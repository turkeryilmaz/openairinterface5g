#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0

"""Adversarial archive/state tests for the R0 terminal controller."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import contextvars
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
import time
import unittest
from unittest import mock

import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.profiling.memory import oai_memprof_r0_terminal_archive as archive  # noqa: E402


MANIFEST = "manifest.json"


class DuplicateKeyMapping(Mapping[str, object]):
    """Mapping adversary whose iterator exposes one key twice."""

    def __getitem__(self, key: str) -> object:
        if key != "duplicate":
            raise KeyError(key)
        return 1

    def __iter__(self) -> Iterator[str]:
        return iter(("duplicate", "duplicate"))

    def __len__(self) -> int:
        return 2


class CallbackCancellation(BaseException):
    """Sentinel proving exact cooperative-cancellation propagation."""


class TerminalArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="oai-memprof-r0-archive-test-"
        )
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def new_directory(self, name: str = "active") -> Path:
        path = self.root / name
        archive.create_directory_exclusive(path)
        self.assertTrue(path.is_dir())
        return path

    def manifest_limits(self) -> dict[str, int | str]:
        return {
            "manifest_relative_path": MANIFEST,
            "max_regular_files_excluding_manifest": 64,
            "max_directories_excluding_root": 64,
            "max_regular_file_bytes": 1024 * 1024,
            "max_total_regular_file_bytes": 4 * 1024 * 1024,
        }

    def create_verified_manifest(self, root: Path) -> archive.ManifestSummary:
        summary = archive.create_manifest(
            root,
            **self.manifest_limits(),
            max_manifest_bytes=256 * 1024,
        )
        verified = archive.verify_manifest(root, **self.manifest_limits())
        self.assertEqual(verified, summary)
        return summary

    def assert_archive_error(
        self,
        expected: archive.ArchiveErrorCode,
        operation,
    ) -> archive.ArchiveError:
        with self.assertRaises(archive.ArchiveError) as context:
            operation()
        self.assertEqual(context.exception.code, expected, context.exception)
        return context.exception

    def callback_sentinels(self, label: str) -> tuple[BaseException, ...]:
        return (
            CallbackCancellation(label),
            OSError(errno.ECANCELED, label),
            archive.ArchiveError(
                archive.ArchiveErrorCode.STATE_INVALID,
                "callback-sentinel",
                label,
                publication_phase=archive.PublicationPhase.VERIFIED,
            ),
        )

    def assert_exact_callback_failure(
        self,
        sentinel: BaseException,
        operation,
    ) -> BaseException:
        with self.assertRaises(BaseException) as context:
            operation()
        self.assertIs(context.exception, sentinel)
        self.assertIsNone(context.exception.__cause__)
        self.assertIsNone(context.exception.__context__)
        return context.exception

    def assert_callback_exception_graph_is_public(
        self,
        error: BaseException,
    ) -> None:
        pending = [error]
        seen: set[int] = set()
        while pending:
            current = pending.pop()
            identity = id(current)
            self.assertNotIn(identity, seen, "exception graph contains a cycle")
            seen.add(identity)
            self.assertNotIsInstance(current, archive._CallbackOrigin)
            if current.__cause__ is not None:
                pending.append(current.__cause__)
            if current.__context__ is not None:
                pending.append(current.__context__)

    def chained_callback_sentinel(
        self,
        label: str,
    ) -> tuple[BaseException, BaseException, BaseException]:
        cause = ValueError(f"{label} cause")
        context = KeyError(f"{label} context")
        sentinel = CallbackCancellation(label)
        sentinel.__cause__ = cause
        sentinel.__context__ = context
        sentinel.__suppress_context__ = True
        return sentinel, cause, context

    def callback_on_call(
        self,
        sentinel: BaseException,
        target_call: int,
    ):
        calls = 0

        def callback() -> None:
            nonlocal calls
            calls += 1
            if calls == target_call:
                raise sentinel

        return callback

    def test_canonical_json_has_one_exact_platform_independent_encoding(self) -> None:
        value = {"z": (True, None), "a": 1}
        literal = b'{"a":1,"z":[true,null]}'
        self.assertEqual(archive.canonical_json_bytes(value), literal)
        self.assertEqual(
            archive.canonical_json_sha256(value),
            hashlib.sha256(literal).hexdigest(),
        )
        self.assertEqual(
            archive.canonical_json_bytes({"non_ascii": "\N{LATIN SMALL LETTER E WITH ACUTE}"}),
            b'{"non_ascii":"\\u00e9"}',
        )

    def test_canonical_json_rejects_ambiguous_or_unbounded_values(self) -> None:
        cyclic: list[object] = []
        cyclic.append(cyclic)
        deep: object = None
        for _ in range(66):
            deep = [deep]
        cases = (
            1.0,
            float("nan"),
            b"bytes",
            {1: "non-string key"},
            1 << 63,
            -(1 << 63) - 1,
            "\ud800",
            DuplicateKeyMapping(),
            cyclic,
            deep,
        )
        for value in cases:
            with self.subTest(value_type=type(value).__name__):
                self.assert_archive_error(
                    archive.ArchiveErrorCode.INVALID_JSON,
                    lambda value=value: archive.canonical_json_bytes(value),
                )

    def test_run_id_and_path_validators_reject_traversal_and_ambiguity(self) -> None:
        self.assertEqual(archive.validate_run_id("r0-x86_64-0001"), "r0-x86_64-0001")
        for invalid in ("", ".", "..", "../escape", "a/b", "/absolute", "\u00e9", "a" * 129):
            with self.subTest(run_id=invalid):
                self.assert_archive_error(
                    archive.ArchiveErrorCode.INVALID_RUN_ID,
                    lambda invalid=invalid: archive.validate_run_id(invalid),
                )

        absolute = self.root / "target"
        self.assertEqual(
            archive.validate_absolute_path(absolute, "target"), absolute
        )
        for invalid in ("relative", "../relative", "", "//ambiguous", "/tmp/control\x01"):
            with self.subTest(absolute=invalid):
                self.assert_archive_error(
                    archive.ArchiveErrorCode.INVALID_ABSOLUTE_PATH,
                    lambda invalid=invalid: archive.validate_absolute_path(
                        invalid, "target"
                    ),
                )

        self.assertEqual(archive.validate_relative_path("a/b-c_1"), "a/b-c_1")
        for invalid in (
            "",
            ".",
            "..",
            "../a",
            "a/../b",
            "/a",
            "a//b",
            "a/./b",
            "control\x01",
        ):
            with self.subTest(relative=invalid):
                self.assert_archive_error(
                    archive.ArchiveErrorCode.INVALID_RELATIVE_PATH,
                    lambda invalid=invalid: archive.validate_relative_path(invalid),
                )

    def test_safe_join_rejects_lexical_escape_and_existing_symlink_component(self) -> None:
        archive_root = self.new_directory()
        self.assertEqual(
            archive.safe_join(archive_root, "nested/file"),
            archive_root / "nested" / "file",
        )
        outside = self.root / "outside"
        outside.mkdir()
        (archive_root / "link").symlink_to(outside, target_is_directory=True)
        self.assert_archive_error(
            archive.ArchiveErrorCode.SYMLINK_DENIED,
            lambda: archive.safe_join(archive_root, "link/file"),
        )
        self.assert_archive_error(
            archive.ArchiveErrorCode.INVALID_RELATIVE_PATH,
            lambda: archive.safe_join(archive_root, "../outside"),
        )

    def test_directory_and_file_creation_are_exclusive(self) -> None:
        directory = self.new_directory()
        self.assert_archive_error(
            archive.ArchiveErrorCode.PATH_EXISTS,
            lambda: archive.create_directory_exclusive(directory),
        )

        destination = directory / "evidence.bin"
        record = archive.write_bytes_exclusive(
            destination, b"evidence", max_bytes=8
        )
        self.assertEqual(record.path, "evidence.bin")
        self.assertEqual(record.size, 8)
        self.assertEqual(record.sha256, hashlib.sha256(b"evidence").hexdigest())
        information = os.lstat(destination)
        self.assertEqual(
            (record.device, record.inode, record.mtime_ns),
            (information.st_dev, information.st_ino, information.st_mtime_ns),
        )
        self.assertEqual(record.line_count, 1)
        sentinel = destination.read_bytes()
        self.assert_archive_error(
            archive.ArchiveErrorCode.PATH_EXISTS,
            lambda: archive.write_bytes_exclusive(
                destination, b"replacement", max_bytes=32
            ),
        )
        self.assertEqual(destination.read_bytes(), sentinel)

    def test_file_record_line_count_has_exact_empty_and_final_lf_semantics(self) -> None:
        directory = self.new_directory()
        vectors = (
            (b"", 0),
            (b"one", 1),
            (b"one\n", 1),
            (b"one\ntwo", 2),
            (b"\n\n", 2),
        )
        for index, (payload, expected_lines) in enumerate(vectors):
            with self.subTest(payload=payload):
                destination = directory / "line-vector-{}".format(index)
                record = archive.write_bytes_exclusive(
                    destination, payload, max_bytes=len(payload)
                )
                information = os.lstat(destination)
                self.assertEqual(record.line_count, expected_lines)
                self.assertEqual(
                    (record.device, record.inode, record.mtime_ns),
                    (
                        information.st_dev,
                        information.st_ino,
                        information.st_mtime_ns,
                    ),
                )

    def test_file_bounds_are_checked_before_or_during_io_without_partial_output(self) -> None:
        directory = self.new_directory()
        oversized_output = directory / "oversized-output"
        self.assert_archive_error(
            archive.ArchiveErrorCode.FILE_TOO_LARGE,
            lambda: archive.write_bytes_exclusive(
                oversized_output, b"12345", max_bytes=4
            ),
        )
        self.assertFalse(oversized_output.exists())

        source = directory / "source"
        source.write_bytes(b"12345")
        self.assert_archive_error(
            archive.ArchiveErrorCode.FILE_TOO_LARGE,
            lambda: archive.read_regular_file_bounded(source, 4),
        )
        copied = directory / "copied"
        self.assert_archive_error(
            archive.ArchiveErrorCode.FILE_TOO_LARGE,
            lambda: archive.copy_file_exclusive(
                source, copied, max_bytes=4
            ),
        )
        self.assertFalse(copied.exists())

    def test_exclusive_write_failure_and_fsync_failure_leave_no_partial_path(self) -> None:
        directory = self.new_directory()
        real_write = archive.os.write
        destination = directory / "write-failure"
        calls = 0

        def fail_second_write(descriptor: int, data: bytes) -> int:
            nonlocal calls
            calls += 1
            if calls == 1:
                return real_write(descriptor, data[:1])
            raise OSError(errno.EIO, "injected write failure")

        with mock.patch.object(archive.os, "write", fail_second_write):
            self.assert_archive_error(
                archive.ArchiveErrorCode.WRITE_FAILED,
                lambda: archive.write_bytes_exclusive(
                    destination, b"more-than-one-byte", max_bytes=64
                ),
            )
        self.assertFalse(destination.exists())

        fsync_destination = directory / "fsync-failure"
        with mock.patch.object(
            archive.os,
            "fsync",
            side_effect=OSError(errno.EIO, "injected fsync failure"),
        ):
            self.assert_archive_error(
                archive.ArchiveErrorCode.FSYNC_FAILED,
                lambda: archive.write_bytes_exclusive(
                    fsync_destination, b"payload", max_bytes=64
                ),
            )
        self.assertFalse(fsync_destination.exists())

    def test_primary_write_and_copy_errors_survive_cleanup_fsync_failure(self) -> None:
        directory = self.new_directory()
        source = directory / "source"
        source.write_bytes(b"source")
        cleanup_failure = archive.ArchiveError(
            archive.ArchiveErrorCode.FSYNC_FAILED,
            "injected-cleanup",
            "directory fsync failed",
        )

        for operation, destination in (
            (
                lambda destination: archive.write_bytes_exclusive(
                    destination, b"payload", max_bytes=64
                ),
                directory / "failed-write",
            ),
            (
                lambda destination: archive.copy_file_exclusive(
                    source, destination, max_bytes=64
                ),
                directory / "failed-copy",
            ),
        ):
            with self.subTest(destination=destination.name):
                with mock.patch.object(
                    archive.os,
                    "write",
                    side_effect=OSError(errno.EIO, "injected primary write failure"),
                ), mock.patch.object(
                    archive, "_fsync_directory", side_effect=cleanup_failure
                ):
                    error = self.assert_archive_error(
                        archive.ArchiveErrorCode.WRITE_FAILED,
                        lambda operation=operation, destination=destination: operation(
                            destination
                        ),
                    )
                self.assertFalse(destination.exists())
                notes = getattr(error, "__notes__", ())
                self.assertTrue(
                    any("cleanup" in note or "fsync" in note for note in notes),
                    notes,
                )

    def test_streaming_copy_exceeds_one_chunk_and_removes_partial_on_write_failure(self) -> None:
        directory = self.new_directory()
        source = directory / "large-source"
        payload = bytes(range(251)) * 5000
        source.write_bytes(payload)
        destination = directory / "large-copy"
        with mock.patch.object(
            archive,
            "read_regular_file_bounded",
            side_effect=AssertionError("copy must stream independently"),
        ):
            record = archive.copy_file_exclusive(
                source, destination, max_bytes=len(payload)
            )
        self.assertEqual(record.size, len(payload))
        self.assertEqual(destination.read_bytes(), payload)
        information = os.lstat(destination)
        self.assertEqual(
            (record.device, record.inode, record.mtime_ns),
            (information.st_dev, information.st_ino, information.st_mtime_ns),
        )
        self.assertEqual(
            record.line_count,
            payload.count(b"\n") + int(bool(payload) and not payload.endswith(b"\n")),
        )

        failed_destination = directory / "failed-large-copy"
        real_write = archive.os.write
        calls = 0

        def fail_later_write(descriptor: int, data: bytes) -> int:
            nonlocal calls
            calls += 1
            if calls <= 2:
                return real_write(descriptor, data)
            raise OSError(errno.EIO, "injected streaming write failure")

        with mock.patch.object(archive.os, "write", fail_later_write):
            self.assert_archive_error(
                archive.ArchiveErrorCode.WRITE_FAILED,
                lambda: archive.copy_file_exclusive(
                    source, failed_destination, max_bytes=len(payload)
                ),
            )
        self.assertFalse(failed_destination.exists())

    def test_streaming_copy_cancellation_and_source_failures_are_typed_and_clean(self) -> None:
        directory = self.new_directory()
        source = directory / "source"
        source.write_bytes(b"source payload")

        class InjectedCancellation(BaseException):
            pass

        cancellation = InjectedCancellation("injected cancellation")
        cancelled_destination = directory / "cancelled-copy"
        with mock.patch.object(archive.os, "read", side_effect=cancellation):
            with self.assertRaises(InjectedCancellation) as context:
                archive.copy_file_exclusive(
                    source, cancelled_destination, max_bytes=64
                )
        self.assertIs(context.exception, cancellation)
        self.assertFalse(cancelled_destination.exists())

        eagain_destination = directory / "eagain-copy"
        with mock.patch.object(
            archive.os,
            "read",
            side_effect=BlockingIOError(errno.EAGAIN, "injected source EAGAIN"),
        ):
            self.assert_archive_error(
                archive.ArchiveErrorCode.READ_FAILED,
                lambda: archive.copy_file_exclusive(
                    source, eagain_destination, max_bytes=64
                ),
            )
        self.assertFalse(eagain_destination.exists())

        disappeared_destination = directory / "disappeared-copy"
        real_lstat = archive.os.lstat

        def missing_source_after_copy(path, *args, **kwargs):
            if Path(path) == source:
                raise FileNotFoundError(errno.ENOENT, "injected disappearance", path)
            return real_lstat(path, *args, **kwargs)

        with mock.patch.object(archive.os, "lstat", missing_source_after_copy):
            self.assert_archive_error(
                archive.ArchiveErrorCode.FILE_CHANGED,
                lambda: archive.copy_file_exclusive(
                    source, disappeared_destination, max_bytes=64
                ),
            )
        self.assertFalse(disappeared_destination.exists())

    def test_bounded_read_detects_pathname_replacement_during_open_file_read(self) -> None:
        directory = self.new_directory()
        source = directory / "source"
        source.write_bytes(b"A" * 16384)
        replacement = directory / "replacement"
        replacement.write_bytes(b"B" * 16384)
        real_read = archive.os.read
        replaced = False

        def replace_after_first_read(descriptor: int, count: int) -> bytes:
            nonlocal replaced
            data = real_read(descriptor, count)
            if not replaced:
                replaced = True
                os.replace(replacement, source)
            return data

        with mock.patch.object(archive.os, "read", replace_after_first_read):
            self.assert_archive_error(
                archive.ArchiveErrorCode.FILE_CHANGED,
                lambda: archive.read_regular_file_bounded(source, 32768),
            )

    def test_post_create_hardlink_injection_is_rejected(self) -> None:
        directory = self.new_directory()
        destination = directory / "destination"
        alias = directory / "injected-hardlink"
        real_open = archive.os.open

        def open_and_link(path, flags, mode=0o777, *args, **kwargs):
            descriptor = real_open(path, flags, mode, *args, **kwargs)
            if Path(path) == destination:
                os.link(destination, alias)
            return descriptor

        with mock.patch.object(archive.os, "open", open_and_link):
            self.assert_archive_error(
                archive.ArchiveErrorCode.PATH_TYPE,
                lambda: archive.write_bytes_exclusive(
                    destination, b"payload", max_bytes=64
                ),
            )
        self.assertFalse(destination.exists())

    def test_regular_file_reader_rejects_symlink_directory_and_fifo_without_blocking(self) -> None:
        directory = self.new_directory()
        target = directory / "target"
        target.write_bytes(b"target")
        symlink = directory / "symlink"
        symlink.symlink_to(target)
        fifo = directory / "fifo"
        os.mkfifo(fifo, 0o600)

        cases = (
            (symlink, archive.ArchiveErrorCode.SYMLINK_DENIED),
            (directory, archive.ArchiveErrorCode.PATH_TYPE),
            (fifo, archive.ArchiveErrorCode.PATH_TYPE),
        )
        for path, expected in cases:
            with self.subTest(path=path.name):
                started = time.monotonic()
                self.assert_archive_error(
                    expected,
                    lambda path=path: archive.read_regular_file_bounded(path, 64),
                )
                self.assertLess(time.monotonic() - started, 0.5)

    def test_copy_rejects_symlink_source_and_destination_collision(self) -> None:
        directory = self.new_directory()
        source = directory / "source"
        source.write_bytes(b"copy-source")
        source_link = directory / "source-link"
        source_link.symlink_to(source)
        destination = directory / "destination"
        self.assert_archive_error(
            archive.ArchiveErrorCode.SYMLINK_DENIED,
            lambda: archive.copy_file_exclusive(
                source_link, destination, max_bytes=64
            ),
        )
        destination.write_bytes(b"sentinel")
        self.assert_archive_error(
            archive.ArchiveErrorCode.PATH_EXISTS,
            lambda: archive.copy_file_exclusive(
                source, destination, max_bytes=64
            ),
        )
        self.assertEqual(destination.read_bytes(), b"sentinel")

    def test_state_checkpoint_round_trip_and_atomic_failure_preserves_predecessor(self) -> None:
        active = self.new_directory()
        first = {"sequence": 1, "phase": "STARTED"}
        second = {"sequence": 2, "phase": "SUCCEEDED"}
        first_record = archive.write_state_checkpoint(active, first, max_bytes=4096)
        self.assertEqual(archive.read_state_checkpoint(active, max_bytes=4096), first)

        real_replace = archive.os.replace

        def fail_replace(source, destination, *args, **kwargs):
            raise OSError(errno.EIO, "injected checkpoint rename failure")

        with mock.patch.object(archive.os, "replace", fail_replace):
            self.assert_archive_error(
                archive.ArchiveErrorCode.WRITE_FAILED,
                lambda: archive.write_state_checkpoint(
                    active, second, max_bytes=4096
                ),
            )
        self.assertEqual(archive.read_state_checkpoint(active, max_bytes=4096), first)
        self.assertEqual(
            (active / first_record.path).read_bytes(),
            archive.canonical_json_bytes(first),
        )
        leftovers = [
            path.name
            for path in active.iterdir()
            if path.name != first_record.path
        ]
        self.assertEqual(leftovers, [])
        self.assertIs(archive.os.replace, real_replace)

        second_record = archive.write_state_checkpoint(active, second, max_bytes=4096)
        self.assertEqual(second_record.path, first_record.path)
        self.assertEqual(archive.read_state_checkpoint(active, max_bytes=4096), second)

    def test_state_checkpoint_rejects_nonexact_bound_before_serialization_or_mutation(self) -> None:
        active = self.new_directory()
        predecessor = {"sequence": 1, "phase": "STARTED"}
        record = archive.write_state_checkpoint(
            active,
            predecessor,
            max_bytes=4096,
        )
        state_path = active / record.path
        before_information = os.lstat(state_path)
        before_bytes = state_path.read_bytes()
        before_population = sorted(path.name for path in active.iterdir())

        for invalid in (False, -1, "4096", 1 << 63):
            with self.subTest(max_bytes=invalid):
                with mock.patch.object(
                    archive,
                    "canonical_json_bytes",
                    side_effect=AssertionError(
                        "invalid bound reached checkpoint serialization"
                    ),
                ) as serializer:
                    self.assert_archive_error(
                        archive.ArchiveErrorCode.INVALID_ARGUMENT,
                        lambda invalid=invalid: archive.write_state_checkpoint(
                            active,
                            {"sequence": 2},
                            max_bytes=invalid,
                        ),
                    )
                serializer.assert_not_called()
                after_information = os.lstat(state_path)
                self.assertEqual(
                    (after_information.st_dev, after_information.st_ino),
                    (before_information.st_dev, before_information.st_ino),
                )
                self.assertEqual(state_path.read_bytes(), before_bytes)
                self.assertEqual(
                    sorted(path.name for path in active.iterdir()),
                    before_population,
                )
        self.assertEqual(
            archive.read_state_checkpoint(active, max_bytes=4096),
            predecessor,
        )

    def test_state_checkpoint_cleanup_is_identity_safe_and_preserves_primary_error(self) -> None:
        active = self.new_directory()
        predecessor = {"sequence": 1}
        archive.write_state_checkpoint(active, predecessor, max_bytes=4096)
        replacement_path: Path | None = None

        def replace_temporary_then_fail(source, destination):
            nonlocal replacement_path
            replacement_path = Path(source)
            os.unlink(source)
            replacement_path.write_bytes(b"foreign inode")
            raise OSError(errno.EIO, "injected checkpoint rename failure")

        with mock.patch.object(
            archive.os,
            "replace",
            side_effect=replace_temporary_then_fail,
        ):
            self.assert_archive_error(
                archive.ArchiveErrorCode.WRITE_FAILED,
                lambda: archive.write_state_checkpoint(
                    active, {"sequence": 2}, max_bytes=4096
                ),
            )
        self.assertIsNotNone(replacement_path)
        self.assertEqual(replacement_path.read_bytes(), b"foreign inode")
        self.assertEqual(
            archive.read_state_checkpoint(active, max_bytes=4096),
            predecessor,
        )

        replacement_path.unlink()
        real_fsync_directory = archive._fsync_directory
        fsync_calls = 0

        def fail_cleanup_fsync(path):
            nonlocal fsync_calls
            fsync_calls += 1
            if fsync_calls == 2:
                raise archive.ArchiveError(
                    archive.ArchiveErrorCode.FSYNC_FAILED,
                    "injected-checkpoint-cleanup",
                    "cleanup directory fsync failed",
                    Path(path),
                )
            return real_fsync_directory(path)

        with (
            mock.patch.object(
                archive.os,
                "replace",
                side_effect=OSError(
                    errno.EIO, "primary checkpoint rename failure"
                ),
            ),
            mock.patch.object(
                archive,
                "_fsync_directory",
                side_effect=fail_cleanup_fsync,
            ),
        ):
            error = self.assert_archive_error(
                archive.ArchiveErrorCode.WRITE_FAILED,
                lambda: archive.write_state_checkpoint(
                    active, {"sequence": 3}, max_bytes=4096
                ),
            )
        self.assertTrue(
            any(
                "temporary unlink cleanup raised" in note
                for note in getattr(error, "__notes__", ())
            ),
            getattr(error, "__notes__", ()),
        )
        self.assertEqual(
            archive.read_state_checkpoint(active, max_bytes=4096),
            predecessor,
        )
        self.assertEqual(
            sorted(path.name for path in active.iterdir()),
            ["state.json"],
        )

    def test_state_reader_rejects_invalid_json_and_symlink(self) -> None:
        active = self.new_directory()
        record = archive.write_state_checkpoint(
            active, {"sequence": 1, "phase": "STARTED"}, max_bytes=4096
        )
        state_path = active / record.path
        state_path.write_bytes(b"{not-json")
        self.assert_archive_error(
            archive.ArchiveErrorCode.STATE_INVALID,
            lambda: archive.read_state_checkpoint(active, max_bytes=4096),
        )
        state_path.unlink()
        target = self.root / "outside-state"
        target.write_text("{}", encoding="ascii")
        state_path.symlink_to(target)
        self.assert_archive_error(
            archive.ArchiveErrorCode.SYMLINK_DENIED,
            lambda: archive.read_state_checkpoint(active, max_bytes=4096),
        )

    def test_journal_sequence_and_hash_chain_round_trip(self) -> None:
        active = self.new_directory()
        cursor0 = archive.initialize_journal(active)
        self.assertEqual(cursor0.next_sequence, 0)
        cursor1 = archive.append_journal(
            active,
            {"event": "STARTED"},
            cursor=cursor0,
            max_entry_bytes=4096,
        )
        cursor2 = archive.append_journal(
            active,
            {"event": "SUCCEEDED"},
            cursor=cursor1,
            max_entry_bytes=4096,
        )
        observed = archive.read_journal(
            active, max_entries=2, max_entry_bytes=4096
        )
        self.assertEqual(observed.cursor, cursor2)
        self.assertEqual(len(observed.entries), 2)
        self.assertEqual(observed.entries[0]["payload"], {"event": "STARTED"})
        self.assertEqual(observed.entries[1]["payload"], {"event": "SUCCEEDED"})

        self.assert_archive_error(
            archive.ArchiveErrorCode.JOURNAL_SEQUENCE,
            lambda: archive.append_journal(
                active,
                {"event": "STALE"},
                cursor=cursor0,
                max_entry_bytes=4096,
            ),
        )

    def test_journal_detects_missing_tampered_symlink_and_entry_overlimit(self) -> None:
        active = self.new_directory()
        before = set(active.iterdir())
        cursor = archive.initialize_journal(active)
        cursor = archive.append_journal(
            active,
            {"event": "ONE"},
            cursor=cursor,
            max_entry_bytes=4096,
        )
        archive.append_journal(
            active,
            {"event": "TWO"},
            cursor=cursor,
            max_entry_bytes=4096,
        )
        journal_files = set(active.iterdir()) - before
        self.assertEqual(len(journal_files), 1)
        journal_path = journal_files.pop()

        self.assert_archive_error(
            archive.ArchiveErrorCode.DIRECTORY_TOO_LARGE,
            lambda: archive.read_journal(
                active, max_entries=1, max_entry_bytes=4096
            ),
        )
        original = journal_path.read_bytes()
        journal_path.write_bytes(original.replace(b"ONE", b"BAD", 1))
        self.assert_archive_error(
            archive.ArchiveErrorCode.JOURNAL_HASH,
            lambda: archive.read_journal(
                active, max_entries=2, max_entry_bytes=4096
            ),
        )

        journal_path.unlink()
        self.assert_archive_error(
            archive.ArchiveErrorCode.PATH_MISSING,
            lambda: archive.read_journal(
                active, max_entries=2, max_entry_bytes=4096
            ),
        )
        outside = self.root / "outside-journal"
        outside.write_bytes(original)
        journal_path.symlink_to(outside)
        self.assert_archive_error(
            archive.ArchiveErrorCode.SYMLINK_DENIED,
            lambda: archive.read_journal(
                active, max_entries=2, max_entry_bytes=4096
            ),
        )

    def test_journal_primary_error_survives_directory_fsync_cleanup_failure(self) -> None:
        active = self.new_directory()
        stale = archive.initialize_journal(active)
        current = archive.append_journal(
            active,
            {"event": "ONE"},
            cursor=stale,
            max_entry_bytes=4096,
        )
        cleanup_failure = archive.ArchiveError(
            archive.ArchiveErrorCode.FSYNC_FAILED,
            "injected-cleanup",
            "directory fsync failed",
        )
        with mock.patch.object(
            archive, "_fsync_directory", side_effect=cleanup_failure
        ):
            error = self.assert_archive_error(
                archive.ArchiveErrorCode.JOURNAL_SEQUENCE,
                lambda: archive.append_journal(
                    active,
                    {"event": "STALE"},
                    cursor=stale,
                    max_entry_bytes=4096,
                ),
            )
        notes = getattr(error, "__notes__", ())
        self.assertTrue(any("directory-fsync" in note for note in notes), notes)
        self.assertEqual(
            archive.read_journal(
                active, max_entries=2, max_entry_bytes=4096
            ).cursor,
            current,
        )

    def test_journal_rejects_boolean_sequence_even_when_integer_hash_matches(self) -> None:
        active = self.new_directory()
        before = set(active.iterdir())
        cursor = archive.initialize_journal(active)
        archive.append_journal(
            active,
            {"event": "ONE"},
            cursor=cursor,
            max_entry_bytes=4096,
        )
        journal_files = set(active.iterdir()) - before
        self.assertEqual(len(journal_files), 1)
        journal_path = journal_files.pop()
        line = journal_path.read_bytes().removesuffix(b"\n")
        document = json.loads(line)
        self.assertEqual(document["sequence"], 0)
        document["sequence"] = False
        journal_path.write_bytes(archive.canonical_json_bytes(document) + b"\n")
        self.assert_archive_error(
            archive.ArchiveErrorCode.JOURNAL_INVALID,
            lambda: archive.read_journal(
                active, max_entries=1, max_entry_bytes=4096
            ),
        )

    def test_journal_rejects_nonexact_hash_and_cursor_field_types(self) -> None:
        for index, field in enumerate(
            ("entry_sha256", "payload_sha256", "previous_sha256")
        ):
            with self.subTest(field=field):
                active = self.new_directory(f"hash-type-{index}")
                cursor = archive.initialize_journal(active)
                archive.append_journal(
                    active,
                    {"event": "ONE"},
                    cursor=cursor,
                    max_entry_bytes=4096,
                )
                journal_path = active / "journal.jsonl"
                document = json.loads(
                    journal_path.read_bytes().removesuffix(b"\n")
                )
                document[field] = False
                journal_path.write_bytes(
                    archive.canonical_json_bytes(document) + b"\n"
                )
                self.assert_archive_error(
                    archive.ArchiveErrorCode.JOURNAL_INVALID,
                    lambda active=active: archive.read_journal(
                        active, max_entries=1, max_entry_bytes=4096
                    ),
                )

        active = self.new_directory("cursor-types")
        cursor = archive.initialize_journal(active)
        invalid_cursors = (
            archive.JournalCursor(False, cursor.previous_sha256),
            archive.JournalCursor(0, False),
            archive.JournalCursor((1 << 63) - 1, cursor.previous_sha256),
        )
        for invalid in invalid_cursors:
            with self.subTest(cursor=invalid):
                self.assert_archive_error(
                    archive.ArchiveErrorCode.INVALID_ARGUMENT,
                    lambda invalid=invalid: archive.append_journal(
                        active,
                        {"event": "INVALID"},
                        cursor=invalid,
                        max_entry_bytes=4096,
                    ),
                )
        self.assertEqual(
            archive.read_journal(
                active, max_entries=0, max_entry_bytes=4096
            ).cursor,
            cursor,
        )

    def test_manifest_create_and_verify_bind_exact_regular_file_population(self) -> None:
        active = self.new_directory()
        archive.write_bytes_exclusive(active / "a", b"alpha", max_bytes=64)
        nested = active / "nested"
        archive.create_directory_exclusive(nested)
        archive.write_bytes_exclusive(nested / "b", b"beta", max_bytes=64)

        summary = self.create_verified_manifest(active)
        self.assertEqual(summary.regular_file_count, 2)
        self.assertEqual(summary.directory_count, 1)
        self.assertEqual(summary.total_regular_file_bytes, 9)
        self.assertEqual(summary.manifest_path, MANIFEST)
        manifest_path = active / MANIFEST
        self.assertEqual(
            summary.manifest_sha256,
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        )

    def test_manifest_rejects_missing_tampered_and_symlinked_member(self) -> None:
        for mutation in ("missing", "tampered", "symlink"):
            with self.subTest(mutation=mutation):
                active = self.new_directory("active-{}".format(mutation))
                member = active / "member"
                member.write_bytes(b"original")
                self.create_verified_manifest(active)
                if mutation == "missing":
                    member.unlink()
                    expected = archive.ArchiveErrorCode.PATH_MISSING
                elif mutation == "tampered":
                    member.write_bytes(b"modified")
                    expected = archive.ArchiveErrorCode.HASH_MISMATCH
                else:
                    member.unlink()
                    outside = self.root / "outside-member"
                    if not outside.exists():
                        outside.write_bytes(b"original")
                    member.symlink_to(outside)
                    expected = archive.ArchiveErrorCode.SYMLINK_DENIED
                self.assert_archive_error(
                    expected,
                    lambda active=active: archive.verify_manifest(
                        active, **self.manifest_limits()
                    ),
                )

    def test_manifest_rejects_unmanifested_empty_directory(self) -> None:
        active = self.new_directory()
        (active / "member").write_bytes(b"member")
        self.create_verified_manifest(active)
        (active / "unmanifested-empty-directory").mkdir()
        self.assert_archive_error(
            archive.ArchiveErrorCode.MANIFEST_INVALID,
            lambda: archive.verify_manifest(active, **self.manifest_limits()),
        )

    def test_manifest_binds_sorted_directory_population_and_separate_bound(self) -> None:
        active = self.new_directory()
        (active / "z-empty").mkdir()
        (active / "a").mkdir()
        (active / "a" / "nested-empty").mkdir()
        (active / "member").write_bytes(b"member")
        summary = self.create_verified_manifest(active)
        self.assertEqual(summary.regular_file_count, 1)
        self.assertEqual(summary.directory_count, 3)
        document = json.loads((active / MANIFEST).read_text(encoding="ascii"))
        self.assertEqual(
            document["directories"],
            ["a", "a/nested-empty", "z-empty"],
        )
        self.assertEqual(document["directory_count"], 3)

        (active / "z-empty").rmdir()
        self.assert_archive_error(
            archive.ArchiveErrorCode.PATH_MISSING,
            lambda: archive.verify_manifest(active, **self.manifest_limits()),
        )

        bounded = self.new_directory("directory-bound")
        (bounded / "empty").mkdir()
        self.assert_archive_error(
            archive.ArchiveErrorCode.DIRECTORY_TOO_LARGE,
            lambda: archive.create_manifest(
                bounded,
                manifest_relative_path=MANIFEST,
                max_regular_files_excluding_manifest=64,
                max_directories_excluding_root=0,
                max_regular_file_bytes=64,
                max_total_regular_file_bytes=64,
                max_manifest_bytes=4096,
            ),
        )
        self.assertFalse((bounded / MANIFEST).exists())

    def test_manifest_rejects_nonexact_or_unsorted_directory_schema(self) -> None:
        for index, replacement in enumerate(
            (["z", "a"], ["a", "a"], ["a", False])
        ):
            with self.subTest(replacement=replacement):
                active = self.new_directory(f"directory-schema-{index}")
                (active / "a").mkdir()
                (active / "z").mkdir()
                self.create_verified_manifest(active)
                manifest_path = active / MANIFEST
                document = json.loads(manifest_path.read_text(encoding="ascii"))
                document["directories"] = replacement
                document["directory_count"] = len(replacement)
                manifest_path.write_bytes(archive.canonical_json_bytes(document))
                self.assert_archive_error(
                    archive.ArchiveErrorCode.MANIFEST_INVALID,
                    lambda active=active: archive.verify_manifest(
                        active, **self.manifest_limits()
                    ),
                )

    def test_manifest_bounds_fail_for_count_individual_and_total_size(self) -> None:
        active = self.new_directory()
        (active / "a").write_bytes(b"1234")
        (active / "b").write_bytes(b"5678")
        cases = (
            (
                archive.ArchiveErrorCode.DIRECTORY_TOO_LARGE,
                {
                    "max_regular_files_excluding_manifest": 1,
                    "max_directories_excluding_root": 64,
                    "max_regular_file_bytes": 8,
                    "max_total_regular_file_bytes": 16,
                },
            ),
            (
                archive.ArchiveErrorCode.FILE_TOO_LARGE,
                {
                    "max_regular_files_excluding_manifest": 2,
                    "max_directories_excluding_root": 64,
                    "max_regular_file_bytes": 3,
                    "max_total_regular_file_bytes": 16,
                },
            ),
            (
                archive.ArchiveErrorCode.TOTAL_TOO_LARGE,
                {
                    "max_regular_files_excluding_manifest": 2,
                    "max_directories_excluding_root": 64,
                    "max_regular_file_bytes": 8,
                    "max_total_regular_file_bytes": 7,
                },
            ),
        )
        for index, (expected, bounds) in enumerate(cases):
            with self.subTest(expected=expected):
                manifest = "manifest-{}.json".format(index)
                self.assert_archive_error(
                    expected,
                    lambda manifest=manifest, bounds=bounds: archive.create_manifest(
                        active,
                        manifest_relative_path=manifest,
                        max_manifest_bytes=4096,
                        **bounds,
                    ),
                )
                self.assertFalse((active / manifest).exists())

    def test_manifest_population_bound_stops_scandir_after_max_plus_one(self) -> None:
        active = self.new_directory()
        for index in range(2):
            (active / "entry-{:04d}".format(index)).write_bytes(b"member")

        class SyntheticEntry:
            def __init__(self, index: int) -> None:
                self.name = "entry-{:04d}".format(index)
                self.path = str(active / self.name)

        class CountingScandir:
            def __init__(self, population: int) -> None:
                self.population = population
                self.consumed = 0

            def __enter__(self):
                return self

            def __exit__(self, exception_type, exception, traceback):
                return False

            def __iter__(self):
                return self

            def __next__(self):
                if self.consumed >= self.population:
                    raise StopIteration
                entry = SyntheticEntry(self.consumed)
                self.consumed += 1
                return entry

        iterator = CountingScandir(10_000)
        with mock.patch.object(archive.os, "scandir", return_value=iterator):
            self.assert_archive_error(
                archive.ArchiveErrorCode.DIRECTORY_TOO_LARGE,
                lambda: archive.create_manifest(
                    active,
                    manifest_relative_path=MANIFEST,
                    max_regular_files_excluding_manifest=2,
                    max_directories_excluding_root=0,
                    max_regular_file_bytes=64,
                    max_total_regular_file_bytes=128,
                    max_manifest_bytes=4096,
                ),
            )
        self.assertEqual(iterator.consumed, 3)
        self.assertFalse((active / MANIFEST).exists())

    def test_manifest_rejects_symlink_population_and_self_reference(self) -> None:
        active = self.new_directory()
        outside = self.root / "outside"
        outside.write_bytes(b"outside")
        (active / "member-link").symlink_to(outside)
        self.assert_archive_error(
            archive.ArchiveErrorCode.SYMLINK_DENIED,
            lambda: archive.create_manifest(
                active,
                **self.manifest_limits(),
                max_manifest_bytes=4096,
            ),
        )

        (active / "member-link").unlink()
        (active / "member").write_bytes(b"member")
        self.create_verified_manifest(active)
        manifest_path = active / MANIFEST
        document = json.loads(manifest_path.read_text(encoding="ascii"))
        self.assertIsInstance(document.get("regular_files"), list)
        document["regular_files"].append(
            {
                "mode": stat.S_IFREG | 0o600,
                "mtime_ns": 0,
                "path": MANIFEST,
                "sha256": "0" * 64,
                "size": 0,
            }
        )
        manifest_path.write_bytes(archive.canonical_json_bytes(document))
        error = self.assert_archive_error(
            archive.ArchiveErrorCode.MANIFEST_INVALID,
            lambda: archive.verify_manifest(active, **self.manifest_limits()),
        )
        self.assertIn("contains the manifest itself", error.detail)

    def test_manifest_rejects_hardlink_fifo_nested_symlink_and_mtime_change(self) -> None:
        for mutation in ("hardlink", "fifo", "nested-symlink"):
            with self.subTest(mutation=mutation):
                active = self.new_directory("special-{}".format(mutation))
                if mutation == "hardlink":
                    member = active / "member"
                    member.write_bytes(b"member")
                    os.link(member, active / "member-alias")
                elif mutation == "fifo":
                    os.mkfifo(active / "fifo", 0o600)
                else:
                    outside = self.root / "outside-directory"
                    outside.mkdir(exist_ok=True)
                    (active / "nested").symlink_to(outside, target_is_directory=True)
                self.assert_archive_error(
                    archive.ArchiveErrorCode.PATH_TYPE
                    if mutation != "nested-symlink"
                    else archive.ArchiveErrorCode.SYMLINK_DENIED,
                    lambda active=active: archive.create_manifest(
                        active,
                        **self.manifest_limits(),
                        max_manifest_bytes=4096,
                    ),
                )

        active = self.new_directory("mtime-change")
        member = active / "member"
        member.write_bytes(b"stable bytes")
        self.create_verified_manifest(active)
        information = member.stat()
        os.utime(
            member,
            ns=(information.st_atime_ns, information.st_mtime_ns + 1_000_000),
        )
        self.assert_archive_error(
            archive.ArchiveErrorCode.FILE_CHANGED,
            lambda: archive.verify_manifest(active, **self.manifest_limits()),
        )

    def test_manifest_rejects_boolean_and_negative_numeric_fields(self) -> None:
        mutations = (
            ("regular_file_count", True),
            ("regular_file_count", -1),
            ("directory_count", True),
            ("directory_count", -1),
            ("total_regular_file_bytes", True),
            ("total_regular_file_bytes", -1),
            ("mode", True),
            ("mode", -1),
            ("size", True),
            ("size", -1),
            ("mtime_ns", True),
            ("mtime_ns", -1),
        )
        for index, (field, replacement) in enumerate(mutations):
            with self.subTest(field=field, replacement=replacement):
                active = self.new_directory("manifest-number-{}".format(index))
                (active / "member").write_bytes(b"member")
                self.create_verified_manifest(active)
                manifest_path = active / MANIFEST
                document = json.loads(manifest_path.read_text(encoding="ascii"))
                if field in {
                    "regular_file_count",
                    "directory_count",
                    "total_regular_file_bytes",
                }:
                    document[field] = replacement
                else:
                    document["regular_files"][0][field] = replacement
                manifest_path.write_bytes(archive.canonical_json_bytes(document))
                self.assert_archive_error(
                    archive.ArchiveErrorCode.MANIFEST_INVALID,
                    lambda active=active: archive.verify_manifest(
                        active, **self.manifest_limits()
                    ),
                )

    def test_manifest_distinguishes_hash_and_metadata_only_divergence(self) -> None:
        hash_active = self.new_directory("hash-only")
        hash_member = hash_active / "member"
        hash_member.write_bytes(b"original")
        self.create_verified_manifest(hash_active)
        information = hash_member.stat()
        hash_member.write_bytes(b"modified")
        os.utime(
            hash_member,
            ns=(information.st_atime_ns, information.st_mtime_ns),
        )
        self.assert_archive_error(
            archive.ArchiveErrorCode.HASH_MISMATCH,
            lambda: archive.verify_manifest(
                hash_active, **self.manifest_limits()
            ),
        )

        metadata_active = self.new_directory("metadata-only")
        metadata_member = metadata_active / "member"
        metadata_member.write_bytes(b"stable")
        self.create_verified_manifest(metadata_active)
        original_mode = stat.S_IMODE(metadata_member.stat().st_mode)
        os.chmod(metadata_member, original_mode ^ stat.S_IXUSR)
        self.assert_archive_error(
            archive.ArchiveErrorCode.FILE_CHANGED,
            lambda: archive.verify_manifest(
                metadata_active, **self.manifest_limits()
            ),
        )

    def test_publish_is_verified_same_filesystem_and_no_replace(self) -> None:
        active = self.new_directory()
        (active / "member").write_bytes(b"member")
        expected = self.create_verified_manifest(active)
        final = self.root / "final"
        observed = archive.publish_verified_archive(
            active, final, **self.manifest_limits()
        )
        self.assertEqual(observed.summary, expected)
        self.assertEqual(observed.phase, archive.PublicationPhase.VERIFIED)
        self.assertFalse(active.exists())
        self.assertTrue(final.is_dir())
        self.assertEqual(
            archive.verify_manifest(final, **self.manifest_limits()), expected
        )

        active2 = self.new_directory("active2")
        (active2 / "member").write_bytes(b"second")
        self.create_verified_manifest(active2)
        self.assert_archive_error(
            archive.ArchiveErrorCode.PUBLISH_EXISTS,
            lambda: archive.publish_verified_archive(
                active2, final, **self.manifest_limits()
            ),
        )
        self.assertTrue(active2.is_dir())
        self.assertEqual((final / "member").read_bytes(), b"member")

    def test_publish_classifies_same_parent_post_rename_fsync_failure(self) -> None:
        active = self.new_directory()
        (active / "member").write_bytes(b"member")
        self.create_verified_manifest(active)
        final = self.root / "final"
        real_fsync_directory = archive._fsync_directory

        def fail_final_parent_after_rename(path):
            path = Path(path)
            if path == final.parent and final.exists():
                raise archive.ArchiveError(
                    archive.ArchiveErrorCode.FSYNC_FAILED,
                    "injected-final-parent-fsync",
                    "post-rename parent fsync failed",
                    path,
                )
            return real_fsync_directory(path)

        with mock.patch.object(
            archive,
            "_fsync_directory",
            side_effect=fail_final_parent_after_rename,
        ):
            error = self.assert_archive_error(
                archive.ArchiveErrorCode.FSYNC_FAILED,
                lambda: archive.publish_verified_archive(
                    active, final, **self.manifest_limits()
                ),
            )
        self.assertEqual(
            error.publication_phase,
            archive.PublicationPhase.RENAMED_UNSYNCED,
        )
        self.assertFalse(active.exists())
        self.assertTrue(final.is_dir())
        archive.verify_manifest(final, **self.manifest_limits())

    def test_publish_classifies_distinct_source_parent_fsync_failure(self) -> None:
        source_parent = self.root / "source-parent"
        final_parent = self.root / "final-parent"
        archive.create_directory_exclusive(source_parent)
        archive.create_directory_exclusive(final_parent)
        active = source_parent / "active"
        archive.create_directory_exclusive(active)
        (active / "member").write_bytes(b"member")
        self.create_verified_manifest(active)
        final = final_parent / "final"
        real_fsync_directory = archive._fsync_directory

        def fail_source_parent_after_final_parent_sync(path):
            path = Path(path)
            if path == source_parent and final.exists():
                raise archive.ArchiveError(
                    archive.ArchiveErrorCode.FSYNC_FAILED,
                    "injected-source-parent-fsync",
                    "source parent fsync failed",
                    path,
                )
            return real_fsync_directory(path)

        with mock.patch.object(
            archive,
            "_fsync_directory",
            side_effect=fail_source_parent_after_final_parent_sync,
        ):
            error = self.assert_archive_error(
                archive.ArchiveErrorCode.FSYNC_FAILED,
                lambda: archive.publish_verified_archive(
                    active, final, **self.manifest_limits()
                ),
            )
        self.assertEqual(
            error.publication_phase,
            archive.PublicationPhase.FINAL_PARENT_SYNCED,
        )
        self.assertFalse(active.exists())
        self.assertTrue(final.is_dir())
        archive.verify_manifest(final, **self.manifest_limits())

    def test_publish_classifies_post_rename_verification_failure(self) -> None:
        active = self.new_directory()
        (active / "member").write_bytes(b"member")
        self.create_verified_manifest(active)
        final = self.root / "final"
        real_verify_manifest = archive.verify_manifest.__wrapped__

        def fail_final_verification(root, **bounds):
            if Path(root) == final:
                raise archive.ArchiveError(
                    archive.ArchiveErrorCode.HASH_MISMATCH,
                    "injected-post-publish-verify",
                    "post-publication verification failed",
                    final,
                )
            return real_verify_manifest(root, **bounds)

        with mock.patch.object(
            archive.verify_manifest,
            "__wrapped__",
            new=fail_final_verification,
        ):
            error = self.assert_archive_error(
                archive.ArchiveErrorCode.HASH_MISMATCH,
                lambda: archive.publish_verified_archive(
                    active, final, **self.manifest_limits()
                ),
            )
        self.assertEqual(
            error.publication_phase,
            archive.PublicationPhase.PARENTS_SYNCED,
        )
        self.assertFalse(active.exists())
        self.assertTrue(final.is_dir())
        archive.verify_manifest(final, **self.manifest_limits())

    def test_publish_cross_device_failure_leaves_verified_active_tree_untouched(self) -> None:
        active = self.new_directory()
        (active / "member").write_bytes(b"member")
        self.create_verified_manifest(active)
        final = self.root / "final"

        with mock.patch.object(
            archive,
            "_renameat2_noreplace",
            side_effect=OSError(errno.EXDEV, "injected cross-device rename"),
        ):
            error = self.assert_archive_error(
                archive.ArchiveErrorCode.CROSS_DEVICE,
                lambda: archive.publish_verified_archive(
                    active, final, **self.manifest_limits()
                ),
            )
        self.assertEqual(
            error.publication_phase,
            archive.PublicationPhase.PRE_RENAME,
        )
        self.assertTrue(active.is_dir())
        self.assertFalse(final.exists())
        archive.verify_manifest(active, **self.manifest_limits())

    def test_publish_enosys_alias_and_injected_fsync_fail_without_fallback(self) -> None:
        active = self.new_directory("enosys-active")
        (active / "member").write_bytes(b"member")
        self.create_verified_manifest(active)
        final = self.root / "enosys-final"
        with mock.patch.object(
            archive,
            "_renameat2_noreplace",
            side_effect=OSError(errno.ENOSYS, "injected unavailable renameat2"),
        ):
            error = self.assert_archive_error(
                archive.ArchiveErrorCode.PUBLISH_FAILED,
                lambda: archive.publish_verified_archive(
                    active, final, **self.manifest_limits()
                ),
            )
        self.assertEqual(
            error.publication_phase,
            archive.PublicationPhase.PRE_RENAME,
        )
        self.assertTrue(active.is_dir())
        self.assertFalse(final.exists())

        error = self.assert_archive_error(
            archive.ArchiveErrorCode.INVALID_ARGUMENT,
            lambda: archive.publish_verified_archive(
                active, active, **self.manifest_limits()
            ),
        )
        self.assertEqual(
            error.publication_phase,
            archive.PublicationPhase.PRE_RENAME,
        )

        if hasattr(archive, "_fsync_directory"):
            with mock.patch.object(
                archive,
                "_fsync_directory",
                side_effect=archive.ArchiveError(
                    archive.ArchiveErrorCode.FSYNC_FAILED,
                    "injected-fsync",
                    "directory fsync failed",
                ),
            ):
                error = self.assert_archive_error(
                    archive.ArchiveErrorCode.FSYNC_FAILED,
                    lambda: archive.publish_verified_archive(
                        active, final, **self.manifest_limits()
                    ),
                )
            self.assertEqual(
                error.publication_phase,
                archive.PublicationPhase.PRE_RENAME,
            )
            self.assertTrue(active.is_dir())
            self.assertFalse(final.exists())

    def test_publish_rejects_ancestor_and_descendant_destination_aliases(self) -> None:
        active = self.new_directory()
        (active / "member").write_bytes(b"member")
        self.create_verified_manifest(active)
        for final in (active / "nested-final", active.parent):
            with self.subTest(final=final):
                self.assert_archive_error(
                    archive.ArchiveErrorCode.INVALID_ARGUMENT,
                    lambda final=final: archive.publish_verified_archive(
                        active, final, **self.manifest_limits()
                    ),
                )
                self.assertTrue(active.is_dir())
                archive.verify_manifest(active, **self.manifest_limits())

    def test_recovery_inspection_never_upgrades_partial_or_tampered_evidence(self) -> None:
        active = self.new_directory()
        state = {
            "scientific_case_state": "INCOMPLETE",
            "archive_integrity": "UNVERIFIED",
            "inclusion": "EXCLUDED",
            "profiler_stream_state": "NOT_APPLICABLE_R0",
        }
        archive.write_state_checkpoint(active, state, max_bytes=4096)
        cursor = archive.initialize_journal(active)
        archive.append_journal(
            active,
            {"event": "STARTED"},
            cursor=cursor,
            max_entry_bytes=4096,
        )

        before_manifest = archive.inspect_incomplete(
            active,
            max_state_bytes=4096,
            max_journal_entries=8,
            max_journal_entry_bytes=4096,
            **self.manifest_limits(),
        )
        self.assertEqual(before_manifest.state, state)
        self.assertFalse(before_manifest.manifest_present)
        self.assertFalse(before_manifest.manifest_verified)

        self.create_verified_manifest(active)
        (active / MANIFEST).write_bytes(b"{}")
        after_tamper = archive.inspect_incomplete(
            active,
            max_state_bytes=4096,
            max_journal_entries=8,
            max_journal_entry_bytes=4096,
            **self.manifest_limits(),
        )
        self.assertEqual(after_tamper.state, state)
        self.assertTrue(after_tamper.manifest_present)
        self.assertFalse(after_tamper.manifest_verified)
        self.assertTrue(after_tamper.reason)

    def test_progress_callback_chunk_cadence_and_same_thread(self) -> None:
        chunk_bytes = 64 << 10
        self.assertEqual(archive._READ_CHUNK, chunk_bytes)
        payload = b"x" * (2 * chunk_bytes + 17)
        source = self.root / "progress-source"
        source.write_bytes(payload)
        caller_thread = threading.get_ident()

        read_threads: list[int] = []
        image = archive.read_regular_file_bounded(
            source,
            len(payload),
            progress_callback=lambda: read_threads.append(
                threading.get_ident()
            ),
        )
        self.assertEqual(image.data, payload)
        self.assertGreaterEqual(len(read_threads), 4)
        self.assertEqual(set(read_threads), {caller_thread})

        write_threads: list[int] = []
        destination = self.root / "progress-write"
        archive.write_bytes_exclusive(
            destination,
            payload,
            max_bytes=len(payload),
            progress_callback=lambda: write_threads.append(
                threading.get_ident()
            ),
        )
        self.assertEqual(destination.read_bytes(), payload)
        self.assertGreaterEqual(len(write_threads), 8)
        self.assertEqual(set(write_threads), {caller_thread})

        copy_threads: list[int] = []
        copied = self.root / "progress-copy"
        archive.copy_file_exclusive(
            source,
            copied,
            max_bytes=len(payload),
            progress_callback=lambda: copy_threads.append(
                threading.get_ident()
            ),
        )
        self.assertEqual(copied.read_bytes(), payload)
        self.assertGreaterEqual(len(copy_threads), 11)
        self.assertEqual(set(copy_threads), {caller_thread})

    def test_mid_copy_callback_cancellation_cleans_exact_partial_only(self) -> None:
        payload = b"c" * (3 * (64 << 10))
        source = self.root / "copy-cancel-source"
        source.write_bytes(payload)

        destination = self.root / "copy-cancel-destination"
        cancellation = CallbackCancellation("cancel ordinary partial")
        calls = 0

        def cancel_after_first_write() -> None:
            nonlocal calls
            calls += 1
            if calls == 3:
                raise cancellation

        with self.assertRaises(CallbackCancellation) as context:
            archive.copy_file_exclusive(
                source,
                destination,
                max_bytes=len(payload),
                progress_callback=cancel_after_first_write,
            )
        self.assertIs(context.exception, cancellation)
        self.assertFalse(destination.exists())
        self.assertEqual(source.read_bytes(), payload)

        foreign_source = self.root / "foreign-source"
        foreign_source.write_bytes(b"foreign replacement")
        destination = self.root / "copy-foreign-destination"
        cancellation = CallbackCancellation("cancel after replacement")
        calls = 0

        def replace_partial_then_cancel() -> None:
            nonlocal calls
            calls += 1
            if calls == 3:
                os.unlink(destination)
                os.rename(foreign_source, destination)
                raise cancellation

        with self.assertRaises(CallbackCancellation) as context:
            archive.copy_file_exclusive(
                source,
                destination,
                max_bytes=len(payload),
                progress_callback=replace_partial_then_cancel,
            )
        self.assertIs(context.exception, cancellation)
        self.assertEqual(destination.read_bytes(), b"foreign replacement")

    def test_callback_baseexception_remains_primary_when_cleanup_fsync_fails(self) -> None:
        destination = self.root / "callback-cleanup"
        cancellation = CallbackCancellation("primary callback failure")
        calls = 0

        def cancel_after_write() -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise cancellation

        cleanup_failure = archive.ArchiveError(
            archive.ArchiveErrorCode.FSYNC_FAILED,
            "injected-callback-cleanup",
            "cleanup directory fsync failed",
            self.root,
        )
        with (
            mock.patch.object(
                archive,
                "_fsync_directory",
                side_effect=cleanup_failure,
            ),
            self.assertRaises(CallbackCancellation) as context,
        ):
            archive.write_bytes_exclusive(
                destination,
                b"partial",
                max_bytes=64,
                progress_callback=cancel_after_write,
            )
        self.assertIs(context.exception, cancellation)
        self.assertFalse(destination.exists())
        self.assertTrue(
            any(
                "cleanup raised ArchiveError" in note
                for note in getattr(cancellation, "__notes__", ())
            ),
            getattr(cancellation, "__notes__", ()),
        )

    def test_mid_manifest_callback_cancellation_leaves_population_unmodified(self) -> None:
        active = self.new_directory()
        for index in range(130):
            (active / f"empty-{index:03d}").mkdir()
        before = sorted(path.name for path in active.iterdir())
        cancellation = CallbackCancellation("cancel directory scan")
        calls = 0

        def cancel_first_directory_batch() -> None:
            nonlocal calls
            calls += 1
            if calls == 3:
                raise cancellation

        with self.assertRaises(CallbackCancellation) as context:
            archive.create_manifest(
                active,
                manifest_relative_path=MANIFEST,
                max_regular_files_excluding_manifest=0,
                max_directories_excluding_root=130,
                max_regular_file_bytes=0,
                max_total_regular_file_bytes=0,
                max_manifest_bytes=64 * 1024,
                progress_callback=cancel_first_directory_batch,
            )
        self.assertIs(context.exception, cancellation)
        self.assertFalse((active / MANIFEST).exists())
        self.assertEqual(
            sorted(path.name for path in active.iterdir()),
            before,
        )

    def test_verify_callback_cancellation_is_exact_and_read_only(self) -> None:
        active = self.new_directory()
        (active / "member").write_bytes(b"v" * (2 * (64 << 10)))
        self.create_verified_manifest(active)
        before = {
            path.relative_to(active).as_posix(): (
                os.lstat(path).st_ino,
                path.read_bytes(),
            )
            for path in active.iterdir()
            if path.is_file()
        }
        cancellation = CallbackCancellation("cancel verification")
        calls = 0

        def cancel_during_verification() -> None:
            nonlocal calls
            calls += 1
            if calls == 4:
                raise cancellation

        with self.assertRaises(CallbackCancellation) as context:
            archive.verify_manifest(
                active,
                **self.manifest_limits(),
                progress_callback=cancel_during_verification,
            )
        self.assertIs(context.exception, cancellation)
        after = {
            path.relative_to(active).as_posix(): (
                os.lstat(path).st_ino,
                path.read_bytes(),
            )
            for path in active.iterdir()
            if path.is_file()
        }
        self.assertEqual(after, before)

    def test_publication_callback_brackets_every_fsync_and_rename(self) -> None:
        active = self.new_directory()
        (active / "member").write_bytes(b"member")
        expected = self.create_verified_manifest(active)
        final = self.root / "progress-final"
        events: list[str] = []
        real_fsync = archive.os.fsync
        real_rename = archive._renameat2_noreplace

        def observed_fsync(descriptor: int) -> None:
            events.append("fsync")
            real_fsync(descriptor)

        def observed_rename(source: Path, destination: Path) -> None:
            events.append("rename")
            real_rename(source, destination)

        with (
            mock.patch.object(
                archive.os,
                "fsync",
                side_effect=observed_fsync,
            ),
            mock.patch.object(
                archive,
                "_renameat2_noreplace",
                side_effect=observed_rename,
            ),
        ):
            result = archive.publish_verified_archive(
                active,
                final,
                **self.manifest_limits(),
                progress_callback=lambda: events.append("callback"),
            )
        self.assertEqual(result.summary, expected)
        self.assertEqual(result.phase, archive.PublicationPhase.VERIFIED)
        boundaries = [
            index
            for index, event in enumerate(events)
            if event in {"fsync", "rename"}
        ]
        self.assertTrue(boundaries, events)
        for index in boundaries:
            self.assertGreater(index, 0, events)
            self.assertLess(index + 1, len(events), events)
            self.assertEqual(events[index - 1], "callback", events)
            self.assertEqual(events[index + 1], "callback", events)

    def test_post_rename_callback_cancellation_preserves_exact_exception_and_stage(self) -> None:
        active = self.new_directory()
        (active / "member").write_bytes(b"member")
        self.create_verified_manifest(active)
        final = self.root / "cancelled-final"
        cancellation = CallbackCancellation("cancel after rename")
        renamed = False
        real_rename = archive._renameat2_noreplace

        def observed_rename(source: Path, destination: Path) -> None:
            nonlocal renamed
            real_rename(source, destination)
            renamed = True

        def cancel_after_rename() -> None:
            if renamed:
                raise cancellation

        with (
            mock.patch.object(
                archive,
                "_renameat2_noreplace",
                side_effect=observed_rename,
            ),
            self.assertRaises(CallbackCancellation) as context,
        ):
            archive.publish_verified_archive(
                active,
                final,
                **self.manifest_limits(),
                progress_callback=cancel_after_rename,
            )
        self.assertIs(context.exception, cancellation)
        self.assertFalse(active.exists())
        self.assertTrue(final.is_dir())
        self.assertTrue(
            any(
                "publication_phase=renamed_unsynced" in note
                for note in getattr(cancellation, "__notes__", ())
            ),
            getattr(cancellation, "__notes__", ()),
        )
        archive.verify_manifest(final, **self.manifest_limits())

    def test_progress_callback_reaches_state_journal_and_recovery_io(self) -> None:
        active = self.new_directory()
        calls = 0

        def progress() -> None:
            nonlocal calls
            calls += 1

        archive.write_state_checkpoint(
            active,
            {"sequence": 1},
            max_bytes=4096,
            progress_callback=progress,
        )
        after_state = calls
        self.assertGreater(after_state, 0)
        cursor = archive.initialize_journal(
            active,
            progress_callback=progress,
        )
        cursor = archive.append_journal(
            active,
            {"event": "STARTED"},
            cursor=cursor,
            max_entry_bytes=4096,
            progress_callback=progress,
        )
        self.assertGreater(calls, after_state)
        self.assertEqual(cursor.next_sequence, 1)
        before_recovery = calls
        inspection = archive.inspect_incomplete(
            active,
            max_state_bytes=4096,
            max_journal_entries=8,
            max_journal_entry_bytes=4096,
            **self.manifest_limits(),
            progress_callback=progress,
        )
        self.assertGreater(calls, before_recovery)
        self.assertIsNotNone(inspection.state)
        self.assertIsNotNone(inspection.journal)


    def test_callback_origin_is_exact_across_read_write_and_fsync(self) -> None:
        payload = b"origin" * (12 * 1024)
        for seam, target_call in (("read", 2), ("write", 2), ("fsync", 3)):
            for index, sentinel in enumerate(self.callback_sentinels(seam)):
                with self.subTest(seam=seam, sentinel=type(sentinel).__name__):
                    if seam == "read":
                        source = self.root / f"origin-read-{index}"
                        source.write_bytes(payload)
                        operation = lambda source=source, sentinel=sentinel: (
                            archive.read_regular_file_bounded(
                                source,
                                len(payload),
                                progress_callback=self.callback_on_call(
                                    sentinel,
                                    target_call,
                                ),
                            )
                        )
                    else:
                        destination = self.root / f"origin-{seam}-{index}"
                        operation = (
                            lambda destination=destination, sentinel=sentinel: (
                                archive.write_bytes_exclusive(
                                    destination,
                                    payload,
                                    max_bytes=len(payload),
                                    progress_callback=self.callback_on_call(
                                        sentinel,
                                        target_call,
                                    ),
                                )
                            )
                        )
                    self.assert_exact_callback_failure(sentinel, operation)
                    if seam != "read":
                        self.assertFalse(destination.exists())

    def test_callback_catching_reentrant_public_api_observes_exact_original(
        self,
    ) -> None:
        sentinel, cause, exception_context = self.chained_callback_sentinel(
            "direct reentrant public API",
        )
        outer = self.root / "reentrant-outer"
        inner = self.root / "reentrant-inner"
        observed: list[BaseException] = []
        invoked = False

        def reenter_and_catch() -> None:
            nonlocal invoked
            if invoked:
                return
            invoked = True
            try:
                archive.create_directory_exclusive(
                    inner,
                    progress_callback=lambda: (_ for _ in ()).throw(sentinel),
                )
            except BaseException as error:
                observed.append(error)

        archive.create_directory_exclusive(
            outer,
            progress_callback=reenter_and_catch,
        )
        self.assertEqual(observed, [sentinel])
        self.assertIs(observed[0].__cause__, cause)
        self.assertIs(observed[0].__context__, exception_context)
        self.assertTrue(observed[0].__suppress_context__)
        self.assert_callback_exception_graph_is_public(observed[0])
        self.assertTrue(outer.is_dir())
        self.assertFalse(inner.exists())
        self.assertEqual(archive._ARCHIVE_API_DEPTH.get(), 0)

    def test_callback_active_copied_context_observes_exact_original(
        self,
    ) -> None:
        sentinel, cause, exception_context = self.chained_callback_sentinel(
            "active copied context",
        )
        outer = self.root / "active-context-outer"
        inner = self.root / "active-context-inner"
        observed: list[BaseException] = []
        invoked = False

        def copy_reenter_and_catch() -> None:
            nonlocal invoked
            if invoked:
                return
            invoked = True
            copied = contextvars.copy_context()
            try:
                copied.run(
                    archive.create_directory_exclusive,
                    inner,
                    progress_callback=lambda: (_ for _ in ()).throw(sentinel),
                )
            except BaseException as error:
                observed.append(error)

        archive.create_directory_exclusive(
            outer,
            progress_callback=copy_reenter_and_catch,
        )
        self.assertEqual(observed, [sentinel])
        self.assertIs(observed[0].__cause__, cause)
        self.assertIs(observed[0].__context__, exception_context)
        self.assertTrue(observed[0].__suppress_context__)
        self.assert_callback_exception_graph_is_public(observed[0])
        self.assertTrue(outer.is_dir())
        self.assertFalse(inner.exists())
        self.assertEqual(archive._ARCHIVE_API_DEPTH.get(), 0)

    def test_callback_copied_context_after_return_is_clean_public_root(
        self,
    ) -> None:
        sentinel, cause, exception_context = self.chained_callback_sentinel(
            "copied context after return",
        )
        outer = self.root / "saved-context-outer"
        inner = self.root / "saved-context-inner"
        saved: list[contextvars.Context] = []

        def capture_once() -> None:
            if not saved:
                saved.append(contextvars.copy_context())

        archive.create_directory_exclusive(
            outer,
            progress_callback=capture_once,
        )
        self.assertEqual(len(saved), 1)
        self.assertEqual(
            saved[0].run(archive._ARCHIVE_API_DEPTH.get),
            0,
            "caller-copied context retained private archive depth",
        )
        with self.assertRaises(BaseException) as caught:
            saved[0].run(
                archive.create_directory_exclusive,
                inner,
                progress_callback=lambda: (_ for _ in ()).throw(sentinel),
            )
        self.assertIs(caught.exception, sentinel)
        self.assertIs(caught.exception.__cause__, cause)
        self.assertIs(caught.exception.__context__, exception_context)
        self.assertTrue(caught.exception.__suppress_context__)
        self.assert_callback_exception_graph_is_public(caught.exception)
        self.assertTrue(outer.is_dir())
        self.assertFalse(inner.exists())
        self.assertEqual(saved[0].run(archive._ARCHIVE_API_DEPTH.get), 0)
        self.assertEqual(archive._ARCHIVE_API_DEPTH.get(), 0)

    def test_callback_origin_survives_state_journal_and_manifest_nesting(self) -> None:
        for index, sentinel in enumerate(self.callback_sentinels("state")):
            with self.subTest(seam="state", sentinel=type(sentinel).__name__):
                active = self.new_directory(f"origin-state-{index}")
                replaced = False
                real_replace = archive.os.replace

                def observed_replace(source, destination) -> None:
                    nonlocal replaced
                    real_replace(source, destination)
                    replaced = True

                def cancel_after_replace() -> None:
                    if replaced:
                        raise sentinel

                with mock.patch.object(
                    archive.os,
                    "replace",
                    side_effect=observed_replace,
                ):
                    self.assert_exact_callback_failure(
                        sentinel,
                        lambda: archive.write_state_checkpoint(
                            active,
                            {"state": index},
                            max_bytes=4096,
                            progress_callback=cancel_after_replace,
                        ),
                    )
                self.assertEqual(
                    archive.read_state_checkpoint(active, max_bytes=4096),
                    {"state": index},
                )

        for index, sentinel in enumerate(self.callback_sentinels("journal")):
            with self.subTest(seam="journal", sentinel=type(sentinel).__name__):
                active = self.new_directory(f"origin-journal-{index}")
                cursor = archive.initialize_journal(active)
                synced = False
                real_fsync = archive.os.fsync

                def observed_fsync(descriptor: int) -> None:
                    nonlocal synced
                    real_fsync(descriptor)
                    synced = True

                def cancel_after_fsync() -> None:
                    if synced:
                        raise sentinel

                with mock.patch.object(
                    archive.os,
                    "fsync",
                    side_effect=observed_fsync,
                ):
                    self.assert_exact_callback_failure(
                        sentinel,
                        lambda: archive.append_journal(
                            active,
                            {"event": "STARTED"},
                            cursor=cursor,
                            max_entry_bytes=4096,
                            progress_callback=cancel_after_fsync,
                        ),
                    )
                observed = archive.read_journal(
                    active,
                    max_entries=2,
                    max_entry_bytes=4096,
                )
                self.assertEqual(len(observed.entries), 1)

        for index, sentinel in enumerate(self.callback_sentinels("manifest")):
            with self.subTest(seam="manifest", sentinel=type(sentinel).__name__):
                active = self.new_directory(f"origin-manifest-{index}")
                for directory_index in range(65):
                    (active / f"empty-{directory_index:03d}").mkdir()
                self.assert_exact_callback_failure(
                    sentinel,
                    lambda: archive.create_manifest(
                        active,
                        manifest_relative_path=MANIFEST,
                        max_regular_files_excluding_manifest=0,
                        max_directories_excluding_root=65,
                        max_regular_file_bytes=0,
                        max_total_regular_file_bytes=0,
                        max_manifest_bytes=64 * 1024,
                        progress_callback=self.callback_on_call(sentinel, 3),
                    ),
                )
                self.assertFalse((active / MANIFEST).exists())

    def test_callback_origin_is_not_remapped_by_verify_or_recovery(self) -> None:
        for index, sentinel in enumerate(self.callback_sentinels("verify")):
            with self.subTest(seam="verify", sentinel=type(sentinel).__name__):
                active = self.new_directory(f"origin-verify-{index}")
                (active / "member").write_bytes(b"member")
                self.create_verified_manifest(active)
                self.assert_exact_callback_failure(
                    sentinel,
                    lambda: archive.verify_manifest(
                        active,
                        **self.manifest_limits(),
                        progress_callback=self.callback_on_call(sentinel, 3),
                    ),
                )

        for index, sentinel in enumerate(self.callback_sentinels("recovery")):
            with self.subTest(seam="recovery", sentinel=type(sentinel).__name__):
                active = self.new_directory(f"origin-recovery-{index}")
                archive.write_state_checkpoint(
                    active,
                    {"state": index},
                    max_bytes=4096,
                )
                self.assert_exact_callback_failure(
                    sentinel,
                    lambda: archive.inspect_incomplete(
                        active,
                        max_state_bytes=4096,
                        max_journal_entries=2,
                        max_journal_entry_bytes=4096,
                        **self.manifest_limits(),
                        progress_callback=self.callback_on_call(sentinel, 2),
                    ),
                )


    def test_callback_primary_survives_cleanup_failures(self) -> None:
        for index, sentinel in enumerate(self.callback_sentinels("read-close")):
            with self.subTest(seam="read-close", sentinel=type(sentinel).__name__):
                source = self.root / f"origin-read-close-{index}"
                source.write_bytes(b"read-close")
                real_close = archive.os.close

                def close_then_fail(descriptor: int) -> None:
                    real_close(descriptor)
                    raise OSError(errno.EIO, "injected read close failure")

                with mock.patch.object(
                    archive.os,
                    "close",
                    side_effect=close_then_fail,
                ):
                    error = self.assert_exact_callback_failure(
                        sentinel,
                        lambda: archive.read_regular_file_bounded(
                            source,
                            4096,
                            progress_callback=self.callback_on_call(
                                sentinel,
                                2,
                            ),
                        ),
                    )
                self.assertTrue(
                    any(
                        "close" in note
                        for note in getattr(error, "__notes__", ())
                    ),
                    getattr(error, "__notes__", ()),
                )

    def test_scandir_callback_primary_survives_close_cleanup_failure(self) -> None:
        active = self.new_directory("origin-scan-close")
        for index in range(65):
            (active / f"empty-{index:03d}").mkdir()
        sentinel = CallbackCancellation("scan callback primary")
        real_scandir = archive.os.scandir

        class FailingScandir:
            def __init__(self, path) -> None:
                self.inner = real_scandir(path)

            def __enter__(self):
                return self

            def __iter__(self):
                return iter(self.inner)

            def __exit__(self, exception_type, exception, traceback) -> None:
                self.inner.close()
                raise OSError(errno.EIO, "injected scandir close failure")

        with mock.patch.object(
            archive.os,
            "scandir",
            side_effect=FailingScandir,
        ):
            error = self.assert_exact_callback_failure(
                sentinel,
                lambda: archive.create_manifest(
                    active,
                    manifest_relative_path=MANIFEST,
                    max_regular_files_excluding_manifest=0,
                    max_directories_excluding_root=65,
                    max_regular_file_bytes=0,
                    max_total_regular_file_bytes=0,
                    max_manifest_bytes=64 * 1024,
                    progress_callback=self.callback_on_call(sentinel, 3),
                ),
            )
        self.assertTrue(
            any("scandir" in note for note in getattr(error, "__notes__", ())),
            getattr(error, "__notes__", ()),
        )

    def test_hash_callback_primary_survives_close_cleanup_failure(self) -> None:
        active = self.new_directory("origin-hash-close")
        member = active / "member"
        member.write_bytes(b"hash-close")
        self.create_verified_manifest(active)
        sentinel = CallbackCancellation("hash callback primary")
        target_descriptor: int | None = None
        real_open = archive.os.open
        real_close = archive.os.close

        def observed_open(path, flags, mode=0o777) -> int:
            nonlocal target_descriptor
            descriptor = real_open(path, flags, mode)
            if Path(path) == member:
                target_descriptor = descriptor
            return descriptor

        def fail_target_close(descriptor: int) -> None:
            real_close(descriptor)
            if descriptor == target_descriptor:
                raise OSError(errno.EIO, "injected hash close failure")

        def cancel_hash_read() -> None:
            if target_descriptor is not None:
                raise sentinel

        with (
            mock.patch.object(archive.os, "open", side_effect=observed_open),
            mock.patch.object(archive.os, "close", side_effect=fail_target_close),
        ):
            error = self.assert_exact_callback_failure(
                sentinel,
                lambda: archive.verify_manifest(
                    active,
                    **self.manifest_limits(),
                    progress_callback=cancel_hash_read,
                ),
            )
        self.assertTrue(
            any("close" in note for note in getattr(error, "__notes__", ())),
            getattr(error, "__notes__", ()),
        )

    def test_tree_fsync_callback_primary_survives_close_cleanup_failure(self) -> None:
        active = self.new_directory("origin-tree-close")
        (active / "member").write_bytes(b"tree-close")
        self.create_verified_manifest(active)
        final = self.root / "origin-tree-close-final"
        sentinel = CallbackCancellation("tree fsync callback primary")
        fsync_completed = False
        real_fsync = archive.os.fsync
        real_close = archive.os.close

        def observed_fsync(descriptor: int) -> None:
            nonlocal fsync_completed
            real_fsync(descriptor)
            fsync_completed = True

        def close_after_fsync_then_fail(descriptor: int) -> None:
            real_close(descriptor)
            if fsync_completed:
                raise OSError(errno.EIO, "injected tree close failure")

        def cancel_after_tree_fsync() -> None:
            if fsync_completed:
                raise sentinel

        with (
            mock.patch.object(archive.os, "fsync", side_effect=observed_fsync),
            mock.patch.object(
                archive.os,
                "close",
                side_effect=close_after_fsync_then_fail,
            ),
        ):
            error = self.assert_exact_callback_failure(
                sentinel,
                lambda: archive.publish_verified_archive(
                    active,
                    final,
                    **self.manifest_limits(),
                    progress_callback=cancel_after_tree_fsync,
                ),
            )
        self.assertTrue(active.is_dir())
        self.assertFalse(final.exists())
        self.assertTrue(
            any("close" in note for note in getattr(error, "__notes__", ())),
            getattr(error, "__notes__", ()),
        )


    def test_callback_origin_is_exact_during_streaming_copy(self) -> None:
        payload = b"copy-origin" * (8 * 1024)
        source = self.root / "origin-copy-source"
        source.write_bytes(payload)
        for index, sentinel in enumerate(self.callback_sentinels("copy")):
            with self.subTest(sentinel=type(sentinel).__name__):
                destination = self.root / f"origin-copy-{index}"
                self.assert_exact_callback_failure(
                    sentinel,
                    lambda: archive.copy_file_exclusive(
                        source,
                        destination,
                        max_bytes=len(payload),
                        progress_callback=self.callback_on_call(sentinel, 3),
                    ),
                )
                self.assertFalse(destination.exists())

    def test_directory_fsync_close_cannot_replace_callback_primary(self) -> None:
        sentinel = OSError(errno.ECANCELED, "directory fsync callback")
        destination = self.root / "origin-directory-fsync"
        real_close = archive.os.close

        def close_then_fail(descriptor: int) -> None:
            real_close(descriptor)
            raise OSError(errno.EIO, "injected directory close failure")

        with mock.patch.object(
            archive.os,
            "close",
            side_effect=close_then_fail,
        ):
            error = self.assert_exact_callback_failure(
                sentinel,
                lambda: archive.create_directory_exclusive(
                    destination,
                    progress_callback=self.callback_on_call(sentinel, 3),
                ),
            )
        self.assertTrue(destination.is_dir())
        self.assertTrue(
            any("close" in note for note in getattr(error, "__notes__", ())),
            getattr(error, "__notes__", ()),
        )

    def test_callback_archive_error_always_reports_factual_publication_phase(self) -> None:
        cases = (
            (False, "before-rename", archive.PublicationPhase.PRE_RENAME),
            (False, "after-rename", archive.PublicationPhase.RENAMED_UNSYNCED),
            (False, "before-parent-fsync", archive.PublicationPhase.RENAMED_UNSYNCED),
            (False, "after-parent-fsync", archive.PublicationPhase.PARENTS_SYNCED),
            (False, "after-verify", archive.PublicationPhase.VERIFIED),
            (True, "before-rename", archive.PublicationPhase.PRE_RENAME),
            (True, "after-rename", archive.PublicationPhase.RENAMED_UNSYNCED),
            (True, "before-final-fsync", archive.PublicationPhase.RENAMED_UNSYNCED),
            (True, "after-final-fsync", archive.PublicationPhase.FINAL_PARENT_SYNCED),
            (True, "before-source-fsync", archive.PublicationPhase.FINAL_PARENT_SYNCED),
            (True, "after-source-fsync", archive.PublicationPhase.PARENTS_SYNCED),
            (True, "after-verify", archive.PublicationPhase.VERIFIED),
        )
        for index, (distinct, target, expected_phase) in enumerate(cases):
            with self.subTest(distinct=distinct, target=target):
                if distinct:
                    source_parent = self.root / f"phase-source-{index}"
                    final_parent = self.root / f"phase-final-parent-{index}"
                    archive.create_directory_exclusive(source_parent)
                    archive.create_directory_exclusive(final_parent)
                    active = source_parent / "active"
                    archive.create_directory_exclusive(active)
                    final = final_parent / "final"
                else:
                    active = self.new_directory(f"phase-active-{index}")
                    source_parent = active.parent
                    final_parent = active.parent
                    final = self.root / f"phase-final-{index}"
                (active / "member").write_bytes(b"phase member")
                self.create_verified_manifest(active)
                preloaded_phase = (
                    archive.PublicationPhase.PRE_RENAME
                    if expected_phase == archive.PublicationPhase.VERIFIED
                    else archive.PublicationPhase.VERIFIED
                )
                sentinel = archive.ArchiveError(
                    archive.ArchiveErrorCode.STATE_INVALID,
                    "callback-phase-sentinel",
                    target,
                    publication_phase=preloaded_phase,
                )
                state = {
                    "tree_done": False,
                    "renamed": False,
                    "final_synced": False,
                    "parents_synced": False,
                    "verified": False,
                    "renamed_callbacks": 0,
                    "final_callbacks": 0,
                    "parents_callbacks": 0,
                }
                real_tree = archive._fsync_tree
                real_rename = archive._renameat2_noreplace
                real_directory_fsync = archive._fsync_directory
                real_verify = archive.verify_manifest.__wrapped__

                def observed_tree(*arguments, **keywords) -> None:
                    real_tree(*arguments, **keywords)
                    state["tree_done"] = True

                def observed_rename(source: Path, destination: Path) -> None:
                    real_rename(source, destination)
                    state["renamed"] = True

                def observed_directory_fsync(path, *arguments, **keywords) -> None:
                    real_directory_fsync(path, *arguments, **keywords)
                    if not state["renamed"]:
                        return
                    if not distinct:
                        state["parents_synced"] = True
                    elif Path(path) == final_parent:
                        state["final_synced"] = True
                    elif Path(path) == source_parent:
                        state["parents_synced"] = True

                def observed_verify(root, **keywords):
                    result = real_verify(root, **keywords)
                    if Path(root) == final:
                        state["verified"] = True
                    return result

                def cancel_at_target_phase() -> None:
                    point: str | None = None
                    if state["verified"]:
                        point = "after-verify"
                    elif not state["renamed"]:
                        if state["tree_done"]:
                            point = "before-rename"
                    elif state["parents_synced"]:
                        state["parents_callbacks"] += 1
                        if state["parents_callbacks"] == 1:
                            point = (
                                "after-source-fsync"
                                if distinct
                                else "after-parent-fsync"
                            )
                    elif state["final_synced"]:
                        state["final_callbacks"] += 1
                        point = (
                            "after-final-fsync"
                            if state["final_callbacks"] == 1
                            else "before-source-fsync"
                        )
                    else:
                        state["renamed_callbacks"] += 1
                        if state["renamed_callbacks"] == 1:
                            point = "after-rename"
                        elif state["renamed_callbacks"] == 2:
                            point = (
                                "before-final-fsync"
                                if distinct
                                else "before-parent-fsync"
                            )
                    if point == target:
                        raise sentinel

                with (
                    mock.patch.object(
                        archive,
                        "_fsync_tree",
                        side_effect=observed_tree,
                    ),
                    mock.patch.object(
                        archive,
                        "_renameat2_noreplace",
                        side_effect=observed_rename,
                    ),
                    mock.patch.object(
                        archive,
                        "_fsync_directory",
                        side_effect=observed_directory_fsync,
                    ),
                    mock.patch.object(
                        archive.verify_manifest,
                        "__wrapped__",
                        new=observed_verify,
                    ),
                ):
                    error = self.assert_exact_callback_failure(
                        sentinel,
                        lambda: archive.publish_verified_archive(
                            active,
                            final,
                            **self.manifest_limits(),
                            progress_callback=cancel_at_target_phase,
                        ),
                    )
                self.assertEqual(error.publication_phase, expected_phase)
                if expected_phase == archive.PublicationPhase.PRE_RENAME:
                    self.assertTrue(active.is_dir())
                    self.assertFalse(final.exists())
                else:
                    self.assertFalse(active.exists())
                    self.assertTrue(final.is_dir())


if __name__ == "__main__":
    unittest.main()
