#!/usr/bin/env python3
"""Clean-build red/green regression for authenticated archive composition."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import os
import pathlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
import uuid


ROOT = pathlib.Path(__file__).resolve().parents[4]
COMPOSER_PATH = ROOT / "tools/profiling/memory/oai_memprof_archive_composer.py"
API_DEFINITION_SHA256 = (
    "93056c4cfd071c1df396ba09bf82b4cbe807923977c4bca988b0aee1b8c94610"
)


def load(name: str, path: pathlib.Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"module unavailable: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


COMPOSER = load("_oai_memprof_archive_composer_test", COMPOSER_PATH)
BUILD_EVIDENCE = COMPOSER.BUILD_EVIDENCE
COVERAGE = COMPOSER.VERIFIER.COVERAGE
CONFIG = COMPOSER.VERIFIER.CONFIG
WIRE = COMPOSER.VERIFIER.WIRE
HANDOFF = COMPOSER.HANDOFF
SELECTION = COMPOSER.VERIFIER.SELECTION


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


TRUSTED_RELEASE_AUTHORITY_FIXTURE_SHA256 = (
    "48b7946d9338cd455fe6b2f48a4fc7a7fc1c7a46cd22c5c5c7ae029edb3d1a16"
)


class BuildArtifactPreflightTests(unittest.TestCase):
    """Reject hostile artifact plans before the first pathname is opened."""

    @staticmethod
    def row(
        index: int = 0,
        *,
        byte_count: int = 0,
        digest: str | None = None,
        path: str | None = None,
    ) -> dict:
        return {
            "bytes": byte_count,
            "path": path or f"input/build-evidence/artifact-{index}",
            "sha256": digest or sha(b""),
        }

    def assert_preflight_rejects_without_read(
        self, entries: list, message: str
    ) -> None:
        with mock.patch.object(
            COMPOSER,
            "_read_frozen",
            side_effect=AssertionError("artifact reader was invoked"),
        ) as reader:
            with self.assertRaisesRegex(COMPOSER.ArchiveComposerError, message):
                COMPOSER._read_build_evidence_artifacts(
                    {"entries": entries}, pathlib.Path("/not-opened")
                )
        reader.assert_not_called()

    def test_invalid_plans_reject_before_artifact_open(self) -> None:
        over_count = [
            self.row(index) for index in range(COMPOSER.MAX_BUILD_ARTIFACT_ENTRIES + 1)
        ]
        aggregate = [
            self.row(0, byte_count=COMPOSER.MAX_BUILD_ARTIFACT_BYTES),
            self.row(1, byte_count=COMPOSER.MAX_BUILD_ARTIFACT_BYTES),
            self.row(2, byte_count=1),
        ]
        malformed_shape = [self.row() | {"unexpected": True}]
        boolean_size = [self.row(byte_count=True)]
        uppercase_digest = [self.row(digest="A" * 64)]
        malformed_path = [self.row(path="input/build-evidence/../artifact")]
        duplicate_path = [self.row(0), self.row(0)]
        for label, entries, message in (
            ("over-count", over_count, "artifact-entry limit exceeded"),
            ("aggregate", aggregate, "aggregate declared artifact bytes exceeded"),
            ("shape", malformed_shape, "exact members required"),
            ("boolean-size", boolean_size, "bounded nonnegative bytes required"),
            ("uppercase-digest", uppercase_digest, "lowercase SHA-256 required"),
            ("path", malformed_path, "normalized archive-relative POSIX path required"),
            ("duplicate", duplicate_path, "duplicate artifact path"),
        ):
            with self.subTest(label=label):
                self.assert_preflight_rejects_without_read(entries, message)

    def test_actual_artifact_mismatch_rejects_after_one_bounded_read(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-memprof-artifact-plan-") as root:
            evidence_root = pathlib.Path(root)
            artifact_path = evidence_root / "input/build-evidence/artifact-0"
            artifact_path.parent.mkdir(parents=True)
            artifact_path.write_bytes(b"x")
            with mock.patch.object(
                COMPOSER, "_read_frozen", wraps=COMPOSER._read_frozen
            ) as reader:
                with self.assertRaisesRegex(
                    COMPOSER.ArchiveComposerError, "path/size/digest mismatch"
                ):
                    COMPOSER._read_build_evidence_artifacts(
                        {"entries": [self.row(byte_count=1, digest="0" * 64)]},
                        evidence_root,
                    )
            self.assertEqual(reader.call_count, 1)
            self.assertEqual(reader.call_args.args[1], 1)


def replace_maps_file_identity(
    raw: bytes,
    source_path: pathlib.Path,
    *,
    replacement_path: pathlib.Path | None = None,
    replacement_device: int | None = None,
    replacement_inode: int | None = None,
) -> bytes:
    """Replace every maps group for one absolute path with a test identity."""

    source = os.fsencode(str(source_path))
    result = []
    replacements = 0
    for line in raw.splitlines():
        match = COMPOSER._MAP_RE.fullmatch(line)
        if match is None:
            raise AssertionError(f"fixture maps grammar mismatch: {line!r}")
        start, end, permissions, offset, major, minor, inode, path = match.groups()
        if path != source:
            result.append(line)
            continue
        current_device = os.makedev(int(major, 16), int(minor, 16))
        device = current_device if replacement_device is None else replacement_device
        new_inode = int(inode) if replacement_inode is None else replacement_inode
        new_path = source if replacement_path is None else os.fsencode(str(replacement_path))
        result.append(
            b" ".join(
                (
                    start + b"-" + end,
                    permissions,
                    offset,
                    f"{os.major(device):x}:{os.minor(device):x}".encode("ascii"),
                    str(new_inode).encode("ascii"),
                    new_path,
                )
            )
        )
        replacements += 1
    if replacements == 0:
        raise AssertionError(f"fixture maps lack {source_path}")
    return b"\n".join(result) + (b"\n" if raw.endswith(b"\n") else b"")


def handoff_with_writer_outcome(
    handoff_raw: bytes, *, writer_status: int, clock_status: int
) -> bytes:
    source = HANDOFF.decode_process_handoff(handoff_raw)
    if source.writer.status != 0 or not source.writer.prefooter_closed:
        raise AssertionError("fixture source must be a closed successful handoff")
    if (writer_status, clock_status) not in ((9, 0), (11, 5)):
        raise AssertionError("fixture supports only IO_ERROR and CLOCK_ERROR")
    mutant = bytearray(handoff_raw)
    mutant[104:112] = (1).to_bytes(8, "little")
    mutant[640:644] = writer_status.to_bytes(4, "little")
    mutant[648:652] = clock_status.to_bytes(4, "little")
    mutant[656:660] = (5 if writer_status == 9 else 0).to_bytes(4, "little")
    mutant[-32:] = hashlib.sha256(mutant[:-32]).digest()
    encoded = bytes(mutant)
    decoded = HANDOFF.decode_process_handoff(encoded)
    if (
        decoded.writer.status != writer_status
        or decoded.writer.clock_status != clock_status
        or decoded.writer_io_or_finalization_failures != 1
        or not decoded.writer.prefooter_closed
    ):
        raise AssertionError("negative handoff fixture outcome mismatch")
    return encoded


def run(
    arguments: tuple[str, ...],
    *,
    cwd: pathlib.Path | None = None,
    environment: dict[str, str] | None = None,
) -> bytes:
    base_environment = {
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if environment:
        base_environment.update(environment)
    completed = subprocess.run(
        arguments,
        cwd=None if cwd is None else str(cwd),
        env=base_environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed {completed.returncode}: {arguments!r}\n"
            f"stdout:\n{completed.stdout.decode('utf-8', 'replace')}\n"
            f"stderr:\n{completed.stderr.decode('utf-8', 'replace')}"
        )
    return completed.stdout


class SoftmodemInstrumentationBoundaryTests(unittest.TestCase):
    def test_profiler_starts_after_late_modules_and_before_workload_release(self) -> None:
        gnb = (ROOT / "executables/nr-softmodem.c").read_text(encoding="utf-8")
        ue = (ROOT / "executables/nr-uesoftmodem.c").read_text(encoding="utf-8")

        self.assertEqual(gnb.count("oai_memprof_softmodem_session_start_v1("), 1)
        gnb_main = gnb.index("int main(")
        gnb_init_ru = gnb.index("init_NR_RU(uniqCfg", gnb_main)
        gnb_wait_ru = gnb.rindex("wait_RUs();")
        gnb_init_after_ru = gnb.rindex("init_eNB_afterRU();")
        gnb_start = gnb.index("oai_memprof_softmodem_session_start_v1(", gnb_main)
        gnb_release = gnb.index("sync_var=0;", gnb_wait_ru)
        self.assertLess(gnb_init_ru, gnb_wait_ru)
        self.assertLess(gnb_wait_ru, gnb_init_after_ru)
        self.assertLess(gnb_init_after_ru, gnb_start)
        self.assertLess(gnb_start, gnb_release)

        self.assertEqual(ue.count("oai_memprof_softmodem_session_start_v1("), 1)
        ue_main = ue.index("int main(")
        ue_join = ue.index("pthread_join(ru_start_thread", ue_main)
        ue_start = ue.index("oai_memprof_softmodem_session_start_v1(", ue_main)
        ue_release = ue.index("init_NR_UE_threads(", ue_join)
        self.assertLess(ue_join, ue_start)
        self.assertLess(ue_start, ue_release)


class BuildEvidenceCommandCaptureTests(unittest.TestCase):
    def test_composer_handoff_limit_matches_authenticated_decoder(self) -> None:
        self.assertEqual(HANDOFF.MAX_WIRE_BYTES, 46_203_168)
        self.assertEqual(COMPOSER.MAX_HANDOFF_BYTES, HANDOFF.MAX_WIRE_BYTES)

    def test_supported_api_table_is_exactly_coverage_policy(self) -> None:
        expected = tuple(
            (
                row["api_id"],
                row["import_symbol"],
                row["wrapper_symbol"],
                row["wrap_option"],
            )
            for row in BUILD_EVIDENCE.COVERAGE.API_RULES
        )
        self.assertEqual(BUILD_EVIDENCE.VERSION, {"major": 1, "minor": 4})
        self.assertEqual(BUILD_EVIDENCE._SUPPORTED, expected)
        self.assertEqual([row[0] for row in expected], list(range(1, 13)))
        self.assertEqual(
            BUILD_EVIDENCE.COVERAGE.expected_supported_symbol_version(5, 1),
            "GLIBC_2.26",
        )
        self.assertEqual(
            BUILD_EVIDENCE.COVERAGE.expected_supported_symbol_version(6, 1),
            "GLIBC_2.16",
        )
        self.assertEqual(
            BUILD_EVIDENCE.COVERAGE.expected_supported_symbol_version(6, 2),
            "GLIBC_2.17",
        )

    def test_archive_appender_target_is_guarded_production_only_once(self) -> None:
        cmake = (ROOT / "common/utils/memprof/CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        tests_cmake = (
            ROOT / "common/utils/memprof/tests/CMakeLists.txt"
        ).read_text(encoding="utf-8")
        marker = "add_executable(oai_memprof_archive_append EXCLUDE_FROM_ALL"
        self.assertEqual(cmake.count(marker), 1)
        self.assertNotIn("add_executable(oai_memprof_archive_append", tests_cmake)
        start = cmake.rfind("if(", 0, cmake.index(marker))
        end = cmake.index("endif()", cmake.index(marker))
        block = " ".join(cmake[start:end].split())
        self.assertTrue(block.startswith("if(OAI_MEMPROF_ACTIVE OR ENABLE_TESTS)"))
        self.assertIn("oai_memprof_stream_finalizer.c", block)
        self.assertIn("OAI_MEMPROF_STREAM_FINALIZER_OFFLINE_ONLY=1", block)
        self.assertIn("-static", block)
        self.assertIn("-no-pie", block)
        self.assertIn("LINKER:-Map,${OAI_MEMPROF_ARCHIVE_APPEND_MAP}", block)
        self.assertIn("LINKER:--cref", block)
        unit_marker = "add_executable(test_oai_memprof_archive_append"
        unit_start = tests_cmake.index(unit_marker)
        unit_end = tests_cmake.index(
            "add_dependencies(tests test_oai_memprof_archive_append)", unit_start
        )
        unit_block = " ".join(tests_cmake[unit_start:unit_end].split())
        self.assertIn("../oai_memprof_stream_finalizer.c", unit_block)
        self.assertIn("OAI_MEMPROF_STREAM_FINALIZER_OFFLINE_ONLY=1", unit_block)
        self.assertIn("-static", unit_block)
        self.assertIn("-no-pie", unit_block)
        self.assertNotIn("oai_memprof_stream_finalizer)", unit_block)

    def test_dfts_wrap_set_is_architecture_exact(self) -> None:
        cmake = (ROOT / "common/utils/memprof/CMakeLists.txt").read_text(encoding="utf-8")
        dfts_marker = cmake.index("oai_memprof_wrap_active_c(dfts")
        x86_marker = cmake.rfind("if(", 0, dfts_marker)
        self.assertNotEqual(x86_marker, -1)
        aarch_marker = cmake.index("elseif(", dfts_marker)
        else_marker = cmake.index("else()", aarch_marker)
        x86_block = " ".join(cmake[x86_marker:aarch_marker].split())
        aarch_block = " ".join(cmake[aarch_marker:else_marker].split())

        self.assertIn("x86_64", x86_block)
        self.assertIn("amd64", x86_block)
        self.assertEqual(x86_block.count("oai_memprof_wrap_active_c(dfts"), 1)
        self.assertIn("oai_memprof_wrap_active_c(dfts free aligned_alloc posix_memalign)", x86_block)
        self.assertIn("aarch64", aarch_block)
        self.assertIn("arm64", aarch_block)
        self.assertEqual(aarch_block.count("oai_memprof_add_active_map(dfts)"), 1)
        self.assertNotIn("oai_memprof_wrap_active_c(dfts", aarch_block)

    def test_rfsimulator_wrap_set_is_conditional_and_exact(self) -> None:
        cmake = (ROOT / "common/utils/memprof/CMakeLists.txt").read_text(encoding="utf-8")
        rfsimulator_marker = cmake.index("if(TARGET rfsimulator)")
        rfsimulator_end = cmake.index("endif()", rfsimulator_marker) + len("endif()")
        rfsimulator_block = " ".join(cmake[rfsimulator_marker:rfsimulator_end].split())

        self.assertEqual(
            rfsimulator_block,
            (
                "if(TARGET rfsimulator) "
                "oai_memprof_wrap_active_c(rfsimulator malloc calloc free strdup) "
                "elseif(OAI_SIMU) "
                "message(FATAL_ERROR "
                "\"OAI_MEMPROF_ACTIVE with OAI_SIMU requires target rfsimulator\") "
                "endif()"
            ),
        )

    def test_requests_only_final_ninja_command(self) -> None:
        expected = b"final link command\n"
        ninja = pathlib.Path("/usr/bin/ninja")
        build = pathlib.Path("/tmp/oai-build")
        target = "common/utils/memprof/liboai_memprof_active_runtime.so.1.0.0"
        self.assertEqual(
            BUILD_EVIDENCE._target(target, "test.target"), target
        )
        with mock.patch.object(
            BUILD_EVIDENCE, "_run", return_value=expected
        ) as runner:
            actual = BUILD_EVIDENCE._ninja_final_command(
                ninja, build, target
            )
        self.assertEqual(actual, expected)
        runner.assert_called_once_with(
            (
                str(ninja),
                "-C",
                str(build),
                "-t",
                "commands",
                "-s",
                target,
            )
        )


class ArchiveComposerBuildEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case_root = pathlib.Path(
            tempfile.mkdtemp(prefix="oai-memprof-archive-composer-test-")
        )
        print(f"retained_archive_composer_test_root={cls.case_root}")
        cls.source_root = cls.case_root / "source"
        cls.build_root = cls.case_root / "build"
        cls.source_root.joinpath("common/utils").mkdir(parents=True)
        shutil.copytree(
            ROOT / "common/utils/memprof",
            cls.source_root / "common/utils/memprof",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        for repository_relative_path in (
            COMPOSER.VERIFIER.TRUSTED_RELEASE_SOURCE_PATHS.values()
        ):
            source = ROOT / repository_relative_path
            destination = cls.source_root / repository_relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        cls.source_root.joinpath("unselected_fixture.c").write_text(
            "#include <unistd.h>\n"
            "pid_t oai_memprof_unselected_fixture(void) { return getpid(); }\n",
            encoding="utf-8",
        )
        cls.source_root.joinpath("CMakeLists.txt").write_text(
            "\n".join(
                (
                    "cmake_minimum_required(VERSION 3.19)",
                    "project(oai_memprof_archive_composer_test LANGUAGES C)",
                    "set(OPENAIR_DIR \"${CMAKE_CURRENT_SOURCE_DIR}\")",
                    "set(ENABLE_TESTS ON)",
                    "add_custom_target(tests)",
                    "enable_testing()",
                    "add_subdirectory(common/utils/memprof memprof)",
                    "add_library(oai_memprof_unselected_fixture SHARED unselected_fixture.c)",
                    "target_link_options(oai_memprof_unselected_fixture PRIVATE",
                    "  \"LINKER:-Map,${CMAKE_CURRENT_BINARY_DIR}/oai_memprof_unselected_fixture.map\"",
                    "  \"LINKER:--cref\")",
                    "",
                )
            ),
            encoding="utf-8",
        )
        run(("/usr/bin/git", "init", "-q"), cwd=cls.source_root)
        run(
            (
                "/usr/bin/git",
                "add",
                "CMakeLists.txt",
                "common",
                "tools",
                "unselected_fixture.c",
            ),
            cwd=cls.source_root,
        )
        run(
            (
                "/usr/bin/git",
                "-c",
                "user.name=OAI Memory Profiler Evidence",
                "-c",
                "user.email=memprof-evidence@invalid.example",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-q",
                "-m",
                "Freeze clean profiler evidence fixture",
            ),
            cwd=cls.source_root,
            environment={
                "GIT_AUTHOR_DATE": "2026-08-14T01:00:00+02:00",
                "GIT_COMMITTER_DATE": "2026-08-14T01:00:00+02:00",
            },
        )
        cls.assert_clean_source()
        run(
            (
                "/usr/bin/cmake",
                "-S",
                str(cls.source_root),
                "-B",
                str(cls.build_root),
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
                "-DCMAKE_C_COMPILER=/usr/bin/gcc",
            )
        )
        run(
            (
                "/usr/bin/cmake",
                "--build",
                str(cls.build_root),
                "--target",
                "test_oai_memprof_archive_producer",
                "oai_memprof_active_runtime",
                "oai_memprof_archive_append",
                "oai_memprof_unselected_fixture",
                "--parallel",
                "1",
            )
        )
        cls.assert_clean_source()
        cls.producer = (
            cls.build_root / "memprof/tests/test_oai_memprof_archive_producer"
        )
        cls.appender = cls.build_root / "memprof/oai_memprof_archive_append"
        cls.evidence_root = cls.case_root / "prepared"
        cls.evidence_root.mkdir()
        logical_elfs = (
            BUILD_EVIDENCE.LogicalElfInput(
                "archive_fixture",
                "memprof/tests/test_oai_memprof_archive_producer",
                cls.producer,
                cls.build_root
                / "memprof/tests/test_oai_memprof_archive_producer.map",
                "common/utils/memprof/tests/test_oai_memprof_archive_producer",
                1,
                (1,),
            ),
            BUILD_EVIDENCE.LogicalElfInput(
                "oai_memprof_active_runtime",
                "memprof/liboai_memprof_active_runtime.so.1.0.0",
                cls.build_root
                / "memprof/liboai_memprof_active_runtime.so.1.0.0",
                cls.build_root / "memprof/liboai_memprof_active_runtime.map",
                "common/utils/memprof/liboai_memprof_active_runtime.so.1",
                2,
                (1, 2),
            ),
            BUILD_EVIDENCE.LogicalElfInput(
                "unselected_fixture",
                "liboai_memprof_unselected_fixture.so",
                cls.build_root / "liboai_memprof_unselected_fixture.so",
                cls.build_root / "oai_memprof_unselected_fixture.map",
                "unselected_fixture.c",
                2,
                (1,),
                module_selection={
                    "operator_id": 1,
                    "predicates": [
                        {
                            "configuration_key": "unselected_fixture_enabled",
                            "expected_value": "1",
                            "predicate_id": 2,
                        }
                    ],
                },
            ),
        )
        cls.prepared = BUILD_EVIDENCE.prepare_measured_build_evidence(
            repository=cls.source_root,
            build_directory=cls.build_root,
            evidence_root=cls.evidence_root,
            logical_elfs=logical_elfs,
            auxiliary_executables=(
                BUILD_EVIDENCE.AuxiliaryExecutableInput(
                    BUILD_EVIDENCE.ARCHIVE_APPEND_AUXILIARY_ID,
                    BUILD_EVIDENCE.ARCHIVE_APPEND_TARGET,
                    cls.appender,
                    cls.build_root / "memprof/oai_memprof_archive_append.map",
                ),
            ),
            primary_logical_elf_id="archive_fixture",
            libc_path=pathlib.Path(
                "/lib/x86_64-linux-gnu/libc.so.6"
            ).resolve(),
            api_definition_sha256=API_DEFINITION_SHA256,
        )
        cls.artifacts = dict(cls.prepared.artifacts)
        cls.evidence_value = COVERAGE.parse_canonical(
            cls.prepared.evidence_bytes
        )
        cls.build = COVERAGE.parse_canonical(cls.prepared.coverage_bytes)
        cls.assert_clean_source()
        trusted_sources = {
            archive_path: (
                cls.source_root / repository_relative_path
            ).read_bytes()
            for archive_path, repository_relative_path in COMPOSER.VERIFIER.TRUSTED_RELEASE_SOURCE_PATHS.items()
        }
        identity = cls.build["build_identity"]
        cls.trusted_release_authority_raw = (
            COMPOSER.VERIFIER.make_trusted_release_authority_bytes(
                commit=identity["source_commit"],
                tree=identity["source_tree"],
                source_bytes=trusted_sources,
            )
        )
        actual_authority_sha256 = sha(cls.trusted_release_authority_raw)
        if actual_authority_sha256 != TRUSTED_RELEASE_AUTHORITY_FIXTURE_SHA256:
            raise AssertionError(
                "trusted-release authority fixture changed; review and explicitly "
                "update its literal pin "
                f"(observed {actual_authority_sha256})"
            )

    @classmethod
    def assert_clean_source(cls) -> None:
        status = run(
            (
                "/usr/bin/git",
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
            ),
            cwd=cls.source_root,
        )
        if status != b"":
            raise AssertionError(f"clean fixture source became dirty: {status!r}")

    @classmethod
    def write_evidence(
        cls,
        root: pathlib.Path,
        evidence_raw: bytes,
        artifacts: dict[str, bytes],
    ) -> None:
        root.mkdir()
        for relative, raw in artifacts.items():
            path = root.joinpath(*relative.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        path = root / BUILD_EVIDENCE.EVIDENCE_ARCHIVE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(evidence_raw)

    @classmethod
    def paired_mutation(
        cls, relative_path: str, replacement: bytes
    ) -> tuple[bytes, dict[str, bytes]]:
        value = copy.deepcopy(cls.evidence_value)
        artifacts = dict(cls.artifacts)
        artifacts[relative_path] = replacement
        row = next(item for item in value["entries"] if item["path"] == relative_path)
        row["bytes"] = len(replacement)
        row["sha256"] = sha(replacement)
        value["build_coverage_sha256"] = "0" * 64
        derived = BUILD_EVIDENCE._derive(
            value,
            artifacts,
            api_definition_sha256=API_DEFINITION_SHA256,
            enforce_advertised_digest=False,
        )
        value["build_coverage_sha256"] = sha(derived)
        return COVERAGE.canonical_bytes(value), artifacts

    @classmethod
    def opening_bytes(cls, config_raw: bytes) -> bytes:
        _members, bundle = COMPOSER.VERIFIER._accepted_static_members()
        primary = next(
            row
            for row in cls.build["entries"]
            if row["logical_id"]
            == cls.build["build_identity"]["primary_logical_elf_id"]
        )

        return WIRE.encode_opening_header(
            WIRE.OpeningHeader(
                page_size_bytes=4096,
                scope_kind=1,
                role_kind=1,
                clock_kind=1,
                calibration_kind=1,
                process_generation=1,
                counter_frequency_numerator=1,
                counter_frequency_denominator=1,
                calibration_error_bound_ns=1_000_000,
                calibration_span_ns=0,
                start_counter=1,
                start_monotonic_raw_ns=1,
                start_realtime_unix_ns=1,
                pid=1,
                configured_thread_capacity=1,
                run_uuid=uuid.UUID(
                    "00112233-4455-4677-8899-aabbccddeeff"
                ).bytes,
                process_uuid=uuid.UUID(
                    "ffeeddcc-bbaa-4988-8776-655443322110"
                ).bytes,
                source_object_kind=1,
                source_object_algorithm=1,
                source_object_length=20,
                source_object_value=bytes.fromhex(
                    cls.build["build_identity"]["source_commit"]
                )
                + bytes(12),
                primary_binary_sha256=bytes.fromhex(primary["sha256"]),
                schema_bundle_definition_sha256=hashlib.sha256(bundle).digest(),
                api_catalog_definition_sha256=bytes.fromhex(
                    COMPOSER.VERIFIER.ACCEPTED_MEMBER_SHA256[4]
                ),
                callsite_catalog_definition_sha256=bytes.fromhex(
                    COMPOSER.VERIFIER.ACCEPTED_MEMBER_SHA256[5]
                ),
                configuration_instance_sha256=hashlib.sha256(
                    config_raw
                ).digest(),
                primary_build_id_sha256=hashlib.sha256(
                    bytes.fromhex(primary["build_id"])
                ).digest(),
            )
        )

    @classmethod
    def trusted_release_inputs(
        cls, label: str
    ) -> tuple[pathlib.Path, pathlib.Path, str]:
        authority_path = cls.case_root / f"{label}-trusted-release-authority-v1.json"
        authority_path.write_bytes(cls.trusted_release_authority_raw)
        return (
            authority_path,
            cls.source_root,
            TRUSTED_RELEASE_AUTHORITY_FIXTURE_SHA256,
        )

    def produce(
        self,
        label: str,
        *,
        evidence_raw: bytes | None = None,
        artifacts: dict[str, bytes] | None = None,
        bound_digest: str | None = None,
        bound_authority_digest: str | None = None,
        config_output_directory: str | None = None,
        mode_id: int = 4,
        preload_unselected_fixture: bool = False,
        runtime_threshold_delta: int = 0,
        unselected_fixture_enabled: str | None = "0",
    ) -> tuple[pathlib.Path, pathlib.Path, bytes, bytes]:
        evidence_raw = (
            self.prepared.evidence_bytes if evidence_raw is None else evidence_raw
        )
        artifacts = dict(self.artifacts) if artifacts is None else artifacts
        evidence_root = self.case_root / f"{label}-evidence"
        self.write_evidence(evidence_root, evidence_raw, artifacts)
        archive = self.case_root / f"{label}-archive"
        archive.joinpath("streams").mkdir(parents=True)
        bound_digest = sha(evidence_raw) if bound_digest is None else bound_digest
        _authority_path, _authority_root, authority_digest = self.trusted_release_inputs(
            label
        )
        bound_authority_digest = (
            authority_digest
            if bound_authority_digest is None
            else bound_authority_digest
        )
        if mode_id == 3:
            sample_seed_bytes = b""
            sample_seed_k = 0
            sample_threshold = 0
            for _attempt in range(16):
                sample_seed_bytes = os.getrandom(8)
                if len(sample_seed_bytes) != 8:
                    raise AssertionError("getrandom did not return eight bytes")
                sample_seed_k = int.from_bytes(sample_seed_bytes, "big")
                values = tuple(
                    SELECTION.selection_value(1, 1, sequence, sample_seed_k)
                    for sequence in range(1, 5)
                )
                if max(values) != (1 << 64) - 1:
                    sample_threshold = max(values) + 1
                    break
            if sample_threshold == 0:
                raise AssertionError("bounded getrandom draws did not admit exact fixture q")
            sample_seed_hex = sample_seed_bytes.hex()
            sample_seed_provenance_id = 1
            sample_seed_status_id = 1
        elif mode_id == 4:
            sample_seed_k = 0
            sample_seed_hex = None
            sample_seed_provenance_id = 20
            sample_seed_status_id = 20
            sample_threshold = 0
        else:
            raise AssertionError("fixture mode must be 3 or 4")
        selection_values = [
            {"key": "build_evidence_sha256", "value": bound_digest},
            {
                "key": "trusted_release_authority_sha256",
                "value": bound_authority_digest,
            },
        ]
        if unselected_fixture_enabled is not None:
            selection_values.append(
                {
                    "key": "unselected_fixture_enabled",
                    "value": unselected_fixture_enabled,
                }
            )
        config = CONFIG.make_effective_configuration(
            flush_records=4,
            flush_us=1_000,
            max_threads=1,
            mode_id=mode_id,
            output_directory=(
                str(archive)
                if config_output_directory is None
                else config_output_directory
            ),
            ring_records=64,
            role_id=1,
            run_id=label.replace("_", "-"),
            sample_seed_hex=sample_seed_hex,
            sample_seed_provenance_id=sample_seed_provenance_id,
            sample_seed_status_id=sample_seed_status_id,
            sample_threshold=sample_threshold,
            scope_kind=1,
            selection_values=selection_values,
            table_entries=512,
            table_probes=8,
        )
        config_raw = CONFIG.serialize_effective_configuration(config)
        config_path = self.case_root / f"{label}-effective-config.json"
        opening_path = self.case_root / f"{label}-opening.bin"
        build_path = self.case_root / f"{label}-build-coverage.json"
        config_path.write_bytes(config_raw)
        opening_path.write_bytes(self.opening_bytes(config_raw))
        build_path.write_bytes(self.prepared.coverage_bytes)
        arguments = (
            str(self.producer),
            str(archive),
            str(config_path),
            str(opening_path),
        )
        if mode_id == 3:
            runtime_threshold = sample_threshold + runtime_threshold_delta
            if not 1 <= runtime_threshold <= (1 << 64) - 1:
                raise AssertionError("runtime threshold mutation left u64 nonzero domain")
            arguments += (
                "3",
                str(sample_seed_k),
                str(runtime_threshold),
            )
        producer_environment = {
            "LD_LIBRARY_PATH": str(self.build_root / "memprof")
        }
        if preload_unselected_fixture:
            producer_environment["LD_PRELOAD"] = str(
                (self.build_root / "liboai_memprof_unselected_fixture.so").resolve(
                    strict=True
                )
            )
        output = run(
            arguments,
            environment=producer_environment,
        )
        self.assertIn(b"archive producer emitted", output)
        stream = archive / "streams/memory-lifetime.bin"
        handoff = archive / "streams/process-handoff.bin"
        return archive, evidence_root, stream.read_bytes(), handoff.read_bytes()

    def sealed_appender(self, path: pathlib.Path) -> COMPOSER._SealedAppender:
        return COMPOSER._create_sealed_appender(path.read_bytes())

    def probe_contract_path(self, path: pathlib.Path) -> None:
        appender = self.sealed_appender(path)
        try:
            COMPOSER._require_appender_contract(appender)
        finally:
            appender.close()

    def compose(
        self,
        archive: pathlib.Path,
        evidence_root: pathlib.Path,
        *,
        append_executable: pathlib.Path | None = None,
    ):
        label = archive.name.removesuffix("-archive")
        authority_path, authority_root, authority_sha256 = self.trusted_release_inputs(
            label
        )
        return COMPOSER.compose(
            archive,
            build_coverage_path=self.case_root / f"{label}-build-coverage.json",
            build_evidence_path=evidence_root
            / BUILD_EVIDENCE.EVIDENCE_ARCHIVE_PATH,
            build_evidence_root=evidence_root,
            append_executable=(
                self.appender if append_executable is None else append_executable
            ),
            trusted_release_authority_path=authority_path,
            trusted_release_source_root=authority_root,
            trusted_release_authority_sha256=authority_sha256,
        )

    def handoff_substituting_appender(
        self,
        archive: pathlib.Path,
        *,
        label: str,
        replacement: bytes,
    ) -> pathlib.Path:
        process_directory = archive / "streams"
        replacement_path = process_directory / f"{label}-handoff-replacement.bin"
        replacement_path.write_bytes(replacement)
        handoff_leaf = pathlib.PurePosixPath(
            COMPOSER.PROCESS_HANDOFF_ARCHIVE_PATH
        ).name
        wrapper = self.case_root / f"{label}-handoff-substituting-appender.py"
        wrapper.write_text(
            "\n".join(
                (
                    "#!/usr/bin/python3",
                    "import os",
                    "import sys",
                    f"if sys.argv[1:] == [{COMPOSER.APPENDER_CONTRACT_PROBE_ARGUMENT!r}]:",
                    f"    os.execv({str(self.appender)!r}, [{str(self.appender)!r}, *sys.argv[1:]])",
                    f"os.replace({str(replacement_path)!r}, os.path.join(sys.argv[1], {handoff_leaf!r}))",
                    f"os.execv({str(self.appender)!r}, [{str(self.appender)!r}, *sys.argv[1:]])",
                    "",
                )
            ),
            encoding="utf-8",
        )
        wrapper.chmod(0o700)
        return wrapper

    def contract_probe_executable(
        self,
        label: str,
        *,
        stdout_text: str,
        stderr_text: str,
        return_code: int,
    ) -> pathlib.Path:
        wrapper = self.case_root / f"{label}-contract-probe-appender.py"
        wrapper.write_text(
            "\n".join(
                (
                    "#!/usr/bin/python3",
                    "import sys",
                    f"sys.stdout.write({stdout_text!r})",
                    f"sys.stderr.write({stderr_text!r})",
                    f"raise SystemExit({return_code})",
                    "",
                )
            ),
            encoding="utf-8",
        )
        wrapper.chmod(0o700)
        return wrapper

    def sleeping_contract_probe_executable(
        self, label: str
    ) -> tuple[pathlib.Path, pathlib.Path]:
        pid_path = self.case_root / f"{label}-contract-probe.pid"
        wrapper = self.case_root / f"{label}-contract-probe-sleeper.py"
        wrapper.write_text(
            "\n".join(
                (
                    "#!/usr/bin/python3",
                    "import os",
                    "import pathlib",
                    "import time",
                    f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid()), encoding='ascii')",
                    "time.sleep(30)",
                    "",
                )
            ),
            encoding="utf-8",
        )
        wrapper.chmod(0o700)
        return wrapper, pid_path

    def descendant_contract_probe_executable(
        self,
        label: str,
        *,
        alternate_process_group: bool = False,
    ) -> tuple[pathlib.Path, pathlib.Path]:
        child_path = self.case_root / f"{label}-contract-probe-child.pid"
        process_path = self.case_root / f"{label}-contract-probe-processes.txt"
        wrapper = self.case_root / f"{label}-contract-probe-descendant.py"
        wrapper.write_text(
            "\n".join(
                (
                    "#!/usr/bin/python3",
                    "import os",
                    "import pathlib",
                    "import signal",
                    "import sys",
                    "import time",
                    f"child_path = pathlib.Path({str(child_path)!r})",
                    f"process_path = pathlib.Path({str(process_path)!r})",
                    "child = os.fork()",
                    "if child == 0:",
                    f"    if {alternate_process_group!r}:",
                    "        os.setpgid(0, 0)",
                    "    signal.signal(signal.SIGTERM, signal.SIG_IGN)",
                    "    child_path.write_text(str(os.getpid()), encoding='ascii')",
                    "    while True:",
                    "        time.sleep(1)",
                    "deadline = time.monotonic() + 1",
                    "while not child_path.exists() and time.monotonic() < deadline:",
                    "    time.sleep(0.001)",
                    "if not child_path.exists():",
                    "    raise SystemExit(1)",
                    "process_path.write_text(",
                    "    f'{os.getpid()} {child} {os.getpgrp()} {os.getpgid(child)} {os.getsid(0)}',",
                    "    encoding='ascii',",
                    ")",
                    f"sys.stdout.write({COMPOSER.APPENDER_CONTRACT_PROBE_OUTPUT.decode('ascii')!r})",
                    "sys.stdout.flush()",
                    "os._exit(0)",
                    "",
                )
            ),
            encoding="utf-8",
        )
        wrapper.chmod(0o700)
        return wrapper, process_path

    def assert_process_gone(self, pid: int) -> None:
        deadline = time.monotonic() + 1
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            if time.monotonic() >= deadline:
                self.fail(f"process {pid} remained after contract probe cleanup")
            time.sleep(0.01)

    def assert_probe_session_quiescent(self, session: int) -> None:
        deadline = time.monotonic() + 1
        while True:
            members = COMPOSER._scan_probe_session(session, deadline)
            if not members:
                return
            if time.monotonic() >= deadline:
                self.fail(
                    f"probe session {session} retained members after containment: {members}"
                )
            time.sleep(0.01)

    def read_contract_probe_pid(self, pid_path: pathlib.Path) -> int:
        deadline = time.monotonic() + 1
        while True:
            if pid_path.exists():
                raw = pid_path.read_text(encoding="ascii")
                if raw:
                    return int(raw)
            if time.monotonic() >= deadline:
                self.fail(f"contract probe PID file was not populated: {pid_path}")
            time.sleep(0.001)

    def assert_negative_terminal_archive(
        self,
        *,
        label: str,
        writer_status: int,
        clock_status: int,
        expected_outcome: str,
        expected_terminal: tuple[int, int, int],
    ) -> None:
        archive, evidence_root, prefix, handoff_raw = self.produce(label)
        negative_handoff = handoff_with_writer_outcome(
            handoff_raw,
            writer_status=writer_status,
            clock_status=clock_status,
        )
        handoff_path = archive / COMPOSER.PROCESS_HANDOFF_ARCHIVE_PATH
        handoff_path.write_bytes(negative_handoff)

        result = self.compose(archive, evidence_root)
        self.assertEqual(result.terminal_outcome, expected_outcome)
        self.assertFalse(result.scientific_admission_complete)
        self.assertTrue(result.admission_blockers)
        self.assertEqual(
            result.admission_blockers[0],
            "negative terminal outcome is retained and never promoted",
        )
        self.assertEqual(result.handoff_sha256, sha(negative_handoff))

        stream_path = archive / COMPOSER.STREAM_ARCHIVE_PATH
        stream_raw = stream_path.read_bytes()
        self.assertGreater(len(stream_raw), len(prefix))
        decoded = WIRE.decode_container(stream_raw)
        header = decoded.trailer_body.header
        observed_terminal = (
            header.lifecycle_state,
            header.terminal_reason_code,
            header.payload_writer_state,
        )
        self.assertEqual(observed_terminal, expected_terminal)

        status_raw = (
            archive / COMPOSER.VERIFIER.STATUS.PRE_FOOTER_PATH
        ).read_bytes()
        status = COMPOSER.VERIFIER.STATUS.parse_canonical(status_raw)
        COMPOSER.VERIFIER.STATUS.validate_pre_footer(status)
        self.assertEqual(
            (
                status["lifecycle_state"],
                status["reason_code"],
                status["payload_writer_state"],
            ),
            expected_terminal,
        )

        receipt_path = archive / COMPOSER.VERIFIER.STATUS.POST_CLOSE_PATH
        receipt_raw = receipt_path.read_bytes()
        receipt = COMPOSER.VERIFIER.STATUS.parse_canonical(receipt_raw)
        COMPOSER.VERIFIER.STATUS.validate_post_close(receipt)
        self.assertEqual(receipt["physical_bytes"], len(stream_raw))
        self.assertEqual(receipt["whole_stream_sha256"], sha(stream_raw))

        manifest_raw = (
            archive / COMPOSER.VERIFIER.STATUS.MANIFEST_PATH
        ).read_bytes()
        manifest = COMPOSER.VERIFIER.STATUS.parse_canonical(manifest_raw)
        entries = COMPOSER.VERIFIER.STATUS.validate_manifest(manifest)
        by_path = {
            row.path: (row.byte_count, row.sha256) for row in entries
        }
        expected_bindings = {
            COMPOSER.STREAM_ARCHIVE_PATH: stream_raw,
            COMPOSER.PROCESS_HANDOFF_ARCHIVE_PATH: negative_handoff,
            COMPOSER.VERIFIER.STATUS.PRE_FOOTER_PATH: status_raw,
            COMPOSER.VERIFIER.STATUS.POST_CLOSE_PATH: receipt_raw,
        }
        for path, raw in expected_bindings.items():
            self.assertEqual(by_path[path], (len(raw), sha(raw)))
        self.assertNotIn(COMPOSER.VERIFIER.STATUS.MANIFEST_PATH, by_path)
        self.assertEqual(result.stream_sha256, sha(stream_raw))
        self.assertEqual(result.manifest_sha256, sha(manifest_raw))

    def test_prepared_root_publishes_exact_build_coverage(self) -> None:
        published = self.evidence_root.joinpath(
            *COVERAGE.BUILD_COVERAGE_ARCHIVE_PATH.split("/")
        ).read_bytes()
        self.assertEqual(published, self.prepared.coverage_bytes)
        self.assertEqual(
            sha(published),
            self.evidence_value["build_coverage_sha256"],
        )
        COVERAGE.validate_build_coverage_bytes(
            published, api_definition_sha256=API_DEFINITION_SHA256
        )

    def test_unregistered_elf_in_primary_build_directory_rejects(self) -> None:
        _archive, _evidence_root, _prefix, handoff_raw = self.produce(
            "unregistered_mapped_elf"
        )
        extra_elf = self.producer.parent / "unregistered-oai-module.so"
        shutil.copy2("/bin/true", extra_elf)
        identity = extra_elf.stat()
        mapped_line = (
            f"700000000000-700000001000 r-xp 00000000 "
            f"{os.major(identity.st_dev):x}:{os.minor(identity.st_dev):x} "
            f"{identity.st_ino} {extra_elf}\n"
        ).encode("utf-8")
        handoff = HANDOFF.decode_process_handoff(handoff_raw)
        separator = b"" if handoff.maps_bytes.endswith(b"\n") else b"\n"
        mutated = dataclasses.replace(
            handoff,
            maps_bytes=handoff.maps_bytes + separator + mapped_line,
        )
        with self.assertRaisesRegex(
            COMPOSER.ArchiveComposerError,
            "unregistered mapped ELF under primary build directory",
        ):
            COMPOSER._runtime_objects(mutated, self.build)

    def test_final_path_rebind_compares_complete_file_identity(self) -> None:
        path = self.case_root / "complete-frozen-identity.bin"
        path.write_bytes(b"frozen-input")
        initial = path.stat()
        fields = {
            "st_mode": initial.st_mode ^ 0o200,
            "st_nlink": initial.st_nlink + 1,
            "st_mtime_ns": initial.st_mtime_ns + 1,
            "st_ctime_ns": initial.st_ctime_ns + 1,
        }
        for field, value in fields.items():
            with self.subTest(field=field):
                rebound = mock.Mock()
                rebound.st_dev = initial.st_dev
                rebound.st_ino = initial.st_ino
                rebound.st_mode = initial.st_mode
                rebound.st_nlink = initial.st_nlink
                rebound.st_size = initial.st_size
                rebound.st_mtime_ns = initial.st_mtime_ns
                rebound.st_ctime_ns = initial.st_ctime_ns
                setattr(rebound, field, value)
                with mock.patch.object(COMPOSER.os, "lstat", return_value=rebound):
                    with self.assertRaisesRegex(
                        COMPOSER.ArchiveComposerError,
                        "path changed after read",
                    ):
                        COMPOSER._read_frozen(path, initial.st_size)

    def test_mapped_primary_inode_mismatch_at_retained_path_rejects(self) -> None:
        _archive, _evidence_root, _prefix, handoff_raw = self.produce(
            "mapped_primary_inode_mismatch"
        )
        handoff = HANDOFF.decode_process_handoff(handoff_raw)
        mutated = dataclasses.replace(
            handoff,
            maps_bytes=replace_maps_file_identity(
                handoff.maps_bytes,
                self.producer,
                replacement_inode=self.producer.stat().st_ino + 1,
            ),
        )
        with self.assertRaisesRegex(
            COMPOSER.ArchiveComposerError,
            "primary logical ELF must have exactly one runtime mapping",
        ):
            COMPOSER._runtime_objects(mutated, self.build)

    def test_alternate_glibc_mapping_rejects_before_append(self) -> None:
        archive, _evidence_root, prefix, handoff_raw = self.produce(
            "alternate_glibc_mapping"
        )
        handoff = HANDOFF.decode_process_handoff(handoff_raw)
        candidates = COMPOSER._glibc_runtime_map_candidates(handoff)
        self.assertEqual(len(candidates), 1)
        actual_path = pathlib.Path(candidates[0]["loaded_path"])
        alternate_directory = self.case_root / "alternate-glibc"
        alternate_directory.mkdir()
        alternate = alternate_directory / "libc-999.0.so"
        shutil.copy2("/bin/true", alternate)
        alternate_identity = alternate.stat()
        mutated = dataclasses.replace(
            handoff,
            maps_bytes=replace_maps_file_identity(
                handoff.maps_bytes,
                actual_path,
                replacement_path=alternate,
                replacement_device=alternate_identity.st_dev,
                replacement_inode=alternate_identity.st_ino,
            ),
        )
        with self.assertRaisesRegex(
            COMPOSER.ArchiveComposerError,
            "authenticated glibc runtime mapped bytes differ from build evidence",
        ):
            COMPOSER._require_glibc_runtime_binding(
                mutated,
                self.build,
                self.evidence_value,
                self.artifacts,
            )
        self.assertEqual(
            (archive / "streams/memory-lifetime.bin").read_bytes(), prefix
        )

    def test_current_appender_contract_probe_and_integration_succeeds(self) -> None:
        raw = self.appender.read_bytes()
        self.assertEqual(struct.unpack_from("<H", raw, 16)[0], 2)
        program_offset = struct.unpack_from("<Q", raw, 32)[0]
        program_entry_size = struct.unpack_from("<H", raw, 54)[0]
        program_count = struct.unpack_from("<H", raw, 56)[0]
        program_types = {
            struct.unpack_from("<I", raw, program_offset + index * program_entry_size)[0]
            for index in range(program_count)
        }
        self.assertNotIn(2, program_types)  # PT_DYNAMIC
        self.assertNotIn(3, program_types)  # PT_INTERP
        _machine, sections, _bodies = BUILD_EVIDENCE._parse_sections(raw)
        self.assertFalse(any(section.kind in (6, 11) for section in sections))
        auxiliary_paths = BUILD_EVIDENCE._auxiliary_paths(
            BUILD_EVIDENCE.ARCHIVE_APPEND_AUXILIARY_ID
        )
        command_raw = self.artifacts[auxiliary_paths["link_command_path"]]
        self.assertEqual(len(re.findall(rb"(?<!\S)-static(?!\S)", command_raw)), 1)
        self.assertEqual(len(re.findall(rb"(?<!\S)-no-pie(?!\S)", command_raw)), 1)
        self.assertIn(b"-Wl,-Map,", command_raw)
        self.assertIn(b"-Wl,--cref", command_raw)
        self.assertNotIn(b"liboai_memprof_active_runtime", command_raw)
        self.assertNotIn(b"rpath", command_raw.lower())
        appender = COMPOSER._require_authenticated_appender(
            self.appender,
            self.evidence_value,
            self.artifacts,
        )
        descriptor = appender.pass_descriptor()
        executable = appender.executable_path()
        self.assertFalse(os.get_inheritable(descriptor))
        required_seals = (
            COMPOSER.fcntl.F_SEAL_WRITE
            | COMPOSER.fcntl.F_SEAL_GROW
            | COMPOSER.fcntl.F_SEAL_SHRINK
            | COMPOSER.fcntl.F_SEAL_SEAL
        )
        self.assertEqual(appender.seals & required_seals, required_seals)
        with self.assertRaises(PermissionError):
            os.write(descriptor, b"x")
        original_raw = self.appender.read_bytes()
        original_mode = self.appender.stat().st_mode & 0o7777
        replacement = self.contract_probe_executable(
            "sealed-appender-path-replacement",
            stdout_text="evil\n",
            stderr_text="",
            return_code=0,
        )
        try:
            os.replace(replacement, self.appender)
            self.appender.write_bytes(b"not an executable image\n")
            with mock.patch.object(
                COMPOSER.subprocess, "Popen", wraps=subprocess.Popen
            ) as runner:
                COMPOSER._require_appender_contract(appender)
        finally:
            self.appender.write_bytes(original_raw)
            self.appender.chmod(original_mode)
        runner.assert_called_once_with(
            [COMPOSER.APPENDER_CALLER_ARGV0, COMPOSER.APPENDER_CONTRACT_PROBE_ARGUMENT],
            executable=executable,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            start_new_session=True,
            close_fds=True,
            pass_fds=(descriptor,),
            env={},
        )
        completed = subprocess.CompletedProcess([], 0, b"", b"")
        with mock.patch.object(
            COMPOSER.subprocess, "run", return_value=completed
        ) as production:
            self.assertIs(
                COMPOSER._run_sealed_appender(
                    appender,
                    self.case_root,
                    "stream.bin",
                    "handoff.bin",
                    "trailer.bin",
                    "a" * 64,
                ),
                completed,
            )
        production.assert_called_once_with(
            [
                COMPOSER.APPENDER_CALLER_ARGV0,
                str(self.case_root),
                "stream.bin",
                "handoff.bin",
                "trailer.bin",
                "a" * 64,
            ],
            executable=executable,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            close_fds=True,
            pass_fds=(descriptor,),
            env={},
        )
        appender.close()
        with self.assertRaisesRegex(COMPOSER.ArchiveComposerError, "used after close"):
            appender.executable_path()
        with self.assertRaisesRegex(COMPOSER.ArchiveComposerError, "closed more than once"):
            appender.close()

    def test_appender_authentication_and_seal_fail_closed(self) -> None:
        alias = self.case_root / "archive-appender-alias"
        shutil.copy2(self.appender, alias)
        with self.assertRaisesRegex(
            COMPOSER.ArchiveComposerError,
            "exact canonical authenticated build output",
        ):
            COMPOSER._require_authenticated_appender(
                alias, self.evidence_value, self.artifacts
            )
        original_mode = self.appender.stat().st_mode & 0o7777
        original_raw = self.appender.read_bytes()
        try:
            self.appender.chmod(original_mode & ~0o111)
            with self.assertRaisesRegex(
                COMPOSER.ArchiveComposerError, "requires an executable mode bit"
            ):
                COMPOSER._require_authenticated_appender(
                    self.appender, self.evidence_value, self.artifacts
                )
            self.appender.chmod(original_mode)
            self.appender.write_bytes(original_raw + b"\n")
            with self.assertRaisesRegex(
                COMPOSER.ArchiveComposerError, "bytes differ from build evidence"
            ):
                COMPOSER._require_authenticated_appender(
                    self.appender, self.evidence_value, self.artifacts
                )
        finally:
            self.appender.write_bytes(original_raw)
            self.appender.chmod(original_mode)
        with mock.patch.object(COMPOSER.sys, "platform", "not-linux"):
            with self.assertRaisesRegex(
                COMPOSER.ArchiveComposerError, "sealed archive appender unavailable"
            ):
                COMPOSER._create_sealed_appender(original_raw)
        with mock.patch.object(COMPOSER.os, "write", return_value=0), mock.patch.object(
            COMPOSER.os, "close", wraps=os.close
        ) as closer:
            with self.assertRaisesRegex(
                COMPOSER.ArchiveComposerError, "short write"
            ):
                COMPOSER._create_sealed_appender(original_raw)
        self.assertEqual(closer.call_count, 1)

        real_close = os.close

        def close_then_ordinary_error(descriptor: int) -> None:
            real_close(descriptor)
            raise OSError("synthetic setup close failure")

        with mock.patch.object(COMPOSER.os, "write", return_value=0), mock.patch.object(
            COMPOSER.os, "close", side_effect=close_then_ordinary_error
        ) as closer:
            with self.assertRaisesRegex(
                COMPOSER.ArchiveComposerError, "short write"
            ) as captured:
                COMPOSER._create_sealed_appender(original_raw)
        self.assertEqual(closer.call_count, 1)
        self.assertIn(
            "sealed archive appender setup close failure: OSError",
            "\n".join(captured.exception.__notes__),
        )

        close_interruption = KeyboardInterrupt()

        def close_then_interrupt(descriptor: int) -> None:
            real_close(descriptor)
            raise close_interruption

        with mock.patch.object(COMPOSER.os, "write", return_value=0), mock.patch.object(
            COMPOSER.os, "close", side_effect=close_then_interrupt
        ) as closer:
            with self.assertRaisesRegex(
                COMPOSER.ArchiveComposerError, "short write"
            ) as captured:
                COMPOSER._create_sealed_appender(original_raw)
        self.assertEqual(closer.call_count, 1)
        self.assertIn(
            "sealed archive appender setup close interruption: KeyboardInterrupt",
            "\n".join(captured.exception.__notes__),
        )

        for interruption in (KeyboardInterrupt(), SystemExit(17)):
            with self.subTest(interruption=type(interruption).__name__), mock.patch.object(
                COMPOSER.os, "fchmod", side_effect=interruption
            ), mock.patch.object(COMPOSER.os, "close", wraps=os.close) as closer:
                with self.assertRaises(type(interruption)) as captured:
                    COMPOSER._create_sealed_appender(original_raw)
            self.assertIs(captured.exception, interruption)
            self.assertEqual(closer.call_count, 1)

    def test_sealed_appender_final_close_precedence(self) -> None:
        primary = COMPOSER.ArchiveComposerError("synthetic mutation primary")
        ordinary_close = COMPOSER.ArchiveComposerError("synthetic ordinary close")
        appender = mock.Mock()
        appender.close.side_effect = ordinary_close
        COMPOSER._close_sealed_appender(appender, primary)
        self.assertIn(
            "sealed archive appender close failure: ArchiveComposerError",
            "\n".join(primary.__notes__),
        )

        appender = mock.Mock()
        appender.close.side_effect = ordinary_close
        with self.assertRaises(COMPOSER.ArchiveComposerError) as captured:
            COMPOSER._close_sealed_appender(appender, None)
        self.assertIs(captured.exception, ordinary_close)

        interruption = KeyboardInterrupt()
        appender = mock.Mock()
        appender.close.side_effect = interruption
        COMPOSER._close_sealed_appender(appender, primary)
        self.assertIn(
            "sealed archive appender close interruption: KeyboardInterrupt",
            "\n".join(primary.__notes__),
        )

        appender = mock.Mock()
        appender.close.side_effect = interruption
        with self.assertRaises(KeyboardInterrupt) as captured:
            COMPOSER._close_sealed_appender(appender, None)
        self.assertIs(captured.exception, interruption)

    def test_stale_or_wrong_appender_contract_rejects_before_stream_mutation(
        self,
    ) -> None:
        cases = (
            (
                "stale_appender_contract",
                "",
                "usage: oai_memprof_archive_append DIRECTORY STREAM_LEAF HANDOFF_LEAF TRAILER_LEAF\n",
                64,
                "append utility contract probe failed status=64",
            ),
            (
                "wrong_appender_contract",
                "oai_memprof_archive_append contract-v0\n",
                "",
                0,
                "append utility contract probe stdout mismatch",
            ),
        )
        for label, stdout_text, stderr_text, return_code, error in cases:
            with self.subTest(label=label):
                wrapper = self.contract_probe_executable(
                    label,
                    stdout_text=stdout_text,
                    stderr_text=stderr_text,
                    return_code=return_code,
                )
                appender = self.sealed_appender(wrapper)
                try:
                    with self.assertRaisesRegex(COMPOSER.ArchiveComposerError, error):
                        COMPOSER._require_appender_contract(appender)
                finally:
                    appender.close()

        archive, evidence_root, prefix, handoff = self.produce(
            "unmeasured_wrapper_prepublication_rejection"
        )
        wrapper = self.contract_probe_executable(
            "unmeasured_wrapper_prepublication_rejection",
            stdout_text=COMPOSER.APPENDER_CONTRACT_PROBE_OUTPUT.decode("ascii"),
            stderr_text="",
            return_code=0,
        )
        with self.assertRaisesRegex(
            COMPOSER.ArchiveComposerError,
            "exact canonical authenticated build output",
        ):
            self.compose(archive, evidence_root, append_executable=wrapper)
        self.assertEqual((archive / COMPOSER.STREAM_ARCHIVE_PATH).read_bytes(), prefix)
        self.assertEqual(
            (archive / COMPOSER.PROCESS_HANDOFF_ARCHIVE_PATH).read_bytes(), handoff
        )
        self.assertFalse((archive / "streams/trailer.bin").exists())
        self.assertFalse((archive / COMPOSER.VERIFIER.STATUS.MANIFEST_PATH).exists())

    def test_contract_probe_rejects_stderr_timeout_and_output_overflow(self) -> None:
        stderr_appender = self.contract_probe_executable(
            "stderr_appender_contract",
            stdout_text=COMPOSER.APPENDER_CONTRACT_PROBE_OUTPUT.decode("ascii"),
            stderr_text="unexpected probe diagnostic\n",
            return_code=0,
        )
        with self.assertRaisesRegex(
            COMPOSER.ArchiveComposerError,
            "append utility contract probe emitted stderr",
        ):
            self.probe_contract_path(stderr_appender)

        sleeper, pid_path = self.sleeping_contract_probe_executable(
            "timeout_appender_contract"
        )
        with mock.patch.object(
            COMPOSER.subprocess,
            "Popen",
            wraps=subprocess.Popen,
        ) as runner, mock.patch.object(
            COMPOSER, "APPENDER_CONTRACT_PROBE_TIMEOUT_SECONDS", 0.2
        ):
            with self.assertRaisesRegex(
                COMPOSER.ArchiveComposerError,
                "append utility contract probe timed out",
            ):
                self.probe_contract_path(sleeper)
        self.assertTrue(pid_path.exists())
        direct_pid = self.read_contract_probe_pid(pid_path)
        self.assert_process_gone(direct_pid)
        self.assertEqual(runner.call_count, 1)

        for stream, error in (
            ("stdout", "append utility contract probe stdout exceeded byte limit"),
            ("stderr", "append utility contract probe stderr exceeded byte limit"),
        ):
            with self.subTest(stream=stream):
                appender = self.contract_probe_executable(
                    f"{stream}_overflow_appender_contract",
                    stdout_text=(
                        "x" * (COMPOSER.APPENDER_CONTRACT_PROBE_STDOUT_LIMIT_BYTES + 1)
                        if stream == "stdout"
                        else ""
                    ),
                    stderr_text=(
                        "x" * (COMPOSER.APPENDER_CONTRACT_PROBE_STDERR_LIMIT_BYTES + 1)
                        if stream == "stderr"
                        else ""
                    ),
                    return_code=0,
                )
                with self.assertRaisesRegex(COMPOSER.ArchiveComposerError, error):
                    self.probe_contract_path(appender)

    def test_contract_probe_rejects_same_session_descendant_and_reaps_it(self) -> None:
        appender, process_path = self.descendant_contract_probe_executable(
            "descendant_appender_contract"
        )
        with self.assertRaisesRegex(
            COMPOSER.ArchiveComposerError,
            "append utility contract probe left session descendants",
        ):
            self.probe_contract_path(appender)
        self.assertTrue(process_path.exists())
        leader_pid, child_pid, leader_group, child_group, session = map(
            int, process_path.read_text(encoding="ascii").split()
        )
        self.assertEqual(leader_pid, leader_group)
        self.assertEqual(child_group, leader_group)
        self.assert_process_gone(leader_pid)
        self.assert_process_gone(child_pid)
        self.assert_probe_session_quiescent(session)

    def test_contract_probe_rejects_alternate_group_same_session_descendant(self) -> None:
        appender, process_path = self.descendant_contract_probe_executable(
            "alternate_group_descendant_appender_contract",
            alternate_process_group=True,
        )
        with self.assertRaisesRegex(
            COMPOSER.ArchiveComposerError,
            "append utility contract probe left session descendants",
        ):
            self.probe_contract_path(appender)
        self.assertTrue(process_path.exists())
        leader_pid, child_pid, leader_group, child_group, session = map(
            int, process_path.read_text(encoding="ascii").split()
        )
        self.assertEqual(leader_pid, leader_group)
        self.assertEqual(child_pid, child_group)
        self.assertNotEqual(child_group, leader_group)
        self.assert_process_gone(leader_pid)
        self.assert_process_gone(child_pid)
        self.assert_probe_session_quiescent(session)

    def test_contract_probe_cleanup_timeout_when_session_persists(self) -> None:
        probe = mock.Mock()
        leader = COMPOSER._ProbeProcessIdentity(
            pid=12345,
            process_group=12345,
            session=12345,
            start_time_ticks=1,
            state="S",
        )
        persistent_child = COMPOSER._ProbeProcessIdentity(
            pid=12346,
            process_group=12346,
            session=12345,
            start_time_ticks=2,
            state="S",
        )
        with mock.patch.object(
            COMPOSER, "_probe_leader_exited", return_value=True
        ), mock.patch.object(
            COMPOSER, "_probe_session_members", return_value=[leader, persistent_child]
        ), mock.patch.object(
            COMPOSER, "_signal_probe_identity"
        ) as signal_identity, mock.patch.object(
            COMPOSER, "APPENDER_CONTRACT_PROBE_CLEANUP_GRACE_SECONDS", 0.001
        ):
            with self.assertRaisesRegex(
                COMPOSER.ArchiveComposerError,
                "append utility contract probe cleanup timed out",
            ):
                cleanup_failure = COMPOSER._terminate_probe_session(
                    probe, 99, leader
                )
                if cleanup_failure is not None:
                    raise cleanup_failure
        self.assertTrue(
            all(
                call.args[0] == persistent_child
                for call in signal_identity.call_args_list
            )
        )
        self.assertIn(
            mock.call(persistent_child, COMPOSER.signal.SIGTERM),
            signal_identity.call_args_list,
        )
        self.assertIn(
            mock.call(persistent_child, COMPOSER.signal.SIGKILL),
            signal_identity.call_args_list,
        )

    def test_contract_probe_does_not_signal_recycled_group_after_setsid_escape(
        self,
    ) -> None:
        leader = COMPOSER._ProbeProcessIdentity(
            pid=12345,
            process_group=12345,
            session=12345,
            start_time_ticks=1,
            state="Z",
        )
        with mock.patch.object(
            COMPOSER, "_probe_leader_exited", return_value=True
        ), mock.patch.object(
            COMPOSER, "_probe_session_members", return_value=[leader]
        ), mock.patch.object(
            COMPOSER, "_signal_probe_identity"
        ) as signal_identity, mock.patch.object(COMPOSER.os, "killpg") as killpg:
            self.assertTrue(
                COMPOSER._wait_for_probe_session_quiescence(
                    99,
                    leader,
                    time.monotonic() + 1,
                    COMPOSER.signal.SIGKILL,
                )
            )
        signal_identity.assert_not_called()
        killpg.assert_not_called()

    def test_contract_probe_adapts_setup_and_close_failures(self) -> None:
        sleeper, pid_path = self.sleeping_contract_probe_executable(
            "setup_failure_appender_contract"
        )
        real_popen = subprocess.Popen
        launched = []

        def launch(*arguments, **kwargs):
            probe = real_popen(*arguments, **kwargs)
            launched.append(probe)
            return probe

        def fail_set_blocking(*_args, **_kwargs):
            deadline = time.monotonic() + 1
            while not pid_path.exists() and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertTrue(pid_path.exists())
            raise OSError("synthetic set-blocking failure")

        with mock.patch.object(
            COMPOSER.subprocess, "Popen", side_effect=launch
        ), mock.patch.object(
            COMPOSER.os, "set_blocking", side_effect=fail_set_blocking
        ):
            with self.assertRaisesRegex(
                COMPOSER.ArchiveComposerError,
                "append utility contract probe I/O setup failed",
            ):
                self.probe_contract_path(sleeper)
        self.assertEqual(len(launched), 1)
        self.assert_process_gone(self.read_contract_probe_pid(pid_path))

        sleeper, pid_path = self.sleeping_contract_probe_executable(
            "missing_pipe_appender_contract"
        )
        launched_without_stdout = []

        def launch_without_stdout(*arguments, **kwargs):
            probe = real_popen(*arguments, **kwargs)
            launched_without_stdout.append(probe)
            stdout = probe.stdout
            probe.stdout = None
            if stdout is not None:
                stdout.close()
            return probe

        with mock.patch.object(
            COMPOSER.subprocess, "Popen", side_effect=launch_without_stdout
        ):
            with self.assertRaisesRegex(
                COMPOSER.ArchiveComposerError,
                "append utility contract probe pipes unavailable",
            ) as captured:
                self.probe_contract_path(sleeper)
        self.assertIsNot(captured.exception.__cause__, captured.exception)
        self.assertEqual(len(launched_without_stdout), 1)
        self.assert_process_gone(launched_without_stdout[0].pid)

        sleeper, pid_path = self.sleeping_contract_probe_executable(
            "close_failure_appender_contract"
        )
        original_close = COMPOSER._close_probe_resource

        def close_with_failure(name, resource):
            original_failure = original_close(name, resource)
            if name == "selector":
                return COMPOSER.ArchiveComposerError("synthetic selector close failure")
            return original_failure

        with mock.patch.object(
            COMPOSER, "_close_probe_resource", side_effect=close_with_failure
        ) as closer, mock.patch.object(
            COMPOSER, "APPENDER_CONTRACT_PROBE_TIMEOUT_SECONDS", 0.2
        ):
            with self.assertRaisesRegex(
                COMPOSER.ArchiveComposerError,
                "append utility contract probe timed out",
            ) as captured:
                self.probe_contract_path(sleeper)
        self.assertIn(
            "cleanup/close failure: synthetic selector close failure",
            "\n".join(captured.exception.__notes__),
        )
        self.assertEqual(
            [call.args[0] for call in closer.call_args_list],
            ["selector", "stdout", "stderr"],
        )
        self.assert_process_gone(self.read_contract_probe_pid(pid_path))

    def test_contract_probe_unidentified_fallback_reaps_session_descendants(
        self,
    ) -> None:
        original_read = COMPOSER._read_probe_process_identity
        for alternate_process_group in (False, True):
            with self.subTest(alternate_process_group=alternate_process_group):
                appender, process_path = self.descendant_contract_probe_executable(
                    f"fallback_descendant_{int(alternate_process_group)}",
                    alternate_process_group=alternate_process_group,
                )
                failed = False

                def fail_first_leader_read(pid: int):
                    nonlocal failed
                    if not failed:
                        failed = True
                        deadline = time.monotonic() + 1
                        fields = ()
                        while time.monotonic() < deadline:
                            if process_path.exists():
                                fields = tuple(
                                    process_path.read_text(encoding="ascii").split()
                                )
                                if len(fields) == 5:
                                    break
                            time.sleep(0.001)
                        self.assertEqual(len(fields), 5)
                        raise COMPOSER.ArchiveComposerError(
                            "synthetic leader setup failure"
                        )
                    return original_read(pid)

                with mock.patch.object(
                    COMPOSER,
                    "_read_probe_process_identity",
                    side_effect=fail_first_leader_read,
                ):
                    with self.assertRaisesRegex(
                        COMPOSER.ArchiveComposerError,
                        "synthetic leader setup failure",
                    ):
                        self.probe_contract_path(appender)
                leader_pid, child_pid, _leader_group, child_group, session = map(
                    int, process_path.read_text(encoding="ascii").split()
                )
                if alternate_process_group:
                    self.assertEqual(child_pid, child_group)
                self.assert_process_gone(leader_pid)
                self.assert_process_gone(child_pid)
                self.assert_probe_session_quiescent(session)

    def test_contract_probe_pidfd_unavailable_fails_closed_before_reap(self) -> None:
        sleeper, _pid_path = self.sleeping_contract_probe_executable(
            "pidfd_unavailable_appender_contract"
        )
        real_popen = subprocess.Popen
        real_killpg = os.killpg
        launched = []
        events = []

        def launch(*arguments, **kwargs):
            probe = real_popen(*arguments, **kwargs)
            launched.append(probe)
            original_wait = probe.wait

            def record_wait(*wait_arguments, **wait_kwargs):
                events.append("wait")
                return original_wait(*wait_arguments, **wait_kwargs)

            probe.wait = record_wait
            return probe

        def record_group_signal(process_group, signal_number):
            events.append(("killpg", process_group, signal_number))
            return real_killpg(process_group, signal_number)

        with mock.patch.object(
            COMPOSER.subprocess, "Popen", side_effect=launch
        ), mock.patch.object(
            COMPOSER.os, "pidfd_open", side_effect=OSError("unavailable")
        ), mock.patch.object(
            COMPOSER.os, "killpg", side_effect=record_group_signal
        ), mock.patch.object(
            COMPOSER, "APPENDER_CONTRACT_PROBE_CLEANUP_GRACE_SECONDS", 0.05
        ):
            with self.assertRaisesRegex(
                COMPOSER.ArchiveComposerError,
                "append utility contract probe pidfd unavailable",
            ):
                self.probe_contract_path(sleeper)
        self.assertEqual(len(launched), 1)
        self.assert_process_gone(launched[0].pid)
        wait_index = events.index("wait")
        self.assertTrue(
            all(index < wait_index for index, event in enumerate(events) if event != "wait")
        )
        self.assertIn(
            ("killpg", launched[0].pid, COMPOSER.signal.SIGTERM), events
        )
        self.assertIn(
            ("killpg", launched[0].pid, COMPOSER.signal.SIGKILL), events
        )

    def test_contract_probe_fences_invalid_leader_group_and_session(self) -> None:
        original_read = COMPOSER._read_probe_process_identity
        original_scan = COMPOSER._scan_probe_session
        original_signal = COMPOSER._signal_probe_identity
        real_popen = subprocess.Popen
        for field in ("process_group", "session"):
            with self.subTest(field=field):
                sleeper, _pid_path = self.sleeping_contract_probe_executable(
                    f"invalid_leader_{field}_appender_contract"
                )
                launched = []
                scan_sessions = []
                signalled = []
                injected = None

                def launch(*arguments, **kwargs):
                    probe = real_popen(*arguments, **kwargs)
                    launched.append(probe)
                    return probe

                def invalid_first_leader_read(pid: int):
                    nonlocal injected
                    identity = original_read(pid)
                    if injected is None:
                        self.assertIsNotNone(identity)
                        injected = dataclasses.replace(
                            identity,
                            **{field: identity.pid + 100_000},
                        )
                        return injected
                    return identity

                def record_scan(session, deadline):
                    scan_sessions.append(session)
                    return original_scan(session, deadline)

                def record_signal(identity, signal_number):
                    signalled.append(identity)
                    return original_signal(identity, signal_number)

                with mock.patch.object(
                    COMPOSER.subprocess, "Popen", side_effect=launch
                ), mock.patch.object(
                    COMPOSER,
                    "_read_probe_process_identity",
                    side_effect=invalid_first_leader_read,
                ), mock.patch.object(
                    COMPOSER, "_scan_probe_session", side_effect=record_scan
                ), mock.patch.object(
                    COMPOSER, "_signal_probe_identity", side_effect=record_signal
                ), mock.patch.object(
                    COMPOSER, "APPENDER_CONTRACT_PROBE_CLEANUP_GRACE_SECONDS", 0.05
                ):
                    with self.assertRaisesRegex(
                        COMPOSER.ArchiveComposerError,
                        "append utility contract probe leader identity unavailable",
                    ):
                        self.probe_contract_path(sleeper)
                self.assertEqual(len(launched), 1)
                self.assertTrue(scan_sessions)
                self.assertEqual(scan_sessions, [launched[0].pid] * len(scan_sessions))
                self.assertNotIn(injected, signalled)
                self.assert_process_gone(launched[0].pid)

    def test_contract_probe_never_reads_past_stream_limit_after_overflow(self) -> None:
        appender = self.contract_probe_executable(
            "large_stdout_appender_contract",
            stdout_text="x" * (COMPOSER.APPENDER_CONTRACT_PROBE_STDOUT_LIMIT_BYTES * 32),
            stderr_text="",
            return_code=0,
        )
        real_popen = subprocess.Popen
        real_read = os.read
        stream_fds = set()
        stream_bytes = {}

        def launch(*arguments, **kwargs):
            probe = real_popen(*arguments, **kwargs)
            for pipe in (probe.stdout, probe.stderr):
                if pipe is not None:
                    stream_fds.add(pipe.fileno())
                    stream_bytes[pipe.fileno()] = 0
            return probe

        def count_stream_read(descriptor, size):
            chunk = real_read(descriptor, size)
            if descriptor in stream_fds:
                stream_bytes[descriptor] += len(chunk)
            return chunk

        with mock.patch.object(
            COMPOSER.subprocess, "Popen", side_effect=launch
        ), mock.patch.object(COMPOSER.os, "read", side_effect=count_stream_read):
            with self.assertRaisesRegex(
                COMPOSER.ArchiveComposerError,
                "append utility contract probe stdout exceeded byte limit",
            ):
                self.probe_contract_path(appender)
        self.assertTrue(stream_bytes)
        self.assertTrue(
            all(
                count <= COMPOSER.APPENDER_CONTRACT_PROBE_STDOUT_LIMIT_BYTES + 1
                for count in stream_bytes.values()
            )
        )

    def test_contract_probe_cleanup_interrupt_reaps_before_propagation(self) -> None:
        appender, process_path = self.descendant_contract_probe_executable(
            "cleanup_interrupt_appender_contract"
        )
        interruption = KeyboardInterrupt()
        original_wait = COMPOSER._wait_for_probe_session_quiescence
        original_close = COMPOSER._close_probe_resource
        close_names = []

        def interrupt_term_wait(pidfd, leader, deadline, signal_number):
            if signal_number == COMPOSER.signal.SIGTERM:
                raise interruption
            return original_wait(pidfd, leader, deadline, signal_number)

        def record_close(name, resource):
            close_names.append(name)
            return original_close(name, resource)

        with mock.patch.object(
            COMPOSER,
            "_wait_for_probe_session_quiescence",
            side_effect=interrupt_term_wait,
        ), mock.patch.object(
            COMPOSER, "_close_probe_resource", side_effect=record_close
        ):
            with self.assertRaises(KeyboardInterrupt) as captured:
                self.probe_contract_path(appender)
        self.assertIs(captured.exception, interruption)
        self.assertEqual(close_names, ["selector", "stdout", "stderr"])
        leader_pid, child_pid, _leader_group, _child_group, session = map(
            int, process_path.read_text(encoding="ascii").split()
        )
        self.assert_process_gone(leader_pid)
        self.assert_process_gone(child_pid)
        self.assert_probe_session_quiescent(session)

    def test_contract_probe_close_interrupt_still_closes_later_resources(self) -> None:
        sleeper, pid_path = self.sleeping_contract_probe_executable(
            "close_interrupt_appender_contract"
        )
        interruption = KeyboardInterrupt()
        original_close = COMPOSER._close_probe_resource
        close_names = []

        def interrupt_selector_close(name, resource):
            close_names.append(name)
            result = original_close(name, resource)
            if name == "selector":
                raise interruption
            return result

        with mock.patch.object(
            COMPOSER, "_close_probe_resource", side_effect=interrupt_selector_close
        ), mock.patch.object(
            COMPOSER, "APPENDER_CONTRACT_PROBE_TIMEOUT_SECONDS", 0.2
        ):
            with self.assertRaises(KeyboardInterrupt) as captured:
                self.probe_contract_path(sleeper)
        self.assertIs(captured.exception, interruption)
        self.assertEqual(close_names, ["selector", "stdout", "stderr"])
        self.assert_process_gone(self.read_contract_probe_pid(pid_path))

    def test_post_sealed_appender_handoff_substitution_rejects(self) -> None:
        archive, evidence_root, prefix, handoff_raw = self.produce(
            "handoff_content_substitution"
        )
        replacement = handoff_with_writer_outcome(
            handoff_raw,
            writer_status=9,
            clock_status=0,
        )
        replacement_path = archive / "streams/handoff-content-replacement.bin"
        replacement_path.write_bytes(replacement)
        original_run = COMPOSER._run_sealed_appender

        def run_then_substitute(*arguments, **kwargs):
            completed = original_run(*arguments, **kwargs)
            os.replace(
                replacement_path,
                archive / COMPOSER.PROCESS_HANDOFF_ARCHIVE_PATH,
            )
            return completed

        with self.assertRaisesRegex(
            COMPOSER.ArchiveComposerError,
            "persisted handoff differs after C append authentication",
        ), mock.patch.object(
            COMPOSER, "_run_sealed_appender", side_effect=run_then_substitute
        ) as runner:
            self.compose(archive, evidence_root, append_executable=self.appender)
        runner.assert_called_once()
        self.assertGreater(
            len((archive / "streams/memory-lifetime.bin").read_bytes()), len(prefix)
        )
        self.assertEqual(
            (archive / COMPOSER.PROCESS_HANDOFF_ARCHIVE_PATH).read_bytes(),
            replacement,
        )
        self.assertFalse((archive / "manifest.json").exists())

    def test_post_appender_handoff_identity_rebind_rejects(self) -> None:
        archive, evidence_root, prefix, handoff_raw = self.produce(
            "handoff_identity_substitution"
        )
        replacement_path = archive / "streams/handoff-identity-replacement.bin"
        replacement_path.write_bytes(handoff_raw)
        original_run = COMPOSER._run_sealed_appender

        def run_then_rebind(*arguments, **kwargs):
            completed = original_run(*arguments, **kwargs)
            os.replace(
                replacement_path,
                archive / COMPOSER.PROCESS_HANDOFF_ARCHIVE_PATH,
            )
            return completed

        with self.assertRaisesRegex(
            COMPOSER.ArchiveComposerError,
            "persisted handoff differs after C append authentication",
        ), mock.patch.object(
            COMPOSER, "_run_sealed_appender", side_effect=run_then_rebind
        ) as runner:
            self.compose(archive, evidence_root, append_executable=self.appender)
        runner.assert_called_once()
        stream = (archive / "streams/memory-lifetime.bin").read_bytes()
        self.assertGreater(len(stream), len(prefix))
        self.assertEqual(
            (archive / COMPOSER.PROCESS_HANDOFF_ARCHIVE_PATH).read_bytes(),
            handoff_raw,
        )
        self.assertFalse((archive / "manifest.json").exists())

    def test_real_sealed_appender_ignores_replaced_and_mutated_path(self) -> None:
        archive, evidence_root, prefix, handoff_raw = self.produce(
            "sealed_real_path_mutation"
        )
        original_raw = self.appender.read_bytes()
        original_mode = self.appender.stat().st_mode & 0o7777
        replacement = self.contract_probe_executable(
            "sealed-real-path-replacement",
            stdout_text="evil\n",
            stderr_text="",
            return_code=0,
        )
        original_authenticate = COMPOSER._require_authenticated_appender
        sealed = []

        def seal_then_mutate(*arguments, **kwargs):
            appender = original_authenticate(*arguments, **kwargs)
            sealed.append(appender)
            os.replace(replacement, self.appender)
            self.appender.write_bytes(b"not the sealed executable\n")
            return appender

        try:
            with mock.patch.object(
                COMPOSER,
                "_require_authenticated_appender",
                side_effect=seal_then_mutate,
            ):
                result = self.compose(
                    archive,
                    evidence_root,
                    append_executable=self.appender,
                )
        finally:
            self.appender.write_bytes(original_raw)
            self.appender.chmod(original_mode)
        self.assertEqual(len(sealed), 1)
        self.assertEqual(result.terminal_outcome, "complete")
        stream = (archive / COMPOSER.STREAM_ARCHIVE_PATH).read_bytes()
        self.assertGreater(len(stream), len(prefix))
        self.assertTrue(stream.startswith(prefix))
        self.assertEqual(
            (archive / COMPOSER.PROCESS_HANDOFF_ARCHIVE_PATH).read_bytes(), handoff_raw
        )
        self.assertFalse((archive / "streams/trailer.bin").exists())
        self.assertTrue((archive / COMPOSER.VERIFIER.STATUS.MANIFEST_PATH).exists())

    def test_post_publication_archive_and_manifest_rebind_reject(self) -> None:
        for index, target in enumerate(
            ("catalog/module.json", COMPOSER.VERIFIER.STATUS.MANIFEST_PATH)
        ):
            with self.subTest(target=target):
                archive, evidence_root, _prefix, _handoff = self.produce(
                    f"post_publication_rebind_{index}"
                )
                original_publish = COMPOSER._publish_once

                def replace_after_publish(
                    root: pathlib.Path,
                    relative_path: str,
                    raw: bytes,
                    *,
                    target_path: str = target,
                ) -> COMPOSER.FrozenInput:
                    published = original_publish(root, relative_path, raw)
                    if relative_path == target_path:
                        destination = root.joinpath(*target_path.split("/"))
                        replacement = destination.with_name(
                            f"{destination.name}.replacement"
                        )
                        replacement.write_bytes(b"post-publication replacement\n")
                        os.replace(replacement, destination)
                    return published

                with mock.patch.object(
                    COMPOSER,
                    "_publish_once",
                    side_effect=replace_after_publish,
                ):
                    with self.assertRaisesRegex(
                        COMPOSER.ArchiveComposerError,
                        "published archive re-read differs",
                    ):
                        self.compose(archive, evidence_root)

    def test_clean_build_with_unselected_dso_reaches_scientific_admission(self) -> None:
        self.assertEqual(BUILD_EVIDENCE.VERSION, {"major": 1, "minor": 4})
        self.assertEqual(self.evidence_value["version"], BUILD_EVIDENCE.VERSION)
        archive, evidence_root, _prefix, _handoff = self.produce("positive")
        result = self.compose(archive, evidence_root)
        self.assertTrue(result.scientific_admission_complete)
        decoded = WIRE.decode_container(
            (archive / "streams/memory-lifetime.bin").read_bytes()
        )
        records = [row for chunk in decoded.chunks for row in chunk.records]
        self.assertEqual(
            [(row.api_id, row.event_kind) for row in records],
            [(1, 1), (2, 1), (3, 2), (4, 3)],
        )
        run = COVERAGE.parse_canonical(
            (archive / "catalog/run-coverage.json").read_bytes()
        )
        unselected = next(
            row
            for row in run["module_population"]
            if row["logical_id"] == "unselected_fixture"
        )
        self.assertEqual(
            (
                unselected["configured"],
                unselected["observed"],
                unselected["load_state_id"],
            ),
            (False, False, 11),
        )
        module_catalog = COMPOSER.VERIFIER.SEMANTIC.parse_canonical(
            (archive / "catalog/module.json").read_bytes()
        )
        self.assertNotIn(
            "unselected_fixture",
            {row["logical_id"] for row in module_catalog["entries"]},
        )
        manifest = COMPOSER.VERIFIER.STATUS.parse_canonical(
            (archive / "manifest.json").read_bytes()
        )
        manifest_path_rows = [row["path"] for row in manifest["entries"]]
        manifest_paths = set(manifest_path_rows)
        expected = set(self.artifacts) | {
            BUILD_EVIDENCE.EVIDENCE_ARCHIVE_PATH,
            COMPOSER.PROCESS_HANDOFF_ARCHIVE_PATH,
            COMPOSER.STREAM_ARCHIVE_PATH,
            COMPOSER.VERIFIER.STATUS.POST_CLOSE_PATH,
            COMPOSER.VERIFIER.TRUSTED_RELEASE_AUTHORITY_PATH,
        }
        expected.update(COMPOSER.VERIFIER.TRUSTED_RELEASE_SOURCE_PATHS)
        expected.update(
            path for path, _flags in COMPOSER.VERIFIER.EXTERNAL_BY_KIND.values()
        )
        self.assertEqual(manifest_paths, expected)
        self.assertEqual(len(manifest_path_rows), len(manifest_paths))

    def test_conditional_population_real_paths_reject_before_append(self) -> None:
        for label, selected_value, preload, expected in (
            ("selected_missing", "1", False, (True, False, 10)),
            ("unselected_observed", "0", True, (False, True, 20)),
        ):
            with self.subTest(label=label):
                archive, evidence_root, prefix, handoff_raw = self.produce(
                    label,
                    preload_unselected_fixture=preload,
                    unselected_fixture_enabled=selected_value,
                )
                config = CONFIG.validate_effective_configuration_bytes(
                    (self.case_root / f"{label}-effective-config.json").read_bytes()
                )
                handoff = HANDOFF.decode_process_handoff(handoff_raw)
                _module_raw, population, _modules = COMPOSER._runtime_objects(
                    handoff, self.build, config
                )
                row = next(
                    item
                    for item in population
                    if item["logical_id"] == "unselected_fixture"
                )
                self.assertEqual(
                    (row["configured"], row["observed"], row["load_state_id"]),
                    expected,
                )
                with mock.patch.object(COMPOSER, "_publish_once") as publisher:
                    with mock.patch.object(
                        COMPOSER.subprocess, "run"
                    ) as appender:
                        with self.assertRaisesRegex(
                            COMPOSER.ArchiveComposerError,
                            "requires each build row to be selected and observed or unselected and unobserved",
                        ):
                            self.compose(archive, evidence_root)
                publisher.assert_not_called()
                appender.assert_not_called()
                self.assertEqual(
                    (archive / "streams/memory-lifetime.bin").read_bytes(), prefix
                )
                self.assertFalse((archive / "manifest.json").exists())

    def test_missing_conditional_selection_key_rejects_before_append(self) -> None:
        label = "missing_selection_key"
        archive, evidence_root, prefix, _handoff = self.produce(
            label, unselected_fixture_enabled=None
        )
        with mock.patch.object(COMPOSER, "_publish_once") as publisher:
            with mock.patch.object(COMPOSER.subprocess, "run") as appender:
                with self.assertRaisesRegex(
                    COMPOSER.ArchiveComposerError,
                    "missing required key 'unselected_fixture_enabled'",
                ):
                    self.compose(archive, evidence_root)
        publisher.assert_not_called()
        appender.assert_not_called()
        self.assertEqual(
            (archive / "streams/memory-lifetime.bin").read_bytes(), prefix
        )
        self.assertFalse((archive / "manifest.json").exists())

    def test_complete_rejects_noneligible_population_before_append(self) -> None:
        for label, population in (
            (
                "configured_missing",
                [{"configured": True, "observed": False, "load_state_id": 10}],
            ),
            (
                "unconfigured_observed",
                [{"configured": False, "observed": True, "load_state_id": 20}],
            ),
        ):
            with self.subTest(label=label):
                archive, evidence_root, _prefix, _handoff = self.produce(label)
                with mock.patch.object(
                    COMPOSER,
                    "_runtime_objects",
                    return_value=(b"", population, []),
                ):
                    with mock.patch.object(COMPOSER, "_publish_once") as publisher:
                        with mock.patch.object(
                            COMPOSER.subprocess, "run"
                        ) as appender:
                            with self.assertRaisesRegex(
                                COMPOSER.ArchiveComposerError,
                                "requires each build row to be selected and observed or unselected and unobserved",
                            ):
                                self.compose(archive, evidence_root)
                publisher.assert_not_called()
                appender.assert_not_called()

    def test_trusted_release_source_root_rejects_symlink_and_nonregular_leaf(self) -> None:
        root_link = self.case_root / "trusted-release-source-root-link"
        root_link.symlink_to(self.source_root, target_is_directory=True)
        with self.assertRaisesRegex(
            COMPOSER.ArchiveComposerError,
            "source root must be a non-symlink directory",
        ):
            COMPOSER._read_frozen_trusted_release_sources(root_link)

        copied_root = self.case_root / "trusted-release-source-copy"
        shutil.copytree(
            self.source_root,
            copied_root,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "*.pyo"),
        )
        repository_relative_path = next(
            iter(COMPOSER.VERIFIER.TRUSTED_RELEASE_SOURCE_PATHS.values())
        )
        source_leaf = copied_root / repository_relative_path
        source_leaf.unlink()
        source_leaf.symlink_to(ROOT / repository_relative_path)
        with self.assertRaisesRegex(
            COMPOSER.ArchiveComposerError,
            "source leaf unavailable",
        ):
            COMPOSER._read_frozen_trusted_release_sources(copied_root)

        source_leaf.unlink()
        source_leaf.mkdir()
        with self.assertRaisesRegex(
            COMPOSER.ArchiveComposerError,
            "bounded single-link regular source required",
        ):
            COMPOSER._read_frozen_trusted_release_sources(copied_root)

    def test_io_error_handoff_composes_failed_archive_without_admission(self) -> None:
        self.assert_negative_terminal_archive(
            label="writer_io_failure",
            writer_status=9,
            clock_status=0,
            expected_outcome="failed",
            expected_terminal=(6, 6, 6),
        )

    def test_clock_error_handoff_composes_incomplete_archive_without_admission(
        self,
    ) -> None:
        self.assert_negative_terminal_archive(
            label="writer_clock_failure",
            writer_status=11,
            clock_status=5,
            expected_outcome="incomplete",
            expected_terminal=(7, 7, 5),
        )

    def test_sampled_process_handoff_composes_to_scientific_admission(self) -> None:
        archive, evidence_root, _prefix, handoff_raw = self.produce(
            "sampled_positive", mode_id=3
        )
        result = self.compose(archive, evidence_root)
        self.assertTrue(result.scientific_admission_complete)
        config = CONFIG.validate_effective_configuration_bytes(
            (self.case_root / "sampled_positive-effective-config.json").read_bytes()
        )
        handoff = HANDOFF.decode_process_handoff(handoff_raw)
        self.assertEqual(handoff.writer.runtime_snapshot.mode_id, 3)
        self.assertEqual(handoff.writer.runtime_snapshot.table_entries, 512)
        self.assertEqual(
            handoff.writer.runtime_snapshot.sample_seed,
            int(config["sample_seed_hex"], 16),
        )
        self.assertEqual(
            handoff.writer.runtime_snapshot.sample_threshold,
            config["sample_threshold"],
        )
        self.assertEqual(handoff.writer.runtime_snapshot.table_probes, 8)
        self.assertEqual(handoff.writer.runtime_snapshot.table_shards, 256)
        self.assertEqual(
            (
                handoff.threads[0].sample_insertion_failures,
                handoff.threads[0].sample_lookup_failures,
                handoff.threads[0].sample_probe_exhaustions,
                handoff.threads[0].sample_pairing_failures,
            ),
            (0, 0, 0, 0),
        )
        decoded = WIRE.decode_container(
            (archive / "streams/memory-lifetime.bin").read_bytes()
        )
        records = [row for chunk in decoded.chunks for row in chunk.records]
        self.assertEqual(len(records), 4)
        self.assertEqual(
            [bool(row.flags & (1 << 16)) for row in records],
            [True, True, True, False],
        )
        self.assertEqual(
            [bool(row.flags & (1 << 15)) for row in records],
            [False, False, True, True],
        )

    def test_sampled_runtime_control_mismatch_rejects_before_append(self) -> None:
        archive, evidence_root, prefix, handoff = self.produce(
            "sampled_mismatch",
            mode_id=3,
            runtime_threshold_delta=-1,
        )
        with self.assertRaisesRegex(
            COMPOSER.ArchiveComposerError,
            "runtime configuration differs from authenticated catalog/coverage",
        ):
            self.compose(archive, evidence_root)
        self.assertEqual(
            (archive / "streams/memory-lifetime.bin").read_bytes(),
            prefix,
        )
        self.assertEqual(
            (archive / "streams/process-handoff.bin").read_bytes(),
            handoff,
        )
        self.assertFalse((archive / "manifest.json").exists())

    def test_paired_dirty_status_rejects_before_append(self) -> None:
        value = copy.deepcopy(self.evidence_value)
        artifacts = dict(self.artifacts)
        path = value["toolchain"]["source_status_path"]
        artifacts[path] = b"?? untracked-file\0"
        row = next(item for item in value["entries"] if item["path"] == path)
        row.update(bytes=len(artifacts[path]), sha256=sha(artifacts[path]))
        evidence_raw = COVERAGE.canonical_bytes(value)
        archive, evidence_root, prefix, handoff = self.produce(
            "dirty_status", evidence_raw=evidence_raw, artifacts=artifacts
        )
        with self.assertRaisesRegex(
            COMPOSER.ArchiveComposerError,
            "measured source status must be exactly empty",
        ):
            self.compose(archive, evidence_root)
        self.assertEqual(
            (archive / "streams/memory-lifetime.bin").read_bytes(), prefix
        )
        self.assertEqual(
            (archive / "streams/process-handoff.bin").read_bytes(), handoff
        )

    def test_paired_map_elf_and_libc_substitutions_reject(self) -> None:
        paths = (
            "input/build-evidence/archive_fixture.link-map.txt",
            "input/build-evidence/archive_fixture.elf",
            self.evidence_value["toolchain"]["libc_path"],
        )
        for index, path in enumerate(paths):
            with self.subTest(path=path):
                evidence_raw, artifacts = self.paired_mutation(
                    path, self.artifacts[path] + b"\n"
                )
                label = f"paired_substitution_{index}"
                archive, evidence_root, prefix, _handoff = self.produce(
                    label, evidence_raw=evidence_raw, artifacts=artifacts
                )
                with self.assertRaisesRegex(
                    COMPOSER.ArchiveComposerError,
                    "exact canonical build coverage bytes differ",
                ):
                    self.compose(archive, evidence_root)
                self.assertEqual(
                    (archive / "streams/memory-lifetime.bin").read_bytes(),
                    prefix,
                )

    def test_raw_commit_object_mismatch_rejects(self) -> None:
        value = copy.deepcopy(self.evidence_value)
        artifacts = dict(self.artifacts)
        path = value["toolchain"]["source_commit_object_path"]
        artifacts[path] += b"\n"
        row = next(item for item in value["entries"] if item["path"] == path)
        row.update(bytes=len(artifacts[path]), sha256=sha(artifacts[path]))
        evidence_raw = COVERAGE.canonical_bytes(value)
        with self.assertRaisesRegex(
            BUILD_EVIDENCE.BuildEvidenceError,
            "raw commit object does not hash to source HEAD",
        ):
            BUILD_EVIDENCE.validate_build_evidence_bytes(
                evidence_raw,
                artifacts,
                self.prepared.coverage_bytes,
                api_definition_sha256=API_DEFINITION_SHA256,
            )

    def test_missing_active_runtime_closure_rejects(self) -> None:
        value = copy.deepcopy(self.evidence_value)
        artifacts = dict(self.artifacts)
        value["logical_elfs"] = [
            row
            for row in value["logical_elfs"]
            if row["logical_id"] != "oai_memprof_active_runtime"
        ]
        removed = {
            path
            for path in artifacts
            if path.startswith(
                "input/build-evidence/oai_memprof_active_runtime."
            )
        }
        for path in removed:
            artifacts.pop(path)
        value["entries"] = [
            row for row in value["entries"] if row["path"] not in removed
        ]
        with self.assertRaisesRegex(
            BUILD_EVIDENCE.BuildEvidenceError,
            "supported imports require exactly one active-runtime DSO row",
        ):
            BUILD_EVIDENCE._derive(
                value,
                artifacts,
                api_definition_sha256=API_DEFINITION_SHA256,
                enforce_advertised_digest=False,
            )

    def test_configuration_output_directory_must_match_archive(self) -> None:
        archive, evidence_root, prefix, _handoff = self.produce(
            "output_mismatch", config_output_directory="/tmp/not-the-case-archive"
        )
        with self.assertRaisesRegex(
            COMPOSER.ArchiveComposerError,
            "effective configuration output directory differs from archive directory",
        ):
            self.compose(archive, evidence_root)
        self.assertEqual(
            (archive / "streams/memory-lifetime.bin").read_bytes(), prefix
        )

    def test_configuration_must_bind_exact_evidence_digest(self) -> None:
        archive, evidence_root, prefix, _handoff = self.produce(
            "config_mismatch", bound_digest="0" * 64
        )
        with self.assertRaisesRegex(
            COMPOSER.ArchiveComposerError,
            "effective configuration does not bind the exact build evidence",
        ):
            self.compose(archive, evidence_root)
        self.assertEqual(
            (archive / "streams/memory-lifetime.bin").read_bytes(), prefix
        )

    def test_configuration_must_bind_exact_trusted_release_authority_digest(self) -> None:
        archive, evidence_root, prefix, _handoff = self.produce(
            "authority_config_mismatch", bound_authority_digest="0" * 64
        )
        with self.assertRaisesRegex(
            COMPOSER.ArchiveComposerError,
            "effective configuration does not bind the external trusted-release authority",
        ):
            self.compose(archive, evidence_root)
        self.assertEqual(
            (archive / "streams/memory-lifetime.bin").read_bytes(), prefix
        )
        self.assertFalse((archive / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
