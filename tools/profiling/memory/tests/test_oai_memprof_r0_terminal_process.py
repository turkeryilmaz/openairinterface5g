#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0

"""Adversarial process/safety tests for the R0 terminal controller.

These tests use a real bounded child fixture instead of mocking process-group
semantics.  Mocks are restricted to documented internal observation seams for
cancellation and deterministic safety changes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import stat
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.profiling.memory import oai_memprof_r0_terminal_process as process  # noqa: E402


CHILD = Path(__file__).with_name("r0_terminal_test_child.py").resolve()


class TestCancellation(BaseException):
    """Deliberate BaseException used to verify cleanup before propagation."""


def _terminal_style_test_handler(signum, frame) -> None:
    """Model the R0 CLI's finite supported-signal cancellation handler."""

    del frame
    raise SystemExit(128 + int(signum))


def _wait_pid_absent(pid: int, timeout_seconds: float = 2.0) -> bool:
    stat_path = Path("/proc") / str(pid) / "stat"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not stat_path.exists():
            return True
        time.sleep(0.01)
    return not stat_path.exists()


def _direct_child_pids_bounded(
    parent_pid: int,
    *,
    max_proc_entries: int = 32768,
    deadline_seconds: float = 1.0,
) -> tuple[int, ...]:
    """Independently census direct children without production scan helpers."""

    deadline = time.monotonic() + deadline_seconds
    children: list[int] = []
    entries = 0
    with os.scandir("/proc") as iterator:
        for entry in iterator:
            if time.monotonic() >= deadline:
                raise AssertionError("test /proc census exceeded its deadline")
            if not entry.name.isascii() or not entry.name.isdecimal():
                continue
            entries += 1
            if entries > max_proc_entries:
                raise AssertionError("test /proc census exceeded its entry cap")
            try:
                with open(entry.path + "/stat", "rb", buffering=0) as stream:
                    data = stream.read(4097)
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if len(data) > 4096:
                raise AssertionError("test /proc stat record exceeded 4096 bytes")
            close = data.rfind(b")")
            fields = data[close + 2 :].split() if close >= 0 else []
            if len(fields) < 2:
                continue
            try:
                observed_parent = int(fields[1])
            except ValueError:
                continue
            if observed_parent == parent_pid:
                children.append(int(entry.name))
    return tuple(sorted(children))


class TerminalProcessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_supported_signal_handlers = {
            item: signal.getsignal(item)
            for item in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
        }
        for item, handler in self.original_supported_signal_handlers.items():
            self.addCleanup(signal.signal, item, handler)
        signal.signal(signal.SIGHUP, _terminal_style_test_handler)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, _terminal_style_test_handler)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="oai-memprof-r0-process-test-"
        )
        self.root = Path(self.temporary.name)
        self.cwd = self.root / "cwd"
        self.cwd.mkdir(mode=0o700)
        self.sensor = self.root / "temperature"
        self.sensor.write_text("42000\n", encoding="ascii")
        self.stdout = self.root / "stdout.bin"
        self.stderr = self.root / "stderr.bin"
        self.fixture_pids: set[int] = set()

    def tearDown(self) -> None:
        for pid in self.fixture_pids:
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
        for pid in self.fixture_pids:
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
        if process._CONTAINMENT_POISONED_DETAIL is not None:
            process._reset_containment_poison_for_tests()
        self.temporary.cleanup()

    def safety(
        self,
        *,
        ceiling: int = 90000,
        minimum_free_bytes: int = 0,
        poll_interval_seconds: float = 0.01,
        max_samples: int = 1000,
        sensor: Path | None = None,
    ) -> process.SafetySpec:
        return process.SafetySpec(
            sensors=(
                process.ThermalSensor(
                    path=self.sensor if sensor is None else sensor,
                    ceiling_millicelsius=ceiling,
                    minimum_plausible_millicelsius=1000,
                ),
            ),
            storage_path=self.root,
            minimum_free_bytes=minimum_free_bytes,
            poll_interval_seconds=poll_interval_seconds,
            max_samples=max_samples,
            sensor_max_bytes=32,
        )

    def limits(
        self,
        *,
        wall_seconds: float = 2.0,
        stdout_bytes: int = 4096,
        stderr_bytes: int = 4096,
        cleanup_seconds: float = 1.0,
        max_proc_entries: int = 4096,
        max_observed_live_descendants: int = 32,
        max_descendant_identities: int = 256,
        read_chunk_bytes: int = 512,
    ) -> process.CommandLimits:
        return process.CommandLimits(
            wall_seconds=wall_seconds,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            cleanup_seconds=cleanup_seconds,
            max_proc_entries=max_proc_entries,
            max_observed_live_descendants=max_observed_live_descendants,
            max_descendant_identities=max_descendant_identities,
            read_chunk_bytes=read_chunk_bytes,
        )

    def command(self, *child_arguments: str) -> process.CommandSpec:
        return process.CommandSpec(
            argv=(sys.executable, str(CHILD), *child_arguments),
            cwd=self.cwd,
            environment={"LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1"},
            stdout_path=self.stdout,
            stderr_path=self.stderr,
        )

    def run_child(
        self,
        *child_arguments: str,
        limits: process.CommandLimits | None = None,
        safety: process.SafetySpec | None = None,
    ) -> process.ProcessResult:
        result = process.run_bounded_command(
            self.command(*child_arguments),
            limits=self.limits() if limits is None else limits,
            safety=self.safety() if safety is None else safety,
        )
        self.assert_result_ledger(result)
        return result

    def assert_result_ledger(self, result: process.ProcessResult) -> None:
        supported = process.SUPPORTED_CANCELLATION_SIGNALS
        priority = process.SUPPORTED_CANCELLATION_SIGNAL_PRIORITY
        self.assertIs(type(supported), tuple)
        self.assertEqual(supported, tuple(sorted(set(supported))))
        self.assertTrue(supported)
        self.assertTrue(
            all(type(item) is int and item > 0 for item in supported),
            supported,
        )
        self.assertIs(type(priority), tuple)
        self.assertEqual(
            priority,
            (int(signal.SIGINT), int(signal.SIGTERM), int(signal.SIGHUP)),
        )
        self.assertEqual(set(priority), set(supported))
        for signal_numbers in (
            result.caller_signal_mask_numbers,
            result.deferred_supported_signal_numbers,
        ):
            self.assertIs(type(signal_numbers), tuple)
            self.assertEqual(
                signal_numbers,
                tuple(sorted(set(signal_numbers))),
            )
            self.assertTrue(
                all(type(item) is int and item > 0 for item in signal_numbers),
                signal_numbers,
            )
        self.assertFalse(
            set(result.caller_signal_mask_numbers).intersection(supported)
        )
        self.assertTrue(
            set(result.deferred_supported_signal_numbers).issubset(supported)
        )
        self.assertIs(result.signal_mask_restored, True)
        self.assertFalse(hasattr(result, "observed_descendant_identity_count"))
        self.assertFalse(hasattr(result, "unexpected_descendant_count"))
        populations = (
            (
                result.observed_descendant_identity_lower_bound,
                result.observed_descendant_identity_complete,
                result.observed_descendant_identities,
                result.observed_descendant_identities_truncated,
            ),
            (
                result.unexpected_descendant_identity_lower_bound,
                result.unexpected_descendant_identity_complete,
                result.unexpected_descendant_identities,
                result.unexpected_descendant_identities_truncated,
            ),
        )
        for lower_bound, complete, retained, truncated in populations:
            self.assertIs(type(lower_bound), int)
            self.assertGreaterEqual(lower_bound, 0)
            self.assertIs(type(complete), bool)
            self.assertIs(type(truncated), bool)
            self.assertGreaterEqual(lower_bound, len(retained))
            if complete:
                if not truncated:
                    self.assertEqual(lower_bound, len(retained))
            else:
                self.assertGreaterEqual(lower_bound, len(retained) + 1)
                self.assertTrue(truncated)
        self.assertEqual(
            len(result.unexpected_descendant_pids),
            len(result.unexpected_descendant_identities),
        )
        self.assertEqual(
            tuple(item[0] for item in result.unexpected_descendant_identities),
            result.unexpected_descendant_pids,
        )
        if not result.observed_descendant_identities_truncated:
            self.assertTrue(
                set(result.unexpected_descendant_identities).issubset(
                    set(result.observed_descendant_identities)
                )
            )

    def assert_failure(
        self,
        result: process.ProcessResult,
        expected: process.ProcessErrorCode,
    ) -> None:
        self.assertEqual(result.failure_code, expected, result.failure_detail)
        self.assertTrue(result.kill_attempted)
        self.assertTrue(result.cleanup_complete, result.failure_detail)
        self.assertIsNone(result.cleanup_failure_code)
        self.assertIsNone(result.cleanup_failure_detail)

    def test_sample_safety_returns_exact_observations(self) -> None:
        sample = process.sample_safety(self.safety(), monotonic_ns=123456)
        self.assertEqual(sample.monotonic_ns, 123456)
        self.assertEqual(
            sample.thermal_millicelsius,
            ((str(self.sensor), 42000),),
        )
        self.assertGreaterEqual(sample.free_bytes, 0)

    def test_sample_safety_rejects_missing_malformed_and_hot_sensor(self) -> None:
        missing = self.root / "missing-temperature"
        with self.assertRaises(process.ProcessError) as missing_context:
            process.sample_safety(self.safety(sensor=missing))
        self.assertEqual(
            missing_context.exception.code,
            process.ProcessErrorCode.SAFETY_READ_FAILED,
        )

        for malformed in ("", "42000 junk\n", "42000\n\n", "-1\n", "9" * 40):
            with self.subTest(malformed=malformed):
                self.sensor.write_text(malformed, encoding="ascii")
                with self.assertRaises(process.ProcessError) as context:
                    process.sample_safety(self.safety())
                self.assertEqual(
                    context.exception.code,
                    process.ProcessErrorCode.SAFETY_READ_FAILED,
                )

        self.sensor.write_text("90000\n", encoding="ascii")
        with self.assertRaises(process.ProcessError) as hot_context:
            process.sample_safety(self.safety(ceiling=90000))
        self.assertEqual(
            hot_context.exception.code,
            process.ProcessErrorCode.THERMAL_LIMIT,
        )

    def test_sample_safety_rejects_free_space_below_threshold(self) -> None:
        with self.assertRaises(process.ProcessError) as context:
            process.sample_safety(
                self.safety(minimum_free_bytes=(1 << 63) - 1)
            )
        self.assertEqual(
            context.exception.code,
            process.ProcessErrorCode.FREE_SPACE_LIMIT,
        )

    def test_thermal_minimum_rejects_false_safe_zero_and_accepts_equality(self) -> None:
        self.sensor.write_text("0\n", encoding="ascii")
        with self.assertRaises(process.ProcessError) as context:
            process.sample_safety(self.safety())
        self.assertEqual(
            context.exception.code,
            process.ProcessErrorCode.SAFETY_READ_FAILED,
        )
        self.assertIsNotNone(context.exception.sample)

        self.sensor.write_text("1000\n", encoding="ascii")
        sample = process.sample_safety(self.safety())
        self.assertEqual(
            sample.thermal_millicelsius,
            ((str(self.sensor), 1000),),
        )
        self.sensor.write_text("42000\n", encoding="ascii")

    def test_containment_preconditions_reject_threads_children_busy_and_prctl_failure(self) -> None:
        stop = threading.Event()
        thread = threading.Thread(target=stop.wait, daemon=False)
        thread.start()
        try:
            with self.assertRaises(process.ProcessError) as thread_context:
                self.run_child("exit", "0")
            self.assertEqual(
                thread_context.exception.code,
                process.ProcessErrorCode.CALLER_MULTITHREADED,
            )
            self.assertFalse(self.stdout.exists())
            self.assertFalse(self.stderr.exists())
        finally:
            stop.set()
            thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())

        preexisting = subprocess.Popen(
            [sys.executable, str(CHILD), "sleep", "2"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        self.fixture_pids.add(preexisting.pid)
        prior_subreaper = process._get_child_subreaper()
        try:
            with self.assertRaises(process.ProcessError) as child_context:
                self.run_child("exit", "0")
            self.assertEqual(
                child_context.exception.code,
                process.ProcessErrorCode.PREEXISTING_CHILD,
            )
            self.assertIsNone(preexisting.poll())
            self.assertEqual(process._get_child_subreaper(), prior_subreaper)
        finally:
            preexisting.kill()
            preexisting.wait(timeout=1.0)
            self.fixture_pids.discard(preexisting.pid)

        self.assertTrue(process._RUN_LOCK.acquire(blocking=False))
        try:
            with self.assertRaises(process.ProcessError) as busy_context:
                self.run_child("exit", "0")
            self.assertEqual(
                busy_context.exception.code,
                process.ProcessErrorCode.RUNNER_BUSY,
            )
        finally:
            process._RUN_LOCK.release()

        injected = process.ProcessError(
            process.ProcessErrorCode.SUBREAPER_FAILED,
            "prctl-test",
            "injected syscall failure",
        )
        with mock.patch.object(process, "_get_child_subreaper", side_effect=injected):
            with self.assertRaises(process.ProcessError) as prctl_context:
                self.run_child("exit", "0")
        self.assertEqual(
            prctl_context.exception.code,
            process.ProcessErrorCode.SUBREAPER_FAILED,
        )

    def test_second_containment_scan_child_is_ambiguous_and_poisoned_without_signal(self) -> None:
        prior_subreaper = process._get_child_subreaper()
        if prior_subreaper:
            process._set_child_subreaper(False)
        original_scan = process._scan_process_table
        scan_count = 0
        injected_pid: int | None = None

        def scan_with_second_call_child(max_entries, deadline, *, deadline_code):
            nonlocal scan_count, injected_pid
            scan_count += 1
            if scan_count == 2:
                injected_pid = os.fork()
                if injected_pid == 0:
                    try:
                        time.sleep(5)
                    finally:
                        os._exit(0)
                self.fixture_pids.add(injected_pid)
            table = original_scan(
                max_entries,
                deadline,
                deadline_code=deadline_code,
            )
            return table

        command_popen = mock.Mock(
            side_effect=AssertionError("command spawned after second-scan child")
        )
        try:
            with mock.patch.object(
                process, "_scan_process_table", scan_with_second_call_child
            ), mock.patch.object(process.subprocess, "Popen", command_popen):
                with self.assertRaises(process.ProcessError) as context:
                    self.run_child("exit", "0")
            self.assertEqual(
                context.exception.code,
                process.ProcessErrorCode.PREEXISTING_CHILD,
            )
            command_popen.assert_not_called()
            self.assertIsNotNone(injected_pid)
            self.assertFalse(_wait_pid_absent(injected_pid, 0.05), injected_pid)
            self.assertTrue(process._get_child_subreaper())
            self.assertIsNotNone(process._CONTAINMENT_POISONED_DETAIL)
            self.stdout = self.root / "ambiguous-child-poison.stdout"
            self.stderr = self.root / "ambiguous-child-poison.stderr"
            with self.assertRaises(process.ProcessError) as poisoned:
                self.run_child("exit", "0")
            self.assertEqual(
                poisoned.exception.code,
                process.ProcessErrorCode.CONTAINMENT_POISONED,
            )
        finally:
            if injected_pid is not None and not _wait_pid_absent(injected_pid, 0.05):
                try:
                    os.kill(injected_pid, 9)
                except ProcessLookupError:
                    pass
                try:
                    os.waitpid(injected_pid, 0)
                except ChildProcessError:
                    pass
                self.fixture_pids.discard(injected_pid)
            if process._CONTAINMENT_POISONED_DETAIL is not None:
                process._reset_containment_poison_for_tests()
            else:
                process._set_child_subreaper(prior_subreaper)

    def test_second_containment_scan_error_preserves_primary_and_poisons(self) -> None:
        prior_subreaper = process._get_child_subreaper()
        if prior_subreaper:
            process._set_child_subreaper(False)
        original_scan = process._scan_process_table
        scan_count = 0
        injected = process.ProcessError(
            process.ProcessErrorCode.PROCFS_LIMIT,
            "containment-second-scan",
            "injected second-scan error",
        )

        def fail_second_scan_once(max_entries, deadline, *, deadline_code):
            nonlocal scan_count
            scan_count += 1
            if scan_count == 2:
                raise injected
            return original_scan(
                max_entries,
                deadline,
                deadline_code=deadline_code,
            )

        command_popen = mock.Mock(
            side_effect=AssertionError("command spawned after second-scan error")
        )
        try:
            with mock.patch.object(
                process, "_scan_process_table", fail_second_scan_once
            ), mock.patch.object(process.subprocess, "Popen", command_popen):
                with self.assertRaises(process.ProcessError) as context:
                    self.run_child("exit", "0")
            self.assertIs(context.exception, injected)
            command_popen.assert_not_called()
            self.assertTrue(process._get_child_subreaper())
            self.assertIsNotNone(process._CONTAINMENT_POISONED_DETAIL)
        finally:
            if process._CONTAINMENT_POISONED_DETAIL is not None:
                process._reset_containment_poison_for_tests()
            else:
                process._set_child_subreaper(prior_subreaper)

    def test_second_scan_transient_observation_restores_only_after_empty_proof(self) -> None:
        prior_subreaper = process._get_child_subreaper()
        if prior_subreaper:
            process._set_child_subreaper(False)
        original_scan = process._scan_process_table
        original_set = process._set_child_subreaper
        owner_pid = os.getpid()
        scan_count = 0
        consecutive_empty = 0
        restoration_empty_counts: list[int] = []
        self_identity = process._read_proc_identity(owner_pid)
        self.assertIsNotNone(self_identity)
        synthetic = dataclasses.replace(
            self_identity,
            pid=2_147_000_001,
            parent_pid=owner_pid,
            start_time_ticks=self_identity.start_time_ticks + 1,
        )

        def transient_second_scan(max_entries, deadline, *, deadline_code):
            nonlocal scan_count, consecutive_empty
            scan_count += 1
            table = original_scan(
                max_entries,
                deadline,
                deadline_code=deadline_code,
            )
            if scan_count == 2:
                table[synthetic.pid] = synthetic
            if scan_count > 2:
                direct = [
                    identity
                    for identity in table.values()
                    if identity.parent_pid == owner_pid
                ]
                consecutive_empty = 0 if direct else consecutive_empty + 1
            return table

        def track_restoration(enabled: bool) -> None:
            if not enabled:
                restoration_empty_counts.append(consecutive_empty)
            original_set(enabled)

        command_popen = mock.Mock(
            side_effect=AssertionError("command spawned after second-scan child")
        )
        try:
            with mock.patch.object(
                process, "_scan_process_table", transient_second_scan
            ), mock.patch.object(
                process, "_set_child_subreaper", track_restoration
            ), mock.patch.object(
                process,
                "_signal_identity",
                side_effect=AssertionError("ambiguous child must not be signalled"),
            ), mock.patch.object(process.subprocess, "Popen", command_popen):
                with self.assertRaises(process.ProcessError) as context:
                    self.run_child("exit", "0")
            self.assertEqual(
                context.exception.code,
                process.ProcessErrorCode.PREEXISTING_CHILD,
            )
            command_popen.assert_not_called()
            self.assertTrue(restoration_empty_counts)
            self.assertGreaterEqual(restoration_empty_counts[-1], 2)
            self.assertFalse(process._get_child_subreaper())
            self.assertIsNone(process._CONTAINMENT_POISONED_DETAIL)
        finally:
            if process._CONTAINMENT_POISONED_DETAIL is not None:
                process._reset_containment_poison_for_tests()
            else:
                process._set_child_subreaper(prior_subreaper)

    def test_baseexception_immediately_after_lease_restores_state_and_unlocks(self) -> None:
        prior = process._get_child_subreaper()
        with mock.patch.object(
            process,
            "_open_output_parent",
            side_effect=TestCancellation("cancel immediately after containment lease"),
        ):
            with self.assertRaisesRegex(TestCancellation, "immediately after"):
                self.run_child("exit", "0")
        self.assertEqual(process._get_child_subreaper(), prior)
        self.assertIsNone(process._CONTAINMENT_POISONED_DETAIL)
        self.assertFalse(self.stdout.exists())
        self.assertFalse(self.stderr.exists())
        self.assertTrue(process._RUN_LOCK.acquire(blocking=False))
        process._RUN_LOCK.release()

    def test_subreaper_state_and_repeated_quiescence_are_recorded_and_restored(self) -> None:
        prior = process._get_child_subreaper()
        scan_counts = {"leader": 0, "post_reap": 0}
        original_scan = process._scan_descendants

        def count_scans(owner_pid, leader, max_entries, deadline, *, deadline_code):
            scan_counts["post_reap" if leader is None else "leader"] += 1
            return original_scan(
                owner_pid,
                leader,
                max_entries,
                deadline,
                deadline_code=deadline_code,
            )

        process._set_child_subreaper(True)
        try:
            with mock.patch.object(process, "_scan_descendants", count_scans):
                result = self.run_child("exit", "0")
            self.assertIsNone(result.failure_code, result.failure_detail)
            self.assertEqual(result.containment_mode, process.CONTAINMENT_MODEL)
            self.assertEqual(result.containment_caller_thread_count, 1)
            self.assertEqual(result.containment_preexisting_child_count, 0)
            self.assertTrue(result.containment_subreaper_previously_enabled)
            self.assertTrue(result.containment_subreaper_restored)
            self.assertTrue(process._get_child_subreaper())
            self.assertGreaterEqual(scan_counts["leader"], 2)
            self.assertGreaterEqual(scan_counts["post_reap"], 2)
            self.assertEqual(
                result.descendant_scan_count,
                scan_counts["leader"] + scan_counts["post_reap"],
            )
            self.assertIsNotNone(result.first_descendant_scan_monotonic_ns)
            self.assertIsNotNone(result.last_descendant_scan_monotonic_ns)
            self.assertIsNotNone(result.maximum_observed_descendant_scan_gap_ns)
            self.assertLessEqual(
                result.first_descendant_scan_monotonic_ns,
                result.last_descendant_scan_monotonic_ns,
            )
        finally:
            process._set_child_subreaper(prior)

    def test_descendant_scan_cadence_accumulator_is_exact_and_rejects_clock_reversal(self) -> None:
        cadence = process._ScanCadence()
        with mock.patch.object(
            process,
            "_monotonic_ns",
            side_effect=(100, 140, 165),
        ):
            cadence.record()
            cadence.record()
            cadence.record()
        self.assertEqual(cadence.count, 3)
        self.assertEqual(cadence.first_monotonic_ns, 100)
        self.assertEqual(cadence.last_monotonic_ns, 165)
        self.assertEqual(cadence.maximum_gap_ns, 40)

        with mock.patch.object(process, "_monotonic_ns", return_value=164):
            with self.assertRaises(process.ProcessError) as context:
                cadence.record()
        self.assertEqual(
            context.exception.code,
            process.ProcessErrorCode.CONTAINMENT_FAILED,
        )
        self.assertEqual(cadence.count, 3)

    def test_sysfs_style_reported_size_does_not_replace_bounded_content_length(self) -> None:
        information = os.lstat(self.sensor)
        pseudo_file = types.SimpleNamespace(
            st_dev=information.st_dev,
            st_ino=information.st_ino,
            st_mode=information.st_mode,
            st_nlink=information.st_nlink,
            st_size=4096,
            st_mtime_ns=information.st_mtime_ns,
            st_ctime_ns=information.st_ctime_ns,
        )
        with mock.patch.object(process.os, "fstat", return_value=pseudo_file):
            sample = process.sample_safety(self.safety())
        self.assertEqual(
            sample.thermal_millicelsius,
            ((str(self.sensor), 42000),),
        )

        self.sensor.write_bytes(b"123456")
        with mock.patch.object(process.os, "fstat", return_value=pseudo_file):
            with self.assertRaises(process.ProcessError) as context:
                process.sample_safety(
                    process.SafetySpec(
                        sensors=(
                            process.ThermalSensor(
                                path=self.sensor,
                                ceiling_millicelsius=90000,
                                minimum_plausible_millicelsius=1000,
                            ),
                        ),
                        storage_path=self.root,
                        minimum_free_bytes=0,
                        poll_interval_seconds=0.01,
                        max_samples=1000,
                        sensor_max_bytes=5,
                    )
                )
        self.assertEqual(
            context.exception.code,
            process.ProcessErrorCode.SAFETY_READ_FAILED,
        )

        self.sensor.write_bytes(b"42000\n")
        changed_pseudo_file = types.SimpleNamespace(**vars(pseudo_file))
        changed_pseudo_file.st_mtime_ns += 1
        with mock.patch.object(
            process.os,
            "fstat",
            side_effect=(pseudo_file, changed_pseudo_file),
        ):
            with self.assertRaises(process.ProcessError) as context:
                process.sample_safety(self.safety())
        self.assertEqual(
            context.exception.code,
            process.ProcessErrorCode.SAFETY_READ_FAILED,
        )

    def test_hot_preflight_never_spawns_the_command(self) -> None:
        pid_file = self.root / "must-not-start.pid"
        self.sensor.write_text("90000\n", encoding="ascii")
        try:
            result = self.run_child(
                "hold",
                "1",
                str(pid_file),
                safety=self.safety(ceiling=90000),
            )
        except process.ProcessError as error:
            self.assertEqual(error.code, process.ProcessErrorCode.THERMAL_LIMIT)
        else:
            self.assertEqual(
                result.failure_code,
                process.ProcessErrorCode.THERMAL_LIMIT,
                result.failure_detail,
            )
        self.assertFalse(pid_file.exists())

    def test_normal_exit_captures_exact_bytes_hashes_and_clean_environment(self) -> None:
        result = self.run_child("write", "both", "17")
        self.assertIsNone(result.failure_code, result.failure_detail)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout_bytes, 17)
        self.assertEqual(result.stderr_bytes, 17)
        self.assertEqual(self.stdout.read_bytes(), b"O" * 17)
        self.assertEqual(self.stderr.read_bytes(), b"E" * 17)
        self.assertEqual(result.stdout_sha256, hashlib.sha256(b"O" * 17).hexdigest())
        self.assertEqual(result.stderr_sha256, hashlib.sha256(b"E" * 17).hexdigest())
        self.assertFalse(result.kill_attempted)
        self.assertTrue(result.cleanup_complete)
        self.assertIsNone(result.cleanup_failure_code)
        self.assertIsNone(result.cleanup_failure_detail)
        self.assertEqual(result.unexpected_descendant_identity_lower_bound, 0)
        self.assertTrue(result.unexpected_descendant_identity_complete)
        self.assertGreaterEqual(len(result.safety_samples), 1)

        self.stdout = self.root / "environment.stdout"
        self.stderr = self.root / "environment.stderr"
        ambient_name = "OAI_MEMPROF_TEST_AMBIENT_SENTINEL"
        with mock.patch.dict(os.environ, {ambient_name: "must-not-leak"}):
            environment_result = self.run_child("environment", ambient_name)
        self.assertIsNone(
            environment_result.failure_code, environment_result.failure_detail
        )
        self.assertEqual(self.stdout.read_bytes(), b"<MISSING>")

    def test_nonzero_exit_is_a_structured_runtime_failure(self) -> None:
        result = self.run_child("write-exit", "both", "17", "7")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(
            result.failure_code,
            process.ProcessErrorCode.PROCESS_EXIT_NONZERO,
        )
        self.assertTrue(result.cleanup_complete, result.failure_detail)
        self.assertEqual(self.stdout.read_bytes(), b"O" * 17)
        self.assertEqual(self.stderr.read_bytes(), b"E" * 17)
        self.assertEqual(result.stdout_bytes, 17)
        self.assertEqual(result.stderr_bytes, 17)
        self.assertEqual(result.stdout_sha256, hashlib.sha256(b"O" * 17).hexdigest())
        self.assertEqual(result.stderr_sha256, hashlib.sha256(b"E" * 17).hexdigest())

    def test_preexisting_output_is_rejected_without_modification(self) -> None:
        sentinel = b"preexisting-evidence"
        self.stdout.write_bytes(sentinel)
        with self.assertRaises(process.ProcessError) as context:
            self.run_child("exit", "0")
        self.assertEqual(
            context.exception.code,
            process.ProcessErrorCode.OUTPUT_CREATE_FAILED,
        )
        self.assertEqual(self.stdout.read_bytes(), sentinel)
        self.assertFalse(self.stderr.exists())

    def test_output_parent_symlink_and_non_directory_are_rejected(self) -> None:
        outside = self.root / "outside"
        outside.mkdir()
        link = self.root / "output-link"
        link.symlink_to(outside, target_is_directory=True)
        command = self.command("exit", "0")
        command = process.CommandSpec(
            argv=command.argv,
            cwd=command.cwd,
            environment=command.environment,
            stdout_path=link / "stdout",
            stderr_path=link / "stderr",
        )
        with self.assertRaises(process.ProcessError) as link_context:
            process.run_bounded_command(
                command, limits=self.limits(), safety=self.safety()
            )
        self.assertIn(
            link_context.exception.code,
            (
                process.ProcessErrorCode.PATH_INVALID,
                process.ProcessErrorCode.OUTPUT_CREATE_FAILED,
            ),
        )
        self.assertEqual(list(outside.iterdir()), [])

        regular_parent = self.root / "regular-parent"
        regular_parent.write_bytes(b"not-a-directory")
        command = process.CommandSpec(
            argv=command.argv,
            cwd=command.cwd,
            environment=command.environment,
            stdout_path=regular_parent / "stdout",
            stderr_path=regular_parent / "stderr",
        )
        with self.assertRaises(process.ProcessError):
            process.run_bounded_command(
                command, limits=self.limits(), safety=self.safety()
            )

    def test_wall_time_limit_kills_signal_resistant_child(self) -> None:
        pid_directory = self.root / "timeout-nested"
        pid_directory.mkdir()
        result = self.run_child(
            "leader-running",
            "5",
            str(pid_directory),
            "1",
            "--new-session",
            "--close-pipes",
            limits=self.limits(wall_seconds=0.08, cleanup_seconds=0.8),
            safety=self.safety(poll_interval_seconds=0.005),
        )
        self.assert_failure(result, process.ProcessErrorCode.WALL_TIME_LIMIT)
        child_pid = int((pid_directory / "child-0.pid").read_text(encoding="ascii"))
        self.assertTrue(_wait_pid_absent(child_pid))

    def test_stdout_and_stderr_limits_are_independent_and_hard(self) -> None:
        cases = (
            ("stdout", process.ProcessErrorCode.STDOUT_LIMIT),
            ("stderr", process.ProcessErrorCode.STDERR_LIMIT),
        )
        for stream, code in cases:
            with self.subTest(stream=stream):
                self.stdout = self.root / "{}.stdout".format(stream)
                self.stderr = self.root / "{}.stderr".format(stream)
                result = self.run_child(
                    "write",
                    stream,
                    "131072",
                    limits=self.limits(stdout_bytes=1024, stderr_bytes=1024),
                )
                self.assert_failure(result, code)
                self.assertLessEqual(self.stdout.stat().st_size, 1024)
                self.assertLessEqual(self.stderr.stat().st_size, 1024)

    def test_nested_setsid_output_overflow_cleans_all_descendants(self) -> None:
        pid_directory = self.root / "output-overflow-nested"
        pid_directory.mkdir()
        result = self.run_child(
            "leader-running",
            "5",
            str(pid_directory),
            "1",
            "--new-session",
            "--stdout-bytes",
            "131072",
            limits=self.limits(
                stdout_bytes=1024,
                wall_seconds=1.0,
                cleanup_seconds=1.0,
            ),
            safety=self.safety(poll_interval_seconds=0.005),
        )
        self.assert_failure(result, process.ProcessErrorCode.STDOUT_LIMIT)
        self.assertLessEqual(self.stdout.stat().st_size, 1024)
        for pid in result.unexpected_descendant_pids:
            self.assertTrue(_wait_pid_absent(pid), pid)

    def test_output_fsync_failure_closes_fds_and_preserves_primary_write_error(self) -> None:
        for stream in ("stdout", "stderr"):
            with self.subTest(stream=stream):
                self.stdout = self.root / "fsync-{}.stdout".format(stream)
                self.stderr = self.root / "fsync-{}.stderr".format(stream)
                real_open = process.os.open
                real_write = process.os.write
                real_fsync = process.os.fsync
                real_close = process.os.close
                tracked: dict[str, int] = {}
                closed: set[int] = set()
                target_fsync_calls = 0

                def tracking_open(path, flags, mode=0o777, *args, **kwargs):
                    descriptor = real_open(path, flags, mode, *args, **kwargs)
                    if (
                        flags & os.O_WRONLY
                        and Path(path).name == self.stdout.name
                    ):
                        tracked["stdout"] = descriptor
                    elif (
                        flags & os.O_WRONLY
                        and Path(path).name == self.stderr.name
                    ):
                        tracked["stderr"] = descriptor
                    return descriptor

                def fail_target_write(descriptor: int, data: bytes) -> int:
                    if descriptor == tracked.get(stream):
                        raise OSError(5, "injected output write failure")
                    return real_write(descriptor, data)

                def fail_target_fsync(descriptor: int) -> None:
                    nonlocal target_fsync_calls
                    if descriptor == tracked.get(stream):
                        target_fsync_calls += 1
                        if target_fsync_calls == 2:
                            raise OSError(5, "injected final output fsync failure")
                    real_fsync(descriptor)

                def tracking_close(descriptor: int) -> None:
                    if descriptor in tracked.values():
                        closed.add(descriptor)
                    real_close(descriptor)

                with mock.patch.object(process.os, "open", tracking_open), mock.patch.object(
                    process.os, "write", fail_target_write
                ), mock.patch.object(
                    process.os, "fsync", fail_target_fsync
                ), mock.patch.object(
                    process.os, "close", tracking_close
                ):
                    with self.assertRaises(process.ProcessError) as context:
                        self.run_child("write", stream, "17")
                self.assertEqual(
                    context.exception.code,
                    process.ProcessErrorCode.OUTPUT_WRITE_FAILED,
                )
                self.assertEqual(set(tracked.values()), closed)
                self.assertTrue(
                    any(
                        "descriptor close failures" in note
                        for note in getattr(context.exception, "__notes__", ())
                    ),
                    getattr(context.exception, "__notes__", ()),
                )
                for descriptor in tracked.values():
                    with self.assertRaises(OSError):
                        os.fstat(descriptor)

    def test_output_parent_fsync_failure_cannot_return_success_or_replace_primary(self) -> None:
        real_fsync = process.os.fsync
        directory_fsync_calls = 0

        def fail_directory_fsync(descriptor: int) -> None:
            nonlocal directory_fsync_calls
            if stat.S_ISDIR(os.fstat(descriptor).st_mode):
                directory_fsync_calls += 1
                if directory_fsync_calls == 2:
                    raise OSError(5, "injected final output-parent fsync failure")
            real_fsync(descriptor)

        with mock.patch.object(process.os, "fsync", fail_directory_fsync):
            result = self.run_child("exit", "0")
        self.assertEqual(
            result.failure_code,
            process.ProcessErrorCode.OUTPUT_WRITE_FAILED,
            result.failure_detail,
        )
        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.cleanup_complete)
        self.assertTrue(result.containment_subreaper_restored)

        self.stdout = self.root / "nonzero-dir-fsync.stdout"
        self.stderr = self.root / "nonzero-dir-fsync.stderr"
        directory_fsync_calls = 0
        with mock.patch.object(process.os, "fsync", fail_directory_fsync):
            nonzero = self.run_child("exit", "7")
        self.assertEqual(
            nonzero.failure_code,
            process.ProcessErrorCode.PROCESS_EXIT_NONZERO,
            nonzero.failure_detail,
        )
        self.assertIn(
            "finalization=output_write_failed",
            nonzero.failure_detail or "",
        )

        self.stdout = self.root / "write-dir-fsync.stdout"
        self.stderr = self.root / "write-dir-fsync.stderr"
        primary = process.ProcessError(
            process.ProcessErrorCode.OUTPUT_WRITE_FAILED,
            "write-output",
            "injected primary write failure",
            self.stdout,
        )
        directory_fsync_calls = 0
        with mock.patch.object(process, "_write_all", side_effect=primary), mock.patch.object(
            process.os,
            "fsync",
            fail_directory_fsync,
        ):
            with self.assertRaises(process.ProcessError) as context:
                self.run_child("write", "stdout", "17")
        self.assertIs(context.exception, primary)
        self.assertTrue(
            any(
                "bounded descriptor close failures" in note
                for note in getattr(context.exception, "__notes__", ())
            ),
            getattr(context.exception, "__notes__", ()),
        )

    def test_output_files_and_unique_parents_are_durable_before_spawn_and_at_exit(self) -> None:
        stdout_parent = self.root / "durability-stdout"
        stderr_parent = self.root / "durability-stderr"
        stdout_parent.mkdir()
        stderr_parent.mkdir()
        self.stdout = stdout_parent / "stdout.bin"
        self.stderr = stderr_parent / "stderr.bin"
        real_open = process.os.open
        real_fsync = process.os.fsync
        real_popen = process.subprocess.Popen
        descriptor_roles: dict[int, str] = {}
        parent_paths: dict[int, Path] = {}
        events: list[str] = []

        def tracking_open(path, flags, mode=0o777, *args, **kwargs):
            descriptor = real_open(path, flags, mode, *args, **kwargs)
            if flags & os.O_DIRECTORY:
                parent = Path(path)
                parent_paths[descriptor] = parent
                descriptor_roles[descriptor] = f"parent:{parent.name}"
            elif flags & os.O_WRONLY:
                parent_descriptor = kwargs.get("dir_fd")
                parent = parent_paths[parent_descriptor]
                absolute = parent / Path(path)
                role = "stdout" if absolute == self.stdout else "stderr"
                descriptor_roles[descriptor] = f"file:{role}"
            return descriptor

        def tracking_fsync(descriptor: int) -> None:
            role = descriptor_roles.get(descriptor)
            if role is not None:
                events.append(role)
            real_fsync(descriptor)

        def tracking_popen(*args, **kwargs):
            events.append("popen")
            return real_popen(*args, **kwargs)

        with mock.patch.object(process.os, "open", tracking_open), mock.patch.object(
            process.os, "fsync", tracking_fsync
        ), mock.patch.object(process.subprocess, "Popen", tracking_popen):
            result = self.run_child("exit", "0")
        self.assertIsNone(result.failure_code, result.failure_detail)
        spawn_index = events.index("popen")
        before_spawn = events[:spawn_index]
        after_spawn = events[spawn_index + 1 :]
        self.assertEqual(before_spawn[:2], ["file:stdout", "file:stderr"])
        self.assertCountEqual(
            before_spawn[2:],
            [
                f"parent:{stdout_parent.name}",
                f"parent:{stderr_parent.name}",
            ],
        )
        self.assertEqual(after_spawn[:2], ["file:stdout", "file:stderr"])
        self.assertCountEqual(
            after_spawn[2:],
            [
                f"parent:{stdout_parent.name}",
                f"parent:{stderr_parent.name}",
            ],
        )

    def test_presync_file_or_parent_failure_never_spawns_and_restores_after_proof(self) -> None:
        prior_subreaper = process._get_child_subreaper()
        if prior_subreaper:
            process._set_child_subreaper(False)
        self.addCleanup(process._set_child_subreaper, prior_subreaper)
        for target in ("file", "parent"):
            with self.subTest(target=target):
                stdout_parent = self.root / f"presync-{target}-stdout"
                stderr_parent = self.root / f"presync-{target}-stderr"
                stdout_parent.mkdir()
                stderr_parent.mkdir()
                self.stdout = stdout_parent / "stdout.bin"
                self.stderr = stderr_parent / "stderr.bin"
                real_open = process.os.open
                real_close = process.os.close
                real_fsync = process.os.fsync
                real_scan = process._scan_descendants
                real_set_subreaper = process._set_child_subreaper
                descriptor_roles: dict[int, str] = {}
                parent_paths: dict[int, Path] = {}
                tracked_descriptors: set[int] = set()
                closed_descriptors: set[int] = set()
                lifecycle: list[str] = []
                injected = False

                def tracking_open(path, flags, mode=0o777, *args, **kwargs):
                    descriptor = real_open(path, flags, mode, *args, **kwargs)
                    if flags & os.O_DIRECTORY:
                        parent = Path(path)
                        parent_paths[descriptor] = parent
                        descriptor_roles[descriptor] = "parent"
                        tracked_descriptors.add(descriptor)
                    elif flags & os.O_WRONLY:
                        parent = parent_paths[kwargs.get("dir_fd")]
                        absolute = parent / Path(path)
                        descriptor_roles[descriptor] = (
                            "stdout" if absolute == self.stdout else "stderr"
                        )
                        tracked_descriptors.add(descriptor)
                    return descriptor

                def fail_presync(descriptor: int) -> None:
                    nonlocal injected
                    role = descriptor_roles.get(descriptor)
                    should_fail = (
                        target == "file" and role == "stdout"
                    ) or (target == "parent" and role == "parent")
                    if should_fail and not injected:
                        injected = True
                        raise OSError(5, f"injected pre-spawn {target} fsync failure")
                    real_fsync(descriptor)

                def tracking_close(descriptor: int) -> None:
                    if descriptor in tracked_descriptors:
                        closed_descriptors.add(descriptor)
                    real_close(descriptor)

                def proof_scan(*args, **kwargs):
                    descendants = real_scan(*args, **kwargs)
                    lifecycle.append(
                        "proof-nonempty" if descendants else "proof-empty"
                    )
                    return descendants

                def tracking_set_subreaper(enabled: bool) -> None:
                    lifecycle.append(f"subreaper:{int(enabled)}")
                    real_set_subreaper(enabled)

                popen = mock.Mock(
                    side_effect=AssertionError("Popen reached before output durability")
                )
                with mock.patch.object(process.os, "open", tracking_open), mock.patch.object(
                    process.os, "fsync", fail_presync
                ), mock.patch.object(
                    process.os, "close", tracking_close
                ), mock.patch.object(
                    process, "_scan_descendants", proof_scan
                ), mock.patch.object(
                    process, "_set_child_subreaper", tracking_set_subreaper
                ), mock.patch.object(process.subprocess, "Popen", popen):
                    with self.assertRaises(process.ProcessError) as context:
                        self.run_child("exit", "0")
                self.assertTrue(injected)
                self.assertEqual(
                    context.exception.code,
                    process.ProcessErrorCode.OUTPUT_WRITE_FAILED,
                )
                popen.assert_not_called()
                self.assertEqual(tracked_descriptors, closed_descriptors)
                restoration = len(lifecycle) - 1 - lifecycle[::-1].index(
                    "subreaper:0"
                )
                self.assertGreaterEqual(restoration, 2)
                self.assertEqual(
                    lifecycle[restoration - 2 : restoration],
                    ["proof-empty", "proof-empty"],
                )
                self.assertFalse(process._get_child_subreaper())
                self.assertIsNone(process._CONTAINMENT_POISONED_DETAIL)
                self.assertTrue(self.stdout.exists())
                self.assertTrue(self.stderr.exists())

    def _assert_descendant_case(self, *options: str) -> None:
        pid_file = self.root / ("descendant-{}.pid".format(len(options)))
        result = self.run_child(
            "leader-exit",
            "5",
            str(pid_file),
            *options,
            limits=self.limits(wall_seconds=1.0, cleanup_seconds=1.0),
        )
        self.assertEqual(
            result.failure_code,
            process.ProcessErrorCode.DESCENDANT_REMAINS,
            result.failure_detail,
        )
        self.assertTrue(result.kill_attempted)
        self.assertTrue(result.cleanup_complete, result.failure_detail)
        descendant_pid = int(pid_file.read_text(encoding="ascii"))
        self.fixture_pids.add(descendant_pid)
        self.assertIn(descendant_pid, result.unexpected_descendant_pids)
        self.assertEqual(
            result.unexpected_descendant_identity_lower_bound,
            len(result.unexpected_descendant_identities),
        )
        self.assertTrue(result.unexpected_descendant_identity_complete)
        self.assertFalse(result.unexpected_descendant_identities_truncated)
        self.assertGreaterEqual(result.maximum_observed_live_descendant_count, 1)
        self.assertTrue(_wait_pid_absent(descendant_pid))
        self.fixture_pids.discard(descendant_pid)

    def test_leader_exit_descendant_retaining_pipes_is_detected_and_killed(self) -> None:
        self._assert_descendant_case()

    def test_leader_exit_descendant_closing_pipes_is_detected_and_killed(self) -> None:
        self._assert_descendant_case("--close-pipes")

    def test_same_session_descendant_in_alternate_process_group_is_contained(self) -> None:
        self._assert_descendant_case(
            "--close-pipes",
            "--alternate-process-group",
            "--ignore-term",
        )

    def test_new_session_descendants_with_open_and_closed_pipes_are_contained(self) -> None:
        for options in (("--new-session",), ("--new-session", "--close-pipes")):
            with self.subTest(options=options):
                self.stdout = self.root / ("setsid-{}.stdout".format(len(options)))
                self.stderr = self.root / ("setsid-{}.stderr".format(len(options)))
                self._assert_descendant_case(*options)

    def test_double_fork_leader_exit_reaps_exact_grandchild_without_zombie(self) -> None:
        pid_file = self.root / "double-fork-grandchild.pid"
        result = self.run_child(
            "double-fork",
            "5",
            str(pid_file),
            "--close-pipes",
            limits=self.limits(cleanup_seconds=1.0),
            safety=self.safety(poll_interval_seconds=0.005),
        )
        grandchild_pid = int(pid_file.read_text(encoding="ascii"))
        self.fixture_pids.add(grandchild_pid)
        self.assertEqual(
            result.failure_code,
            process.ProcessErrorCode.DESCENDANT_REMAINS,
            result.failure_detail,
        )
        self.assertTrue(result.cleanup_complete, result.failure_detail)
        self.assertIn(grandchild_pid, result.unexpected_descendant_pids)
        self.assertTrue(_wait_pid_absent(grandchild_pid))
        self.fixture_pids.discard(grandchild_pid)

    def test_rapid_double_fork_reparent_race_contains_every_grandchild(self) -> None:
        pid_directory = self.root / "rapid-double-fork"
        pid_directory.mkdir()
        result = self.run_child(
            "rapid-double-fork",
            "5",
            str(pid_directory),
            "4",
            limits=self.limits(cleanup_seconds=1.5),
            safety=self.safety(poll_interval_seconds=0.003),
        )
        pids = {
            int(path.read_text(encoding="ascii"))
            for path in pid_directory.glob("grandchild-*.pid")
        }
        self.fixture_pids.update(pids)
        self.assertEqual(len(pids), 4)
        self.assertEqual(
            result.failure_code,
            process.ProcessErrorCode.DESCENDANT_REMAINS,
            result.failure_detail,
        )
        self.assertTrue(result.cleanup_complete, result.failure_detail)
        self.assertTrue(pids.issubset(set(result.unexpected_descendant_pids)))
        for pid in pids:
            self.assertTrue(_wait_pid_absent(pid), pid)
            self.fixture_pids.discard(pid)

    def test_transient_descendants_are_observed_without_becoming_residual_failures(self) -> None:
        pid_directory = self.root / "transient-sequential"
        pid_directory.mkdir()
        result = self.run_child(
            "sequential-descendants",
            "0.08",
            str(pid_directory),
            "2",
            limits=self.limits(
                max_observed_live_descendants=1,
                max_descendant_identities=8,
                cleanup_seconds=1.0,
            ),
            safety=self.safety(poll_interval_seconds=0.003),
        )
        self.assertIsNone(result.failure_code, result.failure_detail)
        self.assertTrue(result.cleanup_complete, result.failure_detail)
        self.assertEqual(result.maximum_observed_live_descendant_count, 1)
        self.assertGreaterEqual(result.observed_descendant_identity_lower_bound, 2)
        self.assertTrue(result.observed_descendant_identity_complete)
        self.assertEqual(result.unexpected_descendant_identity_lower_bound, 0)
        self.assertTrue(result.unexpected_descendant_identity_complete)
        self.assertEqual(result.unexpected_descendant_pids, ())

    def test_sequential_fork_churn_enforces_cumulative_identity_bound(self) -> None:
        pid_directory = self.root / "bounded-sequential"
        pid_directory.mkdir()
        result = self.run_child(
            "sequential-descendants",
            "0.08",
            str(pid_directory),
            "5",
            limits=self.limits(
                max_observed_live_descendants=1,
                max_descendant_identities=2,
                cleanup_seconds=1.0,
            ),
            safety=self.safety(poll_interval_seconds=0.003),
        )
        self.assertEqual(
            result.failure_code,
            process.ProcessErrorCode.DESCENDANT_IDENTITY_LIMIT,
            result.failure_detail,
        )
        self.assertTrue(result.cleanup_complete, result.failure_detail)
        self.assertLessEqual(result.maximum_observed_live_descendant_count, 1)
        self.assertGreaterEqual(result.observed_descendant_identity_lower_bound, 3)
        self.assertFalse(result.observed_descendant_identity_complete)
        self.assertTrue(result.observed_descendant_identities_truncated)
        self.assertGreaterEqual(result.unexpected_descendant_identity_lower_bound, 3)
        self.assertFalse(result.unexpected_descendant_identity_complete)
        self.assertTrue(result.unexpected_descendant_identities_truncated)

    def test_identity_population_bounds_are_self_describing_when_complete_or_overflowed(self) -> None:
        complete_directory = self.root / "complete-identity-population"
        complete_directory.mkdir()
        complete = process.run_bounded_command(
            self.command(
                "sequential-descendants",
                "0.08",
                str(complete_directory),
                "2",
            ),
            limits=self.limits(
                max_observed_live_descendants=1,
                max_descendant_identities=8,
            ),
            safety=self.safety(poll_interval_seconds=0.003),
        )
        self.assertIsNone(complete.failure_code, complete.failure_detail)
        self.assertTrue(complete.observed_descendant_identity_complete)
        self.assertEqual(
            complete.observed_descendant_identity_lower_bound,
            len(complete.observed_descendant_identities),
        )
        self.assertFalse(complete.observed_descendant_identities_truncated)
        self.assertTrue(complete.unexpected_descendant_identity_complete)
        self.assertEqual(complete.unexpected_descendant_identity_lower_bound, 0)
        self.assertEqual(complete.unexpected_descendant_identities, ())
        self.assertFalse(complete.unexpected_descendant_identities_truncated)
        self.assertFalse(hasattr(complete, "observed_descendant_identity_count"))
        self.assertFalse(hasattr(complete, "unexpected_descendant_count"))

        self.stdout = self.root / "overflow-identities.stdout"
        self.stderr = self.root / "overflow-identities.stderr"
        overflow_directory = self.root / "overflow-identity-population"
        overflow_directory.mkdir()
        overflow = process.run_bounded_command(
            self.command(
                "sequential-descendants",
                "0.08",
                str(overflow_directory),
                "5",
            ),
            limits=self.limits(
                max_observed_live_descendants=1,
                max_descendant_identities=2,
            ),
            safety=self.safety(poll_interval_seconds=0.003),
        )
        self.assertEqual(
            overflow.failure_code,
            process.ProcessErrorCode.DESCENDANT_IDENTITY_LIMIT,
        )
        self.assertFalse(overflow.observed_descendant_identity_complete)
        self.assertGreaterEqual(
            overflow.observed_descendant_identity_lower_bound,
            len(overflow.observed_descendant_identities) + 1,
        )
        self.assertTrue(overflow.observed_descendant_identities_truncated)
        self.assertFalse(overflow.unexpected_descendant_identity_complete)
        self.assertGreaterEqual(
            overflow.unexpected_descendant_identity_lower_bound,
            len(overflow.unexpected_descendant_identities) + 1,
        )
        self.assertTrue(overflow.unexpected_descendant_identities_truncated)
        self.assertFalse(hasattr(overflow, "observed_descendant_identity_count"))
        self.assertFalse(hasattr(overflow, "unexpected_descendant_count"))

        self.stdout = self.root / "zero-cap-identities.stdout"
        self.stderr = self.root / "zero-cap-identities.stderr"
        zero_cap_pid = self.root / "zero-cap-child.pid"
        zero_cap = process.run_bounded_command(
            self.command(
                "leader-exit",
                "5",
                str(zero_cap_pid),
                "--new-session",
                "--close-pipes",
            ),
            limits=self.limits(
                max_observed_live_descendants=0,
                max_descendant_identities=0,
            ),
            safety=self.safety(poll_interval_seconds=0.003),
        )
        self.assertEqual(
            zero_cap.failure_code,
            process.ProcessErrorCode.DESCENDANT_LIMIT,
        )
        self.assertFalse(zero_cap.observed_descendant_identity_complete)
        self.assertGreaterEqual(zero_cap.observed_descendant_identity_lower_bound, 1)
        self.assertEqual(zero_cap.observed_descendant_identities, ())
        self.assertTrue(zero_cap.observed_descendant_identities_truncated)
        self.assertFalse(zero_cap.unexpected_descendant_identity_complete)
        self.assertGreaterEqual(zero_cap.unexpected_descendant_identity_lower_bound, 1)
        self.assertEqual(zero_cap.unexpected_descendant_identities, ())
        self.assertTrue(zero_cap.unexpected_descendant_identities_truncated)
        self.assertTrue(zero_cap.cleanup_complete, zero_cap.cleanup_failure_detail)
        self.assertEqual(_direct_child_pids_bounded(os.getpid()), ())
        if zero_cap_pid.exists():
            child_pid = int(zero_cap_pid.read_text(encoding="ascii"))
            self.assertTrue(_wait_pid_absent(child_pid), child_pid)

    def test_descendant_population_limit_is_causal(self) -> None:
        pid_file = self.root / "limited-descendant.pid"
        result = self.run_child(
            "leader-exit",
            "5",
            str(pid_file),
            "--close-pipes",
            limits=self.limits(
                max_observed_live_descendants=0,
                cleanup_seconds=1.0,
            ),
        )
        self.assertEqual(
            result.failure_code,
            process.ProcessErrorCode.DESCENDANT_LIMIT,
            result.failure_detail,
        )
        self.assertTrue(result.cleanup_complete, result.failure_detail)
        self.assertGreaterEqual(
            result.unexpected_descendant_identity_lower_bound,
            1,
        )
        self.assertTrue(result.unexpected_descendant_pids)
        for descendant_pid in result.unexpected_descendant_pids:
            self.assertTrue(_wait_pid_absent(descendant_pid))

    def test_live_leader_descendant_population_limit_is_enforced_immediately(self) -> None:
        pid_directory = self.root / "live-descendants"
        pid_directory.mkdir()
        result = self.run_child(
            "leader-running",
            "5",
            str(pid_directory),
            "2",
            limits=self.limits(
                wall_seconds=2.0,
                max_observed_live_descendants=1,
                cleanup_seconds=1.0,
            ),
            safety=self.safety(poll_interval_seconds=0.005),
        )
        self.assertEqual(
            result.failure_code,
            process.ProcessErrorCode.DESCENDANT_LIMIT,
            result.failure_detail,
        )
        self.assertTrue(result.kill_attempted)
        self.assertTrue(result.cleanup_complete, result.failure_detail)
        self.assertIsNone(result.cleanup_failure_code)
        self.assertIsNone(result.cleanup_failure_detail)
        self.assertLess(
            result.ended_monotonic_ns - result.started_monotonic_ns,
            1_000_000_000,
        )
        self.assertGreaterEqual(
            result.unexpected_descendant_identity_lower_bound,
            2,
        )
        self.assertGreaterEqual(len(result.unexpected_descendant_pids), 2)
        for child_pid in result.unexpected_descendant_pids:
            self.assertTrue(_wait_pid_absent(child_pid))

    def test_procfs_scan_limit_fails_closed_and_cleans_the_child(self) -> None:
        pid_directory = self.root / "procfs-limit"
        pid_directory.mkdir()
        pid_file = pid_directory / "child-0.pid"
        original_scan = process._scan_process_table
        post_spawn_scans = 0

        def fail_after_spawn(max_entries, deadline, *, deadline_code):
            nonlocal post_spawn_scans
            if pid_file.exists():
                post_spawn_scans += 1
            if post_spawn_scans > 1:
                raise process.ProcessError(
                    process.ProcessErrorCode.PROCFS_LIMIT,
                    "procfs-scan",
                    "injected post-spawn process-table bound",
                )
            return original_scan(
                max_entries,
                deadline,
                deadline_code=deadline_code,
            )

        with mock.patch.object(process, "_scan_process_table", fail_after_spawn):
            result = self.run_child(
                "leader-running",
                "5",
                str(pid_directory),
                "1",
                "--new-session",
                "--close-pipes",
                limits=self.limits(cleanup_seconds=0.25),
                safety=self.safety(poll_interval_seconds=0.005),
            )
        self.assertEqual(
            result.failure_code,
            process.ProcessErrorCode.PROCFS_LIMIT,
            result.failure_detail,
        )
        self.assertTrue(result.kill_attempted)
        self.assertFalse(result.cleanup_complete)
        self.assertEqual(
            result.cleanup_failure_code,
            process.ProcessErrorCode.PROCFS_LIMIT,
        )
        self.assertTrue(result.cleanup_failure_detail)
        self.assertIsNotNone(result.pid)
        self.assertTrue(_wait_pid_absent(result.pid))
        descendant_pid = int(pid_file.read_text(encoding="ascii"))
        self.assertTrue(_wait_pid_absent(descendant_pid))
        with self.assertRaises(process.ProcessError) as poisoned_context:
            self.stdout = self.root / "poisoned.stdout"
            self.stderr = self.root / "poisoned.stderr"
            self.run_child("exit", "0")
        self.assertEqual(
            poisoned_context.exception.code,
            process.ProcessErrorCode.CONTAINMENT_POISONED,
        )
        process._reset_containment_poison_for_tests()

    def test_transient_procfs_deadline_preserves_primary_and_recovers_cleanup(self) -> None:
        pid_directory = self.root / "deadline-nested"
        pid_directory.mkdir()
        pid_file = pid_directory / "child-0.pid"
        original_scan = process._scan_process_table
        injected = False

        def fail_once(max_entries, deadline, *, deadline_code):
            nonlocal injected
            if pid_file.exists() and not injected:
                injected = True
                raise process.ProcessError(
                    process.ProcessErrorCode.WALL_TIME_LIMIT,
                    "procfs-scan",
                    "injected complete-scan deadline",
                )
            return original_scan(
                max_entries,
                deadline,
                deadline_code=deadline_code,
            )

        with mock.patch.object(process, "_scan_process_table", fail_once):
            result = self.run_child(
                "leader-running",
                "5",
                str(pid_directory),
                "1",
                "--new-session",
                "--close-pipes",
                limits=self.limits(cleanup_seconds=1.0),
                safety=self.safety(poll_interval_seconds=0.005),
            )
        self.assertTrue(injected)
        self.assertEqual(
            result.failure_code,
            process.ProcessErrorCode.WALL_TIME_LIMIT,
            result.failure_detail,
        )
        self.assertTrue(result.cleanup_complete, result.failure_detail)
        self.assertTrue(result.containment_subreaper_restored)
        child_pid = int(pid_file.read_text(encoding="ascii"))
        self.assertTrue(_wait_pid_absent(child_pid))

    def test_descendant_identity_error_never_signals_reused_identity_and_leaves_no_zombie(self) -> None:
        identity = process._read_proc_identity(os.getpid())
        self.assertIsNotNone(identity)
        changed = dataclasses.replace(
            identity,
            start_time_ticks=identity.start_time_ticks + 1,
        )
        with mock.patch.object(process, "_read_proc_identity", return_value=changed):
            with self.assertRaises(process.ProcessError) as mismatch_context:
                process._open_verified_pidfd(identity)
        self.assertEqual(
            mismatch_context.exception.code,
            process.ProcessErrorCode.CONTAINMENT_FAILED,
        )

        pid_directory = self.root / "identity-error-nested"
        pid_directory.mkdir()
        pid_file = pid_directory / "child-0.pid"
        original_open = process._open_verified_pidfd
        injected = False

        def fail_first_descendant(observed_identity):
            nonlocal injected
            if pid_file.exists() and not injected:
                injected = True
                raise process.ProcessError(
                    process.ProcessErrorCode.CONTAINMENT_FAILED,
                    "pidfd-verify-descendant",
                    "injected stable-identity mismatch",
                )
            return original_open(observed_identity)

        with mock.patch.object(process, "_open_verified_pidfd", fail_first_descendant):
            result = self.run_child(
                "leader-running",
                "5",
                str(pid_directory),
                "1",
                "--new-session",
                "--close-pipes",
                limits=self.limits(wall_seconds=0.08, cleanup_seconds=1.0),
                safety=self.safety(poll_interval_seconds=0.005),
            )
        self.assertTrue(injected)
        self.assertEqual(
            result.failure_code,
            process.ProcessErrorCode.WALL_TIME_LIMIT,
            result.failure_detail,
        )
        self.assertFalse(result.cleanup_complete)
        self.assertEqual(
            result.cleanup_failure_code,
            process.ProcessErrorCode.CONTAINMENT_FAILED,
        )
        self.assertFalse(result.containment_subreaper_restored)
        child_pid = int(pid_file.read_text(encoding="ascii"))
        self.assertTrue(_wait_pid_absent(child_pid))
        process._reset_containment_poison_for_tests()

    def test_cancellation_cleans_child_before_reraising_baseexception(self) -> None:
        pid_directory = self.root / "cancelled-nested"
        pid_directory.mkdir()
        pid_file = pid_directory / "child-0.pid"
        original_sample = process._sample_safety

        def cancel_after_spawn(policy: process.SafetySpec) -> process.SafetySample:
            if pid_file.exists():
                raise TestCancellation("cancel after observed spawn")
            return original_sample(policy)

        with mock.patch.object(process, "_sample_safety", cancel_after_spawn):
            with self.assertRaisesRegex(TestCancellation, "observed spawn"):
                self.run_child(
                    "leader-running",
                    "5",
                    str(pid_directory),
                    "1",
                    "--new-session",
                    "--close-pipes",
                    limits=self.limits(cleanup_seconds=1.0),
                    safety=self.safety(poll_interval_seconds=0.005),
                )
        child_pid = int(pid_file.read_text(encoding="ascii"))
        self.assertTrue(_wait_pid_absent(child_pid))
        self.assertEqual(self.stdout.read_bytes(), b"")
        self.assertEqual(self.stderr.read_bytes(), b"")

    def test_cancellation_before_leader_identity_contains_nested_setsid_subtree(self) -> None:
        pid_directory = self.root / "cancel-before-leader-identity"
        pid_directory.mkdir()
        nested_pid_file = pid_directory / "child-0.pid"
        real_popen = process.subprocess.Popen
        real_read_identity = process._read_proc_identity
        real_scan_descendants = process._scan_descendants
        leader_pid: int | None = None
        injected = TestCancellation("cancel before leader identity capture")
        cancellation_raised = False
        owner_scan_populations: list[tuple[int, ...]] = []

        def remember_spawn(*args, **kwargs):
            nonlocal leader_pid
            spawned = real_popen(*args, **kwargs)
            leader_pid = spawned.pid
            self.fixture_pids.add(spawned.pid)
            return spawned

        def cancel_at_leader_identity(pid: int):
            nonlocal cancellation_raised
            if pid == leader_pid and not cancellation_raised:
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline and not nested_pid_file.exists():
                    time.sleep(0.002)
                if not nested_pid_file.exists():
                    raise AssertionError("nested setsid fixture was not created")
                cancellation_raised = True
                raise injected
            return real_read_identity(pid)

        def record_owner_scans(owner_pid, leader, max_entries, deadline, *, deadline_code):
            descendants = real_scan_descendants(
                owner_pid,
                leader,
                max_entries,
                deadline,
                deadline_code=deadline_code,
            )
            if leader is None:
                owner_scan_populations.append(
                    tuple(sorted(identity.pid for identity in descendants))
                )
            return descendants

        prior_subreaper = process._get_child_subreaper()
        try:
            with mock.patch.object(
                process.subprocess, "Popen", remember_spawn
            ), mock.patch.object(
                process, "_read_proc_identity", cancel_at_leader_identity
            ), mock.patch.object(
                process, "_scan_descendants", record_owner_scans
            ):
                with self.assertRaises(TestCancellation) as context:
                    self.run_child(
                        "leader-running",
                        "5",
                        str(pid_directory),
                        "1",
                        "--new-session",
                        "--close-pipes",
                        limits=self.limits(cleanup_seconds=1.0),
                        safety=self.safety(poll_interval_seconds=0.005),
                    )
            self.assertIs(context.exception, injected)
            self.assertTrue(cancellation_raised)
            self.assertIsNotNone(leader_pid)
            nested_pid = int(nested_pid_file.read_text(encoding="ascii"))
            self.fixture_pids.add(nested_pid)
            self.assertTrue(
                any(nested_pid in population for population in owner_scan_populations),
                owner_scan_populations,
            )
            self.assertGreaterEqual(len(owner_scan_populations), 3)
            self.assertEqual(owner_scan_populations[-2:], [(), ()])
            self.assertTrue(_wait_pid_absent(leader_pid), leader_pid)
            self.assertTrue(_wait_pid_absent(nested_pid), nested_pid)
            self.fixture_pids.discard(leader_pid)
            self.fixture_pids.discard(nested_pid)
            self.assertEqual(process._get_child_subreaper(), prior_subreaper)
            self.assertIsNone(process._CONTAINMENT_POISONED_DETAIL)
        finally:
            for pid in tuple(self.fixture_pids):
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass
                try:
                    os.waitpid(pid, 0)
                except ChildProcessError:
                    pass
                self.fixture_pids.discard(pid)
            if process._CONTAINMENT_POISONED_DETAIL is not None:
                process._reset_containment_poison_for_tests()

    def test_noncallable_supported_dispositions_are_rejected_before_spawn(self) -> None:
        prior_subreaper = process._get_child_subreaper()
        variants = (
            (signal.SIGTERM, signal.SIG_DFL),
            (signal.SIGTERM, signal.SIG_IGN),
            (signal.SIGHUP, signal.SIG_DFL),
            (signal.SIGHUP, signal.SIG_IGN),
        )
        for target, disposition in variants:
            label = f"{signal.Signals(target).name}-{int(disposition)}"
            with self.subTest(signal=target, disposition=disposition):
                self.stdout = self.root / f"noncallable-{label}.stdout"
                self.stderr = self.root / f"noncallable-{label}.stderr"
                prior_handler = signal.getsignal(target)
                signal.signal(target, disposition)
                try:
                    with mock.patch.object(
                        process.subprocess,
                        "Popen",
                        side_effect=AssertionError(
                            "spawn reached with a non-callable disposition"
                        ),
                    ):
                        with self.assertRaises(process.ProcessError) as context:
                            self.run_child("exit", "0")
                    self.assertEqual(
                        context.exception.code,
                        process.ProcessErrorCode.SIGNAL_DISPOSITION_PRECONDITION,
                    )
                    self.assertEqual(
                        context.exception.operation,
                        "signal-disposition-precondition",
                    )
                    self.assertFalse(self.stdout.exists())
                    self.assertFalse(self.stderr.exists())
                    self.assertEqual(_direct_child_pids_bounded(os.getpid()), ())
                    self.assertEqual(
                        process._get_child_subreaper(),
                        prior_subreaper,
                    )
                    self.assertIsNone(process._CONTAINMENT_POISONED_DETAIL)
                finally:
                    signal.signal(target, prior_handler)

    def test_exact_default_sigint_disposition_is_accepted_and_preserved(self) -> None:
        self.assertIs(
            signal.getsignal(signal.SIGINT),
            signal.default_int_handler,
        )
        result = self.run_child("exit", "0")
        self.assertIsNone(result.failure_code, result.failure_detail)
        self.assertIs(
            signal.getsignal(signal.SIGINT),
            signal.default_int_handler,
        )

    def test_terminal_style_systemexit_handlers_preserve_exact_exception(self) -> None:
        real_popen = process.subprocess.Popen
        prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        prior_subreaper = process._get_child_subreaper()
        for cancellation_signal in (signal.SIGTERM, signal.SIGHUP):
            label = signal.Signals(cancellation_signal).name.lower()
            with self.subTest(signal=cancellation_signal):
                self.stdout = self.root / f"system-exit-{label}.stdout"
                self.stderr = self.root / f"system-exit-{label}.stderr"
                pid_directory = self.root / f"system-exit-{label}-pids"
                pid_directory.mkdir()
                nested_pid_file = pid_directory / "child-0.pid"
                injected = SystemExit(128 + int(cancellation_signal))
                spawned: list[subprocess.Popen[bytes]] = []
                prior_handler = signal.getsignal(cancellation_signal)
                test_case = self

                def terminal_handler(signum, frame):
                    self.assertEqual(int(signum), int(cancellation_signal))
                    self.assertIsNone(frame)
                    raise injected

                class PendingSystemExitPopen(real_popen):
                    def _execute_child(self, *args, **kwargs):
                        super()._execute_child(*args, **kwargs)
                        spawned.append(self)
                        if self.pid is not None:
                            test_case.fixture_pids.add(self.pid)
                        deadline = time.monotonic() + 1.0
                        while (
                            time.monotonic() < deadline
                            and not nested_pid_file.exists()
                        ):
                            time.sleep(0.002)
                        if not nested_pid_file.exists():
                            raise AssertionError(
                                "nested setsid fixture was not created"
                            )
                        os.kill(os.getpid(), cancellation_signal)

                signal.signal(cancellation_signal, terminal_handler)
                try:
                    with mock.patch.object(
                        process.subprocess,
                        "Popen",
                        PendingSystemExitPopen,
                    ):
                        with self.assertRaises(SystemExit) as context:
                            self.run_child(
                                "leader-running",
                                "5",
                                str(pid_directory),
                                "1",
                                "--new-session",
                                "--close-pipes",
                                limits=self.limits(cleanup_seconds=1.0),
                                safety=self.safety(
                                    poll_interval_seconds=0.005
                                ),
                            )
                    self.assertIs(context.exception, injected)
                    self.assertTrue(spawned)
                    leader_pid = spawned[0].pid
                    nested_pid = int(
                        nested_pid_file.read_text(encoding="ascii")
                    )
                    self.fixture_pids.update((leader_pid, nested_pid))
                    self.assertTrue(_wait_pid_absent(leader_pid), leader_pid)
                    self.assertTrue(_wait_pid_absent(nested_pid), nested_pid)
                    self.assertEqual(_direct_child_pids_bounded(os.getpid()), ())
                    self.assertEqual(
                        process._get_child_subreaper(),
                        prior_subreaper,
                    )
                    self.assertIsNone(process._CONTAINMENT_POISONED_DETAIL)
                    self.assertEqual(
                        signal.pthread_sigmask(signal.SIG_BLOCK, set()),
                        prior_mask,
                    )
                    self.fixture_pids.discard(leader_pid)
                    self.fixture_pids.discard(nested_pid)
                finally:
                    signal.signal(cancellation_signal, prior_handler)
                    signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
                    for spawned_process in spawned:
                        if spawned_process.pid is None:
                            continue
                        try:
                            os.kill(spawned_process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        try:
                            spawned_process.wait(timeout=1.0)
                        except subprocess.TimeoutExpired:
                            pass
                        self.fixture_pids.discard(spawned_process.pid)
                    if nested_pid_file.exists():
                        self.fixture_pids.add(
                            int(nested_pid_file.read_text(encoding="ascii"))
                        )
                    for fixture_pid in tuple(self.fixture_pids):
                        try:
                            os.kill(fixture_pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        try:
                            os.waitpid(fixture_pid, 0)
                        except ChildProcessError:
                            pass
                        self.fixture_pids.discard(fixture_pid)
                    if process._CONTAINMENT_POISONED_DETAIL is not None:
                        process._reset_containment_poison_for_tests()

    def test_supported_disposition_identity_drift_is_rejected_before_fork(self) -> None:
        prior_subreaper = process._get_child_subreaper()
        real_block = process._block_supported_signals

        def captured_handler(signum, frame):
            raise AssertionError((signum, frame, "captured handler invoked"))

        def replacement_handler(signum, frame):
            raise AssertionError((signum, frame, "replacement handler invoked"))

        for phase in ("after-mask", "before-fork"):
            with self.subTest(phase=phase):
                self.stdout = self.root / f"disposition-drift-{phase}.stdout"
                self.stderr = self.root / f"disposition-drift-{phase}.stderr"
                signal.signal(signal.SIGTERM, captured_handler)
                thread_observations = 0

                def block_then_maybe_drift(contract, operation):
                    real_block(contract, operation)
                    if phase == "after-mask":
                        signal.signal(signal.SIGTERM, replacement_handler)

                def count_threads_then_maybe_drift(maximum):
                    nonlocal thread_observations
                    del maximum
                    thread_observations += 1
                    if phase == "before-fork" and thread_observations == 2:
                        signal.signal(signal.SIGTERM, replacement_handler)
                    return 1

                with mock.patch.object(
                    process,
                    "_block_supported_signals",
                    block_then_maybe_drift,
                ), mock.patch.object(
                    process,
                    "_kernel_thread_count",
                    count_threads_then_maybe_drift,
                ), mock.patch.object(
                    process.subprocess,
                    "Popen",
                    side_effect=AssertionError(
                        "spawn reached after signal-disposition drift"
                    ),
                ):
                    with self.assertRaises(process.ProcessError) as context:
                        self.run_child("exit", "0")
                self.assertEqual(
                    context.exception.code,
                    process.ProcessErrorCode.SIGNAL_DISPOSITION_DRIFT,
                )
                self.assertEqual(
                    context.exception.operation,
                    f"signal-disposition-{phase}",
                )
                self.assertIs(
                    signal.getsignal(signal.SIGTERM),
                    replacement_handler,
                )
                self.assertTrue(self.stdout.exists())
                self.assertTrue(self.stderr.exists())
                self.assertEqual(_direct_child_pids_bounded(os.getpid()), ())
                self.assertEqual(
                    process._get_child_subreaper(),
                    prior_subreaper,
                )
                self.assertIsNone(process._CONTAINMENT_POISONED_DETAIL)
                signal.signal(signal.SIGTERM, _terminal_style_test_handler)

    def test_hard_exit_handler_documents_unwind_bypass_boundary(self) -> None:
        """A trusted callable using os._exit is outside containment claims."""

        leader_pid_file = self.root / "hard-exit-leader.pid"
        controller_pid = os.fork()
        if controller_pid == 0:
            real_popen = process.subprocess.Popen

            def hard_exit_handler(signum, frame):
                del signum, frame
                os._exit(77)

            class HardExitPopen(real_popen):
                def _execute_child(self, *args, **kwargs):
                    super()._execute_child(*args, **kwargs)
                    leader_pid_file.write_text(
                        str(self.pid),
                        encoding="ascii",
                    )
                    os.kill(os.getpid(), signal.SIGTERM)

            signal.signal(signal.SIGHUP, _terminal_style_test_handler)
            signal.signal(signal.SIGINT, signal.default_int_handler)
            signal.signal(signal.SIGTERM, hard_exit_handler)
            try:
                with mock.patch.object(
                    process.subprocess,
                    "Popen",
                    HardExitPopen,
                ):
                    self.run_child("sleep", "5")
            except BaseException:
                os._exit(91)
            os._exit(92)

        self.fixture_pids.add(controller_pid)
        _, controller_status = os.waitpid(controller_pid, 0)
        self.fixture_pids.discard(controller_pid)
        leader_pid: int | None = None
        if leader_pid_file.exists():
            leader_pid = int(leader_pid_file.read_text(encoding="ascii"))
            self.fixture_pids.add(leader_pid)
        leader_survived_unwind_bypass = bool(
            leader_pid is not None
            and (Path("/proc") / str(leader_pid) / "stat").exists()
        )
        if leader_pid is not None:
            try:
                os.kill(leader_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.assertTrue(_wait_pid_absent(leader_pid), leader_pid)
            self.fixture_pids.discard(leader_pid)
        self.assertTrue(os.WIFEXITED(controller_status), controller_status)
        self.assertEqual(os.WEXITSTATUS(controller_status), 77)
        self.assertTrue(leader_survived_unwind_bypass)
        self.assertEqual(_direct_child_pids_bounded(os.getpid()), ())

    def test_supported_signal_or_constructor_cancellation_after_fork_is_contained(self) -> None:
        real_popen = process.subprocess.Popen
        supported = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
        prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        self.assertFalse(supported.intersection(prior_mask), prior_mask)

        variants = (
            ("sighup", signal.SIGHUP),
            ("sigint", signal.SIGINT),
            ("sigterm", signal.SIGTERM),
            ("deterministic", None),
        )
        for trigger, cancellation_signal in variants:
            with self.subTest(trigger=trigger):
                self.stdout = self.root / f"spawn-window-{trigger}.stdout"
                self.stderr = self.root / f"spawn-window-{trigger}.stderr"
                pid_directory = self.root / f"spawn-window-{trigger}-pids"
                pid_directory.mkdir()
                nested_pid_file = pid_directory / "child-0.pid"
                injected = TestCancellation(f"spawn-window-{trigger}")
                spawned: list[subprocess.Popen[bytes]] = []
                descriptor_roles: dict[int, str] = {}
                fsync_roles: list[str] = []
                closed: set[int] = set()
                real_open = process.os.open
                real_fsync = process.os.fsync
                real_close = process.os.close
                prior_handler = (
                    signal.getsignal(cancellation_signal)
                    if cancellation_signal is not None
                    else None
                )
                test_case = self

                def cancellation_handler(signum, frame):
                    self.assertEqual(signum, cancellation_signal)
                    raise injected

                class InterruptingPopen(real_popen):
                    def _execute_child(self, *args, **kwargs):
                        super()._execute_child(*args, **kwargs)
                        spawned.append(self)
                        self_pid = self.pid
                        if self_pid is not None:
                            test_case.fixture_pids.add(self_pid)
                        deadline = time.monotonic() + 1.0
                        while (
                            time.monotonic() < deadline
                            and not nested_pid_file.exists()
                        ):
                            time.sleep(0.002)
                        if not nested_pid_file.exists():
                            raise AssertionError("nested setsid fixture was not created")
                        if cancellation_signal is not None:
                            os.kill(os.getpid(), cancellation_signal)
                            return
                        raise injected

                def tracking_open(path, flags, mode=0o777, *args, **kwargs):
                    descriptor = real_open(path, flags, mode, *args, **kwargs)
                    if flags & os.O_DIRECTORY and Path(path) == self.root:
                        descriptor_roles[descriptor] = "parent"
                    elif flags & os.O_WRONLY and Path(path).name == self.stdout.name:
                        descriptor_roles[descriptor] = "stdout"
                    elif flags & os.O_WRONLY and Path(path).name == self.stderr.name:
                        descriptor_roles[descriptor] = "stderr"
                    return descriptor

                def tracking_fsync(descriptor: int) -> None:
                    if descriptor in descriptor_roles:
                        fsync_roles.append(descriptor_roles[descriptor])
                    real_fsync(descriptor)

                def tracking_close(descriptor: int) -> None:
                    if descriptor in descriptor_roles:
                        closed.add(descriptor)
                    real_close(descriptor)

                if cancellation_signal is not None:
                    signal.signal(cancellation_signal, cancellation_handler)
                try:
                    with mock.patch.object(
                        process.subprocess, "Popen", InterruptingPopen
                    ), mock.patch.object(
                        process.os, "open", tracking_open
                    ), mock.patch.object(
                        process.os, "fsync", tracking_fsync
                    ), mock.patch.object(
                        process.os, "close", tracking_close
                    ):
                        with self.assertRaises(TestCancellation) as context:
                            self.run_child(
                                "leader-running",
                                "5",
                                str(pid_directory),
                                "1",
                                "--new-session",
                                "--close-pipes",
                                limits=self.limits(cleanup_seconds=1.0),
                                safety=self.safety(poll_interval_seconds=0.005),
                            )
                    self.assertIs(context.exception, injected)
                    self.assertEqual(
                        signal.pthread_sigmask(signal.SIG_BLOCK, set()),
                        prior_mask,
                    )
                    self.assertTrue(spawned)
                    leader_pid = spawned[0].pid
                    nested_pid = int(nested_pid_file.read_text(encoding="ascii"))
                    self.fixture_pids.update((leader_pid, nested_pid))
                    self.assertTrue(_wait_pid_absent(leader_pid), leader_pid)
                    self.assertTrue(_wait_pid_absent(nested_pid), nested_pid)
                    self.assertEqual(_direct_child_pids_bounded(os.getpid()), ())
                    self.assertIsNone(process._CONTAINMENT_POISONED_DETAIL)
                    self.assertEqual(fsync_roles.count("stdout"), 2)
                    self.assertEqual(fsync_roles.count("stderr"), 2)
                    self.assertEqual(fsync_roles.count("parent"), 2)
                    self.assertEqual(set(descriptor_roles), closed)
                    for descriptor in descriptor_roles:
                        with self.assertRaises(OSError):
                            os.fstat(descriptor)
                    self.fixture_pids.discard(leader_pid)
                    self.fixture_pids.discard(nested_pid)
                finally:
                    if cancellation_signal is not None:
                        signal.signal(cancellation_signal, prior_handler)
                    signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
                    for spawned_process in spawned:
                        if spawned_process.pid is None:
                            continue
                        try:
                            os.kill(spawned_process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        try:
                            spawned_process.wait(timeout=1.0)
                        except subprocess.TimeoutExpired:
                            pass
                        self.fixture_pids.discard(spawned_process.pid)
                    if nested_pid_file.exists():
                        nested_pid = int(nested_pid_file.read_text(encoding="ascii"))
                        try:
                            os.kill(nested_pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        self.fixture_pids.add(nested_pid)
                    for fixture_pid in tuple(self.fixture_pids):
                        try:
                            os.waitpid(fixture_pid, 0)
                        except ChildProcessError:
                            pass
                        self.fixture_pids.discard(fixture_pid)
                    if process._CONTAINMENT_POISONED_DETAIL is not None:
                        process._reset_containment_poison_for_tests()

    def test_reaped_leader_numeric_pid_reuse_is_signalled_and_reaped_by_identity(self) -> None:
        owner_pid = os.getpid()
        reused_pid = 987654
        original_start = 100
        reused_start = 200
        live = process._ProcIdentity(
            pid=reused_pid,
            parent_pid=owner_pid,
            state="R",
            process_group=reused_pid,
            session=reused_pid,
            start_time_ticks=reused_start,
        )
        zombie = dataclasses.replace(live, state="Z")

        class ReapedLeader:
            pid = reused_pid
            returncode = None

            def kill(self) -> None:
                return None

            def wait(self, timeout=None) -> int:
                self.returncode = -signal.SIGKILL
                return self.returncode

        fake_process = ReapedLeader()
        lease = process._ContainmentLease(
            owner_pid=owner_pid,
            caller_thread_count=1,
            preexisting_child_count=0,
            previous_subreaper=False,
        )
        populations = iter(([live], [zombie], [], []))

        def scan(*args, **kwargs):
            return list(next(populations))

        def read_identity(pid: int):
            self.assertEqual(pid, reused_pid)
            return zombie

        waited = types.SimpleNamespace(si_pid=reused_pid)
        with mock.patch.object(
            process, "_scan_descendants", side_effect=scan
        ), mock.patch.object(
            process, "_kill_group"
        ), mock.patch.object(
            process, "_signal_identity"
        ) as signal_identity, mock.patch.object(
            process, "_read_proc_identity", side_effect=read_identity
        ), mock.patch.object(
            process.os, "waitid", return_value=waited
        ) as waitid:
            cleanup = process._cleanup_unidentified_leader(
                fake_process,
                lease,
                limits=self.limits(cleanup_seconds=1.0),
                scan_cadence=process._ScanCadence(),
            )
        self.assertTrue(cleanup.complete, cleanup.detail)
        signal_identity.assert_called_once_with(live)
        waitid.assert_called_once_with(
            os.P_PID,
            reused_pid,
            os.WEXITED | os.WNOHANG,
        )
        self.assertNotEqual(original_start, reused_start)

    def test_multiple_pending_signals_before_popen_return_select_one_primary(self) -> None:
        real_popen = process.subprocess.Popen
        supported = {signal.SIGHUP, signal.SIGINT, signal.SIGTERM}
        prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        self.assertFalse(supported.intersection(prior_mask), prior_mask)
        prior_subreaper = process._get_child_subreaper()
        prior_hup = signal.getsignal(signal.SIGHUP)
        prior_term = signal.getsignal(signal.SIGTERM)
        pid_directory = self.root / "multi-pending-spawn-window"
        pid_directory.mkdir()
        nested_pid_file = pid_directory / "child-0.pid"
        hup_error = TestCancellation("coalesced SIGHUP must not run")
        term_error = TestCancellation("selected SIGTERM primary")
        handler_calls: list[int] = []
        pending_at_send: tuple[int, ...] = ()
        pending_at_primary: tuple[int, ...] | None = None
        spawned: list[subprocess.Popen[bytes]] = []
        test_case = self

        def hup_handler(signum, frame):
            handler_calls.append(int(signum))
            raise hup_error

        def term_handler(signum, frame):
            nonlocal pending_at_primary
            handler_calls.append(int(signum))
            pending_at_primary = tuple(
                sorted(int(item) for item in signal.sigpending() if item in supported)
            )
            raise term_error

        class MultiPendingPopen(real_popen):
            def _execute_child(self, *args, **kwargs):
                nonlocal pending_at_send
                super()._execute_child(*args, **kwargs)
                spawned.append(self)
                if self.pid is not None:
                    test_case.fixture_pids.add(self.pid)
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline and not nested_pid_file.exists():
                    time.sleep(0.002)
                if not nested_pid_file.exists():
                    raise AssertionError("nested setsid fixture was not created")
                os.kill(os.getpid(), signal.SIGHUP)
                os.kill(os.getpid(), signal.SIGTERM)
                pending_at_send = tuple(
                    sorted(
                        int(item)
                        for item in signal.sigpending()
                        if item in supported
                    )
                )

        signal.signal(signal.SIGHUP, hup_handler)
        signal.signal(signal.SIGTERM, term_handler)
        try:
            with mock.patch.object(
                process.subprocess,
                "Popen",
                MultiPendingPopen,
            ):
                with self.assertRaises(TestCancellation) as context:
                    self.run_child(
                        "leader-running",
                        "5",
                        str(pid_directory),
                        "1",
                        "--new-session",
                        "--close-pipes",
                        limits=self.limits(cleanup_seconds=1.0),
                        safety=self.safety(poll_interval_seconds=0.005),
                    )
            self.assertIs(context.exception, term_error)
            self.assertEqual(
                pending_at_send,
                tuple(sorted((int(signal.SIGHUP), int(signal.SIGTERM)))),
            )
            self.assertEqual(handler_calls, [int(signal.SIGTERM)])
            self.assertEqual(pending_at_primary, ())
            expected_note = (
                "supported-signal-drain "
                "boundary=signal-mask-after-spawn-registration "
                f"priority=SIGINT({int(signal.SIGINT)})>"
                f"SIGTERM({int(signal.SIGTERM)})>SIGHUP({int(signal.SIGHUP)}) "
                f"observed=SIGHUP({int(signal.SIGHUP)}),"
                f"SIGTERM({int(signal.SIGTERM)}) "
                f"selected=SIGTERM({int(signal.SIGTERM)}) "
                f"coalesced=SIGHUP({int(signal.SIGHUP)})"
            )
            self.assertIn(
                expected_note,
                getattr(context.exception, "__notes__", ()),
            )
            self.assertFalse(supported.intersection(signal.sigpending()))
            self.assertTrue(spawned)
            leader_pid = spawned[0].pid
            nested_pid = int(nested_pid_file.read_text(encoding="ascii"))
            self.fixture_pids.update((leader_pid, nested_pid))
            self.assertTrue(_wait_pid_absent(leader_pid), leader_pid)
            self.assertTrue(_wait_pid_absent(nested_pid), nested_pid)
            self.assertEqual(_direct_child_pids_bounded(os.getpid()), ())
            self.assertEqual(process._get_child_subreaper(), prior_subreaper)
            self.assertIsNone(process._CONTAINMENT_POISONED_DETAIL)
            self.assertEqual(
                signal.pthread_sigmask(signal.SIG_BLOCK, set()),
                prior_mask,
            )
            self.fixture_pids.discard(leader_pid)
            self.fixture_pids.discard(nested_pid)
        finally:
            signal.signal(signal.SIGHUP, prior_hup)
            signal.signal(signal.SIGTERM, prior_term)
            signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
            for spawned_process in spawned:
                if spawned_process.pid is None:
                    continue
                try:
                    os.kill(spawned_process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    spawned_process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
                self.fixture_pids.discard(spawned_process.pid)
            if nested_pid_file.exists():
                self.fixture_pids.add(
                    int(nested_pid_file.read_text(encoding="ascii"))
                )
            for fixture_pid in tuple(self.fixture_pids):
                try:
                    os.kill(fixture_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    os.waitpid(fixture_pid, 0)
                except ChildProcessError:
                    pass
                self.fixture_pids.discard(fixture_pid)
            if process._CONTAINMENT_POISONED_DETAIL is not None:
                process._reset_containment_poison_for_tests()

    def test_primary_signal_coalesces_both_later_pending_signals_during_cleanup(self) -> None:
        pid_directory = self.root / "second-supported-signal"
        pid_directory.mkdir()
        nested_pid_file = pid_directory / "child-0.pid"
        first = TestCancellation("first SIGINT cancellation")
        later_hup = TestCancellation("coalesced SIGHUP must not run")
        later_term = TestCancellation("coalesced SIGTERM must not run")
        first_delivered = False
        later_handler_calls: list[int] = []
        later_sent = False
        pending_at_send: tuple[int, ...] = ()
        prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        prior_hup = signal.getsignal(signal.SIGHUP)
        prior_int = signal.getsignal(signal.SIGINT)
        prior_term = signal.getsignal(signal.SIGTERM)
        real_popen = process.subprocess.Popen
        real_sample = process._sample_safety
        real_scan = process._scan_descendants
        leaders: list[subprocess.Popen[bytes]] = []

        def remember_spawn(*args, **kwargs):
            spawned = real_popen(*args, **kwargs)
            leaders.append(spawned)
            self.fixture_pids.add(spawned.pid)
            return spawned

        def first_handler(signum, frame):
            nonlocal first_delivered
            first_delivered = True
            raise first

        def hup_handler(signum, frame):
            later_handler_calls.append(int(signum))
            raise later_hup

        def term_handler(signum, frame):
            later_handler_calls.append(int(signum))
            raise later_term

        def send_first(policy: process.SafetySpec) -> process.SafetySample:
            if nested_pid_file.exists() and not first_delivered:
                os.kill(os.getpid(), signal.SIGINT)
            return real_sample(policy)

        def send_later_during_cleanup(*args, **kwargs):
            nonlocal later_sent, pending_at_send
            if first_delivered and not later_sent:
                later_sent = True
                os.kill(os.getpid(), signal.SIGHUP)
                os.kill(os.getpid(), signal.SIGHUP)
                os.kill(os.getpid(), signal.SIGTERM)
                os.kill(os.getpid(), signal.SIGTERM)
                pending_at_send = tuple(
                    sorted(
                        int(item)
                        for item in signal.sigpending()
                        if item in {signal.SIGHUP, signal.SIGINT, signal.SIGTERM}
                    )
                )
            return real_scan(*args, **kwargs)

        signal.signal(signal.SIGHUP, hup_handler)
        signal.signal(signal.SIGINT, first_handler)
        signal.signal(signal.SIGTERM, term_handler)
        try:
            with mock.patch.object(
                process, "_sample_safety", send_first
            ), mock.patch.object(
                process, "_scan_descendants", send_later_during_cleanup
            ), mock.patch.object(
                process.subprocess, "Popen", remember_spawn
            ):
                with self.assertRaises(TestCancellation) as context:
                    self.run_child(
                        "leader-running",
                        "5",
                        str(pid_directory),
                        "1",
                        "--new-session",
                        "--close-pipes",
                        limits=self.limits(cleanup_seconds=1.0),
                        safety=self.safety(poll_interval_seconds=0.005),
                    )
            self.assertIs(context.exception, first)
            self.assertTrue(first_delivered)
            self.assertTrue(later_sent)
            self.assertEqual(
                pending_at_send,
                tuple(sorted((int(signal.SIGHUP), int(signal.SIGTERM)))),
            )
            self.assertEqual(later_handler_calls, [])
            notes = getattr(context.exception, "__notes__", ())
            expected_note = (
                "supported-signal-drain boundary=signal-mask-final-restore "
                f"priority=SIGINT({int(signal.SIGINT)})>"
                f"SIGTERM({int(signal.SIGTERM)})>SIGHUP({int(signal.SIGHUP)}) "
                f"observed=SIGHUP({int(signal.SIGHUP)}),"
                f"SIGTERM({int(signal.SIGTERM)}) "
                "selected=existing-primary "
                f"coalesced=SIGHUP({int(signal.SIGHUP)}),"
                f"SIGTERM({int(signal.SIGTERM)})"
            )
            self.assertIn(expected_note, notes)
            self.assertFalse(
                {signal.SIGHUP, signal.SIGINT, signal.SIGTERM}.intersection(
                    signal.sigpending()
                )
            )
            nested_pid = int(nested_pid_file.read_text(encoding="ascii"))
            self.fixture_pids.add(nested_pid)
            self.assertTrue(_wait_pid_absent(nested_pid), nested_pid)
            self.assertEqual(_direct_child_pids_bounded(os.getpid()), ())
            self.assertEqual(
                signal.pthread_sigmask(signal.SIG_BLOCK, set()),
                prior_mask,
            )
            self.assertIsNone(process._CONTAINMENT_POISONED_DETAIL)
            self.fixture_pids.discard(nested_pid)
        finally:
            signal.signal(signal.SIGHUP, prior_hup)
            signal.signal(signal.SIGINT, prior_int)
            signal.signal(signal.SIGTERM, prior_term)
            signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
            if nested_pid_file.exists():
                self.fixture_pids.add(
                    int(nested_pid_file.read_text(encoding="ascii"))
                )
            for fixture_pid in tuple(self.fixture_pids):
                try:
                    os.kill(fixture_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            for spawned in leaders:
                try:
                    spawned.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
                self.fixture_pids.discard(spawned.pid)
            for fixture_pid in tuple(self.fixture_pids):
                try:
                    os.waitpid(fixture_pid, 0)
                except ChildProcessError:
                    pass
                self.fixture_pids.discard(fixture_pid)
            if process._CONTAINMENT_POISONED_DETAIL is not None:
                process._reset_containment_poison_for_tests()

    def test_selected_supported_signal_handler_return_is_fail_closed(self) -> None:
        real_popen = process.subprocess.Popen
        prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        prior_subreaper = process._get_child_subreaper()
        prior_term = signal.getsignal(signal.SIGTERM)
        handler_calls = 0
        spawned: list[subprocess.Popen[bytes]] = []
        test_case = self

        def returning_handler(signum, frame):
            nonlocal handler_calls
            self.assertEqual(signum, signal.SIGTERM)
            handler_calls += 1

        class ReturningHandlerPopen(real_popen):
            def _execute_child(self, *args, **kwargs):
                super()._execute_child(*args, **kwargs)
                spawned.append(self)
                if self.pid is not None:
                    test_case.fixture_pids.add(self.pid)
                os.kill(os.getpid(), signal.SIGTERM)
                test_case.assertIn(signal.SIGTERM, signal.sigpending())

        signal.signal(signal.SIGTERM, returning_handler)
        try:
            with mock.patch.object(
                process.subprocess,
                "Popen",
                ReturningHandlerPopen,
            ):
                with self.assertRaises(process.ProcessError) as context:
                    self.run_child("sleep", "0.05")
            self.assertEqual(
                context.exception.code.value,
                "cancellation_handler_returned",
            )
            self.assertEqual(handler_calls, 1)
            expected_note = (
                "supported-signal-drain "
                "boundary=signal-mask-after-spawn-registration "
                f"priority=SIGINT({int(signal.SIGINT)})>"
                f"SIGTERM({int(signal.SIGTERM)})>SIGHUP({int(signal.SIGHUP)}) "
                f"observed=SIGTERM({int(signal.SIGTERM)}) "
                f"selected=SIGTERM({int(signal.SIGTERM)}) coalesced="
            )
            self.assertIn(
                expected_note,
                getattr(context.exception, "__notes__", ()),
            )
            self.assertFalse(
                {signal.SIGHUP, signal.SIGINT, signal.SIGTERM}.intersection(
                    signal.sigpending()
                )
            )
            self.assertEqual(_direct_child_pids_bounded(os.getpid()), ())
            self.assertEqual(process._get_child_subreaper(), prior_subreaper)
            self.assertIsNone(process._CONTAINMENT_POISONED_DETAIL)
            self.assertEqual(
                signal.pthread_sigmask(signal.SIG_BLOCK, set()),
                prior_mask,
            )
        finally:
            signal.signal(signal.SIGTERM, prior_term)
            signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
            for spawned_process in spawned:
                if spawned_process.pid is None:
                    continue
                try:
                    os.kill(spawned_process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    spawned_process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
                self.fixture_pids.discard(spawned_process.pid)
            for fixture_pid in tuple(self.fixture_pids):
                try:
                    os.kill(fixture_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    os.waitpid(fixture_pid, 0)
                except ChildProcessError:
                    pass
                self.fixture_pids.discard(fixture_pid)
            if process._CONTAINMENT_POISONED_DETAIL is not None:
                process._reset_containment_poison_for_tests()

    def test_supported_signal_drain_has_one_snapshot_linearization_boundary(self) -> None:
        contract = process._SignalContract(
            caller_mask_numbers=(),
            blocked=True,
        )
        selected = TestCancellation("selected one-snapshot SIGTERM")
        selected_handler = mock.Mock(side_effect=selected)
        contract.supported_signal_handlers = tuple(
            (item, selected_handler if item == int(signal.SIGTERM) else mock.Mock())
            for item in process.SUPPORTED_CANCELLATION_SIGNALS
        )
        pending_sets = iter(
            (
                (int(signal.SIGTERM),),
                (int(signal.SIGHUP),),
            )
        )

        with mock.patch.object(
            process,
            "_pending_supported_signals",
            side_effect=lambda operation: next(pending_sets),
        ) as pending, mock.patch.object(
            process,
            "_consume_pending_supported_signal",
        ) as consume, mock.patch.object(
            process.signal,
            "pthread_sigmask",
            return_value=set(),
        ), mock.patch.object(
            process,
            "_query_signal_mask",
            return_value=(),
        ), mock.patch.object(
            process.signal,
            "raise_signal",
            side_effect=AssertionError(
                "delivery re-read the mutable kernel disposition"
            ),
        ) as raise_signal:
            with self.assertRaises(TestCancellation) as context:
                process._restore_caller_signal_mask(
                    contract,
                    "test-one-snapshot-boundary",
                    primary_exists=False,
                )

        self.assertIs(context.exception, selected)
        pending.assert_called_once_with("test-one-snapshot-boundary")
        consume.assert_called_once_with(
            int(signal.SIGTERM),
            "test-one-snapshot-boundary",
        )
        selected_handler.assert_called_once_with(int(signal.SIGTERM), None)
        raise_signal.assert_not_called()
        self.assertEqual(
            contract.deferred_supported_signal_numbers,
            {int(signal.SIGTERM)},
        )
        expected_note = (
            "supported-signal-drain boundary=test-one-snapshot-boundary "
            f"priority=SIGINT({int(signal.SIGINT)})>"
            f"SIGTERM({int(signal.SIGTERM)})>SIGHUP({int(signal.SIGHUP)}) "
            f"observed=SIGTERM({int(signal.SIGTERM)}) "
            f"selected=SIGTERM({int(signal.SIGTERM)}) coalesced="
        )
        self.assertEqual(
            getattr(context.exception, "__notes__", ()),
            [expected_note],
        )
        self.assertNotIn(
            int(signal.SIGHUP),
            contract.deferred_supported_signal_numbers,
        )

    def test_child_exec_has_default_supported_dispositions_and_exact_caller_mask(self) -> None:
        supported = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
        prior_handlers = {item: signal.getsignal(item) for item in supported}
        prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})
        caller_mask = set(prior_mask) | {signal.SIGUSR1}
        self.assertFalse(set(supported).intersection(caller_mask), caller_mask)
        real_popen = process.subprocess.Popen
        preexec_values: list[object] = []

        def controller_handler(signum, frame):
            raise AssertionError(f"unexpected parent signal {signum}")

        def inspect_popen(*args, **kwargs):
            preexec_values.append(kwargs.get("preexec_fn"))
            return real_popen(*args, **kwargs)

        self.stdout = self.root / "exec-signal-state.stdout"
        self.stderr = self.root / "exec-signal-state.stderr"
        command = process.CommandSpec(
            argv=("/bin/cat", "/proc/self/status"),
            cwd=self.cwd,
            environment={"LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1"},
            stdout_path=self.stdout,
            stderr_path=self.stderr,
        )
        for item in supported:
            signal.signal(item, controller_handler)
        try:
            with mock.patch.object(process.subprocess, "Popen", inspect_popen):
                result = process.run_bounded_command(
                    command,
                    limits=self.limits(),
                    safety=self.safety(),
            )
            self.assertIsNone(result.failure_code, result.failure_detail)
            status_fields = {}
            for line in self.stdout.read_text(encoding="ascii").splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    status_fields[key] = value.strip()
            expected_mask = sum(1 << (int(item) - 1) for item in caller_mask)
            self.assertEqual(int(status_fields["SigBlk"], 16), expected_mask)
            caught = int(status_fields["SigCgt"], 16)
            ignored = int(status_fields["SigIgn"], 16)
            for item in supported:
                bit = 1 << (int(item) - 1)
                self.assertEqual(caught & bit, 0, (item, status_fields))
                self.assertEqual(ignored & bit, 0, (item, status_fields))
            self.assertEqual(
                signal.pthread_sigmask(signal.SIG_BLOCK, set()),
                caller_mask,
            )
            self.assertEqual(len(preexec_values), 1)
            self.assertTrue(callable(preexec_values[0]), preexec_values)
        finally:
            for item, handler in prior_handlers.items():
                signal.signal(item, handler)
            signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)

    def test_preblocked_supported_signal_is_rejected_before_spawn(self) -> None:
        prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        try:
            with self.assertRaises(process.ProcessError) as context:
                self.run_child("exit", "0")
            self.assertEqual(
                context.exception.code.value,
                "signal_mask_precondition",
            )
            self.assertFalse(self.stdout.exists())
            self.assertFalse(self.stderr.exists())
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)

    def test_one_thread_precondition_is_rechecked_immediately_before_fork(self) -> None:
        prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        prior_subreaper = process._get_child_subreaper()
        thread_counts = iter((1, 2))

        with mock.patch.object(
            process,
            "_kernel_thread_count",
            side_effect=lambda maximum: next(thread_counts),
        ), mock.patch.object(
            process.subprocess,
            "Popen",
            side_effect=AssertionError("spawn reached after pre-fork thread change"),
        ):
            with self.assertRaises(process.ProcessError) as context:
                self.run_child("exit", "0")

        self.assertEqual(
            context.exception.code,
            process.ProcessErrorCode.CALLER_MULTITHREADED,
        )
        self.assertEqual(context.exception.operation, "containment-pre-spawn")
        self.assertTrue(self.stdout.exists())
        self.assertTrue(self.stderr.exists())
        self.assertEqual(self.stdout.stat().st_size, 0)
        self.assertEqual(self.stderr.stat().st_size, 0)
        self.assertEqual(_direct_child_pids_bounded(os.getpid()), ())
        self.assertEqual(process._get_child_subreaper(), prior_subreaper)
        self.assertIsNone(process._CONTAINMENT_POISONED_DETAIL)
        self.assertEqual(
            signal.pthread_sigmask(signal.SIG_BLOCK, set()),
            prior_mask,
        )

    def test_signal_mask_block_failure_never_spawns_or_poisons_child_free_runner(self) -> None:
        real_pthread_sigmask = process.signal.pthread_sigmask
        owner_pid = os.getpid()
        prior_mask = real_pthread_sigmask(signal.SIG_BLOCK, set())
        injected_calls = 0

        def fail_supported_block(how, mask):
            nonlocal injected_calls
            numbers = {int(item) for item in mask}
            if (
                os.getpid() == owner_pid
                and how == signal.SIG_BLOCK
                and numbers == set(process.SUPPORTED_CANCELLATION_SIGNALS)
            ):
                injected_calls += 1
                raise OSError(5, "injected supported-signal block failure")
            return real_pthread_sigmask(how, mask)

        prior_subreaper = process._get_child_subreaper()
        with mock.patch.object(
            process.signal, "pthread_sigmask", fail_supported_block
        ), mock.patch.object(
            process.subprocess,
            "Popen",
            side_effect=AssertionError("spawn reached after signal-mask failure"),
        ):
            with self.assertRaises(process.ProcessError) as context:
                self.run_child("exit", "0")
        self.assertEqual(
            context.exception.code,
            process.ProcessErrorCode.SIGNAL_MASK_FAILED,
        )
        self.assertGreaterEqual(injected_calls, 2)
        self.assertEqual(_direct_child_pids_bounded(os.getpid()), ())
        self.assertEqual(process._get_child_subreaper(), prior_subreaper)
        self.assertIsNone(process._CONTAINMENT_POISONED_DETAIL)
        self.assertEqual(
            real_pthread_sigmask(signal.SIG_BLOCK, set()),
            prior_mask,
        )

    def test_signal_mask_final_restore_failure_is_typed_after_cleanup(self) -> None:
        real_pthread_sigmask = process.signal.pthread_sigmask
        owner_pid = os.getpid()
        prior_mask = real_pthread_sigmask(signal.SIG_BLOCK, set())
        parent_restore_calls = 0

        def fail_second_parent_restore(how, mask):
            nonlocal parent_restore_calls
            if os.getpid() == owner_pid and how == signal.SIG_SETMASK:
                parent_restore_calls += 1
                if parent_restore_calls == 2:
                    raise OSError(5, "injected final signal-mask restore failure")
            return real_pthread_sigmask(how, mask)

        prior_subreaper = process._get_child_subreaper()
        try:
            with mock.patch.object(
                process.signal,
                "pthread_sigmask",
                fail_second_parent_restore,
            ):
                with self.assertRaises(process.ProcessError) as context:
                    self.run_child("exit", "0")
            self.assertEqual(
                context.exception.code,
                process.ProcessErrorCode.SIGNAL_MASK_FAILED,
            )
            self.assertEqual(parent_restore_calls, 2)
            self.assertEqual(_direct_child_pids_bounded(os.getpid()), ())
            self.assertEqual(process._get_child_subreaper(), prior_subreaper)
            self.assertIsNone(process._CONTAINMENT_POISONED_DETAIL)
        finally:
            real_pthread_sigmask(signal.SIG_SETMASK, prior_mask)

    def test_child_preexec_failure_is_typed_and_leaves_no_child(self) -> None:
        real_popen = process.subprocess.Popen
        prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        preexec_seen = False

        def fail_child_preexec() -> None:
            raise RuntimeError("injected child preexec failure")

        def replace_preexec(*args, **kwargs):
            nonlocal preexec_seen
            preexec_seen = callable(kwargs.get("preexec_fn"))
            if not preexec_seen:
                raise AssertionError("runner did not provide child preexec_fn")
            kwargs["preexec_fn"] = fail_child_preexec
            return real_popen(*args, **kwargs)

        with mock.patch.object(process.subprocess, "Popen", replace_preexec):
            with self.assertRaises(process.ProcessError) as context:
                self.run_child("exit", "0")
        self.assertTrue(preexec_seen)
        self.assertEqual(context.exception.code, process.ProcessErrorCode.SPAWN_FAILED)
        self.assertEqual(_direct_child_pids_bounded(os.getpid()), ())
        self.assertEqual(
            signal.pthread_sigmask(signal.SIG_BLOCK, set()),
            prior_mask,
        )
        self.assertIsNone(process._CONTAINMENT_POISONED_DETAIL)

    def test_thermal_limit_during_execution_kills_and_preserves_primary_reason(self) -> None:
        pid_directory = self.root / "thermal-nested"
        pid_directory.mkdir()
        pid_file = pid_directory / "child-0.pid"
        original_sample = process._sample_safety

        def become_hot(policy: process.SafetySpec) -> process.SafetySample:
            if pid_file.exists():
                self.sensor.write_text("91000\n", encoding="ascii")
            return original_sample(policy)

        with mock.patch.object(process, "_sample_safety", become_hot):
            result = self.run_child(
                "leader-running",
                "5",
                str(pid_directory),
                "1",
                "--new-session",
                "--close-pipes",
                limits=self.limits(cleanup_seconds=1.0),
                safety=self.safety(ceiling=90000, poll_interval_seconds=0.005),
            )
        self.assert_failure(result, process.ProcessErrorCode.THERMAL_LIMIT)
        observed_temperatures = [
            temperature
            for sample in result.safety_samples
            for _, temperature in sample.thermal_millicelsius
        ]
        self.assertIn(42000, observed_temperatures)
        self.assertIn(91000, observed_temperatures)

    def test_safety_sample_limit_is_not_reported_as_success(self) -> None:
        result = self.run_child(
            "sleep",
            "5",
            "--ignore-term",
            limits=self.limits(cleanup_seconds=1.0),
            safety=self.safety(poll_interval_seconds=0.005, max_samples=1),
        )
        self.assert_failure(result, process.ProcessErrorCode.SAFETY_SAMPLE_LIMIT)
        self.assertEqual(len(result.safety_samples), 1)

    def test_invalid_specs_are_rejected_before_creating_evidence(self) -> None:
        with self.assertRaises(process.ProcessError) as signal_context:
            process._signal_mask_numbers((True,))
        self.assertEqual(
            signal_context.exception.code,
            process.ProcessErrorCode.SIGNAL_MASK_FAILED,
        )
        invalid_commands = (
            (
                process.CommandSpec(
                    argv=(),
                    cwd=self.cwd,
                    environment={},
                    stdout_path=self.stdout,
                    stderr_path=self.stderr,
                ),
                process.ProcessErrorCode.INVALID_ARGUMENT,
            ),
            (
                process.CommandSpec(
                    argv=(sys.executable, str(CHILD), "exit", "0"),
                    cwd=Path("relative"),
                    environment={},
                    stdout_path=self.stdout,
                    stderr_path=self.stderr,
                ),
                process.ProcessErrorCode.PATH_INVALID,
            ),
            (
                process.CommandSpec(
                    argv=(sys.executable, str(CHILD), "exit", "0"),
                    cwd=self.cwd,
                    environment={"BAD=KEY": "value"},
                    stdout_path=self.stdout,
                    stderr_path=self.stderr,
                ),
                process.ProcessErrorCode.INVALID_ARGUMENT,
            ),
        )
        for command, expected in invalid_commands:
            with self.subTest(command=command):
                with self.assertRaises(process.ProcessError) as context:
                    process.run_bounded_command(
                        command,
                        limits=self.limits(),
                        safety=self.safety(),
                    )
                self.assertEqual(context.exception.code, expected)
                self.assertFalse(self.stdout.exists())
                self.assertFalse(self.stderr.exists())


if __name__ == "__main__":
    unittest.main()
