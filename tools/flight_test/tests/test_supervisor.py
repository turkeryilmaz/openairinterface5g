#!/usr/bin/env python3
"""Real-process lifecycle checks for the UE-only recovery session."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch


TOOL_DIR = Path(__file__).resolve().parents[1]
WORKTREE = TOOL_DIR.parents[1]
CAPTURE = TOOL_DIR / "capture.py"
VALIDATION_ROOT = WORKTREE / "cmake_targets/log/FlightTests/Validation/recovery_implementation_2026-09-19"
sys.path.insert(0, str(TOOL_DIR))

import capture as capture_module
from capture import FlightCapture, RecoverySession, StopLatch, build_parser, validate_args


class RecoverySupervisorFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        VALIDATION_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)

    def temporary_directory(self) -> tempfile.TemporaryDirectory[str]:
        return tempfile.TemporaryDirectory(prefix="flight-supervisor-test-", dir=VALIDATION_ROOT)

    @staticmethod
    def wait_for(predicate, timeout: float, message: str) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.05)
        raise AssertionError(message)

    @staticmethod
    def session(output: Path) -> Path:
        sessions = sorted(output.glob("ue-session-*"))
        if len(sessions) != 1:
            raise AssertionError(f"expected one recovery session, found {len(sessions)}")
        return sessions[0]

    @staticmethod
    def status(session: Path) -> dict:
        return json.loads((session / "status.json").read_text(encoding="utf-8"))

    def fixture(self, root: Path, source: str) -> Path:
        path = root / "worker.py"
        path.write_text(textwrap.dedent(source), encoding="utf-8")
        return path

    def command(self, output: Path, worker: Path, *worker_args: Path | str, recovery: bool = True) -> list[str]:
        command = [
            sys.executable,
            "-B",
            str(CAPTURE),
            "--role",
            "ue",
            "--output",
            str(output),
            "--min-free-bytes",
            "0",
            "--startup-grace",
            "0",
            "--health-interval",
            "0.05",
            "--post-exit-drain-timeout",
            "0.2",
        ]
        if recovery:
            command.extend(["--recovery", "--recovery-stall", "1", "--recovery-attempt", "1"])
        command.extend(["--", sys.executable, "-B", str(worker), *(str(value) for value in worker_args)])
        return command

    @staticmethod
    def native_prelude() -> str:
        return textwrap.indent(
            textwrap.dedent("""\
            import json
            import os
            import socket
            import time

            channel = socket.socket(fileno=int(os.environ["_OAI_FLIGHT_MONITOR_FD"]))
            sequence = 0
            def native(values):
                global sequence
                sequence += 1
                channel.send(json.dumps({
                    "schema_version": 1,
                    "kind": "native_progress",
                    "pid": os.getpid(),
                    "sequence": sequence,
                    "mono_ns": time.monotonic_ns(),
                    "send_drops": 0,
                    "values": values,
                }).encode("utf-8"))
            """),
            "                ",
        )

    def test_retryable_cause_101_clean_exit_launches_second_worker(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            counter = root / "launches"
            worker = self.fixture(root, self.native_prelude() + f'''
                import pathlib
                counter = pathlib.Path({str(counter)!r})
                try:
                    launch = int(counter.read_text(encoding="ascii"))
                except FileNotFoundError:
                    launch = 0
                counter.write_text(str(launch + 1), encoding="ascii")
                values = {{"rx_samples": launch + 1}}
                if launch == 0:
                    values["nas_reject"] = (1 << 56) | (101 << 48) | (1 << 40) | (0x42 << 32) | 10
                native(values)
                time.sleep(0.1)
            ''')
            output = root / "output"
            completed = subprocess.run(self.command(output, worker), text=True, capture_output=True, timeout=18, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(counter.read_text(encoding="ascii"), "2")
            status = self.status(self.session(output))
            self.assertEqual(status["state"], "policy_stop")
            self.assertEqual(status["attempt_count"], 2)
            self.assertEqual(status["recent_attempts"][0]["policy_decision"]["action"], "retry")
            self.assertEqual(status["recent_attempts"][1]["policy_decision"]["reason"], "unclassified_zero_exit")

    def test_operator_stop_during_backoff_never_relaunches(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            counter = root / "launches"
            worker = self.fixture(root, self.native_prelude() + f'''
                import pathlib
                counter = pathlib.Path({str(counter)!r})
                counter.write_text("1", encoding="ascii")
                native({{"rx_samples": 1, "nas_reject": (1 << 56) | (101 << 48) | (1 << 40) | (0x42 << 32) | 10}})
                time.sleep(0.1)
            ''')
            output = root / "output"
            process = subprocess.Popen(self.command(output, worker), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                def in_backoff() -> bool:
                    sessions = list(output.glob("ue-session-*")) if output.exists() else []
                    return bool(sessions) and self.status(sessions[0])["state"] == "backoff"

                self.wait_for(in_backoff, 6, "session did not reach retry backoff")
                process.send_signal(signal.SIGTERM)
                stdout, stderr = process.communicate(timeout=12)
                self.assertEqual(process.returncode, 128 + signal.SIGTERM, stdout + stderr)
                status = self.status(self.session(output))
                self.assertEqual(status["state"], "operator_stop")
                self.assertEqual(status["operator_stop_signal"], signal.SIGTERM)
                self.assertEqual(status["attempt_count"], 1)
                self.assertEqual(counter.read_text(encoding="ascii"), "1")
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    process.communicate(timeout=12)

    def test_permanent_and_unknown_native_restrictions_stop(self) -> None:
        for policy in (2, 3):
            with self.subTest(policy=policy), self.temporary_directory() as temporary:
                root = Path(temporary)
                counter = root / "launches"
                worker = self.fixture(root, self.native_prelude() + f'''
                import pathlib
                counter = pathlib.Path({str(counter)!r})
                counter.write_text("1", encoding="ascii")
                native({{"rx_samples": 1, "nas_reject": (1 << 56) | (101 << 48) | ({policy} << 40)}})
                time.sleep(0.1)
                ''')
                output = root / "output"
                completed = subprocess.run(self.command(output, worker), text=True, capture_output=True, timeout=8, check=False)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                status = self.status(self.session(output))
                self.assertEqual(status["state"], "policy_stop")
                self.assertEqual(status["attempt_count"], 1)
                self.assertEqual(counter.read_text(encoding="ascii"), "1")
                decision = status["recent_attempts"][0]["policy_decision"]
                self.assertEqual(decision["action"], "stop")
                self.assertEqual(decision["reason"], f"nas_reject_101_policy_{policy}")

    def test_live_native_input_with_async_completion_stall_recovers(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            counter = root / "launches"
            stopped = root / "stopped"
            worker = self.fixture(root, self.native_prelude() + f'''
                import pathlib
                import signal
                counter = pathlib.Path({str(counter)!r})
                stopped = pathlib.Path({str(stopped)!r})
                try:
                    launch = int(counter.read_text(encoding="ascii"))
                except FileNotFoundError:
                    launch = 0
                counter.write_text(str(launch + 1), encoding="ascii")
                if launch:
                    native({{"rx_samples": 2}})
                    time.sleep(0.1)
                else:
                    def finish(_signum, _frame):
                        stopped.write_text("sigint", encoding="ascii")
                        raise SystemExit(0)
                    signal.signal(signal.SIGINT, finish)
                    for sample in range(80):
                        native({{"rx_samples": sample + 1, "ue_slot_inputs": sample + 1, "ue_dl_completed": 1, "ue_tx_completed": 1}})
                        time.sleep(0.05)
            ''')
            output = root / "output"
            completed = subprocess.run(self.command(output, worker), text=True, capture_output=True, timeout=14, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(counter.read_text(encoding="ascii"), "2")
            self.assertEqual(stopped.read_text(encoding="ascii"), "sigint")
            status = self.status(self.session(output))
            self.assertEqual(status["attempt_count"], 2)
            self.assertEqual(status["recent_attempts"][0]["recovery_stop_reason"], "ue_dl_completed_progress_stalled")
            self.assertEqual(status["recent_attempts"][0]["policy_decision"]["action"], "retry")

    def test_descendant_in_owned_radio_group_is_fenced_before_retry(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            counter = root / "launches"
            child_stopped = root / "child-stopped"
            worker = self.fixture(root, self.native_prelude() + f'''
                import pathlib
                import subprocess
                counter = pathlib.Path({str(counter)!r})
                child_stopped = pathlib.Path({str(child_stopped)!r})
                try:
                    launch = int(counter.read_text(encoding="ascii"))
                except FileNotFoundError:
                    launch = 0
                counter.write_text(str(launch + 1), encoding="ascii")
                if launch:
                    native({{"rx_samples": 2}})
                    time.sleep(0.1)
                else:
                    child = "import pathlib, signal, time; marker=pathlib.Path(" + repr(str(child_stopped)) + "); signal.signal(signal.SIGINT, lambda _s, _f: (marker.write_text('stopped', encoding='ascii'), (_ for _ in ()).throw(SystemExit(0)))); time.sleep(30)"
                    subprocess.Popen([os.environ.get("PYTHON", "{sys.executable}"), "-c", child], close_fds=False)
                    native({{"rx_samples": 1, "nas_reject": (1 << 56) | (101 << 48) | (1 << 40) | (0x42 << 32) | 10}})
                    time.sleep(0.1)
            ''')
            output = root / "output"
            completed = subprocess.run(self.command(output, worker), text=True, capture_output=True, timeout=20, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(counter.read_text(encoding="ascii"), "2")
            self.assertEqual(child_stopped.read_text(encoding="ascii"), "stopped")
            status = self.status(self.session(output))
            first = status["recent_attempts"][0]
            self.assertIn(first["group_fence_state"], {"fenced", "fenced_after_kill"})
            self.assertEqual(status["attempt_count"], 2)

    def test_observation_only_log_run_never_relaunches(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            counter = root / "launches"
            worker = self.fixture(root, self.native_prelude() + f'''
                import pathlib
                counter = pathlib.Path({str(counter)!r})
                counter.write_text("1", encoding="ascii")
                native({{"rx_samples": 1, "nas_reject": (1 << 56) | (101 << 48) | (1 << 40) | (0x42 << 32) | 10}})
                time.sleep(0.1)
            ''')
            output = root / "output"
            completed = subprocess.run(self.command(output, worker, recovery=False), text=True, capture_output=True, timeout=8, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(counter.read_text(encoding="ascii"), "1")
            runs = [path for path in output.iterdir() if path.is_dir()]
            self.assertEqual(len(runs), 1)
            self.assertTrue(runs[0].name.startswith("ue-"))
            status = json.loads((runs[0] / "status.json").read_text(encoding="utf-8"))
            self.assertFalse(status["recovery"]["enabled"])
            self.assertEqual(status["recovery"]["policy_decision"]["action"], "retry")


    def test_latched_prelaunch_stop_never_calls_popen(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            run_dir = root / "attempt"
            run_dir.mkdir(mode=0o700)
            parser = build_parser()
            args = parser.parse_args([
                "--role", "ue", "--recovery", "--output", str(root / "output"), "--min-free-bytes", "0",
                "--", sys.executable, "-c", "raise SystemExit(0)",
            ])
            command = validate_args(args, parser)
            latch = StopLatch()
            latch.signal_number = signal.SIGTERM
            capture = FlightCapture(
                args,
                command,
                run_dir=run_dir,
                stop_latch=latch,
                recovery_enabled=True,
            )
            with patch.object(capture_module.subprocess, "Popen") as popen:
                self.assertEqual(capture.run(), 128 + signal.SIGTERM)
                popen.assert_not_called()
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "operator_stop_before_launch")
            self.assertEqual(status["recovery"]["policy_decision"]["reason"], "operator_stop_before_launch")

    def test_unknown_cleanup_blocks_signal_and_forced_kill_follows_ten_seconds(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--role", "ue", "--recovery", "--output", "/tmp/flight-supervisor-status", "--min-free-bytes", "0",
            "--", sys.executable, "-c", "raise SystemExit(0)",
        ])
        command = validate_args(args, parser)
        session = RecoverySession(args, command)
        fake_capture = SimpleNamespace(process=SimpleNamespace(pid=4242), process_start_ticks=1)
        with patch.object(capture_module.os, "killpg", side_effect=PermissionError):
            self.assertEqual(session._fence_owned_group(fake_capture), "ownership_unknown")

        with (
            patch.object(session, "_owned_group_state", side_effect=["owned"] * 12),
            patch.object(capture_module.os, "killpg") as killpg,
            patch.object(capture_module.time, "monotonic_ns", side_effect=[0, 10_000_000_000, 10_000_000_000]),
            patch.object(capture_module.time, "sleep"),
        ):
            self.assertEqual(session._fence_owned_group(fake_capture), "cleanup_failed")
        self.assertEqual(
            killpg.call_args_list,
            [
                ((4242, signal.SIGINT),),
                ((4242, signal.SIGKILL),),
            ],
        )


if __name__ == "__main__":
    unittest.main()
