#!/usr/bin/env python3
"""Deterministic stdlib fixtures for the flight capture tool."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parents[1]
WORKTREE = TOOL_DIR.parents[1]
CAPTURE = TOOL_DIR / "capture.py"
VALIDATION_ROOT = (
    WORKTREE
    / "cmake_targets/log/FlightTests/Validation/capture"
)
sys.path.insert(0, str(TOOL_DIR))

from flight_health import HealthInput, HealthStateMachine, SystemHealthCollector
from capture import redact_argv


class CaptureFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        VALIDATION_ROOT.mkdir(parents=True, exist_ok=True)

    def temporary_directory(self) -> tempfile.TemporaryDirectory[str]:
        return tempfile.TemporaryDirectory(prefix="flight-capture-test-", dir=VALIDATION_ROOT)

    @staticmethod
    def run_directory(output: Path) -> Path:
        runs = [path for path in output.iterdir() if path.is_dir()]
        if len(runs) != 1:
            raise AssertionError(f"expected one run, found {len(runs)}")
        return runs[0]

    @staticmethod
    def contents(run: Path, prefix: str) -> str:
        return "".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in sorted(run.glob(f"{prefix}.*.log"))
        )

    def invoke(
        self,
        output: Path,
        child: list[str],
        extra: list[str] | None = None,
        timeout: float = 15.0,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            "-B",
            str(CAPTURE),
            "--role",
            "ue",
            "--output",
            str(output),
            "--health-interval",
            "0.02",
            "--startup-grace",
            "0",
        ]
        if extra:
            command.extend(extra)
        command.extend(["--", *child])
        return subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)

    def test_small_end_to_end_exit_status_and_streams(self) -> None:
        with self.temporary_directory() as temporary:
            output = Path(temporary) / "output"
            code = "import os, pathlib, sys; assert os.environ['LD_LIBRARY_PATH'].split(os.pathsep)[0] == str(pathlib.Path(sys.executable).resolve().parent); pathlib.Path('cwd-marker').write_text('ok'); print('fixture-safe'); print('fixture-stderr', file=sys.stderr); raise SystemExit(7)"
            result = self.invoke(output, [sys.executable, "-c", code])
            self.assertEqual(result.returncode, 7)
            run = self.run_directory(output)
            self.assertIn("fixture-safe", self.contents(run, "stdout"))
            self.assertTrue((run / "working" / "cwd-marker").is_file())
            self.assertIn("fixture-stderr", self.contents(run, "stderr"))
            status = json.loads((run / "status.json").read_text())
            self.assertEqual(status["exit_code"], 7)
            self.assertEqual(status["state"], "exited_nonzero")

    def test_launch_failure_has_stable_nonzero_status(self) -> None:
        with self.temporary_directory() as temporary:
            output = Path(temporary) / "output"
            result = self.invoke(output, ["/definitely-not-a-flight-binary"])
            self.assertEqual(result.returncode, 127)
            status = json.loads((self.run_directory(output) / "status.json").read_text())
            self.assertEqual(status["state"], "launch_failed")
            self.assertEqual(status["exit_code"], 127)

    def test_actual_uicc_key_argv_forms_are_redacted(self) -> None:
        redacted = redact_argv([
            "/absolute/nr-uesoftmodem",
            "--uicc0.key",
            "first-synthetic-secret",
            "--uicc0.key=second-synthetic-secret",
            "--key",
            "third-synthetic-secret",
            "--key=fourth-synthetic-secret",
        ])
        joined = " ".join(redacted)
        for secret in ("first-synthetic-secret", "second-synthetic-secret", "third-synthetic-secret", "fourth-synthetic-secret"):
            self.assertNotIn(secret, joined)
        self.assertEqual(redacted[1], "<redacted-argument>")
        self.assertEqual(redacted[2], "<redacted-value>")

    def test_relative_config_is_rejected_before_creating_run(self) -> None:
        with self.temporary_directory() as temporary:
            output = Path(temporary) / "output"
            result = self.invoke(output, [sys.executable, "-c", "raise SystemExit(0)"], ["--config", "relative.yaml"])
            self.assertEqual(result.returncode, 2)
            self.assertFalse(output.exists())

    def test_giant_line_and_sensitive_block_are_dropped(self) -> None:
        with self.temporary_directory() as temporary:
            output = Path(temporary) / "output"
            code = (
                "import sys; "
                "sys.stdout.write('X' * 9000 + '\\n[NAS] K_' + chr(27) + '[31mNASenc = should-not-persist\\n[RRC] deadbeefdeadbeef\\n\\nCMDLINE --uicc0.key a1b2c3d4e5f60708\\n\\n[RRC] KgNB 1029384756abcdef\\n\\n[RRC] deriving kRRCenc, kRRCint from KgNB=89abcdef01234567\\n\\n[SAFE] after-boundary\\n'); "
                "sys.stdout.flush()"
            )
            result = self.invoke(output, [sys.executable, "-c", code])
            self.assertEqual(result.returncode, 0)
            run = self.run_directory(output)
            captured = self.contents(run, "stdout")
            self.assertNotIn("private key", captured.lower())
            self.assertNotIn("deadbeef", captured.lower())
            self.assertNotIn("a1b2c3d4e5f60708", captured)
            self.assertNotIn("1029384756abcdef", captured)
            self.assertNotIn("89abcdef01234567", captured)
            self.assertNotIn("X" * 128, captured)
            self.assertIn("after-boundary", captured)
            status = json.loads((run / "status.json").read_text())
            sanitize = status["extra"]["stdout_sanitizer"]
            self.assertGreaterEqual(sanitize["unterminated_line_truncated"], 1)
            self.assertGreaterEqual(sanitize["sensitive_blocks"], 1)
            self.assertGreaterEqual(sanitize["continuity_lines_redacted"], 1)

    def test_recognized_module_diagnostic_ends_sensitive_block_without_blank(self) -> None:
        with self.temporary_directory() as temporary:
            output = Path(temporary) / "output"
            payload = (
                b"\x1b[31m2026-09-17T12:34:56.789 [NAS] K_NASenc=0011223344556677\n"
                b"[RRC] deadbeefdeadbeef\n"
                b"[RRC] 0000: de ad be ef 01 23 45 67\n"
                b"de ad be ef 01 23 45 67\n"
                b"unrecognized continuation must-not-persist\n"
                b"\x1b[32m2026-09-17 12:34:57.001 [MAC] controlled-stop socket diagnostic\n"
            )
            code = f"import sys; sys.stdout.buffer.write({payload!r}); sys.stdout.flush()"
            result = self.invoke(output, [sys.executable, "-c", code])
            self.assertEqual(result.returncode, 0)
            captured = self.contents(self.run_directory(output), "stdout")
            for value in (
                "0011223344556677",
                "deadbeefdeadbeef",
                "de ad be ef 01 23 45 67",
                "unrecognized continuation",
            ):
                self.assertNotIn(value, captured)
            self.assertIn("[MAC] controlled-stop socket diagnostic", captured)

    def test_rotation_and_disk_quota_record_loss_without_blocking_child(self) -> None:
        with self.temporary_directory() as temporary:
            output = Path(temporary) / "output"
            code = "for index in range(1000): print('line-%04d' % index)"
            result = self.invoke(
                output,
                [sys.executable, "-c", code],
                ["--stdout-budget", "1024", "--chunk-bytes", "256"],
            )
            self.assertEqual(result.returncode, 0)
            run = self.run_directory(output)
            files = sorted(run.glob("stdout.*.log"))
            self.assertGreaterEqual(len(files), 2)
            self.assertLessEqual(sum(path.stat().st_size for path in files), 1024)
            status = json.loads((run / "status.json").read_text())
            self.assertFalse(status["capture"]["healthy"])
            self.assertGreater(status["writers"]["stdout"]["dropped_bytes"], 0)

    def test_term_is_forwarded_to_only_child_group(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            output = root / "output"
            marker = root / "term-marker"
            code = (
                "import pathlib, signal, time; "
                f"marker=pathlib.Path({str(marker)!r}); "
                "signal.signal(signal.SIGTERM, lambda _s, _f: (marker.write_text('term'), time.sleep(0.3), (_ for _ in ()).throw(SystemExit(0)))); "
                "print('ready', flush=True); time.sleep(20)"
            )
            command = [
                sys.executable,
                "-B",
                str(CAPTURE),
                "--role",
                "ue",
                "--output",
                str(output),
                "--health-interval",
                "0.02",
                "--",
                sys.executable,
                "-c",
                code,
            ]
            capture = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 5.0
                ready = False
                while time.monotonic() < deadline:
                    if output.exists():
                        runs = [path for path in output.iterdir() if path.is_dir()]
                        if runs and "ready" in self.contents(runs[0], "stdout"):
                            ready = True
                            break
                    time.sleep(0.02)
                self.assertTrue(ready, "capture did not reach child-ready state")
                capture.send_signal(signal.SIGTERM)
                marker_deadline = time.monotonic() + 3.0
                while not marker.exists() and time.monotonic() < marker_deadline:
                    time.sleep(0.01)
                self.assertTrue(marker.exists(), "child did not receive forwarded TERM")
                try:
                    os.kill(capture.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                capture.wait(timeout=10)
            finally:
                if capture.poll() is None:
                    capture.kill()
                    capture.wait()
                capture.communicate(timeout=1)
            self.assertEqual(capture.returncode, 0)
            self.assertEqual(marker.read_text(), "term")
            status = json.loads((self.run_directory(output) / "status.json").read_text())
            self.assertEqual(status["stop_signal"], signal.SIGTERM)
            self.assertEqual(status["counters"]["stop_requests"], 1)
            self.assertEqual(status["counters"]["graceful_group_signals"], 1)

    def test_post_leader_pipe_timeout_preserves_exit_status(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            output = root / "output"
            pid_file = root / "descendant-pid"
            descendant_code = "import time; time.sleep(20)"
            leader_code = (
                "import pathlib, subprocess, sys; "
                f"child=subprocess.Popen([sys.executable, \"-c\", {descendant_code!r}]); "
                f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid)); "
                "print('leader-exiting', flush=True); raise SystemExit(7)"
            )
            started = time.monotonic()
            try:
                result = self.invoke(
                    output,
                    [sys.executable, "-c", leader_code],
                    ["--post-exit-drain-timeout", "0.2"],
                    timeout=10,
                )
                self.assertEqual(result.returncode, 7)
                self.assertLess(time.monotonic() - started, 4.0)
                status = json.loads((self.run_directory(output) / "status.json").read_text())
                drain = status["extra"]["pipe_drain"]
                self.assertEqual(status["exit_code"], 7)
                self.assertEqual(drain["leader_returncode"], 7)
                self.assertEqual(drain["residual_output_status"], "incomplete")
                self.assertGreaterEqual(drain["residual_open_pipe_streams"], 1)
            finally:
                if pid_file.exists():
                    try:
                        os.kill(int(pid_file.read_text()), signal.SIGTERM)
                    except (ProcessLookupError, ValueError):
                        pass

    def test_unrelated_process_is_untouched_by_capture_shutdown(self) -> None:
        with self.temporary_directory() as temporary:
            output = Path(temporary) / "output"
            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
            capture = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    str(CAPTURE),
                    "--role",
                    "ue",
                    "--output",
                    str(output),
                    "--",
                    sys.executable,
                    "-c",
                    "import time; time.sleep(20)",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                time.sleep(0.3)
                capture.send_signal(signal.SIGTERM)
                capture.wait(timeout=10)
                self.assertIsNone(unrelated.poll())
            finally:
                if capture.poll() is None:
                    capture.kill()
                    capture.wait()
                capture.communicate(timeout=1)
                if unrelated.poll() is None:
                    unrelated.terminate()
                    unrelated.wait(timeout=5)

    def test_unavailable_clock_tools_and_gpsd_are_not_zero(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            collector = SystemHealthCollector(
                interface="missing-tunnel",
                gpsd=("127.0.0.1", 1),
                proc_root=root / "no-proc",
                sys_root=root / "no-sys",
                command_paths={"chronyc": (), "ip": (), "ping": ()},
            )
            observation = collector.sample(999999, True)
            self.assertEqual(observation["chrony"]["status"], "unavailable")
            self.assertEqual(observation["gpsd"]["status"], "unavailable")
            self.assertIsNone(observation["cpu"]["percent"])
            self.assertEqual(observation["interface"]["state"], "absent")

    def test_gnb_missing_default_ue_tunnel_becomes_healthy(self) -> None:
        machine = HealthStateMachine(grace_seconds=0, hysteresis_samples=3, role="gnb")
        for mono in (0, 1_000_000_000):
            current = machine.observe(HealthInput(mono, True, False, None))
            self.assertEqual(current["state"], "acquiring")
        healthy = machine.observe(HealthInput(2_000_000_000, True, False, None))
        self.assertEqual(healthy["state"], "healthy")
        self.assertEqual(healthy["role"], "gnb")
        self.assertIn("missing UE tunnel", healthy["evidence"][1]["scope"])

    def test_enabled_recorder_budget_boundaries(self) -> None:
        for budget in (8192, 134217728):
            with self.subTest(budget=budget), self.temporary_directory() as temporary:
                output = Path(temporary) / "output"
                result = self.invoke(
                    output,
                    [sys.executable, "-c", "raise SystemExit(0)"],
                    ["--recorder-budget", str(budget)],
                )
                self.assertEqual(result.returncode, 0)
                metadata = json.loads((self.run_directory(output) / "metadata.json").read_text())
                self.assertEqual(metadata["recorder"]["maximum_bytes"], budget)
                privacy = metadata["privacy"]
                self.assertFalse(privacy["numeric_recorder_packet_payloads_saved"])
                self.assertFalse(privacy["host_collector_packet_payloads_saved"])
                self.assertNotIn("packet_payloads_saved", privacy)

    def test_enabled_recorder_budget_outside_boundaries_is_rejected(self) -> None:
        for budget in (8191, 134217729):
            with self.subTest(budget=budget), self.temporary_directory() as temporary:
                output = Path(temporary) / "output"
                result = self.invoke(
                    output,
                    [sys.executable, "-c", "raise SystemExit(0)"],
                    ["--recorder-budget", str(budget)],
                )
                self.assertEqual(result.returncode, 2)
                self.assertFalse(output.exists())

    def test_state_machine_grace_hysteresis_and_process_exit(self) -> None:
        machine = HealthStateMachine(grace_seconds=10, hysteresis_samples=3)
        starting = machine.observe(HealthInput(0, True, False, None))
        self.assertEqual(starting["state"], "starting")
        acquiring = machine.observe(HealthInput(11_000_000_000, True, False, None))
        self.assertEqual(acquiring["state"], "acquiring")
        for mono in (12_000_000_000, 13_000_000_000):
            current = machine.observe(HealthInput(mono, True, True, True))
            self.assertEqual(current["state"], "acquiring")
        healthy = machine.observe(HealthInput(14_000_000_000, True, True, True))
        self.assertEqual(healthy["state"], "healthy")
        for mono in (15_000_000_000, 16_000_000_000):
            current = machine.observe(HealthInput(mono, True, True, False))
            self.assertEqual(current["state"], "healthy")
        degraded = machine.observe(HealthInput(17_000_000_000, True, True, False))
        self.assertEqual(degraded["state"], "degraded")
        exited = machine.observe(HealthInput(18_000_000_000, False, True, True))
        self.assertEqual(exited["state"], "exited")
        self.assertEqual(exited["state_basis"], "direct")


if __name__ == "__main__":
    unittest.main(verbosity=2)
