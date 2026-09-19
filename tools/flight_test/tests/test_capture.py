#!/usr/bin/env python3
"""Deterministic stdlib fixtures for the flight capture tool."""

from __future__ import annotations

import json
import os
import signal
import socket
from unittest.mock import patch
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
import capture as capture_module
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

    def test_free_space_reserve_keeps_child_running_and_reports_loss(self) -> None:
        with self.temporary_directory() as temporary:
            output = Path(temporary) / "output"
            result = self.invoke(output, [sys.executable, "-c", "print('still-running'); raise SystemExit(7)"],
                                 ["--min-free-bytes", str((1 << 63) - 1)])
            self.assertEqual(result.returncode, 7)
            status = json.loads((self.run_directory(output) / "status.json").read_text())
            self.assertFalse(status["capture"]["healthy"])
            self.assertGreater(status["writers"]["stdout"]["dropped_bytes"], 0)

    def test_console_mirror_is_redacted(self) -> None:
        with self.temporary_directory() as temporary:
            output = Path(temporary) / "output"
            result = self.invoke(output, [sys.executable, "-c", "print('password=synthetic-secret'); print(); print('[SAFE] console-copy')"],
                                 ["--console"])
            self.assertEqual(result.returncode, 0)
            self.assertIn("console-copy", result.stdout)
            self.assertNotIn("synthetic-secret", result.stdout)

    def test_small_end_to_end_exit_status_and_streams(self) -> None:
        with self.temporary_directory() as temporary:
            output = Path(temporary) / "output"
            code = "import os, pathlib, stat, sys; assert os.environ['LD_LIBRARY_PATH'].split(os.pathsep)[0] == str(pathlib.Path(sys.executable).resolve().parent); native_fd=int(os.environ['_OAI_FLIGHT_MONITOR_FD']); assert stat.S_ISSOCK(os.fstat(native_fd).st_mode); pathlib.Path('cwd-marker').write_text('ok'); print('fixture-safe'); print('fixture-stderr', file=sys.stderr); raise SystemExit(7)"
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
        self.assertEqual(healthy["state"], "host_ready")
        self.assertEqual(healthy["service_state"], "unverified")
        self.assertEqual(healthy["role"], "gnb")
        self.assertIn("missing UE tunnel", healthy["evidence"][1]["scope"])

    def test_enabled_recorder_budget_boundaries(self) -> None:
        for budget in (0, 8192, 134217728, 1073741824):
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
        for budget in (-1, 8191, 1 << 63):
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
        self.assertEqual(healthy["state"], "host_ready")
        self.assertEqual(healthy["service_state"], "unverified")
        for mono in (15_000_000_000, 16_000_000_000):
            current = machine.observe(HealthInput(mono, True, True, False))
            self.assertEqual(current["state"], "host_ready")
        degraded = machine.observe(HealthInput(17_000_000_000, True, True, False))
        self.assertEqual(degraded["state"], "degraded")
        exited = machine.observe(HealthInput(18_000_000_000, False, True, True))
        self.assertEqual(exited["state"], "exited")
        self.assertEqual(exited["state_basis"], "direct")


    def test_udp_and_process_status_are_bounded_observer_evidence(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary) / "proc"
            (root / "net").mkdir(parents=True)
            (root / "net" / "snmp").write_text(
                "Udp: InDatagrams NoPorts InErrors OutDatagrams RcvbufErrors SndbufErrors\nUdp: 1 2 3 4 5 6\n"
            )
            pid_root = root / "4242"
            pid_root.mkdir()
            (pid_root / "status").write_text(
                "State:\tS (sleeping)\nThreads:\t7\nVmRSS:\t8 kB\nvoluntary_ctxt_switches:\t9\nnonvoluntary_ctxt_switches:\t10\n"
            )
            collector = SystemHealthCollector("missing", proc_root=root, sys_root=root / "sys", command_paths={"chronyc": (), "ip": (), "ping": ()})
            observation = collector.sample(4242, True)
            udp = observation["udp_host_counters"]
            self.assertEqual(udp["state"], "available")
            self.assertEqual(udp["counters"], {"InDatagrams": 1, "OutDatagrams": 4, "InErrors": 3, "RcvbufErrors": 5, "SndbufErrors": 6, "NoPorts": 2})
            status = observation["process"]["status_observation"]
            self.assertEqual(status["state"], "available")
            self.assertEqual(status["process_state"], "S (sleeping)")
            self.assertEqual(status["threads"], 7)
            self.assertEqual(status["vm_rss_bytes"], 8192)
            self.assertEqual(status["voluntary_context_switches"], 9)
            self.assertEqual(status["nonvoluntary_context_switches"], 10)

    def test_recovery_thresholds_must_be_finite_and_at_least_one_second(self) -> None:
        with self.temporary_directory() as temporary:
            for option, value in (("--recovery-stall", "nan"), ("--recovery-attempt", "inf"), ("--recovery-stall", "0.5")):
                with self.subTest(option=option, value=value):
                    output = Path(temporary) / f"output-{option[11:]}-{value}"
                    result = self.invoke(
                        output,
                        [sys.executable, "-c", "raise SystemExit(0)"],
                        [option, value],
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertFalse(output.exists())

    def test_final_native_reject_between_drain_and_exit_poll_is_retained(self) -> None:
        class MemoryWriter:
            def __init__(self):
                self.data = bytearray()
            def write(self, data):
                self.data.extend(data)

        class IdleHealthWorker:
            def __init__(self, *args): pass
            def start(self): pass
            def stop(self): pass
            def join(self, **kwargs): pass
            def is_alive(self): return False

        class ExitingProcess:
            pid = 12345678
            def __init__(self, *args, **kwargs):
                self.sender = socket.socket(fileno=os.dup(kwargs["pass_fds"][0]))
                self.stdout = self.eof_pipe()
                self.stderr = self.eof_pipe()
                self.polls = 0
                self.send(1, {"rx_samples": 1})
            @staticmethod
            def eof_pipe():
                reader, writer = os.pipe()
                os.close(writer)
                return os.fdopen(reader, "rb")
            def send(self, sequence, values):
                self.sender.send(json.dumps(dict(kind="native_progress", schema_version=1,
                    pid=self.pid, sequence=sequence, mono_ns=time.monotonic_ns(),
                    send_drops=0, values=values)).encode())
            def poll(self):
                self.polls += 1
                if self.polls == 1:
                    return None
                if self.polls == 2:
                    # The worker's atexit message races the collector's poll.
                    self.send(2, {"rx_samples": 1,
                        "nas_reject": (1 << 56) | (101 << 48) | (1 << 40) | (0x42 << 32) | 10})
                    self.sender.close()
                return 0
            def wait(self): return 0

        with self.temporary_directory() as temporary:
            root = Path(temporary)
            args = capture_module.build_parser().parse_args([
                "--role", "ue", "--output", str(root), "--disable-recorder", "--", sys.executable])
            subject = capture_module.FlightCapture(args, [sys.executable], run_dir=root)
            subject._final_status = lambda *args, **kwargs: None
            outputs = [MemoryWriter() for _ in range(4)]
            with patch.object(capture_module.subprocess, "Popen", ExitingProcess), \
                    patch.object(capture_module, "HealthWorker", IdleHealthWorker):
                code = subject._launch_and_supervise(root, root, *outputs)
            events = [json.loads(line) for line in outputs[3].data.splitlines()]
            sequences = [e["source_sequence"] for e in events if e["kind"] == "native_progress_snapshot"]
            self.assertEqual(code, 0)
            self.assertEqual(sequences, [1, 2])
            self.assertEqual(subject.native_channel["sequence"], 2)
            self.assertEqual(subject.policy_decision.action, "retry")
            self.assertEqual(subject.policy.last_reject_cause, 101)

    def test_absent_group_recovery_signal_does_not_latch_controlled_reason(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            args = capture_module.build_parser().parse_args([
                "--role", "ue", "--output", str(root), "--disable-recorder", "--", sys.executable])
            subject = capture_module.FlightCapture(args, [sys.executable], run_dir=root, recovery_enabled=True)

            class AbsentGroup:
                pid = 12345679

            with patch.object(capture_module.os, "killpg", side_effect=ProcessLookupError) as killpg:
                subject._request_recovery_stop(AbsentGroup(), "radio_rx_progress_stalled")
            killpg.assert_called_once_with(AbsentGroup.pid, signal.SIGINT)
            self.assertIsNone(subject.recovery_stop_reason)
            self.assertIsNone(subject.recovery_stop_sent_ns)
            self.assertNotIn("recovery_stop_requests", subject.counters)

    def test_exited_zero_never_becomes_controlled_when_restart_tick_races_exit(self) -> None:
        class MemoryWriter:
            def __init__(self):
                self.data = bytearray()
            def write(self, data):
                self.data.extend(data)

        class IdleHealthWorker:
            def __init__(self, *args): pass
            def start(self): pass
            def stop(self): pass
            def join(self, **kwargs): pass
            def is_alive(self): return False

        class ExitedProcess:
            pid = 12345680
            def __init__(self, *args, **kwargs):
                sender = socket.socket(fileno=os.dup(kwargs["pass_fds"][0]))
                sender.send(json.dumps(dict(kind="native_progress", schema_version=1,
                    pid=self.pid, sequence=1, mono_ns=time.monotonic_ns(),
                    send_drops=0, values={"rx_samples": 1})).encode())
                sender.close()
                self.stdout = self.eof_pipe()
                self.stderr = self.eof_pipe()
            @staticmethod
            def eof_pipe():
                reader, writer = os.pipe()
                os.close(writer)
                return os.fdopen(reader, "rb")
            def poll(self): return 0
            def wait(self): return 0

        with self.temporary_directory() as temporary:
            root = Path(temporary)
            args = capture_module.build_parser().parse_args([
                "--role", "ue", "--output", str(root), "--disable-recorder", "--", sys.executable])
            subject = capture_module.FlightCapture(args, [sys.executable], run_dir=root, recovery_enabled=True)
            subject._final_status = lambda *args, **kwargs: None
            outputs = [MemoryWriter() for _ in range(4)]
            with patch.object(capture_module.subprocess, "Popen", ExitedProcess), \
                    patch.object(capture_module, "HealthWorker", IdleHealthWorker), \
                    patch.object(subject.policy, "tick", return_value=capture_module.Decision("restart", "synthetic_stall")), \
                    patch.object(capture_module.os, "killpg", side_effect=ProcessLookupError) as killpg:
                code = subject._launch_and_supervise(root, root, *outputs)
            self.assertEqual(code, 0)
            killpg.assert_not_called()
            self.assertIsNone(subject.recovery_stop_reason)
            self.assertIsNone(subject.recovery_stop_sent_ns)
            self.assertEqual(subject.policy_decision.action, "stop")
            self.assertEqual(subject.policy_decision.reason, "unclassified_zero_exit")

    def test_one_shot_log_records_valid_native_datagram_without_relaunch(self) -> None:
        with self.temporary_directory() as temporary:
            output = Path(temporary) / "output"
            code = "import json, os, socket, time; channel=socket.socket(fileno=int(os.environ['_OAI_FLIGHT_MONITOR_FD'])); channel.send(json.dumps({'kind':'native_progress','schema_version':1,'pid':os.getpid(),'sequence':1,'mono_ns':time.monotonic_ns(),'send_drops':0,'values':{'rx_samples':1}}).encode())"
            result = self.invoke(output, [sys.executable, "-c", code])
            self.assertEqual(result.returncode, 0)
            run = self.run_directory(output)
            status = json.loads((run / "status.json").read_text())
            self.assertEqual(status["recovery"]["native_channel"]["sequence"], 1)
            self.assertEqual(len([path for path in output.iterdir() if path.is_dir()]), 1)
            events = [json.loads(line) for line in self.contents(run, "recovery").splitlines()]
            self.assertTrue(any(event["kind"] == "native_progress_snapshot" for event in events))

if __name__ == "__main__":
    unittest.main(verbosity=2)
