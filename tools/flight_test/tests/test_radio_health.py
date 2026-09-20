#!/usr/bin/env python3
"""Deterministic evidence and role-supervision coverage for radio health."""

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


TOOL_DIR = Path(__file__).resolve().parents[1]
WORKTREE = TOOL_DIR.parents[1]
CAPTURE = TOOL_DIR / "capture.py"
VALIDATION_ROOT = WORKTREE / "cmake_targets/log/FlightTests/Validation/radio_health_2026-09-20"
sys.path.insert(0, str(TOOL_DIR))

from flight_recovery import NativeChannel, RecoveryPolicy
from radio_health import (
    MAX_RADIO_HEALTH_DATAGRAM,
    RadioHealthDiagnostics,
    parse_radio_health_datagram,
)


NS = 1_000_000_000


def snapshot(
    *,
    pid: int = 321,
    sequence: int = 1,
    mono_ns: int = NS,
    device_id: int = 0,
    backend: str = "uhd",
    device_type: int = 0,
    active: bool = True,
    supported: list[str] | None = None,
    values: dict[str, int] | None = None,
    send_drops: int = 0,
) -> dict:
    return {
        "kind": "radio_health",
        "schema_version": 1,
        "pid": pid,
        "sequence": sequence,
        "mono_ns": mono_ns,
        "send_drops": send_drops,
        "device_id": device_id,
        "backend": backend,
        "device_type": device_type,
        "active": active,
        "supported": supported or [],
        "values": values or {},
    }


class RadioHealthWireAndEvidence(unittest.TestCase):
    def test_uhd_and_future_metric_are_preserved_while_unobserved_stays_unavailable(self) -> None:
        raw = snapshot(
            supported=["tx_send_requested_samples", "tx_send_accepted_samples", "future_counter"],
            values={"tx_send_requested_samples": 20, "future_counter": (1 << 64) - 1},
        )
        parsed = parse_radio_health_datagram(json.dumps(raw).encode(), 321, 2 * NS)
        self.assertEqual(parsed["backend"], "uhd")
        self.assertEqual(parsed["values"]["future_counter"], (1 << 64) - 1)
        self.assertNotIn("tx_send_accepted_samples", parsed["values"])

        generic = snapshot(sequence=2, mono_ns=2 * NS, backend="rfsimulator", device_type=(1 << 32) - 1)
        self.assertEqual(parse_radio_health_datagram(json.dumps(generic).encode(), 321, 3 * NS)["backend"], "rfsimulator")

    def test_parser_rejects_malformed_supported_and_oversized_input(self) -> None:
        malformed = snapshot(supported=[["not", "a", "metric"]])
        with self.assertRaises(ValueError):
            parse_radio_health_datagram(json.dumps(malformed).encode(), 321, 2 * NS)
        with self.assertRaises(ValueError):
            parse_radio_health_datagram(b"x" * (MAX_RADIO_HEALTH_DATAGRAM + 1), 321, 2 * NS)

    def test_native_channel_routes_only_valid_health_and_keeps_legacy_progress(self) -> None:
        channel = NativeChannel()
        self.addCleanup(channel.close)
        legacy = {
            "kind": "native_progress",
            "schema_version": 1,
            "pid": 321,
            "sequence": 1,
            "mono_ns": NS,
            "send_drops": 0,
            "values": {"rx_samples": 1},
        }
        channel.child.send(json.dumps(legacy).encode())
        channel.child.send(json.dumps(snapshot(sequence=1, supported=["rx_returned_samples"], values={"rx_returned_samples": 1})).encode())
        channel.child.send(json.dumps(snapshot(sequence=2, mono_ns=NS + 1, pid=999)).encode())
        self.assertEqual(len(channel.receive(321, 2 * NS)), 1)
        health = channel.take_radio_health()
        self.assertEqual(len(health), 1)
        self.assertEqual(health[0]["values"]["rx_returned_samples"], 1)
        self.assertEqual(channel.radio_health_invalid, 1)

    def test_derived_tick_and_channel_changes_require_validity_at_both_endpoints(self) -> None:
        diagnostics = RadioHealthDiagnostics()
        supported = [
            "tx_async_last_event_channel",
            "tx_async_last_event_channel_valid",
            "tx_async_last_event_device_ticks",
            "tx_async_last_event_device_time_valid",
            "rx_last_device_ticks",
            "rx_last_device_time_valid",
        ]
        first_values = {
            "tx_async_last_event_channel": 3,
            "tx_async_last_event_channel_valid": 1,
            "tx_async_last_event_device_ticks": 12_345,
            "tx_async_last_event_device_time_valid": 1,
            "rx_last_device_ticks": 54_321,
            "rx_last_device_time_valid": 1,
        }
        invalid_values = {
            "tx_async_last_event_channel": 0,
            "tx_async_last_event_channel_valid": 0,
            "tx_async_last_event_device_ticks": 0,
            "tx_async_last_event_device_time_valid": 0,
            "rx_last_device_ticks": 0,
            "rx_last_device_time_valid": 0,
        }
        valid_again_values = {
            "tx_async_last_event_channel": 4,
            "tx_async_last_event_channel_valid": 1,
            "tx_async_last_event_device_ticks": 20,
            "tx_async_last_event_device_time_valid": 1,
            "rx_last_device_ticks": 30,
            "rx_last_device_time_valid": 1,
        }
        diagnostics.observe(snapshot(supported=supported, values=first_values), NS)
        invalid_events = diagnostics.observe(
            snapshot(sequence=2, mono_ns=2 * NS, supported=supported, values=invalid_values),
            2 * NS,
        )
        invalid_deltas = next(event for event in invalid_events if event["kind"] == "radio_health_interval")["metric_deltas"]
        for name in (
            "tx_async_last_event_channel",
            "tx_async_last_event_device_ticks",
            "rx_last_device_ticks",
        ):
            self.assertEqual(invalid_deltas[name]["state"], "derived_invalid_or_unavailable")
            self.assertIsNone(invalid_deltas[name]["change"])
            self.assertNotIn("value", invalid_deltas[name])
        restored_events = diagnostics.observe(
            snapshot(sequence=3, mono_ns=3 * NS, supported=supported, values=valid_again_values),
            3 * NS,
        )
        restored_deltas = next(event for event in restored_events if event["kind"] == "radio_health_interval")["metric_deltas"]
        for name in (
            "tx_async_last_event_channel",
            "tx_async_last_event_device_ticks",
            "rx_last_device_ticks",
        ):
            self.assertEqual(restored_deltas[name]["state"], "gauge_observed_without_valid_previous")
            self.assertIsNone(restored_deltas[name]["change"])
        final_values = dict(valid_again_values)
        final_values.update({
            "tx_async_last_event_channel": 5,
            "tx_async_last_event_device_ticks": 21,
            "rx_last_device_ticks": 31,
        })
        final_events = diagnostics.observe(
            snapshot(sequence=4, mono_ns=4 * NS, supported=supported, values=final_values),
            4 * NS,
        )
        final_deltas = next(event for event in final_events if event["kind"] == "radio_health_interval")["metric_deltas"]
        self.assertEqual(final_deltas["tx_async_last_event_channel"]["change"], 1)
        self.assertEqual(final_deltas["tx_async_last_event_device_ticks"]["change"], 1)
        self.assertEqual(final_deltas["rx_last_device_ticks"]["change"], 1)

    def test_derived_tick_and_channel_values_without_validity_are_unavailable(self) -> None:
        diagnostics = RadioHealthDiagnostics()
        supported = [
            "tx_async_last_event_channel",
            "tx_async_last_event_channel_valid",
            "tx_async_last_event_device_ticks",
            "tx_async_last_event_device_time_valid",
            "rx_last_device_ticks",
            "rx_last_device_time_valid",
        ]
        values = {
            "tx_async_last_event_channel": 3,
            "tx_async_last_event_device_ticks": 12_345,
            "rx_last_device_ticks": 54_321,
        }
        diagnostics.observe(snapshot(supported=supported, values=values), NS)
        events = diagnostics.observe(
            snapshot(sequence=2, mono_ns=2 * NS, supported=supported, values=values),
            2 * NS,
        )
        deltas = next(event for event in events if event["kind"] == "radio_health_interval")["metric_deltas"]
        for name in (
            "tx_async_last_event_channel",
            "tx_async_last_event_device_ticks",
            "rx_last_device_ticks",
        ):
            self.assertEqual(deltas[name]["state"], "derived_invalid_or_unavailable")
            self.assertIsNone(deltas[name]["change"])
            self.assertNotIn("value", deltas[name])

    def test_diagnostics_retains_gaps_resets_lifecycle_and_observe_only_pending(self) -> None:
        diagnostics = RadioHealthDiagnostics()
        supported = ["tx_send_requested_samples", "tx_send_accepted_samples", "tx_queue_depth", "tx_async_underflow", "tx_send_inflight", "rx_recv_inflight"]
        first = snapshot(supported=supported, values={"tx_send_requested_samples": 0, "tx_send_accepted_samples": 0, "tx_queue_depth": 0, "tx_send_inflight": 0, "rx_recv_inflight": 0})
        self.assertTrue(any(event["kind"] == "radio_health_snapshot" for event in diagnostics.observe(first, NS)))
        second = snapshot(
            sequence=3,
            mono_ns=2 * NS,
            supported=supported,
            values={"tx_send_requested_samples": 20, "tx_send_accepted_samples": 10, "tx_queue_depth": 5, "tx_send_inflight": 1, "rx_recv_inflight": 1},
        )
        events = diagnostics.observe(second, 2 * NS)
        self.assertTrue(any(event["kind"] == "radio_health_source_gap" for event in events))
        interval = next(event for event in events if event["kind"] == "radio_health_interval")
        self.assertEqual(interval["metric_deltas"]["tx_async_underflow"]["state"], "supported_unobserved")
        self.assertEqual(interval["tx_progress"]["accepted_samples"]["value"], 10)
        self.assertEqual(interval["tx_progress"]["delivered_samples"]["state"], "unsupported_by_native_schema")
        self.assertFalse(any(event["kind"] == "radio_health_observation" for event in events))

        third = snapshot(
            sequence=4,
            mono_ns=3 * NS,
            supported=supported,
            values={"tx_send_requested_samples": 40, "tx_send_accepted_samples": 10, "tx_queue_depth": 5, "tx_send_inflight": 1, "rx_recv_inflight": 1},
        )
        diagnostics.observe(third, 3 * NS)
        fourth = snapshot(
            sequence=5,
            mono_ns=4 * NS,
            supported=supported,
            values={"tx_send_requested_samples": 60, "tx_send_accepted_samples": 10, "tx_queue_depth": 5, "tx_send_inflight": 1, "rx_recv_inflight": 1},
        )
        events = diagnostics.observe(fourth, 4 * NS)
        pending = [event for event in events if event["kind"] == "radio_health_observation"]
        self.assertEqual(pending[0]["reason"], "pending_work_without_observed_transport_progress")
        self.assertEqual(pending[0]["candidate_action"], "observe")
        self.assertFalse(pending[0]["qualified"])
        missing = diagnostics.tick(7 * NS)
        self.assertEqual(missing[0]["reason"], "radio_health_missing_with_pending_work")

        reset = snapshot(
            sequence=6,
            mono_ns=5 * NS,
            supported=supported,
            values={"tx_send_requested_samples": 1, "tx_send_accepted_samples": 1, "tx_queue_depth": 0, "tx_send_inflight": 0, "rx_recv_inflight": 0},
        )
        events = diagnostics.observe(reset, 5 * NS)
        interval = next(event for event in events if event["kind"] == "radio_health_interval")
        self.assertEqual(interval["metric_deltas"]["tx_send_requested_samples"]["state"], "counter_decreased_or_reset")
        closed = snapshot(sequence=7, mono_ns=6 * NS, active=False, supported=supported)
        self.assertTrue(any(event.get("event") == "closed" for event in diagnostics.observe(closed, 6 * NS)))
        summary = diagnostics.snapshot()
        self.assertEqual(summary["first_native_event"]["sequence"], 1)
        self.assertEqual(summary["last_native_event"]["sequence"], 7)


    def test_pending_work_keeps_missing_sample_progress_unavailable(self) -> None:
        diagnostics = RadioHealthDiagnostics()
        supported = ["tx_send_inflight", "tx_queue_dequeues"]
        diagnostics.observe(
            snapshot(
                supported=supported,
                values={"tx_send_inflight": 0, "tx_queue_dequeues": 7},
            ),
            NS,
        )
        events = diagnostics.observe(
            snapshot(
                sequence=2,
                mono_ns=2 * NS,
                supported=supported,
                values={"tx_send_inflight": 1, "tx_queue_dequeues": 7},
            ),
            2 * NS,
        )
        interval = next(event for event in events if event["kind"] == "radio_health_interval")
        pending = interval["pending_work"]
        self.assertIsNone(pending["tx_send_accepted_samples_delta"])
        self.assertEqual(pending["tx_queue_dequeues_delta"], 0)
        self.assertFalse(pending["pending"])
        self.assertFalse(any(event["kind"] == "radio_health_observation" for event in events))


class GnbPolicy(unittest.TestCase):
    @staticmethod
    def active_health(sequence: int, accepted: int, source_ns: int) -> dict:
        return snapshot(
            sequence=sequence,
            mono_ns=source_ns,
            supported=["tx_send_accepted_samples", "rx_returned_samples"],
            values={"tx_send_accepted_samples": accepted, "rx_returned_samples": 0},
        )

    def test_gnb_requires_two_fresh_accepted_or_returned_sample_observations(self) -> None:
        policy = RecoveryPolicy(role="gnb")
        policy.begin_attempt(NS)
        policy.observe_radio_health(self.active_health(1, 5, NS), NS)
        policy.observe_radio_health(self.active_health(2, 5, 2 * NS), 2 * NS)
        self.assertFalse(policy.radio_progress_observed)
        self.assertEqual(policy.exited(2 * NS, 9, False).reason, "startup_failed_without_positive_radio_progress")

        policy.begin_attempt(3 * NS)
        policy.observe_radio_health(self.active_health(1, 5, 3 * NS), 3 * NS)
        policy.observe_radio_health(self.active_health(2, 6, 4 * NS), 4 * NS)
        decision = policy.exited(4 * NS, -11, False)
        self.assertEqual(decision.action, "retry")
        self.assertIn("after_positive_radio_progress", decision.reason)
        self.assertEqual(policy.exited(4 * NS, 0, False).reason, "unclassified_zero_exit")
        self.assertEqual(policy.exited(4 * NS, -11, True).reason, "operator_stop")

    def test_gnb_does_not_treat_missing_sample_counters_as_zero(self) -> None:
        policy = RecoveryPolicy(role="gnb")
        policy.begin_attempt(NS)
        supported = ["tx_send_accepted_samples"]
        policy.observe_radio_health(snapshot(sequence=1, supported=supported), NS)
        policy.observe_radio_health(
            snapshot(
                sequence=2,
                mono_ns=2 * NS,
                supported=supported,
                values={"tx_send_accepted_samples": 100},
            ),
            2 * NS,
        )
        self.assertFalse(policy.radio_progress_observed)
        self.assertEqual(
            policy.exited(2 * NS, -11, False).reason,
            "startup_failed_without_positive_radio_progress",
        )

        policy.begin_attempt(3 * NS)
        policy.observe_radio_health(
            snapshot(
                sequence=1,
                mono_ns=3 * NS,
                supported=supported,
                values={"tx_send_accepted_samples": 7},
            ),
            3 * NS,
        )
        policy.observe_radio_health(snapshot(sequence=2, mono_ns=4 * NS, supported=supported), 4 * NS)
        policy.observe_radio_health(
            snapshot(
                sequence=3,
                mono_ns=5 * NS,
                supported=supported,
                values={"tx_send_accepted_samples": 8},
            ),
            5 * NS,
        )
        self.assertFalse(policy.radio_progress_observed)
        policy.observe_radio_health(
            snapshot(
                sequence=4,
                mono_ns=6 * NS,
                supported=supported,
                values={"tx_send_accepted_samples": 9},
            ),
            6 * NS,
        )
        self.assertTrue(policy.radio_progress_observed)
        self.assertEqual(policy.exited(6 * NS, -11, False).action, "retry")

    def test_legacy_ue_exit_policy_is_unchanged(self) -> None:
        policy = RecoveryPolicy()
        policy.begin_attempt(NS)
        policy.observe({"mono_ns": NS, "values": {"rx_samples": 1}}, NS)
        self.assertEqual(policy.exited(NS, -11, False).action, "retry")


class GnbSupervisorFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        VALIDATION_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)

    def temporary_directory(self) -> tempfile.TemporaryDirectory[str]:
        return tempfile.TemporaryDirectory(prefix="radio-health-supervisor-", dir=VALIDATION_ROOT)

    @staticmethod
    def command(output: Path, worker: Path) -> list[str]:
        return [
            sys.executable,
            "-B",
            str(CAPTURE),
            "--role",
            "gnb",
            "--recovery",
            "--output",
            str(output),
            "--min-free-bytes",
            "0",
            "--startup-grace",
            "0",
            "--health-interval",
            "0.05",
            "--post-exit-drain-timeout",
            "0.1",
            "--",
            sys.executable,
            "-B",
            str(worker),
        ]

    @staticmethod
    def write_worker(root: Path, source: str) -> Path:
        worker = root / "worker.py"
        worker.write_text(textwrap.dedent(source), encoding="utf-8")
        return worker

    @staticmethod
    def health_prelude() -> str:
        return textwrap.dedent("""\
            import json
            import os
            import socket
            import time

            channel = socket.socket(fileno=int(os.environ["_OAI_FLIGHT_MONITOR_FD"]))
            sequence = 0
            def health(accepted):
                global sequence
                sequence += 1
                channel.send(json.dumps({
                    "kind": "radio_health", "schema_version": 1,
                    "pid": os.getpid(), "sequence": sequence,
                    "mono_ns": time.monotonic_ns(), "send_drops": 0,
                    "device_id": 0, "backend": "uhd", "device_type": 0,
                    "active": True,
                    "supported": ["tx_send_accepted_samples"],
                    "values": {"tx_send_accepted_samples": accepted},
                }).encode("utf-8"))
            """)

    @staticmethod
    def session(output: Path) -> Path:
        sessions = list(output.glob("gnb-session-*"))
        if len(sessions) != 1:
            raise AssertionError(f"expected one gNB session, found {len(sessions)}")
        return sessions[0]

    def test_positive_radio_progress_allows_only_unexpected_nonzero_retry(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            counter = root / "launches"
            worker = self.write_worker(root, self.health_prelude() + textwrap.dedent(f"""
                import pathlib
                counter = pathlib.Path({str(counter)!r})
                try:
                    launch = int(counter.read_text())
                except FileNotFoundError:
                    launch = 0
                counter.write_text(str(launch + 1))
                if launch == 0:
                    health(5)
                    time.sleep(0.05)
                    health(6)
                    raise SystemExit(9)
                raise SystemExit(0)
            """))
            output = root / "output"
            completed = subprocess.run(self.command(output, worker), text=True, capture_output=True, timeout=12, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(counter.read_text(), "2")
            status = json.loads((self.session(output) / "status.json").read_text())
            self.assertEqual(status["attempt_count"], 2)
            self.assertEqual(status["recent_attempts"][0]["policy_decision"]["action"], "retry")

    def test_gnb_startup_failure_and_clean_zero_exit_do_not_retry(self) -> None:
        for mode in ("startup", "zero"):
            with self.subTest(mode=mode), self.temporary_directory() as temporary:
                root = Path(temporary)
                counter = root / "launches"
                worker_source = self.health_prelude() + (
                    f"import pathlib\n"
                    f"counter = pathlib.Path({str(counter)!r})\n"
                    'counter.write_text("1")\n'
                )
                if mode == "zero":
                    worker_source += "health(5)\ntime.sleep(0.05)\nhealth(6)\nraise SystemExit(0)\n"
                else:
                    worker_source += "raise SystemExit(9)\n"
                worker = self.write_worker(root, worker_source)
                output = root / "output"
                completed = subprocess.run(self.command(output, worker), text=True, capture_output=True, timeout=8, check=False)
                self.assertEqual(completed.returncode, 9 if mode == "startup" else 0, completed.stderr)
                status = json.loads((self.session(output) / "status.json").read_text())
                self.assertEqual(status["attempt_count"], 1)
                self.assertEqual(counter.read_text(), "1")
                reason = status["recent_attempts"][0]["policy_decision"]["reason"]
                self.assertEqual(reason, "startup_failed_without_positive_radio_progress" if mode == "startup" else "unclassified_zero_exit")

    def test_gnb_operator_stop_never_relaunches(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            counter = root / "launches"
            worker = self.write_worker(root, self.health_prelude() + textwrap.dedent(f"""
                import pathlib
                counter = pathlib.Path({str(counter)!r})
                counter.write_text("1")
                health(5)
                time.sleep(0.05)
                health(6)
                time.sleep(30)
            """))
            output = root / "output"
            process = subprocess.Popen(self.command(output, worker), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    sessions = list(output.glob("gnb-session-*")) if output.exists() else []
                    if sessions and json.loads((sessions[0] / "status.json").read_text())["state"] == "running":
                        break
                    time.sleep(0.05)
                else:
                    self.fail("gNB session did not start")
                process.send_signal(signal.SIGTERM)
                stdout, stderr = process.communicate(timeout=8)
                self.assertEqual(process.returncode, 128 + signal.SIGTERM, stdout + stderr)
                status = json.loads((self.session(output) / "status.json").read_text())
                self.assertEqual(status["state"], "operator_stop")
                self.assertEqual(status["attempt_count"], 1)
                self.assertEqual(counter.read_text(), "1")
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    process.communicate(timeout=8)


if __name__ == "__main__":
    unittest.main()
