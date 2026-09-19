#!/usr/bin/env python3
"""Policy safety boundaries; no RF or wall-clock waiting required."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flight_recovery import NativeChannel, RecoveryPolicy, NANOSECOND as NS


def reject(cause=101, policy=1, wait=10, t3502=0x42, t3502_present=True):
    policy_byte = policy | (0x80 if t3502_present else 0)
    return (1 << 56) | (cause << 48) | (policy_byte << 40) | (t3502 << 32) | wait


def sample(seconds, **values):
    return {"mono_ns": int(seconds * NS), "values": values}


class Policy(unittest.TestCase):
    def test_search_and_missing_telemetry_do_not_trigger_restart(self):
        p = RecoveryPolicy()
        p.begin_attempt(0)
        for second in range(1, 301):
            p.observe(sample(second, rx_samples=second * 7680000, search_attempts=second), second * NS)
            self.assertEqual(p.tick(second * NS).action, "observe")
        self.assertEqual(p.tick(400 * NS).reason, "native_telemetry_unavailable")

    def test_radio_stall_requires_fresh_monitor_and_confirmation(self):
        p = RecoveryPolicy()
        p.begin_attempt(0)
        for second in (1, 11, 12):
            p.observe(sample(second, rx_samples=100), second * NS)
            decision = p.tick(second * NS)
            self.assertEqual(decision.action, "restart" if second == 12 else "observe")
        self.assertEqual(decision.reason, "radio_rx_progress_stalled")

    def test_async_worker_stall_despite_live_radio(self):
        for stuck in ("ue_dl_completed", "ue_tx_completed"):
            p = RecoveryPolicy()
            p.begin_attempt(0)
            for second in (1, 11, 12):
                values = dict(rx_samples=second * 100, ue_slot_inputs=second * 100,
                              ue_dl_completed=second * 100, ue_tx_completed=second * 100)
                values[stuck] = 1
                p.observe(sample(second, **values), second * NS)
                d = p.tick(second * NS)
            self.assertEqual(d.reason, stuck + "_progress_stalled")
            self.assertEqual(d.action, "restart")
            # No newly queued work during reacquisition is not worker starvation.
            p.observe(sample(20, rx_samples=2000, ue_slot_inputs=1200,
                             ue_dl_completed=1, ue_tx_completed=1), 20 * NS)
            self.assertEqual(p.tick(20 * NS).action, "observe")

    def test_lost_pdu_recovery_deadline_and_rrc_hold(self):
        p = RecoveryPolicy(attempt_seconds=20)
        p.begin_attempt(0)
        p.observe(sample(1, rx_samples=1, sync_successes=1, pdu_active=1, pdu_accepts=1), NS)
        p.observe(sample(2, rx_samples=2, sync_successes=1, pdu_active=0), 2 * NS)
        p.observe(sample(22, rx_samples=22, rrc_hold_until_ns=40 * NS), 22 * NS)
        self.assertEqual(p.tick(22 * NS).reason, "protocol_backoff")
        p.observe(sample(41, rx_samples=41), 41 * NS)
        self.assertEqual(p.tick(41 * NS).action, "restart")

    def test_restored_drb_after_pdu_accept_cancels_deadline_until_next_loss(self):
        p = RecoveryPolicy(attempt_seconds=20)
        p.begin_attempt(0)
        p.observe(sample(1, rx_samples=1, sync_successes=1, pdu_active=1, pdu_accepts=1), NS)
        p.observe(sample(2, rx_samples=2, pdu_active=0, drb_context_active=0), 2 * NS)
        events = p.observe(sample(3, rx_samples=3, drb_context_active=1), 3 * NS)
        self.assertTrue(any(e["event"] == "drb_context_restored" for e in events))
        for second in range(4, 100):
            p.observe(sample(second, rx_samples=second, drb_context_active=1), second * NS)
            self.assertEqual(p.tick(second * NS).reason, "drb_context_restored_path_unverified")
        p.observe(sample(100, rx_samples=100, drb_context_active=0), 100 * NS)
        p.observe(sample(121, rx_samples=121, drb_context_active=0), 121 * NS)
        self.assertEqual(p.tick(121 * NS).reason, "cell_present_attempt_expired")

    def test_rrc_or_drb_without_prior_pdu_accept_does_not_prove_restoration(self):
        for values in ({"rrc_state": 2}, {"drb_context_active": 1}):
            p = RecoveryPolicy(attempt_seconds=20)
            p.begin_attempt(0)
            for second in range(1, 23):
                p.observe(sample(second, rx_samples=second, sync_successes=1, **values), second * NS)
            self.assertEqual(p.tick(22 * NS).reason, "cell_present_attempt_expired")
        p.begin_attempt(23 * NS)
        p.observe(sample(24, drb_context_active=1), 24 * NS)
        self.assertFalse(p.context_available)

    def test_reject_wait_survives_worker_and_operator_stop_wins(self):
        p = RecoveryPolicy()
        p.begin_attempt(0)
        p.observe(sample(1, nas_reject=reject(wait=37)), NS)
        self.assertEqual(p.exited(2 * NS, 0, False).not_before_ns, 38 * NS)
        self.assertEqual(p.exited(2 * NS, 0, True).reason, "operator_stop")
        p.begin_attempt(40 * NS)
        self.assertEqual(p.not_before_ns, 38 * NS)
        self.assertEqual(p.registration_failures, 1)

    def test_protocol_errors_and_fifth_failure_use_t3502(self):
        p = RecoveryPolicy()
        for attempt in range(5):
            p.begin_attempt(attempt * NS)
            p.observe(sample(attempt, nas_reject=reject()), attempt * NS)
        self.assertEqual(p.not_before_ns, 724 * NS)
        p = RecoveryPolicy()
        p.begin_attempt(0)
        p.observe(sample(1, nas_reject=reject(cause=95, t3502=0x21)), NS)
        self.assertEqual(p.not_before_ns, 61 * NS)

    def test_network_t3502_survives_later_omissions_and_worker_restarts(self):
        p = RecoveryPolicy()
        for attempt, second in enumerate((1, 12, 28, 59, 120)):
            p.begin_attempt(second * NS)
            packed = reject(t3502=0x5f if attempt == 0 else 0x42,
                            t3502_present=attempt == 0)
            p.observe(sample(second, nas_reject=packed), second * NS)
            decision = p.exited(second * NS, 0, False)
            self.assertEqual(decision.action, "retry")
            self.assertEqual(p.t3502_raw, 0x5f)
        self.assertEqual(decision.not_before_ns, 11280 * NS)
        self.assertEqual(p.snapshot()["t3502_raw"], 0x5f)

    def test_permanent_unknown_and_deactivated_rejects_block(self):
        for packed in (reject(cause=3, policy=2), reject(policy=3), reject(cause=95, t3502=0xe0)):
            p = RecoveryPolicy()
            p.begin_attempt(0)
            p.observe(sample(1, nas_reject=packed), NS)
            self.assertEqual(p.tick(100 * NS).action, "observe")
            self.assertEqual(p.exited(100 * NS, -9, False).action, "stop")

    def test_unclassified_success_exit_and_startup_failure_stop(self):
        p = RecoveryPolicy()
        p.begin_attempt(0)
        self.assertEqual(p.exited(NS, 1, False).reason, "startup_failed_without_native_telemetry")
        p.observe(sample(1, rx_samples=1), NS)
        self.assertEqual(p.exited(2 * NS, 0, False).reason, "unclassified_zero_exit")
        self.assertEqual(p.exited(2 * NS, 0, False, True).action, "retry")
        self.assertEqual(p.exited(2 * NS, -11, False).action, "retry")

    def test_stable_protocol_progress_resets_only_process_escalation(self):
        p = RecoveryPolicy()
        p.begin_attempt(0)
        p.failures = 3
        p.not_before_ns = 200 * NS
        events = []
        for second in range(1, 64):
            events.extend(p.observe(sample(second, pdu_active=1, rx_samples=second,
                                          ue_slot_inputs=second, ue_dl_completed=second,
                                          ue_tx_completed=second), second * NS))
        self.assertEqual(p.failures, 0)
        self.assertEqual(p.not_before_ns, 200 * NS)
        self.assertEqual(sum(e.get("process_backoff_reset", False) for e in events), 1)

    def test_pdu_alone_or_telemetry_gap_does_not_reset_backoff(self):
        p = RecoveryPolicy()
        p.begin_attempt(0)
        p.failures = 3
        for second in range(1, 100):
            p.observe(sample(second, pdu_active=1, rx_samples=second), second * NS)
        self.assertEqual(p.failures, 3)
        for second in list(range(100, 130)) + list(range(160, 190)):
            p.observe(sample(second, pdu_active=1, rx_samples=second,
                             ue_slot_inputs=second, ue_dl_completed=second,
                             ue_tx_completed=second), second * NS)
        self.assertEqual(p.failures, 3)

    def test_backoff_is_bounded_but_retry_count_is_not(self):
        p = RecoveryPolicy()
        for i in range(8):
            p.begin_attempt(100 * i * NS)
            p.observe(sample(100 * i, rx_samples=1), 100 * i * NS)
            d = p.exited(100 * i * NS, -11, False)
            self.assertEqual(d.not_before_ns // NS - 100 * i, (5, 15, 30, 60)[min(i, 3)])


class Channel(unittest.TestCase):
    def test_receipt_clock_is_sampled_after_each_datagram(self):
        channel = NativeChannel()
        self.addCleanup(channel.close)
        order = []
        clock = [NS]
        message = dict(kind="native_progress", schema_version=1, pid=123,
                       sequence=1, mono_ns=NS + 331, send_drops=0,
                       values={"rx_samples": 100})

        class DeliveredDuringReceive:
            def recv(self, size):
                order.append("recv")
                if clock[0] == 2 * NS:
                    raise BlockingIOError
                # A producer can send after receive() starts but before recv()
                # returns. A clock sampled before recv() would reject it.
                clock[0] = 2 * NS
                return json.dumps(message).encode()

        def receipt_clock():
            order.append("clock")
            return clock[0]

        with patch.object(channel, "reader", DeliveredDuringReceive()), \
                patch("flight_recovery.time.monotonic_ns", side_effect=receipt_clock):
            result = channel.receive(123)
        self.assertEqual(result, [message])
        self.assertEqual(channel.invalid, 0)
        self.assertEqual(order, ["recv", "clock", "recv"])

    def test_validation_and_gap_accounting(self):
        channel = NativeChannel()
        self.addCleanup(channel.close)
        def send(**overrides):
            message = dict(kind="native_progress", schema_version=1, pid=123, sequence=1,
                           mono_ns=NS, send_drops=0, values={"rx_samples":100})
            message.update(overrides)
            channel.child.send(json.dumps(message).encode())
        send(pid=124)
        send(values={"rx_samples":True})
        send(values={"unknown":1})
        send(mono_ns=10 * NS)
        send()
        send(sequence=3)
        send(sequence=2)
        result = channel.receive(123, 2 * NS)
        self.assertEqual(len(result), 2)
        self.assertEqual(channel.invalid, 5)
        self.assertEqual(channel.gaps, 1)


if __name__ == "__main__":
    unittest.main()
