#!/usr/bin/env python3
"""Flight-only recovery policy. Protocol restrictions outlive modem workers.

No policy input is inferred from text logs, ping, or interface existence.
The native datagram stream is independent of the recorder and its disk writer.
"""

from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from typing import Any, Optional

from radio_health import MAX_RADIO_HEALTH_DATAGRAM, parse_radio_health_datagram, radio_sample_progress_increased

NANOSECOND = 1_000_000_000
MAX_DATAGRAM = MAX_RADIO_HEALTH_DATAGRAM
FIELDS = frozenset((
    "rx_samples", "tx_samples", "search_attempts", "sync_successes",
    "rrc_messages", "nas_messages", "rrc_state", "pdu_accepts", "pdu_active",
    "nas_reject", "rrc_hold_until_ns", "ue_slot_inputs", "ue_dl_completed", "ue_tx_completed",
    "drb_context_active",
))


class NativeChannel:
    """One private socketpair per worker, with bounded, validated input."""

    def __init__(self) -> None:
        self.reader, self.child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.reader.setblocking(False)
        self.sequence = -1
        self.invalid = 0
        self.gaps = 0
        self.latest: Optional[dict[str, Any]] = None
        self.radio_health: list[dict[str, Any]] = []
        self.radio_health_invalid = 0
        self.radio_health_received = 0

    def receive(self, pid: int, now_ns: Optional[int] = None) -> list[dict[str, Any]]:
        # Production validates against a clock sampled after recv. An explicit
        # clock is reserved for deterministic policy/channel fixtures.
        result = []
        for _ in range(32):
            value: Any = None
            try:
                raw = self.reader.recv(MAX_DATAGRAM + 1)
                receipt_ns = time.monotonic_ns() if now_ns is None else now_ns
            except BlockingIOError:
                break
            try:
                value = json.loads(raw)
                if isinstance(value, dict) and value.get("kind") == "radio_health":
                    snapshot = parse_radio_health_datagram(raw, pid, receipt_ns)
                    self.radio_health.append(snapshot)
                    self.radio_health_received += 1
                    continue
                if (len(raw) > MAX_DATAGRAM or not isinstance(value, dict)
                        or value.get("kind") != "native_progress"
                        or type(value.get("schema_version")) is not int or value["schema_version"] != 1
                        or type(value.get("pid")) is not int or value["pid"] != pid):
                    raise ValueError("invalid envelope")
                for key in ("sequence", "mono_ns", "send_drops"):
                    if type(value[key]) is not int or not 0 <= value[key] < 1 << 64:
                        raise ValueError("invalid counter")
                if (value["sequence"] <= self.sequence
                        or not 0 <= receipt_ns - value["mono_ns"] <= 5 * NANOSECOND):
                    raise ValueError("stale or unordered sample")
                fields = value["values"]
                if (not isinstance(fields, dict) or not fields.keys() <= FIELDS
                        or not all(type(v) is int and 0 <= v < 1 << 64 for v in fields.values())):
                    raise ValueError("invalid fields")
            except (ValueError, KeyError, TypeError):
                self.invalid += 1
                if isinstance(value, dict) and value.get("kind") == "radio_health":
                    self.radio_health_invalid += 1
                continue
            if self.sequence >= 0:
                self.gaps += value["sequence"] - self.sequence - 1
            self.sequence = value["sequence"]
            self.latest = value
            result.append(value)
        return result

    def take_radio_health(self) -> list[dict[str, Any]]:
        """Return parsed radio-health snapshots after one socket drain."""

        snapshots = self.radio_health
        self.radio_health = []
        return snapshots

    def close(self) -> None:
        for stream in (self.reader, self.child):
            try:
                stream.close()
            except OSError:
                pass


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    not_before_ns: int = 0


class RecoveryPolicy:
    """One session's retry restrictions and one worker's measured progress.

    nas_reject is one atomic word: generation[63:56], cause[55:48],
    policy/presence[47:40], T3502 timer octet[39:32], wait_seconds[31:0]. Policy 1 permits a
    cause-aware retry with preserved timer lower bounds; 2 blocks retry; 3 marks unsupported/malformed
    restrictions (also blocks). No packet contents or subscription IDs enter it.
    """

    def __init__(self, stall_seconds: float = 10.0, attempt_seconds: float = 120.0, role: str = "ue") -> None:
        if role not in ("ue", "gnb"):
            raise ValueError("role must be ue or gnb")
        self.role = role
        self.stall_ns = int(stall_seconds * NANOSECOND)
        self.attempt_ns = int(attempt_seconds * NANOSECOND)
        self.not_before_ns = 0
        self.blocked_reason: Optional[str] = None
        self.failures = 0
        self.registration_failures = 0
        self.t3502_raw = 0x42  # Session-learned timer survives worker replacement.
        self.last_reject_cause: Optional[int] = None
        self.generation = 0
        self._reset_attempt(0)

    def begin_attempt(self, now_ns: int) -> None:
        self.generation += 1
        self._reset_attempt(now_ns)

    def _reset_attempt(self, now_ns: int) -> None:
        self.started_ns = now_ns
        self.last_sample_ns: Optional[int] = None
        self.rx_changed_ns: Optional[int] = None
        self.cell_attempt_ns: Optional[int] = None
        self.rx_samples: Optional[int] = None
        self.search_attempts = 0
        self.sync_successes = 0
        self.pdu_accepts = 0
        self.pdu_active = False
        self.context_available = False
        self.last_reject_word = 0
        self.values: dict[str, int] = {}
        self.stall_suspected_ns: Optional[int] = None
        self.rrc_hold_ns = 0
        self.retryable_reject = False
        self.slot_inputs = 0
        self.input_changed_ns: Optional[int] = None
        self.completions: dict[str, tuple[int, int]] = {}
        self.worker_suspected_ns: dict[str, int] = {}
        self.stable_progress_since_ns: Optional[int] = None
        self.radio_health_initialized = False
        self.radio_progress_observed = False
        self.radio_progress_observed_ns: Optional[int] = None
        self.radio_health_values: dict[int, dict[str, int]] = {}
        self.radio_health_source_ns: dict[int, int] = {}

    def observe_radio_health(self, snapshot: dict[str, Any], now_ns: int) -> list[dict[str, Any]]:
        """Record gNB crash-retry eligibility without deriving transport faults."""

        if self.role != "gnb":
            return []
        device_id = snapshot["device_id"]
        if not snapshot["active"]:
            self.radio_health_values.pop(device_id, None)
            self.radio_health_source_ns.pop(device_id, None)
            return []
        self.radio_health_initialized = True
        previous_values = self.radio_health_values.get(device_id)
        previous_source_ns = self.radio_health_source_ns.get(device_id)
        fresh_pair = previous_source_ns is not None and 0 < snapshot["mono_ns"] - previous_source_ns <= 3 * NANOSECOND
        self.radio_health_values[device_id] = dict(snapshot["values"])
        self.radio_health_source_ns[device_id] = snapshot["mono_ns"]
        if fresh_pair and radio_sample_progress_increased(previous_values, snapshot):
            was_observed = self.radio_progress_observed
            self.radio_progress_observed = True
            self.radio_progress_observed_ns = now_ns
            if not was_observed:
                return [{
                    "event": "positive_radio_progress_observed",
                    "candidate_action": "observe",
                    "qualified": False,
                }]
        return []

    def observe(self, sample: dict[str, Any], now_ns: int) -> list[dict[str, Any]]:
        if self.role == "gnb":
            # gNB has no UE NAS, DRB, attach or tunnel deadline. Only positive
            # radio health permits a later unexpected-crash retry.
            self.last_sample_ns = sample["mono_ns"]
            self.values = sample["values"]
            return []
        changes = []
        v = sample["values"]
        previous_values = self.values
        previous_sample_ns = self.last_sample_ns
        self.last_sample_ns = sample["mono_ns"]
        self.values = v
        rx = v.get("rx_samples")
        if rx is not None and rx != self.rx_samples:
            self.rx_samples = rx
            self.rx_changed_ns = now_ns
            self.stall_suspected_ns = None
        inputs = v.get("ue_slot_inputs", self.slot_inputs)
        if inputs > self.slot_inputs:
            self.input_changed_ns = now_ns
        self.slot_inputs = inputs
        for field in ("ue_dl_completed", "ue_tx_completed"):
            if field in v:
                previous = self.completions.get(field)
                if previous is None or previous[0] != v[field]:
                    self.completions[field] = (v[field], now_ns)
                    self.worker_suspected_ns.pop(field, None)
        self.search_attempts = v.get("search_attempts", self.search_attempts)
        sync = v.get("sync_successes", self.sync_successes)
        if sync > self.sync_successes and self.cell_attempt_ns is None:
            self.cell_attempt_ns = now_ns
            changes.append({"event": "cell_acquired", "count": sync})
        self.sync_successes = sync
        accepts = v.get("pdu_accepts", self.pdu_accepts)
        if accepts > self.pdu_accepts:
            changes.append({"event": "pdu_accepted", "count": accepts})
            self.registration_failures = 0
            self.last_reject_cause = None
        self.pdu_accepts = accepts
        previous_context = self.context_available
        self.pdu_active = bool(v.get("pdu_active", 0))
        # Reestablishment/service recovery can resume an accepted session without
        # another PDU Session Establishment Accept. Require an actual configured
        # or resumed DRB plus this worker's earlier NAS acceptance. RRC_CONNECTED
        # and PHY progress alone do not establish a user-plane context.
        self.context_available = self.pdu_active or (
            self.pdu_accepts > 0 and bool(v.get("drb_context_active", 0)))
        if not previous_context and self.context_available and not self.pdu_active:
            changes.append({"event": "drb_context_restored", "application_delivery_verified": False})
        if previous_context and not self.context_available:
            self.cell_attempt_ns = now_ns
            changes.append({"event": "pdu_path_lost"})
        if self.context_available:
            self.cell_attempt_ns = None
        self.rrc_hold_ns = v.get("rrc_hold_until_ns", 0)
        packed = v.get("nas_reject", 0)
        if packed and packed != self.last_reject_word:
            self.last_reject_word = packed
            cause = (packed >> 48) & 255
            policy_byte = (packed >> 40) & 255
            policy = policy_byte & 0x7f
            t3502_present = bool(policy_byte & 0x80)
            if t3502_present:
                self.t3502_raw = (packed >> 32) & 255
            wait = packed & 0xFFFFFFFF
            self.last_reject_cause = cause
            if policy == 1:
                self.retryable_reject = True
                self.registration_failures += 1
                # TS 24.501 5.5.1.2.7: selected protocol errors exhaust
                # the short retry counter; the fifth failure starts T3502.
                if cause in (95, 96, 97, 99, 111):
                    self.registration_failures = 5
                if self.registration_failures >= 5:
                    encoded = self.t3502_raw
                    unit, value = encoded >> 5, encoded & 31
                    factor = 2 if unit == 0 else 360 if unit == 2 else 60
                    if unit == 7:
                        self.blocked_reason = "unsupported_deactivated_t3502"
                    wait = max(wait, value * factor)
                    self.registration_failures = 0
                self.not_before_ns = max(self.not_before_ns, now_ns + wait * NANOSECOND)
            else:
                self.blocked_reason = f"nas_reject_{cause}_policy_{policy}"
            changes.append({"event": "nas_reject", "cause": cause, "policy": policy,
                            "wait_seconds": wait, "not_before_ns": self.not_before_ns,
                            "registration_failures": self.registration_failures,
                            "t3502_present": t3502_present, "session_t3502_raw": self.t3502_raw})
        # Reset process-crash escalation only after a minute of continuously
        # observed protocol/PHY progress. This does not prove application
        # delivery and never clears a network restriction or NAS retry counter.
        progress_fields = ("rx_samples", "ue_slot_inputs", "ue_dl_completed", "ue_tx_completed")
        progressing = (previous_context and self.context_available and previous_sample_ns is not None
                       and 0 < sample["mono_ns"] - previous_sample_ns <= 3 * NANOSECOND
                       and all(field in previous_values and v.get(field, 0) > previous_values[field]
                               for field in progress_fields))
        if not progressing:
            self.stable_progress_since_ns = None
        elif self.stable_progress_since_ns is None:
            self.stable_progress_since_ns = now_ns
        elif self.failures and now_ns - self.stable_progress_since_ns >= 60 * NANOSECOND:
            previous_failures = self.failures
            self.failures = 0
            changes.append({"event": "stable_protocol_progress", "process_backoff_reset": True,
                            "previous_process_failures": previous_failures,
                            "observed_duration_ns": now_ns - self.stable_progress_since_ns})
        return changes

    def tick(self, now_ns: int) -> Decision:
        if self.role == "gnb":
            if self.radio_progress_observed:
                return Decision("observe", "gnb_radio_progress_observed")
            if self.radio_health_initialized:
                return Decision("observe", "gnb_radio_initialized_without_positive_progress")
            return Decision("observe", "gnb_radio_health_unavailable")
        if self.blocked_reason:
            return Decision("observe", self.blocked_reason)
        if self.last_sample_ns is None or now_ns - self.last_sample_ns > 3 * NANOSECOND:
            # Telemetry absence is not proof that radio processing has stopped.
            return Decision("observe", "native_telemetry_unavailable")
        if self.rx_changed_ns is not None and now_ns - self.rx_changed_ns >= self.stall_ns:
            if self.stall_suspected_ns is None:
                self.stall_suspected_ns = now_ns
            elif now_ns - self.stall_suspected_ns >= NANOSECOND:
                return Decision("restart", "radio_rx_progress_stalled")
        if self.input_changed_ns is not None and now_ns - self.input_changed_ns <= 3 * NANOSECOND:
            for field, (_, changed_ns) in self.completions.items():
                if now_ns - changed_ns >= self.stall_ns:
                    suspected = self.worker_suspected_ns.setdefault(field, now_ns)
                    if now_ns - suspected >= NANOSECOND:
                        return Decision("restart", field + "_progress_stalled")
        else:
            self.worker_suspected_ns.clear()
        if now_ns < max(self.not_before_ns, self.rrc_hold_ns):
            return Decision("observe", "protocol_backoff", max(self.not_before_ns, self.rrc_hold_ns))
        if (not self.context_available and self.cell_attempt_ns is not None
                and now_ns - self.cell_attempt_ns >= self.attempt_ns):
            return Decision("restart", "cell_present_attempt_expired")
        if self.pdu_active:
            return Decision("observe", "pdu_established_path_unverified")
        return Decision("observe", "drb_context_restored_path_unverified" if self.context_available else "acquiring")

    def exited(self, now_ns: int, returncode: int, operator_stop: bool, controlled_recovery: bool = False) -> Decision:
        if operator_stop:
            return Decision("stop", "operator_stop")
        if self.role == "gnb":
            if returncode == 0:
                return Decision("stop", "unclassified_zero_exit")
            if not self.radio_progress_observed:
                return Decision("stop", "startup_failed_without_positive_radio_progress")
            self.failures += 1
            delay = (5, 15, 30, 60)[min(self.failures - 1, 3)]
            return Decision("retry", f"unexpected_nonzero_exit_{returncode}_after_positive_radio_progress", now_ns + delay * NANOSECOND)
        if self.blocked_reason:
            return Decision("stop", self.blocked_reason)
        if self.last_sample_ns is None:
            return Decision("stop", "startup_failed_without_native_telemetry")
        if returncode == 0 and not (self.retryable_reject or controlled_recovery):
            return Decision("stop", "unclassified_zero_exit")
        self.failures += 1
        delay = (5, 15, 30, 60)[min(self.failures - 1, 3)]
        deadline = max(self.not_before_ns, self.rrc_hold_ns, now_ns + delay * NANOSECOND)
        return Decision("retry", f"unrequested_exit_{returncode}", deadline)

    def snapshot(self) -> dict[str, Any]:
        return {"role": self.role, "generation": self.generation, "failures": self.failures,
                "registration_failures": self.registration_failures, "t3502_raw": self.t3502_raw,
                "not_before_ns": self.not_before_ns, "blocked_reason": self.blocked_reason,
                "last_reject_cause": self.last_reject_cause, "native_values": self.values,
                "radio_health_initialized": self.radio_health_initialized,
                "radio_progress_observed": self.radio_progress_observed,
                "radio_progress_observed_ns": self.radio_progress_observed_ns}
