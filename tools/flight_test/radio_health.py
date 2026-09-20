#!/usr/bin/env python3
"""Validation and evidence-only diagnostics for native radio-health datagrams.

The monitor socket is lossy by design.  This module therefore records what was
actually observed and deliberately represents missing, reset, or unsupported
values explicitly.  It never classifies a transport counter as a radio failure
and has no control-plane or process-control dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


NANOSECOND = 1_000_000_000
MAX_RADIO_HEALTH_DATAGRAM = 8192
MAX_DEVICE_SLOTS = 4
MAX_SUPPORTED_METRICS = 64
MAX_METRIC_NAME_BYTES = 96
MAX_BACKEND_BYTES = 96
MAX_SOURCE_AGE_NS = 5 * NANOSECOND
DIAGNOSTIC_INTERVAL_NS = NANOSECOND
MISSING_PENDING_AFTER_NS = 3 * NANOSECOND

# Current native schema v1 has this fixed catalog.  The parser intentionally
# permits another bounded numeric name so an additive producer update remains
# evidence-preserving until its semantics are documented here.
KNOWN_METRICS = frozenset(
    (
        "tx_send_calls",
        "tx_send_requested_samples",
        "tx_send_accepted_samples",
        "tx_send_short_calls",
        "tx_send_exceptions",
        "tx_send_inflight",
        "tx_sample_rate_hz",
        "tx_sample_rate_microhz",
        "tx_async_polls",
        "tx_async_messages",
        "tx_async_exceptions",
        "tx_async_time_error",
        "tx_async_underflow",
        "tx_async_underflow_in_packet",
        "tx_async_seq_error",
        "tx_async_seq_error_in_burst",
        "tx_async_burst_ack",
        "tx_async_unknown",
        "tx_async_last_poll_mono_ns",
        "tx_async_last_event_mono_ns",
        "tx_async_last_event_raw_code",
        "tx_async_last_event_channel",
        "tx_async_last_event_channel_valid",
        "tx_async_last_event_device_ticks",
        "tx_async_last_event_device_time_valid",
        "rx_recv_calls",
        "rx_recv_inflight",
        "rx_requested_samples",
        "rx_returned_samples",
        "rx_short_calls",
        "rx_zero_return_calls",
        "rx_error_none",
        "rx_error_timeout",
        "rx_error_late_command",
        "rx_error_broken_chain",
        "rx_error_overflow",
        "rx_error_alignment",
        "rx_error_bad_packet",
        "rx_error_other",
        "rx_out_of_sequence",
        "rx_timestamp_gaps",
        "rx_sample_rate_hz",
        "rx_sample_rate_microhz",
        "rx_last_error_raw_code",
        "rx_last_device_ticks",
        "rx_last_device_time_valid",
        "tx_queue_enqueues",
        "tx_queue_dequeues",
        "tx_queue_depth",
        "tx_queue_high_water",
        "tx_queue_overflow_discards",
    )
)

GAUGE_METRICS = frozenset(
    (
        "tx_sample_rate_hz", "rx_sample_rate_hz", "tx_sample_rate_microhz", "rx_sample_rate_microhz",
        "tx_send_inflight", "rx_recv_inflight",
        "tx_async_last_poll_mono_ns", "tx_async_last_event_mono_ns", "tx_async_last_event_raw_code",
        "tx_async_last_event_channel", "tx_async_last_event_channel_valid", "tx_async_last_event_device_ticks",
        "tx_async_last_event_device_time_valid", "rx_last_error_raw_code", "rx_last_device_ticks",
        "rx_last_device_time_valid", "tx_queue_depth", "tx_queue_high_water",
    )
)

# These values are producer placeholders unless their paired native validity
# gauge was observed as one. Keep invalid raw values in the snapshot only.
DERIVED_VALUE_VALIDITY_COMPANIONS = {
    "tx_async_last_event_channel": "tx_async_last_event_channel_valid",
    "tx_async_last_event_device_ticks": "tx_async_last_event_device_time_valid",
    "rx_last_device_ticks": "rx_last_device_time_valid",
}

PROGRESS_METRICS = frozenset(
    (
        "tx_send_calls",
        "tx_send_requested_samples",
        "tx_send_accepted_samples",
        "tx_async_polls",
        "rx_recv_calls",
        "rx_requested_samples",
        "rx_returned_samples",
        "tx_queue_enqueues",
        "tx_queue_dequeues",
    )
)


def _is_uint64(value: Any) -> bool:
    return type(value) is int and 0 <= value < (1 << 64)


def _valid_name(value: Any) -> bool:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > MAX_METRIC_NAME_BYTES:
        return False
    return value.isascii() and all(character.isalnum() or character == "_" for character in value)


def metric_category(name: str) -> str:
    """Return a descriptive group without assigning a fault or threshold."""

    if name.startswith("tx_sample_rate_"):
        return "tx_stream_config"
    if name.startswith("tx_send_"):
        return "tx_send"
    if name.startswith("tx_async_"):
        return "tx_async"
    if name.startswith("rx_"):
        return "rx_stream"
    if name.startswith("tx_queue_"):
        return "tx_queue"
    if name.startswith("tx_delivered"):
        return "tx_delivered"
    if name.startswith("tx_requested"):
        return "tx_requested"
    if name.startswith("tx_accepted"):
        return "tx_accepted"
    return "future_unknown_metric"


def metric_measurement_kind(name: str) -> str:
    if name in GAUGE_METRICS:
        return "gauge"
    if name in KNOWN_METRICS:
        return "counter"
    return "opaque_future_metric"


def parse_radio_health_datagram(raw: bytes, expected_pid: int, receipt_ns: int) -> dict[str, Any]:
    """Return one validated v1 snapshot or raise ``ValueError``.

    `receipt_ns` must be taken after ``recv`` returns.  The caller owns the
    socket and records malformed/truncated input as a bounded counter rather
    than exposing it to the policy.
    """

    if len(raw) > MAX_RADIO_HEALTH_DATAGRAM:
        raise ValueError("truncated_or_oversized_datagram")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid_json") from exc
    if not isinstance(value, dict):
        raise ValueError("invalid_envelope")
    if (
        value.get("kind") != "radio_health"
        or value.get("schema_version") != 1
        or type(value.get("schema_version")) is not int
        or type(value.get("pid")) is not int
        or value["pid"] != expected_pid
    ):
        raise ValueError("invalid_envelope")
    for key in ("sequence", "mono_ns", "send_drops"):
        if not _is_uint64(value.get(key)):
            raise ValueError("invalid_counter")
    if not 0 <= receipt_ns - value["mono_ns"] <= MAX_SOURCE_AGE_NS:
        raise ValueError("stale_or_future_source_clock")
    if type(value.get("device_id")) is not int or not 0 <= value["device_id"] < MAX_DEVICE_SLOTS:
        raise ValueError("invalid_device_id")
    if type(value.get("device_type")) is not int or not 0 <= value["device_type"] < (1 << 32):
        raise ValueError("invalid_device_type")
    backend = value.get("backend")
    if not isinstance(backend, str) or not backend or len(backend.encode("utf-8")) > MAX_BACKEND_BYTES:
        raise ValueError("invalid_backend")
    if type(value.get("active")) is not bool:
        raise ValueError("invalid_active")
    supported = value.get("supported")
    if (
        not isinstance(supported, list)
        or len(supported) > MAX_SUPPORTED_METRICS
        or not all(_valid_name(name) for name in supported)
        or len(set(supported)) != len(supported)
    ):
        raise ValueError("invalid_supported")
    values = value.get("values")
    if (
        not isinstance(values, dict)
        or not values.keys() <= set(supported)
        or not all(_valid_name(name) and _is_uint64(counter) for name, counter in values.items())
    ):
        raise ValueError("invalid_values")
    return {
        "kind": "radio_health",
        "schema_version": 1,
        "pid": value["pid"],
        "sequence": value["sequence"],
        "mono_ns": value["mono_ns"],
        "send_drops": value["send_drops"],
        "device_id": value["device_id"],
        "backend": backend,
        "device_type": value["device_type"],
        "active": value["active"],
        "supported": list(supported),
        "values": dict(values),
    }


def radio_sample_progress_increased(previous_values: Optional[dict[str, int]], snapshot: dict[str, Any]) -> bool:
    """Require a fresh active RX return or TX acceptance sample increase.

    Requested samples can advance before hardware accepts anything. Static
    positive values prove neither fresh work nor RF delivery, so they do not
    qualify a gNB crash for retry.
    """

    if not snapshot["active"] or previous_values is None:
        return False
    values = snapshot["values"]
    return any(
        metric in previous_values
        and metric in values
        and values[metric] > previous_values[metric]
        for metric in (
            "tx_send_accepted_samples",
            "rx_returned_samples",
        )
    )


@dataclass
class _DeviceState:
    sequence: Optional[int] = None
    source_mono_ns: Optional[int] = None
    receipt_ns: Optional[int] = None
    last_report_source_ns: Optional[int] = None
    last_report_values: dict[str, int] = field(default_factory=dict)
    last_report_supported: set[str] = field(default_factory=set)
    last_report_send_drops: Optional[int] = None
    supported: set[str] = field(default_factory=set)
    active: Optional[bool] = None
    values: dict[str, int] = field(default_factory=dict)
    pending_since_ns: Optional[int] = None
    last_missing_report_ns: Optional[int] = None


class RadioHealthDiagnostics:
    """Per-attempt, per-device evidence reducer for the collector.

    Raw accepted snapshots are emitted individually.  Delta reports are reduced
    to at most one per source second per device.  Every event is intentionally
    marked as observer-only, allowing offline analysis without granting an
    unqualified counter any restart authority.
    """

    def __init__(self) -> None:
        self.devices: dict[int, _DeviceState] = {}
        self.accepted = 0
        self.invalid = 0
        self.sequence_gaps = 0
        self.first_native_event: Optional[dict[str, int]] = None
        self.last_native_event: Optional[dict[str, int]] = None

    @staticmethod
    def _event(kind: str, **fields: Any) -> dict[str, Any]:
        event: dict[str, Any] = {
            "kind": kind,
            "candidate_action": "observe",
            "qualified": False,
        }
        event.update(fields)
        return event

    @staticmethod
    def _identity(snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            "device_id": snapshot["device_id"],
            "backend": snapshot["backend"],
            "device_type": snapshot["device_type"],
        }

    @staticmethod
    def _value_state(name: str, values: dict[str, int]) -> dict[str, Any]:
        if name not in values:
            return {"state": "supported_unobserved", "category": metric_category(name)}
        return {"state": "observed", "value": values[name], "category": metric_category(name)}

    def _interval(self, state: _DeviceState, snapshot: dict[str, Any]) -> dict[str, Any]:
        values = snapshot["values"]
        supported = set(snapshot["supported"])
        interval_ns = snapshot["mono_ns"] - state.last_report_source_ns
        metric_deltas: dict[str, dict[str, Any]] = {}
        counter_resets: list[str] = []
        for name in sorted(supported | state.last_report_supported):
            category = metric_category(name)
            measurement_kind = metric_measurement_kind(name)
            if name not in supported:
                record = {"state": "no_longer_supported", "category": category}
            elif name not in values:
                record = {"state": "supported_unobserved", "category": category}
            elif name in DERIVED_VALUE_VALIDITY_COMPANIONS:
                validity_companion = DERIVED_VALUE_VALIDITY_COMPANIONS[name]
                current_valid = values.get(validity_companion) == 1
                previous_valid = state.last_report_values.get(validity_companion) == 1
                if not current_valid:
                    record = {
                        "state": "derived_invalid_or_unavailable",
                        "change": None,
                        "category": category,
                        "validity_companion": validity_companion,
                        "current_validity": values.get(validity_companion),
                        "previous_validity": state.last_report_values.get(validity_companion),
                    }
                elif name not in state.last_report_values:
                    record = {
                        "state": "newly_observed",
                        "value": values[name],
                        "delta": None,
                        "category": category,
                    }
                elif not previous_valid:
                    record = {
                        "state": "gauge_observed_without_valid_previous",
                        "value": values[name],
                        "change": None,
                        "category": category,
                        "validity_companion": validity_companion,
                    }
                else:
                    record = {
                        "state": "gauge_observed",
                        "value": values[name],
                        "previous_value": state.last_report_values[name],
                        "change": values[name] - state.last_report_values[name],
                        "category": category,
                    }
            elif name not in state.last_report_values:
                record = {"state": "newly_observed", "value": values[name], "delta": None, "category": category}
            elif measurement_kind == "gauge":
                record = {"state": "gauge_observed", "value": values[name], "previous_value": state.last_report_values[name], "change": values[name] - state.last_report_values[name], "category": category}
            elif measurement_kind == "opaque_future_metric":
                record = {"state": "opaque_observed", "value": values[name], "previous_value": state.last_report_values[name], "delta": None, "category": category}
            elif values[name] < state.last_report_values[name]:
                record = {"state": "counter_decreased_or_reset", "value": values[name], "previous_value": state.last_report_values[name], "delta": None, "category": category}
                counter_resets.append(name)
            else:
                record = {"state": "counter_observed", "value": values[name], "delta": values[name] - state.last_report_values[name], "category": category}
            record["measurement_kind"] = measurement_kind
            metric_deltas[name] = record
        send_drop_delta = None if state.last_report_send_drops is None or snapshot["send_drops"] < state.last_report_send_drops else snapshot["send_drops"] - state.last_report_send_drops
        tx_progress = {
            "requested_samples": self._value_state("tx_send_requested_samples", values),
            "accepted_samples": self._value_state("tx_send_accepted_samples", values),
            "delivered_samples": self._value_state("tx_delivered_samples", values) if "tx_delivered_samples" in supported else {"state": "unsupported_by_native_schema", "category": "tx_delivered"},
        }
        return self._event("radio_health_interval", device=self._identity(snapshot), source_sequence=snapshot["sequence"], source_clock={"monotonic_ns": snapshot["mono_ns"], "utc_wall_state": "unavailable_from_native_abi"}, interval_duration_ns=interval_ns, source_send_drop_delta=send_drop_delta, metric_deltas=metric_deltas, metric_categories=sorted({metric_category(name) for name in supported}), counter_decreased_or_reset_metrics=counter_resets, tx_progress=tx_progress)

    @staticmethod
    def _pending_work(metric_deltas: dict[str, dict[str, Any]], values: dict[str, int]) -> dict[str, Any]:
        def known_counter_delta(name: str) -> Optional[int]:
            record = metric_deltas.get(name)
            if record is None or record.get("state") != "counter_observed":
                return None
            return record["delta"]

        tx_requested_samples = known_counter_delta("tx_send_requested_samples")
        tx_accepted_samples = known_counter_delta("tx_send_accepted_samples")
        tx_queue_enqueues = known_counter_delta("tx_queue_enqueues")
        tx_queue_dequeues = known_counter_delta("tx_queue_dequeues")
        rx_returned_samples = known_counter_delta("rx_returned_samples")
        inflight = {
            name: values[name]
            for name in ("tx_send_inflight", "rx_recv_inflight")
            if values.get(name) == 1
        }
        queue_depth = values.get("tx_queue_depth")
        reasons: list[str] = []
        if values.get("tx_send_inflight") == 1 and tx_accepted_samples == 0:
            reasons.append("tx_send_inflight_without_known_accepted_sample_progress")
        if values.get("rx_recv_inflight") == 1 and rx_returned_samples == 0:
            reasons.append("rx_recv_inflight_without_known_rx_return_progress")
        if tx_requested_samples is not None and tx_requested_samples > 0 and tx_accepted_samples == 0:
            reasons.append("requested_samples_without_known_accepted_sample_progress")
        if tx_queue_enqueues is not None and tx_queue_enqueues > 0 and tx_queue_dequeues == 0:
            reasons.append("enqueued_items_without_known_queue_dequeue_progress")
        if queue_depth is not None and queue_depth > 0 and tx_queue_dequeues == 0:
            reasons.append("queue_depth_without_known_queue_dequeue_progress")
        return {
            "pending": bool(reasons),
            "reasons": reasons,
            "inflight": inflight,
            "tx_send_requested_samples_delta": tx_requested_samples,
            "tx_send_accepted_samples_delta": tx_accepted_samples,
            "tx_queue_enqueues_delta": tx_queue_enqueues,
            "tx_queue_dequeues_delta": tx_queue_dequeues,
            "rx_returned_samples_delta": rx_returned_samples,
        }

    def observe(self, snapshot: dict[str, Any], receipt_ns: int) -> list[dict[str, Any]]:
        """Accept one parsed snapshot and return bounded evidence events."""

        device_id = snapshot["device_id"]
        state = self.devices.setdefault(device_id, _DeviceState())
        if state.sequence is not None and snapshot["sequence"] <= state.sequence:
            self.invalid += 1
            return [
                self._event(
                    "radio_health_rejected",
                    reason="stale_or_unordered_device_sequence",
                    device=self._identity(snapshot),
                    source_sequence=snapshot["sequence"],
                )
            ]
        if state.source_mono_ns is not None and snapshot["mono_ns"] <= state.source_mono_ns:
            self.invalid += 1
            return [
                self._event(
                    "radio_health_rejected",
                    reason="stale_or_unordered_device_clock",
                    device=self._identity(snapshot),
                    source_sequence=snapshot["sequence"],
                )
            ]

        events = [
            self._event(
                "radio_health_snapshot",
                device=self._identity(snapshot),
                source_sequence=snapshot["sequence"],
                source_clock={"monotonic_ns": snapshot["mono_ns"], "utc_wall_state": "unavailable_from_native_abi"},
                source_send_drops=snapshot["send_drops"],
                active=snapshot["active"],
                native_snapshot=snapshot,
            )
        ]
        if state.sequence is not None and snapshot["sequence"] > state.sequence + 1:
            gap = snapshot["sequence"] - state.sequence - 1
            self.sequence_gaps += gap
            events.append(
                self._event(
                    "radio_health_source_gap",
                    device=self._identity(snapshot),
                    first_missing_sequence=state.sequence + 1,
                    last_missing_sequence=snapshot["sequence"] - 1,
                    missing_count=gap,
                    time_span={"previous_source_mono_ns": state.source_mono_ns, "next_source_mono_ns": snapshot["mono_ns"]},
                )
            )
        current_supported = set(snapshot["supported"])
        if state.active is None:
            events.append(self._event("radio_health_lifecycle", event="first_native_event", device=self._identity(snapshot), active=snapshot["active"]))
        elif state.active != snapshot["active"]:
            events.append(
                self._event(
                    "radio_health_lifecycle",
                    event="active" if snapshot["active"] else "closed",
                    device=self._identity(snapshot),
                    active=snapshot["active"],
                    time_span={"previous_source_mono_ns": state.source_mono_ns, "next_source_mono_ns": snapshot["mono_ns"]},
                )
            )
        if state.supported and current_supported != state.supported:
            events.append(
                self._event(
                    "radio_health_lifecycle",
                    event="supported_metrics_changed",
                    device=self._identity(snapshot),
                    added_supported=sorted(current_supported - state.supported),
                    removed_supported=sorted(state.supported - current_supported),
                )
            )

        if state.last_report_source_ns is not None and snapshot["mono_ns"] - state.last_report_source_ns >= DIAGNOSTIC_INTERVAL_NS:
            interval = self._interval(state, snapshot)
            metric_deltas = interval["metric_deltas"]
            pending = self._pending_work(metric_deltas, snapshot["values"])
            interval["pending_work"] = pending
            events.append(interval)
            if pending["pending"]:
                if state.pending_since_ns is None:
                    state.pending_since_ns = snapshot["mono_ns"]
                if snapshot["mono_ns"] - state.pending_since_ns >= DIAGNOSTIC_INTERVAL_NS:
                    events.append(self._event("radio_health_observation", reason="pending_work_without_observed_transport_progress", device=self._identity(snapshot), time_span={"started_source_mono_ns": state.pending_since_ns, "ended_source_mono_ns": snapshot["mono_ns"], "observed_duration_ns": snapshot["mono_ns"] - state.pending_since_ns}, pending_evidence=pending, transport_evidence="observational_only_no_restart_qualification"))
            else:
                state.pending_since_ns = None
            state.last_report_source_ns = snapshot["mono_ns"]
            state.last_report_values = dict(snapshot["values"])
            state.last_report_supported = current_supported
            state.last_report_send_drops = snapshot["send_drops"]
        elif state.last_report_source_ns is None:
            state.last_report_source_ns = snapshot["mono_ns"]
            state.last_report_values = dict(snapshot["values"])
            state.last_report_supported = current_supported
            state.last_report_send_drops = snapshot["send_drops"]

        state.sequence = snapshot["sequence"]
        state.source_mono_ns = snapshot["mono_ns"]
        state.receipt_ns = receipt_ns
        state.supported = current_supported
        state.active = snapshot["active"]
        state.values = dict(snapshot["values"])
        identity = {"device_id": device_id, "sequence": snapshot["sequence"], "mono_ns": snapshot["mono_ns"]}
        if self.first_native_event is None:
            self.first_native_event = identity
        self.last_native_event = identity
        self.accepted += 1
        return events

    def tick(self, receipt_ns: int) -> list[dict[str, Any]]:
        """Record a real data absence with pending work; never request action."""

        events: list[dict[str, Any]] = []
        for device_id, state in self.devices.items():
            if (
                not state.active
                or state.receipt_ns is None
                or state.pending_since_ns is None
                or receipt_ns - state.receipt_ns < MISSING_PENDING_AFTER_NS
                or (
                    state.last_missing_report_ns is not None
                    and receipt_ns - state.last_missing_report_ns < DIAGNOSTIC_INTERVAL_NS
                )
            ):
                continue
            state.last_missing_report_ns = receipt_ns
            events.append(
                self._event(
                    "radio_health_observation",
                    reason="radio_health_missing_with_pending_work",
                    device={"device_id": device_id},
                    time_span={
                        "last_receipt_monotonic_ns": state.receipt_ns,
                        "observation_monotonic_ns": receipt_ns,
                        "missing_duration_ns": receipt_ns - state.receipt_ns,
                    },
                    transport_evidence="observational_only_no_restart_qualification",
                )
            )
        return events

    def snapshot(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "invalid": self.invalid,
            "sequence_gaps": self.sequence_gaps,
            "first_native_event": self.first_native_event,
            "last_native_event": self.last_native_event,
            "devices": {
                str(device_id): {
                    "active": state.active,
                    "last_source_sequence": state.sequence,
                    "last_source_monotonic_ns": state.source_mono_ns,
                    "supported_metrics": sorted(state.supported),
                    "observed_metrics": sorted(state.values),
                    "pending_work_since_source_mono_ns": state.pending_since_ns,
                }
                for device_id, state in sorted(self.devices.items())
            },
        }
