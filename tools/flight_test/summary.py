#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Bounded live session totals and offline capture facts, without inferring outages."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any, Iterator, Optional


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _traffic_outcome(event: dict[str, Any]) -> Optional[str]:
    if event.get("kind") == "traffic_launch_failed":
        return "failed"
    if event.get("kind") != "traffic_generation_finished":
        return None
    outcome = event.get("outcome")
    if outcome in ("completed", "stopped", "failed"):
        return outcome
    # Older recordings have only completed/reason/forced_kill. Keep their
    # reported stop reason; an absent kill field cannot prove a clean stop.
    if event.get("completed"):
        return "completed"
    if (event.get("reason") == "main_worker_stopping" and event.get("forced_kill") is False
            and not event.get("cleanup_failed")):
        return "stopped"
    return "failed"


class _DirectionSummary:
    """One ordered iperf stream per direction; old/overlapping reports are excluded."""

    def __init__(self) -> None:
        self.reports = self.positive_reports = self.accounted = self.ambiguous = 0
        self.duplicates = self.overlaps = self.loss_reports = 0
        self.received_bytes = self.covered_seconds = 0.0
        self.maximum_bytes: Optional[float] = None
        self.lost = self.datagrams = 0
        self.first_report_ns: Optional[int] = None
        self.first_positive_ns: Optional[int] = None
        self.last_report_ns: Optional[int] = None
        self.gaps = 0
        self.gap_seconds = self.longest_gap = 0.0
        self.key: Optional[tuple[int, int, int]] = None
        self.stream: Optional[int] = None
        self.end = 0.0

    def observe(self, event: dict[str, Any], attempt: int, now_ns: int) -> None:
        row = event["interval"]
        value, start, end = row.get("bytes"), row.get("start_seconds"), row.get("end_seconds")
        if not all(_number(v) for v in (value, start, end)) or end <= start:
            return
        self.reports += 1
        self.maximum_bytes = max(self.maximum_bytes or 0, value)
        self.positive_reports += value > 0
        if self.first_report_ns is None:
            self.first_report_ns = now_ns
        if value > 0 and self.first_positive_ns is None:
            self.first_positive_ns = now_ns
        generation, connection, stream = event.get("generation"), event.get("connection_index"), row.get("stream_id")
        if (type(generation) is not int or generation <= 0 or type(connection) is not int or connection <= 0
                or type(stream) is not int or stream < 0):
            self.ambiguous += 1
            return
        key = (attempt, generation, connection)
        if self.key is not None and key < self.key:
            self.duplicates += 1
            return
        if key != self.key:
            self.key, self.stream, self.end = key, stream, 0.0
        if stream != self.stream:
            # Production requests one stream per direction. Do not silently sum
            # unexpected parallel streams whose time coverage would overlap.
            self.ambiguous += 1
            return
        if end <= self.end:
            self.duplicates += 1
            return
        previous_end = self.end
        self.end = end
        if start < previous_end:
            self.overlaps += 1
            return
        self.accounted += 1
        self.received_bytes += value
        self.covered_seconds += end - start
        lost, total = row.get("lost_datagrams"), row.get("total_datagrams")
        if type(lost) is int and type(total) is int and 0 <= lost <= total:
            self.loss_reports += 1
            self.lost += lost
            self.datagrams += total
        if self.last_report_ns is not None and now_ns > self.last_report_ns:
            gap = (now_ns - self.last_report_ns) / 1e9 - 1.0
            if gap > 0.5:
                self.gaps += 1
                self.gap_seconds += gap
                self.longest_gap = max(self.longest_gap, gap)
        self.last_report_ns = max(self.last_report_ns or now_ns, now_ns)

    def snapshot(self, start_ns: int, end_ns: int) -> dict[str, Any]:
        def elapsed(value: Optional[int]) -> Optional[float]:
            return max(0, value - start_ns) / 1e9 if value is not None else None
        return {
            "delivery": "observed_positive" if self.positive_reports else "observed_zero" if self.reports else "unavailable",
            "receiver_reports": self.reports, "accounted_intervals": self.accounted,
            "ambiguous_receiver_reports": self.ambiguous, "duplicate_or_old_intervals": self.duplicates,
            "overlapping_intervals_excluded": self.overlaps,
            "received_bytes": self.received_bytes if self.accounted else None,
            "covered_seconds": self.covered_seconds if self.accounted else None,
            "received_bits_per_second": self.received_bytes * 8 / self.covered_seconds if self.covered_seconds else None,
            "lost_datagrams": self.lost if self.loss_reports else None,
            "total_datagrams": self.datagrams if self.loss_reports else None,
            "loss_percent": 100 * self.lost / self.datagrams if self.datagrams else None,
            "intervals_with_loss_fields": self.loss_reports,
            "first_receiver_report_seconds": elapsed(self.first_report_ns),
            "first_positive_receiver_report_seconds": elapsed(self.first_positive_ns),
            "reporting_gaps": self.gaps if self.accounted >= 2 else None,
            "reporting_gap_seconds": self.gap_seconds if self.accounted >= 2 else None,
            "longest_reporting_gap_seconds": self.longest_gap if self.accounted >= 2 else None,
            "trailing_unobserved_seconds": max(0, end_ns - self.last_report_ns) / 1e9 if self.last_report_ns else None,
        }


class AttemptTimeline:
    """Fixed-size observer state; no delivery or exact protocol-time inference."""

    RADIO_COUNTERS = (
        "tx_async_time_error", "tx_async_underflow", "tx_async_underflow_in_packet",
        "tx_async_seq_error", "tx_async_seq_error_in_burst",
        "tx_send_short_calls", "tx_send_exceptions", "tx_async_exceptions",
        "rx_short_calls", "rx_error_overflow", "rx_out_of_sequence", "rx_timestamp_gaps",
        "tx_queue_overflow_discards",
    )

    def __init__(self, start_ns: int, role: str) -> None:
        self.start_ns = start_ns
        self.role = role
        self._counters: dict[str, int] = {}
        self._pdu_accepted = False
        self._bearer: Optional[str] = None
        self._had_ready = False

    def _line(self, pid: int, receipt_ns: int, text: str, phase: str) -> str:
        elapsed = max(0, receipt_ns - self.start_ns) / 1e9
        return f"[FLIGHT OBSERVED] worker_pid={pid} monotonic_elapsed={elapsed:.3f}s phase={phase} {text}\n"

    def native_lines(self, sample: dict[str, Any], pid: int, receipt_ns: int, *, phase: str = "running") -> list[str]:
        """Describe only explicit UE observations from the private worker channel."""
        if self.role != "ue":
            return []
        values = sample["values"]
        lines = []
        for name in ("sync_successes", "pdu_accepts"):
            value = values.get(name)
            if type(value) is not int or value < 0:
                continue
            previous = self._counters.get(name)
            self._counters[name] = value
            if name == "pdu_accepts" and value > 0:
                self._pdu_accepted = True
            if previous is not None and value < previous:
                text = f"{name} counter decreased/reset previous={previous} current={value}; increment=unknown"
                lines.append(self._line(pid, receipt_ns, text, phase))
            elif value > (previous or 0):
                initial = " (first observed counter)" if previous is None else ""
                lines.append(self._line(pid, receipt_ns, f"{name} +{value - (previous or 0)} total={value}{initial}", phase))

        active = values.get("drb_context_active")
        if type(active) is not int or active < 0:
            return lines
        # A missing DRB observation never changes the last observed state. TUN
        # configuration (pdu_active) is not evidence of a configured bearer.
        bearer = "ready" if active > 0 and self._pdu_accepted else "unknown" if active > 0 else "inactive"
        if bearer != self._bearer:
            if bearer == "ready":
                text = "data bearer restored" if self._had_ready else "data bearer ready"
                text += " (PDU acceptance and active DRB observed)"
                self._had_ready = True
            elif bearer == "inactive":
                text = "data bearer lost (DRB context inactive)" if self._bearer == "ready" else "DRB context inactive"
                if self._bearer == "ready" and phase == "requested_shutdown":
                    text = "data bearer inactive during requested shutdown"
                if self._bearer is None:
                    text += " (initial observation)"
            else:
                text = "DRB context active; bearer readiness=unknown (PDU acceptance unobserved)"
            lines.append(self._line(pid, receipt_ns, text + "; delivery=unavailable", phase))
            self._bearer = bearer
        return lines

    def radio_line(self, event: dict[str, Any], pid: int, receipt_ns: int, *, phase: str = "running") -> Optional[str]:
        """Format known positive deltas or explicit resets from the reduced interval."""
        if event.get("kind") != "radio_health_interval":
            return None
        metrics = event["metric_deltas"]
        positive = []
        unknown: dict[str, list[str]] = {}
        for name in self.RADIO_COUNTERS:
            metric = metrics.get(name)
            if metric is None:
                continue
            state, delta = metric["state"], metric.get("delta")
            if state == "counter_observed" and type(delta) is int and delta > 0:
                positive.append(f"{name}=+{delta}")
            elif state != "counter_observed":
                unknown.setdefault(state, []).append(name)
        if not positive and "counter_decreased_or_reset" not in unknown:
            return None
        device = event["device"]
        # Backend names originate in a datagram; quote control characters so a
        # bounded observation remains one ASCII log line.
        backend = json.dumps(device["backend"], ensure_ascii=True)
        text = (f"radio device={device['device_id']} backend={backend} device_type={device['device_type']} "
                f"source_sequence={event['source_sequence']} source_interval={event['interval_duration_ns'] / 1e9:.3f}s "
                "observed_counter_deltas: " + (" ".join(positive) if positive else "no known positive delta"))
        for state, names in unknown.items():
            text += f"; delta_unknown[{state}]=" + ",".join(names)
        return self._line(pid, receipt_ns, text + "; transport counters only; delivery=unavailable", phase)


class SessionSummary:
    """Fixed-size live totals, independent of retry count and raw log size."""

    def __init__(self, start_ns: int) -> None:
        self.start_ns = start_ns
        self.attempts_started = self.attempts_finished = self.worker_starts = 0
        self.directions = {name: _DirectionSummary() for name in ("ul", "dl")}
        self.first_pdu_ns: Optional[int] = None
        self.sync_successes = self.sync_reacquisitions = self.context_restorations = self.context_losses = 0
        self.pdu_accepts = self.native_counter_resets = self.native_send_drops = 0
        self.native_samples = self.traffic_gaps = self.traffic_failures = self.partial_output = 0
        self.traffic_stops = 0
        self.sync_samples = self.context_samples = self.send_drop_samples = self.pdu_samples = 0
        self.integrity_counts = {name: 0 for name in ("complete", "degraded", "incomplete", "disabled")}
        self.native_losses: dict[str, int] = {}
        self.writer_dropped_bytes = 0
        self.python_unhealthy_attempts = 0
        self._native_previous: dict[str, int] = {}
        self._context = self._had_context = False

    def begin_attempt(self) -> None:
        self.attempts_started += 1
        self._native_previous.clear()
        self._context = self._had_context = False

    def worker_started(self) -> None:
        self.worker_starts += 1

    def observe_traffic(self, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        self.traffic_gaps += kind == "traffic_gap"
        outcome = _traffic_outcome(event)
        self.traffic_failures += outcome == "failed"
        self.traffic_stops += outcome == "stopped"
        self.partial_output += kind in ("traffic_output_truncated", "traffic_output_incomplete")
        row = event.get("interval")
        if (kind != "traffic_output" or event.get("partial") or not isinstance(row, dict)
                or row.get("reporter") != "receiver" or row.get("receiver_report_available") is False
                or row.get("report_type") != "interval" or row.get("source") != "local_output"):
            return
        direction = row.get("direction")
        now_ns = event.get("clock", {}).get("monotonic_ns")
        if direction in self.directions and type(now_ns) is int and now_ns >= self.start_ns:
            self.directions[direction].observe(event, self.attempts_started, now_ns)

    def observe_native(self, sample: dict[str, Any], receipt_ns: int) -> None:
        values = sample.get("values", {})
        self.native_samples += 1
        for name, value in (("sync_successes", values.get("sync_successes")),
                            ("pdu_accepts", values.get("pdu_accepts")), ("send_drops", sample.get("send_drops"))):
            if type(value) is not int or value < 0:
                continue
            previous = self._native_previous.get(name, 0)
            if value < previous:
                self.native_counter_resets += 1
                delta = 0
            else:
                delta = value - previous
            self._native_previous[name] = value
            if name == "sync_successes":
                self.sync_samples += 1
                self.sync_successes += delta
                self.sync_reacquisitions += max(0, delta - 1) if previous == 0 else delta
            elif name == "pdu_accepts":
                self.pdu_samples += 1
                self.pdu_accepts += delta
                if value > 0 and self.first_pdu_ns is None:
                    self.first_pdu_ns = receipt_ns
            else:
                self.send_drop_samples += 1
                self.native_send_drops += delta
        if "pdu_active" in values or "drb_context_active" in values:
            self.context_samples += 1
            context = bool(values.get("pdu_active", 0) or
                           (self._native_previous.get("pdu_accepts", 0) and values.get("drb_context_active", 0)))
            if context and not self._context and self._had_context:
                self.context_restorations += 1
            if self._context and not context:
                self.context_losses += 1
            self._context = context
            self._had_context |= context

    def finish_attempt(self, capture: dict[str, Any]) -> None:
        self.attempts_finished += 1
        integrity = capture.get("native_integrity", "incomplete")
        self.integrity_counts[integrity if integrity in self.integrity_counts else "incomplete"] += 1
        losses = capture.get("footer_losses")
        if isinstance(losses, dict):
            for key in NATIVE_LOSS_FIELDS:
                value = losses.get(key)
                if type(value) is int and value >= 0:
                    self.native_losses[key] = self.native_losses.get(key, 0) + value
        writers = capture.get("writer_dropped_bytes")
        if isinstance(writers, dict):
            self.writer_dropped_bytes += sum(v for v in writers.values() if type(v) is int and v > 0)
        self.python_unhealthy_attempts += (capture.get("reported_capture") or {}).get("healthy") is False

    def snapshot(self, end_ns: int) -> dict[str, Any]:
        unfinished = max(0, self.attempts_started - self.attempts_finished)
        native = ("incomplete" if unfinished or self.integrity_counts["incomplete"] else
                  "degraded" if self.integrity_counts["degraded"] else
                  "complete" if self.integrity_counts["complete"] else
                  "disabled" if self.integrity_counts["disabled"] else "unavailable")
        observer_loss = self.writer_dropped_bytes or self.python_unhealthy_attempts or self.native_send_drops or self.partial_output
        overall = "degraded" if native in ("complete", "disabled") and observer_loss else native
        return {
            "schema_version": 1, "kind": "flight_session_summary",
            "elapsed_seconds": max(0, end_ns - self.start_ns) / 1e9,
            "attempts": {"started": self.attempts_started, "finished": self.attempts_finished,
                         "worker_starts": self.worker_starts, "process_restarts": max(0, self.worker_starts - 1),
                         "unfinished": unfinished},
            "attachment": {"first_pdu_accept_observed_seconds": max(0, self.first_pdu_ns - self.start_ns) / 1e9
                           if self.first_pdu_ns is not None else None,
                           "pdu_accepts_observed": self.pdu_accepts if self.pdu_samples else None},
            "traffic": {name: value.snapshot(self.start_ns, end_ns) for name, value in self.directions.items()},
            "recovery": {"sync_successes_observed": self.sync_successes if self.sync_samples else None,
                         "within_worker_sync_reacquisitions": self.sync_reacquisitions if self.sync_samples else None,
                         "within_worker_context_losses": self.context_losses if self.context_samples else None,
                         "within_worker_context_restorations": self.context_restorations if self.context_samples else None,
                         "native_samples": self.native_samples, "native_counter_resets": self.native_counter_resets,
                         "native_monitor_send_drops": self.native_send_drops if self.send_drop_samples else None},
            "capture": {"integrity": overall, "native_integrity": native,
                        "attempt_integrity_counts": dict(self.integrity_counts),
                        "native_footer_losses": dict(self.native_losses) if self.native_losses else None,
                        "writer_dropped_bytes": self.writer_dropped_bytes,
                        "python_unhealthy_attempts": self.python_unhealthy_attempts,
                        "traffic_gap_records": self.traffic_gaps,
                        "traffic_failed_or_interrupted_generations": self.traffic_failures,
                        "traffic_expected_stops": self.traffic_stops,
                        "traffic_incomplete_output_records": self.partial_output},
            "scope": {"timing": "monotonic receipt observations from session start, not exact protocol or first-packet timestamps",
                      "traffic": "rounded local receiver intervals; unique disjoint connection coverage; "
                                 "sender delivery unavailable",
                      "gaps": "receiver receipt spacing over 1 s plus 0.5 s tolerance; not measured RF outage duration",
                      "memory": "two directions, one ordered stream checkpoint each, fixed native and attempt counters"},
        }


def render_summary(summary: dict[str, Any]) -> str:
    """Bounded plain text for the final console and summary.txt."""
    def number(value: Any, suffix: str = "") -> str:
        return f"{value:.3f}{suffix}" if _number(value) else "unavailable"
    def count(value: Any) -> str:
        return str(value) if type(value) is int else "unavailable"
    attempts, recovery, capture = summary["attempts"], summary["recovery"], summary["capture"]
    lines = ["Flight session summary",
             f"Elapsed: {number(summary['elapsed_seconds'], ' s')}; worker starts: {attempts['worker_starts']}; "
             f"process restarts: {attempts['process_restarts']}; attempts finished: {attempts['finished']}/{attempts['started']}",
             "First PDU acceptance observed: " + number(summary['attachment']['first_pdu_accept_observed_seconds'], " s"),
             f"Within-worker synchronization reacquisitions: {count(recovery['within_worker_sync_reacquisitions'])}; "
             f"context losses/restorations: {count(recovery['within_worker_context_losses'])}/"
             f"{count(recovery['within_worker_context_restorations'])}"]
    for name, row in summary["traffic"].items():
        rate = row["received_bits_per_second"]
        lines.append(f"{name.upper()}: delivery={row['delivery']}; first positive receiver report: "
                     f"{number(row['first_positive_receiver_report_seconds'], ' s')}; receive rate: "
                     f"{number(rate / 1e6 if rate is not None else None, ' Mbps')}; covered: "
                     f"{number(row['covered_seconds'], ' s')}; datagrams lost/total: "
                     f"{row['lost_datagrams'] if row['lost_datagrams'] is not None else 'unavailable'}/"
                     f"{row['total_datagrams'] if row['total_datagrams'] is not None else 'unavailable'} "
                     f"({number(row['loss_percent'], '%')}); reporting gaps: {count(row['reporting_gaps'])}; "
                     f"loss-report intervals: {row['intervals_with_loss_fields']}/{row['accounted_intervals']}; "
                     f"longest: {number(row['longest_reporting_gap_seconds'], ' s')}; "
                     f"trailing unobserved: {number(row['trailing_unobserved_seconds'], ' s')}; "
                     f"ambiguous/overlapping intervals excluded: {row['ambiguous_receiver_reports']}/"
                     f"{row['overlapping_intervals_excluded']}")
    losses = (json.dumps(capture['native_footer_losses'], sort_keys=True)
              if capture['native_footer_losses'] is not None else 'unavailable')
    lines += [f"Traffic generations: failed/interrupted: {capture['traffic_failed_or_interrupted_generations']}; "
              f"expected stops: {count(capture.get('traffic_expected_stops'))}",
              f"Capture integrity: {capture['integrity']}; native: {capture['native_integrity']}; "
              f"native footer losses: {losses}; "
              f"writer bytes dropped: {capture['writer_dropped_bytes']}",
              "Times are first receipt observations. Rates cover unique receiver intervals only.",
              "Reporting gaps and trailing unobserved time are observer coverage, not measured RF outages."]
    return "\n".join(lines) + "\n"


NATIVE_LOSS_FIELDS = ("writer_errors", "dropped_ring_full", "dropped_no_slot", "no_slot_threads",
                      "invalid_timestamp_records", "payload_truncated")


class BoundedLogs:
    def __init__(self, maximum_bytes: int = 64 * 1024 * 1024) -> None:
        self.remaining = maximum_bytes
        self.read_bytes = 0
        self.invalid_lines = 0
        self.limited = False
        self.unavailable_files = 0
        self.budget_partial_lines = 0

    def records(self, paths: list[Path]) -> Iterator[dict[str, Any]]:
        # Rotating text writers can split a JSON line across adjacent files.
        buffer = bytearray()
        discarding = False
        for path in paths:
            try:
                with path.open("rb") as source:
                    while self.remaining > 0:
                        data = source.read(min(16384, self.remaining))
                        if not data:
                            break
                        self.remaining -= len(data)
                        self.read_bytes += len(data)
                        for piece in data.splitlines(keepends=True):
                            complete = piece.endswith(b"\n")
                            if not discarding:
                                buffer.extend(piece)
                                if len(buffer) > 65536:
                                    self.invalid_lines += 1
                                    buffer.clear()
                                    discarding = True
                            if complete:
                                if not discarding and buffer.strip():
                                    try:
                                        value = json.loads(buffer)
                                        if isinstance(value, dict):
                                            yield value
                                        else:
                                            self.invalid_lines += 1
                                    except (ValueError, UnicodeError):
                                        self.invalid_lines += 1
                                buffer.clear()
                                discarding = False
                    if self.remaining == 0:
                        self.limited = True
                        break
            except OSError:
                self.unavailable_files += 1
        if buffer and self.remaining == 0:
            self.budget_partial_lines += 1
        elif buffer:
            self.invalid_lines += 1


def _paths(directory: Path, prefix: str, reader: BoundedLogs) -> list[Path]:
    try:
        entries = list(itertools.islice(directory.iterdir(), 1025))
    except OSError:
        reader.unavailable_files += 1
        return []
    reader.limited |= len(entries) > 1024
    matches = [path for path in entries[:1024] if path.name.startswith(prefix) and
               path.suffix in (".log", ".ndjson") and not path.is_symlink()]
    reader.limited |= len(matches) > 256
    def order(path: Path) -> tuple[str, int]:
        match = re.search(r"(\d+)\.(?:log|ndjson)$", path.name)
        return (path.name[:match.start()] if match else path.name, int(match[1]) if match else 0)
    return sorted(matches, key=order)[:256]


def _status(directory: Path, name: str = "status.json") -> dict[str, Any]:
    try:
        with (directory / name).open("rb") as source:
            data = source.read(512 * 1024 + 1)
        result = json.loads(data) if len(data) <= 512 * 1024 else None
        return result if isinstance(result, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def _footer_records(paths: list[Path], reader: BoundedLogs) -> Iterator[dict[str, Any]]:
    """Inspect complete records in bounded file tails independently of the event budget.

    The native writer appends its footer last. A large healthy capture must not
    lose this evidence just because the sequential event scan has stopped.
    This verifies footer records, not the integrity of all preceding events.
    """
    for path in paths:
        try:
            with path.open("rb") as source:
                size = source.seek(0, 2)
                offset = max(0, size - 65536)
                source.seek(offset)
                data = source.read(65536)
            reader.read_bytes += len(data)
            # A tail may begin or end inside a record; neither fragment is proof
            # of a complete footer. Native records do not span rotated files.
            if offset:
                data = data.partition(b"\n")[2]
            for line in data.splitlines(keepends=True):
                if not line.endswith(b"\n") or b'"capture_footer"' not in line:
                    continue
                try:
                    value = json.loads(line)
                except (ValueError, UnicodeError):
                    reader.invalid_lines += 1
                    continue
                if isinstance(value, dict) and value.get("kind") == "capture_footer":
                    yield value
        except OSError:
            reader.unavailable_files += 1


def summarize_run(directory: Path, maximum_bytes: int = 64 * 1024 * 1024, *,
                  live: Optional[SessionSummary] = None, final_status: Optional[dict[str, Any]] = None,
                  recorder_enabled: Optional[bool] = None) -> dict[str, Any]:
    reader = BoundedLogs(maximum_bytes)
    receiver_reports = 0
    delivered: Any = None
    maximum_received: Any = None
    gaps = failures = stops = partial_output = 0
    for event in reader.records(_paths(directory, "traffic.", reader)) if live is None else ():
        interval = event.get("interval")
        if (isinstance(interval, dict) and interval.get("reporter") == "receiver"
                and interval.get("receiver_report_available") is not False):
            value = interval.get("bytes")
            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                receiver_reports += 1
                maximum_received = max(maximum_received or 0, value)
                delivered = bool(delivered or value > 0)
        gaps += event.get("kind") == "traffic_gap"
        outcome = _traffic_outcome(event)
        failures += outcome == "failed"
        stops += outcome == "stopped"
        partial_output += event.get("kind") in ("traffic_output_truncated", "traffic_output_incomplete")
    pdu_accept_observed: Any = None
    for event in reader.records(_paths(directory, "recovery.", reader)) if live is None else ():
        if event.get("event") == "pdu_accepted":
            pdu_accept_observed = True
        snapshot = event.get("native_values")
        if (isinstance(snapshot, dict) and type(snapshot.get("pdu_accepts")) is int
                and snapshot["pdu_accepts"] > 0):
            pdu_accept_observed = True
    footer_count = clean_footers = invalid_footers = 0
    async_repeat_summaries = async_repeated_records = 0 if live is None else None
    losses: dict[str, int] = {}
    native_paths_reader = BoundedLogs()
    recorder_paths = _paths(directory / "recorder", "oai-flight-recorder-", native_paths_reader)
    reader.limited |= native_paths_reader.limited
    reader.unavailable_files += native_paths_reader.unavailable_files
    for event in reader.records(recorder_paths) if live is None else ():
        if event.get("kind") == "async_repeat_summary":
            count = event.get("repeat_count")
            if type(count) is int and count > 0:
                async_repeat_summaries += 1
                async_repeated_records += count
        if event.get("kind") == "event" and event.get("event") == 15 and event.get("d") == 1:
            pdu_accept_observed = True
    footer_reader = BoundedLogs()
    for event in _footer_records(recorder_paths, footer_reader):
        if event.get("kind") == "capture_footer":
            footer_count += 1
            required = ("clean", "writer_errors", "dropped_ring_full", "dropped_no_slot", "no_slot_threads",
                        "invalid_timestamp_records", "payload_truncated", "rings_assigned")
            valid_footer = all(type(event.get(key)) is int and event[key] >= 0 for key in required)
            invalid_footers += not valid_footer
            clean_footers += valid_footer and event["clean"] == 1
            for key in NATIVE_LOSS_FIELDS:
                value = event.get(key)
                if type(value) is int and value >= 0:
                    losses[key] = losses.get(key, 0) + value
    status = final_status if final_status is not None else _status(directory)
    if recorder_enabled is None:
        recorder_enabled = _status(directory, "metadata.json").get("recorder", {}).get("enabled")
    writers = status.get("writers")
    writer_losses = {name: writer.get("dropped_bytes") for name, writer in writers.items()
                     if isinstance(writer, dict)} if isinstance(writers, dict) else None
    reasons = []
    if recorder_enabled is False:
        integrity = "disabled"
    elif footer_count != 1 or clean_footers != 1 or invalid_footers:
        integrity = "incomplete"
        reasons.append("missing_or_invalid_clean_footer")
    elif native_paths_reader.limited or footer_reader.unavailable_files or footer_reader.invalid_lines:
        integrity = "incomplete"
        reasons.append("native_scan_incomplete")
    elif any(losses.values()):
        integrity = "degraded"
        reasons.append("native_footer_reports_loss")
    else:
        integrity = "complete"
    if live is not None:
        receiver_reports = sum(value.reports for value in live.directions.values())
        delivered = any(value.positive_reports for value in live.directions.values()) if receiver_reports else None
        maximum_received = max((value.maximum_bytes for value in live.directions.values()
                                if value.maximum_bytes is not None), default=None)
        gaps, failures, partial_output = live.traffic_gaps, live.traffic_failures, live.partial_output
        stops = live.traffic_stops
        pdu_accept_observed = True if live.first_pdu_ns is not None else None
    return {"schema_version": 1, "kind": "flight_capture_summary", "capture_directory": str(directory),
            "control_plane": {"pdu_accept_observed": pdu_accept_observed,
                              "scope": "decoded PDU acceptance only; absent evidence is unavailable"},
            "traffic": {"delivered_bytes_observed": delivered, "receiver_reports": receiver_reports,
                        "maximum_receiver_report_bytes": maximum_received, "gap_records": gaps,
                        "failed_or_interrupted_generations": failures, "incomplete_output_records": partial_output,
                        "expected_stops": stops,
                        "scope": "rounded receiver text reports only; duplicate reports are not summed; no video-delivery claim"},
            "capture": {"final_status_present": bool(status), "reported_capture": status.get("capture"),
                        "native_integrity": integrity, "native_integrity_reasons": reasons,
                        "recorder_enabled": recorder_enabled,
                        "writer_dropped_bytes": writer_losses, "footer_count": footer_count,
                        "clean_footer_observed": clean_footers == 1 and footer_count == 1,
                        "invalid_footer_records": invalid_footers,
                        "async_repeat_summaries": async_repeat_summaries,
                        "async_repeated_records": async_repeated_records,
                        "footer_losses": losses if footer_count else None,
                        "missing_footer_means": "unavailable completeness; crash, kill, live capture, recorder failure or scan limits possible",
                        "footer_scan": {"bytes_read": footer_reader.read_bytes,
                                        "file_selection_limited": native_paths_reader.limited,
                                        "unavailable_files": footer_reader.unavailable_files,
                                        "invalid_lines": footer_reader.invalid_lines,
                                        "scope": "complete records in last 65536 bytes of each selected native file; not full capture integrity"}},
            "scan": {"bytes_read": reader.read_bytes, "maximum_bytes": maximum_bytes,
                     "event_scan_performed": live is None,
                     "limited": reader.limited, "invalid_or_partial_lines": reader.invalid_lines,
                     "budget_partial_lines": reader.budget_partial_lines,
                     "unavailable_files": reader.unavailable_files,
                     "limits": "1024 entries/directory, 256 files/stream, 65536 bytes/line"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_directory", type=Path, help="one attempt directory, not a recovery session parent")
    args = parser.parse_args()
    print(json.dumps(summarize_run(args.capture_directory), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
