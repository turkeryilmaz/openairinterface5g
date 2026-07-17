#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Map OAI profiler counter ticks onto archived host clock domains."""

from __future__ import annotations

import csv
import math
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClockAnchor:
    row_number: int
    tick: int
    monotonic_raw_ns: int
    realtime_ns: int
    bounds_status: str
    monotonic_bracket_width_ns: int | None
    tick_bracket_width: int | None

    @property
    def bounded(self) -> bool:
        return self.bounds_status == "bounded"


@dataclass(frozen=True)
class ClockPoint:
    tick: int
    monotonic_raw_ns: int | None
    realtime_ns: int | None
    method: str
    position: str
    status: str
    realtime_status: str
    acquisition_uncertainty_ns: int | None
    uncertainty_scope: str
    left_anchor_row: int | None
    right_anchor_row: int | None
    segment_tick_delta: int | None
    segment_monotonic_delta_ns: int | None
    segment_realtime_delta_ns: int | None


def parse_optional_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def rounded_ratio(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    sign = -1 if numerator < 0 else 1
    return sign * ((abs(numerator) + denominator // 2) // denominator)


def ceil_ratio(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("ceil_ratio requires a nonnegative numerator and positive denominator")
    return (numerator + denominator - 1) // denominator


class ClockMapper:
    """Piecewise tick-to-clock mapping with explicit evidence quality."""

    def __init__(
        self,
        anchors: list[ClockAnchor],
        counter_hz: int,
        *,
        sync_present: bool,
        sample_count: int,
        read_error_count: int,
        invalid_anchor_count: int,
        malformed_bounds_count: int,
        realtime_regression_count: int,
    ) -> None:
        self.anchors = anchors
        self.counter_hz = counter_hz
        self.sync_present = sync_present
        self.sample_count = sample_count
        self.read_error_count = read_error_count
        self.invalid_anchor_count = invalid_anchor_count
        self.malformed_bounds_count = malformed_bounds_count
        self.realtime_regression_count = realtime_regression_count
        self.bounded_anchor_count = sum(anchor.bounded for anchor in anchors)
        self._ticks = [anchor.tick for anchor in anchors]
        self._anchor_by_tick = {anchor.tick: anchor for anchor in anchors}

    @classmethod
    def from_sync(cls, path: Path, counter_hz: int) -> "ClockMapper":
        if not path.is_file():
            return cls(
                [],
                counter_hz,
                sync_present=False,
                sample_count=0,
                read_error_count=0,
                invalid_anchor_count=0,
                malformed_bounds_count=0,
                realtime_regression_count=0,
            )

        anchors: list[ClockAnchor] = []
        sample_count = 0
        read_errors = 0
        invalid_anchors = 0
        malformed_bounds = 0
        realtime_regressions = 0
        with path.open(newline="") as stream:
            for row_number, row in enumerate(csv.DictReader(stream), start=2):
                sample_count += 1
                row_status = str(row.get("status", "")).strip()
                if row_status not in {"", "ok"}:
                    read_errors += 1
                    continue
                realtime_ns = parse_optional_int(row.get("realtime_ns"))
                monotonic_ns = parse_optional_int(row.get("monotonic_raw_ns"))
                tick = parse_optional_int(row.get("tick"))
                if (
                    realtime_ns is None
                    or monotonic_ns is None
                    or tick is None
                    or realtime_ns <= 0
                    or monotonic_ns <= 0
                    or tick <= 0
                ):
                    invalid_anchors += 1
                    continue

                bound_names = (
                    "monotonic_raw_before_ns",
                    "monotonic_raw_after_ns",
                    "monotonic_raw_uncertainty_ns",
                    "tick_before",
                    "tick_after",
                    "tick_uncertainty",
                )
                bounds_declared = any(str(row.get(name, "")).strip() for name in bound_names)
                bounds_status = "legacy_unbounded"
                monotonic_width: int | None = None
                tick_width: int | None = None
                if bounds_declared:
                    before_ns = parse_optional_int(row.get("monotonic_raw_before_ns"))
                    after_ns = parse_optional_int(row.get("monotonic_raw_after_ns"))
                    declared_ns = parse_optional_int(row.get("monotonic_raw_uncertainty_ns"))
                    tick_before = parse_optional_int(row.get("tick_before"))
                    tick_after = parse_optional_int(row.get("tick_after"))
                    declared_tick = parse_optional_int(row.get("tick_uncertainty"))
                    bounds_valid = (
                        before_ns is not None
                        and after_ns is not None
                        and declared_ns is not None
                        and tick_before is not None
                        and tick_after is not None
                        and declared_tick is not None
                        and before_ns > 0
                        and before_ns <= monotonic_ns <= after_ns
                        and declared_ns == after_ns - before_ns
                        and tick_before > 0
                        and tick_before <= tick <= tick_after
                        and declared_tick == tick_after - tick_before
                    )
                    if bounds_valid:
                        bounds_status = "bounded"
                        monotonic_width = declared_ns
                        tick_width = declared_tick
                    else:
                        bounds_status = "malformed_bounds"
                        malformed_bounds += 1

                if anchors and (
                    tick <= anchors[-1].tick
                    or monotonic_ns <= anchors[-1].monotonic_raw_ns
                ):
                    invalid_anchors += 1
                    continue
                if anchors and realtime_ns < anchors[-1].realtime_ns:
                    realtime_regressions += 1
                anchors.append(
                    ClockAnchor(
                        row_number=row_number,
                        tick=tick,
                        monotonic_raw_ns=monotonic_ns,
                        realtime_ns=realtime_ns,
                        bounds_status=bounds_status,
                        monotonic_bracket_width_ns=monotonic_width,
                        tick_bracket_width=tick_width,
                    )
                )

        return cls(
            anchors,
            counter_hz,
            sync_present=True,
            sample_count=sample_count,
            read_error_count=read_errors,
            invalid_anchor_count=invalid_anchors,
            malformed_bounds_count=malformed_bounds,
            realtime_regression_count=realtime_regressions,
        )

    @property
    def status(self) -> str:
        if not self.sync_present:
            return "sync_missing"
        if not self.anchors:
            return "no_valid_anchors"
        if len(self.anchors) == 1 and self.counter_hz <= 0:
            return "single_anchor_without_counter_hz"
        if self.invalid_anchor_count or self.read_error_count:
            return "usable_with_rejected_samples"
        if self.malformed_bounds_count:
            return "usable_with_malformed_bounds"
        if self.realtime_regression_count:
            return "monotonic_usable_realtime_regression"
        if self.bounded_anchor_count == 0:
            return "usable_unbounded"
        if self.bounded_anchor_count < len(self.anchors):
            return "usable_partially_bounded"
        return "ok"

    def _unavailable(self, tick: int, status: str) -> ClockPoint:
        return ClockPoint(
            tick=tick,
            monotonic_raw_ns=None,
            realtime_ns=None,
            method="unavailable",
            position="unavailable",
            status=status,
            realtime_status="unavailable",
            acquisition_uncertainty_ns=None,
            uncertainty_scope="unavailable",
            left_anchor_row=None,
            right_anchor_row=None,
            segment_tick_delta=None,
            segment_monotonic_delta_ns=None,
            segment_realtime_delta_ns=None,
        )

    @staticmethod
    def _anchor_uncertainty(anchor: ClockAnchor) -> int | None:
        if not anchor.bounded:
            return None
        return anchor.monotonic_bracket_width_ns

    def _exact_anchor(self, anchor: ClockAnchor) -> ClockPoint:
        uncertainty = self._anchor_uncertainty(anchor)
        return ClockPoint(
            tick=anchor.tick,
            monotonic_raw_ns=anchor.monotonic_raw_ns,
            realtime_ns=anchor.realtime_ns,
            method="anchor",
            position="anchor_exact",
            status=f"anchor_{anchor.bounds_status}",
            realtime_status="anchor_exact",
            acquisition_uncertainty_ns=uncertainty,
            uncertainty_scope=(
                "anchor_acquisition_bracket"
                if uncertainty is not None
                else "unbounded_anchor_acquisition"
            ),
            left_anchor_row=anchor.row_number,
            right_anchor_row=anchor.row_number,
            segment_tick_delta=0,
            segment_monotonic_delta_ns=0,
            segment_realtime_delta_ns=0,
        )

    def _single_anchor(self, tick: int, anchor: ClockAnchor) -> ClockPoint:
        if self.counter_hz <= 0:
            return self._unavailable(tick, "insufficient_anchors_and_counter_hz")
        counter_delta_ns = rounded_ratio(
            (tick - anchor.tick) * 1_000_000_000,
            self.counter_hz,
        )
        uncertainty = self._anchor_uncertainty(anchor)
        return ClockPoint(
            tick=tick,
            monotonic_raw_ns=anchor.monotonic_raw_ns + counter_delta_ns,
            realtime_ns=anchor.realtime_ns + counter_delta_ns,
            method="nominal_counter_hz",
            position="single_anchor_extrapolation",
            status=f"single_anchor_{anchor.bounds_status}",
            realtime_status="nominal_rate_extrapolation",
            acquisition_uncertainty_ns=uncertainty,
            uncertainty_scope="anchor_acquisition_only_drift_unbounded",
            left_anchor_row=anchor.row_number,
            right_anchor_row=anchor.row_number,
            segment_tick_delta=None,
            segment_monotonic_delta_ns=None,
            segment_realtime_delta_ns=None,
        )

    def _segment_uncertainty(
        self,
        left: ClockAnchor,
        right: ClockAnchor,
        monotonic_delta_ns: int,
        tick_delta: int,
    ) -> int | None:
        bounded = [anchor for anchor in (left, right) if anchor.bounded]
        if not bounded:
            return None
        monotonic_width = max(anchor.monotonic_bracket_width_ns or 0 for anchor in bounded)
        tick_width = max(anchor.tick_bracket_width or 0 for anchor in bounded)
        return monotonic_width + ceil_ratio(
            tick_width * monotonic_delta_ns,
            tick_delta,
        )

    def map_tick(self, tick: int) -> ClockPoint:
        if tick <= 0:
            return self._unavailable(tick, "invalid_event_tick")
        exact = self._anchor_by_tick.get(tick)
        if exact is not None:
            return self._exact_anchor(exact)
        if not self.anchors:
            return self._unavailable(tick, self.status)
        if len(self.anchors) == 1:
            return self._single_anchor(tick, self.anchors[0])

        index = bisect_right(self._ticks, tick) - 1
        if index < 0:
            left = self.anchors[0]
            right = self.anchors[1]
            position = "extrapolated_before"
        elif index >= len(self.anchors) - 1:
            left = self.anchors[-2]
            right = self.anchors[-1]
            position = "extrapolated_after"
        else:
            left = self.anchors[index]
            right = self.anchors[index + 1]
            position = "interpolated"

        tick_delta = right.tick - left.tick
        monotonic_delta = right.monotonic_raw_ns - left.monotonic_raw_ns
        realtime_delta = right.realtime_ns - left.realtime_ns
        if tick_delta <= 0 or monotonic_delta <= 0:
            return self._unavailable(tick, "invalid_mapping_segment")

        offset_tick = tick - left.tick
        monotonic_ns = left.monotonic_raw_ns + rounded_ratio(
            offset_tick * monotonic_delta,
            tick_delta,
        )
        if realtime_delta > 0:
            realtime_ns: int | None = left.realtime_ns + rounded_ratio(
                offset_tick * realtime_delta,
                tick_delta,
            )
            realtime_status = (
                "piecewise_linear"
                if position == "interpolated"
                else "piecewise_linear_extrapolation_drift_unbounded"
            )
        else:
            realtime_ns = None
            realtime_status = (
                "segment_realtime_regression"
                if realtime_delta < 0
                else "segment_realtime_nonprogressing"
            )

        bounded_count = int(left.bounded) + int(right.bounded)
        bound_status = ("bounded", "partially_bounded", "unbounded")[2 - bounded_count]
        uncertainty = self._segment_uncertainty(
            left,
            right,
            monotonic_delta,
            tick_delta,
        )
        uncertainty_scope = (
            "local_anchor_acquisition_only"
            if position == "interpolated"
            else "local_anchor_acquisition_only_drift_unbounded"
        )
        if uncertainty is None:
            uncertainty_scope = (
                "unbounded_anchor_acquisition"
                if position == "interpolated"
                else "unbounded_anchor_acquisition_and_drift"
            )
        return ClockPoint(
            tick=tick,
            monotonic_raw_ns=monotonic_ns,
            realtime_ns=realtime_ns,
            method="piecewise_linear",
            position=position,
            status=f"{position}_{bound_status}",
            realtime_status=realtime_status,
            acquisition_uncertainty_ns=uncertainty,
            uncertainty_scope=uncertainty_scope,
            left_anchor_row=left.row_number,
            right_anchor_row=right.row_number,
            segment_tick_delta=tick_delta,
            segment_monotonic_delta_ns=monotonic_delta,
            segment_realtime_delta_ns=realtime_delta,
        )

IDENTITY_FIELDS = [
    "profile_dir",
    "run_id",
    "experiment_id",
    "campaign_id",
    "variant",
    "trial",
    "role",
    "hostname",
]

EVENT_FIELDS = [
    "schema_version",
    "seq",
    "tid",
    "thread_name",
    "event_id",
    "event_name",
    "event_role",
    "subsystem",
    "event_class",
    "event_kind",
    "detail_level",
    "nesting_depth",
    "frame",
    "slot",
    "absolute_slot",
    "correlation_id",
    "span_id",
    "parent_id",
    "cpu_start",
    "cpu_end",
    "cpu_migrated",
    "flags",
    "aux0",
    "aux0_name",
    "aux0_unit",
    "aux1",
    "aux1_name",
    "aux1_unit",
    "aux2",
    "aux2_name",
    "aux2_unit",
    "aux3",
    "aux3_name",
    "aux3_unit",
    "start_tick",
    "duration_tick",
    "end_tick",
    "duration_us",
]

MAPPING_FIELDS = [
    "event_tick_status",
    "counter_hz",
    "sync_status",
    "sync_sample_count",
    "sync_valid_anchor_count",
    "sync_bounded_anchor_count",
    "sync_read_error_count",
    "sync_invalid_anchor_count",
    "sync_malformed_bounds_count",
    "sync_realtime_regression_count",
    "start_monotonic_raw_ns_estimate",
    "end_monotonic_raw_ns_estimate",
    "mapped_duration_ns",
    "recorded_duration_ns",
    "duration_difference_ns",
    "duration_difference_ppm",
    "monotonic_interval_status",
    "start_realtime_ns_estimate",
    "end_realtime_ns_estimate",
    "realtime_interval_status",
    "start_mapping_method",
    "start_mapping_position",
    "start_mapping_status",
    "start_realtime_mapping_status",
    "start_acquisition_uncertainty_ns",
    "start_uncertainty_scope",
    "start_left_anchor_row",
    "start_right_anchor_row",
    "start_segment_tick_delta",
    "start_segment_monotonic_delta_ns",
    "start_segment_realtime_delta_ns",
    "end_mapping_method",
    "end_mapping_position",
    "end_mapping_status",
    "end_realtime_mapping_status",
    "end_acquisition_uncertainty_ns",
    "end_uncertainty_scope",
    "end_left_anchor_row",
    "end_right_anchor_row",
    "end_segment_tick_delta",
    "end_segment_monotonic_delta_ns",
    "end_segment_realtime_delta_ns",
    "interval_acquisition_uncertainty_ns",
    "interval_uncertainty_scope",
]

EVENT_TIMELINE_FIELDS = IDENTITY_FIELDS + EVENT_FIELDS + MAPPING_FIELDS


def parse_finite_float(value: object) -> float | None:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _interval_status(start: ClockPoint, end: ClockPoint, *, realtime: bool) -> str:
    start_ns = start.realtime_ns if realtime else start.monotonic_raw_ns
    end_ns = end.realtime_ns if realtime else end.monotonic_raw_ns
    if start_ns is None or end_ns is None:
        return "unavailable"
    if end_ns < start_ns:
        return "clock_regression"
    positions = {start.position, end.position}
    if any("extrapolation" in position or "extrapolated" in position for position in positions):
        return "valid_with_extrapolation_drift_unbounded"
    return "valid"


def _point_fields(prefix: str, point: ClockPoint) -> dict[str, object]:
    return {
        f"{prefix}_mapping_method": point.method,
        f"{prefix}_mapping_position": point.position,
        f"{prefix}_mapping_status": point.status,
        f"{prefix}_realtime_mapping_status": point.realtime_status,
        f"{prefix}_acquisition_uncertainty_ns": point.acquisition_uncertainty_ns,
        f"{prefix}_uncertainty_scope": point.uncertainty_scope,
        f"{prefix}_left_anchor_row": point.left_anchor_row,
        f"{prefix}_right_anchor_row": point.right_anchor_row,
        f"{prefix}_segment_tick_delta": point.segment_tick_delta,
        f"{prefix}_segment_monotonic_delta_ns": point.segment_monotonic_delta_ns,
        f"{prefix}_segment_realtime_delta_ns": point.segment_realtime_delta_ns,
    }


def event_timeline_row(
    profile_dir: Path,
    metadata: dict[str, str],
    event: dict[str, str],
    mapper: ClockMapper,
) -> dict[str, object]:
    start_tick_value = parse_optional_int(event.get("start_tick"))
    duration_tick_value = parse_optional_int(event.get("duration_tick"))
    event_ticks_valid = (
        start_tick_value is not None
        and duration_tick_value is not None
        and start_tick_value > 0
        and duration_tick_value >= 0
    )
    start_tick = start_tick_value if start_tick_value is not None else 0
    duration_tick = duration_tick_value if duration_tick_value is not None else -1
    end_tick = start_tick + duration_tick if event_ticks_valid else None
    start_point = mapper.map_tick(start_tick)
    end_point = mapper.map_tick(end_tick if end_tick is not None else 0)

    monotonic_status = _interval_status(start_point, end_point, realtime=False)
    realtime_status = _interval_status(start_point, end_point, realtime=True)
    mapped_duration_ns: int | None = None
    if (
        monotonic_status.startswith("valid")
        and start_point.monotonic_raw_ns is not None
        and end_point.monotonic_raw_ns is not None
    ):
        mapped_duration_ns = end_point.monotonic_raw_ns - start_point.monotonic_raw_ns

    duration_us = parse_finite_float(event.get("duration_us"))
    recorded_duration_ns = (
        round(duration_us * 1000)
        if duration_us is not None and duration_us >= 0
        else None
    )
    duration_difference_ns = (
        mapped_duration_ns - recorded_duration_ns
        if mapped_duration_ns is not None and recorded_duration_ns is not None
        else None
    )
    duration_difference_ppm = (
        duration_difference_ns * 1_000_000 / recorded_duration_ns
        if duration_difference_ns is not None and recorded_duration_ns
        else None
    )

    point_uncertainties = [
        value
        for value in (
            start_point.acquisition_uncertainty_ns,
            end_point.acquisition_uncertainty_ns,
        )
        if value is not None
    ]
    interval_uncertainty = sum(point_uncertainties) if len(point_uncertainties) == 2 else None
    uncertainty_scopes = sorted(
        {
            start_point.uncertainty_scope,
            end_point.uncertainty_scope,
        }
        - {""}
    )

    identity = {
        "profile_dir": str(profile_dir),
        "run_id": metadata.get("run_id", profile_dir.name),
        "experiment_id": metadata.get("experiment_id", ""),
        "campaign_id": metadata.get("campaign_id", ""),
        "variant": metadata.get("variant", ""),
        "trial": metadata.get("trial", ""),
        "role": metadata.get("role", "unknown"),
        "hostname": metadata.get("hostname", "unknown"),
    }
    event_values = {field: event.get(field, "") for field in EVENT_FIELDS}
    event_values["start_tick"] = start_tick if start_tick_value is not None else ""
    event_values["duration_tick"] = duration_tick if duration_tick_value is not None else ""
    event_values["end_tick"] = end_tick
    event_values["duration_us"] = duration_us
    return {
        **identity,
        **event_values,
        "event_tick_status": "valid" if event_ticks_valid else "invalid",
        "counter_hz": mapper.counter_hz,
        "sync_status": mapper.status,
        "sync_sample_count": mapper.sample_count,
        "sync_valid_anchor_count": len(mapper.anchors),
        "sync_bounded_anchor_count": mapper.bounded_anchor_count,
        "sync_read_error_count": mapper.read_error_count,
        "sync_invalid_anchor_count": mapper.invalid_anchor_count,
        "sync_malformed_bounds_count": mapper.malformed_bounds_count,
        "sync_realtime_regression_count": mapper.realtime_regression_count,
        "start_monotonic_raw_ns_estimate": start_point.monotonic_raw_ns,
        "end_monotonic_raw_ns_estimate": end_point.monotonic_raw_ns,
        "mapped_duration_ns": mapped_duration_ns,
        "recorded_duration_ns": recorded_duration_ns,
        "duration_difference_ns": duration_difference_ns,
        "duration_difference_ppm": duration_difference_ppm,
        "monotonic_interval_status": monotonic_status,
        "start_realtime_ns_estimate": start_point.realtime_ns,
        "end_realtime_ns_estimate": end_point.realtime_ns,
        "realtime_interval_status": realtime_status,
        **_point_fields("start", start_point),
        **_point_fields("end", end_point),
        "interval_acquisition_uncertainty_ns": interval_uncertainty,
        "interval_uncertainty_scope": ";".join(uncertainty_scopes),
    }
