#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Reconstruct and summarize nrUE transmit-deadline evidence.

Runtime monotonic classification and offline radio-tick reconstruction are
kept separate. Missing, malformed, or uncorrelated observations remain
explicitly invalid rather than being replaced by inferred values.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


DEADLINE_VALID = 1 << 0
DEADLINE_MISSED = 1 << 1
DEADLINE_BEFORE_ANCHOR = 1 << 2
DEADLINE_COMPUTE_CLOCK_ERROR = 1 << 3
DEADLINE_CHECK_CLOCK_ERROR = 1 << 4
DEADLINE_ARITHMETIC_ERROR = 1 << 5
DEADLINE_LEGACY_REALTIME_ERROR = 1 << 6

DEADLINE_EVENT_NAMES = {
    "UE_TX_DEADLINE_MISS",
    "UE_TX_DEADLINE_COMPUTE",
    "UE_TX_DEADLINE_CHECK",
    "USRP_RX_RECV",
    "UE_RF_READ",
    "UE_RF_READ_DRIFT",
}

DEADLINE_CHECK_FIELDS = [
    "profile_dir",
    "process_name",
    "schema_version",
    "run_id",
    "experiment_id",
    "campaign_id",
    "variant",
    "trial",
    "hostname",
    "check_seq",
    "check_tid",
    "check_thread_name",
    "frame",
    "slot",
    "absolute_slot",
    "correlation_id",
    "check_start_tick",
    "deadline_flags",
    "deadline_error_code",
    "valid_flag",
    "missed_flag",
    "before_anchor_flag",
    "compute_clock_error_flag",
    "check_clock_error_flag",
    "arithmetic_error_flag",
    "legacy_realtime_error_flag",
    "runtime_status",
    "runtime_valid",
    "runtime_current_monotonic_raw_ns",
    "runtime_deadline_monotonic_raw_ns",
    "runtime_signed_lateness_ns",
    "runtime_signed_lateness_us",
    "runtime_headroom_us",
    "runtime_missed",
    "compute_status",
    "compute_candidate_count",
    "compute_seq",
    "compute_start_tick",
    "compute_flags",
    "radio_anchor_timestamp",
    "radio_deadline_timestamp",
    "radio_sample_offset",
    "samples_per_subframe",
    "sample_rate_hz",
    "counter_hz",
    "radio_anchor_source",
    "radio_anchor_match",
    "radio_anchor_seq",
    "radio_anchor_start_tick",
    "radio_anchor_end_tick",
    "radio_anchor_to_compute_us",
    "compute_to_check_us",
    "reconstruction_status",
    "reconstruction_valid",
    "reconstructed_offset_tick",
    "reconstructed_deadline_tick",
    "reconstructed_signed_lateness_tick",
    "reconstructed_signed_lateness_us",
    "reconstructed_headroom_us",
    "reconstructed_missed",
    "classification_agreement",
    "reconstruction_minus_runtime_lateness_us",
    "status",
]

_DISTRIBUTION_NAMES = ("min", "p50", "p90", "p99", "max", "mean")
_SUMMARY_DISTRIBUTIONS = (
    "runtime_signed_lateness_us",
    "runtime_headroom_us",
    "reconstructed_signed_lateness_us",
    "reconstructed_headroom_us",
    "reconstruction_minus_runtime_lateness_us",
    "radio_anchor_to_compute_us",
    "compute_to_check_us",
)

DEADLINE_SUMMARY_FIELDS = [
    "profile_dir",
    "process_name",
    "schema_version",
    "run_id",
    "experiment_id",
    "campaign_id",
    "variant",
    "trial",
    "hostname",
    "compute_events",
    "check_events",
    "legacy_miss_events",
    "paired_checks",
    "unpaired_checks",
    "unmatched_compute_events",
    "runtime_valid_checks",
    "runtime_invalid_checks",
    "runtime_misses",
    "runtime_miss_rate",
    "reconstruction_valid_checks",
    "reconstruction_invalid_checks",
    "reconstructed_misses",
    "reconstructed_miss_rate",
    "classification_comparable_checks",
    "classification_agreements",
    "classification_disagreements",
    "classification_agreement_rate",
    "usrp_receive_anchors",
    "ue_rf_read_drift_anchors",
    "ue_rf_read_anchors",
    *[
        f"{distribution}_{stat}"
        for distribution in _SUMMARY_DISTRIBUTIONS
        for stat in _DISTRIBUTION_NAMES
    ],
    "status",
]


@dataclass(frozen=True)
class DeadlineReports:
    check_rows: list[dict[str, object]]
    summary_rows: list[dict[str, object]]


def parse_int(value: object, default: int = 0) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def event_start_tick(row: dict[str, str]) -> int:
    return parse_int(row.get("start_tick"), -1)


def event_end_tick(row: dict[str, str]) -> int:
    start = event_start_tick(row)
    duration = parse_int(row.get("duration_tick"), -1)
    return start + duration if start >= 0 and duration >= 0 else -1


def round_div_signed(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def quantile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def distribution(values: Iterable[float]) -> dict[str, float]:
    samples = sorted(value for value in values if math.isfinite(value))
    if not samples:
        return {name: math.nan for name in _DISTRIBUTION_NAMES}
    return {
        "min": samples[0],
        "p50": quantile(samples, 0.5),
        "p90": quantile(samples, 0.9),
        "p99": quantile(samples, 0.99),
        "max": samples[-1],
        "mean": sum(samples) / len(samples),
    }


def identity(profile_dir: str, metadata: dict[str, str]) -> dict[str, object]:
    return {
        "profile_dir": profile_dir,
        "process_name": metadata.get("process_name", "unknown"),
        "schema_version": metadata.get("schema_version", "1"),
        "run_id": metadata.get("run_id", ""),
        "experiment_id": metadata.get("experiment_id", ""),
        "campaign_id": metadata.get("campaign_id", ""),
        "variant": metadata.get("variant", ""),
        "trial": metadata.get("trial", ""),
        "hostname": metadata.get("hostname", ""),
    }


def latest_before(rows: list[dict[str, str]], cutoff_tick: int, *, use_end_tick: bool = False) -> dict[str, str] | None:
    if use_end_tick:
        return max(
            (row for row in rows if 0 <= event_end_tick(row) <= cutoff_tick),
            key=lambda row: (event_end_tick(row), parse_int(row.get("seq"))),
            default=None,
        )

    tick = event_end_tick if use_end_tick else event_start_tick
    lower = 0
    upper = len(rows)
    while lower < upper:
        middle = (lower + upper) // 2
        if tick(rows[middle]) <= cutoff_tick:
            lower = middle + 1
        else:
            upper = middle
    return rows[lower - 1] if lower else None


def rows_before(rows: list[dict[str, str]], cutoff_tick: int, *, use_end_tick: bool = False) -> list[dict[str, str]]:
    if use_end_tick:
        return [row for row in rows if 0 <= event_end_tick(row) <= cutoff_tick]

    tick = event_end_tick if use_end_tick else event_start_tick
    lower = 0
    upper = len(rows)
    while lower < upper:
        middle = (lower + upper) // 2
        if tick(rows[middle]) <= cutoff_tick:
            lower = middle + 1
        else:
            upper = middle
    return rows[:lower]


def select_radio_anchor(
    correlated_by_name: dict[str, list[dict[str, str]]],
    compute_tick: int,
) -> tuple[dict[str, str] | None, str]:
    outer_candidates = [
        candidate
        for name in ("UE_RF_READ_DRIFT", "UE_RF_READ")
        if (candidate := latest_before(correlated_by_name.get(name, []), compute_tick, use_end_tick=True)) is not None
    ]
    outer = max(outer_candidates, key=lambda row: (event_end_tick(row), parse_int(row.get("seq"))), default=None)
    usrp_candidates = rows_before(correlated_by_name.get("USRP_RX_RECV", []), compute_tick, use_end_tick=True)

    if outer is not None:
        outer_start = event_start_tick(outer)
        outer_end = event_end_tick(outer)
        outer_span = parse_int(outer.get("span_id"))
        parent_matches = [
            row
            for row in usrp_candidates
            if outer_span > 0
            and parse_int(row.get("parent_id")) == outer_span
            and outer_start <= event_start_tick(row)
            and event_end_tick(row) <= outer_end
        ]
        if parent_matches:
            return max(parent_matches, key=lambda row: (event_end_tick(row), parse_int(row.get("seq")))), "parent_span"
        contained = [
            row
            for row in usrp_candidates
            if outer_start <= event_start_tick(row) and event_end_tick(row) <= outer_end
        ]
        if contained:
            return (
                max(contained, key=lambda row: (event_end_tick(row), parse_int(row.get("seq")))),
                "temporal_containment",
            )
        return outer, "outer_wrapper_fallback"

    if usrp_candidates:
        return max(usrp_candidates, key=lambda row: (event_end_tick(row), parse_int(row.get("seq")))), "standalone"
    return None, "missing"


def runtime_evidence(check: dict[str, str]) -> dict[str, object]:
    flags = parse_int(check.get("flags"))
    current_ns = parse_int(check.get("aux0"), -1)
    deadline_ns = parse_int(check.get("aux1"), -1)
    lateness_ns = parse_int(check.get("aux2"))
    error_code = parse_int(check.get("aux3"))
    reasons: list[str] = []
    if (flags & DEADLINE_VALID) == 0:
        reasons.append("valid_flag_clear")
    if flags & (DEADLINE_COMPUTE_CLOCK_ERROR | DEADLINE_CHECK_CLOCK_ERROR | DEADLINE_ARITHMETIC_ERROR):
        reasons.append("failure_flag_set")
    if current_ns < 0 or deadline_ns < 0:
        reasons.append("negative_monotonic_value")
    if current_ns >= 0 and deadline_ns >= 0 and current_ns - deadline_ns != lateness_ns:
        reasons.append("lateness_mismatch")
    if bool(flags & DEADLINE_MISSED) != (lateness_ns > 0):
        reasons.append("miss_flag_mismatch")

    valid = not reasons
    return {
        "flags": flags,
        "error_code": error_code,
        "current_ns": current_ns,
        "deadline_ns": deadline_ns,
        "lateness_ns": lateness_ns,
        "valid": valid,
        "status": "valid" if valid else ";".join(reasons),
        "missed": int(lateness_ns > 0) if valid else "",
        "lateness_us": lateness_ns / 1000 if valid else math.nan,
        "headroom_us": max(-lateness_ns, 0) / 1000 if valid else math.nan,
    }


def build_check_row(
    profile_dir: str,
    metadata: dict[str, str],
    check: dict[str, str],
    computes: list[dict[str, str]],
    correlated_by_name: dict[str, list[dict[str, str]]],
    counter_hz: int,
) -> tuple[dict[str, object], tuple[int, int] | None]:
    row = identity(profile_dir, metadata)
    check_tick = event_start_tick(check)
    correlation_id = parse_int(check.get("correlation_id"))
    frame = parse_int(check.get("frame"), -1)
    slot = parse_int(check.get("slot"), -1)
    runtime = runtime_evidence(check)
    row.update(
        {
            "check_seq": check.get("seq", "0"),
            "check_tid": check.get("tid", "0"),
            "check_thread_name": check.get("thread_name", "unknown"),
            "frame": frame,
            "slot": slot,
            "absolute_slot": check.get("absolute_slot", "-1"),
            "correlation_id": correlation_id,
            "check_start_tick": check_tick,
            "deadline_flags": runtime["flags"],
            "deadline_error_code": runtime["error_code"],
            "valid_flag": int(bool(int(runtime["flags"]) & DEADLINE_VALID)),
            "missed_flag": int(bool(int(runtime["flags"]) & DEADLINE_MISSED)),
            "before_anchor_flag": int(bool(int(runtime["flags"]) & DEADLINE_BEFORE_ANCHOR)),
            "compute_clock_error_flag": int(bool(int(runtime["flags"]) & DEADLINE_COMPUTE_CLOCK_ERROR)),
            "check_clock_error_flag": int(bool(int(runtime["flags"]) & DEADLINE_CHECK_CLOCK_ERROR)),
            "arithmetic_error_flag": int(bool(int(runtime["flags"]) & DEADLINE_ARITHMETIC_ERROR)),
            "legacy_realtime_error_flag": int(bool(int(runtime["flags"]) & DEADLINE_LEGACY_REALTIME_ERROR)),
            "runtime_status": runtime["status"],
            "runtime_valid": int(bool(runtime["valid"])),
            "runtime_current_monotonic_raw_ns": runtime["current_ns"],
            "runtime_deadline_monotonic_raw_ns": runtime["deadline_ns"],
            "runtime_signed_lateness_ns": runtime["lateness_ns"],
            "runtime_signed_lateness_us": runtime["lateness_us"],
            "runtime_headroom_us": runtime["headroom_us"],
            "runtime_missed": runtime["missed"],
            "counter_hz": counter_hz,
        }
    )

    candidates = rows_before(computes, check_tick)
    matching = [
        candidate
        for candidate in candidates
        if parse_int(candidate.get("frame"), -1) == frame and parse_int(candidate.get("slot"), -1) == slot
    ]
    compute = matching[-1] if correlation_id > 0 and matching else None
    if correlation_id <= 0:
        compute_status = "zero_correlation_id"
    elif not candidates:
        compute_status = "missing"
    elif not matching:
        compute_status = "frame_slot_mismatch"
    elif len(matching) > 1:
        compute_status = "multiple_candidates_latest_selected"
    else:
        compute_status = "paired"
    row["compute_status"] = compute_status
    row["compute_candidate_count"] = len(matching)

    used_compute: tuple[int, int] | None = None
    if compute is None:
        row.update(
            {
                "compute_seq": "",
                "compute_start_tick": "",
                "compute_flags": "",
                "radio_anchor_timestamp": "",
                "radio_deadline_timestamp": "",
                "radio_sample_offset": "",
                "samples_per_subframe": "",
                "sample_rate_hz": "",
            }
        )
    else:
        compute_tick = event_start_tick(compute)
        compute_flags = parse_int(compute.get("flags"))
        radio_anchor_timestamp = parse_int(compute.get("aux0"))
        radio_deadline_timestamp = parse_int(compute.get("aux1"))
        samples_per_subframe = parse_int(compute.get("aux2"))
        row.update(
            {
                "compute_seq": compute.get("seq", "0"),
                "compute_start_tick": compute_tick,
                "compute_flags": compute_flags,
                "radio_anchor_timestamp": radio_anchor_timestamp,
                "radio_deadline_timestamp": radio_deadline_timestamp,
                "radio_sample_offset": radio_deadline_timestamp - radio_anchor_timestamp,
                "samples_per_subframe": samples_per_subframe,
                "sample_rate_hz": samples_per_subframe * 1000 if samples_per_subframe > 0 else "",
            }
        )
        used_compute = (parse_int(compute.get("seq")), compute_tick)

    reconstruction_status = "missing_compute"
    anchor: dict[str, str] | None = None
    anchor_match = "missing"
    reconstructed_lateness_us = math.nan
    reconstructed_headroom_us = math.nan
    reconstructed_missed: int | str = ""
    reconstructed_offset_tick: int | str = ""
    reconstructed_deadline_tick: int | str = ""
    reconstructed_lateness_tick: int | str = ""
    if compute is not None:
        compute_tick = event_start_tick(compute)
        compute_flags = parse_int(compute.get("flags"))
        radio_anchor_timestamp = parse_int(compute.get("aux0"))
        radio_deadline_timestamp = parse_int(compute.get("aux1"))
        samples_per_subframe = parse_int(compute.get("aux2"))
        anchor, anchor_match = select_radio_anchor(correlated_by_name, compute_tick)
        if compute_flags & DEADLINE_ARITHMETIC_ERROR:
            reconstruction_status = "compute_arithmetic_error"
        elif counter_hz <= 0:
            reconstruction_status = "invalid_counter_hz"
        elif samples_per_subframe <= 0:
            reconstruction_status = "invalid_samples_per_subframe"
        elif anchor is None:
            reconstruction_status = "missing_radio_anchor_event"
        else:
            anchor_end = event_end_tick(anchor)
            sample_offset = radio_deadline_timestamp - radio_anchor_timestamp
            reconstructed_offset_tick = round_div_signed(
                sample_offset * counter_hz,
                samples_per_subframe * 1000,
            )
            reconstructed_deadline_tick = anchor_end + reconstructed_offset_tick
            if anchor_end < 0 or reconstructed_deadline_tick < 0 or check_tick < 0:
                reconstruction_status = "invalid_tick_interval"
            else:
                reconstructed_lateness_tick = check_tick - reconstructed_deadline_tick
                reconstructed_lateness_us = reconstructed_lateness_tick * 1e6 / counter_hz
                reconstructed_headroom_us = max(-reconstructed_lateness_tick, 0) * 1e6 / counter_hz
                reconstructed_missed = int(reconstructed_lateness_tick > 0)
                reconstruction_status = "valid"

    anchor_source = anchor.get("event_name", "") if anchor is not None else ""
    anchor_start = event_start_tick(anchor) if anchor is not None else -1
    anchor_end = event_end_tick(anchor) if anchor is not None else -1
    compute_tick_value = event_start_tick(compute) if compute is not None else -1
    row.update(
        {
            "radio_anchor_source": anchor_source,
            "radio_anchor_match": anchor_match,
            "radio_anchor_seq": anchor.get("seq", "") if anchor is not None else "",
            "radio_anchor_start_tick": anchor_start if anchor is not None else "",
            "radio_anchor_end_tick": anchor_end if anchor is not None else "",
            "radio_anchor_to_compute_us": (compute_tick_value - anchor_end) * 1e6 / counter_hz
            if anchor is not None and counter_hz > 0
            else math.nan,
            "compute_to_check_us": (check_tick - compute_tick_value) * 1e6 / counter_hz
            if compute is not None and counter_hz > 0
            else math.nan,
            "reconstruction_status": reconstruction_status,
            "reconstruction_valid": int(reconstruction_status == "valid"),
            "reconstructed_offset_tick": reconstructed_offset_tick,
            "reconstructed_deadline_tick": reconstructed_deadline_tick,
            "reconstructed_signed_lateness_tick": reconstructed_lateness_tick,
            "reconstructed_signed_lateness_us": reconstructed_lateness_us,
            "reconstructed_headroom_us": reconstructed_headroom_us,
            "reconstructed_missed": reconstructed_missed,
        }
    )

    comparable = bool(runtime["valid"]) and reconstruction_status == "valid"
    agreement: int | str = int(runtime["missed"] == reconstructed_missed) if comparable else ""
    row["classification_agreement"] = agreement
    row["reconstruction_minus_runtime_lateness_us"] = (
        reconstructed_lateness_us - float(runtime["lateness_us"]) if comparable else math.nan
    )
    if compute is None:
        status = "unpaired"
    elif comparable and agreement:
        status = "ok"
    elif comparable:
        status = "classification_disagreement"
    elif reconstruction_status == "valid":
        status = "runtime_invalid"
    elif runtime["valid"]:
        status = "reconstruction_invalid"
    else:
        status = "invalid"
    row["status"] = status
    return row, used_compute


def is_nrue_profile(metadata: dict[str, str]) -> bool:
    return metadata.get("role") == "nrUE" or "uesoftmodem" in metadata.get("process_name", "")


def build_deadline_reports(
    events_by_dir: dict[str, list[dict[str, str]]],
    metadata_by_dir: dict[str, dict[str, str]],
) -> DeadlineReports:
    check_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for profile_dir in sorted(events_by_dir):
        metadata = metadata_by_dir.get(profile_dir, {})
        if not is_nrue_profile(metadata):
            continue
        events = events_by_dir[profile_dir]
        by_name: dict[str, list[dict[str, str]]] = defaultdict(list)
        computes_by_correlation: dict[int, list[dict[str, str]]] = defaultdict(list)
        by_correlation_name: dict[int, dict[str, list[dict[str, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for event in events:
            name = event.get("event_name", "")
            by_name[name].append(event)
            correlation_id = parse_int(event.get("correlation_id"))
            if correlation_id > 0:
                by_correlation_name[correlation_id][name].append(event)
            if name == "UE_TX_DEADLINE_COMPUTE":
                computes_by_correlation[correlation_id].append(event)
        for rows in by_name.values():
            rows.sort(key=lambda row: (event_start_tick(row), parse_int(row.get("seq"))))
        for rows in computes_by_correlation.values():
            rows.sort(key=lambda row: (event_start_tick(row), parse_int(row.get("seq"))))
        for correlated_events in by_correlation_name.values():
            for rows in correlated_events.values():
                rows.sort(key=lambda row: (event_start_tick(row), parse_int(row.get("seq"))))

        counter_hz = parse_int(metadata.get("counter_hz"))
        used_computes: set[tuple[int, int]] = set()
        profile_checks: list[dict[str, object]] = []
        for check in by_name.get("UE_TX_DEADLINE_CHECK", []):
            correlation_id = parse_int(check.get("correlation_id"))
            row, used_compute = build_check_row(
                profile_dir,
                metadata,
                check,
                computes_by_correlation.get(correlation_id, []),
                by_correlation_name.get(correlation_id, {}),
                counter_hz,
            )
            profile_checks.append(row)
            if used_compute is not None:
                used_computes.add(used_compute)
        check_rows.extend(profile_checks)

        compute_count = len(by_name.get("UE_TX_DEADLINE_COMPUTE", []))
        check_count = len(profile_checks)
        runtime_valid = [row for row in profile_checks if row["runtime_valid"] == 1]
        reconstructed_valid = [row for row in profile_checks if row["reconstruction_valid"] == 1]
        comparable = [row for row in profile_checks if row["classification_agreement"] != ""]
        agreements = [row for row in comparable if row["classification_agreement"] == 1]
        anchor_counts = defaultdict(int)
        for row in profile_checks:
            if row["radio_anchor_source"]:
                anchor_counts[str(row["radio_anchor_source"])] += 1

        summary = identity(profile_dir, metadata)
        summary.update(
            {
                "compute_events": compute_count,
                "check_events": check_count,
                "legacy_miss_events": len(by_name.get("UE_TX_DEADLINE_MISS", [])),
                "paired_checks": sum(row["compute_seq"] != "" for row in profile_checks),
                "unpaired_checks": sum(row["compute_seq"] == "" for row in profile_checks),
                "unmatched_compute_events": max(compute_count - len(used_computes), 0),
                "runtime_valid_checks": len(runtime_valid),
                "runtime_invalid_checks": check_count - len(runtime_valid),
                "runtime_misses": sum(row["runtime_missed"] == 1 for row in runtime_valid),
                "runtime_miss_rate": sum(row["runtime_missed"] == 1 for row in runtime_valid) / len(runtime_valid)
                if runtime_valid
                else math.nan,
                "reconstruction_valid_checks": len(reconstructed_valid),
                "reconstruction_invalid_checks": check_count - len(reconstructed_valid),
                "reconstructed_misses": sum(row["reconstructed_missed"] == 1 for row in reconstructed_valid),
                "reconstructed_miss_rate": sum(row["reconstructed_missed"] == 1 for row in reconstructed_valid)
                / len(reconstructed_valid)
                if reconstructed_valid
                else math.nan,
                "classification_comparable_checks": len(comparable),
                "classification_agreements": len(agreements),
                "classification_disagreements": len(comparable) - len(agreements),
                "classification_agreement_rate": len(agreements) / len(comparable) if comparable else math.nan,
                "usrp_receive_anchors": anchor_counts["USRP_RX_RECV"],
                "ue_rf_read_drift_anchors": anchor_counts["UE_RF_READ_DRIFT"],
                "ue_rf_read_anchors": anchor_counts["UE_RF_READ"],
            }
        )
        for name in _SUMMARY_DISTRIBUTIONS:
            values = [
                float(row[name])
                for row in profile_checks
                if isinstance(row.get(name), (int, float)) and math.isfinite(float(row[name]))
            ]
            summary.update({f"{name}_{stat}": value for stat, value in distribution(values).items()})

        if check_count == 0:
            if compute_count:
                summary["status"] = "compute_only_no_hardware_check"
            elif summary["legacy_miss_events"]:
                summary["status"] = "legacy_only"
            else:
                summary["status"] = "not_recorded"
        elif (
            len(runtime_valid) == check_count
            and len(reconstructed_valid) == check_count
            and len(agreements) == check_count
            and summary["unpaired_checks"] == 0
            and summary["unmatched_compute_events"] == 0
        ):
            summary["status"] = "ok"
        else:
            summary["status"] = "partial"
        summary_rows.append(summary)

    return DeadlineReports(check_rows, summary_rows)
