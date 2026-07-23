#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Build publication-oriented reports from OAI profiler side streams.

The module deliberately performs no measurement and no implicit clock
alignment. It summarizes archived evidence and labels unavailable, invalid,
multiplexed, or unresolved observations instead of filling data gaps.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from oai_profile_archive import (
    MANIFEST_NAME,
    has_symlink_component,
    is_safe_relative_path,
    verify_archive,
)

THREAD_METRIC_CPU_FREQUENCY_VALID = 1 << 3
CAMPAIGN_SUCCESS_STOP_REASONS = {"duration_elapsed", "measurement_complete"}
CAMPAIGN_VALID_DURATION_STATUSES = {"valid", "legacy_realtime_fallback"}


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


@dataclass(frozen=True)
class Report:
    fields: list[str]
    rows: list[dict[str, object]]


def parse_int(value: object, default: int = 0) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def parse_float(value: object, default: float = math.nan) -> float:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def campaign_member_succeeded(row: dict[str, object]) -> bool:
    return (
        row.get("status") == "finished"
        and parse_int(row.get("return_code"), -1) == 0
        and row.get("stop_reason") in CAMPAIGN_SUCCESS_STOP_REASONS
        and row.get("duration_status") in CAMPAIGN_VALID_DURATION_STATUSES
    )


def elapsed_duration(
    start_monotonic_ns: int,
    end_monotonic_ns: int,
    start_realtime_ns: int,
    end_realtime_ns: int,
) -> tuple[float, str, str]:
    if start_monotonic_ns > 0 or end_monotonic_ns > 0:
        if start_monotonic_ns > 0 and end_monotonic_ns >= start_monotonic_ns:
            return (
                (end_monotonic_ns - start_monotonic_ns) / 1e9,
                "CLOCK_MONOTONIC_RAW",
                "valid",
            )
        return math.nan, "CLOCK_MONOTONIC_RAW", "invalid_monotonic_bounds"
    if start_realtime_ns > 0 or end_realtime_ns > 0:
        if start_realtime_ns > 0 and end_realtime_ns >= start_realtime_ns:
            return (
                (end_realtime_ns - start_realtime_ns) / 1e9,
                "CLOCK_REALTIME",
                "legacy_realtime_fallback",
            )
        return math.nan, "CLOCK_REALTIME", "invalid_realtime_bounds"
    return math.nan, "unavailable", "unavailable"


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
        return {
            "min": math.nan,
            "p50": math.nan,
            "p90": math.nan,
            "p95": math.nan,
            "p99": math.nan,
            "p99_9": math.nan,
            "max": math.nan,
            "mean": math.nan,
            "stdev": math.nan,
        }
    average = sum(samples) / len(samples)
    variance = (
        sum((value - average) ** 2 for value in samples) / (len(samples) - 1)
        if len(samples) > 1
        else math.nan
    )
    return {
        "min": samples[0],
        "p50": quantile(samples, 0.5),
        "p90": quantile(samples, 0.9),
        "p95": quantile(samples, 0.95),
        "p99": quantile(samples, 0.99),
        "p99_9": quantile(samples, 0.999),
        "max": samples[-1],
        "mean": average,
        "stdev": math.sqrt(variance) if math.isfinite(variance) else math.nan,
    }


def prefixed_distribution(values: Iterable[float], prefix: str, unit: str = "") -> dict[str, float]:
    suffix = f"_{unit}" if unit else ""
    return {f"{prefix}_{name}{suffix}": value for name, value in distribution(values).items()}


def joined_status(values: Iterable[str], default: str = "") -> str:
    unique = sorted({value for value in values if value})
    if not unique:
        return default
    return unique[0] if len(unique) == 1 else ";".join(unique)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_campaign(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        return {}, "missing"
    try:
        value = json.loads(path.read_text(errors="replace"))
    except (OSError, json.JSONDecodeError) as error:
        return {}, f"parse_error:{error}"
    if not isinstance(value, dict):
        return {}, "parse_error:not_an_object"
    return value, "ok"


def discover_run_dirs(inputs: Iterable[Path], profile_dirs: Iterable[Path]) -> list[Path]:
    discovered = {path.resolve() for path in profile_dirs}
    for input_path in inputs:
        path = input_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            if path.name in {"campaign_run.json", MANIFEST_NAME, "metadata.txt"}:
                discovered.add(path.parent)
            continue
        if any((path / marker).is_file() for marker in ("campaign_run.json", MANIFEST_NAME)):
            discovered.add(path)
        for marker in ("campaign_run.json", MANIFEST_NAME):
            discovered.update(found.parent.resolve() for found in path.rglob(marker))
    return sorted(discovered)


def identity_for(
    run_dir: Path,
    metadata_by_dir: dict[str, dict[str, str]],
    campaign_by_dir: dict[str, dict[str, Any]],
) -> dict[str, object]:
    key = str(run_dir)
    metadata = metadata_by_dir.get(key, {})
    campaign = campaign_by_dir.get(key, {})
    return {
        "profile_dir": key,
        "run_id": metadata.get("run_id", str(campaign.get("run_id", run_dir.name))),
        "experiment_id": metadata.get("experiment_id", str(campaign.get("experiment_id", ""))),
        "campaign_id": metadata.get("campaign_id", str(campaign.get("campaign_id", ""))),
        "variant": metadata.get("variant", str(campaign.get("variant", ""))),
        "trial": metadata.get("trial", str(campaign.get("trial", ""))),
        "role": metadata.get("role", str(campaign.get("role", "unknown"))),
        "hostname": metadata.get("hostname", str(campaign.get("hostname", "unknown"))),
    }


def campaign_report(
    run_dirs: list[Path],
    metadata_by_dir: dict[str, dict[str, str]],
    campaign_by_dir: dict[str, dict[str, Any]],
    campaign_status_by_dir: dict[str, str],
) -> Report:
    fields = IDENTITY_FIELDS + [
        "campaign_metadata_status",
        "runner_version",
        "case",
        "host",
        "profile_enabled",
        "pmu_mode",
        "status",
        "return_code",
        "stop_reason",
        "start_realtime_ns",
        "end_realtime_ns",
        "start_monotonic_raw_ns",
        "end_monotonic_raw_ns",
        "duration_s",
        "duration_clock",
        "duration_status",
        "realtime_clock_regressed",
        "anchor_clock_scope",
        "launch_index",
        "sidecar_tool",
        "sidecar_status",
        "sidecar_artifact",
        "archive_status",
        "archive_manifest_present",
        "profiler_metadata_present",
        "events_present",
        "command_json",
        "environment_json",
        "notes_json",
    ]
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        key = str(run_dir)
        campaign = campaign_by_dir[key]
        start_ns = parse_int(campaign.get("start_realtime_ns"))
        end_ns = parse_int(campaign.get("end_realtime_ns"))
        start_monotonic_ns = parse_int(campaign.get("start_monotonic_raw_ns"))
        end_monotonic_ns = parse_int(campaign.get("end_monotonic_raw_ns"))
        duration_s, duration_clock, duration_status = elapsed_duration(
            start_monotonic_ns,
            end_monotonic_ns,
            start_ns,
            end_ns,
        )
        rows.append(
            {
                **identity_for(run_dir, metadata_by_dir, campaign_by_dir),
                "campaign_metadata_status": campaign_status_by_dir[key],
                "runner_version": campaign.get("runner_version", ""),
                "case": campaign.get("case", ""),
                "host": campaign.get("host", ""),
                "profile_enabled": campaign.get("profile_enabled", bool(metadata_by_dir.get(key))),
                "pmu_mode": campaign.get("pmu_mode", metadata_by_dir.get(key, {}).get("pmu_mode", "")),
                "status": campaign.get("status", "campaign_metadata_missing"),
                "return_code": campaign.get("return_code", ""),
                "stop_reason": campaign.get("stop_reason", ""),
                "start_realtime_ns": start_ns,
                "end_realtime_ns": end_ns,
                "start_monotonic_raw_ns": start_monotonic_ns,
                "end_monotonic_raw_ns": end_monotonic_ns,
                "duration_s": duration_s,
                "duration_clock": duration_clock,
                "duration_status": duration_status,
                "realtime_clock_regressed": int(start_ns > 0 and end_ns < start_ns),
                "anchor_clock_scope": campaign.get("anchor_clock_scope", ""),
                "launch_index": campaign.get("launch_index", ""),
                "sidecar_tool": campaign.get("sidecar_tool", "none"),
                "sidecar_status": campaign.get("sidecar_status", ""),
                "sidecar_artifact": campaign.get("sidecar_artifact", ""),
                "archive_status": campaign.get("archive_status", ""),
                "archive_manifest_present": int((run_dir / MANIFEST_NAME).is_file()),
                "profiler_metadata_present": int((run_dir / "metadata.txt").is_file()),
                "events_present": int((run_dir / "events.csv").is_file()),
                "command_json": json.dumps(campaign.get("command", []), sort_keys=True),
                "environment_json": json.dumps(campaign.get("environment", {}), sort_keys=True),
                "notes_json": json.dumps(campaign.get("notes", []), sort_keys=True),
            }
        )
    return Report(fields, rows)


def clock_quality_report(
    profile_dirs: list[Path],
    metadata_by_dir: dict[str, dict[str, str]],
    campaign_by_dir: dict[str, dict[str, Any]],
) -> Report:
    fields = IDENTITY_FIELDS + [
        "sync_present",
        "sample_count",
        "valid_sample_count",
        "bounded_sample_count",
        "clock_read_error_count",
        "realtime_regression_count",
        "monotonic_regression_count",
        "tick_regression_count",
        "uncertainty_min_ns",
        "uncertainty_p50_ns",
        "uncertainty_p99_ns",
        "uncertainty_max_ns",
        "offset_first_ns",
        "offset_last_ns",
        "offset_drift_ns",
        "fit_reference_monotonic_ns",
        "fit_reference_realtime_ns",
        "fit_slope_realtime_per_monotonic",
        "fit_rate_error_ppm",
        "fit_rmse_ns",
        "fit_max_abs_residual_ns",
        "status",
    ]
    rows: list[dict[str, object]] = []
    for profile_dir in profile_dirs:
        sync_path = profile_dir / "sync.csv"
        samples = read_csv(sync_path)
        valid: list[tuple[int, int, int]] = []
        uncertainties: list[float] = []
        bounded_count = 0
        read_errors = 0
        realtime_regressions = 0
        monotonic_regressions = 0
        tick_regressions = 0
        previous_realtime = 0
        previous_monotonic = 0
        previous_tick = 0
        for sample in samples:
            realtime_ns = parse_int(sample.get("realtime_ns"))
            monotonic_ns = parse_int(sample.get("monotonic_raw_ns"))
            tick = parse_int(sample.get("tick"))
            row_status = sample.get("status", "")
            if row_status not in {"", "ok"}:
                read_errors += 1
                continue
            if realtime_ns <= 0 or monotonic_ns <= 0 or tick <= 0:
                read_errors += 1
                continue
            if previous_realtime and realtime_ns < previous_realtime:
                realtime_regressions += 1
            if previous_monotonic and monotonic_ns <= previous_monotonic:
                monotonic_regressions += 1
            if previous_tick and tick < previous_tick:
                tick_regressions += 1
            previous_realtime = realtime_ns
            previous_monotonic = monotonic_ns
            previous_tick = tick
            valid.append((monotonic_ns, realtime_ns, tick))
            before_ns = parse_int(sample.get("monotonic_raw_before_ns"))
            after_ns = parse_int(sample.get("monotonic_raw_after_ns"))
            uncertainty_ns = parse_int(sample.get("monotonic_raw_uncertainty_ns"), -1)
            if before_ns > 0 and after_ns >= before_ns and uncertainty_ns == after_ns - before_ns:
                bounded_count += 1
                uncertainties.append(float(uncertainty_ns))

        uncertainty = distribution(uncertainties)
        offset_first = valid[0][1] - valid[0][0] if valid else math.nan
        offset_last = valid[-1][1] - valid[-1][0] if valid else math.nan
        offset_drift = offset_last - offset_first if valid else math.nan
        fit_slope = math.nan
        fit_rate_error_ppm = math.nan
        fit_rmse_ns = math.nan
        fit_max_residual_ns = math.nan
        if len(valid) >= 2:
            reference_monotonic = valid[0][0]
            reference_realtime = valid[0][1]
            x = [(sample[0] - reference_monotonic) / 1e9 for sample in valid]
            y = [(sample[1] - reference_realtime) / 1e9 for sample in valid]
            x_mean = sum(x) / len(x)
            y_mean = sum(y) / len(y)
            denominator = sum((value - x_mean) ** 2 for value in x)
            if denominator > 0:
                fit_slope = sum(
                    (x_value - x_mean) * (y_value - y_mean)
                    for x_value, y_value in zip(x, y)
                ) / denominator
                residuals_ns = [
                    (y_value - (y_mean + fit_slope * (x_value - x_mean))) * 1e9
                    for x_value, y_value in zip(x, y)
                ]
                fit_rate_error_ppm = (fit_slope - 1.0) * 1e6
                fit_rmse_ns = math.sqrt(
                    sum(residual * residual for residual in residuals_ns) / len(residuals_ns)
                )
                fit_max_residual_ns = max(abs(residual) for residual in residuals_ns)
        if not sync_path.is_file():
            status = "missing"
        elif len(valid) < 2 or not math.isfinite(fit_slope):
            status = "insufficient_valid_samples"
        elif realtime_regressions or monotonic_regressions or tick_regressions:
            status = "clock_regression"
        elif bounded_count == 0:
            status = "legacy_unbounded"
        elif bounded_count < len(valid):
            status = "partially_bounded"
        else:
            status = "ok"
        rows.append(
            {
                **identity_for(profile_dir, metadata_by_dir, campaign_by_dir),
                "sync_present": int(sync_path.is_file()),
                "sample_count": len(samples),
                "valid_sample_count": len(valid),
                "bounded_sample_count": bounded_count,
                "clock_read_error_count": read_errors,
                "realtime_regression_count": realtime_regressions,
                "monotonic_regression_count": monotonic_regressions,
                "tick_regression_count": tick_regressions,
                "uncertainty_min_ns": uncertainty["min"],
                "uncertainty_p50_ns": uncertainty["p50"],
                "uncertainty_p99_ns": uncertainty["p99"],
                "uncertainty_max_ns": uncertainty["max"],
                "offset_first_ns": offset_first,
                "offset_last_ns": offset_last,
                "offset_drift_ns": offset_drift,
                "fit_reference_monotonic_ns": valid[0][0] if valid else "",
                "fit_reference_realtime_ns": valid[0][1] if valid else "",
                "fit_slope_realtime_per_monotonic": fit_slope,
                "fit_rate_error_ppm": fit_rate_error_ppm,
                "fit_rmse_ns": fit_rmse_ns,
                "fit_max_abs_residual_ns": fit_max_residual_ns,
                "status": status,
            }
        )
    return Report(fields, rows)


HOST_DISTRIBUTIONS = (
    ("thermal_zone0", "thermal_zone0_millicelsius", "millicelsius", "thermal"),
    ("thermal_max", "thermal_max_millicelsius", "millicelsius", "thermal"),
    ("cpu_frequency_sample_min", "cpu_frequency_min_khz", "khz", "frequency"),
    ("cpu_frequency_sample_avg", "cpu_frequency_avg_khz", "khz", "frequency"),
    ("cpu_frequency_sample_max", "cpu_frequency_max_khz", "khz", "frequency"),
    ("cpu_busy", "cpu_busy_percent", "percent", "cpu_busy"),
    ("load1", "load1", "", "load1"),
    ("load5", "load5", "", "load5"),
    ("load15", "load15", "", "load15"),
    ("mem_available", "mem_available_kb", "kb", "nonnegative"),
    ("swap_free", "swap_free_kb", "kb", "nonnegative"),
    ("process_rss", "process_rss_kb", "kb", "nonnegative"),
    ("process_maxrss", "process_maxrss_kb", "kb", "rusage"),
    ("acquisition_duration", "acquisition_duration_us", "us", "acquisition"),
)

HOST_COUNTERS = (
    ("process_user", "process_user_us", "us"),
    ("process_system", "process_system_us", "us"),
    ("voluntary_context_switches", "voluntary_context_switches", "count"),
    ("involuntary_context_switches", "involuntary_context_switches", "count"),
    ("minor_faults", "minor_faults", "count"),
    ("major_faults", "major_faults", "count"),
    ("block_input_ops", "block_input_ops", "count"),
    ("block_output_ops", "block_output_ops", "count"),
)

RPI_THROTTLED_BITS = (
    ("undervoltage_now", 1 << 0),
    ("arm_frequency_capped_now", 1 << 1),
    ("throttled_now", 1 << 2),
    ("soft_temperature_limit_now", 1 << 3),
    ("undervoltage_occurred", 1 << 16),
    ("arm_frequency_capped_occurred", 1 << 17),
    ("throttled_occurred", 1 << 18),
    ("soft_temperature_limit_occurred", 1 << 19),
)

HOST_ERROR_BITS = (
    ("realtime_clock", 1 << 0),
    ("monotonic_start", 1 << 1),
    ("monotonic_end", 1 << 2),
    ("monotonic_regression", 1 << 3),
    ("counter_regression", 1 << 4),
    ("loadavg", 1 << 5),
    ("getrusage", 1 << 6),
)

DISTRIBUTION_NAMES = tuple(distribution(()).keys())


def distribution_columns(prefix: str, unit: str = "") -> list[str]:
    suffix = f"_{unit}" if unit else ""
    return [f"{prefix}_{name}{suffix}" for name in DISTRIBUTION_NAMES]


def host_acquisition_status(row: dict[str, str]) -> str:
    required = (
        "end_monotonic_raw_ns",
        "end_tick",
        "acquisition_duration_monotonic_raw_ns",
        "acquisition_duration_tick",
        "acquisition_duration_us",
    )
    if not any(field in row for field in required):
        return "legacy_unbounded"
    if not all(field in row and row.get(field, "") != "" for field in required):
        return "incomplete"

    error_mask = parse_int(row.get("error_mask"))
    if error_mask & ((1 << 1) | (1 << 2) | (1 << 3)):
        return "monotonic_invalid"
    if error_mask & (1 << 4):
        return "counter_invalid"

    start_monotonic_ns = parse_int(row.get("monotonic_raw_ns"), -1)
    end_monotonic_ns = parse_int(row.get("end_monotonic_raw_ns"), -1)
    duration_monotonic_ns = parse_int(row.get("acquisition_duration_monotonic_raw_ns"), -1)
    if (
        start_monotonic_ns <= 0
        or end_monotonic_ns < start_monotonic_ns
        or duration_monotonic_ns != end_monotonic_ns - start_monotonic_ns
    ):
        return "monotonic_duration_mismatch"

    start_tick = parse_int(row.get("tick"), -1)
    end_tick = parse_int(row.get("end_tick"), -1)
    duration_tick = parse_int(row.get("acquisition_duration_tick"), -1)
    if start_tick < 0 or end_tick < start_tick or duration_tick != end_tick - start_tick:
        return "counter_duration_mismatch"

    duration_us = parse_float(row.get("acquisition_duration_us"))
    if not math.isfinite(duration_us) or duration_us < 0:
        return "duration_us_invalid"
    if abs(duration_us * 1000.0 - duration_monotonic_ns) > 0.01:
        return "duration_us_mismatch"
    return "ok"


def host_metric_value(row: dict[str, str], source: str, gate: str) -> float | None:
    value = parse_float(row.get(source))
    if not math.isfinite(value) or value < 0:
        return None
    if gate == "thermal" and parse_int(row.get("thermal_samples")) <= 0:
        return None
    if gate == "frequency" and parse_int(row.get("cpu_frequency_samples")) <= 0:
        return None
    if gate == "cpu_busy" and value > 100.0:
        return None
    if gate in {"load1", "load5", "load15"} and "getloadavg_count" in row:
        required = {"load1": 1, "load5": 2, "load15": 3}[gate]
        if parse_int(row.get("getloadavg_count"), -1) < required:
            return None
    if gate == "rusage" and "getrusage_status" in row and row.get("getrusage_status") != "ok":
        return None
    if gate == "acquisition" and host_acquisition_status(row) != "ok":
        return None
    return value


def host_counter_report(
    rows: list[dict[str, str]],
    prefix: str,
    source: str,
    unit: str,
) -> dict[str, object]:
    valid_samples = 0
    invalid_samples = 0
    valid_intervals = 0
    invalid_intervals = 0
    delta_total = 0.0
    interval_total_ns = 0
    previous: tuple[int, float] | None = None

    for row in rows:
        monotonic_ns = parse_int(row.get("monotonic_raw_ns"), -1)
        value = parse_float(row.get(source))
        acquisition_status = host_acquisition_status(row)
        rusage_valid = "getrusage_status" not in row or row.get("getrusage_status") == "ok"
        acquisition_valid = acquisition_status in {"ok", "legacy_unbounded"}
        if (
            monotonic_ns <= 0
            or not math.isfinite(value)
            or value < 0
            or not rusage_valid
            or not acquisition_valid
        ):
            invalid_samples += 1
            previous = None
            continue

        valid_samples += 1
        if previous is not None:
            previous_monotonic_ns, previous_value = previous
            interval_ns = monotonic_ns - previous_monotonic_ns
            delta = value - previous_value
            if interval_ns <= 0 or delta < 0:
                invalid_intervals += 1
            else:
                valid_intervals += 1
                delta_total += delta
                interval_total_ns += interval_ns
        previous = (monotonic_ns, value)

    interval_total_s = interval_total_ns / 1e9
    rate = delta_total / interval_total_s if interval_total_s > 0 else math.nan
    return {
        f"{prefix}_valid_samples": valid_samples,
        f"{prefix}_invalid_samples": invalid_samples,
        f"{prefix}_valid_intervals": valid_intervals,
        f"{prefix}_invalid_intervals": invalid_intervals,
        f"{prefix}_delta_total_{unit}": delta_total,
        f"{prefix}_observed_s": interval_total_s,
        f"{prefix}_rate_per_second": rate,
    }


def host_metrics_report(
    profile_dirs: list[Path],
    metadata_by_dir: dict[str, dict[str, str]],
    campaign_by_dir: dict[str, dict[str, Any]],
) -> Report:
    quality_fields = [
        "stream_status",
        "sample_count",
        "monotonic_valid_samples",
        "fully_valid_samples",
        "bounded_acquisition_samples",
        "invalid_bounded_acquisition_samples",
        "legacy_acquisition_unbounded_samples",
        "acquisition_statuses",
        "sample_statuses",
        "error_mask_nonzero_samples",
        "error_mask_or",
        "error_mask_unknown_bits_or",
        "first_monotonic_raw_ns",
        "last_monotonic_raw_ns",
        "observation_span_s",
        "sampling_cadence_valid_intervals",
        "sampling_cadence_invalid_intervals",
        "sampling_cadence_missing_samples",
        "writer_cpu_observed_samples",
        "writer_cpu_migration_samples",
        "writer_cpu_migration_percent",
        "getloadavg_full_samples",
        "getloadavg_partial_samples",
        "getloadavg_error_samples",
        "getloadavg_unverified_legacy_samples",
        "getrusage_ok_samples",
        "getrusage_error_samples",
        "getrusage_unverified_legacy_samples",
        "rpi_throttled_valid_samples",
        "rpi_throttled_raw_or",
        "rpi_throttled_unknown_bits_or",
        "quality_status",
        "acquisition_semantics",
        "process_counter_semantics",
        "probe_simultaneity",
    ]
    error_fields = [f"error_{name}_samples" for name, _ in HOST_ERROR_BITS]
    rpi_fields: list[str] = []
    for name, _ in RPI_THROTTLED_BITS:
        rpi_fields.extend((f"rpi_{name}_samples", f"rpi_{name}_percent"))

    metric_fields: list[str] = []
    for prefix, _, unit, _ in HOST_DISTRIBUTIONS:
        metric_fields.append(f"{prefix}_valid_samples")
        metric_fields.extend(distribution_columns(prefix, unit))
    metric_fields.extend(distribution_columns("sampling_cadence", "s"))

    counter_fields: list[str] = []
    for prefix, _, unit in HOST_COUNTERS:
        counter_fields.extend(
            (
                f"{prefix}_valid_samples",
                f"{prefix}_invalid_samples",
                f"{prefix}_valid_intervals",
                f"{prefix}_invalid_intervals",
                f"{prefix}_delta_total_{unit}",
                f"{prefix}_observed_s",
                f"{prefix}_rate_per_second",
            )
        )
    counter_fields.extend(
        (
            "process_user_cpu_percent",
            "process_system_cpu_percent",
            "process_total_cpu_percent",
        )
    )

    fields = (
        IDENTITY_FIELDS
        + quality_fields
        + error_fields
        + rpi_fields
        + metric_fields
        + counter_fields
    )
    rows_out: list[dict[str, object]] = []
    known_error_mask = sum(mask for _, mask in HOST_ERROR_BITS)
    known_rpi_mask = sum(mask for _, mask in RPI_THROTTLED_BITS)

    for profile_dir in profile_dirs:
        path = profile_dir / "host_metrics.csv"
        samples = read_csv(path)
        output: dict[str, object] = {
            **identity_for(profile_dir, metadata_by_dir, campaign_by_dir),
            **{field: "" for field in fields if field not in IDENTITY_FIELDS},
            "stream_status": "recorded" if samples else ("empty" if path.is_file() else "missing"),
            "sample_count": len(samples),
            "monotonic_valid_samples": 0,
            "fully_valid_samples": 0,
            "bounded_acquisition_samples": 0,
            "invalid_bounded_acquisition_samples": 0,
            "legacy_acquisition_unbounded_samples": 0,
            "sampling_cadence_valid_intervals": 0,
            "sampling_cadence_invalid_intervals": 0,
            "sampling_cadence_missing_samples": 0,
            "writer_cpu_observed_samples": 0,
            "writer_cpu_migration_samples": 0,
            "getloadavg_full_samples": 0,
            "getloadavg_partial_samples": 0,
            "getloadavg_error_samples": 0,
            "getloadavg_unverified_legacy_samples": 0,
            "getrusage_ok_samples": 0,
            "getrusage_error_samples": 0,
            "getrusage_unverified_legacy_samples": 0,
            "rpi_throttled_valid_samples": 0,
            "rpi_throttled_raw_or": 0,
            "rpi_throttled_unknown_bits_or": 0,
            "error_mask_nonzero_samples": 0,
            "error_mask_or": 0,
            "error_mask_unknown_bits_or": 0,
            "acquisition_semantics": "sequential_probes_bounded_by_start_and_end_when_available",
            "process_counter_semantics": "aggregate_process_cumulative_delta_per_monotonic_elapsed_time",
            "probe_simultaneity": "not_simultaneous",
        }
        for name, _ in HOST_ERROR_BITS:
            output[f"error_{name}_samples"] = 0
        for name, _ in RPI_THROTTLED_BITS:
            output[f"rpi_{name}_samples"] = 0
            output[f"rpi_{name}_percent"] = math.nan
        for prefix, _, unit, _ in HOST_DISTRIBUTIONS:
            output[f"{prefix}_valid_samples"] = 0
            output.update(prefixed_distribution([], prefix, unit))
        output.update(prefixed_distribution([], "sampling_cadence", "s"))
        for prefix, _, unit in HOST_COUNTERS:
            output.update(host_counter_report([], prefix, "", unit))
        output["process_user_cpu_percent"] = math.nan
        output["process_system_cpu_percent"] = math.nan
        output["process_total_cpu_percent"] = math.nan

        if not samples:
            output["quality_status"] = "empty" if path.is_file() else "missing"
            rows_out.append(output)
            continue

        acquisition_statuses: list[str] = []
        sample_statuses: list[str] = []
        monotonic_values: list[int] = []
        cadence_s: list[float] = []
        previous_monotonic_ns: int | None = None
        cadence_invalid = 0
        cadence_missing = 0
        error_mask_or = 0
        rpi_raw_or = 0
        writer_cpu_observed = 0
        writer_migrations = 0
        metric_values: dict[str, list[float]] = {
            prefix: [] for prefix, _, _, _ in HOST_DISTRIBUTIONS
        }

        load_full = 0
        load_partial = 0
        load_error = 0
        load_legacy = 0
        rusage_ok = 0
        rusage_error = 0
        rusage_legacy = 0
        fully_valid = 0
        bounded = 0
        invalid_bounded = 0
        legacy = 0

        for sample in samples:
            acquisition_status = host_acquisition_status(sample)
            acquisition_statuses.append(acquisition_status)
            if acquisition_status == "ok":
                bounded += 1
            elif acquisition_status == "legacy_unbounded":
                legacy += 1
            else:
                invalid_bounded += 1

            sample_status = sample.get("status", "") or "legacy_unverified"
            sample_statuses.append(sample_status)
            error_mask = parse_int(sample.get("error_mask"))
            error_mask_or |= error_mask
            if error_mask == 0 and sample_status == "ok" and acquisition_status == "ok":
                fully_valid += 1
            for name, mask in HOST_ERROR_BITS:
                if error_mask & mask:
                    output[f"error_{name}_samples"] = int(output[f"error_{name}_samples"]) + 1

            monotonic_ns = parse_int(sample.get("monotonic_raw_ns"), -1)
            if monotonic_ns <= 0:
                cadence_missing += 1
                previous_monotonic_ns = None
            else:
                monotonic_values.append(monotonic_ns)
                if previous_monotonic_ns is not None:
                    interval_ns = monotonic_ns - previous_monotonic_ns
                    if interval_ns <= 0:
                        cadence_invalid += 1
                    else:
                        cadence_s.append(interval_ns / 1e9)
                previous_monotonic_ns = monotonic_ns

            writer_cpu = parse_int(sample.get("writer_cpu"), -1)
            writer_cpu_end = parse_int(sample.get("writer_cpu_end"), -1)
            if writer_cpu >= 0 and writer_cpu_end >= 0:
                writer_cpu_observed += 1
                writer_migrations += int(parse_int(sample.get("writer_cpu_migrated")) != 0)

            if "getloadavg_count" not in sample:
                load_legacy += 1
            else:
                load_count = parse_int(sample.get("getloadavg_count"), -1)
                if load_count >= 3:
                    load_full += 1
                elif load_count >= 0:
                    load_partial += 1
                else:
                    load_error += 1

            if "getrusage_status" not in sample:
                rusage_legacy += 1
            elif sample.get("getrusage_status") == "ok":
                rusage_ok += 1
            else:
                rusage_error += 1

            if parse_int(sample.get("rpi_throttled_valid")) != 0:
                output["rpi_throttled_valid_samples"] = int(output["rpi_throttled_valid_samples"]) + 1
                raw = parse_int(sample.get("rpi_throttled_raw"))
                rpi_raw_or |= raw
                for name, mask in RPI_THROTTLED_BITS:
                    if raw & mask:
                        output[f"rpi_{name}_samples"] = int(output[f"rpi_{name}_samples"]) + 1

            for prefix, source, _, gate in HOST_DISTRIBUTIONS:
                value = host_metric_value(sample, source, gate)
                if value is not None:
                    metric_values[prefix].append(value)

        output.update(
            {
                "monotonic_valid_samples": len(monotonic_values),
                "fully_valid_samples": fully_valid,
                "bounded_acquisition_samples": bounded,
                "invalid_bounded_acquisition_samples": invalid_bounded,
                "legacy_acquisition_unbounded_samples": legacy,
                "acquisition_statuses": joined_status(acquisition_statuses, "unknown"),
                "sample_statuses": joined_status(sample_statuses, "unknown"),
                "error_mask_nonzero_samples": sum(
                    int(parse_int(sample.get("error_mask")) != 0) for sample in samples
                ),
                "error_mask_or": error_mask_or,
                "error_mask_unknown_bits_or": error_mask_or & ~known_error_mask,
                "first_monotonic_raw_ns": monotonic_values[0] if monotonic_values else "",
                "last_monotonic_raw_ns": monotonic_values[-1] if monotonic_values else "",
                "observation_span_s": (
                    (monotonic_values[-1] - monotonic_values[0]) / 1e9
                    if monotonic_values and cadence_invalid == 0
                    else math.nan
                ),
                "sampling_cadence_valid_intervals": len(cadence_s),
                "sampling_cadence_invalid_intervals": cadence_invalid,
                "sampling_cadence_missing_samples": cadence_missing,
                "writer_cpu_observed_samples": writer_cpu_observed,
                "writer_cpu_migration_samples": writer_migrations,
                "writer_cpu_migration_percent": (
                    100.0 * writer_migrations / writer_cpu_observed
                    if writer_cpu_observed
                    else math.nan
                ),
                "getloadavg_full_samples": load_full,
                "getloadavg_partial_samples": load_partial,
                "getloadavg_error_samples": load_error,
                "getloadavg_unverified_legacy_samples": load_legacy,
                "getrusage_ok_samples": rusage_ok,
                "getrusage_error_samples": rusage_error,
                "getrusage_unverified_legacy_samples": rusage_legacy,
                "rpi_throttled_raw_or": rpi_raw_or,
                "rpi_throttled_unknown_bits_or": rpi_raw_or & ~known_rpi_mask,
            }
        )
        output.update(prefixed_distribution(cadence_s, "sampling_cadence", "s"))
        for prefix, _, unit, _ in HOST_DISTRIBUTIONS:
            output[f"{prefix}_valid_samples"] = len(metric_values[prefix])
            output.update(prefixed_distribution(metric_values[prefix], prefix, unit))
        for prefix, source, unit in HOST_COUNTERS:
            output.update(host_counter_report(samples, prefix, source, unit))

        rpi_valid = int(output["rpi_throttled_valid_samples"])
        for name, _ in RPI_THROTTLED_BITS:
            count = int(output[f"rpi_{name}_samples"])
            output[f"rpi_{name}_percent"] = 100.0 * count / rpi_valid if rpi_valid else math.nan

        user_rate = parse_float(output.get("process_user_rate_per_second"))
        system_rate = parse_float(output.get("process_system_rate_per_second"))
        output["process_user_cpu_percent"] = (
            user_rate / 10000.0 if math.isfinite(user_rate) else math.nan
        )
        output["process_system_cpu_percent"] = (
            system_rate / 10000.0 if math.isfinite(system_rate) else math.nan
        )
        output["process_total_cpu_percent"] = (
            (user_rate + system_rate) / 10000.0
            if math.isfinite(user_rate) and math.isfinite(system_rate)
            else math.nan
        )

        if not monotonic_values:
            quality_status = "no_valid_monotonic_samples"
        elif cadence_invalid:
            quality_status = "monotonic_regression"
        elif invalid_bounded:
            quality_status = "acquisition_invalid"
        elif error_mask_or & 1:
            quality_status = "realtime_clock_error"
        elif error_mask_or & ((1 << 5) | (1 << 6)) or load_partial or load_error or rusage_error:
            quality_status = "partial_probe_error"
        elif legacy and bounded:
            quality_status = "mixed_legacy_and_bounded"
        elif legacy == len(samples):
            quality_status = "legacy_acquisition_unbounded"
        elif bounded == len(samples):
            quality_status = "ok"
        else:
            quality_status = "unclassified"
        output["quality_status"] = quality_status
        rows_out.append(output)

    return Report(fields, rows_out)


def archive_integrity_report(
    run_dirs: list[Path],
    metadata_by_dir: dict[str, dict[str, str]],
    campaign_by_dir: dict[str, dict[str, Any]],
) -> Report:
    fields = IDENTITY_FIELDS + [
        "manifest_present",
        "relative_path",
        "status",
        "valid",
        "expected_size_bytes",
        "observed_size_bytes",
        "expected_mtime_ns",
        "observed_mtime_ns",
        "expected_sha256",
        "observed_sha256",
        "notes",
    ]
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        identity = identity_for(run_dir, metadata_by_dir, campaign_by_dir)
        manifest_present = (run_dir / MANIFEST_NAME).is_file()
        if not manifest_present:
            rows.append(
                {
                    **identity,
                    "manifest_present": 0,
                    "relative_path": "",
                    "status": "manifest_missing",
                    "valid": 0,
                    "expected_size_bytes": 0,
                    "observed_size_bytes": 0,
                    "expected_mtime_ns": 0,
                    "observed_mtime_ns": 0,
                    "expected_sha256": "",
                    "observed_sha256": "",
                    "notes": "archive consistency cannot be verified",
                }
            )
            continue
        try:
            results = verify_archive(run_dir)
        except Exception as error:
            rows.append(
                {
                    **identity,
                    "manifest_present": 1,
                    "relative_path": "",
                    "status": "verification_error",
                    "valid": 0,
                    "expected_size_bytes": 0,
                    "observed_size_bytes": 0,
                    "expected_mtime_ns": 0,
                    "observed_mtime_ns": 0,
                    "expected_sha256": "",
                    "observed_sha256": "",
                    "notes": str(error),
                }
            )
            continue
        for result in results:
            rows.append(
                {
                    **identity,
                    "manifest_present": 1,
                    "relative_path": result.relative_path,
                    "status": result.status,
                    "valid": int(result.valid),
                    "expected_size_bytes": result.expected_size_bytes,
                    "observed_size_bytes": result.observed_size_bytes,
                    "expected_mtime_ns": result.expected_mtime_ns,
                    "observed_mtime_ns": result.observed_mtime_ns,
                    "expected_sha256": result.expected_sha256,
                    "observed_sha256": result.observed_sha256,
                    "notes": result.notes,
                }
            )
    return Report(fields, rows)


def primitive_overhead_report(
    profile_dirs: list[Path],
    metadata_by_dir: dict[str, dict[str, str]],
    campaign_by_dir: dict[str, dict[str, Any]],
) -> Report:
    fields = IDENTITY_FIELDS + [
        "phase",
        "primitive",
        "event_kind",
        "samples_total",
        "samples_valid",
        "samples_invalid",
        "event_expected_count",
        "event_recorded_count",
        "drop_total",
        "cpu_observed_count",
        "cpu_migrations",
        "status",
        "baseline_primitive",
        "baseline_p50_us",
        "median_excess_over_counter_pair_us",
        "excess_estimate_semantics",
        "duration_min_us",
        "duration_p50_us",
        "duration_p90_us",
        "duration_p95_us",
        "duration_p99_us",
        "duration_p99_9_us",
        "duration_max_us",
        "duration_mean_us",
        "duration_stdev_us",
    ]
    grouped: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for profile_dir in profile_dirs:
        for row in read_csv(profile_dir / "profiler_primitive_overhead.csv"):
            key = (str(profile_dir), row.get("phase", ""), row.get("primitive", ""), row.get("event_kind", ""))
            group = grouped.setdefault(
                key,
                {
                    "durations": [],
                    "total": 0,
                    "valid": 0,
                    "expected": 0,
                    "recorded": 0,
                    "drops": 0,
                    "cpu_observed": 0,
                    "migrations": 0,
                    "statuses": [],
                },
            )
            group["total"] = int(group["total"]) + 1
            status = row.get("status", "")
            group["statuses"].append(status)
            duration = parse_float(row.get("outer_duration_us"))
            if status == "ok" and math.isfinite(duration) and duration >= 0:
                group["durations"].append(duration)
                group["valid"] = int(group["valid"]) + 1
            group["expected"] = int(group["expected"]) + parse_int(row.get("event_record_expected"))
            group["recorded"] = int(group["recorded"]) + parse_int(row.get("event_recorded"))
            group["drops"] = int(group["drops"]) + parse_int(row.get("drop_delta"))
            cpu_start = parse_int(row.get("cpu_start"), -1)
            cpu_end = parse_int(row.get("cpu_end"), -1)
            group["cpu_observed"] = int(group["cpu_observed"]) + int(cpu_start >= 0 and cpu_end >= 0)
            group["migrations"] = int(group["migrations"]) + parse_int(row.get("cpu_migrated"))

    baseline: dict[tuple[str, str], float] = {}
    for (profile_dir, phase, primitive, _), group in grouped.items():
        if primitive == "counter_pair":
            baseline[(profile_dir, phase)] = distribution(group["durations"])["p50"]

    rows: list[dict[str, object]] = []
    for (profile_key, phase, primitive, event_kind), group in sorted(grouped.items()):
        profile_dir = Path(profile_key)
        stats = prefixed_distribution(group["durations"], "duration", "us")
        baseline_p50 = baseline.get((profile_key, phase), math.nan)
        excess = stats["duration_p50_us"] - baseline_p50 if math.isfinite(baseline_p50) else math.nan
        rows.append(
            {
                **identity_for(profile_dir, metadata_by_dir, campaign_by_dir),
                "phase": phase,
                "primitive": primitive,
                "event_kind": event_kind,
                "samples_total": group["total"],
                "samples_valid": group["valid"],
                "samples_invalid": int(group["total"]) - int(group["valid"]),
                "event_expected_count": group["expected"],
                "event_recorded_count": group["recorded"],
                "drop_total": group["drops"],
                "cpu_observed_count": group["cpu_observed"],
                "cpu_migrations": group["migrations"],
                "status": joined_status(group["statuses"], "unknown"),
                "baseline_primitive": "counter_pair",
                "baseline_p50_us": baseline_p50,
                "median_excess_over_counter_pair_us": excess,
                "excess_estimate_semantics": "difference_of_phase_medians_not_per_sample_correction",
                **stats,
            }
        )
    return Report(fields, rows)


def pmu_reports(
    profile_dirs: list[Path],
    metadata_by_dir: dict[str, dict[str, str]],
    settings_by_dir: dict[str, dict[str, str]],
    campaign_by_dir: dict[str, dict[str, Any]],
) -> tuple[Report, Report, Report]:
    availability_fields = IDENTITY_FIELDS + [
        "pmu_mode",
        "stream_status",
        "thread_index",
        "tid",
        "thread_name",
        "event_id",
        "event_name",
        "domain",
        "requested",
        "available",
        "status",
        "error_code",
    ]
    summary_fields = IDENTITY_FIELDS + [
        "thread_index",
        "tid",
        "thread_name",
        "event_name",
        "domain",
        "unit",
        "samples_total",
        "usable_samples",
        "delta_scaled_total",
        "interval_total_s",
        "estimated_rate_per_second",
        "sample_rate_p50_per_second",
        "sample_rate_p99_per_second",
        "multiplex_ratio_p50",
        "multiplex_ratio_min",
        "status",
        "attribution_scope",
    ]
    quality_fields = IDENTITY_FIELDS + [
        "thread_index",
        "tid",
        "thread_name",
        "event_name",
        "samples_total",
        "delta_valid_count",
        "scaling_valid_count",
        "usable_count",
        "invalid_count",
        "read_error_count",
        "multiplex_ratio_min",
        "multiplex_ratio_p10",
        "multiplex_ratio_p50",
        "status",
        "quality_note",
    ]
    availability_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, str, str, str, str, str], dict[str, object]] = {}
    profiles_with_samples: set[str] = set()
    for profile_dir in profile_dirs:
        identity = identity_for(profile_dir, metadata_by_dir, campaign_by_dir)
        pmu_mode = settings_by_dir.get(str(profile_dir), {}).get(
            "profile.pmu_mode", metadata_by_dir.get(str(profile_dir), {}).get("pmu_mode", "unknown")
        )
        availability_path = profile_dir / "pmu_availability.csv"
        availability = read_csv(availability_path)
        if not availability:
            availability_rows.append(
                {
                    **identity,
                    "pmu_mode": pmu_mode,
                    "stream_status": "empty" if availability_path.is_file() else "missing",
                    "thread_index": "",
                    "tid": "",
                    "thread_name": "",
                    "event_id": "",
                    "event_name": "",
                    "domain": "",
                    "requested": 0,
                    "available": 0,
                    "status": "no_availability_rows",
                    "error_code": "",
                }
            )
        for row in availability:
            availability_rows.append(
                {
                    **identity,
                    "pmu_mode": pmu_mode,
                    "stream_status": "recorded",
                    **{
                        field: row.get(field, "")
                        for field in availability_fields
                        if field not in IDENTITY_FIELDS + ["pmu_mode", "stream_status"]
                    },
                }
            )

        for row in read_csv(profile_dir / "pmu_samples.csv"):
            profiles_with_samples.add(str(profile_dir))
            key = (
                str(profile_dir),
                row.get("thread_index", ""),
                row.get("tid", ""),
                row.get("thread_name", ""),
                row.get("event_name", ""),
                row.get("domain", ""),
                row.get("unit", ""),
            )
            group = grouped.setdefault(
                key,
                {
                    "total": 0,
                    "delta_valid": 0,
                    "scaling_valid": 0,
                    "usable": 0,
                    "errors": 0,
                    "delta_total": 0.0,
                    "interval_total_ns": 0,
                    "rates": [],
                    "multiplex": [],
                    "statuses": [],
                },
            )
            group["total"] = int(group["total"]) + 1
            delta_valid = parse_int(row.get("delta_valid")) != 0
            scaling_valid = parse_int(row.get("scaling_valid")) != 0
            interval_ns = parse_int(row.get("interval_ns"))
            delta_scaled = parse_float(row.get("delta_scaled"))
            multiplex = parse_float(row.get("multiplex_ratio"))
            group["delta_valid"] = int(group["delta_valid"]) + int(delta_valid)
            group["scaling_valid"] = int(group["scaling_valid"]) + int(scaling_valid)
            group["errors"] = int(group["errors"]) + int(parse_int(row.get("error_code")) != 0)
            group["statuses"].append(row.get("status", ""))
            if delta_valid and scaling_valid and math.isfinite(multiplex) and multiplex >= 0:
                group["multiplex"].append(multiplex)
            if delta_valid and scaling_valid and interval_ns > 0 and math.isfinite(delta_scaled):
                group["usable"] = int(group["usable"]) + 1
                group["delta_total"] = float(group["delta_total"]) + delta_scaled
                group["interval_total_ns"] = int(group["interval_total_ns"]) + interval_ns
                group["rates"].append(delta_scaled * 1e9 / interval_ns)

    summary_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        profile_key, thread_index, tid, thread_name, event_name, domain, unit = key
        identity = identity_for(Path(profile_key), metadata_by_dir, campaign_by_dir)
        rate_stats = distribution(group["rates"])
        multiplex_stats = distribution(group["multiplex"])
        interval_s = int(group["interval_total_ns"]) / 1e9
        aggregate_rate = float(group["delta_total"]) / interval_s if interval_s > 0 else math.nan
        summary_rows.append(
            {
                **identity,
                "thread_index": thread_index,
                "tid": tid,
                "thread_name": thread_name,
                "event_name": event_name,
                "domain": domain,
                "unit": unit,
                "samples_total": group["total"],
                "usable_samples": group["usable"],
                "delta_scaled_total": group["delta_total"],
                "interval_total_s": interval_s,
                "estimated_rate_per_second": aggregate_rate,
                "sample_rate_p50_per_second": rate_stats["p50"],
                "sample_rate_p99_per_second": rate_stats["p99"],
                "multiplex_ratio_p50": multiplex_stats["p50"],
                "multiplex_ratio_min": multiplex_stats["min"],
                "status": joined_status(group["statuses"], "unknown"),
                "attribution_scope": "per_thread_sampling_interval_not_function_exact",
            }
        )
        quality_rows.append(
            {
                **identity,
                "thread_index": thread_index,
                "tid": tid,
                "thread_name": thread_name,
                "event_name": event_name,
                "samples_total": group["total"],
                "delta_valid_count": group["delta_valid"],
                "scaling_valid_count": group["scaling_valid"],
                "usable_count": group["usable"],
                "invalid_count": int(group["total"]) - int(group["usable"]),
                "read_error_count": group["errors"],
                "multiplex_ratio_min": multiplex_stats["min"],
                "multiplex_ratio_p10": quantile(sorted(group["multiplex"]), 0.1),
                "multiplex_ratio_p50": multiplex_stats["p50"],
                "status": joined_status(group["statuses"], "unknown"),
                "quality_note": "rates and multiplex ratios require delta_valid=1 and scaling_valid=1",
            }
        )
    for profile_dir in profile_dirs:
        if str(profile_dir) not in profiles_with_samples:
            quality_rows.append(
                {
                    **identity_for(profile_dir, metadata_by_dir, campaign_by_dir),
                    "thread_index": "",
                    "tid": "",
                    "thread_name": "",
                    "event_name": "",
                    "samples_total": 0,
                    "delta_valid_count": 0,
                    "scaling_valid_count": 0,
                    "usable_count": 0,
                    "invalid_count": 0,
                    "read_error_count": 0,
                    "multiplex_ratio_min": math.nan,
                    "multiplex_ratio_p10": math.nan,
                    "multiplex_ratio_p50": math.nan,
                    "status": "stream_empty_or_missing",
                    "quality_note": "no PMU sample rows; consult availability and pmu_mode",
                }
            )
    return (
        Report(availability_fields, availability_rows),
        Report(summary_fields, summary_rows),
        Report(quality_fields, quality_rows),
    )


def scheduler_report(
    profile_dirs: list[Path],
    metadata_by_dir: dict[str, dict[str, str]],
    campaign_by_dir: dict[str, dict[str, Any]],
) -> Report:
    fields = IDENTITY_FIELDS + [
        "thread_index",
        "tid",
        "thread_name",
        "samples_total",
        "delta_valid_count",
        "interval_total_s",
        "runtime_total_s",
        "runqueue_wait_total_s",
        "cpu_runtime_percent",
        "runqueue_wait_percent_of_elapsed",
        "scheduler_wait_fraction",
        "timeslices_total",
        "timeslices_per_second",
        "minor_faults_total",
        "major_faults_total",
        "voluntary_context_switches_total",
        "involuntary_context_switches_total",
        "context_switches_per_second",
        "cpu_change_samples",
        "cpu_frequency_p50_khz",
        "cpu_frequency_min_khz",
        "cpu_frequency_max_khz",
        "status",
        "error_count",
    ]
    grouped: dict[tuple[str, str, str, str], dict[str, object]] = {}
    profiles_with_rows: set[str] = set()
    for profile_dir in profile_dirs:
        for row in read_csv(profile_dir / "thread_metrics.csv"):
            profiles_with_rows.add(str(profile_dir))
            key = (str(profile_dir), row.get("thread_index", ""), row.get("tid", ""), row.get("thread_name", ""))
            group = grouped.setdefault(
                key,
                {
                    "total": 0,
                    "valid": 0,
                    "interval": 0,
                    "runtime": 0,
                    "wait": 0,
                    "timeslices": 0,
                    "minor": 0,
                    "major": 0,
                    "voluntary": 0,
                    "involuntary": 0,
                    "cpu_changes": 0,
                    "frequency": [],
                    "statuses": [],
                    "errors": 0,
                },
            )
            group["total"] = int(group["total"]) + 1
            group["statuses"].append(row.get("status", ""))
            group["errors"] = int(group["errors"]) + int(parse_int(row.get("error_code")) != 0)
            group["cpu_changes"] = int(group["cpu_changes"]) + parse_int(row.get("cpu_changed_since_previous"))
            frequency = parse_float(row.get("cpu_frequency_khz"))
            if (
                parse_int(row.get("valid_mask")) & THREAD_METRIC_CPU_FREQUENCY_VALID
                and math.isfinite(frequency)
                and frequency >= 0
            ):
                group["frequency"].append(frequency)
            if parse_int(row.get("delta_valid")) == 0:
                continue
            interval = parse_int(row.get("interval_ns"))
            if interval <= 0:
                continue
            group["valid"] = int(group["valid"]) + 1
            group["interval"] = int(group["interval"]) + interval
            for target, source in (
                ("runtime", "delta_runtime_ns"),
                ("wait", "delta_runqueue_wait_ns"),
                ("timeslices", "delta_timeslices"),
                ("minor", "delta_minor_faults"),
                ("major", "delta_major_faults"),
                ("voluntary", "delta_voluntary_context_switches"),
                ("involuntary", "delta_involuntary_context_switches"),
            ):
                group[target] = int(group[target]) + parse_int(row.get(source))

    rows: list[dict[str, object]] = []
    for (profile_key, thread_index, tid, thread_name), group in sorted(grouped.items()):
        interval_ns = int(group["interval"])
        runtime_ns = int(group["runtime"])
        wait_ns = int(group["wait"])
        interval_s = interval_ns / 1e9
        frequency = distribution(group["frequency"])
        rows.append(
            {
                **identity_for(Path(profile_key), metadata_by_dir, campaign_by_dir),
                "thread_index": thread_index,
                "tid": tid,
                "thread_name": thread_name,
                "samples_total": group["total"],
                "delta_valid_count": group["valid"],
                "interval_total_s": interval_s,
                "runtime_total_s": runtime_ns / 1e9,
                "runqueue_wait_total_s": wait_ns / 1e9,
                "cpu_runtime_percent": 100.0 * runtime_ns / interval_ns if interval_ns else math.nan,
                "runqueue_wait_percent_of_elapsed": 100.0 * wait_ns / interval_ns if interval_ns else math.nan,
                "scheduler_wait_fraction": wait_ns / (runtime_ns + wait_ns) if runtime_ns + wait_ns else math.nan,
                "timeslices_total": group["timeslices"],
                "timeslices_per_second": int(group["timeslices"]) / interval_s if interval_s else math.nan,
                "minor_faults_total": group["minor"],
                "major_faults_total": group["major"],
                "voluntary_context_switches_total": group["voluntary"],
                "involuntary_context_switches_total": group["involuntary"],
                "context_switches_per_second": (int(group["voluntary"]) + int(group["involuntary"])) / interval_s
                if interval_s
                else math.nan,
                "cpu_change_samples": group["cpu_changes"],
                "cpu_frequency_p50_khz": frequency["p50"],
                "cpu_frequency_min_khz": frequency["min"],
                "cpu_frequency_max_khz": frequency["max"],
                "status": joined_status(group["statuses"], "unknown"),
                "error_count": group["errors"],
            }
        )
    for profile_dir in profile_dirs:
        if str(profile_dir) not in profiles_with_rows:
            rows.append(
                {
                    **identity_for(profile_dir, metadata_by_dir, campaign_by_dir),
                    **{field: "" for field in fields if field not in IDENTITY_FIELDS},
                    "samples_total": 0,
                    "delta_valid_count": 0,
                    "status": "stream_empty_or_missing",
                    "error_count": 0,
                }
            )
    return Report(fields, rows)


def kernel_interference_report(
    profile_dirs: list[Path],
    metadata_by_dir: dict[str, dict[str, str]],
    campaign_by_dir: dict[str, dict[str, Any]],
) -> Report:
    fields = IDENTITY_FIELDS + [
        "stream",
        "source",
        "label",
        "description",
        "cpu",
        "radio_relevant",
        "metric_kind",
        "samples_total",
        "valid_delta_count",
        "invalid_delta_count",
        "delta_total",
        "interval_total_s",
        "rate_per_second",
        "raw_p50",
        "raw_p99",
        "status",
    ]
    grouped: dict[tuple[str, str, str, str, str, str, str, str], dict[str, object]] = {}
    profiles_with_rows: set[str] = set()
    for profile_dir in profile_dirs:
        for row in read_csv(profile_dir / "kernel_activity.csv"):
            profiles_with_rows.add(str(profile_dir))
            cumulative = parse_int(row.get("cumulative")) != 0
            key = (
                str(profile_dir),
                "kernel_activity",
                "proc_stat",
                row.get("metric", ""),
                "",
                "",
                "1",
                "cumulative" if cumulative else "gauge",
            )
            group = grouped.setdefault(
                key,
                {"total": 0, "valid": 0, "delta": 0, "interval": 0, "raw": [], "statuses": []},
            )
            group["total"] = int(group["total"]) + 1
            group["raw"].append(parse_float(row.get("raw_value")))
            group["statuses"].append(row.get("status", ""))
            if cumulative and parse_int(row.get("delta_valid")) and parse_int(row.get("interval_ns")) > 0:
                group["valid"] = int(group["valid"]) + 1
                group["delta"] = int(group["delta"]) + parse_int(row.get("delta_value"))
                group["interval"] = int(group["interval"]) + parse_int(row.get("interval_ns"))
        for filename in ("interrupts.csv", "softirqs.csv"):
            for row in read_csv(profile_dir / filename):
                profiles_with_rows.add(str(profile_dir))
                key = (
                    str(profile_dir),
                    filename.removesuffix(".csv"),
                    row.get("source", ""),
                    row.get("label", ""),
                    row.get("description", ""),
                    row.get("cpu", ""),
                    row.get("radio_relevant", ""),
                    "cumulative",
                )
                group = grouped.setdefault(
                    key,
                    {"total": 0, "valid": 0, "delta": 0, "interval": 0, "raw": [], "statuses": []},
                )
                group["total"] = int(group["total"]) + 1
                group["raw"].append(parse_float(row.get("raw_count")))
                group["statuses"].append(row.get("status", ""))
                if parse_int(row.get("delta_valid")) and parse_int(row.get("interval_ns")) > 0:
                    group["valid"] = int(group["valid"]) + 1
                    group["delta"] = int(group["delta"]) + parse_int(row.get("delta_count"))
                    group["interval"] = int(group["interval"]) + parse_int(row.get("interval_ns"))

    rows: list[dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        profile_key, stream, source, label, description, cpu, radio_relevant, metric_kind = key
        interval_s = int(group["interval"]) / 1e9
        raw = distribution(group["raw"])
        rows.append(
            {
                **identity_for(Path(profile_key), metadata_by_dir, campaign_by_dir),
                "stream": stream,
                "source": source,
                "label": label,
                "description": description,
                "cpu": cpu,
                "radio_relevant": radio_relevant,
                "metric_kind": metric_kind,
                "samples_total": group["total"],
                "valid_delta_count": group["valid"],
                "invalid_delta_count": int(group["total"]) - int(group["valid"]) if metric_kind == "cumulative" else 0,
                "delta_total": group["delta"],
                "interval_total_s": interval_s,
                "rate_per_second": int(group["delta"]) / interval_s if interval_s else math.nan,
                "raw_p50": raw["p50"],
                "raw_p99": raw["p99"],
                "status": joined_status(group["statuses"], "unknown"),
            }
        )
    for profile_dir in profile_dirs:
        if str(profile_dir) not in profiles_with_rows:
            rows.append(
                {
                    **identity_for(profile_dir, metadata_by_dir, campaign_by_dir),
                    **{field: "" for field in fields if field not in IDENTITY_FIELDS},
                    "samples_total": 0,
                    "valid_delta_count": 0,
                    "invalid_delta_count": 0,
                    "status": "streams_empty_or_missing",
                }
            )
    return Report(fields, rows)


def collection_overhead_report(
    profile_dirs: list[Path],
    metadata_by_dir: dict[str, dict[str, str]],
    campaign_by_dir: dict[str, dict[str, Any]],
) -> Report:
    fields = IDENTITY_FIELDS + [
        "source",
        "thread_index",
        "tid",
        "thread_name",
        "samples_total",
        "rows_or_observations_total",
        "read_errors_total",
        "status",
        "duration_min_us",
        "duration_p50_us",
        "duration_p90_us",
        "duration_p95_us",
        "duration_p99_us",
        "duration_p99_9_us",
        "duration_max_us",
        "duration_mean_us",
        "duration_stdev_us",
    ]
    grouped: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    profiles_with_rows: set[str] = set()
    for profile_dir in profile_dirs:
        for row in read_csv(profile_dir / "pmu_read_overhead.csv"):
            profiles_with_rows.add(str(profile_dir))
            key = (
                str(profile_dir),
                "pmu_read",
                row.get("thread_index", ""),
                row.get("tid", ""),
                row.get("thread_name", ""),
            )
            group = grouped.setdefault(key, {"durations": [], "count": 0, "work": 0, "errors": 0, "statuses": []})
            group["count"] = int(group["count"]) + 1
            duration = parse_float(row.get("duration_us"))
            read_errors = parse_int(row.get("read_errors"))
            counter_status = row.get("counter_status", "ok") or "ok"
            if counter_status == "ok" and math.isfinite(duration) and duration >= 0:
                group["durations"].append(duration)
            group["work"] = int(group["work"]) + parse_int(row.get("observations"))
            group["errors"] = int(group["errors"]) + read_errors + int(counter_status != "ok")
            read_status = "ok" if read_errors == 0 else "read_error"
            group["statuses"].append(counter_status if counter_status != "ok" else read_status)
        for row in read_csv(profile_dir / "system_read_overhead.csv"):
            profiles_with_rows.add(str(profile_dir))
            key = (str(profile_dir), row.get("source", "system"), "", "", "writer")
            group = grouped.setdefault(key, {"durations": [], "count": 0, "work": 0, "errors": 0, "statuses": []})
            group["count"] = int(group["count"]) + 1
            group["durations"].append(parse_float(row.get("duration_us")))
            group["work"] = int(group["work"]) + parse_int(row.get("rows"))
            group["errors"] = int(group["errors"]) + int(parse_int(row.get("error_code")) != 0)
            group["statuses"].append(row.get("status", ""))
        for row in read_csv(profile_dir / "profiler_primitive_overhead.csv"):
            profiles_with_rows.add(str(profile_dir))
            source = f"profiler_primitive:{row.get('phase', '')}:{row.get('primitive', '')}"
            key = (str(profile_dir), source, "", "", "initializing_thread")
            group = grouped.setdefault(key, {"durations": [], "count": 0, "work": 0, "errors": 0, "statuses": []})
            group["count"] = int(group["count"]) + 1
            duration = parse_float(row.get("outer_duration_us"))
            if row.get("status") == "ok" and math.isfinite(duration) and duration >= 0:
                group["durations"].append(duration)
            group["work"] = int(group["work"]) + parse_int(row.get("event_recorded"))
            group["errors"] = int(group["errors"]) + int(row.get("status", "") != "ok")
            group["statuses"].append(row.get("status", ""))

    rows: list[dict[str, object]] = []
    for (profile_key, source, thread_index, tid, thread_name), group in sorted(grouped.items()):
        rows.append(
            {
                **identity_for(Path(profile_key), metadata_by_dir, campaign_by_dir),
                "source": source,
                "thread_index": thread_index,
                "tid": tid,
                "thread_name": thread_name,
                "samples_total": group["count"],
                "rows_or_observations_total": group["work"],
                "read_errors_total": group["errors"],
                "status": joined_status(group["statuses"], "unknown"),
                **prefixed_distribution(group["durations"], "duration", "us"),
            }
        )
    for profile_dir in profile_dirs:
        if str(profile_dir) not in profiles_with_rows:
            rows.append(
                {
                    **identity_for(profile_dir, metadata_by_dir, campaign_by_dir),
                    **{field: "" for field in fields if field not in IDENTITY_FIELDS},
                    "source": "all",
                    "samples_total": 0,
                    "rows_or_observations_total": 0,
                    "read_errors_total": 0,
                    "status": "streams_empty_or_missing",
                }
            )
    return Report(fields, rows)


def transport_reports(
    summary_rows: list[dict[str, object]],
    fault_rows: list[dict[str, str]],
    metadata_by_dir: dict[str, dict[str, str]],
    campaign_by_dir: dict[str, dict[str, Any]],
) -> tuple[Report, Report]:
    summary_fields = IDENTITY_FIELDS + [
        "event_name",
        "event_class",
        "event_kind",
        "detail_level",
        "count",
        "total_duration_us",
        "min_us",
        "p50_us",
        "p90_us",
        "p95_us",
        "p99_us",
        "p99_9_us",
        "max_us",
        "mean_us",
        "stdev_us",
        "cpu_migrations",
        "drops_total",
    ]
    fault_fields = IDENTITY_FIELDS + [
        "schema_version",
        "seq",
        "tid",
        "thread_name",
        "event_name",
        "event_class",
        "frame",
        "slot",
        "absolute_slot",
        "correlation_id",
        "parent_id",
        "cpu_start",
        "cpu_end",
        "cpu_migrated",
        "flags",
        "aux0_name",
        "aux0_unit",
        "aux0",
        "aux1_name",
        "aux1_unit",
        "aux1",
        "aux2_name",
        "aux2_unit",
        "aux2",
        "aux3_name",
        "aux3_unit",
        "aux3",
        "start_tick",
    ]
    transport_names = {
        "UE_RF_READ",
        "UE_RF_READ_DRIFT",
        "UE_RF_WRITE",
        "GNB_RF_READ",
        "GNB_RF_READ_ALIGN",
        "GNB_RF_WRITE",
    }
    rows: list[dict[str, object]] = []
    for summary in summary_rows:
        event_name = str(summary.get("event_name", ""))
        if (
            summary.get("subsystem") != "rf_usrp"
            and not event_name.startswith("USRP_")
            and event_name not in transport_names
        ):
            continue
        profile_key = str(summary["profile_dir"])
        count = parse_int(summary.get("count"))
        mean_us = parse_float(summary.get("mean_us"))
        rows.append(
            {
                **identity_for(Path(profile_key), metadata_by_dir, campaign_by_dir),
                **{
                    field: summary.get(field, "")
                    for field in summary_fields
                    if field not in IDENTITY_FIELDS + ["total_duration_us"]
                },
                "total_duration_us": mean_us * count if math.isfinite(mean_us) else math.nan,
            }
        )
    normalized_faults: list[dict[str, object]] = []
    for fault in fault_rows:
        profile_key = fault["profile_dir"]
        normalized_faults.append(
            {
                **identity_for(Path(profile_key), metadata_by_dir, campaign_by_dir),
                **{field: fault.get(field, "") for field in fault_fields if field not in IDENTITY_FIELDS},
            }
        )
    return Report(summary_fields, rows), Report(fault_fields, normalized_faults)


def source_alignment_status(row: dict[str, str]) -> str:
    method = row.get("alignment_method", "")
    if method in {"", "unknown", "unresolved", "alignment_pending"}:
        return "unresolved"
    if row.get("status", "") not in {"recorded", "ok", "complete"}:
        return "source_not_recorded"
    start_monotonic = parse_int(row.get("start_monotonic_raw_ns"))
    end_monotonic = parse_int(row.get("end_monotonic_raw_ns"))
    start_realtime = parse_int(row.get("start_realtime_ns"))
    end_realtime = parse_int(row.get("end_realtime_ns"))
    monotonic_present = start_monotonic > 0 or end_monotonic > 0
    realtime_present = start_realtime > 0 or end_realtime > 0
    monotonic_valid = start_monotonic > 0 and end_monotonic >= start_monotonic
    realtime_valid = start_realtime > 0 and end_realtime >= start_realtime
    uncertainty_text = row.get("alignment_uncertainty_ns", "")
    if uncertainty_text and parse_int(uncertainty_text, -1) < 0:
        return "invalid_alignment_uncertainty"
    if method == "aggregate_run_interval":
        if monotonic_valid or realtime_valid:
            return "aggregate_only"
        if monotonic_present or realtime_present:
            return "aggregate_only_invalid_anchors"
        return "aggregate_only_anchors_incomplete"
    if method == "shared_monotonic_raw":
        if monotonic_valid:
            return "aligned_with_declared_method"
        return "required_monotonic_anchors_invalid" if monotonic_present else "required_monotonic_anchors_incomplete"
    if method == "shared_realtime":
        if realtime_valid:
            return "aligned_with_declared_method"
        return "required_realtime_anchors_invalid" if realtime_present else "required_realtime_anchors_incomplete"
    return "unrecognized_alignment_method"


def registered_artifact(run_dir: Path, artifact_text: str) -> tuple[Path | None, str]:
    if not is_safe_relative_path(artifact_text):
        return None, "unsafe_path"
    if has_symlink_component(run_dir, artifact_text):
        return None, "unsafe_symlink"
    artifact = run_dir.joinpath(*PurePosixPath(artifact_text).parts)
    try:
        resolved = artifact.resolve(strict=True)
        resolved.relative_to(run_dir.resolve())
    except (FileNotFoundError, NotADirectoryError):
        return None, "missing_or_nonregular"
    except ValueError:
        return None, "path_escape"
    if not resolved.is_file():
        return None, "missing_or_nonregular"
    return artifact, "present"


def external_sources_report(
    run_dirs: list[Path],
    metadata_by_dir: dict[str, dict[str, str]],
    campaign_by_dir: dict[str, dict[str, Any]],
) -> Report:
    source_fields = [
        "source_id",
        "source_type",
        "clock_domain",
        "clock_unit",
        "artifact_path",
        "command",
        "tool_version",
        "start_realtime_ns",
        "end_realtime_ns",
        "start_monotonic_raw_ns",
        "end_monotonic_raw_ns",
        "status",
        "alignment_method",
        "alignment_uncertainty_ns",
        "notes",
    ]
    fields = IDENTITY_FIELDS + [
        "catalog_status",
        *source_fields,
        "artifact_status",
        "artifact_size_bytes",
        "alignment_status",
    ]
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        catalog = run_dir / "external_sources.csv"
        sources = read_csv(catalog)
        if not sources:
            rows.append(
                {
                    **identity_for(run_dir, metadata_by_dir, campaign_by_dir),
                    "catalog_status": "empty" if catalog.is_file() else "missing",
                    **{field: "" for field in source_fields},
                    "artifact_status": "not_registered",
                    "artifact_size_bytes": "",
                    "alignment_status": "not_applicable",
                }
            )
            continue
        for source in sources:
            artifact_text = source.get("artifact_path", "")
            artifact, artifact_status = registered_artifact(run_dir, artifact_text)
            artifact_size: object = artifact.stat().st_size if artifact is not None else ""
            rows.append(
                {
                    **identity_for(run_dir, metadata_by_dir, campaign_by_dir),
                    "catalog_status": "recorded",
                    **{field: source.get(field, "") for field in source_fields},
                    "artifact_status": artifact_status,
                    "artifact_size_bytes": artifact_size,
                    "alignment_status": source_alignment_status(source),
                }
            )
    return Report(fields, rows)


def parse_perf_stat_line(line: str) -> dict[str, object] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    parts = [part.strip() for part in stripped.split(";")]
    while parts and not parts[-1]:
        parts.pop()
    if len(parts) < 3:
        return {
            "status": "unparsed",
            "event": "",
            "unit": "",
            "value": math.nan,
            "interval_s": math.nan,
            "running_percent": math.nan,
        }
    interval = math.nan
    value_index = 0
    first = parse_float(parts[0])
    second = parse_float(parts[1]) if len(parts) > 1 else math.nan
    second_is_counter = math.isfinite(second) or (len(parts) > 1 and parts[1].startswith("<"))
    if math.isfinite(first) and second_is_counter and len(parts) >= 4:
        interval = first
        value_index = 1
    value_text = parts[value_index]
    if value_text.startswith("<"):
        status = value_text.strip("<>").replace(" ", "_")
        value = math.nan
    else:
        value = parse_float(value_text)
        status = "ok" if math.isfinite(value) else "unparsed"
    unit = parts[value_index + 1] if len(parts) > value_index + 1 else ""
    event = parts[value_index + 2] if len(parts) > value_index + 2 else ""
    running_percent = math.nan
    for candidate in parts[value_index + 3 :]:
        if candidate.endswith("%"):
            running_percent = parse_float(candidate.removesuffix("%"))
            break
    return {
        "status": status,
        "event": event,
        "unit": unit,
        "value": value,
        "interval_s": interval,
        "running_percent": running_percent,
    }


def perf_stat_report(
    run_dirs: list[Path],
    metadata_by_dir: dict[str, dict[str, str]],
    campaign_by_dir: dict[str, dict[str, Any]],
) -> Report:
    fields = IDENTITY_FIELDS + [
        "source_id",
        "event",
        "unit",
        "samples_total",
        "valid_samples",
        "unsupported_or_uncounted_samples",
        "unparsed_samples",
        "value_total",
        "value_mean",
        "value_p50",
        "value_p99",
        "running_percent_p50",
        "first_interval_s",
        "last_interval_s",
        "status",
    ]
    grouped: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for run_dir in run_dirs:
        for source in read_csv(run_dir / "external_sources.csv"):
            if source.get("source_type") != "perf_stat":
                continue
            artifact, artifact_status = registered_artifact(run_dir, source.get("artifact_path", ""))
            if artifact is None or artifact_status != "present":
                continue
            for line in artifact.read_text(errors="replace").splitlines():
                parsed = parse_perf_stat_line(line)
                if parsed is None:
                    continue
                key = (str(run_dir), source.get("source_id", ""), str(parsed["event"]), str(parsed["unit"]))
                group = grouped.setdefault(
                    key,
                    {
                        "total": 0,
                        "valid": 0,
                        "unsupported": 0,
                        "unparsed": 0,
                        "values": [],
                        "running": [],
                        "intervals": [],
                        "statuses": [],
                    },
                )
                group["total"] = int(group["total"]) + 1
                group["statuses"].append(str(parsed["status"]))
                if parsed["status"] == "ok":
                    group["valid"] = int(group["valid"]) + 1
                    group["values"].append(float(parsed["value"]))
                elif parsed["status"] in {"not_supported", "not_counted"}:
                    group["unsupported"] = int(group["unsupported"]) + 1
                else:
                    group["unparsed"] = int(group["unparsed"]) + 1
                if math.isfinite(float(parsed["running_percent"])):
                    group["running"].append(float(parsed["running_percent"]))
                if math.isfinite(float(parsed["interval_s"])):
                    group["intervals"].append(float(parsed["interval_s"]))
    rows: list[dict[str, object]] = []
    for (profile_key, source_id, event, unit), group in sorted(grouped.items()):
        values = distribution(group["values"])
        running = distribution(group["running"])
        intervals = sorted(group["intervals"])
        rows.append(
            {
                **identity_for(Path(profile_key), metadata_by_dir, campaign_by_dir),
                "source_id": source_id,
                "event": event,
                "unit": unit,
                "samples_total": group["total"],
                "valid_samples": group["valid"],
                "unsupported_or_uncounted_samples": group["unsupported"],
                "unparsed_samples": group["unparsed"],
                "value_total": sum(group["values"]),
                "value_mean": values["mean"],
                "value_p50": values["p50"],
                "value_p99": values["p99"],
                "running_percent_p50": running["p50"],
                "first_interval_s": intervals[0] if intervals else math.nan,
                "last_interval_s": intervals[-1] if intervals else math.nan,
                "status": joined_status(group["statuses"], "unknown"),
            }
        )
    return Report(fields, rows)


def observer_effect_report(
    campaign_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    metadata_by_dir: dict[str, dict[str, str]],
    campaign_by_dir: dict[str, dict[str, Any]],
) -> Report:
    fields = [
        "campaign_id",
        "case",
        "role",
        "metric_scope",
        "metric_name",
        "unit",
        "variant",
        "baseline_variant",
        "baseline_status",
        "run_statistic",
        "sample_count",
        "mean",
        "stdev",
        "p50",
        "p90",
        "p99",
        "baseline_sample_count",
        "baseline_mean",
        "baseline_p50",
        "p50_delta",
        "p50_delta_percent",
        "effect_estimator",
        "paired_sample_count",
        "paired_delta_mean",
        "paired_delta_stdev",
        "paired_delta_p50",
        "paired_delta_p90",
        "paired_delta_p99",
        "paired_delta_percent_p50",
        "unpaired_variant_samples",
        "unpaired_baseline_samples",
        "ambiguous_trial_count",
        "interpretation",
    ]
    grouped: dict[
        tuple[str, str, str, str, str, str],
        dict[str, list[tuple[str, float]]],
    ] = defaultdict(lambda: defaultdict(list))
    campaign_success_by_dir = {
        str(row.get("profile_dir", "")): campaign_member_succeeded(row)
        for row in campaign_rows
    }
    for row in campaign_rows:
        campaign_id = str(row.get("campaign_id", ""))
        case = str(row.get("case", ""))
        role = str(row.get("role", ""))
        variant = str(row.get("variant", ""))
        trial = str(row.get("trial", ""))
        if not campaign_id or not case or not variant:
            continue
        success = int(campaign_member_succeeded(row))
        duration = parse_float(row.get("duration_s"))
        if success and math.isfinite(duration):
            grouped[(campaign_id, case, role, "process", "process_duration", "s")][variant].append(
                (trial, duration)
            )
        grouped[(campaign_id, case, role, "process", "process_success", "fraction")][variant].append(
            (trial, float(success))
        )

    for row in summary_rows:
        if row.get("event_kind") != "duration" or row.get("event_name") == "PROFILER_PRIMITIVE_CALIBRATION":
            continue
        profile_key = str(row.get("profile_dir", ""))
        if not campaign_success_by_dir.get(profile_key, False):
            continue
        metadata = metadata_by_dir.get(profile_key, {})
        campaign = campaign_by_dir.get(profile_key, {})
        campaign_id = metadata.get("campaign_id", str(campaign.get("campaign_id", "")))
        case = str(campaign.get("case", ""))
        role = metadata.get("role", str(campaign.get("role", "")))
        variant = metadata.get("variant", str(campaign.get("variant", "")))
        trial = metadata.get("trial", str(campaign.get("trial", "")))
        value = parse_float(row.get("p50_us"))
        if campaign_id and case and variant and math.isfinite(value):
            grouped[(campaign_id, case, role, "event", str(row.get("event_name", "")), "us")][variant].append(
                (trial, value)
            )

    rows: list[dict[str, object]] = []
    for key, variants in sorted(grouped.items()):
        campaign_id, case, role, scope, metric_name, unit = key
        baseline_variant = "disabled" if scope == "process" else "in-process"
        baseline_observations = variants.get(baseline_variant, [])
        baseline_values = [value for _, value in baseline_observations]
        baseline_by_trial: dict[str, list[float]] = defaultdict(list)
        for trial, value in baseline_observations:
            if trial:
                baseline_by_trial[trial].append(value)
        baseline = distribution(baseline_values)
        baseline_status = "available" if baseline_values else "missing"
        for variant, observations in sorted(variants.items()):
            values = [value for _, value in observations]
            stats = distribution(values)
            delta = stats["p50"] - baseline["p50"] if baseline_values else math.nan
            delta_percent = 100.0 * delta / baseline["p50"] if baseline_values and baseline["p50"] != 0 else math.nan
            variant_by_trial: dict[str, list[float]] = defaultdict(list)
            for trial, value in observations:
                if trial:
                    variant_by_trial[trial].append(value)
            paired_deltas: list[float] = []
            paired_delta_percentages: list[float] = []
            ambiguous_trials = 0
            paired_trials: set[str] = set()
            for trial in sorted(set(variant_by_trial) | set(baseline_by_trial)):
                variant_trial = variant_by_trial.get(trial, [])
                baseline_trial = baseline_by_trial.get(trial, [])
                if len(variant_trial) == 1 and len(baseline_trial) == 1:
                    paired_trials.add(trial)
                    paired_delta = variant_trial[0] - baseline_trial[0]
                    paired_deltas.append(paired_delta)
                    if baseline_trial[0] != 0:
                        paired_delta_percentages.append(100.0 * paired_delta / baseline_trial[0])
                elif len(variant_trial) > 1 or len(baseline_trial) > 1:
                    ambiguous_trials += 1
            paired = distribution(paired_deltas)
            paired_percent = distribution(paired_delta_percentages)
            unpaired_variant = sum(
                len(trial_values)
                for trial, trial_values in variant_by_trial.items()
                if trial not in paired_trials
            ) + sum(1 for trial, _ in observations if not trial)
            unpaired_baseline = sum(
                len(trial_values)
                for trial, trial_values in baseline_by_trial.items()
                if trial not in paired_trials
            ) + sum(1 for trial, _ in baseline_observations if not trial)
            effect_estimator = (
                "paired_trial_delta"
                if paired_deltas
                else "difference_of_medians_unpaired"
            )
            interpretation = (
                "trial-matched campaign process outcome; disabled is the observer baseline"
                if scope == "process"
                else (
                    "trial-matched differences between per-run event medians; in-process is the baseline "
                    "for incremental PMU/sidecar observer effect"
                )
            )
            rows.append(
                {
                    "campaign_id": campaign_id,
                    "case": case,
                    "role": role,
                    "metric_scope": scope,
                    "metric_name": metric_name,
                    "unit": unit,
                    "variant": variant,
                    "baseline_variant": baseline_variant,
                    "baseline_status": baseline_status,
                    "run_statistic": "campaign_value" if scope == "process" else "per_run_p50",
                    "sample_count": len(values),
                    "mean": stats["mean"],
                    "stdev": stats["stdev"],
                    "p50": stats["p50"],
                    "p90": stats["p90"],
                    "p99": stats["p99"],
                    "baseline_sample_count": len(baseline_values),
                    "baseline_mean": baseline["mean"],
                    "baseline_p50": baseline["p50"],
                    "p50_delta": delta,
                    "p50_delta_percent": delta_percent,
                    "effect_estimator": effect_estimator,
                    "paired_sample_count": len(paired_deltas),
                    "paired_delta_mean": paired["mean"],
                    "paired_delta_stdev": paired["stdev"],
                    "paired_delta_p50": paired["p50"],
                    "paired_delta_p90": paired["p90"],
                    "paired_delta_p99": paired["p99"],
                    "paired_delta_percent_p50": paired_percent["p50"],
                    "unpaired_variant_samples": unpaired_variant,
                    "unpaired_baseline_samples": unpaired_baseline,
                    "ambiguous_trial_count": ambiguous_trials,
                    "interpretation": interpretation,
                }
            )
    return Report(fields, rows)


def campaign_completeness_report(
    campaign_rows: list[dict[str, object]], integrity_rows: list[dict[str, object]]
) -> Report:
    fields = [
        "campaign_id",
        "experiment_id",
        "case",
        "variant",
        "trial",
        "roles_present",
        "member_count",
        "role_count",
        "missing_roles",
        "unexpected_roles",
        "duplicate_roles",
        "finished_roles",
        "successful_roles",
        "finalized_roles",
        "integrity_valid_roles",
        "paired_complete",
        "status",
    ]
    integrity_by_dir: dict[str, bool] = {}
    for row in integrity_rows:
        key = str(row.get("profile_dir", ""))
        integrity_by_dir[key] = integrity_by_dir.get(key, True) and bool(parse_int(row.get("valid")))
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in campaign_rows:
        campaign_id = str(row.get("campaign_id", ""))
        experiment_id = str(row.get("experiment_id", ""))
        if campaign_id and experiment_id:
            key = (
                campaign_id,
                experiment_id,
                str(row.get("case", "")),
                str(row.get("variant", "")),
                str(row.get("trial", "")),
            )
            grouped[key].append(row)
    rows: list[dict[str, object]] = []
    expected = {"gNB", "nrUE"}
    for key, members in sorted(grouped.items()):
        role_counts = Counter(str(member.get("role", "")) for member in members)
        roles = set(role_counts)
        missing_roles = expected - roles
        unexpected_roles = roles - expected
        duplicate_roles = {role: count for role, count in role_counts.items() if count > 1}
        finished = {str(member.get("role", "")) for member in members if member.get("status") == "finished"}
        successful = {
            str(member.get("role", ""))
            for member in members
            if campaign_member_succeeded(member)
        }
        finalized = {
            str(member.get("role", ""))
            for member in members
            if member.get("archive_status") == "finalization_pending"
            and parse_int(member.get("archive_manifest_present"))
        }
        valid = {
            str(member.get("role", ""))
            for member in members
            if integrity_by_dir.get(str(member.get("profile_dir", "")), False)
        }
        complete = (
            len(members) == len(expected)
            and role_counts == Counter({"gNB": 1, "nrUE": 1})
            and finished == expected
            and successful == expected
            and finalized == expected
            and valid == expected
        )
        issues = []
        if missing_roles:
            issues.append("missing_role")
        if unexpected_roles:
            issues.append("unexpected_role")
        if duplicate_roles:
            issues.append("duplicate_role")
        if finished != expected:
            issues.append("unfinished_role")
        if successful != expected:
            issues.append("unsuccessful_role")
        if finalized != expected:
            issues.append("unfinalized_role")
        if valid != expected:
            issues.append("integrity_invalid")
        rows.append(
            {
                "campaign_id": key[0],
                "experiment_id": key[1],
                "case": key[2],
                "variant": key[3],
                "trial": key[4],
                "roles_present": ";".join(sorted(roles)),
                "member_count": len(members),
                "role_count": len(roles),
                "missing_roles": ";".join(sorted(missing_roles)),
                "unexpected_roles": ";".join(sorted(unexpected_roles)),
                "duplicate_roles": ";".join(
                    f"{role}:{count}" for role, count in sorted(duplicate_roles.items())
                ),
                "finished_roles": ";".join(sorted(finished)),
                "successful_roles": ";".join(sorted(successful)),
                "finalized_roles": ";".join(sorted(finalized)),
                "integrity_valid_roles": ";".join(sorted(valid)),
                "paired_complete": int(complete),
                "status": "complete" if complete else ";".join(issues or ["incomplete"]),
            }
        )
    return Report(fields, rows)


def build_extended_reports(
    profile_dirs: list[Path],
    run_dirs: list[Path],
    metadata_by_dir: dict[str, dict[str, str]],
    settings_by_dir: dict[str, dict[str, str]],
    summary_rows: list[dict[str, object]],
    transport_fault_rows: list[dict[str, str]],
) -> dict[str, Report]:
    campaign_by_dir: dict[str, dict[str, Any]] = {}
    campaign_status_by_dir: dict[str, str] = {}
    for run_dir in run_dirs:
        campaign, status = read_campaign(run_dir / "campaign_run.json")
        campaign_by_dir[str(run_dir)] = campaign
        campaign_status_by_dir[str(run_dir)] = status

    campaign = campaign_report(run_dirs, metadata_by_dir, campaign_by_dir, campaign_status_by_dir)
    integrity = archive_integrity_report(run_dirs, metadata_by_dir, campaign_by_dir)
    availability, pmu, pmu_quality = pmu_reports(profile_dirs, metadata_by_dir, settings_by_dir, campaign_by_dir)
    transport, faults = transport_reports(summary_rows, transport_fault_rows, metadata_by_dir, campaign_by_dir)
    reports = {
        "clock_quality.csv": clock_quality_report(
            profile_dirs, metadata_by_dir, campaign_by_dir
        ),
        "host_metrics_summary.csv": host_metrics_report(
            profile_dirs, metadata_by_dir, campaign_by_dir
        ),
        "profiler_primitive_overhead_summary.csv": primitive_overhead_report(
            profile_dirs, metadata_by_dir, campaign_by_dir
        ),
        "pmu_availability_summary.csv": availability,
        "pmu_summary.csv": pmu,
        "pmu_quality.csv": pmu_quality,
        "thread_scheduler_summary.csv": scheduler_report(profile_dirs, metadata_by_dir, campaign_by_dir),
        "kernel_interference_summary.csv": kernel_interference_report(profile_dirs, metadata_by_dir, campaign_by_dir),
        "transport_summary.csv": transport,
        "transport_faults.csv": faults,
        "collection_overhead_summary.csv": collection_overhead_report(profile_dirs, metadata_by_dir, campaign_by_dir),
        "archive_integrity.csv": integrity,
        "external_sources.csv": external_sources_report(run_dirs, metadata_by_dir, campaign_by_dir),
        "perf_stat_summary.csv": perf_stat_report(run_dirs, metadata_by_dir, campaign_by_dir),
        "campaign_runs.csv": campaign,
        "campaign_completeness.csv": campaign_completeness_report(campaign.rows, integrity.rows),
        "observer_effect_summary.csv": observer_effect_report(
            campaign.rows, summary_rows, metadata_by_dir, campaign_by_dir
        ),
    }
    return reports
