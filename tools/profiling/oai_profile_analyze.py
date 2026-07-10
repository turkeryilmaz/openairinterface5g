#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Summarize OAI profiler CSV output.

The C profiler writes archival event records.  This script intentionally stays
outside the real-time process and performs the heavier statistical work offline.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


QUANTILES = (0.5, 0.9, 0.95, 0.99, 0.999)


@dataclass
class GroupStats:
    durations_us: list[float] = field(default_factory=list)
    aux0_values: list[int] = field(default_factory=list)
    aux1_values: list[int] = field(default_factory=list)
    aux2_values: list[int] = field(default_factory=list)
    aux3_values: list[int] = field(default_factory=list)

    def add(self, row: dict[str, str]) -> None:
        self.durations_us.append(float(row["duration_us"]))
        self.aux0_values.append(int(row["aux0"]))
        self.aux1_values.append(int(row["aux1"]))
        self.aux2_values.append(int(row["aux2"]))
        self.aux3_values.append(int(row["aux3"]))


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return sorted_values[lo]
    weight = position - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def mean(values: list[float]) -> float:
    return math.nan if not values else sum(values) / len(values)


def stdev(values: list[float], avg: float) -> float:
    if len(values) < 2:
        return math.nan
    return math.sqrt(sum((v - avg) ** 2 for v in values) / (len(values) - 1))


def read_metadata(profile_dir: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    metadata_path = profile_dir / "metadata.txt"
    if not metadata_path.exists():
        return metadata
    for line in metadata_path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key] = value
    return metadata


def read_settings(profile_dir: Path) -> dict[str, str]:
    settings: dict[str, str] = {}
    settings_path = profile_dir / "settings.csv"
    if not settings_path.exists():
        return settings
    with settings_path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = row.get("key", "")
            if key:
                settings[key] = row.get("value", "")
    return settings


def parse_int(value: str | None, default: int = 0) -> int:
    try:
        return int(value or "")
    except ValueError:
        return default


def read_drops(profile_dir: Path) -> int:
    drops_path = profile_dir / "drops.csv"
    if not drops_path.exists():
        return 0
    total = 0
    with drops_path.open(newline="") as f:
        for row in csv.DictReader(f):
            total += int(row.get("dropped_records", "0") or 0)
    return total


def discover_profile_dirs(inputs: Iterable[Path]) -> list[Path]:
    discovered: set[Path] = set()
    for input_path in inputs:
        path = input_path.resolve()
        if (path / "events.csv").is_file() and (path / "metadata.txt").is_file():
            discovered.add(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)
        for metadata_path in path.rglob("metadata.txt"):
            profile_dir = metadata_path.parent
            if (profile_dir / "events.csv").is_file():
                discovered.add(profile_dir.resolve())
    if not discovered:
        raise FileNotFoundError("no profiler directories containing metadata.txt and events.csv were found")
    return sorted(discovered)


def read_sync_bounds(profile_dir: Path) -> tuple[int, int]:
    sync_path = profile_dir / "sync.csv"
    first = 0
    last = 0
    if not sync_path.exists():
        return first, last
    with sync_path.open(newline="") as f:
        for row in csv.DictReader(f):
            realtime_ns = parse_int(row.get("realtime_ns"))
            if realtime_ns <= 0:
                continue
            if first == 0:
                first = realtime_ns
            last = realtime_ns
    return first, last


def finite_max(current: float, value: str | None) -> float:
    try:
        parsed = float(value or "")
    except ValueError:
        return current
    return parsed if math.isnan(current) or parsed > current else current


def finite_min(current: float, value: str | None) -> float:
    try:
        parsed = float(value or "")
    except ValueError:
        return current
    if parsed < 0:
        return current
    return parsed if math.isnan(current) or parsed < current else current


def summarize_host_metrics(profile_dir: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "profile_dir": str(profile_dir),
        "samples": 0,
        "temperature_max_millicelsius": math.nan,
        "rpi_throttled_valid_samples": 0,
        "rpi_throttled_or": 0,
        "cpu_frequency_min_khz": math.nan,
        "cpu_frequency_mean_khz": math.nan,
        "cpu_frequency_max_khz": math.nan,
        "cpu_busy_max_percent": math.nan,
        "load1_max": math.nan,
        "mem_available_min_kb": math.nan,
        "process_rss_max_kb": math.nan,
        "involuntary_context_switches_max": math.nan,
    }
    metrics_path = profile_dir / "host_metrics.csv"
    if not metrics_path.exists():
        return result

    frequency_sum = 0.0
    frequency_count = 0
    with metrics_path.open(newline="") as f:
        for row in csv.DictReader(f):
            result["samples"] = int(result["samples"]) + 1
            result["temperature_max_millicelsius"] = finite_max(
                float(result["temperature_max_millicelsius"]), row.get("thermal_max_millicelsius")
            )
            if parse_int(row.get("rpi_throttled_valid")):
                result["rpi_throttled_valid_samples"] = int(result["rpi_throttled_valid_samples"]) + 1
                result["rpi_throttled_or"] = int(result["rpi_throttled_or"]) | parse_int(row.get("rpi_throttled_raw"))
            result["cpu_frequency_min_khz"] = finite_min(
                float(result["cpu_frequency_min_khz"]), row.get("cpu_frequency_min_khz")
            )
            result["cpu_frequency_max_khz"] = finite_max(
                float(result["cpu_frequency_max_khz"]), row.get("cpu_frequency_max_khz")
            )
            frequency = float(row.get("cpu_frequency_avg_khz", "-1") or -1)
            if frequency >= 0:
                frequency_sum += frequency
                frequency_count += 1
            result["cpu_busy_max_percent"] = finite_max(
                float(result["cpu_busy_max_percent"]), row.get("cpu_busy_percent")
            )
            result["load1_max"] = finite_max(float(result["load1_max"]), row.get("load1"))
            result["mem_available_min_kb"] = finite_min(
                float(result["mem_available_min_kb"]), row.get("mem_available_kb")
            )
            result["process_rss_max_kb"] = finite_max(
                float(result["process_rss_max_kb"]), row.get("process_rss_kb")
            )
            result["involuntary_context_switches_max"] = finite_max(
                float(result["involuntary_context_switches_max"]), row.get("involuntary_context_switches")
            )
    if frequency_count:
        result["cpu_frequency_mean_khz"] = frequency_sum / frequency_count
    return result


def build_run_inventory(
    profile_dirs: Iterable[Path],
    metadata_by_dir: dict[str, dict[str, str]],
    settings_by_dir: dict[str, dict[str, str]],
    drops_by_dir: dict[str, int],
    host_by_dir: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for profile_dir in profile_dirs:
        key = str(profile_dir)
        metadata = metadata_by_dir[key]
        settings = settings_by_dir[key]
        sync_start, sync_end = read_sync_bounds(profile_dir)
        start_ns = parse_int(metadata.get("start_realtime_ns"), sync_start)
        end_ns = parse_int(metadata.get("end_realtime_ns"), sync_end or start_ns)
        if end_ns < start_ns:
            end_ns = start_ns
        process_name = metadata.get("process_name", "unknown")
        role = metadata.get("role", "")
        if not role:
            role = "nrUE" if "uesoftmodem" in process_name else "gNB" if "softmodem" in process_name else "unknown"
        runs.append(
            {
                "profile_dir": key,
                "run_id": metadata.get("run_id", profile_dir.name),
                "experiment_id": metadata.get("experiment_id", ""),
                "role": role,
                "process_name": process_name,
                "hostname": metadata.get("hostname", "unknown"),
                "start_realtime_ns": start_ns,
                "end_realtime_ns": end_ns,
                "duration_s": (end_ns - start_ns) / 1e9,
                "clean_shutdown": metadata.get("clean_shutdown", "0"),
                "config_source": metadata.get("config_source", ""),
                "build_oai_version": metadata.get("build_oai_version", metadata.get("oai_version", "")),
                "runtime_git_branch": metadata.get("runtime_git_branch", ""),
                "runtime_git_head": metadata.get("runtime_git_head", ""),
                "runtime_git_dirty": metadata.get("runtime_git_dirty", ""),
                "min_rxtxtime": settings.get("gnb.min_rxtxtime", ""),
                "usrp_tx_thread": settings.get("softmodem.usrp_tx_thread", ""),
                "continuous_tx": settings.get("softmodem.continuous_tx", ""),
                "sample_advance": settings.get("softmodem.sample_advance", ""),
                "thread_pool": settings.get("softmodem.thread_pool", ""),
                "drops_total": drops_by_dir[key],
                "host_metric_samples": host_by_dir[key]["samples"],
            }
        )
    return sorted(runs, key=lambda row: (int(row["start_realtime_ns"]), str(row["role"]), str(row["profile_dir"])))


def run_overlap_ns(gnb: dict[str, object], ue: dict[str, object]) -> int:
    return max(
        0,
        min(int(gnb["end_realtime_ns"]), int(ue["end_realtime_ns"]))
        - max(int(gnb["start_realtime_ns"]), int(ue["start_realtime_ns"])),
    )


def paired_row(
    status: str,
    method: str,
    gnb: dict[str, object] | None,
    ue: dict[str, object] | None,
    notes: str = "",
) -> dict[str, object]:
    gnb_start = int(gnb["start_realtime_ns"]) if gnb else 0
    ue_start = int(ue["start_realtime_ns"]) if ue else 0
    overlap_ns = run_overlap_ns(gnb, ue) if gnb and ue else 0
    return {
        "status": status,
        "method": method,
        "experiment_id": (ue or gnb or {}).get("experiment_id", ""),
        "gnb_run_id": gnb.get("run_id", "") if gnb else "",
        "ue_run_id": ue.get("run_id", "") if ue else "",
        "gnb_profile_dir": gnb.get("profile_dir", "") if gnb else "",
        "ue_profile_dir": ue.get("profile_dir", "") if ue else "",
        "start_delta_ms": (ue_start - gnb_start) / 1e6 if gnb and ue else math.nan,
        "overlap_s": overlap_ns / 1e9,
        "notes": notes,
    }


def build_pairs(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    gnbs = [run for run in runs if run["role"] == "gNB"]
    ues = [run for run in runs if run["role"] == "nrUE"]
    rows: list[dict[str, object]] = []
    paired_gnb_dirs: set[str] = set()

    for ue in ues:
        experiment_id = str(ue["experiment_id"])
        exact = [gnb for gnb in gnbs if experiment_id and gnb["experiment_id"] == experiment_id]
        if len(exact) == 1:
            rows.append(paired_row("paired", "experiment_id", exact[0], ue))
            paired_gnb_dirs.add(str(exact[0]["profile_dir"]))
            continue
        if len(exact) > 1:
            candidates = ";".join(str(gnb["profile_dir"]) for gnb in exact)
            rows.append(paired_row("ambiguous", "experiment_id", None, ue, f"gNB candidates: {candidates}"))
            continue

        overlapping = [gnb for gnb in gnbs if run_overlap_ns(gnb, ue) > 0]
        if len(overlapping) == 1:
            rows.append(paired_row("paired", "wallclock_overlap", overlapping[0], ue))
            paired_gnb_dirs.add(str(overlapping[0]["profile_dir"]))
        elif len(overlapping) > 1:
            candidates = ";".join(str(gnb["profile_dir"]) for gnb in overlapping)
            rows.append(paired_row("ambiguous", "wallclock_overlap", None, ue, f"gNB candidates: {candidates}"))
        else:
            rows.append(paired_row("unmatched", "none", None, ue, "no overlapping gNB profile"))

    for gnb in gnbs:
        if str(gnb["profile_dir"]) not in paired_gnb_dirs:
            rows.append(paired_row("unmatched", "none", gnb, None, "no uniquely paired nrUE profile"))
    return rows


def write_rows(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def iter_event_rows(profile_dirs: Iterable[Path], event_filter: set[str] | None) -> Iterable[tuple[Path, dict[str, str]]]:
    for profile_dir in profile_dirs:
        events_path = profile_dir / "events.csv"
        if not events_path.exists():
            raise FileNotFoundError(f"missing {events_path}")
        with events_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if event_filter and row["event_name"] not in event_filter:
                    continue
                yield profile_dir, row


def write_summary(path: Path | None, rows: list[dict[str, object]]) -> None:
    fields = [
        "profile_dir",
        "process_name",
        "event_name",
        "count",
        "min_us",
        "p50_us",
        "p90_us",
        "p95_us",
        "p99_us",
        "p99_9_us",
        "max_us",
        "mean_us",
        "stdev_us",
        "drops_total",
    ]
    if path is None:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_by_thread(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["profile_dir", "process_name", "thread_name", "event_name", "count", "mean_us", "p99_us", "max_us"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_deadline_misses(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "profile_dir",
        "seq",
        "tid",
        "thread_name",
        "frame",
        "slot",
        "current_time_us",
        "deadline_us",
        "miss_us",
        "start_tick",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_rows(groups: dict[tuple[str, str], GroupStats], metadata_by_dir: dict[str, dict[str, str]], drops_by_dir: dict[str, int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (profile_dir, event_name), stats in sorted(groups.items()):
        durations = sorted(stats.durations_us)
        avg = mean(durations)
        q = {name: quantile(durations, value) for name, value in [("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99), ("p99_9", 0.999)]}
        rows.append(
            {
                "profile_dir": profile_dir,
                "process_name": metadata_by_dir.get(profile_dir, {}).get("process_name", "unknown"),
                "event_name": event_name,
                "count": len(durations),
                "min_us": durations[0] if durations else math.nan,
                "p50_us": q["p50"],
                "p90_us": q["p90"],
                "p95_us": q["p95"],
                "p99_us": q["p99"],
                "p99_9_us": q["p99_9"],
                "max_us": durations[-1] if durations else math.nan,
                "mean_us": avg,
                "stdev_us": stdev(durations, avg),
                "drops_total": drops_by_dir.get(profile_dir, 0),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize OAI profiler process directories or archive roots")
    parser.add_argument(
        "profile_dir",
        nargs="+",
        type=Path,
        help="A process profile directory or a root containing profile directories",
    )
    parser.add_argument("--event", action="append", help="Event name to include; may be given multiple times")
    parser.add_argument("--output-dir", type=Path, help="Write all CSV analysis outputs here")
    args = parser.parse_args()

    profile_dirs = discover_profile_dirs(args.profile_dir)
    event_filter = set(args.event) if args.event else None
    metadata_by_dir = {str(p): read_metadata(p) for p in profile_dirs}
    settings_by_dir = {str(p): read_settings(p) for p in profile_dirs}
    drops_by_dir = {str(p): read_drops(p) for p in profile_dirs}
    host_by_dir = {str(p): summarize_host_metrics(p) for p in profile_dirs}
    run_rows = build_run_inventory(profile_dirs, metadata_by_dir, settings_by_dir, drops_by_dir, host_by_dir)
    pair_rows = build_pairs(run_rows)

    event_groups: dict[tuple[str, str], GroupStats] = defaultdict(GroupStats)
    thread_groups: dict[tuple[str, str, str], GroupStats] = defaultdict(GroupStats)
    deadline_rows: list[dict[str, str]] = []

    for profile_dir, row in iter_event_rows(profile_dirs, event_filter):
        profile_key = str(profile_dir)
        event_groups[(profile_key, row["event_name"])].add(row)
        thread_groups[(profile_key, row["thread_name"], row["event_name"])].add(row)
        if row["event_name"] == "UE_TX_DEADLINE_MISS":
            deadline_rows.append(
                {
                    "profile_dir": profile_key,
                    "seq": row["seq"],
                    "tid": row["tid"],
                    "thread_name": row["thread_name"],
                    "frame": row["frame"],
                    "slot": row["slot"],
                    "current_time_us": row["aux0"],
                    "deadline_us": row["aux1"],
                    "miss_us": row["aux2"],
                    "start_tick": row["start_tick"],
                }
            )

    summary_rows = build_rows(event_groups, metadata_by_dir, drops_by_dir)
    thread_rows: list[dict[str, object]] = []
    for (profile_dir, thread_name, event_name), stats in sorted(thread_groups.items()):
        durations = sorted(stats.durations_us)
        avg = mean(durations)
        thread_rows.append(
            {
                "profile_dir": profile_dir,
                "process_name": metadata_by_dir.get(profile_dir, {}).get("process_name", "unknown"),
                "thread_name": thread_name,
                "event_name": event_name,
                "count": len(durations),
                "mean_us": avg,
                "p99_us": quantile(durations, 0.99),
                "max_us": durations[-1] if durations else math.nan,
            }
        )

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_summary(args.output_dir / "summary.csv", summary_rows)
        write_by_thread(args.output_dir / "by_thread.csv", thread_rows)
        write_deadline_misses(args.output_dir / "deadline_misses.csv", deadline_rows)
        write_rows(
            args.output_dir / "runs.csv",
            [
                "profile_dir",
                "run_id",
                "experiment_id",
                "role",
                "process_name",
                "hostname",
                "start_realtime_ns",
                "end_realtime_ns",
                "duration_s",
                "clean_shutdown",
                "config_source",
                "build_oai_version",
                "runtime_git_branch",
                "runtime_git_head",
                "runtime_git_dirty",
                "min_rxtxtime",
                "usrp_tx_thread",
                "continuous_tx",
                "sample_advance",
                "thread_pool",
                "drops_total",
                "host_metric_samples",
            ],
            run_rows,
        )
        write_rows(
            args.output_dir / "pairs.csv",
            [
                "status",
                "method",
                "experiment_id",
                "gnb_run_id",
                "ue_run_id",
                "gnb_profile_dir",
                "ue_profile_dir",
                "start_delta_ms",
                "overlap_s",
                "notes",
            ],
            pair_rows,
        )
        write_rows(
            args.output_dir / "host_summary.csv",
            [
                "profile_dir",
                "samples",
                "temperature_max_millicelsius",
                "rpi_throttled_valid_samples",
                "rpi_throttled_or",
                "cpu_frequency_min_khz",
                "cpu_frequency_mean_khz",
                "cpu_frequency_max_khz",
                "cpu_busy_max_percent",
                "load1_max",
                "mem_available_min_kb",
                "process_rss_max_kb",
                "involuntary_context_switches_max",
            ],
            [host_by_dir[str(profile_dir)] for profile_dir in profile_dirs],
        )
    else:
        write_summary(None, summary_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
