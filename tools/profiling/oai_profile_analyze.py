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


def read_drops(profile_dir: Path) -> int:
    drops_path = profile_dir / "drops.csv"
    if not drops_path.exists():
        return 0
    total = 0
    with drops_path.open(newline="") as f:
        for row in csv.DictReader(f):
            total += int(row.get("dropped_records", "0") or 0)
    return total


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
    parser = argparse.ArgumentParser(description="Summarize OAI profiler CSV output directories")
    parser.add_argument("profile_dir", nargs="+", type=Path, help="Directory containing events.csv, metadata.txt, and drops.csv")
    parser.add_argument("--event", action="append", help="Event name to include; may be given multiple times")
    parser.add_argument("--output-dir", type=Path, help="Write summary.csv, by_thread.csv, and deadline_misses.csv here")
    args = parser.parse_args()

    profile_dirs = [p.resolve() for p in args.profile_dir]
    event_filter = set(args.event) if args.event else None
    metadata_by_dir = {str(p): read_metadata(p) for p in profile_dirs}
    drops_by_dir = {str(p): read_drops(p) for p in profile_dirs}

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
    else:
        write_summary(None, summary_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
