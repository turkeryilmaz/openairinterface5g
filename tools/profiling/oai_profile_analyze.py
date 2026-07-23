#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Summarize OAI profiler CSV output.

The C profiler writes archival event records.  This script intentionally stays
outside the real-time process and performs the heavier statistical work offline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from array import array
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from oai_profile_clock import ClockMapper, EVENT_TIMELINE_FIELDS, event_timeline_row
from oai_profile_deadlines import (
    DEADLINE_CHECK_FIELDS,
    DEADLINE_EVENT_NAMES,
    DEADLINE_SUMMARY_FIELDS,
    build_deadline_reports,
    is_nrue_profile,
)
from oai_profile_reports import build_extended_reports, discover_run_dirs



QUANTILES = (0.5, 0.9, 0.95, 0.99, 0.999)
HIERARCHY_FIELDS = [
    "profile_dir",
    "seq",
    "tid",
    "thread_name",
    "event_name",
    "frame",
    "slot",
    "absolute_slot",
    "correlation_id",
    "span_id",
    "parent_id",
    "parent_event_name",
    "parent_relation",
    "absolute_slot_delta_from_parent",
    "nesting_depth",
    "inclusive_us",
    "direct_children",
    "duration_children",
    "instant_children",
    "contained_duration_children",
    "noncontained_duration_children",
    "correlation_mismatch_children",
    "child_duration_sum_us",
    "child_interval_union_us",
    "child_overlap_us",
    "exclusive_us",
    "exclusive_valid",
]
HIERARCHY_ANOMALY_FIELDS = [
    "profile_dir",
    "seq",
    "event_name",
    "span_id",
    "parent_id",
    "relation",
    "correlation_id",
    "parent_correlation_id",
    "absolute_slot",
    "parent_absolute_slot",
    "start_tick",
    "end_tick",
    "parent_start_tick",
    "parent_end_tick",
]
CORRELATION_FIELDS = [
    "profile_dir",
    "correlation_id",
    "absolute_slot_min",
    "absolute_slot_max",
    "first_tick",
    "last_tick",
    "elapsed_us",
    "event_count",
    "duration_count",
    "instant_count",
    "root_count",
    "parented_count",
    "missing_parent_count",
    "max_nesting_depth",
    "thread_count",
    "cpu_migrations",
    "root_events",
]
CAUSAL_EDGE_SUMMARY_FIELDS = [
    "profile_dir",
    "parent_event_name",
    "event_name",
    "relation",
    "absolute_slot_delta_from_parent",
    "temporal_shape",
    "count",
    "correlation_match_count",
    "correlation_unknown_count",
    "boundary_distance_min_us",
    "boundary_distance_p50_us",
    "boundary_distance_p90_us",
    "boundary_distance_p99_us",
    "boundary_distance_p99_9_us",
    "boundary_distance_max_us",
    "boundary_distance_mean_us",
]
ANALYSIS_INPUT_FIELDS = [
    "input_path",
    "input_kind",
    "run_id",
    "experiment_id",
    "campaign_id",
    "variant",
    "trial",
    "role",
    "hostname",
    "schema_version",
    "archive_manifest_status",
    "archive_manifest_path",
    "archive_manifest_sha256",
]
ANALYSIS_MANIFEST_FIELDS = [
    "artifact",
    "status",
    "row_count",
    "size_bytes",
    "sha256",
    "output_profile",
    "notes",
]
ANALYSIS_PROVENANCE_FIELDS = [
    "record_type",
    "name",
    "value",
    "path",
    "sha256",
]
DEADLINE_MISS_FIELDS = [
    "profile_dir",
    "schema_version",
    "seq",
    "tid",
    "thread_name",
    "frame",
    "slot",
    "absolute_slot",
    "correlation_id",
    "span_id",
    "parent_id",
    "cpu_start",
    "cpu_end",
    "cpu_migrated",
    "current_time_us",
    "deadline_us",
    "miss_us",
    "start_tick",
]
MIGRATION_FIELDS = [
    "profile_dir",
    "schema_version",
    "seq",
    "tid",
    "thread_name",
    "event_name",
    "event_kind",
    "frame",
    "slot",
    "absolute_slot",
    "correlation_id",
    "span_id",
    "parent_id",
    "nesting_depth",
    "cpu_start",
    "cpu_end",
    "start_tick",
    "duration_tick",
    "duration_us",
]


@dataclass
class GroupStats:
    durations_us: array[float] = field(default_factory=lambda: array("d"))
    schema_versions: set[str] = field(default_factory=set)
    event_roles: set[str] = field(default_factory=set)
    subsystems: set[str] = field(default_factory=set)
    event_classes: set[str] = field(default_factory=set)
    event_kinds: set[str] = field(default_factory=set)
    detail_levels: set[str] = field(default_factory=set)
    correlated_count: int = 0
    parented_count: int = 0
    absolute_slot_count: int = 0
    cpu_observed_count: int = 0
    cpu_migrations: int = 0

    def add(self, row: dict[str, str]) -> None:
        self.durations_us.append(float(row["duration_us"]))
        self.schema_versions.add(row["schema_version"])
        self.event_roles.add(row["event_role"])
        self.subsystems.add(row["subsystem"])
        self.event_classes.add(row["event_class"])
        self.event_kinds.add(row["event_kind"])
        self.detail_levels.add(row["detail_level"])
        self.correlated_count += parse_int(row.get("correlation_id")) > 0
        self.parented_count += parse_int(row.get("parent_id")) > 0
        self.absolute_slot_count += parse_int(row.get("absolute_slot"), -1) >= 0
        cpu_start = parse_int(row.get("cpu_start"), -1)
        cpu_end = parse_int(row.get("cpu_end"), -1)
        self.cpu_observed_count += cpu_start >= 0 and cpu_end >= 0
        self.cpu_migrations += parse_int(row.get("cpu_migrated")) != 0


@dataclass(slots=True)
class HierarchyRecord:
    schema_version: str
    seq: int
    tid: int
    thread_name: str
    event_name: str
    event_kind: str
    nesting_depth: int
    frame: int
    slot: int
    absolute_slot: int
    correlation_id: int
    span_id: int
    parent_id: int
    cpu_start: int
    cpu_end: int
    cpu_migrated: int
    start_tick: int
    duration_tick: int
    duration_us: float

    @property
    def end_tick(self) -> int:
        return self.start_tick + self.duration_tick


@dataclass
class ExclusiveStats:
    inclusive_us: array[float] = field(default_factory=lambda: array("d"))
    exclusive_us: array[float] = field(default_factory=lambda: array("d"))
    child_union_us: array[float] = field(default_factory=lambda: array("d"))
    total_count: int = 0
    valid_count: int = 0
    overlapping_children_count: int = 0
    noncontained_children: int = 0
    correlation_mismatches: int = 0


@dataclass
class CorrelationStats:
    absolute_slot_min: int = -1
    absolute_slot_max: int = -1
    first_tick: int = 0
    last_tick: int = 0
    event_count: int = 0
    duration_count: int = 0
    instant_count: int = 0
    root_count: int = 0
    parented_count: int = 0
    missing_parent_count: int = 0
    max_nesting_depth: int = 0
    tids: set[int] = field(default_factory=set)
    cpu_migrations: int = 0
    root_events: set[str] = field(default_factory=set)

    def add(self, record: HierarchyRecord, relation: str) -> None:
        if self.event_count == 0:
            self.first_tick = record.start_tick
            self.last_tick = record.end_tick
            self.max_nesting_depth = record.nesting_depth
        else:
            self.first_tick = min(self.first_tick, record.start_tick)
            self.last_tick = max(self.last_tick, record.end_tick)
        if record.absolute_slot >= 0:
            if self.absolute_slot_min < 0:
                self.absolute_slot_min = record.absolute_slot
                self.absolute_slot_max = record.absolute_slot
            else:
                self.absolute_slot_min = min(self.absolute_slot_min, record.absolute_slot)
                self.absolute_slot_max = max(self.absolute_slot_max, record.absolute_slot)
        self.event_count += 1
        self.duration_count += record.event_kind == "duration"
        self.instant_count += record.event_kind == "instant"
        self.root_count += record.parent_id == 0
        self.parented_count += record.parent_id > 0
        self.missing_parent_count += relation == "missing_parent"
        self.max_nesting_depth = max(self.max_nesting_depth, record.nesting_depth)
        self.tids.add(record.tid)
        self.cpu_migrations += record.cpu_migrated != 0
        if record.parent_id == 0:
            self.root_events.add(record.event_name)


@dataclass
class CausalEdgeStats:
    boundary_distance_us: array[float] = field(default_factory=lambda: array("d"))
    correlation_match_count: int = 0
    correlation_unknown_count: int = 0


@dataclass(frozen=True)
class ArtifactInfo:
    artifact: str
    row_count: int
    size_bytes: int
    sha256: str


class _HashingTextSink:
    def __init__(self, path: Path) -> None:
        self._stream = path.open("wb")
        self._hash = hashlib.sha256()
        self.size_bytes = 0
        self._closed = False

    def write(self, value: str) -> int:
        data = value.encode("utf-8")
        self._stream.write(data)
        self._hash.update(data)
        self.size_bytes += len(data)
        return len(value)

    def close(self) -> str:
        if self._closed:
            raise RuntimeError("hashing sink already closed")
        try:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            return self._hash.hexdigest()
        finally:
            self._stream.close()
            self._closed = True

    def abort(self) -> None:
        if not self._closed:
            self._stream.close()
            self._closed = True


class TrackedCsvWriter:
    def __init__(self, path: Path, fields: list[str]) -> None:
        self.path = path
        self.fields = fields
        self.row_count = 0
        self._closed = True
        self._sink: _HashingTextSink | None = None
        try:
            self._sink = _HashingTextSink(path)
            self._writer = csv.DictWriter(self._sink, fieldnames=fields)
            self._writer.writeheader()
            self._closed = False
        except BaseException:
            if self._sink is not None:
                self._sink.abort()
            raise

    @property
    def closed(self) -> bool:
        return self._closed

    def writerow(self, row: dict[str, object]) -> None:
        self._writer.writerow(row)
        self.row_count += 1

    def writerows(self, rows: Iterable[dict[str, object]]) -> None:
        for row in rows:
            self.writerow(row)

    def close(self) -> ArtifactInfo:
        if self._closed:
            raise RuntimeError(f"CSV writer already closed: {self.path}")
        assert self._sink is not None
        try:
            digest = self._sink.close()
            return ArtifactInfo(self.path.name, self.row_count, self._sink.size_bytes, digest)
        finally:
            self._closed = True


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


def one_or_mixed(values: set[str], default: str = "") -> str:
    nonempty = sorted(value for value in values if value)
    if not nonempty:
        return default
    return nonempty[0] if len(nonempty) == 1 else "mixed"


def resolve_event_role(descriptor_role: str, process_role: str) -> str:
    if descriptor_role in {"shared", "nrUE/gNB"} and process_role:
        return process_role
    return descriptor_role or process_role or "unknown"


def read_drop_diagnostics(profile_dir: Path) -> dict[str, int]:
    diagnostics = {
        "dropped_records": 0,
        "span_stack_overflows": 0,
        "span_stack_mismatches": 0,
    }
    drops_path = profile_dir / "drops.csv"
    if not drops_path.exists():
        return diagnostics
    with drops_path.open(newline="") as f:
        for row in csv.DictReader(f):
            for key in diagnostics:
                diagnostics[key] += parse_int(row.get(key))
    return diagnostics


def read_event_catalog(profile_dir: Path) -> dict[str, dict[str, str]]:
    catalog_path = profile_dir / "event_catalog.csv"
    if not catalog_path.exists():
        return {}
    catalog: dict[str, dict[str, str]] = {}
    with catalog_path.open(newline="") as f:
        for row in csv.DictReader(f):
            event_name = row.get("event_name", "")
            if event_name:
                catalog[event_name] = row
    return catalog


def normalize_event_row(
    row: dict[str, str],
    metadata: dict[str, str],
    catalog: dict[str, dict[str, str]],
) -> dict[str, str]:
    normalized = dict(row)
    event_name = normalized.get("event_name") or "UNKNOWN"
    descriptor = catalog.get(event_name, {})
    schema_version = normalized.get("schema_version") or metadata.get("schema_version") or "1"
    default_kind = descriptor.get("default_kind") or ("unknown" if schema_version == "1" else "duration")
    defaults = {
        "schema_version": schema_version,
        "seq": "0",
        "tid": "0",
        "thread_name": "unknown",
        "event_id": "0",
        "event_name": event_name,
        "event_kind": default_kind,
        "nesting_depth": "0",
        "frame": "-1",
        "slot": "-1",
        "absolute_slot": "-1",
        "correlation_id": "0",
        "span_id": "0",
        "parent_id": "0",
        "cpu_start": "-1",
        "cpu_end": "-1",
        "flags": "0",
        "aux0": "0",
        "aux1": "0",
        "aux2": "0",
        "aux3": "0",
        "aux0_name": descriptor.get("aux0_name") or "",
        "aux0_unit": descriptor.get("aux0_unit") or "",
        "aux1_name": descriptor.get("aux1_name") or "",
        "aux1_unit": descriptor.get("aux1_unit") or "",
        "aux2_name": descriptor.get("aux2_name") or "",
        "aux2_unit": descriptor.get("aux2_unit") or "",
        "aux3_name": descriptor.get("aux3_name") or "",
        "aux3_unit": descriptor.get("aux3_unit") or "",
        "start_tick": "0",
        "duration_tick": "0",
        "duration_us": "0",
        "event_role": resolve_event_role(descriptor.get("role", ""), metadata.get("role", "")),
        "subsystem": descriptor.get("subsystem") or "unknown",
        "event_class": descriptor.get("event_class") or "unknown",
        "detail_level": descriptor.get("detail_level") or "boundary",
    }
    for key, value in defaults.items():
        if not normalized.get(key):
            normalized[key] = value
    if not normalized.get("cpu_migrated"):
        cpu_start = parse_int(normalized.get("cpu_start"), -1)
        cpu_end = parse_int(normalized.get("cpu_end"), -1)
        normalized["cpu_migrated"] = str(int(cpu_start >= 0 and cpu_end >= 0 and cpu_start != cpu_end))
    return normalized


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
    return sorted(discovered)


def read_sync_bounds(profile_dir: Path) -> tuple[int, int, int, int]:
    sync_path = profile_dir / "sync.csv"
    first_realtime = 0
    last_realtime = 0
    first_monotonic = 0
    last_monotonic = 0
    if not sync_path.exists():
        return first_realtime, last_realtime, first_monotonic, last_monotonic
    with sync_path.open(newline="") as f:
        for row in csv.DictReader(f):
            realtime_ns = parse_int(row.get("realtime_ns"))
            monotonic_ns = parse_int(row.get("monotonic_raw_ns"))
            if realtime_ns > 0:
                if first_realtime == 0:
                    first_realtime = realtime_ns
                last_realtime = realtime_ns
            if monotonic_ns > 0:
                if first_monotonic == 0:
                    first_monotonic = monotonic_ns
                last_monotonic = monotonic_ns
    return first_realtime, last_realtime, first_monotonic, last_monotonic


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
    drop_diagnostics_by_dir: dict[str, dict[str, int]],
    host_by_dir: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for profile_dir in profile_dirs:
        key = str(profile_dir)
        metadata = metadata_by_dir[key]
        settings = settings_by_dir[key]
        sync_start, sync_end, sync_monotonic_start, sync_monotonic_end = read_sync_bounds(profile_dir)
        start_ns = parse_int(metadata.get("start_realtime_ns"), sync_start)
        end_ns = parse_int(metadata.get("end_realtime_ns"), sync_end)
        start_monotonic_ns = parse_int(metadata.get("start_monotonic_raw_ns"), sync_monotonic_start)
        end_monotonic_ns = parse_int(metadata.get("end_monotonic_raw_ns"), sync_monotonic_end)
        duration_s, duration_clock, duration_status = elapsed_duration(
            start_monotonic_ns,
            end_monotonic_ns,
            start_ns,
            end_ns,
        )
        process_name = metadata.get("process_name", "unknown")
        role = metadata.get("role", "")
        if not role:
            role = "nrUE" if "uesoftmodem" in process_name else "gNB" if "softmodem" in process_name else "unknown"
        runs.append(
            {
                "profile_dir": key,
                "schema_version": metadata.get("schema_version", "1"),
                "event_record_size_bytes": metadata.get("event_record_size_bytes", ""),
                "max_nesting_depth": metadata.get("max_nesting_depth", ""),
                "run_id": metadata.get("run_id", profile_dir.name),
                "experiment_id": metadata.get("experiment_id", ""),
                "campaign_id": metadata.get("campaign_id", ""),
                "variant": metadata.get("variant", ""),
                "trial": metadata.get("trial", ""),
                "role": role,
                "process_name": process_name,
                "hostname": metadata.get("hostname", "unknown"),
                "start_realtime_ns": start_ns,
                "end_realtime_ns": end_ns,
                "start_monotonic_raw_ns": start_monotonic_ns,
                "end_monotonic_raw_ns": end_monotonic_ns,
                "duration_s": duration_s,
                "duration_clock": duration_clock,
                "duration_status": duration_status,
                "realtime_clock_regressed": int(start_ns > 0 and end_ns < start_ns),
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
                "span_stack_overflows": drop_diagnostics_by_dir[key]["span_stack_overflows"],
                "span_stack_mismatches": drop_diagnostics_by_dir[key]["span_stack_mismatches"],
                "host_metric_samples": host_by_dir[key]["samples"],
            }
        )
    return sorted(runs, key=lambda row: (int(row["start_realtime_ns"]), str(row["role"]), str(row["profile_dir"])))


def run_overlap_ns(gnb: dict[str, object], ue: dict[str, object]) -> int:
    if not realtime_bounds_valid(gnb) or not realtime_bounds_valid(ue):
        return 0
    return max(
        0,
        min(int(gnb["end_realtime_ns"]), int(ue["end_realtime_ns"]))
        - max(int(gnb["start_realtime_ns"]), int(ue["start_realtime_ns"])),
    )


def realtime_bounds_valid(run: dict[str, object]) -> bool:
    start_ns = int(run["start_realtime_ns"])
    end_ns = int(run["end_realtime_ns"])
    return start_ns > 0 and end_ns >= start_ns


def paired_row(
    status: str,
    method: str,
    gnb: dict[str, object] | None,
    ue: dict[str, object] | None,
    notes: str = "",
    gnb_candidate_count: int = 0,
    ue_candidate_count: int = 0,
) -> dict[str, object]:
    gnb_start = int(gnb["start_realtime_ns"]) if gnb else 0
    ue_start = int(ue["start_realtime_ns"]) if ue else 0
    overlap_ns = run_overlap_ns(gnb, ue) if gnb and ue else 0
    realtime_valid = bool(gnb and ue and realtime_bounds_valid(gnb) and realtime_bounds_valid(ue))
    return {
        "status": status,
        "method": method,
        "experiment_id": (ue or gnb or {}).get("experiment_id", ""),
        "gnb_run_id": gnb.get("run_id", "") if gnb else "",
        "ue_run_id": ue.get("run_id", "") if ue else "",
        "gnb_profile_dir": gnb.get("profile_dir", "") if gnb else "",
        "ue_profile_dir": ue.get("profile_dir", "") if ue else "",
        "gnb_candidate_count": gnb_candidate_count,
        "ue_candidate_count": ue_candidate_count,
        "clock_domain": "CLOCK_REALTIME" if gnb and ue else "not_applicable",
        "clock_status": "bounds_valid_alignment_unverified" if realtime_valid else "invalid_or_unavailable",
        "start_delta_ms": (ue_start - gnb_start) / 1e6 if realtime_valid else math.nan,
        "overlap_s": overlap_ns / 1e9,
        "notes": notes,
    }


def build_pairs(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    gnbs = [run for run in runs if run["role"] == "gNB"]
    ues = [run for run in runs if run["role"] == "nrUE"]
    rows: list[dict[str, object]] = []
    identified: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: {"gNB": [], "nrUE": []}
    )
    for run in (*gnbs, *ues):
        experiment_id = str(run["experiment_id"])
        if experiment_id:
            identified[experiment_id][str(run["role"])].append(run)

    for experiment_id, members in sorted(identified.items()):
        experiment_gnbs = members["gNB"]
        experiment_ues = members["nrUE"]
        if len(experiment_gnbs) == 1 and len(experiment_ues) == 1:
            rows.append(
                paired_row(
                    "paired",
                    "experiment_id",
                    experiment_gnbs[0],
                    experiment_ues[0],
                    gnb_candidate_count=1,
                    ue_candidate_count=1,
                )
            )
            continue
        status = "ambiguous" if len(experiment_gnbs) > 1 or len(experiment_ues) > 1 else "unmatched"
        notes = (
            f"experiment_id group contains {len(experiment_gnbs)} gNB and {len(experiment_ues)} nrUE runs; "
            "nonempty experiment IDs do not use wallclock fallback"
        )
        for ue in experiment_ues:
            rows.append(
                paired_row(
                    status,
                    "experiment_id",
                    None,
                    ue,
                    notes,
                    len(experiment_gnbs),
                    len(experiment_ues),
                )
            )
        for gnb in experiment_gnbs:
            rows.append(
                paired_row(
                    status,
                    "experiment_id",
                    gnb,
                    None,
                    notes,
                    len(experiment_gnbs),
                    len(experiment_ues),
                )
            )

    legacy_gnbs = [gnb for gnb in gnbs if not str(gnb["experiment_id"])]
    legacy_ues = [ue for ue in ues if not str(ue["experiment_id"])]
    gnb_candidates = {
        str(gnb["profile_dir"]): [ue for ue in legacy_ues if run_overlap_ns(gnb, ue) > 0]
        for gnb in legacy_gnbs
    }
    ue_candidates = {
        str(ue["profile_dir"]): [gnb for gnb in legacy_gnbs if run_overlap_ns(gnb, ue) > 0]
        for ue in legacy_ues
    }
    paired_gnbs: set[str] = set()
    paired_ues: set[str] = set()
    for ue in legacy_ues:
        ue_key = str(ue["profile_dir"])
        candidates = ue_candidates[ue_key]
        if len(candidates) != 1:
            continue
        gnb = candidates[0]
        gnb_key = str(gnb["profile_dir"])
        if len(gnb_candidates[gnb_key]) != 1:
            continue
        rows.append(
            paired_row(
                "paired",
                "wallclock_overlap_mutual_unique",
                gnb,
                ue,
                "legacy fallback; realtime clock synchronization quality is unverified",
                1,
                1,
            )
        )
        paired_gnbs.add(gnb_key)
        paired_ues.add(ue_key)

    for ue in legacy_ues:
        ue_key = str(ue["profile_dir"])
        if ue_key in paired_ues:
            continue
        candidates = ue_candidates[ue_key]
        status = "unmatched" if not candidates else "ambiguous"
        notes = (
            "no overlapping gNB profile with valid realtime bounds"
            if not candidates
            else "wallclock overlap is not reciprocal and unique"
        )
        rows.append(
            paired_row(
                status,
                "wallclock_overlap_mutual_unique" if candidates else "none",
                None,
                ue,
                notes,
                len(candidates),
                0,
            )
        )
    for gnb in legacy_gnbs:
        gnb_key = str(gnb["profile_dir"])
        if gnb_key in paired_gnbs:
            continue
        candidates = gnb_candidates[gnb_key]
        status = "unmatched" if not candidates else "ambiguous"
        notes = (
            "no overlapping nrUE profile with valid realtime bounds"
            if not candidates
            else "wallclock overlap is not reciprocal and unique"
        )
        rows.append(
            paired_row(
                status,
                "wallclock_overlap_mutual_unique" if candidates else "none",
                gnb,
                None,
                notes,
                0,
                len(candidates),
            )
        )
    return rows


def write_rows(
    path: Path,
    fields: list[str],
    rows: Iterable[dict[str, object]],
) -> ArtifactInfo:
    writer = TrackedCsvWriter(path, fields)
    try:
        writer.writerows(rows)
        return writer.close()
    except BaseException:
        if not writer.closed:
            try:
                writer.close()
            except BaseException:
                pass
        raise


def iter_event_rows(
    profile_dirs: Iterable[Path],
    event_filter: set[str] | None,
    metadata_by_dir: dict[str, dict[str, str]],
    catalogs_by_dir: dict[str, dict[str, dict[str, str]]],
) -> Iterable[tuple[Path, dict[str, str]]]:
    for profile_dir in profile_dirs:
        events_path = profile_dir / "events.csv"
        if not events_path.exists():
            raise FileNotFoundError(f"missing {events_path}")
        with events_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                normalized = normalize_event_row(
                    row,
                    metadata_by_dir[str(profile_dir)],
                    catalogs_by_dir[str(profile_dir)],
                )
                if event_filter and normalized["event_name"] not in event_filter:
                    continue
                yield profile_dir, normalized


def write_event_timeline(
    path: Path,
    profile_dirs: list[Path],
    metadata_by_dir: dict[str, dict[str, str]],
    catalogs_by_dir: dict[str, dict[str, dict[str, str]]],
) -> ArtifactInfo:
    mappers = {
        str(profile_dir): ClockMapper.from_sync(
            profile_dir / "sync.csv",
            parse_int(metadata_by_dir[str(profile_dir)].get("counter_hz")),
        )
        for profile_dir in profile_dirs
    }
    writer = TrackedCsvWriter(path, EVENT_TIMELINE_FIELDS)
    try:
        for profile_dir, row in iter_event_rows(
            profile_dirs,
            None,
            metadata_by_dir,
            catalogs_by_dir,
        ):
            writer.writerow(
                event_timeline_row(
                    profile_dir,
                    metadata_by_dir[str(profile_dir)],
                    row,
                    mappers[str(profile_dir)],
                )
            )
        return writer.close()
    except BaseException:
        if not writer.closed:
            try:
                writer.close()
            except BaseException:
                pass
        raise


def hierarchy_record_from_row(row: dict[str, str]) -> HierarchyRecord | None:
    if row.get("schema_version") != "2":
        return None
    span_id = parse_int(row.get("span_id"))
    if span_id <= 0:
        return None
    try:
        duration_us = float(row.get("duration_us", "0") or 0)
    except ValueError:
        duration_us = 0.0
    return HierarchyRecord(
        schema_version=row["schema_version"],
        seq=parse_int(row.get("seq")),
        tid=parse_int(row.get("tid")),
        thread_name=row.get("thread_name", "unknown"),
        event_name=row.get("event_name", "UNKNOWN"),
        event_kind=row.get("event_kind", "unknown"),
        nesting_depth=parse_int(row.get("nesting_depth")),
        frame=parse_int(row.get("frame"), -1),
        slot=parse_int(row.get("slot"), -1),
        absolute_slot=parse_int(row.get("absolute_slot"), -1),
        correlation_id=parse_int(row.get("correlation_id")),
        span_id=span_id,
        parent_id=parse_int(row.get("parent_id")),
        cpu_start=parse_int(row.get("cpu_start"), -1),
        cpu_end=parse_int(row.get("cpu_end"), -1),
        cpu_migrated=parse_int(row.get("cpu_migrated")),
        start_tick=parse_int(row.get("start_tick")),
        duration_tick=parse_int(row.get("duration_tick")),
        duration_us=duration_us,
    )


def interval_union_ticks(intervals: list[tuple[int, int]]) -> tuple[int, int]:
    positive = sorted((start, end) for start, end in intervals if end > start)
    if not positive:
        return 0, 0
    duration_sum = sum(end - start for start, end in positive)
    union = 0
    current_start, current_end = positive[0]
    for start, end in positive[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            union += current_end - current_start
            current_start, current_end = start, end
    union += current_end - current_start
    return union, duration_sum - union


def parent_relation(record: HierarchyRecord, by_span: dict[int, HierarchyRecord]) -> tuple[str, HierarchyRecord | None]:
    if record.parent_id == 0:
        return "root", None
    parent = by_span.get(record.parent_id)
    if parent is None:
        return "missing_parent", None
    if record.correlation_id and parent.correlation_id and record.correlation_id != parent.correlation_id:
        return "correlation_mismatch", parent
    if record.start_tick >= parent.start_tick and record.end_tick <= parent.end_tick:
        return "temporally_contained", parent
    return "causal_noncontained", parent


def hierarchy_anomaly_row(
    profile_dir: str,
    record: HierarchyRecord,
    relation: str,
    parent: HierarchyRecord | None,
) -> dict[str, object]:
    return {
        "profile_dir": profile_dir,
        "seq": record.seq,
        "event_name": record.event_name,
        "span_id": record.span_id,
        "parent_id": record.parent_id,
        "relation": relation,
        "correlation_id": record.correlation_id,
        "parent_correlation_id": parent.correlation_id if parent else 0,
        "absolute_slot": record.absolute_slot,
        "parent_absolute_slot": parent.absolute_slot if parent else -1,
        "start_tick": record.start_tick,
        "end_tick": record.end_tick,
        "parent_start_tick": parent.start_tick if parent else 0,
        "parent_end_tick": parent.end_tick if parent else 0,
    }


def causal_edge_shape(record: HierarchyRecord, parent: HierarchyRecord) -> tuple[str, int]:
    if record.start_tick >= parent.end_tick:
        return "starts_after_parent", record.start_tick - parent.end_tick
    if record.end_tick <= parent.start_tick:
        return "ends_before_parent", parent.start_tick - record.end_tick
    if record.start_tick < parent.start_tick and record.end_tick > parent.end_tick:
        return (
            "encloses_parent",
            max(parent.start_tick - record.start_tick, record.end_tick - parent.end_tick),
        )
    if record.start_tick < parent.start_tick:
        return "starts_before_parent", parent.start_tick - record.start_tick
    return "overruns_parent", record.end_tick - parent.end_tick


def build_hierarchy_outputs(
    profile_dir: str,
    records: list[HierarchyRecord],
    counter_hz: int,
    hierarchy_writer: TrackedCsvWriter | None,
    anomaly_writer: TrackedCsvWriter,
    correlation_writer: TrackedCsvWriter,
    causal_writer: TrackedCsvWriter | None,
) -> tuple[dict[str, object], dict[tuple[str, str], ExclusiveStats]]:
    by_span: dict[int, HierarchyRecord] = {}
    duplicate_span_ids: set[int] = set()
    for record in records:
        if record.span_id in by_span:
            duplicate_span_ids.add(record.span_id)
        else:
            by_span[record.span_id] = record

    children: dict[int, list[HierarchyRecord]] = defaultdict(list)
    for record in records:
        if record.parent_id > 0:
            children[record.parent_id].append(record)

    exclusive_groups: dict[tuple[str, str], ExclusiveStats] = defaultdict(ExclusiveStats)
    causal_groups: dict[tuple[str, str, int | None, str], CausalEdgeStats] = defaultdict(
        CausalEdgeStats
    )
    for record in records:
        relation, parent = parent_relation(record, by_span)
        if relation in {"missing_parent", "correlation_mismatch", "causal_noncontained"}:
            if relation != "causal_noncontained" or causal_writer is None:
                anomaly_writer.writerow(hierarchy_anomaly_row(profile_dir, record, relation, parent))
            if relation == "causal_noncontained" and parent is not None and causal_writer is not None:
                slot_delta = (
                    record.absolute_slot - parent.absolute_slot
                    if record.absolute_slot >= 0 and parent.absolute_slot >= 0
                    else None
                )
                shape, boundary_distance_tick = causal_edge_shape(record, parent)
                stats = causal_groups[(parent.event_name, record.event_name, slot_delta, shape)]
                stats.boundary_distance_us.append(
                    boundary_distance_tick * 1e6 / counter_hz if counter_hz > 0 else math.nan
                )
                if record.correlation_id > 0 and record.correlation_id == parent.correlation_id:
                    stats.correlation_match_count += 1
                else:
                    stats.correlation_unknown_count += 1

    for span_id in sorted(duplicate_span_ids):
        record = by_span[span_id]
        anomaly_writer.writerow(
            hierarchy_anomaly_row(profile_dir, record, "duplicate_span_id", None)
        )

    for record in records:
        if record.event_kind != "duration":
            continue
        direct_children = children.get(record.span_id, [])
        contained_intervals: list[tuple[int, int]] = []
        duration_children = 0
        contained_children = 0
        noncontained_children = 0
        correlation_mismatches = 0
        instant_children = 0
        child_duration_sum_us = 0.0
        for child in direct_children:
            if child.event_kind != "duration":
                instant_children += 1
                continue
            duration_children += 1
            child_duration_sum_us += child.duration_us
            relation, _ = parent_relation(child, by_span)
            if relation == "temporally_contained":
                contained_children += 1
                contained_intervals.append((child.start_tick, child.end_tick))
            elif relation == "correlation_mismatch":
                correlation_mismatches += 1
            else:
                noncontained_children += 1

        child_union_tick, child_overlap_tick = interval_union_ticks(contained_intervals)
        tick_to_us = record.duration_us / record.duration_tick if record.duration_tick > 0 else 0.0
        child_union_us = child_union_tick * tick_to_us
        child_overlap_us = child_overlap_tick * tick_to_us
        exclusive_tick = max(0, record.duration_tick - child_union_tick)
        exclusive_us = exclusive_tick * tick_to_us
        exclusive_valid = (
            record.duration_tick >= 0
            and record.span_id not in duplicate_span_ids
            and noncontained_children == 0
            and correlation_mismatches == 0
        )
        relation, parent = parent_relation(record, by_span)
        absolute_slot_delta = (
            record.absolute_slot - parent.absolute_slot
            if parent is not None and record.absolute_slot >= 0 and parent.absolute_slot >= 0
            else 0
        )
        if hierarchy_writer is not None:
            hierarchy_writer.writerow(
                {
                "profile_dir": profile_dir,
                "seq": record.seq,
                "tid": record.tid,
                "thread_name": record.thread_name,
                "event_name": record.event_name,
                "frame": record.frame,
                "slot": record.slot,
                "absolute_slot": record.absolute_slot,
                "correlation_id": record.correlation_id,
                "span_id": record.span_id,
                "parent_id": record.parent_id,
                "parent_event_name": parent.event_name if parent else "",
                "parent_relation": relation,
                "absolute_slot_delta_from_parent": absolute_slot_delta,
                "nesting_depth": record.nesting_depth,
                "inclusive_us": record.duration_us,
                "direct_children": len(direct_children),
                "duration_children": duration_children,
                "instant_children": instant_children,
                "contained_duration_children": contained_children,
                "noncontained_duration_children": noncontained_children,
                "correlation_mismatch_children": correlation_mismatches,
                "child_duration_sum_us": child_duration_sum_us,
                "child_interval_union_us": child_union_us,
                "child_overlap_us": child_overlap_us,
                "exclusive_us": exclusive_us,
                "exclusive_valid": int(exclusive_valid),
                }
            )
        stats = exclusive_groups[(profile_dir, record.event_name)]
        stats.total_count += 1
        stats.inclusive_us.append(record.duration_us)
        stats.noncontained_children += noncontained_children
        stats.correlation_mismatches += correlation_mismatches
        stats.overlapping_children_count += child_overlap_tick > 0
        if exclusive_valid:
            stats.valid_count += 1
            stats.exclusive_us.append(exclusive_us)
            stats.child_union_us.append(child_union_us)

    relation_counts: dict[str, int] = defaultdict(int)
    absolute_slot_mismatches = 0
    correlation_groups: dict[int, CorrelationStats] = defaultdict(CorrelationStats)
    for record in records:
        relation, parent = parent_relation(record, by_span)
        relation_counts[relation] += 1
        if parent is not None and record.absolute_slot >= 0 and parent.absolute_slot >= 0:
            absolute_slot_mismatches += record.absolute_slot != parent.absolute_slot
        if record.correlation_id > 0:
            correlation_groups[record.correlation_id].add(record, relation)
    integrity = {
        "profile_dir": profile_dir,
        "schema2_records": len(records),
        "duration_records": sum(record.event_kind == "duration" for record in records),
        "instant_records": sum(record.event_kind == "instant" for record in records),
        "unique_span_ids": len(by_span),
        "duplicate_span_ids": len(duplicate_span_ids),
        "root_records": relation_counts["root"],
        "parented_records": len(records) - relation_counts["root"],
        "temporally_contained_edges": relation_counts["temporally_contained"],
        "causal_noncontained_edges": relation_counts["causal_noncontained"],
        "missing_parent_edges": relation_counts["missing_parent"],
        "correlation_mismatch_edges": relation_counts["correlation_mismatch"],
        "absolute_slot_mismatch_edges": absolute_slot_mismatches,
        "unknown_correlation_records": sum(record.correlation_id == 0 for record in records),
        "unknown_absolute_slot_records": sum(record.absolute_slot < 0 for record in records),
        "max_nesting_depth": max((record.nesting_depth for record in records), default=0),
        "counter_hz": counter_hz,
    }

    for correlation_id, stats in sorted(correlation_groups.items()):
        correlation_writer.writerow(
            {
                "profile_dir": profile_dir,
                "correlation_id": correlation_id,
                "absolute_slot_min": stats.absolute_slot_min,
                "absolute_slot_max": stats.absolute_slot_max,
                "first_tick": stats.first_tick,
                "last_tick": stats.last_tick,
                "elapsed_us": (stats.last_tick - stats.first_tick) * 1e6 / counter_hz
                if counter_hz > 0
                else math.nan,
                "event_count": stats.event_count,
                "duration_count": stats.duration_count,
                "instant_count": stats.instant_count,
                "root_count": stats.root_count,
                "parented_count": stats.parented_count,
                "missing_parent_count": stats.missing_parent_count,
                "max_nesting_depth": stats.max_nesting_depth,
                "thread_count": len(stats.tids),
                "cpu_migrations": stats.cpu_migrations,
                "root_events": ";".join(sorted(stats.root_events)),
            }
        )

    if causal_writer is not None:
        causal_items = sorted(
            causal_groups.items(),
            key=lambda item: (
                item[0][0],
                item[0][1],
                item[0][2] is None,
                item[0][2] or 0,
                item[0][3],
            ),
        )
        for (parent_event, event_name, slot_delta, shape), stats in causal_items:
            distances = sorted(stats.boundary_distance_us)
            causal_writer.writerow(
                {
                    "profile_dir": profile_dir,
                    "parent_event_name": parent_event,
                    "event_name": event_name,
                    "relation": "causal_noncontained",
                    "absolute_slot_delta_from_parent": "" if slot_delta is None else slot_delta,
                    "temporal_shape": shape,
                    "count": len(distances),
                    "correlation_match_count": stats.correlation_match_count,
                    "correlation_unknown_count": stats.correlation_unknown_count,
                    "boundary_distance_min_us": distances[0] if distances else math.nan,
                    "boundary_distance_p50_us": quantile(distances, 0.5),
                    "boundary_distance_p90_us": quantile(distances, 0.9),
                    "boundary_distance_p99_us": quantile(distances, 0.99),
                    "boundary_distance_p99_9_us": quantile(distances, 0.999),
                    "boundary_distance_max_us": distances[-1] if distances else math.nan,
                    "boundary_distance_mean_us": mean(distances),
                }
            )

    return integrity, exclusive_groups


def build_exclusive_summary(groups: dict[tuple[str, str], ExclusiveStats]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (profile_dir, event_name), stats in sorted(groups.items()):
        inclusive = sorted(stats.inclusive_us)
        exclusive = sorted(stats.exclusive_us)
        child_union = sorted(stats.child_union_us)
        rows.append(
            {
                "profile_dir": profile_dir,
                "event_name": event_name,
                "total_count": stats.total_count,
                "exclusive_valid_count": stats.valid_count,
                "exclusive_invalid_count": stats.total_count - stats.valid_count,
                "inclusive_mean_us": mean(inclusive),
                "inclusive_p50_us": quantile(inclusive, 0.5),
                "inclusive_p99_us": quantile(inclusive, 0.99),
                "exclusive_mean_us": mean(exclusive),
                "exclusive_p50_us": quantile(exclusive, 0.5),
                "exclusive_p99_us": quantile(exclusive, 0.99),
                "child_union_mean_us": mean(child_union),
                "parents_with_overlapping_children": stats.overlapping_children_count,
                "noncontained_children": stats.noncontained_children,
                "correlation_mismatch_children": stats.correlation_mismatches,
            }
        )
    return rows


def write_summary(path: Path | None, rows: list[dict[str, object]]) -> ArtifactInfo | None:
    fields = [
        "profile_dir",
        "process_name",
        "schema_version",
        "event_name",
        "event_role",
        "subsystem",
        "event_class",
        "event_kind",
        "detail_level",
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
        "correlated_count",
        "parented_count",
        "absolute_slot_count",
        "cpu_observed_count",
        "cpu_migrations",
        "cpu_migration_rate",
        "drops_total",
        "span_stack_overflows",
        "span_stack_mismatches",
    ]
    if path is None:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        return None
    return write_rows(path, fields, rows)


def write_by_thread(path: Path, rows: list[dict[str, object]]) -> ArtifactInfo:
    fields = [
        "profile_dir",
        "process_name",
        "schema_version",
        "thread_name",
        "event_name",
        "event_kind",
        "detail_level",
        "count",
        "mean_us",
        "p99_us",
        "max_us",
        "cpu_observed_count",
        "cpu_migrations",
        "cpu_migration_rate",
    ]
    return write_rows(path, fields, rows)


def write_deadline_misses(path: Path, rows: list[dict[str, str]]) -> ArtifactInfo:
    return write_rows(path, DEADLINE_MISS_FIELDS, rows)


def write_migrations(path: Path, rows: list[dict[str, str]]) -> ArtifactInfo:
    return write_rows(path, MIGRATION_FIELDS, rows)


def build_rows(
    groups: dict[tuple[str, str], GroupStats],
    metadata_by_dir: dict[str, dict[str, str]],
    drop_diagnostics_by_dir: dict[str, dict[str, int]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (profile_dir, event_name), stats in sorted(groups.items()):
        durations = sorted(stats.durations_us)
        avg = mean(durations)
        quantiles = [("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99), ("p99_9", 0.999)]
        q = {name: quantile(durations, value) for name, value in quantiles}
        diagnostics = drop_diagnostics_by_dir[profile_dir]
        count = len(durations)
        rows.append(
            {
                "profile_dir": profile_dir,
                "process_name": metadata_by_dir.get(profile_dir, {}).get("process_name", "unknown"),
                "schema_version": one_or_mixed(stats.schema_versions, "1"),
                "event_name": event_name,
                "event_role": one_or_mixed(stats.event_roles, "unknown"),
                "subsystem": one_or_mixed(stats.subsystems, "unknown"),
                "event_class": one_or_mixed(stats.event_classes, "unknown"),
                "event_kind": one_or_mixed(stats.event_kinds, "unknown"),
                "detail_level": one_or_mixed(stats.detail_levels, "boundary"),
                "count": count,
                "min_us": durations[0] if durations else math.nan,
                "p50_us": q["p50"],
                "p90_us": q["p90"],
                "p95_us": q["p95"],
                "p99_us": q["p99"],
                "p99_9_us": q["p99_9"],
                "max_us": durations[-1] if durations else math.nan,
                "mean_us": avg,
                "stdev_us": stdev(durations, avg),
                "correlated_count": stats.correlated_count,
                "parented_count": stats.parented_count,
                "absolute_slot_count": stats.absolute_slot_count,
                "cpu_observed_count": stats.cpu_observed_count,
                "cpu_migrations": stats.cpu_migrations,
                "cpu_migration_rate": stats.cpu_migrations / stats.cpu_observed_count
                if stats.cpu_observed_count
                else math.nan,
                "drops_total": diagnostics["dropped_records"],
                "span_stack_overflows": diagnostics["span_stack_overflows"],
                "span_stack_mismatches": diagnostics["span_stack_mismatches"],
            }
        )
    return rows


def build_thread_rows(
    groups: dict[tuple[str, str, str], GroupStats],
    metadata_by_dir: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (profile_dir, thread_name, event_name), stats in sorted(groups.items()):
        durations = sorted(stats.durations_us)
        avg = mean(durations)
        rows.append(
            {
                "profile_dir": profile_dir,
                "process_name": metadata_by_dir.get(profile_dir, {}).get("process_name", "unknown"),
                "schema_version": one_or_mixed(stats.schema_versions, "1"),
                "thread_name": thread_name,
                "event_name": event_name,
                "event_kind": one_or_mixed(stats.event_kinds, "unknown"),
                "detail_level": one_or_mixed(stats.detail_levels, "boundary"),
                "count": len(durations),
                "mean_us": avg,
                "p99_us": quantile(durations, 0.99),
                "max_us": durations[-1] if durations else math.nan,
                "cpu_observed_count": stats.cpu_observed_count,
                "cpu_migrations": stats.cpu_migrations,
                "cpu_migration_rate": stats.cpu_migrations / stats.cpu_observed_count
                if stats.cpu_observed_count
                else math.nan,
            }
        )
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analysis_input_rows(
    profile_dirs: list[Path],
    run_dirs: list[Path],
    metadata_by_dir: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    profile_keys = {str(path) for path in profile_dirs}
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        key = str(run_dir)
        metadata = metadata_by_dir.get(key, {})
        campaign: dict[str, object] = {}
        campaign_path = run_dir / "campaign_run.json"
        if campaign_path.is_file():
            try:
                value = json.loads(campaign_path.read_text(errors="replace"))
                if isinstance(value, dict):
                    campaign = value
            except (OSError, json.JSONDecodeError):
                campaign = {}
        manifest_path = run_dir / "archive_manifest.csv"
        manifest_present = manifest_path.is_file()
        rows.append(
            {
                "input_path": key,
                "input_kind": "profile" if key in profile_keys else "campaign_run",
                "run_id": metadata.get("run_id", str(campaign.get("run_id", run_dir.name))),
                "experiment_id": metadata.get("experiment_id", str(campaign.get("experiment_id", ""))),
                "campaign_id": metadata.get("campaign_id", str(campaign.get("campaign_id", ""))),
                "variant": metadata.get("variant", str(campaign.get("variant", ""))),
                "trial": metadata.get("trial", str(campaign.get("trial", ""))),
                "role": metadata.get("role", str(campaign.get("role", "unknown"))),
                "hostname": metadata.get("hostname", str(campaign.get("hostname", "unknown"))),
                "schema_version": metadata.get("schema_version", ""),
                "archive_manifest_status": "present" if manifest_present else "missing",
                "archive_manifest_path": str(manifest_path) if manifest_present else "",
                "archive_manifest_sha256": sha256_file(manifest_path) if manifest_present else "",
            }
        )
    return rows


def analysis_provenance_rows(
    args: argparse.Namespace,
    invocation_argv: list[str],
) -> list[dict[str, object]]:
    invocation_values = [
        ("argv_json", json.dumps(invocation_argv, ensure_ascii=False)),
        (
            "input_arguments_json",
            json.dumps([str(path) for path in args.profile_dir], ensure_ascii=False),
        ),
        ("event_filter_json", json.dumps(args.event or [], ensure_ascii=False)),
        ("output_profile", args.output_profile),
        ("output_dir_argument", str(args.output_dir) if args.output_dir is not None else ""),
        ("working_directory", str(Path.cwd())),
        ("python_version", sys.version),
        ("python_executable", sys.executable),
        ("python_implementation", sys.implementation.name),
    ]
    rows: list[dict[str, object]] = [
        {
            "record_type": "invocation",
            "name": name,
            "value": value,
            "path": "",
            "sha256": "",
        }
        for name, value in invocation_values
    ]
    module_paths = [
        ("oai_profile_analyze", Path(__file__).resolve()),
        ("oai_profile_clock", Path(__file__).resolve().with_name("oai_profile_clock.py")),
        (
            "oai_profile_deadlines",
            Path(__file__).resolve().with_name("oai_profile_deadlines.py"),
        ),
        ("oai_profile_reports", Path(__file__).resolve().with_name("oai_profile_reports.py")),
        ("oai_profile_archive", Path(__file__).resolve().with_name("oai_profile_archive.py")),
    ]
    for module_name, module_path in module_paths:
        rows.append(
            {
                "record_type": "module",
                "name": module_name,
                "value": "",
                "path": str(module_path),
                "sha256": sha256_file(module_path),
            }
        )
    return rows


def prepare_output_directory(output_dir: Path) -> tuple[Path, Path]:
    final_dir = output_dir.expanduser().resolve()
    if final_dir.exists():
        raise FileExistsError(f"analysis output directory already exists: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{final_dir.name}.partial-", dir=final_dir.parent)
    )
    return final_dir, staging_dir


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_incomplete_marker(
    directory: Path,
    output_profile: str,
    error: BaseException,
    published: bool,
) -> bool:
    if not directory.is_dir():
        return False
    marker = directory / "ANALYSIS_INCOMPLETE.txt"
    try:
        with marker.open("w") as stream:
            stream.write(
                f"publication_state={'published_incomplete' if published else 'unpublished_partial'}\n"
                f"output_profile={output_profile}\n"
                f"error_type={type(error).__name__}\n"
                f"error={error}\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        fsync_directory(directory)
        return True
    except BaseException:
        return False


def write_analysis_manifest(
    output_dir: Path,
    artifacts: dict[str, ArtifactInfo],
    omitted: set[str],
    output_profile: str,
) -> None:
    rows: list[dict[str, object]] = []
    for filename, artifact in sorted(artifacts.items()):
        rows.append(
            {
                "artifact": filename,
                "status": "generated",
                "row_count": artifact.row_count,
                "size_bytes": artifact.size_bytes,
                "sha256": artifact.sha256,
                "output_profile": output_profile,
                "notes": "",
            }
        )
    for filename in sorted(omitted):
        rows.append(
            {
                "artifact": filename,
                "status": "omitted_by_output_profile",
                "row_count": "",
                "size_bytes": "",
                "sha256": "",
                "output_profile": output_profile,
                "notes": "Regenerable detail output intentionally omitted.",
            }
        )
    rows.append(
        {
            "artifact": "analysis_manifest.csv",
            "status": "generated_self_unhashed",
            "row_count": len(rows) + 1,
            "size_bytes": "",
            "sha256": "",
            "output_profile": output_profile,
            "notes": "The manifest excludes its own digest to avoid a self-reference.",
        }
    )
    write_rows(output_dir / "analysis_manifest.csv", ANALYSIS_MANIFEST_FIELDS, rows)


def main() -> int:
    invocation_argv = list(sys.argv)
    parser = argparse.ArgumentParser(description="Summarize OAI profiler process directories or archive roots")
    parser.add_argument(
        "profile_dir",
        nargs="+",
        type=Path,
        help="A process profile directory or a root containing profile directories",
    )
    parser.add_argument("--event", action="append", help="Event name to include; may be given multiple times")
    parser.add_argument("--output-dir", type=Path, help="Write all CSV analysis outputs here")
    parser.add_argument(
        "--output-profile",
        choices=("full", "publication"),
        default="full",
        help="full emits all detail CSVs; publication omits regenerable event/hierarchy detail",
    )
    args = parser.parse_args()
    if args.output_profile != "full" and args.output_dir is None:
        parser.error("--output-profile publication requires --output-dir")
    captured_provenance_rows = analysis_provenance_rows(args, invocation_argv)

    profile_dirs = discover_profile_dirs(args.profile_dir)
    run_dirs = discover_run_dirs(args.profile_dir, profile_dirs)
    if not run_dirs:
        raise FileNotFoundError("no profiler or campaign run directories were found")
    event_filter = set(args.event) if args.event else None
    metadata_by_dir = {str(p): read_metadata(p) for p in profile_dirs}
    settings_by_dir = {str(p): read_settings(p) for p in profile_dirs}
    catalogs_by_dir = {str(p): read_event_catalog(p) for p in profile_dirs}
    drop_diagnostics_by_dir = {str(p): read_drop_diagnostics(p) for p in profile_dirs}
    drops_by_dir = {key: diagnostics["dropped_records"] for key, diagnostics in drop_diagnostics_by_dir.items()}
    host_by_dir = {str(p): summarize_host_metrics(p) for p in profile_dirs}
    run_rows = build_run_inventory(
        profile_dirs,
        metadata_by_dir,
        settings_by_dir,
        drops_by_dir,
        drop_diagnostics_by_dir,
        host_by_dir,
    )
    pair_rows = build_pairs(run_rows)

    final_output_dir: Path | None = None
    output_dir: Path | None = None
    artifacts: dict[str, ArtifactInfo] = {}
    open_writers: list[TrackedCsvWriter] = []
    hierarchy_writer: TrackedCsvWriter | None = None
    anomaly_writer: TrackedCsvWriter | None = None
    correlation_writer: TrackedCsvWriter | None = None
    causal_writer: TrackedCsvWriter | None = None
    deadline_miss_writer: TrackedCsvWriter | None = None
    deadline_check_writer: TrackedCsvWriter | None = None
    migration_writer: TrackedCsvWriter | None = None

    all_summary_rows: list[dict[str, object]] = []
    thread_rows: list[dict[str, object]] = []
    deadline_summary_rows: list[dict[str, object]] = []
    transport_fault_rows: list[dict[str, str]] = []
    hierarchy_integrity_rows: list[dict[str, object]] = []
    exclusive_summary_rows: list[dict[str, object]] = []
    published = False

    def open_output_writer(path: Path, fields: list[str]) -> TrackedCsvWriter:
        writer = TrackedCsvWriter(path, fields)
        try:
            open_writers.append(writer)
        except BaseException:
            if not writer.closed:
                try:
                    writer.close()
                except BaseException:
                    pass
            raise
        return writer

    if args.output_dir is not None:
        final_output_dir, output_dir = prepare_output_directory(args.output_dir)

    try:
        if output_dir is not None:
            if args.output_profile == "full":
                hierarchy_writer = open_output_writer(
                    output_dir / "hierarchy.csv",
                    HIERARCHY_FIELDS,
                )
            anomaly_writer = open_output_writer(
                output_dir / "hierarchy_anomalies.csv",
                HIERARCHY_ANOMALY_FIELDS,
            )
            correlation_writer = open_output_writer(
                output_dir / "correlations.csv",
                CORRELATION_FIELDS,
            )
            deadline_miss_writer = open_output_writer(
                output_dir / "deadline_misses.csv",
                DEADLINE_MISS_FIELDS,
            )
            deadline_check_writer = open_output_writer(
                output_dir / "deadline_checks.csv",
                DEADLINE_CHECK_FIELDS,
            )
            migration_writer = open_output_writer(
                output_dir / "migrations.csv",
                MIGRATION_FIELDS,
            )
            if args.output_profile == "publication":
                causal_writer = open_output_writer(
                    output_dir / "causal_edges_summary.csv",
                    CAUSAL_EDGE_SUMMARY_FIELDS,
                )

        for profile_dir in profile_dirs:
            profile_key = str(profile_dir)
            metadata = metadata_by_dir[profile_key]
            is_nrue = is_nrue_profile(metadata)
            event_groups: dict[tuple[str, str], GroupStats] = defaultdict(GroupStats)
            thread_groups: dict[tuple[str, str, str], GroupStats] = defaultdict(GroupStats)
            hierarchy_records: list[HierarchyRecord] = []
            deadline_event_rows: list[dict[str, str]] = []

            for _, row in iter_event_rows(
                [profile_dir],
                None,
                metadata_by_dir,
                catalogs_by_dir,
            ):
                if output_dir is not None:
                    hierarchy_record = hierarchy_record_from_row(row)
                    if hierarchy_record is not None:
                        hierarchy_records.append(hierarchy_record)
                    if row["subsystem"] == "rf_usrp" and row["event_kind"] == "instant":
                        transport_fault_rows.append({"profile_dir": profile_key, **row})
                event_groups[(profile_key, row["event_name"])].add(row)
                if (
                    output_dir is not None
                    and is_nrue
                    and row["event_name"] in DEADLINE_EVENT_NAMES
                ):
                    deadline_event_rows.append(row)

                if event_filter and row["event_name"] not in event_filter:
                    continue
                if output_dir is not None:
                    thread_groups[(profile_key, row["thread_name"], row["event_name"])].add(row)
                    if is_nrue and row["event_name"] == "UE_TX_DEADLINE_MISS":
                        assert deadline_miss_writer is not None
                        deadline_miss_writer.writerow(
                            {
                                "profile_dir": profile_key,
                                "schema_version": row["schema_version"],
                                "seq": row["seq"],
                                "tid": row["tid"],
                                "thread_name": row["thread_name"],
                                "frame": row["frame"],
                                "slot": row["slot"],
                                "absolute_slot": row["absolute_slot"],
                                "correlation_id": row["correlation_id"],
                                "span_id": row["span_id"],
                                "parent_id": row["parent_id"],
                                "cpu_start": row["cpu_start"],
                                "cpu_end": row["cpu_end"],
                                "cpu_migrated": row["cpu_migrated"],
                                "current_time_us": row["aux0"],
                                "deadline_us": row["aux1"],
                                "miss_us": row["aux2"],
                                "start_tick": row["start_tick"],
                            }
                        )
                    if parse_int(row.get("cpu_migrated")):
                        assert migration_writer is not None
                        migration_writer.writerow(
                            {
                                "profile_dir": profile_key,
                                "schema_version": row["schema_version"],
                                "seq": row["seq"],
                                "tid": row["tid"],
                                "thread_name": row["thread_name"],
                                "event_name": row["event_name"],
                                "event_kind": row["event_kind"],
                                "frame": row["frame"],
                                "slot": row["slot"],
                                "absolute_slot": row["absolute_slot"],
                                "correlation_id": row["correlation_id"],
                                "span_id": row["span_id"],
                                "parent_id": row["parent_id"],
                                "nesting_depth": row["nesting_depth"],
                                "cpu_start": row["cpu_start"],
                                "cpu_end": row["cpu_end"],
                                "start_tick": row["start_tick"],
                                "duration_tick": row["duration_tick"],
                                "duration_us": row["duration_us"],
                            }
                        )

            profile_summary_rows = build_rows(
                event_groups,
                metadata_by_dir,
                drop_diagnostics_by_dir,
            )
            all_summary_rows.extend(profile_summary_rows)
            if output_dir is not None:
                thread_rows.extend(build_thread_rows(thread_groups, metadata_by_dir))
                if is_nrue:
                    assert deadline_check_writer is not None
                    profile_deadlines = build_deadline_reports(
                        {profile_key: deadline_event_rows},
                        metadata_by_dir,
                    )
                    deadline_check_writer.writerows(profile_deadlines.check_rows)
                    deadline_summary_rows.extend(profile_deadlines.summary_rows)
                    del profile_deadlines
                del event_groups
                del thread_groups
                del deadline_event_rows
                row = None
                assert anomaly_writer is not None
                assert correlation_writer is not None
                run_integrity, run_exclusive = build_hierarchy_outputs(
                    profile_key,
                    hierarchy_records,
                    parse_int(metadata.get("counter_hz")),
                    hierarchy_writer,
                    anomaly_writer,
                    correlation_writer,
                    causal_writer,
                )
                hierarchy_integrity_rows.append(run_integrity)
                exclusive_summary_rows.extend(build_exclusive_summary(run_exclusive))
                del hierarchy_records
                del run_integrity
                del run_exclusive
            else:
                del event_groups
                del thread_groups
                del deadline_event_rows
                del hierarchy_records
                row = None
            del profile_summary_rows

        summary_rows = (
            [row for row in all_summary_rows if row["event_name"] in event_filter]
            if event_filter
            else all_summary_rows
        )
        if output_dir is None:
            write_summary(None, summary_rows)
            return 0

        for writer in open_writers:
            if not writer.closed:
                artifact = writer.close()
                artifacts[artifact.artifact] = artifact

        omitted: set[str] = set()
        if args.output_profile == "full":
            artifact = write_event_timeline(
                output_dir / "event_timeline.csv",
                profile_dirs,
                metadata_by_dir,
                catalogs_by_dir,
            )
            artifacts[artifact.artifact] = artifact
        else:
            omitted.update(("event_timeline.csv", "hierarchy.csv"))

        extended_reports = build_extended_reports(
            profile_dirs,
            run_dirs,
            metadata_by_dir,
            settings_by_dir,
            all_summary_rows,
            transport_fault_rows,
        )

        output_artifacts = [
            write_summary(output_dir / "summary.csv", summary_rows),
            write_by_thread(output_dir / "by_thread.csv", thread_rows),
            write_rows(
                output_dir / "deadline_summary.csv",
                DEADLINE_SUMMARY_FIELDS,
                deadline_summary_rows,
            ),
            write_rows(
                output_dir / "exclusive_summary.csv",
                [
                    "profile_dir",
                    "event_name",
                    "total_count",
                    "exclusive_valid_count",
                    "exclusive_invalid_count",
                    "inclusive_mean_us",
                    "inclusive_p50_us",
                    "inclusive_p99_us",
                    "exclusive_mean_us",
                    "exclusive_p50_us",
                    "exclusive_p99_us",
                    "child_union_mean_us",
                    "parents_with_overlapping_children",
                    "noncontained_children",
                    "correlation_mismatch_children",
                ],
                exclusive_summary_rows,
            ),
            write_rows(
                output_dir / "hierarchy_integrity.csv",
                [
                    "profile_dir",
                    "schema2_records",
                    "duration_records",
                    "instant_records",
                    "unique_span_ids",
                    "duplicate_span_ids",
                    "root_records",
                    "parented_records",
                    "temporally_contained_edges",
                    "causal_noncontained_edges",
                    "missing_parent_edges",
                    "correlation_mismatch_edges",
                    "absolute_slot_mismatch_edges",
                    "unknown_correlation_records",
                    "unknown_absolute_slot_records",
                    "max_nesting_depth",
                    "counter_hz",
                ],
                hierarchy_integrity_rows,
            ),
            write_rows(
                output_dir / "runs.csv",
                [
                    "profile_dir",
                    "schema_version",
                    "event_record_size_bytes",
                    "max_nesting_depth",
                    "run_id",
                    "experiment_id",
                    "campaign_id",
                    "variant",
                    "trial",
                    "role",
                    "process_name",
                    "hostname",
                    "start_realtime_ns",
                    "end_realtime_ns",
                    "start_monotonic_raw_ns",
                    "end_monotonic_raw_ns",
                    "duration_s",
                    "duration_clock",
                    "duration_status",
                    "realtime_clock_regressed",
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
                    "span_stack_overflows",
                    "span_stack_mismatches",
                    "host_metric_samples",
                ],
                run_rows,
            ),
            write_rows(
                output_dir / "pairs.csv",
                [
                    "status",
                    "method",
                    "experiment_id",
                    "gnb_run_id",
                    "ue_run_id",
                    "gnb_profile_dir",
                    "ue_profile_dir",
                    "gnb_candidate_count",
                    "ue_candidate_count",
                    "clock_domain",
                    "clock_status",
                    "start_delta_ms",
                    "overlap_s",
                    "notes",
                ],
                pair_rows,
            ),
            write_rows(
                output_dir / "host_summary.csv",
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
            ),
            write_rows(
                output_dir / "analysis_inputs.csv",
                ANALYSIS_INPUT_FIELDS,
                analysis_input_rows(profile_dirs, run_dirs, metadata_by_dir),
            ),
            write_rows(
                output_dir / "analysis_provenance.csv",
                ANALYSIS_PROVENANCE_FIELDS,
                captured_provenance_rows,
            ),
        ]
        for artifact in output_artifacts:
            if artifact is not None:
                artifacts[artifact.artifact] = artifact
        for filename, report in extended_reports.items():
            artifact = write_rows(output_dir / filename, report.fields, report.rows)
            artifacts[artifact.artifact] = artifact

        write_analysis_manifest(output_dir, artifacts, omitted, args.output_profile)
        assert final_output_dir is not None
        fsync_directory(output_dir)
        os.replace(output_dir, final_output_dir)
        published = True
        fsync_directory(final_output_dir.parent)
    except BaseException as error:
        for writer in open_writers:
            if not writer.closed:
                try:
                    writer.close()
                except BaseException:
                    pass
        marker_dir = final_output_dir if published else output_dir
        marked = False
        if marker_dir is not None:
            marked = write_incomplete_marker(
                marker_dir,
                args.output_profile,
                error,
                published,
            )
        if (
            published
            and not marked
            and final_output_dir is not None
            and output_dir is not None
        ):
            try:
                os.replace(final_output_dir, output_dir)
                published = False
                write_incomplete_marker(
                    output_dir,
                    args.output_profile,
                    error,
                    published,
                )
            except OSError:
                pass
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
