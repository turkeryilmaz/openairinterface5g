#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0

import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from array import array
from pathlib import Path
from unittest import mock


ANALYZER = Path(__file__).resolve().parents[1] / "oai_profile_analyze.py"

ARCHIVE_TOOL = Path(__file__).resolve().parents[1] / "oai_profile_archive.py"
sys.path.insert(0, str(ANALYZER.parent))

from oai_profile_deadlines import build_deadline_reports, round_div_signed  # noqa: E402
import oai_profile_analyze as analyze_module  # noqa: E402
from oai_profile_analyze import (  # noqa: E402
    CsvDiagnostics,
    DropDiagnostics,
    ExclusiveStats,
    GroupStats,
    build_exclusive_summary,
    build_pairs,
    build_rows,
    build_run_inventory,
    iter_event_rows,
    profile_coverage_status,
    read_drop_diagnostics,
    read_event_catalog,
)
from oai_profile_reports import (  # noqa: E402
    campaign_completeness_report,
    external_sources_report,
    host_metrics_report,
    matches_native_decimal,
    observer_effect_report,
    perf_stat_report,
    pmu_reports,
    pmu_sample_row_is_valid,
    primitive_overhead_report,
    source_alignment_status,
)
HOST_METRIC_FIELDS = (
    "realtime_ns,monotonic_raw_ns,tick,writer_cpu,"
    "thermal_zone0_millicelsius,thermal_max_millicelsius,thermal_samples,"
    "rpi_throttled_valid,rpi_throttled_raw,"
    "cpu_frequency_samples,cpu_frequency_min_khz,cpu_frequency_avg_khz,cpu_frequency_max_khz,"
    "cpu_busy_percent,load1,load5,load15,mem_available_kb,swap_free_kb,"
    "process_rss_kb,process_maxrss_kb,process_user_us,process_system_us,"
    "voluntary_context_switches,involuntary_context_switches,minor_faults,major_faults,"
    "block_input_ops,block_output_ops,end_monotonic_raw_ns,end_tick,writer_cpu_end,"
    "writer_cpu_migrated,acquisition_duration_monotonic_raw_ns,acquisition_duration_tick,"
    "acquisition_duration_us,status,getloadavg_count,getrusage_status,error_mask"
).split(",")
PMU_AVAILABILITY_HEADER = (
    "schema_version,run_id,experiment_id,campaign_id,role,hostname,thread_index,"
    "tid,thread_name,event_id,event_name,domain,requested,available,status,error_code"
)
PMU_SAMPLE_HEADER = (
    "schema_version,sample_id,realtime_ns,monotonic_raw_ns,tick,run_id,"
    "experiment_id,campaign_id,variant,trial,role,hostname,thread_index,tid,"
    "thread_name,target_cpu,event_id,event_name,domain,unit,raw_value,delta_raw,"
    "time_enabled_ns,time_running_ns,delta_enabled_ns,delta_running_ns,"
    "scaled_value,delta_scaled,multiplex_ratio,interval_ns,delta_valid,"
    "scaling_valid,status,error_code"
)

def write_text(path: Path, text: str) -> None:
    path.write_text(text)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))

def write_csv_rows(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)




def write_campaign_run(
    run_dir: Path,
    role: str,
    variant: str,
    duration_s: int,
    profile_enabled: bool,
    *,
    sidecar_tool: str = "none",
    sidecar_status: str = "not_requested",
    sidecar_artifact: str = "",
) -> None:
    run_dir.mkdir(parents=True)
    start_realtime_ns = 1_000_000_000
    start_monotonic_ns = 500_000_000
    experiment_id = f"scientific-baseline-{variant}-t001"
    value = {
        "schema_version": 1,
        "runner_version": 1,
        "campaign_id": "scientific-baseline",
        "experiment_id": experiment_id,
        "case": "band28-25prb",
        "variant": variant,
        "trial": 1,
        "role": role,
        "host": "local",
        "hostname": f"{role.lower()}-host",
        "profile_enabled": profile_enabled,
        "pmu_mode": "software" if profile_enabled else "off",
        "status": "finished",
        "return_code": 0,
        "stop_reason": "duration_elapsed",
        "start_realtime_ns": start_realtime_ns,
        "end_realtime_ns": start_realtime_ns + duration_s * 1_000_000_000,
        "start_monotonic_raw_ns": start_monotonic_ns,
        "end_monotonic_raw_ns": start_monotonic_ns + duration_s * 1_000_000_000,
        "anchor_clock_scope": "measured_host",
        "launch_index": 0 if role == "gNB" else 1,
        "sidecar_tool": sidecar_tool,
        "sidecar_status": sidecar_status,
        "sidecar_artifact": sidecar_artifact,
        "archive_status": "finalization_pending",
        "command": ["./nr-softmodem" if role == "gNB" else "./nr-uesoftmodem"],
        "environment": {"OAI_PROFILE": "1" if profile_enabled else "0"},
        "notes": [],
    }
    write_text(run_dir / "campaign_run.json", json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_profile_metadata(run_dir: Path, role: str, variant: str, duration_s: int) -> None:
    process = "nr-softmodem" if role == "gNB" else "nr-uesoftmodem"
    duration_ns = duration_s * 1_000_000_000
    write_text(
        run_dir / "metadata.txt",
        "schema_version=2\nevent_record_size_bytes=120\nmax_nesting_depth=64\ncounter_hz=10000000\n"
        f"process_name={process}\nrole={role}\nrun_id={variant}-{role}\n"
        f"experiment_id=scientific-baseline-{variant}-t001\ncampaign_id=scientific-baseline\n"
        f"variant={variant}\ntrial=1\nhostname={role.lower()}-host\npmu_mode=software\n"
        f"start_realtime_ns=1000000000\nend_realtime_ns={1_000_000_000 + duration_ns}\n"
        f"duration_realtime_ns={duration_ns}\n"
        f"start_monotonic_raw_ns=500000000\nend_monotonic_raw_ns={500_000_000 + duration_ns}\n"
        f"duration_monotonic_raw_ns={duration_ns}\n"
        "duration_clock=CLOCK_MONOTONIC_RAW\n"
        "realtime_clock_regressed=0\nmonotonic_raw_clock_regressed=0\n"
        "clean_shutdown=1\n",
    )


def finalize_archive(run_dir: Path) -> None:
    subprocess.run([sys.executable, str(ARCHIVE_TOOL), "finalize", str(run_dir)], check=True, capture_output=True, text=True)

class AnalyzerSchemaCompatibilityTest(unittest.TestCase):
    def test_drop_diagnostics_preserve_unavailable_and_legacy_states(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-drop-state-") as temporary:
            root = Path(temporary)

            missing = root / "missing"
            missing.mkdir()
            self.assertEqual(read_drop_diagnostics(missing).status, "missing")
            self.assertIsNone(read_drop_diagnostics(missing).dropped_records)

            zero_byte = root / "zero-byte"
            zero_byte.mkdir()
            (zero_byte / "drops.csv").write_bytes(b"")
            zero_byte_diagnostics = read_drop_diagnostics(zero_byte)
            self.assertEqual(zero_byte_diagnostics.status, "zero_byte")
            self.assertIsNone(zero_byte_diagnostics.span_stack_overflows)

            header_only = root / "header-only"
            header_only.mkdir()
            write_text(
                header_only / "drops.csv",
                "thread_index,tid,thread_name,dropped_records,span_stack_overflows,"
                "span_stack_mismatches,counter_regressions\n",
            )
            header_only_diagnostics = read_drop_diagnostics(header_only)
            self.assertEqual(header_only_diagnostics.status, "header_only")
            self.assertIsNone(header_only_diagnostics.dropped_records)

            recorded = root / "recorded"
            recorded.mkdir()
            write_text(
                recorded / "drops.csv",
                "thread_index,tid,thread_name,dropped_records,span_stack_overflows,"
                "span_stack_mismatches,counter_regressions\n"
                "0,100,worker-a,0,0,0,0\n"
                "1,101,worker-b,2,3,4,5\n",
            )
            recorded_diagnostics = read_drop_diagnostics(recorded)
            self.assertEqual(recorded_diagnostics.status, "recorded")
            self.assertEqual(recorded_diagnostics.row_count, 2)
            self.assertEqual(recorded_diagnostics.dropped_records, 2)
            self.assertEqual(recorded_diagnostics.span_stack_overflows, 3)
            self.assertEqual(recorded_diagnostics.span_stack_mismatches, 4)
            self.assertEqual(recorded_diagnostics.counter_regressions, 5)

            legacy = root / "legacy"
            legacy.mkdir()
            write_text(
                legacy / "drops.csv",
                "thread_index,tid,thread_name,dropped_records\n"
                "0,100,worker-a,7\n",
            )
            legacy_diagnostics = read_drop_diagnostics(legacy)
            self.assertEqual(legacy_diagnostics.status, "legacy_partial")
            self.assertEqual(legacy_diagnostics.dropped_records, 7)
            self.assertIsNone(legacy_diagnostics.span_stack_overflows)
            self.assertIsNone(legacy_diagnostics.counter_regressions)

            malformed_values = (
                "not-an-integer",
                "-1",
                "",
            )
            for index, value in enumerate(malformed_values):
                malformed = root / f"malformed-{index}"
                malformed.mkdir()
                write_text(
                    malformed / "drops.csv",
                    "thread_index,tid,thread_name,dropped_records,span_stack_overflows,"
                    "span_stack_mismatches,counter_regressions\n"
                    f"0,100,worker-a,{value},0,0,0\n",
                )
                diagnostics = read_drop_diagnostics(malformed)
                self.assertEqual(diagnostics.status, "malformed")
                self.assertIsNone(diagnostics.dropped_records)

            malformed_header = root / "malformed-header"
            malformed_header.mkdir()
            write_text(malformed_header / "drops.csv", '"dropped_records\n')
            self.assertEqual(
                read_drop_diagnostics(malformed_header).status,
                "malformed",
            )

            excess_column = root / "excess-column"
            excess_column.mkdir()
            write_text(
                excess_column / "drops.csv",
                "thread_index,tid,thread_name,dropped_records,span_stack_overflows,"
                "span_stack_mismatches,counter_regressions\n"
                "0,100,worker-a,0,0,0,0,unexpected\n",
            )
            self.assertEqual(
                read_drop_diagnostics(excess_column).status,
                "malformed",
            )

            missing_identity = root / "missing-identity"
            missing_identity.mkdir()
            write_text(
                missing_identity / "drops.csv",
                "dropped_records,span_stack_overflows,span_stack_mismatches,"
                "counter_regressions\n"
                "0,0,0,0\n",
            )
            self.assertEqual(
                read_drop_diagnostics(missing_identity).status,
                "malformed",
            )

            blank_identity = root / "blank-identity"
            blank_identity.mkdir()
            write_text(
                blank_identity / "drops.csv",
                "thread_index,tid,thread_name,dropped_records,span_stack_overflows,"
                "span_stack_mismatches,counter_regressions\n"
                "0,100,,0,0,0,0\n",
            )
            self.assertEqual(
                read_drop_diagnostics(blank_identity).status,
                "malformed",
            )

            duplicate_identity = root / "duplicate-identity"
            duplicate_identity.mkdir()
            write_text(
                duplicate_identity / "drops.csv",
                "thread_index,tid,thread_name,dropped_records,span_stack_overflows,"
                "span_stack_mismatches,counter_regressions\n"
                "0,100,worker-a,0,0,0,0\n"
                "0,101,worker-b,0,0,0,0\n",
            )
            self.assertEqual(
                read_drop_diagnostics(duplicate_identity).status,
                "malformed",
            )

            reused_tid = root / "reused-tid"
            reused_tid.mkdir()
            write_text(
                reused_tid / "drops.csv",
                "thread_index,tid,thread_name,dropped_records,span_stack_overflows,"
                "span_stack_mismatches,counter_regressions\n"
                "0,100,worker-a,0,0,0,0\n"
                "1,100,worker-b,0,0,0,0\n",
            )
            reused_tid_diagnostics = read_drop_diagnostics(reused_tid)
            self.assertEqual(reused_tid_diagnostics.status, "recorded")
            self.assertEqual(reused_tid_diagnostics.row_count, 2)

            zero_diagnostics = DropDiagnostics(
                status="recorded",
                row_count=1,
                dropped_records=0,
                span_stack_overflows=0,
                span_stack_mismatches=0,
                counter_regressions=0,
            )
            recorded_csv = CsvDiagnostics(status="recorded", row_count=1)
            complete_metadata = {
                "schema_version": "2",
                "counter_hz": "10000000",
                "clean_shutdown": "1",
                "start_realtime_ns": "100",
                "end_realtime_ns": "200",
                "start_monotonic_raw_ns": "50",
                "end_monotonic_raw_ns": "150",
                "duration_realtime_ns": "100",
                "duration_monotonic_raw_ns": "100",
                "duration_clock": "CLOCK_MONOTONIC_RAW",
                "realtime_clock_regressed": "0",
                "monotonic_raw_clock_regressed": "0",
            }
            self.assertEqual(
                profile_coverage_status(
                    complete_metadata,
                    zero_diagnostics,
                    recorded_csv,
                    recorded_csv,
                ),
                "complete",
            )
            for invalid_metadata in (
                {**complete_metadata, "end_realtime_ns": "99"},
                {**complete_metadata, "end_monotonic_raw_ns": "49"},
                {**complete_metadata, "realtime_clock_regressed": "1"},
                {**complete_metadata, "monotonic_raw_clock_regressed": "1"},
            ):
                self.assertEqual(
                    profile_coverage_status(
                        invalid_metadata,
                        zero_diagnostics,
                        recorded_csv,
                        recorded_csv,
                    ),
                    "lifecycle_clock_invalid",
                )
            for invalid_counter_metadata in (
                {key: value for key, value in complete_metadata.items() if key != "counter_hz"},
                {**complete_metadata, "counter_hz": "0"},
                {**complete_metadata, "counter_hz": "not-an-integer"},
            ):
                self.assertEqual(
                    profile_coverage_status(
                        invalid_counter_metadata,
                        zero_diagnostics,
                        recorded_csv,
                        recorded_csv,
                    ),
                    "counter_hz_invalid",
                )
            incomplete_metadata = dict(complete_metadata)
            del incomplete_metadata["realtime_clock_regressed"]
            self.assertEqual(
                profile_coverage_status(
                    incomplete_metadata,
                    zero_diagnostics,
                    recorded_csv,
                    recorded_csv,
                ),
                "lifecycle_metadata_incomplete",
            )

    def test_event_catalog_and_drop_thread_coverage_are_strict(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-event-state-") as temporary:
            profile_dir = Path(temporary)
            profile_key = str(profile_dir)
            write_text(
                profile_dir / "metadata.txt",
                "schema_version=2\ncounter_hz=10000000\n"
                "process_name=nr-uesoftmodem\nrole=nrUE\n"
                "start_realtime_ns=100\nend_realtime_ns=200\n"
                "duration_realtime_ns=100\n"
                "start_monotonic_raw_ns=50\nend_monotonic_raw_ns=150\n"
                "duration_monotonic_raw_ns=100\n"
                "duration_clock=CLOCK_MONOTONIC_RAW\n"
                "realtime_clock_regressed=0\n"
                "monotonic_raw_clock_regressed=0\nclean_shutdown=1\n",
            )
            event_header = (
                "schema_version,seq,tid,thread_name,event_id,event_name,event_kind,"
                "nesting_depth,frame,slot,absolute_slot,correlation_id,span_id,"
                "parent_id,cpu_start,cpu_end,cpu_migrated,flags,aux0,aux1,aux2,"
                "aux3,start_tick,duration_tick,duration_us\n"
            )
            event_rows = (
                "2,0,100,producer-a,1,EVENT_A,duration,0,1,1,21,1,1,0,0,0,0,"
                "0,0,0,0,0,100,10,1.000\n"
                "2,1,101,producer-b,2,EVENT_B,instant,0,1,2,22,2,2,0,1,1,0,"
                "0,0,0,0,0,120,0,0.000\n"
                "2,2,100,producer-a,3,PROFILER_PRIMITIVE_CALIBRATION,instant,0,"
                "-1,-1,-1,0,3,0,0,0,0,0,0,0,0,0,130,0,0.000\n"
            )
            write_text(profile_dir / "events.csv", event_header + event_rows)
            catalog_header = (
                "schema_version,event_id,event_name,role,subsystem,event_class,"
                "default_kind,detail_level,aux0_name,aux0_unit,aux1_name,"
                "aux1_unit,aux2_name,aux2_unit,aux3_name,aux3_unit,flags_name\n"
            )
            catalog_rows = (
                "2,1,EVENT_A,nrUE,test,duration,duration,stage,,,,,,,,,\n"
                "2,2,EVENT_B,nrUE,test,instant,instant,stage,,,,,,,,,\n"
                "2,3,PROFILER_PRIMITIVE_CALIBRATION,nrUE,profiler,calibration,"
                "duration,primitive,,,,,,,,,\n"
            )
            write_text(
                profile_dir / "event_catalog.csv",
                catalog_header + catalog_rows,
            )
            write_text(
                profile_dir / "drops.csv",
                "thread_index,tid,thread_name,dropped_records,"
                "span_stack_overflows,span_stack_mismatches,counter_regressions\n"
                "0,100,producer-a,0,0,0,0\n",
            )

            metadata_by_dir = {
                profile_key: analyze_module.read_metadata(profile_dir)
            }
            catalog, catalog_diagnostics = read_event_catalog(profile_dir, "2")
            event_diagnostics_by_dir: dict[str, CsvDiagnostics] = {}
            parsed = list(
                iter_event_rows(
                    [profile_dir],
                    None,
                    metadata_by_dir,
                    {profile_key: catalog},
                    event_diagnostics_by_dir,
                )
            )
            self.assertEqual(len(parsed), 3)
            event_diagnostics = event_diagnostics_by_dir[profile_key]
            self.assertEqual(event_diagnostics.status, "recorded")
            self.assertEqual(event_diagnostics.row_count, 3)
            self.assertEqual(
                event_diagnostics.producer_threads,
                frozenset({(100, "producer-a"), (101, "producer-b")}),
            )
            drop_diagnostics = read_drop_diagnostics(profile_dir)
            self.assertEqual(
                profile_coverage_status(
                    metadata_by_dir[profile_key],
                    drop_diagnostics,
                    event_diagnostics,
                    catalog_diagnostics,
                ),
                "drop_diagnostics_missing_event_threads",
            )

            write_text(
                profile_dir / "drops.csv",
                "thread_index,tid,thread_name,dropped_records,"
                "span_stack_overflows,span_stack_mismatches,counter_regressions\n"
                "0,100,producer-a,0,0,0,0\n"
                "1,101,producer-b,0,0,0,0\n",
            )
            self.assertEqual(
                profile_coverage_status(
                    metadata_by_dir[profile_key],
                    read_drop_diagnostics(profile_dir),
                    event_diagnostics,
                    catalog_diagnostics,
                ),
                "complete",
            )

            invalid_native_rows = (
                (
                    "instant-nonzero-duration",
                    "2,0,101,producer-b,2,EVENT_B,instant,0,1,2,22,2,2,0,1,1,0,"
                    "0,0,0,0,0,120,1,0.100\n",
                ),
                (
                    "instant-cpu-mismatch",
                    "2,0,101,producer-b,2,EVENT_B,instant,0,1,2,22,2,2,0,1,2,1,"
                    "0,0,0,0,0,120,0,0.000\n",
                ),
                (
                    "self-parent",
                    "2,0,100,producer-a,1,EVENT_A,duration,0,1,1,21,1,1,1,0,0,0,"
                    "0,0,0,0,0,100,10,1.000\n",
                ),
                (
                    "tick-time-mismatch",
                    "2,0,100,producer-a,1,EVENT_A,duration,0,1,1,21,1,1,0,0,0,0,"
                    "0,0,0,0,0,100,10,2.000\n",
                ),
            )
            for name, invalid_row in invalid_native_rows:
                with self.subTest(name=name):
                    write_text(
                        profile_dir / "events.csv",
                        event_header + invalid_row,
                    )
                    event_diagnostics_by_dir.clear()
                    self.assertEqual(
                        list(
                            iter_event_rows(
                                [profile_dir],
                                None,
                                metadata_by_dir,
                                {profile_key: catalog},
                                event_diagnostics_by_dir,
                            )
                        ),
                        [],
                    )
                    self.assertEqual(
                        event_diagnostics_by_dir[profile_key].status,
                        "malformed",
                    )

            write_text(
                profile_dir / "events.csv",
                event_header
                + event_rows.splitlines(keepends=True)[0]
                + "2,1,101,producer-b\n",
            )
            event_diagnostics_by_dir.clear()
            self.assertEqual(
                len(
                    list(
                        iter_event_rows(
                            [profile_dir],
                            None,
                            metadata_by_dir,
                            {profile_key: catalog},
                            event_diagnostics_by_dir,
                        )
                    )
                ),
                1,
            )
            self.assertEqual(
                event_diagnostics_by_dir[profile_key],
                CsvDiagnostics(
                    status="malformed",
                    row_count=1,
                    producer_threads=frozenset({(100, "producer-a")}),
                ),
            )
            self.assertEqual(
                profile_coverage_status(
                    metadata_by_dir[profile_key],
                    read_drop_diagnostics(profile_dir),
                    event_diagnostics_by_dir[profile_key],
                    catalog_diagnostics,
                ),
                "event_stream_malformed",
            )

            write_text(profile_dir / "events.csv", event_header)
            event_diagnostics_by_dir.clear()
            self.assertEqual(
                list(
                    iter_event_rows(
                        [profile_dir],
                        None,
                        metadata_by_dir,
                        {profile_key: catalog},
                        event_diagnostics_by_dir,
                    )
                ),
                [],
            )
            self.assertEqual(
                event_diagnostics_by_dir[profile_key].status,
                "header_only",
            )
            (profile_dir / "events.csv").write_bytes(b"")
            event_diagnostics_by_dir.clear()
            list(
                iter_event_rows(
                    [profile_dir],
                    None,
                    metadata_by_dir,
                    {profile_key: catalog},
                    event_diagnostics_by_dir,
                )
            )
            self.assertEqual(
                event_diagnostics_by_dir[profile_key].status,
                "zero_byte",
            )
            (profile_dir / "events.csv").unlink()
            self.assertEqual(
                analyze_module.discover_profile_dirs([profile_dir]),
                [profile_dir],
            )
            event_diagnostics_by_dir.clear()
            list(
                iter_event_rows(
                    [profile_dir],
                    None,
                    metadata_by_dir,
                    {profile_key: catalog},
                    event_diagnostics_by_dir,
                )
            )
            self.assertEqual(
                event_diagnostics_by_dir[profile_key].status,
                "missing",
            )

            (profile_dir / "event_catalog.csv").unlink()
            self.assertEqual(
                read_event_catalog(profile_dir, "2")[1].status,
                "missing",
            )
            (profile_dir / "event_catalog.csv").write_bytes(b"")
            self.assertEqual(
                read_event_catalog(profile_dir, "2")[1].status,
                "zero_byte",
            )
            write_text(profile_dir / "event_catalog.csv", '"schema_version\n')
            self.assertEqual(
                read_event_catalog(profile_dir, "2")[1].status,
                "malformed",
            )

    def test_partial_schema2_clock_bounds_remain_explicit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-clock-scope-") as temporary:
            profile_dir = Path(temporary)
            write_text(
                profile_dir / "sync.csv",
                "realtime_ns,monotonic_raw_ns,tick,status\n"
                "100,50,1,ok\n"
                "999,999,9,clock_error\n"
                "200,150,2,ok\n",
            )
            self.assertEqual(
                analyze_module.read_sync_bounds(profile_dir),
                (100, 200, 50, 150),
            )
            metadata = {
                "schema_version": "2",
                "counter_hz": "10000000",
                "clean_shutdown": "0",
                "start_realtime_ns": "100",
                "end_realtime_ns": "200",
                "end_monotonic_raw_ns": "150",
                "process_name": "nr-uesoftmodem",
                "role": "nrUE",
            }
            diagnostics = DropDiagnostics(
                status="recorded",
                row_count=1,
                dropped_records=0,
                span_stack_overflows=0,
                span_stack_mismatches=0,
                counter_regressions=0,
            )
            profile_key = str(profile_dir)
            event_diagnostics = CsvDiagnostics(status="recorded", row_count=1)
            catalog_diagnostics = CsvDiagnostics(status="recorded", row_count=1)
            rows = build_run_inventory(
                [profile_dir],
                {profile_key: metadata},
                {profile_key: {}},
                {profile_key: diagnostics},
                {profile_key: event_diagnostics},
                {profile_key: catalog_diagnostics},
                {
                    profile_key: profile_coverage_status(
                        metadata,
                        diagnostics,
                        event_diagnostics,
                        catalog_diagnostics,
                    )
                },
                {profile_key: {"samples": 0}},
            )
            self.assertEqual(rows[0]["duration_scope"], "mixed_metadata_sync_bounds")
            self.assertEqual(
                rows[0]["duration_status"],
                "valid_mixed_metadata_sync_bounds",
            )

            invalid_footer = {
                **metadata,
                "clean_shutdown": "1",
                "start_realtime_ns": "100",
                "end_realtime_ns": "200",
                "duration_realtime_ns": "100",
                "start_monotonic_raw_ns": "50",
                "end_monotonic_raw_ns": "150",
                "duration_monotonic_raw_ns": "99",
                "duration_clock": "CLOCK_MONOTONIC_RAW",
                "realtime_clock_regressed": "0",
                "monotonic_raw_clock_regressed": "0",
            }
            invalid_rows = build_run_inventory(
                [profile_dir],
                {profile_key: invalid_footer},
                {profile_key: {}},
                {profile_key: diagnostics},
                {profile_key: event_diagnostics},
                {profile_key: catalog_diagnostics},
                {
                    profile_key: profile_coverage_status(
                        invalid_footer,
                        diagnostics,
                        event_diagnostics,
                        catalog_diagnostics,
                    )
                },
                {profile_key: {"samples": 0}},
            )
            self.assertEqual(
                invalid_rows[0]["duration_scope"],
                "invalid_lifecycle_metadata",
            )
            self.assertEqual(
                invalid_rows[0]["duration_status"],
                "lifecycle_clock_invalid",
            )
            self.assertTrue(math.isnan(float(invalid_rows[0]["duration_s"])))
            self.assertEqual(invalid_rows[0]["realtime_clock_regressed"], "")

    def test_crash_prefix_does_not_fabricate_integrity_or_pmu_evidence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-crash-prefix-") as temporary:
            root = Path(temporary)
            run = root / "synthetic-crash-nrUE"
            output = root / "analysis"
            run.mkdir()

            write_text(
                run / "metadata.txt",
                "schema_version=2\n"
                "event_record_size_bytes=120\n"
                "counter_hz=54000000\n"
                "process_name=nr-uesoftmodem\n"
                "role=nrUE\n"
                "hostname=synthetic-host\n"
                "run_id=synthetic-crash\n"
                "experiment_id=synthetic-crash-t001\n"
                "campaign_id=synthetic-campaign\n"
                "variant=in-process\n"
                "trial=1\n"
                "pmu_mode=off\n"
                "start_realtime_ns=1000000000\n"
                "start_monotonic_raw_ns=500000000\n",
            )
            write_text(
                run / "events.csv",
                "schema_version,seq,tid,thread_name,event_id,event_name,event_kind,nesting_depth,"
                "frame,slot,absolute_slot,correlation_id,span_id,parent_id,cpu_start,cpu_end,"
                "cpu_migrated,flags,aux0,aux1,aux2,aux3,start_tick,duration_tick,duration_us\n"
                "2,1,20,ue-test,107,USRP_RX_RECV,duration,0,0,0,0,1,1,0,2,2,0,0,"
                "512,512,1,0,54000000,54000,1000.000\n",
            )
            write_text(
                run / "event_catalog.csv",
                "schema_version,event_id,event_name,role,subsystem,event_class,default_kind,"
                "detail_level,aux0_name,aux0_unit,aux1_name,aux1_unit,aux2_name,aux2_unit,"
                "aux3_name,aux3_unit,flags_name\n"
                "2,107,USRP_RX_RECV,nrUE,rf_usrp,io,duration,transport,requested_samples,"
                "sample,returned_samples,sample,channel_count,count,error_code,errno,io_status\n",
            )
            write_text(
                run / "sync.csv",
                "realtime_ns,monotonic_raw_ns,tick\n"
                "1000000000,500000000,54000000\n"
                "2000000000,1500000000,108000000\n",
            )
            write_text(run / "settings.csv", "key,value\nprofile.pmu_mode,off\n")
            (run / "drops.csv").write_bytes(b"")
            (run / "pmu_availability.csv").write_bytes(b"")
            (run / "pmu_samples.csv").write_bytes(b"")
            (run / "pmu_read_overhead.csv").write_bytes(b"")
            write_csv_rows(
                run / "system_read_overhead.csv",
                (
                    "schema_version,sample_id,realtime_ns,monotonic_raw_ns,run_id,"
                    "experiment_id,campaign_id,variant,trial,role,hostname,source,"
                    "duration_tick,duration_us,rows,status,error_code"
                ).split(","),
                [
                    {
                        "schema_version": 2,
                        "sample_id": 1,
                        "realtime_ns": 1_500_000_000,
                        "monotonic_raw_ns": 1_000_000_000,
                        "run_id": "synthetic-crash",
                        "experiment_id": "synthetic-crash-t001",
                        "campaign_id": "synthetic-campaign",
                        "variant": "in-process",
                        "trial": 1,
                        "role": "nrUE",
                        "hostname": "synthetic-host",
                        "source": "host_metrics",
                        "duration_tick": 54,
                        "duration_us": 1.0,
                        "rows": 1,
                        "status": "ok",
                        "error_code": 0,
                    }
                ],
            )
            write_text(
                run / "campaign_run.json",
                json.dumps(
                    {
                        "schema_version": 1,
                        "runner_version": 1,
                        "run_id": "synthetic-crash",
                        "experiment_id": "synthetic-crash-t001",
                        "campaign_id": "synthetic-campaign",
                        "case": "synthetic-case",
                        "variant": "in-process",
                        "trial": 1,
                        "role": "nrUE",
                        "hostname": "synthetic-host",
                        "profile_enabled": True,
                        "pmu_mode": "off",
                        "status": "exited_nonzero",
                        "return_code": 139,
                        "stop_reason": "process_exit",
                        "start_realtime_ns": 1000000000,
                        "end_realtime_ns": 2000000000,
                        "start_monotonic_raw_ns": 500000000,
                        "end_monotonic_raw_ns": 1500000000,
                        "workload_status": "not_configured",
                        "network_cleanup_status": "not_configured",
                    }
                ),
            )

            finalize_archive(run)
            subprocess.run(
                [sys.executable, str(ANALYZER), str(run), "--output-dir", str(output)],
                check=True,
            )

            run_row = read_rows(output / "runs.csv")[0]
            self.assertEqual(run_row["clean_shutdown"], "unknown")
            self.assertEqual(run_row["lifecycle_status"], "unknown")
            self.assertEqual(run_row["duration_scope"], "sync_prefix")
            self.assertEqual(run_row["duration_status"], "valid_sync_prefix")
            self.assertEqual(run_row["drops_total"], "")
            self.assertEqual(run_row["span_stack_overflows"], "")
            self.assertEqual(run_row["span_stack_mismatches"], "")
            self.assertEqual(run_row["counter_regressions"], "")
            self.assertEqual(run_row["drop_diagnostics_status"], "zero_byte")
            self.assertEqual(
                run_row["profile_coverage_status"],
                "lifecycle_unknown;drop_diagnostics_zero_byte",
            )

            summary_row = next(
                row
                for row in read_rows(output / "summary.csv")
                if row["event_name"] == "USRP_RX_RECV"
            )
            self.assertEqual(summary_row["drops_total"], "")
            self.assertEqual(summary_row["drop_diagnostics_status"], "zero_byte")
            self.assertEqual(
                summary_row["profile_coverage_status"],
                "lifecycle_unknown;drop_diagnostics_zero_byte",
            )

            transport_row = next(
                row
                for row in read_rows(output / "transport_summary.csv")
                if row["event_name"] == "USRP_RX_RECV"
            )
            self.assertEqual(transport_row["drops_total"], "")
            self.assertEqual(transport_row["counter_regressions"], "")
            self.assertEqual(transport_row["drop_diagnostics_status"], "zero_byte")

            hierarchy_row = read_rows(output / "hierarchy_integrity.csv")[0]
            self.assertEqual(hierarchy_row["clean_shutdown"], "unknown")
            self.assertEqual(hierarchy_row["drops_total"], "")
            self.assertEqual(
                hierarchy_row["profile_coverage_status"],
                "lifecycle_unknown;drop_diagnostics_zero_byte",
            )

            availability = read_rows(output / "pmu_availability_summary.csv")[0]
            self.assertEqual(availability["stream_status"], "zero_byte")
            self.assertEqual(availability["requested"], "")
            self.assertEqual(availability["available"], "")
            self.assertEqual(availability["status"], "stream_zero_byte")

            quality = read_rows(output / "pmu_quality.csv")[0]
            self.assertEqual(quality["status"], "stream_zero_byte")
            self.assertEqual(quality["samples_total"], "")
            self.assertEqual(quality["read_error_count"], "")

            collection_overhead = read_rows(
                output / "collection_overhead_summary.csv"
            )
            pmu_overhead = next(
                row for row in collection_overhead if row["source"] == "pmu_read"
            )
            self.assertEqual(pmu_overhead["stream_status"], "zero_byte")
            self.assertEqual(pmu_overhead["status"], "stream_zero_byte")
            self.assertEqual(pmu_overhead["samples_total"], "")
            self.assertEqual(pmu_overhead["duration_p50_us"], "")
            system_overhead = next(
                row
                for row in collection_overhead
                if row["source"] == "host_metrics"
            )
            self.assertEqual(system_overhead["stream_status"], "recorded")
            self.assertEqual(system_overhead["samples_total"], "1")

            drop_integrity = next(
                row
                for row in read_rows(output / "archive_integrity.csv")
                if row["relative_path"] == "drops.csv"
            )
            self.assertEqual(drop_integrity["valid"], "1")
            self.assertEqual(drop_integrity["status"], "ok")
            self.assertEqual(drop_integrity["observed_size_bytes"], "0")

            campaign_run = read_rows(output / "campaign_runs.csv")[0]
            self.assertEqual(campaign_run["status"], "exited_nonzero")
            self.assertEqual(campaign_run["return_code"], "139")
            self.assertEqual(campaign_run["stop_reason"], "process_exit")

            observer_rows = read_rows(output / "observer_effect_summary.csv")
            process_success = next(
                row
                for row in observer_rows
                if row["metric_scope"] == "process"
                and row["metric_name"] == "process_success"
            )
            self.assertEqual(process_success["sample_count"], "1")
            self.assertEqual(process_success["mean"], "0.0")
            self.assertFalse(
                any(
                    row["metric_scope"] == "event"
                    and row["metric_name"] == "USRP_RX_RECV"
                    for row in observer_rows
                )
            )

            completeness = read_rows(output / "campaign_completeness.csv")[0]
            self.assertEqual(completeness["paired_complete"], "0")
            self.assertEqual(completeness["profile_incomplete_roles"], "nrUE")
            self.assertEqual(completeness["profile_evidence_complete"], "0")
            self.assertEqual(
                completeness["profile_evidence_status"],
                "operational_incomplete;profile_incomplete",
            )

    def test_clean_pmu_off_header_only_streams_are_not_requested(self) -> None:
        self.assertTrue(matches_native_decimal(0.333333, 1.0 / 3.0, 6))
        self.assertFalse(matches_native_decimal(0.333334, 1.0 / 3.0, 6))
        self.assertTrue(matches_native_decimal(0.666666667, 2.0 / 3.0, 9))
        self.assertFalse(matches_native_decimal(0.666666668, 2.0 / 3.0, 9))
        with tempfile.TemporaryDirectory(prefix="oai-profile-pmu-off-") as temporary:
            profile_dir = Path(temporary)
            write_text(
                profile_dir / "pmu_availability.csv",
                PMU_AVAILABILITY_HEADER + "\n",
            )
            write_text(
                profile_dir / "pmu_samples.csv",
                PMU_SAMPLE_HEADER + "\n",
            )
            metadata_by_dir = {
                str(profile_dir): {
                    "run_id": "clean-pmu-off",
                    "role": "nrUE",
                    "hostname": "synthetic-host",
                }
            }
            settings_by_dir = {
                str(profile_dir): {"profile.pmu_mode": "off"}
            }
            availability, summary, quality = pmu_reports(
                [profile_dir],
                metadata_by_dir,
                settings_by_dir,
                {},
            )
            self.assertEqual(len(summary.rows), 0)
            self.assertEqual(availability.rows[0]["stream_status"], "header_only")
            self.assertEqual(availability.rows[0]["status"], "not_requested")
            self.assertEqual(availability.rows[0]["requested"], 0)
            self.assertEqual(availability.rows[0]["available"], 0)
            self.assertEqual(quality.rows[0]["status"], "not_requested")
            self.assertEqual(quality.rows[0]["samples_total"], 0)
            self.assertEqual(quality.rows[0]["read_error_count"], 0)

            pmu_on = profile_dir / "pmu-on"
            pmu_on.mkdir()
            write_text(
                pmu_on / "pmu_availability.csv",
                PMU_AVAILABILITY_HEADER + "\n",
            )
            write_text(
                pmu_on / "pmu_samples.csv",
                PMU_SAMPLE_HEADER + "\n",
            )
            pmu_on_key = str(pmu_on)
            availability, summary, quality = pmu_reports(
                [pmu_on],
                {
                    pmu_on_key: {
                        "run_id": "pmu-on-header-only",
                        "role": "nrUE",
                        "hostname": "synthetic-host",
                    }
                },
                {pmu_on_key: {"profile.pmu_mode": "software"}},
                {},
            )
            self.assertEqual(len(summary.rows), 0)
            self.assertEqual(availability.rows[0]["status"], "stream_header_only")
            self.assertEqual(availability.rows[0]["requested"], "")
            self.assertEqual(availability.rows[0]["available"], "")
            self.assertEqual(quality.rows[0]["status"], "stream_header_only")
            self.assertEqual(quality.rows[0]["samples_total"], "")
            self.assertEqual(quality.rows[0]["read_error_count"], "")

            truncated = profile_dir / "truncated"
            truncated.mkdir()
            write_text(
                truncated / "pmu_availability.csv",
                PMU_AVAILABILITY_HEADER + "\n",
            )
            write_text(
                truncated / "pmu_samples.csv",
                PMU_SAMPLE_HEADER + "\n"
                "2,1,100\n",
            )
            truncated_key = str(truncated)
            _, summary, quality = pmu_reports(
                [truncated],
                {
                    truncated_key: {
                        "run_id": "pmu-truncated",
                        "role": "nrUE",
                        "hostname": "synthetic-host",
                    }
                },
                {truncated_key: {"profile.pmu_mode": "software"}},
                {},
            )
            self.assertEqual(len(summary.rows), 0)
            self.assertEqual(quality.rows[0]["status"], "stream_malformed")
            self.assertEqual(quality.rows[0]["samples_total"], "")

            sample_fields = PMU_SAMPLE_HEADER.split(",")
            valid_sample = {
                "schema_version": 2,
                "sample_id": 1,
                "realtime_ns": 1000,
                "monotonic_raw_ns": 900,
                "tick": 9000,
                "run_id": "valid",
                "experiment_id": "",
                "campaign_id": "",
                "variant": "",
                "trial": "",
                "role": "nrUE",
                "hostname": "synthetic-host",
                "thread_index": 0,
                "tid": 100,
                "thread_name": "producer",
                "target_cpu": 2,
                "event_id": 1,
                "event_name": "cpu_cycles",
                "domain": "hardware",
                "unit": "count",
                "raw_value": 1000,
                "delta_raw": 100,
                "time_enabled_ns": 1000,
                "time_running_ns": 1000,
                "delta_enabled_ns": 100,
                "delta_running_ns": 100,
                "scaled_value": 1000,
                "delta_valid": 1,
                "scaling_valid": 1,
                "delta_scaled": 100,
                "multiplex_ratio": 1.0,
                "interval_ns": 1000,
                "status": "ok",
                "error_code": 0,
            }
            common_nonusable = {
                "delta_raw": 0,
                "delta_enabled_ns": 0,
                "delta_running_ns": 0,
                "delta_scaled": 0,
                "delta_valid": 0,
            }
            valid_nonusable_states = (
                {
                    **common_nonusable,
                    "interval_ns": 0,
                    "status": "warmup",
                },
                {
                    **common_nonusable,
                    "interval_ns": 0,
                    "status": "clock_regression",
                },
                {
                    **common_nonusable,
                    "interval_ns": 1000,
                    "status": "counter_reset_or_reconfigured",
                },
                {
                    "delta_enabled_ns": 0,
                    "delta_running_ns": 1,
                    "delta_scaled": 0,
                    "scaling_valid": 0,
                    "status": "not_running",
                },
                {
                    "raw_value": 0,
                    "delta_raw": 0,
                    "time_enabled_ns": 0,
                    "time_running_ns": 0,
                    "delta_enabled_ns": 0,
                    "delta_running_ns": 0,
                    "scaled_value": 0,
                    "delta_scaled": 0,
                    "multiplex_ratio": 0,
                    "delta_valid": 0,
                    "scaling_valid": 0,
                    "status": "read_error",
                    "error_code": 5,
                },
                {
                    "raw_value": 0,
                    "delta_raw": 0,
                    "time_enabled_ns": 0,
                    "time_running_ns": 0,
                    "delta_enabled_ns": 0,
                    "delta_running_ns": 0,
                    "scaled_value": 0,
                    "delta_scaled": 0,
                    "multiplex_ratio": 0,
                    "delta_valid": 0,
                    "scaling_valid": 0,
                    "status": "malformed_group_read",
                    "error_code": 5,
                },
            )
            for override in valid_nonusable_states:
                self.assertTrue(
                    pmu_sample_row_is_valid(
                        {
                            key: str(value)
                            for key, value in {
                                **valid_sample,
                                **override,
                            }.items()
                        }
                    )
                )
            for name, override in (
                ("invalid-delta-flag", {"delta_valid": 2}),
                ("invalid-scaling-flag", {"scaling_valid": -1}),
                ("nonnumeric-error", {"error_code": "nope"}),
                ("nonfinite-delta", {"delta_scaled": "nan"}),
                ("negative-interval", {"interval_ns": -1}),
                ("negative-multiplex", {"multiplex_ratio": -0.1}),
                ("negative-delta", {"delta_scaled": -1}),
                ("multiplex-above-one", {"multiplex_ratio": 1.1}),
                ("inconsistent-scaled-value", {"scaled_value": 999}),
                ("inconsistent-delta-scaled", {"delta_scaled": 101}),
                ("inconsistent-multiplex", {"multiplex_ratio": 0.5}),
                ("usable-zero-delta-running", {"delta_running_ns": 0}),
                ("scaling-valid-zero-running", {"time_running_ns": 0}),
                (
                    "usable-running-above-enabled",
                    {"delta_enabled_ns": 0, "delta_running_ns": 1},
                ),
                (
                    "scaling-running-above-enabled",
                    {
                        "delta_valid": 0,
                        "time_enabled_ns": 0,
                        "time_running_ns": 1,
                    },
                ),
                (
                    "nonusable-ok-status",
                    {
                        "delta_raw": 0,
                        "delta_enabled_ns": 0,
                        "delta_running_ns": 0,
                        "delta_scaled": 0,
                        "interval_ns": 0,
                        "delta_valid": 0,
                        "status": "ok",
                    },
                ),
                (
                    "warmup-invalid-scaling-flag",
                    {
                        "delta_raw": 0,
                        "delta_enabled_ns": 0,
                        "delta_running_ns": 0,
                        "delta_scaled": 0,
                        "interval_ns": 0,
                        "delta_valid": 0,
                        "scaling_valid": 0,
                        "status": "warmup",
                    },
                ),
                (
                    "not-running-wrong-status",
                    {
                        "delta_running_ns": 0,
                        "delta_scaled": 0,
                        "scaling_valid": 0,
                        "status": "ok",
                    },
                ),
                (
                    "read-error-nonzero-evidence",
                    {
                        "raw_value": 1,
                        "delta_raw": 0,
                        "time_enabled_ns": 0,
                        "time_running_ns": 0,
                        "delta_enabled_ns": 0,
                        "delta_running_ns": 0,
                        "scaled_value": 0,
                        "delta_scaled": 0,
                        "multiplex_ratio": 0,
                        "delta_valid": 0,
                        "scaling_valid": 0,
                        "status": "read_error",
                        "error_code": 5,
                    },
                ),
                (
                    "read-error-wrong-status",
                    {
                        "raw_value": 0,
                        "delta_raw": 0,
                        "time_enabled_ns": 0,
                        "time_running_ns": 0,
                        "delta_enabled_ns": 0,
                        "delta_running_ns": 0,
                        "scaled_value": 0,
                        "delta_scaled": 0,
                        "multiplex_ratio": 0,
                        "delta_valid": 0,
                        "scaling_valid": 0,
                        "status": "warmup",
                        "error_code": 5,
                    },
                ),
                ("zero-valid-interval", {"interval_ns": 0}),
                ("usable-error-code", {"error_code": 5}),
                ("usable-error-status", {"status": "read_error"}),
            ):
                invalid = profile_dir / name
                invalid.mkdir()
                write_csv_rows(
                    invalid / "pmu_availability.csv",
                    PMU_AVAILABILITY_HEADER.split(","),
                    [
                        {
                            "schema_version": 2,
                            "run_id": name,
                            "experiment_id": "",
                            "campaign_id": "",
                            "role": "nrUE",
                            "hostname": "synthetic-host",
                            "thread_index": 0,
                            "tid": 100,
                            "thread_name": "producer",
                            "event_id": 1,
                            "event_name": "cpu_cycles",
                            "domain": "hardware",
                            "requested": 1,
                            "available": 1,
                            "status": "available",
                            "error_code": 0,
                        }
                    ],
                )
                write_csv_rows(
                    invalid / "pmu_samples.csv",
                    sample_fields,
                    [{**valid_sample, "run_id": name, **override}],
                )
                invalid_key = str(invalid)
                _, invalid_summary, invalid_quality = pmu_reports(
                    [invalid],
                    {
                        invalid_key: {
                            "run_id": name,
                            "role": "nrUE",
                            "hostname": "synthetic-host",
                        }
                    },
                    {invalid_key: {"profile.pmu_mode": "software"}},
                    {},
                )
                self.assertEqual(invalid_summary.rows, [])
                self.assertEqual(
                    invalid_quality.rows[0]["status"],
                    "stream_malformed",
                )
                self.assertEqual(invalid_quality.rows[0]["samples_total"], "")

            valid_not_running = profile_dir / "valid-not-running"
            valid_not_running.mkdir()
            write_csv_rows(
                valid_not_running / "pmu_availability.csv",
                PMU_AVAILABILITY_HEADER.split(","),
                [
                    {
                        "schema_version": 2,
                        "run_id": "valid-not-running",
                        "experiment_id": "",
                        "campaign_id": "",
                        "role": "nrUE",
                        "hostname": "synthetic-host",
                        "thread_index": 0,
                        "tid": 100,
                        "thread_name": "producer",
                        "event_id": 1,
                        "event_name": "cpu_cycles",
                        "domain": "hardware",
                        "requested": 1,
                        "available": 1,
                        "status": "available",
                        "error_code": 0,
                    }
                ],
            )
            write_csv_rows(
                valid_not_running / "pmu_samples.csv",
                sample_fields,
                [
                    {
                        **valid_sample,
                        "run_id": "valid-not-running",
                        "delta_enabled_ns": 0,
                        "delta_running_ns": 1,
                        "delta_scaled": 0,
                        "scaling_valid": 0,
                        "status": "not_running",
                    }
                ],
            )
            valid_not_running_key = str(valid_not_running)
            _, not_running_summary, not_running_quality = pmu_reports(
                [valid_not_running],
                {
                    valid_not_running_key: {
                        "run_id": "valid-not-running",
                        "role": "nrUE",
                        "hostname": "synthetic-host",
                    }
                },
                {
                    valid_not_running_key: {
                        "profile.pmu_mode": "software"
                    }
                },
                {},
            )
            self.assertEqual(len(not_running_summary.rows), 1)
            self.assertEqual(
                not_running_summary.rows[0]["usable_samples"],
                0,
            )
            self.assertTrue(
                math.isnan(
                    not_running_summary.rows[0]["estimated_rate_per_second"]
                )
            )
            self.assertEqual(not_running_quality.rows[0]["samples_total"], 1)
            self.assertEqual(
                not_running_quality.rows[0]["delta_valid_count"],
                1,
            )
            self.assertEqual(
                not_running_quality.rows[0]["scaling_valid_count"],
                0,
            )
            self.assertEqual(not_running_quality.rows[0]["usable_count"], 0)
            self.assertEqual(not_running_quality.rows[0]["invalid_count"], 1)
            self.assertEqual(
                not_running_quality.rows[0]["status"],
                "not_running",
            )

            invalid_availability = profile_dir / "invalid-availability"
            invalid_availability.mkdir()
            write_csv_rows(
                invalid_availability / "pmu_availability.csv",
                PMU_AVAILABILITY_HEADER.split(","),
                [
                    {
                        "schema_version": 2,
                        "run_id": "invalid-availability",
                        "experiment_id": "",
                        "campaign_id": "",
                        "role": "nrUE",
                        "hostname": "synthetic-host",
                        "thread_index": 0,
                        "tid": 100,
                        "thread_name": "producer",
                        "event_id": 1,
                        "event_name": "cpu_cycles",
                        "domain": "hardware",
                        "requested": 0,
                        "available": 1,
                        "status": "available",
                        "error_code": 0,
                    }
                ],
            )
            write_text(
                invalid_availability / "pmu_samples.csv",
                PMU_SAMPLE_HEADER + "\n",
            )
            invalid_availability_key = str(invalid_availability)
            invalid_availability_report, _, _ = pmu_reports(
                [invalid_availability],
                {
                    invalid_availability_key: {
                        "run_id": "invalid-availability",
                        "role": "nrUE",
                        "hostname": "synthetic-host",
                    }
                },
                {
                    invalid_availability_key: {
                        "profile.pmu_mode": "software"
                    }
                },
                {},
            )
            self.assertEqual(
                invalid_availability_report.rows[0]["stream_status"],
                "malformed",
            )
            self.assertEqual(
                invalid_availability_report.rows[0]["requested"],
                "",
            )

            write_csv_rows(
                invalid_availability / "pmu_availability.csv",
                PMU_AVAILABILITY_HEADER.split(","),
                [
                    {
                        "schema_version": 2,
                        "run_id": "invalid-availability",
                        "experiment_id": "",
                        "campaign_id": "",
                        "role": "nrUE",
                        "hostname": "synthetic-host",
                        "thread_index": 0,
                        "tid": 100,
                        "thread_name": "producer",
                        "event_id": 1,
                        "event_name": "cpu_cycles",
                        "domain": "hardware",
                        "requested": 1,
                        "available": 1,
                        "status": "permission_denied",
                        "error_code": 13,
                    }
                ],
            )
            contradictory_availability_report, _, _ = pmu_reports(
                [invalid_availability],
                {
                    invalid_availability_key: {
                        "run_id": "invalid-availability",
                        "role": "nrUE",
                        "hostname": "synthetic-host",
                    }
                },
                {
                    invalid_availability_key: {
                        "profile.pmu_mode": "software"
                    }
                },
                {},
            )
            self.assertEqual(
                contradictory_availability_report.rows[0]["stream_status"],
                "malformed",
            )
            self.assertEqual(
                contradictory_availability_report.rows[0]["available"],
                "",
            )

            missing_identity = profile_dir / "missing-sample-identity"
            missing_identity.mkdir()
            write_text(
                missing_identity / "pmu_availability.csv",
                PMU_AVAILABILITY_HEADER + "\n",
            )
            write_text(
                missing_identity / "pmu_samples.csv",
                "event_name,delta_valid,scaling_valid,delta_scaled,"
                "multiplex_ratio,interval_ns,status,error_code\n"
                "cpu_cycles,1,1,100,1.0,1000,ok,0\n",
            )
            missing_identity_key = str(missing_identity)
            _, missing_summary, missing_quality = pmu_reports(
                [missing_identity],
                {
                    missing_identity_key: {
                        "run_id": "missing-sample-identity",
                        "role": "nrUE",
                        "hostname": "synthetic-host",
                    }
                },
                {missing_identity_key: {"profile.pmu_mode": "software"}},
                {},
            )
            self.assertEqual(missing_summary.rows, [])
            self.assertEqual(
                missing_quality.rows[0]["status"],
                "stream_malformed",
            )

            malformed = profile_dir / "malformed"
            malformed.mkdir()
            write_text(malformed / "pmu_availability.csv", '"event_name\n')
            write_text(malformed / "pmu_samples.csv", '"event_name\n')
            malformed_key = str(malformed)
            availability, summary, quality = pmu_reports(
                [malformed],
                {
                    malformed_key: {
                        "run_id": "pmu-malformed",
                        "role": "nrUE",
                        "hostname": "synthetic-host",
                    }
                },
                {malformed_key: {"profile.pmu_mode": "software"}},
                {},
            )
            self.assertEqual(len(summary.rows), 0)
            self.assertEqual(availability.rows[0]["stream_status"], "malformed")
            self.assertEqual(availability.rows[0]["status"], "stream_malformed")
            self.assertEqual(quality.rows[0]["status"], "stream_malformed")

            blank = profile_dir / "blank"
            blank.mkdir()
            write_text(
                blank / "pmu_availability.csv",
                PMU_AVAILABILITY_HEADER
                + "\n"
                + "," * (len(PMU_AVAILABILITY_HEADER.split(",")) - 1)
                + "\n",
            )
            write_text(
                blank / "pmu_samples.csv",
                PMU_SAMPLE_HEADER
                + "\n"
                + "," * (len(PMU_SAMPLE_HEADER.split(",")) - 1)
                + "\n",
            )
            blank_key = str(blank)
            availability, summary, quality = pmu_reports(
                [blank],
                {
                    blank_key: {
                        "run_id": "pmu-blank",
                        "role": "nrUE",
                        "hostname": "synthetic-host",
                    }
                },
                {blank_key: {"profile.pmu_mode": "software"}},
                {},
            )
            self.assertEqual(len(summary.rows), 0)
            self.assertEqual(availability.rows[0]["stream_status"], "malformed")
            self.assertEqual(availability.rows[0]["status"], "stream_malformed")
            self.assertEqual(quality.rows[0]["status"], "stream_malformed")
            self.assertEqual(quality.rows[0]["samples_total"], "")

    def test_primitive_overhead_malformed_row_does_not_fabricate_zeros(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="oai-profile-primitive-malformed-"
        ) as temporary:
            profile_dir = Path(temporary)
            fields = (
                "schema_version,run_id,experiment_id,campaign_id,variant,trial,"
                "role,hostname,phase,sample_index,primitive,event_kind,cpu_start,"
                "cpu_end,cpu_migrated,outer_start_tick,outer_end_tick,"
                "outer_duration_tick,outer_duration_us,event_record_expected,"
                "event_recorded,event_seq,event_duration_tick,event_duration_us,"
                "drop_delta,status"
            ).split(",")
            base_row = {
                "schema_version": 2,
                "run_id": "primitive-malformed",
                "experiment_id": "",
                "campaign_id": "",
                "variant": "in-process",
                "trial": 1,
                "role": "nrUE",
                "hostname": "synthetic-host",
                "phase": "measurement",
                "sample_index": 0,
                "primitive": "counter_pair",
                "event_kind": "unknown",
                "cpu_start": 2,
                "cpu_end": 2,
                "cpu_migrated": 0,
                "outer_start_tick": 100,
                "outer_end_tick": 110,
                "outer_duration_tick": 10,
                "outer_duration_us": 1.0,
                "event_record_expected": 0,
                "event_recorded": 0,
                "event_seq": 0,
                "event_duration_tick": 0,
                "event_duration_us": 0.0,
                "drop_delta": 0,
                "status": "ok",
            }
            write_csv_rows(
                profile_dir / "profiler_primitive_overhead.csv",
                fields,
                [{**base_row, "sample_index": ""}],
            )
            profile_key = str(profile_dir)
            report = primitive_overhead_report(
                [profile_dir],
                {
                    profile_key: {
                        "run_id": "primitive-malformed",
                        "experiment_id": "",
                        "campaign_id": "",
                        "variant": "in-process",
                        "trial": "1",
                        "role": "nrUE",
                        "hostname": "synthetic-host",
                    }
                },
                {},
            )
            self.assertEqual(len(report.rows), 1)
            self.assertEqual(report.rows[0]["stream_status"], "malformed")
            self.assertEqual(report.rows[0]["status"], "stream_malformed")
            self.assertEqual(report.rows[0]["samples_total"], "")
            self.assertEqual(report.rows[0]["event_recorded_count"], "")
            self.assertEqual(report.rows[0]["duration_p50_us"], "")

            contradictory_rows = (
                {
                    **base_row,
                    "primitive": "span_start_stop",
                    "event_kind": "duration",
                    "event_record_expected": 1,
                    "status": "ok",
                },
                {
                    **base_row,
                    "primitive": "counter_pair",
                    "event_kind": "duration",
                    "event_record_expected": 1,
                    "event_recorded": 1,
                    "event_seq": 1,
                    "event_duration_tick": 5,
                    "event_duration_us": 0.5,
                },
                {
                    **base_row,
                    "primitive": "fabricated_primitive",
                },
            )
            for contradictory_row in contradictory_rows:
                write_csv_rows(
                    profile_dir / "profiler_primitive_overhead.csv",
                    fields,
                    [contradictory_row],
                )
                report = primitive_overhead_report(
                    [profile_dir],
                    {
                        profile_key: {
                            "run_id": "primitive-malformed",
                            "experiment_id": "",
                            "campaign_id": "",
                            "variant": "in-process",
                            "trial": "1",
                            "role": "nrUE",
                            "hostname": "synthetic-host",
                        }
                    },
                    {},
                )
                self.assertEqual(report.rows[0]["stream_status"], "malformed")
                self.assertEqual(report.rows[0]["status"], "stream_malformed")
                self.assertEqual(report.rows[0]["samples_total"], "")

            write_csv_rows(
                profile_dir / "profiler_primitive_overhead.csv",
                fields,
                [
                    {
                        **base_row,
                        "phase": "setup",
                        "primitive": "calibration",
                        "cpu_start": -1,
                        "cpu_end": -1,
                        "outer_start_tick": 0,
                        "outer_end_tick": 0,
                        "outer_duration_tick": 0,
                        "outer_duration_us": 0.0,
                        "status": "allocation_failed",
                    }
                ],
            )
            allocation_report = primitive_overhead_report(
                [profile_dir],
                {
                    profile_key: {
                        "run_id": "primitive-malformed",
                        "experiment_id": "",
                        "campaign_id": "",
                        "variant": "in-process",
                        "trial": "1",
                        "role": "nrUE",
                        "hostname": "synthetic-host",
                    }
                },
                {},
            )
            self.assertEqual(
                allocation_report.rows[0]["stream_status"],
                "recorded",
            )
            self.assertEqual(
                allocation_report.rows[0]["status"],
                "allocation_failed",
            )
            self.assertEqual(allocation_report.rows[0]["samples_total"], 1)
            self.assertEqual(allocation_report.rows[0]["samples_valid"], 0)

            write_csv_rows(
                profile_dir / "profiler_primitive_overhead.csv",
                fields,
                [
                    {
                        **base_row,
                        "primitive": "span_start_stop",
                        "event_kind": "duration",
                        "event_record_expected": 1,
                        "event_recorded": 1,
                        "event_seq": 0,
                        "event_duration_tick": 5,
                        "event_duration_us": 0.5,
                    }
                ],
            )
            zero_sequence_report = primitive_overhead_report(
                [profile_dir],
                {
                    profile_key: {
                        "run_id": "primitive-malformed",
                        "experiment_id": "",
                        "campaign_id": "",
                        "variant": "in-process",
                        "trial": "1",
                        "role": "nrUE",
                        "hostname": "synthetic-host",
                    }
                },
                {},
            )
            self.assertEqual(
                zero_sequence_report.rows[0]["stream_status"],
                "recorded",
            )
            self.assertEqual(zero_sequence_report.rows[0]["status"], "ok")
            self.assertEqual(zero_sequence_report.rows[0]["samples_valid"], 1)
            self.assertEqual(
                zero_sequence_report.rows[0]["event_recorded_count"],
                1,
            )

    def test_deadline_reconstruction_rejects_uncorrelated_and_malformed_evidence(self) -> None:
        self.assertEqual(round_div_signed(5, 2), 3)
        self.assertEqual(round_div_signed(-5, 2), -3)

        def event(
            name: str,
            seq: int,
            start_tick: int,
            correlation_id: int,
            *,
            duration_tick: int = 0,
            frame: int = 1,
            slot: int = 2,
            span_id: int = 0,
            parent_id: int = 0,
            flags: int = 0,
            aux0: int = 0,
            aux1: int = 0,
            aux2: int = 0,
            aux3: int = 0,
        ) -> dict[str, str]:
            return {
                "schema_version": "2",
                "seq": str(seq),
                "tid": "20",
                "thread_name": "ue-test",
                "event_name": name,
                "event_kind": "duration" if duration_tick else "instant",
                "frame": str(frame),
                "slot": str(slot),
                "absolute_slot": str(frame * 20 + slot),
                "correlation_id": str(correlation_id),
                "span_id": str(span_id),
                "parent_id": str(parent_id),
                "flags": str(flags),
                "aux0": str(aux0),
                "aux1": str(aux1),
                "aux2": str(aux2),
                "aux3": str(aux3),
                "start_tick": str(start_tick),
                "duration_tick": str(duration_tick),
            }

        events = [
            event(
                "UE_RF_READ",
                1,
                100,
                11,
                duration_tick=100,
                span_id=500,
            ),
            event(
                "USRP_RX_RECV",
                2,
                120,
                11,
                duration_tick=30,
            ),
            event(
                "USRP_RX_RECV",
                3,
                160,
                99,
                duration_tick=30,
            ),
            event(
                "UE_TX_DEADLINE_COMPUTE",
                4,
                210,
                11,
                flags=1,
                aux0=1000,
                aux1=2000,
                aux2=1000,
                aux3=1_000_000,
            ),
            event(
                "UE_TX_DEADLINE_CHECK",
                5,
                1160,
                11,
                flags=3,
                aux0=2_010_000,
                aux1=2_000_000,
                aux2=10_000,
            ),
            event(
                "UE_TX_DEADLINE_CHECK",
                6,
                500,
                12,
                slot=4,
                flags=1,
                aux0=1_900_000,
                aux1=2_000_000,
                aux2=-100_000,
            ),
            event(
                "UE_TX_DEADLINE_COMPUTE",
                7,
                320,
                13,
                slot=3,
                flags=1,
                aux0=1000,
                aux1=2000,
                aux2=1000,
                aux3=1_000_000,
            ),
            event(
                "UE_TX_DEADLINE_CHECK",
                8,
                1330,
                13,
                slot=3,
                flags=1,
                aux0=2_010_000,
                aux1=2_000_000,
                aux2=9000,
            ),
        ]
        reports = build_deadline_reports(
            {"/profile": events},
            {
                "/profile": {
                    "schema_version": "2",
                    "process_name": "nr-uesoftmodem",
                    "role": "nrUE",
                    "counter_hz": "1000000",
                }
            },
        )
        rows = {int(row["correlation_id"]): row for row in reports.check_rows}

        valid = rows[11]
        self.assertEqual(valid["status"], "ok")
        self.assertEqual(valid["radio_anchor_seq"], "2")
        self.assertEqual(valid["radio_anchor_match"], "temporal_containment")
        self.assertEqual(valid["reconstructed_offset_tick"], 1000)
        self.assertEqual(valid["reconstructed_deadline_tick"], 1150)
        self.assertEqual(valid["reconstructed_signed_lateness_tick"], 10)
        self.assertAlmostEqual(float(valid["reconstructed_signed_lateness_us"]), 10.0)
        self.assertAlmostEqual(float(valid["runtime_signed_lateness_us"]), 10.0)
        self.assertEqual(valid["classification_agreement"], 1)

        self.assertEqual(rows[12]["status"], "unpaired")
        self.assertEqual(rows[12]["compute_status"], "missing")
        self.assertIn("lateness_mismatch", str(rows[13]["runtime_status"]))
        self.assertEqual(rows[13]["reconstruction_status"], "missing_radio_anchor_event")
        self.assertEqual(rows[13]["status"], "invalid")

        summary = reports.summary_rows[0]
        self.assertEqual(summary["compute_events"], 2)
        self.assertEqual(summary["check_events"], 3)
        self.assertEqual(summary["paired_checks"], 2)
        self.assertEqual(summary["unpaired_checks"], 1)
        self.assertEqual(summary["runtime_valid_checks"], 2)
        self.assertEqual(summary["runtime_invalid_checks"], 1)
        self.assertEqual(summary["reconstruction_valid_checks"], 1)
        self.assertEqual(summary["classification_agreements"], 1)
        self.assertEqual(summary["status"], "partial")

    def test_external_alignment_paths_and_duplicate_campaign_roles(self) -> None:
        realtime_only = {
            "status": "recorded",
            "alignment_method": "shared_monotonic_raw",
            "start_realtime_ns": "100",
            "end_realtime_ns": "200",
        }
        self.assertEqual(
            source_alignment_status(realtime_only),
            "required_monotonic_anchors_incomplete",
        )
        self.assertEqual(
            source_alignment_status(
                {
                    **realtime_only,
                    "start_monotonic_raw_ns": "300",
                    "end_monotonic_raw_ns": "400",
                }
            ),
            "aligned_with_declared_method",
        )
        self.assertEqual(
            source_alignment_status(
                {
                    **realtime_only,
                    "start_monotonic_raw_ns": "400",
                    "end_monotonic_raw_ns": "300",
                }
            ),
            "required_monotonic_anchors_invalid",
        )
        self.assertEqual(
            source_alignment_status(
                {
                    "status": "recorded",
                    "alignment_method": "aggregate_run_interval",
                }
            ),
            "aggregate_only_anchors_incomplete",
        )

        with tempfile.TemporaryDirectory(prefix="oai-profile-report-symlink-") as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            outside = root / "outside"
            outside.mkdir()
            write_text(outside / "perf_stat.csv", "1.000;100;;cycles;100.00%\n")
            (run_dir / "sidecars").symlink_to(outside, target_is_directory=True)
            write_csv_rows(
                run_dir / "external_sources.csv",
                [
                    "source_id",
                    "source_type",
                    "artifact_path",
                    "status",
                    "alignment_method",
                ],
                [
                    {
                        "source_id": "unsafe-perf",
                        "source_type": "perf_stat",
                        "artifact_path": "sidecars/perf_stat.csv",
                        "status": "recorded",
                        "alignment_method": "aggregate_run_interval",
                    }
                ],
            )
            external = external_sources_report([run_dir], {}, {})
            self.assertEqual(external.rows[0]["artifact_status"], "unsafe_symlink")
            self.assertEqual(perf_stat_report([run_dir], {}, {}).rows, [])

        campaign_rows = [
            {
                "profile_dir": "/run/gnb-a",
                "campaign_id": "campaign",
                "experiment_id": "experiment",
                "case": "case",
                "variant": "variant",
                "trial": "1",
                "role": "gNB",
                "status": "finished",
                "archive_status": "finalization_pending",
                "archive_manifest_present": 1,
            },
            {
                "profile_dir": "/run/gnb-b",
                "campaign_id": "campaign",
                "experiment_id": "experiment",
                "case": "case",
                "variant": "variant",
                "trial": "1",
                "role": "gNB",
                "status": "finished",
                "archive_status": "finalization_pending",
                "archive_manifest_present": 1,
            },
            {
                "profile_dir": "/run/ue",
                "campaign_id": "campaign",
                "experiment_id": "experiment",
                "case": "case",
                "variant": "variant",
                "trial": "1",
                "role": "nrUE",
                "status": "finished",
                "archive_status": "finalization_pending",
                "archive_manifest_present": 1,
            },
        ]
        integrity_rows = [
            {"profile_dir": row["profile_dir"], "valid": 1} for row in campaign_rows
        ]
        completeness = campaign_completeness_report(campaign_rows, integrity_rows)
        self.assertEqual(len(completeness.rows), 1)
        self.assertEqual(completeness.rows[0]["member_count"], 3)
        self.assertEqual(completeness.rows[0]["duplicate_roles"], "gNB:2")
        self.assertEqual(completeness.rows[0]["paired_complete"], 0)
        self.assertIn("duplicate_role", str(completeness.rows[0]["status"]))

    def test_campaign_profile_evidence_is_separate_from_operations(self) -> None:
        def member(role: str, profile_enabled: bool) -> dict[str, object]:
            return {
                "profile_dir": f"/run/{role}",
                "campaign_id": "campaign",
                "experiment_id": "experiment",
                "case": "case",
                "variant": "in-process" if profile_enabled else "disabled",
                "trial": "1",
                "role": role,
                "profile_enabled": profile_enabled,
                "status": "finished",
                "return_code": 0,
                "stop_reason": "duration_elapsed",
                "duration_status": "valid",
                "archive_status": "finalization_pending",
                "archive_manifest_present": 1,
            }

        profiled_members = [member("gNB", True), member("nrUE", True)]
        profiled_integrity = [
            {"profile_dir": row["profile_dir"], "valid": 1}
            for row in profiled_members
        ]
        incomplete = campaign_completeness_report(
            profiled_members,
            profiled_integrity,
            {
                "/run/gNB": "complete",
                "/run/nrUE": "lifecycle_unknown;drop_diagnostics_zero_byte",
            },
        ).rows[0]
        self.assertEqual(incomplete["paired_complete"], 1)
        self.assertEqual(incomplete["status"], "complete")
        self.assertEqual(incomplete["profile_complete_roles"], "gNB")
        self.assertEqual(incomplete["profile_incomplete_roles"], "nrUE")
        self.assertEqual(
            incomplete["profile_coverage_status_by_role"],
            "gNB:complete|nrUE:lifecycle_unknown;drop_diagnostics_zero_byte",
        )
        self.assertEqual(incomplete["profile_evidence_complete"], 0)
        self.assertEqual(incomplete["profile_evidence_status"], "profile_incomplete")

        complete = campaign_completeness_report(
            profiled_members,
            profiled_integrity,
            {"/run/gNB": "complete", "/run/nrUE": "complete"},
        ).rows[0]
        self.assertEqual(complete["paired_complete"], 1)
        self.assertEqual(complete["profile_complete_roles"], "gNB;nrUE")
        self.assertEqual(complete["profile_evidence_complete"], 1)
        self.assertEqual(complete["profile_evidence_status"], "complete")

        disabled_members = [member("gNB", False), member("nrUE", False)]
        disabled_integrity = [
            {"profile_dir": row["profile_dir"], "valid": 1}
            for row in disabled_members
        ]
        disabled = campaign_completeness_report(
            disabled_members,
            disabled_integrity,
            {},
        ).rows[0]
        self.assertEqual(disabled["paired_complete"], 1)
        self.assertEqual(
            disabled["profile_not_applicable_roles"],
            "gNB;nrUE",
        )
        self.assertEqual(disabled["profile_incomplete_roles"], "")
        self.assertEqual(disabled["profile_evidence_complete"], 1)
        self.assertEqual(disabled["profile_evidence_status"], "complete")

        mismatched_members = [
            {**member("gNB", False), "variant": "in-process"},
            {**member("nrUE", False), "variant": "in-process"},
        ]
        mismatched_integrity = [
            {"profile_dir": row["profile_dir"], "valid": 1}
            for row in mismatched_members
        ]
        mismatch = campaign_completeness_report(
            mismatched_members,
            mismatched_integrity,
            {},
        ).rows[0]
        self.assertEqual(mismatch["paired_complete"], 1)
        self.assertEqual(
            mismatch["profile_setting_mismatch_roles"],
            "gNB;nrUE",
        )
        self.assertEqual(mismatch["profile_evidence_complete"], 0)
        self.assertEqual(
            mismatch["profile_evidence_status"],
            "profile_setting_mismatch",
        )

        contaminated_disabled = campaign_completeness_report(
            disabled_members,
            disabled_integrity,
            {"/run/gNB": "complete"},
        ).rows[0]
        self.assertEqual(contaminated_disabled["paired_complete"], 1)
        self.assertEqual(
            contaminated_disabled["unexpected_profile_evidence_roles"],
            "gNB",
        )
        self.assertEqual(
            contaminated_disabled["profile_not_applicable_roles"],
            "nrUE",
        )
        self.assertEqual(
            contaminated_disabled["profile_coverage_status_by_role"],
            "gNB:unexpected_profile_evidence:complete|nrUE:not_applicable",
        )
        self.assertEqual(
            contaminated_disabled["profile_evidence_complete"],
            0,
        )
        self.assertEqual(
            contaminated_disabled["profile_evidence_status"],
            "unexpected_profile_evidence",
        )

        events_only_disabled_members = [
            {
                **disabled_members[0],
                "profiler_metadata_present": 0,
                "events_present": 1,
            },
            {
                **disabled_members[1],
                "profiler_metadata_present": 0,
                "events_present": 0,
            },
        ]
        events_only_contamination = campaign_completeness_report(
            events_only_disabled_members,
            disabled_integrity,
            {},
        ).rows[0]
        self.assertEqual(
            events_only_contamination["unexpected_profile_evidence_roles"],
            "gNB",
        )
        self.assertEqual(
            events_only_contamination["profile_coverage_status_by_role"],
            "gNB:unexpected_profile_evidence:artifact_presence|nrUE:not_applicable",
        )
        self.assertEqual(
            events_only_contamination["profile_evidence_complete"],
            0,
        )

        unevaluated = campaign_completeness_report(
            profiled_members,
            profiled_integrity,
        ).rows[0]
        self.assertEqual(unevaluated["paired_complete"], 1)
        self.assertEqual(unevaluated["profile_evidence_complete"], "")
        self.assertEqual(unevaluated["profile_evidence_status"], "not_evaluated")

    def test_early_campaign_exit_is_unsuccessful_and_excluded_from_duration(self) -> None:
        successful = {
            "profile_dir": "/run/valid",
            "campaign_id": "campaign",
            "experiment_id": "valid-experiment",
            "case": "case",
            "variant": "disabled",
            "trial": "1",
            "role": "gNB",
            "status": "finished",
            "return_code": 0,
            "stop_reason": "duration_elapsed",
            "duration_s": 120.0,
            "duration_status": "valid",
            "archive_status": "finalization_pending",
            "archive_manifest_present": 1,
        }
        early = {
            **successful,
            "profile_dir": "/run/early",
            "experiment_id": "early-experiment",
            "trial": "2",
            "stop_reason": "paired_role_exited",
            "duration_s": 2.0,
        }
        observer = observer_effect_report([successful, early], [], {}, {}, {})
        duration = next(row for row in observer.rows if row["metric_name"] == "process_duration")
        success = next(row for row in observer.rows if row["metric_name"] == "process_success")
        self.assertEqual(duration["sample_count"], 1)
        self.assertEqual(duration["p50"], 120.0)
        self.assertEqual(success["sample_count"], 2)
        self.assertEqual(success["mean"], 0.5)

        successful_event = {
            **successful,
            "profile_dir": "/run/valid-event",
            "variant": "in-process",
            "profile_enabled": True,
        }
        early_event = {
            **early,
            "profile_dir": "/run/early-event",
            "variant": "in-process",
        }
        summary_rows = [
            {
                "profile_dir": successful_event["profile_dir"],
                "event_kind": "duration",
                "event_name": "UE_SLOT_PROCESS",
                "p50_us": 100.0,
                "profile_coverage_status": "complete",
            },
            {
                "profile_dir": early_event["profile_dir"],
                "event_kind": "duration",
                "event_name": "UE_SLOT_PROCESS",
                "p50_us": 1.0,
                "profile_coverage_status": "complete",
            },
        ]
        metadata_by_dir = {
            str(row["profile_dir"]): {
                "campaign_id": "campaign",
                "role": "gNB",
                "variant": "in-process",
                "trial": str(row["trial"]),
            }
            for row in (successful_event, early_event)
        }
        campaign_by_dir = {
            str(row["profile_dir"]): {"case": "case"}
            for row in (successful_event, early_event)
        }
        missing_event_profile = {
            **successful_event,
            "profile_dir": "/run/missing-event",
            "experiment_id": "missing-event-experiment",
            "trial": "4",
        }
        missing_event_key = str(missing_event_profile["profile_dir"])
        metadata_by_dir[missing_event_key] = {
            "campaign_id": "campaign",
            "role": "gNB",
            "variant": "in-process",
            "trial": "4",
        }
        campaign_by_dir[missing_event_key] = {"case": "case"}
        observer = observer_effect_report(
            [successful_event, early_event, missing_event_profile],
            summary_rows,
            metadata_by_dir,
            campaign_by_dir,
            {
                str(successful_event["profile_dir"]): "complete",
                str(early_event["profile_dir"]): "complete",
                missing_event_key: "event_stream_header_only",
            },
        )
        event = next(row for row in observer.rows if row["metric_name"] == "UE_SLOT_PROCESS")
        self.assertEqual(event["sample_count"], 1)
        self.assertEqual(event["p50"], 100.0)
        self.assertEqual(
            event["excluded_successful_incomplete_profile_count"],
            1,
        )

        baseline = {
            **successful_event,
            "profile_dir": "/run/incomplete-baseline",
            "trial": "3",
        }
        variant = {
            **successful_event,
            "profile_dir": "/run/complete-variant",
            "variant": "pmu-software",
            "trial": "3",
        }
        baseline_key = str(baseline["profile_dir"])
        variant_key = str(variant["profile_dir"])
        metadata_by_dir = {
            baseline_key: {
                "campaign_id": "campaign",
                "role": "gNB",
                "variant": "in-process",
                "trial": "3",
            },
            variant_key: {
                "campaign_id": "campaign",
                "role": "gNB",
                "variant": "pmu-software",
                "trial": "3",
            },
        }
        campaign_by_dir = {
            baseline_key: {"case": "case"},
            variant_key: {"case": "case"},
        }
        observer = observer_effect_report(
            [baseline, variant],
            [
                {
                    "profile_dir": baseline_key,
                    "event_kind": "duration",
                    "event_name": "UE_SLOT_PROCESS",
                    "p50_us": 100.0,
                    "profile_coverage_status": "lifecycle_unknown",
                },
                {
                    "profile_dir": variant_key,
                    "event_kind": "duration",
                    "event_name": "UE_SLOT_PROCESS",
                    "p50_us": 110.0,
                    "profile_coverage_status": "complete",
                },
            ],
            metadata_by_dir,
            campaign_by_dir,
            {
                baseline_key: "lifecycle_unknown",
                variant_key: "complete",
            },
        )
        pmu_variant = next(
            row
            for row in observer.rows
            if row["metric_name"] == "UE_SLOT_PROCESS"
            and row["variant"] == "pmu-software"
        )
        self.assertEqual(pmu_variant["sample_count"], 1)
        self.assertEqual(
            pmu_variant["baseline_status"],
            "excluded_incomplete_profile",
        )
        self.assertEqual(
            pmu_variant["effect_estimator"],
            "unavailable_incomplete_baseline",
        )
        self.assertTrue(math.isnan(float(pmu_variant["p50_delta"])))

        missing_baseline = observer_effect_report(
            [variant],
            [
                {
                    "profile_dir": variant_key,
                    "event_kind": "duration",
                    "event_name": "UE_SLOT_PROCESS",
                    "p50_us": 110.0,
                    "profile_coverage_status": "complete",
                }
            ],
            {variant_key: metadata_by_dir[variant_key]},
            {variant_key: campaign_by_dir[variant_key]},
            {variant_key: "complete"},
        )
        missing_baseline_variant = next(
            row
            for row in missing_baseline.rows
            if row["metric_name"] == "UE_SLOT_PROCESS"
        )
        self.assertEqual(missing_baseline_variant["baseline_status"], "missing")
        self.assertEqual(
            missing_baseline_variant["effect_estimator"],
            "unavailable_missing_baseline",
        )

        paired_members = [
            {**successful, "experiment_id": "paired", "role": "gNB"},
            {**early, "experiment_id": "paired", "role": "nrUE"},
        ]
        integrity_rows = [
            {"profile_dir": row["profile_dir"], "valid": 1} for row in paired_members
        ]
        completeness = campaign_completeness_report(paired_members, integrity_rows)
        self.assertEqual(completeness.rows[0]["successful_roles"], "gNB")
        self.assertEqual(completeness.rows[0]["paired_complete"], 0)
        self.assertIn("unsuccessful_role", str(completeness.rows[0]["status"]))

    def test_pairing_is_bijective_and_experiment_ids_are_authoritative(self) -> None:
        def run(name: str, role: str, experiment_id: str, start: int, end: int) -> dict[str, object]:
            return {
                "profile_dir": name,
                "run_id": name,
                "role": role,
                "experiment_id": experiment_id,
                "start_realtime_ns": start,
                "end_realtime_ns": end,
            }

        exact = build_pairs(
            [
                run("exact-gNB", "gNB", "experiment-a", 100, 300),
                run("exact-nrUE", "nrUE", "experiment-a", 150, 250),
            ]
        )
        self.assertEqual(len(exact), 1)
        self.assertEqual(exact[0]["status"], "paired")
        self.assertEqual(exact[0]["method"], "experiment_id")

        duplicate = build_pairs(
            [
                run("duplicate-gNB-a", "gNB", "experiment-b", 100, 300),
                run("duplicate-gNB-b", "gNB", "experiment-b", 100, 300),
                run("duplicate-nrUE", "nrUE", "experiment-b", 150, 250),
            ]
        )
        self.assertEqual(len(duplicate), 3)
        self.assertEqual({row["status"] for row in duplicate}, {"ambiguous"})
        represented = {
            str(row["gnb_profile_dir"] or row["ue_profile_dir"])
            for row in duplicate
        }
        self.assertEqual(
            represented,
            {"duplicate-gNB-a", "duplicate-gNB-b", "duplicate-nrUE"},
        )

        mismatched_ids = build_pairs(
            [
                run("mismatch-gNB", "gNB", "experiment-c", 100, 300),
                run("mismatch-nrUE", "nrUE", "experiment-d", 150, 250),
            ]
        )
        self.assertEqual(len(mismatched_ids), 2)
        self.assertEqual({row["status"] for row in mismatched_ids}, {"unmatched"})
        self.assertEqual({row["method"] for row in mismatched_ids}, {"experiment_id"})

        legacy_ambiguous = build_pairs(
            [
                run("legacy-gNB", "gNB", "", 100, 400),
                run("legacy-nrUE-a", "nrUE", "", 150, 250),
                run("legacy-nrUE-b", "nrUE", "", 200, 300),
            ]
        )
        self.assertEqual(len(legacy_ambiguous), 3)
        self.assertEqual({row["status"] for row in legacy_ambiguous}, {"ambiguous"})
        self.assertFalse(any(row["status"] == "paired" for row in legacy_ambiguous))

        legacy_unique = build_pairs(
            [
                run("legacy-unique-gNB", "gNB", "", 100, 300),
                run("legacy-unique-nrUE", "nrUE", "", 150, 250),
            ]
        )
        self.assertEqual(len(legacy_unique), 1)
        self.assertEqual(legacy_unique[0]["status"], "paired")
        self.assertEqual(legacy_unique[0]["method"], "wallclock_overlap_mutual_unique")

    def test_phase1_and_phase2a_profiles(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-analyzer-test-") as tmp:
            root = Path(tmp)
            v1 = root / "v1_nrUE"
            v2 = root / "v2_nrUE"
            output = root / "analysis"
            explicit_full_output = root / "analysis_explicit_full"
            publication_output = root / "analysis_publication"
            filtered_output = root / "analysis_filtered"
            v1.mkdir()
            v2.mkdir()

            write_text(
                v1 / "metadata.txt",
                "process_name=nr-uesoftmodem\nrole=nrUE\nrun_id=v1\n"
                "start_realtime_ns=1000000000\nend_realtime_ns=2000000000\nclean_shutdown=1\n",
            )
            write_text(
                v1 / "events.csv",
                "seq,tid,thread_name,event_id,event_name,frame,slot,flags,aux0,aux1,aux2,aux3,"
                "start_tick,duration_tick,duration_us\n"
                "0,10,ue-v1,2,UE_RF_READ,1,2,0,512,1,512,1000,100,20,2.000\n"
                "1,10,ue-v1,17,UE_TX_DEADLINE_MISS,1,2,0,100,90,10,0,130,1,0.100\n",
            )
            write_text(v1 / "event_catalog.csv", "event_id,event_name\n2,UE_RF_READ\n17,UE_TX_DEADLINE_MISS\n")
            write_text(v1 / "drops.csv", "thread_index,tid,thread_name,dropped_records\n0,10,ue-v1,3\n")
            write_text(
                v1 / "sync.csv",
                "realtime_ns,monotonic_raw_ns,tick\n1000000000,1,100\n2000000000,2,200\n",
            )

            write_text(
                v2 / "metadata.txt",
                "schema_version=2\nevent_record_size_bytes=120\nmax_nesting_depth=64\ncounter_hz=10000000\n"
                "process_name=nr-uesoftmodem\nrole=nrUE\nrun_id=v2\n"
                "start_realtime_ns=3000000000\nend_realtime_ns=4000000000\n"
                "duration_realtime_ns=1000000000\n"
                "start_monotonic_raw_ns=3\nend_monotonic_raw_ns=4\n"
                "duration_monotonic_raw_ns=1\n"
                "duration_clock=CLOCK_MONOTONIC_RAW\n"
                "realtime_clock_regressed=0\nmonotonic_raw_clock_regressed=0\n"
                "clean_shutdown=1\n",
            )
            write_text(
                v2 / "events.csv",
                "schema_version,seq,tid,thread_name,event_id,event_name,event_kind,nesting_depth,frame,slot,"
                "absolute_slot,correlation_id,span_id,parent_id,cpu_start,cpu_end,cpu_migrated,flags,"
                "aux0,aux1,aux2,aux3,start_tick,duration_tick,duration_us\n"
                "2,0,20,ue-v2,1,UE_SLOT_LOOP,duration,0,10,3,203,77,281474976710657,0,1,1,0,0,"
                "6,512,512,32,1000,100,10.000\n"
                "2,1,20,ue-v2,2,UE_RF_READ,duration,1,10,3,203,77,281474976710658,"
                "281474976710657,1,2,1,0,512,1,512,2000,1020,50,5.000\n"
                "2,2,20,ue-v2,17,UE_TX_DEADLINE_MISS,instant,1,10,3,203,77,281474976710659,"
                "281474976710657,2,2,0,0,200,190,10,0,1080,0,0.000\n"
                "2,3,20,ue-v2,100,TEST_PARALLEL_ROOT,duration,0,10,4,204,78,1000,0,1,1,0,0,"
                "0,0,0,0,2000,1000,100.000\n"
                "2,4,21,worker-a,101,TEST_PARALLEL_CHILD_A,duration,1,10,4,204,78,1001,1000,2,2,0,0,"
                "0,0,0,0,2100,500,50.000\n"
                "2,5,22,worker-b,102,TEST_PARALLEL_CHILD_B,duration,1,10,4,204,78,1002,1000,3,3,0,0,"
                "0,0,0,0,2300,500,50.000\n"
                "2,6,20,ue-v2,103,TEST_DISPATCH,duration,0,10,5,205,79,2000,0,1,1,0,0,"
                "0,0,0,0,4000,10,1.000\n"
                "2,7,21,worker-a,104,TEST_ASYNC_WORKER,duration,1,10,5,205,79,2001,2000,2,2,0,0,"
                "0,0,0,0,4100,100,10.000\n"
                "2,8,22,worker-b,105,TEST_ORPHAN,duration,1,10,6,206,80,3000,9999,3,3,0,0,"
                "0,0,0,0,5000,10,1.000\n"
                "2,9,21,worker-a,106,LDPC_DECODER_SEGMENT,duration,0,10,7,207,81,4000,0,2,2,0,1,"
                "0,0,0,0,6000,100,10.000\n"
                "2,10,20,ue-v2,107,USRP_RX_RECV,duration,1,10,8,208,82,281474976710660,0,2,2,0,4,"
                "512,512,1,0,7000,100,10.000\n"
                "2,11,20,ue-v2,108,UE_TX_DEADLINE_COMPUTE,instant,1,10,9,209,82,281474976710661,0,2,2,0,1,"
                "100000,107680,7680,1999000000,7200,0,0.000\n"
                "2,12,21,worker-a,109,UE_TX_DEADLINE_CHECK,instant,1,10,9,209,82,562949953421315,0,2,2,0,3,"
                "2000010000,2000000000,10000,0,17200,0,0.000\n",
            )
            write_text(
                v2 / "event_catalog.csv",
                "schema_version,event_id,event_name,role,subsystem,event_class,default_kind,detail_level,"
                "aux0_name,aux0_unit,aux1_name,aux1_unit,aux2_name,aux2_unit,aux3_name,aux3_unit,flags_name\n"
                "2,1,UE_SLOT_LOOP,nrUE,orchestration,loop,duration,boundary,duration_rx_to_tx,slot,"
                "read_samples,sample,write_samples,sample,timing_advance,sample,variant\n"
                "2,2,UE_RF_READ,nrUE,radio,io,duration,boundary,requested_samples,sample,antenna_count,"
                "count,returned_samples,sample,device_timestamp,sample,\n"
                "2,17,UE_TX_DEADLINE_MISS,nrUE,timing,deadline,instant,boundary,current_time,us,"
                "deadline,us,lateness,us,,,\n"
                "2,100,TEST_PARALLEL_ROOT,nrUE,test,root,duration,stage,,,,,,,,,\n"
                "2,101,TEST_PARALLEL_CHILD_A,nrUE,test,worker,duration,stage,,,,,,,,,\n"
                "2,102,TEST_PARALLEL_CHILD_B,nrUE,test,worker,duration,stage,,,,,,,,,\n"
                "2,103,TEST_DISPATCH,nrUE,test,dispatch,duration,stage,,,,,,,,,\n"
                "2,104,TEST_ASYNC_WORKER,nrUE,test,worker,duration,stage,,,,,,,,,\n"
                "2,105,TEST_ORPHAN,nrUE,test,worker,duration,stage,,,,,,,,,\n"
                "2,106,LDPC_DECODER_SEGMENT,nrUE/gNB,ldpc_decoder,segment,duration,stage,"
                "transport_block_index,index,segment,index,segments,count,transport_block,bit,decode_success\n"
                "2,107,USRP_RX_RECV,nrUE,radio,io,duration,transport,requested_samples,sample,"
                "returned_samples,sample,channel_count,count,error_code,errno,io_status\n"
                "2,108,UE_TX_DEADLINE_COMPUTE,nrUE,timing,deadline,instant,boundary,"
                "radio_anchor_timestamp,sample,radio_deadline_timestamp,sample,samples_per_subframe,sample,"
                "anchor_monotonic_raw,ns,deadline_flags\n"
                "2,109,UE_TX_DEADLINE_CHECK,nrUE,timing,deadline,instant,boundary,current_monotonic_raw,ns,"
                "deadline_monotonic_raw,ns,signed_lateness,ns,error_code,errno,deadline_flags\n",
            )
            write_text(
                v2 / "drops.csv",
                "thread_index,tid,thread_name,dropped_records,span_stack_overflows,"
                "span_stack_mismatches,counter_regressions\n"
                "0,20,ue-v2,1,2,3,4\n"
                "1,21,worker-a,0,0,0,0\n"
                "2,22,worker-b,0,0,0,0\n",
            )
            write_text(
                v2 / "sync.csv",
                "realtime_ns,monotonic_raw_ns,tick\n3000000000,3,1000\n4000000000,4,1100\n",
            )

            subprocess.run(
                [sys.executable, str(ANALYZER), str(root), "--output-dir", str(output)],
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ANALYZER),
                    str(root),
                    "--output-profile",
                    "full",
                    "--output-dir",
                    str(explicit_full_output),
                ],
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ANALYZER),
                    str(root),
                    "--output-profile",
                    "publication",
                    "--output-dir",
                    str(publication_output),
                ],
                check=True,
            )

            summary = read_rows(output / "summary.csv")
            v1_rf = next(
                row for row in summary if row["profile_dir"] == str(v1) and row["event_name"] == "UE_RF_READ"
            )
            self.assertEqual(v1_rf["schema_version"], "1")
            self.assertEqual(v1_rf["event_kind"], "unknown")
            self.assertEqual(v1_rf["cpu_observed_count"], "0")
            self.assertEqual(v1_rf["drops_total"], "3")
            self.assertEqual(v1_rf["span_stack_overflows"], "")
            self.assertEqual(v1_rf["span_stack_mismatches"], "")
            self.assertEqual(v1_rf["counter_regressions"], "")
            self.assertEqual(v1_rf["drop_diagnostics_status"], "legacy_partial")

            v2_rf = next(
                row for row in summary if row["profile_dir"] == str(v2) and row["event_name"] == "UE_RF_READ"
            )
            self.assertEqual(v2_rf["schema_version"], "2")
            self.assertEqual(v2_rf["subsystem"], "radio")
            self.assertEqual(v2_rf["correlated_count"], "1")
            self.assertEqual(v2_rf["parented_count"], "1")
            self.assertEqual(v2_rf["absolute_slot_count"], "1")
            self.assertEqual(v2_rf["cpu_observed_count"], "1")
            self.assertEqual(v2_rf["cpu_migrations"], "1")
            self.assertEqual(float(v2_rf["cpu_migration_rate"]), 1.0)
            self.assertEqual(v2_rf["span_stack_overflows"], "2")
            self.assertEqual(v2_rf["span_stack_mismatches"], "3")
            self.assertEqual(v2_rf["counter_regressions"], "4")
            self.assertEqual(v2_rf["drop_diagnostics_status"], "recorded")
            self.assertEqual(
                v2_rf["profile_coverage_status"],
                "known_record_drops;known_span_stack_overflow;"
                "known_span_stack_mismatch;known_counter_regression",
            )

            ldpc = next(
                row
                for row in summary
                if row["profile_dir"] == str(v2) and row["event_name"] == "LDPC_DECODER_SEGMENT"
            )
            self.assertEqual(ldpc["event_role"], "nrUE")

            deadlines = read_rows(output / "deadline_misses.csv")
            v1_deadline = next(row for row in deadlines if row["profile_dir"] == str(v1))
            v2_deadline = next(row for row in deadlines if row["profile_dir"] == str(v2))
            self.assertEqual(v1_deadline["absolute_slot"], "-1")
            self.assertEqual(v1_deadline["cpu_start"], "-1")
            self.assertEqual(v2_deadline["correlation_id"], "77")
            self.assertEqual(v2_deadline["parent_id"], "281474976710657")

            deadline_checks = read_rows(output / "deadline_checks.csv")
            self.assertEqual(len(deadline_checks), 1)
            deadline_check = deadline_checks[0]
            self.assertEqual(deadline_check["status"], "ok")
            self.assertEqual(deadline_check["compute_status"], "paired")
            self.assertEqual(deadline_check["radio_anchor_source"], "USRP_RX_RECV")
            self.assertEqual(deadline_check["radio_anchor_match"], "standalone")
            self.assertEqual(deadline_check["reconstructed_offset_tick"], "10000")
            self.assertEqual(deadline_check["reconstructed_deadline_tick"], "17100")
            self.assertEqual(deadline_check["reconstructed_signed_lateness_tick"], "100")
            self.assertAlmostEqual(float(deadline_check["runtime_signed_lateness_us"]), 10.0)
            self.assertAlmostEqual(float(deadline_check["reconstructed_signed_lateness_us"]), 10.0)
            self.assertEqual(deadline_check["classification_agreement"], "1")
            self.assertAlmostEqual(float(deadline_check["reconstruction_minus_runtime_lateness_us"]), 0.0)

            deadline_summaries = {row["run_id"]: row for row in read_rows(output / "deadline_summary.csv")}
            self.assertEqual(deadline_summaries["v1"]["status"], "legacy_only")
            self.assertEqual(deadline_summaries["v2"]["status"], "ok")
            self.assertEqual(deadline_summaries["v2"]["compute_events"], "1")
            self.assertEqual(deadline_summaries["v2"]["check_events"], "1")
            self.assertEqual(deadline_summaries["v2"]["classification_agreements"], "1")
            self.assertEqual(
                deadline_summaries["v2"]["profile_coverage_status"],
                "known_record_drops;known_span_stack_overflow;"
                "known_span_stack_mismatch;known_counter_regression",
            )

            migrations = read_rows(output / "migrations.csv")
            self.assertEqual(len(migrations), 1)
            self.assertEqual(migrations[0]["event_name"], "UE_RF_READ")
            self.assertEqual(migrations[0]["cpu_start"], "1")
            self.assertEqual(migrations[0]["cpu_end"], "2")

            runs = read_rows(output / "runs.csv")
            v2_run = next(row for row in runs if row["run_id"] == "v2")
            self.assertEqual(v2_run["event_record_size_bytes"], "120")
            self.assertEqual(v2_run["span_stack_overflows"], "2")
            self.assertEqual(v2_run["span_stack_mismatches"], "3")
            self.assertEqual(v2_run["counter_regressions"], "4")
            self.assertEqual(v2_run["drop_diagnostics_status"], "recorded")
            self.assertEqual(v2_run["lifecycle_status"], "clean")
            self.assertEqual(v2_run["duration_scope"], "complete")

            expected_outputs = {
                "summary.csv",
                "by_thread.csv",
                "deadline_misses.csv",
                "deadline_checks.csv",
                "deadline_summary.csv",
                "migrations.csv",
                "runs.csv",
                "pairs.csv",
                "host_summary.csv",
                "host_metrics_summary.csv",
                "hierarchy.csv",
                "exclusive_summary.csv",
                "hierarchy_anomalies.csv",
                "hierarchy_integrity.csv",
                "profiler_primitive_overhead_summary.csv",
                "pmu_availability_summary.csv",
                "pmu_summary.csv",
                "pmu_quality.csv",
                "thread_scheduler_summary.csv",
                "kernel_interference_summary.csv",
                "transport_summary.csv",
                "transport_faults.csv",
                "collection_overhead_summary.csv",
                "archive_integrity.csv",
                "external_sources.csv",
                "perf_stat_summary.csv",
                "campaign_runs.csv",
                "campaign_completeness.csv",
                "observer_effect_summary.csv",
                "correlations.csv",
                "clock_quality.csv",
                "event_timeline.csv",
                "analysis_inputs.csv",
                "analysis_provenance.csv",
                "analysis_manifest.csv",
            }
            self.assertEqual(expected_outputs, {path.name for path in output.iterdir()})
            self.assertEqual(expected_outputs, {path.name for path in explicit_full_output.iterdir()})
            invocation_dependent_outputs = {
                "analysis_provenance.csv",
                "analysis_manifest.csv",
            }
            for filename in expected_outputs - invocation_dependent_outputs:
                self.assertEqual(
                    (output / filename).read_bytes(),
                    (explicit_full_output / filename).read_bytes(),
                    f"default and explicit full differ for {filename}",
                )

            expected_publication_outputs = (
                expected_outputs - {"event_timeline.csv", "hierarchy.csv"}
            ) | {"causal_edges_summary.csv"}
            self.assertEqual(
                expected_publication_outputs,
                {path.name for path in publication_output.iterdir()},
            )
            for filename in expected_outputs - {
                "event_timeline.csv",
                "hierarchy.csv",
                "hierarchy_anomalies.csv",
                "analysis_provenance.csv",
                "analysis_manifest.csv",
            }:
                self.assertEqual(
                    (output / filename).read_bytes(),
                    (publication_output / filename).read_bytes(),
                    f"full/publication canonical output differs for {filename}",
                )

            publication_anomalies = read_rows(publication_output / "hierarchy_anomalies.csv")
            self.assertEqual(
                {(row["event_name"], row["relation"]) for row in publication_anomalies},
                {("TEST_ORPHAN", "missing_parent")},
            )
            causal_edges = read_rows(publication_output / "causal_edges_summary.csv")
            self.assertEqual(len(causal_edges), 1)
            self.assertEqual(causal_edges[0]["parent_event_name"], "TEST_DISPATCH")
            self.assertEqual(causal_edges[0]["event_name"], "TEST_ASYNC_WORKER")
            self.assertEqual(causal_edges[0]["relation"], "causal_noncontained")
            self.assertEqual(causal_edges[0]["absolute_slot_delta_from_parent"], "0")
            self.assertEqual(causal_edges[0]["temporal_shape"], "starts_after_parent")
            self.assertEqual(causal_edges[0]["count"], "1")
            self.assertAlmostEqual(float(causal_edges[0]["boundary_distance_p50_us"]), 9.0)

            publication_manifest = {
                row["artifact"]: row
                for row in read_rows(publication_output / "analysis_manifest.csv")
            }
            for omitted in ("event_timeline.csv", "hierarchy.csv"):
                self.assertEqual(
                    publication_manifest[omitted]["status"],
                    "omitted_by_output_profile",
                )
                self.assertEqual(publication_manifest[omitted]["sha256"], "")
            summary_manifest = publication_manifest["summary.csv"]
            self.assertEqual(summary_manifest["status"], "generated")
            self.assertEqual(
                int(summary_manifest["row_count"]),
                len(read_rows(publication_output / "summary.csv")),
            )
            self.assertEqual(
                int(summary_manifest["size_bytes"]),
                (publication_output / "summary.csv").stat().st_size,
            )
            self.assertEqual(
                summary_manifest["sha256"],
                hashlib.sha256((publication_output / "summary.csv").read_bytes()).hexdigest(),
            )
            provenance_manifest = publication_manifest["analysis_provenance.csv"]
            self.assertEqual(provenance_manifest["status"], "generated")
            self.assertEqual(
                provenance_manifest["sha256"],
                hashlib.sha256(
                    (publication_output / "analysis_provenance.csv").read_bytes()
                ).hexdigest(),
            )
            provenance = {
                (row["record_type"], row["name"]): row
                for row in read_rows(publication_output / "analysis_provenance.csv")
            }
            self.assertEqual(
                provenance[("invocation", "output_profile")]["value"],
                "publication",
            )
            self.assertEqual(
                json.loads(provenance[("invocation", "input_arguments_json")]["value"]),
                [str(root)],
            )
            self.assertEqual(
                json.loads(provenance[("invocation", "event_filter_json")]["value"]),
                [],
            )
            self.assertEqual(
                provenance[("invocation", "python_version")]["value"],
                sys.version,
            )
            expected_modules = {
                "oai_profile_analyze",
                "oai_profile_clock",
                "oai_profile_deadlines",
                "oai_profile_reports",
                "oai_profile_archive",
            }
            module_rows = {
                name: row
                for (record_type, name), row in provenance.items()
                if record_type == "module"
            }
            self.assertEqual(set(module_rows), expected_modules)
            for row in module_rows.values():
                module_path = Path(row["path"])
                self.assertTrue(module_path.is_file())
                self.assertEqual(
                    row["sha256"],
                    hashlib.sha256(module_path.read_bytes()).hexdigest(),
                )
            for report in output.iterdir():
                with report.open(newline="") as stream:
                    header = next(csv.reader(stream))
                self.assertEqual(len(header), len(set(header)), f"duplicate columns in {report.name}")

            timeline = read_rows(output / "event_timeline.csv")
            self.assertEqual(len(timeline), 15)
            mapped_root = next(
                row
                for row in timeline
                if row["profile_dir"] == str(v2)
                and row["event_name"] == "UE_SLOT_LOOP"
            )
            self.assertEqual(mapped_root["start_mapping_position"], "anchor_exact")
            self.assertEqual(mapped_root["end_mapping_position"], "anchor_exact")
            self.assertEqual(mapped_root["start_monotonic_raw_ns_estimate"], "3")
            self.assertEqual(mapped_root["end_monotonic_raw_ns_estimate"], "4")
            self.assertEqual(mapped_root["mapped_duration_ns"], "1")
            self.assertEqual(mapped_root["duration_difference_ns"], "-9999")
            self.assertEqual(mapped_root["monotonic_interval_status"], "valid")

            hierarchy = read_rows(output / "hierarchy.csv")
            parallel_root = next(row for row in hierarchy if row["event_name"] == "TEST_PARALLEL_ROOT")
            self.assertEqual(parallel_root["duration_children"], "2")
            self.assertAlmostEqual(float(parallel_root["child_duration_sum_us"]), 100.0)
            self.assertAlmostEqual(float(parallel_root["child_interval_union_us"]), 70.0)
            self.assertAlmostEqual(float(parallel_root["child_overlap_us"]), 30.0)
            self.assertAlmostEqual(float(parallel_root["exclusive_us"]), 30.0)
            self.assertEqual(parallel_root["exclusive_valid"], "1")

            dispatch = next(row for row in hierarchy if row["event_name"] == "TEST_DISPATCH")
            self.assertEqual(dispatch["noncontained_duration_children"], "1")
            self.assertEqual(dispatch["exclusive_valid"], "0")

            anomalies = read_rows(output / "hierarchy_anomalies.csv")
            anomaly_relations = {(row["event_name"], row["relation"]) for row in anomalies}
            self.assertIn(("TEST_ASYNC_WORKER", "causal_noncontained"), anomaly_relations)
            self.assertIn(("TEST_ORPHAN", "missing_parent"), anomaly_relations)

            integrity = next(
                row for row in read_rows(output / "hierarchy_integrity.csv") if row["profile_dir"] == str(v2)
            )
            self.assertEqual(integrity["schema2_records"], "13")
            self.assertEqual(integrity["duration_records"], "10")
            self.assertEqual(integrity["instant_records"], "3")
            self.assertEqual(integrity["causal_noncontained_edges"], "1")
            self.assertEqual(integrity["missing_parent_edges"], "1")

            correlations = read_rows(output / "correlations.csv")
            parallel_correlation = next(row for row in correlations if row["correlation_id"] == "78")
            self.assertEqual(parallel_correlation["event_count"], "3")
            self.assertEqual(parallel_correlation["thread_count"], "3")
            self.assertAlmostEqual(float(parallel_correlation["elapsed_us"]), 100.0)

            subprocess.run(
                [
                    sys.executable,
                    str(ANALYZER),
                    str(root),
                    "--event",
                    "LDPC_DECODER_SEGMENT",
                    "--output-dir",
                    str(filtered_output),
                ],
                check=True,
            )
            filtered_summary = read_rows(filtered_output / "summary.csv")
            self.assertEqual({row["event_name"] for row in filtered_summary}, {"LDPC_DECODER_SEGMENT"})
            filtered_hierarchy = read_rows(filtered_output / "hierarchy.csv")
            self.assertIn("TEST_PARALLEL_ROOT", {row["event_name"] for row in filtered_hierarchy})

            filtered_timeline = read_rows(filtered_output / "event_timeline.csv")
            self.assertEqual(len(filtered_timeline), 15)
            self.assertIn("TEST_PARALLEL_ROOT", {row["event_name"] for row in filtered_timeline})
            self.assertEqual(len(read_rows(filtered_output / "deadline_checks.csv")), 1)

            filtered_transport = read_rows(filtered_output / "transport_summary.csv")
            self.assertIn("UE_RF_READ", {row["event_name"] for row in filtered_transport})
            filtered_provenance = {
                (row["record_type"], row["name"]): row
                for row in read_rows(filtered_output / "analysis_provenance.csv")
            }
            self.assertEqual(
                json.loads(
                    filtered_provenance[("invocation", "event_filter_json")][
                        "value"
                    ]
                ),
                ["LDPC_DECODER_SEGMENT"],
            )

    def test_array_backed_statistics_are_exact(self) -> None:
        stats = GroupStats()
        for value in (0.125, 0.5, 1.25, 8.0):
            stats.add(
                {
                    "duration_us": str(value),
                    "schema_version": "2",
                    "event_role": "nrUE",
                    "subsystem": "test",
                    "event_class": "duration",
                    "event_kind": "duration",
                    "detail_level": "stage",
                    "correlation_id": "1",
                    "parent_id": "0",
                    "absolute_slot": "1",
                    "cpu_start": "0",
                    "cpu_end": "0",
                    "cpu_migrated": "0",
                }
            )
        self.assertIsInstance(stats.durations_us, array)
        self.assertEqual(stats.durations_us.typecode, "d")
        self.assertEqual(stats.durations_us.itemsize, 8)
        for name in ("aux0_values", "aux1_values", "aux2_values", "aux3_values"):
            self.assertFalse(hasattr(stats, name))
        row = build_rows(
            {("/profile", "TEST"): stats},
            {"/profile": {"process_name": "nr-uesoftmodem"}},
            {
                "/profile": DropDiagnostics(
                    status="recorded",
                    row_count=1,
                    dropped_records=0,
                    span_stack_overflows=0,
                    span_stack_mismatches=0,
                    counter_regressions=0,
                )
            },
            {"/profile": CsvDiagnostics(status="recorded", row_count=1)},
            {"/profile": CsvDiagnostics(status="recorded", row_count=1)},
            {"/profile": "complete"},
        )[0]
        values = [0.125, 0.5, 1.25, 8.0]
        average = sum(values) / len(values)
        expected_stdev = math.sqrt(
            sum((value - average) ** 2 for value in values) / (len(values) - 1)
        )
        self.assertEqual(row["count"], 4)
        self.assertEqual(row["min_us"], 0.125)
        self.assertEqual(row["max_us"], 8.0)
        self.assertEqual(row["mean_us"], average)
        self.assertAlmostEqual(float(row["p50_us"]), 0.875)
        self.assertAlmostEqual(float(row["p99_us"]), 7.7975)
        self.assertAlmostEqual(float(row["stdev_us"]), expected_stdev)

        exclusive = ExclusiveStats()
        exclusive.inclusive_us.extend(values)
        exclusive.exclusive_us.extend((0.125, 0.25, 1.0, 4.0))
        exclusive.child_union_us.extend((0.0, 0.25, 0.25, 4.0))
        exclusive.total_count = 4
        exclusive.valid_count = 4
        for samples in (
            exclusive.inclusive_us,
            exclusive.exclusive_us,
            exclusive.child_union_us,
        ):
            self.assertIsInstance(samples, array)
            self.assertEqual(samples.typecode, "d")
            self.assertEqual(samples.itemsize, 8)
        exclusive_row = build_exclusive_summary({("/profile", "TEST"): exclusive})[0]
        self.assertEqual(exclusive_row["inclusive_mean_us"], average)
        self.assertAlmostEqual(float(exclusive_row["inclusive_p50_us"]), 0.875)
        self.assertAlmostEqual(float(exclusive_row["exclusive_mean_us"]), 1.34375)
        self.assertAlmostEqual(float(exclusive_row["child_union_mean_us"]), 1.125)

    def test_publication_hard_anomalies_and_profile_isolation(self) -> None:
        event_fields = [
            "schema_version",
            "seq",
            "tid",
            "thread_name",
            "event_id",
            "event_name",
            "event_kind",
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
            "aux1",
            "aux2",
            "aux3",
            "start_tick",
            "duration_tick",
            "duration_us",
        ]

        def event(
            seq: int,
            name: str,
            span_id: int,
            parent_id: int,
            correlation_id: int,
            start_tick: int,
            duration_tick: int = 10,
        ) -> dict[str, object]:
            return {
                "schema_version": 2,
                "seq": seq,
                "tid": 10 + seq,
                "thread_name": f"thread-{seq}",
                "event_id": 100 + seq,
                "event_name": name,
                "event_kind": "duration",
                "nesting_depth": int(parent_id != 0),
                "frame": 1,
                "slot": 1,
                "absolute_slot": 21,
                "correlation_id": correlation_id,
                "span_id": span_id,
                "parent_id": parent_id,
                "cpu_start": 0,
                "cpu_end": 0,
                "cpu_migrated": 0,
                "flags": 0,
                "aux0": 0,
                "aux1": 0,
                "aux2": 0,
                "aux3": 0,
                "start_tick": start_tick,
                "duration_tick": duration_tick,
                "duration_us": duration_tick / 10.0,
            }

        with tempfile.TemporaryDirectory(prefix="oai-profile-hierarchy-isolation-") as temporary:
            root = Path(temporary)
            profile_a = root / "a_nrUE"
            profile_b = root / "b_nrUE"
            output = root / "analysis"
            profile_a.mkdir()
            profile_b.mkdir()
            for profile, role, process in (
                (profile_a, "nrUE", "nr-uesoftmodem"),
                (profile_b, "gNB", "nr-softmodem"),
            ):
                write_text(
                    profile / "metadata.txt",
                    "schema_version=2\ncounter_hz=10000000\n"
                    f"process_name={process}\nrole={role}\n"
                    f"run_id={profile.name}\n",
                )
            write_csv_rows(
                profile_a / "events.csv",
                event_fields,
                [
                    event(1, "A_ROOT", 1, 0, 10, 100, 100),
                    event(2, "A_CHILD", 2, 1, 10, 110),
                    event(3, "A_ASYNC", 3, 1, 10, 250),
                    event(4, "A_MISSING", 4, 999, 10, 130),
                    event(5, "A_CORRELATION_MISMATCH", 5, 1, 11, 140),
                    event(6, "A_DUPLICATE_FIRST", 6, 1, 10, 150),
                    event(7, "A_DUPLICATE_SECOND", 6, 1, 10, 160),
                ],
            )
            write_csv_rows(
                profile_b / "events.csv",
                event_fields,
                [
                    event(1, "B_ROOT", 1, 0, 20, 100, 100),
                    event(2, "B_CHILD", 2, 1, 20, 110),
                    event(3, "UE_TX_DEADLINE_MISS", 3, 0, 20, 300, 0),
                ],
            )

            subprocess.run(
                [
                    sys.executable,
                    str(ANALYZER),
                    str(root),
                    "--output-profile",
                    "publication",
                    "--output-dir",
                    str(output),
                ],
                check=True,
            )

            anomalies = read_rows(output / "hierarchy_anomalies.csv")
            anomalies_a = [row for row in anomalies if row["profile_dir"] == str(profile_a)]
            anomalies_b = [row for row in anomalies if row["profile_dir"] == str(profile_b)]
            self.assertEqual(
                {row["relation"] for row in anomalies_a},
                {"missing_parent", "correlation_mismatch", "duplicate_span_id"},
            )
            self.assertEqual(anomalies_b, [])
            self.assertNotIn("causal_noncontained", {row["relation"] for row in anomalies})

            causal = read_rows(output / "causal_edges_summary.csv")
            self.assertEqual(len(causal), 1)
            self.assertEqual(causal[0]["profile_dir"], str(profile_a))
            self.assertEqual(causal[0]["parent_event_name"], "A_ROOT")
            self.assertEqual(causal[0]["event_name"], "A_ASYNC")

            integrity = {
                row["profile_dir"]: row
                for row in read_rows(output / "hierarchy_integrity.csv")
            }
            self.assertEqual(integrity[str(profile_a)]["duplicate_span_ids"], "1")
            self.assertEqual(integrity[str(profile_b)]["duplicate_span_ids"], "0")
            self.assertEqual(integrity[str(profile_b)]["temporally_contained_edges"], "1")
            correlation_profiles = {
                row["profile_dir"] for row in read_rows(output / "correlations.csv")
            }
            self.assertEqual(correlation_profiles, {str(profile_a), str(profile_b)})
            self.assertIn(
                "UE_TX_DEADLINE_MISS",
                {
                    row["event_name"]
                    for row in read_rows(output / "summary.csv")
                    if row["profile_dir"] == str(profile_b)
                },
            )
            self.assertNotIn(
                str(profile_b),
                {
                    row["profile_dir"]
                    for row in read_rows(output / "deadline_misses.csv")
                },
            )
            self.assertNotIn(
                str(profile_b),
                {
                    row["profile_dir"]
                    for row in read_rows(output / "deadline_summary.csv")
                },
            )

    def test_output_refusal_and_malformed_evidence_publishing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-incomplete-") as temporary:
            root = Path(temporary)
            profile = root / "bad_nrUE"
            profile.mkdir()
            write_text(
                profile / "metadata.txt",
                "schema_version=2\ncounter_hz=10000000\n"
                "process_name=nr-uesoftmodem\nrole=nrUE\nrun_id=bad\n",
            )
            write_text(
                profile / "events.csv",
                "schema_version,seq,tid,thread_name,event_name,event_kind,span_id,"
                "start_tick,duration_tick,duration_us\n"
                "2,1,1,bad,TEST,duration,1,100,10,not-a-number\n",
            )

            existing = root / "existing"
            existing.mkdir()
            refusal = subprocess.run(
                [
                    sys.executable,
                    str(ANALYZER),
                    str(profile),
                    "--output-dir",
                    str(existing),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(refusal.returncode, 0)
            self.assertEqual(list(existing.iterdir()), [])
            self.assertEqual(list(root.glob(".existing.partial-*")), [])

            requested = root / "analysis"
            publication = subprocess.run(
                [
                    sys.executable,
                    str(ANALYZER),
                    str(profile),
                    "--output-profile",
                    "publication",
                    "--output-dir",
                    str(requested),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(publication.returncode, 0, publication.stderr)
            self.assertTrue(requested.is_dir())
            self.assertEqual(list(root.glob(".analysis.partial-*")), [])
            self.assertFalse((requested / "ANALYSIS_INCOMPLETE.txt").exists())
            run = read_rows(requested / "runs.csv")[0]
            self.assertEqual(run["event_stream_status"], "malformed")
            self.assertEqual(run["event_stream_rows"], "0")
            self.assertEqual(run["event_catalog_status"], "missing")
            self.assertEqual(run["drop_diagnostics_status"], "missing")
            self.assertEqual(
                run["profile_coverage_status"],
                "lifecycle_unknown;event_stream_malformed;"
                "event_catalog_missing;drop_diagnostics_missing",
            )

    def test_writer_initialization_and_post_rename_failures_are_marked(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-transaction-") as temporary:
            root = Path(temporary)
            profile = root / "nrUE"
            profile.mkdir()
            write_text(
                profile / "metadata.txt",
                "schema_version=2\ncounter_hz=10000000\n"
                "process_name=nr-uesoftmodem\nrole=nrUE\nrun_id=transaction\n",
            )
            write_text(
                profile / "events.csv",
                "schema_version,seq,tid,thread_name,event_name,event_kind,span_id,"
                "start_tick,duration_tick,duration_us\n"
                "2,1,1,test,TEST,duration,1,100,10,1.0\n",
            )

            sink = analyze_module._HashingTextSink(root / "constructor.csv")
            with mock.patch.object(
                analyze_module,
                "_HashingTextSink",
                return_value=sink,
            ), mock.patch.object(
                sink,
                "write",
                side_effect=OSError("injected header failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected header failure"):
                    analyze_module.TrackedCsvWriter(
                        root / "ignored.csv",
                        ["value"],
                    )
            self.assertTrue(sink._closed)
            self.assertTrue(sink._stream.closed)

            real_writer = analyze_module.TrackedCsvWriter
            opened_writers: list[analyze_module.TrackedCsvWriter] = []

            def fail_second_writer(
                path: Path,
                fields: list[str],
            ) -> analyze_module.TrackedCsvWriter:
                if opened_writers:
                    raise KeyboardInterrupt("injected writer initialization interrupt")
                writer = real_writer(path, fields)
                opened_writers.append(writer)
                return writer

            initialization_output = root / "initialization_failure"
            with mock.patch.object(
                analyze_module,
                "TrackedCsvWriter",
                side_effect=fail_second_writer,
            ), mock.patch.object(
                sys,
                "argv",
                [
                    str(ANALYZER),
                    str(profile),
                    "--output-dir",
                    str(initialization_output),
                ],
            ):
                with self.assertRaisesRegex(
                    KeyboardInterrupt,
                    "injected writer initialization interrupt",
                ):
                    analyze_module.main()
            self.assertEqual(len(opened_writers), 1)
            self.assertTrue(opened_writers[0].closed)
            self.assertFalse(initialization_output.exists())
            initialization_partials = list(
                root.glob(".initialization_failure.partial-*")
            )
            self.assertEqual(len(initialization_partials), 1)
            initialization_marker = (
                initialization_partials[0] / "ANALYSIS_INCOMPLETE.txt"
            ).read_text()
            self.assertIn(
                "publication_state=unpublished_partial",
                initialization_marker,
            )
            self.assertIn(
                "error_type=KeyboardInterrupt",
                initialization_marker,
            )

            fsync_calls = 0
            real_fsync_directory = analyze_module.fsync_directory

            def fail_parent_fsync(path: Path) -> None:
                nonlocal fsync_calls
                fsync_calls += 1
                if fsync_calls == 2:
                    raise OSError("injected parent fsync failure")
                real_fsync_directory(path)

            published_output = root / "published_failure"
            with mock.patch.object(
                analyze_module,
                "fsync_directory",
                side_effect=fail_parent_fsync,
            ), mock.patch.object(
                sys,
                "argv",
                [
                    str(ANALYZER),
                    str(profile),
                    "--output-dir",
                    str(published_output),
                ],
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "injected parent fsync failure",
                ):
                    analyze_module.main()
            self.assertTrue(published_output.is_dir())
            published_marker = (
                published_output / "ANALYSIS_INCOMPLETE.txt"
            ).read_text()
            self.assertIn(
                "publication_state=published_incomplete",
                published_marker,
            )
            self.assertIn("error_type=OSError", published_marker)
            self.assertEqual(list(root.glob(".published_failure.partial-*")), [])


    def test_publication_reports_and_campaign_only_runs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-publication-test-") as temporary:
            root = Path(temporary)
            disabled_gnb = root / "disabled_gNB"
            disabled_ue = root / "disabled_nrUE"
            profiled_gnb = root / "profiled_gNB"
            profiled_ue = root / "profiled_nrUE"
            output = root / "analysis"
            disabled_output = root / "disabled_analysis"

            write_campaign_run(disabled_gnb, "gNB", "disabled", 10, False)
            write_campaign_run(disabled_ue, "nrUE", "disabled", 10, False)
            write_campaign_run(profiled_gnb, "gNB", "in-process", 11, True)
            write_campaign_run(
                profiled_ue,
                "nrUE",
                "in-process",
                11,
                True,
                sidecar_tool="perf_stat",
                sidecar_status="registered",
                sidecar_artifact=str(profiled_ue / "sidecars/perf_stat.csv"),
            )
            write_profile_metadata(profiled_gnb, "gNB", "in-process", 11)
            write_profile_metadata(profiled_ue, "nrUE", "in-process", 11)
            host_rows = []
            for index, acquisition_us in enumerate((100, 200, 300)):
                monotonic_ns = 500_000_000 + index * 1_000_000_000
                acquisition_ns = acquisition_us * 1000
                tick = 1000 + index * 10_000_000
                acquisition_tick = acquisition_ns // 100
                writer_cpu_end = 3 if index == 1 else 2
                host_rows.append(
                    {
                        "realtime_ns": 1_000_000_000 + index * 1_000_000_000,
                        "monotonic_raw_ns": monotonic_ns,
                        "tick": tick,
                        "writer_cpu": 2,
                        "thermal_zone0_millicelsius": (60_000, 70_000, 80_000)[index],
                        "thermal_max_millicelsius": (65_000, 75_000, 85_000)[index],
                        "thermal_samples": 2,
                        "rpi_throttled_valid": 1,
                        "rpi_throttled_raw": (0, 1, 65_537)[index],
                        "cpu_frequency_samples": 4,
                        "cpu_frequency_min_khz": (1_800_000, 1_700_000, 1_600_000)[index],
                        "cpu_frequency_avg_khz": (2_400_000, 2_200_000, 2_000_000)[index],
                        "cpu_frequency_max_khz": 2_400_000,
                        "cpu_busy_percent": (-1, 40, 60)[index],
                        "load1": (0.5, 0.6, 0.7)[index],
                        "load5": (0.4, 0.5, 0.6)[index],
                        "load15": (0.3, 0.4, 0.5)[index],
                        "mem_available_kb": (10_000_000, 9_000_000, 8_000_000)[index],
                        "swap_free_kb": (4_000_000, 3_900_000, 3_800_000)[index],
                        "process_rss_kb": (100_000, 200_000, 300_000)[index],
                        "process_maxrss_kb": (1_000, 1_100, 1_200)[index],
                        "process_user_us": (100_000, 300_000, 500_000)[index],
                        "process_system_us": (50_000, 150_000, 250_000)[index],
                        "voluntary_context_switches": (1, 4, 7)[index],
                        "involuntary_context_switches": (0, 1, 3)[index],
                        "minor_faults": (10, 20, 40)[index],
                        "major_faults": (0, 1, 1)[index],
                        "block_input_ops": (0, 2, 4)[index],
                        "block_output_ops": (0, 4, 10)[index],
                        "end_monotonic_raw_ns": monotonic_ns + acquisition_ns,
                        "end_tick": tick + acquisition_tick,
                        "writer_cpu_end": writer_cpu_end,
                        "writer_cpu_migrated": int(writer_cpu_end != 2),
                        "acquisition_duration_monotonic_raw_ns": acquisition_ns,
                        "acquisition_duration_tick": acquisition_tick,
                        "acquisition_duration_us": acquisition_us,
                        "status": "ok",
                        "getloadavg_count": 3,
                        "getrusage_status": "ok",
                        "error_mask": 0,
                    }
                )
            write_csv_rows(profiled_ue / "host_metrics.csv", HOST_METRIC_FIELDS, host_rows)

            event_fields = (
                "schema_version,seq,tid,thread_name,event_id,event_name,event_kind,nesting_depth,frame,slot,"
                "absolute_slot,correlation_id,span_id,parent_id,cpu_start,cpu_end,cpu_migrated,flags,"
                "aux0,aux1,aux2,aux3,start_tick,duration_tick,duration_us"
            ).split(",")
            write_csv_rows(
                profiled_gnb / "events.csv",
                event_fields,
                [
                    {
                        "schema_version": 2,
                        "seq": 1,
                        "tid": 100,
                        "thread_name": "gnb-rx",
                        "event_id": 1,
                        "event_name": "GNB_RF_READ",
                        "event_kind": "duration",
                        "nesting_depth": 0,
                        "frame": 1,
                        "slot": 1,
                        "absolute_slot": 21,
                        "correlation_id": 1,
                        "span_id": 1,
                        "parent_id": 0,
                        "cpu_start": 1,
                        "cpu_end": 1,
                        "cpu_migrated": 0,
                        "flags": 0,
                        "aux0": 512,
                        "aux1": 1,
                        "aux2": 512,
                        "aux3": 0,
                        "start_tick": 100,
                        "duration_tick": 20,
                        "duration_us": 2.0,
                    }
                ],
            )
            write_csv_rows(
                profiled_ue / "events.csv",
                event_fields,
                [
                    {
                        "schema_version": 2,
                        "seq": 1,
                        "tid": 200,
                        "thread_name": "ue-rx",
                        "event_id": 2,
                        "event_name": "USRP_RX_RECV",
                        "event_kind": "duration",
                        "nesting_depth": 0,
                        "frame": 1,
                        "slot": 1,
                        "absolute_slot": 21,
                        "correlation_id": 1,
                        "span_id": 1,
                        "parent_id": 0,
                        "cpu_start": 2,
                        "cpu_end": 2,
                        "cpu_migrated": 0,
                        "flags": 4,
                        "aux0": 512,
                        "aux1": 512,
                        "aux2": 1,
                        "aux3": 0,
                        "start_tick": 200,
                        "duration_tick": 40,
                        "duration_us": 4.0,
                    },
                    {
                        "schema_version": 2,
                        "seq": 2,
                        "tid": 200,
                        "thread_name": "ue-rx",
                        "event_id": 3,
                        "event_name": "USRP_RX_SHORT_READ",
                        "event_kind": "instant",
                        "nesting_depth": 0,
                        "frame": 1,
                        "slot": 1,
                        "absolute_slot": 21,
                        "correlation_id": 1,
                        "span_id": 2,
                        "parent_id": 0,
                        "cpu_start": 2,
                        "cpu_end": 2,
                        "cpu_migrated": 0,
                        "flags": 4,
                        "aux0": 512,
                        "aux1": 500,
                        "aux2": 1,
                        "aux3": 2,
                        "start_tick": 245,
                        "duration_tick": 0,
                        "duration_us": 0.0,
                    },
                ],
            )
            write_text(
                profiled_gnb / "drops.csv",
                "thread_index,tid,thread_name,dropped_records,span_stack_overflows,"
                "span_stack_mismatches,counter_regressions\n"
                "0,100,gnb-rx,0,0,0,0\n",
            )
            write_text(
                profiled_ue / "drops.csv",
                "thread_index,tid,thread_name,dropped_records,span_stack_overflows,"
                "span_stack_mismatches,counter_regressions\n"
                "0,200,ue-rx,0,0,0,0\n",
            )
            catalog_fields = (
                "schema_version,event_id,event_name,role,subsystem,event_class,default_kind,detail_level,"
                "aux0_name,aux0_unit,aux1_name,aux1_unit,aux2_name,aux2_unit,aux3_name,aux3_unit,flags_name"
            ).split(",")
            write_csv_rows(
                profiled_gnb / "event_catalog.csv",
                catalog_fields,
                [
                    {
                        "schema_version": 2,
                        "event_id": 1,
                        "event_name": "GNB_RF_READ",
                        "role": "gNB",
                        "subsystem": "radio",
                        "event_class": "io",
                        "default_kind": "duration",
                        "detail_level": "boundary",
                    }
                ],
            )
            write_csv_rows(
                profiled_ue / "event_catalog.csv",
                catalog_fields,
                [
                    {
                        "schema_version": 2,
                        "event_id": 2,
                        "event_name": "USRP_RX_RECV",
                        "role": "nrUE/gNB",
                        "subsystem": "rf_usrp",
                        "event_class": "receive",
                        "default_kind": "duration",
                        "detail_level": "kernel",
                        "aux0_name": "requested_samples",
                        "aux0_unit": "count",
                        "aux1_name": "returned_samples",
                        "aux1_unit": "count",
                        "aux2_name": "channels",
                        "aux2_unit": "count",
                    },
                    {
                        "schema_version": 2,
                        "event_id": 3,
                        "event_name": "USRP_RX_SHORT_READ",
                        "role": "nrUE/gNB",
                        "subsystem": "rf_usrp",
                        "event_class": "short_transfer",
                        "default_kind": "instant",
                        "detail_level": "kernel",
                        "aux0_name": "requested_samples",
                        "aux0_unit": "count",
                        "aux1_name": "received_samples",
                        "aux1_unit": "count",
                        "aux2_name": "channels",
                        "aux2_unit": "count",
                        "aux3_name": "error_code",
                        "aux3_unit": "uhd_enum",
                    },
                ],
            )

            primitive_fields = (
                "schema_version,run_id,experiment_id,campaign_id,variant,trial,role,hostname,phase,sample_index,"
                "primitive,event_kind,cpu_start,cpu_end,cpu_migrated,outer_start_tick,outer_end_tick,"
                "outer_duration_tick,outer_duration_us,event_record_expected,event_recorded,event_seq,"
                "event_duration_tick,event_duration_us,drop_delta,status"
            ).split(",")
            primitive_identity = {
                "schema_version": 2,
                "run_id": "in-process-nrUE",
                "experiment_id": "scientific-baseline-in-process-t001",
                "campaign_id": "scientific-baseline",
                "variant": "in-process",
                "trial": 1,
                "role": "nrUE",
                "hostname": "nrue-host",
                "phase": "measurement",
                "cpu_start": 2,
                "cpu_end": 2,
                "status": "ok",
            }
            primitive_rows = []
            for index, duration in enumerate((1.0, 3.0)):
                duration_tick = int(duration * 10)
                outer_start_tick = 1000 + index * 100
                primitive_rows.append(
                    {
                        **primitive_identity,
                        "sample_index": index,
                        "primitive": "counter_pair",
                        "event_kind": "unknown",
                        "cpu_migrated": 0,
                        "outer_start_tick": outer_start_tick,
                        "outer_end_tick": outer_start_tick + duration_tick,
                        "outer_duration_tick": duration_tick,
                        "outer_duration_us": duration,
                        "event_record_expected": 0,
                        "event_recorded": 0,
                        "event_seq": 0,
                        "event_duration_tick": 0,
                        "event_duration_us": 0.0,
                        "drop_delta": 0,
                    }
                )
            for index, duration in enumerate((5.0, 7.0)):
                duration_tick = int(duration * 10)
                outer_start_tick = 2000 + index * 100
                primitive_rows.append(
                    {
                        **primitive_identity,
                        "sample_index": index,
                        "primitive": "span_start_stop",
                        "event_kind": "duration",
                        "cpu_migrated": 0,
                        "outer_start_tick": outer_start_tick,
                        "outer_end_tick": outer_start_tick + duration_tick,
                        "outer_duration_tick": duration_tick,
                        "outer_duration_us": duration,
                        "event_record_expected": 1,
                        "event_recorded": 1,
                        "event_seq": 10 + index,
                        "event_duration_tick": duration_tick - 10,
                        "event_duration_us": duration - 1.0,
                        "drop_delta": 0,
                    }
                )
            write_csv_rows(profiled_ue / "profiler_primitive_overhead.csv", primitive_fields, primitive_rows)

            availability_fields = (
                "schema_version,run_id,experiment_id,campaign_id,role,hostname,thread_index,tid,thread_name,"
                "event_id,event_name,domain,requested,available,status,error_code"
            ).split(",")
            write_csv_rows(
                profiled_ue / "pmu_availability.csv",
                availability_fields,
                [
                    {
                        "schema_version": 2,
                        "run_id": "in-process-nrUE",
                        "experiment_id": "scientific-baseline-in-process-t001",
                        "campaign_id": "scientific-baseline",
                        "role": "nrUE",
                        "hostname": "nrue-host",
                        "thread_index": 0,
                        "tid": 200,
                        "thread_name": "ue-rx",
                        "event_id": 1,
                        "event_name": "cpu_cycles",
                        "domain": "hardware",
                        "requested": 1,
                        "available": 1,
                        "status": "available",
                        "error_code": 0,
                    }
                ],
            )
            pmu_fields = (
                "schema_version,sample_id,realtime_ns,monotonic_raw_ns,tick,run_id,experiment_id,campaign_id,"
                "variant,trial,role,hostname,thread_index,tid,thread_name,target_cpu,event_id,event_name,domain,unit,"
                "raw_value,delta_raw,time_enabled_ns,time_running_ns,delta_enabled_ns,delta_running_ns,scaled_value,"
                "delta_scaled,multiplex_ratio,interval_ns,delta_valid,scaling_valid,status,error_code"
            ).split(",")
            pmu_common = {
                "schema_version": 2,
                "realtime_ns": 1_000_000_000,
                "monotonic_raw_ns": 500_000_000,
                "tick": 100,
                "run_id": "in-process-nrUE",
                "experiment_id": "scientific-baseline-in-process-t001",
                "campaign_id": "scientific-baseline",
                "variant": "in-process",
                "trial": 1,
                "role": "nrUE",
                "hostname": "nrue-host",
                "thread_index": 0,
                "tid": 200,
                "thread_name": "ue-rx",
                "target_cpu": 2,
                "event_id": 1,
                "event_name": "cpu_cycles",
                "domain": "hardware",
                "unit": "count",
                "raw_value": 100,
                "delta_raw": 0,
                "time_enabled_ns": 100,
                "time_running_ns": 100,
                "delta_enabled_ns": 0,
                "delta_running_ns": 0,
                "scaled_value": 100,
                "delta_scaled": 0,
                "multiplex_ratio": 1.0,
                "interval_ns": 0,
                "delta_valid": 0,
                "scaling_valid": 1,
                "status": "warmup",
                "error_code": 0,
            }
            write_csv_rows(
                profiled_ue / "pmu_samples.csv",
                pmu_fields,
                [
                    {**pmu_common, "sample_id": 1},
                    {
                        **pmu_common,
                        "sample_id": 2,
                        "realtime_ns": 2_000_000_000,
                        "monotonic_raw_ns": 1_500_000_000,
                        "tick": 200,
                        "raw_value": 150,
                        "delta_raw": 50,
                        "time_enabled_ns": 1100,
                        "time_running_ns": 600,
                        "delta_enabled_ns": 1000,
                        "delta_running_ns": 500,
                        "scaled_value": 275,
                        "delta_scaled": 100,
                        "multiplex_ratio": 0.5,
                        "interval_ns": 1_000_000_000,
                        "delta_valid": 1,
                        "scaling_valid": 1,
                        "status": "ok",
                    },
                    {
                        **pmu_common,
                        "sample_id": 3,
                        "realtime_ns": 3_000_000_000,
                        "monotonic_raw_ns": 2_500_000_000,
                        "tick": 300,
                        "raw_value": 350,
                        "delta_raw": 200,
                        "time_enabled_ns": 2100,
                        "time_running_ns": 1600,
                        "delta_enabled_ns": 1000,
                        "delta_running_ns": 1000,
                        "scaled_value": 459.375,
                        "delta_scaled": 200,
                        "multiplex_ratio": 1.0,
                        "interval_ns": 1_000_000_000,
                        "delta_valid": 1,
                        "scaling_valid": 1,
                        "status": "ok",
                    },
                ],
            )
            pmu_overhead_fields = (
                "schema_version,sample_id,realtime_ns,monotonic_raw_ns,end_monotonic_raw_ns,"
                "timestamp_uncertainty_ns,run_id,experiment_id,campaign_id,thread_index,"
                "tid,thread_name,duration_tick,duration_us,available_events,active_groups,group_reads,observations,read_errors,counter_status"
            ).split(",")
            write_csv_rows(
                profiled_ue / "pmu_read_overhead.csv",
                pmu_overhead_fields,
                [
                    {
                        "schema_version": 2,
                        "sample_id": 2,
                        "realtime_ns": 2_000_000_000,
                        "monotonic_raw_ns": 1_500_000_000,
                        "end_monotonic_raw_ns": 1_500_000_100,
                        "timestamp_uncertainty_ns": 100,
                        "run_id": "in-process-nrUE",
                        "experiment_id": "scientific-baseline-in-process-t001",
                        "campaign_id": "scientific-baseline",
                        "thread_index": 0,
                        "tid": 200,
                        "thread_name": "ue-rx",
                        "duration_tick": 100,
                        "duration_us": 10.0,
                        "available_events": 1,
                        "active_groups": 1,
                        "group_reads": 1,
                        "observations": 1,
                        "read_errors": 0,
                        "counter_status": "ok",
                    },
                    {
                        "schema_version": 2,
                        "sample_id": 3,
                        "realtime_ns": 3_000_000_000,
                        "monotonic_raw_ns": 2_500_000_000,
                        "end_monotonic_raw_ns": 2_500_000_000,
                        "timestamp_uncertainty_ns": 0,
                        "run_id": "in-process-nrUE",
                        "experiment_id": "scientific-baseline-in-process-t001",
                        "campaign_id": "scientific-baseline",
                        "thread_index": 0,
                        "tid": 200,
                        "thread_name": "ue-rx",
                        "duration_tick": 0,
                        "duration_us": 0.0,
                        "available_events": 1,
                        "active_groups": 1,
                        "group_reads": 0,
                        "observations": 0,
                        "read_errors": 0,
                        "counter_status": "counter_regression",
                    },
                ],
            )

            thread_fields = (
                "schema_version,sample_id,realtime_ns,monotonic_raw_ns,run_id,experiment_id,campaign_id,variant,trial,"
                "role,hostname,thread_index,tid,thread_name,valid_mask,state,cpu,priority,nice,rt_priority,policy,"
                "cpu_frequency_khz,runtime_ns,runqueue_wait_ns,timeslices,minor_faults,major_faults,user_ticks,"
                "system_ticks,voluntary_context_switches,involuntary_context_switches,interval_ns,delta_runtime_ns,"
                "delta_runqueue_wait_ns,delta_timeslices,delta_minor_faults,delta_major_faults,delta_user_ticks,"
                "delta_system_ticks,delta_voluntary_context_switches,delta_involuntary_context_switches,delta_valid,"
                "cpu_changed_since_previous,status,error_code"
            ).split(",")
            thread_common = {"schema_version": 2, "thread_index": 0, "tid": 200, "thread_name": "ue-rx", "valid_mask": 15, "cpu": 2, "cpu_frequency_khz": 2_400_000, "status": "ok", "error_code": 0}
            write_csv_rows(
                profiled_ue / "thread_metrics.csv",
                thread_fields,
                [
                    {**thread_common, "sample_id": 1, "delta_valid": 0},
                    {**thread_common, "sample_id": 2, "interval_ns": 1_000_000_000, "delta_runtime_ns": 400_000_000, "delta_runqueue_wait_ns": 100_000_000, "delta_timeslices": 10, "delta_minor_faults": 2, "delta_major_faults": 1, "delta_voluntary_context_switches": 3, "delta_involuntary_context_switches": 2, "delta_valid": 1, "cpu_changed_since_previous": 1},
                    {**thread_common, "sample_id": 3, "valid_mask": 7, "cpu_frequency_khz": 9_999_999, "delta_valid": 0},
                ],
            )
            kernel_fields = (
                "schema_version,sample_id,realtime_ns,monotonic_raw_ns,run_id,experiment_id,campaign_id,variant,trial,"
                "role,hostname,metric,raw_value,delta_value,interval_ns,cumulative,delta_valid,status,error_code"
            ).split(",")
            write_csv_rows(
                profiled_ue / "kernel_activity.csv",
                kernel_fields,
                [
                    {"schema_version": 2, "sample_id": 1, "metric": "interrupts", "raw_value": 1000, "cumulative": 1, "delta_valid": 0, "status": "warmup", "error_code": 0},
                    {"schema_version": 2, "sample_id": 2, "metric": "interrupts", "raw_value": 1100, "delta_value": 100, "interval_ns": 1_000_000_000, "cumulative": 1, "delta_valid": 1, "status": "ok", "error_code": 0},
                ],
            )
            activity_fields = (
                "schema_version,sample_id,realtime_ns,monotonic_raw_ns,run_id,experiment_id,campaign_id,variant,trial,"
                "role,hostname,source,label,description,cpu,raw_count,delta_count,interval_ns,delta_valid,radio_relevant,status"
            ).split(",")
            write_csv_rows(
                profiled_ue / "interrupts.csv",
                activity_fields,
                [{"schema_version": 2, "sample_id": 2, "source": "hardirq", "label": "123", "description": "xhci_hcd", "cpu": 2, "raw_count": 200, "delta_count": 20, "interval_ns": 1_000_000_000, "delta_valid": 1, "radio_relevant": 1, "status": "ok"}],
            )
            write_csv_rows(
                profiled_ue / "softirqs.csv",
                activity_fields,
                [{"schema_version": 2, "sample_id": 2, "source": "softirq", "label": "NET_RX", "description": "network receive", "cpu": 2, "raw_count": 300, "delta_count": 30, "interval_ns": 1_000_000_000, "delta_valid": 1, "radio_relevant": 1, "status": "ok"}],
            )
            system_overhead_fields = (
                "schema_version,sample_id,realtime_ns,monotonic_raw_ns,run_id,experiment_id,campaign_id,variant,trial,"
                "role,hostname,source,duration_tick,duration_us,rows,status,error_code"
            ).split(",")
            write_csv_rows(
                profiled_ue / "system_read_overhead.csv",
                system_overhead_fields,
                [
                    {
                        "schema_version": 2,
                        "sample_id": 2,
                        "realtime_ns": 2_000_000_000,
                        "monotonic_raw_ns": 1_500_000_000,
                        "run_id": "in-process-nrUE",
                        "experiment_id": "scientific-baseline-in-process-t001",
                        "campaign_id": "scientific-baseline",
                        "variant": "in-process",
                        "trial": 1,
                        "role": "nrUE",
                        "hostname": "nrue-host",
                        "source": "thread_metrics",
                        "duration_tick": 50,
                        "duration_us": 5.0,
                        "rows": 1,
                        "status": "ok",
                        "error_code": 0,
                    },
                    {
                        "schema_version": 2,
                        "sample_id": 2,
                        "realtime_ns": 2_000_000_000,
                        "monotonic_raw_ns": 1_500_000_000,
                        "run_id": "in-process-nrUE",
                        "experiment_id": "scientific-baseline-in-process-t001",
                        "campaign_id": "scientific-baseline",
                        "variant": "in-process",
                        "trial": 1,
                        "role": "nrUE",
                        "hostname": "nrue-host",
                        "source": "kernel_activity",
                        "duration_tick": 60,
                        "duration_us": 6.0,
                        "rows": 1,
                        "status": "ok",
                        "error_code": 0,
                    },
                ],
            )

            write_csv_rows(
                profiled_ue / "settings.csv",
                ["realtime_ns", "key", "value", "source"],
                [{"realtime_ns": 1, "key": "profile.pmu_mode", "value": "software", "source": "resolved"}],
            )
            write_csv_rows(
                profiled_ue / "sync.csv",
                [
                    "realtime_ns",
                    "monotonic_raw_ns",
                    "tick",
                    "monotonic_raw_before_ns",
                    "monotonic_raw_after_ns",
                    "monotonic_raw_uncertainty_ns",
                    "tick_before",
                    "tick_after",
                    "tick_uncertainty",
                    "status",
                ],
                [
                    {
                        "realtime_ns": 1_000_000_000,
                        "monotonic_raw_ns": 500_000_000,
                        "tick": 100,
                        "monotonic_raw_before_ns": 499_999_900,
                        "monotonic_raw_after_ns": 500_000_100,
                        "monotonic_raw_uncertainty_ns": 200,
                        "tick_before": 99,
                        "tick_after": 101,
                        "tick_uncertainty": 2,
                        "status": "ok",
                    },
                    {
                        "realtime_ns": 2_000_000_000,
                        "monotonic_raw_ns": 1_500_000_000,
                        "tick": 200,
                        "monotonic_raw_before_ns": 1_499_999_900,
                        "monotonic_raw_after_ns": 1_500_000_100,
                        "monotonic_raw_uncertainty_ns": 200,
                        "tick_before": 199,
                        "tick_after": 201,
                        "tick_uncertainty": 2,
                        "status": "ok",
                    },
                ],
            )
            sidecars = profiled_ue / "sidecars"
            sidecars.mkdir()
            write_text(sidecars / "perf_stat.csv", "1.000;100;;cycles;100.00%\n2.000;200;;cycles;50.00%\n")
            external_fields = (
                "schema_version,run_id,experiment_id,campaign_id,variant,trial,role,hostname,source_id,source_type,"
                "clock_domain,clock_unit,artifact_path,command,tool_version,start_realtime_ns,end_realtime_ns,"
                "start_monotonic_raw_ns,end_monotonic_raw_ns,status,alignment_method,alignment_uncertainty_ns,notes"
            ).split(",")
            write_csv_rows(
                profiled_ue / "external_sources.csv",
                external_fields,
                [{"schema_version": 2, "source_id": "perf-stat-nrUE-t001", "source_type": "perf_stat", "clock_domain": "process_lifetime", "clock_unit": "mixed", "artifact_path": "sidecars/perf_stat.csv", "command": "perf stat", "tool_version": "perf version test", "start_realtime_ns": 1_000_000_000, "end_realtime_ns": 12_000_000_000, "status": "recorded", "alignment_method": "aggregate_run_interval", "notes": "synthetic"}],
            )

            for run_dir in (disabled_gnb, disabled_ue, profiled_gnb, profiled_ue):
                finalize_archive(run_dir)

            subprocess.run([sys.executable, str(ANALYZER), str(root), "--output-dir", str(output)], check=True)

            inputs = {
                row["input_path"]: row
                for row in read_rows(output / "analysis_inputs.csv")
            }
            self.assertEqual(
                set(inputs),
                {str(disabled_gnb), str(disabled_ue), str(profiled_gnb), str(profiled_ue)},
            )
            for run_dir in (disabled_gnb, disabled_ue, profiled_gnb, profiled_ue):
                row = inputs[str(run_dir)]
                manifest = run_dir / "archive_manifest.csv"
                self.assertEqual(row["archive_manifest_status"], "present")
                self.assertEqual(row["archive_manifest_path"], str(manifest))
                self.assertEqual(
                    row["archive_manifest_sha256"],
                    hashlib.sha256(manifest.read_bytes()).hexdigest(),
                )

            campaigns = read_rows(output / "campaign_runs.csv")
            self.assertEqual(len(campaigns), 4)
            disabled = next(row for row in campaigns if row["profile_dir"] == str(disabled_ue))
            self.assertEqual(disabled["profile_enabled"], "False")
            self.assertEqual(disabled["events_present"], "0")

            completeness = read_rows(output / "campaign_completeness.csv")
            self.assertEqual(len(completeness), 2)
            self.assertEqual({row["status"] for row in completeness}, {"complete"})

            integrity = read_rows(output / "archive_integrity.csv")
            self.assertTrue(integrity)
            self.assertEqual({row["status"] for row in integrity}, {"ok"})
            self.assertEqual({row["valid"] for row in integrity}, {"1"})

            clock_quality = next(
                row
                for row in read_rows(output / "clock_quality.csv")
                if row["profile_dir"] == str(profiled_ue)
            )
            self.assertEqual(clock_quality["status"], "ok")
            self.assertEqual(clock_quality["bounded_sample_count"], "2")
            self.assertAlmostEqual(float(clock_quality["uncertainty_p99_ns"]), 200.0)
            self.assertAlmostEqual(
                float(clock_quality["fit_slope_realtime_per_monotonic"]),
                1.0,
            )
            self.assertAlmostEqual(float(clock_quality["fit_rate_error_ppm"]), 0.0)
            host = next(
                row
                for row in read_rows(output / "host_metrics_summary.csv")
                if row["profile_dir"] == str(profiled_ue)
            )
            self.assertEqual(host["quality_status"], "ok")
            self.assertEqual(host["sample_count"], "3")
            self.assertEqual(host["bounded_acquisition_samples"], "3")
            self.assertEqual(host["fully_valid_samples"], "3")
            self.assertEqual(host["writer_cpu_migration_samples"], "1")
            self.assertAlmostEqual(float(host["thermal_zone0_p50_millicelsius"]), 70_000.0)
            self.assertAlmostEqual(float(host["acquisition_duration_p50_us"]), 200.0)
            self.assertAlmostEqual(float(host["sampling_cadence_p50_s"]), 1.0)
            self.assertEqual(host["rpi_throttled_raw_or"], "65537")
            self.assertEqual(host["rpi_undervoltage_now_samples"], "2")
            self.assertAlmostEqual(float(host["rpi_undervoltage_now_percent"]), 200.0 / 3.0)
            self.assertEqual(host["rpi_undervoltage_occurred_samples"], "1")
            self.assertAlmostEqual(float(host["process_user_delta_total_us"]), 400_000.0)
            self.assertAlmostEqual(float(host["process_user_cpu_percent"]), 20.0)
            self.assertAlmostEqual(float(host["process_system_cpu_percent"]), 10.0)
            self.assertAlmostEqual(float(host["process_total_cpu_percent"]), 30.0)
            self.assertAlmostEqual(
                float(host["voluntary_context_switches_rate_per_second"]),
                3.0,
            )
            self.assertAlmostEqual(
                float(host["involuntary_context_switches_rate_per_second"]),
                1.5,
            )

            primitive = read_rows(output / "profiler_primitive_overhead_summary.csv")

            timeline = read_rows(output / "event_timeline.csv")
            mapped_receive = next(
                row
                for row in timeline
                if row["profile_dir"] == str(profiled_ue)
                and row["event_name"] == "USRP_RX_RECV"
            )
            self.assertEqual(mapped_receive["sync_status"], "ok")
            self.assertEqual(mapped_receive["start_mapping_position"], "anchor_exact")
            self.assertEqual(mapped_receive["end_mapping_position"], "extrapolated_after")
            self.assertEqual(mapped_receive["start_monotonic_raw_ns_estimate"], "1500000000")
            self.assertEqual(mapped_receive["end_monotonic_raw_ns_estimate"], "1900000000")
            self.assertEqual(mapped_receive["mapped_duration_ns"], "400000000")
            self.assertEqual(mapped_receive["recorded_duration_ns"], "4000")
            self.assertEqual(mapped_receive["duration_difference_ns"], "399996000")
            self.assertEqual(
                mapped_receive["monotonic_interval_status"],
                "valid_with_extrapolation_drift_unbounded",
            )
            span = next(row for row in primitive if row["primitive"] == "span_start_stop")
            self.assertAlmostEqual(float(span["baseline_p50_us"]), 2.0)
            self.assertAlmostEqual(float(span["duration_p50_us"]), 6.0)
            self.assertAlmostEqual(float(span["median_excess_over_counter_pair_us"]), 4.0)
            self.assertEqual(span["stream_status"], "recorded")

            pmu = next(row for row in read_rows(output / "pmu_summary.csv") if row["event_name"] == "cpu_cycles")
            self.assertEqual(pmu["usable_samples"], "2")
            self.assertAlmostEqual(float(pmu["delta_scaled_total"]), 300.0)
            self.assertAlmostEqual(float(pmu["estimated_rate_per_second"]), 150.0)
            quality = next(row for row in read_rows(output / "pmu_quality.csv") if row["event_name"] == "cpu_cycles")
            self.assertEqual(quality["delta_valid_count"], "2")
            self.assertEqual(quality["scaling_valid_count"], "3")
            self.assertEqual(quality["usable_count"], "2")
            self.assertAlmostEqual(float(quality["multiplex_ratio_min"]), 0.5)
            self.assertAlmostEqual(float(quality["multiplex_ratio_p50"]), 0.75)

            overhead = next(row for row in read_rows(output / "collection_overhead_summary.csv") if row["source"] == "pmu_read")
            self.assertEqual(overhead["samples_total"], "2")
            self.assertEqual(overhead["read_errors_total"], "1")
            self.assertEqual(overhead["status"], "counter_regression;ok")
            self.assertAlmostEqual(float(overhead["duration_p50_us"]), 10.0)


            scheduler = next(row for row in read_rows(output / "thread_scheduler_summary.csv") if row["tid"] == "200")
            self.assertEqual(scheduler["samples_total"], "3")
            self.assertAlmostEqual(float(scheduler["cpu_frequency_p50_khz"]), 2_400_000.0)
            self.assertAlmostEqual(float(scheduler["cpu_frequency_min_khz"]), 2_400_000.0)
            self.assertAlmostEqual(float(scheduler["cpu_frequency_max_khz"]), 2_400_000.0)
            self.assertAlmostEqual(float(scheduler["cpu_runtime_percent"]), 40.0)
            self.assertAlmostEqual(float(scheduler["runqueue_wait_percent_of_elapsed"]), 10.0)
            self.assertAlmostEqual(float(scheduler["scheduler_wait_fraction"]), 0.2)
            self.assertAlmostEqual(float(scheduler["context_switches_per_second"]), 5.0)

            interference = read_rows(output / "kernel_interference_summary.csv")
            hardirq = next(row for row in interference if row["description"] == "xhci_hcd")
            softirq = next(row for row in interference if row["label"] == "NET_RX")
            self.assertAlmostEqual(float(hardirq["rate_per_second"]), 20.0)
            self.assertAlmostEqual(float(softirq["rate_per_second"]), 30.0)

            transport = next(row for row in read_rows(output / "transport_summary.csv") if row["event_name"] == "USRP_RX_RECV")
            self.assertEqual(transport["count"], "1")
            self.assertAlmostEqual(float(transport["total_duration_us"]), 4.0)
            faults = read_rows(output / "transport_faults.csv")
            self.assertEqual(len(faults), 1)
            self.assertEqual(faults[0]["event_name"], "USRP_RX_SHORT_READ")

            observer = read_rows(output / "observer_effect_summary.csv")
            process_duration = next(
                row
                for row in observer
                if row["role"] == "nrUE"
                and row["metric_name"] == "process_duration"
                and row["variant"] == "in-process"
            )
            self.assertEqual(process_duration["effect_estimator"], "paired_trial_delta")
            self.assertEqual(process_duration["paired_sample_count"], "1")
            self.assertAlmostEqual(float(process_duration["paired_delta_p50"]), 1.0)
            self.assertAlmostEqual(float(process_duration["paired_delta_percent_p50"]), 10.0)
            self.assertEqual(process_duration["unpaired_variant_samples"], "0")
            self.assertEqual(process_duration["unpaired_baseline_samples"], "0")
    def test_host_metrics_report_rejects_invalid_chains_and_labels_legacy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-host-quality-") as temporary:
            root = Path(temporary)
            gapped = root / "gapped"
            malformed = root / "malformed"
            legacy = root / "legacy"
            missing = root / "missing"
            for path in (gapped, malformed, legacy, missing):
                path.mkdir()

            def bounded_row(
                monotonic_ns: int,
                tick: int,
                process_user_us: int,
                *,
                rusage_status: str = "ok",
                error_mask: int = 0,
            ) -> dict[str, object]:
                acquisition_ns = 100_000
                acquisition_tick = 1000
                return {
                    "realtime_ns": monotonic_ns + 500_000_000,
                    "monotonic_raw_ns": monotonic_ns,
                    "tick": tick,
                    "writer_cpu": 2,
                    "process_user_us": process_user_us,
                    "process_system_us": process_user_us // 2,
                    "process_maxrss_kb": 1000,
                    "end_monotonic_raw_ns": monotonic_ns + acquisition_ns,
                    "end_tick": tick + acquisition_tick,
                    "writer_cpu_end": 2,
                    "writer_cpu_migrated": 0,
                    "acquisition_duration_monotonic_raw_ns": acquisition_ns,
                    "acquisition_duration_tick": acquisition_tick,
                    "acquisition_duration_us": 100,
                    "status": "ok" if error_mask == 0 else "partial_probe_error",
                    "getloadavg_count": 3,
                    "getrusage_status": rusage_status,
                    "error_mask": error_mask,
                }

            gapped_rows = [
                bounded_row(1_000_000_000, 1000, 100),
                bounded_row(
                    2_000_000_000,
                    10_001_000,
                    0,
                    rusage_status="error",
                    error_mask=1 << 6,
                ),
                bounded_row(3_000_000_000, 20_001_000, 300),
                bounded_row(4_000_000_000, 30_001_000, 200),
                bounded_row(5_000_000_000, 40_001_000, 400),
            ]
            write_csv_rows(gapped / "host_metrics.csv", HOST_METRIC_FIELDS, gapped_rows)

            malformed_row = bounded_row(1_000_000_000, 1000, 100)
            malformed_row["acquisition_duration_monotonic_raw_ns"] = 99_999
            write_csv_rows(
                malformed / "host_metrics.csv",
                HOST_METRIC_FIELDS,
                [malformed_row],
            )
            write_csv_rows(
                legacy / "host_metrics.csv",
                ["monotonic_raw_ns", "process_user_us"],
                [{"monotonic_raw_ns": 100, "process_user_us": 10}],
            )

            report = host_metrics_report(
                [gapped, malformed, legacy, missing],
                {},
                {},
            )
            rows = {str(row["profile_dir"]): row for row in report.rows}

            gapped_summary = rows[str(gapped)]
            self.assertEqual(gapped_summary["quality_status"], "partial_probe_error")
            self.assertEqual(gapped_summary["getrusage_error_samples"], 1)
            self.assertEqual(gapped_summary["error_getrusage_samples"], 1)
            self.assertEqual(gapped_summary["process_user_valid_samples"], 4)
            self.assertEqual(gapped_summary["process_user_invalid_samples"], 1)
            self.assertEqual(gapped_summary["process_user_valid_intervals"], 1)
            self.assertEqual(gapped_summary["process_user_invalid_intervals"], 1)
            self.assertAlmostEqual(
                float(gapped_summary["process_user_delta_total_us"]),
                200.0,
            )
            self.assertAlmostEqual(
                float(gapped_summary["process_user_rate_per_second"]),
                200.0,
            )
            self.assertAlmostEqual(
                float(gapped_summary["process_user_cpu_percent"]),
                0.02,
            )

            malformed_summary = rows[str(malformed)]
            self.assertEqual(malformed_summary["quality_status"], "acquisition_invalid")
            self.assertEqual(malformed_summary["bounded_acquisition_samples"], 0)
            self.assertEqual(malformed_summary["invalid_bounded_acquisition_samples"], 1)
            self.assertEqual(
                malformed_summary["acquisition_statuses"],
                "monotonic_duration_mismatch",
            )

            legacy_summary = rows[str(legacy)]
            self.assertEqual(
                legacy_summary["quality_status"],
                "legacy_acquisition_unbounded",
            )
            self.assertEqual(legacy_summary["legacy_acquisition_unbounded_samples"], 1)
            self.assertEqual(legacy_summary["getrusage_unverified_legacy_samples"], 1)

            missing_summary = rows[str(missing)]
            self.assertEqual(missing_summary["stream_status"], "missing")
            self.assertEqual(missing_summary["quality_status"], "missing")
            self.assertEqual(missing_summary["sample_count"], 0)


if __name__ == "__main__":
    unittest.main()
