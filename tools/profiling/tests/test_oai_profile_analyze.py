#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ANALYZER = Path(__file__).resolve().parents[1] / "oai_profile_analyze.py"

ARCHIVE_TOOL = Path(__file__).resolve().parents[1] / "oai_profile_archive.py"
sys.path.insert(0, str(ANALYZER.parent))

from oai_profile_deadlines import build_deadline_reports, round_div_signed  # noqa: E402
from oai_profile_analyze import build_pairs  # noqa: E402
from oai_profile_reports import (  # noqa: E402
    campaign_completeness_report,
    external_sources_report,
    host_metrics_report,
    perf_stat_report,
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
        "stop_reason": "measurement_complete",
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
    write_text(
        run_dir / "metadata.txt",
        "schema_version=2\nevent_record_size_bytes=120\nmax_nesting_depth=64\ncounter_hz=10000000\n"
        f"process_name={process}\nrole={role}\nrun_id={variant}-{role}\n"
        f"experiment_id=scientific-baseline-{variant}-t001\ncampaign_id=scientific-baseline\n"
        f"variant={variant}\ntrial=1\nhostname={role.lower()}-host\npmu_mode=software\n"
        f"start_realtime_ns=1000000000\nend_realtime_ns={1_000_000_000 + duration_s * 1_000_000_000}\n"
        "clean_shutdown=1\n",
    )


def finalize_archive(run_dir: Path) -> None:
    subprocess.run([sys.executable, str(ARCHIVE_TOOL), "finalize", str(run_dir)], check=True, capture_output=True, text=True)

class AnalyzerSchemaCompatibilityTest(unittest.TestCase):
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
                "start_realtime_ns=3000000000\nend_realtime_ns=4000000000\nclean_shutdown=1\n",
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
                "2,10,20,ue-v2,107,USRP_RX_RECV,duration,1,10,8,208,82,0,0,2,2,0,4,"
                "512,512,1,0,7000,100,10.000\n"
                "2,11,20,ue-v2,108,UE_TX_DEADLINE_COMPUTE,instant,1,10,9,209,82,0,0,2,2,0,1,"
                "100000,107680,7680,1999000000,7200,0,0.000\n"
                "2,12,21,worker-a,109,UE_TX_DEADLINE_CHECK,instant,1,10,9,209,82,0,0,2,2,0,3,"
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
                "thread_index,tid,thread_name,dropped_records,span_stack_overflows,span_stack_mismatches\n"
                "0,20,ue-v2,1,2,3\n",
            )
            write_text(
                v2 / "sync.csv",
                "realtime_ns,monotonic_raw_ns,tick\n3000000000,3,1000\n4000000000,4,1100\n",
            )

            subprocess.run(
                [sys.executable, str(ANALYZER), str(root), "--output-dir", str(output)],
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
            }
            self.assertEqual(expected_outputs, {path.name for path in output.iterdir()})
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
            self.assertEqual(integrity["schema2_records"], "10")
            self.assertEqual(integrity["duration_records"], "9")
            self.assertEqual(integrity["instant_records"], "1")
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
                        "frame": 1,
                        "slot": 1,
                        "absolute_slot": 21,
                        "cpu_start": 1,
                        "cpu_end": 1,
                        "aux0": 512,
                        "aux1": 1,
                        "aux2": 512,
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
                        "frame": 1,
                        "slot": 1,
                        "absolute_slot": 21,
                        "cpu_start": 2,
                        "cpu_end": 2,
                        "flags": 4,
                        "aux0": 512,
                        "aux1": 512,
                        "aux2": 1,
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
                        "frame": 1,
                        "slot": 1,
                        "absolute_slot": 21,
                        "cpu_start": 2,
                        "cpu_end": 2,
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
                primitive_rows.append(
                    {
                        **primitive_identity,
                        "sample_index": index,
                        "primitive": "counter_pair",
                        "event_kind": "unknown",
                        "outer_duration_us": duration,
                    }
                )
            for index, duration in enumerate((5.0, 7.0)):
                primitive_rows.append(
                    {
                        **primitive_identity,
                        "sample_index": index,
                        "primitive": "span_start_stop",
                        "event_kind": "duration",
                        "outer_duration_us": duration,
                        "event_record_expected": 1,
                        "event_recorded": 1,
                        "event_seq": 10 + index,
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
                "error_code": 0,
            }
            write_csv_rows(
                profiled_ue / "pmu_samples.csv",
                pmu_fields,
                [
                    {**pmu_common, "sample_id": 1, "multiplex_ratio": 1.0, "interval_ns": 0, "delta_valid": 0, "scaling_valid": 0, "status": "warmup"},
                    {**pmu_common, "sample_id": 2, "delta_raw": 50, "delta_scaled": 100, "multiplex_ratio": 0.5, "interval_ns": 1_000_000_000, "delta_valid": 1, "scaling_valid": 1, "status": "ok"},
                    {**pmu_common, "sample_id": 3, "delta_raw": 200, "delta_scaled": 200, "multiplex_ratio": 1.0, "interval_ns": 1_000_000_000, "delta_valid": 1, "scaling_valid": 1, "status": "ok"},
                ],
            )
            pmu_overhead_fields = (
                "schema_version,sample_id,realtime_ns,monotonic_raw_ns,run_id,experiment_id,campaign_id,thread_index,"
                "tid,thread_name,duration_tick,duration_us,available_events,active_groups,group_reads,observations,read_errors,counter_status"
            ).split(",")
            write_csv_rows(
                profiled_ue / "pmu_read_overhead.csv",
                pmu_overhead_fields,
                [
                    {"schema_version": 2, "sample_id": 2, "thread_index": 0, "tid": 200, "thread_name": "ue-rx", "duration_us": 10.0, "observations": 1, "read_errors": 0, "counter_status": "ok"},
                    {"schema_version": 2, "sample_id": 3, "thread_index": 0, "tid": 200, "thread_name": "ue-rx", "duration_us": 0.0, "observations": 0, "read_errors": 0, "counter_status": "counter_regression"},
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
                    {"schema_version": 2, "sample_id": 2, "source": "thread_metrics", "duration_us": 5.0, "rows": 1, "status": "ok", "error_code": 0},
                    {"schema_version": 2, "sample_id": 2, "source": "kernel_activity", "duration_us": 6.0, "rows": 1, "status": "ok", "error_code": 0},
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

            pmu = next(row for row in read_rows(output / "pmu_summary.csv") if row["event_name"] == "cpu_cycles")
            self.assertEqual(pmu["usable_samples"], "2")
            self.assertAlmostEqual(float(pmu["delta_scaled_total"]), 300.0)
            self.assertAlmostEqual(float(pmu["estimated_rate_per_second"]), 150.0)
            quality = next(row for row in read_rows(output / "pmu_quality.csv") if row["event_name"] == "cpu_cycles")
            self.assertEqual(quality["delta_valid_count"], "2")
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
