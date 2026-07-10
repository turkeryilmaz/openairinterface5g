#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ANALYZER = Path(__file__).resolve().parents[1] / "oai_profile_analyze.py"


def write_text(path: Path, text: str) -> None:
    path.write_text(text)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


class AnalyzerSchemaCompatibilityTest(unittest.TestCase):
    def test_phase1_and_phase2a_profiles(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-analyzer-test-") as tmp:
            root = Path(tmp)
            v1 = root / "v1_nrUE"
            v2 = root / "v2_nrUE"
            output = root / "analysis"
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
                "schema_version=2\nevent_record_size_bytes=120\nmax_nesting_depth=64\n"
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
                "281474976710657,2,2,0,0,200,190,10,0,1080,0,0.000\n",
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
                "deadline,us,lateness,us,,,\n",
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

            deadlines = read_rows(output / "deadline_misses.csv")
            v1_deadline = next(row for row in deadlines if row["profile_dir"] == str(v1))
            v2_deadline = next(row for row in deadlines if row["profile_dir"] == str(v2))
            self.assertEqual(v1_deadline["absolute_slot"], "-1")
            self.assertEqual(v1_deadline["cpu_start"], "-1")
            self.assertEqual(v2_deadline["correlation_id"], "77")
            self.assertEqual(v2_deadline["parent_id"], "281474976710657")

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
                "migrations.csv",
                "runs.csv",
                "pairs.csv",
                "host_summary.csv",
            }
            self.assertEqual(expected_outputs, {path.name for path in output.iterdir()})


if __name__ == "__main__":
    unittest.main()
