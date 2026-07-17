#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0

import csv
import sys
import tempfile
import unittest
from pathlib import Path

PROFILING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROFILING_DIR))


from oai_profile_clock import (
    EVENT_TIMELINE_FIELDS,
    ClockMapper,
    event_timeline_row,
)


SYNC_FIELDS = [
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
]


def bounded_anchor(realtime_ns: int, monotonic_ns: int, tick: int) -> dict[str, object]:
    return {
        "realtime_ns": realtime_ns,
        "monotonic_raw_ns": monotonic_ns,
        "tick": tick,
        "monotonic_raw_before_ns": monotonic_ns - 100,
        "monotonic_raw_after_ns": monotonic_ns + 100,
        "monotonic_raw_uncertainty_ns": 200,
        "tick_before": tick - 1,
        "tick_after": tick + 1,
        "tick_uncertainty": 2,
        "status": "ok",
    }


def write_sync(path: Path, rows: list[dict[str, object]], fields: list[str] = SYNC_FIELDS) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class ClockMapperTest(unittest.TestCase):
    def test_exact_interpolation_and_extrapolation_are_integer_and_explicit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-clock-map-") as temporary:
            sync_path = Path(temporary) / "sync.csv"
            write_sync(
                sync_path,
                [
                    bounded_anchor(2_000_000, 1_000_000, 1000),
                    bounded_anchor(3_000_000, 2_000_000, 2000),
                ],
            )
            mapper = ClockMapper.from_sync(sync_path, 1_000_000)
            self.assertEqual(mapper.status, "ok")
            self.assertEqual(mapper.bounded_anchor_count, 2)

            exact = mapper.map_tick(1000)
            self.assertEqual(exact.monotonic_raw_ns, 1_000_000)
            self.assertEqual(exact.realtime_ns, 2_000_000)
            self.assertEqual(exact.method, "anchor")
            self.assertEqual(exact.acquisition_uncertainty_ns, 200)

            interpolated = mapper.map_tick(1500)
            self.assertEqual(interpolated.monotonic_raw_ns, 1_500_000)
            self.assertEqual(interpolated.realtime_ns, 2_500_000)
            self.assertEqual(interpolated.position, "interpolated")
            self.assertEqual(interpolated.status, "interpolated_bounded")
            self.assertEqual(interpolated.acquisition_uncertainty_ns, 2200)
            self.assertEqual(
                interpolated.uncertainty_scope,
                "local_anchor_acquisition_only",
            )

            before = mapper.map_tick(500)
            after = mapper.map_tick(2500)
            self.assertEqual(before.monotonic_raw_ns, 500_000)
            self.assertEqual(after.monotonic_raw_ns, 2_500_000)
            self.assertEqual(before.position, "extrapolated_before")
            self.assertEqual(after.position, "extrapolated_after")
            self.assertIn("drift_unbounded", before.uncertainty_scope)
            self.assertIn("drift_unbounded", after.realtime_status)

    def test_single_anchor_uses_declared_counter_rate_without_claiming_drift_bound(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-clock-single-") as temporary:
            sync_path = Path(temporary) / "sync.csv"
            write_sync(
                sync_path,
                [{"realtime_ns": 2_000_000, "monotonic_raw_ns": 1_000_000, "tick": 1000}],
                ["realtime_ns", "monotonic_raw_ns", "tick"],
            )
            mapper = ClockMapper.from_sync(sync_path, 1_000_000)
            mapped = mapper.map_tick(1100)
            self.assertEqual(mapped.monotonic_raw_ns, 1_100_000)
            self.assertEqual(mapped.realtime_ns, 2_100_000)
            self.assertEqual(mapped.method, "nominal_counter_hz")
            self.assertEqual(mapped.status, "single_anchor_legacy_unbounded")
            self.assertEqual(
                mapped.uncertainty_scope,
                "anchor_acquisition_only_drift_unbounded",
            )

            no_rate = ClockMapper.from_sync(sync_path, 0).map_tick(1100)
            self.assertIsNone(no_rate.monotonic_raw_ns)
            self.assertEqual(no_rate.status, "insufficient_anchors_and_counter_hz")

    def test_bad_anchors_are_rejected_and_realtime_regression_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-clock-invalid-") as temporary:
            sync_path = Path(temporary) / "sync.csv"
            duplicate = bounded_anchor(2_500_000, 1_500_000, 1000)
            regressed_realtime = bounded_anchor(1_500_000, 2_000_000, 2000)
            regressed_realtime["monotonic_raw_uncertainty_ns"] = 201
            write_sync(
                sync_path,
                [
                    bounded_anchor(2_000_000, 1_000_000, 1000),
                    duplicate,
                    regressed_realtime,
                ],
            )
            mapper = ClockMapper.from_sync(sync_path, 1_000_000)
            self.assertEqual(len(mapper.anchors), 2)
            self.assertEqual(mapper.invalid_anchor_count, 1)
            self.assertEqual(mapper.malformed_bounds_count, 1)
            self.assertEqual(mapper.realtime_regression_count, 1)
            self.assertEqual(mapper.status, "usable_with_rejected_samples")

            mapped = mapper.map_tick(1500)
            self.assertEqual(mapped.monotonic_raw_ns, 1_500_000)
            self.assertIsNone(mapped.realtime_ns)
            self.assertEqual(mapped.status, "interpolated_partially_bounded")
            self.assertEqual(mapped.realtime_status, "segment_realtime_regression")

    def test_missing_sync_and_invalid_ticks_remain_unavailable(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-clock-missing-") as temporary:
            mapper = ClockMapper.from_sync(Path(temporary) / "sync.csv", 1_000_000)
            self.assertEqual(mapper.status, "sync_missing")
            self.assertEqual(mapper.map_tick(100).status, "sync_missing")
            self.assertEqual(mapper.map_tick(0).status, "invalid_event_tick")

    def test_event_timeline_preserves_identity_and_duration_consistency(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-event-timeline-") as temporary:
            profile_dir = Path(temporary)
            sync_path = profile_dir / "sync.csv"
            write_sync(
                sync_path,
                [
                    bounded_anchor(2_000_000, 1_000_000, 1000),
                    bounded_anchor(3_000_000, 2_000_000, 2000),
                ],
            )
            mapper = ClockMapper.from_sync(sync_path, 1_000_000)
            metadata = {
                "run_id": "run-1",
                "experiment_id": "experiment-1",
                "campaign_id": "campaign-1",
                "variant": "in-process",
                "trial": "1",
                "role": "nrUE",
                "hostname": "cm5",
            }
            event = {
                "schema_version": "2",
                "seq": "7",
                "tid": "42",
                "thread_name": "ue-rx",
                "event_id": "2",
                "event_name": "UE_RF_READ",
                "event_role": "nrUE",
                "subsystem": "radio",
                "event_class": "io",
                "event_kind": "duration",
                "detail_level": "boundary",
                "start_tick": "1500",
                "duration_tick": "100",
                "duration_us": "100.0",
            }
            row = event_timeline_row(profile_dir, metadata, event, mapper)
            self.assertEqual(len(EVENT_TIMELINE_FIELDS), len(set(EVENT_TIMELINE_FIELDS)))
            self.assertEqual(row["run_id"], "run-1")
            self.assertEqual(row["hostname"], "cm5")
            self.assertEqual(row["end_tick"], 1600)
            self.assertEqual(row["start_monotonic_raw_ns_estimate"], 1_500_000)
            self.assertEqual(row["end_monotonic_raw_ns_estimate"], 1_600_000)
            self.assertEqual(row["mapped_duration_ns"], 100_000)
            self.assertEqual(row["recorded_duration_ns"], 100_000)
            self.assertEqual(row["duration_difference_ns"], 0)
            self.assertEqual(row["monotonic_interval_status"], "valid")
            self.assertEqual(row["realtime_interval_status"], "valid")
            self.assertEqual(row["interval_acquisition_uncertainty_ns"], 4400)

            invalid = event_timeline_row(
                profile_dir,
                metadata,
                {**event, "duration_tick": "-1"},
                mapper,
            )
            self.assertEqual(invalid["event_tick_status"], "invalid")
            self.assertEqual(invalid["monotonic_interval_status"], "unavailable")
            self.assertIsNone(invalid["mapped_duration_ns"])


if __name__ == "__main__":
    unittest.main()
