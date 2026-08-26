#!/usr/bin/env python3
"""Deterministic tests for bundle-owned runtime catalog schema v1."""

from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "runtime_catalog_v1.py"
LITERAL_PATH = ROOT / "archive/definition/runtime-catalog-schema-v1.json"
H1 = "1" * 64
H2 = "2" * 64


def load_module():
    spec = importlib.util.spec_from_file_location("runtime_catalog_v1_under_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("module loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RuntimeCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def catalog(self, catalog_id, entries):
        return {
            "catalog_id": catalog_id,
            "entries": copy.deepcopy(entries),
            "schema": dict(self.module.THREAD_SCHEMA),
            "version": {"major": 1, "minor": 0},
        }

    def thread_catalog(self, entries=None):
        rows = [
            {"process_generation": 7, "registration_ordinal": 1, "thread_index": 1},
            {"process_generation": 7, "registration_ordinal": 2, "thread_index": 4},
        ] if entries is None else entries
        return self.catalog("oai_memprof_thread", rows)

    def validate_thread(self, value, **overrides):
        options = {
            "expected_process_generation": 7,
            "configured_thread_capacity": 4,
            "record_count": 2,
            "mode_id": 4,
            "population_state": self.module.POPULATION_COMPLETE,
            "lifecycle_state": 5,
        }
        options.update(overrides)
        return self.module.validate_thread_catalog(value, **options)

    def segment(self, start, end, offset=0, permissions="r-xp"):
        return {
            "end_address": end,
            "file_offset": offset,
            "permissions": permissions,
            "start_address": start,
        }

    def module_row(self, logical_id="nr_softmodem", module_id=1, *, build_logical=True,
                   load_state=1, start=0x400000, sha=H1):
        row = {
            "build_id": "01" if module_id == 1 else "02",
            "build_logical_id": logical_id if build_logical else None,
            "byte_count": 123,
            "device": 8,
            "inode": 98 + module_id,
            "load_state_id": load_state,
            "loaded_path": f"/opt/oai/{logical_id}",
            "logical_id": logical_id,
            "module_generation": 7,
            "module_id": module_id,
            "module_map_sha256": H2,
            "namespace_id": 0,
            "process_generation": 7,
            "segments": [self.segment(start, start + 0x1000)],
            "sha256": sha,
        }
        row["module_map_sha256"] = self.module.module_map_sha256(row)
        return row

    def module_rows(self):
        rows = [
            self.module_row(),
            self.module_row("runtime", 2, start=0x500000, sha=H2),
        ]
        return rows

    def module_catalog(self, rows=None):
        return self.catalog("oai_memprof_module", self.module_rows() if rows is None else rows)

    def validate_module(self, value, **overrides):
        options = {
            "expected_process_generation": 7,
            "population_state": self.module.POPULATION_COMPLETE,
            "lifecycle_state": 5,
        }
        options.update(overrides)
        return self.module.validate_module_catalog(value, **options)

    def run_population(self, rows=None):
        modules = self.module_rows() if rows is None else rows
        return [
            {
                "build_logical_id": row["build_logical_id"],
                "load_generation": row["module_generation"],
                "load_state_id": row["load_state_id"],
                "loaded_path": row["loaded_path"],
                "logical_id": row["logical_id"],
                "observed": True,
                "runtime_identity": {
                    "build_id": row["build_id"],
                    "byte_count": row["byte_count"],
                    "module_map_sha256": row["module_map_sha256"],
                    "sha256": row["sha256"],
                },
            }
            for row in modules
        ]

    def opening(self, *, architecture=1, calibration=1, numerator=1_000_000_000,
                denominator=1, error=1, span=0):
        del architecture
        return {
            "calibration_error_bound_ns": error,
            "calibration_kind": calibration,
            "calibration_span_ns": span,
            "clock_kind": 1,
            "counter_frequency_denominator": denominator,
            "counter_frequency_numerator": numerator,
            "process_generation": 7,
            "start_counter": 1000,
            "start_monotonic_raw_ns": 5000,
            "start_realtime_unix_ns": 1_700_000_000_000_000_000,
        }

    def clock_catalog(self, opening=None, *, architecture=1, source=1,
                      counter_invalid=False, realtime_discontinuity=False):
        identity = self.opening() if opening is None else opening
        final_mono = identity["start_monotonic_raw_ns"] + 1000
        final_real = identity["start_realtime_unix_ns"] + 1000
        row = dict(identity)
        row.update({
            "acquisition_source_id": source,
            "acquisition_status_id": 1,
            "architecture_id": architecture,
            "counter_invalid_observed": counter_invalid,
            "counter_stability_status_id": 1,
            "observed_max_error_ns": 0,
            "realtime_discontinuity_observed": realtime_discontinuity,
            "samples": [
                {
                    "counter": 1000,
                    "monotonic_raw_after_ns": 5001,
                    "monotonic_raw_before_ns": 4999,
                    "realtime_unix_ns": identity["start_realtime_unix_ns"],
                    "sample_ordinal": 1,
                },
                {
                    "counter": 2000,
                    "monotonic_raw_after_ns": final_mono + 1,
                    "monotonic_raw_before_ns": final_mono - 1,
                    "realtime_unix_ns": final_real,
                    "sample_ordinal": 2,
                },
            ],
        })
        return self.catalog("oai_memprof_clock", [row])

    def validate_clock(self, value, opening=None, **overrides):
        identity = self.opening() if opening is None else opening
        options = {
            "opening_identity": identity,
            "architecture_id": 1,
            "final_counter": 2000,
            "final_monotonic_raw_ns": identity["start_monotonic_raw_ns"] + 1000,
            "final_realtime_unix_ns": identity["start_realtime_unix_ns"] + 1000,
            "counter_invalid_observed": False,
            "realtime_discontinuity_observed": False,
        }
        options.update(overrides)
        return self.module.validate_clock_catalog(value, **options)

    def test_definition_literal_is_exact_canonical_registered_member(self):
        raw = LITERAL_PATH.read_bytes()
        self.assertEqual(raw, self.module.DEFINITION_BYTES)
        self.assertEqual(self.module.validate_definition_bytes(raw), self.module.RUNTIME_DEFINITION)
        self.assertEqual(self.module.BUNDLE_MEMBER_PROPOSAL, {
            "name": "runtime_catalog_schema",
            "object_type": 13,
            "owner": "runtime",
            "path": "definition/runtime-catalog-schema-v1.json",
        })
        self.assertEqual(self.module.BUNDLE_ENTRY_PROPOSAL["sha256"], self.module.DEFINITION_SHA256)
        self.assertEqual(len(self.module.BUNDLE_CROSS_RELATION_PROPOSALS), 3)

    def test_thread_complete_history_resolves_and_negative_partial_retains_unresolved(self):
        raw = self.module.SEMANTIC.canonical_bytes(self.thread_catalog())
        parsed, population = self.module.validate_thread_catalog_bytes(
            raw,
            expected_process_generation=7,
            configured_thread_capacity=4,
            record_count=2,
            mode_id=4,
            population_state=1,
            lifecycle_state=5,
        )
        self.assertEqual(self.module.SEMANTIC.canonical_bytes(parsed), raw)
        self.assertEqual(population.thread_keys, frozenset({(7, 1), (7, 4)}))
        self.assertEqual(population.ready_thread_indices, (1, 4))
        self.assertTrue(population.population_complete)
        self.assertEqual(population.record_count, 2)
        with self.assertRaisesRegex(self.module.RuntimeCatalogError, "record count differs"):
            self.module.reconcile_thread_records(
                population,
                process_generation=7,
                records=[SimpleNamespace(thread_index=1)],
            )
        self.assertEqual(
            self.module.reconcile_thread_records(
                population,
                process_generation=7,
                records=[SimpleNamespace(thread_index=1), {"thread_index": 4}],
            ),
            (),
        )
        partial = self.validate_thread(
            self.thread_catalog([]),
            population_state=2,
            lifecycle_state=7,
            record_count=1,
        )
        self.assertEqual(
            self.module.reconcile_thread_records(
                partial,
                process_generation=7,
                records=[SimpleNamespace(thread_index=3)],
            ),
            (3,),
        )

    def test_thread_ordinal_reuse_capacity_type_and_complete_resolution_reject(self):
        cases = []
        gap = self.thread_catalog(); gap["entries"][1]["registration_ordinal"] = 3
        cases.append((gap, "contiguous"))
        reuse = self.thread_catalog(); reuse["entries"][1]["thread_index"] = 1
        cases.append((reuse, "reuse"))
        capacity = self.thread_catalog(); capacity["entries"][1]["thread_index"] = 5
        cases.append((capacity, "capacity"))
        typed = self.thread_catalog(); typed["entries"][0]["registration_ordinal"] = True
        cases.append((typed, "u64 integer"))
        for value, pattern in cases:
            with self.subTest(pattern=pattern), self.assertRaisesRegex(self.module.RuntimeCatalogError, pattern):
                self.validate_thread(value)
        population = self.validate_thread(self.thread_catalog())
        with self.assertRaisesRegex(self.module.RuntimeCatalogError, "unresolved"):
            self.module.reconcile_thread_records(
                population,
                process_generation=7,
                records=[SimpleNamespace(thread_index=1), SimpleNamespace(thread_index=2)],
            )
        with self.assertRaisesRegex(self.module.RuntimeCatalogError, "requires complete population"):
            self.validate_thread(self.thread_catalog(), population_state=2, lifecycle_state=5)

    def test_module_exact_segments_digest_and_run_population_reconcile(self):
        rows = self.module_rows()
        raw = self.module.SEMANTIC.canonical_bytes(self.module_catalog(rows))
        _parsed, validated = self.module.validate_module_catalog_bytes(
            raw, expected_process_generation=7, population_state=1, lifecycle_state=5
        )
        self.module.reconcile_module_relations(
            validated,
            callsite_module_keys=[(7, 1), (7, 2)],
            run_module_population=self.run_population(rows),
        )
        projection = self.module.module_map_projection(rows[0])
        self.assertEqual(
            self.module.module_map_sha256(rows[0]),
            self.module.SEMANTIC.sha256_hex(self.module.SEMANTIC.canonical_bytes(projection)),
        )

    def test_module_duplicate_logical_zero_digest_segment_namespace_and_map_mutants_reject(self):
        cases = []
        duplicate = self.module_rows(); duplicate[1]["logical_id"] = "nr_softmodem"; duplicate[1]["build_logical_id"] = "nr_softmodem"
        cases.append((duplicate, "logical-ID order|duplicate build"))
        zero = self.module_rows(); zero[0]["sha256"] = "0" * 64
        cases.append((zero, "all-zero SHA"))
        overlap = self.module_rows(); overlap[0]["segments"].append(self.segment(0x400800, 0x402000)); overlap[0]["module_map_sha256"] = self.module.module_map_sha256(overlap[0])
        cases.append((overlap, "nonoverlap"))
        bad_permissions = self.module.rows if False else self.module_rows(); bad_permissions[0]["segments"][0]["permissions"] = "rwx"; bad_permissions[0]["module_map_sha256"] = H2
        cases.append((bad_permissions, "permissions"))
        namespace = self.module_rows(); namespace[0]["namespace_id"] = 1; namespace[0]["module_map_sha256"] = self.module.module_map_sha256(namespace[0])
        cases.append((namespace, "namespace zero"))
        stale = self.module_rows(); stale[0]["segments"][0]["end_address"] += 1
        cases.append((stale, "projection digest"))
        for rows, pattern in cases:
            with self.subTest(pattern=pattern), self.assertRaisesRegex(self.module.RuntimeCatalogError, pattern):
                self.validate_module(self.module_catalog(rows))

    def test_module_unexpected_row_and_one_to_one_multiplicity(self):
        unexpected = self.module_row("unexpected_x", 1, build_logical=False, load_state=20)
        validated = self.validate_module(
            self.module_catalog([unexpected]), population_state=2, lifecycle_state=6
        )
        self.module.reconcile_module_relations(
            validated,
            callsite_module_keys=[],
            run_module_population=self.run_population([unexpected]),
        )
        with self.assertRaisesRegex(self.module.RuntimeCatalogError, "null build logical"):
            invalid = copy.deepcopy(unexpected); invalid["build_logical_id"] = "unexpected_x"
            self.validate_module(self.module_catalog([invalid]), population_state=2, lifecycle_state=6)
        rows = self.module_rows()
        validated = self.validate_module(self.module_catalog(rows))
        run = self.run_population(rows)
        run[1]["logical_id"] = run[0]["logical_id"]
        with self.assertRaisesRegex(self.module.RuntimeCatalogError, "duplicate observed"):
            self.module.reconcile_module_relations(validated, callsite_module_keys=[], run_module_population=run)

    def test_clock_acquisition_samples_bind_opening_terminal_and_transform(self):
        value = self.clock_catalog()
        raw = self.module.SEMANTIC.canonical_bytes(value)
        _parsed, row = self.module.validate_clock_catalog_bytes(
            raw,
            opening_identity=self.opening(),
            architecture_id=1,
            final_counter=2000,
            final_monotonic_raw_ns=6000,
            final_realtime_unix_ns=1_700_000_000_000_001_000,
            counter_invalid_observed=False,
            realtime_discontinuity_observed=False,
        )
        self.assertEqual(row["observed_max_error_ns"], 0)
        self.assertEqual(
            self.module.reconcile_record_counters(
                row,
                final_counter=2000,
                records=[SimpleNamespace(flags=0x60, counter_enter=1000, counter_exit=1100)],
            ),
            [(5000, 5100)],
        )

    def test_clock_missing_reordered_and_mismatched_acquisition_reject(self):
        cases = []
        missing = self.clock_catalog(); missing["entries"][0]["samples"] = []
        cases.append((missing, {}, "at least two"))
        duplicate = self.clock_catalog(); duplicate["entries"][0]["samples"][1]["counter"] = 1000
        cases.append((duplicate, {}, "strictly increase"))
        terminal = self.clock_catalog(); terminal["entries"][0]["samples"][1]["counter"] = 1999
        cases.append((terminal, {}, "terminal anchor"))
        source = self.clock_catalog(); source["entries"][0]["acquisition_source_id"] = 2
        cases.append((source, {}, "source mismatch"))
        error = self.clock_catalog(); error["entries"][0]["samples"][0]["monotonic_raw_before_ns"] = 4997
        cases.append((error, {}, "observed bracket envelope"))
        unflagged = self.clock_catalog()
        unflagged["entries"][0]["samples"].insert(1, {
            "counter": 1500,
            "monotonic_raw_after_ns": 5501,
            "monotonic_raw_before_ns": 5499,
            "realtime_unix_ns": 1_700_000_000_000_000_600,
            "sample_ordinal": 2,
        })
        unflagged["entries"][0]["samples"][2]["sample_ordinal"] = 3
        cases.append((unflagged, {}, "realtime discontinuity"))
        for value, options, pattern in cases:
            with self.subTest(pattern=pattern), self.assertRaisesRegex(self.module.RuntimeCatalogError, pattern):
                self.validate_clock(value, **options)

    def test_clock_aarch64_exact_rate_and_measured_affine_reject(self):
        opening = self.opening()
        opening["clock_kind"] = 2
        value = self.clock_catalog(opening, architecture=2, source=2)
        row = self.validate_clock(
            value,
            opening=opening,
            architecture_id=2,
        )
        self.assertEqual(row["acquisition_source_id"], 2)
        measured = self.opening(calibration=2, error=1, span=1002)
        measured["clock_kind"] = 2
        with self.assertRaisesRegex(self.module.RuntimeCatalogError, "exact architectural rate"):
            self.validate_clock(
                self.clock_catalog(measured, architecture=2, source=4),
                opening=measured,
                architecture_id=2,
            )
        mutant = self.clock_catalog(); mutant["entries"][0]["acquisition_status_id"] = True
        with self.assertRaisesRegex(self.module.RuntimeCatalogError, "status 1"):
            self.validate_clock(mutant)

    def test_counter_sentinels_ranges_and_invalid_observation(self):
        row = self.validate_clock(self.clock_catalog())
        cases = [
            (SimpleNamespace(flags=0, counter_enter=1, counter_exit=0), "canonical zero"),
            (SimpleNamespace(flags=1 << 5, counter_enter=999, counter_exit=0), "outside opening/final"),
            (SimpleNamespace(flags=0x60, counter_enter=1200, counter_exit=1100), "precedes"),
        ]
        for record, pattern in cases:
            with self.subTest(pattern=pattern), self.assertRaisesRegex(self.module.RuntimeCatalogError, pattern):
                self.module.reconcile_record_counters(row, final_counter=2000, records=[record])
        with self.assertRaisesRegex(self.module.RuntimeCatalogError, "lacks explicit observation"):
            self.module.reconcile_record_counters(
                row, final_counter=2000,
                records=[SimpleNamespace(flags=0, counter_enter=0, counter_exit=0)],
            )

    def test_canonical_schema_and_definition_mutants_reject(self):
        value = self.thread_catalog()
        value["schema"]["object_type"] = 12
        with self.assertRaisesRegex(self.module.RuntimeCatalogError, "object type required"):
            self.validate_thread(value)
        raw = self.module.SEMANTIC.canonical_bytes(self.thread_catalog())
        with self.assertRaisesRegex(self.module.RuntimeCatalogError, "canonical serialization"):
            self.module.validate_thread_catalog_bytes(
                raw.replace(b'"catalog_id"', b' "catalog_id"', 1),
                expected_process_generation=7,
                configured_thread_capacity=4,
                record_count=2,
                mode_id=4,
                population_state=1,
                lifecycle_state=5,
            )
        definition = copy.deepcopy(self.module.RUNTIME_DEFINITION)
        definition["version"]["major"] = True
        with self.assertRaisesRegex(self.module.RuntimeCatalogError, "exact type"):
            self.module.validate_definition(definition)


if __name__ == "__main__":
    unittest.main()
