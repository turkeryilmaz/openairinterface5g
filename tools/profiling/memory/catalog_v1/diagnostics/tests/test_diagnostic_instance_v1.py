#!/usr/bin/env python3
"""Deterministic tests for the schema-v1 diagnostics instance projection."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "diagnostic_instance_v1.py"
SEMANTIC_ROOT = ROOT.parent / "semantic"
DEFINITION_PATH = SEMANTIC_ROOT / "archive" / "definition" / "diagnostic-v1.json"

FROZEN_MEMBER_8_DEFINITION_SHA256 = (
    "f46d99a638da45105006fa5bdd70547674aa948fd1c35012c68d5dce2a274162"
)
FROZEN_MEMBER_8_SCHEMA_PATH = "definition/diagnostic-v1.json"
FROZEN_MODE_3_CANONICAL_SHA256 = (
    "8d1f67cb94c6208c4b608188743acb335cb91e8706183082b95c438d12b53678"
)
FROZEN_A05_MODE_5_COUNTER_KEYS = (
    (9, 1, 2, 1),
    (9, 1, 2, 16),
    (9, 1, 2, 17),
    (9, 1, 2, 18),
    (9, 1, 2, 32),
    (9, 1, 2, 64),
    (9, 1, 7, 1),
    (9, 1, 7, 16),
    (9, 1, 7, 17),
    (9, 1, 7, 18),
    (9, 1, 7, 32),
    (9, 1, 7, 64),
    (9, 2, 1, 2),
    (9, 2, 1, 3),
    (9, 3, 1, 80),
    (9, 4, 1, 96),
)
FROZEN_A05_MODE_5_REASON_64_PRODUCER_KEYS = (
    (9, 1, 2, 64),
    (9, 1, 7, 64),
)
FROZEN_A05_MODE_5_ABSENT_REASON_IDS = (48, 49, 50, 51)
FROZEN_MODE_3_EXPECTED_CRITICAL = {
    "catalog_id": "oai_memprof_diagnostics",
    "mode_id": 3,
    "process_generation": 9,
    "schema": {
        "object_type": 8,
        "path": FROZEN_MEMBER_8_SCHEMA_PATH,
        "sha256": FROZEN_MEMBER_8_DEFINITION_SHA256,
    },
    "version": {"major": 1, "minor": 0},
    "entry_count": 24,
    "first_entry": {
        "counter_scope_id": 2,
        "counter_scope_kind": 1,
        "process_generation": 9,
        "reason_id": 1,
        "saturated": False,
        "value": 0,
    },
    "last_entry": {
        "counter_scope_id": 1,
        "counter_scope_kind": 4,
        "process_generation": 9,
        "reason_id": 96,
        "saturated": False,
        "value": 0,
    },
    "reason_totals": (
        {"class_flags": 257, "nonzero_counter_instances": 0, "reason_id": 1,
         "saturated_counter_instances": 0, "saturating_total": 0, "summary_flags": 0},
        {"class_flags": 257, "nonzero_counter_instances": 0, "reason_id": 2,
         "saturated_counter_instances": 0, "saturating_total": 0, "summary_flags": 0},
        {"class_flags": 257, "nonzero_counter_instances": 0, "reason_id": 3,
         "saturated_counter_instances": 0, "saturating_total": 0, "summary_flags": 0},
        {"class_flags": 258, "nonzero_counter_instances": 0, "reason_id": 16,
         "saturated_counter_instances": 0, "saturating_total": 0, "summary_flags": 0},
        {"class_flags": 4, "nonzero_counter_instances": 0, "reason_id": 17,
         "saturated_counter_instances": 0, "saturating_total": 0, "summary_flags": 0},
        {"class_flags": 4, "nonzero_counter_instances": 0, "reason_id": 18,
         "saturated_counter_instances": 0, "saturating_total": 0, "summary_flags": 0},
        {"class_flags": 8, "nonzero_counter_instances": 0, "reason_id": 32,
         "saturated_counter_instances": 0, "saturating_total": 0, "summary_flags": 0},
        {"class_flags": 528, "nonzero_counter_instances": 0, "reason_id": 48,
         "saturated_counter_instances": 0, "saturating_total": 0, "summary_flags": 0},
        {"class_flags": 528, "nonzero_counter_instances": 0, "reason_id": 49,
         "saturated_counter_instances": 0, "saturating_total": 0, "summary_flags": 0},
        {"class_flags": 528, "nonzero_counter_instances": 0, "reason_id": 50,
         "saturated_counter_instances": 0, "saturating_total": 0, "summary_flags": 0},
        {"class_flags": 528, "nonzero_counter_instances": 0, "reason_id": 51,
         "saturated_counter_instances": 0, "saturating_total": 0, "summary_flags": 0},
        {"class_flags": 320, "nonzero_counter_instances": 0, "reason_id": 64,
         "saturated_counter_instances": 0, "saturating_total": 0, "summary_flags": 0},
        {"class_flags": 288, "nonzero_counter_instances": 0, "reason_id": 80,
         "saturated_counter_instances": 0, "saturating_total": 0, "summary_flags": 0},
        {"class_flags": 384, "nonzero_counter_instances": 0, "reason_id": 96,
         "saturated_counter_instances": 0, "saturating_total": 0, "summary_flags": 0},
    ),
    "projection": {
        "aggregate_saturated": False,
        "diagnostic_bypass_sum": 0,
        "diagnostic_loss_sum": 0,
        "population_partial": False,
        "saturated_counter_instances": 0,
    },
}


def load_module():
    spec = importlib.util.spec_from_file_location("diagnostic_instance_v1_under_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("module loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DiagnosticInstanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()
        cls.definition_digest = FROZEN_MEMBER_8_DEFINITION_SHA256

    def rows(
        self,
        *,
        mode_id: int = 3,
        threads: tuple[int, ...] = (2, 7),
        saturated_reason: int | None = None,
        registration_available: bool = True,
        writer_available: bool = True,
        aggregate_available: bool = True,
    ) -> list[dict[str, object]]:
        module = self.module
        aggregate, not_applicable, producer, registration, writer = module._mode_scopes(mode_id)
        rows: list[dict[str, object]] = []
        for reason_id in module.REASON_IDS:
            scopes: list[tuple[int, int]] = []
            if reason_id in not_applicable:
                scopes = []
            elif reason_id in producer:
                scopes = [(1, thread_id) for thread_id in threads]
            elif reason_id in registration and registration_available:
                scopes = [(2, 1)]
            elif reason_id in writer and writer_available:
                scopes = [(3, 1)]
            elif reason_id in aggregate and aggregate_available:
                scopes = [(4, 1)]
            for scope_kind, scope_id in scopes:
                saturated = reason_id == saturated_reason and scope_kind == 1 and scope_id == threads[0]
                value = module.UINT64_MAX if saturated else 0
                if reason_id == 96:
                    value = 1 if saturated_reason is not None else 0
                rows.append(
                    {
                        "counter_scope_id": scope_id,
                        "counter_scope_kind": scope_kind,
                        "process_generation": 9,
                        "reason_id": reason_id,
                        "saturated": saturated,
                        "value": value,
                    }
                )
        rows.sort(
            key=lambda row: (
                row["process_generation"],
                row["counter_scope_kind"],
                row["counter_scope_id"],
                row["reason_id"],
            )
        )
        return rows

    def make(
        self,
        *,
        rows: list[dict[str, object]] | None = None,
        producer_complete: bool = True,
        registration_available: bool = True,
    ) -> bytes:
        if rows is None:
            rows = self.rows(registration_available=registration_available)
        return self.module.make_diagnostics_bytes(
            definition_sha256=self.definition_digest,
            mode_id=3,
            process_generation=9,
            counter_rows=rows,
            ready_thread_indices=(2, 7),
            producer_population_complete=producer_complete,
            registration_available=registration_available,
        )

    def validate(self, raw: bytes, **overrides):
        options = {
            "definition_sha256": self.definition_digest,
            "expected_mode_id": 3,
            "expected_process_generation": 9,
            "ready_thread_indices": (2, 7),
            "producer_population_complete": True,
        }
        options.update(overrides)
        return self.module.validate_diagnostics_bytes(raw, **options)

    def test_complete_fixture_projects_exactly_and_is_canonical(self) -> None:
        raw = self.make()
        value, projection = self.validate(raw)
        self.assertEqual(
            hashlib.sha256(DEFINITION_PATH.read_bytes()).hexdigest(),
            FROZEN_MEMBER_8_DEFINITION_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(), FROZEN_MODE_3_CANONICAL_SHA256
        )
        critical = {
            "catalog_id": value["catalog_id"],
            "mode_id": value["mode_id"],
            "process_generation": value["process_generation"],
            "schema": value["schema"],
            "version": value["version"],
            "entry_count": len(value["entries"]),
            "first_entry": value["entries"][0],
            "last_entry": value["entries"][-1],
            "reason_totals": tuple(dict(row) for row in value["reason_totals"]),
            "projection": {
                "aggregate_saturated": projection["aggregate_saturated"],
                "diagnostic_bypass_sum": projection["diagnostic_bypass_sum"],
                "diagnostic_loss_sum": projection["diagnostic_loss_sum"],
                "population_partial": projection["population_partial"],
                "saturated_counter_instances": projection["saturated_counter_instances"],
            },
        }
        self.assertEqual(critical, FROZEN_MODE_3_EXPECTED_CRITICAL)
        self.assertEqual(value["schema"]["path"], FROZEN_MEMBER_8_SCHEMA_PATH)
        self.assertEqual(
            value["schema"]["sha256"], FROZEN_MEMBER_8_DEFINITION_SHA256
        )
        self.assertEqual(self.module.SEMANTIC.canonical_bytes(value), raw)
        self.assertEqual(projection["entry_count"], 24)
        self.assertEqual(len(projection["reason_totals"]), 14)
        self.assertFalse(projection["aggregate_saturated"])
        self.assertFalse(projection["population_partial"])
        self.assertEqual(projection["diagnostic_loss_sum"], 0)
        self.assertEqual(projection["diagnostic_bypass_sum"], 0)
        self.assertEqual(projection["saturated_counter_instances"], 0)

    def test_every_complete_mode_uses_its_exact_scope_population(self) -> None:
        expected_counts = {2: 12, 3: 24, 4: 16, 5: 16}
        for mode_id, expected_count in expected_counts.items():
            with self.subTest(mode_id=mode_id):
                rows = self.rows(mode_id=mode_id)
                raw = self.module.make_diagnostics_bytes(
                    definition_sha256=self.definition_digest,
                    mode_id=mode_id,
                    process_generation=9,
                    counter_rows=rows,
                    ready_thread_indices=(2, 7),
                    producer_population_complete=True,
                )
                _value, projection = self.module.validate_diagnostics_bytes(
                    raw,
                    definition_sha256=self.definition_digest,
                    expected_mode_id=mode_id,
                    expected_process_generation=9,
                    ready_thread_indices=(2, 7),
                    producer_population_complete=True,
                )
                self.assertEqual(projection["entry_count"], expected_count)
                self.assertFalse(projection["population_partial"])

    def test_a05_mode_5_uses_fixed_scope_membership(self) -> None:
        rows = [
            {
                "counter_scope_id": scope_id,
                "counter_scope_kind": scope_kind,
                "process_generation": generation,
                "reason_id": reason_id,
                "saturated": False,
                "value": 0,
            }
            for generation, scope_kind, scope_id, reason_id in FROZEN_A05_MODE_5_COUNTER_KEYS
        ]
        raw = self.module.make_diagnostics_bytes(
            definition_sha256=self.definition_digest,
            mode_id=5,
            process_generation=9,
            counter_rows=rows,
            ready_thread_indices=(2, 7),
            producer_population_complete=True,
        )
        value, projection = self.module.validate_diagnostics_bytes(
            raw,
            definition_sha256=self.definition_digest,
            expected_mode_id=5,
            expected_process_generation=9,
            ready_thread_indices=(2, 7),
            producer_population_complete=True,
        )
        actual_keys = tuple(
            (
                row["process_generation"],
                row["counter_scope_kind"],
                row["counter_scope_id"],
                row["reason_id"],
            )
            for row in value["entries"]
        )
        self.assertEqual(actual_keys, FROZEN_A05_MODE_5_COUNTER_KEYS)
        self.assertEqual(
            tuple(key for key in actual_keys if key[3] == 64),
            FROZEN_A05_MODE_5_REASON_64_PRODUCER_KEYS,
        )
        self.assertFalse(
            {key[3] for key in actual_keys} & set(FROZEN_A05_MODE_5_ABSENT_REASON_IDS)
        )
        self.assertEqual(projection["entry_count"], len(FROZEN_A05_MODE_5_COUNTER_KEYS))
        self.assertFalse(projection["population_partial"])

    def test_saturation_projection_and_terminal_reconciliation(self) -> None:
        raw = self.make(rows=self.rows(saturated_reason=1))
        _value, projection = self.validate(raw)
        self.assertTrue(projection["aggregate_saturated"])
        self.assertEqual(projection["diagnostic_loss_sum"], self.module.UINT64_MAX)
        self.assertEqual(projection["saturated_counter_instances"], 1)
        terminal = [SimpleNamespace(**row) for row in projection["reason_totals"]]
        self.module.reconcile_terminal(
            projection,
            terminal_entries=terminal,
            terminal_flags=self.module.TERMINAL_AGGREGATE_SATURATED,
            diagnostic_loss_sum=self.module.UINT64_MAX,
            diagnostic_bypass_sum=0,
            saturated_counter_instances=1,
        )
        with self.assertRaisesRegex(self.module.DiagnosticError, "aggregate-saturated bit mismatch"):
            self.module.reconcile_terminal(
                projection,
                terminal_entries=terminal,
                terminal_flags=0,
                diagnostic_loss_sum=self.module.UINT64_MAX,
                diagnostic_bypass_sum=0,
                saturated_counter_instances=1,
            )

    def test_partial_producer_population_is_explicit_not_zero(self) -> None:
        raw = self.make(producer_complete=False)
        _value, projection = self.validate(raw, producer_population_complete=False)
        self.assertTrue(projection["population_partial"])
        producer_reason_ids = {1, 16, 17, 18, 32, 48, 49, 50, 51, 64}
        for row in projection["reason_totals"]:
            self.assertEqual(
                bool(row["summary_flags"] & self.module.PARTIAL_COUNTER_POPULATION),
                row["reason_id"] in producer_reason_ids,
            )
        terminal = projection["reason_totals"]
        self.module.reconcile_terminal(
            projection,
            terminal_entries=terminal,
            terminal_flags=self.module.TERMINAL_POPULATION_PARTIAL,
            diagnostic_loss_sum=0,
            diagnostic_bypass_sum=0,
            saturated_counter_instances=0,
        )

    def test_unavailable_registration_has_only_explicit_partial_totals(self) -> None:
        rows = self.rows(registration_available=False)
        raw = self.make(rows=rows, registration_available=False)
        _value, projection = self.validate(raw, registration_available=False)
        partial = {
            row["reason_id"]
            for row in projection["reason_totals"]
            if row["summary_flags"] & self.module.PARTIAL_COUNTER_POPULATION
        }
        self.assertEqual(partial, {2, 3})
        self.assertNotIn(2, {row["reason_id"] for row in rows})
        self.assertNotIn(3, {row["reason_id"] for row in rows})

    def test_one_unavailable_counter_preserves_other_rows_and_marks_one_reason(self) -> None:
        rows = self.rows()
        missing_key = (9, 2, 1, 3)
        rows = [
            row
            for row in rows
            if (
                row["process_generation"],
                row["counter_scope_kind"],
                row["counter_scope_id"],
                row["reason_id"],
            )
            != missing_key
        ]
        raw = self.module.make_diagnostics_bytes(
            definition_sha256=self.definition_digest,
            mode_id=3,
            process_generation=9,
            counter_rows=rows,
            ready_thread_indices=(2, 7),
            producer_population_complete=True,
            unavailable_counter_keys=(missing_key,),
        )
        _value, projection = self.module.validate_diagnostics_bytes(
            raw,
            definition_sha256=self.definition_digest,
            expected_mode_id=3,
            expected_process_generation=9,
            ready_thread_indices=(2, 7),
            producer_population_complete=True,
            unavailable_counter_keys=(missing_key,),
        )
        partial = {
            row["reason_id"]
            for row in projection["reason_totals"]
            if row["summary_flags"] & self.module.PARTIAL_COUNTER_POPULATION
        }
        self.assertEqual(partial, {3})
        self.assertIn(2, {row["reason_id"] for row in rows})
        with self.assertRaisesRegex(self.module.DiagnosticError, "required counter row missing"):
            self.module.validate_diagnostics_bytes(
                raw,
                definition_sha256=self.definition_digest,
                expected_mode_id=3,
                expected_process_generation=9,
                ready_thread_indices=(2, 7),
                producer_population_complete=True,
            )
        present_key = (9, 2, 1, 2)
        with self.assertRaisesRegex(self.module.DiagnosticError, "declared-unavailable counter is present"):
            self.module.validate_diagnostics_bytes(
                raw,
                definition_sha256=self.definition_digest,
                expected_mode_id=3,
                expected_process_generation=9,
                ready_thread_indices=(2, 7),
                producer_population_complete=True,
                unavailable_counter_keys=(present_key,),
            )

    def test_one_known_producer_counter_can_be_explicitly_unavailable(self) -> None:
        missing_key = (9, 1, 2, 1)
        rows = [
            row
            for row in self.rows()
            if (
                row["process_generation"], row["counter_scope_kind"],
                row["counter_scope_id"], row["reason_id"],
            ) != missing_key
        ]
        raw = self.module.make_diagnostics_bytes(
            definition_sha256=self.definition_digest,
            mode_id=3,
            process_generation=9,
            counter_rows=rows,
            ready_thread_indices=(2, 7),
            producer_population_complete=True,
            unavailable_counter_keys=(missing_key,),
        )
        _value, projection = self.module.validate_diagnostics_bytes(
            raw,
            definition_sha256=self.definition_digest,
            expected_mode_id=3,
            expected_process_generation=9,
            ready_thread_indices=(2, 7),
            producer_population_complete=True,
            unavailable_counter_keys=(missing_key,),
        )
        partial = {
            row["reason_id"] for row in projection["reason_totals"]
            if row["summary_flags"] & self.module.PARTIAL_COUNTER_POPULATION
        }
        self.assertEqual(partial, {1})
        with self.assertRaisesRegex(self.module.DiagnosticError, "required counter row missing"):
            self.module.make_diagnostics_bytes(
                definition_sha256=self.definition_digest,
                mode_id=3,
                process_generation=9,
                counter_rows=rows,
                ready_thread_indices=(2, 7),
                producer_population_complete=False,
            )
        raw = self.module.make_diagnostics_bytes(
            definition_sha256=self.definition_digest,
            mode_id=3,
            process_generation=9,
            counter_rows=rows,
            ready_thread_indices=(2, 7),
            producer_population_complete=False,
            unavailable_counter_keys=(missing_key,),
        )
        _value, projection = self.module.validate_diagnostics_bytes(
            raw,
            definition_sha256=self.definition_digest,
            expected_mode_id=3,
            expected_process_generation=9,
            ready_thread_indices=(2, 7),
            producer_population_complete=False,
            unavailable_counter_keys=(missing_key,),
        )
        self.assertTrue(projection["population_partial"])

    def test_partial_id96_still_obeys_finite_schema_population_bound(self) -> None:
        rows = self.rows(registration_available=False)
        for row in rows:
            if row["reason_id"] == 96:
                row["value"] = self.module.UINT64_MAX
        with self.assertRaisesRegex(self.module.DiagnosticError, "finite schema-v1"):
            self.module.make_diagnostics_bytes(
                definition_sha256=self.definition_digest,
                mode_id=3,
                process_generation=9,
                counter_rows=rows,
                ready_thread_indices=(2, 7),
                producer_population_complete=True,
                registration_available=False,
            )

    def test_required_population_missing_extra_and_order_reject(self) -> None:
        rows = self.rows()
        with self.assertRaisesRegex(self.module.DiagnosticError, "required counter row missing"):
            self.make(rows=rows[1:])
        extra = copy.deepcopy(rows)
        extra.append(
            {
                "counter_scope_id": 99,
                "counter_scope_kind": 4,
                "process_generation": 9,
                "reason_id": 96,
                "saturated": False,
                "value": 0,
            }
        )
        with self.assertRaisesRegex(self.module.DiagnosticError, "not in the admitted mode population"):
            self.make(rows=extra)
        reversed_rows = list(reversed(rows))
        with self.assertRaisesRegex(self.module.DiagnosticError, "strict counter-key order"):
            self.make(rows=reversed_rows)

    def test_reason_total_and_id96_mutants_reject(self) -> None:
        raw = self.make()
        value = self.module.SEMANTIC.parse_canonical(raw)
        bad_total = copy.deepcopy(value)
        bad_total["reason_totals"][0]["class_flags"] ^= 1
        with self.assertRaisesRegex(self.module.DiagnosticError, "authoritative counter projection"):
            self.module.validate_diagnostics(
                bad_total,
                definition_sha256=self.definition_digest,
                expected_mode_id=3,
                expected_process_generation=9,
                ready_thread_indices=(2, 7),
                producer_population_complete=True,
            )
        bad_id96 = copy.deepcopy(value)
        for row in bad_id96["entries"]:
            if row["reason_id"] == 96:
                row["value"] = 1
        bad_id96["reason_totals"][-1]["nonzero_counter_instances"] = 1
        bad_id96["reason_totals"][-1]["saturating_total"] = 1
        with self.assertRaisesRegex(self.module.DiagnosticError, "saturation-transition reconciliation"):
            self.module.validate_diagnostics(
                bad_id96,
                definition_sha256=self.definition_digest,
                expected_mode_id=3,
                expected_process_generation=9,
                ready_thread_indices=(2, 7),
                producer_population_complete=True,
            )

    def test_schema_mode_generation_and_boolean_types_reject(self) -> None:
        raw = self.make()
        value = self.module.SEMANTIC.parse_canonical(raw)
        with self.assertRaisesRegex(self.module.DiagnosticError, "definition digest mismatch"):
            self.validate(raw, definition_sha256="0" * 64)
        with self.assertRaisesRegex(self.module.DiagnosticError, "expected mode mismatch"):
            self.validate(raw, expected_mode_id=4)
        with self.assertRaisesRegex(self.module.DiagnosticError, "expected generation mismatch"):
            self.validate(raw, expected_process_generation=10)
        mutant = copy.deepcopy(value)
        mutant["entries"][0]["saturated"] = 1
        with self.assertRaisesRegex(self.module.DiagnosticError, "boolean required"):
            self.module.validate_diagnostics(
                mutant,
                definition_sha256=self.definition_digest,
                expected_mode_id=3,
                expected_process_generation=9,
                ready_thread_indices=(2, 7),
                producer_population_complete=True,
            )
        mutant = copy.deepcopy(value)
        mutant["version"]["major"] = True
        with self.assertRaisesRegex(self.module.DiagnosticError, "exact version 1.0"):
            self.module.validate_diagnostics(
                mutant,
                definition_sha256=self.definition_digest,
                expected_mode_id=3,
                expected_process_generation=9,
                ready_thread_indices=(2, 7),
                producer_population_complete=True,
            )

    def test_noncanonical_bytes_and_terminal_row_mutant_reject(self) -> None:
        raw = self.make()
        with self.assertRaisesRegex(self.module.DiagnosticError, "canonical serialization"):
            self.validate(raw.replace(b'"catalog_id"', b' "catalog_id"', 1))
        _value, projection = self.validate(raw)
        terminal = copy.deepcopy(projection["reason_totals"])
        terminal[0]["saturating_total"] = 1
        with self.assertRaisesRegex(self.module.DiagnosticError, "diagnostics projection mismatch"):
            self.module.reconcile_terminal(
                projection,
                terminal_entries=terminal,
                terminal_flags=0,
                diagnostic_loss_sum=0,
                diagnostic_bypass_sum=0,
                saturated_counter_instances=0,
            )
        malformed = copy.deepcopy(projection["reason_totals"])
        malformed[0]["extra"] = 0
        with self.assertRaisesRegex(self.module.DiagnosticError, "exact diagnostic-entry fields"):
            self.module.reconcile_terminal(
                projection,
                terminal_entries=malformed,
                terminal_flags=0,
                diagnostic_loss_sum=0,
                diagnostic_bypass_sum=0,
                saturated_counter_instances=0,
            )
        for terminal in (
            copy.deepcopy(projection["reason_totals"]),
            [SimpleNamespace(**row) for row in projection["reason_totals"]],
        ):
            if isinstance(terminal[0], dict):
                terminal[0]["reason_id"] = True
            else:
                terminal[0].reason_id = True
            with self.assertRaisesRegex(self.module.DiagnosticError, "u16 integer required"):
                self.module.reconcile_terminal(
                    projection,
                    terminal_entries=terminal,
                    terminal_flags=0,
                    diagnostic_loss_sum=0,
                    diagnostic_bypass_sum=0,
                    saturated_counter_instances=0,
                )


if __name__ == "__main__":
    unittest.main()
