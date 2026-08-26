#!/usr/bin/env python3
"""Deterministic tests for the concrete schema-v1 callsite catalog."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "callsite_catalog_v1.py"
DEFINITION = ROOT.parent / "semantic/archive/definition/callsite-rule-v1.json"

FROZEN_MEMBER_5_DEFINITION_SHA256 = (
    "510c852c65888dcf563d10e9e416ad0ab96c8503af1e412e06ff75dcb14caa18"
)
FROZEN_MEMBER_5_SCHEMA_PATH = "definition/callsite-rule-v1.json"
FROZEN_A05_CANONICAL_SHA256 = (
    "e697152c8bbefeec792c9397120dc7727d405692127159dcb6d644c1ecbb698b"
)
FROZEN_A05_EXPECTED = {
    "catalog_id": "oai_memprof_callsite",
    "entries": [
        {
            "callsite_id": 1,
            "module_generation": 2,
            "module_id": 4,
            "process_generation": 7,
            "raw_address": 0,
        },
        {
            "callsite_id": 8,
            "module_generation": 8,
            "module_id": 9,
            "process_generation": 7,
            "raw_address": 4660,
        },
    ],
    "schema": {
        "object_type": 5,
        "path": FROZEN_MEMBER_5_SCHEMA_PATH,
        "sha256": FROZEN_MEMBER_5_DEFINITION_SHA256,
    },
    "version": {"major": 1, "minor": 0},
}


def load_module():
    spec = importlib.util.spec_from_file_location("callsite_catalog_v1_under_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("module loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CallsiteCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()
        cls.definition_sha256 = FROZEN_MEMBER_5_DEFINITION_SHA256
        cls.modules = ((2, 4), (8, 9))
        cls.rows = (
            {"callsite_id": 1, "module_generation": 2, "module_id": 4,
             "process_generation": 7, "raw_address": 0},
            {"callsite_id": 8, "module_generation": 8, "module_id": 9,
             "process_generation": 7, "raw_address": 0x1234},
        )

    def make(self, *, rows=None, mode_id=5):
        return self.module.serialize_callsite_catalog(
            definition_sha256=self.definition_sha256,
            entries=self.rows if rows is None else rows,
            expected_process_generation=7,
            mode_id=mode_id,
            module_keys=self.modules,
        )

    def validate(self, raw: bytes, **overrides):
        options = {
            "definition_sha256": self.definition_sha256,
            "expected_process_generation": 7,
            "mode_id": 5,
            "module_keys": self.modules,
        }
        options.update(overrides)
        return self.module.validate_callsite_catalog_bytes(raw, **options)

    def test_a05_literal_is_canonical_resolved_and_bound(self) -> None:
        raw = self.make()
        value, keys = self.validate(raw)
        self.assertEqual(
            hashlib.sha256(DEFINITION.read_bytes()).hexdigest(),
            FROZEN_MEMBER_5_DEFINITION_SHA256,
        )
        self.assertEqual(hashlib.sha256(raw).hexdigest(), FROZEN_A05_CANONICAL_SHA256)
        self.assertEqual(value, FROZEN_A05_EXPECTED)
        self.assertEqual(value["schema"]["path"], FROZEN_MEMBER_5_SCHEMA_PATH)
        self.assertEqual(
            value["schema"]["sha256"], FROZEN_MEMBER_5_DEFINITION_SHA256
        )
        self.assertEqual(
            self.module.validate_callsite_catalog(
                FROZEN_A05_EXPECTED,
                definition_sha256=FROZEN_MEMBER_5_DEFINITION_SHA256,
                expected_process_generation=7,
                mode_id=5,
                module_keys=((2, 4), (8, 9)),
            ),
            {(7, 1), (7, 8)},
        )
        self.assertEqual(self.module.SEMANTIC.canonical_bytes(value), raw)
        self.assertEqual(keys, {(7, 1), (7, 8)})
        arguments = {
            "definition_sha256": self.definition_sha256,
            "expected_process_generation": 7,
            "mode_id": 5,
            "module_keys": self.modules,
        }
        descriptor = self.module.semantic_object_descriptor(raw, **arguments)
        self.assertEqual(
            descriptor,
            {"byte_count": 416, "entry_count": 2, "object_flags": 0x1B,
             "format_id": 1, "object_kind": 4, "path": "catalog/callsite.json",
             "schema_revision": 1, "sha256": FROZEN_A05_CANONICAL_SHA256},
        )
        self.module.validate_semantic_object_descriptor(descriptor, raw, **arguments)
        for field, wrong in (
            ("entry_count", 0),
            ("entry_count", 3),
            ("entry_count", True),
            ("format_id", True),
            ("schema_revision", True),
            ("byte_count", len(raw) + 1),
            ("sha256", "f" * 64),
            ("object_flags", 0x1A),
        ):
            changed = dict(descriptor)
            changed[field] = wrong
            with self.subTest(field=field, wrong=wrong), self.assertRaisesRegex(
                self.module.CallsiteError, "snapshot metadata"
            ):
                self.module.validate_semantic_object_descriptor(
                    changed, raw, **arguments
                )
        mutated = raw.replace(b'"raw_address":0', b'"raw_address":1', 1)
        with self.assertRaisesRegex(self.module.CallsiteError, "snapshot metadata"):
            self.module.validate_semantic_object_descriptor(
                descriptor, mutated, **arguments
            )

    def test_non_a05_requires_empty_catalog_and_zero_record_ids(self) -> None:
        raw = self.make(rows=(), mode_id=4)
        _value, keys = self.validate(raw, mode_id=4)
        self.assertEqual(keys, set())
        self.module.reconcile_records(
            keys, process_generation=7, mode_id=4,
            records=(SimpleNamespace(callsite_id=0),),
        )
        with self.assertRaisesRegex(self.module.CallsiteError, "canonical zero"):
            self.module.reconcile_records(
                keys, process_generation=7, mode_id=4,
                records=(SimpleNamespace(callsite_id=1),),
            )
        with self.assertRaisesRegex(self.module.CallsiteError, "only A05"):
            self.make(mode_id=4)

    def test_a05_requires_nonempty_catalog_and_every_record_resolves(self) -> None:
        with self.assertRaisesRegex(self.module.CallsiteError, "at least one"):
            self.make(rows=(), mode_id=5)
        _value, keys = self.validate(self.make())
        self.module.reconcile_records(
            keys, process_generation=7, mode_id=5,
            records=(SimpleNamespace(callsite_id=1), SimpleNamespace(callsite_id=8)),
        )
        for bad_id in (0, 2):
            with self.subTest(bad_id=bad_id), self.assertRaisesRegex(
                self.module.CallsiteError, "exact resolved nonzero"
            ):
                self.module.reconcile_records(
                    keys, process_generation=7, mode_id=5,
                    records=(SimpleNamespace(callsite_id=bad_id),),
                )
        for records in (
            (),
            (SimpleNamespace(callsite_id=1),),
        ):
            with self.subTest(records=records), self.assertRaisesRegex(
                self.module.CallsiteError, "sets must be equal"
            ):
                self.module.reconcile_records(
                    keys, process_generation=7, mode_id=5, records=records
                )
        self.module.reconcile_records(
            keys,
            process_generation=7,
            mode_id=5,
            records=(SimpleNamespace(callsite_id=1),),
            complete=False,
        )
        for mode in (7, 65535, True):
            with self.subTest(mode=mode), self.assertRaises(self.module.CallsiteError):
                self.module.reconcile_records(
                    set(), process_generation=7, mode_id=mode, records=()
                )

    def test_generation_module_order_duplicate_and_type_mutants_reject(self) -> None:
        base = self.module.SEMANTIC.parse_canonical(self.make())
        mutants = []
        wrong_generation = copy.deepcopy(base)
        wrong_generation["entries"][0]["process_generation"] = 8
        mutants.append((wrong_generation, "generation mismatch"))
        wrong_module = copy.deepcopy(base)
        wrong_module["entries"][0]["module_id"] = 5
        mutants.append((wrong_module, "unresolved module"))
        reverse = copy.deepcopy(base)
        reverse["entries"].reverse()
        mutants.append((reverse, "strict process-generation"))
        bool_id = copy.deepcopy(base)
        bool_id["entries"][0]["callsite_id"] = True
        mutants.append((bool_id, "u32 integer"))
        for mutant, pattern in mutants:
            with self.subTest(pattern=pattern), self.assertRaisesRegex(
                self.module.CallsiteError, pattern
            ):
                self.module.validate_callsite_catalog(
                    mutant, definition_sha256=self.definition_sha256,
                    expected_process_generation=7, mode_id=5,
                    module_keys=self.modules,
                )
        duplicate_raw = copy.deepcopy(base)
        duplicate_raw["entries"][1]["module_generation"] = 2
        duplicate_raw["entries"][1]["module_id"] = 4
        duplicate_raw["entries"][1]["raw_address"] = 0
        self.module.validate_callsite_catalog(
            duplicate_raw,
            definition_sha256=self.definition_sha256,
            expected_process_generation=7,
            mode_id=5,
            module_keys=self.modules,
        )

    def test_object_limits_are_enforced_before_descriptor_publication(self) -> None:
        raw = self.make()
        original_byte_limit = self.module.MAX_OBJECT_BYTES
        original_entry_limit = self.module.MAX_OBJECT_ENTRIES
        try:
            self.module.MAX_OBJECT_BYTES = len(raw) - 1
            with self.assertRaisesRegex(self.module.CallsiteError, "byte limit"):
                self.module.semantic_object_descriptor(
                    raw,
                    definition_sha256=self.definition_sha256,
                    expected_process_generation=7,
                    mode_id=5,
                    module_keys=self.modules,
                )
            self.module.MAX_OBJECT_BYTES = original_byte_limit
            self.module.MAX_OBJECT_ENTRIES = 1
            with self.assertRaisesRegex(self.module.CallsiteError, "entry limit"):
                self.module.semantic_object_descriptor(
                    raw,
                    definition_sha256=self.definition_sha256,
                    expected_process_generation=7,
                    mode_id=5,
                    module_keys=self.modules,
                )
        finally:
            self.module.MAX_OBJECT_BYTES = original_byte_limit
            self.module.MAX_OBJECT_ENTRIES = original_entry_limit

    def test_schema_and_canonical_mutants_reject(self) -> None:
        raw = self.make()
        with self.assertRaisesRegex(self.module.CallsiteError, "definition digest mismatch"):
            self.validate(raw, definition_sha256="0" * 64)
        with self.assertRaisesRegex(self.module.CallsiteError, "canonical serialization"):
            self.validate(raw.replace(b'"catalog_id"', b' "catalog_id"', 1))
        value = self.module.SEMANTIC.parse_canonical(raw)
        value["version"]["major"] = True
        with self.assertRaisesRegex(self.module.CallsiteError, "typed version"):
            self.module.validate_callsite_catalog(
                value, definition_sha256=self.definition_sha256,
                expected_process_generation=7, mode_id=5,
                module_keys=self.modules,
            )


if __name__ == "__main__":
    unittest.main()
