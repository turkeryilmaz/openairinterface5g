#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "selection_rule_v1.py"
SPEC = importlib.util.spec_from_file_location("selection_rule_v1_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
sel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sel)


class SelectionRuleTests(unittest.TestCase):
    @staticmethod
    def _definition_only_mapping(entry, generation, thread, sequence):
        mask = (1 << 64) - 1
        constants = entry["domain_constants"]
        multipliers = entry["mix_multipliers"]

        def mix(value):
            value ^= value >> 30
            value = (value * multipliers["multiplier_1"]) & mask
            value ^= value >> 27
            value = (value * multipliers["multiplier_2"]) & mask
            value ^= value >> 31
            return value & mask

        def rotl(value, distance):
            return ((value << distance) | (value >> (64 - distance))) & mask

        thread_word = ((thread << 32) | thread) & mask
        state = mix(
            constants["initial_state"]
            ^ generation
            ^ constants["generation_domain"]
        )
        state = mix(
            state
            ^ rotl(
                thread_word ^ constants["thread_domain"],
                entry["rotations"]["thread"],
            )
        )
        return mix(
            state
            ^ rotl(
                sequence ^ constants["sequence_domain"],
                entry["rotations"]["sequence"],
            )
        )

    def test_key_encoding_is_exact_little_endian_literal(self) -> None:
        raw = sel.encode_instance_key(
            0x0102030405060708, 0x11223344, 0x8899AABBCCDDEEFF
        )
        self.assertEqual(
            raw.hex(),
            "080706050403020144332211ffeeddccbbaa9988",
        )
        for values in (
            (0, 1, 1),
            (1, 0, 1),
            (1, 0xFFFFFFFF, 1),
            (1, 1, 0),
            (True, 1, 1),
            (1, 1.0, 1),
        ):
            with self.assertRaises(sel.SelectionRuleError):
                sel.encode_instance_key(*values)

    def test_mapping_and_decision_literals(self) -> None:
        vectors = (
            (1, 1, 1, 0xF881B6F2EEF5F925),
            (1, 1, 2, 0x54936737C945F86F),
            (7, 31, 99, 0x6D0FFB5472684179),
            (
                0x0102030405060708,
                0x11223344,
                0x8899AABBCCDDEEFF,
                0x89A052BD3F9A4F4A,
            ),
        )
        for generation, thread, sequence, expected in vectors:
            self.assertEqual(sel.mapping64(generation, thread, sequence), expected)
            self.assertEqual(
                self._definition_only_mapping(
                    sel.DEFINITION["entries"][0], generation, thread, sequence
                ),
                expected,
            )
        key = vectors[0][:3]
        seed = 0x0123456789ABCDEF
        expected_u = vectors[0][3] ^ seed
        self.assertEqual(sel.selection_value(*key, seed), expected_u)
        self.assertFalse(sel.selected(*key, seed_k=seed, threshold=expected_u))
        self.assertTrue(sel.selected(*key, seed_k=seed, threshold=expected_u + 1))

    def test_seed_hex_is_exact_numeric_big_endian_and_provenance_typed(self) -> None:
        self.assertEqual(
            sel.parse_seed_hex(
                "0001020304050607", 1, 1, require_publication_seed=True
            ),
            0x0001020304050607,
        )
        self.assertEqual(
            sel.parse_seed_hex(
                "0000000000000000", 1, 1, require_publication_seed=True
            ),
            0,
        )
        self.assertEqual(
            sel.parse_seed_hex(
                "0123456789abcdef", 2, 2, require_publication_seed=False
            ),
            0x0123456789ABCDEF,
        )
        for seed_hex, provenance, status, publication in (
            ("0123456789abcdef", 2, 2, True),
            ("0123456789abcdeF", 1, 1, False),
            ("00", 1, 1, False),
            (None, 1, 1, False),
            ("0000000000000000", True, 1, False),
            ("0000000000000000", 1, True, False),
            (None, 20, 20, False),
            ("0000000000000000", 20, 20, False),
            ("0000000000000000", 1, 2, False),
        ):
            with self.assertRaises(sel.SelectionRuleError):
                sel.parse_seed_hex(
                    seed_hex, provenance, status, require_publication_seed=publication
                )
        for publication in (0, 1, None, "yes"):
            with self.assertRaises(sel.SelectionRuleError):
                sel.parse_seed_hex(
                    "0000000000000000", 1, 1,
                    require_publication_seed=publication,
                )

    def test_threshold_boundaries_and_inactive_canonical_zero(self) -> None:
        self.assertEqual(sel.validate_threshold(1, active_a03=True), 1)
        self.assertEqual(
            sel.validate_threshold((1 << 64) - 1, active_a03=True),
            (1 << 64) - 1,
        )
        self.assertEqual(sel.validate_threshold(0, active_a03=False), 0)
        for threshold, active in (
            (0, True),
            (1, False),
            (1 << 64, True),
            (True, True),
            (1, 1),
        ):
            with self.assertRaises(sel.SelectionRuleError):
                sel.validate_threshold(threshold, active_a03=active)

    def test_definition_is_canonical_typed_and_hash_bound(self) -> None:
        raw = sel.definition_bytes()
        literal = (
            MODULE_PATH.parent / "archive" / sel.DEFINITION_PATH
        ).read_bytes()
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(raw, sel.SEMANTIC.canonical_bytes(sel.DEFINITION))
        self.assertEqual(literal, raw)
        self.assertEqual(len(raw), 1689)
        self.assertEqual(
            sel.definition_sha256(),
            "6168e7d23ae1a514cca8b111bd0a99b0a6b7a903c59fbc00291ca949ce1110c8",
        )
        self.assertEqual(sel.SEMANTIC.parse_canonical(raw), sel.DEFINITION)
        self.assertEqual(
            sel.BUNDLE_MEMBER_PROPOSAL,
            {
                "name": "selection_rule_definition",
                "object_type": 12,
                "owner": "sampling",
                "path": "definition/selection-rule-v1.json",
            },
        )
        self.assertEqual(
            sel.BUNDLE_ENTRY_PROPOSAL,
            {
                "bytes": 1689,
                "object_type": 12,
                "path": "definition/selection-rule-v1.json",
                "sha256": "6168e7d23ae1a514cca8b111bd0a99b0a6b7a903c59fbc00291ca949ce1110c8",
            },
        )
        entry = sel.DEFINITION["entries"][0]
        self.assertEqual(len(entry["mapping_operations"]), 4)
        self.assertEqual(len(entry["mix64_operations"]), 5)
        self.assertEqual(
            entry["rotate_left_64"],
            "((x<<n)|(x>>(64-n))) mod 2^64 for 0<n<64",
        )
        self.assertTrue(entry["key_interpretation"]["encoded_key_bytes_are_not_hashed"])
        changed = copy.deepcopy(sel.DEFINITION)
        changed["entries"][0]["rotations"]["thread"] = 22
        with self.assertRaises(sel.SelectionRuleError):
            sel.validate_definition(changed)
        changed = copy.deepcopy(sel.DEFINITION)
        changed["version"]["major"] = True
        with self.assertRaises(sel.SelectionRuleError):
            sel.validate_definition(changed)

    def test_owner_literal_rejects_byte_and_semantic_substitution(self) -> None:
        raw = sel.definition_bytes()
        with self.assertRaises(sel.SEMANTIC.SemanticError):
            sel.SEMANTIC.parse_canonical(raw[:-1] + b" \n")
        changed = copy.deepcopy(sel.DEFINITION)
        changed["entries"][0]["mapping_name"] = "substituted"
        with self.assertRaises(sel.SelectionRuleError):
            sel.validate_definition(changed)


if __name__ == "__main__":
    unittest.main()
