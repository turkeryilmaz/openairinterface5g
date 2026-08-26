#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import semantic_catalog_v1 as subject


EXPECTED = {
    1: (1359, "definition/canonical-json-v1.json", "af27d4cae8233c0e049a014e30e3844d86e3d74c3a59143b89794f655ec2b88b"),
    2: (5327, "definition/schema-bundle-schema-v1.json", "4b1560613f7ed69d3bc0c6f9a9d8067091354cccf03c71e626e870b3549b69fa"),
    3: (33626, "definition/event-semantics-v1.json", "8dbe428939592cdfc86ba8730078563672de4a13081ebdb868fa97f543dfab89"),
    4: (1855, "catalog/api.json", "93056c4cfd071c1df396ba09bf82b4cbe807923977c4bca988b0aee1b8c94610"),
    5: (1064, "definition/callsite-rule-v1.json", "510c852c65888dcf563d10e9e416ad0ab96c8503af1e412e06ff75dcb14caa18"),
    6: (932, "definition/context-schema-v1.json", "a05ee0bb16451fa11c965b8db50bdba3de080473329b44a5f8ffdfb8110c9333"),
    7: (646, "definition/phase-v1.json", "5bc30647a2512ab89e1d1507d3175068586bcb2ae020c19fab7f90696f3c1c1f"),
    8: (9039, "definition/diagnostic-v1.json", "f46d99a638da45105006fa5bdd70547674aa948fd1c35012c68d5dce2a274162"),
}

OWNED_MEMBER_EXPECTED = {
    9: (11942, "definition/coverage-policy-v1.json", "2d01a2e3f066787378e7bebfe50f618d94dd46e4f38ede300adc0ad178f31969"),
    10: (11504, "definition/coverage-instance-schema-v1.json", "86176978f48ef9e888bfb373de713a9729e47e0bce4d9cc2fa8c917fc13e6773"),
}
CONFIG_MEMBER = (
    5588,
    "definition/effective-config-schema-v1.json",
    "592bbf3d41790752140f567213ab6cf688c0d571fc89b9dda132875302b8f9cb",
)
SAMPLING_MEMBER = (
    1689,
    "definition/selection-rule-v1.json",
    "6168e7d23ae1a514cca8b111bd0a99b0a6b7a903c59fbc00291ca949ce1110c8",
)
RUNTIME_MEMBER = (
    5916,
    "definition/runtime-catalog-schema-v1.json",
    "70626b468c7ebb89c8a053103957e400418221fcdf38395444a967d0dea917a1",
)
INVENTORY_EXPECTED = {
    **EXPECTED,
    **OWNED_MEMBER_EXPECTED,
    11: CONFIG_MEMBER,
    12: SAMPLING_MEMBER,
    13: RUNTIME_MEMBER,
}

HISTORICAL_MEMBER2_V11 = (4795, "a1424128a826e59f4048439824809d7233fc7fb9ee68f5e92db51ac1a1a028d2")
HISTORICAL_MEMBER2_V12 = (4795, "5037e2b00c14a3fbb6d4874a3141def2f25bc2d35e59faab63314b4048c225da")
HISTORICAL_MEMBER2_V13 = (5327, "123e076049fa99ebe1e65538b0e63d205d5d204954473cc00d6dfeb896ed4599")
HISTORICAL_MEMBER9 = (7802, "fd6ea8030d797f914f8e0afbfc64acd71830e7561212c6e7738a99ef6ed1e4ec")
HISTORICAL_MEMBER9_V11 = (7846, "a3560d89f837469f3b25a4e3ffd5e3bb239c63fea34889046deaacfaceb6fb7c")
HISTORICAL_BUNDLE_V11 = (2037, "f00405e045aff35cb8e53406f585d3d72a36e32bbdc8e08286d1e8a127cc860f")
HISTORICAL_BUNDLE_V12 = (2037, "edacc504f55f13dc082dc5b1ddead6bcd90b4485f4b93f28f2bdf254aae7c866")
HISTORICAL_BUNDLE_V13 = (2196, "df8088e751f77b871db7f0a2f2e1ab53ca594bd7ff4882caceebd363b2b122f7")
HISTORICAL_BUNDLE_V14 = (2196, "5c4fa1ec85ccbb850c31f5973f2416f98fc0bd8cb7383673e5899c7a4e533f09")
HISTORICAL_BUNDLE_V14_ENTRIES = {
    2: (5327, "396b80e1be2ae765afa29346c57f88a279598fd1fd4b63f4870c0255711c1eed"),
    3: (14457, "f7a258d830eb7284ddf6d3feae65f2c7da25566104e3a254857778f7b342f6d8"),
    4: (737, "c315630f1a0cd4023cb693436dd7e5c72bd2684b46433a91e7a348adf6d525cb"),
    9: (9334, "851cf5899affe8420bb10da47998bf33304cea2f4deb5742ec54859cce9a9ae7"),
    11: (5588, "28f5b3e2a4cc4475a61919e4ca32b59521ea17a5a1bc2400902159ecfe8382a3"),
}
RUNTIME_RELATION_IDS = {
    "clock_catalog_schema",
    "module_catalog_schema",
    "thread_catalog_schema",
}


def canonical(value):
    return subject.canonical_bytes(value)


def historical_member2(raw, minor):
    value = subject.parse_canonical(raw)
    value["member_registry"] = [
        row for row in value["member_registry"] if row["object_type"] != 13
    ]
    value["cross_hash_relations"] = [
        row
        for row in value["cross_hash_relations"]
        if row["relation_id"] not in RUNTIME_RELATION_IDS
    ]
    value["version"]["minor"] = minor
    return canonical(value)


def historical_bundle(raw, *, member2, minor, member9):
    value = subject.parse_canonical(raw)
    value["entries"] = [
        row for row in value["entries"] if row["object_type"] != 13
    ]
    value["version"]["minor"] = minor
    value["schema"]["sha256"] = member2[1]
    value["entries"][1]["bytes"] = member2[0]
    value["entries"][1]["sha256"] = member2[1]
    value["entries"][8]["bytes"] = member9[0]
    value["entries"][8]["sha256"] = member9[1]
    return canonical(value)


class CanonicalJsonTests(unittest.TestCase):
    def test_minimal_literal_and_hash_domain(self):
        raw = b'{"a":1,"b":"x"}\n'
        self.assertEqual(subject.parse_canonical(raw), {"a": 1, "b": "x"})
        self.assertEqual(subject.sha256_hex(raw), hashlib.sha256(raw).hexdigest())
        self.assertNotEqual(subject.sha256_hex(raw), hashlib.sha256(raw[:-1]).hexdigest())

    def test_rejects_noncanonical_encodings(self):
        invalid = (
            b'\xef\xbb\xbf{"a":1}\n',
            b'{"a":1}',
            b'{"a":1}\n\n',
            b'{ "a":1}\n',
            b'{"b":1,"a":2}\n',
            b'{"a":1,"a":2}\n',
            b'{"a":1.0}\n',
            b'{"a":1e0}\n',
            b'{"a":-0}\n',
            '{"a":"e\u0301"}\n'.encode("utf-8"),
            b'{"Bad":1}\n',
            b'{"a":"\\n"}\n',
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(subject.SemanticError):
                subject.parse_canonical(raw)

    def test_parser_rejects_nonbytes_and_oversize_before_json(self):
        for raw in ('{"a":1}\n', bytearray(b'{"a":1}\n')):
            with self.subTest(raw=raw), self.assertRaisesRegex(
                subject.SemanticError, "raw: bytes required"
            ):
                subject.parse_canonical(raw)
        with mock.patch.object(subject, "CANONICAL_RAW_MAX_BYTES", 7), mock.patch.object(
            subject.json, "loads"
        ) as loads:
            with self.assertRaisesRegex(subject.SemanticError, "raw: exceeds 7 byte limit"):
                subject.parse_canonical(b'{"a":1}\n')
        loads.assert_not_called()

    @staticmethod
    def _nested_canonical_raw(depth: int) -> bytes:
        opening = [b'{"a":']
        closing = [b"}"]
        for index in range(1, depth):
            if index % 2:
                opening.append(b"[")
                closing.append(b"]")
            else:
                opening.append(b'{"a":')
                closing.append(b"}")
        return b"".join(opening) + b'"[{}]\\"\\\\"' + b"".join(reversed(closing)) + b"\n"

    def test_parser_nesting_limit_is_string_escape_aware(self):
        at_limit = self._nested_canonical_raw(subject.CANONICAL_JSON_MAX_NESTING_DEPTH)
        self.assertEqual(subject.canonical_bytes(subject.parse_canonical(at_limit)), at_limit)
        with mock.patch.object(subject.json, "loads") as loads:
            with self.assertRaisesRegex(subject.SemanticError, "nesting depth exceeds"):
                subject.parse_canonical(
                    self._nested_canonical_raw(subject.CANONICAL_JSON_MAX_NESTING_DEPTH + 1)
                )
        loads.assert_not_called()

    def test_recursion_errors_are_semantic_errors(self):
        with mock.patch.object(subject, "_walk", side_effect=RecursionError("test")):
            with self.assertRaises(subject.SemanticError) as parsed:
                subject.parse_canonical(b'{"a":1}\n')
        self.assertIsInstance(parsed.exception.__cause__, RecursionError)
        with mock.patch.object(subject, "_canonical_value", side_effect=RecursionError("test")):
            with self.assertRaises(subject.SemanticError) as serialized:
                subject.canonical_bytes({"a": 1})
        self.assertIsInstance(serialized.exception.__cause__, RecursionError)

    def test_serialization_uses_frozen_bytewise_key_order(self):
        raw = b'{"a":3,"aa":2,"b":1}\n'
        self.assertEqual(canonical({"b": 1, "aa": 2, "a": 3}), raw)
        self.assertEqual(subject.parse_canonical(raw), {"a": 3, "aa": 2, "b": 1})

    def test_control_characters_use_lowercase_u00xx(self):
        value = {"a": "tab\tdelete\x7fquote\"slash\\"}
        self.assertEqual(
            canonical(value),
            b'{"a":"tab\\u0009delete\\u007fquote\\"slash\\\\"}\n',
        )
        self.assertEqual(subject.parse_canonical(canonical(value)), value)

    def test_generator_rejects_invalid_member_names_recursively(self):
        mutants = (
            {"Upper": 1},
            {"_leading": 1},
            {"a": {"Upper": 1}},
            {"a": [{"_leading": 1}]},
        )
        for mutant in mutants:
            with self.subTest(mutant=mutant), self.assertRaises(subject.SemanticError):
                canonical(mutant)

    def test_path_domains_are_distinct_and_fail_closed(self):
        for path in ("a/b", "catalog/api.json", "na\u00efve/file"):
            self.assertEqual(subject._relative_path(path, "path"), path)
        for path in ("", "/a", "a/", "a//b", "a/./b", "a/../b", "a\\b", "a\x01b"):
            with self.subTest(path=path), self.assertRaises(subject.SemanticError):
                subject._relative_path(path, "path")
        self.assertEqual(subject._absolute_path("/opt/oai/bin", "path"), "/opt/oai/bin")
        for path in ("/", "opt/oai", "/opt//oai", "/opt/../oai", "/opt/oai/"):
            with self.subTest(path=path), self.assertRaises(subject.SemanticError):
                subject._absolute_path(path, "path")


class StaticLiteralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loaded = subject.validate_semantic_root(ROOT)
        cls.event = subject.parse_canonical(cls.loaded[3][1])

    def test_all_semantic_owned_members_have_frozen_bytes_and_hashes(self):
        self.assertEqual(set(self.loaded), set(range(1, 9)))
        for object_type, (expected_bytes, expected_path, expected_hash) in EXPECTED.items():
            path, raw = self.loaded[object_type]
            with self.subTest(object_type=object_type):
                self.assertEqual(path.relative_to(ROOT / "archive").as_posix(), expected_path)
                self.assertEqual(len(raw), expected_bytes)
                self.assertEqual(subject.sha256_hex(raw), expected_hash)

    def test_hash_inventory_is_canonical_and_independent(self):
        inventory_path = ROOT / "literals" / "expected-static-hashes-v1.json"
        inventory = subject.parse_canonical(inventory_path.read_bytes())
        expected_rows = [
            {"bytes": size, "object_type": object_type, "path": path, "sha256": digest}
            for object_type, (size, path, digest) in INVENTORY_EXPECTED.items()
        ]
        self.assertEqual(inventory["entries"], expected_rows)
        self.assertEqual(
            inventory["scope"],
            "static_bundle_members_1_through_13_runtime_schemas_registered_not_producer_evidence",
        )
        self.assertEqual(inventory["version"], {"major": 1, "minor": 5})

    def test_cross_hash_relation_tuples_and_mutants(self):
        definition = subject.parse_canonical(self.loaded[2][1])
        expected = [
            ("api_bundle_member", "opening.api_catalog_definition_sha256", "schema_bundle.entries[object_type=4].sha256"),
            ("api_object_binding", "opening.api_catalog_definition_sha256", "object_bindings[object_kind=2].sha256"),
            ("build_coverage_api_definition", "object_bindings[object_kind=8].api_definition", "schema_bundle.entries[object_type=4].sha256"),
            ("build_coverage_policy", "object_bindings[object_kind=8].policy", "schema_bundle.entries[object_type=9].sha256"),
            ("build_coverage_schema", "object_bindings[object_kind=8].schema", "schema_bundle.entries[object_type=10].sha256"),
            ("callsite_header", "opening.callsite_catalog_definition_sha256", "schema_bundle.entries[object_type=5].sha256"),
            ("clock_catalog_schema", "catalog/clock.json.schema.sha256", "schema_bundle.entries[object_type=13].sha256"),
            ("context_schema", "catalog/context.json.schema.sha256", "schema_bundle.entries[object_type=6].sha256"),
            ("effective_configuration", "opening.configuration_instance_sha256", "object_bindings[object_kind=10].sha256"),
            ("effective_configuration_schema", "metadata/effective-config.json.schema.sha256", "schema_bundle.entries[object_type=11].sha256"),
            ("module_catalog_schema", "catalog/module.json.schema.sha256", "schema_bundle.entries[object_type=13].sha256"),
            ("phase_context", "catalog/context.json.entries[*].phase_id", "schema_bundle.entries[object_type=7].defined_phase_id"),
            ("run_coverage_build", "object_bindings[object_kind=9].build_coverage", "object_bindings[object_kind=8].sha256"),
            ("run_coverage_configuration", "object_bindings[object_kind=9].configuration_instance_sha256", "opening.configuration_instance_sha256"),
            ("run_coverage_policy", "object_bindings[object_kind=9].policy", "schema_bundle.entries[object_type=9].sha256"),
            ("run_coverage_schema", "object_bindings[object_kind=9].schema", "schema_bundle.entries[object_type=10].sha256"),
            ("schema_bundle_object", "opening.schema_bundle_definition_sha256", "object_bindings[object_kind=1].sha256"),
            ("schema_bundle_schema", "catalog/schema-bundle.json.schema.sha256", "schema_bundle.entries[object_type=2].sha256"),
            ("status_diagnostic_schema", "status/diagnostics.json.schema.sha256", "schema_bundle.entries[object_type=8].sha256"),
            ("thread_catalog_schema", "catalog/thread.json.schema.sha256", "schema_bundle.entries[object_type=13].sha256"),
        ]
        observed = [
            (row["relation_id"], row["left"], row["right"])
            for row in definition["cross_hash_relations"]
        ]
        self.assertEqual(observed, expected)
        self.assertFalse(
            any(
                "build_configuration_sha256" in endpoint
                for row in definition["cross_hash_relations"]
                for endpoint in (row["left"], row["right"])
            )
        )
        for index, field, replacement in (
            (2, "relation_id", "build_coverage_api"),
            (2, "left", "object_bindings[object_kind=8].api"),
            (8, "relation_id", "effective_config_schema"),
            (8, "left", "metadata/effective-config.json.bytes"),
            (10, "right", "object_bindings[object_kind=8].bytes"),
        ):
            mutant = copy.deepcopy(definition)
            mutant["cross_hash_relations"][index][field] = replacement
            with self.subTest(index=index, field=field), self.assertRaisesRegex(
                subject.SemanticError, "exact frozen relation tuples"
            ):
                subject.validate_schema_bundle_schema(mutant)

    def test_config_member11_proposal_and_literal_are_adopted_exactly(self):
        config = subject._load_config_module()
        self.assertEqual(
            config.BUNDLE_MEMBER_PROPOSAL,
            {
                "name": "effective_config_schema",
                "object_type": 11,
                "owner": "config",
                "path": CONFIG_MEMBER[1],
            },
        )
        self.assertEqual(
            config.BUNDLE_CROSS_RELATION_PROPOSAL,
            {
                "left": "metadata/effective-config.json.schema.sha256",
                "relation_id": "effective_configuration_schema",
                "right": "schema_bundle.entries[object_type=11].sha256",
            },
        )
        path = ROOT.parent / "config" / "archive" / CONFIG_MEMBER[1]
        raw = path.read_bytes()
        self.assertEqual((len(raw), subject.sha256_hex(raw)), (CONFIG_MEMBER[0], CONFIG_MEMBER[2]))
        self.assertEqual(config.BUNDLE_ENTRY_PROPOSAL["sha256"], CONFIG_MEMBER[2])
        subject._validate_config_member(raw)

    def test_sampling_member12_proposal_and_literal_are_adopted_exactly(self):
        sampling = subject._load_sampling_module()
        self.assertEqual(
            sampling.BUNDLE_MEMBER_PROPOSAL,
            {
                "name": "selection_rule_definition",
                "object_type": 12,
                "owner": "sampling",
                "path": SAMPLING_MEMBER[1],
            },
        )
        path = ROOT.parent / "sampling" / "archive" / SAMPLING_MEMBER[1]
        raw = path.read_bytes()
        self.assertEqual(
            (len(raw), subject.sha256_hex(raw)),
            (SAMPLING_MEMBER[0], SAMPLING_MEMBER[2]),
        )
        self.assertEqual(sampling.BUNDLE_ENTRY_PROPOSAL["sha256"], SAMPLING_MEMBER[2])
        subject._validate_sampling_member(raw)

    def test_runtime_member13_proposals_relations_and_literal_are_adopted_exactly(self):
        runtime = subject._load_runtime_module()
        self.assertEqual(runtime.BUNDLE_MEMBER_PROPOSAL, {
            "name": "runtime_catalog_schema",
            "object_type": 13,
            "owner": "runtime",
            "path": RUNTIME_MEMBER[1],
        })
        path = ROOT.parent / "runtime" / "archive" / RUNTIME_MEMBER[1]
        raw = path.read_bytes()
        self.assertEqual((len(raw), subject.sha256_hex(raw)), (RUNTIME_MEMBER[0], RUNTIME_MEMBER[2]))
        self.assertEqual(runtime.BUNDLE_ENTRY_PROPOSAL["sha256"], RUNTIME_MEMBER[2])
        self.assertEqual(
            tuple(row["relation_id"] for row in runtime.BUNDLE_CROSS_RELATION_PROPOSALS),
            ("clock_catalog_schema", "module_catalog_schema", "thread_catalog_schema"),
        )
        subject._validate_runtime_member(raw)

    def test_complete_diagnostic_definition_matches_accepted_design(self):
        proposal = (ROOT / "DIAGNOSTIC_REASON_PROPOSAL.md").read_bytes()
        self.assertEqual(
            hashlib.sha256(proposal).hexdigest(),
            "3e2e2c3add69a04249f7d01ddec72e95f063885f11fdc830db7b1544003f3d8d",
        )
        diagnostic = subject.parse_canonical(self.loaded[8][1])
        self.assertIs(diagnostic["complete_capable"], True)
        self.assertEqual(
            [(row["reason_id"], row["name"], row["class_flags"]) for row in diagnostic["reason_entries"]],
            [
                (1, "ring_full", 0x0101),
                (2, "unregistered_active_thread", 0x0101),
                (3, "registration_capacity_failure", 0x0101),
                (16, "recursion_bypass", 0x0102),
                (17, "profiler_internal_bypass", 0x0004),
                (18, "unsupported_api_or_domain", 0x0004),
                (32, "size_unknown", 0x0008),
                (48, "sample_membership_insertion_failure", 0x0210),
                (49, "membership_lookup_failure", 0x0210),
                (50, "bounded_probe_exhaustion", 0x0210),
                (51, "invalid_or_ambiguous_pointer_pairing", 0x0210),
                (64, "clock_regression_or_invalid_counter", 0x0140),
                (80, "writer_io_or_finalization_failure", 0x0120),
                (96, "diagnostic_counter_saturation", 0x0180),
            ],
        )
        self.assertEqual(
            [(row["scope_kind"], row["name"]) for row in diagnostic["counter_scope_kinds"]],
            [(1, "producer_thread"), (2, "registration"), (3, "writer"), (4, "diagnostic_aggregate")],
        )
        modes = {row["mode_id"]: row for row in diagnostic["mode_scope_arrays"]}
        self.assertEqual(sorted(modes), [2, 3, 4, 5])
        self.assertEqual(modes[2]["not_applicable_reason_ids"], [1, 48, 49, 50, 51, 64])
        self.assertEqual(modes[3]["not_applicable_reason_ids"], [])
        self.assertEqual(modes[4]["not_applicable_reason_ids"], [48, 49, 50, 51])
        self.assertEqual(modes[5]["producer_thread_reason_ids"], [1, 16, 17, 18, 32, 64])
        for mode in modes.values():
            population = (
                mode["diagnostic_aggregate_reason_ids"]
                + mode["not_applicable_reason_ids"]
                + mode["producer_thread_reason_ids"]
                + mode["registration_reason_ids"]
                + mode["writer_reason_ids"]
            )
            self.assertEqual(sorted(population), [1, 2, 3, 16, 17, 18, 32, 48, 49, 50, 51, 64, 80, 96])
        self.assertEqual(
            [row["name"] for row in diagnostic["counter_row_schema"]["fields"]],
            ["counter_scope_id", "counter_scope_kind", "process_generation", "reason_id", "saturated", "value"],
        )
        self.assertEqual(
            [row["name"] for row in diagnostic["reason_total_schema"]["fields"]],
            ["class_flags", "nonzero_counter_instances", "reason_id", "saturated_counter_instances", "saturating_total", "summary_flags"],
        )
        self.assertEqual(diagnostic["reason_total_schema"]["count"], 14)
        self.assertEqual(diagnostic["projection_rules"]["id96"]["self_saturation"], "forbidden")
        self.assertEqual(
            diagnostic["projection_rules"]["partial"]["summary_flag"],
            "partial_counter_population_set_iff_required_instance_or_value_unavailable",
        )

    def test_diagnostic_definition_rejects_exact_semantic_mutants(self):
        diagnostic = subject.parse_canonical(self.loaded[8][1])
        mutants = []
        wrong_mask = copy.deepcopy(diagnostic)
        wrong_mask["reason_entries"][13]["class_flags"] = 0x0080
        mutants.append(("reason_mask", wrong_mask))
        missing_scope = copy.deepcopy(diagnostic)
        missing_scope["mode_scope_arrays"][1]["producer_thread_reason_ids"].remove(50)
        mutants.append(("mode_scope", missing_scope))
        wrong_projection = copy.deepcopy(diagnostic)
        wrong_projection["projection_rules"]["id96"]["self_saturation"] = "allowed"
        mutants.append(("id96_projection", wrong_projection))
        wrong_type = copy.deepcopy(diagnostic)
        wrong_type["complete_capable"] = 1
        mutants.append(("boolean_type", wrong_type))
        for name, mutant in mutants:
            with self.subTest(name=name), self.assertRaises(subject.SemanticError):
                subject.validate_diagnostic_definition(mutant)

    def test_id_domains_and_valid_pairs(self):
        self.assertEqual(
            [(row["event_kind"], row["name"]) for row in self.event["event_kinds"]],
            [(1, "allocation_result"), (2, "reallocation_result"), (3, "release_call")],
        )
        self.assertEqual(
            [(row["api_id"], row["event_kind"]) for row in self.event["valid_pairs"]],
            [
                (1, 1), (2, 1), (3, 2), (4, 3), (5, 2), (6, 1),
                (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1),
            ],
        )
        api = subject.parse_canonical(self.loaded[4][1])
        self.assertEqual([row["api_id"] for row in api["entries"]], list(range(1, 13)))
        self.assertEqual(
            [row["name"] for row in api["entries"]],
            [
                "malloc", "calloc", "realloc", "free", "reallocarray",
                "aligned_alloc", "posix_memalign", "memalign", "valloc",
                "pvalloc", "strdup", "strndup",
            ],
        )

    def test_flag_domains_are_disjoint_and_exhaustive(self):
        masks = self.event["flag_masks"]
        domains = (
            masks["base_fixed_mask"],
            masks["operand_validity_mask"],
            masks["selection_mask"],
            masks["common_evidence_mask"],
        )
        for index, left in enumerate(domains):
            for right in domains[index + 1 :]:
                self.assertEqual(left & right, 0)
        self.assertEqual(domains[0] | domains[1] | domains[2] | domains[3], masks["known_mask"])
        self.assertEqual(masks["known_mask"] | masks["reserved_mask"], 0xFFFFFFFF)
        self.assertEqual(masks["known_mask"] & masks["reserved_mask"], 0)

    def _variant_masks(self, transition_id, operand_profile, selection_case):
        transition = next(row for row in self.event["transitions"] if row["transition_id"] == transition_id)
        operand = next(row for row in self.event["operand_profiles"] if row["name"] == operand_profile)
        selection = next(row for row in self.event["selection_cases"] if row["name"] == selection_case)
        one = transition["required_one_mask"] | operand["required_one_mask"] | selection["required_one_mask"]
        zero = transition["required_zero_mask"] | operand["required_zero_mask"] | selection["required_zero_mask"]
        return one, zero

    def test_every_declared_variant_composes_exactly(self):
        transitions = {row["transition_id"]: row for row in self.event["transitions"]}
        selections = {row["name"]: row for row in self.event["selection_cases"]}
        for mode in self.event["mode_rules"]:
            for mapping in mode["selection_cases"]:
                for transition_id in mapping["transition_ids"]:
                    transition = transitions[transition_id]
                    for case_name in mapping["selection_cases"]:
                        case = selections[case_name]
                        profile_name = (
                            transition["matched_operand_profile"]
                            if case["arg_match_valid"]
                            else transition["unmatched_operand_profile"]
                        )
                        self.assertIsNotNone(profile_name)
                        one, zero = self._variant_masks(transition_id, profile_name, case_name)
                        with self.subTest(mode=mode["mode_id"], transition=transition_id, case=case_name):
                            self.assertEqual(one & zero, 0)
                            self.assertEqual(one | zero | subject.COMMON_EVIDENCE_MASK, subject.KNOWN_FLAGS_MASK)

    def test_a03_malloc_requires_successor_selected_and_no_predecessor_bits(self):
        one, zero = self._variant_masks(103, "malloc", "successor_only")
        self.assertTrue(one & (1 << 16))
        self.assertEqual(one & ((1 << 14) | (1 << 15) | (1 << 17)), 0)
        self.assertEqual(zero & ((1 << 14) | (1 << 15) | (1 << 17)), (1 << 14) | (1 << 15) | (1 << 17))

    def test_a04_a05_forbid_all_selection_and_match_bits(self):
        for transition_id, profile in ((103, "malloc"), (204, "calloc_valid"), (308, "realloc_unmatched"), (401, "free_exact")):
            one, zero = self._variant_masks(transition_id, profile, "none")
            with self.subTest(transition=transition_id):
                self.assertEqual(one & subject.SELECTION_MASK, 0)
                self.assertEqual(zero & subject.SELECTION_MASK, subject.SELECTION_MASK)

    def test_realloc_zero_policy_truth_table(self):
        policies = {row["policy_id"]: row for row in self.event["realloc_zero_policies"]}
        self.assertFalse(policies[0]["admitted"])
        self.assertEqual((policies[1]["predecessor_ended"], policies[1]["operation_failed"]), (True, False))
        self.assertEqual((policies[2]["predecessor_ended"], policies[2]["operation_failed"]), (False, True))
        transitions = {row["transition_id"]: row for row in self.event["transitions"]}
        self.assertEqual(transitions[300]["realloc_zero_policy_id"], None)
        self.assertEqual(transitions[304]["realloc_zero_policy_id"], 1)
        self.assertEqual(transitions[305]["realloc_zero_policy_id"], 2)
        self.assertEqual(transitions[300]["required_one_mask"] & ((1 << 11) | (1 << 12) | (1 << 13)), 0)

    def test_deferred_c_api_result_and_requested_byte_rules(self):
        flag_bits = {row["bit"]: row["name"] for row in self.event["flag_bits"]}
        self.assertEqual(flag_bits[19], "reallocarray_size_product_overflow")
        self.assertEqual(self.event["flag_masks"]["known_mask"], 0x030FFFFF)
        self.assertEqual(self.event["flag_masks"]["reserved_mask"], 0xFCF00000)
        transitions = {row["transition_id"]: row for row in self.event["transitions"]}
        self.assertEqual(len(transitions), 57)
        self.assertTrue(transitions[500]["required_one_mask"] & (1 << 19))
        self.assertTrue(transitions[501]["required_one_mask"] & (1 << 19))
        self.assertTrue(transitions[502]["required_zero_mask"] & (1 << 19))
        for transition_id, transition in transitions.items():
            with self.subTest(transition=transition_id):
                self.assertEqual(
                    bool(transition["required_one_mask"] & (1 << 25)),
                    700 <= transition_id <= 704,
                )
        result_kinds = {row["name"]: row for row in self.event["result_kinds"]}
        self.assertTrue(result_kinds["errno"]["first_slice_allowed"])
        self.assertTrue(result_kinds["direct_return"]["first_slice_allowed"])
        self.assertFalse(result_kinds["none"]["first_slice_allowed"])
        requested = {row["api_id"]: row for row in self.event["requested_byte_rules"]}
        self.assertEqual(set(requested), set(range(1, 13)))
        self.assertEqual(requested[5]["source"], "arg0")
        self.assertEqual(requested[7]["source"], "arg1")
        for api_id in (11, 12):
            self.assertEqual(
                requested[api_id]["availability"],
                "unavailable_without_second_scan",
            )
            self.assertEqual(requested[api_id]["unknown_reason_id"], 32)

    def test_mode_truth_table(self):
        modes = {row["mode_id"]: row for row in self.event["mode_rules"]}
        for mode_id in (0, 1, 2, 6):
            self.assertEqual(modes[mode_id]["allowed_transition_ids"], [])
        self.assertEqual(
            modes[3]["allowed_transition_ids"],
            [
                101, 103, 202, 204, 301, 303, 304, 306, 308, 401, 402,
                503, 504, 506, 508, 510, 601, 603, 703, 704, 801, 803,
                901, 903, 1001, 1003, 1101, 1201,
            ],
        )
        self.assertEqual(len(modes[4]["allowed_transition_ids"]), 56)
        self.assertEqual(modes[4]["allowed_transition_ids"], modes[5]["allowed_transition_ids"])
        self.assertEqual(modes[4]["callsite_rule"], "canonical_zero")
        self.assertEqual(modes[5]["callsite_rule"], "nonzero_resolved")

    def test_event_flag_helper_rejects_noncomplementary_caller_masks(self):
        for required_one, required_zero in (
            (0, 0),
            (
                subject.KNOWN_FLAGS_MASK & ~subject.COMMON_EVIDENCE_MASK,
                1 << 5,
            ),
        ):
            with self.subTest(
                required_one=required_one, required_zero=required_zero
            ), self.assertRaisesRegex(
                subject.SemanticError, "exact complementary variant masks"
            ):
                subject.validate_event_flags(
                    flags=0x01000000,
                    required_one_mask=required_one,
                    required_zero_mask=required_zero,
                    address_before=0, address_after=0, arg0=0, arg1=0, arg2=0,
                    counter_enter=0, counter_exit=0,
                    cpu_enter=0xFFFF, cpu_exit=0xFFFF, result_code=0,
                )

    def test_invalid_field_sentinels(self):
        one, zero = self._variant_masks(103, "malloc", "none")
        flags = one
        subject.validate_event_flags(
            flags=flags,
            required_one_mask=one,
            required_zero_mask=zero,
            address_before=0,
            address_after=0x1000,
            arg0=64,
            arg1=0,
            arg2=0,
            counter_enter=0,
            counter_exit=0,
            cpu_enter=0xFFFF,
            cpu_exit=0xFFFF,
            result_code=0,
        )
        with self.assertRaisesRegex(subject.SemanticError, "arg1"):
            subject.validate_event_flags(
                flags=flags,
                required_one_mask=one,
                required_zero_mask=zero,
                address_before=0,
                address_after=0x1000,
                arg0=64,
                arg1=1,
                arg2=0,
                counter_enter=0,
                counter_exit=0,
                cpu_enter=0xFFFF,
                cpu_exit=0xFFFF,
                result_code=0,
            )


class ContextAndBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loaded = subject.validate_semantic_root(ROOT)

    def _context(self, entries):
        return {
            "catalog_id": "oai_memprof_context",
            "entries": entries,
            "schema": {
                "object_type": 6,
                "path": "definition/context-schema-v1.json",
                "sha256": subject.sha256_hex(self.loaded[6][1]),
            },
            "version": {"major": 1, "minor": 0},
        }

    def _member_files(self):
        files = {}
        owner_roots = {
            "config": ROOT.parent / "config",
            "coverage": ROOT.parent / "coverage",
            "runtime": ROOT.parent / "runtime",
            "sampling": ROOT.parent / "sampling",
            "semantic": ROOT,
        }
        for object_type, _name, archive_path, owner in subject.MEMBER_REGISTRY:
            files[object_type] = owner_roots[owner] / "archive" / archive_path
        return files

    def test_context_generation_sort_and_phase_resolution(self):
        value = self._context(
            [
                {"context_id": 2, "phase_id": 0, "process_generation": 1},
                {"context_id": 1, "phase_id": 0, "process_generation": 2},
            ]
        )
        subject.validate_context_catalog(
            value,
            context_schema_sha256=subject.sha256_hex(self.loaded[6][1]),
            phase_ids={0},
        )
        bad = copy.deepcopy(value)
        bad["entries"].reverse()
        with self.assertRaisesRegex(subject.SemanticError, "sorted"):
            subject.validate_context_catalog(
                bad,
                context_schema_sha256=subject.sha256_hex(self.loaded[6][1]),
                phase_ids={0},
            )
        bad = self._context([{"context_id": 0, "phase_id": 0, "process_generation": 1}])
        with self.assertRaises(subject.SemanticError):
            subject.validate_context_catalog(bad, context_schema_sha256=subject.sha256_hex(self.loaded[6][1]), phase_ids={0})
        bad = self._context([{"context_id": 1, "phase_id": 7, "process_generation": 1}])
        with self.assertRaisesRegex(subject.SemanticError, "unresolved phase"):
            subject.validate_context_catalog(bad, context_schema_sha256=subject.sha256_hex(self.loaded[6][1]), phase_ids={0})

    def test_complete_bundle_generation_and_binding_with_all_owned_members(self):
        files = self._member_files()
        raw = subject.build_bundle(files)
        member_bytes = {object_type: path.read_bytes() for object_type, path in files.items()}
        bundle = subject.validate_bundle(raw, member_bytes)
        self.assertEqual([row["object_type"] for row in bundle["entries"]], list(range(1, 14)))
        self.assertEqual(bundle["entries"][3]["sha256"], EXPECTED[4][2])
        self.assertEqual(bundle["schema"]["sha256"], EXPECTED[2][2])
        self.assertEqual(subject.VERSION, {"major": 1, "minor": 0})
        self.assertEqual(subject.BUNDLE_VERSION, {"major": 1, "minor": 5})
        self.assertEqual(bundle["version"], subject.BUNDLE_VERSION)
        self.assertEqual(subject.parse_canonical(member_bytes[2])["version"], subject.BUNDLE_VERSION)
        coverage = subject._load_coverage_module()
        self.assertEqual(coverage.POLICY_VERSION, {"major": 2, "minor": 0})
        self.assertEqual(coverage.INSTANCE_VERSION, {"major": 1, "minor": 0})
        self.assertEqual(subject.parse_canonical(member_bytes[9])["version"], coverage.POLICY_VERSION)
        self.assertEqual(subject.parse_canonical(member_bytes[10])["version"], coverage.INSTANCE_VERSION)
        for object_type, (size, path, digest) in INVENTORY_EXPECTED.items():
            entry = bundle["entries"][object_type - 1]
            with self.subTest(object_type=object_type):
                self.assertEqual(
                    entry,
                    {
                        "bytes": size,
                        "object_type": object_type,
                        "path": path,
                        "sha256": digest,
                    },
                )
        missing = dict(member_bytes)
        missing.pop(13)
        with self.assertRaisesRegex(subject.SemanticError, "exact object_type set 1..13"):
            subject.validate_bundle(raw, missing)
        mutant = bytearray(raw)
        marker = EXPECTED[4][2].encode("ascii")
        offset = mutant.index(marker)
        mutant[offset] = ord("0") if mutant[offset] != ord("0") else ord("1")
        with self.assertRaises(subject.SemanticError):
            subject.validate_bundle(bytes(mutant), member_bytes)

    def test_validate_bundle_rejects_self_consistent_event_semantics_replacement(self):
        files = self._member_files()
        valid_bundle = subject.parse_canonical(subject.build_bundle(files))
        valid_member_bytes = {
            object_type: path.read_bytes() for object_type, path in files.items()
        }
        replacement_value = subject.parse_canonical(valid_member_bytes[3])
        replacement_value["event_kinds"][0]["name"] = "allocation"
        replacement_raw = canonical(replacement_value)
        replacement_bundle = copy.deepcopy(valid_bundle)
        replacement_entry = replacement_bundle["entries"][2]
        replacement_entry["bytes"] = len(replacement_raw)
        replacement_entry["sha256"] = subject.sha256_hex(replacement_raw)
        replacement_members = dict(valid_member_bytes)
        replacement_members[3] = replacement_raw

        self.assertEqual(
            (replacement_entry["bytes"], replacement_entry["sha256"]),
            (len(replacement_raw), subject.sha256_hex(replacement_raw)),
        )
        with self.assertRaisesRegex(
            subject.SemanticError, "event_semantics.event_kinds: frozen IDs differ"
        ):
            subject.validate_bundle(canonical(replacement_bundle), replacement_members)

    def test_validate_bundle_rejects_self_consistent_api_cross_binding_replacement(self):
        files = self._member_files()
        valid_bundle = subject.parse_canonical(subject.build_bundle(files))
        valid_member_bytes = {
            object_type: path.read_bytes() for object_type, path in files.items()
        }
        replacement_value = subject.parse_canonical(valid_member_bytes[4])
        replacement_value["schema"]["sha256"] = "0" * 64
        replacement_raw = canonical(replacement_value)
        replacement_bundle = copy.deepcopy(valid_bundle)
        replacement_entry = replacement_bundle["entries"][3]
        replacement_entry["bytes"] = len(replacement_raw)
        replacement_entry["sha256"] = subject.sha256_hex(replacement_raw)
        replacement_members = dict(valid_member_bytes)
        replacement_members[4] = replacement_raw

        self.assertEqual(
            (replacement_entry["bytes"], replacement_entry["sha256"]),
            (len(replacement_raw), subject.sha256_hex(replacement_raw)),
        )
        with self.assertRaisesRegex(
            subject.SemanticError,
            "api.schema.sha256: does not bind exact event-semantics bytes",
        ):
            subject.validate_bundle(canonical(replacement_bundle), replacement_members)

    def test_historical_bundle_v14_is_reconstructible_but_rejected(self):
        files = self._member_files()
        current_raw = subject.build_bundle(files)
        current_members = {
            object_type: path.read_bytes() for object_type, path in files.items()
        }
        historical = subject.parse_canonical(current_raw)
        historical["version"] = {"major": 1, "minor": 4}
        historical["schema"]["sha256"] = HISTORICAL_BUNDLE_V14_ENTRIES[2][1]
        for object_type, (byte_count, digest) in HISTORICAL_BUNDLE_V14_ENTRIES.items():
            entry = historical["entries"][object_type - 1]
            entry["bytes"] = byte_count
            entry["sha256"] = digest
        historical_raw = canonical(historical)
        self.assertEqual(
            (len(historical_raw), subject.sha256_hex(historical_raw)),
            HISTORICAL_BUNDLE_V14,
        )
        with self.assertRaises(subject.SemanticError):
            subject.validate_bundle(historical_raw, current_members)

    def test_validate_bundle_rejects_paired_stale_member2_and_wrong_member9_size(self):
        files = self._member_files()
        valid_raw = subject.build_bundle(files)
        valid_bundle = subject.parse_canonical(valid_raw)
        valid_members = {
            object_type: path.read_bytes() for object_type, path in files.items()
        }

        historical_schema = subject.parse_canonical(valid_members[2])
        historical_schema["version"]["minor"] = 3
        historical_schema_raw = canonical(historical_schema)
        self.assertEqual(
            (len(historical_schema_raw), subject.sha256_hex(historical_schema_raw)),
            HISTORICAL_MEMBER2_V13,
        )
        stale_bundle = copy.deepcopy(valid_bundle)
        stale_bundle["schema"]["sha256"] = HISTORICAL_MEMBER2_V13[1]
        stale_bundle["entries"][1]["bytes"] = HISTORICAL_MEMBER2_V13[0]
        stale_bundle["entries"][1]["sha256"] = HISTORICAL_MEMBER2_V13[1]
        stale_members = dict(valid_members)
        stale_members[2] = historical_schema_raw
        with self.assertRaisesRegex(subject.SemanticError, "version 1.5"):
            subject.validate_bundle(canonical(stale_bundle), stale_members)

        wrong_size = copy.deepcopy(valid_bundle)
        wrong_size["entries"][8]["bytes"] = HISTORICAL_BUNDLE_V14_ENTRIES[9][0]
        with self.assertRaisesRegex(subject.SemanticError, "byte count mismatch"):
            subject.validate_bundle(canonical(wrong_size), valid_members)

    def test_build_and_validation_reject_canonical_bogus_coverage_members(self):
        files = self._member_files()
        valid_raw = subject.build_bundle(files)
        valid_bundle = subject.parse_canonical(valid_raw)
        valid_member_bytes = {
            object_type: path.read_bytes() for object_type, path in files.items()
        }
        with tempfile.TemporaryDirectory(prefix="oai-memprof-semantic-") as temporary:
            temporary_root = Path(temporary)
            for object_type in (9, 10):
                raw = valid_member_bytes[object_type]
                expected = OWNED_MEMBER_EXPECTED[object_type]
                self.assertEqual((len(raw), subject.sha256_hex(raw)), (expected[0], expected[2]))
                subject._validate_coverage_member(object_type, raw)
                bogus = canonical(
                    {
                        "definition_id": f"bogus_member_{object_type}",
                        "version": {"major": 2 if object_type == 9 else 1, "minor": 0},
                    }
                )
                bogus_path = temporary_root / f"member-{object_type}.json"
                bogus_path.write_bytes(bogus)
                build_files = dict(files)
                build_files[object_type] = bogus_path
                with self.subTest(operation="build", object_type=object_type), self.assertRaisesRegex(
                    subject.SemanticError, "exact coverage literal validation failed"
                ):
                    subject.build_bundle(build_files)

                bogus_bundle = copy.deepcopy(valid_bundle)
                entry = bogus_bundle["entries"][object_type - 1]
                entry["bytes"] = len(bogus)
                entry["sha256"] = subject.sha256_hex(bogus)
                member_bytes = dict(valid_member_bytes)
                member_bytes[object_type] = bogus
                with self.subTest(operation="validate", object_type=object_type), self.assertRaisesRegex(
                    subject.SemanticError, "exact coverage literal validation failed"
                ):
                    subject.validate_bundle(canonical(bogus_bundle), member_bytes)

    def test_build_and_validation_reject_canonical_bogus_config_member(self):
        files = self._member_files()
        valid_raw = subject.build_bundle(files)
        valid_bundle = subject.parse_canonical(valid_raw)
        valid_member_bytes = {
            object_type: path.read_bytes() for object_type, path in files.items()
        }
        bogus = canonical(
            {
                "definition_id": "bogus_effective_config_schema",
                "version": {"major": 1, "minor": 1},
            }
        )
        with tempfile.TemporaryDirectory(prefix="oai-memprof-semantic-") as temporary:
            bogus_path = Path(temporary) / "member-11.json"
            bogus_path.write_bytes(bogus)
            build_files = dict(files)
            build_files[11] = bogus_path
            with self.assertRaisesRegex(
                subject.SemanticError, "config entry proposal differs"
            ):
                subject.build_bundle(build_files)

        bogus_bundle = copy.deepcopy(valid_bundle)
        bogus_bundle["entries"][10]["bytes"] = len(bogus)
        bogus_bundle["entries"][10]["sha256"] = subject.sha256_hex(bogus)
        member_bytes = dict(valid_member_bytes)
        member_bytes[11] = bogus
        with self.assertRaisesRegex(
            subject.SemanticError, "config entry proposal differs"
        ):
            subject.validate_bundle(canonical(bogus_bundle), member_bytes)


    def test_build_and_validation_reject_canonical_bogus_sampling_member(self):
        files = self._member_files()
        valid_raw = subject.build_bundle(files)
        valid_bundle = subject.parse_canonical(valid_raw)
        valid_member_bytes = {
            object_type: path.read_bytes() for object_type, path in files.items()
        }
        bogus_value = subject.parse_canonical(valid_member_bytes[12])
        bogus_value["entries"][0]["rotations"]["thread"] = 22
        bogus = canonical(bogus_value)
        with tempfile.TemporaryDirectory(prefix="oai-memprof-semantic-") as temporary:
            bogus_path = Path(temporary) / "member-12.json"
            bogus_path.write_bytes(bogus)
            build_files = dict(files)
            build_files[12] = bogus_path
            with self.assertRaisesRegex(
                subject.SemanticError, "sampling entry proposal differs"
            ):
                subject.build_bundle(build_files)

        bogus_bundle = copy.deepcopy(valid_bundle)
        bogus_bundle["entries"][11]["bytes"] = len(bogus)
        bogus_bundle["entries"][11]["sha256"] = subject.sha256_hex(bogus)
        member_bytes = dict(valid_member_bytes)
        member_bytes[12] = bogus
        with self.assertRaisesRegex(
            subject.SemanticError, "sampling entry proposal differs"
        ):
            subject.validate_bundle(canonical(bogus_bundle), member_bytes)

        byte_substitution = valid_member_bytes[12][:-1] + b" \n"
        bad_bytes = dict(valid_member_bytes)
        bad_bytes[12] = byte_substitution
        with self.assertRaises(subject.SemanticError):
            subject.validate_bundle(valid_raw, bad_bytes)

    def test_build_and_validation_reject_canonical_bogus_runtime_member(self):
        files = self._member_files()
        valid_raw = subject.build_bundle(files)
        valid_bundle = subject.parse_canonical(valid_raw)
        valid_member_bytes = {object_type: path.read_bytes() for object_type, path in files.items()}
        bogus_value = subject.parse_canonical(valid_member_bytes[13])
        bogus_value["clock_acquisition_sources"][0]["source_id"] = 9
        bogus = canonical(bogus_value)
        with tempfile.TemporaryDirectory(prefix="oai-memprof-semantic-") as temporary:
            bogus_path = Path(temporary) / "member-13.json"
            bogus_path.write_bytes(bogus)
            build_files = dict(files)
            build_files[13] = bogus_path
            with self.assertRaisesRegex(subject.SemanticError, "runtime entry proposal differs"):
                subject.build_bundle(build_files)
        bogus_bundle = copy.deepcopy(valid_bundle)
        bogus_bundle["entries"][12]["bytes"] = len(bogus)
        bogus_bundle["entries"][12]["sha256"] = subject.sha256_hex(bogus)
        member_bytes = dict(valid_member_bytes)
        member_bytes[13] = bogus
        with self.assertRaisesRegex(subject.SemanticError, "runtime entry proposal differs"):
            subject.validate_bundle(canonical(bogus_bundle), member_bytes)

if __name__ == "__main__":
    unittest.main()
