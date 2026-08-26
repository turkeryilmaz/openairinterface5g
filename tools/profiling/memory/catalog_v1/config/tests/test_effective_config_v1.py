#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "effective_config_v1.py"
SPEC = importlib.util.spec_from_file_location("effective_config_v1", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
cfg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cfg)


def fixture(
    *,
    mode_id: int = 3,
    seed: str | None = "0001020304050607",
    provenance_id: int = 2,
    status_id: int = 2,
    threshold: int = 1 << 60,
    run_id: str = "synthetic-run-0001",
):
    return cfg.make_effective_configuration(
        flush_records=256,
        flush_us=1000,
        max_threads=64,
        mode_id=mode_id,
        output_directory="/tmp/oai-memprof",
        ring_records=4096,
        role_id=1,
        run_id=run_id,
        sample_seed_hex=seed,
        sample_seed_provenance_id=provenance_id,
        sample_seed_status_id=status_id,
        sample_threshold=threshold,
        scope_kind=1,
        selection_values=[
            {"key": "nrscope", "value": "disabled"},
            {"key": "telnetsrv", "value": "enabled"},
        ],
        table_entries=8192,
        table_probes=16,
    )


class EffectiveConfigurationTests(unittest.TestCase):
    def test_schema_literal_and_codec_identity(self):
        cfg.validate_literal_file()
        self.assertEqual(cfg.SCHEMA_BYTES, cfg.canonical_bytes(cfg.EFFECTIVE_CONFIG_SCHEMA_DEFINITION))
        self.assertEqual(cfg.SCHEMA_SHA256, cfg.sha256_hex(cfg.SCHEMA_BYTES))
        self.assertEqual(cfg.parse_canonical(cfg.SCHEMA_BYTES), cfg.EFFECTIVE_CONFIG_SCHEMA_DEFINITION)
        self.assertEqual(cfg.VERSION, {"major": 1, "minor": 0})
        self.assertEqual(cfg.INSTANCE_VERSION, {"major": 1, "minor": 0})
        self.assertEqual(cfg.SCHEMA_VERSION, {"major": 1, "minor": 1})
        self.assertEqual(
            cfg.EFFECTIVE_CONFIG_SCHEMA_DEFINITION["version"], cfg.SCHEMA_VERSION
        )

    def test_round_trip_binding_and_explicit_generator(self):
        value = fixture()
        raw = cfg.serialize_effective_configuration(value)
        self.assertEqual(cfg.validate_effective_configuration_bytes(raw), value)
        self.assertEqual(value["version"], cfg.INSTANCE_VERSION)
        binding = cfg.wire_object_binding_fields(value)
        self.assertEqual(binding["object_kind"], 10)
        self.assertEqual(binding["object_flags"], 0x07)
        self.assertEqual(binding["entry_count"], 1)
        self.assertEqual(binding["byte_count"], len(raw))
        self.assertEqual(binding["sha256"], bytes.fromhex(cfg.sha256_hex(raw)))
        wire_path = (
            ROOT.parents[1] / "oai_memprof_container_wire.py"
        )
        wire_spec = importlib.util.spec_from_file_location("container_wire", wire_path)
        assert wire_spec is not None and wire_spec.loader is not None
        wire = importlib.util.module_from_spec(wire_spec)
        sys.modules[wire_spec.name] = wire
        wire_spec.loader.exec_module(wire)
        row = wire.ObjectBindingEntry(**binding)
        encoded = wire.encode_object_binding_entry(row)
        self.assertEqual(wire.decode_object_binding_entry(encoded), row)

    def test_pre_active_and_coverage_bindings(self):
        value = fixture()
        digest = cfg.configuration_sha256(value)
        cfg.validate_pre_active_bindings(
            value,
            opening_configuration_sha256=digest,
            opening_configured_thread_capacity=64,
            opening_role_kind=1,
            opening_scope_kind=1,
            run_coverage={"configuration_instance_sha256": digest, "role_id": 1},
        )
        for keyword, wrong in (
            ("opening_configuration_sha256", "f" * 64),
            ("opening_configured_thread_capacity", 65),
            ("opening_role_kind", 2),
            ("opening_scope_kind", 2),
        ):
            arguments = {
                "opening_configuration_sha256": digest,
                "opening_configured_thread_capacity": 64,
                "opening_role_kind": 1,
                "opening_scope_kind": 1,
            }
            arguments[keyword] = wrong
            with self.assertRaises(cfg.EffectiveConfigError):
                cfg.validate_pre_active_bindings(value, **arguments)
        for keyword, wrong in (
            ("opening_configuration_sha256", None),
            ("opening_configured_thread_capacity", True),
            ("opening_configured_thread_capacity", 64.0),
            ("opening_role_kind", True),
            ("opening_role_kind", 1.0),
            ("opening_scope_kind", True),
            ("opening_scope_kind", 1.0),
        ):
            arguments = {
                "opening_configuration_sha256": digest,
                "opening_configured_thread_capacity": 64,
                "opening_role_kind": 1,
                "opening_scope_kind": 1,
            }
            arguments[keyword] = wrong
            with self.assertRaises(cfg.EffectiveConfigError, msg=keyword):
                cfg.validate_pre_active_bindings(value, **arguments)
        for run_row in (
            {"configuration_instance_sha256": "f" * 64, "role_id": 1},
            {"configuration_instance_sha256": digest, "role_id": 2},
            {"configuration_instance_sha256": True, "role_id": 1},
            {"configuration_instance_sha256": digest, "role_id": 1.0},
        ):
            with self.assertRaises(cfg.EffectiveConfigError):
                cfg.validate_pre_active_bindings(
                    value,
                    opening_configuration_sha256=digest,
                    opening_configured_thread_capacity=64,
                    opening_role_kind=1,
                    opening_scope_kind=1,
                    run_coverage=run_row,
                )

    def test_module_selection_is_derived_from_config_and_role(self):
        value = fixture()
        digest = cfg.configuration_sha256(value)
        build = {
            "build_identity": {"build_configuration_sha256": "a" * 64},
            "entries": [
                {
                    "logical_id": "primary",
                    "module_selection": {
                        "operator_id": 1,
                        "predicates": [
                            {"configuration_key": None, "expected_value": None, "predicate_id": 1},
                            {"configuration_key": "telnetsrv", "expected_value": "enabled", "predicate_id": 2},
                            {"configuration_key": None, "expected_value": "gnb", "predicate_id": 3},
                        ],
                    },
                    "role_ids": [1, 2],
                },
                {
                    "logical_id": "nr_ue_only",
                    "module_selection": {
                        "operator_id": 1,
                        "predicates": [
                            {"configuration_key": None, "expected_value": None, "predicate_id": 1}
                        ],
                    },
                    "role_ids": [2],
                },
            ],
        }
        run = {
            "configuration_instance_sha256": digest,
            "module_population": [
                {"build_logical_id": "nr_ue_only", "configured": False},
                {"build_logical_id": "primary", "configured": True},
            ],
            "role_id": 1,
        }
        cfg.validate_module_selection_bindings(value, build_coverage=build, run_coverage=run)
        changed = copy.deepcopy(run)
        changed["module_population"][1]["configured"] = False
        with self.assertRaises(cfg.EffectiveConfigError):
            cfg.validate_module_selection_bindings(value, build_coverage=build, run_coverage=changed)
        missing = copy.deepcopy(value)
        missing["selection_values"] = [{"key": "nrscope", "value": "disabled"}]
        new_digest = cfg.configuration_sha256(missing)
        missing_run = copy.deepcopy(run)
        missing_run["configuration_instance_sha256"] = new_digest
        with self.assertRaises(cfg.EffectiveConfigError):
            cfg.validate_module_selection_bindings(
                missing, build_coverage=build, run_coverage=missing_run
            )

    def test_module_selection_any_true_and_false(self):
        value = fixture()
        digest = cfg.configuration_sha256(value)
        build = {
            "entries": [
                {
                    "logical_id": "any_match",
                    "module_selection": {
                        "operator_id": 2,
                        "predicates": [
                            {"configuration_key": "nrscope", "expected_value": "enabled", "predicate_id": 2},
                            {"configuration_key": "telnetsrv", "expected_value": "enabled", "predicate_id": 2},
                        ],
                    },
                    "role_ids": [1],
                },
                {
                    "logical_id": "any_miss",
                    "module_selection": {
                        "operator_id": 2,
                        "predicates": [
                            {"configuration_key": "nrscope", "expected_value": "enabled", "predicate_id": 2},
                            {"configuration_key": "telnetsrv", "expected_value": "disabled", "predicate_id": 2},
                        ],
                    },
                    "role_ids": [1],
                },
            ],
        }
        run = {
            "configuration_instance_sha256": digest,
            "module_population": [
                {"build_logical_id": "any_match", "configured": True},
                {"build_logical_id": "any_miss", "configured": False},
            ],
            "role_id": 1,
        }
        cfg.validate_module_selection_bindings(value, build_coverage=build, run_coverage=run)

    def test_two_run_instances_share_one_build_configuration(self):
        first = fixture(run_id="trial-a", seed="0001020304050607")
        second = fixture(run_id="trial-b", seed="8899aabbccddeeff")
        self.assertNotEqual(cfg.configuration_sha256(first), cfg.configuration_sha256(second))
        build = {
            "build_identity": {"build_configuration_sha256": "a" * 64},
            "entries": [
                {
                    "logical_id": "primary",
                    "module_selection": {
                        "operator_id": 1,
                        "predicates": [
                            {"configuration_key": None, "expected_value": None, "predicate_id": 1}
                        ],
                    },
                    "role_ids": [1],
                }
            ],
        }
        for value in (first, second):
            run = {
                "configuration_instance_sha256": cfg.configuration_sha256(value),
                "module_population": [
                    {"build_logical_id": "primary", "configured": True}
                ],
                "role_id": 1,
            }
            cfg.validate_module_selection_bindings(
                value, build_coverage=build, run_coverage=run
            )

    def test_noncanonical_bytes_rejected(self):
        raw = cfg.serialize_effective_configuration(fixture())
        with self.assertRaises(cfg.EffectiveConfigError):
            cfg.validate_effective_configuration_bytes(raw[:-1])
        with self.assertRaises(cfg.EffectiveConfigError):
            cfg.validate_effective_configuration_bytes(raw.replace(b'"flush_records":256', b'"flush_records": 256'))

    def test_seed_is_exactly_mode_scoped(self):
        fixture(
            mode_id=2, seed=None, provenance_id=20, status_id=20, threshold=0
        )
        for mode_id, seed, provenance, status, threshold in (
            (3, "0001020304050607", 1, 1, 0),
            (2, "0001020304050607", 20, 20, 0),
            (2, None, 20, 20, 1),
            (3, None, 2, 2, 1),
            (3, "A001020304050607", 2, 2, 1),
            (3, "00", 2, 2, 1),
            (3, "0001020304050607", 1, 2, 1),
            (3, "0001020304050607", 20, 20, 1),
        ):
            with self.assertRaises(cfg.EffectiveConfigError):
                fixture(
                    mode_id=mode_id,
                    seed=seed,
                    provenance_id=provenance,
                    status_id=status,
                    threshold=threshold,
                )
        measured = fixture(provenance_id=1, status_id=1)
        cfg.validate_sample_seed_admissibility(measured)
        with self.assertRaises(cfg.EffectiveConfigError):
            cfg.validate_sample_seed_admissibility(fixture())
        seed_bytes = bytes.fromhex(measured["sample_seed_hex"])
        self.assertEqual(int.from_bytes(seed_bytes, "big"), 0x0001020304050607)
        self.assertEqual(f"{int.from_bytes(seed_bytes, 'big'):016x}", measured["sample_seed_hex"])

    def test_paths_ids_and_exact_keys_rejected(self):
        for output in ("relative/path", "/", "/tmp//x", "/tmp/../x", "/tmp/./x", "/tmp/x\\y"):
            value = fixture()
            value["output_directory"] = output
            with self.assertRaises(cfg.EffectiveConfigError):
                cfg.validate_effective_configuration(value)
        for run_id in ("", "bad/id", "-leading", "x" * 128):
            value = fixture()
            value["run_id"] = run_id
            with self.assertRaises(cfg.EffectiveConfigError):
                cfg.validate_effective_configuration(value)
        value = fixture()
        value["extra"] = 1
        with self.assertRaises(cfg.EffectiveConfigError):
            cfg.validate_effective_configuration(value)

    def test_selection_order_and_uniqueness_rejected(self):
        for rows in (
            [{"key": "b", "value": "1"}, {"key": "a", "value": "2"}],
            [{"key": "a", "value": "1"}, {"key": "a", "value": "2"}],
            [{"key": "Bad", "value": "1"}],
            [{"key": "a", "value": ""}],
            [{"extra": "x", "key": "a", "value": "1"}],
        ):
            value = fixture()
            value["selection_values"] = rows
            with self.assertRaises(cfg.EffectiveConfigError):
                cfg.validate_effective_configuration(value)
        with self.assertRaises(cfg.EffectiveConfigError):
            cfg.make_effective_configuration(
                flush_records=256,
                flush_us=1000,
                max_threads=64,
                mode_id=3,
                output_directory="/tmp/oai-memprof",
                ring_records=4096,
                role_id=1,
                run_id="synthetic-run-0001",
                sample_seed_hex="0001020304050607",
                sample_seed_provenance_id=2,
                sample_seed_status_id=2,
                sample_threshold=1 << 60,
                scope_kind=1,
                selection_values=[{"extra": "x", "key": "a", "value": "1"}],
                table_entries=8192,
                table_probes=16,
            )

    def test_numeric_domains_and_probe_relation_rejected(self):
        cases = (
            ("flush_records", 0),
            ("flush_us", -1),
            ("max_threads", True),
            ("max_threads", cfg.UINT32_MAX),
            ("max_threads", cfg.UINT32_MAX + 1),
            ("mode_id", 1),
            ("mode_id", 6),
            ("ring_records", 0),
            ("role_id", 3),
            ("sample_seed_provenance_id", 3),
            ("sample_seed_status_id", 3),
            ("sample_threshold", cfg.UINT64_MAX + 1),
            ("scope_kind", 3),
            ("table_entries", 0),
            ("table_probes", 0),
        )
        for field, invalid in cases:
            value = fixture()
            value[field] = invalid
            with self.assertRaises(cfg.EffectiveConfigError, msg=field):
                cfg.validate_effective_configuration(value)
        value = fixture()
        value["table_entries"] = 4
        value["table_probes"] = 5
        with self.assertRaises(cfg.EffectiveConfigError):
            cfg.validate_effective_configuration(value)

    def test_versions_and_schema_definition_use_strict_json_types(self):
        for invalid in (True, False, 1.0, 0.0):
            value = fixture()
            value["version"]["major"] = invalid
            with self.assertRaises(cfg.EffectiveConfigError):
                cfg.validate_effective_configuration(value)
        for invalid in (True, 1.0):
            definition = copy.deepcopy(cfg.EFFECTIVE_CONFIG_SCHEMA_DEFINITION)
            definition["version"]["major"] = invalid
            with self.assertRaises(cfg.EffectiveConfigError):
                cfg.validate_schema_definition(definition)

    def test_schema_freezes_units_inactive_rules_and_a03_law(self):
        semantics = {
            row["field"]: row for row in cfg.EFFECTIVE_CONFIG_SCHEMA_DEFINITION["field_semantics"]
        }
        self.assertEqual(semantics["flush_records"]["unit"], "records")
        self.assertIn("may_be_partial", semantics["flush_records"]["rule"])
        self.assertEqual(semantics["flush_us"]["unit"], "microseconds")
        self.assertIn("zero_disables", semantics["flush_us"]["rule"])
        self.assertIn("per_producer", semantics["ring_records"]["rule"])
        self.assertIn("process_total", semantics["table_entries"]["rule"])
        self.assertIn("per_operation", semantics["table_probes"]["rule"])
        self.assertIn("U_less_than_q", semantics["sample_threshold"]["meaning"])
        self.assertIn("requires_1_to_2_pow_64_minus_1", semantics["sample_threshold"]["rule"])
        self.assertIn("exact_p_one_unrepresentable", semantics["sample_threshold"]["rule"])
        self.assertIn("numeric_big_endian", semantics["sample_seed_hex"]["rule"])

    def test_schema_binding_is_exact_and_future_bundle_gate_is_visible(self):
        self.assertEqual(cfg.SCHEMA_OBJECT_TYPE, 11)
        self.assertEqual(
            cfg.BUNDLE_MEMBER_PROPOSAL,
            {
                "name": "effective_config_schema",
                "object_type": 11,
                "owner": "config",
                "path": "definition/effective-config-schema-v1.json",
            },
        )
        self.assertEqual(cfg.BUNDLE_ENTRY_PROPOSAL["bytes"], len(cfg.SCHEMA_BYTES))
        self.assertEqual(cfg.BUNDLE_ENTRY_PROPOSAL["sha256"], cfg.SCHEMA_SHA256)
        self.assertEqual(
            cfg.BUNDLE_CROSS_RELATION_PROPOSAL,
            {
                "left": "metadata/effective-config.json.schema.sha256",
                "relation_id": "effective_configuration_schema",
                "right": "schema_bundle.entries[object_type=11].sha256",
            },
        )
        value = fixture()
        for field, invalid in (
            ("object_type", 10),
            ("path", "definition/other.json"),
            ("sha256", "f" * 64),
        ):
            changed = copy.deepcopy(value)
            changed["schema"][field] = invalid
            with self.assertRaises(cfg.EffectiveConfigError):
                cfg.validate_effective_configuration(changed)


if __name__ == "__main__":
    unittest.main()
