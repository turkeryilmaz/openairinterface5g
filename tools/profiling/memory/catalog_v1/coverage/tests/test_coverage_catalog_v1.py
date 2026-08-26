#!/usr/bin/env python3
"""Focused independent oracles for the static and generated coverage layer."""

from __future__ import annotations

import copy
import importlib.util
import pathlib
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "coverage_catalog_v1", ROOT / "coverage_catalog_v1.py"
)
assert SPEC and SPEC.loader
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)

H = "1" * 64
API_H = "93056c4cfd071c1df396ba09bf82b4cbe807923977c4bca988b0aee1b8c94610"


def binding(object_type: int, path: str, digest: str) -> dict:
    return {"object_type": object_type, "path": path, "sha256": digest}


def logical_elf(*, admission: int = 1, with_unsupported: bool = True) -> dict:
    origins = [
        {"api_id": 1, "classification_id": 1, "origin_id": 1,
         "origin_kind_id": 1, "symbol": "malloc", "symbol_version": "GLIBC_2.2.5"},
        {"api_id": 2, "classification_id": 1, "origin_id": 2,
         "origin_kind_id": 1, "symbol": "calloc", "symbol_version": "GLIBC_2.2.5"},
        {"api_id": 3, "classification_id": 1, "origin_id": 3,
         "origin_kind_id": 1, "symbol": "realloc", "symbol_version": "GLIBC_2.2.5"},
        {"api_id": 4, "classification_id": 1, "origin_id": 4,
         "origin_kind_id": 1, "symbol": "free", "symbol_version": "GLIBC_2.2.5"},
    ]
    if with_unsupported:
        origins.append(
            {"api_id": None, "classification_id": 10, "origin_id": 103,
             "origin_kind_id": 2, "symbol": "__asprintf_chk",
             "symbol_version": "GLIBC_2.8"}
        )
    return {
        "admission_state_id": admission,
        "aliases": [],
        "build_id": "01",
        "byte_count": 4096,
        "dt_needed": ["libc.so.6"],
        "elf_kind_id": 1,
        "elf_machine": 62,
        "evidence_state_id": 1,
        "hidden_wrapper_symbols": [
            "__wrap_calloc", "__wrap_free", "__wrap_malloc", "__wrap_realloc"
        ],
        "import_relocations": [
            {"origin_id": origin["origin_id"], "relocation_kind_id": 1}
            for origin in origins
        ],
        "link_command_sha256": H,
        "link_map_sha256": H,
        "logical_id": "nr_softmodem",
        "module_selection": {
            "operator_id": 1,
            "predicates": [
                {"configuration_key": None, "expected_value": None, "predicate_id": 1}
            ],
        },
        "realloc_zero_policy_id": 1,
        "realloc_zero_semantic_oracle_sha256": (
            C.REALLOC_ZERO_SEMANTIC_ORACLE_BINDING["sha256"]
        ),
        "repo_path": "cmake_targets/ran_build/build/nr-softmodem",
        "role_ids": [1],
        "sha256": H,
        "shared_runtime_binding": {
            "dependency_id": "glibc_runtime", "evidence_state_id": 1,
            "soname": "libc.so.6",
        },
        "soname": None,
        "symbol_origins": origins,
        "wrap_options": ["--wrap=calloc", "--wrap=free", "--wrap=malloc", "--wrap=realloc"],
    }


def add_known_unsupported(
    logical: dict, *, origin_id: int, symbol: str, symbol_version: str
) -> None:
    logical["symbol_origins"].append(
        {
            "api_id": None,
            "classification_id": 10,
            "origin_id": origin_id,
            "origin_kind_id": 2,
            "symbol": symbol,
            "symbol_version": symbol_version,
        }
    )
    logical["symbol_origins"].sort(key=lambda row: row["origin_id"])
    logical["import_relocations"].append(
        {"origin_id": origin_id, "relocation_kind_id": 1}
    )
    logical["import_relocations"].sort(key=lambda row: row["origin_id"])


def add_supported(logical: dict, *, api_id: int, architecture_id: int) -> None:
    rule = {row["api_id"]: row for row in C.API_RULES}[api_id]
    logical["symbol_origins"].append(
        {
            "api_id": api_id,
            "classification_id": 1,
            "origin_id": api_id,
            "origin_kind_id": 1,
            "symbol": rule["import_symbol"],
            "symbol_version": C.expected_supported_symbol_version(
                api_id, architecture_id
            ),
        }
    )
    logical["symbol_origins"].sort(key=lambda row: row["origin_id"])
    logical["import_relocations"].append(
        {"origin_id": api_id, "relocation_kind_id": 1}
    )
    logical["import_relocations"].sort(key=lambda row: row["origin_id"])
    logical["hidden_wrapper_symbols"].append(rule["wrapper_symbol"])
    logical["hidden_wrapper_symbols"].sort()
    logical["wrap_options"].append(rule["wrap_option"])
    logical["wrap_options"].sort()


def build(*, synthetic: bool = True) -> dict:
    return {
        "api_definition": binding(4, "catalog/api.json", API_H),
        "architecture_id": 1,
        "build_identity": {
            "build_configuration_sha256": H,
            "compiler_id": "gcc", "compiler_version": "15.1.0",
            "dirty": False,
            "libc_id": "glibc", "libc_version": "2.41",
            "linker_id": "gnu_ld", "linker_version": "2.44",
            "operating_system": "linux", "primary_logical_elf_id": "nr_softmodem",
            "source_commit": "1" * 40, "source_tree": "2" * 40,
            "target_triple": "x86_64-linux-gnu",
        },
        "catalog_id": "oai_memprof_build_coverage",
        "dependencies": [
            {"dependency_id": "glibc_runtime", "evidence_state_id": 1,
             "name": "libc.so.6", "sha256": H, "version": "2.41"}
        ],
        "entries": [logical_elf()],
        "evidence_origin_id": 2 if synthetic else 1,
        "failure_ids": [90] if synthetic else [],
        "policy": binding(9, C.POLICY_ARCHIVE_PATH, C.POLICY_SHA256),
        "schema": binding(10, C.INSTANCE_SCHEMA_ARCHIVE_PATH, C.INSTANCE_SCHEMA_SHA256),
        "verdict_id": 10 if synthetic else 1,
        "version": {"major": 1, "minor": 0},
    }


def run(
    build_row: dict, *, synthetic: bool = True, configuration_digest: str = H
) -> dict:
    logical = build_row["entries"][0]
    module_map = "3" * 64
    return {
        "build_coverage": {
            "object_kind": 8, "path": C.BUILD_COVERAGE_ARCHIVE_PATH,
            "sha256": C.sha256_hex(C.canonical_bytes(build_row)),
        },
        "catalog_id": "oai_memprof_run_coverage",
        "configuration_instance_sha256": configuration_digest,
        "eligible_exact_domain": not synthetic,
        "evidence_origin_id": 2 if synthetic else 1,
        "failure_ids": [90] if synthetic else [],
        "module_population": [
            {"admission_state_id": 1, "build_logical_id": "nr_softmodem",
             "classifications": [
                 {"classification_id": origin["classification_id"],
                  "origin_id": origin["origin_id"]}
                 for origin in logical["symbol_origins"]
             ],
             "configured": True, "load_generation": 7, "load_state_id": 1,
             "loaded_path": "/opt/oai/nr-softmodem", "logical_id": "nr_softmodem",
             "observed": True,
             "runtime_identity": C._expected_runtime_identity(logical, module_map)},
        ],
        "policy": binding(9, C.POLICY_ARCHIVE_PATH, C.POLICY_SHA256),
        "process_generation": 7,
        "process_uuid": "12345678-1234-4234-8234-123456789abc",
        "role_id": 1,
        "run_uuid": "abcdefab-cdef-4abc-8def-abcdefabcdef",
        "schema": binding(10, C.INSTANCE_SCHEMA_ARCHIVE_PATH, C.INSTANCE_SCHEMA_SHA256),
        "snapshot_state_id": 1,
        "verdict_id": 10 if synthetic else 2,
        "version": {"major": 1, "minor": 0},
    }


def validate_run(
    value: dict,
    build_row: dict,
    *,
    expected_configuration_digest: str | None = None,
) -> None:
    C.validate_run_coverage(
        value,
        build_coverage=build_row,
        api_definition_sha256=API_H,
        expected_configuration_instance_sha256=(
            value["configuration_instance_sha256"]
            if expected_configuration_digest is None
            else expected_configuration_digest
        ),
    )


def population_row(
    logical: dict, module_map_sha256: str, *, active: bool
) -> dict:
    return {
        "admission_state_id": logical["admission_state_id"],
        "build_logical_id": logical["logical_id"],
        "classifications": [
            {
                "classification_id": origin["classification_id"],
                "origin_id": origin["origin_id"],
            }
            for origin in logical["symbol_origins"]
        ],
        "configured": active,
        "load_generation": 7 if active else None,
        "load_state_id": 1 if active else 11,
        "loaded_path": f"/opt/oai/{logical['logical_id']}" if active else None,
        "logical_id": logical["logical_id"],
        "observed": active,
        "runtime_identity": (
            C._expected_runtime_identity(logical, module_map_sha256)
            if active
            else None
        ),
    }


def secondary_logical(logical: dict, *, policy_id: int = 1) -> dict:
    row = copy.deepcopy(logical)
    row.update(
        {
            "build_id": "02",
            "elf_kind_id": 2,
            "logical_id": "optional_module",
            "realloc_zero_policy_id": policy_id,
            "repo_path": "cmake_targets/ran_build/build/liboptional.so",
            "sha256": "6" * 64,
            "soname": "liboptional.so",
        }
    )
    return row


def without_realloc(logical: dict) -> dict:
    row = copy.deepcopy(logical)
    row["symbol_origins"] = [
        origin for origin in row["symbol_origins"] if origin["api_id"] != 3
    ]
    origin_ids = {origin["origin_id"] for origin in row["symbol_origins"]}
    row["import_relocations"] = [
        relation
        for relation in row["import_relocations"]
        if relation["origin_id"] in origin_ids
    ]
    row["hidden_wrapper_symbols"].remove("__wrap_realloc")
    row["wrap_options"].remove("--wrap=realloc")
    row["realloc_zero_policy_id"] = None
    row["realloc_zero_semantic_oracle_sha256"] = None
    return row


def measured_run_for_population(
    build_row: dict, rows: list[tuple[dict, bool]]
) -> dict:
    build_row["entries"] = sorted(
        (logical for logical, _active in rows),
        key=lambda logical: logical["logical_id"],
    )
    value = run(build_row, synthetic=False)
    value["module_population"] = [
        population_row(logical, str(index + 3) * 64, active=active)
        for index, (logical, active) in enumerate(
            sorted(rows, key=lambda item: item[0]["logical_id"])
        )
    ]
    value["build_coverage"]["sha256"] = C.sha256_hex(
        C.canonical_bytes(build_row)
    )
    return value


def resolve_policy(
    value: dict, build_row: dict, *, expected_configuration_digest: str | None = None
):
    return C.resolve_run_realloc_zero_policy(
        value,
        build_coverage=build_row,
        api_definition_sha256=API_H,
        expected_configuration_instance_sha256=(
            value["configuration_instance_sha256"]
            if expected_configuration_digest is None
            else expected_configuration_digest
        ),
    )


class CoverageTests(unittest.TestCase):
    def assert_rejects(self, callable_) -> None:
        with self.assertRaises(C.CoverageError):
            callable_()

    def test_static_definitions_are_exact_canonical_members(self) -> None:
        self.assertEqual(C.POLICY_VERSION, {"major": 2, "minor": 0})
        self.assertEqual(C.INSTANCE_VERSION, {"major": 1, "minor": 0})
        self.assertEqual(len(C.POLICY_BYTES), 11942)
        self.assertEqual(
            C.POLICY_SHA256,
            "2d01a2e3f066787378e7bebfe50f618d94dd46e4f38ede300adc0ad178f31969",
        )
        self.assertEqual(len(C.INSTANCE_SCHEMA_BYTES), 11504)
        self.assertEqual(
            C.INSTANCE_SCHEMA_SHA256,
            "86176978f48ef9e888bfb373de713a9729e47e0bce4d9cc2fa8c917fc13e6773",
        )
        self.assertEqual(len(C.API_RULES), 12)
        self.assertEqual(
            C.FAIL_CLOSED_RULES[-1],
            "supported_origin_symbol_versions_are_architecture_exact",
        )
        self.assertEqual(C.parse_canonical(C.POLICY_BYTES), C.COVERAGE_POLICY_DEFINITION)
        self.assertEqual(C.parse_canonical(C.INSTANCE_SCHEMA_BYTES), C.COVERAGE_INSTANCE_SCHEMA_DEFINITION)
        for path, raw in C.static_members().items():
            self.assertEqual(C.validate_static_member(path, raw), C.parse_canonical(raw))
            self.assertEqual((ROOT / "archive" / path).read_bytes(), raw)
        self.assertNotIn("sha256", C.COVERAGE_POLICY_DEFINITION)
        self.assertNotIn("sha256", C.COVERAGE_INSTANCE_SCHEMA_DEFINITION)
        self.assertEqual(
            C.COVERAGE_POLICY_DEFINITION["realloc_zero_oracle_binding"],
            C.REALLOC_ZERO_SEMANTIC_ORACLE_BINDING,
        )
        oracle = C.REALLOC_ZERO_SEMANTIC_ORACLE_BINDING
        oracle_bytes = (ROOT.parent / "semantic" / "archive" / oracle["path"]).read_bytes()
        self.assertEqual(C.canonical_bytes(C.parse_canonical(oracle_bytes)), oracle_bytes)
        self.assertEqual(C.sha256_hex(oracle_bytes), oracle["sha256"])

    def test_canonical_codec_matches_del_rule_and_rejects_noncanonical(self) -> None:
        self.assertEqual(C.canonical_bytes({"a": "\x7f"}), b'{"a":"\\u007f"}\n')
        self.assert_rejects(lambda: C.parse_canonical(b'{"a":"\x7f"}\n'))
        self.assert_rejects(lambda: C.parse_canonical(b'{"a":1, "b":2}\n'))
        self.assert_rejects(lambda: C.parse_canonical(b'{"a":1,"a":2}\n'))
        self.assert_rejects(lambda: C.parse_canonical(b'{"a":1.0}\n'))

    def test_canonical_parser_limits_are_string_aware_and_typed(self) -> None:
        nested: object = 'literal [{]} escaped quote " and slash \\ [}]'
        for _ in range(C.MAX_JSON_NESTING_DEPTH - 1):
            nested = [nested]
        valid = C.canonical_bytes({"a": nested})
        self.assertEqual(C.parse_canonical(valid), {"a": nested})

        over_limit_raw = (
            b'{"a":'
            + b"[" * C.MAX_JSON_NESTING_DEPTH
            + b"0"
            + b"]" * C.MAX_JSON_NESTING_DEPTH
            + b"}\n"
        )
        with mock.patch.object(C.json, "loads", side_effect=AssertionError("json.loads called")):
            with self.assertRaisesRegex(C.CoverageError, "nesting depth"):
                C.parse_canonical(over_limit_raw)

        original_limit = C.MAX_CANONICAL_RAW_BYTES
        try:
            C.MAX_CANONICAL_RAW_BYTES = len(valid) - 1
            with mock.patch.object(C.json, "loads", side_effect=AssertionError("json.loads called")):
                with self.assertRaisesRegex(C.CoverageError, "bytes exceed"):
                    C.parse_canonical(valid)
        finally:
            C.MAX_CANONICAL_RAW_BYTES = original_limit

        with mock.patch.object(C.json, "loads", side_effect=RecursionError("synthetic")):
            with self.assertRaisesRegex(C.CoverageError, "nesting depth"):
                C.parse_canonical(b'{"a":0}\n')
        with mock.patch.object(C, "_encode", side_effect=RecursionError("synthetic")):
            with self.assertRaisesRegex(C.CoverageError, "nesting depth"):
                C.canonical_bytes({"a": 0})

    def test_schema_separates_admission_classification_and_population_union(self) -> None:
        schemas = {row["name"]: row for row in C.OBJECT_SCHEMAS}
        population_fields = {row["name"] for row in schemas["module_population"]["fields"]}
        run_fields = {row["name"] for row in schemas["run_coverage"]["fields"]}
        self.assertIn("admission_state_id", population_fields)
        self.assertIn("classifications", population_fields)
        self.assertIn("module_population", run_fields)
        self.assertNotIn("loaded_modules", run_fields)
        identity_fields = {row["name"] for row in schemas["build_identity"]["fields"]}
        self.assertIn("build_configuration_sha256", identity_fields)
        self.assertNotIn("configuration_sha256", identity_fields)

    def test_synthetic_and_measured_builds_validate(self) -> None:
        synthetic = build()
        C.validate_build_coverage(synthetic, api_definition_sha256=API_H)
        measured = build(synthetic=False)
        C.validate_build_coverage(measured, api_definition_sha256=API_H)
        self.assertEqual(measured["entries"][0]["admission_state_id"], 1)
        self.assertIn(10, [row["classification_id"] for row in measured["entries"][0]["symbol_origins"]])

    def test_runtime_binding_soname_must_occur_in_dt_needed(self) -> None:
        missing_needed = build(synthetic=False)
        logical = missing_needed["entries"][0]
        runtime = logical["shared_runtime_binding"]
        dependency = missing_needed["dependencies"][0]
        self.assertEqual(runtime["dependency_id"], dependency["dependency_id"])
        self.assertEqual(runtime["soname"], dependency["name"])
        self.assertIn(runtime["soname"], logical["dt_needed"])
        logical["dt_needed"] = []

        with self.assertRaisesRegex(C.CoverageError, "admitted state contradicts failures"):
            C.validate_build_coverage(missing_needed, api_definition_sha256=API_H)

        retained = copy.deepcopy(missing_needed)
        retained["entries"][0]["admission_state_id"] = 62
        retained["failure_ids"] = [62]
        retained["verdict_id"] = 10
        C.validate_build_coverage(retained, api_definition_sha256=API_H)

    def test_runtime_binding_dependency_name_must_equal_soname(self) -> None:
        mismatched_name = build(synthetic=False)
        logical = mismatched_name["entries"][0]
        runtime = logical["shared_runtime_binding"]
        dependency = mismatched_name["dependencies"][0]
        self.assertEqual(runtime["dependency_id"], dependency["dependency_id"])
        self.assertEqual(runtime["evidence_state_id"], 1)
        self.assertEqual(dependency["evidence_state_id"], 1)
        self.assertIn(runtime["soname"], logical["dt_needed"])
        dependency["name"] = "libc_wrong.so.6"
        self.assertNotEqual(runtime["soname"], dependency["name"])

        with self.assertRaisesRegex(C.CoverageError, "admitted state contradicts failures"):
            C.validate_build_coverage(mismatched_name, api_definition_sha256=API_H)

        retained = copy.deepcopy(mismatched_name)
        retained["entries"][0]["admission_state_id"] = 62
        retained["failure_ids"] = [62]
        retained["verdict_id"] = 10
        C.validate_build_coverage(retained, api_definition_sha256=API_H)

    def test_shared_module_soname_is_nullable_but_if_present_is_typed(self) -> None:
        measured = build(synthetic=False)
        module = copy.deepcopy(measured["entries"][0])
        module.update(
            {
                "build_id": "02",
                "elf_kind_id": 2,
                "logical_id": "optional_module",
                "repo_path": "cmake_targets/ran_build/build/liboptional.so",
                "sha256": "6" * 64,
                "soname": None,
            }
        )
        measured["entries"].append(module)
        C.validate_build_coverage(measured, api_definition_sha256=API_H)

        named = copy.deepcopy(measured)
        named["entries"][1]["soname"] = "liboptional.so"
        C.validate_build_coverage(named, api_definition_sha256=API_H)

        malformed = copy.deepcopy(measured)
        malformed["entries"][1]["soname"] = True
        self.assert_rejects(
            lambda: C.validate_build_coverage(
                malformed, api_definition_sha256=API_H
            )
        )

    def test_supported_and_unsupported_versions_are_architecture_exact(self) -> None:
        self.assertEqual([row["api_id"] for row in C.API_RULES], list(range(1, 13)))
        self.assertEqual(
            [row["origin_id"] for row in C.KNOWN_UNSUPPORTED_ORIGINS],
            list(range(101, 105)),
        )
        self.assertEqual(C.expected_known_unsupported_symbol_version(103, 1), "GLIBC_2.8")
        self.assertEqual(C.expected_supported_symbol_version(5, 1), "GLIBC_2.26")
        self.assertEqual(C.expected_supported_symbol_version(6, 1), "GLIBC_2.16")
        self.assertEqual(C.expected_supported_symbol_version(6, 2), "GLIBC_2.17")
        self.assertEqual(C.expected_supported_symbol_version(12, 2), "GLIBC_2.17")

        x86 = build(synthetic=False)
        add_supported(x86["entries"][0], api_id=5, architecture_id=1)
        C.validate_build_coverage(x86, api_definition_sha256=API_H)
        wrong_x86 = copy.deepcopy(x86)
        next(row for row in wrong_x86["entries"][0]["symbol_origins"] if row["origin_id"] == 5)["symbol_version"] = "GLIBC_2.2.5"
        self.assert_rejects(lambda: C.validate_build_coverage(wrong_x86, api_definition_sha256=API_H))

        aarch64 = build(synthetic=False)
        aarch64["architecture_id"] = 2
        aarch64["build_identity"]["target_triple"] = "aarch64-linux-gnu"
        aarch64["entries"][0]["elf_machine"] = 183
        for origin in aarch64["entries"][0]["symbol_origins"]:
            origin["symbol_version"] = "GLIBC_2.17"
        add_supported(aarch64["entries"][0], api_id=6, architecture_id=2)
        C.validate_build_coverage(aarch64, api_definition_sha256=API_H)
        wrong_aarch64 = copy.deepcopy(aarch64)
        next(row for row in wrong_aarch64["entries"][0]["symbol_origins"] if row["origin_id"] == 6)["symbol_version"] = "GLIBC_2.16"
        self.assert_rejects(lambda: C.validate_build_coverage(wrong_aarch64, api_definition_sha256=API_H))

        promoted = build(synthetic=False)
        add_known_unsupported(
            promoted["entries"][0], origin_id=105,
            symbol="aligned_alloc", symbol_version="GLIBC_2.16",
        )
        self.assert_rejects(lambda: C.validate_build_coverage(promoted, api_definition_sha256=API_H))

        for helper, identifier, architecture in (
            (C.expected_supported_symbol_version, True, 1),
            (C.expected_supported_symbol_version, 13, 1),
            (C.expected_supported_symbol_version, 5, True),
            (C.expected_known_unsupported_symbol_version, 105, 1),
            (C.expected_known_unsupported_symbol_version, 103, 3),
        ):
            with self.subTest(helper=helper.__name__, identifier=identifier, architecture=architecture):
                self.assert_rejects(lambda helper=helper, identifier=identifier, architecture=architecture: helper(identifier, architecture))

    def test_policy_major_version_rejects_a_v12_tag(self) -> None:
        historical_tag = copy.deepcopy(C.COVERAGE_POLICY_DEFINITION)
        historical_tag["version"] = {"major": 1, "minor": 2}
        self.assert_rejects(lambda: C.validate_coverage_policy(historical_tag))

    def test_api5_alone_participates_in_run_wide_realloc_policy_resolution(self) -> None:
        measured_build = build(synthetic=False)
        primary = without_realloc(measured_build["entries"][0])
        add_supported(primary, api_id=5, architecture_id=1)
        primary["realloc_zero_policy_id"] = 2
        primary["realloc_zero_semantic_oracle_sha256"] = (
            C.REALLOC_ZERO_SEMANTIC_ORACLE_BINDING["sha256"]
        )
        measured_run = measured_run_for_population(measured_build, [(primary, True)])
        resolution = resolve_policy(measured_run, measured_build)
        self.assertEqual(resolution.policy_id, 2)
        self.assertEqual(
            resolution.semantic_oracle_sha256,
            C.REALLOC_ZERO_SEMANTIC_ORACLE_BINDING["sha256"],
        )

    def test_build_fails_closed_on_oracle_ids_order_paths_and_symbols(self) -> None:
        mutants = []
        mutated = build(); mutated["entries"][0]["realloc_zero_semantic_oracle_sha256"] = None; mutants.append(mutated)
        mutated = build(); mutated["entries"][0]["admission_state_id"] = 99; mutants.append(mutated)
        mutated = build(); mutated["entries"][0]["repo_path"] = "../nr-softmodem"; mutants.append(mutated)
        mutated = build(); mutated["entries"][0]["symbol_origins"][0]["symbol_version"] = "GLIBC_2.17"; mutants.append(mutated)
        mutated = build(); mutated["entries"][0]["symbol_origins"].reverse(); mutants.append(mutated)
        mutated = build(); mutated["entries"][0]["hidden_wrapper_symbols"].reverse(); mutants.append(mutated)
        mutated = build(); mutated["entries"][0]["import_relocations"].pop(); mutants.append(mutated)
        for mutant in mutants:
            self.assert_rejects(lambda mutant=mutant: C.validate_build_coverage(mutant, api_definition_sha256=API_H))

    def test_build_rejects_wrong_realloc_zero_oracle_digest(self) -> None:
        mutant = build(synthetic=False)
        mutant["entries"][0]["realloc_zero_semantic_oracle_sha256"] = "3" * 64
        self.assertNotEqual(
            mutant["entries"][0]["realloc_zero_semantic_oracle_sha256"],
            C.REALLOC_ZERO_SEMANTIC_ORACLE_BINDING["sha256"],
        )
        self.assert_rejects(
            lambda: C.validate_build_coverage(mutant, api_definition_sha256=API_H)
        )

    def test_synthetic_and_measured_run_derived_eligibility(self) -> None:
        synthetic_build = build()
        validate_run(run(synthetic_build), synthetic_build)
        measured_build = build(synthetic=False)
        measured_run = run(measured_build, synthetic=False)
        validate_run(measured_run, measured_build)
        mutant = copy.deepcopy(measured_run)
        mutant["eligible_exact_domain"] = False
        self.assert_rejects(lambda: validate_run(mutant, measured_build))

    def test_two_per_run_configurations_bind_same_build_independently(self) -> None:
        measured_build = build(synthetic=False)
        first = run(measured_build, synthetic=False, configuration_digest="3" * 64)
        second = run(measured_build, synthetic=False, configuration_digest="4" * 64)
        validate_run(first, measured_build)
        validate_run(second, measured_build)
        self.assertEqual(
            measured_build["build_identity"]["build_configuration_sha256"], H
        )
        changed_build = copy.deepcopy(measured_build)
        changed_build["build_identity"]["build_configuration_sha256"] = "5" * 64
        changed_run = run(changed_build, synthetic=False, configuration_digest="3" * 64)
        validate_run(changed_run, changed_build)

    def test_run_digest_mismatch_is_failure34_not_build_equality(self) -> None:
        measured_build = build(synthetic=False)
        mismatched = run(
            measured_build, synthetic=False, configuration_digest="3" * 64
        )
        self.assert_rejects(
            lambda: validate_run(
                mismatched,
                measured_build,
                expected_configuration_digest="4" * 64,
            )
        )
        mismatched["eligible_exact_domain"] = False
        mismatched["failure_ids"] = [34]
        mismatched["verdict_id"] = 10
        validate_run(
            mismatched,
            measured_build,
            expected_configuration_digest="4" * 64,
        )

    def test_run_bytes_requires_opening_configuration_digest(self) -> None:
        measured_build = build(synthetic=False)
        measured_run = run(
            measured_build, synthetic=False, configuration_digest="3" * 64
        )
        raw = C.canonical_bytes(measured_run)
        self.assertEqual(
            C.validate_run_coverage_bytes(
                raw,
                build_coverage=measured_build,
                api_definition_sha256=API_H,
                expected_configuration_instance_sha256="3" * 64,
            ),
            measured_run,
        )
        self.assert_rejects(
            lambda: C.validate_run_coverage_bytes(
                raw,
                build_coverage=measured_build,
                api_definition_sha256=API_H,
                expected_configuration_instance_sha256="4" * 64,
            )
        )

    def test_run_population_exact_union_identity_and_active_time_fail_closed(self) -> None:
        measured_build = build(synthetic=False)
        base = run(measured_build, synthetic=False)
        missing = copy.deepcopy(base); missing["module_population"] = []
        changed = copy.deepcopy(base); changed["module_population"][0]["runtime_identity"]["sha256"] = "4" * 64
        generation = copy.deepcopy(base); generation["module_population"][0]["load_generation"] = 8
        for mutant in (missing, changed, generation):
            self.assert_rejects(lambda mutant=mutant: validate_run(mutant, measured_build))
        active = copy.deepcopy(base)
        active["module_population"][0]["load_state_id"] = 21
        active["eligible_exact_domain"] = False
        active["failure_ids"] = [33]
        active["verdict_id"] = 10
        validate_run(active, measured_build)

    def test_configured_population_mismatch_still_requires_failure34(self) -> None:
        measured_build = build(synthetic=False)
        secondary = copy.deepcopy(measured_build["entries"][0])
        secondary.update(
            {
                "build_id": "02",
                "elf_kind_id": 2,
                "logical_id": "optional_module",
                "repo_path": "cmake_targets/ran_build/build/liboptional.so",
                "sha256": "6" * 64,
                "soname": "liboptional.so",
            }
        )
        measured_build["entries"].append(secondary)
        mismatch = run(measured_build, synthetic=False)
        mismatch["module_population"].append(
            {
                "admission_state_id": 1,
                "build_logical_id": "optional_module",
                "classifications": [
                    {
                        "classification_id": origin["classification_id"],
                        "origin_id": origin["origin_id"],
                    }
                    for origin in secondary["symbol_origins"]
                ],
                "configured": False,
                "load_generation": 7,
                "load_state_id": 20,
                "loaded_path": "/opt/oai/liboptional.so",
                "logical_id": "optional_module",
                "observed": True,
                "runtime_identity": C._expected_runtime_identity(
                    secondary, "7" * 64
                ),
            }
        )
        mismatch["build_coverage"]["sha256"] = C.sha256_hex(
            C.canonical_bytes(measured_build)
        )
        mismatch["eligible_exact_domain"] = False
        mismatch["verdict_id"] = 10
        self.assert_rejects(lambda: validate_run(mismatch, measured_build))
        mismatch["failure_ids"] = [34]
        validate_run(mismatch, measured_build)

    def test_unexpected_module_requires_exact_union_failure_shape(self) -> None:
        measured_build = build(synthetic=False)
        base = run(measured_build, synthetic=False)
        extra = {
            "admission_state_id": 61, "build_logical_id": None,
            "classifications": [{"classification_id": 61, "origin_id": None}],
            "configured": False, "load_generation": 7, "load_state_id": 20,
            "loaded_path": "/tmp/extra.so", "logical_id": "extra_module", "observed": True,
            "runtime_identity": {
                "build_id": "02", "byte_count": 1,
                "hidden_wrapper_symbols_sha256": H, "import_relocations_sha256": H,
                "module_map_sha256": H,
                "sha256": H, "shared_runtime_binding_sha256": H,
                "symbol_origins_sha256": H,
            },
        }
        base["module_population"].insert(0, extra)
        base["eligible_exact_domain"] = False
        base["failure_ids"] = [21]
        base["verdict_id"] = 10
        validate_run(base, measured_build)
        base["module_population"][0]["admission_state_id"] = 1
        self.assert_rejects(lambda: validate_run(base, measured_build))

    def test_realloc_policy_id_is_strict_known_integer(self) -> None:
        for invalid in (True, False, 1.0, 0.0, None, 3, -1, "1", []):
            measured_build = build(synthetic=False)
            measured_build["entries"][0]["realloc_zero_policy_id"] = invalid
            with self.subTest(invalid=repr(invalid)):
                self.assert_rejects(
                    lambda measured_build=measured_build: C.validate_build_coverage(
                        measured_build, api_definition_sha256=API_H
                    )
                )

    def test_realloc_policy_fields_validate_nonnull_side_before_pair_completeness(
        self,
    ) -> None:
        retained_missing = build(synthetic=False)
        retained_missing["entries"][0]["admission_state_id"] = 41
        retained_missing["entries"][0]["realloc_zero_policy_id"] = None
        retained_missing["entries"][0]["realloc_zero_semantic_oracle_sha256"] = None
        retained_missing["failure_ids"] = [13]
        retained_missing["verdict_id"] = 10
        C.validate_build_coverage(retained_missing, api_definition_sha256=API_H)

        for invalid_policy in (True, False, 1.0, 0.0, 3, -1, "1", []):
            candidate = copy.deepcopy(retained_missing)
            candidate["entries"][0]["realloc_zero_policy_id"] = invalid_policy
            with self.subTest(field="policy", invalid=repr(invalid_policy)):
                self.assert_rejects(
                    lambda candidate=candidate: C.validate_build_coverage(
                        candidate, api_definition_sha256=API_H
                    )
                )

        for invalid_oracle in (True, 1, "", "1" * 63, "g" * 64, []):
            candidate = copy.deepcopy(retained_missing)
            candidate["entries"][0][
                "realloc_zero_semantic_oracle_sha256"
            ] = invalid_oracle
            with self.subTest(field="oracle", invalid=repr(invalid_oracle)):
                self.assert_rejects(
                    lambda candidate=candidate: C.validate_build_coverage(
                        candidate, api_definition_sha256=API_H
                    )
                )

    def test_resolver_single_policy_is_exact_immutable_and_nonmutating(self) -> None:
        measured_build = build(synthetic=False)
        logical = measured_build["entries"][0]
        measured_run = measured_run_for_population(
            measured_build, [(logical, True)]
        )
        build_snapshot = copy.deepcopy(measured_build)
        run_snapshot = copy.deepcopy(measured_run)
        result = resolve_policy(measured_run, measured_build)
        self.assertEqual(
            result,
            C.ReallocZeroPolicyResolution(
                "resolved",
                1,
                C.REALLOC_ZERO_SEMANTIC_ORACLE_BINDING["sha256"],
            ),
        )
        with self.assertRaises(AttributeError):
            result.policy_id = 2
        self.assertEqual(measured_build, build_snapshot)
        self.assertEqual(measured_run, run_snapshot)

    def test_resolver_is_order_independent_and_ignores_active_nonrealloc(self) -> None:
        measured_build = build(synthetic=False)
        primary = measured_build["entries"][0]
        primary["realloc_zero_policy_id"] = 2
        secondary = secondary_logical(primary, policy_id=2)
        measured_run = measured_run_for_population(
            measured_build, [(primary, True), (secondary, True)]
        )
        self.assertEqual(resolve_policy(measured_run, measured_build).policy_id, 2)
        build_rows = {
            row["logical_id"]: row for row in measured_build["entries"]
        }
        population = measured_run["module_population"]
        forward = C._active_realloc_policy_pairs(
            population, build_rows, [True] * len(population)
        )
        reverse = C._active_realloc_policy_pairs(
            list(reversed(population)), build_rows, [True] * len(population)
        )
        self.assertEqual(forward, reverse)

        measured_build = build(synthetic=False)
        primary = measured_build["entries"][0]
        secondary = without_realloc(secondary_logical(primary))
        measured_run = measured_run_for_population(
            measured_build, [(primary, True), (secondary, True)]
        )
        self.assertEqual(resolve_policy(measured_run, measured_build).policy_id, 1)

    def test_mixed_active_policies_require_failure22_and_never_resolve(self) -> None:
        measured_build = build(synthetic=False)
        primary = measured_build["entries"][0]
        secondary = secondary_logical(primary, policy_id=2)
        measured_run = measured_run_for_population(
            measured_build, [(primary, True), (secondary, True)]
        )
        with self.assertRaisesRegex(C.CoverageError, "missing causal failure IDs"):
            validate_run(measured_run, measured_build)

        measured_run["eligible_exact_domain"] = False
        measured_run["failure_ids"] = [22]
        measured_run["verdict_id"] = 10
        validate_run(measured_run, measured_build)
        measured_run["verdict_id"] = 20
        with self.assertRaisesRegex(
            C.CoverageError, "mixed active realloc policies require 10"
        ):
            validate_run(measured_run, measured_build)
        measured_run["verdict_id"] = 10
        build_snapshot = copy.deepcopy(measured_build)
        run_snapshot = copy.deepcopy(measured_run)
        self.assert_rejects(lambda: resolve_policy(measured_run, measured_build))
        self.assertEqual(measured_build, build_snapshot)
        self.assertEqual(measured_run, run_snapshot)

    def test_inactive_different_policy_does_not_conflict(self) -> None:
        measured_build = build(synthetic=False)
        primary = measured_build["entries"][0]
        secondary = secondary_logical(primary, policy_id=2)
        measured_run = measured_run_for_population(
            measured_build, [(primary, True), (secondary, False)]
        )
        result = resolve_policy(measured_run, measured_build)
        self.assertEqual(result.policy_id, 1)
        self.assertEqual(
            result.semantic_oracle_sha256,
            C.REALLOC_ZERO_SEMANTIC_ORACLE_BINDING["sha256"],
        )

    def test_resolver_not_applicable_only_for_measured_eligible_run(self) -> None:
        measured_build = build(synthetic=False)
        primary = without_realloc(measured_build["entries"][0])
        measured_run = measured_run_for_population(
            measured_build, [(primary, True)]
        )
        self.assertEqual(
            resolve_policy(measured_run, measured_build),
            C.ReallocZeroPolicyResolution("not_applicable", None, None),
        )

        synthetic_build = build()
        synthetic_primary = without_realloc(synthetic_build["entries"][0])
        synthetic_build["entries"] = [synthetic_primary]
        synthetic_run = run(synthetic_build)
        self.assert_rejects(
            lambda: resolve_policy(synthetic_run, synthetic_build)
        )

    def test_resolver_rejects_other_retained_ineligible_run(self) -> None:
        measured_build = build(synthetic=False)
        measured_run = run(
            measured_build, synthetic=False, configuration_digest="3" * 64
        )
        measured_run["eligible_exact_domain"] = False
        measured_run["failure_ids"] = [34]
        measured_run["verdict_id"] = 10
        validate_run(
            measured_run,
            measured_build,
            expected_configuration_digest="4" * 64,
        )
        self.assert_rejects(
            lambda: resolve_policy(
                measured_run,
                measured_build,
                expected_configuration_digest="4" * 64,
            )
        )

    def test_versions_numeric_fields_and_definitions_reject_bool_or_float(self) -> None:
        for invalid in (True, False, 1.0, 0.0):
            measured_build = build(synthetic=False)
            measured_build["version"]["major"] = invalid
            self.assert_rejects(
                lambda measured_build=measured_build: C.validate_build_coverage(
                    measured_build, api_definition_sha256=API_H
                )
            )
        for field_path, invalid in (("version", True), ("process_generation", 7.0)):
            measured_build = build(synthetic=False)
            measured_run = run(measured_build, synthetic=False)
            if field_path == "version":
                measured_run["version"]["major"] = invalid
            else:
                measured_run[field_path] = invalid
            self.assert_rejects(
                lambda measured_run=measured_run, measured_build=measured_build: validate_run(
                    measured_run, measured_build
                )
            )
        for field, invalid in (("architecture_id", True), ("architecture_id", 1.0)):
            measured_build = build(synthetic=False)
            measured_build[field] = invalid
            self.assert_rejects(
                lambda measured_build=measured_build: C.validate_build_coverage(
                    measured_build, api_definition_sha256=API_H
                )
            )
        for definition, validator in (
            (C.COVERAGE_POLICY_DEFINITION, C.validate_coverage_policy),
            (C.COVERAGE_INSTANCE_SCHEMA_DEFINITION, C.validate_coverage_instance_schema),
        ):
            for invalid in (True, 1.0):
                mutant = copy.deepcopy(definition)
                mutant["version"]["major"] = invalid
                self.assert_rejects(lambda mutant=mutant, validator=validator: validator(mutant))


if __name__ == "__main__":
    unittest.main()
