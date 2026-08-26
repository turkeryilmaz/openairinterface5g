#!/usr/bin/env python3
"""Exact candidate v1 effective-configuration schema and instance validator.

This standard-library-only module performs no host discovery and no writes. Every
run-specific value is supplied by the caller; the constructor adds only static
schema metadata.  The schema is object type 11 in the frozen v1.5 bundle.
This module closes the kind-10 object itself, but that static registration
alone does not claim archive-level semantic admission.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence


INSTANCE_VERSION = {"major": 1, "minor": 0}
SCHEMA_VERSION = {"major": 1, "minor": 1}
# Backward-compatible public name for the external kind-10 instance version.
VERSION = INSTANCE_VERSION
CONFIGURATION_ID = "oai_memprof_effective_config"
CONFIGURATION_ARCHIVE_PATH = "metadata/effective-config.json"
SCHEMA_ARCHIVE_PATH = "definition/effective-config-schema-v1.json"
SCHEMA_OBJECT_TYPE = 11
EVENT_SEMANTICS_OBJECT_TYPE = 3
EVENT_SEMANTICS_ARCHIVE_PATH = "definition/event-semantics-v1.json"
EVENT_SEMANTICS_SHA256 = "8dbe428939592cdfc86ba8730078563672de4a13081ebdb868fa97f543dfab89"
OBJECT_KIND = 10
OBJECT_FLAGS = 0x07
FORMAT_ID = 1
SCHEMA_REVISION = 1
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1

_IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9_]{0,62}\Z", re.ASCII)
_RUN_ID_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z._-]{0,126}\Z", re.ASCII)
_SEED_RE = re.compile(r"[0-9a-f]{16}\Z", re.ASCII)

BUNDLE_MEMBER_PROPOSAL = {
    "name": "effective_config_schema",
    "object_type": SCHEMA_OBJECT_TYPE,
    "owner": "config",
    "path": SCHEMA_ARCHIVE_PATH,
}
BUNDLE_CROSS_RELATION_PROPOSAL = {
    "left": "metadata/effective-config.json.schema.sha256",
    "relation_id": "effective_configuration_schema",
    "right": "schema_bundle.entries[object_type=11].sha256",
}


class EffectiveConfigError(ValueError):
    """Deterministic schema, instance, or cross-binding rejection."""


def _load_semantic_codec() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "semantic"
        / "semantic_catalog_v1.py"
    )
    spec = importlib.util.spec_from_file_location(
        "oai_memprof_frozen_semantic_catalog_v1", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen semantic codec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SEMANTIC = _load_semantic_codec()
canonical_bytes = _SEMANTIC.canonical_bytes
parse_canonical = _SEMANTIC.parse_canonical
sha256_hex = _SEMANTIC.sha256_hex


MODE_CATALOG = (
    (2, "counters"),
    (3, "sampled"),
    (4, "exact_events"),
    (5, "exact_callsite"),
)
ROLE_CATALOG = ((1, "gnb"), (2, "nr_ue"))
SCOPE_CATALOG = (
    (1, "measurement_interval"),
    (2, "process_lifetime_reserved_incomplete_only"),
)
SAMPLE_SEED_PROVENANCE_CATALOG = (
    (1, "linux_getrandom_exactly_eight_bytes"),
    (2, "explicit_synthetic_fixture"),
    (20, "not_applicable"),
)
SAMPLE_SEED_STATUS_CATALOG = (
    (1, "acquired"),
    (2, "synthetic_fixture"),
    (20, "not_applicable"),
)


def _field(name: str, type_name: str, nullable: bool = False) -> dict[str, Any]:
    return {"name": name, "nullable": nullable, "type": type_name}


EFFECTIVE_CONFIG_SCHEMA_DEFINITION: dict[str, Any] = {
    "definition_id": "oai_memprof_effective_config_schema",
    "event_semantics": {
        "object_type": EVENT_SEMANTICS_OBJECT_TYPE,
        "path": EVENT_SEMANTICS_ARCHIVE_PATH,
        "sha256": EVENT_SEMANTICS_SHA256,
    },
    "field_types": [
        {
            "constraint": "absolute_nonroot_posix_utf8_nfc_1_to_4096_bytes_no_empty_dot_dotdot_backslash_or_control",
            "name": "host_path",
            "type": "json_string",
        },
        {
            "constraint": "[a-z][a-z0-9_]{0,62}",
            "name": "selection_key",
            "type": "json_string",
        },
        {
            "constraint": "utf8_nfc_1_to_4096_bytes",
            "name": "selection_value",
            "type": "json_string",
        },
        {
            "constraint": "[0-9A-Za-z][0-9A-Za-z._-]{0,126}",
            "name": "run_id",
            "type": "json_string",
        },
        {
            "constraint": "exactly_16_lowercase_hex_digits_or_null",
            "name": "sample_seed_hex",
            "type": "json_string_or_null",
        },
        {
            "constraint": "0_to_4294967295",
            "name": "u32",
            "type": "json_integer",
        },
        {
            "constraint": "1_to_4294967295",
            "name": "u32_nonzero",
            "type": "json_integer",
        },
        {
            "constraint": "1_to_4294967294",
            "name": "u32_capacity",
            "type": "json_integer",
        },
        {
            "constraint": "0_to_18446744073709551615",
            "name": "u64",
            "type": "json_integer",
        },
        {
            "constraint": "1_to_18446744073709551615",
            "name": "u64_nonzero",
            "type": "json_integer",
        },
    ],
    "field_semantics": [
        {
            "field": "flush_records",
            "meaning": "maximum_records_per_writer_payload_chunk",
            "rule": "record_threshold_chunk_has_exact_count_timer_or_final_chunk_may_be_partial",
            "unit": "records",
        },
        {
            "field": "flush_us",
            "meaning": "maximum_age_of_oldest_staged_record_before_partial_chunk_flush",
            "rule": "zero_disables_time_trigger_nonzero_is_elapsed_monotonic_duration",
            "unit": "microseconds",
        },
        {
            "field": "max_threads",
            "meaning": "preallocated_never_reused_producer_descriptor_capacity",
            "rule": "one_value_reserved_for_unregistered_sentinel",
            "unit": "producer_descriptors",
        },
        {
            "field": "ring_records",
            "meaning": "capacity_of_each_registered_producer_spsc_ring",
            "rule": "per_producer_not_process_total",
            "unit": "record_slots_per_producer",
        },
        {
            "field": "sample_seed_hex",
            "meaning": "K_as_exact_eight_bytes_and_equivalent_u64",
            "rule": "lowercase_percent_016x_numeric_big_endian_hex_bytes_first_byte_most_significant",
            "unit": "u64",
        },
        {
            "field": "sample_threshold",
            "meaning": "q_in_A03_predicate_U_less_than_q_where_U_equals_F_x_xor_K",
            "rule": "A03_requires_1_to_2_pow_64_minus_1_and_uses_q_over_2_pow_64_non_A03_requires_zero_exact_p_one_unrepresentable",
            "unit": "u64",
        },
        {
            "field": "table_entries",
            "meaning": "total_preallocated_membership_slots_across_all_shards",
            "rule": "process_total_not_per_shard",
            "unit": "membership_slots",
        },
        {
            "field": "table_probes",
            "meaning": "maximum_open_address_slots_examined_by_one_membership_operation",
            "rule": "per_operation_bound_not_per_shard_or_process_total",
            "unit": "probes_per_operation",
        },
    ],
    "mode_catalog": [
        {"mode_id": mode_id, "name": name} for mode_id, name in MODE_CATALOG
    ],
    "object_schema": {
        "entry_count": 1,
        "fields": [
            _field("catalog_id", "literal:oai_memprof_effective_config"),
            _field("flush_records", "u32_nonzero"),
            _field("flush_us", "u64"),
            _field("max_threads", "u32_capacity"),
            _field("mode_id", "enum:2_3_4_5"),
            _field("output_directory", "host_path"),
            _field("ring_records", "u64_nonzero"),
            _field("role_id", "enum:1_2"),
            _field("run_id", "run_id"),
            _field("sample_seed_hex", "sample_seed_hex", True),
            _field("sample_seed_provenance_id", "enum:1_2_20"),
            _field("sample_seed_status_id", "enum:1_2_20"),
            _field("sample_threshold", "u64"),
            _field(
                "schema",
                "definition_binding:self_object_type_11_path_definition_effective_config_schema_v1",
            ),
            _field("scope_kind", "enum:1_2"),
            _field("selection_values", "array:selection_value"),
            _field("table_entries", "u64_nonzero"),
            _field("table_probes", "u32_nonzero"),
            _field("version", "version_1_0"),
        ],
        "flags": OBJECT_FLAGS,
        "format_id": FORMAT_ID,
        "object_kind": OBJECT_KIND,
        "path": CONFIGURATION_ARCHIVE_PATH,
        "schema_revision": SCHEMA_REVISION,
        "unknown_fields_allowed": False,
    },
    "relational_rules": [
        "instance_schema_binding_equals_this_member",
        "mode_id_is_stream_capable",
        "output_directory_is_materialized_absolute_host_path",
        "run_coverage_configured_equals_role_and_build_module_selection",
        "sample_seed_big_endian_hex_equals_percent_016x_of_seed_u64",
        "sample_seed_provenance_status_pairs_are_1_1_2_2_or_20_20",
        "sample_seed_is_nonnull_exactly_for_mode_3",
        "sample_threshold_is_nonzero_exactly_for_mode_3",
        "sample_threshold_is_zero_outside_mode_3",
        "selection_values_are_key_sorted_unique",
        "table_probes_not_greater_than_table_entries",
    ],
    "role_catalog": [
        {"name": name, "role_id": role_id} for role_id, name in ROLE_CATALOG
    ],
    "sample_seed_provenance_catalog": [
        {"name": name, "provenance_id": provenance_id}
        for provenance_id, name in SAMPLE_SEED_PROVENANCE_CATALOG
    ],
    "sample_seed_status_catalog": [
        {"name": name, "status_id": status_id}
        for status_id, name in SAMPLE_SEED_STATUS_CATALOG
    ],
    "scope_catalog": [
        {"name": name, "scope_kind": scope_kind}
        for scope_kind, name in SCOPE_CATALOG
    ],
    "selection_value_schema": {
        "array_order": "key_ascii_ascending",
        "fields": [
            _field("key", "selection_key"),
            _field("value", "selection_value"),
        ],
        "uniqueness": "key",
        "unknown_fields_allowed": False,
    },
    "version": SCHEMA_VERSION,
}


SCHEMA_BYTES = canonical_bytes(EFFECTIVE_CONFIG_SCHEMA_DEFINITION)
SCHEMA_SHA256 = sha256_hex(SCHEMA_BYTES)
BUNDLE_ENTRY_PROPOSAL = {
    "bytes": len(SCHEMA_BYTES),
    "object_type": SCHEMA_OBJECT_TYPE,
    "path": SCHEMA_ARCHIVE_PATH,
    "sha256": SCHEMA_SHA256,
}


def _exact_keys(value: Any, keys: Iterable[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise EffectiveConfigError(f"{where}: object required")
    expected = tuple(sorted(keys))
    if tuple(value.keys()) != expected:
        raise EffectiveConfigError(f"{where}: exact keys {expected!r} required")
    return value


def _uint(value: Any, bits: int, where: str, *, nonzero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EffectiveConfigError(f"{where}: u{bits} integer required")
    lower = 1 if nonzero else 0
    upper = (1 << bits) - 1
    if not lower <= value <= upper:
        raise EffectiveConfigError(f"{where}: outside u{bits} range")
    return value


def _string(value: Any, where: str, *, maximum_bytes: int = 4096) -> str:
    if not isinstance(value, str) or not value:
        raise EffectiveConfigError(f"{where}: nonempty string required")
    try:
        _SEMANTIC._validate_string(value, where)
    except ValueError as error:
        raise EffectiveConfigError(str(error)) from error
    if len(value.encode("utf-8")) > maximum_bytes:
        raise EffectiveConfigError(f"{where}: exceeds {maximum_bytes} UTF-8 bytes")
    return value


def _hash(value: Any, where: str) -> str:
    digest = _string(value, where, maximum_bytes=64)
    if not re.fullmatch(r"[0-9a-f]{64}", digest, re.ASCII):
        raise EffectiveConfigError(f"{where}: SHA-256 lowercase hexadecimal required")
    return digest


def _host_path(value: Any, where: str) -> str:
    path = _string(value, where)
    if not path.startswith("/") or "\\" in path:
        raise EffectiveConfigError(f"{where}: absolute POSIX path required")
    components = path.split("/")[1:]
    if not components or any(component in ("", ".", "..") for component in components):
        raise EffectiveConfigError(f"{where}: canonical non-root absolute path required")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise EffectiveConfigError(f"{where}: control character forbidden")
    return value


def _version(value: Any, where: str) -> None:
    _exact_keys(value, ("major", "minor"), where)
    major = _uint(value["major"], 16, f"{where}.major")
    minor = _uint(value["minor"], 16, f"{where}.minor")
    if (major, minor) != (1, 0):
        raise EffectiveConfigError(f"{where}: exact version 1.0 required")


def _schema_binding(value: Any, where: str) -> None:
    _exact_keys(value, ("object_type", "path", "sha256"), where)
    object_type = _uint(value["object_type"], 16, f"{where}.object_type", nonzero=True)
    path = _string(value["path"], f"{where}.path")
    digest = _hash(value["sha256"], f"{where}.sha256")
    if (object_type, path, digest) != (
        SCHEMA_OBJECT_TYPE, SCHEMA_ARCHIVE_PATH, SCHEMA_SHA256
    ):
        raise EffectiveConfigError(f"{where}: exact proposed member-11 binding required")


def _selection_values(value: Any, where: str) -> None:
    if not isinstance(value, list):
        raise EffectiveConfigError(f"{where}: array required")
    keys: list[str] = []
    for index, row in enumerate(value):
        location = f"{where}[{index}]"
        _exact_keys(row, ("key", "value"), location)
        key = _string(row["key"], f"{location}.key", maximum_bytes=63)
        if not _IDENTIFIER_RE.fullmatch(key):
            raise EffectiveConfigError(f"{location}.key: selection-key grammar required")
        _string(row["value"], f"{location}.value")
        keys.append(key)
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise EffectiveConfigError(f"{where}: keys must be unique in ASCII order")


def validate_schema_definition(value: Mapping[str, Any]) -> None:
    """Require the exact in-code schema value, without accepting extensions."""

    try:
        encoded = canonical_bytes(value)
    except (TypeError, ValueError) as error:
        raise EffectiveConfigError(f"schema_definition: {error}") from error
    if encoded != SCHEMA_BYTES:
        raise EffectiveConfigError("schema_definition: exact frozen candidate differs")


def validate_effective_configuration(value: Mapping[str, Any]) -> None:
    """Validate one caller-materialized effective configuration fail closed."""

    _exact_keys(
        value,
        (
            "catalog_id",
            "flush_records",
            "flush_us",
            "max_threads",
            "mode_id",
            "output_directory",
            "ring_records",
            "role_id",
            "run_id",
            "sample_seed_hex",
            "sample_seed_provenance_id",
            "sample_seed_status_id",
            "sample_threshold",
            "schema",
            "scope_kind",
            "selection_values",
            "table_entries",
            "table_probes",
            "version",
        ),
        "effective_config",
    )
    if value["catalog_id"] != CONFIGURATION_ID:
        raise EffectiveConfigError("effective_config.catalog_id: exact ID required")
    _version(value["version"], "effective_config.version")
    _schema_binding(value["schema"], "effective_config.schema")
    _uint(value["flush_records"], 32, "effective_config.flush_records", nonzero=True)
    _uint(value["flush_us"], 64, "effective_config.flush_us")
    max_threads = _uint(
        value["max_threads"], 32, "effective_config.max_threads", nonzero=True
    )
    if max_threads == UINT32_MAX:
        raise EffectiveConfigError(
            "effective_config.max_threads: UINT32_MAX is reserved as sentinel"
        )
    mode_id = _uint(value["mode_id"], 16, "effective_config.mode_id", nonzero=True)
    if mode_id not in dict(MODE_CATALOG):
        raise EffectiveConfigError("effective_config.mode_id: stream-capable mode 2..5 required")
    _host_path(value["output_directory"], "effective_config.output_directory")
    _uint(value["ring_records"], 64, "effective_config.ring_records", nonzero=True)
    role_id = _uint(value["role_id"], 16, "effective_config.role_id", nonzero=True)
    if role_id not in dict(ROLE_CATALOG):
        raise EffectiveConfigError("effective_config.role_id: known role required")
    run_id = _string(value["run_id"], "effective_config.run_id", maximum_bytes=127)
    if not _RUN_ID_RE.fullmatch(run_id):
        raise EffectiveConfigError("effective_config.run_id: canonical run-ID grammar required")
    threshold = _uint(value["sample_threshold"], 64, "effective_config.sample_threshold")
    seed = value["sample_seed_hex"]
    provenance_id = _uint(
        value["sample_seed_provenance_id"],
        16,
        "effective_config.sample_seed_provenance_id",
        nonzero=True,
    )
    status_id = _uint(
        value["sample_seed_status_id"],
        16,
        "effective_config.sample_seed_status_id",
        nonzero=True,
    )
    if provenance_id not in dict(SAMPLE_SEED_PROVENANCE_CATALOG):
        raise EffectiveConfigError(
            "effective_config.sample_seed_provenance_id: unknown provenance"
        )
    if status_id not in dict(SAMPLE_SEED_STATUS_CATALOG):
        raise EffectiveConfigError("effective_config.sample_seed_status_id: unknown status")
    if mode_id == 3:
        if not isinstance(seed, str) or not _SEED_RE.fullmatch(seed):
            raise EffectiveConfigError(
                "effective_config.sample_seed_hex: mode 3 requires exact 64-bit lowercase hex"
            )
        if (provenance_id, status_id) not in ((1, 1), (2, 2)):
            raise EffectiveConfigError(
                "effective_config: mode 3 seed provenance/status must be 1/1 or 2/2"
            )
        if threshold == 0:
            raise EffectiveConfigError(
                "effective_config.sample_threshold: mode 3 requires q in 1..2^64-1"
            )
    elif seed is not None or threshold != 0 or (provenance_id, status_id) != (20, 20):
        raise EffectiveConfigError(
            "effective_config: non-sampled mode requires null seed, zero threshold, and 20/20 provenance/status"
        )
    scope_kind = _uint(value["scope_kind"], 16, "effective_config.scope_kind", nonzero=True)
    if scope_kind not in dict(SCOPE_CATALOG):
        raise EffectiveConfigError("effective_config.scope_kind: known scope required")
    _selection_values(value["selection_values"], "effective_config.selection_values")
    table_entries = _uint(
        value["table_entries"], 64, "effective_config.table_entries", nonzero=True
    )
    table_probes = _uint(
        value["table_probes"], 32, "effective_config.table_probes", nonzero=True
    )
    if table_probes > table_entries:
        raise EffectiveConfigError(
            "effective_config.table_probes: cannot exceed table_entries"
        )


def validate_effective_configuration_bytes(raw: bytes) -> dict[str, Any]:
    """Parse exact format-1 bytes and validate the kind-10 instance."""

    try:
        value = parse_canonical(raw)
    except ValueError as error:
        raise EffectiveConfigError(str(error)) from error
    validate_effective_configuration(value)
    return value


def make_effective_configuration(
    *,
    flush_records: int,
    flush_us: int,
    max_threads: int,
    mode_id: int,
    output_directory: str,
    ring_records: int,
    role_id: int,
    run_id: str,
    sample_seed_hex: str | None,
    sample_seed_provenance_id: int,
    sample_seed_status_id: int,
    sample_threshold: int,
    scope_kind: int,
    selection_values: Sequence[Mapping[str, str]],
    table_entries: int,
    table_probes: int,
) -> dict[str, Any]:
    """Construct from explicit values only; no defaulting or observation occurs."""

    copied_selection_values: list[dict[str, Any]] = []
    for index, row in enumerate(selection_values):
        if not isinstance(row, Mapping):
            raise EffectiveConfigError(
                f"effective_config.selection_values[{index}]: object required"
            )
        _exact_keys(row, ("key", "value"), f"effective_config.selection_values[{index}]")
        copied_selection_values.append({"key": row["key"], "value": row["value"]})
    value = {
        "catalog_id": CONFIGURATION_ID,
        "flush_records": flush_records,
        "flush_us": flush_us,
        "max_threads": max_threads,
        "mode_id": mode_id,
        "output_directory": output_directory,
        "ring_records": ring_records,
        "role_id": role_id,
        "run_id": run_id,
        "sample_seed_hex": sample_seed_hex,
        "sample_seed_provenance_id": sample_seed_provenance_id,
        "sample_seed_status_id": sample_seed_status_id,
        "sample_threshold": sample_threshold,
        "schema": {
            "object_type": SCHEMA_OBJECT_TYPE,
            "path": SCHEMA_ARCHIVE_PATH,
            "sha256": SCHEMA_SHA256,
        },
        "scope_kind": scope_kind,
        "selection_values": copied_selection_values,
        "table_entries": table_entries,
        "table_probes": table_probes,
        "version": dict(INSTANCE_VERSION),
    }
    validate_effective_configuration(value)
    return value


def serialize_effective_configuration(value: Mapping[str, Any]) -> bytes:
    validate_effective_configuration(value)
    return canonical_bytes(value)


def configuration_sha256(value: Mapping[str, Any]) -> str:
    return sha256_hex(serialize_effective_configuration(value))


def wire_object_binding_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return exact kwargs for Artifact10's 64-byte ObjectBindingEntry."""

    raw = serialize_effective_configuration(value)
    return {
        "byte_count": len(raw),
        "entry_count": 1,
        "object_flags": OBJECT_FLAGS,
        "format_id": FORMAT_ID,
        "object_kind": OBJECT_KIND,
        "schema_revision": SCHEMA_REVISION,
        "sha256": bytes.fromhex(sha256_hex(raw)),
    }


def validate_sample_seed_admissibility(value: Mapping[str, Any]) -> None:
    """Reject A03 synthetic-fixture seeds at the scientific admission boundary."""

    validate_effective_configuration(value)
    if value["mode_id"] == 3 and (
        value["sample_seed_provenance_id"], value["sample_seed_status_id"]
    ) != (1, 1):
        raise EffectiveConfigError(
            "effective_config.sample_seed: scientific A03 admission requires measured getrandom provenance"
        )


def validate_pre_active_bindings(
    value: Mapping[str, Any],
    *,
    opening_configuration_sha256: str,
    opening_configured_thread_capacity: int,
    opening_role_kind: int,
    opening_scope_kind: int,
    run_coverage: Mapping[str, Any] | None = None,
) -> None:
    """Bind exact configuration identity to opening and optional coverage rows.

    The caller supplies already decoded/trusted structures.  This function does
    not discover or infer a host, role, module population, or run outcome.
    """

    validate_effective_configuration(value)
    digest = configuration_sha256(value)
    opening_digest = _hash(
        opening_configuration_sha256, "opening.configuration_instance_sha256"
    )
    if opening_digest != digest:
        raise EffectiveConfigError("opening.configuration_instance_sha256: digest mismatch")
    opening_capacity = _uint(
        opening_configured_thread_capacity, 32, "opening.configured_thread_capacity",
        nonzero=True,
    )
    if opening_capacity != value["max_threads"]:
        raise EffectiveConfigError("opening.configured_thread_capacity: configuration mismatch")
    opening_role = _uint(opening_role_kind, 16, "opening.role_kind", nonzero=True)
    if opening_role != value["role_id"]:
        raise EffectiveConfigError("opening.role_kind: configuration mismatch")
    opening_scope = _uint(opening_scope_kind, 16, "opening.scope_kind", nonzero=True)
    if opening_scope != value["scope_kind"]:
        raise EffectiveConfigError("opening.scope_kind: configuration mismatch")
    if run_coverage is not None:
        if not isinstance(run_coverage, Mapping):
            raise EffectiveConfigError("run_coverage: object required")
        run_digest = _hash(
            run_coverage.get("configuration_instance_sha256"),
            "run_coverage.configuration_instance_sha256",
        )
        if run_digest != digest:
            raise EffectiveConfigError("run_coverage.configuration_instance_sha256: mismatch")
        run_role = _uint(
            run_coverage.get("role_id"), 16, "run_coverage.role_id", nonzero=True
        )
        if run_role != value["role_id"]:
            raise EffectiveConfigError("run_coverage.role_id: configuration mismatch")


def _module_selected(
    selection: Mapping[str, Any],
    *,
    configuration_values: Mapping[str, str],
    role_name: str,
    where: str,
) -> bool:
    _exact_keys(selection, ("operator_id", "predicates"), where)
    operator_id = _uint(selection["operator_id"], 16, f"{where}.operator_id", nonzero=True)
    if operator_id not in (1, 2):
        raise EffectiveConfigError(f"{where}.operator_id: exact all/any operator required")
    predicates = selection["predicates"]
    if not isinstance(predicates, list) or not predicates:
        raise EffectiveConfigError(f"{where}.predicates: nonempty array required")
    results: list[bool] = []
    ordering: list[tuple[int, str, str]] = []
    for index, predicate in enumerate(predicates):
        location = f"{where}.predicates[{index}]"
        _exact_keys(
            predicate,
            ("configuration_key", "expected_value", "predicate_id"),
            location,
        )
        predicate_id = _uint(
            predicate["predicate_id"], 16, f"{location}.predicate_id", nonzero=True
        )
        key = predicate["configuration_key"]
        expected = predicate["expected_value"]
        if predicate_id == 1:
            if key is not None or expected is not None:
                raise EffectiveConfigError(f"{location}: always forbids operands")
            results.append(True)
            ordering.append((1, "", ""))
        elif predicate_id == 2:
            if not isinstance(key, str) or not _IDENTIFIER_RE.fullmatch(key):
                raise EffectiveConfigError(f"{location}.configuration_key: grammar mismatch")
            expected = _string(expected, f"{location}.expected_value", maximum_bytes=255)
            if key not in configuration_values:
                raise EffectiveConfigError(
                    f"effective_config.selection_values: missing required key {key!r}"
                )
            results.append(configuration_values[key] == expected)
            ordering.append((2, key, expected))
        elif predicate_id == 3:
            if key is not None or expected not in ("gnb", "nr_ue"):
                raise EffectiveConfigError(f"{location}: role_equals operands mismatch")
            results.append(role_name == expected)
            ordering.append((3, "", expected))
        else:
            raise EffectiveConfigError(f"{location}.predicate_id: unknown predicate")
    if ordering != sorted(ordering) or len(ordering) != len(set(ordering)):
        raise EffectiveConfigError(f"{where}.predicates: order/uniqueness mismatch")
    return all(results) if operator_id == 1 else any(results)


def validate_module_selection_bindings(
    value: Mapping[str, Any],
    *,
    build_coverage: Mapping[str, Any],
    run_coverage: Mapping[str, Any],
) -> None:
    """Reconcile effective selection inputs with each build-domain run row.

    This complements, but does not replace, the accepted coverage validators.
    Callers must first validate both coverage objects with that module.  Here
    missing selection inputs reject instead of being interpreted as false.
    """

    validate_effective_configuration(value)
    digest = configuration_sha256(value)
    if not isinstance(build_coverage, Mapping):
        raise EffectiveConfigError("build_coverage: object required")
    if not isinstance(run_coverage, Mapping):
        raise EffectiveConfigError("run_coverage: object required")
    run_digest = _hash(
        run_coverage.get("configuration_instance_sha256"),
        "run_coverage.configuration_instance_sha256",
    )
    if run_digest != digest:
        raise EffectiveConfigError("run_coverage.configuration_instance_sha256: mismatch")
    run_role = _uint(
        run_coverage.get("role_id"), 16, "run_coverage.role_id", nonzero=True
    )
    if run_role != value["role_id"]:
        raise EffectiveConfigError("run_coverage.role_id: configuration mismatch")
    build_rows = build_coverage.get("entries")
    population = run_coverage.get("module_population")
    if not isinstance(build_rows, list) or not build_rows:
        raise EffectiveConfigError("build_coverage.entries: nonempty array required")
    if not isinstance(population, list):
        raise EffectiveConfigError("run_coverage.module_population: array required")
    run_by_build_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(population):
        if not isinstance(row, dict):
            raise EffectiveConfigError(f"run_coverage.module_population[{index}]: object required")
        logical_id = row.get("build_logical_id")
        if logical_id is None:
            continue
        if not isinstance(logical_id, str) or logical_id in run_by_build_id:
            raise EffectiveConfigError(
                "run_coverage.module_population: unique build_logical_id required"
            )
        run_by_build_id[logical_id] = row
    config_values = {
        row["key"]: row["value"] for row in value["selection_values"]
    }
    role_name = dict(ROLE_CATALOG)[value["role_id"]]
    for index, build_row in enumerate(build_rows):
        if not isinstance(build_row, dict):
            raise EffectiveConfigError(f"build_coverage.entries[{index}]: object required")
        logical_id = build_row.get("logical_id")
        if not isinstance(logical_id, str) or logical_id not in run_by_build_id:
            raise EffectiveConfigError(
                f"run_coverage.module_population: missing build row {logical_id!r}"
            )
        role_ids = build_row.get("role_ids")
        if not isinstance(role_ids, list):
            raise EffectiveConfigError(f"build_coverage.entries[{index}].role_ids: array required")
        selected = value["role_id"] in role_ids and _module_selected(
            build_row.get("module_selection"),
            configuration_values=config_values,
            role_name=role_name,
            where=f"build_coverage.entries[{index}].module_selection",
        )
        if run_by_build_id[logical_id].get("configured") is not selected:
            raise EffectiveConfigError(
                f"run_coverage.module_population[{logical_id!r}].configured: derived value differs"
            )


def validate_literal_file(path: Path | None = None) -> None:
    literal_path = path or Path(__file__).resolve().parent / "archive" / SCHEMA_ARCHIVE_PATH
    raw = literal_path.read_bytes()
    if raw != SCHEMA_BYTES:
        raise EffectiveConfigError("schema_literal: bytes differ from exact in-code definition")
    parsed = parse_canonical(raw)
    validate_schema_definition(parsed)


if __name__ == "__main__":
    validate_literal_file()
    print(f"schema_bytes={len(SCHEMA_BYTES)}")
    print(f"schema_sha256={SCHEMA_SHA256}")
