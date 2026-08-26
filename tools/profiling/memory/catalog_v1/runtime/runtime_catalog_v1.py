#!/usr/bin/env python3
"""Bundle-owned schema-v1 runtime catalog definitions and pure validators.

The module freezes semantic member 13 for external catalog object kinds 5, 6,
and 7.  It validates immutable canonical bytes only; host discovery and catalog
production belong to the tracked runtime implementation that consumes this
contract.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple, Sequence


VERSION = {"major": 1, "minor": 0}
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
MAX_THREAD_INDEX = UINT32_MAX - 1

POPULATION_COMPLETE = 1
POPULATION_NEGATIVE_PARTIAL = 2
LIFECYCLE_COMPLETE = 5
LIFECYCLE_FAILED = 6
LIFECYCLE_INCOMPLETE = 7

RUNTIME_DEFINITION_OBJECT_TYPE = 13
RUNTIME_DEFINITION_PATH = "definition/runtime-catalog-schema-v1.json"
THREAD_CATALOG_PATH = "catalog/thread.json"
MODULE_CATALOG_PATH = "catalog/module.json"
CLOCK_CATALOG_PATH = "catalog/clock.json"

THREAD_ROW_FIELDS = (
    "process_generation",
    "registration_ordinal",
    "thread_index",
)
MODULE_SEGMENT_FIELDS = (
    "end_address",
    "file_offset",
    "permissions",
    "start_address",
)
MODULE_ROW_FIELDS = (
    "build_id",
    "build_logical_id",
    "byte_count",
    "device",
    "inode",
    "load_state_id",
    "loaded_path",
    "logical_id",
    "module_generation",
    "module_id",
    "module_map_sha256",
    "namespace_id",
    "process_generation",
    "segments",
    "sha256",
)
CLOCK_SAMPLE_FIELDS = (
    "counter",
    "monotonic_raw_after_ns",
    "monotonic_raw_before_ns",
    "realtime_unix_ns",
    "sample_ordinal",
)
CLOCK_ROW_FIELDS = (
    "acquisition_source_id",
    "acquisition_status_id",
    "architecture_id",
    "calibration_error_bound_ns",
    "calibration_kind",
    "calibration_span_ns",
    "clock_kind",
    "counter_frequency_denominator",
    "counter_frequency_numerator",
    "counter_invalid_observed",
    "counter_stability_status_id",
    "observed_max_error_ns",
    "process_generation",
    "realtime_discontinuity_observed",
    "samples",
    "start_counter",
    "start_monotonic_raw_ns",
    "start_realtime_unix_ns",
)
OPENING_CLOCK_FIELDS = (
    "calibration_error_bound_ns",
    "calibration_kind",
    "calibration_span_ns",
    "clock_kind",
    "counter_frequency_denominator",
    "counter_frequency_numerator",
    "process_generation",
    "start_counter",
    "start_monotonic_raw_ns",
    "start_realtime_unix_ns",
)

_BUILD_ID_RE = re.compile(r"(?:[0-9a-f]{2}){1,64}\Z", re.ASCII)
_PERMISSION_RE = re.compile(r"[r-][w-][x-][ps]\Z", re.ASCII)


class RuntimeCatalogError(ValueError):
    """Deterministic runtime definition, instance, or relation rejection."""


class ThreadPopulation(NamedTuple):
    """Validated observed thread history and its completeness state."""

    thread_keys: frozenset[tuple[int, int]]
    ready_thread_indices: tuple[int, ...]
    population_complete: bool
    record_count: int


def _load_semantic() -> Any:
    path = Path(__file__).resolve().parent.parent / "semantic" / "semantic_catalog_v1.py"
    spec = importlib.util.spec_from_file_location("_oai_memprof_semantic_for_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeCatalogError("semantic: adjacent validator unavailable")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise RuntimeCatalogError("semantic: adjacent validator failed to load") from error
    return module


SEMANTIC = _load_semantic()


def _field(name: str, value_type: str, *, nullable: bool = False) -> dict[str, Any]:
    return {"name": name, "nullable": nullable, "type": value_type}


RUNTIME_DEFINITION = {
    "catalog_schemas": [
        {
            "array_order": "registration_ordinal_strict_ascending",
            "catalog_id": "oai_memprof_thread",
            "external_path": THREAD_CATALOG_PATH,
            "object_flags": 0x1B,
            "object_kind": 5,
            "row_fields": [
                _field("process_generation", "u64_nonzero"),
                _field("registration_ordinal", "u64_nonzero"),
                _field("thread_index", "thread_index"),
            ],
            "rules": [
                "complete_population_ordinals_are_exactly_1_through_entry_count",
                "negative_partial_population_retains_sorted_observed_ordinals_and_reports_unresolved_records",
                "thread_index_unique_and_never_reused_in_one_process_generation",
                "every_complete_stream_record_thread_index_resolves_exactly_once",
                "entry_count_not_greater_than_configured_thread_capacity",
            ],
            "unknown_fields_allowed": False,
        },
        {
            "array_order": "logical_id_ascii_ascending_with_module_id_equal_one_based_position",
            "catalog_id": "oai_memprof_module",
            "external_path": MODULE_CATALOG_PATH,
            "object_flags": 0x13,
            "object_kind": 6,
            "row_fields": [
                _field("build_id", "build_id_hex"),
                _field("build_logical_id", "identifier", nullable=True),
                _field("byte_count", "u64_nonzero"),
                _field("device", "u64"),
                _field("inode", "u64_nonzero"),
                _field("load_state_id", "loaded_state"),
                _field("loaded_path", "absolute_host_path"),
                _field("logical_id", "identifier"),
                _field("module_generation", "u64_nonzero"),
                _field("module_id", "u32_nonzero"),
                _field("module_map_sha256", "sha256_nonzero"),
                _field("namespace_id", "literal:0"),
                _field("process_generation", "u64_nonzero"),
                _field("segments", "array:module_segment"),
                _field("sha256", "sha256_nonzero"),
            ],
            "rules": [
                "v1_supports_only_base_loader_namespace_zero",
                "module_generation_equals_process_generation_pre_active_snapshot",
                "logical_id_unique_and_every_nonnull_build_logical_id_unique",
                "load_state_1_or_21_requires_build_logical_id_equal_logical_id",
                "load_state_20_requires_null_build_logical_id",
                "module_map_sha256_is_sha256_of_canonical_module_map_projection",
                "observed_run_population_reconciles_one_to_one_including_unexpected_rows",
                "active_time_changed_state_21_requires_negative_partial_terminal",
            ],
            "unknown_fields_allowed": False,
        },
        {
            "array_order": "exactly_one_opening_bound_record_clock",
            "catalog_id": "oai_memprof_clock",
            "external_path": CLOCK_CATALOG_PATH,
            "object_flags": 0x03,
            "object_kind": 7,
            "row_fields": [
                _field("acquisition_source_id", "clock_acquisition_source"),
                _field("acquisition_status_id", "literal:1"),
                _field("architecture_id", "architecture"),
                _field("calibration_error_bound_ns", "u64"),
                _field("calibration_kind", "calibration_kind"),
                _field("calibration_span_ns", "u64"),
                _field("clock_kind", "record_clock_kind"),
                _field("counter_frequency_denominator", "u64_nonzero"),
                _field("counter_frequency_numerator", "u64_nonzero"),
                _field("counter_invalid_observed", "boolean"),
                _field("counter_stability_status_id", "literal:1"),
                _field("observed_max_error_ns", "u64"),
                _field("process_generation", "u64_nonzero"),
                _field("realtime_discontinuity_observed", "boolean"),
                _field("samples", "array:clock_sample"),
                _field("start_counter", "u64"),
                _field("start_monotonic_raw_ns", "u64"),
                _field("start_realtime_unix_ns", "u64"),
            ],
            "rules": [
                "at_least_two_ordered_bracketed_samples",
                "first_sample_binds_opening_anchor_and_last_sample_binds_terminal_anchor",
                "counters_strictly_increase_and_monotonic_brackets_do_not_overlap",
                "declared_calibration_error_bound_covers_every_transform_to_bracket_endpoint_distance",
                "observed_max_error_equals_maximum_transform_distance_outside_bracket",
                "v1_semantic_admission_requires_exact_architectural_rate_and_zero_span",
                "realtime_without_discontinuity_tracks_monotonic_anchor_within_error_bound",
            ],
            "unknown_fields_allowed": False,
        },
    ],
    "clock_acquisition_sources": [
        {"architecture_id": 1, "calibration_kind": 1, "name": "x86_cpuid_15_exact", "source_id": 1},
        {"architecture_id": 2, "calibration_kind": 1, "name": "aarch64_cntfrq_el0_exact", "source_id": 2},
    ],
    "definition_id": "oai_memprof_runtime_catalog_schema",
    "field_types": [
        {"constraint": "absolute_posix_utf8_nfc", "encoding": "json_string", "name": "absolute_host_path"},
        {"constraint": "(?:[0-9a-f]{2}){1,64}_nonzero", "encoding": "json_string", "name": "build_id_hex"},
        {"constraint": "exact_true_or_false", "encoding": "json_boolean", "name": "boolean"},
        {"constraint": "[a-z][a-z0-9_]*", "encoding": "json_string", "name": "identifier"},
        {"constraint": "[r-][w-][x-][ps]", "encoding": "json_string", "name": "map_permissions"},
        {"constraint": "[0-9a-f]{64}_nonzero", "encoding": "json_string", "name": "sha256_nonzero"},
        {"constraint": "1_to_4294967294", "encoding": "json_integer", "name": "thread_index"},
        {"constraint": "0_to_18446744073709551615", "encoding": "json_integer", "name": "u64"},
        {"constraint": "1_to_18446744073709551615", "encoding": "json_integer", "name": "u64_nonzero"},
    ],
    "module_map_digest": {
        "algorithm": "sha256",
        "canonical_domain": "canonical_json_including_sole_final_lf",
        "projection_fields": ["device", "inode", "loaded_path", "namespace_id", "segments"],
        "segment_fields": list(MODULE_SEGMENT_FIELDS),
        "segment_order": "strict_start_address_ascending_nonoverlapping",
    },
    "population_states": [
        {"name": "complete", "population_state_id": POPULATION_COMPLETE},
        {"name": "negative_partial", "population_state_id": POPULATION_NEGATIVE_PARTIAL},
    ],
    "version": VERSION,
}


def definition_bytes() -> bytes:
    return SEMANTIC.canonical_bytes(RUNTIME_DEFINITION)


DEFINITION_BYTES = definition_bytes()
DEFINITION_SHA256 = hashlib.sha256(DEFINITION_BYTES).hexdigest()
BUNDLE_MEMBER_PROPOSAL = {
    "name": "runtime_catalog_schema",
    "object_type": RUNTIME_DEFINITION_OBJECT_TYPE,
    "owner": "runtime",
    "path": RUNTIME_DEFINITION_PATH,
}
BUNDLE_ENTRY_PROPOSAL = {
    "bytes": len(DEFINITION_BYTES),
    "object_type": RUNTIME_DEFINITION_OBJECT_TYPE,
    "path": RUNTIME_DEFINITION_PATH,
    "sha256": DEFINITION_SHA256,
}
THREAD_SCHEMA = {
    "object_type": RUNTIME_DEFINITION_OBJECT_TYPE,
    "path": RUNTIME_DEFINITION_PATH,
    "sha256": DEFINITION_SHA256,
}
MODULE_SCHEMA = dict(THREAD_SCHEMA)
CLOCK_SCHEMA = dict(THREAD_SCHEMA)
BUNDLE_CROSS_RELATION_PROPOSALS = (
    {
        "left": "catalog/clock.json.schema.sha256",
        "relation_id": "clock_catalog_schema",
        "right": "schema_bundle.entries[object_type=13].sha256",
    },
    {
        "left": "catalog/module.json.schema.sha256",
        "relation_id": "module_catalog_schema",
        "right": "schema_bundle.entries[object_type=13].sha256",
    },
    {
        "left": "catalog/thread.json.schema.sha256",
        "relation_id": "thread_catalog_schema",
        "right": "schema_bundle.entries[object_type=13].sha256",
    },
)


def _fail(where: str, detail: str) -> None:
    raise RuntimeCatalogError(f"{where}: {detail}")


def _exact_keys(value: Any, keys: Sequence[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(where, "object required")
    if len(value) != len(keys) or set(value) != set(keys):
        _fail(where, f"exact fields {tuple(sorted(keys))!r} required")
    return value


def _uint(value: Any, bits: int, where: str, *, nonzero: bool = False) -> int:
    if type(value) is not int:
        _fail(where, f"u{bits} integer required")
    if not (1 if nonzero else 0) <= value <= (1 << bits) - 1:
        _fail(where, f"outside u{bits} range")
    return value


def _boolean(value: Any, where: str) -> bool:
    if type(value) is not bool:
        _fail(where, "boolean required")
    return value


def _version(value: Any, where: str) -> None:
    row = _exact_keys(value, ("major", "minor"), where)
    if type(row["major"]) is not int or type(row["minor"]) is not int or row != VERSION:
        _fail(where, "exact typed version 1.0 required")


def _schema_binding(value: Any, where: str) -> None:
    row = _exact_keys(value, ("object_type", "path", "sha256"), where)
    if _uint(row["object_type"], 16, f"{where}.object_type", nonzero=True) != RUNTIME_DEFINITION_OBJECT_TYPE:
        _fail(f"{where}.object_type", "exact runtime member object type required")
    if row["path"] != RUNTIME_DEFINITION_PATH:
        _fail(f"{where}.path", "exact runtime definition path required")
    if _nonzero_hash(row["sha256"], f"{where}.sha256") != DEFINITION_SHA256:
        _fail(f"{where}.sha256", "exact runtime definition digest required")


def _catalog(value: Any, *, catalog_id: str, where: str) -> list[Any]:
    top = _exact_keys(value, ("catalog_id", "entries", "schema", "version"), where)
    if top["catalog_id"] != catalog_id:
        _fail(f"{where}.catalog_id", f"exact ID {catalog_id!r} required")
    _version(top["version"], f"{where}.version")
    _schema_binding(top["schema"], f"{where}.schema")
    if not isinstance(top["entries"], list):
        _fail(f"{where}.entries", "array required")
    return top["entries"]


def _population_state(population_state: Any, lifecycle_state: Any, where: str) -> int:
    state = _uint(population_state, 8, f"{where}.population_state", nonzero=True)
    lifecycle = _uint(lifecycle_state, 16, f"{where}.lifecycle_state", nonzero=True)
    if lifecycle == LIFECYCLE_COMPLETE:
        if state != POPULATION_COMPLETE:
            _fail(where, "COMPLETE lifecycle requires complete population state 1")
    elif lifecycle in (LIFECYCLE_FAILED, LIFECYCLE_INCOMPLETE):
        if state != POPULATION_NEGATIVE_PARTIAL:
            _fail(where, "negative terminal requires negative-partial population state 2")
    else:
        _fail(f"{where}.lifecycle_state", "terminal lifecycle 5, 6, or 7 required")
    return state


def _record_field(record: Any, name: str, where: str) -> Any:
    if isinstance(record, Mapping):
        if name not in record:
            _fail(where, f"{name} field required")
        return record[name]
    try:
        return getattr(record, name)
    except AttributeError as error:
        raise RuntimeCatalogError(f"{where}: {name} field required") from error


def _identifier(value: Any, where: str) -> str:
    try:
        return SEMANTIC._identifier(value, where)
    except Exception as error:
        raise RuntimeCatalogError(str(error)) from error


def _absolute_path(value: Any, where: str) -> str:
    try:
        return SEMANTIC._absolute_path(value, where)
    except Exception as error:
        raise RuntimeCatalogError(str(error)) from error


def _nonzero_hash(value: Any, where: str) -> str:
    try:
        result = SEMANTIC._hash(value, where)
    except Exception as error:
        raise RuntimeCatalogError(str(error)) from error
    if not int(result, 16):
        _fail(where, "all-zero SHA-256 is forbidden")
    return result


def _build_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _BUILD_ID_RE.fullmatch(value):
        _fail(where, "1..64 lowercase hexadecimal Build-ID bytes required")
    if not int(value, 16):
        _fail(where, "all-zero Build-ID descriptor forbidden")
    return value


def _require_exact_literal(value: Any, expected: Any, where: str) -> None:
    if type(value) is not type(expected):
        _fail(where, "exact type differs")
    if isinstance(expected, dict):
        if set(value) != set(expected):
            _fail(where, "exact keys differ")
        for key in expected:
            _require_exact_literal(value[key], expected[key], f"{where}.{key}")
    elif isinstance(expected, list):
        if len(value) != len(expected):
            _fail(where, "exact array length differs")
        for index, (actual, wanted) in enumerate(zip(value, expected)):
            _require_exact_literal(actual, wanted, f"{where}[{index}]")
    elif value != expected:
        _fail(where, "exact value differs")


def validate_definition(value: Mapping[str, Any]) -> None:
    _require_exact_literal(value, RUNTIME_DEFINITION, "runtime_definition")


def validate_definition_bytes(raw: bytes) -> dict[str, Any]:
    if not isinstance(raw, bytes):
        _fail("runtime_definition", "immutable bytes required")
    try:
        value = SEMANTIC.parse_canonical(raw)
    except Exception as error:
        raise RuntimeCatalogError(str(error)) from error
    if raw != DEFINITION_BYTES:
        _fail("runtime_definition", "literal/module bytes differ")
    validate_definition(value)
    return value


def validate_thread_catalog(
    value: Mapping[str, Any],
    *,
    expected_process_generation: int,
    configured_thread_capacity: int,
    record_count: int,
    mode_id: int,
    population_state: int,
    lifecycle_state: int,
) -> ThreadPopulation:
    """Validate retained registration history and return observed population."""

    entries = _catalog(value, catalog_id="oai_memprof_thread", where="thread_catalog")
    generation = _uint(expected_process_generation, 64, "expected_process_generation", nonzero=True)
    capacity = _uint(configured_thread_capacity, 32, "configured_thread_capacity", nonzero=True)
    count = _uint(record_count, 64, "record_count")
    mode = _uint(mode_id, 16, "mode_id")
    if mode > 6:
        _fail("mode_id", "outside frozen mode catalog 0..6")
    if mode in (0, 1, 2, 6) and count:
        _fail("record_count", "selected mode emits no first-slice records")
    state = _population_state(population_state, lifecycle_state, "thread_catalog")
    if len(entries) > capacity:
        _fail("thread_catalog.entries", "row count exceeds configured thread capacity")

    keys: set[tuple[int, int]] = set()
    indices: list[int] = []
    previous_ordinal = 0
    for index, value_row in enumerate(entries):
        where = f"thread_catalog.entries[{index}]"
        row = _exact_keys(value_row, THREAD_ROW_FIELDS, where)
        row_generation = _uint(row["process_generation"], 64, f"{where}.process_generation", nonzero=True)
        ordinal = _uint(row["registration_ordinal"], 64, f"{where}.registration_ordinal", nonzero=True)
        thread_index = _uint(row["thread_index"], 32, f"{where}.thread_index", nonzero=True)
        if thread_index > MAX_THREAD_INDEX:
            _fail(f"{where}.thread_index", "reserved 0xffffffff is forbidden")
        if thread_index > capacity:
            _fail(f"{where}.thread_index", "exceeds configured thread capacity")
        if row_generation != generation:
            _fail(f"{where}.process_generation", "active process-generation mismatch")
        if ordinal <= previous_ordinal:
            _fail("thread_catalog.entries", "strict registration-ordinal order required")
        if state == POPULATION_COMPLETE and ordinal != index + 1:
            _fail("thread_catalog.entries", "complete registration ordinals must be contiguous from one")
        key = (row_generation, thread_index)
        if key in keys:
            _fail(where, "thread index reuse within process generation is forbidden")
        previous_ordinal = ordinal
        keys.add(key)
        indices.append(thread_index)
    if state == POPULATION_COMPLETE and indices != sorted(indices):
        _fail("thread_catalog.entries", "complete population must assign indices in registration order")
    if state == POPULATION_COMPLETE and count and not entries:
        _fail("thread_catalog.entries", "nonzero records require a nonempty complete population")
    return ThreadPopulation(
        frozenset(keys),
        tuple(indices),
        state == POPULATION_COMPLETE,
        count,
    )


def reconcile_thread_records(
    population: ThreadPopulation | Iterable[tuple[int, int]],
    *,
    process_generation: int,
    records: Sequence[Any],
) -> tuple[int, ...]:
    """Resolve complete records; retain exact unresolved IDs only for negatives."""

    generation = _uint(process_generation, 64, "process_generation", nonzero=True)
    if isinstance(population, ThreadPopulation):
        keys = set(population.thread_keys)
        complete = population.population_complete
        if complete and len(records) != population.record_count:
            _fail("records", "record count differs from validated complete population")
    else:
        keys = set(population)
        complete = True
    unresolved: list[int] = []
    for index, record in enumerate(records):
        where = f"records[{index}]"
        thread_index = _uint(_record_field(record, "thread_index", where), 32, f"{where}.thread_index", nonzero=True)
        if thread_index > MAX_THREAD_INDEX:
            _fail(f"{where}.thread_index", "reserved thread index")
        if (generation, thread_index) not in keys:
            if complete:
                _fail(f"{where}.thread_index", "unresolved thread index in complete population")
            unresolved.append(thread_index)
    return tuple(unresolved)


def validate_thread_catalog_bytes(raw: bytes, **kwargs: Any) -> tuple[dict[str, Any], ThreadPopulation]:
    try:
        value = SEMANTIC.parse_canonical(raw)
    except Exception as error:
        raise RuntimeCatalogError(str(error)) from error
    return value, validate_thread_catalog(value, **kwargs)


def _validate_segments(value: Any, where: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        _fail(where, "nonempty mapping-segment array required")
    result: list[dict[str, Any]] = []
    previous_end = 0
    for index, item in enumerate(value):
        location = f"{where}[{index}]"
        row = dict(_exact_keys(item, MODULE_SEGMENT_FIELDS, location))
        start = _uint(row["start_address"], 64, f"{location}.start_address", nonzero=True)
        end = _uint(row["end_address"], 64, f"{location}.end_address", nonzero=True)
        file_offset = _uint(row["file_offset"], 64, f"{location}.file_offset")
        if end <= start:
            _fail(location, "segment end must exceed start")
        if file_offset + (end - start) > UINT64_MAX:
            _fail(location, "file-offset plus segment length overflows u64")
        if index and start < previous_end:
            _fail(where, "strict start order and nonoverlap required")
        if not isinstance(row["permissions"], str) or not _PERMISSION_RE.fullmatch(row["permissions"]):
            _fail(f"{location}.permissions", "exact four-character mapping permissions required")
        previous_end = end
        result.append(row)
    return result


def module_map_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact canonical mapping identity hashed by run coverage."""

    return {
        "device": row["device"],
        "inode": row["inode"],
        "loaded_path": row["loaded_path"],
        "namespace_id": row["namespace_id"],
        "segments": [dict(segment) for segment in row["segments"]],
    }


def module_map_sha256(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(SEMANTIC.canonical_bytes(module_map_projection(row))).hexdigest()


def validate_module_catalog(
    value: Mapping[str, Any],
    *,
    expected_process_generation: int,
    population_state: int,
    lifecycle_state: int,
) -> dict[tuple[int, int], Mapping[str, Any]]:
    """Validate exact base-namespace module rows keyed by generation/ID."""

    entries = _catalog(value, catalog_id="oai_memprof_module", where="module_catalog")
    if not entries:
        _fail("module_catalog.entries", "object kind 6 requires at least one observed loaded row")
    generation = _uint(expected_process_generation, 64, "expected_process_generation", nonzero=True)
    state = _population_state(population_state, lifecycle_state, "module_catalog")
    rows: dict[tuple[int, int], Mapping[str, Any]] = {}
    logical_ids: list[str] = []
    build_logical_ids: set[str] = set()
    raw_identities: set[tuple[int, int, int, str]] = set()
    for index, value_row in enumerate(entries):
        where = f"module_catalog.entries[{index}]"
        row = dict(_exact_keys(value_row, MODULE_ROW_FIELDS, where))
        _build_id(row["build_id"], f"{where}.build_id")
        logical_id = _identifier(row["logical_id"], f"{where}.logical_id")
        build_logical_id = row["build_logical_id"]
        if build_logical_id is not None:
            build_logical_id = _identifier(build_logical_id, f"{where}.build_logical_id")
        _absolute_path(row["loaded_path"], f"{where}.loaded_path")
        _uint(row["byte_count"], 64, f"{where}.byte_count", nonzero=True)
        device = _uint(row["device"], 64, f"{where}.device")
        inode = _uint(row["inode"], 64, f"{where}.inode", nonzero=True)
        load_state = _uint(row["load_state_id"], 16, f"{where}.load_state_id", nonzero=True)
        if load_state not in (1, 20, 21):
            _fail(f"{where}.load_state_id", "observed loaded state 1, 20, or 21 required")
        if load_state in (1, 21) and build_logical_id != logical_id:
            _fail(where, "configured observed row requires build logical ID equal logical ID")
        if load_state == 20 and build_logical_id is not None:
            _fail(where, "observed-unexpected row requires null build logical ID")
        if load_state in (20, 21) and state != POPULATION_NEGATIVE_PARTIAL:
            _fail(where, "unexpected/active-time load requires negative-partial terminal")
        module_generation = _uint(row["module_generation"], 64, f"{where}.module_generation", nonzero=True)
        module_id = _uint(row["module_id"], 32, f"{where}.module_id", nonzero=True)
        process_generation = _uint(row["process_generation"], 64, f"{where}.process_generation", nonzero=True)
        namespace_id = _uint(row["namespace_id"], 64, f"{where}.namespace_id")
        if namespace_id != 0:
            _fail(f"{where}.namespace_id", "v1 admits only base loader namespace zero")
        if process_generation != generation or module_generation != generation:
            _fail(where, "process/module generation must equal active generation")
        if module_id != index + 1:
            _fail(f"{where}.module_id", "must equal one-based logical-ID order")
        if logical_ids and logical_id <= logical_ids[-1]:
            _fail("module_catalog.entries", "strict logical-ID order and uniqueness required")
        if build_logical_id is not None:
            if build_logical_id in build_logical_ids:
                _fail(where, "duplicate build logical ID")
            build_logical_ids.add(build_logical_id)
        row["segments"] = _validate_segments(row["segments"], f"{where}.segments")
        map_digest = _nonzero_hash(row["module_map_sha256"], f"{where}.module_map_sha256")
        if map_digest != module_map_sha256(row):
            _fail(f"{where}.module_map_sha256", "canonical mapping projection digest mismatch")
        _nonzero_hash(row["sha256"], f"{where}.sha256")
        raw_identity = (device, inode, namespace_id, map_digest)
        if raw_identity in raw_identities:
            _fail(where, "duplicate base-namespace mapping identity")
        raw_identities.add(raw_identity)
        logical_ids.append(logical_id)
        rows[(module_generation, module_id)] = row
    return rows


def reconcile_module_relations(
    module_rows: Mapping[tuple[int, int], Mapping[str, Any]],
    *,
    callsite_module_keys: Iterable[tuple[int, int]],
    run_module_population: Sequence[Mapping[str, Any]],
) -> None:
    """Reconcile calls and every observed run row exactly once by logical ID."""

    keys = set(module_rows)
    for index, key in enumerate(callsite_module_keys):
        where = f"callsite_module_keys[{index}]"
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            _fail(where, "exact (module_generation,module_id) pair required")
        normalized = (
            _uint(key[0], 64, f"{where}[0]", nonzero=True),
            _uint(key[1], 32, f"{where}[1]", nonzero=True),
        )
        if normalized not in keys:
            _fail(where, "unresolved module generation/ID")

    observed_run: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(run_module_population):
        where = f"run_module_population[{index}]"
        if not isinstance(row, Mapping):
            _fail(where, "prevalidated run-population row required")
        required = (
            "build_logical_id", "load_generation", "load_state_id", "loaded_path",
            "logical_id", "observed", "runtime_identity",
        )
        if any(name not in row for name in required):
            _fail(where, "run-coverage reconciliation fields missing")
        logical_id = row["logical_id"]
        if not isinstance(logical_id, str):
            _fail(f"{where}.logical_id", "identifier required")
        if row["loaded_path"] is None:
            if row["observed"] is not False or row["load_generation"] is not None or row["runtime_identity"] is not None:
                _fail(where, "not-loaded projection has runtime identity")
            continue
        if row["observed"] is not True:
            _fail(where, "loaded projection must be observed")
        if logical_id in observed_run:
            _fail(where, "duplicate observed logical ID")
        observed_run[logical_id] = row

    modules_by_logical = {row["logical_id"]: row for row in module_rows.values()}
    if set(modules_by_logical) != set(observed_run):
        _fail("module_rows", "observed run population does not reconcile exactly once")
    for logical_id, module in modules_by_logical.items():
        run = observed_run[logical_id]
        runtime = run["runtime_identity"]
        if not isinstance(runtime, Mapping):
            _fail(f"run_module_population[{logical_id!r}].runtime_identity", "object required")
        expected = {
            "build_id": module["build_id"],
            "byte_count": module["byte_count"],
            "module_map_sha256": module["module_map_sha256"],
            "sha256": module["sha256"],
        }
        observed = {name: runtime.get(name) for name in expected}
        if observed != expected:
            _fail(f"module_rows[{logical_id!r}]", "runtime build/map identity mismatch")
        if (
            run["build_logical_id"] != module["build_logical_id"]
            or run["load_generation"] != module["module_generation"]
            or run["load_state_id"] != module["load_state_id"]
            or run["loaded_path"] != module["loaded_path"]
        ):
            _fail(f"module_rows[{logical_id!r}]", "run load identity mismatch")


def validate_module_catalog_bytes(raw: bytes, **kwargs: Any) -> tuple[dict[str, Any], dict[tuple[int, int], Mapping[str, Any]]]:
    try:
        value = SEMANTIC.parse_canonical(raw)
    except Exception as error:
        raise RuntimeCatalogError(str(error)) from error
    return value, validate_module_catalog(value, **kwargs)


def _resolve_counter(row: Mapping[str, Any], counter: int) -> int:
    delta = counter - row["start_counter"]
    return row["start_monotonic_raw_ns"] + (
        delta * 1_000_000_000 * row["counter_frequency_denominator"]
    ) // row["counter_frequency_numerator"]


def _validate_clock_samples(
    row: Mapping[str, Any],
    *,
    final_counter: int,
    final_monotonic_raw_ns: int,
    final_realtime_unix_ns: int,
) -> None:
    samples = row["samples"]
    if not isinstance(samples, list) or len(samples) < 2:
        _fail("clock_catalog.entries[0].samples", "at least two acquisition samples required")
    previous_counter: int | None = None
    previous_after: int | None = None
    envelope_errors: list[int] = []
    exterior_errors: list[int] = []
    parsed: list[dict[str, int]] = []
    for index, item in enumerate(samples):
        where = f"clock_catalog.entries[0].samples[{index}]"
        sample = dict(_exact_keys(item, CLOCK_SAMPLE_FIELDS, where))
        counter = _uint(sample["counter"], 64, f"{where}.counter")
        before = _uint(sample["monotonic_raw_before_ns"], 64, f"{where}.monotonic_raw_before_ns")
        after = _uint(sample["monotonic_raw_after_ns"], 64, f"{where}.monotonic_raw_after_ns")
        realtime = _uint(sample["realtime_unix_ns"], 64, f"{where}.realtime_unix_ns")
        ordinal = _uint(sample["sample_ordinal"], 32, f"{where}.sample_ordinal", nonzero=True)
        if ordinal != index + 1:
            _fail(f"{where}.sample_ordinal", "exact contiguous order from one required")
        if after < before:
            _fail(where, "monotonic acquisition bracket is reversed")
        if previous_counter is not None and counter <= previous_counter:
            _fail(where, "counter samples must strictly increase")
        if previous_after is not None and before < previous_after:
            _fail(where, "monotonic sample brackets overlap or regress")
        predicted = _resolve_counter(row, counter)
        envelope_errors.append(max(abs(predicted - before), abs(after - predicted)))
        exterior_errors.append(max(before - predicted, predicted - after, 0))
        parsed.append({
            "counter": counter,
            "before": before,
            "after": after,
            "realtime": realtime,
        })
        previous_counter = counter
        previous_after = after
    first = parsed[0]
    last = parsed[-1]
    if (
        first["counter"] != row["start_counter"]
        or not first["before"] <= row["start_monotonic_raw_ns"] <= first["after"]
        or first["realtime"] != row["start_realtime_unix_ns"]
    ):
        _fail("clock_catalog.entries[0].samples[0]", "opening anchor mismatch")
    if (
        last["counter"] != final_counter
        or not last["before"] <= final_monotonic_raw_ns <= last["after"]
        or last["realtime"] != final_realtime_unix_ns
    ):
        _fail("clock_catalog.entries[0].samples[-1]", "terminal anchor mismatch")
    if max(envelope_errors) > row["calibration_error_bound_ns"]:
        _fail("clock_catalog.entries[0].calibration_error_bound_ns", "declared bound does not cover observed bracket envelope")
    if row["observed_max_error_ns"] != max(exterior_errors):
        _fail("clock_catalog.entries[0].observed_max_error_ns", "must equal observed outside-bracket error")
    if row["realtime_discontinuity_observed"] is False:
        for index, sample in enumerate(parsed):
            realtime_delta = sample["realtime"] - first["realtime"]
            monotonic_delta = ((sample["before"] + sample["after"]) // 2) - row["start_monotonic_raw_ns"]
            if abs(realtime_delta - monotonic_delta) > row["calibration_error_bound_ns"]:
                _fail(f"clock_catalog.entries[0].samples[{index}]", "unflagged realtime discontinuity")
    if row["calibration_span_ns"] != 0:
        _fail("clock_catalog.entries[0].calibration_span_ns", "v1 exact-rate admission requires zero span")


def validate_clock_catalog(
    value: Mapping[str, Any],
    *,
    opening_identity: Mapping[str, Any],
    architecture_id: int,
    final_counter: int,
    final_monotonic_raw_ns: int,
    final_realtime_unix_ns: int,
    counter_invalid_observed: bool,
    realtime_discontinuity_observed: bool,
) -> Mapping[str, Any]:
    """Validate the sole opening/final-bound record-clock evidence row."""

    entries = _catalog(value, catalog_id="oai_memprof_clock", where="clock_catalog")
    if len(entries) != 1:
        _fail("clock_catalog.entries", "exactly one opening-bound clock row required")
    opening = _exact_keys(opening_identity, OPENING_CLOCK_FIELDS, "opening_identity")
    row = dict(_exact_keys(entries[0], CLOCK_ROW_FIELDS, "clock_catalog.entries[0]"))
    architecture = _uint(architecture_id, 16, "architecture_id", nonzero=True)
    if architecture not in (1, 2):
        _fail("architecture_id", "linux x86_64(1) or aarch64(2) required")
    row_architecture = _uint(row["architecture_id"], 16, "clock_catalog.entries[0].architecture_id", nonzero=True)
    if row_architecture != architecture:
        _fail("clock_catalog.entries[0].architecture_id", "build/run architecture mismatch")
    clock_kind = _uint(row["clock_kind"], 8, "clock_catalog.entries[0].clock_kind", nonzero=True)
    if clock_kind != {1: 1, 2: 2}[architecture]:
        _fail("clock_catalog.entries[0].clock_kind", "clock kind incompatible with architecture")
    for name in OPENING_CLOCK_FIELDS:
        if type(row[name]) is not int or row[name] != opening[name]:
            _fail(f"clock_catalog.entries[0].{name}", "typed opening-header identity mismatch")
    _uint(row["process_generation"], 64, "clock_catalog.entries[0].process_generation", nonzero=True)
    numerator = _uint(row["counter_frequency_numerator"], 64, "clock_catalog.entries[0].counter_frequency_numerator", nonzero=True)
    denominator = _uint(row["counter_frequency_denominator"], 64, "clock_catalog.entries[0].counter_frequency_denominator", nonzero=True)
    if math.gcd(numerator, denominator) != 1:
        _fail("clock_catalog.entries[0]", "counter-frequency rational must be in lowest terms")
    calibration = _uint(row["calibration_kind"], 16, "clock_catalog.entries[0].calibration_kind", nonzero=True)
    if calibration != 1:
        _fail("clock_catalog.entries[0].calibration_kind", "runtime schema v1 admits exact architectural rate only")
    source = _uint(row["acquisition_source_id"], 16, "clock_catalog.entries[0].acquisition_source_id", nonzero=True)
    expected_source = {1: 1, 2: 2}[architecture]
    if source != expected_source:
        _fail("clock_catalog.entries[0].acquisition_source_id", "architecture/calibration source mismatch")
    if row["acquisition_status_id"] != 1 or type(row["acquisition_status_id"]) is not int:
        _fail("clock_catalog.entries[0].acquisition_status_id", "verified acquisition status 1 required")
    if row["counter_stability_status_id"] != 1 or type(row["counter_stability_status_id"]) is not int:
        _fail("clock_catalog.entries[0].counter_stability_status_id", "observed monotonic status 1 required")
    _uint(row["calibration_error_bound_ns"], 64, "clock_catalog.entries[0].calibration_error_bound_ns")
    _uint(row["calibration_span_ns"], 64, "clock_catalog.entries[0].calibration_span_ns")
    _uint(row["observed_max_error_ns"], 64, "clock_catalog.entries[0].observed_max_error_ns")
    start = _uint(row["start_counter"], 64, "clock_catalog.entries[0].start_counter")
    final = _uint(final_counter, 64, "final_counter")
    final_mono = _uint(final_monotonic_raw_ns, 64, "final_monotonic_raw_ns")
    final_real = _uint(final_realtime_unix_ns, 64, "final_realtime_unix_ns")
    if final < start:
        _fail("final_counter", "raw counter wrap or regression")
    counter_invalid = _boolean(row["counter_invalid_observed"], "clock_catalog.entries[0].counter_invalid_observed")
    realtime_discontinuity = _boolean(row["realtime_discontinuity_observed"], "clock_catalog.entries[0].realtime_discontinuity_observed")
    if counter_invalid != _boolean(counter_invalid_observed, "counter_invalid_observed"):
        _fail("clock_catalog.entries[0].counter_invalid_observed", "terminal flag mismatch")
    if realtime_discontinuity != _boolean(realtime_discontinuity_observed, "realtime_discontinuity_observed"):
        _fail("clock_catalog.entries[0].realtime_discontinuity_observed", "terminal flag mismatch")
    _validate_clock_samples(
        row,
        final_counter=final,
        final_monotonic_raw_ns=final_mono,
        final_realtime_unix_ns=final_real,
    )
    return row


def reconcile_record_counters(
    clock_row: Mapping[str, Any],
    *,
    final_counter: int,
    records: Sequence[Any],
) -> list[tuple[int | None, int | None]]:
    """Validate record counter sentinels/ranges and return monotonic resolutions."""

    start = _uint(clock_row.get("start_counter"), 64, "clock_row.start_counter")
    final = _uint(final_counter, 64, "final_counter")
    if final < start:
        _fail("final_counter", "raw counter wrap or regression")
    resolved: list[tuple[int | None, int | None]] = []
    saw_invalid = False
    for index, record in enumerate(records):
        where = f"records[{index}]"
        flags = _uint(_record_field(record, "flags", where), 32, f"{where}.flags")
        values: list[int | None] = []
        raw_values: list[int | None] = []
        for name, mask in (("counter_enter", 1 << 5), ("counter_exit", 1 << 6)):
            raw = _uint(_record_field(record, name, where), 64, f"{where}.{name}")
            if flags & mask:
                if not start <= raw <= final:
                    _fail(f"{where}.{name}", "valid counter outside opening/final range")
                raw_values.append(raw)
                values.append(_resolve_counter(clock_row, raw))
            else:
                saw_invalid = True
                if raw != 0:
                    _fail(f"{where}.{name}", "invalid counter must be canonical zero")
                raw_values.append(None)
                values.append(None)
        if raw_values[0] is not None and raw_values[1] is not None and raw_values[1] < raw_values[0]:
            _fail(where, "counter exit precedes counter enter")
        resolved.append((values[0], values[1]))
    if saw_invalid and clock_row.get("counter_invalid_observed") is not True:
        _fail("clock_row.counter_invalid_observed", "invalid record counter lacks explicit observation")
    return resolved


def validate_clock_catalog_bytes(raw: bytes, **kwargs: Any) -> tuple[dict[str, Any], Mapping[str, Any]]:
    try:
        value = SEMANTIC.parse_canonical(raw)
    except Exception as error:
        raise RuntimeCatalogError(str(error)) from error
    return value, validate_clock_catalog(value, **kwargs)


__all__ = [
    "BUNDLE_CROSS_RELATION_PROPOSALS",
    "BUNDLE_ENTRY_PROPOSAL",
    "BUNDLE_MEMBER_PROPOSAL",
    "CLOCK_ROW_FIELDS",
    "CLOCK_SAMPLE_FIELDS",
    "CLOCK_SCHEMA",
    "DEFINITION_BYTES",
    "DEFINITION_SHA256",
    "MODULE_ROW_FIELDS",
    "MODULE_SCHEMA",
    "MODULE_SEGMENT_FIELDS",
    "OPENING_CLOCK_FIELDS",
    "POPULATION_COMPLETE",
    "POPULATION_NEGATIVE_PARTIAL",
    "RUNTIME_DEFINITION",
    "RUNTIME_DEFINITION_OBJECT_TYPE",
    "RUNTIME_DEFINITION_PATH",
    "RuntimeCatalogError",
    "SEMANTIC",
    "THREAD_ROW_FIELDS",
    "THREAD_SCHEMA",
    "ThreadPopulation",
    "definition_bytes",
    "module_map_projection",
    "module_map_sha256",
    "reconcile_module_relations",
    "reconcile_record_counters",
    "reconcile_thread_records",
    "validate_clock_catalog",
    "validate_clock_catalog_bytes",
    "validate_definition",
    "validate_definition_bytes",
    "validate_module_catalog",
    "validate_module_catalog_bytes",
    "validate_thread_catalog",
    "validate_thread_catalog_bytes",
]
