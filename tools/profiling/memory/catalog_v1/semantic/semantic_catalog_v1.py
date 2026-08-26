#!/usr/bin/env python3
"""Independent schema-v1 canonical JSON and static semantic validator.

This module is deliberately standard-library only and performs no writes.
It rejects non-canonical bytes before applying exact semantic validators.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


VERSION = {"major": 1, "minor": 0}
EVENT_API_VERSION = {"major": 1, "minor": 1}
BUNDLE_VERSION = {"major": 1, "minor": 5}
EVENT_SEMANTICS_SHA256 = "8dbe428939592cdfc86ba8730078563672de4a13081ebdb868fa97f543dfab89"
API_CATALOG_SHA256 = "93056c4cfd071c1df396ba09bf82b4cbe807923977c4bca988b0aee1b8c94610"
UINT16_MAX = (1 << 16) - 1
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
CANONICAL_RAW_MAX_BYTES = 64 * 1024 * 1024
CANONICAL_JSON_MAX_NESTING_DEPTH = 64
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1

KNOWN_FLAGS_MASK = 0x030FFFFF
RESERVED_FLAGS_MASK = 0xFCF00000
RESULT_KIND_MASK = 0x03000000
COMMON_EVIDENCE_MASK = 0x000401E0
SELECTION_MASK = 0x0003C000
OPERAND_VALIDITY_MASK = 0x0000001C
BASE_FIXED_MASK = KNOWN_FLAGS_MASK & ~(COMMON_EVIDENCE_MASK | SELECTION_MASK | OPERAND_VALIDITY_MASK)

MEMBER_REGISTRY = (
    (1, "canonical_json", "definition/canonical-json-v1.json", "semantic"),
    (2, "schema_bundle_schema", "definition/schema-bundle-schema-v1.json", "semantic"),
    (3, "event_semantics", "definition/event-semantics-v1.json", "semantic"),
    (4, "api_definition", "catalog/api.json", "semantic"),
    (5, "callsite_rule_definition", "definition/callsite-rule-v1.json", "semantic"),
    (6, "context_schema_definition", "definition/context-schema-v1.json", "semantic"),
    (7, "phase_definition", "definition/phase-v1.json", "semantic"),
    (8, "diagnostic_definition", "definition/diagnostic-v1.json", "semantic"),
    (9, "coverage_policy", "definition/coverage-policy-v1.json", "coverage"),
    (10, "coverage_instance_schema", "definition/coverage-instance-schema-v1.json", "coverage"),
    (11, "effective_config_schema", "definition/effective-config-schema-v1.json", "config"),
    (12, "selection_rule_definition", "definition/selection-rule-v1.json", "sampling"),
    (13, "runtime_catalog_schema", "definition/runtime-catalog-schema-v1.json", "runtime"),
)

EXTERNAL_OBJECTS = (
    (1, "catalog/schema-bundle.json", 0x05),
    (2, "catalog/api.json", 0x05),
    (3, "catalog/context.json", 0x0B),
    (4, "catalog/callsite.json", 0x1B),
    (5, "catalog/thread.json", 0x1B),
    (6, "catalog/module.json", 0x13),
    (7, "catalog/clock.json", 0x03),
    (8, "catalog/build-coverage.json", 0x03),
    (9, "catalog/run-coverage.json", 0x13),
    (10, "metadata/effective-config.json", 0x07),
    (11, "status/diagnostics.json", 0x03),
    (12, "status/pre-footer-status.json", 0x03),
)

CROSS_HASH_RELATIONS = (
    (
        "api_bundle_member",
        "opening.api_catalog_definition_sha256",
        "schema_bundle.entries[object_type=4].sha256",
    ),
    (
        "api_object_binding",
        "opening.api_catalog_definition_sha256",
        "object_bindings[object_kind=2].sha256",
    ),
    (
        "build_coverage_api_definition",
        "object_bindings[object_kind=8].api_definition",
        "schema_bundle.entries[object_type=4].sha256",
    ),
    (
        "build_coverage_policy",
        "object_bindings[object_kind=8].policy",
        "schema_bundle.entries[object_type=9].sha256",
    ),
    (
        "build_coverage_schema",
        "object_bindings[object_kind=8].schema",
        "schema_bundle.entries[object_type=10].sha256",
    ),
    (
        "callsite_header",
        "opening.callsite_catalog_definition_sha256",
        "schema_bundle.entries[object_type=5].sha256",
    ),
    (
        "clock_catalog_schema",
        "catalog/clock.json.schema.sha256",
        "schema_bundle.entries[object_type=13].sha256",
    ),
    (
        "context_schema",
        "catalog/context.json.schema.sha256",
        "schema_bundle.entries[object_type=6].sha256",
    ),
    (
        "effective_configuration",
        "opening.configuration_instance_sha256",
        "object_bindings[object_kind=10].sha256",
    ),
    (
        "effective_configuration_schema",
        "metadata/effective-config.json.schema.sha256",
        "schema_bundle.entries[object_type=11].sha256",
    ),
    (
        "module_catalog_schema",
        "catalog/module.json.schema.sha256",
        "schema_bundle.entries[object_type=13].sha256",
    ),
    (
        "phase_context",
        "catalog/context.json.entries[*].phase_id",
        "schema_bundle.entries[object_type=7].defined_phase_id",
    ),
    (
        "run_coverage_build",
        "object_bindings[object_kind=9].build_coverage",
        "object_bindings[object_kind=8].sha256",
    ),
    (
        "run_coverage_configuration",
        "object_bindings[object_kind=9].configuration_instance_sha256",
        "opening.configuration_instance_sha256",
    ),
    (
        "run_coverage_policy",
        "object_bindings[object_kind=9].policy",
        "schema_bundle.entries[object_type=9].sha256",
    ),
    (
        "run_coverage_schema",
        "object_bindings[object_kind=9].schema",
        "schema_bundle.entries[object_type=10].sha256",
    ),
    (
        "schema_bundle_object",
        "opening.schema_bundle_definition_sha256",
        "object_bindings[object_kind=1].sha256",
    ),
    (
        "schema_bundle_schema",
        "catalog/schema-bundle.json.schema.sha256",
        "schema_bundle.entries[object_type=2].sha256",
    ),
    (
        "status_diagnostic_schema",
        "status/diagnostics.json.schema.sha256",
        "schema_bundle.entries[object_type=8].sha256",
    ),
    (
        "thread_catalog_schema",
        "catalog/thread.json.schema.sha256",
        "schema_bundle.entries[object_type=13].sha256",
    ),
)

_IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9_]*\Z", re.ASCII)
_HASH_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


class SemanticError(ValueError):
    """A deterministic canonical or semantic rejection."""


def _reject_float(_: str) -> None:
    raise SemanticError("number: floats and exponents are forbidden")


def _reject_constant(_: str) -> None:
    raise SemanticError("number: non-finite values are forbidden")


def _pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    previous: bytes | None = None
    for key, value in pairs:
        if key in out:
            raise SemanticError(f"member: duplicate key {key!r}")
        if not isinstance(key, str) or not _IDENTIFIER_RE.fullmatch(key):
            raise SemanticError(f"member: invalid ASCII identifier {key!r}")
        encoded = key.encode("ascii")
        if previous is not None and encoded <= previous:
            raise SemanticError("member: keys are not in bytewise ASCII order")
        previous = encoded
        out[key] = value
    return out


def _validate_string(value: str, where: str) -> None:
    if "\x00" in value:
        raise SemanticError(f"{where}: U+0000 is forbidden")
    if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise SemanticError(f"{where}: surrogate code point is forbidden")
    if unicodedata.normalize("NFC", value) != value:
        raise SemanticError(f"{where}: string is not NFC")


def _check_json_nesting(raw: bytes) -> None:
    """Reject source nesting beyond the bounded canonical JSON grammar."""

    depth = 0
    in_string = False
    escaped = False
    for byte in raw:
        if in_string:
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                in_string = False
        elif byte == ord('"'):
            in_string = True
        elif byte in (ord("["), ord("{")):
            depth += 1
            if depth > CANONICAL_JSON_MAX_NESTING_DEPTH:
                raise SemanticError(
                    f"json: nesting depth exceeds {CANONICAL_JSON_MAX_NESTING_DEPTH}"
                )
        elif byte in (ord("]"), ord("}")):
            depth -= 1


def _walk(value: Any, where: str = "$", depth: int = 0) -> None:
    if isinstance(value, str):
        _validate_string(value, where)
    elif isinstance(value, list):
        if depth >= CANONICAL_JSON_MAX_NESTING_DEPTH:
            raise SemanticError(
                f"{where}: nesting depth exceeds {CANONICAL_JSON_MAX_NESTING_DEPTH}"
            )
        for index, item in enumerate(value):
            _walk(item, f"{where}[{index}]", depth + 1)
    elif isinstance(value, dict):
        if depth >= CANONICAL_JSON_MAX_NESTING_DEPTH:
            raise SemanticError(
                f"{where}: nesting depth exceeds {CANONICAL_JSON_MAX_NESTING_DEPTH}"
            )
        for key, item in value.items():
            if not isinstance(key, str) or not _IDENTIFIER_RE.fullmatch(key):
                raise SemanticError(f"{where}: invalid ASCII identifier key {key!r}")
            _validate_string(key, f"{where}.<key>")
            _walk(item, f"{where}.{key}", depth + 1)
    elif value is None or isinstance(value, (bool, int)):
        return
    else:
        raise SemanticError(f"{where}: unsupported JSON value type")


def _canonical_string(value: str) -> str:
    encoded: list[str] = ['"']
    for char in value:
        codepoint = ord(char)
        if char == '"':
            encoded.append('\\"')
        elif char == "\\":
            encoded.append("\\\\")
        elif codepoint <= 0x1F or codepoint == 0x7F:
            encoded.append(f"\\u00{codepoint:02x}")
        else:
            encoded.append(char)
    encoded.append('"')
    return "".join(encoded)


def _canonical_value(value: Any, depth: int = 0) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return _canonical_string(value)
    if isinstance(value, list):
        if depth >= CANONICAL_JSON_MAX_NESTING_DEPTH:
            raise SemanticError(
                "canonicalization: nesting depth exceeds "
                f"{CANONICAL_JSON_MAX_NESTING_DEPTH}"
            )
        return "[" + ",".join(_canonical_value(item, depth + 1) for item in value) + "]"
    if isinstance(value, dict):
        if depth >= CANONICAL_JSON_MAX_NESTING_DEPTH:
            raise SemanticError(
                "canonicalization: nesting depth exceeds "
                f"{CANONICAL_JSON_MAX_NESTING_DEPTH}"
            )
        return "{" + ",".join(
            _canonical_string(key) + ":" + _canonical_value(value[key], depth + 1)
            for key in sorted(value)
        ) + "}"
    raise SemanticError("canonicalization: unsupported value type")


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    """Return exact format-1 bytes for an already-typed top-level object."""

    if not isinstance(value, dict):
        raise SemanticError("root: exactly one object is required")
    try:
        _walk(value)
        return _canonical_value(value).encode("utf-8") + b"\n"
    except RecursionError as error:
        raise SemanticError("canonicalization: recursion limit exceeded") from error


def parse_canonical(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise SemanticError("raw: bytes required")
    if len(raw) > CANONICAL_RAW_MAX_BYTES:
        raise SemanticError(f"raw: exceeds {CANONICAL_RAW_MAX_BYTES} byte limit")
    _check_json_nesting(raw)
    if raw.startswith(b"\xef\xbb\xbf"):
        raise SemanticError("encoding: BOM is forbidden")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise SemanticError("encoding: exactly one final LF is required")
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise SemanticError("encoding: invalid UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise SemanticError(f"json: {error.msg}") from error
    except RecursionError as error:
        raise SemanticError("json: recursion limit exceeded") from error
    if not isinstance(value, dict):
        raise SemanticError("root: exactly one object is required")
    try:
        _walk(value)
    except RecursionError as error:
        raise SemanticError("json: recursion limit exceeded") from error
    if canonical_bytes(value) != raw:
        raise SemanticError("encoding: bytes are not the canonical serialization")
    return value


def sha256_hex(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _exact_keys(value: Mapping[str, Any], keys: Iterable[str], where: str) -> None:
    expected = tuple(sorted(keys))
    actual = tuple(value.keys())
    if actual != expected:
        raise SemanticError(f"{where}: exact keys {expected!r} required, got {actual!r}")


def _uint(value: Any, bits: int, where: str, *, nonzero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SemanticError(f"{where}: u{bits} integer required")
    lower = 1 if nonzero else 0
    upper = (1 << bits) - 1
    if not lower <= value <= upper:
        raise SemanticError(f"{where}: outside u{bits} range")
    return value


def _i32(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not INT32_MIN <= value <= INT32_MAX:
        raise SemanticError(f"{where}: i32 integer required")
    return value


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise SemanticError(f"{where}: boolean required")
    return value


def _string(value: Any, where: str, *, nonempty: bool = True, maximum_bytes: int = 4096) -> str:
    if not isinstance(value, str):
        raise SemanticError(f"{where}: string required")
    _validate_string(value, where)
    if nonempty and not value:
        raise SemanticError(f"{where}: empty string forbidden")
    if len(value.encode("utf-8")) > maximum_bytes:
        raise SemanticError(f"{where}: string exceeds {maximum_bytes} UTF-8 bytes")
    return value


def _identifier(value: Any, where: str) -> str:
    text = _string(value, where, maximum_bytes=63)
    if not _IDENTIFIER_RE.fullmatch(text):
        raise SemanticError(f"{where}: ASCII identifier required")
    return text


def _hash(value: Any, where: str) -> str:
    text = _string(value, where, maximum_bytes=64)
    if not _HASH_RE.fullmatch(text):
        raise SemanticError(f"{where}: 64 lowercase hexadecimal characters required")
    return text


def _relative_path(value: Any, where: str) -> str:
    text = _string(value, where, maximum_bytes=4096)
    if text.startswith("/") or text.endswith("/") or "\\" in text or _CONTROL_RE.search(text):
        raise SemanticError(f"{where}: invalid relative POSIX path")
    components = text.split("/")
    if any(component in ("", ".", "..") for component in components):
        raise SemanticError(f"{where}: empty/dot/dot-dot component forbidden")
    return text


def _absolute_path(value: Any, where: str) -> str:
    text = _string(value, where, maximum_bytes=4096)
    if text == "/" or not text.startswith("/") or text.endswith("/") or "\\" in text or _CONTROL_RE.search(text):
        raise SemanticError(f"{where}: invalid absolute POSIX path")
    components = text[1:].split("/")
    if any(component in ("", ".", "..") for component in components):
        raise SemanticError(f"{where}: empty/dot/dot-dot component forbidden")
    return text


def _version(value: Any, where: str) -> None:
    if not isinstance(value, dict):
        raise SemanticError(f"{where}: version object required")
    _exact_keys(value, ("major", "minor"), where)
    if value != VERSION:
        raise SemanticError(f"{where}: exact version 1.0 required")


def _event_api_version(value: Any, where: str) -> None:
    if not isinstance(value, dict):
        raise SemanticError(f"{where}: version object required")
    _exact_keys(value, ("major", "minor"), where)
    if value != EVENT_API_VERSION:
        raise SemanticError(f"{where}: exact version 1.1 required")


def _bundle_version(value: Any, where: str) -> None:
    if not isinstance(value, dict):
        raise SemanticError(f"{where}: version object required")
    _exact_keys(value, ("major", "minor"), where)
    if value != BUNDLE_VERSION:
        raise SemanticError(f"{where}: exact version 1.5 required")


def _binding(value: Any, where: str, *, object_type: int, path: str) -> None:
    if not isinstance(value, dict):
        raise SemanticError(f"{where}: binding object required")
    _exact_keys(value, ("object_type", "path", "sha256"), where)
    if _uint(value["object_type"], 16, f"{where}.object_type", nonzero=True) != object_type:
        raise SemanticError(f"{where}: object_type must be {object_type}")
    if _relative_path(value["path"], f"{where}.path") != path:
        raise SemanticError(f"{where}: path must be {path!r}")
    _hash(value["sha256"], f"{where}.sha256")


def load_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.stat().st_size > CANONICAL_RAW_MAX_BYTES:
        raise SemanticError(f"raw: exceeds {CANONICAL_RAW_MAX_BYTES} byte limit")
    raw = path.read_bytes()
    return parse_canonical(raw), raw


def validate_definition_envelope(value: Mapping[str, Any], expected_id: str, where: str) -> None:
    if value.get("definition_id") != expected_id:
        raise SemanticError(f"{where}.definition_id: expected {expected_id!r}")
    _version(value.get("version"), f"{where}.version")


def validate_canonical_definition(value: Mapping[str, Any]) -> None:
    _exact_keys(value, ("canonicalization", "definition_id", "path_types", "version"), "canonical_json")
    validate_definition_envelope(value, "oai_memprof_canonical_json", "canonical_json")
    canonicalization = value["canonicalization"]
    required = {
        "arrays_schema_ordered": True,
        "bom_allowed": False,
        "final_lf_count": 1,
        "floats_allowed": False,
        "insignificant_whitespace_allowed": False,
        "member_name_pattern": "[a-z][a-z0-9_]*",
        "member_order": "bytewise_ascii_ascending",
        "negative_zero_allowed": False,
        "normalization": "nfc",
        "root_type": "object",
        "sha256_domain": "all_canonical_bytes_including_final_lf",
        "string_escape": "quote_backslash_control_only_controls_lowercase_u00xx",
        "surrogates_allowed": False,
        "u0000_allowed": False,
        "unknown_members_allowed": False,
        "utf8": "strict",
    }
    if canonicalization != required:
        raise SemanticError("canonical_json.canonicalization: frozen rules differ")
    path_types = value["path_types"]
    if not isinstance(path_types, list) or [row.get("name") for row in path_types] != [
        "absolute_path",
        "archive_path",
        "repo_path",
    ]:
        raise SemanticError("canonical_json.path_types: exact ordered path domains required")
    for index, row in enumerate(path_types):
        if not isinstance(row, dict):
            raise SemanticError(f"canonical_json.path_types[{index}]: object required")
        _exact_keys(row, ("encoding", "name", "root", "rules"), f"canonical_json.path_types[{index}]")
        _identifier(row["name"], f"canonical_json.path_types[{index}].name")
        if row["encoding"] != "utf8_nfc_fail_closed":
            raise SemanticError("canonical_json.path_types: frozen encoding required")
        if row["root"] not in ("host_filesystem", "evidence_archive", "repository"):
            raise SemanticError("canonical_json.path_types: invalid root")
        if not isinstance(row["rules"], list) or not all(isinstance(item, str) for item in row["rules"]):
            raise SemanticError("canonical_json.path_types: rules array of strings required")


def validate_schema_bundle_schema(value: Mapping[str, Any]) -> None:
    _exact_keys(
        value,
        ("cross_hash_relations", "definition_id", "external_objects", "member_registry", "schema", "version"),
        "schema_bundle_schema",
    )
    if value.get("definition_id") != "oai_memprof_schema_bundle_schema":
        raise SemanticError("schema_bundle_schema.definition_id: exact ID required")
    _bundle_version(value.get("version"), "schema_bundle_schema.version")
    schema = value["schema"]
    if not isinstance(schema, dict):
        raise SemanticError("schema_bundle_schema.schema: object required")
    _exact_keys(schema, ("entry_keys", "entry_order", "top_keys"), "schema_bundle_schema.schema")
    if schema != {
        "entry_keys": ["bytes", "object_type", "path", "sha256"],
        "entry_order": "strictly_increasing_object_type",
        "top_keys": ["catalog_id", "entries", "schema", "version"],
    }:
        raise SemanticError("schema_bundle_schema.schema: exact schema differs")
    registry = value["member_registry"]
    if not isinstance(registry, list) or len(registry) != len(MEMBER_REGISTRY):
        raise SemanticError("schema_bundle_schema.member_registry: thirteen rows required")
    observed = []
    for index, row in enumerate(registry):
        if not isinstance(row, dict):
            raise SemanticError("schema_bundle_schema.member_registry: row object required")
        _exact_keys(row, ("name", "object_type", "owner", "path"), f"member_registry[{index}]")
        observed.append(
            (
                _uint(row["object_type"], 16, f"member_registry[{index}].object_type", nonzero=True),
                _identifier(row["name"], f"member_registry[{index}].name"),
                _relative_path(row["path"], f"member_registry[{index}].path"),
                _identifier(row["owner"], f"member_registry[{index}].owner"),
            )
        )
    if tuple(observed) != MEMBER_REGISTRY:
        raise SemanticError("schema_bundle_schema.member_registry: frozen IDs/paths differ")
    objects = value["external_objects"]
    if not isinstance(objects, list) or len(objects) != len(EXTERNAL_OBJECTS):
        raise SemanticError("schema_bundle_schema.external_objects: twelve rows required")
    observed_objects = []
    for index, row in enumerate(objects):
        if not isinstance(row, dict):
            raise SemanticError("schema_bundle_schema.external_objects: row object required")
        _exact_keys(row, ("flags", "object_kind", "path"), f"external_objects[{index}]")
        observed_objects.append(
            (
                _uint(row["object_kind"], 16, f"external_objects[{index}].object_kind", nonzero=True),
                _relative_path(row["path"], f"external_objects[{index}].path"),
                _uint(row["flags"], 32, f"external_objects[{index}].flags"),
            )
        )
    if tuple(observed_objects) != EXTERNAL_OBJECTS:
        raise SemanticError("schema_bundle_schema.external_objects: frozen path/flag mapping differs")
    relations = value["cross_hash_relations"]
    if not isinstance(relations, list):
        raise SemanticError("schema_bundle_schema.cross_hash_relations: array required")
    observed_relations = []
    for index, row in enumerate(relations):
        if not isinstance(row, dict):
            raise SemanticError("cross_hash_relations: row object required")
        _exact_keys(row, ("left", "relation_id", "right"), f"cross_hash_relations[{index}]")
        observed_relations.append(
            (
                _identifier(row["relation_id"], f"cross_hash_relations[{index}].relation_id"),
                _string(row["left"], f"cross_hash_relations[{index}].left"),
                _string(row["right"], f"cross_hash_relations[{index}].right"),
            )
        )
    if tuple(observed_relations) != CROSS_HASH_RELATIONS:
        raise SemanticError("cross_hash_relations: exact frozen relation tuples required")


def _validate_flag_rows(rows: Any) -> None:
    expected_names = [
        "address_before_valid",
        "address_after_valid",
        "arg0_valid",
        "arg1_valid",
        "arg2_valid",
        "counter_enter_valid",
        "counter_exit_valid",
        "cpu_enter_valid",
        "cpu_exit_valid",
        "zero_size_request",
        "calloc_size_product_overflow",
        "successor_created",
        "predecessor_ended",
        "operation_failed",
        "predecessor_match_valid",
        "predecessor_selected",
        "successor_selected",
        "cross_thread_endpoint",
        "boundary_straddling",
        "reallocarray_size_product_overflow",
    ]
    if not isinstance(rows, list) or len(rows) != len(expected_names):
        raise SemanticError("event_semantics.flag_bits: bits 0..19 required")
    bits: list[int] = []
    names: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SemanticError("event_semantics.flag_bits: row object required")
        _exact_keys(row, ("bit", "mask", "name"), f"flag_bits[{index}]")
        bit = _uint(row["bit"], 8, f"flag_bits[{index}].bit")
        mask = _uint(row["mask"], 32, f"flag_bits[{index}].mask")
        names.append(_identifier(row["name"], f"flag_bits[{index}].name"))
        if mask != 1 << bit:
            raise SemanticError("event_semantics.flag_bits: bit/mask mismatch")
        bits.append(bit)
    if bits != list(range(20)) or names != expected_names:
        raise SemanticError("event_semantics.flag_bits: exact bit/name order required")


EXPECTED_TRANSITION_IDS = tuple(
    list(range(100, 104))
    + list(range(200, 205))
    + list(range(300, 309))
    + [400, 401, 402]
    + list(range(500, 511))
    + list(range(600, 604))
    + list(range(700, 705))
    + list(range(800, 804))
    + list(range(900, 904))
    + list(range(1000, 1004))
    + [1100, 1101, 1200, 1201]
)


def _validate_transition_rows(rows: Any) -> set[int]:
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_TRANSITION_IDS):
        raise SemanticError("event_semantics.transitions: exactly 57 rows required")
    ids: list[int] = []
    for index, row in enumerate(rows):
        where = f"transitions[{index}]"
        _exact_keys(
            row,
            (
                "address_after_null",
                "address_before_null",
                "api_id",
                "arg0_zero",
                "calloc_product",
                "matched_operand_profile",
                "name",
                "realloc_zero_policy_id",
                "reallocarray_product",
                "required_one_mask",
                "required_zero_mask",
                "result_code_zero",
                "transition_id",
                "unmatched_operand_profile",
            ),
            where,
        )
        transition_id = _uint(
            row["transition_id"], 16, f"{where}.transition_id", nonzero=True
        )
        api_id = _uint(row["api_id"], 16, f"{where}.api_id", nonzero=True)
        if transition_id // 100 != api_id:
            raise SemanticError(f"{where}: transition hundred-group must equal API ID")
        _identifier(row["name"], f"{where}.name")
        for name in ("address_after_null", "address_before_null", "arg0_zero"):
            if row[name] is not None:
                _bool(row[name], f"{where}.{name}")
        for name, applicable_api in (
            ("calloc_product", 2),
            ("reallocarray_product", 5),
        ):
            state = row[name]
            if state not in ("not_applicable", "overflow", "positive", "zero"):
                raise SemanticError(f"{where}.{name}: invalid state")
            if (api_id == applicable_api) != (state != "not_applicable"):
                raise SemanticError(f"{where}.{name}: API applicability differs")
        if api_id == 7:
            _bool(row["result_code_zero"], f"{where}.result_code_zero")
        elif row["result_code_zero"] is not None:
            raise SemanticError(f"{where}.result_code_zero: only API 7 uses direct return")
        if row["matched_operand_profile"] is not None:
            _identifier(
                row["matched_operand_profile"],
                f"{where}.matched_operand_profile",
            )
        _identifier(
            row["unmatched_operand_profile"],
            f"{where}.unmatched_operand_profile",
        )
        policy = row["realloc_zero_policy_id"]
        if policy is not None:
            if api_id not in (3, 5) or _uint(
                policy, 16, f"{where}.realloc_zero_policy_id", nonzero=True
            ) not in (1, 2):
                raise SemanticError(f"{where}.realloc_zero_policy_id: invalid binding")
        one = _uint(row["required_one_mask"], 32, f"{where}.required_one_mask")
        zero = _uint(row["required_zero_mask"], 32, f"{where}.required_zero_mask")
        if (
            one & zero
            or one & ~BASE_FIXED_MASK
            or zero & ~BASE_FIXED_MASK
            or (one | zero) != BASE_FIXED_MASK
        ):
            raise SemanticError(f"{where}: exact complementary base masks required")
        expected_result = 0x02000000 if api_id == 7 else 0x01000000
        if one & RESULT_KIND_MASK != expected_result:
            raise SemanticError(f"{where}: result-kind bit differs from API")
        ids.append(transition_id)
    if tuple(ids) != EXPECTED_TRANSITION_IDS:
        raise SemanticError("event_semantics.transitions: exact sorted ID population required")
    return set(ids)


def validate_event_semantics(value: Mapping[str, Any]) -> None:
    _exact_keys(
        value,
        (
            "definition_id",
            "event_kinds",
            "field_rules",
            "flag_bits",
            "flag_masks",
            "mask_composition",
            "mode_rules",
            "operand_profiles",
            "realloc_zero_policies",
            "requested_byte_rules",
            "result_kinds",
            "selection_cases",
            "transitions",
            "valid_pairs",
            "version",
        ),
        "event_semantics",
    )
    if value["definition_id"] != "oai_memprof_event_semantics":
        raise SemanticError("event_semantics.definition_id: exact ID required")
    _event_api_version(value["version"], "event_semantics.version")
    kinds = value["event_kinds"]
    expected_kinds = [
        (1, "allocation_result"),
        (2, "reallocation_result"),
        (3, "release_call"),
    ]
    if not isinstance(kinds, list):
        raise SemanticError("event_semantics.event_kinds: array required")
    observed_kinds = []
    for index, row in enumerate(kinds):
        _exact_keys(row, ("event_kind", "name"), f"event_kinds[{index}]")
        observed_kinds.append(
            (
                _uint(row["event_kind"], 16, "event_kind", nonzero=True),
                _identifier(row["name"], "name"),
            )
        )
    if observed_kinds != expected_kinds:
        raise SemanticError("event_semantics.event_kinds: frozen IDs differ")
    expected_pairs = [
        {"api_id": api_id, "event_kind": event_kind}
        for api_id, event_kind in (
            (1, 1), (2, 1), (3, 2), (4, 3), (5, 2), (6, 1),
            (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1)
        )
    ]
    if value["valid_pairs"] != expected_pairs:
        raise SemanticError("event_semantics.valid_pairs: exact sorted pairs required")
    _validate_flag_rows(value["flag_bits"])
    if value["flag_masks"] != {
        "base_fixed_mask": BASE_FIXED_MASK,
        "common_evidence_mask": COMMON_EVIDENCE_MASK,
        "known_mask": KNOWN_FLAGS_MASK,
        "operand_validity_mask": OPERAND_VALIDITY_MASK,
        "reserved_mask": RESERVED_FLAGS_MASK,
        "result_kind_mask": RESULT_KIND_MASK,
        "selection_mask": SELECTION_MASK,
    }:
        raise SemanticError("event_semantics.flag_masks: frozen values differ")
    if value["mask_composition"] != {
        "final_required_one": "transition.required_one_mask|operand_profile.required_one_mask|selection_case.required_one_mask",
        "final_required_zero": "transition.required_zero_mask|operand_profile.required_zero_mask|selection_case.required_zero_mask",
        "invariants": "pairwise_domains_disjoint_and_final_masks_cover_known_mask_except_common_evidence_mask",
        "variable_mask": COMMON_EVIDENCE_MASK,
    }:
        raise SemanticError("event_semantics.mask_composition: exact composition rule differs")
    expected_profiles = [
        ("alignment_size", 0x0C, 0x10),
        ("calloc_overflow", 0x0C, 0x10),
        ("calloc_valid", 0x1C, 0),
        ("free_exact", 0, 0x1C),
        ("free_matched", 0x1C, 0),
        ("free_matched_unknown", 0x18, 0x04),
        ("malloc", 0x04, 0x18),
        ("realloc_matched", 0x1C, 0),
        ("realloc_unmatched", 0x04, 0x18),
        ("reallocarray_matched", 0x1C, 0),
        ("reallocarray_overflow", 0x18, 0x04),
        ("reallocarray_unmatched", 0x1C, 0),
        ("single_size", 0x04, 0x18),
        ("string_source", 0x04, 0x18),
        ("string_source_limit", 0x0C, 0x10),
    ]
    profiles = value["operand_profiles"]
    if not isinstance(profiles, list):
        raise SemanticError("event_semantics.operand_profiles: array required")
    observed_profiles = []
    for index, row in enumerate(profiles):
        _exact_keys(
            row,
            ("name", "required_one_mask", "required_zero_mask"),
            f"operand_profiles[{index}]",
        )
        name = _identifier(row["name"], f"operand_profiles[{index}].name")
        one = _uint(row["required_one_mask"], 32, "required_one_mask")
        zero = _uint(row["required_zero_mask"], 32, "required_zero_mask")
        if one & zero or (one | zero) != OPERAND_VALIDITY_MASK:
            raise SemanticError(
                "event_semantics.operand_profiles: exact complementary operand masks required"
            )
        observed_profiles.append((name, one, zero))
    if observed_profiles != expected_profiles:
        raise SemanticError("event_semantics.operand_profiles: frozen profiles differ")
    expected_results = [
        (0, "none", False),
        (0x01000000, "errno", True),
        (0x02000000, "direct_return", True),
        (0x03000000, "exception_category", False),
    ]
    observed_results = []
    for index, row in enumerate(value["result_kinds"]):
        _exact_keys(
            row,
            ("first_slice_allowed", "name", "value"),
            f"result_kinds[{index}]",
        )
        observed_results.append(
            (
                _uint(row["value"], 32, "value"),
                _identifier(row["name"], "name"),
                _bool(row["first_slice_allowed"], "first_slice_allowed"),
            )
        )
    if observed_results != expected_results:
        raise SemanticError("event_semantics.result_kinds: frozen values differ")
    expected_policies = [
        {"admitted": False, "operation_failed": None, "policy_id": 0, "predecessor_ended": None, "name": "invalid_unavailable"},
        {"admitted": True, "operation_failed": False, "policy_id": 1, "predecessor_ended": True, "name": "predecessor_ended_no_successor"},
        {"admitted": True, "operation_failed": True, "policy_id": 2, "predecessor_ended": False, "name": "predecessor_preserved_failure"},
    ]
    if value["realloc_zero_policies"] != expected_policies:
        raise SemanticError("event_semantics.realloc_zero_policies: exact policy table differs")
    expected_requested = [
        {"api_id": 1, "availability": "always", "source": "arg0", "unknown_reason_id": None},
        {"api_id": 2, "availability": "unless_product_overflow", "source": "arg2", "unknown_reason_id": 32},
        {"api_id": 3, "availability": "always", "source": "arg0", "unknown_reason_id": None},
        {"api_id": 4, "availability": "selected_predecessor_known_only", "source": "arg0", "unknown_reason_id": None},
        {"api_id": 5, "availability": "unless_product_overflow", "source": "arg0", "unknown_reason_id": 32},
        {"api_id": 6, "availability": "always", "source": "arg1", "unknown_reason_id": None},
        {"api_id": 7, "availability": "always", "source": "arg1", "unknown_reason_id": None},
        {"api_id": 8, "availability": "always", "source": "arg1", "unknown_reason_id": None},
        {"api_id": 9, "availability": "always", "source": "arg0", "unknown_reason_id": None},
        {"api_id": 10, "availability": "always", "source": "arg0", "unknown_reason_id": None},
        {"api_id": 11, "availability": "unavailable_without_second_scan", "source": "none", "unknown_reason_id": 32},
        {"api_id": 12, "availability": "unavailable_without_second_scan", "source": "none", "unknown_reason_id": 32},
    ]
    if value["requested_byte_rules"] != expected_requested:
        raise SemanticError("event_semantics.requested_byte_rules: exact rules differ")
    transition_ids = _validate_transition_rows(value["transitions"])
    cases = value["selection_cases"]
    expected_case_names = [
        "both_cross", "both_same", "none", "predecessor_cross",
        "predecessor_same", "successor_only",
    ]
    if not isinstance(cases, list) or [row.get("name") for row in cases] != expected_case_names:
        raise SemanticError("event_semantics.selection_cases: exact ordered cases required")
    observed_cases = []
    for index, row in enumerate(cases):
        _exact_keys(
            row,
            ("arg_match_valid", "name", "required_one_mask", "required_zero_mask"),
            f"selection_cases[{index}]",
        )
        _bool(row["arg_match_valid"], f"selection_cases[{index}].arg_match_valid")
        one = _uint(row["required_one_mask"], 32, "required_one_mask")
        zero = _uint(row["required_zero_mask"], 32, "required_zero_mask")
        if one & zero or (one | zero) != SELECTION_MASK:
            raise SemanticError(
                "event_semantics.selection_cases: exact complementary selection masks required"
            )
        observed_cases.append((row["name"], row["arg_match_valid"], one, zero))
    if observed_cases != [
        ("both_cross", True, 245760, 0),
        ("both_same", True, 114688, 131072),
        ("none", False, 0, 245760),
        ("predecessor_cross", True, 180224, 65536),
        ("predecessor_same", True, 49152, 196608),
        ("successor_only", False, 65536, 180224),
    ]:
        raise SemanticError("event_semantics.selection_cases: exact case truth table differs")
    modes = value["mode_rules"]
    if not isinstance(modes, list) or [row.get("mode_id") for row in modes] != list(range(7)):
        raise SemanticError("event_semantics.mode_rules: A00..A06 rows required")
    observed_modes = []
    for index, row in enumerate(modes):
        _exact_keys(
            row,
            ("allowed_transition_ids", "callsite_rule", "emission", "mode_id", "name", "selection_cases"),
            f"mode_rules[{index}]",
        )
        ids = row["allowed_transition_ids"]
        if not isinstance(ids, list) or any(
            _uint(item, 16, "allowed_transition_id", nonzero=True) not in transition_ids
            for item in ids
        ):
            raise SemanticError("event_semantics.mode_rules: unknown transition ID")
        if ids != sorted(set(ids)):
            raise SemanticError("event_semantics.mode_rules: transition IDs must be sorted and unique")
        if row["mode_id"] in (0, 1, 2, 6) and ids:
            raise SemanticError("event_semantics.mode_rules: non-event mode has transitions")
        if row["mode_id"] in (3, 4, 5) and not ids:
            raise SemanticError("event_semantics.mode_rules: event mode lacks transitions")
        selections = row["selection_cases"]
        if not isinstance(selections, list):
            raise SemanticError("event_semantics.mode_rules: selection_cases array required")
        covered: list[int] = []
        for selection_index, selection in enumerate(selections):
            _exact_keys(
                selection,
                ("selection_cases", "transition_ids"),
                f"mode_rules[{index}].selection_cases[{selection_index}]",
            )
            names = selection["selection_cases"]
            selected_ids = selection["transition_ids"]
            if (
                not isinstance(names, list)
                or not names
                or names != sorted(set(names))
                or any(name not in expected_case_names for name in names)
            ):
                raise SemanticError("event_semantics.mode_rules: invalid selection-case names")
            if (
                not isinstance(selected_ids, list)
                or selected_ids != sorted(set(selected_ids))
                or any(item not in ids for item in selected_ids)
            ):
                raise SemanticError("event_semantics.mode_rules: invalid selection transition IDs")
            covered.extend(selected_ids)
        if row["mode_id"] in (3, 4, 5) and sorted(set(covered)) != ids:
            raise SemanticError("event_semantics.mode_rules: every allowed transition needs a selection mapping")
        observed_modes.append(
            (
                row["mode_id"], row["name"], row["emission"], row["callsite_rule"],
                tuple(ids),
                tuple(
                    (tuple(mapping["selection_cases"]), tuple(mapping["transition_ids"]))
                    for mapping in selections
                ),
            )
        )
    sample_successor = (
        101, 103, 202, 204, 301, 303, 503, 508, 601, 603, 703, 704,
        801, 803, 901, 903, 1001, 1003, 1101, 1201,
    )
    sample_predecessor = (304, 401, 402, 504)
    sample_both = (306, 308, 506, 510)
    sample_all = tuple(sorted(sample_successor + sample_predecessor + sample_both))
    exact_transition_ids = tuple(item for item in EXPECTED_TRANSITION_IDS if item != 402)
    if observed_modes != [
        (0, "absent", "none", "canonical_zero", (), ()),
        (1, "present_off", "none", "canonical_zero", (), ()),
        (2, "counters", "none", "canonical_zero", (), ()),
        (
            3,
            "sampled",
            "sampled_selected_successor_or_ended_selected_predecessor",
            "canonical_zero",
            sample_all,
            (
                (("successor_only",), sample_successor),
                (("predecessor_cross", "predecessor_same"), sample_predecessor),
                (("both_cross", "both_same", "predecessor_cross", "predecessor_same", "successor_only"), sample_both),
            ),
        ),
        (4, "exact_events", "every_completed_supported_admitted_transaction", "canonical_zero", exact_transition_ids, ((("none",), exact_transition_ids),)),
        (5, "exact_callsite", "every_completed_supported_admitted_transaction", "nonzero_resolved", exact_transition_ids, ((("none",), exact_transition_ids),)),
        (6, "diagnostic_stacks", "outside_first_slice", "canonical_zero", (), ()),
    ]:
        raise SemanticError("event_semantics.mode_rules: exact A00-A06 truth table differs")
    expected_fields = [
        {"field": "address_after", "invalid_sentinel": 0, "nullable": False, "type": "u64", "validity_flag": "address_after_valid"},
        {"field": "address_before", "invalid_sentinel": 0, "nullable": False, "type": "u64", "validity_flag": "address_before_valid"},
        {"field": "arg0", "invalid_sentinel": 0, "nullable": False, "type": "u64", "validity_flag": "arg0_valid"},
        {"field": "arg1", "invalid_sentinel": 0, "nullable": False, "type": "u64", "validity_flag": "arg1_valid"},
        {"field": "arg2", "invalid_sentinel": 0, "nullable": False, "type": "u64", "validity_flag": "arg2_valid"},
        {"field": "callsite_id", "maximum": UINT32_MAX, "minimum": 0, "nullable": False, "resolution": "zero_unavailable_nonzero_exactly_once", "type": "u32"},
        {"field": "context_id", "maximum": UINT32_MAX, "minimum": 0, "nullable": False, "resolution": "zero_no_context_nonzero_same_generation_exactly_once", "type": "u32"},
        {"field": "counter_enter", "invalid_sentinel": 0, "nullable": False, "type": "u64", "validity_flag": "counter_enter_valid"},
        {"field": "counter_exit", "invalid_sentinel": 0, "nullable": False, "type": "u64", "validity_flag": "counter_exit_valid"},
        {"field": "cpu_enter", "invalid_sentinel": UINT16_MAX, "maximum_valid": UINT16_MAX - 1, "nullable": False, "type": "u16", "validity_flag": "cpu_enter_valid"},
        {"field": "cpu_exit", "invalid_sentinel": UINT16_MAX, "maximum_valid": UINT16_MAX - 1, "nullable": False, "type": "u16", "validity_flag": "cpu_exit_valid"},
        {"field": "event_kind", "maximum": UINT16_MAX, "minimum": 1, "nullable": False, "type": "u16"},
        {"field": "flags", "maximum": UINT32_MAX, "minimum": 0, "nullable": False, "type": "u32"},
        {"exact": 0, "field": "reserved_zero", "nullable": False, "type": "u32"},
        {"field": "result_code", "maximum": INT32_MAX, "minimum": INT32_MIN, "nullable": False, "semantics": "errno_for_pointer_return_apis_or_direct_return_for_posix_memalign_captured_immediately_after_real_call_and_restored_before_return", "type": "i32"},
        {"field": "thread_index", "maximum": UINT32_MAX - 1, "minimum": 1, "nullable": False, "reuse": "forbidden_within_process_generation", "type": "u32"},
        {"field": "thread_sequence", "maximum": UINT64_MAX, "minimum": 1, "nullable": False, "progression": "increments_once_for_every_supported_active_admitted_transaction_even_if_not_emitted_no_wrap", "type": "u64"},
    ]
    if value["field_rules"] != expected_fields:
        raise SemanticError("event_semantics.field_rules: exact field schemas differ")
    if sha256_hex(canonical_bytes(dict(value))) != EVENT_SEMANTICS_SHA256:
        raise SemanticError("event_semantics: exact frozen v1.1 definition differs")


def validate_api_catalog(value: Mapping[str, Any], event_digest: str) -> None:
    _exact_keys(value, ("catalog_id", "entries", "schema", "version"), "api")
    if value["catalog_id"] != "oai_memprof_api":
        raise SemanticError("api.catalog_id: exact ID required")
    _event_api_version(value["version"], "api.version")
    _binding(
        value["schema"],
        "api.schema",
        object_type=3,
        path="definition/event-semantics-v1.json",
    )
    if value["schema"]["sha256"] != event_digest:
        raise SemanticError("api.schema.sha256: does not bind exact event-semantics bytes")
    expected = [
        (1, 1, "malloc", "__real_malloc", "errno", "__wrap_malloc"),
        (2, 1, "calloc", "__real_calloc", "errno", "__wrap_calloc"),
        (3, 2, "realloc", "__real_realloc", "errno", "__wrap_realloc"),
        (4, 3, "free", "__real_free", "errno", "__wrap_free"),
        (5, 2, "reallocarray", "__real_reallocarray", "errno", "__wrap_reallocarray"),
        (6, 1, "aligned_alloc", "__real_aligned_alloc", "errno", "__wrap_aligned_alloc"),
        (7, 1, "posix_memalign", "__real_posix_memalign", "direct_return", "__wrap_posix_memalign"),
        (8, 1, "memalign", "__real_memalign", "errno", "__wrap_memalign"),
        (9, 1, "valloc", "__real_valloc", "errno", "__wrap_valloc"),
        (10, 1, "pvalloc", "__real_pvalloc", "errno", "__wrap_pvalloc"),
        (11, 1, "strdup", "__real_strdup", "errno", "__wrap_strdup"),
        (12, 1, "strndup", "__real_strndup", "errno", "__wrap_strndup"),
    ]
    entries = value["entries"]
    if not isinstance(entries, list) or len(entries) != len(expected):
        raise SemanticError("api.entries: exactly twelve rows required")
    observed = []
    for index, row in enumerate(entries):
        _exact_keys(
            row,
            ("api_id", "event_kind", "name", "real_symbol", "result_kind", "wrapper_symbol"),
            f"api.entries[{index}]",
        )
        observed.append(
            (
                _uint(row["api_id"], 16, "api_id", nonzero=True),
                _uint(row["event_kind"], 16, "event_kind", nonzero=True),
                _identifier(row["name"], "name"),
                _string(row["real_symbol"], "real_symbol", maximum_bytes=63),
                _identifier(row["result_kind"], "result_kind"),
                _string(row["wrapper_symbol"], "wrapper_symbol", maximum_bytes=63),
            )
        )
    if observed != expected:
        raise SemanticError("api.entries: frozen first-slice rows differ")
    if sha256_hex(canonical_bytes(dict(value))) != API_CATALOG_SHA256:
        raise SemanticError("api: exact frozen v1.1 catalog differs")


def validate_callsite_definition(value: Mapping[str, Any]) -> None:
    _exact_keys(value, ("catalog_schema", "definition_id", "mode_rules", "version"), "callsite")
    validate_definition_envelope(value, "oai_memprof_callsite_rule", "callsite")
    modes = value["mode_rules"]
    if modes != [
        {"callsite_id": "canonical_zero", "mode_id": 0},
        {"callsite_id": "canonical_zero", "mode_id": 1},
        {"callsite_id": "canonical_zero", "mode_id": 2},
        {"callsite_id": "canonical_zero", "mode_id": 3},
        {"callsite_id": "canonical_zero", "mode_id": 4},
        {"callsite_id": "nonzero_resolved", "mode_id": 5},
        {"callsite_id": "canonical_zero", "mode_id": 6},
    ]:
        raise SemanticError("callsite.mode_rules: exact A00..A06 rules required")
    schema = value["catalog_schema"]
    if not isinstance(schema, dict):
        raise SemanticError("callsite.catalog_schema: object required")
    _exact_keys(schema, ("row_fields", "sort", "top_keys", "uniqueness"), "callsite.catalog_schema")
    if schema["top_keys"] != ["catalog_id", "entries", "schema", "version"]:
        raise SemanticError("callsite.catalog_schema.top_keys: exact envelope required")
    if schema["sort"] != ["process_generation", "callsite_id"] or schema["uniqueness"] != ["process_generation", "callsite_id"]:
        raise SemanticError("callsite.catalog_schema: exact order/key required")
    if schema["row_fields"] != [
        {"maximum": UINT32_MAX, "minimum": 1, "name": "callsite_id", "nullable": False, "type": "u32"},
        {"maximum": UINT64_MAX, "minimum": 1, "name": "module_generation", "nullable": False, "type": "u64"},
        {"maximum": UINT32_MAX, "minimum": 1, "name": "module_id", "nullable": False, "type": "u32"},
        {"maximum": UINT64_MAX, "minimum": 1, "name": "process_generation", "nullable": False, "type": "u64"},
        {"maximum": UINT64_MAX, "minimum": 0, "name": "raw_address", "nullable": False, "type": "u64"},
    ]:
        raise SemanticError("callsite.catalog_schema.row_fields: exact schemas differ")


def validate_phase_definition(value: Mapping[str, Any]) -> None:
    _exact_keys(value, ("definition_id", "entries", "row_schema", "sort", "uniqueness", "version"), "phase")
    validate_definition_envelope(value, "oai_memprof_phase", "phase")
    if value["sort"] != ["phase_id"] or value["uniqueness"] != ["phase_id"]:
        raise SemanticError("phase: phase_id order/key required")
    if value["entries"] != [
        {"description": "explicitly unspecified; never inferred", "name": "unspecified", "parent_phase_id": None, "phase_id": 0}
    ]:
        raise SemanticError("phase.entries: v1.0 freezes only phase zero")
    if value["row_schema"] != [
        {"maximum_utf8_bytes": 256, "minimum_utf8_bytes": 1, "name": "description", "nullable": False, "type": "utf8_nfc"},
        {"maximum_utf8_bytes": 63, "name": "name", "nullable": False, "pattern": "[a-z][a-z0-9_]*", "type": "ascii_identifier"},
        {"maximum": UINT32_MAX, "minimum": 1, "name": "parent_phase_id", "nullable": True, "type": "u32"},
        {"maximum": UINT32_MAX, "minimum": 0, "name": "phase_id", "nullable": False, "type": "u32"},
    ]:
        raise SemanticError("phase.row_schema: exact sorted fields required")


def validate_context_schema(value: Mapping[str, Any], phase_digest: str) -> None:
    _exact_keys(
        value,
        ("catalog_schema", "definition_id", "phase_definition", "record_binding", "version"),
        "context",
    )
    validate_definition_envelope(value, "oai_memprof_context_schema", "context")
    _binding(value["phase_definition"], "context.phase_definition", object_type=7, path="definition/phase-v1.json")
    if value["phase_definition"]["sha256"] != phase_digest:
        raise SemanticError("context.phase_definition.sha256: does not bind exact phase bytes")
    schema = value["catalog_schema"]
    if not isinstance(schema, dict):
        raise SemanticError("context.catalog_schema: object required")
    _exact_keys(schema, ("row_fields", "sort", "top_keys", "uniqueness"), "context.catalog_schema")
    if schema["top_keys"] != ["catalog_id", "entries", "schema", "version"]:
        raise SemanticError("context.catalog_schema.top_keys: exact envelope required")
    if schema["sort"] != ["process_generation", "context_id"] or schema["uniqueness"] != [
        "process_generation",
        "context_id",
    ]:
        raise SemanticError("context.catalog_schema: exact order/key required")
    if schema["row_fields"] != [
        {"maximum": UINT32_MAX, "minimum": 1, "name": "context_id", "nullable": False, "type": "u32"},
        {"maximum": UINT32_MAX, "minimum": 0, "name": "phase_id", "nullable": False, "type": "u32"},
        {"maximum": UINT64_MAX, "minimum": 1, "name": "process_generation", "nullable": False, "type": "u64"},
    ]:
        raise SemanticError("context.catalog_schema.row_fields: exact schemas differ")
    binding = value["record_binding"]
    if binding != {
        "context_id_zero": "sole_no_context_representation_no_catalog_row",
        "nonzero_context_id": "resolve_exactly_once_in_same_process_generation",
        "nonzero_phase_id": "resolve_exactly_once_in_bound_phase_definition",
        "phase_inference": "forbidden",
    }:
        raise SemanticError("context.record_binding: exact rules required")


DIAGNOSTIC_CLASS_FLAG_NAMES = (
    "event_loss",
    "unaccounted_bypass",
    "intentional_outside_domain",
    "unknown_value",
    "sampled_membership",
    "writer_or_finalization",
    "counter_or_clock_invalid",
    "saturation",
    "exact_event_gate",
    "sampled_gate",
)

DIAGNOSTIC_REASON_ROWS = (
    (
        1,
        "ring_full",
        0x0101,
        "exact_event",
        "completed_supported_admitted_transaction_not_emitted_because_producer_ring_full",
    ),
    (
        2,
        "unregistered_active_thread",
        0x0101,
        "exact_event",
        "active_supported_call_without_ready_producer_for_non_capacity_registration_or_invariant_cause",
    ),
    (
        3,
        "registration_capacity_failure",
        0x0101,
        "exact_event",
        "active_supported_first_use_registration_reservation_failed_because_preallocated_capacity_exhausted",
    ),
    (
        16,
        "recursion_bypass",
        0x0102,
        "exact_event",
        "active_supported_call_bypassed_because_profiler_recursion_was_detected",
    ),
    (
        17,
        "profiler_internal_bypass",
        0x0004,
        "domain_disclosure",
        "operation_matches_exact_static_catalog_profiler_internal_exclusion",
    ),
    (
        18,
        "unsupported_api_or_domain",
        0x0004,
        "denominator_domain",
        "operation_matches_exact_coverage_policy_known_unsupported_api_or_domain",
    ),
    (
        32,
        "size_unknown",
        0x0008,
        "requested_byte_estimands",
        "requested_or_size_dependent_value_unavailable",
    ),
    (
        48,
        "sample_membership_insertion_failure",
        0x0210,
        "sampled_lifetime",
        "selected_live_instance_insertion_not_committed_including_capacity_exhaustion_after_conclusive_search",
    ),
    (
        49,
        "membership_lookup_failure",
        0x0210,
        "sampled_lifetime",
        "conclusive_bounded_lookup_did_not_resolve_required_selected_predecessor",
    ),
    (
        50,
        "bounded_probe_exhaustion",
        0x0210,
        "sampled_lifetime",
        "bounded_collision_or_probe_budget_exhausted_before_conclusive_insertion_or_lookup",
    ),
    (
        51,
        "invalid_or_ambiguous_pointer_pairing",
        0x0210,
        "sampled_lifetime",
        "completed_lookup_found_invalid_or_ambiguous_selected_pointer_instance_transition",
    ),
    (
        64,
        "clock_regression_or_invalid_counter",
        0x0140,
        "timing_lifetime",
        "clock_regression_or_invalid_counter_observed",
    ),
    (
        80,
        "writer_io_or_finalization_failure",
        0x0120,
        "stream_completeness",
        "writer_io_or_finalization_failure_observed",
    ),
    (
        96,
        "diagnostic_counter_saturation",
        0x0180,
        "affected_exact_total",
        "first_unsaturated_to_saturated_transition_of_each_other_authoritative_counter_instance",
    ),
)

DIAGNOSTIC_SCOPE_ROWS = (
    (1, "producer_thread", "stable_nonzero_thread_index_of_final_ready_slot_in_generation"),
    (2, "registration", "exactly_1"),
    (3, "writer", "exactly_1"),
    (4, "diagnostic_aggregate", "exactly_1"),
)

DIAGNOSTIC_MODE_SCOPE_ROWS = (
    (
        2,
        "counters",
        (96,),
        (1, 48, 49, 50, 51, 64),
        (16, 17, 18, 32),
        (2, 3),
        (80,),
    ),
    (
        3,
        "sampled",
        (96,),
        (),
        (1, 16, 17, 18, 32, 48, 49, 50, 51, 64),
        (2, 3),
        (80,),
    ),
    (
        4,
        "exact_events",
        (96,),
        (48, 49, 50, 51),
        (1, 16, 17, 18, 32, 64),
        (2, 3),
        (80,),
    ),
    (
        5,
        "exact_callsite",
        (96,),
        (48, 49, 50, 51),
        (1, 16, 17, 18, 32, 64),
        (2, 3),
        (80,),
    ),
)


def _expected_diagnostic_definition() -> dict[str, Any]:
    return {
        "class_flag_bits": [
            {"bit": bit, "mask": 1 << bit, "name": name}
            for bit, name in enumerate(DIAGNOSTIC_CLASS_FLAG_NAMES)
        ],
        "complete_capable": True,
        "counter_row_schema": {
            "array_order": "process_generation_counter_scope_kind_counter_scope_id_reason_id_strictly_increasing",
            "fields": [
                {
                    "maximum": UINT32_MAX,
                    "minimum": 1,
                    "name": "counter_scope_id",
                    "nullable": False,
                    "type": "u32",
                },
                {
                    "maximum": 4,
                    "minimum": 1,
                    "name": "counter_scope_kind",
                    "nullable": False,
                    "type": "u16",
                },
                {
                    "maximum": UINT64_MAX,
                    "minimum": 1,
                    "name": "process_generation",
                    "nullable": False,
                    "type": "u64",
                },
                {
                    "maximum": UINT16_MAX,
                    "minimum": 1,
                    "name": "reason_id",
                    "nullable": False,
                    "type": "u16",
                },
                {"name": "saturated", "nullable": False, "type": "boolean"},
                {
                    "maximum": UINT64_MAX,
                    "minimum": 0,
                    "name": "value",
                    "nullable": False,
                    "type": "u64",
                },
            ],
            "uniqueness": [
                "process_generation",
                "counter_scope_kind",
                "counter_scope_id",
                "reason_id",
            ],
        },
        "counter_scope_kinds": [
            {
                "counter_scope_id_rule": scope_id_rule,
                "name": name,
                "scope_kind": scope_kind,
            }
            for scope_kind, name, scope_id_rule in DIAGNOSTIC_SCOPE_ROWS
        ],
        "definition_id": "oai_memprof_diagnostic",
        "mode_scope_arrays": [
            {
                "diagnostic_aggregate_reason_ids": list(aggregate),
                "mode_id": mode_id,
                "mode_name": mode_name,
                "not_applicable_reason_ids": list(not_applicable),
                "producer_thread_reason_ids": list(producer),
                "registration_reason_ids": list(registration),
                "writer_reason_ids": list(writer),
            }
            for (
                mode_id,
                mode_name,
                aggregate,
                not_applicable,
                producer,
                registration,
                writer,
            ) in DIAGNOSTIC_MODE_SCOPE_ROWS
        ],
        "object_schema": {
            "entry_count_rule": "length_of_entries_only_reason_totals_excluded",
            "fields": [
                {
                    "name": "catalog_id",
                    "nullable": False,
                    "type": "literal:oai_memprof_diagnostics",
                },
                {"name": "entries", "nullable": False, "type": "array:counter_row"},
                {"name": "mode_id", "nullable": False, "type": "enum:2_3_4_5"},
                {
                    "name": "process_generation",
                    "nullable": False,
                    "type": "u64_nonzero",
                },
                {
                    "name": "reason_totals",
                    "nullable": False,
                    "type": "array:reason_total",
                },
                {
                    "name": "schema",
                    "nullable": False,
                    "type": "definition_binding:self_object_type_8_path_definition_diagnostic_v1",
                },
                {"name": "version", "nullable": False, "type": "version_1_0"},
            ],
            "process_generation_rule": "top_level_value_equals_every_counter_row",
            "unknown_fields_allowed": False,
        },
        "projection_rules": {
            "complete": {
                "counter_rows": "exact_mode_population_no_missing_duplicate_or_extra_scope_reason_pair",
                "empty_population": "schema_known_empty_set_has_explicit_zero_reason_total",
                "partial_counter_population": "forbidden",
                "reason_totals": "exactly_14_in_increasing_reason_id_order",
            },
            "id96": {
                "complete_reconciliation": "stored_value_equals_sum_of_reason_total_saturated_counter_instances",
                "partial_reconciliation": "not_asserted_both_values_are_independent_lower_bounds",
                "self_saturation": "forbidden",
                "source_population": "first_saturation_transition_of_every_other_authoritative_counter_instance",
            },
            "partial": {
                "counter_rows": "available_rows_only_missing_required_instance_or_value_is_never_implicit_zero",
                "reason_totals": "exactly_14_available_lower_bound_projections_in_increasing_reason_id_order",
                "summary_flag": "partial_counter_population_set_iff_required_instance_or_value_unavailable",
            },
            "reason_total": {
                "class_flags": "exact_reason_entry_mask",
                "nonzero_counter_instances": "count_available_contributing_values_greater_than_zero",
                "saturated_counter_instances": "count_available_contributing_counters_with_saturated_true",
                "saturating_total": "mathematical_sum_of_available_stored_values_clamped_to_uint64_max",
                "total_saturated": "set_iff_contributing_counter_saturated_or_available_unsaturated_sum_exceeds_uint64_max",
            },
            "terminal": {
                "bypass_sum": "saturating_sum_of_complete_reason_totals_with_class_bit_1",
                "entry_projection": "every_terminal_reason_entry_exactly_equals_corresponding_reason_total",
                "loss_sum": "saturating_sum_of_complete_reason_totals_with_class_bit_0",
                "partial_saturated_instances": "sum_of_available_reason_total_saturated_counter_instances",
                "reserved_zero": "exact_zero",
            },
        },
        "reason_entries": [
            {
                "claim_gate": claim_gate,
                "class_flags": class_flags,
                "increment_rule": increment_rule,
                "name": name,
                "reason_id": reason_id,
            }
            for reason_id, name, class_flags, claim_gate, increment_rule in DIAGNOSTIC_REASON_ROWS
        ],
        "reason_total_schema": {
            "array_order": "reason_id_strictly_increasing",
            "count": 14,
            "fields": [
                {
                    "maximum": 1023,
                    "minimum": 0,
                    "name": "class_flags",
                    "nullable": False,
                    "type": "u16",
                },
                {
                    "maximum": UINT32_MAX,
                    "minimum": 0,
                    "name": "nonzero_counter_instances",
                    "nullable": False,
                    "type": "u32",
                },
                {
                    "maximum": UINT16_MAX,
                    "minimum": 1,
                    "name": "reason_id",
                    "nullable": False,
                    "type": "u16",
                },
                {
                    "maximum": UINT32_MAX,
                    "minimum": 0,
                    "name": "saturated_counter_instances",
                    "nullable": False,
                    "type": "u32",
                },
                {
                    "maximum": UINT64_MAX,
                    "minimum": 0,
                    "name": "saturating_total",
                    "nullable": False,
                    "type": "u64",
                },
                {
                    "maximum": 3,
                    "minimum": 0,
                    "name": "summary_flags",
                    "nullable": False,
                    "type": "u32",
                },
            ],
            "relational_rule": "saturated_counter_instances_not_greater_than_nonzero_counter_instances",
            "uniqueness": ["reason_id"],
        },
        "scope_population_rules": {
            "diagnostic_aggregate": "singleton_scope_kind_4_scope_id_1",
            "not_ready_reservation": "excluded_and_no_thread_identity_fabricated",
            "producer_thread": "one_per_final_ready_slot_below_reservation_high_water_for_active_generation",
            "registration": "singleton_scope_kind_2_scope_id_1",
            "unknown_required_population": "partial_counter_population",
            "writer": "singleton_scope_kind_3_scope_id_1",
        },
        "summary_flag_bits": [
            {"bit": 0, "mask": 1, "name": "total_saturated"},
            {"bit": 1, "mask": 2, "name": "partial_counter_population"},
        ],
        "version": VERSION,
    }


def _require_exact_literal(value: Any, expected: Any, where: str) -> None:
    if type(value) is not type(expected):
        raise SemanticError(f"{where}: exact type differs")
    if isinstance(expected, dict):
        if set(value) != set(expected):
            raise SemanticError(f"{where}: exact keys differ")
        for key in expected:
            _require_exact_literal(value[key], expected[key], f"{where}.{key}")
    elif isinstance(expected, list):
        if len(value) != len(expected):
            raise SemanticError(f"{where}: exact array length differs")
        for index, (item, expected_item) in enumerate(zip(value, expected)):
            _require_exact_literal(item, expected_item, f"{where}[{index}]")
    elif value != expected:
        raise SemanticError(f"{where}: exact value differs")


def validate_diagnostic_definition(value: Mapping[str, Any]) -> None:
    _require_exact_literal(value, _expected_diagnostic_definition(), "diagnostic")


VALIDATORS: dict[int, Callable[[Mapping[str, Any]], None]] = {
    1: validate_canonical_definition,
    2: validate_schema_bundle_schema,
    3: validate_event_semantics,
    5: validate_callsite_definition,
    7: validate_phase_definition,
    8: validate_diagnostic_definition,
}

_COVERAGE_MODULE: Any | None = None
_CONFIG_MODULE: Any | None = None
_SAMPLING_MODULE: Any | None = None
_RUNTIME_MODULE: Any | None = None


def _load_coverage_module() -> Any:
    """Load the adjacent standard-library-only coverage validator lazily."""

    global _COVERAGE_MODULE
    if _COVERAGE_MODULE is not None:
        return _COVERAGE_MODULE
    path = Path(__file__).resolve().parent.parent / "coverage" / "coverage_catalog_v1.py"
    spec = importlib.util.spec_from_file_location("_oai_memprof_coverage_catalog_v1", path)
    if spec is None or spec.loader is None:
        raise SemanticError("coverage: adjacent validator module is unavailable")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise SemanticError("coverage: adjacent validator module failed to load") from error
    _COVERAGE_MODULE = module
    return module


def _validate_coverage_member(object_type: int, raw: bytes) -> None:
    """Require exact adjacent coverage-owned literal bytes for type 9 or 10."""

    if object_type not in (9, 10):
        raise SemanticError("coverage: only object_type 9 or 10 is coverage-owned")
    registry_path = MEMBER_REGISTRY[object_type - 1][2]
    coverage = _load_coverage_module()
    adjacent_paths = {
        9: coverage.POLICY_ARCHIVE_PATH,
        10: coverage.INSTANCE_SCHEMA_ARCHIVE_PATH,
    }
    if adjacent_paths[object_type] != registry_path:
        raise SemanticError(f"bundle.member[{object_type}]: coverage registry path mismatch")
    try:
        coverage.validate_static_member(registry_path, raw)
    except Exception as error:
        raise SemanticError(
            f"bundle.member[{object_type}]: exact coverage literal validation failed"
        ) from error


def _load_config_module() -> Any:
    """Load the adjacent standard-library-only effective-config validator lazily."""

    global _CONFIG_MODULE
    if _CONFIG_MODULE is not None:
        return _CONFIG_MODULE
    path = Path(__file__).resolve().parent.parent / "config" / "effective_config_v1.py"
    spec = importlib.util.spec_from_file_location("_oai_memprof_effective_config_v1", path)
    if spec is None or spec.loader is None:
        raise SemanticError("config: adjacent validator module is unavailable")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise SemanticError("config: adjacent validator module failed to load") from error
    _CONFIG_MODULE = module
    return module


def _validate_config_member(raw: bytes) -> None:
    """Require the exact config-owned member-11 proposal and literal bytes."""

    config = _load_config_module()
    registry = MEMBER_REGISTRY[10]
    expected_member = {
        "name": registry[1],
        "object_type": registry[0],
        "owner": registry[3],
        "path": registry[2],
    }
    if config.BUNDLE_MEMBER_PROPOSAL != expected_member:
        raise SemanticError("bundle.member[11]: config member proposal differs")
    relation = next(
        row for row in CROSS_HASH_RELATIONS
        if row[0] == "effective_configuration_schema"
    )
    expected_relation = {
        "left": relation[1],
        "relation_id": relation[0],
        "right": relation[2],
    }
    if config.BUNDLE_CROSS_RELATION_PROPOSAL != expected_relation:
        raise SemanticError("bundle.member[11]: config cross-relation proposal differs")
    expected_entry = {
        "bytes": len(raw),
        "object_type": registry[0],
        "path": registry[2],
        "sha256": sha256_hex(raw),
    }
    if config.BUNDLE_ENTRY_PROPOSAL != expected_entry:
        raise SemanticError("bundle.member[11]: config entry proposal differs")
    try:
        value = parse_canonical(raw)
        if raw != config.SCHEMA_BYTES:
            raise SemanticError("config literal bytes differ")
        config.validate_schema_definition(value)
    except Exception as error:
        raise SemanticError("bundle.member[11]: exact config literal validation failed") from error


def _load_sampling_module() -> Any:
    """Load the adjacent standard-library-only sampled-selection owner lazily."""

    global _SAMPLING_MODULE
    if _SAMPLING_MODULE is not None:
        return _SAMPLING_MODULE
    path = Path(__file__).resolve().parent.parent / "sampling" / "selection_rule_v1.py"
    spec = importlib.util.spec_from_file_location("_oai_memprof_selection_rule_v1", path)
    if spec is None or spec.loader is None:
        raise SemanticError("sampling: adjacent validator module is unavailable")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise SemanticError("sampling: adjacent validator module failed to load") from error
    _SAMPLING_MODULE = module
    return module


def _validate_sampling_member(raw: bytes) -> None:
    """Require the exact sampling-owned member-12 proposal and literal bytes."""

    sampling = _load_sampling_module()
    registry = MEMBER_REGISTRY[11]
    expected_member = {
        "name": registry[1],
        "object_type": registry[0],
        "owner": registry[3],
        "path": registry[2],
    }
    if sampling.BUNDLE_MEMBER_PROPOSAL != expected_member:
        raise SemanticError("bundle.member[12]: sampling member proposal differs")
    expected_entry = {
        "bytes": len(raw),
        "object_type": registry[0],
        "path": registry[2],
        "sha256": sha256_hex(raw),
    }
    if sampling.BUNDLE_ENTRY_PROPOSAL != expected_entry:
        raise SemanticError("bundle.member[12]: sampling entry proposal differs")
    try:
        value = parse_canonical(raw)
        if raw != sampling.definition_bytes():
            raise SemanticError("sampling literal bytes differ")
        sampling.validate_definition(value)
    except Exception as error:
        raise SemanticError("bundle.member[12]: exact sampling literal validation failed") from error


def _load_runtime_module() -> Any:
    """Load the adjacent standard-library-only runtime-catalog owner lazily."""

    global _RUNTIME_MODULE
    if _RUNTIME_MODULE is not None:
        return _RUNTIME_MODULE
    path = Path(__file__).resolve().parent.parent / "runtime" / "runtime_catalog_v1.py"
    spec = importlib.util.spec_from_file_location("_oai_memprof_runtime_catalog_v1", path)
    if spec is None or spec.loader is None:
        raise SemanticError("runtime: adjacent validator module is unavailable")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise SemanticError("runtime: adjacent validator module failed to load") from error
    _RUNTIME_MODULE = module
    return module


def _validate_runtime_member(raw: bytes) -> None:
    """Require the exact runtime-owned member-13 proposals and literal bytes."""

    runtime = _load_runtime_module()
    registry = MEMBER_REGISTRY[12]
    expected_member = {
        "name": registry[1],
        "object_type": registry[0],
        "owner": registry[3],
        "path": registry[2],
    }
    if runtime.BUNDLE_MEMBER_PROPOSAL != expected_member:
        raise SemanticError("bundle.member[13]: runtime member proposal differs")
    expected_relations = tuple(
        {
            "left": relation[1],
            "relation_id": relation[0],
            "right": relation[2],
        }
        for relation in CROSS_HASH_RELATIONS
        if relation[0] in {
            "clock_catalog_schema", "module_catalog_schema", "thread_catalog_schema"
        }
    )
    if runtime.BUNDLE_CROSS_RELATION_PROPOSALS != expected_relations:
        raise SemanticError("bundle.member[13]: runtime cross-relation proposals differ")
    expected_entry = {
        "bytes": len(raw),
        "object_type": registry[0],
        "path": registry[2],
        "sha256": sha256_hex(raw),
    }
    if runtime.BUNDLE_ENTRY_PROPOSAL != expected_entry:
        raise SemanticError("bundle.member[13]: runtime entry proposal differs")
    try:
        if raw != runtime.DEFINITION_BYTES:
            raise SemanticError("runtime literal bytes differ")
        runtime.validate_definition_bytes(raw)
    except Exception as error:
        raise SemanticError("bundle.member[13]: exact runtime literal validation failed") from error


def validate_semantic_root(root: Path) -> dict[int, tuple[Path, bytes]]:
    """Validate semantic-owned static members and all internal hash links."""

    archive = root / "archive"
    loaded: dict[int, tuple[Path, bytes]] = {}
    values: dict[int, dict[str, Any]] = {}
    for object_type, _name, archive_path, owner in MEMBER_REGISTRY:
        if owner != "semantic":
            continue
        path = archive / archive_path
        value, raw = load_canonical(path)
        loaded[object_type] = (path, raw)
        values[object_type] = value
        validator = VALIDATORS.get(object_type)
        if validator is not None:
            validator(value)
    event_digest = sha256_hex(loaded[3][1])
    validate_api_catalog(values[4], event_digest)
    phase_digest = sha256_hex(loaded[7][1])
    validate_context_schema(values[6], phase_digest)
    return loaded


def _validate_semantic_owned_members(
    parsed: Mapping[int, Mapping[str, Any]], raws: Mapping[int, bytes]
) -> None:
    """Apply the complete semantic-owned validation and digest bindings."""

    validate_canonical_definition(parsed[1])
    validate_schema_bundle_schema(parsed[2])
    validate_event_semantics(parsed[3])
    validate_api_catalog(parsed[4], sha256_hex(raws[3]))
    validate_callsite_definition(parsed[5])
    validate_context_schema(parsed[6], sha256_hex(raws[7]))
    validate_phase_definition(parsed[7])
    validate_diagnostic_definition(parsed[8])


def build_bundle(member_files: Mapping[int, Path]) -> bytes:
    """Build a complete thirteen-member v1.4 bundle and return canonical bytes."""

    expected_ids = {row[0] for row in MEMBER_REGISTRY}
    if set(member_files) != expected_ids:
        raise SemanticError("bundle: exact object_type set 1..13 required")
    entries = []
    registry_by_id = {row[0]: row for row in MEMBER_REGISTRY}
    parsed: dict[int, dict[str, Any]] = {}
    raws: dict[int, bytes] = {}
    for object_type in sorted(member_files):
        name_path = registry_by_id[object_type]
        path = member_files[object_type]
        value, raw = load_canonical(path)
        parsed[object_type] = value
        raws[object_type] = raw
        entries.append(
            {
                "bytes": len(raw),
                "object_type": object_type,
                "path": name_path[2],
                "sha256": sha256_hex(raw),
            }
        )
    _validate_semantic_owned_members(parsed, raws)
    _validate_coverage_member(9, raws[9])
    _validate_coverage_member(10, raws[10])
    _validate_config_member(raws[11])
    _validate_sampling_member(raws[12])
    _validate_runtime_member(raws[13])
    schema_binding = {
        "object_type": 2,
        "path": registry_by_id[2][2],
        "sha256": sha256_hex(raws[2]),
    }
    return canonical_bytes(
        {
            "catalog_id": "oai_memprof_schema_bundle",
            "entries": entries,
            "schema": schema_binding,
            "version": BUNDLE_VERSION,
        }
    )


def validate_bundle(raw: bytes, member_bytes: Mapping[int, bytes]) -> dict[str, Any]:
    expected_ids = {row[0] for row in MEMBER_REGISTRY}
    if set(member_bytes) != expected_ids:
        raise SemanticError("bundle.member_bytes: exact object_type set 1..13 required")
    value = parse_canonical(raw)
    _exact_keys(value, ("catalog_id", "entries", "schema", "version"), "bundle")
    if value["catalog_id"] != "oai_memprof_schema_bundle":
        raise SemanticError("bundle.catalog_id: exact ID required")
    _bundle_version(value["version"], "bundle.version")
    _binding(value["schema"], "bundle.schema", object_type=2, path=MEMBER_REGISTRY[1][2])
    entries = value["entries"]
    if not isinstance(entries, list) or len(entries) != 13:
        raise SemanticError("bundle.entries: exact thirteen-member static bundle required")
    registry_by_id = {row[0]: row for row in MEMBER_REGISTRY}
    observed_ids = []
    for index, row in enumerate(entries):
        where = f"bundle.entries[{index}]"
        if not isinstance(row, dict):
            raise SemanticError(f"{where}: object required")
        _exact_keys(row, ("bytes", "object_type", "path", "sha256"), where)
        object_type = _uint(row["object_type"], 16, f"{where}.object_type", nonzero=True)
        observed_ids.append(object_type)
        registry = registry_by_id.get(object_type)
        if registry is None:
            raise SemanticError(f"{where}: unknown object_type")
        if _relative_path(row["path"], f"{where}.path") != registry[2]:
            raise SemanticError(f"{where}: registry path mismatch")
        raw_member = member_bytes.get(object_type)
        if raw_member is None:
            raise SemanticError(f"{where}: member bytes unavailable")
        if _uint(row["bytes"], 64, f"{where}.bytes", nonzero=True) != len(raw_member):
            raise SemanticError(f"{where}: byte count mismatch")
        if _hash(row["sha256"], f"{where}.sha256") != sha256_hex(raw_member):
            raise SemanticError(f"{where}: digest mismatch")
    if observed_ids != list(range(1, 14)):
        raise SemanticError("bundle.entries: exact ascending object_type sequence required")
    if value["schema"]["sha256"] != sha256_hex(member_bytes[2]):
        raise SemanticError("bundle.schema: schema hash mismatch")
    semantic_raws = {object_type: member_bytes[object_type] for object_type in range(1, 9)}
    semantic_values = {
        object_type: parse_canonical(member_raw)
        for object_type, member_raw in semantic_raws.items()
    }
    _validate_semantic_owned_members(semantic_values, semantic_raws)
    _validate_coverage_member(9, member_bytes[9])
    _validate_coverage_member(10, member_bytes[10])
    _validate_config_member(member_bytes[11])
    _validate_sampling_member(member_bytes[12])
    _validate_runtime_member(member_bytes[13])
    return value


def validate_context_catalog(value: Mapping[str, Any], *, context_schema_sha256: str, phase_ids: set[int]) -> None:
    _exact_keys(value, ("catalog_id", "entries", "schema", "version"), "context_catalog")
    if value["catalog_id"] != "oai_memprof_context":
        raise SemanticError("context_catalog.catalog_id: exact ID required")
    _version(value["version"], "context_catalog.version")
    _binding(value["schema"], "context_catalog.schema", object_type=6, path="definition/context-schema-v1.json")
    if value["schema"]["sha256"] != context_schema_sha256:
        raise SemanticError("context_catalog.schema: context-schema digest mismatch")
    entries = value["entries"]
    if not isinstance(entries, list):
        raise SemanticError("context_catalog.entries: array required")
    previous: tuple[int, int] | None = None
    for index, row in enumerate(entries):
        where = f"context_catalog.entries[{index}]"
        if not isinstance(row, dict):
            raise SemanticError(f"{where}: object required")
        _exact_keys(row, ("context_id", "phase_id", "process_generation"), where)
        context_id = _uint(row["context_id"], 32, f"{where}.context_id", nonzero=True)
        generation = _uint(row["process_generation"], 64, f"{where}.process_generation", nonzero=True)
        phase_id = _uint(row["phase_id"], 32, f"{where}.phase_id")
        if phase_id not in phase_ids:
            raise SemanticError(f"{where}.phase_id: unresolved phase")
        key = (generation, context_id)
        if previous is not None and key <= previous:
            raise SemanticError("context_catalog.entries: strict sorted uniqueness required")
        previous = key


def validate_event_flags(
    *,
    flags: int,
    required_one_mask: int,
    required_zero_mask: int,
    address_before: int,
    address_after: int,
    arg0: int,
    arg1: int,
    arg2: int,
    counter_enter: int,
    counter_exit: int,
    cpu_enter: int,
    cpu_exit: int,
    result_code: int,
) -> None:
    """Validate exact masks and canonical invalid-field sentinels for one composed variant."""

    flags = _uint(flags, 32, "event.flags")
    required_one_mask = _uint(required_one_mask, 32, "required_one_mask")
    required_zero_mask = _uint(required_zero_mask, 32, "required_zero_mask")
    if (
        required_one_mask & required_zero_mask
        or required_one_mask & ~KNOWN_FLAGS_MASK
        or required_zero_mask & ~KNOWN_FLAGS_MASK
        or (required_one_mask | required_zero_mask) & COMMON_EVIDENCE_MASK
        or (required_one_mask | required_zero_mask)
        != KNOWN_FLAGS_MASK & ~COMMON_EVIDENCE_MASK
    ):
        raise SemanticError(
            "event.flags: exact complementary variant masks required"
        )
    if flags & RESERVED_FLAGS_MASK:
        raise SemanticError("event.flags: reserved bit set")
    if flags & required_one_mask != required_one_mask or flags & required_zero_mask:
        raise SemanticError("event.flags: required-one/required-zero mismatch")
    if flags & RESULT_KIND_MASK not in (0x01000000, 0x02000000):
        raise SemanticError("event.flags: first slice requires ERRNO or direct-return result kind")
    for value, bit, where in (
        (address_before, 0, "address_before"),
        (address_after, 1, "address_after"),
        (arg0, 2, "arg0"),
        (arg1, 3, "arg1"),
        (arg2, 4, "arg2"),
        (counter_enter, 5, "counter_enter"),
        (counter_exit, 6, "counter_exit"),
    ):
        _uint(value, 64, f"event.{where}")
        if not flags & (1 << bit) and value != 0:
            raise SemanticError(f"event.{where}: invalid field must be canonical zero")
    for value, bit, where in ((cpu_enter, 7, "cpu_enter"), (cpu_exit, 8, "cpu_exit")):
        _uint(value, 16, f"event.{where}")
        if flags & (1 << bit):
            if value == UINT16_MAX:
                raise SemanticError(f"event.{where}: valid CPU cannot be 0xffff")
        elif value != UINT16_MAX:
            raise SemanticError(f"event.{where}: invalid CPU must be 0xffff")
    _i32(result_code, "event.result_code")


def _command_check(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    loaded = validate_semantic_root(root)
    for object_type in sorted(loaded):
        path, raw = loaded[object_type]
        print(f"{object_type:02d} {len(raw):8d} {sha256_hex(raw)} {path.relative_to(root)}")
    return 0


def _command_bundle(args: argparse.Namespace) -> int:
    semantic_root = Path(args.semantic_root).resolve()
    coverage_root = Path(args.coverage_root).resolve()
    config_root = Path(args.config_root).resolve()
    sampling_root = Path(args.sampling_root).resolve()
    runtime_root = Path(args.runtime_root).resolve()
    owner_roots = {
        "config": config_root,
        "coverage": coverage_root,
        "runtime": runtime_root,
        "sampling": sampling_root,
        "semantic": semantic_root,
    }
    files: dict[int, Path] = {}
    for object_type, _name, archive_path, owner in MEMBER_REGISTRY:
        owner_root = owner_roots[owner]
        files[object_type] = owner_root / "archive" / archive_path
    raw = build_bundle(files)
    sys.stdout.buffer.write(raw)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    check = subparsers.add_parser("check")
    check.add_argument("root")
    check.set_defaults(func=_command_check)
    bundle = subparsers.add_parser("bundle")
    bundle.add_argument("semantic_root")
    bundle.add_argument("coverage_root")
    bundle.add_argument("config_root")
    bundle.add_argument("sampling_root")
    bundle.add_argument("runtime_root")
    bundle.set_defaults(func=_command_bundle)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, SemanticError) as error:
        print(f"semantic-catalog-v1: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
