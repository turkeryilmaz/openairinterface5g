#!/usr/bin/env python3
"""Exact schema-v1 coverage definitions and generated-instance validators.

This standard-library-only module performs no discovery or file
writes.  It owns static schema-bundle members 9 and 10.  Generated build/run
instances remain caller-supplied evidence and are validated fail-closed.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import uuid
from typing import Any, Iterable, Mapping, NamedTuple, Sequence


POLICY_VERSION = {"major": 2, "minor": 0}
INSTANCE_VERSION = {"major": 1, "minor": 0}
UINT64_MAX = (1 << 64) - 1
POLICY_ARCHIVE_PATH = "definition/coverage-policy-v1.json"
INSTANCE_SCHEMA_ARCHIVE_PATH = "definition/coverage-instance-schema-v1.json"
BUILD_COVERAGE_ARCHIVE_PATH = "catalog/build-coverage.json"
RUN_COVERAGE_ARCHIVE_PATH = "catalog/run-coverage.json"

_IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9_]*\Z", re.ASCII)
_HASH_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_GIT_OBJECT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z", re.ASCII)
_BUILD_ID_RE = re.compile(r"(?:[0-9a-f]{2}){1,64}\Z", re.ASCII)
_VERSION_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z.+:~_-]{0,126}\Z", re.ASCII)
_SONAME_RE = re.compile(r"[0-9A-Za-z_][0-9A-Za-z.+_-]{0,254}\Z", re.ASCII)
_SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,126}\Z", re.ASCII)
_SYMBOL_VERSION_RE = re.compile(r"GLIBC_[0-9]+\.[0-9]+(?:\.[0-9]+)?\Z", re.ASCII)
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

# The generic parser admits composed build/run coverage inputs. The composer
# boundary is at most 16 MiB; retain deliberate headroom while bounding work
# before UTF-8 decoding and JSON allocation.
MAX_CANONICAL_RAW_BYTES = 64 * 1024 * 1024
MAX_JSON_NESTING_DEPTH = 64


class CoverageError(ValueError):
    """A deterministic canonical, schema, or coverage rejection."""


class ReallocZeroPolicyResolution(NamedTuple):
    """Trusted immutable result derived from one measured eligible run."""

    status: str
    policy_id: int | None
    semantic_oracle_sha256: str | None


def _catalog(rows: Sequence[tuple[int, str]]) -> list[dict[str, Any]]:
    return [{"id": identifier, "name": name} for identifier, name in rows]


ARCHITECTURES = [
    {
        "abi_oracle_id": 1,
        "architecture_id": 1,
        "elf_class": "elf64",
        "elf_data": "little_endian",
        "elf_machine": 62,
        "glibc_symbol_version": "GLIBC_2.2.5",
        "machine": "x86_64",
        "name": "linux_gnu_x86_64_lp64",
    },
    {
        "abi_oracle_id": 2,
        "architecture_id": 2,
        "elf_class": "elf64",
        "elf_data": "little_endian",
        "elf_machine": 183,
        "glibc_symbol_version": "GLIBC_2.17",
        "machine": "aarch64",
        "name": "linux_gnu_aarch64_lp64",
    },
]

CLASSIFICATIONS = _catalog(
    (
        (1, "admitted_supported"), (2, "admitted_zero_import"),
        (10, "known_unsupported_origin"), (20, "configuration_excluded"),
        (21, "not_selected"), (22, "not_loaded"), (40, "unsupported_abi"),
        (41, "semantic_oracle_missing"), (42, "identity_unverifiable"),
        (60, "new_import_unadmitted"), (61, "new_module_unadmitted"),
        (62, "domain_gap"),
    )
)

ADMISSION_STATES = _catalog(
    (
        (1, "admitted_supported"), (2, "admitted_zero_import"),
        (20, "configuration_excluded"), (21, "not_selected"),
        (22, "not_loaded"), (40, "unsupported_abi"),
        (41, "semantic_oracle_missing"), (42, "identity_unverifiable"),
        (60, "new_import_unadmitted"), (61, "new_module_unadmitted"),
        (62, "domain_gap"),
    )
)

LOAD_STATES = _catalog(
    ((1, "loaded_verified"), (10, "expected_not_loaded"),
     (11, "not_selected"), (20, "observed_unexpected"),
     (21, "active_time_changed"))
)
EVIDENCE_STATES = _catalog(
    ((1, "verified"), (10, "unavailable"), (11, "invalid"),
     (12, "mismatch"))
)
EVIDENCE_ORIGINS = _catalog(((1, "measured"), (2, "synthetic_fixture")))
VERDICTS = _catalog(
    ((1, "build_coverage_complete"), (2, "exact_domain_eligible"),
     (10, "ineligible"), (20, "unsupported"))
)
FAILURES = _catalog(
    (
        (1, "coverage_policy_mismatch"), (2, "instance_schema_mismatch"),
        (3, "build_identity_mismatch"), (4, "dependency_identity_mismatch"),
        (10, "missing_logical_elf"), (11, "duplicate_logical_elf"),
        (12, "unsupported_abi"), (13, "semantic_oracle_missing"),
        (14, "identity_unverifiable"), (20, "new_import_unadmitted"),
        (21, "new_module_unadmitted"), (22, "domain_gap"),
        (30, "expected_module_not_loaded"), (31, "module_identity_mismatch"),
        (32, "runtime_binding_mismatch"), (33, "active_time_module_change"),
        (34, "configuration_mismatch"), (40, "symbol_version_mismatch"),
        (41, "wrapper_binding_mismatch"), (42, "unknown_catalog_id"),
        (50, "evidence_unavailable"), (51, "population_reconciliation_failed"),
        (60, "dirty_source"), (61, "unsupported_platform"),
        (62, "dependency_unresolved"), (90, "synthetic_fixture_not_admissible"),
    )
)

API_RULES = [
    {
        "api_id": 1,
        "import_symbol": "malloc",
        "name": "malloc",
        "real_symbol": "__real_malloc",
        "realloc_zero_oracle_required": False,
        "signature": "pointer(size_t)",
        "symbol_versions": [
            {"architecture_id": 1, "symbol_version": "GLIBC_2.2.5"},
            {"architecture_id": 2, "symbol_version": "GLIBC_2.17"},
        ],
        "wrapper_symbol": "__wrap_malloc",
        "wrap_option": "--wrap=malloc",
    },
    {
        "api_id": 2,
        "import_symbol": "calloc",
        "name": "calloc",
        "real_symbol": "__real_calloc",
        "realloc_zero_oracle_required": False,
        "signature": "pointer(size_t,size_t)",
        "symbol_versions": [
            {"architecture_id": 1, "symbol_version": "GLIBC_2.2.5"},
            {"architecture_id": 2, "symbol_version": "GLIBC_2.17"},
        ],
        "wrapper_symbol": "__wrap_calloc",
        "wrap_option": "--wrap=calloc",
    },
    {
        "api_id": 3,
        "import_symbol": "realloc",
        "name": "realloc",
        "real_symbol": "__real_realloc",
        "realloc_zero_oracle_required": True,
        "signature": "pointer(pointer,size_t)",
        "symbol_versions": [
            {"architecture_id": 1, "symbol_version": "GLIBC_2.2.5"},
            {"architecture_id": 2, "symbol_version": "GLIBC_2.17"},
        ],
        "wrapper_symbol": "__wrap_realloc",
        "wrap_option": "--wrap=realloc",
    },
    {
        "api_id": 4,
        "import_symbol": "free",
        "name": "free",
        "real_symbol": "__real_free",
        "realloc_zero_oracle_required": False,
        "signature": "void(pointer)",
        "symbol_versions": [
            {"architecture_id": 1, "symbol_version": "GLIBC_2.2.5"},
            {"architecture_id": 2, "symbol_version": "GLIBC_2.17"},
        ],
        "wrapper_symbol": "__wrap_free",
        "wrap_option": "--wrap=free",
    },
    {
        "api_id": 5,
        "import_symbol": "reallocarray",
        "name": "reallocarray",
        "real_symbol": "__real_reallocarray",
        "realloc_zero_oracle_required": True,
        "signature": "pointer(pointer,size_t,size_t)",
        "symbol_versions": [
            {"architecture_id": 1, "symbol_version": "GLIBC_2.26"},
            {"architecture_id": 2, "symbol_version": "GLIBC_2.26"},
        ],
        "wrapper_symbol": "__wrap_reallocarray",
        "wrap_option": "--wrap=reallocarray",
    },
    {
        "api_id": 6,
        "import_symbol": "aligned_alloc",
        "name": "aligned_alloc",
        "real_symbol": "__real_aligned_alloc",
        "realloc_zero_oracle_required": False,
        "signature": "pointer(size_t,size_t)",
        "symbol_versions": [
            {"architecture_id": 1, "symbol_version": "GLIBC_2.16"},
            {"architecture_id": 2, "symbol_version": "GLIBC_2.17"},
        ],
        "wrapper_symbol": "__wrap_aligned_alloc",
        "wrap_option": "--wrap=aligned_alloc",
    },
    {
        "api_id": 7,
        "import_symbol": "posix_memalign",
        "name": "posix_memalign",
        "real_symbol": "__real_posix_memalign",
        "realloc_zero_oracle_required": False,
        "signature": "int(pointer_to_pointer,size_t,size_t)",
        "symbol_versions": [
            {"architecture_id": 1, "symbol_version": "GLIBC_2.2.5"},
            {"architecture_id": 2, "symbol_version": "GLIBC_2.17"},
        ],
        "wrapper_symbol": "__wrap_posix_memalign",
        "wrap_option": "--wrap=posix_memalign",
    },
    {
        "api_id": 8,
        "import_symbol": "memalign",
        "name": "memalign",
        "real_symbol": "__real_memalign",
        "realloc_zero_oracle_required": False,
        "signature": "pointer(size_t,size_t)",
        "symbol_versions": [
            {"architecture_id": 1, "symbol_version": "GLIBC_2.2.5"},
            {"architecture_id": 2, "symbol_version": "GLIBC_2.17"},
        ],
        "wrapper_symbol": "__wrap_memalign",
        "wrap_option": "--wrap=memalign",
    },
    {
        "api_id": 9,
        "import_symbol": "valloc",
        "name": "valloc",
        "real_symbol": "__real_valloc",
        "realloc_zero_oracle_required": False,
        "signature": "pointer(size_t)",
        "symbol_versions": [
            {"architecture_id": 1, "symbol_version": "GLIBC_2.2.5"},
            {"architecture_id": 2, "symbol_version": "GLIBC_2.17"},
        ],
        "wrapper_symbol": "__wrap_valloc",
        "wrap_option": "--wrap=valloc",
    },
    {
        "api_id": 10,
        "import_symbol": "pvalloc",
        "name": "pvalloc",
        "real_symbol": "__real_pvalloc",
        "realloc_zero_oracle_required": False,
        "signature": "pointer(size_t)",
        "symbol_versions": [
            {"architecture_id": 1, "symbol_version": "GLIBC_2.2.5"},
            {"architecture_id": 2, "symbol_version": "GLIBC_2.17"},
        ],
        "wrapper_symbol": "__wrap_pvalloc",
        "wrap_option": "--wrap=pvalloc",
    },
    {
        "api_id": 11,
        "import_symbol": "strdup",
        "name": "strdup",
        "real_symbol": "__real_strdup",
        "realloc_zero_oracle_required": False,
        "signature": "pointer(const_char_pointer)",
        "symbol_versions": [
            {"architecture_id": 1, "symbol_version": "GLIBC_2.2.5"},
            {"architecture_id": 2, "symbol_version": "GLIBC_2.17"},
        ],
        "wrapper_symbol": "__wrap_strdup",
        "wrap_option": "--wrap=strdup",
    },
    {
        "api_id": 12,
        "import_symbol": "strndup",
        "name": "strndup",
        "real_symbol": "__real_strndup",
        "realloc_zero_oracle_required": False,
        "signature": "pointer(const_char_pointer,size_t)",
        "symbol_versions": [
            {"architecture_id": 1, "symbol_version": "GLIBC_2.2.5"},
            {"architecture_id": 2, "symbol_version": "GLIBC_2.17"},
        ],
        "wrapper_symbol": "__wrap_strndup",
        "wrap_option": "--wrap=strndup",
    },
]

FAIL_CLOSED_RULES = [
    "active_time_module_change_is_ineligible",
    "alias_without_canonical_logical_elf_is_ineligible",
    "changed_module_identity_is_ineligible",
    "classification_id_must_be_known",
    "configured_or_observed_oai_elf_reconciles_once",
    "dependency_identity_must_be_verified",
    "evidence_unavailable_is_not_zero",
    "extra_elf_is_ineligible",
    "generation_mismatch_is_ineligible",
    "missing_classification_is_ineligible",
    "missing_logical_elf_row_is_ineligible",
    "module_admission_state_is_distinct_from_origin_classification",
    "new_import_is_ineligible",
    "new_module_is_ineligible",
    "realloc_requires_build_bound_semantic_oracle",
    "run_population_is_expected_observed_union",
    "unknown_abi_is_ineligible",
    "unknown_alias_is_ineligible",
    "unknown_catalog_id_is_ineligible",
    "unknown_elf_is_ineligible",
    "unknown_symbol_version_is_ineligible",
    "unverified_identity_is_ineligible",
    "active_realloc_importers_share_one_policy",
    "known_unsupported_origins_are_outside_first_slice_denominator",
    "known_unsupported_symbol_versions_are_architecture_exact",
    "supported_origin_symbol_versions_are_architecture_exact",
]


def _reject_float(_: str) -> None:
    raise CoverageError("number: floats and exponents are forbidden")


def _reject_constant(_: str) -> None:
    raise CoverageError("number: non-finite values are forbidden")


def _pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    previous: bytes | None = None
    for key, value in pairs:
        if key in out:
            raise CoverageError(f"member: duplicate key {key!r}")
        if not isinstance(key, str) or not _IDENTIFIER_RE.fullmatch(key):
            raise CoverageError(f"member: invalid ASCII identifier {key!r}")
        encoded = key.encode("ascii")
        if previous is not None and encoded <= previous:
            raise CoverageError("member: keys are not in bytewise ASCII order")
        previous = encoded
        out[key] = value
    return out


def _validate_string(value: str, where: str) -> None:
    if "\x00" in value:
        raise CoverageError(f"{where}: U+0000 is forbidden")
    if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise CoverageError(f"{where}: surrogate code point is forbidden")
    if unicodedata.normalize("NFC", value) != value:
        raise CoverageError(f"{where}: string is not NFC")


def _quote(value: str, where: str) -> str:
    _validate_string(value, where)
    pieces = ['"']
    for char in value:
        codepoint = ord(char)
        if char == '"':
            pieces.append('\\"')
        elif char == "\\":
            pieces.append("\\\\")
        elif codepoint < 0x20 or codepoint == 0x7F:
            pieces.append(f"\\u{codepoint:04x}")
        else:
            pieces.append(char)
    pieces.append('"')
    return "".join(pieces)


def _check_json_depth(depth: int) -> None:
    if depth > MAX_JSON_NESTING_DEPTH:
        raise CoverageError(
            f"json: nesting depth exceeds {MAX_JSON_NESTING_DEPTH}"
        )


def _scan_json_depth(raw: bytes) -> None:
    """Reject excessive structural nesting without decoding or parsing JSON."""

    depth = 0
    in_string = False
    escaped = False
    for byte in raw:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:  # backslash
                escaped = True
            elif byte == 0x22:  # double quote
                in_string = False
            continue
        if byte == 0x22:  # double quote
            in_string = True
        elif byte in (0x5B, 0x7B):  # '[' or '{'
            depth += 1
            _check_json_depth(depth)
        elif byte in (0x5D, 0x7D):  # ']' or '}'
            depth -= 1


def _encode(value: Any, where: str, depth: int = 1) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return _quote(value, where)
    if isinstance(value, list):
        _check_json_depth(depth)
        return "[" + ",".join(
            _encode(item, f"{where}[{index}]", depth + 1)
            for index, item in enumerate(value)
        ) + "]"
    if isinstance(value, dict):
        _check_json_depth(depth)
        encoded_keys: list[tuple[bytes, str]] = []
        for key in value:
            if not isinstance(key, str) or not _IDENTIFIER_RE.fullmatch(key):
                raise CoverageError(f"{where}: invalid ASCII identifier key {key!r}")
            encoded_keys.append((key.encode("ascii"), key))
        encoded_keys.sort()
        members = (
            _quote(key, f"{where}.<key>")
            + ":"
            + _encode(value[key], f"{where}.{key}", depth + 1)
            for _, key in encoded_keys
        )
        return "{" + ",".join(members) + "}"
    raise CoverageError(f"{where}: unsupported JSON value type")


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    """Return exact canonical format-1 bytes including the sole final LF."""

    if not isinstance(value, dict):
        raise CoverageError("root: exactly one object is required")
    try:
        return (_encode(value, "$") + "\n").encode("utf-8")
    except RecursionError as error:
        raise CoverageError(
            f"json: nesting depth exceeds {MAX_JSON_NESTING_DEPTH}"
        ) from error


def parse_canonical(raw: bytes) -> dict[str, Any]:
    """Parse only bytes already in the exact canonical representation."""

    if not isinstance(raw, bytes):
        raise CoverageError("encoding: bytes required")
    if len(raw) > MAX_CANONICAL_RAW_BYTES:
        raise CoverageError(
            f"encoding: bytes exceed {MAX_CANONICAL_RAW_BYTES}"
        )
    if raw.startswith(b"\xef\xbb\xbf"):
        raise CoverageError("encoding: BOM is forbidden")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise CoverageError("encoding: exactly one final LF is required")
    _scan_json_depth(raw)
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise CoverageError("encoding: invalid UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except RecursionError as error:
        raise CoverageError(
            f"json: nesting depth exceeds {MAX_JSON_NESTING_DEPTH}"
        ) from error
    except json.JSONDecodeError as error:
        raise CoverageError(f"json: {error.msg}") from error
    if not isinstance(value, dict):
        raise CoverageError("root: exactly one object is required")
    if canonical_bytes(value) != raw:
        raise CoverageError("encoding: bytes are not the canonical serialization")
    return value


def sha256_hex(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


ROLES = _catalog(((1, "gnb"), (2, "nr_ue")))
ELF_KINDS = _catalog(((1, "executable"), (2, "shared_object")))
ORIGIN_KINDS = _catalog(
    ((1, "supported_api"), (2, "known_unsupported"),
     (3, "new_import_unadmitted"))
)
REALLOC_ZERO_POLICIES = _catalog(
    ((1, "predecessor_ended_no_successor"),
     (2, "predecessor_preserved_failure"))
)
REALLOC_ZERO_SEMANTIC_ORACLE_BINDING = {
    "binding_field": "logical_elf.realloc_zero_semantic_oracle_sha256",
    "equality": "field_equals_sha256",
    "hash_domain": "exact_canonical_member_bytes_including_sole_final_lf",
    "object_type": 3,
    "path": "definition/event-semantics-v1.json",
    "sha256": "8dbe428939592cdfc86ba8730078563672de4a13081ebdb868fa97f543dfab89",
}
RELOCATION_KINDS = _catalog(((1, "plt"), (2, "global_data")))
REQUIRED_EVIDENCE = [
    {"evidence_id": 1, "name": "elf_identity",
     "required_fields": ["build_id", "byte_count", "sha256", "soname"]},
    {"evidence_id": 2, "name": "final_link",
     "required_fields": ["link_command_sha256", "link_map_sha256"]},
    {"evidence_id": 3, "name": "imports",
     "required_fields": ["symbol_origins"]},
    {"evidence_id": 4, "name": "relocations",
     "required_fields": ["import_relocations"]},
    {"evidence_id": 5, "name": "hidden_exports",
     "required_fields": ["hidden_wrapper_symbols"]},
    {"evidence_id": 6, "name": "wrap_set",
     "required_fields": ["wrap_options"]},
    {"evidence_id": 7, "name": "dependencies",
     "required_fields": ["dt_needed", "shared_runtime_binding"]},
    {"evidence_id": 8, "name": "runtime_identity",
     "required_fields": ["module_map_sha256", "runtime_identity"]},
]


def _unsupported_origin(
    origin_id: int,
    name: str,
    symbol: str,
    *,
    x86_64_version: str,
    aarch64_version: str = "GLIBC_2.17",
) -> dict[str, Any]:
    return {
        "classification_id": 10,
        "name": name,
        "origin_id": origin_id,
        "symbol": symbol,
        "symbol_versions": [
            {"architecture_id": 1, "symbol_version": x86_64_version},
            {"architecture_id": 2, "symbol_version": aarch64_version},
        ],
    }


KNOWN_UNSUPPORTED_ORIGINS = [
    _unsupported_origin(
        101, "asprintf", "asprintf", x86_64_version="GLIBC_2.2.5"
    ),
    _unsupported_origin(
        102, "vasprintf", "vasprintf", x86_64_version="GLIBC_2.2.5"
    ),
    _unsupported_origin(
        103, "fortified_asprintf", "__asprintf_chk", x86_64_version="GLIBC_2.8"
    ),
    _unsupported_origin(
        104, "fortified_vasprintf", "__vasprintf_chk", x86_64_version="GLIBC_2.8"
    ),
]
PATH_GRAMMARS = [
    {"encoding": "utf8_nfc_fail_closed", "name": "archive_path",
     "pattern": "relative_posix_no_empty_dot_dotdot_backslash_or_control",
     "root": "evidence_archive"},
    {"encoding": "utf8_nfc_fail_closed", "name": "host_path",
     "pattern": "absolute_posix_no_empty_dot_dotdot_backslash_or_control",
     "root": "host_filesystem"},
    {"encoding": "utf8_nfc_fail_closed", "name": "repo_path",
     "pattern": "relative_posix_no_empty_dot_dotdot_backslash_or_control",
     "root": "repository"},
]
SELECTION_GRAMMAR = {
    "operators": _catalog(((1, "all"), (2, "any"))),
    "predicate_kinds": [
        {"configuration_key": "forbidden", "expected_value": "forbidden",
         "predicate_id": 1, "name": "always"},
        {"configuration_key": "required", "expected_value": "required",
         "predicate_id": 2, "name": "configuration_equals"},
        {"configuration_key": "forbidden", "expected_value": "role_name",
         "predicate_id": 3, "name": "role_equals"},
    ],
    "predicate_order": ["predicate_id", "configuration_key", "expected_value"],
}

COVERAGE_POLICY_DEFINITION = {
    "admission_states": ADMISSION_STATES,
    "architectures": ARCHITECTURES,
    "classification_catalog": CLASSIFICATIONS,
    "definition_id": "oai_memprof_coverage_policy",
    "domain": {
        "abi": "gnu",
        "compiler_family": "gcc",
        "elf_class": "elf64",
        "elf_data": "little_endian",
        "libc_family": "glibc",
        "operating_system": "linux",
    },
    "elf_kinds": ELF_KINDS,
    "evidence_origins": EVIDENCE_ORIGINS,
    "evidence_states": EVIDENCE_STATES,
    "fail_closed_rules": FAIL_CLOSED_RULES,
    "failure_catalog": FAILURES,
    "first_slice_apis": API_RULES,
    "known_unsupported_origins": KNOWN_UNSUPPORTED_ORIGINS,
    "load_states": LOAD_STATES,
    "module_selection": SELECTION_GRAMMAR,
    "origin_kinds": ORIGIN_KINDS,
    "path_grammars": PATH_GRAMMARS,
    "realloc_zero_oracle_binding": REALLOC_ZERO_SEMANTIC_ORACLE_BINDING,
    "realloc_zero_policies": REALLOC_ZERO_POLICIES,
    "relocation_kinds": RELOCATION_KINDS,
    "required_evidence": REQUIRED_EVIDENCE,
    "roles": ROLES,
    "verdicts": VERDICTS,
    "version": POLICY_VERSION,
}


def _field(name: str, field_type: str, nullable: bool = False) -> dict[str, Any]:
    return {"name": name, "nullable": nullable, "type": field_type}


def _object_schema(
    name: str,
    fields: Sequence[tuple[str, str, bool]],
    *,
    array_order: str = "not_applicable",
    uniqueness: str = "not_applicable",
) -> dict[str, Any]:
    return {
        "array_order": array_order,
        "fields": [_field(field, kind, nullable) for field, kind, nullable in fields],
        "name": name,
        "unknown_fields_allowed": False,
        "uniqueness": uniqueness,
    }


OBJECT_SCHEMAS = [
    _object_schema("build_coverage", (
        ("api_definition", "definition_binding", False),
        ("architecture_id", "u16_nonzero", False),
        ("build_identity", "build_identity", False),
        ("catalog_id", "literal:oai_memprof_build_coverage", False),
        ("dependencies", "array:dependency_identity", False),
        ("entries", "array:logical_elf", False),
        ("evidence_origin_id", "u16_nonzero", False),
        ("failure_ids", "array:u16_nonzero", False),
        ("policy", "definition_binding", False),
        ("schema", "definition_binding", False),
        ("verdict_id", "u16_nonzero", False),
        ("version", "version_1_0", False),
    )),
    _object_schema("build_identity", (
        ("build_configuration_sha256", "sha256_hex", False),
        ("compiler_id", "ascii_identifier", False),
        ("compiler_version", "version_string", False),
        ("dirty", "boolean", False),
        ("libc_id", "literal:glibc", False),
        ("libc_version", "version_string", False),
        ("linker_id", "ascii_identifier", False),
        ("linker_version", "version_string", False),
        ("operating_system", "literal:linux", False),
        ("primary_logical_elf_id", "logical_elf_id", False),
        ("source_commit", "git_object_hex", False),
        ("source_tree", "git_object_hex", False),
        ("target_triple", "ascii_token", False),
    )),
    _object_schema("definition_binding", (
        ("object_type", "u16_nonzero", False),
        ("path", "archive_path", False),
        ("sha256", "sha256_hex", False),
    )),
    _object_schema("dependency_identity", (
        ("dependency_id", "ascii_identifier", False),
        ("evidence_state_id", "u16_nonzero", False),
        ("name", "ascii_token", False),
        ("sha256", "sha256_hex", False),
        ("version", "version_string", False),
    ), array_order="dependency_id_ascii_ascending", uniqueness="dependency_id"),
    _object_schema("instance_binding", (
        ("object_kind", "u16_nonzero", False),
        ("path", "archive_path", False),
        ("sha256", "sha256_hex", False),
    )),
    _object_schema("logical_elf", (
        ("admission_state_id", "u16_nonzero", False),
        ("aliases", "array:repo_path", False),
        ("build_id", "build_id_hex", False),
        ("byte_count", "u64_nonzero", False),
        ("dt_needed", "array:soname", False),
        ("elf_kind_id", "u16_nonzero", False),
        ("elf_machine", "u16_nonzero", False),
        ("evidence_state_id", "u16_nonzero", False),
        ("hidden_wrapper_symbols", "array:symbol", False),
        ("import_relocations", "array:symbol_relocation", False),
        ("link_command_sha256", "sha256_hex", False),
        ("link_map_sha256", "sha256_hex", False),
        ("logical_id", "logical_elf_id", False),
        ("module_selection", "module_selection", False),
        ("realloc_zero_policy_id", "u16_nonzero", True),
        ("realloc_zero_semantic_oracle_sha256", "sha256_hex", True),
        ("repo_path", "repo_path", False),
        ("role_ids", "array:u16_nonzero", False),
        ("sha256", "sha256_hex", False),
        ("shared_runtime_binding", "shared_runtime_binding", True),
        ("soname", "soname", True),
        ("symbol_origins", "array:symbol_origin", False),
        ("wrap_options", "array:wrap_option", False),
    ), array_order="logical_id_ascii_ascending", uniqueness="logical_id"),
    _object_schema("module_classification", (
        ("classification_id", "u16_nonzero", False),
        ("origin_id", "u16_nonzero", True),
    ), array_order="origin_id_null_last_ascending", uniqueness="origin_id"),
    _object_schema("module_population", (
        ("admission_state_id", "u16_nonzero", False),
        ("build_logical_id", "logical_elf_id", True),
        ("classifications", "array:module_classification", False),
        ("configured", "boolean", False),
        ("load_generation", "u64_nonzero", True),
        ("load_state_id", "u16_nonzero", False),
        ("loaded_path", "host_path", True),
        ("logical_id", "logical_elf_id", False),
        ("observed", "boolean", False),
        ("runtime_identity", "runtime_identity", True),
    ), array_order="logical_id_ascii_ascending", uniqueness="logical_id"),
    _object_schema("module_selection", (
        ("operator_id", "u16_nonzero", False),
        ("predicates", "array:selection_predicate", False),
    )),
    _object_schema("run_coverage", (
        ("build_coverage", "instance_binding", False),
        ("catalog_id", "literal:oai_memprof_run_coverage", False),
        ("configuration_instance_sha256", "sha256_hex", False),
        ("eligible_exact_domain", "boolean", False),
        ("evidence_origin_id", "u16_nonzero", False),
        ("failure_ids", "array:u16_nonzero", False),
        ("module_population", "array:module_population", False),
        ("policy", "definition_binding", False),
        ("process_generation", "u64_nonzero", False),
        ("process_uuid", "uuid_lowercase", False),
        ("role_id", "u16_nonzero", False),
        ("run_uuid", "uuid_lowercase", False),
        ("schema", "definition_binding", False),
        ("snapshot_state_id", "literal:1", False),
        ("verdict_id", "u16_nonzero", False),
        ("version", "version_1_0", False),
    )),
    _object_schema("runtime_identity", (
        ("build_id", "build_id_hex", False),
        ("byte_count", "u64_nonzero", False),
        ("hidden_wrapper_symbols_sha256", "sha256_hex", False),
        ("import_relocations_sha256", "sha256_hex", False),
        ("module_map_sha256", "sha256_hex", False),
        ("sha256", "sha256_hex", False),
        ("shared_runtime_binding_sha256", "sha256_hex", False),
        ("symbol_origins_sha256", "sha256_hex", False),
    )),
    _object_schema("selection_predicate", (
        ("configuration_key", "ascii_identifier", True),
        ("expected_value", "string", True),
        ("predicate_id", "u16_nonzero", False),
    ), array_order="predicate_id_key_value_ascii_ascending",
       uniqueness="predicate_id_configuration_key_expected_value"),
    _object_schema("shared_runtime_binding", (
        ("dependency_id", "ascii_identifier", False),
        ("evidence_state_id", "u16_nonzero", False),
        ("soname", "soname", False),
    )),
    _object_schema("symbol_origin", (
        ("api_id", "u16_nonzero", True),
        ("classification_id", "u16_nonzero", False),
        ("origin_id", "u16_nonzero", False),
        ("origin_kind_id", "u16_nonzero", False),
        ("symbol", "symbol", False),
        ("symbol_version", "symbol_version", False),
    ), array_order="origin_id_ascending", uniqueness="origin_id"),
    _object_schema("symbol_relocation", (
        ("origin_id", "u16_nonzero", False),
        ("relocation_kind_id", "u16_nonzero", False),
    ), array_order="origin_id_ascending", uniqueness="origin_id"),
]

COVERAGE_INSTANCE_SCHEMA_DEFINITION = {
    "definition_id": "oai_memprof_coverage_instance_schema",
    "field_types": [
        {"constraint": "[a-z][a-z0-9_]{0,62}", "encoding": "json_string", "name": "ascii_identifier"},
        {"constraint": "[0-9A-Za-z][0-9A-Za-z.+:~_-]{0,254}", "encoding": "json_string", "name": "ascii_token"},
        {"constraint": "exact_true_or_false", "encoding": "json_boolean", "name": "boolean"},
        {"constraint": "(?:[0-9a-f]{2}){1,64}", "encoding": "json_string", "name": "build_id_hex"},
        {"constraint": "absolute_posix_utf8_nfc", "encoding": "json_string", "name": "host_path"},
        {"constraint": "[a-z][a-z0-9_]{0,62}", "encoding": "json_string", "name": "logical_elf_id"},
        {"constraint": "relative_posix_utf8_nfc", "encoding": "json_string", "name": "repo_path"},
        {"constraint": "[0-9a-f]{64}", "encoding": "json_string", "name": "sha256_hex"},
        {"constraint": "bounded_ascii_soname", "encoding": "json_string", "name": "soname"},
        {"constraint": "utf8_nfc_1_to_4096_bytes", "encoding": "json_string", "name": "string"},
        {"constraint": "[A-Za-z_][A-Za-z0-9_]{0,126}", "encoding": "json_string", "name": "symbol"},
        {"constraint": "GLIBC_[0-9]+\\.[0-9]+(?:\\.[0-9]+)?", "encoding": "json_string", "name": "symbol_version"},
        {"constraint": "1_to_65535", "encoding": "json_integer", "name": "u16_nonzero"},
        {"constraint": "1_to_18446744073709551615", "encoding": "json_integer", "name": "u64_nonzero"},
        {"constraint": "rfc4122_hyphenated_lowercase", "encoding": "json_string", "name": "uuid_lowercase"},
        {"constraint": "exact_object_major_1_minor_0", "encoding": "json_object", "name": "version_1_0"},
        {"constraint": "bounded_ascii_version", "encoding": "json_string", "name": "version_string"},
        {"constraint": "--wrap_equals_symbol", "encoding": "json_string", "name": "wrap_option"},
    ],
    "grammars": {
        "ascii_identifier": "[a-z][a-z0-9_]{0,62}",
        "build_id_hex": "(?:[0-9a-f]{2}){1,64}",
        "git_object_hex": "[0-9a-f]{40}|[0-9a-f]{64}",
        "logical_elf_id": "[a-z][a-z0-9_]{0,62}",
        "sha256_hex": "[0-9a-f]{64}",
        "symbol": "[A-Za-z_][A-Za-z0-9_]{0,126}",
        "symbol_version": "GLIBC_[0-9]+\\.[0-9]+(?:\\.[0-9]+)?",
        "uuid_lowercase": "rfc4122_hyphenated_lowercase",
    },
    "object_schemas": OBJECT_SCHEMAS,
    "schema_ids": [
        {"catalog_id": "oai_memprof_build_coverage", "object_kind": 8,
         "path": BUILD_COVERAGE_ARCHIVE_PATH},
        {"catalog_id": "oai_memprof_run_coverage", "object_kind": 9,
         "path": RUN_COVERAGE_ARCHIVE_PATH},
    ],
    "version": INSTANCE_VERSION,
}

POLICY_BYTES = canonical_bytes(COVERAGE_POLICY_DEFINITION)
POLICY_SHA256 = sha256_hex(POLICY_BYTES)
INSTANCE_SCHEMA_BYTES = canonical_bytes(COVERAGE_INSTANCE_SCHEMA_DEFINITION)
INSTANCE_SCHEMA_SHA256 = sha256_hex(INSTANCE_SCHEMA_BYTES)


def _exact_keys(value: Any, keys: Iterable[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise CoverageError(f"{where}: object required")
    expected = tuple(sorted(keys))
    if tuple(value.keys()) != expected:
        raise CoverageError(f"{where}: exact keys {expected!r} required")
    return value


def _uint(value: Any, bits: int, where: str, *, nonzero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CoverageError(f"{where}: u{bits} integer required")
    lower = 1 if nonzero else 0
    if not lower <= value <= (1 << bits) - 1:
        raise CoverageError(f"{where}: outside u{bits} range")
    return value


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise CoverageError(f"{where}: boolean required")
    return value


def _string(value: Any, where: str, *, maximum_bytes: int = 4096) -> str:
    if not isinstance(value, str) or not value:
        raise CoverageError(f"{where}: nonempty string required")
    _validate_string(value, where)
    if len(value.encode("utf-8")) > maximum_bytes:
        raise CoverageError(f"{where}: string exceeds {maximum_bytes} UTF-8 bytes")
    return value


def _identifier(value: Any, where: str) -> str:
    text = _string(value, where, maximum_bytes=63)
    if not _IDENTIFIER_RE.fullmatch(text):
        raise CoverageError(f"{where}: ASCII identifier required")
    return text


def _hash(value: Any, where: str) -> str:
    text = _string(value, where, maximum_bytes=64)
    if not _HASH_RE.fullmatch(text):
        raise CoverageError(f"{where}: SHA-256 lowercase hexadecimal required")
    return text


def _git_object(value: Any, where: str) -> str:
    text = _string(value, where, maximum_bytes=64)
    if not _GIT_OBJECT_RE.fullmatch(text):
        raise CoverageError(f"{where}: 40- or 64-character Git object required")
    return text


def _build_id(value: Any, where: str) -> str:
    text = _string(value, where, maximum_bytes=128)
    if not _BUILD_ID_RE.fullmatch(text):
        raise CoverageError(f"{where}: 1..64-byte lowercase hexadecimal Build-ID required")
    return text


def _version_string(value: Any, where: str) -> str:
    text = _string(value, where, maximum_bytes=127)
    if not _VERSION_RE.fullmatch(text):
        raise CoverageError(f"{where}: bounded ASCII version required")
    return text


def _symbol(value: Any, where: str) -> str:
    text = _string(value, where, maximum_bytes=127)
    if not _SYMBOL_RE.fullmatch(text):
        raise CoverageError(f"{where}: bounded ELF symbol required")
    return text


def _symbol_version(value: Any, where: str) -> str:
    text = _string(value, where, maximum_bytes=31)
    if not _SYMBOL_VERSION_RE.fullmatch(text):
        raise CoverageError(f"{where}: GLIBC symbol version required")
    return text


def _soname(value: Any, where: str) -> str:
    text = _string(value, where, maximum_bytes=255)
    if not _SONAME_RE.fullmatch(text):
        raise CoverageError(f"{where}: bounded SONAME required")
    return text


def _relative_path(value: Any, where: str) -> str:
    text = _string(value, where)
    if text.startswith("/") or text.endswith("/") or "\\" in text or _CONTROL_RE.search(text):
        raise CoverageError(f"{where}: invalid relative POSIX path")
    if any(part in ("", ".", "..") for part in text.split("/")):
        raise CoverageError(f"{where}: empty/dot/dot-dot component forbidden")
    return text


def _host_path(value: Any, where: str) -> str:
    text = _string(value, where)
    if text == "/" or not text.startswith("/") or text.endswith("/") or "\\" in text or _CONTROL_RE.search(text):
        raise CoverageError(f"{where}: invalid absolute POSIX host path")
    if any(part in ("", ".", "..") for part in text[1:].split("/")):
        raise CoverageError(f"{where}: empty/dot/dot-dot component forbidden")
    return text


def _uuid(value: Any, where: str) -> str:
    text = _string(value, where, maximum_bytes=36)
    try:
        parsed = uuid.UUID(text)
    except (ValueError, AttributeError) as error:
        raise CoverageError(f"{where}: RFC 4122 UUID required") from error
    if str(parsed) != text:
        raise CoverageError(f"{where}: canonical lowercase UUID required")
    return text


def _known_id(value: Any, catalog: Sequence[Mapping[str, Any]], where: str) -> int:
    identifier = _uint(value, 16, where, nonzero=True)
    known = {row.get("id", row.get("architecture_id")) for row in catalog}
    if identifier not in known:
        raise CoverageError(f"{where}: unknown catalog ID {identifier}")
    return identifier


def expected_supported_symbol_version(
    api_id: Any, architecture_id: Any
) -> str:
    """Return the exact registered GLIBC version for one supported API."""

    checked_api = _uint(api_id, 16, "supported_api.api_id", nonzero=True)
    checked_architecture = _known_id(
        architecture_id, ARCHITECTURES, "supported_api.architecture_id"
    )
    rules = {row["api_id"]: row for row in API_RULES}
    if checked_api not in rules:
        raise CoverageError(
            f"supported_api.api_id: unknown catalog ID {checked_api}"
        )
    versions = {
        _known_id(
            row["architecture_id"],
            ARCHITECTURES,
            "supported_api.symbol_versions.architecture_id",
        ): _symbol_version(
            row["symbol_version"],
            "supported_api.symbol_versions.symbol_version",
        )
        for row in rules[checked_api]["symbol_versions"]
    }
    if set(versions) != {row["architecture_id"] for row in ARCHITECTURES}:
        raise CoverageError(
            "supported_api.symbol_versions: exact architecture population required"
        )
    return versions[checked_architecture]


def expected_known_unsupported_symbol_version(
    origin_id: Any, architecture_id: Any
) -> str:
    """Return the frozen symbol version for one registered unsupported origin."""

    checked_origin = _uint(
        origin_id, 16, "known_unsupported.origin_id", nonzero=True
    )
    checked_architecture = _known_id(
        architecture_id, ARCHITECTURES, "known_unsupported.architecture_id"
    )
    origins = {row["origin_id"]: row for row in KNOWN_UNSUPPORTED_ORIGINS}
    if checked_origin not in origins:
        raise CoverageError(
            f"known_unsupported.origin_id: unknown catalog ID {checked_origin}"
        )
    versions = {
        _known_id(
            row["architecture_id"],
            ARCHITECTURES,
            "known_unsupported.symbol_versions.architecture_id",
        ): _symbol_version(
            row["symbol_version"],
            "known_unsupported.symbol_versions.symbol_version",
        )
        for row in origins[checked_origin]["symbol_versions"]
    }
    if set(versions) != {row["architecture_id"] for row in ARCHITECTURES}:
        raise CoverageError(
            "known_unsupported.symbol_versions: exact architecture population required"
        )
    return versions[checked_architecture]


def _sorted_unique(values: Any, validate: Any, where: str) -> list[Any]:
    if not isinstance(values, list):
        raise CoverageError(f"{where}: array required")
    checked = [validate(value, f"{where}[{index}]") for index, value in enumerate(values)]
    if checked != sorted(checked) or len(checked) != len(set(checked)):
        raise CoverageError(f"{where}: strict ascending unique order required")
    return checked


def _version_object(value: Any, where: str) -> None:
    _exact_keys(value, ("major", "minor"), where)
    major = _uint(value["major"], 16, f"{where}.major")
    minor = _uint(value["minor"], 16, f"{where}.minor")
    if (major, minor) != (1, 0):
        raise CoverageError(f"{where}: exact version 1.0 required")


def _definition_binding(
    value: Any, where: str, *, object_type: int, path: str, digest: str
) -> None:
    _exact_keys(value, ("object_type", "path", "sha256"), where)
    if _uint(value["object_type"], 16, f"{where}.object_type", nonzero=True) != object_type:
        raise CoverageError(f"{where}.object_type: expected {object_type}")
    if _relative_path(value["path"], f"{where}.path") != path:
        raise CoverageError(f"{where}.path: expected {path!r}")
    if _hash(value["sha256"], f"{where}.sha256") != digest:
        raise CoverageError(f"{where}.sha256: exact definition digest mismatch")


def _instance_binding(
    value: Any, where: str, *, object_kind: int, path: str, digest: str
) -> None:
    _exact_keys(value, ("object_kind", "path", "sha256"), where)
    if _uint(value["object_kind"], 16, f"{where}.object_kind", nonzero=True) != object_kind:
        raise CoverageError(f"{where}.object_kind: expected {object_kind}")
    if _relative_path(value["path"], f"{where}.path") != path:
        raise CoverageError(f"{where}.path: expected {path!r}")
    if _hash(value["sha256"], f"{where}.sha256") != digest:
        raise CoverageError(f"{where}.sha256: exact instance digest mismatch")


def validate_coverage_policy(value: Mapping[str, Any]) -> None:
    if canonical_bytes(value) != POLICY_BYTES:
        raise CoverageError("coverage_policy: frozen definition differs")


def validate_coverage_instance_schema(value: Mapping[str, Any]) -> None:
    if canonical_bytes(value) != INSTANCE_SCHEMA_BYTES:
        raise CoverageError("coverage_instance_schema: frozen definition differs")


def _nonzero_hash(value: Any, where: str) -> str:
    digest = _hash(value, where)
    if not int(digest, 16):
        raise CoverageError(f"{where}: all-zero digest forbidden")
    return digest


def _validate_build_identity(value: Any) -> Mapping[str, Any]:
    where = "build_coverage.build_identity"
    _exact_keys(value, (
        "build_configuration_sha256", "compiler_id", "compiler_version", "dirty",
        "libc_id", "libc_version", "linker_id", "linker_version",
        "operating_system", "primary_logical_elf_id", "source_commit",
        "source_tree", "target_triple",
    ), where)
    _nonzero_hash(
        value["build_configuration_sha256"],
        f"{where}.build_configuration_sha256",
    )
    _identifier(value["compiler_id"], f"{where}.compiler_id")
    _version_string(value["compiler_version"], f"{where}.compiler_version")
    _boolean(value["dirty"], f"{where}.dirty")
    if value["libc_id"] != "glibc" or value["operating_system"] != "linux":
        raise CoverageError(f"{where}: exact Linux/glibc domain required")
    _version_string(value["libc_version"], f"{where}.libc_version")
    _identifier(value["linker_id"], f"{where}.linker_id")
    _version_string(value["linker_version"], f"{where}.linker_version")
    _identifier(value["primary_logical_elf_id"], f"{where}.primary_logical_elf_id")
    _git_object(value["source_commit"], f"{where}.source_commit")
    _git_object(value["source_tree"], f"{where}.source_tree")
    target = _version_string(value["target_triple"], f"{where}.target_triple")
    if target not in ("aarch64-linux-gnu", "x86_64-linux-gnu"):
        raise CoverageError(f"{where}.target_triple: unsupported target")
    return value


def _validate_dependency(value: Any, index: int) -> Mapping[str, Any]:
    where = f"build_coverage.dependencies[{index}]"
    _exact_keys(value, ("dependency_id", "evidence_state_id", "name", "sha256", "version"), where)
    _identifier(value["dependency_id"], f"{where}.dependency_id")
    _known_id(value["evidence_state_id"], EVIDENCE_STATES, f"{where}.evidence_state_id")
    _soname(value["name"], f"{where}.name")
    _nonzero_hash(value["sha256"], f"{where}.sha256")
    _version_string(value["version"], f"{where}.version")
    return value


def _validate_selection(value: Any, where: str) -> None:
    _exact_keys(value, ("operator_id", "predicates"), where)
    if _known_id(value["operator_id"], SELECTION_GRAMMAR["operators"], f"{where}.operator_id") not in (1, 2):
        raise CoverageError(f"{where}.operator_id: unsupported operator")
    predicates = value["predicates"]
    if not isinstance(predicates, list) or not predicates:
        raise CoverageError(f"{where}.predicates: nonempty array required")
    observed: list[tuple[int, str, str]] = []
    for index, row in enumerate(predicates):
        location = f"{where}.predicates[{index}]"
        _exact_keys(row, ("configuration_key", "expected_value", "predicate_id"), location)
        predicate_id = _uint(row["predicate_id"], 16, f"{location}.predicate_id", nonzero=True)
        key = row["configuration_key"]
        expected = row["expected_value"]
        if predicate_id == 1:
            if key is not None or expected is not None:
                raise CoverageError(f"{location}: always forbids operands")
        elif predicate_id == 2:
            key = _identifier(key, f"{location}.configuration_key")
            expected = _string(expected, f"{location}.expected_value", maximum_bytes=255)
        elif predicate_id == 3:
            if key is not None or expected not in ("gnb", "nr_ue"):
                raise CoverageError(f"{location}: role_equals requires one role name")
        else:
            raise CoverageError(f"{location}.predicate_id: unknown predicate")
        observed.append((predicate_id, key or "", expected or ""))
    if observed != sorted(observed) or len(observed) != len(set(observed)):
        raise CoverageError(f"{where}.predicates: strict grammar order and uniqueness required")


def _validate_symbol_origin(
    value: Any, index: int, *, architecture: Mapping[str, Any]
) -> tuple[int, int, int]:
    where = f"logical_elf.symbol_origins[{index}]"
    _exact_keys(value, (
        "api_id", "classification_id", "origin_id", "origin_kind_id",
        "symbol", "symbol_version",
    ), where)
    origin_id = _uint(value["origin_id"], 16, f"{where}.origin_id", nonzero=True)
    kind = _known_id(value["origin_kind_id"], ORIGIN_KINDS, f"{where}.origin_kind_id")
    classification = _known_id(
        value["classification_id"], CLASSIFICATIONS, f"{where}.classification_id"
    )
    symbol = _symbol(value["symbol"], f"{where}.symbol")
    version = _symbol_version(value["symbol_version"], f"{where}.symbol_version")
    api = None if value["api_id"] is None else _uint(
        value["api_id"], 16, f"{where}.api_id", nonzero=True
    )
    api_rules = {row["api_id"]: row for row in API_RULES}
    unsupported = {row["origin_id"]: row for row in KNOWN_UNSUPPORTED_ORIGINS}
    if kind == 1:
        if origin_id not in api_rules or api != origin_id or classification != 1:
            raise CoverageError(f"{where}: supported origin must bind its exact API/classification")
        if symbol != api_rules[origin_id]["import_symbol"]:
            raise CoverageError(f"{where}.symbol: first-slice import mismatch")
        expected_version = expected_supported_symbol_version(
            origin_id, architecture["architecture_id"]
        )
    elif kind == 2:
        if origin_id not in unsupported or api is not None or classification != 10:
            raise CoverageError(f"{where}: known-unsupported origin binding mismatch")
        if symbol != unsupported[origin_id]["symbol"]:
            raise CoverageError(f"{where}.symbol: known-unsupported symbol mismatch")
        expected_version = expected_known_unsupported_symbol_version(
            origin_id, architecture["architecture_id"]
        )
    else:
        if origin_id < 32768 or api is not None or classification != 60:
            raise CoverageError(f"{where}: new imports require private ID >=32768 and class 60")
        expected_version = architecture["glibc_symbol_version"]
    if version != expected_version:
        raise CoverageError(f"{where}.symbol_version: architecture oracle mismatch")
    return origin_id, kind, classification


def _subobject_sha256(member: str, value: Any) -> str:
    return sha256_hex(canonical_bytes({member: value}))


def _expected_runtime_identity(logical: Mapping[str, Any], module_map_sha256: str) -> dict[str, Any]:
    return {
        "build_id": logical["build_id"],
        "byte_count": logical["byte_count"],
        "hidden_wrapper_symbols_sha256": _subobject_sha256(
            "hidden_wrapper_symbols", logical["hidden_wrapper_symbols"]
        ),
        "import_relocations_sha256": _subobject_sha256(
            "import_relocations", logical["import_relocations"]
        ),
        "module_map_sha256": module_map_sha256,
        "sha256": logical["sha256"],
        "shared_runtime_binding_sha256": _subobject_sha256(
            "shared_runtime_binding", logical["shared_runtime_binding"]
        ),
        "symbol_origins_sha256": _subobject_sha256("symbol_origins", logical["symbol_origins"]),
    }


def _validate_logical_elf(
    value: Any,
    index: int,
    *,
    architecture: Mapping[str, Any],
    dependencies: Mapping[str, Mapping[str, Any]],
) -> tuple[set[int], bool]:
    where = f"build_coverage.entries[{index}]"
    _exact_keys(value, (
        "admission_state_id", "aliases", "build_id", "byte_count", "dt_needed",
        "elf_kind_id", "elf_machine", "evidence_state_id", "hidden_wrapper_symbols",
        "import_relocations",
        "link_command_sha256", "link_map_sha256", "logical_id", "module_selection",
        "realloc_zero_policy_id", "realloc_zero_semantic_oracle_sha256", "repo_path",
        "role_ids", "sha256", "shared_runtime_binding", "soname", "symbol_origins",
        "wrap_options",
    ), where)
    admission = _known_id(
        value["admission_state_id"], ADMISSION_STATES, f"{where}.admission_state_id"
    )
    aliases = _sorted_unique(value["aliases"], _relative_path, f"{where}.aliases")
    build_id = _build_id(value["build_id"], f"{where}.build_id")
    if not int(build_id, 16):
        raise CoverageError(f"{where}.build_id: all-zero descriptor forbidden")
    _uint(value["byte_count"], 64, f"{where}.byte_count", nonzero=True)
    dt_needed = _sorted_unique(value["dt_needed"], _soname, f"{where}.dt_needed")
    elf_kind = _known_id(value["elf_kind_id"], ELF_KINDS, f"{where}.elf_kind_id")
    elf_machine = _uint(value["elf_machine"], 16, f"{where}.elf_machine", nonzero=True)
    evidence = _known_id(
        value["evidence_state_id"], EVIDENCE_STATES, f"{where}.evidence_state_id"
    )
    wrappers = _sorted_unique(
        value["hidden_wrapper_symbols"], _symbol, f"{where}.hidden_wrapper_symbols"
    )
    _nonzero_hash(value["link_command_sha256"], f"{where}.link_command_sha256")
    _nonzero_hash(value["link_map_sha256"], f"{where}.link_map_sha256")
    _identifier(value["logical_id"], f"{where}.logical_id")
    _validate_selection(value["module_selection"], f"{where}.module_selection")
    repo_path = _relative_path(value["repo_path"], f"{where}.repo_path")
    if repo_path in aliases:
        raise CoverageError(f"{where}.aliases: canonical repo_path must not be duplicated")
    role_ids = _sorted_unique(
        value["role_ids"],
        lambda item, location: _known_id(item, ROLES, location),
        f"{where}.role_ids",
    )
    if not role_ids:
        raise CoverageError(f"{where}.role_ids: at least one applicable role required")
    _nonzero_hash(value["sha256"], f"{where}.sha256")
    if elf_kind == 1:
        if value["soname"] is not None:
            raise CoverageError(f"{where}.soname: executable requires null")
    elif value["soname"] is not None:
        _soname(value["soname"], f"{where}.soname")

    origins = value["symbol_origins"]
    if not isinstance(origins, list):
        raise CoverageError(f"{where}.symbol_origins: array required")
    origin_rows = [
        _validate_symbol_origin(row, origin_index, architecture=architecture)
        for origin_index, row in enumerate(origins)
    ]
    origin_ids = [row[0] for row in origin_rows]
    if origin_ids != sorted(origin_ids) or len(origin_ids) != len(set(origin_ids)):
        raise CoverageError(f"{where}.symbol_origins: origin_id order/uniqueness required")
    relocations = value["import_relocations"]
    if not isinstance(relocations, list):
        raise CoverageError(f"{where}.import_relocations: array required")
    relocation_origins: list[int] = []
    for relocation_index, relocation in enumerate(relocations):
        location = f"{where}.import_relocations[{relocation_index}]"
        _exact_keys(relocation, ("origin_id", "relocation_kind_id"), location)
        relocation_origins.append(
            _uint(relocation["origin_id"], 16, f"{location}.origin_id", nonzero=True)
        )
        _known_id(
            relocation["relocation_kind_id"], RELOCATION_KINDS,
            f"{location}.relocation_kind_id",
        )
    if relocation_origins != origin_ids:
        raise CoverageError(f"{where}.import_relocations: exact import-origin coverage required")
    supported_ids = [origin_id for origin_id, kind, _ in origin_rows if kind == 1]
    has_new_import = any(kind == 3 for _, kind, _ in origin_rows)
    api_rules = {row["api_id"]: row for row in API_RULES}
    expected_wrappers = sorted(api_rules[api_id]["wrapper_symbol"] for api_id in supported_ids)
    options = _sorted_unique(
        value["wrap_options"],
        lambda item, location: _string(item, location, maximum_bytes=63),
        f"{where}.wrap_options",
    )
    expected_options = sorted(api_rules[api_id]["wrap_option"] for api_id in supported_ids)

    failures: set[int] = set()
    if elf_machine != architecture["elf_machine"]:
        failures.add(12)
        if admission != 40:
            raise CoverageError(f"{where}.admission_state_id: unsupported ABI requires 40")
    if evidence != 1:
        failures.update((14, 50))
        if admission != 42:
            raise CoverageError(f"{where}.admission_state_id: unverified identity requires 42")
    if has_new_import:
        failures.add(20)
        if admission != 60:
            raise CoverageError(f"{where}.admission_state_id: new import requires 60")
    if wrappers != expected_wrappers or options != expected_options:
        failures.add(41)
        if admission != 62:
            raise CoverageError(f"{where}.admission_state_id: wrapper mismatch requires 62")

    runtime = value["shared_runtime_binding"]
    if supported_ids:
        if runtime is None:
            failures.add(62)
        else:
            location = f"{where}.shared_runtime_binding"
            _exact_keys(runtime, ("dependency_id", "evidence_state_id", "soname"), location)
            dependency_id = _identifier(runtime["dependency_id"], f"{location}.dependency_id")
            runtime_evidence = _known_id(
                runtime["evidence_state_id"], EVIDENCE_STATES, f"{location}.evidence_state_id"
            )
            runtime_soname = _soname(runtime["soname"], f"{location}.soname")
            dependency = dependencies.get(dependency_id)
            if (
                dependency is None
                or runtime_evidence != 1
                or dependency["evidence_state_id"] != 1
                or dependency["name"] != runtime_soname
                or runtime_soname not in dt_needed
                or runtime_soname != "libc.so.6"
            ):
                failures.add(62)
    elif runtime is not None:
        raise CoverageError(f"{where}.shared_runtime_binding: zero-import row requires null")

    admits_realloc = bool({3, 5} & set(supported_ids))
    policy_id = value["realloc_zero_policy_id"]
    oracle = value["realloc_zero_semantic_oracle_sha256"]
    if policy_id is not None:
        _known_id(
            policy_id,
            REALLOC_ZERO_POLICIES,
            f"{where}.realloc_zero_policy_id",
        )
    if oracle is not None:
        if _nonzero_hash(
            oracle, f"{where}.realloc_zero_semantic_oracle_sha256"
        ) != REALLOC_ZERO_SEMANTIC_ORACLE_BINDING["sha256"]:
            raise CoverageError(
                f"{where}.realloc_zero_semantic_oracle_sha256: exact semantic member digest required"
            )
    if admits_realloc:
        if policy_id is None or oracle is None:
            failures.add(13)
            if admission != 41:
                raise CoverageError(f"{where}.admission_state_id: missing realloc oracle requires 41")
    elif policy_id is not None or oracle is not None:
        raise CoverageError(f"{where}: realloc oracle fields forbidden without realloc import")

    if admission == 61:
        failures.add(21)
    elif admission == 62 and not failures:
        failures.add(22)
    nominal = not failures
    if nominal and admission not in (20, 21, 22):
        expected_admission = 1 if supported_ids else 2
        if admission != expected_admission:
            raise CoverageError(
                f"{where}.admission_state_id: expected {expected_admission} for exact import census"
            )
    if failures and admission in (1, 2):
        raise CoverageError(f"{where}.admission_state_id: admitted state contradicts failures")
    return failures, nominal or admission in (20, 21, 22)


def validate_build_coverage(
    value: Mapping[str, Any], *, api_definition_sha256: str
) -> None:
    where = "build_coverage"
    _exact_keys(value, (
        "api_definition", "architecture_id", "build_identity", "catalog_id",
        "dependencies", "entries", "evidence_origin_id", "failure_ids",
        "policy", "schema", "verdict_id", "version",
    ), where)
    if value["catalog_id"] != "oai_memprof_build_coverage":
        raise CoverageError(f"{where}.catalog_id: exact ID required")
    _version_object(value["version"], f"{where}.version")
    _definition_binding(
        value["policy"], f"{where}.policy", object_type=9,
        path=POLICY_ARCHIVE_PATH, digest=POLICY_SHA256,
    )
    _definition_binding(
        value["schema"], f"{where}.schema", object_type=10,
        path=INSTANCE_SCHEMA_ARCHIVE_PATH, digest=INSTANCE_SCHEMA_SHA256,
    )
    api_digest = _nonzero_hash(api_definition_sha256, "api_definition_sha256")
    _definition_binding(
        value["api_definition"], f"{where}.api_definition", object_type=4,
        path="catalog/api.json", digest=api_digest,
    )
    architecture_id = _known_id(
        value["architecture_id"], ARCHITECTURES, f"{where}.architecture_id"
    )
    architecture = next(row for row in ARCHITECTURES if row["architecture_id"] == architecture_id)
    identity = _validate_build_identity(value["build_identity"])
    expected_target = "x86_64-linux-gnu" if architecture_id == 1 else "aarch64-linux-gnu"
    if identity["target_triple"] != expected_target:
        raise CoverageError(f"{where}.build_identity.target_triple: architecture mismatch")

    dependencies_value = value["dependencies"]
    if not isinstance(dependencies_value, list) or not dependencies_value:
        raise CoverageError(f"{where}.dependencies: nonempty identity array required")
    dependency_rows = [
        _validate_dependency(row, index) for index, row in enumerate(dependencies_value)
    ]
    dependency_ids = [row["dependency_id"] for row in dependency_rows]
    if dependency_ids != sorted(dependency_ids) or len(dependency_ids) != len(set(dependency_ids)):
        raise CoverageError(f"{where}.dependencies: dependency_id order/uniqueness required")
    dependencies = {row["dependency_id"]: row for row in dependency_rows}

    entries = value["entries"]
    if not isinstance(entries, list) or not entries:
        raise CoverageError(f"{where}.entries: at least one logical ELF required")
    row_results = [
        _validate_logical_elf(
            row, index, architecture=architecture, dependencies=dependencies
        ) for index, row in enumerate(entries)
    ]
    logical_ids = [row["logical_id"] for row in entries]
    if logical_ids != sorted(logical_ids) or len(logical_ids) != len(set(logical_ids)):
        raise CoverageError(f"{where}.entries: logical_id order/uniqueness required")
    primary_id = identity["primary_logical_elf_id"]
    primary = next((row for row in entries if row["logical_id"] == primary_id), None)
    if primary is None or primary["elf_kind_id"] != 1:
        raise CoverageError(f"{where}.build_identity.primary_logical_elf_id: executable row required")

    evidence_origin = _known_id(
        value["evidence_origin_id"], EVIDENCE_ORIGINS, f"{where}.evidence_origin_id"
    )
    failure_ids = _sorted_unique(
        value["failure_ids"],
        lambda item, location: _known_id(item, FAILURES, location),
        f"{where}.failure_ids",
    )
    required_failures = set().union(*(result[0] for result in row_results))
    if identity["dirty"]:
        required_failures.add(60)
    if any(row["evidence_state_id"] != 1 for row in dependency_rows):
        required_failures.update((50, 62))
    if not required_failures.issubset(failure_ids):
        raise CoverageError(f"{where}.failure_ids: missing causal failure IDs")
    if evidence_origin == 2 and 90 not in failure_ids:
        raise CoverageError(f"{where}.failure_ids: synthetic fixture requires failure 90")

    verdict = _known_id(value["verdict_id"], VERDICTS, f"{where}.verdict_id")
    complete = (
        evidence_origin == 1 and not failure_ids and not identity["dirty"]
        and all(row["evidence_state_id"] == 1 for row in dependency_rows)
        and all(result[1] for result in row_results)
    )
    if complete != (verdict == 1):
        raise CoverageError(f"{where}.verdict_id: build-complete verdict is derived")
    if not complete and verdict not in (10, 20):
        raise CoverageError(f"{where}.verdict_id: failed build coverage must be ineligible/unsupported")
    if evidence_origin == 2 and verdict != 10:
        raise CoverageError(f"{where}.verdict_id: synthetic fixture is always ineligible")


def validate_build_coverage_bytes(raw: bytes, *, api_definition_sha256: str) -> dict[str, Any]:
    value = parse_canonical(raw)
    validate_build_coverage(value, api_definition_sha256=api_definition_sha256)
    return value


def _validate_runtime_identity(value: Any, where: str) -> Mapping[str, Any]:
    _exact_keys(value, (
        "build_id", "byte_count", "hidden_wrapper_symbols_sha256",
        "import_relocations_sha256",
        "module_map_sha256", "sha256", "shared_runtime_binding_sha256",
        "symbol_origins_sha256",
    ), where)
    build_id = _build_id(value["build_id"], f"{where}.build_id")
    if not int(build_id, 16):
        raise CoverageError(f"{where}.build_id: all-zero descriptor forbidden")
    _uint(value["byte_count"], 64, f"{where}.byte_count", nonzero=True)
    for field in (
        "hidden_wrapper_symbols_sha256", "import_relocations_sha256",
        "module_map_sha256", "sha256",
        "shared_runtime_binding_sha256", "symbol_origins_sha256",
    ):
        _nonzero_hash(value[field], f"{where}.{field}")
    return value


def _validate_classifications(
    value: Any, where: str
) -> list[tuple[int | None, int]]:
    if not isinstance(value, list):
        raise CoverageError(f"{where}: array required")
    rows: list[tuple[int | None, int]] = []
    for index, row in enumerate(value):
        location = f"{where}[{index}]"
        _exact_keys(row, ("classification_id", "origin_id"), location)
        classification = _known_id(
            row["classification_id"], CLASSIFICATIONS, f"{location}.classification_id"
        )
        origin = None if row["origin_id"] is None else _uint(
            row["origin_id"], 16, f"{location}.origin_id", nonzero=True
        )
        rows.append((origin, classification))
    order = [(UINT64_MAX if origin is None else origin, classification) for origin, classification in rows]
    if order != sorted(order) or len(rows) != len(set(rows)):
        raise CoverageError(f"{where}: strict origin order and uniqueness required")
    return rows


def _validate_population_row(
    value: Any,
    index: int,
    *,
    process_generation: int,
    logical: Mapping[str, Any] | None,
) -> tuple[set[int], bool]:
    where = f"run_coverage.module_population[{index}]"
    _exact_keys(value, (
        "admission_state_id", "build_logical_id", "classifications", "configured",
        "load_generation", "load_state_id", "loaded_path", "logical_id", "observed",
        "runtime_identity",
    ), where)
    admission = _known_id(
        value["admission_state_id"], ADMISSION_STATES, f"{where}.admission_state_id"
    )
    build_logical_id = value["build_logical_id"]
    if build_logical_id is not None:
        build_logical_id = _identifier(build_logical_id, f"{where}.build_logical_id")
    logical_id = _identifier(value["logical_id"], f"{where}.logical_id")
    classifications = _validate_classifications(value["classifications"], f"{where}.classifications")
    configured = _boolean(value["configured"], f"{where}.configured")
    observed = _boolean(value["observed"], f"{where}.observed")
    load_state = _known_id(value["load_state_id"], LOAD_STATES, f"{where}.load_state_id")
    runtime = value["runtime_identity"]
    loaded_path = value["loaded_path"]
    load_generation = value["load_generation"]
    if observed:
        _host_path(loaded_path, f"{where}.loaded_path")
        if _uint(load_generation, 64, f"{where}.load_generation", nonzero=True) != process_generation:
            raise CoverageError(f"{where}.load_generation: process generation mismatch")
        runtime = _validate_runtime_identity(runtime, f"{where}.runtime_identity")
    elif any(item is not None for item in (loaded_path, load_generation, runtime)):
        raise CoverageError(f"{where}: unloaded row requires null load identity")

    failures: set[int] = set()
    identity_ok = True
    if logical is None:
        if (
            build_logical_id is not None or not observed or configured or load_state != 20
            or admission != 61 or classifications != [(None, 61)]
        ):
            raise CoverageError(f"{where}: extra observed ELF must use exact unadmitted shape")
        failures.add(21)
        return failures, False

    if build_logical_id != logical_id or logical_id != logical["logical_id"]:
        raise CoverageError(f"{where}.build_logical_id: exact logical ELF binding required")
    if admission != logical["admission_state_id"]:
        raise CoverageError(f"{where}.admission_state_id: must copy build admission state")
    expected_classifications = [
        (origin["origin_id"], origin["classification_id"])
        for origin in logical["symbol_origins"]
    ]
    if classifications != expected_classifications:
        raise CoverageError(f"{where}.classifications: per-origin build classifications differ")

    if observed:
        assert isinstance(runtime, Mapping)
        expected = _expected_runtime_identity(logical, runtime["module_map_sha256"])
        stable_fields = ("build_id", "byte_count", "sha256")
        semantic_fields = (
            "hidden_wrapper_symbols_sha256", "import_relocations_sha256",
            "shared_runtime_binding_sha256",
            "symbol_origins_sha256",
        )
        if any(runtime[field] != expected[field] for field in stable_fields):
            failures.add(31)
            identity_ok = False
        if any(runtime[field] != expected[field] for field in semantic_fields):
            failures.add(32)
            identity_ok = False
        if configured:
            if load_state == 21:
                failures.add(33)
            elif load_state != 1:
                raise CoverageError(f"{where}.load_state_id: configured observed row requires 1 or 21")
        else:
            if load_state != 20:
                raise CoverageError(f"{where}.load_state_id: unconfigured observed row requires 20")
            failures.add(34)
    else:
        expected_state = 10 if configured else 11
        if load_state != expected_state:
            raise CoverageError(f"{where}.load_state_id: exact unloaded selection state required")
        if configured:
            failures.add(30)

    row_eligible = (
        admission in (1, 2) and identity_ok
        and ((configured and observed and load_state == 1) or
             (not configured and not observed and load_state == 11))
    )
    return failures, row_eligible


def _active_realloc_policy_pairs(
    population: Sequence[Mapping[str, Any]],
    build_rows: Mapping[str, Mapping[str, Any]],
    row_eligible: Sequence[bool],
) -> frozenset[tuple[int, str]]:
    """Derive exact active realloc-family policy/oracle pairs."""

    pairs: set[tuple[int, str]] = set()
    for row, eligible in zip(population, row_eligible, strict=True):
        if not eligible or not (
            row["admission_state_id"] == 1
            and row["configured"]
            and row["observed"]
            and row["load_state_id"] == 1
        ):
            continue
        logical = build_rows[row["build_logical_id"]]
        if not any(
            origin["origin_kind_id"] == 1
            and origin["classification_id"] == 1
            and origin["api_id"] in (3, 5)
            for origin in logical["symbol_origins"]
        ):
            continue
        policy_id = logical["realloc_zero_policy_id"]
        oracle = logical["realloc_zero_semantic_oracle_sha256"]
        if type(policy_id) is not int or not isinstance(oracle, str):
            raise CoverageError(
                "realloc_policy_resolution: validated pair unavailable"
            )
        pairs.add((policy_id, oracle))
    return frozenset(pairs)


def validate_run_coverage(
    value: Mapping[str, Any],
    *,
    build_coverage: Mapping[str, Any],
    api_definition_sha256: str,
    expected_configuration_instance_sha256: str,
) -> None:
    """Validate one generated run against its exact generated build instance."""

    validate_build_coverage(
        build_coverage, api_definition_sha256=api_definition_sha256
    )
    where = "run_coverage"
    _exact_keys(value, (
        "build_coverage", "catalog_id", "configuration_instance_sha256",
        "eligible_exact_domain", "evidence_origin_id", "failure_ids",
        "module_population", "policy", "process_generation", "process_uuid",
        "role_id", "run_uuid", "schema", "snapshot_state_id", "verdict_id",
        "version",
    ), where)
    if value["catalog_id"] != "oai_memprof_run_coverage":
        raise CoverageError(f"{where}.catalog_id: exact ID required")
    _version_object(value["version"], f"{where}.version")
    _definition_binding(
        value["policy"], f"{where}.policy", object_type=9,
        path=POLICY_ARCHIVE_PATH, digest=POLICY_SHA256,
    )
    _definition_binding(
        value["schema"], f"{where}.schema", object_type=10,
        path=INSTANCE_SCHEMA_ARCHIVE_PATH, digest=INSTANCE_SCHEMA_SHA256,
    )
    build_digest = sha256_hex(canonical_bytes(build_coverage))
    _instance_binding(
        value["build_coverage"], f"{where}.build_coverage", object_kind=8,
        path=BUILD_COVERAGE_ARCHIVE_PATH, digest=build_digest,
    )
    process_generation = _uint(
        value["process_generation"], 64, f"{where}.process_generation", nonzero=True
    )
    _uuid(value["process_uuid"], f"{where}.process_uuid")
    _uuid(value["run_uuid"], f"{where}.run_uuid")
    role_id = _known_id(value["role_id"], ROLES, f"{where}.role_id")
    if value["snapshot_state_id"] != 1:
        raise CoverageError(f"{where}.snapshot_state_id: pre-ACTIVE snapshot state 1 required")
    configuration_digest = _nonzero_hash(
        value["configuration_instance_sha256"],
        f"{where}.configuration_instance_sha256",
    )
    expected_configuration_digest = _nonzero_hash(
        expected_configuration_instance_sha256,
        "expected_configuration_instance_sha256",
    )

    build_rows = {row["logical_id"]: row for row in build_coverage["entries"]}
    population = value["module_population"]
    if not isinstance(population, list) or not population:
        raise CoverageError(f"{where}.module_population: nonempty exact union required")
    logical_ids = []
    represented_build_ids: list[str] = []
    row_results: list[tuple[set[int], bool]] = []
    for index, row in enumerate(population):
        if not isinstance(row, dict):
            raise CoverageError(f"{where}.module_population[{index}]: object required")
        logical_id = row.get("logical_id")
        logical = build_rows.get(logical_id) if isinstance(logical_id, str) else None
        result = _validate_population_row(
            row, index, process_generation=process_generation, logical=logical
        )
        logical_ids.append(logical_id)
        if row["build_logical_id"] is not None:
            represented_build_ids.append(row["build_logical_id"])
        row_results.append(result)
    if logical_ids != sorted(logical_ids) or len(logical_ids) != len(set(logical_ids)):
        raise CoverageError(f"{where}.module_population: logical_id order/uniqueness required")
    if sorted(represented_build_ids) != sorted(build_rows):
        raise CoverageError(
            f"{where}.module_population: every build logical ELF must reconcile exactly once"
        )

    primary_id = build_coverage["build_identity"]["primary_logical_elf_id"]
    primary_population = next(row for row in population if row["logical_id"] == primary_id)
    if not primary_population["configured"]:
        raise CoverageError(f"{where}.module_population: primary executable must be configured")
    if role_id not in build_rows[primary_id]["role_ids"]:
        raise CoverageError(f"{where}.role_id: primary executable does not admit role")

    evidence_origin = _known_id(
        value["evidence_origin_id"], EVIDENCE_ORIGINS, f"{where}.evidence_origin_id"
    )
    failure_ids = _sorted_unique(
        value["failure_ids"],
        lambda item, location: _known_id(item, FAILURES, location),
        f"{where}.failure_ids",
    )
    required_failures = set().union(*(result[0] for result in row_results))
    active_realloc_pairs = _active_realloc_policy_pairs(
        population,
        build_rows,
        [result[1] for result in row_results],
    )
    mixed_active_realloc_policies = len(active_realloc_pairs) > 1
    if mixed_active_realloc_policies:
        required_failures.add(22)
    configuration_matches = configuration_digest == expected_configuration_digest
    if not configuration_matches:
        required_failures.add(34)
    if build_coverage["verdict_id"] != 1 and evidence_origin == 1:
        required_failures.add(3)
    if evidence_origin == 2:
        required_failures.add(90)
    if not required_failures.issubset(failure_ids):
        raise CoverageError(f"{where}.failure_ids: missing causal failure IDs")

    claimed_eligible = _boolean(
        value["eligible_exact_domain"], f"{where}.eligible_exact_domain"
    )
    derived_eligible = (
        evidence_origin == 1
        and build_coverage["evidence_origin_id"] == 1
        and build_coverage["verdict_id"] == 1
        and configuration_matches
        and not failure_ids
        and all(result[1] for result in row_results)
    )
    if claimed_eligible != derived_eligible:
        raise CoverageError(f"{where}.eligible_exact_domain: derived value differs")
    verdict = _known_id(value["verdict_id"], VERDICTS, f"{where}.verdict_id")
    if derived_eligible != (verdict == 2):
        raise CoverageError(f"{where}.verdict_id: exact-domain verdict is derived")
    if mixed_active_realloc_policies and verdict != 10:
        raise CoverageError(f"{where}.verdict_id: mixed active realloc policies require 10")
    if not derived_eligible and verdict not in (10, 20):
        raise CoverageError(f"{where}.verdict_id: ineligible run requires 10 or 20")
    if evidence_origin == 2 and (claimed_eligible or verdict != 10):
        raise CoverageError(f"{where}: synthetic fixture cannot be an eligible/final row")


def resolve_run_realloc_zero_policy(
    run_coverage: Mapping[str, Any],
    *,
    build_coverage: Mapping[str, Any],
    api_definition_sha256: str,
    expected_configuration_instance_sha256: str,
) -> ReallocZeroPolicyResolution:
    """Resolve realloc semantics only from a measured eligible run.

    No policy or oracle is accepted from the caller. Both input objects are
    snapshotted as canonical bytes, completely validated, and reduced from the
    exact active module population. Retained negative and synthetic evidence
    cannot mint a trusted result.
    """

    run_snapshot = parse_canonical(canonical_bytes(run_coverage))
    build_snapshot = parse_canonical(canonical_bytes(build_coverage))
    validate_run_coverage(
        run_snapshot,
        build_coverage=build_snapshot,
        api_definition_sha256=api_definition_sha256,
        expected_configuration_instance_sha256=(
            expected_configuration_instance_sha256
        ),
    )
    if not (
        run_snapshot["evidence_origin_id"] == 1
        and run_snapshot["eligible_exact_domain"] is True
        and run_snapshot["verdict_id"] == 2
        and build_snapshot["evidence_origin_id"] == 1
        and build_snapshot["verdict_id"] == 1
    ):
        raise CoverageError(
            "realloc_policy_resolution: measured eligible exact-domain run required"
        )
    build_rows = {
        row["logical_id"]: row for row in build_snapshot["entries"]
    }
    population = run_snapshot["module_population"]
    pairs = _active_realloc_policy_pairs(
        population,
        build_rows,
        [True] * len(population),
    )
    if not pairs:
        return ReallocZeroPolicyResolution("not_applicable", None, None)
    if len(pairs) != 1:
        raise CoverageError(
            "realloc_policy_resolution: active realloc policy is ambiguous"
        )
    policy_id, oracle = next(iter(pairs))
    return ReallocZeroPolicyResolution("resolved", policy_id, oracle)


def validate_run_coverage_bytes(
    raw: bytes,
    *,
    build_coverage: Mapping[str, Any],
    api_definition_sha256: str,
    expected_configuration_instance_sha256: str,
) -> dict[str, Any]:
    value = parse_canonical(raw)
    validate_run_coverage(
        value,
        build_coverage=build_coverage,
        api_definition_sha256=api_definition_sha256,
        expected_configuration_instance_sha256=expected_configuration_instance_sha256,
    )
    return value


def validate_static_member(path: str, raw: bytes) -> dict[str, Any]:
    """Validate one coverage-owned static schema-bundle member by exact bytes."""

    expected = {
        POLICY_ARCHIVE_PATH: (POLICY_BYTES, validate_coverage_policy),
        INSTANCE_SCHEMA_ARCHIVE_PATH: (
            INSTANCE_SCHEMA_BYTES, validate_coverage_instance_schema,
        ),
    }
    if path not in expected:
        raise CoverageError("static_member.path: unknown coverage-owned path")
    value = parse_canonical(raw)
    expected_bytes, validator = expected[path]
    if raw != expected_bytes:
        raise CoverageError("static_member: canonical bytes differ from frozen literal")
    validator(value)
    return value


def static_members() -> dict[str, bytes]:
    """Return fresh immutable-byte bindings; no generated evidence is included."""

    return {
        POLICY_ARCHIVE_PATH: POLICY_BYTES,
        INSTANCE_SCHEMA_ARCHIVE_PATH: INSTANCE_SCHEMA_BYTES,
    }
