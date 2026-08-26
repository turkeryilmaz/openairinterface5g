#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0

"""Strict schema-v1 status, receipt, and manifest binding primitives.

This version-controlled module closes only the acyclic structural relationship
between a pre-footer status object, a post-close verification receipt, and an
archive manifest.  It performs no file I/O and does not independently verify
stream bytes, ELF coverage, catalogs, diagnostics, or scientific eligibility.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_PRE_FOOTER = "oai.memprof.pre-footer-status/v1"
SCHEMA_POST_CLOSE = "oai.memprof.post-close-verification/v1"
SCHEMA_MANIFEST = "oai.memprof.archive-manifest/v1"
PRE_FOOTER_PATH = "status/pre-footer-status.json"
POST_CLOSE_PATH = "status/post-close-verification.json"
MANIFEST_PATH = "manifest.json"

LIFECYCLE_COMPLETE = 5
LIFECYCLE_FAILED = 6
LIFECYCLE_INCOMPLETE = 7
WRITER_CLOSED_VERIFIED = 5
WRITER_IO_FAILED_CLOSED_VERIFIED = 6
STAGE_PRE_SYNC_MATERIAL_FROZEN = 6
REASON_NONE = 0
REASON_QUIESCENCE_TIMEOUT = 1
REASON_RING_DRAIN_FAILED = 2
REASON_CATALOG_FREEZE_FAILED = 3
REASON_DIAGNOSTICS_FREEZE_FAILED = 4
REASON_PAYLOAD_SYNC_FAILED = 5
REASON_PAYLOAD_IO_FAILED = 6
REASON_COUNTER_OR_TIME_INVALID = 7
REASON_OPERATOR_CANCELLED = 8
REASON_UNSUPPORTED_SCOPE = 9
SCOPE_MEASUREMENT_INTERVAL = 1
SCOPE_PROCESS_LIFETIME = 2
TERMINAL_FLAGS_MASK = 0x1FFFF
COMPLETE_REQUIRED_FLAGS = 0x0FFF

# Schema-v1 trust-boundary ceilings. The status and receipt schemas are
# fixed-size apart from an archive-relative path limited to 4096 UTF-8 bytes,
# so 16 KiB leaves deliberate headroom. A manifest may describe a large
# archive: 8192 entries with maximum-sized paths fit comfortably below 64 MiB.
MAX_PRE_FOOTER_RAW_BYTES = 16 * 1024
MAX_POST_CLOSE_RAW_BYTES = 16 * 1024
MAX_MANIFEST_RAW_BYTES = 64 * 1024 * 1024
MAX_JSON_NESTING_DEPTH = 64
MAX_MANIFEST_ENTRIES = 8192

_HASH_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"\A[a-z][a-z0-9_]*\Z")
_RESERVED_CHAIN_PATHS = frozenset((PRE_FOOTER_PATH, POST_CLOSE_PATH, MANIFEST_PATH))


class StatusChainError(ValueError):
    """Typed rejection of malformed or inconsistent status-chain evidence."""

    def __init__(self, code: str, field: str, detail: str = ""):
        message = f"{code}: {field}"
        if detail:
            message += f": {detail}"
        super().__init__(message)
        self.code = code
        self.field = field
        self.detail = detail


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class VerifiedStreamIdentity:
    """Trusted output of the separate exact-EOF stream verifier."""

    stream_path: str
    physical_bytes: int
    whole_stream_sha256: str
    pre_footer_status_bytes: int
    pre_footer_status_sha256: str
    footer_preimage_sha256: str
    opening_header_sha256: str
    prefix_sha256: str
    trailer_body_sha256: str
    verifier_definition_sha256: str


@dataclass(frozen=True)
class PromotionBinding:
    pre_footer_sha256: str
    receipt_sha256: str
    stream_sha256: str
    stream_bytes: int
    stream_path: str


def _fail(code: str, field: str, detail: str = "") -> None:
    raise StatusChainError(code, field, detail)


def _reject_float(_: str) -> None:
    _fail("canonical", "number", "floating point is forbidden")


def _reject_constant(_: str) -> None:
    _fail("canonical", "number", "non-finite value is forbidden")


def _pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate", key)
        result[key] = value
    return result


def _check_depth(depth: int, field: str) -> None:
    if depth > MAX_JSON_NESTING_DEPTH:
        _fail("range", field, f"JSON nesting depth exceeds {MAX_JSON_NESTING_DEPTH}")


def _walk(value: Any, field: str = "$", depth: int = 1) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        return
    if isinstance(value, str):
        if "\x00" in value:
            _fail("canonical", field, "NUL is forbidden")
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            _fail("canonical", field, "surrogate code point is forbidden")
        if unicodedata.normalize("NFC", value) != value:
            _fail("canonical", field, "string must be NFC")
        try:
            value.encode("utf-8", "strict")
        except UnicodeError as error:
            raise StatusChainError("canonical", field, "invalid Unicode") from error
        return
    if isinstance(value, list):
        _check_depth(depth, field)
        for index, item in enumerate(value):
            _walk(item, f"{field}[{index}]", depth + 1)
        return
    if isinstance(value, dict):
        _check_depth(depth, field)
        for key, item in value.items():
            if not isinstance(key, str) or not _IDENTIFIER_RE.fullmatch(key):
                _fail("canonical", field, "member names must match [a-z][a-z0-9_]*")
            _walk(item, f"{field}.{key}", depth + 1)
        return
    _fail("canonical", field, f"unsupported JSON type {type(value).__name__}")


def _canonical_string(value: str) -> str:
    encoded: list[str] = ['"']
    for character in value:
        codepoint = ord(character)
        if character == '"':
            encoded.append('\\"')
        elif character == "\\":
            encoded.append("\\\\")
        elif codepoint <= 0x1F or codepoint == 0x7F:
            encoded.append(f"\\u00{codepoint:02x}")
        else:
            encoded.append(character)
    encoded.append('"')
    return "".join(encoded)


def _canonical_value(value: Any, depth: int = 1) -> str:
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
        _check_depth(depth, "$")
        return "[" + ",".join(_canonical_value(item, depth + 1) for item in value) + "]"
    if isinstance(value, dict):
        _check_depth(depth, "$")
        return "{" + ",".join(
            _canonical_string(key) + ":" + _canonical_value(value[key], depth + 1)
            for key in sorted(value)
        ) + "}"
    _fail("canonical", "$", f"unsupported JSON type {type(value).__name__}")


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    if not isinstance(value, dict):
        _fail("type", "$", "top-level object required")
    try:
        _walk(value)
        return _canonical_value(value).encode("utf-8") + b"\n"
    except RecursionError as error:
        raise StatusChainError(
            "depth", "$", f"JSON nesting depth exceeds {MAX_JSON_NESTING_DEPTH}"
        ) from error


def _scan_json_depth(raw: bytes, field: str) -> None:
    """Reject excessive structural depth without decoding or parsing JSON."""

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
            _check_depth(depth, field)
        elif byte in (0x5D, 0x7D):  # ']' or '}'
            depth -= 1


def _parse_canonical(raw: bytes, *, maximum_bytes: int, field: str) -> dict[str, Any]:
    if not isinstance(raw, bytes):
        _fail("type", field, "bytes required")
    if len(raw) > maximum_bytes:
        _fail("range", field, f"canonical bytes exceed {maximum_bytes}")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        _fail("canonical", field, "exactly one final LF required")
    if raw.startswith(b"\xef\xbb\xbf"):
        _fail("canonical", field, "BOM forbidden")
    _scan_json_depth(raw, field)
    try:
        text = raw.decode("utf-8", "strict")
        value = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except StatusChainError:
        raise
    except RecursionError as error:
        raise StatusChainError(
            "depth", field, f"JSON nesting depth exceeds {MAX_JSON_NESTING_DEPTH}"
        ) from error
    except (UnicodeError, json.JSONDecodeError) as error:
        raise StatusChainError("json", field, str(error)) from error
    if not isinstance(value, dict):
        _fail("type", "$", "top-level object required")
    if canonical_bytes(value) != raw:
        _fail("canonical", field, "bytes are not canonical")
    return value


def parse_canonical(raw: bytes) -> dict[str, Any]:
    """Parse canonical JSON under the largest schema-v1 artifact ceiling.

    The artifact-specific binder applies the narrower pre-footer and receipt
    ceilings before parsing those populations.
    """

    return _parse_canonical(raw, maximum_bytes=MAX_MANIFEST_RAW_BYTES, field="raw")


def sha256_hex(raw: bytes) -> str:
    if not isinstance(raw, bytes):
        _fail("type", "raw", "bytes required")
    return hashlib.sha256(raw).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], field: str) -> None:
    expected_set = set(expected)
    observed_set = set(value)
    if observed_set != expected_set:
        missing = sorted(expected_set - observed_set)
        extra = sorted(observed_set - expected_set)
        _fail("keys", field, f"missing={missing!r} extra={extra!r}")


def _uint(value: Any, bits: int, field: str, *, nonzero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("type", field, f"u{bits} required")
    if value < 0 or value >= (1 << bits) or (nonzero and value == 0):
        _fail("range", field, f"u{bits}{' nonzero' if nonzero else ''} required")
    return value


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        _fail("type", field, "boolean required")
    return value


def _string(value: Any, field: str, *, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > maximum:
        _fail("type", field, f"nonempty UTF-8 string <= {maximum} bytes required")
    return value


def _hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        _fail("type", field, "64 lowercase hexadecimal SHA-256 required")
    if value == "0" * 64:
        _fail("range", field, "all-zero SHA-256 forbidden")
    return value


def _relative_path(value: Any, field: str) -> str:
    path = _string(value, field, maximum=4096)
    if path.startswith("/") or "\\" in path or any(part in ("", ".", "..") for part in path.split("/")):
        _fail("path", field, "normalized archive-relative POSIX path required")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in path):
        _fail("path", field, "control character forbidden")
    return path


def _validated_stream_identity(value: Any) -> VerifiedStreamIdentity:
    if type(value) is not VerifiedStreamIdentity:
        _fail("type", "verified_stream", "VerifiedStreamIdentity required")
    return VerifiedStreamIdentity(
        stream_path=_relative_path(value.stream_path, "verified_stream.stream_path"),
        physical_bytes=_uint(
            value.physical_bytes, 64, "verified_stream.physical_bytes", nonzero=True
        ),
        whole_stream_sha256=_hash(
            value.whole_stream_sha256, "verified_stream.whole_stream_sha256"
        ),
        pre_footer_status_bytes=_uint(
            value.pre_footer_status_bytes,
            64,
            "verified_stream.pre_footer_status_bytes",
            nonzero=True,
        ),
        pre_footer_status_sha256=_hash(
            value.pre_footer_status_sha256,
            "verified_stream.pre_footer_status_sha256",
        ),
        footer_preimage_sha256=_hash(
            value.footer_preimage_sha256, "verified_stream.footer_preimage_sha256"
        ),
        opening_header_sha256=_hash(
            value.opening_header_sha256, "verified_stream.opening_header_sha256"
        ),
        prefix_sha256=_hash(value.prefix_sha256, "verified_stream.prefix_sha256"),
        trailer_body_sha256=_hash(
            value.trailer_body_sha256, "verified_stream.trailer_body_sha256"
        ),
        verifier_definition_sha256=_hash(
            value.verifier_definition_sha256,
            "verified_stream.verifier_definition_sha256",
        ),
    )


def _validate_outcome(value: Mapping[str, Any]) -> None:
    lifecycle = _uint(value["lifecycle_state"], 16, "lifecycle_state")
    writer = _uint(value["payload_writer_state"], 16, "payload_writer_state")
    stage = _uint(value["finalization_stage"], 16, "finalization_stage")
    reason = _uint(value["reason_code"], 32, "reason_code")
    scope = _uint(value["scope_kind"], 16, "scope_kind")
    flags = _uint(value["terminal_flags"], 64, "terminal_flags")
    if lifecycle not in (LIFECYCLE_COMPLETE, LIFECYCLE_FAILED, LIFECYCLE_INCOMPLETE):
        _fail("enum", "lifecycle_state")
    if writer not in (WRITER_CLOSED_VERIFIED, WRITER_IO_FAILED_CLOSED_VERIFIED):
        _fail("enum", "payload_writer_state")
    if stage > STAGE_PRE_SYNC_MATERIAL_FROZEN or reason > REASON_UNSUPPORTED_SCOPE:
        _fail("enum", "finalization_stage/reason_code")
    if scope not in (SCOPE_MEASUREMENT_INTERVAL, SCOPE_PROCESS_LIFETIME):
        _fail("enum", "scope_kind")
    if flags & ~TERMINAL_FLAGS_MASK:
        _fail("reserved", "terminal_flags")
    stage_mask = (1 << (stage + 1)) - 1
    if flags & 0x7F != stage_mask:
        _fail("relation", "terminal_flags", "primary stage bits")
    if writer == WRITER_CLOSED_VERIFIED:
        if flags & (1 << 7) == 0 or flags & (1 << 8) == 0:
            _fail("relation", "terminal_flags", "writer state 5 requires bits 7 and 8")
    elif flags & (1 << 8) == 0 or flags & (1 << 7):
        _fail("relation", "terminal_flags", "writer state 6 requires bit 8 and forbids bit 7")

    fixed = {
        REASON_NONE: (LIFECYCLE_COMPLETE, 6, 5),
        REASON_QUIESCENCE_TIMEOUT: (LIFECYCLE_INCOMPLETE, 1, 5),
        REASON_RING_DRAIN_FAILED: (LIFECYCLE_FAILED, 2, 5),
        REASON_CATALOG_FREEZE_FAILED: (LIFECYCLE_FAILED, 3, 5),
        REASON_DIAGNOSTICS_FREEZE_FAILED: (LIFECYCLE_FAILED, 4, 5),
        REASON_PAYLOAD_SYNC_FAILED: (LIFECYCLE_FAILED, 6, 6),
        REASON_UNSUPPORTED_SCOPE: (LIFECYCLE_INCOMPLETE, 0, 5),
    }
    if reason in fixed and (lifecycle, stage, writer) != fixed[reason]:
        _fail("relation", "primary_outcome")
    if reason == REASON_PAYLOAD_IO_FAILED and (lifecycle != LIFECYCLE_FAILED or writer != 6):
        _fail("relation", "primary_outcome")
    if reason in (REASON_COUNTER_OR_TIME_INVALID, REASON_OPERATOR_CANCELLED) and (
        lifecycle != LIFECYCLE_INCOMPLETE or writer not in (5, 6)
    ):
        _fail("relation", "primary_outcome")
    if lifecycle == LIFECYCLE_COMPLETE:
        if scope != SCOPE_MEASUREMENT_INTERVAL or reason != REASON_NONE:
            _fail("relation", "complete_scope_reason")
        if flags & COMPLETE_REQUIRED_FLAGS != COMPLETE_REQUIRED_FLAGS or flags & ((1 << 13) | (1 << 16)):
            _fail("relation", "complete_flags")


def validate_pre_footer(value: Mapping[str, Any]) -> None:
    keys = (
        "active_generation",
        "active_start_counter",
        "active_start_monotonic_raw_ns",
        "cutoff_after_counter",
        "cutoff_before_counter",
        "diagnostic_population_partial",
        "final_counter",
        "final_monotonic_raw_ns",
        "final_realtime_unix_ns",
        "finalization_stage",
        "lifecycle_state",
        "payload_writer_state",
        "process_generation",
        "quiescence_complete_counter",
        "reason_code",
        "schema",
        "scope_kind",
        "terminal_flags",
    )
    _exact_keys(value, keys, "pre_footer")
    if value["schema"] != SCHEMA_PRE_FOOTER:
        _fail("schema", "schema")
    generation = _uint(value["process_generation"], 64, "process_generation", nonzero=True)
    if _uint(value["active_generation"], 64, "active_generation", nonzero=True) != generation:
        _fail("relation", "active_generation")
    _bool(value["diagnostic_population_partial"], "diagnostic_population_partial")
    _validate_outcome(value)
    active = _uint(value["active_start_counter"], 64, "active_start_counter")
    final = _uint(value["final_counter"], 64, "final_counter")
    active_mono = _uint(value["active_start_monotonic_raw_ns"], 64, "active_start_monotonic_raw_ns")
    final_mono = _uint(value["final_monotonic_raw_ns"], 64, "final_monotonic_raw_ns")
    _uint(value["final_realtime_unix_ns"], 64, "final_realtime_unix_ns")
    if active > final or active_mono > final_mono:
        _fail("relation", "active/final")
    stage = value["finalization_stage"]
    before = _uint(value["cutoff_before_counter"], 64, "cutoff_before_counter")
    after = _uint(value["cutoff_after_counter"], 64, "cutoff_after_counter")
    quiescence = _uint(value["quiescence_complete_counter"], 64, "quiescence_complete_counter")
    if stage < 1:
        if before != 0 or after != 0:
            _fail("relation", "cutoff", "unreached fields must be zero")
    elif not active <= before <= after <= final:
        _fail("relation", "cutoff")
    if stage < 2:
        if quiescence != 0:
            _fail("relation", "quiescence_complete_counter", "unreached field must be zero")
    elif not after <= quiescence <= final:
        _fail("relation", "quiescence_complete_counter")
    if value["diagnostic_population_partial"] != bool(value["terminal_flags"] & (1 << 13)):
        _fail("relation", "diagnostic_population_partial")


def validate_post_close(value: Mapping[str, Any]) -> None:
    keys = (
        "appender_close",
        "exact_eof",
        "footer_preimage_sha256",
        "opening_header_sha256",
        "physical_bytes",
        "prefix_sha256",
        "schema",
        "stream_path",
        "trailer_body_sha256",
        "verifier_definition_sha256",
        "whole_stream_sha256",
    )
    _exact_keys(value, keys, "post_close")
    if value["schema"] != SCHEMA_POST_CLOSE:
        _fail("schema", "schema")
    if value["appender_close"] != "success":
        _fail("relation", "appender_close")
    if _bool(value["exact_eof"], "exact_eof") is not True:
        _fail("relation", "exact_eof")
    _relative_path(value["stream_path"], "stream_path")
    _uint(value["physical_bytes"], 64, "physical_bytes", nonzero=True)
    for field in (
        "footer_preimage_sha256",
        "opening_header_sha256",
        "prefix_sha256",
        "trailer_body_sha256",
        "verifier_definition_sha256",
        "whole_stream_sha256",
    ):
        _hash(value[field], field)


def validate_manifest(value: Mapping[str, Any]) -> tuple[ManifestEntry, ...]:
    _exact_keys(value, ("entries", "schema"), "manifest")
    if value["schema"] != SCHEMA_MANIFEST:
        _fail("schema", "schema")
    if not isinstance(value["entries"], list) or not value["entries"]:
        _fail("type", "entries", "nonempty array required")
    if len(value["entries"]) > MAX_MANIFEST_ENTRIES:
        _fail("range", "entries", f"at most {MAX_MANIFEST_ENTRIES} entries permitted")
    entries: list[ManifestEntry] = []
    previous = ""
    for index, row in enumerate(value["entries"]):
        if not isinstance(row, dict):
            _fail("type", f"entries[{index}]")
        _exact_keys(row, ("bytes", "path", "sha256"), f"entries[{index}]")
        path = _relative_path(row["path"], f"entries[{index}].path")
        if path == MANIFEST_PATH:
            _fail("cycle", f"entries[{index}].path", "manifest must exclude itself")
        if path <= previous:
            _fail("order", f"entries[{index}].path")
        previous = path
        entries.append(
            ManifestEntry(
                path=path,
                byte_count=_uint(row["bytes"], 64, f"entries[{index}].bytes"),
                sha256=_hash(row["sha256"], f"entries[{index}].sha256"),
            )
        )
    return tuple(entries)


def bind_complete_candidate(
    pre_footer_raw: bytes,
    receipt_raw: bytes,
    manifest_raw: bytes,
    *,
    verified_stream: VerifiedStreamIdentity,
) -> PromotionBinding:
    """Bind a structurally verified COMPLETE candidate to receipt and manifest.

    ``verified_stream`` must come from the separate accepted exact-EOF stream
    verifier.  This function validates that typed identity, compares every
    receipt identity field with it, and verifies the manifest relationships;
    it cannot turn an unverified stream into COMPLETE scientific evidence.
    """

    trusted_stream = _validated_stream_identity(verified_stream)
    if trusted_stream.stream_path in _RESERVED_CHAIN_PATHS:
        _fail(
            "cycle",
            "verified_stream.stream_path",
            "stream path must differ from all reserved status-chain paths",
        )
    required_paths = (PRE_FOOTER_PATH, POST_CLOSE_PATH, trusted_stream.stream_path)
    if len(set(required_paths)) != 3:
        _fail("relation", "chain_paths", "pre-footer, receipt, and stream must be distinct")

    pre_footer = _parse_canonical(
        pre_footer_raw,
        maximum_bytes=MAX_PRE_FOOTER_RAW_BYTES,
        field="pre_footer_raw",
    )
    receipt = _parse_canonical(
        receipt_raw,
        maximum_bytes=MAX_POST_CLOSE_RAW_BYTES,
        field="receipt_raw",
    )
    manifest = _parse_canonical(
        manifest_raw,
        maximum_bytes=MAX_MANIFEST_RAW_BYTES,
        field="manifest_raw",
    )
    validate_pre_footer(pre_footer)
    validate_post_close(receipt)
    entries = validate_manifest(manifest)
    if pre_footer["lifecycle_state"] != LIFECYCLE_COMPLETE:
        _fail("outcome", "lifecycle_state", "negative terminal retained, not promoted")
    if len(pre_footer_raw) != trusted_stream.pre_footer_status_bytes:
        _fail(
            "binding",
            "verified_stream.pre_footer_status_bytes",
            "object-kind-12 byte count mismatch",
        )
    if sha256_hex(pre_footer_raw) != trusted_stream.pre_footer_status_sha256:
        _fail(
            "binding",
            "verified_stream.pre_footer_status_sha256",
            "object-kind-12 digest mismatch",
        )
    for field in (
        "stream_path",
        "physical_bytes",
        "whole_stream_sha256",
        "footer_preimage_sha256",
        "opening_header_sha256",
        "prefix_sha256",
        "trailer_body_sha256",
        "verifier_definition_sha256",
    ):
        if receipt[field] != getattr(trusted_stream, field):
            _fail("binding", f"receipt.{field}", "trusted stream identity mismatch")

    by_path = {entry.path: entry for entry in entries}
    for path in required_paths:
        if path not in by_path:
            _fail("missing", path)
    expected = {
        PRE_FOOTER_PATH: (len(pre_footer_raw), sha256_hex(pre_footer_raw)),
        POST_CLOSE_PATH: (len(receipt_raw), sha256_hex(receipt_raw)),
        trusted_stream.stream_path: (
            trusted_stream.physical_bytes,
            trusted_stream.whole_stream_sha256,
        ),
    }
    for path, (byte_count, digest) in expected.items():
        entry = by_path[path]
        if entry.byte_count != byte_count or entry.sha256 != digest:
            _fail("binding", path)
    return PromotionBinding(
        pre_footer_sha256=expected[PRE_FOOTER_PATH][1],
        receipt_sha256=expected[POST_CLOSE_PATH][1],
        stream_sha256=trusted_stream.whole_stream_sha256,
        stream_bytes=trusted_stream.physical_bytes,
        stream_path=trusted_stream.stream_path,
    )


__all__ = [
    "MANIFEST_PATH",
    "MAX_JSON_NESTING_DEPTH",
    "MAX_MANIFEST_ENTRIES",
    "MAX_MANIFEST_RAW_BYTES",
    "MAX_POST_CLOSE_RAW_BYTES",
    "MAX_PRE_FOOTER_RAW_BYTES",
    "POST_CLOSE_PATH",
    "PRE_FOOTER_PATH",
    "ManifestEntry",
    "PromotionBinding",
    "StatusChainError",
    "VerifiedStreamIdentity",
    "bind_complete_candidate",
    "canonical_bytes",
    "parse_canonical",
    "sha256_hex",
    "validate_manifest",
    "validate_post_close",
    "validate_pre_footer",
]
