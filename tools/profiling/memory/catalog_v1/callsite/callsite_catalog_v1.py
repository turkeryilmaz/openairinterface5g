#!/usr/bin/env python3
"""Concrete schema-v1 callsite catalog validation and record resolution."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


VERSION = {"major": 1, "minor": 0}
DEFINITION_OBJECT_TYPE = 5
DEFINITION_PATH = "definition/callsite-rule-v1.json"
CATALOG_PATH = "catalog/callsite.json"
OBJECT_KIND = 4
OBJECT_FLAGS = 0x1B
FORMAT_ID = 1
SCHEMA_REVISION = 1
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
MAX_OBJECT_ENTRIES = 16_777_216
MAX_OBJECT_BYTES = 268_435_456


class CallsiteError(ValueError):
    """Deterministic callsite-catalog or resolution rejection."""


def _load_semantic() -> Any:
    path = Path(__file__).resolve().parent.parent / "semantic" / "semantic_catalog_v1.py"
    spec = importlib.util.spec_from_file_location("_oai_memprof_semantic_for_callsite", path)
    if spec is None or spec.loader is None:
        raise CallsiteError("semantic: adjacent validator unavailable")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise CallsiteError("semantic: adjacent validator failed to load") from error
    return module


SEMANTIC = _load_semantic()


def sha256_hex(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _fail(where: str, detail: str) -> None:
    raise CallsiteError(f"{where}: {detail}")


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


def _version(value: Any) -> None:
    row = _exact_keys(value, ("major", "minor"), "callsite_catalog.version")
    if type(row["major"]) is not int or type(row["minor"]) is not int or row != VERSION:
        _fail("callsite_catalog.version", "exact typed version 1.0 required")


def _mode(value: Any) -> int:
    mode = _uint(value, 16, "mode_id")
    if mode > 6:
        _fail("mode_id", "outside frozen mode catalog 0..6")
    return mode


def _schema(value: Any, expected_sha256: str) -> None:
    row = _exact_keys(value, ("object_type", "path", "sha256"), "callsite_catalog.schema")
    if _uint(row["object_type"], 16, "callsite_catalog.schema.object_type", nonzero=True) != 5:
        _fail("callsite_catalog.schema.object_type", "exact object type 5 required")
    if row["path"] != DEFINITION_PATH:
        _fail("callsite_catalog.schema.path", f"exact path {DEFINITION_PATH!r} required")
    try:
        observed = SEMANTIC._hash(row["sha256"], "callsite_catalog.schema.sha256")
        expected = SEMANTIC._hash(expected_sha256, "expected_definition_sha256")
    except Exception as error:
        raise CallsiteError(str(error)) from error
    if observed != expected:
        _fail("callsite_catalog.schema.sha256", "definition digest mismatch")


def validate_callsite_catalog(
    value: Mapping[str, Any],
    *,
    definition_sha256: str,
    expected_process_generation: int,
    mode_id: int,
    module_keys: Iterable[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Validate exact rows and return resolved ``(generation, callsite_id)`` keys."""

    top = _exact_keys(value, ("catalog_id", "entries", "schema", "version"), "callsite_catalog")
    if top["catalog_id"] != "oai_memprof_callsite":
        _fail("callsite_catalog.catalog_id", "exact ID required")
    _version(top["version"])
    _schema(top["schema"], definition_sha256)
    generation = _uint(
        expected_process_generation, 64, "expected_process_generation", nonzero=True
    )
    mode = _mode(mode_id)
    entries = top["entries"]
    if not isinstance(entries, list):
        _fail("callsite_catalog.entries", "array required")
    if mode == 5 and not entries:
        _fail("callsite_catalog.entries", "A05 requires at least one resolved callsite row")
    if mode != 5 and entries:
        _fail("callsite_catalog.entries", "only A05 permits observed callsite rows")
    modules: set[tuple[int, int]] = set()
    for index, key in enumerate(module_keys):
        where = f"module_keys[{index}]"
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            _fail(where, "exact (module_generation,module_id) pair required")
        modules.add(
            (
                _uint(key[0], 64, f"{where}[0]", nonzero=True),
                _uint(key[1], 32, f"{where}[1]", nonzero=True),
            )
        )
    keys: list[tuple[int, int]] = []
    for index, value_row in enumerate(entries):
        where = f"callsite_catalog.entries[{index}]"
        row = _exact_keys(
            value_row,
            ("callsite_id", "module_generation", "module_id", "process_generation", "raw_address"),
            where,
        )
        callsite_id = _uint(row["callsite_id"], 32, f"{where}.callsite_id", nonzero=True)
        module_generation = _uint(
            row["module_generation"], 64, f"{where}.module_generation", nonzero=True
        )
        module_id = _uint(row["module_id"], 32, f"{where}.module_id", nonzero=True)
        row_generation = _uint(
            row["process_generation"], 64, f"{where}.process_generation", nonzero=True
        )
        raw_address = _uint(row["raw_address"], 64, f"{where}.raw_address")
        if row_generation != generation:
            _fail(f"{where}.process_generation", "active process-generation mismatch")
        if (module_generation, module_id) not in modules:
            _fail(where, "unresolved module generation/ID")
        key = (row_generation, callsite_id)
        if keys and key <= keys[-1]:
            _fail("callsite_catalog.entries", "strict process-generation/callsite-ID order required")
        keys.append(key)
    return set(keys)


def validate_callsite_catalog_bytes(raw: bytes, **kwargs: Any) -> tuple[dict[str, Any], set[tuple[int, int]]]:
    try:
        value = SEMANTIC.parse_canonical(raw)
    except Exception as error:
        raise CallsiteError(str(error)) from error
    return value, validate_callsite_catalog(value, **kwargs)


def serialize_callsite_catalog(
    *,
    definition_sha256: str,
    entries: Sequence[Mapping[str, Any]],
    expected_process_generation: int,
    mode_id: int,
    module_keys: Iterable[tuple[int, int]],
) -> bytes:
    copied: list[dict[str, Any]] = []
    for index, row in enumerate(entries):
        _exact_keys(
            row,
            ("callsite_id", "module_generation", "module_id", "process_generation", "raw_address"),
            f"callsite_catalog.entries[{index}]",
        )
        copied.append(dict(row))
    value = {
        "catalog_id": "oai_memprof_callsite",
        "entries": copied,
        "schema": {
            "object_type": DEFINITION_OBJECT_TYPE,
            "path": DEFINITION_PATH,
            "sha256": definition_sha256,
        },
        "version": dict(VERSION),
    }
    validate_callsite_catalog(
        value,
        definition_sha256=definition_sha256,
        expected_process_generation=expected_process_generation,
        mode_id=mode_id,
        module_keys=module_keys,
    )
    return SEMANTIC.canonical_bytes(value)


def semantic_object_descriptor(
    raw: bytes,
    *,
    definition_sha256: str,
    expected_process_generation: int,
    mode_id: int,
    module_keys: Iterable[tuple[int, int]],
) -> dict[str, Any]:
    """Return metadata derived from one exact validated canonical snapshot.

    This is an archive descriptor, not the 64-byte binary object-table row.
    The descriptor deliberately includes the archive path; its digest is hex.
    """

    if not isinstance(raw, bytes):
        _fail("catalog_bytes", "bytes required")
    if len(raw) > MAX_OBJECT_BYTES:
        _fail("catalog_bytes", "object byte limit exceeded")
    value, _ = validate_callsite_catalog_bytes(
        raw,
        definition_sha256=definition_sha256,
        expected_process_generation=expected_process_generation,
        mode_id=mode_id,
        module_keys=module_keys,
    )
    entry_count = len(value["entries"])
    if entry_count > MAX_OBJECT_ENTRIES:
        _fail("callsite_catalog.entries", "object entry limit exceeded")
    return {
        "byte_count": len(raw),
        "entry_count": entry_count,
        "object_flags": OBJECT_FLAGS,
        "format_id": FORMAT_ID,
        "object_kind": OBJECT_KIND,
        "path": CATALOG_PATH,
        "schema_revision": SCHEMA_REVISION,
        "sha256": sha256_hex(raw),
    }


def validate_semantic_object_descriptor(
    descriptor: Mapping[str, Any], raw: bytes, **catalog_arguments: Any
) -> None:
    expected = semantic_object_descriptor(raw, **catalog_arguments)
    row = _exact_keys(
        descriptor,
        (
            "byte_count",
            "entry_count",
            "format_id",
            "object_flags",
            "object_kind",
            "path",
            "schema_revision",
            "sha256",
        ),
        "callsite_object_descriptor",
    )
    if any(
        type(row[field]) is not type(expected[field])
        or row[field] != expected[field]
        for field in expected
    ):
        _fail("callsite_object_descriptor", "exact validated snapshot metadata required")


def reconcile_records(
    resolved_keys: set[tuple[int, int]],
    *,
    process_generation: int,
    mode_id: int,
    records: Sequence[Any],
    complete: bool = True,
) -> None:
    generation = _uint(process_generation, 64, "process_generation", nonzero=True)
    mode = _mode(mode_id)
    if type(complete) is not bool:
        _fail("complete", "boolean required")
    referenced: set[tuple[int, int]] = set()
    for index, record in enumerate(records):
        try:
            callsite_id = _uint(record.callsite_id, 32, f"records[{index}].callsite_id")
        except AttributeError as error:
            raise CallsiteError(f"records[{index}]: callsite_id field required") from error
        if mode == 5:
            if callsite_id == 0 or (generation, callsite_id) not in resolved_keys:
                _fail(f"records[{index}].callsite_id", "A05 requires exact resolved nonzero ID")
            referenced.add((generation, callsite_id))
        elif callsite_id != 0:
            _fail(f"records[{index}].callsite_id", "canonical zero required outside A05")
    if mode == 5 and complete and referenced != resolved_keys:
        _fail("callsite_catalog.entries", "COMPLETE A05 record/catalog sets must be equal")
    if mode != 5 and resolved_keys:
        _fail("callsite_catalog.entries", "empty catalog required outside A05")


__all__ = [
    "CATALOG_PATH",
    "CallsiteError",
    "DEFINITION_PATH",
    "reconcile_records",
    "semantic_object_descriptor",
    "serialize_callsite_catalog",
    "sha256_hex",
    "validate_callsite_catalog",
    "validate_callsite_catalog_bytes",
    "validate_semantic_object_descriptor",
]
