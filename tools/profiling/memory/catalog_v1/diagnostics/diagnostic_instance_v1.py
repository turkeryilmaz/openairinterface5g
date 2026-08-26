#!/usr/bin/env python3
"""Canonical schema-v1 diagnostics-instance validation and projection.

This standard-library-only module implements the concrete object-kind
11 grammar frozen by semantic bundle member 8.  It performs no discovery or
writes and never converts unavailable counter evidence into zero.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any, Mapping, Sequence


UINT16_MAX = (1 << 16) - 1
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
VERSION = {"major": 1, "minor": 0}
DEFINITION_PATH = "definition/diagnostic-v1.json"
INSTANCE_PATH = "status/diagnostics.json"
TOTAL_SATURATED = 1
PARTIAL_COUNTER_POPULATION = 2
TERMINAL_AGGREGATE_SATURATED = 1 << 12
TERMINAL_POPULATION_PARTIAL = 1 << 13


class DiagnosticError(ValueError):
    """Deterministic diagnostics schema or projection rejection."""


def _load_semantic() -> Any:
    path = Path(__file__).resolve().parent.parent / "semantic" / "semantic_catalog_v1.py"
    spec = importlib.util.spec_from_file_location("_oai_memprof_semantic_for_diagnostics", path)
    if spec is None or spec.loader is None:
        raise DiagnosticError("semantic: adjacent validator unavailable")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise DiagnosticError("semantic: adjacent validator failed to load") from error
    return module


SEMANTIC = _load_semantic()
REASON_ROWS = tuple(SEMANTIC.DIAGNOSTIC_REASON_ROWS)
REASON_IDS = tuple(row[0] for row in REASON_ROWS)
REASON_CLASS_FLAGS = {row[0]: row[2] for row in REASON_ROWS}
MODE_ROWS = {row[0]: row for row in SEMANTIC.DIAGNOSTIC_MODE_SCOPE_ROWS}


def sha256_hex(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _fail(where: str, detail: str) -> None:
    raise DiagnosticError(f"{where}: {detail}")


def _exact_keys(value: Any, keys: Sequence[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(where, "object required")
    expected = tuple(sorted(keys))
    actual = tuple(value.keys())
    if actual != expected:
        _fail(where, f"exact keys {expected!r} required, got {actual!r}")
    return value


def _uint(value: Any, bits: int, where: str, *, nonzero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(where, f"u{bits} integer required")
    lower = 1 if nonzero else 0
    upper = (1 << bits) - 1
    if not lower <= value <= upper:
        _fail(where, f"outside u{bits} range")
    return value


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        _fail(where, "boolean required")
    return value


def _hash(value: Any, where: str) -> str:
    try:
        return SEMANTIC._hash(value, where)
    except Exception as error:
        raise DiagnosticError(str(error)) from error


def _version(value: Any, where: str) -> None:
    if (
        not isinstance(value, dict)
        or tuple(value.keys()) != ("major", "minor")
        or type(value["major"]) is not int
        or type(value["minor"]) is not int
        or value != VERSION
    ):
        _fail(where, "exact version 1.0 required")


def _schema_binding(value: Any, expected_sha256: str) -> None:
    row = _exact_keys(value, ("object_type", "path", "sha256"), "diagnostics.schema")
    if _uint(row["object_type"], 16, "diagnostics.schema.object_type", nonzero=True) != 8:
        _fail("diagnostics.schema.object_type", "exact object type 8 required")
    if row["path"] != DEFINITION_PATH:
        _fail("diagnostics.schema.path", f"exact path {DEFINITION_PATH!r} required")
    if _hash(row["sha256"], "diagnostics.schema.sha256") != _hash(
        expected_sha256, "expected_definition_sha256"
    ):
        _fail("diagnostics.schema.sha256", "definition digest mismatch")


def _ready_threads(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        _fail("population.ready_thread_indices", "integer sequence required")
    result = tuple(
        _uint(value, 32, f"population.ready_thread_indices[{index}]", nonzero=True)
        for index, value in enumerate(values)
    )
    if tuple(sorted(set(result))) != result:
        _fail("population.ready_thread_indices", "strict increasing uniqueness required")
    return result


def _mode_scopes(mode_id: int) -> tuple[set[int], set[int], set[int], set[int], set[int]]:
    row = MODE_ROWS.get(mode_id)
    if row is None:
        _fail("diagnostics.mode_id", "exact mode 2, 3, 4, or 5 required")
    _mode, _name, aggregate, not_applicable, producer, registration, writer = row
    return set(aggregate), set(not_applicable), set(producer), set(registration), set(writer)


def _scope_population(
    *,
    mode_id: int,
    generation: int,
    ready_thread_indices: tuple[int, ...],
    producer_population_complete: bool,
    registration_available: bool,
    writer_available: bool,
    aggregate_available: bool,
) -> tuple[set[tuple[int, int, int, int]], dict[int, bool], dict[int, str]]:
    aggregate, not_applicable, producer, registration, writer = _mode_scopes(mode_id)
    allowed: set[tuple[int, int, int, int]] = set()
    partial: dict[int, bool] = {reason_id: False for reason_id in REASON_IDS}
    family: dict[int, str] = {}
    for reason_id in REASON_IDS:
        if reason_id in not_applicable:
            family[reason_id] = "not_applicable"
        elif reason_id in producer:
            family[reason_id] = "producer"
            partial[reason_id] = not producer_population_complete
            allowed.update((generation, 1, thread_id, reason_id) for thread_id in ready_thread_indices)
        elif reason_id in registration:
            family[reason_id] = "registration"
            partial[reason_id] = not registration_available
            if registration_available:
                allowed.add((generation, 2, 1, reason_id))
        elif reason_id in writer:
            family[reason_id] = "writer"
            partial[reason_id] = not writer_available
            if writer_available:
                allowed.add((generation, 3, 1, reason_id))
        elif reason_id in aggregate:
            family[reason_id] = "aggregate"
            partial[reason_id] = not aggregate_available
            if aggregate_available:
                allowed.add((generation, 4, 1, reason_id))
        else:
            _fail("diagnostic_definition", f"reason {reason_id} has no mode scope")
    return allowed, partial, family


def _validate_counter_rows(
    entries: Any,
    *,
    generation: int,
    allowed: set[tuple[int, int, int, int]],
    partial_by_reason: dict[int, bool],
    family_by_reason: Mapping[int, str],
    ready_thread_indices: tuple[int, ...],
    producer_population_complete: bool,
    unavailable_counter_keys: set[tuple[int, int, int, int]],
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    if not isinstance(entries, list) or not entries:
        _fail("diagnostics.entries", "nonempty counter-row array required")
    rows: list[dict[str, Any]] = []
    by_reason: dict[int, list[dict[str, Any]]] = {reason_id: [] for reason_id in REASON_IDS}
    previous: tuple[int, int, int, int] | None = None
    observed: set[tuple[int, int, int, int]] = set()
    for index, value in enumerate(entries):
        where = f"diagnostics.entries[{index}]"
        row = dict(
            _exact_keys(
                value,
                (
                    "counter_scope_id",
                    "counter_scope_kind",
                    "process_generation",
                    "reason_id",
                    "saturated",
                    "value",
                ),
                where,
            )
        )
        key = (
            _uint(row["process_generation"], 64, f"{where}.process_generation", nonzero=True),
            _uint(row["counter_scope_kind"], 16, f"{where}.counter_scope_kind", nonzero=True),
            _uint(row["counter_scope_id"], 32, f"{where}.counter_scope_id", nonzero=True),
            _uint(row["reason_id"], 16, f"{where}.reason_id", nonzero=True),
        )
        if key[0] != generation:
            _fail(f"{where}.process_generation", "top-level generation mismatch")
        if previous is not None and key <= previous:
            _fail("diagnostics.entries", "strict counter-key order and uniqueness required")
        previous = key
        if key not in allowed:
            _fail(where, "counter scope/reason is not in the admitted mode population")
        observed.add(key)
        _boolean(row["saturated"], f"{where}.saturated")
        _uint(row["value"], 64, f"{where}.value")
        if key[3] == 96 and row["saturated"]:
            _fail(f"{where}.saturated", "ID 96 self-saturation is forbidden in schema v1")
        rows.append(row)
        by_reason[key[3]].append(row)

    unknown_unavailable = unavailable_counter_keys - allowed
    if unknown_unavailable:
        _fail(
            "population.unavailable_counter_keys",
            f"key is not in the admitted population: {min(unknown_unavailable)!r}",
        )
    overlap = observed & unavailable_counter_keys
    if overlap:
        _fail(
            "population.unavailable_counter_keys",
            f"declared-unavailable counter is present: {min(overlap)!r}",
        )
    for key in unavailable_counter_keys:
        partial_by_reason[key[3]] = True
    missing = allowed - observed - unavailable_counter_keys
    for key in missing:
        _fail("diagnostics.entries", f"required counter row missing for key {key!r}")
    if producer_population_complete:
        expected_threads = set(ready_thread_indices)
        for reason_id, family in family_by_reason.items():
            if family != "producer":
                continue
            observed_threads = {
                row["counter_scope_id"] for row in by_reason[reason_id]
            }
            observed_threads.update(
                key[2]
                for key in unavailable_counter_keys
                if key[1] == 1 and key[3] == reason_id
            )
            if observed_threads != expected_threads:
                _fail(
                    "diagnostics.entries",
                    f"complete producer population mismatch for reason {reason_id}",
                )
    return rows, by_reason


def _unavailable_keys(values: Sequence[Sequence[int]]) -> set[tuple[int, int, int, int]]:
    if isinstance(values, (str, bytes, bytearray)):
        _fail("population.unavailable_counter_keys", "counter-key sequence required")
    result: set[tuple[int, int, int, int]] = set()
    previous: tuple[int, int, int, int] | None = None
    for index, value in enumerate(values):
        where = f"population.unavailable_counter_keys[{index}]"
        if not isinstance(value, (tuple, list)) or len(value) != 4:
            _fail(where, "exact four-integer key required")
        key = (
            _uint(value[0], 64, f"{where}[0]", nonzero=True),
            _uint(value[1], 16, f"{where}[1]", nonzero=True),
            _uint(value[2], 32, f"{where}[2]", nonzero=True),
            _uint(value[3], 16, f"{where}[3]", nonzero=True),
        )
        if previous is not None and key <= previous:
            _fail("population.unavailable_counter_keys", "strict increasing uniqueness required")
        previous = key
        result.add(key)
    return result


def _project_reason_totals(
    by_reason: Mapping[int, Sequence[Mapping[str, Any]]],
    partial_by_reason: Mapping[int, bool],
) -> list[dict[str, int]]:
    result: list[dict[str, int]] = []
    for reason_id in REASON_IDS:
        rows = by_reason[reason_id]
        mathematical_sum = sum(row["value"] for row in rows)
        saturated_instances = sum(1 for row in rows if row["saturated"])
        nonzero_instances = sum(1 for row in rows if row["value"] > 0)
        summary_flags = 0
        if saturated_instances or mathematical_sum > UINT64_MAX:
            summary_flags |= TOTAL_SATURATED
        if partial_by_reason[reason_id]:
            summary_flags |= PARTIAL_COUNTER_POPULATION
        result.append(
            {
                "class_flags": REASON_CLASS_FLAGS[reason_id],
                "nonzero_counter_instances": nonzero_instances,
                "reason_id": reason_id,
                "saturated_counter_instances": saturated_instances,
                "saturating_total": min(mathematical_sum, UINT64_MAX),
                "summary_flags": summary_flags,
            }
        )
    return result


def _validate_reason_totals(value: Any, expected: Sequence[Mapping[str, int]]) -> None:
    if not isinstance(value, list) or len(value) != len(REASON_IDS):
        _fail("diagnostics.reason_totals", "exact 14-row array required")
    fields = (
        "class_flags",
        "nonzero_counter_instances",
        "reason_id",
        "saturated_counter_instances",
        "saturating_total",
        "summary_flags",
    )
    for index, (actual, projected) in enumerate(zip(value, expected)):
        where = f"diagnostics.reason_totals[{index}]"
        row = _exact_keys(actual, fields, where)
        _uint(row["class_flags"], 16, f"{where}.class_flags")
        _uint(row["nonzero_counter_instances"], 32, f"{where}.nonzero_counter_instances")
        _uint(row["reason_id"], 16, f"{where}.reason_id", nonzero=True)
        saturated = _uint(
            row["saturated_counter_instances"],
            32,
            f"{where}.saturated_counter_instances",
        )
        nonzero = row["nonzero_counter_instances"]
        if saturated > nonzero:
            _fail(where, "saturated instances exceed nonzero instances")
        _uint(row["saturating_total"], 64, f"{where}.saturating_total")
        flags = _uint(row["summary_flags"], 32, f"{where}.summary_flags")
        if flags & ~3:
            _fail(f"{where}.summary_flags", "reserved flag set")
        if dict(row) != dict(projected):
            _fail(where, "row does not equal authoritative counter projection")


def _projection(reason_totals: Sequence[Mapping[str, int]]) -> dict[str, Any]:
    loss_math = sum(row["saturating_total"] for row in reason_totals if row["class_flags"] & 1)
    bypass_math = sum(row["saturating_total"] for row in reason_totals if row["class_flags"] & 2)
    saturated_instances = sum(row["saturated_counter_instances"] for row in reason_totals)
    if saturated_instances > UINT64_MAX:
        _fail("projection.saturated_counter_instances", "u64 aggregate overflow")
    aggregate_saturated = bool(
        any(row["summary_flags"] & TOTAL_SATURATED for row in reason_totals)
        or loss_math > UINT64_MAX
        or bypass_math > UINT64_MAX
    )
    return {
        "aggregate_saturated": aggregate_saturated,
        "diagnostic_bypass_sum": min(bypass_math, UINT64_MAX),
        "diagnostic_loss_sum": min(loss_math, UINT64_MAX),
        "population_partial": any(
            row["summary_flags"] & PARTIAL_COUNTER_POPULATION for row in reason_totals
        ),
        "reason_totals": [dict(row) for row in reason_totals],
        "saturated_counter_instances": saturated_instances,
    }


def validate_diagnostics(
    value: Mapping[str, Any],
    *,
    definition_sha256: str,
    expected_mode_id: int,
    expected_process_generation: int,
    ready_thread_indices: Sequence[int],
    producer_population_complete: bool,
    registration_available: bool = True,
    writer_available: bool = True,
    aggregate_available: bool = True,
    unavailable_counter_keys: Sequence[Sequence[int]] = (),
) -> dict[str, Any]:
    """Validate one parsed diagnostics object and return its exact projection."""

    top = _exact_keys(
        value,
        ("catalog_id", "entries", "mode_id", "process_generation", "reason_totals", "schema", "version"),
        "diagnostics",
    )
    if top["catalog_id"] != "oai_memprof_diagnostics":
        _fail("diagnostics.catalog_id", "exact ID required")
    _version(top["version"], "diagnostics.version")
    _schema_binding(top["schema"], definition_sha256)
    mode_id = _uint(top["mode_id"], 16, "diagnostics.mode_id", nonzero=True)
    generation = _uint(
        top["process_generation"], 64, "diagnostics.process_generation", nonzero=True
    )
    if mode_id != _uint(expected_mode_id, 16, "expected_mode_id", nonzero=True):
        _fail("diagnostics.mode_id", "expected mode mismatch")
    if generation != _uint(
        expected_process_generation, 64, "expected_process_generation", nonzero=True
    ):
        _fail("diagnostics.process_generation", "expected generation mismatch")
    threads = _ready_threads(ready_thread_indices)
    for flag, where in (
        (producer_population_complete, "producer_population_complete"),
        (registration_available, "registration_available"),
        (writer_available, "writer_available"),
        (aggregate_available, "aggregate_available"),
    ):
        _boolean(flag, where)
    allowed, partial, family = _scope_population(
        mode_id=mode_id,
        generation=generation,
        ready_thread_indices=threads,
        producer_population_complete=producer_population_complete,
        registration_available=registration_available,
        writer_available=writer_available,
        aggregate_available=aggregate_available,
    )
    unavailable = _unavailable_keys(unavailable_counter_keys)
    rows, by_reason = _validate_counter_rows(
        top["entries"],
        generation=generation,
        allowed=allowed,
        partial_by_reason=partial,
        family_by_reason=family,
        ready_thread_indices=threads,
        producer_population_complete=producer_population_complete,
        unavailable_counter_keys=unavailable,
    )
    totals = _project_reason_totals(by_reason, partial)
    _validate_reason_totals(top["reason_totals"], totals)
    result = _projection(totals)
    id96 = by_reason[96]
    if id96:
        producer_reason_count = sum(1 for kind in family.values() if kind == "producer")
        maximum_transitions = producer_reason_count * UINT32_MAX + 3
        if id96[0]["value"] > maximum_transitions:
            _fail(
                "diagnostics.entries",
                "ID 96 exceeds the finite schema-v1 counter-instance population bound",
            )
    if not result["population_partial"]:
        if len(id96) != 1:
            _fail("diagnostics.entries", "complete ID 96 singleton required")
        if id96[0]["value"] != result["saturated_counter_instances"]:
            _fail("diagnostics.entries", "complete ID 96 saturation-transition reconciliation failed")
    result.update(
        {
            "entry_count": len(rows),
            "mode_id": mode_id,
            "process_generation": generation,
        }
    )
    return result


def validate_diagnostics_bytes(raw: bytes, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require canonical bytes, validate semantics, and return value/projection."""

    try:
        value = SEMANTIC.parse_canonical(raw)
    except Exception as error:
        raise DiagnosticError(str(error)) from error
    return value, validate_diagnostics(value, **kwargs)


def reconcile_terminal(
    projection: Mapping[str, Any],
    *,
    terminal_entries: Sequence[Any],
    terminal_flags: int,
    diagnostic_loss_sum: int,
    diagnostic_bypass_sum: int,
    saturated_counter_instances: int,
) -> None:
    """Require exact object-to-terminal diagnostic table/header projection."""

    if len(terminal_entries) != len(REASON_IDS):
        _fail("terminal.diagnostic_entries", "exact 14-row table required")
    fields = (
        "reason_id",
        "class_flags",
        "summary_flags",
        "saturating_total",
        "nonzero_counter_instances",
        "saturated_counter_instances",
    )
    widths = {
        "reason_id": 16,
        "class_flags": 16,
        "summary_flags": 32,
        "saturating_total": 64,
        "nonzero_counter_instances": 32,
        "saturated_counter_instances": 32,
    }
    for index, (entry, expected) in enumerate(zip(terminal_entries, projection["reason_totals"])):
        if isinstance(entry, Mapping):
            if len(entry) != len(fields) or set(entry) != set(fields):
                _fail(
                    f"terminal.diagnostic_entries[{index}]",
                    "exact diagnostic-entry fields required",
                )
            actual = {field: entry[field] for field in fields}
        else:
            try:
                actual = {field: getattr(entry, field) for field in fields}
            except (AttributeError, TypeError) as error:
                raise DiagnosticError(
                    f"terminal.diagnostic_entries[{index}]: compatible entry required"
                ) from error
        normalized = {
            field: _uint(actual[field], widths[field], f"terminal.diagnostic_entries[{index}].{field}")
            for field in fields
        }
        if normalized != {field: expected[field] for field in fields}:
            _fail(f"terminal.diagnostic_entries[{index}]", "diagnostics projection mismatch")
    if _uint(diagnostic_loss_sum, 64, "terminal.diagnostic_loss_sum") != projection[
        "diagnostic_loss_sum"
    ]:
        _fail("terminal.diagnostic_loss_sum", "projection mismatch")
    if _uint(diagnostic_bypass_sum, 64, "terminal.diagnostic_bypass_sum") != projection[
        "diagnostic_bypass_sum"
    ]:
        _fail("terminal.diagnostic_bypass_sum", "projection mismatch")
    if _uint(
        saturated_counter_instances, 64, "terminal.saturated_counter_instances"
    ) != projection["saturated_counter_instances"]:
        _fail("terminal.saturated_counter_instances", "projection mismatch")
    flags = _uint(terminal_flags, 64, "terminal.terminal_flags")
    if bool(flags & TERMINAL_AGGREGATE_SATURATED) != projection["aggregate_saturated"]:
        _fail("terminal.terminal_flags", "aggregate-saturated bit mismatch")
    if bool(flags & TERMINAL_POPULATION_PARTIAL) != projection["population_partial"]:
        _fail("terminal.terminal_flags", "partial-population bit mismatch")


def make_diagnostics_bytes(
    *,
    definition_sha256: str,
    mode_id: int,
    process_generation: int,
    counter_rows: Sequence[Mapping[str, Any]],
    ready_thread_indices: Sequence[int],
    producer_population_complete: bool,
    registration_available: bool = True,
    writer_available: bool = True,
    aggregate_available: bool = True,
    unavailable_counter_keys: Sequence[Sequence[int]] = (),
) -> bytes:
    """Build canonical fixture/producer bytes from explicit counter evidence."""

    generation = _uint(process_generation, 64, "process_generation", nonzero=True)
    threads = _ready_threads(ready_thread_indices)
    allowed, partial, family = _scope_population(
        mode_id=mode_id,
        generation=generation,
        ready_thread_indices=threads,
        producer_population_complete=producer_population_complete,
        registration_available=registration_available,
        writer_available=writer_available,
        aggregate_available=aggregate_available,
    )
    unavailable = _unavailable_keys(unavailable_counter_keys)
    rows, by_reason = _validate_counter_rows(
        list(counter_rows),
        generation=generation,
        allowed=allowed,
        partial_by_reason=partial,
        family_by_reason=family,
        ready_thread_indices=threads,
        producer_population_complete=producer_population_complete,
        unavailable_counter_keys=unavailable,
    )
    totals = _project_reason_totals(by_reason, partial)
    value = {
        "catalog_id": "oai_memprof_diagnostics",
        "entries": rows,
        "mode_id": mode_id,
        "process_generation": generation,
        "reason_totals": totals,
        "schema": {
            "object_type": 8,
            "path": DEFINITION_PATH,
            "sha256": _hash(definition_sha256, "definition_sha256"),
        },
        "version": VERSION,
    }
    raw = SEMANTIC.canonical_bytes(value)
    validate_diagnostics_bytes(
        raw,
        definition_sha256=definition_sha256,
        expected_mode_id=mode_id,
        expected_process_generation=generation,
        ready_thread_indices=threads,
        producer_population_complete=producer_population_complete,
        registration_available=registration_available,
        writer_available=writer_available,
        aggregate_available=aggregate_available,
        unavailable_counter_keys=unavailable_counter_keys,
    )
    return raw


__all__ = [
    "DEFINITION_PATH",
    "DiagnosticError",
    "INSTANCE_PATH",
    "PARTIAL_COUNTER_POPULATION",
    "REASON_IDS",
    "TERMINAL_AGGREGATE_SATURATED",
    "TERMINAL_POPULATION_PARTIAL",
    "TOTAL_SATURATED",
    "make_diagnostics_bytes",
    "reconcile_terminal",
    "sha256_hex",
    "validate_diagnostics",
    "validate_diagnostics_bytes",
]
