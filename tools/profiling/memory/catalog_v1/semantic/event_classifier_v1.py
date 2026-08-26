#!/usr/bin/env python3
"""Executable conformance and A03 replay for frozen event-semantics v1.1.

This standard-library-only module derives record meaning from decoded evidence.
It accepts no transition ID, operand profile, selection case, or flag mask from
the caller.  The composed archive verifier supplies a realloc-zero policy only
after resolving it from validated measured build/run coverage.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence


UINT16_MAX = (1 << 16) - 1
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1


class EventClassifierError(ValueError):
    """Deterministic record-classification or replay rejection."""


class EventClassification(NamedTuple):
    transition_id: int
    transition_name: str
    operand_profile: str
    selection_case: str
    required_one_mask: int
    required_zero_mask: int


class A03ReplayResult(NamedTuple):
    selected_successors: int
    ended_predecessors: int
    live_selected_successors: int


class _SelectedInstance(NamedTuple):
    address: int
    requested_bytes: int | None
    birth_counter: int


class _CompiledEventSemantics(NamedTuple):
    transitions: Mapping[int, Mapping[str, Any]]
    operand_profiles: Mapping[str, Mapping[str, Any]]
    selection_cases: Mapping[int, Mapping[str, Any]]
    mode_rules: Mapping[int, Mapping[str, Any]]
    allowed_selection_cases: Mapping[int, Mapping[int, frozenset[str]]]


def _load_semantic() -> Any:
    path = Path(__file__).resolve().with_name("semantic_catalog_v1.py")
    spec = importlib.util.spec_from_file_location(
        "_oai_memprof_semantic_for_event_classifier", path
    )
    if spec is None or spec.loader is None:
        raise EventClassifierError("semantic: adjacent validator unavailable")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise EventClassifierError(
            "semantic: adjacent validator failed to load"
        ) from error
    return module


SEMANTIC = _load_semantic()


def _fail(where: str, detail: str) -> None:
    raise EventClassifierError(f"{where}: {detail}")


def _uint(value: Any, bits: int, where: str, *, nonzero: bool = False) -> int:
    if type(value) is not int:
        _fail(where, f"u{bits} integer required")
    minimum = 1 if nonzero else 0
    if not minimum <= value <= (1 << bits) - 1:
        _fail(where, f"outside u{bits} range")
    return value


def _boolean(value: Any, where: str) -> bool:
    if type(value) is not bool:
        _fail(where, "boolean required")
    return value


def _field(record: Any, name: str, where: str) -> Any:
    if isinstance(record, Mapping):
        if name not in record:
            _fail(where, f"missing field {name}")
        return record[name]
    try:
        return getattr(record, name)
    except AttributeError:
        _fail(where, f"missing field {name}")


def _validated_definition(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("event_semantics", "object required")
    try:
        SEMANTIC.validate_event_semantics(value)
    except Exception as error:
        _fail("event_semantics", str(error))
    return value


def _compiled_definition(value: Any) -> _CompiledEventSemantics:
    definition = _validated_definition(value)
    transitions = {
        row["transition_id"]: row for row in definition["transitions"]
    }
    operand_profiles = {
        row["name"]: row for row in definition["operand_profiles"]
    }
    selection_cases = {
        row["required_one_mask"]: row for row in definition["selection_cases"]
    }
    mode_rules = {row["mode_id"]: row for row in definition["mode_rules"]}
    allowed_selection_cases: dict[int, dict[int, frozenset[str]]] = {}
    for mode_id, mode_rule in mode_rules.items():
        allowed: dict[int, set[str]] = {}
        for mapping in mode_rule["selection_cases"]:
            cases = mapping["selection_cases"]
            for transition_id in mapping["transition_ids"]:
                allowed.setdefault(transition_id, set()).update(cases)
        allowed_selection_cases[mode_id] = {
            transition_id: frozenset(cases)
            for transition_id, cases in allowed.items()
        }
    return _CompiledEventSemantics(
        transitions,
        operand_profiles,
        selection_cases,
        mode_rules,
        allowed_selection_cases,
    )


def _selection_case(
    definition: _CompiledEventSemantics, flags: int
) -> Mapping[str, Any]:
    bits = flags & SEMANTIC.SELECTION_MASK
    selection = definition.selection_cases.get(bits)
    if selection is None:
        _fail("event.selection", "impossible frozen bit combination")
    return selection


def _transition_id(
    *,
    api_id: int,
    address_before: int,
    address_after: int,
    arg0: int,
    arg1: int,
    arg2: int,
    flags: int,
    result_code: Any,
    realloc_zero_policy_id: Any,
) -> int:
    if api_id not in (3, 5) and realloc_zero_policy_id is not None:
        _fail(
            "realloc_zero_policy_id",
            "only realloc-family APIs consume a policy",
        )

    if api_id == 1:
        return {
            (True, True): 100,
            (False, True): 101,
            (True, False): 102,
            (False, False): 103,
        }[(address_after == 0, arg0 == 0)]

    if api_id == 2:
        product = arg0 * arg1
        if product > UINT64_MAX:
            if arg2 != 0 or address_after != 0:
                _fail(
                    "event.calloc",
                    "overflow requires canonical arg2 zero and NULL result",
                )
            return 200
        if arg2 != product:
            _fail("event.calloc", "arg2 differs from checked u64 product")
        return {
            (True, True): 201,
            (False, True): 202,
            (True, False): 203,
            (False, False): 204,
        }[(address_after == 0, product == 0)]

    if api_id in (3, 5):
        policy = _uint(
            realloc_zero_policy_id,
            16,
            "realloc_zero_policy_id",
            nonzero=True,
        )
        if policy not in (1, 2):
            _fail("realloc_zero_policy_id", "admitted policy 1 or 2 required")
        before_null = address_before == 0
        after_null = address_after == 0
        if api_id == 3:
            zero_request = arg0 == 0
            if before_null:
                return {
                    (True, True): 300,
                    (False, True): 301,
                    (True, False): 302,
                    (False, False): 303,
                }[(after_null, zero_request)]
            if zero_request and after_null:
                return {1: 304, 2: 305}[policy]
            if zero_request:
                return 306
            if after_null:
                return 307
            return 308

        if flags & (1 << 14):
            if before_null:
                _fail(
                    "event.reallocarray",
                    "matched predecessor requires a non-NULL old pointer",
                )
            if flags & (1 << 19):
                _fail(
                    "event.reallocarray",
                    "overflow cannot emit a matched sampled endpoint",
                )
            zero_request = arg0 == 0
            if zero_request and after_null:
                return {1: 504, 2: 505}[policy]
            if zero_request:
                return 506
            if after_null:
                return 509
            return 510

        product = arg1 * arg2
        if product > UINT64_MAX:
            if arg0 != 0 or address_after != 0:
                _fail(
                    "event.reallocarray",
                    "overflow requires canonical arg0 zero and NULL result",
                )
            return 500 if before_null else 501
        if arg0 != product:
            _fail(
                "event.reallocarray",
                "arg0 differs from checked u64 product",
            )
        if before_null:
            return {
                (True, True): 502,
                (False, True): 503,
                (True, False): 507,
                (False, False): 508,
            }[(after_null, product == 0)]
        if product == 0 and after_null:
            return {1: 504, 2: 505}[policy]
        if product == 0:
            return 506
        if after_null:
            return 509
        return 510

    if api_id == 4:
        if address_before == 0:
            return 400
        if flags & (1 << 14) and not flags & (1 << 2):
            return 402
        return 401

    def pointer_allocation(base: int, size: int) -> int:
        return base + {
            (True, True): 0,
            (False, True): 1,
            (True, False): 2,
            (False, False): 3,
        }[(address_after == 0, size == 0)]

    if api_id == 6:
        return pointer_allocation(600, arg1)
    if api_id == 7:
        if type(result_code) is not int or not 0 <= result_code <= (1 << 31) - 1:
            _fail("event.posix_memalign", "nonnegative direct return required")
        if result_code != 0:
            if address_after != 0:
                _fail(
                    "event.posix_memalign",
                    "failure leaves output pointer unavailable",
                )
            return 700 if arg1 == 0 else 701
        if arg1 == 0:
            return 702 if address_after == 0 else 703
        if address_after == 0:
            _fail(
                "event.posix_memalign",
                "positive-size success requires a non-NULL result",
            )
        return 704
    if api_id == 8:
        return pointer_allocation(800, arg1)
    if api_id == 9:
        return pointer_allocation(900, arg0)
    if api_id == 10:
        return pointer_allocation(1000, arg0)
    if api_id == 11:
        return 1100 if address_after == 0 else 1101
    if api_id == 12:
        return 1200 if address_after == 0 else 1201
    _fail("event.api_id", "unknown first-slice API")


def _classification_context(
    *,
    mode_id: Any,
    cutoff_before_counter: Any,
    cutoff_after_counter: Any,
    seal_bracket_available: Any,
    accepted_modes: tuple[int, ...] = (3, 4, 5),
) -> tuple[int, int, int, bool]:
    mode = _uint(mode_id, 16, "mode_id")
    if mode not in accepted_modes:
        _fail("mode_id", "selected mode emits no first-slice records")
    cutoff_before = _uint(cutoff_before_counter, 64, "cutoff_before_counter")
    cutoff_after = _uint(cutoff_after_counter, 64, "cutoff_after_counter")
    bracket_available = _boolean(
        seal_bracket_available, "seal_bracket_available"
    )
    if cutoff_after < cutoff_before:
        _fail("event.boundary", "cutoff bracket is reversed")
    return mode, cutoff_before, cutoff_after, bracket_available


def _classify_record(
    definition: _CompiledEventSemantics,
    record: Any,
    *,
    mode: int,
    realloc_zero_policy_id: Any,
    cutoff_before: int,
    cutoff_after: int,
    bracket_available: bool,
) -> EventClassification:
    where = "event"
    flags = _uint(_field(record, "flags", where), 32, "event.flags")
    api_id = _uint(_field(record, "api_id", where), 16, "event.api_id", nonzero=True)
    event_kind = _uint(
        _field(record, "event_kind", where), 16, "event.event_kind", nonzero=True
    )
    if {
        1: 1, 2: 1, 3: 2, 4: 3, 5: 2, 6: 1,
        7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1,
    }.get(api_id) != event_kind:
        _fail("event.api_id/event_kind", "invalid frozen pair")
    thread_index = _uint(
        _field(record, "thread_index", where), 32, "event.thread_index", nonzero=True
    )
    if thread_index == UINT32_MAX:
        _fail("event.thread_index", "0xffffffff is reserved")
    _uint(
        _field(record, "thread_sequence", where),
        64,
        "event.thread_sequence",
        nonzero=True,
    )
    address_before = _uint(
        _field(record, "address_before", where), 64, "event.address_before"
    )
    address_after = _uint(
        _field(record, "address_after", where), 64, "event.address_after"
    )
    arg0 = _uint(_field(record, "arg0", where), 64, "event.arg0")
    arg1 = _uint(_field(record, "arg1", where), 64, "event.arg1")
    arg2 = _uint(_field(record, "arg2", where), 64, "event.arg2")
    result_code = _field(record, "result_code", where)
    callsite_id = _uint(
        _field(record, "callsite_id", where), 32, "event.callsite_id"
    )
    _uint(_field(record, "context_id", where), 32, "event.context_id")
    counter_enter = _uint(
        _field(record, "counter_enter", where), 64, "event.counter_enter"
    )
    counter_exit = _uint(
        _field(record, "counter_exit", where), 64, "event.counter_exit"
    )
    transition_id = _transition_id(
        api_id=api_id,
        address_before=address_before,
        address_after=address_after,
        arg0=arg0,
        arg1=arg1,
        arg2=arg2,
        flags=flags,
        result_code=result_code,
        realloc_zero_policy_id=realloc_zero_policy_id,
    )
    transition = definition.transitions[transition_id]
    selection = _selection_case(definition, flags)
    mode_row = definition.mode_rules[mode]
    allowed_cases = definition.allowed_selection_cases[mode].get(
        transition_id, frozenset()
    )
    if transition_id not in mode_row["allowed_transition_ids"]:
        _fail("event.transition", "transition is not emitted in selected mode")
    if selection["name"] not in allowed_cases:
        _fail("event.selection", "case is not emitted for transition/mode")
    if mode_row["callsite_rule"] == "canonical_zero":
        if callsite_id != 0:
            _fail("event.callsite_id", "selected mode requires canonical zero")
    elif callsite_id == 0:
        _fail("event.callsite_id", "A05 requires nonzero resolved ID")

    profile_name = (
        transition["matched_operand_profile"]
        if selection["arg_match_valid"]
        else transition["unmatched_operand_profile"]
    )
    if profile_name is None:
        _fail("event.operand_profile", "selected case has no profile")
    profile = definition.operand_profiles[profile_name]
    if selection["arg_match_valid"] and (
        arg1 == 0 or arg1 >= UINT32_MAX or arg2 == 0
    ):
        _fail("event.predecessor_match", "exact nonzero origin key required")
    if selection["arg_match_valid"] and bool(flags & (1 << 17)) != (
        arg1 != thread_index
    ):
        _fail("event.predecessor_match", "cross-thread bit differs from origin")

    required_one = (
        transition["required_one_mask"]
        | profile["required_one_mask"]
        | selection["required_one_mask"]
    )
    required_zero = (
        transition["required_zero_mask"]
        | profile["required_zero_mask"]
        | selection["required_zero_mask"]
    )
    if (
        required_one & required_zero
        or (required_one | required_zero | SEMANTIC.COMMON_EVIDENCE_MASK)
        != SEMANTIC.KNOWN_FLAGS_MASK
    ):
        _fail("event.flags", "frozen masks are not complementary")
    try:
        SEMANTIC.validate_event_flags(
            flags=flags,
            required_one_mask=required_one,
            required_zero_mask=required_zero,
            address_before=address_before,
            address_after=address_after,
            arg0=arg0,
            arg1=arg1,
            arg2=arg2,
            counter_enter=counter_enter,
            counter_exit=counter_exit,
            cpu_enter=_field(record, "cpu_enter", where),
            cpu_exit=_field(record, "cpu_exit", where),
            result_code=result_code,
        )
    except Exception as error:
        _fail("event.flags", str(error))

    has_enter = bool(flags & (1 << 5))
    has_exit = bool(flags & (1 << 6))
    boundary = bool(flags & (1 << 18))
    if boundary and not bracket_available:
        _fail("event.boundary_straddling", "seal bracket is unavailable")
    if has_enter and has_exit and counter_exit < counter_enter:
        _fail("event.counter", "exit precedes entry")
    if bracket_available and has_exit:
        if boundary and counter_exit < cutoff_before:
            _fail("event.boundary_straddling", "post-seal flag has pre-seal exit")
        if not boundary and counter_exit > cutoff_after:
            _fail("event.boundary_straddling", "pre-seal flag has post-seal exit")

    return EventClassification(
        transition_id,
        transition["name"],
        profile_name,
        selection["name"],
        required_one,
        required_zero,
    )


def classify_record(
    event_semantics: Mapping[str, Any],
    record: Any,
    *,
    mode_id: Any,
    realloc_zero_policy_id: Any,
    cutoff_before_counter: Any,
    cutoff_after_counter: Any,
    seal_bracket_available: Any,
) -> EventClassification:
    """Derive and validate exactly one frozen record variant."""

    definition = _compiled_definition(event_semantics)
    mode, cutoff_before, cutoff_after, bracket_available = (
        _classification_context(
            mode_id=mode_id,
            cutoff_before_counter=cutoff_before_counter,
            cutoff_after_counter=cutoff_after_counter,
            seal_bracket_available=seal_bracket_available,
        )
    )
    return _classify_record(
        definition,
        record,
        mode=mode,
        realloc_zero_policy_id=realloc_zero_policy_id,
        cutoff_before=cutoff_before,
        cutoff_after=cutoff_after,
        bracket_available=bracket_available,
    )


def classify_records(
    event_semantics: Mapping[str, Any],
    records: Sequence[Any],
    *,
    mode_id: Any,
    realloc_zero_policy_id: Any,
    cutoff_before_counter: Any,
    cutoff_after_counter: Any,
    seal_bracket_available: Any,
) -> tuple[EventClassification, ...]:
    """Classify a stream population and enforce per-thread sequence rules."""

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
        _fail("records", "sequence required")
    definition = _compiled_definition(event_semantics)
    mode, cutoff_before, cutoff_after, bracket_available = (
        _classification_context(
            mode_id=mode_id,
            cutoff_before_counter=cutoff_before_counter,
            cutoff_after_counter=cutoff_after_counter,
            seal_bracket_available=seal_bracket_available,
            accepted_modes=(0, 1, 2, 3, 4, 5, 6),
        )
    )
    if mode in (0, 1, 2, 6):
        if records:
            _fail("records", "no-emission mode contains a first-slice record")
        return ()
    classified = tuple(
        _classify_record(
            definition,
            record,
            mode=mode,
            realloc_zero_policy_id=(
                realloc_zero_policy_id
                if _field(record, "api_id", f"records[{index}]") in (3, 5)
                else None
            ),
            cutoff_before=cutoff_before,
            cutoff_after=cutoff_after,
            bracket_available=bracket_available,
        )
        for index, record in enumerate(records)
    )
    by_thread: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        thread = _uint(
            _field(record, "thread_index", f"records[{index}]"),
            32,
            f"records[{index}].thread_index",
            nonzero=True,
        )
        sequence = _uint(
            _field(record, "thread_sequence", f"records[{index}]"),
            64,
            f"records[{index}].thread_sequence",
            nonzero=True,
        )
        by_thread.setdefault(thread, []).append(sequence)
    for thread, values in by_thread.items():
        if len(values) != len(set(values)):
            _fail("records.thread_sequence", f"thread {thread}: duplicate sequence")
        if values != sorted(values):
            _fail("records.thread_sequence", f"thread {thread}: physical order regresses")
    return classified


def exact_mode_sequence_gap_count(
    records: Sequence[Any], *, mode_id: Any
) -> int:
    """Return exact-mode gaps and validate canonical no-emission populations."""

    mode = _uint(mode_id, 16, "mode_id")
    if mode not in (0, 1, 2, 3, 4, 5, 6):
        _fail("mode_id", "unknown mode")
    if not isinstance(records, Sequence) or isinstance(
        records, (str, bytes, bytearray)
    ):
        _fail("records", "sequence required")
    if mode in (0, 1, 2, 6):
        if records:
            _fail("records", "no-emission mode contains a first-slice record")
        return 0
    if mode == 3:
        return 0
    by_thread: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        where = f"records[{index}]"
        thread = _uint(
            _field(record, "thread_index", where),
            32,
            f"{where}.thread_index",
            nonzero=True,
        )
        sequence = _uint(
            _field(record, "thread_sequence", where),
            64,
            f"{where}.thread_sequence",
            nonzero=True,
        )
        by_thread.setdefault(thread, []).append(sequence)
    gaps = 0
    for thread, values in by_thread.items():
        if len(values) != len(set(values)) or values != sorted(values):
            _fail("records.thread_sequence", f"thread {thread}: invalid order")
        gaps += values[-1] - len(values)
    return gaps


def _requested_bytes_for_successor(
    record: Any, api_id: int, flags: int, where: str
) -> int | None:
    source = {
        1: ("arg0", 2),
        2: ("arg2", 4),
        3: ("arg0", 2),
        5: ("arg0", 2),
        6: ("arg1", 3),
        7: ("arg1", 3),
        8: ("arg1", 3),
        9: ("arg0", 2),
        10: ("arg0", 2),
        11: None,
        12: None,
    }.get(api_id)
    if source is None:
        return None
    field, bit = source
    if not flags & (1 << bit):
        return None
    return _uint(_field(record, field, where), 64, f"{where}.requested_bytes")


def reconcile_a03_predecessors(
    records: Sequence[Any],
    classifications: Sequence[EventClassification],
    *,
    process_generation: Any,
) -> A03ReplayResult:
    """Replay persisted selected A03 identities and exact predecessor links."""

    generation = _uint(
        process_generation, 64, "process_generation", nonzero=True
    )
    if len(records) != len(classifications):
        _fail("records", "classification cardinality mismatch")
    selected: dict[tuple[int, int, int], _SelectedInstance] = {}
    for index, (record, classification) in enumerate(zip(records, classifications)):
        flags = _uint(_field(record, "flags", f"records[{index}]"), 32, "flags")
        if type(classification) is not EventClassification:
            _fail(f"classifications[{index}]", "exact EventClassification required")
        if (
            flags & classification.required_one_mask
            != classification.required_one_mask
            or flags & classification.required_zero_mask
        ):
            _fail(f"classifications[{index}]", "classification/flags mismatch")
        if not flags & (1 << 16):
            continue
        key = (
            generation,
            _uint(_field(record, "thread_index", f"records[{index}]"), 32, "thread_index", nonzero=True),
            _uint(_field(record, "thread_sequence", f"records[{index}]"), 64, "thread_sequence", nonzero=True),
        )
        if key in selected:
            _fail(f"records[{index}]", "duplicate selected successor identity")
        address = _uint(_field(record, "address_after", f"records[{index}]"), 64, "address_after")
        if address == 0:
            _fail(f"records[{index}]", "selected successor has NULL address")
        api_id = _uint(
            _field(record, "api_id", f"records[{index}]"),
            16,
            "api_id",
            nonzero=True,
        )
        requested = _requested_bytes_for_successor(
            record, api_id, flags, f"records[{index}]"
        )
        if not flags & (1 << 6):
            _fail(
                f"records[{index}]",
                "selected successor requires counter_exit for lifetime interval",
            )
        birth = _uint(
            _field(record, "counter_exit", f"records[{index}]"),
            64,
            "counter_exit",
        )
        selected[key] = _SelectedInstance(address, requested, birth)

    ended: dict[tuple[int, int, int], int] = {}
    for index, (record, classification) in enumerate(zip(records, classifications)):
        flags = _uint(_field(record, "flags", f"records[{index}]"), 32, "flags")
        if not flags & (1 << 15):
            continue
        origin = (
            generation,
            _uint(_field(record, "arg1", f"records[{index}]"), 32, "arg1", nonzero=True),
            _uint(_field(record, "arg2", f"records[{index}]"), 64, "arg2", nonzero=True),
        )
        predecessor = selected.get(origin)
        if predecessor is None:
            _fail(f"records[{index}]", "selected predecessor origin is unresolved")
        if origin in ended:
            _fail(f"records[{index}]", "selected predecessor ends more than once")
        address_before = _uint(
            _field(record, "address_before", f"records[{index}]"), 64, "address_before"
        )
        if address_before != predecessor.address:
            _fail(f"records[{index}]", "selected predecessor address mismatch")
        api_id = _uint(
            _field(record, "api_id", f"records[{index}]"),
            16,
            "api_id",
            nonzero=True,
        )
        if api_id == 4:
            requested = (
                _uint(_field(record, "arg0", f"records[{index}]"), 64, "arg0")
                if flags & (1 << 2)
                else None
            )
            if requested != predecessor.requested_bytes:
                _fail(
                    f"records[{index}]",
                    "selected free requested-size availability/value mismatch",
                )
            expected_free_transition = (
                402 if predecessor.requested_bytes is None else 401
            )
            if classification.transition_id != expected_free_transition:
                _fail(
                    f"records[{index}]",
                    "selected free transition differs from predecessor size state",
                )
        current_thread = _uint(
            _field(record, "thread_index", f"records[{index}]"),
            32,
            "thread_index",
            nonzero=True,
        )
        current_sequence = _uint(
            _field(record, "thread_sequence", f"records[{index}]"),
            64,
            "thread_sequence",
            nonzero=True,
        )
        current_key = (generation, current_thread, current_sequence)
        if origin[1] == current_thread and origin[2] >= current_sequence:
            _fail(
                f"records[{index}]",
                "same-thread predecessor does not precede transaction",
            )
        if flags & (1 << 16) and current_key == origin:
            _fail(f"records[{index}]", "realloc successor must have a new identity")
        if not flags & (1 << 5):
            _fail(
                f"records[{index}]",
                "selected predecessor end requires counter_enter for lifetime interval",
            )
        death = _uint(
            _field(record, "counter_enter", f"records[{index}]"),
            64,
            "counter_enter",
        )
        if death < predecessor.birth_counter:
            _fail(f"records[{index}]", "predecessor death precedes birth")
        ending_transitions = {
            304: 3,
            306: 3,
            308: 3,
            401: 4,
            402: 4,
            504: 5,
            506: 5,
            510: 5,
        }
        if (
            ending_transitions.get(classification.transition_id) != api_id
            or not flags & (1 << 12)
        ):
            _fail(f"records[{index}]", "selected predecessor is not ended")
        ended[origin] = death

    # A successful selected successor is live from counter_exit.  Its matched
    # predecessor is no longer live from the ending call's counter_enter, so
    # each identity has [birth_counter, death_counter), or an open end while
    # still live.  After sorting by address and birth, accepting each interval
    # proves every earlier same-address interval ended by the prior birth; the
    # immediately preceding interval is therefore the only overlap candidate.
    intervals = sorted(
        (instance.address, instance.birth_counter, key, ended.get(key))
        for key, instance in selected.items()
    )
    previous_address: int | None = None
    previous_death: int | None = None
    for address, birth, _key, death in intervals:
        if address == previous_address and (
            previous_death is None or birth < previous_death
        ):
            _fail("records", "selected identities have intersecting same-address lifetimes")
        previous_address = address
        previous_death = death
    return A03ReplayResult(len(selected), len(ended), len(selected) - len(ended))


__all__ = [
    "A03ReplayResult",
    "EventClassification",
    "EventClassifierError",
    "classify_record",
    "classify_records",
    "exact_mode_sequence_gap_count",
    "reconcile_a03_predecessors",
]
