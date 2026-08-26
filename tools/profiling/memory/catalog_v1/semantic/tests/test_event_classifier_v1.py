#!/usr/bin/env python3
"""Independent deterministic vectors for executable event semantics v1."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "event_classifier_v1", ROOT / "event_classifier_v1.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("event classifier module specification unavailable")
C = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = C
SPEC.loader.exec_module(C)

EVENT_RAW = (ROOT / "archive/definition/event-semantics-v1.json").read_bytes()
EVENT = C.SEMANTIC.parse_canonical(EVENT_RAW)
EVENT_SHA256 = "8dbe428939592cdfc86ba8730078563672de4a13081ebdb868fa97f543dfab89"


@dataclasses.dataclass(frozen=True)
class Record:
    thread_sequence: int
    counter_enter: int
    counter_exit: int
    address_before: int
    address_after: int
    arg0: int
    arg1: int
    arg2: int
    context_id: int
    callsite_id: int
    thread_index: int
    flags: int
    result_code: int
    api_id: int
    event_kind: int
    cpu_enter: int
    cpu_exit: int


def record(**changes: int) -> Record:
    values = {
        "thread_sequence": 1,
        "counter_enter": 0,
        "counter_exit": 0,
        "address_before": 0,
        "address_after": 0,
        "arg0": 0,
        "arg1": 0,
        "arg2": 0,
        "context_id": 0,
        "callsite_id": 0,
        "thread_index": 1,
        "flags": 0,
        "result_code": 0,
        "api_id": 1,
        "event_kind": 1,
        "cpu_enter": 0xFFFF,
        "cpu_exit": 0xFFFF,
    }
    values.update(changes)
    return Record(**values)


EXACT_TRANSITIONS = {
    100: record(flags=0x01000206),
    101: record(address_after=0x1000, flags=0x01000A06),
    102: record(arg0=64, flags=0x01002006),
    103: record(address_after=0x1000, arg0=64, flags=0x01000806),
    200: record(api_id=2, arg0=1 << 63, arg1=2, flags=0x0100240E),
    201: record(api_id=2, arg1=8, flags=0x0100021E),
    202: record(api_id=2, address_after=0x1000, arg1=8, flags=0x01000A1E),
    203: record(api_id=2, arg0=2, arg1=8, arg2=16, flags=0x0100201E),
    204: record(
        api_id=2, address_after=0x1000, arg0=2, arg1=8, arg2=16,
        flags=0x0100081E,
    ),
    300: record(api_id=3, event_kind=2, flags=0x01000207),
    301: record(
        api_id=3, event_kind=2, address_after=0x1000, flags=0x01000A07,
    ),
    302: record(api_id=3, event_kind=2, arg0=64, flags=0x01002007),
    303: record(
        api_id=3, event_kind=2, address_after=0x1000, arg0=64,
        flags=0x01000807,
    ),
    304: record(
        api_id=3, event_kind=2, address_before=0x1000, flags=0x01001207,
    ),
    305: record(
        api_id=3, event_kind=2, address_before=0x1000, flags=0x01002207,
    ),
    306: record(
        api_id=3, event_kind=2, address_before=0x1000,
        address_after=0x2000, flags=0x01001A07,
    ),
    307: record(
        api_id=3, event_kind=2, address_before=0x1000, arg0=64,
        flags=0x01002007,
    ),
    308: record(
        api_id=3, event_kind=2, address_before=0x1000,
        address_after=0x2000, arg0=64, flags=0x01001807,
    ),
    400: record(api_id=4, event_kind=3, flags=0x01000001),
    401: record(
        api_id=4, event_kind=3, address_before=0x1000, flags=0x01001001,
    ),
    500: record(api_id=5, event_kind=2, arg1=1 << 63, arg2=2, flags=0x0108201B),
    501: record(api_id=5, event_kind=2, address_before=0x1000, arg1=1 << 63, arg2=2, flags=0x0108201B),
    502: record(api_id=5, event_kind=2, arg1=0, arg2=8, flags=0x0100021F),
    503: record(api_id=5, event_kind=2, address_after=0x2000, arg1=0, arg2=8, flags=0x01000A1F),
    504: record(api_id=5, event_kind=2, address_before=0x1000, arg1=0, arg2=8, flags=0x0100121F),
    505: record(api_id=5, event_kind=2, address_before=0x1000, arg1=0, arg2=8, flags=0x0100221F),
    506: record(api_id=5, event_kind=2, address_before=0x1000, address_after=0x2000, arg1=0, arg2=8, flags=0x01001A1F),
    507: record(api_id=5, event_kind=2, arg0=16, arg1=2, arg2=8, flags=0x0100201F),
    508: record(api_id=5, event_kind=2, address_after=0x2000, arg0=16, arg1=2, arg2=8, flags=0x0100081F),
    509: record(api_id=5, event_kind=2, address_before=0x1000, arg0=16, arg1=2, arg2=8, flags=0x0100201F),
    510: record(api_id=5, event_kind=2, address_before=0x1000, address_after=0x2000, arg0=16, arg1=2, arg2=8, flags=0x0100181F),
    600: record(api_id=6, arg0=64, flags=0x0100020E),
    601: record(api_id=6, address_after=0x1000, arg0=64, flags=0x01000A0E),
    602: record(api_id=6, arg0=64, arg1=64, flags=0x0100200E),
    603: record(api_id=6, address_after=0x1000, arg0=64, arg1=64, flags=0x0100080E),
    700: record(api_id=7, arg0=64, result_code=22, flags=0x0200220C),
    701: record(api_id=7, arg0=64, arg1=64, result_code=22, flags=0x0200200C),
    702: record(api_id=7, arg0=64, result_code=0, flags=0x0200020E),
    703: record(api_id=7, address_after=0x1000, arg0=64, result_code=0, flags=0x02000A0E),
    704: record(api_id=7, address_after=0x1000, arg0=64, arg1=64, result_code=0, flags=0x0200080E),
    800: record(api_id=8, arg0=64, flags=0x0100020E),
    801: record(api_id=8, address_after=0x1000, arg0=64, flags=0x01000A0E),
    802: record(api_id=8, arg0=64, arg1=64, flags=0x0100200E),
    803: record(api_id=8, address_after=0x1000, arg0=64, arg1=64, flags=0x0100080E),
    900: record(api_id=9, flags=0x01000206),
    901: record(api_id=9, address_after=0x1000, flags=0x01000A06),
    902: record(api_id=9, arg0=64, flags=0x01002006),
    903: record(api_id=9, address_after=0x1000, arg0=64, flags=0x01000806),
    1000: record(api_id=10, flags=0x01000206),
    1001: record(api_id=10, address_after=0x1000, flags=0x01000A06),
    1002: record(api_id=10, arg0=64, flags=0x01002006),
    1003: record(api_id=10, address_after=0x1000, arg0=64, flags=0x01000806),
    1100: record(api_id=11, arg0=0x3000, flags=0x01002006),
    1101: record(api_id=11, address_after=0x1000, arg0=0x3000, flags=0x01000806),
    1200: record(api_id=12, arg0=0x3000, arg1=64, flags=0x0100200E),
    1201: record(api_id=12, address_after=0x1000, arg0=0x3000, arg1=64, flags=0x0100080E),
}


def classify(
    value: Record,
    *,
    mode: int = 4,
    policy: int | None = None,
    cutoff_before: int = 120,
    cutoff_after: int = 121,
    seal_bracket_available: bool = True,
):
    if value.api_id in (3, 5) and policy is None:
        policy = 2 if value.flags & (1 << 13) else 1
    return C.classify_record(
        EVENT,
        value,
        mode_id=mode,
        realloc_zero_policy_id=policy,
        cutoff_before_counter=cutoff_before,
        cutoff_after_counter=cutoff_after,
        seal_bracket_available=seal_bracket_available,
    )


def sampled_successor(
    *, thread: int = 1, sequence: int = 1, address: int = 0x1000,
    requested: int = 64, enter: int = 110, exit_: int = 111,
) -> Record:
    return record(
        thread_index=thread,
        thread_sequence=sequence,
        address_after=address,
        arg0=requested,
        counter_enter=enter,
        counter_exit=exit_,
        flags=0x01010866,
    )


def sampled_free(
    *, thread: int = 2, sequence: int = 1, origin_thread: int = 1,
    origin_sequence: int = 1, address: int = 0x1000, requested: int = 64,
    enter: int = 112, exit_: int = 113,
) -> Record:
    return record(
        thread_index=thread,
        thread_sequence=sequence,
        counter_enter=enter,
        counter_exit=exit_,
        address_before=address,
        arg0=requested,
        arg1=origin_thread,
        arg2=origin_sequence,
        flags=0x0102D07D if thread != origin_thread else 0x0100D07D,
        api_id=4,
        event_kind=3,
    )


def sampled_unknown_successor() -> Record:
    return record(
        thread_index=1,
        thread_sequence=1,
        counter_enter=110,
        counter_exit=111,
        address_after=0x1000,
        arg0=0x3000,
        flags=0x01010866,
        api_id=11,
    )


def sampled_unknown_free() -> Record:
    return record(
        thread_index=2,
        thread_sequence=1,
        counter_enter=112,
        counter_exit=113,
        address_before=0x1000,
        arg1=1,
        arg2=1,
        flags=0x0102D079,
        api_id=4,
        event_kind=3,
    )


class EventClassifierTests(unittest.TestCase):
    def test_batch_validates_and_compiles_event_semantics_once(self) -> None:
        records = tuple(
            dataclasses.replace(EXACT_TRANSITIONS[103], thread_sequence=sequence)
            for sequence in (1, 2, 3)
        )
        original_records = tuple(records)
        original_event = C.SEMANTIC.canonical_bytes(EVENT)
        original_validate = C._validated_definition
        original_compile = C._compiled_definition
        validate_calls = 0
        compile_calls = 0

        def counted_validate(value):
            nonlocal validate_calls
            validate_calls += 1
            return original_validate(value)

        def counted_compile(value):
            nonlocal compile_calls
            compile_calls += 1
            return original_compile(value)

        C._validated_definition = counted_validate
        C._compiled_definition = counted_compile
        try:
            classifications = C.classify_records(
                EVENT,
                records,
                mode_id=4,
                realloc_zero_policy_id=None,
                cutoff_before_counter=120,
                cutoff_after_counter=121,
                seal_bracket_available=True,
            )
        finally:
            C._compiled_definition = original_compile
            C._validated_definition = original_validate

        self.assertEqual(
            [classification.transition_id for classification in classifications],
            [103, 103, 103],
        )
        self.assertEqual((compile_calls, validate_calls), (1, 1))
        self.assertEqual(records, original_records)
        self.assertEqual(C.SEMANTIC.canonical_bytes(EVENT), original_event)

    def test_literal_identity_and_all_fifty_six_exact_transitions(self) -> None:
        self.assertEqual(hashlib.sha256(EVENT_RAW).hexdigest(), EVENT_SHA256)
        observed = {}
        for expected, value in EXACT_TRANSITIONS.items():
            with self.subTest(transition_id=expected):
                result = classify(value)
                observed[expected] = result.transition_name
                self.assertEqual(result.transition_id, expected)
                self.assertEqual(result.selection_case, "none")
                self.assertEqual(
                    result.required_one_mask
                    | result.required_zero_mask
                    | C.SEMANTIC.COMMON_EVIDENCE_MASK,
                    C.SEMANTIC.KNOWN_FLAGS_MASK,
                )
        self.assertEqual(
            set(observed),
            set(EVENT["mode_rules"][4]["allowed_transition_ids"]),
        )

    def test_checked_calloc_and_realloc_policy_edges(self) -> None:
        invalid = (
            dataclasses.replace(EXACT_TRANSITIONS[200], arg2=1),
            dataclasses.replace(EXACT_TRANSITIONS[200], address_after=1),
            dataclasses.replace(EXACT_TRANSITIONS[204], arg2=15),
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(C.EventClassifierError):
                classify(value)
        for policy in (None, True, 0, 3, 1.0, "1"):
            with self.subTest(policy=policy), self.assertRaises(C.EventClassifierError):
                C.classify_record(
                    EVENT,
                    EXACT_TRANSITIONS[304],
                    mode_id=4,
                    realloc_zero_policy_id=policy,
                    cutoff_before_counter=120,
                    cutoff_after_counter=121,
                    seal_bracket_available=True,
                )
        self.assertEqual(classify(EXACT_TRANSITIONS[304], policy=1).transition_id, 304)
        self.assertEqual(classify(EXACT_TRANSITIONS[305], policy=2).transition_id, 305)

    def test_modes_callsites_strict_types_and_sequences(self) -> None:
        with self.assertRaises(C.EventClassifierError):
            classify(EXACT_TRANSITIONS[103], mode=3)
        with self.assertRaises(C.EventClassifierError):
            classify(dataclasses.replace(EXACT_TRANSITIONS[103], callsite_id=1), mode=4)
        self.assertEqual(
            classify(dataclasses.replace(EXACT_TRANSITIONS[103], callsite_id=1), mode=5).transition_id,
            103,
        )
        with self.assertRaises(C.EventClassifierError):
            classify(EXACT_TRANSITIONS[103], mode=5)
        for field, bad in (("context_id", True), ("thread_index", 0xFFFFFFFF), ("flags", 1.0)):
            with self.subTest(field=field), self.assertRaises(C.EventClassifierError):
                classify(dataclasses.replace(EXACT_TRANSITIONS[103], **{field: bad}))
        records = (
            dataclasses.replace(EXACT_TRANSITIONS[103], thread_sequence=1),
            dataclasses.replace(EXACT_TRANSITIONS[103], thread_sequence=3),
        )
        self.assertEqual(
            C.exact_mode_sequence_gap_count(records, mode_id=4), 1
        )
        self.assertEqual(
            len(C.classify_records(
                EVENT, records, mode_id=4, realloc_zero_policy_id=None,
                cutoff_before_counter=120, cutoff_after_counter=121,
                seal_bracket_available=True,
            )),
            2,
        )
        reversed_records = tuple(reversed((
            dataclasses.replace(EXACT_TRANSITIONS[103], thread_sequence=1),
            dataclasses.replace(EXACT_TRANSITIONS[103], thread_sequence=2),
        )))
        with self.assertRaises(C.EventClassifierError):
            C.classify_records(
                EVENT, reversed_records, mode_id=4, realloc_zero_policy_id=None,
                cutoff_before_counter=120, cutoff_after_counter=121,
                seal_bracket_available=True,
            )
        sampled = (
            dataclasses.replace(sampled_successor(sequence=1), thread_sequence=1),
            dataclasses.replace(sampled_successor(sequence=3), thread_sequence=3),
        )
        self.assertEqual(
            len(C.classify_records(
                EVENT, sampled, mode_id=3, realloc_zero_policy_id=None,
                cutoff_before_counter=120, cutoff_after_counter=121,
                seal_bracket_available=True,
            )),
            2,
        )
        self.assertEqual(C.exact_mode_sequence_gap_count(sampled, mode_id=3), 0)
        for no_emission_mode in (0, 1, 2, 6):
            with self.subTest(no_emission_mode=no_emission_mode):
                self.assertEqual(
                    C.exact_mode_sequence_gap_count((), mode_id=no_emission_mode),
                    0,
                )
                self.assertEqual(
                    C.classify_records(
                        EVENT,
                        (),
                        mode_id=no_emission_mode,
                        realloc_zero_policy_id=None,
                        cutoff_before_counter=120,
                        cutoff_after_counter=121,
                        seal_bracket_available=True,
                    ),
                    (),
                )
                with self.assertRaises(C.EventClassifierError):
                    C.exact_mode_sequence_gap_count(
                        (EXACT_TRANSITIONS[103],), mode_id=no_emission_mode
                    )
                with self.assertRaises(C.EventClassifierError):
                    C.classify_records(
                        EVENT,
                        (EXACT_TRANSITIONS[103],),
                        mode_id=no_emission_mode,
                        realloc_zero_policy_id=None,
                        cutoff_before_counter=120,
                        cutoff_after_counter=121,
                        seal_bracket_available=True,
                    )
        with self.assertRaises(C.EventClassifierError):
            C.classify_records(
                EVENT,
                (),
                mode_id=2,
                realloc_zero_policy_id=None,
                cutoff_before_counter=122,
                cutoff_after_counter=121,
                seal_bracket_available=True,
            )
        for bad_mode in (True, 7):
            with self.subTest(bad_mode=bad_mode), self.assertRaises(
                C.EventClassifierError
            ):
                C.exact_mode_sequence_gap_count(records, mode_id=bad_mode)
        with self.assertRaises(C.EventClassifierError):
            C.exact_mode_sequence_gap_count("not-records", mode_id=4)

    def test_boundary_is_authoritative_but_valid_counters_must_agree(self) -> None:
        base = EXACT_TRANSITIONS[103]
        unknown = dataclasses.replace(base, flags=base.flags | (1 << 18))
        self.assertEqual(classify(unknown).transition_id, 103)
        with self.assertRaises(C.EventClassifierError):
            classify(unknown, seal_bracket_available=False)
        with self.assertRaises(C.EventClassifierError):
            classify(base, seal_bracket_available=1)
        straddling = dataclasses.replace(
            base,
            counter_enter=120,
            counter_exit=121,
            flags=base.flags | (1 << 5) | (1 << 6) | (1 << 18),
        )
        self.assertEqual(classify(straddling).transition_id, 103)
        for value in (
            dataclasses.replace(straddling, counter_exit=119),
            dataclasses.replace(
                straddling,
                counter_enter=122,
                counter_exit=123,
                flags=straddling.flags ^ (1 << 18),
            ),
        ):
            with self.subTest(value=value), self.assertRaises(C.EventClassifierError):
                classify(value)
        bracket_true = straddling
        bracket_false = dataclasses.replace(
            bracket_true, flags=bracket_true.flags ^ (1 << 18)
        )
        self.assertEqual(
            classify(bracket_true, cutoff_after=122).transition_id, 103
        )
        self.assertEqual(
            classify(bracket_false, cutoff_after=122).transition_id, 103
        )
        equality_true = dataclasses.replace(straddling, counter_exit=120)
        equality_false = dataclasses.replace(
            straddling,
            flags=straddling.flags ^ (1 << 18),
        )
        self.assertEqual(classify(equality_true).transition_id, 103)
        self.assertEqual(classify(equality_false).transition_id, 103)
        one_sided = dataclasses.replace(
            base, counter_enter=122, flags=base.flags | (1 << 5) | (1 << 18)
        )
        self.assertEqual(classify(one_sided).transition_id, 103)

    def test_cross_thread_bit_must_match_encoded_origin(self) -> None:
        value = sampled_free()
        self.assertEqual(classify(value, mode=3).transition_id, 401)
        for mutant in (
            dataclasses.replace(value, flags=value.flags ^ (1 << 17)),
            dataclasses.replace(value, thread_index=value.arg1, flags=value.flags),
        ):
            with self.subTest(mutant=mutant), self.assertRaises(
                C.EventClassifierError
            ):
                classify(mutant, mode=3)

    def test_sampled_reallocarray_uses_arg0_with_matched_origin(self) -> None:
        matched_positive = record(
            api_id=5,
            event_kind=2,
            address_before=0x1000,
            address_after=0x2000,
            arg0=48,
            arg1=1,
            arg2=1,
            flags=0x0101D81F,
        )
        positive = classify(matched_positive, mode=3, policy=1)
        self.assertEqual(positive.transition_id, 510)
        self.assertEqual(positive.selection_case, "both_same")

        matched_zero_nonnull = dataclasses.replace(
            matched_positive,
            arg0=0,
            flags=0x0101DA1F,
        )
        zero_nonnull = classify(matched_zero_nonnull, mode=3, policy=1)
        self.assertEqual(zero_nonnull.transition_id, 506)
        self.assertEqual(zero_nonnull.selection_case, "both_same")

        matched_zero_null = dataclasses.replace(
            matched_zero_nonnull,
            address_after=0,
            flags=0x0100D21F,
        )
        zero_null = classify(matched_zero_null, mode=3, policy=1)
        self.assertEqual(zero_null.transition_id, 504)
        self.assertEqual(zero_null.selection_case, "predecessor_same")

        matched_cross = dataclasses.replace(
            matched_positive,
            thread_index=2,
            flags=matched_positive.flags | (1 << 17),
        )
        self.assertEqual(
            classify(matched_cross, mode=3, policy=1).selection_case,
            "both_cross",
        )

        for mutant in (
            dataclasses.replace(matched_positive, address_before=0),
            dataclasses.replace(
                matched_positive,
                flags=matched_positive.flags | (1 << 19),
            ),
        ):
            with self.subTest(mutant=mutant), self.assertRaises(
                C.EventClassifierError
            ):
                classify(mutant, mode=3, policy=1)

        self.assertEqual(classify(EXACT_TRANSITIONS[510]).transition_id, 510)

    def test_a03_replay_resolves_and_ends_one_selected_instance(self) -> None:
        records = (sampled_successor(), sampled_free())
        classifications = C.classify_records(
            EVENT, records, mode_id=3, realloc_zero_policy_id=None,
            cutoff_before_counter=120, cutoff_after_counter=121,
            seal_bracket_available=True,
        )
        self.assertEqual(
            C.reconcile_a03_predecessors(
                records, classifications, process_generation=7
            ),
            C.A03ReplayResult(1, 1, 0),
        )
        self.assertEqual(records, (sampled_successor(), sampled_free()))

    def test_a03_unknown_requested_size_replays_and_uses_transition_402(self) -> None:
        records = (sampled_unknown_successor(), sampled_unknown_free())
        classifications = C.classify_records(
            EVENT, records, mode_id=3, realloc_zero_policy_id=None,
            cutoff_before_counter=120, cutoff_after_counter=121,
            seal_bracket_available=True,
        )
        self.assertEqual(classifications[0].transition_id, 1101)
        self.assertEqual(classifications[1].transition_id, 402)
        self.assertEqual(
            C.reconcile_a03_predecessors(
                records, classifications, process_generation=7
            ),
            C.A03ReplayResult(1, 1, 0),
        )

    def test_a03_replay_rejects_origin_address_size_order_and_double_end(self) -> None:
        birth = sampled_successor()
        end = sampled_free()
        mutations = (
            (birth, dataclasses.replace(end, arg2=2)),
            (birth, dataclasses.replace(end, address_before=0x2000)),
            (birth, dataclasses.replace(end, arg0=65)),
        )
        for records in mutations:
            with self.subTest(records=records), self.assertRaises(C.EventClassifierError):
                classes = C.classify_records(
                    EVENT, records, mode_id=3, realloc_zero_policy_id=None,
                    cutoff_before_counter=120, cutoff_after_counter=121,
                    seal_bracket_available=True,
                )
                C.reconcile_a03_predecessors(records, classes, process_generation=7)
        death_before_birth = (
            birth, dataclasses.replace(end, counter_enter=109, counter_exit=110)
        )
        classes = C.classify_records(
            EVENT, death_before_birth, mode_id=3, realloc_zero_policy_id=None,
            cutoff_before_counter=120, cutoff_after_counter=121,
            seal_bracket_available=True,
        )
        with self.assertRaisesRegex(
            C.EventClassifierError, "predecessor death precedes birth"
        ):
            C.reconcile_a03_predecessors(
                death_before_birth, classes, process_generation=7
            )
        double = (birth, end, dataclasses.replace(end, thread_sequence=2))
        classes = C.classify_records(
            EVENT, double, mode_id=3, realloc_zero_policy_id=None,
            cutoff_before_counter=120, cutoff_after_counter=121,
            seal_bracket_available=True,
        )
        with self.assertRaises(C.EventClassifierError):
            C.reconcile_a03_predecessors(double, classes, process_generation=7)
        duplicate_live = (
            birth,
            sampled_successor(
                thread=2, sequence=1, address=0x1000, enter=112, exit_=113
            ),
        )
        classes = C.classify_records(
            EVENT, duplicate_live, mode_id=3, realloc_zero_policy_id=None,
            cutoff_before_counter=120, cutoff_after_counter=121,
            seal_bracket_available=True,
        )
        with self.assertRaises(C.EventClassifierError):
            C.reconcile_a03_predecessors(
                duplicate_live, classes, process_generation=7
            )

        overlapping_ended = (
            sampled_successor(
                thread=1, sequence=1, address=0x1000, enter=110, exit_=111
            ),
            sampled_successor(
                thread=2, sequence=1, address=0x1000, enter=112, exit_=113
            ),
            sampled_free(
                thread=1,
                sequence=2,
                origin_thread=1,
                origin_sequence=1,
                address=0x1000,
                enter=114,
                exit_=115,
            ),
            sampled_free(
                thread=2,
                sequence=2,
                origin_thread=2,
                origin_sequence=1,
                address=0x1000,
                enter=116,
                exit_=117,
            ),
        )
        classes = C.classify_records(
            EVENT, overlapping_ended, mode_id=3, realloc_zero_policy_id=None,
            cutoff_before_counter=120, cutoff_after_counter=121,
            seal_bracket_available=True,
        )
        with self.assertRaisesRegex(
            C.EventClassifierError, "intersecting same-address lifetimes"
        ):
            C.reconcile_a03_predecessors(
                overlapping_ended, classes, process_generation=7
            )

    def test_same_address_realloc_creates_a_new_identity(self) -> None:
        birth = sampled_successor(thread=1, sequence=1)
        realloc = record(
            thread_index=1,
            thread_sequence=2,
            counter_enter=112,
            counter_exit=113,
            address_before=0x1000,
            address_after=0x1000,
            arg0=128,
            arg1=1,
            arg2=1,
            flags=0x0101D87F,
            api_id=3,
            event_kind=2,
        )
        records = (birth, realloc)
        classes = C.classify_records(
            EVENT, records, mode_id=3, realloc_zero_policy_id=1,
            cutoff_before_counter=120, cutoff_after_counter=121,
            seal_bracket_available=True,
        )
        self.assertEqual(classes[1].transition_id, 308)
        self.assertEqual(
            C.reconcile_a03_predecessors(records, classes, process_generation=7),
            C.A03ReplayResult(2, 1, 1),
        )
        adjacent = dataclasses.replace(realloc, counter_enter=113)
        adjacent_records = (birth, adjacent)
        adjacent_classes = C.classify_records(
            EVENT, adjacent_records, mode_id=3, realloc_zero_policy_id=1,
            cutoff_before_counter=120, cutoff_after_counter=121,
            seal_bracket_available=True,
        )
        self.assertEqual(
            C.reconcile_a03_predecessors(
                adjacent_records, adjacent_classes, process_generation=7
            ),
            C.A03ReplayResult(2, 1, 1),
        )
        stale = dataclasses.replace(realloc, thread_sequence=1, arg2=1)
        stale_records = (birth, stale)
        with self.assertRaises(C.EventClassifierError):
            stale_classes = C.classify_records(
                EVENT, stale_records, mode_id=3, realloc_zero_policy_id=1,
                cutoff_before_counter=120, cutoff_after_counter=121,
                seal_bracket_available=True,
            )
            C.reconcile_a03_predecessors(
                stale_records, stale_classes, process_generation=7
            )


if __name__ == "__main__":
    unittest.main()
