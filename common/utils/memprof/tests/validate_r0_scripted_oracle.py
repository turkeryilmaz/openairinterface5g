#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Independent absolute and differential oracle for R0 allocation wrappers."""

from __future__ import annotations

import argparse
import dataclasses
import errno
import hashlib
import os
import pathlib
import re
import sys
import tempfile
from collections import Counter, defaultdict
from typing import Callable, Iterable, Sequence

from r0_harness_common import (
    BoundedFileError,
    BoundedFileImage,
    BoundedProcessResult,
    format_byte_evidence,
    loader_environment_variables,
    read_regular_file_bounded,
    run_process_bounded,
)


SCHEMA = "oai-memprof-r0-raw-v1"
MUTATION_PHYSICAL_SCHEMA = "oai-memprof-r0-mutation-physical-v1"
EXPECTED_CANONICAL_SHA256 = "9ec1834b04b1ba23be60417ee9a5cb04cccb8b7617b4c14b0498f9d2ba8a2606"
POINTER_TOKENS = ("NULL", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8")
MAX_RAW_BYTES = 1 << 20
MAX_RAW_RECORDS = 256
MAX_RAW_LINE_BYTES = 512
MAX_CHILD_STDERR_BYTES = 64 << 10
DEFAULT_CHILD_TIMEOUT_SECONDS = 10.0
DECLARED_MUTATIONS = (
    "duplicate-real",
    "operand",
    "errno",
    "result",
    "suppress-free-null",
    "context",
)


@dataclasses.dataclass(frozen=True)
class Evaluator:
    identifier: int
    operand: str
    kind: str
    value: int | str


@dataclasses.dataclass(frozen=True)
class Transaction:
    identifier: int
    phase: str
    api: str
    arg0: int | str
    arg1: int
    result: str
    evaluators: tuple[Evaluator, ...] = ()
    failure: bool = False

    @property
    def errno_in(self) -> int:
        return 1000 + self.identifier

    @property
    def errno_out(self) -> int:
        return errno.ENOMEM if self.failure else self.errno_in


# This table is intentionally independent of the C backend's switch tables.
# A change to either side requires an explicit review and canonical-hash update.
EXPECTED_TRANSACTIONS: tuple[Transaction, ...] = (
    Transaction(1, "CTOR", "malloc", 16, 0, "P1", (Evaluator(1, "size", "SIZE", 16),)),
    Transaction(
        2,
        "CTOR",
        "calloc",
        2,
        8,
        "P2",
        (Evaluator(2, "count", "SIZE", 2), Evaluator(3, "size", "SIZE", 8)),
    ),
    Transaction(3, "MAIN", "malloc", 0, 0, "P3", (Evaluator(4, "size", "SIZE", 0),)),
    Transaction(4, "MAIN", "malloc", 64, 0, "NULL", (Evaluator(5, "size", "SIZE", 64),), True),
    Transaction(
        5,
        "MAIN",
        "calloc",
        0,
        4,
        "P4",
        (Evaluator(6, "count", "SIZE", 0), Evaluator(7, "size", "SIZE", 4)),
    ),
    Transaction(
        6,
        "MAIN",
        "calloc",
        3,
        0,
        "P5",
        (Evaluator(8, "count", "SIZE", 3), Evaluator(9, "size", "SIZE", 0)),
    ),
    Transaction(
        7,
        "MAIN",
        "calloc",
        4,
        8,
        "NULL",
        (Evaluator(10, "count", "SIZE", 4), Evaluator(11, "size", "SIZE", 8)),
        True,
    ),
    Transaction(
        8,
        "MAIN",
        "calloc",
        (1 << 64) - 1,
        2,
        "NULL",
        (Evaluator(12, "count", "SIZE", (1 << 64) - 1), Evaluator(13, "size", "SIZE", 2)),
        True,
    ),
    Transaction(
        9,
        "MAIN",
        "realloc",
        "NULL",
        24,
        "P6",
        (Evaluator(14, "pointer", "PTR", "NULL"), Evaluator(15, "size", "SIZE", 24)),
    ),
    Transaction(
        10,
        "MAIN",
        "realloc",
        "P1",
        32,
        "P1",
        (Evaluator(16, "pointer", "PTR", "P1"), Evaluator(17, "size", "SIZE", 32)),
    ),
    Transaction(
        11,
        "MAIN",
        "realloc",
        "P2",
        48,
        "P7",
        (Evaluator(18, "pointer", "PTR", "P2"), Evaluator(19, "size", "SIZE", 48)),
    ),
    Transaction(
        12,
        "MAIN",
        "realloc",
        "P7",
        96,
        "NULL",
        (Evaluator(20, "pointer", "PTR", "P7"), Evaluator(21, "size", "SIZE", 96)),
        True,
    ),
    Transaction(
        13,
        "MAIN",
        "realloc",
        "P3",
        0,
        "NULL",
        (Evaluator(22, "pointer", "PTR", "P3"), Evaluator(23, "size", "SIZE", 0)),
    ),
    Transaction(
        14,
        "MAIN",
        "realloc",
        "NULL",
        0,
        "P8",
        (Evaluator(24, "pointer", "PTR", "NULL"), Evaluator(25, "size", "SIZE", 0)),
    ),
    Transaction(
        15, "MAIN", "free", "NULL", 0, "NULL", (Evaluator(26, "pointer", "PTR", "NULL"),)
    ),
    Transaction(16, "MAIN", "free", "P1", 0, "NULL", (Evaluator(27, "pointer", "PTR", "P1"),)),
    Transaction(17, "MAIN", "free", "P4", 0, "NULL", (Evaluator(28, "pointer", "PTR", "P4"),)),
    Transaction(18, "MAIN", "free", "P5", 0, "NULL"),
    Transaction(19, "MAIN", "free", "P6", 0, "NULL"),
    Transaction(
        20,
        "MAIN",
        "realloc",
        "P8",
        7,
        "P8",
        (Evaluator(29, "pointer", "PTR", "P8"), Evaluator(30, "size", "SIZE", 7)),
    ),
    Transaction(21, "DTOR", "free", "P7", 0, "NULL"),
    Transaction(22, "DTOR", "free", "P8", 0, "NULL"),
)


@dataclasses.dataclass(frozen=True)
class Problem:
    code: str
    detail: str


@dataclasses.dataclass(frozen=True)
class MetaRecord:
    line: int
    key: str
    value: str


@dataclasses.dataclass(frozen=True)
class TokenRecord:
    line: int
    token: str
    address: int


@dataclasses.dataclass(frozen=True)
class EvalRecord:
    line: int
    transaction: int
    phase: str
    operand: str
    evaluator: int
    value_kind: str
    value: int


@dataclasses.dataclass(frozen=True)
class RealRecord:
    line: int
    sequence: int
    transaction: int
    api: str
    arg0: int
    arg1: int
    result: int
    errno_in: int
    errno_out: int


@dataclasses.dataclass(frozen=True)
class CallerRecord:
    line: int
    transaction: int
    phase: str
    api: str
    arg0: int
    arg1: int
    result: int
    errno_in: int
    errno_out: int


@dataclasses.dataclass(frozen=True)
class SummaryRecord:
    line: int
    real_calls: int
    evaluator_calls: int
    live_allocations: int
    context_probes: int
    evaluator_faults: int
    emit_failures: int


TypedRecord = MetaRecord | TokenRecord | EvalRecord | RealRecord | CallerRecord | SummaryRecord


@dataclasses.dataclass(frozen=True)
class ParsedTranscript:
    records: tuple[TypedRecord, ...]
    problems: tuple[Problem, ...]
    source_image: BoundedFileImage | None = None


@dataclasses.dataclass(frozen=True)
class Validation:
    problems: tuple[Problem, ...]
    canonical: str
    physical_normalized: str = ""
    physical_mode: str = ""
    physical_problems: tuple[Problem, ...] = ()


_CANONICAL_DECIMAL = re.compile(r"(?:0|-?[1-9][0-9]*)\Z")
_CANONICAL_HEX = re.compile(r"0x(?:0|[1-9a-f][0-9a-f]*)\Z")
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1


def _decimal(text: str, minimum: int, maximum: int) -> int:
    if _CANONICAL_DECIMAL.fullmatch(text) is None:
        raise ValueError(f"noncanonical decimal {text!r}")
    value = int(text, 10)
    if not minimum <= value <= maximum:
        raise ValueError(f"decimal {text!r} is outside [{minimum}, {maximum}]")
    return value


def _hexadecimal(text: str) -> int:
    if _CANONICAL_HEX.fullmatch(text) is None:
        raise ValueError(f"noncanonical hexadecimal {text!r}")
    value = int(text, 16)
    if value > _UINT64_MAX:
        raise ValueError(f"hexadecimal {text!r} exceeds 64 bits")
    return value


def _typed_record(line_number: int, fields: tuple[str, ...]) -> TypedRecord:
    kind = fields[0]
    if kind == "META":
        return MetaRecord(line_number, fields[1], fields[2])
    if kind == "TOKEN":
        return TokenRecord(line_number, fields[1], _hexadecimal(fields[2]))
    if kind == "EVAL":
        return EvalRecord(
            line_number,
            _decimal(fields[1], 1, _UINT32_MAX),
            fields[2],
            fields[3],
            _decimal(fields[4], 1, _UINT32_MAX),
            fields[5],
            _hexadecimal(fields[6]),
        )
    if kind == "REAL":
        return RealRecord(
            line_number,
            _decimal(fields[1], 1, _UINT32_MAX),
            _decimal(fields[2], 1, _UINT32_MAX),
            fields[3],
            _hexadecimal(fields[4]),
            _hexadecimal(fields[5]),
            _hexadecimal(fields[6]),
            _decimal(fields[7], _INT32_MIN, _INT32_MAX),
            _decimal(fields[8], _INT32_MIN, _INT32_MAX),
        )
    if kind == "CALLER":
        return CallerRecord(
            line_number,
            _decimal(fields[1], 1, _UINT32_MAX),
            fields[2],
            fields[3],
            _hexadecimal(fields[4]),
            _hexadecimal(fields[5]),
            _hexadecimal(fields[6]),
            _decimal(fields[7], _INT32_MIN, _INT32_MAX),
            _decimal(fields[8], _INT32_MIN, _INT32_MAX),
        )
    if kind == "SUMMARY":
        return SummaryRecord(
            line_number,
            *(_decimal(field, 0, _UINT32_MAX) for field in fields[1:]),
        )
    raise ValueError(f"unknown record kind {kind!r}")


def parse_transcript_bytes(raw: bytes, source: str = "<memory>") -> ParsedTranscript:
    problems: list[Problem] = []
    if len(raw) > MAX_RAW_BYTES:
        return ParsedTranscript((), (Problem("RAW_SIZE", f"{source}: exceeds {MAX_RAW_BYTES} bytes"),))
    for offset, byte in enumerate(raw):
        if byte != 0x0A and not 0x20 <= byte <= 0x7E:
            return ParsedTranscript(
                (),
                (Problem("RAW_CHARACTER", f"{source}: byte {offset} is 0x{byte:02x}, expected printable ASCII or LF"),),
            )

    terminated = raw.endswith(b"\n")
    if not terminated:
        problems.append(Problem("TERMINATION", f"{source}: raw transcript does not end in LF"))
    raw_lines = raw.split(b"\n")
    if terminated:
        raw_lines = raw_lines[:-1]
    if len(raw_lines) > MAX_RAW_RECORDS:
        return ParsedTranscript(
            (),
            tuple(
                problems
                + [Problem("RECORD_LIMIT", f"{source}: {len(raw_lines)} records exceed limit {MAX_RAW_RECORDS}")]
            ),
        )
    for line_number, line in enumerate(raw_lines, start=1):
        if len(line) > MAX_RAW_LINE_BYTES:
            return ParsedTranscript(
                (),
                tuple(
                    problems
                    + [Problem("LINE_SIZE", f"{source}: line {line_number} exceeds {MAX_RAW_LINE_BYTES} bytes")]
                ),
            )

    expected_lengths = {"META": 3, "TOKEN": 3, "EVAL": 7, "REAL": 9, "CALLER": 9, "SUMMARY": 7}
    records: list[TypedRecord] = []
    for line_number, raw_line in enumerate(raw_lines, start=1):
        line = raw_line.decode("ascii")
        fields = tuple(line.split("|"))
        kind = fields[0] if fields else ""
        if kind not in expected_lengths:
            problems.append(Problem("RECORD_KIND", f"line {line_number}: unknown record {kind!r}"))
            continue
        if len(fields) != expected_lengths[kind]:
            problems.append(
                Problem(
                    "FIELD_COUNT",
                    f"line {line_number}: {kind} has {len(fields)}, expected {expected_lengths[kind]}",
                )
            )
            continue
        try:
            records.append(_typed_record(line_number, fields))
        except ValueError as error:
            problems.append(Problem("INTEGER", f"line {line_number}: {error}"))
    return ParsedTranscript(tuple(records), tuple(problems))


def parse_transcript(path: pathlib.Path) -> ParsedTranscript:
    try:
        image = read_regular_file_bounded(path, MAX_RAW_BYTES)
    except BoundedFileError as error:
        return ParsedTranscript((), (Problem(error.code, error.detail),))
    parsed = parse_transcript_bytes(image.data, str(path))
    return ParsedTranscript(parsed.records, parsed.problems, image)


def add(problems: list[Problem], condition: bool, code: str, detail: str) -> None:
    if not condition:
        problems.append(Problem(code, detail))


def expected_raw(value: int | str, tokens: dict[str, int]) -> int | None:
    return tokens.get(value) if isinstance(value, str) else value


def canonical_value(value: int, pointer: bool, reverse_tokens: dict[int, str]) -> str:
    if pointer:
        return reverse_tokens.get(value, f"UNKNOWN@0x{value:x}")
    return str(value)


def _physical_record_line(record: TypedRecord, reverse_tokens: dict[int, str]) -> str:
    if isinstance(record, MetaRecord):
        return f"META|{record.key}|{record.value}"
    if isinstance(record, TokenRecord):
        return f"TOKEN|{record.token}|{canonical_value(record.address, True, reverse_tokens)}"
    if isinstance(record, EvalRecord):
        return "EVAL|{}|{}|{}|{}|{}|{}".format(
            record.transaction,
            record.phase,
            record.operand,
            record.evaluator,
            record.value_kind,
            canonical_value(record.value, record.value_kind == "PTR", reverse_tokens),
        )
    if isinstance(record, RealRecord):
        return "REAL|{}|{}|{}|{}|{}|{}|{}|{}".format(
            record.sequence,
            record.transaction,
            record.api,
            canonical_value(record.arg0, record.api in {"realloc", "free"}, reverse_tokens),
            record.arg1,
            canonical_value(record.result, True, reverse_tokens),
            record.errno_in,
            record.errno_out,
        )
    if isinstance(record, CallerRecord):
        return "CALLER|{}|{}|{}|{}|{}|{}|{}|{}".format(
            record.transaction,
            record.phase,
            record.api,
            canonical_value(record.arg0, record.api in {"realloc", "free"}, reverse_tokens),
            record.arg1,
            canonical_value(record.result, True, reverse_tokens),
            record.errno_in,
            record.errno_out,
        )
    return "SUMMARY|{}|{}|{}|{}|{}|{}".format(
        record.real_calls,
        record.evaluator_calls,
        record.live_allocations,
        record.context_probes,
        record.evaluator_faults,
        record.emit_failures,
    )


def normalize_physical_transcript(parsed: ParsedTranscript) -> tuple[str, tuple[Problem, ...]]:
    """Normalize addresses and only same-transaction contiguous EVAL order.

    Every successfully parsed physical record remains represented in its
    original position.  The sole ordering equivalence is a contiguous run of
    evaluators carrying one transaction identifier, because C does not specify
    function-argument evaluation order.
    """

    token_records = [record for record in parsed.records if isinstance(record, TokenRecord)]
    token_names = [record.token for record in token_records]
    physical_problems: list[Problem] = []
    if Counter(token_names) != Counter(POINTER_TOKENS):
        physical_problems.append(Problem("PHYSICAL_TOKEN_SET", f"physical tokens are {token_names!r}"))
    if token_records:
        null_records = [record for record in token_records if record.token == "NULL"]
        if len(null_records) != 1 or null_records[0].address != 0:
            physical_problems.append(Problem("PHYSICAL_NULL_TOKEN", "physical NULL token is not uniquely zero"))
        nonnull_addresses = [record.address for record in token_records if record.token != "NULL"]
        if any(address == 0 for address in nonnull_addresses) or len(set(nonnull_addresses)) != len(
            nonnull_addresses
        ):
            physical_problems.append(
                Problem("PHYSICAL_TOKEN_IDENTITY", "physical non-NULL token addresses are zero or aliased")
            )

    reverse_tokens = {record.address: record.token for record in token_records}
    lines: list[str] = []
    cursor = 0
    while cursor < len(parsed.records):
        record = parsed.records[cursor]
        if isinstance(record, EvalRecord):
            transaction = record.transaction
            run: list[str] = []
            while cursor < len(parsed.records):
                candidate = parsed.records[cursor]
                if not isinstance(candidate, EvalRecord) or candidate.transaction != transaction:
                    break
                run.append(_physical_record_line(candidate, reverse_tokens))
                cursor += 1
            lines.extend(sorted(run))
            continue
        lines.append(_physical_record_line(record, reverse_tokens))
        cursor += 1
    return "\n".join(lines) + "\n", tuple(physical_problems)


def mutation_physical_sha256(name: str, mode: str, normalized: str) -> str:
    domain = f"{MUTATION_PHYSICAL_SCHEMA}\nMODE|{mode}\nMUTATION|{name}\n"
    return hashlib.sha256((domain + normalized).encode("ascii")).hexdigest()


def _record_description(record: TypedRecord | None) -> str:
    if record is None:
        return "EOF"
    return f"line {record.line} {type(record).__name__}"


def _validate_global_order(parsed: ParsedTranscript, expected_mode: str, problems: list[Problem]) -> None:
    records = parsed.records
    cursor = 0

    def take(predicate: Callable[[TypedRecord], bool], expected: str) -> TypedRecord | None:
        nonlocal cursor
        record = records[cursor] if cursor < len(records) else None
        if record is None or not predicate(record):
            problems.append(
                Problem(
                    "GLOBAL_ORDER",
                    f"record {cursor + 1}: expected {expected}, found {_record_description(record)}",
                )
            )
            return None
        cursor += 1
        return record

    if (
        take(lambda item: isinstance(item, MetaRecord) and item.key == "schema" and item.value == SCHEMA, "schema META")
        is None
    ):
        return
    if (
        take(
            lambda item: isinstance(item, MetaRecord) and item.key == "mode" and item.value == expected_mode,
            "mode META",
        )
        is None
    ):
        return
    for token in POINTER_TOKENS:
        if (
            take(
                lambda item, token=token: isinstance(item, TokenRecord) and item.token == token,
                f"TOKEN {token}",
            )
            is None
        ):
            return
    for transaction in EXPECTED_TRANSACTIONS:
        for _ in transaction.evaluators:
            if take(
                lambda item, identifier=transaction.identifier: isinstance(item, EvalRecord)
                and item.transaction == identifier,
                f"transaction {transaction.identifier} EVAL",
            ) is None:
                return
        if take(
            lambda item, identifier=transaction.identifier: isinstance(item, RealRecord)
            and item.transaction == identifier,
            f"transaction {transaction.identifier} REAL",
        ) is None:
            return
        if take(
            lambda item, identifier=transaction.identifier: isinstance(item, CallerRecord)
            and item.transaction == identifier,
            f"transaction {transaction.identifier} CALLER",
        ) is None:
            return
    if take(lambda item: isinstance(item, SummaryRecord), "terminal SUMMARY") is None:
        return
    if cursor != len(records):
        problems.append(
            Problem("GLOBAL_ORDER", f"record {cursor + 1}: unexpected trailing {_record_description(records[cursor])}")
        )


def validate_parsed(parsed: ParsedTranscript, expected_mode: str) -> Validation:
    problems = list(parsed.problems)

    metadata: dict[str, str] = {}
    tokens: dict[str, int] = {}
    evaluations: list[EvalRecord] = []
    reals: list[RealRecord] = []
    callers: list[CallerRecord] = []
    summaries: list[SummaryRecord] = []
    for record in parsed.records:
        if isinstance(record, MetaRecord):
            if record.key in metadata:
                problems.append(Problem("META_DUPLICATE", f"line {record.line}: duplicate metadata key {record.key}"))
            else:
                metadata[record.key] = record.value
        elif isinstance(record, TokenRecord):
            if record.token in tokens:
                problems.append(Problem("TOKEN_DUPLICATE", f"line {record.line}: duplicate token {record.token}"))
            else:
                tokens[record.token] = record.address
        elif isinstance(record, EvalRecord):
            evaluations.append(record)
        elif isinstance(record, RealRecord):
            reals.append(record)
        elif isinstance(record, CallerRecord):
            callers.append(record)
        else:
            summaries.append(record)

    add(problems, metadata.get("schema") == SCHEMA, "SCHEMA", f"schema is {metadata.get('schema')!r}")
    add(problems, metadata.get("mode") == expected_mode, "MODE", f"mode is {metadata.get('mode')!r}")
    add(
        problems,
        set(metadata) == {"schema", "mode"},
        "META_SET",
        f"metadata keys are {sorted(metadata)}",
    )
    add(problems, set(tokens) == set(POINTER_TOKENS), "TOKEN_SET", f"tokens are {sorted(tokens)}")
    if set(tokens) == set(POINTER_TOKENS):
        add(problems, tokens["NULL"] == 0, "NULL_TOKEN", "NULL token is nonzero")
        nonnull = [tokens[token] for token in POINTER_TOKENS[1:]]
        add(problems, all(nonnull), "NONNULL_TOKEN", "one or more Pn tokens are zero")
        add(problems, len(set(nonnull)) == len(nonnull), "TOKEN_ALIAS", "Pn token addresses are not unique")

    reverse_tokens = {address: token for token, address in tokens.items()}
    real_by_transaction: dict[int, list[RealRecord]] = defaultdict(list)
    caller_by_transaction: dict[int, list[CallerRecord]] = defaultdict(list)
    eval_by_transaction: dict[int, list[EvalRecord]] = defaultdict(list)
    for record in reals:
        real_by_transaction[record.transaction].append(record)
    for record in callers:
        caller_by_transaction[record.transaction].append(record)
    for record in evaluations:
        eval_by_transaction[record.transaction].append(record)

    canonical_lines: list[str] = [f"SCHEMA|{SCHEMA}"]
    seen_evaluator_ids: list[int] = []
    for expected in EXPECTED_TRANSACTIONS:
        real_records = real_by_transaction.get(expected.identifier, [])
        caller_records = caller_by_transaction.get(expected.identifier, [])
        eval_records = eval_by_transaction.get(expected.identifier, [])
        add(
            problems,
            len(real_records) == 1,
            "REAL_MULTIPLICITY",
            f"transaction {expected.identifier}: {len(real_records)} REAL records",
        )
        add(
            problems,
            len(caller_records) == 1,
            "CALLER_MULTIPLICITY",
            f"transaction {expected.identifier}: {len(caller_records)} CALLER records",
        )

        expected_evaluators = {
            (item.identifier, item.operand, item.kind, expected_raw(item.value, tokens)) for item in expected.evaluators
        }
        actual_evaluators: list[tuple[int, str, str, int]] = []
        for record in eval_records:
            seen_evaluator_ids.append(record.evaluator)
            add(
                problems,
                record.phase == expected.phase,
                "EVAL_PHASE",
                f"transaction {expected.identifier}: evaluator phase {record.phase!r}",
            )
            actual_evaluators.append((record.evaluator, record.operand, record.value_kind, record.value))
        add(
            problems,
            Counter(actual_evaluators) == Counter(expected_evaluators),
            "EVALUATOR_SET",
            f"transaction {expected.identifier}: evaluator set differs",
        )

        if len(real_records) == 1:
            record = real_records[0]
            add(
                problems,
                record.sequence == expected.identifier,
                "REAL_SEQUENCE",
                f"transaction {expected.identifier}: sequence {record.sequence}",
            )
            add(
                problems,
                record.api == expected.api,
                "REAL_API",
                f"transaction {expected.identifier}: API {record.api}",
            )
            add(
                problems,
                record.arg0 == expected_raw(expected.arg0, tokens) and record.arg1 == expected.arg1,
                "REAL_OPERAND",
                f"transaction {expected.identifier}: operands {record.arg0}, {record.arg1}",
            )
            add(
                problems,
                record.result == expected_raw(expected.result, tokens),
                "REAL_RESULT",
                f"transaction {expected.identifier}: result {record.result}",
            )
            add(
                problems,
                record.errno_in == expected.errno_in and record.errno_out == expected.errno_out,
                "REAL_ERRNO",
                f"transaction {expected.identifier}: errno {record.errno_in}->{record.errno_out}",
            )
            arg0_pointer = expected.api in {"realloc", "free"}
            canonical_lines.append(
                "REAL|{}|{}|{}|{}|{}|{}|{}|{}".format(
                    record.sequence,
                    expected.identifier,
                    record.api,
                    canonical_value(record.arg0, arg0_pointer, reverse_tokens),
                    canonical_value(record.arg1, False, reverse_tokens),
                    canonical_value(record.result, True, reverse_tokens),
                    record.errno_in,
                    record.errno_out,
                )
            )

        if len(caller_records) == 1:
            record = caller_records[0]
            add(
                problems,
                record.phase == expected.phase,
                "CALLER_PHASE",
                f"transaction {expected.identifier}: phase {record.phase}",
            )
            add(
                problems,
                record.api == expected.api,
                "CALLER_API",
                f"transaction {expected.identifier}: API {record.api}",
            )
            add(
                problems,
                record.arg0 == expected_raw(expected.arg0, tokens) and record.arg1 == expected.arg1,
                "CALLER_OPERAND",
                f"transaction {expected.identifier}: operands {record.arg0}, {record.arg1}",
            )
            add(
                problems,
                record.result == expected_raw(expected.result, tokens),
                "CALLER_RESULT",
                f"transaction {expected.identifier}: result {record.result}",
            )
            add(
                problems,
                record.errno_in == expected.errno_in and record.errno_out == expected.errno_out,
                "CALLER_ERRNO",
                f"transaction {expected.identifier}: errno {record.errno_in}->{record.errno_out}",
            )
            arg0_pointer = expected.api in {"realloc", "free"}
            canonical_lines.append(
                "CALLER|{}|{}|{}|{}|{}|{}|{}|{}".format(
                    expected.identifier,
                    record.phase,
                    record.api,
                    canonical_value(record.arg0, arg0_pointer, reverse_tokens),
                    canonical_value(record.arg1, False, reverse_tokens),
                    canonical_value(record.result, True, reverse_tokens),
                    record.errno_in,
                    record.errno_out,
                )
            )

        for record in sorted(eval_records, key=lambda item: item.evaluator):
            canonical_lines.append(
                "EVAL|{}|{}|{}|{}|{}|{}".format(
                    expected.identifier,
                    record.phase,
                    record.evaluator,
                    record.operand,
                    record.value_kind,
                    canonical_value(record.value, record.value_kind == "PTR", reverse_tokens),
                )
            )

    expected_ids = list(range(1, 31))
    add(
        problems,
        sorted(seen_evaluator_ids) == expected_ids,
        "EVALUATOR_IDS",
        f"evaluator IDs are {sorted(seen_evaluator_ids)}",
    )
    add(
        problems,
        set(real_by_transaction) == set(range(1, 23)),
        "REAL_TRANSACTION_SET",
        f"REAL transaction IDs are {sorted(real_by_transaction)}",
    )
    add(
        problems,
        set(caller_by_transaction) == set(range(1, 23)),
        "CALLER_TRANSACTION_SET",
        f"CALLER transaction IDs are {sorted(caller_by_transaction)}",
    )

    if len(summaries) != 1:
        problems.append(Problem("SUMMARY_MULTIPLICITY", f"found {len(summaries)} SUMMARY records"))
    else:
        summary = summaries[0]
        values = (
            summary.real_calls,
            summary.evaluator_calls,
            summary.live_allocations,
            summary.context_probes,
            summary.evaluator_faults,
            summary.emit_failures,
        )
        expected_summary = (22, 30, 0, 0, 0, 0)
        for index, (actual, expected_value) in enumerate(zip(values, expected_summary), start=1):
            code = ("REAL_COUNT", "EVAL_COUNT", "LIVE_COUNT", "CONTEXT_PROBE", "EVAL_FAULT", "EMIT_FAILURE")[index - 1]
            add(
                problems,
                actual == expected_value,
                code,
                f"SUMMARY field {index} is {actual}, expected {expected_value}",
            )
        canonical_lines.append("SUMMARY|" + "|".join(str(value) for value in values))

    _validate_global_order(parsed, expected_mode, problems)
    canonical = "\n".join(canonical_lines) + "\n"
    physical_normalized, physical_problems = normalize_physical_transcript(parsed)
    return Validation(tuple(problems), canonical, physical_normalized, expected_mode, physical_problems)


def validate_bytes(raw: bytes, expected_mode: str, source: str = "<memory>") -> Validation:
    return validate_parsed(parse_transcript_bytes(raw, source), expected_mode)


def validate(path: pathlib.Path, expected_mode: str) -> Validation:
    return validate_parsed(parse_transcript(path), expected_mode)


def expected_canonical() -> str:
    lines = [f"SCHEMA|{SCHEMA}"]
    for tx in EXPECTED_TRANSACTIONS:
        arg0_pointer = tx.api in {"realloc", "free"}
        arg0 = str(tx.arg0) if arg0_pointer else str(int(tx.arg0))
        lines.append(
            f"REAL|{tx.identifier}|{tx.identifier}|{tx.api}|{arg0}|{tx.arg1}|{tx.result}|{tx.errno_in}|{tx.errno_out}"
        )
        lines.append(
            f"CALLER|{tx.identifier}|{tx.phase}|{tx.api}|{arg0}|{tx.arg1}|{tx.result}|{tx.errno_in}|{tx.errno_out}"
        )
        for evaluator in sorted(tx.evaluators, key=lambda item: item.identifier):
            lines.append(
                f"EVAL|{tx.identifier}|{tx.phase}|{evaluator.identifier}|{evaluator.operand}|"
                f"{evaluator.kind}|{evaluator.value}"
            )
    lines.append("SUMMARY|22|30|0|0|0|0")
    return "\n".join(lines) + "\n"


def expected_mutation_physical(name: str) -> str:
    """Generate one exact mutation transcript without consuming fixture output."""

    if name not in DECLARED_MUTATIONS:
        raise ValueError(f"unknown mutation {name!r}")
    lines = [f"META|schema|{SCHEMA}", "META|mode|A01"]
    lines.extend(f"TOKEN|{token}|{token}" for token in POINTER_TOKENS)
    real_sequence = 0
    for transaction in EXPECTED_TRANSACTIONS:
        evaluator_lines: list[str] = []
        for evaluator in transaction.evaluators:
            value = evaluator.value
            if name == "result" and transaction.identifier == 10 and evaluator.operand == "pointer":
                value = "NULL"
            evaluator_lines.append(
                f"EVAL|{transaction.identifier}|{transaction.phase}|{evaluator.operand}|"
                f"{evaluator.identifier}|{evaluator.kind}|{value}"
            )
        lines.extend(sorted(evaluator_lines))

        suppress_real = name == "suppress-free-null" and transaction.identifier == 15
        if not suppress_real:
            real_sequence += 1
            real_arg0 = transaction.arg0
            if name == "operand" and transaction.identifier == 2:
                real_arg0 = 3
            if name == "result" and transaction.identifier == 10:
                real_arg0 = "NULL"
            lines.append(
                f"REAL|{real_sequence}|{transaction.identifier}|{transaction.api}|{real_arg0}|"
                f"{transaction.arg1}|{transaction.result}|{transaction.errno_in}|{transaction.errno_out}"
            )
            if name == "duplicate-real" and transaction.identifier == 1:
                real_sequence += 1
                lines.append(
                    f"REAL|{real_sequence}|{transaction.identifier}|{transaction.api}|{real_arg0}|"
                    f"{transaction.arg1}|{transaction.result}|{transaction.errno_in}|{transaction.errno_out}"
                )

        caller_arg0 = transaction.arg0
        caller_result = transaction.result
        caller_errno_out = transaction.errno_out
        if name == "errno" and transaction.identifier == 1:
            caller_errno_out = errno.EIO
        if name == "result" and transaction.identifier == 1:
            caller_result = "NULL"
        if name == "result" and transaction.identifier == 10:
            caller_arg0 = "NULL"
        lines.append(
            f"CALLER|{transaction.identifier}|{transaction.phase}|{transaction.api}|{caller_arg0}|"
            f"{transaction.arg1}|{caller_result}|{transaction.errno_in}|{caller_errno_out}"
        )

    real_calls = 23 if name == "duplicate-real" else 21 if name == "suppress-free-null" else 22
    context_probes = 1 if name == "context" else 0
    lines.append(f"SUMMARY|{real_calls}|30|0|{context_probes}|0|0")
    return "\n".join(lines) + "\n"


EXPECTED_MUTATION_PHYSICAL_SHA256: dict[str, str] = {
    "duplicate-real": "6a6fcfbe3e787ad29dfe3630395ab7f53317c0a70a8f6b29e5358c6fa29755a3",
    "operand": "1f2df547757c5a20dc91bc93b89fa6593d49529343b6ee4aa9a65b39c338ad51",
    "errno": "9d20e2433e874c87a32c94a79234a0984fb07ebdeffc40de622f0f5ecfad02a8",
    "result": "35286b8fbf1e24c1250c21ae4b46e075171fd5b437052e308a619ad0c7295ea6",
    "suppress-free-null": "e24d7ed7ea7243482a578e40c2958ea9c3d981ce18077b15bac717e6cd0f085e",
    "context": "2ef65395129983024ea4f995b6aa7621e466c66548af53699b7338f9b08170c9",
}


def verify_frozen_mutation_expected(name: str) -> str:
    if tuple(EXPECTED_MUTATION_PHYSICAL_SHA256) != DECLARED_MUTATIONS:
        raise RuntimeError("frozen mutation digest catalog does not match the declared mutation domain")
    if tuple(MUTATION_EXPECTATIONS) != DECLARED_MUTATIONS or tuple(MUTATION_PROBLEM_CODES) != DECLARED_MUTATIONS:
        raise RuntimeError("mutation diagnostic catalogs do not match the declared mutation domain")
    expected = expected_mutation_physical(name)
    generated_digest = mutation_physical_sha256(name, "A01", expected)
    frozen_digest = EXPECTED_MUTATION_PHYSICAL_SHA256[name]
    if frozen_digest == "TO_BE_FROZEN" or generated_digest != frozen_digest:
        raise RuntimeError(
            f"independent {name} physical template hash changed: {generated_digest}, frozen {frozen_digest}"
        )
    return frozen_digest


def print_problems(label: str, problems: Iterable[Problem]) -> None:
    for problem in problems:
        print(f"{label}: {problem.code}: {problem.detail}", file=sys.stderr)


def verify_frozen_expected() -> str:
    canonical = expected_canonical()
    digest = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    if EXPECTED_CANONICAL_SHA256 == "TO_BE_FROZEN" or digest != EXPECTED_CANONICAL_SHA256:
        raise RuntimeError(
            f"independent expected table hash changed: {digest}, frozen {EXPECTED_CANONICAL_SHA256}"
        )
    return canonical


MUTATION_EXPECTATIONS: dict[str, tuple[str, str]] = {
    "duplicate-real": ("REAL_MULTIPLICITY", "transaction 1: 2 REAL records"),
    "operand": ("REAL_OPERAND", "transaction 2: operands 3, 8"),
    "errno": ("CALLER_ERRNO", "transaction 1: errno 1001->5"),
    "result": ("CALLER_RESULT", "transaction 1: result 0"),
    "suppress-free-null": ("REAL_MULTIPLICITY", "transaction 15: 0 REAL records"),
    "context": ("CONTEXT_PROBE", "SUMMARY field 4 is 1, expected 0"),
}

MUTATION_PROBLEM_CODES: dict[str, tuple[str, ...]] = {
    "duplicate-real": (
        "REAL_MULTIPLICITY",
        *("REAL_SEQUENCE" for _ in range(21)),
        "REAL_COUNT",
        "GLOBAL_ORDER",
        "CANONICAL_ABSOLUTE",
    ),
    "operand": ("REAL_OPERAND", "CANONICAL_ABSOLUTE"),
    "errno": ("CALLER_ERRNO", "CANONICAL_ABSOLUTE"),
    "result": (
        "CALLER_RESULT",
        "EVALUATOR_SET",
        "REAL_OPERAND",
        "CALLER_OPERAND",
        "CANONICAL_ABSOLUTE",
    ),
    "suppress-free-null": (
        "REAL_MULTIPLICITY",
        *("REAL_SEQUENCE" for _ in range(7)),
        "REAL_TRANSACTION_SET",
        "REAL_COUNT",
        "GLOBAL_ORDER",
        "CANONICAL_ABSOLUTE",
    ),
    "context": ("CONTEXT_PROBE", "CANONICAL_ABSOLUTE"),
}

MUTATION_GLOBAL_ORDER_CAUSES = {
    "duplicate-real": "expected transaction 1 CALLER",
    "suppress-free-null": "expected transaction 15 REAL",
}


def check_absolute(path: pathlib.Path, mode: str) -> Validation:
    expected = verify_frozen_expected()
    result = validate(path, mode)
    problems = list(result.problems)
    if result.canonical != expected:
        problems.append(Problem("CANONICAL_ABSOLUTE", "canonical transcript differs from independent literal table"))
    return dataclasses.replace(result, problems=tuple(problems))


def check_absolute_bytes(raw: bytes, mode: str, source: str = "<memory>") -> Validation:
    expected = verify_frozen_expected()
    result = validate_bytes(raw, mode, source)
    problems = list(result.problems)
    if result.canonical != expected:
        problems.append(Problem("CANONICAL_ABSOLUTE", "canonical transcript differs from independent literal table"))
    return dataclasses.replace(result, problems=tuple(problems))


def mutation_first_defect_matches(name: str, result: Validation) -> bool:
    if name not in DECLARED_MUTATIONS or not result.problems:
        return False
    if result.physical_mode != "A01" or result.physical_problems or not result.physical_normalized:
        return False
    expected_physical_digest = verify_frozen_mutation_expected(name)
    actual_physical_digest = mutation_physical_sha256(
        name, result.physical_mode, result.physical_normalized
    )
    if actual_physical_digest != expected_physical_digest:
        return False
    code, detail = MUTATION_EXPECTATIONS[name]
    first = result.problems[0]
    if first != Problem(code, detail):
        return False
    if tuple(problem.code for problem in result.problems) != MUTATION_PROBLEM_CODES[name]:
        return False
    expected_global_cause = MUTATION_GLOBAL_ORDER_CAUSES.get(name)
    if expected_global_cause is not None:
        global_problem = next(problem for problem in result.problems if problem.code == "GLOBAL_ORDER")
        if expected_global_cause not in global_problem.detail:
            return False
    return True


def _report_pair(a00: Validation, a01: Validation) -> int:
    if a00.problems:
        print_problems("A00", a00.problems)
    if a01.problems:
        print_problems("A01", a01.problems)
    if a00.problems or a01.problems:
        return 1
    if a00.canonical != a01.canonical:
        print("PAIR: DIFFERENTIAL: A00 and A01 canonical transcripts differ", file=sys.stderr)
        return 1
    digest = hashlib.sha256(a00.canonical.encode("ascii")).hexdigest()
    print(f"PASS pair: 22 transactions, 30 one-time evaluators, canonical_sha256={digest}")
    return 0


def command_pair(a00_path: pathlib.Path, a01_path: pathlib.Path) -> int:
    return _report_pair(check_absolute(a00_path, "A00"), check_absolute(a01_path, "A01"))


def _report_single(mode: str, result: Validation) -> int:
    if result.problems:
        print_problems(mode, result.problems)
        return 1
    digest = hashlib.sha256(result.canonical.encode("ascii")).hexdigest()
    print(f"PASS {mode}: canonical_sha256={digest}")
    return 0


def command_single(mode: str, path: pathlib.Path) -> int:
    return _report_single(mode, check_absolute(path, mode))


def _report_mutation(name: str, result: Validation) -> int:
    if not result.problems:
        print(f"MUTATION {name}: transcript unexpectedly passed the absolute oracle", file=sys.stderr)
        return 1
    code, detail = MUTATION_EXPECTATIONS[name]
    first = result.problems[0]
    if not mutation_first_defect_matches(name, result):
        actual_physical_digest = mutation_physical_sha256(
            name, result.physical_mode, result.physical_normalized
        )
        expected_physical_digest = verify_frozen_mutation_expected(name)
        if result.physical_problems:
            print_problems(f"MUTATION {name} PHYSICAL", result.physical_problems)
        if actual_physical_digest != expected_physical_digest:
            print(
                f"MUTATION {name}: normalized physical transcript differs; "
                f"expected sha256={expected_physical_digest}, found sha256={actual_physical_digest}",
                file=sys.stderr,
            )
        elif first != Problem(code, detail):
            print(
                f"MUTATION {name}: first defect must be exactly {code!r}: {detail!r}, "
                f"found {first.code!r}: {first.detail}",
                file=sys.stderr,
            )
        else:
            actual_codes = tuple(problem.code for problem in result.problems)
            print(
                f"MUTATION {name}: downstream diagnostic cascade differs; "
                f"expected {MUTATION_PROBLEM_CODES[name]!r}, found {actual_codes!r}",
                file=sys.stderr,
            )
        print_problems(f"MUTATION {name} SECONDARY", result.problems)
        return 1
    physical_digest = mutation_physical_sha256(name, result.physical_mode, result.physical_normalized)
    print(
        f"PASS mutation control {name}: first defect detected by {code}, "
        f"physical_sha256={physical_digest}"
    )
    return 0


def command_mutation(name: str, path: pathlib.Path) -> int:
    return _report_mutation(name, check_absolute(path, "A01"))


def run_child_bounded(executable: pathlib.Path, timeout_seconds: float) -> BoundedProcessResult:
    result = run_process_bounded(
        [str(executable.resolve())],
        timeout_seconds=timeout_seconds,
        max_stdout_bytes=MAX_RAW_BYTES,
        max_stderr_bytes=MAX_CHILD_STDERR_BYTES,
        remove_environment=loader_environment_variables(),
        environment_updates={"LANG": "C", "LC_ALL": "C"},
    )
    failure = result.failure
    if failure is None and result.returncode != 0:
        failure = f"child exited with status {result.returncode}"
    if failure is None and result.stderr:
        failure = f"child wrote {len(result.stderr)} bytes to stderr"
    return dataclasses.replace(result, failure=failure)


def persist_raw(path: pathlib.Path, raw: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("zero-length write while persisting raw transcript")
            offset += written
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    os.close(descriptor)


def report_child_failure(label: str, result: BoundedProcessResult) -> None:
    print(f"{label}: CHILD: {result.failure}", file=sys.stderr)
    if result.stderr:
        print(format_byte_evidence(f"{label}: CHILD_STDERR", result.stderr), file=sys.stderr)


def raw_path(directory: pathlib.Path, name: str, requested: pathlib.Path | None, raw: bytes) -> pathlib.Path:
    path = requested if requested is not None else directory / name
    persist_raw(path, raw)
    return path


def command_launch_pair(
    a00_executable: pathlib.Path,
    a01_executable: pathlib.Path,
    timeout_seconds: float,
    a00_raw: pathlib.Path | None,
    a01_raw: pathlib.Path | None,
) -> int:
    a00 = run_child_bounded(a00_executable, timeout_seconds)
    a01 = run_child_bounded(a01_executable, timeout_seconds)
    with tempfile.TemporaryDirectory(prefix="oai-memprof-r0-launch-") as directory_text:
        directory = pathlib.Path(directory_text)
        try:
            raw_path(directory, "a00.raw", a00_raw, a00.stdout)
            raw_path(directory, "a01.raw", a01_raw, a01.stdout)
        except OSError as exc:
            print(f"LAUNCH PAIR: could not persist raw transcript: {exc}", file=sys.stderr)
            return 1
        if a00.failure is not None:
            report_child_failure("A00", a00)
        if a01.failure is not None:
            report_child_failure("A01", a01)
        if a00.failure is not None or a01.failure is not None:
            return 1
        return _report_pair(
            check_absolute_bytes(a00.stdout, "A00", str(a00_executable)),
            check_absolute_bytes(a01.stdout, "A01", str(a01_executable)),
        )


def command_launch_single(
    mode: str, executable: pathlib.Path, timeout_seconds: float, raw_output: pathlib.Path | None
) -> int:
    result = run_child_bounded(executable, timeout_seconds)
    with tempfile.TemporaryDirectory(prefix="oai-memprof-r0-launch-") as directory_text:
        try:
            raw_path(pathlib.Path(directory_text), f"{mode.lower()}.raw", raw_output, result.stdout)
        except OSError as exc:
            print(f"LAUNCH {mode}: could not persist raw transcript: {exc}", file=sys.stderr)
            return 1
        if result.failure is not None:
            report_child_failure(mode, result)
            return 1
        return _report_single(mode, check_absolute_bytes(result.stdout, mode, str(executable)))


def command_launch_mutation(
    name: str, executable: pathlib.Path, timeout_seconds: float, raw_output: pathlib.Path | None
) -> int:
    result = run_child_bounded(executable, timeout_seconds)
    with tempfile.TemporaryDirectory(prefix="oai-memprof-r0-launch-") as directory_text:
        try:
            raw_path(pathlib.Path(directory_text), "mutant.raw", raw_output, result.stdout)
        except OSError as exc:
            print(f"LAUNCH MUTATION {name}: could not persist raw transcript: {exc}", file=sys.stderr)
            return 1
        if result.failure is not None:
            report_child_failure(f"MUTATION {name}", result)
            return 1
        return _report_mutation(name, check_absolute_bytes(result.stdout, "A01", str(executable)))


def command_bounds_self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="oai-memprof-r0-bounds-") as directory:
        oversized = pathlib.Path(directory) / "oversized.raw"
        with oversized.open("wb") as stream:
            stream.truncate(MAX_RAW_BYTES + 1)
        result = validate(oversized, "A00")
    if not any(problem.code == "RAW_SIZE" for problem in result.problems):
        print("BOUNDS: oversized input did not trigger RAW_SIZE", file=sys.stderr)
        return 1
    print(f"PASS bounds: rejected input larger than {MAX_RAW_BYTES} bytes")
    return 0


def timeout_argument(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not 0.1 <= value <= 60.0:
        raise argparse.ArgumentTypeError("timeout must be between 0.1 and 60 seconds")
    return value


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pair = subparsers.add_parser("pair", help="validate A00 and A01 absolutely, then compare them")
    pair.add_argument("a00", type=pathlib.Path)
    pair.add_argument("a01", type=pathlib.Path)

    single = subparsers.add_parser("single", help="validate one absolute transcript")
    single.add_argument("mode", choices=("A00", "A01"))
    single.add_argument("path", type=pathlib.Path)

    mutation = subparsers.add_parser("mutation", help="require a named mutant to be rejected for its causal defect")
    mutation.add_argument("name", choices=DECLARED_MUTATIONS)
    mutation.add_argument("path", type=pathlib.Path)

    launch_pair = subparsers.add_parser("launch-pair", help="run bounded A00/A01 fixtures, then validate the pair")
    launch_pair.add_argument("a00_executable", type=pathlib.Path)
    launch_pair.add_argument("a01_executable", type=pathlib.Path)
    launch_pair.add_argument("--timeout-seconds", type=timeout_argument, default=DEFAULT_CHILD_TIMEOUT_SECONDS)
    launch_pair.add_argument("--a00-raw", type=pathlib.Path)
    launch_pair.add_argument("--a01-raw", type=pathlib.Path)

    launch_single = subparsers.add_parser("launch-single", help="run one bounded fixture, then validate it absolutely")
    launch_single.add_argument("mode", choices=("A00", "A01"))
    launch_single.add_argument("executable", type=pathlib.Path)
    launch_single.add_argument("--timeout-seconds", type=timeout_argument, default=DEFAULT_CHILD_TIMEOUT_SECONDS)
    launch_single.add_argument("--raw-output", type=pathlib.Path)

    launch_mutation = subparsers.add_parser(
        "launch-mutation", help="run one bounded mutant fixture, then require its causal rejection"
    )
    launch_mutation.add_argument("name", choices=DECLARED_MUTATIONS)
    launch_mutation.add_argument("executable", type=pathlib.Path)
    launch_mutation.add_argument("--timeout-seconds", type=timeout_argument, default=DEFAULT_CHILD_TIMEOUT_SECONDS)
    launch_mutation.add_argument("--raw-output", type=pathlib.Path)

    subparsers.add_parser("dump-expected", help="write the independent canonical literal table to stdout")
    subparsers.add_parser("bounds-self-test", help="prove that an oversized raw input is rejected before parsing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        if args.command == "pair":
            return command_pair(args.a00, args.a01)
        if args.command == "single":
            return command_single(args.mode, args.path)
        if args.command == "mutation":
            return command_mutation(args.name, args.path)
        if args.command == "launch-pair":
            return command_launch_pair(
                args.a00_executable,
                args.a01_executable,
                args.timeout_seconds,
                args.a00_raw,
                args.a01_raw,
            )
        if args.command == "launch-single":
            return command_launch_single(args.mode, args.executable, args.timeout_seconds, args.raw_output)
        if args.command == "launch-mutation":
            return command_launch_mutation(args.name, args.executable, args.timeout_seconds, args.raw_output)
        if args.command == "bounds-self-test":
            return command_bounds_self_test()
        sys.stdout.write(verify_frozen_expected())
        return 0
    except RuntimeError as exc:
        print(f"ORACLE: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
