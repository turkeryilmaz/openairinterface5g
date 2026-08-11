#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Adversarial self-tests for the R0 transcript and process harness."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import os
import signal
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from typing import Callable
from unittest import mock

import r0_harness_common as harness
import run_r0_actual_differential as actual_launcher
import validate_r0_scripted_oracle as oracle
from r0_harness_common import (
    BoundedFileError,
    format_byte_evidence,
    known_loader_environment_variables,
    loader_environment_variables,
    read_regular_file_bounded,
    run_process_bounded,
)


TOKEN_ADDRESSES = {
    "NULL": 0,
    "P1": 0x1000,
    "P2": 0x2000,
    "P3": 0x3000,
    "P4": 0x4000,
    "P5": 0x5000,
    "P6": 0x6000,
    "P7": 0x7000,
    "P8": 0x8000,
}


def hex_field(value: int | str) -> str:
    numeric = TOKEN_ADDRESSES[value] if isinstance(value, str) else value
    return f"0x{numeric:x}"


def transcript_parts(mode: str = "A00") -> tuple[list[str], list[list[str]], str]:
    header = [f"META|schema|{oracle.SCHEMA}", f"META|mode|{mode}"]
    header.extend(f"TOKEN|{token}|{hex_field(token)}" for token in oracle.POINTER_TOKENS)
    blocks: list[list[str]] = []
    for transaction in oracle.EXPECTED_TRANSACTIONS:
        block = [
            "EVAL|{}|{}|{}|{}|{}|{}".format(
                transaction.identifier,
                transaction.phase,
                evaluator.operand,
                evaluator.identifier,
                evaluator.kind,
                hex_field(evaluator.value),
            )
            for evaluator in transaction.evaluators
        ]
        block.append(
            "REAL|{}|{}|{}|{}|{}|{}|{}|{}".format(
                transaction.identifier,
                transaction.identifier,
                transaction.api,
                hex_field(transaction.arg0),
                hex_field(transaction.arg1),
                hex_field(transaction.result),
                transaction.errno_in,
                transaction.errno_out,
            )
        )
        block.append(
            "CALLER|{}|{}|{}|{}|{}|{}|{}|{}".format(
                transaction.identifier,
                transaction.phase,
                transaction.api,
                hex_field(transaction.arg0),
                hex_field(transaction.arg1),
                hex_field(transaction.result),
                transaction.errno_in,
                transaction.errno_out,
            )
        )
        blocks.append(block)
    return header, blocks, "SUMMARY|22|30|0|0|0|0"


def encode_parts(header: list[str], blocks: list[list[str]], summary: str) -> bytes:
    lines = header + [line for block in blocks for line in block] + [summary]
    return ("\n".join(lines) + "\n").encode("ascii")


def valid_transcript(mode: str = "A00") -> bytes:
    return encode_parts(*transcript_parts(mode))


def replace_line(raw: bytes, prefix: bytes, transform: Callable[[bytes], bytes]) -> bytes:
    lines = raw.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = transform(line)
            return b"\n".join(lines) + b"\n"
    raise AssertionError(f"missing line prefix {prefix!r}")


def problem_codes(result: oracle.Validation) -> list[str]:
    return [problem.code for problem in result.problems]


def process_state_starttime(pid: int) -> tuple[str, int] | None:
    try:
        data = Path(f"/proc/{pid}/stat").read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = data.rsplit(b") ", 1)[1].split()
    return fields[0].decode("ascii"), int(fields[19], 10)


def assert_process_quiescent(testcase: unittest.TestCase, pid: int, starttime: int) -> None:
    deadline = time.monotonic() + 1.0
    current: tuple[str, int] | None = None
    while time.monotonic() < deadline:
        current = process_state_starttime(pid)
        if current is None or current[1] != starttime or current[0] == "Z":
            return
        time.sleep(0.01)

    pidfd: int | None = None
    try:
        pidfd = os.pidfd_open(pid, 0)
        current = process_state_starttime(pid)
        if current is not None and current[1] == starttime and current[0] != "Z":
            signal.pidfd_send_signal(pidfd, signal.SIGKILL, None, 0)
    except (ProcessLookupError, FileNotFoundError):
        return
    finally:
        if pidfd is not None:
            os.close(pidfd)
    testcase.fail(f"process {pid} starttime {starttime} remained live with state {current!r}")


def transaction_bounds(lines: list[bytes], identifier: int) -> tuple[int, int]:
    indexes: list[int] = []
    for index, line in enumerate(lines):
        fields = line.split(b"|")
        if fields[0] == b"EVAL" and int(fields[1]) == identifier:
            indexes.append(index)
        elif fields[0] == b"REAL" and int(fields[2]) == identifier:
            indexes.append(index)
        elif fields[0] == b"CALLER" and int(fields[1]) == identifier:
            indexes.append(index)
    if not indexes:
        raise AssertionError(f"transaction {identifier} not found")
    return min(indexes), max(indexes) + 1


def swap_adjacent_transactions(raw: bytes, first: int, second: int) -> bytes:
    lines = raw.splitlines()
    first_begin, first_end = transaction_bounds(lines, first)
    second_begin, second_end = transaction_bounds(lines, second)
    if first_end != second_begin:
        raise AssertionError("test helper requires adjacent transaction blocks")
    lines[first_begin:second_end] = lines[second_begin:second_end] + lines[first_begin:first_end]
    return b"\n".join(lines) + b"\n"


class TranscriptGrammarTests(unittest.TestCase):
    def assert_rejected(self, raw: bytes, code: str) -> oracle.Validation:
        result = oracle.check_absolute_bytes(raw, "A00", self.id())
        self.assertIn(code, problem_codes(result), result.problems)
        return result

    def test_valid_literal_preserves_frozen_hash(self) -> None:
        result = oracle.check_absolute_bytes(valid_transcript(), "A00", self.id())
        self.assertEqual(result.problems, ())
        self.assertEqual(hashlib.sha256(result.canonical.encode("ascii")).hexdigest(), oracle.EXPECTED_CANONICAL_SHA256)

    def test_reversed_multi_operand_evaluators_are_valid(self) -> None:
        header, blocks, summary = transcript_parts()
        for block, transaction in zip(blocks, oracle.EXPECTED_TRANSACTIONS):
            evaluator_count = len(transaction.evaluators)
            block[:evaluator_count] = reversed(block[:evaluator_count])
        result = oracle.check_absolute_bytes(encode_parts(header, blocks, summary), "A00", self.id())
        self.assertEqual(result.problems, ())

    def test_summary_must_be_terminal(self) -> None:
        header, blocks, summary = transcript_parts()
        raw = (summary + "\n").encode("ascii") + encode_parts(header, blocks, summary).removesuffix(
            (summary + "\n").encode("ascii")
        )
        self.assert_rejected(raw, "GLOBAL_ORDER")

    def test_transactions_must_be_globally_ordered(self) -> None:
        header, blocks, summary = transcript_parts()
        blocks[0], blocks[1] = blocks[1], blocks[0]
        self.assert_rejected(encode_parts(header, blocks, summary), "GLOBAL_ORDER")

    def test_headers_must_precede_tokens(self) -> None:
        header, blocks, summary = transcript_parts()
        mode = header.pop(1)
        header.insert(3, mode)
        self.assert_rejected(encode_parts(header, blocks, summary), "GLOBAL_ORDER")

    def test_carriage_return_is_not_a_record_separator(self) -> None:
        self.assert_rejected(valid_transcript().replace(b"\n", b"\r", 1), "RAW_CHARACTER")

    def test_unknown_record_is_rejected(self) -> None:
        self.assert_rejected(b"BOGUS\n" + valid_transcript(), "RECORD_KIND")

    def test_malformed_evaluator_identifier_is_structured(self) -> None:
        raw = valid_transcript().replace(b"EVAL|1|CTOR|size|1|SIZE|", b"EVAL|1|CTOR|size|x|SIZE|", 1)
        result = self.assert_rejected(raw, "INTEGER")
        self.assertEqual(result.problems[0].code, "INTEGER")

    def test_non_ascii_is_rejected(self) -> None:
        self.assert_rejected(b"\xff" + valid_transcript(), "RAW_CHARACTER")

    def test_unterminated_input_is_rejected(self) -> None:
        self.assert_rejected(valid_transcript()[:-1], "TERMINATION")

    def test_oversized_line_is_rejected(self) -> None:
        self.assert_rejected(b"X" * (oracle.MAX_RAW_LINE_BYTES + 1) + b"\n", "LINE_SIZE")

    def test_excess_record_count_is_rejected(self) -> None:
        self.assert_rejected(b"X\n" * (oracle.MAX_RAW_RECORDS + 1), "RECORD_LIMIT")

    def test_noncanonical_numbers_are_rejected(self) -> None:
        raw = valid_transcript().replace(b"REAL|1|1|malloc|", b"REAL|01|1|malloc|", 1)
        self.assert_rejected(raw, "INTEGER")

    def test_noncanonical_hex_is_rejected(self) -> None:
        raw = valid_transcript().replace(b"TOKEN|P1|0x1000", b"TOKEN|P1|0X1000", 1)
        self.assert_rejected(raw, "INTEGER")

    def test_typed_integer_ranges_are_enforced(self) -> None:
        cases = {
            "uint32-transaction": valid_transcript().replace(
                b"EVAL|1|CTOR|size|1|", b"EVAL|4294967296|CTOR|size|1|", 1
            ),
            "uint32-evaluator": valid_transcript().replace(
                b"EVAL|1|CTOR|size|1|", b"EVAL|1|CTOR|size|4294967296|", 1
            ),
            "uint64-hex": valid_transcript().replace(
                b"TOKEN|P1|0x1000", b"TOKEN|P1|0x10000000000000000", 1
            ),
            "int32-errno": valid_transcript().replace(b"|1001|1001\n", b"|2147483648|1001\n", 1),
            "uint32-summary": valid_transcript().replace(
                b"SUMMARY|22|30|0|0|0|0", b"SUMMARY|4294967296|30|0|0|0|0", 1
            ),
        }
        for label, raw in cases.items():
            with self.subTest(label=label):
                result = self.assert_rejected(raw, "INTEGER")
                self.assertEqual(result.problems[0].code, "INTEGER")

    def test_exact_field_count_is_enforced(self) -> None:
        raw = valid_transcript().replace(b"META|schema|", b"META|extra|schema|", 1)
        result = self.assert_rejected(raw, "FIELD_COUNT")
        self.assertEqual(result.problems[0].code, "FIELD_COUNT")

    def test_fifo_input_is_rejected_without_waiting_for_a_writer(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-memprof-r0-fifo-") as directory:
            fifo = Path(directory) / "input.fifo"
            os.mkfifo(fifo)
            started = time.monotonic()
            parsed = oracle.parse_transcript(fifo)
            elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.5)
        self.assertEqual(parsed.problems[0].code, "INPUT_TYPE")


class MutationCausalityTests(unittest.TestCase):
    def mutation_transcript(self, name: str) -> bytes:
        raw = valid_transcript("A01")
        if name == "duplicate-real":
            lines = raw.splitlines()
            index = next(i for i, line in enumerate(lines) if line.startswith(b"REAL|1|1|"))
            fields = lines[index].split(b"|")
            fields[1] = b"2"
            lines.insert(index + 1, b"|".join(fields))
            for line_index, line in enumerate(lines):
                fields = line.split(b"|")
                if fields[0] == b"REAL" and int(fields[2]) >= 2:
                    fields[1] = str(int(fields[1]) + 1).encode("ascii")
                    lines[line_index] = b"|".join(fields)
            lines[-1] = b"SUMMARY|23|30|0|0|0|0"
            return b"\n".join(lines) + b"\n"
        if name == "operand":
            return replace_line(
                raw,
                b"REAL|2|2|calloc|",
                lambda line: b"|".join(line.split(b"|")[:4] + [b"0x3"] + line.split(b"|")[5:]),
            )
        if name == "errno":
            return replace_line(raw, b"CALLER|1|CTOR|malloc|", lambda line: b"|".join(line.split(b"|")[:-1] + [b"5"]))
        if name == "result":
            raw = replace_line(
                raw,
                b"CALLER|1|CTOR|malloc|",
                lambda line: b"|".join(line.split(b"|")[:6] + [b"0x0"] + line.split(b"|")[7:]),
            )
            raw = replace_line(
                raw,
                b"EVAL|10|MAIN|pointer|",
                lambda line: b"|".join(line.split(b"|")[:-1] + [b"0x0"]),
            )
            raw = replace_line(
                raw,
                b"REAL|10|10|realloc|",
                lambda line: b"|".join(line.split(b"|")[:4] + [b"0x0"] + line.split(b"|")[5:]),
            )
            return replace_line(
                raw,
                b"CALLER|10|MAIN|realloc|",
                lambda line: b"|".join(line.split(b"|")[:4] + [b"0x0"] + line.split(b"|")[5:]),
            )
        if name == "suppress-free-null":
            lines = [line for line in raw.splitlines() if not line.startswith(b"REAL|15|15|")]
            for line_index, line in enumerate(lines):
                fields = line.split(b"|")
                if fields[0] == b"REAL" and int(fields[2]) >= 16:
                    fields[1] = str(int(fields[1]) - 1).encode("ascii")
                    lines[line_index] = b"|".join(fields)
            lines[-1] = b"SUMMARY|21|30|0|0|0|0"
            return b"\n".join(lines) + b"\n"
        if name == "context":
            return raw.replace(b"SUMMARY|22|30|0|0|0|0", b"SUMMARY|22|30|0|1|0|0", 1)
        raise AssertionError(name)

    def reverse_contiguous_evaluators(self, raw: bytes) -> bytes:
        lines = raw.splitlines()
        cursor = 0
        while cursor < len(lines):
            fields = lines[cursor].split(b"|")
            if fields[0] != b"EVAL":
                cursor += 1
                continue
            transaction = fields[1]
            end = cursor + 1
            while end < len(lines):
                candidate = lines[end].split(b"|")
                if candidate[0] != b"EVAL" or candidate[1] != transaction:
                    break
                end += 1
            lines[cursor:end] = reversed(lines[cursor:end])
            cursor = end
        return b"\n".join(lines) + b"\n"

    def test_each_mutant_has_its_declared_first_defect(self) -> None:
        for name in oracle.MUTATION_EXPECTATIONS:
            with self.subTest(name=name):
                result = oracle.check_absolute_bytes(self.mutation_transcript(name), "A01", name)
                self.assertTrue(oracle.mutation_first_defect_matches(name, result), result.problems)

    def test_frozen_physical_templates_match_independent_mutants(self) -> None:
        self.assertEqual(tuple(oracle.EXPECTED_MUTATION_PHYSICAL_SHA256), oracle.DECLARED_MUTATIONS)
        for name in oracle.DECLARED_MUTATIONS:
            with self.subTest(name=name):
                frozen = oracle.verify_frozen_mutation_expected(name)
                result = oracle.check_absolute_bytes(self.mutation_transcript(name), "A01", name)
                actual = oracle.mutation_physical_sha256(name, result.physical_mode, result.physical_normalized)
                self.assertEqual(actual, frozen)

    def test_physical_hash_domain_includes_mode_and_mutation_name(self) -> None:
        normalized = oracle.expected_mutation_physical("operand")
        frozen = oracle.EXPECTED_MUTATION_PHYSICAL_SHA256["operand"]
        self.assertEqual(oracle.mutation_physical_sha256("operand", "A01", normalized), frozen)
        self.assertNotEqual(oracle.mutation_physical_sha256("operand", "A00", normalized), frozen)
        self.assertNotEqual(oracle.mutation_physical_sha256("errno", "A01", normalized), frozen)

    def test_only_contiguous_same_transaction_evaluator_order_is_normalized(self) -> None:
        for name in oracle.DECLARED_MUTATIONS:
            with self.subTest(name=name):
                raw = self.reverse_contiguous_evaluators(self.mutation_transcript(name))
                result = oracle.check_absolute_bytes(raw, "A01", name)
                self.assertTrue(oracle.mutation_first_defect_matches(name, result), result.problems)

    def test_later_transaction_swaps_cannot_preserve_mutation_identity(self) -> None:
        cases = (
            ("duplicate-real", 2, 3),
            ("suppress-free-null", 16, 17),
        )
        for name, first, second in cases:
            with self.subTest(name=name):
                raw = swap_adjacent_transactions(self.mutation_transcript(name), first, second)
                result = oracle.check_absolute_bytes(raw, "A01", name)
                self.assertEqual(result.problems[0], oracle.Problem(*oracle.MUTATION_EXPECTATIONS[name]))
                self.assertFalse(oracle.mutation_first_defect_matches(name, result))

    def test_extra_valid_record_cannot_preserve_mutation_identity(self) -> None:
        raw = self.mutation_transcript("duplicate-real")
        lines = raw.splitlines()
        real22 = next(line for line in lines if line.startswith(b"REAL|23|22|"))
        caller22 = next(index for index, line in enumerate(lines) if line.startswith(b"CALLER|22|"))
        lines.insert(caller22, real22)
        result = oracle.check_absolute_bytes(b"\n".join(lines) + b"\n", "A01", self.id())
        self.assertEqual(result.problems[0], oracle.Problem(*oracle.MUTATION_EXPECTATIONS["duplicate-real"]))
        self.assertFalse(oracle.mutation_first_defect_matches("duplicate-real", result))

    def test_wrong_mutation_values_cannot_claim_named_controls(self) -> None:
        cases = {
            "operand": self.mutation_transcript("operand").replace(
                b"REAL|2|2|calloc|0x3|", b"REAL|2|2|calloc|0x4|", 1
            ),
            "errno": self.mutation_transcript("errno").replace(
                b"CALLER|1|CTOR|malloc|0x10|0x0|0x1000|1001|5",
                b"CALLER|1|CTOR|malloc|0x10|0x0|0x1000|1001|6",
                1,
            ),
            "result": self.mutation_transcript("result").replace(
                b"CALLER|1|CTOR|malloc|0x10|0x0|0x0|1001|1001",
                b"CALLER|1|CTOR|malloc|0x10|0x0|0x2000|1001|1001",
                1,
            ),
            "context": self.mutation_transcript("context").replace(
                b"SUMMARY|22|30|0|1|0|0", b"SUMMARY|22|30|0|2|0|0", 1
            ),
        }
        for name, raw in cases.items():
            with self.subTest(name=name):
                result = oracle.check_absolute_bytes(raw, "A01", name)
                self.assertFalse(oracle.mutation_first_defect_matches(name, result))

    def test_suppressing_a_different_free_is_not_the_declared_control(self) -> None:
        lines = [line for line in valid_transcript("A01").splitlines() if not line.startswith(b"REAL|16|16|")]
        for line_index, line in enumerate(lines):
            fields = line.split(b"|")
            if fields[0] == b"REAL" and int(fields[2]) >= 17:
                fields[1] = str(int(fields[1]) - 1).encode("ascii")
                lines[line_index] = b"|".join(fields)
        lines[-1] = b"SUMMARY|21|30|0|0|0|0"
        result = oracle.check_absolute_bytes(b"\n".join(lines) + b"\n", "A01", self.id())
        self.assertEqual(result.problems[0], oracle.Problem("REAL_MULTIPLICITY", "transaction 16: 0 REAL records"))
        self.assertFalse(oracle.mutation_first_defect_matches("suppress-free-null", result))

    def test_prefixed_mutant_cannot_claim_causal_success(self) -> None:
        raw = b"BOGUS\n" + self.mutation_transcript("duplicate-real")
        result = oracle.check_absolute_bytes(raw, "A01", self.id())
        self.assertEqual(result.problems[0].code, "RECORD_KIND")
        self.assertFalse(oracle.mutation_first_defect_matches("duplicate-real", result))

    def test_lifecycle_reordered_mutant_cannot_claim_causal_success(self) -> None:
        lines = self.mutation_transcript("duplicate-real").splitlines()
        summary = lines.pop()
        raw = b"\n".join([summary, *lines]) + b"\n"
        result = oracle.check_absolute_bytes(raw, "A01", self.id())
        self.assertFalse(oracle.mutation_first_defect_matches("duplicate-real", result))


class BoundedFileTests(unittest.TestCase):
    def test_exact_cap_is_valid_and_digest_binds_read_bytes(self) -> None:
        data = bytes(range(128))
        with tempfile.TemporaryDirectory(prefix="oai-memprof-r0-file-") as directory:
            path = Path(directory) / "exact.bin"
            path.write_bytes(data)
            image = read_regular_file_bounded(path, len(data))
        self.assertEqual(image.data, data)
        self.assertEqual(image.size, len(data))
        self.assertGreater(image.inode, 0)
        self.assertEqual(image.sha256, hashlib.sha256(data).hexdigest())

    def test_transcript_path_exposes_same_fd_byte_identity(self) -> None:
        data = valid_transcript()
        with tempfile.TemporaryDirectory(prefix="oai-memprof-r0-file-") as directory:
            path = Path(directory) / "valid.raw"
            path.write_bytes(data)
            parsed = oracle.parse_transcript(path)
        self.assertIsNotNone(parsed.source_image)
        assert parsed.source_image is not None
        self.assertEqual(parsed.source_image.data, data)
        self.assertEqual(parsed.source_image.size, len(data))
        self.assertEqual(parsed.source_image.sha256, hashlib.sha256(data).hexdigest())
        self.assertEqual(oracle.validate_parsed(parsed, "A00").problems, ())

    def test_changed_file_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-memprof-r0-file-") as directory:
            path = Path(directory) / "changing.bin"
            path.write_bytes(b"x" * 32)
            real_fstat = harness.os.fstat
            calls = 0

            def changing_fstat(fd: int):
                nonlocal calls
                value = real_fstat(fd)
                calls += 1
                if calls == 1:
                    return value
                return types.SimpleNamespace(
                    st_mode=value.st_mode,
                    st_dev=value.st_dev,
                    st_ino=value.st_ino,
                    st_size=value.st_size,
                    st_mtime_ns=value.st_mtime_ns + 1,
                    st_ctime_ns=value.st_ctime_ns,
                )

            with mock.patch.object(harness.os, "fstat", side_effect=changing_fstat):
                with self.assertRaises(BoundedFileError) as context:
                    harness.read_regular_file_bounded(path, 32)
        self.assertEqual(context.exception.code, "INPUT_CHANGED")

    def test_cap_plus_one_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-memprof-r0-file-") as directory:
            path = Path(directory) / "large.bin"
            path.write_bytes(b"x" * 129)
            with self.assertRaises(BoundedFileError) as context:
                read_regular_file_bounded(path, 128)
        self.assertEqual(context.exception.code, "RAW_SIZE")

    def test_symlink_input_is_rejected_without_following_it(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-memprof-r0-file-") as directory:
            target = Path(directory) / "target.bin"
            link = Path(directory) / "link.bin"
            target.write_bytes(b"bounded")
            link.symlink_to(target.name)
            with self.assertRaises(BoundedFileError) as context:
                read_regular_file_bounded(link, 128)
        self.assertEqual(context.exception.code, "READ")


class ByteEvidenceTests(unittest.TestCase):
    def assert_round_trip(self, data: bytes) -> None:
        evidence = format_byte_evidence("stderr", data)
        self.assertTrue(all(0x20 <= ord(character) <= 0x7E for character in evidence))
        fields = dict(field.split("=", 1) for field in evidence.split()[1:])
        self.assertEqual(int(fields["bytes"]), len(data))
        self.assertEqual(fields["sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(base64.b64decode(fields["base64"], validate=True), data)

    def test_lossless_non_utf8_escape_and_newline(self) -> None:
        for data in (b"", b"\xff\x80", b"\x1b[31mred\x1b[0m", b"line1\nline2\r\n"):
            with self.subTest(data=data):
                self.assert_round_trip(data)

    def test_evidence_label_cannot_inject_a_control_or_new_record(self) -> None:
        for label in ("", "stderr\nINJECT", "stderr\x1b"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                format_byte_evidence(label, b"payload")

    def test_validator_never_renders_raw_child_controls(self) -> None:
        data = b"\xff\x1b[31m\n"
        result = harness.BoundedProcessResult(("fixture",), 1, 1, b"", data, "fixture failure", False, True)
        diagnostic = io.StringIO()
        with contextlib.redirect_stderr(diagnostic):
            oracle.report_child_failure("A01", result)
        rendered = diagnostic.getvalue()
        self.assertNotIn("\x1b", rendered)
        self.assertNotIn("\ufffd", rendered)
        self.assertIn(format_byte_evidence("A01: CHILD_STDERR", data), rendered)
        self.assertEqual(len(rendered.splitlines()), 2)


class BoundedProcessTests(unittest.TestCase):
    def run_python(self, source: str, *, timeout: float = 1.0, stdout: int = 4096, stderr: int = 4096):
        return run_process_bounded(
            [sys.executable, "-c", source],
            timeout_seconds=timeout,
            max_stdout_bytes=stdout,
            max_stderr_bytes=stderr,
            environment_updates={"LANG": "C", "LC_ALL": "C"},
        )

    def test_normal_eof(self) -> None:
        result = self.run_python("import os; os.write(1, b'normal\\n')")
        self.assertIsNone(result.failure)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"normal\n")
        self.assertEqual(result.stderr, b"")
        self.assertFalse(result.killpg_attempted)
        self.assertTrue(result.cleanup_complete)

    def test_known_loader_catalog_is_frozen_but_ambient_ld_names_are_scrubbed(self) -> None:
        frozen = known_loader_environment_variables()
        self.assertEqual(frozen, tuple(sorted(frozen)))
        self.assertIn("GLIBC_TUNABLES", frozen)
        self.assertIn("LD_BIND_NOT", frozen)
        self.assertIn("LD_POINTER_GUARD", frozen)
        self.assertIn("LD_PREFER_MAP_32BIT_EXEC", frozen)
        self.assertIn("LD_PRELOAD", frozen)
        novel = "LD_R0_ADVERSARIAL_ONLY"
        self.assertNotIn(novel, frozen)
        with mock.patch.dict(os.environ, {novel: "present"}, clear=False):
            self.assertEqual(known_loader_environment_variables(), frozen)
            self.assertIn(novel, loader_environment_variables())

    def test_loader_environment_is_scrubbed_before_exec(self) -> None:
        names = (
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "LD_AUDIT",
            "LD_BIND_NOT",
            "LD_DEBUG_OUTPUT",
            "LD_POINTER_GUARD",
            "LD_PREFER_MAP_32BIT_EXEC",
            "GLIBC_TUNABLES",
        )
        source = "import os; print(','.join(name for name in {} if name in os.environ))".format(repr(names))
        with mock.patch.dict(os.environ, {name: "adversarial" for name in names}, clear=False):
            result = run_process_bounded(
                [sys.executable, "-c", source],
                timeout_seconds=1.0,
                max_stdout_bytes=128,
                max_stderr_bytes=128,
                remove_environment=loader_environment_variables(),
                environment_updates={"LANG": "C", "LC_ALL": "C"},
            )
        self.assertIsNone(result.failure)
        self.assertEqual(result.stdout, b"\n")
        self.assertEqual(result.stderr, b"")

    def test_timeout_kills_process_group(self) -> None:
        result = self.run_python("import time; time.sleep(30)", timeout=0.2)
        self.assertIn("timeout", result.failure or "")
        self.assertTrue(result.killpg_attempted)
        self.assertTrue(result.cleanup_complete)

    def test_stdout_overflow(self) -> None:
        result = self.run_python("import os; os.write(1, b'x' * 1024)", stdout=128)
        self.assertIn("stdout exceeded 128 bytes", result.failure or "")
        self.assertEqual(result.stdout, b"x" * 128)
        self.assertTrue(result.killpg_attempted)
        self.assertTrue(result.cleanup_complete)

    def test_stderr_overflow(self) -> None:
        result = self.run_python("import os; os.write(2, b'x' * 1024)", stderr=128)
        self.assertIn("stderr exceeded 128 bytes", result.failure or "")
        self.assertEqual(result.stderr, b"x" * 128)
        self.assertTrue(result.killpg_attempted)
        self.assertTrue(result.cleanup_complete)

    def test_exact_output_caps_and_non_utf8_bytes_are_preserved(self) -> None:
        stdout_data = b"A" * 128
        stderr_data = b"\xff\x1b\n" + b"B" * 125
        source = f"import os; os.write(1, {stdout_data!r}); os.write(2, {stderr_data!r})"
        result = self.run_python(source, stdout=128, stderr=128)
        self.assertIsNone(result.failure)
        self.assertEqual(result.stdout, stdout_data)
        self.assertEqual(result.stderr, stderr_data)

    def test_output_overflow_kills_alternate_process_group_descendant(self) -> None:
        source = (
            "import os,time\n"
            "pid=os.fork()\n"
            "if pid == 0:\n"
            " os.setpgid(0,0); time.sleep(30); os._exit(0)\n"
            "time.sleep(0.02)\n"
            "fields=open('/proc/'+str(pid)+'/stat','rb').read().rsplit(b') ',1)[1].split()\n"
            "os.write(1, (str(pid)+' '+fields[19].decode('ascii')+'\\n').encode('ascii'))\n"
            "os.write(1, b'x'*1024)\n"
            "time.sleep(30)\n"
        )
        result = self.run_python(source, timeout=1.0, stdout=128)
        self.assertIn("stdout exceeded 128 bytes", result.failure or "")
        self.assertEqual(len(result.stdout), 128)
        self.assertTrue(result.killpg_attempted)
        self.assertTrue(result.cleanup_complete)
        descendant, starttime = (int(field) for field in result.stdout.splitlines()[0].split())
        assert_process_quiescent(self, descendant, starttime)

    def test_exited_leader_with_descendant_holding_pipes_is_killed(self) -> None:
        source = (
            "import os,time\n"
            "pid=os.fork()\n"
            "if pid == 0:\n"
            " time.sleep(30)\n"
            " os._exit(0)\n"
            "fields=open('/proc/'+str(pid)+'/stat','rb').read().rsplit(b') ',1)[1].split()\n"
            "os.write(1, (str(pid)+' '+fields[19].decode('ascii')+'\\n').encode('ascii'))\n"
            "os._exit(0)\n"
        )
        started = time.monotonic()
        result = self.run_python(source, timeout=0.25)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 2.0)
        self.assertIn("unexpected live same-session descendants after leader exit", result.failure or "")
        self.assertTrue(result.killpg_attempted)
        self.assertTrue(result.cleanup_complete)
        descendant, starttime = (int(field) for field in result.stdout.split())
        assert_process_quiescent(self, descendant, starttime)

    def test_exited_leader_with_descendant_closing_pipes_is_still_rejected(self) -> None:
        source = (
            "import os,time\n"
            "pid=os.fork()\n"
            "if pid == 0:\n"
            " os.close(1); os.close(2); time.sleep(30); os._exit(0)\n"
            "fields=open('/proc/'+str(pid)+'/stat','rb').read().rsplit(b') ',1)[1].split()\n"
            "os.write(1, (str(pid)+' '+fields[19].decode('ascii')+'\\n').encode('ascii'))\n"
            "os._exit(0)\n"
        )
        result = self.run_python(source, timeout=0.5)
        self.assertIn("unexpected live same-session descendants after leader exit", result.failure or "")
        self.assertTrue(result.cleanup_complete)
        descendant, starttime = (int(field) for field in result.stdout.split())
        assert_process_quiescent(self, descendant, starttime)

    def test_alternate_process_group_in_same_session_is_pidfd_killed(self) -> None:
        source = (
            "import os,time\n"
            "pid=os.fork()\n"
            "if pid == 0:\n"
            " os.setpgid(0,0); time.sleep(30); os._exit(0)\n"
            "time.sleep(0.02)\n"
            "fields=open('/proc/'+str(pid)+'/stat','rb').read().rsplit(b') ',1)[1].split()\n"
            "os.write(1, (str(pid)+' '+fields[19].decode('ascii')+'\\n').encode('ascii'))\n"
            "os._exit(0)\n"
        )
        result = self.run_python(source, timeout=0.5)
        self.assertIn("unexpected live same-session descendants after leader exit", result.failure or "")
        self.assertTrue(result.cleanup_complete)
        descendant, starttime = (int(field) for field in result.stdout.split())
        assert_process_quiescent(self, descendant, starttime)

    def test_keyboard_interrupt_cleans_before_reraising(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-memprof-r0-cancel-") as directory:
            identity_path = Path(directory) / "identity"
            source = (
                "import os,time,pathlib\n"
                "fields=open('/proc/self/stat','rb').read().rsplit(b') ',1)[1].split()\n"
                f"pathlib.Path({str(identity_path)!r}).write_text(str(os.getpid())+' '+fields[19].decode('ascii'))\n"
                "time.sleep(30)\n"
            )
            original_drain = harness._drain_ready
            injected = False

            def interrupt_once(*args, **kwargs):
                nonlocal injected
                if not injected:
                    deadline = time.monotonic() + 1.0
                    while not identity_path.exists() and time.monotonic() < deadline:
                        time.sleep(0.005)
                    injected = True
                    raise KeyboardInterrupt("injected cancellation")
                return original_drain(*args, **kwargs)

            with mock.patch.object(harness, "_drain_ready", side_effect=interrupt_once):
                with self.assertRaises(KeyboardInterrupt) as context:
                    self.run_python(source, timeout=2.0)
            pid, starttime = (int(field) for field in identity_path.read_text(encoding="ascii").split())
        self.assertTrue(any("R0 bounded cleanup complete=1" in note for note in context.exception.__notes__))
        assert_process_quiescent(self, pid, starttime)

    def test_pidfd_initialization_interrupt_uses_anchored_emergency_cleanup(self) -> None:
        captured: dict[str, int] = {}

        def interrupt_pidfd(pid: int, flags: int) -> int:
            identity = process_state_starttime(pid)
            self.assertIsNotNone(identity)
            assert identity is not None
            captured.update(pid=pid, starttime=identity[1])
            raise KeyboardInterrupt("pidfd initialization")

        with mock.patch.object(harness.os, "pidfd_open", side_effect=interrupt_pidfd):
            with self.assertRaises(KeyboardInterrupt) as context:
                self.run_python("import time; time.sleep(30)", timeout=2.0)
        self.assertTrue(any("R0 bounded cleanup complete=1" in note for note in context.exception.__notes__))
        assert_process_quiescent(self, captured["pid"], captured["starttime"])

    def test_repeated_primary_cleanup_interrupt_uses_same_session_fallback(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-memprof-r0-cleanup-fallback-") as directory:
            identity_path = Path(directory) / "descendant"
            source = (
                "import os,time,pathlib\n"
                "pid=os.fork()\n"
                "if pid == 0:\n"
                " os.setpgid(0,0); time.sleep(30); os._exit(0)\n"
                "time.sleep(0.02)\n"
                "fields=open('/proc/'+str(pid)+'/stat','rb').read().rsplit(b') ',1)[1].split()\n"
                f"pathlib.Path({str(identity_path)!r}).write_text(str(pid)+' '+fields[19].decode('ascii'))\n"
                "time.sleep(30)\n"
            )
            with mock.patch.object(
                harness,
                "_cleanup_spawned_process",
                side_effect=KeyboardInterrupt("repeated primary cleanup interruption"),
            ):
                with self.assertRaises(KeyboardInterrupt) as context:
                    self.run_python(source, timeout=0.2)
            descendant, starttime = (
                int(field) for field in identity_path.read_text(encoding="ascii").split()
            )
        self.assertTrue(
            any("R0 bounded cleanup raised KeyboardInterrupt" in note for note in context.exception.__notes__)
        )
        self.assertFalse(any("fallback cleanup raised" in note for note in context.exception.__notes__))
        self.assertTrue(any("R0 bounded cleanup complete=1" in note for note in context.exception.__notes__))
        assert_process_quiescent(self, descendant, starttime)

    def test_pidfd_signal_revalidates_starttime(self) -> None:
        identity = harness._read_proc_identity(os.getpid())
        self.assertIsNotNone(identity)
        assert identity is not None
        stale = harness._ProcIdentity(
            identity.pid,
            identity.state,
            identity.process_group,
            identity.session,
            identity.start_time_ticks + 1,
        )
        with mock.patch.object(harness.signal, "pidfd_send_signal") as send:
            self.assertFalse(harness._signal_identity(stale))
        send.assert_not_called()

    def test_zombie_descendants_are_quiescent(self) -> None:
        with mock.patch.object(harness, "_scan_session", return_value=([], 1)):
            result = self.run_python("pass")
        self.assertIsNone(result.failure)
        self.assertFalse(result.killpg_attempted)
        self.assertTrue(result.cleanup_complete)

    def test_descriptor_close_interrupt_preserves_cleanup_evidence(self) -> None:
        real_selector = harness.selectors.DefaultSelector

        class InterruptingSelector:
            def __init__(self):
                self.delegate = real_selector()

            def __getattr__(self, name: str):
                return getattr(self.delegate, name)

            def close(self) -> None:
                self.delegate.close()
                raise KeyboardInterrupt("selector close")

        with mock.patch.object(harness.selectors, "DefaultSelector", InterruptingSelector):
            with self.assertRaises(KeyboardInterrupt) as context:
                self.run_python("pass")
        self.assertTrue(any("R0 bounded cleanup complete=1" in note for note in context.exception.__notes__))
        self.assertTrue(any("descriptor close errors" in note for note in context.exception.__notes__))

    def test_proc_scan_deadline_is_fail_closed(self) -> None:
        with self.assertRaises(harness._ContainmentError):
            harness._scan_session(os.getsid(0), os.getpid(), time.monotonic() - 1.0)

    def test_proc_scan_entry_count_is_hard_bounded(self) -> None:
        entries = [types.SimpleNamespace(name="1"), types.SimpleNamespace(name="2")]

        class FakeScandir:
            def __enter__(self):
                return self

            def __iter__(self):
                return iter(entries)

            def __exit__(self, *unused) -> None:
                return None

        with (
            mock.patch.object(harness, "_PROC_SCAN_MAX_ENTRIES", 1),
            mock.patch.object(harness.os, "scandir", return_value=FakeScandir()),
            mock.patch.object(harness, "_read_proc_identity", return_value=None),
            self.assertRaises(harness._ContainmentError),
        ):
            harness._scan_session(1, 1, time.monotonic() + 1.0)

    def test_proc_stat_record_size_is_hard_bounded(self) -> None:
        read_fd, write_fd = os.pipe()
        os.write(write_fd, b"x" * (harness._PROC_STAT_MAX_BYTES + 1))
        os.close(write_fd)
        with (
            mock.patch.object(harness.os, "open", return_value=read_fd),
            self.assertRaises(harness._ContainmentError),
        ):
            harness._read_proc_identity(123)


class ActualLauncherCliTests(unittest.TestCase):
    def test_failure_evidence_is_lossless_and_control_free(self) -> None:
        stderr = b"\xff\x1b[31m\n"
        result = harness.BoundedProcessResult(("fixture",), 1, 1, b"unexpected\n", stderr, None, False, True)
        with self.assertRaises(actual_launcher.LaunchError) as context:
            actual_launcher.require_case("A01", result)
        rendered = str(context.exception)
        self.assertNotIn("\x1b", rendered)
        self.assertNotIn("\ufffd", rendered)
        self.assertIn(format_byte_evidence("stdout", result.stdout), rendered)
        self.assertIn(format_byte_evidence("stderr", result.stderr), rendered)

    def test_runtime_path_is_passed_only_to_a01(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-memprof-r0-actual-cli-") as directory_text:
            directory = Path(directory_text)
            a00_dso = directory / "a00.so"
            a01_dso = directory / "a01.so"
            runtime = directory / "liboai_memprof_runtime.so.1"
            for path in (a00_dso, a01_dso, runtime):
                path.write_bytes(b"fixture")
            fake_executable = directory / "fixture.py"
            fake_executable.write_text(
                "#!/usr/bin/env python3\n"
                "import os,sys\n"
                f"a00_dso={str(a00_dso.resolve())!r}\n"
                f"a01_dso={str(a01_dso.resolve())!r}\n"
                f"runtime={str(runtime.resolve())!r}\n"
                "args=sys.argv[1:]\n"
                "for name in ('LD_PRELOAD','LD_AUDIT','LD_DEBUG','LD_DEBUG_OUTPUT','LD_PROFILE',"
                "'LD_LIBRARY_PATH','LD_ORIGIN_PATH','LD_POINTER_GUARD','LD_PREFER_MAP_32BIT_EXEC',"
                "'LD_ASSUME_KERNEL','LD_BIND_NOT','LD_BIND_NOW',"
                "'GLIBC_TUNABLES'):\n"
                " assert name not in os.environ, name\n"
                "if args == ['--dso',a00_dso,'--runtime','absent']:\n"
                " pass\n"
                "elif args == ['--dso',a01_dso,'--runtime','present-off','--runtime-path',runtime]:\n"
                " pass\n"
                "elif args == ['--dso',a00_dso,'--runtime','present-off','--runtime-path',runtime]:\n"
                " os.write(2, b'R0_ACTUAL_V1 error=dso_link_identity\\n')\n"
                " raise SystemExit(80)\n"
                "elif args == ['--dso',a01_dso,'--runtime','absent']:\n"
                " os.write(2, b'R0_ACTUAL_V1 error=dso_link_identity\\n')\n"
                " raise SystemExit(80)\n"
                "else:\n"
                " os.write(2, ('unexpected argv: '+repr(args)+'\\n').encode('ascii'))\n"
                " raise SystemExit(91)\n"
                "os.write(1, b'R0_ACTUAL_V1 semantic=pass pre_main=pass main=pass dso_constructor=pass '"
                "b'dso_destructor=pass process_destructor=pass\\n')\n",
                encoding="ascii",
            )
            fake_executable.chmod(0o755)
            launcher = Path(__file__).with_name("run_r0_actual_differential.py")
            result = run_process_bounded(
                [
                    sys.executable,
                    "-B",
                    str(launcher),
                    "--a00-exe",
                    str(fake_executable),
                    "--a00-dso",
                    str(a00_dso),
                    "--a01-exe",
                    str(fake_executable),
                    "--a01-dso",
                    str(a01_dso),
                    "--runtime-path",
                    str(runtime),
                    "--timeout-seconds",
                    "1",
                    "--max-output-bytes",
                    "4096",
                ],
                timeout_seconds=3.0,
                max_stdout_bytes=4096,
                max_stderr_bytes=4096,
            )
        self.assertIsNone(result.failure)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            b"R0_ACTUAL_DIFFERENTIAL_V1 pass cases=4 positives=2 "
            b"identity-negatives=2 transcript=byte-identical\n",
        )
        self.assertEqual(result.stderr, b"")


if __name__ == "__main__":
    unittest.main(verbosity=2)
