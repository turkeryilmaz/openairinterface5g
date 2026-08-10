#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0

"""Deterministic tests for the independent schema-v1 event wire oracle."""

from __future__ import annotations

import re
import sys
import unittest
from dataclasses import replace
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.profiling.memory.oai_memprof_wire import (  # noqa: E402
    EVENT_RECORD_V1_LAYOUT,
    EVENT_RECORD_V1_SIZE,
    EventRecordV1,
    EventRecordWireError,
    decode_event_record_v1,
    encode_event_record_v1,
)


LITERAL_PATH = Path(__file__).with_name("event_record_v1_literal.hex")
LITERAL_VALUES = EventRecordV1(
    thread_sequence=0x0102030405060708,
    counter_enter=0x1112131415161718,
    counter_exit=0x2122232425262728,
    address_before=0x3132333435363738,
    address_after=0x4142434445464748,
    arg0=0x5152535455565758,
    arg1=0x6162636465666768,
    arg2=0x7172737475767778,
    context_id=0x81828384,
    callsite_id=0x91929394,
    thread_index=0xA1A2A3A4,
    flags=0xB1B2B3B4,
    result_code=-0x12345678,
    api_id=0xC1C2,
    event_kind=0xD1D2,
    cpu_enter=0xE1E2,
    cpu_exit=0xF1F2,
)


class EventRecordV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        literal_text = LITERAL_PATH.read_text(encoding="ascii")
        self.assertRegex(literal_text, re.compile(r"\A[0-9a-f]{192}\n?\Z"))
        self.literal = bytes.fromhex(literal_text)

    def test_normative_layout_is_exact_and_contiguous(self) -> None:
        expected = (
            ("thread_sequence", 0, 8, False),
            ("counter_enter", 8, 8, False),
            ("counter_exit", 16, 8, False),
            ("address_before", 24, 8, False),
            ("address_after", 32, 8, False),
            ("arg0", 40, 8, False),
            ("arg1", 48, 8, False),
            ("arg2", 56, 8, False),
            ("context_id", 64, 4, False),
            ("callsite_id", 68, 4, False),
            ("thread_index", 72, 4, False),
            ("flags", 76, 4, False),
            ("result_code", 80, 4, True),
            ("api_id", 84, 2, False),
            ("event_kind", 86, 2, False),
            ("cpu_enter", 88, 2, False),
            ("cpu_exit", 90, 2, False),
            ("reserved_zero", 92, 4, False),
        )
        observed = tuple(
            (field.name, field.offset, field.width, field.signed)
            for field in EVENT_RECORD_V1_LAYOUT
        )
        self.assertEqual(observed, expected)
        self.assertEqual(EVENT_RECORD_V1_SIZE, 96)
        self.assertEqual(len(self.literal), EVENT_RECORD_V1_SIZE)

    def test_frozen_literal_encodes_exactly(self) -> None:
        self.assertEqual(encode_event_record_v1(LITERAL_VALUES), self.literal)

    def test_frozen_literal_decodes_exactly(self) -> None:
        self.assertEqual(decode_event_record_v1(self.literal), LITERAL_VALUES)

    def test_frozen_literal_exercises_each_field_offset(self) -> None:
        for spec in EVENT_RECORD_V1_LAYOUT:
            with self.subTest(field=spec.name):
                expected = getattr(LITERAL_VALUES, spec.name)
                observed = int.from_bytes(
                    self.literal[spec.offset : spec.offset + spec.width],
                    byteorder="little",
                    signed=spec.signed,
                )
                self.assertEqual(observed, expected)

    def test_decode_rejects_wrong_lengths(self) -> None:
        for size in (0, EVENT_RECORD_V1_SIZE - 1, EVENT_RECORD_V1_SIZE + 1):
            with self.subTest(size=size):
                with self.assertRaisesRegex(EventRecordWireError, "expected 96"):
                    decode_event_record_v1(bytes(size))

    def test_decode_rejects_nonzero_reserved_bytes(self) -> None:
        malformed = bytearray(self.literal)
        malformed[92] = 1
        with self.assertRaisesRegex(EventRecordWireError, "reserved_zero"):
            decode_event_record_v1(malformed)

    def test_encode_rejects_nonzero_reserved_value(self) -> None:
        with self.assertRaisesRegex(EventRecordWireError, "reserved_zero"):
            encode_event_record_v1(replace(LITERAL_VALUES, reserved_zero=1))

    def test_unsigned_fields_reject_out_of_range_values(self) -> None:
        for spec in EVENT_RECORD_V1_LAYOUT:
            if spec.signed or spec.name == "reserved_zero":
                continue
            maximum = (1 << (spec.width * 8)) - 1
            for invalid in (-1, maximum + 1):
                with self.subTest(field=spec.name, invalid=invalid):
                    record = replace(LITERAL_VALUES, **{spec.name: invalid})
                    with self.assertRaisesRegex(ValueError, spec.name):
                        encode_event_record_v1(record)

    def test_signed_field_rejects_out_of_range_values(self) -> None:
        for invalid in (-(1 << 31) - 1, 1 << 31):
            with self.subTest(invalid=invalid):
                record = replace(LITERAL_VALUES, result_code=invalid)
                with self.assertRaisesRegex(ValueError, "result_code"):
                    encode_event_record_v1(record)

    def test_fixed_width_boundary_values_round_trip(self) -> None:
        for spec in EVENT_RECORD_V1_LAYOUT:
            if spec.name == "reserved_zero":
                values = (0,)
            elif spec.signed:
                values = (-(1 << (spec.width * 8 - 1)), (1 << (spec.width * 8 - 1)) - 1)
            else:
                values = (0, (1 << (spec.width * 8)) - 1)

            for value in values:
                with self.subTest(field=spec.name, value=value):
                    record = replace(LITERAL_VALUES, **{spec.name: value})
                    decoded = decode_event_record_v1(encode_event_record_v1(record))
                    self.assertEqual(getattr(decoded, spec.name), value)

    def test_bool_is_not_accepted_as_an_integer_field(self) -> None:
        with self.assertRaisesRegex(TypeError, "thread_sequence"):
            encode_event_record_v1(replace(LITERAL_VALUES, thread_sequence=True))

    def test_decode_accepts_each_declared_bytes_like_input(self) -> None:
        for data in (self.literal, bytearray(self.literal), memoryview(self.literal)):
            with self.subTest(type=type(data).__name__):
                self.assertEqual(decode_event_record_v1(data), LITERAL_VALUES)

    def test_decode_rejects_ambiguous_memoryview_layouts(self) -> None:
        non_contiguous = memoryview(bytes(range(192)))[::2]
        multidimensional = memoryview(bytearray(EVENT_RECORD_V1_SIZE)).cast(
            "B", shape=(12, 8)
        )
        non_byte = memoryview(bytearray(EVENT_RECORD_V1_SIZE)).cast("I")

        for data in (non_contiguous, multidimensional, non_byte):
            with self.subTest(
                shape=data.shape,
                itemsize=data.itemsize,
                c_contiguous=data.c_contiguous,
            ):
                self.assertEqual(data.nbytes, EVENT_RECORD_V1_SIZE)
                with self.assertRaisesRegex(EventRecordWireError, "memoryview"):
                    decode_event_record_v1(data)

    def test_public_functions_reject_wrong_object_types(self) -> None:
        with self.assertRaisesRegex(TypeError, "EventRecordV1"):
            encode_event_record_v1({})  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "bytes"):
            decode_event_record_v1(EVENT_RECORD_V1_SIZE)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
