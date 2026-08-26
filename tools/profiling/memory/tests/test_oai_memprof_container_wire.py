"""Targeted independent tests for the Artifact 10 structural wire oracle."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import sys
import unittest


MEMORY_DIR = Path(__file__).resolve().parents[1]
if str(MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_DIR))

import oai_memprof_container_wire as wire


LITERAL_DIR = Path(__file__).with_name("literals")


def literal(name: str) -> bytes:
    return bytes.fromhex((LITERAL_DIR / name).read_text(encoding="ascii"))


def opening_header() -> wire.OpeningHeader:
    return wire.OpeningHeader(
        page_size_bytes=4096,
        scope_kind=1,
        role_kind=1,
        clock_kind=1,
        calibration_kind=1,
        process_generation=0x0102030405060708,
        counter_frequency_numerator=2_400_000_000,
        counter_frequency_denominator=1,
        calibration_error_bound_ns=100,
        calibration_span_ns=0,
        start_counter=0x1122334455667788,
        start_monotonic_raw_ns=0x0102030405060708,
        start_realtime_unix_ns=1_700_000_000_000_000_000,
        pid=12345,
        configured_thread_capacity=256,
        run_uuid=bytes.fromhex("00112233445546778899aabbccddeeff"),
        process_uuid=bytes.fromhex("ffeeddccbbaa49888776655443322110"),
        source_object_kind=1,
        source_object_algorithm=1,
        source_object_length=20,
        source_object_value=bytes(range(0x00, 0x14)) + bytes(12),
        primary_binary_sha256=bytes(range(0x20, 0x40)),
        schema_bundle_definition_sha256=bytes(range(0x40, 0x60)),
        api_catalog_definition_sha256=bytes(range(0x60, 0x80)),
        callsite_catalog_definition_sha256=bytes(range(0x80, 0xA0)),
        configuration_instance_sha256=bytes(range(0xA0, 0xC0)),
        primary_build_id_sha256=bytes(range(0xC0, 0xE0)),
    )


def event_record() -> wire.EventRecord:
    return wire.EventRecord(
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


OBJECT_FLAGS = {
    1: 0x05,
    2: 0x05,
    3: 0x0B,
    4: 0x1B,
    5: 0x1B,
    6: 0x13,
    7: 0x03,
    8: 0x03,
    9: 0x13,
    10: 0x07,
    11: 0x03,
    12: 0x03,
}

OBJECT_COUNTS = {1: 7, 2: 12, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1}


def object_entry(kind: int, opening: wire.OpeningHeader | None = None) -> wire.ObjectBindingEntry:
    if opening is not None and kind == 1:
        digest = opening.schema_bundle_definition_sha256
    elif opening is not None and kind == 2:
        digest = opening.api_catalog_definition_sha256
    elif opening is not None and kind == 10:
        digest = opening.configuration_instance_sha256
    else:
        digest = bytes([kind]) * 32
    return wire.ObjectBindingEntry(
        object_kind=kind,
        format_id=1,
        object_flags=OBJECT_FLAGS[kind],
        schema_revision=1,
        entry_count=OBJECT_COUNTS[kind],
        byte_count=1,
        sha256=digest,
    )


def incomplete_trailer_header() -> wire.TrailerHeader:
    return wire.TrailerHeader(
        trailer_body_bytes=384,
        process_generation=0x0102030405060708,
        scope_kind=1,
        lifecycle_state=7,
        payload_writer_state=5,
        finalization_stage=0,
        terminal_flags=(1 << 0) | (1 << 7) | (1 << 8),
        chunk_count=0,
        record_count=0,
        payload_bytes=0,
        first_chunk_offset=512,
        chunks_end_offset=512,
        active_generation=0x0102030405060708,
        active_start_counter=0x1122334455667788,
        cutoff_before_counter=0,
        cutoff_after_counter=0,
        quiescence_complete_counter=0,
        final_counter=0x1122334455667799,
        active_start_monotonic_raw_ns=0x0102030405060708,
        final_monotonic_raw_ns=0x0102030405060808,
        final_realtime_unix_ns=1_700_000_000_000_000_000,
        event_entry_count=0,
        diagnostic_entry_count=0,
        object_entry_count=2,
        event_table_offset=256,
        diagnostic_table_offset=256,
        object_table_offset=256,
        terminal_reason_code=8,
        diagnostic_loss_sum=0,
        diagnostic_bypass_sum=0,
        saturated_counter_instances=0,
    )


def complete_trailer_body(
    opening: wire.OpeningHeader,
    records: tuple[wire.EventRecord, ...] = (),
    trailer_offset: int = 512,
) -> wire.TrailerBody:
    if records:
        totals = {}
        for record in records:
            key = (record.event_kind, record.api_id)
            totals[key] = totals.get(key, 0) + 1
        event_entries = tuple(wire.EventTotalEntry(key[0], key[1], count) for key, count in sorted(totals.items()))
    else:
        event_entries = ()
    diagnostics = (wire.DiagnosticTotalEntry(1, 0, 0, 0, 0, 0),)
    objects = tuple(object_entry(kind, opening) for kind in range(1, 13))
    event_offset = 256
    diagnostic_offset = event_offset + len(event_entries) * 32
    object_offset = diagnostic_offset + len(diagnostics) * 32
    body_bytes = object_offset + len(objects) * 64
    count = len(records)
    header = wire.TrailerHeader(
        trailer_body_bytes=body_bytes,
        process_generation=opening.process_generation,
        scope_kind=opening.scope_kind,
        lifecycle_state=5,
        payload_writer_state=5,
        finalization_stage=6,
        terminal_flags=0xFFF,
        chunk_count=1 if records else 0,
        record_count=count,
        payload_bytes=count * 96,
        first_chunk_offset=512,
        chunks_end_offset=trailer_offset,
        active_generation=opening.process_generation,
        active_start_counter=opening.start_counter,
        cutoff_before_counter=opening.start_counter + 1,
        cutoff_after_counter=opening.start_counter + 2,
        quiescence_complete_counter=opening.start_counter + 3,
        final_counter=opening.start_counter + 4,
        active_start_monotonic_raw_ns=opening.start_monotonic_raw_ns,
        final_monotonic_raw_ns=opening.start_monotonic_raw_ns + 10,
        final_realtime_unix_ns=opening.start_realtime_unix_ns + 10,
        event_entry_count=len(event_entries),
        diagnostic_entry_count=len(diagnostics),
        object_entry_count=len(objects),
        event_table_offset=event_offset,
        diagnostic_table_offset=diagnostic_offset,
        object_table_offset=object_offset,
        terminal_reason_code=0,
        diagnostic_loss_sum=0,
        diagnostic_bypass_sum=0,
        saturated_counter_instances=0,
    )
    return wire.TrailerBody(header, event_entries, diagnostics, objects)


def complete_container(records: tuple[wire.EventRecord, ...] = ()) -> bytes:
    opening = opening_header()
    opening_bytes = wire.encode_opening_header(opening)
    if records:
        chunk_bytes = wire.encode_chunk(0, records)
    else:
        chunk_bytes = b""
    prefix = opening_bytes + chunk_bytes
    trailer = complete_trailer_body(opening, records, len(prefix))
    trailer_bytes = wire.encode_trailer_body(trailer)
    footer = wire.Footer(
        trailer_offset=len(prefix),
        trailer_body_bytes=len(trailer_bytes),
        stream_bytes=len(prefix) + len(trailer_bytes) + 256,
        prefix_bytes=len(prefix),
        chunk_count=trailer.header.chunk_count,
        record_count=trailer.header.record_count,
        prefix_sha256=sha256(prefix).digest(),
        trailer_body_sha256=sha256(trailer_bytes).digest(),
        opening_header_sha256=sha256(opening_bytes).digest(),
    )
    return prefix + trailer_bytes + wire.encode_footer(footer)


class DescriptorAndCrcTests(unittest.TestCase):
    def test_descriptors_are_exact_and_cover_every_byte(self) -> None:
        expected_sizes = (512, 96, 32, 256, 32, 32, 64, 256)
        self.assertEqual(tuple(descriptor.size for descriptor in wire.ALL_DESCRIPTORS), expected_sizes)
        for descriptor in wire.ALL_DESCRIPTORS:
            wire.validate_descriptor(descriptor)

    def test_descriptor_offsets_are_independently_fixed(self) -> None:
        offsets = {
            descriptor.name: {field.name: (field.offset, field.size) for field in descriptor.fields}
            for descriptor in wire.ALL_DESCRIPTORS
        }
        self.assertEqual(offsets["opening_header_v1"]["header_crc32c"], (508, 4))
        self.assertEqual(offsets["chunk_header_v1"]["payload_crc32c"], (24, 4))
        self.assertEqual(offsets["trailer_header_v1"]["event_table_offset"], (192, 8))
        self.assertEqual(offsets["event_total_entry_v1"]["record_count"], (8, 8))
        self.assertEqual(offsets["diagnostic_total_entry_v1"]["reserved_zero"], (24, 8))
        self.assertEqual(offsets["object_binding_entry_v1"]["sha256"], (32, 32))
        self.assertEqual(offsets["footer_v1"]["footer_sha256"], (224, 32))

    def test_crc32c_frozen_vectors(self) -> None:
        event = literal("event_record_v1.hex")
        self.assertEqual(wire.crc32c(b""), 0x00000000)
        self.assertEqual(wire.crc32c(b"123456789"), 0xE3069283)
        self.assertEqual(wire.crc32c(bytes(96)), 0x51F204AB)
        self.assertEqual(wire.crc32c(bytes(508)), 0xEC57A9C3)
        self.assertEqual(wire.crc32c(event), 0xB639CC1E)
        self.assertEqual(wire.crc32c(b"123456789" + (0xE3069283).to_bytes(4, "little")), 0x48674BC7)

    def test_crc32c_bytes_like_is_strict_and_nonmutating(self) -> None:
        mutable = bytearray(b"123456789")
        before = mutable[:]
        self.assertEqual(wire.crc32c(memoryview(mutable)), 0xE3069283)
        self.assertEqual(mutable, before)
        for rejected in ("123456789", [1, 2, 3], 123):
            with self.subTest(value=rejected):
                with self.assertRaisesRegex(wire.WireError, "^type: data"):
                    wire.crc32c(rejected)
        with self.assertRaises(wire.WireError):
            wire.crc32c(memoryview(bytearray(b"abcdef"))[::2])


class OpeningAndChunkTests(unittest.TestCase):
    def test_opening_header_matches_authoritative_literal(self) -> None:
        expected = literal("opening_header_v1.hex")
        self.assertEqual(len(expected), 512)
        self.assertEqual(wire.encode_opening_header(opening_header()), expected)
        self.assertEqual(wire.decode_opening_header(expected), opening_header())
        self.assertEqual(wire.crc32c(expected[:508]), 0xDF1CA3F4)
        self.assertEqual(sha256(expected).hexdigest(), "cddbbcda4dbda6ed63a32de290734834877c76addf59142dd7be0a5cf4a5cfc0")

    def test_opening_header_precedence_and_input_unchanged(self) -> None:
        valid = literal("opening_header_v1.hex")
        mutable = bytearray(valid)
        before = mutable[:]
        with self.assertRaisesRegex(wire.WireError, "^wrong_size: opening_header"):
            wire.decode_opening_header(mutable[:-1])
        bad_magic = bytearray(valid)
        bad_magic[0] ^= 1
        with self.assertRaisesRegex(wire.WireError, "^bad_magic: magic"):
            wire.decode_opening_header(bad_magic)
        bad_version = bytearray(valid)
        bad_version[8] = 2
        with self.assertRaisesRegex(wire.WireError, "^checksum: header_crc32c"):
            wire.decode_opening_header(bad_version)
        bad_version[508:512] = wire.crc32c(bad_version[:508]).to_bytes(4, "little")
        with self.assertRaisesRegex(wire.WireError, "^version: container_version"):
            wire.decode_opening_header(bad_version)
        bad_minimum_reader_and_length = bytearray(valid)
        bad_minimum_reader_and_length[12:14] = (511).to_bytes(2, "little")
        bad_minimum_reader_and_length[18:20] = (1).to_bytes(2, "little")
        bad_minimum_reader_and_length[508:512] = wire.crc32c(bad_minimum_reader_and_length[:508]).to_bytes(4, "little")
        bad_minimum_reader_and_length_before = bad_minimum_reader_and_length[:]
        with self.assertRaisesRegex(wire.WireError, "^version: minimum_reader_minor"):
            wire.decode_opening_header(bad_minimum_reader_and_length)
        self.assertEqual(bad_minimum_reader_and_length, bad_minimum_reader_and_length_before)
        bad_endian_and_features = bytearray(valid)
        bad_endian_and_features[20:24] = bytes(4)
        bad_endian_and_features[24:28] = bytes(4)
        bad_endian_and_features[508:512] = wire.crc32c(bad_endian_and_features[:508]).to_bytes(4, "little")
        bad_endian_and_features_before = bad_endian_and_features[:]
        with self.assertRaisesRegex(wire.WireError, "^fixed_value: endian_marker"):
            wire.decode_opening_header(bad_endian_and_features)
        self.assertEqual(bad_endian_and_features, bad_endian_and_features_before)
        reserved = bytearray(valid)
        reserved[507] = 1
        reserved[508:512] = wire.crc32c(reserved[:508]).to_bytes(4, "little")
        with self.assertRaisesRegex(wire.WireError, "^nonzero_reserved: reserved_zero_2"):
            wire.decode_opening_header(reserved)
        bad_enum_and_scalar = bytearray(valid)
        bad_enum_and_scalar[28:32] = (3).to_bytes(4, "little")
        bad_enum_and_scalar[33] = 0
        bad_enum_and_scalar[508:512] = wire.crc32c(bad_enum_and_scalar[:508]).to_bytes(4, "little")
        with self.assertRaisesRegex(wire.WireError, "^enum: scope_kind"):
            wire.decode_opening_header(bad_enum_and_scalar)
        self.assertEqual(mutable, before)

    def test_opening_header_rejects_bool_range_and_cross_field_errors(self) -> None:
        cases = (
            replace(opening_header(), page_size_bytes=True),
            replace(opening_header(), page_size_bytes=6000),
            replace(opening_header(), process_generation=0),
            replace(opening_header(), counter_frequency_numerator=6, counter_frequency_denominator=4),
            replace(opening_header(), calibration_kind=1, calibration_span_ns=1),
            replace(opening_header(), calibration_kind=2, calibration_span_ns=1, calibration_error_bound_ns=0),
            replace(opening_header(), process_uuid=opening_header().run_uuid),
            replace(opening_header(), source_object_length=32),
            replace(opening_header(), source_object_value=bytes(range(32))),
        )
        for value in cases:
            with self.subTest(value=value):
                with self.assertRaises(wire.WireError):
                    wire.encode_opening_header(value)

    def test_event_record_matches_independent_literal_and_signed_extremes(self) -> None:
        expected = literal("event_record_v1.hex")
        self.assertEqual(len(expected), 96)
        self.assertEqual(wire.encode_event_record(event_record()), expected)
        self.assertEqual(wire.decode_event_record(expected), event_record())
        self.assertEqual(sha256(expected).hexdigest(), "7b9e3d2f44a8f15e7ecd6b94614e464552d128044a06f608994b90f74143974f")
        for code, encoded in ((-(1 << 31), b"\x00\x00\x00\x80"), ((1 << 31) - 1, b"\xff\xff\xff\x7f"), (-1, b"\xff" * 4)):
            candidate = replace(event_record(), result_code=code)
            raw = wire.encode_event_record(candidate)
            self.assertEqual(raw[80:84], encoded)
            self.assertEqual(wire.decode_event_record(raw), candidate)

    def test_event_record_is_structurally_opaque_but_reserved_is_zero(self) -> None:
        opaque = wire.EventRecord(*(0,) * 11, flags=0xFFFFFFFF, result_code=0, api_id=0xFFFF, event_kind=0xFFFF, cpu_enter=0, cpu_exit=0)
        self.assertEqual(wire.decode_event_record(wire.encode_event_record(opaque)), opaque)
        invalid = bytearray(literal("event_record_v1.hex"))
        invalid[95] = 1
        before = invalid[:]
        with self.assertRaisesRegex(wire.WireError, "^nonzero_reserved: reserved_zero"):
            wire.decode_event_record(invalid)
        self.assertEqual(invalid, before)
        with self.assertRaisesRegex(wire.WireError, "^type: flags"):
            wire.encode_event_record(replace(event_record(), flags=True))

    def test_chunk_matches_frozen_header_and_payload(self) -> None:
        expected_header = literal("chunk_header_v1.hex")
        self.assertEqual(len(expected_header), 32)
        chunk = wire.encode_chunk(0, (event_record(),))
        self.assertEqual(chunk[:32], expected_header)
        self.assertEqual(chunk[32:], literal("event_record_v1.hex"))
        decoded = wire.decode_chunk(bytearray(chunk), expected_sequence=0)
        self.assertEqual(decoded.header, wire.ChunkHeader(0, 1, 96, 0xB639CC1E))
        self.assertEqual(decoded.records, (event_record(),))

    def test_chunk_rejection_order_length_crc_sequence_and_limits(self) -> None:
        chunk = wire.encode_chunk(0, (event_record(),))
        with self.assertRaisesRegex(wire.WireError, "^truncated: chunk_header"):
            wire.decode_chunk(chunk[:31])
        with self.assertRaisesRegex(wire.WireError, "^truncated: chunk_payload"):
            wire.decode_chunk(chunk[:-1])
        with self.assertRaisesRegex(wire.WireError, "^overlong: chunk"):
            wire.decode_chunk(chunk + b"\x00")
        bad_crc = bytearray(chunk)
        bad_crc[-1] ^= 1
        with self.assertRaisesRegex(wire.WireError, "^checksum: payload_crc32c"):
            wire.decode_chunk(bad_crc, expected_sequence=1)
        with self.assertRaisesRegex(wire.WireError, "^sequence: writer_chunk_sequence"):
            wire.decode_chunk(chunk, expected_sequence=1)
        with self.assertRaisesRegex(wire.WireError, "^range: max_records"):
            wire.decode_chunk_header(literal("chunk_header_v1.hex"), max_records=0)
        with self.assertRaisesRegex(wire.WireError, "^type: expected_sequence"):
            wire.decode_chunk(chunk, expected_sequence=True)


class TerminalFixedStructureTests(unittest.TestCase):
    def test_terminal_header_matches_independent_literal(self) -> None:
        expected = literal("trailer_header_v1.hex")
        self.assertEqual(len(expected), 256)
        value = incomplete_trailer_header()
        self.assertEqual(wire.encode_trailer_header(value), expected)
        self.assertEqual(wire.decode_trailer_header(expected), value)

    def test_fixed_table_entries_match_independent_literals(self) -> None:
        event = wire.EventTotalEntry(0x0102, 0x0304, 0x0102030405060708)
        self.assertEqual(wire.encode_event_total_entry(event), literal("event_total_entry_v1.hex"))
        self.assertEqual(wire.decode_event_total_entry(literal("event_total_entry_v1.hex")), event)
        diagnostic = wire.DiagnosticTotalEntry(1, 3, 1, (1 << 64) - 1, 2, 1)
        self.assertEqual(wire.encode_diagnostic_total_entry(diagnostic), literal("diagnostic_total_entry_v1.hex"))
        self.assertEqual(wire.decode_diagnostic_total_entry(literal("diagnostic_total_entry_v1.hex")), diagnostic)
        missing_saturation_summary = replace(diagnostic, summary_flags=0)
        with self.assertRaisesRegex(wire.WireError, "^relation: summary_flags"):
            wire.encode_diagnostic_total_entry(missing_saturation_summary)
        missing_saturation_summary_raw = bytearray(literal("diagnostic_total_entry_v1.hex"))
        missing_saturation_summary_raw[4:8] = bytes(4)
        with self.assertRaisesRegex(wire.WireError, "^relation: summary_flags"):
            wire.decode_diagnostic_total_entry(missing_saturation_summary_raw)
        entries = literal("object_binding_entries_v1.hex")
        first = replace(object_entry(1, opening_header()), byte_count=32)
        last = wire.ObjectBindingEntry(12, 1, 3, 1, 1, 1, bytes(range(0xE0, 0x100)))
        self.assertEqual(wire.encode_object_binding_entry(first) + wire.encode_object_binding_entry(last), entries)
        self.assertEqual(wire.decode_object_binding_entry(entries[:64]), first)
        self.assertEqual(wire.decode_object_binding_entry(entries[64:]), last)
        api = object_entry(2, opening_header())
        api_raw = wire.encode_object_binding_entry(api)
        self.assertEqual(wire.decode_object_binding_entry(api_raw), api)
        with self.assertRaisesRegex(wire.WireError, "^relation: entry_count: API slice requires exactly 12 entries"):
            wire.encode_object_binding_entry(replace(api, entry_count=4))
        bad_api_raw = bytearray(api_raw)
        bad_api_raw[16:24] = (4).to_bytes(8, "little")
        with self.assertRaisesRegex(wire.WireError, "^relation: entry_count: API slice requires exactly 12 entries"):
            wire.decode_object_binding_entry(bad_api_raw)

    def test_terminal_stage_writer_and_reason_pairs_are_exact(self) -> None:
        base = incomplete_trailer_header()
        for mutation in (
            replace(base, finalization_stage=7),
            replace(base, payload_writer_state=4),
            replace(base, payload_writer_state=6),
            replace(base, terminal_flags=base.terminal_flags | (1 << 1)),
            replace(base, terminal_reason_code=2),
            replace(base, terminal_reason_code=10),
            replace(base, active_generation=0),
            replace(base, cutoff_before_counter=1),
        ):
            with self.subTest(mutation=mutation):
                with self.assertRaises(wire.WireError):
                    wire.encode_trailer_header(mutation)
        writer6 = replace(base, payload_writer_state=6, terminal_flags=(1 << 0) | (1 << 8), terminal_reason_code=8)
        self.assertEqual(wire.decode_trailer_header(wire.encode_trailer_header(writer6)), writer6)

    def test_trailer_and_footer_prefix_layout_is_exact_and_overflow_checked(self) -> None:
        base = incomplete_trailer_header()
        with self.assertRaisesRegex(wire.WireError, "^relation: chunks_end_offset"):
            wire.encode_trailer_header(replace(base, chunks_end_offset=544))
        trailer_mutant = bytearray(literal("trailer_header_v1.hex"))
        trailer_mutant[88:96] = (544).to_bytes(8, "little")
        with self.assertRaisesRegex(wire.WireError, "^relation: chunks_end_offset"):
            wire.decode_trailer_header(trailer_mutant)
        with self.assertRaisesRegex(wire.WireError, "^relation: chunk_count"):
            wire.encode_trailer_header(replace(base, chunk_count=1))
        with self.assertRaisesRegex(wire.WireError, "^relation: chunk_count"):
            wire.encode_trailer_header(
                replace(base, chunk_count=2, record_count=1, payload_bytes=96, chunks_end_offset=672)
            )

        footer = wire.Footer(
            trailer_offset=512,
            trailer_body_bytes=384,
            stream_bytes=1152,
            prefix_bytes=512,
            chunk_count=0,
            record_count=0,
            prefix_sha256=bytes(range(0x10, 0x30)),
            trailer_body_sha256=bytes(range(0x30, 0x50)),
            opening_header_sha256=bytes(range(0x50, 0x70)),
        )
        with self.assertRaisesRegex(wire.WireError, "^relation: trailer_offset"):
            wire.encode_footer(replace(footer, trailer_offset=544, prefix_bytes=544, stream_bytes=1184))
        footer_preimage_mutant = bytearray(literal("footer_preimage_v1.hex"))
        footer_preimage_mutant[32:40] = (544).to_bytes(8, "little")
        footer_preimage_mutant[48:56] = (1184).to_bytes(8, "little")
        footer_preimage_mutant[56:64] = (544).to_bytes(8, "little")
        footer_mutant = footer_preimage_mutant + sha256(footer_preimage_mutant).digest()
        with self.assertRaisesRegex(wire.WireError, "^relation: trailer_offset"):
            wire.decode_footer(footer_mutant)
        with self.assertRaisesRegex(wire.WireError, "^relation: chunk_count"):
            wire.encode_footer(replace(footer, chunk_count=1))
        footer_preimage_mutant = bytearray(literal("footer_preimage_v1.hex"))
        footer_preimage_mutant[72:80] = (1).to_bytes(8, "little")
        footer_mutant = footer_preimage_mutant + sha256(footer_preimage_mutant).digest()
        with self.assertRaisesRegex(wire.WireError, "^relation: chunk_count"):
            wire.decode_footer(footer_mutant)
        with self.assertRaisesRegex(wire.WireError, "^relation: chunk_count"):
            wire.encode_footer(replace(footer, chunk_count=2, record_count=1))
        with self.assertRaisesRegex(wire.WireError, "^range: payload_bytes"):
            wire.encode_footer(replace(footer, chunk_count=1, record_count=(1 << 64) - 1))
        odd_body = replace(footer, trailer_body_bytes=257, stream_bytes=1025)
        with self.assertRaisesRegex(wire.WireError, "^range: trailer_body_bytes"):
            wire.encode_footer(odd_body)
        odd_body_raw = bytearray(wire.encode_footer(footer))
        odd_body_raw[40:48] = (257).to_bytes(8, "little")
        odd_body_raw[48:56] = (1025).to_bytes(8, "little")
        odd_body_raw[224:256] = sha256(odd_body_raw[:224]).digest()
        with self.assertRaisesRegex(wire.WireError, "^range: trailer_body_bytes"):
            wire.decode_footer(odd_body_raw)

    def test_object_rows_have_exact_kind_flags_revision_and_counts(self) -> None:
        for kind in range(1, 13):
            value = object_entry(kind, opening_header())
            self.assertEqual(wire.decode_object_binding_entry(wire.encode_object_binding_entry(value)), value)
            with self.assertRaises(wire.WireError):
                wire.encode_object_binding_entry(replace(value, object_flags=value.object_flags ^ 1))
            with self.assertRaises(wire.WireError):
                wire.encode_object_binding_entry(replace(value, schema_revision=2))
        with self.assertRaises(wire.WireError):
            wire.encode_object_binding_entry(replace(object_entry(1), entry_count=6))
        with self.assertRaises(wire.WireError):
            wire.encode_object_binding_entry(replace(object_entry(10), entry_count=0))
        with self.assertRaises(wire.WireError):
            wire.encode_object_binding_entry(replace(object_entry(12), sha256=bytes(32)))

    def test_table_order_counts_aggregates_and_saturation_reconcile(self) -> None:
        opening = opening_header()
        body = complete_trailer_body(opening)
        self.assertEqual(wire.decode_trailer_body(wire.encode_trailer_body(body)), body)
        reversed_objects = replace(body, object_entries=tuple(reversed(body.object_entries)))
        with self.assertRaisesRegex(wire.WireError, "table_order: object_entries"):
            wire.encode_trailer_body(reversed_objects)
        bad_count = replace(body, header=replace(body.header, object_entry_count=11))
        with self.assertRaises(wire.WireError):
            wire.encode_trailer_body(bad_count)
        saturated_diag = wire.DiagnosticTotalEntry(1, 1, 1, (1 << 64) - 1, 1, 1)
        saturated_header = replace(
            body.header,
            terminal_flags=body.header.terminal_flags | (1 << 12),
            diagnostic_loss_sum=(1 << 64) - 1,
            saturated_counter_instances=1,
        )
        saturated_body = replace(body, header=saturated_header, diagnostic_entries=(saturated_diag,))
        self.assertEqual(wire.decode_trailer_body(wire.encode_trailer_body(saturated_body)), saturated_body)
        with self.assertRaises(wire.WireError):
            wire.encode_trailer_body(replace(saturated_body, header=replace(saturated_header, terminal_flags=body.header.terminal_flags)))

        partial_diag = replace(body.diagnostic_entries[0], summary_flags=2)
        partial_body = replace(body, diagnostic_entries=(partial_diag,))
        with self.assertRaisesRegex(wire.WireError, "^relation: terminal_flags"):
            wire.encode_trailer_body(partial_body)
        partial_body_raw = bytearray(wire.encode_trailer_body(body))
        partial_body_raw[body.header.diagnostic_table_offset + 4 : body.header.diagnostic_table_offset + 8] = (2).to_bytes(
            4, "little"
        )
        with self.assertRaisesRegex(wire.WireError, "^relation: terminal_flags"):
            wire.decode_trailer_body(partial_body_raw)

    def test_reserved_bytes_reject_without_mutating_input(self) -> None:
        cases = (
            (wire.decode_event_total_entry, literal("event_total_entry_v1.hex"), 4),
            (wire.decode_diagnostic_total_entry, literal("diagnostic_total_entry_v1.hex"), 31),
            (wire.decode_object_binding_entry, literal("object_binding_entries_v1.hex")[:64], 12),
            (wire.decode_trailer_header, literal("trailer_header_v1.hex"), 255),
        )
        for decoder, raw, offset in cases:
            mutable = bytearray(raw)
            mutable[offset] ^= 1
            before = mutable[:]
            with self.assertRaises(wire.WireError):
                decoder(mutable)
            self.assertEqual(mutable, before)

    def test_footer_preimage_literal_and_self_hash(self) -> None:
        preimage = literal("footer_preimage_v1.hex")
        self.assertEqual(len(preimage), 224)
        footer = wire.Footer(
            trailer_offset=512,
            trailer_body_bytes=384,
            stream_bytes=1152,
            prefix_bytes=512,
            chunk_count=0,
            record_count=0,
            prefix_sha256=bytes(range(0x10, 0x30)),
            trailer_body_sha256=bytes(range(0x30, 0x50)),
            opening_header_sha256=bytes(range(0x50, 0x70)),
        )
        encoded = wire.encode_footer(footer)
        self.assertEqual(encoded[:224], preimage)
        self.assertEqual(encoded[224:], sha256(preimage).digest())
        self.assertEqual(wire.decode_footer(encoded), replace(footer, footer_sha256=sha256(preimage).digest()))

        opaque_zero_hashes = replace(
            footer,
            prefix_sha256=bytes(32),
            trailer_body_sha256=bytes(32),
            opening_header_sha256=bytes(32),
            footer_sha256=bytes(32),
        )
        opaque_encoded = wire.encode_footer(opaque_zero_hashes)
        self.assertEqual(opaque_encoded[96:192], bytes(96))
        self.assertEqual(
            wire.decode_footer(opaque_encoded),
            replace(opaque_zero_hashes, footer_sha256=sha256(opaque_encoded[:224]).digest()),
        )
        with self.assertRaisesRegex(wire.WireError, "^hash: footer_sha256"):
            wire.encode_footer(replace(footer, footer_sha256=bytes([1]) * 32))

    def test_footer_hash_precedence_and_arithmetic(self) -> None:
        preimage = literal("footer_preimage_v1.hex")
        encoded = preimage + sha256(preimage).digest()
        bad_magic = bytearray(encoded)
        bad_magic[0] ^= 1
        with self.assertRaisesRegex(wire.WireError, "^bad_magic: magic"):
            wire.decode_footer(bad_magic)
        bad_layout = bytearray(encoded)
        bad_layout[48] ^= 1
        with self.assertRaisesRegex(wire.WireError, "^hash: footer_sha256"):
            wire.decode_footer(bad_layout)
        bad_layout[224:] = sha256(bad_layout[:224]).digest()
        with self.assertRaisesRegex(wire.WireError, "^relation: stream_bytes"):
            wire.decode_footer(bad_layout)


class WholeContainerTests(unittest.TestCase):
    def test_zero_record_complete_candidate(self) -> None:
        encoded = complete_container()
        decoded = wire.decode_container(memoryview(encoded))
        self.assertEqual(decoded.opening_header, opening_header())
        self.assertEqual(decoded.chunks, ())
        self.assertEqual(decoded.trailer_body.header.lifecycle_state, 5)
        self.assertEqual(decoded.footer.stream_bytes, len(encoded))

    def test_one_record_container_rederives_chunk_and_event_totals(self) -> None:
        encoded = complete_container((event_record(),))
        decoded = wire.decode_container(encoded)
        self.assertEqual(len(decoded.chunks), 1)
        self.assertEqual(decoded.chunks[0].records, (event_record(),))
        self.assertEqual(decoded.trailer_body.event_entries, (wire.EventTotalEntry(0xD1D2, 0xC1C2, 1),))

    def test_structurally_valid_incomplete_is_retained_negative_evidence(self) -> None:
        opening = opening_header()
        opening_bytes = wire.encode_opening_header(opening)
        header = incomplete_trailer_header()
        body = wire.TrailerBody(header, (), (), (object_entry(1, opening), object_entry(12, opening)))
        body_bytes = wire.encode_trailer_body(body)
        footer = wire.Footer(
            trailer_offset=512,
            trailer_body_bytes=len(body_bytes),
            stream_bytes=512 + len(body_bytes) + 256,
            prefix_bytes=512,
            chunk_count=0,
            record_count=0,
            prefix_sha256=sha256(opening_bytes).digest(),
            trailer_body_sha256=sha256(body_bytes).digest(),
            opening_header_sha256=sha256(opening_bytes).digest(),
        )
        decoded = wire.decode_container(opening_bytes + body_bytes + wire.encode_footer(footer))
        self.assertEqual(decoded.trailer_body.header.lifecycle_state, 7)
        self.assertEqual(decoded.trailer_body.header.terminal_reason_code, 8)

    def test_exact_eof_truncation_append_and_hash_mutants_reject(self) -> None:
        encoded = complete_container()
        for mutant in (encoded[:-1], encoded + b"\x00"):
            with self.subTest(length=len(mutant)):
                with self.assertRaises(wire.WireError):
                    wire.decode_container(mutant)
        body_mutant = bytearray(encoded)
        body_mutant[512] ^= 1
        with self.assertRaisesRegex(wire.WireError, "^hash: trailer_body_sha256"):
            wire.decode_container(body_mutant)
        opening_mutant = bytearray(encoded)
        opening_mutant[100] ^= 1
        before = opening_mutant[:]
        with self.assertRaisesRegex(wire.WireError, "^checksum: header_crc32c"):
            wire.decode_container(opening_mutant)
        self.assertEqual(opening_mutant, before)

    def test_inner_chunk_mutant_recomputes_outer_hashes_to_reach_crc(self) -> None:
        encoded = bytearray(complete_container((event_record(),)))
        trailer_offset = int.from_bytes(encoded[-224:-216], "little")
        encoded[512 + 32] ^= 1
        prefix = bytes(encoded[:trailer_offset])
        encoded[-160:-128] = sha256(prefix).digest()
        encoded[-32:] = sha256(encoded[-256:-32]).digest()
        before = encoded[:]
        with self.assertRaisesRegex(wire.WireError, "^checksum: payload_crc32c"):
            wire.decode_container(encoded)
        self.assertEqual(encoded, before)

    def test_opening_hash_precedes_event_table_relations(self) -> None:
        encoded = bytearray(complete_container((event_record(),)))
        footer_offset = len(encoded) - wire.FOOTER_BYTES
        trailer_offset = int.from_bytes(encoded[footer_offset + 32 : footer_offset + 40], "little")

        encoded[96:104] = (opening_header().start_realtime_unix_ns + 1).to_bytes(8, "little")
        encoded[508:512] = wire.crc32c(encoded[:508]).to_bytes(4, "little")

        event_table_offset = int.from_bytes(encoded[trailer_offset + 192 : trailer_offset + 200], "little")
        event_total_offset = trailer_offset + event_table_offset
        encoded[event_total_offset + 8 : event_total_offset + 16] = (2).to_bytes(8, "little")

        encoded[footer_offset + 96 : footer_offset + 128] = sha256(encoded[:trailer_offset]).digest()
        encoded[footer_offset + 128 : footer_offset + 160] = sha256(encoded[trailer_offset:footer_offset]).digest()
        encoded[footer_offset + 224 : footer_offset + 256] = sha256(encoded[footer_offset : footer_offset + 224]).digest()

        before = encoded[:]
        with self.assertRaisesRegex(wire.WireError, "^hash: opening_header_sha256"):
            wire.decode_container(encoded)
        self.assertEqual(encoded, before)


if __name__ == "__main__":
    unittest.main()
