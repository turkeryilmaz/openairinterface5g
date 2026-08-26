"""Independent structural oracle for the OAI memory-profiler v1.0 container.

This module deliberately implements only the byte-level contract frozen by
Artifact 10.  It has no file I/O, writer, ring, archive, JSON, or catalog
semantics.  All multibyte integers are placed with explicit offsets and
``int.to_bytes``; native structure layouts are never used.

Decode functions are functional: they either return a new immutable value or
raise :class:`WireError`.  Consequently, a failed decode cannot partially
modify caller-owned output state.  Bytes-like inputs are snapshotted before
validation so mutable inputs are also left unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import gcd
from typing import Dict, Iterable, Optional, Sequence, Tuple


CONTAINER_MAJOR = 1
CONTAINER_MINOR = 0
MINIMUM_READER_MINOR = 0
OPENING_HEADER_BYTES = 512
EVENT_RECORD_BYTES = 96
CHUNK_HEADER_BYTES = 32
TRAILER_HEADER_BYTES = 256
EVENT_TOTAL_ENTRY_BYTES = 32
DIAGNOSTIC_TOTAL_ENTRY_BYTES = 32
OBJECT_BINDING_ENTRY_BYTES = 64
FOOTER_BYTES = 256

OPENING_MAGIC = b"OAIMEM01"
CHUNK_MAGIC = b"OMC1"
TRAILER_MAGIC = b"OAI_MEMPROF_TR1\x00"
FOOTER_MAGIC = b"OAI_MEMPROF_END\x00"

REQUIRED_FEATURES = 0x0000000F
ENDIAN_MARKER = 0x01020304
POINTER_WIDTH_BYTES = 8
FOOTER_FLAGS = 0x000000000000000F
TERMINAL_FLAGS_MASK = 0x000000000001FFFF
DIAGNOSTIC_CLASS_FLAGS_MASK = 0x03FF
DIAGNOSTIC_SUMMARY_FLAGS_MASK = 0x00000003
OBJECT_FLAGS_MASK = 0x0000001F

MAX_CHUNK_RECORD_COUNT = 0xFFFFFFFF // EVENT_RECORD_BYTES
MAX_EVENT_ENTRIES = 16384
MAX_DIAGNOSTIC_ENTRIES = 4096
MAX_OBJECT_ENTRIES = 64
MAX_TRAILER_BODY_BYTES = 1048576
MAX_OBJECT_BYTE_COUNT = 268435456
MAX_OBJECT_ENTRY_COUNT = 16777216

UINT16_MAX = (1 << 16) - 1
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


class ErrorCode:
    """Stable error categories used by the independent oracle."""

    TYPE = "type"
    WRONG_SIZE = "wrong_size"
    TRUNCATED = "truncated"
    OVERLONG = "overlong"
    BAD_MAGIC = "bad_magic"
    CHECKSUM = "checksum"
    HASH = "hash"
    VERSION = "version"
    FIXED_VALUE = "fixed_value"
    NONZERO_RESERVED = "nonzero_reserved"
    RANGE = "range"
    ENUM = "enum"
    RELATION = "relation"
    SEQUENCE = "sequence"
    TABLE_ORDER = "table_order"


class WireError(ValueError):
    """A deterministic structural rejection."""

    def __init__(self, code: str, field: str, detail: str = "") -> None:
        self.code = code
        self.field = field
        self.detail = detail
        message = "%s: %s" % (code, field)
        if detail:
            message += ": " + detail
        super().__init__(message)


@dataclass(frozen=True)
class FieldDescriptor:
    name: str
    offset: int
    size: int
    kind: str
    constant: object = None


@dataclass(frozen=True)
class StructureDescriptor:
    name: str
    size: int
    fields: Tuple[FieldDescriptor, ...]


def _fd(name: str, offset: int, size: int, kind: str, constant: object = None) -> FieldDescriptor:
    return FieldDescriptor(name, offset, size, kind, constant)


OPENING_HEADER_DESCRIPTOR = StructureDescriptor(
    "opening_header_v1",
    OPENING_HEADER_BYTES,
    (
        _fd("magic", 0, 8, "bytes", OPENING_MAGIC),
        _fd("container_major", 8, 2, "u16", CONTAINER_MAJOR),
        _fd("container_minor", 10, 2, "u16", CONTAINER_MINOR),
        _fd("header_bytes", 12, 2, "u16", OPENING_HEADER_BYTES),
        _fd("event_record_bytes", 14, 2, "u16", EVENT_RECORD_BYTES),
        _fd("chunk_header_bytes", 16, 2, "u16", CHUNK_HEADER_BYTES),
        _fd("minimum_reader_minor", 18, 2, "u16", MINIMUM_READER_MINOR),
        _fd("required_features", 20, 4, "u32", REQUIRED_FEATURES),
        _fd("endian_marker", 24, 4, "u32", ENDIAN_MARKER),
        _fd("page_size_bytes", 28, 4, "u32"),
        _fd("pointer_width_bytes", 32, 1, "u8", POINTER_WIDTH_BYTES),
        _fd("scope_kind", 33, 1, "u8"),
        _fd("role_kind", 34, 1, "u8"),
        _fd("clock_kind", 35, 1, "u8"),
        _fd("calibration_kind", 36, 2, "u16"),
        _fd("reserved_zero_0", 38, 2, "reserved", 0),
        _fd("process_generation", 40, 8, "u64"),
        _fd("counter_frequency_numerator", 48, 8, "u64"),
        _fd("counter_frequency_denominator", 56, 8, "u64"),
        _fd("calibration_error_bound_ns", 64, 8, "u64"),
        _fd("calibration_span_ns", 72, 8, "u64"),
        _fd("start_counter", 80, 8, "u64"),
        _fd("start_monotonic_raw_ns", 88, 8, "u64"),
        _fd("start_realtime_unix_ns", 96, 8, "u64"),
        _fd("pid", 104, 4, "u32"),
        _fd("configured_thread_capacity", 108, 4, "u32"),
        _fd("run_uuid", 112, 16, "bytes"),
        _fd("process_uuid", 128, 16, "bytes"),
        _fd("source_object_kind", 144, 2, "u16"),
        _fd("source_object_algorithm", 146, 2, "u16"),
        _fd("source_object_length", 148, 2, "u16"),
        _fd("reserved_zero_1", 150, 2, "reserved", 0),
        _fd("source_object_value", 152, 32, "bytes"),
        _fd("primary_binary_sha256", 184, 32, "bytes"),
        _fd("schema_bundle_definition_sha256", 216, 32, "bytes"),
        _fd("api_catalog_definition_sha256", 248, 32, "bytes"),
        _fd("callsite_catalog_definition_sha256", 280, 32, "bytes"),
        _fd("configuration_instance_sha256", 312, 32, "bytes"),
        _fd("primary_build_id_sha256", 344, 32, "bytes"),
        _fd("reserved_zero_2", 376, 132, "reserved", 0),
        _fd("header_crc32c", 508, 4, "u32"),
    ),
)


EVENT_RECORD_DESCRIPTOR = StructureDescriptor(
    "event_record_v1",
    EVENT_RECORD_BYTES,
    (
        _fd("thread_sequence", 0, 8, "u64"),
        _fd("counter_enter", 8, 8, "u64"),
        _fd("counter_exit", 16, 8, "u64"),
        _fd("address_before", 24, 8, "u64"),
        _fd("address_after", 32, 8, "u64"),
        _fd("arg0", 40, 8, "u64"),
        _fd("arg1", 48, 8, "u64"),
        _fd("arg2", 56, 8, "u64"),
        _fd("context_id", 64, 4, "u32"),
        _fd("callsite_id", 68, 4, "u32"),
        _fd("thread_index", 72, 4, "u32"),
        _fd("flags", 76, 4, "u32"),
        _fd("result_code", 80, 4, "i32"),
        _fd("api_id", 84, 2, "u16"),
        _fd("event_kind", 86, 2, "u16"),
        _fd("cpu_enter", 88, 2, "u16"),
        _fd("cpu_exit", 90, 2, "u16"),
        _fd("reserved_zero", 92, 4, "reserved", 0),
    ),
)


CHUNK_HEADER_DESCRIPTOR = StructureDescriptor(
    "chunk_header_v1",
    CHUNK_HEADER_BYTES,
    (
        _fd("magic", 0, 4, "bytes", CHUNK_MAGIC),
        _fd("chunk_major", 4, 1, "u8", CONTAINER_MAJOR),
        _fd("chunk_minor", 5, 1, "u8", CONTAINER_MINOR),
        _fd("header_bytes", 6, 2, "u16", CHUNK_HEADER_BYTES),
        _fd("writer_chunk_sequence", 8, 8, "u64"),
        _fd("record_count", 16, 4, "u32"),
        _fd("payload_bytes", 20, 4, "u32"),
        _fd("payload_crc32c", 24, 4, "u32"),
        _fd("flags", 28, 4, "u32", 0),
    ),
)


TRAILER_HEADER_DESCRIPTOR = StructureDescriptor(
    "trailer_header_v1",
    TRAILER_HEADER_BYTES,
    (
        _fd("magic", 0, 16, "bytes", TRAILER_MAGIC),
        _fd("schema_major", 16, 2, "u16", CONTAINER_MAJOR),
        _fd("schema_minor", 18, 2, "u16", CONTAINER_MINOR),
        _fd("fixed_header_bytes", 20, 4, "u32", TRAILER_HEADER_BYTES),
        _fd("trailer_body_bytes", 24, 8, "u64"),
        _fd("process_generation", 32, 8, "u64"),
        _fd("scope_kind", 40, 2, "u16"),
        _fd("lifecycle_state", 42, 2, "u16"),
        _fd("payload_writer_state", 44, 2, "u16"),
        _fd("finalization_stage", 46, 2, "u16"),
        _fd("terminal_flags", 48, 8, "u64"),
        _fd("chunk_count", 56, 8, "u64"),
        _fd("record_count", 64, 8, "u64"),
        _fd("payload_bytes", 72, 8, "u64"),
        _fd("first_chunk_offset", 80, 8, "u64"),
        _fd("chunks_end_offset", 88, 8, "u64"),
        _fd("active_generation", 96, 8, "u64"),
        _fd("active_start_counter", 104, 8, "u64"),
        _fd("cutoff_before_counter", 112, 8, "u64"),
        _fd("cutoff_after_counter", 120, 8, "u64"),
        _fd("quiescence_complete_counter", 128, 8, "u64"),
        _fd("final_counter", 136, 8, "u64"),
        _fd("active_start_monotonic_raw_ns", 144, 8, "u64"),
        _fd("final_monotonic_raw_ns", 152, 8, "u64"),
        _fd("final_realtime_unix_ns", 160, 8, "u64"),
        _fd("event_entry_count", 168, 4, "u32"),
        _fd("event_entry_bytes", 172, 4, "u32", EVENT_TOTAL_ENTRY_BYTES),
        _fd("diagnostic_entry_count", 176, 4, "u32"),
        _fd("diagnostic_entry_bytes", 180, 4, "u32", DIAGNOSTIC_TOTAL_ENTRY_BYTES),
        _fd("object_entry_count", 184, 4, "u32"),
        _fd("object_entry_bytes", 188, 4, "u32", OBJECT_BINDING_ENTRY_BYTES),
        _fd("event_table_offset", 192, 8, "u64"),
        _fd("diagnostic_table_offset", 200, 8, "u64"),
        _fd("object_table_offset", 208, 8, "u64"),
        _fd("terminal_reason_code", 216, 4, "u32"),
        _fd("reserved_zero_0", 220, 4, "reserved", 0),
        _fd("diagnostic_loss_sum", 224, 8, "u64"),
        _fd("diagnostic_bypass_sum", 232, 8, "u64"),
        _fd("saturated_counter_instances", 240, 8, "u64"),
        _fd("reserved_zero_1", 248, 8, "reserved", 0),
    ),
)


EVENT_TOTAL_ENTRY_DESCRIPTOR = StructureDescriptor(
    "event_total_entry_v1",
    EVENT_TOTAL_ENTRY_BYTES,
    (
        _fd("event_kind", 0, 2, "u16"),
        _fd("api_id", 2, 2, "u16"),
        _fd("reserved_zero_0", 4, 4, "reserved", 0),
        _fd("record_count", 8, 8, "u64"),
        _fd("reserved_zero_1", 16, 16, "reserved", 0),
    ),
)


DIAGNOSTIC_TOTAL_ENTRY_DESCRIPTOR = StructureDescriptor(
    "diagnostic_total_entry_v1",
    DIAGNOSTIC_TOTAL_ENTRY_BYTES,
    (
        _fd("reason_id", 0, 2, "u16"),
        _fd("class_flags", 2, 2, "u16"),
        _fd("summary_flags", 4, 4, "u32"),
        _fd("saturating_total", 8, 8, "u64"),
        _fd("nonzero_counter_instances", 16, 4, "u32"),
        _fd("saturated_counter_instances", 20, 4, "u32"),
        _fd("reserved_zero", 24, 8, "reserved", 0),
    ),
)


OBJECT_BINDING_ENTRY_DESCRIPTOR = StructureDescriptor(
    "object_binding_entry_v1",
    OBJECT_BINDING_ENTRY_BYTES,
    (
        _fd("object_kind", 0, 2, "u16"),
        _fd("format_id", 2, 2, "u16", 1),
        _fd("object_flags", 4, 4, "u32"),
        _fd("schema_revision", 8, 4, "u32"),
        _fd("reserved_zero", 12, 4, "reserved", 0),
        _fd("entry_count", 16, 8, "u64"),
        _fd("byte_count", 24, 8, "u64"),
        _fd("sha256", 32, 32, "bytes"),
    ),
)


FOOTER_DESCRIPTOR = StructureDescriptor(
    "footer_v1",
    FOOTER_BYTES,
    (
        _fd("magic", 0, 16, "bytes", FOOTER_MAGIC),
        _fd("schema_major", 16, 2, "u16", CONTAINER_MAJOR),
        _fd("schema_minor", 18, 2, "u16", CONTAINER_MINOR),
        _fd("footer_bytes", 20, 4, "u32", FOOTER_BYTES),
        _fd("footer_flags", 24, 8, "u64", FOOTER_FLAGS),
        _fd("trailer_offset", 32, 8, "u64"),
        _fd("trailer_body_bytes", 40, 8, "u64"),
        _fd("stream_bytes", 48, 8, "u64"),
        _fd("prefix_bytes", 56, 8, "u64"),
        _fd("header_bytes", 64, 8, "u64", OPENING_HEADER_BYTES),
        _fd("chunk_count", 72, 8, "u64"),
        _fd("record_count", 80, 8, "u64"),
        _fd("reserved_zero_0", 88, 8, "reserved", 0),
        _fd("prefix_sha256", 96, 32, "bytes"),
        _fd("trailer_body_sha256", 128, 32, "bytes"),
        _fd("opening_header_sha256", 160, 32, "bytes"),
        _fd("reserved_zero_1", 192, 32, "reserved", 0),
        _fd("footer_sha256", 224, 32, "bytes"),
    ),
)


ALL_DESCRIPTORS = (
    OPENING_HEADER_DESCRIPTOR,
    EVENT_RECORD_DESCRIPTOR,
    CHUNK_HEADER_DESCRIPTOR,
    TRAILER_HEADER_DESCRIPTOR,
    EVENT_TOTAL_ENTRY_DESCRIPTOR,
    DIAGNOSTIC_TOTAL_ENTRY_DESCRIPTOR,
    OBJECT_BINDING_ENTRY_DESCRIPTOR,
    FOOTER_DESCRIPTOR,
)


@dataclass(frozen=True)
class OpeningHeader:
    page_size_bytes: int
    scope_kind: int
    role_kind: int
    clock_kind: int
    calibration_kind: int
    process_generation: int
    counter_frequency_numerator: int
    counter_frequency_denominator: int
    calibration_error_bound_ns: int
    calibration_span_ns: int
    start_counter: int
    start_monotonic_raw_ns: int
    start_realtime_unix_ns: int
    pid: int
    configured_thread_capacity: int
    run_uuid: bytes
    process_uuid: bytes
    source_object_kind: int
    source_object_algorithm: int
    source_object_length: int
    source_object_value: bytes
    primary_binary_sha256: bytes
    schema_bundle_definition_sha256: bytes
    api_catalog_definition_sha256: bytes
    callsite_catalog_definition_sha256: bytes
    configuration_instance_sha256: bytes
    primary_build_id_sha256: bytes


@dataclass(frozen=True)
class EventRecord:
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


@dataclass(frozen=True)
class ChunkHeader:
    writer_chunk_sequence: int
    record_count: int
    payload_bytes: int
    payload_crc32c: int


@dataclass(frozen=True)
class Chunk:
    header: ChunkHeader
    records: Tuple[EventRecord, ...]


@dataclass(frozen=True)
class TrailerHeader:
    trailer_body_bytes: int
    process_generation: int
    scope_kind: int
    lifecycle_state: int
    payload_writer_state: int
    finalization_stage: int
    terminal_flags: int
    chunk_count: int
    record_count: int
    payload_bytes: int
    first_chunk_offset: int
    chunks_end_offset: int
    active_generation: int
    active_start_counter: int
    cutoff_before_counter: int
    cutoff_after_counter: int
    quiescence_complete_counter: int
    final_counter: int
    active_start_monotonic_raw_ns: int
    final_monotonic_raw_ns: int
    final_realtime_unix_ns: int
    event_entry_count: int
    diagnostic_entry_count: int
    object_entry_count: int
    event_table_offset: int
    diagnostic_table_offset: int
    object_table_offset: int
    terminal_reason_code: int
    diagnostic_loss_sum: int
    diagnostic_bypass_sum: int
    saturated_counter_instances: int


@dataclass(frozen=True)
class EventTotalEntry:
    event_kind: int
    api_id: int
    record_count: int


@dataclass(frozen=True)
class DiagnosticTotalEntry:
    reason_id: int
    class_flags: int
    summary_flags: int
    saturating_total: int
    nonzero_counter_instances: int
    saturated_counter_instances: int


@dataclass(frozen=True)
class ObjectBindingEntry:
    object_kind: int
    format_id: int
    object_flags: int
    schema_revision: int
    entry_count: int
    byte_count: int
    sha256: bytes


@dataclass(frozen=True)
class TrailerBody:
    header: TrailerHeader
    event_entries: Tuple[EventTotalEntry, ...]
    diagnostic_entries: Tuple[DiagnosticTotalEntry, ...]
    object_entries: Tuple[ObjectBindingEntry, ...]


@dataclass(frozen=True)
class Footer:
    trailer_offset: int
    trailer_body_bytes: int
    stream_bytes: int
    prefix_bytes: int
    chunk_count: int
    record_count: int
    prefix_sha256: bytes
    trailer_body_sha256: bytes
    opening_header_sha256: bytes
    footer_sha256: bytes = b""


@dataclass(frozen=True)
class Container:
    opening_header: OpeningHeader
    chunks: Tuple[Chunk, ...]
    trailer_body: TrailerBody
    footer: Footer


def _fail(code: str, field: str, detail: str = "") -> None:
    raise WireError(code, field, detail)


def _bytes_like(value: object, field: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        _fail(ErrorCode.TYPE, field, "expected bytes, bytearray, or byte-oriented memoryview")
    if isinstance(value, memoryview):
        if value.ndim != 1 or value.itemsize != 1 or not value.c_contiguous:
            _fail(ErrorCode.TYPE, field, "memoryview must be one-dimensional, byte-sized, and C-contiguous")
        return value.tobytes()
    return bytes(value)


def _exact_input(value: object, size: int, field: str) -> bytes:
    raw = _bytes_like(value, field)
    if len(raw) != size:
        _fail(ErrorCode.WRONG_SIZE, field, "expected %d bytes, got %d" % (size, len(raw)))
    return raw


def _uint(value: object, bits: int, field: str) -> int:
    if type(value) is not int:
        _fail(ErrorCode.TYPE, field, "expected an integer (bool is not accepted)")
    maximum = (1 << bits) - 1
    if value < 0 or value > maximum:
        _fail(ErrorCode.RANGE, field, "outside u%d" % bits)
    return value


def _sint(value: object, bits: int, field: str) -> int:
    if type(value) is not int:
        _fail(ErrorCode.TYPE, field, "expected an integer (bool is not accepted)")
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    if value < minimum or value > maximum:
        _fail(ErrorCode.RANGE, field, "outside i%d" % bits)
    return value


def _fixed_bytes(value: object, size: int, field: str) -> bytes:
    raw = _bytes_like(value, field)
    if len(raw) != size:
        _fail(ErrorCode.WRONG_SIZE, field, "expected %d bytes, got %d" % (size, len(raw)))
    return raw


def _read_u(raw: bytes, offset: int, size: int) -> int:
    return int.from_bytes(raw[offset : offset + size], "little", signed=False)


def _read_i(raw: bytes, offset: int, size: int) -> int:
    return int.from_bytes(raw[offset : offset + size], "little", signed=True)


def _put_u(target: bytearray, offset: int, size: int, value: object, field: str) -> None:
    checked = _uint(value, size * 8, field)
    target[offset : offset + size] = checked.to_bytes(size, "little", signed=False)


def _put_i(target: bytearray, offset: int, size: int, value: object, field: str) -> None:
    checked = _sint(value, size * 8, field)
    target[offset : offset + size] = checked.to_bytes(size, "little", signed=True)


def _put_bytes(target: bytearray, offset: int, size: int, value: object, field: str) -> None:
    target[offset : offset + size] = _fixed_bytes(value, size, field)


def _expect_bytes(raw: bytes, offset: int, expected: bytes, field: str, code: str) -> None:
    if raw[offset : offset + len(expected)] != expected:
        _fail(code, field)


def _expect_u(raw: bytes, offset: int, size: int, expected: int, field: str, code: str) -> None:
    if _read_u(raw, offset, size) != expected:
        _fail(code, field)


def _expect_zero(raw: bytes, offset: int, size: int, field: str) -> None:
    if any(raw[offset : offset + size]):
        _fail(ErrorCode.NONZERO_RESERVED, field)


def _nonzero_digest(value: object, field: str) -> bytes:
    raw = _fixed_bytes(value, 32, field)
    if not any(raw):
        _fail(ErrorCode.RELATION, field, "all-zero digest is a placeholder")
    return raw


def _checked_add(left: int, right: int, maximum: int, field: str) -> int:
    result = left + right
    if result > maximum:
        _fail(ErrorCode.RANGE, field, "addition overflow")
    return result


def _checked_mul(left: int, right: int, maximum: int, field: str) -> int:
    result = left * right
    if result > maximum:
        _fail(ErrorCode.RANGE, field, "multiplication overflow")
    return result


def crc32c(data: object) -> int:
    """Return CRC-32C/Castagnoli using the frozen reflected parameters."""

    raw = _bytes_like(data, "data")
    register = 0xFFFFFFFF
    for octet in raw:
        register ^= octet
        for _ in range(8):
            if register & 1:
                register = (register >> 1) ^ 0x82F63B78
            else:
                register >>= 1
    return register ^ 0xFFFFFFFF


def _validate_opening_header(value: OpeningHeader) -> None:
    scope_kind = _uint(value.scope_kind, 8, "scope_kind")
    role_kind = _uint(value.role_kind, 8, "role_kind")
    clock_kind = _uint(value.clock_kind, 8, "clock_kind")
    calibration_kind = _uint(value.calibration_kind, 16, "calibration_kind")
    source_object_kind = _uint(value.source_object_kind, 16, "source_object_kind")
    source_object_algorithm = _uint(value.source_object_algorithm, 16, "source_object_algorithm")
    if scope_kind not in (1, 2):
        _fail(ErrorCode.ENUM, "scope_kind")
    if role_kind not in (1, 2):
        _fail(ErrorCode.ENUM, "role_kind")
    if clock_kind not in (1, 2):
        _fail(ErrorCode.ENUM, "clock_kind")
    if calibration_kind not in (1, 2):
        _fail(ErrorCode.ENUM, "calibration_kind")
    if source_object_kind != 1:
        _fail(ErrorCode.ENUM, "source_object_kind")
    if source_object_algorithm not in (1, 2):
        _fail(ErrorCode.ENUM, "source_object_algorithm")

    _uint(value.page_size_bytes, 32, "page_size_bytes")
    if value.page_size_bytes < 4096 or value.page_size_bytes & (value.page_size_bytes - 1):
        _fail(ErrorCode.RELATION, "page_size_bytes", "must be a power of two at least 4096")

    if _uint(value.process_generation, 64, "process_generation") == 0:
        _fail(ErrorCode.RELATION, "process_generation", "must be nonzero")
    numerator = _uint(value.counter_frequency_numerator, 64, "counter_frequency_numerator")
    denominator = _uint(value.counter_frequency_denominator, 64, "counter_frequency_denominator")
    if numerator == 0 or denominator == 0:
        _fail(ErrorCode.RELATION, "counter_frequency", "numerator and denominator must be nonzero")
    if gcd(numerator, denominator) != 1:
        _fail(ErrorCode.RELATION, "counter_frequency", "ratio must be in lowest terms")
    _uint(value.calibration_error_bound_ns, 64, "calibration_error_bound_ns")
    span = _uint(value.calibration_span_ns, 64, "calibration_span_ns")
    if value.calibration_kind == 1 and span != 0:
        _fail(ErrorCode.RELATION, "calibration_span_ns", "EXACT_RATE requires zero span")
    if value.calibration_kind == 2 and (span == 0 or value.calibration_error_bound_ns == 0):
        _fail(ErrorCode.RELATION, "calibration", "MEASURED_AFFINE requires positive span and error bound")

    for name in ("start_counter", "start_monotonic_raw_ns", "start_realtime_unix_ns"):
        _uint(getattr(value, name), 64, name)
    if _uint(value.pid, 32, "pid") == 0:
        _fail(ErrorCode.RELATION, "pid", "must be nonzero")
    if _uint(value.configured_thread_capacity, 32, "configured_thread_capacity") == 0:
        _fail(ErrorCode.RELATION, "configured_thread_capacity", "must be nonzero")

    run_uuid = _fixed_bytes(value.run_uuid, 16, "run_uuid")
    process_uuid = _fixed_bytes(value.process_uuid, 16, "process_uuid")
    for name, identifier in (("run_uuid", run_uuid), ("process_uuid", process_uuid)):
        if not any(identifier):
            _fail(ErrorCode.RELATION, name, "nil UUID")
        if identifier[8] & 0xC0 != 0x80:
            _fail(ErrorCode.RELATION, name, "not an RFC variant UUID")
    if run_uuid == process_uuid:
        _fail(ErrorCode.RELATION, "process_uuid", "must differ from run_uuid")

    algorithm = source_object_algorithm
    length = _uint(value.source_object_length, 16, "source_object_length")
    if algorithm == 1:
        if length != 20:
            _fail(ErrorCode.RELATION, "source_object_length", "GIT_SHA1 requires 20")
    elif algorithm == 2:
        if length != 32:
            _fail(ErrorCode.RELATION, "source_object_length", "GIT_SHA256 requires 32")
    source = _fixed_bytes(value.source_object_value, 32, "source_object_value")
    if not any(source[:length]):
        _fail(ErrorCode.RELATION, "source_object_value", "all-zero digest is a placeholder")
    if any(source[length:]):
        _fail(ErrorCode.RELATION, "source_object_value", "nonzero canonical tail")

    for name in (
        "primary_binary_sha256",
        "schema_bundle_definition_sha256",
        "api_catalog_definition_sha256",
        "callsite_catalog_definition_sha256",
        "configuration_instance_sha256",
        "primary_build_id_sha256",
    ):
        _nonzero_digest(getattr(value, name), name)


def encode_opening_header(value: OpeningHeader) -> bytes:
    if not isinstance(value, OpeningHeader):
        _fail(ErrorCode.TYPE, "opening_header")
    _validate_opening_header(value)
    out = bytearray(OPENING_HEADER_BYTES)
    _put_bytes(out, 0, 8, OPENING_MAGIC, "magic")
    _put_u(out, 8, 2, CONTAINER_MAJOR, "container_major")
    _put_u(out, 10, 2, CONTAINER_MINOR, "container_minor")
    _put_u(out, 12, 2, OPENING_HEADER_BYTES, "header_bytes")
    _put_u(out, 14, 2, EVENT_RECORD_BYTES, "event_record_bytes")
    _put_u(out, 16, 2, CHUNK_HEADER_BYTES, "chunk_header_bytes")
    _put_u(out, 18, 2, MINIMUM_READER_MINOR, "minimum_reader_minor")
    _put_u(out, 20, 4, REQUIRED_FEATURES, "required_features")
    _put_u(out, 24, 4, ENDIAN_MARKER, "endian_marker")
    _put_u(out, 28, 4, value.page_size_bytes, "page_size_bytes")
    _put_u(out, 32, 1, POINTER_WIDTH_BYTES, "pointer_width_bytes")
    _put_u(out, 33, 1, value.scope_kind, "scope_kind")
    _put_u(out, 34, 1, value.role_kind, "role_kind")
    _put_u(out, 35, 1, value.clock_kind, "clock_kind")
    _put_u(out, 36, 2, value.calibration_kind, "calibration_kind")
    _put_u(out, 40, 8, value.process_generation, "process_generation")
    _put_u(out, 48, 8, value.counter_frequency_numerator, "counter_frequency_numerator")
    _put_u(out, 56, 8, value.counter_frequency_denominator, "counter_frequency_denominator")
    _put_u(out, 64, 8, value.calibration_error_bound_ns, "calibration_error_bound_ns")
    _put_u(out, 72, 8, value.calibration_span_ns, "calibration_span_ns")
    _put_u(out, 80, 8, value.start_counter, "start_counter")
    _put_u(out, 88, 8, value.start_monotonic_raw_ns, "start_monotonic_raw_ns")
    _put_u(out, 96, 8, value.start_realtime_unix_ns, "start_realtime_unix_ns")
    _put_u(out, 104, 4, value.pid, "pid")
    _put_u(out, 108, 4, value.configured_thread_capacity, "configured_thread_capacity")
    _put_bytes(out, 112, 16, value.run_uuid, "run_uuid")
    _put_bytes(out, 128, 16, value.process_uuid, "process_uuid")
    _put_u(out, 144, 2, value.source_object_kind, "source_object_kind")
    _put_u(out, 146, 2, value.source_object_algorithm, "source_object_algorithm")
    _put_u(out, 148, 2, value.source_object_length, "source_object_length")
    _put_bytes(out, 152, 32, value.source_object_value, "source_object_value")
    _put_bytes(out, 184, 32, value.primary_binary_sha256, "primary_binary_sha256")
    _put_bytes(out, 216, 32, value.schema_bundle_definition_sha256, "schema_bundle_definition_sha256")
    _put_bytes(out, 248, 32, value.api_catalog_definition_sha256, "api_catalog_definition_sha256")
    _put_bytes(out, 280, 32, value.callsite_catalog_definition_sha256, "callsite_catalog_definition_sha256")
    _put_bytes(out, 312, 32, value.configuration_instance_sha256, "configuration_instance_sha256")
    _put_bytes(out, 344, 32, value.primary_build_id_sha256, "primary_build_id_sha256")
    _put_u(out, 508, 4, crc32c(out[:508]), "header_crc32c")
    return bytes(out)


def decode_opening_header(data: object) -> OpeningHeader:
    raw = _exact_input(data, OPENING_HEADER_BYTES, "opening_header")
    _expect_bytes(raw, 0, OPENING_MAGIC, "magic", ErrorCode.BAD_MAGIC)
    if _read_u(raw, 508, 4) != crc32c(raw[:508]):
        _fail(ErrorCode.CHECKSUM, "header_crc32c")
    if _read_u(raw, 8, 2) != CONTAINER_MAJOR or _read_u(raw, 10, 2) != CONTAINER_MINOR:
        _fail(ErrorCode.VERSION, "container_version")
    _expect_u(raw, 18, 2, MINIMUM_READER_MINOR, "minimum_reader_minor", ErrorCode.VERSION)
    _expect_u(raw, 12, 2, OPENING_HEADER_BYTES, "header_bytes", ErrorCode.FIXED_VALUE)
    _expect_u(raw, 14, 2, EVENT_RECORD_BYTES, "event_record_bytes", ErrorCode.FIXED_VALUE)
    _expect_u(raw, 16, 2, CHUNK_HEADER_BYTES, "chunk_header_bytes", ErrorCode.FIXED_VALUE)
    _expect_u(raw, 24, 4, ENDIAN_MARKER, "endian_marker", ErrorCode.FIXED_VALUE)
    _expect_u(raw, 32, 1, POINTER_WIDTH_BYTES, "pointer_width_bytes", ErrorCode.FIXED_VALUE)
    _expect_u(raw, 20, 4, REQUIRED_FEATURES, "required_features", ErrorCode.FIXED_VALUE)
    _expect_zero(raw, 38, 2, "reserved_zero_0")
    _expect_zero(raw, 150, 2, "reserved_zero_1")
    _expect_zero(raw, 376, 132, "reserved_zero_2")
    value = OpeningHeader(
        page_size_bytes=_read_u(raw, 28, 4),
        scope_kind=_read_u(raw, 33, 1),
        role_kind=_read_u(raw, 34, 1),
        clock_kind=_read_u(raw, 35, 1),
        calibration_kind=_read_u(raw, 36, 2),
        process_generation=_read_u(raw, 40, 8),
        counter_frequency_numerator=_read_u(raw, 48, 8),
        counter_frequency_denominator=_read_u(raw, 56, 8),
        calibration_error_bound_ns=_read_u(raw, 64, 8),
        calibration_span_ns=_read_u(raw, 72, 8),
        start_counter=_read_u(raw, 80, 8),
        start_monotonic_raw_ns=_read_u(raw, 88, 8),
        start_realtime_unix_ns=_read_u(raw, 96, 8),
        pid=_read_u(raw, 104, 4),
        configured_thread_capacity=_read_u(raw, 108, 4),
        run_uuid=raw[112:128],
        process_uuid=raw[128:144],
        source_object_kind=_read_u(raw, 144, 2),
        source_object_algorithm=_read_u(raw, 146, 2),
        source_object_length=_read_u(raw, 148, 2),
        source_object_value=raw[152:184],
        primary_binary_sha256=raw[184:216],
        schema_bundle_definition_sha256=raw[216:248],
        api_catalog_definition_sha256=raw[248:280],
        callsite_catalog_definition_sha256=raw[280:312],
        configuration_instance_sha256=raw[312:344],
        primary_build_id_sha256=raw[344:376],
    )
    _validate_opening_header(value)
    return value


def _validate_event(value: EventRecord) -> None:
    for name in (
        "thread_sequence",
        "counter_enter",
        "counter_exit",
        "address_before",
        "address_after",
        "arg0",
        "arg1",
        "arg2",
    ):
        _uint(getattr(value, name), 64, name)
    for name in ("context_id", "callsite_id", "thread_index", "flags"):
        _uint(getattr(value, name), 32, name)
    _sint(value.result_code, 32, "result_code")
    for name in ("api_id", "event_kind", "cpu_enter", "cpu_exit"):
        _uint(getattr(value, name), 16, name)


def encode_event_record(value: EventRecord) -> bytes:
    if not isinstance(value, EventRecord):
        _fail(ErrorCode.TYPE, "event_record")
    _validate_event(value)
    out = bytearray(EVENT_RECORD_BYTES)
    for offset, name in (
        (0, "thread_sequence"),
        (8, "counter_enter"),
        (16, "counter_exit"),
        (24, "address_before"),
        (32, "address_after"),
        (40, "arg0"),
        (48, "arg1"),
        (56, "arg2"),
    ):
        _put_u(out, offset, 8, getattr(value, name), name)
    for offset, name in ((64, "context_id"), (68, "callsite_id"), (72, "thread_index"), (76, "flags")):
        _put_u(out, offset, 4, getattr(value, name), name)
    _put_i(out, 80, 4, value.result_code, "result_code")
    for offset, name in ((84, "api_id"), (86, "event_kind"), (88, "cpu_enter"), (90, "cpu_exit")):
        _put_u(out, offset, 2, getattr(value, name), name)
    return bytes(out)


def decode_event_record(data: object) -> EventRecord:
    raw = _exact_input(data, EVENT_RECORD_BYTES, "event_record")
    _expect_zero(raw, 92, 4, "reserved_zero")
    value = EventRecord(
        thread_sequence=_read_u(raw, 0, 8),
        counter_enter=_read_u(raw, 8, 8),
        counter_exit=_read_u(raw, 16, 8),
        address_before=_read_u(raw, 24, 8),
        address_after=_read_u(raw, 32, 8),
        arg0=_read_u(raw, 40, 8),
        arg1=_read_u(raw, 48, 8),
        arg2=_read_u(raw, 56, 8),
        context_id=_read_u(raw, 64, 4),
        callsite_id=_read_u(raw, 68, 4),
        thread_index=_read_u(raw, 72, 4),
        flags=_read_u(raw, 76, 4),
        result_code=_read_i(raw, 80, 4),
        api_id=_read_u(raw, 84, 2),
        event_kind=_read_u(raw, 86, 2),
        cpu_enter=_read_u(raw, 88, 2),
        cpu_exit=_read_u(raw, 90, 2),
    )
    _validate_event(value)
    return value


def _validate_chunk_header(value: ChunkHeader, max_records: int = MAX_CHUNK_RECORD_COUNT) -> None:
    sequence = _uint(value.writer_chunk_sequence, 64, "writer_chunk_sequence")
    del sequence
    count = _uint(value.record_count, 32, "record_count")
    configured_max = _uint(max_records, 32, "max_records")
    if configured_max == 0 or configured_max > MAX_CHUNK_RECORD_COUNT:
        _fail(ErrorCode.RANGE, "max_records")
    if count == 0 or count > MAX_CHUNK_RECORD_COUNT or count > configured_max:
        _fail(ErrorCode.RELATION, "record_count", "empty or above configured/wire bound")
    expected_payload = _checked_mul(count, EVENT_RECORD_BYTES, UINT32_MAX, "payload_bytes")
    if _uint(value.payload_bytes, 32, "payload_bytes") != expected_payload:
        _fail(ErrorCode.RELATION, "payload_bytes", "must equal record_count * 96")
    _uint(value.payload_crc32c, 32, "payload_crc32c")


def encode_chunk_header(value: ChunkHeader, max_records: int = MAX_CHUNK_RECORD_COUNT) -> bytes:
    if not isinstance(value, ChunkHeader):
        _fail(ErrorCode.TYPE, "chunk_header")
    _validate_chunk_header(value, max_records)
    out = bytearray(CHUNK_HEADER_BYTES)
    _put_bytes(out, 0, 4, CHUNK_MAGIC, "magic")
    _put_u(out, 4, 1, CONTAINER_MAJOR, "chunk_major")
    _put_u(out, 5, 1, CONTAINER_MINOR, "chunk_minor")
    _put_u(out, 6, 2, CHUNK_HEADER_BYTES, "header_bytes")
    _put_u(out, 8, 8, value.writer_chunk_sequence, "writer_chunk_sequence")
    _put_u(out, 16, 4, value.record_count, "record_count")
    _put_u(out, 20, 4, value.payload_bytes, "payload_bytes")
    _put_u(out, 24, 4, value.payload_crc32c, "payload_crc32c")
    return bytes(out)


def decode_chunk_header(data: object, max_records: int = MAX_CHUNK_RECORD_COUNT) -> ChunkHeader:
    raw = _exact_input(data, CHUNK_HEADER_BYTES, "chunk_header")
    _expect_bytes(raw, 0, CHUNK_MAGIC, "magic", ErrorCode.BAD_MAGIC)
    if _read_u(raw, 4, 1) != CONTAINER_MAJOR or _read_u(raw, 5, 1) != CONTAINER_MINOR:
        _fail(ErrorCode.VERSION, "chunk_version")
    _expect_u(raw, 6, 2, CHUNK_HEADER_BYTES, "header_bytes", ErrorCode.FIXED_VALUE)
    _expect_u(raw, 28, 4, 0, "flags", ErrorCode.FIXED_VALUE)
    value = ChunkHeader(
        writer_chunk_sequence=_read_u(raw, 8, 8),
        record_count=_read_u(raw, 16, 4),
        payload_bytes=_read_u(raw, 20, 4),
        payload_crc32c=_read_u(raw, 24, 4),
    )
    _validate_chunk_header(value, max_records)
    return value


def encode_chunk(writer_chunk_sequence: int, records: Sequence[EventRecord]) -> bytes:
    _uint(writer_chunk_sequence, 64, "writer_chunk_sequence")
    if isinstance(records, (bytes, bytearray, memoryview, str)) or not isinstance(records, Sequence):
        _fail(ErrorCode.TYPE, "records", "expected a sequence of EventRecord")
    if not records or len(records) > MAX_CHUNK_RECORD_COUNT:
        _fail(ErrorCode.RELATION, "records", "chunk must contain 1..MAX records")
    payload_parts = []
    for index, record in enumerate(records):
        if not isinstance(record, EventRecord):
            _fail(ErrorCode.TYPE, "records[%d]" % index)
        payload_parts.append(encode_event_record(record))
    payload = b"".join(payload_parts)
    header = ChunkHeader(writer_chunk_sequence, len(records), len(payload), crc32c(payload))
    return encode_chunk_header(header) + payload


def decode_chunk(
    data: object,
    expected_sequence: Optional[int] = None,
    max_records: int = MAX_CHUNK_RECORD_COUNT,
) -> Chunk:
    raw = _bytes_like(data, "chunk")
    if len(raw) < CHUNK_HEADER_BYTES:
        _fail(ErrorCode.TRUNCATED, "chunk_header")
    header = decode_chunk_header(raw[:CHUNK_HEADER_BYTES], max_records)
    total = CHUNK_HEADER_BYTES + header.payload_bytes
    if len(raw) < total:
        _fail(ErrorCode.TRUNCATED, "chunk_payload")
    if len(raw) > total:
        _fail(ErrorCode.OVERLONG, "chunk")
    payload = raw[CHUNK_HEADER_BYTES:]
    if crc32c(payload) != header.payload_crc32c:
        _fail(ErrorCode.CHECKSUM, "payload_crc32c")
    if expected_sequence is not None:
        expected = _uint(expected_sequence, 64, "expected_sequence")
        if header.writer_chunk_sequence != expected:
            _fail(ErrorCode.SEQUENCE, "writer_chunk_sequence")
    records = tuple(
        decode_event_record(payload[offset : offset + EVENT_RECORD_BYTES])
        for offset in range(0, len(payload), EVENT_RECORD_BYTES)
    )
    return Chunk(header, records)


def _trailer_offsets(event_count: int, diagnostic_count: int, object_count: int) -> Tuple[int, int, int, int]:
    event_offset = TRAILER_HEADER_BYTES
    diagnostic_offset = event_offset + event_count * EVENT_TOTAL_ENTRY_BYTES
    object_offset = diagnostic_offset + diagnostic_count * DIAGNOSTIC_TOTAL_ENTRY_BYTES
    body_bytes = object_offset + object_count * OBJECT_BINDING_ENTRY_BYTES
    return event_offset, diagnostic_offset, object_offset, body_bytes


def _prefix_layout(chunk_count: int, record_count: int) -> Tuple[int, int]:
    chunks = _uint(chunk_count, 64, "chunk_count")
    records = _uint(record_count, 64, "record_count")
    if (chunks == 0) != (records == 0):
        _fail(ErrorCode.RELATION, "chunk_count", "zero chunk and record counts must agree")
    if chunks > records:
        _fail(ErrorCode.RELATION, "chunk_count", "cannot exceed record_count")
    chunk_header_bytes = _checked_mul(chunks, CHUNK_HEADER_BYTES, UINT64_MAX, "chunk_header_bytes")
    payload_bytes = _checked_mul(records, EVENT_RECORD_BYTES, UINT64_MAX, "payload_bytes")
    prefix_bytes = _checked_add(OPENING_HEADER_BYTES, chunk_header_bytes, UINT64_MAX, "prefix_bytes")
    prefix_bytes = _checked_add(prefix_bytes, payload_bytes, UINT64_MAX, "prefix_bytes")
    return prefix_bytes, payload_bytes


def _validate_trailer_header_grammar(value: TrailerHeader) -> None:
    for name in (
        "trailer_body_bytes",
        "process_generation",
        "terminal_flags",
        "chunk_count",
        "record_count",
        "payload_bytes",
        "first_chunk_offset",
        "chunks_end_offset",
        "active_generation",
        "active_start_counter",
        "cutoff_before_counter",
        "cutoff_after_counter",
        "quiescence_complete_counter",
        "final_counter",
        "active_start_monotonic_raw_ns",
        "final_monotonic_raw_ns",
        "final_realtime_unix_ns",
        "event_table_offset",
        "diagnostic_table_offset",
        "object_table_offset",
        "diagnostic_loss_sum",
        "diagnostic_bypass_sum",
        "saturated_counter_instances",
    ):
        _uint(getattr(value, name), 64, name)
    for name in ("event_entry_count", "diagnostic_entry_count", "object_entry_count", "terminal_reason_code"):
        _uint(getattr(value, name), 32, name)
    for name in ("scope_kind", "lifecycle_state", "payload_writer_state", "finalization_stage"):
        _uint(getattr(value, name), 16, name)

    if value.process_generation == 0:
        _fail(ErrorCode.RELATION, "process_generation", "must be nonzero")
    if value.scope_kind not in (1, 2):
        _fail(ErrorCode.ENUM, "scope_kind")
    if value.lifecycle_state not in (5, 6, 7):
        _fail(ErrorCode.ENUM, "lifecycle_state", "terminal body requires COMPLETE, FAILED, or INCOMPLETE")
    if value.payload_writer_state not in (5, 6):
        _fail(ErrorCode.ENUM, "payload_writer_state", "trusted terminal requires verified prefix closure")
    if value.finalization_stage > 6:
        _fail(ErrorCode.ENUM, "finalization_stage")
    if value.terminal_reason_code > 9:
        _fail(ErrorCode.ENUM, "terminal_reason_code")
    if value.terminal_flags & ~TERMINAL_FLAGS_MASK:
        _fail(ErrorCode.FIXED_VALUE, "terminal_flags", "unknown flag bit")

    if value.event_entry_count > MAX_EVENT_ENTRIES:
        _fail(ErrorCode.RANGE, "event_entry_count")
    if value.diagnostic_entry_count > MAX_DIAGNOSTIC_ENTRIES:
        _fail(ErrorCode.RANGE, "diagnostic_entry_count")
    if value.object_entry_count > MAX_OBJECT_ENTRIES:
        _fail(ErrorCode.RANGE, "object_entry_count")
    expected_offsets = _trailer_offsets(
        value.event_entry_count, value.diagnostic_entry_count, value.object_entry_count
    )
    actual_offsets = (
        value.event_table_offset,
        value.diagnostic_table_offset,
        value.object_table_offset,
        value.trailer_body_bytes,
    )
    if actual_offsets != expected_offsets:
        _fail(ErrorCode.RELATION, "trailer_offsets", "table offsets/body size do not match exact arithmetic")
    if value.trailer_body_bytes < TRAILER_HEADER_BYTES or value.trailer_body_bytes > MAX_TRAILER_BODY_BYTES:
        _fail(ErrorCode.RANGE, "trailer_body_bytes")


def _validate_trailer_payload_layout(value: TrailerHeader) -> None:
    expected_prefix, expected_payload = _prefix_layout(value.chunk_count, value.record_count)
    if value.payload_bytes != expected_payload:
        _fail(ErrorCode.RELATION, "payload_bytes", "must equal record_count * 96")
    if value.first_chunk_offset != OPENING_HEADER_BYTES:
        _fail(ErrorCode.FIXED_VALUE, "first_chunk_offset")
    if value.chunks_end_offset != expected_prefix:
        _fail(ErrorCode.RELATION, "chunks_end_offset", "must equal 512 + chunk_count * 32 + record_count * 96")
    if value.record_count == 0:
        if value.event_entry_count != 0:
            _fail(ErrorCode.RELATION, "event_entry_count", "must be zero with zero records")
    elif value.event_entry_count == 0 or value.event_entry_count > min(value.record_count, MAX_EVENT_ENTRIES):
        _fail(ErrorCode.RELATION, "event_entry_count")


def _validate_trailer_finalization(value: TrailerHeader) -> None:
    required_stage_flags = (1 << (value.finalization_stage + 1)) - 1
    if value.terminal_flags & 0x7F != required_stage_flags:
        _fail(ErrorCode.RELATION, "finalization_stage", "stage-governed flags disagree with causal ordinal")
    if value.payload_writer_state == 5:
        if value.terminal_flags & ((1 << 7) | (1 << 8)) != ((1 << 7) | (1 << 8)):
            _fail(ErrorCode.RELATION, "payload_writer_state", "state 5 requires sync and close flags")
    elif not value.terminal_flags & (1 << 8) or value.terminal_flags & (1 << 7):
        _fail(ErrorCode.RELATION, "payload_writer_state", "state 6 requires close and forbids sync")

    if value.active_generation != value.process_generation:
        _fail(ErrorCode.RELATION, "active_generation")
    if value.active_start_counter > value.final_counter:
        _fail(ErrorCode.RELATION, "final_counter")
    if value.active_start_monotonic_raw_ns > value.final_monotonic_raw_ns:
        _fail(ErrorCode.RELATION, "final_monotonic_raw_ns")
    if value.finalization_stage < 1 and (value.cutoff_before_counter or value.cutoff_after_counter):
        _fail(ErrorCode.RELATION, "cutoff_counter", "unreached fields must be canonical zero")
    if value.finalization_stage < 2 and value.quiescence_complete_counter:
        _fail(ErrorCode.RELATION, "quiescence_complete_counter", "unreached field must be canonical zero")
    if value.finalization_stage >= 1 and not (
        value.active_start_counter <= value.cutoff_before_counter <= value.cutoff_after_counter <= value.final_counter
    ):
        _fail(ErrorCode.RELATION, "cutoff_counter")
    if value.finalization_stage >= 2 and not (
        value.cutoff_after_counter <= value.quiescence_complete_counter <= value.final_counter
    ):
        _fail(ErrorCode.RELATION, "quiescence_complete_counter")

    if value.lifecycle_state == 5:
        if value.scope_kind != 1:
            _fail(ErrorCode.RELATION, "scope_kind", "current PROCESS_LIFETIME is not COMPLETE-capable")
        if value.payload_writer_state != 5 or value.finalization_stage != 6:
            _fail(ErrorCode.RELATION, "complete_finalization")
        if value.terminal_reason_code != 0:
            _fail(ErrorCode.RELATION, "terminal_reason_code", "COMPLETE requires NONE")
        if value.terminal_flags & 0xFFF != 0xFFF:
            _fail(ErrorCode.RELATION, "terminal_flags", "COMPLETE requires bits 0..11")
        if value.terminal_flags & ((1 << 13) | (1 << 16)):
            _fail(ErrorCode.RELATION, "terminal_flags", "COMPLETE forbids partial diagnostics/unverified tail")
    else:
        fixed_outcomes = {
            1: (7, 1, 5),
            2: (6, 2, 5),
            3: (6, 3, 5),
            4: (6, 4, 5),
            5: (6, 6, 6),
            9: (7, 0, 5),
        }
        expected = fixed_outcomes.get(value.terminal_reason_code)
        if expected is not None and (
            value.lifecycle_state, value.finalization_stage, value.payload_writer_state
        ) != expected:
            _fail(ErrorCode.RELATION, "terminal_reason_code", "reason/stage/writer tuple is unreachable")
        ranged_outcomes = {
            6: (6, (6,)),
            7: (7, (5, 6)),
            8: (7, (5, 6)),
        }
        ranged = ranged_outcomes.get(value.terminal_reason_code)
        if ranged is not None and (
            value.lifecycle_state != ranged[0] or value.payload_writer_state not in ranged[1]
        ):
            _fail(ErrorCode.RELATION, "terminal_reason_code", "reason/stage/writer tuple is unreachable")
        if expected is None and ranged is None:
            _fail(ErrorCode.RELATION, "terminal_reason_code", "reason/lifecycle pair is unreachable")


def _validate_trailer_header(value: TrailerHeader) -> None:
    _validate_trailer_header_grammar(value)
    _validate_trailer_payload_layout(value)
    _validate_trailer_finalization(value)


def encode_trailer_header(value: TrailerHeader) -> bytes:
    if not isinstance(value, TrailerHeader):
        _fail(ErrorCode.TYPE, "trailer_header")
    _validate_trailer_header(value)
    out = bytearray(TRAILER_HEADER_BYTES)
    _put_bytes(out, 0, 16, TRAILER_MAGIC, "magic")
    _put_u(out, 16, 2, CONTAINER_MAJOR, "schema_major")
    _put_u(out, 18, 2, CONTAINER_MINOR, "schema_minor")
    _put_u(out, 20, 4, TRAILER_HEADER_BYTES, "fixed_header_bytes")
    fields = (
        (24, 8, "trailer_body_bytes"),
        (32, 8, "process_generation"),
        (40, 2, "scope_kind"),
        (42, 2, "lifecycle_state"),
        (44, 2, "payload_writer_state"),
        (46, 2, "finalization_stage"),
        (48, 8, "terminal_flags"),
        (56, 8, "chunk_count"),
        (64, 8, "record_count"),
        (72, 8, "payload_bytes"),
        (80, 8, "first_chunk_offset"),
        (88, 8, "chunks_end_offset"),
        (96, 8, "active_generation"),
        (104, 8, "active_start_counter"),
        (112, 8, "cutoff_before_counter"),
        (120, 8, "cutoff_after_counter"),
        (128, 8, "quiescence_complete_counter"),
        (136, 8, "final_counter"),
        (144, 8, "active_start_monotonic_raw_ns"),
        (152, 8, "final_monotonic_raw_ns"),
        (160, 8, "final_realtime_unix_ns"),
        (168, 4, "event_entry_count"),
        (176, 4, "diagnostic_entry_count"),
        (184, 4, "object_entry_count"),
        (192, 8, "event_table_offset"),
        (200, 8, "diagnostic_table_offset"),
        (208, 8, "object_table_offset"),
        (216, 4, "terminal_reason_code"),
        (224, 8, "diagnostic_loss_sum"),
        (232, 8, "diagnostic_bypass_sum"),
        (240, 8, "saturated_counter_instances"),
    )
    for offset, size, name in fields:
        _put_u(out, offset, size, getattr(value, name), name)
    _put_u(out, 172, 4, EVENT_TOTAL_ENTRY_BYTES, "event_entry_bytes")
    _put_u(out, 180, 4, DIAGNOSTIC_TOTAL_ENTRY_BYTES, "diagnostic_entry_bytes")
    _put_u(out, 188, 4, OBJECT_BINDING_ENTRY_BYTES, "object_entry_bytes")
    return bytes(out)


def _decode_trailer_header_grammar(data: object) -> TrailerHeader:
    raw = _exact_input(data, TRAILER_HEADER_BYTES, "trailer_header")
    _expect_bytes(raw, 0, TRAILER_MAGIC, "magic", ErrorCode.BAD_MAGIC)
    if _read_u(raw, 16, 2) != CONTAINER_MAJOR or _read_u(raw, 18, 2) != CONTAINER_MINOR:
        _fail(ErrorCode.VERSION, "trailer_version")
    _expect_u(raw, 20, 4, TRAILER_HEADER_BYTES, "fixed_header_bytes", ErrorCode.FIXED_VALUE)
    _expect_u(raw, 172, 4, EVENT_TOTAL_ENTRY_BYTES, "event_entry_bytes", ErrorCode.FIXED_VALUE)
    _expect_u(raw, 180, 4, DIAGNOSTIC_TOTAL_ENTRY_BYTES, "diagnostic_entry_bytes", ErrorCode.FIXED_VALUE)
    _expect_u(raw, 188, 4, OBJECT_BINDING_ENTRY_BYTES, "object_entry_bytes", ErrorCode.FIXED_VALUE)
    _expect_zero(raw, 220, 4, "reserved_zero_0")
    _expect_zero(raw, 248, 8, "reserved_zero_1")
    value = TrailerHeader(
        trailer_body_bytes=_read_u(raw, 24, 8),
        process_generation=_read_u(raw, 32, 8),
        scope_kind=_read_u(raw, 40, 2),
        lifecycle_state=_read_u(raw, 42, 2),
        payload_writer_state=_read_u(raw, 44, 2),
        finalization_stage=_read_u(raw, 46, 2),
        terminal_flags=_read_u(raw, 48, 8),
        chunk_count=_read_u(raw, 56, 8),
        record_count=_read_u(raw, 64, 8),
        payload_bytes=_read_u(raw, 72, 8),
        first_chunk_offset=_read_u(raw, 80, 8),
        chunks_end_offset=_read_u(raw, 88, 8),
        active_generation=_read_u(raw, 96, 8),
        active_start_counter=_read_u(raw, 104, 8),
        cutoff_before_counter=_read_u(raw, 112, 8),
        cutoff_after_counter=_read_u(raw, 120, 8),
        quiescence_complete_counter=_read_u(raw, 128, 8),
        final_counter=_read_u(raw, 136, 8),
        active_start_monotonic_raw_ns=_read_u(raw, 144, 8),
        final_monotonic_raw_ns=_read_u(raw, 152, 8),
        final_realtime_unix_ns=_read_u(raw, 160, 8),
        event_entry_count=_read_u(raw, 168, 4),
        diagnostic_entry_count=_read_u(raw, 176, 4),
        object_entry_count=_read_u(raw, 184, 4),
        event_table_offset=_read_u(raw, 192, 8),
        diagnostic_table_offset=_read_u(raw, 200, 8),
        object_table_offset=_read_u(raw, 208, 8),
        terminal_reason_code=_read_u(raw, 216, 4),
        diagnostic_loss_sum=_read_u(raw, 224, 8),
        diagnostic_bypass_sum=_read_u(raw, 232, 8),
        saturated_counter_instances=_read_u(raw, 240, 8),
    )
    _validate_trailer_header_grammar(value)
    return value


def decode_trailer_header(data: object) -> TrailerHeader:
    value = _decode_trailer_header_grammar(data)
    _validate_trailer_payload_layout(value)
    _validate_trailer_finalization(value)
    return value


def _validate_event_total_entry(value: EventTotalEntry) -> None:
    _uint(value.event_kind, 16, "event_kind")
    _uint(value.api_id, 16, "api_id")
    if _uint(value.record_count, 64, "record_count") == 0:
        _fail(ErrorCode.RELATION, "record_count", "event total must be nonzero")


def encode_event_total_entry(value: EventTotalEntry) -> bytes:
    if not isinstance(value, EventTotalEntry):
        _fail(ErrorCode.TYPE, "event_total_entry")
    _validate_event_total_entry(value)
    out = bytearray(EVENT_TOTAL_ENTRY_BYTES)
    _put_u(out, 0, 2, value.event_kind, "event_kind")
    _put_u(out, 2, 2, value.api_id, "api_id")
    _put_u(out, 8, 8, value.record_count, "record_count")
    return bytes(out)


def _decode_event_total_entry_grammar(data: object) -> EventTotalEntry:
    raw = _exact_input(data, EVENT_TOTAL_ENTRY_BYTES, "event_total_entry")
    _expect_zero(raw, 4, 4, "reserved_zero_0")
    _expect_zero(raw, 16, 16, "reserved_zero_1")
    return EventTotalEntry(_read_u(raw, 0, 2), _read_u(raw, 2, 2), _read_u(raw, 8, 8))


def decode_event_total_entry(data: object) -> EventTotalEntry:
    value = _decode_event_total_entry_grammar(data)
    _validate_event_total_entry(value)
    return value


def _validate_diagnostic_total_entry(value: DiagnosticTotalEntry) -> None:
    _uint(value.reason_id, 16, "reason_id")
    class_flags = _uint(value.class_flags, 16, "class_flags")
    summary_flags = _uint(value.summary_flags, 32, "summary_flags")
    _uint(value.saturating_total, 64, "saturating_total")
    nonzero = _uint(value.nonzero_counter_instances, 32, "nonzero_counter_instances")
    saturated = _uint(value.saturated_counter_instances, 32, "saturated_counter_instances")
    if class_flags & ~DIAGNOSTIC_CLASS_FLAGS_MASK:
        _fail(ErrorCode.FIXED_VALUE, "class_flags", "unknown class bit")
    if summary_flags & ~DIAGNOSTIC_SUMMARY_FLAGS_MASK:
        _fail(ErrorCode.FIXED_VALUE, "summary_flags", "unknown summary bit")
    if saturated > nonzero:
        _fail(ErrorCode.RELATION, "saturated_counter_instances")
    if saturated and not summary_flags & 1:
        _fail(ErrorCode.RELATION, "summary_flags", "saturated input requires TOTAL_SATURATED")


def encode_diagnostic_total_entry(value: DiagnosticTotalEntry) -> bytes:
    if not isinstance(value, DiagnosticTotalEntry):
        _fail(ErrorCode.TYPE, "diagnostic_total_entry")
    _validate_diagnostic_total_entry(value)
    out = bytearray(DIAGNOSTIC_TOTAL_ENTRY_BYTES)
    _put_u(out, 0, 2, value.reason_id, "reason_id")
    _put_u(out, 2, 2, value.class_flags, "class_flags")
    _put_u(out, 4, 4, value.summary_flags, "summary_flags")
    _put_u(out, 8, 8, value.saturating_total, "saturating_total")
    _put_u(out, 16, 4, value.nonzero_counter_instances, "nonzero_counter_instances")
    _put_u(out, 20, 4, value.saturated_counter_instances, "saturated_counter_instances")
    return bytes(out)


def _decode_diagnostic_total_entry_grammar(data: object) -> DiagnosticTotalEntry:
    raw = _exact_input(data, DIAGNOSTIC_TOTAL_ENTRY_BYTES, "diagnostic_total_entry")
    _expect_zero(raw, 24, 8, "reserved_zero")
    return DiagnosticTotalEntry(
        reason_id=_read_u(raw, 0, 2),
        class_flags=_read_u(raw, 2, 2),
        summary_flags=_read_u(raw, 4, 4),
        saturating_total=_read_u(raw, 8, 8),
        nonzero_counter_instances=_read_u(raw, 16, 4),
        saturated_counter_instances=_read_u(raw, 20, 4),
    )


def decode_diagnostic_total_entry(data: object) -> DiagnosticTotalEntry:
    value = _decode_diagnostic_total_entry_grammar(data)
    _validate_diagnostic_total_entry(value)
    return value


_OBJECT_FLAGS_BY_KIND = {
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


def _validate_object_binding_entry(value: ObjectBindingEntry) -> None:
    kind = _uint(value.object_kind, 16, "object_kind")
    if kind < 1 or kind > 12:
        _fail(ErrorCode.ENUM, "object_kind")
    if _uint(value.format_id, 16, "format_id") != 1:
        _fail(ErrorCode.FIXED_VALUE, "format_id")
    flags = _uint(value.object_flags, 32, "object_flags")
    if flags & ~OBJECT_FLAGS_MASK:
        _fail(ErrorCode.FIXED_VALUE, "object_flags", "unknown flag bit")
    if _uint(value.schema_revision, 32, "schema_revision") != 1:
        _fail(ErrorCode.FIXED_VALUE, "schema_revision")
    entry_count = _uint(value.entry_count, 64, "entry_count")
    byte_count = _uint(value.byte_count, 64, "byte_count")
    if entry_count > MAX_OBJECT_ENTRY_COUNT:
        _fail(ErrorCode.RANGE, "entry_count")
    if byte_count > MAX_OBJECT_BYTE_COUNT:
        _fail(ErrorCode.RANGE, "byte_count")
    _nonzero_digest(value.sha256, "sha256")
    if flags != _OBJECT_FLAGS_BY_KIND[kind]:
        _fail(ErrorCode.FIXED_VALUE, "object_flags", "kind fixes one exact flag value")
    if kind == 1 and entry_count < 7:
        _fail(ErrorCode.RELATION, "entry_count", "schema bundle requires at least 7 entries")
    if kind == 2 and entry_count != 12:
        _fail(ErrorCode.RELATION, "entry_count", "API slice requires exactly 12 entries")
    if kind in (6, 7, 8, 9, 11) and entry_count < 1:
        _fail(ErrorCode.RELATION, "entry_count", "object kind requires at least one entry")
    if kind in (10, 12) and entry_count != 1:
        _fail(ErrorCode.RELATION, "entry_count", "object kind requires exactly one entry")


def encode_object_binding_entry(value: ObjectBindingEntry) -> bytes:
    if not isinstance(value, ObjectBindingEntry):
        _fail(ErrorCode.TYPE, "object_binding_entry")
    _validate_object_binding_entry(value)
    out = bytearray(OBJECT_BINDING_ENTRY_BYTES)
    _put_u(out, 0, 2, value.object_kind, "object_kind")
    _put_u(out, 2, 2, value.format_id, "format_id")
    _put_u(out, 4, 4, value.object_flags, "object_flags")
    _put_u(out, 8, 4, value.schema_revision, "schema_revision")
    _put_u(out, 16, 8, value.entry_count, "entry_count")
    _put_u(out, 24, 8, value.byte_count, "byte_count")
    _put_bytes(out, 32, 32, value.sha256, "sha256")
    return bytes(out)


def _decode_object_binding_entry_grammar(data: object) -> ObjectBindingEntry:
    raw = _exact_input(data, OBJECT_BINDING_ENTRY_BYTES, "object_binding_entry")
    _expect_zero(raw, 12, 4, "reserved_zero")
    return ObjectBindingEntry(
        object_kind=_read_u(raw, 0, 2),
        format_id=_read_u(raw, 2, 2),
        object_flags=_read_u(raw, 4, 4),
        schema_revision=_read_u(raw, 8, 4),
        entry_count=_read_u(raw, 16, 8),
        byte_count=_read_u(raw, 24, 8),
        sha256=raw[32:64],
    )


def decode_object_binding_entry(data: object) -> ObjectBindingEntry:
    value = _decode_object_binding_entry_grammar(data)
    _validate_object_binding_entry(value)
    return value


def _saturating_sum(values: Iterable[int]) -> int:
    total = 0
    for value in values:
        total += value
        if total > UINT64_MAX:
            return UINT64_MAX
    return total


def _validate_trailer_table_counts(body: TrailerBody) -> None:
    header = body.header
    if len(body.event_entries) != header.event_entry_count:
        _fail(ErrorCode.RELATION, "event_entry_count")
    if len(body.diagnostic_entries) != header.diagnostic_entry_count:
        _fail(ErrorCode.RELATION, "diagnostic_entry_count")
    if len(body.object_entries) != header.object_entry_count:
        _fail(ErrorCode.RELATION, "object_entry_count")


def _validate_event_table(body: TrailerBody) -> None:
    header = body.header
    previous_event = None
    event_sum = 0
    for entry in body.event_entries:
        if not isinstance(entry, EventTotalEntry):
            _fail(ErrorCode.TYPE, "event_entries")
        _validate_event_total_entry(entry)
        key = (entry.event_kind, entry.api_id)
        if previous_event is not None and key <= previous_event:
            _fail(ErrorCode.TABLE_ORDER, "event_entries")
        previous_event = key
        event_sum = _checked_add(event_sum, entry.record_count, UINT64_MAX, "event_record_count_sum")
    if event_sum != header.record_count:
        _fail(ErrorCode.RELATION, "event_record_count_sum")


def _validate_diagnostic_table(body: TrailerBody) -> None:
    header = body.header
    previous_reason = None
    for entry in body.diagnostic_entries:
        if not isinstance(entry, DiagnosticTotalEntry):
            _fail(ErrorCode.TYPE, "diagnostic_entries")
        _validate_diagnostic_total_entry(entry)
        if previous_reason is not None and entry.reason_id <= previous_reason:
            _fail(ErrorCode.TABLE_ORDER, "diagnostic_entries")
        previous_reason = entry.reason_id
        if entry.saturated_counter_instances and not entry.summary_flags & 1:
            _fail(ErrorCode.RELATION, "summary_flags", "saturated input requires TOTAL_SATURATED")
    loss_values = [entry.saturating_total for entry in body.diagnostic_entries if entry.class_flags & (1 << 0)]
    bypass_values = [entry.saturating_total for entry in body.diagnostic_entries if entry.class_flags & (1 << 1)]
    loss = _saturating_sum(loss_values)
    bypass = _saturating_sum(bypass_values)
    saturated_instances = sum(entry.saturated_counter_instances for entry in body.diagnostic_entries)
    if loss != header.diagnostic_loss_sum:
        _fail(ErrorCode.RELATION, "diagnostic_loss_sum")
    if bypass != header.diagnostic_bypass_sum:
        _fail(ErrorCode.RELATION, "diagnostic_bypass_sum")
    if saturated_instances != header.saturated_counter_instances:
        _fail(ErrorCode.RELATION, "saturated_counter_instances")
    aggregate_saturated = (
        any(entry.summary_flags & 1 for entry in body.diagnostic_entries)
        or sum(loss_values) > UINT64_MAX
        or sum(bypass_values) > UINT64_MAX
    )
    if bool(header.terminal_flags & (1 << 12)) != aggregate_saturated:
        _fail(ErrorCode.RELATION, "terminal_flags", "DIAGNOSTIC_AGGREGATE_SATURATED does not reconcile")
    if any(entry.summary_flags & 2 for entry in body.diagnostic_entries) and not header.terminal_flags & (1 << 13):
        _fail(ErrorCode.RELATION, "terminal_flags", "partial diagnostic row requires DIAGNOSTIC_POPULATION_PARTIAL")


def _validate_object_table(body: TrailerBody) -> None:
    header = body.header
    previous_kind = None
    for entry in body.object_entries:
        if not isinstance(entry, ObjectBindingEntry):
            _fail(ErrorCode.TYPE, "object_entries")
        _validate_object_binding_entry(entry)
        if previous_kind is not None and entry.object_kind <= previous_kind:
            _fail(ErrorCode.TABLE_ORDER, "object_entries")
        previous_kind = entry.object_kind
    kinds = tuple(entry.object_kind for entry in body.object_entries)
    if header.lifecycle_state == 5:
        if kinds != tuple(range(1, 13)):
            _fail(ErrorCode.RELATION, "object_entries", "COMPLETE requires exactly kinds 1..12")
    else:
        if 1 not in kinds or 12 not in kinds:
            _fail(ErrorCode.RELATION, "object_entries", "FAILED/INCOMPLETE require kinds 1 and 12")
        if header.diagnostic_entry_count and 11 not in kinds:
            _fail(ErrorCode.RELATION, "object_entries", "diagnostics require kind 11")


def _validate_trailer_tables(body: TrailerBody) -> None:
    _validate_trailer_table_counts(body)
    _validate_event_table(body)
    _validate_diagnostic_table(body)
    _validate_object_table(body)


def encode_trailer_body(value: TrailerBody) -> bytes:
    if not isinstance(value, TrailerBody):
        _fail(ErrorCode.TYPE, "trailer_body")
    _validate_trailer_header(value.header)
    _validate_trailer_tables(value)
    parts = [encode_trailer_header(value.header)]
    parts.extend(encode_event_total_entry(entry) for entry in value.event_entries)
    parts.extend(encode_diagnostic_total_entry(entry) for entry in value.diagnostic_entries)
    parts.extend(encode_object_binding_entry(entry) for entry in value.object_entries)
    encoded = b"".join(parts)
    if len(encoded) != value.header.trailer_body_bytes:
        _fail(ErrorCode.RELATION, "trailer_body_bytes")
    return encoded


def _decode_trailer_body_grammar(data: object) -> TrailerBody:
    raw = _bytes_like(data, "trailer_body")
    if len(raw) < TRAILER_HEADER_BYTES:
        _fail(ErrorCode.TRUNCATED, "trailer_header")
    header = _decode_trailer_header_grammar(raw[:TRAILER_HEADER_BYTES])
    if len(raw) < header.trailer_body_bytes:
        _fail(ErrorCode.TRUNCATED, "trailer_body")
    if len(raw) > header.trailer_body_bytes:
        _fail(ErrorCode.OVERLONG, "trailer_body")
    event_entries = tuple(
        _decode_event_total_entry_grammar(raw[offset : offset + EVENT_TOTAL_ENTRY_BYTES])
        for offset in range(header.event_table_offset, header.diagnostic_table_offset, EVENT_TOTAL_ENTRY_BYTES)
    )
    diagnostic_entries = tuple(
        _decode_diagnostic_total_entry_grammar(raw[offset : offset + DIAGNOSTIC_TOTAL_ENTRY_BYTES])
        for offset in range(header.diagnostic_table_offset, header.object_table_offset, DIAGNOSTIC_TOTAL_ENTRY_BYTES)
    )
    object_entries = tuple(
        _decode_object_binding_entry_grammar(raw[offset : offset + OBJECT_BINDING_ENTRY_BYTES])
        for offset in range(header.object_table_offset, header.trailer_body_bytes, OBJECT_BINDING_ENTRY_BYTES)
    )
    value = TrailerBody(header, event_entries, diagnostic_entries, object_entries)
    _validate_trailer_table_counts(value)
    return value


def decode_trailer_body(data: object) -> TrailerBody:
    value = _decode_trailer_body_grammar(data)
    _validate_trailer_payload_layout(value.header)
    _validate_trailer_finalization(value.header)
    _validate_event_table(value)
    _validate_diagnostic_table(value)
    _validate_object_table(value)
    return value


def _validate_footer(value: Footer) -> None:
    for name in (
        "trailer_offset",
        "trailer_body_bytes",
        "stream_bytes",
        "prefix_bytes",
        "chunk_count",
        "record_count",
    ):
        _uint(getattr(value, name), 64, name)
    if (
        value.trailer_body_bytes < TRAILER_HEADER_BYTES
        or value.trailer_body_bytes > MAX_TRAILER_BODY_BYTES
        or value.trailer_body_bytes % 32
    ):
        _fail(ErrorCode.RANGE, "trailer_body_bytes")
    expected_prefix, unused_payload_bytes = _prefix_layout(value.chunk_count, value.record_count)
    del unused_payload_bytes
    if value.trailer_offset != expected_prefix:
        _fail(ErrorCode.RELATION, "trailer_offset", "must equal 512 + chunk_count * 32 + record_count * 96")
    if value.prefix_bytes != expected_prefix:
        _fail(ErrorCode.RELATION, "prefix_bytes")
    expected_stream = _checked_add(value.trailer_offset, value.trailer_body_bytes, UINT64_MAX, "stream_bytes")
    expected_stream = _checked_add(expected_stream, FOOTER_BYTES, UINT64_MAX, "stream_bytes")
    if value.stream_bytes != expected_stream:
        _fail(ErrorCode.RELATION, "stream_bytes")
    for name in ("prefix_sha256", "trailer_body_sha256", "opening_header_sha256"):
        _fixed_bytes(getattr(value, name), 32, name)
    footer_digest = _bytes_like(value.footer_sha256, "footer_sha256")
    if len(footer_digest) not in (0, 32):
        _fail(ErrorCode.WRONG_SIZE, "footer_sha256", "expected an empty sentinel or 32 bytes")


def encode_footer(value: Footer) -> bytes:
    if not isinstance(value, Footer):
        _fail(ErrorCode.TYPE, "footer")
    _validate_footer(value)
    out = bytearray(FOOTER_BYTES)
    _put_bytes(out, 0, 16, FOOTER_MAGIC, "magic")
    _put_u(out, 16, 2, CONTAINER_MAJOR, "schema_major")
    _put_u(out, 18, 2, CONTAINER_MINOR, "schema_minor")
    _put_u(out, 20, 4, FOOTER_BYTES, "footer_bytes")
    _put_u(out, 24, 8, FOOTER_FLAGS, "footer_flags")
    _put_u(out, 32, 8, value.trailer_offset, "trailer_offset")
    _put_u(out, 40, 8, value.trailer_body_bytes, "trailer_body_bytes")
    _put_u(out, 48, 8, value.stream_bytes, "stream_bytes")
    _put_u(out, 56, 8, value.prefix_bytes, "prefix_bytes")
    _put_u(out, 64, 8, OPENING_HEADER_BYTES, "header_bytes")
    _put_u(out, 72, 8, value.chunk_count, "chunk_count")
    _put_u(out, 80, 8, value.record_count, "record_count")
    _put_bytes(out, 96, 32, value.prefix_sha256, "prefix_sha256")
    _put_bytes(out, 128, 32, value.trailer_body_sha256, "trailer_body_sha256")
    _put_bytes(out, 160, 32, value.opening_header_sha256, "opening_header_sha256")
    digest = sha256(out[:224]).digest()
    supplied_digest = _bytes_like(value.footer_sha256, "footer_sha256")
    if supplied_digest and any(supplied_digest) and supplied_digest != digest:
        _fail(ErrorCode.HASH, "footer_sha256")
    _put_bytes(out, 224, 32, digest, "footer_sha256")
    return bytes(out)


def decode_footer(data: object) -> Footer:
    raw = _exact_input(data, FOOTER_BYTES, "footer")
    _expect_bytes(raw, 0, FOOTER_MAGIC, "magic", ErrorCode.BAD_MAGIC)
    if raw[224:256] != sha256(raw[:224]).digest():
        _fail(ErrorCode.HASH, "footer_sha256")
    if _read_u(raw, 16, 2) != CONTAINER_MAJOR or _read_u(raw, 18, 2) != CONTAINER_MINOR:
        _fail(ErrorCode.VERSION, "footer_version")
    _expect_u(raw, 20, 4, FOOTER_BYTES, "footer_bytes", ErrorCode.FIXED_VALUE)
    _expect_u(raw, 24, 8, FOOTER_FLAGS, "footer_flags", ErrorCode.FIXED_VALUE)
    _expect_u(raw, 64, 8, OPENING_HEADER_BYTES, "header_bytes", ErrorCode.FIXED_VALUE)
    _expect_zero(raw, 88, 8, "reserved_zero_0")
    _expect_zero(raw, 192, 32, "reserved_zero_1")
    value = Footer(
        trailer_offset=_read_u(raw, 32, 8),
        trailer_body_bytes=_read_u(raw, 40, 8),
        stream_bytes=_read_u(raw, 48, 8),
        prefix_bytes=_read_u(raw, 56, 8),
        chunk_count=_read_u(raw, 72, 8),
        record_count=_read_u(raw, 80, 8),
        prefix_sha256=raw[96:128],
        trailer_body_sha256=raw[128:160],
        opening_header_sha256=raw[160:192],
        footer_sha256=raw[224:256],
    )
    _validate_footer(value)
    return value


def _validate_header_trailer(opening: OpeningHeader, trailer: TrailerHeader) -> None:
    if trailer.process_generation != opening.process_generation:
        _fail(ErrorCode.RELATION, "process_generation", "opening/trailer mismatch")
    if trailer.scope_kind != opening.scope_kind:
        _fail(ErrorCode.RELATION, "scope_kind", "opening/trailer mismatch")
    if trailer.terminal_flags & 1:
        if trailer.active_generation != opening.process_generation:
            _fail(ErrorCode.RELATION, "active_generation")
        if trailer.active_start_counter != opening.start_counter:
            _fail(ErrorCode.RELATION, "active_start_counter", "opening/trailer mismatch")
        if trailer.active_start_monotonic_raw_ns != opening.start_monotonic_raw_ns:
            _fail(ErrorCode.RELATION, "active_start_monotonic_raw_ns", "opening/trailer mismatch")


def _validate_object_mirrors(opening: OpeningHeader, entries: Tuple[ObjectBindingEntry, ...]) -> None:
    by_kind = {entry.object_kind: entry for entry in entries}
    expected = {
        1: opening.schema_bundle_definition_sha256,
        2: opening.api_catalog_definition_sha256,
        10: opening.configuration_instance_sha256,
    }
    for kind, digest in expected.items():
        if kind in by_kind and by_kind[kind].sha256 != digest:
            _fail(ErrorCode.RELATION, "object_mirror_%d" % kind)


def decode_container(data: object, max_chunk_records: int = MAX_CHUNK_RECORD_COUNT) -> Container:
    """Decode and validate one complete in-memory structural container.

    External object bytes, JSON canonicalization, record catalog semantics,
    post-close receipts, and archive manifests are intentionally out of scope.
    A structurally encoded FAILED/INCOMPLETE outcome is returned, not promoted
    to COMPLETE and not treated as corrupt solely for being negative evidence.
    """

    raw = _bytes_like(data, "container")
    if len(raw) < OPENING_HEADER_BYTES:
        _fail(ErrorCode.TRUNCATED, "opening_header")
    opening_raw = raw[:OPENING_HEADER_BYTES]
    opening = decode_opening_header(opening_raw)
    if len(raw) < OPENING_HEADER_BYTES + FOOTER_BYTES:
        _fail(ErrorCode.TRUNCATED, "eof_footer")
    footer = decode_footer(raw[-FOOTER_BYTES:])
    if footer.stream_bytes != len(raw):
        _fail(ErrorCode.RELATION, "stream_bytes", "does not equal physical byte length")
    footer_offset = len(raw) - FOOTER_BYTES
    if footer.trailer_offset + footer.trailer_body_bytes != footer_offset:
        _fail(ErrorCode.RELATION, "trailer_extent")
    trailer_raw = raw[footer.trailer_offset:footer_offset]
    if sha256(trailer_raw).digest() != footer.trailer_body_sha256:
        _fail(ErrorCode.HASH, "trailer_body_sha256")
    trailer_body = _decode_trailer_body_grammar(trailer_raw)
    trailer = trailer_body.header
    if sha256(opening_raw).digest() != footer.opening_header_sha256:
        _fail(ErrorCode.HASH, "opening_header_sha256")
    prefix = raw[: footer.trailer_offset]
    if sha256(prefix).digest() != footer.prefix_sha256:
        _fail(ErrorCode.HASH, "prefix_sha256")

    chunks = []
    offset = OPENING_HEADER_BYTES
    expected_sequence = 0
    while offset < footer.trailer_offset:
        remaining = footer.trailer_offset - offset
        if remaining < CHUNK_HEADER_BYTES:
            _fail(ErrorCode.TRUNCATED, "chunk_header")
        chunk_header = decode_chunk_header(raw[offset : offset + CHUNK_HEADER_BYTES], max_chunk_records)
        chunk_bytes = CHUNK_HEADER_BYTES + chunk_header.payload_bytes
        if chunk_bytes > remaining:
            _fail(ErrorCode.TRUNCATED, "chunk_payload")
        chunk = decode_chunk(raw[offset : offset + chunk_bytes], expected_sequence, max_chunk_records)
        chunks.append(chunk)
        expected_sequence += 1
        offset += chunk_bytes
    if offset != footer.trailer_offset:
        _fail(ErrorCode.RELATION, "trailer_offset")

    chunk_count = len(chunks)
    record_count = sum(chunk.header.record_count for chunk in chunks)
    _validate_trailer_payload_layout(trailer)
    if chunk_count != trailer.chunk_count or chunk_count != footer.chunk_count:
        _fail(ErrorCode.RELATION, "chunk_count")
    if record_count != trailer.record_count or record_count != footer.record_count:
        _fail(ErrorCode.RELATION, "record_count")
    if trailer.chunks_end_offset != footer.trailer_offset:
        _fail(ErrorCode.RELATION, "chunks_end_offset")
    if trailer.trailer_body_bytes != footer.trailer_body_bytes:
        _fail(ErrorCode.RELATION, "trailer_body_bytes")

    _validate_event_table(trailer_body)
    derived: Dict[Tuple[int, int], int] = {}
    for chunk in chunks:
        for record in chunk.records:
            key = (record.event_kind, record.api_id)
            derived[key] = derived.get(key, 0) + 1
    table = {(entry.event_kind, entry.api_id): entry.record_count for entry in trailer_body.event_entries}
    if derived != table:
        _fail(ErrorCode.RELATION, "event_entries", "does not equal totals re-derived from records")

    _validate_diagnostic_table(trailer_body)
    _validate_object_table(trailer_body)
    _validate_object_mirrors(opening, trailer_body.object_entries)
    _validate_header_trailer(opening, trailer)
    _validate_trailer_finalization(trailer)
    return Container(opening, tuple(chunks), trailer_body, footer)


def validate_descriptor(descriptor: StructureDescriptor) -> None:
    """Validate exact bounds/non-overlap for a published wire descriptor."""

    if not isinstance(descriptor, StructureDescriptor):
        _fail(ErrorCode.TYPE, "descriptor")
    occupied = [False] * descriptor.size
    for field in descriptor.fields:
        if field.offset < 0 or field.size <= 0 or field.offset + field.size > descriptor.size:
            _fail(ErrorCode.RELATION, descriptor.name, "field outside structure: " + field.name)
        for index in range(field.offset, field.offset + field.size):
            if occupied[index]:
                _fail(ErrorCode.RELATION, descriptor.name, "overlap at byte %d" % index)
            occupied[index] = True
    if not all(occupied):
        _fail(ErrorCode.RELATION, descriptor.name, "descriptor has an untyped byte")


__all__ = [
    "ALL_DESCRIPTORS",
    "CHUNK_HEADER_BYTES",
    "CHUNK_HEADER_DESCRIPTOR",
    "Chunk",
    "ChunkHeader",
    "Container",
    "DIAGNOSTIC_TOTAL_ENTRY_BYTES",
    "DIAGNOSTIC_TOTAL_ENTRY_DESCRIPTOR",
    "DiagnosticTotalEntry",
    "EVENT_RECORD_BYTES",
    "EVENT_RECORD_DESCRIPTOR",
    "EVENT_TOTAL_ENTRY_BYTES",
    "EVENT_TOTAL_ENTRY_DESCRIPTOR",
    "ErrorCode",
    "EventRecord",
    "EventTotalEntry",
    "FOOTER_BYTES",
    "FOOTER_DESCRIPTOR",
    "Footer",
    "MAX_CHUNK_RECORD_COUNT",
    "OBJECT_BINDING_ENTRY_BYTES",
    "OBJECT_BINDING_ENTRY_DESCRIPTOR",
    "OPENING_HEADER_BYTES",
    "OPENING_HEADER_DESCRIPTOR",
    "ObjectBindingEntry",
    "OpeningHeader",
    "TRAILER_HEADER_BYTES",
    "TRAILER_HEADER_DESCRIPTOR",
    "TrailerBody",
    "TrailerHeader",
    "WireError",
    "crc32c",
    "decode_chunk",
    "decode_chunk_header",
    "decode_container",
    "decode_diagnostic_total_entry",
    "decode_event_record",
    "decode_event_total_entry",
    "decode_footer",
    "decode_object_binding_entry",
    "decode_opening_header",
    "decode_trailer_body",
    "decode_trailer_header",
    "encode_chunk",
    "encode_chunk_header",
    "encode_diagnostic_total_entry",
    "encode_event_record",
    "encode_event_total_entry",
    "encode_footer",
    "encode_object_binding_entry",
    "encode_opening_header",
    "encode_trailer_body",
    "encode_trailer_header",
    "validate_descriptor",
]
