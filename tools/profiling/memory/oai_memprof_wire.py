#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0

"""Independent schema-v1 event-record wire oracle.

This module implements only the syntactic 96-byte event record defined by the
version-1 measurement contract.  API, event-kind, and flag meanings belong to
separate versioned catalogs; this codec deliberately does not invent semantic
restrictions that are not part of the frozen record layout.

The implementation uses explicit offsets and ``int.to_bytes``/``from_bytes``
instead of a native or aggregate structure so that padding and host ABI cannot
affect the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union


EVENT_RECORD_V1_SIZE = 96
_BytesLike = Union[bytes, bytearray, memoryview]


@dataclass(frozen=True)
class FieldSpec:
    """One fixed-width integer field in a schema-v1 disk record."""

    name: str
    offset: int
    width: int
    signed: bool = False


EVENT_RECORD_V1_LAYOUT: Tuple[FieldSpec, ...] = (
    FieldSpec("thread_sequence", 0, 8),
    FieldSpec("counter_enter", 8, 8),
    FieldSpec("counter_exit", 16, 8),
    FieldSpec("address_before", 24, 8),
    FieldSpec("address_after", 32, 8),
    FieldSpec("arg0", 40, 8),
    FieldSpec("arg1", 48, 8),
    FieldSpec("arg2", 56, 8),
    FieldSpec("context_id", 64, 4),
    FieldSpec("callsite_id", 68, 4),
    FieldSpec("thread_index", 72, 4),
    FieldSpec("flags", 76, 4),
    FieldSpec("result_code", 80, 4, signed=True),
    FieldSpec("api_id", 84, 2),
    FieldSpec("event_kind", 86, 2),
    FieldSpec("cpu_enter", 88, 2),
    FieldSpec("cpu_exit", 90, 2),
    FieldSpec("reserved_zero", 92, 4),
)


@dataclass(frozen=True)
class EventRecordV1:
    """Decoded values of one schema-v1 disk event record."""

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
    reserved_zero: int = 0


class EventRecordWireError(ValueError):
    """A schema-v1 record violates the fixed wire-format contract."""


def _validate_layout() -> None:
    """Fail at import if this oracle's descriptor is not exactly contiguous."""

    cursor = 0
    names = set()
    for spec in EVENT_RECORD_V1_LAYOUT:
        if spec.name in names:
            raise RuntimeError("duplicate schema-v1 field: {}".format(spec.name))
        if spec.offset != cursor:
            raise RuntimeError(
                "schema-v1 field {} starts at {}, expected {}".format(
                    spec.name, spec.offset, cursor
                )
            )
        if spec.width not in (2, 4, 8):
            raise RuntimeError(
                "schema-v1 field {} has unsupported width {}".format(
                    spec.name, spec.width
                )
            )
        names.add(spec.name)
        cursor += spec.width

    if cursor != EVENT_RECORD_V1_SIZE:
        raise RuntimeError(
            "schema-v1 layout occupies {} bytes, expected {}".format(
                cursor, EVENT_RECORD_V1_SIZE
            )
        )


def _validate_fixed_width(spec: FieldSpec, value: int) -> None:
    # bool is an int subclass, but accepting it would hide a type error in a
    # scientific literal or decoder caller.
    if type(value) is not int:
        raise TypeError(
            "{} must be an int, not {}".format(spec.name, type(value).__name__)
        )

    bits = spec.width * 8
    if spec.signed:
        minimum = -(1 << (bits - 1))
        maximum = (1 << (bits - 1)) - 1
    else:
        minimum = 0
        maximum = (1 << bits) - 1

    if value < minimum or value > maximum:
        kind = "i{}".format(bits) if spec.signed else "u{}".format(bits)
        raise EventRecordWireError(
            "{}={} is outside {} [{}, {}]".format(
                spec.name, value, kind, minimum, maximum
            )
        )


def encode_event_record_v1(record: EventRecordV1) -> bytes:
    """Encode one event as the exact 96-byte little-endian schema-v1 record."""

    if not isinstance(record, EventRecordV1):
        raise TypeError("record must be an EventRecordV1")

    if record.reserved_zero != 0:
        raise EventRecordWireError("reserved_zero must be zero")

    output = bytearray(EVENT_RECORD_V1_SIZE)
    for spec in EVENT_RECORD_V1_LAYOUT:
        value = getattr(record, spec.name)
        _validate_fixed_width(spec, value)
        output[spec.offset : spec.offset + spec.width] = value.to_bytes(
            spec.width, byteorder="little", signed=spec.signed
        )

    return bytes(output)


def decode_event_record_v1(data: _BytesLike) -> EventRecordV1:
    """Decode one exact schema-v1 record and reject malformed wire syntax."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes, bytearray, or memoryview")

    wire_size = data.nbytes if isinstance(data, memoryview) else len(data)
    if wire_size != EVENT_RECORD_V1_SIZE:
        raise EventRecordWireError(
            "schema-v1 event record has {} bytes, expected {}".format(
                wire_size, EVENT_RECORD_V1_SIZE
            )
        )

    if isinstance(data, memoryview):
        if data.ndim != 1 or data.itemsize != 1 or not data.c_contiguous:
            raise EventRecordWireError(
                "memoryview must be one-dimensional, byte-oriented, and C-contiguous"
            )
        raw = data.tobytes()
    else:
        raw = bytes(data)

    values = {}
    for spec in EVENT_RECORD_V1_LAYOUT:
        field_bytes = raw[spec.offset : spec.offset + spec.width]
        values[spec.name] = int.from_bytes(
            field_bytes, byteorder="little", signed=spec.signed
        )

    if values["reserved_zero"] != 0:
        raise EventRecordWireError("reserved_zero must be zero")

    return EventRecordV1(**values)


_validate_layout()
