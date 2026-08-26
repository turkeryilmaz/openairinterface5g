#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Independent whole-container oracle for the native stream finalizer."""

from __future__ import annotations

import pathlib
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.profiling.memory import oai_memprof_container_wire as wire


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: validate_oai_memprof_stream_finalizer.py STREAM RECORDS CHUNKS")
    path = pathlib.Path(sys.argv[1])
    expected_records = int(sys.argv[2], 10)
    expected_chunks = int(sys.argv[3], 10)
    raw = path.read_bytes()
    container = wire.decode_container(raw)
    assert container.trailer_body.header.lifecycle_state == 5
    assert container.trailer_body.header.payload_writer_state == 5
    assert container.trailer_body.header.finalization_stage == 6
    assert container.trailer_body.header.terminal_flags == 0xFFF
    assert len(container.chunks) == expected_chunks
    assert sum(chunk.header.record_count for chunk in container.chunks) == expected_records
    assert container.trailer_body.event_entries == (wire.EventTotalEntry(1, 1, expected_records),)
    assert tuple(row.object_kind for row in container.trailer_body.object_entries) == tuple(range(1, 13))
    assert container.footer.stream_bytes == len(raw)
    assert container.footer.record_count == expected_records
    print(
        f"stream-finalizer oracle passed: bytes={len(raw)} "
        f"chunks={expected_chunks} records={expected_records}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
