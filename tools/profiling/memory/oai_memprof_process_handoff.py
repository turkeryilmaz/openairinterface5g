#!/usr/bin/env python3
"""Strict schema-v1 decoder for the self-hashed OAI memprof process handoff."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


HEADER_BYTES = 1152
API_SLOT_COUNT = 32
ADMITTED_API_COUNT = 12
THREAD_RUNTIME_BYTES = 368
THREAD_BYTES = 448
SELF_SHA256_BYTES = 32
MAX_BOOTSTRAP_BYTES = 65_536
MAX_MAPS_BYTES = 16_777_216
MAX_THREADS = 65_534
MAX_RING_RECORDS = 1_048_576
MAX_FLUSH_RECORDS = 65_536
MAGIC = b"OAIMPHANDOFFV1\x00\x00"
UINT64_MAX = (1 << 64) - 1


class ProcessHandoffError(ValueError):
    """Deterministic rejection of malformed or unauthenticated handoff bytes."""


def _fail(field: str, detail: str) -> None:
    raise ProcessHandoffError(f"{field}: {detail}")


def _load_wire() -> ModuleType:
    path = Path(__file__).resolve().parent / "oai_memprof_container_wire.py"
    spec = importlib.util.spec_from_file_location("_oai_memprof_container_wire_for_handoff", path)
    if spec is None or spec.loader is None:
        _fail("wire", "container decoder unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


WIRE = _load_wire()


@dataclasses.dataclass(frozen=True)
class ClockSample:
    counter: int
    monotonic_raw_before_ns: int
    monotonic_raw_after_ns: int
    realtime_unix_ns: int


@dataclasses.dataclass(frozen=True)
class RuntimeSnapshot:
    process_generation: int
    reservations: int
    ready_threads: int
    registration_capacity_failures: int
    unregistered_active_thread_failures: int
    diagnostic_saturation_transitions: int
    registration_diagnostic_saturated_mask: int
    recursion_bypasses: int
    ring_full_losses: int
    admitted_transactions: int
    completed_transactions: int
    emitted_events: int
    requested_bytes: int
    table_entries: int
    sample_seed: int
    sample_threshold: int
    table_probes: int
    table_shards: int
    state: int
    mode_id: int


@dataclasses.dataclass(frozen=True)
class ClockInfo:
    counter_frequency_numerator: int
    counter_frequency_denominator: int
    architecture_id: int
    acquisition_source_id: int
    clock_kind: int


@dataclasses.dataclass(frozen=True)
class WriterResult:
    status: int
    runtime_status: int
    clock_status: int
    prefooter_closed: bool
    system_errno: int
    runtime_snapshot: RuntimeSnapshot
    clock_info: ClockInfo
    seal_before_sample: ClockSample
    seal_after_sample: ClockSample
    drain_complete_sample: ClockSample
    final_sample: ClockSample
    chunk_count: int
    record_count: int
    payload_bytes: int
    stream_bytes: int
    file_device: int
    file_inode: int


@dataclasses.dataclass(frozen=True)
class ThreadEvidence:
    process_generation: int
    registration_ordinal: int
    thread_sequence: int
    api_attempts: tuple[int, ...]
    requested_bytes: int
    completed_transactions: int
    recursion_bypasses: int
    ring_full_losses: int
    size_unknowns: int
    counter_invalids: int
    sample_insertion_failures: int
    sample_lookup_failures: int
    sample_probe_exhaustions: int
    sample_pairing_failures: int
    thread_index: int
    diagnostic_saturated_mask: int
    diagnostic_values: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class ProcessHandoff:
    raw: bytes
    opening_raw: bytes
    opening: object
    opening_sample: ClockSample
    writer: WriterResult
    bootstrap_bytes: bytes
    maps_bytes: bytes
    threads: tuple[ThreadEvidence, ...]
    unregistered_active_thread_failures: int
    writer_io_or_finalization_failures: int
    diagnostic_saturation_transitions: int
    registration_diagnostic_saturated_mask: int
    ring_records: int
    flush_records: int
    flush_interval_ns: int
    realloc_zero_policy_id: int
    bootstrap_sha256: bytes
    maps_sha256: bytes
    opening_header_sha256: bytes
    prefix_sha256: bytes
    handoff_sha256: bytes


def _u(raw: bytes, offset: int, size: int) -> int:
    return int.from_bytes(raw[offset : offset + size], "little", signed=False)


def _sample(raw: bytes, offset: int) -> ClockSample:
    return ClockSample(*(_u(raw, offset + index * 8, 8) for index in range(4)))


def _writer(
    raw: bytes,
    unregistered_active_thread_failures: int,
    diagnostic_saturation_transitions: int,
    registration_diagnostic_saturated_mask: int,
    table_entries: int,
    sample_seed: int,
    sample_threshold: int,
    table_probes: int,
    table_shards: int,
) -> WriterResult:
    if any(raw[20:24]) or any(raw[106:112]) or any(raw[133:136]):
        _fail("writer.reserved", "nonzero reserved byte")
    status, runtime_status, clock_status, closed, system_errno = (
        _u(raw, offset, 4) for offset in (0, 4, 8, 12, 16)
    )
    if status > 11 or runtime_status > 7 or clock_status > 5 or closed > 1 or system_errno > 0x7FFFFFFF:
        _fail("writer", "enum or signed-error domain mismatch")
    snapshot = RuntimeSnapshot(
        _u(raw, 24, 8),
        _u(raw, 32, 8),
        _u(raw, 40, 8),
        _u(raw, 48, 8),
        unregistered_active_thread_failures,
        diagnostic_saturation_transitions,
        registration_diagnostic_saturated_mask,
        *(_u(raw, offset, 8) for offset in range(56, 104, 8)),
        table_entries,
        sample_seed,
        sample_threshold,
        table_probes,
        table_shards,
        raw[104],
        raw[105],
    )
    clock_info = ClockInfo(
        _u(raw, 112, 8),
        _u(raw, 120, 8),
        _u(raw, 128, 2),
        _u(raw, 130, 2),
        raw[132],
    )
    return WriterResult(
        status,
        runtime_status,
        clock_status,
        bool(closed),
        system_errno,
        snapshot,
        clock_info,
        _sample(raw, 136),
        _sample(raw, 168),
        _sample(raw, 200),
        _sample(raw, 232),
        *(_u(raw, offset, 8) for offset in range(264, 312, 8)),
    )


def _thread(raw: bytes) -> ThreadEvidence:
    requested_offset = 24 + API_SLOT_COUNT * 8
    completed_offset = requested_offset + 8
    recursion_offset = completed_offset + 8
    ring_full_offset = recursion_offset + 8
    size_unknown_offset = ring_full_offset + 8
    counter_invalid_offset = size_unknown_offset + 8
    thread_index_offset = counter_invalid_offset + 8
    saturated_offset = thread_index_offset + 4
    sample_insertion_offset = saturated_offset + 4
    sample_lookup_offset = sample_insertion_offset + 8
    sample_probe_offset = sample_lookup_offset + 8
    sample_pairing_offset = sample_probe_offset + 8
    if sample_pairing_offset + 8 != THREAD_RUNTIME_BYTES:
        _fail("thread.layout", "internal runtime extent mismatch")
    saturated = _u(raw, saturated_offset, 4)
    if saturated & ~0x3FF:
        _fail("thread.diagnostic_saturated_mask", "reserved bit set")
    return ThreadEvidence(
        _u(raw, 0, 8),
        _u(raw, 8, 8),
        _u(raw, 16, 8),
        tuple(_u(raw, 24 + index * 8, 8) for index in range(API_SLOT_COUNT)),
        _u(raw, requested_offset, 8),
        _u(raw, completed_offset, 8),
        _u(raw, recursion_offset, 8),
        _u(raw, ring_full_offset, 8),
        _u(raw, size_unknown_offset, 8),
        _u(raw, counter_invalid_offset, 8),
        _u(raw, sample_insertion_offset, 8),
        _u(raw, sample_lookup_offset, 8),
        _u(raw, sample_probe_offset, 8),
        _u(raw, sample_pairing_offset, 8),
        _u(raw, thread_index_offset, 4),
        saturated,
        tuple(_u(raw, THREAD_RUNTIME_BYTES + index * 8, 8) for index in range(10)),
    )


def _sat_add(left: int, right: int) -> int:
    return min(UINT64_MAX, left + right)


def _mask_matches(values: tuple[int, ...], mask: int) -> bool:
    expected = sum(1 << index for index, value in enumerate(values) if value == UINT64_MAX)
    return expected == mask


def decode_process_handoff(data: object) -> ProcessHandoff:
    if not isinstance(data, bytes):
        _fail("handoff", "immutable bytes required")
    raw = data
    if len(raw) < HEADER_BYTES + SELF_SHA256_BYTES:
        _fail("handoff", "truncated")
    if raw[:16] != MAGIC:
        _fail("magic", "schema-v1 magic mismatch")
    digest = hashlib.sha256(raw[:-32]).digest()
    if digest != raw[-32:]:
        _fail("handoff_sha256", "self-hash mismatch")
    if (_u(raw, 16, 2), _u(raw, 18, 2)) != (1, 5):
        _fail("version", "exact 1.5 required")
    if _u(raw, 20, 4) != HEADER_BYTES or _u(raw, 24, 8) != len(raw) or _u(raw, 76, 4) != THREAD_BYTES:
        _fail("layout", "fixed size field mismatch")
    if any(raw[118:120]) or any(raw[1144:1152]):
        _fail("reserved", "nonzero reserved byte")

    bootstrap_offset, bootstrap_size = _u(raw, 32, 8), _u(raw, 40, 8)
    maps_offset, maps_size = _u(raw, 48, 8), _u(raw, 56, 8)
    threads_offset, thread_count = _u(raw, 64, 8), _u(raw, 72, 4)
    if (
        not 1 <= bootstrap_size <= MAX_BOOTSTRAP_BYTES
        or not 1 <= maps_size <= MAX_MAPS_BYTES
        or thread_count > MAX_THREADS
    ):
        _fail("layout", "bounded section domain mismatch")
    if (
        bootstrap_offset != HEADER_BYTES
        or maps_offset != bootstrap_offset + bootstrap_size
        or threads_offset != maps_offset + maps_size
        or threads_offset + thread_count * THREAD_BYTES + SELF_SHA256_BYTES != len(raw)
    ):
        _fail("layout", "noncanonical section arithmetic")

    opening_raw = raw[128:640]
    try:
        opening = WIRE.decode_opening_header(opening_raw)
    except Exception as error:
        raise ProcessHandoffError(f"opening: {error}") from error

    unregistered = _u(raw, 80, 8)
    saturation_transitions = _u(raw, 88, 8)
    registration_mask = _u(raw, 96, 4)
    writer_failures = _u(raw, 104, 8)
    ring_records = _u(raw, 100, 4)
    flush_records = _u(raw, 112, 4)
    realloc_zero_policy_id = _u(raw, 116, 2)
    flush_interval_ns = _u(raw, 120, 8)
    table_entries = _u(raw, 1112, 8)
    sample_seed = _u(raw, 1120, 8)
    sample_threshold = _u(raw, 1128, 8)
    table_probes = _u(raw, 1136, 4)
    table_shards = _u(raw, 1140, 4)
    if (
        not 2 <= ring_records <= MAX_RING_RECORDS
        or ring_records & (ring_records - 1)
        or not 1 <= flush_records <= MAX_FLUSH_RECORDS
        or realloc_zero_policy_id not in (1, 2)
    ):
        _fail("runtime_configuration", "ring/flush/realloc-policy domain mismatch")
    if registration_mask & ~0x3:
        _fail("registration_diagnostic_saturated_mask", "reserved bit set")
    opening_sample = _sample(raw, 1048)
    prefix_sha = raw[1080:1112]
    if not any(prefix_sha):
        _fail("prefix_sha256", "nonzero producer-authenticated digest required")
    writer = _writer(
        raw[640:952],
        unregistered,
        saturation_transitions,
        registration_mask,
        table_entries,
        sample_seed,
        sample_threshold,
        table_probes,
        table_shards,
    )
    bootstrap = raw[bootstrap_offset:maps_offset]
    maps = raw[maps_offset:threads_offset]
    bootstrap_sha = hashlib.sha256(bootstrap).digest()
    maps_sha = hashlib.sha256(maps).digest()
    opening_sha = hashlib.sha256(opening_raw).digest()
    if bootstrap_sha != raw[952:984] or maps_sha != raw[984:1016] or opening_sha != raw[1016:1048]:
        _fail("component_sha256", "component hash mismatch")
    if bootstrap_sha != opening.configuration_instance_sha256:
        _fail("bootstrap_sha256", "opening configuration digest mismatch")

    threads = tuple(
        _thread(raw[threads_offset + index * THREAD_BYTES : threads_offset + (index + 1) * THREAD_BYTES])
        for index in range(thread_count)
    )
    snapshot = writer.runtime_snapshot
    if (
        snapshot.process_generation != opening.process_generation
        or snapshot.ready_threads != thread_count
        or snapshot.reservations < snapshot.ready_threads
        or snapshot.unregistered_active_thread_failures != unregistered
        or snapshot.diagnostic_saturation_transitions != saturation_transitions
        or snapshot.registration_diagnostic_saturated_mask != registration_mask
    ):
        _fail("runtime_snapshot", "generation/population/diagnostic mismatch")
    if snapshot.state != 3 or snapshot.mode_id not in (2, 3, 4):
        _fail("runtime_snapshot", "DRAINING mode 2/3/4 required")
    sampled = snapshot.mode_id == 3
    shard_limit = min(snapshot.table_entries, 256)
    expected_shards = 1
    while expected_shards * 2 <= shard_limit:
        expected_shards *= 2
    if sampled:
        if (
            snapshot.table_entries == 0
            or snapshot.sample_threshold == 0
            or not 1 <= snapshot.table_probes <= snapshot.table_entries
            or snapshot.table_shards != expected_shards
        ):
            _fail("sampling_controls", "sampled table/seed/threshold/probe/shard relation mismatch")
    elif any(
        (
            snapshot.table_entries,
            snapshot.sample_seed,
            snapshot.sample_threshold,
            snapshot.table_probes,
            snapshot.table_shards,
        )
    ):
        _fail("sampling_controls", "non-sampled controls must be zero")
    if (
        writer.clock_info.clock_kind != opening.clock_kind
        or writer.clock_info.counter_frequency_numerator != opening.counter_frequency_numerator
        or writer.clock_info.counter_frequency_denominator != opening.counter_frequency_denominator
    ):
        _fail("clock_info", "opening identity/status mismatch")
    if (
        opening_sample.counter != opening.start_counter
        or opening_sample.realtime_unix_ns != opening.start_realtime_unix_ns
        or opening_sample.monotonic_raw_before_ns > opening.start_monotonic_raw_ns
        or opening_sample.monotonic_raw_after_ns < opening.start_monotonic_raw_ns
    ):
        _fail("opening_sample", "opening midpoint is not bound to acquisition bracket")
    if writer.seal_before_sample != ClockSample(0, 0, 0, 0) and (
        opening_sample.counter >= writer.seal_before_sample.counter
        or opening_sample.monotonic_raw_after_ns > writer.seal_before_sample.monotonic_raw_before_ns
    ):
        _fail("opening_sample", "opening acquisition does not precede sealing")

    samples = (
        writer.seal_before_sample,
        writer.seal_after_sample,
        writer.drain_complete_sample,
        writer.final_sample,
    )
    reached_zero = False
    previous = None
    for index, sample in enumerate(samples):
        if sample == ClockSample(0, 0, 0, 0):
            reached_zero = True
            continue
        if reached_zero:
            _fail(f"clock_samples[{index}]", "nonzero sample after zero suffix")
        if (
            not sample.counter
            or not sample.realtime_unix_ns
            or sample.monotonic_raw_after_ns < sample.monotonic_raw_before_ns
        ):
            _fail(f"clock_samples[{index}]", "invalid bracket")
        if previous is not None and (
            previous.counter >= sample.counter
            or previous.monotonic_raw_after_ns > sample.monotonic_raw_before_ns
        ):
            _fail(f"clock_samples[{index}]", "non-increasing or overlapping")
        previous = sample
    if writer.clock_status == 0 and reached_zero:
        _fail("clock_samples", "successful clock requires complete sequence")
    if (writer.clock_status == 0 and writer.status == 11) or (
        writer.clock_status != 0 and writer.status == 0
    ):
        _fail("clock_status", "clock failure/writer status mismatch")

    totals = [0, 0, 0, 0, 0]
    saturated_instances = 0
    for index, thread in enumerate(threads):
        if (
            thread.process_generation,
            thread.registration_ordinal,
            thread.thread_index,
        ) != (opening.process_generation, index + 1, index + 1):
            _fail(f"threads[{index}]", "registration order/identity mismatch")
        if not _mask_matches(thread.diagnostic_values, thread.diagnostic_saturated_mask):
            _fail(f"threads[{index}]", "diagnostic saturation mask mismatch")
        attempts = 0
        for value in thread.api_attempts:
            attempts = _sat_add(attempts, value)
        if any(thread.api_attempts[ADMITTED_API_COUNT:]):
            _fail(f"threads[{index}]", "reserved API counter slot is nonzero")
        expected_diagnostics = (
            thread.ring_full_losses,
            thread.recursion_bypasses,
            0,
            0,
            thread.size_unknowns,
            thread.sample_insertion_failures,
            thread.sample_lookup_failures,
            thread.sample_probe_exhaustions,
            thread.sample_pairing_failures,
            thread.counter_invalids,
        )
        if thread.completed_transactions > attempts or thread.diagnostic_values != expected_diagnostics:
            _fail(f"threads[{index}]", "runtime/diagnostic projection mismatch")
        if snapshot.mode_id != 3 and any(
            (
                thread.sample_insertion_failures,
                thread.sample_lookup_failures,
                thread.sample_probe_exhaustions,
                thread.sample_pairing_failures,
            )
        ):
            _fail(f"threads[{index}]", "non-sampled selection diagnostic nonzero")
        if snapshot.mode_id == 2 and (thread.ring_full_losses or thread.counter_invalids):
            _fail(f"threads[{index}]", "counter-mode exact-event diagnostic nonzero")
        saturated_instances += sum(value == UINT64_MAX for value in thread.diagnostic_values)
        for total_index, value in enumerate(
            (
                attempts,
                thread.completed_transactions,
                thread.requested_bytes,
                thread.recursion_bypasses,
                thread.ring_full_losses,
            )
        ):
            totals[total_index] = _sat_add(totals[total_index], value)

    observed = (
        snapshot.admitted_transactions,
        snapshot.completed_transactions,
        snapshot.requested_bytes,
        snapshot.recursion_bypasses,
        snapshot.ring_full_losses,
    )
    if tuple(totals) != observed:
        _fail("runtime_snapshot", "thread aggregate mismatch")
    registration_values = (unregistered, snapshot.registration_capacity_failures)
    for index, value in enumerate(registration_values):
        saturated = value == UINT64_MAX
        if saturated != bool(registration_mask & (1 << index)):
            _fail("registration_diagnostic_saturated_mask", "counter saturation mismatch")
        saturated_instances += saturated
    saturated_instances += writer_failures == UINT64_MAX
    if saturated_instances != saturation_transitions:
        _fail("diagnostic_saturation_transitions", "available population mismatch")

    if (
        writer.record_count > UINT64_MAX // 96
        or writer.chunk_count > UINT64_MAX // 32
        or writer.payload_bytes != writer.record_count * 96
        or writer.stream_bytes < 512
        or (writer.chunk_count == 0) != (writer.record_count == 0)
        or writer.chunk_count > writer.record_count
    ):
        _fail("writer", "stream count/byte relation mismatch")
    complete_prefix = 512 + writer.chunk_count * 32 + writer.payload_bytes
    if (
        complete_prefix > writer.stream_bytes
        or writer.record_count > snapshot.emitted_events
        or (snapshot.mode_id == 2 and (writer.chunk_count or writer.record_count))
    ):
        _fail("writer", "retained complete-prefix relation mismatch")
    if writer.status == 0:
        if (
            writer.runtime_status != 0
            or not writer.prefooter_closed
            or totals[0] != totals[1]
            or writer.record_count != snapshot.emitted_events
            or writer.stream_bytes != complete_prefix
            or writer_failures != 0
        ):
            _fail("writer", "successful writer relation mismatch")
    elif writer_failures == 0:
        _fail("writer", "failed writer missing diagnostic")

    return ProcessHandoff(
        raw,
        opening_raw,
        opening,
        opening_sample,
        writer,
        bootstrap,
        maps,
        threads,
        unregistered,
        writer_failures,
        saturation_transitions,
        registration_mask,
        ring_records,
        flush_records,
        flush_interval_ns,
        realloc_zero_policy_id,
        bootstrap_sha,
        maps_sha,
        opening_sha,
        prefix_sha,
        digest,
    )


__all__ = [
    "ClockInfo",
    "ClockSample",
    "ProcessHandoff",
    "ProcessHandoffError",
    "RuntimeSnapshot",
    "ThreadEvidence",
    "WriterResult",
    "decode_process_handoff",
]
