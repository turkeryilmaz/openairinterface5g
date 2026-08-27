#!/usr/bin/env python3
"""Synthetic-only deterministic tests for the composed archive verifier."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import os
import pathlib
import sys
import types
import unittest
import uuid


HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "archive_semantic_verifier_v1", HERE / "archive_semantic_verifier_v1.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("integration verifier module specification unavailable")
V = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = V
SPEC.loader.exec_module(V)

FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "_integration_coverage_fixtures",
    V.CATALOG_ROOT / "coverage/tests/test_coverage_catalog_v1.py",
)
if FIXTURE_SPEC is None or FIXTURE_SPEC.loader is None:
    raise RuntimeError("coverage fixture module specification unavailable")
F = importlib.util.module_from_spec(FIXTURE_SPEC)
sys.modules[FIXTURE_SPEC.name] = F
FIXTURE_SPEC.loader.exec_module(F)


STREAM_PATH = "streams/memory-lifetime.bin"
VERIFIER_PATH = V.ACCEPTED_VERIFIER_DEFINITION_PATH
VERIFIER_DEFINITION = V.accepted_verifier_definition_bytes()
EVENT_CLASSIFIER_DEFINITION = V.accepted_event_classifier_definition_bytes()
GENERATION = 7
START_COUNTER = 100
START_MONOTONIC = 1_000
START_REALTIME = 1_700_000_000_000_000_000
FINAL_COUNTER = 2_500
FINAL_MONOTONIC = 2_000
FINAL_REALTIME = START_REALTIME + (FINAL_MONOTONIC - START_MONOTONIC)

A03_SEED_HEX = "0123456789abcdef"
A03_SEED_K = int(A03_SEED_HEX, 16)
A03_THRESHOLD = max(
    V.SELECTION.selection_value(GENERATION, 1, 1, A03_SEED_K),
    V.SELECTION.selection_value(GENERATION, 2, 1, A03_SEED_K),
) + 1


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


TRUSTED_RELEASE_AUTHORITY_FIXTURE_SHA256 = (
    "7c7743fa026f0df8a4154bc9330ab1c1420abb26f0291b165a430e32b451431f"
)
_TRUSTED_RELEASE_AUTHORITY_SOURCES = V.accepted_trusted_release_source_bytes()
_TRUSTED_RELEASE_AUTHORITY_RAW = V.make_trusted_release_authority_bytes(
    commit="1" * 40,
    tree="2" * 40,
    source_bytes=_TRUSTED_RELEASE_AUTHORITY_SOURCES,
)
_TRUSTED_RELEASE_AUTHORITY_SHA256 = sha(_TRUSTED_RELEASE_AUTHORITY_RAW)
if _TRUSTED_RELEASE_AUTHORITY_SHA256 != TRUSTED_RELEASE_AUTHORITY_FIXTURE_SHA256:
    raise AssertionError(
        "trusted-release authority fixture changed; review and explicitly "
        "update its literal pin "
        f"(observed {_TRUSTED_RELEASE_AUTHORITY_SHA256})"
    )
VERIFIER_SHA256 = sha(VERIFIER_DEFINITION)
EVENT_CLASSIFIER_SHA256 = sha(EVENT_CLASSIFIER_DEFINITION)

HISTORICAL_BUNDLE_V14 = (2196, "5c4fa1ec85ccbb850c31f5973f2416f98fc0bd8cb7383673e5899c7a4e533f09")
HISTORICAL_BUNDLE_V14_ENTRIES = {
    2: (5327, "396b80e1be2ae765afa29346c57f88a279598fd1fd4b63f4870c0255711c1eed"),
    3: (14457, "f7a258d830eb7284ddf6d3feae65f2c7da25566104e3a254857778f7b342f6d8"),
    4: (737, "c315630f1a0cd4023cb693436dd7e5c72bd2684b46433a91e7a348adf6d525cb"),
    9: (9334, "851cf5899affe8420bb10da47998bf33304cea2f4deb5742ec54859cce9a9ae7"),
    11: (5588, "28f5b3e2a4cc4475a61919e4ca32b59521ea17a5a1bc2400902159ecfe8382a3"),
}


def binding(object_type: int, path: str, digest: str) -> dict:
    return {"object_type": object_type, "path": path, "sha256": digest}


def candidate_catalog(
    catalog_id: str, entries: list[dict], schema: dict[str, str]
) -> bytes:
    return V.SEMANTIC.canonical_bytes(
        {
            "catalog_id": catalog_id,
            "entries": entries,
            "schema": dict(schema),
            "version": {"major": 1, "minor": 0},
        }
    )


def diagnostic_counter_rows(
    *, mode_id: int = 4, ready_thread_indices: tuple[int, ...] = ()
) -> list[dict]:
    producer_reasons = {
        3: (1, 16, 17, 18, 32, 48, 49, 50, 51, 64),
        4: (1, 16, 17, 18, 32, 64),
    }[mode_id]
    keys = [
        (1, thread_index, reason_id)
        for thread_index in ready_thread_indices
        for reason_id in producer_reasons
    ]
    keys.extend(((2, 1, 2), (2, 1, 3), (3, 1, 80), (4, 1, 96)))
    rows = []
    for scope_kind, scope_id, reason_id in keys:
        rows.append(
            {
                "counter_scope_id": scope_id,
                "counter_scope_kind": scope_kind,
                "process_generation": GENERATION,
                "reason_id": reason_id,
                "saturated": False,
                "value": 0,
            }
        )
    return rows



def a03_records() -> tuple:
    return (
        V.WIRE.EventRecord(
            thread_sequence=1,
            counter_enter=110,
            counter_exit=111,
            address_before=0,
            address_after=0x1000,
            arg0=64,
            arg1=0,
            arg2=0,
            context_id=0,
            callsite_id=0,
            thread_index=1,
            flags=0x01010866,
            result_code=0,
            api_id=1,
            event_kind=1,
            cpu_enter=0xFFFF,
            cpu_exit=0xFFFF,
        ),
        V.WIRE.EventRecord(
            thread_sequence=1,
            counter_enter=112,
            counter_exit=113,
            address_before=0x1000,
            address_after=0,
            arg0=64,
            arg1=1,
            arg2=1,
            context_id=0,
            callsite_id=0,
            thread_index=2,
            flags=0x0102D07D,
            result_code=0,
            api_id=4,
            event_kind=3,
            cpu_enter=0xFFFF,
            cpu_exit=0xFFFF,
        ),
    )


def a03_overlapping_ended_records() -> tuple:
    birth, free = a03_records()
    return (
        birth,
        dataclasses.replace(
            birth,
            thread_index=2,
            thread_sequence=1,
            counter_enter=112,
            counter_exit=113,
        ),
        dataclasses.replace(
            free,
            thread_index=2,
            thread_sequence=2,
            counter_enter=114,
            counter_exit=115,
            arg1=1,
            arg2=1,
        ),
        dataclasses.replace(
            free,
            thread_index=1,
            thread_sequence=2,
            counter_enter=116,
            counter_exit=117,
            arg1=2,
            arg2=1,
        ),
    )

def pre_footer(*, complete: bool) -> dict:
    stage = 6 if complete else 1
    lifecycle = 5 if complete else 7
    reason = 0 if complete else 1
    flags = ((1 << (stage + 1)) - 1) | (1 << 7) | (1 << 8)
    if complete:
        flags |= (1 << 9) | (1 << 10) | (1 << 11)
    else:
        flags |= 1 << 13
    return {
        "active_generation": GENERATION,
        "active_start_counter": START_COUNTER,
        "active_start_monotonic_raw_ns": START_MONOTONIC,
        "cutoff_after_counter": 121,
        "cutoff_before_counter": 120,
        "diagnostic_population_partial": not complete,
        "final_counter": FINAL_COUNTER,
        "final_monotonic_raw_ns": FINAL_MONOTONIC,
        "final_realtime_unix_ns": FINAL_REALTIME,
        "finalization_stage": stage,
        "lifecycle_state": lifecycle,
        "payload_writer_state": 5,
        "process_generation": GENERATION,
        "quiescence_complete_counter": 130 if complete else 0,
        "reason_code": reason,
        "schema": V.STATUS.SCHEMA_PRE_FOOTER,
        "scope_kind": 1,
        "terminal_flags": flags,
    }


def base_external_objects(
    *,
    complete: bool = True,
    mode_id: int = 4,
    ready_thread_indices: tuple[int, ...] = (),
    measured_coverage: bool = False,
    trusted_release_authority_sha256: str | None = None,
) -> dict[str, bytes]:
    members, bundle = V._accepted_static_members()
    config = V.CONFIG.make_effective_configuration(
        flush_records=8,
        flush_us=1_000,
        max_threads=8,
        mode_id=mode_id,
        output_directory="/tmp/oai-memprof",
        ring_records=64,
        role_id=1,
        run_id="synthetic-integration",
        sample_seed_hex=A03_SEED_HEX if mode_id == 3 else None,
        sample_seed_provenance_id=2 if mode_id == 3 else 20,
        sample_seed_status_id=2 if mode_id == 3 else 20,
        sample_threshold=A03_THRESHOLD if mode_id == 3 else 0,
        scope_kind=1,
        selection_values=(
            []
            if trusted_release_authority_sha256 is None
            else [
                {
                    "key": "trusted_release_authority_sha256",
                    "value": trusted_release_authority_sha256,
                }
            ]
        ),
        table_entries=64,
        table_probes=8,
    )
    config_raw = V.CONFIG.serialize_effective_configuration(config)
    config_digest = sha(config_raw)
    build = F.build(synthetic=not measured_coverage)
    build["api_definition"]["sha256"] = V.ACCEPTED_MEMBER_SHA256[4]
    run = F.run(
        build,
        synthetic=not measured_coverage,
        configuration_digest=config_digest,
    )
    build_raw = V.COVERAGE.canonical_bytes(build)
    population = run["module_population"][0]
    runtime = population["runtime_identity"]
    segments = [
        {
            "end_address": 0x401000,
            "file_offset": 0,
            "permissions": "r-xp",
            "start_address": 0x400000,
        }
    ]
    module_row = {
        "build_id": runtime["build_id"],
        "build_logical_id": population["build_logical_id"],
        "byte_count": runtime["byte_count"],
        "device": 8,
        "inode": 99,
        "load_state_id": population["load_state_id"],
        "loaded_path": population["loaded_path"],
        "logical_id": population["logical_id"],
        "module_generation": population["load_generation"],
        "module_id": 1,
        "module_map_sha256": "1" * 64,
        "namespace_id": 0,
        "process_generation": GENERATION,
        "segments": segments,
        "sha256": runtime["sha256"],
    }
    module_row["module_map_sha256"] = V.RUNTIME.module_map_sha256(module_row)
    runtime["module_map_sha256"] = module_row["module_map_sha256"]
    run_raw = V.COVERAGE.canonical_bytes(run)
    context_raw = V.SEMANTIC.canonical_bytes(
        {
            "catalog_id": "oai_memprof_context",
            "entries": [],
            "schema": binding(
                6,
                "definition/context-schema-v1.json",
                V.ACCEPTED_MEMBER_SHA256[6],
            ),
            "version": {"major": 1, "minor": 0},
        }
    )
    callsite_raw = V.SEMANTIC.canonical_bytes(
        {
            "catalog_id": "oai_memprof_callsite",
            "entries": [],
            "schema": binding(
                5,
                "definition/callsite-rule-v1.json",
                V.ACCEPTED_MEMBER_SHA256[5],
            ),
            "version": {"major": 1, "minor": 0},
        }
    )
    diagnostics_raw = V.DIAGNOSTICS.make_diagnostics_bytes(
        definition_sha256=V.ACCEPTED_MEMBER_SHA256[8],
        mode_id=mode_id,
        process_generation=GENERATION,
        counter_rows=diagnostic_counter_rows(
            mode_id=mode_id, ready_thread_indices=ready_thread_indices
        ),
        ready_thread_indices=ready_thread_indices,
        producer_population_complete=complete,
    )
    clock_row = {
        "acquisition_source_id": 1,
        "acquisition_status_id": 1,
        "architecture_id": 1,
        "calibration_error_bound_ns": 100,
        "calibration_kind": 1,
        "calibration_span_ns": 0,
        "clock_kind": 1,
        "counter_frequency_denominator": 1,
        "counter_frequency_numerator": 2_400_000_000,
        "counter_invalid_observed": False,
        "counter_stability_status_id": 1,
        "observed_max_error_ns": 0,
        "process_generation": GENERATION,
        "realtime_discontinuity_observed": False,
        "samples": [
            {
                "counter": START_COUNTER,
                "monotonic_raw_after_ns": START_MONOTONIC,
                "monotonic_raw_before_ns": START_MONOTONIC,
                "realtime_unix_ns": START_REALTIME,
                "sample_ordinal": 1,
            },
            {
                "counter": FINAL_COUNTER,
                "monotonic_raw_after_ns": FINAL_MONOTONIC,
                "monotonic_raw_before_ns": FINAL_MONOTONIC,
                "realtime_unix_ns": FINAL_REALTIME,
                "sample_ordinal": 2,
            },
        ],
        "start_counter": START_COUNTER,
        "start_monotonic_raw_ns": START_MONOTONIC,
        "start_realtime_unix_ns": START_REALTIME,
    }
    return {
        "catalog/schema-bundle.json": bundle,
        "catalog/api.json": members[4],
        "catalog/context.json": context_raw,
        "catalog/callsite.json": callsite_raw,
        "catalog/thread.json": candidate_catalog(
            "oai_memprof_thread",
            [
                {
                    "process_generation": GENERATION,
                    "registration_ordinal": ordinal,
                    "thread_index": thread_index,
                }
                for ordinal, thread_index in enumerate(ready_thread_indices, 1)
            ],
            V.RUNTIME.THREAD_SCHEMA,
        ),
        "catalog/module.json": candidate_catalog(
            "oai_memprof_module", [module_row], V.RUNTIME.MODULE_SCHEMA
        ),
        "catalog/clock.json": candidate_catalog(
            "oai_memprof_clock", [clock_row], V.RUNTIME.CLOCK_SCHEMA
        ),
        "catalog/build-coverage.json": build_raw,
        "catalog/run-coverage.json": run_raw,
        "metadata/effective-config.json": config_raw,
        "status/diagnostics.json": diagnostics_raw,
        "status/pre-footer-status.json": V.STATUS.canonical_bytes(
            pre_footer(complete=complete)
        ),
    }


def count_for(kind: int, value: dict) -> int:
    if kind <= 7:
        return len(value["entries"])
    if kind == 8:
        return len(value["entries"])
    if kind == 9:
        return len(value["module_population"])
    if kind in (10, 12):
        return 1
    if kind == 11:
        return len(value["entries"])
    raise AssertionError(kind)


def assemble_stream(
    objects: dict[str, bytes],
    *,
    complete: bool = True,
    records: tuple = (),
    trailer_status: dict | None = None,
) -> bytes:
    config = V.CONFIG.validate_effective_configuration_bytes(
        objects["metadata/effective-config.json"]
    )
    build = V.COVERAGE.parse_canonical(objects["catalog/build-coverage.json"])
    run = V.COVERAGE.parse_canonical(objects["catalog/run-coverage.json"])
    opening = V.WIRE.OpeningHeader(
        page_size_bytes=4096,
        scope_kind=1,
        role_kind=1,
        clock_kind=1,
        calibration_kind=1,
        process_generation=GENERATION,
        counter_frequency_numerator=2_400_000_000,
        counter_frequency_denominator=1,
        calibration_error_bound_ns=100,
        calibration_span_ns=0,
        start_counter=START_COUNTER,
        start_monotonic_raw_ns=START_MONOTONIC,
        start_realtime_unix_ns=START_REALTIME,
        pid=12345,
        configured_thread_capacity=config["max_threads"],
        run_uuid=uuid.UUID(run["run_uuid"]).bytes,
        process_uuid=uuid.UUID(run["process_uuid"]).bytes,
        source_object_kind=1,
        source_object_algorithm=1,
        source_object_length=20,
        source_object_value=bytes.fromhex(build["build_identity"]["source_commit"])
        + bytes(12),
        primary_binary_sha256=bytes.fromhex(build["entries"][0]["sha256"]),
        schema_bundle_definition_sha256=bytes.fromhex(
            sha(objects["catalog/schema-bundle.json"])
        ),
        api_catalog_definition_sha256=bytes.fromhex(
            sha(objects["catalog/api.json"])
        ),
        callsite_catalog_definition_sha256=bytes.fromhex(
            V.ACCEPTED_MEMBER_SHA256[5]
        ),
        configuration_instance_sha256=bytes.fromhex(
            sha(objects["metadata/effective-config.json"])
        ),
        primary_build_id_sha256=bytes.fromhex(
            sha(bytes.fromhex(build["entries"][0]["build_id"]))
        ),
    )
    diagnostic = V.SEMANTIC.parse_canonical(objects["status/diagnostics.json"])
    diagnostic_entries = tuple(
        V.WIRE.DiagnosticTotalEntry(
            reason_id=row["reason_id"],
            class_flags=row["class_flags"],
            summary_flags=row["summary_flags"],
            saturating_total=row["saturating_total"],
            nonzero_counter_instances=row["nonzero_counter_instances"],
            saturated_counter_instances=row["saturated_counter_instances"],
        )
        for row in diagnostic["reason_totals"]
    )
    object_entries = []
    for kind in range(1, 13):
        path, flags = V.EXTERNAL_BY_KIND[kind]
        object_raw = objects[path]
        value = V.SEMANTIC.parse_canonical(object_raw)
        object_entries.append(
            V.WIRE.ObjectBindingEntry(
                object_kind=kind,
                format_id=1,
                object_flags=flags,
                schema_revision=1,
                entry_count=count_for(kind, value),
                byte_count=len(object_raw),
                sha256=hashlib.sha256(object_raw).digest(),
            )
        )
    opening_raw = V.WIRE.encode_opening_header(opening)
    chunk_raw = V.WIRE.encode_chunk(0, records) if records else b""
    prefix_raw = opening_raw + chunk_raw
    event_totals = {}
    for record in records:
        key = (record.event_kind, record.api_id)
        event_totals[key] = event_totals.get(key, 0) + 1
    event_entries = tuple(
        V.WIRE.EventTotalEntry(event_kind, api_id, count)
        for (event_kind, api_id), count in sorted(event_totals.items())
    )
    event_offset = V.WIRE.TRAILER_HEADER_BYTES
    diagnostic_offset = event_offset + len(event_entries) * 32
    object_offset = diagnostic_offset + len(diagnostic_entries) * 32
    body_bytes = object_offset + len(object_entries) * 64
    status = (
        V.STATUS.parse_canonical(objects["status/pre-footer-status.json"])
        if trailer_status is None else trailer_status
    )
    header = V.WIRE.TrailerHeader(
        trailer_body_bytes=body_bytes,
        process_generation=GENERATION,
        scope_kind=1,
        lifecycle_state=status["lifecycle_state"],
        payload_writer_state=status["payload_writer_state"],
        finalization_stage=status["finalization_stage"],
        terminal_flags=status["terminal_flags"],
        chunk_count=1 if records else 0,
        record_count=len(records),
        payload_bytes=len(records) * V.WIRE.EVENT_RECORD_BYTES,
        first_chunk_offset=512,
        chunks_end_offset=len(prefix_raw),
        active_generation=GENERATION,
        active_start_counter=START_COUNTER,
        cutoff_before_counter=120,
        cutoff_after_counter=121,
        quiescence_complete_counter=status["quiescence_complete_counter"],
        final_counter=FINAL_COUNTER,
        active_start_monotonic_raw_ns=START_MONOTONIC,
        final_monotonic_raw_ns=FINAL_MONOTONIC,
        final_realtime_unix_ns=FINAL_REALTIME,
        event_entry_count=len(event_entries),
        diagnostic_entry_count=len(diagnostic_entries),
        object_entry_count=len(object_entries),
        event_table_offset=event_offset,
        diagnostic_table_offset=diagnostic_offset,
        object_table_offset=object_offset,
        terminal_reason_code=status["reason_code"],
        diagnostic_loss_sum=sum(
            row.saturating_total for row in diagnostic_entries if row.class_flags & 1
        ),
        diagnostic_bypass_sum=sum(
            row.saturating_total for row in diagnostic_entries if row.class_flags & 2
        ),
        saturated_counter_instances=sum(
            row.saturated_counter_instances for row in diagnostic_entries
        ),
    )
    body = V.WIRE.TrailerBody(
        header, event_entries, diagnostic_entries, tuple(object_entries)
    )
    trailer_raw = V.WIRE.encode_trailer_body(body)
    footer = V.WIRE.Footer(
        trailer_offset=len(prefix_raw),
        trailer_body_bytes=len(trailer_raw),
        stream_bytes=len(prefix_raw) + len(trailer_raw) + V.WIRE.FOOTER_BYTES,
        prefix_bytes=len(prefix_raw),
        chunk_count=1 if records else 0,
        record_count=len(records),
        prefix_sha256=hashlib.sha256(prefix_raw).digest(),
        trailer_body_sha256=hashlib.sha256(trailer_raw).digest(),
        opening_header_sha256=hashlib.sha256(opening_raw).digest(),
    )
    return prefix_raw + trailer_raw + V.WIRE.encode_footer(footer)


def receipt(stream: bytes, *, verifier_sha256: str = VERIFIER_SHA256) -> bytes:
    decoded = V.WIRE.decode_container(stream)
    footer_preimage = stream[-V.WIRE.FOOTER_BYTES : -32]
    value = {
        "appender_close": "success",
        "exact_eof": True,
        "footer_preimage_sha256": sha(footer_preimage),
        "opening_header_sha256": decoded.footer.opening_header_sha256.hex(),
        "physical_bytes": len(stream),
        "prefix_sha256": decoded.footer.prefix_sha256.hex(),
        "schema": V.STATUS.SCHEMA_POST_CLOSE,
        "stream_path": STREAM_PATH,
        "trailer_body_sha256": decoded.footer.trailer_body_sha256.hex(),
        "verifier_definition_sha256": verifier_sha256,
        "whole_stream_sha256": sha(stream),
    }
    return V.STATUS.canonical_bytes(value)


def manifest(
    objects: dict[str, bytes],
    stream: bytes,
    receipt_raw: bytes,
    *,
    verifier_path: str = VERIFIER_PATH,
    verifier_definition: bytes = VERIFIER_DEFINITION,
    authenticated_artifacts: dict[str, bytes] | None = None,
) -> bytes:
    rows: dict[str, tuple[int, str]] = {}

    def add(path: str, raw: bytes) -> None:
        value = (len(raw), sha(raw))
        previous = rows.get(path)
        if previous is not None and previous != value:
            raise AssertionError(f"fixture manifest collision: {path}")
        rows[path] = value

    for path, raw in objects.items():
        add(path, raw)
    add(V.STATUS.POST_CLOSE_PATH, receipt_raw)
    add(STREAM_PATH, stream)
    add(verifier_path, verifier_definition)
    if authenticated_artifacts is not None:
        for path, raw in authenticated_artifacts.items():
            add(path, raw)

    value = {
        "entries": [
            {"bytes": byte_count, "path": path, "sha256": digest}
            for path, (byte_count, digest) in sorted(rows.items())
        ],
        "schema": V.STATUS.SCHEMA_MANIFEST,
    }
    return V.STATUS.canonical_bytes(value)


def _write_uint(raw: bytearray, offset: int, value: int, size: int) -> None:
    raw[offset : offset + size] = value.to_bytes(size, "little")


def _write_sample(
    raw: bytearray, offset: int, counter: int, monotonic_ns: int
) -> None:
    values = (
        counter,
        monotonic_ns,
        monotonic_ns,
        START_REALTIME + monotonic_ns - START_MONOTONIC,
    )
    for index, value in enumerate(values):
        _write_uint(raw, offset + index * 8, value, 8)


def _authenticated_artifacts(handoff_raw: bytes) -> dict[str, bytes]:
    return {
        V.ACCEPTED_PROCESS_HANDOFF_PATH: handoff_raw,
        V.ACCEPTED_PRODUCER_DEFINITION_PATH: V._PRODUCER_DEFINITION_BYTES,
        V.ACCEPTED_HANDOFF_DECODER_DEFINITION_PATH: (
            V._HANDOFF_DECODER_DEFINITION_BYTES
        ),
    }


def _authentication_kwargs(handoff_raw: bytes) -> dict:
    return {
        "process_handoff_bytes": handoff_raw,
        "producer_definition_path": V.ACCEPTED_PRODUCER_DEFINITION_PATH,
        "producer_definition_bytes": V._PRODUCER_DEFINITION_BYTES,
        "handoff_decoder_definition_path": (
            V.ACCEPTED_HANDOFF_DECODER_DEFINITION_PATH
        ),
        "handoff_decoder_definition_bytes": V._HANDOFF_DECODER_DEFINITION_BYTES,
    }


def trusted_release_authority(
    build: dict | None = None,
) -> tuple[bytes, str, dict[str, bytes], dict[str, bytes]]:
    build = F.build(synthetic=False) if build is None else build
    identity = build["build_identity"]
    if (
        identity["source_commit"] == "1" * 40
        and identity["source_tree"] == "2" * 40
    ):
        sources = _TRUSTED_RELEASE_AUTHORITY_SOURCES
        raw = _TRUSTED_RELEASE_AUTHORITY_RAW
        actual_sha256 = _TRUSTED_RELEASE_AUTHORITY_SHA256
    else:
        sources = V.accepted_trusted_release_source_bytes()
        raw = V.make_trusted_release_authority_bytes(
            commit=identity["source_commit"],
            tree=identity["source_tree"],
            source_bytes=sources,
        )
        actual_sha256 = sha(raw)
    if actual_sha256 != TRUSTED_RELEASE_AUTHORITY_FIXTURE_SHA256:
        raise AssertionError(
            "trusted-release authority fixture changed; review and explicitly "
            f"update its literal pin (observed {actual_sha256})"
        )
    return (
        raw,
        TRUSTED_RELEASE_AUTHORITY_FIXTURE_SHA256,
        sources,
        {V.TRUSTED_RELEASE_AUTHORITY_PATH: raw, **sources},
    )


def encode_authenticated_handoff(
    stream: bytes,
    objects: dict[str, bytes],
    *,
    writer_status: int,
    clock_status: int,
    prefooter_closed: bool = True,
) -> bytes:
    opening_raw = stream[: V.WIRE.OPENING_HEADER_BYTES]
    bootstrap_raw = objects["metadata/effective-config.json"]
    config = V.CONFIG.validate_effective_configuration_bytes(bootstrap_raw)
    module = V.SEMANTIC.parse_canonical(objects["catalog/module.json"])[
        "entries"
    ][0]
    maps_raw = "".join(
        (
            f"{segment['start_address']:x}-{segment['end_address']:x} "
            f"{segment['permissions']} {segment['file_offset']:x} "
            f"{os.major(module['device']):x}:{os.minor(module['device']):x} "
            f"{module['inode']} {module['loaded_path']}\n"
        )
        for segment in module["segments"]
    ).encode("utf-8")
    bootstrap_offset = 1152
    maps_offset = bootstrap_offset + len(bootstrap_raw)
    threads_offset = maps_offset + len(maps_raw)
    total_bytes = threads_offset + 32
    raw = bytearray(total_bytes)
    raw[:16] = b"OAIMPHANDOFFV1\0\0"
    _write_uint(raw, 16, 1, 2)
    _write_uint(raw, 18, 5, 2)
    _write_uint(raw, 20, 1152, 4)
    _write_uint(raw, 24, total_bytes, 8)
    _write_uint(raw, 32, bootstrap_offset, 8)
    _write_uint(raw, 40, len(bootstrap_raw), 8)
    _write_uint(raw, 48, maps_offset, 8)
    _write_uint(raw, 56, len(maps_raw), 8)
    _write_uint(raw, 64, threads_offset, 8)
    _write_uint(raw, 72, 0, 4)
    _write_uint(raw, 76, 448, 4)
    _write_uint(raw, 100, config["ring_records"], 4)
    _write_uint(raw, 104, 0 if writer_status == 0 else 1, 8)
    _write_uint(raw, 112, config["flush_records"], 4)
    _write_uint(raw, 116, 1, 2)
    _write_uint(raw, 120, config["flush_us"] * 1_000, 8)
    raw[128:640] = opening_raw

    writer_offset = 640
    _write_uint(raw, writer_offset, writer_status, 4)
    _write_uint(raw, writer_offset + 4, 0, 4)
    _write_uint(raw, writer_offset + 8, clock_status, 4)
    _write_uint(raw, writer_offset + 12, int(prefooter_closed), 4)
    _write_uint(raw, writer_offset + 16, 5 if writer_status == 9 else 0, 4)
    _write_uint(raw, writer_offset + 24, GENERATION, 8)
    raw[writer_offset + 104] = 3
    raw[writer_offset + 105] = 4
    _write_uint(raw, writer_offset + 112, 2_400_000_000, 8)
    _write_uint(raw, writer_offset + 120, 1, 8)
    _write_uint(raw, writer_offset + 128, 1, 2)
    _write_uint(raw, writer_offset + 130, 1, 2)
    raw[writer_offset + 132] = 1
    _write_sample(raw, writer_offset + 136, 120, 1_100)
    _write_sample(raw, writer_offset + 168, 121, 1_101)
    _write_sample(raw, writer_offset + 200, 130, 1_110)
    _write_sample(raw, writer_offset + 232, FINAL_COUNTER, FINAL_MONOTONIC)
    _write_uint(raw, writer_offset + 288, len(opening_raw), 8)
    _write_uint(raw, writer_offset + 296, 1, 8)
    _write_uint(raw, writer_offset + 304, 1, 8)

    raw[952:984] = hashlib.sha256(bootstrap_raw).digest()
    raw[984:1016] = hashlib.sha256(maps_raw).digest()
    raw[1016:1048] = hashlib.sha256(opening_raw).digest()
    _write_sample(raw, 1048, START_COUNTER, START_MONOTONIC)
    raw[1080:1112] = hashlib.sha256(opening_raw).digest()
    raw[bootstrap_offset:maps_offset] = bootstrap_raw
    raw[maps_offset:threads_offset] = maps_raw
    raw[-32:] = hashlib.sha256(raw[:-32]).digest()
    encoded = bytes(raw)
    decoded = V.HANDOFF.decode_process_handoff(encoded)
    if (
        decoded.writer.status != writer_status
        or decoded.writer.clock_status != clock_status
        or decoded.writer.prefooter_closed != prefooter_closed
    ):
        raise AssertionError("authenticated handoff fixture outcome mismatch")
    return encoded


def authenticated_terminal_fixture(
    writer_status: int, clock_status: int
) -> tuple[dict[str, bytes], bytes, bytes, bytes, dict, dict]:
    objects = base_external_objects(measured_coverage=True)
    seed_stream = assemble_stream(objects)
    handoff_raw = encode_authenticated_handoff(
        seed_stream,
        objects,
        writer_status=writer_status,
        clock_status=clock_status,
    )
    handoff = V.HANDOFF.decode_process_handoff(handoff_raw)
    objects["catalog/clock.json"] = V._catalog_bytes(
        "oai_memprof_clock",
        [V._handoff_clock_row(handoff)],
        V.RUNTIME.CLOCK_SCHEMA,
    )
    diagnostic_raw = V.DIAGNOSTICS.make_diagnostics_bytes(
        definition_sha256=V.ACCEPTED_MEMBER_SHA256[8],
        mode_id=handoff.writer.runtime_snapshot.mode_id,
        process_generation=GENERATION,
        counter_rows=V._handoff_diagnostic_rows(handoff),
        ready_thread_indices=(),
        producer_population_complete=True,
    )
    objects["status/diagnostics.json"] = diagnostic_raw
    _diagnostic, projection = V.DIAGNOSTICS.validate_diagnostics_bytes(
        diagnostic_raw,
        definition_sha256=V.ACCEPTED_MEMBER_SHA256[8],
        expected_mode_id=handoff.writer.runtime_snapshot.mode_id,
        expected_process_generation=GENERATION,
        ready_thread_indices=(),
        producer_population_complete=True,
    )
    status = V._handoff_pre_footer(handoff, projection)
    V.STATUS.validate_pre_footer(status)
    objects["status/pre-footer-status.json"] = V.STATUS.canonical_bytes(status)
    stream = assemble_stream(objects)
    if stream[: V.WIRE.OPENING_HEADER_BYTES] != handoff.opening_raw:
        raise AssertionError("authenticated handoff opening fixture mismatch")
    receipt_raw = receipt(stream)
    artifacts = _authenticated_artifacts(handoff_raw)
    manifest_raw = manifest(
        objects,
        stream,
        receipt_raw,
        authenticated_artifacts=artifacts,
    )
    return (
        objects,
        stream,
        receipt_raw,
        manifest_raw,
        _authentication_kwargs(handoff_raw),
        status,
    )


def fixture(*, complete: bool = True):
    objects = base_external_objects(complete=complete)
    stream = assemble_stream(objects, complete=complete)
    receipt_raw = receipt(stream)
    return objects, stream, receipt_raw, manifest(objects, stream, receipt_raw)


def a03_fixture(records=None):
    selected_records = a03_records() if records is None else records
    ready = (1, 2)
    objects = base_external_objects(
        mode_id=3,
        ready_thread_indices=ready,
    )
    stream = assemble_stream(objects, records=selected_records)
    receipt_raw = receipt(stream)
    return objects, stream, receipt_raw, manifest(objects, stream, receipt_raw), ready


def verify(
    objects,
    stream,
    receipt_raw=None,
    manifest_raw=None,
    *,
    verifier_path=VERIFIER_PATH,
    verifier_definition=VERIFIER_DEFINITION,
    authentication: dict | None = None,
    trusted_release: dict | None = None,
):
    keyword_arguments = {} if authentication is None else dict(authentication)
    if trusted_release is not None:
        keyword_arguments.update(trusted_release)
    return V.verify_archive_candidate(
        stream,
        objects,
        stream_path=STREAM_PATH,
        verifier_definition_path=verifier_path,
        verifier_definition_bytes=verifier_definition,
        post_close_receipt_bytes=receipt_raw,
        manifest_bytes=manifest_raw,
        **keyword_arguments,
    )


class IntegrationTests(unittest.TestCase):
    def test_event_classifier_source_is_exactly_trust_bound(self):
        self.assertEqual(
            EVENT_CLASSIFIER_SHA256, V.ACCEPTED_EVENT_CLASSIFIER_SHA256
        )
        self.assertEqual(
            EVENT_CLASSIFIER_DEFINITION,
            pathlib.Path(V.EVENT_CLASSIFIER.__file__).resolve().read_bytes(),
        )

    def assert_stage(self, expected: str, operation) -> None:
        with self.assertRaises(V.ArchiveVerificationError) as caught:
            operation()
        self.assertEqual(caught.exception.stage, expected)

    def assert_stage_detail(self, expected: str, detail: str, operation) -> None:
        with self.assertRaises(V.ArchiveVerificationError) as caught:
            operation()
        self.assertEqual(caught.exception.stage, expected)
        self.assertIn(detail, caught.exception.detail)

    def test_complete_candidate_reaches_status_promotion_but_not_scientific_admission(self):
        objects, stream, receipt_raw, manifest_raw = fixture()
        bundle = V.SEMANTIC.parse_canonical(objects["catalog/schema-bundle.json"])
        build = V.COVERAGE.parse_canonical(objects["catalog/build-coverage.json"])
        self.assertEqual(len(bundle["entries"]), 13)
        self.assertEqual(bundle["version"], {"major": 1, "minor": 5})
        self.assertEqual(V.SEMANTIC.BUNDLE_VERSION, {"major": 1, "minor": 5})
        self.assertEqual(bundle["entries"][12]["sha256"], V.ACCEPTED_MEMBER_SHA256[13])
        self.assertEqual(V.COVERAGE.POLICY_VERSION, {"major": 2, "minor": 0})
        self.assertEqual(V.COVERAGE.INSTANCE_VERSION, {"major": 1, "minor": 0})
        self.assertEqual(V.CONFIG.SCHEMA_VERSION, {"major": 1, "minor": 1})
        self.assertEqual(V.CONFIG.INSTANCE_VERSION, {"major": 1, "minor": 0})
        self.assertEqual(V.BUILD_EVIDENCE.VERSION, {"major": 1, "minor": 4})
        run = V.COVERAGE.parse_canonical(objects["catalog/run-coverage.json"])
        self.assertEqual(build["version"], {"major": 1, "minor": 0})
        self.assertEqual(run["version"], {"major": 1, "minor": 0})
        self.assertNotEqual(
            build["build_identity"]["build_configuration_sha256"],
            sha(objects["metadata/effective-config.json"]),
        )
        result = verify(objects, stream, receipt_raw, manifest_raw)
        self.assertEqual(result.terminal_outcome, "complete")
        self.assertIsNotNone(result.status_promotion)
        self.assertEqual(result.verified_stream.whole_stream_sha256, sha(stream))
        self.assertFalse(result.scientific_admission_complete)
        self.assertEqual(result.admission_blockers, V.SCIENTIFIC_ADMISSION_BLOCKERS)
        self.assertNotIn(
            "realloc_zero_policy_resolution",
            {field.name for field in dataclasses.fields(result)},
        )

    def test_trusted_release_authority_binds_exact_sources_build_and_manifest(self):
        build = F.build(synthetic=False)
        authority_raw, authority_sha256, sources, artifacts = (
            trusted_release_authority(build)
        )
        self.assertEqual(
            V.validate_trusted_release_authority(
                authority_raw,
                authority_sha256,
                sources,
                build=build,
            ),
            artifacts,
        )
        objects = base_external_objects(
            measured_coverage=True,
            trusted_release_authority_sha256=authority_sha256,
        )
        stream = assemble_stream(objects)
        receipt_raw = receipt(stream)
        manifest_raw = manifest(
            objects,
            stream,
            receipt_raw,
            authenticated_artifacts=artifacts,
        )
        trusted_release = {
            "trusted_release_authority_bytes": authority_raw,
            "trusted_release_authority_sha256": authority_sha256,
            "trusted_release_source_bytes": sources,
        }
        result = verify(
            objects,
            stream,
            receipt_raw,
            manifest_raw,
            trusted_release=trusted_release,
        )
        self.assertIsNotNone(result.status_promotion)
        self.assertFalse(result.scientific_admission_complete)
        self.assertNotIn(
            V.TRUSTED_RELEASE_AUTHORITY_BLOCKER,
            result.admission_blockers,
        )
        self.assertEqual(
            result.admission_blockers,
            (
                V.PROCESS_AUTHENTICATION_BLOCKER,
                V.BUILD_EVIDENCE_AUTHENTICATION_BLOCKER,
            ),
        )

        omitted = dict(artifacts)
        omitted.pop("definition/catalog-v1/runtime-catalog-v1.py")
        omitted_manifest = manifest(
            objects,
            stream,
            receipt_raw,
            authenticated_artifacts=omitted,
        )
        self.assert_stage(
            "status_chain",
            lambda: verify(
                objects,
                stream,
                receipt_raw,
                omitted_manifest,
                trusted_release=trusted_release,
            ),
        )

    def test_trusted_release_authority_rejects_partial_and_noncanonical_inputs(self):
        build = F.build(synthetic=False)
        authority_raw, authority_sha256, sources, _artifacts = (
            trusted_release_authority(build)
        )

        objects, stream, receipt_raw, manifest_raw = fixture()
        self.assert_stage(
            "input",
            lambda: verify(
                objects,
                stream,
                receipt_raw,
                manifest_raw,
                trusted_release={"trusted_release_authority_bytes": authority_raw},
            ),
        )

        def canonical_authority(value: dict) -> tuple[bytes, str]:
            raw = V.COVERAGE.canonical_bytes(value)
            return raw, sha(raw)

        value = V.COVERAGE.parse_canonical(authority_raw)
        malformed_cases: list[tuple[str, bytes, str, dict[str, bytes]]] = [
            ("wrong-pin", authority_raw, "0" * 64, sources),
            ("malformed", b"{}\n", sha(b"{}\n"), sources),
        ]
        exact_keys = copy.deepcopy(value)
        exact_keys["unexpected"] = True
        malformed_cases.append(
            ("exact-keys", *canonical_authority(exact_keys), sources)
        )
        unordered = copy.deepcopy(value)
        unordered["sources"].reverse()
        malformed_cases.append(("source-order", *canonical_authority(unordered), sources))
        duplicate = copy.deepcopy(value)
        duplicate["sources"][-1]["path"] = duplicate["sources"][0]["path"]
        malformed_cases.append(("source-duplicate", *canonical_authority(duplicate), sources))
        missing_row = copy.deepcopy(value)
        missing_row["sources"].pop()
        malformed_cases.append(
            ("source-row-missing", *canonical_authority(missing_row), sources)
        )
        extra_row = copy.deepcopy(value)
        extra_row["sources"].append(copy.deepcopy(extra_row["sources"][0]))
        malformed_cases.append(
            ("source-row-extra", *canonical_authority(extra_row), sources)
        )
        unclean = copy.deepcopy(value)
        unclean["git"]["clean"] = False
        malformed_cases.append(("authority-dirty", *canonical_authority(unclean), sources))
        for label, raw, pin, source_bytes in malformed_cases:
            with self.subTest(label=label):
                self.assert_stage(
                    "trusted_release_authority",
                    lambda raw=raw, pin=pin, source_bytes=source_bytes: V.validate_trusted_release_authority(
                        raw, pin, source_bytes, build=build
                    ),
                )

        missing = dict(sources)
        missing.pop("definition/catalog-v1/runtime-catalog-v1.py")
        extra = dict(sources)
        extra["definition/extra.py"] = b"extra\n"
        mutated = dict(sources)
        mutation_path = "definition/catalog-v1/runtime-catalog-v1.py"
        mutated[mutation_path] = mutated[mutation_path] + b"# mutant\n"
        for label, source_bytes in (
            ("source-missing", missing),
            ("source-extra", extra),
            ("source-byte-mutant", mutated),
        ):
            with self.subTest(label=label):
                self.assert_stage(
                    "trusted_release_authority",
                    lambda source_bytes=source_bytes: V.validate_trusted_release_authority(
                        authority_raw,
                        authority_sha256,
                        source_bytes,
                        build=build,
                    ),
                )

        for label, mutate in (
            (
                "commit",
                lambda identity: identity.update(source_commit="3" * 40),
            ),
            ("tree", lambda identity: identity.update(source_tree="4" * 40)),
            ("dirty", lambda identity: identity.update(dirty=True)),
        ):
            with self.subTest(label=label):
                mismatched_build = copy.deepcopy(build)
                mutate(mismatched_build["build_identity"])
                self.assert_stage(
                    "trusted_release_authority",
                    lambda mismatched_build=mismatched_build: V.validate_trusted_release_authority(
                        authority_raw,
                        authority_sha256,
                        sources,
                        build=mismatched_build,
                    ),
                )

    def test_structural_decode_has_first_rejection_precedence_and_input_is_unchanged(self):
        objects, stream, receipt_raw, manifest_raw = fixture()
        mutant = bytearray(stream)
        mutant[0] ^= 1
        snapshot = bytes(mutant)
        self.assert_stage(
            "input",
            lambda: verify(objects, mutant, receipt_raw, manifest_raw),
        )
        self.assertEqual(bytes(mutant), snapshot)
        corrupt = bytes(mutant)
        self.assert_stage(
            "structural",
            lambda: verify({"unexpected": b"{}\n"}, corrupt, receipt_raw, manifest_raw),
        )

    def test_exact_external_mapping_and_byte_hash_bindings_reject(self):
        objects, stream, receipt_raw, manifest_raw = fixture()
        missing = dict(objects)
        missing.pop("catalog/clock.json")
        self.assert_stage(
            "external_set", lambda: verify(missing, stream, receipt_raw, manifest_raw)
        )
        mutated = dict(objects)
        mutated["catalog/clock.json"] += b" "
        self.assert_stage(
            "external_binding",
            lambda: verify(mutated, stream, receipt_raw, manifest_raw),
        )

    def test_old_v14_bundle_rejects_after_recomputed_outer_bindings(self):
        objects = base_external_objects()
        historical = V.SEMANTIC.parse_canonical(objects["catalog/schema-bundle.json"])
        historical["version"] = {"major": 1, "minor": 4}
        historical["schema"]["sha256"] = HISTORICAL_BUNDLE_V14_ENTRIES[2][1]
        for object_type, (byte_count, digest) in HISTORICAL_BUNDLE_V14_ENTRIES.items():
            entry = historical["entries"][object_type - 1]
            entry["bytes"] = byte_count
            entry["sha256"] = digest
        historical_raw = V.SEMANTIC.canonical_bytes(historical)
        self.assertEqual(
            (len(historical_raw), sha(historical_raw)), HISTORICAL_BUNDLE_V14
        )
        objects["catalog/schema-bundle.json"] = historical_raw
        stream = assemble_stream(objects)
        receipt_raw = receipt(stream)
        self.assert_stage(
            "schema_bundle",
            lambda: verify(
                objects, stream, receipt_raw, manifest(objects, stream, receipt_raw)
            ),
        )

    def test_stale_realloc_policy_bindings_reject_at_coverage(self):
        objects = base_external_objects()
        build = V.COVERAGE.parse_canonical(objects["catalog/build-coverage.json"])
        run = V.COVERAGE.parse_canonical(objects["catalog/run-coverage.json"])
        build["policy"]["sha256"] = HISTORICAL_BUNDLE_V14_ENTRIES[9][1]
        build_raw = V.COVERAGE.canonical_bytes(build)
        run["policy"]["sha256"] = HISTORICAL_BUNDLE_V14_ENTRIES[9][1]
        run["build_coverage"]["sha256"] = sha(build_raw)
        objects["catalog/build-coverage.json"] = build_raw
        objects["catalog/run-coverage.json"] = V.COVERAGE.canonical_bytes(run)
        stream = assemble_stream(objects)
        receipt_raw = receipt(stream)
        self.assert_stage(
            "coverage",
            lambda: verify(
                objects,
                stream,
                receipt_raw,
                manifest(objects, stream, receipt_raw),
            ),
        )

    def test_stale_member2_and_wrong_member9_size_reject_at_schema_bundle(self):
        for mutation in ("stale_member2", "wrong_member9_size"):
            with self.subTest(mutation=mutation):
                objects = base_external_objects()
                bundle = V.SEMANTIC.parse_canonical(
                    objects["catalog/schema-bundle.json"]
                )
                if mutation == "stale_member2":
                    bundle["schema"]["sha256"] = HISTORICAL_BUNDLE_V14_ENTRIES[2][1]
                    bundle["entries"][1]["bytes"] = HISTORICAL_BUNDLE_V14_ENTRIES[2][0]
                    bundle["entries"][1]["sha256"] = HISTORICAL_BUNDLE_V14_ENTRIES[2][1]
                else:
                    bundle["entries"][8]["bytes"] = HISTORICAL_BUNDLE_V14_ENTRIES[9][0]
                objects["catalog/schema-bundle.json"] = V.SEMANTIC.canonical_bytes(
                    bundle
                )
                stream = assemble_stream(objects)
                receipt_raw = receipt(stream)
                self.assert_stage(
                    "schema_bundle",
                    lambda: verify(
                        objects,
                        stream,
                        receipt_raw,
                        manifest(objects, stream, receipt_raw),
                    ),
                )

    def test_wrong_bundle_relation_rejects_after_recomputed_stream_bindings(self):
        objects = base_external_objects()
        value = V.SEMANTIC.parse_canonical(objects["catalog/schema-bundle.json"])
        value["entries"][8]["sha256"] = "b" * 64
        objects["catalog/schema-bundle.json"] = V.SEMANTIC.canonical_bytes(value)
        stream = assemble_stream(objects)
        receipt_raw = receipt(stream)
        self.assert_stage(
            "schema_bundle",
            lambda: verify(objects, stream, receipt_raw, manifest(objects, stream, receipt_raw)),
        )

    def test_wrong_run_to_build_coverage_relation_rejects(self):
        objects = base_external_objects()
        value = V.COVERAGE.parse_canonical(objects["catalog/run-coverage.json"])
        value["build_coverage"]["sha256"] = "b" * 64
        objects["catalog/run-coverage.json"] = V.COVERAGE.canonical_bytes(value)
        stream = assemble_stream(objects)
        receipt_raw = receipt(stream)
        self.assert_stage(
            "coverage",
            lambda: verify(objects, stream, receipt_raw, manifest(objects, stream, receipt_raw)),
        )

    def test_wrong_configuration_coverage_relation_rejects(self):
        objects = base_external_objects()
        value = V.COVERAGE.parse_canonical(objects["catalog/run-coverage.json"])
        value["configuration_instance_sha256"] = "b" * 64
        objects["catalog/run-coverage.json"] = V.COVERAGE.canonical_bytes(value)
        stream = assemble_stream(objects)
        receipt_raw = receipt(stream)
        self.assert_stage(
            "coverage",
            lambda: verify(objects, stream, receipt_raw, manifest(objects, stream, receipt_raw)),
        )

    def test_runtime_module_must_reconcile_to_validated_run_population(self):
        objects = base_external_objects()
        value = V.SEMANTIC.parse_canonical(objects["catalog/module.json"])
        value["entries"][0]["loaded_path"] = "/opt/oai/substituted-softmodem"
        objects["catalog/module.json"] = V.SEMANTIC.canonical_bytes(value)
        stream = assemble_stream(objects)
        receipt_raw = receipt(stream)
        self.assert_stage(
            "runtime_catalog",
            lambda: verify(
                objects,
                stream,
                receipt_raw,
                manifest(objects, stream, receipt_raw),
            ),
        )

    def test_callsite_candidate_uses_accepted_mode_and_module_relations(self):
        objects = base_external_objects()
        value = V.SEMANTIC.parse_canonical(objects["catalog/callsite.json"])
        value["entries"] = [
            {
                "callsite_id": 1,
                "module_generation": GENERATION,
                "module_id": 1,
                "process_generation": GENERATION,
                "raw_address": 4096,
            }
        ]
        objects["catalog/callsite.json"] = V.SEMANTIC.canonical_bytes(value)
        stream = assemble_stream(objects)
        receipt_raw = receipt(stream)
        self.assert_stage(
            "callsite",
            lambda: verify(
                objects,
                stream,
                receipt_raw,
                manifest(objects, stream, receipt_raw),
            ),
        )

    def test_diagnostics_rows_must_project_exactly_to_counters_and_terminal(self):
        objects = base_external_objects()
        value = V.SEMANTIC.parse_canonical(objects["status/diagnostics.json"])
        value["reason_totals"][0]["saturating_total"] = 1
        objects["status/diagnostics.json"] = V.SEMANTIC.canonical_bytes(value)
        stream = assemble_stream(objects)
        receipt_raw = receipt(stream)
        self.assert_stage(
            "diagnostics",
            lambda: verify(objects, stream, receipt_raw, manifest(objects, stream, receipt_raw)),
        )

    def test_recomputed_prefooter_and_manifest_substitution_still_rejects_terminal_mismatch(self):
        objects = base_external_objects()
        value = V.STATUS.parse_canonical(objects[V.STATUS.PRE_FOOTER_PATH])
        value["final_counter"] += 1
        objects[V.STATUS.PRE_FOOTER_PATH] = V.STATUS.canonical_bytes(value)
        stream = assemble_stream(objects)
        receipt_raw = receipt(stream)
        manifest_raw = manifest(objects, stream, receipt_raw)
        self.assert_stage(
            "terminal_binding",
            lambda: verify(objects, stream, receipt_raw, manifest_raw),
        )

    def test_paired_path_bytes_receipt_and_manifest_cannot_substitute_trust_root(self):
        objects, stream, _receipt_raw, _manifest_raw = fixture()
        alternate_path = "definition/alternate-verifier-v1.py"
        alternate_definition = b"fully paired alternate verifier bytes\n"
        alternate_receipt = receipt(stream, verifier_sha256=sha(alternate_definition))
        alternate_manifest = manifest(
            objects,
            stream,
            alternate_receipt,
            verifier_path=alternate_path,
            verifier_definition=alternate_definition,
        )
        self.assert_stage(
            "input",
            lambda: verify(
                objects,
                stream,
                alternate_receipt,
                alternate_manifest,
                verifier_path=alternate_path,
                verifier_definition=alternate_definition,
            ),
        )

    def test_manifest_must_bind_exact_verifier_definition_size(self):
        objects, stream, receipt_raw, manifest_raw = fixture()
        value = V.STATUS.parse_canonical(manifest_raw)
        for entry in value["entries"]:
            if entry["path"] == VERIFIER_PATH:
                entry["bytes"] += 1
                break
        self.assert_stage(
            "status_chain",
            lambda: verify(
                objects,
                stream,
                receipt_raw,
                V.STATUS.canonical_bytes(value),
            ),
        )

    def test_manifest_must_bind_every_external_object_and_accepted_verifier(self):
        objects, stream, receipt_raw, _manifest_raw = fixture()
        rows = [(path, len(raw), sha(raw)) for path, raw in objects.items()]
        rows = [row for row in rows if row[0] != "catalog/clock.json"]
        rows.extend(
            (
                (V.STATUS.POST_CLOSE_PATH, len(receipt_raw), sha(receipt_raw)),
                (STREAM_PATH, len(stream), sha(stream)),
                (VERIFIER_PATH, len(VERIFIER_DEFINITION), VERIFIER_SHA256),
            )
        )
        manifest_raw = V.STATUS.canonical_bytes(
            {
                "entries": [
                    {"bytes": size, "path": path, "sha256": digest}
                    for path, size, digest in sorted(rows)
                ],
                "schema": V.STATUS.SCHEMA_MANIFEST,
            }
        )
        self.assert_stage(
            "status_chain",
            lambda: verify(objects, stream, receipt_raw, manifest_raw),
        )

    def test_event_classifier_rejects_impossible_flags_boundary_and_predecessor(self):
        records = list(a03_records())
        mutants = [
            dataclasses.replace(records[0], flags=records[0].flags | (1 << 13)),
            dataclasses.replace(records[0], flags=records[0].flags | (1 << 18)),
            dataclasses.replace(records[1], address_before=0x2000),
            dataclasses.replace(records[1], arg0=65),
        ]
        for index, mutant in enumerate(mutants):
            candidate = list(records)
            candidate[0 if index < 2 else 1] = mutant
            objects, stream, receipt_raw, manifest_raw, ready = a03_fixture(tuple(candidate))
            with self.subTest(index=index):
                self.assert_stage(
                    "record_semantics" if index < 2 else "record_replay",
                    lambda: verify(
                        objects, stream, receipt_raw, manifest_raw,
                    ),
                )

    def test_exact_mode_sequence_gap_is_retained_as_scientific_blocker(self):
        records = (
            V.WIRE.EventRecord(
                thread_sequence=1, counter_enter=110, counter_exit=111,
                address_before=0, address_after=0x1000, arg0=64, arg1=0, arg2=0,
                context_id=0, callsite_id=0, thread_index=1, flags=0x01000866,
                result_code=0, api_id=1, event_kind=1,
                cpu_enter=0xFFFF, cpu_exit=0xFFFF,
            ),
            V.WIRE.EventRecord(
                thread_sequence=3, counter_enter=112, counter_exit=113,
                address_before=0, address_after=0x2000, arg0=128, arg1=0, arg2=0,
                context_id=0, callsite_id=0, thread_index=1, flags=0x01000866,
                result_code=0, api_id=1, event_kind=1,
                cpu_enter=0xFFFF, cpu_exit=0xFFFF,
            ),
        )
        objects = base_external_objects(
            mode_id=4, ready_thread_indices=(1,)
        )
        stream = assemble_stream(objects, records=records)
        receipt_raw = receipt(stream)
        result = verify(
            objects, stream, receipt_raw, manifest(objects, stream, receipt_raw),
        )
        self.assertEqual(result.terminal_outcome, "complete")
        self.assertIsNotNone(result.status_promotion)
        self.assertFalse(result.scientific_admission_complete)
        self.assertIn(
            "exact-mode thread sequence population has 1 missing transactions",
            result.admission_blockers,
        )

    def test_exact_mode_requested_bytes_match_all_runtime_api_accounting(self):
        source_by_api = {
            1: "arg0",
            2: "arg2",
            3: "arg0",
            5: "arg0",
            6: "arg1",
            7: "arg1",
            8: "arg1",
            9: "arg0",
            10: "arg0",
        }
        records = []
        expected_total = 0
        for api_id in range(1, 13):
            record = V.WIRE.EventRecord(
                thread_sequence=api_id,
                counter_enter=100 + api_id * 2,
                counter_exit=101 + api_id * 2,
                address_before=0,
                address_after=0,
                arg0=api_id * 100 + 1,
                arg1=api_id * 100 + 2,
                arg2=api_id * 100 + 3,
                context_id=0,
                callsite_id=0,
                thread_index=1,
                flags=0,
                result_code=0,
                api_id=api_id,
                event_kind=1,
                cpu_enter=0xFFFF,
                cpu_exit=0xFFFF,
            )
            records.append(record)
            source = source_by_api.get(api_id)
            expected = 0 if source is None else getattr(record, source)
            expected_total = V._sat_add(expected_total, expected)
            with self.subTest(api_id=api_id):
                self.assertEqual(V._record_requested_bytes(record), expected)

        observed_total = 0
        for record in records:
            observed_total = V._sat_add(
                observed_total, V._record_requested_bytes(record)
            )
        self.assertEqual(observed_total, expected_total)

    def test_measured_realloc_policy_resolves_before_transition_classification(self):
        records = (V.WIRE.EventRecord(
            thread_sequence=1,
            counter_enter=110,
            counter_exit=111,
            address_before=0x1000,
            address_after=0,
            arg0=0,
            arg1=0,
            arg2=0,
            context_id=0,
            callsite_id=0,
            thread_index=1,
            flags=0x01001267,
            result_code=0,
            api_id=3,
            event_kind=2,
            cpu_enter=0xFFFF,
            cpu_exit=0xFFFF,
        ),)
        objects = base_external_objects(
            mode_id=4,
            ready_thread_indices=(1,),
            measured_coverage=True,
        )
        stream = assemble_stream(objects, records=records)
        receipt_raw = receipt(stream)
        result = verify(
            objects,
            stream,
            receipt_raw,
            manifest(objects, stream, receipt_raw),
        )
        self.assertEqual(result.terminal_outcome, "complete")
        self.assertFalse(result.scientific_admission_complete)

        build = V.COVERAGE.parse_canonical(objects["catalog/build-coverage.json"])
        build["entries"][0]["realloc_zero_policy_id"] = 2
        objects["catalog/build-coverage.json"] = V.COVERAGE.canonical_bytes(build)
        run = V.COVERAGE.parse_canonical(objects["catalog/run-coverage.json"])
        run["build_coverage"]["sha256"] = sha(
            objects["catalog/build-coverage.json"]
        )
        objects["catalog/run-coverage.json"] = V.COVERAGE.canonical_bytes(run)
        stream = assemble_stream(objects, records=records)
        receipt_raw = receipt(stream)
        self.assert_stage(
            "record_semantics",
            lambda: verify(
                objects,
                stream,
                receipt_raw,
                manifest(objects, stream, receipt_raw),
            ),
        )

    def test_a03_selection_composes_successor_predecessor_and_admission_boundary(self):
        objects, stream, receipt_raw, manifest_raw, ready = a03_fixture()
        result = verify(
            objects,
            stream,
            receipt_raw,
            manifest_raw,
        )
        self.assertEqual(result.terminal_outcome, "complete")
        self.assertIsNotNone(result.status_promotion)
        self.assertFalse(result.scientific_admission_complete)
        self.assertIn(
            "synthetic A03 seed validates mechanics only and is not scientifically admissible",
            result.admission_blockers,
        )
        self.assertEqual(
            result.admission_blockers[-len(V.SCIENTIFIC_ADMISSION_BLOCKERS) :],
            V.SCIENTIFIC_ADMISSION_BLOCKERS,
        )

    def test_a03_successor_selected_bit_must_match_current_instance_key(self):
        records = list(a03_records())
        records[0] = dataclasses.replace(
            records[0], flags=records[0].flags ^ (1 << 16)
        )
        objects, stream, receipt_raw, manifest_raw, ready = a03_fixture(tuple(records))
        self.assert_stage_detail(
            "record_semantics",
            "case is not emitted for transition/mode",
            lambda: verify(
                objects,
                stream,
                receipt_raw,
                manifest_raw,
            ),
        )

    def test_a03_predecessor_selected_bit_must_match_origin_instance_key(self):
        records = list(a03_records())
        records[1] = dataclasses.replace(
            records[1], flags=records[1].flags ^ (1 << 15)
        )
        objects, stream, receipt_raw, manifest_raw, ready = a03_fixture(tuple(records))
        self.assert_stage_detail(
            "record_semantics",
            "impossible frozen bit combination",
            lambda: verify(
                objects,
                stream,
                receipt_raw,
                manifest_raw,
            ),
        )

    def test_a03_emission_requires_at_least_one_selected_endpoint(self):
        unselected = dataclasses.replace(
            a03_records()[0],
            thread_sequence=2,
            flags=0x01000866,
        )
        objects, stream, receipt_raw, manifest_raw, _ready = a03_fixture((unselected,))
        self.assert_stage_detail(
            "record_semantics",
            "case is not emitted for transition/mode",
            lambda: verify(
                objects,
                stream,
                receipt_raw,
                manifest_raw,
            ),
        )

    def test_a03_replay_rejects_composed_same_address_lifetime_overlap(self):
        objects, stream, receipt_raw, manifest_raw, _ready = a03_fixture(
            a03_overlapping_ended_records()
        )
        self.assert_stage_detail(
            "record_replay",
            "intersecting same-address lifetimes",
            lambda: verify(objects, stream, receipt_raw, manifest_raw),
        )

    def test_authenticated_runtime_libc_requires_one_executable_map_identity(self):
        libc_raw = b"synthetic-libc-artifact"
        build = {
            "dependencies": [
                {
                    "dependency_id": "glibc_runtime",
                    "evidence_state_id": 1,
                    "name": "libc.so.6",
                    "sha256": sha(libc_raw),
                    "version": "2.39",
                }
            ]
        }
        executable = (
            b"1000-2000 r--p 0 8:1 71 /usr/lib/libc.so.6\n"
            b"2000-3000 r-xp 1000 8:1 71 /usr/lib/libc.so.6\n"
        )
        artifacts = {"input/build-evidence/libc.so.6": libc_raw}
        V._validate_authenticated_runtime_libc(
            types.SimpleNamespace(maps_bytes=executable), build, artifacts
        )

        mutants = (
            executable.replace(b"r-xp", b"r--p"),
            executable
            + b"4000-5000 r-xp 0 8:1 72 /usr/lib/libc-2.39.so\n",
        )
        for maps_raw in mutants:
            with self.subTest(maps_raw=maps_raw), self.assertRaises(
                V.ArchiveVerificationError
            ):
                V._validate_authenticated_runtime_libc(
                    types.SimpleNamespace(maps_bytes=maps_raw), build, artifacts
                )
        with self.assertRaises(V.ArchiveVerificationError):
            V._validate_authenticated_runtime_libc(
                types.SimpleNamespace(maps_bytes=executable),
                build,
                {"input/build-evidence/libc.so.6": libc_raw + b"!"},
            )

    def test_authenticated_handoff_exact_terminal_mappings(self):
        cases = (
            (0, 0, "complete", (5, 0, 5)),
            (9, 0, "failed", (6, 6, 6)),
            (11, 5, "incomplete", (7, 7, 5)),
        )
        for writer_status, clock_status, outcome, terminal in cases:
            with self.subTest(writer_status=writer_status):
                (
                    objects,
                    stream,
                    receipt_raw,
                    manifest_raw,
                    authentication,
                    status,
                ) = authenticated_terminal_fixture(writer_status, clock_status)
                result = verify(
                    objects,
                    stream,
                    receipt_raw,
                    manifest_raw,
                    authentication=authentication,
                )
                decoded = V.WIRE.decode_container(stream)
                header = decoded.trailer_body.header
                self.assertEqual(result.terminal_outcome, outcome)
                self.assertEqual(
                    (
                        header.lifecycle_state,
                        header.terminal_reason_code,
                        header.payload_writer_state,
                    ),
                    terminal,
                )
                self.assertEqual(
                    (
                        status["lifecycle_state"],
                        status["reason_code"],
                        status["payload_writer_state"],
                    ),
                    terminal,
                )
                self.assertNotIn(
                    V.PROCESS_AUTHENTICATION_BLOCKER,
                    result.admission_blockers,
                )
                if outcome == "complete":
                    self.assertIsNotNone(result.status_promotion)
                else:
                    self.assertIsNone(result.status_promotion)
                    self.assertFalse(result.scientific_admission_complete)
                    self.assertIn(
                        "negative terminal outcome",
                        result.admission_blockers[0],
                    )

    def test_authenticated_unrepresentable_writer_status_rejects(self):
        objects = base_external_objects(measured_coverage=True)
        stream = assemble_stream(objects)
        handoff_raw = encode_authenticated_handoff(
            stream,
            objects,
            writer_status=10,
            clock_status=0,
        )
        receipt_raw = receipt(stream)
        manifest_raw = manifest(
            objects,
            stream,
            receipt_raw,
            authenticated_artifacts=_authenticated_artifacts(handoff_raw),
        )
        self.assert_stage_detail(
            "process_handoff",
            "has no authenticated terminal representation",
            lambda: verify(
                objects,
                stream,
                receipt_raw,
                manifest_raw,
                authentication=_authentication_kwargs(handoff_raw),
            ),
        )

    def test_authenticated_pre_footer_trailer_mismatch_rejects(self):
        (
            objects,
            _stream,
            _receipt,
            _manifest,
            authentication,
            trailer_status,
        ) = authenticated_terminal_fixture(9, 0)
        objects = dict(objects)
        mismatched = dict(trailer_status)
        mismatched["scope_kind"] = 2
        V.STATUS.validate_pre_footer(mismatched)
        objects["status/pre-footer-status.json"] = V.STATUS.canonical_bytes(
            mismatched
        )
        stream = assemble_stream(objects, trailer_status=trailer_status)
        receipt_raw = receipt(stream)
        manifest_raw = manifest(
            objects,
            stream,
            receipt_raw,
            authenticated_artifacts=_authenticated_artifacts(
                authentication["process_handoff_bytes"]
            ),
        )
        self.assert_stage_detail(
            "terminal_binding",
            "pre_footer.scope_kind: trailer mismatch",
            lambda: verify(
                objects,
                stream,
                receipt_raw,
                manifest_raw,
                authentication=authentication,
            ),
        )

    def test_negative_terminal_is_retained_and_never_promoted(self):
        objects, stream, receipt_raw, manifest_raw = fixture(complete=False)
        result = verify(objects, stream, receipt_raw, manifest_raw)
        self.assertEqual(result.terminal_outcome, "incomplete")
        self.assertIsNone(result.status_promotion)
        self.assertIn("negative terminal outcome", result.admission_blockers[0])


if __name__ == "__main__":
    unittest.main()
