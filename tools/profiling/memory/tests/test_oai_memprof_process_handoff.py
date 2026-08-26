#!/usr/bin/env python3
"""Independent literal and mutation tests for the process-handoff v1 decoder."""

from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path.cwd()
MODULE_PATH = ROOT / "tools/profiling/memory/oai_memprof_process_handoff.py"
LITERAL_PATH = ROOT / "common/utils/memprof/tests/process_handoff_v1.hex"


def load_module():
    spec = importlib.util.spec_from_file_location("_handoff_under_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("handoff decoder unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SUBJECT = load_module()


def refreshed(raw: bytearray) -> bytes:
    raw[-32:] = hashlib.sha256(raw[:-32]).digest()
    return bytes(raw)


class ProcessHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.literal = bytes.fromhex(LITERAL_PATH.read_text(encoding="ascii").strip())

    def test_independent_literal_decodes_exactly(self) -> None:
        value = SUBJECT.decode_process_handoff(self.literal)
        self.assertEqual(len(self.literal), 1745)
        self.assertEqual(int.from_bytes(self.literal[16:20], "little"), (5 << 16) | 1)
        self.assertEqual(SUBJECT.ADMITTED_API_COUNT, 12)
        self.assertEqual(value.opening.process_generation, 7)
        self.assertEqual(value.opening_sample.counter, value.opening.start_counter)
        self.assertEqual(value.opening_sample.monotonic_raw_before_ns, 999_999_950)
        self.assertEqual(value.opening_sample.monotonic_raw_after_ns, 1_000_000_050)
        self.assertEqual(value.opening_sample.realtime_unix_ns, value.opening.start_realtime_unix_ns)
        self.assertEqual(value.writer.record_count, 2)
        self.assertEqual(value.writer.chunk_count, 1)
        self.assertEqual(value.writer.payload_bytes, 192)
        self.assertEqual(value.writer.stream_bytes, 736)
        self.assertEqual(value.writer.runtime_snapshot.admitted_transactions, 2)
        self.assertEqual(value.writer.runtime_snapshot.completed_transactions, 2)
        self.assertEqual(value.writer.runtime_snapshot.emitted_events, 2)
        self.assertEqual(value.writer.runtime_snapshot.requested_bytes, 96)
        self.assertEqual(
            (
                value.writer.runtime_snapshot.table_entries,
                value.writer.runtime_snapshot.sample_seed,
                value.writer.runtime_snapshot.sample_threshold,
                value.writer.runtime_snapshot.table_probes,
                value.writer.runtime_snapshot.table_shards,
            ),
            (0, 0, 0, 0, 0),
        )
        self.assertEqual(value.ring_records, 8)
        self.assertEqual(value.flush_records, 2)
        self.assertEqual(value.flush_interval_ns, 100_000_000)
        self.assertEqual(value.realloc_zero_policy_id, 1)
        self.assertEqual(value.threads[0].api_attempts, (1, 1) + (0,) * 30)
        self.assertEqual(value.threads[0].diagnostic_values, (0,) * 10)
        self.assertEqual(value.bootstrap_sha256, value.opening.configuration_instance_sha256)
        self.assertEqual(value.prefix_sha256, self.literal[1080:1112])
        self.assertEqual(value.handoff_sha256, self.literal[-32:])

    def test_requires_immutable_bytes_and_self_hash_precedes_relations(self) -> None:
        with self.assertRaisesRegex(SUBJECT.ProcessHandoffError, r"^handoff: immutable bytes required$"):
            SUBJECT.decode_process_handoff(bytearray(self.literal))
        mutant = bytearray(self.literal)
        mutant[118] = 1
        with self.assertRaisesRegex(SUBJECT.ProcessHandoffError, r"^handoff_sha256: self-hash mismatch$"):
            SUBJECT.decode_process_handoff(bytes(mutant))
        with self.assertRaisesRegex(SUBJECT.ProcessHandoffError, r"^reserved: nonzero reserved byte$"):
            SUBJECT.decode_process_handoff(refreshed(mutant))

    def test_component_and_runtime_relations_reject(self) -> None:
        mutant = bytearray(self.literal)
        mutant[18:20] = (4).to_bytes(2, "little")
        with self.assertRaisesRegex(SUBJECT.ProcessHandoffError, r"^version: exact 1.5 required$"):
            SUBJECT.decode_process_handoff(refreshed(mutant))

        for api in range(4, SUBJECT.ADMITTED_API_COUNT):
            with self.subTest(admitted_api=api + 1):
                mutant = bytearray(self.literal)
                mutant[1265 + 24 + 1 * 8 : 1265 + 24 + 2 * 8] = bytes(8)
                mutant[1265 + 24 + api * 8 : 1265 + 24 + (api + 1) * 8] = (1).to_bytes(8, "little")
                value = SUBJECT.decode_process_handoff(refreshed(mutant))
                self.assertEqual(value.threads[0].api_attempts[1], 0)
                self.assertEqual(value.threads[0].api_attempts[api], 1)

        mutant = bytearray(self.literal)
        mutant[1265 + 24 + 1 * 8 : 1265 + 24 + 2 * 8] = bytes(8)
        reserved = SUBJECT.ADMITTED_API_COUNT
        mutant[1265 + 24 + reserved * 8 : 1265 + 24 + (reserved + 1) * 8] = (1).to_bytes(8, "little")
        with self.assertRaisesRegex(
            SUBJECT.ProcessHandoffError,
            r"^threads\[0\]: reserved API counter slot is nonzero$",
        ):
            SUBJECT.decode_process_handoff(refreshed(mutant))

        mutant = bytearray(self.literal)
        mutant[1152] ^= 1
        mutated = refreshed(mutant)
        with self.assertRaisesRegex(SUBJECT.ProcessHandoffError, r"^component_sha256: component hash mismatch$"):
            SUBJECT.decode_process_handoff(mutated)

        mutant = bytearray(self.literal)
        mutant[1265 + 368] = 1
        with self.assertRaisesRegex(SUBJECT.ProcessHandoffError, r"^threads\[0\]: runtime/diagnostic projection mismatch$"):
            SUBJECT.decode_process_handoff(refreshed(mutant))

        mutant = bytearray(self.literal)
        mutant[640 + 288 : 640 + 296] = (737).to_bytes(8, "little")
        with self.assertRaisesRegex(SUBJECT.ProcessHandoffError, r"^writer: successful writer relation mismatch$"):
            SUBJECT.decode_process_handoff(refreshed(mutant))

    def test_negative_writer_evidence_is_retained_and_diagnostic_bound(self) -> None:
        mutant = bytearray(self.literal)
        mutant[640] = 9
        mutant[640 + 288 : 640 + 296] = (768).to_bytes(8, "little")
        mutant[104:112] = (1).to_bytes(8, "little")
        value = SUBJECT.decode_process_handoff(refreshed(mutant))
        self.assertEqual(value.writer.status, 9)
        self.assertEqual(value.writer.stream_bytes, 768)
        self.assertEqual(value.writer_io_or_finalization_failures, 1)

        mutant = bytearray(self.literal)
        mutant[640] = 11
        mutant[640 + 8 : 640 + 12] = (4).to_bytes(4, "little")
        mutant[640 + 232 : 640 + 264] = bytes(32)
        mutant[104:112] = (1).to_bytes(8, "little")
        value = SUBJECT.decode_process_handoff(refreshed(mutant))
        self.assertEqual(value.writer.clock_status, 4)
        self.assertEqual(value.writer.final_sample, SUBJECT.ClockSample(0, 0, 0, 0))

        mutant[640 + 136 : 640 + 168] = bytes(32)
        with self.assertRaisesRegex(
            SUBJECT.ProcessHandoffError,
            r"^clock_samples\[1\]: nonzero sample after zero suffix$",
        ):
            SUBJECT.decode_process_handoff(refreshed(mutant))

        mutant = bytearray(self.literal)
        mutant[640] = 9
        mutant[640 + 288 : 640 + 296] = (768).to_bytes(8, "little")
        mutant[104:112] = bytes(8)
        with self.assertRaisesRegex(SUBJECT.ProcessHandoffError, r"^writer: failed writer missing diagnostic$"):
            SUBJECT.decode_process_handoff(refreshed(mutant))

    def test_opening_sample_bracket_and_final_reserved_bytes_are_exact(self) -> None:
        mutant = bytearray(self.literal)
        mutant[1056:1064] = (1_000_000_001).to_bytes(8, "little")
        with self.assertRaisesRegex(
            SUBJECT.ProcessHandoffError,
            r"^opening_sample: opening midpoint is not bound to acquisition bracket$",
        ):
            SUBJECT.decode_process_handoff(refreshed(mutant))

        mutant = bytearray(self.literal)
        mutant[1144] = 1
        with self.assertRaisesRegex(SUBJECT.ProcessHandoffError, r"^reserved: nonzero reserved byte$"):
            SUBJECT.decode_process_handoff(refreshed(mutant))

    def test_runtime_configuration_fields_are_self_authenticated_and_bounded(self) -> None:
        mutations = (
            (100, 4, 3),
            (112, 4, 0),
            (112, 4, 65_537),
            (116, 2, 0),
            (116, 2, 3),
        )
        for offset, size, value in mutations:
            with self.subTest(offset=offset, value=value):
                mutant = bytearray(self.literal)
                mutant[offset : offset + size] = value.to_bytes(size, "little")
                with self.assertRaisesRegex(
                    SUBJECT.ProcessHandoffError,
                    r"^runtime_configuration: ring/flush/realloc-policy domain mismatch$",
                ):
                    SUBJECT.decode_process_handoff(refreshed(mutant))

    def test_sampled_controls_and_diagnostic_projection_are_exact(self) -> None:
        sampled = bytearray(self.literal)
        sampled[640 + 105] = 3
        sampled[1112:1120] = (64).to_bytes(8, "little")
        sampled[1120:1128] = (0x0123456789ABCDEF).to_bytes(8, "little")
        sampled[1128:1136] = ((1 << 64) - 1).to_bytes(8, "little")
        sampled[1136:1140] = (8).to_bytes(4, "little")
        sampled[1140:1144] = (64).to_bytes(4, "little")
        value = SUBJECT.decode_process_handoff(refreshed(sampled))
        self.assertEqual(value.writer.runtime_snapshot.mode_id, 3)
        self.assertEqual(
            (
                value.writer.runtime_snapshot.table_entries,
                value.writer.runtime_snapshot.sample_seed,
                value.writer.runtime_snapshot.sample_threshold,
                value.writer.runtime_snapshot.table_probes,
                value.writer.runtime_snapshot.table_shards,
            ),
            (64, 0x0123456789ABCDEF, (1 << 64) - 1, 8, 64),
        )

        paired = bytearray(sampled)
        paired[1265 + 336 : 1265 + 344] = (1).to_bytes(8, "little")
        paired[1265 + 368 + 5 * 8 : 1265 + 368 + 6 * 8] = (1).to_bytes(8, "little")
        value = SUBJECT.decode_process_handoff(refreshed(paired))
        self.assertEqual(value.threads[0].sample_insertion_failures, 1)
        self.assertEqual(value.threads[0].diagnostic_values[5], 1)

        mismatched = bytearray(paired)
        mismatched[1265 + 368 + 5 * 8 : 1265 + 368 + 6 * 8] = (2).to_bytes(8, "little")
        with self.assertRaisesRegex(
            SUBJECT.ProcessHandoffError,
            r"^threads\[0\]: runtime/diagnostic projection mismatch$",
        ):
            SUBJECT.decode_process_handoff(refreshed(mismatched))

        for offset, size, value in ((1140, 4, 32), (1128, 8, 0)):
            with self.subTest(offset=offset):
                mutant = bytearray(sampled)
                mutant[offset : offset + size] = value.to_bytes(size, "little")
                with self.assertRaisesRegex(SUBJECT.ProcessHandoffError, r"^sampling_controls:"):
                    SUBJECT.decode_process_handoff(refreshed(mutant))

        nonsampled = bytearray(self.literal)
        nonsampled[1112:1120] = (64).to_bytes(8, "little")
        with self.assertRaisesRegex(SUBJECT.ProcessHandoffError, r"^sampling_controls:"):
            SUBJECT.decode_process_handoff(refreshed(nonsampled))

    def test_prefix_digest_is_mandatory_and_self_authenticated(self) -> None:
        mutant = bytearray(self.literal)
        mutant[1080:1112] = bytes(32)
        with self.assertRaisesRegex(
            SUBJECT.ProcessHandoffError,
            r"^prefix_sha256: nonzero producer-authenticated digest required$",
        ):
            SUBJECT.decode_process_handoff(refreshed(mutant))

    def test_layout_and_reserved_masks_are_fail_closed(self) -> None:
        mutant = bytearray(self.literal)
        mutant[76:80] = (224).to_bytes(4, "little")
        with self.assertRaisesRegex(SUBJECT.ProcessHandoffError, r"^layout: fixed size field mismatch$"):
            SUBJECT.decode_process_handoff(refreshed(mutant))

        mutant = bytearray(self.literal)
        mutant[96:100] = (4).to_bytes(4, "little")
        with self.assertRaisesRegex(
            SUBJECT.ProcessHandoffError,
            r"^registration_diagnostic_saturated_mask: reserved bit set$",
        ):
            SUBJECT.decode_process_handoff(refreshed(mutant))


if __name__ == "__main__":
    unittest.main()
