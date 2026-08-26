#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Deterministic tests for the schema-v1 acyclic status chain."""

from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import sys
import unittest
from dataclasses import replace
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("status_chain_v1", HERE / "status_chain_v1.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("module specification unavailable")
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def terminal_flags(stage: int, writer: int, *, complete=False, partial=False) -> int:
    value = (1 << (stage + 1)) - 1
    value |= ((1 << 7) | (1 << 8)) if writer == 5 else (1 << 8)
    if complete:
        value |= (1 << 9) | (1 << 10) | (1 << 11)
    if partial:
        value |= 1 << 13
    return value


def pre_footer(*, lifecycle=5, writer=5, stage=6, reason=0, scope=1, partial=False):
    return {
        "active_generation": 17,
        "active_start_counter": 100,
        "active_start_monotonic_raw_ns": 1_000,
        "cutoff_after_counter": 121 if stage >= 1 else 0,
        "cutoff_before_counter": 120 if stage >= 1 else 0,
        "diagnostic_population_partial": partial,
        "final_counter": 140,
        "final_monotonic_raw_ns": 2_000,
        "final_realtime_unix_ns": 1_800_000_000_000_000_000,
        "finalization_stage": stage,
        "lifecycle_state": lifecycle,
        "payload_writer_state": writer,
        "process_generation": 17,
        "quiescence_complete_counter": 130 if stage >= 2 else 0,
        "reason_code": reason,
        "schema": m.SCHEMA_PRE_FOOTER,
        "scope_kind": scope,
        "terminal_flags": terminal_flags(stage, writer, complete=lifecycle == 5, partial=partial),
    }


def receipt(path: str, stream: bytes):
    return {
        "appender_close": "success",
        "exact_eof": True,
        "footer_preimage_sha256": "1" * 64,
        "opening_header_sha256": "2" * 64,
        "physical_bytes": len(stream),
        "prefix_sha256": "3" * 64,
        "schema": m.SCHEMA_POST_CLOSE,
        "stream_path": path,
        "trailer_body_sha256": "4" * 64,
        "verifier_definition_sha256": "5" * 64,
        "whole_stream_sha256": sha(stream),
    }


def verified_stream_identity(path: str, stream: bytes, pre_footer_raw: bytes):
    return m.VerifiedStreamIdentity(
        stream_path=path,
        physical_bytes=len(stream),
        whole_stream_sha256=sha(stream),
        pre_footer_status_bytes=len(pre_footer_raw),
        pre_footer_status_sha256=sha(pre_footer_raw),
        footer_preimage_sha256="1" * 64,
        opening_header_sha256="2" * 64,
        prefix_sha256="3" * 64,
        trailer_body_sha256="4" * 64,
        verifier_definition_sha256="5" * 64,
    )


def manifest(rows):
    return {
        "entries": [
            {"bytes": byte_count, "path": path, "sha256": digest}
            for path, byte_count, digest in sorted(rows)
        ],
        "schema": m.SCHEMA_MANIFEST,
    }


class CanonicalAndOutcomeTests(unittest.TestCase):
    def test_canonical_round_trip_and_rejections(self):
        value = {"entries": [{"bytes": 0, "path": "a", "sha256": "1" * 64}], "schema": m.SCHEMA_MANIFEST}
        raw = m.canonical_bytes(value)
        self.assertEqual(m.canonical_bytes(m.parse_canonical(raw)), raw)
        bad = (
            b'{"schema":"x", "entries":[]}\n',
            b'{"entries":[],"schema":"x","schema":"x"}\n',
            b'{"entries":[1.0],"schema":"x"}\n',
            b'\xef\xbb\xbf{"entries":[],"schema":"x"}\n',
            b'{"Upper":1}\n',
            b'{"_leading":1}\n',
        )
        for mutant in bad:
            with self.subTest(mutant=mutant), self.assertRaises(m.StatusChainError):
                m.parse_canonical(mutant)
        with self.assertRaises(m.StatusChainError):
            m.canonical_bytes({"entries": [], "schema": "e\u0301"})
        escaped = {"entries": [], "schema": "tab\tdelete\x7fquote\"slash\\"}
        self.assertIn(b"tab\\u0009delete\\u007fquote\\\"slash\\\\", m.canonical_bytes(escaped))

    def test_frozen_lexical_canonical_vector_from_nonlexical_insertion_order(self):
        value = {}
        value["zeta"] = "slash\\quote\"\t"
        nested = {}
        nested["omega"] = 2
        nested["beta"] = [True, None]
        value["alpha"] = nested
        self.assertEqual(
            m.canonical_bytes(value),
            b'{"alpha":{"beta":[true,null],"omega":2},"zeta":"slash\\\\quote\\"\\u0009"}\n',
        )

    def test_nesting_limit_is_string_and_escape_aware(self):
        nested: object = 'literal [{]} escaped quote " and slash \\ [}]'
        for _ in range(m.MAX_JSON_NESTING_DEPTH - 1):
            nested = [nested]
        raw = m.canonical_bytes({"a": nested})
        self.assertEqual(m.parse_canonical(raw), {"a": nested})

        over_limit_raw = (
            b'{"a":'
            + b"[" * m.MAX_JSON_NESTING_DEPTH
            + b"0"
            + b"]" * m.MAX_JSON_NESTING_DEPTH
            + b"}\n"
        )
        with self.assertRaises(m.StatusChainError) as caught:
            m.parse_canonical(over_limit_raw)
        self.assertEqual(caught.exception.code, "range")
        self.assertEqual(caught.exception.field, "raw")

    def test_recursion_error_backstop_is_typed(self):
        with mock.patch.object(m, "_canonical_value", side_effect=RecursionError("synthetic")):
            with self.assertRaises(m.StatusChainError) as caught:
                m.canonical_bytes({"a": 0})
        self.assertEqual(caught.exception.code, "depth")
        self.assertEqual(caught.exception.field, "$")

    def test_all_reachable_primary_outcome_shapes(self):
        cases = (
            (5, 6, 5, 0, 1), (7, 1, 5, 1, 1), (6, 2, 5, 2, 1),
            (6, 3, 5, 3, 1), (6, 4, 5, 4, 1), (6, 6, 6, 5, 1),
            (6, 4, 6, 6, 1), (7, 3, 5, 7, 1), (7, 0, 5, 8, 1),
            (7, 0, 5, 9, 2),
        )
        for lifecycle, stage, writer, reason, scope in cases:
            with self.subTest(reason=reason):
                value = pre_footer(lifecycle=lifecycle, stage=stage, writer=writer, reason=reason, scope=scope)
                m.validate_pre_footer(m.parse_canonical(m.canonical_bytes(value)))

    def test_relational_mutants_reject(self):
        mutants = []
        for field, replacement in (("scope_kind", 2), ("reason_code", 1), ("active_generation", 18)):
            value = pre_footer()
            value[field] = replacement
            mutants.append(value)
        value = pre_footer(); value["terminal_flags"] &= ~(1 << 10); mutants.append(value)
        value = pre_footer(lifecycle=7, stage=1, writer=5, reason=1); value["quiescence_complete_counter"] = 130; mutants.append(value)
        value = pre_footer(lifecycle=6, stage=2, writer=5, reason=2); value["cutoff_after_counter"] = 141; mutants.append(value)
        value = pre_footer(lifecycle=6, stage=2, writer=5, reason=2, partial=True); value["diagnostic_population_partial"] = False; mutants.append(value)
        for index, mutant in enumerate(mutants):
            with self.subTest(index=index), self.assertRaises(m.StatusChainError):
                m.validate_pre_footer(mutant)


class ReceiptManifestTests(unittest.TestCase):
    def setUp(self):
        self.path = "streams/memory-lifetime.bin"
        self.stream = b"synthetic independently verified stream bytes"
        self.pre = m.canonical_bytes(pre_footer())
        self.receipt = m.canonical_bytes(receipt(self.path, self.stream))
        self.verified_stream = verified_stream_identity(self.path, self.stream, self.pre)
        self.rows = [
            (m.POST_CLOSE_PATH, len(self.receipt), sha(self.receipt)),
            (m.PRE_FOOTER_PATH, len(self.pre), sha(self.pre)),
            (self.path, len(self.stream), sha(self.stream)),
        ]
        self.manifest = m.canonical_bytes(manifest(self.rows))

    def bind(self, pre=None, receipt_raw=None, manifest_raw=None, verified_stream=None):
        return m.bind_complete_candidate(
            self.pre if pre is None else pre,
            self.receipt if receipt_raw is None else receipt_raw,
            self.manifest if manifest_raw is None else manifest_raw,
            verified_stream=(
                self.verified_stream if verified_stream is None else verified_stream
            ),
        )

    def test_complete_candidate_binds_all_three_artifacts(self):
        binding = self.bind()
        self.assertEqual(binding.pre_footer_sha256, sha(self.pre))
        self.assertEqual(binding.receipt_sha256, sha(self.receipt))
        self.assertEqual(binding.stream_sha256, sha(self.stream))

    def test_negative_terminal_retained_but_never_promoted(self):
        negative = m.canonical_bytes(pre_footer(lifecycle=7, stage=1, writer=5, reason=1))
        rows = [
            (m.POST_CLOSE_PATH, len(self.receipt), sha(self.receipt)),
            (m.PRE_FOOTER_PATH, len(negative), sha(negative)),
            (self.path, len(self.stream), sha(self.stream)),
        ]
        m.validate_pre_footer(m.parse_canonical(negative))
        with self.assertRaisesRegex(m.StatusChainError, "negative terminal retained"):
            self.bind(pre=negative, manifest_raw=m.canonical_bytes(manifest(rows)))

    def test_missing_wrong_or_cyclic_manifest_binding_rejects(self):
        cases = [manifest([row for row in self.rows if row[0] != m.POST_CLOSE_PATH])]
        wrong = list(self.rows); wrong[-1] = (self.path, len(self.stream), "f" * 64); cases.append(manifest(wrong))
        cases.append(manifest(self.rows + [(m.MANIFEST_PATH, 1, "e" * 64)]))
        cases.append(manifest(self.rows + [self.rows[-1]]))
        for case in cases:
            with self.subTest(case=case), self.assertRaises(m.StatusChainError):
                self.bind(manifest_raw=m.canonical_bytes(case))

    def test_receipt_close_eof_and_stream_mismatch_reject(self):
        for field, replacement in (("appender_close", "failed"), ("exact_eof", False)):
            value = receipt(self.path, self.stream); value[field] = replacement
            with self.subTest(field=field), self.assertRaises(m.StatusChainError):
                m.validate_post_close(value)
        with self.assertRaises(m.StatusChainError):
            self.bind(
                verified_stream=replace(
                    self.verified_stream, physical_bytes=len(self.stream) + 1
                )
            )
        with self.assertRaises(m.StatusChainError):
            self.bind(
                verified_stream=replace(self.verified_stream, whole_stream_sha256="a" * 64)
            )

    def test_pre_footer_raw_byte_ceiling_rejects_before_json_load(self):
        oversized = b" " * (m.MAX_PRE_FOOTER_RAW_BYTES + 1)
        with mock.patch.object(m.json, "loads", side_effect=AssertionError("json.loads called")):
            with self.assertRaises(m.StatusChainError) as caught:
                self.bind(pre=oversized)
        self.assertEqual(caught.exception.code, "range")
        self.assertEqual(caught.exception.field, "pre_footer_raw")

    def test_flat_manifest_entry_ceiling_rejects(self):
        value = {
            "entries": [
                {"bytes": 1, "path": f"entries/{index:05d}", "sha256": "1" * 64}
                for index in range(m.MAX_MANIFEST_ENTRIES + 1)
            ],
            "schema": m.SCHEMA_MANIFEST,
        }
        with self.assertRaises(m.StatusChainError) as caught:
            m.validate_manifest(value)
        self.assertEqual(caught.exception.code, "range")
        self.assertEqual(caught.exception.field, "entries")

    def test_all_footer_and_verifier_receipt_digests_bind_trusted_identity(self):
        fields = (
            "footer_preimage_sha256",
            "opening_header_sha256",
            "prefix_sha256",
            "trailer_body_sha256",
            "verifier_definition_sha256",
        )
        for field in fields:
            value = receipt(self.path, self.stream)
            value[field] = "a" * 64
            mutated_receipt = m.canonical_bytes(value)
            rows = [
                (m.POST_CLOSE_PATH, len(mutated_receipt), sha(mutated_receipt)),
                (m.PRE_FOOTER_PATH, len(self.pre), sha(self.pre)),
                (self.path, len(self.stream), sha(self.stream)),
            ]
            with self.subTest(field=field), self.assertRaisesRegex(
                m.StatusChainError, rf"receipt\.{field}"
            ):
                self.bind(
                    receipt_raw=mutated_receipt,
                    manifest_raw=m.canonical_bytes(manifest(rows)),
                )

    def test_stream_object_kind_12_byte_count_mismatch_rejects_precisely(self):
        different = pre_footer()
        different["active_generation"] = 1000
        different["process_generation"] = 1000
        different_raw = m.canonical_bytes(different)
        self.assertNotEqual(len(different_raw), len(self.pre))
        rows = [
            (m.POST_CLOSE_PATH, len(self.receipt), sha(self.receipt)),
            (m.PRE_FOOTER_PATH, len(different_raw), sha(different_raw)),
            (self.path, len(self.stream), sha(self.stream)),
        ]
        with self.assertRaises(m.StatusChainError) as caught:
            self.bind(
                pre=different_raw,
                manifest_raw=m.canonical_bytes(manifest(rows)),
            )
        self.assertEqual(caught.exception.field, "verified_stream.pre_footer_status_bytes")

    def test_stream_object_kind_12_digest_mismatch_rejects_precisely(self):
        different = pre_footer()
        different["active_generation"] = 18
        different["process_generation"] = 18
        different_raw = m.canonical_bytes(different)
        self.assertEqual(len(different_raw), len(self.pre))
        self.assertNotEqual(sha(different_raw), sha(self.pre))
        rows = [
            (m.POST_CLOSE_PATH, len(self.receipt), sha(self.receipt)),
            (m.PRE_FOOTER_PATH, len(different_raw), sha(different_raw)),
            (self.path, len(self.stream), sha(self.stream)),
        ]
        with self.assertRaises(m.StatusChainError) as caught:
            self.bind(
                pre=different_raw,
                manifest_raw=m.canonical_bytes(manifest(rows)),
            )
        self.assertEqual(caught.exception.field, "verified_stream.pre_footer_status_sha256")

    def test_receipt_stream_path_mismatch_rejects_with_recomputed_manifest(self):
        mismatched_receipt = m.canonical_bytes(receipt("streams/other.bin", self.stream))
        rows = [
            (m.POST_CLOSE_PATH, len(mismatched_receipt), sha(mismatched_receipt)),
            (m.PRE_FOOTER_PATH, len(self.pre), sha(self.pre)),
            (self.path, len(self.stream), sha(self.stream)),
        ]
        with self.assertRaises(m.StatusChainError) as caught:
            self.bind(
                receipt_raw=mismatched_receipt,
                manifest_raw=m.canonical_bytes(manifest(rows)),
            )
        self.assertEqual(caught.exception.field, "receipt.stream_path")

    def test_stream_path_must_not_alias_any_reserved_chain_path(self):
        for path in (m.PRE_FOOTER_PATH, m.POST_CLOSE_PATH, m.MANIFEST_PATH):
            with self.subTest(path=path), self.assertRaises(m.StatusChainError) as caught:
                self.bind(
                    receipt_raw=m.canonical_bytes(receipt(path, self.stream)),
                    verified_stream=verified_stream_identity(path, self.stream, self.pre),
                )
            self.assertEqual(caught.exception.code, "cycle")


if __name__ == "__main__":
    unittest.main()
