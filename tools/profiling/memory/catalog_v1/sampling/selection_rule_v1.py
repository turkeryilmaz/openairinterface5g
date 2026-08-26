#!/usr/bin/env python3
"""Exact schema-v1 A03 allocation-instance selection candidate.

The implementation is a pure, dependency-free reference.  Arithmetic is
explicitly modulo 2**64 so an eventual C implementation can be compared
without relying on host overflow behavior.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any, Mapping, Sequence


UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
MASK64 = UINT64_MAX

DEFINITION_PATH = "definition/selection-rule-v1.json"
DEFINITION_ID = "oai_memprof_selection_rule"
VERSION = {"major": 1, "minor": 0}
DEFINITION_OBJECT_TYPE = 12
DEFINITION_OWNER = "sampling"

SEED_STATUS_GETRANDOM_EXACT = 1
SEED_STATUS_SYNTHETIC_FIXTURE = 2
SEED_STATUS_NOT_APPLICABLE = 20

_INITIAL_STATE = 0x243F6A8885A308D3
_GENERATION_DOMAIN = 0x13198A2E03707344
_THREAD_DOMAIN = 0xA4093822299F31D0
_SEQUENCE_DOMAIN = 0x082EFA98EC4E6C89
_MIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
_MIX_MULTIPLIER_2 = 0x94D049BB133111EB


class SelectionRuleError(ValueError):
    """Deterministic schema, key, seed, or threshold rejection."""


def _load_semantic() -> Any:
    path = Path(__file__).resolve().parent.parent / "semantic" / "semantic_catalog_v1.py"
    spec = importlib.util.spec_from_file_location("_oai_memprof_semantic_for_selection", path)
    if spec is None or spec.loader is None:
        raise SelectionRuleError("semantic: adjacent canonical codec unavailable")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise SelectionRuleError("semantic: adjacent canonical codec failed to load") from error
    return module


SEMANTIC = _load_semantic()


def _fail(where: str, detail: str) -> None:
    raise SelectionRuleError(f"{where}: {detail}")


def _uint(value: Any, bits: int, where: str, *, nonzero: bool = False) -> int:
    if type(value) is not int:
        _fail(where, f"u{bits} integer required")
    minimum = 1 if nonzero else 0
    if not minimum <= value <= (1 << bits) - 1:
        _fail(where, f"outside u{bits} range")
    return value


def _exact_keys(value: Any, keys: Sequence[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(where, "object required")
    if len(value) != len(keys) or set(value) != set(keys):
        _fail(where, f"exact fields {tuple(sorted(keys))!r} required")
    return value


def _rotl64(value: int, distance: int) -> int:
    return ((value << distance) | (value >> (64 - distance))) & MASK64


def _mix64(value: int) -> int:
    """SplitMix64 finalizer with explicitly wrapped products."""

    value ^= value >> 30
    value = (value * _MIX_MULTIPLIER_1) & MASK64
    value ^= value >> 27
    value = (value * _MIX_MULTIPLIER_2) & MASK64
    value ^= value >> 31
    return value & MASK64


def validate_instance_key(
    process_generation: Any, thread_index: Any, thread_sequence: Any
) -> tuple[int, int, int]:
    generation = _uint(process_generation, 64, "instance_key.process_generation", nonzero=True)
    thread = _uint(thread_index, 32, "instance_key.thread_index", nonzero=True)
    if thread == UINT32_MAX:
        _fail("instance_key.thread_index", "0xffffffff is reserved")
    sequence = _uint(thread_sequence, 64, "instance_key.thread_sequence", nonzero=True)
    return generation, thread, sequence


def encode_instance_key(
    process_generation: Any, thread_index: Any, thread_sequence: Any
) -> bytes:
    """Return exact 20-byte LE encoding ``g:u64 || t:u32 || s:u64``."""

    generation, thread, sequence = validate_instance_key(
        process_generation, thread_index, thread_sequence
    )
    return (
        generation.to_bytes(8, "little")
        + thread.to_bytes(4, "little")
        + sequence.to_bytes(8, "little")
    )


def mapping64(process_generation: Any, thread_index: Any, thread_sequence: Any) -> int:
    """Compute frozen candidate ``F(g,t,s)`` using three domain-separated rounds."""

    generation, thread, sequence = validate_instance_key(
        process_generation, thread_index, thread_sequence
    )
    duplicated_thread = ((thread << 32) | thread) & MASK64
    state = _mix64(_INITIAL_STATE ^ generation ^ _GENERATION_DOMAIN)
    state = _mix64(state ^ _rotl64(duplicated_thread ^ _THREAD_DOMAIN, 21))
    return _mix64(state ^ _rotl64(sequence ^ _SEQUENCE_DOMAIN, 42))


def parse_seed_hex(
    seed_hex: Any,
    provenance_id: Any,
    status_id: Any,
    *,
    require_publication_seed: bool,
) -> int:
    """Decode exact eight acquired bytes as conventional numeric-hex ``K``.

    All-zero bytes are a valid possible uniform draw.  Publication admission
    requires status 1; status 2 is retained only for deterministic tests.
    The first hex byte is the most-significant byte of the equivalent u64.
    Status 20 is the non-A03 not-applicable state and must carry ``None``.
    """

    if type(require_publication_seed) is not bool:
        _fail("require_publication_seed", "boolean required")
    provenance = _uint(provenance_id, 16, "seed.provenance_id", nonzero=True)
    status = _uint(status_id, 16, "seed.status_id", nonzero=True)
    if (provenance, status) not in ((1, 1), (2, 2), (20, 20)):
        _fail("seed", "frozen provenance/status pair required")
    if status == SEED_STATUS_NOT_APPLICABLE:
        if seed_hex is not None:
            _fail("seed.seed_hex", "unavailable status requires null bytes")
        _fail("seed", "selection not applicable")
    if not isinstance(seed_hex, str) or len(seed_hex) != 16:
        _fail("seed.seed_hex", "exact 16 lowercase hexadecimal characters required")
    if any(character not in "0123456789abcdef" for character in seed_hex):
        _fail("seed.seed_hex", "exact 16 lowercase hexadecimal characters required")
    if require_publication_seed and (provenance, status) != (1, 1):
        _fail("seed", "publication A03 requires exact getrandom provenance/status 1/1")
    return int(seed_hex, 16)


def validate_threshold(threshold: Any, *, active_a03: bool) -> int:
    if type(active_a03) is not bool:
        _fail("active_a03", "boolean required")
    value = _uint(threshold, 64, "sample_threshold")
    if active_a03:
        if value == 0:
            _fail("sample_threshold", "A03 requires nonzero q")
    elif value != 0:
        _fail("sample_threshold", "canonical zero required outside A03")
    return value


def selection_value(
    process_generation: Any,
    thread_index: Any,
    thread_sequence: Any,
    seed_k: Any,
) -> int:
    seed = _uint(seed_k, 64, "seed_k")
    return mapping64(process_generation, thread_index, thread_sequence) ^ seed


def selected(
    process_generation: Any,
    thread_index: Any,
    thread_sequence: Any,
    *,
    seed_k: Any,
    threshold: Any,
) -> bool:
    q = validate_threshold(threshold, active_a03=True)
    return selection_value(
        process_generation, thread_index, thread_sequence, seed_k
    ) < q


DEFINITION = {
    "catalog_id": DEFINITION_ID,
    "entries": [
        {
            "arithmetic": "unsigned_modulo_2_power_64",
            "domain_constants": {
                "generation_domain": _GENERATION_DOMAIN,
                "initial_state": _INITIAL_STATE,
                "sequence_domain": _SEQUENCE_DOMAIN,
                "thread_domain": _THREAD_DOMAIN,
            },
            "instance_key_encoding": "little_endian_u64_generation_u32_thread_u64_sequence",
            "key_interpretation": {
                "encoded_key_bytes_are_not_hashed": True,
                "mapping_inputs": "decoded_unsigned_numeric_components",
                "thread_word": "zero_extend_u32_t_shift_left_32_or_t",
            },
            "mapping_id": 1,
            "mapping_name": "domain_separated_splitmix64_finalizer_v1",
            "rotate_left_64": "((x<<n)|(x>>(64-n))) mod 2^64 for 0<n<64",
            "mapping_operations": [
                "dup_t=((u64)t<<32)|(u64)t",
                "state=mix64(initial_state xor g xor generation_domain)",
                "state=mix64(state xor rotl64(dup_t xor thread_domain,21))",
                "F=mix64(state xor rotl64(s xor sequence_domain,42))",
            ],
            "mix64_operations": [
                "x=x xor (x>>30)",
                "x=(x*multiplier_1) mod 2^64",
                "x=x xor (x>>27)",
                "x=(x*multiplier_2) mod 2^64",
                "x=x xor (x>>31)",
            ],
            "mix_multipliers": {
                "multiplier_1": _MIX_MULTIPLIER_1,
                "multiplier_2": _MIX_MULTIPLIER_2,
            },
            "rotations": {
                "sequence": 42,
                "thread": 21,
            },
            "selection_predicate": "mapping64_xor_seed_u64_less_than_threshold_u64",
            "seed_byte_order": "numeric_big_endian_hex_first_byte_most_significant",
            "seed_bytes": 8,
            "seed_provenance_status_pairs": [
                {"name": "linux_getrandom_exact_eight_bytes_acquired", "provenance_id": 1, "status_id": 1},
                {"name": "explicit_synthetic_fixture", "provenance_id": 2, "status_id": 2},
                {"name": "not_applicable", "provenance_id": 20, "status_id": 20},
            ],
            "threshold_domain": "a03_1_to_2_power_64_minus_1_else_zero",
        }
    ],
    "schema": "oai.memprof.selection-rule-definition",
    "version": dict(VERSION),
}


def definition_bytes() -> bytes:
    return SEMANTIC.canonical_bytes(DEFINITION)


def definition_sha256() -> str:
    return hashlib.sha256(definition_bytes()).hexdigest()


BUNDLE_MEMBER_PROPOSAL = {
    "name": "selection_rule_definition",
    "object_type": DEFINITION_OBJECT_TYPE,
    "owner": DEFINITION_OWNER,
    "path": DEFINITION_PATH,
}
BUNDLE_ENTRY_PROPOSAL = {
    "bytes": len(definition_bytes()),
    "object_type": DEFINITION_OBJECT_TYPE,
    "path": DEFINITION_PATH,
    "sha256": definition_sha256(),
}


def validate_definition(value: Any) -> None:
    try:
        if SEMANTIC.canonical_bytes(value) != definition_bytes():
            _fail("selection_definition", "frozen definition differs")
    except SelectionRuleError:
        raise
    except Exception as error:
        raise SelectionRuleError(str(error)) from error


__all__ = [
    "BUNDLE_ENTRY_PROPOSAL",
    "BUNDLE_MEMBER_PROPOSAL",
    "DEFINITION",
    "DEFINITION_OBJECT_TYPE",
    "DEFINITION_OWNER",
    "DEFINITION_PATH",
    "SelectionRuleError",
    "definition_bytes",
    "definition_sha256",
    "encode_instance_key",
    "mapping64",
    "parse_seed_hex",
    "selected",
    "selection_value",
    "validate_definition",
    "validate_instance_key",
    "validate_threshold",
]
