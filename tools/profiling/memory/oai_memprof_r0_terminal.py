#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Publication-grade terminal controller for the bounded OAI memprof R0 gate.

This module deliberately owns scientific protocol semantics, not operating-
system process containment or archive I/O.  Those mechanisms live in the
sibling ``oai_memprof_r0_terminal_process`` and
``oai_memprof_r0_terminal_archive`` modules.  The split is part of the frozen
R0 procedure: this file validates an immutable plan, expands the exact
OFF0/OFF1/ON command population, interprets the resulting evidence, and keeps
scientific completeness distinct from archive integrity and claim eligibility.

R0 is a passive-wrapper *test and generated-graph* gate.  It does not execute
an active allocation stream, build production softmodem ELFs, or claim absence
from production ELFs.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import os
import pathlib
import platform
import pwd
import re
import secrets
import signal
import socket
import stat
import sys
import threading
import time
import types
from datetime import datetime, timezone
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Protocol, TextIO


PLAN_SCHEMA = "oai.memprof.r0.terminal-plan/v1"
PROCEDURE_SCHEMA = "oai.memprof.r0.terminal-procedure/v1"
TERMINAL_RESULT_SCHEMA = "oai.memprof.r0.terminal-result/v1"
COMMAND_LEDGER_SCHEMA = "oai.memprof.r0.command-ledger/v1"
PUBLICATION_SUCCESS_SCHEMA = "oai.memprof.r0.publication-success/v1"
PUBLICATION_LINEARIZATION = (
    "durable_o_excl_file_fsync_parent_fsync_success_marker_commit"
)
PUBLICATION_ACTIVATION = (
    "terminal_axes_activate_only_after_durable_publication_success_marker"
)
PUBLICATION_SIGNAL_CUTOFF = (
    "one_pending_set_snapshot_immediately_before_marker_commit_"
    "conditional_on_later_durable_success"
)
MAXIMUM_PLAN_BYTES = 1024 * 1024
MAXIMUM_DIAGNOSTIC_CHARACTERS = 2048
FALLBACK_MAXIMUM_CLI_BYTES = 64 * 1024
MOUNTINFO_PATH = pathlib.Path("/proc/self/mountinfo")
MAXIMUM_MOUNTINFO_BYTES = 1024 * 1024
MOUNTINFO_READ_CHUNK_BYTES = 64 * 1024
PARSER_PROGRESS_CHUNK_BYTES = 64 * 1024
MAXIMUM_PARSER_LINE_BYTES = 1024 * 1024
ACTIVE_FILESYSTEM_TYPE = "ext4"
MAXIMUM_RUNTIME_CORE_SOURCE_BYTES = 2 * 1024 * 1024
ARCHIVE_CORE_SOURCE_SHA256 = (
    "640699f9f8336b74ba2ca15e2bd5d31e5e6955efa07c929c96936a9355a5b0c6"
)
PROCESS_CORE_SOURCE_SHA256 = (
    "ff98db21401c40f2aafa0fabe7c25f4f48f956da6262750a54bb5f93095983c6"
)
SUPPORTED_CANCELLATION_SIGNAL_PRIORITY = (
    int(signal.SIGINT),
    int(signal.SIGTERM),
    int(signal.SIGHUP),
)

ABSENCE_SCHEMA = "oai.memprof.r0.absence-result/v6"
ABSENCE_CATALOG_VERSION = "oai-memprof-r0-absence-catalog/6"
ABSENCE_CATALOG_ENTRIES = 106
ABSENCE_CATALOG_SHA256 = (
    "a74f21dd955e8995dc1faad14eab70d1b2cc0ca0fe0aaebc0c9af13bc43a8c14"
)
ABSENCE_CHECKER_RELATIVE_PATH = (
    "common/utils/memprof/tests/check_oai_memprof_r0_absence.py"
)
ABSENCE_CHECKER_SHA256 = (
    "12485e9e33c00089abdb53a75acf1415052958c55495b11e2d3fb2280ae0c606"
)
ABSENCE_FINAL_LINK_GRAMMAR = "oai.memprof.r0.final-link-text-grammar/v2"
ABSENCE_REFERENCE_TOOL = "GNU ld 2.42"
ABSENCE_EXPECTED_FINAL_TARGETS = ("nr-softmodem", "nr-uesoftmodem")
PRODUCTION_FINAL_ELF_ABSENCE = "NOT_APPLICABLE_DEFERRED_R0_TEST_ONLY"
ABSENCE_CLAIM = {
    "absence_scope": "bounded_structural_content_only",
    "exit_zero_common_origin": "not_common_origin_proof",
    "final_elf": "independent_final_elf_validation_required",
    "final_output_identity": "external_final_file_validation_required",
    "gnu_wrap_absence": "frozen_textual_grammar_only",
    "gnu_wrap_shell_expansion": "shell_expansion_constructed_spellings_outside_grammar",
}
ABSENCE_FINAL_LINK_MODEL = {
    "containment": "lexical_bare_target_basename_only",
    "driver": "/usr/bin/c++",
    "driver_claim": "exact_textual_usr_bin_cxx_only",
    "grammar": ABSENCE_FINAL_LINK_GRAMMAR,
    "scaffold": "exact_cmake_ninja_colon_guard_only",
}
ABSENCE_PROVENANCE = {
    "capture_origin": "external_terminal_ledger_required",
    "content_identity": "self_hashed_content_only",
    "relative_final_outputs": "bare_basename_external_cwd_premise_required",
}
ABSENCE_ROOT_MODEL = {
    "domain": "canonical_ascii_path_components_only",
    "grammar": "oai.memprof.r0.declared-root-grammar/v1",
}
ABSENCE_WRAP_MODEL = {
    "expansion_constructed_spellings": "shell_expansion_constructed_spellings_outside_grammar",
    "expansion_example": r"--wr${EMPTY}ap=malloc",
    "grammar": "oai.memprof.gnu-ld-wrap-text-grammar/v2",
    "reference_tool": ABSENCE_REFERENCE_TOOL,
}

R0_BUILD_TARGETS = (
    "oai_memprof_runtime",
    "oai_memprof_wrap_c",
    "test_oai_memprof_wire",
    "test_oai_memprof_r0_scripted_a00",
    "test_oai_memprof_r0_scripted_a01",
    "test_oai_memprof_r0_mutant_duplicate_real",
    "test_oai_memprof_r0_mutant_operand",
    "test_oai_memprof_r0_mutant_errno",
    "test_oai_memprof_r0_mutant_result",
    "test_oai_memprof_r0_mutant_suppress_free_null",
    "test_oai_memprof_r0_mutant_context",
    "oai_memprof_r0_actual_dso_a00",
    "oai_memprof_r0_actual_dso_a01",
    "test_oai_memprof_r0_actual_a00",
    "test_oai_memprof_r0_actual_a01",
)

R0_CTESTS = (
    "oai_memprof_r0_mutation_duplicate_real",
    "oai_memprof_r0_mutation_operand",
    "oai_memprof_r0_mutation_errno",
    "oai_memprof_r0_mutation_result",
    "oai_memprof_r0_mutation_suppress_free_null",
    "oai_memprof_r0_mutation_context",
    "oai_memprof_r0_scripted_pair",
    "oai_memprof_r0_scripted_bounds",
    "oai_memprof_r0_actual_a00",
    "oai_memprof_r0_actual_a01",
    "oai_memprof_r0_actual_differential",
    "oai_memprof_r0_elf_validator_selftest",
    "oai_memprof_r0_harness_selftest",
    "oai_memprof_r0_absence_validator_selftest",
    "oai_memprof_r0_elf",
)

WIRE_CTESTS = (
    "oai_memprof_wire_c",
    "oai_memprof_wire_cross_literal",
    "oai_memprof_wire_python",
)

R0_ARTIFACTS = (
    "common/utils/memprof/liboai_memprof_runtime.so.1.0.0",
    "common/utils/memprof/liboai_memprof_wire.a",
    "common/utils/memprof/liboai_memprof_wrap_c.a",
    "common/utils/memprof/tests/liboai_memprof_r0_actual_dso_a00.so",
    "common/utils/memprof/tests/liboai_memprof_r0_actual_dso_a01.so",
    "common/utils/memprof/tests/oai_memprof_r0_a00_dso.map",
    "common/utils/memprof/tests/oai_memprof_r0_a00_exe.map",
    "common/utils/memprof/tests/oai_memprof_r0_a01_dso.map",
    "common/utils/memprof/tests/oai_memprof_r0_a01_exe.map",
    "common/utils/memprof/tests/test_oai_memprof_r0_actual_a00",
    "common/utils/memprof/tests/test_oai_memprof_r0_actual_a01",
    "common/utils/memprof/tests/test_oai_memprof_r0_mutant_context",
    "common/utils/memprof/tests/test_oai_memprof_r0_mutant_duplicate_real",
    "common/utils/memprof/tests/test_oai_memprof_r0_mutant_errno",
    "common/utils/memprof/tests/test_oai_memprof_r0_mutant_operand",
    "common/utils/memprof/tests/test_oai_memprof_r0_mutant_result",
    "common/utils/memprof/tests/test_oai_memprof_r0_mutant_suppress_free_null",
    "common/utils/memprof/tests/test_oai_memprof_r0_scripted_a00",
    "common/utils/memprof/tests/test_oai_memprof_r0_scripted_a01",
    "common/utils/memprof/tests/test_oai_memprof_wire",
)

ELF_ARTIFACTS = frozenset(
    path
    for path in R0_ARTIFACTS
    if path.endswith(".so")
    or path.endswith(".so.1.0.0")
    or "/test_oai_memprof_" in path
)
ARCHIVE_ARTIFACTS = frozenset(path for path in R0_ARTIFACTS if path.endswith(".a"))
MAP_ARTIFACTS = frozenset(path for path in R0_ARTIFACTS if path.endswith(".map"))

R0_CONDITIONS = (
    ("OFF0", False, False),
    ("OFF1", False, True),
    ("ON", True, False),
)

OPTIONAL_CMAKE_FEATURES_DISABLED = (
    "ENABLE_CHANNEL_SIM_CUDA",
    "ENBSCOPE",
    "IMSCOPE",
    "IMSCOPE_RECORD",
    "LDPC_AAL",
    "LTTNG",
    "NRSCOPE",
    "TELNETSRV",
    "UESCOPE",
    "USE_CPU_EXECUTION_TIME",
    "VCD",
    "VCD_FIFO",
    "WEBSRV",
)

REQUIRED_TOOL_NAMES = (
    "python",
    "git",
    "cmake",
    "ninja",
    "ctest",
    "cc",
    "cxx",
    "readelf",
    "nm",
    "objdump",
    "ar",
    "ld",
)

REQUIRED_STEP_LIMIT_NAMES = (
    "controller_selftest",
    "git_clone",
    "git_checkout",
    "tool_probe",
    "configure",
    "ninja_targets",
    "ninja_commands",
    "absence_gate",
    "build",
    "ctest_inventory",
    "ctest_run",
    "elf_gate",
)

_PLAN_TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "case_label",
        "expected_commit",
        "expected_architecture",
        "expected_hostname",
        "source_repository",
        "roots",
        "dependencies",
        "tools",
        "safety",
        "execution",
    }
)

_SAFE_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_RECEIPT_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_FULL_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_V6_ROOT_PATH = re.compile(r"/(?:[A-Za-z0-9._+-]+/)*[A-Za-z0-9._+-]+\Z")
_CTEST_LINE = re.compile(r"^\s*Test\s+#(?P<number>[1-9][0-9]*):\s+(?P<name>\S+)\s*$")
_CTEST_TOTAL = re.compile(r"^\s*Total Tests:\s+(?P<count>[0-9]+)\s*$")
_CTEST_PASS = re.compile(
    r"^100% tests passed, 0 tests failed out of (?P<count>[0-9]+)$", re.MULTILINE
)
_CTEST_RESULT = re.compile(
    r"^\s*(?:[1-9][0-9]*/[1-9][0-9]*\s+)?Test\s+#[1-9][0-9]*:\s+"
    r"(?P<name>\S+)\s+\.*\s+(?P<status>Passed|Skipped|Not Run|Failed|\*\*\*Failed)"
    r"(?:\s+.*)?$"
)
_LD_242 = re.compile(
    r"GNU ld(?: \([ -~]{1,128}\))? 2\.42(?:\.[0-9]+)?(?:[ -~]{0,128})\Z"
)


class PlanError(ValueError):
    """The immutable run plan does not satisfy the frozen schema."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = _bounded_text(detail)
        super().__init__(f"{code}: {self.detail}")


class GateError(ValueError):
    """A bounded scientific postcondition was not established."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = _bounded_text(detail)
        super().__init__(f"{code}: {self.detail}")


class ControllerError(RuntimeError):
    """The controller could not safely continue or publish evidence."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        publication_phase: str | None = None,
    ):
        self.code = code
        self.detail = _bounded_text(detail)
        self.publication_phase = publication_phase
        super().__init__(f"{code}: {self.detail}")


@dataclasses.dataclass(frozen=True)
class RuntimeCoreSourceBinding:
    role: str
    path: str
    bytes: int
    sha256: str
    device: int
    inode: int
    mode: int
    mtime_ns: int
    ctime_ns: int


def _bind_runtime_core_source(
    role: str, path: pathlib.Path, expected_sha256: str
) -> RuntimeCoreSourceBinding:
    """Hash one exact sibling source with bounded, no-follow, stable I/O."""

    if (
        not isinstance(role, str)
        or not role
        or not path.is_absolute()
        or str(path).startswith("//")
        or _SHA256.fullmatch(expected_sha256) is None
    ):
        raise ControllerError("runtime_core_binding_argument", role)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ControllerError(
            "runtime_core_source_open", f"{role}: {error}"
        ) from error
    primary: BaseException | None = None
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= MAXIMUM_RUNTIME_CORE_SOURCE_BYTES
        ):
            raise ControllerError(
                "runtime_core_source_type",
                f"{role}: mode={before.st_mode} nlink={before.st_nlink} "
                f"bytes={before.st_size}",
            )
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, min(64 * 1024, before.st_size - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > MAXIMUM_RUNTIME_CORE_SOURCE_BYTES:
                raise ControllerError("runtime_core_source_bound", role)
            digest.update(chunk)
        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if total != before.st_size or any(
            getattr(before, field) != getattr(after, field)
            for field in stable_fields
        ):
            raise ControllerError("runtime_core_source_changed", role)
        observed_sha256 = digest.hexdigest()
        if observed_sha256 != expected_sha256:
            raise ControllerError(
                "runtime_core_source_hash",
                f"{role}: expected {expected_sha256}, observed {observed_sha256}",
            )
        return RuntimeCoreSourceBinding(
            role=role,
            path=str(path),
            bytes=total,
            sha256=observed_sha256,
            device=after.st_dev,
            inode=after.st_ino,
            mode=after.st_mode,
            mtime_ns=after.st_mtime_ns,
            ctime_ns=after.st_ctime_ns,
        )
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            os.close(descriptor)
        except OSError as close_error:
            if primary is not None:
                primary.add_note(
                    f"runtime core source close failed for {role}: {close_error}"
                )
            else:
                raise ControllerError(
                    "runtime_core_source_close", f"{role}: {close_error}"
                ) from close_error


@dataclasses.dataclass(frozen=True)
class StepLimit:
    wall_seconds: int
    stdout_bytes: int
    stderr_bytes: int


@dataclasses.dataclass(frozen=True)
class ThermalSensorPlan:
    label: str
    path: str
    minimum_plausible_millicelsius: int


@dataclasses.dataclass(frozen=True)
class SafetyPlan:
    thermal_ceiling_millicelsius: int
    poll_interval_milliseconds: int
    maximum_poll_gap_milliseconds: int
    mandatory_thermal_sensors: tuple[ThermalSensorPlan, ...]
    minimum_free_start_bytes: int
    minimum_free_during_bytes: int
    maximum_observed_free_space_decline_bytes: int
    maximum_archive_bytes: int
    maximum_safety_samples: int
    maximum_sensor_bytes: int


@dataclasses.dataclass(frozen=True)
class ExecutionPlan:
    build_jobs: int
    ctest_jobs: int
    clean_path: str
    cleanup_grace_seconds: int
    maximum_proc_entries: int
    maximum_observed_live_descendants: int
    maximum_descendant_identities: int
    read_chunk_bytes: int
    maximum_evidence_bytes: int
    maximum_cli_output_bytes: int
    maximum_json_evidence_bytes: int
    maximum_archive_regular_files_excluding_manifest: int
    maximum_archive_directories_excluding_root: int
    maximum_manifest_bytes: int
    maximum_state_bytes: int
    maximum_journal_entries: int
    maximum_journal_entry_bytes: int
    step_limits: Mapping[str, StepLimit]


@dataclasses.dataclass(frozen=True)
class CpmDependency:
    path: str
    sha256: str


@dataclasses.dataclass(frozen=True)
class GitDependency:
    path: str
    commit: str


@dataclasses.dataclass(frozen=True)
class DependencyPlan:
    cpm_cmake: CpmDependency
    googletest: GitDependency
    benchmark: GitDependency


@dataclasses.dataclass(frozen=True)
class RootPlan:
    acquisition_parent: str
    archive_parent: str


@dataclasses.dataclass(frozen=True)
class RunPlan:
    schema: str
    case_label: str
    expected_commit: str
    expected_architecture: str
    expected_hostname: str
    source_repository: str
    roots: RootPlan
    dependencies: DependencyPlan
    tools: Mapping[str, str]
    safety: SafetyPlan
    execution: ExecutionPlan
    canonical_bytes: bytes
    sha256: str


@dataclasses.dataclass(frozen=True)
class EvidenceIdentity:
    role: str
    path: str
    bytes: int
    lines: int
    sha256: str
    device: int
    inode: int


@dataclasses.dataclass(frozen=True)
class BoundedFileImage:
    """One callback-sampled bounded read and its stable content identity."""

    data: bytes
    device: int
    inode: int
    size: int
    mode: int
    mtime_ns: int
    ctime_ns: int
    sha256: str


@dataclasses.dataclass(frozen=True)
class ToolBinding:
    declared_path: str
    declared_device: int
    declared_inode: int
    declared_mode: int
    declared_nlink: int
    declared_size: int
    declared_mtime_ns: int
    declared_ctime_ns: int
    resolved_path: str
    resolved_device: int
    resolved_inode: int
    resolved_mode: int
    resolved_nlink: int
    resolved_size: int
    resolved_mtime_ns: int
    resolved_ctime_ns: int
    sha256: str


@dataclasses.dataclass(frozen=True)
class MountIdentity:
    mount_id: int
    parent_id: int
    device_major: int
    device_minor: int
    root: str
    mount_point: str
    filesystem_type: str
    source: str
    mount_options: tuple[str, ...]
    optional_fields: tuple[str, ...]
    super_options: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class AbsenceDecision:
    condition: str
    tool_reference_match: bool
    eligibility: str
    reason: str


@dataclasses.dataclass(frozen=True)
class ClaimDecision:
    claim: str
    eligibility: str
    reasons: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class TerminalAxes:
    scientific_case_state: str
    archive_integrity: str
    inclusion: str
    profiler_stream_state: str
    production_final_elf_absence: str
    reasons: tuple[str, ...]
    claims: tuple[ClaimDecision, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "archive_integrity": self.archive_integrity,
            "claims": [
                {
                    "claim": item.claim,
                    "eligibility": item.eligibility,
                    "reasons": list(item.reasons),
                }
                for item in self.claims
            ],
            "inclusion": self.inclusion,
            "production_final_elf_absence": self.production_final_elf_absence,
            "profiler_stream_state": self.profiler_stream_state,
            "reasons": list(self.reasons),
            "scientific_case_state": self.scientific_case_state,
        }


@dataclasses.dataclass(frozen=True)
class ConditionPaths:
    name: str
    root: pathlib.Path
    build: pathlib.Path
    dependency_root: pathlib.Path
    log_root: pathlib.Path


@dataclasses.dataclass(frozen=True)
class RunWorkspace:
    run_id: str
    scratch: pathlib.Path
    archive_stage: pathlib.Path
    archive_destination: pathlib.Path
    source: pathlib.Path
    temporary: pathlib.Path
    xdg_cache: pathlib.Path
    xdg_config: pathlib.Path
    xdg_data: pathlib.Path
    xdg_state: pathlib.Path
    conditions: Mapping[str, ConditionPaths]


class ControllerServices(Protocol):
    """High-level seam used by deterministic fake-tool controller tests.

    The production implementation is an adapter over the sibling process and
    archive modules.  No implementation of containment or archive publication
    belongs in this protocol consumer.
    """

    def create_workspace(self, plan: RunPlan, descriptor: Mapping[str, object]) -> RunWorkspace:
        ...

    def read_regular(
        self,
        workspace: RunWorkspace,
        path: pathlib.Path,
        maximum_bytes: int,
    ) -> BoundedFileImage:
        ...

    def progress(self, workspace: RunWorkspace) -> None:
        """Reach one cooperative controller-CPU safety/cancellation boundary."""

        ...

    def write_json(self, workspace: RunWorkspace, relative_path: str, value: object) -> None:
        ...

    def copy_scratch_regular(
        self,
        workspace: RunWorkspace,
        source: pathlib.Path,
        destination: pathlib.Path,
        maximum_bytes: int,
    ) -> EvidenceIdentity:
        """Copy to an absolute validated destination below active scratch."""
        ...

    def capture_regular(
        self,
        workspace: RunWorkspace,
        source: pathlib.Path,
        archive_relative_path: str,
        maximum_bytes: int,
    ) -> EvidenceIdentity:
        """Copy below archive_stage and return that actual absolute identity."""
        ...

    def validate_captured_artifact(
        self,
        workspace: RunWorkspace,
        path: pathlib.Path,
        relative_path: str,
        identity: EvidenceIdentity,
        architecture: str,
        *,
        maximum_bytes: int,
        read_chunk_bytes: int,
    ) -> None:
        ...

    def run_step(
        self,
        workspace: RunWorkspace,
        plan: RunPlan,
        step: str,
        instance: str,
        argv: Sequence[str],
        cwd: pathlib.Path,
        environment: Mapping[str, str],
    ) -> object:
        ...

    def command_stdout(self, result: object, maximum_bytes: int) -> bytes:
        ...

    def command_stderr(self, result: object, maximum_bytes: int) -> bytes:
        ...

    def command_succeeded(self, result: object) -> bool:
        ...

    def command_record(self, result: object) -> Mapping[str, object]:
        ...

    def checkpoint(self, workspace: RunWorkspace, state: Mapping[str, object]) -> None:
        ...

    def publish(
        self,
        workspace: RunWorkspace,
        axes: TerminalAxes,
        maximum_archive_bytes: int,
        receipt_validator: Callable[
            [Mapping[str, object]], Mapping[str, object]
        ],
    ) -> Mapping[str, object]:
        ...

    def recover_incomplete(self, case_path: pathlib.Path) -> Mapping[str, object]:
        ...

    def verify_archive(self, archive_path: pathlib.Path) -> Mapping[str, object]:
        ...


def _bounded_text(value: object, limit: int = MAXIMUM_DIAGNOSTIC_CHARACTERS) -> str:
    text = str(value).replace("\x00", "?")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _archive_module() -> Any:
    return importlib.import_module(
        "tools.profiling.memory.oai_memprof_r0_terminal_archive"
    )


def canonical_json_bytes(value: object) -> bytes:
    """Delegate the one authoritative serialization to the archive module."""

    return _archive_module().canonical_json_bytes(value)


def _descriptor_json_bytes(value: object) -> bytes:
    """Private bootstrap serializer for the import-time frozen descriptor.

    The descriptor contains only strings, booleans, lists, and dictionaries;
    its bytes are asserted against a literal digest below.  Runtime plan and
    archive serialization always delegates to the archive module.
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


_PROCEDURE_DESCRIPTOR: dict[str, object] = {
    "absence_gate": {
        "catalog_entries": ABSENCE_CATALOG_ENTRIES,
        "catalog_sha256": ABSENCE_CATALOG_SHA256,
        "catalog_version": ABSENCE_CATALOG_VERSION,
        "checker_relative_path": ABSENCE_CHECKER_RELATIVE_PATH,
        "checker_sha256": ABSENCE_CHECKER_SHA256,
        "expected_final_targets": list(ABSENCE_EXPECTED_FINAL_TARGETS),
        "final_link_grammar": ABSENCE_FINAL_LINK_GRAMMAR,
        "reference_tool": ABSENCE_REFERENCE_TOOL,
        "schema": ABSENCE_SCHEMA,
    },
    "artifacts": list(R0_ARTIFACTS),
    "build_targets": list(R0_BUILD_TARGETS),
    "claim_boundary": {
        "active_profiler_stream": "NOT_APPLICABLE_R0",
        "archive_content_hashes": (
            "content_identity_and_accidental_change_detection_not_authenticity"
        ),
        "archive_filesystem": (
            "trusted_single_writer_same_exact_ext4_mount_renameat2_fsync_required"
        ),
        "original_location_activation": (
            "lexical_path_plus_archived_and_current_acquisition_archive_parent_"
            "inode_mode_device_and_exact_ext4_mount_identity_required"
        ),
        "archive_portability": {
            "local_strict": ["mode", "mtime_ns"],
            "portable_content": ["relative_path", "size", "sha256"],
        },
        "publication_completion": {
            "linearization": PUBLICATION_LINEARIZATION,
            "marker_retention": (
                "acquisition_scratch_marker_required_for_complete_recovery"
            ),
            "precommit_cadence_deadline": (
                "immediately_after_receipt_validation_monotonic_now_must_not_"
                "exceed_last_scientific_safety_sample_plus_plan_maximum_poll_"
                "gap_before_callback_free_marker_commit"
            ),
            "post_durability_operational_boundary": (
                "restore_exact_caller_signal_mask_without_scientific_mutation"
            ),
            "post_durability_scientific_operations": "none",
            "postcompletion_operational_receipt": (
                "receipt_only_not_success_marker_and_never_downgrades_durable_axes"
            ),
            "postcompletion_operational_cli_status": 2,
            "schema": PUBLICATION_SUCCESS_SCHEMA,
            "supported_signal_order_cutoff": PUBLICATION_SIGNAL_CUTOFF,
        },
        "atlas_gvfs_smb": "verified_backup_only_not_active_or_publication_root",
        "controller_unit_suites": (
            "exact_commit_preterminal_prerequisite_not_nested_under_runner"
        ),
        "runtime_core_sources": {
            "archive_sha256": ARCHIVE_CORE_SOURCE_SHA256,
            "binding_phase": "before_production_adapter_use",
            "content_claim": (
                "bounded_stable_sha256_identity_not_authenticity_or_authorship"
            ),
            "maximum_source_bytes": MAXIMUM_RUNTIME_CORE_SOURCE_BYTES,
            "process_sha256": PROCESS_CORE_SOURCE_SHA256,
        },
        "process_containment": {
            "admitted_tools": "trusted_exact_commands_not_hostile_fork_load",
            "hard_failure": (
                "sigkill_controller_crash_power_loss_may_require_operator_cleanup"
            ),
            "model": "linux_subreaper_ppid_pidfd_v1",
            "population": "bounded_discrete_observation_not_kernel_pids_limit",
            "signals": {
                "caller_mask": (
                    "exact_pre_post_observation_supported_signals_must_be_unblocked"
                ),
                "deferred_evidence": (
                    "sorted_unique_subset_of_sighup_sigint_sigterm"
                ),
                "restoration": "exact_process_result_true_and_caller_mask_unchanged",
                "supported": ["SIGHUP", "SIGINT", "SIGTERM"],
                "terminal_handler_binding": (
                    "exact_default_int_handler_sigint_and_exact_catchable_"
                    "systemexit_handlers_sigterm_sighup"
                ),
            },
        },
        "production_final_elf_absence": PRODUCTION_FINAL_ELF_ABSENCE,
        "production_softmodem_builds": False,
    },
    "cmake": {
        "build_type": "RelWithDebInfo",
        "disabled_optional_features": list(OPTIONAL_CMAKE_FEATURES_DISABLED),
        "fetchcontent_fully_disconnected": True,
        "fetchcontent_updates_disconnected": True,
        "generator": "Ninja",
    },
    "conditions": [
        {"enable_physim_tests": physim, "enable_tests": tests, "name": name}
        for name, tests, physim in R0_CONDITIONS
    ],
    "ctests": {"r0": list(R0_CTESTS), "wire": list(WIRE_CTESTS)},
    "execution": {
        "archive_bounds_are_plan_supplied": True,
        "build_jobs": 4,
        "cli": {
            "direct_script_invocation": "rejected_before_runtime_import",
            "invocation": (
                "python_-B_-m_tools.profiling.memory."
                "oai_memprof_r0_terminal"
            ),
            "repository_root_cwd_required": True,
        },
        "ctest_jobs": 1,
        "evidence_parser": {
            "maximum_line_bytes": MAXIMUM_PARSER_LINE_BYTES,
            "progress_chunk_bytes": PARSER_PROGRESS_CHUNK_BYTES,
            "utf8": "strict_lf_delimited_crlf_accepted_bare_cr_not_separator",
        },
        "recovery_reserve": {
            "bytes": "maximum_json_evidence_bytes",
            "live_population_check": (
                "before_mutation_exact_marker_bytes_plus_one_file_and_"
                "maximum_manifest_bytes"
            ),
            "minimum_pre_recovery_regular_files": 5,
            "regular_files": 1,
        },
        "recovery_timeline": (
            "one_shared_plan_bound_global_timeline_through_scan_marker_"
            "manifest_publish_and_postpublication_verify"
        ),
        "initial_failure_recovery_capacity": (
            "plan_plus_procedure_plus_preflight_json_max_plus_state_max_plus_"
            "journal_entry_max_plus_recovery_json_max_plus_manifest_max"
        ),
        "minimum_journal_entries": 11,
        "required_step_limits": list(REQUIRED_STEP_LIMIT_NAMES),
    },
    "safety": {
        "all_limits_are_plan_supplied": True,
        "archive_progress": (
            "same_thread_due_only_callback_plus_forced_operation_boundaries"
        ),
        "archive_payload_vs_global_decline": (
            "maximum_archive_bytes_is_payload_cap_and_must_not_exceed_"
            "same_mount_maximum_observed_free_space_decline_bytes"
        ),
        "global_decline_headroom": (
            "plan_must_accommodate_scratch_build_capture_manifest_marker_"
            "journal_and_state_allocation_not_proven_by_payload_inequality"
        ),
        "global_same_mount_baseline": (
            "maximum_of_acquisition_and_archive_preflight_free_byte_observations"
        ),
        "global_timeline": (
            "commands_controller_archive_io_and_bounded_cpu_parsers_share_one_"
            "monotonic_sample_and_maximum_gap_ledger"
        ),
        "publication_minimum_sample_operations": (
            "one_global_timeline_initialization_plus_three_named_publication_"
            "safety_samples_independent_of_monotonic_timestamp_equality"
        ),
        "verification_return_boundary": (
            "final_forced_shared_timeline_sample_after_marker_parsing_before_"
            "terminal_axes_activation_return"
        ),
        "syscall_stall_boundary": (
            "python_cannot_sample_inside_a_blocked_kernel_or_filesystem_syscall"
        ),
        "thermal_ceiling_millicelsius": 90000,
    },
    "schema": PROCEDURE_SCHEMA,
}

PROCEDURE_DESCRIPTOR_SHA256 = (
    "fb4711a0668fc88fb6ea66f6070f20cc97ee075260437adaf9803480632aa625"
)


def procedure_descriptor() -> dict[str, object]:
    """Return an isolated JSON-valued copy of the frozen descriptor."""

    return json.loads(_descriptor_json_bytes(_PROCEDURE_DESCRIPTOR).decode("utf-8"))


def verify_procedure_descriptor() -> None:
    observed = hashlib.sha256(_descriptor_json_bytes(_PROCEDURE_DESCRIPTOR)).hexdigest()
    if observed != PROCEDURE_DESCRIPTOR_SHA256:
        raise ControllerError(
            "procedure_descriptor_hash_mismatch",
            f"expected {PROCEDURE_DESCRIPTOR_SHA256}, observed {observed}",
        )


def _duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PlanError("duplicate_key", f"duplicate JSON member {key!r}")
        result[key] = value
    return result


def _reject_float(value: str) -> object:
    raise PlanError("non_integer_number", f"non-integer JSON number {value!r}")


def _reject_constant(value: str) -> object:
    raise PlanError("invalid_number", f"invalid JSON number {value!r}")


def _exact_keys(value: object, expected: frozenset[str], location: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise PlanError("wrong_type", f"{location} must be an object")
    actual = frozenset(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise PlanError("missing_key", f"{location} missing {missing}")
    if unknown:
        raise PlanError("unknown_key", f"{location} has unknown {unknown}")
    return value


def _text(value: object, location: str, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise PlanError("invalid_string", f"{location} must be a nonempty string <= {maximum}")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise PlanError("invalid_string", f"{location} contains a control character")
    return value


def _integer(value: object, location: str, minimum: int = 1, maximum: int = 2**63 - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanError("wrong_type", f"{location} must be an integer")
    if not minimum <= value <= maximum:
        raise PlanError("out_of_range", f"{location} must be in {minimum}..{maximum}")
    return value


def _absolute_path(value: object, location: str, *, permit_root: bool = False) -> str:
    text = _text(value, location)
    path = pathlib.PurePosixPath(text)
    if (
        text.startswith("//")
        or not path.is_absolute()
        or str(path) != text
        or ".." in path.parts
    ):
        raise PlanError("noncanonical_path", f"{location} must be a canonical absolute POSIX path")
    if not permit_root and text == "/":
        raise PlanError("unsafe_path", f"{location} cannot be filesystem root")
    return text


def _active_local_root(value: object, location: str) -> str:
    """Validate an active root and reject the known institutional backup path."""

    text = _absolute_path(value, location)
    components = tuple(component.lower() for component in pathlib.PurePosixPath(text).parts)
    if "gvfs" in components or any(
        component.startswith("smb-share:") for component in components
    ):
        raise PlanError(
            "backup_mount_active_root",
            f"{location} is in a GVFS/SMB backup namespace, not the active local-filesystem domain",
        )
    return text


_MANIFEST_RECEIPT_KEYS = frozenset(
    {
        "directory_count_excluding_root",
        "identity_claim",
        "local_strict_identity_fields",
        "manifest_bytes",
        "manifest_path",
        "manifest_sha256",
        "observed_archive_entries_including_manifest",
        "observed_directories",
        "observed_regular_files_including_manifest",
        "portable_content_identity_fields",
        "regular_file_count_excluding_manifest",
        "total_regular_file_bytes_excluding_manifest",
        "total_regular_file_bytes_including_manifest",
    }
)


def _canonical_receipt_path(value: object, role: str) -> pathlib.PurePosixPath:
    if not isinstance(value, str) or not value or value.startswith("//"):
        raise ControllerError("receipt_path", f"{role}: invalid absolute path")
    path = pathlib.PurePosixPath(value)
    if not path.is_absolute() or str(path) != value or ".." in path.parts:
        raise ControllerError("receipt_path", f"{role}: noncanonical absolute path")
    return path


def _validate_receipt_run_id(value: object, role: str) -> str:
    if (
        type(value) is not str
        or _RECEIPT_RUN_ID.fullmatch(value) is None
        or value in {".", ".."}
    ):
        raise ControllerError("receipt_run_id", role)
    return value


def _validate_manifest_receipt(
    value: object,
    *,
    expected_manifest_path: pathlib.PurePosixPath,
    maximum_regular_files_excluding_manifest: int | None = None,
    maximum_directories_excluding_root: int | None = None,
    maximum_manifest_bytes: int | None = None,
    maximum_total_bytes_including_manifest: int | None = None,
) -> Mapping[str, object]:
    """Validate the exact portable/local manifest summary receipt schema."""

    if not isinstance(value, Mapping) or set(value) != _MANIFEST_RECEIPT_KEYS:
        raise ControllerError("manifest_receipt_shape", "manifest receipt members differ")
    numeric_fields = (
        "directory_count_excluding_root",
        "manifest_bytes",
        "observed_archive_entries_including_manifest",
        "observed_directories",
        "observed_regular_files_including_manifest",
        "regular_file_count_excluding_manifest",
        "total_regular_file_bytes_excluding_manifest",
        "total_regular_file_bytes_including_manifest",
    )
    if any(type(value[field]) is not int or value[field] < 0 for field in numeric_fields):
        raise ControllerError(
            "manifest_receipt_numeric", "manifest counts and sizes must be exact nonnegative integers"
        )
    observed_path = _canonical_receipt_path(value["manifest_path"], "manifest")
    if (
        observed_path != expected_manifest_path
        or not isinstance(value["manifest_sha256"], str)
        or _SHA256.fullmatch(value["manifest_sha256"]) is None
        or value["total_regular_file_bytes_including_manifest"]
        != value["total_regular_file_bytes_excluding_manifest"]
        + value["manifest_bytes"]
        or value["directory_count_excluding_root"]
        != value["observed_directories"]
        or value["regular_file_count_excluding_manifest"] + 1
        != value["observed_regular_files_including_manifest"]
        or value["observed_archive_entries_including_manifest"]
        != value["observed_directories"]
        + value["observed_regular_files_including_manifest"]
        or value["identity_claim"]
        != "content_hashes_are_not_authenticity_or_authorship_proof"
        or value["local_strict_identity_fields"] != ["mode", "mtime_ns"]
        or value["portable_content_identity_fields"]
        != ["relative_path", "size", "sha256"]
    ):
        raise ControllerError("manifest_receipt_binding", "manifest summary relation differs")
    bounded_fields = (
        (
            "regular_file_count_excluding_manifest",
            maximum_regular_files_excluding_manifest,
        ),
        ("directory_count_excluding_root", maximum_directories_excluding_root),
        ("manifest_bytes", maximum_manifest_bytes),
        (
            "total_regular_file_bytes_including_manifest",
            maximum_total_bytes_including_manifest,
        ),
    )
    for field, maximum in bounded_fields:
        if maximum is not None and (
            type(maximum) is not int or maximum < 0 or value[field] > maximum
        ):
            raise ControllerError(
                "manifest_receipt_bound", f"{field} exceeds its immutable bound"
            )
    return value


def _validate_safety_sample_receipt(
    value: object, role: str
) -> Mapping[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"free_bytes", "monotonic_ns", "thermal_millicelsius"}
        or type(value["free_bytes"]) is not int
        or value["free_bytes"] < 0
        or type(value["monotonic_ns"]) is not int
        or value["monotonic_ns"] < 0
        or not isinstance(value["thermal_millicelsius"], list)
        or not value["thermal_millicelsius"]
    ):
        raise ControllerError("safety_receipt_shape", role)
    paths: list[str] = []
    for item in value["thermal_millicelsius"]:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"path", "value"}
            or type(item["value"]) is not int
            or not 0 <= item["value"] < 90_000
        ):
            raise ControllerError("safety_receipt_sensor", role)
        path = _canonical_receipt_path(item["path"], f"{role} thermal sensor")
        paths.append(str(path))
    if len(paths) != len(set(paths)):
        raise ControllerError("safety_receipt_sensor", f"{role}: duplicate sensor path")
    return value


def _validate_plan_safety_sample_receipt(
    value: object,
    role: str,
    plan: RunPlan,
    *,
    minimum_free_bytes: int,
) -> Mapping[str, object]:
    """Bind a safety sample to the exact ordered sensor population and plan."""

    if type(minimum_free_bytes) is not int or minimum_free_bytes < 0:
        raise ControllerError("safety_receipt_floor", role)
    sample = _validate_safety_sample_receipt(value, role)
    thermal = sample["thermal_millicelsius"]
    expected_sensors = plan.safety.mandatory_thermal_sensors
    if (
        sample["free_bytes"] < minimum_free_bytes
        or len(thermal) != len(expected_sensors)
        or tuple(item["path"] for item in thermal)
        != tuple(sensor.path for sensor in expected_sensors)
        or any(
            not (
                sensor.minimum_plausible_millicelsius
                <= thermal[index]["value"]
                < plan.safety.thermal_ceiling_millicelsius
            )
            for index, sensor in enumerate(expected_sensors)
        )
    ):
        raise ControllerError("safety_receipt_plan_binding", role)
    return sample


def _validate_global_safety_timeline_receipt(
    value: object, role: str, plan: RunPlan
) -> Mapping[str, object]:
    return _validate_global_safety_timeline_limits(
        value,
        role,
        maximum_safety_samples=plan.safety.maximum_safety_samples,
        maximum_poll_gap_milliseconds=(
            plan.safety.maximum_poll_gap_milliseconds
        ),
    )


def _validate_global_safety_timeline_limits(
    value: object,
    role: str,
    *,
    maximum_safety_samples: int,
    maximum_poll_gap_milliseconds: int,
) -> Mapping[str, object]:
    if (
        type(maximum_safety_samples) is not int
        or maximum_safety_samples < 1
        or type(maximum_poll_gap_milliseconds) is not int
        or maximum_poll_gap_milliseconds < 1
    ):
        raise ControllerError("global_safety_timeline_limit", role)
    timeline = _validate_global_safety_timeline_structure(value, role)
    if timeline["sample_count"] > maximum_safety_samples:
        raise ControllerError("global_safety_timeline_shape", role)
    gap = timeline["maximum_observed_gap_ns"]
    if (
        timeline["sample_count"] > 1
        and gap > maximum_poll_gap_milliseconds * 1_000_000
    ):
        raise ControllerError("global_safety_timeline_gap", role)
    return timeline


def _validate_global_safety_timeline_structure(
    value: object,
    role: str,
) -> Mapping[str, object]:
    """Validate exact timeline evidence without imposing a plan cadence cap."""

    if (
        type(value) is not dict
        or set(value)
        != {
            "first_monotonic_ns",
            "last_monotonic_ns",
            "maximum_observed_gap_ns",
            "sample_count",
            "schema",
        }
        or value["schema"] != "oai.memprof.r0.global-safety-timeline/v1"
        or type(value["sample_count"]) is not int
        or value["sample_count"] < 1
        or type(value["first_monotonic_ns"]) is not int
        or type(value["last_monotonic_ns"]) is not int
        or not 0 <= value["first_monotonic_ns"] <= value["last_monotonic_ns"]
    ):
        raise ControllerError("global_safety_timeline_shape", role)
    gap = value["maximum_observed_gap_ns"]
    sample_count = value["sample_count"]
    span_ns = value["last_monotonic_ns"] - value["first_monotonic_ns"]
    minimum_possible_maximum_gap_ns = (
        None
        if sample_count == 1
        else (span_ns + sample_count - 2) // (sample_count - 1)
    )
    if (
        (
            sample_count == 1
            and (
                gap is not None
                or value["first_monotonic_ns"] != value["last_monotonic_ns"]
            )
        )
        or (
            sample_count > 1
            and (
                type(gap) is not int
                or gap < 0
                or gap < minimum_possible_maximum_gap_ns
                or gap > span_ns
            )
        )
    ):
        raise ControllerError("global_safety_timeline_gap", role)
    return value


def _validate_postcompletion_operational_problem(
    publication: object,
) -> bool:
    if type(publication) is not dict:
        raise ControllerError(
            "publication_receipt_type", "publication must be an exact object"
        )
    verified = publication.get("verified")
    if type(verified) is not bool:
        raise ControllerError(
            "publication_receipt_boolean", "verified must be an exact boolean"
        )
    if verified is False:
        if "postcompletion_operational_problem" in publication:
            raise ControllerError(
                "postcompletion_operational_problem",
                "unverified publication cannot report a postcompletion problem",
            )
        return False
    if "postcompletion_operational_problem" not in publication:
        raise ControllerError(
            "postcompletion_operational_problem",
            "verified publication omitted the exact operational-status member",
        )
    problem = publication["postcompletion_operational_problem"]
    if problem is None:
        return False
    if (
        type(problem) is not dict
        or set(problem) != {"code", "message", "phase", "scientific_effect"}
        or type(problem["code"]) is not str
        or not problem["code"]
        or len(problem["code"]) > 128
        or type(problem["message"]) is not str
        or not problem["message"]
        or len(problem["message"]) > MAXIMUM_DIAGNOSTIC_CHARACTERS
        or problem["phase"] != "postcompletion_signal_mask_restoration"
        or problem["scientific_effect"]
        != "none_durable_success_marker_already_committed"
    ):
        raise ControllerError(
            "postcompletion_operational_problem",
            "postcompletion operational status differs",
        )
    return True


_MOUNTINFO_ESCAPES = {
    "011": "\t",
    "012": "\n",
    "040": " ",
    "134": "\\",
}


def _decode_mountinfo_token(token: str, role: str) -> str:
    """Decode only the four escapes documented for Linux mountinfo fields."""

    if not token:
        raise ControllerError("mountinfo_empty_token", role)
    decoded: list[str] = []
    index = 0
    while index < len(token):
        character = token[index]
        codepoint = ord(character)
        if character == "\\":
            escape = token[index + 1 : index + 4]
            if len(escape) != 3 or escape not in _MOUNTINFO_ESCAPES:
                raise ControllerError(
                    "mountinfo_escape", f"{role}: {token!r}"
                )
            decoded.append(_MOUNTINFO_ESCAPES[escape])
            index += 4
            continue
        if codepoint < 0x21 or codepoint > 0x7E:
            raise ControllerError(
                "mountinfo_character", f"{role}: U+{codepoint:04X}"
            )
        decoded.append(character)
        index += 1
    return "".join(decoded)


def _mountinfo_decimal(token: str, role: str, *, permit_zero: bool) -> int:
    if not re.fullmatch(r"0|[1-9][0-9]*", token):
        raise ControllerError("mountinfo_integer", f"{role}: {token!r}")
    value = int(token)
    if not permit_zero and value == 0:
        raise ControllerError("mountinfo_integer", f"{role}: zero")
    return value


def _mountinfo_options(token: str, role: str) -> tuple[str, ...]:
    decoded = _decode_mountinfo_token(token, role)
    values = tuple(decoded.split(","))
    if not values or any(not value for value in values):
        raise ControllerError("mountinfo_options", role)
    return values


def _parse_mountinfo_snapshot(
    data: bytes, *, progress_callback: Callable[[], None] | None = None
) -> tuple[MountIdentity, ...]:
    """Parse one complete, bounded Linux ``/proc/self/mountinfo`` snapshot."""

    if (
        type(data) is not bytes
        or not data
        or len(data) > MAXIMUM_MOUNTINFO_BYTES
        or not data.endswith(b"\n")
    ):
        raise ControllerError("mountinfo_snapshot_shape", "invalid size or terminator")
    try:
        text = data.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise ControllerError("mountinfo_encoding", error) from error
    records: list[MountIdentity] = []
    mount_ids: set[int] = set()
    for line_number, line in enumerate(text[:-1].split("\n"), 1):
        if progress_callback is not None:
            progress_callback()
        if not line or line.startswith(" ") or line.endswith(" ") or "  " in line:
            raise ControllerError("mountinfo_line_shape", line_number)
        fields = line.split(" ")
        separators = [index for index, field in enumerate(fields) if field == "-"]
        if len(separators) != 1:
            raise ControllerError("mountinfo_separator", line_number)
        separator = separators[0]
        if separator < 6 or len(fields) != separator + 4:
            raise ControllerError("mountinfo_field_population", line_number)
        mount_id = _mountinfo_decimal(
            fields[0], f"line {line_number} mount_id", permit_zero=False
        )
        if mount_id in mount_ids:
            raise ControllerError("mountinfo_duplicate_id", mount_id)
        mount_ids.add(mount_id)
        parent_id = _mountinfo_decimal(
            fields[1], f"line {line_number} parent_id", permit_zero=True
        )
        device_match = re.fullmatch(r"(0|[1-9][0-9]*):(0|[1-9][0-9]*)", fields[2])
        if device_match is None:
            raise ControllerError("mountinfo_device", line_number)
        root = _decode_mountinfo_token(fields[3], f"line {line_number} root")
        mount_point = _decode_mountinfo_token(
            fields[4], f"line {line_number} mount_point"
        )
        if not mount_point.startswith("/"):
            raise ControllerError("mountinfo_path", line_number)
        records.append(
            MountIdentity(
                mount_id=mount_id,
                parent_id=parent_id,
                device_major=int(device_match.group(1)),
                device_minor=int(device_match.group(2)),
                root=root,
                mount_point=mount_point,
                filesystem_type=_decode_mountinfo_token(
                    fields[separator + 1], f"line {line_number} filesystem_type"
                ),
                source=_decode_mountinfo_token(
                    fields[separator + 2], f"line {line_number} source"
                ),
                mount_options=_mountinfo_options(
                    fields[5], f"line {line_number} mount_options"
                ),
                optional_fields=tuple(
                    _decode_mountinfo_token(
                        field, f"line {line_number} optional_field"
                    )
                    for field in fields[6:separator]
                ),
                super_options=_mountinfo_options(
                    fields[separator + 3], f"line {line_number} super_options"
                ),
            )
        )
    if not records:
        raise ControllerError("mountinfo_empty", MOUNTINFO_PATH)
    return tuple(records)


def _read_mountinfo_snapshot(
    *, progress_callback: Callable[[], None] | None = None
) -> tuple[MountIdentity, ...]:
    """Open mountinfo once and return a strict bounded snapshot."""

    try:
        if progress_callback is not None:
            progress_callback()
        descriptor = os.open(
            MOUNTINFO_PATH,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        if progress_callback is not None:
            progress_callback()
    except OSError as error:
        raise ControllerError("mountinfo_open", error) from error
    primary: BaseException | None = None
    try:
        if progress_callback is not None:
            progress_callback()
        information = os.fstat(descriptor)
        if progress_callback is not None:
            progress_callback()
        if not stat.S_ISREG(information.st_mode):
            raise ControllerError("mountinfo_type", MOUNTINFO_PATH)
        chunks: list[bytes] = []
        total = 0
        while True:
            if progress_callback is not None:
                progress_callback()
            chunk = os.read(descriptor, MOUNTINFO_READ_CHUNK_BYTES)
            if progress_callback is not None:
                progress_callback()
            if not chunk:
                break
            total += len(chunk)
            if total > MAXIMUM_MOUNTINFO_BYTES:
                raise ControllerError("mountinfo_size", total)
            chunks.append(chunk)
        if progress_callback is not None:
            progress_callback()
        result = _parse_mountinfo_snapshot(
            b"".join(chunks), progress_callback=progress_callback
        )
        if progress_callback is not None:
            progress_callback()
        return result
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            os.close(descriptor)
        except OSError as close_error:
            if primary is not None:
                primary.add_note(f"mountinfo close failed: {close_error}")
            else:
                raise ControllerError("mountinfo_close", close_error) from close_error


def _select_mount_identity(
    path: pathlib.Path,
    information: os.stat_result,
    records: Sequence[MountIdentity],
) -> MountIdentity:
    path_text = str(path)
    matches = [
        record
        for record in records
        if record.mount_point == "/"
        or path_text == record.mount_point
        or path_text.startswith(record.mount_point.rstrip("/") + "/")
    ]
    if not matches:
        raise ControllerError("mountinfo_path_unmatched", path)
    longest = max(len(record.mount_point) for record in matches)
    winners = [record for record in matches if len(record.mount_point) == longest]
    if len(winners) != 1:
        raise ControllerError("mountinfo_path_ambiguous", path)
    selected = winners[0]
    observed_device = (os.major(information.st_dev), os.minor(information.st_dev))
    declared_device = (selected.device_major, selected.device_minor)
    if observed_device != declared_device:
        raise ControllerError(
            "mountinfo_device_mismatch",
            f"{path}: stat {observed_device}, mountinfo {declared_device}",
        )
    if selected.filesystem_type != ACTIVE_FILESYSTEM_TYPE:
        raise ControllerError(
            "active_filesystem_type",
            f"{path}: expected {ACTIVE_FILESYSTEM_TYPE}, observed {selected.filesystem_type}",
        )
    if not selected.root.startswith("/"):
        raise ControllerError("active_mount_root", f"{path}: {selected.root!r}")
    return selected


def _mount_identity_record(identity: MountIdentity) -> dict[str, object]:
    return {
        "device_major": identity.device_major,
        "device_minor": identity.device_minor,
        "filesystem_type": identity.filesystem_type,
        "mount_id": identity.mount_id,
        "mount_options": list(identity.mount_options),
        "mount_point": identity.mount_point,
        "optional_fields": list(identity.optional_fields),
        "parent_id": identity.parent_id,
        "root": identity.root,
        "source": identity.source,
        "super_options": list(identity.super_options),
    }


def _mount_identity_from_record(value: object, role: str) -> MountIdentity:
    expected_keys = {
        "device_major",
        "device_minor",
        "filesystem_type",
        "mount_id",
        "mount_options",
        "mount_point",
        "optional_fields",
        "parent_id",
        "root",
        "source",
        "super_options",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise ControllerError("mount_identity_record_shape", role)
    for field in ("mount_id", "parent_id"):
        if type(value[field]) is not int or value[field] <= 0:
            raise ControllerError("mount_identity_record_integer", f"{role}:{field}")
    for field in ("device_major", "device_minor"):
        if type(value[field]) is not int or value[field] < 0:
            raise ControllerError("mount_identity_record_integer", f"{role}:{field}")
    for field in ("filesystem_type", "mount_point", "root", "source"):
        if type(value[field]) is not str or not value[field]:
            raise ControllerError("mount_identity_record_string", f"{role}:{field}")
    for field in ("mount_options", "optional_fields", "super_options"):
        if (
            type(value[field]) is not list
            or any(type(item) is not str or not item for item in value[field])
        ):
            raise ControllerError("mount_identity_record_list", f"{role}:{field}")
    if (
        value["filesystem_type"] != ACTIVE_FILESYSTEM_TYPE
        or not value["mount_point"].startswith("/")
        or not value["root"].startswith("/")
    ):
        raise ControllerError("mount_identity_record_domain", role)
    return MountIdentity(
        mount_id=value["mount_id"],
        parent_id=value["parent_id"],
        device_major=value["device_major"],
        device_minor=value["device_minor"],
        root=value["root"],
        mount_point=value["mount_point"],
        filesystem_type=value["filesystem_type"],
        source=value["source"],
        mount_options=tuple(value["mount_options"]),
        optional_fields=tuple(value["optional_fields"]),
        super_options=tuple(value["super_options"]),
    )


def _round_allocation_bytes(value: int, allocation_unit_bytes: int) -> int:
    if (
        type(value) is not int
        or value < 0
        or type(allocation_unit_bytes) is not int
        or allocation_unit_bytes <= 0
    ):
        raise ControllerError(
            "allocation_rounding_argument", f"{value}/{allocation_unit_bytes}"
        )
    if value == 0:
        return 0
    rounded = ((value + allocation_unit_bytes - 1) // allocation_unit_bytes) * (
        allocation_unit_bytes
    )
    if rounded > 2**63 - 1:
        raise ControllerError("allocation_rounding_overflow", rounded)
    return rounded


def _prospective_free_threshold(
    *,
    absolute_floor_bytes: int,
    baseline_free_bytes: int,
    maximum_decline_bytes: int,
    prospective_write_bytes: int,
    allocation_unit_bytes: int,
) -> int:
    return _prospective_free_threshold_for_files(
        absolute_floor_bytes=absolute_floor_bytes,
        baseline_free_bytes=baseline_free_bytes,
        maximum_decline_bytes=maximum_decline_bytes,
        prospective_file_bytes=(prospective_write_bytes,),
        allocation_unit_bytes=allocation_unit_bytes,
    )


def _prospective_free_threshold_for_files(
    *,
    absolute_floor_bytes: int,
    baseline_free_bytes: int,
    maximum_decline_bytes: int,
    prospective_file_bytes: Sequence[int],
    allocation_unit_bytes: int,
) -> int:
    values = (
        absolute_floor_bytes,
        baseline_free_bytes,
        maximum_decline_bytes,
        allocation_unit_bytes,
    )
    if (
        any(type(value) is not int or value < 0 for value in values)
        or not isinstance(prospective_file_bytes, Sequence)
        or any(
            type(value) is not int or value < 0
            for value in prospective_file_bytes
        )
    ):
        raise ControllerError(
            "free_threshold_argument", (values, prospective_file_bytes)
        )
    relative_floor = max(0, baseline_free_bytes - maximum_decline_bytes)
    controlled_floor = max(absolute_floor_bytes, relative_floor)
    prospective_allocation = sum(
        _round_allocation_bytes(value, allocation_unit_bytes)
        for value in prospective_file_bytes
    )
    threshold = controlled_floor + prospective_allocation
    if threshold > 2**63 - 1:
        raise ControllerError("free_threshold_overflow", threshold)
    return threshold


def _sha1(value: object, location: str) -> str:
    text = _text(value, location, 40)
    if not _FULL_SHA1.fullmatch(text):
        raise PlanError("invalid_commit", f"{location} must be 40 lowercase hexadecimal characters")
    return text


def _sha256(value: object, location: str) -> str:
    text = _text(value, location, 64)
    if not _SHA256.fullmatch(text):
        raise PlanError("invalid_sha256", f"{location} must be 64 lowercase hexadecimal characters")
    return text


def _parse_dependency(value: object, location: str) -> GitDependency:
    item = _exact_keys(value, frozenset({"path", "commit"}), location)
    return GitDependency(
        path=_absolute_path(item["path"], f"{location}.path"),
        commit=_sha1(item["commit"], f"{location}.commit"),
    )


def parse_plan(data: bytes | str) -> RunPlan:
    """Parse a byte-for-byte canonical, duplicate-key-free immutable plan."""

    if isinstance(data, str):
        try:
            raw = data.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise PlanError("invalid_utf8", error) from error
    elif isinstance(data, bytes):
        raw = data
    else:
        raise PlanError("wrong_type", "plan input must be bytes or str")
    if not raw or len(raw) > MAXIMUM_PLAN_BYTES:
        raise PlanError("plan_size", f"plan must contain 1..{MAXIMUM_PLAN_BYTES} bytes")
    try:
        decoded = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise PlanError("invalid_utf8", error) from error
    try:
        value = json.loads(
            decoded,
            object_pairs_hook=_duplicate_object,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except PlanError:
        raise
    except (json.JSONDecodeError, RecursionError) as error:
        raise PlanError("invalid_json", error) from error

    try:
        canonical = canonical_json_bytes(value)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as error:
        raise PlanError("invalid_json_domain", error) from error
    if raw != canonical:
        raise PlanError("noncanonical_json", "plan bytes differ from canonical JSON encoding")

    top = _exact_keys(value, _PLAN_TOP_LEVEL_KEYS, "plan")
    if top["schema"] != PLAN_SCHEMA:
        raise PlanError("schema_mismatch", f"plan.schema must equal {PLAN_SCHEMA}")
    label = _text(top["case_label"], "plan.case_label", 64)
    if not _SAFE_LABEL.fullmatch(label):
        raise PlanError("invalid_case_label", "case label is outside the safe 1..64 character grammar")
    architecture = _text(top["expected_architecture"], "plan.expected_architecture", 16)
    if architecture not in {"x86_64", "aarch64"}:
        raise PlanError("unsupported_architecture", "expected_architecture must be x86_64 or aarch64")

    roots_value = _exact_keys(
        top["roots"], frozenset({"acquisition_parent", "archive_parent"}), "plan.roots"
    )
    roots = RootPlan(
        acquisition_parent=_active_local_root(
            roots_value["acquisition_parent"], "plan.roots.acquisition_parent"
        ),
        archive_parent=_active_local_root(
            roots_value["archive_parent"], "plan.roots.archive_parent"
        ),
    )
    if roots.acquisition_parent == roots.archive_parent:
        raise PlanError("root_alias", "active scratch and immutable archive parents must differ")
    acquisition_root = pathlib.PurePosixPath(roots.acquisition_parent)
    archive_root = pathlib.PurePosixPath(roots.archive_parent)
    if acquisition_root in archive_root.parents or archive_root in acquisition_root.parents:
        raise PlanError(
            "root_overlap",
            "active scratch and immutable archive parents must not contain one another",
        )
    if not _V6_ROOT_PATH.fullmatch(roots.acquisition_parent):
        raise PlanError(
            "v6_acquisition_root_grammar",
            "acquisition_parent components must satisfy the frozen v6 ASCII root grammar",
        )

    dependency_value = _exact_keys(
        top["dependencies"],
        frozenset({"cpm_cmake", "googletest", "benchmark"}),
        "plan.dependencies",
    )
    cpm_value = _exact_keys(
        dependency_value["cpm_cmake"], frozenset({"path", "sha256"}), "plan.dependencies.cpm_cmake"
    )
    dependencies = DependencyPlan(
        cpm_cmake=CpmDependency(
            path=_absolute_path(cpm_value["path"], "plan.dependencies.cpm_cmake.path"),
            sha256=_sha256(cpm_value["sha256"], "plan.dependencies.cpm_cmake.sha256"),
        ),
        googletest=_parse_dependency(dependency_value["googletest"], "plan.dependencies.googletest"),
        benchmark=_parse_dependency(dependency_value["benchmark"], "plan.dependencies.benchmark"),
    )

    tool_value = _exact_keys(top["tools"], frozenset(REQUIRED_TOOL_NAMES), "plan.tools")
    tools = {
        name: _absolute_path(tool_value[name], f"plan.tools.{name}")
        for name in REQUIRED_TOOL_NAMES
    }
    if len(set(tools.values())) != len(tools):
        raise PlanError("tool_alias", "each declared tool must have a distinct exact path")
    if tools["cc"] != "/usr/bin/cc" or tools["cxx"] != "/usr/bin/c++":
        raise PlanError(
            "nonfrozen_compiler_driver",
            "R0 requires exact /usr/bin/cc and /usr/bin/c++ compiler-driver paths",
        )

    safety_value = _exact_keys(
        top["safety"],
        frozenset(
            {
                "thermal_ceiling_millicelsius",
                "poll_interval_milliseconds",
                "maximum_poll_gap_milliseconds",
                "mandatory_thermal_sensors",
                "minimum_free_start_bytes",
                "minimum_free_during_bytes",
                "maximum_observed_free_space_decline_bytes",
                "maximum_archive_bytes",
                "maximum_safety_samples",
                "maximum_sensor_bytes",
            }
        ),
        "plan.safety",
    )
    sensor_value = safety_value["mandatory_thermal_sensors"]
    if not isinstance(sensor_value, list) or not sensor_value:
        raise PlanError("invalid_sensor_population", "at least one mandatory thermal sensor is required")
    if len(sensor_value) > 64:
        raise PlanError("invalid_sensor_population", "at most 64 mandatory thermal sensors are allowed")
    sensors: list[ThermalSensorPlan] = []
    for index, raw_sensor in enumerate(sensor_value):
        location = f"plan.safety.mandatory_thermal_sensors[{index}]"
        sensor = _exact_keys(
            raw_sensor,
            frozenset(
                {"label", "path", "minimum_plausible_millicelsius"}
            ),
            location,
        )
        sensor_label = _text(sensor["label"], f"{location}.label", 64)
        if not _SAFE_LABEL.fullmatch(sensor_label):
            raise PlanError("invalid_sensor_label", f"{location}.label is outside the safe grammar")
        sensors.append(
            ThermalSensorPlan(
                label=sensor_label,
                path=_absolute_path(sensor["path"], f"{location}.path"),
                minimum_plausible_millicelsius=_integer(
                    sensor["minimum_plausible_millicelsius"],
                    f"{location}.minimum_plausible_millicelsius",
                    0,
                    89_999,
                ),
            )
        )
    if len({item.label for item in sensors}) != len(sensors):
        raise PlanError("duplicate_sensor_label", "thermal sensor labels must be unique")
    if len({item.path for item in sensors}) != len(sensors):
        raise PlanError("duplicate_sensor_path", "thermal sensor paths must be unique")
    safety = SafetyPlan(
        thermal_ceiling_millicelsius=_integer(
            safety_value["thermal_ceiling_millicelsius"],
            "plan.safety.thermal_ceiling_millicelsius",
            1,
            200_000,
        ),
        poll_interval_milliseconds=_integer(
            safety_value["poll_interval_milliseconds"],
            "plan.safety.poll_interval_milliseconds",
            1,
            3_600_000,
        ),
        maximum_poll_gap_milliseconds=_integer(
            safety_value["maximum_poll_gap_milliseconds"],
            "plan.safety.maximum_poll_gap_milliseconds",
            1,
            3_600_000,
        ),
        mandatory_thermal_sensors=tuple(sensors),
        minimum_free_start_bytes=_integer(
            safety_value["minimum_free_start_bytes"], "plan.safety.minimum_free_start_bytes"
        ),
        minimum_free_during_bytes=_integer(
            safety_value["minimum_free_during_bytes"], "plan.safety.minimum_free_during_bytes"
        ),
        maximum_observed_free_space_decline_bytes=_integer(
            safety_value["maximum_observed_free_space_decline_bytes"],
            "plan.safety.maximum_observed_free_space_decline_bytes",
        ),
        maximum_archive_bytes=_integer(
            safety_value["maximum_archive_bytes"], "plan.safety.maximum_archive_bytes"
        ),
        maximum_safety_samples=_integer(
            safety_value["maximum_safety_samples"], "plan.safety.maximum_safety_samples", 1, 10_000_000
        ),
        maximum_sensor_bytes=_integer(
            safety_value["maximum_sensor_bytes"], "plan.safety.maximum_sensor_bytes", 1, 65_536
        ),
    )
    if safety.maximum_poll_gap_milliseconds < safety.poll_interval_milliseconds:
        raise PlanError("invalid_poll_gap", "maximum poll gap must not be below polling interval")
    if safety.thermal_ceiling_millicelsius != 90_000:
        raise PlanError(
            "nonfrozen_thermal_ceiling",
            "the R0 protocol requires an explicitly declared 90000 mC ceiling",
        )
    if safety.minimum_free_start_bytes < safety.minimum_free_during_bytes:
        raise PlanError("invalid_storage_threshold", "start free-space threshold must be >= during threshold")
    if (
        safety.maximum_archive_bytes
        > safety.maximum_observed_free_space_decline_bytes
    ):
        raise PlanError(
            "invalid_storage_bound",
            "maximum_archive_bytes is an archive payload cap and cannot exceed "
            "the global observed free-space-decline envelope on the exact shared mount",
        )
    execution_value = _exact_keys(
        top["execution"],
        frozenset(
            {
                "build_jobs",
                "ctest_jobs",
                "clean_path",
                "cleanup_grace_seconds",
                "maximum_proc_entries",
                "maximum_observed_live_descendants",
                "maximum_descendant_identities",
                "read_chunk_bytes",
                "maximum_evidence_bytes",
                "maximum_cli_output_bytes",
                "maximum_json_evidence_bytes",
                "maximum_archive_regular_files_excluding_manifest",
                "maximum_archive_directories_excluding_root",
                "maximum_manifest_bytes",
                "maximum_state_bytes",
                "maximum_journal_entries",
                "maximum_journal_entry_bytes",
                "step_limits",
            }
        ),
        "plan.execution",
    )
    build_jobs = _integer(execution_value["build_jobs"], "plan.execution.build_jobs", 1, 1024)
    ctest_jobs = _integer(execution_value["ctest_jobs"], "plan.execution.ctest_jobs", 1, 1024)
    if build_jobs != 4 or ctest_jobs != 1:
        raise PlanError("nonfrozen_parallelism", "R0 requires build_jobs=4 and ctest_jobs=1")
    clean_path = _text(execution_value["clean_path"], "plan.execution.clean_path", 8192)
    path_items = clean_path.split(":")
    if any(not item for item in path_items) or len(set(path_items)) != len(path_items):
        raise PlanError("invalid_clean_path", "clean_path must contain unique nonempty absolute components")
    for index, item in enumerate(path_items):
        _absolute_path(item, f"plan.execution.clean_path[{index}]", permit_root=False)
    missing_tool_parents = sorted(
        {str(pathlib.PurePosixPath(path).parent) for path in tools.values()} - set(path_items)
    )
    if missing_tool_parents:
        raise PlanError("incomplete_clean_path", f"clean_path omits tool parents {missing_tool_parents}")

    step_value = _exact_keys(
        execution_value["step_limits"], frozenset(REQUIRED_STEP_LIMIT_NAMES), "plan.execution.step_limits"
    )
    step_limits: dict[str, StepLimit] = {}
    for name in REQUIRED_STEP_LIMIT_NAMES:
        item = _exact_keys(
            step_value[name], frozenset({"wall_seconds", "stdout_bytes", "stderr_bytes"}),
            f"plan.execution.step_limits.{name}",
        )
        step_limits[name] = StepLimit(
            wall_seconds=_integer(item["wall_seconds"], f"plan.execution.step_limits.{name}.wall_seconds"),
            stdout_bytes=_integer(item["stdout_bytes"], f"plan.execution.step_limits.{name}.stdout_bytes"),
            stderr_bytes=_integer(item["stderr_bytes"], f"plan.execution.step_limits.{name}.stderr_bytes"),
        )
    execution = ExecutionPlan(
        build_jobs=build_jobs,
        ctest_jobs=ctest_jobs,
        clean_path=clean_path,
        cleanup_grace_seconds=_integer(
            execution_value["cleanup_grace_seconds"], "plan.execution.cleanup_grace_seconds", 1, 3600
        ),
        maximum_proc_entries=_integer(
            execution_value["maximum_proc_entries"], "plan.execution.maximum_proc_entries", 1, 10_000_000
        ),
        maximum_observed_live_descendants=_integer(
            execution_value["maximum_observed_live_descendants"],
            "plan.execution.maximum_observed_live_descendants",
            1,
            1_000_000,
        ),
        maximum_descendant_identities=_integer(
            execution_value["maximum_descendant_identities"],
            "plan.execution.maximum_descendant_identities",
            1,
            10_000_000,
        ),
        read_chunk_bytes=_integer(
            execution_value["read_chunk_bytes"], "plan.execution.read_chunk_bytes", 1, 16 * 1024 * 1024
        ),
        maximum_evidence_bytes=_integer(
            execution_value["maximum_evidence_bytes"],
            "plan.execution.maximum_evidence_bytes",
            1,
            512 * 1024 * 1024,
        ),
        maximum_cli_output_bytes=_integer(
            execution_value["maximum_cli_output_bytes"],
            "plan.execution.maximum_cli_output_bytes",
            256,
            16 * 1024 * 1024,
        ),
        maximum_json_evidence_bytes=_integer(
            execution_value["maximum_json_evidence_bytes"],
            "plan.execution.maximum_json_evidence_bytes",
            256,
            safety.maximum_archive_bytes,
        ),
        maximum_archive_regular_files_excluding_manifest=_integer(
            execution_value["maximum_archive_regular_files_excluding_manifest"],
            "plan.execution.maximum_archive_regular_files_excluding_manifest",
            1,
            1_000_000,
        ),
        maximum_archive_directories_excluding_root=_integer(
            execution_value["maximum_archive_directories_excluding_root"],
            "plan.execution.maximum_archive_directories_excluding_root",
            1,
            1_000_000,
        ),
        maximum_manifest_bytes=_integer(
            execution_value["maximum_manifest_bytes"],
            "plan.execution.maximum_manifest_bytes",
            256,
            safety.maximum_archive_bytes,
        ),
        maximum_state_bytes=_integer(
            execution_value["maximum_state_bytes"],
            "plan.execution.maximum_state_bytes",
            256,
            safety.maximum_archive_bytes,
        ),
        maximum_journal_entries=_integer(
            execution_value["maximum_journal_entries"],
            "plan.execution.maximum_journal_entries",
            11,
            1_000_000,
        ),
        maximum_journal_entry_bytes=_integer(
            execution_value["maximum_journal_entry_bytes"],
            "plan.execution.maximum_journal_entry_bytes",
            256,
            safety.maximum_archive_bytes,
        ),
        step_limits=types.MappingProxyType(step_limits),
    )
    if (
        execution.maximum_descendant_identities
        < execution.maximum_observed_live_descendants
    ):
        raise PlanError(
            "invalid_descendant_bounds",
            "maximum_descendant_identities must be >= "
            "maximum_observed_live_descendants",
        )
    if execution.maximum_archive_regular_files_excluding_manifest < 6:
        raise PlanError(
            "invalid_archive_population_bound",
            "maximum_archive_regular_files_excluding_manifest must hold the "
            "five mandatory pre-recovery files plus one recovery marker reserve",
        )
    if execution.maximum_archive_directories_excluding_root < 8:
        raise PlanError(
            "invalid_archive_population_bound",
            "maximum_archive_directories_excluding_root must be at least 8 for the frozen initial layout",
        )
    if (
        execution.maximum_manifest_bytes
        + execution.maximum_json_evidence_bytes
        >= safety.maximum_archive_bytes
    ):
        raise PlanError(
            "invalid_storage_bound",
            "manifest and recovery-marker reserves must leave a positive acquisition payload budget",
        )
    descriptor_bytes = _descriptor_json_bytes(_PROCEDURE_DESCRIPTOR)
    if len(descriptor_bytes) > execution.maximum_json_evidence_bytes:
        raise PlanError(
            "invalid_storage_bound",
            "maximum_json_evidence_bytes cannot hold the frozen procedure descriptor",
        )
    initial_payload_bytes = len(canonical) + len(descriptor_bytes)
    minimum_initial_failure_sealing_bytes = (
        initial_payload_bytes
        + execution.maximum_json_evidence_bytes
        + execution.maximum_state_bytes
        + execution.maximum_journal_entry_bytes
        + 1
        + execution.maximum_json_evidence_bytes
        + execution.maximum_manifest_bytes
    )
    if minimum_initial_failure_sealing_bytes > safety.maximum_archive_bytes:
        raise PlanError(
            "invalid_storage_bound",
            "archive budget cannot hold plan, procedure, bounded preflight, "
            "first state/journal checkpoint, recovery marker, and manifest",
        )

    return RunPlan(
        schema=PLAN_SCHEMA,
        case_label=label,
        expected_commit=_sha1(top["expected_commit"], "plan.expected_commit"),
        expected_architecture=architecture,
        expected_hostname=_text(top["expected_hostname"], "plan.expected_hostname", 255),
        source_repository=_absolute_path(top["source_repository"], "plan.source_repository"),
        roots=roots,
        dependencies=dependencies,
        tools=types.MappingProxyType(tools),
        safety=safety,
        execution=execution,
        canonical_bytes=canonical,
        sha256=hashlib.sha256(canonical).hexdigest(),
    )


def load_plan(path: os.PathLike[str] | str) -> RunPlan:
    """Read a bounded regular plan through the archive I/O implementation."""

    try:
        raw = _archive_module().read_regular_file_bounded(
            pathlib.Path(path), MAXIMUM_PLAN_BYTES
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as error:
        code = getattr(getattr(error, "code", None), "value", "plan_read")
        detail = getattr(error, "detail", str(error))
        raise PlanError("plan_read", f"{code}: {detail}") from error
    if hasattr(raw, "data"):
        raw = raw.data
    if not isinstance(raw, bytes):
        raise ControllerError("archive_api_contract", "bounded plan read did not return bytes")
    return parse_plan(raw)


def _validated_host_home() -> str:
    """Return the unchanged login home after a strict passwd/ambient binding."""

    try:
        declared = pwd.getpwuid(os.getuid()).pw_dir
    except (KeyError, OSError) as error:
        raise ControllerError("host_home_passwd", error) from error
    try:
        normalized = _absolute_path(declared, "passwd.home", permit_root=False)
    except PlanError as error:
        raise ControllerError("host_home_path", error.detail) from error
    if normalized != declared:
        raise ControllerError(
            "host_home_normalization", f"passwd={declared!r} normalized={normalized!r}"
        )
    path = pathlib.Path(normalized)
    try:
        information = os.lstat(path)
    except OSError as error:
        raise ControllerError("host_home_stat", error) from error
    if stat.S_ISLNK(information.st_mode) or not stat.S_ISDIR(information.st_mode):
        raise ControllerError("host_home_type", normalized)
    ambient = os.environ.get("HOME")
    if ambient is not None and ambient != normalized:
        raise ControllerError(
            "host_home_ambient_mismatch",
            f"passwd={normalized!r} ambient={ambient!r}",
        )
    return normalized


def build_clean_environment(plan: RunPlan, workspace: RunWorkspace, condition: str) -> dict[str, str]:
    """Construct a closed environment plus the unchanged validated login HOME."""

    if condition not in {name for name, _, _ in R0_CONDITIONS}:
        raise ControllerError("unknown_condition", condition)
    dependency_cache = workspace.conditions[condition].dependency_root / "cpm-cache"
    return {
        "CC": plan.tools["cc"],
        "CXX": plan.tools["cxx"],
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": _validated_host_home(),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": plan.execution.clean_path,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": str(workspace.temporary),
        "TZ": "UTC",
        "XDG_CACHE_HOME": str(workspace.xdg_cache),
        "XDG_CONFIG_HOME": str(workspace.xdg_config),
        "XDG_DATA_HOME": str(workspace.xdg_data),
        "XDG_STATE_HOME": str(workspace.xdg_state),
        "CPM_SOURCE_CACHE": str(dependency_cache),
    }


def configure_argv(plan: RunPlan, workspace: RunWorkspace, condition: str) -> tuple[str, ...]:
    condition_spec = {name: (tests, physim) for name, tests, physim in R0_CONDITIONS}
    try:
        tests, physim = condition_spec[condition]
    except KeyError as error:
        raise ControllerError("unknown_condition", condition) from error
    paths = workspace.conditions[condition]
    definitions = {
        "CCACHE_ACTIVE": "OFF",
        "CMAKE_BUILD_TYPE": "RelWithDebInfo",
        "CMAKE_C_COMPILER": plan.tools["cc"],
        "CMAKE_C_COMPILER_LAUNCHER": "",
        "CMAKE_CXX_COMPILER": plan.tools["cxx"],
        "CMAKE_CXX_COMPILER_LAUNCHER": "",
        "CMAKE_DISABLE_FIND_PACKAGE_GTest": "TRUE",
        "CMAKE_DISABLE_FIND_PACKAGE_benchmark": "TRUE",
        "CMAKE_EXPORT_NO_PACKAGE_REGISTRY": "ON",
        "CMAKE_FIND_USE_PACKAGE_REGISTRY": "FALSE",
        "CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY": "FALSE",
        "CMAKE_MAKE_PROGRAM": plan.tools["ninja"],
        "CPM_SOURCE_CACHE": str(paths.dependency_root / "cpm-cache"),
        "ENABLE_PHYSIM_TESTS": "ON" if physim else "OFF",
        "ENABLE_TESTS": "ON" if tests else "OFF",
        "FETCHCONTENT_FULLY_DISCONNECTED": "ON",
        "FETCHCONTENT_SOURCE_DIR_BENCHMARK": str(paths.dependency_root / "benchmark"),
        "FETCHCONTENT_SOURCE_DIR_GOOGLETEST": str(paths.dependency_root / "googletest"),
        "FETCHCONTENT_UPDATES_DISCONNECTED": "ON",
    }
    definitions.update({name: "OFF" for name in OPTIONAL_CMAKE_FEATURES_DISABLED})
    return (
        plan.tools["cmake"],
        "-S",
        str(workspace.source),
        "-B",
        str(paths.build),
        "-G",
        "Ninja",
        *(f"-D{name}={definitions[name]}" for name in sorted(definitions)),
    )


def build_argv(plan: RunPlan, workspace: RunWorkspace) -> tuple[str, ...]:
    build = workspace.conditions["ON"].build
    return (
        plan.tools["ninja"],
        "-C",
        str(build),
        "-j",
        str(plan.execution.build_jobs),
        *R0_BUILD_TARGETS,
    )


def ctest_inventory_argv(plan: RunPlan, workspace: RunWorkspace, population: str) -> tuple[str, ...]:
    if population == "r0":
        expression = "^oai_memprof_r0_"
    elif population == "wire":
        expression = "^oai_memprof_wire_"
    else:
        raise ControllerError("unknown_ctest_population", population)
    return (
        plan.tools["ctest"],
        "--test-dir",
        str(workspace.conditions["ON"].build),
        "-N",
        "-R",
        expression,
    )


def ctest_run_argv(plan: RunPlan, workspace: RunWorkspace, population: str) -> tuple[str, ...]:
    inventory = ctest_inventory_argv(plan, workspace, population)
    return (
        inventory[0],
        "--test-dir",
        inventory[2],
        "--verbose",
        "--no-tests=error",
        "-j",
        str(plan.execution.ctest_jobs),
        "-R",
        inventory[-1],
    )


def absence_argv(
    plan: RunPlan,
    workspace: RunWorkspace,
    condition: str,
    build_ninja: pathlib.Path,
    targets: pathlib.Path,
    commands: pathlib.Path,
) -> tuple[str, ...]:
    build = workspace.conditions[condition].build
    return (
        plan.tools["python"],
        "-B",
        str(workspace.source / ABSENCE_CHECKER_RELATIVE_PATH),
        "--build-ninja",
        str(build_ninja),
        "--targets",
        str(targets),
        "--commands",
        str(commands),
        "--source-root",
        str(workspace.source),
        "--build-root",
        str(build),
        "--max-evidence-bytes",
        str(plan.execution.maximum_evidence_bytes),
    )


def positive_elf_argv(plan: RunPlan, workspace: RunWorkspace) -> tuple[str, ...]:
    build = workspace.conditions["ON"].build
    tests = build / "common/utils/memprof/tests"
    return (
        plan.tools["python"],
        "-B",
        str(workspace.source / "common/utils/memprof/tests/check_oai_memprof_r0_elf.py"),
        "--a00-exe",
        str(tests / "test_oai_memprof_r0_actual_a00"),
        "--a01-exe",
        str(tests / "test_oai_memprof_r0_actual_a01"),
        "--a00-dso",
        str(tests / "liboai_memprof_r0_actual_dso_a00.so"),
        "--a01-dso",
        str(tests / "liboai_memprof_r0_actual_dso_a01.so"),
        "--runtime",
        str(build / "common/utils/memprof/liboai_memprof_runtime.so.1.0.0"),
        "--wrapper-archive",
        str(build / "common/utils/memprof/liboai_memprof_wrap_c.a"),
        "--a00-exe-map",
        str(tests / "oai_memprof_r0_a00_exe.map"),
        "--a01-exe-map",
        str(tests / "oai_memprof_r0_a01_exe.map"),
        "--a00-dso-map",
        str(tests / "oai_memprof_r0_a00_dso.map"),
        "--a01-dso-map",
        str(tests / "oai_memprof_r0_a01_dso.map"),
        "--build-dir",
        str(build),
        "--source-dir",
        str(workspace.source),
        *(item for target in (
            "test_oai_memprof_r0_actual_a00",
            "test_oai_memprof_r0_actual_a01",
            "oai_memprof_r0_actual_dso_a00",
            "oai_memprof_r0_actual_dso_a01",
        ) for item in ("--ninja-target", target)),
    )


def parse_ctest_inventory(
    output: bytes | str,
    expected: Sequence[str],
    *,
    progress_callback: Callable[[], None] | None = None,
) -> tuple[str, ...]:
    names: list[str] = []
    totals: list[int] = []
    numbers: list[int] = []
    for line in _iter_bounded_utf8_lines(
        output, "ctest_inventory", progress_callback=progress_callback
    ):
        if progress_callback is not None:
            progress_callback()
        match = _CTEST_LINE.fullmatch(line)
        if match:
            numbers.append(int(match.group("number")))
            names.append(match.group("name"))
        total = _CTEST_TOTAL.fullmatch(line)
        if total:
            totals.append(int(total.group("count")))
    if len(totals) != 1:
        raise GateError("ctest_inventory_total", "expected exactly one Total Tests line")
    if totals[0] != len(names):
        raise GateError("ctest_inventory_count", f"reported {totals[0]}, parsed {len(names)}")
    if len(numbers) != len(set(numbers)):
        raise GateError("ctest_inventory_duplicate_number", "duplicate CTest number")
    if len(names) != len(set(names)):
        raise GateError("ctest_inventory_duplicate_name", "duplicate CTest name")
    expected_tuple = tuple(expected)
    if tuple(names) != expected_tuple:
        raise GateError("ctest_inventory_population", f"expected {expected_tuple!r}, observed {tuple(names)!r}")
    return tuple(names)


def validate_ctest_run(
    output: bytes | str,
    expected: Sequence[str],
    *,
    progress_callback: Callable[[], None] | None = None,
) -> None:
    expected_tuple = tuple(expected)
    if not expected_tuple or len(expected_tuple) != len(set(expected_tuple)):
        raise GateError("ctest_run_expected_population", "expected names must be unique and nonempty")
    results: list[tuple[str, str]] = []
    summaries: list[int] = []
    nonzero_failure_summary = False
    for line in _iter_bounded_utf8_lines(
        output, "ctest_run", progress_callback=progress_callback
    ):
        if progress_callback is not None:
            progress_callback()
        match = _CTEST_RESULT.fullmatch(line)
        if match:
            results.append((match.group("name"), match.group("status")))
        summary = _CTEST_PASS.fullmatch(line)
        if summary:
            summaries.append(int(summary.group("count")))
        if re.fullmatch(r"[1-9][0-9]* tests failed out of [0-9]+", line):
            nonzero_failure_summary = True
    observed_names = tuple(name for name, _ in results)
    if observed_names != expected_tuple:
        raise GateError(
            "ctest_run_population",
            f"expected {expected_tuple!r}, observed {observed_names!r}",
        )
    invalid = [(name, status) for name, status in results if status != "Passed"]
    if invalid:
        raise GateError("ctest_run_status", f"non-pass terminal status {invalid!r}")
    if len(summaries) != 1:
        raise GateError("ctest_run_summary", "expected exactly one all-pass summary")
    if summaries[0] != len(expected_tuple):
        raise GateError("ctest_run_count", "all-pass summary has the wrong population size")
    if nonzero_failure_summary:
        raise GateError("ctest_run_failure", "output contains a nonzero failed-test summary")


def _iter_bounded_utf8_lines(
    value: bytes | str,
    role: str,
    *,
    progress_callback: Callable[[], None] | None = None,
    chunk_bytes: int = PARSER_PROGRESS_CHUNK_BYTES,
    maximum_line_bytes: int = MAXIMUM_PARSER_LINE_BYTES,
) -> Sequence[str]:
    """Decode LF-delimited evidence with bounded lines and progress boundaries.

    Production call sites supply bytes from a callback-sampled ``BoundedFileImage``
    or retained command capture.  The string form exists only for pure parser
    tests and is converted under the same UTF-8 and line-size contract.
    """

    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise GateError("parser_chunk_bound", role)
    if type(maximum_line_bytes) is not int or maximum_line_bytes <= 0:
        raise GateError("parser_line_bound", role)
    if type(value) is str:
        try:
            data = value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise GateError("invalid_output_utf8", f"{role}: {error}") from error
    elif type(value) is bytes:
        data = value
    else:
        raise GateError("wrong_output_type", f"{role} must be exact bytes or str")

    lines: list[str] = []
    pending = bytearray()

    def emit(raw: bytes) -> None:
        if raw.endswith(b"\r"):
            raw = raw[:-1]
        try:
            lines.append(raw.decode("utf-8", errors="strict"))
        except UnicodeDecodeError as error:
            raise GateError("invalid_output_utf8", f"{role}: {error}") from error

    if progress_callback is not None:
        progress_callback()
    for offset in range(0, len(data), chunk_bytes):
        chunk = data[offset : offset + chunk_bytes]
        cursor = 0
        while cursor < len(chunk):
            newline = chunk.find(b"\n", cursor)
            end = len(chunk) if newline < 0 else newline
            pending.extend(chunk[cursor:end])
            if len(pending) > maximum_line_bytes:
                raise GateError(
                    "parser_line_bound",
                    f"{role} line exceeds {maximum_line_bytes} bytes",
                )
            if newline < 0:
                break
            emit(bytes(pending))
            pending.clear()
            cursor = newline + 1
        if progress_callback is not None:
            progress_callback()
    if pending:
        emit(bytes(pending))
    if progress_callback is not None:
        progress_callback()
    return tuple(lines)


def _decode_bounded_utf8(value: bytes | str, role: str) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, bytes):
        raise GateError("wrong_output_type", f"{role} must be bytes or str")
    try:
        return value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise GateError("invalid_output_utf8", f"{role}: {error}") from error


def _single_ascii_line(value: bytes | str, role: str) -> str:
    text = _decode_bounded_utf8(value, role)
    lines = text.splitlines()
    if len(lines) != 1 or text not in {lines[0], lines[0] + "\n"}:
        raise GateError("invalid_single_line_output", role)
    try:
        lines[0].encode("ascii", errors="strict")
    except UnicodeEncodeError as error:
        raise GateError("invalid_ascii_output", f"{role}: {error}") from error
    return lines[0]


def _strict_result_json(value: bytes | str, role: str) -> Mapping[str, object]:
    text = _decode_bounded_utf8(value, role)
    try:
        result = json.loads(
            text,
            object_pairs_hook=_duplicate_object,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except PlanError as error:
        raise GateError("invalid_result_json", f"{role}: {error}") from error
    except (json.JSONDecodeError, RecursionError) as error:
        raise GateError("invalid_result_json", f"{role}: {error}") from error
    if not isinstance(result, dict):
        raise GateError("invalid_result_shape", f"{role} must be a JSON object")
    return result


def validate_absence_result(
    output: bytes | str,
    *,
    source_root: pathlib.Path,
    build_root: pathlib.Path,
    identities: Sequence[EvidenceIdentity],
    ld_version_output: bytes | str,
    condition: str,
) -> AbsenceDecision:
    if condition not in {"OFF0", "OFF1"}:
        raise GateError("absence_condition", condition)
    result = _strict_result_json(output, "absence_result")
    raw_result = output.encode("utf-8", errors="strict") if isinstance(output, str) else output
    if raw_result != canonical_json_bytes(result) + b"\n":
        raise GateError("absence_result_encoding", "v6 result is not exact canonical JSON plus LF")
    expected_keys = {
        "catalog_digest",
        "catalog_entries",
        "catalog_version",
        "claim",
        "evidence",
        "expected_final_targets",
        "final_link_model",
        "match_count",
        "matches",
        "matches_omitted",
        "provenance",
        "root_model",
        "roots",
        "schema",
        "verdict",
        "wrap_model",
    }
    if set(result) != expected_keys:
        raise GateError(
            "absence_result_members",
            f"missing={sorted(expected_keys - set(result))} unknown={sorted(set(result) - expected_keys)}",
        )
    exact_scalars = {
        "schema": ABSENCE_SCHEMA,
        "catalog_version": ABSENCE_CATALOG_VERSION,
        "catalog_entries": ABSENCE_CATALOG_ENTRIES,
        "catalog_digest": ABSENCE_CATALOG_SHA256,
        "verdict": "pass",
        "match_count": 0,
        "matches_omitted": 0,
    }
    for name, expected in exact_scalars.items():
        observed_scalar = result.get(name)
        if type(observed_scalar) is not type(expected) or observed_scalar != expected:
            raise GateError(
                "absence_identity",
                f"{name} expected {expected!r}, observed {observed_scalar!r}",
            )
    if result.get("matches") != []:
        raise GateError("absence_matches", "passing v6 result must retain no matches")
    if result.get("expected_final_targets") != list(ABSENCE_EXPECTED_FINAL_TARGETS):
        raise GateError("absence_target_population", "v6 final-target population differs")
    roots = result.get("roots")
    if roots != {"build": str(build_root), "source": str(source_root)}:
        raise GateError("absence_roots", f"v6 roots differ: {roots!r}")
    contracts = {
        "claim": ABSENCE_CLAIM,
        "final_link_model": ABSENCE_FINAL_LINK_MODEL,
        "provenance": ABSENCE_PROVENANCE,
        "root_model": ABSENCE_ROOT_MODEL,
        "wrap_model": ABSENCE_WRAP_MODEL,
    }
    for name, expected in contracts.items():
        if result.get(name) != expected:
            raise GateError("absence_claim_contract", f"v6 {name} differs")

    expected_identity = {
        item.role: {
            "bytes": item.bytes,
            "device": item.device,
            "inode": item.inode,
            "lines": item.lines,
            "path": item.path,
            "role": item.role,
            "sha256": item.sha256,
        }
        for item in identities
    }
    if set(expected_identity) != {"build_ninja", "targets", "commands"}:
        raise GateError("absence_ledger_population", "external ledger must contain exactly three v6 roles")
    evidence = result.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != 3:
        raise GateError("absence_evidence_population", "v6 evidence must contain exactly three roles")
    observed: dict[str, object] = {}
    for item in evidence:
        if not isinstance(item, dict) or not isinstance(item.get("role"), str):
            raise GateError("absence_evidence_shape", "malformed v6 evidence member")
        role = item["role"]
        if role in observed:
            raise GateError("absence_evidence_duplicate", f"duplicate evidence role {role}")
        observed[role] = item
    if observed != expected_identity:
        raise GateError("absence_external_identity", "v6 self-hashed identities differ from external ledger")

    ld_text = _decode_bounded_utf8(ld_version_output, "ld_version")
    first_line = ld_text.splitlines()[0] if ld_text.splitlines() else ""
    reference_match = bool(_LD_242.fullmatch(first_line))
    return AbsenceDecision(
        condition=condition,
        tool_reference_match=reference_match,
        eligibility="QUALIFIED" if reference_match else "SUPPORTING_ONLY",
        reason="REFERENCE_TOOL_MATCH" if reference_match else "SUPPORTING_ONLY_TOOL_REFERENCE_MISMATCH",
    )


def validate_positive_elf_output(
    stdout: bytes | str, stderr: bytes | str, architecture: str
) -> None:
    out = _decode_bounded_utf8(stdout, "elf_gate_stdout")
    err = _decode_bounded_utf8(stderr, "elf_gate_stderr")
    expected = (
        f"R0_ELF_GATE_V2 pass architecture={architecture} roles=4 wrappers=4 "
        "runtime_soname=liboai_memprof_runtime.so.1 generator=Ninja compiler=GNU\n"
    )
    if out != expected:
        raise GateError("positive_elf_stdout", f"unexpected positive gate output {out!r}")
    if err != "":
        raise GateError("positive_elf_stderr", "positive ELF gate stderr is not empty")


def validate_controller_selftest_output(
    stdout: bytes | str, stderr: bytes | str
) -> None:
    result = _strict_result_json(stdout, "controller_selftest")
    raw = stdout.encode("utf-8", errors="strict") if isinstance(stdout, str) else stdout
    if raw != canonical_json_bytes(result) + b"\n":
        raise GateError(
            "controller_selftest_encoding",
            "self-test result is not exact canonical JSON plus LF",
        )
    expected = {
        "checks": 13,
        "procedure_sha256": PROCEDURE_DESCRIPTOR_SHA256,
        "schema": "oai.memprof.r0.terminal-self-test/v1",
        "verdict": "pass",
    }
    if result != expected:
        raise GateError(
            "controller_selftest_contract",
            f"expected {expected!r}, observed {dict(result)!r}",
        )
    if _decode_bounded_utf8(stderr, "controller_selftest_stderr") != "":
        raise GateError("controller_selftest_stderr", "self-test stderr is not empty")


def validate_artifact_header(path: str, data: bytes, architecture: str) -> None:
    if path not in R0_ARTIFACTS:
        raise GateError("artifact_population", f"unexpected artifact {path!r}")
    machine = 62 if architecture == "x86_64" else 183 if architecture == "aarch64" else -1
    if not isinstance(data, bytes) or not data:
        raise GateError("artifact_empty", path)
    if path in ELF_ARTIFACTS:
        if len(data) < 20 or data[:6] != b"\x7fELF\x02\x01":
            raise GateError("artifact_elf_identity", path)
        observed_machine = int.from_bytes(data[18:20], "little")
        if observed_machine != machine:
            raise GateError("artifact_architecture", f"{path}: e_machine={observed_machine}")
    elif path in ARCHIVE_ARTIFACTS:
        if not data.startswith(b"!<arch>\n"):
            raise GateError("artifact_archive_identity", path)
    elif path in MAP_ARTIFACTS:
        if b"\x00" in data:
            raise GateError("artifact_map_nul", path)
    else:
        raise GateError("artifact_unclassified", path)


def validate_artifact_headers(
    artifacts: Mapping[str, bytes], architecture: str, *, maximum_total_bytes: int | None = None
) -> None:
    if tuple(sorted(artifacts)) != tuple(sorted(R0_ARTIFACTS)):
        raise GateError("artifact_population", "selected artifact population is not exactly the frozen 20")
    total = 0
    for path in R0_ARTIFACTS:
        data = artifacts[path]
        total += len(data)
        if maximum_total_bytes is not None and total > maximum_total_bytes:
            raise GateError("artifact_total_bytes", "selected artifacts exceed the aggregate bound")
        validate_artifact_header(path, data, architecture)


def validate_captured_artifact_stream(
    path: pathlib.Path,
    relative_path: str,
    identity: EvidenceIdentity,
    architecture: str,
    *,
    maximum_bytes: int,
    read_chunk_bytes: int,
    progress_callback: Callable[[], None] | None = None,
) -> None:
    """Validate a captured artifact with constant memory and a stable identity."""

    if (
        maximum_bytes <= 0
        or read_chunk_bytes <= 0
        or (
            progress_callback is not None
            and not callable(progress_callback)
        )
    ):
        raise GateError("artifact_stream_bound", relative_path)

    def progress() -> None:
        if progress_callback is not None:
            progress_callback()

    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        progress()
        descriptor = os.open(path, flags)
        progress()
    except OSError as error:
        raise GateError("artifact_stream_open", f"{relative_path}: {error}") from error
    try:
        progress()
        before = os.fstat(descriptor)
        progress()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise GateError("artifact_stream_type", relative_path)
        if before.st_size <= 0 or before.st_size > maximum_bytes:
            raise GateError("artifact_stream_size", f"{relative_path}: {before.st_size}")
        digest = hashlib.sha256()
        prefix = bytearray()
        total = 0
        contains_nul = False
        final_byte: int | None = None
        line_count = 0
        while True:
            progress()
            chunk = os.read(descriptor, min(read_chunk_bytes, 64 * 1024))
            progress()
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_bytes:
                raise GateError("artifact_stream_size", relative_path)
            digest.update(chunk)
            if len(prefix) < 64:
                prefix.extend(chunk[: 64 - len(prefix)])
            contains_nul = contains_nul or b"\x00" in chunk
            line_count += chunk.count(b"\n")
            final_byte = chunk[-1]
        progress()
        after = os.fstat(descriptor)
        progress()
    except OSError as error:
        raise GateError("artifact_stream_read", f"{relative_path}: {error}") from error
    finally:
        active_error = sys.exc_info()[1]
        try:
            os.close(descriptor)
        except OSError as error:
            if active_error is not None:
                active_error.add_note(f"artifact stream close failed: {error}")
            else:
                raise GateError(
                    "artifact_stream_close", f"{relative_path}: {error}"
                ) from error
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable):
        raise GateError("artifact_stream_changed", relative_path)
    logical_lines = line_count + int(total > 0 and final_byte != 0x0A)
    if (
        before.st_dev != identity.device
        or before.st_ino != identity.inode
        or total != identity.bytes
        or logical_lines != identity.lines
        or digest.hexdigest() != identity.sha256
    ):
        raise GateError("artifact_capture_identity", relative_path)
    if relative_path in MAP_ARTIFACTS:
        if contains_nul:
            raise GateError("artifact_map_nul", relative_path)
        if total == 0:
            raise GateError("artifact_empty", relative_path)
    else:
        validate_artifact_header(relative_path, bytes(prefix), architecture)


def classify_terminal_result(
    *,
    condition_complete: Mapping[str, bool],
    archive_verified: bool,
    interrupted: bool,
    failure_reasons: Sequence[str] = (),
    absence_decisions: Sequence[AbsenceDecision] = (),
    positive_elf_complete: bool = False,
) -> TerminalAxes:
    reasons = list(dict.fromkeys(_bounded_text(item, 128) for item in failure_reasons))
    expected_conditions = {name for name, _, _ in R0_CONDITIONS}
    if set(condition_complete) != expected_conditions:
        reasons.append("CONDITION_POPULATION_INVALID")
    elif any(type(value) is not bool for value in condition_complete.values()):
        reasons.append("CONDITION_STATE_TYPE_INVALID")
    elif not all(condition_complete.values()):
        reasons.append("CONDITION_INCOMPLETE")
    if type(interrupted) is not bool:
        reasons.append("INTERRUPTION_STATE_TYPE_INVALID")
    if interrupted is not False:
        reasons.append("INTERRUPTED_OR_RECOVERED")
    if type(positive_elf_complete) is not bool:
        reasons.append("POSITIVE_ELF_STATE_TYPE_INVALID")
    if positive_elf_complete is not True:
        reasons.append("POSITIVE_ELF_GATE_INCOMPLETE")
    absence_population_valid = bool(
        len(absence_decisions) == 2
        and all(isinstance(item, AbsenceDecision) for item in absence_decisions)
        and {item.condition for item in absence_decisions} == {"OFF0", "OFF1"}
    )
    absence_contract_valid = bool(
        absence_population_valid
        and all(
            type(item.tool_reference_match) is bool
            and item.eligibility
            == ("QUALIFIED" if item.tool_reference_match else "SUPPORTING_ONLY")
            and item.reason
            == (
                "REFERENCE_TOOL_MATCH"
                if item.tool_reference_match
                else "SUPPORTING_ONLY_TOOL_REFERENCE_MISMATCH"
            )
            for item in absence_decisions
        )
    )
    if not absence_population_valid:
        reasons.append("ABSENCE_DECISION_POPULATION_INVALID")
    elif not absence_contract_valid:
        reasons.append("ABSENCE_DECISION_CONTRACT_INVALID")
    if type(archive_verified) is not bool:
        reasons.append("ARCHIVE_STATE_TYPE_INVALID")
    complete = not reasons
    archive_is_verified = archive_verified is True
    if not archive_is_verified:
        reasons.append("ARCHIVE_UNVERIFIED")
    scientific_state = "COMPLETE" if complete else "INCOMPLETE"
    integrity = "VERIFIED" if archive_is_verified else "UNVERIFIED"
    inclusion = "INCLUDED" if complete and archive_is_verified else "EXCLUDED"

    if not absence_contract_valid:
        invalid_absence_reason = (
            "OFF0_OFF1_DECISION_POPULATION_INVALID"
            if not absence_population_valid
            else "OFF0_OFF1_DECISION_CONTRACT_INVALID"
        )
        absence_claim = ClaimDecision(
            "disabled_generated_graph_absence",
            "UNAVAILABLE",
            (invalid_absence_reason,),
        )
    else:
        supporting = tuple(
            dict.fromkeys(
                decision.reason
                for decision in absence_decisions
                if not decision.tool_reference_match
            )
        )
        absence_claim = ClaimDecision(
            "disabled_generated_graph_absence",
            "QUALIFIED" if not supporting else "SUPPORTING_ONLY",
            supporting or ("REFERENCE_TOOL_MATCH",),
        )
    positive_claim = ClaimDecision(
        "four_role_fixture_final_elf",
        "QUALIFIED" if positive_elf_complete is True else "UNAVAILABLE",
        ("POSITIVE_ELF_GATE_PASS",)
        if positive_elf_complete is True
        else ("POSITIVE_ELF_GATE_INCOMPLETE",),
    )
    production_claim = ClaimDecision(
        "production_final_elf_absence",
        "NOT_APPLICABLE",
        (PRODUCTION_FINAL_ELF_ABSENCE,),
    )
    if not complete:
        if absence_claim.eligibility != "UNAVAILABLE":
            absence_claim = ClaimDecision(
                absence_claim.claim,
                "SUPPORTING_ONLY",
                tuple(dict.fromkeys((*absence_claim.reasons, "CASE_INCOMPLETE"))),
            )
        if positive_claim.eligibility != "UNAVAILABLE":
            positive_claim = ClaimDecision(
                positive_claim.claim,
                "SUPPORTING_ONLY",
                tuple(dict.fromkeys((*positive_claim.reasons, "CASE_INCOMPLETE"))),
            )
    if not archive_is_verified:
        if absence_claim.eligibility != "UNAVAILABLE":
            absence_claim = ClaimDecision(
                absence_claim.claim,
                "UNVERIFIED",
                tuple(dict.fromkeys((*absence_claim.reasons, "ARCHIVE_UNVERIFIED"))),
            )
        if positive_claim.eligibility != "UNAVAILABLE":
            positive_claim = ClaimDecision(
                positive_claim.claim,
                "UNVERIFIED",
                tuple(dict.fromkeys((*positive_claim.reasons, "ARCHIVE_UNVERIFIED"))),
            )
    return TerminalAxes(
        scientific_case_state=scientific_state,
        archive_integrity=integrity,
        inclusion=inclusion,
        profiler_stream_state="NOT_APPLICABLE_R0",
        production_final_elf_absence=PRODUCTION_FINAL_ELF_ABSENCE,
        reasons=tuple(dict.fromkeys(reasons)),
        claims=(absence_claim, positive_claim, production_claim),
    )


def validate_host_binding(plan: RunPlan, *, hostname: str | None = None, architecture: str | None = None) -> None:
    observed_hostname = socket.gethostname() if hostname is None else hostname
    machine = platform.machine() if architecture is None else architecture
    normalized_machine = {"AMD64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    if observed_hostname != plan.expected_hostname:
        raise ControllerError(
            "hostname_mismatch", f"expected {plan.expected_hostname!r}, observed {observed_hostname!r}"
        )
    if normalized_machine != plan.expected_architecture:
        raise ControllerError(
            "architecture_mismatch", f"expected {plan.expected_architecture!r}, observed {normalized_machine!r}"
        )
    try:
        running_python = pathlib.Path(sys.executable).resolve(strict=True)
        declared_python = pathlib.Path(plan.tools["python"]).resolve(strict=True)
    except OSError as error:
        raise ControllerError("python_binding", error) from error
    if running_python != declared_python:
        raise ControllerError(
            "python_binding",
            f"controller={running_python} plan={declared_python}",
        )
    if sys.byteorder != "little" or (8 * __import__("struct").calcsize("P")) != 64:
        raise ControllerError("unsupported_abi", "R0 admits only 64-bit little-endian hosts")
    if platform.system() != "Linux":
        raise ControllerError("unsupported_kernel", "R0 admits only GNU/Linux hosts")
    try:
        libc_identity = os.confstr("CS_GNU_LIBC_VERSION")
    except (OSError, ValueError) as error:
        raise ControllerError("unsupported_libc", f"glibc identity unavailable: {error}") from error
    if not isinstance(libc_identity, str) or not libc_identity.startswith("glibc "):
        raise ControllerError("unsupported_libc", f"observed {libc_identity!r}")


class TerminalController:
    """Expand and execute the frozen R0 protocol through injected services."""

    def __init__(self, services: ControllerServices):
        self.services = services

    def _require_success(self, result: object, step: str, instance: str) -> None:
        if not self.services.command_succeeded(result):
            raise ControllerError("command_failed", f"{step}/{instance}")

    def _run(
        self,
        workspace: RunWorkspace,
        plan: RunPlan,
        step: str,
        instance: str,
        argv: Sequence[str],
        cwd: pathlib.Path,
        environment: Mapping[str, str],
    ) -> object:
        result = self.services.run_step(workspace, plan, step, instance, tuple(argv), cwd, environment)
        self.services.write_json(
            workspace,
            f"ledger/{instance}.json",
            {
                "instance": instance,
                "result": dict(self.services.command_record(result)),
                "schema": COMMAND_LEDGER_SCHEMA,
                "step": step,
            },
        )
        self._require_success(result, step, instance)
        return result

    def _write_platform_provenance(
        self, plan: RunPlan, workspace: RunWorkspace
    ) -> None:
        uname = platform.uname()
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
            configured_cpus = int(os.sysconf("SC_NPROCESSORS_CONF"))
            online_cpus = int(os.sysconf("SC_NPROCESSORS_ONLN"))
            affinity = sorted(os.sched_getaffinity(0))
        except (OSError, ValueError) as error:
            raise ControllerError("platform_topology", error) from error
        self.services.write_json(
            workspace,
            "provenance/platform.json",
            {
                "architecture": plan.expected_architecture,
                "byteorder": sys.byteorder,
                "cpu_count": os.cpu_count(),
                "cpu_identity_scope": (
                    "bounded_kernel_topology_only_full_linux_inventory_deferred_R0"
                ),
                "cpu_topology": {
                    "affinity_cpu_ids": affinity,
                    "configured_logical_cpus": configured_cpus,
                    "online_logical_cpus": online_cpus,
                },
                "hostname": socket.gethostname(),
                "login_home": {
                    "ambient_home_present": "HOME" in os.environ,
                    "binding": "passwd_home_equals_ambient_if_present",
                    "path": _validated_host_home(),
                },
                "kernel": {
                    "machine": uname.machine,
                    "release": uname.release,
                    "system": uname.system,
                    "version": uname.version,
                },
                "libc": os.confstr("CS_GNU_LIBC_VERSION"),
                "memory": {
                    "page_size_bytes": page_size,
                    "physical_pages": physical_pages,
                    "physical_memory_bytes": page_size * physical_pages,
                },
                "plan_sha256": plan.sha256,
                "pointer_bits": 8 * __import__("struct").calcsize("P"),
                "procedure_sha256": PROCEDURE_DESCRIPTOR_SHA256,
                "python": {
                    "executable": sys.executable,
                    "implementation": platform.python_implementation(),
                    "version": platform.python_version(),
                },
                "schema": "oai.memprof.r0.platform-provenance/v1",
            },
        )

    @staticmethod
    def _publication_path_observation(
        workspace: RunWorkspace,
    ) -> dict[str, object]:
        """Observe both transaction paths without following a final component.

        This observation is deliberately independent of an archive exception's
        last-completed publication phase.  It is evidence about the filesystem
        state seen by the controller after the exception, not proof of archive
        integrity or durability.
        """

        result: dict[str, object] = {}
        for role, path in (
            ("active", workspace.archive_stage),
            ("final", workspace.archive_destination),
        ):
            try:
                information = os.lstat(path)
            except FileNotFoundError:
                result[role] = {"path": str(path), "state": "ABSENT"}
            except OSError as error:
                result[role] = {
                    "error": _bounded_text(error),
                    "path": str(path),
                    "state": "STAT_FAILED",
                }
            else:
                if stat.S_ISLNK(information.st_mode):
                    observed_type = "SYMLINK"
                elif stat.S_ISDIR(information.st_mode):
                    observed_type = "DIRECTORY"
                elif stat.S_ISREG(information.st_mode):
                    observed_type = "REGULAR_FILE"
                else:
                    observed_type = "OTHER"
                result[role] = {
                    "device": information.st_dev,
                    "inode": information.st_ino,
                    "mode": information.st_mode,
                    "path": str(path),
                    "state": observed_type,
                }
        return result

    @staticmethod
    def _validate_publication_receipt(
        value: object,
        workspace: RunWorkspace,
        terminal_candidate: TerminalAxes,
        plan: RunPlan,
    ) -> Mapping[str, object]:
        expected_keys = {
            "archive_integrity",
            "archive_path",
            "global_safety_timeline",
            "inclusion",
            "manifest",
            "manifest_safety",
            "mutated",
            "postcompletion_operational_problem",
            "production_final_elf_absence",
            "profiler_stream_state",
            "publication_phase",
            "publication_success_marker",
            "publication_success_marker_precommit_safety",
            "run_id",
            "schema",
            "scientific_case_state",
            "terminal_axes",
            "verified",
        }
        if type(value) is not dict or set(value) != expected_keys:
            raise ControllerError(
                "publication_receipt_shape", "success receipt members differ"
            )
        if (
            value["schema"] != "oai.memprof.r0.publication-receipt/v1"
            or value["verified"] is not True
            or value["mutated"] is not True
            or value["archive_integrity"] != "VERIFIED"
            or value["archive_path"] != str(workspace.archive_destination)
            or value["run_id"] != workspace.run_id
            or value["publication_phase"] != "verified"
            or value["scientific_case_state"]
            != terminal_candidate.scientific_case_state
            or value["inclusion"] != terminal_candidate.inclusion
            or value["profiler_stream_state"]
            != terminal_candidate.profiler_stream_state
            or value["production_final_elf_absence"]
            != terminal_candidate.production_final_elf_absence
            or value["terminal_axes"] != terminal_candidate.as_dict()
            or value["postcompletion_operational_problem"] is not None
        ):
            raise ControllerError(
                "publication_receipt_binding",
                "success receipt does not bind the exact run, path, phase, and axes",
            )
        manifest = _validate_manifest_receipt(
            value["manifest"],
            expected_manifest_path=pathlib.PurePosixPath(
                workspace.archive_destination / "manifest.json"
            ),
            maximum_regular_files_excluding_manifest=(
                plan.execution.maximum_archive_regular_files_excluding_manifest
            ),
            maximum_directories_excluding_root=(
                plan.execution.maximum_archive_directories_excluding_root
            ),
            maximum_manifest_bytes=plan.execution.maximum_manifest_bytes,
            maximum_total_bytes_including_manifest=(
                plan.safety.maximum_archive_bytes
            ),
        )
        expected_manifest_keys = {
            "directory_count_excluding_root",
            "identity_claim",
            "local_strict_identity_fields",
            "manifest_bytes",
            "manifest_path",
            "manifest_sha256",
            "observed_archive_entries_including_manifest",
            "observed_directories",
            "observed_regular_files_including_manifest",
            "portable_content_identity_fields",
            "regular_file_count_excluding_manifest",
            "total_regular_file_bytes_excluding_manifest",
            "total_regular_file_bytes_including_manifest",
        }
        if not isinstance(manifest, Mapping) or set(manifest) != expected_manifest_keys:
            raise ControllerError("publication_manifest_shape", "manifest receipt members differ")
        numeric_manifest_fields = (
            "directory_count_excluding_root",
            "manifest_bytes",
            "observed_archive_entries_including_manifest",
            "observed_directories",
            "observed_regular_files_including_manifest",
            "regular_file_count_excluding_manifest",
            "total_regular_file_bytes_excluding_manifest",
            "total_regular_file_bytes_including_manifest",
        )
        if (
            manifest["manifest_path"]
            != str(workspace.archive_destination / "manifest.json")
            or not isinstance(manifest["manifest_sha256"], str)
            or _SHA256.fullmatch(str(manifest["manifest_sha256"])) is None
            or any(
                type(manifest[field]) is not int or int(manifest[field]) < 0
                for field in numeric_manifest_fields
            )
            or manifest["total_regular_file_bytes_including_manifest"]
            != manifest["total_regular_file_bytes_excluding_manifest"]
            + manifest["manifest_bytes"]
            or manifest["directory_count_excluding_root"]
            != manifest["observed_directories"]
            or manifest["regular_file_count_excluding_manifest"] + 1
            != manifest["observed_regular_files_including_manifest"]
            or manifest["observed_archive_entries_including_manifest"]
            != manifest["observed_directories"]
            + manifest["observed_regular_files_including_manifest"]
            or manifest["regular_file_count_excluding_manifest"]
            > plan.execution.maximum_archive_regular_files_excluding_manifest
            or manifest["directory_count_excluding_root"]
            > plan.execution.maximum_archive_directories_excluding_root
            or manifest["manifest_bytes"] > plan.execution.maximum_manifest_bytes
            or manifest["total_regular_file_bytes_including_manifest"]
            > plan.safety.maximum_archive_bytes
            or manifest["identity_claim"]
            != "content_hashes_are_not_authenticity_or_authorship_proof"
            or manifest["local_strict_identity_fields"] != ["mode", "mtime_ns"]
            or manifest["portable_content_identity_fields"]
            != ["relative_path", "size", "sha256"]
        ):
            raise ControllerError(
                "publication_manifest_binding", "manifest receipt identity differs"
            )
        manifest_safety = value["manifest_safety"]
        if not isinstance(manifest_safety, Mapping) or set(manifest_safety) != {
            "post_write",
            "prospective_maximum_manifest_bytes",
        }:
            raise ControllerError(
                "publication_safety_shape", "manifest safety receipt members differ"
            )
        global_timeline = _validate_global_safety_timeline_receipt(
            value["global_safety_timeline"], "publication receipt", plan
        )
        marker_value = {
            "activation_condition": PUBLICATION_LINEARIZATION,
            "archive_path": str(workspace.archive_destination),
            "global_safety_timeline": dict(global_timeline),
            "manifest_sha256": manifest["manifest_sha256"],
            "plan_sha256": plan.sha256,
            "procedure_sha256": PROCEDURE_DESCRIPTOR_SHA256,
            "publication_phase": "verified",
            "run_id": workspace.run_id,
            "schema": PUBLICATION_SUCCESS_SCHEMA,
            "supported_signal_order_cutoff": PUBLICATION_SIGNAL_CUTOFF,
            "terminal_axes": terminal_candidate.as_dict(),
        }
        marker_bytes = canonical_json_bytes(marker_value)
        marker_receipt = value["publication_success_marker"]
        if (
            type(marker_receipt) is not dict
            or set(marker_receipt) != {"bytes", "path", "sha256", "value"}
            or type(marker_receipt["bytes"]) is not int
            or type(marker_receipt["path"]) is not str
            or type(marker_receipt["sha256"]) is not str
            or type(marker_receipt["value"]) is not dict
        ):
            raise ControllerError(
                "publication_success_marker_binding",
                "prepared success-marker receipt differs",
            )
        try:
            observed_marker_bytes = canonical_json_bytes(marker_receipt["value"])
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:
            raise ControllerError(
                "publication_success_marker_binding",
                "prepared success-marker value is not canonical JSON",
            ) from error
        if (
            marker_receipt["bytes"] != len(marker_bytes)
            or marker_receipt["bytes"]
            > plan.execution.maximum_json_evidence_bytes
            or marker_receipt["path"]
            != str(workspace.scratch / "publication-success.json")
            or marker_receipt["sha256"]
            != hashlib.sha256(marker_bytes).hexdigest()
            or observed_marker_bytes != marker_bytes
            or marker_receipt["value"] != marker_value
        ):
            raise ControllerError(
                "publication_success_marker_binding",
                "prepared success-marker receipt differs",
            )
        expected_sensor_paths = tuple(
            sensor.path for sensor in plan.safety.mandatory_thermal_sensors
        )
        safety_records = (
            (
                "prospective_maximum_manifest_bytes",
                manifest_safety["prospective_maximum_manifest_bytes"],
            ),
            ("post_write", manifest_safety["post_write"]),
            (
                "publication_success_marker_precommit_safety",
                value["publication_success_marker_precommit_safety"],
            ),
        )
        for role, sample in safety_records:
            if (
                not isinstance(sample, Mapping)
                or set(sample)
                != {"free_bytes", "monotonic_ns", "thermal_millicelsius"}
                or type(sample["free_bytes"]) is not int
                or sample["free_bytes"]
                < plan.safety.minimum_free_during_bytes
                or type(sample["monotonic_ns"]) is not int
                or sample["monotonic_ns"] < 0
                or not isinstance(sample["thermal_millicelsius"], list)
            ):
                raise ControllerError("publication_safety_record", role)
            thermal = sample["thermal_millicelsius"]
            if (
                tuple(
                    item.get("path") if isinstance(item, Mapping) else None
                    for item in thermal
                )
                != expected_sensor_paths
                or any(
                    not isinstance(item, Mapping)
                    or set(item) != {"path", "value"}
                    or type(item["value"]) is not int
                    or not (
                        plan.safety.mandatory_thermal_sensors[index]
                        .minimum_plausible_millicelsius
                        <= item["value"]
                        < plan.safety.thermal_ceiling_millicelsius
                    )
                    for index, item in enumerate(thermal)
                )
            ):
                raise ControllerError("publication_safety_sensor", role)
        safety_times = tuple(
            sample["monotonic_ns"] for _role, sample in safety_records
        )
        minimum_publication_sample_operations = 1 + len(safety_records)
        if (
            global_timeline["sample_count"]
            < minimum_publication_sample_operations
            or not (
                global_timeline["first_monotonic_ns"]
                <= safety_times[0]
                <= safety_times[1]
                <= safety_times[2]
                == global_timeline["last_monotonic_ns"]
            )
        ):
            raise ControllerError(
                "publication_safety_timeline",
                "named safety samples are not bound to the global timeline",
            )
        return value

    def run(self, plan: RunPlan) -> Mapping[str, object]:
        verify_procedure_descriptor()
        validate_host_binding(plan)
        workspace = self.services.create_workspace(plan, procedure_descriptor())
        state: dict[str, object] = {
            "archive_integrity": "UNVERIFIED",
            "conditions": {name: "NOT_STARTED" for name, _, _ in R0_CONDITIONS},
            "expected_commit": plan.expected_commit,
            "inclusion": "EXCLUDED",
            "interrupted": False,
            "plan_sha256": plan.sha256,
            "phase": "INITIALIZED",
            "profiler_stream_state": "NOT_APPLICABLE_R0",
            "production_final_elf_absence": PRODUCTION_FINAL_ELF_ABSENCE,
            "run_id": workspace.run_id,
            "schema": TERMINAL_RESULT_SCHEMA,
            "scientific_case_state": "INCOMPLETE",
        }
        self.services.checkpoint(workspace, state)
        absence_decisions: list[AbsenceDecision] = []
        condition_complete = {name: False for name, _, _ in R0_CONDITIONS}
        positive_complete = False
        failure_reasons: list[str] = []
        cancellation: BaseException | None = None
        primary_failure: BaseException | None = None
        try:
            self._write_platform_provenance(plan, workspace)
            tool_bindings = self._bind_all_tools(plan, workspace)
            common_env = build_clean_environment(plan, workspace, "OFF0")
            self_test = self._run(
                workspace,
                plan,
                "controller_selftest",
                "controller-selftest",
                (
                    plan.tools["python"],
                    "-B",
                    "-m",
                    "tools.profiling.memory.oai_memprof_r0_terminal",
                    "self-test",
                ),
                pathlib.Path(plan.source_repository),
                common_env,
            )
            validate_controller_selftest_output(
                self.services.command_stdout(
                    self_test,
                    plan.execution.step_limits["controller_selftest"].stdout_bytes,
                ),
                self.services.command_stderr(
                    self_test,
                    plan.execution.step_limits["controller_selftest"].stderr_bytes,
                ),
            )

            clone = self._run(
                workspace,
                plan,
                "git_clone",
                "source-clone",
                (
                    plan.tools["git"],
                    "clone",
                    "--local",
                    "--no-hardlinks",
                    "--no-checkout",
                    plan.source_repository,
                    str(workspace.source),
                ),
                workspace.scratch,
                common_env,
            )
            del clone
            self._run(
                workspace,
                plan,
                "git_checkout",
                "source-checkout",
                (plan.tools["git"], "-C", str(workspace.source), "checkout", "--detach", plan.expected_commit),
                workspace.scratch,
                common_env,
            )
            self._validate_source_binding(plan, workspace, common_env)
            ld_result = self._run(
                workspace,
                plan,
                "tool_probe",
                "tool-ld-version",
                (plan.tools["ld"], "--version"),
                workspace.scratch,
                common_env,
            )
            ld_output = self.services.command_stdout(
                ld_result, plan.execution.step_limits["tool_probe"].stdout_bytes
            )
            if self.services.command_stderr(
                ld_result, plan.execution.step_limits["tool_probe"].stderr_bytes
            ) != b"":
                raise GateError("ld_version_stderr", "linker version probe emitted stderr")
            self._probe_all_tools(
                plan, workspace, common_env, tool_bindings
            )
            state["phase"] = "SOURCE_BOUND"
            self.services.checkpoint(workspace, state)

            for condition, tests_enabled, _ in R0_CONDITIONS:
                del tests_enabled
                state["phase"] = f"{condition}_RUNNING"
                state["conditions"][condition] = "RUNNING"  # type: ignore[index]
                self.services.checkpoint(workspace, state)
                environment = build_clean_environment(plan, workspace, condition)
                self._prepare_dependencies(plan, workspace, condition, environment)
                self._run(
                    workspace,
                    plan,
                    "configure",
                    f"{condition.lower()}-configure",
                    configure_argv(plan, workspace, condition),
                    workspace.scratch,
                    environment,
                )
                self._validate_cmake_cache(plan, workspace, condition)
                provisional_absence: AbsenceDecision | None = None
                provisional_positive = False
                if condition in {"OFF0", "OFF1"}:
                    provisional_absence = self._run_absence_condition(
                        plan, workspace, condition, environment, ld_output
                    )
                else:
                    self._run_on_condition(plan, workspace, environment)
                    provisional_positive = True
                self._validate_dependency_postcondition(
                    plan, workspace, condition, environment
                )
                if provisional_absence is not None:
                    absence_decisions.append(provisional_absence)
                if provisional_positive:
                    positive_complete = True
                condition_complete[condition] = True
                state["conditions"][condition] = "COMPLETE"  # type: ignore[index]
                state["phase"] = f"{condition}_COMPLETE"
                self.services.checkpoint(workspace, state)
            self._validate_source_postcondition(plan, workspace, common_env)
            self._validate_tool_postcondition(plan, workspace, tool_bindings)
            state["phase"] = "VALIDATION_COMPLETE"
            self.services.checkpoint(workspace, state)
        except (KeyboardInterrupt, SystemExit) as error:
            cancellation = error
            primary_failure = error
            state["interrupted"] = True
            state["phase"] = "INTERRUPTED"
            failure_reasons.append("INTERRUPTED")
            try:
                self.services.checkpoint(workspace, state)
            except (KeyboardInterrupt, SystemExit) as checkpoint_error:
                error.add_note(
                    "repeated cancellation during terminal interruption checkpoint: "
                    f"{type(checkpoint_error).__name__}: "
                    f"{_bounded_text(checkpoint_error)}"
                )
                raise error
            except BaseException as checkpoint_error:
                error.add_note(
                    "terminal interruption checkpoint failed; attempting "
                    "failure-only sealing: "
                    f"{type(checkpoint_error).__name__}: "
                    f"{_bounded_text(checkpoint_error)}"
                )
                failure_reasons.append("INTERRUPTION_CHECKPOINT_FAILED")
                checkpoint_code = getattr(checkpoint_error, "code", None)
                checkpoint_code = getattr(checkpoint_code, "value", checkpoint_code)
                state["failure_checkpoint_problem"] = {
                    "code": (
                        checkpoint_code
                        if isinstance(checkpoint_code, str) and checkpoint_code
                        else "checkpoint_failure"
                    ),
                    "detail": _bounded_text(checkpoint_error),
                    "requested_phase": "INTERRUPTED",
                }
        except (ControllerError, GateError, OSError) as error:
            primary_failure = error
            state["phase"] = "FAILED"
            failure_reasons.append(getattr(error, "code", "CONTROLLER_FAILURE").upper())
            state["failure"] = {
                "code": getattr(error, "code", "controller_failure"),
                "detail": _bounded_text(error),
            }
            try:
                self.services.checkpoint(workspace, state)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as checkpoint_error:
                error.add_note(
                    "terminal failure checkpoint failed; attempting "
                    "failure-only sealing: "
                    f"{type(checkpoint_error).__name__}: "
                    f"{_bounded_text(checkpoint_error)}"
                )
                failure_reasons.append("FAILURE_CHECKPOINT_FAILED")
                checkpoint_code = getattr(checkpoint_error, "code", None)
                checkpoint_code = getattr(checkpoint_code, "value", checkpoint_code)
                state["failure_checkpoint_problem"] = {
                    "code": (
                        checkpoint_code
                        if isinstance(checkpoint_code, str) and checkpoint_code
                        else "checkpoint_failure"
                    ),
                    "detail": _bounded_text(checkpoint_error),
                    "requested_phase": "FAILED",
                }
        except Exception as error:
            primary_failure = error
            state["phase"] = "FAILED"
            failure_reasons.append("INTERNAL_CONTROLLER_FAILURE")
            state["failure"] = {
                "code": "internal_controller_failure",
                "detail": f"{type(error).__name__}: {_bounded_text(error)}",
            }
            try:
                self.services.checkpoint(workspace, state)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as checkpoint_error:
                error.add_note(
                    "internal failure checkpoint failed; attempting "
                    "failure-only sealing: "
                    f"{type(checkpoint_error).__name__}: "
                    f"{_bounded_text(checkpoint_error)}"
                )
                failure_reasons.append("FAILURE_CHECKPOINT_FAILED")
                checkpoint_code = getattr(checkpoint_error, "code", None)
                checkpoint_code = getattr(checkpoint_code, "value", checkpoint_code)
                state["failure_checkpoint_problem"] = {
                    "code": (
                        checkpoint_code
                        if isinstance(checkpoint_code, str) and checkpoint_code
                        else "checkpoint_failure"
                    ),
                    "detail": _bounded_text(checkpoint_error),
                    "requested_phase": "FAILED",
                }
        try:
            result = self._seal_and_publish(
                plan=plan,
                workspace=workspace,
                state=state,
                condition_complete=condition_complete,
                absence_decisions=absence_decisions,
                positive_complete=positive_complete,
                failure_reasons=failure_reasons,
            )
        except BaseException as sealing_error:
            if cancellation is not None:
                cancellation.add_note(
                    "terminal interruption sealing failed: "
                    f"{type(sealing_error).__name__}: "
                    f"{_bounded_text(sealing_error)}"
                )
                raise cancellation
            if primary_failure is not None:
                primary_failure.add_note(
                    "terminal failure sealing failed: "
                    f"{type(sealing_error).__name__}: "
                    f"{_bounded_text(sealing_error)}"
                )
                raise primary_failure
            raise
        if cancellation is not None:
            raise cancellation
        return result

    def _seal_and_publish(
        self,
        *,
        plan: RunPlan,
        workspace: RunWorkspace,
        state: dict[str, object],
        condition_complete: Mapping[str, bool],
        absence_decisions: Sequence[AbsenceDecision],
        positive_complete: bool,
        failure_reasons: Sequence[str],
    ) -> Mapping[str, object]:
        """Seal one terminal candidate; interruption ownership stays in ``run``."""

        terminal_candidate = classify_terminal_result(
            condition_complete=condition_complete,
            archive_verified=True,
            interrupted=bool(state["interrupted"]),
            failure_reasons=failure_reasons,
            absence_decisions=absence_decisions,
            positive_elf_complete=positive_complete,
        )
        state["phase"] = "READY_TO_SEAL"
        state["terminal_candidate"] = {
            "activation_condition": (
                PUBLICATION_ACTIVATION
            ),
            "axes": terminal_candidate.as_dict(),
        }
        self.services.checkpoint(workspace, state)
        publication_succeeded = True
        try:
            publication = self.services.publish(
                workspace,
                terminal_candidate,
                plan.safety.maximum_archive_bytes,
                lambda value: self._validate_publication_receipt(
                    value,
                    workspace,
                    terminal_candidate,
                    plan,
                ),
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            publication_succeeded = False
            path_observation = self._publication_path_observation(workspace)
            active_state = path_observation["active"]
            final_state = path_observation["final"]
            active_directory = (
                isinstance(active_state, dict)
                and active_state.get("state") == "DIRECTORY"
            )
            final_directory = (
                isinstance(final_state, dict)
                and final_state.get("state") == "DIRECTORY"
            )
            if final_directory and not active_directory:
                observed_archive_path: str | None = str(
                    workspace.archive_destination
                )
            elif active_directory and not final_directory:
                observed_archive_path = str(workspace.archive_stage)
            else:
                observed_archive_path = None
            raw_code = getattr(error, "code", None)
            normalized_code = getattr(raw_code, "value", raw_code)
            if not isinstance(normalized_code, str) or not normalized_code:
                normalized_code = "publication_failure"
            raw_phase = getattr(error, "publication_phase", None)
            normalized_phase = getattr(raw_phase, "value", raw_phase)
            publication = {
                "archive_integrity": "UNVERIFIED",
                "archive_path": observed_archive_path,
                "mutated": final_directory,
                "problem": {
                    "code": normalized_code,
                    "message": _bounded_text(error),
                },
                "publication_path_observation": path_observation,
                "publication_phase": (
                    normalized_phase if isinstance(normalized_phase, str) else None
                ),
                "schema": "oai.memprof.r0.publication-receipt/v1",
                "verified": False,
            }
        if publication_succeeded:
            final_axes = terminal_candidate
        else:
            final_axes = classify_terminal_result(
                condition_complete=condition_complete,
                archive_verified=False,
                interrupted=bool(state["interrupted"]),
                failure_reasons=failure_reasons,
                absence_decisions=absence_decisions,
                positive_elf_complete=positive_complete,
            )
        return {
            "axes": final_axes.as_dict(),
            "plan_sha256": plan.sha256,
            "procedure_sha256": PROCEDURE_DESCRIPTOR_SHA256,
            "publication": publication,
            "run_id": workspace.run_id,
            "schema": TERMINAL_RESULT_SCHEMA,
        }

    def _observe_tool_binding(
        self, plan: RunPlan, workspace: RunWorkspace, name: str
    ) -> ToolBinding:
        declared = pathlib.Path(plan.tools[name])
        try:
            declared_before = os.lstat(declared)
            resolved = declared.resolve(strict=True)
            resolved_before = os.stat(resolved, follow_symlinks=False)
        except OSError as error:
            raise GateError("tool_identity_stat", f"{name}: {error}") from error
        if not (
            stat.S_ISREG(declared_before.st_mode)
            or stat.S_ISLNK(declared_before.st_mode)
        ):
            raise GateError(
                "tool_declared_type", f"{name}: declared path is not regular or symlink"
            )
        if not stat.S_ISREG(resolved_before.st_mode) or not os.access(
            resolved, os.X_OK
        ):
            raise GateError(
                "tool_identity_type",
                f"{name}: resolved tool is not an executable regular file",
            )
        tool_image = self.services.read_regular(
            workspace, resolved, plan.execution.maximum_evidence_bytes
        )
        try:
            declared_after = os.lstat(declared)
            resolved_after_path = declared.resolve(strict=True)
            resolved_after = os.stat(resolved_after_path, follow_symlinks=False)
        except OSError as error:
            raise GateError("tool_identity_restat", f"{name}: {error}") from error
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if (
            resolved_after_path != resolved
            or any(
                getattr(declared_before, field) != getattr(declared_after, field)
                for field in stable_fields
            )
            or any(
                getattr(resolved_before, field) != getattr(resolved_after, field)
                for field in stable_fields
            )
            or tool_image.size != resolved_after.st_size
            or tool_image.device != resolved_after.st_dev
            or tool_image.inode != resolved_after.st_ino
            or tool_image.mode != resolved_after.st_mode
            or tool_image.mtime_ns != resolved_after.st_mtime_ns
            or tool_image.ctime_ns != resolved_after.st_ctime_ns
        ):
            raise GateError("tool_identity_changed", name)
        return ToolBinding(
            declared_path=str(declared),
            declared_device=declared_after.st_dev,
            declared_inode=declared_after.st_ino,
            declared_mode=declared_after.st_mode,
            declared_nlink=declared_after.st_nlink,
            declared_size=declared_after.st_size,
            declared_mtime_ns=declared_after.st_mtime_ns,
            declared_ctime_ns=declared_after.st_ctime_ns,
            resolved_path=str(resolved_after_path),
            resolved_device=resolved_after.st_dev,
            resolved_inode=resolved_after.st_ino,
            resolved_mode=resolved_after.st_mode,
            resolved_nlink=resolved_after.st_nlink,
            resolved_size=resolved_after.st_size,
            resolved_mtime_ns=resolved_after.st_mtime_ns,
            resolved_ctime_ns=resolved_after.st_ctime_ns,
            sha256=tool_image.sha256,
        )

    @staticmethod
    def _tool_binding_record(
        name: str, binding: ToolBinding, *, version_ledger: str | None
    ) -> dict[str, object]:
        return {
            **dataclasses.asdict(binding),
            "name": name,
            "version_ledger": version_ledger,
        }

    def _bind_all_tools(
        self, plan: RunPlan, workspace: RunWorkspace
    ) -> Mapping[str, ToolBinding]:
        bindings: dict[str, ToolBinding] = {}
        for name in REQUIRED_TOOL_NAMES:
            bindings[name] = self._observe_tool_binding(plan, workspace, name)
        if len({item.declared_path for item in bindings.values()}) != len(
            bindings
        ):
            raise GateError("tool_declared_alias", "declared tool paths alias")
        self.services.write_json(
            workspace,
            "provenance/tools-preflight.json",
            {
                "binding_phase": "before_first_external_command",
                "schema": "oai.memprof.r0.tool-preflight/v1",
                "tools": [
                    self._tool_binding_record(name, bindings[name], version_ledger=None)
                    for name in REQUIRED_TOOL_NAMES
                ],
            },
        )
        return types.MappingProxyType(bindings)

    def _probe_all_tools(
        self,
        plan: RunPlan,
        workspace: RunWorkspace,
        environment: Mapping[str, str],
        bindings: Mapping[str, ToolBinding],
    ) -> None:
        if set(bindings) != set(REQUIRED_TOOL_NAMES):
            raise GateError("tool_probe_population", "preflight tool population changed")
        provenance: list[dict[str, object]] = []
        for name in REQUIRED_TOOL_NAMES:
            if name == "ld":
                # The separately retained ld probe owns its version and
                # reference-classification bytes; file identity is still
                # included below with every other declared tool.
                result = None
            else:
                result = self._run(
                    workspace,
                    plan,
                    "tool_probe",
                    f"tool-{name}-version",
                    (plan.tools[name], "--version"),
                    workspace.scratch,
                    environment,
                )
                output = self.services.command_stdout(
                    result, plan.execution.step_limits["tool_probe"].stdout_bytes
                )
                if not output:
                    raise GateError("tool_version_empty", name)
            observed = self._observe_tool_binding(plan, workspace, name)
            if observed != bindings[name]:
                raise GateError("tool_probe_identity", name)
            provenance.append(
                self._tool_binding_record(
                    name,
                    observed,
                    version_ledger=(
                        "tool-ld-version"
                        if name == "ld"
                        else f"tool-{name}-version"
                    ),
                )
            )
        self.services.write_json(
            workspace,
            "provenance/tools.json",
            {"schema": "oai.memprof.r0.tool-provenance/v1", "tools": provenance},
        )

    def _validate_tool_postcondition(
        self,
        plan: RunPlan,
        workspace: RunWorkspace,
        bindings: Mapping[str, ToolBinding],
    ) -> None:
        if set(bindings) != set(REQUIRED_TOOL_NAMES):
            raise GateError("tool_post_population", "tool binding population changed")
        for name in REQUIRED_TOOL_NAMES:
            if self._observe_tool_binding(plan, workspace, name) != bindings[name]:
                raise GateError("tool_post_identity", name)

    def _validate_source_binding(
        self, plan: RunPlan, workspace: RunWorkspace, environment: Mapping[str, str]
    ) -> None:
        for label, repository in (("invoker", pathlib.Path(plan.source_repository)), ("clone", workspace.source)):
            head = self._run(
                workspace,
                plan,
                "tool_probe",
                f"source-{label}-head",
                (plan.tools["git"], "-C", str(repository), "rev-parse", "--verify", "HEAD^{commit}"),
                workspace.scratch,
                environment,
            )
            observed = _single_ascii_line(
                self.services.command_stdout(
                    head, plan.execution.step_limits["tool_probe"].stdout_bytes
                ),
                f"source_{label}_head",
            )
            if observed != plan.expected_commit:
                raise GateError("source_commit_mismatch", f"{label}: {observed}")
            status_result = self._run(
                workspace,
                plan,
                "tool_probe",
                f"source-{label}-status",
                (
                    plan.tools["git"], "-C", str(repository), "status", "--porcelain=v1",
                    "--untracked-files=all" if label == "clone" else "--untracked-files=no",
                ),
                workspace.scratch,
                environment,
            )
            if self.services.command_stdout(
                status_result, plan.execution.step_limits["tool_probe"].stdout_bytes
            ) != b"":
                raise GateError("source_tracked_dirty", label)
        submodule = self._run(
            workspace,
            plan,
            "tool_probe",
            "source-submodule-status",
            (plan.tools["git"], "-C", str(workspace.source), "submodule", "status"),
            workspace.scratch,
            environment,
        )
        submodule_text = _decode_bounded_utf8(
            self.services.command_stdout(submodule, plan.execution.step_limits["tool_probe"].stdout_bytes),
            "submodule_status",
        )
        lines = submodule_text.splitlines()
        if len(lines) != 1 or not re.fullmatch(
            r"-[0-9a-f]{40} openair2/E2AP/flexric(?: \([^\r\n]{1,256}\))?", lines[0]
        ):
            raise GateError("submodule_state", "flexric must be the sole uninitialized gitlink")
        controller_paths = {
            "tools/profiling/memory/oai_memprof_r0_terminal.py": pathlib.Path(__file__).resolve(),
            "tools/profiling/memory/oai_memprof_r0_terminal_process.py": pathlib.Path(
                importlib.import_module(
                    "tools.profiling.memory.oai_memprof_r0_terminal_process"
                ).__file__
            ).resolve(),
            "tools/profiling/memory/oai_memprof_r0_terminal_archive.py": pathlib.Path(
                _archive_module().__file__
            ).resolve(),
        }
        for relative, running_path in controller_paths.items():
            committed = self._run(
                workspace,
                plan,
                "tool_probe",
                "controller-committed-" + pathlib.PurePosixPath(relative).stem,
                (
                    plan.tools["git"],
                    "-C",
                    plan.source_repository,
                    "rev-parse",
                    f"{plan.expected_commit}:{relative}",
                ),
                workspace.scratch,
                environment,
            )
            running = self._run(
                workspace,
                plan,
                "tool_probe",
                "controller-running-" + pathlib.PurePosixPath(relative).stem,
                (plan.tools["git"], "hash-object", "--no-filters", str(running_path)),
                workspace.scratch,
                environment,
            )
            committed_hash = _single_ascii_line(
                self.services.command_stdout(
                    committed, plan.execution.step_limits["tool_probe"].stdout_bytes
                ),
                "committed_controller_blob",
            )
            running_hash = _single_ascii_line(
                self.services.command_stdout(
                    running, plan.execution.step_limits["tool_probe"].stdout_bytes
                ),
                "running_controller_blob",
            )
            if not _FULL_SHA1.fullmatch(committed_hash) or running_hash != committed_hash:
                raise GateError(
                    "controller_blob_mismatch",
                    f"{relative}: committed={committed_hash!r} running={running_hash!r}",
                )

    def _prepare_dependencies(
        self,
        plan: RunPlan,
        workspace: RunWorkspace,
        condition: str,
        environment: Mapping[str, str],
    ) -> None:
        root = workspace.conditions[condition].dependency_root
        for name, dependency in (
            ("googletest", plan.dependencies.googletest),
            ("benchmark", plan.dependencies.benchmark),
        ):
            destination = root / name
            self._run(
                workspace,
                plan,
                "git_clone",
                f"{condition.lower()}-{name}-clone",
                (
                    plan.tools["git"],
                    "clone",
                    "--local",
                    "--no-hardlinks",
                    "--no-checkout",
                    dependency.path,
                    str(destination),
                ),
                workspace.scratch,
                environment,
            )
            self._run(
                workspace,
                plan,
                "git_checkout",
                f"{condition.lower()}-{name}-checkout",
                (
                    plan.tools["git"],
                    "-C",
                    str(destination),
                    "checkout",
                    "--detach",
                    dependency.commit,
                ),
                workspace.scratch,
                environment,
            )
            head = self._run(
                workspace,
                plan,
                "tool_probe",
                f"{condition.lower()}-{name}-head",
                (plan.tools["git"], "-C", str(destination), "rev-parse", "--verify", "HEAD^{commit}"),
                workspace.scratch,
                environment,
            )
            observed = _single_ascii_line(
                self.services.command_stdout(
                    head, plan.execution.step_limits["tool_probe"].stdout_bytes
                ),
                f"{condition}_{name}_head",
            )
            if observed != dependency.commit:
                raise GateError("dependency_commit_mismatch", f"{condition}/{name}: {observed}")
            clean = self._run(
                workspace,
                plan,
                "tool_probe",
                f"{condition.lower()}-{name}-status",
                (
                    plan.tools["git"],
                    "-C",
                    str(destination),
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=no",
                ),
                workspace.scratch,
                environment,
            )
            if self.services.command_stdout(
                clean, plan.execution.step_limits["tool_probe"].stdout_bytes
            ) != b"":
                raise GateError("dependency_tracked_dirty", f"{condition}/{name}")
        cpm_destination = root / "cpm-cache/cpm/CPM_0.40.1.cmake"
        identity = self.services.copy_scratch_regular(
            workspace,
            pathlib.Path(plan.dependencies.cpm_cmake.path),
            cpm_destination,
            plan.execution.maximum_evidence_bytes,
        )
        if identity.sha256 != plan.dependencies.cpm_cmake.sha256:
            raise GateError("cpm_identity", f"{condition}: {identity.sha256}")

    def _validate_source_postcondition(
        self,
        plan: RunPlan,
        workspace: RunWorkspace,
        environment: Mapping[str, str],
    ) -> None:
        for label, repository in (
            ("invoker", pathlib.Path(plan.source_repository)),
            ("clone", workspace.source),
        ):
            head_result = self._run(
                workspace,
                plan,
                "tool_probe",
                f"source-{label}-post-head",
                (
                    plan.tools["git"],
                    "-C",
                    str(repository),
                    "rev-parse",
                    "--verify",
                    "HEAD^{commit}",
                ),
                workspace.scratch,
                environment,
            )
            observed_head = _single_ascii_line(
                self.services.command_stdout(
                    head_result,
                    plan.execution.step_limits["tool_probe"].stdout_bytes,
                ),
                f"source_{label}_post_head",
            )
            if observed_head != plan.expected_commit:
                raise GateError(
                    "source_post_commit",
                    f"{label}: expected {plan.expected_commit}, observed {observed_head}",
                )
            result = self._run(
                workspace,
                plan,
                "tool_probe",
                f"source-{label}-post-status",
                (
                    plan.tools["git"],
                    "-C",
                    str(repository),
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all"
                    if label == "clone"
                    else "--untracked-files=no",
                ),
                workspace.scratch,
                environment,
            )
            if self.services.command_stdout(
                result, plan.execution.step_limits["tool_probe"].stdout_bytes
            ) != b"":
                raise GateError("source_post_tracked_dirty", label)

    def _validate_dependency_postcondition(
        self,
        plan: RunPlan,
        workspace: RunWorkspace,
        condition: str,
        environment: Mapping[str, str],
    ) -> None:
        root = workspace.conditions[condition].dependency_root
        cpm_path = root / "cpm-cache/cpm/CPM_0.40.1.cmake"
        cpm_image = self.services.read_regular(
            workspace, cpm_path, plan.execution.maximum_evidence_bytes
        )
        if cpm_image.sha256 != plan.dependencies.cpm_cmake.sha256:
            raise GateError("cpm_post_identity", condition)
        for name, dependency in (
            ("googletest", plan.dependencies.googletest),
            ("benchmark", plan.dependencies.benchmark),
        ):
            repository = root / name
            head = self._run(
                workspace,
                plan,
                "tool_probe",
                f"{condition.lower()}-{name}-post-head",
                (
                    plan.tools["git"],
                    "-C",
                    str(repository),
                    "rev-parse",
                    "--verify",
                    "HEAD^{commit}",
                ),
                workspace.scratch,
                environment,
            )
            if _single_ascii_line(
                self.services.command_stdout(
                    head, plan.execution.step_limits["tool_probe"].stdout_bytes
                ),
                f"{condition}_{name}_post_head",
            ) != dependency.commit:
                raise GateError("dependency_post_commit", f"{condition}/{name}")
            status_result = self._run(
                workspace,
                plan,
                "tool_probe",
                f"{condition.lower()}-{name}-post-status",
                (
                    plan.tools["git"],
                    "-C",
                    str(repository),
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                ),
                workspace.scratch,
                environment,
            )
            if self.services.command_stdout(
                status_result, plan.execution.step_limits["tool_probe"].stdout_bytes
            ) != b"":
                raise GateError("dependency_post_dirty", f"{condition}/{name}")

    def _validate_cmake_cache(self, plan: RunPlan, workspace: RunWorkspace, condition: str) -> None:
        path = workspace.conditions[condition].build / "CMakeCache.txt"
        image = self.services.read_regular(
            workspace, path, plan.execution.maximum_evidence_bytes
        )
        progress = lambda: self.services.progress(workspace)
        expected_tests, expected_physim = {
            name: (tests, physim) for name, tests, physim in R0_CONDITIONS
        }[condition]
        required = {
            "CCACHE_ACTIVE": "OFF",
            "CMAKE_BUILD_TYPE": "RelWithDebInfo",
            "CMAKE_C_COMPILER": plan.tools["cc"],
            "CMAKE_C_COMPILER_LAUNCHER": "",
            "CMAKE_CXX_COMPILER": plan.tools["cxx"],
            "CMAKE_CXX_COMPILER_LAUNCHER": "",
            "CMAKE_DISABLE_FIND_PACKAGE_GTest": "TRUE",
            "CMAKE_DISABLE_FIND_PACKAGE_benchmark": "TRUE",
            "CMAKE_EXPORT_NO_PACKAGE_REGISTRY": "ON",
            "CMAKE_FIND_USE_PACKAGE_REGISTRY": "FALSE",
            "CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY": "FALSE",
            "CMAKE_MAKE_PROGRAM": plan.tools["ninja"],
            "CPM_SOURCE_CACHE": str(
                workspace.conditions[condition].dependency_root / "cpm-cache"
            ),
            "ENABLE_PHYSIM_TESTS": "ON" if expected_physim else "OFF",
            "ENABLE_TESTS": "ON" if expected_tests else "OFF",
            "FETCHCONTENT_FULLY_DISCONNECTED": "ON",
            "FETCHCONTENT_SOURCE_DIR_BENCHMARK": str(
                workspace.conditions[condition].dependency_root / "benchmark"
            ),
            "FETCHCONTENT_SOURCE_DIR_GOOGLETEST": str(
                workspace.conditions[condition].dependency_root / "googletest"
            ),
            "FETCHCONTENT_UPDATES_DISCONNECTED": "ON",
            **{name: "OFF" for name in OPTIONAL_CMAKE_FEATURES_DISABLED},
        }
        internal_required = {
            "CMAKE_CACHEFILE_DIR": str(workspace.conditions[condition].build),
            "CMAKE_GENERATOR": "Ninja",
            "CMAKE_HOME_DIRECTORY": str(workspace.source),
        }
        observed: dict[str, list[str]] = {}
        for line in _iter_bounded_utf8_lines(
            image.data,
            "CMakeCache",
            progress_callback=progress,
        ):
            progress()
            if line.startswith("//") or line.startswith("#") or ":" not in line or "=" not in line:
                continue
            name_type, value = line.split("=", 1)
            name = name_type.split(":", 1)[0]
            if name in required or name in internal_required:
                observed.setdefault(name, []).append(value)
        for name, expected in required.items():
            if observed.get(name) != [expected]:
                raise GateError("cmake_cache_binding", f"{condition} {name}: {observed.get(name)!r}")
        for name, expected in internal_required.items():
            if observed.get(name) != [expected]:
                raise GateError(
                    "cmake_internal_binding",
                    f"{condition} {name}: {observed.get(name)!r}",
                )

    def _run_absence_condition(
        self,
        plan: RunPlan,
        workspace: RunWorkspace,
        condition: str,
        environment: Mapping[str, str],
        ld_output: bytes,
    ) -> AbsenceDecision:
        paths = workspace.conditions[condition]
        target_result = self._run(
            workspace, plan, "ninja_targets", f"{condition.lower()}-targets",
            (plan.tools["ninja"], "-t", "targets", "all"), paths.build, environment,
        )
        command_result = self._run(
            workspace, plan, "ninja_commands", f"{condition.lower()}-commands",
            (plan.tools["ninja"], "-t", "commands", *ABSENCE_EXPECTED_FINAL_TARGETS),
            paths.build, environment,
        )
        build_identity = self.services.capture_regular(
            workspace,
            paths.build / "build.ninja",
            f"conditions/{condition}/absence/build.ninja",
            plan.execution.maximum_evidence_bytes,
        )
        targets_identity = self.services.capture_regular(
            workspace,
            pathlib.Path(
                str(
                    self.services.command_record(target_result)[
                        "stdout_capture_path_during_acquisition"
                    ]
                )
            ),
            f"conditions/{condition}/absence/targets.txt",
            plan.execution.maximum_evidence_bytes,
        )
        commands_identity = self.services.capture_regular(
            workspace,
            pathlib.Path(
                str(
                    self.services.command_record(command_result)[
                        "stdout_capture_path_during_acquisition"
                    ]
                )
            ),
            f"conditions/{condition}/absence/commands.txt",
            plan.execution.maximum_evidence_bytes,
        )
        evidence_paths = {
            "build_ninja": pathlib.Path(build_identity.path),
            "targets": pathlib.Path(targets_identity.path),
            "commands": pathlib.Path(commands_identity.path),
        }
        normalized = tuple(
            dataclasses.replace(identity, role=role)
            for role, identity in (
                ("build_ninja", build_identity),
                ("targets", targets_identity),
                ("commands", commands_identity),
            )
        )
        for identity in normalized:
            captured = pathlib.Path(identity.path)
            try:
                relative = captured.relative_to(workspace.archive_stage)
            except ValueError as error:
                raise GateError(
                    "absence_capture_domain",
                    f"{identity.role}: {captured} is outside archive stage",
                ) from error
            if not captured.is_absolute() or str(captured) != str(workspace.archive_stage / relative):
                raise GateError("absence_capture_path", f"{identity.role}: noncanonical capture path")
        gate = self._run(
            workspace,
            plan,
            "absence_gate",
            f"{condition.lower()}-absence",
            absence_argv(
                plan, workspace, condition, evidence_paths["build_ninja"],
                evidence_paths["targets"], evidence_paths["commands"],
            ),
            paths.build,
            environment,
        )
        stderr = self.services.command_stderr(
            gate, plan.execution.step_limits["absence_gate"].stderr_bytes
        )
        if stderr != b"":
            raise GateError("absence_stderr", condition)
        return validate_absence_result(
            self.services.command_stdout(gate, plan.execution.step_limits["absence_gate"].stdout_bytes),
            source_root=workspace.source,
            build_root=paths.build,
            identities=normalized,
            ld_version_output=ld_output,
            condition=condition,
        )

    def _run_on_condition(
        self,
        plan: RunPlan,
        workspace: RunWorkspace,
        environment: Mapping[str, str],
    ) -> None:
        paths = workspace.conditions["ON"]
        targets = self._run(
            workspace, plan, "ninja_targets", "on-targets",
            (plan.tools["ninja"], "-t", "targets", "all"), paths.build, environment,
        )
        progress = lambda: self.services.progress(workspace)
        target_counts = {target: 0 for target in R0_BUILD_TARGETS}
        for line in _iter_bounded_utf8_lines(
            self.services.command_stdout(
                targets,
                plan.execution.step_limits["ninja_targets"].stdout_bytes,
            ),
            "on_target_inventory",
            progress_callback=progress,
        ):
            progress()
            target = line.split(":", 1)[0]
            if target in target_counts:
                target_counts[target] += 1
        for target in R0_BUILD_TARGETS:
            count = target_counts[target]
            if count != 1:
                raise GateError("on_target_inventory", f"{target}: count={count}")
        self._run(
            workspace, plan, "build", "on-build", build_argv(plan, workspace),
            workspace.scratch, environment,
        )
        for population, expected in (("r0", R0_CTESTS), ("wire", WIRE_CTESTS)):
            inventory = self._run(
                workspace, plan, "ctest_inventory", f"on-{population}-inventory",
                ctest_inventory_argv(plan, workspace, population), workspace.scratch, environment,
            )
            parse_ctest_inventory(
                self.services.command_stdout(
                    inventory, plan.execution.step_limits["ctest_inventory"].stdout_bytes
                ),
                expected,
                progress_callback=progress,
            )
            test_run = self._run(
                workspace, plan, "ctest_run", f"on-{population}-ctest",
                ctest_run_argv(plan, workspace, population), workspace.scratch, environment,
            )
            validate_ctest_run(
                self.services.command_stdout(test_run, plan.execution.step_limits["ctest_run"].stdout_bytes),
                expected,
                progress_callback=progress,
            )
        elf = self._run(
            workspace, plan, "elf_gate", "on-positive-elf", positive_elf_argv(plan, workspace),
            paths.build, environment,
        )
        validate_positive_elf_output(
            self.services.command_stdout(elf, plan.execution.step_limits["elf_gate"].stdout_bytes),
            self.services.command_stderr(elf, plan.execution.step_limits["elf_gate"].stderr_bytes),
            plan.expected_architecture,
        )
        artifact_total = 0
        for relative in R0_ARTIFACTS:
            remaining = plan.safety.maximum_archive_bytes - artifact_total
            if remaining <= 0:
                raise GateError("artifact_total_bytes", "selected artifacts exhaust aggregate archive bound")
            artifact_limit = min(remaining, plan.execution.maximum_evidence_bytes)
            identity = self.services.capture_regular(
                workspace,
                paths.build / relative,
                f"conditions/ON/artifacts/{relative}",
                artifact_limit,
            )
            artifact_total += identity.bytes
            if artifact_total > plan.safety.maximum_archive_bytes:
                raise GateError("artifact_total_bytes", "selected artifacts exceed aggregate archive bound")
            captured = pathlib.Path(identity.path)
            try:
                captured.relative_to(workspace.archive_stage)
            except ValueError as error:
                raise GateError("artifact_capture_domain", relative) from error
            self.services.validate_captured_artifact(
                workspace,
                captured,
                relative,
                identity,
                plan.expected_architecture,
                maximum_bytes=artifact_limit,
                read_chunk_bytes=plan.execution.read_chunk_bytes,
            )

    def resume(self, case_path: os.PathLike[str] | str) -> Mapping[str, object]:
        """Recover archive integrity without changing the preserved science axis."""

        result_value = self.services.recover_incomplete(pathlib.Path(case_path))
        if not isinstance(result_value, Mapping):
            raise ControllerError("recovery_receipt_type", type(result_value).__name__)
        result = dict(result_value)
        action = result.get("recovery_action")
        if result.get("schema") != "oai.memprof.r0.recovery-receipt/v1" or action not in {
            "VERIFY_EXISTING_ORIGINAL",
            "VERIFY_EXISTING_RECOVERED",
            "PRESERVED_INVALID_MANIFEST_ACTIVE",
            "SEALED_AND_PUBLISHED_INCOMPLETE",
        }:
            raise ControllerError("recovery_receipt_schema", action)
        if type(result.get("verified")) is not bool or type(
            result.get("mutated")
        ) is not bool:
            raise ControllerError(
                "recovery_receipt_boolean", "verified/mutated must be exact booleans"
            )
        receipt_archive_path = _canonical_receipt_path(
            result.get("archive_path"), "recovery archive"
        )
        axes = _CoreServices._axes_from_mapping(
            {
                key: result[key]
                for key in (
                    "archive_integrity",
                    "claims",
                    "inclusion",
                    "production_final_elf_absence",
                    "profiler_stream_state",
                    "reasons",
                    "scientific_case_state",
                )
                if key in result
            }
        )
        if (result["verified"] is True) != (
            axes.archive_integrity == "VERIFIED"
        ):
            raise ControllerError(
                "recovery_receipt_integrity", "verified and archive axis differ"
            )
        location = result.get("location_matches_original_publication")
        if "location_matches_original_publication" in result and type(location) is not bool:
            raise ControllerError(
                "recovery_receipt_location", "location flag must be an exact boolean"
            )
        axes_keys = {
            "archive_integrity",
            "claims",
            "inclusion",
            "production_final_elf_absence",
            "profiler_stream_state",
            "reasons",
            "scientific_case_state",
        }
        verification_success_keys = axes_keys | {
            "archive_path",
            "location_matches_original_publication",
            "manifest",
            "mutated",
            "plan_sha256",
            "publication_success_marker_path",
            "publication_success_marker_verified",
            "procedure_sha256",
            "recovery_action",
            "run_id",
            "schema",
            "verified",
            "verification_global_safety_timeline",
        }
        verification_failure_keys = axes_keys | {
            "archive_path",
            "mutated",
            "problem",
            "recovery_action",
            "schema",
            "verified",
            "verification_global_safety_timeline",
        }
        if action in {"VERIFY_EXISTING_ORIGINAL", "VERIFY_EXISTING_RECOVERED"}:
            expected_keys = (
                verification_success_keys
                if result["verified"] is True
                else verification_failure_keys
            )
            if set(result) != expected_keys or result["mutated"] is not False:
                raise ControllerError(
                    "recovery_receipt_shape", f"{action} members or mutation differ"
                )
            if result["verified"] is True and (
                type(result["publication_success_marker_verified"]) is not bool
                or not isinstance(result["publication_success_marker_path"], str)
                or not pathlib.PurePosixPath(
                    result["publication_success_marker_path"]
                ).is_absolute()
                or result["publication_success_marker_path"].startswith("//")
                or str(
                    pathlib.PurePosixPath(
                        result["publication_success_marker_path"]
                    )
                )
                != result["publication_success_marker_path"]
            ):
                raise ControllerError(
                    "recovery_marker_receipt", "marker status or path differs"
                )
            if result["verified"] is True:
                _validate_global_safety_timeline_structure(
                    result["verification_global_safety_timeline"],
                    "verified recovery",
                )
                archive_receipt_path = receipt_archive_path
                run_id = _validate_receipt_run_id(
                    result["run_id"], "verified recovery"
                )
                marker_path = _canonical_receipt_path(
                    result["publication_success_marker_path"],
                    "verified recovery success marker",
                )
                expected_name = (
                    run_id
                    if action == "VERIFY_EXISTING_ORIGINAL"
                    else f"{run_id}.recovered-incomplete"
                )
                if (
                    archive_receipt_path.name != expected_name
                    or marker_path.name != "publication-success.json"
                    or marker_path.parent.name != run_id
                    or result["procedure_sha256"]
                    != PROCEDURE_DESCRIPTOR_SHA256
                    or not isinstance(result["plan_sha256"], str)
                    or _SHA256.fullmatch(result["plan_sha256"]) is None
                ):
                    raise ControllerError(
                        "recovery_receipt_binding",
                        "successful recovery run, path, plan, or procedure identity differs",
                    )
                _validate_manifest_receipt(
                    result["manifest"],
                    expected_manifest_path=archive_receipt_path / "manifest.json",
                )
            if result["verified"] is False:
                problem = result["problem"]
                if (
                    not isinstance(problem, Mapping)
                    or set(problem) != {"code", "message"}
                    or not isinstance(problem["code"], str)
                    or not problem["code"]
                    or not isinstance(problem["message"], str)
                ):
                    raise ControllerError(
                        "recovery_problem_shape", "verification problem differs"
                    )
                verification_timeline = result[
                    "verification_global_safety_timeline"
                ]
                if verification_timeline is not None:
                    _validate_global_safety_timeline_structure(
                        verification_timeline,
                        "failed recovery verification",
                    )
                if (
                    problem["code"] == "maximum_poll_gap"
                    and verification_timeline is None
                ):
                    raise ControllerError(
                        "recovery_problem_timeline",
                        "maximum poll-gap failure omitted its live timeline",
                    )
        elif action == "PRESERVED_INVALID_MANIFEST_ACTIVE":
            if set(result) != axes_keys | {
                "archive_path",
                "mutated",
                "recovery_action",
                "recovery_reason",
                "schema",
                "verified",
            } or result["verified"] is not False or result["mutated"] is not False:
                raise ControllerError(
                    "recovery_receipt_shape", "preserved-invalid receipt differs"
                )
            if (
                not isinstance(result["recovery_reason"], str)
                or not result["recovery_reason"]
                or not receipt_archive_path.name.startswith(".")
                or not receipt_archive_path.name.endswith(".active")
            ):
                raise ControllerError(
                    "recovery_reason_shape", "preserved-invalid reason differs"
                )
        else:
            if set(result) != verification_success_keys | {
                "publication_phase",
                "recovery_global_safety_timeline",
                "recovery_safety",
                "recovery_safety_plan_binding",
            } or result["verified"] is not True or result["mutated"] is not True:
                raise ControllerError(
                    "recovery_receipt_shape", "sealed-incomplete receipt differs"
                )
            run_id = _validate_receipt_run_id(
                result["run_id"], "sealed incomplete recovery"
            )
            marker_path = _canonical_receipt_path(
                result["publication_success_marker_path"],
                "sealed recovery success marker",
            )
            if (
                receipt_archive_path.name
                != f"{run_id}.recovered-incomplete"
                or marker_path.name != "publication-success.json"
                or marker_path.parent.name != run_id
                or result["procedure_sha256"] != PROCEDURE_DESCRIPTOR_SHA256
                or not isinstance(result["plan_sha256"], str)
                or _SHA256.fullmatch(result["plan_sha256"]) is None
                or result["publication_phase"] != "verified"
                or type(result["publication_success_marker_verified"])
                is not bool
            ):
                raise ControllerError(
                    "recovery_receipt_binding",
                    "sealed recovery path, run, phase, plan, or procedure identity differs",
                )
            _validate_manifest_receipt(
                result["manifest"],
                expected_manifest_path=receipt_archive_path / "manifest.json",
            )
            verification_timeline = _validate_global_safety_timeline_structure(
                result["verification_global_safety_timeline"],
                "sealed recovery verification",
            )
            recovery_safety = result["recovery_safety"]
            if not isinstance(recovery_safety, Mapping) or set(
                recovery_safety
            ) != {
                "post_manifest",
                "post_marker_pre_manifest",
                "prospective_marker_and_manifest",
            }:
                raise ControllerError(
                    "recovery_safety_shape", "sealed recovery safety members differ"
                )
            ordered_samples: list[Mapping[str, object]] = []
            for role in (
                "prospective_marker_and_manifest",
                "post_marker_pre_manifest",
                "post_manifest",
            ):
                sample = recovery_safety[role]
                if sample is None:
                    if role == "post_manifest":
                        raise ControllerError(
                            "recovery_safety_shape", "post-manifest sample is required"
                        )
                    continue
                ordered_samples.append(
                    _validate_safety_sample_receipt(sample, role)
                )
            if any(
                right["monotonic_ns"] < left["monotonic_ns"]
                for left, right in zip(ordered_samples, ordered_samples[1:])
            ):
                raise ControllerError(
                    "recovery_safety_timeline",
                    "recovery safety samples are not monotonic",
                )
            safety_binding = result["recovery_safety_plan_binding"]
            if (
                type(safety_binding) is not dict
                or set(safety_binding)
                != {
                    "maximum_poll_gap_milliseconds",
                    "maximum_safety_samples",
                    "plan_sha256",
                    "schema",
                }
                or safety_binding["schema"]
                != "oai.memprof.r0.recovery-safety-plan-binding/v1"
                or type(safety_binding["maximum_poll_gap_milliseconds"])
                is not int
                or safety_binding["maximum_poll_gap_milliseconds"] < 1
                or type(safety_binding["maximum_safety_samples"]) is not int
                or safety_binding["maximum_safety_samples"] < 1
                or safety_binding["plan_sha256"] != result["plan_sha256"]
            ):
                raise ControllerError(
                    "recovery_safety_plan_binding",
                    "sealed recovery safety limits differ",
                )
            global_recovery_timeline = (
                _validate_global_safety_timeline_limits(
                    result["recovery_global_safety_timeline"],
                    "sealed recovery",
                    maximum_safety_samples=(
                        safety_binding["maximum_safety_samples"]
                    ),
                    maximum_poll_gap_milliseconds=(
                        safety_binding["maximum_poll_gap_milliseconds"]
                    ),
                )
            )
            if (
                not ordered_samples
                or global_recovery_timeline["sample_count"]
                < len(ordered_samples)
                or global_recovery_timeline["first_monotonic_ns"]
                > ordered_samples[0]["monotonic_ns"]
                or global_recovery_timeline["last_monotonic_ns"]
                < ordered_samples[-1]["monotonic_ns"]
            ):
                raise ControllerError(
                    "recovery_global_safety_binding",
                    "selected recovery samples are outside the shared timeline",
                )
            if canonical_json_bytes(verification_timeline) != canonical_json_bytes(
                global_recovery_timeline
            ):
                raise ControllerError(
                    "recovery_verification_timeline_binding",
                    "postverification and recovery timelines differ",
                )
        if result["verified"] is True:
            if (
                action == "VERIFY_EXISTING_ORIGINAL"
                and location is False
                and result["publication_success_marker_verified"] is not False
            ):
                raise ControllerError(
                    "recovery_original_location",
                    "an unbound original location cannot activate the marker",
                )
            if action in {
                "VERIFY_EXISTING_RECOVERED",
                "SEALED_AND_PUBLISHED_INCOMPLETE",
            } and (
                location is not False
                or result["publication_success_marker_verified"] is not False
            ):
                raise ControllerError(
                    "recovery_incomplete_location", "recovered receipt binding differs"
                )
        if axes.scientific_case_state == "COMPLETE":
            if not (
                action == "VERIFY_EXISTING_ORIGINAL"
                and result["verified"] is True
                and result["mutated"] is False
                and location is True
                and result["publication_success_marker_verified"] is True
            ):
                raise ControllerError(
                    "recovery_upgrade_forbidden",
                    "only read-only verification of an already-renamed original case may preserve COMPLETE",
                )
            return result
        if (
            axes.scientific_case_state != "INCOMPLETE"
            or axes.inclusion != "EXCLUDED"
            or axes.profiler_stream_state != "NOT_APPLICABLE_R0"
            or axes.production_final_elf_absence != PRODUCTION_FINAL_ELF_ABSENCE
        ):
            raise ControllerError(
                "recovery_axes_preservation",
                "recovery must preserve explicit INCOMPLETE/EXCLUDED/R0-N-A axes",
            )
        return result

    def verify(self, archive_path: os.PathLike[str] | str) -> Mapping[str, object]:
        """Read-only archive verification; it never repairs terminal state."""

        requested = pathlib.Path(archive_path)
        result_value = self.services.verify_archive(requested)
        if not isinstance(result_value, Mapping):
            raise ControllerError("verification_receipt_type", type(result_value).__name__)
        result = dict(result_value)
        requested_path = _canonical_receipt_path(str(requested), "archive")
        if (
            result.get("schema") != "oai.memprof.r0.archive-verification/v1"
            or type(result.get("verified")) is not bool
            or type(result.get("mutated")) is not bool
            or result.get("archive_path") != str(requested)
        ):
            raise ControllerError(
                "verification_receipt_schema",
                "verification receipt schema, booleans, or path differ",
            )
        if result["mutated"] is not False:
            raise ControllerError("verify_mutation_forbidden", "verify operation reported a mutation")
        axes_keys = {
            "archive_integrity",
            "claims",
            "inclusion",
            "production_final_elf_absence",
            "profiler_stream_state",
            "reasons",
            "scientific_case_state",
        }
        axes = _CoreServices._axes_from_mapping(
            {key: result[key] for key in axes_keys if key in result}
        )
        if (result["verified"] is True) != (
            axes.archive_integrity == "VERIFIED"
        ):
            raise ControllerError(
                "verification_receipt_integrity", "verified and archive axis differ"
            )
        if result["verified"] is True:
            expected_keys = axes_keys | {
                "archive_path",
                "location_matches_original_publication",
                "manifest",
                "mutated",
                "plan_sha256",
                "publication_success_marker_path",
                "publication_success_marker_verified",
                "procedure_sha256",
                "run_id",
                "schema",
                "verified",
                "verification_global_safety_timeline",
            }
            if (
                set(result) != expected_keys
                or type(result["location_matches_original_publication"]) is not bool
                or type(result["publication_success_marker_verified"]) is not bool
                or not isinstance(result["publication_success_marker_path"], str)
                or not pathlib.PurePosixPath(
                    result["publication_success_marker_path"]
                ).is_absolute()
                or result["publication_success_marker_path"].startswith("//")
                or result["publication_success_marker_path"]
                != str(pathlib.PurePosixPath(result["publication_success_marker_path"]))
                or (
                    axes.scientific_case_state == "COMPLETE"
                    and (
                        result["publication_success_marker_verified"] is not True
                        or result["location_matches_original_publication"] is not True
                    )
                )
                or (
                    result["publication_success_marker_verified"] is False
                    and "PUBLICATION_COMPLETION_MARKER_UNAVAILABLE"
                    not in axes.reasons
                    and "ORIGINAL_PUBLICATION_LOCATION_UNVERIFIED"
                    not in axes.reasons
                    and not result["archive_path"].endswith(
                        ".recovered-incomplete"
                    )
                )
                or result["procedure_sha256"] != PROCEDURE_DESCRIPTOR_SHA256
                or not isinstance(result["plan_sha256"], str)
                or _SHA256.fullmatch(str(result["plan_sha256"])) is None
            ):
                raise ControllerError(
                    "verification_receipt_shape", "verified receipt members differ"
                )
            run_id = _validate_receipt_run_id(
                result["run_id"], "archive verification"
            )
            recovered_suffix = ".recovered-incomplete"
            expected_archive_name = (
                f"{run_id}{recovered_suffix}"
                if result["archive_path"].endswith(recovered_suffix)
                else run_id
            )
            marker_path = _canonical_receipt_path(
                result["publication_success_marker_path"],
                "verification success marker",
            )
            if (
                requested_path.name != expected_archive_name
                or marker_path.name != "publication-success.json"
                or marker_path.parent.name != run_id
            ):
                raise ControllerError(
                    "verification_receipt_binding",
                    "verified run, archive, or success-marker path differs",
                )
            _validate_manifest_receipt(
                result["manifest"],
                expected_manifest_path=requested_path / "manifest.json",
            )
            _validate_global_safety_timeline_structure(
                result["verification_global_safety_timeline"],
                "successful archive verification",
            )
        else:
            if set(result) != axes_keys | {
                "archive_path",
                "mutated",
                "problem",
                "schema",
                "verified",
                "verification_global_safety_timeline",
            }:
                raise ControllerError(
                    "verification_receipt_shape", "failed receipt members differ"
                )
            problem = result["problem"]
            if (
                not isinstance(problem, Mapping)
                or set(problem) != {"code", "message"}
                or not isinstance(problem["code"], str)
                or not problem["code"]
                or not isinstance(problem["message"], str)
            ):
                raise ControllerError(
                    "verification_problem_shape", "failure problem differs"
                )
            verification_timeline = result[
                "verification_global_safety_timeline"
            ]
            if verification_timeline is not None:
                _validate_global_safety_timeline_structure(
                    verification_timeline,
                    "failed archive verification",
                )
            if (
                problem["code"] == "maximum_poll_gap"
                and verification_timeline is None
            ):
                raise ControllerError(
                    "verification_problem_timeline",
                    "maximum poll-gap failure omitted its live timeline",
                )
        return result


def controller_self_test() -> dict[str, object]:
    checks = 0
    verify_procedure_descriptor()
    checks += 1
    parse_ctest_inventory(
        "Test project /tmp/x\n"
        + "\n".join(f"  Test #{index}: {name}" for index, name in enumerate(WIRE_CTESTS, 1))
        + f"\n\nTotal Tests: {len(WIRE_CTESTS)}\n",
        WIRE_CTESTS,
    )
    checks += 1
    validate_ctest_run(
        "\n".join(
            f"{index}/3 Test #{index}: {name} .... Passed 0.01 sec"
            for index, name in enumerate(WIRE_CTESTS, 1)
        )
        + "\n100% tests passed, 0 tests failed out of 3\n",
        WIRE_CTESTS,
    )
    checks += 1
    validate_positive_elf_output(
        b"R0_ELF_GATE_V2 pass architecture=x86_64 roles=4 wrappers=4 "
        b"runtime_soname=liboai_memprof_runtime.so.1 generator=Ninja compiler=GNU\n",
        b"",
        "x86_64",
    )
    checks += 1
    mutant_checks = 0
    for mutation in (
        lambda: parse_ctest_inventory("Total Tests: 0\n", WIRE_CTESTS),
        lambda: validate_ctest_run(
            "1/3 Test #1: wrong .... Passed 0.01 sec\n"
            "100% tests passed, 0 tests failed out of 3\n",
            WIRE_CTESTS,
        ),
        lambda: validate_ctest_run(
            "\n".join(
                f"{index}/3 Test #{index}: {name} .... "
                f"{'Skipped' if index == 2 else 'Passed'} 0.01 sec"
                for index, name in enumerate(WIRE_CTESTS, 1)
            )
            + "\n100% tests passed, 0 tests failed out of 3\n",
            WIRE_CTESTS,
        ),
        lambda: validate_ctest_run(
            "1/3 Test #1: oai_memprof_wire_c .... Passed 0.01 sec\n"
            "2/3 Test #2: oai_memprof_wire_c .... Passed 0.01 sec\n"
            "3/3 Test #3: oai_memprof_wire_python .... Passed 0.01 sec\n"
            "100% tests passed, 0 tests failed out of 3\n",
            WIRE_CTESTS,
        ),
        lambda: validate_ctest_run(
            "\n".join(
                f"{index}/3 Test #{index}: {name} .... "
                f"{'Not Run' if index == 3 else 'Passed'} 0.01 sec"
                for index, name in enumerate(WIRE_CTESTS, 1)
            )
            + "\n100% tests passed, 0 tests failed out of 3\n",
            WIRE_CTESTS,
        ),
        lambda: validate_positive_elf_output(b"R0_ELF_GATE_V2 pass\n", b"", "x86_64"),
        lambda: classify_terminal_result(
            condition_complete={"OFF0": True, "OFF1": True},
            archive_verified=True,
            interrupted=False,
            positive_elf_complete=True,
        ),
    ):
        try:
            value = mutation()
            if isinstance(value, TerminalAxes) and value.scientific_case_state == "INCOMPLETE":
                mutant_checks += 1
                continue
        except GateError:
            mutant_checks += 1
            continue
        raise ControllerError("selftest_mutant_survived", f"mutant {mutant_checks + 1}")
    checks += mutant_checks
    unavailable = classify_terminal_result(
        condition_complete={name: False for name, _, _ in R0_CONDITIONS},
        archive_verified=False,
        interrupted=False,
    )
    if tuple(claim.eligibility for claim in unavailable.claims[:2]) != (
        "UNAVAILABLE",
        "UNAVAILABLE",
    ):
        raise ControllerError(
            "selftest_unavailable_claim", "missing evidence was not preserved as unavailable"
        )
    checks += 1
    supporting = classify_terminal_result(
        condition_complete={name: True for name, _, _ in R0_CONDITIONS},
        archive_verified=True,
        interrupted=False,
        failure_reasons=("POSTCONDITION_FAILURE",),
        absence_decisions=(
            AbsenceDecision("OFF0", True, "QUALIFIED", "REFERENCE_TOOL_MATCH"),
            AbsenceDecision("OFF1", True, "QUALIFIED", "REFERENCE_TOOL_MATCH"),
        ),
        positive_elf_complete=True,
    )
    if tuple(claim.eligibility for claim in supporting.claims[:2]) != (
        "SUPPORTING_ONLY",
        "SUPPORTING_ONLY",
    ):
        raise ControllerError(
            "selftest_incomplete_claim", "incomplete case retained a qualified claim"
        )
    checks += 1
    return {
        "checks": checks,
        "procedure_sha256": PROCEDURE_DESCRIPTOR_SHA256,
        "schema": "oai.memprof.r0.terminal-self-test/v1",
        "verdict": "pass",
    }


@dataclasses.dataclass(frozen=True)
class _CoreCommandObservation:
    spec: object
    result: object
    stdout_capture: EvidenceIdentity
    stderr_capture: EvidenceIdentity
    started_utc_ns: int
    ended_utc_ns: int
    maximum_observed_poll_gap_ns: int | None
    local_failure_code: str | None
    local_failure_detail: str | None
    post_safety_sample: object | None
    caller_signal_mask_before: tuple[int, ...]
    caller_signal_mask_after: tuple[int, ...]
    maximum_descendant_identities: int
    safety_timeline: _SafetyTimelineState


@dataclasses.dataclass
class _SafetyTimelineState:
    plan: RunPlan
    storage_path: pathlib.Path
    baseline_free_bytes: int
    sample_count: int = 0
    first_monotonic_ns: int | None = None
    last_monotonic_ns: int | None = None
    maximum_observed_gap_ns: int | None = None
    last_sample: object | None = None


@dataclasses.dataclass
class _CoreContext:
    plan: RunPlan
    workspace: RunWorkspace
    journal_cursor: object
    baseline_free_bytes: int
    baseline_archive_free_bytes: int
    archive_payload_bytes: int
    archive_regular_file_count: int
    archive_directory_count: int
    archive_entry_count: int
    root_identities: tuple[tuple[str, int, int, int], ...]
    scratch_identity: tuple[str, int, int, int]
    archive_stage_identity: tuple[str, int, int, int]
    thermal_sensor_identities: tuple[tuple[str, int, int, int], ...]
    active_mount_identity: MountIdentity
    acquisition_allocation_unit_bytes: int
    archive_allocation_unit_bytes: int
    safety_timeline: _SafetyTimelineState
    last_phase: str | None = None


class _ArchiveSignalScope:
    """Defer the finite supported set to declared archive callback boundaries."""

    def __init__(
        self,
        supported: tuple[int, ...],
        handlers: Mapping[int, Callable[[int, object], object]],
    ) -> None:
        self.supported = supported
        self.supported_set = frozenset(supported)
        self.handlers = dict(handlers)
        self.caller_mask: tuple[int, ...] = ()
        self.blocked = False
        self.signal_primary: BaseException | None = None
        self.events: list[str] = []

    @staticmethod
    def _mask_numbers(values: Sequence[int]) -> tuple[int, ...]:
        if any(type(value) is not int or value <= 0 for value in values):
            raise ControllerError(
                "archive_signal_mask", "signal mask contains a nonpositive non-integer"
            )
        result = tuple(sorted(set(values)))
        if len(result) != len(values):
            raise ControllerError(
                "archive_signal_mask", "signal mask contains duplicate identities"
            )
        return result

    def enter(self) -> None:
        try:
            observed = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        except (OSError, ValueError, AttributeError) as error:
            raise ControllerError("archive_signal_mask", error) from error
        self.caller_mask = self._mask_numbers(tuple(int(item) for item in observed))
        incompatible = self.supported_set.intersection(self.caller_mask)
        if incompatible:
            raise ControllerError(
                "archive_signal_mask_precondition",
                "supported cancellation signal is already blocked",
            )
        try:
            previous = signal.pthread_sigmask(signal.SIG_BLOCK, self.supported_set)
        except (OSError, ValueError) as error:
            raise ControllerError("archive_signal_mask", error) from error
        self.blocked = True
        try:
            if (
                self._mask_numbers(tuple(int(item) for item in previous))
                != self.caller_mask
            ):
                raise ControllerError(
                    "archive_signal_mask",
                    "caller mask changed before archive primitive",
                )
            self.validate_handlers("archive-signal-scope-enter")
        except BaseException as primary:
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, set(self.caller_mask))
                self.blocked = False
            except BaseException as restore_error:
                primary.add_note(
                    "archive signal-scope entry restoration raised "
                    f"{type(restore_error).__name__}: {_bounded_text(restore_error)}"
                )
            raise

    def validate_handlers(self, boundary: str) -> None:
        try:
            observed = {
                number: signal.getsignal(number) for number in self.supported
            }
        except (OSError, ValueError, AttributeError) as error:
            raise ControllerError("archive_signal_handler", error) from error
        changed = tuple(
            self._name(number)
            for number in self.supported
            if observed[number] is not self.handlers[number]
        )
        if changed:
            raise ControllerError(
                "archive_signal_handler_drift",
                f"{boundary}: {','.join(changed)}",
            )

    def _pending_snapshot(self) -> tuple[int, ...]:
        try:
            pending = signal.sigpending()
        except (OSError, ValueError) as error:
            raise ControllerError("archive_signal_pending", error) from error
        return tuple(
            number
            for number in self._mask_numbers(tuple(int(item) for item in pending))
            if number in self.supported_set
        )

    @staticmethod
    def _name(number: int) -> str:
        return f"{signal.Signals(number).name}({number})"

    def checkpoint(self, boundary: str) -> None:
        """Consume one pending-set snapshot and deliver at most one primary."""

        pending = self._pending_snapshot()
        if not pending:
            return
        selected = None
        if self.signal_primary is None:
            selected = next(
                number
                for number in SUPPORTED_CANCELLATION_SIGNAL_PRIORITY
                if number in pending
            )
        for number in pending:
            try:
                consumed = signal.sigwait({number})
            except (OSError, ValueError) as error:
                raise ControllerError("archive_signal_consume", error) from error
            if (
                isinstance(consumed, bool)
                or not isinstance(consumed, int)
                or int(consumed) != number
            ):
                raise ControllerError(
                    "archive_signal_consume",
                    "sigwait returned a different signal identity",
                )
        selected_text = "existing-primary" if selected is None else self._name(selected)
        event = (
            f"archive-supported-signal boundary={boundary} "
            f"observed={','.join(self._name(number) for number in pending)} "
            f"selected={selected_text}"
        )
        self.events.append(event)
        if self.signal_primary is not None:
            self.signal_primary.add_note(event)
            return
        if selected is None:
            raise AssertionError("selected signal required without a primary")
        handler = self.handlers[selected]
        try:
            handler(selected, None)
        except BaseException as error:
            self.signal_primary = error
            error.add_note(event)
            raise
        error = ControllerError(
            "archive_cancellation_handler_returned",
            f"captured handler returned for {self._name(selected)}",
        )
        error.add_note(event)
        self.signal_primary = error
        raise error

    def finish(self, primary: BaseException | None) -> None:
        """Drain once, restore exactly, and never replace an existing primary."""

        generated: BaseException | None = None
        if primary is not None and self.signal_primary is None:
            self.signal_primary = primary
        try:
            self.checkpoint("archive-primitive-final-boundary")
        except BaseException as error:
            generated = error
        restore_error: BaseException | None = None
        if self.blocked:
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, set(self.caller_mask))
                self.blocked = False
                observed = signal.pthread_sigmask(signal.SIG_BLOCK, set())
                if self._mask_numbers(
                    tuple(int(item) for item in observed)
                ) != self.caller_mask:
                    raise ControllerError(
                        "archive_signal_mask", "caller mask was not restored exactly"
                    )
            except BaseException as error:
                restore_error = error
        effective_primary = (
            primary
            if primary is not None
            else (
                self.signal_primary
                if self.signal_primary is not None
                else generated
            )
        )
        if effective_primary is not None:
            if generated is not None and generated is not effective_primary:
                effective_primary.add_note(
                    "archive final signal drain raised "
                    f"{type(generated).__name__}: {_bounded_text(generated)}"
                )
            if restore_error is not None:
                effective_primary.add_note(
                    "archive caller signal-mask restoration raised "
                    f"{type(restore_error).__name__}: {_bounded_text(restore_error)}"
                )
            if primary is None:
                raise effective_primary
            return
        if restore_error is not None:
            raise restore_error


_NONTERMINAL_PHASES = (
    "INITIALIZED",
    "SOURCE_BOUND",
    "OFF0_RUNNING",
    "OFF0_COMPLETE",
    "OFF1_RUNNING",
    "OFF1_COMPLETE",
    "ON_RUNNING",
    "ON_COMPLETE",
    "VALIDATION_COMPLETE",
)
_STATE_TRANSITIONS = {
    None: frozenset({"INITIALIZED"}),
    "INITIALIZED": frozenset({"SOURCE_BOUND", "FAILED", "INTERRUPTED"}),
    "SOURCE_BOUND": frozenset({"OFF0_RUNNING", "FAILED", "INTERRUPTED"}),
    "OFF0_RUNNING": frozenset({"OFF0_COMPLETE", "FAILED", "INTERRUPTED"}),
    "OFF0_COMPLETE": frozenset({"OFF1_RUNNING", "FAILED", "INTERRUPTED"}),
    "OFF1_RUNNING": frozenset({"OFF1_COMPLETE", "FAILED", "INTERRUPTED"}),
    "OFF1_COMPLETE": frozenset({"ON_RUNNING", "FAILED", "INTERRUPTED"}),
    "ON_RUNNING": frozenset({"ON_COMPLETE", "FAILED", "INTERRUPTED"}),
    "ON_COMPLETE": frozenset({"VALIDATION_COMPLETE", "FAILED", "INTERRUPTED"}),
    "VALIDATION_COMPLETE": frozenset(
        {"READY_TO_SEAL", "FAILED", "INTERRUPTED"}
    ),
    "FAILED": frozenset({"READY_TO_SEAL"}),
    "INTERRUPTED": frozenset({"READY_TO_SEAL"}),
    "READY_TO_SEAL": frozenset(),
}


class _CoreServices:
    """Production adapter over the separately tested process/archive cores."""

    def __init__(self) -> None:
        try:
            module_directory = pathlib.Path(__file__).resolve(strict=True).parent
        except OSError as error:
            raise ControllerError("runtime_controller_source", error) from error
        archive_path = module_directory / "oai_memprof_r0_terminal_archive.py"
        process_path = module_directory / "oai_memprof_r0_terminal_process.py"
        self._runtime_core_bindings = types.MappingProxyType(
            {
                "archive": _bind_runtime_core_source(
                    "archive", archive_path, ARCHIVE_CORE_SOURCE_SHA256
                ),
                "process": _bind_runtime_core_source(
                    "process", process_path, PROCESS_CORE_SOURCE_SHA256
                ),
            }
        )
        self.archive = _archive_module()
        self.process = importlib.import_module(
            "tools.profiling.memory.oai_memprof_r0_terminal_process"
        )
        for role, module, expected_path in (
            ("archive", self.archive, archive_path),
            ("process", self.process, process_path),
        ):
            module_path_value = getattr(module, "__file__", None)
            if not isinstance(module_path_value, str):
                raise ControllerError(
                    "runtime_core_module_path", f"{role}: missing source path"
                )
            try:
                module_path = pathlib.Path(module_path_value).resolve(strict=True)
            except OSError as error:
                raise ControllerError(
                    "runtime_core_module_path", f"{role}: {error}"
                ) from error
            if module_path != expected_path:
                raise ControllerError(
                    "runtime_core_module_path",
                    f"{role}: expected {expected_path}, observed {module_path}",
                )
        expected_supported_signals = tuple(
            sorted({int(signal.SIGHUP), int(signal.SIGINT), int(signal.SIGTERM)})
        )
        observed_supported_signals = getattr(
            self.process, "SUPPORTED_CANCELLATION_SIGNALS", None
        )
        if (
            type(observed_supported_signals) is not tuple
            or observed_supported_signals != expected_supported_signals
            or any(type(number) is not int for number in observed_supported_signals)
        ):
            raise ControllerError(
                "process_signal_api",
                "SUPPORTED_CANCELLATION_SIGNALS differs from exact SIGHUP/SIGINT/SIGTERM",
            )
        self._supported_cancellation_signals = observed_supported_signals
        observed_priority = getattr(
            self.process, "SUPPORTED_CANCELLATION_SIGNAL_PRIORITY", None
        )
        if (
            type(observed_priority) is not tuple
            or observed_priority != SUPPORTED_CANCELLATION_SIGNAL_PRIORITY
            or any(type(number) is not int for number in observed_priority)
        ):
            raise ControllerError(
                "process_signal_api",
                "SUPPORTED_CANCELLATION_SIGNAL_PRIORITY differs from SIGINT/SIGTERM/SIGHUP",
            )
        if getattr(self.process, "CONTAINMENT_MODEL", None) != (
            "linux_subreaper_ppid_pidfd_v1"
        ):
            raise ControllerError(
                "process_containment_api", "accepted containment model differs"
            )
        expected_process_result_fields = (
            "argv",
            "cwd",
            "pid",
            "returncode",
            "started_monotonic_ns",
            "ended_monotonic_ns",
            "stdout_bytes",
            "stderr_bytes",
            "stdout_sha256",
            "stderr_sha256",
            "safety_samples",
            "failure_code",
            "failure_detail",
            "kill_attempted",
            "cleanup_complete",
            "cleanup_failure_code",
            "cleanup_failure_detail",
            "unexpected_descendant_identity_lower_bound",
            "unexpected_descendant_identity_complete",
            "unexpected_descendant_pids",
            "maximum_observed_live_descendant_count",
            "observed_descendant_identity_lower_bound",
            "observed_descendant_identity_complete",
            "observed_descendant_identities",
            "observed_descendant_identities_truncated",
            "unexpected_descendant_identities",
            "unexpected_descendant_identities_truncated",
            "caller_signal_mask_numbers",
            "signal_mask_restored",
            "deferred_supported_signal_numbers",
            "containment_mode",
            "containment_caller_thread_count",
            "containment_preexisting_child_count",
            "containment_subreaper_previously_enabled",
            "containment_subreaper_restored",
            "descendant_scan_count",
            "first_descendant_scan_monotonic_ns",
            "last_descendant_scan_monotonic_ns",
            "maximum_observed_descendant_scan_gap_ns",
        )
        try:
            observed_process_result_fields = tuple(
                field.name for field in dataclasses.fields(self.process.ProcessResult)
            )
        except (AttributeError, TypeError) as error:
            raise ControllerError("process_result_api", error) from error
        if observed_process_result_fields != expected_process_result_fields:
            raise ControllerError(
                "process_result_api",
                "accepted ProcessResult field population/order differs",
            )
        self._contexts: dict[str, _CoreContext] = {}

    @staticmethod
    def _current_signal_mask_numbers() -> tuple[int, ...]:
        try:
            observed = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        except (OSError, ValueError, AttributeError) as error:
            raise ControllerError("signal_mask_observation", error) from error
        numbers = tuple(sorted(int(item) for item in observed))
        if (
            any(number <= 0 for number in numbers)
            or len(numbers) != len(set(numbers))
        ):
            raise ControllerError(
                "signal_mask_observation", "caller mask is not sorted/unique/positive"
            )
        return numbers

    @staticmethod
    def _validate_terminal_signal_handlers() -> Mapping[int, Callable[[int, object], object]]:
        if threading.current_thread() is not threading.main_thread():
            raise ControllerError(
                "signal_handler_precondition",
                "archive and process operations require the Python main thread",
            )
        expected: dict[int, Callable[[int, object], object]] = {
            int(signal.SIGINT): signal.default_int_handler,
            int(signal.SIGTERM): _CatchableTermination._raise_system_exit,
            int(signal.SIGHUP): _CatchableTermination._raise_system_exit,
        }
        observed: dict[int, object] = {}
        try:
            for number in SUPPORTED_CANCELLATION_SIGNAL_PRIORITY:
                observed[number] = signal.getsignal(number)
        except (OSError, ValueError, AttributeError) as error:
            raise ControllerError("signal_handler_precondition", error) from error
        mismatches = tuple(
            signal.Signals(number).name
            for number, handler in expected.items()
            if observed[number] is not handler
        )
        if mismatches:
            raise ControllerError(
                "signal_handler_precondition",
                "terminal signal handler binding differs for " + ",".join(mismatches),
            )
        return types.MappingProxyType(expected)

    def _observe_timeline_sample(
        self,
        timeline: _SafetyTimelineState,
        sample: object,
        *,
        minimum_free_bytes: int,
    ) -> None:
        plan = timeline.plan
        if type(minimum_free_bytes) is not int or minimum_free_bytes < 0:
            raise ControllerError("global_safety_threshold", minimum_free_bytes)
        monotonic_ns = getattr(sample, "monotonic_ns", None)
        free_bytes = getattr(sample, "free_bytes", None)
        thermal = getattr(sample, "thermal_millicelsius", None)
        expected_paths = tuple(
            sensor.path for sensor in plan.safety.mandatory_thermal_sensors
        )
        if (
            type(monotonic_ns) is not int
            or monotonic_ns < 0
            or type(free_bytes) is not int
            or free_bytes < 0
            or type(thermal) is not tuple
            or tuple(path for path, _value in thermal) != expected_paths
            or any(
                type(path) is not str or type(value) is not int
                for path, value in thermal
            )
        ):
            raise ControllerError(
                "global_safety_sample_contract",
                "sample timestamp, storage, or ordered thermal population differs",
            )
        if timeline.sample_count >= plan.safety.maximum_safety_samples:
            raise ControllerError(
                "global_safety_sample_limit",
                plan.safety.maximum_safety_samples,
            )
        for sensor, (path, value) in zip(
            plan.safety.mandatory_thermal_sensors, thermal, strict=True
        ):
            if value < sensor.minimum_plausible_millicelsius:
                raise ControllerError(
                    "thermal_sensor_plausibility",
                    f"{path}={value}<{sensor.minimum_plausible_millicelsius}",
                )
            if value >= plan.safety.thermal_ceiling_millicelsius:
                raise ControllerError(
                    "thermal_limit",
                    f"{path}={value}>={plan.safety.thermal_ceiling_millicelsius}",
                )
        if free_bytes < minimum_free_bytes:
            raise ControllerError(
                "free_space_limit", f"{free_bytes}<{minimum_free_bytes}"
            )
        maximum_gap_ns = plan.safety.maximum_poll_gap_milliseconds * 1_000_000
        if timeline.last_monotonic_ns is not None:
            gap_ns = monotonic_ns - timeline.last_monotonic_ns
            if gap_ns < 0:
                raise ControllerError(
                    "global_safety_timeline", "monotonic sample time moved backwards"
                )
            timeline.maximum_observed_gap_ns = max(
                timeline.maximum_observed_gap_ns or 0, gap_ns
            )
            timeline.last_monotonic_ns = monotonic_ns
            timeline.last_sample = sample
            timeline.sample_count += 1
            if gap_ns > maximum_gap_ns:
                raise ControllerError(
                    "maximum_poll_gap", f"{gap_ns}>{maximum_gap_ns}"
                )
            return
        timeline.first_monotonic_ns = monotonic_ns
        timeline.last_monotonic_ns = monotonic_ns
        timeline.last_sample = sample
        timeline.sample_count = 1

    def _timeline_minimum_free(self, timeline: _SafetyTimelineState) -> int:
        return max(
            timeline.plan.safety.minimum_free_during_bytes,
            timeline.baseline_free_bytes
            - timeline.plan.safety.maximum_observed_free_space_decline_bytes,
        )

    def _force_timeline_sample(
        self,
        timeline: _SafetyTimelineState,
        *,
        minimum_free_bytes: int | None = None,
    ) -> object:
        threshold = (
            self._timeline_minimum_free(timeline)
            if minimum_free_bytes is None
            else minimum_free_bytes
        )
        sample = self.process.sample_safety(
            self._safety_spec(timeline.plan, timeline.storage_path, threshold)
        )
        self._observe_timeline_sample(
            timeline, sample, minimum_free_bytes=threshold
        )
        return sample

    def _timeline_progress(self, timeline: _SafetyTimelineState) -> None:
        if timeline.last_monotonic_ns is None:
            self._force_timeline_sample(timeline)
            return
        now_ns = time.monotonic_ns()
        if now_ns < timeline.last_monotonic_ns:
            raise ControllerError(
                "global_safety_timeline", "monotonic clock moved backwards"
            )
        interval_ns = timeline.plan.safety.poll_interval_milliseconds * 1_000_000
        if now_ns - timeline.last_monotonic_ns >= interval_ns:
            self._force_timeline_sample(timeline)

    def _new_live_safety_timeline(
        self, plan: RunPlan, storage_path: pathlib.Path
    ) -> _SafetyTimelineState:
        self._validate_terminal_signal_handlers()
        initial = self.process.sample_safety(
            self._safety_spec(
                plan,
                storage_path,
                plan.safety.minimum_free_during_bytes,
            )
        )
        free_bytes = getattr(initial, "free_bytes", None)
        if type(free_bytes) is not int or free_bytes < 0:
            raise ControllerError(
                "global_safety_sample_contract", "initial free bytes differ"
            )
        timeline = _SafetyTimelineState(
            plan=plan,
            storage_path=storage_path,
            baseline_free_bytes=free_bytes,
        )
        self._observe_timeline_sample(
            timeline,
            initial,
            minimum_free_bytes=plan.safety.minimum_free_during_bytes,
        )
        return timeline

    @staticmethod
    def _timeline_receipt(timeline: _SafetyTimelineState) -> Mapping[str, object]:
        return {
            "first_monotonic_ns": timeline.first_monotonic_ns,
            "last_monotonic_ns": timeline.last_monotonic_ns,
            "maximum_observed_gap_ns": timeline.maximum_observed_gap_ns,
            "sample_count": timeline.sample_count,
            "schema": "oai.memprof.r0.global-safety-timeline/v1",
        }

    def _archive_call(
        self,
        timeline: _SafetyTimelineState | None,
        operation: str,
        function: Callable[..., Any],
        *args: object,
        force_boundaries: bool = True,
        **kwargs: object,
    ) -> Any:
        if "progress_callback" in kwargs:
            raise ControllerError(
                "archive_progress_callback", "caller supplied a duplicate callback"
            )
        handlers = self._validate_terminal_signal_handlers()
        scope = _ArchiveSignalScope(
            self._supported_cancellation_signals, handlers
        )
        scope.enter()
        primary: BaseException | None = None

        def progress() -> None:
            scope.checkpoint(f"{operation}:progress-before-safety")
            if scope.signal_primary is not None:
                return
            if timeline is not None:
                self._timeline_progress(timeline)
            scope.checkpoint(f"{operation}:progress-after-safety")

        try:
            scope.checkpoint(f"{operation}:before")
            if force_boundaries and timeline is not None:
                self._force_timeline_sample(timeline)
            scope.validate_handlers(f"{operation}:before-primitive")
            result = function(*args, **kwargs, progress_callback=progress)
            scope.checkpoint(f"{operation}:returned-before-safety")
            if force_boundaries and timeline is not None:
                self._force_timeline_sample(timeline)
            scope.checkpoint(f"{operation}:returned-after-safety")
            return result
        except BaseException as error:
            primary = error
            raise
        finally:
            scope.finish(primary)

    def _commit_publication_success_marker(
        self,
        path: pathlib.Path,
        data: bytes,
        *,
        maximum_bytes: int,
    ) -> Mapping[str, object] | None:
        """Commit the final marker with no post-commit scientific operation.

        All evidence validation and safety sampling is complete before entry.
        The final pending-set snapshot immediately before the write is the
        supported-signal ordering cutoff, conditional on later durable success.
        Newly pending signals after that snapshot are post-cutoff even when the
        kernel observes them before fsync returns.  Supported signals remain
        blocked through the bounded exclusive write, file fsync, and parent
        fsync.  The archive primitive intentionally gets no progress callback:
        its successful return is the external durability linearization point.
        Signal-mask restoration after that return is an operational
        post-completion boundary and cannot mutate the marker.
        """

        handlers = self._validate_terminal_signal_handlers()
        scope = _ArchiveSignalScope(
            self._supported_cancellation_signals, handlers
        )
        scope.enter()
        primary: BaseException | None = None
        committed = False
        postcompletion_problem: Mapping[str, object] | None = None
        try:
            scope.checkpoint("publication-success-marker:before-commit")
            self.archive.write_bytes_exclusive(
                path,
                data,
                max_bytes=maximum_bytes,
                progress_callback=None,
            )
            committed = True
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                scope.finish(primary)
            except BaseException as error:
                if committed:
                    error.add_note(
                        "publication-success marker was durably committed before "
                        "post-completion signal restoration"
                    )
                    if isinstance(error, (KeyboardInterrupt, SystemExit)):
                        raise
                    raw_code = getattr(error, "code", None)
                    normalized_code = getattr(raw_code, "value", raw_code)
                    if type(normalized_code) is not str or not normalized_code:
                        normalized_code = type(error).__name__
                    postcompletion_problem = {
                        "code": _bounded_text(normalized_code, 128),
                        "message": _bounded_text(getattr(error, "detail", error)),
                        "phase": "postcompletion_signal_mask_restoration",
                        "scientific_effect": (
                            "none_durable_success_marker_already_committed"
                        ),
                    }
                else:
                    raise
        return postcompletion_problem

    @staticmethod
    def _error(code: str, error: BaseException) -> ControllerError:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise error
        raw_code = getattr(error, "code", None)
        core_code = getattr(raw_code, "value", raw_code)
        detail = getattr(error, "detail", str(error))
        phase = getattr(error, "publication_phase", None)
        phase_value = getattr(phase, "value", phase)
        return ControllerError(
            error.code if isinstance(error, ControllerError) else code,
            f"{core_code or type(error).__name__}: {detail}",
            publication_phase=(
                None if phase_value is None else _bounded_text(phase_value, 64)
            ),
        )

    def _context(self, workspace: RunWorkspace) -> _CoreContext:
        try:
            context = self._contexts[str(workspace.scratch)]
        except KeyError as error:
            raise ControllerError("workspace_context_missing", workspace.scratch) from error
        if context.workspace != workspace:
            raise ControllerError("workspace_context_mismatch", workspace.run_id)
        return context

    def _ensure_directory(
        self,
        root: pathlib.Path,
        directory: pathlib.Path,
        *,
        timeline: _SafetyTimelineState,
        maximum_creations: int | None = None,
    ) -> int:
        try:
            relative = directory.relative_to(root)
        except ValueError as error:
            raise ControllerError("directory_domain", f"{directory} outside {root}") from error
        current = root
        created = 0
        for component in relative.parts:
            current = current / component
            try:
                information = os.lstat(current)
            except FileNotFoundError:
                if maximum_creations is not None and created >= maximum_creations:
                    raise ControllerError(
                        "archive_directory_limit", f"before creating {current}"
                    )
                try:
                    self._archive_call(
                        timeline,
                        "create-directory",
                        self.archive.create_directory_exclusive,
                        current,
                    )
                    created += 1
                except BaseException as error:
                    raise self._error("directory_create", error) from error
                continue
            except OSError as error:
                raise ControllerError("directory_stat", error) from error
            if stat.S_ISLNK(information.st_mode) or not stat.S_ISDIR(information.st_mode):
                raise ControllerError("directory_type", current)
        return created

    def _safety_spec(
        self,
        plan: RunPlan,
        storage_path: pathlib.Path,
        minimum_free_bytes: int,
    ) -> object:
        return self.process.SafetySpec(
            sensors=tuple(
                self.process.ThermalSensor(
                    path=pathlib.Path(sensor.path),
                    ceiling_millicelsius=plan.safety.thermal_ceiling_millicelsius,
                    minimum_plausible_millicelsius=(
                        sensor.minimum_plausible_millicelsius
                    ),
                )
                for sensor in plan.safety.mandatory_thermal_sensors
            ),
            storage_path=storage_path,
            minimum_free_bytes=minimum_free_bytes,
            poll_interval_seconds=plan.safety.poll_interval_milliseconds / 1000.0,
            max_samples=plan.safety.maximum_safety_samples,
            sensor_max_bytes=plan.safety.maximum_sensor_bytes,
        )

    @staticmethod
    def _sample_record(sample: object) -> dict[str, object]:
        return {
            "free_bytes": int(sample.free_bytes),
            "monotonic_ns": int(sample.monotonic_ns),
            "thermal_millicelsius": [
                {"path": path, "value": value}
                for path, value in sample.thermal_millicelsius
            ],
        }

    @staticmethod
    def _require_path_absent(path: pathlib.Path, code: str) -> None:
        try:
            information = os.lstat(path)
        except FileNotFoundError:
            return
        except OSError as error:
            raise ControllerError(f"{code}_stat", error) from error
        if stat.S_ISLNK(information.st_mode):
            observed = "symlink"
        elif stat.S_ISDIR(information.st_mode):
            observed = "directory"
        elif stat.S_ISFIFO(information.st_mode):
            observed = "fifo"
        elif stat.S_ISREG(information.st_mode):
            observed = "regular_file"
        else:
            observed = "other"
        raise ControllerError(code, f"{path}: preexisting {observed}")

    def _mount_snapshot(
        self, timeline: _SafetyTimelineState, operation: str
    ) -> tuple[MountIdentity, ...]:
        try:
            return self._archive_call(
                timeline, operation, _read_mountinfo_snapshot
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except ControllerError:
            raise
        except BaseException as error:
            raise self._error("mountinfo_snapshot", error) from error

    def _recheck_workspace_parent_bindings(
        self,
        acquisition_parent: pathlib.Path,
        archive_parent: pathlib.Path,
        acquisition_expected: os.stat_result,
        archive_expected: os.stat_result,
        expected_mount: MountIdentity,
        timeline: _SafetyTimelineState,
    ) -> None:
        observed_information: list[os.stat_result] = []
        for role, path, expected in (
            ("acquisition", acquisition_parent, acquisition_expected),
            ("archive", archive_parent, archive_expected),
        ):
            try:
                information = os.lstat(path)
                resolved = path.resolve(strict=True)
            except OSError as error:
                raise ControllerError(
                    "workspace_parent_identity_changed", f"{role}: {error}"
                ) from error
            if (
                stat.S_ISLNK(information.st_mode)
                or not stat.S_ISDIR(information.st_mode)
                or resolved != path
                or (
                    information.st_dev,
                    information.st_ino,
                    information.st_mode,
                )
                != (expected.st_dev, expected.st_ino, expected.st_mode)
            ):
                raise ControllerError(
                    "workspace_parent_identity_changed", role
                )
            observed_information.append(information)
        records = self._mount_snapshot(
            timeline, "recheck-workspace-parent-mounts"
        )
        observed_mounts = tuple(
            _select_mount_identity(path, information, records)
            for path, information in zip(
                (acquisition_parent, archive_parent), observed_information
            )
        )
        if observed_mounts != (expected_mount, expected_mount):
            raise ControllerError(
                "active_mount_identity_changed",
                "workspace parent mount identity changed before mutation",
            )

    def _bind_created_directory_mount(
        self,
        path: pathlib.Path,
        expected_mount: MountIdentity,
        timeline: _SafetyTimelineState,
        role: str,
    ) -> tuple[str, int, int, int]:
        try:
            information = os.lstat(path)
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise ControllerError(
                "workspace_child_identity", f"{role}: {error}"
            ) from error
        if (
            stat.S_ISLNK(information.st_mode)
            or not stat.S_ISDIR(information.st_mode)
            or resolved != path
        ):
            raise ControllerError("workspace_child_identity", role)
        records = self._mount_snapshot(
            timeline, f"bind-{role}-mount"
        )
        if _select_mount_identity(path, information, records) != expected_mount:
            raise ControllerError(
                "workspace_child_mount_mismatch",
                f"{role}: {path} is not on the exact active ext4 mount",
            )
        return (
            str(path),
            information.st_dev,
            information.st_ino,
            information.st_mode,
        )

    def create_workspace(
        self, plan: RunPlan, descriptor: Mapping[str, object]
    ) -> RunWorkspace:
        acquisition_parent = pathlib.Path(plan.roots.acquisition_parent)
        archive_parent = pathlib.Path(plan.roots.archive_parent)
        try:
            acquisition_info = os.lstat(acquisition_parent)
            archive_info = os.lstat(archive_parent)
        except OSError as error:
            raise ControllerError("workspace_parent_stat", error) from error
        for path, information in (
            (acquisition_parent, acquisition_info),
            (archive_parent, archive_info),
        ):
            if stat.S_ISLNK(information.st_mode) or not stat.S_ISDIR(information.st_mode):
                raise ControllerError("workspace_parent_type", path)
            try:
                resolved = path.resolve(strict=True)
            except OSError as error:
                raise ControllerError("workspace_parent_resolve", error) from error
            if resolved != path:
                raise ControllerError(
                    "workspace_parent_symlink_ancestor",
                    f"{path} resolves to {resolved}",
                )
        if (acquisition_info.st_dev, acquisition_info.st_ino) == (
            archive_info.st_dev,
            archive_info.st_ino,
        ):
            raise ControllerError(
                "workspace_parent_inode_alias",
                "acquisition and archive parents name the same directory inode",
            )
        mount_snapshot = _read_mountinfo_snapshot()
        acquisition_mount = _select_mount_identity(
            acquisition_parent, acquisition_info, mount_snapshot
        )
        archive_mount = _select_mount_identity(
            archive_parent, archive_info, mount_snapshot
        )
        if acquisition_mount != archive_mount:
            raise ControllerError(
                "active_mount_mismatch",
                "acquisition and archive parents must use one exact ext4 mount identity",
            )
        try:
            acquisition_vfs = os.statvfs(acquisition_parent)
            archive_vfs = os.statvfs(archive_parent)
        except OSError as error:
            raise ControllerError("workspace_parent_statvfs", error) from error
        if (
            type(acquisition_vfs.f_frsize) is not int
            or acquisition_vfs.f_frsize <= 0
            or type(archive_vfs.f_frsize) is not int
            or archive_vfs.f_frsize <= 0
        ):
            raise ControllerError(
                "workspace_allocation_unit",
                f"{acquisition_vfs.f_frsize}/{archive_vfs.f_frsize}",
            )
        try:
            acquisition_preflight = self.process.sample_safety(
                self._safety_spec(
                    plan,
                    acquisition_parent,
                    plan.safety.minimum_free_start_bytes,
                )
            )
            archive_preflight = self.process.sample_safety(
                self._safety_spec(
                    plan,
                    archive_parent,
                    plan.safety.minimum_free_start_bytes,
                )
            )
        except BaseException as error:
            raise self._error("workspace_safety_preflight", error) from error
        global_same_mount_baseline_free_bytes = max(
            int(acquisition_preflight.free_bytes),
            int(archive_preflight.free_bytes),
        )
        timeline = _SafetyTimelineState(
            plan=plan,
            storage_path=acquisition_parent,
            baseline_free_bytes=global_same_mount_baseline_free_bytes,
        )
        self._observe_timeline_sample(
            timeline,
            acquisition_preflight,
            minimum_free_bytes=plan.safety.minimum_free_start_bytes,
        )
        self._observe_timeline_sample(
            timeline,
            archive_preflight,
            minimum_free_bytes=plan.safety.minimum_free_start_bytes,
        )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        scratch: pathlib.Path | None = None
        archive_destination: pathlib.Path | None = None
        archive_stage: pathlib.Path | None = None
        scratch_identity: tuple[str, int, int, int] | None = None
        run_id = ""
        for _ in range(16):
            run_id = f"{plan.case_label}-{timestamp}-{secrets.token_hex(8)}"
            try:
                self.archive.validate_run_id(run_id)
                candidate = acquisition_parent / run_id
                candidate_destination = archive_parent / run_id
                candidate_stage = archive_parent / f".{run_id}.active"
                try:
                    os.lstat(candidate)
                except FileNotFoundError:
                    pass
                except OSError as error:
                    raise ControllerError(
                        "workspace_candidate_stat", error
                    ) from error
                else:
                    continue
                self._require_path_absent(
                    candidate_destination, "archive_destination_exists"
                )
                self._require_path_absent(
                    candidate_stage, "archive_stage_exists"
                )
                self._recheck_workspace_parent_bindings(
                    acquisition_parent,
                    archive_parent,
                    acquisition_info,
                    archive_info,
                    acquisition_mount,
                    timeline,
                )
                self._archive_call(
                    timeline,
                    "create-workspace",
                    self.archive.create_directory_exclusive,
                    candidate,
                )
                scratch = candidate
                archive_destination = candidate_destination
                archive_stage = candidate_stage
                scratch_identity = self._bind_created_directory_mount(
                    scratch, acquisition_mount, timeline, "scratch"
                )
                break
            except BaseException as error:
                if getattr(getattr(error, "code", None), "value", None) == "path_exists":
                    continue
                raise self._error("workspace_create", error) from error
        if (
            scratch is None
            or archive_destination is None
            or archive_stage is None
            or scratch_identity is None
        ):
            raise ControllerError("workspace_collision_limit", plan.case_label)
        work = scratch / "work"
        runtime = scratch / "runtime"
        try:
            self._archive_call(
                timeline,
                "create-archive-stage",
                self.archive.create_directory_exclusive,
                archive_stage,
            )
            archive_stage_identity = self._bind_created_directory_mount(
                archive_stage, acquisition_mount, timeline, "archive-stage"
            )
        except BaseException as error:
            raise self._error("workspace_archive_stage", error) from error
        for directory in (work, runtime):
            try:
                self._archive_call(
                    timeline,
                    "create-workspace-layout",
                    self.archive.create_directory_exclusive,
                    directory,
                )
            except BaseException as error:
                raise self._error("workspace_layout", error) from error
        temporary = runtime / "tmp"
        xdg_cache = runtime / "xdg-cache"
        xdg_config = runtime / "xdg-config"
        xdg_data = runtime / "xdg-data"
        xdg_state = runtime / "xdg-state"
        command_output = runtime / "command-output"
        condition_parent = work / "conditions"
        for directory in (
            temporary,
            xdg_cache,
            xdg_config,
            xdg_data,
            xdg_state,
            command_output,
            condition_parent,
        ):
            try:
                self._archive_call(
                    timeline,
                    "create-runtime-layout",
                    self.archive.create_directory_exclusive,
                    directory,
                )
            except BaseException as error:
                raise self._error("workspace_layout", error) from error

        conditions: dict[str, ConditionPaths] = {}
        archive_directory_count = 0
        for name, _, _ in R0_CONDITIONS:
            root = condition_parent / name
            dependency_root = root / "dependencies"
            log_root = archive_stage / "conditions" / name
            for directory in (root, dependency_root):
                try:
                    self._archive_call(
                        timeline,
                        "create-condition-layout",
                        self.archive.create_directory_exclusive,
                        directory,
                    )
                except BaseException as error:
                    raise self._error("workspace_condition_layout", error) from error
            archive_directory_count += self._ensure_directory(
                archive_stage,
                log_root,
                timeline=timeline,
                maximum_creations=(
                    plan.execution.maximum_archive_directories_excluding_root
                    - archive_directory_count
                ),
            )
            conditions[name] = ConditionPaths(
                name=name,
                root=root,
                build=root / "build",
                dependency_root=dependency_root,
                log_root=log_root,
            )
        for relative in ("inputs", "provenance", "ledger", "commands"):
            archive_directory_count += self._ensure_directory(
                archive_stage,
                archive_stage / relative,
                timeline=timeline,
                maximum_creations=(
                    plan.execution.maximum_archive_directories_excluding_root
                    - archive_directory_count
                ),
            )
        workspace = RunWorkspace(
            run_id=run_id,
            scratch=scratch,
            archive_stage=archive_stage,
            archive_destination=archive_destination,
            source=work / "source",
            temporary=temporary,
            xdg_cache=xdg_cache,
            xdg_config=xdg_config,
            xdg_data=xdg_data,
            xdg_state=xdg_state,
            conditions=types.MappingProxyType(conditions),
        )
        self._recheck_workspace_parent_bindings(
            acquisition_parent,
            archive_parent,
            acquisition_info,
            archive_info,
            acquisition_mount,
            timeline,
        )
        if self._bind_created_directory_mount(
            scratch, acquisition_mount, timeline, "scratch-before-evidence"
        ) != scratch_identity:
            raise ControllerError(
                "workspace_child_identity_changed", "scratch"
            )
        if self._bind_created_directory_mount(
            archive_stage,
            acquisition_mount,
            timeline,
            "archive-stage-before-evidence",
        ) != archive_stage_identity:
            raise ControllerError(
                "workspace_child_identity_changed", "archive-stage"
            )
        try:
            descriptor_bytes = canonical_json_bytes(descriptor)
            if hashlib.sha256(descriptor_bytes).hexdigest() != PROCEDURE_DESCRIPTOR_SHA256:
                raise ControllerError(
                    "procedure_descriptor_hash_mismatch",
                    "workspace descriptor differs",
                )
            initial_payload_bytes = len(plan.canonical_bytes) + len(descriptor_bytes)
            payload_limit = (
                plan.safety.maximum_archive_bytes
                - plan.execution.maximum_manifest_bytes
                - plan.execution.maximum_json_evidence_bytes
            )
            if initial_payload_bytes > payload_limit:
                raise ControllerError(
                    "archive_initial_payload_limit",
                    f"{initial_payload_bytes}>{payload_limit}",
                )
            pointer_value = {
                "archive_stage": str(archive_stage),
                "run_id": run_id,
                "schema": "oai.memprof.r0.archive-stage-pointer/v1",
            }
            pointer_bytes = canonical_json_bytes(pointer_value)
            if len(pointer_bytes) > plan.execution.maximum_json_evidence_bytes:
                raise ControllerError(
                    "archive_stage_pointer_size",
                    f"{len(pointer_bytes)}>{plan.execution.maximum_json_evidence_bytes}",
                )
            initialization_threshold = _prospective_free_threshold_for_files(
                absolute_floor_bytes=plan.safety.minimum_free_during_bytes,
                baseline_free_bytes=global_same_mount_baseline_free_bytes,
                maximum_decline_bytes=(
                    plan.safety.maximum_observed_free_space_decline_bytes
                ),
                prospective_file_bytes=(
                    len(plan.canonical_bytes),
                    len(descriptor_bytes),
                    archive_vfs.f_frsize,
                    len(pointer_bytes),
                ),
                allocation_unit_bytes=archive_vfs.f_frsize,
            )
            initialization_pre_sample = self._force_timeline_sample(
                timeline, minimum_free_bytes=initialization_threshold
            )
            plan_record = self._archive_call(
                timeline,
                "write-plan",
                self.archive.write_bytes_exclusive,
                archive_stage / "inputs/plan.json",
                plan.canonical_bytes,
                max_bytes=MAXIMUM_PLAN_BYTES,
            )
            descriptor_record = self._archive_call(
                timeline,
                "write-procedure",
                self.archive.write_bytes_exclusive,
                archive_stage / "inputs/procedure.json",
                descriptor_bytes,
                max_bytes=plan.execution.maximum_json_evidence_bytes,
            )
            cursor = self._archive_call(
                timeline,
                "initialize-journal",
                self.archive.initialize_journal,
                archive_stage,
            )
            pointer_record = self._archive_call(
                timeline,
                "write-stage-pointer",
                self.archive.write_json_exclusive,
                scratch / "archive-stage-pointer.json",
                pointer_value,
                max_bytes=plan.execution.maximum_json_evidence_bytes,
            )
            initialization_post_sample = self._force_timeline_sample(
                timeline,
                minimum_free_bytes=_prospective_free_threshold(
                    absolute_floor_bytes=plan.safety.minimum_free_during_bytes,
                    baseline_free_bytes=global_same_mount_baseline_free_bytes,
                    maximum_decline_bytes=(
                        plan.safety.maximum_observed_free_space_decline_bytes
                    ),
                    prospective_write_bytes=0,
                    allocation_unit_bytes=archive_vfs.f_frsize,
                ),
            )
        except ControllerError:
            raise
        except BaseException as error:
            raise self._error("workspace_evidence_initialize", error) from error
        initial_archive_entries = archive_directory_count + 3
        if archive_directory_count > (
            plan.execution.maximum_archive_directories_excluding_root
        ) or 3 > (
            plan.execution.maximum_archive_regular_files_excluding_manifest - 1
        ):
            raise ControllerError(
                "archive_population_limit", initial_archive_entries
            )
        if (
            plan_record.size + descriptor_record.size != initial_payload_bytes
            or pointer_record.size != len(pointer_bytes)
            or pointer_record.sha256
            != hashlib.sha256(pointer_bytes).hexdigest()
        ):
            raise ControllerError(
                "archive_initial_payload_identity",
                "retained plan, procedure, or pointer identity differs from its encoded input",
            )
        context = _CoreContext(
            plan=plan,
            workspace=workspace,
            journal_cursor=cursor,
            baseline_free_bytes=global_same_mount_baseline_free_bytes,
            baseline_archive_free_bytes=global_same_mount_baseline_free_bytes,
            archive_payload_bytes=initial_payload_bytes,
            archive_regular_file_count=3,
            archive_directory_count=archive_directory_count,
            archive_entry_count=initial_archive_entries,
            root_identities=(
                (
                    str(acquisition_parent),
                    acquisition_info.st_dev,
                    acquisition_info.st_ino,
                    acquisition_info.st_mode,
                ),
                (
                    str(archive_parent),
                    archive_info.st_dev,
                    archive_info.st_ino,
                    archive_info.st_mode,
                ),
            ),
            scratch_identity=scratch_identity,
            archive_stage_identity=archive_stage_identity,
            thermal_sensor_identities=(),
            active_mount_identity=acquisition_mount,
            acquisition_allocation_unit_bytes=acquisition_vfs.f_frsize,
            archive_allocation_unit_bytes=archive_vfs.f_frsize,
            safety_timeline=timeline,
        )
        self._contexts[str(scratch)] = context
        try:
            sample_values = dict(acquisition_preflight.thermal_millicelsius)
            sensor_provenance = []
            for sensor in plan.safety.mandatory_thermal_sensors:
                sensor_path = pathlib.Path(sensor.path)
                information = os.lstat(sensor_path)
                if stat.S_ISLNK(information.st_mode) or not stat.S_ISREG(
                    information.st_mode
                ):
                    raise ControllerError("thermal_sensor_type", sensor.path)
                sensor_provenance.append(
                    {
                        "device": information.st_dev,
                        "inode": information.st_ino,
                        "label": sensor.label,
                        "minimum_plausible_millicelsius": (
                            sensor.minimum_plausible_millicelsius
                        ),
                        "mode": information.st_mode,
                        "mtime_ns": information.st_mtime_ns,
                        "path": sensor.path,
                        "value_millicelsius": sample_values[sensor.path],
                    }
                )
            context.thermal_sensor_identities = tuple(
                (
                    str(item["path"]),
                    int(item["device"]),
                    int(item["inode"]),
                    int(item["mode"]),
                )
                for item in sensor_provenance
            )
        except (OSError, KeyError) as error:
            raise ControllerError("preflight_provenance", error) from error

        def filesystem_record(
            path: pathlib.Path,
            information: os.stat_result,
            filesystem: os.statvfs_result,
            mount_identity: MountIdentity,
        ) -> dict[str, object]:
            return {
                "available_blocks": filesystem.f_bavail,
                "available_bytes": filesystem.f_bavail * filesystem.f_frsize,
                "block_size_bytes": filesystem.f_bsize,
                "device": information.st_dev,
                "fragment_size_bytes": filesystem.f_frsize,
                "free_blocks": filesystem.f_bfree,
                "free_inodes": filesystem.f_ffree,
                "inode": information.st_ino,
                "mode": information.st_mode,
                "mount": _mount_identity_record(mount_identity),
                "path": str(path),
                "total_blocks": filesystem.f_blocks,
                "total_inodes": filesystem.f_files,
            }

        self.write_json(
            workspace,
            "provenance/preflight-safety.json",
            {
                "acquisition_sample": self._sample_record(acquisition_preflight),
                "archive_sample": self._sample_record(archive_preflight),
                "filesystems": {
                    "acquisition": filesystem_record(
                        acquisition_parent,
                        acquisition_info,
                        acquisition_vfs,
                        acquisition_mount,
                    ),
                    "archive": filesystem_record(
                        archive_parent,
                        archive_info,
                        archive_vfs,
                        archive_mount,
                    ),
                },
                "global_same_mount_baseline_free_bytes": (
                    global_same_mount_baseline_free_bytes
                ),
                "initial_workspace_evidence_safety": {
                    "allocation_rounding": "each_file_ceiling_to_fragment_size",
                    "post_write": self._sample_record(
                        initialization_post_sample
                    ),
                    "prospective_file_bytes": [
                        len(plan.canonical_bytes),
                        len(descriptor_bytes),
                        archive_vfs.f_frsize,
                        len(pointer_bytes),
                    ],
                    "prospective_threshold_free_bytes": initialization_threshold,
                    "prospective_write": self._sample_record(
                        initialization_pre_sample
                    ),
                },
                "schema": "oai.memprof.r0.safety-preflight/v1",
                "thermal_sensors": sensor_provenance,
                "thermal_ceiling_millicelsius": plan.safety.thermal_ceiling_millicelsius,
            },
        )
        return workspace

    @staticmethod
    def _validate_preflight_identities(context: _CoreContext) -> None:
        for role, expected_population in (
            ("root", context.root_identities),
            ("thermal_sensor", context.thermal_sensor_identities),
        ):
            if not expected_population:
                raise ControllerError(f"{role}_identity_population", "empty")
            for path_text, device, inode, mode in expected_population:
                path = pathlib.Path(path_text)
                try:
                    information = os.lstat(path)
                except OSError as error:
                    raise ControllerError(f"{role}_identity_stat", error) from error
                observed = (
                    information.st_dev,
                    information.st_ino,
                    information.st_mode,
                )
                if observed != (device, inode, mode):
                    raise ControllerError(
                        f"{role}_identity_changed",
                        f"{path}: expected {(device, inode, mode)}, observed {observed}",
                    )
                if role == "root" and not stat.S_ISDIR(information.st_mode):
                    raise ControllerError("root_identity_type", path)
                if role == "thermal_sensor" and not stat.S_ISREG(
                    information.st_mode
                ):
                    raise ControllerError("thermal_sensor_identity_type", path)

    def _validate_mount_identity(self, context: _CoreContext) -> None:
        records = self._mount_snapshot(
            context.safety_timeline, "validate-active-mount-identities"
        )

        def exact_directory(
            identity: tuple[str, int, int, int], role: str
        ) -> tuple[pathlib.Path, os.stat_result]:
            path = pathlib.Path(identity[0])
            try:
                information = os.lstat(path)
                resolved = path.resolve(strict=True)
            except OSError as error:
                raise ControllerError(
                    "mount_recheck_stat", f"{role}: {error}"
                ) from error
            if (
                stat.S_ISLNK(information.st_mode)
                or not stat.S_ISDIR(information.st_mode)
                or resolved != path
                or (
                    information.st_dev,
                    information.st_ino,
                    information.st_mode,
                )
                != identity[1:]
            ):
                raise ControllerError(
                    "mount_recheck_identity_changed", role
                )
            if (
                _select_mount_identity(path, information, records)
                != context.active_mount_identity
            ):
                raise ControllerError(
                    "active_mount_identity_changed", role
                )
            return path, information

        for index, identity in enumerate(context.root_identities):
            exact_directory(identity, f"root-{index}")
        scratch_path, _scratch_information = exact_directory(
            context.scratch_identity, "scratch"
        )
        if scratch_path != context.workspace.scratch:
            raise ControllerError(
                "mount_recheck_identity_changed", "scratch path"
            )

        active_population: list[tuple[pathlib.Path, os.stat_result]] = []
        for role, path in (
            ("archive-stage", context.workspace.archive_stage),
            ("archive-final", context.workspace.archive_destination),
        ):
            try:
                information = os.lstat(path)
            except FileNotFoundError:
                continue
            except OSError as error:
                raise ControllerError(
                    "mount_recheck_stat", f"{role}: {error}"
                ) from error
            if stat.S_ISLNK(information.st_mode) or not stat.S_ISDIR(
                information.st_mode
            ):
                raise ControllerError("mount_recheck_identity_changed", role)
            active_population.append((path, information))
        if len(active_population) != 1:
            raise ControllerError(
                "archive_location_population",
                "exactly one of active stage and final archive must exist",
            )
        archive_path, archive_information = active_population[0]
        try:
            archive_resolved = archive_path.resolve(strict=True)
        except OSError as error:
            raise ControllerError("mount_recheck_stat", error) from error
        if (
            archive_resolved != archive_path
            or (
                archive_information.st_dev,
                archive_information.st_ino,
                archive_information.st_mode,
            )
            != context.archive_stage_identity[1:]
            or _select_mount_identity(
                archive_path, archive_information, records
            )
            != context.active_mount_identity
        ):
            raise ControllerError(
                "active_mount_identity_changed",
                f"archive stage/final binding differs at {archive_path}",
            )

    def read_regular(
        self,
        workspace: RunWorkspace,
        path: pathlib.Path,
        maximum_bytes: int,
    ) -> BoundedFileImage:
        context = self._context(workspace)
        try:
            image = self._archive_call(
                context.safety_timeline,
                "workspace-bounded-read",
                self.archive.read_regular_file_bounded,
                path,
                maximum_bytes,
            )
        except BaseException as error:
            raise self._error("bounded_read", error) from error
        return BoundedFileImage(
            data=image.data,
            device=image.device,
            inode=image.inode,
            size=image.size,
            mode=image.mode,
            mtime_ns=image.mtime_ns,
            ctime_ns=image.ctime_ns,
            sha256=image.sha256,
        )

    def progress(self, workspace: RunWorkspace) -> None:
        context = self._context(workspace)
        self._validate_terminal_signal_handlers()
        self._timeline_progress(context.safety_timeline)

    def _reserve_archive_write(
        self,
        context: _CoreContext,
        maximum_new_bytes: int,
        *,
        new_files: int,
        replaced_bytes: int = 0,
        additional_write_bytes: Sequence[int] = (),
    ) -> object:
        if (
            type(maximum_new_bytes) is not int
            or maximum_new_bytes < 0
            or type(replaced_bytes) is not int
            or replaced_bytes < 0
            or new_files not in {0, 1}
            or not isinstance(additional_write_bytes, Sequence)
            or any(
                type(value) is not int or value < 0
                for value in additional_write_bytes
            )
        ):
            raise ControllerError("archive_reservation_argument", "invalid write reservation")
        total_new_bytes = maximum_new_bytes + sum(additional_write_bytes)
        if total_new_bytes > 2**63 - 1:
            raise ControllerError("archive_reservation_overflow", total_new_bytes)
        maximum_delta = max(0, total_new_bytes - replaced_bytes)
        payload_limit = (
            context.plan.safety.maximum_archive_bytes
            - context.plan.execution.maximum_manifest_bytes
            - context.plan.execution.maximum_json_evidence_bytes
        )
        if context.archive_payload_bytes + maximum_delta > payload_limit:
            raise ControllerError(
                "archive_payload_limit",
                f"{context.archive_payload_bytes}+{maximum_delta}>{payload_limit}",
            )
        file_limit = (
            context.plan.execution.maximum_archive_regular_files_excluding_manifest
            - 1
        )
        if context.archive_regular_file_count + new_files > file_limit:
            raise ControllerError(
                "archive_regular_file_limit",
                f"{context.archive_regular_file_count}+{new_files}>{file_limit}",
            )
        threshold = _prospective_free_threshold_for_files(
            absolute_floor_bytes=context.plan.safety.minimum_free_during_bytes,
            baseline_free_bytes=context.baseline_archive_free_bytes,
            maximum_decline_bytes=(
                context.plan.safety.maximum_observed_free_space_decline_bytes
            ),
            prospective_file_bytes=(
                maximum_new_bytes,
                *tuple(additional_write_bytes),
            ),
            allocation_unit_bytes=context.archive_allocation_unit_bytes,
        )
        try:
            return self._force_timeline_sample(
                context.safety_timeline,
                minimum_free_bytes=threshold,
            )
        except BaseException as error:
            raise self._error("archive_write_safety", error) from error

    @staticmethod
    def _account_archive_directories(
        context: _CoreContext, created_directories: int
    ) -> None:
        if created_directories < 0:
            raise ControllerError(
                "archive_directory_accounting", created_directories
            )
        context.archive_directory_count += created_directories
        context.archive_entry_count += created_directories
        directory_limit = (
            context.plan.execution.maximum_archive_directories_excluding_root
        )
        if context.archive_directory_count > directory_limit:
            raise ControllerError(
                "archive_directory_accounting", context.archive_directory_count
            )

    @staticmethod
    def _account_archive_write(
        context: _CoreContext,
        record: object,
        *,
        new_files: int,
        replaced_bytes: int = 0,
    ) -> None:
        context.archive_payload_bytes += int(record.size) - replaced_bytes
        context.archive_regular_file_count += new_files
        context.archive_entry_count += new_files
        payload_limit = (
            context.plan.safety.maximum_archive_bytes
            - context.plan.execution.maximum_manifest_bytes
            - context.plan.execution.maximum_json_evidence_bytes
        )
        if context.archive_payload_bytes < 0 or context.archive_payload_bytes > payload_limit:
            raise ControllerError("archive_payload_accounting", context.archive_payload_bytes)
        if context.archive_regular_file_count > (
            context.plan.execution.maximum_archive_regular_files_excluding_manifest
            - 1
        ):
            raise ControllerError(
                "archive_regular_file_accounting",
                context.archive_regular_file_count,
            )

    @staticmethod
    def _identity_from_record(
        destination: pathlib.Path, record: object
    ) -> EvidenceIdentity:
        return EvidenceIdentity(
            role="",
            path=str(destination),
            bytes=int(record.size),
            lines=int(record.line_count),
            sha256=str(record.sha256),
            device=int(record.device),
            inode=int(record.inode),
        )

    def write_json(
        self, workspace: RunWorkspace, relative_path: str, value: object
    ) -> None:
        context = self._context(workspace)
        try:
            destination = self.archive.safe_join(workspace.archive_stage, relative_path)
            self._account_archive_directories(
                context,
                self._ensure_directory(
                    workspace.archive_stage,
                    destination.parent,
                    timeline=context.safety_timeline,
                    maximum_creations=(
                        context.plan.execution.maximum_archive_directories_excluding_root
                        - context.archive_directory_count
                    ),
                ),
            )
            encoded = canonical_json_bytes(value)
            if len(encoded) > context.plan.execution.maximum_json_evidence_bytes:
                raise ControllerError("json_evidence_limit", relative_path)
            self._reserve_archive_write(context, len(encoded), new_files=1)
            record = self._archive_call(
                context.safety_timeline,
                "write-json-evidence",
                self.archive.write_bytes_exclusive,
                destination,
                encoded,
                max_bytes=context.plan.execution.maximum_json_evidence_bytes,
            )
            self._account_archive_write(context, record, new_files=1)
            self._sample_archive_safety(context, reserve_bytes=0)
        except ControllerError:
            raise
        except BaseException as error:
            raise self._error("json_evidence_write", error) from error

    def copy_scratch_regular(
        self,
        workspace: RunWorkspace,
        source: pathlib.Path,
        destination: pathlib.Path,
        maximum_bytes: int,
    ) -> EvidenceIdentity:
        context = self._context(workspace)
        try:
            relative = destination.relative_to(workspace.scratch)
        except ValueError as error:
            raise ControllerError("scratch_copy_domain", destination) from error
        if relative.parts and relative.parts[0] == "archive-stage":
            raise ControllerError("scratch_copy_archive_alias", destination)
        self._ensure_directory(
            workspace.scratch,
            destination.parent,
            timeline=context.safety_timeline,
        )
        try:
            source_information = os.lstat(source)
        except OSError as error:
            raise ControllerError("scratch_copy_source_stat", error) from error
        if stat.S_ISLNK(source_information.st_mode) or not stat.S_ISREG(
            source_information.st_mode
        ):
            raise ControllerError("scratch_copy_source_type", source)
        if source_information.st_size > maximum_bytes:
            raise ControllerError("scratch_copy_source_limit", source)
        scratch_threshold = _prospective_free_threshold(
            absolute_floor_bytes=context.plan.safety.minimum_free_during_bytes,
            baseline_free_bytes=context.baseline_free_bytes,
            maximum_decline_bytes=(
                context.plan.safety.maximum_observed_free_space_decline_bytes
            ),
            prospective_write_bytes=0,
            allocation_unit_bytes=context.acquisition_allocation_unit_bytes,
        )
        self._validate_preflight_identities(context)
        try:
            self._force_timeline_sample(
                context.safety_timeline,
                minimum_free_bytes=_prospective_free_threshold(
                    absolute_floor_bytes=(
                        context.plan.safety.minimum_free_during_bytes
                    ),
                    baseline_free_bytes=context.baseline_free_bytes,
                    maximum_decline_bytes=(
                        context.plan.safety.maximum_observed_free_space_decline_bytes
                    ),
                    prospective_write_bytes=source_information.st_size,
                    allocation_unit_bytes=(
                        context.acquisition_allocation_unit_bytes
                    ),
                ),
            )
            record = self._archive_call(
                context.safety_timeline,
                "copy-scratch-regular",
                self.archive.copy_file_exclusive,
                source,
                destination,
                max_bytes=maximum_bytes,
            )
            self._force_timeline_sample(
                context.safety_timeline,
                minimum_free_bytes=scratch_threshold,
            )
        except BaseException as error:
            raise self._error("scratch_copy", error) from error
        return self._identity_from_record(destination, record)

    def capture_regular(
        self,
        workspace: RunWorkspace,
        source: pathlib.Path,
        archive_relative_path: str,
        maximum_bytes: int,
    ) -> EvidenceIdentity:
        try:
            destination = self.archive.safe_join(
                workspace.archive_stage, archive_relative_path
            )
        except BaseException as error:
            raise self._error("capture_path", error) from error
        context = self._context(workspace)
        self._account_archive_directories(
            context,
            self._ensure_directory(
                workspace.archive_stage,
                destination.parent,
                timeline=context.safety_timeline,
                maximum_creations=(
                    context.plan.execution.maximum_archive_directories_excluding_root
                    - context.archive_directory_count
                ),
            ),
        )
        try:
            information = os.lstat(source)
            if stat.S_ISLNK(information.st_mode) or not stat.S_ISREG(information.st_mode):
                raise ControllerError("capture_source_type", source)
            if information.st_size > maximum_bytes:
                raise ControllerError("capture_source_limit", source)
            self._reserve_archive_write(
                context, int(information.st_size), new_files=1
            )
            record = self._archive_call(
                context.safety_timeline,
                "capture-archive-regular",
                self.archive.copy_file_exclusive,
                source,
                destination,
                max_bytes=maximum_bytes,
            )
            self._account_archive_write(context, record, new_files=1)
            self._sample_archive_safety(context, reserve_bytes=0)
        except ControllerError:
            raise
        except BaseException as error:
            raise self._error("capture_copy", error) from error
        return self._identity_from_record(destination, record)

    def validate_captured_artifact(
        self,
        workspace: RunWorkspace,
        path: pathlib.Path,
        relative_path: str,
        identity: EvidenceIdentity,
        architecture: str,
        *,
        maximum_bytes: int,
        read_chunk_bytes: int,
    ) -> None:
        context = self._context(workspace)
        try:
            self._archive_call(
                context.safety_timeline,
                "validate-captured-artifact",
                validate_captured_artifact_stream,
                path,
                relative_path,
                identity,
                architecture,
                maximum_bytes=maximum_bytes,
                read_chunk_bytes=read_chunk_bytes,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except (ControllerError, GateError):
            raise
        except BaseException as error:
            raise self._error("artifact_stream_validation", error) from error

    def run_step(
        self,
        workspace: RunWorkspace,
        plan: RunPlan,
        step: str,
        instance: str,
        argv: Sequence[str],
        cwd: pathlib.Path,
        environment: Mapping[str, str],
    ) -> object:
        context = self._context(workspace)
        if context.plan is not plan:
            raise ControllerError("run_plan_identity", instance)
        self._validate_preflight_identities(context)
        if step not in REQUIRED_STEP_LIMIT_NAMES or not _SAFE_LABEL.fullmatch(instance):
            raise ControllerError("run_step_identity", f"{step}/{instance}")
        output_root = workspace.scratch / "runtime" / "command-output"
        stdout_path = output_root / f"{instance}.stdout.bin"
        stderr_path = output_root / f"{instance}.stderr.bin"
        limit = plan.execution.step_limits[step]
        process_limits = self.process.CommandLimits(
            wall_seconds=limit.wall_seconds,
            stdout_bytes=limit.stdout_bytes,
            stderr_bytes=limit.stderr_bytes,
            cleanup_seconds=plan.execution.cleanup_grace_seconds,
            max_proc_entries=plan.execution.maximum_proc_entries,
            max_observed_live_descendants=(
                plan.execution.maximum_observed_live_descendants
            ),
            max_descendant_identities=plan.execution.maximum_descendant_identities,
            read_chunk_bytes=plan.execution.read_chunk_bytes,
        )
        scratch_threshold = max(
            plan.safety.minimum_free_during_bytes,
            context.baseline_free_bytes
            - plan.safety.maximum_observed_free_space_decline_bytes,
        )
        safety = self._safety_spec(plan, workspace.scratch, scratch_threshold)
        spec = self.process.CommandSpec(
            argv=tuple(argv),
            cwd=cwd,
            environment=dict(environment),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        self._validate_terminal_signal_handlers()
        self._force_timeline_sample(
            context.safety_timeline,
            minimum_free_bytes=scratch_threshold,
        )
        caller_signal_mask_before = self._current_signal_mask_numbers()
        if set(caller_signal_mask_before).intersection(
            self._supported_cancellation_signals
        ):
            raise ControllerError(
                "signal_mask_precondition",
                "caller blocked at least one supported cancellation signal",
            )
        started_utc_ns = time.time_ns()
        result: object | None = None
        runner_error: BaseException | None = None
        try:
            self._validate_terminal_signal_handlers()
            result = self.process.run_bounded_command(
                spec,
                limits=process_limits,
                safety=safety,
            )
        except BaseException as error:
            runner_error = error
        ended_utc_ns = time.time_ns()
        try:
            if result is None:
                self._force_timeline_sample(
                    context.safety_timeline,
                    minimum_free_bytes=scratch_threshold,
                )
            else:
                process_samples = getattr(result, "safety_samples", None)
                if type(process_samples) is not tuple or not process_samples:
                    raise ControllerError(
                        "process_safety_sample_contract",
                        "runner returned no exact safety-sample tuple",
                    )
                for sample in process_samples:
                    self._observe_timeline_sample(
                        context.safety_timeline,
                        sample,
                        minimum_free_bytes=scratch_threshold,
                    )
        except BaseException as timeline_error:
            if runner_error is not None:
                runner_error.add_note(
                    "command global-safety timeline failed: "
                    f"{type(timeline_error).__name__}: {_bounded_text(timeline_error)}"
                )
            else:
                runner_error = timeline_error
        caller_signal_mask_after: tuple[int, ...] = ()
        try:
            caller_signal_mask_after = self._current_signal_mask_numbers()
            if caller_signal_mask_after != caller_signal_mask_before:
                raise ControllerError(
                    "signal_mask_restoration",
                    f"before={caller_signal_mask_before} after={caller_signal_mask_after}",
                )
        except BaseException as mask_error:
            if runner_error is not None:
                runner_error.add_note(
                    "post-command signal-mask observation failed: "
                    f"{type(mask_error).__name__}: {_bounded_text(mask_error)}"
                )
            else:
                runner_error = mask_error
        captures: dict[str, EvidenceIdentity] = {}
        capture_failures: list[tuple[str, BaseException]] = []
        for role, raw_path, maximum in (
            ("stdout", stdout_path, limit.stdout_bytes),
            ("stderr", stderr_path, limit.stderr_bytes),
        ):
            try:
                os.lstat(raw_path)
            except FileNotFoundError:
                if runner_error is None:
                    capture_failures.append(
                        (role, ControllerError("command_output_missing", raw_path))
                    )
                continue
            except (KeyboardInterrupt, SystemExit) as capture_cancellation:
                if isinstance(runner_error, (KeyboardInterrupt, SystemExit)):
                    runner_error.add_note(
                        "repeated cancellation while locating retained "
                        f"{role}: {type(capture_cancellation).__name__}"
                    )
                    raise runner_error
                raise
            except BaseException as error:
                capture_failures.append((role, error))
                continue
            try:
                captures[role] = self.capture_regular(
                    workspace,
                    raw_path,
                    f"commands/{instance}.{role}.bin",
                    maximum,
                )
            except (KeyboardInterrupt, SystemExit) as capture_cancellation:
                if isinstance(runner_error, (KeyboardInterrupt, SystemExit)):
                    runner_error.add_note(
                        "repeated cancellation while capturing retained "
                        f"{role}: {type(capture_cancellation).__name__}"
                    )
                    raise runner_error
                raise
            except BaseException as error:
                capture_failures.append((role, error))
        if runner_error is not None:
            try:
                self.write_json(
                    workspace,
                    f"ledger/{instance}.runner-exception.json",
                    {
                        "argv": list(argv),
                        "captures": {
                            role: dataclasses.asdict(identity)
                            for role, identity in sorted(captures.items())
                        },
                        "cwd": str(cwd),
                        "ended_utc_ns": ended_utc_ns,
                        "environment": dict(sorted(environment.items())),
                        "exception": {
                            "message": _bounded_text(runner_error),
                            "type": type(runner_error).__name__,
                        },
                        "instance": instance,
                        "schema": "oai.memprof.r0.runner-exception/v1",
                        "started_utc_ns": started_utc_ns,
                        "step": step,
                    },
                )
            except (KeyboardInterrupt, SystemExit) as evidence_error:
                if isinstance(runner_error, (KeyboardInterrupt, SystemExit)):
                    runner_error.add_note(
                        "repeated cancellation while retaining runner error: "
                        f"{type(evidence_error).__name__}"
                    )
                    raise runner_error
                evidence_error.add_note(
                    "runner failed before cancellation: "
                    f"{type(runner_error).__name__}: {_bounded_text(runner_error)}"
                )
                raise
            except BaseException as evidence_error:
                runner_error.add_note(
                    "runner-exception evidence write failed: "
                    f"{type(evidence_error).__name__}: {_bounded_text(evidence_error)}"
                )
            for role, capture_error in capture_failures:
                runner_error.add_note(
                    f"{role} capture failed: {type(capture_error).__name__}: "
                    f"{_bounded_text(capture_error)}"
                )
            if isinstance(runner_error, self.process.ProcessError):
                raw_code = getattr(runner_error.code, "value", None)
                if not isinstance(raw_code, str) or not raw_code:
                    raw_code = "process_core_failure"
                wrapped = ControllerError(raw_code, _bounded_text(runner_error))
                for note in getattr(runner_error, "__notes__", ()):
                    wrapped.add_note(_bounded_text(note))
                raise wrapped from runner_error
            raise runner_error
        if capture_failures:
            role, capture_error = capture_failures[0]
            raise ControllerError(
                "command_output_capture",
                f"{role}: {type(capture_error).__name__}: {_bounded_text(capture_error)}",
            ) from capture_error
        if result is None or set(captures) != {"stdout", "stderr"}:
            raise ControllerError("command_output_capture", "incomplete output capture population")
        if type(result) is not self.process.ProcessResult:
            raise ControllerError(
                "process_result_contract",
                f"expected ProcessResult, observed {type(result).__name__}",
            )

        post_sample: object | None = None
        local_code: str | None = None
        local_detail: str | None = None
        maximum_gap: int | None = None
        try:
            self._validate_preflight_identities(context)
        except ControllerError as error:
            local_code = error.code
            local_detail = error.detail

        def valid_signal_population(value: object) -> bool:
            return bool(
                type(value) is tuple
                and all(type(number) is int and number > 0 for number in value)
                and tuple(sorted(value)) == value
                and len(value) == len(set(value))
            )

        def valid_error_pair(code: object, detail: object) -> bool:
            if code is None:
                return detail is None
            return bool(
                type(code) is self.process.ProcessErrorCode
                and type(detail) is str
                and detail
            )

        def valid_safety_sample(sample: object) -> bool:
            return bool(
                type(sample) is self.process.SafetySample
                and type(sample.monotonic_ns) is int
                and sample.monotonic_ns >= 0
                and type(sample.free_bytes) is int
                and sample.free_bytes >= 0
                and type(sample.thermal_millicelsius) is tuple
                and all(
                    type(item) is tuple
                    and len(item) == 2
                    and type(item[0]) is str
                    and type(item[1]) is int
                    for item in sample.thermal_millicelsius
                )
            )

        def exact_identity_population_shape(value: object) -> bool:
            return bool(
                type(value) is tuple
                and all(
                    type(identity) is tuple
                    and len(identity) == 2
                    and type(identity[0]) is int
                    and identity[0] > 0
                    and type(identity[1]) is int
                    and identity[1] > 0
                    for identity in value
                )
            )

        def valid_descendant_scan_contract() -> bool:
            scan_count = result.descendant_scan_count
            first_scan = result.first_descendant_scan_monotonic_ns
            last_scan = result.last_descendant_scan_monotonic_ns
            scan_gap = result.maximum_observed_descendant_scan_gap_ns
            if (
                type(scan_count) is not int
                or scan_count < 1
                or type(first_scan) is not int
                or type(last_scan) is not int
                or not (
                    result.started_monotonic_ns
                    <= first_scan
                    <= last_scan
                    <= result.ended_monotonic_ns
                )
            ):
                return False
            if scan_count == 1:
                return first_scan == last_scan and scan_gap is None
            if type(scan_gap) is not int:
                return False
            span_ns = last_scan - first_scan
            minimum_possible_maximum_gap_ns = (
                span_ns + scan_count - 2
            ) // (scan_count - 1)
            return minimum_possible_maximum_gap_ns <= scan_gap <= span_ns

        caller_signal_mask = result.caller_signal_mask_numbers
        deferred_supported_signals = result.deferred_supported_signal_numbers
        result_contract_valid = bool(
            type(result.argv) is tuple
            and all(type(argument) is str for argument in result.argv)
            and result.argv == spec.argv
            and type(result.cwd) is str
            and result.cwd == str(cwd)
            and type(result.started_monotonic_ns) is int
            and type(result.ended_monotonic_ns) is int
            and 0 <= result.started_monotonic_ns <= result.ended_monotonic_ns
            and type(result.pid) is int
            and result.pid > 0
            and type(result.returncode) is int
            and type(result.stdout_bytes) is int
            and 0 <= result.stdout_bytes <= limit.stdout_bytes
            and type(result.stderr_bytes) is int
            and 0 <= result.stderr_bytes <= limit.stderr_bytes
            and type(result.stdout_sha256) is str
            and _SHA256.fullmatch(result.stdout_sha256) is not None
            and type(result.stderr_sha256) is str
            and _SHA256.fullmatch(result.stderr_sha256) is not None
            and type(result.safety_samples) is tuple
            and bool(result.safety_samples)
            and all(valid_safety_sample(sample) for sample in result.safety_samples)
            and valid_error_pair(result.failure_code, result.failure_detail)
            and valid_error_pair(
                result.cleanup_failure_code, result.cleanup_failure_detail
            )
            and (
                result.cleanup_failure_code is None
                or result.failure_code is not None
            )
            and type(result.kill_attempted) is bool
            and type(result.cleanup_complete) is bool
            and (
                result.cleanup_complete
                == (result.cleanup_failure_code is None)
            )
            and valid_signal_population(caller_signal_mask)
            and caller_signal_mask == caller_signal_mask_before
            and caller_signal_mask_after == caller_signal_mask_before
            and not set(caller_signal_mask).intersection(
                self._supported_cancellation_signals
            )
            and result.signal_mask_restored is True
            and valid_signal_population(deferred_supported_signals)
            and set(deferred_supported_signals).issubset(
                self._supported_cancellation_signals
            )
            and type(result.containment_mode) is str
            and result.containment_mode == self.process.CONTAINMENT_MODEL
            and type(result.containment_caller_thread_count) is int
            and result.containment_caller_thread_count == 1
            and type(result.containment_preexisting_child_count) is int
            and result.containment_preexisting_child_count == 0
            and type(result.containment_subreaper_previously_enabled) is bool
            and type(result.containment_subreaper_restored) is bool
            and type(result.maximum_observed_live_descendant_count) is int
            and result.maximum_observed_live_descendant_count >= 0
            and type(result.observed_descendant_identity_lower_bound) is int
            and result.observed_descendant_identity_lower_bound >= 0
            and type(result.observed_descendant_identity_complete) is bool
            and exact_identity_population_shape(
                result.observed_descendant_identities
            )
            and type(result.observed_descendant_identities_truncated) is bool
            and result.maximum_observed_live_descendant_count
            <= result.observed_descendant_identity_lower_bound
            and type(result.unexpected_descendant_identity_lower_bound) is int
            and result.unexpected_descendant_identity_lower_bound >= 0
            and type(result.unexpected_descendant_identity_complete) is bool
            and type(result.unexpected_descendant_pids) is tuple
            and all(
                type(pid) is int and pid > 0
                for pid in result.unexpected_descendant_pids
            )
            and exact_identity_population_shape(
                result.unexpected_descendant_identities
            )
            and type(result.unexpected_descendant_identities_truncated) is bool
            and valid_descendant_scan_contract()
        )
        if not result_contract_valid and local_code is None:
            local_code = "process_result_contract"
            local_detail = "runner argv/CWD/status/timestamp/output contract differs"
        try:
            post_sample = self._force_timeline_sample(
                context.safety_timeline,
                minimum_free_bytes=scratch_threshold,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:
            local_code = "post_command_safety"
            local_detail = _bounded_text(error)
        samples = list(result.safety_samples)
        if post_sample is not None:
            samples.append(post_sample)
        expected_sensors = tuple(
            sensor.path for sensor in plan.safety.mandatory_thermal_sensors
        )
        if not samples:
            local_code, local_detail = "safety_sample_missing", "no safety sample returned"
        else:
            observed_times: list[int] = []
            for sample in samples:
                observed_times.append(int(sample.monotonic_ns))
                if tuple(path for path, _ in sample.thermal_millicelsius) != expected_sensors:
                    local_code, local_detail = "safety_sensor_population", "mandatory sensor population differs"
                    break
                if any(
                    value >= plan.safety.thermal_ceiling_millicelsius
                    for _, value in sample.thermal_millicelsius
                ):
                    local_code, local_detail = "thermal_limit", "sample reached the declared ceiling"
                    break
                if sample.free_bytes < scratch_threshold:
                    local_code, local_detail = "free_space_limit", "sample is below dynamic scratch threshold"
                    break
            if (
                observed_times
                and observed_times[0] <= result.started_monotonic_ns
                and observed_times[-1] >= result.ended_monotonic_ns
                and all(
                    right >= left
                    for left, right in zip(observed_times, observed_times[1:])
                )
            ):
                maximum_gap = max(
                    (
                        right - left
                        for left, right in zip(observed_times, observed_times[1:])
                    ),
                    default=0,
                )
                allowed_gap = plan.safety.maximum_poll_gap_milliseconds * 1_000_000
                if maximum_gap > allowed_gap and local_code is None:
                    local_code = "maximum_poll_gap"
                    local_detail = f"observed {maximum_gap} ns above {allowed_gap} ns"
            elif local_code is None:
                local_code, local_detail = "safety_timeline", "safety timestamps are not ordered within command"
        if (
            captures["stdout"].bytes != result.stdout_bytes
            or captures["stdout"].sha256 != result.stdout_sha256
            or captures["stderr"].bytes != result.stderr_bytes
            or captures["stderr"].sha256 != result.stderr_sha256
        ) and local_code is None:
            local_code, local_detail = "command_output_identity", "runner and retained output differ"
        scan_count = result.descendant_scan_count
        first_scan = result.first_descendant_scan_monotonic_ns
        last_scan = result.last_descendant_scan_monotonic_ns
        scan_gap = result.maximum_observed_descendant_scan_gap_ns
        scan_contract_valid = valid_descendant_scan_contract()
        if not scan_contract_valid and local_code is None:
            local_code = "descendant_scan_contract"
            local_detail = "descendant-scan population/timeline is inconsistent"
        allowed_gap = plan.safety.maximum_poll_gap_milliseconds * 1_000_000
        if (
            scan_contract_valid
            and scan_gap is not None
            and scan_gap > allowed_gap
            and local_code is None
        ):
            local_code = "maximum_descendant_scan_gap"
            local_detail = f"observed {scan_gap} ns above {allowed_gap} ns"
        if (
            result.maximum_observed_live_descendant_count
            > plan.execution.maximum_observed_live_descendants
            and local_code is None
        ):
            local_code = "observed_descendant_population_bound"
            local_detail = "runner reported an observed population above the plan bound"
        observed_identities_value = result.observed_descendant_identities
        unexpected_identities_value = result.unexpected_descendant_identities
        observed_identities = (
            observed_identities_value
            if type(observed_identities_value) is tuple
            else ()
        )
        unexpected_identities = (
            unexpected_identities_value
            if type(unexpected_identities_value) is tuple
            else ()
        )

        def valid_identity_population(identities: tuple[object, ...]) -> bool:
            if not all(
                type(identity) is tuple
                and len(identity) == 2
                and type(identity[0]) is int
                and identity[0] > 0
                and type(identity[1]) is int
                and identity[1] > 0
                for identity in identities
            ):
                return False
            return bool(
                len(identities) == len(set(identities))
                and tuple(sorted(identities)) == identities
            )

        def valid_identity_summary(
            identities: tuple[object, ...],
            lower_bound: object,
            complete: object,
            truncated: object,
        ) -> bool:
            if (
                type(lower_bound) is not int
                or lower_bound < 0
                or type(complete) is not bool
                or type(truncated) is not bool
                or not valid_identity_population(identities)
                or lower_bound < len(identities)
            ):
                return False
            if complete:
                return bool(
                    (not truncated and lower_bound == len(identities))
                    or (truncated and lower_bound > len(identities))
                )
            return bool(truncated and lower_bound > len(identities))

        identity_contract_valid = bool(
            type(observed_identities_value) is tuple
            and type(unexpected_identities_value) is tuple
            and type(result.maximum_observed_live_descendant_count) is int
            and result.maximum_observed_live_descendant_count >= 0
            and result.maximum_observed_live_descendant_count
            <= plan.execution.maximum_observed_live_descendants
            and result.maximum_observed_live_descendant_count
            <= result.observed_descendant_identity_lower_bound
            and valid_identity_summary(
                observed_identities,
                result.observed_descendant_identity_lower_bound,
                result.observed_descendant_identity_complete,
                result.observed_descendant_identities_truncated,
            )
            and valid_identity_summary(
                unexpected_identities,
                result.unexpected_descendant_identity_lower_bound,
                result.unexpected_descendant_identity_complete,
                result.unexpected_descendant_identities_truncated,
            )
            and result.unexpected_descendant_identity_lower_bound
            <= result.observed_descendant_identity_lower_bound
            and type(result.unexpected_descendant_pids) is tuple
            and all(
                type(pid) is int and pid > 0
                for pid in result.unexpected_descendant_pids
            )
            and tuple(sorted(result.unexpected_descendant_pids))
            == result.unexpected_descendant_pids
            and len(result.unexpected_descendant_pids)
            == len(set(result.unexpected_descendant_pids))
            and tuple(result.unexpected_descendant_pids)
            == tuple(identity[0] for identity in unexpected_identities)
            and (
                result.observed_descendant_identities_truncated
                or set(unexpected_identities).issubset(observed_identities)
            )
        )
        if not identity_contract_valid and local_code is None:
            local_code = "descendant_identity_contract"
            local_detail = (
                "observed-live, cumulative-identity, and residual ledgers differ"
            )
        return _CoreCommandObservation(
            spec=spec,
            result=result,
            stdout_capture=captures["stdout"],
            stderr_capture=captures["stderr"],
            started_utc_ns=started_utc_ns,
            ended_utc_ns=ended_utc_ns,
            maximum_observed_poll_gap_ns=maximum_gap,
            local_failure_code=local_code,
            local_failure_detail=local_detail,
            post_safety_sample=post_sample,
            caller_signal_mask_before=caller_signal_mask_before,
            caller_signal_mask_after=caller_signal_mask_after,
            maximum_descendant_identities=(
                plan.execution.maximum_descendant_identities
            ),
            safety_timeline=context.safety_timeline,
        )

    def command_stdout(self, result: object, maximum_bytes: int) -> bytes:
        if not isinstance(result, _CoreCommandObservation):
            raise ControllerError("command_result_type", "stdout")
        return self._read_command_capture(
            result, result.stdout_capture, maximum_bytes
        )

    def command_stderr(self, result: object, maximum_bytes: int) -> bytes:
        if not isinstance(result, _CoreCommandObservation):
            raise ControllerError("command_result_type", "stderr")
        return self._read_command_capture(
            result, result.stderr_capture, maximum_bytes
        )

    def _read_command_capture(
        self,
        result: _CoreCommandObservation,
        identity: EvidenceIdentity,
        maximum_bytes: int,
    ) -> bytes:
        if identity.bytes > maximum_bytes:
            raise ControllerError("command_capture_read_limit", identity.path)
        try:
            image = self._archive_call(
                result.safety_timeline,
                "read-command-capture",
                self.archive.read_regular_file_bounded,
                pathlib.Path(identity.path),
                maximum_bytes,
            )
        except BaseException as error:
            raise self._error("command_capture_read", error) from error
        logical_lines = image.data.count(b"\n") + int(
            bool(image.data) and not image.data.endswith(b"\n")
        )
        if (
            image.device != identity.device
            or image.inode != identity.inode
            or image.size != identity.bytes
            or image.sha256 != identity.sha256
            or logical_lines != identity.lines
        ):
            raise ControllerError("command_capture_identity", identity.path)
        return image.data

    def command_succeeded(self, result: object) -> bool:
        if not isinstance(result, _CoreCommandObservation):
            return False
        process_result = result.result
        return bool(
            result.local_failure_code is None
            and type(process_result) is self.process.ProcessResult
            and type(process_result.argv) is tuple
            and all(type(argument) is str for argument in process_result.argv)
            and type(process_result.cwd) is str
            and type(process_result.pid) is int
            and process_result.pid > 0
            and process_result.failure_code is None
            and process_result.failure_detail is None
            and process_result.cleanup_failure_code is None
            and process_result.cleanup_failure_detail is None
            and type(process_result.returncode) is int
            and process_result.returncode == 0
            and process_result.kill_attempted is False
            and process_result.cleanup_complete is True
            and type(process_result.descendant_scan_count) is int
            and process_result.descendant_scan_count >= 1
            and type(process_result.maximum_observed_live_descendant_count) is int
            and process_result.maximum_observed_live_descendant_count >= 0
            and process_result.observed_descendant_identity_complete is True
            and type(process_result.observed_descendant_identity_lower_bound)
            is int
            and process_result.maximum_observed_live_descendant_count
            <= process_result.observed_descendant_identity_lower_bound
            and process_result.observed_descendant_identity_lower_bound
            <= result.maximum_descendant_identities
            and type(process_result.observed_descendant_identities) is tuple
            and type(process_result.observed_descendant_identities_truncated) is bool
            and type(process_result.unexpected_descendant_identity_lower_bound)
            is int
            and process_result.unexpected_descendant_identity_lower_bound == 0
            and process_result.unexpected_descendant_identity_complete is True
            and type(process_result.unexpected_descendant_pids) is tuple
            and type(process_result.unexpected_descendant_identities) is tuple
            and process_result.unexpected_descendant_pids == ()
            and process_result.unexpected_descendant_identities == ()
            and process_result.unexpected_descendant_identities_truncated is False
            and process_result.containment_mode == self.process.CONTAINMENT_MODEL
            and type(process_result.containment_caller_thread_count) is int
            and process_result.containment_caller_thread_count == 1
            and type(process_result.containment_preexisting_child_count) is int
            and process_result.containment_preexisting_child_count == 0
            and type(process_result.containment_subreaper_previously_enabled) is bool
            and process_result.containment_subreaper_restored is True
            and process_result.signal_mask_restored is True
            and process_result.caller_signal_mask_numbers
            == result.caller_signal_mask_before
            == result.caller_signal_mask_after
            and process_result.deferred_supported_signal_numbers == ()
        )

    def command_record(self, result: object) -> Mapping[str, object]:
        if not isinstance(result, _CoreCommandObservation):
            raise ControllerError("command_result_type", "record")
        process_result = result.result

        def code(value: object) -> str | None:
            return None if value is None else str(value.value)

        stdout_relative = f"commands/{pathlib.Path(result.stdout_capture.path).name}"
        stderr_relative = f"commands/{pathlib.Path(result.stderr_capture.path).name}"
        return {
            "argv": list(process_result.argv),
            "cleanup": {
                "complete": process_result.cleanup_complete,
                "containment_caller_thread_count": (
                    process_result.containment_caller_thread_count
                ),
                "containment_mode": process_result.containment_mode,
                "containment_preexisting_child_count": (
                    process_result.containment_preexisting_child_count
                ),
                "containment_subreaper_previously_enabled": (
                    process_result.containment_subreaper_previously_enabled
                ),
                "containment_subreaper_restored": (
                    process_result.containment_subreaper_restored
                ),
                "descendant_scan_count": process_result.descendant_scan_count,
                "first_descendant_scan_monotonic_ns": (
                    process_result.first_descendant_scan_monotonic_ns
                ),
                "failure_code": code(process_result.cleanup_failure_code),
                "failure_detail": process_result.cleanup_failure_detail,
                "kill_attempted": process_result.kill_attempted,
                "last_descendant_scan_monotonic_ns": (
                    process_result.last_descendant_scan_monotonic_ns
                ),
                "maximum_observed_live_descendant_count": (
                    process_result.maximum_observed_live_descendant_count
                ),
                "maximum_observed_descendant_scan_gap_ns": (
                    process_result.maximum_observed_descendant_scan_gap_ns
                ),
                "signal_mask": {
                    "caller_after": list(result.caller_signal_mask_after),
                    "caller_before": list(result.caller_signal_mask_before),
                    "deferred_supported_signals": list(
                        process_result.deferred_supported_signal_numbers
                    ),
                    "process_caller": list(
                        process_result.caller_signal_mask_numbers
                    ),
                    "restored": process_result.signal_mask_restored,
                    "supported_cancellation_signals": list(
                        self._supported_cancellation_signals
                    ),
                },
                "observed_descendant_identities": [
                    {"pid": pid, "start_time_ticks": start_time_ticks}
                    for pid, start_time_ticks in (
                        process_result.observed_descendant_identities
                    )
                ],
                "observed_descendant_identities_truncated": (
                    process_result.observed_descendant_identities_truncated
                ),
                "observed_descendant_identity_complete": (
                    process_result.observed_descendant_identity_complete
                ),
                "observed_descendant_identity_lower_bound": (
                    process_result.observed_descendant_identity_lower_bound
                ),
                "unexpected_descendant_identity_complete": (
                    process_result.unexpected_descendant_identity_complete
                ),
                "unexpected_descendant_identity_lower_bound": (
                    process_result.unexpected_descendant_identity_lower_bound
                ),
                "unexpected_descendant_identities": [
                    {"pid": pid, "start_time_ticks": start_time_ticks}
                    for pid, start_time_ticks in (
                        process_result.unexpected_descendant_identities
                    )
                ],
                "unexpected_descendant_identities_truncated": (
                    process_result.unexpected_descendant_identities_truncated
                ),
                "unexpected_descendant_pids": list(process_result.unexpected_descendant_pids),
            },
            "cwd": process_result.cwd,
            "ended_monotonic_ns": process_result.ended_monotonic_ns,
            "ended_utc_ns": result.ended_utc_ns,
            "environment": dict(sorted(result.spec.environment.items())),
            "local_failure_code": result.local_failure_code,
            "local_failure_detail": result.local_failure_detail,
            "maximum_observed_poll_gap_ns": result.maximum_observed_poll_gap_ns,
            "pid": process_result.pid,
            "post_safety_sample": (
                None
                if result.post_safety_sample is None
                else self._sample_record(result.post_safety_sample)
            ),
            "returncode": process_result.returncode,
            "runtime_failure_code": code(process_result.failure_code),
            "runtime_failure_detail": process_result.failure_detail,
            "safety_samples": [
                self._sample_record(sample) for sample in process_result.safety_samples
            ],
            "started_monotonic_ns": process_result.started_monotonic_ns,
            "started_utc_ns": result.started_utc_ns,
            "stderr_bytes": process_result.stderr_bytes,
            "stderr_acquisition_path": str(result.spec.stderr_path),
            "stderr_capture_bytes": result.stderr_capture.bytes,
            "stderr_capture_path_during_acquisition": result.stderr_capture.path,
            "stderr_capture_sha256": result.stderr_capture.sha256,
            "stderr_device_during_acquisition": result.stderr_capture.device,
            "stderr_inode_during_acquisition": result.stderr_capture.inode,
            "stderr_lines": result.stderr_capture.lines,
            "stderr_path": stderr_relative,
            "stderr_sha256": process_result.stderr_sha256,
            "stdout_bytes": process_result.stdout_bytes,
            "stdout_acquisition_path": str(result.spec.stdout_path),
            "stdout_capture_bytes": result.stdout_capture.bytes,
            "stdout_capture_path_during_acquisition": result.stdout_capture.path,
            "stdout_capture_sha256": result.stdout_capture.sha256,
            "stdout_device_during_acquisition": result.stdout_capture.device,
            "stdout_inode_during_acquisition": result.stdout_capture.inode,
            "stdout_lines": result.stdout_capture.lines,
            "stdout_path": stdout_relative,
            "stdout_sha256": process_result.stdout_sha256,
        }

    def checkpoint(
        self, workspace: RunWorkspace, state: Mapping[str, object]
    ) -> None:
        context = self._context(workspace)
        phase = state.get("phase")
        if not isinstance(phase, str) or phase not in _STATE_TRANSITIONS:
            raise ControllerError("state_phase", phase)
        failure_checkpoint_bypass = bool(
            phase == "READY_TO_SEAL"
            and context.last_phase in _NONTERMINAL_PHASES
            and isinstance(state.get("failure_checkpoint_problem"), dict)
            and (
                state.get("failure") is not None
                or state.get("interrupted") is True
            )
        )
        if (
            phase not in _STATE_TRANSITIONS[context.last_phase]
            and not failure_checkpoint_bypass
        ):
            raise ControllerError(
                "state_transition", f"{context.last_phase!r} -> {phase!r}"
            )
        if context.journal_cursor.next_sequence >= context.plan.execution.maximum_journal_entries:
            raise ControllerError("journal_entry_limit", phase)
        try:
            encoded_state = canonical_json_bytes(dict(state))
            if len(encoded_state) > context.plan.execution.maximum_state_bytes:
                raise ControllerError("state_size_limit", phase)
            state_path = workspace.archive_stage / "state.json"
            journal_path = workspace.archive_stage / "journal.jsonl"
            try:
                state_before = os.lstat(state_path)
                if stat.S_ISLNK(state_before.st_mode) or not stat.S_ISREG(
                    state_before.st_mode
                ):
                    raise ControllerError("state_type", state_path)
                replaced_bytes = int(state_before.st_size)
                new_state_file = 0
            except FileNotFoundError:
                replaced_bytes = 0
                new_state_file = 1
            journal_before = os.lstat(journal_path)
            if stat.S_ISLNK(journal_before.st_mode) or not stat.S_ISREG(
                journal_before.st_mode
            ):
                raise ControllerError("journal_type", journal_path)
            self._reserve_archive_write(
                context,
                len(encoded_state),
                new_files=new_state_file,
                replaced_bytes=replaced_bytes,
                additional_write_bytes=(
                    context.plan.execution.maximum_journal_entry_bytes + 1,
                ),
            )
            state_record = self._archive_call(
                context.safety_timeline,
                "write-state-checkpoint",
                self.archive.write_state_checkpoint,
                workspace.archive_stage,
                dict(state),
                max_bytes=context.plan.execution.maximum_state_bytes,
            )
            context.journal_cursor = self._archive_call(
                context.safety_timeline,
                "append-state-journal",
                self.archive.append_journal,
                workspace.archive_stage,
                {"event": "STATE_CHECKPOINT", "state": dict(state)},
                cursor=context.journal_cursor,
                max_entry_bytes=context.plan.execution.maximum_journal_entry_bytes,
            )
            journal_after = os.lstat(journal_path)
            journal_growth = journal_after.st_size - journal_before.st_size
            if journal_growth <= 0:
                raise ControllerError("journal_growth", phase)
            context.archive_payload_bytes += (
                state_record.size - replaced_bytes + journal_growth
            )
            context.archive_regular_file_count += new_state_file
            context.archive_entry_count += new_state_file
            payload_limit = (
                context.plan.safety.maximum_archive_bytes
                - context.plan.execution.maximum_manifest_bytes
                - context.plan.execution.maximum_json_evidence_bytes
            )
            if context.archive_payload_bytes > payload_limit:
                raise ControllerError(
                    "archive_payload_accounting", context.archive_payload_bytes
                )
            self._sample_archive_safety(context, reserve_bytes=0)
        except BaseException as error:
            if isinstance(error, ControllerError):
                raise
            raise self._error("state_checkpoint", error) from error
        context.last_phase = phase

    @staticmethod
    def _manifest_arguments(plan: RunPlan) -> dict[str, object]:
        payload_limit = (
            plan.safety.maximum_archive_bytes
            - plan.execution.maximum_manifest_bytes
        )
        return {
            "manifest_relative_path": "manifest.json",
            "max_regular_files_excluding_manifest": (
                plan.execution.maximum_archive_regular_files_excluding_manifest
            ),
            "max_directories_excluding_root": (
                plan.execution.maximum_archive_directories_excluding_root
            ),
            "max_regular_file_bytes": payload_limit,
            "max_total_regular_file_bytes": payload_limit,
        }

    @staticmethod
    def _archive_population(
        root: pathlib.Path,
        maximum_regular_files_excluding_manifest: int,
        maximum_directories_excluding_root: int,
        maximum_regular_file_bytes: int,
        maximum_total_regular_file_bytes: int,
        expected_mount: MountIdentity,
        mount_records: Sequence[MountIdentity],
        *,
        progress_callback: Callable[[], None] | None = None,
    ) -> tuple[int, int, int, int]:
        try:
            root_information = os.lstat(root)
        except OSError as error:
            raise ControllerError("archive_population_root", error) from error
        if stat.S_ISLNK(root_information.st_mode) or not stat.S_ISDIR(
            root_information.st_mode
        ):
            raise ControllerError("archive_population_root", root)
        stack = [root]
        entries = 0
        directories = 0
        regular_files = 0
        total_regular_file_bytes = 0
        while stack:
            directory = stack.pop()
            if progress_callback is not None:
                progress_callback()
            try:
                with os.scandir(directory) as iterator:
                    for child in iterator:
                        if progress_callback is not None:
                            progress_callback()
                        entries += 1
                        path = pathlib.Path(child.path)
                        try:
                            information = os.lstat(path)
                        except OSError as error:
                            raise ControllerError(
                                "archive_population_stat", error
                            ) from error
                        if information.st_dev != root_information.st_dev:
                            raise ControllerError("archive_population_device", path)
                        if (
                            _select_mount_identity(path, information, mount_records)
                            != expected_mount
                        ):
                            raise ControllerError(
                                "archive_population_mount", path
                            )
                        if stat.S_ISLNK(information.st_mode):
                            raise ControllerError("archive_population_symlink", path)
                        if stat.S_ISDIR(information.st_mode):
                            directories += 1
                            if directories > maximum_directories_excluding_root:
                                raise ControllerError(
                                    "archive_directory_limit", root
                                )
                            stack.append(path)
                        elif stat.S_ISREG(information.st_mode):
                            regular_files += 1
                            if information.st_size > maximum_regular_file_bytes:
                                raise ControllerError(
                                    "archive_regular_file_bytes", path
                                )
                            total_regular_file_bytes += information.st_size
                            if (
                                total_regular_file_bytes
                                > maximum_total_regular_file_bytes
                            ):
                                raise ControllerError(
                                    "archive_total_regular_file_bytes", root
                                )
                            if regular_files > maximum_regular_files_excluding_manifest:
                                raise ControllerError(
                                    "archive_regular_file_limit", root
                                )
                        else:
                            raise ControllerError("archive_population_type", path)
            except OSError as error:
                raise ControllerError("archive_population_scan", error) from error
        if progress_callback is not None:
            progress_callback()
        return entries, directories, regular_files, total_regular_file_bytes

    def _scan_recovery_population_and_reserve(
        self,
        plan: RunPlan,
        stage: pathlib.Path,
        timeline: _SafetyTimelineState,
        recovery_bytes: int,
        recovery_marker_present: bool,
        expected_mount: MountIdentity,
    ) -> tuple[int, int, int, int]:
        """Prove the live manifest-less stage can add recovery.json and manifest.

        The population is observed before any recovery mutation.  One regular
        file and the exact canonical recovery-marker bytes are reserved only
        when the idempotent marker is absent; the maximum manifest byte budget
        is reserved in the payload inequality in both cases.
        """

        if (
            type(recovery_bytes) is not int
            or recovery_bytes < 0
            or type(recovery_marker_present) is not bool
        ):
            raise ControllerError(
                "recovery_population_argument", "invalid reserve parameters"
            )
        payload_limit = (
            plan.safety.maximum_archive_bytes
            - plan.execution.maximum_manifest_bytes
        )
        mount_records = self._mount_snapshot(
            timeline, "scan-recovery-population-mounts"
        )
        try:
            population = self._archive_call(
                timeline,
                "scan-recovery-population",
                self._archive_population,
                stage,
                plan.execution.maximum_archive_regular_files_excluding_manifest,
                plan.execution.maximum_archive_directories_excluding_root,
                payload_limit,
                payload_limit,
                expected_mount,
                mount_records,
            )
        except BaseException as error:
            raise self._error("recovery_population", error) from error
        _entries, _directories, regular_files, regular_bytes = population
        new_files = 0 if recovery_marker_present else 1
        new_bytes = 0 if recovery_marker_present else recovery_bytes
        if (
            regular_files + new_files
            > plan.execution.maximum_archive_regular_files_excluding_manifest
        ):
            raise ControllerError(
                "recovery_regular_file_reserve",
                f"{regular_files}+{new_files}>"
                f"{plan.execution.maximum_archive_regular_files_excluding_manifest}",
            )
        if regular_bytes + new_bytes > payload_limit:
            raise ControllerError(
                "recovery_payload_reserve",
                f"{regular_bytes}+{new_bytes}>{payload_limit}",
            )
        return population

    @staticmethod
    def _manifest_record(
        summary: object,
        manifest_size: int,
        population: tuple[int, int, int],
        *,
        manifest_path: pathlib.Path | None = None,
    ) -> dict[str, object]:
        entries, directories, regular_files = population
        return {
            "directory_count_excluding_root": int(summary.directory_count),
            "identity_claim": (
                "content_hashes_are_not_authenticity_or_authorship_proof"
            ),
            "local_strict_identity_fields": ["mode", "mtime_ns"],
            "manifest_bytes": manifest_size,
            "manifest_path": str(
                summary.manifest_path if manifest_path is None else manifest_path
            ),
            "manifest_sha256": str(summary.manifest_sha256),
            "observed_archive_entries_including_manifest": entries,
            "observed_directories": directories,
            "observed_regular_files_including_manifest": regular_files,
            "portable_content_identity_fields": [
                "relative_path",
                "size",
                "sha256",
            ],
            "regular_file_count_excluding_manifest": int(
                summary.regular_file_count
            ),
            "total_regular_file_bytes_excluding_manifest": int(
                summary.total_regular_file_bytes
            ),
            "total_regular_file_bytes_including_manifest": (
                int(summary.total_regular_file_bytes) + manifest_size
            ),
        }

    def _sample_terminal_safety(self, context: _CoreContext) -> dict[str, object]:
        self._validate_preflight_identities(context)
        scratch_threshold = max(
            context.plan.safety.minimum_free_during_bytes,
            context.baseline_free_bytes
            - context.plan.safety.maximum_observed_free_space_decline_bytes,
        )
        try:
            scratch_sample = self._force_timeline_sample(
                context.safety_timeline,
                minimum_free_bytes=scratch_threshold,
            )
        except BaseException as error:
            raise self._error("terminal_safety", error) from error
        return {
            "archive": self._sample_archive_safety(context, reserve_bytes=0),
            "scratch": self._sample_record(scratch_sample),
        }

    def _sample_archive_safety(
        self, context: _CoreContext, *, reserve_bytes: int
    ) -> dict[str, object]:
        if type(reserve_bytes) is not int or reserve_bytes < 0:
            raise ControllerError("archive_safety_reserve", reserve_bytes)
        self._validate_preflight_identities(context)
        threshold = _prospective_free_threshold(
            absolute_floor_bytes=context.plan.safety.minimum_free_during_bytes,
            baseline_free_bytes=context.baseline_archive_free_bytes,
            maximum_decline_bytes=(
                context.plan.safety.maximum_observed_free_space_decline_bytes
            ),
            prospective_write_bytes=reserve_bytes,
            allocation_unit_bytes=context.archive_allocation_unit_bytes,
        )
        try:
            sample = self._force_timeline_sample(
                context.safety_timeline,
                minimum_free_bytes=threshold,
            )
        except BaseException as error:
            raise self._error("archive_safety", error) from error
        return self._sample_record(sample)

    def publish(
        self,
        workspace: RunWorkspace,
        axes: TerminalAxes,
        maximum_archive_bytes: int,
        receipt_validator: Callable[
            [Mapping[str, object]], Mapping[str, object]
        ],
    ) -> Mapping[str, object]:
        context = self._context(workspace)
        plan = context.plan
        if maximum_archive_bytes != plan.safety.maximum_archive_bytes:
            raise ControllerError("publication_bound_mismatch", maximum_archive_bytes)
        if context.last_phase != "READY_TO_SEAL":
            raise ControllerError("publication_phase", context.last_phase)
        if axes.archive_integrity != "VERIFIED":
            raise ControllerError(
                "publication_candidate_integrity", axes.archive_integrity
            )
        self._validate_mount_identity(context)
        verify_procedure_descriptor()
        terminal_safety = self._sample_terminal_safety(context)
        publication_contract = {
            "activation_condition": (
                PUBLICATION_ACTIVATION
            ),
            "archive_budget": {
                "maximum_archive_bytes": plan.safety.maximum_archive_bytes,
                "maximum_archive_directories_excluding_root": (
                    plan.execution.maximum_archive_directories_excluding_root
                ),
                "maximum_archive_regular_files_excluding_manifest": (
                    plan.execution.maximum_archive_regular_files_excluding_manifest
                ),
                "maximum_manifest_bytes": plan.execution.maximum_manifest_bytes,
            },
            "plan_sha256": plan.sha256,
            "prepublication_safety": terminal_safety,
            "procedure_sha256": PROCEDURE_DESCRIPTOR_SHA256,
            "original_archive_destination": str(workspace.archive_destination),
            "run_id": workspace.run_id,
            "schema": "oai.memprof.r0.publication-contract/v1",
            "terminal_axes": axes.as_dict(),
        }
        self.write_json(
            workspace, "publication-contract.json", publication_contract
        )
        pre_manifest_safety = self._sample_archive_safety(
            context, reserve_bytes=plan.execution.maximum_manifest_bytes
        )
        manifest_arguments = self._manifest_arguments(plan)
        try:
            summary = self._archive_call(
                context.safety_timeline,
                "create-manifest",
                self.archive.create_manifest,
                workspace.archive_stage,
                **manifest_arguments,
                max_manifest_bytes=plan.execution.maximum_manifest_bytes,
            )
            post_manifest_safety = self._sample_archive_safety(
                context, reserve_bytes=0
            )
            verified = self._archive_call(
                context.safety_timeline,
                "verify-staged-manifest",
                self.archive.verify_manifest,
                workspace.archive_stage,
                **manifest_arguments,
            )
            if verified != summary:
                raise ControllerError(
                    "manifest_verification_mismatch", workspace.run_id
                )
            manifest_image = self._archive_call(
                context.safety_timeline,
                "read-staged-manifest",
                self.archive.read_regular_file_bounded,
                workspace.archive_stage / "manifest.json",
                plan.execution.maximum_manifest_bytes,
            )
            population = (
                int(summary.regular_file_count)
                + int(summary.directory_count)
                + 1,
                int(summary.directory_count),
                int(summary.regular_file_count) + 1,
            )
            if (
                summary.regular_file_count != context.archive_regular_file_count
                or summary.directory_count != context.archive_directory_count
                or summary.total_regular_file_bytes != context.archive_payload_bytes
                or population
                != (
                    context.archive_entry_count + 1,
                    context.archive_directory_count,
                    context.archive_regular_file_count + 1,
                )
                or summary.total_regular_file_bytes + manifest_image.size
                > plan.safety.maximum_archive_bytes
            ):
                raise ControllerError(
                    "manifest_accounting_mismatch",
                    f"files={summary.regular_file_count}/{context.archive_regular_file_count} "
                    f"directories={summary.directory_count}/{context.archive_directory_count} "
                    f"bytes={summary.total_regular_file_bytes}/{context.archive_payload_bytes}",
                )
            publication_result = self._archive_call(
                context.safety_timeline,
                "publish-verified-archive",
                self.archive.publish_verified_archive,
                workspace.archive_stage,
                workspace.archive_destination,
                **manifest_arguments,
            )
            publication_phase = getattr(
                publication_result.phase, "value", publication_result.phase
            )
            if (
                publication_result.summary != summary
                or publication_phase
                != self.archive.PublicationPhase.VERIFIED.value
            ):
                raise ControllerError(
                    "publication_verification_mismatch", workspace.run_id
                )
            self._validate_mount_identity(context)
            marker_threshold = _prospective_free_threshold(
                absolute_floor_bytes=plan.safety.minimum_free_during_bytes,
                baseline_free_bytes=context.baseline_free_bytes,
                maximum_decline_bytes=(
                    plan.safety.maximum_observed_free_space_decline_bytes
                ),
                prospective_write_bytes=plan.execution.maximum_json_evidence_bytes,
                allocation_unit_bytes=context.acquisition_allocation_unit_bytes,
            )
            marker_precommit_sample = self._force_timeline_sample(
                context.safety_timeline,
                minimum_free_bytes=marker_threshold,
            )
            manifest_record = self._manifest_record(
                summary,
                manifest_image.size,
                population,
                manifest_path=workspace.archive_destination / "manifest.json",
            )
            marker_value = {
                "activation_condition": PUBLICATION_LINEARIZATION,
                "archive_path": str(workspace.archive_destination),
                "global_safety_timeline": self._timeline_receipt(
                    context.safety_timeline
                ),
                "manifest_sha256": manifest_record["manifest_sha256"],
                "plan_sha256": plan.sha256,
                "procedure_sha256": PROCEDURE_DESCRIPTOR_SHA256,
                "publication_phase": publication_phase,
                "run_id": workspace.run_id,
                "schema": PUBLICATION_SUCCESS_SCHEMA,
                "supported_signal_order_cutoff": PUBLICATION_SIGNAL_CUTOFF,
                "terminal_axes": axes.as_dict(),
            }
            marker_bytes = canonical_json_bytes(marker_value)
            if len(marker_bytes) > plan.execution.maximum_json_evidence_bytes:
                raise ControllerError(
                    "publication_success_marker_size",
                    f"{len(marker_bytes)}>{plan.execution.maximum_json_evidence_bytes}",
                )
            marker_path = workspace.scratch / "publication-success.json"
            prepared_receipt = {
                "archive_integrity": "VERIFIED",
                "archive_path": str(workspace.archive_destination),
                "inclusion": axes.inclusion,
                "manifest": manifest_record,
                "manifest_safety": {
                    "post_write": post_manifest_safety,
                    "prospective_maximum_manifest_bytes": pre_manifest_safety,
                },
                "global_safety_timeline": self._timeline_receipt(
                    context.safety_timeline
                ),
                "mutated": True,
                "postcompletion_operational_problem": None,
                "production_final_elf_absence": axes.production_final_elf_absence,
                "profiler_stream_state": axes.profiler_stream_state,
                "publication_phase": publication_phase,
                "publication_success_marker": {
                    "bytes": len(marker_bytes),
                    "path": str(marker_path),
                    "sha256": hashlib.sha256(marker_bytes).hexdigest(),
                    "value": marker_value,
                },
                "publication_success_marker_precommit_safety": (
                    self._sample_record(marker_precommit_sample)
                ),
                "run_id": workspace.run_id,
                "schema": "oai.memprof.r0.publication-receipt/v1",
                "scientific_case_state": axes.scientific_case_state,
                "terminal_axes": axes.as_dict(),
                "verified": True,
            }
            validated_receipt = receipt_validator(prepared_receipt)
            precommit_now_ns = time.monotonic_ns()
            precommit_deadline_ns = (
                marker_precommit_sample.monotonic_ns
                + plan.safety.maximum_poll_gap_milliseconds * 1_000_000
            )
            if (
                type(precommit_now_ns) is not int
                or precommit_now_ns < marker_precommit_sample.monotonic_ns
            ):
                raise ControllerError(
                    "global_safety_timeline",
                    "precommit monotonic observation is invalid",
                )
            if precommit_now_ns > precommit_deadline_ns:
                raise ControllerError(
                    "maximum_poll_gap",
                    f"precommit {precommit_now_ns}>{precommit_deadline_ns}",
                )
        except ControllerError:
            raise
        except BaseException as error:
            raise self._error("archive_publication", error) from error
        try:
            postcompletion_problem = self._commit_publication_success_marker(
                marker_path,
                marker_bytes,
                maximum_bytes=plan.execution.maximum_json_evidence_bytes,
            )
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise ControllerError(
                "publication_success_marker_commit",
                _bounded_text(error),
                publication_phase="verified",
            ) from error
        if postcompletion_problem is None:
            return validated_receipt
        if (
            type(postcompletion_problem) is not dict
            or set(postcompletion_problem)
            != {"code", "message", "phase", "scientific_effect"}
            or type(postcompletion_problem["code"]) is not str
            or not postcompletion_problem["code"]
            or type(postcompletion_problem["message"]) is not str
            or postcompletion_problem["phase"]
            != "postcompletion_signal_mask_restoration"
            or postcompletion_problem["scientific_effect"]
            != "none_durable_success_marker_already_committed"
        ):
            raise AssertionError("invalid internal postcompletion problem")
        completed_receipt = dict(validated_receipt)
        completed_receipt["postcompletion_operational_problem"] = dict(
            postcompletion_problem
        )
        return completed_receipt

    @staticmethod
    def _canonical_object(data: bytes, role: str) -> Mapping[str, object]:
        value = _strict_result_json(data, role)
        if canonical_json_bytes(value) != data:
            raise GateError("noncanonical_archive_json", role)
        return value

    @staticmethod
    def _axes_from_mapping(value: object) -> TerminalAxes:
        expected_keys = {
            "archive_integrity",
            "claims",
            "inclusion",
            "production_final_elf_absence",
            "profiler_stream_state",
            "reasons",
            "scientific_case_state",
        }
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise GateError("terminal_axes_shape", "terminal axes members differ")
        scalar_contract = {
            "archive_integrity": {"VERIFIED", "UNVERIFIED"},
            "inclusion": {"INCLUDED", "EXCLUDED"},
            "profiler_stream_state": {"NOT_APPLICABLE_R0"},
            "production_final_elf_absence": {PRODUCTION_FINAL_ELF_ABSENCE},
            "scientific_case_state": {"COMPLETE", "INCOMPLETE"},
        }
        for name, admitted in scalar_contract.items():
            if value[name] not in admitted:
                raise GateError("terminal_axes_value", f"{name}: {value[name]!r}")
        raw_reasons = value["reasons"]
        if (
            not isinstance(raw_reasons, list)
            or any(
                not isinstance(reason, str)
                or not reason
                or len(reason) > 128
                or any(ord(character) < 0x20 for character in reason)
                for reason in raw_reasons
            )
            or len(raw_reasons) != len(set(raw_reasons))
        ):
            raise GateError("terminal_reason_shape", "terminal reasons are invalid")
        raw_claims = value["claims"]
        expected_claims = (
            "disabled_generated_graph_absence",
            "four_role_fixture_final_elf",
            "production_final_elf_absence",
        )
        if not isinstance(raw_claims, list) or len(raw_claims) != 3:
            raise GateError("terminal_claim_population", "expected exactly three claims")
        claims: list[ClaimDecision] = []
        for expected_name, raw_claim in zip(expected_claims, raw_claims):
            if not isinstance(raw_claim, dict) or set(raw_claim) != {
                "claim",
                "eligibility",
                "reasons",
            }:
                raise GateError("terminal_claim_shape", expected_name)
            if raw_claim["claim"] != expected_name or raw_claim["eligibility"] not in {
                "QUALIFIED",
                "SUPPORTING_ONLY",
                "UNAVAILABLE",
                "UNVERIFIED",
                "NOT_APPLICABLE",
            }:
                raise GateError("terminal_claim_value", expected_name)
            claim_reasons = raw_claim["reasons"]
            if (
                not isinstance(claim_reasons, list)
                or not claim_reasons
                or any(
                not isinstance(reason, str)
                or not reason
                or len(reason) > 128
                or any(
                    ord(character) < 0x20 or ord(character) == 0x7F
                    for character in reason
                )
                for reason in claim_reasons
            )
                or len(claim_reasons) != len(set(claim_reasons))
            ):
                raise GateError("terminal_claim_reasons", expected_name)
            claims.append(
                ClaimDecision(
                    expected_name,
                    str(raw_claim["eligibility"]),
                    tuple(str(reason) for reason in claim_reasons),
                )
            )
        evidence_eligibilities = {
            "QUALIFIED",
            "SUPPORTING_ONLY",
            "UNAVAILABLE",
            "UNVERIFIED",
        }
        if any(
            claim.eligibility not in evidence_eligibilities
            for claim in claims[:2]
        ):
            raise GateError(
                "terminal_claim_domain",
                "R0 evidence claims cannot be NOT_APPLICABLE",
            )
        axes = TerminalAxes(
            scientific_case_state=str(value["scientific_case_state"]),
            archive_integrity=str(value["archive_integrity"]),
            inclusion=str(value["inclusion"]),
            profiler_stream_state=str(value["profiler_stream_state"]),
            production_final_elf_absence=str(value["production_final_elf_absence"]),
            reasons=tuple(str(reason) for reason in raw_reasons),
            claims=tuple(claims),
        )
        expected_inclusion = (
            "INCLUDED"
            if axes.scientific_case_state == "COMPLETE"
            and axes.archive_integrity == "VERIFIED"
            else "EXCLUDED"
        )
        if axes.inclusion != expected_inclusion:
            raise GateError("terminal_inclusion_relation", "axes inclusion differs")
        archive_reason_present = "ARCHIVE_UNVERIFIED" in axes.reasons
        scientific_reasons = tuple(
            reason for reason in axes.reasons if reason != "ARCHIVE_UNVERIFIED"
        )
        if axes.archive_integrity == "VERIFIED":
            if archive_reason_present or any(
                claim.eligibility == "UNVERIFIED" for claim in axes.claims
            ):
                raise GateError(
                    "terminal_claim_integrity", "verified axes contain unverified state"
                )
        elif not archive_reason_present or any(
            claim.eligibility in {"QUALIFIED", "SUPPORTING_ONLY"}
            for claim in axes.claims[:2]
        ):
            raise GateError(
                "terminal_claim_integrity", "unverified axes contain verified evidence claim"
            )
        if axes.scientific_case_state == "COMPLETE":
            if scientific_reasons:
                raise GateError(
                    "terminal_complete_relation", "complete axes have scientific failure"
                )
            if axes.archive_integrity == "VERIFIED":
                if axes.claims[0].eligibility not in {
                    "QUALIFIED",
                    "SUPPORTING_ONLY",
                } or axes.claims[1].eligibility != "QUALIFIED":
                    raise GateError(
                        "terminal_complete_claim_population",
                        "verified COMPLETE axes lack both produced R0 claims",
                    )
            elif any(
                claim.eligibility != "UNVERIFIED" for claim in axes.claims[:2]
            ):
                raise GateError(
                    "terminal_complete_claim_population",
                    "unverified COMPLETE axes do not preserve both produced claims",
                )
        elif not scientific_reasons:
            raise GateError(
                "terminal_incomplete_relation", "incomplete axes lack scientific failure"
            )
        if axes.scientific_case_state != "COMPLETE" and any(
            claim.eligibility == "QUALIFIED" for claim in axes.claims
        ):
            raise GateError("terminal_claim_completion", "incomplete case has qualified claim")
        production = axes.claims[2]
        if production.eligibility != "NOT_APPLICABLE" or production.reasons != (
            PRODUCTION_FINAL_ELF_ABSENCE,
        ):
            raise GateError("terminal_production_claim", "production claim boundary differs")
        return axes

    @staticmethod
    def _recovered_axes(*, verified: bool, reason: str) -> TerminalAxes:
        integrity = "VERIFIED" if verified else "UNVERIFIED"
        eligibility = "UNAVAILABLE"
        claim_reason = "RECOVERED_AFTER_INTERRUPTION" if verified else "ARCHIVE_UNVERIFIED"
        return TerminalAxes(
            scientific_case_state="INCOMPLETE",
            archive_integrity=integrity,
            inclusion="EXCLUDED",
            profiler_stream_state="NOT_APPLICABLE_R0",
            production_final_elf_absence=PRODUCTION_FINAL_ELF_ABSENCE,
            reasons=(reason,) if verified else (reason, "ARCHIVE_UNVERIFIED"),
            claims=(
                ClaimDecision(
                    "disabled_generated_graph_absence", eligibility, (claim_reason,)
                ),
                ClaimDecision(
                    "four_role_fixture_final_elf", eligibility, (claim_reason,)
                ),
                ClaimDecision(
                    "production_final_elf_absence",
                    "NOT_APPLICABLE",
                    (PRODUCTION_FINAL_ELF_ABSENCE,),
                ),
            ),
        )

    @staticmethod
    def _axes_without_completion_marker(candidate: TerminalAxes) -> TerminalAxes:
        return _CoreServices._axes_without_terminal_activation(
            candidate, "PUBLICATION_COMPLETION_MARKER_UNAVAILABLE"
        )

    @staticmethod
    def _axes_without_original_location(candidate: TerminalAxes) -> TerminalAxes:
        return _CoreServices._axes_without_terminal_activation(
            candidate, "ORIGINAL_PUBLICATION_LOCATION_UNVERIFIED"
        )

    @staticmethod
    def _axes_without_terminal_activation(
        candidate: TerminalAxes, reason: str
    ) -> TerminalAxes:
        claims: list[ClaimDecision] = []
        for claim in candidate.claims[:2]:
            if claim.eligibility == "UNAVAILABLE":
                claims.append(claim)
            else:
                claims.append(
                    ClaimDecision(
                        claim.claim,
                        "SUPPORTING_ONLY",
                        tuple(dict.fromkeys((*claim.reasons, "CASE_INCOMPLETE"))),
                    )
                )
        claims.append(candidate.claims[2])
        return TerminalAxes(
            scientific_case_state="INCOMPLETE",
            archive_integrity="VERIFIED",
            inclusion="EXCLUDED",
            profiler_stream_state="NOT_APPLICABLE_R0",
            production_final_elf_absence=PRODUCTION_FINAL_ELF_ABSENCE,
            reasons=tuple(dict.fromkeys((*candidate.reasons, reason))),
            claims=tuple(claims),
        )

    def _original_publication_location_matches(
        self,
        plan: RunPlan,
        archive_path: pathlib.Path,
        archive_information: os.stat_result,
        run_id: str,
        timeline: _SafetyTimelineState,
    ) -> bool:
        """Bind the original lexical location to archived and current ext4 identity."""

        acquisition_parent = pathlib.Path(plan.roots.acquisition_parent)
        archive_parent = pathlib.Path(plan.roots.archive_parent)
        if archive_path != archive_parent / run_id:
            return False
        preflight_image = self._archive_call(
            timeline,
            "read-location-preflight",
            self.archive.read_regular_file_bounded,
            archive_path / "provenance/preflight-safety.json",
            plan.execution.maximum_json_evidence_bytes,
        )
        try:
            preflight = self._canonical_object(
                preflight_image.data, "location_preflight_safety"
            )
            if (
                set(preflight)
                != {
                    "acquisition_sample",
                    "archive_sample",
                    "filesystems",
                    "global_same_mount_baseline_free_bytes",
                    "initial_workspace_evidence_safety",
                    "schema",
                    "thermal_ceiling_millicelsius",
                    "thermal_sensors",
                }
                or preflight["schema"] != "oai.memprof.r0.safety-preflight/v1"
                or type(preflight["filesystems"]) is not dict
                or set(preflight["filesystems"]) != {"acquisition", "archive"}
            ):
                return False
        except (ControllerError, GateError):
            return False
        filesystem_keys = {
            "available_blocks",
            "available_bytes",
            "block_size_bytes",
            "device",
            "fragment_size_bytes",
            "free_blocks",
            "free_inodes",
            "inode",
            "mode",
            "mount",
            "path",
            "total_blocks",
            "total_inodes",
        }
        try:
            root_information = (
                os.lstat(acquisition_parent),
                os.lstat(archive_parent),
            )
            current_archive_information = os.lstat(archive_path)
            if (
                any(
                    stat.S_ISLNK(item.st_mode)
                    or not stat.S_ISDIR(item.st_mode)
                    for item in root_information
                )
                or acquisition_parent.resolve(strict=True) != acquisition_parent
                or archive_parent.resolve(strict=True) != archive_parent
                or (root_information[0].st_dev, root_information[0].st_ino)
                == (root_information[1].st_dev, root_information[1].st_ino)
                or stat.S_ISLNK(archive_information.st_mode)
                or not stat.S_ISDIR(archive_information.st_mode)
                or archive_path.resolve(strict=True) != archive_path
                or (
                    current_archive_information.st_dev,
                    current_archive_information.st_ino,
                    current_archive_information.st_mode,
                )
                != (
                    archive_information.st_dev,
                    archive_information.st_ino,
                    archive_information.st_mode,
                )
            ):
                return False
            mount_records = self._mount_snapshot(
                timeline, "verify-original-location-mounts"
            )
            current_mounts = (
                _select_mount_identity(
                    acquisition_parent, root_information[0], mount_records
                ),
                _select_mount_identity(
                    archive_parent, root_information[1], mount_records
                ),
            )
            archive_directory_mount = _select_mount_identity(
                archive_path, current_archive_information, mount_records
            )
        except OSError:
            return False
        except ControllerError as error:
            if error.code in {
                "active_filesystem_type",
                "active_mount_root",
                "mountinfo_device_mismatch",
                "mountinfo_path_ambiguous",
                "mountinfo_path_unmatched",
            }:
                return False
            raise
        archived_mounts: list[MountIdentity] = []
        for role, path, information, current_mount in zip(
            ("acquisition", "archive"),
            (acquisition_parent, archive_parent),
            root_information,
            current_mounts,
            strict=True,
        ):
            filesystem = preflight["filesystems"][role]
            if type(filesystem) is not dict or set(filesystem) != filesystem_keys:
                return False
            try:
                archived_mount = _mount_identity_from_record(
                    filesystem["mount"], f"original-location-{role}"
                )
            except ControllerError:
                return False
            if (
                type(filesystem["path"]) is not str
                or filesystem["path"] != str(path)
                or type(filesystem["device"]) is not int
                or filesystem["device"] != information.st_dev
                or type(filesystem["inode"]) is not int
                or filesystem["inode"] != information.st_ino
                or type(filesystem["mode"]) is not int
                or filesystem["mode"] != information.st_mode
                or archived_mount != current_mount
            ):
                return False
            archived_mounts.append(archived_mount)
        return bool(
            archived_mounts[0] == archived_mounts[1]
            and current_mounts[0] == current_mounts[1]
            and archive_directory_mount == current_mounts[1]
        )

    def _verification_failure(
        self,
        archive_path: pathlib.Path,
        error: BaseException,
        timeline: _SafetyTimelineState | None,
    ) -> Mapping[str, object]:
        raw_code = getattr(error, "code", None)
        normalized_code = getattr(raw_code, "value", raw_code)
        if not isinstance(normalized_code, str) or not normalized_code:
            normalized_code = type(error).__name__
        axes = self._recovered_axes(
            verified=False, reason="ARCHIVE_VERIFICATION_FAILED"
        )
        verification_timeline = None
        if timeline is not None:
            verification_timeline = dict(
                _validate_global_safety_timeline_structure(
                    self._timeline_receipt(timeline),
                    "failed archive verification",
                )
            )
        return {
            **axes.as_dict(),
            "archive_path": str(archive_path),
            "mutated": False,
            "problem": {
                "code": _bounded_text(normalized_code, 128),
                "message": _bounded_text(getattr(error, "detail", error)),
            },
            "schema": "oai.memprof.r0.archive-verification/v1",
            "verified": False,
            "verification_global_safety_timeline": verification_timeline,
        }

    def verify_archive(
        self,
        archive_path: pathlib.Path,
        *,
        timeline: _SafetyTimelineState | None = None,
        expected_plan: RunPlan | None = None,
    ) -> Mapping[str, object]:
        try:
            archive_path = self.archive.validate_absolute_path(
                archive_path, "archive_path"
            )
            information = os.lstat(archive_path)
            if stat.S_ISLNK(information.st_mode) or not stat.S_ISDIR(
                information.st_mode
            ):
                raise ControllerError("archive_type", archive_path)
            if timeline is None:
                if expected_plan is not None:
                    raise ControllerError(
                        "verification_timeline_binding",
                        "expected plan requires the shared recovery timeline",
                    )
                plan = load_plan(archive_path / "inputs/plan.json")
                timeline = self._new_live_safety_timeline(plan, archive_path)
            else:
                if (
                    expected_plan is None
                    or timeline.plan.sha256 != expected_plan.sha256
                    or timeline.plan.canonical_bytes
                    != expected_plan.canonical_bytes
                ):
                    raise ControllerError(
                        "verification_timeline_binding",
                        "shared recovery timeline and immutable plan differ",
                    )
                plan_image = self._archive_call(
                    timeline,
                    "read-shared-verification-plan",
                    self.archive.read_regular_file_bounded,
                    archive_path / "inputs/plan.json",
                    MAXIMUM_PLAN_BYTES,
                )
                plan = parse_plan(plan_image.data)
                if (
                    plan.sha256 != expected_plan.sha256
                    or plan.canonical_bytes != expected_plan.canonical_bytes
                ):
                    raise ControllerError(
                        "verification_plan_binding",
                        "postpublication archived plan differs",
                    )
            verify_procedure_descriptor()
            descriptor_image = self._archive_call(
                timeline,
                "read-archived-procedure",
                self.archive.read_regular_file_bounded,
                archive_path / "inputs/procedure.json",
                plan.execution.maximum_json_evidence_bytes,
            )
            if (
                descriptor_image.data != canonical_json_bytes(procedure_descriptor())
                or descriptor_image.sha256 != PROCEDURE_DESCRIPTOR_SHA256
            ):
                raise GateError("archived_procedure_identity", archive_path)
            manifest_arguments = self._manifest_arguments(plan)
            summary = self._archive_call(
                timeline,
                "verify-published-manifest",
                self.archive.verify_manifest,
                archive_path,
                **manifest_arguments,
            )
            manifest_image = self._archive_call(
                timeline,
                "read-published-manifest",
                self.archive.read_regular_file_bounded,
                archive_path / "manifest.json",
                plan.execution.maximum_manifest_bytes,
            )
            population = (
                int(summary.regular_file_count)
                + int(summary.directory_count)
                + 1,
                int(summary.directory_count),
                int(summary.regular_file_count) + 1,
            )
            if (
                summary.regular_file_count
                > plan.execution.maximum_archive_regular_files_excluding_manifest
                or summary.directory_count
                > plan.execution.maximum_archive_directories_excluding_root
                or summary.total_regular_file_bytes + manifest_image.size
                > plan.safety.maximum_archive_bytes
            ):
                raise GateError("archive_total_bound", archive_path)
            recovered_suffix = ".recovered-incomplete"
            is_recovered = archive_path.name.endswith(recovered_suffix)
            run_id = (
                archive_path.name[: -len(recovered_suffix)]
                if is_recovered
                else archive_path.name
            )
            self.archive.validate_run_id(run_id)
            location_matches_original_publication = bool(
                not is_recovered
                and self._original_publication_location_matches(
                    plan,
                    archive_path,
                    information,
                    run_id,
                    timeline,
                )
            )
            marker_path = (
                pathlib.Path(plan.roots.acquisition_parent)
                / run_id
                / "publication-success.json"
            )
            marker_verified = False
            if is_recovered:
                axes = self._recovered_axes(
                    verified=True, reason="RECOVERED_AFTER_INTERRUPTION"
                )
            else:
                contract_image = self._archive_call(
                    timeline,
                    "read-publication-contract",
                    self.archive.read_regular_file_bounded,
                    archive_path / "publication-contract.json",
                    plan.execution.maximum_json_evidence_bytes,
                )
                contract = self._canonical_object(
                    contract_image.data, "publication_contract"
                )
                if set(contract) != {
                    "activation_condition",
                    "archive_budget",
                    "plan_sha256",
                    "prepublication_safety",
                    "procedure_sha256",
                    "original_archive_destination",
                    "run_id",
                    "schema",
                    "terminal_axes",
                }:
                    raise GateError("publication_contract_shape", archive_path)
                if (
                    contract["schema"] != "oai.memprof.r0.publication-contract/v1"
                    or contract["activation_condition"]
                    != PUBLICATION_ACTIVATION
                    or contract["run_id"] != run_id
                    or contract["plan_sha256"] != plan.sha256
                    or contract["procedure_sha256"] != PROCEDURE_DESCRIPTOR_SHA256
                    or contract["original_archive_destination"]
                    != str(pathlib.Path(plan.roots.archive_parent) / run_id)
                    or contract["archive_budget"]
                    != {
                        "maximum_archive_bytes": plan.safety.maximum_archive_bytes,
                        "maximum_archive_directories_excluding_root": (
                            plan.execution.maximum_archive_directories_excluding_root
                        ),
                        "maximum_archive_regular_files_excluding_manifest": (
                            plan.execution.maximum_archive_regular_files_excluding_manifest
                        ),
                        "maximum_manifest_bytes": plan.execution.maximum_manifest_bytes,
                    }
                ):
                    raise GateError("publication_contract_binding", archive_path)
                axes = self._axes_from_mapping(contract["terminal_axes"])
                if axes.archive_integrity != "VERIFIED":
                    raise GateError("publication_contract_integrity", archive_path)
                prepublication_safety = contract["prepublication_safety"]
                if not isinstance(prepublication_safety, dict) or set(
                    prepublication_safety
                ) != {"archive", "scratch"}:
                    raise GateError(
                        "publication_contract_safety_shape", archive_path
                    )
                scratch_safety = _validate_plan_safety_sample_receipt(
                    prepublication_safety["scratch"],
                    "archived prepublication scratch",
                    plan,
                    minimum_free_bytes=plan.safety.minimum_free_during_bytes,
                )
                archive_safety = _validate_plan_safety_sample_receipt(
                    prepublication_safety["archive"],
                    "archived prepublication archive",
                    plan,
                    minimum_free_bytes=plan.safety.minimum_free_during_bytes,
                )
                if archive_safety["monotonic_ns"] < scratch_safety["monotonic_ns"]:
                    raise GateError(
                        "publication_contract_safety_timeline", archive_path
                    )
                try:
                    marker_image = self._archive_call(
                        timeline,
                        "read-publication-success-marker",
                        self.archive.read_regular_file_bounded,
                        marker_path,
                        plan.execution.maximum_json_evidence_bytes,
                    )
                    observed_marker = self._canonical_object(
                        marker_image.data, "publication_success_marker"
                    )
                    if set(observed_marker) != {
                        "activation_condition",
                        "archive_path",
                        "global_safety_timeline",
                        "manifest_sha256",
                        "plan_sha256",
                        "procedure_sha256",
                        "publication_phase",
                        "run_id",
                        "schema",
                        "supported_signal_order_cutoff",
                        "terminal_axes",
                    }:
                        raise GateError(
                            "publication_success_marker_shape", marker_path
                        )
                    global_timeline = _validate_global_safety_timeline_receipt(
                        observed_marker["global_safety_timeline"],
                        "publication success marker",
                        plan,
                    )
                    expected_marker_value = {
                        "activation_condition": PUBLICATION_LINEARIZATION,
                        "archive_path": str(
                            pathlib.Path(plan.roots.archive_parent) / run_id
                        ),
                        "global_safety_timeline": dict(global_timeline),
                        "manifest_sha256": str(summary.manifest_sha256),
                        "plan_sha256": plan.sha256,
                        "procedure_sha256": PROCEDURE_DESCRIPTOR_SHA256,
                        "publication_phase": "verified",
                        "run_id": run_id,
                        "schema": PUBLICATION_SUCCESS_SCHEMA,
                        "supported_signal_order_cutoff": (
                            PUBLICATION_SIGNAL_CUTOFF
                        ),
                        "terminal_axes": axes.as_dict(),
                    }
                    expected_marker_bytes = canonical_json_bytes(
                        expected_marker_value
                    )
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException:
                    marker_image = None
                marker_verified = bool(
                    location_matches_original_publication
                    and
                    marker_image is not None
                    and marker_image.data == expected_marker_bytes
                    and marker_image.sha256
                    == hashlib.sha256(expected_marker_bytes).hexdigest()
                )
                if not location_matches_original_publication:
                    axes = self._axes_without_original_location(axes)
                elif not marker_verified:
                    axes = self._axes_without_completion_marker(axes)
            verification_result = {
                **axes.as_dict(),
                "archive_path": str(archive_path),
                "manifest": self._manifest_record(
                    summary,
                    manifest_image.size,
                    population,
                    manifest_path=archive_path / "manifest.json",
                ),
                "mutated": False,
                "location_matches_original_publication": (
                    location_matches_original_publication
                ),
                "plan_sha256": plan.sha256,
                "publication_success_marker_path": str(marker_path),
                "publication_success_marker_verified": marker_verified,
                "procedure_sha256": PROCEDURE_DESCRIPTOR_SHA256,
                "run_id": run_id,
                "schema": "oai.memprof.r0.archive-verification/v1",
                "verified": True,
            }
            self._force_timeline_sample(timeline)
            verification_timeline = self._timeline_receipt(timeline)
            _validate_global_safety_timeline_receipt(
                verification_timeline,
                "successful archive verification",
                plan,
            )
            verification_result["verification_global_safety_timeline"] = dict(
                verification_timeline
            )
            return verification_result
        except (KeyboardInterrupt, SystemExit):
            raise
        except ControllerError as error:
            if expected_plan is not None and error.code == "maximum_poll_gap":
                raise
            return self._verification_failure(
                pathlib.Path(archive_path), error, timeline
            )
        except BaseException as error:
            return self._verification_failure(
                pathlib.Path(archive_path), error, timeline
            )

    def _resolve_recovery_stage(
        self, case_path: pathlib.Path
    ) -> tuple[pathlib.Path, str]:
        try:
            case_path = self.archive.validate_absolute_path(case_path, "case_path")
        except BaseException as error:
            raise self._error("recovery_case_path", error) from error
        pointer_path = case_path / "archive-stage-pointer.json"
        try:
            pointer_image = self._archive_call(
                None,
                "read-recovery-pointer-bootstrap",
                self.archive.read_regular_file_bounded,
                pointer_path,
                MAXIMUM_PLAN_BYTES,
            )
        except BaseException as error:
            if getattr(getattr(error, "code", None), "value", None) != "path_missing":
                raise self._error("recovery_pointer", error) from error
            stage = case_path
            if not stage.name.startswith(".") or not stage.name.endswith(".active"):
                raise ControllerError("recovery_stage_name", stage)
            run_id = stage.name[1:-7]
        else:
            pointer = self._canonical_object(
                pointer_image.data, "archive_stage_pointer"
            )
            if set(pointer) != {"archive_stage", "run_id", "schema"} or pointer[
                "schema"
            ] != "oai.memprof.r0.archive-stage-pointer/v1":
                raise ControllerError("recovery_pointer_shape", pointer_path)
            stage = pathlib.Path(str(pointer["archive_stage"]))
            run_id = str(pointer["run_id"])
            if case_path.name != run_id:
                raise ControllerError("recovery_pointer_run_id", case_path)
        try:
            self.archive.validate_run_id(run_id)
            self.archive.validate_absolute_path(stage, "archive_stage")
        except BaseException as error:
            raise self._error("recovery_identity", error) from error
        return stage, run_id

    @staticmethod
    def _directory_presence(path: pathlib.Path, role: str) -> bool:
        try:
            information = os.lstat(path)
        except FileNotFoundError:
            return False
        except OSError as error:
            raise ControllerError(f"{role}_stat", error) from error
        if stat.S_ISLNK(information.st_mode) or not stat.S_ISDIR(
            information.st_mode
        ):
            raise ControllerError(f"{role}_type", path)
        return True

    def _recovery_storage_binding(
        self,
        plan: RunPlan,
        stage: pathlib.Path,
        timeline: _SafetyTimelineState,
    ) -> tuple[int, int, MountIdentity]:
        acquisition_parent = pathlib.Path(plan.roots.acquisition_parent)
        archive_parent = pathlib.Path(plan.roots.archive_parent)
        root_information: list[os.stat_result] = []
        for role, root in (
            ("acquisition", acquisition_parent),
            ("archive", archive_parent),
        ):
            try:
                information = os.lstat(root)
                resolved = root.resolve(strict=True)
            except OSError as error:
                raise ControllerError(f"recovery_{role}_root_stat", error) from error
            if (
                stat.S_ISLNK(information.st_mode)
                or not stat.S_ISDIR(information.st_mode)
                or resolved != root
            ):
                raise ControllerError(
                    f"recovery_{role}_root_identity", f"{root} resolves to {resolved}"
                )
            root_information.append(information)
        if (
            root_information[0].st_dev,
            root_information[0].st_ino,
        ) == (
            root_information[1].st_dev,
            root_information[1].st_ino,
        ):
            raise ControllerError(
                "recovery_root_inode_alias",
                "acquisition and archive parents name the same directory inode",
            )
        records = self._mount_snapshot(
            timeline, "recovery-storage-mounts"
        )
        acquisition_mount = _select_mount_identity(
            acquisition_parent, root_information[0], records
        )
        archive_mount = _select_mount_identity(
            archive_parent, root_information[1], records
        )
        if acquisition_mount != archive_mount:
            raise ControllerError(
                "recovery_mount_mismatch",
                "recovery roots no longer share one exact ext4 mount",
            )
        try:
            stage_information = os.lstat(stage)
            acquisition_vfs = os.statvfs(acquisition_parent)
            archive_vfs = os.statvfs(archive_parent)
        except OSError as error:
            raise ControllerError("recovery_stage_filesystem", error) from error
        if (
            stat.S_ISLNK(stage_information.st_mode)
            or not stat.S_ISDIR(stage_information.st_mode)
            or stage_information.st_dev != root_information[1].st_dev
            or type(acquisition_vfs.f_frsize) is not int
            or acquisition_vfs.f_frsize <= 0
            or type(archive_vfs.f_frsize) is not int
            or archive_vfs.f_frsize <= 0
        ):
            raise ControllerError("recovery_stage_filesystem", stage)
        stage_mount = _select_mount_identity(stage, stage_information, records)
        if stage_mount != archive_mount:
            raise ControllerError(
                "recovery_stage_mount",
                "active recovery stage is not on the archived ext4 mount identity",
            )
        try:
            preflight_image = self._archive_call(
                timeline,
                "read-recovery-preflight",
                self.archive.read_regular_file_bounded,
                stage / "provenance/preflight-safety.json",
                plan.execution.maximum_json_evidence_bytes,
            )
        except BaseException as error:
            raise self._error("recovery_preflight_read", error) from error
        preflight = self._canonical_object(
            preflight_image.data, "recovery_preflight_safety"
        )
        if set(preflight) != {
            "acquisition_sample",
            "archive_sample",
            "filesystems",
            "global_same_mount_baseline_free_bytes",
            "initial_workspace_evidence_safety",
            "schema",
            "thermal_ceiling_millicelsius",
            "thermal_sensors",
        } or preflight["schema"] != "oai.memprof.r0.safety-preflight/v1":
            raise ControllerError("recovery_preflight_shape", stage)
        acquisition_sample = preflight["acquisition_sample"]
        archive_sample = preflight["archive_sample"]
        filesystems = preflight["filesystems"]
        acquisition_sample = _validate_plan_safety_sample_receipt(
            acquisition_sample,
            "recovery acquisition preflight",
            plan,
            minimum_free_bytes=plan.safety.minimum_free_start_bytes,
        )
        archive_sample = _validate_plan_safety_sample_receipt(
            archive_sample,
            "recovery archive preflight",
            plan,
            minimum_free_bytes=plan.safety.minimum_free_start_bytes,
        )
        if (
            not isinstance(filesystems, dict)
            or set(filesystems) != {"acquisition", "archive"}
            or type(preflight["global_same_mount_baseline_free_bytes"])
            is not int
            or preflight["global_same_mount_baseline_free_bytes"]
            != max(
                acquisition_sample["free_bytes"],
                archive_sample["free_bytes"],
            )
            or type(preflight["thermal_ceiling_millicelsius"]) is not int
            or preflight["thermal_ceiling_millicelsius"]
            != plan.safety.thermal_ceiling_millicelsius
        ):
            raise ControllerError("recovery_preflight_binding", stage)
        if archive_sample["monotonic_ns"] < acquisition_sample["monotonic_ns"]:
            raise ControllerError("recovery_preflight_timeline", stage)
        filesystem_keys = {
            "available_blocks",
            "available_bytes",
            "block_size_bytes",
            "device",
            "fragment_size_bytes",
            "free_blocks",
            "free_inodes",
            "inode",
            "mode",
            "mount",
            "path",
            "total_blocks",
            "total_inodes",
        }
        filesystem_numeric_keys = filesystem_keys - {"mount", "path"}
        for role, expected_path, expected_information, current_vfs in (
            (
                "acquisition",
                acquisition_parent,
                root_information[0],
                acquisition_vfs,
            ),
            ("archive", archive_parent, root_information[1], archive_vfs),
        ):
            filesystem = filesystems[role]
            if (
                not isinstance(filesystem, dict)
                or set(filesystem) != filesystem_keys
                or any(
                    type(filesystem[field]) is not int
                    or filesystem[field] < 0
                    for field in filesystem_numeric_keys
                )
                or filesystem["path"] != str(expected_path)
                or filesystem["mount"]
                != _mount_identity_record(archive_mount)
                or filesystem["device"] != expected_information.st_dev
                or filesystem["inode"] != expected_information.st_ino
                or filesystem["mode"] != expected_information.st_mode
                or filesystem["fragment_size_bytes"] != current_vfs.f_frsize
                or filesystem["block_size_bytes"] != current_vfs.f_bsize
                or filesystem["available_bytes"]
                != filesystem["available_blocks"]
                * filesystem["fragment_size_bytes"]
            ):
                raise ControllerError(
                    "recovery_preflight_mount_binding", role
                )
        thermal_sensors = preflight["thermal_sensors"]
        expected_sensors = plan.safety.mandatory_thermal_sensors
        acquisition_thermal = {
            item["path"]: item["value"]
            for item in acquisition_sample["thermal_millicelsius"]
        }
        sensor_keys = {
            "device",
            "inode",
            "label",
            "minimum_plausible_millicelsius",
            "mode",
            "mtime_ns",
            "path",
            "value_millicelsius",
        }
        if not isinstance(thermal_sensors, list) or len(thermal_sensors) != len(
            expected_sensors
        ):
            raise ControllerError("recovery_preflight_sensor_population", stage)
        for index, expected_sensor in enumerate(expected_sensors):
            item = thermal_sensors[index]
            if (
                not isinstance(item, dict)
                or set(item) != sensor_keys
                or any(
                    type(item[field]) is not int or item[field] < 0
                    for field in (
                        "device",
                        "inode",
                        "minimum_plausible_millicelsius",
                        "mode",
                        "mtime_ns",
                        "value_millicelsius",
                    )
                )
                or item["label"] != expected_sensor.label
                or item["path"] != expected_sensor.path
                or item["minimum_plausible_millicelsius"]
                != expected_sensor.minimum_plausible_millicelsius
                or item["value_millicelsius"]
                != acquisition_thermal[expected_sensor.path]
            ):
                raise ControllerError("recovery_preflight_sensor_binding", index)
            try:
                sensor_information = os.lstat(expected_sensor.path)
            except OSError as error:
                raise ControllerError(
                    "recovery_preflight_sensor_stat", error
                ) from error
            if (
                stat.S_ISLNK(sensor_information.st_mode)
                or not stat.S_ISREG(sensor_information.st_mode)
                or (
                    sensor_information.st_dev,
                    sensor_information.st_ino,
                    sensor_information.st_mode,
                )
                != (item["device"], item["inode"], item["mode"])
            ):
                raise ControllerError(
                    "recovery_preflight_sensor_identity", expected_sensor.path
                )

        initialization = preflight["initial_workspace_evidence_safety"]
        if not isinstance(initialization, dict) or set(initialization) != {
            "allocation_rounding",
            "post_write",
            "prospective_file_bytes",
            "prospective_threshold_free_bytes",
            "prospective_write",
        }:
            raise ControllerError("recovery_initial_safety_shape", stage)
        descriptor_bytes = _descriptor_json_bytes(_PROCEDURE_DESCRIPTOR)
        pointer_bytes = canonical_json_bytes(
            {
                "archive_stage": str(stage),
                "run_id": stage.name[1:-7],
                "schema": "oai.memprof.r0.archive-stage-pointer/v1",
            }
        )
        expected_initial_file_bytes = [
            len(plan.canonical_bytes),
            len(descriptor_bytes),
            archive_vfs.f_frsize,
            len(pointer_bytes),
        ]
        expected_initial_threshold = _prospective_free_threshold_for_files(
            absolute_floor_bytes=plan.safety.minimum_free_during_bytes,
            baseline_free_bytes=preflight[
                "global_same_mount_baseline_free_bytes"
            ],
            maximum_decline_bytes=(
                plan.safety.maximum_observed_free_space_decline_bytes
            ),
            prospective_file_bytes=expected_initial_file_bytes,
            allocation_unit_bytes=archive_vfs.f_frsize,
        )
        prospective_sample = _validate_plan_safety_sample_receipt(
            initialization["prospective_write"],
            "recovery initial prospective write",
            plan,
            minimum_free_bytes=expected_initial_threshold,
        )
        post_sample = _validate_plan_safety_sample_receipt(
            initialization["post_write"],
            "recovery initial post write",
            plan,
            minimum_free_bytes=plan.safety.minimum_free_during_bytes,
        )
        if (
            initialization["allocation_rounding"]
            != "each_file_ceiling_to_fragment_size"
            or initialization["prospective_file_bytes"]
            != expected_initial_file_bytes
            or type(initialization["prospective_threshold_free_bytes"])
            is not int
            or initialization["prospective_threshold_free_bytes"]
            != expected_initial_threshold
            or prospective_sample["monotonic_ns"]
            < archive_sample["monotonic_ns"]
            or post_sample["monotonic_ns"]
            < prospective_sample["monotonic_ns"]
        ):
            raise ControllerError("recovery_initial_safety_binding", stage)
        archived_baseline = max(
            int(acquisition_sample["free_bytes"]),
            int(archive_sample["free_bytes"]),
        )
        timeline.baseline_free_bytes = archived_baseline
        self._force_timeline_sample(timeline)
        return (
            archived_baseline,
            archive_vfs.f_frsize,
            archive_mount,
        )

    def recover_incomplete(self, case_path: pathlib.Path) -> Mapping[str, object]:
        stage, run_id = self._resolve_recovery_stage(case_path)
        original = stage.parent / run_id
        recovered_id = f"{run_id}.recovered-incomplete"
        try:
            self.archive.validate_run_id(recovered_id)
        except BaseException as error:
            raise self._error("recovery_run_id", error) from error
        recovered = stage.parent / recovered_id
        stage_present = self._directory_presence(stage, "recovery_stage")
        original_present = self._directory_presence(
            original, "recovery_original_destination"
        )
        recovered_present = self._directory_presence(
            recovered, "recovery_incomplete_destination"
        )
        if not stage_present:
            if original_present and recovered_present:
                raise ControllerError(
                    "recovery_destination_conflict", f"{original} and {recovered}"
                )
            if recovered_present:
                result = dict(self.verify_archive(recovered))
                result["recovery_action"] = "VERIFY_EXISTING_RECOVERED"
                result["schema"] = "oai.memprof.r0.recovery-receipt/v1"
                return result
            if original_present:
                result = dict(self.verify_archive(original))
                result["recovery_action"] = "VERIFY_EXISTING_ORIGINAL"
                result["schema"] = "oai.memprof.r0.recovery-receipt/v1"
                return result
            raise ControllerError("recovery_stage_missing", stage)
        plan = load_plan(stage / "inputs/plan.json")
        verify_procedure_descriptor()
        timeline = self._new_live_safety_timeline(plan, stage.parent)
        try:
            descriptor_image = self._archive_call(
                timeline,
                "read-recovery-procedure",
                self.archive.read_regular_file_bounded,
                stage / "inputs/procedure.json",
                plan.execution.maximum_json_evidence_bytes,
            )
        except BaseException as error:
            raise self._error("recovery_procedure_read", error) from error
        expected_descriptor_bytes = _descriptor_json_bytes(_PROCEDURE_DESCRIPTOR)
        if descriptor_image.data != expected_descriptor_bytes:
            raise ControllerError(
                "recovery_procedure_binding",
                "archived procedure descriptor differs from this controller",
            )
        expected_stage = pathlib.Path(plan.roots.archive_parent) / f".{run_id}.active"
        if stage != expected_stage:
            raise ControllerError("recovery_stage_binding", f"{stage} != {expected_stage}")
        (
            recovery_baseline_free_bytes,
            recovery_allocation_unit_bytes,
            recovery_mount_identity,
        ) = self._recovery_storage_binding(plan, stage, timeline)
        manifest_arguments = self._manifest_arguments(plan)
        try:
            inspection = self._archive_call(
                timeline,
                "inspect-incomplete",
                self.archive.inspect_incomplete,
                stage,
                max_state_bytes=plan.execution.maximum_state_bytes,
                max_journal_entries=plan.execution.maximum_journal_entries,
                max_journal_entry_bytes=plan.execution.maximum_journal_entry_bytes,
                **manifest_arguments,
            )
        except BaseException as error:
            raise self._error("recovery_inspection", error) from error
        if recovered_present or original_present:
            raise ControllerError(
                "recovery_destination_exists",
                str(recovered if recovered_present else original),
            )
        if inspection.manifest_present and not inspection.manifest_verified:
            return {
                **self._recovered_axes(
                    verified=False, reason="RECOVERY_MANIFEST_INVALID"
                ).as_dict(),
                "archive_path": str(stage),
                "mutated": False,
                "recovery_action": "PRESERVED_INVALID_MANIFEST_ACTIVE",
                "recovery_reason": inspection.reason,
                "schema": "oai.memprof.r0.recovery-receipt/v1",
                "verified": False,
            }
        pre_marker_safety: dict[str, object] | None = None
        pre_manifest_safety: dict[str, object] | None = None
        if not inspection.manifest_present:
            recovery_value = {
                "original_run_id": run_id,
                "recovery_reason": inspection.reason or "INTERRUPTED_BEFORE_MANIFEST",
                "schema": "oai.memprof.r0.recovery/v1",
                "terminal_axes": self._recovered_axes(
                    verified=True, reason="RECOVERED_AFTER_INTERRUPTION"
                ).as_dict(),
            }
            recovery_bytes = canonical_json_bytes(recovery_value)
            if len(recovery_bytes) > plan.execution.maximum_json_evidence_bytes:
                raise ControllerError(
                    "recovery_marker_size",
                    f"{len(recovery_bytes)}>{plan.execution.maximum_json_evidence_bytes}",
                )
            recovery_path = stage / "recovery.json"
            recovery_marker_present = False
            try:
                recovery_information = os.lstat(recovery_path)
            except FileNotFoundError:
                pass
            except OSError as error:
                raise ControllerError("recovery_marker_stat", error) from error
            else:
                if stat.S_ISLNK(
                    recovery_information.st_mode
                ) or not stat.S_ISREG(recovery_information.st_mode):
                    raise ControllerError("recovery_marker_type", recovery_path)
                try:
                    recovery_image = self._archive_call(
                        timeline,
                        "read-recovery-marker",
                        self.archive.read_regular_file_bounded,
                        recovery_path,
                        plan.execution.maximum_json_evidence_bytes,
                    )
                except BaseException as error:
                    raise self._error("recovery_marker_read", error) from error
                if recovery_image.data != recovery_bytes:
                    raise ControllerError(
                        "recovery_marker_identity", recovery_path
                    )
                recovery_marker_present = True
            self._scan_recovery_population_and_reserve(
                plan,
                stage,
                timeline,
                len(recovery_bytes),
                recovery_marker_present,
                recovery_mount_identity,
            )
            try:
                if not recovery_marker_present:
                    marker_and_manifest_threshold = (
                        _prospective_free_threshold_for_files(
                            absolute_floor_bytes=(
                                plan.safety.minimum_free_during_bytes
                            ),
                            baseline_free_bytes=recovery_baseline_free_bytes,
                            maximum_decline_bytes=(
                                plan.safety.maximum_observed_free_space_decline_bytes
                            ),
                            prospective_file_bytes=(
                                plan.execution.maximum_json_evidence_bytes,
                                plan.execution.maximum_manifest_bytes,
                            ),
                            allocation_unit_bytes=(
                                recovery_allocation_unit_bytes
                            ),
                        )
                    )
                    pre_marker_sample = self._force_timeline_sample(
                        timeline,
                        minimum_free_bytes=marker_and_manifest_threshold,
                    )
                    pre_marker_safety = self._sample_record(pre_marker_sample)
                    recovery_record = self._archive_call(
                        timeline,
                        "write-recovery-marker",
                        self.archive.write_json_exclusive,
                        recovery_path,
                        recovery_value,
                        max_bytes=plan.execution.maximum_json_evidence_bytes,
                    )
                    if (
                        recovery_record.size != len(recovery_bytes)
                        or recovery_record.sha256
                        != hashlib.sha256(recovery_bytes).hexdigest()
                    ):
                        raise ControllerError(
                            "recovery_marker_write_identity", recovery_path
                        )
                manifest_threshold = _prospective_free_threshold(
                    absolute_floor_bytes=plan.safety.minimum_free_during_bytes,
                    baseline_free_bytes=recovery_baseline_free_bytes,
                    maximum_decline_bytes=(
                        plan.safety.maximum_observed_free_space_decline_bytes
                    ),
                    prospective_write_bytes=plan.execution.maximum_manifest_bytes,
                    allocation_unit_bytes=recovery_allocation_unit_bytes,
                )
                pre_manifest_sample = self._force_timeline_sample(
                    timeline,
                    minimum_free_bytes=manifest_threshold,
                )
                pre_manifest_safety = self._sample_record(pre_manifest_sample)
                self._archive_call(
                    timeline,
                    "create-recovery-manifest",
                    self.archive.create_manifest,
                    stage,
                    **manifest_arguments,
                    max_manifest_bytes=plan.execution.maximum_manifest_bytes,
                )
            except BaseException as error:
                raise self._error("recovery_seal", error) from error
        try:
            post_manifest_threshold = _prospective_free_threshold(
                absolute_floor_bytes=plan.safety.minimum_free_during_bytes,
                baseline_free_bytes=recovery_baseline_free_bytes,
                maximum_decline_bytes=(
                    plan.safety.maximum_observed_free_space_decline_bytes
                ),
                prospective_write_bytes=0,
                allocation_unit_bytes=recovery_allocation_unit_bytes,
            )
            post_manifest_sample = self._force_timeline_sample(
                timeline,
                minimum_free_bytes=post_manifest_threshold,
            )
            post_manifest_safety = self._sample_record(post_manifest_sample)
            summary = self._archive_call(
                timeline,
                "verify-recovery-manifest",
                self.archive.verify_manifest,
                stage,
                **manifest_arguments,
            )
            manifest_image = self._archive_call(
                timeline,
                "read-recovery-manifest",
                self.archive.read_regular_file_bounded,
                stage / "manifest.json",
                plan.execution.maximum_manifest_bytes,
            )
            if (
                summary.total_regular_file_bytes + manifest_image.size
                > plan.safety.maximum_archive_bytes
            ):
                raise ControllerError("recovery_archive_bound", stage)
            if self._recovery_storage_binding(plan, stage, timeline) != (
                recovery_baseline_free_bytes,
                recovery_allocation_unit_bytes,
                recovery_mount_identity,
            ):
                raise ControllerError(
                    "recovery_storage_binding_changed", stage
                )
            publication_result = self._archive_call(
                timeline,
                "publish-recovered-incomplete",
                self.archive.publish_verified_archive,
                stage,
                recovered,
                **manifest_arguments,
            )
            publication_phase = getattr(
                publication_result.phase, "value", publication_result.phase
            )
            if (
                publication_result.summary != summary
                or publication_phase
                != self.archive.PublicationPhase.VERIFIED.value
            ):
                raise ControllerError(
                    "recovery_publication_verification", recovered
                )
        except ControllerError:
            raise
        except BaseException as error:
            raise self._error("recovery_publication", error) from error
        result = dict(
            self.verify_archive(
                recovered,
                timeline=timeline,
                expected_plan=plan,
            )
        )
        if (
            result.get("verified") is not True
            or result.get("mutated") is not False
            or result.get("archive_path") != str(recovered)
            or result.get("scientific_case_state") != "INCOMPLETE"
            or result.get("inclusion") != "EXCLUDED"
        ):
            raise ControllerError(
                "recovery_postpublication_verification",
                "published incomplete archive did not independently reverify as incomplete",
            )
        result["mutated"] = True
        result["publication_phase"] = publication_phase
        result["recovery_action"] = "SEALED_AND_PUBLISHED_INCOMPLETE"
        result["recovery_safety"] = {
            "post_manifest": post_manifest_safety,
            "post_marker_pre_manifest": pre_manifest_safety,
            "prospective_marker_and_manifest": pre_marker_safety,
        }
        recovery_timeline = self._timeline_receipt(timeline)
        _validate_global_safety_timeline_receipt(
            recovery_timeline, "sealed recovery", plan
        )
        verification_timeline = _validate_global_safety_timeline_structure(
            result.get("verification_global_safety_timeline"),
            "sealed recovery postverification",
        )
        if canonical_json_bytes(verification_timeline) != canonical_json_bytes(
            recovery_timeline
        ):
            raise ControllerError(
                "recovery_verification_timeline_binding",
                "postverification and recovery timelines differ",
            )
        result["recovery_global_safety_timeline"] = dict(recovery_timeline)
        result["recovery_safety_plan_binding"] = {
            "maximum_poll_gap_milliseconds": (
                plan.safety.maximum_poll_gap_milliseconds
            ),
            "maximum_safety_samples": plan.safety.maximum_safety_samples,
            "plan_sha256": plan.sha256,
            "schema": "oai.memprof.r0.recovery-safety-plan-binding/v1",
        }
        result["schema"] = "oai.memprof.r0.recovery-receipt/v1"
        return result

def _default_services() -> ControllerServices:
    """Construct the production adapter over the frozen sibling cores."""

    return _CoreServices()


def _emit(value: object, maximum_bytes: int, stream: TextIO | None = None) -> bool:
    """Emit one bounded canonical record and report whether it is the full value."""

    payload = canonical_json_bytes(value) + b"\n"
    full_result = True
    if len(payload) > maximum_bytes:
        fallback = canonical_json_bytes(
            {
                "problem": {"code": "cli_output_limit", "message": "canonical result exceeds declared bound"},
                "schema": TERMINAL_RESULT_SCHEMA,
                "verdict": "invalid",
            }
        ) + b"\n"
        if len(fallback) > maximum_bytes:
            raise ControllerError("cli_output_limit", "declared CLI bound cannot hold the failure result")
        payload = fallback
        full_result = False
    output = sys.stdout if stream is None else stream
    output.write(payload.decode("utf-8", errors="strict"))
    output.flush()
    return full_result


class _CanonicalArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ControllerError("argument_parse_error", _bounded_text(message))


class _CatchableTermination:
    """Translate one catchable termination signal into normal Python unwinding."""

    def __init__(self) -> None:
        self.previous: dict[int, object] = {}

    @staticmethod
    def _raise_system_exit(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    def __enter__(self) -> "_CatchableTermination":
        try:
            if signal.getsignal(signal.SIGINT) is not signal.default_int_handler:
                raise ControllerError(
                    "signal_handler_precondition",
                    "SIGINT must retain Python's default KeyboardInterrupt mapping",
                )
            caller_mask = {
                int(item)
                for item in signal.pthread_sigmask(signal.SIG_BLOCK, set())
            }
        except (OSError, ValueError, AttributeError) as error:
            raise ControllerError("signal_handler_precondition", error) from error
        supported = {int(signal.SIGHUP), int(signal.SIGINT), int(signal.SIGTERM)}
        if caller_mask.intersection(supported):
            raise ControllerError(
                "signal_handler_precondition",
                "SIGHUP, SIGINT, and SIGTERM must enter the terminal CLI unblocked",
            )
        for signum in (signal.SIGTERM, signal.SIGHUP):
            try:
                self.previous[signum] = signal.getsignal(signum)
                signal.signal(signum, self._raise_system_exit)
            except (OSError, ValueError) as error:
                for installed, previous in self.previous.items():
                    signal.signal(installed, previous)
                self.previous.clear()
                raise ControllerError("signal_handler_install", error) from error
        return self

    def __exit__(self, _kind: object, _value: object, _traceback: object) -> None:
        failures: list[str] = []
        for signum, previous in self.previous.items():
            try:
                signal.signal(signum, previous)
            except (OSError, ValueError) as error:
                failures.append(f"{signum}:{error}")
        self.previous.clear()
        if failures and _value is not None and isinstance(_value, BaseException):
            _value.add_note("signal handler restoration failed: " + "; ".join(failures))
        elif failures:
            raise ControllerError("signal_handler_restore", "; ".join(failures))


def _parser() -> argparse.ArgumentParser:
    parser = _CanonicalArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    subparsers.add_parser("self-test")
    validate = subparsers.add_parser("validate-plan")
    validate.add_argument("--plan", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--plan", required=True)
    resume = subparsers.add_parser("resume")
    resume.add_argument("--case", required=True)
    verify = subparsers.add_parser("verify-archive")
    verify.add_argument("--archive", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, services: ControllerServices | None = None) -> int:
    maximum_output = FALLBACK_MAXIMUM_CLI_BYTES
    try:
        arguments = _parser().parse_args(sys.argv[1:] if argv is None else argv)
        if arguments.operation == "self-test":
            return 0 if _emit(controller_self_test(), maximum_output) else 2
        if arguments.operation in {"validate-plan", "run"}:
            plan = load_plan(arguments.plan)
            maximum_output = plan.execution.maximum_cli_output_bytes
            if arguments.operation == "validate-plan":
                verify_procedure_descriptor()
                validate_host_binding(plan)
                emitted = _emit(
                    {
                        "plan_sha256": plan.sha256,
                        "procedure_sha256": PROCEDURE_DESCRIPTOR_SHA256,
                        "schema": PLAN_SCHEMA,
                        "verdict": "pass",
                    },
                    maximum_output,
                )
                return 0 if emitted else 2
            with _CatchableTermination():
                result = TerminalController(services or _default_services()).run(plan)
            postcompletion_operational_failure = (
                _validate_postcompletion_operational_problem(
                    result.get("publication")
                )
            )
            if not _emit(result, maximum_output):
                return 2
            if postcompletion_operational_failure:
                return 2
            axes = result.get("axes", {})
            if isinstance(axes, dict) and axes.get("archive_integrity") == "VERIFIED":
                return 0 if axes.get("scientific_case_state") == "COMPLETE" else 1
            return 2
        controller = TerminalController(services or _default_services())
        with _CatchableTermination():
            if arguments.operation == "resume":
                result = controller.resume(arguments.case)
            else:
                result = controller.verify(arguments.archive)
        if not _emit(result, maximum_output):
            return 2
        if result.get("archive_integrity") != "VERIFIED":
            return 2
        return 0 if result.get("scientific_case_state") == "COMPLETE" else 1
    except (PlanError, GateError, ControllerError, OSError) as error:
        _emit(
            {
                "problem": {
                    "code": getattr(error, "code", "io_error"),
                    "message": _bounded_text(error),
                },
                "schema": TERMINAL_RESULT_SCHEMA,
                "verdict": "invalid",
            },
            maximum_output,
        )
        return 2
    except Exception as error:
        _emit(
            {
                "problem": {
                    "code": "internal_controller_failure",
                    "message": _bounded_text(
                        f"{type(error).__name__}: {error}"
                    ),
                },
                "schema": TERMINAL_RESULT_SCHEMA,
                "verdict": "invalid",
            },
            maximum_output,
        )
        return 2


if __name__ == "__main__":
    if (
        __spec__ is None
        or __spec__.name
        != "tools.profiling.memory.oai_memprof_r0_terminal"
    ):
        direct_failure = _descriptor_json_bytes(
            {
                "problem": {
                    "code": "direct_script_invocation_not_admitted",
                    "message": (
                        "use python -B -m tools.profiling.memory."
                        "oai_memprof_r0_terminal from the repository root"
                    ),
                },
                "schema": TERMINAL_RESULT_SCHEMA,
                "verdict": "invalid",
            }
        ) + b"\n"
        try:
            os.write(1, direct_failure)
        except OSError:
            os._exit(2)
        raise SystemExit(2)
    raise SystemExit(main())
