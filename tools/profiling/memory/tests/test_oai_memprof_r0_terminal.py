#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0

"""Independent plan, protocol, and injected-service tests for R0 terminal control."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.profiling.memory import oai_memprof_r0_terminal as terminal  # noqa: E402


CHILD = Path(__file__).with_name("r0_terminal_test_child.py").resolve()


def _pid_is_live(pid: int) -> bool:
    try:
        text = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii")
    except FileNotFoundError:
        return False
    right_parenthesis = text.rfind(")")
    fields = text[right_parenthesis + 2 :].split() if right_parenthesis >= 0 else []
    return not fields or fields[0] != "Z"


def _wait_pid_gone(pid: int, timeout_seconds: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _pid_is_live(pid):
            return True
        time.sleep(0.01)
    return not _pid_is_live(pid)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _valid_plan_value() -> dict[str, object]:
    tools = {
        "ar": "/usr/bin/ar",
        "cc": "/usr/bin/cc",
        "cmake": "/usr/bin/cmake",
        "ctest": "/usr/bin/ctest",
        "cxx": "/usr/bin/c++",
        "git": "/usr/bin/git",
        "ld": "/usr/bin/ld",
        "ninja": "/usr/bin/ninja",
        "nm": "/usr/bin/nm",
        "objdump": "/usr/bin/objdump",
        "python": "/usr/bin/python3",
        "readelf": "/usr/bin/readelf",
    }
    step_limits = {
        name: {
            "stderr_bytes": 1024 * 1024,
            "stdout_bytes": 1024 * 1024,
            "wall_seconds": 60,
        }
        for name in terminal.REQUIRED_STEP_LIMIT_NAMES
    }
    return {
        "case_label": "r0-x86-0001",
        "dependencies": {
            "benchmark": {"commit": "2" * 40, "path": "/tmp/benchmark"},
            "cpm_cmake": {
                "path": "/tmp/CPM.cmake",
                "sha256": hashlib.sha256(b"synthetic-cpm").hexdigest(),
            },
            "googletest": {"commit": "1" * 40, "path": "/tmp/googletest"},
        },
        "execution": {
            "build_jobs": 4,
            "clean_path": "/usr/bin",
            "cleanup_grace_seconds": 2,
            "ctest_jobs": 1,
            "maximum_archive_directories_excluding_root": 256,
            "maximum_archive_regular_files_excluding_manifest": 1024,
            "maximum_cli_output_bytes": 65536,
            "maximum_descendant_identities": 256,
            "maximum_evidence_bytes": 32 * 1024 * 1024,
            "maximum_json_evidence_bytes": 65536,
            "maximum_journal_entries": 256,
            "maximum_journal_entry_bytes": 65536,
            "maximum_manifest_bytes": 65536,
            "maximum_observed_live_descendants": 64,
            "maximum_proc_entries": 1000000,
            "maximum_state_bytes": 65536,
            "read_chunk_bytes": 4096,
            "step_limits": step_limits,
        },
        "expected_architecture": platform.machine(),
        "expected_commit": "a" * 40,
        "expected_hostname": socket.gethostname(),
        "roots": {
            "acquisition_parent": "/tmp/oai-r0-acquisition",
            "archive_parent": "/tmp/oai-r0-archive",
        },
        "safety": {
            "mandatory_thermal_sensors": [
                {
                    "label": "cpu",
                    "minimum_plausible_millicelsius": 1000,
                    "path": "/tmp/oai-r0-temperature",
                }
            ],
            "maximum_archive_bytes": 32 * 1024 * 1024,
            "maximum_poll_gap_milliseconds": 20,
            "maximum_safety_samples": 10000,
            "maximum_observed_free_space_decline_bytes": 64 * 1024 * 1024,
            "maximum_sensor_bytes": 32,
            "minimum_free_during_bytes": 1024 * 1024,
            "minimum_free_start_bytes": 2 * 1024 * 1024,
            "poll_interval_milliseconds": 10,
            "thermal_ceiling_millicelsius": 90000,
        },
        "schema": terminal.PLAN_SCHEMA,
        "source_repository": "/tmp/oai-source-repository",
        "tools": tools,
    }


def _workspace(root: Path) -> terminal.RunWorkspace:
    scratch = root / "scratch"
    archive_stage = root / "archive-stage"
    conditions = {
        name: terminal.ConditionPaths(
            name=name,
            root=scratch / "conditions" / name,
            build=scratch / "conditions" / name / "build",
            dependency_root=scratch / "conditions" / name / "dependencies",
            log_root=archive_stage / "conditions" / name / "logs",
        )
        for name, _, _ in terminal.R0_CONDITIONS
    }
    return terminal.RunWorkspace(
        run_id="r0-x86-0001",
        scratch=scratch,
        archive_stage=archive_stage,
        archive_destination=root / "archive-final",
        source=scratch / "source",
        temporary=scratch / "tmp",
        xdg_cache=scratch / "xdg-cache",
        xdg_config=scratch / "xdg-config",
        xdg_data=scratch / "xdg-data",
        xdg_state=scratch / "xdg-state",
        conditions=conditions,
    )


class PlanAndPureProtocolTests(unittest.TestCase):
    def plan(self) -> terminal.RunPlan:
        return terminal.parse_plan(_canonical(_valid_plan_value()))

    def test_procedure_descriptor_digest_matches_current_contract(self) -> None:
        terminal.verify_procedure_descriptor()

    def test_canonical_plan_round_trip_is_exact_and_immutable(self) -> None:
        raw = _canonical(_valid_plan_value())
        plan = terminal.parse_plan(raw)
        self.assertEqual(plan.canonical_bytes, raw)
        self.assertEqual(terminal.canonical_json_bytes(_valid_plan_value()), raw)
        self.assertEqual(len(plan.sha256), 64)
        with self.assertRaises(TypeError):
            plan.tools["git"] = "/mutated"  # type: ignore[index]
        with self.assertRaises(TypeError):
            plan.execution.step_limits["build"] = terminal.StepLimit(1, 1, 1)  # type: ignore[index]

    def test_plan_rejects_duplicate_unknown_missing_noncanonical_and_float(self) -> None:
        canonical = _canonical(_valid_plan_value())
        cases = (
            (b'{"schema":"x","schema":"y"}', "duplicate_key"),
            (canonical + b"\n", "noncanonical_json"),
            (canonical.replace(b'"build_jobs":4', b'"build_jobs":4.0'), "non_integer_number"),
        )
        for raw, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(terminal.PlanError) as context:
                    terminal.parse_plan(raw)
                self.assertEqual(context.exception.code, code)

        for mutation, code in (("unknown", "unknown_key"), ("missing", "missing_key")):
            value = _valid_plan_value()
            if mutation == "unknown":
                value["unexpected"] = 1
            else:
                del value["schema"]
            with self.subTest(mutation=mutation):
                with self.assertRaises(terminal.PlanError) as context:
                    terminal.parse_plan(_canonical(value))
                self.assertEqual(context.exception.code, code)

    def test_plan_semantic_mutants_fail_with_specific_causes(self) -> None:
        mutations = (
            (("case_label",), "../escape", "invalid_case_label"),
            (("expected_architecture",), "riscv64", "unsupported_architecture"),
            (("roots", "archive_parent"), "/tmp/oai-r0-acquisition", "root_alias"),
            (("tools", "cc"), "/usr/local/bin/cc", "nonfrozen_compiler_driver"),
            (("safety", "thermal_ceiling_millicelsius"), 89999, "nonfrozen_thermal_ceiling"),
            (("safety", "maximum_poll_gap_milliseconds"), 9, "invalid_poll_gap"),
            (("safety", "minimum_free_start_bytes"), 1, "invalid_storage_threshold"),
            (("safety", "maximum_archive_bytes"), 65 * 1024 * 1024, "invalid_storage_bound"),
            (("execution", "build_jobs"), 3, "nonfrozen_parallelism"),
            (("execution", "ctest_jobs"), 2, "nonfrozen_parallelism"),
        )
        for path, replacement, code in mutations:
            with self.subTest(path=path, code=code):
                value = _valid_plan_value()
                target = value
                for component in path[:-1]:
                    target = target[component]  # type: ignore[index,assignment]
                target[path[-1]] = replacement  # type: ignore[index]
                with self.assertRaises(terminal.PlanError) as context:
                    terminal.parse_plan(_canonical(value))
                self.assertEqual(context.exception.code, code)

    def test_clean_environment_has_exact_population_and_no_ambient_inheritance(self) -> None:
        plan = self.plan()
        with tempfile.TemporaryDirectory(prefix="oai-r0-terminal-pure-") as temporary:
            workspace = _workspace(Path(temporary))
            ambient = "OAI_MEMPROF_AMBIENT_MUST_NOT_LEAK"
            with mock.patch.dict(os.environ, {ambient: "secret"}):
                observed = terminal.build_clean_environment(plan, workspace, "OFF0")
        self.assertNotIn(ambient, observed)
        self.assertEqual(
            set(observed),
            {
                "CC",
                "CPM_SOURCE_CACHE",
                "CXX",
                "GIT_CONFIG_GLOBAL",
                "GIT_CONFIG_NOSYSTEM",
                "GIT_TERMINAL_PROMPT",
                "HOME",
                "LANG",
                "LC_ALL",
                "PATH",
                "PYTHONDONTWRITEBYTECODE",
                "PYTHONHASHSEED",
                "PYTHONNOUSERSITE",
                "TMPDIR",
                "TZ",
                "XDG_CACHE_HOME",
                "XDG_CONFIG_HOME",
                "XDG_DATA_HOME",
                "XDG_STATE_HOME",
            },
        )
        self.assertEqual(observed["HOME"], terminal._validated_host_home())
        self.assertEqual(observed["PATH"], "/usr/bin")
        self.assertEqual(observed["LC_ALL"], "C")
        self.assertEqual(observed["TZ"], "UTC")

    def test_exact_condition_command_expansion(self) -> None:
        plan = self.plan()
        with tempfile.TemporaryDirectory(prefix="oai-r0-terminal-builders-") as temporary:
            workspace = _workspace(Path(temporary))
            for condition, tests, physim in terminal.R0_CONDITIONS:
                argv = terminal.configure_argv(plan, workspace, condition)
                self.assertEqual(argv[:6], (
                    "/usr/bin/cmake",
                    "-S",
                    str(workspace.source),
                    "-B",
                    str(workspace.conditions[condition].build),
                    "-G",
                ))
                self.assertIn("Ninja", argv)
                self.assertIn("-DENABLE_TESTS={}".format("ON" if tests else "OFF"), argv)
                self.assertIn(
                    "-DENABLE_PHYSIM_TESTS={}".format("ON" if physim else "OFF"),
                    argv,
                )
                self.assertIn("-DFETCHCONTENT_FULLY_DISCONNECTED=ON", argv)
                self.assertIn("-DFETCHCONTENT_UPDATES_DISCONNECTED=ON", argv)

            build = terminal.build_argv(plan, workspace)
            self.assertEqual(build[:5], (
                "/usr/bin/ninja",
                "-C",
                str(workspace.conditions["ON"].build),
                "-j",
                "4",
            ))
            self.assertEqual(build[5:], terminal.R0_BUILD_TARGETS)
            self.assertEqual(
                terminal.ctest_inventory_argv(plan, workspace, "r0")[-1],
                "^oai_memprof_r0_",
            )
            self.assertEqual(
                terminal.ctest_inventory_argv(plan, workspace, "wire")[-1],
                "^oai_memprof_wire_",
            )
            self.assertEqual(
                terminal.ctest_run_argv(plan, workspace, "r0")[3:7],
                ("--verbose", "--no-tests=error", "-j", "1"),
            )

    def test_ctest_parsers_require_exact_population_and_success_summary(self) -> None:
        expected = terminal.WIRE_CTESTS
        inventory = (
            "Test project /tmp/build\n"
            + "\n".join(
                "  Test #{}: {}".format(index, name)
                for index, name in enumerate(expected, 1)
            )
            + "\n\nTotal Tests: {}\n".format(len(expected))
        )
        self.assertEqual(terminal.parse_ctest_inventory(inventory, expected), expected)
        passing = (
            "\n".join(
                f"{index}/3 Test #{index}: {name} .... Passed 0.01 sec"
                for index, name in enumerate(expected, 1)
            )
            + "\n100% tests passed, 0 tests failed out of 3\n"
        )
        terminal.validate_ctest_run(passing, expected)
        for mutant in (
            inventory.replace(expected[0], "wrong", 1),
            inventory.replace("Total Tests: 3", "Total Tests: 2"),
            inventory + "Total Tests: 3\n",
        ):
            with self.assertRaises(terminal.GateError):
                terminal.parse_ctest_inventory(mutant, expected)
        with self.assertRaises(terminal.GateError):
            terminal.validate_ctest_run(
                passing.replace("Passed 0.01 sec", "Skipped 0.01 sec", 1),
                expected,
            )

    def test_terminal_axes_keep_scientific_integrity_and_inclusion_independent(self) -> None:
        complete = {name: True for name, _, _ in terminal.R0_CONDITIONS}
        decisions = (
            terminal.AbsenceDecision(
                "OFF0", True, "QUALIFIED", "REFERENCE_TOOL_MATCH"
            ),
            terminal.AbsenceDecision(
                "OFF1", True, "QUALIFIED", "REFERENCE_TOOL_MATCH"
            ),
        )
        included = terminal.classify_terminal_result(
            condition_complete=complete,
            archive_verified=True,
            interrupted=False,
            absence_decisions=decisions,
            positive_elf_complete=True,
        )
        self.assertEqual(included.scientific_case_state, "COMPLETE")
        self.assertEqual(included.archive_integrity, "VERIFIED")
        self.assertEqual(included.inclusion, "INCLUDED")

        unverified = terminal.classify_terminal_result(
            condition_complete=complete,
            archive_verified=False,
            interrupted=False,
            absence_decisions=decisions,
            positive_elf_complete=True,
        )
        self.assertEqual(unverified.scientific_case_state, "COMPLETE")
        self.assertEqual(unverified.archive_integrity, "UNVERIFIED")
        self.assertEqual(unverified.inclusion, "EXCLUDED")

        interrupted = terminal.classify_terminal_result(
            condition_complete=complete,
            archive_verified=True,
            interrupted=True,
            absence_decisions=decisions,
            positive_elf_complete=True,
        )
        self.assertEqual(interrupted.scientific_case_state, "INCOMPLETE")
        self.assertEqual(interrupted.archive_integrity, "VERIFIED")
        self.assertEqual(interrupted.inclusion, "EXCLUDED")

    def test_terminal_axes_tuple_fields_round_trip_through_json_lists(self) -> None:
        complete = {name: True for name, _, _ in terminal.R0_CONDITIONS}
        axes = terminal.classify_terminal_result(
            condition_complete=complete,
            archive_verified=True,
            interrupted=False,
            absence_decisions=(
                terminal.AbsenceDecision(
                    "OFF0", True, "QUALIFIED", "REFERENCE_TOOL_MATCH"
                ),
                terminal.AbsenceDecision(
                    "OFF1", True, "QUALIFIED", "REFERENCE_TOOL_MATCH"
                ),
            ),
            positive_elf_complete=True,
        )

        wire_value = json.loads(terminal.canonical_json_bytes(axes.as_dict()))
        self.assertIsInstance(wire_value["reasons"], list)
        self.assertTrue(
            all(isinstance(claim["reasons"], list) for claim in wire_value["claims"])
        )
        restored = terminal._CoreServices._axes_from_mapping(wire_value)
        self.assertEqual(restored, axes)
        self.assertIsInstance(restored.reasons, tuple)
        self.assertIsInstance(restored.claims, tuple)
        self.assertTrue(
            all(isinstance(claim.reasons, tuple) for claim in restored.claims)
        )

        reasons_as_tuple = copy.deepcopy(wire_value)
        reasons_as_tuple["reasons"] = tuple(reasons_as_tuple["reasons"])
        with self.assertRaises(terminal.GateError) as context:
            terminal._CoreServices._axes_from_mapping(reasons_as_tuple)
        self.assertEqual(context.exception.code, "terminal_reason_shape")

        claim_reasons_as_tuple = copy.deepcopy(wire_value)
        claim_reasons_as_tuple["claims"][0]["reasons"] = tuple(
            claim_reasons_as_tuple["claims"][0]["reasons"]
        )
        with self.assertRaises(terminal.GateError) as context:
            terminal._CoreServices._axes_from_mapping(claim_reasons_as_tuple)
        self.assertEqual(context.exception.code, "terminal_claim_reasons")

    def test_absence_zero_counts_reject_json_false(self) -> None:
        source = Path("/tmp/oai-r0-source")
        build = Path("/tmp/oai-r0-build")
        identities = tuple(
            terminal.EvidenceIdentity(
                role=role,
                path=str(build / filename),
                bytes=0,
                lines=0,
                sha256=hashlib.sha256(b"").hexdigest(),
                device=1,
                inode=index,
            )
            for index, (role, filename) in enumerate(
                (
                    ("build_ninja", "build.ninja"),
                    ("targets", "targets.txt"),
                    ("commands", "commands.txt"),
                ),
                1,
            )
        )
        value = {
            "catalog_digest": terminal.ABSENCE_CATALOG_SHA256,
            "catalog_entries": terminal.ABSENCE_CATALOG_ENTRIES,
            "catalog_version": terminal.ABSENCE_CATALOG_VERSION,
            "claim": terminal.ABSENCE_CLAIM,
            "evidence": [
                {
                    "bytes": identity.bytes,
                    "device": identity.device,
                    "inode": identity.inode,
                    "lines": identity.lines,
                    "path": identity.path,
                    "role": identity.role,
                    "sha256": identity.sha256,
                }
                for identity in identities
            ],
            "expected_final_targets": list(terminal.ABSENCE_EXPECTED_FINAL_TARGETS),
            "final_link_model": terminal.ABSENCE_FINAL_LINK_MODEL,
            "match_count": 0,
            "matches": [],
            "matches_omitted": 0,
            "provenance": terminal.ABSENCE_PROVENANCE,
            "root_model": terminal.ABSENCE_ROOT_MODEL,
            "roots": {"build": str(build), "source": str(source)},
            "schema": terminal.ABSENCE_SCHEMA,
            "verdict": "pass",
            "wrap_model": terminal.ABSENCE_WRAP_MODEL,
        }
        decision = terminal.validate_absence_result(
            terminal.canonical_json_bytes(value) + b"\n",
            source_root=source,
            build_root=build,
            identities=identities,
            ld_version_output=b"GNU ld (GNU Binutils for Ubuntu) 2.42\n",
            condition="OFF0",
        )
        self.assertTrue(decision.tool_reference_match)

        for field in ("match_count", "matches_omitted"):
            with self.subTest(field=field):
                mutant = copy.deepcopy(value)
                mutant[field] = False
                with self.assertRaises(terminal.GateError) as context:
                    terminal.validate_absence_result(
                        terminal.canonical_json_bytes(mutant) + b"\n",
                        source_root=source,
                        build_root=build,
                        identities=identities,
                        ld_version_output=(
                            b"GNU ld (GNU Binutils for Ubuntu) 2.42\n"
                        ),
                        condition="OFF0",
                    )
                self.assertEqual(context.exception.code, "absence_identity")

    def test_prospective_free_threshold_uses_one_global_floor_and_per_file_rounding(self) -> None:
        observed = terminal._prospective_free_threshold_for_files(
            absolute_floor_bytes=1024,
            baseline_free_bytes=8192,
            maximum_decline_bytes=4096,
            prospective_file_bytes=(0, 1, 4096, 4097),
            allocation_unit_bytes=4096,
        )
        self.assertEqual(observed, 4096 + 0 + 4096 + 4096 + 8192)
        self.assertEqual(
            terminal._prospective_free_threshold(
                absolute_floor_bytes=10000,
                baseline_free_bytes=8192,
                maximum_decline_bytes=4096,
                prospective_write_bytes=1,
                allocation_unit_bytes=4096,
            ),
            10000 + 4096,
        )
        self.assertEqual(terminal._round_allocation_bytes(4096, 4096), 4096)
        self.assertEqual(terminal._round_allocation_bytes(4097, 4096), 8192)

        for field in (
            "absolute_floor_bytes",
            "baseline_free_bytes",
            "maximum_decline_bytes",
            "allocation_unit_bytes",
        ):
            with self.subTest(field=field):
                arguments = {
                    "absolute_floor_bytes": 1024,
                    "baseline_free_bytes": 8192,
                    "maximum_decline_bytes": 4096,
                    "prospective_file_bytes": (1,),
                    "allocation_unit_bytes": 4096,
                }
                arguments[field] = False
                with self.assertRaises(terminal.ControllerError) as context:
                    terminal._prospective_free_threshold_for_files(**arguments)
                self.assertEqual(context.exception.code, "free_threshold_argument")

        with self.assertRaises(terminal.ControllerError) as context:
            terminal._prospective_free_threshold_for_files(
                absolute_floor_bytes=1024,
                baseline_free_bytes=8192,
                maximum_decline_bytes=4096,
                prospective_file_bytes=(False,),
                allocation_unit_bytes=4096,
            )
        self.assertEqual(context.exception.code, "free_threshold_argument")

    def test_large_parser_paths_propagate_in_operation_timeline_failure(
        self,
    ) -> None:
        target_output = (
            b"target: phony\n"
            * (2 * terminal.PARSER_PROGRESS_CHUNK_BYTES // 14 + 1)
        )
        inventory = (
            b"Test project /tmp/build\n"
            b"  Test #1: one\n"
            b"\nTotal Tests: 1\n"
        )
        test_run = (
            b"1/1 Test #1: one .... Passed 0.01 sec\n"
            b"100% tests passed, 0 tests failed out of 1\n"
        )
        operations = (
            lambda callback: terminal._iter_bounded_utf8_lines(
                target_output,
                "target-cadence",
                progress_callback=callback,
            ),
            lambda callback: terminal.parse_ctest_inventory(
                inventory,
                ("one",),
                progress_callback=callback,
            ),
            lambda callback: terminal.validate_ctest_run(
                test_run,
                ("one",),
                progress_callback=callback,
            ),
        )
        for operation in operations:
            with self.subTest(operation=operation):
                failure = terminal.ControllerError(
                    "maximum_poll_gap", "injected intra-parse clock jump"
                )
                callback_count = 0

                def progress() -> None:
                    nonlocal callback_count
                    callback_count += 1
                    if callback_count == 2:
                        raise failure

                with self.assertRaises(terminal.ControllerError) as raised:
                    operation(progress)
                self.assertIs(raised.exception, failure)
                self.assertEqual(raised.exception.code, "maximum_poll_gap")
                self.assertEqual(callback_count, 2)


class FakeControllerServices:
    """Deterministic service double that still exercises every controller gate."""

    def __init__(
        self,
        root: Path,
        *,
        fail_instance: str | None = None,
        interrupt_instance: str | None = None,
        interrupt_checkpoint_phase: str | None = None,
        checkpoint_failure_phase: str | None = None,
        interruption: BaseException | None = None,
        publish_verified: bool = True,
        capture_outside_archive: bool = False,
        first_artifact_bytes: int | None = None,
        progress_failure_after: int | None = None,
        publication_receipt_mutator=None,
    ) -> None:
        self.workspace = _workspace(root)
        self.fail_instance = fail_instance
        self.interrupt_instance = interrupt_instance
        self.interrupt_checkpoint_phase = interrupt_checkpoint_phase
        self.checkpoint_failure_phase = checkpoint_failure_phase
        self.interruption = interruption or KeyboardInterrupt("injected cancellation")
        self.publish_verified = publish_verified
        self.capture_outside_archive = capture_outside_archive
        self.first_artifact_bytes = first_artifact_bytes
        self.progress_failure_after = progress_failure_after
        self.publication_receipt_mutator = publication_receipt_mutator
        self.calls: list[dict[str, object]] = []
        self.checkpoints: list[dict[str, object]] = []
        self.json_records: dict[str, object] = {}
        self.captures: dict[str, terminal.EvidenceIdentity] = {}
        self.publications: list[terminal.TerminalAxes] = []
        self.publication_receipts: list[dict[str, object]] = []
        self.read_workspaces: list[terminal.RunWorkspace] = []
        self.progress_workspaces: list[terminal.RunWorkspace] = []
        self.artifact_validation_workspaces: list[
            terminal.RunWorkspace
        ] = []
        self.plan: terminal.RunPlan | None = None
        self._interruption_delivered = False
        self._checkpoint_failure_delivered = False

    def create_workspace(
        self,
        plan: terminal.RunPlan,
        descriptor: object,
    ) -> terminal.RunWorkspace:
        del descriptor
        self.plan = plan
        for path in (
            self.workspace.scratch,
            self.workspace.archive_stage,
            self.workspace.source,
            self.workspace.temporary,
            self.workspace.xdg_cache,
            self.workspace.xdg_config,
            self.workspace.xdg_data,
            self.workspace.xdg_state,
        ):
            path.mkdir(parents=True, exist_ok=True)
        for paths in self.workspace.conditions.values():
            for path in (paths.root, paths.build, paths.dependency_root, paths.log_root):
                path.mkdir(parents=True, exist_ok=True)
        return self.workspace

    def read_regular(
        self,
        workspace: terminal.RunWorkspace,
        path: Path,
        maximum_bytes: int,
    ) -> terminal.BoundedFileImage:
        if workspace is not self.workspace:
            raise AssertionError("bounded read used a different workspace")
        self.read_workspaces.append(workspace)
        if path.name == "CPM_0.40.1.cmake":
            data = b"synthetic-cpm"
            information = os.lstat(self.workspace.scratch)
        else:
            data = path.read_bytes()
            information = os.lstat(path)
        if len(data) > maximum_bytes:
            raise OSError("fake bounded read exceeded")
        return terminal.BoundedFileImage(
            data=data,
            device=information.st_dev,
            inode=information.st_ino,
            size=len(data),
            mode=information.st_mode,
            mtime_ns=information.st_mtime_ns,
            ctime_ns=information.st_ctime_ns,
            sha256=hashlib.sha256(data).hexdigest(),
        )

    def progress(self, workspace: terminal.RunWorkspace) -> None:
        if workspace is not self.workspace:
            raise AssertionError("parser progress used a different workspace")
        self.progress_workspaces.append(workspace)
        if (
            self.progress_failure_after is not None
            and len(self.progress_workspaces) >= self.progress_failure_after
        ):
            raise terminal.ControllerError(
                "maximum_poll_gap", "injected parser timeline gap"
            )

    def write_json(
        self,
        workspace: terminal.RunWorkspace,
        relative_path: str,
        value: object,
    ) -> None:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise AssertionError("write_json escaped archive-stage domain")
        destination = workspace.archive_stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(terminal.canonical_json_bytes(value))
        self.json_records[relative_path] = copy.deepcopy(value)

    def copy_scratch_regular(
        self,
        workspace: terminal.RunWorkspace,
        source: Path,
        destination: Path,
        maximum_bytes: int,
    ) -> terminal.EvidenceIdentity:
        del source
        if not destination.is_absolute():
            raise AssertionError("scratch destination is not absolute")
        destination.relative_to(workspace.scratch)
        with self.assert_not_below(destination, workspace.archive_stage):
            pass
        data = b"synthetic-cpm"
        if len(data) > maximum_bytes:
            raise OSError("copy bound exceeded")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return self._identity(destination, "scratch-copy", data)

    @contextlib.contextmanager
    def assert_not_below(self, path: Path, root: Path):
        try:
            path.relative_to(root)
        except ValueError:
            yield
            return
        raise AssertionError("scratch copy entered archive-stage domain")

    def capture_regular(
        self,
        workspace: terminal.RunWorkspace,
        source: Path,
        archive_relative_path: str,
        maximum_bytes: int,
    ) -> terminal.EvidenceIdentity:
        relative = Path(archive_relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise AssertionError("capture path is not archive-relative")
        base = workspace.scratch if self.capture_outside_archive else workspace.archive_stage
        destination = base / relative
        if source.exists():
            data = source.read_bytes()
        else:
            data = self._artifact_bytes(source)
        if len(data) > maximum_bytes:
            raise OSError("capture bound exceeded")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        identity = self._identity(destination, relative.name, data)
        self.captures[archive_relative_path] = identity
        return identity

    def validate_captured_artifact(
        self,
        workspace: terminal.RunWorkspace,
        path: Path,
        relative_path: str,
        identity: terminal.EvidenceIdentity,
        architecture: str,
        *,
        maximum_bytes: int,
        read_chunk_bytes: int,
    ) -> None:
        if workspace is not self.workspace:
            raise AssertionError(
                "artifact validation used a different workspace"
            )
        self.artifact_validation_workspaces.append(workspace)
        terminal.validate_captured_artifact_stream(
            path,
            relative_path,
            identity,
            architecture,
            maximum_bytes=maximum_bytes,
            read_chunk_bytes=read_chunk_bytes,
        )

    def _artifact_bytes(self, source: Path) -> bytes:
        for relative in terminal.R0_ARTIFACTS:
            if str(source).endswith(relative):
                if relative in terminal.ELF_ARTIFACTS:
                    size = (
                        self.first_artifact_bytes
                        if relative == terminal.R0_ARTIFACTS[0]
                        and self.first_artifact_bytes is not None
                        else 20
                    )
                    data = bytearray(max(20, size))
                    data[:6] = b"\x7fELF\x02\x01"
                    machine = 62 if platform.machine() == "x86_64" else 183
                    data[18:20] = machine.to_bytes(2, "little")
                    return bytes(data)
                if relative in terminal.ARCHIVE_ARTIFACTS:
                    return b"!<arch>\n"
                if relative in terminal.MAP_ARTIFACTS:
                    return b"synthetic map\n"
        raise FileNotFoundError(source)

    @staticmethod
    def _identity(path: Path, role: str, data: bytes) -> terminal.EvidenceIdentity:
        information = path.stat()
        return terminal.EvidenceIdentity(
            role=role,
            path=str(path),
            bytes=len(data),
            lines=len(data.splitlines()),
            sha256=hashlib.sha256(data).hexdigest(),
            device=information.st_dev,
            inode=information.st_ino,
        )

    def run_step(
        self,
        workspace: terminal.RunWorkspace,
        plan: terminal.RunPlan,
        step: str,
        instance: str,
        argv,
        cwd: Path,
        environment,
    ) -> dict[str, object]:
        if instance == self.interrupt_instance and not self._interruption_delivered:
            self._interruption_delivered = True
            raise self.interruption
        stdout = self._stdout_for(workspace, plan, instance, tuple(argv))
        stderr = b""
        output_root = workspace.scratch / "command-output"
        output_root.mkdir(parents=True, exist_ok=True)
        stdout_path = output_root / (instance + ".stdout")
        stderr_path = output_root / (instance + ".stderr")
        stdout_path.write_bytes(stdout)
        stderr_path.write_bytes(stderr)
        result = {
            "instance": instance,
            "stderr": stderr,
            "stderr_path": str(stderr_path),
            "stdout": stdout,
            "stdout_path": str(stdout_path),
            "succeeded": instance != self.fail_instance,
        }
        self.calls.append(
            {
                "argv": tuple(argv),
                "cwd": cwd,
                "environment": dict(environment),
                "instance": instance,
                "step": step,
            }
        )
        return result

    def _stdout_for(
        self,
        workspace: terminal.RunWorkspace,
        plan: terminal.RunPlan,
        instance: str,
        argv: tuple[str, ...],
    ) -> bytes:
        del argv
        if instance == "controller-selftest":
            return terminal.canonical_json_bytes(
                {
                    "checks": 13,
                    "procedure_sha256": terminal.PROCEDURE_DESCRIPTOR_SHA256,
                    "schema": "oai.memprof.r0.terminal-self-test/v1",
                    "verdict": "pass",
                }
            ) + b"\n"
        if instance in {
            "source-invoker-head",
            "source-clone-head",
            "source-invoker-post-head",
            "source-clone-post-head",
        }:
            return (plan.expected_commit + "\n").encode("ascii")
        if instance == "source-submodule-status":
            return ("-" + "4" * 40 + " openair2/E2AP/flexric\n").encode("ascii")
        if instance.startswith("controller-committed-") or instance.startswith(
            "controller-running-"
        ):
            return ("5" * 40 + "\n").encode("ascii")
        if instance == "tool-ld-version":
            return b"GNU ld (GNU Binutils for Ubuntu) 2.42\n"
        if instance.startswith("tool-") and instance.endswith("-version"):
            return (instance + " synthetic version\n").encode("ascii")
        if instance.endswith("-googletest-head") or instance.endswith(
            "-googletest-post-head"
        ):
            return (plan.dependencies.googletest.commit + "\n").encode("ascii")
        if instance.endswith("-benchmark-head") or instance.endswith(
            "-benchmark-post-head"
        ):
            return (plan.dependencies.benchmark.commit + "\n").encode("ascii")
        if instance.endswith("-configure"):
            condition = instance.split("-", 1)[0].upper()
            self._write_cmake_cache(workspace, plan, condition)
            (workspace.conditions[condition].build / "build.ninja").write_bytes(
                b"synthetic build graph\n"
            )
            return b""
        if instance in {"off0-targets", "off1-targets"}:
            return b"nr-softmodem: phony\nnr-uesoftmodem: phony\n"
        if instance in {"off0-commands", "off1-commands"}:
            return b"cc -o nr-softmodem objects\ncc -o nr-uesoftmodem objects\n"
        if instance in {"off0-absence", "off1-absence"}:
            condition = instance.split("-", 1)[0].upper()
            return self._absence_result(workspace, condition)
        if instance == "on-targets":
            return "".join(
                "{}: phony\n".format(target) for target in terminal.R0_BUILD_TARGETS
            ).encode("utf-8")
        if instance == "on-r0-inventory":
            return self._ctest_inventory(terminal.R0_CTESTS)
        if instance == "on-wire-inventory":
            return self._ctest_inventory(terminal.WIRE_CTESTS)
        if instance == "on-r0-ctest":
            return self._ctest_run(terminal.R0_CTESTS)
        if instance == "on-wire-ctest":
            return self._ctest_run(terminal.WIRE_CTESTS)
        if instance == "on-positive-elf":
            return (
                "R0_ELF_GATE_V2 pass architecture={} roles=4 wrappers=4 "
                "runtime_soname=liboai_memprof_runtime.so.1 generator=Ninja "
                "compiler=GNU\n".format(plan.expected_architecture)
            ).encode("ascii")
        return b""

    def _write_cmake_cache(
        self,
        workspace: terminal.RunWorkspace,
        plan: terminal.RunPlan,
        condition: str,
    ) -> None:
        argv = terminal.configure_argv(plan, workspace, condition)
        definitions = {
            item[2:].split("=", 1)[0]: item[2:].split("=", 1)[1]
            for item in argv
            if item.startswith("-D")
        }
        definitions.update(
            {
                "CMAKE_CACHEFILE_DIR": str(workspace.conditions[condition].build),
                "CMAKE_GENERATOR": "Ninja",
                "CMAKE_HOME_DIRECTORY": str(workspace.source),
            }
        )
        text = "".join(
            "{}:STRING={}\n".format(name, value)
            for name, value in sorted(definitions.items())
        )
        (workspace.conditions[condition].build / "CMakeCache.txt").write_text(
            text, encoding="utf-8"
        )

    @staticmethod
    def _ctest_inventory(expected) -> bytes:
        return (
            "Test project /tmp/build\n"
            + "\n".join(
                "  Test #{}: {}".format(index, name)
                for index, name in enumerate(expected, 1)
            )
            + "\n\nTotal Tests: {}\n".format(len(expected))
        ).encode("utf-8")

    @staticmethod
    def _ctest_run(expected) -> bytes:
        count = len(expected)
        return (
            "\n".join(
                f"{index}/{count} Test #{index}: {name} .... Passed 0.01 sec"
                for index, name in enumerate(expected, 1)
            )
            + f"\n100% tests passed, 0 tests failed out of {count}\n"
        ).encode("utf-8")

    def _absence_result(
        self, workspace: terminal.RunWorkspace, condition: str
    ) -> bytes:
        identities = []
        for role, filename in (
            ("build_ninja", "build.ninja"),
            ("targets", "targets.txt"),
            ("commands", "commands.txt"),
        ):
            identity = self.captures[
                "conditions/{}/absence/{}".format(condition, filename)
            ]
            identities.append(
                {
                    "bytes": identity.bytes,
                    "device": identity.device,
                    "inode": identity.inode,
                    "lines": identity.lines,
                    "path": identity.path,
                    "role": role,
                    "sha256": identity.sha256,
                }
            )
        value = {
            "catalog_digest": terminal.ABSENCE_CATALOG_SHA256,
            "catalog_entries": terminal.ABSENCE_CATALOG_ENTRIES,
            "catalog_version": terminal.ABSENCE_CATALOG_VERSION,
            "claim": terminal.ABSENCE_CLAIM,
            "evidence": identities,
            "expected_final_targets": list(terminal.ABSENCE_EXPECTED_FINAL_TARGETS),
            "final_link_model": terminal.ABSENCE_FINAL_LINK_MODEL,
            "match_count": 0,
            "matches": [],
            "matches_omitted": 0,
            "provenance": terminal.ABSENCE_PROVENANCE,
            "root_model": terminal.ABSENCE_ROOT_MODEL,
            "roots": {
                "build": str(workspace.conditions[condition].build),
                "source": str(workspace.source),
            },
            "schema": terminal.ABSENCE_SCHEMA,
            "verdict": "pass",
            "wrap_model": terminal.ABSENCE_WRAP_MODEL,
        }
        return terminal.canonical_json_bytes(value) + b"\n"

    def command_stdout(self, result: dict[str, object], maximum_bytes: int) -> bytes:
        value = result["stdout"]
        assert isinstance(value, bytes)
        if len(value) > maximum_bytes:
            raise OSError("stdout bound exceeded")
        return value

    def command_stderr(self, result: dict[str, object], maximum_bytes: int) -> bytes:
        value = result["stderr"]
        assert isinstance(value, bytes)
        if len(value) > maximum_bytes:
            raise OSError("stderr bound exceeded")
        return value

    @staticmethod
    def command_succeeded(result: dict[str, object]) -> bool:
        return result["succeeded"] is True

    @staticmethod
    def command_record(result: dict[str, object]) -> dict[str, object]:
        return {
            "stderr_capture_path_during_acquisition": result["stderr_path"],
            "stderr_path": result["stderr_path"],
            "stdout_capture_path_during_acquisition": result["stdout_path"],
            "stdout_path": result["stdout_path"],
            "succeeded": result["succeeded"],
        }

    def checkpoint(
        self,
        workspace: terminal.RunWorkspace,
        state: object,
    ) -> None:
        del workspace
        assert isinstance(state, dict)
        self.checkpoints.append(copy.deepcopy(state))
        phase = state.get("phase")
        if (
            phase == self.interrupt_checkpoint_phase
            and not self._interruption_delivered
        ):
            self._interruption_delivered = True
            raise self.interruption
        if (
            phase == self.checkpoint_failure_phase
            and not self._checkpoint_failure_delivered
        ):
            self._checkpoint_failure_delivered = True
            raise terminal.ControllerError(
                "injected_checkpoint_failure", str(phase)
            )

    def publish(
        self,
        workspace: terminal.RunWorkspace,
        axes: terminal.TerminalAxes,
        maximum_archive_bytes: int,
        receipt_validator,
    ) -> dict[str, object]:
        self.publications.append(axes)
        if not self.publish_verified:
            raise terminal.ControllerError(
                "injected_publication_failure",
                "fake publication stopped before rename",
                publication_phase="pre_rename",
            )
        plan = self.plan
        if plan is None:
            raise AssertionError("publish called before create_workspace")
        self.assert_equal(maximum_archive_bytes, plan.safety.maximum_archive_bytes)
        manifest = {
            "directory_count_excluding_root": 1,
            "identity_claim": (
                "content_hashes_are_not_authenticity_or_authorship_proof"
            ),
            "local_strict_identity_fields": ["mode", "mtime_ns"],
            "manifest_bytes": 128,
            "manifest_path": str(workspace.archive_destination / "manifest.json"),
            "manifest_sha256": "b" * 64,
            "observed_archive_entries_including_manifest": 3,
            "observed_directories": 1,
            "observed_regular_files_including_manifest": 2,
            "portable_content_identity_fields": [
                "relative_path",
                "size",
                "sha256",
            ],
            "regular_file_count_excluding_manifest": 1,
            "total_regular_file_bytes_excluding_manifest": 256,
            "total_regular_file_bytes_including_manifest": 384,
        }
        global_safety_timeline = {
            "first_monotonic_ns": 0,
            "last_monotonic_ns": 3,
            "maximum_observed_gap_ns": 1,
            "sample_count": 4,
            "schema": "oai.memprof.r0.global-safety-timeline/v1",
        }
        marker_value = {
            "activation_condition": terminal.PUBLICATION_LINEARIZATION,
            "archive_path": str(workspace.archive_destination),
            "global_safety_timeline": global_safety_timeline,
            "manifest_sha256": manifest["manifest_sha256"],
            "plan_sha256": plan.sha256,
            "procedure_sha256": terminal.PROCEDURE_DESCRIPTOR_SHA256,
            "publication_phase": "verified",
            "run_id": workspace.run_id,
            "schema": terminal.PUBLICATION_SUCCESS_SCHEMA,
            "supported_signal_order_cutoff": terminal.PUBLICATION_SIGNAL_CUTOFF,
            "terminal_axes": axes.as_dict(),
        }
        marker_bytes = terminal.canonical_json_bytes(marker_value)

        def safety_sample(monotonic_ns: int) -> dict[str, object]:
            return {
                "free_bytes": max(
                    plan.safety.minimum_free_during_bytes,
                    1024 * 1024 * 1024,
                ),
                "monotonic_ns": monotonic_ns,
                "thermal_millicelsius": [
                    {"path": sensor.path, "value": 42000}
                    for sensor in plan.safety.mandatory_thermal_sensors
                ],
            }

        receipt = {
            "archive_integrity": "VERIFIED",
            "archive_path": str(workspace.archive_destination),
            "global_safety_timeline": global_safety_timeline,
            "inclusion": axes.inclusion,
            "manifest": manifest,
            "manifest_safety": {
                "post_write": safety_sample(2),
                "prospective_maximum_manifest_bytes": safety_sample(1),
            },
            "mutated": True,
            "postcompletion_operational_problem": None,
            "production_final_elf_absence": axes.production_final_elf_absence,
            "profiler_stream_state": axes.profiler_stream_state,
            "publication_phase": "verified",
            "publication_success_marker": {
                "bytes": len(marker_bytes),
                "path": str(workspace.scratch / "publication-success.json"),
                "sha256": hashlib.sha256(marker_bytes).hexdigest(),
                "value": marker_value,
            },
            "publication_success_marker_precommit_safety": safety_sample(3),
            "run_id": workspace.run_id,
            "schema": "oai.memprof.r0.publication-receipt/v1",
            "scientific_case_state": axes.scientific_case_state,
            "terminal_axes": axes.as_dict(),
            "verified": True,
        }
        if self.publication_receipt_mutator is not None:
            self.publication_receipt_mutator(receipt)
        self.publication_receipts.append(copy.deepcopy(receipt))
        return dict(receipt_validator(receipt))

    @staticmethod
    def assert_equal(observed: object, expected: object) -> None:
        if observed != expected:
            raise AssertionError(f"{observed!r} != {expected!r}")

    def recover_incomplete(self, case_path: Path) -> dict[str, object]:
        del case_path
        return {"archive_integrity": "VERIFIED"}

    def verify_archive(self, archive_path: Path) -> dict[str, object]:
        del archive_path
        return {"archive_integrity": "VERIFIED", "mutated": False}


class InjectedControllerTests(unittest.TestCase):
    def prepare_case(self, *, plan_mutator=None, **service_options):
        temporary = tempfile.TemporaryDirectory(prefix="oai-r0-controller-fake-")
        self.addCleanup(temporary.cleanup)
        services = FakeControllerServices(Path(temporary.name), **service_options)
        plan_value = _valid_plan_value()
        if plan_mutator is not None:
            plan_mutator(plan_value)
        plan = terminal.parse_plan(_canonical(plan_value))
        return services, plan

    @staticmethod
    def run_controller(services, plan):
        with mock.patch.object(
            terminal, "verify_procedure_descriptor", return_value=None
        ):
            return terminal.TerminalController(services).run(plan)

    def run_case(self, *, plan_mutator=None, **service_options):
        services, plan = self.prepare_case(
            plan_mutator=plan_mutator, **service_options
        )
        result = self.run_controller(services, plan)
        return services, plan, result

    def test_fake_services_execute_exact_off0_off1_on_population_and_domains(self) -> None:
        services, plan, result = self.run_case()
        axes = result["axes"]
        self.assertEqual(axes["scientific_case_state"], "COMPLETE")
        self.assertEqual(axes["archive_integrity"], "VERIFIED")
        self.assertEqual(axes["inclusion"], "INCLUDED")
        instances = [call["instance"] for call in services.calls]
        self.assertEqual(
            [item for item in instances if str(item).endswith("-configure")],
            ["off0-configure", "off1-configure", "on-configure"],
        )
        self.assertEqual(
            [item for item in instances if str(item).endswith("-absence")],
            ["off0-absence", "off1-absence"],
        )
        self.assertEqual(instances.count("on-positive-elf"), 1)
        on_build = next(call for call in services.calls if call["instance"] == "on-build")
        self.assertEqual(on_build["argv"], terminal.build_argv(plan, services.workspace))
        self.assertNotIn("nr-softmodem", on_build["argv"])
        for call in services.calls:
            environment = call["environment"]
            self.assertIsInstance(environment, dict)
            self.assertNotIn("OAI_MEMPROF_AMBIENT_MUST_NOT_LEAK", environment)
        for identity in services.captures.values():
            Path(identity.path).relative_to(services.workspace.archive_stage)
        self.assertTrue(services.read_workspaces)
        self.assertTrue(
            all(
                workspace is services.workspace
                for workspace in services.read_workspaces
            )
        )
        self.assertEqual(
            services.artifact_validation_workspaces,
            [services.workspace] * len(terminal.R0_ARTIFACTS),
        )
        self.assertTrue(services.progress_workspaces)
        self.assertTrue(
            all(
                workspace is services.workspace
                for workspace in services.progress_workspaces
            )
        )
        self.assertEqual(
            services.checkpoints[-1]["phase"],
            "READY_TO_SEAL",
        )
        self.assertEqual(
            services.checkpoints[-1]["terminal_candidate"]["axes"],
            result["axes"],
        )
        self.assertEqual(services.publications[0].as_dict(), result["axes"])

    def test_parser_timeline_failure_fails_case_closed(self) -> None:
        services, _, result = self.run_case(progress_failure_after=1)
        axes = result["axes"]
        self.assertEqual(axes["scientific_case_state"], "INCOMPLETE")
        self.assertEqual(axes["inclusion"], "EXCLUDED")
        self.assertIn("MAXIMUM_POLL_GAP", axes["reasons"])
        self.assertEqual(services.progress_workspaces, [services.workspace])

    def test_tool_and_cpm_use_sampled_file_digest_without_rehash(self) -> None:
        services, plan = self.prepare_case()
        workspace = services.create_workspace(plan, {})
        executable = Path(plan.tools["cc"]).resolve(strict=True)
        information = os.stat(executable, follow_symlinks=False)
        sampled_digest = "d" * 64
        tool_image = terminal.BoundedFileImage(
            data=executable.read_bytes(),
            device=information.st_dev,
            inode=information.st_ino,
            size=information.st_size,
            mode=information.st_mode,
            mtime_ns=information.st_mtime_ns,
            ctime_ns=information.st_ctime_ns,
            sha256=sampled_digest,
        )
        controller = terminal.TerminalController(services)
        with mock.patch.object(
            services,
            "read_regular",
            return_value=tool_image,
        ), mock.patch.object(
            terminal.hashlib,
            "sha256",
            side_effect=AssertionError("monolithic tool rehash"),
        ) as rehasher:
            binding = controller._observe_tool_binding(
                plan, workspace, "cc"
            )
            rehasher.assert_not_called()
        self.assertEqual(binding.sha256, sampled_digest)

        cpm_path = (
            workspace.conditions["OFF0"].dependency_root
            / "cpm-cache/cpm/CPM_0.40.1.cmake"
        )
        cpm_path.parent.mkdir(parents=True, exist_ok=True)
        cpm_path.write_bytes(b"synthetic-cpm")
        cpm_information = os.lstat(cpm_path)
        cpm_image = terminal.BoundedFileImage(
            data=cpm_path.read_bytes(),
            device=cpm_information.st_dev,
            inode=cpm_information.st_ino,
            size=cpm_information.st_size,
            mode=cpm_information.st_mode,
            mtime_ns=cpm_information.st_mtime_ns,
            ctime_ns=cpm_information.st_ctime_ns,
            sha256=plan.dependencies.cpm_cmake.sha256,
        )
        with mock.patch.object(
            services,
            "read_regular",
            return_value=cpm_image,
        ), mock.patch.object(
            terminal.hashlib,
            "sha256",
            side_effect=AssertionError("monolithic CPM rehash"),
        ) as rehasher:
            controller._validate_dependency_postcondition(
                plan,
                workspace,
                "OFF0",
                {"LC_ALL": "C"},
            )
            rehasher.assert_not_called()

    def test_command_failure_is_archived_as_incomplete_and_excluded(self) -> None:
        services, _, result = self.run_case(fail_instance="off1-configure")
        axes = result["axes"]
        self.assertEqual(axes["scientific_case_state"], "INCOMPLETE")
        self.assertEqual(axes["archive_integrity"], "VERIFIED")
        self.assertEqual(axes["inclusion"], "EXCLUDED")
        self.assertIn("COMMAND_FAILED", axes["reasons"])
        self.assertEqual(services.checkpoints[-2]["phase"], "FAILED")
        self.assertEqual(services.checkpoints[-1]["phase"], "READY_TO_SEAL")
        self.assertEqual(
            services.checkpoints[-1]["terminal_candidate"]["axes"]
            ["scientific_case_state"],
            "INCOMPLETE",
        )

    def test_keyboard_interruption_is_archived_but_never_upgraded(self) -> None:
        cancellation = KeyboardInterrupt("first cancellation")
        services, plan = self.prepare_case(
            interrupt_instance="off1-configure",
            interruption=cancellation,
        )
        with self.assertRaises(KeyboardInterrupt) as context:
            self.run_controller(services, plan)
        self.assertIs(context.exception, cancellation)
        axes = services.publications[-1]
        self.assertEqual(axes.scientific_case_state, "INCOMPLETE")
        self.assertEqual(axes.inclusion, "EXCLUDED")
        self.assertIn("INTERRUPTED_OR_RECOVERED", axes.reasons)
        self.assertEqual(services.checkpoints[-2]["phase"], "INTERRUPTED")
        self.assertEqual(services.checkpoints[-1]["phase"], "READY_TO_SEAL")

    def test_cancellation_at_validation_complete_is_sealed_then_rethrown_exactly(self) -> None:
        cancellation = SystemExit(143)
        services, plan = self.prepare_case(
            interrupt_checkpoint_phase="VALIDATION_COMPLETE",
            interruption=cancellation,
        )
        with self.assertRaises(SystemExit) as context:
            self.run_controller(services, plan)
        self.assertIs(context.exception, cancellation)
        phases = [checkpoint["phase"] for checkpoint in services.checkpoints]
        self.assertEqual(
            phases[-3:],
            ["VALIDATION_COMPLETE", "INTERRUPTED", "READY_TO_SEAL"],
        )
        axes = services.publications[-1]
        self.assertEqual(axes.scientific_case_state, "INCOMPLETE")
        self.assertEqual(axes.inclusion, "EXCLUDED")
        self.assertIn("INTERRUPTED_OR_RECOVERED", axes.reasons)

    def test_interruption_checkpoint_failure_preserves_first_cancellation(self) -> None:
        cancellation = KeyboardInterrupt("first cancellation")
        services, plan = self.prepare_case(
            interrupt_instance="off1-configure",
            checkpoint_failure_phase="INTERRUPTED",
            interruption=cancellation,
        )
        with self.assertRaises(KeyboardInterrupt) as context:
            self.run_controller(services, plan)
        self.assertIs(context.exception, cancellation)
        self.assertEqual(services.checkpoints[-2]["phase"], "INTERRUPTED")
        ready = services.checkpoints[-1]
        self.assertEqual(ready["phase"], "READY_TO_SEAL")
        self.assertEqual(
            ready["failure_checkpoint_problem"],
            {
                "code": "injected_checkpoint_failure",
                "detail": (
                    "injected_checkpoint_failure: INTERRUPTED"
                ),
                "requested_phase": "INTERRUPTED",
            },
        )
        axes = services.publications[-1]
        self.assertIn("INTERRUPTION_CHECKPOINT_FAILED", axes.reasons)
        notes = getattr(cancellation, "__notes__", ())
        self.assertTrue(
            any("interruption checkpoint failed" in note for note in notes),
            notes,
        )

    def test_unverified_publication_keeps_complete_science_out_of_inclusion(self) -> None:
        services, _, result = self.run_case(publish_verified=False)
        axes = result["axes"]
        self.assertEqual(axes["scientific_case_state"], "COMPLETE")
        self.assertEqual(axes["archive_integrity"], "UNVERIFIED")
        self.assertEqual(axes["inclusion"], "EXCLUDED")
        candidate = services.publications[0]
        self.assertEqual(candidate.scientific_case_state, "COMPLETE")
        self.assertEqual(candidate.archive_integrity, "VERIFIED")
        self.assertEqual(candidate.inclusion, "INCLUDED")

    def test_publication_safety_sequence_is_bound_to_global_timeline(self) -> None:
        def set_named_times(receipt, prospective, post_write, marker) -> None:
            receipt["manifest_safety"]["prospective_maximum_manifest_bytes"][
                "monotonic_ns"
            ] = prospective
            receipt["manifest_safety"]["post_write"][
                "monotonic_ns"
            ] = post_write
            receipt["publication_success_marker_precommit_safety"][
                "monotonic_ns"
            ] = marker

        def rebind_marker(receipt) -> None:
            marker_receipt = receipt["publication_success_marker"]
            marker_value = marker_receipt["value"]
            marker_value["global_safety_timeline"] = copy.deepcopy(
                receipt["global_safety_timeline"]
            )
            marker_bytes = terminal.canonical_json_bytes(marker_value)
            marker_receipt["bytes"] = len(marker_bytes)
            marker_receipt["sha256"] = hashlib.sha256(
                marker_bytes
            ).hexdigest()

        def all_named_samples_after_global_window(receipt) -> None:
            set_named_times(receipt, 100, 101, 102)

        def prospective_sample_before_global_first(receipt) -> None:
            timeline = receipt["global_safety_timeline"]
            timeline.update(
                first_monotonic_ns=10,
                last_monotonic_ns=13,
                maximum_observed_gap_ns=1,
                sample_count=4,
            )
            set_named_times(receipt, 9, 12, 13)
            rebind_marker(receipt)

        def marker_sample_after_global_last(receipt) -> None:
            set_named_times(receipt, 1, 2, 4)

        def marker_sample_before_global_last(receipt) -> None:
            timeline = receipt["global_safety_timeline"]
            timeline.update(
                first_monotonic_ns=0,
                last_monotonic_ns=4,
                maximum_observed_gap_ns=1,
                sample_count=5,
            )
            set_named_times(receipt, 1, 2, 3)
            rebind_marker(receipt)

        def insufficient_global_population(receipt) -> None:
            timeline = receipt["global_safety_timeline"]
            timeline.update(
                first_monotonic_ns=0,
                last_monotonic_ns=3,
                maximum_observed_gap_ns=2,
                sample_count=3,
            )
            set_named_times(receipt, 1, 2, 3)
            rebind_marker(receipt)

        def collapsed_named_population(sample_count):
            def mutate(receipt) -> None:
                timeline = receipt["global_safety_timeline"]
                timeline.update(
                    first_monotonic_ns=7,
                    last_monotonic_ns=7,
                    maximum_observed_gap_ns=(
                        None if sample_count == 1 else 0
                    ),
                    sample_count=sample_count,
                )
                set_named_times(receipt, 7, 7, 7)
                rebind_marker(receipt)

            return mutate

        mutants = (
            ("all-named-after-window", all_named_samples_after_global_window),
            ("prospective-before-first", prospective_sample_before_global_first),
            ("marker-after-last", marker_sample_after_global_last),
            ("marker-before-last", marker_sample_before_global_last),
            ("insufficient-population", insufficient_global_population),
            (
                "collapsed-named-population-count-1",
                collapsed_named_population(1),
            ),
            (
                "collapsed-named-population-count-2",
                collapsed_named_population(2),
            ),
            (
                "collapsed-named-population-count-3",
                collapsed_named_population(3),
            ),
        )
        for label, mutate in mutants:
            with self.subTest(label=label):
                services, _, result = self.run_case(
                    publication_receipt_mutator=mutate
                )
                self.assertIn(
                    "problem", result["publication"], result["publication"]
                )
                self.assertEqual(
                    result["publication"]["problem"]["code"],
                    "publication_safety_timeline",
                )
                self.assertIs(result["publication"]["verified"], False)
                self.assertEqual(
                    result["axes"]["archive_integrity"], "UNVERIFIED"
                )
                self.assertEqual(result["axes"]["inclusion"], "EXCLUDED")
                self.assertNotEqual(
                    services.publications[0].inclusion,
                    result["axes"]["inclusion"],
                )

        with self.subTest(label="collapsed-named-population-count-4-valid"):
            services, _, result = self.run_case(
                publication_receipt_mutator=collapsed_named_population(4)
            )
            self.assertNotIn(
                "problem", result["publication"], result["publication"]
            )
            self.assertIs(result["publication"]["verified"], True)
            self.assertEqual(result["axes"]["archive_integrity"], "VERIFIED")
            self.assertEqual(result["axes"]["inclusion"], "INCLUDED")
            self.assertEqual(services.publications[0].inclusion, "INCLUDED")

    def test_publication_marker_nested_timeline_rejects_bool_equal_integer(
        self,
    ) -> None:
        prepared_marker_bytes: list[bytes] = []

        def nested_bool_mutant(receipt) -> None:
            marker_receipt = receipt["publication_success_marker"]
            marker_value = marker_receipt["value"]
            prepared = terminal.canonical_json_bytes(marker_value)
            self.assertEqual(marker_receipt["bytes"], len(prepared))
            self.assertEqual(
                marker_receipt["sha256"],
                hashlib.sha256(prepared).hexdigest(),
            )
            prepared_marker_bytes.append(prepared)
            marker_value["global_safety_timeline"] = copy.deepcopy(
                marker_value["global_safety_timeline"]
            )
            marker_value["global_safety_timeline"][
                "first_monotonic_ns"
            ] = False
            outer_first = receipt["global_safety_timeline"][
                "first_monotonic_ns"
            ]
            self.assertIs(type(outer_first), int)
            self.assertEqual(
                outer_first,
                0,
            )

        services, _, result = self.run_case(
            publication_receipt_mutator=nested_bool_mutant
        )
        marker_value = copy.deepcopy(
            services.publication_receipts[0][
                "publication_success_marker"
            ]["value"]
        )
        marker_value["global_safety_timeline"]["first_monotonic_ns"] = 0
        self.assertEqual(len(prepared_marker_bytes), 1)
        self.assertEqual(
            terminal.canonical_json_bytes(marker_value),
            prepared_marker_bytes[0],
        )
        self.assertNotEqual(
            terminal.canonical_json_bytes(
                services.publication_receipts[0][
                    "publication_success_marker"
                ]["value"]
            ),
            prepared_marker_bytes[0],
        )
        self.assertIn(
            "problem", result["publication"], result["publication"]
        )
        self.assertEqual(
            result["publication"]["problem"]["code"],
            "publication_success_marker_binding",
        )
        self.assertIs(result["publication"]["verified"], False)
        self.assertEqual(result["axes"]["archive_integrity"], "UNVERIFIED")
        self.assertEqual(result["axes"]["inclusion"], "EXCLUDED")

    def test_artifact_aggregate_bound_fails_before_second_capture(self) -> None:
        archive_bytes = 1024 * 1024

        def lower_archive_bound(value: dict[str, object]) -> None:
            value["safety"]["maximum_archive_bytes"] = archive_bytes  # type: ignore[index]

        services, _, result = self.run_case(
            plan_mutator=lower_archive_bound,
            first_artifact_bytes=archive_bytes,
        )
        axes = result["axes"]
        self.assertEqual(axes["scientific_case_state"], "INCOMPLETE")
        self.assertEqual(axes["inclusion"], "EXCLUDED")
        self.assertIn("ARTIFACT_TOTAL_BYTES", axes["reasons"])
        artifact_captures = [
            path
            for path in services.captures
            if path.startswith("conditions/ON/artifacts/")
        ]
        self.assertEqual(
            artifact_captures,
            ["conditions/ON/artifacts/{}".format(terminal.R0_ARTIFACTS[0])],
        )

    def test_capture_destination_outside_archive_stage_fails_closed(self) -> None:
        _, _, result = self.run_case(capture_outside_archive=True)
        axes = result["axes"]
        self.assertEqual(axes["scientific_case_state"], "INCOMPLETE")
        self.assertEqual(axes["inclusion"], "EXCLUDED")
        self.assertTrue(
            any("CAPTURE_DOMAIN" in reason for reason in axes["reasons"]),
            axes["reasons"],
        )


class CoreServicesAdapterTests(unittest.TestCase):
    def prepare_adapter_inputs(
        self, *, maximum_poll_gap_milliseconds: int = 1000
    ):
        temporary = tempfile.TemporaryDirectory(prefix="oai-r0-core-services-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        acquisition = root / "acquisition"
        archive_root = root / "archive"
        source = root / "source"
        sensor = root / "temperature"
        for directory in (acquisition, archive_root, source):
            directory.mkdir()
        sensor.write_bytes(b"42000\n")
        value = _valid_plan_value()
        value["roots"] = {
            "acquisition_parent": str(acquisition),
            "archive_parent": str(archive_root),
        }
        value["source_repository"] = str(source)
        value["safety"]["mandatory_thermal_sensors"] = [  # type: ignore[index]
            {
                "label": "cpu",
                "minimum_plausible_millicelsius": 1000,
                "path": str(sensor),
            }
        ]
        value["safety"]["maximum_poll_gap_milliseconds"] = (  # type: ignore[index]
            maximum_poll_gap_milliseconds
        )
        plan = terminal.parse_plan(_canonical(value))
        services = terminal._CoreServices()
        handler_scope = terminal._CatchableTermination()
        handler_scope.__enter__()
        self.addCleanup(handler_scope.__exit__, None, None, None)
        descriptor = terminal.procedure_descriptor()
        descriptor_digest = hashlib.sha256(
            terminal.canonical_json_bytes(descriptor)
        ).hexdigest()
        descriptor_patch = mock.patch.object(
            terminal, "PROCEDURE_DESCRIPTOR_SHA256", descriptor_digest
        )
        descriptor_patch.start()
        self.addCleanup(descriptor_patch.stop)
        return services, plan, descriptor, root, acquisition, archive_root

    def create_adapter(self, *, maximum_poll_gap_milliseconds: int = 1000):
        services, plan, descriptor, _, _, _ = self.prepare_adapter_inputs(
            maximum_poll_gap_milliseconds=maximum_poll_gap_milliseconds
        )
        workspace = services.create_workspace(plan, descriptor)
        return services, plan, workspace

    def prepare_publishable_adapter(
        self, *, maximum_poll_gap_milliseconds: int = 1000
    ):
        services, plan, workspace = self.create_adapter(
            maximum_poll_gap_milliseconds=maximum_poll_gap_milliseconds
        )
        for phase in (
            "INITIALIZED",
            "SOURCE_BOUND",
            "OFF0_RUNNING",
            "OFF0_COMPLETE",
            "OFF1_RUNNING",
            "OFF1_COMPLETE",
            "ON_RUNNING",
            "ON_COMPLETE",
            "VALIDATION_COMPLETE",
            "READY_TO_SEAL",
        ):
            services.checkpoint(workspace, {"phase": phase})
        return services, plan, workspace

    def publish_complete_archive(
        self, *, maximum_poll_gap_milliseconds: int = 1000
    ):
        services, plan, workspace = self.prepare_publishable_adapter(
            maximum_poll_gap_milliseconds=maximum_poll_gap_milliseconds
        )
        axes = _complete_axes(verified=True)
        receipt = services.publish(
            workspace,
            axes,
            plan.safety.maximum_archive_bytes,
            lambda value: terminal.TerminalController._validate_publication_receipt(
                value, workspace, axes, plan
            ),
        )
        return services, plan, workspace, axes, receipt

    def _successful_process_result(
        self,
        services,
        spec,
        *,
        started_monotonic_ns: int,
        ended_monotonic_ns: int,
        safety_samples: tuple[object, ...],
    ):
        return services.process.ProcessResult(
            argv=tuple(spec.argv),
            cwd=str(spec.cwd),
            pid=4242,
            returncode=0,
            started_monotonic_ns=started_monotonic_ns,
            ended_monotonic_ns=ended_monotonic_ns,
            stdout_bytes=0,
            stderr_bytes=0,
            stdout_sha256=hashlib.sha256(b"").hexdigest(),
            stderr_sha256=hashlib.sha256(b"").hexdigest(),
            safety_samples=safety_samples,
            failure_code=None,
            failure_detail=None,
            kill_attempted=False,
            cleanup_complete=True,
            cleanup_failure_code=None,
            cleanup_failure_detail=None,
            unexpected_descendant_identity_lower_bound=0,
            unexpected_descendant_identity_complete=True,
            unexpected_descendant_pids=(),
            caller_signal_mask_numbers=services._current_signal_mask_numbers(),
            signal_mask_restored=True,
            containment_mode=services.process.CONTAINMENT_MODEL,
            containment_caller_thread_count=1,
            containment_preexisting_child_count=0,
            containment_subreaper_restored=True,
            descendant_scan_count=1,
            first_descendant_scan_monotonic_ns=ended_monotonic_ns,
            last_descendant_scan_monotonic_ns=ended_monotonic_ns,
        )

    def test_nonzero_and_cancellation_keep_raw_outputs_inside_archive_stage(self) -> None:
        services, plan, workspace = self.create_adapter()
        observation = services.run_step(
            workspace,
            plan,
            "controller_selftest",
            "adapter-nonzero",
            (sys.executable, str(CHILD), "write-exit", "both", "17", "7"),
            workspace.scratch,
            {"LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertFalse(services.command_succeeded(observation))
        record = services.command_record(observation)
        self.assertEqual(record["runtime_failure_code"], "process_exit_nonzero")
        stdout_relative = Path(record["stdout_path"])
        stderr_relative = Path(record["stderr_path"])
        self.assertFalse(stdout_relative.is_absolute())
        self.assertFalse(stderr_relative.is_absolute())
        self.assertNotIn("..", stdout_relative.parts)
        self.assertNotIn("..", stderr_relative.parts)
        stdout_path = workspace.archive_stage / stdout_relative
        stderr_path = workspace.archive_stage / stderr_relative
        self.assertEqual(
            Path(record["stdout_capture_path_during_acquisition"]), stdout_path
        )
        self.assertEqual(
            Path(record["stderr_capture_path_during_acquisition"]), stderr_path
        )
        Path(record["stdout_acquisition_path"]).relative_to(workspace.scratch)
        Path(record["stderr_acquisition_path"]).relative_to(workspace.scratch)
        self.assertEqual(stdout_path.read_bytes(), b"O" * 17)
        self.assertEqual(stderr_path.read_bytes(), b"E" * 17)

        pid_file = workspace.scratch / "adapter-cancel.pid"
        original_sample = services.process._sample_safety
        interrupted = False

        def interrupt_after_spawn(policy, monotonic_ns=None):
            nonlocal interrupted
            if pid_file.exists() and not interrupted:
                interrupted = True
                raise KeyboardInterrupt("injected adapter interruption")
            return original_sample(policy, monotonic_ns)

        with mock.patch.object(
            services.process, "_sample_safety", interrupt_after_spawn
        ):
            with self.assertRaisesRegex(
                KeyboardInterrupt, "adapter interruption"
            ) as interruption:
                services.run_step(
                    workspace,
                    plan,
                    "controller_selftest",
                    "adapter-cancel",
                    (
                        sys.executable,
                        str(CHILD),
                        "hold",
                        "5",
                        str(pid_file),
                        "--ignore-term",
                        "--stdout-bytes",
                        "23",
                    ),
                    workspace.scratch,
                    {"LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1"},
                )
        child_pid = int(pid_file.read_text(encoding="ascii"))
        self.assertTrue(_wait_pid_gone(child_pid))
        cancelled_stdout = (
            workspace.archive_stage / "commands/adapter-cancel.stdout.bin"
        )
        cancelled_stderr = (
            workspace.archive_stage / "commands/adapter-cancel.stderr.bin"
        )
        notes = getattr(interruption.exception, "__notes__", ())
        self.assertTrue(cancelled_stdout.exists(), notes)
        self.assertTrue(cancelled_stderr.exists(), notes)
        self.assertEqual(cancelled_stdout.read_bytes(), b"O" * 23)
        self.assertEqual(cancelled_stderr.read_bytes(), b"")

    def test_preexisting_final_or_active_path_causes_zero_workspace_mutation(
        self,
    ) -> None:
        real_datetime = terminal.datetime
        fixed = real_datetime(2026, 8, 11, 12, 34, 56, tzinfo=terminal.timezone.utc)
        token = "0123456789abcdef"
        run_id = "r0-x86-0001-20260811T123456Z-" + token
        for role, expected_code in (
            ("final", "archive_destination_exists"),
            ("active", "archive_stage_exists"),
        ):
            for path_type in ("symlink", "directory", "fifo"):
                with self.subTest(role=role, path_type=path_type):
                    (
                        services,
                        plan,
                        descriptor,
                        root,
                        acquisition,
                        archive_root,
                    ) = self.prepare_adapter_inputs()
                    path = (
                        archive_root / run_id
                        if role == "final"
                        else archive_root / f".{run_id}.active"
                    )
                    if path_type == "symlink":
                        target = root / f"{role}-symlink-target"
                        target.mkdir()
                        path.symlink_to(target, target_is_directory=True)
                    elif path_type == "directory":
                        path.mkdir()
                    else:
                        os.mkfifo(path)
                    before = os.lstat(path)
                    acquisition_population = tuple(acquisition.iterdir())
                    archive_population = tuple(archive_root.iterdir())
                    with mock.patch.object(
                        terminal,
                        "datetime",
                        wraps=real_datetime,
                    ) as datetime_type, mock.patch.object(
                        terminal.secrets,
                        "token_hex",
                        return_value=token,
                    ), mock.patch.object(
                        services.archive,
                        "create_directory_exclusive",
                        wraps=services.archive.create_directory_exclusive,
                    ) as creator:
                        datetime_type.now.return_value = fixed
                        with self.assertRaises(terminal.ControllerError) as raised:
                            services.create_workspace(plan, descriptor)
                    self.assertEqual(raised.exception.code, expected_code)
                    creator.assert_not_called()
                    after = os.lstat(path)
                    self.assertEqual(
                        (after.st_dev, after.st_ino, after.st_mode),
                        (before.st_dev, before.st_ino, before.st_mode),
                    )
                    self.assertEqual(
                        tuple(acquisition.iterdir()), acquisition_population
                    )
                    self.assertEqual(
                        tuple(archive_root.iterdir()), archive_population
                    )

    def test_parent_binding_drift_is_detected_before_first_create(self) -> None:
        for expected_code in (
            "workspace_parent_identity_changed",
            "active_mount_identity_changed",
        ):
            with self.subTest(placement=expected_code):
                (
                    services,
                    plan,
                    descriptor,
                    _,
                    acquisition,
                    archive_root,
                ) = self.prepare_adapter_inputs()
                before = (
                    tuple(acquisition.iterdir()),
                    tuple(archive_root.iterdir()),
                )
                with mock.patch.object(
                    services,
                    "_recheck_workspace_parent_bindings",
                    side_effect=terminal.ControllerError(
                        expected_code, "injected pre-create drift"
                    ),
                ) as recheck, mock.patch.object(
                    services.archive,
                    "create_directory_exclusive",
                    wraps=services.archive.create_directory_exclusive,
                ) as creator:
                    with self.assertRaises(terminal.ControllerError) as raised:
                        services.create_workspace(plan, descriptor)
                self.assertEqual(raised.exception.code, expected_code)
                recheck.assert_called_once()
                creator.assert_not_called()
                self.assertEqual(
                    (
                        tuple(acquisition.iterdir()),
                        tuple(archive_root.iterdir()),
                    ),
                    before,
                )

        services, plan, _, _, acquisition, archive_root = (
            self.prepare_adapter_inputs()
        )
        acquisition_expected = os.lstat(acquisition)
        archive_expected = os.lstat(archive_root)
        records = terminal._read_mountinfo_snapshot()
        expected_mount = terminal._select_mount_identity(
            acquisition, acquisition_expected, records
        )
        real_lstat = terminal.os.lstat

        def inode_drift(path):
            information = real_lstat(path)
            if Path(path) != acquisition:
                return information
            values = list(information)
            values[1] += 1
            return os.stat_result(values)

        with mock.patch.object(terminal.os, "lstat", side_effect=inode_drift):
            with self.assertRaises(terminal.ControllerError) as raised:
                services._recheck_workspace_parent_bindings(
                    acquisition,
                    archive_root,
                    acquisition_expected,
                    archive_expected,
                    expected_mount,
                    mock.sentinel.timeline,
                )
        self.assertEqual(
            raised.exception.code, "workspace_parent_identity_changed"
        )

        drifted_mount = dataclasses.replace(
            expected_mount, mount_id=expected_mount.mount_id + 1
        )
        with mock.patch.object(
            services,
            "_mount_snapshot",
            return_value=records,
        ), mock.patch.object(
            terminal,
            "_select_mount_identity",
            side_effect=(expected_mount, drifted_mount),
        ):
            with self.assertRaises(terminal.ControllerError) as raised:
                services._recheck_workspace_parent_bindings(
                    acquisition,
                    archive_root,
                    acquisition_expected,
                    archive_expected,
                    expected_mount,
                    mock.sentinel.timeline,
                )
        self.assertEqual(raised.exception.code, "active_mount_identity_changed")

    def test_created_scratch_or_stage_rejects_distinct_mount_id_same_device(
        self,
    ) -> None:
        real_datetime = terminal.datetime
        fixed = real_datetime(2026, 8, 11, 12, 34, 56, tzinfo=terminal.timezone.utc)
        token = "fedcba9876543210"
        run_id = "r0-x86-0001-20260811T123456Z-" + token
        for fault_role in ("scratch", "archive-stage"):
            with self.subTest(fault_role=fault_role):
                (
                    services,
                    plan,
                    descriptor,
                    _,
                    acquisition,
                    archive_root,
                ) = self.prepare_adapter_inputs()
                scratch = acquisition / run_id
                stage = archive_root / f".{run_id}.active"
                target = scratch if fault_role == "scratch" else stage
                real_select = terminal._select_mount_identity
                observed_binding: list[tuple[int, int, int, int]] = []

                def select_with_distinct_mount(path, information, records):
                    selected = real_select(path, information, records)
                    if Path(path) != target:
                        return selected
                    observed_binding.append(
                        (
                            information.st_dev,
                            os.makedev(
                                selected.device_major,
                                selected.device_minor,
                            ),
                            selected.mount_id,
                            selected.mount_id + 1000,
                        )
                    )
                    return dataclasses.replace(
                        selected, mount_id=selected.mount_id + 1000
                    )

                with mock.patch.object(
                    terminal,
                    "datetime",
                    wraps=real_datetime,
                ) as datetime_type, mock.patch.object(
                    terminal.secrets,
                    "token_hex",
                    return_value=token,
                ), mock.patch.object(
                    terminal,
                    "_select_mount_identity",
                    side_effect=select_with_distinct_mount,
                ), mock.patch.object(
                    services.archive,
                    "create_directory_exclusive",
                    wraps=services.archive.create_directory_exclusive,
                ) as creator, mock.patch.object(
                    services.archive, "write_bytes_exclusive"
                ) as byte_writer, mock.patch.object(
                    services.archive, "write_json_exclusive"
                ) as json_writer, mock.patch.object(
                    services.archive, "initialize_journal"
                ) as journal_writer:
                    datetime_type.now.return_value = fixed
                    with self.assertRaises(terminal.ControllerError) as raised:
                        services.create_workspace(plan, descriptor)
                self.assertEqual(
                    raised.exception.code, "workspace_child_mount_mismatch"
                )
                self.assertEqual(len(observed_binding), 1)
                stat_device, mount_device, original_id, changed_id = (
                    observed_binding[0]
                )
                self.assertEqual(stat_device, mount_device)
                self.assertNotEqual(original_id, changed_id)
                expected_calls = 1 if fault_role == "scratch" else 2
                self.assertEqual(creator.call_count, expected_calls)
                self.assertTrue(scratch.is_dir())
                self.assertEqual(tuple(scratch.iterdir()), ())
                if fault_role == "scratch":
                    self.assertFalse(stage.exists())
                else:
                    self.assertTrue(stage.is_dir())
                    self.assertEqual(tuple(stage.iterdir()), ())
                self.assertFalse((archive_root / run_id).exists())
                byte_writer.assert_not_called()
                json_writer.assert_not_called()
                journal_writer.assert_not_called()

    def test_postrename_final_rejects_distinct_mount_id_same_device(self) -> None:
        services, _, workspace = self.create_adapter()
        context = services._context(workspace)
        os.rename(workspace.archive_stage, workspace.archive_destination)
        final_information = os.lstat(workspace.archive_destination)
        self.assertEqual(
            (
                final_information.st_dev,
                final_information.st_ino,
                final_information.st_mode,
            ),
            context.archive_stage_identity[1:],
        )
        records = terminal._read_mountinfo_snapshot()
        real_select = terminal._select_mount_identity
        observed_binding: list[tuple[int, int, int, int]] = []

        def select_final_on_distinct_mount(path, information, snapshot):
            selected = real_select(path, information, snapshot)
            if Path(path) != workspace.archive_destination:
                return selected
            observed_binding.append(
                (
                    information.st_dev,
                    os.makedev(
                        selected.device_major,
                        selected.device_minor,
                    ),
                    selected.mount_id,
                    selected.mount_id + 1000,
                )
            )
            return dataclasses.replace(
                selected, mount_id=selected.mount_id + 1000
            )

        with mock.patch.object(
            services,
            "_mount_snapshot",
            return_value=records,
        ), mock.patch.object(
            terminal,
            "_select_mount_identity",
            side_effect=select_final_on_distinct_mount,
        ):
            with self.assertRaises(terminal.ControllerError) as raised:
                services._validate_mount_identity(context)
        self.assertEqual(raised.exception.code, "active_mount_identity_changed")
        self.assertEqual(len(observed_binding), 1)
        stat_device, mount_device, original_id, changed_id = observed_binding[0]
        self.assertEqual(stat_device, mount_device)
        self.assertNotEqual(original_id, changed_id)
        self.assertFalse(workspace.archive_stage.exists())
        self.assertTrue(workspace.archive_destination.is_dir())

    def test_recovery_population_reserves_exact_file_and_payload_boundaries(
        self,
    ) -> None:
        services, plan, workspace = self.create_adapter()
        context = services._context(workspace)
        maximum_files = (
            plan.execution.maximum_archive_regular_files_excluding_manifest
        )
        payload_limit = (
            plan.safety.maximum_archive_bytes
            - plan.execution.maximum_manifest_bytes
        )
        recovery_value = {
            "original_run_id": workspace.run_id,
            "recovery_reason": "INTERRUPTED_BEFORE_MANIFEST",
            "schema": "oai.memprof.r0.recovery/v1",
            "terminal_axes": services._recovered_axes(
                verified=True,
                reason="RECOVERED_AFTER_INTERRUPTION",
            ).as_dict(),
        }
        recovery_bytes = len(terminal.canonical_json_bytes(recovery_value))
        cases = (
            (
                "one-file-and-exact-byte-reserve-remains",
                maximum_files - 1,
                payload_limit - recovery_bytes,
                False,
                None,
            ),
            (
                "file-capacity-without-marker",
                maximum_files,
                payload_limit - recovery_bytes,
                False,
                "recovery_regular_file_reserve",
            ),
            (
                "one-byte-over-payload-reserve",
                maximum_files - 1,
                payload_limit - recovery_bytes + 1,
                False,
                "recovery_payload_reserve",
            ),
            (
                "idempotent-marker-needs-no-second-reserve",
                maximum_files,
                payload_limit,
                True,
                None,
            ),
        )
        for (
            label,
            regular_files,
            regular_bytes,
            marker_present,
            expected_code,
        ) in cases:
            with self.subTest(label=label):
                population = (
                    regular_files,
                    0,
                    regular_files,
                    regular_bytes,
                )
                with mock.patch.object(
                    services, "_mount_snapshot", return_value=()
                ), mock.patch.object(
                    services, "_archive_call", return_value=population
                ) as scanner:
                    if expected_code is None:
                        observed = (
                            services._scan_recovery_population_and_reserve(
                                plan,
                                workspace.archive_stage,
                                context.safety_timeline,
                                recovery_bytes,
                                marker_present,
                                context.active_mount_identity,
                            )
                        )
                        self.assertEqual(observed, population)
                    else:
                        with self.assertRaises(
                            terminal.ControllerError
                        ) as raised:
                            services._scan_recovery_population_and_reserve(
                                plan,
                                workspace.archive_stage,
                                context.safety_timeline,
                                recovery_bytes,
                                marker_present,
                                context.active_mount_identity,
                            )
                        self.assertEqual(raised.exception.code, expected_code)
                scanner.assert_called_once()

    def test_recovery_capacity_rejection_precedes_marker_mutation(self) -> None:
        services, _, workspace = self.create_adapter()
        services.checkpoint(workspace, {"phase": "INITIALIZED"})
        rejection = terminal.ControllerError(
            "recovery_regular_file_reserve",
            "injected exact-capacity rejection",
        )
        recovered = (
            workspace.archive_stage.parent
            / f"{workspace.run_id}.recovered-incomplete"
        )
        with mock.patch.object(
            services,
            "_scan_recovery_population_and_reserve",
            side_effect=rejection,
        ) as reserve, mock.patch.object(
            services.archive, "write_json_exclusive"
        ) as writer:
            with self.assertRaises(terminal.ControllerError) as raised:
                services.recover_incomplete(workspace.archive_stage)
        self.assertIs(raised.exception, rejection)
        reserve.assert_called_once()
        writer.assert_not_called()
        self.assertFalse((workspace.archive_stage / "recovery.json").exists())
        self.assertFalse((workspace.archive_stage / "manifest.json").exists())
        self.assertFalse(workspace.archive_destination.exists())
        self.assertFalse(recovered.exists())

    def test_recovery_uses_one_global_timeline_through_postverification(
        self,
    ) -> None:
        services, _, workspace = self.create_adapter()
        services.checkpoint(workspace, {"phase": "INITIALIZED"})
        real_new_timeline = services._new_live_safety_timeline
        real_archive_call = services._archive_call
        timelines: list[object] = []
        operations: list[tuple[str, object]] = []

        def capture_timeline(plan, storage_path):
            timeline = real_new_timeline(plan, storage_path)
            timelines.append(timeline)
            return timeline

        def capture_operation(
            timeline,
            operation,
            function,
            *args,
            **kwargs,
        ):
            operations.append((operation, timeline))
            return real_archive_call(
                timeline, operation, function, *args, **kwargs
            )

        with mock.patch.object(
            services,
            "_new_live_safety_timeline",
            side_effect=capture_timeline,
        ), mock.patch.object(
            services,
            "_archive_call",
            side_effect=capture_operation,
        ):
            result = services.recover_incomplete(workspace.archive_stage)
        self.assertEqual(len(timelines), 1)
        timeline = timelines[0]
        for operation in (
            "write-recovery-marker",
            "create-recovery-manifest",
            "publish-recovered-incomplete",
            "read-shared-verification-plan",
        ):
            matching = [
                observed_timeline
                for observed_operation, observed_timeline in operations
                if observed_operation == operation
            ]
            self.assertEqual(matching, [timeline], operation)
        self.assertEqual(
            result["recovery_global_safety_timeline"],
            dict(services._timeline_receipt(timeline)),
        )
        first = result["recovery_global_safety_timeline"][
            "first_monotonic_ns"
        ]
        last = result["recovery_global_safety_timeline"][
            "last_monotonic_ns"
        ]
        self.assertIsInstance(first, int)
        self.assertIsInstance(last, int)
        for sample in result["recovery_safety"].values():
            if sample is not None:
                self.assertLessEqual(first, sample["monotonic_ns"])
                self.assertLessEqual(sample["monotonic_ns"], last)

    def test_recovery_phase_gap_fails_the_shared_global_timeline(self) -> None:
        for target in (
            "write-recovery-marker",
            "create-recovery-manifest",
            "publish-recovered-incomplete",
            "read-shared-verification-plan",
        ):
            with self.subTest(target=target):
                services, plan, workspace = self.create_adapter(
                    maximum_poll_gap_milliseconds=20
                )
                services.checkpoint(workspace, {"phase": "INITIALIZED"})
                real_archive_call = services._archive_call
                real_sample_safety = services.process.sample_safety
                jump_timeline: list[object] = []
                allowed_ns = (
                    plan.safety.maximum_poll_gap_milliseconds * 1_000_000
                )

                def select_gap(
                    timeline,
                    operation,
                    function,
                    *args,
                    **kwargs,
                ):
                    if operation == target:
                        jump_timeline.append(timeline)
                    return real_archive_call(
                        timeline, operation, function, *args, **kwargs
                    )

                def sample_with_selected_gap(*args, **kwargs):
                    sample = real_sample_safety(*args, **kwargs)
                    if not jump_timeline:
                        return sample
                    timeline = jump_timeline.pop()
                    self.assertIsNotNone(timeline)
                    self.assertIsInstance(timeline.last_monotonic_ns, int)
                    return dataclasses.replace(
                        sample,
                        monotonic_ns=(
                            timeline.last_monotonic_ns + allowed_ns + 1
                        ),
                    )

                with mock.patch.object(
                    services,
                    "_archive_call",
                    side_effect=select_gap,
                ), mock.patch.object(
                    services.process,
                    "sample_safety",
                    side_effect=sample_with_selected_gap,
                ):
                    with self.assertRaises(
                        terminal.ControllerError
                    ) as raised:
                        services.recover_incomplete(workspace.archive_stage)
                self.assertEqual(
                    raised.exception.code, "maximum_poll_gap", target
                )
                self.assertEqual(jump_timeline, [])

    def test_standalone_verification_binds_original_exact_mount_identity(
        self,
    ) -> None:
        services, plan, workspace = self.prepare_publishable_adapter()
        axes = _complete_axes(verified=True)
        receipt = services.publish(
            workspace,
            axes,
            plan.safety.maximum_archive_bytes,
            lambda value: terminal.TerminalController._validate_publication_receipt(
                value, workspace, axes, plan
            ),
        )
        marker = workspace.scratch / "publication-success.json"
        marker_bytes = marker.read_bytes()
        self.assertEqual(receipt["scientific_case_state"], "COMPLETE")
        baseline = services.verify_archive(workspace.archive_destination)
        self.assertTrue(baseline["verified"])
        self.assertEqual(baseline["archive_integrity"], "VERIFIED")
        self.assertEqual(baseline["scientific_case_state"], "COMPLETE")
        self.assertEqual(baseline["inclusion"], "INCLUDED")
        self.assertIs(
            baseline["location_matches_original_publication"], True
        )
        self.assertIs(
            baseline["publication_success_marker_verified"], True
        )

        real_select = terminal._select_mount_identity
        drift_paths = {
            workspace.archive_destination.parent,
            workspace.archive_destination,
        }
        observed: list[tuple[int, int, int, int]] = []

        def select_on_different_mount(path, information, records):
            selected = real_select(path, information, records)
            if Path(path) not in drift_paths:
                return selected
            observed.append(
                (
                    information.st_dev,
                    os.makedev(
                        selected.device_major,
                        selected.device_minor,
                    ),
                    selected.mount_id,
                    selected.mount_id + 1000,
                )
            )
            return dataclasses.replace(
                selected, mount_id=selected.mount_id + 1000
            )

        with mock.patch.object(
            terminal,
            "_select_mount_identity",
            side_effect=select_on_different_mount,
        ):
            moved_mount = services.verify_archive(
                workspace.archive_destination
            )
        self.assertTrue(moved_mount["verified"])
        self.assertEqual(moved_mount["archive_integrity"], "VERIFIED")
        self.assertIs(
            moved_mount["location_matches_original_publication"], False
        )
        self.assertIs(
            moved_mount["publication_success_marker_verified"], False
        )
        self.assertEqual(moved_mount["scientific_case_state"], "INCOMPLETE")
        self.assertEqual(moved_mount["inclusion"], "EXCLUDED")
        self.assertIn(
            "ORIGINAL_PUBLICATION_LOCATION_UNVERIFIED",
            moved_mount["reasons"],
        )
        self.assertEqual(len(observed), 2)
        for stat_device, mount_device, original_id, changed_id in observed:
            self.assertEqual(stat_device, mount_device)
            self.assertNotEqual(original_id, changed_id)
        self.assertEqual(marker.read_bytes(), marker_bytes)

    def test_marker_restoration_failure_preserves_durable_completion(
        self,
    ) -> None:
        services, plan, workspace = self.prepare_publishable_adapter()
        axes = _complete_axes(verified=True)
        marker = workspace.scratch / "publication-success.json"
        caller_mask = services._current_signal_mask_numbers()
        real_archive_call = services._archive_call
        real_scope_finish = terminal._ArchiveSignalScope.finish
        real_fsync = services.archive.os.fsync
        real_pthread_sigmask = terminal.signal.pthread_sigmask
        committed = False
        restore_failures = 0
        marker_fsync_count = 0
        archive_operations: list[str] = []
        postcommit_counts: tuple[int, int, int] | None = None

        def observe_archive_call(
            timeline,
            operation,
            function,
            *args,
            **kwargs,
        ):
            archive_operations.append(operation)
            return real_archive_call(
                timeline, operation, function, *args, **kwargs
            )

        def count_marker_fsync(descriptor):
            nonlocal marker_fsync_count
            if marker.exists():
                marker_fsync_count += 1
            return real_fsync(descriptor)

        def snapshot_before_restoration(scope, primary):
            nonlocal committed, postcommit_counts
            if marker.exists() and postcommit_counts is None:
                committed = True
                postcommit_counts = (
                    len(archive_operations),
                    safety_sampler.call_count,
                    hasher.call_count,
                )
            return real_scope_finish(scope, primary)

        def fail_first_postcommit_restore(how, mask):
            nonlocal restore_failures
            if (
                committed
                and how == signal.SIG_SETMASK
                and restore_failures == 0
            ):
                restore_failures += 1
                raise OSError(
                    errno.EIO,
                    "injected postcompletion signal-mask restoration failure",
                )
            return real_pthread_sigmask(how, mask)

        try:
            with mock.patch.object(
                services,
                "_archive_call",
                side_effect=observe_archive_call,
            ), mock.patch.object(
                services,
                "_force_timeline_sample",
                wraps=services._force_timeline_sample,
            ) as safety_sampler, mock.patch.object(
                terminal.hashlib,
                "sha256",
                wraps=terminal.hashlib.sha256,
            ) as hasher, mock.patch.object(
                services.archive.os,
                "fsync",
                side_effect=count_marker_fsync,
            ), mock.patch.object(
                terminal._ArchiveSignalScope,
                "finish",
                new=snapshot_before_restoration,
            ), mock.patch.object(
                terminal.signal,
                "pthread_sigmask",
                side_effect=fail_first_postcommit_restore,
            ):
                receipt = services.publish(
                    workspace,
                    axes,
                    plan.safety.maximum_archive_bytes,
                    lambda value: terminal.TerminalController._validate_publication_receipt(
                        value, workspace, axes, plan
                    ),
                )
                self.assertIsNotNone(postcommit_counts)
                self.assertEqual(
                    (
                        len(archive_operations),
                        safety_sampler.call_count,
                        hasher.call_count,
                    ),
                    postcommit_counts,
                )
        finally:
            real_pthread_sigmask(signal.SIG_SETMASK, set(caller_mask))

        self.assertTrue(committed)
        self.assertEqual(restore_failures, 1)
        self.assertEqual(marker_fsync_count, 2)
        self.assertEqual(
            services._current_signal_mask_numbers(), caller_mask
        )
        problem = receipt["postcompletion_operational_problem"]
        self.assertEqual(
            set(problem),
            {"code", "message", "phase", "scientific_effect"},
        )
        self.assertEqual(problem["code"], "OSError")
        self.assertIn("injected postcompletion", problem["message"])
        self.assertEqual(
            problem["phase"],
            "postcompletion_signal_mask_restoration",
        )
        self.assertEqual(
            problem["scientific_effect"],
            "none_durable_success_marker_already_committed",
        )
        self.assertTrue(
            terminal._validate_postcompletion_operational_problem(receipt)
        )
        self.assertTrue(receipt["verified"])
        self.assertEqual(receipt["archive_integrity"], "VERIFIED")
        self.assertEqual(receipt["scientific_case_state"], "COMPLETE")
        self.assertEqual(receipt["inclusion"], "INCLUDED")
        self.assertEqual(
            receipt["terminal_axes"]["scientific_case_state"], "COMPLETE"
        )
        self.assertEqual(
            marker.read_bytes(),
            terminal.canonical_json_bytes(
                receipt["publication_success_marker"]["value"]
            ),
        )

        recovery = terminal._CoreServices().recover_incomplete(
            workspace.scratch
        )
        self.assertEqual(
            recovery["recovery_action"], "VERIFY_EXISTING_ORIGINAL"
        )
        for field in (
            "scientific_case_state",
            "archive_integrity",
            "inclusion",
            "profiler_stream_state",
            "production_final_elf_absence",
        ):
            self.assertEqual(recovery[field], receipt[field], field)
        self.assertIs(
            recovery["location_matches_original_publication"], True
        )
        self.assertIs(
            recovery["publication_success_marker_verified"], True
        )

    def test_publication_receipt_validation_obeys_precommit_deadline(
        self,
    ) -> None:
        for label, deadline_offset_ns, expected_success in (
            ("one-nanosecond-over", 1, False),
            ("exact-deadline", 0, True),
        ):
            with self.subTest(label=label):
                services, plan, workspace = self.prepare_publishable_adapter(
                    maximum_poll_gap_milliseconds=20
                )
                axes = _complete_axes(verified=True)
                allowed_ns = (
                    plan.safety.maximum_poll_gap_milliseconds * 1_000_000
                )
                forced_now: int | None = None
                late_writer: mock.Mock | None = None
                late_writer_patch = mock.patch.object(
                    services.archive,
                    "write_bytes_exclusive",
                )
                real_monotonic_ns = terminal.time.monotonic_ns

                def controlled_monotonic_ns() -> int:
                    if forced_now is None:
                        return real_monotonic_ns()
                    return forced_now

                def validate_then_advance(receipt):
                    nonlocal forced_now, late_writer
                    validated = (
                        terminal.TerminalController._validate_publication_receipt(
                            receipt, workspace, axes, plan
                        )
                    )
                    marker_sample_time = receipt[
                        "publication_success_marker_precommit_safety"
                    ]["monotonic_ns"]
                    self.assertIsInstance(marker_sample_time, int)
                    if expected_success:
                        forced_now = (
                            marker_sample_time
                            + allowed_ns
                            + deadline_offset_ns
                        )
                    else:
                        late_writer = late_writer_patch.start()
                        burn_deadline = max(
                            marker_sample_time + allowed_ns + 1,
                            real_monotonic_ns() + allowed_ns + 1,
                        )
                        while real_monotonic_ns() < burn_deadline:
                            time.sleep(0.001)
                    return validated

                marker = workspace.scratch / "publication-success.json"
                if expected_success:
                    with mock.patch.object(
                        terminal.time,
                        "monotonic_ns",
                        side_effect=controlled_monotonic_ns,
                    ):
                        receipt = services.publish(
                            workspace,
                            axes,
                            plan.safety.maximum_archive_bytes,
                            validate_then_advance,
                        )
                    self.assertIsNotNone(forced_now)
                    self.assertTrue(receipt["verified"])
                    self.assertEqual(receipt["inclusion"], "INCLUDED")
                    self.assertEqual(
                        marker.read_bytes(),
                        terminal.canonical_json_bytes(
                            receipt["publication_success_marker"]["value"]
                        ),
                    )
                else:
                    try:
                        with self.assertRaises(
                            terminal.ControllerError
                        ) as raised:
                            services.publish(
                                workspace,
                                axes,
                                plan.safety.maximum_archive_bytes,
                                validate_then_advance,
                            )
                    finally:
                        late_writer_patch.stop()
                    self.assertEqual(
                        raised.exception.code, "maximum_poll_gap"
                    )
                    if late_writer is None:
                        self.fail(
                            "receipt validator did not install marker writer guard"
                        )
                    late_writer.assert_not_called()
                    self.assertFalse(marker.exists())

    def test_standalone_verification_final_sample_obeys_deadline(
        self,
    ) -> None:
        _, plan, workspace, _, _ = self.publish_complete_archive(
            maximum_poll_gap_milliseconds=20
        )
        allowed_ns = plan.safety.maximum_poll_gap_milliseconds * 1_000_000

        for label, deadline_offset_ns, expected_success in (
            ("one-nanosecond-over", 1, False),
            ("exact-deadline", 0, True),
        ):
            with self.subTest(label=label):
                verifier = terminal._CoreServices()
                timelines: list[object] = []
                forced_now: int | None = None
                real_new_timeline = verifier._new_live_safety_timeline
                real_canonical_object = verifier._canonical_object
                real_monotonic_ns = terminal.time.monotonic_ns

                def controlled_monotonic_ns() -> int:
                    if forced_now is None:
                        return real_monotonic_ns()
                    return forced_now

                def capture_timeline(*args, **kwargs):
                    timeline = real_new_timeline(*args, **kwargs)
                    timelines.append(timeline)
                    return timeline

                def canonical_with_advance(data, role):
                    nonlocal forced_now
                    if role == "publication_success_marker":
                        self.assertEqual(len(timelines), 1)
                        last = timelines[0].last_monotonic_ns
                        self.assertIsInstance(last, int)
                        forced_now = last + allowed_ns + deadline_offset_ns
                    return real_canonical_object(data, role)

                with mock.patch.object(
                    terminal.time,
                    "monotonic_ns",
                    side_effect=controlled_monotonic_ns,
                ), mock.patch.object(
                    verifier.process,
                    "_monotonic_ns",
                    side_effect=controlled_monotonic_ns,
                ), mock.patch.object(
                    verifier,
                    "_new_live_safety_timeline",
                    side_effect=capture_timeline,
                ), mock.patch.object(
                    verifier,
                    "_canonical_object",
                    side_effect=canonical_with_advance,
                ):
                    result = verifier.verify_archive(
                        workspace.archive_destination
                    )
                self.assertEqual(len(timelines), 1)
                self.assertIsNotNone(forced_now)
                timeline_receipt = dict(
                    verifier._timeline_receipt(timelines[0])
                )
                self.assertIn(
                    "verification_global_safety_timeline",
                    result,
                    result,
                )
                self.assertEqual(
                    result["verification_global_safety_timeline"],
                    timeline_receipt,
                )
                self.assertEqual(
                    timeline_receipt["last_monotonic_ns"], forced_now
                )
                if expected_success:
                    self.assertTrue(result["verified"])
                    self.assertEqual(result["archive_integrity"], "VERIFIED")
                    self.assertEqual(result["scientific_case_state"], "COMPLETE")
                    self.assertEqual(result["inclusion"], "INCLUDED")
                    self.assertIs(
                        result["publication_success_marker_verified"], True
                    )
                else:
                    self.assertIs(result["verified"], False)
                    self.assertEqual(
                        result["problem"]["code"], "maximum_poll_gap"
                    )
                    self.assertEqual(
                        result["archive_integrity"], "UNVERIFIED"
                    )
                    self.assertEqual(
                        result["scientific_case_state"], "INCOMPLETE"
                    )
                    self.assertEqual(result["inclusion"], "EXCLUDED")
                    self.assertGreater(
                        timeline_receipt["maximum_observed_gap_ns"],
                        allowed_ns,
                    )

    def test_controller_verify_accepts_real_core_manifest_path_contract(
        self,
    ) -> None:
        _, plan, workspace, _, _ = self.publish_complete_archive()
        verifier = terminal._CoreServices()
        real_verify_manifest = verifier.archive.verify_manifest
        core_manifest_paths: list[str] = []

        def capture_summary(*args, **kwargs):
            summary = real_verify_manifest(*args, **kwargs)
            core_manifest_paths.append(summary.manifest_path)
            return summary

        try:
            with mock.patch.object(
                verifier.archive,
                "verify_manifest",
                side_effect=capture_summary,
            ):
                result = terminal.TerminalController(verifier).verify(
                    workspace.archive_destination
                )
        except terminal.ControllerError as error:
            self.fail(
                "real core verification receipt was rejected: "
                f"{error.code}: {error.detail}"
            )
        self.assertEqual(core_manifest_paths, ["manifest.json"])
        self.assertTrue(result["verified"])
        self.assertEqual(result["archive_integrity"], "VERIFIED")
        self.assertEqual(result["scientific_case_state"], "COMPLETE")
        self.assertEqual(result["inclusion"], "INCLUDED")
        self.assertEqual(
            result["manifest"]["manifest_path"],
            str(workspace.archive_destination / "manifest.json"),
        )
        terminal._validate_global_safety_timeline_receipt(
            result["verification_global_safety_timeline"],
            "real standalone verification",
            plan,
        )

    def test_controller_resume_accepts_real_core_manifest_path_contract(
        self,
    ) -> None:
        _, plan, workspace, _, _ = self.publish_complete_archive()
        recovery_services = terminal._CoreServices()
        real_verify_manifest = recovery_services.archive.verify_manifest
        core_manifest_paths: list[str] = []

        def capture_summary(*args, **kwargs):
            summary = real_verify_manifest(*args, **kwargs)
            core_manifest_paths.append(summary.manifest_path)
            return summary

        try:
            with mock.patch.object(
                recovery_services.archive,
                "verify_manifest",
                side_effect=capture_summary,
            ):
                result = terminal.TerminalController(
                    recovery_services
                ).resume(workspace.scratch)
        except terminal.ControllerError as error:
            self.fail(
                "real core recovery receipt was rejected: "
                f"{error.code}: {error.detail}"
            )
        self.assertEqual(core_manifest_paths, ["manifest.json"])
        self.assertEqual(result["recovery_action"], "VERIFY_EXISTING_ORIGINAL")
        self.assertTrue(result["verified"])
        self.assertEqual(result["archive_integrity"], "VERIFIED")
        self.assertEqual(result["scientific_case_state"], "COMPLETE")
        self.assertEqual(result["inclusion"], "INCLUDED")
        self.assertEqual(
            result["manifest"]["manifest_path"],
            str(workspace.archive_destination / "manifest.json"),
        )
        terminal._validate_global_safety_timeline_receipt(
            result["verification_global_safety_timeline"],
            "real resumed verification",
            plan,
        )

    def test_poll_gap_above_declared_maximum_fails_global_timeline(self) -> None:
        services, plan, workspace = self.create_adapter(
            maximum_poll_gap_milliseconds=20
        )
        allowed_ns = plan.safety.maximum_poll_gap_milliseconds * 1_000_000
        context = services._context(workspace)

        def delayed_sample_result(spec, *, limits, safety):
            del limits, safety
            spec.stdout_path.write_bytes(b"")
            spec.stderr_path.write_bytes(b"")
            started = context.safety_timeline.last_monotonic_ns
            self.assertIsInstance(started, int)
            deadline = started + allowed_ns + 1
            while time.monotonic_ns() < deadline:
                time.sleep(0.001)
            ended = time.monotonic_ns()
            sample = services.process.SafetySample(
                monotonic_ns=ended,
                thermal_millicelsius=(
                    (plan.safety.mandatory_thermal_sensors[0].path, 42000),
                ),
                free_bytes=1 << 62,
            )
            return self._successful_process_result(
                services,
                spec,
                started_monotonic_ns=started,
                ended_monotonic_ns=ended,
                safety_samples=(sample,),
            )

        with mock.patch.object(
            services.process,
            "run_bounded_command",
            delayed_sample_result,
        ):
            with self.assertRaises(terminal.ControllerError) as raised:
                services.run_step(
                    workspace,
                    plan,
                    "controller_selftest",
                    "adapter-poll-gap",
                    (sys.executable, "-c", "pass"),
                    workspace.scratch,
                    {"LC_ALL": "C"},
                )
        self.assertEqual(raised.exception.code, "maximum_poll_gap")

    def test_dynamic_free_space_decline_threshold_is_passed_and_rechecked(self) -> None:
        services, plan, workspace = self.create_adapter()
        context = services._context(workspace)
        expected_threshold = max(
            plan.safety.minimum_free_during_bytes,
            context.baseline_free_bytes
            - plan.safety.maximum_observed_free_space_decline_bytes,
        )

        def below_threshold_result(spec, *, limits, safety):
            del limits
            self.assertEqual(safety.storage_path, workspace.scratch)
            self.assertEqual(safety.minimum_free_bytes, expected_threshold)
            spec.stdout_path.write_bytes(b"")
            spec.stderr_path.write_bytes(b"")
            started = services._context(
                workspace
            ).safety_timeline.last_monotonic_ns
            self.assertIsInstance(started, int)
            ended = max(started, time.monotonic_ns())
            sample = services.process.SafetySample(
                monotonic_ns=ended,
                thermal_millicelsius=(
                    (plan.safety.mandatory_thermal_sensors[0].path, 42000),
                ),
                free_bytes=max(0, expected_threshold - 1),
            )
            return self._successful_process_result(
                services,
                spec,
                started_monotonic_ns=started,
                ended_monotonic_ns=ended,
                safety_samples=(sample,),
            )

        with mock.patch.object(
            services.process,
            "run_bounded_command",
            below_threshold_result,
        ):
            with self.assertRaises(terminal.ControllerError) as raised:
                services.run_step(
                    workspace,
                    plan,
                    "controller_selftest",
                    "adapter-free-space",
                    (sys.executable, "-c", "pass"),
                    workspace.scratch,
                    {"LC_ALL": "C"},
                )
        self.assertEqual(raised.exception.code, "free_space_limit")

    def test_process_result_exact_contract_rejects_mutant_matrix(self) -> None:
        services, plan, workspace = self.create_adapter()

        class DetailSubclass(str):
            pass

        class TupleSubclass(tuple):
            pass

        @dataclasses.dataclass(frozen=True)
        class SafetySampleLookalike:
            monotonic_ns: int
            thermal_millicelsius: tuple[tuple[str, int], ...]
            free_bytes: int

        dynamic_argv = object()
        dynamic_safety_lookalike = object()
        failure_enum_like = mock.Mock(spec=services.process.ProcessErrorCode)
        failure_enum_like.value = "wall_time_limit"
        cleanup_enum_like = mock.Mock(spec=services.process.ProcessErrorCode)
        cleanup_enum_like.value = "cleanup_timeout"
        self.assertIsInstance(
            failure_enum_like, services.process.ProcessErrorCode
        )
        self.assertIsNot(
            type(failure_enum_like), services.process.ProcessErrorCode
        )
        cases = (
            (
                "orphan-failure-detail",
                {"failure_detail": "orphan"},
                "process_result_contract",
            ),
            (
                "orphan-cleanup-detail",
                {"cleanup_failure_detail": "orphan"},
                "process_result_contract",
            ),
            (
                "enum-like-failure-code",
                {
                    "failure_code": failure_enum_like,
                    "failure_detail": "failure",
                },
                "process_result_contract",
            ),
            (
                "enum-like-cleanup-code",
                {
                    "failure_code": services.process.ProcessErrorCode.WALL_TIME_LIMIT,
                    "failure_detail": "failure",
                    "cleanup_failure_code": cleanup_enum_like,
                    "cleanup_failure_detail": "cleanup",
                    "cleanup_complete": False,
                },
                "process_result_contract",
            ),
            (
                "str-subclass-failure-detail",
                {
                    "failure_code": services.process.ProcessErrorCode.WALL_TIME_LIMIT,
                    "failure_detail": DetailSubclass("failure"),
                },
                "process_result_contract",
            ),
            (
                "str-subclass-cleanup-detail",
                {
                    "failure_code": services.process.ProcessErrorCode.WALL_TIME_LIMIT,
                    "failure_detail": "failure",
                    "cleanup_failure_code": (
                        services.process.ProcessErrorCode.CLEANUP_TIMEOUT
                    ),
                    "cleanup_failure_detail": DetailSubclass("cleanup"),
                    "cleanup_complete": False,
                },
                "process_result_contract",
            ),
            (
                "success-without-pid-or-scan",
                {
                    "pid": None,
                    "returncode": 0,
                    "descendant_scan_count": 0,
                    "first_descendant_scan_monotonic_ns": None,
                    "last_descendant_scan_monotonic_ns": None,
                    "maximum_observed_descendant_scan_gap_ns": None,
                },
                "process_result_contract",
            ),
            (
                "live-descendant-without-cumulative-identity",
                {
                    "maximum_observed_live_descendant_count": 1,
                    "observed_descendant_identity_lower_bound": 0,
                },
                "process_result_contract",
            ),
            (
                "argv-list",
                {"argv": dynamic_argv},
                "process_result_contract",
            ),
            (
                "cwd-str-subclass",
                {"cwd": DetailSubclass(str(workspace.scratch))},
                "process_result_contract",
            ),
            (
                "sha-str-subclass",
                {
                    "stdout_sha256": DetailSubclass(
                        hashlib.sha256(b"").hexdigest()
                    )
                },
                "process_result_contract",
            ),
            (
                "safety-sample-lookalike",
                {"safety_samples": dynamic_safety_lookalike},
                "process_result_contract",
            ),
            (
                "observed-identity-list",
                {"observed_descendant_identities": []},
                "process_result_contract",
            ),
            (
                "unexpected-identity-tuple-subclass",
                {"unexpected_descendant_identities": TupleSubclass()},
                "process_result_contract",
            ),
            (
                "bool-stdout-bytes",
                {"stdout_bytes": False},
                "process_result_contract",
            ),
            (
                "bool-containment-thread-count",
                {"containment_caller_thread_count": True},
                "process_result_contract",
            ),
            (
                "bool-live-descendant-count",
                {"maximum_observed_live_descendant_count": False},
                "process_result_contract",
            ),
            (
                "bool-observed-identity-count",
                {"observed_descendant_identity_lower_bound": False},
                "process_result_contract",
            ),
            (
                "bool-unexpected-identity-count",
                {"unexpected_descendant_identity_lower_bound": False},
                "process_result_contract",
            ),
            (
                "bool-descendant-scan-count",
                {"descendant_scan_count": False},
                "process_result_contract",
            ),
            (
                "integer-signal-mask-restored",
                {"signal_mask_restored": 1},
                "process_result_contract",
            ),
            (
                "integer-subreaper-restored",
                {"containment_subreaper_restored": 1},
                "process_result_contract",
            ),
            (
                "integer-cleanup-complete",
                {"cleanup_complete": 1},
                "process_result_contract",
            ),
        )

        for index, (label, mutations, expected_code) in enumerate(cases):
            with self.subTest(label=label):
                def mutant_result(spec, *, limits, safety):
                    del limits, safety
                    spec.stdout_path.write_bytes(b"")
                    spec.stderr_path.write_bytes(b"")
                    started = services._context(
                        workspace
                    ).safety_timeline.last_monotonic_ns
                    self.assertIsInstance(started, int)
                    ended = max(started, time.monotonic_ns())
                    sample = services.process.SafetySample(
                        monotonic_ns=ended,
                        thermal_millicelsius=(
                            (
                                plan.safety.mandatory_thermal_sensors[0].path,
                                42000,
                            ),
                        ),
                        free_bytes=1 << 62,
                    )
                    valid = self._successful_process_result(
                        services,
                        spec,
                        started_monotonic_ns=ended,
                        ended_monotonic_ns=ended,
                        safety_samples=(sample,),
                    )
                    valid = dataclasses.replace(
                        valid,
                        pid=4242,
                        descendant_scan_count=1,
                        first_descendant_scan_monotonic_ns=ended,
                        last_descendant_scan_monotonic_ns=ended,
                    )
                    effective_mutations = dict(mutations)
                    if effective_mutations.get("argv") is dynamic_argv:
                        effective_mutations["argv"] = list(spec.argv)
                    if (
                        effective_mutations.get("safety_samples")
                        is dynamic_safety_lookalike
                    ):
                        effective_mutations["safety_samples"] = (
                            SafetySampleLookalike(
                                monotonic_ns=sample.monotonic_ns,
                                thermal_millicelsius=(
                                    sample.thermal_millicelsius
                                ),
                                free_bytes=sample.free_bytes,
                            ),
                        )
                    return dataclasses.replace(valid, **effective_mutations)

                with mock.patch.object(
                    services.process,
                    "run_bounded_command",
                    mutant_result,
                ):
                    observation = services.run_step(
                        workspace,
                        plan,
                        "controller_selftest",
                        f"adapter-process-mutant-{index}",
                        (sys.executable, "-c", "pass"),
                        workspace.scratch,
                        {"LC_ALL": "C"},
                    )
                record = services.command_record(observation)
                self.assertFalse(services.command_succeeded(observation))
                self.assertEqual(record["local_failure_code"], expected_code)

    def test_process_result_descendant_scan_gap_matches_population_span(
        self,
    ) -> None:
        services, plan, workspace = self.create_adapter()
        cases = (
            ("count-one-exact-no-gap", 1, 0, None, True),
            ("count-one-nonzero-span", 1, 1, None, False),
            ("count-one-zero-gap", 1, 0, 0, False),
            ("count-two-zero-span-zero-gap", 2, 0, 0, True),
            ("count-two-zero-span-positive-gap", 2, 0, 1, False),
            ("count-three-below-ceiling", 3, 5, 2, False),
            ("count-three-at-ceiling", 3, 5, 3, True),
            ("count-three-at-span", 3, 5, 5, True),
            ("count-three-above-span", 3, 5, 6, False),
        )

        for index, (label, scan_count, scan_span, scan_gap, accepted) in enumerate(
            cases
        ):
            with self.subTest(label=label):
                if scan_count == 1:
                    oracle_accepts = scan_span == 0 and scan_gap is None
                else:
                    minimum_possible_maximum_gap = (
                        scan_span + scan_count - 2
                    ) // (scan_count - 1)
                    oracle_accepts = bool(
                        type(scan_gap) is int
                        and minimum_possible_maximum_gap
                        <= scan_gap
                        <= scan_span
                    )
                self.assertIs(accepted, oracle_accepts)

                def process_result(spec, *, limits, safety):
                    del limits, safety
                    spec.stdout_path.write_bytes(b"")
                    spec.stderr_path.write_bytes(b"")
                    started = services._context(
                        workspace
                    ).safety_timeline.last_monotonic_ns
                    self.assertIsInstance(started, int)
                    ended = max(started + scan_span, time.monotonic_ns())

                    def sample(monotonic_ns):
                        return services.process.SafetySample(
                            monotonic_ns=monotonic_ns,
                            thermal_millicelsius=(
                                (
                                    plan.safety.mandatory_thermal_sensors[0].path,
                                    42000,
                                ),
                            ),
                            free_bytes=1 << 62,
                        )

                    valid = self._successful_process_result(
                        services,
                        spec,
                        started_monotonic_ns=started,
                        ended_monotonic_ns=ended,
                        safety_samples=(sample(started), sample(ended)),
                    )
                    return dataclasses.replace(
                        valid,
                        descendant_scan_count=scan_count,
                        first_descendant_scan_monotonic_ns=ended - scan_span,
                        last_descendant_scan_monotonic_ns=ended,
                        maximum_observed_descendant_scan_gap_ns=scan_gap,
                    )

                with mock.patch.object(
                    services.process,
                    "run_bounded_command",
                    process_result,
                ):
                    observation = services.run_step(
                        workspace,
                        plan,
                        "controller_selftest",
                        f"adapter-process-scan-gap-{index}",
                        (sys.executable, "-c", "pass"),
                        workspace.scratch,
                        {"LC_ALL": "C"},
                    )
                record = services.command_record(observation)
                if accepted:
                    self.assertIsNone(record["local_failure_code"])
                    self.assertTrue(services.command_succeeded(observation))
                else:
                    self.assertEqual(
                        record["local_failure_code"],
                        "process_result_contract",
                    )
                    self.assertFalse(services.command_succeeded(observation))

    def test_terminal_adapter_requires_exact_active_signal_handlers(self) -> None:
        services, plan, workspace = self.create_adapter()
        handlers = services._validate_terminal_signal_handlers()
        self.assertIs(handlers[int(signal.SIGINT)], signal.default_int_handler)
        self.assertIs(
            handlers[int(signal.SIGTERM)],
            terminal._CatchableTermination._raise_system_exit,
        )
        self.assertIs(
            handlers[int(signal.SIGHUP)],
            terminal._CatchableTermination._raise_system_exit,
        )

        wrong_callable = lambda _signum, _frame: None
        for signum, replacement in (
            (signal.SIGTERM, signal.SIG_DFL),
            (signal.SIGHUP, signal.SIG_IGN),
            (signal.SIGINT, wrong_callable),
        ):
            with self.subTest(signum=signal.Signals(signum).name):
                previous = signal.getsignal(signum)
                signal.signal(signum, replacement)
                try:
                    with self.assertRaises(terminal.ControllerError) as raised:
                        services._validate_terminal_signal_handlers()
                finally:
                    signal.signal(signum, previous)
                self.assertEqual(
                    raised.exception.code, "signal_handler_precondition"
                )

        previous = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        try:
            with mock.patch.object(
                services.process, "run_bounded_command"
            ) as runner:
                with self.assertRaises(terminal.ControllerError) as raised:
                    services.run_step(
                        workspace,
                        plan,
                        "controller_selftest",
                        "adapter-handler-precondition",
                        (sys.executable, "-c", "pass"),
                        workspace.scratch,
                        {"LC_ALL": "C"},
                    )
                runner.assert_not_called()
        finally:
            signal.signal(signal.SIGTERM, previous)
        self.assertEqual(raised.exception.code, "signal_handler_precondition")

        source = workspace.scratch / "handler-precondition-source"
        source.write_bytes(b"source")
        destination = workspace.archive_stage / "commands/handler-copy.bin"
        previous = signal.getsignal(signal.SIGHUP)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        try:
            with mock.patch.object(
                services.archive, "copy_file_exclusive"
            ) as copier:
                with self.assertRaises(terminal.ControllerError) as raised:
                    services.capture_regular(
                        workspace,
                        source,
                        "commands/handler-copy.bin",
                        64,
                    )
                copier.assert_not_called()
        finally:
            signal.signal(signal.SIGHUP, previous)
        self.assertEqual(raised.exception.code, "signal_handler_precondition")
        self.assertFalse(destination.exists())


class PublicationSignalCutoffTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="oai-r0-marker-cutoff-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.services = terminal._CoreServices()
        self.payload = b'{"schema":"test-publication-success"}'

    @staticmethod
    def _supported_pending() -> set[int]:
        supported = {int(signal.SIGHUP), int(signal.SIGINT), int(signal.SIGTERM)}
        return {int(item) for item in signal.sigpending()}.intersection(supported)

    def _assert_mask_and_pending_restored(
        self, caller_mask: tuple[int, ...]
    ) -> None:
        self.assertEqual(
            self.services._current_signal_mask_numbers(), caller_mask
        )
        self.assertEqual(self._supported_pending(), set())

    def test_pending_before_marker_cutoff_prevents_all_marker_mutation(self) -> None:
        marker = self.root / "pre-cutoff.json"
        caller_mask = self.services._current_signal_mask_numbers()
        real_enter = terminal._ArchiveSignalScope.enter

        def enter_and_queue(scope) -> None:
            real_enter(scope)
            os.kill(os.getpid(), signal.SIGTERM)
            self.assertIn(int(signal.SIGTERM), self._supported_pending())

        with terminal._CatchableTermination():
            with mock.patch.object(
                terminal._ArchiveSignalScope,
                "enter",
                new=enter_and_queue,
            ), mock.patch.object(
                self.services.archive, "write_bytes_exclusive"
            ) as writer:
                with self.assertRaises(SystemExit) as raised:
                    self.services._commit_publication_success_marker(
                        marker,
                        self.payload,
                        maximum_bytes=len(self.payload),
                    )
                writer.assert_not_called()
        self.assertEqual(raised.exception.args, (128 + int(signal.SIGTERM),))
        notes = tuple(getattr(raised.exception, "__notes__", ()))
        self.assertTrue(
            any("boundary=publication-success-marker:before-commit" in note for note in notes),
            notes,
        )
        self.assertFalse(
            any("durably committed" in note for note in notes), notes
        )
        self.assertFalse(marker.exists())
        self._assert_mask_and_pending_restored(caller_mask)

    def test_signal_after_cutoff_during_fsync_is_post_completion(self) -> None:
        marker = self.root / "post-cutoff-success.json"
        caller_mask = self.services._current_signal_mask_numbers()
        real_fsync = self.services.archive.os.fsync
        fsync_calls = 0

        def queue_during_first_fsync(descriptor: int) -> None:
            nonlocal fsync_calls
            fsync_calls += 1
            if fsync_calls == 1:
                os.kill(os.getpid(), signal.SIGTERM)
                self.assertIn(int(signal.SIGTERM), self._supported_pending())
            real_fsync(descriptor)

        with terminal._CatchableTermination():
            with mock.patch.object(
                self.services.archive.os,
                "fsync",
                side_effect=queue_during_first_fsync,
            ):
                with self.assertRaises(SystemExit) as raised:
                    self.services._commit_publication_success_marker(
                        marker,
                        self.payload,
                        maximum_bytes=len(self.payload),
                    )
        self.assertEqual(fsync_calls, 2)
        self.assertEqual(raised.exception.args, (128 + int(signal.SIGTERM),))
        notes = tuple(getattr(raised.exception, "__notes__", ()))
        self.assertTrue(
            any("boundary=archive-primitive-final-boundary" in note for note in notes),
            notes,
        )
        self.assertTrue(
            any(
                "publication-success marker was durably committed before "
                "post-completion signal restoration" in note
                for note in notes
            ),
            notes,
        )
        self.assertEqual(marker.read_bytes(), self.payload)
        self._assert_mask_and_pending_restored(caller_mask)

    def test_post_cutoff_write_or_fsync_failure_has_no_completion(self) -> None:
        for fault in ("write", "fsync"):
            with self.subTest(fault=fault):
                marker = self.root / f"post-cutoff-{fault}-failure.json"
                caller_mask = self.services._current_signal_mask_numbers()
                expected_code = (
                    self.services.archive.ArchiveErrorCode.WRITE_FAILED
                    if fault == "write"
                    else self.services.archive.ArchiveErrorCode.FSYNC_FAILED
                )
                real_operation = getattr(self.services.archive.os, fault)
                calls = 0

                def queue_then_fail(*args):
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        os.kill(os.getpid(), signal.SIGTERM)
                        self.assertIn(
                            int(signal.SIGTERM), self._supported_pending()
                        )
                        raise OSError(errno.EIO, f"injected marker {fault} failure")
                    return real_operation(*args)

                with terminal._CatchableTermination():
                    with mock.patch.object(
                        self.services.archive.os,
                        fault,
                        side_effect=queue_then_fail,
                    ):
                        with self.assertRaises(
                            self.services.archive.ArchiveError
                        ) as raised:
                            self.services._commit_publication_success_marker(
                                marker,
                                self.payload,
                                maximum_bytes=len(self.payload),
                            )
                self.assertEqual(raised.exception.code, expected_code)
                notes = tuple(getattr(raised.exception, "__notes__", ()))
                self.assertTrue(
                    any(
                        "boundary=archive-primitive-final-boundary" in note
                        and "selected=existing-primary" in note
                        for note in notes
                    ),
                    notes,
                )
                self.assertFalse(
                    any("durably committed" in note for note in notes), notes
                )
                self.assertFalse(marker.exists())
                self._assert_mask_and_pending_restored(caller_mask)


def _complete_axes(*, verified: bool) -> terminal.TerminalAxes:
    return terminal.classify_terminal_result(
        condition_complete={name: True for name, _, _ in terminal.R0_CONDITIONS},
        archive_verified=verified,
        interrupted=False,
        absence_decisions=(
            terminal.AbsenceDecision(
                "OFF0", True, "QUALIFIED", "REFERENCE_TOOL_MATCH"
            ),
            terminal.AbsenceDecision(
                "OFF1", True, "QUALIFIED", "REFERENCE_TOOL_MATCH"
            ),
        ),
        positive_elf_complete=True,
    )


def _manifest_receipt(archive_path: Path) -> dict[str, object]:
    return {
        "directory_count_excluding_root": 1,
        "identity_claim": (
            "content_hashes_are_not_authenticity_or_authorship_proof"
        ),
        "local_strict_identity_fields": ["mode", "mtime_ns"],
        "manifest_bytes": 128,
        "manifest_path": str(archive_path / "manifest.json"),
        "manifest_sha256": "b" * 64,
        "observed_archive_entries_including_manifest": 3,
        "observed_directories": 1,
        "observed_regular_files_including_manifest": 2,
        "portable_content_identity_fields": [
            "relative_path",
            "size",
            "sha256",
        ],
        "regular_file_count_excluding_manifest": 1,
        "total_regular_file_bytes_excluding_manifest": 256,
        "total_regular_file_bytes_including_manifest": 384,
    }


def _verification_global_timeline() -> dict[str, object]:
    return {
        "first_monotonic_ns": 10,
        "last_monotonic_ns": 30,
        "maximum_observed_gap_ns": 10,
        "sample_count": 3,
        "schema": "oai.memprof.r0.global-safety-timeline/v1",
    }


def _verification_receipt(
    archive_path: Path,
    axes: terminal.TerminalAxes,
    *,
    location_matches_original: bool,
    marker_verified: bool,
) -> dict[str, object]:
    receipt = {
        **axes.as_dict(),
        "archive_path": str(archive_path),
        "mutated": False,
        "schema": "oai.memprof.r0.archive-verification/v1",
        "verification_global_safety_timeline": (
            _verification_global_timeline()
            if axes.archive_integrity == "VERIFIED"
            else None
        ),
        "verified": axes.archive_integrity == "VERIFIED",
    }
    if axes.archive_integrity != "VERIFIED":
        receipt["problem"] = {
            "code": "injected_verification_failure",
            "message": "synthetic unverified receipt",
        }
        return receipt
    recovered_suffix = ".recovered-incomplete"
    run_id = (
        archive_path.name[: -len(recovered_suffix)]
        if archive_path.name.endswith(recovered_suffix)
        else archive_path.name
    )
    receipt.update(
        {
            "location_matches_original_publication": location_matches_original,
            "manifest": _manifest_receipt(archive_path),
            "plan_sha256": "c" * 64,
            "publication_success_marker_path": str(
                Path("/tmp/oai-r0-markers")
                / run_id
                / "publication-success.json"
            ),
            "publication_success_marker_verified": marker_verified,
            "procedure_sha256": terminal.PROCEDURE_DESCRIPTOR_SHA256,
            "run_id": run_id,
        }
    )
    return receipt


def _recovery_receipt(
    archive_path: Path,
    axes: terminal.TerminalAxes,
    *,
    action: str,
    location_matches_original: bool,
    marker_verified: bool,
) -> dict[str, object]:
    receipt = _verification_receipt(
        archive_path,
        axes,
        location_matches_original=location_matches_original,
        marker_verified=marker_verified,
    )
    receipt["schema"] = "oai.memprof.r0.recovery-receipt/v1"
    receipt["recovery_action"] = action
    return receipt


def _sealed_recovery_receipt(archive_path: Path) -> dict[str, object]:
    axes = terminal._CoreServices._recovered_axes(
        verified=True,
        reason="RECOVERED_AFTER_INTERRUPTION",
    )
    receipt = _recovery_receipt(
        archive_path,
        axes,
        action="SEALED_AND_PUBLISHED_INCOMPLETE",
        location_matches_original=False,
        marker_verified=False,
    )

    def sample(monotonic_ns: int) -> dict[str, object]:
        return {
            "free_bytes": 1024 * 1024 * 1024,
            "monotonic_ns": monotonic_ns,
            "thermal_millicelsius": [
                {"path": "/tmp/oai-r0-temperature", "value": 42000}
            ],
        }

    receipt.update(
        {
            "mutated": True,
            "publication_phase": "verified",
            "recovery_global_safety_timeline": copy.deepcopy(
                receipt["verification_global_safety_timeline"]
            ),
            "recovery_safety": {
                "post_manifest": sample(30),
                "post_marker_pre_manifest": sample(20),
                "prospective_marker_and_manifest": sample(10),
            },
            "recovery_safety_plan_binding": {
                "maximum_poll_gap_milliseconds": 20,
                "maximum_safety_samples": 100,
                "plan_sha256": receipt["plan_sha256"],
                "schema": "oai.memprof.r0.recovery-safety-plan-binding/v1",
            },
        }
    )
    return receipt


class RecoveryServices:
    def __init__(self, recovery: dict[str, object], verification: dict[str, object]):
        self.recovery = recovery
        self.verification = verification

    def recover_incomplete(self, case_path: Path) -> dict[str, object]:
        del case_path
        return copy.deepcopy(self.recovery)

    def verify_archive(self, archive_path: Path) -> dict[str, object]:
        del archive_path
        return copy.deepcopy(self.verification)


class RecoveryAndCliTests(unittest.TestCase):
    def test_sealed_recovery_global_timeline_is_strict_and_related(self) -> None:
        archive_path = Path("/tmp/r0-x86-0001.recovered-incomplete")
        valid = _sealed_recovery_receipt(archive_path)
        services = RecoveryServices(recovery=valid, verification={})
        observed = terminal.TerminalController(services).resume(
            "/tmp/active-case"
        )
        self.assertEqual(
            observed["recovery_global_safety_timeline"]["sample_count"], 3
        )
        self.assertEqual(
            observed["recovery_global_safety_timeline"],
            observed["verification_global_safety_timeline"],
        )

        def delete_global(value):
            del value["recovery_global_safety_timeline"]

        def add_unknown(value):
            value["recovery_global_safety_timeline"]["unknown"] = 0

        def zero_count(value):
            value["recovery_global_safety_timeline"]["sample_count"] = 0

        def bool_count(value):
            value["recovery_global_safety_timeline"]["sample_count"] = True

        def bool_first(value):
            value["recovery_global_safety_timeline"][
                "first_monotonic_ns"
            ] = False

        def bool_gap(value):
            value["recovery_global_safety_timeline"][
                "maximum_observed_gap_ns"
            ] = False

        def unrelated_window(value):
            timeline = value["recovery_global_safety_timeline"]
            timeline["first_monotonic_ns"] = 40
            timeline["last_monotonic_ns"] = 60

        def too_few_global_samples(value):
            timeline = value["recovery_global_safety_timeline"]
            timeline["sample_count"] = 2
            timeline["maximum_observed_gap_ns"] = 20

        def impossible_zero_gap(value):
            value["recovery_global_safety_timeline"][
                "maximum_observed_gap_ns"
            ] = 0

        def zero_bound(value):
            value["recovery_safety_plan_binding"][
                "maximum_poll_gap_milliseconds"
            ] = 0

        def bool_bound(value):
            value["recovery_safety_plan_binding"][
                "maximum_safety_samples"
            ] = True

        mutants = (
            ("missing-global", delete_global, "recovery_receipt_shape"),
            ("unknown-global", add_unknown, "global_safety_timeline_shape"),
            ("zero-count", zero_count, "global_safety_timeline_shape"),
            ("bool-count", bool_count, "global_safety_timeline_shape"),
            ("bool-first", bool_first, "global_safety_timeline_shape"),
            ("bool-gap", bool_gap, "global_safety_timeline_gap"),
            (
                "unrelated-window",
                unrelated_window,
                "recovery_global_safety_binding",
            ),
            (
                "too-few-global-samples",
                too_few_global_samples,
                "recovery_global_safety_binding",
            ),
            (
                "impossible-zero-gap",
                impossible_zero_gap,
                "global_safety_timeline_gap",
            ),
            (
                "zero-plan-gap-bound",
                zero_bound,
                "recovery_safety_plan_binding",
            ),
            (
                "bool-plan-sample-bound",
                bool_bound,
                "recovery_safety_plan_binding",
            ),
        )
        for label, mutate, expected_code in mutants:
            with self.subTest(label=label):
                services.recovery = copy.deepcopy(valid)
                mutate(services.recovery)
                with self.assertRaises(terminal.ControllerError) as raised:
                    terminal.TerminalController(services).resume(
                        "/tmp/active-case"
                    )
                self.assertEqual(raised.exception.code, expected_code)

    def test_recovery_receipt_preserves_incomplete_unavailable_axes(self) -> None:
        archive_path = Path("/tmp/r0-x86-0001.recovered-incomplete")
        axes = terminal._CoreServices._recovered_axes(
            verified=True,
            reason="RECOVERED_AFTER_INTERRUPTION",
        )
        recovery = _recovery_receipt(
            archive_path,
            axes,
            action="VERIFY_EXISTING_RECOVERED",
            location_matches_original=False,
            marker_verified=False,
        )
        services = RecoveryServices(
            recovery=recovery,
            verification={},
        )
        result = terminal.TerminalController(services).resume("/tmp/active-case")
        self.assertEqual(result["scientific_case_state"], "INCOMPLETE")
        self.assertEqual(result["inclusion"], "EXCLUDED")
        self.assertEqual(result["profiler_stream_state"], "NOT_APPLICABLE_R0")
        self.assertFalse(result["location_matches_original_publication"])
        self.assertEqual(
            [claim["eligibility"] for claim in result["claims"]],
            ["UNAVAILABLE", "UNAVAILABLE", "NOT_APPLICABLE"],
        )

        mutants = (
            ("verified", 1, "recovery_receipt_boolean"),
            ("mutated", 0, "recovery_receipt_boolean"),
            (
                "location_matches_original_publication",
                0,
                "recovery_receipt_location",
            ),
            (
                "publication_success_marker_verified",
                0,
                "recovery_marker_receipt",
            ),
        )
        for field, value, code in mutants:
            with self.subTest(field=field):
                services.recovery = copy.deepcopy(recovery)
                services.recovery[field] = value
                with self.assertRaises(terminal.ControllerError) as context:
                    terminal.TerminalController(services).resume("/tmp/active-case")
                self.assertEqual(context.exception.code, code)

        services.recovery = copy.deepcopy(recovery)
        services.recovery["unknown"] = None
        with self.assertRaises(terminal.ControllerError) as context:
            terminal.TerminalController(services).resume("/tmp/active-case")
        self.assertEqual(context.exception.code, "recovery_receipt_shape")

    def test_only_exact_original_verified_location_may_preserve_complete(self) -> None:
        run_id = "r0-x86-0001"
        original_path = Path("/tmp") / run_id
        complete = _complete_axes(verified=True)
        original = _recovery_receipt(
            original_path,
            complete,
            action="VERIFY_EXISTING_ORIGINAL",
            location_matches_original=True,
            marker_verified=True,
        )
        services = RecoveryServices(recovery=original, verification={})
        result = terminal.TerminalController(services).resume("/tmp/active-case")
        self.assertEqual(result["scientific_case_state"], "COMPLETE")
        self.assertEqual(result["inclusion"], "INCLUDED")
        self.assertIs(result["location_matches_original_publication"], True)

        services.recovery = copy.deepcopy(original)
        services.recovery["location_matches_original_publication"] = False
        with self.assertRaises(terminal.ControllerError) as context:
            terminal.TerminalController(services).resume("/tmp/active-case")
        self.assertEqual(context.exception.code, "recovery_original_location")

        recovered_path = Path("/tmp") / f"{run_id}.recovered-incomplete"
        services.recovery = _recovery_receipt(
            recovered_path,
            complete,
            action="VERIFY_EXISTING_RECOVERED",
            location_matches_original=False,
            marker_verified=False,
        )
        with self.assertRaises(terminal.ControllerError) as context:
            terminal.TerminalController(services).resume("/tmp/active-case")
        self.assertEqual(context.exception.code, "recovery_upgrade_forbidden")

    def test_verify_receipt_is_strict_read_only_and_preserves_unverified_complete(self) -> None:
        archive_path = Path("/tmp/r0-x86-0001")
        verified = _verification_receipt(
            archive_path,
            _complete_axes(verified=True),
            location_matches_original=True,
            marker_verified=True,
        )
        services = RecoveryServices(
            recovery={},
            verification=verified,
        )
        result = terminal.TerminalController(services).verify(archive_path)
        self.assertEqual(result["archive_integrity"], "VERIFIED")
        self.assertEqual(result["scientific_case_state"], "COMPLETE")
        self.assertEqual(result["inclusion"], "INCLUDED")

        services.verification["mutated"] = True
        with self.assertRaises(terminal.ControllerError) as context:
            terminal.TerminalController(services).verify(archive_path)
        self.assertEqual(context.exception.code, "verify_mutation_forbidden")

        unverified = _verification_receipt(
            archive_path,
            _complete_axes(verified=False),
            location_matches_original=False,
            marker_verified=False,
        )
        services.verification = unverified
        result = terminal.TerminalController(services).verify(archive_path)
        self.assertEqual(result["scientific_case_state"], "COMPLETE")
        self.assertEqual(result["archive_integrity"], "UNVERIFIED")
        self.assertEqual(result["inclusion"], "EXCLUDED")
        self.assertEqual(
            [claim["eligibility"] for claim in result["claims"][:2]],
            ["UNVERIFIED", "UNVERIFIED"],
        )

        for field, value, code in (
            ("verified", 1, "verification_receipt_schema"),
            ("mutated", 0, "verification_receipt_schema"),
        ):
            with self.subTest(field=field):
                services.verification = copy.deepcopy(unverified)
                services.verification[field] = value
                with self.assertRaises(terminal.ControllerError) as context:
                    terminal.TerminalController(services).verify(archive_path)
                self.assertEqual(context.exception.code, code)

        services.verification = copy.deepcopy(unverified)
        services.verification["unknown"] = None
        with self.assertRaises(terminal.ControllerError) as context:
            terminal.TerminalController(services).verify(archive_path)
        self.assertEqual(context.exception.code, "verification_receipt_shape")

    def test_cli_argument_error_is_one_canonical_structured_record(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = terminal.main([])
        self.assertEqual(status, 2)
        raw = output.getvalue().encode("utf-8")
        self.assertTrue(raw.endswith(b"\n"))
        value = json.loads(raw)
        self.assertEqual(value["verdict"], "invalid")
        self.assertEqual(value["problem"]["code"], "argument_parse_error")
        self.assertEqual(raw[:-1], terminal.canonical_json_bytes(value))

    def test_cli_oversize_result_emits_bounded_fallback_and_exits_two(self) -> None:
        fallback = {
            "problem": {
                "code": "cli_output_limit",
                "message": "canonical result exceeds declared bound",
            },
            "schema": terminal.TERMINAL_RESULT_SCHEMA,
            "verdict": "invalid",
        }
        maximum_bytes = len(terminal.canonical_json_bytes(fallback) + b"\n")
        output = io.StringIO()
        with mock.patch.object(
            terminal,
            "controller_self_test",
            return_value={"oversized": "x" * (maximum_bytes + 1)},
        ), mock.patch.object(
            terminal, "FALLBACK_MAXIMUM_CLI_BYTES", maximum_bytes
        ), contextlib.redirect_stdout(output):
            status = terminal.main(["self-test"])
        self.assertEqual(status, 2)
        raw = output.getvalue().encode("utf-8")
        self.assertEqual(raw, terminal.canonical_json_bytes(fallback) + b"\n")

    def test_cli_emits_postcompletion_receipt_and_exits_two(self) -> None:
        plan = terminal.parse_plan(_canonical(_valid_plan_value()))
        problem = {
            "code": "OSError",
            "message": "injected postcompletion restoration failure",
            "phase": "postcompletion_signal_mask_restoration",
            "scientific_effect": (
                "none_durable_success_marker_already_committed"
            ),
        }
        result = {
            "axes": {
                "archive_integrity": "VERIFIED",
                "scientific_case_state": "COMPLETE",
            },
            "publication": {
                "postcompletion_operational_problem": problem,
                "verified": True,
            },
            "schema": terminal.TERMINAL_RESULT_SCHEMA,
        }
        output = io.StringIO()
        with mock.patch.object(
            terminal, "load_plan", return_value=plan
        ), mock.patch.object(
            terminal.TerminalController,
            "run",
            return_value=result,
        ) as runner, contextlib.redirect_stdout(output):
            status = terminal.main(
                ["run", "--plan", "/tmp/injected-plan.json"],
                services=mock.sentinel.services,
            )
        runner.assert_called_once_with(plan)
        self.assertEqual(status, 2)
        raw = output.getvalue().encode("utf-8")
        self.assertEqual(raw, terminal.canonical_json_bytes(result) + b"\n")

    def test_cli_admits_module_only_and_rejects_direct_script_invocation(self) -> None:
        script = Path(terminal.__file__).resolve()
        common = {
            "cwd": REPOSITORY_ROOT,
            "env": {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "check": False,
            "timeout": 10,
        }
        direct = subprocess.run(
            (sys.executable, "-B", str(script)),
            **common,
        )
        self.assertEqual(direct.returncode, 2)
        self.assertEqual(direct.stderr, b"")
        self.assertEqual(
            json.loads(direct.stdout)["problem"]["code"],
            "direct_script_invocation_not_admitted",
        )

        module = subprocess.run(
            (
                sys.executable,
                "-B",
                "-m",
                "tools.profiling.memory.oai_memprof_r0_terminal",
            ),
            **common,
        )
        self.assertEqual(module.returncode, 2)
        self.assertEqual(module.stderr, b"")
        self.assertEqual(
            json.loads(module.stdout)["problem"]["code"],
            "argument_parse_error",
        )


if __name__ == "__main__":
    unittest.main()
