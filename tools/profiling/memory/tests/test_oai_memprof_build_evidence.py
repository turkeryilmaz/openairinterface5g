#!/usr/bin/env python3
"""Deterministic source, process, and publication boundary tests."""

from __future__ import annotations

import copy
import importlib.util
import os
import pathlib
import re
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


HERE = pathlib.Path(__file__).resolve().parent
MODULE_PATH = HERE.parent / "oai_memprof_build_evidence.py"
SPEC = importlib.util.spec_from_file_location("oai_memprof_build_evidence", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("build-evidence module specification unavailable")
B = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = B
SPEC.loader.exec_module(B)


class BuildEvidenceBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="oai-memprof-build-evidence-test-"
        )
        self.root = pathlib.Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_host(
        self, arguments: tuple[str, ...], *, cwd: pathlib.Path | None = None
    ) -> bytes:
        completed = subprocess.run(
            arguments,
            cwd=cwd,
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode("utf-8", "replace"),
        )
        return completed.stdout

    def make_project(self, name: str = "source") -> tuple[pathlib.Path, pathlib.Path]:
        source = self.root / name
        build = self.root / f"{name}-build"
        source.mkdir()
        (source / "CMakeLists.txt").write_text(
            "\n".join(
                (
                    "cmake_minimum_required(VERSION 3.16)",
                    "project(memprof_build_evidence_fixture C)",
                    "add_executable(app main.c)",
                    'target_link_options(app PRIVATE "-Wl,-Map,${CMAKE_BINARY_DIR}/app.map" "-Wl,--cref")',
                    "",
                )
            ),
            encoding="utf-8",
        )
        (source / "main.c").write_text(
            "int main(void) { return 0; }\n", encoding="utf-8"
        )
        self.run_host(("/usr/bin/git", "init", "-q"), cwd=source)
        self.run_host(("/usr/bin/git", "add", "CMakeLists.txt", "main.c"), cwd=source)
        self.run_host(
            (
                "/usr/bin/git",
                "-c",
                "user.name=Profiler Test",
                "-c",
                "user.email=profiler-test@example.invalid",
                "commit",
                "-q",
                "--no-gpg-sign",
                "-m",
                "fixture",
            ),
            cwd=source,
        )
        self.run_host(
            (
                "/usr/bin/cmake",
                "-S",
                str(source),
                "-B",
                str(build),
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            )
        )
        return source, build

    def libc_path(self) -> pathlib.Path:
        raw = self.run_host(("/usr/bin/gcc", "-print-file-name=libc.so.6"))
        return pathlib.Path(raw.decode("utf-8", "strict").strip()).resolve(strict=True)

    @staticmethod
    def logical(build: pathlib.Path, *, target: str = "app") -> B.LogicalElfInput:
        return B.LogicalElfInput(
            logical_id="app",
            target=target,
            elf_path=build / "app",
            link_map_path=build / "app.map",
            repo_path="build/app",
            elf_kind_id=1,
            role_ids=(1,),
        )

    def prepare(
        self,
        source: pathlib.Path,
        build: pathlib.Path,
        evidence: pathlib.Path,
        *,
        logical: B.LogicalElfInput | None = None,
        ninja: pathlib.Path = pathlib.Path("/usr/bin/ninja"),
    ) -> B.PreparedBuildEvidence:
        evidence.mkdir()
        return B.prepare_measured_build_evidence(
            repository=source,
            build_directory=build,
            evidence_root=evidence,
            logical_elfs=(self.logical(build) if logical is None else logical,),
            primary_logical_elf_id="app",
            libc_path=self.libc_path(),
            api_definition_sha256="1" * 64,
            ninja=ninja,
        )

    def test_real_cmake_ninja_target_reaches_exact_validated_publication(self) -> None:
        source, build = self.make_project()
        evidence = self.root / "evidence"
        prepared = self.prepare(source, build, evidence)
        artifacts = dict(prepared.artifacts)
        derived = B.validate_build_evidence_bytes(
            prepared.evidence_bytes,
            artifacts,
            prepared.coverage_bytes,
            api_definition_sha256="1" * 64,
        )
        self.assertEqual(derived["build_identity"]["dirty"], False)
        self.assertEqual(derived["build_identity"]["primary_logical_elf_id"], "app")
        self.assertEqual(
            (evidence / B.EVIDENCE_ARCHIVE_PATH).read_bytes(), prepared.evidence_bytes
        )
        self.assertEqual(
            (evidence / B.COVERAGE.BUILD_COVERAGE_ARCHIVE_PATH).read_bytes(),
            prepared.coverage_bytes,
        )

    def test_foreign_cmake_source_root_is_rejected_before_target_construction(self) -> None:
        source, _build = self.make_project("source-a")
        _foreign_source, foreign_build = self.make_project("source-b")
        evidence = self.root / "evidence"
        with self.assertRaisesRegex(B.BuildEvidenceError, "another source root"):
            self.prepare(
                source,
                foreign_build,
                evidence,
                logical=self.logical(foreign_build),
            )
        self.assertEqual(list(evidence.iterdir()), [])

    def test_missing_ninja_target_is_rejected_without_publication(self) -> None:
        source, build = self.make_project()
        evidence = self.root / "evidence"
        with self.assertRaisesRegex(B.BuildEvidenceError, "tool failed"):
            self.prepare(
                source,
                build,
                evidence,
                logical=self.logical(build, target="missing-target"),
            )
        self.assertEqual(list(evidence.iterdir()), [])

    def test_link_map_output_must_name_the_measured_elf(self) -> None:
        source, build = self.make_project()
        self.run_host(("/usr/bin/ninja", "-C", str(build), "app"))
        map_path = build / "app.map"
        original = map_path.read_bytes()
        mutant = re.sub(rb"(?m)^OUTPUT\([^ ]+", b"OUTPUT(not-app", original, count=1)
        self.assertNotEqual(mutant, original)
        map_path.write_bytes(mutant)
        evidence = self.root / "evidence"
        with self.assertRaisesRegex(B.BuildEvidenceError, "OUTPUT does not name"):
            self.prepare(source, build, evidence)
        self.assertEqual(list(evidence.iterdir()), [])

    def test_offline_verifier_rejects_coherent_link_map_output_mutant(self) -> None:
        source, build = self.make_project()
        prepared = self.prepare(source, build, self.root / "evidence")
        evidence = B.COVERAGE.parse_canonical(prepared.evidence_bytes)
        artifacts = dict(prepared.artifacts)
        logical = evidence["logical_elfs"][0]
        map_path = logical["link_map_path"]
        original = artifacts[map_path]
        mutant = re.sub(rb"(?m)^OUTPUT\([^ ]+", b"OUTPUT(not-app", original, count=1)
        self.assertNotEqual(mutant, original)
        artifacts[map_path] = mutant
        manifest_row = next(row for row in evidence["entries"] if row["path"] == map_path)
        manifest_row.update(bytes=len(mutant), sha256=B._sha(mutant))

        coverage = B.COVERAGE.parse_canonical(prepared.coverage_bytes)
        coverage_row = next(
            row
            for row in coverage["entries"]
            if row["logical_id"] == logical["logical_id"]
        )
        coverage_row["link_map_sha256"] = B._sha(mutant)
        logical_by_id = {
            row["logical_id"]: row for row in evidence["logical_elfs"]
        }
        projection = [
            {
                "build_map_path": logical_by_id[row["logical_id"]]["build_map_path"],
                "build_output_path": logical_by_id[row["logical_id"]]["build_output_path"],
                "link_command_sha256": row["link_command_sha256"],
                "link_map_sha256": row["link_map_sha256"],
                "logical_id": row["logical_id"],
            }
            for row in coverage["entries"]
        ]
        identity = coverage["build_identity"]
        configuration = B.COVERAGE.canonical_bytes(
            {
                "build_directory": evidence["build_directory"],
                "compiler_version": identity["compiler_version"],
                "entries": projection,
                "linker_version": identity["linker_version"],
                "source_commit": identity["source_commit"],
                "source_tree": identity["source_tree"],
                "target_triple": identity["target_triple"],
                "version": B.VERSION,
            }
        )
        identity["build_configuration_sha256"] = B._sha(configuration)
        coverage_raw = B.COVERAGE.canonical_bytes(coverage)
        evidence["build_coverage_sha256"] = B._sha(coverage_raw)
        evidence_raw = B.COVERAGE.canonical_bytes(evidence)

        with self.assertRaisesRegex(B.BuildEvidenceError, "OUTPUT does not name"):
            B.validate_build_evidence_bytes(
                evidence_raw,
                artifacts,
                coverage_raw,
                api_definition_sha256="1" * 64,
            )

    def test_missing_fixed_and_logical_artifacts_use_bounded_error_contract(self) -> None:
        source, build = self.make_project()
        prepared = self.prepare(source, build, self.root / "evidence")
        original_evidence = B.COVERAGE.parse_canonical(prepared.evidence_bytes)
        missing_paths = (
            original_evidence["toolchain"]["source_status_path"],
            original_evidence["logical_elfs"][0]["link_map_path"],
        )
        for missing_path in missing_paths:
            with self.subTest(path=missing_path):
                evidence = copy.deepcopy(original_evidence)
                artifacts = dict(prepared.artifacts)
                artifacts.pop(missing_path)
                evidence["entries"] = [
                    row for row in evidence["entries"] if row["path"] != missing_path
                ]
                with self.assertRaisesRegex(
                    B.BuildEvidenceError, "required artifact unavailable"
                ):
                    B.validate_build_evidence_bytes(
                        B.COVERAGE.canonical_bytes(evidence),
                        artifacts,
                        prepared.coverage_bytes,
                        api_definition_sha256="1" * 64,
                    )

    def test_source_mutation_during_target_construction_is_rejected(self) -> None:
        source, build = self.make_project()
        wrapper = self.root / "mutating-ninja"
        wrapper.write_text(
            "\n".join(
                (
                    "#!/usr/bin/python3",
                    "import pathlib, subprocess, sys",
                    "result = subprocess.run(['/usr/bin/ninja', *sys.argv[1:]], stdout=subprocess.PIPE, stderr=subprocess.PIPE)",
                    "sys.stdout.buffer.write(result.stdout)",
                    "sys.stderr.buffer.write(result.stderr)",
                    f"pathlib.Path({str(source / 'main.c')!r}).write_text('int main(void) {{ return 0; }}\\n\\n', encoding='utf-8')",
                    "raise SystemExit(result.returncode)",
                    "",
                )
            ),
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
        evidence = self.root / "evidence"
        with self.assertRaisesRegex(B.BuildEvidenceError, "repository is dirty"):
            self.prepare(source, build, evidence, ninja=wrapper)
        self.assertEqual(list(evidence.iterdir()), [])

    def test_build_artifact_paths_must_remain_under_build_root(self) -> None:
        source, build = self.make_project()
        outside = self.root / "outside-app"
        outside.write_bytes(b"not an ELF")
        logical = self.logical(build)
        logical = B.LogicalElfInput(
            logical.logical_id,
            logical.target,
            outside,
            logical.link_map_path,
            logical.repo_path,
            logical.elf_kind_id,
            logical.role_ids,
        )
        evidence = self.root / "evidence"
        with self.assertRaisesRegex(B.BuildEvidenceError, "escapes the build directory"):
            self.prepare(source, build, evidence, logical=logical)
        self.assertEqual(list(evidence.iterdir()), [])

    def test_publication_rejects_symlink_roots_parents_leaves_and_overwrite(self) -> None:
        root = self.root / "publication"
        outside = self.root / "outside"
        root.mkdir()
        outside.mkdir()
        root_alias = self.root / "root-alias"
        root_alias.symlink_to(root, target_is_directory=True)
        with self.assertRaises(OSError):
            B._publish(root_alias, "value.bin", b"value")

        (root / "redirect").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(OSError):
            B._publish(root, "redirect/value.bin", b"value")
        self.assertFalse((outside / "value.bin").exists())

        outside_leaf = outside / "leaf"
        outside_leaf.write_bytes(b"outside")
        (root / "leaf").symlink_to(outside_leaf)
        with self.assertRaises(FileExistsError):
            B._publish(root, "leaf", b"value")
        self.assertEqual(outside_leaf.read_bytes(), b"outside")

        B._publish(root, "stable.bin", b"stable")
        with self.assertRaises(FileExistsError):
            B._publish(root, "stable.bin", b"replacement")
        self.assertEqual((root / "stable.bin").read_bytes(), b"stable")

    def test_tool_stdout_and_stderr_overflow_are_bounded_and_reaped(self) -> None:
        for descriptor, name in ((1, "stdout"), (2, "stderr")):
            with self.subTest(stream=name):
                pid_path = self.root / f"{name}.pid"
                code = (
                    "import os\n"
                    f"open({str(pid_path)!r}, 'w', encoding='ascii').write(str(os.getpid()))\n"
                    "chunk = b'x' * 4096\n"
                    f"while True: os.write({descriptor}, chunk)\n"
                )
                patches = (
                    mock.patch.object(B, "MAX_TOOL_OUTPUT_BYTES", 128),
                    mock.patch.object(B, "MAX_TOOL_ERROR_BYTES", 128),
                    mock.patch.object(B, "TOOL_TERMINATION_GRACE_SECONDS", 0.05),
                )
                with patches[0], patches[1], patches[2], self.assertRaisesRegex(
                    B.BuildEvidenceError, f"{name} exceeds bound"
                ):
                    B._run(("/usr/bin/python3", "-c", code))
                pid = int(pid_path.read_text(encoding="ascii"))
                with self.assertRaises(ChildProcessError):
                    os.waitpid(pid, os.WNOHANG)

    def test_tool_timeout_is_stable_and_direct_child_is_reaped(self) -> None:
        pid_path = self.root / "timeout.pid"
        code = (
            "import os, time\n"
            f"open({str(pid_path)!r}, 'w', encoding='ascii').write(str(os.getpid()))\n"
            "os.close(1)\n"
            "os.close(2)\n"
            "time.sleep(60)\n"
        )
        with mock.patch.object(B, "TOOL_TIMEOUT_SECONDS", 0.1), mock.patch.object(
            B, "TOOL_TERMINATION_GRACE_SECONDS", 0.05
        ), self.assertRaisesRegex(B.BuildEvidenceError, "tool timeout"):
            B._run(("/usr/bin/python3", "-c", code))
        pid = int(pid_path.read_text(encoding="ascii"))
        with self.assertRaises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)

    def test_selector_setup_failure_terminates_started_process(self) -> None:
        pid_path = self.root / "selector-failure.pid"
        code = (
            "import os, pathlib, time\n"
            f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid()), encoding='ascii')\n"
            "time.sleep(60)\n"
        )

        def fail_after_child_starts() -> None:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not pid_path.exists():
                time.sleep(0.01)
            if not pid_path.exists():
                self.fail("tool did not publish its PID before selector failure")
            raise OSError("fixture selector failure")

        pid: int | None = None
        try:
            with mock.patch.object(
                B.selectors, "DefaultSelector", side_effect=fail_after_child_starts
            ), mock.patch.object(
                B, "TOOL_TERMINATION_GRACE_SECONDS", 0.05
            ), self.assertRaisesRegex(OSError, "fixture selector failure"):
                B._run(("/usr/bin/python3", "-c", code))
            pid = int(pid_path.read_text(encoding="ascii"))
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
            with self.assertRaises(ChildProcessError):
                os.waitpid(pid, os.WNOHANG)
        finally:
            if pid is not None:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_timeout_terminates_pipe_holding_descendant_after_leader_exit(self) -> None:
        descendant_pid_path = self.root / "descendant.pid"
        code = (
            "import os, time\n"
            "child = os.fork()\n"
            "if child == 0:\n"
            f"    with open({str(descendant_pid_path)!r}, 'w', encoding='ascii') as stream:\n"
            "        stream.write(str(os.getpid()))\n"
            "        stream.flush()\n"
            "        os.fsync(stream.fileno())\n"
            "    time.sleep(60)\n"
            "    os._exit(0)\n"
            "os._exit(0)\n"
        )
        descendant_pid: int | None = None
        try:
            with mock.patch.object(B, "TOOL_TIMEOUT_SECONDS", 0.2), mock.patch.object(
                B, "TOOL_TERMINATION_GRACE_SECONDS", 0.05
            ), self.assertRaisesRegex(B.BuildEvidenceError, "tool timeout"):
                B._run(("/usr/bin/python3", "-c", code))
            descendant_pid = int(descendant_pid_path.read_text(encoding="ascii"))
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                try:
                    os.kill(descendant_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.01)
            else:
                self.fail("pipe-holding descendant survived process-group cleanup")
        finally:
            if descendant_pid is not None:
                try:
                    os.kill(descendant_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    unittest.main()
