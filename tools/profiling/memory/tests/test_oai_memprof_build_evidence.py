#!/usr/bin/env python3
"""Deterministic source, process, and publication boundary tests."""

from __future__ import annotations

import copy
import importlib.util
import os
import pathlib
import re
import signal
import struct
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

    def make_project(
        self,
        name: str = "source",
        *,
        archive_append_source: str = "int main(void) { return 0; }\n",
        archive_append_link_options: tuple[str, ...] = (),
    ) -> tuple[pathlib.Path, pathlib.Path]:
        source = self.root / name
        build = self.root / f"{name}-build"
        source.mkdir()
        self.assertTrue(
            all('"' not in option for option in archive_append_link_options)
        )
        archive_append_options = " ".join(
            f'"{option}"'
            for option in (
                "-static",
                "-no-pie",
                "-Wl,-Map,${CMAKE_BINARY_DIR}/oai_memprof_archive_append.map",
                "-Wl,--cref",
                *archive_append_link_options,
            )
        )
        (source / "CMakeLists.txt").write_text(
            "\n".join(
                (
                    "cmake_minimum_required(VERSION 3.16)",
                    "project(memprof_build_evidence_fixture C)",
                    "add_executable(app main.c)",
                    'target_link_options(app PRIVATE "-Wl,-Map,${CMAKE_BINARY_DIR}/app.map" "-Wl,--cref")',
                    "add_executable(oai_memprof_archive_append archive_append.c)",
                    f"target_link_options(oai_memprof_archive_append PRIVATE {archive_append_options})",
                    "",
                )
            ),
            encoding="utf-8",
        )
        (source / "main.c").write_text(
            "int main(void) { return 0; }\n", encoding="utf-8"
        )
        (source / "archive_append.c").write_text(
            archive_append_source, encoding="utf-8"
        )
        self.run_host(("/usr/bin/git", "init", "-q"), cwd=source)
        self.run_host(
            ("/usr/bin/git", "add", "CMakeLists.txt", "archive_append.c", "main.c"),
            cwd=source,
        )
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

    @staticmethod
    def auxiliary(
        build: pathlib.Path,
        *, target: str = B.ARCHIVE_APPEND_TARGET,
    ) -> B.AuxiliaryExecutableInput:
        return B.AuxiliaryExecutableInput(
            auxiliary_id=B.ARCHIVE_APPEND_AUXILIARY_ID,
            target=target,
            elf_path=build / "oai_memprof_archive_append",
            link_map_path=build / "oai_memprof_archive_append.map",
        )

    @staticmethod
    def set_elf_type(raw: bytes, elf_type: int) -> bytes:
        mutant = bytearray(raw)
        struct.pack_into("<H", mutant, 16, elf_type)
        return bytes(mutant)

    @staticmethod
    def clear_program_header_type(raw: bytes, program_type: int) -> bytes:
        program_offset = struct.unpack_from("<Q", raw, 32)[0]
        program_entry_size = struct.unpack_from("<H", raw, 54)[0]
        program_count = struct.unpack_from("<H", raw, 56)[0]
        mutant = bytearray(raw)
        count = 0
        for index in range(program_count):
            offset = program_offset + index * program_entry_size
            if struct.unpack_from("<I", raw, offset)[0] == program_type:
                struct.pack_into("<I", mutant, offset, 0)
                count += 1
        if count == 0:
            raise AssertionError(f"fixture lacks program-header type {program_type}")
        return bytes(mutant)

    @staticmethod
    def replace_artifact(
        evidence: dict[str, object],
        artifacts: dict[str, bytes],
        path: str,
        raw: bytes,
    ) -> None:
        artifacts[path] = raw
        row = next(
            value
            for value in evidence["entries"]
            if isinstance(value, dict) and value.get("path") == path
        )
        row.update(bytes=len(raw), sha256=B._sha(raw))

    @staticmethod
    def mutate_auxiliary_symbol_to_undefined(
        raw: bytes, *, original_name: str, forbidden_name: str
    ) -> bytes:
        _machine, sections, _bodies = B._parse_sections(raw)
        symtabs = [section for section in sections if section.kind == 2]
        if len(symtabs) != 1:
            raise AssertionError("fixture lacks exactly one static symbol table")
        symtab = symtabs[0]
        string_sections = [
            section for section in sections if section.index == symtab.link
        ]
        if len(string_sections) != 1:
            raise AssertionError("fixture static symbol strings are unavailable")
        strings = string_sections[0]
        original = original_name.encode("ascii")
        forbidden = forbidden_name.encode("ascii")
        if len(forbidden) > len(original):
            raise AssertionError("forbidden fixture symbol does not fit in-place")
        mutant = bytearray(raw)
        for index in range(symtab.size // symtab.entry_size):
            entry_offset = symtab.offset + index * symtab.entry_size
            name_offset = struct.unpack_from("<I", raw, entry_offset)[0]
            string_offset = strings.offset + name_offset
            end = raw.find(b"\0", string_offset)
            if end < string_offset or raw[string_offset:end] != original:
                continue
            mutant[string_offset : string_offset + len(forbidden)] = forbidden
            mutant[string_offset + len(forbidden)] = 0
            struct.pack_into("<H", mutant, entry_offset + 6, 0)  # SHN_UNDEF
            struct.pack_into("<Q", mutant, entry_offset + 8, 0)
            struct.pack_into("<Q", mutant, entry_offset + 16, 0)
            return bytes(mutant)
        raise AssertionError(f"fixture lacks static symbol {original_name!r}")

    def prepare(
        self,
        source: pathlib.Path,
        build: pathlib.Path,
        evidence: pathlib.Path,
        *,
        logical: B.LogicalElfInput | None = None,
        auxiliary: B.AuxiliaryExecutableInput | None = None,
        ninja: pathlib.Path = pathlib.Path("/usr/bin/ninja"),
    ) -> B.PreparedBuildEvidence:
        evidence.mkdir()
        return B.prepare_measured_build_evidence(
            repository=source,
            build_directory=build,
            evidence_root=evidence,
            logical_elfs=(self.logical(build) if logical is None else logical,),
            auxiliary_executables=(
                self.auxiliary(build) if auxiliary is None else auxiliary,
            ),
            primary_logical_elf_id="app",
            libc_path=self.libc_path(),
            api_definition_sha256="1" * 64,
            ninja=ninja,
        )

    def assert_auxiliary_command_rejected(
        self,
        prepared: B.PreparedBuildEvidence,
        command_raw: bytes,
        expression: str,
    ) -> None:
        evidence = B.COVERAGE.parse_canonical(prepared.evidence_bytes)
        auxiliary = evidence["auxiliary_executables"][0]
        with self.assertRaisesRegex(B.BuildEvidenceError, expression):
            B._validate_final_link_command(
                command_raw,
                build_directory=evidence["build_directory"],
                build_output_path=auxiliary["build_output_path"],
                build_map_path=auxiliary["build_map_path"],
                expected_wraps=(),
                require_static_executable=True,
                where="test auxiliary final-link command",
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
        self.assertEqual(B.VERSION, {"major": 1, "minor": 4})
        evidence_value = B.COVERAGE.parse_canonical(prepared.evidence_bytes)
        self.assertEqual(
            evidence_value["auxiliary_executables"],
            [
                {
                    "auxiliary_id": B.ARCHIVE_APPEND_AUXILIARY_ID,
                    "build_map_path": "oai_memprof_archive_append.map",
                    "build_output_path": "oai_memprof_archive_append",
                    "elf_path": "input/build-evidence/auxiliary/archive_append.elf",
                    "link_command_path": "input/build-evidence/auxiliary/archive_append.link-command.txt",
                    "link_map_path": "input/build-evidence/auxiliary/archive_append.link-map.txt",
                    "target": B.ARCHIVE_APPEND_TARGET,
                }
            ],
        )
        self.assertNotIn(
            B.ARCHIVE_APPEND_AUXILIARY_ID,
            {row["logical_id"] for row in derived["entries"]},
        )
        auxiliary_paths = B._auxiliary_paths(B.ARCHIVE_APPEND_AUXILIARY_ID)
        auxiliary_raw = artifacts[auxiliary_paths["elf_path"]]
        self.assertEqual(struct.unpack_from("<H", auxiliary_raw, 16)[0], 2)
        program_offset = struct.unpack_from("<Q", auxiliary_raw, 32)[0]
        program_entry_size = struct.unpack_from("<H", auxiliary_raw, 54)[0]
        program_count = struct.unpack_from("<H", auxiliary_raw, 56)[0]
        program_types = {
            struct.unpack_from(
                "<I", auxiliary_raw, program_offset + index * program_entry_size
            )[0]
            for index in range(program_count)
        }
        self.assertNotIn(2, program_types)  # PT_DYNAMIC
        self.assertNotIn(3, program_types)  # PT_INTERP
        _machine, sections, _bodies = B._parse_sections(auxiliary_raw)
        self.assertFalse(any(section.kind in (6, 11) for section in sections))
        command_raw = artifacts[auxiliary_paths["link_command_path"]]
        self.assertEqual(len(re.findall(rb"(?<!\S)-static(?!\S)", command_raw)), 1)
        self.assertEqual(len(re.findall(rb"(?<!\S)-no-pie(?!\S)", command_raw)), 1)
        self.assertIn(b"-Wl,-Map,", command_raw)
        self.assertIn(b"-Wl,--cref", command_raw)
        self.assertNotIn(b"liboai_memprof_active_runtime", command_raw)
        self.assertNotIn(b"rpath", command_raw.lower())
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
        auxiliary = evidence["auxiliary_executables"][0]
        auxiliary_paths = B._auxiliary_paths(auxiliary["auxiliary_id"])
        auxiliary_projection = [
            {
                "auxiliary_id": auxiliary["auxiliary_id"],
                "build_map_path": auxiliary["build_map_path"],
                "build_output_path": auxiliary["build_output_path"],
                "elf_sha256": B._sha(artifacts[auxiliary_paths["elf_path"]]),
                "link_command_sha256": B._sha(
                    artifacts[auxiliary_paths["link_command_path"]]
                ),
                "link_map_sha256": B._sha(
                    artifacts[auxiliary_paths["link_map_path"]]
                ),
                "target": auxiliary["target"],
            }
        ]
        identity = coverage["build_identity"]
        configuration = B.COVERAGE.canonical_bytes(
            {
                "auxiliary_executables": auxiliary_projection,
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
            original_evidence["auxiliary_executables"][0]["link_map_path"],
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

    def test_auxiliary_row_is_exact_and_configuration_bound(self) -> None:
        source, build = self.make_project()
        prepared = self.prepare(source, build, self.root / "evidence")
        original_evidence = B.COVERAGE.parse_canonical(prepared.evidence_bytes)
        auxiliary = original_evidence["auxiliary_executables"][0]
        auxiliary_paths = B._auxiliary_paths(B.ARCHIVE_APPEND_AUXILIARY_ID)
        self.assertEqual(set(auxiliary_paths.values()).issubset(dict(prepared.artifacts)), True)

        for replacement in (
            [],
            [copy.deepcopy(auxiliary), copy.deepcopy(auxiliary)],
        ):
            with self.subTest(row_count=len(replacement)):
                evidence = copy.deepcopy(original_evidence)
                evidence["auxiliary_executables"] = replacement
                with self.assertRaisesRegex(
                    B.BuildEvidenceError, "exactly one row required"
                ):
                    B._derive(
                        evidence,
                        dict(prepared.artifacts),
                        api_definition_sha256="1" * 64,
                        enforce_advertised_digest=False,
                    )

        evidence = copy.deepcopy(original_evidence)
        artifacts = dict(prepared.artifacts)
        elf_path = auxiliary_paths["elf_path"]
        artifacts[elf_path] += b"\n"
        manifest = next(row for row in evidence["entries"] if row["path"] == elf_path)
        manifest.update(bytes=len(artifacts[elf_path]), sha256=B._sha(artifacts[elf_path]))
        with self.assertRaisesRegex(
            B.BuildEvidenceError, "derived build-coverage digest mismatch"
        ):
            B._derive(
                evidence,
                artifacts,
                api_definition_sha256="1" * 64,
            )

    def test_auxiliary_architecture_and_map_are_validated_without_coverage_row(
        self,
    ) -> None:
        source, build = self.make_project()
        prepared = self.prepare(source, build, self.root / "evidence")
        original_evidence = B.COVERAGE.parse_canonical(prepared.evidence_bytes)
        auxiliary_paths = B._auxiliary_paths(B.ARCHIVE_APPEND_AUXILIARY_ID)

        evidence = copy.deepcopy(original_evidence)
        artifacts = dict(prepared.artifacts)
        elf_path = auxiliary_paths["elf_path"]
        elf_mutant = bytearray(artifacts[elf_path])
        observed_machine = int.from_bytes(elf_mutant[18:20], "little")
        elf_mutant[18:20] = (183 if observed_machine == 62 else 62).to_bytes(
            2, "little"
        )
        artifacts[elf_path] = bytes(elf_mutant)
        manifest = next(row for row in evidence["entries"] if row["path"] == elf_path)
        manifest.update(bytes=len(elf_mutant), sha256=B._sha(bytes(elf_mutant)))
        with self.assertRaisesRegex(B.BuildEvidenceError, "architecture differs"):
            B._derive(
                evidence,
                artifacts,
                api_definition_sha256="1" * 64,
                enforce_advertised_digest=False,
            )

        evidence = copy.deepcopy(original_evidence)
        artifacts = dict(prepared.artifacts)
        map_path = auxiliary_paths["link_map_path"]
        map_mutant = re.sub(
            rb"(?m)^OUTPUT\([^ ]+",
            b"OUTPUT(not-archive-appender",
            artifacts[map_path],
            count=1,
        )
        self.assertNotEqual(map_mutant, artifacts[map_path])
        artifacts[map_path] = map_mutant
        manifest = next(row for row in evidence["entries"] if row["path"] == map_path)
        manifest.update(bytes=len(map_mutant), sha256=B._sha(map_mutant))
        with self.assertRaisesRegex(B.BuildEvidenceError, "OUTPUT does not name"):
            B._derive(
                evidence,
                artifacts,
                api_definition_sha256="1" * 64,
                enforce_advertised_digest=False,
            )

        coverage = B.COVERAGE.parse_canonical(prepared.coverage_bytes)
        self.assertNotIn(
            B.ARCHIVE_APPEND_AUXILIARY_ID,
            {row["logical_id"] for row in coverage["entries"]},
        )

    def test_auxiliary_defined_wrap_and_real_symbols_are_rejected(self) -> None:
        sources = (
            (
                "defined-wrap",
                "void __wrap_archive_append_fixture(void) {}\n"
                "int main(void) { return 0; }\n",
            ),
            (
                "defined-real",
                "void __real_archive_append_fixture(void) {}\n"
                "int main(void) { return 0; }\n",
            ),
        )
        for label, archive_append_source in sources:
            with self.subTest(symbol=label):
                source, build = self.make_project(
                    f"source-symbol-{label}",
                    archive_append_source=archive_append_source,
                )
                with self.assertRaisesRegex(
                    B.BuildEvidenceError, "__wrap_/__real_ symbols"
                ):
                    self.prepare(
                        source, build, self.root / f"evidence-symbol-{label}"
                    )

    def test_auxiliary_undefined_wrap_and_real_symbols_are_rejected_offline(
        self,
    ) -> None:
        original_name = (
            "archive_append_symbol_padding_for_undefined_forbidden_prefix_fixture"
        )
        source, build = self.make_project(
            "source-undefined-symbol",
            archive_append_source=(
                "__attribute__((noinline, used))\n"
                f"void {original_name}(void) {{}}\n"
                f"int main(void) {{ {original_name}(); return 0; }}\n"
            ),
        )
        prepared = self.prepare(source, build, self.root / "evidence-undefined-symbol")
        elf_path = B._auxiliary_paths(B.ARCHIVE_APPEND_AUXILIARY_ID)["elf_path"]
        for forbidden_name in (
            "__wrap_archive_append_fixture",
            "__real_archive_append_fixture",
        ):
            with self.subTest(symbol=forbidden_name):
                evidence = B.COVERAGE.parse_canonical(prepared.evidence_bytes)
                artifacts = dict(prepared.artifacts)
                mutant = self.mutate_auxiliary_symbol_to_undefined(
                    artifacts[elf_path],
                    original_name=original_name,
                    forbidden_name=forbidden_name,
                )
                self.replace_artifact(evidence, artifacts, elf_path, mutant)
                _machine, sections, bodies = B._parse_sections(mutant)
                symbols = B._parse_auxiliary_symbols(
                    sections, bodies, "test undefined auxiliary symbol"
                )
                observed = next(
                    symbol for symbol in symbols if symbol.name == forbidden_name
                )
                self.assertFalse(observed.defined)
                with self.assertRaisesRegex(
                    B.BuildEvidenceError, "__wrap_/__real_ symbols"
                ):
                    B._derive(
                        evidence,
                        artifacts,
                        api_definition_sha256="1" * 64,
                        enforce_advertised_digest=False,
                    )

    def test_archive_appender_cannot_be_reclassified_as_logical_elf(self) -> None:
        source, build = self.make_project()
        logical = B.LogicalElfInput(
            logical_id="app",
            target=B.ARCHIVE_APPEND_TARGET,
            elf_path=build / "oai_memprof_archive_append",
            link_map_path=build / "oai_memprof_archive_append.map",
            repo_path="build/oai_memprof_archive_append",
            elf_kind_id=1,
            role_ids=(1,),
        )
        rejected_root = self.root / "evidence-rejected-logical-alias"
        with self.assertRaisesRegex(B.BuildEvidenceError, "auxiliary-only"):
            self.prepare(source, build, rejected_root, logical=logical)
        self.assertEqual(list(rejected_root.iterdir()), [])

        prepared = self.prepare(
            source, build, self.root / "evidence-valid-logical-alias"
        )
        original = B.COVERAGE.parse_canonical(prepared.evidence_bytes)
        auxiliary = original["auxiliary_executables"][0]
        for logical_field, auxiliary_field in (
            ("build_output_path", "build_output_path"),
            ("build_output_path", "build_map_path"),
            ("build_map_path", "build_output_path"),
            ("build_map_path", "build_map_path"),
        ):
            with self.subTest(
                logical_field=logical_field,
                auxiliary_field=auxiliary_field,
            ):
                evidence = copy.deepcopy(original)
                evidence["logical_elfs"][0][logical_field] = auxiliary[
                    auxiliary_field
                ]
                with self.assertRaisesRegex(
                    B.BuildEvidenceError, "globally unique"
                ):
                    B._derive(
                        evidence,
                        dict(prepared.artifacts),
                        api_definition_sha256="1" * 64,
                        enforce_advertised_digest=False,
                    )

    def test_build_output_and_map_paths_are_globally_unique(self) -> None:
        source, build = self.make_project()
        prepared = self.prepare(source, build, self.root / "evidence")
        original = B.COVERAGE.parse_canonical(prepared.evidence_bytes)

        evidence = copy.deepcopy(original)
        auxiliary = evidence["auxiliary_executables"][0]
        auxiliary["build_map_path"] = auxiliary["build_output_path"]
        with self.assertRaisesRegex(B.BuildEvidenceError, "globally unique"):
            B._derive(
                evidence,
                dict(prepared.artifacts),
                api_definition_sha256="1" * 64,
                enforce_advertised_digest=False,
            )

        evidence = copy.deepcopy(original)
        logical = evidence["logical_elfs"][0]
        logical["build_map_path"] = logical["build_output_path"]
        with self.assertRaisesRegex(B.BuildEvidenceError, "globally unique"):
            B._derive(
                evidence,
                dict(prepared.artifacts),
                api_definition_sha256="1" * 64,
                enforce_advertised_digest=False,
            )

        evidence = copy.deepcopy(original)
        duplicate = copy.deepcopy(evidence["logical_elfs"][0])
        duplicate["logical_id"] = "app2"
        duplicate.update(B._entry_paths("app2"))
        evidence["logical_elfs"].append(duplicate)
        with self.assertRaisesRegex(B.BuildEvidenceError, "globally unique"):
            B._derive(
                evidence,
                dict(prepared.artifacts),
                api_definition_sha256="1" * 64,
                enforce_advertised_digest=False,
            )

    def test_online_resolved_and_device_inode_alias_claims_are_rejected(self) -> None:
        first = self.root / "first"
        second = self.root / "second"
        resolved_claims: dict[pathlib.Path, str] = {}
        identity_claims: dict[tuple[int, int], str] = {}
        B._claim_unique_online_artifact(
            resolved_claims,
            identity_claims,
            resolved_path=first,
            identity=(1, 2),
            where="first",
        )
        with self.assertRaisesRegex(B.BuildEvidenceError, "resolved build artifact"):
            B._claim_unique_online_artifact(
                resolved_claims,
                identity_claims,
                resolved_path=first,
                identity=(3, 4),
                where="same resolved path",
            )
        resolved_claims = {}
        identity_claims = {}
        B._claim_unique_online_artifact(
            resolved_claims,
            identity_claims,
            resolved_path=first,
            identity=(1, 2),
            where="first",
        )
        with self.assertRaisesRegex(B.BuildEvidenceError, "device/inode"):
            B._claim_unique_online_artifact(
                resolved_claims,
                identity_claims,
                resolved_path=second,
                identity=(1, 2),
                where="same device/inode",
            )

    def test_auxiliary_wrap_spellings_are_rejected_online_and_offline(self) -> None:
        for name, options in (
            ("wl-comma", ("-Wl,--wrap,archive_append_unused",)),
            (
                "xlinker",
                ("-Xlinker", "--wrap=archive_append_unused"),
            ),
        ):
            with self.subTest(spelling=name):
                source, build = self.make_project(
                    f"source-wrap-{name}",
                    archive_append_link_options=options,
                )
                evidence_root = self.root / f"evidence-wrap-{name}"
                with self.assertRaisesRegex(
                    B.BuildEvidenceError, "exact wrap set differs"
                ):
                    self.prepare(source, build, evidence_root)
                self.assertEqual(list(evidence_root.iterdir()), [])

        source, build = self.make_project("source-wrap-offline")
        prepared = self.prepare(
            source, build, self.root / "evidence-wrap-offline"
        )
        evidence = B.COVERAGE.parse_canonical(prepared.evidence_bytes)
        artifacts = dict(prepared.artifacts)
        command_path = B._auxiliary_paths(B.ARCHIVE_APPEND_AUXILIARY_ID)[
            "link_command_path"
        ]
        original_command = artifacts[command_path]
        mutant_command = original_command.replace(
            b" -o ",
            b" -Wl,--wrap,archive_append_unused -o ",
            1,
        )
        self.assertNotEqual(mutant_command, original_command)
        self.replace_artifact(
            evidence, artifacts, command_path, mutant_command
        )
        with self.assertRaisesRegex(B.BuildEvidenceError, "exact wrap set differs"):
            B._derive(
                evidence,
                artifacts,
                api_definition_sha256="1" * 64,
                enforce_advertised_digest=False,
            )

    def test_frozen_gnu_ld_wrap_grammar_rejects_every_auxiliary_channel(self) -> None:
        source, build = self.make_project("source-wrap-grammar")
        prepared = self.prepare(source, build, self.root / "evidence-wrap-grammar")
        command_path = B._auxiliary_paths(B.ARCHIVE_APPEND_AUXILIARY_ID)[
            "link_command_path"
        ]
        original_command = dict(prepared.artifacts)[command_path]
        spellings = ("-wr", "--wr", "-wra", "--wra", "-wrap", "--wrap")
        for spelling in spellings:
            payloads = (
                ("direct-attached", f"{spelling}=archive_append_unused"),
                ("direct-separate", f"{spelling} archive_append_unused"),
                ("wl-attached", f"-Wl,{spelling}=archive_append_unused"),
                ("wl-comma", f"-Wl,{spelling},archive_append_unused"),
                ("wl-separate", f"-Wl,{spelling} archive_append_unused"),
                ("xlinker-attached", f"-Xlinker {spelling}=archive_append_unused"),
                (
                    "xlinker-separate",
                    f"-Xlinker {spelling} -Xlinker archive_append_unused",
                ),
            )
            for channel, payload in payloads:
                with self.subTest(spelling=spelling, channel=channel):
                    mutant_command = original_command.replace(
                        b" -o ",
                        b" " + payload.encode("ascii") + b" -o ",
                        1,
                    )
                    self.assertNotEqual(mutant_command, original_command)
                    self.assert_auxiliary_command_rejected(
                        prepared, mutant_command, "exact wrap set differs"
                    )

        for payload in (
            b"--WRAP=archive_append_unused",
            b"-WR=archive_append_unused",
            b"--wrap=",
            b"--wrap=bad/symbol",
            b"-Wl,--WRAP,archive_append_unused",
            b"-Wl,--wrap,",
            b"-Xlinker --wrap archive_append_unused",
        ):
            with self.subTest(malformed=payload):
                mutant_command = original_command.replace(
                    b" -o ", b" " + payload + b" -o ", 1
                )
                self.assert_auxiliary_command_rejected(
                    prepared, mutant_command, "wrap|GNU ld|malformed -Wl"
                )

    def test_frozen_gnu_ld_map_grammar_rejects_aliases_in_every_auxiliary_channel(
        self,
    ) -> None:
        source, build = self.make_project("source-map-grammar")
        prepared = self.prepare(source, build, self.root / "evidence-map-grammar")
        command_path = B._auxiliary_paths(B.ARCHIVE_APPEND_AUXILIARY_ID)[
            "link_command_path"
        ]
        original_command = dict(prepared.artifacts)[command_path]
        spellings = ("-M", "--M", "-Ma", "--Ma", "-Map", "--Map")
        expected_errors = {
            ("-Map", "wl-comma"): (
                "exactly one supported GNU ld map option required"
            ),
            ("-Map", "wl-separate"): (
                "-Wl,-Map requires one path in its option group"
            ),
        }
        for spelling in spellings:
            payloads = (
                ("direct-attached", f"{spelling}=shadow.map"),
                ("direct-separate", f"{spelling} shadow.map"),
                ("wl-attached", f"-Wl,{spelling}=shadow.map"),
                ("wl-comma", f"-Wl,{spelling},shadow.map"),
                ("wl-separate", f"-Wl,{spelling} -Wl,shadow.map"),
                ("xlinker-attached", f"-Xlinker {spelling}=shadow.map"),
                (
                    "xlinker-separate",
                    f"-Xlinker {spelling} -Xlinker shadow.map",
                ),
            )
            for channel, payload in payloads:
                with self.subTest(spelling=spelling, channel=channel):
                    mutant_command = original_command.replace(
                        b" -o ",
                        b" " + payload.encode("ascii") + b" -o ",
                        1,
                    )
                    self.assertNotEqual(mutant_command, original_command)
                    self.assert_auxiliary_command_rejected(
                        prepared,
                        mutant_command,
                        expected_errors.get((spelling, channel), "map aliases"),
                    )

        for channel, payload in (
            ("lowercase-emulation", b"-Wl,-m,elf_x86_64"),
            ("gold-map-compatibility", b"-Wl,--map-whole-files"),
        ):
            with self.subTest(unrelated=channel):
                command = original_command.replace(
                    b" -o ", b" " + payload + b" -o ", 1
                )
                self.assertNotEqual(command, original_command)
                self.assertIsNone(
                    B._validate_final_link_command(
                        command,
                        build_directory=str(build),
                        build_output_path="oai_memprof_archive_append",
                        build_map_path="oai_memprof_archive_append.map",
                        expected_wraps=(),
                        require_static_executable=True,
                        where="test case-sensitive GNU ld map grammar",
                    )
                )

    def test_unique_prefix_map_alias_redirect_is_rejected_before_publication(
        self,
    ) -> None:
        shadow_name = "oai_memprof_archive_append.shadow.map"
        source, build = self.make_project(
            "source-map-prefix-online",
            archive_append_link_options=(
                f"-Wl,--Ma,${{CMAKE_BINARY_DIR}}/{shadow_name}",
            ),
        )
        appender = build / "oai_memprof_archive_append"
        canonical_map = build / "oai_memprof_archive_append.map"
        self.run_host(
            (
                "/usr/bin/cc",
                "-static",
                "-no-pie",
                "-o",
                str(appender),
                f"-Wl,-Map,{canonical_map}",
                "-Wl,--cref",
                str(source / "archive_append.c"),
            )
        )
        authenticated_map = canonical_map.read_bytes()
        B._validate_link_map_output(
            authenticated_map,
            build_directory=str(build),
            build_output_path="oai_memprof_archive_append",
            where="test preexisting authenticated map",
        )
        shadow_map = build / shadow_name
        self.assertFalse(shadow_map.exists())

        # The map is a Ninja-untracked side output.  Leave the valid canonical
        # map in place while forcing Ninja to execute the later --Ma redirect.
        os.utime(appender, ns=(0, 0))
        evidence_root = self.root / "evidence-map-prefix-online"
        with self.assertRaisesRegex(B.BuildEvidenceError, "map aliases"):
            self.prepare(source, build, evidence_root)
        self.assertEqual(list(evidence_root.iterdir()), [])
        self.assertEqual(canonical_map.read_bytes(), authenticated_map)
        self.assertTrue(shadow_map.is_file())
        self.assertGreater(shadow_map.stat().st_size, 0)

    def test_response_file_indirection_is_rejected_online_and_offline(self) -> None:
        source, build = self.make_project(
            "source-wrap-response",
            archive_append_link_options=(
                "-Wl,@${CMAKE_BINARY_DIR}/archive-append-wrap.rsp",
            ),
        )
        (build / "archive-append-wrap.rsp").write_text(
            "--wrap=archive_append_unused\n",
            encoding="ascii",
        )
        evidence_root = self.root / "evidence-wrap-response"
        with self.assertRaisesRegex(
            B.BuildEvidenceError, "response-file operands are forbidden"
        ):
            self.prepare(source, build, evidence_root)
        self.assertEqual(list(evidence_root.iterdir()), [])

        source, build = self.make_project("source-wrap-response-offline")
        prepared = self.prepare(
            source, build, self.root / "evidence-wrap-response-offline"
        )
        original_evidence = B.COVERAGE.parse_canonical(prepared.evidence_bytes)
        command_path = B._auxiliary_paths(B.ARCHIVE_APPEND_AUXILIARY_ID)[
            "link_command_path"
        ]
        original_command = dict(prepared.artifacts)[command_path]
        for label, response_option in (
            ("driver", b"@/not-read/driver.rsp"),
            ("wl", b"-Wl,@/not-read/linker.rsp"),
            ("xlinker", b"-Xlinker @/not-read/linker.rsp"),
            ("driver-expansion", b"$RESPONSE"),
            ("wl-expansion", b"-Wl,@$RESPONSE"),
            ("xlinker-expansion", b"-Xlinker $RESPONSE"),
        ):
            with self.subTest(channel=label):
                evidence = copy.deepcopy(original_evidence)
                artifacts = dict(prepared.artifacts)
                mutant_command = original_command.replace(
                    b" -o ", b" " + response_option + b" -o ", 1
                )
                self.assertNotEqual(mutant_command, original_command)
                self.replace_artifact(
                    evidence, artifacts, command_path, mutant_command
                )
                with self.assertRaisesRegex(
                    B.BuildEvidenceError, "response-file operands|shell expansion"
                ):
                    B._derive(
                        evidence,
                        artifacts,
                        api_definition_sha256="1" * 64,
                        enforce_advertised_digest=False,
                    )

    def test_auxiliary_static_link_rejects_shell_output_map_and_loader_indirection(
        self,
    ) -> None:
        source, build = self.make_project("source-auxiliary-command")
        prepared = self.prepare(source, build, self.root / "evidence-auxiliary-command")
        command_path = B._auxiliary_paths(B.ARCHIVE_APPEND_AUXILIARY_ID)[
            "link_command_path"
        ]
        original_command = dict(prepared.artifacts)[command_path]
        for label, payload in (
            ("dollar", b"$OUT"),
            ("command-substitution", b"$(id)"),
            ("backtick", b"`id`"),
            ("quote", b"'quoted'"),
            ("escape", b"\\quoted"),
            ("comment", b"# comment"),
            ("glob", b"*"),
            ("tilde", b"~"),
            ("group", b"("),
            ("redirect", b">/tmp/out"),
        ):
            with self.subTest(shell=label):
                mutant_command = original_command.replace(
                    b" -o ", b" " + payload + b" -o ", 1
                )
                self.assert_auxiliary_command_rejected(
                    prepared, mutant_command, "shell expansion/quoting"
                )
        for label, payload in (
            ("semicolon", b"; :"),
            ("pipe", b"| /bin/true"),
            ("or", b"|| :"),
        ):
            with self.subTest(control=label):
                mutant_command = original_command.replace(
                    b" -o ", b" " + payload + b" -o ", 1
                )
                self.assert_auxiliary_command_rejected(
                    prepared, mutant_command, "final-link scaffold"
                )
        for label, payload in (
            ("response", b"@/not-read/auxiliary.rsp"),
            ("wl-response", b"-Wl,@/not-read/auxiliary.rsp"),
            ("xlinker-response", b"-Xlinker @/not-read/auxiliary.rsp"),
        ):
            with self.subTest(response=label):
                mutant_command = original_command.replace(
                    b" -o ", b" " + payload + b" -o ", 1
                )
                self.assert_auxiliary_command_rejected(
                    prepared, mutant_command, "response-file"
                )

        for label, command, expression in (
            ("joined-o", original_command.replace(b" -o ", b" -o", 1), "output"),
            (
                "long-output",
                original_command.replace(b" -o ", b" --output ", 1),
                "output",
            ),
            (
                "wl-output",
                original_command.replace(b" -o ", b" -Wl,-o,shadow -o ", 1),
                "output",
            ),
            (
                "map-equals",
                original_command.replace(b"-Wl,-Map,", b"-Wl,-Map=", 1),
                "map",
            ),
            (
                "xlinker-map",
                original_command.replace(
                    b" -o ", b" -Xlinker -Map -Xlinker shadow.map -o ", 1
                ),
                "map",
            ),
            ("missing-static", original_command.replace(b" -static ", b" ", 1), "-static"),
            ("missing-no-pie", original_command.replace(b" -no-pie ", b" ", 1), "-no-pie"),
            (
                "duplicate-static",
                original_command.replace(b" -static ", b" -static -static ", 1),
                "-static",
            ),
            ("pie", original_command.replace(b" -o ", b" -pie -o ", 1), "PIE/shared"),
            ("shared", original_command.replace(b" -o ", b" -shared -o ", 1), "PIE/shared"),
            (
                "dynamic-linker",
                original_command.replace(
                    b" -o ", b" -Wl,-dynamic-linker,/tmp/ld -o ", 1
                ),
                "dynamic-loader",
            ),
            (
                "rpath",
                original_command.replace(b" -o ", b" -Wl,-rpath,/tmp -o ", 1),
                "RPATH",
            ),
            (
                "specs",
                original_command.replace(b" -o ", b" -specs=/tmp/specs -o ", 1),
                "specs",
            ),
            (
                "plugin",
                original_command.replace(b" -o ", b" -fplugin=/tmp/plugin.so -o ", 1),
                "plugin",
            ),
            (
                "wl-plugin",
                original_command.replace(b" -o ", b" -Wl,-plugin,/tmp/plugin.so -o ", 1),
                "plugin",
            ),
            (
                "bdynamic",
                original_command.replace(b" -o ", b" -Wl,-Bdynamic -o ", 1),
                "dynamic-loader",
            ),
            (
                "explicit-shared-input",
                original_command.replace(b" -o ", b" -l:libmutable.so -o ", 1),
                "shared-object",
            ),
            (
                "active-runtime",
                original_command.replace(
                    b" -o ", b" /tmp/liboai_memprof_active_runtime.so.1 -o ", 1
                ),
                "active-runtime DSO",
            ),
            (
                "xlinker",
                original_command.replace(b" -o ", b" -Xlinker --as-needed -o ", 1),
                "Xlinker indirection",
            ),
        ):
            with self.subTest(link_option=label):
                self.assertNotEqual(command, original_command)
                self.assert_auxiliary_command_rejected(prepared, command, expression)

        evidence = B.COVERAGE.parse_canonical(prepared.evidence_bytes)
        auxiliary = evidence["auxiliary_executables"][0]
        unsafe_output = (
            ": && /usr/bin/cc -static -no-pie -o '"
            + evidence["build_directory"]
            + "/"
            + auxiliary["build_output_path"]
            + "' -Wl,-Map,"
            + evidence["build_directory"]
            + "/"
            + auxiliary["build_map_path"]
            + " -Wl,--cref && :\n"
        ).encode("utf-8")
        with self.assertRaisesRegex(B.BuildEvidenceError, "shell expansion/quoting"):
            B._link_command(
                unsafe_output,
                build_directory=evidence["build_directory"],
                build_output_path=auxiliary["build_output_path"],
                build_map_path=auxiliary["build_map_path"],
                expected_wraps=(),
                require_static_executable=True,
            )

    def test_dynamic_auxiliary_elf_metadata_is_rejected_online_and_offline(self) -> None:
        source, build = self.make_project("source-dynamic-auxiliary")
        self.run_host(("/usr/bin/ninja", "-C", str(build), "app"))
        dynamic_raw = (build / "app").read_bytes()
        self.assertIn(struct.unpack_from("<H", dynamic_raw, 16)[0], (2, 3))
        et_dyn_dynamic = self.set_elf_type(dynamic_raw, 3)
        et_exec_dynamic = self.set_elf_type(dynamic_raw, 2)
        variants = (
            ("dynamic-type", et_dyn_dynamic, "executable ELF program headers"),
            ("interpreter", et_exec_dynamic, "PT_INTERP"),
            (
                "dynamic-program-header",
                self.clear_program_header_type(et_exec_dynamic, 3),
                "PT_DYNAMIC",
            ),
            (
                "dynamic-section-tags",
                self.clear_program_header_type(
                    self.clear_program_header_type(et_exec_dynamic, 3), 2
                ),
                "SHT_DYNAMIC|DT_NEEDED|DT_SONAME|DT_RPATH|DT_RUNPATH",
            ),
        )
        prepared = self.prepare(source, build, self.root / "evidence-dynamic-offline")
        elf_path = B._auxiliary_paths(B.ARCHIVE_APPEND_AUXILIARY_ID)["elf_path"]
        for label, mutant, expression in variants:
            with self.subTest(offline=label):
                evidence = B.COVERAGE.parse_canonical(prepared.evidence_bytes)
                artifacts = dict(prepared.artifacts)
                self.replace_artifact(evidence, artifacts, elf_path, mutant)
                with self.assertRaisesRegex(B.BuildEvidenceError, expression):
                    B._derive(
                        evidence,
                        artifacts,
                        api_definition_sha256="1" * 64,
                        enforce_advertised_digest=False,
                    )

        source, build = self.make_project("source-dynamic-auxiliary-online")
        self.run_host(
            ("/usr/bin/ninja", "-C", str(build), B.ARCHIVE_APPEND_TARGET, "app")
        )
        (build / "oai_memprof_archive_append").write_bytes(
            self.set_elf_type((build / "app").read_bytes(), 3)
        )
        evidence_root = self.root / "evidence-dynamic-online"
        with self.assertRaisesRegex(
            B.BuildEvidenceError, "executable ELF program headers"
        ):
            self.prepare(source, build, evidence_root)
        self.assertEqual(list(evidence_root.iterdir()), [])

    def test_malformed_auxiliary_request_paths_use_project_error(self) -> None:
        evidence_root = self.root / "request-evidence"
        evidence_root.mkdir()
        base_request = {
            "api_definition_sha256": "1" * 64,
            "auxiliary_executables": [
                {
                    "auxiliary_id": B.ARCHIVE_APPEND_AUXILIARY_ID,
                    "elf_path": "/not-opened/appender",
                    "link_map_path": "/not-opened/appender.map",
                    "target": B.ARCHIVE_APPEND_TARGET,
                }
            ],
            "build_directory": "/not-opened/build",
            "evidence_root": str(evidence_root),
            "libc_path": "/not-opened/libc.so.6",
            "logical_elfs": [],
            "primary_logical_elf_id": "app",
            "repository": "/not-opened/repository",
        }
        for field in ("elf_path", "link_map_path"):
            for index, malformed in enumerate((None, 7, [], {})):
                with self.subTest(field=field, malformed=type(malformed).__name__):
                    request = copy.deepcopy(base_request)
                    request["auxiliary_executables"][0][field] = malformed
                    request_path = self.root / f"request-{field}-{index}.json"
                    request_path.write_bytes(B.COVERAGE.canonical_bytes(request))
                    with mock.patch.object(
                        B, "prepare_measured_build_evidence"
                    ) as prepare, self.assertRaisesRegex(
                        B.BuildEvidenceError, "nonempty path string required"
                    ):
                        B.main(("--request", str(request_path)))
                    prepare.assert_not_called()
                    self.assertEqual(list(evidence_root.iterdir()), [])

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
