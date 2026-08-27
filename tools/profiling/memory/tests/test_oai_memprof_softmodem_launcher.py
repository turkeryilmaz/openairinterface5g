#!/usr/bin/env python3
"""Deterministic regression tests for the authenticated softmodem launcher."""

from __future__ import annotations

import copy
import errno
import hashlib
import importlib.util
import os
import pathlib
import shutil
import stat
import sys
import tempfile
import threading
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[4]
LAUNCHER_PATH = ROOT / "tools/profiling/memory/oai_memprof_softmodem_launcher.py"


def load(name: str, path: pathlib.Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"module unavailable: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


LAUNCHER = load("_oai_memprof_softmodem_launcher_test", LAUNCHER_PATH)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


TRUSTED_RELEASE_AUTHORITY_FIXTURE_SHA256 = (
    "edde0b92154f7170b9565facfa77136bf30300bfe2f75e3d6e1e38dc6f848047"
)
_TRUSTED_RELEASE_AUTHORITY_SOURCES = (
    LAUNCHER.VERIFIER.accepted_trusted_release_source_bytes()
)
_TRUSTED_RELEASE_AUTHORITY_RAW = (
    LAUNCHER.VERIFIER.make_trusted_release_authority_bytes(
        commit="1" * 40,
        tree="2" * 40,
        source_bytes=_TRUSTED_RELEASE_AUTHORITY_SOURCES,
    )
)
_TRUSTED_RELEASE_AUTHORITY_SHA256 = sha(_TRUSTED_RELEASE_AUTHORITY_RAW)
if _TRUSTED_RELEASE_AUTHORITY_SHA256 != TRUSTED_RELEASE_AUTHORITY_FIXTURE_SHA256:
    raise AssertionError(
        "trusted-release authority fixture changed; review and explicitly "
        "update its literal pin "
        f"(observed {_TRUSTED_RELEASE_AUTHORITY_SHA256})"
    )


def always_selected() -> dict:
    return {
        "operator_id": 1,
        "predicates": [
            {
                "configuration_key": None,
                "expected_value": None,
                "predicate_id": 1,
            }
        ],
    }


class ExecveSentinel(Exception):
    pass


class SoftmodemLauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = pathlib.Path(
            tempfile.mkdtemp(prefix="oai-memprof-softmodem-launcher-test-")
        ).resolve()
        self.prepared_launches = []
        self.binary = self.root / "nr-softmodem"
        self.binary_bytes = b"bounded launcher fixture executable\n"
        self.binary.write_bytes(self.binary_bytes)
        self.binary.chmod(0o700)
        self.primary = {
            "admission_state_id": 1,
            "build_id": "01",
            "byte_count": len(self.binary_bytes),
            "logical_id": "nr_softmodem",
            "module_selection": always_selected(),
            "realloc_zero_policy_id": 1,
            "realloc_zero_semantic_oracle_sha256": (
                LAUNCHER.VERIFIER.ACCEPTED_MEMBER_SHA256[3]
            ),
            "role_ids": [1],
            "sha256": sha(self.binary_bytes),
            "soname": None,
            "symbol_origins": [
                {
                    "api_id": 3,
                    "classification_id": 1,
                    "origin_kind_id": 1,
                }
            ],
        }
        self.runtime = {
            "admission_state_id": 2,
            "logical_id": "oai_memprof_active_runtime",
            "module_selection": always_selected(),
            "realloc_zero_policy_id": None,
            "realloc_zero_semantic_oracle_sha256": None,
            "role_ids": [1, 2],
            "soname": "liboai_memprof_active_runtime.so.1",
            "symbol_origins": [],
        }
        self.build = {
            "architecture_id": 1,
            "build_identity": {
                "dirty": False,
                "primary_logical_elf_id": "nr_softmodem",
                "source_commit": "1" * 40,
                "source_tree": "2" * 40,
            },
            "entries": [self.primary, self.runtime],
        }
        self.authenticated = LAUNCHER.AuthenticatedBuild(
            b"canonical build coverage fixture",
            b"canonical build evidence fixture",
            self.build,
            self.primary,
            self.binary,
        )
        self.authority_sources = _TRUSTED_RELEASE_AUTHORITY_SOURCES
        self.authority_raw = _TRUSTED_RELEASE_AUTHORITY_RAW
        self.authority_sha256 = TRUSTED_RELEASE_AUTHORITY_FIXTURE_SHA256
        self.authority_path = self.root / "trusted-release-authority-v1.json"
        self.authority_path.write_bytes(self.authority_raw)
        self.authority_source_root = ROOT

    def tearDown(self) -> None:
        for prepared in reversed(self.prepared_launches):
            LAUNCHER.close_prepared_launch(prepared)
        shutil.rmtree(self.root)

    def request(self, label: str) -> tuple[dict, pathlib.Path, pathlib.Path]:
        archive = self.root / f"archive-{label}"
        bootstrap = self.root / f"bootstrap-{label}"
        archive.mkdir(mode=0o700)
        bootstrap.mkdir(mode=0o700)
        request = {
            "archive_directory": str(archive),
            "argv": [str(self.binary), "--launcher-unit"],
            "bootstrap_directory": str(bootstrap),
            "build_coverage_path": str(self.binary),
            "build_evidence_path": str(self.binary),
            "build_evidence_root": str(self.root),
            "flush_records": 4,
            "flush_us": 1_000,
            "max_threads": 16,
            "mode_id": 4,
            "process_generation": 7,
            "process_uuid": "ffeeddcc-bbaa-4988-8776-655443322110",
            "ring_records": 64,
            "role_id": 1,
            "run_id": f"launcher-{label}",
            "run_uuid": "00112233-4455-4677-8899-aabbccddeeff",
            "sample_threshold": 0,
            "scope_kind": 1,
            "seal_timeout_ns": 2_000_000_000,
            "selection_values": [{"key": "role_profile", "value": "unit"}],
            "table_entries": 64,
            "table_probes": 8,
            "working_directory": str(self.root),
        }
        return request, archive, bootstrap

    def prepare(
        self,
        label: str,
        *,
        request_mutator=None,
        authenticated=None,
    ):
        request, archive, bootstrap = self.request(label)
        if request_mutator is not None:
            request_mutator(request)
        raw = LAUNCHER.COVERAGE.canonical_bytes(request)
        with mock.patch.object(
            LAUNCHER,
            "_authenticate_build",
            return_value=self.authenticated if authenticated is None else authenticated,
        ) as authenticate:
            prepared = LAUNCHER.prepare_launch(
                raw,
                trusted_release_authority_path=self.authority_path,
                trusted_release_source_root=self.authority_source_root,
                trusted_release_authority_sha256=self.authority_sha256,
            )
        authenticate.assert_called_once()
        self.prepared_launches.append(prepared)
        return request, raw, archive, bootstrap, prepared

    def prepare_from_authenticated(self, request, authenticated):
        return LAUNCHER._prepare_from_authenticated(
            request,
            authenticated,
            trusted_release_authority_sha256=self.authority_sha256,
        )

    def test_public_prepare_binds_exact_configuration_opening_and_environment(self) -> None:
        build_before = copy.deepcopy(self.build)
        request, raw, archive, bootstrap, prepared = self.prepare("positive")

        self.assertEqual(raw, LAUNCHER.COVERAGE.canonical_bytes(request))
        self.assertEqual(self.build, build_before)
        self.assertEqual(prepared.binary_path, self.binary)
        self.assertEqual(prepared.argv, (str(self.binary), "--launcher-unit"))
        self.assertEqual(prepared.realloc_zero_policy_id, 1)
        self.assertEqual(prepared.build_coverage_sha256, sha(self.authenticated.build_raw))
        self.assertEqual(prepared.build_evidence_sha256, sha(self.authenticated.evidence_raw))
        self.assertEqual(
            prepared.trusted_release_authority_sha256, self.authority_sha256
        )
        self.assertEqual([path.name for path in archive.iterdir()], ["streams"])
        self.assertEqual(list((archive / "streams").iterdir()), [])

        configuration_raw = prepared.configuration_path.read_bytes()
        configuration = LAUNCHER.CONFIG.validate_effective_configuration_bytes(
            configuration_raw
        )
        self.assertEqual(configuration["output_directory"], str(archive))
        self.assertEqual(configuration["mode_id"], 4)
        self.assertEqual(
            configuration["selection_values"],
            [
                {
                    "key": "build_evidence_sha256",
                    "value": sha(self.authenticated.evidence_raw),
                },
                {"key": "role_profile", "value": "unit"},
                {
                    "key": "trusted_release_authority_sha256",
                    "value": self.authority_sha256,
                },
            ],
        )
        opening_raw = prepared.opening_path.read_bytes()
        opening = LAUNCHER.WIRE.decode_opening_header(opening_raw)
        self.assertEqual(opening.process_generation, 7)
        self.assertEqual(opening.configured_thread_capacity, 16)
        self.assertEqual(opening.role_kind, 1)
        self.assertEqual(opening.scope_kind, 1)
        self.assertEqual(opening.primary_binary_sha256, bytes.fromhex(self.primary["sha256"]))
        self.assertEqual(opening.configuration_instance_sha256, hashlib.sha256(configuration_raw).digest())
        self.assertEqual(prepared.configuration_sha256, sha(configuration_raw))
        self.assertEqual(prepared.opening_sha256, sha(opening_raw))
        self.assertEqual(
            tuple(name for name, _value in prepared.session_environment),
            LAUNCHER.SESSION_ENVIRONMENT_NAMES,
        )
        self.assertEqual(prepared.exec_environment, ())
        environment = dict(prepared.session_environment)
        self.assertEqual(
            environment["OAI_MEMPROF_SESSION_ARCHIVE_FD"],
            str(prepared.archive_directory_fd),
        )
        self.assertEqual(
            environment["OAI_MEMPROF_SESSION_BOOTSTRAP_FD"],
            str(prepared.bootstrap_directory_fd),
        )
        self.assertFalse(os.get_inheritable(prepared.archive_directory_fd))
        self.assertFalse(os.get_inheritable(prepared.bootstrap_directory_fd))
        self.assertFalse(os.get_inheritable(prepared.working_directory_fd))
        self.assertNotIn("OAI_MEMPROF_SESSION_ARCHIVE_DIRECTORY", environment)
        self.assertNotIn("OAI_MEMPROF_SESSION_CONFIGURATION_PATH", environment)
        self.assertNotIn("OAI_MEMPROF_SESSION_OPENING_PATH", environment)
        self.assertEqual(environment["OAI_MEMPROF_SESSION_TABLE_ENTRIES"], "0")
        self.assertEqual(environment["OAI_MEMPROF_SESSION_SAMPLE_SEED"], "0")
        self.assertEqual(environment["OAI_MEMPROF_SESSION_SAMPLE_THRESHOLD"], "0")
        self.assertEqual(environment["OAI_MEMPROF_SESSION_TABLE_PROBES"], "0")
        self.assertIsNone(configuration["sample_seed_hex"])
        self.assertEqual(
            (
                configuration["sample_seed_provenance_id"],
                configuration["sample_seed_status_id"],
            ),
            (20, 20),
        )
        self.assertEqual(configuration["sample_threshold"], 0)
        self.assertEqual(stat.S_IMODE(prepared.configuration_path.stat().st_mode), 0o640)
        self.assertEqual(stat.S_IMODE(prepared.opening_path.stat().st_mode), 0o640)

    def test_sampled_prepare_acquires_and_binds_exact_seed_and_controls(self) -> None:
        seed = bytes.fromhex("0123456789abcdef")
        with mock.patch.object(
            LAUNCHER.os, "getrandom", return_value=seed
        ) as getrandom:
            _request, _raw, archive, _bootstrap, prepared = self.prepare(
                "sampled-positive",
                request_mutator=lambda value: value.update(
                    mode_id=3, sample_threshold=(1 << 64) - 1
                ),
            )
        getrandom.assert_called_once_with(8)
        configuration = LAUNCHER.CONFIG.validate_effective_configuration_bytes(
            prepared.configuration_path.read_bytes()
        )
        self.assertEqual(configuration["output_directory"], str(archive))
        self.assertEqual(configuration["mode_id"], 3)
        self.assertEqual(configuration["sample_seed_hex"], seed.hex())
        self.assertEqual(
            (
                configuration["sample_seed_provenance_id"],
                configuration["sample_seed_status_id"],
            ),
            (1, 1),
        )
        self.assertEqual(configuration["sample_threshold"], (1 << 64) - 1)
        environment = dict(prepared.session_environment)
        self.assertEqual(environment["OAI_MEMPROF_SESSION_TABLE_ENTRIES"], "64")
        self.assertEqual(
            environment["OAI_MEMPROF_SESSION_SAMPLE_SEED"],
            str(int.from_bytes(seed, "big")),
        )
        self.assertEqual(
            environment["OAI_MEMPROF_SESSION_SAMPLE_THRESHOLD"],
            str((1 << 64) - 1),
        )
        self.assertEqual(environment["OAI_MEMPROF_SESSION_TABLE_PROBES"], "8")

        request, rejected_archive, rejected_bootstrap = self.request(
            "sampled-short-seed"
        )
        request.update(mode_id=3, sample_threshold=1)
        with mock.patch.object(LAUNCHER.os, "getrandom", return_value=b"short"):
            with self.assertRaisesRegex(
                LAUNCHER.SoftmodemLauncherError,
                "getrandom must return exactly eight bytes",
            ):
                self.prepare_from_authenticated(request, self.authenticated)
        self.assertEqual(list(rejected_archive.iterdir()), [])
        self.assertEqual(list(rejected_bootstrap.iterdir()), [])

    def test_public_request_is_exact_before_authentication(self) -> None:
        request, archive, bootstrap = self.request("extra")
        request["unexpected"] = 1
        raw = LAUNCHER.COVERAGE.canonical_bytes(request)
        with mock.patch.object(LAUNCHER, "_authenticate_build") as authenticate:
            with self.assertRaisesRegex(
                LAUNCHER.SoftmodemLauncherError, "request: exact members"
            ):
                LAUNCHER.prepare_launch(
                    raw,
                    trusted_release_authority_path=self.authority_path,
                    trusted_release_source_root=self.authority_source_root,
                    trusted_release_authority_sha256=self.authority_sha256,
                )
        authenticate.assert_not_called()
        self.assertEqual(list(archive.iterdir()), [])
        self.assertEqual(list(bootstrap.iterdir()), [])

    def test_publication_is_exclusive_and_does_not_leak_descriptors(self) -> None:
        directory = self.root / "publication"
        directory.mkdir(mode=0o700)
        before = len(list(pathlib.Path("/proc/self/fd").iterdir()))
        path = LAUNCHER._publish_once(directory, "fixture.bin", b"immutable")
        after = len(list(pathlib.Path("/proc/self/fd").iterdir()))
        self.assertEqual(after, before)
        self.assertEqual(path.read_bytes(), b"immutable")
        with self.assertRaises(FileExistsError):
            LAUNCHER._publish_once(directory, "fixture.bin", b"replacement")
        self.assertEqual(
            len(list(pathlib.Path("/proc/self/fd").iterdir())), before
        )
        self.assertEqual(path.read_bytes(), b"immutable")

    def test_policy_role_mode_and_launcher_owned_evidence_fail_closed(self) -> None:
        request, archive, bootstrap = self.request("ambiguous")
        other = copy.deepcopy(self.primary)
        other.update(logical_id="other_module", realloc_zero_policy_id=2)
        ambiguous_build = copy.deepcopy(self.build)
        ambiguous_build["entries"].append(other)
        ambiguous = LAUNCHER.AuthenticatedBuild(
            self.authenticated.build_raw,
            self.authenticated.evidence_raw,
            ambiguous_build,
            ambiguous_build["entries"][0],
            self.binary,
        )
        with self.assertRaisesRegex(
            LAUNCHER.SoftmodemLauncherError, "policy must resolve exactly once"
        ):
            self.prepare_from_authenticated(request, ambiguous)
        self.assertEqual(list(archive.iterdir()), [])
        self.assertEqual(list(bootstrap.iterdir()), [])

        request, archive, bootstrap = self.request("role")
        wrong_build = copy.deepcopy(self.build)
        wrong_build["entries"][0]["role_ids"] = [2]
        wrong_role = LAUNCHER.AuthenticatedBuild(
            self.authenticated.build_raw,
            self.authenticated.evidence_raw,
            wrong_build,
            wrong_build["entries"][0],
            self.binary,
        )
        with self.assertRaisesRegex(
            LAUNCHER.SoftmodemLauncherError, "primary role mismatch"
        ):
            self.prepare_from_authenticated(request, wrong_role)
        self.assertEqual(list(archive.iterdir()), [])
        self.assertEqual(list(bootstrap.iterdir()), [])

        for label, mutate, message in (
            (
                "sampled-zero-threshold",
                lambda value: value.update(mode_id=3),
                "mode 3 requires nonzero q",
            ),
            (
                "owned",
                lambda value: value["selection_values"].append(
                    {"key": "build_evidence_sha256", "value": "0" * 64}
                ),
                "build_evidence_sha256 is launcher-owned",
            ),
            (
                "owned-trusted-release",
                lambda value: value["selection_values"].append(
                    {
                        "key": "trusted_release_authority_sha256",
                        "value": "0" * 64,
                    }
                ),
                "trusted_release_authority_sha256 is launcher-owned",
            ),
        ):
            with self.subTest(label=label):
                request, archive, bootstrap = self.request(label)
                mutate(request)
                with self.assertRaisesRegex(LAUNCHER.SoftmodemLauncherError, message):
                    self.prepare_from_authenticated(request, self.authenticated)
                self.assertEqual(list(archive.iterdir()), [])
                self.assertEqual(list(bootstrap.iterdir()), [])

    def test_public_prepare_rejects_authority_before_bootstrap_publication(self) -> None:
        request, archive, bootstrap = self.request("authority-prepublication")
        raw = LAUNCHER.COVERAGE.canonical_bytes(request)
        with mock.patch.object(
            LAUNCHER, "_authenticate_build", return_value=self.authenticated
        ):
            with self.assertRaisesRegex(
                LAUNCHER.SoftmodemLauncherError,
                "authority digest differs from the external pin",
            ):
                LAUNCHER.prepare_launch(
                    raw,
                    trusted_release_authority_path=self.authority_path,
                    trusted_release_source_root=self.authority_source_root,
                    trusted_release_authority_sha256="0" * 64,
                )
        self.assertEqual(list(archive.iterdir()), [])
        self.assertEqual(list(bootstrap.iterdir()), [])

    def test_reallocarray_only_build_resolves_the_authoritative_policy(self) -> None:
        reallocarray_build = copy.deepcopy(self.build)
        reallocarray_build["entries"][0]["symbol_origins"] = [
            {
                "api_id": 5,
                "classification_id": 1,
                "origin_kind_id": 1,
            }
        ]
        reallocarray = LAUNCHER.AuthenticatedBuild(
            self.authenticated.build_raw,
            self.authenticated.evidence_raw,
            reallocarray_build,
            reallocarray_build["entries"][0],
            self.binary,
        )
        _request, _raw, _archive, _bootstrap, prepared = self.prepare(
            "reallocarray-only", authenticated=reallocarray
        )
        self.assertEqual(prepared.realloc_zero_policy_id, 1)

    def test_build_artifact_preflight_rejects_before_opening_bad_or_overbudget_entries(self) -> None:
        def evidence(entries: list[dict]) -> bytes:
            return LAUNCHER.COVERAGE.canonical_bytes({"entries": entries})

        valid = {
            "bytes": 1,
            "path": "input/build-evidence/first",
            "sha256": "0" * 64,
        }
        invalid_cases = (
            (
                "row-shape",
                [{"bytes": 1, "path": valid["path"]}],
                "exact members",
            ),
            (
                "declared-bytes",
                [{**valid, "bytes": True}],
                "u64 integer required",
            ),
            (
                "digest",
                [{**valid, "sha256": "A" * 64}],
                "lowercase SHA-256 hex required",
            ),
            (
                "duplicate",
                [valid, {**valid, "bytes": 2}],
                "duplicate path",
            ),
        )
        for label, entries, message in invalid_cases:
            with self.subTest(label=label):
                with mock.patch.object(LAUNCHER.COMPOSER, "_read_frozen") as read:
                    with self.assertRaisesRegex(LAUNCHER.SoftmodemLauncherError, message):
                        LAUNCHER._read_build_artifacts(evidence(entries), self.root)
                read.assert_not_called()

        over_budget = [
            valid,
            {
                "bytes": 3,
                "path": "input/build-evidence/second",
                "sha256": "1" * 64,
            },
        ]
        with mock.patch.object(LAUNCHER, "MAX_BUILD_ARTIFACT_TOTAL_BYTES", 3):
            with mock.patch.object(LAUNCHER.COMPOSER, "_read_frozen") as read:
                with self.assertRaisesRegex(
                    LAUNCHER.SoftmodemLauncherError,
                    "aggregate declared artifact bytes exceeded",
                ):
                    LAUNCHER._read_build_artifacts(evidence(over_budget), self.root)
            read.assert_not_called()

        with mock.patch.object(LAUNCHER, "MAX_BUILD_ARTIFACT_ENTRIES", 1):
            with mock.patch.object(LAUNCHER.COMPOSER, "_read_frozen") as read:
                with self.assertRaisesRegex(
                    LAUNCHER.SoftmodemLauncherError,
                    "artifact-entry limit exceeded",
                ):
                    LAUNCHER._read_build_artifacts(evidence(over_budget), self.root)
            read.assert_not_called()

    def test_request_exec_environment_admits_only_secure_uhd_images_directory(self) -> None:
        images_directory = self.root / "uhd-images"
        images_directory.mkdir(mode=0o750)
        request, raw, _archive, _bootstrap, prepared = self.prepare(
            "uhd-images",
            request_mutator=lambda value: value.update(
                exec_environment=[
                    {"name": "UHD_IMAGES_DIR", "value": str(images_directory)}
                ]
            ),
        )
        self.assertEqual(
            request["exec_environment"],
            [{"name": "UHD_IMAGES_DIR", "value": str(images_directory)}],
        )
        self.assertEqual(
            LAUNCHER._request(raw)["exec_environment"],
            request["exec_environment"],
        )
        self.assertEqual(
            prepared.exec_environment,
            (("UHD_IMAGES_DIR", str(images_directory)),),
        )
        self.assertNotIn("UHD_IMAGES_DIR", dict(prepared.session_environment))

        observed = {}

        def intercepted_execve(_executable, _argv, environment):
            observed["environment"] = environment
            raise ExecveSentinel

        stale = {
            "UHD_IMAGES_DIR": "/tmp/ambient-uhd-images",
            "LD_PRELOAD": "/tmp/ambient-preload.so",
            "UNRELATED_LAUNCHER_TEST": "ambient",
        }
        with mock.patch.dict(os.environ, stale, clear=True):
            with mock.patch.object(
                LAUNCHER.os, "execve", side_effect=intercepted_execve
            ):
                with self.assertRaises(ExecveSentinel):
                    LAUNCHER.execute(prepared)
        self.assertEqual(
            observed["environment"],
            dict(prepared.session_environment + prepared.exec_environment),
        )
        self.assertEqual(observed["environment"]["UHD_IMAGES_DIR"], str(images_directory))
        self.assertNotIn("LD_PRELOAD", observed["environment"])
        self.assertNotIn("UNRELATED_LAUNCHER_TEST", observed["environment"])

        _request, _raw, _archive, _bootstrap, empty_prepared = self.prepare(
            "empty-exec-environment",
            request_mutator=lambda value: value.update(exec_environment=[]),
        )
        self.assertEqual(empty_prepared.exec_environment, ())

    def test_request_exec_environment_rejects_invalid_controls_before_publication(self) -> None:
        images_directory = self.root / "uhd-images-valid"
        images_directory.mkdir(mode=0o750)
        symlink_directory = self.root / "uhd-images-link"
        symlink_directory.symlink_to(images_directory, target_is_directory=True)
        writable_directory = self.root / "uhd-images-writable"
        writable_directory.mkdir(mode=0o770)
        writable_directory.chmod(0o770)
        regular_file = self.root / "uhd-images-file"
        regular_file.write_bytes(b"not an image directory\n")
        regular_file.chmod(0o600)
        noncanonical_directory = str(
            images_directory.parent / images_directory.name / ".." / images_directory.name
        )
        cases = (
            ("container", {}, "array required"),
            ("explicit-null", None, "array required"),
            (
                "row-shape",
                [{"name": "UHD_IMAGES_DIR", "unexpected": str(images_directory)}],
                "exact members",
            ),
            (
                "name-type",
                [{"name": [], "value": str(images_directory)}],
                "string required",
            ),
            (
                "name-nul",
                [{"name": "UHD_IMAGES_DIR\x00", "value": str(images_directory)}],
                "NUL-free no-equals name required",
            ),
            (
                "name-equals",
                [{"name": "UHD_IMAGES_DIR=bad", "value": str(images_directory)}],
                "NUL-free no-equals name required",
            ),
            (
                "unknown-name",
                [{"name": "PATH", "value": str(images_directory)}],
                "only UHD_IMAGES_DIR",
            ),
            (
                "profiler-collision",
                [{"name": "OAI_MEMPROF_SESSION_ENABLE", "value": "1"}],
                "profiler control collision",
            ),
            (
                "loader-collision",
                [{"name": "LD_PRELOAD", "value": "/tmp/loader.so"}],
                "loader-dangerous control collision",
            ),
            (
                "duplicate",
                [
                    {"name": "UHD_IMAGES_DIR", "value": str(images_directory)},
                    {"name": "UHD_IMAGES_DIR", "value": str(images_directory)},
                ],
                "entry limit exceeded",
            ),
            (
                "value-type",
                [{"name": "UHD_IMAGES_DIR", "value": []}],
                "string required",
            ),
            (
                "value-nul",
                [{"name": "UHD_IMAGES_DIR", "value": "/images\x00dir"}],
                "bounded NUL-free path required",
            ),
            (
                "relative",
                [{"name": "UHD_IMAGES_DIR", "value": "relative-images"}],
                "absolute path required",
            ),
            (
                "missing",
                [
                    {
                        "name": "UHD_IMAGES_DIR",
                        "value": str(self.root / "missing-images"),
                    }
                ],
                "unavailable directory",
            ),
            (
                "regular-file",
                [{"name": "UHD_IMAGES_DIR", "value": str(regular_file)}],
                "real directory required",
            ),
            (
                "symlink",
                [{"name": "UHD_IMAGES_DIR", "value": str(symlink_directory)}],
                "canonical resolved path required",
            ),
            (
                "noncanonical",
                [{"name": "UHD_IMAGES_DIR", "value": noncanonical_directory}],
                "canonical resolved path required",
            ),
            (
                "writable",
                [{"name": "UHD_IMAGES_DIR", "value": str(writable_directory)}],
                "group/other write forbidden",
            ),
            (
                "bounded",
                [
                    {
                        "name": "UHD_IMAGES_DIR",
                        "value": "/" + "a" * LAUNCHER.MAX_UHD_IMAGES_DIR_BYTES,
                    }
                ],
                "bounded NUL-free path required",
            ),
        )
        for label, exec_environment, message in cases:
            with self.subTest(label=label):
                request, archive, bootstrap = self.request(f"exec-environment-{label}")
                request["exec_environment"] = exec_environment
                with self.assertRaisesRegex(LAUNCHER.SoftmodemLauncherError, message):
                    self.prepare_from_authenticated(request, self.authenticated)
                self.assertEqual(list(archive.iterdir()), [])
                self.assertEqual(list(bootstrap.iterdir()), [])

    def test_execute_rejects_tampered_exec_environment_before_execve(self) -> None:
        writable_directory = self.root / "tampered-uhd-images"
        writable_directory.mkdir(mode=0o770)
        writable_directory.chmod(0o770)
        cases = (
            (
                "outer-list",
                [("UHD_IMAGES_DIR", str(writable_directory))],
                "exact optional environment tuple required",
            ),
            (
                "malformed-row",
                (("UHD_IMAGES_DIR",),),
                "exact optional environment row required",
            ),
            (
                "non-string-name",
                ((1, str(writable_directory)),),
                "name: string required",
            ),
            (
                "non-string-value",
                (("UHD_IMAGES_DIR", 1),),
                "value: string required",
            ),
            (
                "extra-rows",
                (
                    ("UHD_IMAGES_DIR", str(writable_directory)),
                    ("UHD_IMAGES_DIR", str(writable_directory)),
                ),
                "optional environment entry limit exceeded",
            ),
            (
                "unexpected-name",
                (("LD_PRELOAD", "/tmp/loader.so"),),
                "loader-dangerous control collision",
            ),
            (
                "writable-directory",
                (("UHD_IMAGES_DIR", str(writable_directory)),),
                "group/other write forbidden",
            ),
        )
        for label, exec_environment, message in cases:
            with self.subTest(label=label):
                _request, _raw, _archive, _bootstrap, prepared = self.prepare(
                    f"tampered-{label}"
                )
                prepared.exec_environment = exec_environment
                with mock.patch.object(LAUNCHER.os, "execve") as execve:
                    with self.assertRaisesRegex(LAUNCHER.SoftmodemLauncherError, message):
                        LAUNCHER.execute(prepared)
                execve.assert_not_called()

    def test_execute_uses_authenticated_fd_and_strict_owned_environment(self) -> None:
        _request, _raw, _archive, _bootstrap, prepared = self.prepare("exec")
        observed = {}

        def intercepted_execve(executable, argv, environment):
            observed["binary_identity"] = os.fstat(executable)
            observed["archive_inheritable"] = os.get_inheritable(
                prepared.archive_directory_fd
            )
            observed["bootstrap_inheritable"] = os.get_inheritable(
                prepared.bootstrap_directory_fd
            )
            observed["working_inheritable"] = os.get_inheritable(
                prepared.working_directory_fd
            )
            raise ExecveSentinel

        stale = {name: "stale" for name in LAUNCHER.SESSION_ENVIRONMENT_NAMES}
        stale.update(
            {
                "UHD_IMAGES_DIR": "/tmp/ambient-uhd-images",
                "LD_PRELOAD": "/tmp/ambient-preload.so",
                "LD_AUDIT": "/tmp/ambient-audit.so",
                "LD_LIBRARY_PATH": "/tmp/ambient-library",
                "UNRELATED_LAUNCHER_TEST": "ambient",
            }
        )
        with mock.patch.dict(os.environ, stale, clear=True):
            with mock.patch.object(
                LAUNCHER.os, "execve", side_effect=intercepted_execve
            ) as execve:
                with self.assertRaises(ExecveSentinel):
                    LAUNCHER.execute(prepared)
        executable, argv, environment = execve.call_args.args
        self.assertIsInstance(executable, int)
        self.assertEqual(argv, prepared.argv)
        self.assertEqual(environment, dict(prepared.session_environment))
        self.assertEqual(
            (observed["binary_identity"].st_dev, observed["binary_identity"].st_ino),
            (self.binary.stat().st_dev, self.binary.stat().st_ino),
        )
        self.assertTrue(observed["archive_inheritable"])
        self.assertTrue(observed["bootstrap_inheritable"])
        self.assertFalse(observed["working_inheritable"])
        for name in (
            "UHD_IMAGES_DIR",
            "LD_PRELOAD",
            "LD_AUDIT",
            "LD_LIBRARY_PATH",
            "UNRELATED_LAUNCHER_TEST",
        ):
            self.assertNotIn(name, environment)

    def test_execute_closes_unrelated_inheritable_fds_and_restores_them(self) -> None:
        _request, _raw, _archive, _bootstrap, prepared = self.prepare(
            "descriptor-sweep"
        )
        unrelated_read, unrelated_write = os.pipe()
        os.set_inheritable(unrelated_read, True)
        try:
            def intercepted_execve(executable, _argv, _environment):
                self.assertTrue(os.get_inheritable(prepared.archive_directory_fd))
                self.assertTrue(os.get_inheritable(prepared.bootstrap_directory_fd))
                self.assertFalse(os.get_inheritable(prepared.working_directory_fd))
                self.assertFalse(os.get_inheritable(executable))
                self.assertFalse(os.get_inheritable(unrelated_read))
                self.assertFalse(os.get_inheritable(unrelated_write))
                raise OSError(errno.EIO, "forced execve failure")

            with mock.patch.object(
                LAUNCHER.os, "execve", side_effect=intercepted_execve
            ):
                with self.assertRaisesRegex(OSError, "forced execve failure"):
                    LAUNCHER.execute(prepared)
            self.assertTrue(os.get_inheritable(unrelated_read))
            self.assertFalse(os.get_inheritable(unrelated_write))
        finally:
            os.close(unrelated_read)
            os.close(unrelated_write)

    def test_execute_restores_descriptor_state_after_partial_handoff_setup(self) -> None:
        _request, _raw, _archive, _bootstrap, prepared = self.prepare(
            "descriptor-setup-failure"
        )
        unrelated_read, unrelated_write = os.pipe()
        os.set_inheritable(unrelated_read, True)
        bootstrap_descriptor = prepared.bootstrap_directory_fd
        original_set_inheritable = os.set_inheritable
        injected = False

        def interrupted_set_inheritable(descriptor, inheritable):
            nonlocal injected
            if (
                descriptor == bootstrap_descriptor
                and inheritable
                and not injected
            ):
                injected = True
                raise OSError(errno.EIO, "forced root handoff failure")
            return original_set_inheritable(descriptor, inheritable)

        try:
            with mock.patch.object(
                LAUNCHER.os,
                "set_inheritable",
                side_effect=interrupted_set_inheritable,
            ):
                with mock.patch.object(LAUNCHER.os, "execve") as execve:
                    with self.assertRaisesRegex(
                        LAUNCHER.SoftmodemLauncherError,
                        "descriptor inheritance mutation failed",
                    ):
                        LAUNCHER.execute(prepared)
            self.assertTrue(injected)
            execve.assert_not_called()
            self.assertTrue(os.get_inheritable(unrelated_read))
            self.assertFalse(os.get_inheritable(unrelated_write))
        finally:
            os.close(unrelated_read)
            os.close(unrelated_write)

    def test_execute_rejects_multithreaded_descriptor_handoff(self) -> None:
        _request, _raw, _archive, _bootstrap, prepared = self.prepare(
            "multithreaded-exec"
        )
        ready = threading.Event()
        release = threading.Event()

        def hold_second_thread() -> None:
            ready.set()
            release.wait()

        thread = threading.Thread(target=hold_second_thread)
        thread.start()
        try:
            self.assertTrue(ready.wait(timeout=5))
            with mock.patch.object(LAUNCHER.os, "execve") as execve:
                with self.assertRaisesRegex(
                    LAUNCHER.SoftmodemLauncherError,
                    "descriptor exec requires one process thread",
                ):
                    LAUNCHER.execute(prepared)
            execve.assert_not_called()
        finally:
            release.set()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())

    def test_execute_rejects_replaced_primary_file_before_exec(self) -> None:
        _request, _raw, _archive, _bootstrap, prepared = self.prepare(
            "replaced-primary"
        )
        replacement = self.root / "different-primary"
        replacement.write_bytes(b"x" * len(self.binary_bytes))
        replacement.chmod(0o700)
        os.replace(replacement, self.binary)
        with mock.patch.object(LAUNCHER.os, "execve") as execve:
            with self.assertRaisesRegex(
                LAUNCHER.SoftmodemLauncherError,
                "bytes differ from measured primary executable",
            ):
                LAUNCHER.execute(prepared)
        execve.assert_not_called()

    def test_execute_rejects_primary_symlink_before_exec(self) -> None:
        _request, _raw, _archive, _bootstrap, prepared = self.prepare(
            "symlink-primary"
        )
        target = self.root / "symlink-target"
        target.write_bytes(self.binary_bytes)
        target.chmod(0o700)
        self.binary.unlink()
        self.binary.symlink_to(target)
        with mock.patch.object(LAUNCHER.os, "execve") as execve:
            with self.assertRaisesRegex(
                LAUNCHER.SoftmodemLauncherError,
                "final executable open or verification failed",
            ):
                LAUNCHER.execute(prepared)
        execve.assert_not_called()

    def test_execute_rejects_hard_linked_primary_before_exec(self) -> None:
        _request, _raw, _archive, _bootstrap, prepared = self.prepare(
            "hard-linked-primary"
        )
        image = self.root / "hard-link-image"
        image.write_bytes(self.binary_bytes)
        image.chmod(0o700)
        self.binary.unlink()
        os.link(image, self.binary)
        with mock.patch.object(LAUNCHER.os, "execve") as execve:
            with self.assertRaisesRegex(
                LAUNCHER.SoftmodemLauncherError,
                "final executable identity mismatch",
            ):
                LAUNCHER.execute(prepared)
        execve.assert_not_called()

    def test_execute_hands_off_pinned_roots_and_fd_bound_working_directory(self) -> None:
        working = self.root / "working"
        working.mkdir(mode=0o700)
        _request, _raw, archive, bootstrap, prepared = self.prepare(
            "pinned-roots",
            request_mutator=lambda value: value.update(working_directory=str(working)),
        )
        archive_identity = os.fstat(prepared.archive_directory_fd)
        bootstrap_identity = os.fstat(prepared.bootstrap_directory_fd)
        working_identity = os.fstat(prepared.working_directory_fd)
        held_archive = self.root / "archive-held"
        held_bootstrap = self.root / "bootstrap-held"
        held_working = self.root / "working-held"
        archive.rename(held_archive)
        bootstrap.rename(held_bootstrap)
        working.rename(held_working)
        archive.mkdir(mode=0o700)
        bootstrap.mkdir(mode=0o700)
        working.mkdir(mode=0o700)

        def intercepted_execve(_executable, _argv, environment):
            archive_fd = int(environment["OAI_MEMPROF_SESSION_ARCHIVE_FD"])
            bootstrap_fd = int(environment["OAI_MEMPROF_SESSION_BOOTSTRAP_FD"])
            self.assertEqual(
                (os.fstat(archive_fd).st_dev, os.fstat(archive_fd).st_ino),
                (archive_identity.st_dev, archive_identity.st_ino),
            )
            self.assertEqual(
                (os.fstat(bootstrap_fd).st_dev, os.fstat(bootstrap_fd).st_ino),
                (bootstrap_identity.st_dev, bootstrap_identity.st_ino),
            )
            self.assertEqual(
                (os.stat(".").st_dev, os.stat(".").st_ino),
                (working_identity.st_dev, working_identity.st_ino),
            )
            self.assertNotIn("OAI_MEMPROF_SESSION_ARCHIVE_DIRECTORY", environment)
            self.assertNotIn("OAI_MEMPROF_SESSION_CONFIGURATION_PATH", environment)
            self.assertNotIn("OAI_MEMPROF_SESSION_OPENING_PATH", environment)
            raise ExecveSentinel

        with mock.patch.object(
            LAUNCHER.os, "execve", side_effect=intercepted_execve
        ):
            with self.assertRaises(ExecveSentinel):
                LAUNCHER.execute(prepared)

    def test_execute_fails_closed_without_descriptor_exec_support(self) -> None:
        _request, _raw, _archive, _bootstrap, prepared = self.prepare(
            "no-fd-exec"
        )
        with mock.patch.object(LAUNCHER, "EXECVE_SUPPORTS_FD", False):
            with mock.patch.object(LAUNCHER.os, "execve") as execve:
                with self.assertRaisesRegex(
                    LAUNCHER.SoftmodemLauncherError,
                    "descriptor exec unsupported",
                ):
                    LAUNCHER.execute(prepared)
        execve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
