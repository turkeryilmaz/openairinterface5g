#!/usr/bin/env python3
"""Isolated-Git tests for trusted-release authority generation.

These tests create only temporary repositories.  They never add, remove, or
change objects in the checkout that provides this test and its generator.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


HERE = pathlib.Path(__file__).resolve().parent
MEMORY_ROOT = HERE.parent
REPOSITORY_ROOT = MEMORY_ROOT.parents[2]


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module specification unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


G = _load_module(
    "oai_memprof_trusted_release_authority_tested",
    MEMORY_ROOT / "oai_memprof_trusted_release_authority.py",
)
V = _load_module(
    "archive_semantic_verifier_v1_authority_tested",
    MEMORY_ROOT / "catalog_v1/integration/archive_semantic_verifier_v1.py",
)


class IsolatedGitRepository:
    """One clean temporary repository with exactly the 15 required files."""

    def __init__(self, testcase: unittest.TestCase) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="oai-memprof-authority-")
        testcase.addCleanup(self._temporary.cleanup)
        self.base = pathlib.Path(self._temporary.name)
        self.root = self.base / "checkout"
        self.root.mkdir()
        self.source_bytes: dict[str, bytes] = {}
        for authority_path, repository_path in G.validate_fixed_source_mapping(
            G.TRUSTED_RELEASE_SOURCE_PATHS
        ):
            raw = (REPOSITORY_ROOT / repository_path).read_bytes()
            destination = self.root / repository_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
            self.source_bytes[authority_path] = raw
        self.run("git", "init", "-q")
        self.run("git", "config", "user.email", "authority-test@example.invalid")
        self.run("git", "config", "user.name", "Authority Test")
        self.commit_all("initial fixed source population")

    def run(self, *arguments: str) -> str:
        completed = subprocess.run(
            arguments,
            cwd=self.root,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return completed.stdout

    def commit_all(self, message: str) -> None:
        self.run("git", "add", "--all")
        self.run("git", "commit", "-q", "-m", message)

    @property
    def commit(self) -> str:
        return self.run("git", "rev-parse", "HEAD").strip()

    @property
    def tree(self) -> str:
        return self.run("git", "rev-parse", "HEAD^{tree}").strip()

    def fresh_output(self, name: str) -> pathlib.Path:
        output = self.base / name
        output.mkdir()
        return output

    def source_path(self, authority_path: str) -> pathlib.Path:
        return self.root / G.TRUSTED_RELEASE_SOURCE_PATHS[authority_path]


class TrustedReleaseAuthorityGeneratorTests(unittest.TestCase):
    maxDiff = None

    def _generate(
        self, fixture: IsolatedGitRepository, output: pathlib.Path | None = None
    ):
        return G.generate_trusted_release_authority(
            repository_root=fixture.root,
            commit=fixture.commit,
            tree=fixture.tree,
            output_directory=output or fixture.fresh_output("output"),
        )

    def test_valid_deterministic_authority_is_verifier_compatible(self) -> None:
        fixture = IsolatedGitRepository(self)
        first_output = fixture.fresh_output("first")
        second_output = fixture.fresh_output("second")
        first = self._generate(fixture, first_output)
        second = self._generate(fixture, second_output)

        self.assertEqual(first.authority_bytes, second.authority_bytes)
        self.assertEqual(first.authority_sha256, second.authority_sha256)
        self.assertEqual(
            first.authority_sha256,
            hashlib.sha256(first.authority_bytes).hexdigest(),
        )
        self.assertEqual(first.authority_path.read_bytes(), first.authority_bytes)
        self.assertEqual(
            first.receipt_path.read_bytes(), second.receipt_path.read_bytes()
        )
        authority = json.loads(first.authority_bytes)
        self.assertEqual(authority["schema"], V.TRUSTED_RELEASE_AUTHORITY_SCHEMA)
        self.assertEqual(authority["git"], {
            "clean": True,
            "commit": fixture.commit,
            "tree": fixture.tree,
        })
        self.assertEqual(
            [row["path"] for row in authority["sources"]],
            sorted(V.TRUSTED_RELEASE_SOURCE_PATHS),
        )
        self.assertEqual(G.TRUSTED_RELEASE_SOURCE_PATHS, V.TRUSTED_RELEASE_SOURCE_PATHS)
        projected = V.validate_trusted_release_authority(
            first.authority_bytes,
            first.authority_sha256,
            fixture.source_bytes,
            build={
                "build_identity": {
                    "dirty": False,
                    "source_commit": fixture.commit,
                    "source_tree": fixture.tree,
                }
            },
        )
        self.assertEqual(
            set(projected),
            {V.TRUSTED_RELEASE_AUTHORITY_PATH, *V.TRUSTED_RELEASE_SOURCE_PATHS},
        )

    def test_receipt_and_generator_are_not_a_sixteenth_authority_row(self) -> None:
        fixture = IsolatedGitRepository(self)
        result = self._generate(fixture)
        authority = json.loads(result.authority_bytes)
        receipt = json.loads(result.receipt_path.read_bytes())
        authority_paths = [row["path"] for row in authority["sources"]]
        self.assertEqual(len(authority_paths), 15)
        self.assertEqual(set(authority_paths), set(G.TRUSTED_RELEASE_SOURCE_PATHS))
        self.assertNotIn(G.RECEIPT_FILENAME, authority_paths)
        self.assertNotIn("oai_memprof_trusted_release_authority.py", authority_paths)
        self.assertTrue(receipt["not_authority_input"])
        self.assertEqual(receipt["authority"]["sha256"], result.authority_sha256)
        self.assertEqual(len(receipt["source_mapping"]), 15)
        self.assertEqual(len(receipt["git_blobs"]), 15)
        self.assertEqual(
            receipt["generator"]["status"],
            "unavailable_outside_or_absent_from_pinned_tree",
        )

    def test_dirty_tracked_and_untracked_checkout_reject_before_publication(self) -> None:
        for kind in ("tracked", "untracked"):
            with self.subTest(kind=kind):
                fixture = IsolatedGitRepository(self)
                if kind == "tracked":
                    fixture.source_path(
                        "definition/oai-memprof-container-wire-v1.py"
                    ).write_bytes(b"changed tracked source\n")
                else:
                    (fixture.root / "untracked-controller-artifact").write_text(
                        "not allowed\n", encoding="utf-8"
                    )
                output = fixture.fresh_output(f"{kind}-output")
                with self.assertRaisesRegex(
                    G.TrustedReleaseAuthorityError, "status is not empty"
                ):
                    self._generate(fixture, output)
                self.assertEqual(list(output.iterdir()), [])

    def test_repository_and_output_symlink_components_reject_without_writing(self) -> None:
        fixture = IsolatedGitRepository(self)
        alias = fixture.base / "alias"
        alias.symlink_to(fixture.base, target_is_directory=True)
        repository_final = fixture.base / "repository-final-link"
        repository_final.symlink_to(fixture.root, target_is_directory=True)
        protected_output = fixture.fresh_output("protected-output")
        output_final = fixture.base / "output-final-link"
        output_final.symlink_to(protected_output, target_is_directory=True)
        cases = (
            ("repository ancestor", alias / "checkout", protected_output),
            ("repository final", repository_final, protected_output),
            ("output ancestor", fixture.root, alias / "protected-output"),
            ("output final", fixture.root, output_final),
        )
        for label, repository_root, output in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    G.TrustedReleaseAuthorityError, "symlinked path component"
                ):
                    G.generate_trusted_release_authority(
                        repository_root=repository_root,
                        commit=fixture.commit,
                        tree=fixture.tree,
                        output_directory=output,
                    )
                self.assertEqual(list(protected_output.iterdir()), [])

    def test_wrong_commit_or_tree_pin_rejects(self) -> None:
        fixture = IsolatedGitRepository(self)
        with self.assertRaisesRegex(
            G.TrustedReleaseAuthorityError, "HEAD differs"
        ):
            G.generate_trusted_release_authority(
                repository_root=fixture.root,
                commit="0" * 40,
                tree=fixture.tree,
                output_directory=fixture.fresh_output("wrong-commit"),
            )
        with self.assertRaisesRegex(
            G.TrustedReleaseAuthorityError, "commit tree differs"
        ):
            G.generate_trusted_release_authority(
                repository_root=fixture.root,
                commit=fixture.commit,
                tree="0" * 40,
                output_directory=fixture.fresh_output("wrong-tree"),
            )

    def test_fixed_mapping_rejects_missing_extra_and_malformed_rows(self) -> None:
        missing = dict(G.TRUSTED_RELEASE_SOURCE_PATHS)
        missing.pop("definition/oai-memprof-container-wire-v1.py")
        extra = dict(G.TRUSTED_RELEASE_SOURCE_PATHS)
        extra["definition/extra.py"] = "tools/profiling/memory/extra.py"
        malformed = dict(G.TRUSTED_RELEASE_SOURCE_PATHS)
        malformed["definition/bad\\path.py"] = malformed.pop(
            "definition/oai-memprof-container-wire-v1.py"
        )
        for label, mapping in (
            ("missing", missing),
            ("extra", extra),
            ("malformed", malformed),
        ):
            with self.subTest(label=label):
                with self.assertRaises(G.TrustedReleaseAuthorityError):
                    G.validate_fixed_source_mapping(mapping)

    def test_git_blob_oid_width_is_exact_and_rejects_before_cat_file(self) -> None:
        repository_path = "tools/profiling/memory/source.py"
        authority_path = "definition/source.py"
        for width in (39, 41, 63, 65):
            with self.subTest(width=width):
                calls: list[tuple[str, ...]] = []
                listing = (
                    f"100644 blob {'a' * width}\t{repository_path}\0".encode("ascii")
                )

                def fake_run_git(_root, arguments, *, output_limit):
                    calls.append(tuple(arguments))
                    self.assertEqual(arguments[0], "ls-tree")
                    return listing

                with mock.patch.object(G, "_run_git", side_effect=fake_run_git):
                    with self.assertRaisesRegex(
                        G.TrustedReleaseAuthorityError, "malformed blob object ID"
                    ):
                        G._git_blob_from_tree(
                            pathlib.Path("/temporary-root"),
                            "b" * 40,
                            authority_path,
                            repository_path,
                        )
                self.assertEqual(len(calls), 1)

        for width in (40, 64):
            with self.subTest(accepted_width=width):
                oid = "a" * width
                responses = iter(
                    (
                        f"100644 blob {oid}\t{repository_path}\0".encode("ascii"),
                        b"1\n",
                        b"x",
                    )
                )
                with mock.patch.object(
                    G, "_run_git", side_effect=lambda *_args, **_kwargs: next(responses)
                ):
                    identity, raw = G._git_blob_from_tree(
                        pathlib.Path("/temporary-root"),
                        "b" * 40,
                        authority_path,
                        repository_path,
                    )
                self.assertEqual(identity.oid, oid)
                self.assertEqual(raw, b"x")

    def test_git_final_wait_timeout_terminates_and_raises_project_error(self) -> None:
        process = mock.Mock()
        process.stdout = mock.Mock()
        process.stderr = mock.Mock()
        process.wait.side_effect = (
            subprocess.TimeoutExpired("git", 1.0),
            0,
        )
        selector = mock.Mock()
        selector.get_map.return_value = {}
        with mock.patch.object(G.subprocess, "Popen", return_value=process), mock.patch.object(
            G.selectors, "DefaultSelector", return_value=selector
        ), mock.patch.object(G, "_git_executable", return_value="/usr/bin/git"):
            with self.assertRaisesRegex(
                G.TrustedReleaseAuthorityError, "did not exit after bounded I/O"
            ):
                G._run_git(pathlib.Path("/temporary-root"), ("status",))
        process.kill.assert_called_once_with()
        self.assertEqual(process.wait.call_count, 2)

    def test_committed_symlink_and_gitlink_fixed_source_reject(self) -> None:
        for kind in ("symlink", "gitlink"):
            with self.subTest(kind=kind):
                fixture = IsolatedGitRepository(self)
                target = fixture.source_path("definition/oai-memprof-container-wire-v1.py")
                target.unlink()
                if kind == "symlink":
                    target.symlink_to("replacement")
                else:
                    target.mkdir()
                    fixture.run("git", "init", "-q", os.fspath(target))
                    subprocess.run(
                        ("git", "config", "user.email", "authority-test@example.invalid"),
                        cwd=target,
                        check=True,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    subprocess.run(
                        ("git", "config", "user.name", "Authority Test"),
                        cwd=target,
                        check=True,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    (target / "payload").write_text("nested\n", encoding="utf-8")
                    subprocess.run(
                        ("git", "add", "payload"),
                        cwd=target,
                        check=True,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    subprocess.run(
                        ("git", "commit", "-q", "-m", "nested"),
                        cwd=target,
                        check=True,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                fixture.commit_all(f"replace fixed source with {kind}")
                with self.assertRaisesRegex(
                    G.TrustedReleaseAuthorityError, "not a regular blob"
                ):
                    self._generate(fixture, fixture.fresh_output(f"{kind}-output"))

    def test_runtime_current_source_mutation_and_symlink_reject(self) -> None:
        fixture = IsolatedGitRepository(self)
        authority_path = "definition/oai-memprof-container-wire-v1.py"
        repository_path = G.TRUSTED_RELEASE_SOURCE_PATHS[authority_path]
        expected = fixture.source_bytes[authority_path]
        current = fixture.root / repository_path
        current.write_bytes(b"different but uncommitted\n")
        with self.assertRaisesRegex(
            G.TrustedReleaseAuthorityError, "size differs|bytes differ"
        ):
            G.read_stable_current_source(fixture.root, repository_path, expected)
        current.unlink()
        current.symlink_to("replacement")
        with self.assertRaisesRegex(
            G.TrustedReleaseAuthorityError, "non-symlink regular file"):
            G.read_stable_current_source(fixture.root, repository_path, expected)

        parent_fixture = IsolatedGitRepository(self)
        parent_expected = parent_fixture.source_bytes[authority_path]
        tools_directory = parent_fixture.root / "tools"
        redirected_tools = parent_fixture.root / "redirected-tools"
        tools_directory.rename(redirected_tools)
        tools_directory.symlink_to("redirected-tools", target_is_directory=True)
        with self.assertRaisesRegex(
            G.TrustedReleaseAuthorityError, "parent is not a non-symlink directory"
        ):
            G.read_stable_current_source(
                parent_fixture.root,
                repository_path,
                parent_expected,
            )

    def test_nonempty_output_and_overwrite_refuse_without_replacement(self) -> None:
        fixture = IsolatedGitRepository(self)
        nonempty = fixture.fresh_output("nonempty")
        sentinel = nonempty / "sentinel"
        sentinel.write_bytes(b"preserve\n")
        with self.assertRaisesRegex(
            G.TrustedReleaseAuthorityError, "fresh and empty"
        ):
            self._generate(fixture, nonempty)
        self.assertEqual(sentinel.read_bytes(), b"preserve\n")

        output = fixture.fresh_output("published")
        first = self._generate(fixture, output)
        before = first.authority_path.read_bytes()
        with self.assertRaisesRegex(
            G.TrustedReleaseAuthorityError, "fresh and empty"
        ):
            self._generate(fixture, output)
        self.assertEqual(first.authority_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
