#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0

from __future__ import annotations

import os
import shlex
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oai_profile_archive import finalize_archive, verify_archive  # noqa: E402
from oai_profile_collect import (  # noqa: E402
    analysis_command,
    copy_run,
    next_analysis_output_dir,
    normalize_manifest_mtimes,
)


class ProfileCollectorTest(unittest.TestCase):
    def test_analysis_recommendation_uses_unique_nonexistent_static_path(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-collect-") as temporary:
            local_root = Path(temporary) / "profiles with spaces"
            analysis_root = local_root / "Analysis"
            timestamp = "2026-07-24_02-05-00"
            (analysis_root / timestamp).mkdir(parents=True)
            (analysis_root / f"{timestamp}_01").mkdir()

            output_dir = next_analysis_output_dir(local_root, timestamp)

            self.assertEqual(output_dir, analysis_root / f"{timestamp}_02")
            self.assertFalse(output_dir.exists())
            command = analysis_command(local_root, output_dir)
            self.assertNotIn("$(", command)
            self.assertEqual(
                shlex.split(command),
                [
                    str(
                        Path(__file__)
                        .resolve()
                        .parents[1]
                        .joinpath("oai_profile_analyze.py")
                    ),
                    str(local_root),
                    "--output-dir",
                    str(output_dir),
                ],
            )

    def make_archive(self, root: Path) -> tuple[Path, Path]:
        run_dir = root / "2026-07-23_22-00-35_nrUE_cm5"
        run_dir.mkdir()
        artifact = run_dir / "stdout.log"
        artifact.write_bytes(b"original")
        precise_mtime_ns = 1_700_000_000_123_456_789
        os.utime(artifact, ns=(precise_mtime_ns, precise_mtime_ns))
        finalize_archive(run_dir)
        return run_dir, artifact

    def test_normalize_manifest_mtimes_repairs_second_resolution_copy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-collect-") as temporary:
            run_dir, artifact = self.make_archive(Path(temporary))
            observed = artifact.stat()
            second_resolution_ns = observed.st_mtime_ns // 1_000_000_000 * 1_000_000_000
            os.utime(artifact, ns=(observed.st_atime_ns, second_resolution_ns))

            failures = [result for result in verify_archive(run_dir) if not result.valid]
            self.assertEqual([result.status for result in failures], ["mtime_mismatch"])
            self.assertEqual(normalize_manifest_mtimes(run_dir), 1)
            self.assertTrue(all(result.valid for result in verify_archive(run_dir)))

    def test_normalize_manifest_mtimes_rejects_content_corruption(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-collect-") as temporary:
            run_dir, artifact = self.make_archive(Path(temporary))
            artifact.write_bytes(b"tampered")
            observed = artifact.stat()
            second_resolution_ns = observed.st_mtime_ns // 1_000_000_000 * 1_000_000_000
            os.utime(artifact, ns=(observed.st_atime_ns, second_resolution_ns))

            with self.assertRaisesRegex(RuntimeError, "hash_mismatch"):
                normalize_manifest_mtimes(run_dir)
            self.assertEqual(artifact.stat().st_mtime_ns, second_resolution_ns)

    def test_normalize_manifest_mtimes_rejects_arbitrary_mtime_change(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-collect-") as temporary:
            run_dir, artifact = self.make_archive(Path(temporary))
            observed = artifact.stat()
            changed_mtime_ns = observed.st_mtime_ns + 1
            os.utime(artifact, ns=(observed.st_atime_ns, changed_mtime_ns))

            with self.assertRaisesRegex(
                RuntimeError,
                "mtime_mismatch_not_second_truncation",
            ):
                normalize_manifest_mtimes(run_dir)
            self.assertEqual(artifact.stat().st_mtime_ns, changed_mtime_ns)

    def test_manifestless_partial_run_is_left_unmodified(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-collect-") as temporary:
            run_dir = Path(temporary) / "partial"
            run_dir.mkdir()
            artifact = run_dir / "stdout.log"
            artifact.write_bytes(b"partial")
            before = artifact.stat()

            self.assertIsNone(normalize_manifest_mtimes(run_dir))
            after = artifact.stat()
            self.assertEqual(after.st_mtime_ns, before.st_mtime_ns)
            self.assertEqual(artifact.read_bytes(), b"partial")

    def test_manifest_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-collect-") as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            target = root / "external-manifest.csv"
            target.write_text("external")
            (run_dir / "archive_manifest.csv").symlink_to(target)

            with self.assertRaisesRegex(RuntimeError, "manifest is not a regular file"):
                normalize_manifest_mtimes(run_dir)

    def test_copy_run_does_not_publish_corrupted_archive(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-collect-") as temporary:
            root = Path(temporary)
            source_root = root / "source"
            source_root.mkdir()
            source_run, artifact = self.make_archive(source_root)
            artifact.write_bytes(b"tampered")
            local_root = root / "local"
            local_root.mkdir()

            def copy_fixture(command: list[str], check: bool) -> None:
                self.assertTrue(check)
                shutil.copytree(source_run, Path(command[-1]) / source_run.name)

            with patch(
                "oai_profile_collect.subprocess.run",
                side_effect=copy_fixture,
            ):
                with self.assertRaisesRegex(RuntimeError, "hash_mismatch"):
                    copy_run("cm5", "/profiles", local_root, source_run.name, False)

            self.assertFalse((local_root / source_run.name).exists())
            self.assertEqual(list(local_root.glob(".incoming-*")), [])


if __name__ == "__main__":
    unittest.main()
