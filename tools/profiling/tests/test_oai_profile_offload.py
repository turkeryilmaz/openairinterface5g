#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import oai_profile_offload as offload  # noqa: E402


class ProfileOffloadTest(unittest.TestCase):
    def make_policy(self, root: Path) -> offload.StoragePolicy:
        active_root = root / "active"
        archive_mount = root / "archive"
        active_root.mkdir()
        archive_mount.mkdir()
        return offload.StoragePolicy(
            active_root=active_root,
            archive_root=archive_mount / "batches",
            archive_mount=archive_mount,
            archive_uuid="ed46bf11-c2eb-4f27-9df1-2a99a816bf5a",
            inventory_root=archive_mount / "inventory",
            archive_free_reserve_bytes=0,
        )

    def mount_result(self) -> dict[str, str]:
        return {
            "target": "/archive",
            "source": "/dev/test",
            "fstype": "ext4",
            "options": "rw",
            "uuid": "ed46bf11-c2eb-4f27-9df1-2a99a816bf5a",
        }

    def distinct_device(
        self,
        policy: offload.StoragePolicy,
        path: object,
    ) -> int:
        candidate = Path(path)
        if candidate == policy.archive_mount or policy.archive_mount in candidate.parents:
            return 2
        return 1

    def execution_patches(self, policy: offload.StoragePolicy):
        return (
            mock.patch.object(
                offload, "validate_archive_mount", return_value=self.mount_result()
            ),
            mock.patch.object(offload, "sync_all"),
            mock.patch.object(
                offload.shutil,
                "disk_usage",
                return_value=offload.shutil._ntuple_diskusage(10**9, 0, 10**9),
            ),
            mock.patch.object(
                offload,
                "device_id",
                side_effect=lambda path: self.distinct_device(policy, path),
            ),
        )

    def test_move_is_verified_and_removes_source(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-offload-") as temporary:
            root = Path(temporary)
            policy = self.make_policy(root)
            batch = policy.active_root / "RxTxTime3_n1"
            (batch / "profiles" / "empty").mkdir(parents=True)
            (batch / "profiles" / "events.csv").write_bytes(b"event-data\n")
            (batch / "campaign.json").write_text('{"status":"complete"}\n')

            patches = self.execution_patches(policy)
            with patches[0], patches[1], patches[2], patches[3]:
                result = offload.offload_batch(
                    policy,
                    "RxTxTime3_n1",
                    remove_source=True,
                    transaction_id="transaction1",
                )

            destination = policy.archive_root / "RxTxTime3_n1"
            self.assertEqual(result["status"], "complete")
            self.assertTrue(result["source_removed"])
            self.assertFalse(batch.exists())
            self.assertEqual(
                (destination / "profiles" / "events.csv").read_bytes(),
                b"event-data\n",
            )
            self.assertTrue((destination / "profiles" / "empty").is_dir())
            status = json.loads(
                (policy.inventory_root / "transaction1" / "status.json").read_text()
            )
            self.assertEqual(status["tree"]["files"], 2)
            self.assertEqual(len(set(status["manifest_sha256"].values())), 1)
            self.assertEqual(len(status["manifest_sha256"]), 5)
            self.assertEqual(list(policy.archive_root.glob(".incoming-*")), [])

    def test_differing_destination_collision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-offload-") as temporary:
            root = Path(temporary)
            policy = self.make_policy(root)
            source = policy.active_root / "batch"
            destination = policy.archive_root / "batch"
            source.mkdir()
            destination.mkdir(parents=True)
            (source / "value").write_text("source")
            (destination / "value").write_text("destination")

            with mock.patch.object(
                offload, "validate_archive_mount", return_value=self.mount_result()
            ):
                with self.assertRaisesRegex(FileExistsError, "destination already exists"):
                    offload.offload_batch(
                        policy,
                        "batch",
                        remove_source=True,
                        transaction_id="collision",
                    )

            self.assertEqual((source / "value").read_text(), "source")
            self.assertEqual((destination / "value").read_text(), "destination")

    def test_symlink_is_rejected_without_publication(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-offload-") as temporary:
            root = Path(temporary)
            policy = self.make_policy(root)
            batch = policy.active_root / "batch"
            batch.mkdir()
            target = root / "outside"
            target.write_text("outside")
            (batch / "link").symlink_to(target)

            patches = self.execution_patches(policy)
            with patches[0], patches[1], patches[2], patches[3]:
                with self.assertRaisesRegex(ValueError, "unsupported non-regular"):
                    offload.offload_batch(
                        policy,
                        "batch",
                        remove_source=True,
                        transaction_id="symlink",
                    )

            self.assertTrue(batch.exists())
            self.assertFalse((policy.archive_root / "batch").exists())

    def test_source_change_during_copy_aborts_before_publication(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-offload-") as temporary:
            root = Path(temporary)
            policy = self.make_policy(root)
            batch = policy.active_root / "batch"
            batch.mkdir()
            payload = batch / "payload"
            payload.write_text("before")
            original_rsync = offload.run_rsync

            def copy_then_mutate(source: Path, staging: Path, log: Path) -> None:
                original_rsync(source, staging, log)
                payload.write_text("after")

            patches = self.execution_patches(policy)
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                mock.patch.object(offload, "run_rsync", side_effect=copy_then_mutate),
            ):
                with self.assertRaisesRegex(RuntimeError, "source_pre->source_post"):
                    offload.offload_batch(
                        policy,
                        "batch",
                        remove_source=True,
                        transaction_id="mutation",
                    )

            self.assertEqual(payload.read_text(), "after")
            self.assertFalse((policy.archive_root / "batch").exists())
            status = json.loads(
                (policy.inventory_root / "mutation" / "status.json").read_text()
            )
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["phase"], "verify_source_stability")

    def test_destination_race_cannot_replace_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-offload-") as temporary:
            root = Path(temporary)
            policy = self.make_policy(root)
            source = policy.active_root / "batch"
            source.mkdir()
            (source / "payload").write_text("source")
            destination = policy.archive_root / "batch"
            original_rsync = offload.run_rsync

            def copy_then_create_destination(
                source_path: Path,
                staging: Path,
                log: Path,
            ) -> None:
                original_rsync(source_path, staging, log)
                destination.mkdir()

            patches = self.execution_patches(policy)
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                mock.patch.object(
                    offload,
                    "run_rsync",
                    side_effect=copy_then_create_destination,
                ),
            ):
                with self.assertRaises(FileExistsError):
                    offload.offload_batch(
                        policy,
                        "batch",
                        remove_source=True,
                        transaction_id="destinationrace",
                    )

            self.assertEqual((source / "payload").read_text(), "source")
            self.assertTrue(destination.is_dir())
            self.assertEqual(list(destination.iterdir()), [])
            status = json.loads(
                (policy.inventory_root / "destinationrace" / "status.json").read_text()
            )
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["phase"], "atomic_publication")

    def test_invalid_mount_creates_no_archive_side_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-offload-") as temporary:
            root = Path(temporary)
            policy = self.make_policy(root)
            batch = policy.active_root / "batch"
            batch.mkdir()
            (batch / "payload").write_text("source")

            with mock.patch.object(
                offload,
                "validate_archive_mount",
                side_effect=RuntimeError("archive mount unavailable"),
            ):
                with self.assertRaisesRegex(RuntimeError, "mount unavailable"):
                    offload.offload_batch(
                        policy,
                        "batch",
                        remove_source=True,
                        transaction_id="nomount",
                    )

            self.assertFalse(policy.archive_root.exists())
            self.assertFalse(policy.inventory_root.exists())
            self.assertEqual((batch / "payload").read_text(), "source")

    def test_reclamation_failure_preserves_destination_and_retired_source(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-offload-") as temporary:
            root = Path(temporary)
            policy = self.make_policy(root)
            source = policy.active_root / "batch"
            source.mkdir()
            (source / "payload").write_text("source")

            patches = self.execution_patches(policy)
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                mock.patch.object(
                    offload.shutil,
                    "rmtree",
                    side_effect=RuntimeError("synthetic reclamation failure"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "reclamation failure"):
                    offload.offload_batch(
                        policy,
                        "batch",
                        remove_source=True,
                        transaction_id="reclamationfailure",
                    )

            destination = policy.archive_root / "batch"
            retired = policy.active_root / ".offloaded-reclamationfailure"
            self.assertEqual((destination / "payload").read_text(), "source")
            self.assertEqual((retired / "payload").read_text(), "source")
            self.assertFalse(source.exists())
            status = json.loads(
                (
                    policy.inventory_root
                    / "reclamationfailure"
                    / "status.json"
                ).read_text()
            )
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["phase"], "source_reclamation")
            self.assertTrue(status["destination_published"])
            self.assertEqual(status["retired_source"], str(retired))

    def test_mount_loss_before_publication_does_not_rewrite_ledger(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-offload-") as temporary:
            root = Path(temporary)
            policy = self.make_policy(root)
            source = policy.active_root / "batch"
            source.mkdir()
            (source / "payload").write_text("source")
            mount = self.mount_result()

            with (
                mock.patch.object(
                    offload,
                    "validate_archive_mount",
                    side_effect=[
                        mount,
                        RuntimeError("synthetic mount loss"),
                        RuntimeError("synthetic mount remains absent"),
                    ],
                ),
                mock.patch.object(offload, "sync_all"),
                mock.patch.object(
                    offload.shutil,
                    "disk_usage",
                    return_value=offload.shutil._ntuple_diskusage(10**9, 0, 10**9),
                ),
                mock.patch.object(
                    offload,
                    "device_id",
                    side_effect=lambda path: self.distinct_device(policy, path),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic mount loss"):
                    offload.offload_batch(
                        policy,
                        "batch",
                        remove_source=True,
                        transaction_id="mountloss",
                    )

            self.assertEqual((source / "payload").read_text(), "source")
            self.assertFalse((policy.archive_root / "batch").exists())
            self.assertTrue((policy.archive_root / ".incoming-mountloss").is_dir())
            status = json.loads(
                (policy.inventory_root / "mountloss" / "status.json").read_text()
            )
            self.assertEqual(status["status"], "running")
            self.assertEqual(status["phase"], "preflight")

    def test_policy_rejects_unknown_fields(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-offload-") as temporary:
            path = Path(temporary) / "policy.json"
            path.write_text(
                json.dumps(
                    {
                        "active_root": "/active",
                        "archive_root": "/archive/batches",
                        "archive_mount": "/archive",
                        "archive_uuid": "ed46bf11-c2eb-4f27-9df1-2a99a816bf5a",
                        "inventory_root": "/archive/inventory",
                        "archive_free_reserve_bytes": 0,
                        "typo": True,
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "unexpected=\\['typo'\\]"):
                offload.load_policy(path)


if __name__ == "__main__":
    unittest.main()
