#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oai_profile_archive import (  # noqa: E402
    MANIFEST_NAME,
    archive_identity,
    finalize_archive,
    register_external_source,
    sha256_file,
    verify_archive,
)


def write_run(run_dir: Path) -> None:
    run_dir.mkdir()
    (run_dir / "metadata.txt").write_text(
        "schema_version=2\n"
        "run_id=test-run\n"
        "experiment_id=test-experiment\n"
        "campaign_id=test-campaign\n"
        "variant=in-process\n"
        "trial=1\n"
        "role=nrUE\n"
        "hostname=cm5\n"
        "clean_shutdown=1\n"
    )
    (run_dir / "events.csv").write_text("seq,event_name\n0,UE_SLOT_LOOP\n")


class ArchiveIntegrityTest(unittest.TestCase):
    def test_disabled_controlled_sigint_has_consistent_archive_state(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="oai-profile-controlled-disabled-"
        ) as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            campaign = {
                "schema_version": 1,
                "runner_version": 2,
                "run_id": "controlled-disabled",
                "campaign_id": "controlled-campaign",
                "experiment_id": "controlled-experiment",
                "variant": "disabled",
                "trial": 1,
                "role": "gNB",
                "hostname": "laptop",
                "host": "local",
                "status": "exited_nonzero",
                "return_code": -2,
                "stop_reason": "duration_elapsed",
                "completion_classifier_version": 1,
                "controlled_stop_requested": 1,
                "shutdown_stage": "SIGINT",
                "shutdown_verified": 1,
                "remote_completion_identity_verified": "not_applicable",
                "termination_class": "controlled_sigint_signal",
            }
            (run_dir / "campaign_run.json").write_text(
                json.dumps(campaign, sort_keys=True) + "\n"
            )
            self.assertEqual(
                archive_identity(run_dir)["archive_state"],
                "runner_completed_controlled_sigint",
            )
            finalize_archive(run_dir)
            self.assertTrue(all(result.valid for result in verify_archive(run_dir)))
            with (run_dir / MANIFEST_NAME).open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(
                {row["archive_state"] for row in rows},
                {"runner_completed_controlled_sigint"},
            )

            unproven = Path(temporary) / "unproven"
            unproven.mkdir()
            campaign.pop("completion_classifier_version")
            campaign.pop("controlled_stop_requested")
            campaign.pop("shutdown_stage")
            campaign.pop("shutdown_verified")
            campaign.pop("remote_completion_identity_verified")
            (unproven / "campaign_run.json").write_text(
                json.dumps(campaign, sort_keys=True) + "\n"
            )
            self.assertEqual(
                archive_identity(unproven)["archive_state"],
                "runner_incomplete",
            )

            natural = Path(temporary) / "natural"
            natural.mkdir()
            natural_campaign = {
                **campaign,
                "status": "finished",
                "return_code": 0,
                "completion_classifier_version": 1,
                "controlled_stop_requested": 0,
                "shutdown_stage": "none",
                "shutdown_verified": 1,
                "remote_completion_identity_verified": "not_applicable",
                "termination_class": "natural_zero",
            }
            (natural / "campaign_run.json").write_text(
                json.dumps(natural_campaign, sort_keys=True) + "\n"
            )
            self.assertEqual(
                archive_identity(natural)["archive_state"],
                "runner_completed",
            )

            for name, tampered in (
                ("natural-escalated", {**natural_campaign, "shutdown_stage": "SIGTERM"}),
                ("natural-unverified", {**natural_campaign, "shutdown_verified": 0}),
                (
                    "natural-unsupported",
                    {**natural_campaign, "completion_classifier_version": 999},
                ),
                (
                    "remote-natural-unverified",
                    {
                        **natural_campaign,
                        "host": "cm5",
                        "remote_completion_identity_verified": 0,
                    },
                ),
            ):
                invalid = Path(temporary) / name
                invalid.mkdir()
                (invalid / "campaign_run.json").write_text(
                    json.dumps(tampered, sort_keys=True) + "\n"
                )
                self.assertEqual(
                    archive_identity(invalid)["archive_state"],
                    "runner_incomplete",
                )

    def test_verify_rejects_malformed_manifest_rows(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-manifest-validation-") as temporary:
            run_dir = Path(temporary) / "run"
            write_run(run_dir)
            finalize_archive(run_dir)
            manifest = run_dir / MANIFEST_NAME

            def mutate(field: str, value: str) -> str:
                with manifest.open(newline="") as stream:
                    reader = csv.DictReader(stream)
                    fields = list(reader.fieldnames or [])
                    rows = list(reader)
                rows[0][field] = value
                with manifest.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                return rows[0]["relative_path"]

            relative_path = mutate("manifest_version", "999")
            statuses = {
                result.relative_path: result.status for result in verify_archive(run_dir)
            }
            self.assertEqual(statuses[relative_path], "invalid_manifest_version")

            finalize_archive(run_dir, replace=True)
            relative_path = mutate("role", "gNB")
            statuses = {
                result.relative_path: result.status for result in verify_archive(run_dir)
            }
            self.assertEqual(statuses[relative_path], "manifest_identity_mismatch")

            finalize_archive(run_dir, replace=True)
            relative_path = mutate("artifact_type", "unrelated")
            statuses = {
                result.relative_path: result.status for result in verify_archive(run_dir)
            }
            self.assertEqual(statuses[relative_path], "artifact_type_mismatch")

            finalize_archive(run_dir, replace=True)
            relative_path = mutate("sha256", "not-a-sha256")
            statuses = {
                result.relative_path: result.status for result in verify_archive(run_dir)
            }
            self.assertEqual(statuses[relative_path], "invalid_sha256")

            finalize_archive(run_dir, replace=True)
            relative_path = mutate("size_bytes", "-1")
            statuses = {
                result.relative_path: result.status for result in verify_archive(run_dir)
            }
            self.assertEqual(statuses[relative_path], "invalid_manifest_value")


    def test_finalize_rejects_file_replacement_during_hashing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-archive-replacement-") as temporary:
            run_dir = Path(temporary) / "run"
            write_run(run_dir)
            original_sha256_file = sha256_file
            replacement_performed = False

            def replace_events_before_hash(path: Path) -> str:
                nonlocal replacement_performed
                if path.name == "events.csv":
                    replacement = path.with_name("events.replacement")
                    replacement.write_bytes(path.read_bytes())
                    os.replace(replacement, path)
                    replacement_performed = True
                return original_sha256_file(path)

            with mock.patch("oai_profile_archive.sha256_file", side_effect=replace_events_before_hash):
                with self.assertRaisesRegex(RuntimeError, "file changed while hashing"):
                    finalize_archive(run_dir)
    def test_finalize_verify_and_detect_mutation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-archive-tool-") as temporary:
            run_dir = Path(temporary) / "run"
            write_run(run_dir)

            manifest = finalize_archive(run_dir)
            self.assertEqual(manifest, run_dir / MANIFEST_NAME)
            self.assertTrue(all(result.valid for result in verify_archive(run_dir)))

            events = run_dir / "events.csv"
            original = events.read_text()
            events.write_text(original.replace("UE_SLOT_LOOP", "UE_SLOT_LOOX"))
            statuses = {result.relative_path: result.status for result in verify_archive(run_dir)}
            self.assertEqual(statuses["events.csv"], "hash_mismatch")

            events.write_text(original)
            finalize_archive(run_dir, replace=True)
            current = events.stat()
            os.utime(events, ns=(current.st_atime_ns, current.st_mtime_ns + 1))
            statuses = {result.relative_path: result.status for result in verify_archive(run_dir)}
            self.assertEqual(statuses["events.csv"], "mtime_mismatch")

            finalize_archive(run_dir, replace=True)
            (run_dir / "unregistered.txt").write_text("extra\n")
            statuses = {result.relative_path: result.status for result in verify_archive(run_dir)}
            self.assertEqual(statuses["unregistered.txt"], "unmanifested")

    def test_register_future_external_power_artifact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-external-tool-") as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            write_run(run_dir)
            power_file = root / "m5stack-power.csv"
            power_file.write_text("timestamp_ms,voltage_v,current_a,power_w\n1,5.1,2.0,10.2\n")
            args = argparse.Namespace(
                run_dir=run_dir,
                source_id="m5stack-va-001",
                source_type="m5stack_voltage_current_power",
                artifact=power_file,
                copy_artifact=True,
                clock_domain="m5stack_device_clock",
                clock_unit="ms",
                command="",
                tool_version="v1.1.3",
                start_realtime_ns="",
                end_realtime_ns="",
                start_monotonic_raw_ns="",
                end_monotonic_raw_ns="",
                status="recorded",
                alignment_method="unresolved",
                alignment_uncertainty_ns="",
                notes="alignment deferred",
                replace_manifest=False,
            )
            register_external_source(args)

            copied = run_dir / "external" / "m5stack-va-001" / power_file.name
            self.assertEqual(copied.read_text(), power_file.read_text())
            with (run_dir / "external_sources.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source_id"], "m5stack-va-001")
            self.assertEqual(rows[0]["alignment_method"], "unresolved")
            self.assertEqual(rows[0]["artifact_path"], "external/m5stack-va-001/m5stack-power.csv")

            finalize_archive(run_dir)
            self.assertTrue(all(result.valid for result in verify_archive(run_dir)))
            with self.assertRaises(FileExistsError):
                register_external_source(
                    argparse.Namespace(
                        **{
                            **vars(args),
                            "source_id": "m5stack-va-002",
                            "replace_manifest": False,
                        }
                    )
                )

    def test_reject_symlinked_artifact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-symlink-tool-") as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            write_run(run_dir)
            outside = root / "outside"
            outside.write_text("not archival\n")
            (run_dir / "link").symlink_to(outside)
            with self.assertRaises(ValueError):
                finalize_archive(run_dir)

    def test_verify_rejects_intermediate_symlink(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-verify-symlink-") as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            write_run(run_dir)
            nested = run_dir / "nested"
            nested.mkdir()
            (nested / "artifact.txt").write_text("archived\n")
            finalize_archive(run_dir)

            outside = root / "outside"
            nested.rename(outside)
            nested.symlink_to(outside, target_is_directory=True)

            statuses = {result.relative_path: result.status for result in verify_archive(run_dir)}
            self.assertEqual(statuses["nested/artifact.txt"], "unexpected_symlink")
            self.assertEqual(statuses["nested"], "unexpected_symlink")


if __name__ == "__main__":
    unittest.main()
