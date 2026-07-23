#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import oai_profile_campaign as campaign  # noqa: E402
from oai_profile_archive import verify_archive  # noqa: E402
from oai_profile_campaign import (  # noqa: E402
    Endpoint,
    ExperimentPlan,
    ProcessHandle,
    Variant,
    build_plans,
    experiment_failed,
    execute_plan,
    launch_remote,
    load_spec,
    plan_view,
    preflight_endpoint,
    register_sidecar,
    redact_command,
    remote_group_is_running,
    signal_handle,
    sidecar_clock,
    sidecar_source_anchors,
    sidecar_command,
)


GRACEFUL_SLEEP = (
    "import signal,sys,time;"
    "signal.signal(signal.SIGINT,lambda *_:sys.exit(0));"
    "time.sleep(30)"
)


def role(profile_root: Path, name: str, command: list[str], delay_s: float = 0.0) -> dict[str, object]:
    return {
        "host": "local",
        "hostname": name,
        "profile_root": str(profile_root),
        "command": command,
        "launch_delay_s": delay_s,
    }


def write_spec(
    path: Path,
    profile_root: Path,
    gnb_command: list[str],
    ue_command: list[str],
    *,
    duration_s: float = 0.4,
    ue_delay_s: float = 0.0,
    variants: list[dict[str, object]] | None = None,
    cases: list[dict[str, object]] | None = None,
    trials: int = 1,
    start_order: list[str] | None = None,
) -> None:
    value = {
        "schema_version": 1,
        "campaign_id": "campaign-test",
        "duration_s": duration_s,
        "stop_grace_s": 1.0,
        "stop_on_role_exit": True,
        "trials": trials,
        "start_order": start_order or ["gNB", "nrUE"],
        "roles": {
            "gNB": role(profile_root, "gnb-host", gnb_command),
            "nrUE": role(profile_root, "ue-host", ue_command, ue_delay_s),
        },
        "variants": variants or [{"name": "disabled", "profile": False, "pmu": "off", "sidecar": "none"}],
        "cases": cases or [{"name": "baseline"}],
    }
    path.write_text(json.dumps(value))


def read_results(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


class CampaignRunnerTest(unittest.TestCase):
    def test_stop_handles_signals_reverse_launch_order(self) -> None:
        gnb_handle = SimpleNamespace(stop_reason="")
        nrue_handle = SimpleNamespace(stop_reason="")
        handles = [gnb_handle, nrue_handle]
        with (
            patch("oai_profile_campaign.shutdown_target_is_running", return_value=True),
            patch("oai_profile_campaign.signal_handle") as signal,
            patch("oai_profile_campaign.wait_handles", side_effect=[False, False, True]),
        ):
            campaign.stop_handles(handles, "duration_elapsed", 1.0)

        self.assertEqual(
            [entry.args for entry in signal.call_args_list],
            [
                (nrue_handle, "INT"),
                (gnb_handle, "INT"),
                (nrue_handle, "TERM"),
                (gnb_handle, "TERM"),
                (nrue_handle, "KILL"),
                (gnb_handle, "KILL"),
            ],
        )

    def test_remote_launch_waits_for_session_payload(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-remote-launch-") as temporary:
            root = Path(temporary)
            run_dir = root / "profiles" / "run"
            run_dir.mkdir(parents=True)
            marker = root / "payload-finished"
            payload = (
                "import pathlib,sys,time;"
                "time.sleep(0.1);"
                f"pathlib.Path({str(marker)!r}).write_text('done');"
                "sys.exit(7)"
            )
            endpoint = Endpoint(
                role="nrUE",
                host="cm5",
                hostname="cm5",
                profile_root=str(run_dir.parent),
                command=[sys.executable, "-c", payload],
                cwd=str(root),
                sudo=False,
                environment={},
                archive_tool="/repo/tools/profiling/oai_profile_archive.py",
                launch_delay_s=0.0,
            )
            handle = ProcessHandle(
                endpoint=endpoint,
                run_dir=str(run_dir),
                command=endpoint.command,
                environment={},
            )
            with patch("oai_profile_campaign.subprocess.Popen", return_value=object()) as popen:
                launch_remote(handle, root / "control")

            remote_shell = popen.call_args.args[0][-1]
            self.assertIn("exec setsid --wait sh -c", remote_shell)
            assert handle.stdout_handle is not None
            assert handle.stderr_handle is not None
            handle.stdout_handle.close()
            handle.stderr_handle.close()
            completed = subprocess.run(["sh", "-c", remote_shell], check=False)
            self.assertEqual(completed.returncode, 7)
            self.assertEqual(marker.read_text(), "done")
            control_pid = int((run_dir / "control.pid").read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(control_pid, 0)

    def test_remote_preflight_requires_setsid_wait(self) -> None:
        endpoint = Endpoint(
            role="nrUE",
            host="cm5",
            hostname="cm5",
            profile_root="/profiles",
            command=["nr-uesoftmodem"],
            cwd=None,
            sudo=False,
            environment={},
            archive_tool="/repo/tools/profiling/oai_profile_archive.py",
            launch_delay_s=0.0,
        )
        handle = ProcessHandle(
            endpoint=endpoint,
            run_dir="/profiles/run",
            command=endpoint.command,
            environment={},
        )
        completed = subprocess.CompletedProcess(["ssh"], 0, "", "")
        with patch("oai_profile_campaign.remote_run", return_value=completed) as remote:
            preflight_endpoint(handle)
        commands = [call.args[1] for call in remote.call_args_list]
        self.assertIn("setsid --wait sh -c 'exit 0'", commands)

    def test_remote_group_control_survives_ssh_process_exit(self) -> None:
        endpoint = Endpoint(
            role="nrUE",
            host="cm5",
            hostname="cm5",
            profile_root="/profiles",
            command=["nr-uesoftmodem"],
            cwd=None,
            sudo=False,
            environment={},
            archive_tool="/repo/tools/profiling/oai_profile_archive.py",
            launch_delay_s=0.0,
        )
        handle = ProcessHandle(
            endpoint=endpoint,
            run_dir="/profiles/run",
            command=endpoint.command,
            environment={},
            process=object(),  # type: ignore[arg-type]
        )
        completed = subprocess.CompletedProcess(["ssh"], 0, "", "")
        with patch("oai_profile_campaign.remote_run", return_value=completed) as remote:
            self.assertTrue(remote_group_is_running(handle))
            signal_handle(handle, "INT")
        commands = [call.args[1] for call in remote.call_args_list]
        self.assertIn("kill -0", commands[0])
        self.assertIn("kill -INT", commands[1])

    def test_partial_preparation_failure_is_archived(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-campaign-preparation-") as temporary:
            root = Path(temporary)
            spec_path = root / "campaign.json"
            command = [sys.executable, "-c", GRACEFUL_SLEEP]
            write_spec(spec_path, root / "profiles", command, command)
            spec = load_spec(spec_path)
            plan = build_plans(spec, set(), set(), set())[0]
            control_root = root / "control"
            original_atomic_write = campaign.atomic_write_json
            invocation = 0

            def fail_first_manifest(path: Path, value: dict[str, object]) -> None:
                nonlocal invocation
                invocation += 1
                if invocation == 1:
                    raise OSError("synthetic initial manifest failure")
                original_atomic_write(path, value)

            with patch(
                "oai_profile_campaign.atomic_write_json",
                side_effect=fail_first_manifest,
            ):
                with self.assertRaisesRegex(OSError, "synthetic initial manifest failure"):
                    execute_plan(spec, plan, control_root)

            rows = read_results(control_root / "campaign_results.csv")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["role"], "gNB")
            self.assertEqual(rows[0]["run_status"], "preparation_failed")
            self.assertEqual(rows[0]["archive_status"], "finalized")
            self.assertTrue(
                all(result.valid for result in verify_archive(Path(rows[0]["run_dir"])))
            )

    def test_repository_example_is_valid(self) -> None:
        example = Path(__file__).resolve().parents[1] / "campaign_laptop_cm5.example.json"
        spec = load_spec(example)
        plans = build_plans(spec, set(), set(), set())

        self.assertEqual(spec["campaign_id"], "band28-25prb-cm5-baseline")
        self.assertEqual(len(plans), 35)
        self.assertEqual(spec["_start_order"], ["gNB", "nrUE"])
        self.assertEqual(
            {plan.variant.name for plan in plans},
            {"disabled", "in-process", "pmu-software", "pmu-all", "perf-stat", "perf-record", "perf-sched"},
        )

    def test_matrix_order_validation_and_redaction(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-campaign-plan-") as temporary:
            root = Path(temporary)
            spec_path = root / "campaign.json"
            variants = [
                {"name": "disabled", "profile": False, "pmu": "off", "sidecar": "none"},
                {"name": "profiled", "profile": True, "pmu": "software", "sidecar": "none"},
            ]
            cases = [{"name": "case-a"}, {"name": "case-b"}]
            command = [sys.executable, "-c", "pass"]
            write_spec(
                spec_path,
                root / "profiles",
                command,
                command,
                variants=variants,
                cases=cases,
                trials=2,
                start_order=["nrUE", "gNB"],
            )

            spec = load_spec(spec_path)
            plans = build_plans(spec, set(), set(), set())
            self.assertEqual(len(plans), 8)
            view = plan_view(spec, plans[0])
            self.assertEqual(view["start_order"], ["nrUE", "gNB"])
            self.assertEqual(list(view["roles"]), ["nrUE", "gNB"])
            self.assertEqual(view["roles"]["nrUE"]["launch_index"], 0)
            self.assertEqual(
                redact_command(["app", "--password", "secret", "/tmp/client.key", "--mode", "test"]),
                ["app", "--password", "<redacted>", "<redacted>", "--mode", "test"],
            )
            self.assertEqual(
                redact_command(
                    [
                        "app",
                        "--uicc0.imsi",
                        "001010000000001",
                        "--supi=imsi-001010000000001",
                        "--imei",
                        "490154203237518",
                    ]
                ),
                ["app", "--uicc0.imsi", "<redacted>", "--supi=<redacted>", "--imei", "<redacted>"],
            )
            remote_endpoint = Endpoint(
                role="nrUE",
                host="cm5",
                hostname="cm5",
                profile_root="/profiles",
                command=["nr-uesoftmodem"],
                cwd=None,
                sudo=False,
                environment={},
                archive_tool="/repo/tools/profiling/oai_profile_archive.py",
                launch_delay_s=0.0,
            )
            remote_handle = ProcessHandle(
                endpoint=remote_endpoint,
                run_dir="/profiles/run",
                command=remote_endpoint.command,
                environment={},
                start_realtime_ns=1,
                end_realtime_ns=2,
                start_monotonic_ns=3,
                end_monotonic_ns=4,
            )
            remote_anchors = sidecar_source_anchors(remote_handle)
            self.assertEqual(remote_anchors[:4], ("", "", "", ""))
            self.assertIn("orchestrator clock", remote_anchors[4])
            local_endpoint = Endpoint(
                role="gNB",
                host="local",
                hostname="gnb-host",
                profile_root="/profiles",
                command=["nr-softmodem"],
                cwd=None,
                sudo=False,
                environment={},
                archive_tool=None,
                launch_delay_s=0.0,
            )
            local_handle = ProcessHandle(
                endpoint=local_endpoint,
                run_dir="/profiles/run",
                command=local_endpoint.command,
                environment={},
                start_realtime_ns=1,
                end_realtime_ns=2,
                start_monotonic_ns=3,
                end_monotonic_ns=4,
            )
            self.assertEqual(sidecar_source_anchors(local_handle), (1, 2, 3, 4, ""))
            self.assertEqual(sidecar_clock("perf_record")[2], "shared_monotonic_raw")
            self.assertEqual(sidecar_clock("perf_sched")[2], "alignment_pending")
            with self.assertRaises(ValueError):
                sidecar_command(command, {"tool": "perf_record", "frequency_hz": 0}, str(root))

            invalid = json.loads(spec_path.read_text())
            invalid["start_order"] = ["gNB", "gNB"]
            spec_path.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):
                load_spec(spec_path)

    def test_sidecar_registration_stops_when_process_never_launched(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-sidecar-not-launched-") as temporary:
            run_dir = Path(temporary)
            endpoint = Endpoint(
                role="nrUE",
                host="local",
                hostname="ue-host",
                profile_root=str(run_dir),
                command=["nr-uesoftmodem"],
                cwd=None,
                sudo=False,
                environment={},
                archive_tool=None,
                launch_delay_s=0.0,
            )
            variant = Variant("perf-stat", True, "off", {"tool": "perf_stat"}, {})
            plan = ExperimentPlan(
                "campaign-test",
                "baseline",
                variant,
                1,
                "campaign-test-baseline-perf-stat-t001",
                {},
                {},
            )
            handle = ProcessHandle(
                endpoint=endpoint,
                run_dir=str(run_dir),
                command=endpoint.command,
                environment={},
                sidecar_tool="perf_stat",
                sidecar_artifact=str(run_dir / "sidecars" / "perf_stat.csv"),
            )

            register_sidecar(handle, plan)

            self.assertEqual(handle.sidecar_status, "not_launched")
            self.assertEqual(handle.notes, [])

    def test_local_paired_run_is_finalized_and_verifiable(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-campaign-run-") as temporary:
            root = Path(temporary)
            spec_path = root / "campaign.json"
            command = [sys.executable, "-c", GRACEFUL_SLEEP]
            write_spec(spec_path, root / "profiles", command, command)

            spec = load_spec(spec_path)
            plan = build_plans(spec, set(), set(), set())[0]
            control_root = root / "control"
            rows = execute_plan(spec, plan, control_root)

            self.assertEqual([row["role"] for row in rows], ["gNB", "nrUE"])
            for row in rows:
                self.assertEqual(row["run_status"], "finished")
                self.assertEqual(row["return_code"], 0)
                self.assertEqual(row["archive_status"], "finalized")
                self.assertEqual(row["sidecar_status"], "not_requested")
                run_dir = Path(row["run_dir"])
                self.assertTrue(all(result.valid for result in verify_archive(run_dir)))
                campaign = json.loads((run_dir / "campaign_run.json").read_text())
                self.assertEqual(campaign["status"], "finished")
                self.assertEqual(campaign["archive_status"], "finalization_pending")
                self.assertEqual(campaign["experiment_id"], plan.experiment_id)

            results = read_results(control_root / "campaign_results.csv")
            self.assertEqual(len(results), 2)
            self.assertEqual({row["run_status"] for row in results}, {"finished"})

    def test_experiment_failure_requires_complete_duration(self) -> None:
        successful = {
            "role": "gNB",
            "archive_status": "finalized",
            "run_status": "finished",
            "return_code": 0,
            "stop_reason": "duration_elapsed",
            "sidecar": "none",
            "sidecar_status": "not_requested",
        }
        self.assertFalse(experiment_failed([successful], {"gNB"}))
        for field, value in (
            ("archive_status", "finalize_failed"),
            ("run_status", "exited_nonzero"),
            ("return_code", 1),
            ("stop_reason", "paired_role_exited"),
        ):
            failed = dict(successful)
            failed[field] = value
            self.assertTrue(experiment_failed([failed], {"gNB"}), field)
        self.assertTrue(experiment_failed([], {"gNB", "nrUE"}))
        self.assertTrue(experiment_failed([successful], {"gNB", "nrUE"}))
        self.assertTrue(experiment_failed([successful, successful], {"gNB", "nrUE"}))

    def test_main_fails_clean_early_exit_and_preserves_archives(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-campaign-early-exit-") as temporary:
            root = Path(temporary)
            spec_path = root / "campaign.json"
            early_exit = [sys.executable, "-c", "import time;time.sleep(0.15)"]
            peer = [sys.executable, "-c", GRACEFUL_SLEEP]
            write_spec(
                spec_path,
                root / "profiles",
                early_exit,
                peer,
                duration_s=2.0,
            )
            control_root = root / "control"
            argv = [
                "oai_profile_campaign.py",
                str(spec_path),
                "--execute",
                "--control-root",
                str(control_root),
            ]
            with patch.object(sys, "argv", argv):
                self.assertEqual(campaign.main(), 1)

            rows = read_results(control_root / "campaign_results.csv")
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["return_code"] for row in rows}, {"0"})
            self.assertNotIn("duration_elapsed", {row["stop_reason"] for row in rows})
            for row in rows:
                self.assertEqual(row["archive_status"], "finalized")
                self.assertTrue(all(result.valid for result in verify_archive(Path(row["run_dir"]))))

    def test_partial_launch_failure_preserves_both_prepared_archives(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-campaign-failure-") as temporary:
            root = Path(temporary)
            spec_path = root / "campaign.json"
            gnb_command = [sys.executable, "-c", GRACEFUL_SLEEP]
            ue_command = ["/path/that/does/not/exist"]
            write_spec(
                spec_path,
                root / "profiles",
                gnb_command,
                ue_command,
                ue_delay_s=0.2,
            )

            spec = load_spec(spec_path)
            plan = build_plans(spec, set(), set(), set())[0]
            control_root = root / "control"
            with self.assertRaises(FileNotFoundError):
                execute_plan(spec, plan, control_root)

            results = read_results(control_root / "campaign_results.csv")
            self.assertEqual(len(results), 2)
            by_role = {row["role"]: row for row in results}
            self.assertEqual(by_role["gNB"]["run_status"], "finished")
            self.assertEqual(by_role["nrUE"]["run_status"], "launch_failed")
            self.assertEqual(by_role["gNB"]["archive_status"], "finalized")
            self.assertEqual(by_role["nrUE"]["archive_status"], "finalized")
            for row in results:
                self.assertTrue(all(result.valid for result in verify_archive(Path(row["run_dir"]))))


if __name__ == "__main__":
    unittest.main()
