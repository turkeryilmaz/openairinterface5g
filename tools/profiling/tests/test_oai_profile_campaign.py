#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0

from __future__ import annotations

import csv
import json
import os
import shutil
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import oai_profile_campaign as campaign  # noqa: E402
import oai_profile_reports as reports  # noqa: E402
from oai_profile_archive import archive_identity, verify_archive  # noqa: E402
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
DEFAULT_SIGINT_SLEEP = "import time;time.sleep(30)"


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
    workload: dict[str, object] | None = None,
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
    if workload is not None:
        value["workload"] = workload
    path.write_text(json.dumps(value))


def workload_value(root: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "client_role": "nrUE",
        "helper": str(
            Path(__file__).resolve().parents[1] / "oai_profile_workload.py"
        ),
        "interface": "oaitun_ue1",
        "ipv4_subnet": "10.0.0.0/24",
        "server_ipv4": "192.168.70.135",
        "policy_table": 9999,
        "readiness_timeout_s": 30,
        "readiness_poll_s": 0.25,
        "ping_count": 3,
        "ping_timeout_s": 2,
        "duration_s": 120,
        "bitrate_bps": 1_000_000,
        "datagram_bytes": 1200,
        "lease_path": str(root / "ue0-table9999.lock"),
    }


def read_results(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


class CampaignRunnerTest(unittest.TestCase):
    def test_endpoint_text_read_honors_configured_sudo(self) -> None:
        def endpoint(host: str, sudo: bool) -> Endpoint:
            return Endpoint(
                role="nrUE",
                host=host,
                hostname="cm5" if host == "cm5" else "local",
                profile_root="/profiles",
                command=["nr-uesoftmodem"],
                cwd=None,
                sudo=sudo,
                environment={},
                archive_tool=None,
                launch_delay_s=0.0,
            )

        with tempfile.TemporaryDirectory(
            prefix="oai-profile-endpoint-read-"
        ) as temporary:
            local_path = Path(temporary) / "ordinary.json"
            local_path.write_text("local ordinary\n")
            self.assertEqual(
                campaign.read_endpoint_text(
                    endpoint("local", False),
                    str(local_path),
                ),
                "local ordinary\n",
            )

        local_result = subprocess.CompletedProcess(
            ["sudo", "-n", "cat"],
            0,
            "local privileged\n",
            "",
        )
        with patch(
            "oai_profile_campaign.subprocess.run",
            return_value=local_result,
        ) as local_run:
            self.assertEqual(
                campaign.read_endpoint_text(
                    endpoint("local", True),
                    "/profiles/run/workload/workload_run.json",
                ),
                "local privileged\n",
            )
        local_run.assert_called_once_with(
            [
                "sudo",
                "-n",
                "cat",
                "--",
                "/profiles/run/workload/workload_run.json",
            ],
            check=False,
            text=True,
            capture_output=True,
        )

        for sudo in (False, True):
            with self.subTest(remote=True, sudo=sudo):
                remote_result = subprocess.CompletedProcess(
                    ["ssh"],
                    0,
                    f"remote sudo={sudo}\n",
                    "",
                )
                with patch(
                    "oai_profile_campaign.remote_run",
                    return_value=remote_result,
                ) as remote:
                    self.assertEqual(
                        campaign.read_endpoint_text(
                            endpoint("cm5", sudo),
                            "/profiles/run with space/workload_run.json",
                        ),
                        f"remote sudo={sudo}\n",
                    )
                command = ["cat", "--", "/profiles/run with space/workload_run.json"]
                if sudo:
                    command = ["sudo", "-n", *command]
                remote.assert_called_once_with(
                    endpoint("cm5", sudo),
                    shlex.join(command),
                    check=False,
                    capture=True,
                )

        denied = subprocess.CompletedProcess(
            ["ssh"],
            1,
            "",
            "cat: Permission denied",
        )
        with patch(
            "oai_profile_campaign.remote_run",
            return_value=denied,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "status=1, stderr=cat: Permission denied",
            ):
                campaign.read_endpoint_text(
                    endpoint("cm5", True),
                    "/profiles/run/workload/workload_run.json",
                )

    def test_control_records_bind_identity_and_wrapper_completion(self) -> None:
        token = "a" * 32
        start = {
            "schema_version": 1,
            "action": "run",
            "experiment_id": "experiment-t001",
            "token": token,
            "pgid": 4321,
            "start_ticks": 987654,
        }
        start_text = campaign.control_record_text(start, completion=False)
        self.assertEqual(
            campaign.parse_control_record(
                start_text,
                action="run",
                experiment_id="experiment-t001",
                token=token,
                completion=False,
            ),
            start,
        )
        completed = {**start, "return_code": 7}
        self.assertEqual(
            campaign.parse_control_record(
                campaign.control_record_text(completed, completion=True),
                action="run",
                experiment_id="experiment-t001",
                token=token,
                completion=True,
            ),
            completed,
        )
        for malformed in (
            start_text.replace("pgid=4321", "pgid=0"),
            start_text.replace("start_ticks=987654", "start_ticks=-1"),
            start_text + "token=duplicate\n",
            start_text.replace("action=run", "action=cleanup"),
        ):
            with self.subTest(malformed=malformed):
                with self.assertRaises(ValueError):
                    campaign.parse_control_record(
                        malformed,
                        action="run",
                        experiment_id="experiment-t001",
                        token=token,
                        completion=False,
                    )

        wrapper = campaign.remote_control_inner_command(
            ["python3", "helper.py", "run"],
            action="run",
            experiment_id="experiment-t001",
            token=token,
            start_path="/profiles/run/workload/run.control.start",
            completion_path="/profiles/run/workload/run.control.complete",
        )
        self.assertLess(wrapper.index("run.control.start"), wrapper.index("python3"))
        self.assertLess(wrapper.index("python3"), wrapper.index("return_code=$command_rc"))
        self.assertIn("print $5", wrapper)
        self.assertIn('[ "$1" = "$$" ]', wrapper)
        self.assertIn("mv --", wrapper)
        generic_role_wrapper = campaign.remote_control_inner_command(
            ["role-command"],
            action="role:custom_role",
            experiment_id="experiment-t001",
            token=token,
            start_path="/profiles/run/control.start",
            completion_path="/profiles/run/control.complete",
        )
        self.assertIn("action=role:custom_role", generic_role_wrapper)

    def test_remote_control_state_is_token_and_starttime_bound(self) -> None:
        token = "b" * 32
        endpoint = Endpoint(
            role="nrUE",
            host="cm5",
            hostname="cm5",
            profile_root="/profiles",
            command=["nr-uesoftmodem"],
            cwd=None,
            sudo=True,
            environment={},
            archive_tool="/repo/tools/profiling/oai_profile_archive.py",
            launch_delay_s=0.0,
        )
        start = {
            "schema_version": 1,
            "action": "run",
            "experiment_id": "experiment-t001",
            "token": token,
            "pgid": 4321,
            "start_ticks": 987654,
        }
        start_result = subprocess.CompletedProcess(
            ["ssh"],
            0,
            campaign.control_record_text(start, completion=False),
            "",
        )
        missing = subprocess.CompletedProcess(["ssh"], 1, "", "")

        completed_identity = {**start, "return_code": 0}
        completed = subprocess.CompletedProcess(
            ["ssh"],
            0,
            campaign.control_record_text(
                completed_identity,
                completion=True,
            ),
            "",
        )
        for probe_status, expected_state, expected_detail in (
            (
                0,
                None,
                "completion_present_but_original_leader_identity_observed",
            ),
            (1, False, "matching_completion_group_absent"),
            (
                2,
                False,
                "matching_completion_pgid_reused_original_identity_dead",
            ),
            (
                3,
                True,
                "matching_completion_leader_absent_process_group_still_live",
            ),
            (44, None, "identity probe status=44"),
        ):
            with self.subTest(completion_probe_status=probe_status):
                probe = subprocess.CompletedProcess(
                    ["ssh"],
                    probe_status,
                    "",
                    "permission denied" if probe_status == 44 else "",
                )
                with patch(
                    "oai_profile_campaign.remote_run",
                    side_effect=[start_result, completed, probe],
                ) as remote:
                    state, identity, detail = campaign.remote_control_state(
                        endpoint,
                        action="run",
                        experiment_id="experiment-t001",
                        token=token,
                        start_path="/profiles/run/workload/run.control.start",
                        completion_path="/profiles/run/workload/run.control.complete",
                    )
                self.assertIs(state, expected_state)
                self.assertEqual(identity, completed_identity)
                self.assertIn(expected_detail, detail)
                self.assertEqual(remote.call_count, 3)

        for probe_status, expected_state, expected_detail in (
            (0, True, "matching_identity_live"),
            (1, False, "original_identity_absent"),
            (2, False, "pgid_reused_original_identity_dead"),
            (3, None, "leader_absent_process_group_still_live"),
            (44, None, "identity probe status=44"),
        ):
            with self.subTest(probe_status=probe_status):
                probe = subprocess.CompletedProcess(
                    ["ssh"],
                    probe_status,
                    "",
                    "permission denied" if probe_status == 44 else "",
                )
                with patch(
                    "oai_profile_campaign.remote_run",
                    side_effect=[start_result, missing, probe],
                ):
                    state, identity, detail = campaign.remote_control_state(
                        endpoint,
                        action="run",
                        experiment_id="experiment-t001",
                        token=token,
                        start_path="/profiles/run/workload/run.control.start",
                        completion_path="/profiles/run/workload/run.control.complete",
                    )
                self.assertIs(state, expected_state)
                self.assertEqual(identity, start)
                self.assertIn(expected_detail, detail)

        wrong_completion = {
            **start,
            "action": "cleanup",
            "return_code": 0,
        }
        with patch(
            "oai_profile_campaign.remote_run",
            side_effect=[
                start_result,
                subprocess.CompletedProcess(
                    ["ssh"],
                    0,
                    campaign.control_record_text(
                        wrong_completion,
                        completion=True,
                    ),
                    "",
                ),
            ],
        ):
            state, _, detail = campaign.remote_control_state(
                endpoint,
                action="run",
                experiment_id="experiment-t001",
                token=token,
                start_path="/profiles/run/workload/run.control.start",
                completion_path="/profiles/run/workload/run.control.complete",
            )
        self.assertIsNone(state)
        self.assertIn("completion record invalid", detail)

        signal_command = campaign.remote_identity_signal_command(
            "/profiles/run/workload/run.control.start",
            start,
            "INT",
            sudo=True,
        )
        self.assertIn("start_ticks=987654", signal_command)
        self.assertIn("/proc/4321/stat", signal_command)
        self.assertIn("sudo -n /bin/kill -INT -- -4321", signal_command)

    def test_completion_backed_group_liveness_delays_shutdown_admission(self) -> None:
        completed_identity = {
            "schema_version": 1,
            "action": "role:nrUE",
            "experiment_id": "experiment-t001",
            "token": "c" * 32,
            "pgid": 4321,
            "start_ticks": 987654,
            "return_code": 130,
        }
        with (
            patch(
                "oai_profile_campaign.remote_control_state",
                side_effect=[
                    (
                        True,
                        completed_identity,
                        "matching_completion_leader_absent_process_group_still_live",
                    ),
                    (
                        True,
                        completed_identity,
                        "matching_completion_leader_absent_process_group_still_live",
                    ),
                    (
                        False,
                        completed_identity,
                        "matching_completion_group_absent",
                    ),
                ],
            ) as state_probe,
            patch("oai_profile_campaign.time.sleep") as sleep,
        ):
            state, identity, detail = campaign.wait_remote_control_not_live(
                Endpoint(
                    role="nrUE",
                    host="cm5",
                    hostname="cm5",
                    profile_root="/profiles",
                    command=["nr-uesoftmodem"],
                    cwd=None,
                    sudo=True,
                    environment={},
                    archive_tool="/repo/tools/profiling/oai_profile_archive.py",
                    launch_delay_s=0.0,
                ),
                action="role:nrUE",
                experiment_id="experiment-t001",
                token="c" * 32,
                start_path="/profiles/run/control.start",
                completion_path="/profiles/run/control.complete",
                timeout_s=10.0,
            )
        self.assertFalse(state)
        self.assertEqual(identity, completed_identity)
        self.assertEqual(detail, "matching_completion_group_absent")
        self.assertEqual(state_probe.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

        process = SimpleNamespace(poll=lambda: 130, pid=7654, returncode=130)
        handle = ProcessHandle(
            endpoint=Endpoint(
                role="nrUE",
                host="cm5",
                hostname="cm5",
                profile_root="/profiles",
                command=["nr-uesoftmodem"],
                cwd=None,
                sudo=True,
                environment={},
                archive_tool="/repo/tools/profiling/oai_profile_archive.py",
                launch_delay_s=0.0,
            ),
            run_dir="/profiles/run",
            command=["nr-uesoftmodem"],
            environment={},
            process=process,
            experiment_id="experiment-t001",
            control_token="c" * 32,
        )
        with patch(
            "oai_profile_campaign.remote_control_state",
            return_value=(
                True,
                completed_identity,
                "matching_completion_leader_absent_process_group_still_live",
            ),
        ):
            self.assertFalse(campaign.role_shutdown_is_verified(handle))

    def test_synchronous_remote_workload_action_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-remote-action-") as temporary:
            root = Path(temporary)
            endpoint = Endpoint(
                role="nrUE",
                host="cm5",
                hostname="cm5",
                profile_root="/profiles",
                command=["nr-uesoftmodem"],
                cwd=None,
                sudo=True,
                environment={},
                archive_tool="/repo/tools/profiling/oai_profile_archive.py",
                launch_delay_s=0.0,
            )

            def new_workload() -> campaign.WorkloadHandle:
                return campaign.WorkloadHandle(
                    spec=campaign.parse_workload_spec(workload_value(root)),
                    endpoint=endpoint,
                    run_dir="/profiles/run",
                    experiment_id="experiment-t001",
                )

            completed = subprocess.CompletedProcess(["ssh"], 255, "", "transport lost")
            workload = new_workload()
            completed_identity = {
                "schema_version": 1,
                "action": "preflight",
                "experiment_id": "experiment-t001",
                "token": "c" * 32,
                "pgid": 4321,
                "start_ticks": 987654,
                "return_code": 0,
            }
            with (
                patch("oai_profile_campaign.secrets.token_hex", return_value="c" * 32),
                patch("oai_profile_campaign.remote_run", return_value=completed) as remote,
                patch(
                    "oai_profile_campaign.remote_control_state",
                    return_value=(False, completed_identity, "matching_completion"),
                ),
                patch("oai_profile_campaign.stop_remote_controlled_action") as stop,
            ):
                result = campaign.invoke_workload_action(workload, "preflight", 20.0)
            self.assertEqual(result.returncode, 255)
            self.assertTrue(workload.evidence_quiesced)
            self.assertEqual(workload.control_tokens, {"preflight": "c" * 32})
            self.assertEqual(
                workload.control_results["preflight"],
                {
                    "transport_status": "returned",
                    "transport_return_code": 255,
                    "remote_completion_return_code": 0,
                },
            )
            self.assertIn("preflight.control.start", remote.call_args.args[1])
            self.assertIn("preflight.control.complete", remote.call_args.args[1])
            stop.assert_not_called()

            identity = {
                "schema_version": 1,
                "action": "cleanup",
                "experiment_id": "experiment-t001",
                "token": "d" * 32,
                "pgid": 4321,
                "start_ticks": 987654,
            }
            workload = new_workload()
            timeout = subprocess.TimeoutExpired(["ssh"], 20.0)
            completed_cleanup = {**identity, "return_code": 0}
            with (
                patch("oai_profile_campaign.secrets.token_hex", return_value="d" * 32),
                patch("oai_profile_campaign.remote_run", side_effect=timeout),
                patch(
                    "oai_profile_campaign.remote_control_state",
                    side_effect=[
                        (True, identity, "matching_identity_live"),
                        (False, completed_cleanup, "matching_completion"),
                    ],
                ),
                patch(
                    "oai_profile_campaign.stop_remote_controlled_action",
                    return_value=(False, "original_identity_absent"),
                ) as stop,
            ):
                result = campaign.invoke_workload_action(workload, "cleanup", 20.0)
            self.assertTrue(workload.evidence_quiesced)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(
                workload.control_results["cleanup"][
                    "remote_completion_return_code"
                ],
                0,
            )
            stop.assert_called_once()
            self.assertEqual(stop.call_args.kwargs["identity"], identity)

            for state, detail in (
                (None, "start record invalid"),
                (False, "pgid_reused_original_identity_dead"),
            ):
                with self.subTest(state=state):
                    workload = new_workload()
                    with (
                        patch(
                            "oai_profile_campaign.secrets.token_hex",
                            return_value="e" * 32,
                        ),
                        patch(
                            "oai_profile_campaign.remote_run",
                            return_value=completed,
                        ),
                        patch(
                            "oai_profile_campaign.remote_control_state",
                            return_value=(state, None, detail),
                        ),
                        patch(
                            "oai_profile_campaign.stop_remote_controlled_action"
                        ) as stop,
                    ):
                        campaign.invoke_workload_action(
                            workload,
                            "preflight",
                            20.0,
                        )
                    stop.assert_not_called()
                    self.assertIs(workload.evidence_quiesced, state is False)

    def test_unverified_remote_preflight_skips_both_archive_finalizations(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="oai-profile-unverified-preflight-"
        ) as temporary:
            root = Path(temporary)
            spec_path = root / "campaign.json"
            command = [sys.executable, "-c", GRACEFUL_SLEEP]
            write_spec(
                spec_path,
                root / "profiles",
                command,
                command,
                duration_s=120,
                workload=workload_value(root),
            )
            spec = load_spec(spec_path)
            local_client = spec["_endpoints"]["nrUE"]
            spec["_endpoints"]["nrUE"] = Endpoint(
                role=local_client.role,
                host="cm5",
                hostname=local_client.hostname,
                profile_root=local_client.profile_root,
                command=local_client.command,
                cwd=local_client.cwd,
                sudo=local_client.sudo,
                environment=local_client.environment,
                archive_tool="/remote/oai_profile_archive.py",
                launch_delay_s=local_client.launch_delay_s,
            )
            plan = build_plans(spec, set(), set(), set())[0]
            control_root = root / "control"

            def prepare_locally(handle: ProcessHandle) -> None:
                run_dir = Path(handle.run_dir)
                run_dir.mkdir(parents=True)
                (run_dir / "sidecars").mkdir()
                (run_dir / "workload").mkdir()
                handle.prepared = True
                campaign.atomic_write_json(
                    run_dir / "campaign_run.json",
                    handle.initial_manifest,
                )

            transport = subprocess.CompletedProcess(
                ["ssh"],
                255,
                "",
                "transport returned without trustworthy identity evidence",
            )
            with (
                patch("oai_profile_campaign.preflight_endpoint"),
                patch(
                    "oai_profile_campaign.prepare_run_dir",
                    side_effect=prepare_locally,
                ),
                patch(
                    "oai_profile_campaign.remote_run",
                    return_value=transport,
                ),
                patch(
                    "oai_profile_campaign.remote_control_state",
                    return_value=(None, None, "start record invalid"),
                ),
                patch("oai_profile_campaign.read_endpoint_json") as read_json,
                patch("oai_profile_campaign.read_endpoint_text") as read_text,
                patch(
                    "oai_profile_campaign.register_workload_evidence"
                ) as register_workload,
                patch("oai_profile_campaign.update_manifest") as update_manifest,
                patch("oai_profile_campaign.finalize_handle") as finalize_handle,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "workload preflight process shutdown is unverified",
                ):
                    execute_plan(spec, plan, control_root)

            rows = read_results(control_root / "campaign_results.csv")
            self.assertEqual({row["role"] for row in rows}, {"gNB", "nrUE"})
            self.assertEqual(
                {row["archive_status"] for row in rows},
                {"skipped_workload_action_unverified"},
            )
            self.assertEqual(
                {row["workload_status"] for row in rows},
                {"preflight_failed"},
            )
            self.assertEqual({row["workload_artifact"] for row in rows}, {""})
            for row in rows:
                self.assertFalse(
                    (Path(row["run_dir"]) / "archive_manifest.csv").exists()
                )
            read_json.assert_not_called()
            read_text.assert_not_called()
            register_workload.assert_not_called()
            update_manifest.assert_not_called()
            finalize_handle.assert_not_called()

    def test_post_preflight_read_failure_requires_cleanup_before_finalization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="oai-profile-preflight-read-failure-"
        ) as temporary:
            root = Path(temporary)
            spec_path = root / "campaign.json"
            command = [sys.executable, "-c", GRACEFUL_SLEEP]
            write_spec(
                spec_path,
                root / "profiles",
                command,
                command,
                duration_s=120,
                workload=workload_value(root),
            )
            spec = load_spec(spec_path)
            plan = build_plans(spec, set(), set(), set())[0]
            control_root = root / "control"
            cleanup_latches: list[bool] = []

            def successful_preflight(
                workload: campaign.WorkloadHandle,
                action: str,
                timeout_s: float,
            ) -> subprocess.CompletedProcess[str]:
                self.assertEqual(action, "preflight")
                self.assertGreater(timeout_s, 0)
                workload.synchronous_action_attempted = True
                workload.evidence_quiesced = True
                return subprocess.CompletedProcess(["workload"], 0, "", "")

            def failed_cleanup(workload: campaign.WorkloadHandle) -> None:
                cleanup_latches.append(workload.cleanup_required)
                workload.cleanup_status = "failed"
                raise RuntimeError("cleanup evidence unavailable")

            with (
                patch(
                    "oai_profile_campaign.invoke_workload_action",
                    side_effect=successful_preflight,
                ),
                patch(
                    "oai_profile_campaign.refresh_workload_state",
                    side_effect=PermissionError("evidence read blocked"),
                ),
                patch(
                    "oai_profile_campaign.cleanup_workload",
                    side_effect=failed_cleanup,
                ) as cleanup,
                patch(
                    "oai_profile_campaign.register_workload_evidence"
                ) as register_workload,
                patch("oai_profile_campaign.update_manifest") as update_manifest,
                patch("oai_profile_campaign.finalize_handle") as finalize_handle,
            ):
                with self.assertRaisesRegex(
                    PermissionError,
                    "evidence read blocked",
                ):
                    execute_plan(spec, plan, control_root)

            cleanup.assert_called_once()
            self.assertEqual(cleanup_latches, [True])
            rows = read_results(control_root / "campaign_results.csv")
            self.assertEqual({row["role"] for row in rows}, {"gNB", "nrUE"})
            self.assertEqual(
                {row["archive_status"] for row in rows},
                {"skipped_workload_cleanup_unverified"},
            )
            self.assertEqual(
                {row["network_cleanup_status"] for row in rows},
                {"failed"},
            )
            self.assertEqual({row["workload_artifact"] for row in rows}, {""})
            for row in rows:
                self.assertFalse(
                    (Path(row["run_dir"]) / "archive_manifest.csv").exists()
                )
            register_workload.assert_not_called()
            update_manifest.assert_not_called()
            finalize_handle.assert_not_called()

    def test_verified_cleanup_clears_finalization_requirement(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="oai-profile-cleanup-latch-"
        ) as temporary:
            root = Path(temporary)
            endpoint = Endpoint(
                role="nrUE",
                host="local",
                hostname="local",
                profile_root=str(root),
                command=["nr-uesoftmodem"],
                cwd=None,
                sudo=False,
                environment={},
                archive_tool=None,
                launch_delay_s=0.0,
            )
            workload = campaign.WorkloadHandle(
                spec=campaign.parse_workload_spec(workload_value(root)),
                endpoint=endpoint,
                run_dir=str(root / "run"),
                experiment_id="experiment-t001",
                cleanup_required=True,
            )

            def completed_cleanup(
                handle: campaign.WorkloadHandle,
                action: str,
                timeout_s: float,
            ) -> subprocess.CompletedProcess[str]:
                self.assertEqual(action, "cleanup")
                self.assertGreater(timeout_s, 0)
                handle.evidence_quiesced = True
                return subprocess.CompletedProcess(["workload"], 0, "", "")

            with (
                patch(
                    "oai_profile_campaign.invoke_workload_action",
                    side_effect=completed_cleanup,
                ),
                patch(
                    "oai_profile_campaign.refresh_workload_state",
                    return_value={"cleanup_status": "ok"},
                ),
            ):
                campaign.cleanup_workload(workload)

            self.assertEqual(workload.cleanup_status, "ok")
            self.assertFalse(workload.cleanup_required)

    def test_controlled_action_escalation_revalidates_every_signal(self) -> None:
        endpoint = Endpoint(
            role="nrUE",
            host="cm5",
            hostname="cm5",
            profile_root="/profiles",
            command=["nr-uesoftmodem"],
            cwd=None,
            sudo=True,
            environment={},
            archive_tool="/repo/tools/profiling/oai_profile_archive.py",
            launch_delay_s=0.0,
        )
        identity = {
            "schema_version": 1,
            "action": "cleanup",
            "experiment_id": "experiment-t001",
            "token": "f" * 32,
            "pgid": 4321,
            "start_ticks": 987654,
        }
        signal_result = subprocess.CompletedProcess(["ssh"], 0, "", "")
        with (
            patch(
                "oai_profile_campaign.remote_run",
                return_value=signal_result,
            ) as remote,
            patch(
                "oai_profile_campaign.wait_remote_control_not_live",
                side_effect=[
                    (True, identity, "matching_identity_live"),
                    (True, identity, "matching_identity_live"),
                    (False, identity, "original_identity_absent"),
                ],
            ),
        ):
            state, detail = campaign.stop_remote_controlled_action(
                endpoint,
                action="cleanup",
                experiment_id="experiment-t001",
                token="f" * 32,
                start_path="/profiles/run/workload/cleanup.control.start",
                completion_path="/profiles/run/workload/cleanup.control.complete",
                identity=identity,
            )
        self.assertFalse(state)
        self.assertIn("original_identity_absent", detail)
        commands = [call.args[1] for call in remote.call_args_list]
        self.assertEqual(len(commands), 3)
        self.assertIn("sudo -n /bin/kill -INT -- -4321", commands[0])
        self.assertIn("sudo -n /bin/kill -TERM -- -4321", commands[1])
        self.assertIn("sudo -n /bin/kill -KILL -- -4321", commands[2])
        self.assertTrue(
            all("start_ticks=987654" in command for command in commands)
        )

        term_failure = subprocess.CompletedProcess(
            ["ssh"],
            44,
            "",
            "synthetic TERM failure",
        )
        with (
            patch(
                "oai_profile_campaign.remote_run",
                side_effect=[OSError("synthetic INT failure"), term_failure, signal_result],
            ) as remote,
            patch(
                "oai_profile_campaign.wait_remote_control_not_live",
                side_effect=[
                    (None, identity, "identity probe unavailable"),
                    (True, identity, "matching_identity_live"),
                    (False, identity, "original_identity_absent"),
                ],
            ) as wait,
        ):
            state, detail = campaign.stop_remote_controlled_action(
                endpoint,
                action="cleanup",
                experiment_id="experiment-t001",
                token="f" * 32,
                start_path="/profiles/run/workload/cleanup.control.start",
                completion_path="/profiles/run/workload/cleanup.control.complete",
                identity=identity,
            )
        self.assertFalse(state)
        self.assertEqual(remote.call_count, 3)
        self.assertEqual(wait.call_count, 3)
        self.assertIn("synthetic INT failure", detail)
        self.assertIn("synthetic TERM failure", detail)
        self.assertIn("original_identity_absent", detail)

    def test_asynchronous_workload_uses_bound_identity_and_completion(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-async-workload-") as temporary:
            root = Path(temporary)
            endpoint = Endpoint(
                role="nrUE",
                host="cm5",
                hostname="cm5",
                profile_root="/profiles",
                command=["nr-uesoftmodem"],
                cwd=None,
                sudo=True,
                environment={},
                archive_tool="/repo/tools/profiling/oai_profile_archive.py",
                launch_delay_s=0.0,
            )
            workload = campaign.WorkloadHandle(
                spec=campaign.parse_workload_spec(workload_value(root)),
                endpoint=endpoint,
                run_dir="/profiles/run",
                experiment_id="experiment-t001",
            )
            process = SimpleNamespace(
                poll=lambda: None,
                pid=7654,
                returncode=None,
            )
            with (
                patch("oai_profile_campaign.secrets.token_hex", return_value="1" * 32),
                patch(
                    "oai_profile_campaign.subprocess.Popen",
                    return_value=process,
                ) as popen,
            ):
                campaign.launch_workload(workload, root / "control")
            remote_shell = popen.call_args.args[0][-1]
            self.assertIn("run.control.start", remote_shell)
            self.assertIn("run.control.complete", remote_shell)
            self.assertIn("token=" + "1" * 32, remote_shell)
            self.assertEqual(workload.control_tokens, {"run": "1" * 32})
            assert workload.stdout_handle is not None
            assert workload.stderr_handle is not None
            workload.stdout_handle.close()
            workload.stderr_handle.close()

            identity = {
                "schema_version": 1,
                "action": "run",
                "experiment_id": "experiment-t001",
                "token": "1" * 32,
                "pgid": 4321,
                "start_ticks": 987654,
            }
            signal_result = subprocess.CompletedProcess(["ssh"], 0, "", "")
            with (
                patch(
                    "oai_profile_campaign.remote_control_state",
                    return_value=(True, identity, "matching_identity_live"),
                ),
                patch(
                    "oai_profile_campaign.remote_run",
                    return_value=signal_result,
                ) as remote,
                patch("oai_profile_campaign.os.killpg") as local_signal,
            ):
                campaign.signal_workload(workload, "INT")
            self.assertIn(
                "sudo -n /bin/kill -INT -- -4321",
                remote.call_args.args[1],
            )
            local_signal.assert_not_called()

            with (
                patch(
                    "oai_profile_campaign.remote_control_state",
                    return_value=(None, identity, "completion record invalid"),
                ),
                patch("oai_profile_campaign.remote_run") as remote,
                patch("oai_profile_campaign.os.killpg") as local_signal,
            ):
                campaign.signal_workload(workload, "TERM")
            remote.assert_not_called()
            local_signal.assert_not_called()

            with (
                patch(
                    "oai_profile_campaign.remote_control_state",
                    return_value=(
                        False,
                        identity,
                        "pgid_reused_original_identity_dead",
                    ),
                ),
                patch("oai_profile_campaign.remote_run") as remote,
                patch("oai_profile_campaign.os.killpg") as local_signal,
            ):
                campaign.signal_workload(workload, "TERM")
            remote.assert_not_called()
            local_signal.assert_called_once_with(7654, signal.SIGTERM)

            process.poll = lambda: 255
            process.returncode = 255
            completed_identity = {**identity, "return_code": 0}
            with patch(
                "oai_profile_campaign.remote_control_state",
                return_value=(False, completed_identity, "matching_completion"),
            ):
                state, _, _ = campaign.workload_remote_control_state(workload)
            self.assertFalse(state)
            self.assertEqual(
                workload.control_results["run"],
                {
                    "transport_status": "returned",
                    "transport_return_code": 255,
                    "remote_completion_return_code": 0,
                },
            )

    def test_stop_handles_signals_reverse_launch_order(self) -> None:
        stopped_process = SimpleNamespace(poll=lambda: 0)
        local_endpoint = SimpleNamespace(remote=False)
        gnb_handle = SimpleNamespace(
            stop_reason="",
            process=stopped_process,
            endpoint=local_endpoint,
            shutdown_verified=False,
            notes=[],
        )
        nrue_handle = SimpleNamespace(
            stop_reason="",
            process=stopped_process,
            endpoint=local_endpoint,
            shutdown_verified=False,
            notes=[],
        )
        handles = [gnb_handle, nrue_handle]
        with (
            patch("oai_profile_campaign.shutdown_target_is_running", return_value=True),
            patch(
                "oai_profile_campaign.signal_handle",
                side_effect=[True] * 6,
            ) as signal,
            patch("oai_profile_campaign.wait_handles", side_effect=[False, False, True]),
            patch("oai_profile_campaign.role_shutdown_is_verified", return_value=True),
        ):
            stopped = campaign.stop_handles(handles, "duration_elapsed", 1.0)

        self.assertTrue(stopped)
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
        for handle in handles:
            self.assertTrue(handle.controlled_stop_requested)
            self.assertEqual(handle.shutdown_stage, "SIGKILL")

    def test_local_shutdown_escalates_for_live_group_descendant(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="oai-profile-local-descendant-"
        ) as temporary:
            marker = Path(temporary) / "child-ready"
            script = "\n".join(
                (
                    "import os, signal, time",
                    "pid = os.fork()",
                    "if pid == 0:",
                    "    signal.signal(signal.SIGINT, signal.SIG_IGN)",
                    f"    open({str(marker)!r}, 'w').write(str(os.getpid()))",
                    "    while True: time.sleep(30)",
                    "signal.signal(signal.SIGINT, signal.SIG_DFL)",
                    "while True: time.sleep(30)",
                )
            )
            process = subprocess.Popen(
                [sys.executable, "-c", script],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            endpoint = Endpoint(
                role="gNB",
                host="local",
                hostname="local",
                profile_root=temporary,
                command=[sys.executable, "-c", script],
                cwd=None,
                sudo=False,
                environment={},
                archive_tool=None,
                launch_delay_s=0.0,
            )
            handle = ProcessHandle(
                endpoint=endpoint,
                run_dir=temporary,
                command=endpoint.command,
                environment={},
                process=process,
                run_status="running",
            )
            try:
                deadline = time.monotonic() + 5.0
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(marker.exists())
                self.assertTrue(
                    campaign.stop_handles([handle], "duration_elapsed", 0.25)
                )
                campaign.close_handle(handle)
                self.assertEqual(handle.shutdown_stage, "SIGTERM")
                self.assertTrue(handle.shutdown_verified)
                self.assertFalse(campaign.local_process_group_is_running(process))
                self.assertFalse(
                    campaign.campaign_result_process_succeeded(
                        {
                            **campaign.completion_evidence(
                                handle,
                                status_field="run_status",
                            ),
                            "host": "local",
                            "run_status": handle.run_status,
                            "return_code": campaign.authoritative_return_code(handle),
                            "stop_reason": handle.stop_reason,
                        }
                    )
                )
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)

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
                experiment_id="experiment-t001",
            )
            with (
                patch(
                    "oai_profile_campaign.secrets.token_hex",
                    return_value="2" * 32,
                ),
                patch(
                    "oai_profile_campaign.subprocess.Popen",
                    return_value=object(),
                ) as popen,
            ):
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
            start = campaign.parse_control_record(
                (run_dir / "control.start").read_text(),
                action="role:nrUE",
                experiment_id="experiment-t001",
                token="2" * 32,
                completion=False,
            )
            completion = campaign.parse_control_record(
                (run_dir / "control.complete").read_text(),
                action="role:nrUE",
                experiment_id="experiment-t001",
                token="2" * 32,
                completion=True,
            )
            self.assertEqual(completion["return_code"], 7)
            self.assertEqual(completion["pgid"], start["pgid"])
            self.assertEqual(completion["start_ticks"], start["start_ticks"])
            with self.assertRaises(ProcessLookupError):
                os.kill(start["pgid"], 0)

    def test_remote_control_wrapper_records_group_signal_completion(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-remote-signal-") as temporary:
            root = Path(temporary)
            start_path = root / "control.start"
            completion_path = root / "control.complete"
            ready_path = root / "payload-ready"
            payload = (
                "import pathlib,signal,sys,time;"
                "signal.signal(signal.SIGINT,lambda *_:sys.exit(0));"
                f"pathlib.Path({str(ready_path)!r}).write_text('ready');"
                "time.sleep(30)"
            )
            wrapper = campaign.remote_control_inner_command(
                [sys.executable, "-c", payload],
                action="role:nrUE",
                experiment_id="experiment-t001",
                token="3" * 32,
                start_path=str(start_path),
                completion_path=str(completion_path),
            )
            process = subprocess.Popen(
                ["setsid", "--wait", "sh", "-c", wrapper],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            start = None
            try:
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    if start_path.exists() and ready_path.exists():
                        start = campaign.parse_control_record(
                            start_path.read_text(),
                            action="role:nrUE",
                            experiment_id="experiment-t001",
                            token="3" * 32,
                            completion=False,
                        )
                        break
                    if process.poll() is not None:
                        break
                    time.sleep(0.01)
                self.assertIsNotNone(start)
                assert start is not None
                os.killpg(start["pgid"], signal.SIGINT)
                stdout, stderr = process.communicate(timeout=5.0)
            finally:
                if process.poll() is None:
                    if start is not None:
                        os.killpg(start["pgid"], signal.SIGKILL)
                    process.kill()
                    process.communicate()
            self.assertEqual((stdout, stderr), ("", ""))
            self.assertEqual(process.returncode, 0)
            completion = campaign.parse_control_record(
                completion_path.read_text(),
                action="role:nrUE",
                experiment_id="experiment-t001",
                token="3" * 32,
                completion=True,
            )
            self.assertEqual(completion["return_code"], 0)
            self.assertEqual(completion["pgid"], start["pgid"])
            self.assertEqual(completion["start_ticks"], start["start_ticks"])

    def test_emulated_remote_sigint_lifecycle_is_archived_and_accepted(self) -> None:
        cases = (
            (
                "signal",
                "",
                128 + signal.SIGINT,
                "controlled_sigint_signal",
            ),
            (
                "graceful",
                "signal.signal(signal.SIGINT,lambda *_:sys.exit(0));",
                0,
                "controlled_sigint_zero",
            ),
        )
        for index, (name, signal_setup, expected_return, expected_class) in enumerate(
            cases,
            start=5,
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory(
                prefix=f"oai-profile-remote-{name}-"
            ) as temporary:
                root = Path(temporary)
                run_dir = root / "run"
                run_dir.mkdir()
                ready_path = root / "payload.ready"
                payload = (
                    "import pathlib,signal,sys,time;"
                    f"{signal_setup}"
                    f"pathlib.Path({str(ready_path)!r}).write_text('ready');"
                    "time.sleep(30)"
                )
                endpoint = Endpoint(
                    role="nrUE",
                    host="remote-emulated",
                    hostname="remote-emulated",
                    profile_root=str(root),
                    command=[sys.executable, "-c", payload],
                    cwd=None,
                    sudo=False,
                    environment={},
                    archive_tool=str(
                        Path(__file__).resolve().parents[1]
                        / "oai_profile_archive.py"
                    ),
                    launch_delay_s=0.0,
                )
                plan = ExperimentPlan(
                    "campaign-test",
                    "baseline",
                    Variant("disabled", False, "off", {"tool": "none"}, {}),
                    1,
                    f"campaign-test-remote-{name}-t001",
                    {},
                    {},
                )
                token = str(index) * 32
                handle = ProcessHandle(
                    endpoint=endpoint,
                    run_dir=str(run_dir),
                    command=endpoint.command,
                    environment={},
                    start_realtime_ns=time.time_ns(),
                    start_monotonic_ns=time.clock_gettime_ns(
                        time.CLOCK_MONOTONIC_RAW
                    ),
                    experiment_id=plan.experiment_id,
                    control_token=token,
                    prepared=True,
                    run_status="running",
                )
                handle.initial_manifest = campaign.run_manifest(
                    endpoint,
                    plan,
                    str(run_dir),
                    endpoint.command,
                    {},
                )
                campaign.atomic_write_json(
                    run_dir / "campaign_run.json",
                    handle.initial_manifest,
                )
                wrapper = campaign.remote_control_inner_command(
                    endpoint.command,
                    action=handle.control_action,
                    experiment_id=handle.experiment_id,
                    token=handle.control_token,
                    start_path=handle.control_start_path,
                    completion_path=handle.control_completion_path,
                )
                process = subprocess.Popen(
                    ["setsid", "--wait", "sh", "-c", wrapper],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                handle.process = process

                def run_locally(
                    endpoint_argument: Endpoint,
                    command: str,
                    check: bool = True,
                    capture: bool = False,
                    timeout_s: float | None = campaign.REMOTE_CONTROL_TIMEOUT_S,
                ) -> subprocess.CompletedProcess[str]:
                    del endpoint_argument
                    return subprocess.run(
                        ["sh", "-c", command],
                        check=check,
                        text=True,
                        capture_output=capture,
                        timeout=timeout_s,
                    )

                def upload_locally(
                    endpoint_argument: Endpoint,
                    local_path: Path,
                    remote_path: str,
                ) -> None:
                    del endpoint_argument
                    destination = Path(remote_path)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(local_path, destination)

                start_record: dict[str, object] | None = None
                try:
                    deadline = time.monotonic() + 5.0
                    while time.monotonic() < deadline:
                        if (
                            Path(handle.control_start_path).is_file()
                            and ready_path.is_file()
                        ):
                            start_record = campaign.parse_control_record(
                                Path(handle.control_start_path).read_text(),
                                action=handle.control_action,
                                experiment_id=handle.experiment_id,
                                token=handle.control_token,
                                completion=False,
                            )
                            break
                        if process.poll() is not None:
                            break
                        time.sleep(0.01)
                    self.assertIsNotNone(start_record)
                    with (
                        patch(
                            "oai_profile_campaign.remote_run",
                            side_effect=run_locally,
                        ),
                        patch(
                            "oai_profile_campaign.remote_upload",
                            side_effect=upload_locally,
                        ),
                    ):
                        self.assertTrue(
                            campaign.stop_handles(
                                [handle],
                                "duration_elapsed",
                                2.0,
                            )
                        )
                        campaign.close_handle(handle)
                        campaign.update_manifest(handle)
                        campaign.finalize_handle(handle)
                finally:
                    if process.poll() is None:
                        if start_record is not None:
                            try:
                                os.killpg(
                                    int(start_record["pgid"]),
                                    signal.SIGKILL,
                                )
                            except OSError:
                                pass
                        process.kill()
                    process.communicate(timeout=5.0)

                completion = campaign.parse_control_record(
                    Path(handle.control_completion_path).read_text(),
                    action=handle.control_action,
                    experiment_id=handle.experiment_id,
                    token=handle.control_token,
                    completion=True,
                )
                self.assertEqual(completion["return_code"], expected_return)
                self.assertEqual(handle.remote_completion_return_code, expected_return)
                self.assertTrue(handle.remote_completion_identity_verified)
                self.assertEqual(handle.shutdown_stage, "SIGINT")
                self.assertTrue(handle.shutdown_verified)
                self.assertEqual(handle.archive_status, "finalized")
                row = campaign.result_row(handle, plan)
                self.assertEqual(row["return_code"], expected_return)
                self.assertEqual(row["termination_class"], expected_class)
                self.assertFalse(experiment_failed([row], {"nrUE"}))
                self.assertEqual(
                    archive_identity(run_dir)["archive_state"],
                    "runner_completed_controlled_sigint",
                )
                self.assertTrue(
                    all(result.valid for result in verify_archive(run_dir))
                )

    def test_remote_completion_waits_for_live_process_group_descendant(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-remote-descendant-") as temporary:
            root = Path(temporary)
            start_path = root / "control.start"
            completion_path = root / "control.complete"
            child_path = root / "child.pid"
            payload = f"""
import os
import pathlib
import time

child_pid = os.fork()
if child_pid == 0:
    null_fd = os.open(os.devnull, os.O_RDWR)
    for standard_fd in (0, 1, 2):
        os.dup2(null_fd, standard_fd)
    if null_fd > 2:
        os.close(null_fd)
    pathlib.Path({str(child_path)!r}).write_text(str(os.getpid()))
    time.sleep(30)
    os._exit(0)
os._exit(0)
"""
            wrapper = campaign.remote_control_inner_command(
                [sys.executable, "-c", payload],
                action="role:nrUE",
                experiment_id="experiment-t001",
                token="4" * 32,
                start_path=str(start_path),
                completion_path=str(completion_path),
            )
            process = subprocess.Popen(
                ["setsid", "--wait", "sh", "-c", wrapper],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            start = None
            try:
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    if start_path.exists() and completion_path.exists() and child_path.exists():
                        start = campaign.parse_control_record(
                            start_path.read_text(),
                            action="role:nrUE",
                            experiment_id="experiment-t001",
                            token="4" * 32,
                            completion=False,
                        )
                        break
                    if process.poll() is not None and not child_path.exists():
                        break
                    time.sleep(0.01)
                self.assertIsNotNone(start)
                assert start is not None
                stdout, stderr = process.communicate(timeout=5.0)
                self.assertEqual((stdout, stderr), ("", ""))
                self.assertEqual(process.returncode, 0)
                completion = campaign.parse_control_record(
                    completion_path.read_text(),
                    action="role:nrUE",
                    experiment_id="experiment-t001",
                    token="4" * 32,
                    completion=True,
                )
                self.assertEqual(completion["return_code"], 0)
                self.assertEqual(completion["pgid"], start["pgid"])
                self.assertEqual(completion["start_ticks"], start["start_ticks"])
                self.assertFalse(Path(f"/proc/{start['pgid']}/stat").exists())
                os.killpg(start["pgid"], 0)

                def run_locally(
                    endpoint: Endpoint,
                    command: str,
                    check: bool = True,
                    capture: bool = False,
                    timeout_s: float | None = campaign.REMOTE_CONTROL_TIMEOUT_S,
                ) -> subprocess.CompletedProcess[str]:
                    del endpoint
                    return subprocess.run(
                        ["sh", "-c", command],
                        check=check,
                        text=True,
                        capture_output=capture,
                        timeout=timeout_s,
                    )

                endpoint = Endpoint(
                    role="nrUE",
                    host="local-test",
                    hostname="local-test",
                    profile_root=str(root),
                    command=["nr-uesoftmodem"],
                    cwd=None,
                    sudo=False,
                    environment={},
                    archive_tool=str(root / "archive-tool"),
                    launch_delay_s=0.0,
                )
                with patch(
                    "oai_profile_campaign.remote_run",
                    side_effect=run_locally,
                ) as remote:
                    state, identity, detail = campaign.remote_control_state(
                        endpoint,
                        action="role:nrUE",
                        experiment_id="experiment-t001",
                        token="4" * 32,
                        start_path=str(start_path),
                        completion_path=str(completion_path),
                    )
                self.assertTrue(state, detail)
                self.assertEqual(identity, completion)
                self.assertEqual(
                    detail,
                    "matching_completion_leader_absent_process_group_still_live",
                )
                self.assertEqual(remote.call_count, 3)
            finally:
                cleanup_start = start
                if cleanup_start is None and start_path.exists():
                    try:
                        cleanup_start = campaign.parse_control_record(
                            start_path.read_text(),
                            action="role:nrUE",
                            experiment_id="experiment-t001",
                            token="4" * 32,
                            completion=False,
                        )
                    except (OSError, ValueError):
                        pass
                if cleanup_start is not None:
                    try:
                        os.killpg(cleanup_start["pgid"], signal.SIGKILL)
                    except OSError:
                        pass
                elif child_path.exists():
                    try:
                        child_pid_text = child_path.read_text().strip()
                        if not child_pid_text.isdecimal() or int(child_pid_text) <= 0:
                            raise ValueError("invalid test child PID")
                        os.kill(int(child_pid_text), signal.SIGKILL)
                    except (OSError, ValueError):
                        pass
                if process.poll() is None:
                    process.kill()
                process.communicate()

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
            process=SimpleNamespace(
                poll=lambda: 255,
                returncode=255,
                pid=7654,
            ),
            experiment_id="experiment-t001",
            control_token="3" * 32,
        )
        identity = {
            "schema_version": 1,
            "action": "role:nrUE",
            "experiment_id": "experiment-t001",
            "token": "3" * 32,
            "pgid": 4321,
            "start_ticks": 987654,
        }
        completed = subprocess.CompletedProcess(["ssh"], 0, "", "")
        with (
            patch(
                "oai_profile_campaign.remote_control_state",
                return_value=(True, identity, "matching_identity_live"),
            ),
            patch("oai_profile_campaign.remote_run", return_value=completed) as remote,
        ):
            self.assertTrue(remote_group_is_running(handle))
            signal_handle(handle, "INT")
        commands = [call.args[1] for call in remote.call_args_list]
        self.assertEqual(len(commands), 1)
        self.assertIn("/bin/kill -INT", commands[0])
        self.assertIn("start_ticks=987654", commands[0])
        self.assertEqual(handle.transport_return_code, 255)

    def test_remote_group_control_is_tri_state(self) -> None:
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
        identity = {
            "schema_version": 1,
            "action": "role:nrUE",
            "experiment_id": "experiment-t001",
            "token": "4" * 32,
            "pgid": 4321,
            "start_ticks": 987654,
        }
        for state, expected in ((True, True), (False, False), (None, None)):
            with self.subTest(state=state):
                handle = ProcessHandle(
                    endpoint=endpoint,
                    run_dir="/profiles/run",
                    command=endpoint.command,
                    environment={},
                    process=SimpleNamespace(
                        poll=lambda: 255,
                        returncode=255,
                    ),
                    experiment_id="experiment-t001",
                    control_token="4" * 32,
                )
                with patch(
                    "oai_profile_campaign.remote_control_state",
                    return_value=(state, identity, "synthetic remote state"),
                ):
                    self.assertIs(remote_group_is_running(handle), expected)
                if expected is None:
                    self.assertIn("state unavailable", handle.notes[0])

    def test_remote_role_completion_is_authoritative_over_transport(self) -> None:
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
        variant = Variant("in-process", True, "off", {"tool": "none"}, {})
        plan = ExperimentPlan(
            "campaign-test",
            "baseline",
            variant,
            1,
            "experiment-t001",
            {},
            {},
        )
        process = SimpleNamespace(
            poll=lambda: 255,
            returncode=255,
            wait=lambda: 255,
            pid=7654,
        )
        handle = ProcessHandle(
            endpoint=endpoint,
            run_dir="/profiles/run",
            command=endpoint.command,
            environment={},
            process=process,
            experiment_id="experiment-t001",
            control_token="5" * 32,
        )
        completion = {
            "schema_version": 1,
            "action": "role:nrUE",
            "experiment_id": "experiment-t001",
            "token": "5" * 32,
            "pgid": 4321,
            "start_ticks": 987654,
            "return_code": 0,
        }
        with patch(
            "oai_profile_campaign.remote_control_state",
            return_value=(False, completion, "matching_completion"),
        ):
            campaign.close_handle(handle)
        row = campaign.result_row(handle, plan)
        self.assertEqual(handle.run_status, "finished")
        self.assertEqual(row["transport_return_code"], 255)
        self.assertEqual(row["remote_completion_return_code"], 0)
        self.assertEqual(row["return_code"], 0)

        unverified = ProcessHandle(
            endpoint=endpoint,
            run_dir="/profiles/run2",
            command=endpoint.command,
            environment={},
            process=process,
            experiment_id="experiment-t002",
            control_token="6" * 32,
        )
        absent_identity = {
            **completion,
            "experiment_id": "experiment-t002",
            "token": "6" * 32,
        }
        absent_identity.pop("return_code")
        with patch(
            "oai_profile_campaign.remote_control_state",
            return_value=(False, absent_identity, "original_identity_absent"),
        ):
            campaign.close_handle(unverified)
        row = campaign.result_row(unverified, plan)
        self.assertEqual(unverified.run_status, "remote_return_unverified")
        self.assertEqual(row["transport_return_code"], 255)
        self.assertEqual(row["remote_completion_return_code"], "")
        self.assertEqual(row["return_code"], "")

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
        profiling_root = Path(__file__).resolve().parents[1]
        example = profiling_root / "campaign_laptop_cm5.example.json"
        spec = load_spec(example)
        plans = build_plans(spec, set(), set(), set())

        self.assertEqual(spec["campaign_id"], "band28-25prb-cm5-baseline")
        self.assertEqual(len(plans), 35)
        self.assertEqual(spec["_start_order"], ["gNB", "nrUE"])
        self.assertEqual(
            {plan.variant.name for plan in plans},
            {"disabled", "in-process", "pmu-software", "pmu-all", "perf-stat", "perf-record", "perf-sched"},
        )

        loaded_example = (
            profiling_root / "campaign_laptop_cm5.loaded.example.json"
        )
        loaded_spec = load_spec(loaded_example)
        loaded_plans = build_plans(loaded_spec, set(), set(), set())
        self.assertEqual(
            loaded_spec["campaign_id"],
            "band28-25prb-cm5-loaded",
        )
        self.assertEqual(len(loaded_plans), 35)
        workload = loaded_spec["_workload"]
        self.assertIsNotNone(workload)
        assert workload is not None
        self.assertEqual(workload.client_role, "nrUE")
        self.assertEqual(workload.interface, "oaitun_ue1")
        self.assertEqual(str(workload.subnet), "10.0.0.0/24")
        self.assertEqual(str(workload.server), "192.168.70.135")
        self.assertEqual(workload.policy_table, 9999)
        self.assertEqual(workload.duration_s, 120)
        self.assertEqual(workload.bitrate_bps, 1_000_000)
        self.assertEqual(workload.datagram_bytes, 1200)
        loaded_json = json.loads(loaded_example.read_text())
        self.assertNotIn(
            "UHD_IMAGES_DIR",
            json.dumps(loaded_json, sort_keys=True),
        )
        self.assertEqual(
            loaded_json["roles"],
            json.loads(example.read_text())["roles"],
        )

    def test_loaded_workload_contract_and_dry_run_are_side_effect_free(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-profile-workload-plan-") as temporary:
            root = Path(temporary)
            profile_root = root / "profiles"
            spec_path = root / "campaign.json"
            command = [sys.executable, "-c", "pass"]
            workload = workload_value(root)
            write_spec(
                spec_path,
                profile_root,
                command,
                command,
                duration_s=120,
                workload=workload,
            )
            spec = load_spec(spec_path)
            plan = build_plans(spec, set(), set(), set())[0]
            view = plan_view(spec, plan)

            self.assertEqual(len(view["roles"]), 2)
            self.assertEqual(set(view["roles"]), {"gNB", "nrUE"})
            self.assertEqual(view["workload"]["client_role"], "nrUE")
            self.assertTrue(view["workload"]["role_count_unchanged"])
            self.assertEqual(
                view["workload"]["contract"]["traffic_contract"],
                {
                    "protocol": "UDP",
                    "direction": "bidirectional",
                    "bitrate_bps_per_direction": 1_000_000,
                    "datagram_bytes": 1200,
                    "post_readiness_duration_s": 120,
                },
            )
            self.assertFalse(profile_root.exists())
            self.assertFalse(Path(str(workload["lease_path"])).exists())

            mismatched = json.loads(spec_path.read_text())
            mismatched["duration_s"] = 119
            spec_path.write_text(json.dumps(mismatched))
            with self.assertRaisesRegex(ValueError, "exactly equal"):
                load_spec(spec_path)

            unknown = json.loads(json.dumps(workload))
            unknown["unexpected"] = True
            write_spec(
                spec_path,
                profile_root,
                command,
                command,
                duration_s=120,
                workload=unknown,
            )
            with self.assertRaisesRegex(ValueError, "unknown workload fields"):
                load_spec(spec_path)

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
            perf_record_command, artifact = sidecar_command(
                command,
                {
                    "tool": "perf_record",
                    "event": "cycles",
                    "frequency_hz": 199,
                    "call_graph": "dwarf,8192",
                },
                str(root),
            )
            self.assertEqual(artifact, f"{root}/sidecars/perf.data")
            self.assertEqual(
                perf_record_command,
                [
                    "perf",
                    "record",
                    "-o",
                    artifact,
                    "-e",
                    "cycles",
                    "-F",
                    "199",
                    "--call-graph",
                    "dwarf,8192",
                    "--clockid",
                    "CLOCK_MONOTONIC_RAW",
                    "--timestamp",
                    "--",
                    *command,
                ],
            )
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
                self.assertEqual(row["workload_status"], "not_configured")
                self.assertEqual(row["workload_artifact"], "")
                self.assertEqual(row["controlled_stop_requested"], 1)
                self.assertEqual(row["shutdown_stage"], "SIGINT")
                self.assertEqual(row["shutdown_verified"], 1)
                self.assertEqual(row["termination_class"], "controlled_sigint_zero")
                self.assertEqual(
                    row["network_cleanup_status"],
                    "not_configured",
                )
                run_dir = Path(row["run_dir"])
                self.assertTrue(all(result.valid for result in verify_archive(run_dir)))
                campaign = json.loads((run_dir / "campaign_run.json").read_text())
                self.assertEqual(campaign["status"], "finished")
                self.assertEqual(campaign["archive_status"], "finalization_pending")
                self.assertEqual(campaign["experiment_id"], plan.experiment_id)
                self.assertEqual(campaign["workload_status"], "not_configured")
                self.assertEqual(campaign["workload_artifact"], "")
                self.assertEqual(
                    campaign["network_cleanup_status"],
                    "not_configured",
                )

            results = read_results(control_root / "campaign_results.csv")
            self.assertEqual(len(results), 2)
            self.assertEqual({row["run_status"] for row in results}, {"finished"})

    def test_local_sigint_exit_is_preserved_and_accepted(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="oai-profile-campaign-controlled-int-"
        ) as temporary:
            root = Path(temporary)
            spec_path = root / "campaign.json"
            command = [sys.executable, "-c", DEFAULT_SIGINT_SLEEP]
            write_spec(spec_path, root / "profiles", command, command)

            spec = load_spec(spec_path)
            plan = build_plans(spec, set(), set(), set())[0]
            control_root = root / "control"
            rows = execute_plan(spec, plan, control_root)

            self.assertFalse(experiment_failed(rows, {"gNB", "nrUE"}))
            for row in rows:
                self.assertEqual(row["run_status"], "exited_nonzero")
                self.assertEqual(row["return_code"], -signal.SIGINT)
                self.assertEqual(row["stop_reason"], "duration_elapsed")
                self.assertEqual(row["shutdown_stage"], "SIGINT")
                self.assertEqual(row["shutdown_verified"], 1)
                self.assertEqual(
                    row["termination_class"],
                    "controlled_sigint_signal",
                )
                self.assertEqual(row["archive_status"], "finalized")
                run_dir = Path(row["run_dir"])
                self.assertTrue(
                    all(result.valid for result in verify_archive(run_dir))
                )
                manifest = json.loads((run_dir / "campaign_run.json").read_text())
                self.assertEqual(manifest["status"], "exited_nonzero")
                self.assertEqual(manifest["return_code"], -signal.SIGINT)
                self.assertEqual(
                    manifest["termination_class"],
                    "controlled_sigint_signal",
                )

    def test_experiment_failure_requires_complete_duration(self) -> None:
        successful = {
            "role": "gNB",
            "host": "local",
            "archive_status": "finalized",
            "run_status": "finished",
            "return_code": 0,
            "stop_reason": "duration_elapsed",
            "duration_status": "valid",
            "completion_classifier_version": "1",
            "controlled_stop_requested": 1,
            "shutdown_stage": "SIGINT",
            "shutdown_verified": 1,
            "remote_completion_identity_verified": "not_applicable",
            "remote_completion_return_code": "",
            "termination_class": "controlled_sigint_zero",
            "profile_enabled": 1,
            "pmu_mode": "off",
            "sidecar": "none",
            "sidecar_status": "not_requested",
            "sidecar_artifact": "",
            "workload_status": "not_configured",
            "workload_artifact": "",
            "network_cleanup_status": "not_configured",
        }
        self.assertFalse(experiment_failed([successful], {"gNB"}))
        self.assertTrue(
            experiment_failed(
                [{**successful, "sidecar_status": ""}],
                {"gNB"},
            )
        )
        legacy_no_sidecar = {
            key: value
            for key, value in successful.items()
            if key
            not in {
                "completion_classifier_version",
                "controlled_stop_requested",
                "shutdown_stage",
                "shutdown_verified",
                "remote_completion_identity_verified",
                "termination_class",
            }
        }
        legacy_no_sidecar["sidecar_status"] = ""
        self.assertFalse(experiment_failed([legacy_no_sidecar], {"gNB"}))
        legacy_offline_sidecar = {
            **legacy_no_sidecar,
            "sidecar_tool": legacy_no_sidecar.pop("sidecar"),
        }
        self.assertTrue(
            reports.campaign_sidecar_evidence_succeeded(legacy_offline_sidecar)
        )
        self.assertFalse(
            reports.campaign_sidecar_evidence_succeeded(
                {**successful, "sidecar_status": ""}
            )
        )
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
        wrong_legacy_stop = {
            **successful,
            "stop_reason": "measurement_complete",
        }
        self.assertTrue(experiment_failed([wrong_legacy_stop], {"gNB"}))
        inconsistent_legacy = {
            **successful,
            "workload_status": "not_configured",
            "workload_artifact": "workload/workload_run.json",
            "network_cleanup_status": "not_configured",
        }
        self.assertTrue(experiment_failed([inconsistent_legacy], {"gNB"}))

        loaded = {
            **successful,
            "stop_reason": "measurement_complete",
            "workload_status": "completed",
            "workload_artifact": "workload/workload_run.json",
            "network_cleanup_status": "ok",
        }
        loaded_peer = {**loaded, "role": "nrUE"}
        self.assertFalse(experiment_failed([loaded, loaded_peer], {"gNB", "nrUE"}))
        sidecar_loaded = {
            **loaded,
            "sidecar": "perf_stat",
            "sidecar_status": "registered",
            "sidecar_artifact": "/profiles/run/sidecars/perf_stat.csv",
        }
        self.assertFalse(experiment_failed([sidecar_loaded], {"gNB"}))
        self.assertTrue(
            experiment_failed(
                [{**sidecar_loaded, "sidecar_artifact": ""}],
                {"gNB"},
            )
        )
        controlled_loaded = {
            **loaded,
            "run_status": "exited_nonzero",
            "return_code": -signal.SIGINT,
            "termination_class": "controlled_sigint_signal",
            "termination_class_recorded": "controlled_sigint_signal",
            "termination_class_recomputed": "controlled_sigint_signal",
            "termination_class_consistent": 1,
        }
        controlled_remote = {
            **loaded_peer,
            "host": "cm5",
            "run_status": "exited_nonzero",
            "return_code": 128 + signal.SIGINT,
            "remote_completion_return_code": 128 + signal.SIGINT,
            "remote_completion_identity_verified": 1,
            "termination_class": "controlled_sigint_signal",
        }
        self.assertFalse(
            experiment_failed(
                [controlled_loaded, controlled_remote],
                {"gNB", "nrUE"},
            )
        )
        for tampered in (
            {**controlled_loaded, "shutdown_stage": "SIGTERM"},
            {**controlled_loaded, "shutdown_verified": 0},
            {**controlled_loaded, "controlled_stop_requested": 0},
            {**controlled_loaded, "completion_classifier_version": ""},
            {**controlled_loaded, "return_code": 130},
            {**controlled_remote, "remote_completion_identity_verified": 0},
            {**controlled_remote, "remote_completion_return_code": ""},
            {**controlled_remote, "remote_completion_return_code": 0},
            {**controlled_remote, "return_code": -signal.SIGINT},
            {**controlled_remote, "stop_reason": "interrupted"},
            {**controlled_remote, "duration_status": "unavailable"},
        ):
            with self.subTest(tampered=tampered):
                peer = controlled_remote if tampered.get("role") == "gNB" else controlled_loaded
                self.assertTrue(
                    experiment_failed([tampered, peer], {"gNB", "nrUE"})
                )
        wrong_loaded_stop = {
            **loaded_peer,
            "stop_reason": "duration_elapsed",
        }
        self.assertTrue(
            experiment_failed([loaded, wrong_loaded_stop], {"gNB", "nrUE"})
        )
        for field, value in (
            ("workload_status", "failed"),
            ("workload_artifact", ""),
            ("network_cleanup_status", "failed"),
        ):
            failed = dict(loaded_peer)
            failed[field] = value
            self.assertTrue(
                experiment_failed([loaded, failed], {"gNB", "nrUE"}),
                field,
            )

    def test_offline_campaign_success_matches_runtime_workload_mode(self) -> None:
        baseline = {
            "status": "finished",
            "return_code": 0,
            "stop_reason": "duration_elapsed",
            "duration_status": "valid",
            "archive_status": "finalization_pending",
            "archive_manifest_present": 1,
            "campaign_schema_version": 1,
            "campaign_schema_supported": 1,
            "profile_enabled": 0,
            "pmu_mode": "off",
            "sidecar_tool": "none",
            "sidecar_status": "not_requested",
            "sidecar_artifact": "",
            "sidecar_config_json": '{"tool":"none"}',
            "workload_status": "not_configured",
            "workload_artifact": "",
            "network_cleanup_status": "not_configured",
        }
        loaded = {
            **baseline,
            "stop_reason": "measurement_complete",
            "workload_status": "completed",
            "workload_artifact": "workload/workload_run.json",
            "network_cleanup_status": "ok",
        }
        self.assertTrue(reports.campaign_member_succeeded(baseline))
        self.assertTrue(reports.campaign_member_succeeded(loaded))
        controlled_local = {
            **loaded,
            "host": "local",
            "status": "exited_nonzero",
            "return_code": -signal.SIGINT,
            "completion_classifier_version": "1",
            "controlled_stop_requested": 1,
            "shutdown_stage": "SIGINT",
            "shutdown_verified": 1,
            "remote_completion_identity_verified": "not_applicable",
            "remote_completion_return_code": "",
            "termination_class": "controlled_sigint_signal",
            "termination_class_recorded": "controlled_sigint_signal",
        }
        controlled_remote = {
            **controlled_local,
            "host": "cm5",
            "return_code": str(128 + signal.SIGINT),
            "remote_completion_return_code": 128 + signal.SIGINT,
            "remote_completion_identity_verified": 1,
        }
        self.assertTrue(reports.campaign_member_succeeded(controlled_local))
        self.assertTrue(reports.campaign_member_succeeded(controlled_remote))
        self.assertTrue(
            reports.campaign_member_succeeded(
                {**baseline, "duration_status": "legacy_realtime_fallback"}
            )
        )
        for tampered in (
            {**baseline, "stop_reason": "measurement_complete"},
            {
                **baseline,
                "workload_status": "not_configured",
                "workload_artifact": "workload/workload_run.json",
            },
            {**loaded, "stop_reason": "duration_elapsed"},
            {**loaded, "workload_artifact": ""},
            {**loaded, "network_cleanup_status": "failed"},
            {**controlled_local, "shutdown_stage": "SIGTERM"},
            {**controlled_local, "shutdown_verified": 0},
            {**controlled_local, "duration_status": "legacy_realtime_fallback"},
            {**controlled_remote, "remote_completion_identity_verified": 0},
            {**controlled_remote, "return_code": "-2"},
        ):
            with self.subTest(tampered=tampered):
                self.assertFalse(reports.campaign_member_succeeded(tampered))

        self.assertTrue(
            reports.campaign_member_measurement_eligible(
                baseline,
                archive_integrity_valid=True,
            )
        )
        for tampered, integrity_valid in (
            ({**baseline, "archive_manifest_present": 0}, True),
            ({**baseline, "archive_status": "finalize_failed"}, True),
            (
                {
                    **baseline,
                    "sidecar_tool": "perf_stat",
                    "sidecar_status": "artifact_missing",
                },
                True,
            ),
            (baseline, False),
            (
                {
                    **controlled_local,
                    "termination_class_recorded": "natural_zero",
                },
                True,
            ),
        ):
            with self.subTest(tampered=tampered, integrity=integrity_valid):
                self.assertTrue(reports.campaign_member_succeeded(tampered))
                self.assertFalse(
                    reports.campaign_member_measurement_eligible(
                        tampered,
                        archive_integrity_valid=integrity_valid,
                    )
                )

        with tempfile.TemporaryDirectory(
            prefix="oai-profile-campaign-report-"
        ) as temporary:
            run_dir = Path(temporary).resolve()
            key = str(run_dir)
            campaign_metadata = {
                **controlled_remote,
                "schema_version": 1,
                "runner_version": "2",
                "sidecar": {"tool": "none"},
                "transport_return_code": 255,
                "remote_completion_return_code": 128 + signal.SIGINT,
                "workload_control_results": {
                    "run": {
                        "transport_return_code": 255,
                        "remote_completion_return_code": 0,
                    }
                },
            }
            report = reports.campaign_report(
                [run_dir],
                {},
                {key: campaign_metadata},
                {key: "ok"},
            )
        self.assertEqual(len(report.rows), 1)
        row = report.rows[0]
        self.assertEqual(row["workload_status"], "completed")
        self.assertEqual(
            row["workload_artifact"],
            "workload/workload_run.json",
        )
        self.assertEqual(row["network_cleanup_status"], "ok")
        self.assertEqual(row["status"], "exited_nonzero")
        self.assertEqual(row["return_code"], "130")
        self.assertEqual(row["shutdown_stage"], "SIGINT")
        self.assertEqual(row["termination_class"], "controlled_sigint_signal")
        self.assertEqual(row["transport_return_code"], 255)
        self.assertEqual(row["remote_completion_return_code"], 128 + signal.SIGINT)
        self.assertEqual(
            json.loads(str(row["workload_control_results_json"]))["run"][
                "remote_completion_return_code"
            ],
            0,
        )

    def test_controlled_sigint_proof_reaches_completeness_and_observer(self) -> None:
        common = {
            "campaign_id": "controlled-campaign",
            "experiment_id": "controlled-experiment",
            "case": "RxTxTime3",
            "variant": "in-process",
            "trial": "1",
            "profile_enabled": True,
            "pmu_mode": "off",
            "status": "exited_nonzero",
            "stop_reason": "measurement_complete",
            "duration_s": 120.0,
            "duration_status": "valid",
            "completion_classifier_version": "1",
            "controlled_stop_requested": 1,
            "shutdown_stage": "SIGINT",
            "shutdown_verified": 1,
            "termination_class": "controlled_sigint_signal",
            "termination_class_recorded": "controlled_sigint_signal",
            "termination_class_recomputed": "controlled_sigint_signal",
            "termination_class_consistent": 1,
            "sidecar_tool": "none",
            "sidecar_status": "not_requested",
            "sidecar_artifact": "",
            "sidecar_config_json": '{"tool":"none"}',
            "sidecar_evidence_fields_present": "complete",
            "workload_status": "completed",
            "workload_artifact": "workload/workload_run.json",
            "network_cleanup_status": "ok",
            "workload_evidence_fields_present": "complete",
            "archive_status": "finalization_pending",
            "archive_manifest_present": 1,
            "campaign_schema_version": 1,
            "campaign_schema_supported": 1,
            "campaign_profile_identity_consistent": 1,
        }
        gnb = {
            **common,
            "profile_dir": "/run/controlled-gNB",
            "role": "gNB",
            "host": "local",
            "return_code": -signal.SIGINT,
            "remote_completion_return_code": "",
            "remote_completion_identity_verified": "not_applicable",
        }
        nrue = {
            **common,
            "profile_dir": "/run/controlled-nrUE",
            "role": "nrUE",
            "host": "cm5",
            "return_code": 128 + signal.SIGINT,
            "remote_completion_return_code": 128 + signal.SIGINT,
            "remote_completion_identity_verified": 1,
        }
        integrity = [
            {"profile_dir": row["profile_dir"], "valid": 1}
            for row in (gnb, nrue)
        ]
        coverage = {
            str(gnb["profile_dir"]): "complete",
            str(nrue["profile_dir"]): "complete",
        }
        completeness = reports.campaign_completeness_report(
            [gnb, nrue],
            integrity,
            coverage,
        ).rows[0]
        self.assertEqual(completeness["finished_roles"], "")
        self.assertEqual(
            completeness["termination_accepted_roles"],
            "gNB;nrUE",
        )
        self.assertEqual(completeness["controlled_stop_roles"], "gNB;nrUE")
        self.assertEqual(
            completeness["termination_class_by_role"],
            "gNB:controlled_sigint_signal|nrUE:controlled_sigint_signal",
        )
        self.assertEqual(completeness["workload_contract_consistent"], 1)
        self.assertEqual(completeness["successful_roles"], "gNB;nrUE")
        self.assertEqual(completeness["sidecar_valid_roles"], "gNB;nrUE")
        self.assertEqual(
            completeness["measurement_eligible_roles"],
            "gNB;nrUE",
        )
        self.assertEqual(completeness["paired_complete"], 1)
        self.assertEqual(completeness["paired_measurement_complete"], 1)
        self.assertEqual(completeness["status"], "complete")
        self.assertEqual(completeness["profile_evidence_complete"], 1)

        summary_rows = [
            {
                "profile_dir": row["profile_dir"],
                "event_kind": "duration",
                "event_name": "SLOT_PROCESS",
                "p50_us": 100.0 if row["role"] == "gNB" else 200.0,
            }
            for row in (gnb, nrue)
        ]
        metadata_by_dir = {
            str(row["profile_dir"]): {
                "campaign_id": str(row["campaign_id"]),
                "role": str(row["role"]),
                "variant": str(row["variant"]),
                "trial": str(row["trial"]),
            }
            for row in (gnb, nrue)
        }
        campaign_by_dir = {
            str(row["profile_dir"]): {"case": str(row["case"])}
            for row in (gnb, nrue)
        }
        observer = reports.observer_effect_report(
            [gnb, nrue],
            summary_rows,
            metadata_by_dir,
            campaign_by_dir,
            coverage,
            integrity,
        )
        success_rows = [
            row
            for row in observer.rows
            if row["metric_name"] == "process_success"
        ]
        self.assertEqual(len(success_rows), 2)
        self.assertEqual({row["mean"] for row in success_rows}, {1.0})
        duration_rows = [
            row
            for row in observer.rows
            if row["metric_name"] == "process_duration"
        ]
        self.assertEqual(len(duration_rows), 2)
        self.assertEqual({row["sample_count"] for row in duration_rows}, {1})
        event_rows = [
            row for row in observer.rows if row["metric_name"] == "SLOT_PROCESS"
        ]
        self.assertEqual(len(event_rows), 2)
        self.assertEqual({row["sample_count"] for row in event_rows}, {1})

        unproven_nrue = {**nrue, "shutdown_stage": "SIGTERM"}
        failed = reports.campaign_completeness_report(
            [gnb, unproven_nrue],
            integrity,
            coverage,
        ).rows[0]
        self.assertEqual(failed["termination_accepted_roles"], "gNB")
        self.assertEqual(failed["successful_roles"], "gNB")
        self.assertEqual(failed["paired_complete"], 0)
        self.assertIn("termination_unaccepted", str(failed["status"]))
        self.assertIn("unsuccessful_role", str(failed["status"]))

        mismatched_workload = {
            **nrue,
            "workload_status": "not_configured",
            "workload_artifact": "",
            "network_cleanup_status": "not_configured",
            "stop_reason": "duration_elapsed",
        }
        mismatch = reports.campaign_completeness_report(
            [gnb, mismatched_workload],
            integrity,
            coverage,
        ).rows[0]
        self.assertEqual(mismatch["termination_accepted_roles"], "gNB;nrUE")
        self.assertEqual(mismatch["successful_roles"], "gNB;nrUE")
        self.assertEqual(mismatch["workload_contract_consistent"], 0)
        self.assertEqual(mismatch["paired_complete"], 0)
        self.assertIn("workload_mismatch", str(mismatch["status"]))

        mismatch_observer = reports.observer_effect_report(
            [gnb, mismatched_workload],
            summary_rows,
            metadata_by_dir,
            campaign_by_dir,
            coverage,
            integrity,
        )
        self.assertEqual(
            {row["metric_name"] for row in mismatch_observer.rows},
            {"process_success"},
        )
        self.assertEqual(
            {row["mean"] for row in mismatch_observer.rows},
            {1.0},
        )

        for name, acquisition_nrue, acquisition_integrity in (
            (
                "sidecar",
                {
                    **nrue,
                    "sidecar_tool": "perf_stat",
                    "sidecar_status": "artifact_missing",
                    "sidecar_artifact": "",
                },
                integrity,
            ),
            (
                "integrity",
                nrue,
                [
                    integrity[0],
                    {"profile_dir": nrue["profile_dir"], "valid": 0},
                ],
            ),
        ):
            with self.subTest(acquisition_failure=name):
                acquisition_observer = reports.observer_effect_report(
                    [gnb, acquisition_nrue],
                    summary_rows,
                    metadata_by_dir,
                    campaign_by_dir,
                    coverage,
                    acquisition_integrity,
                )
                process_success = [
                    row
                    for row in acquisition_observer.rows
                    if row["metric_name"] == "process_success"
                ]
                process_duration = [
                    row
                    for row in acquisition_observer.rows
                    if row["metric_name"] == "process_duration"
                ]
                event_metrics = [
                    row
                    for row in acquisition_observer.rows
                    if row["metric_name"] == "SLOT_PROCESS"
                ]
                self.assertEqual({row["mean"] for row in process_success}, {1.0})
                self.assertEqual(len(process_duration), 2)
                self.assertEqual(event_metrics, [])

        valid_perf_stat_nrue = {
            **nrue,
            "sidecar_tool": "perf_stat",
            "sidecar_status": "registered",
            "sidecar_artifact": "sidecars/perf-stat.csv",
            "sidecar_config_json": '{"interval_ms":1000,"tool":"perf_stat"}',
        }
        sidecar_mismatch = reports.campaign_completeness_report(
            [gnb, valid_perf_stat_nrue],
            integrity,
            coverage,
        ).rows[0]
        self.assertEqual(sidecar_mismatch["paired_complete"], 1)
        self.assertEqual(sidecar_mismatch["sidecar_contract_consistent"], 0)
        self.assertEqual(sidecar_mismatch["measurement_contract_consistent"], 0)
        self.assertEqual(sidecar_mismatch["paired_measurement_complete"], 0)
        self.assertIn("sidecar_mismatch", str(sidecar_mismatch["measurement_status"]))
        mixed_observer = reports.observer_effect_report(
            [gnb, valid_perf_stat_nrue],
            summary_rows,
            metadata_by_dir,
            campaign_by_dir,
            coverage,
            integrity,
        )
        self.assertEqual(
            {row["metric_name"] for row in mixed_observer.rows},
            {"process_success", "process_duration"},
        )

        blank_identity_nrue = {**nrue, "experiment_id": ""}
        blank_observer = reports.observer_effect_report(
            [gnb, blank_identity_nrue],
            summary_rows,
            metadata_by_dir,
            campaign_by_dir,
            coverage,
            integrity,
        )
        self.assertEqual(
            {row["metric_name"] for row in blank_observer.rows},
            {"process_success"},
        )
        blank_completeness = reports.campaign_completeness_report(
            [gnb, blank_identity_nrue],
            integrity,
            coverage,
        ).rows
        self.assertEqual(len(blank_completeness), 1)
        self.assertEqual(blank_completeness[0]["roles_present"], "gNB")
        self.assertEqual(blank_completeness[0]["paired_complete"], 0)
        self.assertIn("missing_role", str(blank_completeness[0]["status"]))

        unsupported_nrue = {**nrue, "campaign_schema_supported": 0}
        unsupported = reports.campaign_completeness_report(
            [gnb, unsupported_nrue],
            integrity,
            coverage,
        ).rows[0]
        self.assertEqual(unsupported["paired_complete"], 0)
        self.assertIn("campaign_schema_unsupported", str(unsupported["status"]))
        unsupported_observer = reports.observer_effect_report(
            [gnb, unsupported_nrue],
            summary_rows,
            metadata_by_dir,
            campaign_by_dir,
            coverage,
            integrity,
        )
        self.assertEqual(
            {row["metric_name"] for row in unsupported_observer.rows},
            {"process_success"},
        )

    def test_campaign_schema_and_identity_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="oai-profile-schema-identity-"
        ) as temporary:
            root = Path(temporary)
            spec_path = root / "campaign.json"
            write_spec(
                spec_path,
                root / "profiles",
                ["/bin/true"],
                ["/bin/true"],
            )
            original = json.loads(spec_path.read_text())
            for schema_version in (True, 1.0, "1", 2, None):
                value = dict(original)
                if schema_version is None:
                    value.pop("schema_version", None)
                else:
                    value["schema_version"] = schema_version
                spec_path.write_text(json.dumps(value))
                with self.subTest(input_schema=schema_version):
                    with self.assertRaisesRegex(ValueError, "schema_version"):
                        campaign.load_spec(spec_path)

            manifest_path = root / "campaign_run.json"
            for schema_version in (1, True, 1.0, "1", 2, None):
                value = {"schema_version": schema_version}
                if schema_version is None:
                    value = {}
                manifest_path.write_text(json.dumps(value))
                parsed, status = reports.read_campaign(manifest_path)
                self.assertEqual(parsed, value)
                if type(schema_version) is int and schema_version == 1:
                    self.assertEqual(status, "ok")
                else:
                    self.assertTrue(status.startswith("unsupported_schema:"))

            run_dirs = [root / "gNB", root / "nrUE"]
            for run_dir in run_dirs:
                run_dir.mkdir()
                (run_dir / "archive_manifest.csv").touch()
            def manifest(role: str, experiment_id: str) -> dict[str, object]:
                return {
                    "schema_version": 1,
                    "runner_version": "2",
                    "completion_classifier_version": "1",
                    "campaign_id": "identity-campaign",
                    "experiment_id": experiment_id,
                    "case": "RxTxTime3",
                    "variant": "in-process",
                    "trial": 1,
                    "role": role,
                    "host": "local",
                    "hostname": f"{role.lower()}-host",
                    "profile_enabled": True,
                    "pmu_mode": "off",
                    "status": "finished",
                    "return_code": 0,
                    "remote_completion_return_code": "",
                    "stop_reason": "duration_elapsed",
                    "controlled_stop_requested": 0,
                    "shutdown_stage": "none",
                    "shutdown_verified": 1,
                    "remote_completion_identity_verified": "not_applicable",
                    "termination_class": "natural_zero",
                    "start_realtime_ns": 1_000_000_000,
                    "end_realtime_ns": 2_000_000_000,
                    "start_monotonic_raw_ns": 500_000_000,
                    "end_monotonic_raw_ns": 1_500_000_000,
                    "sidecar": {"tool": "none"},
                    "sidecar_tool": "none",
                    "sidecar_status": "not_requested",
                    "sidecar_artifact": "",
                    "workload_status": "not_configured",
                    "workload_artifact": "",
                    "network_cleanup_status": "not_configured",
                    "archive_status": "finalization_pending",
                }

            campaign_by_dir = {
                str(run_dirs[0]): manifest("gNB", "campaign-e1"),
                str(run_dirs[1]): manifest("nrUE", "campaign-e2"),
            }
            metadata_by_dir = {
                str(run_dir): {
                    "experiment_id": "metadata-shared",
                    "campaign_id": "identity-campaign",
                    "variant": "in-process",
                    "trial": "1",
                    "role": role,
                    "hostname": f"{role.lower()}-host",
                }
                for run_dir, role in zip(run_dirs, ("gNB", "nrUE"))
            }
            report = reports.campaign_report(
                run_dirs,
                metadata_by_dir,
                campaign_by_dir,
                {str(run_dir): "ok" for run_dir in run_dirs},
            )
            self.assertEqual(
                {row["experiment_id"] for row in report.rows},
                {"campaign-e1", "campaign-e2"},
            )
            self.assertEqual(
                {row["campaign_profile_identity_consistent"] for row in report.rows},
                {0},
            )
            self.assertTrue(
                all(
                    "experiment_id:value_mismatch"
                    in str(row["campaign_profile_identity_mismatches"])
                    for row in report.rows
                )
            )
            completeness = reports.campaign_completeness_report(
                report.rows,
                [
                    {"profile_dir": str(run_dir), "valid": 1}
                    for run_dir in run_dirs
                ],
            )
            self.assertEqual(len(completeness.rows), 2)
            self.assertTrue(
                all(row["paired_complete"] == 0 for row in completeness.rows)
            )

            missing_experiment = dict(campaign_by_dir[str(run_dirs[0])])
            missing_experiment.pop("experiment_id")
            missing_report = reports.campaign_report(
                [run_dirs[0]],
                {str(run_dirs[0]): metadata_by_dir[str(run_dirs[0])]},
                {str(run_dirs[0]): missing_experiment},
                {str(run_dirs[0]): "ok"},
            ).rows[0]
            self.assertEqual(missing_report["experiment_id"], "")
            self.assertIn(
                "experiment_id:campaign_missing",
                str(missing_report["campaign_profile_identity_mismatches"]),
            )

            disabled = manifest("gNB", "disabled-e1")
            disabled["profile_enabled"] = False
            disabled_report = reports.campaign_report(
                [run_dirs[0]],
                {},
                {str(run_dirs[0]): disabled},
                {str(run_dirs[0]): "ok"},
            ).rows[0]
            self.assertEqual(disabled_report["campaign_profile_identity_consistent"], "")

    def test_campaign_results_migrates_only_known_headers(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="oai-profile-results-migration-"
        ) as temporary:
            root = Path(temporary)
            current_row = {
                field_name: ""
                for field_name in campaign.RESULT_FIELDS
            }
            current_row["campaign_id"] = "new-campaign"
            for name, fields in (
                ("legacy", campaign.LEGACY_RESULT_FIELDS),
                ("pre-identity", campaign.PRE_IDENTITY_RESULT_FIELDS),
                ("pre-completion", campaign.PRE_COMPLETION_RESULT_FIELDS),
                (
                    "pre-sidecar-artifact",
                    campaign.PRE_SIDECAR_ARTIFACT_RESULT_FIELDS,
                ),
            ):
                with self.subTest(name=name):
                    path = root / f"{name}.csv"
                    old_row = {field_name: "" for field_name in fields}
                    old_row["campaign_id"] = "old-campaign"
                    if "workload_status" in old_row:
                        old_row["workload_status"] = "completed"
                        old_row[
                            "workload_artifact"
                        ] = "workload/workload_run.json"
                        old_row["network_cleanup_status"] = "ok"
                    if "transport_return_code" in old_row:
                        old_row["transport_return_code"] = "255"
                        old_row["remote_completion_return_code"] = "130"
                    if "completion_classifier_version" in old_row:
                        old_row["completion_classifier_version"] = "1"
                    with path.open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.DictWriter(stream, fieldnames=fields)
                        writer.writeheader()
                        writer.writerow(old_row)
                    campaign.append_results(path, [current_row])
                    with path.open(newline="", encoding="utf-8") as stream:
                        reader = csv.DictReader(stream)
                        migrated = list(reader)
                        self.assertEqual(
                            reader.fieldnames,
                            campaign.RESULT_FIELDS,
                        )
                    self.assertEqual(len(migrated), 2)
                    preserves_identity = name in {
                        "pre-completion",
                        "pre-sidecar-artifact",
                    }
                    expected_transport = "255" if preserves_identity else ""
                    expected_remote = "130" if preserves_identity else ""
                    self.assertEqual(migrated[0]["transport_return_code"], expected_transport)
                    self.assertEqual(migrated[0]["remote_completion_return_code"], expected_remote)
                    for field_name in campaign.COMPLETION_RESULT_FIELDS:
                        expected = (
                            "1"
                            if name == "pre-sidecar-artifact"
                            and field_name == "completion_classifier_version"
                            else ""
                        )
                        self.assertEqual(migrated[0][field_name], expected)
                    self.assertEqual(migrated[0]["sidecar_artifact"], "")
                    expected_status = (
                        "not_configured" if name == "legacy" else "completed"
                    )
                    self.assertEqual(
                        migrated[0]["workload_status"],
                        expected_status,
                    )

            unknown = root / "unknown.csv"
            unknown.write_text("unexpected\nvalue\n")
            with self.assertRaisesRegex(
                ValueError,
                "unexpected campaign results schema",
            ):
                campaign.append_results(unknown, [current_row])

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
