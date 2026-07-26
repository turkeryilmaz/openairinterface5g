#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import oai_profile_workload as workload  # noqa: E402


def spec_value(root: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "client_role": "nrUE",
        "helper": str(Path(workload.__file__).resolve()),
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


def rule(priority: int, src: str = "all", dst: str = "all") -> dict[str, object]:
    return {
        "priority": priority,
        "src": src,
        "dst": dst,
        "table": 9999,
        "action": "lookup",
    }


def route() -> dict[str, object]:
    return {
        "dst": "default",
        "dev": "oaitun_ue1",
        "gateway": "",
        "table": 9999,
        "protocol": "",
        "scope": "",
        "type": "unicast",
        "metric": None,
    }


def iperf_result() -> dict[str, object]:
    def summary(sender: bool, bitrate_bps: float, lost_packets: int) -> dict[str, object]:
        return {
            "seconds": 120.1,
            "bytes": 15_000_000,
            "bits_per_second": bitrate_bps,
            "jitter_ms": 0.19,
            "lost_packets": lost_packets,
            "packets": 12_500,
            "lost_percent": 100.0 * lost_packets / 12_500,
            "sender": sender,
        }

    return {
        "start": {
            "connected": [
                {
                    "socket": 5,
                    "local_host": "10.0.0.4",
                    "local_port": 40000,
                    "remote_host": "192.168.70.135",
                    "remote_port": 5201,
                },
                {
                    "socket": 7,
                    "local_host": "10.0.0.4",
                    "local_port": 40001,
                    "remote_host": "192.168.70.135",
                    "remote_port": 5201,
                },
            ],
            "connecting_to": {
                "host": "192.168.70.135",
                "port": 5201,
            },
            "test_start": {
                "protocol": "UDP",
                "duration": 120,
                "blksize": 1200,
                "target_bitrate": 1_000_000,
                "bidir": 1,
            },
        },
        "end": {
            "sum_sent": summary(True, 1_002_000.0, 0),
            "sum_received": summary(False, 998_000.0, 2),
            "sum_sent_bidir_reverse": summary(True, 1_001_000.0, 0),
            "sum_received_bidir_reverse": summary(False, 997_000.0, 1),
        },
        "server_output_text": "server-side iperf3 evidence\n",
    }


class WorkloadTest(unittest.TestCase):
    def test_strict_schema_and_exact_publication_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-workload-schema-") as temporary:
            value = spec_value(Path(temporary))
            spec = workload.parse_workload_spec(value)
            self.assertEqual(spec.subnet, workload.EXPECTED_SUBNET)
            self.assertEqual(spec.policy_table, 9999)
            plan = workload.command_plan(spec)
            self.assertEqual(plan["traffic_contract"]["bitrate_bps_per_direction"], 1_000_000)
            self.assertEqual(plan["traffic_contract"]["post_readiness_duration_s"], 120)
            self.assertEqual(
                plan["iperf3_command"],
                [
                    "iperf3",
                    "-c",
                    "192.168.70.135",
                    "-B",
                    "<dynamic-ue-ip>",
                    "--bind-dev",
                    "oaitun_ue1",
                    "-u",
                    "--bidir",
                    "-b",
                    "1000000",
                    "-l",
                    "1200",
                    "-t",
                    "120",
                    "-i",
                    "1",
                    "--get-server-output",
                    "--udp-counters-64bit",
                    "-J",
                ],
            )

            invalid = dict(value)
            invalid["duration_s"] = 119
            with self.assertRaisesRegex(ValueError, "duration_s"):
                workload.parse_workload_spec(invalid)
            invalid = dict(value)
            invalid["server_ipv4"] = "192.168.70.136"
            with self.assertRaisesRegex(ValueError, "192.168.70.135"):
                workload.parse_workload_spec(invalid)
            for field, invalid_value in (
                ("schema_version", 1.9),
                ("schema_version", True),
                ("duration_s", 120.9),
                ("duration_s", "120"),
                ("bitrate_bps", 1_000_000.9),
                ("policy_table", True),
                ("readiness_timeout_s", True),
                ("readiness_poll_s", float("inf")),
            ):
                with self.subTest(field=field, invalid_value=invalid_value):
                    invalid = dict(value)
                    invalid[field] = invalid_value
                    with self.assertRaisesRegex(ValueError, field):
                        workload.parse_workload_spec(invalid)
            invalid = dict(value)
            invalid["typo"] = 1
            with self.assertRaisesRegex(ValueError, "unknown workload fields"):
                workload.parse_workload_spec(invalid)

    def test_discovery_requires_one_global_address_in_exact_subnet(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-workload-address-") as temporary:
            spec = workload.parse_workload_spec(spec_value(Path(temporary)))
            valid = [
                {
                    "ifname": "oaitun_ue1",
                    "addr_info": [
                        {
                            "family": "inet",
                            "local": "10.0.0.4",
                            "prefixlen": 24,
                            "scope": "global",
                        }
                    ],
                }
            ]
            with patch("oai_profile_workload._json_command", return_value=valid) as query:
                self.assertEqual(workload.discover_global_ipv4(spec), "10.0.0.4")
            query.assert_called_once_with(
                [
                    spec.ip_binary,
                    "-j",
                    "-4",
                    "address",
                    "show",
                    "dev",
                    spec.interface,
                    "scope",
                    "global",
                ],
                missing_device_is_pending=True,
            )
            duplicate = valid + [
                {
                    "ifname": "oaitun_ue1",
                    "addr_info": [
                        {
                            "family": "inet",
                            "local": "10.0.0.5",
                            "prefixlen": 24,
                            "scope": "global",
                        }
                    ]
                }
            ]
            with patch("oai_profile_workload._json_command", return_value=duplicate):
                with self.assertRaisesRegex(workload.WorkloadError, "exactly one"):
                    workload.discover_global_ipv4(spec)

    def test_interface_absence_is_distinct_from_ip_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-workload-interface-") as temporary:
            spec = workload.parse_workload_spec(spec_value(Path(temporary)))
            present = subprocess.CompletedProcess(
                ["ip"],
                0,
                '[{"ifname":"oaitun_ue1"}]',
                "",
            )
            absent = subprocess.CompletedProcess(
                ["ip"],
                1,
                "",
                'Device "oaitun_ue1" does not exist.',
            )
            failed = subprocess.CompletedProcess(["ip"], 2, "", "permission denied")
            with patch("oai_profile_workload.run_command", return_value=present):
                self.assertTrue(workload.interface_exists(spec))
            with patch("oai_profile_workload.run_command", return_value=absent):
                self.assertFalse(workload.interface_exists(spec))
            with patch("oai_profile_workload.run_command", return_value=failed):
                with self.assertRaisesRegex(workload.WorkloadError, "inspection failed"):
                    workload.interface_exists(spec)

    def test_exact_policy_and_route_validation_and_ownership(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-workload-policy-") as temporary:
            spec = workload.parse_workload_spec(spec_value(Path(temporary)))
            active = {
                "rules": [
                    rule(12000, src="10.0.0.4/32"),
                    rule(12001, dst="10.0.0.4/32"),
                ],
                "routes": [route()],
            }
            validation = workload.validate_ready_network(spec, "10.0.0.4", active)
            self.assertEqual(validation["source_rule"]["priority"], 12000)
            ownership = workload.build_ownership(
                spec,
                "10.0.0.4",
                {"rules": [], "routes": []},
                active,
            )
            self.assertEqual(len(ownership["rules"]), 2)
            self.assertEqual(len(ownership["routes"]), 1)

            ambiguous = {
                "rules": active["rules"] + [rule(12002, src="10.0.0.4/32")],
                "routes": active["routes"],
            }
            with self.assertRaisesRegex(workload.WorkloadError, "duplicate"):
                workload.validate_ready_network(spec, "10.0.0.4", ambiguous)

    def test_ping_requires_three_of_three_and_is_source_bound(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-workload-ping-") as temporary:
            root = Path(temporary)
            spec = workload.parse_workload_spec(spec_value(root))
            completed = subprocess.CompletedProcess(
                ["ping"],
                0,
                "3 packets transmitted, 3 received, 0% packet loss\n",
                "",
            )
            with patch("oai_profile_workload.run_command", return_value=completed) as run:
                result = workload.run_ping(spec, root, "10.0.0.4")
            self.assertEqual(result["status"], "ok")
            self.assertEqual(run.call_args.args[0][0:3], ["ping", "-I", "10.0.0.4"])

            incomplete = subprocess.CompletedProcess(
                ["ping"],
                0,
                "3 packets transmitted, 2 received, 33% packet loss\n",
                "",
            )
            with patch("oai_profile_workload.run_command", return_value=incomplete):
                with self.assertRaisesRegex(workload.WorkloadError, "3/3"):
                    workload.run_ping(spec, root, "10.0.0.4")

    def test_iperf_command_and_source_host_anchors_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-workload-iperf-") as temporary:
            root = Path(temporary)
            spec = workload.parse_workload_spec(spec_value(root))
            completed = subprocess.CompletedProcess(
                ["iperf3"],
                0,
                json.dumps(iperf_result()),
                "",
            )
            anchors = [
                {"realtime_ns": 10, "monotonic_raw_ns": 20},
                {"realtime_ns": 30, "monotonic_raw_ns": 40},
            ]
            with (
                patch("oai_profile_workload.run_command", return_value=completed) as run,
                patch("oai_profile_workload.clock_anchors", side_effect=anchors),
            ):
                result = workload.run_iperf3(spec, root, "10.0.0.4")
            command = run.call_args.args[0]
            self.assertEqual(
                command,
                [
                    "iperf3",
                    "-c",
                    "192.168.70.135",
                    "-B",
                    "10.0.0.4",
                    "--bind-dev",
                    "oaitun_ue1",
                    "-u",
                    "--bidir",
                    "-b",
                    "1000000",
                    "-l",
                    "1200",
                    "-t",
                    "120",
                    "-i",
                    "1",
                    "--get-server-output",
                    "--udp-counters-64bit",
                    "-J",
                ],
            )
            self.assertEqual(result["start_monotonic_raw_ns"], 20)
            self.assertEqual(result["end_monotonic_raw_ns"], 40)
            self.assertEqual(
                result["start_validation"]["connecting_to_host"],
                "192.168.70.135",
            )
            self.assertEqual(result["start_validation"]["connection_count"], 2)
            self.assertEqual(len(result["views"]), 4)
            self.assertEqual(
                [row["direction"] for row in result["directions"]],
                ["nrUE_to_ext-DN_UL", "ext-DN_to_nrUE_DL"],
            )
            self.assertEqual(result["directions"][0]["receiver_lost_packets"], 2)
            self.assertAlmostEqual(
                result["directions"][1]["achieved_to_requested_ratio"],
                0.997,
            )
            self.assertTrue(result["server_output_text_present"])
            self.assertEqual(
                (root / "workload" / "iperf3_server.stdout.log").read_text(),
                "server-side iperf3 evidence\n",
            )

            compatibility = iperf_result()
            reverse_sender = compatibility["end"]["sum_sent_bidir_reverse"]  # type: ignore[index]
            reverse_sender["packets"] = 0  # type: ignore[index]
            reverse_sender["lost_packets"] = 0  # type: ignore[index]
            completed = subprocess.CompletedProcess(
                ["iperf3"],
                0,
                json.dumps(compatibility),
                "",
            )
            with patch("oai_profile_workload.run_command", return_value=completed):
                compatible_result = workload.run_iperf3(
                    spec,
                    root,
                    "10.0.0.4",
                )
            reverse_view = next(
                row
                for row in compatible_result["views"]
                if row["json_key"] == "end.sum_sent_bidir_reverse"
            )
            self.assertFalse(reverse_view["packet_metrics_valid"])
            self.assertEqual(
                reverse_view["packet_metrics_status"],
                "not_authoritative_inconsistent_zero_packets",
            )
            self.assertIsNone(reverse_view["packets"])
            self.assertEqual(reverse_view["reported_packets"], 0)

            missing_sender_auxiliary = iperf_result()
            sender = missing_sender_auxiliary["end"]["sum_sent"]  # type: ignore[index]
            for field in ("packets", "lost_packets", "lost_percent", "jitter_ms"):
                sender.pop(field)  # type: ignore[union-attr]
            completed = subprocess.CompletedProcess(
                ["iperf3"],
                0,
                json.dumps(missing_sender_auxiliary),
                "",
            )
            with patch("oai_profile_workload.run_command", return_value=completed):
                missing_aux_result = workload.run_iperf3(
                    spec,
                    root,
                    "10.0.0.4",
                )
            sender_view = next(
                row
                for row in missing_aux_result["views"]
                if row["json_key"] == "end.sum_sent"
            )
            self.assertFalse(sender_view["packet_metrics_valid"])
            self.assertIsNone(sender_view["reported_packets"])

            zero_delivery = iperf_result()
            zero_receiver = zero_delivery["end"]["sum_received_bidir_reverse"]  # type: ignore[index]
            zero_receiver.update(  # type: ignore[union-attr]
                {
                    "bytes": 0,
                    "bits_per_second": 0.0,
                    "jitter_ms": 0.0,
                    "lost_packets": 0,
                    "packets": 0,
                    "lost_percent": 0.0,
                }
            )
            completed = subprocess.CompletedProcess(
                ["iperf3"],
                0,
                json.dumps(zero_delivery),
                "",
            )
            with patch("oai_profile_workload.run_command", return_value=completed):
                zero_result = workload.run_iperf3(spec, root, "10.0.0.4")
            zero_direction = next(
                row
                for row in zero_result["directions"]
                if row["direction"] == "ext-DN_to_nrUE_DL"
            )
            self.assertEqual(zero_direction["achieved_bitrate_bps"], 0.0)
            self.assertIsNone(zero_direction["receiver_lost_percent"])
            self.assertFalse(zero_direction["receiver_packet_metrics_valid"])
            self.assertEqual(
                zero_direction["receiver_packet_metrics_status"],
                "no_received_datagrams_loss_unavailable",
            )
            self.assertIsNone(zero_direction["receiver_jitter_ms"])
            self.assertFalse(zero_direction["receiver_jitter_valid"])

            near_total_loss = iperf_result()
            lossy_receiver = near_total_loss["end"]["sum_received"]  # type: ignore[index]
            lossy_receiver.update(  # type: ignore[union-attr]
                {
                    "bytes": 1200,
                    "bits_per_second": 80.0,
                    "jitter_ms": 1.5,
                    "lost_packets": 99,
                    "packets": 100,
                    "lost_percent": 98.5,
                }
            )
            completed = subprocess.CompletedProcess(
                ["iperf3"],
                0,
                json.dumps(near_total_loss),
                "",
            )
            with patch("oai_profile_workload.run_command", return_value=completed):
                lossy_result = workload.run_iperf3(spec, root, "10.0.0.4")
            lossy_direction = next(
                row
                for row in lossy_result["directions"]
                if row["direction"] == "nrUE_to_ext-DN_UL"
            )
            self.assertEqual(lossy_direction["receiver_lost_percent"], 98.5)
            self.assertTrue(lossy_direction["receiver_packet_metrics_valid"])
            self.assertTrue(lossy_direction["receiver_jitter_valid"])

            total_loss = iperf_result()
            total_receiver = total_loss["end"]["sum_received"]  # type: ignore[index]
            total_receiver.update(  # type: ignore[union-attr]
                {
                    "bytes": 0,
                    "bits_per_second": 0.0,
                    "jitter_ms": 0.0,
                    "lost_packets": 100,
                    "packets": 100,
                    "lost_percent": 100.0,
                }
            )
            completed = subprocess.CompletedProcess(
                ["iperf3"],
                0,
                json.dumps(total_loss),
                "",
            )
            with patch("oai_profile_workload.run_command", return_value=completed):
                total_result = workload.run_iperf3(spec, root, "10.0.0.4")
            total_direction = next(
                row
                for row in total_result["directions"]
                if row["direction"] == "nrUE_to_ext-DN_UL"
            )
            self.assertEqual(total_direction["receiver_lost_percent"], 100.0)
            self.assertTrue(total_direction["receiver_packet_metrics_valid"])
            self.assertIsNone(total_direction["receiver_jitter_ms"])
            self.assertFalse(total_direction["receiver_jitter_valid"])
            self.assertEqual(
                total_direction["receiver_jitter_status"],
                "no_received_datagrams_jitter_unavailable",
            )

            inconsistent_zero = iperf_result()
            invalid_receiver = inconsistent_zero["end"]["sum_received"]  # type: ignore[index]
            invalid_receiver["packets"] = 0  # type: ignore[index]
            invalid_receiver["lost_packets"] = 0  # type: ignore[index]
            invalid_receiver["lost_percent"] = 0.0  # type: ignore[index]
            completed = subprocess.CompletedProcess(
                ["iperf3"],
                0,
                json.dumps(inconsistent_zero),
                "",
            )
            with patch("oai_profile_workload.run_command", return_value=completed):
                with self.assertRaisesRegex(
                    workload.WorkloadError,
                    "zero-delivery receiver evidence is inconsistent",
                ):
                    workload.run_iperf3(spec, root, "10.0.0.4")

            duration_boundary = iperf_result()
            for view in duration_boundary["end"].values():  # type: ignore[union-attr]
                view["seconds"] = 122.39  # type: ignore[index]
            completed = subprocess.CompletedProcess(
                ["iperf3"],
                0,
                json.dumps(duration_boundary),
                "",
            )
            with patch("oai_profile_workload.run_command", return_value=completed):
                boundary_result = workload.run_iperf3(
                    spec,
                    root,
                    "10.0.0.4",
                )
            self.assertEqual(boundary_result["status"], "ok")

            wrong_target = iperf_result()
            wrong_target["start"]["connecting_to"]["host"] = "192.168.70.136"  # type: ignore[index]
            completed = subprocess.CompletedProcess(
                ["iperf3"],
                0,
                json.dumps(wrong_target),
                "",
            )
            with patch("oai_profile_workload.run_command", return_value=completed):
                with self.assertRaisesRegex(
                    workload.WorkloadError,
                    "expected 192.168.70.135",
                ):
                    workload.run_iperf3(spec, root, "10.0.0.4")
            self.assertEqual(
                json.loads((root / "workload" / "iperf3.json").read_text())[
                    "start"
                ]["connecting_to"]["host"],
                "192.168.70.136",
            )

            wrong_start = iperf_result()
            wrong_start["start"]["test_start"]["protocol"] = "TCP"  # type: ignore[index]
            completed = subprocess.CompletedProcess(
                ["iperf3"],
                0,
                json.dumps(wrong_start),
                "",
            )
            with patch("oai_profile_workload.run_command", return_value=completed):
                with self.assertRaisesRegex(
                    workload.WorkloadError,
                    "test_start.protocol",
                ):
                    workload.run_iperf3(spec, root, "10.0.0.4")

            missing_reverse = iperf_result()
            del missing_reverse["end"]["sum_received_bidir_reverse"]  # type: ignore[index]
            completed = subprocess.CompletedProcess(
                ["iperf3"],
                0,
                json.dumps(missing_reverse),
                "",
            )
            with patch("oai_profile_workload.run_command", return_value=completed):
                with self.assertRaisesRegex(
                    workload.WorkloadError,
                    "sum_received_bidir_reverse",
                ):
                    workload.run_iperf3(spec, root, "10.0.0.4")
            invalid_evidence = json.loads(
                (root / "workload" / "iperf3_run.json").read_text()
            )
            self.assertEqual(invalid_evidence["status"], "failed")
            self.assertIn(
                "sum_received_bidir_reverse",
                invalid_evidence["validation_error"],
            )

    def test_ping_and_iperf_timeouts_preserve_partial_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-workload-timeout-") as temporary:
            root = Path(temporary)
            spec = workload.parse_workload_spec(spec_value(root))
            ping_timeout = subprocess.TimeoutExpired(
                ["ping"],
                16,
                output=b"3 packets transmitted, 2 received, 33% packet loss\n",
                stderr=b"ping partial diagnostic\n",
            )
            with patch("oai_profile_workload.run_command", side_effect=ping_timeout):
                with self.assertRaisesRegex(workload.WorkloadError, "timed out"):
                    workload.run_ping(spec, root, "10.0.0.4")
            ping = json.loads((root / "workload" / "ping.json").read_text())
            self.assertEqual(ping["status"], "timeout")
            self.assertEqual((ping["transmitted"], ping["received"]), (3, 2))
            self.assertEqual(
                (root / "workload" / "ping.stderr.log").read_text(),
                "ping partial diagnostic\n",
            )

            iperf_timeout = subprocess.TimeoutExpired(
                ["iperf3"],
                150,
                output=b'{"partial":true}\n',
                stderr=b"iperf partial diagnostic\n",
            )
            with patch("oai_profile_workload.run_command", side_effect=iperf_timeout):
                with self.assertRaisesRegex(workload.WorkloadError, "timed out"):
                    workload.run_iperf3(spec, root, "10.0.0.4")
            iperf = json.loads(
                (root / "workload" / "iperf3_run.json").read_text()
            )
            self.assertEqual(iperf["status"], "timeout")
            self.assertEqual(iperf["validation_error"], "subprocess timeout")
            self.assertEqual(
                (root / "workload" / "iperf3.json").read_text(),
                '{"partial":true}\n',
            )
            self.assertEqual(
                (root / "workload" / "iperf3.stderr.log").read_text(),
                "iperf partial diagnostic\n",
            )

    def test_workload_interruption_is_a_terminal_preserved_state(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-workload-interrupt-") as temporary:
            root = Path(temporary)
            spec = workload.parse_workload_spec(spec_value(root))
            with (
                patch("oai_profile_workload.verify_lease"),
                patch(
                    "oai_profile_workload.wait_for_readiness",
                    side_effect=KeyboardInterrupt(),
                ),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    workload.run_workload(spec, root, "experiment-t001")
            state = json.loads(
                (root / "workload" / "workload_run.json").read_text()
            )
            self.assertEqual(state["status"], "interrupted")
            self.assertEqual(state["workload_status"], "interrupted")
            self.assertIn("KeyboardInterrupt", state["error"])

    def test_readiness_retries_absence_but_fails_integrity_immediately(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-workload-readiness-") as temporary:
            root = Path(temporary)
            spec = workload.parse_workload_spec(spec_value(root))
            active = {
                "rules": [
                    rule(12000, src="10.0.0.4/32"),
                    rule(12001, dst="10.0.0.4/32"),
                ],
                "routes": [route()],
            }
            with (
                patch(
                    "oai_profile_workload.discover_global_ipv4",
                    side_effect=[
                        workload.ReadinessPending("not attached"),
                        "10.0.0.4",
                    ],
                ),
                patch("oai_profile_workload.network_snapshot", return_value=active),
                patch("oai_profile_workload.time.sleep"),
            ):
                ipv4, snapshot = workload.wait_for_readiness(spec, root)
            self.assertEqual(ipv4, "10.0.0.4")
            self.assertEqual(snapshot, active)
            ready = json.loads(
                (root / "workload" / "workload_readiness.json").read_text()
            )
            self.assertEqual([row["status"] for row in ready["attempts"]], ["pending", "ready"])

            with (
                patch(
                    "oai_profile_workload.discover_global_ipv4",
                    side_effect=workload.WorkloadError("malformed policy state"),
                ),
                patch("oai_profile_workload.time.sleep") as sleep,
            ):
                with self.assertRaisesRegex(workload.WorkloadError, "malformed"):
                    workload.wait_for_readiness(spec, root)
            sleep.assert_not_called()
            failed = json.loads(
                (root / "workload" / "workload_readiness.json").read_text()
            )
            self.assertEqual(failed["status"], "failed_integrity")
            self.assertEqual(len(failed["attempts"]), 1)

    def test_preflight_detects_residue_without_deleting_it(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-workload-preflight-") as temporary:
            root = Path(temporary)
            spec = workload.parse_workload_spec(spec_value(root))
            run_dir = root / "run"
            residue = {"rules": [rule(12000, src="10.0.0.2/32")], "routes": []}
            with (
                patch("oai_profile_workload.shutil.which", return_value="/usr/bin/tool"),
                patch("oai_profile_workload.process_ids_by_name", return_value=[]),
                patch("oai_profile_workload.interface_exists", return_value=False),
                patch("oai_profile_workload.network_snapshot", return_value=residue),
                patch("oai_profile_workload.run_command") as run,
            ):
                with self.assertRaisesRegex(workload.WorkloadError, "policy residue"):
                    workload.preflight(spec, run_dir, "experiment-t001")
            self.assertFalse(Path(spec.lease_path).exists())
            run.assert_not_called()
            evidence = json.loads(
                (run_dir / "workload" / "network_preflight.json").read_text()
            )
            self.assertEqual(evidence["status"], "failed")

    def test_cleanup_deletes_only_exact_owned_identities_and_releases_lease(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-workload-cleanup-") as temporary:
            root = Path(temporary)
            spec = workload.parse_workload_spec(spec_value(root))
            run_dir = root / "run"
            evidence = run_dir / "workload"
            evidence.mkdir(parents=True)
            workload.acquire_lease(spec, run_dir, "experiment-t001")
            baseline = {"rules": [], "routes": []}
            ownership = {
                "rules": [
                    rule(12000, src="10.0.0.4/32"),
                    rule(12001, dst="10.0.0.4/32"),
                ],
                "routes": [route()],
            }
            workload.atomic_write_json(evidence / "network_baseline.json", baseline)
            workload.atomic_write_json(evidence / "network_ownership.json", ownership)
            before = {
                "rules": ownership["rules"],
                "routes": ownership["routes"],
            }
            after = {"rules": [], "routes": []}
            completed = subprocess.CompletedProcess(["ip"], 0, "", "")
            with (
                patch("oai_profile_workload.process_ids_by_name", return_value=[]),
                patch("oai_profile_workload.interface_exists", return_value=False),
                patch(
                    "oai_profile_workload.network_snapshot",
                    side_effect=[before, after],
                ),
                patch("oai_profile_workload.run_command", return_value=completed) as run,
            ):
                workload.cleanup(spec, run_dir, "experiment-t001")
            self.assertFalse(Path(spec.lease_path).exists())
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(commands[0][0:4], ["ip", "-4", "route", "del"])
            self.assertEqual(commands[1][0:4], ["ip", "-4", "rule", "del"])
            self.assertEqual(commands[2][0:4], ["ip", "-4", "rule", "del"])
            result = json.loads((evidence / "network_cleanup.json").read_text())
            self.assertEqual(result["status"], "ok")
            self.assertTrue(result["lease_released"])

    def test_cleanup_is_idempotent_and_refuses_unowned_state_or_competitor(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oai-workload-safe-cleanup-") as temporary:
            root = Path(temporary)
            spec = workload.parse_workload_spec(spec_value(root))
            run_dir = root / "run"
            evidence = run_dir / "workload"
            evidence.mkdir(parents=True)
            baseline = {"rules": [], "routes": []}
            ownership = {
                "rules": [rule(12000, src="10.0.0.4/32")],
                "routes": [],
            }
            workload.atomic_write_json(evidence / "network_baseline.json", baseline)
            workload.atomic_write_json(evidence / "network_ownership.json", ownership)

            workload.acquire_lease(spec, run_dir, "experiment-t001")
            empty = {"rules": [], "routes": []}
            with (
                patch("oai_profile_workload.process_ids_by_name", return_value=[]),
                patch("oai_profile_workload.interface_exists", return_value=False),
                patch("oai_profile_workload.network_snapshot", side_effect=[empty, empty]),
                patch("oai_profile_workload.run_command") as run,
            ):
                workload.cleanup(spec, run_dir, "experiment-t001")
            run.assert_not_called()
            result = json.loads((evidence / "network_cleanup.json").read_text())
            self.assertEqual(result["status"], "already_absent")

            workload.acquire_lease(spec, run_dir, "experiment-t002")
            with (
                patch("oai_profile_workload.process_ids_by_name", return_value=[321]),
                patch("oai_profile_workload.interface_exists", return_value=False),
                patch("oai_profile_workload.run_command") as run,
            ):
                with self.assertRaisesRegex(workload.WorkloadError, "verified nrUE death"):
                    workload.cleanup(spec, run_dir, "experiment-t002")
            run.assert_not_called()
            self.assertTrue(Path(spec.lease_path).exists())

            Path(spec.lease_path, "owner.json").unlink()
            Path(spec.lease_path).rmdir()

            workload.acquire_lease(spec, run_dir, "experiment-t003")
            unowned = {"rules": [rule(13000, src="10.0.0.9/32")], "routes": []}
            with (
                patch("oai_profile_workload.process_ids_by_name", return_value=[]),
                patch("oai_profile_workload.interface_exists", return_value=False),
                patch("oai_profile_workload.network_snapshot", return_value=unowned),
                patch("oai_profile_workload.run_command") as run,
            ):
                with self.assertRaisesRegex(workload.WorkloadError, "unowned policy rule"):
                    workload.cleanup(spec, run_dir, "experiment-t003")
            run.assert_not_called()
            self.assertTrue(Path(spec.lease_path).exists())


if __name__ == "__main__":
    unittest.main()
