#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0

from __future__ import annotations

import unittest
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oai_profile_campaign_semantics import (  # noqa: E402
    classify_role_termination,
    pair_measurement_contracts_match,
    paired_workload_contract_is_valid,
    role_termination_succeeded,
    sidecar_evidence_is_valid,
    workload_evidence_is_valid,
    workload_contracts_match,
)


def controlled_row(*, remote: bool, return_code: int, status: str = "exited_nonzero") -> dict[str, object]:
    return {
        "host": "cm5" if remote else "local",
        "status": status,
        "return_code": return_code,
        "stop_reason": "measurement_complete",
        "completion_classifier_version": "1",
        "controlled_stop_requested": 1,
        "shutdown_stage": "SIGINT",
        "shutdown_verified": 1,
        "remote_completion_identity_verified": 1 if remote else "not_applicable",
        "remote_completion_return_code": return_code if remote else "",
    }


class CampaignTerminationSemanticsTest(unittest.TestCase):
    def test_accepts_endpoint_specific_proven_controlled_sigint(self) -> None:
        cases = (
            (controlled_row(remote=False, return_code=-2), "controlled_sigint_signal"),
            (
                controlled_row(remote=False, return_code=0, status="finished"),
                "controlled_sigint_zero",
            ),
            (controlled_row(remote=True, return_code=130), "controlled_sigint_signal"),
            (
                controlled_row(remote=True, return_code=0, status="finished"),
                "controlled_sigint_zero",
            ),
        )
        for row, expected in cases:
            with self.subTest(row=row):
                self.assertEqual(classify_role_termination(row), expected)
                self.assertTrue(role_termination_succeeded(row))

    def test_rejects_unproven_wrong_endpoint_and_escalated_stops(self) -> None:
        local = controlled_row(remote=False, return_code=-2)
        remote = controlled_row(remote=True, return_code=130)
        cases = (
            {**local, "completion_classifier_version": ""},
            {**local, "controlled_stop_requested": 0},
            {**local, "shutdown_stage": "none"},
            {**local, "shutdown_stage": "SIGTERM"},
            {**local, "shutdown_stage": "SIGKILL"},
            {**local, "shutdown_verified": 0},
            {**local, "stop_reason": "interrupted"},
            {**local, "stop_reason": "paired_role_exited"},
            {**local, "return_code": 130},
            {**local, "return_code": -15},
            {**local, "return_code": -9},
            {**local, "status": "finished"},
            {**local, "remote_completion_identity_verified": 0},
            {**local, "remote_completion_identity_verified": 1},
            {**local, "remote_completion_return_code": 130},
            {**remote, "remote_completion_identity_verified": 0},
            {**remote, "remote_completion_return_code": ""},
            {**remote, "remote_completion_return_code": 0},
            {**remote, "return_code": -2},
            {**remote, "return_code": 255},
            {**remote, "return_code": 0},
            {**remote, "status": "remote_return_unverified", "return_code": ""},
        )
        for row in cases:
            with self.subTest(row=row):
                self.assertFalse(role_termination_succeeded(row))

    def test_legacy_zero_is_preserved_but_legacy_nonzero_is_not_reclassified(self) -> None:
        legacy_zero = {
            "host": "cm5",
            "status": "finished",
            "return_code": "0",
            "stop_reason": "duration_elapsed",
        }
        self.assertEqual(classify_role_termination(legacy_zero), "natural_zero")
        self.assertTrue(role_termination_succeeded(legacy_zero))
        self.assertTrue(
            role_termination_succeeded(
                {**legacy_zero, "remote_completion_return_code": 0}
            )
        )
        self.assertFalse(
            role_termination_succeeded(
                {**legacy_zero, "remote_completion_return_code": 130}
            )
        )
        self.assertFalse(
            role_termination_succeeded(
                {
                    **legacy_zero,
                    "host": "local",
                    "remote_completion_return_code": 0,
                }
            )
        )
        for return_code in (-2, 130):
            row = {
                **legacy_zero,
                "status": "exited_nonzero",
                "return_code": return_code,
            }
            with self.subTest(return_code=return_code):
                self.assertEqual(classify_role_termination(row), "exited_nonzero")
                self.assertFalse(role_termination_succeeded(row))

    def test_current_and_partial_completion_evidence_fail_closed(self) -> None:
        graceful = controlled_row(remote=False, return_code=0, status="finished")
        for row in (
            {**graceful, "controlled_stop_requested": ""},
            {**graceful, "controlled_stop_requested": "malformed"},
            {**graceful, "completion_classifier_version": "2"},
            {
                **graceful,
                "controlled_stop_requested": 0,
                "shutdown_stage": "SIGINT",
            },
            {
                **graceful,
                "controlled_stop_requested": 0,
                "shutdown_stage": "SIGTERM",
            },
            {
                **graceful,
                "controlled_stop_requested": 0,
                "shutdown_stage": "SIGKILL",
            },
            {
                **graceful,
                "controlled_stop_requested": 0,
                "shutdown_stage": "none",
                "shutdown_verified": 0,
            },
            {
                **controlled_row(remote=True, return_code=0, status="finished"),
                "controlled_stop_requested": 0,
                "shutdown_stage": "none",
                "remote_completion_identity_verified": 0,
            },
            {
                "host": "local",
                "status": "finished",
                "return_code": 0,
                "stop_reason": "duration_elapsed",
                "controlled_stop_requested": 1,
            },
            {
                "host": "local",
                "status": "finished",
                "return_code": 0,
                "stop_reason": "duration_elapsed",
                "runner_version": "2",
            },
            {
                "host": "local",
                "status": "finished",
                "return_code": 0,
                "stop_reason": "duration_elapsed",
                "termination_class": "natural_zero",
            },
        ):
            with self.subTest(row=row):
                self.assertFalse(role_termination_succeeded(row))

        new_natural = {
            **graceful,
            "controlled_stop_requested": 0,
            "shutdown_stage": "none",
        }
        self.assertEqual(classify_role_termination(new_natural), "natural_zero")
        self.assertTrue(role_termination_succeeded(new_natural))
        remote_natural = {
            **controlled_row(remote=True, return_code=0, status="finished"),
            "controlled_stop_requested": 0,
            "shutdown_stage": "none",
        }
        self.assertEqual(classify_role_termination(remote_natural), "natural_zero")
        self.assertTrue(role_termination_succeeded(remote_natural))

    def test_sidecar_evidence_is_exact_for_current_and_compatible_for_legacy(self) -> None:
        current = {
            "completion_classifier_version": "1",
            "sidecar_status": "not_requested",
            "sidecar_artifact": "",
        }
        self.assertTrue(
            sidecar_evidence_is_valid(
                {**current, "sidecar": "none"},
                tool_field="sidecar",
            )
        )
        self.assertFalse(
            sidecar_evidence_is_valid(
                {**current, "sidecar_tool": "none", "sidecar_status": ""},
                tool_field="sidecar_tool",
            )
        )
        self.assertFalse(
            sidecar_evidence_is_valid(
                {
                    **current,
                    "runner_version": "2",
                    "sidecar_tool": "none",
                    "sidecar_status": "",
                },
                tool_field="sidecar_tool",
            )
        )
        self.assertFalse(
            sidecar_evidence_is_valid(
                {
                    **current,
                    "sidecar_tool": "invented",
                    "sidecar_status": "registered",
                    "sidecar_artifact": "sidecars/invented.data",
                },
                tool_field="sidecar_tool",
            )
        )
        self.assertFalse(
            sidecar_evidence_is_valid(
                {
                    **current,
                    "sidecar_tool": "none",
                    "sidecar_config_json": '{"tool":"perf_record"}',
                },
                tool_field="sidecar_tool",
            )
        )
        for partial in (
            {"sidecar_evidence_fields_present": "partial", "sidecar_tool": "none"},
            {
                "sidecar_evidence_fields_present": 0,
                "sidecar_tool": "perf_stat",
            },
            {
                "sidecar_evidence_fields_present": 0,
                "sidecar_status": "registered",
            },
        ):
            with self.subTest(partial=partial):
                self.assertFalse(
                    sidecar_evidence_is_valid(partial, tool_field="sidecar_tool")
                )
        self.assertTrue(
            sidecar_evidence_is_valid(
                {
                    "sidecar_tool": "none",
                    "sidecar_status": "",
                    "sidecar_artifact": "",
                },
                tool_field="sidecar_tool",
            )
        )
        self.assertTrue(
            sidecar_evidence_is_valid(
                {
                    **current,
                    "sidecar_tool": "perf_stat",
                    "sidecar_status": "registered",
                    "sidecar_artifact": "sidecars/perf_stat.csv",
                },
                tool_field="sidecar_tool",
            )
        )
        self.assertFalse(
            sidecar_evidence_is_valid(
                {
                    **current,
                    "sidecar_tool": "perf_stat",
                    "sidecar_status": "registered",
                    "sidecar_artifact": "",
                },
                tool_field="sidecar_tool",
            )
        )

    def test_workload_contract_parity_is_exact(self) -> None:
        unloaded = {
            "workload_status": "not_configured",
            "workload_artifact": "",
            "network_cleanup_status": "not_configured",
        }
        loaded = {
            "workload_status": "completed",
            "workload_artifact": "workload/workload_run.json",
            "network_cleanup_status": "ok",
        }
        self.assertTrue(workload_contracts_match([unloaded, dict(unloaded)]))
        self.assertTrue(workload_contracts_match([loaded, dict(loaded)]))
        self.assertFalse(workload_contracts_match([unloaded, loaded]))
        self.assertFalse(
            workload_contracts_match(
                [loaded, {**loaded, "workload_artifact": "workload/other.json"}]
            )
        )
        current = {
            "completion_classifier_version": "1",
            "stop_reason": "duration_elapsed",
        }
        self.assertFalse(workload_evidence_is_valid(current))
        for partial in (
            {"workload_evidence_fields_present": "partial", "workload_status": "completed"},
            {"workload_evidence_fields_present": 0, "workload_status": "completed"},
            {
                "workload_evidence_fields_present": 0,
                "network_cleanup_status": "ok",
            },
        ):
            with self.subTest(partial=partial):
                self.assertFalse(
                    workload_evidence_is_valid(
                        {**partial, "stop_reason": "duration_elapsed"}
                    )
                )
        current_unloaded = {
            **current,
            "workload_status": "not_configured",
            "workload_artifact": "",
            "network_cleanup_status": "not_configured",
        }
        self.assertTrue(workload_evidence_is_valid(current_unloaded))
        self.assertTrue(
            paired_workload_contract_is_valid(
                [
                    {**current_unloaded, "role": "gNB"},
                    {**current_unloaded, "role": "nrUE"},
                ]
            )
        )

    def test_pair_measurement_contract_is_exact(self) -> None:
        common = {
            "completion_classifier_version": "1",
            "profile_enabled": 1,
            "pmu_mode": "off",
            "sidecar_tool": "none",
            "sidecar_status": "not_requested",
            "sidecar_artifact": "",
            "sidecar_config_json": '{"tool":"none"}',
        }
        pair = [{**common, "role": "gNB"}, {**common, "role": "nrUE"}]
        self.assertTrue(
            pair_measurement_contracts_match(
                pair,
                tool_field="sidecar_tool",
                sidecar_config_field="sidecar_config_json",
            )
        )
        for changed in (
            {**pair[1], "profile_enabled": 0},
            {**pair[1], "pmu_mode": "software"},
            {
                **pair[1],
                "sidecar_tool": "perf_stat",
                "sidecar_status": "registered",
                "sidecar_artifact": "sidecars/perf-stat.csv",
                "sidecar_config_json": '{"tool":"perf_stat"}',
            },
            {**pair[1], "sidecar_config_json": '{"tool":"perf_record"}'},
        ):
            with self.subTest(changed=changed):
                self.assertFalse(
                    pair_measurement_contracts_match(
                        [pair[0], changed],
                        tool_field="sidecar_tool",
                        sidecar_config_field="sidecar_config_json",
                    )
                )

    def test_unlaunched_and_unverified_states_remain_explicit(self) -> None:
        prepared = {
            "host": "local",
            "status": "prepared",
            "return_code": "",
            "completion_classifier_version": "1",
            "controlled_stop_requested": 0,
            "shutdown_stage": "none",
            "shutdown_verified": 0,
        }
        self.assertEqual(classify_role_termination(prepared), "not_launched")
        self.assertEqual(
            classify_role_termination(
                {**prepared, "status": "remote_return_unverified"}
            ),
            "shutdown_unverified",
        )


if __name__ == "__main__":
    unittest.main()
