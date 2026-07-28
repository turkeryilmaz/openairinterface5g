#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Pure, versioned campaign process-termination semantics."""

from __future__ import annotations

import json
import signal
from collections import Counter
from collections.abc import Mapping


COMPLETION_CLASSIFIER_VERSION = "1"
ALLOWED_PMU_MODES = {"off", "auto", "software", "hardware", "all"}
ALLOWED_SIDECAR_TOOLS = {"none", "perf_stat", "perf_record", "perf_sched"}
CONTROLLED_COMPLETION_STOP_REASONS = {
    "duration_elapsed",
    "measurement_complete",
}
SUCCESSFUL_TERMINATION_CLASSES = {
    "natural_zero",
    "controlled_sigint_zero",
    "controlled_sigint_signal",
}
COMPLETION_PROOF_FIELDS = {
    "controlled_stop_requested",
    "shutdown_stage",
    "shutdown_verified",
    "remote_completion_identity_verified",
}


def parse_exact_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def parse_evidence_flag(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    return None


def endpoint_is_remote(row: Mapping[str, object]) -> bool:
    return str(row.get("host", "")) not in {"", "local", "localhost"}


def completion_proof_is_present(row: Mapping[str, object]) -> bool:
    fields_present = any(
        field_name in row and str(row.get(field_name, "")).strip()
        for field_name in COMPLETION_PROOF_FIELDS
    )
    recorded_class = (
        row.get("termination_class_recorded", "")
        if "termination_class_recomputed" in row
        else row.get("termination_class", "")
    )
    return fields_present or bool(str(recorded_class).strip())


def current_evidence_contract_is_present(row: Mapping[str, object]) -> bool:
    runner_version = str(row.get("runner_version", "")).strip()
    return (
        bool(str(row.get("completion_classifier_version", "")).strip())
        or runner_version not in {"", "1"}
        or completion_proof_is_present(row)
    )


def evidence_fields_state(
    row: Mapping[str, object],
    *,
    marker_field: str,
    evidence_fields: tuple[str, ...],
) -> str:
    marker = str(row.get(marker_field, "")).strip().lower()
    if marker in {"absent", "partial", "complete"}:
        return marker
    marker_flag = parse_evidence_flag(row.get(marker_field))
    if marker_flag is True:
        return "complete"
    if marker_flag is False:
        return (
            "partial"
            if any(str(row.get(field_name, "")).strip() for field_name in evidence_fields)
            else "absent"
        )
    present_count = sum(field_name in row for field_name in evidence_fields)
    if present_count == 0:
        return "absent"
    return "complete" if present_count == len(evidence_fields) else "partial"


def workload_contract(row: Mapping[str, object]) -> tuple[str, str, str]:
    state = evidence_fields_state(
        row,
        marker_field="workload_evidence_fields_present",
        evidence_fields=(
            "workload_status",
            "workload_artifact",
            "network_cleanup_status",
        ),
    )
    if not current_evidence_contract_is_present(row) and state == "absent":
        return ("not_configured", "", "not_configured")
    defaults = (
        ("", "", "")
        if current_evidence_contract_is_present(row)
        else ("not_configured", "", "not_configured")
    )
    return (
        str(row.get("workload_status", defaults[0])),
        str(row.get("workload_artifact", defaults[1])),
        str(row.get("network_cleanup_status", defaults[2])),
    )


def workload_evidence_is_valid(row: Mapping[str, object]) -> bool:
    current = current_evidence_contract_is_present(row)
    state = evidence_fields_state(
        row,
        marker_field="workload_evidence_fields_present",
        evidence_fields=(
            "workload_status",
            "workload_artifact",
            "network_cleanup_status",
        ),
    )
    if state == "partial" or (current and state != "complete"):
        return False
    status, artifact, cleanup = workload_contract(row)
    stop_reason = str(row.get("stop_reason", ""))
    if status == "not_configured":
        return (
            not artifact
            and cleanup == "not_configured"
            and stop_reason == "duration_elapsed"
        )
    return (
        status == "completed"
        and bool(artifact)
        and cleanup in {"ok", "already_absent"}
        and stop_reason == "measurement_complete"
    )


def workload_contracts_match(rows: list[Mapping[str, object]]) -> bool:
    return len({workload_contract(row) for row in rows}) <= 1


def paired_workload_contract_is_valid(rows: list[Mapping[str, object]]) -> bool:
    return (
        len(rows) == 2
        and Counter(str(row.get("role", "")) for row in rows)
        == Counter({"gNB": 1, "nrUE": 1})
        and all(workload_evidence_is_valid(row) for row in rows)
        and workload_contracts_match(rows)
    )


def sidecar_evidence_is_valid(
    row: Mapping[str, object],
    *,
    tool_field: str,
) -> bool:
    current = current_evidence_contract_is_present(row)
    state = evidence_fields_state(
        row,
        marker_field="sidecar_evidence_fields_present",
        evidence_fields=(tool_field, "sidecar_status", "sidecar_artifact"),
    )
    if state == "partial" or (current and state != "complete"):
        return False
    if not current and state == "absent":
        tool = "none"
        status = ""
        artifact = ""
    else:
        tool = str(row.get(tool_field, "none" if not current else ""))
        status = str(row.get("sidecar_status", ""))
        artifact = str(row.get("sidecar_artifact", ""))
    if tool not in ALLOWED_SIDECAR_TOOLS:
        return False
    sidecar_config_json = str(row.get("sidecar_config_json", ""))
    if sidecar_config_json:
        try:
            sidecar_config = json.loads(sidecar_config_json)
        except json.JSONDecodeError:
            return False
        if (
            not isinstance(sidecar_config, dict)
            or str(sidecar_config.get("tool", "none")) != tool
        ):
            return False
    if tool == "none":
        legacy_blank = (
            not current
            and status == ""
        )
        return (status == "not_requested" or legacy_blank) and not artifact
    return status == "registered" and bool(artifact)


def sidecar_contracts_match(
    rows: list[Mapping[str, object]],
    *,
    tool_field: str,
) -> bool:
    return (
        bool(rows)
        and
        all(
            sidecar_evidence_is_valid(row, tool_field=tool_field)
            for row in rows
        )
        and len(
            {
                "none"
                if (
                    not current_evidence_contract_is_present(row)
                    and evidence_fields_state(
                        row,
                        marker_field="sidecar_evidence_fields_present",
                        evidence_fields=(
                            tool_field,
                            "sidecar_status",
                            "sidecar_artifact",
                        ),
                    )
                    == "absent"
                )
                else str(row.get(tool_field, ""))
                for row in rows
            }
        )
        <= 1
    )


def pair_measurement_contracts_match(
    rows: list[Mapping[str, object]],
    *,
    tool_field: str,
    sidecar_config_field: str | None = None,
) -> bool:
    if not sidecar_contracts_match(rows, tool_field=tool_field):
        return False
    profile_values = [parse_evidence_flag(row.get("profile_enabled")) for row in rows]
    pmu_values = [str(row.get("pmu_mode", "")) for row in rows]
    if (
        any(value is None for value in profile_values)
        or len(set(profile_values)) != 1
        or any(value not in ALLOWED_PMU_MODES for value in pmu_values)
        or len(set(pmu_values)) != 1
    ):
        return False
    if sidecar_config_field is None:
        return True
    configs = [str(row.get(sidecar_config_field, "")) for row in rows]
    if any(
        current_evidence_contract_is_present(row) and not config
        for row, config in zip(rows, configs)
    ):
        return False
    nonempty_configs = {config for config in configs if config}
    return len(nonempty_configs) <= 1


def natural_zero_evidence_is_consistent(row: Mapping[str, object]) -> bool:
    if str(row.get("shutdown_stage", "")) != "none":
        return False
    if parse_evidence_flag(row.get("shutdown_verified")) is not True:
        return False
    remote_identity = row.get("remote_completion_identity_verified")
    if endpoint_is_remote(row):
        return (
            parse_evidence_flag(remote_identity) is True
            and parse_exact_int(row.get("remote_completion_return_code")) == 0
        )
    return (
        str(remote_identity).strip() == "not_applicable"
        and str(row.get("remote_completion_return_code", "")).strip() == ""
    )


def legacy_endpoint_return_is_consistent(row: Mapping[str, object]) -> bool:
    raw_remote_return = str(row.get("remote_completion_return_code", "")).strip()
    if endpoint_is_remote(row):
        return (
            not raw_remote_return
            or parse_exact_int(raw_remote_return)
            == parse_exact_int(row.get("return_code"))
        )
    return not raw_remote_return


def controlled_sigint_is_proven(row: Mapping[str, object]) -> bool:
    if str(row.get("completion_classifier_version", "")) != COMPLETION_CLASSIFIER_VERSION:
        return False
    if parse_evidence_flag(row.get("controlled_stop_requested")) is not True:
        return False
    if str(row.get("stop_reason", "")) not in CONTROLLED_COMPLETION_STOP_REASONS:
        return False
    if str(row.get("shutdown_stage", "")) != "SIGINT":
        return False
    if parse_evidence_flag(row.get("shutdown_verified")) is not True:
        return False
    remote_identity = row.get("remote_completion_identity_verified")
    if endpoint_is_remote(row):
        if (
            parse_evidence_flag(remote_identity) is not True
            or parse_exact_int(row.get("remote_completion_return_code"))
            != parse_exact_int(row.get("return_code"))
        ):
            return False
    elif (
        str(remote_identity).strip() != "not_applicable"
        or str(row.get("remote_completion_return_code", "")).strip()
    ):
        return False
    return True


def classify_role_termination(
    row: Mapping[str, object],
    *,
    status_field: str = "status",
) -> str:
    status = str(row.get(status_field, ""))
    return_code = parse_exact_int(row.get("return_code"))
    proof_version = str(row.get("completion_classifier_version", ""))
    controlled_requested = (
        parse_evidence_flag(row.get("controlled_stop_requested")) is True
    )
    controlled_sigint = controlled_sigint_is_proven(row)

    if status == "finished" and return_code == 0:
        if not proof_version:
            if (
                completion_proof_is_present(row)
                or str(row.get("runner_version", "")).strip() not in {"", "1"}
            ):
                return "completion_evidence_malformed"
            return (
                "natural_zero"
                if legacy_endpoint_return_is_consistent(row)
                else "return_evidence_mismatch"
            )
        if proof_version != COMPLETION_CLASSIFIER_VERSION:
            return "completion_classifier_unsupported"
        requested_flag = parse_evidence_flag(row.get("controlled_stop_requested"))
        if requested_flag is None:
            return "completion_evidence_malformed"
        if requested_flag:
            return (
                "controlled_sigint_zero"
                if controlled_sigint
                else "controlled_stop_unverified"
            )
        return (
            "natural_zero"
            if natural_zero_evidence_is_consistent(row)
            else "natural_zero_unverified"
        )

    if status == "exited_nonzero" and controlled_sigint:
        expected_return_code = 128 + signal.SIGINT if endpoint_is_remote(row) else -signal.SIGINT
        if return_code == expected_return_code:
            return "controlled_sigint_signal"

    if status in {
        "planned",
        "prepared",
        "launch_failed",
        "launch_not_attempted",
    }:
        return "not_launched"
    if status == "shutdown_failed" or (
        proof_version == COMPLETION_CLASSIFIER_VERSION
        and parse_evidence_flag(row.get("shutdown_verified")) is False
    ):
        return "shutdown_unverified"
    if status in {"remote_return_unverified", "return_unverified"} or return_code is None:
        return "return_unverified"
    if controlled_requested:
        return "controlled_stop_unverified"
    return "exited_nonzero"


def role_termination_succeeded(
    row: Mapping[str, object],
    *,
    status_field: str = "status",
) -> bool:
    return (
        classify_role_termination(row, status_field=status_field)
        in SUCCESSFUL_TERMINATION_CLASSES
    )


def is_controlled_sigint_class(termination_class: str) -> bool:
    return termination_class in {
        "controlled_sigint_zero",
        "controlled_sigint_signal",
    }
