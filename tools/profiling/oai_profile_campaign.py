#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Plan and execute paired, repeated OAI profiling campaigns.

Execution is opt-in through --execute. The default is a read-only plan that
prints every case/variant/trial and its effective, redacted command.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))

from oai_profile_archive import finalize_archive, register_external_source  # noqa: E402
from oai_profile_workload import (  # noqa: E402
    WorkloadSpec,
    command_plan as workload_command_plan,
    parse_workload_spec,
)


CAMPAIGN_SCHEMA_VERSION = 1
RUNNER_VERSION = "1"
CONTROL_SCHEMA_VERSION = 1
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
CONTROL_TOKEN = re.compile(r"^[0-9a-f]{32}$")
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
REMOTE_CONTROL_TIMEOUT_S = 20.0
REMOTE_ACTION_STOP_GRACE_S = 2.0
SENSITIVE_FRAGMENTS = (
    "password",
    "passwd",
    "secret",
    "token",
    "credential",
    ".key",
    ".opc",
    "imsi",
    "supi",
    "imei",
)
DEFAULT_PERF_EVENTS = [
    "task-clock",
    "context-switches",
    "cpu-migrations",
    "page-faults",
    "cycles",
    "instructions",
    "branches",
    "branch-misses",
    "cache-references",
    "cache-misses",
]
RESULT_FIELDS = [
    "campaign_id",
    "experiment_id",
    "case",
    "variant",
    "trial",
    "role",
    "host",
    "run_dir",
    "profile_enabled",
    "pmu_mode",
    "sidecar",
    "start_realtime_ns",
    "end_realtime_ns",
    "start_monotonic_raw_ns",
    "end_monotonic_raw_ns",
    "duration_s",
    "duration_clock",
    "duration_status",
    "realtime_clock_regressed",
    "transport_return_code",
    "remote_completion_return_code",
    "return_code",
    "run_status",
    "stop_reason",
    "archive_status",
    "sidecar_status",
    "workload_status",
    "workload_artifact",
    "network_cleanup_status",
    "notes",
]
PRE_IDENTITY_RESULT_FIELDS = [
    field_name
    for field_name in RESULT_FIELDS
    if field_name
    not in {"transport_return_code", "remote_completion_return_code"}
]
LEGACY_RESULT_FIELDS = [
    field_name
    for field_name in PRE_IDENTITY_RESULT_FIELDS
    if field_name
    not in {"workload_status", "workload_artifact", "network_cleanup_status"}
]


@dataclass(frozen=True)
class Endpoint:
    role: str
    host: str
    hostname: str
    profile_root: str
    command: list[str]
    cwd: str | None
    sudo: bool
    environment: dict[str, str]
    archive_tool: str | None
    launch_delay_s: float

    @property
    def remote(self) -> bool:
        return self.host not in {"", "local", "localhost"}


@dataclass(frozen=True)
class Variant:
    name: str
    profile: bool
    pmu: str
    sidecar: dict[str, Any]
    environment: dict[str, str]


@dataclass(frozen=True)
class ExperimentPlan:
    campaign_id: str
    case_name: str
    variant: Variant
    trial: int
    experiment_id: str
    role_overrides: dict[str, dict[str, Any]]
    environment: dict[str, str]


@dataclass
class ProcessHandle:
    endpoint: Endpoint
    run_dir: str
    command: list[str]
    environment: dict[str, str]
    process: subprocess.Popen[bytes] | None = None
    stdout_handle: Any = None
    stderr_handle: Any = None
    start_realtime_ns: int = 0
    start_monotonic_ns: int = 0
    end_realtime_ns: int = 0
    end_monotonic_ns: int = 0
    stop_reason: str = ""
    sidecar_tool: str = "none"
    sidecar_artifact: str = ""
    sidecar_version: str = ""
    archive_status: str = "pending"
    sidecar_status: str = "not_requested"
    workload_status: str = "not_configured"
    workload_artifact: str = ""
    network_cleanup_status: str = "not_configured"
    workload_control_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    experiment_id: str = ""
    control_token: str = ""
    transport_return_code: int | None = None
    remote_completion_return_code: int | None = None
    shutdown_verified: bool = False
    run_status: str = "planned"
    initial_manifest: dict[str, Any] = field(default_factory=dict)
    prepared: bool = False
    owner_uid: int = -1
    owner_gid: int = -1
    launch_index: int = -1
    notes: list[str] = field(default_factory=list)

    @property
    def control_action(self) -> str:
        return f"role:{self.endpoint.role}"

    @property
    def control_start_path(self) -> str:
        return f"{self.run_dir}/control.start"

    @property
    def control_completion_path(self) -> str:
        return f"{self.run_dir}/control.complete"


@dataclass
class WorkloadHandle:
    spec: WorkloadSpec
    endpoint: Endpoint
    run_dir: str
    experiment_id: str
    process: subprocess.Popen[bytes] | None = None
    stdout_handle: Any = None
    stderr_handle: Any = None
    status: str = "planned"
    cleanup_status: str = "pending"
    preflight_complete: bool = False
    cleanup_required: bool = False
    shutdown_verified: bool = False
    evidence_quiesced: bool = False
    synchronous_action_attempted: bool = False
    closed: bool = False
    control_tokens: dict[str, str] = field(default_factory=dict)
    control_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def action_control_start_path(self, action: str) -> str:
        if action not in {"preflight", "run", "cleanup"}:
            raise ValueError(f"unsupported workload action: {action}")
        return f"{self.run_dir}/workload/{action}.control.start"

    def action_control_completion_path(self, action: str) -> str:
        if action not in {"preflight", "run", "cleanup"}:
            raise ValueError(f"unsupported workload action: {action}")
        return f"{self.run_dir}/workload/{action}.control.complete"


def elapsed_duration(
    start_monotonic_ns: int,
    end_monotonic_ns: int,
    start_realtime_ns: int,
    end_realtime_ns: int,
) -> tuple[float | str, str, str]:
    if start_monotonic_ns > 0 or end_monotonic_ns > 0:
        if start_monotonic_ns > 0 and end_monotonic_ns >= start_monotonic_ns:
            return (
                (end_monotonic_ns - start_monotonic_ns) / 1e9,
                "CLOCK_MONOTONIC_RAW",
                "valid",
            )
        return "", "CLOCK_MONOTONIC_RAW", "invalid_monotonic_bounds"
    if start_realtime_ns > 0 or end_realtime_ns > 0:
        if start_realtime_ns > 0 and end_realtime_ns >= start_realtime_ns:
            return (
                (end_realtime_ns - start_realtime_ns) / 1e9,
                "CLOCK_REALTIME",
                "legacy_realtime_fallback",
            )
        return "", "CLOCK_REALTIME", "invalid_realtime_bounds"
    return "", "unavailable", "unavailable"


def require_safe_name(value: str, label: str) -> str:
    if not SAFE_NAME.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def string_dict(value: Any, label: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ValueError(f"{label} must be an object containing string values")
    return dict(value)


def string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a nonempty string array")
    return list(value)


def parse_endpoint(role: str, value: Any) -> Endpoint:
    if not isinstance(value, dict):
        raise ValueError(f"roles.{role} must be an object")
    host = str(value.get("host", "local"))
    default_hostname = socket.gethostname() if host in {"", "local", "localhost"} else ""
    hostname = require_safe_name(str(value.get("hostname", default_hostname)), f"roles.{role}.hostname")
    profile_root_value = str(value.get("profile_root", ""))
    profile_root = PurePosixPath(profile_root_value)
    if not profile_root_value or not profile_root.is_absolute() or ".." in profile_root.parts:
        raise ValueError(f"roles.{role}.profile_root must be an absolute path without '..'")
    command = string_list(value.get("command"), f"roles.{role}.command")
    launch_delay_s = float(value.get("launch_delay_s", 0.0))
    if launch_delay_s < 0:
        raise ValueError(f"roles.{role}.launch_delay_s must be nonnegative")
    if command[0] == "sudo" or Path(command[0]).name == "sudo":
        raise ValueError(f"roles.{role}.command must omit sudo; use the sudo boolean")
    return Endpoint(
        role=require_safe_name(role, "role"),
        host=host,
        hostname=hostname,
        profile_root=str(profile_root),
        command=command,
        cwd=str(value["cwd"]) if value.get("cwd") else None,
        sudo=bool(value.get("sudo", False)),
        environment=string_dict(value.get("environment"), f"roles.{role}.environment"),
        archive_tool=str(value["archive_tool"]) if value.get("archive_tool") else None,
        launch_delay_s=launch_delay_s,
    )


def parse_variant(value: Any) -> Variant:
    if not isinstance(value, dict):
        raise ValueError("each variant must be an object")
    name = require_safe_name(str(value.get("name", "")), "variant name")
    sidecar_value = value.get("sidecar", "none")
    if isinstance(sidecar_value, str):
        sidecar = {"tool": sidecar_value}
    elif isinstance(sidecar_value, dict):
        sidecar = dict(sidecar_value)
    else:
        raise ValueError(f"variant {name} sidecar must be a string or object")
    tool = sidecar.get("tool", "none")
    if tool not in {"none", "perf_stat", "perf_record", "perf_sched"}:
        raise ValueError(f"variant {name} has unsupported sidecar: {tool}")
    pmu = str(value.get("pmu", "off"))
    if pmu not in {"off", "auto", "software", "hardware", "all"}:
        raise ValueError(f"variant {name} has invalid PMU mode: {pmu}")
    return Variant(
        name=name,
        profile=bool(value.get("profile", True)),
        pmu=pmu,
        sidecar=sidecar,
        environment=string_dict(value.get("environment"), f"variant {name} environment"),
    )


def load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text())
    if spec.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
        raise ValueError(f"campaign schema_version must be {CAMPAIGN_SCHEMA_VERSION}")
    require_safe_name(str(spec.get("campaign_id", "")), "campaign_id")
    if not isinstance(spec.get("roles"), dict) or not spec["roles"]:
        raise ValueError("roles must be a nonempty object")
    if not isinstance(spec.get("variants"), list) or not spec["variants"]:
        raise ValueError("variants must be a nonempty array")
    if not isinstance(spec.get("cases"), list) or not spec["cases"]:
        raise ValueError("cases must be a nonempty array")
    trials = int(spec.get("trials", 1))
    duration_s = float(spec.get("duration_s", 60.0))
    stop_grace_s = float(spec.get("stop_grace_s", 10.0))
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if stop_grace_s < 0:
        raise ValueError("stop_grace_s must be nonnegative")
    if trials < 1:
        raise ValueError("trials must be positive")
    spec["_endpoints"] = {role: parse_endpoint(role, value) for role, value in spec["roles"].items()}
    start_order = spec.get("start_order", list(spec["_endpoints"]))
    if (
        not isinstance(start_order, list)
        or not all(isinstance(role, str) for role in start_order)
        or set(start_order) != set(spec["_endpoints"])
        or len(start_order) != len(spec["_endpoints"])
    ):
        raise ValueError("start_order must contain every role exactly once")
    spec["_start_order"] = start_order
    workload_value = spec.get("workload")
    workload = parse_workload_spec(workload_value) if workload_value is not None else None
    if workload is not None:
        if workload.client_role not in spec["_endpoints"]:
            raise ValueError(f"workload client role does not exist: {workload.client_role}")
        if float(workload.duration_s) != duration_s:
            raise ValueError(
                "campaign duration_s must exactly equal workload duration_s "
                "so the measurement window has one declared duration"
            )
    spec["_workload"] = workload
    spec["_variants"] = [parse_variant(value) for value in spec["variants"]]
    variant_names = [variant.name for variant in spec["_variants"]]
    if len(set(variant_names)) != len(variant_names):
        raise ValueError("variant names must be unique")
    spec["_trials"] = trials
    return spec


def experiment_id(campaign_id: str, case_name: str, variant_name: str, trial: int) -> str:
    return require_safe_name(f"{campaign_id}-{case_name}-{variant_name}-t{trial:03d}", "experiment_id")


def build_plans(spec: dict[str, Any], cases: set[str], variants: set[str], trials: set[int]) -> list[ExperimentPlan]:
    campaign_id = str(spec["campaign_id"])
    plans: list[ExperimentPlan] = []
    case_names: set[str] = set()
    for case in spec["cases"]:
        if not isinstance(case, dict):
            raise ValueError("each case must be an object")
        case_name = require_safe_name(str(case.get("name", "")), "case name")
        if case_name in case_names:
            raise ValueError(f"duplicate case name: {case_name}")
        case_names.add(case_name)
        if cases and case_name not in cases:
            continue
        role_overrides = case.get("roles", {})
        if not isinstance(role_overrides, dict):
            raise ValueError(f"case {case_name} roles must be an object")
        for role, override in role_overrides.items():
            if role not in spec["_endpoints"] or not isinstance(override, dict):
                raise ValueError(f"invalid override for case {case_name} role {role}")
        case_environment = string_dict(case.get("environment"), f"case {case_name} environment")
        for variant in spec["_variants"]:
            if variants and variant.name not in variants:
                continue
            for trial in range(1, spec["_trials"] + 1):
                if trials and trial not in trials:
                    continue
                plans.append(
                    ExperimentPlan(
                        campaign_id=campaign_id,
                        case_name=case_name,
                        variant=variant,
                        trial=trial,
                        experiment_id=experiment_id(campaign_id, case_name, variant.name, trial),
                        role_overrides=role_overrides,
                        environment=case_environment,
                    )
                )
    if not plans:
        raise ValueError("filters selected no experiments")
    return plans


def is_sensitive(value: str) -> bool:
    lowered = value.lower()
    return any(fragment in lowered for fragment in SENSITIVE_FRAGMENTS)


def redact_command(command: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for argument in command:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
        elif is_sensitive(argument):
            if "=" in argument:
                key, _ = argument.split("=", 1)
                redacted.append(f"{key}=<redacted>")
            elif argument.startswith("-"):
                redacted.append(argument)
                redact_next = True
            else:
                redacted.append("<redacted>")
        else:
            redacted.append(argument)
    return redacted


def redact_environment(environment: dict[str, str]) -> dict[str, str]:
    return {key: "<redacted>" if is_sensitive(key) else value for key, value in sorted(environment.items())}


def role_command(endpoint: Endpoint, plan: ExperimentPlan) -> list[str]:
    override = plan.role_overrides.get(endpoint.role, {})
    command = (
        string_list(override["command"], f"case role {endpoint.role} command")
        if "command" in override
        else list(endpoint.command)
    )
    arguments = override.get("arguments", [])
    if not isinstance(arguments, list) or not all(isinstance(item, str) for item in arguments):
        raise ValueError(f"case role {endpoint.role} arguments must be a string array")
    return command + arguments


def role_environment(endpoint: Endpoint, plan: ExperimentPlan, run_dir: str) -> dict[str, str]:
    override = plan.role_overrides.get(endpoint.role, {})
    environment = dict(endpoint.environment)
    environment.update(plan.environment)
    environment.update(plan.variant.environment)
    environment.update(string_dict(override.get("environment"), f"case role {endpoint.role} environment"))
    environment.update(
        {
            "OAI_PROFILE": "1" if plan.variant.profile else "0",
            "OAI_PROFILE_EXPERIMENT_ID": plan.experiment_id,
            "OAI_PROFILE_CAMPAIGN_ID": plan.campaign_id,
            "OAI_PROFILE_VARIANT": plan.variant.name,
            "OAI_PROFILE_TRIAL": str(plan.trial),
        }
    )
    if plan.variant.profile:
        environment["OAI_PROFILE_DIR"] = run_dir
        environment["OAI_PROFILE_PMU"] = plan.variant.pmu
    return environment


def sidecar_command(command: list[str], sidecar: dict[str, Any], run_dir: str) -> tuple[list[str], str]:
    tool = str(sidecar.get("tool", "none"))
    if tool == "none":
        return command, ""
    sidecar_dir = f"{run_dir}/sidecars"
    if tool == "perf_stat":
        artifact = f"{sidecar_dir}/perf_stat.csv"
        events = sidecar.get("events", DEFAULT_PERF_EVENTS)
        if not isinstance(events, list) or not events or not all(isinstance(event, str) for event in events):
            raise ValueError("perf_stat events must be a nonempty string array")
        wrapper = ["perf", "stat", "--no-big-num", "-x", ";", "-o", artifact]
        for event in events:
            wrapper.extend(["-e", event])
        if "interval_ms" in sidecar:
            interval_ms = int(sidecar["interval_ms"])
            if interval_ms <= 0:
                raise ValueError("perf_stat interval_ms must be positive")
            wrapper.extend(["--interval-print", str(interval_ms)])
        return wrapper + ["--"] + command, artifact
    if tool == "perf_record":
        artifact = f"{sidecar_dir}/perf.data"
        event = str(sidecar.get("event", "cycles"))
        frequency = int(sidecar.get("frequency_hz", 199))
        call_graph = str(sidecar.get("call_graph", "dwarf,8192"))
        if frequency <= 0:
            raise ValueError("perf_record frequency_hz must be positive")
        if not event or not call_graph:
            raise ValueError("perf_record event and call_graph must be nonempty")
        return (
            [
                "perf",
                "record",
                "-o",
                artifact,
                "-e",
                event,
                "-F",
                str(frequency),
                "--call-graph",
                call_graph,
                "--clockid",
                "mono_raw",
                "--timestamp",
                "--",
            ]
            + command,
            artifact,
        )
    artifact = f"{sidecar_dir}/perf_sched.data"
    return ["perf", "sched", "record", "-o", artifact, "--"] + command, artifact


def effective_command(
    endpoint: Endpoint,
    command: list[str],
    environment: dict[str, str],
    sidecar: dict[str, Any],
    run_dir: str,
) -> tuple[list[str], str]:
    wrapped, artifact = sidecar_command(command, sidecar, run_dir)
    assignments = [f"{key}={value}" for key, value in sorted(environment.items())]
    if endpoint.sudo:
        return ["sudo", "-n", "env", *assignments, *wrapped], artifact
    if endpoint.remote:
        return ["env", *assignments, *wrapped], artifact
    return wrapped, artifact


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def remote_run(
    endpoint: Endpoint,
    command: str,
    check: bool = True,
    capture: bool = False,
    timeout_s: float | None = REMOTE_CONTROL_TIMEOUT_S,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", *SSH_OPTIONS, endpoint.host, command],
        check=check,
        text=True,
        capture_output=capture,
        timeout=timeout_s,
    )


def remote_upload(endpoint: Endpoint, local_path: Path, remote_path: str) -> None:
    subprocess.run(
        ["scp", *SSH_OPTIONS, "-p", str(local_path), f"{endpoint.host}:{remote_path}"],
        check=True,
    )


def write_remote_json(endpoint: Endpoint, remote_path: str, value: dict[str, Any]) -> None:
    with tempfile.TemporaryDirectory(prefix="oai-campaign-json-") as temporary:
        local_path = Path(temporary) / Path(remote_path).name
        atomic_write_json(local_path, value)
        remote_upload(endpoint, local_path, remote_path)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_endpoint_text(endpoint: Endpoint, path: str) -> str:
    if not endpoint.remote and not endpoint.sudo:
        return Path(path).read_text()
    command = ["cat", "--", path]
    if endpoint.sudo:
        command = ["sudo", "-n", *command]
    if endpoint.remote:
        result = remote_run(
            endpoint,
            shlex.join(command),
            check=False,
            capture=True,
        )
    else:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"could not read {endpoint.host}:{path}: "
            f"status={result.returncode}, stderr={result.stderr.strip()}"
        )
    return result.stdout


def write_endpoint_text(endpoint: Endpoint, path: str, value: str) -> None:
    if not endpoint.remote:
        atomic_write_text(Path(path), value)
        return
    with tempfile.TemporaryDirectory(prefix="oai-campaign-text-") as temporary:
        local_path = Path(temporary) / Path(path).name
        atomic_write_text(local_path, value)
        remote_upload(endpoint, local_path, path)


def read_endpoint_json(endpoint: Endpoint, path: str) -> dict[str, Any]:
    try:
        value = json.loads(read_endpoint_text(endpoint, path))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"malformed JSON evidence at {endpoint.host}:{path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON evidence is not an object at {endpoint.host}:{path}")
    return value


def workload_config_json(spec: WorkloadSpec) -> str:
    return json.dumps(asdict(spec), sort_keys=True, separators=(",", ":"))


def workload_action_command(
    workload: WorkloadHandle,
    action: str,
) -> list[str]:
    command = [
        "python3",
        workload.spec.helper,
        action,
        "--config-json",
        workload_config_json(workload.spec),
        "--run-dir",
        workload.run_dir,
        "--experiment-id",
        workload.experiment_id,
    ]
    return ["sudo", "-n", *command] if workload.endpoint.sudo else command


def control_record_text(record: dict[str, Any], completion: bool) -> str:
    fields = [
        "schema_version",
        "action",
        "experiment_id",
        "token",
        "pgid",
        "start_ticks",
    ]
    if completion:
        fields.append("return_code")
    return "".join(f"{field_name}={record[field_name]}\n" for field_name in fields)


def parse_control_record(
    text: str,
    *,
    action: str,
    experiment_id: str,
    token: str,
    completion: bool,
) -> dict[str, Any]:
    if not CONTROL_TOKEN.fullmatch(token):
        raise ValueError("control token must be 32 lowercase hexadecimal characters")
    expected_fields = {
        "schema_version",
        "action",
        "experiment_id",
        "token",
        "pgid",
        "start_ticks",
    }
    if completion:
        expected_fields.add("return_code")
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line or "=" not in line:
            raise ValueError("control record contains a malformed line")
        name, value = line.split("=", 1)
        if name in values:
            raise ValueError(f"control record repeats {name}")
        values[name] = value
    if set(values) != expected_fields:
        raise ValueError("control record field set is invalid")
    if values["schema_version"] != str(CONTROL_SCHEMA_VERSION):
        raise ValueError("control record schema is unsupported")
    expected_identity = {
        "action": action,
        "experiment_id": require_safe_name(experiment_id, "experiment_id"),
        "token": token,
    }
    for name, expected in expected_identity.items():
        if values[name] != expected:
            raise ValueError(f"control record {name} mismatch")
    for name in ("pgid", "start_ticks"):
        if not values[name].isdecimal() or int(values[name]) <= 0:
            raise ValueError(f"control record {name} must be positive decimal")
    record: dict[str, Any] = {
        "schema_version": CONTROL_SCHEMA_VERSION,
        **expected_identity,
        "pgid": int(values["pgid"]),
        "start_ticks": int(values["start_ticks"]),
    }
    if completion:
        return_code = values["return_code"]
        if not return_code.isdecimal() or not (0 <= int(return_code) <= 255):
            raise ValueError("control record return_code must be in [0, 255]")
        record["return_code"] = int(return_code)
    return record


def remote_control_inner_command(
    command: list[str],
    *,
    action: str,
    experiment_id: str,
    token: str,
    start_path: str,
    completion_path: str,
) -> str:
    if not CONTROL_TOKEN.fullmatch(token):
        raise ValueError("control token must be 32 lowercase hexadecimal characters")
    require_safe_name(experiment_id, "experiment_id")
    if action not in {"preflight", "run", "cleanup"}:
        if not action.startswith("role:"):
            raise ValueError(f"unsupported control action: {action}")
        require_safe_name(action.removeprefix("role:"), "control role")
    fixed_fields = [
        f"schema_version={CONTROL_SCHEMA_VERSION}",
        f"action={action}",
        f"experiment_id={experiment_id}",
        f"token={token}",
    ]
    start_values = " ".join(shlex.quote(value) for value in fixed_fields)
    start_temporary = f"{start_path}.tmp.$$"
    completion_temporary = f"{completion_path}.tmp.$$"
    payload = (
        "pgid=$$; "
        "proc_identity=$(awk '{print $5 \" \" $22}' /proc/$$/stat) || exit 46; "
        "set -- $proc_identity; "
        "[ \"$#\" -eq 2 ] && [ \"$1\" = \"$$\" ] || exit 46; "
        "start_ticks=$2; "
        f"printf '%s\\n' {start_values} \"pgid=$pgid\" "
        f"\"start_ticks=$start_ticks\" > {shlex.quote(start_temporary)} || exit 46; "
        f"mv -- {shlex.quote(start_temporary)} {shlex.quote(start_path)} || exit 46; "
        f"exec {shlex.join(command)}"
    )
    return (
        "umask 077; "
        f"if test -e {shlex.quote(start_path)} || "
        f"test -e {shlex.quote(completion_path)}; then exit 47; fi; "
        f"setsid sh -c {shlex.quote(payload)} & payload_pid=$!; "
        "wait \"$payload_pid\"; command_rc=$?; "
        f"actual=$(cat -- {shlex.quote(start_path)}) || exit 46; "
        "printf '%s\\n' \"$actual\" \"return_code=$command_rc\" "
        f"> {shlex.quote(completion_temporary)} || exit 46; "
        f"mv -- {shlex.quote(completion_temporary)} "
        f"{shlex.quote(completion_path)} || exit 46; "
        "exit \"$command_rc\""
    )


def read_remote_control_record(
    endpoint: Endpoint,
    path: str,
    *,
    action: str,
    experiment_id: str,
    token: str,
    completion: bool,
) -> tuple[str, dict[str, Any] | None, str]:
    command = (
        f"if ! test -e {shlex.quote(path)}; then exit 1; fi; "
        f"if ! test -f {shlex.quote(path)} || "
        f"! test -r {shlex.quote(path)}; then exit 44; fi; "
        f"cat -- {shlex.quote(path)}"
    )
    try:
        result = remote_run(endpoint, command, check=False, capture=True)
    except (OSError, subprocess.SubprocessError) as error:
        return "unknown", None, str(error)
    if result.returncode == 1:
        return "missing", None, ""
    if result.returncode != 0:
        return (
            "unknown",
            None,
            f"status={result.returncode}, stderr={result.stderr.strip()!r}",
        )
    try:
        record = parse_control_record(
            result.stdout,
            action=action,
            experiment_id=experiment_id,
            token=token,
            completion=completion,
        )
    except ValueError as error:
        return "invalid", None, str(error)
    return "present", record, ""


def remote_identity_probe_command(record: dict[str, Any], sudo: bool) -> str:
    pgid = int(record["pgid"])
    start_ticks = int(record["start_ticks"])
    kill = "sudo -n /bin/kill" if sudo else "/bin/kill"
    return (
        f"if ! test -e /proc/{pgid}/stat; then "
        f"probe=$(LC_ALL=C {kill} -0 -- -{pgid} 2>&1); probe_rc=$?; "
        "if [ \"$probe_rc\" -eq 0 ]; then exit 3; fi; "
        "case \"$probe\" in *'No such process'*) exit 1;; "
        "*) printf '%s\\n' \"$probe\" >&2; exit 44;; esac; fi; "
        f"if ! test -r /proc/{pgid}/stat; then exit 44; fi; "
        f"current=$(awk '{{print $22}}' /proc/{pgid}/stat) || exit 44; "
        f"if [ \"$current\" = {shlex.quote(str(start_ticks))} ]; "
        "then exit 0; else exit 2; fi"
    )


def remote_control_state(
    endpoint: Endpoint,
    *,
    action: str,
    experiment_id: str,
    token: str,
    start_path: str,
    completion_path: str,
) -> tuple[bool | None, dict[str, Any] | None, str]:
    start_status, start, start_detail = read_remote_control_record(
        endpoint,
        start_path,
        action=action,
        experiment_id=experiment_id,
        token=token,
        completion=False,
    )
    if start_status != "present" or start is None:
        return None, None, f"start record {start_status}: {start_detail}"
    completion_status, completed, completion_detail = read_remote_control_record(
        endpoint,
        completion_path,
        action=action,
        experiment_id=experiment_id,
        token=token,
        completion=True,
    )
    if completion_status == "present" and completed is not None:
        if (
            completed["pgid"] != start["pgid"]
            or completed["start_ticks"] != start["start_ticks"]
        ):
            return None, start, "completion identity does not match start record"
    elif completion_status != "missing":
        return (
            None,
            start,
            f"completion record {completion_status}: {completion_detail}",
        )
    observed_identity = completed if completed is not None else start
    try:
        result = remote_run(
            endpoint,
            remote_identity_probe_command(start, endpoint.sudo),
            check=False,
            capture=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return None, observed_identity, str(error)
    if result.returncode == 0:
        if completed is not None:
            return (
                None,
                observed_identity,
                "completion_present_but_original_leader_identity_observed",
            )
        return True, observed_identity, "matching_identity_live"
    if result.returncode == 1:
        detail = (
            "matching_completion_group_absent"
            if completed is not None
            else "original_identity_absent"
        )
        return False, observed_identity, detail
    if result.returncode == 2:
        detail = (
            "matching_completion_pgid_reused_original_identity_dead"
            if completed is not None
            else "pgid_reused_original_identity_dead"
        )
        return False, observed_identity, detail
    if result.returncode == 3:
        if completed is not None:
            return (
                True,
                observed_identity,
                "matching_completion_leader_absent_process_group_still_live",
            )
        return None, observed_identity, "leader_absent_process_group_still_live"
    return (
        None,
        observed_identity,
        f"identity probe status={result.returncode}, stderr={result.stderr.strip()!r}",
    )


def remote_identity_signal_command(
    start_path: str,
    record: dict[str, Any],
    signal_name: str,
    sudo: bool,
) -> str:
    expected = control_record_text(record, completion=False).rstrip("\n")
    pgid = int(record["pgid"])
    start_ticks = int(record["start_ticks"])
    kill = "sudo -n /bin/kill" if sudo else "/bin/kill"
    return (
        f"if ! test -r {shlex.quote(start_path)}; then exit 42; fi; "
        f"actual=$(cat -- {shlex.quote(start_path)}) || exit 44; "
        f"if [ \"$actual\" != {shlex.quote(expected)} ]; then exit 43; fi; "
        f"if ! test -r /proc/{pgid}/stat; then exit 45; fi; "
        f"current=$(awk '{{print $22}}' /proc/{pgid}/stat) || exit 44; "
        f"if [ \"$current\" != {shlex.quote(str(start_ticks))} ]; then exit 45; fi; "
        f"{kill} -{signal_name} -- -{pgid}"
    )


def wait_remote_control_not_live(
    endpoint: Endpoint,
    *,
    action: str,
    experiment_id: str,
    token: str,
    start_path: str,
    completion_path: str,
    timeout_s: float,
) -> tuple[bool | None, dict[str, Any] | None, str]:
    deadline = time.monotonic() + timeout_s
    while True:
        state, identity, detail = remote_control_state(
            endpoint,
            action=action,
            experiment_id=experiment_id,
            token=token,
            start_path=start_path,
            completion_path=completion_path,
        )
        if state is not True or time.monotonic() >= deadline:
            return state, identity, detail
        time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))


def stop_remote_controlled_action(
    endpoint: Endpoint,
    *,
    action: str,
    experiment_id: str,
    token: str,
    start_path: str,
    completion_path: str,
    identity: dict[str, Any],
) -> tuple[bool | None, str]:
    current_identity = identity
    last_state: bool | None = True
    details: list[str] = []
    for signal_name in ("INT", "TERM", "KILL"):
        command = remote_identity_signal_command(
            start_path,
            current_identity,
            signal_name,
            endpoint.sudo,
        )
        try:
            result = remote_run(
                endpoint,
                command,
                check=False,
                capture=True,
            )
        except BaseException as error:
            result = None
            details.append(f"{signal_name} signal proof failed: {error}")
        if result is not None and result.returncode != 0:
            details.append(
                f"{signal_name} signal proof status={result.returncode}, "
                f"stderr={result.stderr.strip()!r}"
            )
        try:
            state, observed_identity, detail = wait_remote_control_not_live(
                endpoint,
                action=action,
                experiment_id=experiment_id,
                token=token,
                start_path=start_path,
                completion_path=completion_path,
                timeout_s=REMOTE_ACTION_STOP_GRACE_S,
            )
        except BaseException as error:
            state = None
            observed_identity = None
            detail = f"{signal_name} post-signal identity proof failed: {error}"
        details.append(f"{signal_name} post-signal state: {detail}")
        last_state = state
        if state is False:
            return False, "; ".join(details)
        if state is True and observed_identity is not None:
            current_identity = observed_identity
        elif state is True:
            details.append(
                f"{signal_name} live state omitted matching identity; retaining prior identity"
            )
    return last_state, "; ".join(details)


def invoke_workload_action(
    workload: WorkloadHandle,
    action: str,
    timeout_s: float,
) -> subprocess.CompletedProcess[str]:
    workload.synchronous_action_attempted = True
    workload.evidence_quiesced = False
    command = workload_action_command(workload, action)
    stdout_path = f"{workload.run_dir}/workload/{action}.stdout.log"
    stderr_path = f"{workload.run_dir}/workload/{action}.stderr.log"
    if workload.endpoint.remote:
        if action in workload.control_tokens:
            raise RuntimeError(f"remote workload {action} was already invoked")
        token = secrets.token_hex(16)
        if not CONTROL_TOKEN.fullmatch(token):
            raise RuntimeError("generated workload control token is invalid")
        workload.control_tokens[action] = token
        workload.control_results[action] = {
            "transport_status": "pending",
            "transport_return_code": None,
            "remote_completion_return_code": None,
        }
        start_path = workload.action_control_start_path(action)
        completion_path = workload.action_control_completion_path(action)
        inner = remote_control_inner_command(
            command,
            action=action,
            experiment_id=workload.experiment_id,
            token=token,
            start_path=start_path,
            completion_path=completion_path,
        )
        shell = (
            f"exec setsid --wait sh -c {shlex.quote(inner)} "
            f">{shlex.quote(stdout_path)} 2>{shlex.quote(stderr_path)}"
        )
        result: subprocess.CompletedProcess[str] | None = None
        invocation_error: BaseException | None = None
        try:
            result = remote_run(
                workload.endpoint,
                shell,
                check=False,
                capture=True,
                timeout_s=timeout_s,
            )
            workload.control_results[action].update(
                {
                    "transport_status": "returned",
                    "transport_return_code": result.returncode,
                }
            )
        except BaseException as error:
            invocation_error = error
            workload.control_results[action]["transport_status"] = (
                f"{type(error).__name__}: {error}"
            )
        try:
            state, identity, detail = remote_control_state(
                workload.endpoint,
                action=action,
                experiment_id=workload.experiment_id,
                token=token,
                start_path=start_path,
                completion_path=completion_path,
            )
        except BaseException as error:
            state = None
            identity = None
            detail = f"identity proof raised {type(error).__name__}: {error}"
        if state is True and identity is not None:
            workload.notes.append(
                f"remote workload {action} remained live; starting identity-bound shutdown"
            )
            state, detail = stop_remote_controlled_action(
                workload.endpoint,
                action=action,
                experiment_id=workload.experiment_id,
                token=token,
                start_path=start_path,
                completion_path=completion_path,
                identity=identity,
            )
            if state is False:
                try:
                    _, identity, completion_detail = remote_control_state(
                        workload.endpoint,
                        action=action,
                        experiment_id=workload.experiment_id,
                        token=token,
                        start_path=start_path,
                        completion_path=completion_path,
                    )
                    detail = f"{detail}; final state: {completion_detail}"
                except BaseException as error:
                    identity = None
                    detail = (
                        f"{detail}; final completion proof raised "
                        f"{type(error).__name__}: {error}"
                    )
        if (
            state is False
            and identity is not None
            and isinstance(identity.get("return_code"), int)
        ):
            workload.control_results[action][
                "remote_completion_return_code"
            ] = identity["return_code"]
        workload.evidence_quiesced = state is False
        if not workload.evidence_quiesced:
            workload.notes.append(
                f"remote workload {action} shutdown is unverified: "
                f"{detail or 'matching identity remains active'}"
            )
        if invocation_error is not None:
            completion_return_code = workload.control_results[action][
                "remote_completion_return_code"
            ]
            if isinstance(invocation_error, (KeyboardInterrupt, SystemExit)):
                raise invocation_error
            if completion_return_code is None:
                raise invocation_error
            workload.notes.append(
                f"remote workload {action} transport failed after authoritative "
                f"completion: {type(invocation_error).__name__}: {invocation_error}"
            )
            result = subprocess.CompletedProcess(
                ["ssh", workload.endpoint.host],
                completion_return_code,
                "",
                str(invocation_error),
            )
        if result is None:
            raise RuntimeError(f"remote workload {action} produced no command result")
        return result
    with Path(stdout_path).open("w", encoding="utf-8") as stdout_stream, Path(
        stderr_path
    ).open("w", encoding="utf-8") as stderr_stream:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=stdout_stream,
            stderr=stderr_stream,
            timeout=timeout_s,
        )
    workload.evidence_quiesced = True
    return result


def refresh_workload_state(workload: WorkloadHandle) -> dict[str, Any]:
    state = read_endpoint_json(
        workload.endpoint,
        f"{workload.run_dir}/workload/workload_run.json",
    )
    if state.get("schema_version") != 1:
        raise RuntimeError("workload_run.json has an unsupported schema")
    if state.get("experiment_id") != workload.experiment_id:
        raise RuntimeError("workload_run.json experiment identity mismatch")
    workload.state = state
    return state


def endpoint_path_exists(endpoint: Endpoint, path: str) -> bool:
    if not endpoint.remote:
        return Path(path).exists()
    result = remote_run(endpoint, f"test -e {shlex.quote(path)}", check=False)
    return result.returncode == 0


def prepare_run_dir(handle: ProcessHandle) -> None:
    endpoint = handle.endpoint
    run_dir = handle.run_dir
    if endpoint_path_exists(endpoint, run_dir):
        raise FileExistsError(f"run directory already exists: {endpoint.host}:{run_dir}")
    if endpoint.remote:
        parent = str(PurePosixPath(run_dir).parent)
        workload_directory = (
            f" && mkdir {shlex.quote(run_dir + '/workload')}"
            if handle.workload_status != "not_configured"
            else ""
        )
        remote_run(
            endpoint,
            f"mkdir -p {shlex.quote(parent)} && "
            f"mkdir {shlex.quote(run_dir)} && "
            f"mkdir {shlex.quote(run_dir + '/sidecars')}"
            f"{workload_directory}",
        )
        handle.prepared = True
        write_remote_json(endpoint, f"{run_dir}/campaign_run.json", handle.initial_manifest)
    else:
        Path(run_dir).mkdir(parents=True, exist_ok=False)
        handle.prepared = True
        Path(run_dir, "sidecars").mkdir()
        if handle.workload_status != "not_configured":
            Path(run_dir, "workload").mkdir()
        atomic_write_json(Path(run_dir, "campaign_run.json"), handle.initial_manifest)


def perf_version(endpoint: Endpoint) -> str:
    if endpoint.remote:
        result = remote_run(endpoint, "perf --version", check=False, capture=True)
    else:
        result = subprocess.run(["perf", "--version"], check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"perf unavailable on {endpoint.host}: {result.stderr.strip()}")
    return result.stdout.strip()


def endpoint_numeric_ids(endpoint: Endpoint) -> tuple[int, int]:
    if not endpoint.remote:
        return os.getuid(), os.getgid()
    result = remote_run(endpoint, "id -u && id -g", capture=True)
    values = result.stdout.splitlines()
    if len(values) != 2:
        raise RuntimeError(f"could not determine numeric user identity on {endpoint.host}")
    return int(values[0]), int(values[1])


def preflight_endpoint(handle: ProcessHandle) -> None:
    endpoint = handle.endpoint
    if endpoint.remote:
        if not endpoint.archive_tool:
            raise ValueError(f"remote role {endpoint.role} requires archive_tool")
        result = remote_run(endpoint, f"test -x {shlex.quote(endpoint.archive_tool)}", check=False, capture=True)
        if result.returncode != 0:
            raise RuntimeError(f"archive tool is not executable on {endpoint.host}: {endpoint.archive_tool}")
        result = remote_run(endpoint, "setsid --wait sh -c 'exit 0'", check=False, capture=True)
        if result.returncode != 0:
            raise RuntimeError(f"setsid --wait is unavailable on {endpoint.host}: {result.stderr.strip()}")
    if handle.sidecar_tool != "none":
        handle.sidecar_version = perf_version(endpoint)
    if endpoint.sudo:
        if endpoint.remote:
            result = remote_run(endpoint, "sudo -n true", check=False, capture=True)
        else:
            result = subprocess.run(["sudo", "-n", "true"], check=False, text=True, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"non-interactive sudo is unavailable on {endpoint.host}: {result.stderr.strip()}")
        handle.owner_uid, handle.owner_gid = endpoint_numeric_ids(endpoint)


def launch_local(handle: ProcessHandle) -> None:
    stdout_handle = Path(handle.run_dir, "stdout.log").open("wb")
    stderr_handle = Path(handle.run_dir, "stderr.log").open("wb")
    process_environment = os.environ.copy()
    if not handle.endpoint.sudo:
        process_environment.update(handle.environment)
    handle.start_realtime_ns = time.time_ns()
    handle.start_monotonic_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    try:
        handle.process = subprocess.Popen(
            handle.command,
            cwd=handle.endpoint.cwd,
            env=process_environment,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
    except Exception:
        stdout_handle.close()
        stderr_handle.close()
        raise
    handle.stdout_handle = stdout_handle
    handle.stderr_handle = stderr_handle
    handle.run_status = "launched"


def launch_remote(handle: ProcessHandle, control_dir: Path) -> None:
    if handle.control_token:
        raise RuntimeError(f"remote role {handle.endpoint.role} was already launched")
    handle.control_token = secrets.token_hex(16)
    if not CONTROL_TOKEN.fullmatch(handle.control_token):
        raise RuntimeError("generated role control token is invalid")
    require_safe_name(handle.experiment_id, "experiment_id")
    working_directory = handle.endpoint.cwd or "."
    inner = remote_control_inner_command(
        handle.command,
        action=handle.control_action,
        experiment_id=handle.experiment_id,
        token=handle.control_token,
        start_path=handle.control_start_path,
        completion_path=handle.control_completion_path,
    )
    shell = (
        f"cd {shlex.quote(working_directory)} && exec setsid --wait sh -c {shlex.quote(inner)} "
        f">{shlex.quote(handle.run_dir + '/stdout.log')} 2>{shlex.quote(handle.run_dir + '/stderr.log')}"
    )
    control_dir.mkdir(parents=True, exist_ok=True)
    transport_stdout = Path(control_dir, f"{handle.endpoint.role}_ssh_stdout.log").open("wb")
    transport_stderr = Path(control_dir, f"{handle.endpoint.role}_ssh_stderr.log").open("wb")
    handle.start_realtime_ns = time.time_ns()
    handle.start_monotonic_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    try:
        handle.process = subprocess.Popen(
            ["ssh", *SSH_OPTIONS, handle.endpoint.host, shell],
            stdout=transport_stdout,
            stderr=transport_stderr,
            start_new_session=True,
        )
    except Exception:
        transport_stdout.close()
        transport_stderr.close()
        raise
    handle.stdout_handle = transport_stdout
    handle.stderr_handle = transport_stderr
    handle.run_status = "launched"
    handle.notes.append("start/end monotonic anchors are orchestrator-clock values for remote roles")


def launch_workload(workload: WorkloadHandle, control_dir: Path) -> None:
    workload.evidence_quiesced = False
    command = workload_action_command(workload, "run")
    stdout_path = f"{workload.run_dir}/workload/run.stdout.log"
    stderr_path = f"{workload.run_dir}/workload/run.stderr.log"
    if workload.endpoint.remote:
        if "run" in workload.control_tokens:
            raise RuntimeError("remote workload run was already launched")
        token = secrets.token_hex(16)
        if not CONTROL_TOKEN.fullmatch(token):
            raise RuntimeError("generated workload run control token is invalid")
        workload.control_tokens["run"] = token
        workload.control_results["run"] = {
            "transport_status": "running",
            "transport_return_code": None,
            "remote_completion_return_code": None,
        }
        inner = remote_control_inner_command(
            command,
            action="run",
            experiment_id=workload.experiment_id,
            token=token,
            start_path=workload.action_control_start_path("run"),
            completion_path=workload.action_control_completion_path("run"),
        )
        shell = (
            f"exec setsid --wait sh -c {shlex.quote(inner)} "
            f">{shlex.quote(stdout_path)} 2>{shlex.quote(stderr_path)}"
        )
        control_dir.mkdir(parents=True, exist_ok=True)
        stdout_handle = Path(control_dir, "workload_ssh_stdout.log").open("wb")
        stderr_handle = Path(control_dir, "workload_ssh_stderr.log").open("wb")
        try:
            workload.process = subprocess.Popen(
                ["ssh", *SSH_OPTIONS, workload.endpoint.host, shell],
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
            )
        except Exception:
            stdout_handle.close()
            stderr_handle.close()
            raise
    else:
        stdout_handle = Path(stdout_path).open("wb")
        stderr_handle = Path(stderr_path).open("wb")
        try:
            workload.process = subprocess.Popen(
                command,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
            )
        except Exception:
            stdout_handle.close()
            stderr_handle.close()
            raise
    workload.stdout_handle = stdout_handle
    workload.stderr_handle = stderr_handle
    workload.status = "running"


def preflight_workload(workload: WorkloadHandle) -> None:
    result = invoke_workload_action(workload, "preflight", REMOTE_CONTROL_TIMEOUT_S)
    if not workload.evidence_quiesced:
        workload.status = "preflight_failed"
        raise RuntimeError("workload preflight process shutdown is unverified")
    return_code = (
        workload.control_results.get("preflight", {}).get(
            "remote_completion_return_code"
        )
        if workload.endpoint.remote
        else result.returncode
    )
    if return_code != 0:
        workload.status = "preflight_failed"
        try:
            refresh_workload_state(workload)
        except Exception as error:
            workload.notes.append(f"preflight evidence unavailable: {error}")
        raise RuntimeError(
            f"workload preflight failed with authoritative status {return_code}; "
            f"transport status {result.returncode}"
        )
    workload.cleanup_required = True
    state = refresh_workload_state(workload)
    if state.get("status") != "preflight_complete":
        workload.status = "preflight_failed"
        raise RuntimeError("workload preflight did not produce complete structured evidence")
    workload.preflight_complete = True
    workload.status = "preflight_complete"


def append_workload_note_once(workload: WorkloadHandle, note: str) -> None:
    if note not in workload.notes:
        workload.notes.append(note)


def workload_remote_control_state(
    workload: WorkloadHandle,
) -> tuple[bool | None, dict[str, Any] | None, str]:
    if not workload.endpoint.remote or workload.process is None:
        return False, None, "not_remote_or_not_launched"
    token = workload.control_tokens.get("run")
    if token is None:
        return None, None, "run control token is unavailable"
    state, identity, detail = remote_control_state(
        workload.endpoint,
        action="run",
        experiment_id=workload.experiment_id,
        token=token,
        start_path=workload.action_control_start_path("run"),
        completion_path=workload.action_control_completion_path("run"),
    )
    result = workload.control_results.setdefault(
        "run",
        {
            "transport_status": "unknown",
            "transport_return_code": None,
            "remote_completion_return_code": None,
        },
    )
    if workload.process.poll() is not None:
        result["transport_status"] = "returned"
        result["transport_return_code"] = workload.process.returncode
    if (
        state is False
        and identity is not None
        and isinstance(identity.get("return_code"), int)
    ):
        result["remote_completion_return_code"] = identity["return_code"]
    return state, identity, detail


def workload_remote_group_is_running(workload: WorkloadHandle) -> bool | None:
    state, _, detail = workload_remote_control_state(workload)
    if state is None:
        append_workload_note_once(
            workload,
            f"remote workload process-group state unavailable: {detail}",
        )
    return state


def workload_shutdown_target_is_running(workload: WorkloadHandle) -> bool:
    if workload.process is None:
        return False
    if not workload.endpoint.remote:
        return workload.process.poll() is None
    remote_state = workload_remote_group_is_running(workload)
    return True if remote_state is None else remote_state or workload.process.poll() is None


def workload_shutdown_is_verified(workload: WorkloadHandle) -> bool:
    if workload.process is None:
        return True
    if workload.process.poll() is None:
        return False
    if workload.endpoint.remote:
        return workload_remote_group_is_running(workload) is False
    return True


def signal_workload(workload: WorkloadHandle, signal_name: str) -> None:
    if workload.process is None:
        return
    if workload.endpoint.remote:
        state, identity, detail = workload_remote_control_state(workload)
        if state is None or (state is True and identity is None):
            append_workload_note_once(
                workload,
                f"remote workload {signal_name} refused: {detail}",
            )
            if signal_name == "KILL" and workload.process.poll() is None:
                try:
                    os.killpg(workload.process.pid, signal.SIGKILL)
                    append_workload_note_once(
                        workload,
                        "killed local workload SSH transport after unverifiable remote KILL",
                    )
                except OSError as error:
                    append_workload_note_once(
                        workload,
                        f"local workload SSH KILL failed: {error}",
                    )
            return
        if state is False:
            if workload.process.poll() is None:
                try:
                    os.killpg(
                        workload.process.pid,
                        getattr(signal, f"SIG{signal_name}"),
                    )
                except OSError as error:
                    append_workload_note_once(
                        workload,
                        f"local workload SSH {signal_name} failed: {error}",
                    )
            return
        assert identity is not None
        command = remote_identity_signal_command(
            workload.action_control_start_path("run"),
            identity,
            signal_name,
            workload.endpoint.sudo,
        )
        try:
            result = remote_run(
                workload.endpoint,
                command,
                check=False,
                capture=True,
            )
        except (OSError, subprocess.SubprocessError) as error:
            result = None
            append_workload_note_once(
                workload,
                f"remote workload {signal_name} failed: {error}",
            )
        if result is not None and result.returncode != 0:
            append_workload_note_once(
                workload,
                f"remote workload {signal_name} failed: {result.stderr.strip()}",
            )
        if signal_name == "KILL" and workload.process.poll() is None:
            try:
                os.killpg(workload.process.pid, signal.SIGKILL)
            except OSError as error:
                append_workload_note_once(
                    workload,
                    f"local workload SSH KILL failed: {error}",
                )
        return
    if workload.process.poll() is not None:
        return
    if workload.endpoint.sudo:
        try:
            subprocess.run(
                [
                    "sudo",
                    "-n",
                    "kill",
                    f"-{signal_name}",
                    "--",
                    f"-{workload.process.pid}",
                ],
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            workload.notes.append(f"local workload {signal_name} failed: {error}")
    else:
        try:
            os.killpg(workload.process.pid, getattr(signal, f"SIG{signal_name}"))
        except OSError as error:
            workload.notes.append(f"local workload {signal_name} failed: {error}")


def wait_workload_stopped(workload: WorkloadHandle, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not workload_shutdown_target_is_running(workload):
            return True
        time.sleep(0.25)
    return not workload_shutdown_target_is_running(workload)


def stop_workload(workload: WorkloadHandle, grace_s: float) -> bool:
    if workload.process is None:
        workload.shutdown_verified = True
        return True
    failures: list[BaseException] = []
    try:
        if workload_shutdown_target_is_running(workload):
            signal_workload(workload, "INT")
    except BaseException as error:
        workload.notes.append(f"workload INT shutdown stage failed: {error}")
        failures.append(error)
    try:
        stopped_after_int = wait_workload_stopped(workload, grace_s)
    except BaseException as error:
        failures.append(error)
        stopped_after_int = False
    if not stopped_after_int:
        try:
            signal_workload(workload, "TERM")
        except BaseException as error:
            workload.notes.append(f"workload TERM shutdown stage failed: {error}")
            failures.append(error)
    try:
        stopped_after_term = wait_workload_stopped(workload, grace_s)
    except BaseException as error:
        failures.append(error)
        stopped_after_term = False
    if not stopped_after_term:
        try:
            signal_workload(workload, "KILL")
        except BaseException as error:
            workload.notes.append(f"workload KILL shutdown stage failed: {error}")
            failures.append(error)
    try:
        stopped = wait_workload_stopped(workload, grace_s)
    except BaseException as error:
        failures.append(error)
        stopped = False
    try:
        workload.shutdown_verified = stopped and workload_shutdown_is_verified(workload)
        workload.evidence_quiesced = workload.shutdown_verified
    except BaseException as error:
        workload.shutdown_verified = False
        workload.evidence_quiesced = False
        failures.append(error)
    if failures:
        raise failures[0]
    return workload.shutdown_verified


def close_workload(workload: WorkloadHandle) -> None:
    if workload.closed and workload.shutdown_verified:
        return
    if workload.process is not None:
        workload.shutdown_verified = workload_shutdown_is_verified(workload)
        workload.evidence_quiesced = workload.shutdown_verified
        if workload.shutdown_verified:
            workload.process.wait()
            if workload.process.returncode != 0 and workload.status == "running":
                workload.status = "failed"
        else:
            workload.evidence_quiesced = False
            workload.notes.append("workload process shutdown could not be verified")
            if workload.status == "running":
                workload.status = "failed"
            workload.closed = False
            return
    else:
        workload.shutdown_verified = True
        if not workload.synchronous_action_attempted:
            workload.evidence_quiesced = True
    if workload.stdout_handle is not None:
        workload.stdout_handle.close()
    if workload.stderr_handle is not None:
        workload.stderr_handle.close()
    workload.closed = True


def role_measurement_state(handle: ProcessHandle) -> str:
    if handle.process is None:
        return "ended"
    if handle.process.poll() is None:
        return "running"
    if not handle.endpoint.remote:
        return "ended"
    remote_state = remote_group_is_running(handle)
    if remote_state is True:
        return "running"
    if remote_state is False:
        return "ended"
    return "unknown"


def workload_measurement_state(workload: WorkloadHandle) -> str:
    if workload.process is None:
        return "ended"
    if workload.process.poll() is None:
        return "running"
    if not workload.endpoint.remote:
        return "ended"
    remote_state = workload_remote_group_is_running(workload)
    if remote_state is True:
        return "running"
    if remote_state is False:
        return "ended"
    return "unknown"


def monitor_loaded_measurement(
    handles: list[ProcessHandle],
    workload: WorkloadHandle,
) -> str:
    timeout_s = (
        workload.spec.readiness_timeout_s
        + workload.spec.ping_count * workload.spec.ping_timeout_s
        + workload.spec.duration_s
        + 60.0
    )
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        workload_state = workload_measurement_state(workload)
        if workload_state == "ended":
            close_workload(workload)
            if not workload.shutdown_verified:
                workload.status = "failed"
                workload.notes.append(
                    "workload completion rejected because shutdown was not verified"
                )
                return "workload_shutdown_unverified"
            state = refresh_workload_state(workload)
            if (
                workload.process is not None
                and (
                    workload.control_results.get("run", {}).get(
                        "remote_completion_return_code"
                    )
                    == 0
                    if workload.endpoint.remote
                    else workload.process.returncode == 0
                )
                and state.get("status") == "completed"
                and state.get("workload_status") == "completed"
                and state.get("iperf3_status") == "ok"
            ):
                workload.status = "completed"
                return "measurement_complete"
            workload.status = "failed"
            return "workload_failed"
        if workload_state == "unknown":
            workload.status = "failed"
            workload.notes.append("workload state became unverifiable during measurement")
            return "workload_state_unknown"
        role_states = [role_measurement_state(handle) for handle in handles]
        if "unknown" in role_states:
            workload.status = "aborted"
            return "paired_role_state_unknown"
        if "ended" in role_states:
            workload.status = "aborted_role_exit"
            return "paired_role_exited"
        time.sleep(0.25)
    workload.status = "failed"
    workload.notes.append(f"workload outer watchdog expired after {timeout_s}s")
    return "workload_timeout"


def cleanup_workload(workload: WorkloadHandle) -> None:
    workload.evidence_quiesced = False
    result = invoke_workload_action(workload, "cleanup", 60.0)
    if not workload.evidence_quiesced:
        workload.cleanup_status = "failed"
        raise RuntimeError("workload cleanup process shutdown is unverified")
    try:
        state = refresh_workload_state(workload)
        cleanup_status = str(state.get("cleanup_status", "failed"))
    except Exception as error:
        workload.cleanup_status = "failed"
        workload.notes.append(f"cleanup evidence unavailable: {error}")
        raise
    workload.cleanup_status = cleanup_status
    return_code = (
        workload.control_results.get("cleanup", {}).get(
            "remote_completion_return_code"
        )
        if workload.endpoint.remote
        else result.returncode
    )
    if return_code != 0 or cleanup_status not in {"ok", "already_absent"}:
        raise RuntimeError(
            f"workload cleanup failed: authoritative_status={return_code}, "
            f"transport_status={result.returncode}, "
            f"evidence={cleanup_status}"
        )
    workload.cleanup_required = False


def process_is_running(handle: ProcessHandle) -> bool:
    return handle.process is not None and handle.process.poll() is None


def append_note_once(handle: ProcessHandle, note: str) -> None:
    if note not in handle.notes:
        handle.notes.append(note)


def remote_group_is_running(handle: ProcessHandle) -> bool | None:
    if not handle.endpoint.remote or handle.process is None:
        return False
    if not handle.control_token:
        append_note_once(handle, "remote process-group state unavailable: control token is unavailable")
        return None
    state, identity, detail = remote_control_state(
        handle.endpoint,
        action=handle.control_action,
        experiment_id=handle.experiment_id,
        token=handle.control_token,
        start_path=handle.control_start_path,
        completion_path=handle.control_completion_path,
    )
    if handle.process.poll() is not None:
        handle.transport_return_code = handle.process.returncode
    if (
        state is False
        and identity is not None
        and isinstance(identity.get("return_code"), int)
    ):
        handle.remote_completion_return_code = identity["return_code"]
    if state is None:
        append_note_once(
            handle,
            f"remote process-group state unavailable: {detail}",
        )
    return state


def shutdown_target_is_running(handle: ProcessHandle) -> bool:
    if not handle.endpoint.remote:
        return process_is_running(handle)
    remote_state = remote_group_is_running(handle)
    return True if remote_state is None else remote_state or process_is_running(handle)


def signal_handle(handle: ProcessHandle, signal_name: str) -> None:
    if handle.process is None:
        return
    if handle.endpoint.remote:
        if not handle.control_token:
            state: bool | None = None
            identity = None
            detail = "control token is unavailable"
        else:
            state, identity, detail = remote_control_state(
                handle.endpoint,
                action=handle.control_action,
                experiment_id=handle.experiment_id,
                token=handle.control_token,
                start_path=handle.control_start_path,
                completion_path=handle.control_completion_path,
            )
        if state is None or (state is True and identity is None):
            append_note_once(
                handle,
                f"remote {signal_name} refused: {detail}",
            )
            if signal_name == "KILL" and process_is_running(handle):
                try:
                    os.killpg(handle.process.pid, signal.SIGKILL)
                    append_note_once(
                        handle,
                        "killed local SSH transport after unverifiable remote KILL",
                    )
                except OSError as error:
                    append_note_once(handle, f"local SSH control KILL failed: {error}")
            return
        if state is False:
            if process_is_running(handle):
                try:
                    os.killpg(
                        handle.process.pid,
                        getattr(signal, f"SIG{signal_name}"),
                    )
                except OSError as error:
                    append_note_once(
                        handle,
                        f"local SSH control {signal_name} failed: {error}",
                    )
            return
        assert identity is not None
        command = remote_identity_signal_command(
            handle.control_start_path,
            identity,
            signal_name,
            handle.endpoint.sudo,
        )
        try:
            result = remote_run(handle.endpoint, command, check=False, capture=True)
        except (OSError, subprocess.SubprocessError) as error:
            append_note_once(handle, f"remote {signal_name} failed: {error}")
            result = None
        if result is not None and result.returncode != 0:
            append_note_once(handle, f"remote {signal_name} failed: {result.stderr.strip()}")
        if signal_name == "KILL" and process_is_running(handle):
            try:
                os.killpg(handle.process.pid, signal.SIGKILL)
                append_note_once(handle, "killed local SSH transport after remote KILL")
            except OSError as error:
                append_note_once(handle, f"local SSH control KILL failed: {error}")
        return
    if not process_is_running(handle):
        return
    try:
        if handle.endpoint.sudo:
            subprocess.run(["sudo", "-n", "kill", f"-{signal_name}", "--", f"-{handle.process.pid}"], check=True)
        else:
            os.killpg(handle.process.pid, getattr(signal, f"SIG{signal_name}"))
    except (OSError, subprocess.CalledProcessError) as error:
        handle.notes.append(f"local {signal_name} failed: {error}")


def wait_handles(handles: list[ProcessHandle], timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if all(not shutdown_target_is_running(handle) for handle in handles):
            return True
        time.sleep(0.25)
    return all(not shutdown_target_is_running(handle) for handle in handles)


def role_shutdown_is_verified(handle: ProcessHandle) -> bool:
    if handle.process is None:
        return True
    if handle.process.poll() is None:
        return False
    if handle.endpoint.remote:
        return remote_group_is_running(handle) is False
    return True


def stop_handles(handles: list[ProcessHandle], reason: str, grace_s: float) -> bool:
    failures: list[BaseException] = []
    for handle in reversed(handles):
        try:
            if shutdown_target_is_running(handle):
                handle.stop_reason = reason
                signal_handle(handle, "INT")
        except BaseException as error:
            append_note_once(handle, f"INT shutdown stage failed: {error}")
            failures.append(error)
    try:
        stopped_after_int = wait_handles(handles, grace_s)
    except BaseException as error:
        failures.append(error)
        stopped_after_int = False
    if not stopped_after_int:
        for handle in reversed(handles):
            try:
                signal_handle(handle, "TERM")
            except BaseException as error:
                append_note_once(handle, f"TERM shutdown stage failed: {error}")
                failures.append(error)
    try:
        stopped_after_term = wait_handles(handles, grace_s)
    except BaseException as error:
        failures.append(error)
        stopped_after_term = False
    if not stopped_after_term:
        for handle in reversed(handles):
            try:
                signal_handle(handle, "KILL")
            except BaseException as error:
                append_note_once(handle, f"KILL shutdown stage failed: {error}")
                failures.append(error)
    try:
        stopped = wait_handles(handles, grace_s)
    except BaseException as error:
        failures.append(error)
        stopped = False
    for handle in handles:
        try:
            handle.shutdown_verified = stopped and role_shutdown_is_verified(handle)
        except BaseException as error:
            handle.shutdown_verified = False
            append_note_once(handle, f"shutdown verification failed: {error}")
            failures.append(error)
    if failures:
        raise failures[0]
    return all(handle.shutdown_verified for handle in handles)


def close_handle(handle: ProcessHandle) -> None:
    if handle.process is None:
        return
    handle.shutdown_verified = role_shutdown_is_verified(handle)
    if not handle.shutdown_verified:
        handle.notes.append("process remained alive after shutdown escalation")
        handle.run_status = "shutdown_failed"
    else:
        handle.process.wait()
        if handle.endpoint.remote:
            handle.transport_return_code = handle.process.returncode
            if handle.remote_completion_return_code == 0:
                handle.run_status = "finished"
            elif handle.remote_completion_return_code is None:
                handle.run_status = "remote_return_unverified"
            else:
                handle.run_status = "exited_nonzero"
        else:
            handle.run_status = (
                "finished" if handle.process.returncode == 0 else "exited_nonzero"
            )
    handle.end_realtime_ns = time.time_ns()
    handle.end_monotonic_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    if handle.stdout_handle is not None:
        handle.stdout_handle.close()
    if handle.stderr_handle is not None:
        handle.stderr_handle.close()


def run_manifest(
    endpoint: Endpoint,
    plan: ExperimentPlan,
    run_dir: str,
    command: list[str],
    environment: dict[str, str],
    workload_configured: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "campaign_id": plan.campaign_id,
        "experiment_id": plan.experiment_id,
        "case": plan.case_name,
        "variant": plan.variant.name,
        "trial": plan.trial,
        "role": endpoint.role,
        "host": endpoint.host,
        "hostname": endpoint.hostname,
        "run_dir": run_dir,
        "profile_enabled": plan.variant.profile,
        "pmu_mode": plan.variant.pmu,
        "sidecar": plan.variant.sidecar,
        "command": redact_command(command),
        "environment": redact_environment(environment),
        "orchestrator_hostname": socket.gethostname(),
        "workload_status": "planned" if workload_configured else "not_configured",
        "workload_artifact": "",
        "network_cleanup_status": "pending" if workload_configured else "not_configured",
        "workload_control_results": {},
        "transport_return_code": "",
        "remote_completion_return_code": "",
        "status": "prepared",
    }


def authoritative_return_code(handle: ProcessHandle) -> int | str:
    if handle.process is None:
        return ""
    if handle.endpoint.remote:
        return (
            handle.remote_completion_return_code
            if handle.remote_completion_return_code is not None
            else ""
        )
    return handle.process.returncode if handle.process.returncode is not None else ""


def update_manifest(handle: ProcessHandle) -> dict[str, Any]:
    value = dict(handle.initial_manifest)
    return_code = authoritative_return_code(handle)
    duration_s, duration_clock, duration_status = elapsed_duration(
        handle.start_monotonic_ns,
        handle.end_monotonic_ns,
        handle.start_realtime_ns,
        handle.end_realtime_ns,
    )
    value.update(
        {
            "status": handle.run_status,
            "return_code": return_code,
            "transport_return_code": (
                handle.transport_return_code
                if handle.transport_return_code is not None
                else ""
            ),
            "remote_completion_return_code": (
                handle.remote_completion_return_code
                if handle.remote_completion_return_code is not None
                else ""
            ),
            "stop_reason": handle.stop_reason or ("process_exit" if handle.process is not None else handle.run_status),
            "start_realtime_ns": handle.start_realtime_ns,
            "end_realtime_ns": handle.end_realtime_ns,
            "start_monotonic_raw_ns": handle.start_monotonic_ns,
            "end_monotonic_raw_ns": handle.end_monotonic_ns,
            "duration_s": duration_s,
            "duration_clock": duration_clock,
            "duration_status": duration_status,
            "realtime_clock_regressed": int(
                handle.start_realtime_ns > 0 and handle.end_realtime_ns < handle.start_realtime_ns
            ),
            "anchor_clock_scope": "orchestrator" if handle.endpoint.remote else "measured_host",
            "launch_index": handle.launch_index,
            "sidecar_tool": handle.sidecar_tool,
            "sidecar_artifact": handle.sidecar_artifact,
            "sidecar_tool_version": handle.sidecar_version,
            "sidecar_status": handle.sidecar_status,
            "workload_status": handle.workload_status,
            "workload_artifact": handle.workload_artifact,
            "network_cleanup_status": handle.network_cleanup_status,
            "workload_control_results": handle.workload_control_results,
            "archive_status": "finalization_pending",
            "notes": list(handle.notes),
        }
    )
    path = f"{handle.run_dir}/campaign_run.json"
    if handle.endpoint.remote:
        write_remote_json(handle.endpoint, path, value)
    else:
        atomic_write_json(Path(path), value)
    return value


def normalize_ownership(handle: ProcessHandle) -> None:
    if not handle.endpoint.sudo:
        return
    if handle.owner_uid < 0 or handle.owner_gid < 0:
        raise RuntimeError(f"numeric owner is unavailable for role {handle.endpoint.role}")
    owner = f"{handle.owner_uid}:{handle.owner_gid}"
    command = ["sudo", "-n", "chown", "-R", "--", owner, handle.run_dir]
    if handle.endpoint.remote:
        result = remote_run(handle.endpoint, shlex.join(command), check=False, capture=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
    else:
        subprocess.run(command, check=True, text=True, capture_output=True)


def remote_file_exists(endpoint: Endpoint, path: str) -> bool:
    return remote_run(endpoint, f"test -f {shlex.quote(path)}", check=False).returncode == 0


def sidecar_clock(tool: str) -> tuple[str, str, str, str]:
    if tool == "perf_record":
        return "CLOCK_MONOTONIC_RAW", "ns", "shared_monotonic_raw", ""
    if tool == "perf_stat":
        return "process_lifetime", "mixed", "aggregate_run_interval", ""
    if tool == "perf_sched":
        return "perf_event_clock", "ns", "alignment_pending", "convert timestamps offline before causal joins"
    raise ValueError(f"unsupported sidecar tool: {tool}")


def sidecar_source_anchors(handle: ProcessHandle) -> tuple[int | str, int | str, int | str, int | str, str]:
    if handle.endpoint.remote:
        return (
            "",
            "",
            "",
            "",
            "source-host process bounds unavailable; campaign_run.json launch bounds use the orchestrator clock",
        )
    return (
        handle.start_realtime_ns,
        handle.end_realtime_ns,
        handle.start_monotonic_ns,
        handle.end_monotonic_ns,
        "",
    )


def register_sidecar(handle: ProcessHandle, plan: ExperimentPlan) -> None:
    if handle.sidecar_tool == "none":
        return
    if handle.process is None:
        handle.sidecar_status = "not_launched"
        return
    if process_is_running(handle):
        handle.sidecar_status = "skipped_process_alive"
        return
    exists = (
        remote_file_exists(handle.endpoint, handle.sidecar_artifact)
        if handle.endpoint.remote
        else Path(handle.sidecar_artifact).is_file()
    )
    if not exists:
        handle.sidecar_status = "artifact_missing"
        handle.notes.append(f"missing sidecar artifact: {handle.sidecar_artifact}")
        return
    source_id = require_safe_name(f"{handle.sidecar_tool}-{handle.endpoint.role}-t{plan.trial:03d}", "source_id")
    clock_domain, clock_unit, alignment, source_notes = sidecar_clock(handle.sidecar_tool)
    recorded_command = shlex.join(redact_command(handle.command))
    start_realtime_ns, end_realtime_ns, start_monotonic_ns, end_monotonic_ns, anchor_notes = (
        sidecar_source_anchors(handle)
    )
    source_notes = "; ".join(note for note in (source_notes, anchor_notes) if note)
    if handle.endpoint.remote:
        assert handle.endpoint.archive_tool is not None
        command = [
            handle.endpoint.archive_tool,
            "register-external",
            handle.run_dir,
            "--source-id",
            source_id,
            "--source-type",
            handle.sidecar_tool,
            "--artifact",
            handle.sidecar_artifact,
            "--clock-domain",
            clock_domain,
            "--clock-unit",
            clock_unit,
            "--command",
            recorded_command,
            "--tool-version",
            handle.sidecar_version,
            "--start-realtime-ns",
            str(start_realtime_ns),
            "--end-realtime-ns",
            str(end_realtime_ns),
            "--start-monotonic-raw-ns",
            str(start_monotonic_ns),
            "--end-monotonic-raw-ns",
            str(end_monotonic_ns),
            "--status",
            "recorded",
            "--alignment-method",
            alignment,
            "--notes",
            source_notes,
        ]
        result = remote_run(handle.endpoint, shlex.join(command), check=False, capture=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
    else:
        register_external_source(
            argparse.Namespace(
                run_dir=Path(handle.run_dir),
                source_id=source_id,
                source_type=handle.sidecar_tool,
                artifact=Path(handle.sidecar_artifact),
                copy_artifact=False,
                clock_domain=clock_domain,
                clock_unit=clock_unit,
                command=recorded_command,
                tool_version=handle.sidecar_version,
                start_realtime_ns=start_realtime_ns,
                end_realtime_ns=end_realtime_ns,
                start_monotonic_raw_ns=start_monotonic_ns,
                end_monotonic_raw_ns=end_monotonic_ns,
                status="recorded",
                alignment_method=alignment,
                alignment_uncertainty_ns="",
                notes=source_notes,
                replace_manifest=False,
            )
        )
    handle.sidecar_status = "registered"


def workload_source_anchors(
    state: dict[str, Any],
) -> tuple[int | str, int | str, int | str, int | str]:
    values: list[int | str] = []
    for field_name in (
        "start_realtime_ns",
        "end_realtime_ns",
        "start_monotonic_raw_ns",
        "end_monotonic_raw_ns",
    ):
        value = state.get(field_name)
        values.append(value if isinstance(value, int) and value > 0 else "")
    return values[0], values[1], values[2], values[3]


def register_workload_source(
    handle: ProcessHandle,
    plan: ExperimentPlan,
    workload: WorkloadHandle,
    state: dict[str, Any],
) -> None:
    artifact = f"{handle.run_dir}/workload/workload_run.json"
    source_id = require_safe_name(
        f"campaign-workload-t{plan.trial:03d}",
        "source_id",
    )
    source_status = "recorded" if workload.status == "completed" else "failed"
    alignment = (
        "shared_monotonic_raw"
        if handle.endpoint.role == workload.spec.client_role
        else "alignment_pending"
    )
    notes = (
        f"source host is {workload.endpoint.hostname}; "
        "cross-host temporal alignment is unresolved"
        if alignment == "alignment_pending"
        else f"source host is {workload.endpoint.hostname}"
    )
    anchors = workload_source_anchors(state)
    recorded_command = shlex.join(redact_command(workload_action_command(workload, "run")))
    if handle.endpoint.remote:
        assert handle.endpoint.archive_tool is not None
        command = [
            handle.endpoint.archive_tool,
            "register-external",
            handle.run_dir,
            "--source-id",
            source_id,
            "--source-type",
            "campaign_udp_bidir_workload",
            "--artifact",
            artifact,
            "--clock-domain",
            f"{workload.endpoint.hostname}:CLOCK_MONOTONIC_RAW",
            "--clock-unit",
            "ns",
            "--command",
            recorded_command,
            "--tool-version",
            f"workload-schema-{state.get('schema_version', '')}",
            "--start-realtime-ns",
            str(anchors[0]),
            "--end-realtime-ns",
            str(anchors[1]),
            "--start-monotonic-raw-ns",
            str(anchors[2]),
            "--end-monotonic-raw-ns",
            str(anchors[3]),
            "--status",
            source_status,
            "--alignment-method",
            alignment,
            "--notes",
            notes,
        ]
        result = remote_run(
            handle.endpoint,
            shlex.join(command),
            check=False,
            capture=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
    else:
        register_external_source(
            argparse.Namespace(
                run_dir=Path(handle.run_dir),
                source_id=source_id,
                source_type="campaign_udp_bidir_workload",
                artifact=Path(artifact),
                copy_artifact=False,
                clock_domain=f"{workload.endpoint.hostname}:CLOCK_MONOTONIC_RAW",
                clock_unit="ns",
                command=recorded_command,
                tool_version=f"workload-schema-{state.get('schema_version', '')}",
                start_realtime_ns=anchors[0],
                end_realtime_ns=anchors[1],
                start_monotonic_raw_ns=anchors[2],
                end_monotonic_raw_ns=anchors[3],
                status=source_status,
                alignment_method=alignment,
                alignment_uncertainty_ns="",
                notes=notes,
                replace_manifest=False,
            )
        )
    handle.workload_artifact = "workload/workload_run.json"


def register_workload_evidence(
    handles: list[ProcessHandle],
    plan: ExperimentPlan,
    workload: WorkloadHandle,
) -> None:
    source_path = f"{workload.run_dir}/workload/workload_run.json"
    source_text = read_endpoint_text(workload.endpoint, source_path)
    state = json.loads(source_text)
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != 1
        or state.get("experiment_id") != workload.experiment_id
    ):
        raise RuntimeError("workload summary is malformed or has the wrong experiment identity")
    workload.state = state
    for handle in handles:
        if not handle.prepared or not handle.shutdown_verified:
            continue
        target_path = f"{handle.run_dir}/workload/workload_run.json"
        if not (
            handle.endpoint.role == workload.spec.client_role
            and handle.run_dir == workload.run_dir
        ):
            write_endpoint_text(handle.endpoint, target_path, source_text)
        try:
            register_workload_source(handle, plan, workload, state)
        except Exception as error:
            handle.workload_artifact = ""
            handle.notes.append(f"workload registration failed: {error}")


def propagate_workload_state(
    handles: list[ProcessHandle],
    workload: WorkloadHandle,
) -> None:
    for handle in handles:
        handle.workload_status = workload.status
        handle.network_cleanup_status = workload.cleanup_status
        handle.workload_control_results = {
            action: dict(result)
            for action, result in workload.control_results.items()
        }


def finalize_handle(handle: ProcessHandle) -> None:
    if not handle.prepared:
        handle.archive_status = "not_prepared"
        return
    if not handle.shutdown_verified:
        handle.archive_status = "skipped_shutdown_unverified"
        return
    if handle.endpoint.remote:
        assert handle.endpoint.archive_tool is not None
        result = remote_run(
            handle.endpoint,
            shlex.join([handle.endpoint.archive_tool, "finalize", handle.run_dir]),
            check=False,
            capture=True,
            timeout_s=None,
        )
        if result.returncode != 0:
            handle.archive_status = "finalize_failed"
            handle.notes.append(result.stderr.strip())
            return
    else:
        finalize_archive(Path(handle.run_dir))
    handle.archive_status = "finalized"


def result_row(handle: ProcessHandle, plan: ExperimentPlan) -> dict[str, Any]:
    return_code = authoritative_return_code(handle)
    duration_s, duration_clock, duration_status = elapsed_duration(
        handle.start_monotonic_ns,
        handle.end_monotonic_ns,
        handle.start_realtime_ns,
        handle.end_realtime_ns,
    )
    return {
        "campaign_id": plan.campaign_id,
        "experiment_id": plan.experiment_id,
        "case": plan.case_name,
        "variant": plan.variant.name,
        "trial": plan.trial,
        "role": handle.endpoint.role,
        "host": handle.endpoint.host,
        "run_dir": handle.run_dir,
        "profile_enabled": int(plan.variant.profile),
        "pmu_mode": plan.variant.pmu,
        "sidecar": handle.sidecar_tool,
        "start_realtime_ns": handle.start_realtime_ns,
        "end_realtime_ns": handle.end_realtime_ns,
        "start_monotonic_raw_ns": handle.start_monotonic_ns,
        "end_monotonic_raw_ns": handle.end_monotonic_ns,
        "duration_s": duration_s,
        "duration_clock": duration_clock,
        "duration_status": duration_status,
        "realtime_clock_regressed": int(
            handle.start_realtime_ns > 0 and handle.end_realtime_ns < handle.start_realtime_ns
        ),
        "transport_return_code": (
            handle.transport_return_code
            if handle.transport_return_code is not None
            else ""
        ),
        "remote_completion_return_code": (
            handle.remote_completion_return_code
            if handle.remote_completion_return_code is not None
            else ""
        ),
        "return_code": return_code,
        "run_status": handle.run_status,
        "stop_reason": handle.stop_reason or ("process_exit" if handle.process is not None else handle.run_status),
        "archive_status": handle.archive_status,
        "sidecar_status": handle.sidecar_status,
        "workload_status": handle.workload_status,
        "workload_artifact": handle.workload_artifact,
        "network_cleanup_status": handle.network_cleanup_status,
        "notes": "; ".join(note for note in handle.notes if note),
    }


def append_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    if not new_file:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames
            existing_rows = list(reader)
        fieldname_tuple = tuple(fieldnames or [])
        if fieldname_tuple in {
            tuple(LEGACY_RESULT_FIELDS),
            tuple(PRE_IDENTITY_RESULT_FIELDS),
        }:
            if any(None in row for row in existing_rows):
                raise ValueError(f"legacy campaign results contain extra columns: {path}")
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            try:
                with temporary.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
                    writer.writeheader()
                    for row in existing_rows:
                        if fieldnames == LEGACY_RESULT_FIELDS:
                            row["workload_status"] = "not_configured"
                            row["workload_artifact"] = ""
                            row["network_cleanup_status"] = "not_configured"
                        row["transport_return_code"] = ""
                        row["remote_completion_return_code"] = ""
                        writer.writerow(row)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        elif fieldnames != RESULT_FIELDS:
            raise ValueError(f"unexpected campaign results schema: {path}")
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def finalize_runs(
    handles: list[ProcessHandle],
    plan: ExperimentPlan,
    control_root: Path,
    workload: WorkloadHandle | None = None,
) -> list[dict[str, Any]]:
    prepared = [handle for handle in handles if handle.prepared]
    if workload is not None and (
        not workload.shutdown_verified or not workload.evidence_quiesced
    ):
        if not workload.shutdown_verified:
            archive_status = "skipped_workload_shutdown_unverified"
            note = "archive finalization skipped because workload shutdown is unverified"
        else:
            archive_status = "skipped_workload_action_unverified"
            note = "archive finalization skipped because a workload action may still be writing"
        for handle in prepared:
            handle.workload_artifact = ""
            handle.archive_status = archive_status
            append_note_once(handle, note)
        rows = [result_row(handle, plan) for handle in prepared]
        if rows:
            append_results(control_root / "campaign_results.csv", rows)
        return rows
    if workload is not None and workload.cleanup_required:
        for handle in prepared:
            handle.workload_artifact = ""
            handle.archive_status = "skipped_workload_cleanup_unverified"
            append_note_once(
                handle,
                "archive finalization skipped because workload cleanup is unverified",
            )
        rows = [result_row(handle, plan) for handle in prepared]
        if rows:
            append_results(control_root / "campaign_results.csv", rows)
        return rows
    for handle in prepared:
        if not handle.shutdown_verified:
            handle.sidecar_status = "skipped_process_alive" if handle.sidecar_tool != "none" else "not_requested"
        else:
            try:
                normalize_ownership(handle)
            except Exception as error:
                handle.notes.append(f"ownership normalization failed: {error}")
    if workload is not None:
        try:
            register_workload_evidence(prepared, plan, workload)
        except Exception as error:
            for handle in prepared:
                handle.workload_artifact = ""
                handle.notes.append(f"workload evidence unavailable: {error}")
    for handle in prepared:
        if handle.shutdown_verified:
            try:
                register_sidecar(handle, plan)
            except Exception as error:
                handle.sidecar_status = "registration_failed"
                handle.notes.append(f"sidecar registration failed: {error}")
        handle.archive_status = "finalization_pending"
        try:
            update_manifest(handle)
        except Exception as error:
            handle.archive_status = "manifest_update_failed"
            handle.notes.append(f"campaign manifest update failed: {error}")
            continue
        try:
            finalize_handle(handle)
        except Exception as error:
            handle.archive_status = "finalize_failed"
            handle.notes.append(f"archive finalization failed: {error}")
    rows = [result_row(handle, plan) for handle in prepared]
    if rows:
        append_results(control_root / "campaign_results.csv", rows)
    return rows


def wait_launch_delay(delay_s: float, handles: list[ProcessHandle], stop_on_role_exit: bool) -> None:
    deadline = time.monotonic() + delay_s
    while time.monotonic() < deadline:
        if stop_on_role_exit and any(
            handle.process is not None and handle.process.poll() is not None for handle in handles
        ):
            raise RuntimeError("a previously launched role exited during launch delay")
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))


def execute_plan(spec: dict[str, Any], plan: ExperimentPlan, control_root: Path) -> list[dict[str, Any]]:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    endpoints: dict[str, Endpoint] = spec["_endpoints"]
    start_order: list[str] = spec["_start_order"]
    workload_spec: WorkloadSpec | None = spec["_workload"]
    workload_configured = workload_spec is not None
    handles: list[ProcessHandle] = []
    sidecar_tool = str(plan.variant.sidecar.get("tool", "none"))
    for launch_index, role in enumerate(start_order):
        endpoint = endpoints[role]
        run_name = f"{timestamp}_{endpoint.role}_{endpoint.hostname}"
        run_dir = f"{endpoint.profile_root}/{run_name}"
        environment = role_environment(endpoint, plan, run_dir)
        command = role_command(endpoint, plan)
        if not plan.variant.profile and "-P" in command:
            raise ValueError(f"disabled variant cannot use -P in role {endpoint.role} command")
        effective, artifact = effective_command(endpoint, command, environment, plan.variant.sidecar, run_dir)
        manifest = run_manifest(
            endpoint,
            plan,
            run_dir,
            effective,
            environment,
            workload_configured=workload_configured,
        )
        handles.append(
            ProcessHandle(
                endpoint=endpoint,
                run_dir=run_dir,
                command=effective,
                environment=environment,
                initial_manifest=manifest,
                sidecar_tool=sidecar_tool,
                sidecar_artifact=artifact,
                sidecar_status="planned" if sidecar_tool != "none" else "not_requested",
                workload_status="planned" if workload_configured else "not_configured",
                network_cleanup_status="pending"
                if workload_configured
                else "not_configured",
                experiment_id=plan.experiment_id,
                launch_index=launch_index,
            )
        )

    workload: WorkloadHandle | None = None
    if workload_spec is not None:
        client = next(
            handle
            for handle in handles
            if handle.endpoint.role == workload_spec.client_role
        )
        workload = WorkloadHandle(
            spec=workload_spec,
            endpoint=client.endpoint,
            run_dir=client.run_dir,
            experiment_id=plan.experiment_id,
        )

    duration_s = float(spec.get("duration_s", 60.0))
    grace_s = float(spec.get("stop_grace_s", 10.0))
    stop_on_role_exit = bool(spec.get("stop_on_role_exit", True))
    control_dir = control_root / plan.experiment_id
    phase = "preflight"
    active_handle: ProcessHandle | None = None
    stop_reason = "duration_elapsed"
    failure: BaseException | None = None
    try:
        for handle in handles:
            active_handle = handle
            preflight_endpoint(handle)

        phase = "preparation"
        for handle in handles:
            active_handle = handle
            prepare_run_dir(handle)
            handle.run_status = "prepared"

        if workload is not None:
            active_handle = None
            phase = "workload_preflight"
            preflight_workload(workload)
            propagate_workload_state(handles, workload)

        phase = "launch"
        launched: list[ProcessHandle] = []
        for handle in handles:
            active_handle = handle
            wait_launch_delay(handle.endpoint.launch_delay_s, launched, stop_on_role_exit)
            if stop_on_role_exit and any(
                peer.process is not None and peer.process.poll() is not None for peer in launched
            ):
                raise RuntimeError("a previously launched role exited before its peer was started")
            if handle.endpoint.remote:
                launch_remote(handle, control_dir)
            else:
                launch_local(handle)
            if handle.sidecar_tool != "none":
                handle.sidecar_status = "launched"
            launched.append(handle)

        active_handle = None
        if workload is not None:
            phase = "workload_launch"
            launch_workload(workload, control_dir)
            propagate_workload_state(handles, workload)

        phase = "measurement"
        if workload is not None:
            stop_reason = monitor_loaded_measurement(handles, workload)
            propagate_workload_state(handles, workload)
        else:
            deadline = time.monotonic() + duration_s
            while time.monotonic() < deadline:
                states = [role_measurement_state(handle) for handle in handles]
                if "unknown" in states:
                    stop_reason = "paired_role_state_unknown"
                    break
                if all(state == "ended" for state in states):
                    stop_reason = "all_processes_exited"
                    break
                if "ended" in states and stop_on_role_exit:
                    stop_reason = "paired_role_exited"
                    break
                time.sleep(0.25)
    except BaseException as error:
        failure = error
        stop_reason = "interrupted" if isinstance(error, KeyboardInterrupt) else f"{phase}_failed"
        if workload is not None and workload.status not in {"completed", "preflight_failed"}:
            workload.status = "aborted"
            workload.notes.append(str(error))
        if active_handle is not None and active_handle.prepared and active_handle.process is None:
            active_handle.run_status = f"{phase}_failed"
            active_handle.notes.append(str(error))
        for handle in handles:
            if handle.prepared and handle.process is None and handle.run_status == "prepared":
                handle.run_status = "launch_not_attempted" if phase == "launch" else f"{phase}_aborted"
                handle.stop_reason = stop_reason
    finally:
        if workload is not None:
            try:
                if workload.process is not None and not workload_shutdown_is_verified(
                    workload
                ):
                    stop_workload(workload, grace_s)
                close_workload(workload)
            except BaseException as error:
                workload.shutdown_verified = False
                workload.evidence_quiesced = False
                workload.status = "failed"
                workload.notes.append(f"workload shutdown failed: {error}")
                if failure is None:
                    failure = error
        launched = [handle for handle in handles if handle.process is not None]
        if launched:
            try:
                stop_handles(launched, stop_reason, grace_s)
            except BaseException as error:
                for handle in launched:
                    append_note_once(handle, f"paired role shutdown failed: {error}")
                if failure is None:
                    failure = error
            for handle in reversed(launched):
                try:
                    close_handle(handle)
                except BaseException as error:
                    handle.shutdown_verified = False
                    handle.run_status = "shutdown_failed"
                    append_note_once(handle, f"role close failed: {error}")
                    if failure is None:
                        failure = error
        for handle in handles:
            try:
                handle.shutdown_verified = role_shutdown_is_verified(handle)
            except BaseException as error:
                handle.shutdown_verified = False
                append_note_once(
                    handle,
                    f"independent role shutdown verification failed: {error}",
                )
                if failure is None:
                    failure = error
            if not handle.shutdown_verified:
                append_note_once(handle, "independent role shutdown verification failed")
        if workload is not None:
            try:
                workload.shutdown_verified = workload_shutdown_is_verified(workload)
            except BaseException as error:
                workload.shutdown_verified = False
                workload.evidence_quiesced = False
                workload.notes.append(f"workload shutdown verification failed: {error}")
                if failure is None:
                    failure = error
            if workload.cleanup_required:
                if workload.shutdown_verified and all(
                    handle.shutdown_verified for handle in handles
                ):
                    try:
                        cleanup_workload(workload)
                    except BaseException as error:
                        workload.cleanup_status = "failed"
                        workload.notes.append(str(error))
                        if failure is None:
                            failure = error
                else:
                    workload.cleanup_status = "skipped_unverified_role_death"
                    workload.notes.append(
                        "network cleanup skipped because helper/role death was not proven"
                    )
            elif workload.cleanup_status == "pending":
                workload.cleanup_status = "not_started"
            propagate_workload_state(handles, workload)
        rows = finalize_runs(handles, plan, control_root, workload)

    if failure is not None:
        raise failure
    return rows


def plan_view(spec: dict[str, Any], plan: ExperimentPlan) -> dict[str, Any]:
    roles = {}
    for launch_index, role in enumerate(spec["_start_order"]):
        endpoint = spec["_endpoints"][role]
        run_dir = f"{endpoint.profile_root}/<timestamp>_{endpoint.role}_{endpoint.hostname}"
        environment = role_environment(endpoint, plan, run_dir)
        command = role_command(endpoint, plan)
        effective, artifact = effective_command(endpoint, command, environment, plan.variant.sidecar, run_dir)
        roles[endpoint.role] = {
            "host": endpoint.host,
            "launch_index": launch_index,
            "launch_delay_s": endpoint.launch_delay_s,
            "profile_enabled": plan.variant.profile,
            "pmu_mode": plan.variant.pmu,
            "sidecar": plan.variant.sidecar.get("tool", "none"),
            "sidecar_artifact": artifact,
            "command": redact_command(effective),
            "environment": redact_environment(environment),
        }
    workload_view: dict[str, Any] | None = None
    workload_spec: WorkloadSpec | None = spec["_workload"]
    if workload_spec is not None:
        endpoint = spec["_endpoints"][workload_spec.client_role]
        run_dir = (
            f"{endpoint.profile_root}/<timestamp>_{endpoint.role}_{endpoint.hostname}"
        )
        workload = WorkloadHandle(
            spec=workload_spec,
            endpoint=endpoint,
            run_dir=run_dir,
            experiment_id=plan.experiment_id,
        )
        workload_view = {
            "client_role": workload_spec.client_role,
            "host": endpoint.host,
            "contract": workload_command_plan(workload_spec),
            "actions": {
                action: redact_command(workload_action_command(workload, action))
                for action in ("preflight", "run", "cleanup")
            },
            "role_count_unchanged": True,
        }
    return {
        "campaign_id": plan.campaign_id,
        "experiment_id": plan.experiment_id,
        "case": plan.case_name,
        "variant": plan.variant.name,
        "trial": plan.trial,
        "start_order": list(spec["_start_order"]),
        "roles": roles,
        "workload": workload_view,
    }


def integer_filter(values: list[str]) -> set[int]:
    parsed = {int(value) for value in values}
    if any(value < 1 for value in parsed):
        raise ValueError("trial filters must be positive")
    return parsed


def campaign_result_workload_succeeded(row: dict[str, Any]) -> bool:
    workload_status = str(row.get("workload_status", "not_configured"))
    workload_artifact = str(row.get("workload_artifact", ""))
    cleanup_status = str(
        row.get("network_cleanup_status", "not_configured")
    )
    stop_reason = str(row.get("stop_reason", ""))
    if workload_status == "not_configured":
        return (
            not workload_artifact
            and cleanup_status == "not_configured"
            and stop_reason == "duration_elapsed"
        )
    return (
        workload_status == "completed"
        and bool(workload_artifact)
        and cleanup_status in {"ok", "already_absent"}
        and stop_reason == "measurement_complete"
    )


def experiment_failed(rows: list[dict[str, Any]], expected_roles: set[str]) -> bool:
    observed_roles = [str(row.get("role", "")) for row in rows]
    role_set_invalid = len(observed_roles) != len(expected_roles) or set(observed_roles) != expected_roles
    workload_values = {
        (
            str(row.get("workload_status", "not_configured")),
            str(row.get("workload_artifact", "")),
            str(row.get("network_cleanup_status", "not_configured")),
        )
        for row in rows
    }
    workload_mismatch = len(workload_values) > 1
    return role_set_invalid or workload_mismatch or any(
        row["archive_status"] != "finalized"
        or row["run_status"] != "finished"
        or row["return_code"] != 0
        or not campaign_result_workload_succeeded(row)
        or (row["sidecar"] != "none" and row["sidecar_status"] != "registered")
        for row in rows
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--execute", action="store_true", help="Execute the campaign; otherwise print a read-only plan")
    parser.add_argument("--case", action="append", default=[], help="Select a case; repeat as needed")
    parser.add_argument("--variant", action="append", default=[], help="Select a variant; repeat as needed")
    parser.add_argument("--trial", action="append", default=[], help="Select a 1-based trial; repeat as needed")
    parser.add_argument("--control-root", type=Path, help="Local campaign status directory")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed experiment")
    args = parser.parse_args()

    spec = load_spec(args.spec)
    plans = build_plans(spec, set(args.case), set(args.variant), integer_filter(args.trial))
    if not args.execute:
        for plan in plans:
            print(json.dumps(plan_view(spec, plan), sort_keys=True))
        print(f"planned_experiments={len(plans)}", file=sys.stderr)
        return 0

    control_root = args.control_root or (
        Path(spec.get("control_root", "PerformanceProfiles/Campaigns")) / spec["campaign_id"]
    )
    control_root = control_root.expanduser().resolve()
    control_root.mkdir(parents=True, exist_ok=True)
    failures = 0
    expected_roles = set(spec["_start_order"])
    for plan in plans:
        try:
            rows = execute_plan(spec, plan, control_root)
            failures += experiment_failed(rows, expected_roles)
        except Exception as error:
            failures += 1
            print(f"experiment failed: {plan.experiment_id}: {error}", file=sys.stderr)
            if not args.keep_going:
                break
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
