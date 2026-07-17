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
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))

from oai_profile_archive import finalize_archive, register_external_source  # noqa: E402


CAMPAIGN_SCHEMA_VERSION = 1
RUNNER_VERSION = "1"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
REMOTE_CONTROL_TIMEOUT_S = 20.0
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
    "return_code",
    "run_status",
    "stop_reason",
    "archive_status",
    "sidecar_status",
    "notes",
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
    run_status: str = "planned"
    initial_manifest: dict[str, Any] = field(default_factory=dict)
    prepared: bool = False
    owner_uid: int = -1
    owner_gid: int = -1
    launch_index: int = -1
    notes: list[str] = field(default_factory=list)


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
        remote_run(
            endpoint,
            f"mkdir -p {shlex.quote(parent)} && "
            f"mkdir {shlex.quote(run_dir)} && "
            f"mkdir {shlex.quote(run_dir + '/sidecars')}",
        )
        handle.prepared = True
        write_remote_json(endpoint, f"{run_dir}/campaign_run.json", handle.initial_manifest)
    else:
        Path(run_dir).mkdir(parents=True, exist_ok=False)
        handle.prepared = True
        Path(run_dir, "sidecars").mkdir()
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
    control_pid = f"{handle.run_dir}/control.pid"
    working_directory = handle.endpoint.cwd or "."
    inner = f"printf '%s\\n' $$ > {shlex.quote(control_pid)}; exec {shlex.join(handle.command)}"
    shell = (
        f"cd {shlex.quote(working_directory)} && exec setsid sh -c {shlex.quote(inner)} "
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


def process_is_running(handle: ProcessHandle) -> bool:
    return handle.process is not None and handle.process.poll() is None


def append_note_once(handle: ProcessHandle, note: str) -> None:
    if note not in handle.notes:
        handle.notes.append(note)


def remote_group_is_running(handle: ProcessHandle) -> bool | None:
    if not handle.endpoint.remote or handle.process is None:
        return False
    pid_file = f"{handle.run_dir}/control.pid"
    kill = "sudo -n kill" if handle.endpoint.sudo else "kill"
    command = (
        f"test -s {shlex.quote(pid_file)} && "
        f"{kill} -0 -- -$(cat {shlex.quote(pid_file)})"
    )
    try:
        result = remote_run(handle.endpoint, command, check=False, capture=True)
    except (OSError, subprocess.SubprocessError) as error:
        append_note_once(handle, f"remote process-group state unavailable: {error}")
        return None
    return result.returncode == 0


def shutdown_target_is_running(handle: ProcessHandle) -> bool:
    if not handle.endpoint.remote:
        return process_is_running(handle)
    remote_state = remote_group_is_running(handle)
    return True if remote_state is None else remote_state or process_is_running(handle)


def signal_handle(handle: ProcessHandle, signal_name: str) -> None:
    if handle.process is None:
        return
    if handle.endpoint.remote:
        pid_file = f"{handle.run_dir}/control.pid"
        kill = "sudo -n kill" if handle.endpoint.sudo else "kill"
        command = (
            f"if test -s {shlex.quote(pid_file)}; then "
            f"pgid=$(cat {shlex.quote(pid_file)}); "
            f"if {kill} -0 -- -$pgid; then {kill} -{signal_name} -- -$pgid; fi; "
            "fi"
        )
        try:
            result = remote_run(handle.endpoint, command, check=False, capture=True)
        except (OSError, subprocess.SubprocessError) as error:
            append_note_once(handle, f"remote {signal_name} failed: {error}")
            result = None
        if result is not None and result.returncode != 0:
            append_note_once(handle, f"remote {signal_name} failed: {result.stderr.strip()}")
        if result is None or result.returncode != 0:
            if signal_name == "KILL" and process_is_running(handle):
                try:
                    os.killpg(handle.process.pid, signal.SIGKILL)
                    append_note_once(handle, "killed local SSH control process after remote KILL failure")
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


def stop_handles(handles: list[ProcessHandle], reason: str, grace_s: float) -> None:
    for handle in handles:
        if shutdown_target_is_running(handle):
            handle.stop_reason = reason
            signal_handle(handle, "INT")
    if not wait_handles(handles, grace_s):
        for handle in handles:
            signal_handle(handle, "TERM")
    if not wait_handles(handles, grace_s):
        for handle in handles:
            signal_handle(handle, "KILL")
    wait_handles(handles, grace_s)


def close_handle(handle: ProcessHandle) -> None:
    if handle.process is None:
        return
    if shutdown_target_is_running(handle) or handle.process.poll() is None:
        handle.notes.append("process remained alive after shutdown escalation")
        handle.run_status = "shutdown_failed"
    else:
        handle.process.wait()
        handle.run_status = "finished" if handle.process.returncode == 0 else "exited_nonzero"
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
        "status": "prepared",
    }


def update_manifest(handle: ProcessHandle) -> dict[str, Any]:
    value = dict(handle.initial_manifest)
    return_code: int | str = ""
    if handle.process is not None and handle.process.returncode is not None:
        return_code = handle.process.returncode
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
    if not handle.endpoint.sudo or handle.process is None:
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


def finalize_handle(handle: ProcessHandle) -> None:
    if not handle.prepared:
        handle.archive_status = "not_prepared"
        return
    if process_is_running(handle):
        handle.archive_status = "skipped_process_alive"
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
    return_code: int | str = ""
    if handle.process is not None and handle.process.returncode is not None:
        return_code = handle.process.returncode
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
        "return_code": return_code,
        "run_status": handle.run_status,
        "stop_reason": handle.stop_reason or ("process_exit" if handle.process is not None else handle.run_status),
        "archive_status": handle.archive_status,
        "sidecar_status": handle.sidecar_status,
        "notes": "; ".join(note for note in handle.notes if note),
    }


def append_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
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
) -> list[dict[str, Any]]:
    prepared = [handle for handle in handles if handle.prepared]
    for handle in prepared:
        if shutdown_target_is_running(handle):
            handle.sidecar_status = "skipped_process_alive" if handle.sidecar_tool != "none" else "not_requested"
        else:
            try:
                normalize_ownership(handle)
            except Exception as error:
                handle.notes.append(f"ownership normalization failed: {error}")
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
        manifest = run_manifest(endpoint, plan, run_dir, effective, environment)
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
                launch_index=launch_index,
            )
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
        phase = "measurement"
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            ended = [
                handle
                for handle in handles
                if handle.process is not None and handle.process.poll() is not None
            ]
            if len(ended) == len(handles):
                stop_reason = "all_processes_exited"
                break
            if ended and stop_on_role_exit:
                stop_reason = "paired_role_exited"
                break
            time.sleep(0.25)
    except BaseException as error:
        failure = error
        stop_reason = "interrupted" if isinstance(error, KeyboardInterrupt) else f"{phase}_failed"
        if active_handle is not None and active_handle.prepared and active_handle.process is None:
            active_handle.run_status = f"{phase}_failed"
            active_handle.notes.append(str(error))
        for handle in handles:
            if handle.prepared and handle.process is None and handle.run_status == "prepared":
                handle.run_status = "launch_not_attempted" if phase == "launch" else f"{phase}_aborted"
                handle.stop_reason = stop_reason
    finally:
        launched = [handle for handle in handles if handle.process is not None]
        if launched:
            stop_handles(launched, stop_reason, grace_s)
            for handle in launched:
                close_handle(handle)
        rows = finalize_runs(handles, plan, control_root)

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
    return {
        "campaign_id": plan.campaign_id,
        "experiment_id": plan.experiment_id,
        "case": plan.case_name,
        "variant": plan.variant.name,
        "trial": plan.trial,
        "start_order": list(spec["_start_order"]),
        "roles": roles,
    }


def integer_filter(values: list[str]) -> set[int]:
    parsed = {int(value) for value in values}
    if any(value < 1 for value in parsed):
        raise ValueError("trial filters must be positive")
    return parsed


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
    for plan in plans:
        try:
            rows = execute_plan(spec, plan, control_root)
            failures += any(
                row["archive_status"] != "finalized"
                or row["run_status"] != "finished"
                or (row["sidecar"] != "none" and row["sidecar_status"] != "registered")
                for row in rows
            )
        except Exception as error:
            failures += 1
            print(f"experiment failed: {plan.experiment_id}: {error}", file=sys.stderr)
            if not args.keep_going:
                break
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
