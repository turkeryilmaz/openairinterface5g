#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Run and preserve an attach-gated nrUE publication workload.

This helper is intentionally host-local.  The campaign runner invokes it on
the nrUE host so all clock anchors, interface observations, policy snapshots,
ping evidence, and iperf3 evidence use the source host's clocks and network
namespace.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import MISSING, asdict, dataclass
from math import isfinite
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


WORKLOAD_SCHEMA_VERSION = 1
WORKLOAD_RUN_SCHEMA_VERSION = 1
EXPECTED_CLIENT_ROLE = "nrUE"
EXPECTED_INTERFACE = "oaitun_ue1"
EXPECTED_SUBNET = ipaddress.IPv4Network("10.0.0.0/24")
EXPECTED_SERVER_IPV4 = ipaddress.IPv4Address("192.168.70.135")
EXPECTED_POLICY_TABLE = 9999
EXPECTED_PING_COUNT = 3
EXPECTED_DURATION_S = 120
EXPECTED_BITRATE_BPS = 1_000_000
EXPECTED_DATAGRAM_BYTES = 1200
IPERF_REPORT_INTERVAL_S = 1
IPERF_DURATION_TOLERANCE_FRACTION = 0.02
SAFE_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
PING_SUMMARY = re.compile(r"(?m)^(\d+) packets transmitted, (\d+) received(?:,|$)")


class WorkloadError(RuntimeError):
    """A preserved workload phase failed or could not be proven valid."""


class ReadinessPending(WorkloadError):
    """The UE dataplane is not ready yet but may become ready before timeout."""


@dataclass(frozen=True)
class WorkloadSpec:
    schema_version: int
    client_role: str
    helper: str
    interface: str
    ipv4_subnet: str
    server_ipv4: str
    policy_table: int
    readiness_timeout_s: float
    readiness_poll_s: float
    ping_count: int
    ping_timeout_s: int
    duration_s: int
    bitrate_bps: int
    datagram_bytes: int
    lease_path: str
    ip_binary: str = "ip"
    ping_binary: str = "ping"
    iperf3_binary: str = "iperf3"

    @property
    def subnet(self) -> ipaddress.IPv4Network:
        return ipaddress.IPv4Network(self.ipv4_subnet)

    @property
    def server(self) -> ipaddress.IPv4Address:
        return ipaddress.IPv4Address(self.server_ipv4)


WORKLOAD_SPEC_FIELDS = {field.name for field in WorkloadSpec.__dataclass_fields__.values()}
WORKLOAD_SPEC_REQUIRED_FIELDS = {
    name
    for name, field in WorkloadSpec.__dataclass_fields__.items()
    if field.default is MISSING and field.default_factory is MISSING
}


def require_string(value: dict[str, Any], name: str, default: str | None = None) -> str:
    raw = value.get(name, default)
    if type(raw) is not str:
        raise ValueError(f"workload {name} must be a string")
    return raw


def require_integer(value: dict[str, Any], name: str) -> int:
    raw = value[name]
    if type(raw) is not int:
        raise ValueError(f"workload {name} must be an integer")
    return raw


def require_finite_number(value: dict[str, Any], name: str) -> float:
    raw = value[name]
    if type(raw) not in {int, float} or not isfinite(raw):
        raise ValueError(f"workload {name} must be a finite number")
    return float(raw)


def parse_workload_spec(value: Any) -> WorkloadSpec:
    if not isinstance(value, dict):
        raise ValueError("workload must be an object")
    unknown = set(value) - WORKLOAD_SPEC_FIELDS
    if unknown:
        raise ValueError(f"unknown workload fields: {', '.join(sorted(unknown))}")
    missing = WORKLOAD_SPEC_REQUIRED_FIELDS - set(value)
    if missing:
        raise ValueError(f"missing workload fields: {', '.join(sorted(missing))}")
    try:
        spec = WorkloadSpec(
            schema_version=require_integer(value, "schema_version"),
            client_role=require_string(value, "client_role"),
            helper=require_string(value, "helper"),
            interface=require_string(value, "interface"),
            ipv4_subnet=require_string(value, "ipv4_subnet"),
            server_ipv4=require_string(value, "server_ipv4"),
            policy_table=require_integer(value, "policy_table"),
            readiness_timeout_s=require_finite_number(value, "readiness_timeout_s"),
            readiness_poll_s=require_finite_number(value, "readiness_poll_s"),
            ping_count=require_integer(value, "ping_count"),
            ping_timeout_s=require_integer(value, "ping_timeout_s"),
            duration_s=require_integer(value, "duration_s"),
            bitrate_bps=require_integer(value, "bitrate_bps"),
            datagram_bytes=require_integer(value, "datagram_bytes"),
            lease_path=require_string(value, "lease_path"),
            ip_binary=require_string(value, "ip_binary", "ip"),
            ping_binary=require_string(value, "ping_binary", "ping"),
            iperf3_binary=require_string(value, "iperf3_binary", "iperf3"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid workload field: {error}") from error

    helper = PurePosixPath(spec.helper)
    lease = PurePosixPath(spec.lease_path)
    if spec.schema_version != WORKLOAD_SCHEMA_VERSION:
        raise ValueError(f"workload schema_version must be {WORKLOAD_SCHEMA_VERSION}")
    if spec.client_role != EXPECTED_CLIENT_ROLE:
        raise ValueError(f"workload client_role must be {EXPECTED_CLIENT_ROLE}")
    if not helper.is_absolute() or ".." in helper.parts:
        raise ValueError("workload helper must be an absolute path without '..'")
    if spec.interface != EXPECTED_INTERFACE:
        raise ValueError(f"workload interface must be {EXPECTED_INTERFACE}")
    if spec.subnet != EXPECTED_SUBNET:
        raise ValueError(f"workload ipv4_subnet must be {EXPECTED_SUBNET}")
    if spec.server != EXPECTED_SERVER_IPV4:
        raise ValueError(f"workload server_ipv4 must be {EXPECTED_SERVER_IPV4}")
    if spec.policy_table != EXPECTED_POLICY_TABLE:
        raise ValueError(f"workload policy_table must be {EXPECTED_POLICY_TABLE}")
    if not (0 < spec.readiness_timeout_s <= 600):
        raise ValueError("workload readiness_timeout_s must be in (0, 600]")
    if not (0 < spec.readiness_poll_s <= min(10, spec.readiness_timeout_s)):
        raise ValueError("workload readiness_poll_s must be positive and bounded")
    if spec.ping_count != EXPECTED_PING_COUNT:
        raise ValueError(f"workload ping_count must be {EXPECTED_PING_COUNT}")
    if not (1 <= spec.ping_timeout_s <= 30):
        raise ValueError("workload ping_timeout_s must be in [1, 30]")
    if spec.duration_s != EXPECTED_DURATION_S:
        raise ValueError(f"workload duration_s must be {EXPECTED_DURATION_S}")
    if spec.bitrate_bps != EXPECTED_BITRATE_BPS:
        raise ValueError(f"workload bitrate_bps must be {EXPECTED_BITRATE_BPS}")
    if spec.datagram_bytes != EXPECTED_DATAGRAM_BYTES:
        raise ValueError(f"workload datagram_bytes must be {EXPECTED_DATAGRAM_BYTES}")
    if not lease.is_absolute() or ".." in lease.parts:
        raise ValueError("workload lease_path must be an absolute path without '..'")
    for label, binary in (
        ("ip_binary", spec.ip_binary),
        ("ping_binary", spec.ping_binary),
        ("iperf3_binary", spec.iperf3_binary),
    ):
        if not binary or "\x00" in binary:
            raise ValueError(f"workload {label} must be nonempty")
    return spec


def clock_anchors() -> dict[str, int]:
    return {
        "realtime_ns": time.time_ns(),
        "monotonic_raw_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
    }


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = stream.name
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def run_command(
    command: list[str],
    *,
    timeout_s: float,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    process_environment = os.environ.copy()
    if env:
        process_environment.update(env)
    return subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        env=process_environment,
    )


def timeout_stream_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def parse_table(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return None


def normalize_selector(value: Any) -> str:
    if value in {None, "", "all"}:
        return "all"
    try:
        return str(ipaddress.IPv4Network(str(value), strict=False))
    except ValueError as error:
        raise WorkloadError(f"invalid IPv4 policy selector: {value!r}") from error


def normalize_rule(row: dict[str, Any]) -> dict[str, Any]:
    priority = row.get("priority")
    table = parse_table(row.get("table"))
    if not isinstance(priority, int) or priority < 0:
        raise WorkloadError(f"invalid policy-rule priority: {priority!r}")
    if table is None:
        raise WorkloadError(f"policy rule has nonnumeric table: {row.get('table')!r}")
    return {
        "priority": priority,
        "src": normalize_selector(row.get("src")),
        "dst": normalize_selector(row.get("dst")),
        "table": table,
        "action": str(row.get("action", "lookup")),
    }


def normalize_route(row: dict[str, Any], table: int) -> dict[str, Any]:
    row_table = parse_table(row.get("table"))
    if row_table not in {None, table}:
        raise WorkloadError(f"route table drift: expected {table}, observed {row_table}")
    metric = row.get("metric")
    if metric is not None and (not isinstance(metric, int) or metric < 0):
        raise WorkloadError(f"invalid route metric: {metric!r}")
    return {
        "dst": str(row.get("dst", "default")),
        "dev": str(row.get("dev", "")),
        "gateway": str(row.get("gateway", "")),
        "table": table,
        "protocol": str(row.get("protocol", "")),
        "scope": str(row.get("scope", "")),
        "type": str(row.get("type", "unicast")),
        "metric": metric,
    }


def is_missing_device_error(stderr: str) -> bool:
    return any(
        marker in stderr
        for marker in (
            "Cannot find device",
            "does not exist",
            "No such device",
        )
    )


def _json_command(
    command: list[str],
    *,
    missing_table_is_empty: bool = False,
    missing_device_is_pending: bool = False,
) -> list[dict[str, Any]]:
    try:
        result = run_command(command, timeout_s=10)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise WorkloadError(f"could not execute {' '.join(command)}: {error}") from error
    if result.returncode != 0:
        missing_table = missing_table_is_empty and "FIB table does not exist" in result.stderr
        if missing_table:
            return []
        if missing_device_is_pending and is_missing_device_error(result.stderr):
            raise ReadinessPending(f"{EXPECTED_INTERFACE} does not exist yet")
        raise WorkloadError(
            f"command failed ({result.returncode}): {' '.join(command)}: {result.stderr.strip()}"
        )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise WorkloadError(f"command returned malformed JSON: {' '.join(command)}") from error
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise WorkloadError(f"command returned unexpected JSON shape: {' '.join(command)}")
    return value


def network_snapshot(spec: WorkloadSpec) -> dict[str, Any]:
    start = clock_anchors()
    raw_rules = _json_command([spec.ip_binary, "-N", "-j", "-4", "rule", "show"])
    raw_routes = _json_command(
        [
            spec.ip_binary,
            "-N",
            "-j",
            "-4",
            "route",
            "show",
            "table",
            str(spec.policy_table),
        ],
        missing_table_is_empty=True,
    )
    relevant_rules = []
    for row in raw_rules:
        if parse_table(row.get("table")) == spec.policy_table:
            relevant_rules.append(normalize_rule(row))
    routes = [normalize_route(row, spec.policy_table) for row in raw_routes]
    return {
        "schema_version": WORKLOAD_RUN_SCHEMA_VERSION,
        "start": start,
        "end": clock_anchors(),
        "policy_table": spec.policy_table,
        "rules": sorted(relevant_rules, key=lambda row: (row["priority"], row["src"], row["dst"])),
        "routes": sorted(
            routes,
            key=lambda row: (
                row["dst"],
                row["dev"],
                row["gateway"],
                row["metric"] if row["metric"] is not None else -1,
            ),
        ),
        "raw_rules": raw_rules,
        "raw_routes": raw_routes,
    }


def discover_global_ipv4(spec: WorkloadSpec) -> str:
    rows = _json_command(
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
    addresses: list[ipaddress.IPv4Interface] = []
    for row in rows:
        if row.get("ifname") != spec.interface:
            raise WorkloadError(
                f"ip address JSON returned unexpected interface {row.get('ifname')!r}"
            )
        info = row.get("addr_info", [])
        if not isinstance(info, list):
            raise WorkloadError("ip address JSON has invalid addr_info")
        for address in info:
            if not isinstance(address, dict):
                raise WorkloadError("ip address JSON has invalid address row")
            if address.get("family") != "inet" or address.get("scope") != "global":
                continue
            local = address.get("local")
            prefixlen = address.get("prefixlen")
            if not isinstance(local, str) or not isinstance(prefixlen, int):
                raise WorkloadError("global IPv4 row lacks local/prefixlen")
            addresses.append(ipaddress.IPv4Interface(f"{local}/{prefixlen}"))
    if not addresses:
        raise ReadinessPending(f"no global IPv4 exists on {spec.interface} yet")
    if len(addresses) != 1:
        raise WorkloadError(
            f"expected exactly one global IPv4 on {spec.interface}, observed {len(addresses)}"
        )
    address = addresses[0]
    if address.network != spec.subnet or address.ip not in spec.subnet:
        raise WorkloadError(
            f"global IPv4 {address} is outside the required subnet {spec.subnet}"
        )
    return str(address.ip)


def validate_ready_network(
    spec: WorkloadSpec,
    ipv4: str,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    source = f"{ipaddress.IPv4Address(ipv4)}/32"
    rules = snapshot["rules"]
    routes = snapshot["routes"]
    from_rules = [
        row
        for row in rules
        if row["src"] == source
        and row["dst"] == "all"
        and row["table"] == spec.policy_table
        and row["action"] == "lookup"
    ]
    to_rules = [
        row
        for row in rules
        if row["src"] == "all"
        and row["dst"] == source
        and row["table"] == spec.policy_table
        and row["action"] == "lookup"
    ]
    expected_routes = [
        row
        for row in routes
        if row["dst"] == "default"
        and row["dev"] == spec.interface
        and row["gateway"] == ""
        and row["table"] == spec.policy_table
    ]
    expected_rule_keys = {identity_key(row) for row in from_rules + to_rules}
    observed_rule_keys = {identity_key(row) for row in rules}
    expected_route_keys = {identity_key(row) for row in expected_routes}
    observed_route_keys = {identity_key(row) for row in routes}
    if observed_rule_keys - expected_rule_keys:
        raise WorkloadError(
            f"unexpected or malformed policy rule exists in table {spec.policy_table}"
        )
    if observed_route_keys - expected_route_keys:
        raise WorkloadError(
            f"unexpected or malformed route exists in table {spec.policy_table}"
        )
    if len(from_rules) > 1 or len(to_rules) > 1 or len(expected_routes) > 1:
        raise WorkloadError("duplicate workload policy state exists")
    missing: list[str] = []
    if len(from_rules) != 1:
        missing.append(f"source rule for {source}")
    if len(to_rules) != 1:
        missing.append(f"destination rule for {source}")
    if len(expected_routes) != 1:
        missing.append(f"default dev {spec.interface} route")
    if missing:
        raise ReadinessPending(f"workload policy state is not complete yet: {', '.join(missing)}")
    return {
        "ipv4": ipv4,
        "source_rule": from_rules[0],
        "destination_rule": to_rules[0],
        "default_route": expected_routes[0],
    }


def process_ids_by_name(name: str) -> list[int]:
    matches: list[int] = []
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            command_name = (entry / "comm").read_text(errors="replace").strip()
            command_line = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(
                errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if command_name == name or Path(command_line.split(" ", 1)[0]).name == name:
            matches.append(int(entry.name))
    return sorted(matches)


def interface_exists(spec: WorkloadSpec) -> bool:
    command = [spec.ip_binary, "-N", "-j", "link", "show", "dev", spec.interface]
    try:
        result = run_command(command, timeout_s=10)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise WorkloadError(f"could not inspect {spec.interface}: {error}") from error
    if result.returncode != 0:
        if is_missing_device_error(result.stderr):
            return False
        raise WorkloadError(
            f"interface inspection failed ({result.returncode}): {result.stderr.strip()}"
        )
    try:
        rows = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise WorkloadError("interface inspection returned malformed JSON") from error
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise WorkloadError("interface inspection returned an unexpected JSON shape")
    if rows[0].get("ifname") != spec.interface:
        raise WorkloadError("interface inspection returned the wrong interface")
    return True


def validate_experiment_id(experiment_id: str) -> str:
    if not SAFE_EXPERIMENT_ID.fullmatch(experiment_id) or experiment_id in {".", ".."}:
        raise ValueError(f"invalid experiment ID: {experiment_id!r}")
    return experiment_id


def lease_owner(spec: WorkloadSpec, run_dir: Path, experiment_id: str) -> dict[str, Any]:
    return {
        "schema_version": WORKLOAD_RUN_SCHEMA_VERSION,
        "experiment_id": validate_experiment_id(experiment_id),
        "run_dir": str(run_dir.resolve()),
        "pid": os.getpid(),
        "created": clock_anchors(),
        "interface": spec.interface,
        "policy_table": spec.policy_table,
    }


def acquire_lease(spec: WorkloadSpec, run_dir: Path, experiment_id: str) -> dict[str, Any]:
    path = Path(spec.lease_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError as error:
        raise WorkloadError(f"workload lease already exists: {path}") from error
    owner = lease_owner(spec, run_dir, experiment_id)
    try:
        atomic_write_json(path / "owner.json", owner)
    except Exception:
        path.rmdir()
        raise
    return owner


def verify_lease(spec: WorkloadSpec, run_dir: Path, experiment_id: str) -> dict[str, Any]:
    path = Path(spec.lease_path)
    try:
        owner = json.loads((path / "owner.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise WorkloadError(f"workload lease is missing or malformed: {path}") from error
    if not isinstance(owner, dict):
        raise WorkloadError(f"workload lease owner is not an object: {path}")
    expected = {
        "experiment_id": validate_experiment_id(experiment_id),
        "run_dir": str(run_dir.resolve()),
        "interface": spec.interface,
        "policy_table": spec.policy_table,
    }
    for key, value in expected.items():
        if owner.get(key) != value:
            raise WorkloadError(
                f"workload lease ownership mismatch for {key}: {owner.get(key)!r} != {value!r}"
            )
    return owner


def release_lease(spec: WorkloadSpec, run_dir: Path, experiment_id: str) -> None:
    verify_lease(spec, run_dir, experiment_id)
    path = Path(spec.lease_path)
    (path / "owner.json").unlink()
    path.rmdir()


def workload_dir(run_dir: Path) -> Path:
    return run_dir / "workload"


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise WorkloadError(f"missing or malformed workload evidence: {path}") from error
    if not isinstance(value, dict):
        raise WorkloadError(f"workload evidence must be an object: {path}")
    return value


def update_run_state(run_dir: Path, **updates: Any) -> dict[str, Any]:
    path = workload_dir(run_dir) / "workload_run.json"
    value: dict[str, Any] = {}
    if path.exists():
        value = load_json_object(path)
    value.update(updates)
    value["schema_version"] = WORKLOAD_RUN_SCHEMA_VERSION
    value["updated"] = clock_anchors()
    atomic_write_json(path, value)
    return value


def command_plan(spec: WorkloadSpec) -> dict[str, Any]:
    dynamic = "<dynamic-ue-ip>"
    return {
        "schema_version": WORKLOAD_RUN_SCHEMA_VERSION,
        "client_role": spec.client_role,
        "readiness": {
            "timeout_s": spec.readiness_timeout_s,
            "poll_s": spec.readiness_poll_s,
            "interface": spec.interface,
            "required_subnet": str(spec.subnet),
            "policy_table": spec.policy_table,
            "required_rules": [
                f"from {dynamic}/32 lookup {spec.policy_table}",
                f"to {dynamic}/32 lookup {spec.policy_table}",
            ],
            "required_route": f"default dev {spec.interface} table {spec.policy_table}",
        },
        "ping_command": [
            spec.ping_binary,
            "-I",
            dynamic,
            "-c",
            str(spec.ping_count),
            "-W",
            str(spec.ping_timeout_s),
            str(spec.server),
        ],
        "iperf3_command": [
            spec.iperf3_binary,
            "-c",
            str(spec.server),
            "-B",
            dynamic,
            "--bind-dev",
            spec.interface,
            "-u",
            "--bidir",
            "-b",
            str(spec.bitrate_bps),
            "-l",
            str(spec.datagram_bytes),
            "-t",
            str(spec.duration_s),
            "-i",
            str(IPERF_REPORT_INTERVAL_S),
            "--get-server-output",
            "--udp-counters-64bit",
            "-J",
        ],
        "traffic_contract": {
            "protocol": "UDP",
            "direction": "bidirectional",
            "bitrate_bps_per_direction": spec.bitrate_bps,
            "datagram_bytes": spec.datagram_bytes,
            "post_readiness_duration_s": spec.duration_s,
        },
        "cleanup_contract": {
            "lease_path": spec.lease_path,
            "verified_nrue_death_required": True,
            "competing_nrue_or_tun_refuses_cleanup": True,
            "exact_rule_identity": ["priority", "src", "dst", "table", "action"],
            "flush_table": False,
        },
    }


def preflight(spec: WorkloadSpec, run_dir: Path, experiment_id: str) -> None:
    evidence = workload_dir(run_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    atomic_write_json(evidence / "workload_plan.json", command_plan(spec))
    update_run_state(
        run_dir,
        experiment_id=experiment_id,
        status="preflight_started",
        workload_status="not_started",
        cleanup_status="not_started",
    )
    missing = [
        binary
        for binary in (spec.ip_binary, spec.ping_binary, spec.iperf3_binary)
        if shutil.which(binary) is None
    ]
    if not Path(spec.helper).is_file():
        missing.append(spec.helper)
    if missing:
        update_run_state(run_dir, status="preflight_failed", error=f"missing tools: {missing}")
        raise WorkloadError(f"missing workload tools: {', '.join(missing)}")

    owner: dict[str, Any] | None = None
    preflight_evidence: dict[str, Any] = {
        "schema_version": WORKLOAD_RUN_SCHEMA_VERSION,
        "start": clock_anchors(),
        "status": "running",
    }
    try:
        owner = acquire_lease(spec, run_dir, experiment_id)
        preflight_evidence["lease_owner"] = owner
        competitors = process_ids_by_name("nr-uesoftmodem")
        tun_exists = interface_exists(spec)
        preflight_evidence["competing_nrue_pids"] = competitors
        preflight_evidence["interface_exists"] = tun_exists
        if competitors or tun_exists:
            raise WorkloadError(
                f"preflight requires no nrUE/TUN: pids={competitors}, {spec.interface}={tun_exists}"
            )
        baseline = network_snapshot(spec)
        atomic_write_json(evidence / "network_baseline.json", baseline)
        if baseline["rules"] or baseline["routes"]:
            raise WorkloadError(
                "preflight detected policy residue; preserve and clean it explicitly outside the runner"
            )
        preflight_evidence["status"] = "ok"
        update_run_state(run_dir, status="preflight_complete")
    except Exception as error:
        preflight_evidence["status"] = "failed"
        preflight_evidence["error"] = str(error)
        update_run_state(run_dir, status="preflight_failed", error=str(error))
        if owner is not None:
            release_lease(spec, run_dir, experiment_id)
            preflight_evidence["lease_released"] = True
        raise
    finally:
        preflight_evidence["end"] = clock_anchors()
        atomic_write_json(evidence / "network_preflight.json", preflight_evidence)


def identity_key(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"))


def exact_delta(
    baseline: Iterable[dict[str, Any]],
    active: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_keys = {identity_key(row) for row in baseline}
    return [row for row in active if identity_key(row) not in baseline_keys]


def build_ownership(
    spec: WorkloadSpec,
    ipv4: str,
    baseline: dict[str, Any],
    active: dict[str, Any],
) -> dict[str, Any]:
    validation = validate_ready_network(spec, ipv4, active)
    owned_rules = exact_delta(baseline["rules"], active["rules"])
    owned_routes = exact_delta(baseline["routes"], active["routes"])
    expected_rules = [validation["source_rule"], validation["destination_rule"]]
    if {identity_key(row) for row in owned_rules} != {
        identity_key(row) for row in expected_rules
    }:
        raise WorkloadError("active policy rules are not the exact baseline delta")
    if len(owned_routes) != 1 or owned_routes[0] != validation["default_route"]:
        raise WorkloadError("active default route is not the exact baseline delta")
    return {
        "schema_version": WORKLOAD_RUN_SCHEMA_VERSION,
        "captured": clock_anchors(),
        "ipv4": ipv4,
        "policy_table": spec.policy_table,
        "rules": sorted(owned_rules, key=lambda row: row["priority"]),
        "routes": owned_routes,
    }


def wait_for_readiness(
    spec: WorkloadSpec,
    run_dir: Path,
) -> tuple[str, dict[str, Any]]:
    path = workload_dir(run_dir) / "workload_readiness.json"
    deadline = time.monotonic() + spec.readiness_timeout_s
    attempts: list[dict[str, Any]] = []
    while True:
        attempt: dict[str, Any] = {
            "attempt": len(attempts) + 1,
            "start": clock_anchors(),
            "status": "pending",
        }
        try:
            ipv4 = discover_global_ipv4(spec)
            snapshot = network_snapshot(spec)
            validation = validate_ready_network(spec, ipv4, snapshot)
            attempt.update(
                {
                    "status": "ready",
                    "ipv4": ipv4,
                    "validation": validation,
                    "snapshot": snapshot,
                }
            )
            attempt["end"] = clock_anchors()
            attempts.append(attempt)
            atomic_write_json(
                path,
                {
                    "schema_version": WORKLOAD_RUN_SCHEMA_VERSION,
                    "status": "ready",
                    "attempts": attempts,
                },
            )
            return ipv4, snapshot
        except ReadinessPending as error:
            attempt["reason"] = str(error)
        except WorkloadError as error:
            attempt["status"] = "failed_integrity"
            attempt["reason"] = str(error)
            attempt["end"] = clock_anchors()
            attempts.append(attempt)
            atomic_write_json(
                path,
                {
                    "schema_version": WORKLOAD_RUN_SCHEMA_VERSION,
                    "status": "failed_integrity",
                    "attempts": attempts,
                },
            )
            raise
        except Exception as error:
            attempt["status"] = "failed_error"
            attempt["reason"] = f"{type(error).__name__}: {error}"
            attempt["end"] = clock_anchors()
            attempts.append(attempt)
            atomic_write_json(
                path,
                {
                    "schema_version": WORKLOAD_RUN_SCHEMA_VERSION,
                    "status": "failed_error",
                    "attempts": attempts,
                },
            )
            raise
        attempt["end"] = clock_anchors()
        attempts.append(attempt)
        atomic_write_json(
            path,
            {
                "schema_version": WORKLOAD_RUN_SCHEMA_VERSION,
                "status": "waiting",
                "attempts": attempts,
            },
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            atomic_write_json(
                path,
                {
                    "schema_version": WORKLOAD_RUN_SCHEMA_VERSION,
                    "status": "timeout",
                    "attempts": attempts,
                },
            )
            raise WorkloadError(
                f"nrUE dataplane readiness timed out after {spec.readiness_timeout_s}s"
            )
        time.sleep(min(spec.readiness_poll_s, remaining))


def run_ping(spec: WorkloadSpec, run_dir: Path, ipv4: str) -> dict[str, Any]:
    command = [
        spec.ping_binary,
        "-I",
        ipv4,
        "-c",
        str(spec.ping_count),
        "-W",
        str(spec.ping_timeout_s),
        str(spec.server),
    ]
    start = clock_anchors()
    timeout_s = spec.ping_count * spec.ping_timeout_s + 10
    try:
        result = run_command(
            command,
            timeout_s=timeout_s,
            env={"LC_ALL": "C"},
        )
    except subprocess.TimeoutExpired as error:
        end = clock_anchors()
        stdout = timeout_stream_text(error.stdout)
        stderr = timeout_stream_text(error.stderr)
        evidence = workload_dir(run_dir)
        atomic_write_text(evidence / "ping.stdout.log", stdout)
        atomic_write_text(evidence / "ping.stderr.log", stderr)
        summary = PING_SUMMARY.search(stdout)
        transmitted = int(summary.group(1)) if summary else -1
        received = int(summary.group(2)) if summary else -1
        atomic_write_json(
            evidence / "ping.json",
            {
                "schema_version": WORKLOAD_RUN_SCHEMA_VERSION,
                "command": command,
                "start": start,
                "end": end,
                "return_code": "",
                "transmitted": transmitted,
                "received": received,
                "status": "timeout",
                "timeout_s": timeout_s,
                "error": str(error),
            },
        )
        raise WorkloadError(
            f"bound ping timed out after {timeout_s}s: tx={transmitted}, rx={received}"
        ) from error
    end = clock_anchors()
    evidence = workload_dir(run_dir)
    atomic_write_text(evidence / "ping.stdout.log", result.stdout)
    atomic_write_text(evidence / "ping.stderr.log", result.stderr)
    summary = PING_SUMMARY.search(result.stdout)
    transmitted = int(summary.group(1)) if summary else -1
    received = int(summary.group(2)) if summary else -1
    value = {
        "schema_version": WORKLOAD_RUN_SCHEMA_VERSION,
        "command": command,
        "start": start,
        "end": end,
        "return_code": result.returncode,
        "transmitted": transmitted,
        "received": received,
        "status": "ok"
        if result.returncode == 0
        and transmitted == spec.ping_count
        and received == spec.ping_count
        else "failed",
    }
    atomic_write_json(evidence / "ping.json", value)
    if value["status"] != "ok":
        raise WorkloadError(
            f"bound ping did not prove {spec.ping_count}/{spec.ping_count}: "
            f"rc={result.returncode}, tx={transmitted}, rx={received}"
        )
    return value


def finite_metric(row: dict[str, Any], name: str) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkloadError(f"iperf3 summary has invalid {name}: {value!r}")
    result = float(value)
    if not isfinite(result):
        raise WorkloadError(f"iperf3 summary has non-finite {name}")
    return result


def nonnegative_integer_metric(row: dict[str, Any], name: str) -> int:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkloadError(f"iperf3 summary has invalid {name}: {value!r}")
    return value


def normalized_ipv4_metadata(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise WorkloadError(f"iperf3 {field_name} is not an IPv4 string")
    try:
        return str(ipaddress.IPv4Address(value))
    except ValueError as error:
        raise WorkloadError(f"iperf3 {field_name} is not valid IPv4") from error


def validate_iperf3_start(
    spec: WorkloadSpec,
    parsed: dict[str, Any],
    ipv4: str,
) -> dict[str, Any]:
    start = parsed.get("start")
    if not isinstance(start, dict):
        raise WorkloadError("iperf3 JSON lacks a start object")
    connecting_to = start.get("connecting_to")
    if not isinstance(connecting_to, dict):
        raise WorkloadError("iperf3 start.connecting_to is not an object")
    target = normalized_ipv4_metadata(
        connecting_to.get("host"),
        "start.connecting_to.host",
    )
    if target != str(spec.server):
        raise WorkloadError(
            f"iperf3 connected to {target}, expected {spec.server}"
        )

    connected = start.get("connected")
    if not isinstance(connected, list) or not connected:
        raise WorkloadError("iperf3 start.connected must be a nonempty list")
    expected_local = str(ipaddress.IPv4Address(ipv4))
    connections = []
    for index, row in enumerate(connected):
        if not isinstance(row, dict):
            raise WorkloadError(f"iperf3 start.connected[{index}] is not an object")
        local = normalized_ipv4_metadata(
            row.get("local_host"),
            f"start.connected[{index}].local_host",
        )
        remote = normalized_ipv4_metadata(
            row.get("remote_host"),
            f"start.connected[{index}].remote_host",
        )
        if local != expected_local or remote != str(spec.server):
            raise WorkloadError(
                f"iperf3 start.connected[{index}] endpoint mismatch: "
                f"local={local}, remote={remote}"
            )
        connections.append({"local_host": local, "remote_host": remote})

    test_start = start.get("test_start")
    if not isinstance(test_start, dict):
        raise WorkloadError("iperf3 start.test_start is not an object")
    expected = {
        "protocol": "UDP",
        "bidir": 1,
        "duration": spec.duration_s,
        "blksize": spec.datagram_bytes,
        "target_bitrate": spec.bitrate_bps,
    }
    for name, expected_value in expected.items():
        value = test_start.get(name)
        if type(value) is not type(expected_value) or value != expected_value:
            raise WorkloadError(
                f"iperf3 start.test_start.{name} is {value!r}, "
                f"expected {expected_value!r}"
            )
    return {
        "connecting_to_host": target,
        "connection_count": len(connections),
        "connected": connections,
        "test_start": expected,
    }


def summarize_iperf3_view(
    spec: WorkloadSpec,
    row: Any,
    *,
    direction: str,
    view: str,
    json_key: str,
    expected_sender: bool,
) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise WorkloadError(f"iperf3 JSON lacks {json_key}")
    if row.get("sender") is not expected_sender:
        raise WorkloadError(
            f"iperf3 {json_key} sender flag is {row.get('sender')!r}, "
            f"expected {expected_sender}"
        )
    seconds = finite_metric(row, "seconds")
    bitrate_bps = finite_metric(row, "bits_per_second")
    byte_count = nonnegative_integer_metric(row, "bytes")
    duration_tolerance_s = max(
        1.0,
        spec.duration_s * IPERF_DURATION_TOLERANCE_FRACTION,
    )
    duration_error_s = seconds - spec.duration_s
    if abs(duration_error_s) > duration_tolerance_s:
        raise WorkloadError(
            f"iperf3 {json_key} duration {seconds}s is outside "
            f"{spec.duration_s}±{duration_tolerance_s}s"
        )
    if expected_sender and byte_count == 0:
        raise WorkloadError(f"iperf3 {json_key} has no byte evidence")
    if bitrate_bps < 0 or (expected_sender and bitrate_bps == 0):
        raise WorkloadError(f"iperf3 {json_key} has an invalid rate")

    packets: int | None = None
    lost_packets: int | None = None
    lost_percent: float | None = None
    jitter_ms: float | None = None
    packet_metrics_valid = False
    packet_metrics_status = "not_authoritative_sender_view"
    jitter_valid = False
    jitter_status = "not_authoritative_sender_view"
    if expected_sender:
        reported_packets = row.get("packets")
        if reported_packets == 0 and byte_count > 0:
            packet_metrics_status = "not_authoritative_inconsistent_zero_packets"
    else:
        packets = nonnegative_integer_metric(row, "packets")
        if packets == 0:
            reported_lost_packets = nonnegative_integer_metric(row, "lost_packets")
            reported_lost_percent = finite_metric(row, "lost_percent")
            reported_jitter_ms = finite_metric(row, "jitter_ms")
            if (
                byte_count != 0
                or bitrate_bps != 0
                or reported_lost_packets != 0
                or reported_lost_percent != 0
                or reported_jitter_ms < 0
            ):
                raise WorkloadError(
                    f"iperf3 {json_key} zero-delivery receiver evidence is inconsistent"
                )
            packets = None
            packet_metrics_status = "no_received_datagrams_loss_unavailable"
            jitter_status = "no_received_datagrams_jitter_unavailable"
        else:
            lost_packets = nonnegative_integer_metric(row, "lost_packets")
            lost_percent = finite_metric(row, "lost_percent")
            jitter_ms = finite_metric(row, "jitter_ms")
            if jitter_ms < 0:
                raise WorkloadError(f"iperf3 {json_key} has negative jitter")
            if lost_packets > packets:
                raise WorkloadError(f"iperf3 {json_key} lost_packets exceeds packets")
            if not (0 <= lost_percent <= 100):
                raise WorkloadError(f"iperf3 {json_key} lost_percent is outside [0, 100]")
            if (byte_count == 0) != (bitrate_bps == 0):
                raise WorkloadError(
                    f"iperf3 {json_key} zero bytes/rate evidence is inconsistent"
                )
            if byte_count == 0 and lost_packets != packets:
                raise WorkloadError(
                    f"iperf3 {json_key} zero achieved payload is not explained by total loss"
                )
            packet_metrics_valid = True
            packet_metrics_status = "valid_receiver_view"
            if lost_packets == packets:
                jitter_ms = None
                jitter_status = "no_received_datagrams_jitter_unavailable"
            else:
                jitter_valid = True
                jitter_status = "valid_receiver_view"
    return {
        "direction": direction,
        "view": view,
        "json_key": json_key,
        "sender": expected_sender,
        "requested_bitrate_bps": spec.bitrate_bps,
        "measured_bitrate_bps": bitrate_bps,
        "measured_to_requested_ratio": bitrate_bps / spec.bitrate_bps,
        "requested_duration_s": spec.duration_s,
        "measured_duration_s": seconds,
        "duration_error_s": duration_error_s,
        "duration_tolerance_s": duration_tolerance_s,
        "bytes": byte_count,
        "packets": packets,
        "lost_packets": lost_packets,
        "lost_percent": lost_percent,
        "jitter_ms": jitter_ms,
        "packet_metrics_valid": packet_metrics_valid,
        "packet_metrics_status": packet_metrics_status,
        "jitter_valid": jitter_valid,
        "jitter_status": jitter_status,
        "reported_packets": row.get("packets"),
        "reported_lost_packets": row.get("lost_packets"),
        "reported_lost_percent": row.get("lost_percent"),
        "reported_jitter_ms": row.get("jitter_ms"),
    }


def summarize_iperf3_bidir(
    spec: WorkloadSpec,
    parsed: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    end = parsed.get("end")
    if not isinstance(end, dict):
        raise WorkloadError("iperf3 JSON lacks an end object")
    views = [
        summarize_iperf3_view(
            spec,
            end.get("sum_sent"),
            direction="nrUE_to_ext-DN_UL",
            view="sender",
            json_key="end.sum_sent",
            expected_sender=True,
        ),
        summarize_iperf3_view(
            spec,
            end.get("sum_received"),
            direction="nrUE_to_ext-DN_UL",
            view="receiver",
            json_key="end.sum_received",
            expected_sender=False,
        ),
        summarize_iperf3_view(
            spec,
            end.get("sum_sent_bidir_reverse"),
            direction="ext-DN_to_nrUE_DL",
            view="sender",
            json_key="end.sum_sent_bidir_reverse",
            expected_sender=True,
        ),
        summarize_iperf3_view(
            spec,
            end.get("sum_received_bidir_reverse"),
            direction="ext-DN_to_nrUE_DL",
            view="receiver",
            json_key="end.sum_received_bidir_reverse",
            expected_sender=False,
        ),
    ]
    directions = []
    for direction in ("nrUE_to_ext-DN_UL", "ext-DN_to_nrUE_DL"):
        sender = next(
            row for row in views if row["direction"] == direction and row["view"] == "sender"
        )
        receiver = next(
            row for row in views if row["direction"] == direction and row["view"] == "receiver"
        )
        directions.append(
            {
                "direction": direction,
                "requested_bitrate_bps": spec.bitrate_bps,
                "offered_bitrate_bps": sender["measured_bitrate_bps"],
                "achieved_bitrate_bps": receiver["measured_bitrate_bps"],
                "achieved_to_requested_ratio": receiver["measured_to_requested_ratio"],
                "receiver_packets": receiver["packets"],
                "receiver_lost_packets": receiver["lost_packets"],
                "receiver_lost_percent": receiver["lost_percent"],
                "receiver_packet_metrics_valid": receiver["packet_metrics_valid"],
                "receiver_packet_metrics_status": receiver["packet_metrics_status"],
                "receiver_jitter_ms": receiver["jitter_ms"],
                "receiver_jitter_valid": receiver["jitter_valid"],
                "receiver_jitter_status": receiver["jitter_status"],
                "sender_json_key": sender["json_key"],
                "receiver_json_key": receiver["json_key"],
            }
        )
    return views, directions


def run_iperf3(spec: WorkloadSpec, run_dir: Path, ipv4: str) -> dict[str, Any]:
    command = [
        spec.iperf3_binary,
        "-c",
        str(spec.server),
        "-B",
        ipv4,
        "--bind-dev",
        spec.interface,
        "-u",
        "--bidir",
        "-b",
        str(spec.bitrate_bps),
        "-l",
        str(spec.datagram_bytes),
        "-t",
        str(spec.duration_s),
        "-i",
        str(IPERF_REPORT_INTERVAL_S),
        "--get-server-output",
        "--udp-counters-64bit",
        "-J",
    ]
    start = clock_anchors()
    timeout_s = spec.duration_s + 30
    try:
        result = run_command(command, timeout_s=timeout_s, env={"LC_ALL": "C"})
    except subprocess.TimeoutExpired as error:
        end = clock_anchors()
        stdout = timeout_stream_text(error.stdout)
        stderr = timeout_stream_text(error.stderr)
        evidence = workload_dir(run_dir)
        atomic_write_text(evidence / "iperf3.json", stdout)
        atomic_write_text(evidence / "iperf3.stderr.log", stderr)
        atomic_write_text(evidence / "iperf3_server.stdout.log", "")
        atomic_write_json(
            evidence / "iperf3_run.json",
            {
                "schema_version": WORKLOAD_RUN_SCHEMA_VERSION,
                "command": command,
                "start_realtime_ns": start["realtime_ns"],
                "end_realtime_ns": end["realtime_ns"],
                "start_monotonic_raw_ns": start["monotonic_raw_ns"],
                "end_monotonic_raw_ns": end["monotonic_raw_ns"],
                "return_code": "",
                "status": "timeout",
                "timeout_s": timeout_s,
                "parse_error": "",
                "tool_error": "",
                "validation_error": "subprocess timeout",
                "requested_duration_s": spec.duration_s,
                "bitrate_bps_per_direction": spec.bitrate_bps,
                "datagram_bytes": spec.datagram_bytes,
                "ipv4": ipv4,
                "server_ipv4": str(spec.server),
                "server_output_text_present": False,
                "views": [],
                "directions": [],
                "error": str(error),
            },
        )
        raise WorkloadError(f"iperf3 timed out after {timeout_s}s") from error
    end = clock_anchors()
    evidence = workload_dir(run_dir)
    atomic_write_text(evidence / "iperf3.json", result.stdout)
    atomic_write_text(evidence / "iperf3.stderr.log", result.stderr)
    parsed: Any = None
    parse_error = ""
    try:
        parsed = json.loads(result.stdout)
        if not isinstance(parsed, dict):
            parse_error = "iperf3 JSON root is not an object"
    except json.JSONDecodeError as error:
        parse_error = str(error)
    tool_error = parsed.get("error", "") if isinstance(parsed, dict) else ""
    views: list[dict[str, Any]] = []
    directions: list[dict[str, Any]] = []
    start_validation: dict[str, Any] = {}
    validation_error = ""
    server_output_text = (
        parsed.get("server_output_text", "") if isinstance(parsed, dict) else ""
    )
    if not isinstance(server_output_text, str):
        validation_error = "iperf3 server_output_text is not a string"
        server_output_text = ""
    if isinstance(parsed, dict) and not parse_error and not tool_error and not validation_error:
        try:
            start_validation = validate_iperf3_start(spec, parsed, ipv4)
            views, directions = summarize_iperf3_bidir(spec, parsed)
        except WorkloadError as error:
            validation_error = str(error)
    atomic_write_text(evidence / "iperf3_server.stdout.log", server_output_text)
    status = (
        "ok"
        if result.returncode == 0
        and not parse_error
        and not tool_error
        and not validation_error
        and len(views) == 4
        and len(directions) == 2
        else "failed"
    )
    value = {
        "schema_version": WORKLOAD_RUN_SCHEMA_VERSION,
        "command": command,
        "start_realtime_ns": start["realtime_ns"],
        "end_realtime_ns": end["realtime_ns"],
        "start_monotonic_raw_ns": start["monotonic_raw_ns"],
        "end_monotonic_raw_ns": end["monotonic_raw_ns"],
        "return_code": result.returncode,
        "status": status,
        "parse_error": parse_error,
        "tool_error": tool_error,
        "validation_error": validation_error,
        "requested_duration_s": spec.duration_s,
        "bitrate_bps_per_direction": spec.bitrate_bps,
        "datagram_bytes": spec.datagram_bytes,
        "ipv4": ipv4,
        "server_ipv4": str(spec.server),
        "server_output_text_present": bool(server_output_text),
        "start_validation": start_validation,
        "views": views,
        "directions": directions,
    }
    atomic_write_json(evidence / "iperf3_run.json", value)
    if status != "ok":
        raise WorkloadError(
            f"iperf3 failed: rc={result.returncode}, parse={parse_error!r}, "
            f"error={tool_error!r}, validation={validation_error!r}"
        )
    return value


def run_workload(spec: WorkloadSpec, run_dir: Path, experiment_id: str) -> None:
    verify_lease(spec, run_dir, experiment_id)
    update_run_state(run_dir, status="readiness", workload_status="running")
    try:
        ipv4, active = wait_for_readiness(spec, run_dir)
        evidence = workload_dir(run_dir)
        atomic_write_json(evidence / "network_active.json", active)
        baseline = load_json_object(evidence / "network_baseline.json")
        ownership = build_ownership(spec, ipv4, baseline, active)
        atomic_write_json(evidence / "network_ownership.json", ownership)
        update_run_state(run_dir, status="ping", ipv4=ipv4)
        ping = run_ping(spec, run_dir, ipv4)
        update_run_state(run_dir, status="traffic", ping_status=ping["status"])
        iperf = run_iperf3(spec, run_dir, ipv4)
        update_run_state(
            run_dir,
            status="completed",
            workload_status="completed",
            iperf3_status=iperf["status"],
            source_type="iperf3_udp_bidir",
            artifact_path="workload/iperf3.json",
            start_realtime_ns=iperf["start_realtime_ns"],
            end_realtime_ns=iperf["end_realtime_ns"],
            start_monotonic_raw_ns=iperf["start_monotonic_raw_ns"],
            end_monotonic_raw_ns=iperf["end_monotonic_raw_ns"],
        )
    except BaseException as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        update_run_state(
            run_dir,
            status="interrupted" if interrupted else "failed",
            workload_status="interrupted" if interrupted else "failed",
            error=f"{type(error).__name__}: {error}",
        )
        raise


def rule_delete_command(spec: WorkloadSpec, rule: dict[str, Any]) -> list[str]:
    normalized = normalize_rule(rule)
    if normalized["table"] != spec.policy_table:
        raise WorkloadError("refusing to delete a rule outside the workload table")
    return [
        spec.ip_binary,
        "-4",
        "rule",
        "del",
        "priority",
        str(normalized["priority"]),
        "from",
        normalized["src"],
        "to",
        normalized["dst"],
        "table",
        str(normalized["table"]),
    ]


def route_delete_command(spec: WorkloadSpec, route: dict[str, Any]) -> list[str]:
    normalized = normalize_route(route, spec.policy_table)
    if (
        normalized["dst"] != "default"
        or normalized["dev"] != spec.interface
        or normalized["gateway"]
    ):
        raise WorkloadError("refusing to delete a non-workload route")
    command = [
        spec.ip_binary,
        "-4",
        "route",
        "del",
        "default",
        "dev",
        spec.interface,
        "table",
        str(spec.policy_table),
    ]
    if normalized["metric"] is not None:
        command.extend(["metric", str(normalized["metric"])])
    return command


def validate_cleanup_ownership(
    baseline: dict[str, Any],
    ownership: dict[str, Any],
    current: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_rules = {identity_key(row) for row in baseline["rules"]}
    owned_rules = {identity_key(row) for row in ownership["rules"]}
    baseline_routes = {identity_key(row) for row in baseline["routes"]}
    owned_routes = {identity_key(row) for row in ownership["routes"]}
    current_rule_keys = {identity_key(row) for row in current["rules"]}
    current_route_keys = {identity_key(row) for row in current["routes"]}
    if not current_rule_keys <= baseline_rules | owned_rules:
        raise WorkloadError("unowned policy rule appeared; refusing cleanup")
    if not current_route_keys <= baseline_routes | owned_routes:
        raise WorkloadError("unowned policy route appeared; refusing cleanup")

    current_by_priority: dict[int, list[dict[str, Any]]] = {}
    for row in current["rules"]:
        current_by_priority.setdefault(row["priority"], []).append(row)
    rules_to_delete = []
    for rule in ownership["rules"]:
        at_priority = current_by_priority.get(rule["priority"], [])
        if not at_priority:
            continue
        if len(at_priority) != 1 or at_priority[0] != rule:
            raise WorkloadError(
                f"rule priority {rule['priority']} changed or became ambiguous; refusing cleanup"
            )
        rules_to_delete.append(rule)
    routes_to_delete = [
        route for route in ownership["routes"] if identity_key(route) in current_route_keys
    ]
    return rules_to_delete, routes_to_delete


def cleanup(spec: WorkloadSpec, run_dir: Path, experiment_id: str) -> None:
    verify_lease(spec, run_dir, experiment_id)
    evidence = workload_dir(run_dir)
    result: dict[str, Any] = {
        "schema_version": WORKLOAD_RUN_SCHEMA_VERSION,
        "start": clock_anchors(),
        "status": "running",
        "lease_released": False,
        "deletions": [],
    }
    try:
        competitors = process_ids_by_name("nr-uesoftmodem")
        tun_exists = interface_exists(spec)
        result["competing_nrue_pids"] = competitors
        result["interface_exists"] = tun_exists
        if competitors or tun_exists:
            raise WorkloadError(
                f"cleanup requires verified nrUE death and no TUN: "
                f"pids={competitors}, {spec.interface}={tun_exists}"
            )
        baseline = load_json_object(evidence / "network_baseline.json")
        ownership_path = evidence / "network_ownership.json"
        if ownership_path.exists():
            ownership = load_json_object(ownership_path)
        else:
            ownership = {"rules": [], "routes": []}
        before = network_snapshot(spec)
        atomic_write_json(evidence / "network_cleanup_before.json", before)
        rules, routes = validate_cleanup_ownership(baseline, ownership, before)

        for route in routes:
            command = route_delete_command(spec, route)
            command_result = run_command(command, timeout_s=10)
            deletion = {
                "kind": "route",
                "identity": route,
                "command": command,
                "return_code": command_result.returncode,
                "stderr": command_result.stderr,
            }
            result["deletions"].append(deletion)
            if command_result.returncode != 0:
                raise WorkloadError(
                    f"exact route deletion failed ({command_result.returncode})"
                )
        for rule in rules:
            command = rule_delete_command(spec, rule)
            command_result = run_command(command, timeout_s=10)
            deletion = {
                "kind": "rule",
                "identity": rule,
                "command": command,
                "return_code": command_result.returncode,
                "stderr": command_result.stderr,
            }
            result["deletions"].append(deletion)
            if command_result.returncode != 0:
                raise WorkloadError(
                    f"exact rule deletion failed ({command_result.returncode})"
                )

        after = network_snapshot(spec)
        atomic_write_json(evidence / "network_cleanup_after.json", after)
        if after["rules"] != baseline["rules"] or after["routes"] != baseline["routes"]:
            raise WorkloadError("post-cleanup network state does not match the exact baseline")
        result["status"] = "ok" if result["deletions"] else "already_absent"
        result["end"] = clock_anchors()
        atomic_write_json(evidence / "network_cleanup.json", result)
        release_lease(spec, run_dir, experiment_id)
        result["lease_released"] = True
        atomic_write_json(evidence / "network_cleanup.json", result)
        update_run_state(run_dir, cleanup_status=result["status"])
    except Exception as error:
        result["status"] = "failed"
        result["error"] = str(error)
        result["end"] = clock_anchors()
        atomic_write_json(evidence / "network_cleanup.json", result)
        update_run_state(run_dir, cleanup_status="failed", cleanup_error=str(error))
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "preflight", "run", "cleanup"])
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--experiment-id")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        spec = parse_workload_spec(json.loads(args.config_json))
        if args.action == "plan":
            print(json.dumps(command_plan(spec), sort_keys=True))
            return 0
        if args.run_dir is None or args.experiment_id is None:
            raise ValueError("--run-dir and --experiment-id are required for this action")
        run_dir = args.run_dir.expanduser().resolve()
        experiment_id = validate_experiment_id(args.experiment_id)
        if args.action == "preflight":
            preflight(spec, run_dir, experiment_id)
        elif args.action == "run":
            run_workload(spec, run_dir, experiment_id)
        else:
            cleanup(spec, run_dir, experiment_id)
        return 0
    except Exception as error:
        print(f"workload {args.action} failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
