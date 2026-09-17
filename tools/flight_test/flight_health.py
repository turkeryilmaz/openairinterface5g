#!/usr/bin/env python3
"""Bounded, observer-only host observations for OAI flight captures.

This module has no OAI dependency and deliberately never restarts, kills, or
otherwise controls a radio process. Its state machine is separated from Linux
collection so its safety properties can be tested with deterministic inputs.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import os
import selectors
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Optional, Sequence


SCHEMA_VERSION = 1
DEFAULT_COMMAND_PATHS = {
    "chronyc": ("/usr/bin/chronyc", "/usr/sbin/chronyc", "/bin/chronyc"),
    "ip": ("/usr/sbin/ip", "/sbin/ip", "/usr/bin/ip", "/bin/ip"),
    "ping": ("/usr/bin/ping", "/bin/ping", "/usr/sbin/ping", "/sbin/ping"),
}


class HealthState(str, enum.Enum):
    STARTING = "starting"
    ACQUIRING = "acquiring"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    EXITED = "exited"
    UNAVAILABLE = "unavailable"


@dataclasses.dataclass(frozen=True)
class HealthInput:
    """One normalized observation. None means unavailable."""

    monotonic_ns: int
    process_running: Optional[bool]
    interface_present: Optional[bool]
    interface_up: Optional[bool]
    probe_reachable: Optional[bool] = None
    counter_activity: Optional[bool] = None
    capture_healthy: Optional[bool] = True


def _recommendation(state: HealthState) -> dict[str, Any]:
    """Return advice only; the capture supervisor never acts on it."""

    advice = {
        HealthState.STARTING: "wait through the configured monotonic startup grace",
        HealthState.ACQUIRING: "inspect registration and tunnel acquisition logs",
        HealthState.HEALTHY: "continue observation",
        HealthState.DEGRADED: "inspect direct process/interface evidence before any manual recovery",
        HealthState.EXITED: "inspect the captured exit status before a manual relaunch",
        HealthState.UNAVAILABLE: "restore host observation access before interpreting health",
    }[state]
    return {
        "automatic_action": "none",
        "restart_eligible": False,
        "backoff_seconds": 60,
        "restart_budget_remaining": 0,
        "manual_recommendation": advice,
    }


class HealthStateMachine:
    """Apply role-aware grace and hysteresis without inventing PDU outcomes."""

    def __init__(
        self,
        grace_seconds: float = 30.0,
        hysteresis_samples: int = 3,
        role: str = "ue",
    ) -> None:
        if grace_seconds < 0:
            raise ValueError("grace_seconds must be non-negative")
        if hysteresis_samples < 1:
            raise ValueError("hysteresis_samples must be positive")
        if role not in ("gnb", "ue"):
            raise ValueError("role must be gnb or ue")
        self.role = role
        self.grace_ns = int(grace_seconds * 1_000_000_000)
        self.hysteresis_samples = hysteresis_samples
        self.started_ns: Optional[int] = None
        self.state = HealthState.STARTING
        self.ever_interface_present = False
        self.healthy_streak = 0
        self.bad_streak = 0

    def observe(self, sample: HealthInput) -> dict[str, Any]:
        if self.started_ns is None:
            self.started_ns = sample.monotonic_ns
        elapsed_ns = max(0, sample.monotonic_ns - self.started_ns)
        evidence = [
            {
                "layer": "process",
                "value": sample.process_running,
                "direct": sample.process_running is not None,
            },
            {
                "layer": "interface",
                "value": sample.interface_present,
                "link_up": sample.interface_up,
                "direct": sample.interface_present is not None,
                "scope": (
                    "UE tunnel acquisition evidence only; it does not establish a PDU session"
                    if self.role == "ue"
                    else "gNB host-interface observation only; a missing UE tunnel is not acquisition evidence"
                ),
            },
            {
                "layer": "probe",
                "value": sample.probe_reachable,
                "direct": sample.probe_reachable is not None,
                "scope": "ICMP reachability only; it does not establish a PDU session",
            },
            {
                "layer": "interface_counters",
                "value": sample.counter_activity,
                "direct": sample.counter_activity is not None,
                "scope": "counter movement is host-interface evidence only",
            },
            {
                "layer": "capture",
                "value": sample.capture_healthy,
                "direct": sample.capture_healthy is not None,
            },
        ]

        if sample.interface_present is True:
            self.ever_interface_present = True

        if sample.process_running is False:
            self.state = HealthState.EXITED
            self.healthy_streak = 0
            self.bad_streak = 0
            return self._result(elapsed_ns, evidence, "direct process exit", "direct")

        if (
            sample.process_running is None
            and sample.interface_present is None
            and sample.interface_up is None
        ):
            self.state = HealthState.UNAVAILABLE
            self.healthy_streak = 0
            self.bad_streak = 0
            return self._result(
                elapsed_ns,
                evidence,
                "process and interface observations unavailable",
                "directly_unavailable",
            )

        if elapsed_ns < self.grace_ns:
            self.state = HealthState.STARTING
            self.healthy_streak = 0
            self.bad_streak = 0
            return self._result(
                elapsed_ns,
                evidence,
                "within monotonic startup grace; absent tunnel is not wedge evidence",
                "inferred",
            )

        # Only a UE tunnel that never appeared is acquisition evidence. A gNB
        # normally has no UE tunnel, so absence cannot be a gNB lifecycle claim.
        if self.role == "ue" and sample.interface_present is False and not self.ever_interface_present:
            self.state = HealthState.ACQUIRING
            self.healthy_streak = 0
            self.bad_streak = 0
            return self._result(
                elapsed_ns,
                evidence,
                "tunnel has not appeared; remaining in acquisition",
                "inferred",
            )

        if self.role == "ue" and sample.interface_present is None:
            self.state = HealthState.UNAVAILABLE
            self.healthy_streak = 0
            self.bad_streak = 0
            return self._result(
                elapsed_ns,
                evidence,
                "interface observation unavailable after startup grace",
                "directly_unavailable",
            )

        if self.role == "ue":
            direct_bad = sample.interface_present is False or sample.interface_up is False or sample.process_running is False
            direct_good = (
                sample.process_running is True
                and sample.interface_present is True
                and sample.interface_up is not False
            )
        else:
            direct_bad = sample.process_running is False or (
                sample.interface_present is True and sample.interface_up is False
            )
            direct_good = sample.process_running is True and not (
                sample.interface_present is True and sample.interface_up is False
            )
        if direct_bad:
            self.bad_streak += 1
            self.healthy_streak = 0
            if self.bad_streak >= self.hysteresis_samples:
                self.state = HealthState.DEGRADED
                reason = "three consecutive direct process/interface concerns"
            else:
                reason = "direct concern pending hysteresis"
            return self._result(elapsed_ns, evidence, reason, "inferred")

        if direct_good:
            self.healthy_streak += 1
            self.bad_streak = 0
            if self.healthy_streak >= self.hysteresis_samples:
                self.state = HealthState.HEALTHY
                reason = (
                    "three consecutive direct process/interface observations"
                    if self.role == "ue"
                    else "three consecutive direct gNB process observations"
                )
            else:
                self.state = HealthState.ACQUIRING
                reason = "direct observations pending healthy hysteresis"
            return self._result(elapsed_ns, evidence, reason, "inferred")

        self.state = HealthState.UNAVAILABLE
        self.healthy_streak = 0
        self.bad_streak = 0
        return self._result(
            elapsed_ns,
            evidence,
            "insufficient direct process/interface observations",
            "directly_unavailable",
        )

    def _result(
        self,
        elapsed_ns: int,
        evidence: list[dict[str, Any]],
        reason: str,
        state_basis: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "role": self.role,
            "state": self.state.value,
            "state_basis": state_basis,
            "reason": reason,
            "startup_elapsed_ns": elapsed_ns,
            "healthy_streak": self.healthy_streak,
            "bad_streak": self.bad_streak,
            "hysteresis_samples": self.hysteresis_samples,
            "evidence": evidence,
            "recovery": _recommendation(self.state),
        }


def clock_sample() -> dict[str, Any]:
    """Read paired clocks and state the sampling uncertainty explicitly."""

    before = time.monotonic_ns()
    wall_ns = time.time_ns()
    after = time.monotonic_ns()
    raw_clock = getattr(time, "CLOCK_MONOTONIC_RAW", None)
    try:
        raw_ns: Optional[int] = time.clock_gettime_ns(raw_clock) if raw_clock is not None else None
        raw_state = "available" if raw_ns is not None else "unavailable"
    except (AttributeError, OSError, ValueError):
        raw_ns = None
        raw_state = "unavailable"
    return {
        "utc_wall_ns": wall_ns,
        "monotonic_ns": (before + after) // 2,
        "monotonic_raw_ns": raw_ns,
        "monotonic_raw_state": raw_state,
        "paired_clock_uncertainty_ns": max(0, (after - before) // 2),
    }


def _read_text(path: Path, limit: int = 8192) -> Optional[str]:
    try:
        with path.open("rb") as source:
            value = source.read(limit + 1)
    except (FileNotFoundError, PermissionError, OSError):
        return None
    if len(value) > limit:
        return None
    return value.decode("utf-8", "replace").strip()


def _read_int(path: Path) -> Optional[int]:
    value = _read_text(path, 128)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _safe_executable(candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if os.path.isabs(candidate) and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def bounded_command(
    executable: Optional[str],
    args: Sequence[str],
    timeout_seconds: float,
    max_output_bytes: int = 16384,
    capture_output: bool = True,
) -> dict[str, Any]:
    """Run one fixed absolute executable without a shell or unbounded pipes."""

    if executable is None:
        return {"status": "unavailable", "reason": "fixed executable path absent"}
    if not os.path.isabs(executable):
        return {"status": "invalid", "reason": "executable path was not absolute"}
    try:
        process = subprocess.Popen(
            [executable, *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if capture_output else subprocess.DEVNULL,
            stderr=subprocess.PIPE if capture_output else subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
    except OSError as exc:
        return {"status": "unavailable", "reason": exc.__class__.__name__}

    if not capture_output:
        try:
            returncode = process.wait(timeout=timeout_seconds)
            return {"status": "ok", "returncode": returncode}
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            return {"status": "timeout"}

    output = bytearray()
    truncated = False
    selector = selectors.DefaultSelector()
    assert process.stdout is not None
    assert process.stderr is not None
    for stream, name in ((process.stdout, "stdout"), (process.stderr, "stderr")):
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, name)
    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    while selector.get_map() or process.poll() is None:
        if not timed_out and time.monotonic() >= deadline:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for key, _ in selector.select(timeout=0.05):
            try:
                chunk = os.read(key.fileobj.fileno(), 4096)
            except BlockingIOError:
                continue
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            remaining = max_output_bytes - len(output)
            if remaining > 0:
                output.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated = True
    selector.close()
    process.stdout.close()
    process.stderr.close()
    returncode = process.wait()
    if timed_out:
        return {"status": "timeout", "returncode": returncode, "output_truncated": truncated}
    return {
        "status": "ok",
        "returncode": returncode,
        "output": output.decode("utf-8", "replace"),
        "output_truncated": truncated,
    }


class SystemHealthCollector:
    """Linux-first bounded collector whose unavailable fields remain unavailable."""

    def __init__(
        self,
        interface: str,
        role: str = "ue",
        core_ip: Optional[str] = None,
        probe_ping: bool = False,
        gpsd: Optional[tuple[str, int]] = None,
        proc_root: str | Path = "/proc",
        sys_root: str | Path = "/sys",
        command_paths: Optional[dict[str, Sequence[str]]] = None,
    ) -> None:
        if role not in ("gnb", "ue"):
            raise ValueError("role must be gnb or ue")
        self.interface = interface
        self.role = role
        self.core_ip = core_ip
        self.probe_ping = probe_ping
        self.gpsd = gpsd
        self.proc_root = Path(proc_root)
        self.sys_root = Path(sys_root)
        self.command_paths = command_paths if command_paths is not None else DEFAULT_COMMAND_PATHS
        self.previous_cpu: Optional[tuple[int, int]] = None
        self.previous_counters: Optional[tuple[int, int]] = None
        self.last_chrony_ns = 0
        self.last_route_ns = 0
        self.chrony: dict[str, Any] = {"status": "not_sampled"}
        self.route: dict[str, Any] = {"status": "not_sampled"}

    def sample(self, pid: int, process_running: Optional[bool]) -> dict[str, Any]:
        now = clock_sample()
        now_ns = now["monotonic_ns"]
        interface = self._interface()
        counter_activity = self._counter_activity(interface)
        observation = {
            "schema_version": SCHEMA_VERSION,
            "kind": "host_observation",
            "role": self.role,
            "clock": now,
            "boot_id": self._boot_id(),
            "cpu": self._cpu(),
            "load": self._load(),
            "memory": self._memory(),
            "thermal": self._thermal(),
            "process": self._process(pid, process_running),
            "interface": interface,
            "route": self._route(now_ns),
            "chrony": self._chrony(now_ns),
            "gpsd": self._gpsd(),
            "probe": self._probe(),
            "limits": {
                "thermal_zone_limit": 32,
                "command_output_limit_bytes": 16384,
                "probe_timeout_seconds": 1,
            },
        }
        observation["interface"]["counter_activity"] = counter_activity
        observation["interface"]["scope"] = (
            "UE tunnel host-interface counters only; no packet payloads captured"
            if self.role == "ue"
            else "gNB host-interface counters only; a missing UE tunnel is not acquisition evidence"
        )
        return observation

    def normalized_input(
        self,
        observation: dict[str, Any],
        process_running: Optional[bool],
        capture_healthy: Optional[bool],
    ) -> HealthInput:
        interface = observation["interface"]
        probe = observation["probe"]
        return HealthInput(
            monotonic_ns=observation["clock"]["monotonic_ns"],
            process_running=process_running,
            interface_present=interface.get("present"),
            interface_up=interface.get("up"),
            probe_reachable=probe.get("reachable"),
            counter_activity=interface.get("counter_activity"),
            capture_healthy=capture_healthy,
        )

    def _boot_id(self) -> dict[str, Optional[str]]:
        value = _read_text(self.proc_root / "sys/kernel/random/boot_id", 128)
        return {"state": "available" if value else "unavailable", "value": value}

    def _cpu(self) -> dict[str, Any]:
        value = _read_text(self.proc_root / "stat", 8192)
        if value is None:
            return {"state": "unavailable", "percent": None}
        line = next((line for line in value.splitlines() if line.startswith("cpu ")), None)
        if line is None:
            return {"state": "invalid", "percent": None}
        try:
            fields = [int(field) for field in line.split()[1:]]
            total = sum(fields)
            idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
        except (ValueError, IndexError):
            return {"state": "invalid", "percent": None}
        percent: Optional[float] = None
        if self.previous_cpu is not None:
            old_total, old_idle = self.previous_cpu
            total_delta = total - old_total
            idle_delta = idle - old_idle
            if total_delta > 0:
                percent = round(100.0 * (1.0 - idle_delta / total_delta), 3)
        self.previous_cpu = (total, idle)
        return {"state": "available", "percent": percent}

    def _load(self) -> dict[str, Any]:
        value = _read_text(self.proc_root / "loadavg", 512)
        if value is None:
            return {"state": "unavailable", "one": None, "five": None, "fifteen": None}
        try:
            one, five, fifteen = (float(field) for field in value.split()[:3])
        except (ValueError, IndexError):
            return {"state": "invalid", "one": None, "five": None, "fifteen": None}
        return {"state": "available", "one": one, "five": five, "fifteen": fifteen}

    def _memory(self) -> dict[str, Any]:
        value = _read_text(self.proc_root / "meminfo", 8192)
        if value is None:
            return {"state": "unavailable", "total_bytes": None, "available_bytes": None}
        fields: dict[str, int] = {}
        for line in value.splitlines():
            key, separator, rest = line.partition(":")
            if not separator:
                continue
            part = rest.strip().split()
            if len(part) >= 2 and part[1] == "kB":
                try:
                    fields[key] = int(part[0]) * 1024
                except ValueError:
                    continue
        return {
            "state": "available" if fields else "invalid",
            "total_bytes": fields.get("MemTotal"),
            "available_bytes": fields.get("MemAvailable"),
        }

    def _thermal(self) -> dict[str, Any]:
        root = self.sys_root / "class/thermal"
        try:
            zones = sorted(root.glob("thermal_zone*"))[:32]
        except OSError:
            zones = []
        readings = []
        for zone in zones:
            readings.append(
                {
                    "name": zone.name,
                    "type": _read_text(zone / "type", 256),
                    "millidegrees_c": _read_int(zone / "temp"),
                }
            )
        return {"state": "available" if zones else "unavailable", "readings": readings}

    def _process(self, pid: int, running: Optional[bool]) -> dict[str, Any]:
        result: dict[str, Any] = {"running": running, "state": "available" if running is not None else "unavailable"}
        if running is not True:
            return result
        value = _read_text(self.proc_root / str(pid) / "stat", 8192)
        if value is None or ")" not in value:
            result.update({"metrics_state": "unavailable", "rss_bytes": None})
            return result
        fields = value.rsplit(")", 1)[1].split()
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            result.update(
                {
                    "metrics_state": "available",
                    "kernel_state": fields[0],
                    "utime_ticks": int(fields[11]),
                    "stime_ticks": int(fields[12]),
                    "rss_bytes": int(fields[21]) * page_size,
                }
            )
        except (IndexError, ValueError, OSError):
            result.update({"metrics_state": "invalid", "rss_bytes": None})
        return result

    def _interface(self) -> dict[str, Any]:
        base = self.sys_root / "class/net" / self.interface
        if not base.exists():
            return {
                "state": "absent",
                "present": False,
                "up": None,
                "rx_bytes": None,
                "tx_bytes": None,
            }
        operstate = _read_text(base / "operstate", 64)
        rx_bytes = _read_int(base / "statistics/rx_bytes")
        tx_bytes = _read_int(base / "statistics/tx_bytes")
        rx_packets = _read_int(base / "statistics/rx_packets")
        tx_packets = _read_int(base / "statistics/tx_packets")
        return {
            "state": "available",
            "present": True,
            "up": operstate in ("up", "unknown") if operstate is not None else None,
            "operstate": operstate,
            "rx_bytes": rx_bytes,
            "tx_bytes": tx_bytes,
            "rx_packets": rx_packets,
            "tx_packets": tx_packets,
        }

    def _counter_activity(self, interface: dict[str, Any]) -> Optional[bool]:
        values = (interface.get("rx_bytes"), interface.get("tx_bytes"))
        if None in values:
            self.previous_counters = None
            return None
        counters = (int(values[0]), int(values[1]))
        if self.previous_counters is None:
            self.previous_counters = counters
            return None
        active = counters != self.previous_counters
        self.previous_counters = counters
        return active

    def _route(self, now_ns: int) -> dict[str, Any]:
        if self.core_ip is None:
            return {"status": "unconfigured"}
        if now_ns - self.last_route_ns < 10_000_000_000:
            return self.route
        self.last_route_ns = now_ns
        executable = _safe_executable(self.command_paths.get("ip", ()))
        result = bounded_command(executable, ("-j", "route", "get", self.core_ip), 2.0)
        if result.get("status") != "ok" or result.get("returncode") != 0:
            self.route = {"status": result.get("status", "unavailable"), "reachable": None}
            return self.route
        try:
            parsed = json.loads(result.get("output", ""))
            entry = parsed[0] if isinstance(parsed, list) and parsed else {}
            self.route = {
                "status": "available",
                "reachable": True,
                "device": entry.get("dev") if isinstance(entry, dict) else None,
            }
        except (json.JSONDecodeError, TypeError):
            self.route = {"status": "invalid", "reachable": None}
        return self.route

    def _chrony(self, now_ns: int) -> dict[str, Any]:
        if now_ns - self.last_chrony_ns < 10_000_000_000:
            return self.chrony
        self.last_chrony_ns = now_ns
        executable = _safe_executable(self.command_paths.get("chronyc", ()))
        tracking = bounded_command(executable, ("tracking",), 2.0)
        sources = bounded_command(executable, ("sources",), 2.0)
        self.chrony = {
            "status": "available" if tracking.get("status") == "ok" else tracking.get("status", "unavailable"),
            "tracking": tracking,
            "sources": sources,
            "cadence_seconds": 10,
        }
        return self.chrony

    def _gpsd(self) -> dict[str, Any]:
        if self.gpsd is None:
            return {"status": "unconfigured"}
        host, port = self.gpsd
        receipt = clock_sample()
        try:
            with socket.create_connection((host, port), timeout=1.0) as connection:
                connection.settimeout(1.0)
                connection.sendall(b'?WATCH={"enable":true,"json":true};\n')
                payload = connection.recv(8192)
        except (OSError, ValueError) as exc:
            return {
                "status": "unavailable",
                "reason": exc.__class__.__name__,
                "receipt_utc_wall_ns": receipt["utc_wall_ns"],
                "receipt_monotonic_ns": receipt["monotonic_ns"],
            }
        for line in payload.splitlines():
            try:
                message = json.loads(line.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                continue
            message_class = message.get("class")
            if message_class not in ("TPV", "SKY"):
                continue
            return {
                "status": "available",
                "class": message_class,
                "gnss_utc": message.get("time"),
                "fix_mode": message.get("mode"),
                "accuracy": {
                    key: message.get(key)
                    for key in ("ept", "epx", "epy", "epv", "eps", "epd", "hdop", "vdop", "pdop")
                    if key in message
                },
                "receipt_utc_wall_ns": receipt["utc_wall_ns"],
                "receipt_monotonic_ns": receipt["monotonic_ns"],
            }
        return {
            "status": "unavailable",
            "reason": "no_tpv_or_sky_in_bounded_response",
            "receipt_utc_wall_ns": receipt["utc_wall_ns"],
            "receipt_monotonic_ns": receipt["monotonic_ns"],
        }

    def _probe(self) -> dict[str, Any]:
        if not self.probe_ping:
            return {"status": "unconfigured", "reachable": None}
        executable = _safe_executable(self.command_paths.get("ping", ()))
        started = time.monotonic_ns()
        result = bounded_command(
            executable,
            ("-n", "-c", "1", "-W", "1", str(self.core_ip)),
            1.5,
            capture_output=False,
        )
        return {
            "status": result.get("status", "unavailable"),
            "reachable": result.get("returncode") == 0 if result.get("status") == "ok" else None,
            "duration_ns": time.monotonic_ns() - started,
            "target_scope": "operator-supplied remote core endpoint required; ICMP does not prove PDU service",
        }
