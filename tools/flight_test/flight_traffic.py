# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Bounded iperf3 children owned by one flight capture attempt.

Traffic failures describe the application path only. This module never receives
a radio process or recovery policy and cannot request radio recovery.
"""

from __future__ import annotations

import fcntl
import ipaddress
import json
import os
import re
import selectors
import signal
import socket
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from flight_health import bounded_command, clock_sample


class TrafficError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrafficSettings:
    role: str
    executable: str
    interface: str
    peer: Optional[str] = None
    bind: Optional[str] = None
    port: int = 5201
    direction: str = "ul"
    rate: int = 1100000
    segment_seconds: int = 30
    retry_seconds: float = 5.0

    def metadata(self) -> dict[str, Any]:
        return {**self.__dict__, "datagram_bytes": 1200, "interval_seconds": 1,
                "control_protocol": "TCP", "payload_protocol": "UDP",
                "binding": "current tunnel IP and SO_BINDTODEVICE on UE",
                "radio_recovery_actions": False}


def validate_settings(args: Any) -> Optional[TrafficSettings]:
    if not args.traffic:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,15}", args.interface):
        raise TrafficError("traffic requires a Linux interface name of at most 15 bytes")
    for name in ("traffic_peer", "traffic_bind"):
        value = getattr(args, name)
        if value is not None:
            try:
                address = ipaddress.ip_address(value)
            except ValueError as exc:
                raise TrafficError(f"--{name.replace('_', '-')} must be an IP address") from exc
            if address.is_unspecified or address.is_multicast:
                raise TrafficError(f"--{name.replace('_', '-')} must be a unicast address")
    if args.role == "ue" and not args.traffic_peer:
        raise TrafficError("UE traffic requires explicit --traffic-peer IP")
    if args.role == "ue" and args.traffic_bind:
        raise TrafficError("--traffic-bind is a gNB server setting; UE uses its current tunnel IP")
    if args.role == "gnb" and args.traffic_peer:
        raise TrafficError("--traffic-peer is a UE client setting")
    if not 1 <= args.traffic_port <= 65535 or not 1 <= args.traffic_rate <= 10**12:
        raise TrafficError("traffic port must be 1..65535 and rate 1..1000000000000 bit/s")
    executable = next((str(path) for path in (Path("/usr/bin/iperf3"), Path("/bin/iperf3"),
                                             Path("/usr/local/bin/iperf3"))
                       if path.is_file() and os.access(path, os.X_OK)), None)
    if executable is None:
        raise TrafficError("traffic requires installed iperf3; no package is installed automatically")
    help_result = bounded_command(executable, ("--help",), 2.0)
    required = ["--forceflush", "--rcv-timeout"]
    if args.role == "gnb":
        required.append("--idle-timeout")
    if args.role == "ue":
        required += ["--bind-dev", "--connect-timeout", "--get-server-output"]
        if args.traffic_direction == "both":
            required.append("--bidir")
    if (help_result.get("status") != "ok" or help_result.get("returncode") != 0
            or help_result.get("output_truncated")
            or any(option not in help_result.get("output", "") for option in required)):
        raise TrafficError("installed iperf3 lacks required traffic options or its bounded help probe failed")
    return TrafficSettings(args.role, executable, args.interface, args.traffic_peer, args.traffic_bind,
                           args.traffic_port, args.traffic_direction, args.traffic_rate)


def tunnel_address(interface: str, peer: str) -> Optional[str]:
    """Read a current address without a command, route change, or packet probe."""
    family = ipaddress.ip_address(peer).version
    try:
        socket.if_nametoindex(interface)
        if family == 4:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as control:
                request = struct.pack("256s", interface.encode("ascii"))
                response = fcntl.ioctl(control.fileno(), 0x8915, request)  # SIOCGIFADDR
            return socket.inet_ntoa(response[20:24])
        with open("/proc/net/if_inet6", encoding="ascii") as source:
            for _ in range(256):
                line = source.readline(1024)
                if not line:
                    break
                fields = line.split()
                peer_address = ipaddress.ip_address(peer)
                scope = "10" if peer_address.is_loopback else "20" if peer_address.is_link_local else "00"
                if (len(fields) == 6 and fields[5] == interface and fields[3] == scope
                        and not int(fields[4], 16) & 0x68):
                    return str(ipaddress.IPv6Address(int(fields[0], 16)))
    except (OSError, ValueError, UnicodeError):
        pass
    return None


INTERVAL_RE = re.compile(r"(?P<start>\d+(?:\.\d+)?)\s*-\s*(?P<end>\d+(?:\.\d+)?)\s+sec\s+"
                         r"(?P<bytes>\d+(?:\.\d+)?)\s+(?P<byte_unit>[KMGT]?)Bytes\s+"
                         r"(?P<rate>\d+(?:\.\d+)?)\s+(?P<rate_unit>[KMGT]?)bits/sec")


def interval_evidence(text: str) -> Optional[dict[str, Any]]:
    match = INTERVAL_RE.search(text)
    if match is None:
        return None
    return {"start_seconds": float(match["start"]), "end_seconds": float(match["end"]),
            "bytes": float(match["bytes"]) * 1024 ** (" KMGT".index(match["byte_unit"] or " ")),
            "bits_per_second": float(match["rate"]) * 1000 ** (" KMGT".index(match["rate_unit"] or " ")),
            "reporter": "receiver" if re.search(r"\breceiver\s*$", text) else
                        "sender" if re.search(r"\bsender\s*$", text) else "interval",
            "scope": "rounded iperf3 text units; raw text retains jitter, packet and loss fields"}


class TrafficSupervisor:
    """One ordinary-priority child group, ticked from FlightCapture's loop."""

    def __init__(self, settings: TrafficSettings, writer: Any,
                 address_reader: Callable[[str, str], Optional[str]] = tunnel_address) -> None:
        self.settings = settings
        self.writer = writer
        self.address_reader = address_reader
        self.selector = selectors.DefaultSelector()
        self.process: Optional[subprocess.Popen[bytes]] = None
        self.last_process_group: Optional[int] = None
        self.buffers: dict[str, bytearray] = {}
        self.generation = 0
        self.started = 0.0
        self.next_launch = 0.0
        self.next_address = 0.0
        self.address: Optional[str] = None
        self.stop_started: Optional[float] = None
        self.stop_reason: Optional[str] = None
        self.killed = False
        self.closed = False
        self.cleanup_failed = False
        self.last_gap: Optional[str] = None
        self.last_returncode: Optional[int] = None
        self.output_lines = 0
        self.output_truncated_bytes = 0
        self.completed_segments = 0
        self.failed_segments = 0

    def _event(self, kind: str, **fields: Any) -> None:
        self.writer.write(json.dumps({"schema_version": 1, "kind": kind, "clock": clock_sample(),
                                     "generation": self.generation, **fields}, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8") + b"\n")

    def _gap(self, reason: str) -> None:
        if self.last_gap != reason:
            self._event("traffic_gap", reason=reason, delivered_traffic="unavailable")
            self.last_gap = reason

    def _command(self) -> list[str]:
        s = self.settings
        command = [s.executable, "--port", str(s.port), "--interval", "1", "--format", "k", "--forceflush"]
        if s.role == "gnb":
            command += ["--server", "--rcv-timeout", "5000", "--idle-timeout", "10"]
            if s.bind:
                command += ["--bind", s.bind]
        else:
            assert self.address is not None and s.peer is not None
            command += ["--client", s.peer, "--udp", "--bitrate", str(s.rate), "--length", "1200",
                        "--time", str(s.segment_seconds), "--connect-timeout", "3000",
                        "--bind", self.address, "--bind-dev", s.interface, "--get-server-output",
                        "--version4" if ipaddress.ip_address(s.peer).version == 4 else "--version6"]
            if s.direction == "dl":
                command += ["--reverse", "--rcv-timeout", "5000"]
            elif s.direction == "both":
                command += ["--bidir", "--rcv-timeout", "5000"]
        return command

    def _launch(self, now: float) -> None:
        command = self._command()
        self.generation += 1
        self._event("traffic_generation_started", role=self.settings.role, command=command,
                    tunnel_ip=self.address, previous_gap=self.last_gap)
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, close_fds=True, start_new_session=True)
        except OSError as exc:
            self.failed_segments += 1
            self._event("traffic_launch_failed", error=exc.__class__.__name__)
            self._gap("launch_failed")
            self.next_launch = now + self.settings.retry_seconds
            return
        self.last_process_group = self.process.pid
        self.started = now
        self.stop_started = None
        self.stop_reason = None
        self.killed = False
        self.last_gap = None
        for stream, name in ((self.process.stdout, "stdout"), (self.process.stderr, "stderr")):
            assert stream is not None
            os.set_blocking(stream.fileno(), False)
            self.selector.register(stream, selectors.EVENT_READ, name)
            self.buffers[name] = bytearray()

    def start(self) -> None:
        if self.settings.role != "gnb":
            return
        self.tick()
        # Fail before launching the radio for immediate server startup errors
        # such as a busy port or a nonlocal configured bind address.
        deadline = time.monotonic() + 0.2
        while time.monotonic() < deadline:
            self._drain()
            if self.process is None or self.process.poll() is not None:
                raise TrafficError("iperf3 server failed at startup; inspect traffic log")
            time.sleep(0.01)

    def _line(self, stream: str, raw: bytes, partial: bool = False) -> None:
        text = raw.decode("utf-8", "replace")
        self.output_lines += 1
        self._event("traffic_output", stream=stream, text=text, partial=partial,
                    interval=interval_evidence(text))

    def _drain(self) -> None:
        # At most 32 KiB per tick: a noisy child cannot starve radio supervision.
        for key, _ in self.selector.select(timeout=0)[:8]:
            try:
                data = os.read(key.fileobj.fileno(), 4096)
            except BlockingIOError:
                continue
            buffer = self.buffers[key.data]
            if not data:
                if buffer:
                    self._line(key.data, bytes(buffer), partial=True)
                    buffer.clear()
                self.selector.unregister(key.fileobj)
                key.fileobj.close()
                continue
            buffer.extend(data)
            while b"\n" in buffer:
                line, _, remaining = buffer.partition(b"\n")
                buffer[:] = remaining
                self._line(key.data, line)
            if len(buffer) > 8192:
                self.output_truncated_bytes += len(buffer) - 8192
                del buffer[8192:]
                self._line(key.data, bytes(buffer), partial=True)
                buffer.clear()
                self._event("traffic_output_truncated", reason="overlong_line")

    def _group_exists(self) -> bool:
        if self.process is None:
            return False
        try:
            os.killpg(self.process.pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _signal(self, signum: int) -> None:
        assert self.process is not None
        try:
            os.killpg(self.process.pid, signum)
        except ProcessLookupError:
            pass
        except PermissionError:
            self.cleanup_failed = True
            self._event("traffic_signal_failed", process_group=self.process.pid, signal=signum,
                        reason="permission_denied")

    def _stop(self, reason: str, now: float) -> None:
        if self.process is not None and self.stop_started is None:
            self.stop_started, self.stop_reason = now, reason
            self._event("traffic_stop_requested", reason=reason)
            self._signal(signal.SIGTERM)

    def tick(self, stopping: bool = False) -> None:
        now = time.monotonic()
        self._drain()
        if self.process is not None:
            returncode = self.process.poll()
            if stopping:
                self._stop("main_worker_stopping", now)
            elif returncode is not None:
                self._stop("segment_finished" if returncode == 0 else "segment_failed", now)
            elif self.settings.role == "ue" and now - self.started >= self.settings.segment_seconds + 10:
                self._gap("segment_deadline")
                self._stop("segment_deadline", now)
            if self.stop_started is not None:
                if now - self.stop_started >= 1 and not self.killed and self._group_exists():
                    self._signal(signal.SIGKILL)
                    self.killed = True
                    self._event("traffic_group_killed", reason=self.stop_reason)
                returncode = self.process.poll()
                if returncode is not None and not self._group_exists():
                    self._finish(returncode, now)
                elif now - self.stop_started >= 3:
                    self.cleanup_failed = True
                    self._gap("owned_group_cleanup_failed")
        if self.settings.role == "ue" and not stopping and now >= self.next_address:
            self.next_address = now + 1
            address = self.address_reader(self.settings.interface, str(self.settings.peer))
            if address != self.address:
                self._event("traffic_address_changed", previous=self.address, current=address)
                self.address = address
                self.next_launch = now
                self._stop("tunnel_address_changed", now)
            if address is None:
                self._gap("tunnel_address_unavailable")
        if self.process is None and not stopping and not self.cleanup_failed and now >= self.next_launch:
            if self.settings.role == "gnb" or self.address is not None:
                self._launch(now)

    def _finish(self, returncode: Optional[int], now: float) -> None:
        assert self.process is not None
        # Bounded final drain; anything left is identified rather than discarded silently.
        for _ in range(8):
            if not self.selector.get_map():
                break
            self._drain()
        for key in list(self.selector.get_map().values()):
            self._event("traffic_output_incomplete", stream=key.data, reason="residual_pipe")
            self.selector.unregister(key.fileobj)
            key.fileobj.close()
        for name, buffer in self.buffers.items():
            if buffer:
                self._line(name, bytes(buffer), partial=True)
        success = returncode == 0 and self.stop_reason == "segment_finished"
        self.completed_segments += int(success)
        self.failed_segments += int(not success)
        self.last_returncode = returncode
        self._event("traffic_generation_finished", returncode=returncode, reason=self.stop_reason,
                    completed=success, forced_kill=self.killed)
        self.process = None
        self.buffers.clear()
        self.next_launch = now if success else now + self.settings.retry_seconds
        self._gap("between_segments" if success else str(self.stop_reason))

    def close(self) -> None:
        if self.closed:
            return
        deadline = time.monotonic() + 3.2
        while self.process is not None and time.monotonic() < deadline:
            self.tick(stopping=True)
            if self.process is not None:
                time.sleep(0.02)
        if self.process is not None:
            self._signal(signal.SIGKILL)
            try:
                self.process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
            self.cleanup_failed = self._group_exists()
            self._finish(self.process.returncode, time.monotonic())
        self.selector.close()
        self.closed = True
        self._event("traffic_stopped", **self.snapshot())

    def snapshot(self) -> dict[str, Any]:
        return {"enabled": True, "generation_count": self.generation, "output_lines": self.output_lines,
                "output_truncated_bytes": self.output_truncated_bytes,
                "completed_segments": self.completed_segments, "failed_segments": self.failed_segments,
                "last_returncode": self.last_returncode, "cleanup_failed": self.cleanup_failed,
                "last_process_group": self.last_process_group,
                "active_pid": self.process.pid if self.process else None,
                "delivered_traffic": "inspect_receiver_reports; absent_reports_are_unavailable"}
