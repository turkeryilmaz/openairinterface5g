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


def parse_bitrate(value: str) -> int:
    """Decimal bit/s, optionally K/M/G or Kbps/Mbps/Gbps; never byte rates."""
    message = "rate must be 1..1000000000000 bit/s, e.g. 500K, 1.1M, 1.1Mbps or 1100000"
    if len(value) > 64:
        raise ValueError(message)
    match = re.fullmatch(r"([0-9]+)(?:\.([0-9]+))?([kKmMgG]?)(?:bps)?", value.strip())
    if match is None:
        raise ValueError(message)
    whole, fraction, unit = match.groups()
    fraction = fraction or ""
    # Integer arithmetic preserves fractional input exactly, without silently
    # rounding a nonintegral bit/s rate or interpreting M as a binary multiple.
    scale = 1000 ** {"": 0, "K": 1, "M": 2, "G": 3}[unit.upper()]
    rate, remainder = divmod(int(whole + fraction) * scale, 10 ** len(fraction))
    if remainder or not 1 <= rate <= 10**12:
        raise ValueError(message)
    return rate


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
    segment_seconds: int = 0  # iperf3: no scheduled end; the capture owns shutdown.
    retry_seconds: float = 5.0

    def metadata(self) -> dict[str, Any]:
        return {**self.__dict__, "datagram_bytes": 1200, "interval_seconds": 1,
                "continuous": self.segment_seconds == 0, "client_report_timeout_seconds": 10,
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
    required = ["--forceflush", "--rcv-timeout", "--snd-timeout"]
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


UDP_RECEIVER_RE = re.compile(r"(?P<jitter>\d+(?:\.\d+)?)\s+ms\s+"
                             r"(?P<lost>\d+)\s*/\s*(?P<total>\d+)\s+\((?P<loss>\d+(?:\.\d+)?)%\)")
STREAM_TAG_RE = re.compile(r"\[(?P<tag>(?:TX|RX)-(?:C|S))\]")


def interval_evidence(text: str, role: Optional[str] = None,
                      direction: Optional[str] = None) -> Optional[dict[str, Any]]:
    match = INTERVAL_RE.search(text)
    if match is None:
        return None
    summary = re.search(r"\b(sender|receiver)\s*$", text)
    tag_match = STREAM_TAG_RE.search(text)
    tag = tag_match["tag"] if tag_match else None
    udp = UDP_RECEIVER_RE.search(text, match.end())
    reporter = summary[1] if summary else ("receiver" if tag and tag.startswith("RX") else
                                          "sender" if tag else "receiver" if udp else
                                          "sender" if role else "interval")
    if tag:
        direction = "ul" if tag in ("TX-C", "RX-S") else "dl"
    elif direction not in ("ul", "dl"):
        # The server's configured direction does not select the client's mode.
        # Untagged live UDP reports identify its negotiated direction instead.
        if role and not summary:
            direction = "ul" if (role == "ue") == (reporter == "sender") else "dl"
        else:
            direction = None
    endpoint = None
    if direction in ("ul", "dl") and reporter in ("sender", "receiver"):
        endpoint = "client" if (direction == "ul") == (reporter == "sender") else "server"
    # On interrupted result exchange iperf3 can print a remote 0/0 receiver
    # summary without having received that endpoint's results. Preserve the raw
    # numbers, but do not turn this placeholder into evidence of zero delivery.
    local_endpoint = "client" if role == "ue" else "server" if role == "gnb" else None
    receiver_available = (not (summary and udp and int(udp["total"]) == 0 and float(match["bytes"]) == 0
                              and (local_endpoint is None or endpoint != local_endpoint))) if reporter == "receiver" else None
    result = {"start_seconds": float(match["start"]), "end_seconds": float(match["end"]),
              "bytes": float(match["bytes"]) * 1024 ** (" KMGT".index(match["byte_unit"] or " ")),
              "bits_per_second": float(match["rate"]) * 1000 ** (" KMGT".index(match["rate_unit"] or " ")),
              "reporter": reporter, "reporter_endpoint": endpoint, "direction": direction, "stream_tag": tag,
              "report_type": "summary" if summary else "interval",
              "receiver_report_available": receiver_available,
              "jitter_ms": float(udp["jitter"]) if udp else None,
              "lost_datagrams": int(udp["lost"]) if udp else None,
              "total_datagrams": int(udp["total"]) if udp else None,
              "loss_percent": float(udp["loss"]) if udp else None,
              "scope": "rounded iperf3 text units; sender reports do not establish delivery"}
    sent = re.match(r"\s+(\d+)\s*(?:sender)?\s*$", text[match.end():])
    result["sent_datagrams"] = int(sent[1]) if sent and reporter == "sender" else None
    return result


class TrafficSupervisor:
    """One ordinary-priority child group, ticked from FlightCapture's loop."""

    def __init__(self, settings: TrafficSettings, writer: Any,
                 address_reader: Callable[[str, str], Optional[str]] = tunnel_address,
                 human_writer: Any = None) -> None:
        self.settings = settings
        self.writer = writer
        self.address_reader = address_reader
        self.human_writer = human_writer
        self.human_last: dict[str, float] = {}
        self.last_interval = 0.0
        self.server_output = False
        self.session_direction: Optional[str] = settings.direction if settings.role == "ue" else None
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
            self._human(f"gap={reason} delivery=unavailable")
            self.last_gap = reason

    def _human(self, text: str) -> None:
        if self.human_writer is not None:
            self.human_writer.write(f"[TRAFFIC] gen={self.generation} {text}\n".encode("ascii")[:512])

    def _human_interval(self, interval: dict[str, Any]) -> None:
        direction = interval["direction"] or "unknown"
        now = time.monotonic()
        if now - self.human_last.get(direction, float("-inf")) < 0.9:
            return
        self.human_last[direction] = now
        detail = "delivery=unavailable"
        if interval["reporter"] == "receiver":
            detail = "delivery=receiver_observed"
            if interval["lost_datagrams"] is not None:
                detail += (f" loss={interval['lost_datagrams']}/{interval['total_datagrams']}"
                           f" ({interval['loss_percent']:g}%) jitter={interval['jitter_ms']:g}ms")
        elif interval["sent_datagrams"] is not None:
            detail += f" sent_datagrams={interval['sent_datagrams']}"
        self._human(f"{direction.upper()} {interval['reporter']}={interval['reporter_endpoint'] or 'unknown'} "
                    f"{interval['start_seconds']:.2f}-{interval['end_seconds']:.2f}s "
                    f"{interval['bits_per_second'] / 1e6:.3f}Mbps {detail}")

    def _command(self) -> list[str]:
        s = self.settings
        command = [s.executable, "--port", str(s.port), "--interval", "1", "--format", "k", "--forceflush",
                   "--snd-timeout", "5000"]
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
        self.last_interval = now
        self.server_output = False
        self.session_direction = self.settings.direction if self.settings.role == "ue" else None
        self.human_last.clear()
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
        if stream == "stdout" and text.startswith("Server output:"):
            self.server_output = True
        if stream == "stdout" and text.startswith("Accepted connection from"):
            self.session_direction = None
        role = ("gnb" if self.settings.role == "ue" else "ue") if self.server_output else self.settings.role
        interval = interval_evidence(text, role, self.session_direction) if not partial else None
        if interval is not None:
            interval["source"] = "retrieved_server_output" if self.server_output else "local_output"
            if not self.server_output and interval["report_type"] == "interval":
                self.last_interval = time.monotonic()
                self.session_direction = interval["direction"]
                self._human_interval(interval)
        self.output_lines += 1
        self._event("traffic_output", stream=stream, text=text, partial=partial,
                    interval=interval)

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
            elif (self.settings.role == "ue" and self.settings.segment_seconds > 0
                  and now - self.started >= self.settings.segment_seconds + 10):
                self._gap("segment_deadline")
                self._stop("segment_deadline", now)
            elif self.settings.role == "ue" and now - (self.last_interval or self.started) >= 10:
                self._gap("client_report_timeout")
                self._stop("client_report_timeout", now)
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
