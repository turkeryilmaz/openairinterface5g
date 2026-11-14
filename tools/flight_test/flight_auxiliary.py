#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Optional bounded local core logs and packets for a flight attempt.

Only children created here are signalled. No core/radio control, recovery, SSH,
privilege escalation, or traffic generation is available. Packet bytes are
sensitive and unredacted; an explicit interface, filter, and snaplen scope them.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import selectors
import signal
import socket
import stat
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from flight_health import bounded_command, clock_sample

COMMAND_PATHS = {
    "docker": ("/usr/bin/docker", "/bin/docker"),
    "journalctl": ("/usr/bin/journalctl", "/bin/journalctl"),
    "systemctl": ("/usr/bin/systemctl", "/bin/systemctl"),
    "dumpcap": ("/usr/bin/dumpcap", "/usr/sbin/dumpcap", "/bin/dumpcap"),
}
MAX_RAW_LINE = 65536
DOCKER_TIME = re.compile(rb"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,9})?(?:Z|[+-]\d\d:\d\d)) ")
# Short key labels seen in core logs supplement OutputSanitizer's OAI markers.
CORE_SECRET = re.compile(rb"(?i)(?:^|[^a-z0-9_])(?:k|op|opc|ki|xres(?:_star)?|res(?:_star)?|"
                         rb"kasme|ck|ik|ak|sqn|rand|autn|msisdn|guti|5g[-_]s[-_]tmsi|authorization)[\"']?\s*[:=]")
JOURNAL_FIELDS = "MESSAGE,__REALTIME_TIMESTAMP"
DOCKER_IDENTITY_FORMAT = ('{"id":{{json .Id}},"image":{{json .Image}},'
                          '"restart_count":{{json .RestartCount}},'
                          '"started_at":{{json .State.StartedAt}},"state":{{json .State.Status}}}')
SERVICE_PROPERTIES = ("Id", "LoadState", "ActiveState", "SubState", "MainPID",
                      "ExecMainStartTimestampMonotonic", "NRestarts", "InvocationID")


class AuxiliaryError(RuntimeError):
    pass


@dataclass(frozen=True)
class CoreSource:
    kind: str
    name: str


@dataclass(frozen=True)
class AuxiliarySettings:
    core_logs: tuple[CoreSource, ...] = ()
    pcap_interface: Optional[str] = None
    pcap_filter: Optional[str] = None
    pcap_max_bytes: int = 64 * 1024 * 1024
    pcap_files: int = 8
    pcap_snaplen: int = 256
    pcap_duration: float = 3600.0

    @property
    def pcap_file_kb(self) -> int:
        return self.pcap_max_bytes // self.pcap_files // 1000

    def metadata(self) -> dict[str, Any]:
        return {"core_logs": [{"kind": s.kind, "name": s.name} for s in self.core_logs],
                **{k: getattr(self, k) for k in ("pcap_interface", "pcap_filter", "pcap_max_bytes",
                                                "pcap_files", "pcap_snaplen", "pcap_duration")},
                "pcap_file_threshold_bytes": self.pcap_file_kb * 1000,
                "pcap_retention": "sequential_files_stop_at_quota_no_overwrite",
                "pcap_byte_limit_scope": "nominal thresholds; packet/block/final-statistics overhead may overshoot",
                "pcap_content": "sensitive_unredacted_packet_bytes_with_explicit_filter_and_snaplen",
                "identity_sampling": "start_and_end_only; intermediate_restarts_may_be_unobserved",
                "automatic_recovery": False, "radio_or_core_control": False}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("optional observer-only core logs and packet capture")
    group.add_argument("--core-logs", default="", metavar="docker:NAME,journal:UNIT",
                       help="selected local containers/services; never starts or restarts them")
    group.add_argument("--pcap-interface", help="explicit local Linux interface for optional dumpcap")
    group.add_argument("--pcap-filter", help="explicit BPF filter; required with --pcap-interface")
    group.add_argument("--pcap-max-bytes", type=int, default=64 * 1024 * 1024,
                       help="total nominal byte quota; packet/block boundary overhead is reported")
    group.add_argument("--pcap-files", type=int, default=8, help="1..32 sequential files, without overwriting")
    group.add_argument("--pcap-snaplen", type=int, default=256, help="64..65535 bytes per packet")
    group.add_argument("--pcap-duration", type=float, default=3600.0, help="1..86400 seconds maximum")


def validate_settings(args: Any) -> Optional[AuxiliarySettings]:
    raw_sources = getattr(args, "core_logs", "") or ""
    sources: list[CoreSource] = []
    if len(raw_sources) > 1024:
        raise AuxiliaryError("--core-logs is too long")
    for value in raw_sources.split(",") if raw_sources else ():
        kind, separator, name = value.strip().partition(":")
        pattern = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}" if kind == "docker" else r"[A-Za-z0-9][A-Za-z0-9_.@:-]{0,127}"
        if not separator or kind not in ("docker", "journal") or re.fullmatch(pattern, name) is None:
            raise AuxiliaryError("--core-logs requires docker:NAME or journal:UNIT without options or paths")
        if CoreSource(kind, name) in sources:
            raise AuxiliaryError("--core-logs contains a duplicate source")
        sources.append(CoreSource(kind, name))
    if len(sources) > 6:
        raise AuxiliaryError("--core-logs allows at most six sources")
    interface = getattr(args, "pcap_interface", None)
    capture_filter = getattr(args, "pcap_filter", None)
    if interface is not None and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:-]{0,14}", interface) is None:
        raise AuxiliaryError("--pcap-interface must be a Linux name, not an option, pipe or remote source")
    if interface and (not capture_filter or len(capture_filter) > 2048 or capture_filter.startswith("-")
                      or any(ord(c) < 32 or ord(c) > 126 for c in capture_filter)):
        raise AuxiliaryError("--pcap-interface requires an explicit printable BPF --pcap-filter, at most 2048 bytes")
    if capture_filter is not None and not interface:
        raise AuxiliaryError("--pcap-filter requires --pcap-interface")
    maximum = getattr(args, "pcap_max_bytes", 64 * 1024 * 1024)
    files = getattr(args, "pcap_files", 8)
    snaplen = getattr(args, "pcap_snaplen", 256)
    duration = getattr(args, "pcap_duration", 3600.0)
    if not 1 <= files <= 32 or not files * 65536 <= maximum <= 16 * 1024**3:
        raise AuxiliaryError("pcap needs 1..32 files and a quota of 64 KiB per file through 16 GiB")
    if not 64 <= snaplen <= 65535 or not 1 <= duration <= 86400:
        raise AuxiliaryError("pcap snaplen must be 64..65535 and duration 1..86400 seconds")
    if not sources and not interface:
        return None
    return AuxiliarySettings(tuple(sources), interface, capture_filter, maximum, files, snaplen, duration)


class _RecordWriter:
    """Only sanitized complete lines receive JSON metadata and are persisted."""
    def __init__(self, writer: Any, run_id: str, child: "_Child", stream: str) -> None:
        self.writer, self.run_id, self.child, self.stream = writer, run_id, child, stream
        self.health = writer.health
        self.context: dict[str, Any] = {}

    def write(self, data: bytes) -> Any:
        record = {"schema_version": 1,
                  "kind": "auxiliary_core_output" if self.child.kind != "pcap" else "auxiliary_capture_output",
                  "run_id": self.run_id, "host": socket.gethostname(), "clock": clock_sample(),
                  "source_kind": self.child.kind, "source_name": self.child.name, "stream": self.stream,
                  "text": data.rstrip(b"\n").decode("utf-8", "replace"), **self.context}
        return self.writer.write(json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n")


@dataclass
class _Child:
    key: str
    kind: str
    name: str
    process: Optional[subprocess.Popen[bytes]] = None
    status: str = "not_started"
    identity_start: dict[str, Any] = field(default_factory=dict)
    identity_end: dict[str, Any] = field(default_factory=dict)
    streams: dict[str, dict[str, Any]] = field(default_factory=dict)
    started: float = 0.0
    stop_started: Optional[float] = None
    stop_reason: Optional[str] = None
    returncode: Optional[int] = None
    forced_kill: bool = False
    cleanup_failed: bool = False
    output_incomplete: bool = False
    raw_bytes: int = 0
    discarded_bytes: int = 0
    invalid_records: int = 0
    source_records: int = 0
    secret_guard_lines: int = 0
    last_process_group: Optional[int] = None


class AuxiliarySupervisor:
    """No threads: bounded start/close and nonblocking, bounded per-tick I/O."""
    def __init__(self, settings: AuxiliarySettings, directory: Path, run_id: str,
                 event_writer: Any, writer_factory: Callable[[str], Any],
                 sanitizer_factory: Callable[[Any], Any], directory_factory: Callable[..., None],
                 *, storage_allows_write: Optional[Callable[[], bool]] = None,
                 command_paths: Optional[dict[str, Sequence[str]]] = None) -> None:
        if not run_id or len(run_id) > 128 or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", run_id) is None:
            raise AuxiliaryError("run_id must be 1..128 ASCII identifier characters")
        self.settings, self.directory, self.run_id = settings, Path(directory), run_id
        self.event_writer, self.writer_factory = event_writer, writer_factory
        self.sanitizer_factory, self.directory_factory = sanitizer_factory, directory_factory
        self.storage_allows_write = storage_allows_write or (lambda: True)
        self.command_paths = COMMAND_PATHS if command_paths is None else command_paths
        self.selector = selectors.DefaultSelector()
        self.children = [_Child(f"core-{i:02d}", s.kind, s.name) for i, s in enumerate(settings.core_logs)]
        if settings.pcap_interface:
            self.children.append(_Child("pcap", "pcap", settings.pcap_interface))
        self.started = self.closed = False
        self.read_cursor = 0
        self.pcap_directory: Optional[Path] = None
        self.pcap_directory_identity: Optional[tuple[int, int]] = None
        self.pcap_output_owner: Optional[tuple[int, int]] = None
        self.pcap_directory_handoff = "not_created"
        self.pcap_files: list[dict[str, Any]] = []
        self.pcap_total_bytes = 0
        self.pcap_quota_reached = self.pcap_scan_failed = False
        self.next_pcap_scan = 0.0
        self.pcap_stats: dict[str, Optional[int]] = {"captured_packets": None, "received_packets": None, "dropped_packets": None}
        self.since_utc = self.since_journal = ""

    def _event(self, kind: str, child: Optional[_Child] = None, **fields: Any) -> None:
        event = {"schema_version": 1, "kind": kind, "run_id": self.run_id,
                 "host": socket.gethostname(), "clock": clock_sample(), **fields}
        if child:
            event.update(source_kind=child.kind, source_name=child.name, collector=child.key)
        self.event_writer.write(json.dumps(event, sort_keys=True, separators=(",", ":")).encode() + b"\n")

    def _executable(self, name: str) -> Optional[str]:
        return next((p for p in self.command_paths.get(name, ())
                     if os.path.isabs(p) and os.path.isfile(p) and os.access(p, os.X_OK)), None)

    def _identity(self, child: _Child) -> dict[str, Any]:
        executable = self._executable("docker" if child.kind == "docker" else "systemctl")
        args = (["inspect", "--type", "container", "--format", DOCKER_IDENTITY_FORMAT, "--", child.name]
                if child.kind == "docker" else
                ["show", "--no-pager", "--property=" + ",".join(SERVICE_PROPERTIES), "--", child.name])
        result = bounded_command(executable, args, 2.0, max_output_bytes=4096)
        if result.get("status") != "ok" or result.get("returncode") != 0 or result.get("output_truncated"):
            return {"status": "unavailable", "reason": "identity_query_failed_or_tool_missing",
                    "query_status": result.get("status"), "returncode": result.get("returncode")}
        try:
            if child.kind == "docker":
                raw = json.loads(result["output"])
                if (not isinstance(raw, dict) or re.fullmatch(r"[0-9a-f]{12,64}", str(raw.get("id", ""))) is None
                        or re.fullmatch(r"(?:sha256:)?[0-9a-f]{12,64}", str(raw.get("image", ""))) is None
                        or type(raw.get("restart_count")) is not int or raw["restart_count"] < 0
                        or raw.get("state") not in ("created", "running", "paused", "restarting", "removing", "exited", "dead")
                        or not isinstance(raw.get("started_at"), str) or len(raw["started_at"]) > 64):
                    raise ValueError("invalid restricted identity")
                return {"status": "available", **{k: raw[k] for k in ("id", "image", "restart_count", "started_at", "state")}}
            raw = dict(line.split("=", 1) for line in result["output"].splitlines() if "=" in line)
            identity = {"status": "available"}
            for key in SERVICE_PROPERTIES:
                value = raw.get(key)
                if value is None or len(value) > 128 or re.fullmatch(r"[A-Za-z0-9_.@:-]*", value) is None:
                    raise ValueError("invalid restricted service identity")
                identity[key] = value
            if identity["LoadState"] != "loaded":
                raise ValueError("unit unavailable")
            return identity
        except (ValueError, TypeError, KeyError):
            return {"status": "unavailable", "reason": "invalid_restricted_identity_output"}

    def _command(self, child: _Child, executable: str) -> list[str]:
        if child.kind == "docker":
            return [executable, "logs", "--timestamps", "--since", self.since_utc, "--follow", "--",
                    child.identity_start.get("id", child.name)]
        if child.kind == "journal":
            return [executable, "--no-pager", "--output=json", "--output-fields=" + JOURNAL_FIELDS,
                    "--since", self.since_journal, "--unit=" + child.name, "--follow"]
        assert self.pcap_directory is not None
        s = self.settings
        return [executable, "-i", child.name, "-f", str(s.pcap_filter), "-s", str(s.pcap_snaplen), "-p", "-q",
                "-b", f"filesize:{s.pcap_file_kb}", "-a", f"files:{s.pcap_files}",
                "-a", f"duration:{s.pcap_duration:g}", "-w", "packets.pcapng"]

    def start(self) -> None:
        if self.started or self.closed:
            return
        self.started = True
        wall = datetime.datetime.now(datetime.timezone.utc)
        self.since_utc = wall.isoformat(timespec="microseconds").replace("+00:00", "Z")
        self.since_journal = wall.strftime("%Y-%m-%d %H:%M:%S.%f UTC")
        self._event("auxiliary_started", settings=self.settings.metadata(), since_utc=self.since_utc,
                    journal_since=self.since_journal,
                    clock_alignment="host receipt clocks only; cross-host synchronization not established")
        for child in self.children:
            executable = self._executable("dumpcap" if child.kind == "pcap" else
                                          "journalctl" if child.kind == "journal" else "docker")
            if executable is None:
                child.status = "unavailable"
                self._event("auxiliary_unavailable", child, reason="fixed_executable_missing")
                continue
            try:
                if child.kind == "pcap":
                    if not self.storage_allows_write():
                        child.status = "unavailable"
                        self._event("auxiliary_unavailable", child, reason="storage_reserve")
                        continue
                    self.pcap_directory = self.directory / ("pcap-" + uuid.uuid4().hex)
                    self.directory_factory(self.pcap_directory)
                    self._prepare_packet_directory()
                else:
                    child.identity_start = self._identity(child)
                    self._event("auxiliary_identity", child, phase="start", identity=child.identity_start)
                for stream in ("stdout", "stderr"):
                    writer = self.writer_factory("auxiliary-" + child.key + "-" + stream)
                    adapter = _RecordWriter(writer, self.run_id, child, stream)
                    child.streams[stream] = {"writer": writer, "adapter": adapter,
                                            "sanitizer": self.sanitizer_factory(adapter),
                                            "buffer": bytearray(), "discarding": False, "eof": False}
                if child.kind != "pcap":
                    # Docker preserves application stdout/stderr separately.
                    # One sensitive block can span both pipes; share its guard.
                    child.streams["stderr"]["sanitizer"].redactor = child.streams["stdout"]["sanitizer"].redactor
                command = self._command(child, executable)
                child.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                                 stderr=subprocess.PIPE, close_fds=True, start_new_session=True, umask=0o077,
                                                 cwd=self.pcap_directory if child.kind == "pcap" else None)
                child.started = time.monotonic()
                child.last_process_group = child.process.pid
                child.status = "collecting"
                for stream in ("stdout", "stderr"):
                    pipe = getattr(child.process, stream)
                    assert pipe is not None
                    os.set_blocking(pipe.fileno(), False)
                    self.selector.register(pipe, selectors.EVENT_READ, (child, stream))
                self._event("auxiliary_process_started", child, process_group=child.process.pid, command=command,
                            working_directory=str(self.pcap_directory) if child.kind == "pcap" else None,
                            selected_identity_pinned=child.kind == "docker" and "id" in child.identity_start)
            except (OSError, ValueError, AuxiliaryError) as exc:
                child.status = "unavailable"
                self._event("auxiliary_unavailable", child, reason="launch_or_output_setup_failed", error=type(exc).__name__)
                if child.process:
                    self._request_stop(child, "launch_setup_failed", time.monotonic())

    def _message(self, child: _Child, stream: str, raw: bytes, partial: bool = False) -> None:
        state = child.streams[stream]
        context = {"source_timestamp": None, "source_timestamp_format": "unavailable", "partial": partial}
        if child.kind == "journal" and stream == "stdout":
            try:
                record = json.loads(raw)
                if not isinstance(record, dict) or not isinstance(record.get("MESSAGE"), str):
                    raise ValueError("non-text journal record")
                stamp = record.get("__REALTIME_TIMESTAMP")
                if not isinstance(stamp, str) or re.fullmatch(r"[0-9]{1,20}", stamp) is None:
                    raise ValueError("source clock missing")
                context.update(source_timestamp=stamp, source_timestamp_format="unix_microseconds")
                raw = record["MESSAGE"].encode("utf-8", "replace")
                child.source_records += 1
            except (ValueError, TypeError, UnicodeError):
                child.invalid_records += 1
                child.output_incomplete = True
                state["sanitizer"].feed(b"authentication: discarded invalid journal record\n")
                return
        elif child.kind == "docker":
            match = DOCKER_TIME.match(raw)
            if match:
                context.update(source_timestamp=match[1].decode("ascii"), source_timestamp_format="rfc3339")
                raw = raw[match.end():]
                child.source_records += 1
        state["adapter"].context = context
        if child.kind == "pcap":
            captured = re.search(rb"Packets captured:\s*([0-9]+)", raw)
            dropped = re.search(rb"Packets received/dropped on interface.*?:\s*([0-9]+)/([0-9]+)", raw)
            if captured:
                self.pcap_stats["captured_packets"] = int(captured[1])
            if dropped:
                self.pcap_stats.update(received_packets=int(dropped[1]), dropped_packets=int(dropped[2]))
        lines = raw.split(b"\n", 32)
        omitted = lines.pop() if len(lines) > 32 else None
        if omitted is not None:
            child.discarded_bytes += len(omitted)
            child.output_incomplete = True
        if omitted is None and raw.endswith(b"\n"):
            lines.pop()  # A final terminator is not a separate blank message.
        for line in lines:
            if child.kind != "pcap" and CORE_SECRET.search(line):
                child.secret_guard_lines += 1
                line = b"authentication: redacted auxiliary core key or subscriber field"
            state["sanitizer"].feed(line + b"\n")
            if len(line) > 8192:
                # The shared sanitizer discards the overlong logical line. Its
                # unknown sensitivity must also protect later continuation rows.
                state["sanitizer"].feed(b"authentication: discarded overlong core continuation\n")
        if omitted is not None:
            state["sanitizer"].feed(b"authentication: discarded excess multiline continuation\n")
        if partial:
            child.output_incomplete = True

    def _consume(self, child: _Child, stream: str, budget: int) -> int:
        state = child.streams[stream]
        buffer = state["buffer"]
        used = 0
        while used < budget:
            newline = buffer.find(b"\n")
            if newline < 0:
                break
            raw = bytes(buffer[:newline])
            del buffer[:newline + 1]
            if state["discarding"]:
                child.discarded_bytes += len(raw) + 1
                state["discarding"] = False
            else:
                self._message(child, stream, raw)
            used += 1
        if len(buffer) > MAX_RAW_LINE:
            child.discarded_bytes += len(buffer)
            child.output_incomplete = True
            buffer.clear()
            state["discarding"] = True
            state["sanitizer"].feed(b"authentication: discarded overlong auxiliary record\n")
        if state["eof"] and buffer and used < budget:
            if state["discarding"]:
                child.discarded_bytes += len(buffer)
            else:
                self._message(child, stream, bytes(buffer), partial=True)
            buffer.clear()
            used += 1
        return used

    def _drain(self) -> None:
        budget = 128
        for child in self.children:
            for stream in child.streams:
                budget -= self._consume(child, stream, budget)
        ready = self.selector.select(timeout=0)
        if ready:
            start = self.read_cursor % len(ready)
            ready = (ready[start:] + ready[:start])[:8]
            self.read_cursor += len(ready)
        for key, _ in ready:
            child, stream = key.data
            state = child.streams[stream]
            if budget <= 0 or b"\n" in state["buffer"]:
                continue
            try:
                chunk = os.read(key.fileobj.fileno(), 4096)
            except BlockingIOError:
                continue
            except OSError:
                chunk = b""
                child.output_incomplete = True
            if not chunk:
                self.selector.unregister(key.fileobj)
                key.fileobj.close()
                state["eof"] = True
            else:
                child.raw_bytes += len(chunk)
                state["buffer"].extend(chunk)
            budget -= self._consume(child, stream, budget)

    @staticmethod
    def _group_exists(child: _Child) -> bool:
        if child.last_process_group is None:
            return False
        try:
            os.killpg(child.last_process_group, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _signal(self, child: _Child, signum: int) -> None:
        assert child.last_process_group is not None
        try:
            os.killpg(child.last_process_group, signum)
        except ProcessLookupError:
            pass
        except PermissionError:
            child.cleanup_failed = True
            self._event("auxiliary_signal_failed", child, signal=signum, reason="permission_denied")

    def _request_stop(self, child: _Child, reason: str, now: float) -> None:
        if child.process is None or child.stop_started is not None:
            return
        child.stop_started, child.stop_reason = now, reason
        self._event("auxiliary_stop_requested", child, reason=reason)
        self._signal(child, signal.SIGINT if child.kind == "pcap" else signal.SIGTERM)

    def _prepare_packet_directory(self) -> None:
        assert self.pcap_directory is not None
        owner = self.directory.stat()
        self.pcap_output_owner = (owner.st_uid, owner.st_gid)
        fd = os.open(self.pcap_directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            # dumpcap may retain UID 0 while dropping DAC capabilities. Give
            # only its new private leaf to that UID and use relative output
            # from cwd; inaccessible invoking-user ancestors need no changes.
            info = os.fstat(fd)
            self.pcap_directory_identity = (info.st_dev, info.st_ino)
            self.pcap_directory_handoff = "pending"
            if os.geteuid() == 0:
                os.fchown(fd, os.geteuid(), os.getegid())
            os.fchmod(fd, 0o700)
        finally:
            os.close(fd)

    def _open_packet_directory(self) -> int:
        assert self.pcap_directory is not None
        fd = os.open(self.pcap_directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        info = os.fstat(fd)
        if self.pcap_directory_identity != (info.st_dev, info.st_ino):
            os.close(fd)
            raise OSError("packet directory identity changed")
        return fd

    def _packet_file(self, directory_fd: int, name: str) -> int:
        if not name.endswith(".pcapng"):
            raise OSError("unexpected packet output entry")
        fd = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory_fd)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            os.close(fd)
            raise OSError("packet output is not an owned regular file")
        return fd

    def _handoff_packet_directory(self) -> None:
        if self.pcap_directory is None or self.pcap_directory_handoff != "pending":
            return
        packet = next((c for c in self.children if c.kind == "pcap"), None)
        if packet is not None and self._group_exists(packet):
            return
        try:
            fd = self._open_packet_directory()
            try:
                names = os.listdir(fd)
                if len(names) > self.settings.pcap_files + 2:
                    raise OSError("unexpected packet file count")
                for name in names:
                    packet_fd = self._packet_file(fd, name)
                    try:
                        if os.geteuid() == 0:
                            os.fchown(packet_fd, *self.pcap_output_owner)
                        os.fchmod(packet_fd, 0o600)
                    finally:
                        os.close(packet_fd)
                if os.geteuid() == 0:
                    os.fchown(fd, *self.pcap_output_owner)
                os.fchmod(fd, 0o700)
                self.pcap_directory_handoff = "complete"
            finally:
                os.close(fd)
        except OSError as exc:
            self.pcap_directory_handoff = "failed"
            self.pcap_scan_failed = True
            if packet is not None:
                packet.status = "failed"
                packet.output_incomplete = True
            self._event("auxiliary_packet_handoff_failed", reason=type(exc).__name__)

    def _scan_pcap(self) -> None:
        if self.pcap_directory is None:
            return
        files = []
        try:
            directory_fd = self._open_packet_directory()
            try:
                with os.scandir(directory_fd) as entries:
                    for index, entry in enumerate(entries):
                        if index >= self.settings.pcap_files + 2:
                            raise OSError("unexpected packet file count")
                        fd = self._packet_file(directory_fd, entry.name)
                        try:
                            os.fchmod(fd, 0o600)
                            info = os.fstat(fd)
                        finally:
                            os.close(fd)
                        files.append({"name": entry.name, "size_bytes": info.st_size,
                                      "mode": oct(stat.S_IMODE(info.st_mode))})
            finally:
                os.close(directory_fd)
            self.pcap_files = sorted(files, key=lambda item: item["name"])
            self.pcap_total_bytes = sum(item["size_bytes"] for item in files)
            if (self.pcap_total_bytes >= self.settings.pcap_max_bytes or
                    (len(files) >= self.settings.pcap_files and
                     all(item["size_bytes"] >= self.settings.pcap_file_kb * 1000 for item in files))):
                self.pcap_quota_reached = True
        except OSError as exc:
            if not self.pcap_scan_failed:
                self._event("auxiliary_packet_files_unavailable", reason=type(exc).__name__)
            self.pcap_scan_failed = True

    def _finish(self, child: _Child) -> None:
        for key in list(self.selector.get_map().values()):
            if key.data[0] is child:
                self.selector.unregister(key.fileobj)
                key.fileobj.close()
                child.streams[key.data[1]]["eof"] = True
                child.output_incomplete = True
        for state in child.streams.values():
            if state["buffer"]:
                child.discarded_bytes += len(state["buffer"])
                child.output_incomplete = True
                state["buffer"].clear()
            state["sanitizer"].finish()
        if child.cleanup_failed or child.forced_kill:
            child.status = "failed"
        elif child.stop_reason in ("storage_reserve", "packet_file_observation_failed", "launch_setup_failed"):
            child.status = "failed"
            child.output_incomplete = True
        elif child.stop_reason in ("main_worker_stopping", "duration_limit"):
            child.status = "stopped"
        elif (child.stop_reason == "quota_reached"
              or (child.kind == "pcap" and child.returncode == 0 and self.pcap_quota_reached)):
            child.status = "quota_reached"
        elif child.kind == "pcap" and child.returncode == 0:
            child.status = "completed"
        else:
            observed = child.source_records > 0 or (child.kind == "pcap" and bool(self.pcap_files))
            child.status = "failed" if observed else "unavailable"
        self._event("auxiliary_process_finished", child, status=child.status, returncode=child.returncode,
                    stop_reason=child.stop_reason, forced_kill=child.forced_kill,
                    cleanup_failed=child.cleanup_failed, output_incomplete=child.output_incomplete)
        child.process = None
        if child.kind == "pcap":
            self._handoff_packet_directory()

    def tick(self, stopping: bool = False) -> None:
        if not self.started or self.closed:
            return
        self._drain()
        now = time.monotonic()
        if now >= self.next_pcap_scan:
            self.next_pcap_scan = now + 0.25
            self._scan_pcap()
        for child in self.children:
            if child.process is None:
                continue
            child.returncode = child.process.poll()
            if child.returncode is not None:
                self._request_stop(child, "child_exited", now)
            elif stopping:
                self._request_stop(child, "main_worker_stopping", now)
            elif child.kind == "pcap":
                if not self.storage_allows_write():
                    self._request_stop(child, "storage_reserve", now)
                elif self.pcap_quota_reached:
                    self._request_stop(child, "quota_reached", now)
                elif self.pcap_scan_failed:
                    self._request_stop(child, "packet_file_observation_failed", now)
                elif now - child.started >= self.settings.pcap_duration:
                    self._request_stop(child, "duration_limit", now)
            if child.stop_started is not None:
                if now - child.stop_started >= 1.0 and not child.forced_kill and self._group_exists(child):
                    child.forced_kill = True
                    self._signal(child, signal.SIGKILL)
                    self._event("auxiliary_group_killed", child, process_group=child.last_process_group)
                if child.process.poll() is not None and not self._group_exists(child):
                    child.returncode = child.process.returncode
                    for _ in range(8):
                        self._drain()
                    if child.kind == "pcap":
                        self._scan_pcap()
                    self._finish(child)
                elif now - child.stop_started >= 3.0:
                    child.cleanup_failed = True

    def close(self) -> None:
        if self.closed:
            return
        deadline = time.monotonic() + 3.3
        while any(c.process is not None for c in self.children) and time.monotonic() < deadline:
            self.tick(stopping=True)
            if any(c.process is not None for c in self.children):
                time.sleep(0.02)
        for child in self.children:
            if child.process:
                self._signal(child, signal.SIGKILL)
                child.forced_kill = True
                try:
                    child.process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
                child.returncode = child.process.returncode
                child.cleanup_failed = self._group_exists(child)
                self._finish(child)
            if child.kind != "pcap" and child.identity_start:
                child.identity_end = self._identity(child)
                self._event("auxiliary_identity", child, phase="end", identity=child.identity_end)
            for state in child.streams.values():
                state["writer"].close()
        self._scan_pcap()
        self._handoff_packet_directory()
        self.selector.close()
        self.closed = True
        self._event("auxiliary_stopped", snapshot=self.snapshot())

    def _child_snapshot(self, child: _Child) -> dict[str, Any]:
        comparable = child.identity_start.get("status") == child.identity_end.get("status") == "available"
        streams = {n: {"writer": s["writer"].snapshot(), "sanitizer": s["sanitizer"].snapshot()}
                   for n, s in child.streams.items()}
        loss = (child.output_incomplete or child.status == "failed"
                or any(v["writer"].get("dropped_bytes", 0) or v["sanitizer"].get("discarded_bytes", 0)
                       for v in streams.values()))
        completeness = ("unavailable" if child.status in ("unavailable", "not_started") else
                        "partial" if loss else "collection_window_observed; absence_of_messages_is_not_core_health")
        return {"source_kind": child.kind, "source_name": child.name, "status": child.status,
                "identity_start": child.identity_start, "identity_end": child.identity_end,
                "identity_changed": child.identity_start != child.identity_end if comparable else None,
                "identity_continuity": "start_end_only" if comparable else "unavailable",
                "active_pid": child.process.pid if child.process is not None else None,
                "last_process_group": child.last_process_group, "returncode": child.returncode,
                "stop_reason": child.stop_reason, "forced_kill": child.forced_kill,
                "cleanup_failed": child.cleanup_failed, "output_incomplete": child.output_incomplete,
                "raw_bytes": child.raw_bytes, "discarded_bytes": child.discarded_bytes,
                "invalid_records": child.invalid_records, "source_records": child.source_records,
                "secret_guard_lines": child.secret_guard_lines,
                "streams": streams, "completeness": completeness,
                "redaction_counter_scope": "source_shared_across_streams" if child.kind != "pcap" else "per_stream"}

    def snapshot(self) -> dict[str, Any]:
        pcap = next((self._child_snapshot(c) for c in self.children if c.kind == "pcap"), {"status": "disabled"})
        pcap.update(directory=str(self.pcap_directory) if self.pcap_directory else None,
                    files=self.pcap_files, total_bytes=self.pcap_total_bytes,
                    directory_handoff=self.pcap_directory_handoff,
                    quota_reached=self.pcap_quota_reached, file_scan_failed=self.pcap_scan_failed,
                    byte_quota_overshoot=max(0, self.pcap_total_bytes - self.settings.pcap_max_bytes),
                    packet_statistics=dict(self.pcap_stats),
                    packet_statistics_status="available" if self.pcap_stats["dropped_packets"] is not None else "unavailable",
                    content="sensitive_unredacted_packets", retention="stop_at_quota_no_overwrite")
        return {"enabled": True, "run_id": self.run_id, "started": self.started, "closed": self.closed,
                "cleanup_failed": any(c.cleanup_failed for c in self.children),
                "core_logs": [self._child_snapshot(c) for c in self.children if c.kind != "pcap"], "pcap": pcap}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="parent for a unique private auxiliary directory")
    parser.add_argument("--run-id", required=True, help="shared operator-selected run identity")
    parser.add_argument("--duration", type=float, default=3600, help="1..86400 seconds total observation")
    parser.add_argument("--min-free-bytes", type=int, default=512 * 1024 * 1024,
                        help="free-space reserve shared by text output and dumpcap")
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        settings = validate_settings(args)
        if settings is None or not 1 <= args.duration <= 86400:
            raise AuxiliaryError("select a collection and --duration of 1..86400 seconds")
        if not 0 <= args.min_free_bytes <= (1 << 63) - 1:
            raise AuxiliaryError("--min-free-bytes must be nonnegative and fit a signed 64-bit value")
        if not args.run_id or len(args.run_id) > 128 or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", args.run_id) is None:
            raise AuxiliaryError("--run-id must be 1..128 ASCII identifier characters")
        from capture import (BoundedRotatingWriter, ByteQuota, CaptureHealth, OutputSanitizer,
                             StopLatch, StorageReserve, private_directory, write_json)
        if not args.output.exists():
            private_directory(args.output, parents=True)
        elif args.output.is_symlink() or not args.output.is_dir():
            raise AuxiliaryError("--output must be a directory, without a final symlink")
        directory = args.output / ("auxiliary-" + uuid.uuid4().hex)
        private_directory(directory)
        health, quota = CaptureHealth(), ByteQuota(64 * 1024 * 1024)
        storage = StorageReserve(directory, args.min_free_bytes)

        def writer_factory(prefix: str) -> Any:
            writer = BoundedRotatingWriter(directory, prefix, quota, 8 * 1024 * 1024, health)
            writer.storage = storage
            return writer
        events = writer_factory("auxiliary-events")
        supervisor = AuxiliarySupervisor(settings, directory, args.run_id, events, writer_factory,
                                         OutputSanitizer, private_directory, storage_allows_write=storage.allows_write)
        stop = StopLatch()
        previous = stop.install()
        try:
            supervisor.start()
            deadline = time.monotonic() + args.duration
            while stop.signal_number is None and time.monotonic() < deadline:
                supervisor.tick()
                if all(c.process is None for c in supervisor.children):
                    break
                time.sleep(0.05)
        finally:
            supervisor.close()
            stop.restore(previous)
            metadata = {"schema_version": 1, "run_id": args.run_id, "settings": settings.metadata(),
                        "clock": clock_sample(), "auxiliary": supervisor.snapshot(),
                        "capture_health": health.snapshot(), "events": events.snapshot()}
            written = write_json(directory / "metadata.json", metadata, health)
            events.close()
        print(str(directory))
        failed = any(c.status in ("unavailable", "failed") for c in supervisor.children)
        return 0 if written and health.snapshot()["healthy"] and not failed else 1
    except (AuxiliaryError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
