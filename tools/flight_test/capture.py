#!/usr/bin/env python3
"""Portable OAI flight collector with bounded buffers and sequential output.

The program intentionally uses only the Python standard library. It owns one
child process group, never discovers users or shells out, and performs no
recovery action beyond forwarding a signal that it received itself.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
from dataclasses import dataclass
import ipaddress
import json
import math
import os
import queue
import re
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from flight_health import (
    HealthStateMachine,
    SystemHealthCollector,
    bounded_command,
    clock_sample,
)
from flight_recovery import Decision, NativeChannel, RecoveryPolicy
from radio_health import RadioHealthDiagnostics


SCHEMA_VERSION = 1
MIB = 1024 * 1024
DEFAULT_STDOUT_BUDGET = 0
DEFAULT_RECORDER_BUDGET = 0
MIN_RECORDER_BUDGET = 8192
MAX_RECORDER_BUDGET = (1 << 63) - 1
DEFAULT_HOST_BUDGET = 0
DEFAULT_MIN_FREE_BYTES = 512 * MIB
DEFAULT_CHUNK_BYTES = 8 * MIB
MAX_LINE_BYTES = 8192
READ_CHUNK_BYTES = 8192

SENSITIVE_MARKERS = tuple(
    marker.encode("ascii")
    for marker in (
        "subscriber",
        "imsi",
        "supi",
        "suci",
        "authentication",
        "private key",
        "password",
        "token",
        "secret",
        "[sim]",
        "k_nas",
        "knasenc",
        "knasint",
        "kgnb",
        "kamf",
        "krrcenc",
        "krrcint",
        "krrcupenc",
        "krrcupint",
        "kupenc",
        "kupint",
        "uicc0.key",
        "k_amf",
        "k_gnb",
        "k_nh",
        "kseaf",
        "kausf",
        "res_star",
        "ciphering key",
        "integrity key",
        "security context",
        "opc",
        "op_c",
        "ki:",
        "ki=",
        "sqn",
        "autn",
        " rand",
    )
)
SENSITIVE_ARG_RE = re.compile(
    r"(subscriber|imsi|supi|suci|auth|private.?key|password|token|secret|"
    r"(?:^|[._-])key(?:$|[=._-])|(^|[_-])k_(nas|amf|gnb|nh)|\[sim\]|\bopc?\b|\bki\b)",
    re.IGNORECASE,
)
MODULE_LINE_RE = re.compile(
    br"^\s*(?:(?:\d{4}-\d{2}-\d{2}[T ])?\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,9})?\s+)?"
    br"\[[A-Za-z][A-Za-z0-9_.-]{1,48}\]"
)
MODULE_HEXDUMP_PAYLOAD_RE = re.compile(br"^(?=[0-9A-Fa-fxX])(?=.*[0-9A-Fa-f])[0-9A-Fa-fxX\s:.,|_-]+$")
ANSI_ESCAPE_RE = re.compile(br"\x1b\[[0-?]*[ -/]*[@-~]")


class CaptureError(RuntimeError):
    pass


class CaptureHealth:
    """Thread-safe capture quality state; it never drives process recovery."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reasons: set[str] = set()

    def unhealthy(self, reason: str) -> None:
        with self._lock:
            self._reasons.add(reason)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"healthy": not self._reasons, "reasons": sorted(self._reasons)}


class StopLatch:
    """Latch one operator stop across an attempt and any retry backoff."""

    def __init__(self) -> None:
        self.signal_number: Optional[int] = None

    def install(self) -> dict[int, Any]:
        previous: dict[int, Any] = {}
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._on_signal)
        return previous

    def restore(self, previous: dict[int, Any]) -> None:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    def _on_signal(self, signum: int, _frame: Any) -> None:
        if self.signal_number is None:
            self.signal_number = signum


@dataclass(frozen=True)
class CaptureResult:
    """The bounded outcome of one worker and its owned process group."""

    exit_code: int
    raw_returncode: Optional[int]
    state: str
    policy_decision: Optional[Decision]
    operator_stop: bool
    recovery_stop_reason: Optional[str]
    run_dir: Path
    process_group: Optional[int]



class ByteQuota:
    """An optional byte quota shared by writers; zero retains the complete run."""

    def __init__(self, limit: int) -> None:
        if limit < 0:
            raise ValueError("quota must be non-negative")
        self.limit = limit
        self.used = 0
        self._lock = threading.Lock()

    def reserve(self, wanted: int) -> int:
        with self._lock:
            granted = wanted if self.limit == 0 else max(0, min(wanted, self.limit - self.used))
            self.used += granted
            return granted

    def release(self, amount: int) -> None:
        if amount <= 0:
            return
        with self._lock:
            self.used = max(0, self.used - amount)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {"limit_bytes": self.limit, "used_bytes": self.used}


class ConsoleMirror:
    """A slow terminal may lose console copies, but cannot block disk capture."""

    def __init__(self) -> None:
        self.queue: queue.Queue[bytes] = queue.Queue(maxsize=64)
        self.stop = threading.Event()
        self.dropped_bytes = 0
        self.counter_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def write(self, data: bytes) -> None:
        try:
            self.queue.put_nowait(data)
        except queue.Full:
            with self.counter_lock:
                self.dropped_bytes += len(data)

    def _run(self) -> None:
        while not self.stop.is_set() or not self.queue.empty():
            try:
                data = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                while data:
                    count = os.write(sys.stdout.fileno(), data)
                    if count <= 0:
                        raise OSError("console write failed")
                    data = data[count:]
            except OSError:
                with self.counter_lock:
                    self.dropped_bytes += len(data)

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=0.2)


class StorageReserve:
    """Best-effort free-space reserve, shared by the asynchronous text writers."""

    def __init__(self, directory: Path, minimum: int) -> None:
        self.directory = directory
        self.minimum = minimum
        self.checked_at = 0.0
        self.available = True
        self.lock = threading.Lock()

    def allows_write(self) -> bool:
        if self.minimum == 0:
            return True
        with self.lock:
            now = time.monotonic()
            if now - self.checked_at >= 0.5:
                try:
                    space = os.statvfs(self.directory)
                    self.available = space.f_bavail * space.f_frsize > self.minimum
                except OSError:
                    self.available = False
                self.checked_at = now
            return self.available


class BoundedRotatingWriter:
    """Private numbered files with fixed-size chunks and explicit loss counts."""

    def __init__(
        self,
        directory: Path,
        prefix: str,
        quota: ByteQuota,
        chunk_bytes: int,
        health: CaptureHealth,
    ) -> None:
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        self.directory = directory
        self.prefix = prefix
        self.quota = quota
        self.chunk_bytes = chunk_bytes
        self.health = health
        self.console: Optional[ConsoleMirror] = None
        self.storage: Optional[StorageReserve] = None
        self.file_index = 0
        self.file_bytes = 0
        self.fd: Optional[int] = None
        self.disabled = False
        self._lock = threading.Lock()
        self.stats: dict[str, Any] = {
            "prefix": prefix,
            "files_opened": 0,
            "written_bytes": 0,
            "dropped_bytes": 0,
            "rotations": 0,
            "write_errors": 0,
            "last_error": None,
            "quota_reached": False,
        }

    def write(self, data: bytes) -> bool:
        if not data:
            return True
        if self.console is not None:
            self.console.write(data)
        if self.storage is not None and not self.storage.allows_write():
            with self._lock:
                self._drop(len(data), "free_space_reserve")
            return False
        offset = 0
        with self._lock:
            while offset < len(data):
                if self.disabled:
                    self._drop(len(data) - offset, "writer_disabled")
                    return False
                if self.fd is None or self.file_bytes >= self.chunk_bytes:
                    if not self._open_next():
                        self._drop(len(data) - offset, "writer_open_failure")
                        return False
                wanted = min(len(data) - offset, self.chunk_bytes - self.file_bytes)
                granted = self.quota.reserve(wanted)
                if granted <= 0:
                    self._drop(len(data) - offset, "capture_quota_reached")
                    self.stats["quota_reached"] = True
                    return False
                try:
                    written = os.write(self.fd, data[offset : offset + granted])
                except OSError as exc:
                    self.quota.release(granted)
                    self._write_error(exc)
                    self._drop(len(data) - offset, "writer_io_error")
                    return False
                if written < granted:
                    self.quota.release(granted - written)
                self.file_bytes += written
                self.stats["written_bytes"] += written
                offset += written
                if written < granted:
                    self._write_error(OSError(errno.EIO, "short write"))
                    self._drop(len(data) - offset, "writer_short_write")
                    return False
        return True

    def _open_next(self) -> bool:
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None
            self.file_index += 1
            self.stats["rotations"] += 1
        while True:
            path = self.directory / f"{self.prefix}.{self.file_index:04d}.log"
            try:
                self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                self.file_index += 1
                continue
            except OSError as exc:
                self._write_error(exc)
                return False
            self.file_bytes = 0
            self.stats["files_opened"] += 1
            return True

    def _drop(self, amount: int, reason: str) -> None:
        self.stats["dropped_bytes"] += max(0, amount)
        self.health.unhealthy(f"{self.prefix}:{reason}")

    def _write_error(self, exc: OSError) -> None:
        self.stats["write_errors"] += 1
        self.stats["last_error"] = errno.errorcode.get(exc.errno, exc.__class__.__name__)
        self.disabled = True
        if exc.errno == errno.ENOSPC:
            self.health.unhealthy("disk_enospc")
        else:
            self.health.unhealthy(f"{self.prefix}:write_error")

    def close(self) -> None:
        with self._lock:
            if self.fd is not None:
                try:
                    os.close(self.fd)
                except OSError:
                    pass
                self.fd = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.stats)


class SecretRedactor:
    """Fail closed across sensitive multi-line blocks, not merely one line."""

    def __init__(self) -> None:
        self.in_sensitive_block = False
        self.stats = {
            "lines_redacted": 0,
            "sensitive_blocks": 0,
            "continuity_lines_redacted": 0,
        }

    @staticmethod
    def _normalized(line: bytes) -> bytes:
        return ANSI_ESCAPE_RE.sub(b"", line)

    @classmethod
    def _is_sensitive(cls, line: bytes) -> bool:
        lower = cls._normalized(line).lower()
        return any(marker in lower for marker in SENSITIVE_MARKERS)

    @classmethod
    def _is_independent_module_diagnostic(cls, line: bytes) -> bool:
        normalized = cls._normalized(line)
        header = MODULE_LINE_RE.match(normalized)
        if header is None:
            return False
        payload = normalized[header.end() :].strip()
        return bool(payload) and MODULE_HEXDUMP_PAYLOAD_RE.fullmatch(payload) is None

    def accept(self, line: bytes) -> bool:
        sensitive = self._is_sensitive(line)
        if self.in_sensitive_block:
            normalized = self._normalized(line)
            if not sensitive and self._is_independent_module_diagnostic(line):
                # Preserve the first clearly independent OAI diagnostic. Unknown
                # text and module-prefixed hexadecimal continuation rows remain private.
                self.in_sensitive_block = False
                return True
            if not sensitive and not normalized.strip():
                self.in_sensitive_block = False
            self.stats["lines_redacted"] += 1
            self.stats["continuity_lines_redacted"] += 1
            return False
        if sensitive:
            self.in_sensitive_block = True
            self.stats["lines_redacted"] += 1
            self.stats["sensitive_blocks"] += 1
            return False
        return True


class OutputSanitizer:
    """Bound a line buffer and send only safe complete chunks to one writer."""

    def __init__(self, writer: BoundedRotatingWriter) -> None:
        self.writer = writer
        self.buffer = bytearray()
        self.discarding_long_line = False
        self.redactor = SecretRedactor()
        self.stats = {
            "chunks_received": 0,
            "lines_written": 0,
            "unterminated_lines": 0,
            "unterminated_line_truncated": 0,
            "discarded_bytes": 0,
        }

    def feed(self, data: bytes) -> None:
        self.stats["chunks_received"] += 1
        view = data
        while view:
            if self.discarding_long_line:
                newline = view.find(b"\n")
                if newline < 0:
                    self.stats["discarded_bytes"] += len(view)
                    return
                self.stats["discarded_bytes"] += newline + 1
                self.discarding_long_line = False
                view = view[newline + 1 :]
                continue

            capacity = MAX_LINE_BYTES - len(self.buffer)
            newline = view.find(b"\n", 0, capacity + 1)
            if newline >= 0:
                self.buffer.extend(view[:newline])
                self._emit(bytes(self.buffer))
                self.buffer.clear()
                view = view[newline + 1 :]
                continue
            if len(view) <= capacity:
                self.buffer.extend(view)
                return

            # No newline fits in the bounded line. Discard the full logical line,
            # including already buffered bytes, to avoid persisting a secret after
            # its marker appears later in a giant line.
            self.buffer.extend(view[:capacity])
            self.stats["unterminated_line_truncated"] += 1
            self.stats["discarded_bytes"] += len(self.buffer)
            self.buffer.clear()
            self.discarding_long_line = True
            view = view[capacity:]

    def finish(self) -> None:
        if self.discarding_long_line:
            self.stats["unterminated_lines"] += 1
            self.discarding_long_line = False
        if self.buffer:
            self.stats["unterminated_lines"] += 1
            self._emit(bytes(self.buffer))
            self.buffer.clear()

    def _emit(self, line: bytes) -> None:
        if line.startswith(b"flight recorder disabled:"):
            self.writer.health.unhealthy("recorder_disabled")
        if not self.redactor.accept(line):
            return
        self.writer.write(line + b"\n")
        self.stats["lines_written"] += 1

    def snapshot(self) -> dict[str, Any]:
        result = dict(self.stats)
        result.update(self.redactor.stats)
        return result


def _write_all(fd: int, content: bytes) -> None:
    offset = 0
    while offset < len(content):
        written = os.write(fd, content[offset:])
        if written <= 0:
            raise OSError(errno.EIO, "short metadata write")
        offset += written


def write_json(path: Path, value: dict[str, Any], health: Optional[CaptureHealth] = None) -> bool:
    """Write small metadata with no temporary files and a private mode."""

    content = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    if len(content) > 512 * 1024:
        raise CaptureError("metadata exceeds 512 KiB bound")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            _write_all(fd, content)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.chmod(path, 0o600)
        return True
    except OSError as exc:
        if health is not None:
            health.unhealthy("disk_enospc" if exc.errno == errno.ENOSPC else "metadata_write_error")
        return False


def file_fingerprint(path_text: Optional[str]) -> dict[str, Any]:
    if path_text is None:
        return {"state": "unconfigured", "sha256": None, "size_bytes": None, "basename": None}
    path = Path(path_text)
    try:
        source = path.open("rb")
    except (FileNotFoundError, PermissionError, OSError):
        return {
            "state": "unavailable",
            "sha256": None,
            "size_bytes": None,
            "basename": path.name,
        }
    digest = hashlib.sha256()
    size = 0
    try:
        with source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
    except OSError:
        return {"state": "unavailable", "sha256": None, "size_bytes": None, "basename": path.name}
    return {
        "state": "available",
        "sha256": digest.hexdigest(),
        "size_bytes": size,
        "basename": path.name,
    }


def runtime_module_fingerprints(binary_path: str) -> dict[str, dict[str, Any]]:
    parent = Path(binary_path).resolve().parent
    names = (
        "libparams_libconfig.so",
        "liboai_usrpdevif.so",
        "liboai_device.so",
        "librfsimulator.so",
    )
    return {name: file_fingerprint(str(parent / name)) for name in names}


def _git_executable() -> Optional[str]:
    for candidate in ("/usr/bin/git", "/bin/git"):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _stream_git_summary(executable: Optional[str], args: Sequence[str], count_status: bool = False) -> dict[str, Any]:
    """Hash or count Git output while retaining neither patch nor path names."""

    if executable is None:
        return {"state": "unavailable"}
    try:
        process = subprocess.Popen(
            [executable, *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
    except OSError as exc:
        return {"state": "unavailable", "reason": exc.__class__.__name__}
    assert process.stdout is not None
    digest = hashlib.sha256()
    paths = 0
    prefix = bytearray()
    skip_rename_path = False
    try:
        while True:
            chunk = process.stdout.read(65536)
            if not chunk:
                break
            digest.update(chunk)
            if not count_status:
                continue
            for value in chunk:
                if value == 0:
                    if skip_rename_path:
                        skip_rename_path = False
                    elif len(prefix) == 3 and prefix[2] == 32:
                        paths += 1
                        skip_rename_path = prefix[0] in (ord("R"), ord("C")) or prefix[1] in (ord("R"), ord("C"))
                    prefix.clear()
                elif not skip_rename_path and len(prefix) < 3:
                    prefix.append(value)
    finally:
        process.stdout.close()
    returncode = process.wait()
    if returncode != 0:
        return {"state": "unavailable", "returncode": returncode}
    result: dict[str, Any] = {"state": "available", "sha256": digest.hexdigest()}
    if count_status:
        result["path_count"] = paths
    return result


def safe_source_identity(repo: Optional[str]) -> dict[str, Any]:
    if repo is None:
        return {"state": "unconfigured", "head": None, "dirty_path_count": None, "tracked_patch_sha256": None}
    path = Path(repo)
    if not path.is_absolute() or not path.is_dir():
        return {"state": "invalid", "head": None, "dirty_path_count": None, "tracked_patch_sha256": None}
    executable = _git_executable()
    head = bounded_command(executable, ("-C", str(path), "rev-parse", "--verify", "HEAD"), 2.0)
    if head.get("status") != "ok" or head.get("returncode") != 0:
        return {"state": head.get("status", "unavailable"), "head": None, "dirty_path_count": None, "tracked_patch_sha256": None}
    status = _stream_git_summary(executable, ("-C", str(path), "status", "--porcelain=v1", "-z", "--untracked-files=all"), True)
    patch = _stream_git_summary(executable, ("-C", str(path), "diff", "--no-ext-diff", "--binary", "HEAD"))
    return {
        "state": "available",
        "head": head.get("output", "").strip()[:64],
        "status_state": status.get("state"),
        "dirty_path_count": status.get("path_count"),
        "tracked_patch_state": patch.get("state"),
        "tracked_patch_sha256": patch.get("sha256"),
        "persistence": "Git names and patch bytes are streamed for count/hash only and are not saved",
    }


def redact_argv(argv: Sequence[str]) -> list[str]:
    safe: list[str] = []
    redact_next = False
    for argument in argv:
        if redact_next:
            safe.append("<redacted-value>")
            redact_next = False
            continue
        if SENSITIVE_ARG_RE.search(argument):
            safe.append("<redacted-argument>")
            if "=" not in argument and argument.startswith("-"):
                redact_next = True
            continue
        safe.append(argument)
    return safe


def parse_gpsd(value: str) -> tuple[str, int]:
    if value.startswith("["):
        closing = value.find("]")
        if closing < 1 or value[closing + 1 : closing + 2] != ":":
            raise argparse.ArgumentTypeError("GPSD must use [IPv6]:PORT or HOST:PORT")
        host, port_text = value[1:closing], value[closing + 2 :]
    else:
        host, separator, port_text = value.rpartition(":")
        if not separator:
            raise argparse.ArgumentTypeError("GPSD must use HOST:PORT")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("GPSD port must be numeric") from exc
    if not host or not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("GPSD host and port are invalid")
    return host, port


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def byte_limit(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative byte count") from exc
    if not 0 <= result <= MAX_RECORDER_BUDGET:
        raise argparse.ArgumentTypeError("must be in 0..INT64_MAX")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounded observer-only supervisor for one absolute OAI executable."
    )
    parser.add_argument("--role", choices=("gnb", "ue"), required=True)
    parser.add_argument("--output", metavar="ROOT", help="default: REPO/cmake_targets/log/FlightTests/local-date")
    parser.add_argument("--console", action="store_true", help="mirror redacted console output without blocking capture")
    parser.add_argument("--working-directory", metavar="DIR", help="preserve the softmodem launch directory")
    parser.add_argument("--config", metavar="CONFIG")
    parser.add_argument("--repo", metavar="REPO")
    parser.add_argument("--core-ip", metavar="IP")
    parser.add_argument("--interface", default="oaitun_ue1")
    parser.add_argument("--gpsd", type=parse_gpsd, metavar="HOST:PORT")
    parser.add_argument("--probe-ping", action="store_true")
    parser.add_argument("--disable-recorder", action="store_true")
    parser.add_argument("--recovery", action="store_true", help="allow policy-gated worker recovery for the selected role")
    parser.add_argument("--recovery-stall", type=float, default=10.0)
    parser.add_argument("--recovery-attempt", type=float, default=120.0)
    parser.add_argument("--stop-timeout", type=float, default=10.0)
    parser.add_argument("--post-exit-drain-timeout", type=float, default=2.0)
    parser.add_argument("--startup-grace", type=float, default=30.0)
    parser.add_argument("--health-interval", type=float, default=1.0)
    parser.add_argument("--stdout-budget", type=byte_limit, default=DEFAULT_STDOUT_BUDGET)
    parser.add_argument("--recorder-budget", type=byte_limit, default=DEFAULT_RECORDER_BUDGET)
    parser.add_argument("--host-budget", type=byte_limit, default=DEFAULT_HOST_BUDGET)
    parser.add_argument("--min-free-bytes", type=byte_limit, default=DEFAULT_MIN_FREE_BYTES)
    parser.add_argument("--chunk-bytes", type=positive_int, default=DEFAULT_CHUNK_BYTES)
    parser.add_argument("command", nargs=argparse.REMAINDER, metavar="-- BINARY [ARGS ...]")
    return parser


def default_output(repo: str) -> Path:
    root = Path(repo)
    # Linked worktrees share the user's main checkout log root, not nested copies.
    if (root / ".git").is_file():
        try:
            result = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                capture_output=True, text=True, timeout=2, check=False,
            )
            common = Path(result.stdout.strip())
            if result.returncode == 0 and common.name == ".git" and (common.parent / "cmake_targets").is_dir():
                root = common.parent
        except (OSError, subprocess.TimeoutExpired):
            pass
    return root / "cmake_targets/log/FlightTests" / time.strftime("%Y-%m-%d")


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str]:
    time_values = (args.stop_timeout, args.post_exit_drain_timeout, args.health_interval, args.startup_grace, args.recovery_stall, args.recovery_attempt)
    if not all(math.isfinite(value) for value in time_values):
        parser.error("timeouts and intervals must be finite")
    if args.stop_timeout <= 0 or args.post_exit_drain_timeout <= 0 or args.health_interval <= 0 or args.startup_grace < 0:
        parser.error("timeout/interval must be positive and startup grace non-negative")
    if args.recovery_stall < 1 or args.recovery_attempt < 1:
        parser.error("--recovery-stall and --recovery-attempt must each be at least one second")
    if args.probe_ping and args.core_ip is None:
        parser.error("--probe-ping requires --core-ip for a remote core endpoint")
    if not args.disable_recorder and args.recorder_budget != 0 and not MIN_RECORDER_BUDGET <= args.recorder_budget <= MAX_RECORDER_BUDGET:
        parser.error(
            f"--recorder-budget must be 0 or {MIN_RECORDER_BUDGET}..{MAX_RECORDER_BUDGET} bytes"
        )
    if args.core_ip is not None:
        try:
            ipaddress.ip_address(args.core_ip)
        except ValueError:
            parser.error("--core-ip must be an IP address")
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("an executable is required after --")
    if not os.path.isabs(command[0]):
        parser.error("the executable after -- must be an explicit absolute path")
    if args.repo is not None and not os.path.isabs(args.repo):
        parser.error("--repo must be an absolute path")
    if args.config is not None and not os.path.isabs(args.config):
        parser.error("--config must be an absolute path so child CWD remains contained")
    if args.working_directory is not None:
        args.working_directory = str(Path(args.working_directory).resolve())
        if not Path(args.working_directory).is_dir():
            parser.error("--working-directory must name an existing directory")
    for index, argument in enumerate(command):
        value = None
        if argument in ("-O", "--config") and index + 1 < len(command):
            value = command[index + 1]
        elif argument.startswith("--config="):
            value = argument.split("=", 1)[1]
        elif argument.startswith("-O="):
            value = argument.split("=", 1)[1]
        if value is not None and not os.path.isabs(value) and args.working_directory is None:
            parser.error("configuration argument after -- must be an absolute path")
    if args.output is None:
        if args.repo is None:
            parser.error("default output needs --repo (supplied automatically by the softmodem)")
        args.output = str(default_output(args.repo))
    return command


class RecorderMonitor:
    """Observe, but never control, the separately bounded C recorder."""

    def __init__(self, directory: Path, limit_bytes: int, enabled: bool) -> None:
        self.directory = directory
        self.limit_bytes = limit_bytes
        self.enabled = enabled
        self.previous_exceeded = False

    def sample(self, health: CaptureHealth) -> dict[str, Any]:
        if not self.enabled:
            return {"state": "disabled", "observed_bytes": 0, "files": 0}
        total = 0
        files = 0
        scan_limited = False
        try:
            with os.scandir(self.directory) as entries:
                for entry in entries:
                    files += 1
                    if files > 64:
                        scan_limited = True
                        break
                    try:
                        mode = entry.stat(follow_symlinks=False).st_mode
                        if stat.S_ISREG(mode):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError as exc:
            return {"state": "unavailable", "reason": errno.errorcode.get(exc.errno, exc.__class__.__name__)}
        exceeded = self.limit_bytes != 0 and total > self.limit_bytes
        if exceeded:
            health.unhealthy("recorder_contract_exceeded")
        self.previous_exceeded = self.previous_exceeded or exceeded
        return {
            "state": "exceeded" if exceeded else "partial" if scan_limited else "available",
            "observed_bytes": total,
            "files": min(files, 64),
            "scan_limited": scan_limited,
            "limit_bytes": self.limit_bytes,
            "scope": "observed only; capture never stops radio because of recorder storage",
        }


class HealthWorker(threading.Thread):
    """A normal-priority offline sampler running outside the child output path."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        collector: SystemHealthCollector,
        state: HealthStateMachine,
        recorder: RecorderMonitor,
        host_writer: BoundedRotatingWriter,
        health: CaptureHealth,
        interval_seconds: float,
    ) -> None:
        super().__init__(name="oai-flight-health", daemon=False)
        self.process = process
        self.collector = collector
        self.state = state
        self.recorder = recorder
        self.host_writer = host_writer
        self.health = health
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.samples = 0
        self.failures = 0
        self.last_assessment: Optional[dict[str, Any]] = None

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                running = self.process.poll() is None
                recorder = self.recorder.sample(self.health)
                observation = self.collector.sample(self.process.pid, running)
                assessment = self.state.observe(
                    self.collector.normalized_input(observation, running, self.health.snapshot()["healthy"])
                )
                self.last_assessment = assessment
                event = {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "health",
                    "observation": observation,
                    "assessment": assessment,
                    "recorder": recorder,
                }
                self.host_writer.write(
                    json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
                )
                self.samples += 1
            except Exception as exc:
                self.failures += 1
                self.health.unhealthy("health_collector_failure")
                failure = {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "health_collector_failure",
                    "exception": exc.__class__.__name__,
                    "clock": clock_sample(),
                }
                self.host_writer.write(
                    json.dumps(failure, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
                )
            self.stop_event.wait(self.interval_seconds)

    def stop(self) -> None:
        self.stop_event.set()

    def snapshot(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "failures": self.failures,
            "last_assessment": self.last_assessment,
            "alive_at_snapshot": self.is_alive(),
        }


class FlightCapture:
    """Create one private run directory and supervise one child process group."""

    def __init__(
        self,
        args: argparse.Namespace,
        command: list[str],
        *,
        run_dir: Optional[Path] = None,
        session_id: Optional[str] = None,
        policy: Optional[RecoveryPolicy] = None,
        begin_policy_attempt: bool = True,
        stop_latch: Optional[StopLatch] = None,
        recovery_enabled: bool = False,
        launch_observer: Optional[Callable[[int], None]] = None,
    ) -> None:
        self.args = args
        self.command = command
        self.health = CaptureHealth()
        self.run_dir = run_dir
        self.process: Optional[subprocess.Popen[bytes]] = None
        self.process_start_ticks: Optional[int] = None
        self.console: Optional[ConsoleMirror] = None
        self.stop_signal: Optional[int] = None
        self.stop_sent_ns: Optional[int] = None
        self.recovery_stop_reason: Optional[str] = None
        self.recovery_stop_sent_ns: Optional[int] = None
        self.kill_sent = False
        self.session_id = session_id or uuid.uuid4().hex
        self.policy = policy or RecoveryPolicy(args.recovery_stall, args.recovery_attempt, args.role)
        self.begin_policy_attempt = begin_policy_attempt
        self.stop_latch = stop_latch
        self.recovery_enabled = recovery_enabled
        self.launch_observer = launch_observer
        self.policy_decision: Optional[Decision] = None
        self._last_policy_decision: Optional[tuple[str, str, int]] = None
        self.last_result: Optional[CaptureResult] = None
        self.radio_health = RadioHealthDiagnostics()
        self.radio_health_writer: Optional[BoundedRotatingWriter] = None
        self.counters = {
            "launch_attempts": 0,
            "launch_failures": 0,
            "normal_exits": 0,
            "nonzero_exits": 0,
            "signal_exits": 0,
            "stop_requests": 0,
            "graceful_group_signals": 0,
            "forced_group_kills": 0,
            "post_exit_pipe_timeouts": 0,
            "residual_pipe_streams": 0,
            "shutdowns": 0,
        }

    def run(self) -> int:
        run_dir = self.run_dir or self._create_run_dir()
        self.run_dir = run_dir
        print(f"[FLIGHT] logging enabled: {run_dir}", file=sys.stderr, flush=True)
        recorder_dir = run_dir / "recorder"
        recorder_dir.mkdir(mode=0o700)
        os.chmod(recorder_dir, 0o700)
        child_cwd, cwd_mode = self._child_cwd(run_dir)
        metadata = self._metadata(recorder_dir, child_cwd, cwd_mode)
        write_json(run_dir / "metadata.json", metadata, self.health)

        stdout_quota = ByteQuota(self.args.stdout_budget)
        host_quota = ByteQuota(self.args.host_budget)
        stdout_writer = BoundedRotatingWriter(
            run_dir,
            "stdout",
            stdout_quota,
            self.args.chunk_bytes,
            self.health,
        )
        stderr_writer = BoundedRotatingWriter(
            run_dir,
            "stderr",
            stdout_quota,
            self.args.chunk_bytes,
            self.health,
        )
        host_writer = BoundedRotatingWriter(
            run_dir,
            "host",
            host_quota,
            min(self.args.chunk_bytes, 4 * MIB),
            self.health,
        )
        recovery_writer = BoundedRotatingWriter(
            run_dir,
            "recovery",
            host_quota,
            min(self.args.chunk_bytes, 4 * MIB),
            self.health,
        )
        radio_health_writer = BoundedRotatingWriter(
            run_dir,
            "radio_health",
            host_quota,
            min(self.args.chunk_bytes, 4 * MIB),
            self.health,
        )
        self.radio_health_writer = radio_health_writer
        if self.args.console:
            self.console = ConsoleMirror()
            stdout_writer.console = self.console
            stderr_writer.console = self.console
        storage = StorageReserve(run_dir, self.args.min_free_bytes)
        for writer in (stdout_writer, stderr_writer, host_writer, recovery_writer, radio_health_writer):
            writer.storage = storage
        old_handlers = self._install_signal_handlers() if self.stop_latch is None else None
        try:
            return self._launch_and_supervise(
                recorder_dir,
                child_cwd,
                stdout_writer,
                stderr_writer,
                host_writer,
                recovery_writer,
            )
        finally:
            if old_handlers is not None:
                self._restore_signal_handlers(old_handlers)
            stdout_writer.close()
            stderr_writer.close()
            host_writer.close()
            recovery_writer.close()
            radio_health_writer.close()
            if self.console is not None:
                self.console.close()

    def _create_run_dir(self) -> Path:
        root = Path(self.args.output).resolve()
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(root, 0o700)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        run_dir = Path(tempfile.mkdtemp(prefix=f"{self.args.role}-{stamp}-{os.getpid()}-", dir=root))
        os.chmod(run_dir, 0o700)
        return run_dir

    def _child_cwd(self, run_dir: Path) -> tuple[Path, str]:
        if self.args.working_directory is not None:
            return Path(self.args.working_directory), "original_launch_directory"
        work = run_dir / "working"
        work.mkdir(mode=0o700)
        os.chmod(work, 0o700)
        return work, "per_run_contained"

    def _metadata(self, recorder_dir: Path, child_cwd: Path, cwd_mode: str) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "flight_capture_metadata",
            "run_id": uuid.uuid4().hex,
            "session_id": self.session_id,
            "role": self.args.role,
            "features": ["log", "recovery"] if self.recovery_enabled else ["log"],
            "radio_health": {
                "enabled": True,
                "stream": "radio_health.*.log JSON Lines",
                "wire_kind": "radio_health",
                "wire_schema_version": 1,
                "transport_diagnostics": "observer_only_unqualified_for_restart",
                "budget": "shares the existing host budget and free-space reserve",
            },
            "directory_date_basis": "host local date at launch",
            "created_clock": clock_sample(),
            "command": redact_argv(self.command),
            "binary": file_fingerprint(self.command[0]),
            "config": file_fingerprint(self.args.config),
            "config_was_absolute": bool(self.args.config and os.path.isabs(self.args.config)),
            "source": safe_source_identity(self.args.repo),
            "child_working_directory": {
                "mode": cwd_mode,
                "basename": child_cwd.name,
                "contains_oai_cwd_statistics": cwd_mode == "per_run_contained",
            },
            "runtime": {
                "executable_parent_prepended_to_ld_library_path": True,
                "environment_values_saved": False,
                "adjacent_module_fingerprints": runtime_module_fingerprints(self.command[0]),
            },
            "recorder": {
                "enabled": not self.args.disable_recorder,
                "directory_basename": recorder_dir.name,
                "maximum_bytes": self.args.recorder_budget,
                "enabled_range_bytes": [MIN_RECORDER_BUDGET, MAX_RECORDER_BUDGET],
                "zero_budget_means_no_total_cap": True,
                "retention": "sequential_files_no_overwrite",
                "contract": "OAI_FLIGHT_RECORDER_DIR and OAI_FLIGHT_RECORDER_MAX_BYTES",
            },
            "limits": {
                "stdout_and_stderr_budget_bytes": self.args.stdout_budget,
                "recorder_budget_bytes": self.args.recorder_budget,
                "host_budget_bytes": self.args.host_budget,
                "chunk_bytes": self.args.chunk_bytes,
                "minimum_free_bytes": self.args.min_free_bytes,
                "zero_budget_means_no_total_cap": True,
                "max_line_bytes": MAX_LINE_BYTES,
                "disk_bound_scope": "optional per-category caps; otherwise free-space reserve; metadata separately bounded",
            },
            "privacy": {
            "recovery": {
                "enabled": self.recovery_enabled,
                "native_channel": "one inherited OAI_FLIGHT_MONITOR_FD socketpair per worker",
                "stall_seconds": self.args.recovery_stall,
                "attempt_seconds": self.args.recovery_attempt,
                "actions": "policy observations only unless recovery is enabled",
            },
                "config_contents_saved": False,
                "environment_saved": False,
                "git_diff_saved": False,
                "numeric_recorder_packet_payloads_saved": False,
                "host_collector_packet_payloads_saved": False,
                "child_output_policy": (
                    "sensitive lines and continuous blocks are dropped until a blank line or recognized non-sensitive "
                    "OAI module diagnostic; stdout/stderr may retain protocol dumps, so review private output before export"
                ),
                "timestamp_scope": "capture receipt clocks only; source event timestamps require source instrumentation",
            },
        }

    def _launch_and_supervise(
        self,
        recorder_dir: Path,
        child_cwd: Path,
        stdout_writer: BoundedRotatingWriter,
        stderr_writer: BoundedRotatingWriter,
        host_writer: BoundedRotatingWriter,
        recovery_writer: BoundedRotatingWriter,
    ) -> int:
        environment = os.environ.copy()
        environment["_OAI_FLIGHT_CAPTURE_PARENT"] = str(os.getpid())
        if self.args.disable_recorder:
            environment.pop("OAI_FLIGHT_RECORDER_DIR", None)
            environment.pop("OAI_FLIGHT_RECORDER_MAX_BYTES", None)
        else:
            environment["OAI_FLIGHT_RECORDER_DIR"] = str(recorder_dir)
            environment["OAI_FLIGHT_RECORDER_MAX_BYTES"] = str(self.args.recorder_budget)
            environment["OAI_FLIGHT_RECORDER_MIN_FREE_BYTES"] = str(self.args.min_free_bytes)
        executable_library_dir = str(Path(self.command[0]).resolve().parent)
        existing_library_path = environment.get("LD_LIBRARY_PATH")
        environment["LD_LIBRARY_PATH"] = executable_library_dir + (
            os.pathsep + existing_library_path if existing_library_path else ""
        )
        native = NativeChannel()
        environment["_OAI_FLIGHT_MONITOR_FD"] = str(native.child.fileno())
        if self.begin_policy_attempt:
            self.policy.begin_attempt(time.monotonic_ns())
        # Check at the Popen boundary as well as the session boundary. A latched
        # operator signal must never turn a pending retry into another worker.
        if self._operator_stop_signal() is not None:
            native.close()
            return self._operator_stop_before_launch(
                stdout_writer,
                stderr_writer,
                host_writer,
                recovery_writer,
            )
        self.counters["launch_attempts"] += 1
        try:
            process = subprocess.Popen(
                self.command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=child_cwd,
                env=environment,
                close_fds=True,
                start_new_session=True,
                pass_fds=(native.child.fileno(),),
            )
        except OSError as exc:
            self.counters["launch_failures"] += 1
            native.close()
            self.policy_decision = Decision("stop", "launch_failed")
            return_code = 127 if exc.errno == errno.ENOENT else 126
            self._final_status(
                "launch_failed",
                return_code,
                None,
                {"launch_error": errno.errorcode.get(exc.errno, exc.__class__.__name__)},
                stdout_writer,
                stderr_writer,
                host_writer,
                recovery_writer,
                None,
            )
            assert self.run_dir is not None
            self.last_result = CaptureResult(
                return_code,
                None,
                "launch_failed",
                self.policy_decision,
                self._operator_stop_signal() is not None,
                None,
                self.run_dir,
                None,
            )
            return return_code
        native.child.close()

        self.process = process
        self.process_start_ticks = self._process_start_ticks(process.pid)
        if self.launch_observer is not None:
            try:
                self.launch_observer(process.pid)
            except Exception:
                self.health.unhealthy("launch_observer_failure")
        assert process.stdout is not None
        assert process.stderr is not None
        collector = SystemHealthCollector(
            interface=self.args.interface,
            role=self.args.role,
            core_ip=self.args.core_ip,
            probe_ping=self.args.probe_ping,
            gpsd=self.args.gpsd,
        )
        worker = HealthWorker(
            process,
            collector,
            HealthStateMachine(self.args.startup_grace, 3, self.args.role),
            RecorderMonitor(recorder_dir, self.args.recorder_budget, not self.args.disable_recorder),
            host_writer,
            self.health,
            self.args.health_interval,
        )
        worker.start()
        stdout_sanitizer = OutputSanitizer(stdout_writer)
        stderr_sanitizer = OutputSanitizer(stderr_writer)
        selector = selectors.DefaultSelector()
        for stream, sanitizer in ((process.stdout, stdout_sanitizer), (process.stderr, stderr_sanitizer)):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, sanitizer)

        leader_returncode: Optional[int] = None
        leader_exit_ns: Optional[int] = None
        residual_pipe_timeout = False
        residual_pipe_streams = 0
        pipes_closed_early = False
        try:
            while True:
                self._forward_stop_request(process)
                now_ns = time.monotonic_ns()
                # A clean leader exit is classified before a native policy tick
                # can request recovery. A datagram received after that exit is
                # still recorded below, but cannot turn an unknown zero exit
                # into a controlled restart.
                observed_returncode = process.poll()
                if leader_returncode is None and observed_returncode is not None:
                    leader_returncode = observed_returncode
                    leader_exit_ns = now_ns
                self._drain_native(
                    native,
                    process.pid,
                    recovery_writer,
                    allow_actions=leader_returncode is None,
                )
                self._apply_stop_deadline(process)

                if leader_returncode is not None:
                    if (
                        selector.get_map()
                        and leader_exit_ns is not None
                        and now_ns - leader_exit_ns >= int(self.args.post_exit_drain_timeout * 1_000_000_000)
                    ):
                        residual_pipe_timeout = True
                        residual_pipe_streams = len(selector.get_map())
                        self.counters["post_exit_pipe_timeouts"] += 1
                        self.counters["residual_pipe_streams"] += residual_pipe_streams
                        self.health.unhealthy("post_exit_residual_pipes")
                        for key in list(selector.get_map().values()):
                            selector.unregister(key.fileobj)
                        process.stdout.close()
                        process.stderr.close()
                        pipes_closed_early = True
                    if not selector.get_map():
                        # After an explicit TERM or INT, retain the owned-group
                        # deadline if descendants still exist after leader exit.
                        if (
                            (self.stop_sent_ns is not None or self.recovery_stop_sent_ns is not None)
                            and not self.kill_sent
                            and self._owned_group_exists(process)
                        ):
                            time.sleep(0.05)
                            continue
                        # The worker may send its final native datagram after
                        # the loop's previous drain but before poll sees exit.
                        # Drain once more after exit, without issuing actions.
                        self._drain_native(native, process.pid, recovery_writer, allow_actions=False)
                        break

                if not selector.get_map():
                    time.sleep(0.05)
                    continue
                for key, _ in selector.select(timeout=0.1):
                    try:
                        chunk = os.read(key.fileobj.fileno(), READ_CHUNK_BYTES)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    key.data.feed(chunk)
        finally:
            selector.close()
            if not pipes_closed_early:
                process.stdout.close()
                process.stderr.close()
            native.reader.close()
            worker.stop()
            worker.join(timeout=5.0)
            if worker.is_alive():
                self.health.unhealthy("health_worker_shutdown_timeout")
        stdout_sanitizer.finish()
        stderr_sanitizer.finish()
        returncode = leader_returncode if leader_returncode is not None else process.wait()
        exit_code, exit_kind = self._exit_code(returncode)
        self.policy_decision = self.policy.exited(
            time.monotonic_ns(),
            returncode,
            self._operator_stop_signal() is not None,
            controlled_recovery=self.recovery_stop_reason is not None,
        )
        self._record_policy_decision(recovery_writer, "exited", self.policy_decision, force=True)
        assert self.run_dir is not None
        self.last_result = CaptureResult(
            exit_code,
            returncode,
            exit_kind,
            self.policy_decision,
            self._operator_stop_signal() is not None,
            self.recovery_stop_reason,
            self.run_dir,
            process.pid,
        )

        self._final_status(
            exit_kind,
            exit_code,
            returncode,
            {
                "stdout_sanitizer": stdout_sanitizer.snapshot(),
                "stderr_sanitizer": stderr_sanitizer.snapshot(),
                "pipe_drain": {
                    "post_exit_drain_timeout_seconds": self.args.post_exit_drain_timeout,
                    "leader_returncode": leader_returncode,
                    "residual_output_status": "incomplete" if residual_pipe_timeout else "complete",
                    "residual_open_pipe_streams": residual_pipe_streams,
                    "residual_bytes_not_captured": "unknown" if residual_pipe_timeout else 0,
                },
            },
            stdout_writer,
            stderr_writer,
            host_writer,
            recovery_writer,
            worker,
        )
        return exit_code

    @staticmethod
    def _process_start_ticks(pid: int) -> Optional[int]:
        """Return Linux /proc start ticks without retaining process details."""

        try:
            stat_text = Path("/proc").joinpath(str(pid), "stat").read_text(encoding="ascii")
            fields = stat_text[stat_text.rfind(")") + 2 :].split()
            return int(fields[19]) if len(fields) > 19 else None
        except (OSError, UnicodeDecodeError, ValueError):
            return None

    def _operator_stop_before_launch(
        self,
        stdout_writer: BoundedRotatingWriter,
        stderr_writer: BoundedRotatingWriter,
        host_writer: BoundedRotatingWriter,
        recovery_writer: BoundedRotatingWriter,
    ) -> int:
        signal_number = self._operator_stop_signal()
        assert signal_number is not None
        return_code = 128 + abs(signal_number)
        self.policy_decision = Decision("stop", "operator_stop_before_launch")
        self._final_status(
            "operator_stop_before_launch",
            return_code,
            None,
            {"operator_stop_signal": signal_number},
            stdout_writer,
            stderr_writer,
            host_writer,
            recovery_writer,
            None,
        )
        assert self.run_dir is not None
        self.last_result = CaptureResult(
            return_code,
            None,
            "operator_stop_before_launch",
            self.policy_decision,
            True,
            None,
            self.run_dir,
            None,
        )
        return return_code

    def _write_recovery_event(self, writer: BoundedRotatingWriter, kind: str, **fields: Any) -> None:
        receipt = clock_sample()
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "session_id": self.session_id,
            "attempt_directory": self.run_dir.name if self.run_dir is not None else None,
            "policy_generation": self.policy.snapshot()["generation"],
            "receipt_clock": receipt,
        }
        payload.update(fields)
        writer.write(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")

    def _write_radio_health_event(self, event: dict[str, Any]) -> None:
        """Write one observer-only diagnostic event to the bounded JSONL stream."""

        if self.radio_health_writer is None:
            return
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "attempt_directory": self.run_dir.name if self.run_dir is not None else None,
            "role": self.args.role,
            "process_attempt": {"policy_generation": self.policy.snapshot()["generation"]},
            "receipt_clock": clock_sample(),
        }
        payload.update(event)
        self.radio_health_writer.write(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")

    def _record_policy_decision(self, writer: BoundedRotatingWriter, phase: str, decision: Decision, force: bool = False) -> None:
        identity = (decision.action, decision.reason, decision.not_before_ns)
        if not force and identity == self._last_policy_decision:
            return
        self._last_policy_decision = identity
        self._write_recovery_event(
            writer,
            "recovery_policy_transition",
            policy_phase=phase,
            decision={
                "action": decision.action,
                "reason": decision.reason,
                "not_before_ns": decision.not_before_ns,
            },
            policy_snapshot=self.policy.snapshot(),
        )

    def _drain_native(self, channel: NativeChannel, pid: int,
                      writer: BoundedRotatingWriter, *, allow_actions: bool = True) -> None:
        samples = channel.receive(pid)
        for sample in samples:
            self._write_recovery_event(
                writer,
                "native_progress_snapshot",
                native_pid=pid,
                source_sequence=sample["sequence"],
                source_clock={
                    "monotonic_ns": sample["mono_ns"],
                    "utc_wall_ns": None,
                    "utc_wall_state": "unavailable_from_native_abi",
                },
                native_values=sample["values"],
                native_send_drops=sample["send_drops"],
            )
            for change in self.policy.observe(sample, time.monotonic_ns()):
                self._write_recovery_event(
                    writer,
                    "recovery_policy_transition",
                    policy_phase="native_observe",
                    change=change,
                    policy_snapshot=self.policy.snapshot(),
                )
        receipt_ns = time.monotonic_ns()
        for snapshot in channel.take_radio_health():
            accepted_before = self.radio_health.accepted
            for event in self.radio_health.observe(snapshot, receipt_ns):
                self._write_radio_health_event(event)
            if self.radio_health.accepted == accepted_before:
                continue
            for change in self.policy.observe_radio_health(snapshot, receipt_ns):
                self._write_recovery_event(
                    writer,
                    "recovery_policy_transition",
                    policy_phase="radio_health_observe",
                    change=change,
                    policy_snapshot=self.policy.snapshot(),
                )
        for event in self.radio_health.tick(receipt_ns):
            self._write_radio_health_event(event)
        decision = self.policy.tick(time.monotonic_ns())
        self._record_policy_decision(writer, "tick", decision)
        if allow_actions and self.recovery_enabled and decision.action == "restart":
            self._request_recovery_stop(self.process, decision.reason)
        self.native_channel = {
            "sequence": channel.sequence,
            "invalid": channel.invalid,
            "gaps": channel.gaps,
            "latest_source_monotonic_ns": channel.latest["mono_ns"] if channel.latest is not None else None,
            "radio_health_received": channel.radio_health_received,
            "radio_health_invalid": channel.radio_health_invalid,
            "radio_health": self.radio_health.snapshot(),
        }

    def _install_signal_handlers(self) -> dict[int, Any]:
        previous: dict[int, Any] = {}
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._on_stop_signal)
        return previous

    def _restore_signal_handlers(self, previous: dict[int, Any]) -> None:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    def _on_stop_signal(self, signum: int, _frame: Any) -> None:
        # Python can dispatch another signal handler between bytecodes. Keep this
        # handler flag-only and idempotent; normal-loop code owns side effects.
        if self.stop_signal is None:
            self.stop_signal = signum

    def _operator_stop_signal(self) -> Optional[int]:
        if self.stop_latch is not None:
            return self.stop_latch.signal_number
        return self.stop_signal

    def _forward_stop_request(self, process: subprocess.Popen[bytes]) -> None:
        requested = self._operator_stop_signal()
        if requested is None or self.stop_sent_ns is not None:
            return
        self.stop_signal = requested
        self.stop_sent_ns = time.monotonic_ns()
        self.counters["stop_requests"] += 1
        try:
            # The child created this process group. It can remain after its leader
            # exits, so do not use process.poll() to gate an owned-group signal.
            os.killpg(process.pid, requested)
            self.counters["graceful_group_signals"] += 1
        except ProcessLookupError:
            pass

    def _request_recovery_stop(self, process: subprocess.Popen[bytes], reason: str) -> None:
        if self._operator_stop_signal() is not None or self.recovery_stop_sent_ns is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            # The worker group was already gone. In particular, do not make a
            # later clean leader exit appear to be a controlled recovery.
            return
        self.recovery_stop_reason = reason
        self.recovery_stop_sent_ns = time.monotonic_ns()
        self.counters["recovery_stop_requests"] = self.counters.get("recovery_stop_requests", 0) + 1
        self.counters["graceful_group_signals"] += 1

    def _owned_group_exists(self, process: subprocess.Popen[bytes]) -> bool:
        try:
            os.killpg(process.pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return False

    def _apply_stop_deadline(self, process: subprocess.Popen[bytes]) -> None:
        if self.kill_sent:
            return
        started_ns = self.stop_sent_ns if self.stop_sent_ns is not None else self.recovery_stop_sent_ns
        if started_ns is None:
            return
        timeout_seconds = self.args.stop_timeout if self.stop_sent_ns is not None else 10.0
        elapsed = time.monotonic_ns() - started_ns
        if elapsed < int(timeout_seconds * 1_000_000_000):
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
            self.counters["forced_group_kills"] += 1
        except ProcessLookupError:
            pass
        self.kill_sent = True

    def _exit_code(self, returncode: int) -> tuple[int, str]:
        self.counters["shutdowns"] += 1
        if returncode == 0:
            self.counters["normal_exits"] += 1
            return 0, "exited"
        if returncode > 0:
            self.counters["nonzero_exits"] += 1
            return returncode, "exited_nonzero"
        self.counters["signal_exits"] += 1
        return 128 + abs(returncode), "exited_signal"

    def _final_status(
        self,
        state: str,
        exit_code: int,
        raw_returncode: Optional[int],
        extra: dict[str, Any],
        stdout_writer: BoundedRotatingWriter,
        stderr_writer: BoundedRotatingWriter,
        host_writer: BoundedRotatingWriter,
        recovery_writer: BoundedRotatingWriter,
        worker: Optional[HealthWorker],
    ) -> None:
        if self.run_dir is None:
            return
        if self.console is not None:
            self.console.close()
        status = {
            "console": {
                "enabled": self.console is not None,
                "dropped_bytes": self.console.dropped_bytes if self.console else 0,
                "drain_complete": not self.console.thread.is_alive() if self.console else True,
            },
            "schema_version": SCHEMA_VERSION,
            "kind": "flight_capture_status",
            "session_id": self.session_id,
            "final_clock": clock_sample(),
            "state": state,
            "exit_code": exit_code,
            "child_returncode": raw_returncode,
            "stop_signal": self.stop_signal,
            "capture": self.health.snapshot(),
            "counters": dict(self.counters),
            "writers": {
                "stdout": stdout_writer.snapshot(),
                "stderr": stderr_writer.snapshot(),
                "host": host_writer.snapshot(),
                "recovery": recovery_writer.snapshot(),
                "radio_health": self.radio_health_writer.snapshot() if self.radio_health_writer is not None else None,
            },
            "quotas": {
                "stdout_and_stderr": stdout_writer.quota.snapshot(),
                "host": host_writer.quota.snapshot(),
            },
            "health_worker": worker.snapshot() if worker is not None else None,
            "recovery": {
                "enabled": self.recovery_enabled,
                "stop_reason": self.recovery_stop_reason,
                "policy_decision": self.policy_decision.__dict__ if self.policy_decision is not None else None,
                "policy_snapshot": self.policy.snapshot(),
                "native_channel": getattr(self, "native_channel", None),
                "radio_health": self.radio_health.snapshot(),
            },
            "extra": extra,
        }
        write_json(self.run_dir / "status.json", status, self.health)
        health_state = self.health.snapshot()
        result = f"flight capture run={self.run_dir} state={state} exit={exit_code}"
        if not health_state["healthy"]:
            result += " capture_unhealthy=" + ",".join(health_state["reasons"])
        print(result, file=sys.stderr)



class RecoverySession:
    """Own sequential role-specific workers while keeping restrictions session-wide."""

    _MAX_RECENT_ATTEMPTS = 32
    _PROC_SCAN_LIMIT = 65536

    def __init__(self, args: argparse.Namespace, command: list[str]) -> None:
        self.args = args
        self.command = command
        self.policy = RecoveryPolicy(args.recovery_stall, args.recovery_attempt, args.role)
        self.latch = StopLatch()
        self.health = CaptureHealth()
        self.session_dir: Optional[Path] = None
        self.journal: Optional[BoundedRotatingWriter] = None
        self.attempt_count = 0
        self.dropped_attempt_records = 0
        self.recent_attempts: list[dict[str, Any]] = []
        self.current_attempt: Optional[dict[str, Any]] = None
        self.final_state = "starting"
        self.final_exit_code: Optional[int] = None
        self.group_fence_state: Optional[str] = None

    def _write(self, kind: str, **fields: Any) -> None:
        if self.journal is None or self.session_dir is None:
            return
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "session_id": self.session_dir.name,
            "receipt_clock": clock_sample(),
            "policy_generation": self.policy.snapshot()["generation"],
        }
        payload.update(fields)
        self.journal.write(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")

    def _write_status(self) -> None:
        if self.session_dir is None:
            return
        status = {
            "schema_version": SCHEMA_VERSION,
            "kind": "recovery_session_status",
            "session_id": self.session_dir.name,
            "final_clock": clock_sample(),
            "state": self.final_state,
            "exit_code": self.final_exit_code,
            "operator_stop_signal": self.latch.signal_number,
            "attempt_count": self.attempt_count,
            "dropped_attempt_records": self.dropped_attempt_records,
            "current_attempt": self.current_attempt,
            "recent_attempts": self.recent_attempts,
            "group_fence_state": self.group_fence_state,
            "policy": self.policy.snapshot(),
            "journal": self.journal.snapshot() if self.journal is not None else None,
            "capture": self.health.snapshot(),
        }
        write_json(self.session_dir / "status.json", status, self.health)

    def _open(self) -> Path:
        root = Path(self.args.output).resolve()
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(root, 0o700)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.session_dir = Path(tempfile.mkdtemp(prefix=f"{self.args.role}-session-{stamp}-{os.getpid()}-", dir=root))
        os.chmod(self.session_dir, 0o700)
        attempts_dir = self.session_dir / "attempts"
        attempts_dir.mkdir(mode=0o700)
        os.chmod(attempts_dir, 0o700)
        journal_limit = self.args.host_budget if self.args.host_budget else min(self.args.chunk_bytes, 4 * MIB)
        quota = ByteQuota(journal_limit)
        self.journal = BoundedRotatingWriter(
            self.session_dir,
            "recovery",
            quota,
            min(self.args.chunk_bytes, 4 * MIB),
            self.health,
        )
        self.journal.storage = StorageReserve(self.session_dir, self.args.min_free_bytes)
        write_json(
            self.session_dir / "metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "recovery_session_metadata",
                "session_id": self.session_dir.name,
                "role": self.args.role,
                "created_clock": clock_sample(),
                "command": redact_argv(self.command),
                "recovery": {
                    "stall_seconds": self.args.recovery_stall,
                    "attempt_seconds": self.args.recovery_attempt,
                    "no_total_start_budget": True,
                    "recent_attempt_record_limit": self._MAX_RECENT_ATTEMPTS,
                    "journal_byte_limit": journal_limit,
                },
            },
            self.health,
        )
        return self.session_dir

    @staticmethod
    def _operator_exit_code(signal_number: int) -> int:
        return 128 + abs(signal_number)

    @staticmethod
    def _attempt_dir(session_dir: Path, ordinal: int) -> Path:
        return session_dir / "attempts" / f"{ordinal:06d}"

    @classmethod
    def _owned_group_member_count(cls, group: int) -> Optional[int]:
        """Verify members remain in the session created by the worker leader."""

        members = 0
        scanned = 0
        try:
            entries = os.scandir("/proc")
        except OSError:
            return None
        with entries:
            for entry in entries:
                if not entry.name.isdigit():
                    continue
                scanned += 1
                if scanned > cls._PROC_SCAN_LIMIT:
                    return None
                try:
                    text = Path(entry.path, "stat").read_text(encoding="ascii")
                    fields = text[text.rfind(")") + 2 :].split()
                    process_group = int(fields[2])
                    session = int(fields[3])
                except FileNotFoundError:
                    # A process can exit between scandir and stat; it cannot
                    # remain an unfenced member after this point.
                    continue
                except (OSError, UnicodeDecodeError, ValueError, IndexError):
                    return None
                if process_group != group:
                    continue
                # start_new_session makes the leader's SID and PGID its PID.
                # A member from another session is outside the group we created.
                if session != group:
                    return None
                members += 1
        return members

    def _owned_group_state(self, capture: FlightCapture) -> str:
        if capture.process is None:
            return "not_started"
        group = capture.process.pid
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            return "already_absent"
        except PermissionError:
            return "ownership_unknown"
        current_start_ticks = FlightCapture._process_start_ticks(group)
        if current_start_ticks is not None:
            if capture.process_start_ticks is None or current_start_ticks != capture.process_start_ticks:
                return "pid_reused"
        members = self._owned_group_member_count(group)
        if members is None:
            return "ownership_unknown"
        if members:
            return "owned"
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            return "already_absent"
        except PermissionError:
            return "ownership_unknown"
        return "ownership_unknown"

    def _fence_owned_group(self, capture: FlightCapture) -> str:
        """Drain only the worker session we launched before a replacement."""

        initial = self._owned_group_state(capture)
        if initial != "owned":
            return initial
        assert capture.process is not None
        group = capture.process.pid
        graceful_signal = self.latch.signal_number or signal.SIGINT
        self._write(
            "recovery_group_fence",
            process_group=group,
            signal=graceful_signal,
            state="graceful_sent",
        )
        try:
            os.killpg(group, graceful_signal)
        except ProcessLookupError:
            return "already_absent"
        except PermissionError:
            return "ownership_unknown"
        deadline = time.monotonic_ns() + 10_000_000_000
        while time.monotonic_ns() < deadline:
            state = self._owned_group_state(capture)
            if state == "already_absent":
                return "fenced"
            if state != "owned":
                return state
            time.sleep(0.1)
        state = self._owned_group_state(capture)
        if state == "already_absent":
            return "fenced"
        if state != "owned":
            return state
        self._write("recovery_group_fence", process_group=group, signal=signal.SIGKILL, state="kill_sent")
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            return "fenced"
        except PermissionError:
            return "ownership_unknown"
        for _ in range(10):
            state = self._owned_group_state(capture)
            if state == "already_absent":
                return "fenced_after_kill"
            if state != "owned":
                return state
            time.sleep(0.1)
        return "cleanup_failed"

    def _worker_started(self, ordinal: int, worker_pid: int) -> None:
        if self.current_attempt is None or self.current_attempt["ordinal"] != ordinal:
            return
        self.current_attempt["state"] = "running"
        self.current_attempt["worker_pid"] = worker_pid
        self.current_attempt["worker_started_monotonic_ns"] = time.monotonic_ns()
        self._write(
            "recovery_attempt_worker_started",
            ordinal=ordinal,
            attempt_directory=self.current_attempt["attempt_directory"],
            worker_pid=worker_pid,
        )
        self._write_status()

    def _record_attempt(self, result: CaptureResult, fence_state: str) -> None:
        record = dict(self.current_attempt or {})
        record.update(
            {
                "state": "finished",
                "finished_monotonic_ns": time.monotonic_ns(),
                "exit_code": result.exit_code,
                "child_returncode": result.raw_returncode,
                "capture_state": result.state,
                "operator_stop": result.operator_stop,
                "recovery_stop_reason": result.recovery_stop_reason,
                "policy_decision": result.policy_decision.__dict__ if result.policy_decision is not None else None,
                "process_group": result.process_group,
                "group_fence_state": fence_state,
                "attempt_directory": str(result.run_dir),
            }
        )
        if len(self.recent_attempts) >= self._MAX_RECENT_ATTEMPTS:
            self.recent_attempts.pop(0)
            self.dropped_attempt_records += 1
        self.recent_attempts.append(record)
        self.current_attempt = None

    @staticmethod
    def _fence_failed(state: str) -> bool:
        return state in {"ownership_unknown", "pid_reused", "cleanup_failed"}

    def _wait_for_launch(self, planned_launch_ns: int) -> bool:
        self.final_state = "backoff"
        self.current_attempt = None
        self._write(
            "recovery_backoff",
            planned_launch_monotonic_ns=planned_launch_ns,
            remaining_ns=max(0, planned_launch_ns - time.monotonic_ns()),
            policy_snapshot=self.policy.snapshot(),
        )
        self._write_status()
        while time.monotonic_ns() < planned_launch_ns:
            if self.latch.signal_number is not None:
                return False
            time.sleep(0.1)
        return self.latch.signal_number is None

    def run(self) -> int:
        session_dir = self._open()
        previous_handlers = self.latch.install()
        active_capture: Optional[FlightCapture] = None
        try:
            self._write("recovery_session_started", no_total_start_budget=True, policy_snapshot=self.policy.snapshot())
            self._write_status()
            planned_launch_ns = time.monotonic_ns()
            while True:
                if self.latch.signal_number is not None:
                    self.final_state = "operator_stop"
                    self.final_exit_code = self._operator_exit_code(self.latch.signal_number)
                    self._write("recovery_session_finished", reason="operator_stop", exit_code=self.final_exit_code)
                    break
                if not self._wait_for_launch(planned_launch_ns):
                    assert self.latch.signal_number is not None
                    self.final_state = "operator_stop"
                    self.final_exit_code = self._operator_exit_code(self.latch.signal_number)
                    self._write("recovery_session_finished", reason="operator_stop_during_backoff", exit_code=self.final_exit_code)
                    break

                actual_launch_ns = time.monotonic_ns()
                self.attempt_count += 1
                ordinal = self.attempt_count
                attempt_dir = self._attempt_dir(session_dir, ordinal)
                attempt_dir.mkdir(mode=0o700)
                os.chmod(attempt_dir, 0o700)
                # The policy generation belongs to this accepted worker launch. It
                # is intentionally not consumed when the session object is created.
                self.policy.begin_attempt(time.monotonic_ns())
                self.current_attempt = {
                    "ordinal": ordinal,
                    "state": "launching",
                    "attempt_directory": str(attempt_dir),
                    "planned_launch_monotonic_ns": planned_launch_ns,
                    "actual_launch_monotonic_ns": actual_launch_ns,
                    "launch_delay_ns": max(0, actual_launch_ns - planned_launch_ns),
                    "worker_pid": None,
                }
                self.final_state = "running"
                self._write("recovery_attempt_started", **self.current_attempt, policy_snapshot=self.policy.snapshot())
                self._write_status()
                active_capture = FlightCapture(
                    self.args,
                    self.command,
                    run_dir=attempt_dir,
                    session_id=session_dir.name,
                    policy=self.policy,
                    begin_policy_attempt=False,
                    stop_latch=self.latch,
                    recovery_enabled=True,
                    launch_observer=lambda pid, value=ordinal: self._worker_started(value, pid),
                )
                active_capture.run()
                result = active_capture.last_result
                if result is None:
                    self.final_state = "capture_failure"
                    self.final_exit_code = 125
                    self._write("recovery_session_finished", reason="capture_result_unavailable", exit_code=125)
                    break
                fence_state = self._fence_owned_group(active_capture)
                self.group_fence_state = fence_state
                self._record_attempt(result, fence_state)
                self._write(
                    "recovery_attempt_finished",
                    ordinal=ordinal,
                    attempt_directory=str(result.run_dir),
                    worker_pid=result.process_group,
                    exit_code=result.exit_code,
                    child_returncode=result.raw_returncode,
                    recovery_stop_reason=result.recovery_stop_reason,
                    group_fence_state=fence_state,
                    policy_decision=result.policy_decision.__dict__ if result.policy_decision is not None else None,
                )
                active_capture = None
                if self._fence_failed(fence_state):
                    self.final_state = "cleanup_failed"
                    self.final_exit_code = 125
                    self._write("recovery_session_finished", reason=fence_state, exit_code=125)
                    break
                if self.latch.signal_number is not None or result.operator_stop:
                    signal_number = self.latch.signal_number or signal.SIGINT
                    self.final_state = "operator_stop"
                    self.final_exit_code = self._operator_exit_code(signal_number)
                    self._write("recovery_session_finished", reason="operator_stop", exit_code=self.final_exit_code)
                    break
                decision = result.policy_decision
                if decision is None or decision.action != "retry":
                    self.final_state = "policy_stop"
                    self.final_exit_code = result.exit_code
                    self._write(
                        "recovery_session_finished",
                        reason=decision.reason if decision is not None else "missing_policy_decision",
                        exit_code=self.final_exit_code,
                        policy_snapshot=self.policy.snapshot(),
                    )
                    break
                planned_launch_ns = max(time.monotonic_ns(), decision.not_before_ns)
                self._write_status()
        except (CaptureError, OSError) as exc:
            self.final_state = "session_error"
            self.final_exit_code = 125
            self._write("recovery_session_finished", reason=exc.__class__.__name__, exit_code=125)
        finally:
            if active_capture is not None:
                fence_state = self._fence_owned_group(active_capture)
                self.group_fence_state = fence_state
                if self._fence_failed(fence_state):
                    self.final_state = "cleanup_failed"
                    self.final_exit_code = 125
                    self._write("recovery_session_finished", reason=fence_state, exit_code=125)
            if self.final_exit_code is None:
                self.final_state = "session_error"
                self.final_exit_code = 125
            self._write_status()
            if self.journal is not None:
                self.journal.close()
            self.latch.restore(previous_handlers)
        return self.final_exit_code

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = validate_args(args, parser)
    try:
        if args.recovery:
            return RecoverySession(args, command).run()
        return FlightCapture(args, command).run()
    except CaptureError as exc:
        print(f"flight capture error: {exc}", file=sys.stderr)
        return 125
    except OSError as exc:
        print(f"flight capture error: {exc.__class__.__name__}", file=sys.stderr)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
