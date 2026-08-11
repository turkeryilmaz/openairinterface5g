#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Bounded file and child-process primitives for the R0 test harness."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import os
import selectors
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path


_READ_CHUNK_BYTES = 64 << 10
_SELECT_INTERVAL_SECONDS = 0.05
_CLEANUP_TIMEOUT_SECONDS = 2.0
_PROC_STAT_MAX_BYTES = 4096
_PROC_SCAN_MAX_ENTRIES = 131072
_MAX_REPORTED_DESCENDANT_PIDS = 16

_KNOWN_LOADER_ENVIRONMENT_VARIABLES = frozenset(
    {
        "GLIBC_TUNABLES",
        "LD_ASSUME_KERNEL",
        "LD_AUDIT",
        "LD_BIND_NOT",
        "LD_BIND_NOW",
        "LD_DEBUG",
        "LD_DEBUG_OUTPUT",
        "LD_DYNAMIC_WEAK",
        "LD_HWCAP_MASK",
        "LD_LIBRARY_PATH",
        "LD_ORIGIN_PATH",
        "LD_POINTER_GUARD",
        "LD_PREFER_MAP_32BIT_EXEC",
        "LD_PRELOAD",
        "LD_PROFILE",
        "LD_PROFILE_OUTPUT",
        "LD_SHOW_AUXV",
        "LD_TRACE_LOADED_OBJECTS",
        "LD_TRACE_PRELINKING",
        "LD_USE_LOAD_BIAS",
        "LD_VERBOSE",
        "LD_WARN",
    }
)


class BoundedFileError(RuntimeError):
    """A file could not be read as one stable, bounded regular-file image."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclasses.dataclass(frozen=True)
class BoundedFileImage:
    data: bytes
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    sha256: str


@dataclasses.dataclass(frozen=True)
class BoundedProcessResult:
    command: tuple[str, ...]
    pid: int | None
    returncode: int | None
    stdout: bytes
    stderr: bytes
    failure: str | None
    killpg_attempted: bool
    cleanup_complete: bool


@dataclasses.dataclass(frozen=True)
class _ProcIdentity:
    pid: int
    state: str
    process_group: int
    session: int
    start_time_ticks: int


@dataclasses.dataclass(frozen=True)
class _CleanupOutcome:
    complete: bool
    kill_attempted: bool
    unexpected_descendant_count: int
    unexpected_descendant_pids: tuple[int, ...]
    error: str | None


class _ContainmentError(RuntimeError):
    pass


def known_loader_environment_variables() -> tuple[str, ...]:
    """Return the frozen loader-control catalog, independent of ambient state."""

    return tuple(sorted(_KNOWN_LOADER_ENVIRONMENT_VARIABLES))


def loader_environment_variables() -> tuple[str, ...]:
    """Return every known or currently present dynamic-loader control name."""

    return tuple(
        sorted(set(known_loader_environment_variables()) | {name for name in os.environ if name.startswith("LD_")})
    )


def format_byte_evidence(label: str, data: bytes) -> str:
    """Render bounded bytes losslessly as one printable-ASCII evidence record."""

    if not label or any(not 0x20 <= ord(character) <= 0x7E for character in label):
        raise ValueError("byte-evidence label must be nonempty printable ASCII")
    encoded = base64.b64encode(data).decode("ascii")
    digest = hashlib.sha256(data).hexdigest()
    return f"{label} bytes={len(data)} sha256={digest} base64={encoded}"


def read_regular_file_bounded(path: Path, max_bytes: int) -> BoundedFileImage:
    """Open once and read one unchanged regular-file image with a hard size cap."""

    if max_bytes < 0:
        raise ValueError("max_bytes must be nonnegative")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BoundedFileError("READ", f"{path}: {error}") from error

    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BoundedFileError("INPUT_TYPE", f"{path}: input is not a regular file")
        if before.st_size > max_bytes:
            raise BoundedFileError("RAW_SIZE", f"{path}: exceeds {max_bytes} bytes")

        chunks: list[bytes] = []
        total = 0
        while True:
            try:
                chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, max_bytes - total + 1))
            except BlockingIOError as error:
                raise BoundedFileError("READ_BLOCKED", f"{path}: regular-file read would block") from error
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise BoundedFileError("RAW_SIZE", f"{path}: grew beyond {max_bytes} bytes while reading")

        after = os.fstat(descriptor)
        before_identity = (before.st_dev, before.st_ino)
        after_identity = (after.st_dev, after.st_ino)
        before_version = (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        after_version = (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if before_identity != after_identity or before_version != after_version or total != after.st_size:
            raise BoundedFileError("INPUT_CHANGED", f"{path}: input changed while it was being read")

        data = b"".join(chunks)
        return BoundedFileImage(
            data=data,
            device=after.st_dev,
            inode=after.st_ino,
            size=total,
            mtime_ns=after.st_mtime_ns,
            ctime_ns=after.st_ctime_ns,
            sha256=hashlib.sha256(data).hexdigest(),
        )
    except BoundedFileError:
        raise
    except OSError as error:
        raise BoundedFileError("READ", f"{path}: {error}") from error
    finally:
        os.close(descriptor)


def _kill_process_group(process_group: int) -> bool:
    """Attempt SIGKILL even when the original process-group leader has exited."""

    try:
        os.killpg(process_group, signal.SIGKILL)
        return True
    except ProcessLookupError:
        return True
    except OSError:
        return False


def _leader_exited_unreaped(pidfd: int) -> bool:
    try:
        status = os.waitid(os.P_PIDFD, pidfd, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except (ChildProcessError, OSError) as error:
        raise _ContainmentError(f"cannot inspect leader pidfd: {error}") from error
    return status is not None


def _read_proc_identity(pid: int) -> _ProcIdentity | None:
    path = f"/proc/{pid}/stat"
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK)
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as error:
        raise _ContainmentError(f"cannot open {path}: {error}") from error

    try:
        data = bytearray()
        while len(data) <= _PROC_STAT_MAX_BYTES:
            try:
                chunk = os.read(descriptor, _PROC_STAT_MAX_BYTES + 1 - len(data))
            except BlockingIOError as error:
                raise _ContainmentError(f"nonblocking read would block for {path}") from error
            if not chunk:
                break
            data.extend(chunk)
        if len(data) > _PROC_STAT_MAX_BYTES:
            raise _ContainmentError(f"{path} exceeds {_PROC_STAT_MAX_BYTES} bytes")
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as error:
        raise _ContainmentError(f"cannot read {path}: {error}") from error
    finally:
        os.close(descriptor)

    closing = data.rfind(b") ")
    opening = data.find(b"(")
    if opening <= 0 or closing <= opening:
        raise _ContainmentError(f"malformed {path}")
    try:
        recorded_pid = int(data[:opening].strip(), 10)
        fields = data[closing + 2 :].split()
        state = fields[0].decode("ascii")
        process_group = int(fields[2], 10)
        session = int(fields[3], 10)
        start_time_ticks = int(fields[19], 10)
    except (IndexError, UnicodeDecodeError, ValueError) as error:
        raise _ContainmentError(f"malformed {path}: {error}") from error
    if recorded_pid != pid or len(state) != 1 or process_group < 0 or session < 0 or start_time_ticks <= 0:
        raise _ContainmentError(f"invalid identity fields in {path}")
    return _ProcIdentity(pid, state, process_group, session, start_time_ticks)


def _scan_session(session: int, leader_pid: int, deadline: float) -> tuple[list[_ProcIdentity], int]:
    live: list[_ProcIdentity] = []
    zombies = 0
    entries = 0
    try:
        proc_entries = os.scandir("/proc")
    except OSError as error:
        raise _ContainmentError(f"cannot scan /proc: {error}") from error
    with proc_entries:
        for entry in proc_entries:
            if time.monotonic() >= deadline:
                raise _ContainmentError("process-session scan exceeded cleanup deadline")
            if not entry.name.isascii() or not entry.name.isdigit():
                continue
            entries += 1
            if entries > _PROC_SCAN_MAX_ENTRIES:
                raise _ContainmentError(f"/proc scan exceeds {_PROC_SCAN_MAX_ENTRIES} process entries")
            identity = _read_proc_identity(int(entry.name, 10))
            if identity is None or identity.pid == leader_pid or identity.session != session:
                continue
            if identity.state == "Z":
                zombies += 1
            else:
                live.append(identity)
    return live, zombies


def _signal_identity(identity: _ProcIdentity) -> bool:
    try:
        pidfd = os.pidfd_open(identity.pid, 0)
    except (FileNotFoundError, ProcessLookupError):
        return False
    except OSError as error:
        raise _ContainmentError(f"cannot open pidfd for descendant {identity.pid}: {error}") from error
    try:
        current = _read_proc_identity(identity.pid)
        if current is None or current != identity or current.state == "Z":
            return False
        try:
            signal.pidfd_send_signal(pidfd, signal.SIGKILL, None, 0)
        except ProcessLookupError:
            return False
        except OSError as error:
            raise _ContainmentError(f"cannot signal descendant {identity.pid}: {error}") from error
        return True
    finally:
        os.close(pidfd)


def _drain_ready(
    selector: selectors.BaseSelector,
    buffers: dict[int, bytearray],
    limits: Mapping[int, int],
    labels: Mapping[int, str],
    timeout_seconds: float,
    *,
    discard_excess: bool = False,
) -> str | None:
    for key, _ in selector.select(timeout_seconds):
        descriptor = key.fd
        destination = buffers[descriptor]
        limit = limits[descriptor]
        try:
            read_size = _READ_CHUNK_BYTES if discard_excess else min(_READ_CHUNK_BYTES, limit - len(destination) + 1)
            chunk = os.read(descriptor, read_size)
        except BlockingIOError:
            continue
        except OSError as error:
            return f"could not read child {labels[descriptor]}: {error}"
        if not chunk:
            selector.unregister(descriptor)
            continue
        available = limit - len(destination)
        destination.extend(chunk[:available])
        if len(chunk) > available and not discard_excess:
            return f"{labels[descriptor]} exceeded {limit} bytes"
    return None


def _cleanup_spawned_process(
    process: subprocess.Popen[bytes],
    pidfd: int,
    selector: selectors.BaseSelector | None,
    buffers: dict[int, bytearray],
    limits: Mapping[int, int],
    labels: Mapping[int, str],
    *,
    force_kill: bool,
) -> _CleanupOutcome:
    deadline = time.monotonic() + _CLEANUP_TIMEOUT_SECONDS
    kill_attempted = force_kill
    unexpected_count = 0
    unexpected_pids: list[int] = []
    containment_error: str | None = None
    session = process.pid

    def signal_leader_and_group() -> None:
        nonlocal containment_error
        try:
            if not _leader_exited_unreaped(pidfd):
                signal.pidfd_send_signal(pidfd, signal.SIGKILL, None, 0)
        except ProcessLookupError:
            pass
        except (OSError, _ContainmentError) as error:
            containment_error = f"cannot signal leader: {error}"
        if not _kill_process_group(session) and containment_error is None:
            containment_error = f"cannot signal process group {session}"

    if force_kill:
        signal_leader_and_group()

    quiescent = False
    pipes_complete = selector is None
    while time.monotonic() < deadline:
        if selector is not None and selector.get_map():
            drain_error = _drain_ready(selector, buffers, limits, labels, 0.0, discard_excess=True)
            if drain_error is not None and containment_error is None:
                containment_error = drain_error
        pipes_complete = selector is None or not selector.get_map()

        try:
            leader_exited = _leader_exited_unreaped(pidfd)
            live, _ = _scan_session(session, process.pid, deadline)
        except _ContainmentError as error:
            containment_error = str(error)
            signal_leader_and_group()
            break

        if live and unexpected_count == 0:
            unexpected_count = len(live)
            unexpected_pids.extend(identity.pid for identity in live[:_MAX_REPORTED_DESCENDANT_PIDS])
        if live and not kill_attempted:
            kill_attempted = True
            signal_leader_and_group()

        if kill_attempted:
            for identity in live:
                if time.monotonic() >= deadline:
                    containment_error = "descendant signaling exceeded cleanup deadline"
                    break
                try:
                    _signal_identity(identity)
                except _ContainmentError as error:
                    containment_error = str(error)
                    break

        if leader_exited and not live and pipes_complete:
            quiescent = True
            break

        if selector is not None and selector.get_map():
            remaining = max(0.0, min(_SELECT_INTERVAL_SECONDS, deadline - time.monotonic()))
            drain_error = _drain_ready(selector, buffers, limits, labels, remaining, discard_excess=True)
            if drain_error is not None and containment_error is None:
                containment_error = drain_error
        else:
            time.sleep(max(0.0, min(_SELECT_INTERVAL_SECONDS, deadline - time.monotonic())))

    reaped = False
    try:
        process.wait(timeout=max(0.0, deadline - time.monotonic()))
        reaped = True
    except subprocess.TimeoutExpired:
        if containment_error is None:
            containment_error = "leader could not be reaped before cleanup deadline"
    except OSError as error:
        if containment_error is None:
            containment_error = f"cannot reap leader: {error}"

    complete = quiescent and reaped and containment_error is None
    return _CleanupOutcome(
        complete,
        kill_attempted,
        unexpected_count,
        tuple(unexpected_pids),
        containment_error,
    )


def _cleanup_spawned_process_without_pidfd(process: subprocess.Popen[bytes]) -> _CleanupOutcome:
    """Best-effort fail-closed cleanup when pidfd initialization itself failed."""

    deadline = time.monotonic() + _CLEANUP_TIMEOUT_SECONDS
    kill_attempted = True
    unexpected_count = 0
    unexpected_pids: list[int] = []
    containment_error: str | None = None

    if not _kill_process_group(process.pid):
        containment_error = f"cannot signal process group {process.pid}"

    quiescent = False
    while time.monotonic() < deadline:
        try:
            live, _ = _scan_session(process.pid, process.pid, deadline)
        except _ContainmentError as error:
            containment_error = str(error)
            break
        if live and unexpected_count == 0:
            unexpected_count = len(live)
            unexpected_pids.extend(identity.pid for identity in live[:_MAX_REPORTED_DESCENDANT_PIDS])
        for identity in live:
            if time.monotonic() >= deadline:
                containment_error = "descendant signaling exceeded cleanup deadline"
                break
            try:
                _signal_identity(identity)
            except _ContainmentError as error:
                containment_error = str(error)
                break
        if containment_error is not None:
            break
        if not live:
            quiescent = True
            break
        time.sleep(max(0.0, min(_SELECT_INTERVAL_SECONDS, deadline - time.monotonic())))

    reaped = False
    try:
        process.wait(timeout=max(0.0, deadline - time.monotonic()))
        reaped = True
    except subprocess.TimeoutExpired:
        if containment_error is None:
            containment_error = "leader could not be reaped before cleanup deadline"
    except OSError as error:
        if containment_error is None:
            containment_error = f"cannot reap leader: {error}"

    return _CleanupOutcome(
        quiescent and reaped and containment_error is None,
        kill_attempted,
        unexpected_count,
        tuple(unexpected_pids),
        containment_error,
    )


def run_process_bounded(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    max_stdout_bytes: int,
    max_stderr_bytes: int,
    remove_environment: Sequence[str] = (),
    environment_updates: Mapping[str, str] | None = None,
) -> BoundedProcessResult:
    """Run one Linux process session with bounded output and verified quiescence.

    The session leader remains unreaped while procfs identities are inspected and
    signalled through pidfds.  A child that deliberately escapes with ``setsid``
    is outside this harness's containment claim, but cannot make this call wait
    beyond its declared execution and cleanup bounds.
    """

    frozen_command = tuple(str(part) for part in command)
    if not frozen_command:
        raise ValueError("command must not be empty")
    if not 0.0 < timeout_seconds <= 60.0:
        raise ValueError("timeout_seconds must be in (0, 60]")
    if max_stdout_bytes < 0 or max_stderr_bytes < 0:
        raise ValueError("output limits must be nonnegative")

    environment = os.environ.copy()
    for variable in remove_environment:
        environment.pop(variable, None)
    if environment_updates is not None:
        environment.update(environment_updates)

    deadline = time.monotonic() + timeout_seconds
    try:
        process = subprocess.Popen(
            frozen_command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            close_fds=True,
            start_new_session=True,
        )
    except OSError as error:
        return BoundedProcessResult(
            frozen_command,
            None,
            None,
            b"",
            b"",
            f"cannot start {frozen_command[0]}: {error}",
            False,
            True,
        )

    selector: selectors.BaseSelector | None = None
    pidfd: int | None = None
    stdout_fd: int | None = None
    stderr_fd: int | None = None
    buffers: dict[int, bytearray] = {}
    limits: dict[int, int] = {}
    labels: dict[int, str] = {}
    failure: str | None = None
    cleanup_outcome: _CleanupOutcome | None = None
    observed_descendant_pids: tuple[int, ...] = ()
    try:
        pidfd = os.pidfd_open(process.pid, 0)
        if process.stdout is None or process.stderr is None:
            raise _ContainmentError("Popen did not create both bounded output pipes")
        stdout_fd = process.stdout.fileno()
        stderr_fd = process.stderr.fileno()
        selector = selectors.DefaultSelector()
        buffers = {stdout_fd: bytearray(), stderr_fd: bytearray()}
        limits = {stdout_fd: max_stdout_bytes, stderr_fd: max_stderr_bytes}
        labels = {stdout_fd: "stdout", stderr_fd: "stderr"}
        try:
            for descriptor in (stdout_fd, stderr_fd):
                os.set_blocking(descriptor, False)
                selector.register(descriptor, selectors.EVENT_READ)
        except (OSError, KeyError, ValueError) as error:
            failure = f"cannot initialize bounded child pipes: {error}"

        while True:
            if failure is not None:
                break
            leader_exited = _leader_exited_unreaped(pidfd)
            if leader_exited:
                try:
                    live, _ = _scan_session(process.pid, process.pid, deadline)
                except _ContainmentError as error:
                    failure = f"bounded execution containment failed: {error}"
                    break
                if live:
                    observed_descendant_pids = tuple(
                        identity.pid for identity in live[:_MAX_REPORTED_DESCENDANT_PIDS]
                    )
                    failure = (
                        "unexpected live same-session descendants after leader exit "
                        f"count={len(live)} pids={','.join(str(pid) for pid in observed_descendant_pids)}"
                    )
                    break
                if not selector.get_map():
                    break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure = f"timeout after {timeout_seconds:g} seconds"
                break
            if selector.get_map():
                failure = _drain_ready(selector, buffers, limits, labels, min(remaining, _SELECT_INTERVAL_SECONDS))
                if failure is not None:
                    break
            else:
                time.sleep(min(remaining, _SELECT_INTERVAL_SECONDS))

        cleanup_outcome = _cleanup_spawned_process(
            process,
            pidfd,
            selector,
            buffers,
            limits,
            labels,
            force_kill=failure is not None,
        )
        if cleanup_outcome.unexpected_descendant_count and not observed_descendant_pids:
            observed_descendant_pids = cleanup_outcome.unexpected_descendant_pids
            descendant_failure = (
                "unexpected live same-session descendants "
                f"count={cleanup_outcome.unexpected_descendant_count} "
                f"pids={','.join(str(pid) for pid in observed_descendant_pids)}"
            )
            failure = descendant_failure if failure is None else f"{failure}; {descendant_failure}"
        if cleanup_outcome.error is not None:
            cleanup_failure = f"process-session cleanup failed: {cleanup_outcome.error}"
            failure = cleanup_failure if failure is None else f"{failure}; {cleanup_failure}"
        if not cleanup_outcome.complete:
            suffix = "process-session cleanup could not be verified"
            failure = suffix if failure is None else f"{failure}; {suffix}"

        return BoundedProcessResult(
            command=frozen_command,
            pid=process.pid,
            returncode=process.returncode,
            stdout=bytes(buffers[stdout_fd]),
            stderr=bytes(buffers[stderr_fd]),
            failure=failure,
            killpg_attempted=cleanup_outcome.kill_attempted,
            cleanup_complete=cleanup_outcome.complete,
        )
    except BaseException as error:
        if cleanup_outcome is None:
            try:
                if pidfd is None:
                    cleanup_outcome = _cleanup_spawned_process_without_pidfd(process)
                else:
                    cleanup_outcome = _cleanup_spawned_process(
                        process,
                        pidfd,
                        selector,
                        buffers,
                        limits,
                        labels,
                        force_kill=True,
                    )
            except BaseException as cleanup_error:
                error.add_note(f"R0 bounded cleanup raised {type(cleanup_error).__name__}: {cleanup_error}")
                if process.returncode is None:
                    try:
                        cleanup_outcome = _cleanup_spawned_process_without_pidfd(process)
                    except BaseException as fallback_error:
                        error.add_note(
                            "R0 bounded fallback cleanup raised "
                            f"{type(fallback_error).__name__}: {fallback_error}"
                        )
                        try:
                            _kill_process_group(process.pid)
                            process.wait(timeout=_CLEANUP_TIMEOUT_SECONDS)
                        except BaseException:
                            pass
                else:
                    error.add_note("R0 bounded fallback cleanup skipped because the leader was already reaped")
        if cleanup_outcome is not None:
            error.add_note(
                "R0 bounded cleanup "
                f"complete={int(cleanup_outcome.complete)} "
                f"kill_attempted={int(cleanup_outcome.kill_attempted)} "
                f"descendants={cleanup_outcome.unexpected_descendant_count} "
                f"error={cleanup_outcome.error or 'none'}"
            )
        raise
    finally:
        active_error = sys.exc_info()[1]
        close_errors: list[tuple[str, BaseException]] = []
        closers = (
            ("selector", selector.close if selector is not None else None),
            ("stdout", process.stdout.close if process.stdout is not None else None),
            ("stderr", process.stderr.close if process.stderr is not None else None),
            ("pidfd", (lambda: os.close(pidfd)) if pidfd is not None else None),
        )
        for label, close in closers:
            if close is None:
                continue
            try:
                close()
            except BaseException as close_error:
                close_errors.append((label, close_error))
        if close_errors:
            detail = ", ".join(
                f"{label}={type(close_error).__name__}:{close_error}" for label, close_error in close_errors
            )
            note = f"R0 bounded descriptor close errors after cleanup: {detail}"
            if active_error is not None:
                active_error.add_note(note)
            else:
                close_error = close_errors[0][1]
                if cleanup_outcome is not None:
                    close_error.add_note(
                        "R0 bounded cleanup "
                        f"complete={int(cleanup_outcome.complete)} "
                        f"kill_attempted={int(cleanup_outcome.kill_attempted)} "
                        f"descendants={cleanup_outcome.unexpected_descendant_count} "
                        f"error={cleanup_outcome.error or 'none'}"
                    )
                close_error.add_note(note)
                raise close_error
