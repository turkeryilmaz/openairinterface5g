#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Bounded Linux process execution for the OAI memory-profiler controller.

The runner deliberately accepts an explicit argument vector, working directory,
and complete environment.  It streams the two output channels into distinct
exclusive regular files, periodically enforces caller-supplied thermal and
storage limits, and does not return until all-descendant cleanup is verified.

Containment is deliberately Linux-specific and fail closed.  One single-
threaded, child-free controller process enables ``PR_SET_CHILD_SUBREAPER``
around one command.  Bounded procfs snapshots identify the PPID-transitive
command tree, including descendants which call ``setsid(2)`` and orphans
reparented directly to the controller.  Stable PID/start-time identities and
pidfds are used for signalling; exact direct-child waits and repeated complete
empty snapshots establish terminal quiescence.

The finite supported cancellation set is masked only across child publication
and terminal cleanup.  A one-thread-only child pre-exec hook restores the exact
caller mask after setting those signals to their default dispositions.  That
hook deliberately disables the ``posix_spawn`` fast path; this is cold
controller code, never an OAI hot path.  The parent accepts only callable
supported-signal dispositions, binds their exact identities before containment,
and invokes a selected captured handler directly after restoring the caller
mask.  Arbitrary programmatic exceptions, unbounded signal arrival,
``SIGKILL``, controller crash, OOM, and handlers which call ``_exit``/``exec``
or otherwise bypass Python unwinding remain outside this containment claim.

Each supported-signal drain takes exactly one ``sigpending(2)`` snapshot while
the set remains blocked.  That distinct signal-number set is the operation's
linearization boundary: standard-signal multiplicity within an observed pending
bit is unknowable, no retry loop is used, and a newly pending number not
coalesced into that observed bit is a post-boundary caller event.
"""

from __future__ import annotations

import dataclasses
import ctypes
import enum
import hashlib
import math
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence


class ProcessErrorCode(str, enum.Enum):
    INVALID_ARGUMENT = "invalid_argument"
    PATH_INVALID = "path_invalid"
    OUTPUT_CREATE_FAILED = "output_create_failed"
    OUTPUT_WRITE_FAILED = "output_write_failed"
    SAFETY_READ_FAILED = "safety_read_failed"
    THERMAL_LIMIT = "thermal_limit"
    FREE_SPACE_LIMIT = "free_space_limit"
    SAFETY_SAMPLE_LIMIT = "safety_sample_limit"
    SPAWN_FAILED = "spawn_failed"
    PIDFD_FAILED = "pidfd_failed"
    PIPE_FAILED = "pipe_failed"
    STDOUT_LIMIT = "stdout_limit"
    STDERR_LIMIT = "stderr_limit"
    WALL_TIME_LIMIT = "wall_time_limit"
    PROCFS_LIMIT = "procfs_limit"
    DESCENDANT_LIMIT = "descendant_limit"
    DESCENDANT_IDENTITY_LIMIT = "descendant_identity_limit"
    DESCENDANT_REMAINS = "descendant_remains"
    RUNNER_BUSY = "runner_busy"
    CALLER_MULTITHREADED = "caller_multithreaded"
    PREEXISTING_CHILD = "preexisting_child"
    SIGNAL_MASK_PRECONDITION = "signal_mask_precondition"
    SIGNAL_MASK_FAILED = "signal_mask_failed"
    SIGNAL_DISPOSITION_PRECONDITION = "signal_disposition_precondition"
    SIGNAL_DISPOSITION_DRIFT = "signal_disposition_drift"
    CANCELLATION_HANDLER_RETURNED = "cancellation_handler_returned"
    SUBREAPER_FAILED = "subreaper_failed"
    CONTAINMENT_POISONED = "containment_poisoned"
    CONTAINMENT_FAILED = "containment_failed"
    CLEANUP_TIMEOUT = "cleanup_timeout"
    CLEANUP_UNVERIFIED = "cleanup_unverified"
    PROCESS_EXIT_NONZERO = "process_exit_nonzero"


class ProcessError(RuntimeError):
    """Typed setup or safety-observation failure."""

    def __init__(
        self,
        code: ProcessErrorCode,
        operation: str,
        detail: str,
        path: Path | None = None,
        *,
        sample: SafetySample | None = None,
    ) -> None:
        super().__init__(f"{operation}: {detail}")
        self.code = code
        self.operation = operation
        self.detail = detail
        self.path = path
        self.sample = sample


@dataclasses.dataclass(frozen=True)
class ThermalSensor:
    path: Path
    ceiling_millicelsius: int
    minimum_plausible_millicelsius: int


@dataclasses.dataclass(frozen=True)
class SafetySpec:
    sensors: tuple[ThermalSensor, ...]
    storage_path: Path
    minimum_free_bytes: int
    poll_interval_seconds: float
    max_samples: int
    sensor_max_bytes: int


SafetyPolicy = SafetySpec


@dataclasses.dataclass(frozen=True)
class SafetySample:
    monotonic_ns: int
    thermal_millicelsius: tuple[tuple[str, int], ...]
    free_bytes: int


@dataclasses.dataclass(frozen=True)
class CommandLimits:
    wall_seconds: float
    stdout_bytes: int
    stderr_bytes: int
    cleanup_seconds: float
    max_proc_entries: int
    max_observed_live_descendants: int
    max_descendant_identities: int
    read_chunk_bytes: int


ProcessLimits = CommandLimits


@dataclasses.dataclass(frozen=True)
class CommandSpec:
    argv: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]
    stdout_path: Path
    stderr_path: Path


@dataclasses.dataclass(frozen=True)
class ProcessResult:
    argv: tuple[str, ...]
    cwd: str
    pid: int | None
    returncode: int | None
    started_monotonic_ns: int
    ended_monotonic_ns: int
    stdout_bytes: int
    stderr_bytes: int
    stdout_sha256: str
    stderr_sha256: str
    safety_samples: tuple[SafetySample, ...]
    failure_code: ProcessErrorCode | None
    failure_detail: str | None
    kill_attempted: bool
    cleanup_complete: bool
    cleanup_failure_code: ProcessErrorCode | None
    cleanup_failure_detail: str | None
    unexpected_descendant_identity_lower_bound: int
    unexpected_descendant_identity_complete: bool
    unexpected_descendant_pids: tuple[int, ...]
    maximum_observed_live_descendant_count: int = 0
    observed_descendant_identity_lower_bound: int = 0
    observed_descendant_identity_complete: bool = True
    observed_descendant_identities: tuple[tuple[int, int], ...] = ()
    observed_descendant_identities_truncated: bool = False
    unexpected_descendant_identities: tuple[tuple[int, int], ...] = ()
    unexpected_descendant_identities_truncated: bool = False
    caller_signal_mask_numbers: tuple[int, ...] = ()
    signal_mask_restored: bool = False
    deferred_supported_signal_numbers: tuple[int, ...] = ()
    containment_mode: str = "linux_subreaper_ppid_pidfd_v1"
    containment_caller_thread_count: int = 0
    containment_preexisting_child_count: int = 0
    containment_subreaper_previously_enabled: bool = False
    containment_subreaper_restored: bool = False
    descendant_scan_count: int = 0
    first_descendant_scan_monotonic_ns: int | None = None
    last_descendant_scan_monotonic_ns: int | None = None
    maximum_observed_descendant_scan_gap_ns: int | None = None


@dataclasses.dataclass(frozen=True)
class _ProcIdentity:
    pid: int
    parent_pid: int
    state: str
    process_group: int
    session: int
    start_time_ticks: int


@dataclasses.dataclass(frozen=True)
class _CleanupResult:
    complete: bool
    kill_attempted: bool
    observed_identities: tuple[_ProcIdentity, ...]
    residual_identities: tuple[_ProcIdentity, ...]
    observed_identity_lower_bound: int
    observed_identity_complete: bool
    residual_identity_lower_bound: int
    residual_identity_complete: bool
    maximum_observed_live_descendant_count: int
    code: ProcessErrorCode | None
    detail: str | None


@dataclasses.dataclass(frozen=True)
class _ContainmentLease:
    owner_pid: int
    caller_thread_count: int
    preexisting_child_count: int
    previous_subreaper: bool


@dataclasses.dataclass
class _SignalContract:
    caller_mask_numbers: tuple[int, ...]
    supported_signal_handlers: tuple[
        tuple[int, Callable[[int, object | None], object]], ...
    ] = ()
    blocked: bool = False
    deferred_supported_signal_numbers: set[int] = dataclasses.field(
        default_factory=set
    )
    drain_events: list[_SignalDrainEvent] = dataclasses.field(
        default_factory=list
    )
    noted_drain_event_count: int = 0


@dataclasses.dataclass(frozen=True)
class _SignalDrainEvent:
    boundary: str
    observed: tuple[int, ...]
    selected: int | None
    coalesced: tuple[int, ...]


@dataclasses.dataclass
class _PopenHolder:
    process: subprocess.Popen[bytes] | None = None


@dataclasses.dataclass
class _ScanCadence:
    count: int = 0
    first_monotonic_ns: int | None = None
    last_monotonic_ns: int | None = None
    maximum_gap_ns: int | None = None

    def record(self) -> None:
        observed_ns = _monotonic_ns()
        if self.last_monotonic_ns is not None:
            gap_ns = observed_ns - self.last_monotonic_ns
            if gap_ns < 0:
                raise ProcessError(
                    ProcessErrorCode.CONTAINMENT_FAILED,
                    "descendant-scan-cadence",
                    "monotonic clock moved backwards",
                )
            self.maximum_gap_ns = max(self.maximum_gap_ns or 0, gap_ns)
        else:
            self.first_monotonic_ns = observed_ns
        self.last_monotonic_ns = observed_ns
        self.count += 1


_monotonic = time.monotonic
_monotonic_ns = time.monotonic_ns
_PROC_STAT_MAX_BYTES = 4096
_MAX_REPORTED_PIDS = 32
_QUIESCENT_SCAN_COUNT = 2
_PR_SET_CHILD_SUBREAPER = 36
_PR_GET_CHILD_SUBREAPER = 37
CONTAINMENT_MODEL = "linux_subreaper_ppid_pidfd_v1"
SUPPORTED_CANCELLATION_SIGNALS = tuple(
    sorted((int(signal.SIGHUP), int(signal.SIGINT), int(signal.SIGTERM)))
)
SUPPORTED_CANCELLATION_SIGNAL_PRIORITY = (
    int(signal.SIGINT),
    int(signal.SIGTERM),
    int(signal.SIGHUP),
)
_SUPPORTED_CANCELLATION_SIGNAL_SET = frozenset(
    SUPPORTED_CANCELLATION_SIGNALS
)
_POPEN_TYPE = subprocess.Popen
_RUN_LOCK = threading.Lock()
_CONTAINMENT_POISONED_DETAIL: str | None = None
_CONTAINMENT_POISONED_PRIOR_SUBREAPER: bool | None = None
_LIBC = ctypes.CDLL(None, use_errno=True)


def _error(code: ProcessErrorCode, operation: str, detail: str, path: Path | None = None) -> ProcessError:
    return ProcessError(code, operation, detail, path)


def _signal_mask_numbers(mask: Sequence[int]) -> tuple[int, ...]:
    if any(isinstance(item, bool) or not isinstance(item, int) for item in mask):
        raise ProcessError(
            ProcessErrorCode.SIGNAL_MASK_FAILED,
            "signal-mask-observe",
            "signal numbers must be exact integers",
        )
    numbers = tuple(sorted({int(item) for item in mask}))
    if any(item <= 0 for item in numbers):
        raise ProcessError(
            ProcessErrorCode.SIGNAL_MASK_FAILED,
            "signal-mask-observe",
            "kernel returned a nonpositive signal number",
        )
    return numbers


def _query_signal_mask(operation: str) -> tuple[int, ...]:
    try:
        observed = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    except (OSError, ValueError) as error:
        raise ProcessError(
            ProcessErrorCode.SIGNAL_MASK_FAILED,
            operation,
            str(error),
        ) from error
    return _signal_mask_numbers(tuple(observed))


def _observe_supported_signal_handlers(
    operation: str,
    error_code: ProcessErrorCode,
) -> tuple[tuple[int, Callable[[int, object | None], object]], ...]:
    observed: list[
        tuple[int, Callable[[int, object | None], object]]
    ] = []
    for signal_number in SUPPORTED_CANCELLATION_SIGNALS:
        try:
            handler = signal.getsignal(signal_number)
        except (OSError, ValueError) as error:
            raise ProcessError(error_code, operation, str(error)) from error
        if not callable(handler):
            if handler is signal.SIG_DFL:
                disposition = "SIG_DFL"
            elif handler is signal.SIG_IGN:
                disposition = "SIG_IGN"
            else:
                disposition = type(handler).__name__
            raise ProcessError(
                error_code,
                operation,
                f"{_signal_name_number(signal_number)} has non-callable "
                f"disposition {disposition}",
            )
        observed.append((signal_number, handler))
    return tuple(observed)


def _verify_supported_signal_handlers(
    contract: _SignalContract,
    operation: str,
) -> None:
    observed = _observe_supported_signal_handlers(
        operation,
        ProcessErrorCode.SIGNAL_DISPOSITION_DRIFT,
    )
    if len(observed) != len(contract.supported_signal_handlers):
        raise ProcessError(
            ProcessErrorCode.SIGNAL_DISPOSITION_DRIFT,
            operation,
            "captured supported-signal handler set is incomplete",
        )
    for (expected_number, expected_handler), (
        observed_number,
        observed_handler,
    ) in zip(contract.supported_signal_handlers, observed, strict=True):
        if (
            observed_number != expected_number
            or observed_handler is not expected_handler
        ):
            raise ProcessError(
                ProcessErrorCode.SIGNAL_DISPOSITION_DRIFT,
                operation,
                f"{_signal_name_number(observed_number)} handler identity "
                "changed after precondition capture",
            )


def _captured_supported_signal_handler(
    contract: _SignalContract,
    signal_number: int,
    operation: str,
) -> Callable[[int, object | None], object]:
    expected_numbers = SUPPORTED_CANCELLATION_SIGNALS
    captured_numbers = tuple(
        item for item, _ in contract.supported_signal_handlers
    )
    if captured_numbers != expected_numbers:
        raise ProcessError(
            ProcessErrorCode.SIGNAL_DISPOSITION_DRIFT,
            operation,
            "captured supported-signal handler set is incomplete or unordered",
        )
    for captured_number, handler in contract.supported_signal_handlers:
        if captured_number == signal_number:
            return handler
    raise ProcessError(
        ProcessErrorCode.SIGNAL_DISPOSITION_DRIFT,
        operation,
        f"no captured handler for {_signal_name_number(signal_number)}",
    )


def _preflight_signal_contract() -> _SignalContract:
    caller_mask = _query_signal_mask("signal-mask-precondition")
    incompatible = tuple(
        item for item in caller_mask
        if item in _SUPPORTED_CANCELLATION_SIGNAL_SET
    )
    if incompatible:
        names = ",".join(signal.Signals(item).name for item in incompatible)
        raise ProcessError(
            ProcessErrorCode.SIGNAL_MASK_PRECONDITION,
            "signal-mask-precondition",
            f"supported cancellation signals are already blocked: {names}",
        )
    handlers = _observe_supported_signal_handlers(
        "signal-disposition-precondition",
        ProcessErrorCode.SIGNAL_DISPOSITION_PRECONDITION,
    )
    return _SignalContract(
        caller_mask_numbers=caller_mask,
        supported_signal_handlers=handlers,
    )


def _block_supported_signals(
    contract: _SignalContract,
    operation: str,
) -> None:
    try:
        previous = signal.pthread_sigmask(
            signal.SIG_BLOCK,
            _SUPPORTED_CANCELLATION_SIGNAL_SET,
        )
    except (OSError, ValueError) as error:
        raise ProcessError(
            ProcessErrorCode.SIGNAL_MASK_FAILED,
            operation,
            str(error),
        ) from error
    contract.blocked = True
    previous_numbers = _signal_mask_numbers(tuple(previous))
    if previous_numbers != contract.caller_mask_numbers:
        raise ProcessError(
            ProcessErrorCode.SIGNAL_MASK_FAILED,
            operation,
            "caller signal mask changed after precondition capture",
        )


def _pending_supported_signals(operation: str) -> tuple[int, ...]:
    try:
        pending = signal.sigpending()
    except (OSError, ValueError) as error:
        raise ProcessError(
            ProcessErrorCode.SIGNAL_MASK_FAILED,
            operation,
            str(error),
        ) from error
    return tuple(
        item for item in _signal_mask_numbers(tuple(pending))
        if item in _SUPPORTED_CANCELLATION_SIGNAL_SET
    )


def _signal_name_number(signal_number: int) -> str:
    return f"{signal.Signals(signal_number).name}({signal_number})"


def _format_signal_drain_event(event: _SignalDrainEvent) -> str:
    priority = ">".join(
        _signal_name_number(item)
        for item in SUPPORTED_CANCELLATION_SIGNAL_PRIORITY
    )
    observed = ",".join(
        _signal_name_number(item) for item in event.observed
    )
    selected = (
        _signal_name_number(event.selected)
        if event.selected is not None
        else "existing-primary"
    )
    coalesced = ",".join(
        _signal_name_number(item) for item in event.coalesced
    )
    return (
        f"supported-signal-drain boundary={event.boundary} "
        f"priority={priority} observed={observed} selected={selected} "
        f"coalesced={coalesced}"
    )


def _attach_signal_drain_notes(
    contract: _SignalContract,
    error: BaseException,
) -> None:
    while contract.noted_drain_event_count < len(contract.drain_events):
        event = contract.drain_events[contract.noted_drain_event_count]
        error.add_note(_format_signal_drain_event(event))
        contract.noted_drain_event_count += 1


def _consume_pending_supported_signal(
    signal_number: int,
    operation: str,
) -> None:
    try:
        consumed = signal.sigwait({signal_number})
    except (OSError, ValueError) as error:
        raise ProcessError(
            ProcessErrorCode.SIGNAL_MASK_FAILED,
            operation,
            str(error),
        ) from error
    if (
        isinstance(consumed, bool)
        or not isinstance(consumed, int)
        or int(consumed) != signal_number
    ):
        raise ProcessError(
            ProcessErrorCode.SIGNAL_MASK_FAILED,
            operation,
            "sigwait returned a different or non-integer signal identity",
        )


def _restore_caller_signal_mask(
    contract: _SignalContract,
    operation: str,
    *,
    primary_exists: bool,
) -> tuple[int, ...]:
    pending = _pending_supported_signals(operation)
    contract.deferred_supported_signal_numbers.update(pending)
    selected: int | None = None
    if pending and not primary_exists:
        selected = next(
            item
            for item in SUPPORTED_CANCELLATION_SIGNAL_PRIORITY
            if item in pending
        )
    coalesced = tuple(item for item in pending if item != selected)
    for signal_number in pending:
        _consume_pending_supported_signal(signal_number, operation)
    if pending:
        contract.drain_events.append(
            _SignalDrainEvent(
                boundary=operation,
                observed=pending,
                selected=selected,
                coalesced=coalesced,
            )
        )
    contract.blocked = False
    try:
        signal.pthread_sigmask(
            signal.SIG_SETMASK,
            set(contract.caller_mask_numbers),
        )
    except (OSError, ValueError) as error:
        contract.blocked = True
        raise ProcessError(
            ProcessErrorCode.SIGNAL_MASK_FAILED,
            operation,
            str(error),
        ) from error
    observed = _query_signal_mask(operation)
    if observed != contract.caller_mask_numbers:
        raise ProcessError(
            ProcessErrorCode.SIGNAL_MASK_FAILED,
            operation,
            "caller signal mask was not restored exactly",
        )
    if selected is not None:
        handler = _captured_supported_signal_handler(
            contract,
            selected,
            operation,
        )
        try:
            handler(selected, None)
        except BaseException as error:
            _attach_signal_drain_notes(contract, error)
            raise
        error = ProcessError(
            ProcessErrorCode.CANCELLATION_HANDLER_RETURNED,
            operation,
            "selected supported-cancellation handler returned without raising",
        )
        _attach_signal_drain_notes(contract, error)
        raise error
    return pending


def _child_restore_signal_contract(caller_mask_numbers: tuple[int, ...]) -> None:
    """Reset supported dispositions, then restore the parent's caller mask."""

    for signal_number in SUPPORTED_CANCELLATION_SIGNALS:
        signal.signal(signal_number, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_SETMASK, set(caller_mask_numbers))


def _spawn_prepublished(
    holder: _PopenHolder,
    *args,
    **kwargs,
) -> subprocess.Popen[bytes]:
    """Publish a Popen object before its constructor can create a child."""

    factory = subprocess.Popen
    if isinstance(factory, type) and issubclass(factory, _POPEN_TYPE):
        candidate = factory.__new__(factory)
        holder.process = candidate
        factory.__init__(candidate, *args, **kwargs)
    else:
        candidate = factory(*args, **kwargs)
        holder.process = candidate
    return candidate


def _recover_published_process(
    holder: _PopenHolder,
) -> subprocess.Popen[bytes] | None:
    candidate = holder.process
    if candidate is None:
        return None
    pid = getattr(candidate, "pid", None)
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or getattr(candidate, "_child_created", False) is not True
    ):
        return None
    return candidate


def _validate_absolute_path(path: Path, label: str) -> Path:
    if not isinstance(path, Path):
        raise _error(
            ProcessErrorCode.INVALID_ARGUMENT,
            "validate",
            f"{label} must be a pathlib.Path",
        )
    text = str(path)
    if (
        not text
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in text)
        or text.startswith("//")
        or not path.is_absolute()
        or os.path.normpath(text) != text
    ):
        raise _error(
            ProcessErrorCode.PATH_INVALID,
            "validate",
            f"{label} must be a normalized absolute path",
            path,
        )
    return path


def _validate_specs(
    command: CommandSpec,
    limits: CommandLimits,
    safety: SafetySpec,
) -> None:
    if (
        not isinstance(command, CommandSpec)
        or not isinstance(limits, CommandLimits)
        or not isinstance(safety, SafetySpec)
    ):
        raise _error(
            ProcessErrorCode.INVALID_ARGUMENT,
            "validate",
            "unexpected command, limits, or safety type",
        )
    if not command.argv or any(
        not isinstance(item, str) or not item or "\x00" in item
        for item in command.argv
    ):
        raise _error(
            ProcessErrorCode.INVALID_ARGUMENT,
            "validate",
            "argv must contain nonempty NUL-free strings",
        )
    _validate_absolute_path(command.cwd, "cwd")
    _validate_absolute_path(command.stdout_path, "stdout_path")
    _validate_absolute_path(command.stderr_path, "stderr_path")
    if command.stdout_path == command.stderr_path:
        raise _error(ProcessErrorCode.INVALID_ARGUMENT, "validate", "stdout and stderr paths must differ")
    if not isinstance(command.environment, Mapping):
        raise _error(ProcessErrorCode.INVALID_ARGUMENT, "validate", "environment must be a mapping")
    for name, value in command.environment.items():
        if (
            not isinstance(name, str)
            or not name
            or "=" in name
            or "\x00" in name
            or not isinstance(value, str)
            or "\x00" in value
        ):
            raise _error(
                ProcessErrorCode.INVALID_ARGUMENT,
                "validate",
                "environment contains an invalid name or value",
            )
    numeric_nonnegative = (
        ("stdout_bytes", limits.stdout_bytes),
        ("stderr_bytes", limits.stderr_bytes),
        (
            "max_observed_live_descendants",
            limits.max_observed_live_descendants,
        ),
        ("max_descendant_identities", limits.max_descendant_identities),
        ("minimum_free_bytes", safety.minimum_free_bytes),
    )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for _, value in numeric_nonnegative
    ):
        raise _error(
            ProcessErrorCode.INVALID_ARGUMENT,
            "validate",
            "byte/count limits must be nonnegative integers",
        )
    numeric_positive = (
        ("max_proc_entries", limits.max_proc_entries),
        ("read_chunk_bytes", limits.read_chunk_bytes),
        ("max_samples", safety.max_samples),
        ("sensor_max_bytes", safety.sensor_max_bytes),
    )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for _, value in numeric_positive
    ):
        raise _error(
            ProcessErrorCode.INVALID_ARGUMENT,
            "validate",
            "entry/chunk/sample limits must be positive integers",
        )
    durations = (
        ("wall_seconds", limits.wall_seconds),
        ("cleanup_seconds", limits.cleanup_seconds),
        ("poll_interval_seconds", safety.poll_interval_seconds),
    )
    for label, value in durations:
        try:
            valid = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and value > 0
                and math.isfinite(value)
            )
        except OverflowError:
            valid = False
        if not valid:
            raise _error(
                ProcessErrorCode.INVALID_ARGUMENT,
                "validate",
                f"{label} must be positive and finite",
            )
    _validate_absolute_path(safety.storage_path, "storage_path")
    if not safety.sensors:
        raise _error(
            ProcessErrorCode.INVALID_ARGUMENT,
            "validate",
            "at least one thermal sensor is required",
        )
    for sensor in safety.sensors:
        if not isinstance(sensor, ThermalSensor):
            raise _error(ProcessErrorCode.INVALID_ARGUMENT, "validate", "invalid thermal sensor")
        _validate_absolute_path(sensor.path, "thermal sensor")
        if (
            not isinstance(sensor.ceiling_millicelsius, int)
            or isinstance(sensor.ceiling_millicelsius, bool)
            or sensor.ceiling_millicelsius <= 0
            or not isinstance(sensor.minimum_plausible_millicelsius, int)
            or isinstance(sensor.minimum_plausible_millicelsius, bool)
            or sensor.minimum_plausible_millicelsius < 0
            or sensor.minimum_plausible_millicelsius >= sensor.ceiling_millicelsius
        ):
            raise _error(
                ProcessErrorCode.INVALID_ARGUMENT,
                "validate",
                "thermal bounds must be exact integers satisfying 0 <= minimum < ceiling",
            )


def _open_regular_bounded(path: Path, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ProcessError(ProcessErrorCode.SAFETY_READ_FAILED, "read-safety", str(error), path) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ProcessError(
                ProcessErrorCode.SAFETY_READ_FAILED,
                "read-safety",
                "sensor is not a single-linked regular file",
                path,
            )
        data = bytearray()
        while len(data) <= max_bytes:
            chunk = os.read(descriptor, min(4096, max_bytes + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(descriptor)
        if len(data) > max_bytes:
            raise ProcessError(
                ProcessErrorCode.SAFETY_READ_FAILED,
                "read-safety",
                "sensor exceeds byte limit",
                path,
            )
        if (
            (
                before.st_dev,
                before.st_ino,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
        ):
            raise ProcessError(
                ProcessErrorCode.SAFETY_READ_FAILED,
                "read-safety",
                "sensor changed during read",
                path,
            )
        pathname_after = os.lstat(path)
        if (
            stat.S_ISLNK(pathname_after.st_mode)
            or (pathname_after.st_dev, pathname_after.st_ino)
            != (after.st_dev, after.st_ino)
        ):
            raise ProcessError(
                ProcessErrorCode.SAFETY_READ_FAILED,
                "read-safety",
                "sensor pathname changed during read",
                path,
            )
        return bytes(data)
    except ProcessError:
        raise
    except OSError as error:
        raise ProcessError(ProcessErrorCode.SAFETY_READ_FAILED, "read-safety", str(error), path) from error
    finally:
        os.close(descriptor)


def _sample_safety(policy: SafetySpec, monotonic_ns: int | None = None) -> SafetySample:
    observations: list[tuple[str, int]] = []
    for sensor in policy.sensors:
        raw = _open_regular_bounded(sensor.path, policy.sensor_max_bytes)
        try:
            text = raw.decode("ascii")
            digits = text[:-1] if text.endswith("\n") else text
            if not digits or not digits.isdigit() or "\n" in digits:
                raise ValueError("not one unsigned decimal integer")
            value = int(digits, 10)
        except (UnicodeDecodeError, ValueError) as error:
            raise ProcessError(
                ProcessErrorCode.SAFETY_READ_FAILED,
                "read-safety",
                f"invalid thermal value: {error}",
                sensor.path,
            ) from error
        observations.append((str(sensor.path), value))
    try:
        filesystem = os.statvfs(policy.storage_path)
        free_bytes = filesystem.f_bavail * filesystem.f_frsize
    except OSError as error:
        raise ProcessError(
            ProcessErrorCode.SAFETY_READ_FAILED,
            "statvfs",
            str(error),
            policy.storage_path,
        ) from error
    sample = SafetySample(
        monotonic_ns=_monotonic_ns() if monotonic_ns is None else monotonic_ns,
        thermal_millicelsius=tuple(observations),
        free_bytes=free_bytes,
    )
    for sensor, (_, value) in zip(policy.sensors, observations, strict=True):
        if value < sensor.minimum_plausible_millicelsius:
            raise ProcessError(
                ProcessErrorCode.SAFETY_READ_FAILED,
                "safety",
                f"{sensor.path} observed implausible {value} mC below "
                f"{sensor.minimum_plausible_millicelsius} mC",
                sensor.path,
                sample=sample,
            )
        if value >= sensor.ceiling_millicelsius:
            raise ProcessError(
                ProcessErrorCode.THERMAL_LIMIT,
                "safety",
                f"{sensor.path} observed {value} mC at/above {sensor.ceiling_millicelsius} mC",
                sensor.path,
                sample=sample,
            )
    if free_bytes < policy.minimum_free_bytes:
        raise ProcessError(
            ProcessErrorCode.FREE_SPACE_LIMIT,
            "safety",
            f"{policy.storage_path} has {free_bytes} bytes below {policy.minimum_free_bytes}",
            policy.storage_path,
            sample=sample,
        )
    return sample


def sample_safety(policy: SafetySpec, *, monotonic_ns: int | None = None) -> SafetySample:
    if not isinstance(policy, SafetySpec):
        raise _error(ProcessErrorCode.INVALID_ARGUMENT, "validate", "safety must be a SafetySpec")
    return _sample_safety(policy, monotonic_ns)


def _validate_existing_directory(path: Path, operation: str) -> None:
    try:
        information = os.lstat(path)
    except OSError as error:
        raise ProcessError(ProcessErrorCode.PATH_INVALID, operation, str(error), path) from error
    if stat.S_ISLNK(information.st_mode) or not stat.S_ISDIR(information.st_mode):
        raise ProcessError(ProcessErrorCode.PATH_INVALID, operation, "path is not a non-symlink directory", path)


def _open_output_parent(path: Path) -> int:
    _validate_existing_directory(path.parent, "validate-output-parent")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open(path.parent, flags)
    except OSError as error:
        raise ProcessError(
            ProcessErrorCode.OUTPUT_CREATE_FAILED,
            "open-output-parent",
            str(error),
            path.parent,
        ) from error
    try:
        opened = os.fstat(descriptor)
        pathname = os.lstat(path.parent)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or stat.S_ISLNK(pathname.st_mode)
            or (opened.st_dev, opened.st_ino) != (pathname.st_dev, pathname.st_ino)
        ):
            raise ProcessError(
                ProcessErrorCode.OUTPUT_CREATE_FAILED,
                "open-output-parent",
                "output parent pathname changed during open",
                path.parent,
            )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_output_exclusive(path: Path, parent_descriptor: int) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path.name, flags, 0o600, dir_fd=parent_descriptor)
    except OSError as error:
        raise ProcessError(ProcessErrorCode.OUTPUT_CREATE_FAILED, "create-output", str(error), path) from error
    try:
        information = os.fstat(descriptor)
        if not stat.S_ISREG(information.st_mode) or information.st_nlink != 1:
            raise ProcessError(
                ProcessErrorCode.OUTPUT_CREATE_FAILED,
                "create-output",
                "output is not a single-linked regular file",
                path,
            )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _fsync_open_output(descriptor: int, path: Path, operation: str) -> None:
    """Durably sync an open output object without transferring ownership."""

    try:
        os.fsync(descriptor)
    except OSError as error:
        raise ProcessError(
            ProcessErrorCode.OUTPUT_WRITE_FAILED,
            operation,
            str(error),
            path,
        ) from error


def _close_output_file(descriptor: int, path: Path) -> None:
    primary: BaseException | None = None
    try:
        os.fsync(descriptor)
    except BaseException as error:
        primary = error
    try:
        os.close(descriptor)
    except BaseException as close_error:
        if primary is None:
            primary = close_error
        else:
            primary.add_note(
                f"descriptor close also raised {type(close_error).__name__}: {close_error}"
            )
    if primary is not None:
        if isinstance(primary, ProcessError):
            raise primary
        raise ProcessError(
            ProcessErrorCode.OUTPUT_WRITE_FAILED,
            "finalize-output",
            str(primary),
            path,
        ) from primary


def _fsync_output_parent(descriptor: int, path: Path) -> None:
    primary: BaseException | None = None
    try:
        os.fsync(descriptor)
    except BaseException as error:
        primary = error
    try:
        os.close(descriptor)
    except BaseException as close_error:
        if primary is None:
            primary = close_error
        else:
            primary.add_note(
                f"directory descriptor close also raised {type(close_error).__name__}: {close_error}"
            )
    if primary is not None:
        if isinstance(primary, ProcessError):
            raise primary
        raise ProcessError(
            ProcessErrorCode.OUTPUT_WRITE_FAILED,
            "fsync-output-parent",
            str(primary),
            path,
        ) from primary


def _write_all(descriptor: int, data: bytes, path: Path) -> None:
    offset = 0
    try:
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise OSError("write made no progress")
            offset += written
    except OSError as error:
        raise ProcessError(ProcessErrorCode.OUTPUT_WRITE_FAILED, "write-output", str(error), path) from error


def _read_proc_identity(pid: int) -> _ProcIdentity | None:
    path = Path("/proc") / str(pid) / "stat"
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOFOLLOW)
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as error:
        raise ProcessError(ProcessErrorCode.CONTAINMENT_FAILED, "procfs-open", str(error), path) from error
    try:
        data = bytearray()
        while len(data) <= _PROC_STAT_MAX_BYTES:
            chunk = os.read(descriptor, _PROC_STAT_MAX_BYTES + 1 - len(data))
            if not chunk:
                break
            data.extend(chunk)
        if len(data) > _PROC_STAT_MAX_BYTES:
            raise ProcessError(ProcessErrorCode.CONTAINMENT_FAILED, "procfs-read", "stat record exceeds bound", path)
    except ProcessError:
        raise
    except OSError as error:
        raise ProcessError(ProcessErrorCode.CONTAINMENT_FAILED, "procfs-read", str(error), path) from error
    finally:
        os.close(descriptor)
    opening = data.find(b"(")
    closing = data.rfind(b") ")
    try:
        fields = data[closing + 2 :].split()
        identity = _ProcIdentity(
            pid=int(data[:opening].strip(), 10),
            parent_pid=int(fields[1], 10),
            state=fields[0].decode("ascii"),
            process_group=int(fields[2], 10),
            session=int(fields[3], 10),
            start_time_ticks=int(fields[19], 10),
        )
    except (IndexError, UnicodeDecodeError, ValueError) as error:
        raise ProcessError(
            ProcessErrorCode.CONTAINMENT_FAILED,
            "procfs-parse",
            f"malformed stat record: {error}",
            path,
        ) from error
    if (
        opening <= 0
        or closing <= opening
        or identity.pid != pid
        or len(identity.state) != 1
        or identity.start_time_ticks <= 0
    ):
        raise ProcessError(ProcessErrorCode.CONTAINMENT_FAILED, "procfs-parse", "invalid stat identity", path)
    return identity


def _same_process(left: _ProcIdentity, right: _ProcIdentity) -> bool:
    """Compare only kernel-stable identity fields.

    PPID, state, process group, and session are deliberately allowed to change
    between observation and signalling.  In particular, reparenting and
    ``setsid(2)`` are containment events, not identity changes.
    """

    return (
        left.pid == right.pid
        and left.start_time_ticks == right.start_time_ticks
    )


def _scan_process_table(
    max_entries: int,
    deadline: float,
    *,
    deadline_code: ProcessErrorCode,
) -> dict[int, _ProcIdentity]:
    table: dict[int, _ProcIdentity] = {}
    seen = 0
    try:
        directory = os.scandir("/proc")
    except OSError as error:
        raise ProcessError(
            ProcessErrorCode.CONTAINMENT_FAILED,
            "procfs-scan",
            str(error),
            Path("/proc"),
        ) from error
    with directory:
        for entry in directory:
            if _monotonic() >= deadline:
                raise ProcessError(
                    deadline_code,
                    "procfs-scan",
                    "bounded process-table scan deadline expired",
                )
            if not entry.name.isascii() or not entry.name.isdigit():
                continue
            seen += 1
            if seen > max_entries:
                raise ProcessError(
                    ProcessErrorCode.PROCFS_LIMIT,
                    "procfs-scan",
                    f"more than {max_entries} process entries",
                )
            identity = _read_proc_identity(int(entry.name, 10))
            if identity is not None:
                table[identity.pid] = identity
    return table


def _kernel_thread_count(max_entries: int) -> int:
    count = 0
    path = Path("/proc/self/task")
    try:
        directory = os.scandir(path)
    except OSError as error:
        raise ProcessError(
            ProcessErrorCode.CALLER_MULTITHREADED,
            "containment-precondition",
            f"cannot enumerate controller threads: {error}",
            path,
        ) from error
    with directory:
        for entry in directory:
            if not entry.name.isascii() or not entry.name.isdigit():
                continue
            count += 1
            if count > max_entries:
                raise ProcessError(
                    ProcessErrorCode.PROCFS_LIMIT,
                    "containment-precondition",
                    f"more than {max_entries} controller threads",
                    path,
                )
    return count


def _get_child_subreaper() -> bool:
    if not sys.platform.startswith("linux"):
        raise ProcessError(
            ProcessErrorCode.SUBREAPER_FAILED,
            "prctl-get-child-subreaper",
            "Linux PR_SET_CHILD_SUBREAPER is required",
        )
    value = ctypes.c_int()
    ctypes.set_errno(0)
    result = _LIBC.prctl(
        _PR_GET_CHILD_SUBREAPER,
        ctypes.byref(value),
        0,
        0,
        0,
    )
    if result != 0 or value.value not in (0, 1):
        error_number = ctypes.get_errno()
        detail = os.strerror(error_number) if error_number else "invalid prctl result"
        raise ProcessError(
            ProcessErrorCode.SUBREAPER_FAILED,
            "prctl-get-child-subreaper",
            detail,
        )
    return bool(value.value)


def _set_child_subreaper(enabled: bool) -> None:
    ctypes.set_errno(0)
    result = _LIBC.prctl(
        _PR_SET_CHILD_SUBREAPER,
        int(enabled),
        0,
        0,
        0,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        detail = os.strerror(error_number) if error_number else "prctl failed"
        raise ProcessError(
            ProcessErrorCode.SUBREAPER_FAILED,
            "prctl-set-child-subreaper",
            detail,
        )


def _mark_containment_poisoned(detail: str, previous_subreaper: bool) -> None:
    global _CONTAINMENT_POISONED_DETAIL
    global _CONTAINMENT_POISONED_PRIOR_SUBREAPER
    if _CONTAINMENT_POISONED_DETAIL is None:
        _CONTAINMENT_POISONED_DETAIL = detail
        _CONTAINMENT_POISONED_PRIOR_SUBREAPER = previous_subreaper


def _prove_ambiguous_setup_child_free(
    owner_pid: int,
    limits: CommandLimits,
    deadline: float,
) -> bool:
    """Observe quiescence without signalling or reaping ambiguous children."""

    empty_scans = 0
    while _monotonic() < deadline:
        descendants = _scan_descendants(
            owner_pid,
            None,
            limits.max_proc_entries,
            deadline,
            deadline_code=ProcessErrorCode.CLEANUP_TIMEOUT,
        )
        if descendants:
            empty_scans = 0
        else:
            empty_scans += 1
            if empty_scans >= _QUIESCENT_SCAN_COUNT:
                return True
        time.sleep(min(0.01, max(0.0, deadline - _monotonic())))
    return False


def _enter_containment(limits: CommandLimits) -> _ContainmentLease:
    if not _RUN_LOCK.acquire(blocking=False):
        raise ProcessError(
            ProcessErrorCode.RUNNER_BUSY,
            "containment-precondition",
            "another bounded command owns the process-wide subreaper",
        )
    previous_subreaper: bool | None = None
    post_enable_state = False
    post_enable_failure_handled = False
    try:
        if _CONTAINMENT_POISONED_DETAIL is not None:
            raise ProcessError(
                ProcessErrorCode.CONTAINMENT_POISONED,
                "containment-precondition",
                _CONTAINMENT_POISONED_DETAIL,
            )
        thread_count = _kernel_thread_count(limits.max_proc_entries)
        if thread_count != 1:
            raise ProcessError(
                ProcessErrorCode.CALLER_MULTITHREADED,
                "containment-precondition",
                f"controller has {thread_count} kernel threads; exactly one is required",
            )
        owner_pid = os.getpid()
        deadline = _monotonic() + limits.cleanup_seconds
        table = _scan_process_table(
            limits.max_proc_entries,
            deadline,
            deadline_code=ProcessErrorCode.CLEANUP_TIMEOUT,
        )
        children = tuple(
            identity for identity in table.values()
            if identity.parent_pid == owner_pid
        )
        if children:
            raise ProcessError(
                ProcessErrorCode.PREEXISTING_CHILD,
                "containment-precondition",
                "controller already owns direct child PIDs "
                + ",".join(str(item.pid) for item in sorted(children, key=lambda item: item.pid)[:_MAX_REPORTED_PIDS]),
            )
        previous_subreaper = _get_child_subreaper()
        if not previous_subreaper:
            _set_child_subreaper(True)
        post_enable_state = True
        if not _get_child_subreaper():
            raise ProcessError(
                ProcessErrorCode.SUBREAPER_FAILED,
                "containment-precondition",
                "kernel did not retain enabled child-subreaper state",
            )
        try:
            table = _scan_process_table(
                limits.max_proc_entries,
                deadline,
                deadline_code=ProcessErrorCode.CLEANUP_TIMEOUT,
            )
        except BaseException as error:
            post_enable_failure_handled = True
            _mark_containment_poisoned(
                "post-enable containment scan failed before command ownership "
                f"was established: {type(error).__name__}: {error}",
                previous_subreaper,
            )
            raise
        children = tuple(
            identity for identity in table.values()
            if identity.parent_pid == owner_pid
        )
        if children:
            primary = ProcessError(
                ProcessErrorCode.PREEXISTING_CHILD,
                "containment-precondition",
                "a direct child appeared while establishing exclusive containment",
            )
            post_enable_failure_handled = True
            proof_error: BaseException | None = None
            try:
                child_free = _prove_ambiguous_setup_child_free(
                    owner_pid,
                    limits,
                    deadline,
                )
            except BaseException as error:
                child_free = False
                proof_error = error
            if child_free:
                try:
                    if not previous_subreaper:
                        _set_child_subreaper(False)
                    if _get_child_subreaper() != previous_subreaper:
                        raise ProcessError(
                            ProcessErrorCode.SUBREAPER_FAILED,
                            "containment-precondition",
                            "failed to restore prior child-subreaper state",
                        )
                except BaseException as restore_error:
                    _mark_containment_poisoned(
                        "ambiguous setup became child-free but subreaper "
                        f"restoration failed: {restore_error}",
                        previous_subreaper,
                    )
                    primary.add_note(
                        "subreaper restoration raised "
                        f"{type(restore_error).__name__}: {restore_error}"
                    )
            else:
                detail = (
                    "post-enable direct child ownership remained ambiguous; "
                    "no signal or wait operation was attempted"
                )
                if proof_error is not None:
                    detail += (
                        "; child-free proof raised "
                        f"{type(proof_error).__name__}: {proof_error}"
                    )
                    primary.add_note(detail)
                _mark_containment_poisoned(detail, previous_subreaper)
            raise primary
        return _ContainmentLease(
            owner_pid=owner_pid,
            caller_thread_count=thread_count,
            preexisting_child_count=0,
            previous_subreaper=previous_subreaper,
        )
    except BaseException as error:
        if post_enable_state and not post_enable_failure_handled:
            _mark_containment_poisoned(
                "containment setup failed after enabling subreaper but before "
                f"child-free ownership was proven: {type(error).__name__}: {error}",
                bool(previous_subreaper),
            )
        _RUN_LOCK.release()
        raise


def _restore_containment(lease: _ContainmentLease) -> None:
    if not lease.previous_subreaper:
        _set_child_subreaper(False)
    if _get_child_subreaper() != lease.previous_subreaper:
        raise ProcessError(
            ProcessErrorCode.SUBREAPER_FAILED,
            "prctl-restore-child-subreaper",
            "child-subreaper state does not match its pre-command value",
        )


def _reset_containment_poison_for_tests() -> None:
    """Test-only recovery seam requiring an independent child-free proof."""

    global _CONTAINMENT_POISONED_DETAIL
    global _CONTAINMENT_POISONED_PRIOR_SUBREAPER
    if not _RUN_LOCK.acquire(blocking=False):
        raise RuntimeError("cannot reset containment poison while a run is active")
    try:
        if _CONTAINMENT_POISONED_DETAIL is None:
            return
        if _kernel_thread_count(65536) != 1:
            raise RuntimeError("cannot reset containment poison in a multithreaded process")
        deadline = _monotonic() + 1.0
        owner_pid = os.getpid()
        for _ in range(_QUIESCENT_SCAN_COUNT):
            table = _scan_process_table(
                65536,
                deadline,
                deadline_code=ProcessErrorCode.CLEANUP_TIMEOUT,
            )
            if any(identity.parent_pid == owner_pid for identity in table.values()):
                raise RuntimeError("cannot reset containment poison with a direct child")
        previous = _CONTAINMENT_POISONED_PRIOR_SUBREAPER
        if previous is None:
            raise RuntimeError("poisoned containment has no prior subreaper state")
        _set_child_subreaper(previous)
        if _get_child_subreaper() != previous:
            raise RuntimeError("cannot restore prior subreaper state")
        _CONTAINMENT_POISONED_DETAIL = None
        _CONTAINMENT_POISONED_PRIOR_SUBREAPER = None
    finally:
        _RUN_LOCK.release()


def _scan_descendants(
    owner_pid: int,
    leader: _ProcIdentity | None,
    max_entries: int,
    deadline: float,
    *,
    deadline_code: ProcessErrorCode,
) -> list[_ProcIdentity]:
    table = _scan_process_table(
        max_entries,
        deadline,
        deadline_code=deadline_code,
    )
    if leader is not None:
        current_leader = table.get(leader.pid)
        if current_leader is not None and not _same_process(current_leader, leader):
            raise ProcessError(
                ProcessErrorCode.CONTAINMENT_FAILED,
                "procfs-identity",
                "leader PID/start-time identity changed before it was reaped",
            )
    children_by_parent: dict[int, list[_ProcIdentity]] = {}
    for identity in table.values():
        children_by_parent.setdefault(identity.parent_pid, []).append(identity)

    descendants: dict[tuple[int, int], _ProcIdentity] = {}
    frontier: list[int] = []
    if leader is not None:
        frontier.append(leader.pid)
    for identity in children_by_parent.get(owner_pid, ()):
        if leader is not None and _same_process(identity, leader):
            continue
        key = (identity.pid, identity.start_time_ticks)
        descendants[key] = identity
        frontier.append(identity.pid)

    visited_parents: set[int] = set()
    while frontier:
        parent_pid = frontier.pop()
        if parent_pid in visited_parents:
            continue
        visited_parents.add(parent_pid)
        for identity in children_by_parent.get(parent_pid, ()):
            if identity.pid == owner_pid:
                raise ProcessError(
                    ProcessErrorCode.CONTAINMENT_FAILED,
                    "procfs-lineage",
                    "process-table lineage contains a cycle through the controller",
                )
            if leader is not None and _same_process(identity, leader):
                continue
            key = (identity.pid, identity.start_time_ticks)
            if key not in descendants:
                descendants[key] = identity
                frontier.append(identity.pid)
    return sorted(
        descendants.values(),
        key=lambda item: (item.pid, item.start_time_ticks),
    )


def _leader_exited(pidfd: int | None, leader_pid: int) -> bool:
    id_type = os.P_PIDFD if pidfd is not None else os.P_PID
    identifier = pidfd if pidfd is not None else leader_pid
    try:
        return os.waitid(
            id_type,
            identifier,
            os.WEXITED | os.WNOHANG | os.WNOWAIT,
        ) is not None
    except (ChildProcessError, OSError) as error:
        raise ProcessError(
            ProcessErrorCode.CONTAINMENT_FAILED,
            "pidfd-wait" if pidfd is not None else "leader-wait",
            str(error),
        ) from error


def _open_verified_pidfd(identity: _ProcIdentity) -> int | None:
    try:
        descriptor = os.pidfd_open(identity.pid, 0)
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as error:
        raise ProcessError(
            ProcessErrorCode.CONTAINMENT_FAILED,
            "pidfd-open-descendant",
            str(error),
        ) from error
    try:
        current = _read_proc_identity(identity.pid)
        if current is None:
            os.close(descriptor)
            return None
        if not _same_process(current, identity):
            raise ProcessError(
                ProcessErrorCode.CONTAINMENT_FAILED,
                "pidfd-verify-descendant",
                "PID/start-time identity changed while opening pidfd",
            )
        return descriptor
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _signal_identity(identity: _ProcIdentity) -> None:
    descriptor = _open_verified_pidfd(identity)
    if descriptor is None:
        return
    try:
        current = _read_proc_identity(identity.pid)
        if current is None or not _same_process(current, identity) or current.state == "Z":
            return
        try:
            signal.pidfd_send_signal(descriptor, signal.SIGKILL, None, 0)
        except ProcessLookupError:
            return
        except OSError as error:
            raise ProcessError(
                ProcessErrorCode.CONTAINMENT_FAILED,
                "pidfd-signal-descendant",
                str(error),
            ) from error
    finally:
        os.close(descriptor)


def _kill_group(process_group: int) -> None:
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError as error:
        raise ProcessError(
            ProcessErrorCode.CONTAINMENT_FAILED,
            "kill-process-group",
            str(error),
        ) from error


def _reap_direct_zombies(
    owner_pid: int,
    unreaped_leader: _ProcIdentity | None,
    descendants: Sequence[_ProcIdentity],
) -> None:
    for identity in descendants:
        if (
            identity.parent_pid != owner_pid
            or (
                unreaped_leader is not None
                and _same_process(identity, unreaped_leader)
            )
            or identity.state != "Z"
        ):
            continue
        current = _read_proc_identity(identity.pid)
        if current is None:
            continue
        if (
            not _same_process(current, identity)
            or current.parent_pid != owner_pid
            or current.state != "Z"
        ):
            continue
        try:
            waited = os.waitid(os.P_PID, identity.pid, os.WEXITED | os.WNOHANG)
        except ChildProcessError:
            if _read_proc_identity(identity.pid) is None:
                continue
            raise ProcessError(
                ProcessErrorCode.CONTAINMENT_FAILED,
                "reap-direct-descendant",
                f"PID {identity.pid} was not waitable by its recorded parent",
            )
        except OSError as error:
            raise ProcessError(
                ProcessErrorCode.CONTAINMENT_FAILED,
                "reap-direct-descendant",
                str(error),
            ) from error
        if waited is not None and waited.si_pid != identity.pid:
            raise ProcessError(
                ProcessErrorCode.CONTAINMENT_FAILED,
                "reap-direct-descendant",
                f"waitid returned PID {waited.si_pid} for requested PID {identity.pid}",
            )


def _cleanup(
    process: subprocess.Popen[bytes],
    pidfd: int | None,
    leader: _ProcIdentity,
    lease: _ContainmentLease,
    *,
    limits: CommandLimits,
    force_kill: bool,
    previously_observed: Mapping[tuple[int, int], _ProcIdentity],
    previously_observed_lower_bound: int,
    previously_observed_complete: bool,
    scan_cadence: _ScanCadence,
) -> _CleanupResult:
    deadline = _monotonic() + limits.cleanup_seconds
    kill_attempted = force_kill
    observed: dict[tuple[int, int], _ProcIdentity] = dict(previously_observed)
    residual: dict[tuple[int, int], _ProcIdentity] = {}
    observed_identity_lower_bound = max(
        previously_observed_lower_bound,
        len(observed),
    )
    observed_identity_complete = previously_observed_complete
    residual_identity_lower_bound = 0
    residual_identity_complete = True
    maximum_observed_live_descendant_count = 0
    problem_code: ProcessErrorCode | None = None
    problem_detail: str | None = None
    leader_reaped = False
    pre_reap_empty_scans = 0
    post_reap_empty_scans = 0
    quiescent = False

    def record_problem(error: ProcessError) -> None:
        nonlocal problem_code, problem_detail
        if problem_code is None:
            problem_code, problem_detail = error.code, error.detail

    def remember(descendants: Sequence[_ProcIdentity]) -> None:
        nonlocal observed_identity_complete
        nonlocal observed_identity_lower_bound
        nonlocal residual_identity_complete
        nonlocal residual_identity_lower_bound
        for identity in descendants:
            key = (identity.pid, identity.start_time_ticks)
            if key in residual:
                residual[key] = identity
            elif len(residual) < limits.max_descendant_identities:
                residual[key] = identity
                if residual_identity_complete:
                    residual_identity_lower_bound = len(residual)
            else:
                residual_identity_complete = False
                residual_identity_lower_bound = max(
                    residual_identity_lower_bound,
                    len(residual) + 1,
                )
            if key in observed:
                observed[key] = identity
                continue
            if len(observed) >= limits.max_descendant_identities:
                observed_identity_complete = False
                observed_identity_lower_bound = max(
                    observed_identity_lower_bound,
                    len(observed) + 1,
                )
                continue
            observed[key] = identity
            if observed_identity_complete:
                observed_identity_lower_bound = len(observed)

    def signal_leader_and_group() -> None:
        nonlocal problem_code, problem_detail
        try:
            if not _leader_exited(pidfd, process.pid):
                if pidfd is None:
                    current = _read_proc_identity(leader.pid)
                    if current is not None and _same_process(current, leader):
                        os.kill(leader.pid, signal.SIGKILL)
                else:
                    signal.pidfd_send_signal(pidfd, signal.SIGKILL, None, 0)
            _kill_group(process.pid)
        except (OSError, ProcessError) as error:
            if problem_code is None:
                problem_code = ProcessErrorCode.CONTAINMENT_FAILED
                problem_detail = str(error)

    if force_kill:
        signal_leader_and_group()
        for identity in observed.values():
            try:
                _signal_identity(identity)
            except ProcessError as error:
                record_problem(error)
    while _monotonic() < deadline:
        descendants: list[_ProcIdentity] | None
        try:
            descendants = _scan_descendants(
                lease.owner_pid,
                None if leader_reaped else leader,
                limits.max_proc_entries,
                deadline,
                deadline_code=ProcessErrorCode.CLEANUP_TIMEOUT,
            )
            scan_cadence.record()
        except ProcessError as error:
            descendants = None
            record_problem(error)
        if descendants is None:
            pre_reap_empty_scans = 0
            post_reap_empty_scans = 0
            known_now: list[_ProcIdentity] = []
            for identity in observed.values():
                try:
                    current = _read_proc_identity(identity.pid)
                except ProcessError as error:
                    record_problem(error)
                    continue
                if current is None or not _same_process(current, identity):
                    continue
                known_now.append(current)
                if current.state != "Z":
                    try:
                        _signal_identity(current)
                    except ProcessError as error:
                        record_problem(error)
            try:
                _reap_direct_zombies(
                    lease.owner_pid,
                    None if leader_reaped else leader,
                    known_now,
                )
            except ProcessError as error:
                record_problem(error)
        else:
            maximum_observed_live_descendant_count = max(
                maximum_observed_live_descendant_count,
                len(descendants),
            )
            if descendants:
                remember(descendants)
                kill_attempted = True
                for identity in descendants:
                    if identity.state == "Z":
                        continue
                    try:
                        _signal_identity(identity)
                    except ProcessError as error:
                        record_problem(error)
                try:
                    _reap_direct_zombies(
                        lease.owner_pid,
                        None if leader_reaped else leader,
                        descendants,
                    )
                except ProcessError as error:
                    record_problem(error)
            if not leader_reaped:
                try:
                    leader_done = _leader_exited(pidfd, process.pid)
                except ProcessError as error:
                    record_problem(error)
                    leader_done = False
                if force_kill and not leader_done:
                    signal_leader_and_group()
                if leader_done and not descendants:
                    pre_reap_empty_scans += 1
                else:
                    pre_reap_empty_scans = 0
                if pre_reap_empty_scans >= _QUIESCENT_SCAN_COUNT:
                    try:
                        process.wait(timeout=max(0.0, deadline - _monotonic()))
                    except subprocess.TimeoutExpired:
                        if problem_code is None:
                            problem_code = ProcessErrorCode.CLEANUP_TIMEOUT
                            problem_detail = "leader could not be reaped before cleanup deadline"
                    else:
                        leader_reaped = True
                        post_reap_empty_scans = 0
            else:
                if descendants:
                    post_reap_empty_scans = 0
                else:
                    post_reap_empty_scans += 1
                    if post_reap_empty_scans >= _QUIESCENT_SCAN_COUNT:
                        quiescent = True
                        break
        time.sleep(min(0.01, max(0.0, deadline - _monotonic())))

    if not leader_reaped:
        signal_leader_and_group()
        try:
            process.wait(timeout=0)
        except subprocess.TimeoutExpired:
            if problem_code is None:
                problem_code = ProcessErrorCode.CLEANUP_TIMEOUT
                problem_detail = "leader could not be reaped before cleanup deadline"
        else:
            leader_reaped = True
    complete = (
        quiescent
        and leader_reaped
        and process.returncode is not None
        and problem_code is None
    )
    if not complete and problem_code is None:
        problem_code = ProcessErrorCode.CLEANUP_UNVERIFIED
        problem_detail = "all-descendant cleanup could not be verified"
    ordered_observed = tuple(
        sorted(observed.values(), key=lambda item: (item.pid, item.start_time_ticks))
    )
    ordered_residual = tuple(
        sorted(residual.values(), key=lambda item: (item.pid, item.start_time_ticks))
    )
    return _CleanupResult(
        complete=complete,
        kill_attempted=kill_attempted,
        observed_identities=ordered_observed,
        residual_identities=ordered_residual,
        observed_identity_lower_bound=observed_identity_lower_bound,
        observed_identity_complete=observed_identity_complete,
        residual_identity_lower_bound=residual_identity_lower_bound,
        residual_identity_complete=residual_identity_complete,
        maximum_observed_live_descendant_count=maximum_observed_live_descendant_count,
        code=problem_code,
        detail=problem_detail,
    )


def _cleanup_unidentified_leader(
    process: subprocess.Popen[bytes],
    lease: _ContainmentLease,
    *,
    limits: CommandLimits,
    scan_cadence: _ScanCadence,
) -> _CleanupResult:
    """Clean a command tree after Popen but before leader identity capture.

    The completed containment lease proves that every direct child appearing
    after ``Popen`` belongs to this one command.  The leader remains exactly
    waitable through its ``Popen`` object; after it is reaped, subreaper
    adoption exposes nested ``setsid`` descendants through owner-PPID closure.
    """

    deadline = _monotonic() + limits.cleanup_seconds
    observed: dict[tuple[int, int], _ProcIdentity] = {}
    observed_lower_bound = 0
    observed_complete = True
    maximum_observed_live_descendant_count = 0
    problem_code: ProcessErrorCode | None = None
    problem_detail: str | None = None
    leader_reaped = False
    unreaped_leader: _ProcIdentity | None = None
    empty_scans = 0
    quiescent = False

    def record_problem(error: BaseException) -> None:
        nonlocal problem_code, problem_detail
        if problem_code is None:
            if isinstance(error, ProcessError):
                problem_code, problem_detail = error.code, error.detail
            else:
                problem_code = ProcessErrorCode.CONTAINMENT_FAILED
                problem_detail = str(error)

    def remember(descendants: Sequence[_ProcIdentity]) -> None:
        nonlocal observed_complete, observed_lower_bound
        for identity in descendants:
            key = (identity.pid, identity.start_time_ticks)
            if key in observed:
                observed[key] = identity
            elif len(observed) < limits.max_descendant_identities:
                observed[key] = identity
                if observed_complete:
                    observed_lower_bound = len(observed)
            else:
                observed_complete = False
                observed_lower_bound = max(
                    observed_lower_bound,
                    len(observed) + 1,
                )

    try:
        unreaped_leader = _read_proc_identity(process.pid)
    except ProcessError as error:
        record_problem(error)
    try:
        process.kill()
    except ProcessLookupError:
        pass
    except OSError as error:
        record_problem(error)
    try:
        _kill_group(process.pid)
    except ProcessError as error:
        record_problem(error)
    try:
        process.wait(timeout=max(0.0, deadline - _monotonic()))
    except subprocess.TimeoutExpired as error:
        record_problem(
            ProcessError(
                ProcessErrorCode.CLEANUP_TIMEOUT,
                "cleanup-unidentified-leader",
                f"leader PID {process.pid} did not become waitable: {error}",
            )
        )
    else:
        leader_reaped = True
        unreaped_leader = None

    while _monotonic() < deadline:
        try:
            descendants = _scan_descendants(
                lease.owner_pid,
                unreaped_leader,
                limits.max_proc_entries,
                deadline,
                deadline_code=ProcessErrorCode.CLEANUP_TIMEOUT,
            )
            scan_cadence.record()
        except ProcessError as error:
            record_problem(error)
            empty_scans = 0
            descendants = None
        if descendants is not None:
            maximum_observed_live_descendant_count = max(
                maximum_observed_live_descendant_count,
                len(descendants),
            )
            if descendants:
                empty_scans = 0
                remember(descendants)
                for identity in descendants:
                    if (
                        identity.state == "Z"
                        or (
                            unreaped_leader is not None
                            and _same_process(identity, unreaped_leader)
                        )
                    ):
                        continue
                    try:
                        _signal_identity(identity)
                    except ProcessError as error:
                        record_problem(error)
                try:
                    _reap_direct_zombies(
                        lease.owner_pid,
                        unreaped_leader,
                        descendants,
                    )
                except ProcessError as error:
                    record_problem(error)
            else:
                empty_scans += 1
                if empty_scans >= _QUIESCENT_SCAN_COUNT and leader_reaped:
                    quiescent = True
                    break
        if not leader_reaped:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            except OSError as error:
                record_problem(error)
            try:
                process.wait(timeout=0)
            except subprocess.TimeoutExpired:
                pass
            else:
                leader_reaped = True
                unreaped_leader = None
        time.sleep(min(0.01, max(0.0, deadline - _monotonic())))

    complete = bool(
        leader_reaped
        and process.returncode is not None
        and quiescent
        and problem_code is None
    )
    if not complete and problem_code is None:
        problem_code = ProcessErrorCode.CLEANUP_UNVERIFIED
        problem_detail = (
            "post-Popen unidentified-leader subtree cleanup was not verified"
        )
    ordered = tuple(
        sorted(observed.values(), key=lambda item: (item.pid, item.start_time_ticks))
    )
    return _CleanupResult(
        complete=complete,
        kill_attempted=True,
        observed_identities=ordered,
        residual_identities=ordered,
        observed_identity_lower_bound=observed_lower_bound,
        observed_identity_complete=observed_complete,
        residual_identity_lower_bound=observed_lower_bound,
        residual_identity_complete=observed_complete,
        maximum_observed_live_descendant_count=(
            maximum_observed_live_descendant_count
        ),
        code=problem_code,
        detail=problem_detail,
    )


def _set_failure(
    current_code: ProcessErrorCode | None,
    current_detail: str | None,
    code: ProcessErrorCode,
    detail: str,
) -> tuple[ProcessErrorCode, str]:
    if current_code is None:
        return code, detail
    return current_code, f"{current_detail}; cleanup={code.value}:{detail}"


def run_bounded_command(
    command: CommandSpec,
    *,
    limits: CommandLimits,
    safety: SafetySpec,
) -> ProcessResult:
    """Execute one command under caller-supplied terminal evidence bounds."""

    _validate_specs(command, limits, safety)
    _validate_existing_directory(command.cwd, "validate-cwd")
    _validate_existing_directory(safety.storage_path, "validate-storage")
    signal_contract = _preflight_signal_contract()
    initial_sample = _sample_safety(safety)

    lease: _ContainmentLease | None = None
    stdout_fd: int | None = None
    stderr_fd: int | None = None
    output_parent_fds: dict[Path, int] = {}
    spawn_holder = _PopenHolder()
    process: subprocess.Popen[bytes] | None = None
    leader: _ProcIdentity | None = None
    pidfd: int | None = None
    selector: selectors.BaseSelector | None = None
    cleanup: _CleanupResult | None = None
    result: ProcessResult | None = None
    containment_restored = False
    samples: list[SafetySample] = [initial_sample]
    stdout_hash = hashlib.sha256()
    stderr_hash = hashlib.sha256()
    counts = {"stdout": 0, "stderr": 0}
    started_ns = _monotonic_ns()
    ended_ns = started_ns
    failure_code: ProcessErrorCode | None = None
    failure_detail: str | None = None
    observed_identities: dict[tuple[int, int], _ProcIdentity] = {}
    observed_identity_lower_bound = 0
    observed_identity_complete = True
    offending_identities: dict[tuple[int, int], _ProcIdentity] = {}
    offending_identity_lower_bound = 0
    offending_identity_complete = True
    maximum_observed_live_descendant_count = 0
    scan_cadence = _ScanCadence()

    def retain_observed(descendants: Sequence[_ProcIdentity]) -> None:
        nonlocal observed_identity_complete
        nonlocal observed_identity_lower_bound
        for identity in descendants:
            key = (identity.pid, identity.start_time_ticks)
            if key in observed_identities:
                observed_identities[key] = identity
            elif len(observed_identities) < limits.max_descendant_identities:
                observed_identities[key] = identity
                if observed_identity_complete:
                    observed_identity_lower_bound = len(observed_identities)
            else:
                observed_identity_complete = False
                observed_identity_lower_bound = max(
                    observed_identity_lower_bound,
                    len(observed_identities) + 1,
                )

    def retain_offending(descendants: Sequence[_ProcIdentity]) -> None:
        nonlocal offending_identity_complete
        nonlocal offending_identity_lower_bound
        for identity in descendants:
            key = (identity.pid, identity.start_time_ticks)
            if key not in observed_identities:
                offending_identity_complete = False
                offending_identity_lower_bound = max(
                    offending_identity_lower_bound,
                    len(offending_identities) + 1,
                )
                continue
            if key in offending_identities:
                offending_identities[key] = identity
            elif len(offending_identities) < limits.max_descendant_identities:
                offending_identities[key] = identity
                if offending_identity_complete:
                    offending_identity_lower_bound = len(offending_identities)
            else:
                offending_identity_complete = False
                offending_identity_lower_bound = max(
                    offending_identity_lower_bound,
                    len(offending_identities) + 1,
                )

    def observe(descendants: Sequence[_ProcIdentity]) -> None:
        nonlocal maximum_observed_live_descendant_count
        maximum_observed_live_descendant_count = max(
            maximum_observed_live_descendant_count,
            len(descendants),
        )
        retain_observed(descendants)

    def restore_after_child_free_proof() -> None:
        nonlocal containment_restored
        if lease is None:
            raise AssertionError("containment lease required for restoration")
        deadline = _monotonic() + limits.cleanup_seconds
        for _ in range(_QUIESCENT_SCAN_COUNT):
            descendants = _scan_descendants(
                lease.owner_pid,
                None,
                limits.max_proc_entries,
                deadline,
                deadline_code=ProcessErrorCode.CLEANUP_TIMEOUT,
            )
            if descendants:
                raise ProcessError(
                    ProcessErrorCode.CONTAINMENT_FAILED,
                    "containment-no-command-cleanup",
                    "direct children exist after command setup failed",
                )
        _restore_containment(lease)
        containment_restored = True

    def restore_after_cleanup() -> None:
        nonlocal cleanup, containment_restored, failure_code, failure_detail
        if lease is None:
            raise AssertionError("containment lease required for restoration")
        if cleanup is None:
            raise AssertionError("cleanup result required for containment restoration")
        if not cleanup.complete:
            _mark_containment_poisoned(
                cleanup.detail or "all-descendant cleanup was not verified",
                lease.previous_subreaper,
            )
            return
        try:
            _restore_containment(lease)
        except ProcessError as error:
            _mark_containment_poisoned(
                f"verified cleanup was followed by subreaper restoration failure: {error}",
                lease.previous_subreaper,
            )
            cleanup = dataclasses.replace(
                cleanup,
                complete=False,
                code=error.code,
                detail=error.detail,
            )
            failure_code, failure_detail = _set_failure(
                failure_code,
                failure_detail,
                error.code,
                error.detail,
            )
            return
        containment_restored = True

    try:
        lease = _enter_containment(limits)
        for output_path in (command.stdout_path, command.stderr_path):
            if output_path.parent not in output_parent_fds:
                output_parent_fds[output_path.parent] = _open_output_parent(output_path)
        stdout_fd = _open_output_exclusive(
            command.stdout_path,
            output_parent_fds[command.stdout_path.parent],
        )
        stderr_fd = _open_output_exclusive(
            command.stderr_path,
            output_parent_fds[command.stderr_path.parent],
        )
        _fsync_open_output(
            stdout_fd,
            command.stdout_path,
            "fsync-output-before-spawn",
        )
        _fsync_open_output(
            stderr_fd,
            command.stderr_path,
            "fsync-output-before-spawn",
        )
        for parent_path, parent_descriptor in output_parent_fds.items():
            _fsync_open_output(
                parent_descriptor,
                parent_path,
                "fsync-output-parent-before-spawn",
            )
        _block_supported_signals(signal_contract, "signal-mask-before-spawn")
        _verify_supported_signal_handlers(
            signal_contract,
            "signal-disposition-after-mask",
        )
        spawn_thread_count = _kernel_thread_count(limits.max_proc_entries)
        if spawn_thread_count != 1:
            raise ProcessError(
                ProcessErrorCode.CALLER_MULTITHREADED,
                "containment-pre-spawn",
                "controller has "
                f"{spawn_thread_count} kernel threads immediately before fork; "
                "exactly one is required",
            )

        def child_preexec() -> None:
            _child_restore_signal_contract(
                signal_contract.caller_mask_numbers
            )

        _verify_supported_signal_handlers(
            signal_contract,
            "signal-disposition-before-fork",
        )
        try:
            process = _spawn_prepublished(
                spawn_holder,
                command.argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=command.cwd,
                env=dict(command.environment),
                close_fds=True,
                start_new_session=True,
                preexec_fn=child_preexec,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ProcessError(ProcessErrorCode.SPAWN_FAILED, "spawn", str(error)) from error
        leader = _read_proc_identity(process.pid)
        if leader is None:
            raise ProcessError(
                ProcessErrorCode.CONTAINMENT_FAILED,
                "leader-identity",
                "spawned leader has no procfs identity while unreaped",
            )
        try:
            pidfd = os.pidfd_open(process.pid, 0)
        except OSError as error:
            raise ProcessError(ProcessErrorCode.PIDFD_FAILED, "pidfd-open", str(error)) from error
        current_leader = _read_proc_identity(process.pid)
        if current_leader is None or not _same_process(current_leader, leader):
            raise ProcessError(
                ProcessErrorCode.PIDFD_FAILED,
                "pidfd-verify",
                "leader PID/start-time identity changed while opening pidfd",
            )
        if process.stdout is None or process.stderr is None:
            raise ProcessError(ProcessErrorCode.PIPE_FAILED, "spawn", "missing child output pipe")
        selector = selectors.DefaultSelector()
        streams = {
            process.stdout.fileno(): ("stdout", stdout_fd, command.stdout_path, limits.stdout_bytes, stdout_hash),
            process.stderr.fileno(): ("stderr", stderr_fd, command.stderr_path, limits.stderr_bytes, stderr_hash),
        }
        for descriptor in streams:
            os.set_blocking(descriptor, False)
            selector.register(descriptor, selectors.EVENT_READ)
        _restore_caller_signal_mask(
            signal_contract,
            "signal-mask-after-spawn-registration",
            primary_exists=False,
        )

        start = _monotonic()
        deadline = start + limits.wall_seconds
        next_safety = start + safety.poll_interval_seconds
        while failure_code is None:
            now = _monotonic()
            if now >= deadline:
                failure_code = ProcessErrorCode.WALL_TIME_LIMIT
                failure_detail = f"wall time exceeded {limits.wall_seconds:g} seconds"
                break
            if now >= next_safety:
                if len(samples) >= safety.max_samples:
                    failure_code = ProcessErrorCode.SAFETY_SAMPLE_LIMIT
                    failure_detail = f"safety sample count reached {safety.max_samples}"
                    break
                try:
                    samples.append(_sample_safety(safety))
                except ProcessError as error:
                    if error.sample is not None:
                        samples.append(error.sample)
                    failure_code, failure_detail = error.code, error.detail
                    break
                try:
                    live_during_execution = _scan_descendants(
                        lease.owner_pid,
                        leader,
                        limits.max_proc_entries,
                        deadline,
                        deadline_code=ProcessErrorCode.WALL_TIME_LIMIT,
                    )
                    scan_cadence.record()
                except ProcessError as error:
                    failure_code, failure_detail = error.code, error.detail
                    break
                observe(live_during_execution)
                try:
                    _reap_direct_zombies(
                        lease.owner_pid,
                        leader,
                        live_during_execution,
                    )
                except ProcessError as error:
                    failure_code, failure_detail = error.code, error.detail
                    break
                if len(live_during_execution) > limits.max_observed_live_descendants:
                    retain_offending(live_during_execution)
                    failure_code = ProcessErrorCode.DESCENDANT_LIMIT
                    failure_detail = (
                        "observed live descendant count "
                        f"{len(live_during_execution)} exceeded "
                        f"{limits.max_observed_live_descendants}"
                    )
                    break
                if not observed_identity_complete:
                    retain_offending(observed_identities.values())
                    failure_code = ProcessErrorCode.DESCENDANT_IDENTITY_LIMIT
                    failure_detail = (
                        "observed descendant identity count exceeded "
                        f"{limits.max_descendant_identities}"
                    )
                    break
                next_safety = now + safety.poll_interval_seconds

            leader_done = _leader_exited(pidfd, process.pid)
            if leader_done:
                live = _scan_descendants(
                    lease.owner_pid,
                    leader,
                    limits.max_proc_entries,
                    deadline,
                    deadline_code=ProcessErrorCode.WALL_TIME_LIMIT,
                )
                scan_cadence.record()
                observe(live)
                if live:
                    retain_offending(live)
                    failure_code = (
                        ProcessErrorCode.DESCENDANT_LIMIT
                        if len(live) > limits.max_observed_live_descendants
                        else ProcessErrorCode.DESCENDANT_REMAINS
                    )
                    failure_detail = (
                        f"leader exited with {len(live)} observed residual descendants"
                    )
                    break
                if not observed_identity_complete:
                    retain_offending(observed_identities.values())
                    failure_code = ProcessErrorCode.DESCENDANT_IDENTITY_LIMIT
                    failure_detail = (
                        "observed descendant identity count exceeded "
                        f"{limits.max_descendant_identities}"
                    )
                    break
                if not selector.get_map():
                    break

            wait = min(0.05, deadline - now, max(0.0, next_safety - now))
            for key, _ in selector.select(max(0.0, wait)):
                descriptor = key.fd
                label, output_fd, output_path, maximum, digest = streams[descriptor]
                try:
                    chunk = os.read(descriptor, min(limits.read_chunk_bytes, maximum - counts[label] + 1))
                except BlockingIOError:
                    continue
                except OSError as error:
                    failure_code = ProcessErrorCode.PIPE_FAILED
                    failure_detail = f"cannot read {label}: {error}"
                    break
                if not chunk:
                    selector.unregister(descriptor)
                    continue
                available = maximum - counts[label]
                accepted = chunk[:available]
                if accepted:
                    _write_all(output_fd, accepted, output_path)
                    digest.update(accepted)
                    counts[label] += len(accepted)
                if len(chunk) > available:
                    failure_code = (
                        ProcessErrorCode.STDOUT_LIMIT
                        if label == "stdout"
                        else ProcessErrorCode.STDERR_LIMIT
                    )
                    failure_detail = f"{label} exceeded {maximum} bytes"
                    break

        _block_supported_signals(signal_contract, "signal-mask-before-cleanup")
        cleanup = _cleanup(
            process,
            pidfd,
            leader,
            lease,
            limits=limits,
            force_kill=failure_code is not None,
            previously_observed=observed_identities,
            previously_observed_lower_bound=observed_identity_lower_bound,
            previously_observed_complete=observed_identity_complete,
            scan_cadence=scan_cadence,
        )
        observed_identities = {
            (identity.pid, identity.start_time_ticks): identity
            for identity in cleanup.observed_identities
        }
        observed_identity_lower_bound = cleanup.observed_identity_lower_bound
        observed_identity_complete = cleanup.observed_identity_complete
        maximum_observed_live_descendant_count = max(
            maximum_observed_live_descendant_count,
            cleanup.maximum_observed_live_descendant_count,
        )
        retain_offending(cleanup.residual_identities)
        offending_identity_lower_bound = max(
            offending_identity_lower_bound,
            cleanup.residual_identity_lower_bound,
        )
        offending_identity_complete = bool(
            offending_identity_complete and cleanup.residual_identity_complete
        )
        if cleanup.residual_identities and failure_code is None:
            failure_code = (
                ProcessErrorCode.DESCENDANT_LIMIT
                if cleanup.maximum_observed_live_descendant_count
                > limits.max_observed_live_descendants
                else ProcessErrorCode.DESCENDANT_REMAINS
            )
            failure_detail = (
                "terminal cleanup observed residual command descendants"
            )
        if not observed_identity_complete:
            if failure_code is None:
                failure_code = ProcessErrorCode.DESCENDANT_IDENTITY_LIMIT
                failure_detail = (
                    "observed descendant identity count exceeded "
                    f"{limits.max_descendant_identities}"
                )
            else:
                failure_code, failure_detail = _set_failure(
                    failure_code,
                    failure_detail,
                    ProcessErrorCode.DESCENDANT_IDENTITY_LIMIT,
                    "cumulative observed descendant identity ledger overflowed",
                )
            retain_offending(observed_identities.values())
            offending_identity_complete = False
            offending_identity_lower_bound = max(
                offending_identity_lower_bound,
                observed_identity_lower_bound,
            )
        if cleanup.code is not None:
            failure_code, failure_detail = _set_failure(
                failure_code,
                failure_detail,
                cleanup.code,
                cleanup.detail or "cleanup failed",
            )
        if failure_code is None and process.returncode != 0:
            failure_code = ProcessErrorCode.PROCESS_EXIT_NONZERO
            failure_detail = f"process exited with status {process.returncode}"
        restore_after_cleanup()
        ended_ns = _monotonic_ns()
        ordered_observed = tuple(
            sorted(
                observed_identities.values(),
                key=lambda item: (item.pid, item.start_time_ticks),
            )
        )
        ordered_offending = tuple(
            sorted(
                offending_identities.values(),
                key=lambda item: (item.pid, item.start_time_ticks),
            )
        )
        result = ProcessResult(
            argv=tuple(command.argv),
            cwd=str(command.cwd),
            pid=process.pid,
            returncode=process.returncode,
            started_monotonic_ns=started_ns,
            ended_monotonic_ns=ended_ns,
            stdout_bytes=counts["stdout"],
            stderr_bytes=counts["stderr"],
            stdout_sha256=stdout_hash.hexdigest(),
            stderr_sha256=stderr_hash.hexdigest(),
            safety_samples=tuple(samples),
            failure_code=failure_code,
            failure_detail=failure_detail,
            kill_attempted=cleanup.kill_attempted,
            cleanup_complete=cleanup.complete,
            cleanup_failure_code=cleanup.code,
            cleanup_failure_detail=cleanup.detail,
            unexpected_descendant_identity_lower_bound=(
                offending_identity_lower_bound
            ),
            unexpected_descendant_identity_complete=(
                offending_identity_complete
            ),
            unexpected_descendant_pids=tuple(
                item.pid for item in ordered_offending[:_MAX_REPORTED_PIDS]
            ),
            maximum_observed_live_descendant_count=maximum_observed_live_descendant_count,
            observed_descendant_identity_lower_bound=(
                observed_identity_lower_bound
            ),
            observed_descendant_identity_complete=observed_identity_complete,
            observed_descendant_identities=tuple(
                (item.pid, item.start_time_ticks)
                for item in ordered_observed[:_MAX_REPORTED_PIDS]
            ),
            observed_descendant_identities_truncated=(
                not observed_identity_complete
                or len(ordered_observed) > _MAX_REPORTED_PIDS
            ),
            unexpected_descendant_identities=tuple(
                (item.pid, item.start_time_ticks)
                for item in ordered_offending[:_MAX_REPORTED_PIDS]
            ),
            unexpected_descendant_identities_truncated=(
                not offending_identity_complete
                or len(ordered_offending) > _MAX_REPORTED_PIDS
            ),
            caller_signal_mask_numbers=(
                signal_contract.caller_mask_numbers
            ),
            signal_mask_restored=False,
            deferred_supported_signal_numbers=tuple(
                sorted(signal_contract.deferred_supported_signal_numbers)
            ),
            containment_mode=CONTAINMENT_MODEL,
            containment_caller_thread_count=lease.caller_thread_count,
            containment_preexisting_child_count=lease.preexisting_child_count,
            containment_subreaper_previously_enabled=lease.previous_subreaper,
            containment_subreaper_restored=containment_restored,
            descendant_scan_count=scan_cadence.count,
            first_descendant_scan_monotonic_ns=scan_cadence.first_monotonic_ns,
            last_descendant_scan_monotonic_ns=scan_cadence.last_monotonic_ns,
            maximum_observed_descendant_scan_gap_ns=scan_cadence.maximum_gap_ns,
        )
    except BaseException as error:
        if lease is None:
            raise
        if process is None:
            process = _recover_published_process(spawn_holder)
        if not signal_contract.blocked:
            try:
                _block_supported_signals(
                    signal_contract,
                    "signal-mask-before-exception-cleanup",
                )
            except BaseException as mask_error:
                error.add_note(
                    "supported-signal cleanup mask raised "
                    f"{type(mask_error).__name__}: {mask_error}"
                )
                if process is not None:
                    _mark_containment_poisoned(
                        "supported-signal cleanup mask could not be established: "
                        f"{type(mask_error).__name__}: {mask_error}",
                        lease.previous_subreaper,
                    )
        if process is not None and cleanup is None:
            if leader is not None:
                try:
                    cleanup = _cleanup(
                        process,
                        pidfd,
                        leader,
                        lease,
                        limits=limits,
                        force_kill=True,
                        previously_observed=observed_identities,
                        previously_observed_lower_bound=(
                            observed_identity_lower_bound
                        ),
                        previously_observed_complete=observed_identity_complete,
                        scan_cadence=scan_cadence,
                    )
                except BaseException as cleanup_error:
                    error.add_note(
                        "bounded cancellation cleanup raised "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                    _mark_containment_poisoned(
                        f"cleanup raised {type(cleanup_error).__name__}: {cleanup_error}",
                        lease.previous_subreaper,
                    )
            else:
                try:
                    cleanup = _cleanup_unidentified_leader(
                        process,
                        lease,
                        limits=limits,
                        scan_cadence=scan_cadence,
                    )
                except BaseException as cleanup_error:
                    error.add_note(
                        "unidentified-leader cleanup raised "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                    _mark_containment_poisoned(
                        "unidentified-leader all-descendant cleanup raised "
                        f"{type(cleanup_error).__name__}: {cleanup_error}",
                        lease.previous_subreaper,
                    )
        if cleanup is not None and not containment_restored:
            if cleanup.complete:
                try:
                    _restore_containment(lease)
                except BaseException as restore_error:
                    _mark_containment_poisoned(
                        f"exception cleanup restoration failed: {restore_error}",
                        lease.previous_subreaper,
                    )
                    error.add_note(
                        "subreaper restoration raised "
                        f"{type(restore_error).__name__}: {restore_error}"
                    )
                else:
                    containment_restored = True
            else:
                _mark_containment_poisoned(
                    cleanup.detail or "exception cleanup was not verified",
                    lease.previous_subreaper,
                )
        elif process is None and not containment_restored:
            try:
                restore_after_child_free_proof()
            except BaseException as restore_error:
                _mark_containment_poisoned(
                    f"no-command restoration failed: {restore_error}",
                    lease.previous_subreaper,
                )
                error.add_note(
                    "no-command subreaper restoration raised "
                    f"{type(restore_error).__name__}: {restore_error}"
                )
        if cleanup is not None:
            error.add_note(
                f"bounded cleanup complete={int(cleanup.complete)} kill_attempted={int(cleanup.kill_attempted)} "
                f"residual_descendants={len(cleanup.residual_identities)} "
                f"error={cleanup.detail or 'none'}"
            )
        raise
    finally:
        active_error = sys.exc_info()[1]
        close_failures: list[BaseException] = []
        finalizer_error: BaseException | None = None
        if lease is not None:
            try:
                if (
                    not containment_restored
                    and _CONTAINMENT_POISONED_DETAIL is None
                ):
                    _mark_containment_poisoned(
                        "run exited without verified cleanup and subreaper restoration",
                        lease.previous_subreaper,
                    )
            finally:
                _RUN_LOCK.release()
        closers: list[object] = []
        if selector is not None:
            closers.append(selector.close)
        if process is not None and process.stdout is not None:
            closers.append(process.stdout.close)
        if process is not None and process.stderr is not None:
            closers.append(process.stderr.close)
        if pidfd is not None:
            closers.append(lambda: os.close(pidfd))
        if stdout_fd is not None:
            closers.append(
                lambda: _close_output_file(stdout_fd, command.stdout_path)
            )
        if stderr_fd is not None:
            closers.append(
                lambda: _close_output_file(stderr_fd, command.stderr_path)
            )
        for parent_path, parent_descriptor in output_parent_fds.items():
            closers.append(
                lambda fd=parent_descriptor, path=parent_path: _fsync_output_parent(fd, path)
            )
        for close in closers:
            try:
                close()  # type: ignore[operator]
            except BaseException as close_error:
                close_failures.append(close_error)
        if close_failures:
            note = "; ".join(f"{type(item).__name__}:{item}" for item in close_failures)
            if active_error is not None:
                active_error.add_note(f"bounded descriptor close failures: {note}")
            elif result is not None:
                for close_error in close_failures:
                    close_code = (
                        close_error.code
                        if isinstance(close_error, ProcessError)
                        else ProcessErrorCode.OUTPUT_WRITE_FAILED
                    )
                    close_detail = (
                        close_error.detail
                        if isinstance(close_error, ProcessError)
                        else str(close_error)
                    )
                    if result.failure_code is None:
                        result = dataclasses.replace(
                            result,
                            failure_code=close_code,
                            failure_detail=close_detail,
                        )
                    else:
                        result = dataclasses.replace(
                            result,
                            failure_detail=(
                                f"{result.failure_detail}; finalization="
                                f"{close_code.value}:{close_detail}"
                            ),
                        )
            else:
                finalizer_error = close_failures[0]
        if signal_contract.blocked:
            primary = (
                active_error
                if active_error is not None
                else finalizer_error
            )
            try:
                _restore_caller_signal_mask(
                    signal_contract,
                    "signal-mask-final-restore",
                    primary_exists=primary is not None,
                )
            except BaseException as restore_error:
                if primary is not None:
                    _attach_signal_drain_notes(signal_contract, primary)
                    primary.add_note(
                        "caller signal drain/restore raised "
                        f"{type(restore_error).__name__}: {restore_error}; "
                        "the original exception remains primary"
                    )
                else:
                    _attach_signal_drain_notes(
                        signal_contract,
                        restore_error,
                    )
                    raise
            else:
                if primary is not None:
                    _attach_signal_drain_notes(signal_contract, primary)
                if result is not None:
                    result = dataclasses.replace(
                        result,
                        signal_mask_restored=True,
                        deferred_supported_signal_numbers=tuple(
                            sorted(
                                signal_contract.deferred_supported_signal_numbers
                            )
                        ),
                    )
        if finalizer_error is not None:
            raise finalizer_error
    if result is None:
        raise AssertionError("bounded command produced no result")
    return result


def run_process(
    argv: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    limits: CommandLimits,
    safety: SafetySpec,
) -> ProcessResult:
    """Compatibility wrapper around :func:`run_bounded_command`."""

    return run_bounded_command(
        CommandSpec(tuple(argv), cwd, environment, stdout_path, stderr_path),
        limits=limits,
        safety=safety,
    )
