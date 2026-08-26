#!/usr/bin/env python3
"""Authenticate, prepare, and exec one profiled OAI NR softmodem process.

The launcher is deliberately a same-process ``execve`` handoff.  It validates
the measured build evidence and separately pinned trusted-release authority
before publishing any bootstrap byte, derives the realloc-zero policy from the
selected build domain, and supplies only the strict environment consumed by
``oai_memprof_softmodem_session.c``.  It never uses a shell and never accepts a
policy, semantic-oracle, or trusted-release digest from request JSON.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib.util
import os
import pathlib
import stat
import sys
import uuid
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Mapping, Sequence


ROOT = pathlib.Path(__file__).resolve().parents[3]
COMPOSER_PATH = pathlib.Path(__file__).resolve().with_name(
    "oai_memprof_archive_composer.py"
)
MAX_REQUEST_BYTES = 1 << 20
MAX_BUILD_COVERAGE_BYTES = 16 << 20
MAX_BUILD_EVIDENCE_BYTES = 16 << 20
MAX_BUILD_ARTIFACT_BYTES = 512 << 20
MAX_BUILD_ARTIFACT_ENTRIES = 256
MAX_BUILD_ARTIFACT_TOTAL_BYTES = 1 << 30
MAX_BINARY_BYTES = 2 << 30
MAX_ARGUMENT_BYTES = 1 << 20
MAX_ARGUMENT_COUNT = 4096
MAX_EXEC_DESCRIPTOR_COUNT = 4096
MAX_PROCESS_GENERATION = (1 << 48) - 1
MAX_THREADS = 65534
MAX_RING_RECORDS = 1 << 20
MAX_FLUSH_RECORDS = 65536
CONFIGURATION_LEAF = "effective-config.json"
OPENING_LEAF = "opening.bin"
API_DEFINITION_SHA256 = (
    "93056c4cfd071c1df396ba09bf82b4cbe807923977c4bca988b0aee1b8c94610"
)
SESSION_ENVIRONMENT_NAMES = (
    "OAI_MEMPROF_SESSION_ENABLE",
    "OAI_MEMPROF_SESSION_ARCHIVE_FD",
    "OAI_MEMPROF_SESSION_BOOTSTRAP_FD",
    "OAI_MEMPROF_SESSION_PROCESS_GENERATION",
    "OAI_MEMPROF_SESSION_MAX_THREADS",
    "OAI_MEMPROF_SESSION_RING_RECORDS",
    "OAI_MEMPROF_SESSION_MODE_ID",
    "OAI_MEMPROF_SESSION_TABLE_ENTRIES",
    "OAI_MEMPROF_SESSION_SAMPLE_SEED",
    "OAI_MEMPROF_SESSION_SAMPLE_THRESHOLD",
    "OAI_MEMPROF_SESSION_TABLE_PROBES",
    "OAI_MEMPROF_SESSION_REALLOC_ZERO_POLICY_ID",
    "OAI_MEMPROF_SESSION_FLUSH_RECORDS",
    "OAI_MEMPROF_SESSION_FLUSH_INTERVAL_NS",
    "OAI_MEMPROF_SESSION_SEAL_TIMEOUT_NS",
)
# ``os.execve`` accepts a descriptor only where the platform exposes a real
# descriptor-exec primitive. A pathname fallback would reintroduce the race
# this launcher exists to prevent, so unsupported platforms fail closed.
EXECVE_SUPPORTS_FD = os.execve in getattr(os, "supports_fd", ())


class SoftmodemLauncherError(RuntimeError):
    """A deterministic request, evidence, publication, or exec rejection."""


def _load(name: str, path: pathlib.Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise SoftmodemLauncherError(f"module unavailable: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except Exception as error:
        raise SoftmodemLauncherError(f"module failed to load: {path}") from error
    return module


COMPOSER = _load("_oai_memprof_softmodem_launcher_composer", COMPOSER_PATH)
VERIFIER = COMPOSER.VERIFIER
BUILD_EVIDENCE = COMPOSER.BUILD_EVIDENCE
COVERAGE = VERIFIER.COVERAGE
CONFIG = VERIFIER.CONFIG
WIRE = VERIFIER.WIRE


@dataclass(frozen=True)
class AuthenticatedBuild:
    build_raw: bytes
    evidence_raw: bytes
    build: Mapping[str, Any]
    primary: Mapping[str, Any]
    binary_path: pathlib.Path


@dataclass
class PreparedLaunch:
    archive_directory: pathlib.Path
    bootstrap_directory: pathlib.Path
    configuration_path: pathlib.Path
    opening_path: pathlib.Path
    binary_path: pathlib.Path
    working_directory: pathlib.Path
    archive_directory_fd: int
    bootstrap_directory_fd: int
    working_directory_fd: int
    binary_byte_count: int
    binary_sha256: str
    argv: tuple[str, ...]
    session_environment: tuple[tuple[str, str], ...]
    configuration_sha256: str
    opening_sha256: str
    build_coverage_sha256: str
    build_evidence_sha256: str
    trusted_release_authority_sha256: str
    realloc_zero_policy_id: int


@dataclass(frozen=True)
class _DescriptorInheritanceState:
    descriptor: int
    identity: tuple[int, int, int]
    inheritable: bool


@dataclass
class _ExecDescriptorHygiene:
    root_states: tuple[_DescriptorInheritanceState, _DescriptorInheritanceState]
    cloexec_changed: list[_DescriptorInheritanceState]


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_hex(value: Any, where: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{where}: lowercase SHA-256 hex required",
    )
    return value


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise SoftmodemLauncherError(detail)


def _uint(
    value: Any,
    bits: int,
    where: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    _require(type(value) is int, f"{where}: u{bits} integer required")
    upper = (1 << bits) - 1 if maximum is None else maximum
    _require(minimum <= value <= upper, f"{where}: outside admitted range")
    return value


def _exact_keys(value: Any, keys: Sequence[str], where: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{where}: object required")
    _require(
        len(value) == len(keys) and set(value) == set(keys),
        f"{where}: exact members {tuple(sorted(keys))!r} required",
    )
    return value


def _canonical_existing_path(value: Any, where: str, *, directory: bool) -> pathlib.Path:
    _require(isinstance(value, str) and value.startswith("/"), f"{where}: absolute path required")
    _require("\x00" not in value and len(value.encode("utf-8")) <= 4096, f"{where}: invalid path")
    path = pathlib.Path(value)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SoftmodemLauncherError(f"{where}: unavailable path") from error
    _require(str(resolved) == value, f"{where}: canonical resolved path required")
    status = os.lstat(path)
    if directory:
        _require(stat.S_ISDIR(status.st_mode), f"{where}: real directory required")
    else:
        _require(
            stat.S_ISREG(status.st_mode) and status.st_nlink == 1,
            f"{where}: single-link regular file required",
        )
    return resolved


def _same_file_status(before: Any, after: Any) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )


def _open_directory(
    path: pathlib.Path,
    where: str,
    *,
    require_private: bool,
    require_empty: bool,
) -> int:
    descriptor = -1
    retained = False
    try:
        descriptor = os.open(
            path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        before = os.fstat(descriptor)
        _require(stat.S_ISDIR(before.st_mode), f"{where}: real directory required")
        if require_private:
            _require(
                before.st_uid == os.geteuid(),
                f"{where}: current-user ownership required",
            )
            _require(
                before.st_mode & 0o022 == 0,
                f"{where}: group/other write forbidden",
            )
        if require_empty:
            _require(
                not os.listdir(descriptor),
                f"{where}: existing empty directory required",
            )
        after = os.fstat(descriptor)
        _require(
            _same_file_status(before, after),
            f"{where}: directory changed during validation",
        )
        current = os.lstat(path)
        _require(
            (current.st_dev, current.st_ino, current.st_mode)
            == (before.st_dev, before.st_ino, before.st_mode),
            f"{where}: pathname changed during validation",
        )
        retained = True
        return descriptor
    except OSError as error:
        raise SoftmodemLauncherError(f"{where}: unavailable directory") from error
    finally:
        if descriptor >= 0 and not retained:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _open_empty_private_directory(value: Any, where: str) -> tuple[pathlib.Path, int]:
    path = _canonical_existing_path(value, where, directory=True)
    return path, _open_directory(
        path, where, require_private=True, require_empty=True
    )


def _open_authenticated_binary(
    path: pathlib.Path,
    expected_byte_count: int,
    expected_sha256: str,
) -> int:
    _require(
        type(expected_byte_count) is int
        and 0 < expected_byte_count <= MAX_BINARY_BYTES,
        "prepared binary: bounded expected byte count required",
    )
    expected_sha256 = _sha256_hex(expected_sha256, "prepared binary")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    descriptor = -1
    retained = False
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        _require(
            stat.S_ISREG(before.st_mode)
            and before.st_nlink == 1
            and before.st_size == expected_byte_count
            and before.st_size <= MAX_BINARY_BYTES
            and before.st_mode & 0o111,
            "request.argv[0]: final executable identity mismatch",
        )
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1 << 20))
            _require(chunk, "request.argv[0]: final executable short read")
            digest.update(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        # This is a point-in-time check. A same-UID actor able to mutate the
        # already-open inode after this fstat can defeat any userspace-only
        # immutability claim; pathname swaps, links, and observable rewrites
        # before this point fail closed and execution still uses this exact fd.
        _require(
            _same_file_status(before, after),
            "request.argv[0]: changed during final authentication",
        )
        _require(
            digest.hexdigest() == expected_sha256,
            "request.argv[0]: bytes differ from measured primary executable",
        )
        retained = True
        return descriptor
    except OSError as error:
        raise SoftmodemLauncherError(
            "request.argv[0]: final executable open or verification failed"
        ) from error
    finally:
        if descriptor >= 0 and not retained:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _close_prepared_fds(prepared: PreparedLaunch) -> None:
    first_error: OSError | None = None
    for attribute in (
        "archive_directory_fd",
        "bootstrap_directory_fd",
        "working_directory_fd",
    ):
        descriptor = getattr(prepared, attribute)
        if descriptor < 0:
            continue
        # Clear before close: if close reports an error, retrying later could
        # close an unrelated descriptor after the kernel has reused its number.
        setattr(prepared, attribute, -1)
        try:
            os.close(descriptor)
        except OSError as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise SoftmodemLauncherError("prepared launch: descriptor cleanup failed") from first_error


def close_prepared_launch(prepared: PreparedLaunch) -> None:
    _require(type(prepared) is PreparedLaunch, "prepared launch: exact result required")
    _close_prepared_fds(prepared)


def _validate_directory_fd(descriptor: int, where: str) -> None:
    _require(type(descriptor) is int and descriptor >= 3, f"{where}: live descriptor required")
    try:
        status = os.fstat(descriptor)
    except OSError as error:
        raise SoftmodemLauncherError(f"{where}: unavailable descriptor") from error
    _require(stat.S_ISDIR(status.st_mode), f"{where}: directory descriptor required")


def _descriptor_identity(status: os.stat_result) -> tuple[int, int, int]:
    return (status.st_dev, status.st_ino, stat.S_IFMT(status.st_mode))


def _descriptor_inheritance_state(
    descriptor: int,
    where: str,
    *,
    vanished_ok: bool,
) -> _DescriptorInheritanceState | None:
    try:
        before = os.fstat(descriptor)
        inheritable = os.get_inheritable(descriptor)
        after = os.fstat(descriptor)
    except OSError as error:
        if vanished_ok and error.errno == errno.EBADF:
            return None
        raise SoftmodemLauncherError(f"{where}: descriptor state unavailable") from error
    _require(
        _descriptor_identity(before) == _descriptor_identity(after),
        f"{where}: descriptor changed during state check",
    )
    return _DescriptorInheritanceState(
        descriptor=descriptor,
        identity=_descriptor_identity(before),
        inheritable=inheritable,
    )


def _set_descriptor_inheritable(
    state: _DescriptorInheritanceState, inheritable: bool, where: str
) -> None:
    current = _descriptor_inheritance_state(
        state.descriptor, where, vanished_ok=False
    )
    _require(
        current is not None and current.identity == state.identity,
        f"{where}: descriptor identity changed before inheritance update",
    )
    try:
        os.set_inheritable(state.descriptor, inheritable)
    except OSError as error:
        raise SoftmodemLauncherError(
            f"{where}: descriptor inheritance mutation failed"
        ) from error
    observed = _descriptor_inheritance_state(
        state.descriptor, where, vanished_ok=False
    )
    _require(
        observed is not None
        and observed.identity == state.identity
        and observed.inheritable == inheritable,
        f"{where}: descriptor inheritance mutation was not established",
    )


def _restore_descriptor_inheritance(
    state: _DescriptorInheritanceState, where: str
) -> None:
    current = _descriptor_inheritance_state(
        state.descriptor, where, vanished_ok=True
    )
    # A /proc enumeration descriptor has already disappeared by this point;
    # likewise, a descriptor closed by a concurrent actor cannot cross exec.
    if current is None:
        return
    _require(
        current.identity == state.identity,
        f"{where}: descriptor identity changed before inheritance restoration",
    )
    if current.inheritable != state.inheritable:
        _set_descriptor_inheritable(state, state.inheritable, where)


def _open_descriptors_for_exec_sweep() -> tuple[int, ...]:
    descriptors: list[int] = []
    try:
        # The scandir descriptor is visible in /proc/self/fd while this block
        # runs, then disappears at block exit before the mutation pass below.
        with os.scandir("/proc/self/fd") as entries:
            for entry in entries:
                name = entry.name
                _require(
                    isinstance(name, str) and name.isascii() and name.isdecimal(),
                    "prepared launch: malformed /proc/self/fd entry",
                )
                descriptor = int(name)
                if descriptor >= 3:
                    descriptors.append(descriptor)
                    _require(
                        len(descriptors) <= MAX_EXEC_DESCRIPTOR_COUNT,
                        "prepared launch: descriptor sweep limit exceeded",
                    )
    except OSError as error:
        raise SoftmodemLauncherError(
            "prepared launch: descriptor enumeration unavailable"
        ) from error
    _require(
        len(set(descriptors)) == len(descriptors),
        "prepared launch: duplicate /proc/self/fd entry",
    )
    return tuple(descriptors)


def _require_single_threaded_exec() -> None:
    task_count = 0
    try:
        with os.scandir("/proc/self/task") as entries:
            for entry in entries:
                name = entry.name
                _require(
                    isinstance(name, str) and name.isascii() and name.isdecimal(),
                    "prepared launch: malformed /proc/self/task entry",
                )
                task_count += 1
                _require(
                    task_count <= 1,
                    "prepared launch: descriptor exec requires one process thread",
                )
    except OSError as error:
        raise SoftmodemLauncherError(
            "prepared launch: process-thread enumeration unavailable"
        ) from error
    _require(
        task_count == 1,
        "prepared launch: descriptor exec requires one process thread",
    )


def _restore_exec_descriptor_hygiene(hygiene: _ExecDescriptorHygiene) -> None:
    first_error: BaseException | None = None
    for state in reversed(hygiene.cloexec_changed):
        try:
            _restore_descriptor_inheritance(state, "prepared launch: descriptor sweep")
        except BaseException as error:
            if first_error is None:
                first_error = error
    for state in hygiene.root_states:
        try:
            _restore_descriptor_inheritance(state, "prepared launch: root handoff")
        except BaseException as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise SoftmodemLauncherError(
            "prepared launch: descriptor inheritance restoration failed"
        ) from first_error


def _prepare_exec_descriptor_hygiene(
    archive_directory_fd: int, bootstrap_directory_fd: int
) -> _ExecDescriptorHygiene:
    _require(
        archive_directory_fd != bootstrap_directory_fd,
        "prepared launch: distinct archive/bootstrap roots required",
    )
    archive_state = _descriptor_inheritance_state(
        archive_directory_fd,
        "prepared launch: archive root",
        vanished_ok=False,
    )
    bootstrap_state = _descriptor_inheritance_state(
        bootstrap_directory_fd,
        "prepared launch: bootstrap root",
        vanished_ok=False,
    )
    _require(
        archive_state is not None and bootstrap_state is not None,
        "prepared launch: root descriptors unavailable",
    )
    hygiene = _ExecDescriptorHygiene(
        root_states=(archive_state, bootstrap_state),
        cloexec_changed=[],
    )
    allowed = {archive_directory_fd, bootstrap_directory_fd}
    try:
        # This is a bounded Linux /proc snapshot, not an atomic fd-allocation
        # barrier. execute() must run in the launcher's controlled
        # single-threaded phase: a same-process thread that opens or reuses an
        # fd after this sweep and before execve can race any userspace-only
        # close-on-exec protocol.
        for descriptor in _open_descriptors_for_exec_sweep():
            if descriptor in allowed:
                continue
            state = _descriptor_inheritance_state(
                descriptor,
                "prepared launch: descriptor sweep",
                vanished_ok=True,
            )
            if state is None or not state.inheritable:
                continue
            # Enter the rollback ledger before mutation: a successful fcntl
            # followed by a failed verification must still be reversible.
            hygiene.cloexec_changed.append(state)
            _set_descriptor_inheritable(
                state, False, "prepared launch: descriptor sweep"
            )
        for state in hygiene.root_states:
            _set_descriptor_inheritable(
                state, True, "prepared launch: root handoff"
            )
        return hygiene
    except BaseException:
        try:
            _restore_exec_descriptor_hygiene(hygiene)
        except BaseException as cleanup_error:
            raise SoftmodemLauncherError(
                "prepared launch: descriptor inheritance setup cleanup failed"
            ) from cleanup_error
        raise


def _canonical_uuid(value: Any, where: str) -> uuid.UUID:
    _require(isinstance(value, str), f"{where}: lowercase UUID string required")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as error:
        raise SoftmodemLauncherError(f"{where}: lowercase UUID string required") from error
    _require(str(parsed) == value and parsed.int != 0, f"{where}: canonical nonnil UUID required")
    _require(parsed.variant == uuid.RFC_4122, f"{where}: RFC 4122 variant required")
    return parsed


def _argv(value: Any) -> tuple[str, ...]:
    _require(isinstance(value, list) and value, "request.argv: nonempty array required")
    _require(len(value) <= MAX_ARGUMENT_COUNT, "request.argv: too many arguments")
    copied: list[str] = []
    total = 0
    for index, argument in enumerate(value):
        _require(isinstance(argument, str), f"request.argv[{index}]: string required")
        encoded = argument.encode("utf-8")
        _require(encoded and b"\0" not in encoded, f"request.argv[{index}]: nonempty NUL-free string required")
        total += len(encoded) + 1
        _require(total <= MAX_ARGUMENT_BYTES, "request.argv: bounded encoded size exceeded")
        copied.append(argument)
    _require(copied[0].startswith("/"), "request.argv[0]: absolute executable path required")
    return tuple(copied)


def _request(raw: bytes) -> dict[str, Any]:
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_REQUEST_BYTES, "request: bounded immutable bytes required")
    try:
        value = COVERAGE.parse_canonical(raw)
    except Exception as error:
        raise SoftmodemLauncherError(str(error)) from error
    _exact_keys(
        value,
        (
            "archive_directory",
            "argv",
            "bootstrap_directory",
            "build_coverage_path",
            "build_evidence_path",
            "build_evidence_root",
            "flush_records",
            "flush_us",
            "max_threads",
            "mode_id",
            "process_generation",
            "process_uuid",
            "ring_records",
            "role_id",
            "run_id",
            "run_uuid",
            "sample_threshold",
            "scope_kind",
            "seal_timeout_ns",
            "selection_values",
            "table_entries",
            "table_probes",
            "working_directory",
        ),
        "request",
    )
    return value


def _read_build_artifacts(
    evidence_raw: bytes, evidence_root: pathlib.Path
) -> dict[str, bytes]:
    try:
        evidence = COVERAGE.parse_canonical(evidence_raw)
    except Exception as error:
        raise SoftmodemLauncherError(str(error)) from error
    entries = evidence.get("entries")
    _require(isinstance(entries, list) and entries, "build evidence: nonempty artifact manifest required")
    _require(
        len(entries) <= MAX_BUILD_ARTIFACT_ENTRIES,
        "build evidence: artifact-entry limit exceeded",
    )
    plan: list[tuple[str, int, str]] = []
    seen_paths: set[str] = set()
    declared_total = 0
    for index, row in enumerate(entries):
        _exact_keys(row, ("bytes", "path", "sha256"), f"build evidence entry {index}")
        relative = VERIFIER._archive_path(row["path"], f"build evidence entry {index} path")
        _require(relative.startswith("input/build-evidence/"), f"build evidence entry {index}: owned path required")
        declared_bytes = _uint(
            row["bytes"],
            64,
            f"build evidence entry {index} bytes",
            maximum=MAX_BUILD_ARTIFACT_BYTES,
        )
        declared_sha256 = _sha256_hex(
            row["sha256"], f"build evidence entry {index} sha256"
        )
        _require(
            relative not in seen_paths,
            f"build evidence entry {index}: duplicate path",
        )
        declared_total += declared_bytes
        _require(
            declared_total <= MAX_BUILD_ARTIFACT_TOTAL_BYTES,
            "build evidence: aggregate declared artifact bytes exceeded",
        )
        seen_paths.add(relative)
        plan.append((relative, declared_bytes, declared_sha256))

    artifacts: dict[str, bytes] = {}
    actual_total = 0
    for index, (relative, declared_bytes, declared_sha256) in enumerate(plan):
        # The declared preflight makes the maximum below a per-item and
        # remaining-aggregate bound before this pathname is opened.
        remaining_budget = MAX_BUILD_ARTIFACT_TOTAL_BYTES - actual_total
        try:
            artifact = COMPOSER._read_frozen(
                evidence_root.joinpath(*relative.split("/")),
                min(declared_bytes, remaining_budget),
                allow_empty=True,
            )
        except Exception as error:
            raise SoftmodemLauncherError(
                f"build evidence entry {index}: bounded artifact read failed"
            ) from error
        actual_total += artifact.size
        _require(
            actual_total <= MAX_BUILD_ARTIFACT_TOTAL_BYTES,
            "build evidence: aggregate actual artifact bytes exceeded",
        )
        _require(
            artifact.size == declared_bytes and _sha(artifact.raw) == declared_sha256,
            f"build evidence entry {index}: path/size/digest mismatch",
        )
        artifacts[relative] = artifact.raw
    return artifacts


def _authenticate_build(request: Mapping[str, Any], argv: tuple[str, ...]) -> AuthenticatedBuild:
    coverage_path = _canonical_existing_path(
        request["build_coverage_path"], "request.build_coverage_path", directory=False
    )
    evidence_path = _canonical_existing_path(
        request["build_evidence_path"], "request.build_evidence_path", directory=False
    )
    evidence_root = _canonical_existing_path(
        request["build_evidence_root"], "request.build_evidence_root", directory=True
    )
    _require(
        evidence_path == evidence_root / BUILD_EVIDENCE.EVIDENCE_ARCHIVE_PATH,
        "request.build_evidence_path: exact evidence-root location required",
    )
    coverage_file = COMPOSER._read_frozen(coverage_path, MAX_BUILD_COVERAGE_BYTES)
    evidence_file = COMPOSER._read_frozen(evidence_path, MAX_BUILD_EVIDENCE_BYTES)
    artifacts = _read_build_artifacts(evidence_file.raw, evidence_root)
    try:
        build = COVERAGE.validate_build_coverage_bytes(
            coverage_file.raw, api_definition_sha256=API_DEFINITION_SHA256
        )
        derived = BUILD_EVIDENCE.validate_build_evidence_bytes(
            evidence_file.raw,
            artifacts,
            coverage_file.raw,
            api_definition_sha256=API_DEFINITION_SHA256,
        )
    except Exception as error:
        raise SoftmodemLauncherError(str(error)) from error
    _require(
        build["evidence_origin_id"] == 1
        and build["verdict_id"] == 1
        and not build["build_identity"]["dirty"],
        "build evidence: measured clean complete coverage required",
    )
    _require(derived == build, "build evidence: derived build differs")
    primary_id = build["build_identity"]["primary_logical_elf_id"]
    primary = next(row for row in build["entries"] if row["logical_id"] == primary_id)
    binary_path = _canonical_existing_path(argv[0], "request.argv[0]", directory=False)
    binary = COMPOSER._read_frozen(binary_path, MAX_BINARY_BYTES)
    _require(
        binary.size == primary["byte_count"] and _sha(binary.raw) == primary["sha256"],
        "request.argv[0]: bytes differ from measured primary executable",
    )
    _require(os.access(binary_path, os.X_OK), "request.argv[0]: executable permission required")
    return AuthenticatedBuild(
        coverage_file.raw,
        evidence_file.raw,
        build,
        primary,
        binary_path,
    )


def _selection_values(
    value: Any, evidence_digest: str, authority_digest: str
) -> list[dict[str, str]]:
    _require(isinstance(value, list), "request.selection_values: array required")
    rows: list[dict[str, str]] = []
    for index, row in enumerate(value):
        _exact_keys(row, ("key", "value"), f"request.selection_values[{index}]")
        _require(
            row["key"] != "build_evidence_sha256",
            "request.selection_values: build_evidence_sha256 is launcher-owned",
        )
        _require(
            row["key"] != "trusted_release_authority_sha256",
            "request.selection_values: trusted_release_authority_sha256 is launcher-owned",
        )
        rows.append({"key": row["key"], "value": row["value"]})
    rows.append({"key": "build_evidence_sha256", "value": evidence_digest})
    rows.append(
        {
            "key": "trusted_release_authority_sha256",
            "value": authority_digest,
        }
    )
    rows.sort(key=lambda row: row["key"].encode("ascii"))
    return rows


def _derive_realloc_policy(
    build: Mapping[str, Any], config: Mapping[str, Any]
) -> int:
    configuration_values = {
        row["key"]: row["value"] for row in config["selection_values"]
    }
    role_name = dict(CONFIG.ROLE_CATALOG)[config["role_id"]]
    selected_rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(build["entries"]):
        selected = config["role_id"] in row["role_ids"] and CONFIG._module_selected(
            row["module_selection"],
            configuration_values=configuration_values,
            role_name=role_name,
            where=f"build_coverage.entries[{index}].module_selection",
        )
        if selected:
            selected_rows.append(row)
    _require(build["build_identity"]["primary_logical_elf_id"] in {
        row["logical_id"] for row in selected_rows
    }, "build coverage: primary executable is not selected for role/configuration")
    active_runtime = [
        row
        for row in selected_rows
        if row["soname"] == "liboai_memprof_active_runtime.so.1"
    ]
    _require(
        len(active_runtime) == 1 and active_runtime[0]["admission_state_id"] == 2,
        "build coverage: exactly one selected zero-import ACTIVE runtime required",
    )
    pairs: set[tuple[int, str]] = set()
    for row in selected_rows:
        admits_realloc = any(
            origin["api_id"] in (3, 5)
            and origin["origin_kind_id"] == 1
            and origin["classification_id"] == 1
            for origin in row["symbol_origins"]
        )
        if admits_realloc:
            pairs.add(
                (
                    row["realloc_zero_policy_id"],
                    row["realloc_zero_semantic_oracle_sha256"],
                )
            )
    _require(len(pairs) == 1, "build coverage: selected realloc policy must resolve exactly once")
    policy_id, oracle = next(iter(pairs))
    _require(
        type(policy_id) is int
        and policy_id in (1, 2)
        and oracle == VERIFIER.ACCEPTED_MEMBER_SHA256[3],
        "build coverage: exact selected realloc policy/oracle required",
    )
    return policy_id


def _publication_leaf(leaf: Any, where: str) -> str:
    _require(
        isinstance(leaf, str)
        and leaf not in ("", ".", "..")
        and "/" not in leaf
        and "\x00" not in leaf,
        f"{where}: single relative leaf required",
    )
    return leaf


def _publish_once_at(directory_fd: int, leaf: str, raw: bytes) -> None:
    _validate_directory_fd(directory_fd, "bootstrap publication")
    leaf = _publication_leaf(leaf, "bootstrap publication")
    _require(type(raw) is bytes, f"bootstrap publication bytes required: {leaf}")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            leaf,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o640,
            dir_fd=directory_fd,
        )
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            _require(count > 0, f"bootstrap publication short write: {leaf}")
            offset += count
        os.fsync(descriptor)
        status = os.fstat(descriptor)
        _require(
            stat.S_ISREG(status.st_mode)
            and status.st_nlink == 1
            and status.st_size == len(raw),
            f"bootstrap publication identity mismatch: {leaf}",
        )
        os.fsync(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _publish_once(directory: pathlib.Path, leaf: str, raw: bytes) -> pathlib.Path:
    directory_fd = _open_directory(
        directory,
        "bootstrap publication",
        require_private=False,
        require_empty=False,
    )
    try:
        _publish_once_at(directory_fd, leaf, raw)
    finally:
        os.close(directory_fd)
    return directory / leaf


def _create_streams_at(directory_fd: int) -> None:
    _validate_directory_fd(directory_fd, "archive publication")
    try:
        os.mkdir("streams", 0o750, dir_fd=directory_fd)
        os.fsync(directory_fd)
    except OSError as error:
        raise SoftmodemLauncherError("archive publication: streams directory creation failed") from error


def _prepare_from_authenticated(
    request: Mapping[str, Any],
    authenticated: AuthenticatedBuild,
    *,
    trusted_release_authority_sha256: str,
) -> PreparedLaunch:
    argv = _argv(request["argv"])
    _require(pathlib.Path(argv[0]).resolve(strict=True) == authenticated.binary_path, "request.argv[0]: authenticated path mismatch")
    archive = _canonical_existing_path(
        request["archive_directory"], "request.archive_directory", directory=True
    )
    bootstrap = _canonical_existing_path(
        request["bootstrap_directory"], "request.bootstrap_directory", directory=True
    )
    working = _canonical_existing_path(request["working_directory"], "request.working_directory", directory=True)
    generation = _uint(
        request["process_generation"], 64, "request.process_generation", minimum=1, maximum=MAX_PROCESS_GENERATION
    )
    max_threads = _uint(request["max_threads"], 32, "request.max_threads", minimum=1, maximum=MAX_THREADS)
    ring_records = _uint(request["ring_records"], 32, "request.ring_records", minimum=2, maximum=MAX_RING_RECORDS)
    _require(ring_records & (ring_records - 1) == 0, "request.ring_records: power of two required")
    mode_id = _uint(request["mode_id"], 16, "request.mode_id", minimum=1)
    _require(mode_id in (2, 3, 4), "request.mode_id: implemented runtime admits only 2, 3, or 4")
    role_id = _uint(request["role_id"], 16, "request.role_id", minimum=1)
    _require(role_id in (1, 2), "request.role_id: gNB or nrUE required")
    scope_kind = _uint(request["scope_kind"], 16, "request.scope_kind", minimum=1)
    _require(scope_kind == 1, "request.scope_kind: measurement interval required")
    flush_records = _uint(
        request["flush_records"], 32, "request.flush_records", minimum=1, maximum=MAX_FLUSH_RECORDS
    )
    flush_us = _uint(request["flush_us"], 64, "request.flush_us")
    _require(flush_us <= ((1 << 64) - 1) // 1000, "request.flush_us: nanosecond conversion overflow")
    seal_timeout = _uint(request["seal_timeout_ns"], 64, "request.seal_timeout_ns", minimum=1)
    table_entries = _uint(request["table_entries"], 64, "request.table_entries", minimum=1)
    table_probes = _uint(request["table_probes"], 32, "request.table_probes", minimum=1)
    _require(table_probes <= table_entries, "request.table_probes: cannot exceed table_entries")
    sample_threshold = _uint(
        request["sample_threshold"], 64, "request.sample_threshold"
    )
    sampled = mode_id == 3
    if sampled:
        _require(
            sample_threshold != 0,
            "request.sample_threshold: mode 3 requires nonzero q",
        )
        try:
            sample_seed_bytes = os.getrandom(8)
        except (AttributeError, OSError) as error:
            raise SoftmodemLauncherError(
                "sample seed: exact Linux getrandom acquisition failed"
            ) from error
        _require(
            type(sample_seed_bytes) is bytes and len(sample_seed_bytes) == 8,
            "sample seed: getrandom must return exactly eight bytes",
        )
        sample_seed_hex = sample_seed_bytes.hex()
        sample_seed = int.from_bytes(sample_seed_bytes, "big")
        sample_seed_provenance_id = 1
        sample_seed_status_id = 1
    else:
        _require(
            sample_threshold == 0,
            "request.sample_threshold: non-sampled mode requires zero",
        )
        sample_seed_hex = None
        sample_seed = 0
        sample_seed_provenance_id = 20
        sample_seed_status_id = 20
    run_uuid = _canonical_uuid(request["run_uuid"], "request.run_uuid")
    process_uuid = _canonical_uuid(request["process_uuid"], "request.process_uuid")
    _require(run_uuid != process_uuid, "request.process_uuid: must differ from run UUID")
    _require(role_id in authenticated.primary["role_ids"], "build coverage: primary role mismatch")
    primary_byte_count = _uint(
        authenticated.primary["byte_count"],
        64,
        "build coverage primary byte_count",
        minimum=1,
        maximum=MAX_BINARY_BYTES,
    )
    primary_sha256 = _sha256_hex(
        authenticated.primary["sha256"], "build coverage primary sha256"
    )

    evidence_digest = _sha(authenticated.evidence_raw)
    authority_digest = _sha256_hex(
        trusted_release_authority_sha256,
        "trusted-release authority",
    )
    config = CONFIG.make_effective_configuration(
        flush_records=flush_records,
        flush_us=flush_us,
        max_threads=max_threads,
        mode_id=mode_id,
        output_directory=str(archive),
        ring_records=ring_records,
        role_id=role_id,
        run_id=request["run_id"],
        sample_seed_hex=sample_seed_hex,
        sample_seed_provenance_id=sample_seed_provenance_id,
        sample_seed_status_id=sample_seed_status_id,
        sample_threshold=sample_threshold,
        scope_kind=scope_kind,
        selection_values=_selection_values(
            request["selection_values"], evidence_digest, authority_digest
        ),
        table_entries=table_entries,
        table_probes=table_probes,
    )
    config_raw = CONFIG.serialize_effective_configuration(config)
    policy_id = _derive_realloc_policy(authenticated.build, config)
    _members, bundle_raw = VERIFIER._accepted_static_members()
    architecture_id = authenticated.build["architecture_id"]
    clock_kind = {1: 1, 2: 2}.get(architecture_id)
    _require(clock_kind is not None, "build coverage: unsupported architecture")
    source_commit = authenticated.build["build_identity"]["source_commit"]
    source_raw = bytes.fromhex(source_commit)
    _require(len(source_raw) in (20, 32), "build coverage: unsupported source object width")
    opening = WIRE.OpeningHeader(
        page_size_bytes=os.sysconf("SC_PAGE_SIZE"),
        scope_kind=scope_kind,
        role_kind=role_id,
        clock_kind=clock_kind,
        calibration_kind=1,
        process_generation=generation,
        counter_frequency_numerator=1,
        counter_frequency_denominator=1,
        calibration_error_bound_ns=1_000_000,
        calibration_span_ns=0,
        start_counter=1,
        start_monotonic_raw_ns=1,
        start_realtime_unix_ns=1,
        pid=1,
        configured_thread_capacity=max_threads,
        run_uuid=run_uuid.bytes,
        process_uuid=process_uuid.bytes,
        source_object_kind=1,
        source_object_algorithm=1 if len(source_raw) == 20 else 2,
        source_object_length=len(source_raw),
        source_object_value=source_raw + bytes(32 - len(source_raw)),
        primary_binary_sha256=bytes.fromhex(primary_sha256),
        schema_bundle_definition_sha256=hashlib.sha256(bundle_raw).digest(),
        api_catalog_definition_sha256=bytes.fromhex(VERIFIER.ACCEPTED_MEMBER_SHA256[4]),
        callsite_catalog_definition_sha256=bytes.fromhex(VERIFIER.ACCEPTED_MEMBER_SHA256[5]),
        configuration_instance_sha256=hashlib.sha256(config_raw).digest(),
        primary_build_id_sha256=hashlib.sha256(bytes.fromhex(authenticated.primary["build_id"])).digest(),
    )
    opening_raw = WIRE.encode_opening_header(opening)

    archive_fd = -1
    bootstrap_fd = -1
    working_fd = -1
    try:
        archive_fd = _open_directory(
            archive,
            "request.archive_directory",
            require_private=True,
            require_empty=True,
        )
        bootstrap_fd = _open_directory(
            bootstrap,
            "request.bootstrap_directory",
            require_private=True,
            require_empty=True,
        )
        working_fd = _open_directory(
            working,
            "request.working_directory",
            require_private=False,
            require_empty=False,
        )
        _create_streams_at(archive_fd)
        _publish_once_at(bootstrap_fd, CONFIGURATION_LEAF, config_raw)
        _publish_once_at(bootstrap_fd, OPENING_LEAF, opening_raw)
        configuration_path = bootstrap / CONFIGURATION_LEAF
        opening_path = bootstrap / OPENING_LEAF
        environment = (
            ("OAI_MEMPROF_SESSION_ENABLE", "1"),
            ("OAI_MEMPROF_SESSION_ARCHIVE_FD", str(archive_fd)),
            ("OAI_MEMPROF_SESSION_BOOTSTRAP_FD", str(bootstrap_fd)),
            ("OAI_MEMPROF_SESSION_PROCESS_GENERATION", str(generation)),
            ("OAI_MEMPROF_SESSION_MAX_THREADS", str(max_threads)),
            ("OAI_MEMPROF_SESSION_RING_RECORDS", str(ring_records)),
            ("OAI_MEMPROF_SESSION_MODE_ID", str(mode_id)),
            ("OAI_MEMPROF_SESSION_TABLE_ENTRIES", str(table_entries if sampled else 0)),
            ("OAI_MEMPROF_SESSION_SAMPLE_SEED", str(sample_seed)),
            ("OAI_MEMPROF_SESSION_SAMPLE_THRESHOLD", str(sample_threshold)),
            ("OAI_MEMPROF_SESSION_TABLE_PROBES", str(table_probes if sampled else 0)),
            ("OAI_MEMPROF_SESSION_REALLOC_ZERO_POLICY_ID", str(policy_id)),
            ("OAI_MEMPROF_SESSION_FLUSH_RECORDS", str(flush_records)),
            ("OAI_MEMPROF_SESSION_FLUSH_INTERVAL_NS", str(flush_us * 1000)),
            ("OAI_MEMPROF_SESSION_SEAL_TIMEOUT_NS", str(seal_timeout)),
        )
        _require(
            tuple(name for name, _value in environment) == SESSION_ENVIRONMENT_NAMES,
            "internal environment order mismatch",
        )
        return PreparedLaunch(
            archive_directory=archive,
            bootstrap_directory=bootstrap,
            configuration_path=configuration_path,
            opening_path=opening_path,
            binary_path=authenticated.binary_path,
            working_directory=working,
            archive_directory_fd=archive_fd,
            bootstrap_directory_fd=bootstrap_fd,
            working_directory_fd=working_fd,
            binary_byte_count=primary_byte_count,
            binary_sha256=primary_sha256,
            argv=argv,
            session_environment=environment,
            configuration_sha256=_sha(config_raw),
            opening_sha256=_sha(opening_raw),
            build_coverage_sha256=_sha(authenticated.build_raw),
            build_evidence_sha256=evidence_digest,
            trusted_release_authority_sha256=authority_digest,
            realloc_zero_policy_id=policy_id,
        )
    except BaseException:
        for descriptor in (working_fd, bootstrap_fd, archive_fd):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        raise


def _authenticate_trusted_release(
    authority_path: pathlib.Path,
    source_root: pathlib.Path,
    expected_sha256: str,
    authenticated: AuthenticatedBuild,
) -> str:
    """Validate controller-pinned release bytes before bootstrap publication.

    The trust input is the separately supplied digest, not a digest derived by
    this process. Frozen fd-root reads detect ordinary replacement/symlink
    attacks, but cannot make a malicious modified running process or coordinated
    same-UID rewrite impossible.
    """

    try:
        authority, sources = COMPOSER._read_trusted_release_inputs(
            pathlib.Path(authority_path),
            pathlib.Path(source_root),
        )
        VERIFIER.validate_trusted_release_authority(
            authority.raw,
            expected_sha256,
            sources,
            build=authenticated.build,
        )
    except (
        COMPOSER.ArchiveComposerError,
        OSError,
        VERIFIER.ArchiveVerificationError,
    ) as error:
        raise SoftmodemLauncherError(
            f"trusted-release authority authentication failed: {error}"
        ) from error
    return _sha256_hex(expected_sha256, "trusted-release authority")


def prepare_launch(
    request_raw: bytes,
    *,
    trusted_release_authority_path: pathlib.Path,
    trusted_release_source_root: pathlib.Path,
    trusted_release_authority_sha256: str,
) -> PreparedLaunch:
    request = _request(request_raw)
    argv = _argv(request["argv"])
    authenticated = _authenticate_build(request, argv)
    authority_digest = _authenticate_trusted_release(
        trusted_release_authority_path,
        trusted_release_source_root,
        trusted_release_authority_sha256,
        authenticated,
    )
    return _prepare_from_authenticated(
        request,
        authenticated,
        trusted_release_authority_sha256=authority_digest,
    )


def execute(prepared: PreparedLaunch) -> None:
    _require(type(prepared) is PreparedLaunch, "prepared launch: exact result required")
    try:
        _require(EXECVE_SUPPORTS_FD, "prepared launch: descriptor exec unsupported")
        _validate_directory_fd(prepared.archive_directory_fd, "prepared archive root")
        _validate_directory_fd(prepared.bootstrap_directory_fd, "prepared bootstrap root")
        _validate_directory_fd(prepared.working_directory_fd, "prepared working directory")
        _require(
            tuple(name for name, _value in prepared.session_environment)
            == SESSION_ENVIRONMENT_NAMES,
            "prepared launch: exact owned environment required",
        )
        environment = dict(prepared.session_environment)
        _require(
            len(environment) == len(SESSION_ENVIRONMENT_NAMES),
            "prepared launch: duplicate environment control",
        )
    except BaseException:
        _close_prepared_fds(prepared)
        raise
    executable_fd = -1
    original_working_fd = -1
    descriptor_hygiene: _ExecDescriptorHygiene | None = None
    cleanup_error: BaseException | None = None
    try:
        executable_fd = _open_authenticated_binary(
            prepared.binary_path,
            prepared.binary_byte_count,
            prepared.binary_sha256,
        )
        _require(
            not os.get_inheritable(executable_fd),
            "prepared launch: executable descriptor must close on exec",
        )
        _require(
            not os.get_inheritable(prepared.working_directory_fd),
            "prepared launch: working descriptor must close on exec",
        )
        original_working_fd = os.open(
            ".", os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        _require(
            not os.get_inheritable(original_working_fd),
            "prepared launch: saved working descriptor must close on exec",
        )
        # The archive and bootstrap roots are the only descriptors this launcher
        # deliberately hands to the child. The C session opens its exact leaves
        # beneath them, then closes both inherited roots.
        _require_single_threaded_exec()
        descriptor_hygiene = _prepare_exec_descriptor_hygiene(
            prepared.archive_directory_fd, prepared.bootstrap_directory_fd
        )
        os.fchdir(prepared.working_directory_fd)
        os.execve(executable_fd, prepared.argv, environment)
        raise AssertionError("os.execve returned")
    finally:
        if descriptor_hygiene is not None:
            try:
                _restore_exec_descriptor_hygiene(descriptor_hygiene)
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
        if original_working_fd >= 0:
            try:
                os.fchdir(original_working_fd)
            except OSError as error:
                if cleanup_error is None:
                    cleanup_error = error
            try:
                os.close(original_working_fd)
            except OSError as error:
                if cleanup_error is None:
                    cleanup_error = error
        if executable_fd >= 0:
            try:
                os.close(executable_fd)
            except OSError as error:
                if cleanup_error is None:
                    cleanup_error = error
        try:
            _close_prepared_fds(prepared)
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        if cleanup_error is not None:
            raise SoftmodemLauncherError(
                "prepared launch: failed cleanup after descriptor exec"
            ) from cleanup_error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=pathlib.Path)
    parser.add_argument("--trusted-release-authority", required=True, type=pathlib.Path)
    parser.add_argument("--trusted-release-source-root", required=True, type=pathlib.Path)
    parser.add_argument("--trusted-release-authority-sha256", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    arguments = parser.parse_args(argv)
    request_file = COMPOSER._read_frozen(
        arguments.request.resolve(strict=True), MAX_REQUEST_BYTES
    )
    prepared = prepare_launch(
        request_file.raw,
        trusted_release_authority_path=arguments.trusted_release_authority,
        trusted_release_source_root=arguments.trusted_release_source_root,
        trusted_release_authority_sha256=arguments.trusted_release_authority_sha256,
    )
    try:
        print(
            "softmodem launch prepared "
            f"configuration_sha256={prepared.configuration_sha256} "
            f"opening_sha256={prepared.opening_sha256} "
            f"build_coverage_sha256={prepared.build_coverage_sha256} "
            f"build_evidence_sha256={prepared.build_evidence_sha256} "
            f"trusted_release_authority_sha256={prepared.trusted_release_authority_sha256}"
        )
        if arguments.prepare_only:
            return 0
        execute(prepared)
        return 127
    finally:
        close_prepared_launch(prepared)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SoftmodemLauncherError, OSError, ValueError) as error:
        print(f"softmodem launcher: {error}", file=sys.stderr)
        raise SystemExit(2) from error
