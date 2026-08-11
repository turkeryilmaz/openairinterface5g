#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Crash-resilient evidence archive primitives for the R0 controller."""

from __future__ import annotations

import ctypes
import contextvars
import dataclasses
import enum
import errno
import fcntl
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from collections.abc import Callable, Mapping
from typing import Any


class ArchiveErrorCode(str, enum.Enum):
    INVALID_ARGUMENT = "invalid_argument"
    INVALID_JSON = "invalid_json"
    INVALID_RUN_ID = "invalid_run_id"
    INVALID_ABSOLUTE_PATH = "invalid_absolute_path"
    INVALID_RELATIVE_PATH = "invalid_relative_path"
    PATH_EXISTS = "path_exists"
    PATH_MISSING = "path_missing"
    PATH_TYPE = "path_type"
    SYMLINK_DENIED = "symlink_denied"
    OPEN_FAILED = "open_failed"
    READ_FAILED = "read_failed"
    WRITE_FAILED = "write_failed"
    FSYNC_FAILED = "fsync_failed"
    FILE_TOO_LARGE = "file_too_large"
    FILE_CHANGED = "file_changed"
    DIRECTORY_TOO_LARGE = "directory_too_large"
    TOTAL_TOO_LARGE = "total_too_large"
    HASH_MISMATCH = "hash_mismatch"
    MANIFEST_INVALID = "manifest_invalid"
    MANIFEST_MISSING = "manifest_missing"
    JOURNAL_INVALID = "journal_invalid"
    JOURNAL_SEQUENCE = "journal_sequence"
    JOURNAL_HASH = "journal_hash"
    STATE_INVALID = "state_invalid"
    CROSS_DEVICE = "cross_device"
    PUBLISH_EXISTS = "publish_exists"
    PUBLISH_FAILED = "publish_failed"
    RECOVERY_UNVERIFIED = "recovery_unverified"


class PublicationPhase(str, enum.Enum):
    """Last publication protocol step known to have completed."""

    PRE_RENAME = "pre_rename"
    RENAMED_UNSYNCED = "renamed_unsynced"
    FINAL_PARENT_SYNCED = "final_parent_synced"
    PARENTS_SYNCED = "parents_synced"
    VERIFIED = "verified"


class ArchiveError(RuntimeError):
    def __init__(
        self,
        code: ArchiveErrorCode,
        operation: str,
        detail: str,
        path: Path | None = None,
        publication_phase: PublicationPhase | None = None,
    ) -> None:
        super().__init__(f"{operation}: {detail}")
        self.code = code
        self.operation = operation
        self.detail = detail
        self.path = path
        self.publication_phase = publication_phase


@dataclasses.dataclass(frozen=True)
class FileImage:
    data: bytes
    device: int
    inode: int
    size: int
    mode: int
    mtime_ns: int
    ctime_ns: int
    sha256: str


@dataclasses.dataclass(frozen=True)
class FileRecord:
    path: str
    size: int
    mode: int
    sha256: str
    device: int
    inode: int
    mtime_ns: int
    line_count: int


@dataclasses.dataclass(frozen=True)
class JournalCursor:
    next_sequence: int
    previous_sha256: str


@dataclasses.dataclass(frozen=True)
class JournalRead:
    entries: tuple[dict[str, Any], ...]
    cursor: JournalCursor


@dataclasses.dataclass(frozen=True)
class ManifestSummary:
    manifest_path: str
    regular_file_count: int
    directory_count: int
    total_regular_file_bytes: int
    manifest_sha256: str


@dataclasses.dataclass(frozen=True)
class PublicationResult:
    summary: ManifestSummary
    phase: PublicationPhase


@dataclasses.dataclass(frozen=True)
class RecoveryInspection:
    active_dir: str
    state: object | None
    journal: JournalRead | None
    manifest_present: bool
    manifest_verified: bool
    reason: str | None


_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z", re.ASCII)
_ZERO_HASH = "0" * 64
_STATE_NAME = "state.json"
_JOURNAL_NAME = "journal.jsonl"
_JSON_MAX_DEPTH = 64
_SIGNED_64_MAX = (1 << 63) - 1
_READ_CHUNK = 64 << 10
_DIRECTORY_PROGRESS_BATCH = 64
_ARCHIVE_API_MAX_DEPTH = 64
_CALLBACK_ORIGIN_MAX_NOTES = 32
_CALLBACK_ORIGIN_MAX_NOTE_CHARS = 512
_JOURNAL_SCHEMA = "oai-memprof-r0-journal-v1"
_MANIFEST_SCHEMA = "oai-memprof-r0-manifest-v2"

ProgressCallback = Callable[[], None]


class _CallbackOrigin(BaseException):
    """Internal callback-origin carrier; never exposed by a public API."""

    __slots__ = (
        "original",
        "original_cause",
        "original_context",
        "original_suppress_context",
        "original_traceback",
        "factual_publication_phase",
        "note_count",
        "dropped_note_count",
    )

    def __init__(self, original: BaseException) -> None:
        super().__init__("progress callback raised")
        self.original = original
        self.original_cause = original.__cause__
        self.original_context = original.__context__
        self.original_suppress_context = original.__suppress_context__
        self.original_traceback = original.__traceback__
        self.factual_publication_phase: PublicationPhase | None = None
        self.note_count = 0
        self.dropped_note_count = 0

    def add_note(self, note: str) -> None:
        if type(note) is not str:
            raise TypeError("note must be a string")
        if self.note_count >= _CALLBACK_ORIGIN_MAX_NOTES:
            self.dropped_note_count += 1
            return
        if len(note) > _CALLBACK_ORIGIN_MAX_NOTE_CHARS:
            note = note[: _CALLBACK_ORIGIN_MAX_NOTE_CHARS - 3] + "..."
        BaseException.add_note(self, note)
        self.note_count += 1


_ARCHIVE_API_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "oai_memprof_archive_api_depth",
    default=0,
)


def _transfer_callback_origin(failure: _CallbackOrigin) -> BaseException:
    original = failure.original
    for note in getattr(failure, "__notes__", ()):
        BaseException.add_note(original, note)
    if failure.dropped_note_count:
        BaseException.add_note(
            original,
            "callback cleanup notes dropped by bounded carrier: "
            f"{failure.dropped_note_count}",
        )
    if failure.factual_publication_phase is not None:
        if isinstance(original, ArchiveError):
            original.publication_phase = failure.factual_publication_phase
        else:
            BaseException.add_note(
                original,
                "publication_phase="
                f"{failure.factual_publication_phase.value}",
            )
    original.__cause__ = failure.original_cause
    original.__context__ = failure.original_context
    original.__suppress_context__ = failure.original_suppress_context
    original.__traceback__ = failure.original_traceback
    return original


def _archive_public_api(function):
    @functools.wraps(function)
    def invoke(*arguments, **keywords):
        depth = _ARCHIVE_API_DEPTH.get()
        if depth >= _ARCHIVE_API_MAX_DEPTH:
            raise ArchiveError(
                ArchiveErrorCode.INVALID_ARGUMENT,
                "archive-api",
                "nested archive API depth exceeds bound",
            )
        token = _ARCHIVE_API_DEPTH.set(depth + 1)
        failure: _CallbackOrigin | None = None
        try:
            try:
                return function(*arguments, **keywords)
            except _CallbackOrigin as error:
                failure = error
        finally:
            _ARCHIVE_API_DEPTH.reset(token)
        if failure is None:
            raise AssertionError("callback origin control flow is invalid")
        raise _transfer_callback_origin(failure)

    invoke._oai_memprof_archive_public_api = True
    return invoke


def _call_archive_internal(function, /, *arguments, **keywords):
    """Invoke a composed API implementation without a public boundary."""

    if getattr(function, "_oai_memprof_archive_public_api", False) is not True:
        raise AssertionError("internal archive call target is not a public API")
    implementation = getattr(function, "__wrapped__", None)
    if not callable(implementation):
        raise AssertionError("internal archive call target has no implementation")
    depth = _ARCHIVE_API_DEPTH.get()
    if depth <= 0:
        raise AssertionError("internal archive call is outside a public API")
    if depth >= _ARCHIVE_API_MAX_DEPTH:
        raise ArchiveError(
            ArchiveErrorCode.INVALID_ARGUMENT,
            "archive-api",
            "nested archive API depth exceeds bound",
        )
    token = _ARCHIVE_API_DEPTH.set(depth + 1)
    try:
        return implementation(*arguments, **keywords)
    finally:
        _ARCHIVE_API_DEPTH.reset(token)


def _preserve_cleanup_failure(
    primary: BaseException,
    operation: str,
    cleanup: Callable[[], None],
) -> None:
    try:
        cleanup()
    except BaseException as cleanup_error:
        primary.add_note(
            f"{operation} cleanup raised {type(cleanup_error).__name__}: "
            f"{cleanup_error}"
        )


def _close_descriptor_preserving_primary(
    descriptor: int,
    primary: BaseException | None,
    operation: str,
    failure_code: ArchiveErrorCode,
    path: Path,
) -> None:
    try:
        os.close(descriptor)
    except BaseException as close_error:
        if primary is not None:
            primary.add_note(
                f"{operation} close raised "
                f"{type(close_error).__name__}: {close_error}"
            )
        elif isinstance(close_error, OSError):
            raise ArchiveError(
                failure_code,
                operation,
                f"descriptor close failed: {close_error}",
                path,
            ) from close_error
        else:
            raise


class _ScandirScope:
    """Close a scandir iterator without replacing an active primary."""

    def __init__(self, path: Path) -> None:
        self._iterator = os.scandir(path)

    def __enter__(self):
        return self._iterator.__enter__()

    def __exit__(self, exception_type, exception, traceback) -> bool:
        def cleanup() -> None:
            self._iterator.__exit__(None, None, None)

        if exception is None:
            cleanup()
        else:
            _preserve_cleanup_failure(
                exception,
                "scan-archive scandir close",
                cleanup,
            )
        return False


def _raise(code: ArchiveErrorCode, operation: str, detail: str, path: Path | None = None) -> None:
    raise ArchiveError(code, operation, detail, path)


def _invoke_progress(progress_callback: ProgressCallback | None) -> None:
    if progress_callback is None:
        return
    if not callable(progress_callback):
        _raise(
            ArchiveErrorCode.INVALID_ARGUMENT,
            "progress-callback",
            "progress_callback must be callable or None",
        )
    # Caller code may re-enter a public API or retain a copied context.  Neither
    # case is private in-module composition, so it must observe a clean root.
    token = _ARCHIVE_API_DEPTH.set(0)
    try:
        try:
            progress_callback()
        except _CallbackOrigin:
            raise
        except BaseException as error:
            raise _CallbackOrigin(error) from None
    finally:
        _ARCHIVE_API_DEPTH.reset(token)


def _normalize_json(value: object, depth: int = 0, active: set[int] | None = None) -> object:
    if depth > _JSON_MAX_DEPTH:
        _raise(ArchiveErrorCode.INVALID_JSON, "canonical-json", "maximum nesting depth exceeded")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value < -_SIGNED_64_MAX - 1 or value > _SIGNED_64_MAX:
            _raise(ArchiveErrorCode.INVALID_JSON, "canonical-json", "integer is outside signed 64-bit domain")
        return value
    if isinstance(value, str):
        try:
            value.encode("utf-8", "strict")
        except UnicodeEncodeError as error:
            raise ArchiveError(
                ArchiveErrorCode.INVALID_JSON,
                "canonical-json",
                "string contains a lone surrogate",
            ) from error
        return value
    if isinstance(value, float) or isinstance(value, (bytes, bytearray, memoryview)):
        _raise(ArchiveErrorCode.INVALID_JSON, "canonical-json", f"unsupported scalar type {type(value).__name__}")
    if active is None:
        active = set()
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            _raise(ArchiveErrorCode.INVALID_JSON, "canonical-json", "cycle detected")
        active.add(identity)
        try:
            return [_normalize_json(item, depth + 1, active) for item in value]
        finally:
            active.remove(identity)
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            _raise(ArchiveErrorCode.INVALID_JSON, "canonical-json", "cycle detected")
        active.add(identity)
        try:
            normalized: dict[str, object] = {}
            for key, item in value.items():
                if type(key) is not str:
                    _raise(ArchiveErrorCode.INVALID_JSON, "canonical-json", "object keys must be exact strings")
                if key in normalized:
                    _raise(ArchiveErrorCode.INVALID_JSON, "canonical-json", "duplicate object key")
                _normalize_json(key, depth + 1, active)
                normalized[key] = _normalize_json(item, depth + 1, active)
            return normalized
        finally:
            active.remove(identity)
    _raise(ArchiveErrorCode.INVALID_JSON, "canonical-json", f"unsupported type {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    normalized = _normalize_json(value)
    try:
        return json.dumps(
            normalized,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ArchiveError(ArchiveErrorCode.INVALID_JSON, "canonical-json", str(error)) from error


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_json(data: bytes, code: ArchiveErrorCode, operation: str) -> object:
    def reject_number(text: str) -> object:
        raise ValueError(f"non-integer JSON number {text}")

    try:
        value = json.loads(
            data.decode("ascii"),
            object_pairs_hook=_object_pairs,
            parse_float=reject_number,
            parse_constant=reject_number,
        )
        normalized = _normalize_json(value)
    except (UnicodeError, ValueError, TypeError, ArchiveError) as error:
        raise ArchiveError(code, operation, f"invalid JSON: {error}") from error
    return normalized


def validate_run_id(run_id: str) -> str:
    if type(run_id) is not str or _RUN_ID_RE.fullmatch(run_id) is None or run_id in (".", ".."):
        _raise(ArchiveErrorCode.INVALID_RUN_ID, "validate-run-id", "run ID is not safe bounded ASCII")
    return run_id


def validate_absolute_path(path: Path | str, field: str) -> Path:
    if not isinstance(path, (str, Path)) or not isinstance(field, str) or not field:
        _raise(ArchiveErrorCode.INVALID_ARGUMENT, "validate-path", "invalid path or field type")
    text = str(path)
    if (
        not text
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in text)
        or text.startswith("//")
        or not os.path.isabs(text)
        or os.path.normpath(text) != text
    ):
        _raise(ArchiveErrorCode.INVALID_ABSOLUTE_PATH, "validate-path", f"{field} is not a normalized absolute path")
    return Path(text)


def validate_relative_path(relative: str) -> str:
    if (
        type(relative) is not str
        or not relative
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in relative)
        or relative.startswith("/")
        or "\\" in relative
    ):
        _raise(ArchiveErrorCode.INVALID_RELATIVE_PATH, "validate-relative-path", "relative path is invalid")
    try:
        relative.encode("ascii", "strict")
    except UnicodeEncodeError as error:
        raise ArchiveError(
            ArchiveErrorCode.INVALID_RELATIVE_PATH,
            "validate-relative-path",
            "relative path is not ASCII",
        ) from error
    parts = relative.split("/")
    if any(not part or part in (".", "..") for part in parts) or os.path.normpath(relative) != relative:
        _raise(
            ArchiveErrorCode.INVALID_RELATIVE_PATH,
            "validate-relative-path",
            "relative path is ambiguous or traversing",
        )
    return relative


def _directory_lstat(path: Path, operation: str) -> os.stat_result:
    try:
        information = os.lstat(path)
    except FileNotFoundError as error:
        raise ArchiveError(ArchiveErrorCode.PATH_MISSING, operation, str(error), path) from error
    except OSError as error:
        raise ArchiveError(ArchiveErrorCode.OPEN_FAILED, operation, str(error), path) from error
    if stat.S_ISLNK(information.st_mode):
        _raise(ArchiveErrorCode.SYMLINK_DENIED, operation, "directory is a symlink", path)
    if not stat.S_ISDIR(information.st_mode):
        _raise(ArchiveErrorCode.PATH_TYPE, operation, "path is not a directory", path)
    return information


def safe_join(root: Path | str, relative: str) -> Path:
    root_path = validate_absolute_path(root, "root")
    relative = validate_relative_path(relative)
    _directory_lstat(root_path, "safe-join")
    current = root_path
    parts = relative.split("/")
    for index, part in enumerate(parts):
        current = current / part
        try:
            information = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as error:
            raise ArchiveError(ArchiveErrorCode.OPEN_FAILED, "safe-join", str(error), current) from error
        if stat.S_ISLNK(information.st_mode):
            _raise(ArchiveErrorCode.SYMLINK_DENIED, "safe-join", "existing path component is a symlink", current)
        if index < len(parts) - 1 and not stat.S_ISDIR(information.st_mode):
            _raise(ArchiveErrorCode.PATH_TYPE, "safe-join", "intermediate component is not a directory", current)
    return root_path.joinpath(*parts)


def _fsync_directory(
    path: Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise ArchiveError(
            ArchiveErrorCode.FSYNC_FAILED,
            "fsync-directory",
            str(error),
            path,
        ) from error
    try:
        _invoke_progress(progress_callback)
        try:
            os.fsync(descriptor)
        except OSError as error:
            raise ArchiveError(
                ArchiveErrorCode.FSYNC_FAILED,
                "fsync-directory",
                str(error),
                path,
            ) from error
        _invoke_progress(progress_callback)
    except BaseException as error:
        _preserve_cleanup_failure(
            error,
            "fsync-directory close",
            lambda: os.close(descriptor),
        )
        raise
    try:
        os.close(descriptor)
    except OSError as error:
        raise ArchiveError(
            ArchiveErrorCode.FSYNC_FAILED,
            "fsync-directory",
            f"directory descriptor close failed: {error}",
            path,
        ) from error


def _fsync_directory_with_progress(
    path: Path,
    progress_callback: ProgressCallback | None,
) -> None:
    if progress_callback is None:
        _fsync_directory(path)
        return
    _fsync_directory(path, progress_callback=progress_callback)


@_archive_public_api
def create_directory_exclusive(
    path: Path | str,
    mode: int = 0o700,
    *,
    progress_callback: ProgressCallback | None = None,
) -> None:
    _invoke_progress(progress_callback)
    path = validate_absolute_path(path, "directory")
    _directory_lstat(path.parent, "create-directory-parent")
    try:
        os.mkdir(path, mode)
    except FileExistsError as error:
        raise ArchiveError(ArchiveErrorCode.PATH_EXISTS, "create-directory", str(error), path) from error
    except OSError as error:
        raise ArchiveError(ArchiveErrorCode.WRITE_FAILED, "create-directory", str(error), path) from error
    _directory_lstat(path, "create-directory")
    _fsync_directory_with_progress(path.parent, progress_callback)


def _map_open_error(error: OSError, operation: str, path: Path) -> ArchiveError:
    if error.errno == errno.ELOOP:
        return ArchiveError(ArchiveErrorCode.SYMLINK_DENIED, operation, str(error), path)
    if error.errno == errno.ENOENT:
        return ArchiveError(ArchiveErrorCode.PATH_MISSING, operation, str(error), path)
    if error.errno == errno.EEXIST:
        return ArchiveError(ArchiveErrorCode.PATH_EXISTS, operation, str(error), path)
    return ArchiveError(ArchiveErrorCode.OPEN_FAILED, operation, str(error), path)


@_archive_public_api
def read_regular_file_bounded(
    path: Path | str,
    max_bytes: int,
    *,
    progress_callback: ProgressCallback | None = None,
) -> FileImage:
    _invoke_progress(progress_callback)
    path = validate_absolute_path(path, "file")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 0:
        _raise(ArchiveErrorCode.INVALID_ARGUMENT, "read-file", "max_bytes must be nonnegative")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise _map_open_error(error, "read-file", path) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _raise(ArchiveErrorCode.PATH_TYPE, "read-file", "input is not a regular file", path)
        if before.st_nlink != 1:
            _raise(ArchiveErrorCode.PATH_TYPE, "read-file", "hardlinked input is denied", path)
        if before.st_size > max_bytes:
            _raise(ArchiveErrorCode.FILE_TOO_LARGE, "read-file", f"input exceeds {max_bytes} bytes", path)
        data = bytearray()
        while len(data) <= max_bytes:
            try:
                chunk = os.read(descriptor, min(_READ_CHUNK, max_bytes + 1 - len(data)))
            except BlockingIOError as error:
                raise ArchiveError(
                    ArchiveErrorCode.READ_FAILED,
                    "read-file",
                    "regular-file read would block",
                    path,
                ) from error
            if not chunk:
                break
            data.extend(chunk)
            _invoke_progress(progress_callback)
        after = os.fstat(descriptor)
        if len(data) > max_bytes:
            _raise(ArchiveErrorCode.FILE_TOO_LARGE, "read-file", f"input grew beyond {max_bytes} bytes", path)
        before_version = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mode,
            before.st_nlink,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_version = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mode,
            after.st_nlink,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_version != after_version or len(data) != after.st_size:
            _raise(ArchiveErrorCode.FILE_CHANGED, "read-file", "input changed while being read", path)
        try:
            pathname_after = os.lstat(path)
        except FileNotFoundError as error:
            raise ArchiveError(
                ArchiveErrorCode.FILE_CHANGED,
                "read-file",
                "pathname disappeared after read",
                path,
            ) from error
        if stat.S_ISLNK(pathname_after.st_mode):
            _raise(
                ArchiveErrorCode.SYMLINK_DENIED,
                "read-file",
                "pathname became a symlink during read",
                path,
            )
        if (pathname_after.st_dev, pathname_after.st_ino) != (after.st_dev, after.st_ino):
            _raise(
                ArchiveErrorCode.FILE_CHANGED,
                "read-file",
                "pathname no longer identifies the opened file",
                path,
            )
        raw = bytes(data)
        return FileImage(
            raw,
            after.st_dev,
            after.st_ino,
            len(raw),
            after.st_mode,
            after.st_mtime_ns,
            after.st_ctime_ns,
            hashlib.sha256(raw).hexdigest(),
        )
    except ArchiveError:
        raise
    except OSError as error:
        raise ArchiveError(ArchiveErrorCode.READ_FAILED, "read-file", str(error), path) from error
    finally:
        active_error = sys.exc_info()[1]
        try:
            os.close(descriptor)
        except BaseException as close_error:
            if active_error is not None:
                active_error.add_note(
                    "read-file close raised "
                    f"{type(close_error).__name__}: {close_error}"
                )
            elif isinstance(close_error, OSError):
                raise ArchiveError(
                    ArchiveErrorCode.READ_FAILED,
                    "read-file",
                    f"descriptor close failed: {close_error}",
                    path,
                ) from close_error
            else:
                raise


def _open_exclusive(path: Path, mode: int) -> int:
    _directory_lstat(path.parent, "create-file-parent")
    try:
        return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, mode)
    except OSError as error:
        raise _map_open_error(error, "create-file", path) from error


def _write_all(
    descriptor: int,
    data: bytes,
    path: Path,
    progress_callback: ProgressCallback | None = None,
) -> None:
    offset = 0
    try:
        while offset < len(data):
            written = os.write(
                descriptor,
                data[offset : offset + _READ_CHUNK],
            )
            if written <= 0:
                raise OSError("write made no progress")
            offset += written
            _invoke_progress(progress_callback)
    except OSError as error:
        raise ArchiveError(ArchiveErrorCode.WRITE_FAILED, "write-file", str(error), path) from error


def _line_count(data: bytes) -> int:
    return data.count(b"\n") + int(bool(data) and not data.endswith(b"\n"))


def _unlink_created(path: Path, identity: tuple[int, int]) -> None:
    try:
        current = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISREG(current.st_mode) and (current.st_dev, current.st_ino) == identity:
        os.unlink(path)
        _fsync_directory(path.parent)


def _unlink_created_version(
    path: Path,
    version: tuple[int, int, int, int, int, int, int],
) -> None:
    try:
        current = os.lstat(path)
    except FileNotFoundError:
        return
    current_version = (
        current.st_dev,
        current.st_ino,
        current.st_mode,
        current.st_nlink,
        current.st_size,
        current.st_mtime_ns,
        current.st_ctime_ns,
    )
    if stat.S_ISREG(current.st_mode) and current_version == version:
        os.unlink(path)
        _fsync_directory(path.parent)


@_archive_public_api
def write_bytes_exclusive(
    path: Path | str,
    data: bytes,
    *,
    max_bytes: int,
    mode: int = 0o600,
    progress_callback: ProgressCallback | None = None,
) -> FileRecord:
    _invoke_progress(progress_callback)
    path = validate_absolute_path(path, "destination")
    if not isinstance(data, bytes) or not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 0:
        _raise(ArchiveErrorCode.INVALID_ARGUMENT, "write-file", "invalid data or bound", path)
    if len(data) > max_bytes:
        _raise(ArchiveErrorCode.FILE_TOO_LARGE, "write-file", f"output exceeds {max_bytes} bytes", path)
    descriptor = _open_exclusive(path, mode)
    created = os.fstat(descriptor)
    identity = (created.st_dev, created.st_ino)
    try:
        if not stat.S_ISREG(created.st_mode) or created.st_nlink != 1:
            _raise(
                ArchiveErrorCode.PATH_TYPE,
                "write-file",
                "new output is not a single-linked regular file",
                path,
            )
        _write_all(
            descriptor,
            data,
            path,
            progress_callback,
        )
        try:
            _invoke_progress(progress_callback)
            os.fsync(descriptor)
            _invoke_progress(progress_callback)
        except OSError as error:
            raise ArchiveError(
                ArchiveErrorCode.FSYNC_FAILED,
                "fsync-file",
                str(error),
                path,
            ) from error
        information = os.fstat(descriptor)
        if (
            not stat.S_ISREG(information.st_mode)
            or information.st_nlink != 1
            or (information.st_dev, information.st_ino) != identity
            or information.st_size != len(data)
        ):
            _raise(
                ArchiveErrorCode.WRITE_FAILED,
                "write-file",
                "post-write regular-file invariant failed",
                path,
            )
    except BaseException as error:
        _preserve_cleanup_failure(error, "write-file close", lambda: os.close(descriptor))
        _preserve_cleanup_failure(
            error,
            "write-file unlink",
            lambda: _unlink_created(path, identity),
        )
        if isinstance(error, ArchiveError):
            raise
        if isinstance(error, OSError):
            raise ArchiveError(
                ArchiveErrorCode.WRITE_FAILED,
                "write-file",
                str(error),
                path,
            ) from error
        raise
    os.close(descriptor)
    try:
        _fsync_directory_with_progress(path.parent, progress_callback)
    except BaseException as error:
        _preserve_cleanup_failure(
            error,
            "write-file parent-fsync unlink",
            lambda: _unlink_created(path, identity),
        )
        raise
    return FileRecord(
        path.name,
        len(data),
        information.st_mode,
        hashlib.sha256(data).hexdigest(),
        information.st_dev,
        information.st_ino,
        information.st_mtime_ns,
        _line_count(data),
    )


@_archive_public_api
def write_json_exclusive(
    path: Path | str,
    value: object,
    *,
    max_bytes: int,
    mode: int = 0o600,
    progress_callback: ProgressCallback | None = None,
) -> FileRecord:
    _invoke_progress(progress_callback)
    encoded = canonical_json_bytes(value)
    _invoke_progress(progress_callback)
    return _call_archive_internal(
        write_bytes_exclusive,
        path,
        encoded,
        max_bytes=max_bytes,
        mode=mode,
        progress_callback=progress_callback,
    )


@_archive_public_api
def copy_file_exclusive(
    source: Path | str,
    destination: Path | str,
    *,
    max_bytes: int,
    mode: int = 0o600,
    progress_callback: ProgressCallback | None = None,
) -> FileRecord:
    _invoke_progress(progress_callback)
    source = validate_absolute_path(source, "source")
    destination = validate_absolute_path(destination, "destination")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 0:
        _raise(ArchiveErrorCode.INVALID_ARGUMENT, "copy-file", "max_bytes must be nonnegative")
    try:
        source_fd = os.open(
            source,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
    except OSError as error:
        raise _map_open_error(error, "copy-file", source) from error
    destination_fd: int | None = None
    identity: tuple[int, int] | None = None
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            _raise(
                ArchiveErrorCode.PATH_TYPE,
                "copy-file",
                "source is not a single-linked regular file",
                source,
            )
        if before.st_size > max_bytes:
            _raise(
                ArchiveErrorCode.FILE_TOO_LARGE,
                "copy-file",
                f"source exceeds {max_bytes} bytes",
                source,
            )
        destination_fd = _open_exclusive(destination, mode)
        created = os.fstat(destination_fd)
        identity = (created.st_dev, created.st_ino)
        if not stat.S_ISREG(created.st_mode) or created.st_nlink != 1:
            _raise(
                ArchiveErrorCode.PATH_TYPE,
                "copy-file",
                "destination is not a single-linked regular file",
                destination,
            )
        digest = hashlib.sha256()
        total = 0
        line_count = 0
        final_byte = b""
        while True:
            try:
                chunk = os.read(
                    source_fd,
                    min(_READ_CHUNK, max_bytes - total + 1),
                )
            except OSError as error:
                raise ArchiveError(
                    ArchiveErrorCode.READ_FAILED,
                    "copy-file",
                    str(error),
                    source,
                ) from error
            if not chunk:
                break
            _invoke_progress(progress_callback)
            if total + len(chunk) > max_bytes:
                _raise(
                    ArchiveErrorCode.FILE_TOO_LARGE,
                    "copy-file",
                    "source grew beyond bound",
                    source,
                )
            _write_all(
                destination_fd,
                chunk,
                destination,
                progress_callback,
            )
            digest.update(chunk)
            total += len(chunk)
            line_count += chunk.count(b"\n")
            final_byte = chunk[-1:]
        after = os.fstat(source_fd)
        if (
            (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or total != after.st_size
        ):
            _raise(
                ArchiveErrorCode.FILE_CHANGED,
                "copy-file",
                "source changed while copying",
                source,
            )
        try:
            pathname_after = os.lstat(source)
        except OSError as error:
            raise ArchiveError(
                ArchiveErrorCode.FILE_CHANGED,
                "copy-file",
                f"source pathname unavailable after copy: {error}",
                source,
            ) from error
        if (
            stat.S_ISLNK(pathname_after.st_mode)
            or (pathname_after.st_dev, pathname_after.st_ino)
            != (after.st_dev, after.st_ino)
        ):
            _raise(
                ArchiveErrorCode.FILE_CHANGED,
                "copy-file",
                "source pathname no longer identifies the opened file",
                source,
            )
        try:
            _invoke_progress(progress_callback)
            os.fsync(destination_fd)
            _invoke_progress(progress_callback)
        except OSError as error:
            raise ArchiveError(
                ArchiveErrorCode.FSYNC_FAILED,
                "fsync-copy",
                str(error),
                destination,
            ) from error
        copied = os.fstat(destination_fd)
        if (
            not stat.S_ISREG(copied.st_mode)
            or copied.st_nlink != 1
            or (copied.st_dev, copied.st_ino) != identity
            or copied.st_size != total
        ):
            _raise(
                ArchiveErrorCode.WRITE_FAILED,
                "copy-file",
                "destination post-write invariant failed",
                destination,
            )
        os.close(destination_fd)
        destination_fd = None
        _fsync_directory_with_progress(destination.parent, progress_callback)
        if total and final_byte != b"\n":
            line_count += 1
        return FileRecord(
            destination.name,
            total,
            copied.st_mode,
            digest.hexdigest(),
            copied.st_dev,
            copied.st_ino,
            copied.st_mtime_ns,
            line_count,
        )
    except BaseException as error:
        if destination_fd is not None:
            _preserve_cleanup_failure(
                error,
                "copy-file destination close",
                lambda: os.close(destination_fd),
            )
        if identity is not None:
            _preserve_cleanup_failure(
                error,
                "copy-file destination unlink",
                lambda: _unlink_created(destination, identity),
            )
        if isinstance(error, OSError):
            code = (
                ArchiveErrorCode.WRITE_FAILED
                if identity is not None
                else ArchiveErrorCode.READ_FAILED
            )
            raise ArchiveError(code, "copy-file", str(error), destination) from error
        raise
    finally:
        active_error = sys.exc_info()[1]
        try:
            os.close(source_fd)
        except BaseException as close_error:
            if active_error is not None:
                active_error.add_note(
                    "copy-file source close raised "
                    f"{type(close_error).__name__}: {close_error}"
                )
            else:
                raise


@_archive_public_api
def write_state_checkpoint(
    active_dir: Path | str,
    state: object,
    *,
    max_bytes: int,
    progress_callback: ProgressCallback | None = None,
) -> FileRecord:
    _invoke_progress(progress_callback)
    active = validate_absolute_path(active_dir, "active_dir")
    _directory_lstat(active, "checkpoint-state")
    if (
        type(max_bytes) is not int
        or max_bytes < 0
        or max_bytes > _SIGNED_64_MAX
    ):
        _raise(
            ArchiveErrorCode.INVALID_ARGUMENT,
            "checkpoint-state",
            "max_bytes must be a signed-64 nonnegative integer",
            active,
        )
    data = canonical_json_bytes(state)
    _invoke_progress(progress_callback)
    if len(data) > max_bytes:
        _raise(ArchiveErrorCode.FILE_TOO_LARGE, "checkpoint-state", "state exceeds bound", active)
    destination = active / _STATE_NAME
    temporary = active / f".{_STATE_NAME}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    record = _call_archive_internal(
        write_bytes_exclusive,
        temporary,
        data,
        max_bytes=max_bytes,
        progress_callback=progress_callback,
    )
    temporary_identity = (record.device, record.inode)
    temporary_version: tuple[int, int, int, int, int, int, int] | None = None
    renamed = False
    try:
        image = _call_archive_internal(
            read_regular_file_bounded,
            temporary,
            max_bytes,
            progress_callback=progress_callback,
        )
        if (
            (image.device, image.inode) != temporary_identity
            or image.size != record.size
            or image.mode != record.mode
            or image.mtime_ns != record.mtime_ns
            or image.sha256 != record.sha256
        ):
            _raise(
                ArchiveErrorCode.FILE_CHANGED,
                "checkpoint-state",
                "temporary checkpoint changed before installation",
                temporary,
            )
        temporary_version = (
            image.device,
            image.inode,
            image.mode,
            1,
            image.size,
            image.mtime_ns,
            image.ctime_ns,
        )
        try:
            existing = os.lstat(destination)
        except FileNotFoundError:
            existing = None
        if existing is not None and (stat.S_ISLNK(existing.st_mode) or not stat.S_ISREG(existing.st_mode)):
            code = (
                ArchiveErrorCode.SYMLINK_DENIED
                if stat.S_ISLNK(existing.st_mode)
                else ArchiveErrorCode.PATH_TYPE
            )
            _raise(
                code,
                "checkpoint-state",
                "existing state is not a regular file",
                destination,
            )
        _invoke_progress(progress_callback)
        os.replace(temporary, destination)
        renamed = True
        _invoke_progress(progress_callback)
        installed = os.lstat(destination)
        if (
            not stat.S_ISREG(installed.st_mode)
            or installed.st_nlink != 1
            or (installed.st_dev, installed.st_ino) != temporary_identity
        ):
            _raise(
                ArchiveErrorCode.WRITE_FAILED,
                "checkpoint-state",
                "installed state no longer identifies the checkpoint inode",
                destination,
            )
        _fsync_directory_with_progress(active, progress_callback)
    except BaseException as error:
        if isinstance(error, ArchiveError):
            primary: BaseException = error
        elif isinstance(error, OSError):
            primary = ArchiveError(
                ArchiveErrorCode.WRITE_FAILED,
                "checkpoint-state",
                str(error),
                destination,
            )
        else:
            primary = error
        if not renamed and temporary_version is not None:
            _preserve_cleanup_failure(
                primary,
                "checkpoint-state temporary unlink",
                lambda: _unlink_created_version(
                    temporary,
                    temporary_version,
                ),
            )
        if primary is error:
            raise
        raise primary from error
    return FileRecord(
        _STATE_NAME,
        record.size,
        record.mode,
        record.sha256,
        record.device,
        record.inode,
        record.mtime_ns,
        record.line_count,
    )


@_archive_public_api
def read_state_checkpoint(
    active_dir: Path | str,
    *,
    max_bytes: int,
    progress_callback: ProgressCallback | None = None,
) -> object:
    _invoke_progress(progress_callback)
    active = validate_absolute_path(active_dir, "active_dir")
    image = _call_archive_internal(
        read_regular_file_bounded,
        active / _STATE_NAME,
        max_bytes,
        progress_callback=progress_callback,
    )
    value = _parse_json(image.data, ArchiveErrorCode.STATE_INVALID, "read-state")
    _invoke_progress(progress_callback)
    if canonical_json_bytes(value) != image.data:
        _raise(ArchiveErrorCode.STATE_INVALID, "read-state", "state JSON is not canonical", active / _STATE_NAME)
    return value


@_archive_public_api
def initialize_journal(
    active_dir: Path | str,
    *,
    progress_callback: ProgressCallback | None = None,
) -> JournalCursor:
    _invoke_progress(progress_callback)
    active = validate_absolute_path(active_dir, "active_dir")
    _directory_lstat(active, "initialize-journal")
    _call_archive_internal(
        write_bytes_exclusive,
        active / _JOURNAL_NAME,
        b"",
        max_bytes=0,
        progress_callback=progress_callback,
    )
    return JournalCursor(0, _ZERO_HASH)


def _journal_unsigned(sequence: int, previous: str, payload: object) -> dict[str, object]:
    payload_bytes = canonical_json_bytes(payload)
    return {
        "payload": _normalize_json(payload),
        "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "previous_sha256": previous,
        "schema": _JOURNAL_SCHEMA,
        "sequence": sequence,
    }


def _journal_record(sequence: int, previous: str, payload: object) -> dict[str, object]:
    unsigned = _journal_unsigned(sequence, previous, payload)
    entry_hash = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    return {**unsigned, "entry_sha256": entry_hash}


@_archive_public_api
def read_journal(
    active_dir: Path | str,
    *,
    max_entries: int,
    max_entry_bytes: int,
    progress_callback: ProgressCallback | None = None,
) -> JournalRead:
    _invoke_progress(progress_callback)
    active = validate_absolute_path(active_dir, "active_dir")
    if (
        not isinstance(max_entries, int)
        or isinstance(max_entries, bool)
        or max_entries < 0
        or max_entries > _SIGNED_64_MAX
        or not isinstance(max_entry_bytes, int)
        or isinstance(max_entry_bytes, bool)
        or max_entry_bytes <= 0
        or max_entry_bytes > _SIGNED_64_MAX
    ):
        _raise(ArchiveErrorCode.INVALID_ARGUMENT, "read-journal", "invalid journal bounds")
    maximum = min(
        _SIGNED_64_MAX,
        max_entries * (max_entry_bytes + 1),
    )
    try:
        image = _call_archive_internal(
            read_regular_file_bounded,
            active / _JOURNAL_NAME,
            maximum,
            progress_callback=progress_callback,
        )
    except ArchiveError as error:
        if error.code == ArchiveErrorCode.FILE_TOO_LARGE:
            raise ArchiveError(
                ArchiveErrorCode.DIRECTORY_TOO_LARGE,
                "read-journal",
                "journal exceeds population bound",
                active / _JOURNAL_NAME,
            ) from error
        raise
    if not image.data:
        return JournalRead((), JournalCursor(0, _ZERO_HASH))
    if not image.data.endswith(b"\n"):
        _raise(
            ArchiveErrorCode.JOURNAL_INVALID,
            "read-journal",
            "journal has a partial final entry",
            active / _JOURNAL_NAME,
        )
    lines = image.data[:-1].split(b"\n")
    if len(lines) > max_entries:
        _raise(
            ArchiveErrorCode.DIRECTORY_TOO_LARGE,
            "read-journal",
            "journal entry count exceeds bound",
            active / _JOURNAL_NAME,
        )
    previous = _ZERO_HASH
    entries: list[dict[str, Any]] = []
    exact_keys = {
        "entry_sha256",
        "payload",
        "payload_sha256",
        "previous_sha256",
        "schema",
        "sequence",
    }
    for sequence, line in enumerate(lines):
        if not line or len(line) > max_entry_bytes:
            _raise(
                ArchiveErrorCode.JOURNAL_INVALID,
                "read-journal",
                "journal entry is empty or over bound",
                active / _JOURNAL_NAME,
            )
        value = _parse_json(line, ArchiveErrorCode.JOURNAL_INVALID, "read-journal")
        if not isinstance(value, dict) or set(value) != exact_keys or canonical_json_bytes(value) != line:
            _raise(
                ArchiveErrorCode.JOURNAL_INVALID,
                "read-journal",
                "journal entry schema or encoding is invalid",
                active / _JOURNAL_NAME,
            )
        if value["schema"] != _JOURNAL_SCHEMA:
            _raise(
                ArchiveErrorCode.JOURNAL_INVALID,
                "read-journal",
                "journal schema identifier is invalid",
                active / _JOURNAL_NAME,
            )
        if (
            type(value["sequence"]) is not int
            or value["sequence"] < 0
            or value["sequence"] > _SIGNED_64_MAX
            or any(
                type(value[field]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", value[field]) is None
                for field in (
                    "entry_sha256",
                    "payload_sha256",
                    "previous_sha256",
                )
            )
        ):
            _raise(
                ArchiveErrorCode.JOURNAL_INVALID,
                "read-journal",
                "journal sequence or hash field type is invalid",
                active / _JOURNAL_NAME,
            )
        if value["sequence"] != sequence:
            _raise(
                ArchiveErrorCode.JOURNAL_SEQUENCE,
                "read-journal",
                "journal sequence is not contiguous",
                active / _JOURNAL_NAME,
            )
        if value["previous_sha256"] != previous:
            _raise(
                ArchiveErrorCode.JOURNAL_HASH,
                "read-journal",
                "journal predecessor hash mismatch",
                active / _JOURNAL_NAME,
            )
        payload = value["payload"]
        if value["payload_sha256"] != canonical_json_sha256(payload):
            _raise(
                ArchiveErrorCode.JOURNAL_HASH,
                "read-journal",
                "journal payload hash mismatch",
                active / _JOURNAL_NAME,
            )
        unsigned = _journal_unsigned(sequence, previous, payload)
        expected = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
        if value["entry_sha256"] != expected:
            _raise(
                ArchiveErrorCode.JOURNAL_HASH,
                "read-journal",
                "journal entry hash mismatch",
                active / _JOURNAL_NAME,
            )
        previous = expected
        entries.append(value)
        _invoke_progress(progress_callback)
    return JournalRead(tuple(entries), JournalCursor(len(entries), previous))


@_archive_public_api
def append_journal(
    active_dir: Path | str,
    payload: object,
    *,
    cursor: JournalCursor,
    max_entry_bytes: int,
    progress_callback: ProgressCallback | None = None,
) -> JournalCursor:
    _invoke_progress(progress_callback)
    active = validate_absolute_path(active_dir, "active_dir")
    if (
        not isinstance(cursor, JournalCursor)
        or type(cursor.next_sequence) is not int
        or cursor.next_sequence < 0
        or cursor.next_sequence >= _SIGNED_64_MAX
        or type(cursor.previous_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", cursor.previous_sha256) is None
        or type(max_entry_bytes) is not int
        or max_entry_bytes <= 0
        or max_entry_bytes > _SIGNED_64_MAX
    ):
        _raise(ArchiveErrorCode.INVALID_ARGUMENT, "append-journal", "invalid journal cursor")
    path = active / _JOURNAL_NAME
    try:
        descriptor = os.open(
            path,
            os.O_RDWR | os.O_APPEND | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
    except OSError as error:
        raise _map_open_error(error, "append-journal", path) from error
    try:
        information = os.fstat(descriptor)
        if not stat.S_ISREG(information.st_mode) or information.st_nlink != 1:
            _raise(
                ArchiveErrorCode.PATH_TYPE,
                "append-journal",
                "journal is not a single-linked regular file",
                path,
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            observed = _call_archive_internal(
                read_journal,
                active,
                max_entries=max(cursor.next_sequence + 2, 2),
                max_entry_bytes=max_entry_bytes,
                progress_callback=progress_callback,
            )
        except ArchiveError as error:
            if error.code == ArchiveErrorCode.DIRECTORY_TOO_LARGE:
                raise ArchiveError(
                    ArchiveErrorCode.JOURNAL_SEQUENCE,
                    "append-journal",
                    "journal is ahead of supplied cursor",
                    path,
                ) from error
            raise
        if observed.cursor != cursor:
            _raise(
                ArchiveErrorCode.JOURNAL_SEQUENCE,
                "append-journal",
                "supplied cursor is stale or divergent",
                path,
            )
        record = _journal_record(cursor.next_sequence, cursor.previous_sha256, payload)
        _invoke_progress(progress_callback)
        encoded = canonical_json_bytes(record)
        if len(encoded) > max_entry_bytes:
            _raise(
                ArchiveErrorCode.FILE_TOO_LARGE,
                "append-journal",
                "journal entry exceeds bound",
                path,
            )
        _write_all(
            descriptor,
            encoded + b"\n",
            path,
            progress_callback,
        )
        _invoke_progress(progress_callback)
        os.fsync(descriptor)
        _invoke_progress(progress_callback)
        return JournalCursor(cursor.next_sequence + 1, str(record["entry_sha256"]))
    except ArchiveError:
        raise
    except OSError as error:
        raise ArchiveError(ArchiveErrorCode.WRITE_FAILED, "append-journal", str(error), path) from error
    finally:
        active_error = sys.exc_info()[1]
        try:
            os.close(descriptor)
            _fsync_directory_with_progress(active, progress_callback)
        except BaseException as cleanup_error:
            if active_error is not None:
                active_error.add_note(
                    "journal close/directory-fsync failure: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            else:
                raise


def _validate_manifest_bounds(
    max_regular_files_excluding_manifest: int,
    max_directories_excluding_root: int,
    max_regular_file_bytes: int,
    max_total_regular_file_bytes: int,
) -> None:
    values = (
        max_regular_files_excluding_manifest,
        max_directories_excluding_root,
        max_regular_file_bytes,
        max_total_regular_file_bytes,
    )
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > _SIGNED_64_MAX
        for value in values
    ):
        _raise(
            ArchiveErrorCode.INVALID_ARGUMENT,
            "manifest",
            "manifest bounds must be signed-64 nonnegative integers",
        )


def _hash_regular_file(
    path: Path,
    max_bytes: int,
    root_device: int,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    _invoke_progress(progress_callback)
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
    except OSError as error:
        raise _map_open_error(error, "hash-member", path) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            _raise(
                ArchiveErrorCode.PATH_TYPE,
                "hash-member",
                "archive member is not a single-linked regular file",
                path,
            )
        if before.st_dev != root_device:
            _raise(
                ArchiveErrorCode.CROSS_DEVICE,
                "hash-member",
                "archive member crosses filesystem",
                path,
            )
        if before.st_size > max_bytes:
            _raise(
                ArchiveErrorCode.FILE_TOO_LARGE,
                "hash-member",
                f"archive member exceeds {max_bytes} bytes",
                path,
            )
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, _READ_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                _raise(
                    ArchiveErrorCode.FILE_TOO_LARGE,
                    "hash-member",
                    "archive member grew beyond bound",
                    path,
                )
            digest.update(chunk)
            _invoke_progress(progress_callback)
        after = os.fstat(descriptor)
        version_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        version_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if version_before != version_after or total != after.st_size:
            _raise(
                ArchiveErrorCode.FILE_CHANGED,
                "hash-member",
                "archive member changed while hashing",
                path,
            )
        try:
            pathname_after = os.lstat(path)
        except OSError as error:
            raise ArchiveError(
                ArchiveErrorCode.FILE_CHANGED,
                "hash-member",
                f"member pathname unavailable after hashing: {error}",
                path,
            ) from error
        if stat.S_ISLNK(pathname_after.st_mode):
            _raise(
                ArchiveErrorCode.SYMLINK_DENIED,
                "hash-member",
                "member pathname became a symlink",
                path,
            )
        if (pathname_after.st_dev, pathname_after.st_ino) != (
            after.st_dev,
            after.st_ino,
        ):
            _raise(
                ArchiveErrorCode.FILE_CHANGED,
                "hash-member",
                "member pathname no longer identifies the opened file",
                path,
            )
        return {
            "mode": after.st_mode,
            "mtime_ns": after.st_mtime_ns,
            "sha256": digest.hexdigest(),
            "size": total,
        }
    except ArchiveError:
        raise
    except OSError as error:
        raise ArchiveError(
            ArchiveErrorCode.READ_FAILED,
            "hash-member",
            str(error),
            path,
        ) from error
    finally:
        _close_descriptor_preserving_primary(
            descriptor,
            sys.exc_info()[1],
            "hash-member",
            ArchiveErrorCode.READ_FAILED,
            path,
        )


def _scan_manifest_population(
    root: Path,
    manifest_relative_path: str,
    *,
    max_regular_files_excluding_manifest: int,
    max_directories_excluding_root: int,
    max_regular_file_bytes: int,
    max_total_regular_file_bytes: int,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict[str, object]], list[str], int, list[Path]]:
    _invoke_progress(progress_callback)
    root_info = _directory_lstat(root, "scan-archive")
    root_device = root_info.st_dev
    stack = [root]
    directories = [root]
    entries_seen = 0
    records: list[dict[str, object]] = []
    directory_paths: list[str] = []
    regular_files_seen = 0
    directories_seen = 0
    completed_since_progress = 0
    total = 0
    while stack:
        directory = stack.pop()
        try:
            with _ScandirScope(directory) as iterator:
                for child in iterator:
                    if completed_since_progress == _DIRECTORY_PROGRESS_BATCH:
                        _invoke_progress(progress_callback)
                        completed_since_progress = 0
                    path = Path(child.path)
                    relative = path.relative_to(root).as_posix()
                    completed_since_progress += 1
                    if relative == manifest_relative_path:
                        continue
                    validate_relative_path(relative)
                    entries_seen += 1
                    if entries_seen > (
                        max_regular_files_excluding_manifest
                        + max_directories_excluding_root
                    ):
                        _raise(
                            ArchiveErrorCode.DIRECTORY_TOO_LARGE,
                            "scan-archive",
                            "archive population exceeds the combined regular-file "
                            "and directory bounds",
                            root,
                        )
                    try:
                        information = os.lstat(path)
                    except FileNotFoundError as error:
                        raise ArchiveError(
                            ArchiveErrorCode.FILE_CHANGED,
                            "scan-archive",
                            str(error),
                            path,
                        ) from error
                    if stat.S_ISLNK(information.st_mode):
                        _raise(
                            ArchiveErrorCode.SYMLINK_DENIED,
                            "scan-archive",
                            "archive contains a symlink",
                            path,
                        )
                    if information.st_dev != root_device:
                        _raise(
                            ArchiveErrorCode.CROSS_DEVICE,
                            "scan-archive",
                            "archive member crosses filesystem",
                            path,
                        )
                    if stat.S_ISDIR(information.st_mode):
                        directories_seen += 1
                        if directories_seen > max_directories_excluding_root:
                            _raise(
                                ArchiveErrorCode.DIRECTORY_TOO_LARGE,
                                "scan-archive",
                                "archive directory population exceeds "
                                f"{max_directories_excluding_root}",
                                root,
                            )
                        directory_paths.append(relative)
                        directories.append(path)
                        stack.append(path)
                        continue
                    if not stat.S_ISREG(information.st_mode):
                        _raise(
                            ArchiveErrorCode.PATH_TYPE,
                            "scan-archive",
                            "archive contains a special file",
                            path,
                        )
                    regular_files_seen += 1
                    if regular_files_seen > max_regular_files_excluding_manifest:
                        _raise(
                            ArchiveErrorCode.DIRECTORY_TOO_LARGE,
                            "scan-archive",
                            "archive regular-file population exceeds "
                            f"{max_regular_files_excluding_manifest}",
                            root,
                        )
                    record = _hash_regular_file(
                        path,
                        max_regular_file_bytes,
                        root_device,
                        progress_callback,
                    )
                    record["path"] = relative
                    total += int(record["size"])
                    if total > max_total_regular_file_bytes:
                        _raise(
                            ArchiveErrorCode.TOTAL_TOO_LARGE,
                            "scan-archive",
                            "archive regular files exceed "
                            f"{max_total_regular_file_bytes} total bytes",
                            root,
                        )
                    records.append(record)
        except OSError as error:
            raise ArchiveError(
                ArchiveErrorCode.READ_FAILED,
                "scan-archive",
                str(error),
                directory,
            ) from error
    if completed_since_progress:
        _invoke_progress(progress_callback)
    records.sort(key=lambda item: str(item["path"]))
    directory_paths.sort()
    return records, directory_paths, total, directories


def _manifest_document(
    records: list[dict[str, object]],
    directory_paths: list[str],
    total: int,
) -> dict[str, object]:
    return {
        "directories": directory_paths,
        "directory_count": len(directory_paths),
        "regular_file_count": len(records),
        "regular_files": records,
        "schema": _MANIFEST_SCHEMA,
        "total_regular_file_bytes": total,
    }


@_archive_public_api
def create_manifest(
    root: Path | str,
    *,
    manifest_relative_path: str,
    max_regular_files_excluding_manifest: int,
    max_directories_excluding_root: int,
    max_regular_file_bytes: int,
    max_total_regular_file_bytes: int,
    max_manifest_bytes: int,
    progress_callback: ProgressCallback | None = None,
) -> ManifestSummary:
    _invoke_progress(progress_callback)
    root = validate_absolute_path(root, "root")
    manifest_relative_path = validate_relative_path(manifest_relative_path)
    _validate_manifest_bounds(
        max_regular_files_excluding_manifest,
        max_directories_excluding_root,
        max_regular_file_bytes,
        max_total_regular_file_bytes,
    )
    if (
        type(max_manifest_bytes) is not int
        or max_manifest_bytes < 0
        or max_manifest_bytes > _SIGNED_64_MAX
    ):
        _raise(
            ArchiveErrorCode.INVALID_ARGUMENT,
            "create-manifest",
            "invalid manifest bound",
        )
    records, directory_paths, total, _ = _scan_manifest_population(
        root,
        manifest_relative_path,
        max_regular_files_excluding_manifest=(
            max_regular_files_excluding_manifest
        ),
        max_directories_excluding_root=max_directories_excluding_root,
        max_regular_file_bytes=max_regular_file_bytes,
        max_total_regular_file_bytes=max_total_regular_file_bytes,
        progress_callback=progress_callback,
    )
    document = _manifest_document(records, directory_paths, total)
    encoded = canonical_json_bytes(document)
    _invoke_progress(progress_callback)
    manifest_path = safe_join(root, manifest_relative_path)
    _call_archive_internal(
        write_bytes_exclusive,
        manifest_path,
        encoded,
        max_bytes=max_manifest_bytes,
        progress_callback=progress_callback,
    )
    return ManifestSummary(
        manifest_relative_path,
        len(records),
        len(directory_paths),
        total,
        hashlib.sha256(encoded).hexdigest(),
    )


def _manifest_read_bound(
    max_regular_files_excluding_manifest: int,
    max_directories_excluding_root: int,
) -> int:
    return min(
        _SIGNED_64_MAX,
        4096
        + (
            max_regular_files_excluding_manifest
            + max_directories_excluding_root
        )
        * 8192,
    )


def _manifest_relative(
    value: object,
    *,
    manifest_path: Path,
    population: str,
) -> str:
    if type(value) is not str:
        _raise(
            ArchiveErrorCode.MANIFEST_INVALID,
            "verify-manifest",
            f"manifest {population} path is not a string",
            manifest_path,
        )
    try:
        return validate_relative_path(value)
    except ArchiveError as error:
        raise ArchiveError(
            ArchiveErrorCode.MANIFEST_INVALID,
            "verify-manifest",
            f"manifest {population} path is invalid: {error.detail}",
            manifest_path,
        ) from error


@_archive_public_api
def verify_manifest(
    root: Path | str,
    *,
    manifest_relative_path: str,
    max_regular_files_excluding_manifest: int,
    max_directories_excluding_root: int,
    max_regular_file_bytes: int,
    max_total_regular_file_bytes: int,
    progress_callback: ProgressCallback | None = None,
) -> ManifestSummary:
    _invoke_progress(progress_callback)
    root = validate_absolute_path(root, "root")
    manifest_relative_path = validate_relative_path(manifest_relative_path)
    _validate_manifest_bounds(
        max_regular_files_excluding_manifest,
        max_directories_excluding_root,
        max_regular_file_bytes,
        max_total_regular_file_bytes,
    )
    manifest_path = safe_join(root, manifest_relative_path)
    try:
        image = _call_archive_internal(
            read_regular_file_bounded,
            manifest_path,
            _manifest_read_bound(
                max_regular_files_excluding_manifest,
                max_directories_excluding_root,
            ),
            progress_callback=progress_callback,
        )
    except ArchiveError as error:
        if error.code == ArchiveErrorCode.PATH_MISSING:
            raise ArchiveError(
                ArchiveErrorCode.MANIFEST_MISSING,
                "verify-manifest",
                "manifest is missing",
                manifest_path,
            ) from error
        raise
    document = _parse_json(
        image.data,
        ArchiveErrorCode.MANIFEST_INVALID,
        "verify-manifest",
    )
    if (
        not isinstance(document, dict)
        or set(document)
        != {
            "directories",
            "directory_count",
            "regular_file_count",
            "regular_files",
            "schema",
            "total_regular_file_bytes",
        }
        or canonical_json_bytes(document) != image.data
    ):
        _raise(
            ArchiveErrorCode.MANIFEST_INVALID,
            "verify-manifest",
            "manifest schema or encoding is invalid",
            manifest_path,
        )
    if (
        document["schema"] != _MANIFEST_SCHEMA
        or not isinstance(document["regular_files"], list)
        or not isinstance(document["directories"], list)
    ):
        _raise(
            ArchiveErrorCode.MANIFEST_INVALID,
            "verify-manifest",
            "manifest header is invalid",
            manifest_path,
        )
    expected_records = document["regular_files"]
    expected_directories = document["directories"]
    expected_keys = {"mode", "mtime_ns", "path", "sha256", "size"}
    previous_directory = ""
    normalized_directories: list[str] = []
    for index, value in enumerate(expected_directories, 1):
        relative = _manifest_relative(
            value,
            manifest_path=manifest_path,
            population="directory",
        )
        if relative == manifest_relative_path:
            _raise(
                ArchiveErrorCode.MANIFEST_INVALID,
                "verify-manifest",
                "manifest path appears in the directory population",
                manifest_path,
            )
        if relative <= previous_directory:
            _raise(
                ArchiveErrorCode.MANIFEST_INVALID,
                "verify-manifest",
                "manifest directory population is not strictly sorted",
                manifest_path,
            )
        previous_directory = relative
        normalized_directories.append(relative)
        if index % _DIRECTORY_PROGRESS_BATCH == 0:
            _invoke_progress(progress_callback)
    if (
        expected_directories
        and len(expected_directories) % _DIRECTORY_PROGRESS_BATCH
    ):
        _invoke_progress(progress_callback)
    previous_path = ""
    normalized_file_paths: list[str] = []
    for index, record in enumerate(expected_records, 1):
        if not isinstance(record, dict) or set(record) != expected_keys:
            _raise(
                ArchiveErrorCode.MANIFEST_INVALID,
                "verify-manifest",
                "manifest file record is invalid",
                manifest_path,
            )
        relative = _manifest_relative(
            record["path"],
            manifest_path=manifest_path,
            population="regular-file",
        )
        if relative == manifest_relative_path:
            _raise(
                ArchiveErrorCode.MANIFEST_INVALID,
                "verify-manifest",
                "manifest regular-file population contains the manifest itself",
                manifest_path,
            )
        if relative <= previous_path:
            _raise(
                ArchiveErrorCode.MANIFEST_INVALID,
                "verify-manifest",
                "manifest regular-file population is not strictly sorted",
                manifest_path,
            )
        previous_path = relative
        normalized_file_paths.append(relative)
        if (
            type(record["mode"]) is not int
            or record["mode"] < 0
            or not stat.S_ISREG(record["mode"])
        ):
            _raise(
                ArchiveErrorCode.MANIFEST_INVALID,
                "verify-manifest",
                "manifest mode is invalid",
                manifest_path,
            )
        if (
            type(record["mtime_ns"]) is not int
            or record["mtime_ns"] < 0
            or record["mtime_ns"] > _SIGNED_64_MAX
            or type(record["size"]) is not int
            or record["size"] < 0
            or record["size"] > max_regular_file_bytes
            or type(record["sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
        ):
            _raise(
                ArchiveErrorCode.MANIFEST_INVALID,
                "verify-manifest",
                "manifest metadata is invalid",
                manifest_path,
            )
        if index % _DIRECTORY_PROGRESS_BATCH == 0:
            _invoke_progress(progress_callback)
    if expected_records and len(expected_records) % _DIRECTORY_PROGRESS_BATCH:
        _invoke_progress(progress_callback)
    if (
        type(document["regular_file_count"]) is not int
        or document["regular_file_count"] < 0
        or document["regular_file_count"] != len(expected_records)
        or len(expected_records) > max_regular_files_excluding_manifest
        or type(document["directory_count"]) is not int
        or document["directory_count"] < 0
        or document["directory_count"] != len(expected_directories)
        or len(expected_directories) > max_directories_excluding_root
        or type(document["total_regular_file_bytes"]) is not int
        or document["total_regular_file_bytes"] < 0
        or document["total_regular_file_bytes"]
        > max_total_regular_file_bytes
        or sum(int(record["size"]) for record in expected_records)
        != document["total_regular_file_bytes"]
        or bool(set(normalized_file_paths) & set(normalized_directories))
    ):
        _raise(
            ArchiveErrorCode.MANIFEST_INVALID,
            "verify-manifest",
            "manifest population count or byte total is invalid",
            manifest_path,
        )
    observed_records, observed_directories, observed_total, _ = (
        _scan_manifest_population(
            root,
            manifest_relative_path,
            max_regular_files_excluding_manifest=(
                max_regular_files_excluding_manifest
            ),
            max_directories_excluding_root=max_directories_excluding_root,
            max_regular_file_bytes=max_regular_file_bytes,
            max_total_regular_file_bytes=max_total_regular_file_bytes,
            progress_callback=progress_callback,
        )
    )
    expected_directory_set = set(normalized_directories)
    observed_directory_set = set(observed_directories)
    missing_directories = sorted(expected_directory_set - observed_directory_set)
    if missing_directories:
        _raise(
            ArchiveErrorCode.PATH_MISSING,
            "verify-manifest",
            f"manifest directory is missing: {missing_directories[0]}",
            root / missing_directories[0],
        )
    if observed_directory_set != expected_directory_set:
        _raise(
            ArchiveErrorCode.MANIFEST_INVALID,
            "verify-manifest",
            "archive has an unmanifested directory",
            root,
        )
    expected_by_path = {str(record["path"]): record for record in expected_records}
    observed_by_path = {str(record["path"]): record for record in observed_records}
    missing = sorted(set(expected_by_path) - set(observed_by_path))
    if missing:
        _raise(
            ArchiveErrorCode.PATH_MISSING,
            "verify-manifest",
            f"manifest member is missing: {missing[0]}",
            root / missing[0],
        )
    if set(observed_by_path) != set(expected_by_path):
        _raise(
            ArchiveErrorCode.MANIFEST_INVALID,
            "verify-manifest",
            "archive has an unmanifested file",
            root,
        )
    for relative, expected in expected_by_path.items():
        observed = observed_by_path[relative]
        if observed != expected:
            code = (
                ArchiveErrorCode.HASH_MISMATCH
                if observed["sha256"] != expected["sha256"]
                else ArchiveErrorCode.FILE_CHANGED
            )
            _raise(
                code,
                "verify-manifest",
                f"manifest metadata mismatch for {relative}",
                root / relative,
            )
    if (
        document["total_regular_file_bytes"] != observed_total
        or observed_total > max_total_regular_file_bytes
    ):
        _raise(
            ArchiveErrorCode.HASH_MISMATCH,
            "verify-manifest",
            "manifest total size mismatch",
            root,
        )
    return ManifestSummary(
        manifest_relative_path,
        len(observed_records),
        len(observed_directories),
        observed_total,
        image.sha256,
    )


def _fsync_tree(
    root: Path,
    manifest_relative_path: str,
    *,
    max_regular_files_excluding_manifest: int,
    max_directories_excluding_root: int,
    max_regular_file_bytes: int,
    max_total_regular_file_bytes: int,
    progress_callback: ProgressCallback | None = None,
) -> None:
    _invoke_progress(progress_callback)
    records, _, _, observed_directories = _scan_manifest_population(
        root,
        manifest_relative_path,
        max_regular_files_excluding_manifest=(
            max_regular_files_excluding_manifest
        ),
        max_directories_excluding_root=max_directories_excluding_root,
        max_regular_file_bytes=max_regular_file_bytes,
        max_total_regular_file_bytes=max_total_regular_file_bytes,
        progress_callback=progress_callback,
    )
    records.append({"path": manifest_relative_path})
    root_device = _directory_lstat(root, "fsync-archive-tree").st_dev
    for record in records:
        path = safe_join(root, str(record["path"]))
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            )
            try:
                information = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(information.st_mode)
                    or information.st_nlink != 1
                    or information.st_dev != root_device
                ):
                    _raise(
                        ArchiveErrorCode.PATH_TYPE,
                        "fsync-archive-file",
                        "archive fsync target is not a local single-linked regular file",
                        path,
                    )
                _invoke_progress(progress_callback)
                os.fsync(descriptor)
                _invoke_progress(progress_callback)
                pathname_after = os.lstat(path)
                if (pathname_after.st_dev, pathname_after.st_ino) != (
                    information.st_dev,
                    information.st_ino,
                ):
                    _raise(
                        ArchiveErrorCode.FILE_CHANGED,
                        "fsync-archive-file",
                        "archive fsync pathname changed",
                        path,
                    )
            finally:
                _close_descriptor_preserving_primary(
                    descriptor,
                    sys.exc_info()[1],
                    "fsync-archive-file",
                    ArchiveErrorCode.FSYNC_FAILED,
                    path,
                )
        except ArchiveError:
            raise
        except OSError as error:
            raise ArchiveError(
                ArchiveErrorCode.FSYNC_FAILED,
                "fsync-archive-file",
                str(error),
                path,
            ) from error
    for directory in sorted(
        set(observed_directories),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _fsync_directory_with_progress(directory, progress_callback)


def _renameat2_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable")
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    result = function(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(destination))


@_archive_public_api
def publish_verified_archive(
    active_dir: Path | str,
    final_dir: Path | str,
    *,
    manifest_relative_path: str,
    max_regular_files_excluding_manifest: int,
    max_directories_excluding_root: int,
    max_regular_file_bytes: int,
    max_total_regular_file_bytes: int,
    progress_callback: ProgressCallback | None = None,
) -> PublicationResult:
    phase = PublicationPhase.PRE_RENAME
    try:
        _invoke_progress(progress_callback)
        active = validate_absolute_path(active_dir, "active_dir")
        final = validate_absolute_path(final_dir, "final_dir")
        if active == final or active in final.parents or final in active.parents:
            _raise(
                ArchiveErrorCode.INVALID_ARGUMENT,
                "publish-archive",
                "active and final directories must be distinct and non-nested",
                active,
            )
        active_info = _directory_lstat(active, "publish-archive")
        final_parent_info = _directory_lstat(final.parent, "publish-archive")
        if active_info.st_dev != final_parent_info.st_dev:
            _raise(
                ArchiveErrorCode.CROSS_DEVICE,
                "publish-archive",
                "active and final parent are on different filesystems",
                final,
            )
        summary = _call_archive_internal(
            verify_manifest,
            active,
            manifest_relative_path=manifest_relative_path,
            max_regular_files_excluding_manifest=(
                max_regular_files_excluding_manifest
            ),
            max_directories_excluding_root=max_directories_excluding_root,
            max_regular_file_bytes=max_regular_file_bytes,
            max_total_regular_file_bytes=max_total_regular_file_bytes,
            progress_callback=progress_callback,
        )
        _fsync_tree(
            active,
            manifest_relative_path,
            max_regular_files_excluding_manifest=(
                max_regular_files_excluding_manifest
            ),
            max_directories_excluding_root=max_directories_excluding_root,
            max_regular_file_bytes=max_regular_file_bytes,
            max_total_regular_file_bytes=max_total_regular_file_bytes,
            progress_callback=progress_callback,
        )
        _invoke_progress(progress_callback)
        try:
            _renameat2_noreplace(active, final)
        except OSError as error:
            if error.errno == errno.EEXIST:
                code = ArchiveErrorCode.PUBLISH_EXISTS
            elif error.errno == errno.EXDEV:
                code = ArchiveErrorCode.CROSS_DEVICE
            else:
                code = ArchiveErrorCode.PUBLISH_FAILED
            raise ArchiveError(
                code,
                "publish-archive",
                str(error),
                final,
                phase,
            ) from error
        phase = PublicationPhase.RENAMED_UNSYNCED
        _invoke_progress(progress_callback)
        if active.parent == final.parent:
            _invoke_progress(progress_callback)
            _fsync_directory(final.parent)
            phase = PublicationPhase.PARENTS_SYNCED
            _invoke_progress(progress_callback)
        else:
            _invoke_progress(progress_callback)
            _fsync_directory(final.parent)
            phase = PublicationPhase.FINAL_PARENT_SYNCED
            _invoke_progress(progress_callback)
            _invoke_progress(progress_callback)
            _fsync_directory(active.parent)
            phase = PublicationPhase.PARENTS_SYNCED
            _invoke_progress(progress_callback)
        observed = _call_archive_internal(
            verify_manifest,
            final,
            manifest_relative_path=manifest_relative_path,
            max_regular_files_excluding_manifest=(
                max_regular_files_excluding_manifest
            ),
            max_directories_excluding_root=max_directories_excluding_root,
            max_regular_file_bytes=max_regular_file_bytes,
            max_total_regular_file_bytes=max_total_regular_file_bytes,
            progress_callback=progress_callback,
        )
        if observed != summary:
            _raise(
                ArchiveErrorCode.HASH_MISMATCH,
                "publish-archive",
                "post-publication verification differs",
                final,
            )
        phase = PublicationPhase.VERIFIED
        _invoke_progress(progress_callback)
        return PublicationResult(observed, phase)
    except _CallbackOrigin as error:
        error.factual_publication_phase = phase
        raise
    except ArchiveError as error:
        if error.publication_phase is None:
            error.publication_phase = phase
        raise
    except BaseException as error:
        error.add_note(f"publication_phase={phase.value}")
        raise


@_archive_public_api
def inspect_incomplete(
    active_dir: Path | str,
    *,
    max_state_bytes: int,
    max_journal_entries: int,
    max_journal_entry_bytes: int,
    manifest_relative_path: str,
    max_regular_files_excluding_manifest: int,
    max_directories_excluding_root: int,
    max_regular_file_bytes: int,
    max_total_regular_file_bytes: int,
    progress_callback: ProgressCallback | None = None,
) -> RecoveryInspection:
    _invoke_progress(progress_callback)
    active = validate_absolute_path(active_dir, "active_dir")
    state: object | None = None
    journal: JournalRead | None = None
    reasons: list[str] = []
    try:
        state = _call_archive_internal(
            read_state_checkpoint,
            active,
            max_bytes=max_state_bytes,
            progress_callback=progress_callback,
        )
    except ArchiveError as error:
        reasons.append(f"state:{error.code.value}:{error.detail}")
    try:
        journal = _call_archive_internal(
            read_journal,
            active,
            max_entries=max_journal_entries,
            max_entry_bytes=max_journal_entry_bytes,
            progress_callback=progress_callback,
        )
    except ArchiveError as error:
        reasons.append(f"journal:{error.code.value}:{error.detail}")
    manifest_path = safe_join(active, manifest_relative_path)
    try:
        _invoke_progress(progress_callback)
        os.lstat(manifest_path)
        _invoke_progress(progress_callback)
        manifest_present = True
    except FileNotFoundError:
        manifest_present = False
    except OSError as error:
        manifest_present = True
        reasons.append(f"manifest-open:{error}")
    manifest_verified = False
    if manifest_present:
        try:
            _call_archive_internal(
                verify_manifest,
                active,
                manifest_relative_path=manifest_relative_path,
                max_regular_files_excluding_manifest=(
                    max_regular_files_excluding_manifest
                ),
                max_directories_excluding_root=max_directories_excluding_root,
                max_regular_file_bytes=max_regular_file_bytes,
                max_total_regular_file_bytes=max_total_regular_file_bytes,
                progress_callback=progress_callback,
            )
            manifest_verified = True
        except ArchiveError as error:
            reasons.append(f"manifest:{error.code.value}:{error.detail}")
    result = RecoveryInspection(
        str(active),
        state,
        journal,
        manifest_present,
        manifest_verified,
        "; ".join(reasons) or None,
    )
    _invoke_progress(progress_callback)
    return result
