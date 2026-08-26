#!/usr/bin/env python3
"""Offline composer for authenticated OAI memory-lifetime process evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import pathlib
import re
import stat
import subprocess
import sys
import uuid
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Mapping, Sequence


ROOT = pathlib.Path(__file__).resolve().parents[3]
CATALOG_ROOT = ROOT / "tools/profiling/memory/catalog_v1"
INTEGRATION_PATH = CATALOG_ROOT / "integration/archive_semantic_verifier_v1.py"
HANDOFF_PATH = pathlib.Path(__file__).resolve().with_name("oai_memprof_process_handoff.py")
STREAM_ARCHIVE_PATH = "streams/memory-lifetime.bin"
PROCESS_HANDOFF_ARCHIVE_PATH = "streams/process-handoff.bin"
PRODUCER_DEFINITION_PATH = "definition/oai-memprof-archive-composer-v1.py"
HANDOFF_DECODER_DEFINITION_PATH = "definition/oai-memprof-process-handoff-v1.py"
BUILD_EVIDENCE_DEFINITION_PATH = "definition/oai-memprof-build-evidence-v1.py"
BUILD_EVIDENCE_INPUT_PATH = "input/build-evidence.json"
BUILD_INPUT_PATH = "input/build-coverage.json"
MAX_BUILD_COVERAGE_BYTES = 16 * 1024 * 1024
MAX_BUILD_EVIDENCE_BYTES = 16 * 1024 * 1024
MAX_BUILD_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_BUILD_ARTIFACT_ENTRIES = 256
MAX_BUILD_ARTIFACT_TOTAL_BYTES = 1 * 1024 * 1024 * 1024
MAX_STREAM_BYTES = 64 * 1024 * 1024 * 1024
_MAP_RE = re.compile(
    rb"\A([0-9a-f]+)-([0-9a-f]+) ([r-][w-][x-][ps]) ([0-9a-f]+) "
    rb"([0-9a-f]+):([0-9a-f]+) ([0-9]+)(?: +(.*))?\Z"
)
_GLIBC_RUNTIME_FILENAME_RE = re.compile(
    r"libc(?:\.so\.6|-[0-9]+(?:\.[0-9]+)*\.so)"
)


class ArchiveComposerError(RuntimeError):
    """Fail-closed composition error."""


def _load(name: str, path: pathlib.Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ArchiveComposerError(f"module unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HANDOFF = _load("_oai_memprof_archive_handoff", HANDOFF_PATH)
VERIFIER = _load("_oai_memprof_archive_verifier", INTEGRATION_PATH)
BUILD_EVIDENCE = VERIFIER.BUILD_EVIDENCE
MAX_HANDOFF_BYTES = HANDOFF.MAX_WIRE_BYTES


@dataclass(frozen=True)
class FrozenInput:
    path: pathlib.Path
    raw: bytes
    device: int
    inode: int
    mode: int
    nlink: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class CompositionResult:
    archive_directory: pathlib.Path
    stream_path: pathlib.Path
    handoff_sha256: str
    stream_sha256: str
    manifest_sha256: str
    scientific_admission_complete: bool
    terminal_outcome: str = "complete"
    admission_blockers: tuple[str, ...] = ()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _stat_identity(status: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    """Return the stable identity fields suitable for a no-follow rebind check."""

    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_nlink,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _frozen_identity(file: FrozenInput) -> tuple[int, int, int, int, int, int, int]:
    return (
        file.device,
        file.inode,
        file.mode,
        file.nlink,
        file.size,
        file.mtime_ns,
        file.ctime_ns,
    )


def _same_frozen_identity(left: FrozenInput, right: FrozenInput) -> bool:
    return _frozen_identity(left) == _frozen_identity(right)


def _read_frozen(
    path: pathlib.Path,
    maximum: int,
    *,
    allow_empty: bool = False,
    require_single_link: bool = True,
) -> FrozenInput:
    if not path.is_absolute():
        raise ArchiveComposerError(f"absolute path required: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink < 1
            or (require_single_link and before.st_nlink != 1)
            or (before.st_size == 0 and not allow_empty)
            or before.st_size > maximum
        ):
            raise ArchiveComposerError(f"bounded single-link regular file required: {path}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(remaining, 1 << 20))
            if not chunk:
                raise ArchiveComposerError(f"short read: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(fd)
        if _stat_identity(after) != _stat_identity(before):
            raise ArchiveComposerError(f"file changed during read: {path}")
    finally:
        os.close(fd)
    current = os.lstat(path)
    if _stat_identity(current) != _stat_identity(before):
        raise ArchiveComposerError(f"path changed after read: {path}")
    return FrozenInput(
        path,
        raw,
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )


def _read_frozen_source_file(
    source_root_fd: int,
    source_root: pathlib.Path,
    repository_relative_path: str,
    maximum: int,
) -> FrozenInput:
    """Read one regular source leaf relative to one pinned no-follow root fd."""

    if (
        not isinstance(repository_relative_path, str)
        or not repository_relative_path
        or repository_relative_path.startswith("/")
        or "\\" in repository_relative_path
    ):
        raise ArchiveComposerError("normalized repository-relative source path required")
    parts = repository_relative_path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ArchiveComposerError("normalized repository-relative source path required")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    directory_fd = os.dup(source_root_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(
                part,
                flags | os.O_DIRECTORY,
                dir_fd=directory_fd,
            )
            try:
                if not stat.S_ISDIR(os.fstat(next_fd).st_mode):
                    raise ArchiveComposerError(
                        f"source parent is not a directory: {repository_relative_path}"
                    )
            except Exception:
                os.close(next_fd)
                raise
            os.close(directory_fd)
            directory_fd = next_fd
        fd = os.open(parts[-1], flags, dir_fd=directory_fd)
        try:
            before = os.fstat(fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size <= 0
                or before.st_size > maximum
            ):
                raise ArchiveComposerError(
                    f"bounded single-link regular source required: {repository_relative_path}"
                )
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(fd, min(remaining, 1 << 20))
                if not chunk:
                    raise ArchiveComposerError(
                        f"short source read: {repository_relative_path}"
                    )
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            if _stat_identity(os.fstat(fd)) != _stat_identity(before):
                raise ArchiveComposerError(
                    f"source changed during read: {repository_relative_path}"
                )
        finally:
            os.close(fd)
    finally:
        os.close(directory_fd)
    return FrozenInput(
        source_root.joinpath(*parts),
        raw,
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )


def _read_frozen_trusted_release_sources(
    source_root: pathlib.Path,
) -> dict[str, bytes]:
    """Freeze all authority sources beneath one explicit, no-follow root.

    The controller supplies this root and its separately pinned authority digest.
    The pinned directory fd prevents later path replacement from redirecting a
    source read. This remains a same-process/same-UID best effort: a coordinated
    writer can still race pathname validation before the root fd is acquired.
    """

    source_root = pathlib.Path(source_root)
    if not source_root.is_absolute():
        raise ArchiveComposerError(f"absolute trusted-release source root required: {source_root}")
    try:
        root_before = os.lstat(source_root)
    except OSError as error:
        raise ArchiveComposerError("trusted-release source root unavailable") from error
    if stat.S_ISLNK(root_before.st_mode) or not stat.S_ISDIR(root_before.st_mode):
        raise ArchiveComposerError("trusted-release source root must be a non-symlink directory")
    try:
        root_fd = os.open(
            source_root,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY,
        )
    except OSError as error:
        raise ArchiveComposerError("trusted-release source root unavailable") from error
    try:
        if _stat_identity(os.fstat(root_fd)) != _stat_identity(root_before):
            raise ArchiveComposerError("trusted-release source root changed before read")
        try:
            sources = {
                archive_path: _read_frozen_source_file(
                    root_fd,
                    source_root,
                    repository_relative_path,
                    VERIFIER.MAX_TRUSTED_RELEASE_SOURCE_BYTES,
                ).raw
                for archive_path, repository_relative_path in VERIFIER.TRUSTED_RELEASE_SOURCE_PATHS.items()
            }
        except OSError as error:
            raise ArchiveComposerError("trusted-release source leaf unavailable") from error
        if _stat_identity(os.fstat(root_fd)) != _stat_identity(root_before):
            raise ArchiveComposerError("trusted-release source root changed during read")
    finally:
        os.close(root_fd)
    try:
        root_after = os.lstat(source_root)
    except OSError as error:
        raise ArchiveComposerError("trusted-release source root unavailable") from error
    if _stat_identity(root_after) != _stat_identity(root_before):
        raise ArchiveComposerError("trusted-release source root changed after read")
    return sources


def _read_trusted_release_inputs(
    authority_path: pathlib.Path, source_root: pathlib.Path
) -> tuple[FrozenInput, dict[str, bytes]]:
    """Read the externally pinned authority and all 15 source bytes before publication."""

    authority_path = pathlib.Path(authority_path)
    if not authority_path.is_absolute():
        raise ArchiveComposerError(f"absolute trusted-release authority path required: {authority_path}")
    try:
        authority = _read_frozen(
            authority_path,
            VERIFIER.MAX_TRUSTED_RELEASE_AUTHORITY_BYTES,
        )
    except OSError as error:
        raise ArchiveComposerError("trusted-release authority unavailable") from error
    return authority, _read_frozen_trusted_release_sources(source_root)


def _write_all(fd: int, raw: bytes) -> None:
    offset = 0
    while offset != len(raw):
        count = os.write(fd, raw[offset:])
        if count <= 0:
            raise ArchiveComposerError("short archive write")
        offset += count


def _publish_once(root: pathlib.Path, relative_path: str, raw: bytes) -> FrozenInput:
    canonical = VERIFIER._archive_path(relative_path, "archive publication path")
    root_fd = os.open(
        root, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    directory_fd = os.dup(root_fd)
    published: FrozenInput | None = None
    try:
        parts = canonical.split("/")
        for part in parts[:-1]:
            try:
                os.mkdir(part, 0o750, dir_fd=directory_fd)
                os.fsync(directory_fd)
            except FileExistsError:
                pass
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        fd = os.open(
            parts[-1],
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o640,
            dir_fd=directory_fd,
        )
        try:
            _write_all(fd, raw)
            os.fsync(fd)
            status = os.fstat(fd)
            if (
                not stat.S_ISREG(status.st_mode)
                or status.st_nlink != 1
                or status.st_size != len(raw)
            ):
                raise ArchiveComposerError(
                    f"published file identity mismatch: {canonical}"
                )
            published = FrozenInput(
                root.joinpath(*canonical.split("/")),
                raw,
                status.st_dev,
                status.st_ino,
                status.st_mode,
                status.st_nlink,
                status.st_size,
                status.st_mtime_ns,
                status.st_ctime_ns,
            )
        finally:
            os.close(fd)
        os.fsync(directory_fd)
        os.fsync(root_fd)
    finally:
        os.close(directory_fd)
        os.close(root_fd)
    if published is None:
        raise ArchiveComposerError(f"archive publication did not produce: {canonical}")
    return published


def _unlink_leaf(directory: pathlib.Path, leaf: str) -> None:
    fd = os.open(
        directory, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    try:
        os.unlink(leaf, dir_fd=fd)
        os.fsync(fd)
    finally:
        os.close(fd)


def _re_read_published(
    root: pathlib.Path, published: Mapping[str, FrozenInput]
) -> dict[str, FrozenInput]:
    """Require every retained archive member, including the manifest, to persist."""

    result: dict[str, FrozenInput] = {}
    for relative_path, expected in sorted(published.items()):
        canonical = VERIFIER._archive_path(
            relative_path, "published archive re-read path"
        )
        try:
            observed = _read_frozen(
                root.joinpath(*canonical.split("/")),
                max(expected.size, 1),
                allow_empty=expected.size == 0,
            )
        except (ArchiveComposerError, OSError) as error:
            raise ArchiveComposerError(
                f"published archive re-read failed: {canonical}"
            ) from error
        if observed.raw != expected.raw or not _same_frozen_identity(expected, observed):
            raise ArchiveComposerError(
                f"published archive re-read differs: {canonical}"
            )
        result[canonical] = observed
    return result


def _catalog(catalog_id: str, entries: Sequence[Mapping[str, Any]], schema: Mapping[str, Any]) -> bytes:
    return VERIFIER.SEMANTIC.canonical_bytes(
        {
            "catalog_id": catalog_id,
            "entries": [dict(row) for row in entries],
            "schema": dict(schema),
            "version": {"major": 1, "minor": 0},
        }
    )


def _parse_maps(raw: bytes) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, bytes], list[dict[str, Any]]] = {}
    for index, line in enumerate(raw.splitlines()):
        match = _MAP_RE.fullmatch(line)
        if match is None:
            raise ArchiveComposerError(f"proc maps line {index + 1}: grammar mismatch")
        start, end, permissions, offset, major, minor, inode, path = match.groups()
        inode_value = int(inode)
        if inode_value == 0 or not path or not path.startswith(b"/") or path.endswith(b" (deleted)"):
            continue
        try:
            loaded_path = path.decode("utf-8", "strict")
        except UnicodeDecodeError as error:
            raise ArchiveComposerError(f"proc maps line {index + 1}: invalid UTF-8 path") from error
        device = os.makedev(int(major, 16), int(minor, 16))
        key = (device, inode_value, path)
        groups.setdefault(key, []).append(
            {
                "end_address": int(end, 16),
                "file_offset": int(offset, 16),
                "permissions": permissions.decode("ascii"),
                "start_address": int(start, 16),
            }
        )
    rows = []
    for (device, inode, path_raw), segments in groups.items():
        path = pathlib.Path(path_raw.decode("utf-8"))
        segments.sort(key=lambda row: row["start_address"])
        rows.append({"device": device, "inode": inode, "loaded_path": str(path), "segments": segments})
    return rows


def _build_row_by_runtime(
    build: Mapping[str, Any], mapped: Mapping[str, Any]
) -> tuple[Mapping[str, Any], bytes] | None:
    try:
        image = _read_frozen(pathlib.Path(mapped["loaded_path"]), MAX_BUILD_ARTIFACT_BYTES)
    except (ArchiveComposerError, OSError):
        return None
    if (image.device, image.inode) != (mapped["device"], mapped["inode"]):
        return None
    digest = _sha(image.raw)
    matches = [row for row in build["entries"] if row["sha256"] == digest and row["byte_count"] == image.size]
    if len(matches) != 1:
        return None
    return matches[0], image.raw


def _glibc_runtime_map_candidates(handoff: Any) -> list[dict[str, Any]]:
    """Select the one data-only libc map group retained by the handoff."""

    result = []
    for mapped in _parse_maps(handoff.maps_bytes):
        path = pathlib.Path(mapped["loaded_path"])
        if _GLIBC_RUNTIME_FILENAME_RE.fullmatch(path.name) is None:
            continue
        if not any(segment["permissions"][2] == "x" for segment in mapped["segments"]):
            continue
        result.append(mapped)
    return result


def _require_glibc_runtime_binding(
    handoff: Any,
    build: Mapping[str, Any],
    build_evidence: Mapping[str, Any],
    build_evidence_artifacts: Mapping[str, bytes],
) -> FrozenInput:
    """Bind the measured libc maps identity to the recorded dependency artifact."""

    dependencies = build.get("dependencies")
    if not isinstance(dependencies, list):
        raise ArchiveComposerError("build coverage glibc runtime dependency is unavailable")
    dependency_rows = [
        row
        for row in dependencies
        if isinstance(row, Mapping) and row.get("dependency_id") == "glibc_runtime"
    ]
    if len(dependency_rows) != 1:
        raise ArchiveComposerError(
            "build coverage requires exactly one glibc runtime dependency"
        )
    dependency = dependency_rows[0]
    if (
        dependency.get("evidence_state_id") != 1
        or dependency.get("name") != "libc.so.6"
        or not isinstance(dependency.get("sha256"), str)
    ):
        raise ArchiveComposerError("build coverage glibc runtime dependency is invalid")

    toolchain = build_evidence.get("toolchain")
    if not isinstance(toolchain, Mapping) or not isinstance(toolchain.get("libc_path"), str):
        raise ArchiveComposerError("build evidence libc artifact path is unavailable")
    artifact_path = VERIFIER._archive_path(
        toolchain["libc_path"], "build evidence libc artifact path"
    )
    recorded = build_evidence_artifacts.get(artifact_path)
    if recorded is None:
        raise ArchiveComposerError("build evidence libc artifact is unavailable")
    if _sha(recorded) != dependency["sha256"]:
        raise ArchiveComposerError(
            "build evidence libc artifact digest differs from glibc runtime dependency"
        )
    try:
        _machine, sections, bodies = BUILD_EVIDENCE._parse_sections(recorded)
        _needed, recorded_soname = BUILD_EVIDENCE._parse_dynamic(sections, bodies)
    except BUILD_EVIDENCE.BuildEvidenceError as error:
        raise ArchiveComposerError("build evidence libc artifact is not a valid ELF") from error
    if recorded_soname != dependency["name"]:
        raise ArchiveComposerError(
            "build evidence libc artifact SONAME differs from glibc runtime dependency"
        )

    candidates = _glibc_runtime_map_candidates(handoff)
    if len(candidates) != 1:
        raise ArchiveComposerError(
            "authenticated maps require exactly one executable glibc runtime identity"
        )
    mapped = candidates[0]
    path = pathlib.Path(mapped["loaded_path"])
    try:
        image = _read_frozen(
            path,
            MAX_BUILD_ARTIFACT_BYTES,
            require_single_link=False,
        )
    except (ArchiveComposerError, OSError) as error:
        raise ArchiveComposerError(
            f"authenticated glibc runtime mapped file is unavailable: {path}"
        ) from error
    if (image.device, image.inode) != (mapped["device"], mapped["inode"]):
        raise ArchiveComposerError(
            "authenticated glibc runtime mapped file identity differs from maps"
        )
    if image.raw != recorded or _sha(image.raw) != dependency["sha256"]:
        raise ArchiveComposerError(
            "authenticated glibc runtime mapped bytes differ from build evidence"
        )
    try:
        _machine, sections, bodies = BUILD_EVIDENCE._parse_sections(image.raw)
        _needed, mapped_soname = BUILD_EVIDENCE._parse_dynamic(sections, bodies)
    except BUILD_EVIDENCE.BuildEvidenceError as error:
        raise ArchiveComposerError(
            "authenticated glibc runtime mapped file is not a valid ELF"
        ) from error
    if mapped_soname != dependency["name"]:
        raise ArchiveComposerError(
            "authenticated glibc runtime mapped SONAME differs from build evidence"
        )
    return image


def _runtime_objects(
    handoff: Any,
    build: Mapping[str, Any],
) -> tuple[bytes, list[dict[str, Any]], list[dict[str, Any]]]:
    mapped_rows = _parse_maps(handoff.maps_bytes)
    mapped_matches = [
        (mapped, _build_row_by_runtime(build, mapped)) for mapped in mapped_rows
    ]
    primary_id = build["build_identity"]["primary_logical_elf_id"]
    primary_paths = [
        pathlib.Path(mapped["loaded_path"])
        for mapped, matched in mapped_matches
        if matched is not None and matched[0]["logical_id"] == primary_id
    ]
    if len(primary_paths) != 1:
        raise ArchiveComposerError(
            "primary logical ELF must have exactly one runtime mapping"
        )
    primary_directory = primary_paths[0].parent
    for mapped, matched in mapped_matches:
        if matched is not None:
            continue
        path = pathlib.Path(mapped["loaded_path"])
        try:
            path.relative_to(primary_directory)
        except ValueError:
            continue
        try:
            image = _read_frozen(path, MAX_BUILD_ARTIFACT_BYTES)
        except (ArchiveComposerError, OSError) as error:
            raise ArchiveComposerError(
                f"mapped file under primary build directory is unavailable: {path}"
            ) from error
        if image.raw.startswith(b"\x7fELF"):
            raise ArchiveComposerError(
                f"unregistered mapped ELF under primary build directory: {path}"
            )

    module_entries: list[dict[str, Any]] = []
    population: list[dict[str, Any]] = []
    represented: set[str] = set()
    module_id = 1
    for mapped, matched in mapped_matches:
        if matched is None:
            continue
        build_row, _image = matched
        logical_id = build_row["logical_id"]
        if logical_id in represented:
            raise ArchiveComposerError(f"multiple mappings for build logical ID {logical_id!r}")
        represented.add(logical_id)
        module = {
            "build_id": build_row["build_id"],
            "build_logical_id": logical_id,
            "byte_count": build_row["byte_count"],
            "device": mapped["device"],
            "inode": mapped["inode"],
            "load_state_id": 1,
            "loaded_path": mapped["loaded_path"],
            "logical_id": logical_id,
            "module_generation": handoff.opening.process_generation,
            "module_id": module_id,
            "module_map_sha256": "0" * 64,
            "namespace_id": 0,
            "process_generation": handoff.opening.process_generation,
            "segments": mapped["segments"],
            "sha256": build_row["sha256"],
        }
        module["module_map_sha256"] = VERIFIER.RUNTIME.module_map_sha256(module)
        runtime_identity = VERIFIER.COVERAGE._expected_runtime_identity(
            build_row, module["module_map_sha256"]
        )
        population.append(
            {
                "admission_state_id": build_row["admission_state_id"],
                "build_logical_id": logical_id,
                "classifications": [
                    {
                        "classification_id": origin["classification_id"],
                        "origin_id": origin["origin_id"],
                    }
                    for origin in build_row["symbol_origins"]
                ],
                "configured": True,
                "load_generation": handoff.opening.process_generation,
                "load_state_id": 1,
                "loaded_path": mapped["loaded_path"],
                "logical_id": logical_id,
                "observed": True,
                "runtime_identity": runtime_identity,
            }
        )
        module_entries.append(module)
        module_id += 1
    for row in build["entries"]:
        if row["logical_id"] not in represented:
            population.append(
                {
                    "admission_state_id": row["admission_state_id"],
                    "build_logical_id": row["logical_id"],
                    "classifications": [
                        {
                            "classification_id": origin["classification_id"],
                            "origin_id": origin["origin_id"],
                        }
                        for origin in row["symbol_origins"]
                    ],
                    "configured": True,
                    "load_generation": None,
                    "load_state_id": 10,
                    "loaded_path": None,
                    "logical_id": row["logical_id"],
                    "observed": False,
                    "runtime_identity": None,
                }
            )
    population.sort(key=lambda row: row["logical_id"])
    module_entries.sort(key=lambda row: row["logical_id"])
    for index, row in enumerate(module_entries, 1):
        row["module_id"] = index
    module_raw = _catalog("oai_memprof_module", module_entries, VERIFIER.RUNTIME.MODULE_SCHEMA)
    return module_raw, population, module_entries


def _clock_row(handoff: Any) -> dict[str, Any]:
    samples = (handoff.opening_sample, handoff.writer.final_sample)
    rows = [
        {
            "counter": sample.counter,
            "monotonic_raw_after_ns": sample.monotonic_raw_after_ns,
            "monotonic_raw_before_ns": sample.monotonic_raw_before_ns,
            "realtime_unix_ns": sample.realtime_unix_ns,
            "sample_ordinal": index,
        }
        for index, sample in enumerate(samples, 1)
    ]
    numerator = handoff.opening.counter_frequency_numerator
    denominator = handoff.opening.counter_frequency_denominator
    predicted = lambda counter: handoff.opening.start_monotonic_raw_ns + (
        (counter - handoff.opening.start_counter) * 1_000_000_000 * denominator
    ) // numerator
    observed_max = max(
        max(sample.monotonic_raw_before_ns - predicted(sample.counter),
            predicted(sample.counter) - sample.monotonic_raw_after_ns,
            0)
        for sample in samples
    )
    realtime_discontinuity = any(
        abs(
            (sample.realtime_unix_ns - samples[0].realtime_unix_ns)
            - (
                (sample.monotonic_raw_before_ns + sample.monotonic_raw_after_ns) // 2
                - handoff.opening.start_monotonic_raw_ns
            )
        )
        > handoff.opening.calibration_error_bound_ns
        for sample in samples
    )
    return {
        "acquisition_source_id": handoff.writer.clock_info.acquisition_source_id,
        "acquisition_status_id": 1,
        "architecture_id": handoff.writer.clock_info.architecture_id,
        "calibration_error_bound_ns": handoff.opening.calibration_error_bound_ns,
        "calibration_kind": handoff.opening.calibration_kind,
        "calibration_span_ns": handoff.opening.calibration_span_ns,
        "clock_kind": handoff.opening.clock_kind,
        "counter_frequency_denominator": denominator,
        "counter_frequency_numerator": numerator,
        "counter_invalid_observed": (
            handoff.writer.clock_status != 0
            or any(thread.counter_invalids for thread in handoff.threads)
        ),
        "counter_stability_status_id": 1,
        "observed_max_error_ns": observed_max,
        "process_generation": handoff.opening.process_generation,
        "realtime_discontinuity_observed": realtime_discontinuity,
        "samples": rows,
        "start_counter": handoff.opening.start_counter,
        "start_monotonic_raw_ns": handoff.opening.start_monotonic_raw_ns,
        "start_realtime_unix_ns": handoff.opening.start_realtime_unix_ns,
    }


def _clock_object(handoff: Any) -> bytes:
    return _catalog(
        "oai_memprof_clock", [_clock_row(handoff)], VERIFIER.RUNTIME.CLOCK_SCHEMA
    )


def _diagnostic_rows(handoff: Any) -> list[dict[str, Any]]:
    all_reason_ids = (1, 16, 17, 18, 32, 48, 49, 50, 51, 64)
    mode_row = VERIFIER.DIAGNOSTICS.MODE_ROWS.get(
        handoff.writer.runtime_snapshot.mode_id
    )
    if mode_row is None:
        raise ArchiveComposerError("diagnostic mode is not registered")
    producer_reason_ids = set(mode_row[4])
    rows = []
    for thread in handoff.threads:
        for index, reason_id in enumerate(all_reason_ids):
            if reason_id not in producer_reason_ids:
                continue
            value = thread.diagnostic_values[index]
            rows.append(
                {
                    "counter_scope_id": thread.thread_index,
                    "counter_scope_kind": 1,
                    "process_generation": handoff.opening.process_generation,
                    "reason_id": reason_id,
                    "saturated": bool(thread.diagnostic_saturated_mask & (1 << index)),
                    "value": value,
                }
            )
    registration_values = (
        handoff.unregistered_active_thread_failures,
        handoff.writer.runtime_snapshot.registration_capacity_failures,
    )
    for index, reason_id in enumerate((2, 3)):
        rows.append(
            {
                "counter_scope_id": 1,
                "counter_scope_kind": 2,
                "process_generation": handoff.opening.process_generation,
                "reason_id": reason_id,
                "saturated": bool(handoff.registration_diagnostic_saturated_mask & (1 << index)),
                "value": registration_values[index],
            }
        )
    rows.extend(
        (
            {
                "counter_scope_id": 1,
                "counter_scope_kind": 3,
                "process_generation": handoff.opening.process_generation,
                "reason_id": 80,
                "saturated": handoff.writer_io_or_finalization_failures == (1 << 64) - 1,
                "value": handoff.writer_io_or_finalization_failures,
            },
            {
                "counter_scope_id": 1,
                "counter_scope_kind": 4,
                "process_generation": handoff.opening.process_generation,
                "reason_id": 96,
                "saturated": False,
                "value": handoff.diagnostic_saturation_transitions,
            },
        )
    )
    return sorted(
        rows,
        key=lambda row: (
            row["process_generation"],
            row["counter_scope_kind"],
            row["counter_scope_id"],
            row["reason_id"],
        ),
    )


def _pre_footer(
    handoff: Any, diagnostic_projection: Mapping[str, Any]
) -> dict[str, Any]:
    writer = handoff.writer
    try:
        lifecycle_state, reason_code, payload_writer_state, terminal_outcome = (
            VERIFIER._handoff_terminal_projection(writer)
        )
    except VERIFIER.ArchiveVerificationError as error:
        raise ArchiveComposerError(str(error)) from error
    if diagnostic_projection["population_partial"]:
        raise ArchiveComposerError(
            "authenticated terminal requires the exact diagnostic counter population"
        )
    clock_row = _clock_row(handoff)
    terminal_flags = 0x017F if payload_writer_state == 6 else 0x01FF
    if terminal_outcome == "complete":
        terminal_flags |= (1 << 9) | (1 << 10) | (1 << 11)
    if diagnostic_projection["aggregate_saturated"]:
        terminal_flags |= 1 << 12
    if clock_row["realtime_discontinuity_observed"]:
        terminal_flags |= 1 << 14
    if clock_row["counter_invalid_observed"]:
        terminal_flags |= 1 << 15
    return {
        "active_generation": handoff.opening.process_generation,
        "active_start_counter": handoff.opening.start_counter,
        "active_start_monotonic_raw_ns": handoff.opening.start_monotonic_raw_ns,
        "cutoff_after_counter": writer.seal_after_sample.counter,
        "cutoff_before_counter": writer.seal_before_sample.counter,
        "diagnostic_population_partial": False,
        "final_counter": writer.final_sample.counter,
        "final_monotonic_raw_ns": (
            writer.final_sample.monotonic_raw_before_ns
            + writer.final_sample.monotonic_raw_after_ns
        )
        // 2,
        "final_realtime_unix_ns": writer.final_sample.realtime_unix_ns,
        "finalization_stage": 6,
        "lifecycle_state": lifecycle_state,
        "payload_writer_state": payload_writer_state,
        "process_generation": handoff.opening.process_generation,
        "quiescence_complete_counter": writer.drain_complete_sample.counter,
        "reason_code": reason_code,
        "schema": VERIFIER.STATUS.SCHEMA_PRE_FOOTER,
        "scope_kind": handoff.opening.scope_kind,
        "terminal_flags": terminal_flags,
    }


def _count(kind: int, value: Mapping[str, Any]) -> int:
    if kind in tuple(range(1, 9)) + (11,):
        return len(value["entries"])
    if kind == 9:
        return len(value["module_population"])
    if kind in (10, 12):
        return 1
    raise ArchiveComposerError(f"unsupported object kind {kind}")


def _event_totals(stream_prefix: bytes) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    records = []
    offset = VERIFIER.WIRE.OPENING_HEADER_BYTES
    sequence = 0
    while offset < len(stream_prefix):
        if len(stream_prefix) - offset < VERIFIER.WIRE.CHUNK_HEADER_BYTES:
            raise ArchiveComposerError("truncated pre-footer chunk header")
        chunk_header = VERIFIER.WIRE.decode_chunk_header(
            stream_prefix[offset : offset + VERIFIER.WIRE.CHUNK_HEADER_BYTES]
        )
        size = (
            VERIFIER.WIRE.CHUNK_HEADER_BYTES
            + chunk_header.record_count * VERIFIER.WIRE.EVENT_RECORD_BYTES
        )
        if size > len(stream_prefix) - offset:
            raise ArchiveComposerError("truncated pre-footer chunk body")
        chunk = VERIFIER.WIRE.decode_chunk(
            stream_prefix[offset : offset + size], expected_sequence=sequence
        )
        records.extend(chunk.records)
        offset += size
        sequence += 1
    if offset != len(stream_prefix):
        raise ArchiveComposerError("pre-footer chunk boundary mismatch")
    totals: dict[tuple[int, int], int] = {}
    for record in records:
        key = (record.event_kind, record.api_id)
        totals[key] = totals.get(key, 0) + 1
    entries = tuple(
        VERIFIER.WIRE.EventTotalEntry(event_kind, api_id, count)
        for (event_kind, api_id), count in sorted(totals.items())
    )
    return tuple(records), entries


def _trailer_bytes(handoff: Any, objects: Mapping[str, bytes], stream_prefix: bytes) -> bytes:
    diagnostics = VERIFIER.SEMANTIC.parse_canonical(objects[VERIFIER.DIAGNOSTICS.INSTANCE_PATH])
    diagnostic_entries = tuple(
        VERIFIER.WIRE.DiagnosticTotalEntry(
            row["reason_id"],
            row["class_flags"],
            row["summary_flags"],
            row["saturating_total"],
            row["nonzero_counter_instances"],
            row["saturated_counter_instances"],
        )
        for row in diagnostics["reason_totals"]
    )
    records, event_entries = _event_totals(stream_prefix)
    object_entries = []
    for kind in range(1, 13):
        path, flags = VERIFIER.EXTERNAL_BY_KIND[kind]
        raw = objects[path]
        value = VERIFIER.SEMANTIC.parse_canonical(raw)
        object_entries.append(
            VERIFIER.WIRE.ObjectBindingEntry(
                kind, 1, flags, 1, _count(kind, value), len(raw), hashlib.sha256(raw).digest()
            )
        )
    event_offset = VERIFIER.WIRE.TRAILER_HEADER_BYTES
    diagnostic_offset = event_offset + len(event_entries) * 32
    object_offset = diagnostic_offset + len(diagnostic_entries) * 32
    body_bytes = object_offset + len(object_entries) * 64
    status = VERIFIER.STATUS.parse_canonical(objects[VERIFIER.STATUS.PRE_FOOTER_PATH])
    projection = VERIFIER.DIAGNOSTICS.validate_diagnostics_bytes(
        objects[VERIFIER.DIAGNOSTICS.INSTANCE_PATH],
        definition_sha256=VERIFIER.ACCEPTED_MEMBER_SHA256[8],
        expected_mode_id=handoff.writer.runtime_snapshot.mode_id,
        expected_process_generation=handoff.opening.process_generation,
        ready_thread_indices=tuple(thread.thread_index for thread in handoff.threads),
        producer_population_complete=True,
    )[1]
    header = VERIFIER.WIRE.TrailerHeader(
        trailer_body_bytes=body_bytes,
        process_generation=handoff.opening.process_generation,
        scope_kind=handoff.opening.scope_kind,
        lifecycle_state=status["lifecycle_state"],
        payload_writer_state=status["payload_writer_state"],
        finalization_stage=status["finalization_stage"],
        terminal_flags=status["terminal_flags"],
        chunk_count=handoff.writer.chunk_count,
        record_count=handoff.writer.record_count,
        payload_bytes=handoff.writer.payload_bytes,
        first_chunk_offset=512,
        chunks_end_offset=handoff.writer.stream_bytes,
        active_generation=handoff.opening.process_generation,
        active_start_counter=handoff.opening.start_counter,
        cutoff_before_counter=status["cutoff_before_counter"],
        cutoff_after_counter=status["cutoff_after_counter"],
        quiescence_complete_counter=status["quiescence_complete_counter"],
        final_counter=status["final_counter"],
        active_start_monotonic_raw_ns=handoff.opening.start_monotonic_raw_ns,
        final_monotonic_raw_ns=status["final_monotonic_raw_ns"],
        final_realtime_unix_ns=status["final_realtime_unix_ns"],
        event_entry_count=len(event_entries),
        diagnostic_entry_count=len(diagnostic_entries),
        object_entry_count=12,
        event_table_offset=event_offset,
        diagnostic_table_offset=diagnostic_offset,
        object_table_offset=object_offset,
        terminal_reason_code=status["reason_code"],
        diagnostic_loss_sum=projection["diagnostic_loss_sum"],
        diagnostic_bypass_sum=projection["diagnostic_bypass_sum"],
        saturated_counter_instances=projection["saturated_counter_instances"],
    )
    if len(records) != handoff.writer.record_count:
        raise ArchiveComposerError("pre-footer record count mismatch")
    return VERIFIER.WIRE.encode_trailer_body(
        VERIFIER.WIRE.TrailerBody(header, event_entries, diagnostic_entries, tuple(object_entries))
    )


def _receipt(stream: bytes) -> bytes:
    decoded = VERIFIER.WIRE.decode_container(stream)
    verifier_raw = VERIFIER.accepted_verifier_definition_bytes()
    footer_preimage = stream[-VERIFIER.WIRE.FOOTER_BYTES : -32]
    return VERIFIER.STATUS.canonical_bytes(
        {
            "appender_close": "success",
            "exact_eof": True,
            "footer_preimage_sha256": _sha(footer_preimage),
            "opening_header_sha256": decoded.footer.opening_header_sha256.hex(),
            "physical_bytes": len(stream),
            "prefix_sha256": decoded.footer.prefix_sha256.hex(),
            "schema": VERIFIER.STATUS.SCHEMA_POST_CLOSE,
            "stream_path": STREAM_ARCHIVE_PATH,
            "trailer_body_sha256": decoded.footer.trailer_body_sha256.hex(),
            "verifier_definition_sha256": _sha(verifier_raw),
            "whole_stream_sha256": _sha(stream),
        }
    )


def _manifest(
    objects: Mapping[str, bytes],
    stream: bytes,
    handoff: bytes,
    receipt: bytes,
    trusted_release_artifacts: Mapping[str, bytes],
    build_evidence: bytes,
    build_evidence_artifacts: Mapping[str, bytes],
) -> bytes:
    rows: dict[str, tuple[int, str]] = {}

    def add(path: str, raw: bytes) -> None:
        if path in rows:
            raise ArchiveComposerError(f"manifest role path duplicated: {path}")
        rows[path] = (len(raw), _sha(raw))

    for path, raw in objects.items():
        add(path, raw)
    for path, raw in trusted_release_artifacts.items():
        add(path, raw)
    for path, raw in build_evidence_artifacts.items():
        add(path, raw)
    add(VERIFIER.STATUS.POST_CLOSE_PATH, receipt)
    add(PROCESS_HANDOFF_ARCHIVE_PATH, handoff)
    add(STREAM_ARCHIVE_PATH, stream)
    add(BUILD_EVIDENCE_INPUT_PATH, build_evidence)
    return VERIFIER.STATUS.canonical_bytes(
        {
            "entries": [
                {"bytes": size, "path": path, "sha256": digest}
                for path, (size, digest) in sorted(rows.items())
            ],
            "schema": VERIFIER.STATUS.SCHEMA_MANIFEST,
        }
    )


def _read_build_evidence_artifacts(
    evidence_value: Any, build_evidence_root: pathlib.Path
) -> dict[str, bytes]:
    """Preflight and bounded-read the complete build-artifact population.

    The authoritative build-evidence validator still runs after this helper.
    This earlier two-phase pass only ensures that malformed evidence cannot
    cause an unbounded number or aggregate size of artifact reads before that
    semantic rejection.
    """

    if type(evidence_value) is not dict:
        raise ArchiveComposerError("build evidence object required")
    evidence_rows = evidence_value.get("entries")
    if type(evidence_rows) is not list or not evidence_rows:
        raise ArchiveComposerError("build evidence requires a nonempty artifact manifest")
    if len(evidence_rows) > MAX_BUILD_ARTIFACT_ENTRIES:
        raise ArchiveComposerError("build evidence artifact-entry limit exceeded")

    plan: list[tuple[str, int, str]] = []
    seen_paths: set[str] = set()
    declared_total = 0
    for index, row in enumerate(evidence_rows):
        if type(row) is not dict or set(row) != {"bytes", "path", "sha256"}:
            raise ArchiveComposerError(
                f"build evidence entry {index}: exact members required"
            )
        try:
            relative = VERIFIER._archive_path(
                row["path"], f"build evidence entry {index} path"
            )
        except VERIFIER.ArchiveVerificationError as error:
            raise ArchiveComposerError(str(error)) from error
        if not relative.startswith("input/build-evidence/"):
            raise ArchiveComposerError(
                f"build evidence entry {index}: owned path prefix required"
            )
        declared_bytes = row["bytes"]
        if (
            type(declared_bytes) is not int
            or declared_bytes < 0
            or declared_bytes > MAX_BUILD_ARTIFACT_BYTES
        ):
            raise ArchiveComposerError(
                f"build evidence entry {index}: bounded nonnegative bytes required"
            )
        declared_sha256 = row["sha256"]
        if (
            not isinstance(declared_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", declared_sha256, re.ASCII) is None
        ):
            raise ArchiveComposerError(
                f"build evidence entry {index}: lowercase SHA-256 required"
            )
        if relative in seen_paths:
            raise ArchiveComposerError(
                f"build evidence entry {index}: duplicate artifact path"
            )
        declared_total += declared_bytes
        if declared_total > MAX_BUILD_ARTIFACT_TOTAL_BYTES:
            raise ArchiveComposerError(
                "build evidence aggregate declared artifact bytes exceeded"
            )
        seen_paths.add(relative)
        plan.append((relative, declared_bytes, declared_sha256))

    artifacts: dict[str, bytes] = {}
    actual_total = 0
    for index, (relative, declared_bytes, declared_sha256) in enumerate(plan):
        remaining_budget = MAX_BUILD_ARTIFACT_TOTAL_BYTES - actual_total
        artifact = _read_frozen(
            build_evidence_root.joinpath(*relative.split("/")),
            min(declared_bytes, remaining_budget),
            allow_empty=True,
        )
        actual_total += artifact.size
        if actual_total > MAX_BUILD_ARTIFACT_TOTAL_BYTES:
            raise ArchiveComposerError(
                "build evidence aggregate actual artifact bytes exceeded"
            )
        if artifact.size != declared_bytes or _sha(artifact.raw) != declared_sha256:
            raise ArchiveComposerError(
                f"build evidence entry {index}: path/size/digest mismatch"
            )
        artifacts[relative] = artifact.raw
    return artifacts


def compose(
    archive_directory: pathlib.Path,
    *,
    build_coverage_path: pathlib.Path,
    build_evidence_path: pathlib.Path,
    build_evidence_root: pathlib.Path,
    append_executable: pathlib.Path,
    trusted_release_authority_path: pathlib.Path,
    trusted_release_source_root: pathlib.Path,
    trusted_release_authority_sha256: str,
) -> CompositionResult:
    archive_directory = archive_directory.resolve(strict=True)
    authority_file, trusted_release_source_bytes = _read_trusted_release_inputs(
        trusted_release_authority_path,
        trusted_release_source_root,
    )
    stream_path = archive_directory / STREAM_ARCHIVE_PATH
    handoff_path = archive_directory / PROCESS_HANDOFF_ARCHIVE_PATH
    handoff_file = _read_frozen(handoff_path, MAX_HANDOFF_BYTES)
    stream_file = _read_frozen(stream_path, MAX_STREAM_BYTES)
    build_file = _read_frozen(
        build_coverage_path.resolve(strict=True), MAX_BUILD_COVERAGE_BYTES
    )
    build_evidence_file = _read_frozen(
        build_evidence_path.resolve(strict=True), MAX_BUILD_EVIDENCE_BYTES
    )
    build_evidence_root = build_evidence_root.resolve(strict=True)
    try:
        evidence_value = BUILD_EVIDENCE.COVERAGE.parse_canonical(
            build_evidence_file.raw
        )
    except BUILD_EVIDENCE.COVERAGE.CoverageError as error:
        raise ArchiveComposerError(str(error)) from error
    build_evidence_artifacts = _read_build_evidence_artifacts(
        evidence_value, build_evidence_root
    )
    handoff = HANDOFF.decode_process_handoff(handoff_file.raw)
    try:
        _lifecycle, _reason, _payload, expected_terminal_outcome = (
            VERIFIER._handoff_terminal_projection(handoff.writer)
        )
    except VERIFIER.ArchiveVerificationError as error:
        raise ArchiveComposerError(str(error)) from error
    if (stream_file.device, stream_file.inode, stream_file.size) != (
        handoff.writer.file_device,
        handoff.writer.file_inode,
        handoff.writer.stream_bytes,
    ):
        raise ArchiveComposerError("pre-footer file identity differs from authenticated handoff")
    if stream_file.raw[:512] != handoff.opening_raw:
        raise ArchiveComposerError("pre-footer opening differs from authenticated handoff")
    build = VERIFIER.COVERAGE.validate_build_coverage_bytes(
        build_file.raw, api_definition_sha256=VERIFIER.ACCEPTED_MEMBER_SHA256[4]
    )
    if build["evidence_origin_id"] != 1 or build["verdict_id"] != 1:
        raise ArchiveComposerError("measured complete build coverage required")
    try:
        derived_build = BUILD_EVIDENCE.validate_build_evidence_bytes(
            build_evidence_file.raw,
            build_evidence_artifacts,
            build_file.raw,
            api_definition_sha256=VERIFIER.ACCEPTED_MEMBER_SHA256[4],
        )
    except BUILD_EVIDENCE.BuildEvidenceError as error:
        raise ArchiveComposerError(str(error)) from error
    if derived_build != build:
        raise ArchiveComposerError("derived build coverage differs after validation")
    _require_glibc_runtime_binding(
        handoff,
        build,
        evidence_value,
        build_evidence_artifacts,
    )
    primary = next(
        row
        for row in build["entries"]
        if row["logical_id"] == build["build_identity"]["primary_logical_elf_id"]
    )
    if (
        primary["sha256"] != handoff.opening.primary_binary_sha256.hex()
        or _sha(bytes.fromhex(primary["build_id"]))
        != handoff.opening.primary_build_id_sha256.hex()
    ):
        raise ArchiveComposerError("build coverage differs from opening primary ELF")

    config = VERIFIER.CONFIG.validate_effective_configuration_bytes(
        handoff.bootstrap_bytes
    )
    if config["output_directory"] != str(archive_directory):
        raise ArchiveComposerError(
            "effective configuration output directory differs from archive directory"
        )
    selection_values = {
        row["key"]: row["value"] for row in config["selection_values"]
    }
    if selection_values.get("build_evidence_sha256") != _sha(
        build_evidence_file.raw
    ):
        raise ArchiveComposerError(
            "effective configuration does not bind the exact build evidence"
        )
    try:
        trusted_release_artifacts = VERIFIER.validate_trusted_release_authority(
            authority_file.raw,
            trusted_release_authority_sha256,
            trusted_release_source_bytes,
            build=build,
        )
    except VERIFIER.ArchiveVerificationError as error:
        raise ArchiveComposerError(str(error)) from error
    if selection_values.get("trusted_release_authority_sha256") != (
        trusted_release_authority_sha256
    ):
        raise ArchiveComposerError(
            "effective configuration does not bind the external trusted-release authority"
        )
    module_raw, population, _modules = _runtime_objects(handoff, build)
    if any(row["load_state_id"] != 1 for row in population):
        raise ArchiveComposerError(
            "v1 COMPLETE composition requires every measured build row loaded and observed"
        )
    run = {
        "build_coverage": {
            "object_kind": 8,
            "path": VERIFIER.COVERAGE.BUILD_COVERAGE_ARCHIVE_PATH,
            "sha256": _sha(build_file.raw),
        },
        "catalog_id": "oai_memprof_run_coverage",
        "configuration_instance_sha256": _sha(handoff.bootstrap_bytes),
        "eligible_exact_domain": True,
        "evidence_origin_id": 1,
        "failure_ids": [],
        "module_population": population,
        "policy": {
            "object_type": 9,
            "path": VERIFIER.COVERAGE.POLICY_ARCHIVE_PATH,
            "sha256": VERIFIER.COVERAGE.POLICY_SHA256,
        },
        "process_generation": handoff.opening.process_generation,
        "process_uuid": str(uuid.UUID(bytes=handoff.opening.process_uuid)),
        "role_id": handoff.opening.role_kind,
        "run_uuid": str(uuid.UUID(bytes=handoff.opening.run_uuid)),
        "schema": {
            "object_type": 10,
            "path": VERIFIER.COVERAGE.INSTANCE_SCHEMA_ARCHIVE_PATH,
            "sha256": VERIFIER.COVERAGE.INSTANCE_SCHEMA_SHA256,
        },
        "snapshot_state_id": 1,
        "verdict_id": 2,
        "version": {"major": 1, "minor": 0},
    }
    run_raw = VERIFIER.COVERAGE.canonical_bytes(run)
    VERIFIER.COVERAGE.validate_run_coverage_bytes(
        run_raw,
        build_coverage=build,
        api_definition_sha256=VERIFIER.ACCEPTED_MEMBER_SHA256[4],
        expected_configuration_instance_sha256=_sha(handoff.bootstrap_bytes),
    )
    resolution = VERIFIER.COVERAGE.resolve_run_realloc_zero_policy(
        run,
        build_coverage=build,
        api_definition_sha256=VERIFIER.ACCEPTED_MEMBER_SHA256[4],
        expected_configuration_instance_sha256=_sha(handoff.bootstrap_bytes),
    )
    try:
        VERIFIER.validate_handoff_runtime_configuration(
            handoff,
            config,
            resolution.policy_id,
        )
    except VERIFIER.ArchiveVerificationError as error:
        raise ArchiveComposerError(str(error)) from error
    members, bundle = VERIFIER._accepted_static_members()
    empty_context = _catalog(
        "oai_memprof_context",
        [],
        {"object_type": 6, "path": "definition/context-schema-v1.json", "sha256": VERIFIER.ACCEPTED_MEMBER_SHA256[6]},
    )
    empty_callsite = _catalog(
        "oai_memprof_callsite",
        [],
        {"object_type": 5, "path": "definition/callsite-rule-v1.json", "sha256": VERIFIER.ACCEPTED_MEMBER_SHA256[5]},
    )
    thread_raw = _catalog(
        "oai_memprof_thread",
        [
            {
                "process_generation": thread.process_generation,
                "registration_ordinal": thread.registration_ordinal,
                "thread_index": thread.thread_index,
            }
            for thread in handoff.threads
        ],
        VERIFIER.RUNTIME.THREAD_SCHEMA,
    )
    diagnostic_raw = VERIFIER.DIAGNOSTICS.make_diagnostics_bytes(
        definition_sha256=VERIFIER.ACCEPTED_MEMBER_SHA256[8],
        mode_id=handoff.writer.runtime_snapshot.mode_id,
        process_generation=handoff.opening.process_generation,
        counter_rows=_diagnostic_rows(handoff),
        ready_thread_indices=tuple(thread.thread_index for thread in handoff.threads),
        producer_population_complete=True,
    )
    projection = VERIFIER.DIAGNOSTICS.validate_diagnostics_bytes(
        diagnostic_raw,
        definition_sha256=VERIFIER.ACCEPTED_MEMBER_SHA256[8],
        expected_mode_id=handoff.writer.runtime_snapshot.mode_id,
        expected_process_generation=handoff.opening.process_generation,
        ready_thread_indices=tuple(thread.thread_index for thread in handoff.threads),
        producer_population_complete=True,
    )[1]
    pre_footer_raw = VERIFIER.STATUS.canonical_bytes(
        _pre_footer(handoff, projection)
    )
    objects = {
        "catalog/schema-bundle.json": bundle,
        "catalog/api.json": members[4],
        "catalog/context.json": empty_context,
        "catalog/callsite.json": empty_callsite,
        "catalog/thread.json": thread_raw,
        "catalog/module.json": module_raw,
        "catalog/clock.json": _clock_object(handoff),
        "catalog/build-coverage.json": build_file.raw,
        "catalog/run-coverage.json": run_raw,
        "metadata/effective-config.json": handoff.bootstrap_bytes,
        "status/diagnostics.json": diagnostic_raw,
        "status/pre-footer-status.json": pre_footer_raw,
    }
    trailer_raw = _trailer_bytes(handoff, objects, stream_file.raw)
    process_directory = stream_path.parent
    trailer_leaf = "trailer.bin"
    trailer_path = process_directory / trailer_leaf
    _publish_once(archive_directory, f"streams/{trailer_leaf}", trailer_raw)
    try:
        completed = subprocess.run(
            [
                str(append_executable.resolve(strict=True)),
                str(process_directory),
                stream_path.name,
                handoff_path.name,
                trailer_leaf,
                _sha(handoff_file.raw),
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        if completed.returncode != 0:
            raise ArchiveComposerError(
                f"C append utility failed {completed.returncode}: {completed.stderr.decode('utf-8', 'replace')}"
            )
    finally:
        _unlink_leaf(process_directory, trailer_leaf)
    handoff_after_append = _read_frozen(handoff_path, MAX_HANDOFF_BYTES)
    if (
        handoff_after_append.raw != handoff_file.raw
        or not _same_frozen_identity(handoff_file, handoff_after_append)
    ):
        raise ArchiveComposerError(
            "persisted handoff differs after C append authentication"
        )
    complete_stream_file = _read_frozen(stream_path, MAX_STREAM_BYTES)
    complete_stream = complete_stream_file.raw
    receipt = _receipt(complete_stream)
    producer_definition = trusted_release_artifacts[PRODUCER_DEFINITION_PATH]
    handoff_decoder_definition = trusted_release_artifacts[
        HANDOFF_DECODER_DEFINITION_PATH
    ]
    build_evidence_definition = trusted_release_artifacts[
        BUILD_EVIDENCE_DEFINITION_PATH
    ]
    verifier_definition = trusted_release_artifacts[
        VERIFIER.ACCEPTED_VERIFIER_DEFINITION_PATH
    ]
    manifest = _manifest(
        objects,
        complete_stream,
        handoff_file.raw,
        receipt,
        trusted_release_artifacts,
        build_evidence_file.raw,
        build_evidence_artifacts,
    )
    result = VERIFIER.verify_archive_candidate(
        complete_stream,
        objects,
        stream_path=STREAM_ARCHIVE_PATH,
        verifier_definition_path=VERIFIER.ACCEPTED_VERIFIER_DEFINITION_PATH,
        verifier_definition_bytes=verifier_definition,
        process_handoff_bytes=handoff_file.raw,
        producer_definition_path=PRODUCER_DEFINITION_PATH,
        producer_definition_bytes=producer_definition,
        handoff_decoder_definition_path=HANDOFF_DECODER_DEFINITION_PATH,
        handoff_decoder_definition_bytes=handoff_decoder_definition,
        build_evidence_definition_path=BUILD_EVIDENCE_DEFINITION_PATH,
        build_evidence_definition_bytes=build_evidence_definition,
        build_evidence_bytes=build_evidence_file.raw,
        build_evidence_artifact_bytes=build_evidence_artifacts,
        trusted_release_authority_bytes=authority_file.raw,
        trusted_release_authority_sha256=trusted_release_authority_sha256,
        trusted_release_source_bytes=trusted_release_source_bytes,
        post_close_receipt_bytes=receipt,
        manifest_bytes=manifest,
    )
    if result.terminal_outcome != expected_terminal_outcome:
        raise ArchiveComposerError(
            "composed verifier terminal outcome differs from authenticated handoff: "
            f"expected {expected_terminal_outcome}, observed {result.terminal_outcome}"
        )
    if expected_terminal_outcome == "complete":
        if result.status_promotion is None:
            raise ArchiveComposerError("composed verifier did not promote COMPLETE")
        if not result.scientific_admission_complete or result.admission_blockers:
            raise ArchiveComposerError(
                f"scientific admission incomplete: {result.admission_blockers!r}"
            )
    else:
        if result.status_promotion is not None:
            raise ArchiveComposerError("negative terminal outcome was promoted")
        if result.scientific_admission_complete:
            raise ArchiveComposerError("negative terminal outcome reached scientific admission")
        if not result.admission_blockers:
            raise ArchiveComposerError("negative terminal outcome requires retained blockers")
    published = {
        PROCESS_HANDOFF_ARCHIVE_PATH: handoff_after_append,
        STREAM_ARCHIVE_PATH: complete_stream_file,
    }
    for path, raw in objects.items():
        published[path] = _publish_once(archive_directory, path, raw)
    for path, raw in trusted_release_artifacts.items():
        if path in published:
            raise ArchiveComposerError(f"trusted-release publication path duplicates: {path}")
        published[path] = _publish_once(archive_directory, path, raw)
    published[BUILD_EVIDENCE_INPUT_PATH] = _publish_once(
        archive_directory,
        BUILD_EVIDENCE_INPUT_PATH,
        build_evidence_file.raw,
    )
    for path, raw in build_evidence_artifacts.items():
        published[path] = _publish_once(archive_directory, path, raw)
    published[VERIFIER.STATUS.POST_CLOSE_PATH] = _publish_once(
        archive_directory, VERIFIER.STATUS.POST_CLOSE_PATH, receipt
    )
    published[VERIFIER.STATUS.MANIFEST_PATH] = _publish_once(
        archive_directory, VERIFIER.STATUS.MANIFEST_PATH, manifest
    )
    persisted = _re_read_published(archive_directory, published)
    persisted_handoff = persisted[PROCESS_HANDOFF_ARCHIVE_PATH].raw
    persisted_stream = persisted[STREAM_ARCHIVE_PATH].raw
    persisted_manifest = persisted[VERIFIER.STATUS.MANIFEST_PATH].raw
    return CompositionResult(
        archive_directory=archive_directory,
        stream_path=stream_path,
        handoff_sha256=_sha(persisted_handoff),
        stream_sha256=_sha(persisted_stream),
        manifest_sha256=_sha(persisted_manifest),
        scientific_admission_complete=result.scientific_admission_complete,
        terminal_outcome=result.terminal_outcome,
        admission_blockers=tuple(result.admission_blockers),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-directory", required=True, type=pathlib.Path)
    parser.add_argument("--build-coverage", required=True, type=pathlib.Path)
    parser.add_argument("--build-evidence", required=True, type=pathlib.Path)
    parser.add_argument("--build-evidence-root", required=True, type=pathlib.Path)
    parser.add_argument("--append-executable", required=True, type=pathlib.Path)
    parser.add_argument("--trusted-release-authority", required=True, type=pathlib.Path)
    parser.add_argument("--trusted-release-source-root", required=True, type=pathlib.Path)
    parser.add_argument("--trusted-release-authority-sha256", required=True)
    args = parser.parse_args(argv)
    result = compose(
        args.archive_directory,
        build_coverage_path=args.build_coverage,
        build_evidence_path=args.build_evidence,
        build_evidence_root=args.build_evidence_root,
        append_executable=args.append_executable,
        trusted_release_authority_path=args.trusted_release_authority,
        trusted_release_source_root=args.trusted_release_source_root,
        trusted_release_authority_sha256=args.trusted_release_authority_sha256,
    )
    print(
        f"archive {result.terminal_outcome} stream_sha256={result.stream_sha256} "
        f"manifest_sha256={result.manifest_sha256} admission={str(result.scientific_admission_complete).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
