#!/usr/bin/env python3
"""Prepare and independently verify measured OAI memory-profiler build evidence.

The canonical coverage object is derived from immutable source, ELF, final-link,
link-map, toolchain, and dependency byte images. Caller-supplied coverage fields
are never treated as physical identity. The same pure derivation is used by the
offline archive verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import posixpath
import re
import selectors
import shlex
import signal
import stat
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Mapping, Sequence


ROOT = pathlib.Path(__file__).resolve().parents[3]
COVERAGE_PATH = (
    ROOT
    / "tools/profiling/memory/catalog_v1/coverage"
    / "coverage_catalog_v1.py"
)
EVIDENCE_ARCHIVE_PATH = "input/build-evidence.json"
DEFINITION_ARCHIVE_PATH = "definition/oai-memprof-build-evidence-v1.py"
VERSION = {"major": 1, "minor": 3}
MAX_INPUT_BYTES = 512 * 1024 * 1024
MAX_TOOL_OUTPUT_BYTES = 32 * 1024 * 1024
MAX_TOOL_ERROR_BYTES = 1024 * 1024
TOOL_TIMEOUT_SECONDS = 30
TOOL_TERMINATION_GRACE_SECONDS = 1.0
_UNREGISTERED_ALLOCATION_SYMBOLS = frozenset(
    {
        "aligned_alloc",
        "cfree",
        "malloc_usable_size",
        "memalign",
        "posix_memalign",
        "pvalloc",
        "reallocarray",
        "strdup",
        "strndup",
        "valloc",
    }
)
_HASH_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9_]{0,62}\Z", re.ASCII)
_TARGET_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./+-]{0,254}\Z", re.ASCII)
_VERSION_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z.+:~_-]{0,126}\Z", re.ASCII)


class BuildEvidenceError(ValueError):
    """Fail-closed measured-build preparation or verification error."""


def _load(name: str, path: pathlib.Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise BuildEvidenceError(f"module unavailable: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


COVERAGE = _load("_oai_memprof_build_evidence_coverage", COVERAGE_PATH)
_SUPPORTED = tuple(
    (
        row["api_id"],
        row["import_symbol"],
        row["wrapper_symbol"],
        row["wrap_option"],
    )
    for row in COVERAGE.API_RULES
)
_KNOWN_UNSUPPORTED = tuple(
    (row["origin_id"], row["symbol"])
    for row in COVERAGE.KNOWN_UNSUPPORTED_ORIGINS
)


@dataclass(frozen=True)
class LogicalElfInput:
    logical_id: str
    target: str
    elf_path: pathlib.Path
    link_map_path: pathlib.Path
    repo_path: str
    elf_kind_id: int
    role_ids: tuple[int, ...]
    aliases: tuple[str, ...] = ()
    module_selection: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PreparedBuildEvidence:
    coverage_bytes: bytes
    evidence_bytes: bytes
    artifacts: tuple[tuple[str, bytes], ...]

    @property
    def evidence_sha256(self) -> str:
        return _sha(self.evidence_bytes)


@dataclass(frozen=True)
class _SourceSnapshot:
    head: bytes
    head_tree: bytes
    index_tree: bytes
    status: bytes


@dataclass(frozen=True)
class _Section:
    index: int
    name: str
    kind: int
    offset: int
    size: int
    link: int
    entry_size: int


@dataclass(frozen=True)
class _Symbol:
    index: int
    name: str
    binding: int
    visibility: int
    section_index: int

    @property
    def defined(self) -> bool:
        return self.section_index != 0


@dataclass(frozen=True)
class _Elf:
    machine: int
    build_id: str
    needed: tuple[str, ...]
    soname: str | None
    origins: tuple[dict[str, Any], ...]
    relocations: tuple[dict[str, Any], ...]
    wrappers: tuple[str, ...]


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise BuildEvidenceError(detail)


def _read_regular(
    path: pathlib.Path,
    *,
    maximum: int = MAX_INPUT_BYTES,
    allow_empty: bool = False,
    require_single_link: bool = True,
) -> bytes:
    path = path.resolve(strict=True)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        _require(
            stat.S_ISREG(before.st_mode)
            and (not require_single_link or before.st_nlink == 1)
            and before.st_nlink >= 1
            and before.st_size <= maximum
            and (allow_empty or before.st_size > 0),
            f"bounded single-link regular file required: {path}",
        )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1 << 20))
            _require(bool(chunk), f"short read: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        _require(
            (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            == (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ),
            f"file changed while read: {path}",
        )
    finally:
        os.close(descriptor)
    current = os.lstat(path)
    _require(
        (current.st_dev, current.st_ino, current.st_mode, current.st_nlink, current.st_size)
        == (before.st_dev, before.st_ino, before.st_mode, before.st_nlink, before.st_size),
        f"path changed after read: {path}",
    )
    return raw


def _decode(raw: bytes, where: str) -> str:
    try:
        return raw.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise BuildEvidenceError(f"{where}: strict UTF-8 required") from error


def _signal_process_group(process_group_id: int, selected_signal: signal.Signals) -> None:
    try:
        os.killpg(process_group_id, selected_signal)
    except ProcessLookupError:
        pass


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    return True


def _terminate_and_reap(process: subprocess.Popen[bytes]) -> None:
    """Terminate the original session even when its direct leader exited first."""

    process_group_id = process.pid
    _signal_process_group(process_group_id, signal.SIGTERM)
    deadline = time.monotonic() + TOOL_TERMINATION_GRACE_SECONDS
    while _process_group_exists(process_group_id):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.01, remaining))
    if _process_group_exists(process_group_id):
        _signal_process_group(process_group_id, signal.SIGKILL)
    process.wait()


def _run(arguments: Sequence[str], *, cwd: pathlib.Path | None = None) -> bytes:
    _require(
        bool(arguments)
        and all(isinstance(item, str) and item for item in arguments),
        "exact command required",
    )
    environment = {
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }
    process = subprocess.Popen(
        list(arguments),
        cwd=None if cwd is None else str(cwd),
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        bufsize=0,
    )
    selector: selectors.BaseSelector | None = None
    try:
        _require(
            process.stdout is not None and process.stderr is not None,
            "tool pipes unavailable",
        )
        stdout_fd = process.stdout.fileno()
        stderr_fd = process.stderr.fileno()
        streams = {
            stdout_fd: ("stdout", bytearray(), MAX_TOOL_OUTPUT_BYTES),
            stderr_fd: ("stderr", bytearray(), MAX_TOOL_ERROR_BYTES),
        }
        selector = selectors.DefaultSelector()
        for descriptor in streams:
            os.set_blocking(descriptor, False)
            selector.register(descriptor, selectors.EVENT_READ)
        deadline = time.monotonic() + TOOL_TIMEOUT_SECONDS
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BuildEvidenceError(f"tool timeout: {shlex.join(arguments)}")
            events = selector.select(remaining)
            if not events:
                raise BuildEvidenceError(f"tool timeout: {shlex.join(arguments)}")
            for key, _mask in events:
                name, captured, limit = streams[key.fd]
                try:
                    chunk = os.read(key.fd, min(65536, limit - len(captured) + 1))
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fd)
                    continue
                captured.extend(chunk)
                if len(captured) > limit:
                    raise BuildEvidenceError(
                        f"tool {name} exceeds bound: {arguments[0]}"
                    )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BuildEvidenceError(f"tool timeout: {shlex.join(arguments)}")
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            raise BuildEvidenceError(
                f"tool timeout: {shlex.join(arguments)}"
            ) from error
        stdout = bytes(streams[stdout_fd][1])
        stderr = bytes(streams[stderr_fd][1])
        _require(
            returncode == 0 and not stderr,
            f"tool failed: {shlex.join(arguments)}: rc={returncode}: "
            f"{stderr.decode('utf-8', 'replace')}",
        )
        return stdout
    except BaseException:
        _terminate_and_reap(process)
        raise
    finally:
        if selector is not None:
            selector.close()
        if process.stdout is not None and not process.stdout.closed:
            process.stdout.close()
        if process.stderr is not None and not process.stderr.closed:
            process.stderr.close()


def _cstring(raw: bytes, offset: int, where: str) -> str:
    _require(0 <= offset < len(raw), f"{where}: string offset outside table")
    end = raw.find(b"\0", offset)
    _require(end >= offset, f"{where}: unterminated string")
    text = _decode(raw[offset:end], where)
    _require(bool(text) and "/" not in text and "\x00" not in text, f"{where}: invalid ELF string")
    return text


def _slice(raw: bytes, offset: int, size: int, where: str) -> bytes:
    _require(offset >= 0 and size >= 0 and offset + size <= len(raw), f"{where}: range outside ELF")
    return raw[offset : offset + size]


def _parse_sections(raw: bytes) -> tuple[int, list[_Section], dict[int, bytes]]:
    _require(
        len(raw) >= 64
        and raw[:4] == b"\x7fELF"
        and raw[4] == 2
        and raw[5] == 1
        and raw[6] == 1,
        "ELF: exact ELF64 little-endian identity required",
    )
    (
        elf_type,
        machine,
        version,
        _entry,
        _program_offset,
        section_offset,
        _flags,
        elf_header_size,
        _program_entry_size,
        _program_count,
        section_entry_size,
        section_count,
        section_names_index,
    ) = struct.unpack_from("<HHIQQQIHHHHHH", raw, 16)
    _require(
        elf_type in (2, 3)
        and machine in (62, 183)
        and version == 1
        and elf_header_size == 64
        and section_entry_size == 64
        and 1 < section_count <= 65535
        and 0 < section_names_index < section_count,
        "ELF: unsupported header or section table",
    )
    _require(section_offset + section_count * section_entry_size <= len(raw), "ELF: section table outside image")
    records = [
        struct.unpack_from("<IIQQQQIIQQ", raw, section_offset + index * section_entry_size)
        for index in range(section_count)
    ]
    name_record = records[section_names_index]
    names = _slice(raw, name_record[4], name_record[5], "ELF section-name table")
    sections: list[_Section] = []
    section_bytes: dict[int, bytes] = {}
    observed_names: set[str] = set()
    for index, record in enumerate(records):
        name_offset, kind, _flags, _address, offset, size, link, _info, _alignment, entry_size = record
        if index == 0:
            _require(name_offset == 0 and kind == 0, "ELF: invalid null section")
            continue
        name = _cstring(names, name_offset, f"ELF section {index} name")
        _require(name not in observed_names, f"ELF: duplicate section name {name!r}")
        observed_names.add(name)
        body = b"" if kind == 8 else _slice(raw, offset, size, f"ELF section {name}")
        sections.append(_Section(index, name, kind, offset, size, link, entry_size))
        section_bytes[index] = body
    return machine, sections, section_bytes


def _parse_notes(sections: Sequence[_Section], bodies: Mapping[int, bytes]) -> str:
    build_ids: list[str] = []
    for section in sections:
        if section.kind != 7:
            continue
        raw = bodies[section.index]
        cursor = 0
        while cursor < len(raw):
            _require(len(raw) - cursor >= 12, f"ELF note {section.name}: truncated header")
            name_size, description_size, note_type = struct.unpack_from("<III", raw, cursor)
            cursor += 12
            name = _slice(raw, cursor, name_size, f"ELF note {section.name} name")
            cursor = (cursor + name_size + 3) & ~3
            description = _slice(
                raw, cursor, description_size, f"ELF note {section.name} description"
            )
            cursor = (cursor + description_size + 3) & ~3
            _require(cursor <= len(raw), f"ELF note {section.name}: alignment outside section")
            if name.rstrip(b"\0") == b"GNU" and note_type == 3:
                _require(1 <= len(description) <= 64, "ELF: Build-ID length outside 1..64")
                build_ids.append(description.hex())
    _require(len(build_ids) == 1 and int(build_ids[0], 16) != 0, "ELF: exactly one nonzero GNU Build-ID required")
    return build_ids[0]


def _parse_dynamic(
    sections: Sequence[_Section], bodies: Mapping[int, bytes]
) -> tuple[tuple[str, ...], str | None]:
    dynamic_sections = [section for section in sections if section.kind == 6]
    _require(len(dynamic_sections) == 1, "ELF: exactly one dynamic section required")
    dynamic = dynamic_sections[0]
    _require(dynamic.entry_size == 16 and dynamic.size % 16 == 0, "ELF: invalid dynamic entry width")
    strings = bodies.get(dynamic.link)
    _require(strings is not None, "ELF: dynamic string table unavailable")
    needed: list[str] = []
    sonames: list[str] = []
    terminal = False
    for offset in range(0, dynamic.size, 16):
        tag, value = struct.unpack_from("<QQ", bodies[dynamic.index], offset)
        if terminal:
            _require(tag == 0 and value == 0, "ELF: nonzero dynamic entry follows DT_NULL")
            continue
        if tag == 0:
            terminal = True
        elif tag == 1:
            needed.append(_cstring(strings, value, "ELF DT_NEEDED"))
        elif tag == 14:
            sonames.append(_cstring(strings, value, "ELF DT_SONAME"))
    _require(terminal, "ELF: dynamic table lacks DT_NULL")
    _require(len(needed) == len(set(needed)), "ELF: duplicate DT_NEEDED")
    _require(len(sonames) <= 1, "ELF: multiple DT_SONAME values")
    return tuple(sorted(needed)), sonames[0] if sonames else None


def _parse_symbols(
    sections: Sequence[_Section], bodies: Mapping[int, bytes]
) -> tuple[_Section, tuple[_Symbol, ...], tuple[_Symbol, ...]]:
    symbol_lists = {
        kind: [section for section in sections if section.kind == kind]
        for kind in (2, 11)
    }
    _require(
        all(len(symbol_lists[kind]) == 1 for kind in (2, 11)),
        "ELF: exactly one dynsym and one symtab are required",
    )
    symbol_sections = {kind: symbol_lists[kind][0] for kind in (2, 11)}
    parsed: dict[int, tuple[_Symbol, ...]] = {}
    for kind in (11, 2):
        section = symbol_sections[kind]
        _require(section.entry_size == 24 and section.size % 24 == 0, f"ELF: invalid {section.name} width")
        strings = bodies.get(section.link)
        _require(strings is not None, f"ELF: {section.name} string table unavailable")
        rows: list[_Symbol] = []
        for index, offset in enumerate(range(0, section.size, 24)):
            name_offset, info, other, section_index, _value, _size = struct.unpack_from(
                "<IBBHQQ", bodies[section.index], offset
            )
            name = "" if name_offset == 0 else _cstring(strings, name_offset, f"ELF {section.name} symbol {index}")
            rows.append(_Symbol(index, name, info >> 4, other & 3, section_index))
        parsed[kind] = tuple(rows)
    return symbol_sections[11], parsed[11], parsed[2]


def _parse_versions(
    sections: Sequence[_Section],
    bodies: Mapping[int, bytes],
    dynsym: _Section,
    symbol_count: int,
) -> dict[int, str]:
    versym_sections = [section for section in sections if section.kind == 0x6FFFFFFF]
    need_sections = [section for section in sections if section.kind == 0x6FFFFFFE]
    _require(len(versym_sections) == 1 and len(need_sections) == 1, "ELF: exact GNU symbol-version sections required")
    versym = versym_sections[0]
    _require(
        versym.link == dynsym.index and versym.size == symbol_count * 2,
        "ELF: .gnu.version does not align with dynsym",
    )
    need = need_sections[0]
    strings = bodies.get(need.link)
    _require(strings is not None, "ELF: version-need string table unavailable")
    version_names: dict[int, str] = {}
    raw = bodies[need.index]
    cursor = 0
    visited = 0
    while cursor < len(raw):
        _require(len(raw) - cursor >= 16, "ELF: truncated version-need row")
        version, count, _file, auxiliary_offset, next_offset = struct.unpack_from("<HHIII", raw, cursor)
        _require(version == 1 and count > 0 and auxiliary_offset >= 16, "ELF: invalid version-need row")
        auxiliary = cursor + auxiliary_offset
        for index in range(count):
            _require(len(raw) - auxiliary >= 16, "ELF: truncated version auxiliary row")
            _hash_value, _flags, other, name_offset, next_auxiliary = struct.unpack_from(
                "<IHHII", raw, auxiliary
            )
            key = other & 0x7FFF
            name = _cstring(strings, name_offset, "ELF symbol version")
            _require(key > 1 and key not in version_names, "ELF: duplicate version index")
            version_names[key] = name
            if index + 1 == count:
                _require(next_auxiliary == 0, "ELF: final auxiliary version row must terminate")
            else:
                _require(next_auxiliary >= 16, "ELF: version auxiliary chain does not advance")
                auxiliary += next_auxiliary
        visited += 1
        if next_offset == 0:
            _require(cursor + 16 <= len(raw), "ELF: invalid version-need terminal")
            break
        _require(next_offset >= 16, "ELF: version-need chain does not advance")
        cursor += next_offset
        _require(cursor < len(raw), "ELF: version-need chain outside section")
    _require(visited > 0, "ELF: empty version-need section")
    result: dict[int, str] = {}
    versym_raw = bodies[versym.index]
    for index in range(symbol_count):
        version_index = struct.unpack_from("<H", versym_raw, index * 2)[0] & 0x7FFF
        if version_index in version_names:
            result[index] = version_names[version_index]
    return result


def _parse_origins(
    machine: int,
    sections: Sequence[_Section],
    bodies: Mapping[int, bytes],
    dynsym_section: _Section,
    dynsyms: Sequence[_Symbol],
    symtab: Sequence[_Symbol],
    versions: Mapping[int, str],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], tuple[str, ...]]:
    architecture = next(row for row in COVERAGE.ARCHITECTURES if row["elf_machine"] == machine)
    supported = {name: (api_id, wrapper, option) for api_id, name, wrapper, option in _SUPPORTED}
    unsupported = {name: origin_id for origin_id, name in _KNOWN_UNSUPPORTED}
    undefined = {
        symbol.name: symbol
        for symbol in dynsyms
        if symbol.name and not symbol.defined
    }
    recognized = set(undefined) & (
        set(supported) | set(unsupported) | _UNREGISTERED_ALLOCATION_SYMBOLS
    )
    new_symbols = sorted(
        (recognized & _UNREGISTERED_ALLOCATION_SYMBOLS)
        - set(unsupported)
        - set(supported)
    )
    _require(
        not new_symbols,
        "ELF: allocation origins are not registered by the frozen coverage "
        f"policy: {new_symbols!r}",
    )
    origin_by_symbol: dict[str, dict[str, Any]] = {}
    for name in sorted(recognized):
        symbol = undefined[name]
        _require(
            symbol.index in versions,
            f"ELF: allocation origin {name!r} lacks a symbol version",
        )
        version = versions[symbol.index]
        if name in supported:
            expected_version = COVERAGE.expected_supported_symbol_version(
                supported[name][0], architecture["architecture_id"]
            )
        else:
            expected_version = COVERAGE.expected_known_unsupported_symbol_version(
                unsupported[name], architecture["architecture_id"]
            )
        _require(
            version == expected_version,
            f"ELF: allocation origin {name!r} version {version!r} differs "
            f"from {expected_version!r}",
        )
        if name in supported:
            api_id, _wrapper, _option = supported[name]
            origin = {
                "api_id": api_id,
                "classification_id": 1,
                "origin_id": api_id,
                "origin_kind_id": 1,
                "symbol": name,
                "symbol_version": version,
            }
        else:
            origin_id = unsupported[name]
            origin = {
                "api_id": None,
                "classification_id": 10,
                "origin_id": origin_id,
                "origin_kind_id": 2,
                "symbol": name,
                "symbol_version": version,
            }
        origin_by_symbol[name] = origin
    relocation_rows: list[dict[str, Any]] = []
    seen_relocations: set[str] = set()
    relocation_types = {62: {7: 1, 6: 2}, 183: {1026: 1, 1025: 2}}[machine]
    for section in sections:
        if section.kind not in (4, 9) or section.link != dynsym_section.index:
            continue
        width = 24 if section.kind == 4 else 16
        _require(section.entry_size == width and section.size % width == 0, f"ELF: invalid relocation section {section.name}")
        for offset in range(0, section.size, width):
            if section.kind == 4:
                _relocation_offset, info, _addend = struct.unpack_from("<QQq", bodies[section.index], offset)
            else:
                _relocation_offset, info = struct.unpack_from("<QQ", bodies[section.index], offset)
            symbol_index = info >> 32
            relocation_type = info & 0xFFFFFFFF
            _require(symbol_index < len(dynsyms), f"ELF: relocation symbol outside dynsym in {section.name}")
            name = dynsyms[symbol_index].name
            if name not in origin_by_symbol:
                continue
            _require(relocation_type in relocation_types, f"ELF: unsupported relocation kind {relocation_type} for {name}")
            _require(name not in seen_relocations, f"ELF: multiple import relocations for {name}")
            seen_relocations.add(name)
            relocation_rows.append(
                {
                    "origin_id": origin_by_symbol[name]["origin_id"],
                    "relocation_kind_id": relocation_types[relocation_type],
                }
            )
    _require(seen_relocations == set(origin_by_symbol), "ELF: allocation imports and relocations differ")
    origins = tuple(sorted(origin_by_symbol.values(), key=lambda row: row["origin_id"]))
    relocations = tuple(sorted(relocation_rows, key=lambda row: row["origin_id"]))
    defined = {symbol.name: symbol for symbol in symtab if symbol.name and symbol.defined}
    expected_wrappers = {
        supported[row["symbol"]][1] for row in origins if row["origin_kind_id"] == 1
    }
    observed_wrappers = {
        name for name in defined if name.startswith("__wrap_")
    }
    _require(
        observed_wrappers == expected_wrappers,
        "ELF: exact supported import wrapper definitions are required",
    )
    exported_wrappers = {
        symbol.name
        for symbol in dynsyms
        if symbol.name.startswith("__wrap_") and symbol.defined
    }
    _require(not exported_wrappers, f"ELF: wrapper symbols must not be dynamically exported: {sorted(exported_wrappers)!r}")
    return origins, relocations, tuple(sorted(observed_wrappers))


def _parse_elf(raw: bytes) -> _Elf:
    machine, sections, bodies = _parse_sections(raw)
    build_id = _parse_notes(sections, bodies)
    needed, soname = _parse_dynamic(sections, bodies)
    dynsym_section, dynsyms, symtab = _parse_symbols(sections, bodies)
    versions = _parse_versions(sections, bodies, dynsym_section, len(dynsyms))
    origins, relocations, wrappers = _parse_origins(
        machine, sections, bodies, dynsym_section, dynsyms, symtab, versions
    )
    return _Elf(machine, build_id, needed, soname, origins, relocations, wrappers)


def _archive_path(value: Any, where: str) -> str:
    _require(isinstance(value, str) and value, f"{where}: nonempty path required")
    _require(
        not value.startswith("/")
        and not value.endswith("/")
        and "\\" not in value
        and all(part not in ("", ".", "..") for part in value.split("/"))
        and all(ord(character) >= 0x20 and ord(character) != 0x7F for character in value),
        f"{where}: normalized archive-relative POSIX path required",
    )
    return value


def _identifier(value: Any, where: str) -> str:
    _require(isinstance(value, str) and _IDENTIFIER_RE.fullmatch(value) is not None, f"{where}: identifier required")
    return value


def _target(value: Any, where: str) -> str:
    _require(
        isinstance(value, str) and _TARGET_RE.fullmatch(value) is not None,
        f"{where}: bounded Ninja target required",
    )
    return _archive_path(value, where)


def _version_line(raw: bytes, where: str, pattern: str) -> str:
    text = _decode(raw, where)
    lines = text.splitlines()
    _require(len(lines) == 1, f"{where}: exactly one output line required")
    match = re.fullmatch(pattern, lines[0])
    _require(match is not None, f"{where}: unexpected output {lines[0]!r}")
    value = match.group(1)
    _require(_VERSION_RE.fullmatch(value) is not None, f"{where}: bounded version required")
    return value


def _source_object(raw: bytes, where: str) -> str:
    text = _decode(raw, where)
    _require(re.fullmatch(r"[0-9a-f]{40}\n", text) is not None, f"{where}: exact SHA-1 object plus LF required")
    return text[:-1]


def _source_snapshot(
    git: pathlib.Path, repository: pathlib.Path
) -> _SourceSnapshot:
    top_raw = _run(
        (str(git), "-C", str(repository), "rev-parse", "--show-toplevel")
    )
    top_text = _decode(top_raw, "Git top-level")
    _require(
        top_text.endswith("\n") and top_text.count("\n") == 1,
        "Git top-level: exactly one path line required",
    )
    top = pathlib.Path(top_text[:-1]).resolve(strict=True)
    _require(top == repository, "measured repository is not the exact Git work-tree root")
    status = _run(
        (
            str(git),
            "-C",
            str(repository),
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        )
    )
    _require(status == b"", "measured build source repository is dirty")
    head = _run((str(git), "-C", str(repository), "rev-parse", "HEAD"))
    head_tree = _run(
        (str(git), "-C", str(repository), "rev-parse", "HEAD^{tree}")
    )
    index_tree = _run((str(git), "-C", str(repository), "write-tree"))
    _source_object(head, "source HEAD")
    expected_tree = _source_object(head_tree, "source HEAD tree")
    _require(
        _source_object(index_tree, "source index tree") == expected_tree,
        "source index tree differs from HEAD tree",
    )
    return _SourceSnapshot(head, head_tree, index_tree, status)


def _require_source_snapshot(
    expected: _SourceSnapshot,
    git: pathlib.Path,
    repository: pathlib.Path,
    where: str,
) -> None:
    _require(
        _source_snapshot(git, repository) == expected,
        f"measured source identity changed {where}",
    )


def _validate_cmake_build_root(
    build_directory: pathlib.Path, repository: pathlib.Path
) -> None:
    cache_raw = _read_regular(
        build_directory / "CMakeCache.txt", maximum=16 * 1024 * 1024
    )
    cache = _decode(cache_raw, "CMake cache")

    def cache_value(name: str) -> str:
        matches = []
        prefix = f"{name}:"
        for line in cache.splitlines():
            if line.startswith(prefix) and "=" in line:
                matches.append(line.split("=", 1)[1])
        _require(len(matches) == 1 and matches[0], f"CMake cache: exact {name} required")
        return matches[0]

    source_root = pathlib.Path(cache_value("CMAKE_HOME_DIRECTORY")).resolve(
        strict=True
    )
    _require(
        source_root == repository,
        "CMake build directory was configured from another source root",
    )
    _require(
        cache_value("CMAKE_GENERATOR") == "Ninja",
        "CMake build directory requires the Ninja generator",
    )
    build_ninja = (build_directory / "build.ninja").resolve(strict=True)
    _require(
        build_ninja.parent == build_directory and build_ninja.is_file(),
        "CMake build directory requires a local build.ninja",
    )


def _resolve_under(
    root: pathlib.Path, path: pathlib.Path, where: str
) -> pathlib.Path:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise BuildEvidenceError(f"{where}: path escapes the build directory") from error
    _require(relative.parts, f"{where}: build root itself is not an artifact")
    return resolved


def _selection(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {
            "operator_id": 1,
            "predicates": [
                {
                    "configuration_key": None,
                    "expected_value": None,
                    "predicate_id": 1,
                }
            ],
        }
    return json.loads(json.dumps(value, ensure_ascii=True, allow_nan=False))


def _ninja_final_command(
    ninja: pathlib.Path, build_directory: pathlib.Path, target: str
) -> bytes:
    """Return only Ninja's final command for one measured logical ELF."""

    return _run(
        (
            str(ninja),
            "-C",
            str(build_directory),
            "-t",
            "commands",
            "-s",
            target,
        )
    )


def _build_directory_identity(value: Any, where: str) -> str:
    _require(
        isinstance(value, str)
        and value.startswith("/")
        and not value.startswith("//")
        and value != "/"
        and "\\" not in value
        and all(ord(character) >= 0x20 and ord(character) != 0x7F for character in value)
        and posixpath.normpath(value) == value,
        f"{where}: normalized absolute POSIX build directory required",
    )
    return value


def _normalized_build_artifact_path(
    value: Any,
    *,
    build_directory: str,
    where: str,
) -> str:
    _require(
        isinstance(value, str)
        and value
        and "\\" not in value
        and all(ord(character) >= 0x20 and ord(character) != 0x7F for character in value)
        and posixpath.normpath(value) == value,
        f"{where}: normalized POSIX path required",
    )
    absolute = value if posixpath.isabs(value) else posixpath.join(build_directory, value)
    _require(
        posixpath.commonpath((build_directory, absolute)) == build_directory
        and absolute != build_directory,
        f"{where}: path escapes the authenticated build directory",
    )
    return _archive_path(
        posixpath.relpath(absolute, build_directory),
        f"{where} build-relative path",
    )


def _command_output_tokens(tokens: Sequence[str], where: str) -> list[str]:
    outputs: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "-o":
            _require(index + 1 < len(tokens) and tokens[index + 1], f"{where}: -o value required")
            outputs.append(tokens[index + 1])
            index += 2
            continue
        if token.startswith("-o") and len(token) > 2:
            outputs.append(token[2:])
        index += 1
    return outputs


def _command_map_tokens(tokens: Sequence[str]) -> list[str]:
    values: list[str] = []
    for token in tokens:
        if token.startswith("-Wl,-Map,") and len(token) > len("-Wl,-Map,"):
            values.append(token[len("-Wl,-Map,") :])
        elif token.startswith("-Wl,-Map=") and len(token) > len("-Wl,-Map="):
            values.append(token[len("-Wl,-Map=") :])
    return values


def _validate_final_link_command(
    command_raw: bytes,
    *,
    build_directory: str,
    build_output_path: str,
    build_map_path: str,
    expected_wraps: Sequence[str],
    where: str,
) -> None:
    command = _decode(command_raw, where)
    _require(
        command.endswith("\n") and command.count("\n") == 1,
        f"{where}: exactly one final-link command line required",
    )
    try:
        tokens = shlex.split(command.strip())
    except ValueError as error:
        raise BuildEvidenceError(f"{where}: malformed shell quoting") from error
    outputs = _command_output_tokens(tokens, where)
    _require(len(outputs) == 1, f"{where}: exactly one -o output required")
    _require(
        _normalized_build_artifact_path(
            outputs[0], build_directory=build_directory, where=f"{where} -o"
        )
        == build_output_path,
        f"{where}: -o does not name the measured ELF",
    )
    map_values = _command_map_tokens(tokens)
    _require(
        len(map_values) == 1,
        f"{where}: exactly one supported GNU ld map option required",
    )
    _require(
        _normalized_build_artifact_path(
            map_values[0], build_directory=build_directory, where=f"{where} map option"
        )
        == build_map_path,
        f"{where}: map option does not name the authenticated link map",
    )
    observed_wraps = sorted(
        token[4:] for token in tokens if token.startswith("-Wl,--wrap=")
    )
    _require(
        observed_wraps == sorted(expected_wraps),
        f"{where}: exact wrap set differs",
    )


def _link_command(
    output: bytes,
    *,
    build_directory: str,
    build_output_path: str,
    build_map_path: str,
    expected_wraps: Sequence[str],
) -> bytes:
    text = _decode(output, "ninja command output")
    build_directory = _build_directory_identity(
        build_directory, "final link command build directory"
    )
    build_output_path = _archive_path(
        build_output_path, "final link command output"
    )
    build_map_path = _archive_path(build_map_path, "final link command map")
    accepted_outputs = {
        build_output_path,
        posixpath.join(build_directory, build_output_path),
    }
    candidates: list[str] = []
    for line in text.splitlines():
        try:
            tokens = shlex.split(line)
        except ValueError as error:
            raise BuildEvidenceError("ninja command output: malformed shell quoting") from error
        if any(
            value in accepted_outputs
            for value in _command_output_tokens(tokens, "ninja command output")
        ):
            candidates.append(line)
    _require(
        len(candidates) == 1,
        f"final link command for {build_output_path}: exactly one Ninja command required",
    )
    command_raw = (candidates[0] + "\n").encode("utf-8")
    _validate_final_link_command(
        command_raw,
        build_directory=build_directory,
        build_output_path=build_output_path,
        build_map_path=build_map_path,
        expected_wraps=expected_wraps,
        where=f"final link command for {build_output_path}",
    )
    return command_raw


def _validate_link_map_output(
    map_raw: bytes,
    *,
    build_directory: str,
    build_output_path: str,
    where: str,
) -> None:
    _require(map_raw and b"\0" not in map_raw, f"{where}: nonempty NUL-free bytes required")
    text = _decode(map_raw, where)
    outputs: list[str] = []
    for line in text.splitlines():
        match = re.fullmatch(r"OUTPUT\(([^\s()]+) [^()]+\)", line)
        if match is not None:
            outputs.append(match.group(1))
    _require(len(outputs) == 1, f"{where}: exactly one GNU ld OUTPUT record required")
    _require(
        _normalized_build_artifact_path(
            outputs[0], build_directory=build_directory, where=f"{where} OUTPUT"
        )
        == build_output_path,
        f"{where}: OUTPUT does not name the measured ELF",
    )


def _entry_paths(logical_id: str) -> dict[str, str]:
    base = f"input/build-evidence/{logical_id}"
    return {
        "elf_path": f"{base}.elf",
        "link_command_path": f"{base}.link-command.txt",
        "link_map_path": f"{base}.link-map.txt",
    }


def _tool_paths() -> dict[str, str]:
    return {
        "compiler_version_path": "input/build-evidence/tool-gcc-version.txt",
        "libc_path": "input/build-evidence/libc.so.6",
        "libc_version_path": "input/build-evidence/tool-libc-version.txt",
        "linker_version_path": "input/build-evidence/tool-ld-version.txt",
        "source_commit_object_path": "input/build-evidence/source-commit-object.bin",
        "source_head_path": "input/build-evidence/source-head.txt",
        "source_status_path": "input/build-evidence/source-status.bin",
        "source_tree_path": "input/build-evidence/source-tree.txt",
        "target_triple_path": "input/build-evidence/tool-target-triple.txt",
    }


def _exact_artifact_entries(artifacts: Mapping[str, bytes]) -> list[dict[str, Any]]:
    return [
        {"bytes": len(raw), "path": path, "sha256": _sha(raw)}
        for path, raw in sorted(artifacts.items())
    ]


def _required_artifact(
    artifacts: Mapping[str, bytes], path: str, where: str
) -> bytes:
    _require(path in artifacts, f"{where}: required artifact unavailable: {path}")
    raw = artifacts[path]
    _require(type(raw) is bytes, f"{where}: immutable artifact bytes required: {path}")
    return raw


def _derive(
    evidence: Mapping[str, Any],
    artifacts: Mapping[str, bytes],
    *,
    api_definition_sha256: str,
    enforce_advertised_digest: bool = True,
) -> bytes:
    expected_keys = {
        "api_definition_sha256",
        "build_directory",
        "build_coverage_sha256",
        "catalog_id",
        "entries",
        "logical_elfs",
        "primary_logical_elf_id",
        "toolchain",
        "version",
    }
    _require(set(evidence) == expected_keys, "build evidence: exact root members required")
    _require(evidence["catalog_id"] == "oai_memprof_build_evidence", "build evidence: catalog ID mismatch")
    _require(evidence["version"] == VERSION, f"build evidence: version {VERSION['major']}.{VERSION['minor']} required")
    build_directory = _build_directory_identity(
        evidence["build_directory"], "build evidence build_directory"
    )
    _require(
        isinstance(evidence["api_definition_sha256"], str)
        and evidence["api_definition_sha256"] == api_definition_sha256
        and _HASH_RE.fullmatch(api_definition_sha256) is not None,
        "build evidence: API definition digest mismatch",
    )
    manifest_rows = evidence["entries"]
    _require(isinstance(manifest_rows, list), "build evidence entries: array required")
    expected_manifest = _exact_artifact_entries(artifacts)
    _require(manifest_rows == expected_manifest, "build evidence entries: exact path/size/digest set required")
    for path in artifacts:
        _archive_path(path, f"build evidence artifact {path!r}")
    tools = evidence["toolchain"]
    tool_keys = set(_tool_paths())
    _require(isinstance(tools, dict) and set(tools) == tool_keys, "build evidence toolchain: exact path set required")
    for key, expected_path in _tool_paths().items():
        _require(tools[key] == expected_path, f"build evidence toolchain.{key}: exact path required")
    tool_artifacts = {
        key: _required_artifact(
            artifacts, path, f"build evidence toolchain.{key}"
        )
        for key, path in tools.items()
    }
    status_raw = tool_artifacts["source_status_path"]
    _require(status_raw == b"", "build evidence: measured source status must be exactly empty")
    source_commit = _source_object(tool_artifacts["source_head_path"], "source HEAD")
    source_tree = _source_object(tool_artifacts["source_tree_path"], "source tree")
    commit_object = tool_artifacts["source_commit_object_path"]
    _require(
        hashlib.sha1(
            b"commit " + str(len(commit_object)).encode("ascii") + b"\0" + commit_object
        ).hexdigest()
        == source_commit,
        "build evidence: raw commit object does not hash to source HEAD",
    )
    _require(
        commit_object.startswith(f"tree {source_tree}\n".encode("ascii")),
        "build evidence: source HEAD does not bind the declared source tree",
    )
    compiler_version = _version_line(
        tool_artifacts["compiler_version_path"], "gcc version", r"([0-9]+(?:\.[0-9]+)+)\n?"
    )
    target_triple = _version_line(
        tool_artifacts["target_triple_path"], "target triple", r"([a-z0-9_]+-linux-gnu)\n?"
    )
    linker_version = _version_line(
        tool_artifacts["linker_version_path"],
        "GNU ld version",
        r"GNU ld \(GNU Binutils for [^)]+\) ([0-9]+(?:\.[0-9]+)+)\n?",
    )
    libc_version = _version_line(
        tool_artifacts["libc_version_path"], "glibc version", r"glibc ([0-9]+(?:\.[0-9]+)+)\n?"
    )
    architecture_id = {"x86_64-linux-gnu": 1, "aarch64-linux-gnu": 2}.get(target_triple)
    _require(architecture_id is not None, f"build evidence: unsupported target triple {target_triple!r}")
    architecture = next(row for row in COVERAGE.ARCHITECTURES if row["architecture_id"] == architecture_id)
    libc_raw = tool_artifacts["libc_path"]
    libc_machine, libc_sections, libc_bodies = _parse_sections(libc_raw)
    _libc_needed, libc_soname = _parse_dynamic(libc_sections, libc_bodies)
    _require(
        libc_machine == architecture["elf_machine"] and libc_soname == "libc.so.6",
        "build evidence: libc ELF identity mismatch",
    )
    logical_values = evidence["logical_elfs"]
    _require(isinstance(logical_values, list) and logical_values, "build evidence logical_elfs: nonempty array required")
    logical_ids = [row.get("logical_id") for row in logical_values if isinstance(row, dict)]
    _require(
        len(logical_ids) == len(logical_values)
        and logical_ids == sorted(logical_ids)
        and len(logical_ids) == len(set(logical_ids)),
        "build evidence logical_elfs: strict logical-ID order/uniqueness required",
    )
    build_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, str]] = []
    required_artifact_paths = set(tools.values())
    for index, specification in enumerate(logical_values):
        where = f"build evidence logical_elfs[{index}]"
        keys = {
            "aliases",
            "build_map_path",
            "build_output_path",
            "elf_kind_id",
            "elf_path",
            "link_command_path",
            "link_map_path",
            "logical_id",
            "module_selection",
            "repo_path",
            "role_ids",
        }
        _require(isinstance(specification, dict) and set(specification) == keys, f"{where}: exact members required")
        logical_id = _identifier(specification["logical_id"], f"{where}.logical_id")
        build_output_path = _archive_path(
            specification["build_output_path"], f"{where}.build_output_path"
        )
        build_map_path = _archive_path(
            specification["build_map_path"], f"{where}.build_map_path"
        )
        paths = _entry_paths(logical_id)
        for key, path in paths.items():
            _require(specification[key] == path, f"{where}.{key}: exact derived archive path required")
        required_artifact_paths.update(paths.values())
        elf_raw = _required_artifact(artifacts, paths["elf_path"], where)
        map_raw = _required_artifact(artifacts, paths["link_map_path"], where)
        command_raw = _required_artifact(
            artifacts, paths["link_command_path"], where
        )
        parsed = _parse_elf(elf_raw)
        _require(parsed.machine == architecture["elf_machine"], f"{where}: ELF architecture mismatch")
        elf_kind = specification["elf_kind_id"]
        _require(type(elf_kind) is int and elf_kind in (1, 2), f"{where}.elf_kind_id: 1 or 2 required")
        if elf_kind == 1:
            _require(parsed.soname is None, f"{where}: executable forbids SONAME")
        supported_ids = [row["api_id"] for row in parsed.origins if row["api_id"] is not None]
        wraps = sorted(
            option for api_id, _name, _wrapper, option in _SUPPORTED if api_id in supported_ids
        )
        _validate_final_link_command(
            command_raw,
            build_directory=build_directory,
            build_output_path=build_output_path,
            build_map_path=build_map_path,
            expected_wraps=wraps,
            where=f"{where}.link_command",
        )
        _validate_link_map_output(
            map_raw,
            build_directory=build_directory,
            build_output_path=build_output_path,
            where=f"{where}.link_map",
        )
        aliases = specification["aliases"]
        role_ids = specification["role_ids"]
        policy = 1 if 3 in supported_ids else None
        _require(isinstance(aliases, list), f"{where}.aliases: array required")
        _require(isinstance(role_ids, list), f"{where}.role_ids: array required")
        row = {
            "admission_state_id": 1 if supported_ids else 2,
            "aliases": aliases,
            "build_id": parsed.build_id,
            "byte_count": len(elf_raw),
            "dt_needed": list(parsed.needed),
            "elf_kind_id": elf_kind,
            "elf_machine": parsed.machine,
            "evidence_state_id": 1,
            "hidden_wrapper_symbols": list(parsed.wrappers),
            "import_relocations": list(parsed.relocations),
            "link_command_sha256": _sha(command_raw),
            "link_map_sha256": _sha(map_raw),
            "logical_id": logical_id,
            "module_selection": specification["module_selection"],
            "realloc_zero_policy_id": policy,
            "realloc_zero_semantic_oracle_sha256": (
                COVERAGE.REALLOC_ZERO_SEMANTIC_ORACLE_BINDING["sha256"]
                if policy is not None
                else None
            ),
            "repo_path": specification["repo_path"],
            "role_ids": role_ids,
            "sha256": _sha(elf_raw),
            "shared_runtime_binding": (
                {
                    "dependency_id": "glibc_runtime",
                    "evidence_state_id": 1,
                    "soname": "libc.so.6",
                }
                if supported_ids
                else None
            ),
            "soname": parsed.soname,
            "symbol_origins": list(parsed.origins),
            "wrap_options": wraps,
        }
        build_rows.append(row)
        projection_rows.append(
            {
                "build_map_path": build_map_path,
                "build_output_path": build_output_path,
                "link_command_sha256": row["link_command_sha256"],
                "link_map_sha256": row["link_map_sha256"],
                "logical_id": logical_id,
            }
        )
    _require(
        set(artifacts) == required_artifact_paths,
        "build evidence: exact required raw-artifact path set required",
    )
    supported_rows = [
        row
        for row in build_rows
        if any(origin["origin_kind_id"] == 1 for origin in row["symbol_origins"])
    ]
    runtime_rows = [
        row
        for row in build_rows
        if row["soname"] == "liboai_memprof_active_runtime.so.1"
    ]
    if supported_rows:
        _require(
            len(runtime_rows) == 1,
            "build evidence: supported imports require exactly one active-runtime DSO row",
        )
        runtime_row = runtime_rows[0]
        _require(
            runtime_row["logical_id"] == "oai_memprof_active_runtime"
            and runtime_row["elf_kind_id"] == 2
            and runtime_row["admission_state_id"] == 2
            and not runtime_row["symbol_origins"]
            and runtime_row["dt_needed"] == ["libc.so.6"],
            "build evidence: active-runtime DSO exact zero-import identity required",
        )
        _require(
            all(
                "liboai_memprof_active_runtime.so.1" in row["dt_needed"]
                for row in supported_rows
            ),
            "build evidence: every supported logical ELF must bind the active-runtime SONAME",
        )
    else:
        _require(
            not runtime_rows,
            "build evidence: active-runtime row is forbidden without supported imports",
        )
    primary_id = _identifier(evidence["primary_logical_elf_id"], "build evidence primary_logical_elf_id")
    _require(
        any(row["logical_id"] == primary_id and row["elf_kind_id"] == 1 for row in build_rows),
        "build evidence: primary logical ELF must name one executable",
    )
    build_configuration = COVERAGE.canonical_bytes(
        {
            "build_directory": build_directory,
            "compiler_version": compiler_version,
            "entries": projection_rows,
            "linker_version": linker_version,
            "source_commit": source_commit,
            "source_tree": source_tree,
            "target_triple": target_triple,
            "version": VERSION,
        }
    )
    build = {
        "api_definition": {
            "object_type": 4,
            "path": "catalog/api.json",
            "sha256": api_definition_sha256,
        },
        "architecture_id": architecture_id,
        "build_identity": {
            "build_configuration_sha256": _sha(build_configuration),
            "compiler_id": "gcc",
            "compiler_version": compiler_version,
            "dirty": False,
            "libc_id": "glibc",
            "libc_version": libc_version,
            "linker_id": "gnu_ld",
            "linker_version": linker_version,
            "operating_system": "linux",
            "primary_logical_elf_id": primary_id,
            "source_commit": source_commit,
            "source_tree": source_tree,
            "target_triple": target_triple,
        },
        "catalog_id": "oai_memprof_build_coverage",
        "dependencies": [
            {
                "dependency_id": "glibc_runtime",
                "evidence_state_id": 1,
                "name": "libc.so.6",
                "sha256": _sha(libc_raw),
                "version": libc_version,
            }
        ],
        "entries": build_rows,
        "evidence_origin_id": 1,
        "failure_ids": [],
        "policy": {
            "object_type": 9,
            "path": COVERAGE.POLICY_ARCHIVE_PATH,
            "sha256": COVERAGE.POLICY_SHA256,
        },
        "schema": {
            "object_type": 10,
            "path": COVERAGE.INSTANCE_SCHEMA_ARCHIVE_PATH,
            "sha256": COVERAGE.INSTANCE_SCHEMA_SHA256,
        },
        "verdict_id": 1,
        "version": {"major": 1, "minor": 0},
    }
    coverage_raw = COVERAGE.canonical_bytes(build)
    COVERAGE.validate_build_coverage_bytes(
        coverage_raw, api_definition_sha256=api_definition_sha256
    )
    if enforce_advertised_digest:
        _require(
            evidence["build_coverage_sha256"] == _sha(coverage_raw),
            "build evidence: derived build-coverage digest mismatch",
        )
    return coverage_raw


def validate_build_evidence_bytes(
    evidence_raw: bytes,
    artifact_bytes: Mapping[str, bytes],
    build_coverage_raw: bytes,
    *,
    api_definition_sha256: str,
) -> dict[str, Any]:
    """Validate exact raw evidence and return the canonical derived build object."""

    _require(isinstance(evidence_raw, bytes), "build evidence: immutable bytes required")
    evidence = COVERAGE.parse_canonical(evidence_raw)
    _require(isinstance(artifact_bytes, Mapping), "build evidence artifacts: mapping required")
    artifacts: dict[str, bytes] = {}
    for path, raw in artifact_bytes.items():
        canonical = _archive_path(path, f"build evidence artifact {path!r}")
        _require(type(raw) is bytes, f"build evidence artifact {canonical}: immutable bytes required")
        artifacts[canonical] = bytes(raw)
    derived = _derive(evidence, artifacts, api_definition_sha256=api_definition_sha256)
    _require(type(build_coverage_raw) is bytes and derived == build_coverage_raw, "build evidence: exact canonical build coverage bytes differ")
    return COVERAGE.parse_canonical(derived)


def _open_directory_path(path: pathlib.Path) -> int:
    absolute = pathlib.Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in absolute.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        result = os.fstat(descriptor)
        _require(stat.S_ISDIR(result.st_mode), f"directory required: {absolute}")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_or_create_directory(parent: int, component: str) -> int:
    try:
        os.mkdir(component, mode=0o750, dir_fd=parent)
    except FileExistsError:
        pass
    descriptor = os.open(
        component,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=parent,
    )
    result = os.fstat(descriptor)
    _require(stat.S_ISDIR(result.st_mode), f"publication directory required: {component}")
    return descriptor


def _publish_at(root_descriptor: int, relative: str, raw: bytes) -> None:
    relative = _archive_path(relative, "evidence publication path")
    _require(type(raw) is bytes, f"evidence publication immutable bytes required: {relative}")
    components = relative.split("/")
    parent = os.dup(root_descriptor)
    try:
        for component in components[:-1]:
            child = _open_or_create_directory(parent, component)
            os.close(parent)
            parent = child
        descriptor = os.open(
            components[-1],
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o640,
            dir_fd=parent,
        )
        try:
            created = os.fstat(descriptor)
            _require(
                stat.S_ISREG(created.st_mode)
                and created.st_nlink == 1
                and created.st_size == 0,
                f"evidence publication fresh regular file required: {relative}",
            )
            view = memoryview(raw)
            offset = 0
            while offset < len(view):
                count = os.write(descriptor, view[offset:])
                _require(count > 0, f"evidence publication short write: {relative}")
                offset += count
            os.fsync(descriptor)
            result = os.fstat(descriptor)
            _require(
                stat.S_ISREG(result.st_mode)
                and result.st_nlink == 1
                and result.st_size == len(raw)
                and (result.st_dev, result.st_ino) == (created.st_dev, created.st_ino),
                f"evidence publication identity mismatch: {relative}",
            )
            os.lseek(descriptor, 0, os.SEEK_SET)
            captured = bytearray()
            while len(captured) < len(raw):
                chunk = os.read(descriptor, min(1 << 20, len(raw) - len(captured)))
                _require(bool(chunk), f"evidence publication short verification read: {relative}")
                captured.extend(chunk)
            _require(
                bytes(captured) == raw and os.read(descriptor, 1) == b"",
                f"evidence publication content mismatch: {relative}",
            )
        finally:
            os.close(descriptor)
        anchored = os.stat(components[-1], dir_fd=parent, follow_symlinks=False)
        _require(
            (anchored.st_dev, anchored.st_ino, anchored.st_mode, anchored.st_nlink, anchored.st_size)
            == (result.st_dev, result.st_ino, result.st_mode, result.st_nlink, result.st_size),
            f"evidence publication path changed after close: {relative}",
        )
        os.fsync(parent)
    finally:
        os.close(parent)


def _publish(root: pathlib.Path, relative: str, raw: bytes) -> None:
    root_descriptor = _open_directory_path(root)
    try:
        _publish_at(root_descriptor, relative, raw)
    finally:
        os.close(root_descriptor)


def _request_logical(value: Mapping[str, Any], index: int) -> LogicalElfInput:
    where = f"request.logical_elfs[{index}]"
    keys = {
        "aliases",
        "elf_kind_id",
        "elf_path",
        "link_map_path",
        "logical_id",
        "module_selection",
        "repo_path",
        "role_ids",
        "target",
    }
    _require(isinstance(value, dict) and set(value) == keys, f"{where}: exact members required")
    return LogicalElfInput(
        logical_id=_identifier(value["logical_id"], f"{where}.logical_id"),
        target=_target(value["target"], f"{where}.target"),
        elf_path=pathlib.Path(value["elf_path"]),
        link_map_path=pathlib.Path(value["link_map_path"]),
        repo_path=value["repo_path"],
        elf_kind_id=value["elf_kind_id"],
        role_ids=tuple(value["role_ids"]),
        aliases=tuple(value["aliases"]),
        module_selection=value["module_selection"],
    )


def prepare_measured_build_evidence(
    *,
    repository: pathlib.Path,
    build_directory: pathlib.Path,
    evidence_root: pathlib.Path,
    logical_elfs: Sequence[LogicalElfInput],
    primary_logical_elf_id: str,
    libc_path: pathlib.Path,
    api_definition_sha256: str,
    git: pathlib.Path = pathlib.Path("/usr/bin/git"),
    gcc: pathlib.Path = pathlib.Path("/usr/bin/gcc"),
    ld: pathlib.Path = pathlib.Path("/usr/bin/ld"),
    ninja: pathlib.Path = pathlib.Path("/usr/bin/ninja"),
) -> PreparedBuildEvidence:
    """Capture a clean build and publish exact raw evidence beneath an empty root."""

    repository = repository.resolve(strict=True)
    build_directory = build_directory.resolve(strict=True)
    build_directory_identity = _build_directory_identity(
        str(build_directory), "measured build directory"
    )
    evidence_root = pathlib.Path(os.path.abspath(os.fspath(evidence_root)))
    root_probe = _open_directory_path(evidence_root)
    try:
        initial_root = os.fstat(root_probe)
        _require(
            stat.S_ISDIR(initial_root.st_mode) and not os.listdir(root_probe),
            "evidence root must be an existing empty no-symlink directory",
        )
    finally:
        os.close(root_probe)
    try:
        evidence_root.relative_to(repository)
    except ValueError:
        pass
    else:
        raise BuildEvidenceError("evidence root must remain outside the source repository")
    _require(logical_elfs and len(logical_elfs) == len({row.logical_id for row in logical_elfs}), "logical ELF inputs must be nonempty and unique")
    _require([row.logical_id for row in logical_elfs] == sorted(row.logical_id for row in logical_elfs), "logical ELF inputs must be sorted")
    _validate_cmake_build_root(build_directory, repository)
    source_snapshot = _source_snapshot(git, repository)
    status = source_snapshot.status
    head = source_snapshot.head
    tree = source_snapshot.head_tree
    head_object = _source_object(head, "source HEAD")
    compiler_version = _run((str(gcc), "-dumpfullversion"))
    target_triple = _run((str(gcc), "-dumpmachine"))
    linker_version = _run((str(ld), "--version")).splitlines(keepends=True)[0]
    libc_version = _run(("/usr/bin/getconf", "GNU_LIBC_VERSION"))
    artifacts: dict[str, bytes] = {
        _tool_paths()["compiler_version_path"]: compiler_version,
        _tool_paths()["libc_path"]: _read_regular(
            libc_path, require_single_link=False
        ),
        _tool_paths()["libc_version_path"]: libc_version,
        _tool_paths()["linker_version_path"]: linker_version,
        _tool_paths()["source_commit_object_path"]: _run(
            (str(git), "-C", str(repository), "cat-file", "commit", head_object)
        ),
        _tool_paths()["source_head_path"]: head,
        _tool_paths()["source_status_path"]: status,
        _tool_paths()["source_tree_path"]: tree,
        _tool_paths()["target_triple_path"]: target_triple,
    }
    logical_values: list[dict[str, Any]] = []
    for logical in logical_elfs:
        logical_id = _identifier(logical.logical_id, "logical ELF ID")
        paths = _entry_paths(logical_id)
        _run(
            (
                str(ninja),
                "-C",
                str(build_directory),
                _target(logical.target, f"logical ELF {logical_id} target"),
            )
        )
        _require_source_snapshot(
            source_snapshot,
            git,
            repository,
            f"while constructing Ninja target {logical.target}",
        )
        elf_path = _resolve_under(
            build_directory, logical.elf_path, f"logical ELF {logical_id} image"
        )
        map_path = _resolve_under(
            build_directory, logical.link_map_path, f"logical ELF {logical_id} map"
        )
        build_output_path = _archive_path(
            elf_path.relative_to(build_directory).as_posix(),
            f"logical ELF {logical_id} build output",
        )
        build_map_path = _archive_path(
            map_path.relative_to(build_directory).as_posix(),
            f"logical ELF {logical_id} build map",
        )
        elf_raw = _read_regular(elf_path)
        parsed = _parse_elf(elf_raw)
        expected_wraps = sorted(
            option
            for api_id, _name, _wrapper, option in _SUPPORTED
            if any(row["api_id"] == api_id for row in parsed.origins)
        )
        commands = _ninja_final_command(
            ninja, build_directory, logical.target
        )
        command_raw = _link_command(
            commands,
            build_directory=build_directory_identity,
            build_output_path=build_output_path,
            build_map_path=build_map_path,
            expected_wraps=expected_wraps,
        )
        map_raw = _read_regular(map_path)
        _validate_link_map_output(
            map_raw,
            build_directory=build_directory_identity,
            build_output_path=build_output_path,
            where=f"logical ELF {logical_id} link map",
        )
        artifacts[paths["elf_path"]] = elf_raw
        artifacts[paths["link_command_path"]] = command_raw
        artifacts[paths["link_map_path"]] = map_raw
        logical_values.append(
            {
                "aliases": list(logical.aliases),
                "build_map_path": build_map_path,
                "build_output_path": build_output_path,
                "elf_kind_id": logical.elf_kind_id,
                "elf_path": paths["elf_path"],
                "link_command_path": paths["link_command_path"],
                "link_map_path": paths["link_map_path"],
                "logical_id": logical_id,
                "module_selection": _selection(logical.module_selection),
                "repo_path": logical.repo_path,
                "role_ids": list(logical.role_ids),
            }
        )
        _require_source_snapshot(
            source_snapshot,
            git,
            repository,
            f"while freezing artifacts for Ninja target {logical.target}",
        )
    provisional = {
        "api_definition_sha256": api_definition_sha256,
        "build_directory": build_directory_identity,
        "build_coverage_sha256": "0" * 64,
        "catalog_id": "oai_memprof_build_evidence",
        "entries": _exact_artifact_entries(artifacts),
        "logical_elfs": logical_values,
        "primary_logical_elf_id": primary_logical_elf_id,
        "toolchain": _tool_paths(),
        "version": VERSION,
    }
    coverage_raw = _derive(
        provisional,
        artifacts,
        api_definition_sha256=api_definition_sha256,
        enforce_advertised_digest=False,
    )
    evidence = dict(provisional)
    evidence["build_coverage_sha256"] = _sha(coverage_raw)
    evidence_raw = COVERAGE.canonical_bytes(evidence)
    _require(
        _derive(
            evidence,
            artifacts,
            api_definition_sha256=api_definition_sha256,
        )
        == coverage_raw,
        "build evidence: two-pass derivation mismatch",
    )
    _require_source_snapshot(
        source_snapshot,
        git,
        repository,
        "before evidence publication",
    )
    root_descriptor = _open_directory_path(evidence_root)
    try:
        current_root = os.fstat(root_descriptor)
        _require(
            (current_root.st_dev, current_root.st_ino, current_root.st_mode)
            == (initial_root.st_dev, initial_root.st_ino, initial_root.st_mode)
            and not os.listdir(root_descriptor),
            "evidence root identity or emptiness changed before publication",
        )
        for path, raw in artifacts.items():
            _publish_at(root_descriptor, path, raw)
        _publish_at(root_descriptor, EVIDENCE_ARCHIVE_PATH, evidence_raw)
        _publish_at(
            root_descriptor, COVERAGE.BUILD_COVERAGE_ARCHIVE_PATH, coverage_raw
        )
        os.fsync(root_descriptor)
    finally:
        os.close(root_descriptor)
    _require_source_snapshot(
        source_snapshot,
        git,
        repository,
        "after evidence publication",
    )
    root_post = _open_directory_path(evidence_root)
    try:
        final_root = os.fstat(root_post)
        _require(
            (final_root.st_dev, final_root.st_ino, final_root.st_mode)
            == (initial_root.st_dev, initial_root.st_ino, initial_root.st_mode),
            "evidence root path changed after publication",
        )
    finally:
        os.close(root_post)
    return PreparedBuildEvidence(
        coverage_raw,
        evidence_raw,
        tuple((path, raw) for path, raw in sorted(artifacts.items())),
    )

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=pathlib.Path)
    arguments = parser.parse_args(argv)
    request_raw = _read_regular(arguments.request.resolve(strict=True))
    request = COVERAGE.parse_canonical(request_raw)
    required = {
        "api_definition_sha256",
        "build_directory",
        "evidence_root",
        "libc_path",
        "logical_elfs",
        "primary_logical_elf_id",
        "repository",
    }
    _require(set(request) == required, "request: exact members required")
    prepared = prepare_measured_build_evidence(
        repository=pathlib.Path(request["repository"]),
        build_directory=pathlib.Path(request["build_directory"]),
        evidence_root=pathlib.Path(request["evidence_root"]),
        logical_elfs=tuple(
            _request_logical(value, index)
            for index, value in enumerate(request["logical_elfs"])
        ),
        primary_logical_elf_id=request["primary_logical_elf_id"],
        libc_path=pathlib.Path(request["libc_path"]),
        api_definition_sha256=request["api_definition_sha256"],
    )
    print(
        f"build evidence complete coverage_sha256={_sha(prepared.coverage_bytes)} "
        f"evidence_sha256={prepared.evidence_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
