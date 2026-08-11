#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Fail-closed ELF, archive, compile, and final-link gate for R0 A00/A01."""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import shlex
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Iterable, Sequence

from r0_harness_common import (
    BoundedFileError,
    BoundedFileImage,
    known_loader_environment_variables,
    loader_environment_variables,
    read_regular_file_bounded,
    run_process_bounded,
)


APIS = ("malloc", "calloc", "realloc", "free")
WRAPPERS = tuple(f"__wrap_{api}" for api in APIS)
REALS = tuple(f"__real_{api}" for api in APIS)
CONTROL = "oai_memprof_control_v1"
RUNTIME_VERSION = "OAI_MEMPROF_RUNTIME_1.0"
RUNTIME_SONAME = "liboai_memprof_runtime.so.1"
ROLE_RUNPATH = "$ORIGIN/.."
EXPECTED_NINJA_TARGETS = (
    "test_oai_memprof_r0_actual_a00",
    "test_oai_memprof_r0_actual_a01",
    "oai_memprof_r0_actual_dso_a00",
    "oai_memprof_r0_actual_dso_a01",
)
EXPECTED_R0_CTESTS = (
    "oai_memprof_r0_mutation_duplicate_real",
    "oai_memprof_r0_mutation_operand",
    "oai_memprof_r0_mutation_errno",
    "oai_memprof_r0_mutation_result",
    "oai_memprof_r0_mutation_suppress_free_null",
    "oai_memprof_r0_mutation_context",
    "oai_memprof_r0_scripted_pair",
    "oai_memprof_r0_scripted_bounds",
    "oai_memprof_r0_actual_a00",
    "oai_memprof_r0_actual_a01",
    "oai_memprof_r0_actual_differential",
    "oai_memprof_r0_elf_validator_selftest",
    "oai_memprof_r0_harness_selftest",
    "oai_memprof_r0_absence_validator_selftest",
    "oai_memprof_r0_elf",
)
EXPECTED_R0_MUTATIONS = (
    ("duplicate_real", "duplicate-real"),
    ("operand", "operand"),
    ("errno", "errno"),
    ("result", "result"),
    ("suppress_free_null", "suppress-free-null"),
    ("context", "context"),
)
EXPECTED_R0_CTEST_TIMEOUT_SECONDS = {
    test_name: 900 if test_name == "oai_memprof_r0_elf" else 120
    for test_name in EXPECTED_R0_CTESTS
}
PASSIVE_NEGATIVE_FLAGS = (
    "-fplt",
    "-fno-instrument-functions",
    "-fno-lto",
    "-fno-profile-arcs",
    "-fno-sanitize=all",
    "-fno-stack-protector",
    "-fno-test-coverage",
)
ACTUAL_NO_BUILTIN_FLAGS = tuple(f"-fno-builtin-{api}" for api in APIS)
OWNED_COMPILE_PATHS = (
    Path("common/utils/memprof/oai_memprof_runtime.c"),
    Path("common/utils/memprof/oai_memprof_wrap_malloc.c"),
    Path("common/utils/memprof/oai_memprof_wrap_calloc.c"),
    Path("common/utils/memprof/oai_memprof_wrap_realloc.c"),
    Path("common/utils/memprof/oai_memprof_wrap_free.c"),
    Path("common/utils/memprof/tests/r0_actual_fixture_exe.c"),
    Path("common/utils/memprof/tests/r0_actual_fixture_dso.c"),
)
MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
MAX_TOOL_STDOUT_BYTES = 16 * 1024 * 1024
MAX_TOOL_STDERR_BYTES = 1024 * 1024
MAX_COMMAND_TOKENS = 200_000
TOOL_TIMEOUT_SECONDS = 5.0
FORBIDDEN_LOADER_DYNAMIC_TAGS = frozenset({"AUDIT", "DEPAUDIT", "FILTER", "AUXILIARY"})

FORBIDDEN_CLOSURE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"^__asan_",
        r"^__hwasan_",
        r"^__lsan_",
        r"^__msan_",
        r"^__sanitizer_",
        r"^__tsan_",
        r"^__ubsan_",
        r"^__stack_chk_",
        r"^__cyg_profile_",
        r"^_?mcount$",
        r"^__gcov_",
        r"^__llvm_profile_",
        r"^__atomic_",
        r"^dlsym$",
        r"^dlopen$",
        r"^pthread_",
        r"^clock_gettime$",
        r"^syscall$",
        r"^(f?printf|fprintf|fopen|open|open64|write|pwrite|pwrite64)$",
        r"^oai_memprof_(wire|ring|writer|catalog|context|clock)",
    )
)


class GateError(RuntimeError):
    """A closed-gate result with one causal diagnostic."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def base_symbol(name: str) -> str:
    return name.split("@", 1)[0].split("+", 1)[0]


def read_bounded_image(path: Path, maximum: int = MAX_EVIDENCE_BYTES) -> BoundedFileImage:
    try:
        image = read_regular_file_bounded(path, maximum)
    except BoundedFileError as error:
        raise GateError(f"evidence read rejected [{error.code}]: {error.detail}") from error
    require(image.size > 0, f"empty evidence file: {path}")
    return image


def decode_strict_utf8(data: bytes, context: str) -> str:
    try:
        return data.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise GateError(f"{context} is not strict UTF-8: {error}") from error


def read_bounded_text(path: Path, maximum: int = MAX_EVIDENCE_BYTES) -> str:
    return decode_strict_utf8(read_bounded_image(path, maximum).data, f"evidence {path}")


@dataclasses.dataclass(frozen=True)
class ArtifactSnapshot:
    label: str
    source: Path
    path: Path
    source_image: BoundedFileImage
    snapshot_image: BoundedFileImage


def write_private_frozen_file(destination: Path, data: bytes, label: str) -> BoundedFileImage:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o400,
        )
        offset = 0
        view = memoryview(data)
        while offset < len(data):
            written = os.write(descriptor, view[offset:])
            require(written > 0, f"short snapshot write for {label}")
            offset += written
    except GateError:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise
    except OSError as error:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise GateError(f"cannot create exact snapshot for {label}: {error}") from error
    assert descriptor is not None
    try:
        os.close(descriptor)
    except OSError as error:
        raise GateError(f"cannot close exact snapshot for {label}: {error}") from error
    frozen = read_bounded_image(destination)
    require(
        (frozen.size, frozen.data) == (len(data), data),
        f"snapshot byte identity mismatch for {label}",
    )
    return frozen


def snapshot_regular_file(source: Path, root: Path, label: str) -> ArtifactSnapshot:
    """Freeze one input byte image so all external tools inspect the same bytes."""

    image = read_bounded_image(source)
    directory = root / label
    try:
        directory.mkdir(mode=0o700)
    except OSError as error:
        raise GateError(f"cannot create private snapshot directory for {label}: {error}") from error
    destination = directory / source.name
    frozen = write_private_frozen_file(destination, image.data, label)
    require(
        (frozen.size, frozen.sha256, frozen.data) == (image.size, image.sha256, image.data),
        f"snapshot byte identity mismatch for {label}",
    )
    return ArtifactSnapshot(label, source, destination, image, frozen)


def verify_file_image_unchanged(path: Path, expected: BoundedFileImage, context: str) -> None:
    current = read_bounded_image(path)
    require(
        (
            current.device,
            current.inode,
            current.size,
            current.mtime_ns,
            current.ctime_ns,
            current.sha256,
        )
        == (
            expected.device,
            expected.inode,
            expected.size,
            expected.mtime_ns,
            expected.ctime_ns,
            expected.sha256,
        ),
        f"file changed during validation: {context}: {path}",
    )


def verify_snapshot_source_unchanged(snapshot: ArtifactSnapshot) -> None:
    verify_file_image_unchanged(snapshot.path, snapshot.snapshot_image, f"snapshot {snapshot.label}")
    verify_file_image_unchanged(snapshot.source, snapshot.source_image, f"source {snapshot.label}")


def run_bounded_bytes(
    arguments: Sequence[str],
    *,
    stdout_limit: int = MAX_TOOL_STDOUT_BYTES,
    stderr_limit: int = MAX_TOOL_STDERR_BYTES,
    cwd: Path | None = None,
) -> bytes:
    """Run one admitted tool through the shared fail-closed process-group runner."""
    require(arguments, "empty tool command")
    require(cwd is None, "checker tool execution does not admit a mutable working directory")
    result = run_process_bounded(
        arguments,
        timeout_seconds=TOOL_TIMEOUT_SECONDS,
        max_stdout_bytes=stdout_limit,
        max_stderr_bytes=stderr_limit,
        remove_environment=loader_environment_variables(),
        environment_updates={"LANG": "C", "LC_ALL": "C"},
    )
    require(result.cleanup_complete, f"tool cleanup was not verified: {shlex.join(arguments)}")
    require(result.failure is None, f"tool execution rejected: {shlex.join(arguments)}: {result.failure}")
    error_text = decode_strict_utf8(result.stderr, f"tool stderr for {shlex.join(arguments)}")
    require(result.returncode == 0, f"tool failed ({result.returncode}): {shlex.join(arguments)}\n{error_text}")
    require(not result.stderr, f"tool emitted stderr: {shlex.join(arguments)}\n{error_text}")
    return result.stdout


def run_bounded_text(arguments: Sequence[str], *, cwd: Path | None = None) -> str:
    return decode_strict_utf8(
        run_bounded_bytes(arguments, cwd=cwd),
        f"tool stdout for {shlex.join(arguments)}",
    )


@dataclasses.dataclass(frozen=True)
class ElfHeader:
    elf_class: str
    data: str
    elf_type: str
    machine_text: str
    architecture: str
    entry: int


def parse_header(text: str) -> ElfHeader:
    names = ("Class", "Data", "Type", "Machine", "Entry point address")
    fields: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^\s*(" + "|".join(re.escape(name) for name in names) + r"):\s+(.+?)\s*$", line)
        if match:
            require(match.group(1) not in fields, f"duplicate ELF header field: {match.group(1)}")
            fields[match.group(1)] = match.group(2)
    require(set(fields) == set(names), f"incomplete ELF header output: {sorted(fields)}")
    require(fields["Class"] == "ELF64", f"unsupported ELF class: {fields['Class']}")
    require(fields["Data"] == "2's complement, little endian", f"unsupported ELF byte order: {fields['Data']}")
    machine = fields["Machine"]
    if machine == "Advanced Micro Devices X86-64":
        architecture = "x86_64"
    elif machine == "AArch64":
        architecture = "aarch64"
    else:
        raise GateError(f"unsupported ELF machine: {machine}")
    type_match = re.match(r"^(\S+)\s+\(", fields["Type"])
    require(type_match is not None, f"malformed ELF type: {fields['Type']}")
    try:
        entry = int(fields["Entry point address"], 0)
    except ValueError as error:
        raise GateError(f"malformed ELF entry point: {fields['Entry point address']}") from error
    return ElfHeader(fields["Class"], fields["Data"], type_match.group(1), machine, architecture, entry)


@dataclasses.dataclass(frozen=True)
class Symbol:
    ordinal: int
    value: int
    size: int
    kind: str
    binding: str
    visibility: str
    index: str
    name: str

    @property
    def base_name(self) -> str:
        return base_symbol(self.name)

    @property
    def defined(self) -> bool:
        return self.index != "UND"


def parse_symbols(text: str, table_name: str) -> list[Symbol]:
    header_pattern = re.compile(r"^Symbol table '([^']+)' contains (\d+) entries:$")
    row_pattern = re.compile(
        r"^\s*(\d+):\s+([0-9A-Fa-f]+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)(?:\s+(.*?))?\s*$"
    )
    active = False
    found = False
    expected = -1
    saw_columns = False
    symbols: list[Symbol] = []
    for line in text.splitlines():
        header = header_pattern.match(line)
        if header:
            if active:
                require(len(symbols) == expected, f"{table_name} symbol count mismatch: {len(symbols)} != {expected}")
            active = header.group(1) == table_name
            if active:
                require(not found, f"duplicate symbol table {table_name}")
                found = True
                expected = int(header.group(2))
                symbols = []
                saw_columns = False
            continue
        if not active:
            continue
        if line.lstrip().startswith("Num:"):
            require(not saw_columns, f"duplicate column header in {table_name}")
            saw_columns = True
            continue
        if not line.strip():
            if saw_columns:
                require(len(symbols) == expected, f"{table_name} symbol count mismatch: {len(symbols)} != {expected}")
                active = False
            continue
        require(saw_columns, f"symbol row before columns in {table_name}: {line}")
        match = row_pattern.match(line)
        require(match is not None, f"malformed {table_name} symbol row: {line}")
        ordinal = int(match.group(1))
        require(ordinal == len(symbols), f"non-sequential {table_name} symbol ordinal: {ordinal}")
        name = match.group(8) or ""
        name = re.sub(r"\s+\(\d+\)$", "", name)
        symbols.append(
            Symbol(
                ordinal=ordinal,
                value=int(match.group(2), 16),
                size=int(match.group(3)),
                kind=match.group(4),
                binding=match.group(5),
                visibility=match.group(6),
                index=match.group(7),
                name=name,
            )
        )
    require(found, f"missing symbol table {table_name}")
    require(len(symbols) == expected, f"{table_name} symbol count mismatch: {len(symbols)} != {expected}")
    return symbols


@dataclasses.dataclass(frozen=True)
class Section:
    index: int
    name: str
    kind: str
    address: int
    offset: int
    size: int
    entry_size: int
    flags: str
    link: int
    info: int
    alignment: int


def parse_sections(text: str) -> list[Section]:
    count_match = re.search(r"There are (\d+) section headers", text)
    require(count_match is not None, "missing section-header count")
    expected = int(count_match.group(1))
    row_pattern = re.compile(
        r"^\s*\[\s*(\d+)\]\s+(\S+)\s+(\S+)\s+([0-9A-Fa-f]+)\s+([0-9A-Fa-f]+)\s+"
        r"([0-9A-Fa-f]+)\s+([0-9A-Fa-f]+)\s*([A-Z]*)\s+(\d+)\s+(\d+)\s+(\d+)\s*$"
    )
    sections: list[Section] = []
    for line in text.splitlines():
        match = row_pattern.match(line)
        if match is None:
            continue
        index = int(match.group(1))
        if index == 0:
            continue
        sections.append(
            Section(
                index=index,
                name=match.group(2),
                kind=match.group(3),
                address=int(match.group(4), 16),
                offset=int(match.group(5), 16),
                size=int(match.group(6), 16),
                entry_size=int(match.group(7), 16),
                flags=match.group(8),
                link=int(match.group(9)),
                info=int(match.group(10)),
                alignment=int(match.group(11)),
            )
        )
    require(len(sections) == expected - 1, f"section count mismatch: {len(sections) + 1} != {expected}")
    require([section.index for section in sections] == list(range(1, expected)), "non-sequential section indices")
    require(len({section.name for section in sections}) == len(sections), "duplicate section name")
    return sections


@dataclasses.dataclass(frozen=True)
class DynamicEntry:
    tag: str
    value: str


def parse_dynamic(text: str) -> list[DynamicEntry]:
    header_pattern = re.compile(
        r"^Dynamic section at offset 0x[0-9A-Fa-f]+ contains (\d+) entr(?:y|ies):$"
    )
    column_pattern = re.compile(r"^\s*Tag\s+Type\s+Name/Value\s*$")
    row_pattern = re.compile(r"^\s*0x[0-9A-Fa-f]+\s+\(([A-Z0-9_]+)\)\s+(.+?)\s*$")
    entries: list[DynamicEntry] = []
    declared_count: int | None = None
    saw_columns = False
    saw_terminal_null = False
    for line in text.splitlines():
        if not line.strip():
            continue
        header = header_pattern.fullmatch(line)
        if line.startswith("Dynamic section"):
            require(header is not None, f"malformed dynamic-section header: {line}")
        if header is not None:
            require(declared_count is None, "duplicate dynamic-section header")
            declared_count = int(header.group(1))
            continue
        if column_pattern.fullmatch(line):
            require(declared_count is not None, "dynamic column header precedes section header")
            require(not saw_columns and not entries, "duplicate or late dynamic column header")
            saw_columns = True
            continue
        row = row_pattern.fullmatch(line)
        if row is not None:
            require(declared_count is not None and saw_columns, f"dynamic row precedes complete header: {line}")
            require(not saw_terminal_null, f"dynamic row follows terminal DT_NULL: {line}")
            entry = DynamicEntry(row.group(1), row.group(2))
            entries.append(entry)
            saw_terminal_null = entry.tag == "NULL"
            require(len(entries) <= declared_count, "dynamic table has more rows than declared")
            continue
        raise GateError(f"unexpected dynamic-section output: {line}")
    require(declared_count is not None, "missing dynamic-section header")
    require(saw_columns, "missing dynamic column header")
    require(declared_count > 0, "dynamic section declares no DT_NULL row")
    require(len(entries) == declared_count, f"dynamic entry count mismatch: {len(entries)} != {declared_count}")
    require(saw_terminal_null and entries[-1].tag == "NULL", "dynamic table does not terminate with DT_NULL")
    require(sum(entry.tag == "NULL" for entry in entries) == 1, "dynamic table DT_NULL multiplicity is not one")
    return entries


def bracket_values(entries: Sequence[DynamicEntry], tag: str) -> list[str]:
    values = []
    for entry in entries:
        if entry.tag != tag:
            continue
        match = re.search(r"\[([^]]*)\]", entry.value)
        require(match is not None, f"malformed {tag} dynamic entry: {entry.value}")
        values.append(match.group(1))
    return values


def require_exact_dynamic_loader_policy(
    entries: Sequence[DynamicEntry],
    context: str,
    *,
    expected_flags: Sequence[str],
    expected_flags_1: Sequence[str],
) -> None:
    forbidden = [entry.tag for entry in entries if entry.tag in FORBIDDEN_LOADER_DYNAMIC_TAGS]
    require(not forbidden, f"{context} contains loader injection tags: {forbidden}")
    flags = [entry.value for entry in entries if entry.tag == "FLAGS"]
    flags_1 = [entry.value for entry in entries if entry.tag == "FLAGS_1"]
    require(flags == list(expected_flags), f"{context} DT_FLAGS are not exact: {flags}")
    require(flags_1 == list(expected_flags_1), f"{context} DT_FLAGS_1 are not exact: {flags_1}")


@dataclasses.dataclass(frozen=True)
class Relocation:
    section: str
    offset: int
    kind: str
    symbol: str

    @property
    def base_name(self) -> str:
        return base_symbol(self.symbol)


def parse_relocations(text: str) -> list[Relocation]:
    nonempty = [line for line in text.splitlines() if line.strip()]
    if nonempty == ["There are no relocations in this file."]:
        return []
    require("There are no relocations in this file." not in nonempty, "contradictory relocation output")
    header_pattern = re.compile(
        r"^Relocation section '([^']+)' at offset 0x[0-9A-Fa-f]+ contains (\d+) entr(?:y|ies):$"
    )
    column_pattern = re.compile(
        r"^\s*Offset\s+Info\s+Type\s+Symbol's Value\s+Symbol's Name \+ Addend\s*$"
    )
    row_pattern = re.compile(r"^\s*([0-9A-Fa-f]+)\s+([0-9A-Fa-f]+)\s+(R_\S+)(?:\s+(.+?))?\s*$")
    relocations: list[Relocation] = []
    current_section: str | None = None
    declared_count: int | None = None
    parsed_count = 0
    saw_columns = False
    found_section = False
    for line in text.splitlines():
        if not line.strip():
            continue
        header = header_pattern.match(line)
        if line.startswith("Relocation section '"):
            require(header is not None, f"malformed relocation-section header: {line}")
        if header:
            if current_section is not None:
                require(
                    parsed_count == declared_count,
                    f"relocation count mismatch in {current_section}: {parsed_count} != {declared_count}",
                )
                require(saw_columns, f"missing relocation column header in {current_section}")
            current_section = header.group(1)
            declared_count = int(header.group(2))
            parsed_count = 0
            saw_columns = False
            found_section = True
            continue
        if column_pattern.fullmatch(line):
            require(current_section is not None, "relocation column header precedes section header")
            require(not saw_columns and parsed_count == 0, f"duplicate or late relocation columns in {current_section}")
            saw_columns = True
            continue
        row = row_pattern.fullmatch(line)
        if row is None:
            raise GateError(f"unexpected relocation output in {current_section}: {line}")
        require(current_section is not None and declared_count is not None and saw_columns,
                f"relocation row precedes complete header: {line}")
        require(parsed_count < declared_count,
                f"extra relocation row beyond declared count in {current_section}: {line}")
        parts = line.split()
        require(len(parts) >= 4, f"malformed relocation row in {current_section}: {line}")
        require(parts[0] == row.group(1) and parts[1] == row.group(2) and parts[2] == row.group(3),
                f"ambiguous relocation row in {current_section}: {line}")
        if len(parts) == 4:
            require(re.fullmatch(r"-?[0-9A-Fa-f]+", parts[3]) is not None,
                    f"malformed symbol-free relocation addend in {current_section}: {line}")
            symbol = ""
        else:
            require(
                len(parts) == 7
                and re.fullmatch(r"[0-9A-Fa-f]+", parts[3]) is not None
                and parts[4]
                and parts[5] in {"+", "-"}
                and re.fullmatch(r"[0-9A-Fa-f]+", parts[6]) is not None,
                f"malformed symbol relocation fields in {current_section}: {line}",
            )
            symbol = parts[4]
        relocations.append(Relocation(current_section, int(row.group(1), 16), row.group(3), symbol))
        parsed_count += 1
    require(found_section and current_section is not None and declared_count is not None,
            "could not parse relocation output")
    require(saw_columns, f"missing relocation column header in {current_section}")
    require(
        parsed_count == declared_count,
        f"relocation count mismatch in {current_section}: {parsed_count} != {declared_count}",
    )
    return relocations


@dataclasses.dataclass(frozen=True)
class Instruction:
    address: int
    mnemonic: str
    operands: str


def parse_instructions(text: str, symbol: str) -> list[Instruction]:
    header = re.compile(rf"^\s*[0-9A-Fa-f]+\s+<{re.escape(symbol)}>:\s*$")
    row = re.compile(r"^\s*([0-9A-Fa-f]+):\s+(\S+)(?:\s+(.*?))?\s*$")
    active = False
    found = False
    instructions: list[Instruction] = []
    for line in text.splitlines():
        if header.match(line):
            require(not found, f"duplicate disassembly for {symbol}")
            found = True
            active = True
            continue
        if not active:
            continue
        if re.match(r"^\s*[0-9A-Fa-f]+\s+<[^>]+>:\s*$", line):
            break
        match = row.match(line)
        if match:
            instructions.append(Instruction(int(match.group(1), 16), match.group(2), match.group(3) or ""))
            continue
        if line.strip() and line.startswith(" "):
            raise GateError(f"malformed instruction row for {symbol}: {line}")
    require(found and instructions, f"no instructions parsed for {symbol}")
    require(
        all(left.address < right.address for left, right in zip(instructions, instructions[1:])),
        f"non-increasing instruction addresses for {symbol}",
    )
    return instructions


def direct_transfer_target(instruction: Instruction, architecture: str) -> str | None:
    if architecture == "x86_64":
        is_transfer = instruction.mnemonic.startswith("call") or instruction.mnemonic.startswith("jmp")
    elif architecture == "aarch64":
        is_transfer = (
            instruction.mnemonic in {"b", "bl", "br", "blr"}
            or instruction.mnemonic.startswith("b.")
        )
    else:
        raise GateError(f"unsupported disassembly architecture: {architecture}")
    if not is_transfer:
        return None
    match = re.search(r"<([^>]+)>", instruction.operands)
    require(
        match is not None,
        f"indirect or unresolved transfer rejected: {instruction.mnemonic} {instruction.operands}",
    )
    require(
        instruction.mnemonic not in {"br", "blr"},
        f"indirect AArch64 transfer rejected: {instruction.mnemonic} {instruction.operands}",
    )
    return base_symbol(match.group(1))


def extract_transfers(instructions: Sequence[Instruction], architecture: str) -> list[str]:
    transfers = []
    for instruction in instructions:
        target = direct_transfer_target(instruction, architecture)
        if target is not None:
            transfers.append(target)
    return transfers


def strip_landing_pad(instructions: Sequence[Instruction], architecture: str) -> list[Instruction]:
    result = list(instructions)
    if result and architecture == "x86_64" and result[0].mnemonic == "endbr64":
        result.pop(0)
    if result and architecture == "aarch64" and result[0].mnemonic == "bti":
        require(result[0].operands in {"c", "jc"}, f"unexpected BTI form: {result[0].operands}")
        result.pop(0)
    return result


def wrapper_instruction_signature(
    instructions: Sequence[Instruction],
    architecture: str,
    *,
    api: str,
    expected_final_target: str | None,
) -> tuple[str, ...]:
    require(api in APIS, f"unsupported wrapper API for register proof: {api}")
    argument_count = 2 if api in {"calloc", "realloc"} else 1
    body = strip_landing_pad(instructions, architecture)
    if architecture == "x86_64":
        require([item.mnemonic for item in body] == ["mov", "mov", "jmp"], f"invalid x86-64 wrapper shape: {body}")
        got_load = re.fullmatch(
            r".*\(%rip\),\s*(%[a-z0-9]+)(?:\s+#\s+.+)?",
            body[0].operands,
        )
        value_load = re.fullmatch(r"\((%[a-z0-9]+)\),\s*(%[a-z0-9]+)", body[1].operands)
        require(got_load is not None, f"x86-64 wrapper lacks one exact GOT-address load: {body[0]}")
        require(value_load is not None, f"x86-64 wrapper lacks one exact control-value load: {body[1]}")
        scratch = got_load.group(1)
        require(
            scratch == value_load.group(1) == value_load.group(2),
            f"x86-64 wrapper does not carry the control pointer/value through one register: {body[:2]}",
        )
        argument_registers = ("%rdi", "%rsi")[:argument_count]
        require(scratch not in argument_registers,
                f"x86-64 wrapper clobbers an API argument register: {scratch}/{argument_registers}")
    elif architecture == "aarch64":
        require([item.mnemonic for item in body] == ["adrp", "ldr", "ldar", "b"],
                f"invalid AArch64 wrapper shape: {body}")
        adrp = re.match(r"^(x\d+),", body[0].operands)
        got_load = re.fullmatch(
            r"(x\d+),\s*\[(x\d+)(?:,\s*#(?:0x)?[0-9A-Fa-f]+)?\](?:\s+//.*)?",
            body[1].operands,
        )
        value_load = re.match(r"^([xw]\d+),\s*\[(x\d+)\]$", body[2].operands)
        require(adrp is not None and got_load is not None and value_load is not None,
                f"malformed AArch64 wrapper data-load operands: {body}")
        scratch = adrp.group(1)
        require(
            scratch
            == got_load.group(1)
            == got_load.group(2)
            == value_load.group(1)
            == value_load.group(2),
            f"AArch64 wrapper does not carry the control pointer/value through one register: {body[:3]}",
        )
        argument_registers = tuple(f"x{index}" for index in range(argument_count))
        require(scratch not in argument_registers,
                f"AArch64 wrapper clobbers an API argument register: {scratch}/{argument_registers}")
    else:
        raise GateError(f"unsupported wrapper architecture: {architecture}")
    transfers = extract_transfers(body, architecture)
    require(len(transfers) == 1, f"wrapper does not contain exactly one direct transfer: {transfers}")
    if expected_final_target is not None:
        require(transfers == [expected_final_target], f"wrapper routes {transfers}, expected [{expected_final_target}]")
    return (*tuple(item.mnemonic for item in instructions), f"scratch={scratch}")


def validate_callsite_instructions(
    instructions: Sequence[Instruction], architecture: str, expected_target: str
) -> None:
    body = strip_landing_pad(instructions, architecture)
    expected_mnemonic = "jmp" if architecture == "x86_64" else "b"
    require(len(body) == 1 and body[0].mnemonic == expected_mnemonic, f"invalid callsite shape: {body}")
    require(extract_transfers(body, architecture) == [expected_target],
            f"callsite does not route exactly to {expected_target}: {body}")


def forbidden_symbols(symbols: Iterable[Symbol]) -> set[str]:
    result: set[str] = set()
    for symbol in symbols:
        if any(pattern.search(symbol.base_name) for pattern in FORBIDDEN_CLOSURE_PATTERNS):
            result.add(symbol.base_name)
    return result


def reject_positive_instrumentation(tokens: Sequence[str], context: str) -> None:
    rejected: list[str] = []
    for token in tokens:
        if token.startswith("-fno-"):
            continue
        if (
            token == "-flto"
            or token.startswith("-flto=")
            or token == "-fuse-linker-plugin"
            or token in {"--coverage", "-coverage", "-p", "-pg", "-finstrument-functions", "-ftest-coverage"}
            or token.startswith("-fsanitize=")
            or token.startswith("-fsanitize-coverage=")
            or token.startswith("-fstack-protector")
            or token.startswith("-fprofile-")
            or token.startswith("-fauto-profile")
            or token.startswith("-fplugin")
            or token.startswith("-fpatchable-function-entry")
            or token == "-fcoverage-mapping"
        ):
            rejected.append(token)
    for argument in normalized_linker_arguments(tokens):
        if argument in {"-plugin", "--plugin"} or argument.startswith("-plugin-opt"):
            rejected.append(argument)
    require(not rejected, f"{context} enables instrumentation/LTO: {rejected}")


def require_passive_flags(tokens: Sequence[str], context: str) -> None:
    for flag in PASSIVE_NEGATIVE_FLAGS:
        require(tokens.count(flag) == 1, f"{context} requires exactly one {flag}, found {tokens.count(flag)}")
    reject_positive_instrumentation(tokens, context)


def require_no_linker_driver_override(tokens: Sequence[str], context: str) -> None:
    rejected = [
        token
        for token in tokens
        if token == "-fuse-ld"
        or token.startswith("-fuse-ld=")
        or token == "-B"
        or (token.startswith("-B") and len(token) > 2)
        or token in {"-specs", "--specs", "-wrapper", "--wrapper"}
        or token.startswith("-specs=")
        or token.startswith("--specs=")
        or token.startswith("-wrapper=")
        or token.startswith("--wrapper=")
    ]
    linker_arguments = normalized_linker_arguments(tokens)
    loader_options = {"--audit", "-P", "--depaudit", "--filter", "-F", "--auxiliary", "-f"}
    loader_prefixes = ("--audit=", "--depaudit=", "--filter=", "--auxiliary=")
    for index, argument in enumerate(linker_arguments):
        if argument in loader_options or argument.startswith(loader_prefixes):
            rejected.append(argument)
        if argument == "-z" and index + 1 < len(linker_arguments):
            if linker_arguments[index + 1] in {"interpose", "nodefaultlib"}:
                rejected.extend((argument, linker_arguments[index + 1]))
        if argument in {"-zinterpose", "-znodefaultlib", "-z=interpose", "-z=nodefaultlib"}:
            rejected.append(argument)
    require(not rejected, f"{context} contains a toolchain/loader override or injection: {rejected}")


def normalized_linker_arguments(tokens: Sequence[str]) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("-Wl,"):
            result.extend(part for part in token[4:].split(",") if part)
        elif token == "-Xlinker":
            require(index + 1 < len(tokens), "dangling -Xlinker option")
            index += 1
            result.append(tokens[index])
        elif token.startswith("--wrap="):
            result.append(token)
        index += 1
    return result


def wrapped_symbols(tokens: Sequence[str]) -> list[str]:
    pattern = re.compile(r"^-Wl,--wrap=([A-Za-z_][A-Za-z0-9_]*)$")
    result: list[str] = []
    malformed: list[str] = []
    for token in tokens:
        match = pattern.fullmatch(token)
        if match is not None:
            result.append(match.group(1))
        elif "--wrap" in token:
            malformed.append(token)
    require(not malformed, f"GNU wrap must use standalone -Wl,--wrap=<symbol> tokens: {malformed}")
    return result


def output_path(tokens: Sequence[str], base_directory: Path) -> Path | None:
    for index, token in enumerate(tokens):
        if token == "-o":
            require(index + 1 < len(tokens), "dangling -o option")
            candidate = Path(tokens[index + 1])
            return (base_directory / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        if token.startswith("-o") and len(token) > 2:
            candidate = Path(token[2:])
            return (base_directory / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    return None


def expand_response_files(
    tokens: Sequence[str],
    base_directory: Path,
    *,
    depth: int = 0,
    active: frozenset[Path] = frozenset(),
) -> list[str]:
    require(depth <= 2, "nested response-file depth exceeds two")
    expanded: list[str] = []
    for token in tokens:
        if not token.startswith("@"):
            expanded.append(token)
            continue
        response = Path(token[1:])
        if not response.is_absolute():
            response = base_directory / response
        response = response.resolve()
        require(response not in active, f"response-file cycle: {response}")
        try:
            response_tokens = shlex.split(read_bounded_text(response), posix=True)
        except ValueError as error:
            raise GateError(f"cannot parse response file {response}: {error}") from error
        expanded.extend(
            expand_response_files(
                response_tokens,
                response.parent,
                depth=depth + 1,
                active=active | {response},
            )
        )
        require(len(expanded) <= MAX_COMMAND_TOKENS, "expanded command exceeds token bound")
    return expanded


def parse_command_lines(evidence: str, base_directory: Path) -> list[list[str]]:
    commands: list[list[str]] = []
    for line in evidence.splitlines():
        if not line.strip():
            continue
        try:
            raw = shlex.split(line, posix=True)
        except ValueError as error:
            raise GateError(f"cannot parse Ninja command line: {error}") from error
        tokens = expand_response_files(raw, base_directory)
        require(len(tokens) <= MAX_COMMAND_TOKENS, "Ninja command exceeds token bound")
        commands.append(tokens)
    require(commands, "empty Ninja command evidence")
    return commands


def find_output_command(commands: Sequence[Sequence[str]], output: Path, base_directory: Path) -> list[str]:
    matches = [list(tokens) for tokens in commands if output_path(tokens, base_directory) == output.resolve()]
    require(len(matches) == 1, f"expected one command producing {output}, found {len(matches)}")
    return matches[0]


def resolved_token_indices(tokens: Sequence[str], target: Path, base_directory: Path) -> list[int]:
    expected = target.resolve()
    result = []
    for index, token in enumerate(tokens):
        if token.startswith("-") or token in {":", "&&", "||", ";"}:
            continue
        candidate = Path(token)
        candidate = (base_directory / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        if candidate == expected:
            result.append(index)
    return result


def explicit_object_inputs(tokens: Sequence[str]) -> tuple[str, ...]:
    objects = tuple(os.path.normpath(token) for token in tokens if token.endswith((".o", ".obj")))
    require(objects, "final link command has no explicit fixture object")
    return objects


def _map_token_matches(token: str, map_path: Path, base_directory: Path) -> bool:
    if token.startswith("-Wl,-Map,"):
        candidate_text = token[len("-Wl,-Map,") :]
    elif token.startswith("-Map="):
        candidate_text = token[len("-Map=") :]
    else:
        return False
    candidate = Path(candidate_text)
    candidate = (base_directory / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    return candidate == map_path.resolve()


@dataclasses.dataclass(frozen=True)
class Role:
    name: str
    path: Path
    map_path: Path
    wrapped: bool
    dso: bool
    ninja_target: str


def normalize_final_link(
    role: Role,
    tokens: Sequence[str],
    wrapper_archive: Path,
    runtime: Path,
    base_directory: Path,
) -> tuple[str, ...]:
    require_passive_flags(tokens, f"{role.name} final link")
    actual_wraps = wrapped_symbols(tokens)
    require(actual_wraps == (list(APIS) if role.wrapped else []),
            f"{role.name} wrap order/multiplicity mismatch: {actual_wraps}")
    wrap_indices = {
        index for index, token in enumerate(tokens) if re.fullmatch(r"-Wl,--wrap=[A-Za-z_][A-Za-z0-9_]*", token)
    }
    archive_indices = resolved_token_indices(tokens, wrapper_archive, base_directory)
    runtime_indices = resolved_token_indices(tokens, runtime, base_directory)
    require(len(archive_indices) == (1 if role.wrapped else 0),
            f"{role.name} wrapper-archive multiplicity is {len(archive_indices)}")
    require(len(runtime_indices) == (1 if role.wrapped else 0),
            f"{role.name} runtime multiplicity is {len(runtime_indices)}")
    if role.wrapped:
        require(archive_indices[0] < runtime_indices[0], f"{role.name} runtime precedes wrapper archive")

    normalized: list[str] = []
    skip_next = False
    removed_output = 0
    removed_map = 0
    profiler_indices = set(archive_indices + runtime_indices) | wrap_indices
    for index, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if token == "-o":
            require(index + 1 < len(tokens), f"{role.name} has dangling -o")
            normalized.extend(("-o", "<ROLE_OUTPUT>"))
            skip_next = True
            removed_output += 1
            continue
        if token.startswith("-o") and len(token) > 2:
            normalized.append("-o<ROLE_OUTPUT>")
            removed_output += 1
            continue
        if _map_token_matches(token, role.map_path, base_directory):
            normalized.append("-Wl,-Map,<ROLE_MAP>")
            removed_map += 1
            continue
        if role.wrapped and index in profiler_indices:
            continue
        normalized.append(token)
    require(removed_output == 1, f"{role.name} output normalization count is {removed_output}")
    require(removed_map == 1, f"{role.name} map normalization count is {removed_map}")
    return tuple(normalized)


@dataclasses.dataclass
class Toolchain:
    readelf: str
    nm: str
    objdump: str
    ar: str
    ninja: str
    ld: str

    @staticmethod
    def resolve(name: str) -> str:
        path = shutil.which(name) if os.sep not in name else name
        require(path is not None and Path(path).is_file(), f"tool not found: {name}")
        return str(Path(path).resolve())

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Toolchain":
        return cls(*(cls.resolve(getattr(args, field.name)) for field in dataclasses.fields(cls)))

    def verify_versions(self) -> None:
        for path, prefix in (
            (self.readelf, "GNU readelf"),
            (self.nm, "GNU nm"),
            (self.objdump, "GNU objdump"),
            (self.ar, "GNU ar"),
            (self.ld, "GNU ld"),
        ):
            first = run_bounded_text((path, "--version")).splitlines()
            require(first and first[0].startswith(prefix), f"non-GNU or unexpected tool identity: {path}: {first[:1]}")
            require(re.search(r"\d+\.\d+", first[0]) is not None, f"missing GNU tool version: {first[0]}")
        ninja_version = run_bounded_text((self.ninja, "--version")).strip()
        require(re.fullmatch(r"\d+\.\d+(?:\.\d+)?", ninja_version) is not None,
                f"unexpected Ninja version output: {ninja_version}")

    def header(self, path: Path) -> ElfHeader:
        return parse_header(run_bounded_text((self.readelf, "-hW", str(path))))

    def sections(self, path: Path) -> tuple[list[Section], str]:
        output = run_bounded_text((self.readelf, "-SW", str(path)))
        return parse_sections(output), output

    def program_headers(self, path: Path) -> str:
        return run_bounded_text((self.readelf, "-lW", str(path)))

    def dynamic(self, path: Path) -> list[DynamicEntry]:
        return parse_dynamic(run_bounded_text((self.readelf, "-dW", str(path))))

    def relocations(self, path: Path) -> list[Relocation]:
        return parse_relocations(run_bounded_text((self.readelf, "-rW", str(path))))

    def symtab(self, path: Path) -> list[Symbol]:
        return parse_symbols(run_bounded_text((self.readelf, "-sW", str(path))), ".symtab")

    def dynsym(self, path: Path) -> list[Symbol]:
        return parse_symbols(run_bounded_text((self.readelf, "-sW", str(path))), ".dynsym")

    def version_info(self, path: Path) -> str:
        return run_bounded_text((self.readelf, "-VW", str(path)))

    def disassemble(self, path: Path, symbol: str) -> list[Instruction]:
        output = run_bounded_text(
            (self.objdump, "-d", "--no-show-raw-insn", f"--disassemble={symbol}", str(path))
        )
        return parse_instructions(output, symbol)


def require_no_tls_or_lto(toolchain: Toolchain, path: Path) -> list[Section]:
    sections, section_output = toolchain.sections(path)
    names = {section.name for section in sections}
    require(not ({".tdata", ".tbss"} & names), f"TLS section present in {path}")
    require(not any(name.startswith(".gnu.lto_") for name in names), f"LTO section present in {path}")
    require(not any("T" in section.flags for section in sections), f"TLS section flag present in {path}")
    require(" TLS " not in toolchain.program_headers(path), f"PT_TLS present in {path}")
    require("LTO" not in section_output, f"unrecognized LTO marker in {path}")
    return sections


def require_clean_relocations(toolchain: Toolchain, path: Path) -> list[Relocation]:
    relocations = toolchain.relocations(path)
    for relocation in relocations:
        require("COPY" not in relocation.kind, f"COPY relocation present in {path}: {relocation}")
        require(
            not re.search(r"TLS|TPOFF|DTPMOD|DTPREL|TLSDESC", relocation.kind),
            f"TLS relocation present in {path}: {relocation}",
        )
    return relocations


def validate_runtime(toolchain: Toolchain, runtime: Path) -> tuple[str, str]:
    header = toolchain.header(runtime)
    require(header.elf_type == "DYN" and header.entry == 0, f"runtime is not an entryless ET_DYN: {header}")
    sections = require_no_tls_or_lto(toolchain, runtime)
    relocations = require_clean_relocations(toolchain, runtime)
    require(not relocations, f"data-only runtime has relocations: {relocations}")
    program_headers = toolchain.program_headers(runtime)
    require("INTERP" not in program_headers, "runtime unexpectedly has PT_INTERP")
    require(not re.search(r"^\s*LOAD\s+.*\bR E\b", program_headers, re.MULTILINE),
            "runtime has an executable PT_LOAD")
    executable_sections = [
        section.name
        for section in sections
        if section.size and "A" in section.flags and "X" in section.flags
    ]
    require(not executable_sections, f"runtime has allocated executable sections: {executable_sections}")

    dynamic = toolchain.dynamic(runtime)
    require_exact_dynamic_loader_policy(
        dynamic,
        "data-only runtime",
        expected_flags=(),
        expected_flags_1=(),
    )
    needed = bracket_values(dynamic, "NEEDED")
    sonames = bracket_values(dynamic, "SONAME")
    require(not needed, f"data-only runtime has DT_NEEDED dependencies: {needed}")
    require(sonames == [RUNTIME_SONAME], f"runtime SONAME is not exact: {sonames}")
    require(not bracket_values(dynamic, "RPATH"), "runtime has DT_RPATH")
    require(not bracket_values(dynamic, "RUNPATH"), "runtime has DT_RUNPATH")
    forbidden_tags = {"INIT", "FINI", "INIT_ARRAY", "INIT_ARRAYSZ", "FINI_ARRAY", "FINI_ARRAYSZ"}
    present_forbidden = [entry.tag for entry in dynamic if entry.tag in forbidden_tags]
    require(not present_forbidden, f"data-only runtime has lifecycle tags: {present_forbidden}")

    dynsym = toolchain.dynsym(runtime)
    symtab = toolchain.symtab(runtime)
    controls = [symbol for symbol in dynsym if symbol.name == f"{CONTROL}@@{RUNTIME_VERSION}" and symbol.defined]
    require(len(controls) == 1, f"runtime lacks one exact default-version control definition: {controls}")
    control = controls[0]
    require(
        (control.kind, control.binding, control.visibility, control.size)
        == ("OBJECT", "GLOBAL", "PROTECTED", 8),
        f"invalid runtime control ABI: {control}",
    )
    require(control.value % 64 == 0, f"runtime control is not 64-byte aligned: {control.value:#x}")
    require(not any(symbol.name == f"{CONTROL}@{RUNTIME_VERSION}" for symbol in dynsym),
            "runtime has a non-default control version alias")
    control_section = next(
        (section for section in sections if str(section.index) == control.index),
        None,
    )
    require(control_section is not None, f"runtime control has invalid section index: {control.index}")
    require(
        control_section.name == ".bss"
        and control_section.kind == "NOBITS"
        and control_section.address == control.value
        and control_section.size == 8
        and control_section.alignment >= 64
        and "W" in control_section.flags
        and "A" in control_section.flags
        and "X" not in control_section.flags,
        f"runtime control is not the sole aligned zero-fill object: {control_section}",
    )
    defined_functions = [symbol for symbol in symtab if symbol.defined and symbol.kind == "FUNC"]
    require(not defined_functions, f"data-only runtime defines executable functions: {defined_functions}")
    application_objects = [
        symbol
        for symbol in symtab
        if symbol.defined and symbol.kind == "OBJECT" and symbol.name not in {"_DYNAMIC", RUNTIME_VERSION}
    ]
    require(
        len(application_objects) == 1
        and application_objects[0].name == CONTROL
        and application_objects[0].value == control.value
        and application_objects[0].size == control.size,
        f"runtime application OBJECT set is not exactly the control object: {application_objects}",
    )
    undefined_globals = [
        symbol for symbol in dynsym if not symbol.defined and symbol.binding in {"GLOBAL", "WEAK"} and symbol.name
    ]
    require(not undefined_globals, f"runtime has dynamic undefined symbols: {undefined_globals}")
    defined_dynamic = {
        symbol.name
        for symbol in dynsym
        if symbol.defined and symbol.binding in {"GLOBAL", "WEAK"} and symbol.name
    }
    require(
        defined_dynamic == {RUNTIME_VERSION, f"{CONTROL}@@{RUNTIME_VERSION}"},
        f"runtime dynamic export closure is not exact: {sorted(defined_dynamic)}",
    )
    require(not forbidden_symbols(symtab), f"runtime has forbidden symbol closure: {sorted(forbidden_symbols(symtab))}")
    version_info = toolchain.version_info(runtime)
    version_names = re.findall(r"\bName:\s+(\S+)", version_info)
    require(
        version_names == [RUNTIME_SONAME, RUNTIME_VERSION],
        f"runtime version-definition closure is not exact: {version_names}",
    )
    return RUNTIME_SONAME, header.architecture


def validate_member_relocations(relocations: Sequence[Relocation], architecture: str, api: str) -> None:
    relevant = [
        relocation
        for relocation in relocations
        if relocation.base_name in {CONTROL, f"__real_{api}"}
    ]
    require(all(relocation.section in {".rela.text", ".rel.text"} for relocation in relevant),
            f"{api} profiler relocations are outside .text: {relevant}")
    control = [relocation.kind for relocation in relevant if relocation.base_name == CONTROL]
    real = [relocation.kind for relocation in relevant if relocation.base_name == f"__real_{api}"]
    if architecture == "x86_64":
        require(
            len(control) == 1 and control[0] in {"R_X86_64_REX_GOTPCRELX", "R_X86_64_GOTPCRELX"},
            f"{api} x86-64 control relocation is not one admitted GOT reference: {control}",
        )
        require(real == ["R_X86_64_PLT32"], f"{api} x86-64 real-call relocation is not exact: {real}")
    elif architecture == "aarch64":
        require(
            control == ["R_AARCH64_ADR_GOT_PAGE", "R_AARCH64_LD64_GOT_LO12_NC"],
            f"{api} AArch64 control relocation pair is not exact: {control}",
        )
        require(real == ["R_AARCH64_JUMP26"], f"{api} AArch64 real-call relocation is not exact: {real}")
    else:
        raise GateError(f"unsupported relocation architecture: {architecture}")
    require(len(relevant) == len(control) + len(real), f"{api} has unexpected profiler relocations: {relevant}")


@dataclasses.dataclass(frozen=True)
class ExtractedMember:
    name: str
    path: Path
    image: BoundedFileImage


def extract_archive(toolchain: Toolchain, archive: Path, directory: Path) -> list[ExtractedMember]:
    listing = run_bounded_text((toolchain.ar, "t", str(archive)))
    members = listing.splitlines()
    expected = tuple(f"oai_memprof_wrap_{api}.c.o" for api in APIS)
    require(tuple(members) == expected, f"wrapper archive order/names are not exact: {members}")
    paths: list[ExtractedMember] = []
    for index, member in enumerate(members):
        require(Path(member).name == member and member not in {".", ".."}, f"unsafe archive member: {member}")
        data = run_bounded_bytes((toolchain.ar, "p", str(archive), member))
        require(data.startswith(b"\x7fELF"), f"archive member is not ELF: {member}")
        output = directory / f"{index}_{member}"
        image = write_private_frozen_file(output, data, f"archive member {member}")
        paths.append(ExtractedMember(member, output, image))
    return paths


@dataclasses.dataclass(frozen=True)
class MemberProof:
    name: str
    size: int
    signature: tuple[str, ...]


def validate_wrapper_archive(toolchain: Toolchain, archive: Path, architecture: str) -> dict[str, MemberProof]:
    proofs: dict[str, MemberProof] = {}
    with tempfile.TemporaryDirectory(prefix="oai-memprof-r0-ar-") as temporary:
        for extracted in extract_archive(toolchain, archive, Path(temporary)):
            member_name = extracted.name
            member = extracted.path
            header = toolchain.header(member)
            require(header.elf_type == "REL" and header.architecture == architecture,
                    f"wrapper member has wrong ELF identity: {member_name}: {header}")
            require_no_tls_or_lto(toolchain, member)
            relocations = require_clean_relocations(toolchain, member)
            symbols = toolchain.symtab(member)
            defined_functions = [symbol for symbol in symbols if symbol.defined and symbol.kind == "FUNC"]
            require(len(defined_functions) == 1, f"{member_name} defined FUNC set is not singular: {defined_functions}")
            wrapper = defined_functions[0]
            require(wrapper.base_name in WRAPPERS, f"{member_name} defines the wrong function: {wrapper}")
            api = wrapper.base_name.removeprefix("__wrap_")
            require(member_name == f"oai_memprof_wrap_{api}.c.o", f"member/function mismatch: {member_name}/{api}")
            require(api not in proofs, f"duplicate wrapper proof for {api}")
            require(
                (wrapper.binding, wrapper.visibility) == ("GLOBAL", "HIDDEN"),
                f"{member_name} wrapper is not GLOBAL HIDDEN: {wrapper}",
            )
            undefined = {symbol.base_name for symbol in symbols if not symbol.defined and symbol.name}
            required = {CONTROL, f"__real_{api}"}
            require(required <= undefined, f"{member_name} lacks required undefined symbols: {undefined}")
            require(undefined <= required | {"_GLOBAL_OFFSET_TABLE_"},
                    f"{member_name} has helper/TLS/errno undefined symbols: {sorted(undefined - required)}")
            control_imports = [symbol for symbol in symbols if not symbol.defined and symbol.base_name == CONTROL]
            require(
                len(control_imports) == 1 and control_imports[0].visibility == "DEFAULT",
                f"{member_name} control import is not DEFAULT visibility: {control_imports}",
            )
            require(
                not forbidden_symbols(symbols),
                f"{member_name} has forbidden closure: {sorted(forbidden_symbols(symbols))}",
            )
            validate_member_relocations(relocations, architecture, api)
            instructions = toolchain.disassemble(member, wrapper.base_name)
            signature = wrapper_instruction_signature(
                instructions,
                architecture,
                api=api,
                expected_final_target=None,
            )
            proofs[api] = MemberProof(member_name, wrapper.size, signature)
            verify_file_image_unchanged(member, extracted.image, f"archive member {member_name}")
    require(set(proofs) == set(APIS), f"wrapper archive API set mismatch: {sorted(proofs)}")
    return proofs


def symbols_named(symbols: Iterable[Symbol], names: Iterable[str]) -> list[Symbol]:
    accepted = set(names)
    return [symbol for symbol in symbols if symbol.base_name in accepted]


def parse_interpreters(program_headers: str) -> list[str]:
    return re.findall(r"\[Requesting program interpreter:\s*([^]]+)\]", program_headers)


def validate_role_elf(
    toolchain: Toolchain,
    role: Role,
    architecture: str,
    member_proofs: dict[str, MemberProof],
) -> None:
    header = toolchain.header(role.path)
    require(header.architecture == architecture and header.elf_type == "DYN",
            f"{role.name} has wrong ELF identity: {header}")
    program_headers = toolchain.program_headers(role.path)
    interpreters = parse_interpreters(program_headers)
    if role.dso:
        require(
            header.entry == 0 and not interpreters,
            f"{role.name} MODULE has entry/interpreter: {header.entry}/{interpreters}",
        )
    else:
        expected_interpreter = {
            "x86_64": "/lib64/ld-linux-x86-64.so.2",
            "aarch64": "/lib/ld-linux-aarch64.so.1",
        }[architecture]
        require(header.entry != 0 and interpreters == [expected_interpreter],
                f"{role.name} PIE interpreter is not exact: entry={header.entry:#x}, interp={interpreters}")

    require_no_tls_or_lto(toolchain, role.path)
    relocations = require_clean_relocations(toolchain, role.path)
    dynamic = toolchain.dynamic(role.path)
    require_exact_dynamic_loader_policy(
        dynamic,
        role.name,
        expected_flags=() if role.dso else ("BIND_NOW",),
        expected_flags_1=() if role.dso else ("Flags: NOW PIE",),
    )
    needed = bracket_values(dynamic, "NEEDED")
    expected_needed = ([RUNTIME_SONAME] if role.wrapped else []) + ["libc.so.6"]
    require(needed == expected_needed, f"{role.name} DT_NEEDED order/closure is not exact: {needed}")
    require(bracket_values(dynamic, "RUNPATH") == [ROLE_RUNPATH],
            f"{role.name} RUNPATH is not exactly {ROLE_RUNPATH}: {bracket_values(dynamic, 'RUNPATH')}")
    require(not bracket_values(dynamic, "RPATH"), f"{role.name} has legacy DT_RPATH")
    require(not bracket_values(dynamic, "SONAME"), f"{role.name} unexpectedly defines DT_SONAME")
    symtab = toolchain.symtab(role.path)
    dynsym = toolchain.dynsym(role.path)
    profiler_defined = [
        symbol for symbol in symtab if symbol.defined and symbol.base_name.startswith("__wrap_")
    ]
    if role.wrapped:
        require({symbol.base_name for symbol in profiler_defined} == set(WRAPPERS)
                and len(profiler_defined) == len(WRAPPERS),
                f"{role.name} final wrapper FUNC set is not exact: {profiler_defined}")
        require(all(symbol.kind == "FUNC" for symbol in profiler_defined),
                f"{role.name} has non-FUNC wrapper definitions: {profiler_defined}")
        require(
            all(
                (symbol.binding, symbol.visibility) in {("LOCAL", "DEFAULT"), ("GLOBAL", "HIDDEN")}
                for symbol in profiler_defined
            ),
            f"{role.name} has a preemptible final wrapper definition: {profiler_defined}",
        )
        require(not symbols_named(dynsym, WRAPPERS), f"{role.name} exports wrappers dynamically")
        require(not symbols_named(symtab, REALS), f"{role.name} retains __real aliases")
        dyn_controls = [symbol for symbol in dynsym if not symbol.defined and symbol.base_name == CONTROL]
        sym_controls = [symbol for symbol in symtab if not symbol.defined and symbol.base_name == CONTROL]
        for table_name, controls in ((".dynsym", dyn_controls), (".symtab", sym_controls)):
            require(
                len(controls) == 1
                and controls[0].name == f"{CONTROL}@{RUNTIME_VERSION}"
                and controls[0].kind == "OBJECT"
                and controls[0].binding == "GLOBAL"
                and controls[0].visibility == "DEFAULT",
                f"{role.name} {table_name} control import is not exact: {controls}",
            )
        control_relocations = [relocation for relocation in relocations if relocation.base_name == CONTROL]
        expected_relocation = "R_X86_64_GLOB_DAT" if architecture == "x86_64" else "R_AARCH64_GLOB_DAT"
        require(
            len(control_relocations) == 1 and control_relocations[0].kind == expected_relocation,
            f"{role.name} final shared-control relocation is not singular/exact: {control_relocations}",
        )
    else:
        require(not profiler_defined, f"{role.name} A00 defines wrapper-prefixed functions: {profiler_defined}")
        require(not symbols_named(symtab, WRAPPERS + REALS + (CONTROL,)),
                f"{role.name} A00 contains profiler symbols")
        require(not symbols_named(dynsym, WRAPPERS + REALS + (CONTROL,)),
                f"{role.name} A00 dynamically references profiler symbols")
        require(not any(relocation.base_name == CONTROL for relocation in relocations),
                f"{role.name} A00 relocates profiler control")

    prefix = "oai_memprof_r0_dso_call_" if role.dso else "oai_memprof_r0_exe_call_"
    for api in APIS:
        callsite = f"{prefix}{api}"
        expected = f"__wrap_{api}" if role.wrapped else api
        validate_callsite_instructions(toolchain.disassemble(role.path, callsite), architecture, expected)
        if role.wrapped:
            wrapper_name = f"__wrap_{api}"
            wrapper_symbol = next(symbol for symbol in profiler_defined if symbol.base_name == wrapper_name)
            proof = member_proofs[api]
            require(wrapper_symbol.size == proof.size,
                    f"{role.name}:{wrapper_name} final/member size mismatch: {wrapper_symbol.size}/{proof.size}")
            instructions = toolchain.disassemble(role.path, wrapper_name)
            signature = wrapper_instruction_signature(
                instructions,
                architecture,
                api=api,
                expected_final_target=api,
            )
            require(signature == proof.signature,
                    f"{role.name}:{wrapper_name} final/member instruction signature mismatch: "
                    f"{signature}/{proof.signature}")
            if architecture == "x86_64":
                require(any(f"{CONTROL}@{RUNTIME_VERSION}" in item.operands for item in instructions),
                        f"{role.name}:{wrapper_name} final code is not bound to exact control version")


def validate_map(role: Role, wrapper_archive: Path, runtime: Path, member_proofs: dict[str, MemberProof]) -> None:
    text = read_bounded_text(role.map_path)
    markers = tuple(WRAPPERS) + (CONTROL, wrapper_archive.name, runtime.name, RUNTIME_SONAME)
    if role.wrapped:
        for marker in WRAPPERS + (CONTROL, wrapper_archive.name):
            require(marker in text, f"{role.name} link map lacks {marker}")
        require(runtime.name in text or RUNTIME_SONAME in text, f"{role.name} link map lacks exact runtime")
        for proof in member_proofs.values():
            require(proof.name in text, f"{role.name} link map lacks extracted member {proof.name}")
    else:
        present = sorted(marker for marker in markers if marker in text)
        require(not present, f"{role.name} A00 map contains profiler markers: {present}")


def compiler_token_count(tokens: Sequence[str], compiler: Path) -> int:
    expected = compiler.resolve()
    count = 0
    for token in tokens:
        if token.startswith("-"):
            continue
        path = Path(token)
        if path.is_file() and path.resolve() == expected:
            count += 1
    return count


def require_direct_compiler(tokens: Sequence[str], compiler: Path, context: str) -> None:
    require(
        tokens
        and Path(tokens[0]).is_file()
        and Path(tokens[0]).resolve() == compiler.resolve(),
        f"{context} is not launched directly by the configured GNU C compiler: {tokens[:2]}",
    )
    require(
        compiler_token_count(tokens, compiler) == 1,
        f"{context} does not contain the exact configured GNU C compiler exactly once",
    )


def require_direct_link_compiler(tokens: Sequence[str], compiler: Path, context: str) -> None:
    shell_controls = {"&&", "||", ";", "|", "&", "(", ")"}
    require(
        compiler_token_count(tokens, compiler) == 1,
        f"{context} does not contain the configured GNU C compiler exactly once",
    )
    if tokens and Path(tokens[0]).is_file() and Path(tokens[0]).resolve() == compiler.resolve():
        require(not any(token in shell_controls for token in tokens), f"{context} has shell control tokens")
        return
    require(
        len(tokens) >= 5
        and tokens[:2] == [":", "&&"]
        and tokens[-2:] == ["&&", ":"]
        and Path(tokens[2]).is_file()
        and Path(tokens[2]).resolve() == compiler.resolve(),
        f"{context} is not a direct compiler link or exact CMake no-op scaffold: {tokens[:4]}",
    )
    observed_controls = [
        (index, token)
        for index, token in enumerate(tokens)
        if token in shell_controls or token == ":"
    ]
    expected_controls = [(0, ":"), (1, "&&"), (len(tokens) - 2, "&&"), (len(tokens) - 1, ":")]
    require(observed_controls == expected_controls, f"{context} has unadmitted shell structure: {observed_controls}")


def cache_value(cache: str, key: str) -> str:
    matches = re.findall(rf"^{re.escape(key)}:[^=]+=(.*)$", cache, re.MULTILINE)
    require(len(matches) == 1, f"CMake cache key {key} multiplicity is {len(matches)}")
    return matches[0]


@dataclasses.dataclass(frozen=True)
class BuildProvenance:
    build_dir: Path
    source_dir: Path
    compiler: Path
    python_executable: Path
    commands_by_target: dict[str, list[list[str]]]
    all_commands: list[list[str]]


def validate_build_provenance(
    args: argparse.Namespace,
    toolchain: Toolchain,
    roles: Sequence[Role],
) -> BuildProvenance:
    require(args.build_dir is not None, "the frozen terminal gate requires --build-dir")
    require(args.source_dir is not None, "the frozen terminal gate requires --source-dir")
    require(args.command_evidence is None, "--command-evidence is not admitted; exact Ninja queries are required")
    build_dir = Path(args.build_dir).resolve()
    source_dir = Path(args.source_dir).resolve()
    require(build_dir.is_dir() and (build_dir / "build.ninja").is_file(), f"invalid Ninja build directory: {build_dir}")
    require(source_dir.is_dir(), f"invalid source directory: {source_dir}")
    cache = read_bounded_text(build_dir / "CMakeCache.txt")
    require(cache_value(cache, "CMAKE_GENERATOR") == "Ninja", "CMake generator is not exact single-config Ninja")
    require("CMAKE_CONFIGURATION_TYPES:" not in cache, "multi-config CMake cache is not admitted")
    cmake_home = Path(cache_value(cache, "CMAKE_HOME_DIRECTORY")).resolve()
    require(cmake_home == source_dir, f"--source-dir differs from CMAKE_HOME_DIRECTORY: {source_dir} != {cmake_home}")
    compiler = Path(cache_value(cache, "CMAKE_C_COMPILER")).resolve()
    require(compiler.is_file(), f"configured C compiler is missing: {compiler}")
    python_executable = Path(cache_value(cache, "_Python3_EXECUTABLE"))
    require(
        python_executable.is_absolute() and python_executable.is_file(),
        f"configured Python interpreter is invalid: {python_executable}",
    )
    require(Path(cache_value(cache, "CMAKE_AR")).resolve() == Path(toolchain.ar),
            "checker ar differs from configured CMAKE_AR")
    require(Path(cache_value(cache, "CMAKE_LINKER")).resolve() == Path(toolchain.ld),
            "checker ld differs from configured CMAKE_LINKER")
    require(Path(cache_value(cache, "CMAKE_MAKE_PROGRAM")).resolve() == Path(toolchain.ninja),
            "checker Ninja differs from configured CMAKE_MAKE_PROGRAM")
    compiler_files = tuple((build_dir / "CMakeFiles").glob("*/CMakeCCompiler.cmake"))
    require(len(compiler_files) == 1, f"compiler-ID evidence multiplicity is {len(compiler_files)}")
    compiler_id_text = read_bounded_text(compiler_files[0])
    require(len(re.findall(r'^set\(CMAKE_C_COMPILER_ID "GNU"\)$', compiler_id_text, re.MULTILINE)) == 1,
            "CMake compiler ID is not exactly GNU")
    compiler_version = run_bounded_text((str(compiler), "-dumpfullversion", "-dumpversion")).strip()
    require(re.fullmatch(r"\d+(?:\.\d+){1,2}", compiler_version) is not None,
            f"configured GNU compiler version is malformed: {compiler_version}")

    require(tuple(args.ninja_target) == EXPECTED_NINJA_TARGETS,
            f"Ninja target order/set is not exact: {args.ninja_target}")
    role_by_target = {role.ninja_target: role for role in roles}
    require(set(role_by_target) == set(EXPECTED_NINJA_TARGETS), "role/Ninja target mapping is incomplete")
    commands_by_target: dict[str, list[list[str]]] = {}
    unique_lines: dict[str, None] = {}
    for target in EXPECTED_NINJA_TARGETS:
        query = run_bounded_text((toolchain.ninja, "-C", str(build_dir), "-t", "query", target))
        lines = query.splitlines()
        require(lines and lines[0] == f"{target}:", f"Ninja query does not resolve exact target {target}")
        require("  input: phony" in lines, f"Ninja target is not a CMake phony alias: {target}")
        relative_artifact = str(role_by_target[target].path.resolve().relative_to(build_dir))
        require(f"    {relative_artifact}" in lines, f"Ninja target {target} does not name exact role artifact")
        evidence = run_bounded_text((toolchain.ninja, "-C", str(build_dir), "-t", "commands", target))
        commands_by_target[target] = parse_command_lines(evidence, build_dir)
        for line in evidence.splitlines():
            if line.strip():
                unique_lines.setdefault(line, None)
    all_commands = parse_command_lines("\n".join(unique_lines), build_dir)
    return BuildProvenance(
        build_dir,
        source_dir,
        compiler,
        python_executable,
        commands_by_target,
        all_commands,
    )


def validate_compile_and_runtime_link_commands(
    provenance: BuildProvenance,
    toolchain: Toolchain,
    runtime: Path,
    wrapper_archive: Path,
) -> None:
    expected_sources = tuple((provenance.source_dir / relative).resolve() for relative in OWNED_COMPILE_PATHS)
    by_source: dict[Path, list[list[str]]] = {source: [] for source in expected_sources}
    for tokens in provenance.all_commands:
        for source in expected_sources:
            basename_hits = [
                token
                for token in tokens
                if not token.startswith("-") and Path(token).name == source.name
            ]
            for token in basename_hits:
                candidate = Path(token)
                candidate = (
                    (provenance.build_dir / candidate).resolve()
                    if not candidate.is_absolute()
                    else candidate.resolve()
                )
                require(
                    candidate == source,
                    f"owned source basename resolves outside --source-dir: {token} != {source}",
                )
            if resolved_token_indices(tokens, source, provenance.build_dir):
                by_source[source].append(tokens)
    for source, matches in by_source.items():
        require(len(matches) == 1, f"compile command multiplicity for {source} is {len(matches)}")
        tokens = matches[0]
        require(len(resolved_token_indices(tokens, source, provenance.build_dir)) == 1,
                f"owned source path multiplicity for {source} is not one")
        require(tokens.count("-c") == 1, f"{source} command is not one compilation")
        require_direct_compiler(tokens, provenance.compiler, f"{source} compile")
        require_no_linker_driver_override(tokens, f"{source} compile")
        require_passive_flags(tokens, f"{source} compile")
        if source.name.startswith("r0_actual_fixture_"):
            for flag in ACTUAL_NO_BUILTIN_FLAGS:
                require(tokens.count(flag) == 1, f"{source} requires exactly one {flag}")

    runtime_link = find_output_command(provenance.all_commands, runtime, provenance.build_dir)
    require_direct_link_compiler(runtime_link, provenance.compiler, "runtime link")
    reject_positive_instrumentation(runtime_link, "runtime link")
    require_no_linker_driver_override(runtime_link, "runtime link")
    require(runtime_link.count("-nostartfiles") == 1, "runtime link lacks exact -nostartfiles")
    require(runtime_link.count("-shared") == 1, "runtime link is not exactly one shared link")
    require(sum(token.startswith("-Wl,-soname,") and token.endswith(RUNTIME_SONAME) for token in runtime_link) == 1,
            "runtime link SONAME token is not exact")
    require(
        sum(
            "--version-script=" in token and token.endswith("oai_memprof_runtime.map")
            for token in runtime_link
        )
        == 1,
        "runtime link version-script token is not exact",
    )
    require(not any("rpath" in token.lower() for token in runtime_link), "runtime link command contains RPATH")

    archive_commands = [
        tokens
        for tokens in provenance.all_commands
        if any(Path(token).name == wrapper_archive.name for token in tokens if not token.startswith("-"))
        and "qc" in tokens
    ]
    require(len(archive_commands) == 1, f"wrapper archive command multiplicity is {len(archive_commands)}")
    archive_command = archive_commands[0]
    require(compiler_token_count(archive_command, Path(toolchain.ar)) == 1,
            "wrapper archive command does not use configured GNU ar")
    member_inputs = tuple(Path(token).name for token in archive_command if token.endswith(".c.o"))
    require(member_inputs == tuple(f"oai_memprof_wrap_{api}.c.o" for api in APIS),
            f"archive command member order/multiplicity is not exact: {member_inputs}")


def expected_r0_ctest_commands(
    build_dir: Path,
    source_dir: Path,
    python_executable: Path,
) -> dict[str, tuple[str, ...]]:
    memprof_build = build_dir / "common/utils/memprof"
    test_build = memprof_build / "tests"
    test_source = source_dir / "common/utils/memprof/tests"
    python = str(python_executable)
    oracle = str(test_source / "validate_r0_scripted_oracle.py")
    runtime = str(memprof_build / "liboai_memprof_runtime.so.1.0.0")
    wrapper_archive = str(memprof_build / "liboai_memprof_wrap_c.a")
    a00_exe = str(test_build / "test_oai_memprof_r0_actual_a00")
    a01_exe = str(test_build / "test_oai_memprof_r0_actual_a01")
    a00_dso = str(test_build / "liboai_memprof_r0_actual_dso_a00.so")
    a01_dso = str(test_build / "liboai_memprof_r0_actual_dso_a01.so")
    a00_exe_map = str(test_build / "oai_memprof_r0_a00_exe.map")
    a01_exe_map = str(test_build / "oai_memprof_r0_a01_exe.map")
    a00_dso_map = str(test_build / "oai_memprof_r0_a00_dso.map")
    a01_dso_map = str(test_build / "oai_memprof_r0_a01_dso.map")

    commands: dict[str, tuple[str, ...]] = {}
    for suffix, mutation_name in EXPECTED_R0_MUTATIONS:
        commands[f"oai_memprof_r0_mutation_{suffix}"] = (
            python,
            "-B",
            oracle,
            "launch-mutation",
            mutation_name,
            str(test_build / f"test_oai_memprof_r0_mutant_{suffix}"),
        )
    commands.update(
        {
            "oai_memprof_r0_scripted_pair": (
                python,
                "-B",
                oracle,
                "launch-pair",
                str(test_build / "test_oai_memprof_r0_scripted_a00"),
                str(test_build / "test_oai_memprof_r0_scripted_a01"),
            ),
            "oai_memprof_r0_scripted_bounds": (python, "-B", oracle, "bounds-self-test"),
            "oai_memprof_r0_actual_a00": (
                a00_exe,
                "--dso",
                a00_dso,
                "--runtime",
                "absent",
            ),
            "oai_memprof_r0_actual_a01": (
                a01_exe,
                "--dso",
                a01_dso,
                "--runtime",
                "present-off",
                "--runtime-path",
                runtime,
            ),
            "oai_memprof_r0_actual_differential": (
                python,
                "-B",
                str(test_source / "run_r0_actual_differential.py"),
                "--a00-exe",
                a00_exe,
                "--a00-dso",
                a00_dso,
                "--a01-exe",
                a01_exe,
                "--a01-dso",
                a01_dso,
                "--runtime-path",
                runtime,
                "--timeout-seconds",
                "10",
                "--max-output-bytes",
                "4096",
            ),
            "oai_memprof_r0_elf_validator_selftest": (
                python,
                "-B",
                str(test_source / "check_oai_memprof_r0_elf.py"),
                "--self-test",
            ),
            "oai_memprof_r0_harness_selftest": (
                python,
                "-B",
                str(test_source / "test_r0_harness.py"),
            ),
            "oai_memprof_r0_absence_validator_selftest": (
                python,
                "-B",
                str(test_source / "check_oai_memprof_r0_absence.py"),
                "--self-test",
            ),
            "oai_memprof_r0_elf": (
                python,
                "-B",
                str(test_source / "check_oai_memprof_r0_elf.py"),
                "--a00-exe",
                a00_exe,
                "--a01-exe",
                a01_exe,
                "--a00-dso",
                a00_dso,
                "--a01-dso",
                a01_dso,
                "--runtime",
                runtime,
                "--wrapper-archive",
                wrapper_archive,
                "--a00-exe-map",
                a00_exe_map,
                "--a01-exe-map",
                a01_exe_map,
                "--a00-dso-map",
                a00_dso_map,
                "--a01-dso-map",
                a01_dso_map,
                "--build-dir",
                str(build_dir),
                "--source-dir",
                str(source_dir),
                "--ninja-target",
                EXPECTED_NINJA_TARGETS[0],
                "--ninja-target",
                EXPECTED_NINJA_TARGETS[1],
                "--ninja-target",
                EXPECTED_NINJA_TARGETS[2],
                "--ninja-target",
                EXPECTED_NINJA_TARGETS[3],
            ),
        }
    )
    require(set(commands) == set(EXPECTED_R0_CTESTS), "internal R0 CTest command catalog is incomplete")
    return commands


def parse_ctest_property_block(block: str, test_name: str) -> dict[str, str]:
    pattern = re.compile(r'(?:^|\s)([A-Z_][A-Z0-9_]*) "([^"]*)"')
    entries: dict[str, str] = {}
    cursor = 0
    for match in pattern.finditer(block):
        require(not block[cursor : match.start()].strip(), f"{test_name} has malformed CTest properties")
        key, value = match.group(1), match.group(2)
        require(key not in entries, f"{test_name} has duplicate CTest property {key}")
        entries[key] = value
        cursor = match.end()
    require(not block[cursor:].strip(), f"{test_name} has malformed trailing CTest properties")
    expected_keys = {"ENVIRONMENT", "ENVIRONMENT_MODIFICATION", "LABELS", "TIMEOUT", "_BACKTRACE_TRIPLES"}
    require(set(entries) == expected_keys, f"{test_name} CTest property closure is not exact: {sorted(entries)}")
    require(";add_test;" in entries["_BACKTRACE_TRIPLES"], f"{test_name} lacks generated add_test provenance")
    return entries


def validate_ctest_contract_text(
    text: str,
    *,
    build_dir: Path,
    source_dir: Path,
    python_executable: Path,
) -> None:
    add_pattern = re.compile(r"^add_test\(\[=\[([^]]+)\]=\]\s+(.+)\)$")
    property_pattern = re.compile(
        r"^set_tests_properties\(\[=\[([^]]+)\]=\]\s+PROPERTIES\s+(.*)\)$"
    )
    additions: dict[str, list[tuple[str, ...]]] = {}
    properties: dict[str, list[str]] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        add = add_pattern.fullmatch(line)
        if add is not None:
            try:
                command = tuple(shlex.split(add.group(2), posix=True))
            except ValueError as error:
                raise GateError(f"cannot parse generated CTest command {add.group(1)}: {error}") from error
            additions.setdefault(add.group(1), []).append(command)
        prop = property_pattern.fullmatch(line)
        if prop is not None:
            properties.setdefault(prop.group(1), []).append(prop.group(2))
        require(add is not None or prop is not None, f"unadmitted generated leaf CTest statement: {line}")

    generated_r0 = tuple(name for name in additions if name.startswith("oai_memprof_r0_"))
    require(
        set(generated_r0) == set(EXPECTED_R0_CTESTS) and len(generated_r0) == len(EXPECTED_R0_CTESTS),
        f"generated R0 CTest name closure is not exact: {generated_r0}",
    )
    expected_scrub = set(known_loader_environment_variables())
    expected_commands = expected_r0_ctest_commands(build_dir, source_dir, python_executable)
    for test_name, blocks in properties.items():
        if test_name in EXPECTED_R0_CTESTS:
            continue
        for block in blocks:
            labels = re.findall(r'(?:^|\s)LABELS "([^"]*)"', block)
            require(
                all("r0" not in label_set.split(";") for label_set in labels),
                f"non-R0 test {test_name} enters the frozen r0 label selection: {labels}",
            )
    for test_name in EXPECTED_R0_CTESTS:
        require(len(additions.get(test_name, ())) == 1, f"CTest add_test multiplicity for {test_name} is not one")
        require(
            additions[test_name][0] == expected_commands[test_name],
            f"{test_name} CTest command/arguments are not exact: {additions[test_name][0]}",
        )
        blocks = properties.get(test_name, ())
        require(len(blocks) == 1, f"CTest property multiplicity for {test_name} is not one")
        parsed_properties = parse_ctest_property_block(blocks[0], test_name)
        require(
            parsed_properties["ENVIRONMENT"] == "LC_ALL=C",
            f"{test_name} CTest environment is not exact: {parsed_properties['ENVIRONMENT']}",
        )
        require(parsed_properties["LABELS"] == "memprof;r0", f"{test_name} CTest labels are not exact")
        expected_timeout = EXPECTED_R0_CTEST_TIMEOUT_SECONDS[test_name]
        require(
            parsed_properties["TIMEOUT"] == str(expected_timeout),
            f"{test_name} CTest timeout is not exactly {expected_timeout} seconds: "
            f"{parsed_properties['TIMEOUT']}",
        )
        modification = parsed_properties["ENVIRONMENT_MODIFICATION"]
        entries = modification.split(";") if modification else []
        parsed: list[str] = []
        for entry in entries:
            match = re.fullmatch(r"([A-Z][A-Z0-9_]*)=unset:", entry)
            require(match is not None, f"{test_name} has malformed environment modification: {entry!r}")
            parsed.append(match.group(1))
        require(len(parsed) == len(set(parsed)), f"{test_name} has duplicate loader scrub entries: {parsed}")
        require(set(parsed) == expected_scrub, f"{test_name} loader scrub is not authoritative/exact: {parsed}")


def validate_ctest_contract(provenance: BuildProvenance) -> None:
    generated = provenance.build_dir / "common/utils/memprof/tests/CTestTestfile.cmake"
    validate_ctest_contract_text(
        read_bounded_text(generated),
        build_dir=provenance.build_dir,
        source_dir=provenance.source_dir,
        python_executable=provenance.python_executable,
    )


def validate_target_isolation(
    provenance: BuildProvenance,
    roles: Sequence[Role],
    wrapper_archive: Path,
    runtime: Path,
) -> None:
    forbidden_markers = (
        wrapper_archive.name,
        runtime.name,
        RUNTIME_SONAME,
        "oai_memprof_runtime.c",
        "oai_memprof_runtime.map",
        *(f"oai_memprof_wrap_{api}.c" for api in APIS),
    )
    for role in roles:
        commands = provenance.commands_by_target[role.ninja_target]
        evidence = "\n".join(shlex.join(tokens) for tokens in commands)
        if not role.wrapped:
            present = [marker for marker in forbidden_markers if marker in evidence]
            require(not present, f"{role.name} Ninja closure is not isolated from profiler production: {present}")
            require("--wrap=" not in evidence, f"{role.name} Ninja closure contains GNU wrap")


def validate_final_link_pairs(
    provenance: BuildProvenance,
    roles: Sequence[Role],
    wrapper_archive: Path,
    runtime: Path,
) -> None:
    role_commands: dict[str, list[str]] = {}
    normalized: dict[str, tuple[str, ...]] = {}
    for role in roles:
        commands = provenance.commands_by_target[role.ninja_target]
        tokens = find_output_command(commands, role.path, provenance.build_dir)
        require_direct_link_compiler(tokens, provenance.compiler, f"{role.name} final link")
        require_no_linker_driver_override(tokens, f"{role.name} final link")
        role_commands[role.name] = tokens
        normalized[role.name] = normalize_final_link(
            role, tokens, wrapper_archive, runtime, provenance.build_dir
        )
        linker_arguments = normalized_linker_arguments(tokens)
        require(linker_arguments.count("--cref") == 1, f"{role.name} requires exactly one --cref")
        rpath_tokens = [token for token in tokens if "rpath" in token.lower()]
        require(rpath_tokens == [f"-Wl,-rpath,\\{ROLE_RUNPATH}"],
                f"{role.name} final-link RUNPATH token is not exact: {rpath_tokens}")
        if role.dso:
            require(tokens.count("-shared") == 1 and "-Wl,-pie" not in tokens,
                    f"{role.name} final link is not exactly MODULE-shaped")
            require("-z" in linker_arguments and "defs" in linker_arguments,
                    f"{role.name} MODULE link lacks -z defs")
        else:
            require(tokens.count("-shared") == 0 and tokens.count("-Wl,-pie") == 1,
                    f"{role.name} final link is not exactly PIE-shaped")

    require(explicit_object_inputs(role_commands["a00-exe"]) == explicit_object_inputs(role_commands["a01-exe"]),
            "A00/A01 executable links do not preserve object order/multiplicity")
    require(explicit_object_inputs(role_commands["a00-dso"]) == explicit_object_inputs(role_commands["a01-dso"]),
            "A00/A01 MODULE links do not preserve object order/multiplicity")
    require(normalized["a00-exe"] == normalized["a01-exe"],
            "executable final links differ beyond admitted A01 wrap/archive/runtime deltas")
    require(normalized["a00-dso"] == normalized["a01-dso"],
            "MODULE final links differ beyond admitted A01 wrap/archive/runtime deltas")


def validate(args: argparse.Namespace) -> None:
    toolchain = Toolchain.from_args(args)
    toolchain.verify_versions()
    wrapper_archive = Path(args.wrapper_archive).resolve()
    runtime = Path(args.runtime).resolve()
    roles = (
        Role(
            "a00-exe",
            Path(args.a00_exe).resolve(),
            Path(args.a00_exe_map).resolve(),
            False,
            False,
            EXPECTED_NINJA_TARGETS[0],
        ),
        Role(
            "a01-exe",
            Path(args.a01_exe).resolve(),
            Path(args.a01_exe_map).resolve(),
            True,
            False,
            EXPECTED_NINJA_TARGETS[1],
        ),
        Role(
            "a00-dso",
            Path(args.a00_dso).resolve(),
            Path(args.a00_dso_map).resolve(),
            False,
            True,
            EXPECTED_NINJA_TARGETS[2],
        ),
        Role(
            "a01-dso",
            Path(args.a01_dso).resolve(),
            Path(args.a01_dso_map).resolve(),
            True,
            True,
            EXPECTED_NINJA_TARGETS[3],
        ),
    )
    with tempfile.TemporaryDirectory(prefix="oai-memprof-r0-snapshot-") as temporary:
        snapshot_root = Path(temporary)
        snapshots = [
            snapshot_regular_file(wrapper_archive, snapshot_root, "00-wrapper-archive"),
            snapshot_regular_file(runtime, snapshot_root, "01-runtime"),
        ]
        for index, role in enumerate(roles):
            snapshots.append(snapshot_regular_file(role.path, snapshot_root, f"{index + 2:02d}-{role.name}-elf"))
            snapshots.append(snapshot_regular_file(role.map_path, snapshot_root, f"{index + 6:02d}-{role.name}-map"))

        snapshot_by_label = {snapshot.label: snapshot for snapshot in snapshots}
        frozen_wrapper = snapshot_by_label["00-wrapper-archive"].path
        frozen_runtime = snapshot_by_label["01-runtime"].path
        frozen_roles = tuple(
            dataclasses.replace(
                role,
                path=snapshot_by_label[f"{index + 2:02d}-{role.name}-elf"].path,
                map_path=snapshot_by_label[f"{index + 6:02d}-{role.name}-map"].path,
            )
            for index, role in enumerate(roles)
        )

        runtime_soname, architecture = validate_runtime(toolchain, frozen_runtime)
        require(runtime_soname == RUNTIME_SONAME, "internal runtime SONAME invariant failed")
        member_proofs = validate_wrapper_archive(toolchain, frozen_wrapper, architecture)
        for role in frozen_roles:
            validate_role_elf(toolchain, role, architecture, member_proofs)
            validate_map(role, frozen_wrapper, frozen_runtime, member_proofs)

        provenance = validate_build_provenance(args, toolchain, roles)
        validate_compile_and_runtime_link_commands(provenance, toolchain, runtime, wrapper_archive)
        validate_ctest_contract(provenance)
        validate_target_isolation(provenance, roles, wrapper_archive, runtime)
        validate_final_link_pairs(provenance, roles, wrapper_archive, runtime)
        for snapshot in snapshots:
            verify_snapshot_source_unchanged(snapshot)
    print(
        f"R0_ELF_GATE_V2 pass architecture={architecture} roles=4 wrappers=4 "
        f"runtime_soname={RUNTIME_SONAME} generator=Ninja compiler=GNU"
    )


class SensitivitySelfTest(unittest.TestCase):
    def assertGate(self, function, *arguments, **keywords) -> None:
        with self.assertRaises(GateError):
            function(*arguments, **keywords)

    def test_symbol_parser_exact_versions_and_count(self) -> None:
        text = """Symbol table '.dynsym' contains 2 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND
     1: 0000000000000040     8 OBJECT  GLOBAL PROTECTED    3 oai_memprof_control_v1@@OAI_MEMPROF_RUNTIME_1.0
"""
        symbols = parse_symbols(text, ".dynsym")
        self.assertEqual(symbols[1].name, f"{CONTROL}@@{RUNTIME_VERSION}")

    def test_symbol_parser_rejects_malformed_row(self) -> None:
        text = """Symbol table '.dynsym' contains 2 entries:
   Num: Value Size Type Bind Vis Ndx Name
     0: 0000000000000000 0 NOTYPE LOCAL DEFAULT UND
     1: malformed control row
"""
        self.assertGate(parse_symbols, text, ".dynsym")

    def test_symbol_parser_rejects_silent_count_loss(self) -> None:
        text = """Symbol table '.dynsym' contains 2 entries:
   Num: Value Size Type Bind Vis Ndx Name
     0: 0000000000000000 0 NOTYPE LOCAL DEFAULT UND
"""
        self.assertGate(parse_symbols, text, ".dynsym")

    def test_wrap_order_and_duplicates_are_observable(self) -> None:
        tokens = ["cc", "-Wl,--wrap=malloc", "-Wl,--wrap=calloc", "-Wl,--wrap=malloc"]
        self.assertEqual(wrapped_symbols(tokens), ["malloc", "calloc", "malloc"])

    def test_wrap_tokens_must_be_exact_standalone_driver_arguments(self) -> None:
        for tokens in (
            ["cc", "-Wl,--wrap=malloc,--as-needed"],
            ["cc", "-Wl,-z,defs,--wrap=malloc"],
            ["cc", "-Xlinker", "--wrap=malloc"],
            ["cc", "--wrap=malloc"],
        ):
            with self.subTest(tokens=tokens):
                self.assertGate(wrapped_symbols, tokens)

    def test_positive_instrumentation_matrix_rejected(self) -> None:
        for token in (
            "-flto=auto",
            "-fuse-linker-plugin",
            "-fsanitize=address",
            "--coverage",
            "-finstrument-functions",
            "-fstack-protector-strong",
            "-fprofile-generate",
            "-fplugin=/tmp/instrument.so",
            "-fplugin-arg-instrument-mode=active",
            "-fcoverage-mapping",
        ):
            with self.subTest(token=token):
                self.assertGate(reject_positive_instrumentation, ["cc", token], "selftest")
        self.assertGate(reject_positive_instrumentation, ["cc", "-Wl,-plugin-opt=jobs=2"], "selftest")

    def test_negative_flags_must_be_exact(self) -> None:
        require_passive_flags(["cc", *PASSIVE_NEGATIVE_FLAGS], "selftest")
        self.assertGate(require_passive_flags, ["cc", *PASSIVE_NEGATIVE_FLAGS, "-fno-lto"], "selftest")
        self.assertGate(require_passive_flags, ["cc", *PASSIVE_NEGATIVE_FLAGS[:-1]], "selftest")

    def test_linker_driver_override_rejected(self) -> None:
        require_no_linker_driver_override(["cc", "-Wl,-z,defs"], "selftest")
        for tokens in (
            ["cc", "-fuse-ld=lld"],
            ["cc", "-B/tmp/toolchain"],
            ["cc", "-B", "/tmp/toolchain"],
            ["cc", "-specs=/tmp/instrument.specs"],
            ["cc", "--specs=/tmp/instrument.specs"],
            ["cc", "-wrapper", "/tmp/wrapper"],
        ):
            with self.subTest(tokens=tokens):
                self.assertGate(require_no_linker_driver_override, tokens, "selftest")

    def test_loader_injection_linker_options_are_rejected(self) -> None:
        for tokens in (
            ["cc", "-Wl,--audit,/tmp/audit.so"],
            ["cc", "-Wl,--depaudit,/tmp/audit.so"],
            ["cc", "-Wl,--filter,/tmp/filter.so"],
            ["cc", "-Wl,--auxiliary,/tmp/auxiliary.so"],
            ["cc", "-Wl,-z,interpose"],
            ["cc", "-Wl,-z,nodefaultlib"],
            ["cc", "-Xlinker", "--audit=/tmp/audit.so"],
        ):
            with self.subTest(tokens=tokens):
                self.assertGate(require_no_linker_driver_override, tokens, "selftest")

    def test_x86_relocation_signature(self) -> None:
        good = [
            Relocation(".rela.text", 7, "R_X86_64_REX_GOTPCRELX", CONTROL),
            Relocation(".rela.text", 15, "R_X86_64_PLT32", "__real_malloc"),
        ]
        validate_member_relocations(good, "x86_64", "malloc")
        self.assertGate(validate_member_relocations, good + [good[0]], "x86_64", "malloc")

    def test_singular_relocation_section_is_parsed(self) -> None:
        text = """Relocation section '.rela.text' at offset 0x100 contains 1 entry:
    Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
0000000000000007  0000000b0000002a R_X86_64_REX_GOTPCRELX 0000000000000000 oai_memprof_control_v1 - 4
"""
        relocations = parse_relocations(text)
        self.assertEqual(relocations, [Relocation(".rela.text", 7, "R_X86_64_REX_GOTPCRELX", CONTROL)])

    def test_relocation_parser_rejects_count_and_row_mutations(self) -> None:
        row = (
            "0000000000000007  0000000b0000002a R_X86_64_REX_GOTPCRELX "
            "0000000000000000 oai_memprof_control_v1 - 4"
        )
        template = """Relocation section '.rela.text' at offset 0x100 contains {count} {noun}:
    Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
{rows}
"""
        self.assertGate(parse_relocations, template.format(count=2, noun="entries", rows=row))
        self.assertGate(parse_relocations, template.format(count=0, noun="entries", rows=row))
        self.assertGate(
            parse_relocations,
            template.format(count=1, noun="entry", rows=f"{row}\n{row}"),
        )
        self.assertGate(
            parse_relocations,
            template.format(count=1, noun="entry", rows="0000000000000007 malformed row"),
        )

    def test_aarch64_paired_got_relocations(self) -> None:
        good = [
            Relocation(".rela.text", 0, "R_AARCH64_ADR_GOT_PAGE", CONTROL),
            Relocation(".rela.text", 4, "R_AARCH64_LD64_GOT_LO12_NC", CONTROL),
            Relocation(".rela.text", 12, "R_AARCH64_JUMP26", "__real_malloc"),
        ]
        validate_member_relocations(good, "aarch64", "malloc")
        self.assertGate(validate_member_relocations, good[1:], "aarch64", "malloc")
        self.assertGate(validate_member_relocations, [good[1], good[0], good[2]], "aarch64", "malloc")

    def test_x86_wrapper_shape_and_extra_load(self) -> None:
        good = [
            Instruction(0, "endbr64", ""),
            Instruction(4, "mov", "0x0(%rip),%rax"),
            Instruction(11, "mov", "(%rax),%rax"),
            Instruction(14, "jmp", "20 <malloc@plt>"),
        ]
        self.assertEqual(
            wrapper_instruction_signature(good, "x86_64", api="malloc", expected_final_target="malloc"),
            ("endbr64", "mov", "mov", "jmp", "scratch=%rax"),
        )
        bad = good[:-1] + [Instruction(13, "mov", "(%rax),%rax"), good[-1]]
        self.assertGate(wrapper_instruction_signature, bad, "x86_64", api="malloc", expected_final_target="malloc")
        mismatched_base = [good[0], good[1], Instruction(11, "mov", "(%rcx),%rax"), good[-1]]
        self.assertGate(
            wrapper_instruction_signature,
            mismatched_base,
            "x86_64",
            api="malloc",
            expected_final_target="malloc",
        )
        clobbered_argument = [good[0], good[1], Instruction(11, "mov", "(%rax),%rdi"), good[-1]]
        self.assertGate(
            wrapper_instruction_signature,
            clobbered_argument,
            "x86_64",
            api="malloc",
            expected_final_target="malloc",
        )
        argument_chain = [
            good[0],
            Instruction(4, "mov", "0x0(%rip),%rdi"),
            Instruction(11, "mov", "(%rdi),%rdi"),
            good[-1],
        ]
        self.assertGate(
            wrapper_instruction_signature,
            argument_chain,
            "x86_64",
            api="malloc",
            expected_final_target="malloc",
        )

    def test_aarch64_wrapper_shape(self) -> None:
        good = [
            Instruction(0, "bti", "c"),
            Instruction(4, "adrp", "x2, 0 <control>"),
            Instruction(8, "ldr", "x2, [x2]"),
            Instruction(12, "ldar", "x2, [x2]"),
            Instruction(16, "b", "20 <malloc>"),
        ]
        wrapper_instruction_signature(good, "aarch64", api="malloc", expected_final_target="malloc")
        for bad in (
            [good[0], good[1], Instruction(8, "ldr", "x3, [x2]"), good[3], good[4]],
            [good[0], good[1], good[2], Instruction(12, "ldar", "x0, [x2]"), good[4]],
        ):
            with self.subTest(bad=bad):
                self.assertGate(
                    wrapper_instruction_signature,
                    bad,
                    "aarch64",
                    api="malloc",
                    expected_final_target="malloc",
                )
        argument_chain = [
            good[0],
            Instruction(4, "adrp", "x0, 0 <control>"),
            Instruction(8, "ldr", "x0, [x0]"),
            Instruction(12, "ldar", "x0, [x0]"),
            good[4],
        ]
        self.assertGate(
            wrapper_instruction_signature,
            argument_chain,
            "aarch64",
            api="malloc",
            expected_final_target="malloc",
        )

    def test_indirect_transfers_fail_closed(self) -> None:
        self.assertGate(extract_transfers, [Instruction(0, "call", "*%rax")], "x86_64")
        self.assertGate(extract_transfers, [Instruction(0, "blr", "x0")], "aarch64")

    def test_callsite_extra_instruction_rejected(self) -> None:
        good = [Instruction(0, "endbr64", ""), Instruction(4, "jmp", "8 <malloc@plt>")]
        validate_callsite_instructions(good, "x86_64", "malloc")
        self.assertGate(
            validate_callsite_instructions,
            [good[0], Instruction(4, "nop", ""), good[1]],
            "x86_64",
            "malloc",
        )

    def test_unknown_machine_fails_closed(self) -> None:
        text = """Class: ELF64
Data: 2's complement, little endian
Type: DYN (Shared object file)
Machine: RISC-V
Entry point address: 0x0
"""
        self.assertGate(parse_header, text)

    def test_dynamic_parser_preserves_order_and_multiplicity(self) -> None:
        text = """Dynamic section at offset 0x100 contains 3 entries:
  Tag        Type                         Name/Value
 0x1 (NEEDED) Shared library: [a.so]
 0x1 (NEEDED) Shared library: [a.so]
 0x0 (NULL) 0x0
"""
        entries = parse_dynamic(text)
        self.assertEqual(bracket_values(entries, "NEEDED"), ["a.so", "a.so"])

    def test_dynamic_loader_policy_rejects_injection_tags_and_extra_flag_bits(self) -> None:
        executable = [
            DynamicEntry("FLAGS", "BIND_NOW"),
            DynamicEntry("FLAGS_1", "Flags: NOW PIE"),
            DynamicEntry("NULL", "0x0"),
        ]
        require_exact_dynamic_loader_policy(
            executable,
            "selftest",
            expected_flags=("BIND_NOW",),
            expected_flags_1=("Flags: NOW PIE",),
        )
        for extra in (
            DynamicEntry("AUDIT", "Audit library: [/tmp/audit.so]"),
            DynamicEntry("DEPAUDIT", "Dependency audit library: [/tmp/audit.so]"),
            DynamicEntry("FILTER", "Filter library: [/tmp/filter.so]"),
            DynamicEntry("AUXILIARY", "Auxiliary library: [/tmp/auxiliary.so]"),
        ):
            with self.subTest(extra=extra):
                self.assertGate(
                    require_exact_dynamic_loader_policy,
                    [*executable[:-1], extra, executable[-1]],
                    "selftest",
                    expected_flags=("BIND_NOW",),
                    expected_flags_1=("Flags: NOW PIE",),
                )
        for mutated in (
            [DynamicEntry("FLAGS", "BIND_NOW INTERPOSE"), executable[1], executable[2]],
            [executable[0], DynamicEntry("FLAGS_1", "Flags: NOW NODEFLIB PIE"), executable[2]],
        ):
            with self.subTest(mutated=mutated):
                self.assertGate(
                    require_exact_dynamic_loader_policy,
                    mutated,
                    "selftest",
                    expected_flags=("BIND_NOW",),
                    expected_flags_1=("Flags: NOW PIE",),
                )

    def test_dynamic_parser_rejects_header_count_row_and_null_mutations(self) -> None:
        valid = """Dynamic section at offset 0x100 contains 2 entries:
  Tag        Type                         Name/Value
 0x1 (NEEDED) Shared library: [a.so]
 0x0 (NULL) 0x0
"""
        parse_dynamic(valid)
        for mutated in (
            valid.replace("contains 2 entries", "contains 3 entries"),
            valid.replace(" 0x1 (NEEDED) Shared library: [a.so]", " malformed dynamic row"),
            valid.replace(" 0x0 (NULL) 0x0", " 0x2 (PLTRELSZ) 24 (bytes)"),
            valid.replace(" 0x0 (NULL) 0x0", " 0x0 (NULL) 0x0\n 0x2 (PLTRELSZ) 24 (bytes)")
            .replace("contains 2 entries", "contains 3 entries"),
            valid.replace("Dynamic section at offset 0x100 contains 2 entries:", "Dynamic section malformed"),
            valid.replace("  Tag        Type                         Name/Value\n", "")
            + "  Tag        Type                         Name/Value\n",
        ):
            with self.subTest(mutated=mutated):
                self.assertGate(parse_dynamic, mutated)

    def test_configured_compiler_must_be_first_without_launcher(self) -> None:
        compiler = Path(sys.executable).resolve()
        require_direct_compiler([str(compiler), "-c", "fixture.c"], compiler, "selftest")
        self.assertGate(
            require_direct_compiler,
            ["/usr/bin/env", str(compiler), "-c", "fixture.c"],
            compiler,
            "selftest",
        )
        self.assertGate(
            require_direct_compiler,
            ["/usr/bin/env", str(compiler), "-Wl,-pie", "fixture.o"],
            compiler,
            "selftest-link",
        )
        require_direct_link_compiler(
            [":", "&&", str(compiler), "-Wl,-pie", "fixture.o", "&&", ":"],
            compiler,
            "selftest-cmake-link",
        )
        for tokens in (
            [":", "&&", "/usr/bin/env", str(compiler), "fixture.o", "&&", ":"],
            [":", "&&", str(compiler), "fixture.o", "&&", "/bin/true", "&&", ":"],
            [str(compiler), "fixture.o", "&&", "/bin/true"],
        ):
            with self.subTest(tokens=tokens):
                self.assertGate(
                    require_direct_link_compiler,
                    tokens,
                    compiler,
                    "selftest-cmake-link",
                )

    def test_tool_stderr_is_rejected_even_on_success(self) -> None:
        self.assertEqual(run_bounded_bytes((sys.executable, "-c", "pass")), b"")
        self.assertGate(
            run_bounded_bytes,
            (sys.executable, "-c", "import sys;sys.stderr.write('unexpected')"),
        )

    def test_artifact_snapshot_binds_exact_bytes_and_detects_source_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.elf"
            source.write_bytes(b"original-byte-image")
            snapshot_root = root / "snapshots"
            snapshot_root.mkdir()
            snapshot = snapshot_regular_file(source, snapshot_root, "artifact")
            self.assertEqual(snapshot.path.read_bytes(), b"original-byte-image")
            verify_snapshot_source_unchanged(snapshot)
            source.write_bytes(b"mutated-byte-image")
            self.assertGate(verify_snapshot_source_unchanged, snapshot)

    def test_ctest_contract_is_exact_for_each_frozen_r0_test(self) -> None:
        build_dir = Path("/build")
        source_dir = Path("/source")
        python_executable = Path("/usr/bin/python3")
        call_keywords = {
            "build_dir": build_dir,
            "source_dir": source_dir,
            "python_executable": python_executable,
        }
        scrub = ";".join(f"{name}=unset:" for name in known_loader_environment_variables())
        expected_commands = expected_r0_ctest_commands(build_dir, source_dir, python_executable)
        backtrace = f"{source_dir}/common/utils/memprof/tests/CMakeLists.txt;1;add_test;{source_dir};0;"

        def property_line(test_name: str, environment_scrub: str, timeout: int) -> str:
            return (
                f'set_tests_properties([=[{test_name}]=] PROPERTIES '
                f'ENVIRONMENT "LC_ALL=C" ENVIRONMENT_MODIFICATION "{environment_scrub}" '
                f'LABELS "memprof;r0" TIMEOUT "{timeout}" _BACKTRACE_TRIPLES "{backtrace}")'
            )

        lines: list[str] = []
        for test_name in EXPECTED_R0_CTESTS:
            arguments = " ".join(f'"{argument}"' for argument in expected_commands[test_name])
            lines.append(f"add_test([=[{test_name}]=] {arguments})")
            lines.append(property_line(test_name, scrub, EXPECTED_R0_CTEST_TIMEOUT_SECONDS[test_name]))
        valid = "\n".join(lines) + "\n"
        validate_ctest_contract_text(valid, **call_keywords)

        missing_entry = ";".join(
            f"{name}=unset:" for name in known_loader_environment_variables()[1:]
        )
        missing_one_variable = list(lines)
        missing_one_variable[1] = property_line(
            EXPECTED_R0_CTESTS[0],
            missing_entry,
            EXPECTED_R0_CTEST_TIMEOUT_SECONDS[EXPECTED_R0_CTESTS[0]],
        )
        self.assertGate(
            validate_ctest_contract_text,
            "\n".join(missing_one_variable) + "\n",
            **call_keywords,
        )
        self.assertGate(validate_ctest_contract_text, "\n".join(lines[2:]) + "\n", **call_keywords)
        self.assertGate(validate_ctest_contract_text, valid + lines[1] + "\n", **call_keywords)

        wrong_command = list(lines)
        wrong_command[0] = f'add_test([=[{EXPECTED_R0_CTESTS[0]}]=] "/bin/true")'
        self.assertGate(validate_ctest_contract_text, "\n".join(wrong_command) + "\n", **call_keywords)

        missing_timeout = list(lines)
        missing_timeout[1] = lines[1].replace(
            f' TIMEOUT "{EXPECTED_R0_CTEST_TIMEOUT_SECONDS[EXPECTED_R0_CTESTS[0]]}"',
            "",
        )
        self.assertGate(validate_ctest_contract_text, "\n".join(missing_timeout) + "\n", **call_keywords)

        wrong_timeout = list(lines)
        wrong_timeout[-1] = property_line(
            EXPECTED_R0_CTESTS[-1],
            scrub,
            120,
        )
        self.assertGate(validate_ctest_contract_text, "\n".join(wrong_timeout) + "\n", **call_keywords)

        missing_label = list(lines)
        missing_label[1] = lines[1].replace(' LABELS "memprof;r0"', "")
        self.assertGate(validate_ctest_contract_text, "\n".join(missing_label) + "\n", **call_keywords)

        for property_name, value in (
            ("DISABLED", "TRUE"),
            ("WILL_FAIL", "TRUE"),
            ("SKIP_RETURN_CODE", "0"),
            ("PASS_REGULAR_EXPRESSION", "pass"),
            ("FAIL_REGULAR_EXPRESSION", "fail"),
            ("SKIP_REGULAR_EXPRESSION", "skip"),
            ("WORKING_DIRECTORY", "/tmp"),
        ):
            with self.subTest(property_name=property_name):
                mutated = list(lines)
                mutated[1] = lines[1][:-1] + f' {property_name} "{value}")'
                self.assertGate(
                    validate_ctest_contract_text,
                    "\n".join(mutated) + "\n",
                    **call_keywords,
                )

        for alternate_statement in (
            'set_property(TEST [=[oai_memprof_r0_scripted_pair]=] PROPERTY DISABLED "TRUE")',
            'set_tests_properties([=[oai_memprof_r0_scripted_pair]=] '
            '[=[oai_memprof_r0_scripted_bounds]=] PROPERTIES WILL_FAIL "TRUE")',
            'add_test(oai_memprof_r0_scripted_pair "/bin/true")',
            'include("/tmp/ctest-injection.cmake")',
        ):
            with self.subTest(alternate_statement=alternate_statement):
                self.assertGate(
                    validate_ctest_contract_text,
                    valid + alternate_statement + "\n",
                    **call_keywords,
                )

        extra_labeled_test = (
            valid
            + 'add_test([=[unrelated_name]=] "/bin/true")\n'
            + 'set_tests_properties([=[unrelated_name]=] PROPERTIES LABELS "r0")\n'
        )
        self.assertGate(validate_ctest_contract_text, extra_labeled_test, **call_keywords)

    def test_response_file_cycle_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cycle.rsp"
            path.write_text("@cycle.rsp\n", encoding="utf-8")
            self.assertGate(expand_response_files, ["@cycle.rsp"], Path(temporary))

    def test_object_inputs_preserve_order_and_multiplicity(self) -> None:
        self.assertEqual(explicit_object_inputs(["cc", "a.o", "a.o", "b.o"]), ("a.o", "a.o", "b.o"))

    def test_final_link_normalization_admits_only_frozen_a01_delta(self) -> None:
        base = Path("/tmp")
        wrapper = base / "liboai_memprof_wrap_c.a"
        runtime = base / "liboai_memprof_runtime.so.1.0.0"
        common = ["cc", *PASSIVE_NEGATIVE_FLAGS, "-Wl,-pie"]
        a00_role = Role("a00-exe", base / "a00", base / "a00.map", False, False, "a00")
        a01_role = Role("a01-exe", base / "a01", base / "a01.map", True, False, "a01")
        a00 = [
            *common,
            "-Wl,-Map,/tmp/a00.map",
            "-Wl,--cref",
            "/tmp/fixture.o",
            "-o",
            "/tmp/a00",
            r"-Wl,-rpath,\$ORIGIN/..",
            "-ldl",
        ]
        a01 = [
            *common,
            "-Wl,-Map,/tmp/a01.map",
            "-Wl,--cref",
            *(f"-Wl,--wrap={api}" for api in APIS),
            "/tmp/fixture.o",
            "-o",
            "/tmp/a01",
            r"-Wl,-rpath,\$ORIGIN/..",
            "-ldl",
            str(wrapper),
            str(runtime),
        ]
        baseline = normalize_final_link(a00_role, a00, wrapper, runtime, base)
        treatment = normalize_final_link(a01_role, a01, wrapper, runtime, base)
        self.assertEqual(baseline, treatment)
        self.assertNotEqual(
            baseline,
            normalize_final_link(a01_role, [*a01, "-Wl,--as-needed"], wrapper, runtime, base),
        )
        self.assertGate(normalize_final_link, a01_role, [*a01, str(wrapper)], wrapper, runtime, base)
        for api in APIS:
            mutant = list(a01)
            mutant.remove(f"-Wl,--wrap={api}")
            with self.subTest(missing_wrap=api):
                self.assertGate(normalize_final_link, a01_role, mutant, wrapper, runtime, base)
        combined = list(a01)
        combined[combined.index("-Wl,--wrap=malloc")] = "-Wl,--wrap=malloc,--as-needed"
        self.assertGate(normalize_final_link, a01_role, combined, wrapper, runtime, base)

    def test_final_link_wrap_order_is_sensitive(self) -> None:
        base = Path("/tmp")
        wrapper = base / "liboai_memprof_wrap_c.a"
        runtime = base / "liboai_memprof_runtime.so.1.0.0"
        role = Role("a01-dso", base / "a01.so", base / "a01.map", True, True, "a01")
        tokens = [
            "cc",
            *PASSIVE_NEGATIVE_FLAGS,
            "-Wl,-Map,/tmp/a01.map",
            "-Wl,--wrap=calloc",
            "-Wl,--wrap=malloc",
            "-Wl,--wrap=realloc",
            "-Wl,--wrap=free",
            "/tmp/fixture.o",
            "-o",
            "/tmp/a01.so",
            str(wrapper),
            str(runtime),
        ]
        self.assertGate(normalize_final_link, role, tokens, wrapper, runtime, base)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--self-test", action="store_true")
    for name in ("a00-exe", "a01-exe", "a00-dso", "a01-dso", "runtime", "wrapper-archive"):
        result.add_argument(f"--{name}")
    for name in ("a00-exe-map", "a01-exe-map", "a00-dso-map", "a01-dso-map"):
        result.add_argument(f"--{name}")
    result.add_argument("--command-evidence")
    result.add_argument("--build-dir")
    result.add_argument("--source-dir")
    result.add_argument("--ninja-target", action="append", default=[])
    result.add_argument("--readelf", default="readelf")
    result.add_argument("--nm", default="nm")
    result.add_argument("--objdump", default="objdump")
    result.add_argument("--ar", default="ar")
    result.add_argument("--ninja", default="ninja")
    result.add_argument("--ld", default="ld")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SensitivitySelfTest)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    required = (
        "a00_exe",
        "a01_exe",
        "a00_dso",
        "a01_dso",
        "runtime",
        "wrapper_archive",
        "a00_exe_map",
        "a01_exe_map",
        "a00_dso_map",
        "a01_dso_map",
        "build_dir",
        "source_dir",
    )
    missing = [name.replace("_", "-") for name in required if getattr(args, name) is None]
    if missing:
        parser().error(f"missing required arguments: {', '.join(missing)}")
    try:
        validate(args)
    except GateError as error:
        print(f"R0_ELF_GATE_V2 fail: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
