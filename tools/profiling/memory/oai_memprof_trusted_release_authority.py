#!/usr/bin/env python3
"""Generate a pinned trusted-release authority from one clean Git checkout.

The archive semantic verifier deliberately treats its authority document and
the source bytes as externally pinned inputs.  This controller-side utility
creates those inputs without using worktree bytes to build authority rows:
every row is read from the caller-pinned Git tree.  It then independently
stable-reads the current files without following links, so publishing is
refused if the live checkout no longer matches those committed blobs.

The accompanying controller receipt is diagnostic provenance only.  It is not
an archive member and is never an input to archive_semantic_verifier_v1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import selectors
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


AUTHORITY_SCHEMA = "oai_memprof_trusted_release_authority_v1"
RECEIPT_SCHEMA = "oai_memprof_trusted_release_controller_receipt_v1"
AUTHORITY_FILENAME = "trusted-release-authority-v1.json"
RECEIPT_FILENAME = "trusted-release-authority-v1.controller-receipt.json"
MAX_SOURCE_BYTES = 4 << 20
MAX_SOURCE_TOTAL_BYTES = 16 << 20
_GIT_TIMEOUT_SECONDS = 15.0
_GIT_SMALL_OUTPUT_BYTES = 1 << 20
_HEX40_RE = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
_GIT_OBJECT_RE = re.compile(rb"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z", re.ASCII)

# This intentionally mirrors the exported contract in
# catalog_v1/integration/archive_semantic_verifier_v1.py.  The mapping is
# frozen here so that this generator stays controller-side and does not import
# executing worktree code as a source of trust.  validate_fixed_source_mapping
# rejects every population other than this exact verifier contract.
TRUSTED_RELEASE_SOURCE_PATHS: dict[str, str] = {
    "definition/oai-memprof-container-wire-v1.py": "tools/profiling/memory/oai_memprof_container_wire.py",
    "definition/catalog-v1/semantic-catalog-v1.py": "tools/profiling/memory/catalog_v1/semantic/semantic_catalog_v1.py",
    "definition/catalog-v1/event-classifier-v1.py": "tools/profiling/memory/catalog_v1/semantic/event_classifier_v1.py",
    "definition/catalog-v1/coverage-catalog-v1.py": "tools/profiling/memory/catalog_v1/coverage/coverage_catalog_v1.py",
    "definition/catalog-v1/effective-config-v1.py": "tools/profiling/memory/catalog_v1/config/effective_config_v1.py",
    "definition/catalog-v1/selection-rule-v1.py": "tools/profiling/memory/catalog_v1/sampling/selection_rule_v1.py",
    "definition/catalog-v1/callsite-catalog-v1.py": "tools/profiling/memory/catalog_v1/callsite/callsite_catalog_v1.py",
    "definition/catalog-v1/runtime-catalog-v1.py": "tools/profiling/memory/catalog_v1/runtime/runtime_catalog_v1.py",
    "definition/catalog-v1/diagnostic-instance-v1.py": "tools/profiling/memory/catalog_v1/diagnostics/diagnostic_instance_v1.py",
    "definition/catalog-v1/status-chain-v1.py": "tools/profiling/memory/catalog_v1/status/status_chain_v1.py",
    "definition/oai-memprof-process-handoff-v1.py": "tools/profiling/memory/oai_memprof_process_handoff.py",
    "definition/oai-memprof-build-evidence-v1.py": "tools/profiling/memory/oai_memprof_build_evidence.py",
    "definition/oai-memprof-archive-composer-v1.py": "tools/profiling/memory/oai_memprof_archive_composer.py",
    "definition/archive-semantic-verifier-v1.py": "tools/profiling/memory/catalog_v1/integration/archive_semantic_verifier_v1.py",
    "definition/oai-memprof-softmodem-launcher-v1.py": "tools/profiling/memory/oai_memprof_softmodem_launcher.py",
}
if len(TRUSTED_RELEASE_SOURCE_PATHS) != 15:
    raise RuntimeError("trusted-release source population must contain exactly fifteen paths")


class TrustedReleaseAuthorityError(RuntimeError):
    """Fail-closed rejection while producing one controller authority."""


@dataclass(frozen=True)
class GitBlobIdentity:
    """One fixed source's identity in the caller-pinned Git tree."""

    authority_path: str
    repository_path: str
    oid: str
    mode: str
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class TrustedReleaseAuthorityResult:
    """Published authority, non-trust receipt, and immutable generation facts."""

    authority_path: Path
    receipt_path: Path
    authority_bytes: bytes
    authority_sha256: str
    commit: str
    tree: str
    source_blobs: tuple[GitBlobIdentity, ...]


def _fail(detail: str) -> None:
    raise TrustedReleaseAuthorityError(detail)


def _canonical_relative_path(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{where}: nonempty relative POSIX path required")
    if (
        value.startswith("/")
        or "\\" in value
        or any(part in ("", ".", "..") for part in value.split("/"))
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        _fail(f"{where}: canonical relative POSIX path required")
    return value


def validate_fixed_source_mapping(mapping: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    """Return the sorted verifier population or reject missing, extra, or bad rows.

    The generator itself has no caller-configurable mapping: making it
    configurable could silently expand, contract, or redirect the verifier's
    fixed authority population.  This callable validation seam makes that
    invariant explicit and testable.
    """

    if not isinstance(mapping, Mapping):
        _fail("source mapping: mapping required")
    normalized: dict[str, str] = {}
    for authority_path, repository_path in mapping.items():
        archive = _canonical_relative_path(authority_path, "source mapping archive path")
        repository = _canonical_relative_path(repository_path, "source mapping repository path")
        if archive in normalized:
            _fail("source mapping: duplicate archive path")
        normalized[archive] = repository
    if normalized != TRUSTED_RELEASE_SOURCE_PATHS:
        _fail("source mapping: exact fifteen-path verifier population required")
    return tuple(sorted(normalized.items()))


def _hex40(value: Any, where: str) -> str:
    if not isinstance(value, str) or _HEX40_RE.fullmatch(value) is None:
        _fail(f"{where}: lowercase 40-hex Git object ID required")
    return value


def _pinned_directory(value: str | os.PathLike[str], where: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        _fail(f"{where}: absolute path required")
    normalized = os.path.normpath(os.fspath(path))
    if normalized != os.fspath(path):
        _fail(f"{where}: normalized absolute path required")
    try:
        resolved = os.fspath(path.resolve(strict=True))
    except (OSError, RuntimeError) as error:
        _fail(f"{where}: strict resolution failed: {error}")
    if resolved != normalized:
        _fail(f"{where}: symlinked path component is forbidden")
    try:
        before = os.lstat(normalized)
    except OSError as error:
        _fail(f"{where}: unavailable: {error}")
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        _fail(f"{where}: non-symlink directory required")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow == 0:
        _fail(f"{where}: O_NOFOLLOW support is required")
    try:
        fd = os.open(normalized, flags | nofollow)
    except OSError as error:
        _fail(f"{where}: no-follow open failed: {error}")
    try:
        opened = os.fstat(fd)
        if not _same_identity(before, opened):
            _fail(f"{where}: changed while opening")
    finally:
        os.close(fd)
    return Path(normalized)


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
    )


def _git_executable() -> str:
    executable = shutil.which("git", path="/usr/bin:/bin")
    if executable is None:
        _fail("git executable unavailable in controlled PATH")
    return executable


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    try:
        process.kill()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass


def _run_git(
    repository_root: Path,
    arguments: Sequence[str],
    *,
    output_limit: int = _GIT_SMALL_OUTPUT_BYTES,
) -> bytes:
    """Run a fixed Git command with a timeout and hard stdout/stderr bounds."""

    command = [
        _git_executable(),
        "--no-optional-locks",
        "-C",
        os.fspath(repository_root),
        *arguments,
    ]
    environment = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "LANG": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            env=environment,
        )
    except OSError as error:
        _fail(f"git launch failed: {error}")
    assert process.stdout is not None and process.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + _GIT_TIMEOUT_SECONDS
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_process(process)
                _fail("git command exceeded bounded timeout")
            events = selector.select(remaining)
            if not events:
                continue
            for key, _unused in events:
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                destination = captured[key.data]
                destination.extend(chunk)
                if len(destination) > output_limit:
                    _terminate_process(process)
                    _fail("git command exceeded bounded output")
        status = process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        _terminate_process(process)
        _fail("git command did not exit after bounded I/O")
    except OSError as error:
        _terminate_process(process)
        _fail(f"git command I/O failed: {error}")
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
    stdout = bytes(captured["stdout"])
    stderr = bytes(captured["stderr"])
    if status != 0:
        detail = stderr.decode("utf-8", "replace").strip()
        _fail(f"git command failed ({arguments[0]}): {detail or 'exit status ' + str(status)}")
    return stdout


def _single_git_line(raw: bytes, where: str) -> str:
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        _fail(f"{where}: exactly one Git output line required")
    try:
        return raw[:-1].decode("ascii", "strict")
    except UnicodeDecodeError as error:
        _fail(f"{where}: ASCII output required: {error}")


def _verify_pinned_repository(repository_root: Path, commit: str, tree: str) -> None:
    top = _run_git(repository_root, ("rev-parse", "--show-toplevel"))
    top_text = os.fsdecode(top[:-1]) if top.endswith(b"\n") else ""
    if top != os.fsencode(os.fspath(repository_root)) + b"\n":
        _fail(f"repository top differs from pinned root: {top_text!r}")
    status = _run_git(
        repository_root,
        ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
    )
    if status:
        _fail("repository status is not empty, including untracked files")
    head = _single_git_line(_run_git(repository_root, ("rev-parse", "HEAD")), "HEAD")
    if head != commit:
        _fail("HEAD differs from caller-pinned commit")
    commit_object = _single_git_line(
        _run_git(repository_root, ("rev-parse", "--verify", f"{commit}^{{commit}}")),
        "commit object",
    )
    if commit_object != commit:
        _fail("caller-pinned commit does not resolve exactly to a commit object")
    commit_tree = _single_git_line(
        _run_git(repository_root, ("rev-parse", "--verify", f"{commit}^{{tree}}")),
        "commit tree",
    )
    if commit_tree != tree:
        _fail("caller-pinned commit tree differs from caller-pinned tree")
    index_tree = _single_git_line(_run_git(repository_root, ("write-tree",)), "index tree")
    if index_tree != tree:
        _fail("index tree differs from caller-pinned tree")
    raw_commit = _run_git(repository_root, ("cat-file", "commit", commit))
    expected_first_line = b"tree " + tree.encode("ascii")
    if raw_commit.split(b"\n", 1)[0] != expected_first_line:
        _fail("raw commit first line does not bind caller-pinned tree")


def _git_blob_from_tree(
    repository_root: Path, tree: str, authority_path: str, repository_path: str
) -> tuple[GitBlobIdentity, bytes]:
    listing = _run_git(
        repository_root,
        ("ls-tree", "-z", "--full-tree", tree, "--", repository_path),
        output_limit=8192,
    )
    if not listing.endswith(b"\0"):
        _fail(f"Git tree listing is not NUL-terminated: {repository_path}")
    records = listing[:-1].split(b"\0")
    if len(records) != 1 or not records[0]:
        _fail(f"Git tree listing must contain exactly one source: {repository_path}")
    header, separator, listed_path = records[0].partition(b"\t")
    parts = header.split(b" ")
    expected_path = repository_path.encode("utf-8")
    if separator != b"\t" or len(parts) != 3 or listed_path != expected_path:
        _fail(f"Git tree listing is malformed or redirected: {repository_path}")
    mode_raw, object_type, oid_raw = parts
    if mode_raw not in (b"100644", b"100755") or object_type != b"blob":
        _fail(f"Git tree source is not a regular blob: {repository_path}")
    if _GIT_OBJECT_RE.fullmatch(oid_raw) is None:
        _fail(f"Git tree source has malformed blob object ID: {repository_path}")
    oid = oid_raw.decode("ascii")
    size_raw = _run_git(
        repository_root, ("cat-file", "-s", oid), output_limit=128
    )
    if re.fullmatch(rb"[0-9]+\n", size_raw) is None:
        _fail(f"Git blob size is malformed: {repository_path}")
    byte_count = int(size_raw[:-1])
    if not 0 < byte_count <= MAX_SOURCE_BYTES:
        _fail(f"Git blob bytes outside authority bound: {repository_path}")
    blob = _run_git(
        repository_root,
        ("cat-file", "blob", oid),
        output_limit=byte_count,
    )
    if len(blob) != byte_count:
        _fail(f"Git blob size changed or differs from cat-file size: {repository_path}")
    return (
        GitBlobIdentity(
            authority_path=authority_path,
            repository_path=repository_path,
            oid=oid,
            mode=mode_raw.decode("ascii"),
            byte_count=byte_count,
            sha256=hashlib.sha256(blob).hexdigest(),
        ),
        blob,
    )


def _open_directory_no_follow(path: Path, where: str) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow == 0:
        _fail(f"{where}: O_NOFOLLOW support is required")
    try:
        before = os.lstat(path)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            _fail(f"{where}: non-symlink directory required")
        fd = os.open(path, flags | nofollow)
    except OSError as error:
        _fail(f"{where}: no-follow directory open failed: {error}")
    opened = os.fstat(fd)
    if not _same_identity(before, opened):
        os.close(fd)
        _fail(f"{where}: directory changed while opening")
    return fd, opened


def _read_exact_bounded(fd: int, expected_size: int, where: str) -> bytes:
    chunks: list[bytes] = []
    remaining = expected_size + 1
    while remaining:
        chunk = os.read(fd, min(65536, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    raw = b"".join(chunks)
    if len(raw) != expected_size:
        _fail(f"{where}: bytes changed while reading")
    return raw


def read_stable_current_source(
    repository_root: str | os.PathLike[str], repository_path: str, expected: bytes
) -> bytes:
    """No-follow stable-read one current source and require its Git blob bytes.

    Every parent component beneath ``repository_root`` is opened by descriptor
    without following a link and checked again after the leaf read.  This
    cannot defeat a same-UID attacker that rewrites all observations between
    checks; that residual limitation is recorded in the receipt.
    """

    if not isinstance(expected, bytes) or not 0 < len(expected) <= MAX_SOURCE_BYTES:
        _fail("current source expected bytes outside authority bound")
    root = _pinned_directory(repository_root, "repository root")
    relative = _canonical_relative_path(repository_path, "current source path")
    root_fd, root_identity = _open_directory_no_follow(root, "repository root")
    opened: list[tuple[int, str, int, os.stat_result]] = []
    current_fd = root_fd
    leaf_fd = -1
    try:
        components = relative.split("/")
        for component in components[:-1]:
            before = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                _fail(f"current source parent is not a non-symlink directory: {relative}")
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
            child_fd = os.open(component, flags, dir_fd=current_fd)
            opened_identity = os.fstat(child_fd)
            if not _same_identity(before, opened_identity):
                os.close(child_fd)
                _fail(f"current source parent changed while opening: {relative}")
            opened.append((current_fd, component, child_fd, opened_identity))
            current_fd = child_fd
        leaf = components[-1]
        before_leaf = os.stat(leaf, dir_fd=current_fd, follow_symlinks=False)
        if stat.S_ISLNK(before_leaf.st_mode) or not stat.S_ISREG(before_leaf.st_mode):
            _fail(f"current source leaf is not a non-symlink regular file: {relative}")
        if before_leaf.st_size != len(expected):
            _fail(f"current source size differs from committed blob: {relative}")
        leaf_fd = os.open(
            leaf,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=current_fd,
        )
        opened_leaf = os.fstat(leaf_fd)
        if not _same_identity(before_leaf, opened_leaf) or not stat.S_ISREG(opened_leaf.st_mode):
            _fail(f"current source leaf changed while opening: {relative}")
        raw = _read_exact_bounded(leaf_fd, len(expected), f"current source {relative}")
        after_leaf = os.stat(leaf, dir_fd=current_fd, follow_symlinks=False)
        if not _same_identity(opened_leaf, after_leaf):
            _fail(f"current source leaf changed while reading: {relative}")
        if raw != expected:
            _fail(f"current source bytes differ from committed blob: {relative}")
        for parent_fd, name, child_fd, child_identity in reversed(opened):
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if not _same_identity(child_identity, current):
                _fail(f"current source parent changed while reading: {relative}")
        current_root = os.lstat(root)
        if not _same_identity(root_identity, current_root):
            _fail(f"repository root changed while reading current source: {relative}")
        return raw
    except OSError as error:
        _fail(f"current source no-follow read failed: {relative}: {error}")
    finally:
        if leaf_fd >= 0:
            os.close(leaf_fd)
        for _parent_fd, _name, child_fd, _identity in reversed(opened):
            os.close(child_fd)
        os.close(root_fd)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Canonical ASCII-only JSON used by the verifier's authority decoder."""

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        _fail(f"canonical JSON encoding failed: {error}")
    try:
        raw = text.encode("utf-8", "strict") + b"\n"
    except UnicodeEncodeError as error:
        _fail(f"canonical JSON UTF-8 encoding failed: {error}")
    if b"\r" in raw:
        _fail("canonical JSON contains forbidden carriage return")
    return raw


def _write_no_overwrite(directory_fd: int, filename: str, raw: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        fd = os.open(filename, flags, 0o600, dir_fd=directory_fd)
    except OSError as error:
        _fail(f"publication create refused for {filename}: {error}")
    try:
        written = 0
        while written < len(raw):
            count = os.write(fd, raw[written:])
            if count <= 0:
                _fail(f"publication write failed for {filename}")
            written += count
        os.fsync(fd)
    except OSError as error:
        _fail(f"publication write/fsync failed for {filename}: {error}")
    finally:
        os.close(fd)


def _stable_reread_output(directory_fd: int, filename: str, expected: bytes) -> None:
    try:
        before = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            _fail(f"published output is not a regular file: {filename}")
        if before.st_size != len(expected):
            _fail(f"published output size differs after write: {filename}")
        fd = os.open(
            filename,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
    except OSError as error:
        _fail(f"published output reread failed for {filename}: {error}")
    try:
        opened = os.fstat(fd)
        if not _same_identity(before, opened):
            _fail(f"published output changed while opening: {filename}")
        raw = _read_exact_bounded(fd, len(expected), f"published output {filename}")
        after = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        if not _same_identity(opened, after) or raw != expected:
            _fail(f"published output changed while rereading: {filename}")
    finally:
        os.close(fd)


def _generator_identity(repository_root: Path, tree: str) -> dict[str, Any]:
    """Best-effort generator Git identity for a receipt, never an authority row."""

    try:
        generator_file = Path(__file__).resolve()
        relative = generator_file.relative_to(repository_root).as_posix()
        relative = _canonical_relative_path(relative, "generator repository path")
        identity, _raw = _git_blob_from_tree(
            repository_root, tree, "controller/generator", relative
        )
        return {
            "status": "available_from_pinned_tree",
            "repository_path": relative,
            "blob": {
                "oid": identity.oid,
                "mode": identity.mode,
                "bytes": identity.byte_count,
                "sha256": identity.sha256,
            },
        }
    except (OSError, ValueError, TrustedReleaseAuthorityError):
        return {
            "status": "unavailable_outside_or_absent_from_pinned_tree",
            "repository_path": None,
            "blob": None,
        }


def generate_trusted_release_authority(
    *,
    repository_root: str | os.PathLike[str],
    commit: str,
    tree: str,
    output_directory: str | os.PathLike[str],
) -> TrustedReleaseAuthorityResult:
    """Publish one verifier-compatible authority and a separate controller receipt.

    ``repository_root``, ``commit``, ``tree``, and an existing empty
    ``output_directory`` are caller-pinned inputs.  The function never creates
    output directories, never overwrites a file, and leaves a partial result in
    place if a later receipt publication fails; retry therefore requires a new
    fresh output directory.
    """

    root = _pinned_directory(repository_root, "repository root")
    output = _pinned_directory(output_directory, "output directory")
    commit = _hex40(commit, "commit")
    tree = _hex40(tree, "tree")
    mapping = validate_fixed_source_mapping(TRUSTED_RELEASE_SOURCE_PATHS)
    try:
        initial_contents = os.listdir(output)
    except OSError as error:
        _fail(f"output directory listing failed: {error}")
    if initial_contents:
        _fail("output directory must be fresh and empty")
    _verify_pinned_repository(root, commit, tree)

    identities: list[GitBlobIdentity] = []
    committed_sources: dict[str, bytes] = {}
    total = 0
    for authority_path, repository_path in mapping:
        identity, blob = _git_blob_from_tree(root, tree, authority_path, repository_path)
        total += identity.byte_count
        if total > MAX_SOURCE_TOTAL_BYTES:
            _fail("trusted-release source aggregate bytes exceeded")
        identities.append(identity)
        committed_sources[repository_path] = blob

    # This live read is strictly a post-Git-object equality proof.  It never
    # contributes bytes used in source rows or the authority document.
    for identity in identities:
        read_stable_current_source(root, identity.repository_path, committed_sources[identity.repository_path])

    authority = {
        "schema": AUTHORITY_SCHEMA,
        "git": {"clean": True, "commit": commit, "tree": tree},
        "sources": [
            {
                "bytes": identity.byte_count,
                "path": identity.authority_path,
                "sha256": identity.sha256,
            }
            for identity in identities
        ],
    }
    authority_bytes = _canonical_json_bytes(authority)
    authority_sha256 = hashlib.sha256(authority_bytes).hexdigest()
    generator = _generator_identity(root, tree)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "generated_from_clean_pinned_git_tree",
        "not_authority_input": True,
        "authority": {
            "filename": AUTHORITY_FILENAME,
            "bytes": len(authority_bytes),
            "sha256": authority_sha256,
        },
        "git": {"clean": True, "commit": commit, "tree": tree},
        "source_mapping": [
            {"authority_path": authority_path, "repository_path": repository_path}
            for authority_path, repository_path in mapping
        ],
        "git_blobs": [
            {
                "authority_path": identity.authority_path,
                "repository_path": identity.repository_path,
                "oid": identity.oid,
                "mode": identity.mode,
                "bytes": identity.byte_count,
                "sha256": identity.sha256,
            }
            for identity in identities
        ],
        "generator": generator,
        "limitations": [
            "The receipt is controller provenance only and is not a trusted archive/verifier input.",
            "Git object and worktree equality checks do not prove Git authorship, remote provenance, or durable storage.",
            "No same-UID process can be excluded from replacing paths between bounded observations.",
        ],
    }
    receipt_bytes = _canonical_json_bytes(receipt)

    output_fd, output_identity = _open_directory_no_follow(output, "output directory")
    try:
        if os.listdir(output_fd):
            _fail("output directory changed and is no longer empty")
        _write_no_overwrite(output_fd, AUTHORITY_FILENAME, authority_bytes)
        _stable_reread_output(output_fd, AUTHORITY_FILENAME, authority_bytes)
        _write_no_overwrite(output_fd, RECEIPT_FILENAME, receipt_bytes)
        _stable_reread_output(output_fd, RECEIPT_FILENAME, receipt_bytes)
        current_output = os.lstat(output)
        if not _same_identity(output_identity, current_output):
            _fail("output directory changed while publishing")
        try:
            os.fsync(output_fd)
        except OSError as error:
            _fail(f"output directory fsync failed: {error}")
    finally:
        os.close(output_fd)
    return TrustedReleaseAuthorityResult(
        authority_path=output / AUTHORITY_FILENAME,
        receipt_path=output / RECEIPT_FILENAME,
        authority_bytes=authority_bytes,
        authority_sha256=authority_sha256,
        commit=commit,
        tree=tree,
        source_blobs=tuple(identities),
    )


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a clean Git-object trusted-release authority and receipt."
    )
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded CLI; it prints only the external authority pin summary."""

    arguments = _parse_arguments(argv)
    try:
        result = generate_trusted_release_authority(
            repository_root=arguments.repository_root,
            commit=arguments.commit,
            tree=arguments.tree,
            output_directory=arguments.output_directory,
        )
    except TrustedReleaseAuthorityError as error:
        print(f"trusted-release authority: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "authority_path": os.fspath(result.authority_path),
                "authority_sha256": result.authority_sha256,
                "receipt_path": os.fspath(result.receipt_path),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
