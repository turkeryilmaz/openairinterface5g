#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Atomically offload a completed profiler batch to a durable archive.

The active batch and destination are hashed independently before publication.
The source is removed only after the atomically published destination has also
been re-read and matched to the original source manifest.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


CHUNK_SIZE = 8 * 1024 * 1024
AT_FDCWD = -100
RENAME_NOREPLACE = 1
BATCH_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
REQUIRED_POLICY_FIELDS = {
    "active_root",
    "archive_root",
    "archive_mount",
    "archive_uuid",
    "inventory_root",
    "archive_free_reserve_bytes",
}


@dataclass(frozen=True)
class StoragePolicy:
    active_root: Path
    archive_root: Path
    archive_mount: Path
    archive_uuid: str
    inventory_root: Path
    archive_free_reserve_bytes: int


@dataclass(frozen=True)
class TreeRecord:
    path: str
    kind: str
    size_bytes: int
    mtime_ns: int
    mode: int
    uid: int
    gid: int
    sha256: str


def absolute_path(value: object, field: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{field} must be an absolute path")
    return path.resolve(strict=False)


def load_policy(path: Path) -> StoragePolicy:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("storage policy must be a JSON object")
    missing = sorted(REQUIRED_POLICY_FIELDS - set(value))
    unexpected = sorted(set(value) - REQUIRED_POLICY_FIELDS)
    if missing or unexpected:
        raise ValueError(
            f"storage policy fields mismatch: missing={missing} unexpected={unexpected}"
        )
    archive_uuid = str(value["archive_uuid"]).lower()
    try:
        uuid.UUID(archive_uuid)
    except ValueError as error:
        raise ValueError("archive_uuid must be a valid UUID") from error
    reserve = value["archive_free_reserve_bytes"]
    if not isinstance(reserve, int) or isinstance(reserve, bool) or reserve < 0:
        raise ValueError("archive_free_reserve_bytes must be a nonnegative integer")
    policy = StoragePolicy(
        active_root=absolute_path(value["active_root"], "active_root"),
        archive_root=absolute_path(value["archive_root"], "archive_root"),
        archive_mount=absolute_path(value["archive_mount"], "archive_mount"),
        archive_uuid=archive_uuid,
        inventory_root=absolute_path(value["inventory_root"], "inventory_root"),
        archive_free_reserve_bytes=reserve,
    )
    try:
        policy.archive_root.relative_to(policy.archive_mount)
        policy.inventory_root.relative_to(policy.archive_mount)
    except ValueError as error:
        raise ValueError("archive_root and inventory_root must be below archive_mount") from error
    return policy


def batch_paths(policy: StoragePolicy, batch_name: str) -> tuple[Path, Path]:
    if not BATCH_NAME.fullmatch(batch_name):
        raise ValueError("batch name must contain only letters, digits, '.', '_' or '-'")
    return policy.active_root / batch_name, policy.archive_root / batch_name


def device_id(path: Path) -> int:
    return path.stat().st_dev


def is_directory_noexcept(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        return path.is_dir()
    except OSError:
        return False


def validate_archive_mount(policy: StoragePolicy) -> dict[str, str]:
    result = subprocess.run(
        [
            "findmnt",
            "--json",
            "--target",
            str(policy.archive_mount),
            "--output",
            "TARGET,SOURCE,FSTYPE,OPTIONS,UUID",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    filesystems = json.loads(result.stdout).get("filesystems", [])
    candidates = [
        item
        for item in filesystems
        if Path(str(item.get("target", ""))).resolve(strict=False)
        == policy.archive_mount
        and str(item.get("uuid", "")).lower() == policy.archive_uuid
        and "rw" in str(item.get("options", "")).split(",")
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "archive mount identity is not uniquely writable: "
            f"target={policy.archive_mount} uuid={policy.archive_uuid} "
            f"matching_rw_entries={len(candidates)}"
        )
    candidate = candidates[0]
    return {
        "target": str(candidate.get("target", "")),
        "source": str(candidate.get("source", "")),
        "fstype": str(candidate.get("fstype", "")),
        "options": str(candidate.get("options", "")),
        "uuid": str(candidate.get("uuid", "")).lower(),
    }


def rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename without replacing a destination that races into place."""
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) is unavailable") from error
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            f"{source} -> {destination}",
        )


def stable_digest(path: Path, metadata: os.stat_result) -> str:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
        )
        expected = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_size,
            metadata.st_mtime_ns,
        )
        if identity != expected or not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"file changed before hashing: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, CHUNK_SIZE):
            digest.update(chunk)
        after = os.fstat(descriptor)
        observed = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
        )
        if observed != identity:
            raise RuntimeError(f"file changed while hashing: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def snapshot_tree(root: Path, hash_files: bool = True) -> list[TreeRecord]:
    root = root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"batch source must be a real directory: {root}")
    records: list[TreeRecord] = []
    for directory, names, files in os.walk(root, followlinks=False):
        names.sort()
        files.sort()
        directory_path = Path(directory)
        directory_metadata = directory_path.lstat()
        if not stat.S_ISDIR(directory_metadata.st_mode):
            raise ValueError(f"unsupported non-directory path: {directory_path}")
        records.append(
            TreeRecord(
                path=directory_path.relative_to(root).as_posix(),
                kind="directory",
                size_bytes=0,
                mtime_ns=directory_metadata.st_mtime_ns,
                mode=stat.S_IMODE(directory_metadata.st_mode),
                uid=directory_metadata.st_uid,
                gid=directory_metadata.st_gid,
                sha256="",
            )
        )
        for name in names:
            child = directory_path / name
            if not stat.S_ISDIR(child.lstat().st_mode):
                raise ValueError(f"unsupported non-directory path: {child}")
        for name in files:
            child = directory_path / name
            child_metadata = child.lstat()
            if not stat.S_ISREG(child_metadata.st_mode):
                raise ValueError(f"unsupported non-regular path: {child}")
            records.append(
                TreeRecord(
                    path=child.relative_to(root).as_posix(),
                    kind="file",
                    size_bytes=child_metadata.st_size,
                    mtime_ns=child_metadata.st_mtime_ns,
                    mode=stat.S_IMODE(child_metadata.st_mode),
                    uid=child_metadata.st_uid,
                    gid=child_metadata.st_gid,
                    sha256=stable_digest(child, child_metadata) if hash_files else "",
                )
            )
    return records


def compare_snapshots(
    expected: list[TreeRecord],
    observed: list[TreeRecord],
    expected_label: str,
    observed_label: str,
) -> None:
    expected_by_path = {record.path: record for record in expected}
    observed_by_path = {record.path: record for record in observed}
    if len(expected_by_path) != len(expected) or len(observed_by_path) != len(observed):
        raise RuntimeError("duplicate paths in tree snapshot")
    missing = sorted(set(expected_by_path) - set(observed_by_path))
    unexpected = sorted(set(observed_by_path) - set(expected_by_path))
    mismatched = [
        path
        for path in sorted(set(expected_by_path) & set(observed_by_path))
        if expected_by_path[path] != observed_by_path[path]
    ]
    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"tree mismatch {expected_label}->{observed_label}: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"mismatched={len(mismatched)} first_missing={missing[:1]} "
            f"first_unexpected={unexpected[:1]} first_mismatched={mismatched[:1]}"
        )


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def write_snapshot(path: Path, records: list[TreeRecord]) -> None:
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), sort_keys=True, separators=(",", ":")))
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def sync_all() -> None:
    os.sync()


def run_rsync(source: Path, staging: Path, log_path: Path) -> None:
    staging.mkdir(parents=False)
    command = [
        "rsync",
        "-aHAX",
        "--numeric-ids",
        "--protect-args",
        "--stats",
        "--",
        f"{source}/",
        f"{staging}/",
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    write_json_atomic(
        log_path,
        {
            "argv": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
    )
    if result.returncode != 0:
        raise RuntimeError(f"rsync failed with status {result.returncode}")


def restore_directory_mtimes(staging: Path, source_records: list[TreeRecord]) -> None:
    """Restore nanosecond directory mtimes that rsync may quick-check by second."""
    directories = [record for record in source_records if record.kind == "directory"]
    for record in sorted(directories, key=lambda item: item.path.count("/"), reverse=True):
        path = staging if record.path == "." else staging / record.path
        metadata = path.stat()
        os.utime(
            path,
            ns=(metadata.st_atime_ns, record.mtime_ns),
            follow_symlinks=False,
        )


def snapshot_summary(records: list[TreeRecord]) -> dict[str, int]:
    return {
        "directories": sum(record.kind == "directory" for record in records),
        "files": sum(record.kind == "file" for record in records),
        "bytes": sum(record.size_bytes for record in records if record.kind == "file"),
    }


def manifest_sha256(path: Path) -> str:
    return stable_digest(path, path.stat())


def transaction_name(batch_name: str) -> str:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f")
    return f"{timestamp}_{batch_name}_{uuid.uuid4().hex[:8]}"


def preflight(
    policy: StoragePolicy,
    batch_name: str,
    hash_files: bool,
    mount: dict[str, str] | None = None,
) -> tuple[Path, Path, dict[str, str], list[TreeRecord]]:
    source, destination = batch_paths(policy, batch_name)
    mount = mount or validate_archive_mount(policy)
    if not policy.active_root.is_dir():
        raise FileNotFoundError(f"active_root does not exist: {policy.active_root}")
    if not source.is_dir() or source.is_symlink():
        raise FileNotFoundError(f"completed batch does not exist: {source}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"archive destination already exists: {destination}")
    if device_id(policy.archive_mount) == device_id(policy.active_root):
        raise RuntimeError("active_root and archive_mount must use different filesystems")
    records = snapshot_tree(source, hash_files=hash_files)
    required = snapshot_summary(records)["bytes"] + policy.archive_free_reserve_bytes
    available = shutil.disk_usage(policy.archive_mount).free
    if available < required:
        raise RuntimeError(
            f"insufficient archive space: available={available} required={required}"
        )
    return source, destination, mount, records


def plan_offload(policy: StoragePolicy, batch_name: str) -> dict[str, Any]:
    source, destination, mount, records = preflight(policy, batch_name, hash_files=False)
    return {
        "status": "dry_run",
        "batch": batch_name,
        "source": str(source),
        "destination": str(destination),
        "mount": mount,
        "reserve_bytes": policy.archive_free_reserve_bytes,
        "archive_free_bytes": shutil.disk_usage(policy.archive_mount).free,
        "tree": snapshot_summary(records),
    }


def offload_batch(
    policy: StoragePolicy,
    batch_name: str,
    remove_source: bool,
    transaction_id: str | None = None,
) -> dict[str, Any]:
    transaction_id = transaction_id or transaction_name(batch_name)
    if not BATCH_NAME.fullmatch(transaction_id):
        raise ValueError("transaction_id contains unsupported characters")
    initial_mount = validate_archive_mount(policy)
    policy.archive_root.mkdir(parents=True, exist_ok=True)
    policy.inventory_root.mkdir(parents=True, exist_ok=True)
    ledger = policy.inventory_root / transaction_id
    ledger.mkdir()
    status_path = ledger / "status.json"
    started_at = datetime.now().astimezone().isoformat()
    staging = policy.archive_root / f".incoming-{transaction_id}"
    source: Path | None = None
    destination: Path | None = None
    retired: Path | None = None
    phase = "preflight"
    try:
        write_json_atomic(
            status_path,
            {
                "status": "running",
                "phase": phase,
                "batch": batch_name,
                "started_at": started_at,
            },
        )
        source, destination, mount, source_pre = preflight(
            policy,
            batch_name,
            hash_files=True,
            mount=initial_mount,
        )
        write_snapshot(ledger / "source_pre.jsonl", source_pre)

        phase = "copy_to_staging"
        run_rsync(source, staging, ledger / "rsync.json")
        restore_directory_mtimes(staging, source_pre)
        sync_all()

        phase = "verify_staging"
        staging_snapshot = snapshot_tree(staging)
        write_snapshot(ledger / "staging.jsonl", staging_snapshot)
        compare_snapshots(source_pre, staging_snapshot, "source_pre", "staging")

        phase = "verify_source_stability"
        source_post = snapshot_tree(source)
        write_snapshot(ledger / "source_post.jsonl", source_post)
        compare_snapshots(source_pre, source_post, "source_pre", "source_post")

        phase = "atomic_publication"
        if validate_archive_mount(policy) != initial_mount:
            raise RuntimeError("archive mount identity changed before publication")
        if device_id(staging) != device_id(policy.archive_mount):
            raise RuntimeError("staging is no longer on the archive filesystem")
        rename_noreplace(staging, destination)
        fsync_directory(policy.archive_root)
        sync_all()

        phase = "verify_published_destination"
        destination_post = snapshot_tree(destination)
        write_snapshot(ledger / "destination_post.jsonl", destination_post)
        compare_snapshots(
            source_pre,
            destination_post,
            "source_pre",
            "destination_post",
        )

        phase = "verify_source_before_reclamation"
        source_final = snapshot_tree(source)
        write_snapshot(ledger / "source_final.jsonl", source_final)
        compare_snapshots(
            source_pre,
            source_final,
            "source_pre",
            "source_final",
        )

        phase = "source_reclamation"
        if remove_source:
            if validate_archive_mount(policy) != initial_mount:
                raise RuntimeError("archive mount identity changed before reclamation")
            if device_id(destination) != device_id(policy.archive_mount):
                raise RuntimeError("destination is no longer on the archive filesystem")
            retired = policy.active_root / f".offloaded-{transaction_id}"
            if retired.exists() or retired.is_symlink():
                raise FileExistsError(f"retired source path already exists: {retired}")
            rename_noreplace(source, retired)
            fsync_directory(policy.active_root)
            shutil.rmtree(retired)
            fsync_directory(policy.active_root)
            retired = None

        phase = "complete"
        manifests = {
            name: manifest_sha256(ledger / name)
            for name in (
                "source_pre.jsonl",
                "staging.jsonl",
                "source_post.jsonl",
                "destination_post.jsonl",
                "source_final.jsonl",
            )
        }
        result = {
            "status": "complete",
            "phase": phase,
            "batch": batch_name,
            "transaction_id": transaction_id,
            "started_at": started_at,
            "finished_at": datetime.now().astimezone().isoformat(),
            "source": str(source),
            "destination": str(destination),
            "source_removed": remove_source,
            "mount": mount,
            "tree": snapshot_summary(source_pre),
            "manifest_sha256": manifests,
        }
        write_json_atomic(status_path, result)
        return result
    except Exception as error:
        failure = {
            "status": "failed",
            "phase": phase,
            "batch": batch_name,
            "transaction_id": transaction_id,
            "started_at": started_at,
            "finished_at": datetime.now().astimezone().isoformat(),
            "source": str(source) if source is not None else "",
            "destination": str(destination) if destination is not None else "",
            "destination_published": is_directory_noexcept(destination),
            "staging": str(staging),
            "retired_source": str(retired) if retired is not None else "",
            "error": str(error),
        }
        try:
            current_mount = validate_archive_mount(policy)
            ledger_is_on_archive = (
                ledger.is_dir()
                and device_id(ledger) == device_id(policy.archive_mount)
            )
            if current_mount == initial_mount and ledger_is_on_archive:
                write_json_atomic(status_path, failure)
            else:
                raise RuntimeError("archive identity is unavailable for failure ledger")
        except Exception as ledger_error:
            print(
                f"OAI_PROFILE_OFFLOAD_LEDGER=SKIPPED: {ledger_error}",
                file=sys.stderr,
            )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify and atomically offload one completed OAI profiler batch"
    )
    parser.add_argument("policy", type=Path, help="Storage-policy JSON file")
    parser.add_argument("batch", help="One child directory below policy active_root")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Copy, verify and atomically publish; default is a read-only dry run",
    )
    parser.add_argument(
        "--remove-source",
        action="store_true",
        help="After publication proof, remove the active source batch",
    )
    args = parser.parse_args()
    if args.remove_source and not args.execute:
        parser.error("--remove-source requires --execute")
    policy = load_policy(args.policy)
    result = (
        offload_batch(policy, args.batch, args.remove_source)
        if args.execute
        else plan_offload(policy, args.batch)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"OAI_PROFILE_OFFLOAD=FAIL: {error}", file=sys.stderr)
        raise SystemExit(2)
