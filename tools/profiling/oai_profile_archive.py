#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Finalize and verify OAI profile archives without external dependencies.

The SHA-256 manifest detects accidental archive modification. It is not an
authenticity proof: sign or otherwise protect the manifest externally when a
chain of custody is required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


MANIFEST_VERSION = "1"
MANIFEST_NAME = "archive_manifest.csv"
HASH_CHUNK_BYTES = 1024 * 1024
SOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_FIELDS = [
    "manifest_version",
    "schema_version",
    "run_id",
    "experiment_id",
    "campaign_id",
    "variant",
    "trial",
    "role",
    "hostname",
    "archive_state",
    "relative_path",
    "artifact_type",
    "size_bytes",
    "mtime_ns",
    "sha256",
]
EXTERNAL_SOURCE_FIELDS = [
    "schema_version",
    "run_id",
    "experiment_id",
    "campaign_id",
    "variant",
    "trial",
    "role",
    "hostname",
    "source_id",
    "source_type",
    "clock_domain",
    "clock_unit",
    "artifact_path",
    "command",
    "tool_version",
    "start_realtime_ns",
    "end_realtime_ns",
    "start_monotonic_raw_ns",
    "end_monotonic_raw_ns",
    "status",
    "alignment_method",
    "alignment_uncertainty_ns",
    "notes",
]


@dataclass(frozen=True)
class VerificationResult:
    relative_path: str
    status: str
    expected_size_bytes: int = 0
    observed_size_bytes: int = 0
    expected_mtime_ns: int = 0
    observed_mtime_ns: int = 0
    expected_sha256: str = ""
    observed_sha256: str = ""
    notes: str = ""

    @property
    def valid(self) -> bool:
        return self.status == "ok"


def read_metadata(run_dir: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    path = run_dir / "metadata.txt"
    if not path.is_file():
        return metadata
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key] = value
    return metadata


def archive_identity(run_dir: Path) -> dict[str, str]:
    metadata = read_metadata(run_dir)
    campaign: dict[str, object] = {}
    campaign_path = run_dir / "campaign_run.json"
    if campaign_path.is_file():
        loaded = json.loads(campaign_path.read_text())
        if not isinstance(loaded, dict):
            raise ValueError(f"campaign metadata must be an object: {campaign_path}")
        campaign = loaded
    clean_shutdown = metadata.get("clean_shutdown", "0")
    if metadata:
        archive_state = "clean_shutdown" if clean_shutdown == "1" else "unclean_or_unknown"
    else:
        archive_state = "runner_completed" if campaign.get("status") == "finished" else "runner_incomplete"
    return {
        "schema_version": metadata.get("schema_version", str(campaign.get("schema_version", "unknown"))),
        "run_id": metadata.get("run_id", str(campaign.get("run_id", run_dir.name))),
        "experiment_id": metadata.get("experiment_id", str(campaign.get("experiment_id", ""))),
        "campaign_id": metadata.get("campaign_id", str(campaign.get("campaign_id", ""))),
        "variant": metadata.get("variant", str(campaign.get("variant", ""))),
        "trial": metadata.get("trial", str(campaign.get("trial", ""))),
        "role": metadata.get("role", str(campaign.get("role", "unknown"))),
        "hostname": metadata.get("hostname", str(campaign.get("hostname", "unknown"))),
        "archive_state": archive_state,
    }


def classify_artifact(relative_path: str) -> str:
    name = Path(relative_path).name
    exact = {
        "events.csv": "event_stream",
        "event_catalog.csv": "event_catalog",
        "metadata.txt": "run_metadata",
        "settings.csv": "run_settings",
        "sync.csv": "clock_anchor",
        "clock_catalog.csv": "clock_catalog",
        "drops.csv": "integrity_diagnostics",
        "host_metrics.csv": "host_metrics",
        "profiler_primitive_overhead.csv": "profiler_calibration",
        "external_sources.csv": "external_source_catalog",
        "stdout.log": "process_stdout",
        "stderr.log": "process_stderr",
    }
    if name in exact:
        return exact[name]
    if name.startswith("pmu_"):
        return "pmu"
    if name in {"thread_metrics.csv", "kernel_activity.csv", "interrupts.csv", "softirqs.csv", "system_catalog.csv"}:
        return "scheduler_or_kernel"
    if name == "system_read_overhead.csv":
        return "system_collection_overhead"
    if name.endswith("perf.data") or name == "perf.data":
        return "perf_record"
    if name.startswith("perf_stat"):
        return "perf_stat"
    if relative_path.startswith("external/"):
        return "external_artifact"
    if relative_path.startswith("sidecars/"):
        return "sidecar_artifact"
    return "other"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def same_file_snapshot(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
        and before.st_ctime_ns == after.st_ctime_ns
    )


def iter_archive_files(run_dir: Path) -> Iterable[Path]:
    for path in sorted(run_dir.rglob("*")):
        if path.name == MANIFEST_NAME or path.name.startswith(f".{MANIFEST_NAME}."):
            continue
        if path.is_symlink():
            raise ValueError(f"archive symlinks are not supported: {path}")
        if path.is_file():
            yield path


def build_manifest_rows(run_dir: Path) -> list[dict[str, object]]:
    identity = archive_identity(run_dir)
    rows: list[dict[str, object]] = []
    for path in iter_archive_files(run_dir):
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        if not same_file_snapshot(before, after):
            raise RuntimeError(f"file changed while hashing: {path}")
        relative_path = path.relative_to(run_dir).as_posix()
        rows.append(
            {
                "manifest_version": MANIFEST_VERSION,
                **identity,
                "relative_path": relative_path,
                "artifact_type": classify_artifact(relative_path),
                "size_bytes": after.st_size,
                "mtime_ns": after.st_mtime_ns,
                "sha256": digest,
            }
        )
    return rows


def atomic_write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def finalize_archive(run_dir: Path, replace: bool = False) -> Path:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    manifest_path = run_dir / MANIFEST_NAME
    if manifest_path.exists() and not replace:
        raise FileExistsError(f"manifest already exists; verify it or use --replace: {manifest_path}")
    rows = build_manifest_rows(run_dir)
    if not rows:
        raise ValueError(f"archive contains no regular files: {run_dir}")
    atomic_write_csv(manifest_path, MANIFEST_FIELDS, rows)
    return manifest_path


def read_manifest(run_dir: Path) -> list[dict[str, str]]:
    manifest_path = run_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = [field for field in MANIFEST_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"manifest is missing columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"manifest has no data rows: {manifest_path}")
    return rows


def is_safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def has_symlink_component(run_dir: Path, relative_path: str) -> bool:
    candidate = run_dir
    for component in PurePosixPath(relative_path).parts:
        candidate /= component
        if candidate.is_symlink():
            return True
    return False


def verify_archive(run_dir: Path) -> list[VerificationResult]:
    run_dir = run_dir.expanduser().resolve()
    rows = read_manifest(run_dir)
    expected_identity = archive_identity(run_dir)
    results: list[VerificationResult] = []
    manifested: set[str] = set()
    for row in rows:
        relative_path = row.get("relative_path", "")
        if relative_path in manifested:
            results.append(VerificationResult(relative_path, "duplicate_manifest_path"))
            continue
        manifested.add(relative_path)
        if row.get("manifest_version") != MANIFEST_VERSION:
            results.append(VerificationResult(relative_path, "invalid_manifest_version"))
            continue
        if any(row.get(field, "") != value for field, value in expected_identity.items()):
            results.append(
                VerificationResult(
                    relative_path,
                    "manifest_identity_mismatch",
                    notes="manifest identity does not match archived metadata",
                )
            )
            continue
        if not is_safe_relative_path(relative_path):
            results.append(VerificationResult(relative_path, "invalid_relative_path"))
            continue
        if row.get("artifact_type") != classify_artifact(relative_path):
            results.append(VerificationResult(relative_path, "artifact_type_mismatch"))
            continue
        expected_hash = row.get("sha256", "")
        if SHA256_PATTERN.fullmatch(expected_hash) is None:
            results.append(VerificationResult(relative_path, "invalid_sha256"))
            continue
        candidate = run_dir / relative_path
        if has_symlink_component(run_dir, relative_path):
            results.append(VerificationResult(relative_path, "unexpected_symlink"))
            continue
        try:
            resolved_candidate = candidate.resolve(strict=True)
        except (FileNotFoundError, NotADirectoryError):
            results.append(VerificationResult(relative_path, "missing"))
            continue
        try:
            resolved_candidate.relative_to(run_dir)
        except ValueError:
            results.append(VerificationResult(relative_path, "path_escape"))
            continue
        if not resolved_candidate.is_file():
            results.append(VerificationResult(relative_path, "missing"))
            continue
        try:
            expected_size = int(row["size_bytes"])
            expected_mtime = int(row["mtime_ns"])
            if expected_size < 0 or expected_mtime < 0:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            results.append(VerificationResult(relative_path, "invalid_manifest_value"))
            continue
        observed_before = candidate.stat()
        observed_hash = sha256_file(candidate)
        observed = candidate.stat()
        if not same_file_snapshot(observed_before, observed):
            status = "changed_during_verification"
        elif observed.st_size != expected_size:
            status = "size_mismatch"
        elif observed_hash != expected_hash:
            status = "hash_mismatch"
        elif observed.st_mtime_ns != expected_mtime:
            status = "mtime_mismatch"
        else:
            status = "ok"
        results.append(
            VerificationResult(
                relative_path=relative_path,
                status=status,
                expected_size_bytes=expected_size,
                observed_size_bytes=observed.st_size,
                expected_mtime_ns=expected_mtime,
                observed_mtime_ns=observed.st_mtime_ns,
                expected_sha256=expected_hash,
                observed_sha256=observed_hash,
            )
        )

    actual: set[str] = set()
    unexpected_symlinks: set[str] = set()
    for path in sorted(run_dir.rglob("*")):
        if path.name == MANIFEST_NAME or path.name.startswith(f".{MANIFEST_NAME}."):
            continue
        relative_path = path.relative_to(run_dir).as_posix()
        if has_symlink_component(run_dir, relative_path):
            unexpected_symlinks.add(relative_path)
        elif path.is_file():
            actual.add(relative_path)
    for relative_path in sorted(actual - manifested):
        results.append(VerificationResult(relative_path, "unmanifested"))
    for relative_path in sorted(unexpected_symlinks - manifested):
        results.append(VerificationResult(relative_path, "unexpected_symlink"))
    return sorted(results, key=lambda result: (result.relative_path, result.status))


def write_verification_report(path: Path, results: Iterable[VerificationResult]) -> None:
    fields = [
        "relative_path",
        "status",
        "expected_size_bytes",
        "observed_size_bytes",
        "expected_mtime_ns",
        "observed_mtime_ns",
        "expected_sha256",
        "observed_sha256",
        "notes",
    ]
    rows = [result.__dict__ for result in results]
    atomic_write_csv(path, fields, rows)


def ensure_relative_artifact(run_dir: Path, artifact: Path, copy_artifact: bool, source_id: str) -> Path:
    run_dir = run_dir.resolve()
    artifact = artifact.expanduser().resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    try:
        artifact.relative_to(run_dir)
        return artifact
    except ValueError:
        if not copy_artifact:
            raise ValueError("artifact is outside the run directory; use --copy-artifact") from None
    destination_dir = run_dir / "external" / source_id
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / artifact.name
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copy2(artifact, destination)
    return destination


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != EXTERNAL_SOURCE_FIELDS:
            raise ValueError(f"unexpected external source schema in {path}")
        return list(reader)


def register_external_source(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve()
    if not SOURCE_ID_PATTERN.fullmatch(args.source_id) or args.source_id in {".", ".."}:
        raise ValueError(f"invalid source_id: {args.source_id}")
    manifest_path = run_dir / MANIFEST_NAME
    if manifest_path.exists() and not args.replace_manifest:
        raise FileExistsError("register sources before finalization, or pass --replace-manifest")
    artifact = ensure_relative_artifact(run_dir, args.artifact, args.copy_artifact, args.source_id)
    catalog_path = run_dir / "external_sources.csv"
    rows = read_csv_rows(catalog_path)
    if any(row["source_id"] == args.source_id for row in rows):
        raise ValueError(f"duplicate source_id: {args.source_id}")
    identity = archive_identity(run_dir)
    rows.append(
        {
            "schema_version": identity["schema_version"],
            "run_id": identity["run_id"],
            "experiment_id": identity["experiment_id"],
            "campaign_id": identity["campaign_id"],
            "variant": identity["variant"],
            "trial": identity["trial"],
            "role": identity["role"],
            "hostname": identity["hostname"],
            "source_id": args.source_id,
            "source_type": args.source_type,
            "clock_domain": args.clock_domain,
            "clock_unit": args.clock_unit,
            "artifact_path": artifact.relative_to(run_dir).as_posix(),
            "command": args.command,
            "tool_version": args.tool_version,
            "start_realtime_ns": args.start_realtime_ns,
            "end_realtime_ns": args.end_realtime_ns,
            "start_monotonic_raw_ns": args.start_monotonic_raw_ns,
            "end_monotonic_raw_ns": args.end_monotonic_raw_ns,
            "status": args.status,
            "alignment_method": args.alignment_method,
            "alignment_uncertainty_ns": args.alignment_uncertainty_ns,
            "notes": args.notes,
        }
    )
    atomic_write_csv(catalog_path, EXTERNAL_SOURCE_FIELDS, rows)
    if manifest_path.exists():
        finalize_archive(run_dir, replace=True)


def print_verification(results: list[VerificationResult]) -> None:
    failures = [result for result in results if not result.valid]
    for result in failures:
        print(f"{result.status}: {result.relative_path}", file=sys.stderr)
    print(f"verified={len(results) - len(failures)} failed={len(failures)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    finalize = subparsers.add_parser("finalize", help="Hash all regular files and atomically write a manifest")
    finalize.add_argument("run_dir", type=Path)
    finalize.add_argument("--replace", action="store_true", help="Replace an existing manifest")

    verify = subparsers.add_parser("verify", help="Verify hashes, sizes, mtimes, missing files, and extra files")
    verify.add_argument("run_dir", type=Path)
    verify.add_argument("--report", type=Path, help="Write detailed verification results as CSV")

    external = subparsers.add_parser("register-external", help="Register a sidecar or future power artifact")
    external.add_argument("run_dir", type=Path)
    external.add_argument("--source-id", required=True)
    external.add_argument("--source-type", required=True)
    external.add_argument("--artifact", type=Path, required=True)
    external.add_argument("--copy-artifact", action="store_true")
    external.add_argument("--clock-domain", default="unknown")
    external.add_argument("--clock-unit", default="unknown")
    external.add_argument("--command", default="")
    external.add_argument("--tool-version", default="")
    external.add_argument("--start-realtime-ns", default="")
    external.add_argument("--end-realtime-ns", default="")
    external.add_argument("--start-monotonic-raw-ns", default="")
    external.add_argument("--end-monotonic-raw-ns", default="")
    external.add_argument("--status", default="recorded")
    external.add_argument("--alignment-method", default="unresolved")
    external.add_argument("--alignment-uncertainty-ns", default="")
    external.add_argument("--notes", default="")
    external.add_argument("--replace-manifest", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.action == "finalize":
        print(finalize_archive(args.run_dir, args.replace))
        return 0
    if args.action == "verify":
        results = verify_archive(args.run_dir)
        if args.report:
            write_verification_report(args.report, results)
        print_verification(results)
        return 0 if all(result.valid for result in results) else 1
    if args.action == "register-external":
        register_external_source(args)
        return 0
    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
