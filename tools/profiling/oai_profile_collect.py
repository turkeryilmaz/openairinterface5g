#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Collect timestamped nrUE profiler runs from a remote host with scp."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import stat
import subprocess
import tempfile
from pathlib import Path

from oai_profile_archive import MANIFEST_NAME, verify_archive


DEFAULT_REMOTE_ROOT = "/mnt/ssd/Documents/OpenAirInterface/PerformanceProfiles"
RUN_NAME = re.compile(
    r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_nrUE_[A-Za-z0-9_.-]+(?:_\d+)?$"
)


def default_local_root() -> Path:
    repository = Path(__file__).resolve().parents[2]
    return repository.parent / "PerformanceProfiles"


def list_remote_runs(remote: str, remote_root: str) -> list[str]:
    remote_command = (
        f"find {shlex.quote(remote_root)} -mindepth 1 -maxdepth 1 "
        "-type d -printf '%f\\n'"
    )
    result = subprocess.run(
        ["ssh", remote, remote_command],
        check=True,
        text=True,
        capture_output=True,
    )
    names = []
    for line in result.stdout.splitlines():
        name = line.strip()
        if RUN_NAME.fullmatch(name):
            names.append(name)
    return sorted(set(names))


def normalize_manifest_mtimes(run_dir: Path) -> int | None:
    manifest = run_dir / MANIFEST_NAME
    try:
        manifest_mode = manifest.lstat().st_mode
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(manifest_mode):
        raise RuntimeError("transferred archive manifest is not a regular file")
    results = verify_archive(run_dir)
    failures = [result for result in results if not result.valid]
    invalid = []
    for result in failures:
        expected_second = result.expected_mtime_ns // 1_000_000_000 * 1_000_000_000
        if result.status != "mtime_mismatch":
            invalid.append(f"{result.relative_path}:{result.status}")
        elif result.observed_mtime_ns != expected_second:
            invalid.append(f"{result.relative_path}:mtime_mismatch_not_second_truncation")
    if invalid:
        statuses = ", ".join(invalid)
        raise RuntimeError(f"transferred archive failed verification: {statuses}")
    for result in failures:
        path = run_dir / result.relative_path
        observed = path.stat()
        os.utime(
            path,
            ns=(observed.st_atime_ns, result.expected_mtime_ns),
            follow_symlinks=False,
        )
    final_failures = [result for result in verify_archive(run_dir) if not result.valid]
    if final_failures:
        statuses = ", ".join(
            f"{result.relative_path}:{result.status}" for result in final_failures
        )
        raise RuntimeError(f"transferred archive failed verification after mtime restore: {statuses}")
    return len(failures)


def copy_run(remote: str, remote_root: str, local_root: Path, name: str, dry_run: bool) -> bool:
    destination = local_root / name
    if destination.exists():
        print(f"skip existing: {destination}")
        return False

    source = f"{remote}:{remote_root.rstrip('/')}/{name}"
    if dry_run:
        print(f"would copy: {source} -> {destination}")
        return True

    with tempfile.TemporaryDirectory(prefix=".incoming-", dir=local_root) as temporary:
        staging = Path(temporary)
        subprocess.run(["scp", "-p", "-r", source, str(staging)], check=True)
        copied = staging / name
        if not copied.is_dir():
            raise RuntimeError(f"scp did not create expected directory: {copied}")
        restored_mtimes = normalize_manifest_mtimes(copied)
        if restored_mtimes is None:
            print(f"warning: unverified manifestless partial run: {name}")
        else:
            print(f"verified manifest: {name} (restored_mtimes={restored_mtimes})")
        copied.rename(destination)
    print(f"collected: {destination}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect missing nrUE OAI profiler runs from a remote CM5")
    parser.add_argument("remote", help="SSH target, for example turker@cm5")
    parser.add_argument(
        "--remote-root",
        default=DEFAULT_REMOTE_ROOT,
        help=f"Remote archive root (default: {DEFAULT_REMOTE_ROOT})",
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        default=default_local_root(),
        help="Laptop archive root (default: sibling PerformanceProfiles directory)",
    )
    parser.add_argument("--dry-run", action="store_true", help="List copies without transferring data")
    args = parser.parse_args()

    local_root = args.local_root.expanduser().resolve()
    local_root.mkdir(parents=True, exist_ok=True)
    remote_runs = list_remote_runs(args.remote, args.remote_root)
    if not remote_runs:
        print("no timestamped nrUE profile directories found")
        return 0

    copied = 0
    for name in remote_runs:
        copied += copy_run(args.remote, args.remote_root, local_root, name, args.dry_run)

    print(f"{copied} run(s) {'would be copied' if args.dry_run else 'copied'}")
    print(
        "analyze with: "
        f"{Path(__file__).with_name('oai_profile_analyze.py')} "
        f"{local_root} --output-dir {local_root / 'Analysis'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
