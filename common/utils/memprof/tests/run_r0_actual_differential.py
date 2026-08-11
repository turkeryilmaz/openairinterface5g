#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Run bounded A00/A01 actual-libc fixtures and compare canonical output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from r0_harness_common import (
    BoundedProcessResult,
    format_byte_evidence,
    loader_environment_variables,
    run_process_bounded,
)


EXPECTED_OUTPUT = (
    b"R0_ACTUAL_V1 semantic=pass pre_main=pass main=pass dso_constructor=pass "
    b"dso_destructor=pass process_destructor=pass\n"
)
EXPECTED_DSO_IDENTITY_STDERR = b"R0_ACTUAL_V1 error=dso_link_identity\n"
EXPECTED_DSO_IDENTITY_RETURN_CODE = 80


class LaunchError(RuntimeError):
    pass


def positive_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 60.0:
        raise argparse.ArgumentTypeError("timeout must be in (0, 60] seconds")
    return parsed


def bounded_size(value: str) -> int:
    parsed = int(value)
    if not 128 <= parsed <= 1024 * 1024:
        raise argparse.ArgumentTypeError("output bound must be between 128 and 1048576 bytes")
    return parsed


def run_bounded(command: list[str], timeout_seconds: float, max_output_bytes: int) -> BoundedProcessResult:
    return run_process_bounded(
        command,
        timeout_seconds=timeout_seconds,
        max_stdout_bytes=max_output_bytes,
        max_stderr_bytes=max_output_bytes,
        remove_environment=loader_environment_variables(),
        environment_updates={"LANG": "C", "LC_ALL": "C"},
    )


def require_case(label: str, result: BoundedProcessResult) -> None:
    evidence = f"{format_byte_evidence('stdout', result.stdout)}; {format_byte_evidence('stderr', result.stderr)}"
    if result.failure is not None:
        raise LaunchError(f"{label} bounded launch failed: {result.failure!a}; {evidence}")
    if result.returncode != 0:
        raise LaunchError(f"{label} returned {result.returncode}; {evidence}")
    if result.stderr:
        raise LaunchError(f"{label} produced stderr; {evidence}")
    if result.stdout != EXPECTED_OUTPUT:
        raise LaunchError(f"{label} stdout is not the frozen teardown transcript; {evidence}")


def require_dso_identity_negative(label: str, result: BoundedProcessResult) -> None:
    evidence = f"{format_byte_evidence('stdout', result.stdout)}; {format_byte_evidence('stderr', result.stderr)}"
    if result.failure is not None:
        raise LaunchError(f"{label} bounded launch failed: {result.failure!a}; {evidence}")
    if result.returncode != EXPECTED_DSO_IDENTITY_RETURN_CODE:
        raise LaunchError(
            f"{label} returned {result.returncode}, expected {EXPECTED_DSO_IDENTITY_RETURN_CODE}; {evidence}"
        )
    if result.stdout:
        raise LaunchError(f"{label} produced stdout; {evidence}")
    if result.stderr != EXPECTED_DSO_IDENTITY_STDERR:
        raise LaunchError(f"{label} stderr is not the frozen DSO identity rejection; {evidence}")


def existing_file(value: str) -> str:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {value}")
    return str(path.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a00-exe", required=True, type=existing_file)
    parser.add_argument("--a00-dso", required=True, type=existing_file)
    parser.add_argument("--a01-exe", required=True, type=existing_file)
    parser.add_argument("--a01-dso", required=True, type=existing_file)
    parser.add_argument("--runtime-path", required=True, type=existing_file)
    parser.add_argument("--timeout-seconds", type=positive_float, default=10.0)
    parser.add_argument("--max-output-bytes", type=bounded_size, default=4096)
    args = parser.parse_args()

    try:
        a00 = run_bounded(
            [args.a00_exe, "--dso", args.a00_dso, "--runtime", "absent"],
            args.timeout_seconds,
            args.max_output_bytes,
        )
        require_case("A00", a00)
        a01 = run_bounded(
            [
                args.a01_exe,
                "--dso",
                args.a01_dso,
                "--runtime",
                "present-off",
                "--runtime-path",
                args.runtime_path,
            ],
            args.timeout_seconds,
            args.max_output_bytes,
        )
        require_case("A01", a01)
        if a00.stdout != a01.stdout:
            raise LaunchError("A00 and A01 normalized transcripts differ")
        a01_with_a00_dso = run_bounded(
            [
                args.a01_exe,
                "--dso",
                args.a00_dso,
                "--runtime",
                "present-off",
                "--runtime-path",
                args.runtime_path,
            ],
            args.timeout_seconds,
            args.max_output_bytes,
        )
        require_dso_identity_negative("A01_EXE_A00_DSO", a01_with_a00_dso)
        a00_with_a01_dso = run_bounded(
            [args.a00_exe, "--dso", args.a01_dso, "--runtime", "absent"],
            args.timeout_seconds,
            args.max_output_bytes,
        )
        require_dso_identity_negative("A00_EXE_A01_DSO", a00_with_a01_dso)
    except LaunchError as error:
        print(f"R0_ACTUAL_DIFFERENTIAL_V1 fail: {error}", file=sys.stderr)
        return 1

    print(
        "R0_ACTUAL_DIFFERENTIAL_V1 pass cases=4 positives=2 "
        "identity-negatives=2 transcript=byte-identical"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
