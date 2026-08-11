#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0

"""Bounded child-process fixture for the R0 terminal-controller tests.

The production runner deliberately accepts only explicit argument vectors.  A
small external program is therefore a more faithful oracle than mocks for
process-group, inherited-pipe, signal, and output-limit behavior.  Every mode
has a finite upper bound so a failing controller test cannot intentionally
leave an immortal fixture behind.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Sequence


MAX_BYTES = 8 * 1024 * 1024
MAX_SLEEP_SECONDS = 20.0


def _bounded_nonnegative_int(text: str) -> int:
    value = int(text, 10)
    if value < 0 or value > MAX_BYTES:
        raise argparse.ArgumentTypeError(
            "value must be in [0, {}]".format(MAX_BYTES)
        )
    return value


def _bounded_seconds(text: str) -> float:
    value = float(text)
    if value < 0.0 or value > MAX_SLEEP_SECONDS:
        raise argparse.ArgumentTypeError(
            "seconds must be in [0, {:.1f}]".format(MAX_SLEEP_SECONDS)
        )
    return value


def _bounded_count(text: str) -> int:
    value = int(text, 10)
    if value < 1 or value > 8:
        raise argparse.ArgumentTypeError("count must be in [1, 8]")
    return value


def _write_all(fd: int, count: int, byte: bytes) -> None:
    chunk = byte * min(65536, count)
    remaining = count
    while remaining:
        current = chunk[: min(len(chunk), remaining)]
        try:
            written = os.write(fd, current)
        except BrokenPipeError:
            return
        if written <= 0:
            raise RuntimeError("os.write made no progress")
        remaining -= written


def _write_pid_exclusive(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        data = "{}\n".format(os.getpid()).encode("ascii")
        if os.write(descriptor, data) != len(data):
            raise RuntimeError("short PID-file write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _hold(
    seconds: float,
    pid_file: Path,
    *,
    close_pipes: bool,
    alternate_process_group: bool,
    ignore_term: bool,
    stdout_bytes: int,
) -> int:
    if alternate_process_group:
        os.setpgid(0, 0)
    if close_pipes:
        devnull = os.open(os.devnull, os.O_RDWR | os.O_CLOEXEC)
        try:
            os.dup2(devnull, sys.stdout.fileno())
            os.dup2(devnull, sys.stderr.fileno())
        finally:
            if devnull > sys.stderr.fileno():
                os.close(devnull)
    if ignore_term:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    _write_all(sys.stdout.fileno(), stdout_bytes, b"O")
    _write_pid_exclusive(pid_file)
    time.sleep(seconds)
    return 0


def _spawn_descendant(
    seconds: float,
    pid_file: Path,
    *,
    close_pipes: bool,
    alternate_process_group: bool,
    ignore_term: bool,
    new_session: bool,
    stdout_bytes: int,
) -> int:
    arguments = [
        sys.executable,
        str(Path(__file__).resolve()),
        "hold",
        str(seconds),
        str(pid_file),
    ]
    if close_pipes:
        arguments.append("--close-pipes")
    if alternate_process_group:
        arguments.append("--alternate-process-group")
    if ignore_term:
        arguments.append("--ignore-term")
    if stdout_bytes:
        arguments.extend(("--stdout-bytes", str(stdout_bytes)))
    subprocess.Popen(
        arguments,
        close_fds=False,
        start_new_session=new_session,
    )

    # Wait for the PID ledger before the leader exits.  This makes a missing
    # descendant a controller failure, not a fixture launch race.
    deadline = time.monotonic() + 2.0
    while not pid_file.exists():
        if time.monotonic() >= deadline:
            return 70
        time.sleep(0.005)
    return 0


def _run_leader_with_descendants(
    seconds: float,
    pid_directory: Path,
    count: int,
    *,
    close_pipes: bool,
    new_session: bool,
    stdout_bytes: int,
) -> int:
    pid_files = [pid_directory / "child-{}.pid".format(index) for index in range(count)]
    for pid_file in pid_files:
        arguments = [
            sys.executable,
            str(Path(__file__).resolve()),
            "hold",
            str(seconds),
            str(pid_file),
        ]
        if close_pipes:
            arguments.append("--close-pipes")
        if stdout_bytes:
            arguments.extend(("--stdout-bytes", str(stdout_bytes)))
        subprocess.Popen(
            arguments,
            close_fds=False,
            start_new_session=new_session,
        )
    deadline = time.monotonic() + 2.0
    while not all(path.exists() for path in pid_files):
        if time.monotonic() >= deadline:
            return 70
        time.sleep(0.005)
    time.sleep(seconds)
    return 0


def _double_fork_descendant(
    seconds: float,
    pid_file: Path,
    *,
    close_pipes: bool,
) -> int:
    intermediate = os.fork()
    if intermediate == 0:
        try:
            os.setsid()
            grandchild = os.fork()
            if grandchild == 0:
                try:
                    status = _hold(
                        seconds,
                        pid_file,
                        close_pipes=close_pipes,
                        alternate_process_group=False,
                        ignore_term=True,
                        stdout_bytes=0,
                    )
                except BaseException:
                    status = 72
                os._exit(status)
            os._exit(0)
        except BaseException:
            os._exit(71)
    waited, status = os.waitpid(intermediate, 0)
    if waited != intermediate or status != 0:
        return 71
    deadline = time.monotonic() + 2.0
    while not pid_file.exists():
        if time.monotonic() >= deadline:
            return 70
        time.sleep(0.005)
    return 0


def _rapid_double_fork(
    seconds: float,
    pid_directory: Path,
    count: int,
) -> int:
    pid_files = [
        pid_directory / "grandchild-{}.pid".format(index)
        for index in range(count)
    ]
    for pid_file in pid_files:
        status = _double_fork_descendant(seconds, pid_file, close_pipes=True)
        if status != 0:
            return status
    return 0


def _sequential_descendants(
    seconds: float,
    pid_directory: Path,
    count: int,
) -> int:
    for index in range(count):
        pid_file = pid_directory / "sequential-{}.pid".format(index)
        child = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "hold",
                str(seconds),
                str(pid_file),
                "--close-pipes",
            ],
            close_fds=False,
            start_new_session=True,
        )
        if child.wait() != 0:
            return 73
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    exit_parser = subparsers.add_parser("exit")
    exit_parser.add_argument("status", type=int, choices=range(0, 126))

    write_parser = subparsers.add_parser("write")
    write_parser.add_argument("stream", choices=("stdout", "stderr", "both"))
    write_parser.add_argument("count", type=_bounded_nonnegative_int)

    write_exit_parser = subparsers.add_parser("write-exit")
    write_exit_parser.add_argument("stream", choices=("stdout", "stderr", "both"))
    write_exit_parser.add_argument("count", type=_bounded_nonnegative_int)
    write_exit_parser.add_argument("status", type=int, choices=range(1, 126))

    environment_parser = subparsers.add_parser("environment")
    environment_parser.add_argument("name")

    sleep_parser = subparsers.add_parser("sleep")
    sleep_parser.add_argument("seconds", type=_bounded_seconds)
    sleep_parser.add_argument("--ignore-term", action="store_true")

    hold_parser = subparsers.add_parser("hold")
    hold_parser.add_argument("seconds", type=_bounded_seconds)
    hold_parser.add_argument("pid_file", type=Path)
    hold_parser.add_argument("--close-pipes", action="store_true")
    hold_parser.add_argument("--alternate-process-group", action="store_true")
    hold_parser.add_argument("--ignore-term", action="store_true")
    hold_parser.add_argument("--stdout-bytes", type=_bounded_nonnegative_int, default=0)

    descendant_parser = subparsers.add_parser("leader-exit")
    descendant_parser.add_argument("seconds", type=_bounded_seconds)
    descendant_parser.add_argument("pid_file", type=Path)
    descendant_parser.add_argument("--close-pipes", action="store_true")
    descendant_parser.add_argument("--alternate-process-group", action="store_true")
    descendant_parser.add_argument("--ignore-term", action="store_true")
    descendant_parser.add_argument("--new-session", action="store_true")
    descendant_parser.add_argument("--stdout-bytes", type=_bounded_nonnegative_int, default=0)

    live_leader_parser = subparsers.add_parser("leader-running")
    live_leader_parser.add_argument("seconds", type=_bounded_seconds)
    live_leader_parser.add_argument("pid_directory", type=Path)
    live_leader_parser.add_argument("count", type=_bounded_count)
    live_leader_parser.add_argument("--close-pipes", action="store_true")
    live_leader_parser.add_argument("--new-session", action="store_true")
    live_leader_parser.add_argument("--stdout-bytes", type=_bounded_nonnegative_int, default=0)

    double_fork_parser = subparsers.add_parser("double-fork")
    double_fork_parser.add_argument("seconds", type=_bounded_seconds)
    double_fork_parser.add_argument("pid_file", type=Path)
    double_fork_parser.add_argument("--close-pipes", action="store_true")

    rapid_parser = subparsers.add_parser("rapid-double-fork")
    rapid_parser.add_argument("seconds", type=_bounded_seconds)
    rapid_parser.add_argument("pid_directory", type=Path)
    rapid_parser.add_argument("count", type=_bounded_count)

    sequential_parser = subparsers.add_parser("sequential-descendants")
    sequential_parser.add_argument("seconds", type=_bounded_seconds)
    sequential_parser.add_argument("pid_directory", type=Path)
    sequential_parser.add_argument("count", type=_bounded_count)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.mode == "exit":
        return arguments.status
    if arguments.mode in ("write", "write-exit"):
        if arguments.stream in ("stdout", "both"):
            _write_all(sys.stdout.fileno(), arguments.count, b"O")
        if arguments.stream in ("stderr", "both"):
            _write_all(sys.stderr.fileno(), arguments.count, b"E")
        return arguments.status if arguments.mode == "write-exit" else 0
    if arguments.mode == "environment":
        value = os.environ.get(arguments.name, "<MISSING>").encode("utf-8")
        _write_all(sys.stdout.fileno(), len(value), value)
        return 0
    if arguments.mode == "sleep":
        if arguments.ignore_term:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(arguments.seconds)
        return 0
    if arguments.mode == "hold":
        return _hold(
            arguments.seconds,
            arguments.pid_file,
            close_pipes=arguments.close_pipes,
            alternate_process_group=arguments.alternate_process_group,
            ignore_term=arguments.ignore_term,
            stdout_bytes=arguments.stdout_bytes,
        )
    if arguments.mode == "leader-exit":
        return _spawn_descendant(
            arguments.seconds,
            arguments.pid_file,
            close_pipes=arguments.close_pipes,
            alternate_process_group=arguments.alternate_process_group,
            ignore_term=arguments.ignore_term,
            new_session=arguments.new_session,
            stdout_bytes=arguments.stdout_bytes,
        )
    if arguments.mode == "leader-running":
        return _run_leader_with_descendants(
            arguments.seconds,
            arguments.pid_directory,
            arguments.count,
            close_pipes=arguments.close_pipes,
            new_session=arguments.new_session,
            stdout_bytes=arguments.stdout_bytes,
        )
    if arguments.mode == "double-fork":
        return _double_fork_descendant(
            arguments.seconds,
            arguments.pid_file,
            close_pipes=arguments.close_pipes,
        )
    if arguments.mode == "rapid-double-fork":
        return _rapid_double_fork(
            arguments.seconds,
            arguments.pid_directory,
            arguments.count,
        )
    if arguments.mode == "sequential-descendants":
        return _sequential_descendants(
            arguments.seconds,
            arguments.pid_directory,
            arguments.count,
        )
    raise AssertionError("unreachable mode")


if __name__ == "__main__":
    raise SystemExit(main())
