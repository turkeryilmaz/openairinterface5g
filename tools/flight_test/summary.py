#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Bounded offline facts from one capture attempt, without inferring delivery."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any, Iterator


class BoundedLogs:
    def __init__(self, maximum_bytes: int = 64 * 1024 * 1024) -> None:
        self.remaining = maximum_bytes
        self.read_bytes = 0
        self.invalid_lines = 0
        self.limited = False
        self.unavailable_files = 0
        self.budget_partial_lines = 0

    def records(self, paths: list[Path]) -> Iterator[dict[str, Any]]:
        # Rotating text writers can split a JSON line across adjacent files.
        buffer = bytearray()
        discarding = False
        for path in paths:
            try:
                with path.open("rb") as source:
                    while self.remaining > 0:
                        data = source.read(min(16384, self.remaining))
                        if not data:
                            break
                        self.remaining -= len(data)
                        self.read_bytes += len(data)
                        for piece in data.splitlines(keepends=True):
                            complete = piece.endswith(b"\n")
                            if not discarding:
                                buffer.extend(piece)
                                if len(buffer) > 65536:
                                    self.invalid_lines += 1
                                    buffer.clear()
                                    discarding = True
                            if complete:
                                if not discarding and buffer.strip():
                                    try:
                                        value = json.loads(buffer)
                                        if isinstance(value, dict):
                                            yield value
                                        else:
                                            self.invalid_lines += 1
                                    except (ValueError, UnicodeError):
                                        self.invalid_lines += 1
                                buffer.clear()
                                discarding = False
                    if self.remaining == 0:
                        self.limited = True
                        break
            except OSError:
                self.unavailable_files += 1
        if buffer and self.remaining == 0:
            self.budget_partial_lines += 1
        elif buffer:
            self.invalid_lines += 1


def _paths(directory: Path, prefix: str, reader: BoundedLogs) -> list[Path]:
    try:
        entries = list(itertools.islice(directory.iterdir(), 1025))
    except OSError:
        reader.unavailable_files += 1
        return []
    reader.limited |= len(entries) > 1024
    matches = [path for path in entries[:1024] if path.name.startswith(prefix) and
               path.suffix in (".log", ".ndjson") and not path.is_symlink()]
    reader.limited |= len(matches) > 256
    def order(path: Path) -> tuple[str, int]:
        match = re.search(r"(\d+)\.(?:log|ndjson)$", path.name)
        return (path.name[:match.start()] if match else path.name, int(match[1]) if match else 0)
    return sorted(matches, key=order)[:256]


def _status(directory: Path) -> dict[str, Any]:
    try:
        with (directory / "status.json").open("rb") as source:
            data = source.read(512 * 1024 + 1)
        result = json.loads(data) if len(data) <= 512 * 1024 else None
        return result if isinstance(result, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def _footer_records(paths: list[Path], reader: BoundedLogs) -> Iterator[dict[str, Any]]:
    """Inspect complete records in bounded file tails independently of the event budget.

    The native writer appends its footer last. A large healthy capture must not
    lose this evidence just because the sequential event scan has stopped.
    This verifies footer records, not the integrity of all preceding events.
    """
    for path in paths:
        try:
            with path.open("rb") as source:
                size = source.seek(0, 2)
                offset = max(0, size - 65536)
                source.seek(offset)
                data = source.read(65536)
            reader.read_bytes += len(data)
            # A tail may begin or end inside a record; neither fragment is proof
            # of a complete footer. Native records do not span rotated files.
            if offset:
                data = data.partition(b"\n")[2]
            for line in data.splitlines(keepends=True):
                if not line.endswith(b"\n") or b'"capture_footer"' not in line:
                    continue
                try:
                    value = json.loads(line)
                except (ValueError, UnicodeError):
                    reader.invalid_lines += 1
                    continue
                if isinstance(value, dict) and value.get("kind") == "capture_footer":
                    yield value
        except OSError:
            reader.unavailable_files += 1


def summarize_run(directory: Path, maximum_bytes: int = 64 * 1024 * 1024) -> dict[str, Any]:
    reader = BoundedLogs(maximum_bytes)
    receiver_reports = 0
    delivered: Any = None
    maximum_received: Any = None
    gaps = failures = partial_output = 0
    for event in reader.records(_paths(directory, "traffic.", reader)):
        interval = event.get("interval")
        if (isinstance(interval, dict) and interval.get("reporter") == "receiver"
                and interval.get("receiver_report_available") is not False):
            value = interval.get("bytes")
            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                receiver_reports += 1
                maximum_received = max(maximum_received or 0, value)
                delivered = bool(delivered or value > 0)
        gaps += event.get("kind") == "traffic_gap"
        failures += event.get("kind") == "traffic_launch_failed" or (event.get("kind") == "traffic_generation_finished"
                                                                   and not event.get("completed"))
        partial_output += event.get("kind") in ("traffic_output_truncated", "traffic_output_incomplete")
    pdu_accept_observed: Any = None
    for event in reader.records(_paths(directory, "recovery.", reader)):
        if event.get("event") == "pdu_accepted":
            pdu_accept_observed = True
        snapshot = event.get("native_values")
        if (isinstance(snapshot, dict) and type(snapshot.get("pdu_accepts")) is int
                and snapshot["pdu_accepts"] > 0):
            pdu_accept_observed = True
    footer_count = clean_footers = invalid_footers = 0
    async_repeat_summaries = async_repeated_records = 0
    losses: dict[str, int] = {}
    recorder_paths = _paths(directory / "recorder", "oai-flight-recorder-", reader)
    for event in reader.records(recorder_paths):
        if event.get("kind") == "async_repeat_summary":
            count = event.get("repeat_count")
            if type(count) is int and count > 0:
                async_repeat_summaries += 1
                async_repeated_records += count
        if event.get("kind") == "event" and event.get("event") == 15 and event.get("d") == 1:
            pdu_accept_observed = True
    footer_reader = BoundedLogs()
    for event in _footer_records(recorder_paths, footer_reader):
        if event.get("kind") == "capture_footer":
            footer_count += 1
            required = ("clean", "writer_errors", "dropped_ring_full", "dropped_no_slot", "no_slot_threads",
                        "invalid_timestamp_records", "payload_truncated", "rings_assigned")
            valid_footer = all(type(event.get(key)) is int and event[key] >= 0 for key in required)
            invalid_footers += not valid_footer
            clean_footers += valid_footer and event["clean"] == 1
            for key in ("writer_errors", "dropped_ring_full", "dropped_no_slot", "invalid_timestamp_records",
                        "payload_truncated"):
                value = event.get(key)
                if type(value) is int and value >= 0:
                    losses[key] = losses.get(key, 0) + value
    status = _status(directory)
    writers = status.get("writers")
    writer_losses = {name: writer.get("dropped_bytes") for name, writer in writers.items()
                     if isinstance(writer, dict)} if isinstance(writers, dict) else None
    return {"schema_version": 1, "kind": "flight_capture_summary", "capture_directory": str(directory),
            "control_plane": {"pdu_accept_observed": pdu_accept_observed,
                              "scope": "decoded PDU acceptance only; absent evidence is unavailable"},
            "traffic": {"delivered_bytes_observed": delivered, "receiver_reports": receiver_reports,
                        "maximum_receiver_report_bytes": maximum_received, "gap_records": gaps,
                        "failed_or_interrupted_generations": failures, "incomplete_output_records": partial_output,
                        "scope": "rounded receiver text reports only; duplicate reports are not summed; no video-delivery claim"},
            "capture": {"final_status_present": bool(status), "reported_capture": status.get("capture"),
                        "writer_dropped_bytes": writer_losses, "footer_count": footer_count,
                        "clean_footer_observed": clean_footers == 1 and footer_count == 1,
                        "invalid_footer_records": invalid_footers,
                        "async_repeat_summaries": async_repeat_summaries,
                        "async_repeated_records": async_repeated_records,
                        "footer_losses": losses if footer_count else None,
                        "missing_footer_means": "unavailable completeness; crash, kill, live capture, recorder failure or scan limits possible",
                        "footer_scan": {"bytes_read": footer_reader.read_bytes,
                                        "unavailable_files": footer_reader.unavailable_files,
                                        "invalid_lines": footer_reader.invalid_lines,
                                        "scope": "complete records in last 65536 bytes of each selected native file; not full capture integrity"}},
            "scan": {"bytes_read": reader.read_bytes, "maximum_bytes": maximum_bytes,
                     "limited": reader.limited, "invalid_or_partial_lines": reader.invalid_lines,
                     "budget_partial_lines": reader.budget_partial_lines,
                     "unavailable_files": reader.unavailable_files,
                     "limits": "1024 entries/directory, 256 files/stream, 65536 bytes/line"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_directory", type=Path, help="one attempt directory, not a recovery session parent")
    args = parser.parse_args()
    print(json.dumps(summarize_run(args.capture_directory), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
