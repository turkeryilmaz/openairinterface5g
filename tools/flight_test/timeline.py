#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Offline flight timelines: full selected CSV/JSONL and bounded standalone plots.

Example: python3 timeline.py --gnb GNB_SESSION --ue UE_SESSION --output NEWDIR
Inputs are attempts or recovery sessions containing attempts/. Input files are
read only. NEWDIR must not exist and must be outside every input.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

try:
    from . import decode_events as decoder
except ImportError:
    import decode_events as decoder

SELECTED = set(range(10, 28)) | set(range(40, 70)) | set(decoder.TIMING_LAYOUTS)
COLUMNS = ["capture", "attempt", "role", "host", "boot_id", "clock_domain", "clock_segment",
           "time_basis", "monotonic_ns", "utc_wall_ns", "alignment", "event", "name", "category",
           "source_file", "source_line", "ring", "sequence", "frame", "slot", "rnti", "sync_epoch",
           "fields", "metrics"]
SCOPES = [
    "Native clocks are source event clocks; host, recovery, traffic and core times are collector observations.",
    "Monotonic times compare only within a proven host boot. Unknown boots use separate attempt domains.",
    "Host UTC alignment is unverified. Shared experiment/run IDs and chrony tracking are not clock calibration.",
    "Clock segments split when consecutive wall-minus-monotonic offsets change beyond the selected threshold.",
    "No one-way latency, cross-host RACH pairing or SFN-only links are inferred. SFN wraps every 10.24 seconds.",
    "Core source timestamps remain original fields; collector receipt UTC is plotted. Packet payloads are not read.",
    "Recovery durations use collector receipts within one attempt, proven host boot and clock segment only.",
    "Positive local receiver intervals show observed delivery; idle time and absent reports are not failures.",
    "PDU acceptance and active DRB are separate from delivery; interface/TUN presence does not prove a working bearer.",
    "Gain readbacks are device settings, not calibrated RF power. Different units have separate plots.",
    "SQLite bounds normalization/sorting memory; production decode() retains its own typed records and sequence map.",
]


def packed(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def integer(value: Any) -> int | None:
    return value if type(value) is int and value != decoder.INT64_MIN else None


def number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def read_metadata(directory: Path) -> dict:
    path = directory / "metadata.json"
    if not path.is_file():
        return {}
    with path.open("rb") as source:
        content = source.read(1024 * 1024 + 1)
    if len(content) > 1024 * 1024:
        raise ValueError(f"metadata exceeds 1 MiB: {path}")
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError(f"metadata is not an object: {path}")
    return value


def rotation_key(path: Path) -> tuple:
    match = re.search(r"(\d+)\.(?:log|ndjson)$", path.name)
    return (path.name[:match.start()] if match else path.name, int(match[1]) if match else 0, path.name)


def records(directory: Path, prefix: str, warnings: Counter) -> Iterator[tuple[dict, str, int]]:
    """Stream rotating JSONL, including a record split between adjacent files."""
    pending = bytearray()
    reference = None
    discarding = False
    paths = sorted((p for p in directory.glob(prefix + ".*.log") if p.is_file()), key=rotation_key)
    for path in paths:
        with path.open("rb") as source:
            line, continuation = 0, False
            while True:
                piece = source.readline(65537)
                if not piece:
                    break
                if not continuation:
                    line += 1
                continuation = not piece.endswith(b"\n")
                if reference is None:
                    reference = (str(path), line)
                if not discarding:
                    pending.extend(piece)
                    if len(pending) > 1024 * 1024:
                        warnings["overlong_jsonl_records"] += 1
                        pending.clear()
                        discarding = True
                if piece.endswith(b"\n"):
                    if not discarding and pending.strip():
                        try:
                            value = json.loads(pending)
                            if isinstance(value, dict):
                                yield value, reference[0], reference[1]
                            else:
                                warnings["non_object_jsonl_records"] += 1
                        except (ValueError, UnicodeError):
                            warnings["invalid_jsonl_records"] += 1
                    pending.clear()
                    reference = None
                    discarding = False
    if pending or discarding:
        warnings["partial_jsonl_records"] += 1


def boot_value(observation: dict) -> str | None:
    value = observation.get("boot_id")
    if isinstance(value, dict):
        value = value.get("value") if value.get("state") == "available" else None
    return value if isinstance(value, str) and value else None


def clock_values(clock: dict) -> tuple:
    mono, utc = integer(clock.get("monotonic_ns")), integer(clock.get("utc_wall_ns"))
    return (mono if mono is not None and mono >= 0 else None, utc if utc is not None and utc >= 0 else None)


def native_fields(event: dict) -> tuple[dict, list]:
    """Reuse typed production decoders; interpret only simple schema-v1 columns."""
    kind = event["event"]
    typed = {}
    for decode in (decoder.decode_radio_timing, decoder.decode_radio_gain_event,
                   decoder.decode_radio_rx_decision_event, decoder.decode_radio_rx_peak_envelope,
                   decoder.decode_relative_tx, decoder.decode_gnb_prach, decoder.decode_tx_reference_event,
                   decoder.decode_radio_tx_power_event, decoder.decode_ue_tx_control,
                   decoder.decode_ue_pathloss_state, decoder.decode_ue_ssb_measurement):
        try:
            value = decode(event)
        except (ValueError, KeyError, TypeError) as error:
            return {"raw": {key: event[key] for key in "abcdef"}, "typed": None,
                    "typed_decode_error": str(error)}, []
        if value is not None:
            typed = {key: value for key, value in value.items() if not key.startswith("_")}
            break
    fields = {"raw": {key: event[key] for key in "abcdef"}, "typed": typed}
    metrics = []

    def add(name: str, value: Any, unit: str, entity: str = "") -> None:
        if number(value) and value != decoder.INT64_MIN:
            metrics.append([name, value, unit, entity])

    def frame_slot(value: int) -> None:
        if value >= 0:
            fields.update(frame=value // 1000, slot=value % 1000, frame_slot_raw=value)

    if kind in (12, 13, 16):
        frame_slot(event["b"])
    elif kind == 23:
        frame_slot(event["c"])
    elif kind == 21:
        fields["frame"] = integer(event["c"])
    fields.update({key: typed[key] for key in ("frame", "slot", "rnti", "sync_epoch", "prach_token",
                                              "frame_slot", "absolute_tx_slot", "absolute_rx_slot") if key in typed})
    if type(typed.get("frame_slot")) is int:
        frame_slot(typed["frame_slot"])
    if kind in (13, 16):
        fields["rnti"] = integer(event["c"])
    elif kind in (21, 23):
        fields["rnti"] = integer(event["b"])
    elif kind in (22, 24, 25, 26):
        fields["rnti"] = integer(event["a"])
    entity = f"RNTI {fields['rnti']}" if fields.get("rnti") is not None else ""
    outcome = ""
    if kind == 10:
        fields["sync_success"] = event["b"] == 1 if event["b"] in (0, 1) else None
        outcome = {0: "failure", 1: "success"}.get(event["b"], "unknown outcome")
    elif kind == 13:
        fields["ra_outcome"] = {1: "success", 2: "contention failure", 3: "RAR failure"}.get(event["d"], "unknown")
        outcome = fields["ra_outcome"]
    elif kind == 23:
        fields["ra_stage"] = {0: "initiation", 1: "positive feedback", 2: "negative feedback",
                              3: "Msg3 exhausted"}.get(event["a"], "unknown")
        outcome = fields["ra_stage"]
    elif kind == 19:
        fields["control_action"] = {1: "RRC transition", 2: "RLF", 3: "idle fallback", 4: "NAS reject"}.get(event["b"], "unknown")
        outcome = fields["control_action"]
    elif kind == 15:
        fields["pdu_accept_observed"] = event["d"] == 1
        outcome = "accept" if event["d"] == 1 else "unknown outcome"
    if kind in (10, 13, 14, 15, 17, 18, 19, 23):
        add(decoder.NAMES[kind] + (" " + outcome if outcome else ""), kind, "event code", entity)
    if kind == 12:
        add("UE SS-RSRP", integer(event["e"]), "dBm")
        add("UE L1 SINR", integer(event["f"]), "dB")
    elif kind == 11:
        add("UE legacy configured gain minus offset", integer(event["d"]), "gain setting dB")
    elif kind == 21:
        add("gNB DL MAC SDU counter", integer(event["d"]), "bytes", entity)
        add("gNB UL MAC SDU counter", integer(event["e"]), "bytes", entity)
    elif kind == 22:
        for key, name, unit in (("b", "DL MCS", "MCS index"), ("c", "UL MCS", "MCS index"),
                                ("d", "DL HARQ errors", "count"), ("e", "UL HARQ errors", "count"),
                                ("f", "UL DTX", "count")):
            add("gNB " + name, integer(event[key]), unit, entity)
    elif kind == 24:
        for key, name, unit, divisor in (("b", "PH", "dB", 1), ("c", "PCMAX", "dBm", 1),
                                         ("d", "PUCCH DTX", "count", 1), ("e", "mean RSRP", "dBm", 1),
                                         ("f", "mean SINR", "dB", 10)):
            value = integer(event[key])
            add("gNB " + name, value / divisor if value is not None else None, unit, entity)
    elif kind in (25, 26):
        direction = "DL" if kind == 25 else "UL"
        for index, key in enumerate("bcde"):
            add(f"gNB {direction} HARQ round {index}", integer(event[key]), "count", entity)
        add(f"gNB {direction} MAC transport counter", integer(event["f"]), "bytes", entity)
    typed_metrics = {
        40: (("rx_readback_db", "RX device gain", "gain setting dB"), ("tx_readback_db", "TX device gain", "gain setting dB")),
        42: (("mean_power_dbfs", "sampled RX mean", "dBFS"), ("peak_component_dbfs", "sampled RX peak", "dBFS")),
        53: (("ssb_pathloss_db", "UE serving SSB pathloss", "dB"),
             ("adjustment_state_after_db", "UE closed-loop adjustment", "dB"),
             ("configured_network_pmax_dbm", "UE configured network p-Max", "dBm")),
        54: (("rsrp_dbm", "UE pathloss input RSRP", "dBm"), ("ssb_pathloss_db", "UE checked SSB pathloss", "dB")),
        55: (("accepted_rsrp_dbm", "UE accepted SSB RSRP", "dBm"),),
        56: (("reported_rx_gain_db", "SSB context reported gain", "gain setting dB"),),
        65: (("candidate_energy_db", "gNB PRACH candidate energy", "dB"), ("noise_i0_db", "gNB PRACH noise reference", "dB")),
        110: (("signal_noise_ratio_db", "gNB PUSCH signal/noise ratio", "dB"),),
        111: (("ul_cqi_raw", "gNB PUSCH CQI raw", "raw CQI code"),),
    }
    for key, name, unit in typed_metrics.get(kind, ()):
        add(name, typed.get(key), unit, entity)
    if kind in (80, 84, 87, 88, 90, 91, 119):
        add(decoder.NAMES[kind], kind, "event code", entity)
    if kind == 111:
        add("gNB PUSCH CRC failed", int(typed["crc_failed"]), "boolean", entity)
        add("gNB PUSCH DTX", int(typed["dtx"]), "boolean", entity)
    return fields, metrics


class Report:
    def __init__(self, output: Path, clock_step_ms: float, max_plot_points: int, max_plot_series: int):
        self.output = output
        self.clock_step_ns = int(clock_step_ms * 1e6)
        self.max_plot_points = max_plot_points
        self.max_plot_series = max_plot_series
        self.warnings = Counter()
        self.inputs = []
        self.db = sqlite3.connect(output / "ordering.sqlite3")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.execute("PRAGMA cache_size=-8192")
        self.db.execute("CREATE TABLE events(id INTEGER PRIMARY KEY, capture TEXT, attempt TEXT, role TEXT, host TEXT, "
                        "boot TEXT, domain TEXT, mono INTEGER, utc INTEGER, segment INTEGER DEFAULT 0, "
                        "name TEXT, kind INTEGER, origin TEXT, category TEXT, source TEXT, line INTEGER, "
                        "ring INTEGER, sequence INTEGER, fields TEXT, metrics TEXT)")
        self.db.execute("CREATE TABLE points(series TEXT, family TEXT, unit TEXT, event_id INTEGER, value REAL)")

    def add(self, context: dict, *, name: str, kind: int | None, origin: str, category: str,
            mono: int | None, utc: int | None, source: str, line: int, fields: dict, metrics: list,
            boot: str | None = None, ring: int | None = None, sequence: int | None = None) -> None:
        boot = boot or context.get("boot")
        host = context["host"]
        domain = packed([host, boot]) if boot and context["host_known"] else packed([context["capture"], context["attempt"], host])
        self.db.execute("INSERT INTO events(capture,attempt,role,host,boot,domain,mono,utc,name,kind,origin,category,"
                        "source,line,ring,sequence,fields,metrics) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (context["capture"], context["attempt"], context["role"], host, boot, domain,
                         mono, utc, name, kind, origin, category, source, line, ring, sequence, packed(fields), packed(metrics)))

    def ingest(self, root: Path, role: str, capture: str) -> None:
        if role == "core":
            self.ingest_core(root, capture)
            return
        session_meta = read_metadata(root)
        attempts_root = root / "attempts"
        attempts = sorted((p for p in attempts_root.iterdir() if p.is_dir()), key=lambda p: p.name) if attempts_root.is_dir() else [root]
        if not attempts:
            raise ValueError(f"no attempt directories: {root}")
        for ordinal, attempt in enumerate(attempts, 1):
            meta = read_metadata(attempt)
            identity = dict(session_meta, **meta)
            host = identity.get("runtime", {}).get("host", {}).get("nodename")
            host_known = isinstance(host, str) and bool(host)
            if not host_known:
                host = f"unknown-host:{capture}"
            context = dict(capture=capture, attempt=attempt.name, role=role, host=host, host_known=host_known, boot=None)
            boots = set()
            before = self.warnings.copy()
            for row, source, line in records(attempt, "host", self.warnings):
                observation = row.get("observation", row)
                if row.get("kind") != "health" and observation.get("kind") != "host_observation":
                    continue
                boot = boot_value(observation)
                if boot:
                    boots.add(boot)
                mono, utc = clock_values(observation.get("clock", {}))
                fields = {"clock": observation.get("clock", {}), "boot_id": observation.get("boot_id"),
                          "chrony": observation.get("chrony", {}), "assessment": row.get("assessment")}
                self.add(context, name="host_health", kind=None, origin="collector observation", category="host",
                         mono=mono, utc=utc, source=source, line=line, fields=fields, metrics=[], boot=boot)
            context["boot"] = next(iter(boots)) if len(boots) == 1 else None
            if len(boots) > 1:
                self.warnings["attempts_with_multiple_boot_ids_native_boot_unavailable"] += 1
            info = {"capture": capture, "attempt": attempt.name, "role": role, "directory": str(attempt),
                    "host": host, "host_identity_available": host_known, "boot_ids": sorted(boots),
                    "experiment_id": identity.get("experiment_id"), "run_id": identity.get("user_run_id", identity.get("run_id")),
                    "session_id": identity.get("session_id"), "decoder": None}
            recorder = attempt / "recorder"
            if recorder.is_dir() and any(recorder.glob("oai-flight-recorder-*.ndjson")):
                decoded = self.output / "decoded" / capture / f"{ordinal:06d}"
                status = decoder.decode(recorder, decoded)
                info["decoder"] = {"directory": str(decoded.relative_to(self.output)),
                    **{key: status.get(key) for key in ("files", "events", "invalid_lines", "unsupported_schema",
                                                       "clean_footer_observed", "footers", "sampling_config")}}
                with (decoded / "events.csv").open(newline="") as source_csv:
                    for row in csv.DictReader(source_csv):
                        event = {key: int(row[key]) for key in decoder.FIELDS[3:]}
                        if event["event"] not in SELECTED:
                            continue
                        event.update(source_file=row["source_file"], source_line=int(row["source_line"]))
                        fields, metrics = native_fields(event)
                        if "typed_decode_error" in fields:
                            self.warnings["invalid_native_typed_payloads"] += 1
                        self.add(context, name=decoder.NAMES.get(event["event"], "UNKNOWN"), kind=event["event"],
                                 origin="source event", category="native", mono=integer(event["mono_ns"]),
                                 utc=integer(event["realtime_ns"]), source=str(recorder / row["source_file"]),
                                 line=int(row["source_line"]), fields=fields, metrics=metrics,
                                 ring=event["ring"], sequence=event["sequence"])
            else:
                self.warnings["attempts_without_native_recorder"] += 1
            for prefix in ("recovery", "traffic"):
                self.ingest_observations(attempt, prefix, context)
            info["scan_warnings"] = dict(self.warnings - before)
            self.inputs.append(info)
        if attempts_root.is_dir():
            context = dict(capture=capture, attempt="session", role=role, host=f"unknown-host:{capture}",
                           host_known=False, boot=None)
            self.ingest_observations(root, "recovery", context)
        self.db.commit()

    def ingest_core(self, root: Path, capture: str) -> None:
        """Keep receipt and original source timestamps separate; never read packet payloads."""
        meta = read_metadata(root)
        prefixes = set()
        for path in root.glob("auxiliary-*.log"):
            match = re.fullmatch(r"(.+)\.[0-9]+\.log", path.name)
            if match:
                prefixes.add(match[1])
        if not prefixes:
            raise ValueError(f"no auxiliary JSONL streams: {root}")
        count = 0
        for prefix in sorted(prefixes):
            for row, source, line in records(root, prefix, self.warnings):
                name = row.get("kind")
                if not isinstance(name, str) or not name.startswith("auxiliary_"):
                    self.warnings["invalid_auxiliary_records"] += 1
                    continue
                host = row.get("host")
                known = isinstance(host, str) and bool(host)
                context = dict(capture=capture, attempt=root.name, role="core",
                               host=host if known else f"unknown-host:{capture}", host_known=known, boot=None)
                mono, utc = clock_values(row.get("clock", {}))
                fields = {key: value for key, value in row.items() if key != "clock"}
                fields["source_timestamp_alignment"] = "original source timestamp; relation to collector unverified"
                entity = "/".join(str(row.get(key, "")) for key in ("source_kind", "source_name", "stream"))
                self.add(context, name=name, kind=None, origin="collector observation", category="core",
                         mono=mono, utc=utc, source=source, line=line, fields=fields,
                         metrics=[["core receipt " + name, 1, "core observation", entity]])
                count += 1
        self.inputs.append({"capture": capture, "attempt": root.name, "role": "core", "directory": str(root),
                            "experiment_id": meta.get("run_id"), "run_id": meta.get("run_id"),
                            "auxiliary_records": count, "boot_identity": "unavailable",
                            "alignment": "collector UTC unverified; source timestamps retained separately"})
        self.db.commit()

    def ingest_observations(self, directory: Path, prefix: str, context: dict) -> None:
        for row, source, line in records(directory, prefix, self.warnings):
            name = row.get("kind")
            if not isinstance(name, str):
                self.warnings["observation_without_kind"] += 1
                continue
            if prefix == "traffic" and name == "traffic_output" and not isinstance(row.get("interval"), dict):
                continue
            mono, utc = clock_values(row.get("receipt_clock", row.get("clock", {})))
            metrics = []
            fields = {key: value for key, value in row.items() if key not in ("text", "clock", "receipt_clock")}
            interval = row.get("interval")
            if (prefix == "traffic" and name == "traffic_output" and isinstance(interval, dict)
                    and not row.get("partial") and interval.get("reporter") == "receiver"
                    and interval.get("receiver_report_available") is not False
                    and interval.get("report_type") == "interval" and interval.get("source") == "local_output"):
                direction = interval.get("direction")
                value, start, end = (interval.get(key) for key in ("bytes", "start_seconds", "end_seconds"))
                if direction in ("ul", "dl") and all(number(v) and v >= 0 for v in (value, start, end)) and end > start:
                    fields["local_receiver_interval_valid"] = True
                    fields["delivery"] = "observed_positive" if value > 0 else "observed_zero"
                    metrics.append([f"{direction.upper()} local receiver rate", value * 8 / (end - start), "bit/s", ""])
                    if number(interval.get("loss_percent")) and 0 <= interval["loss_percent"] <= 100:
                        metrics.append([f"{direction.upper()} receiver datagram loss", interval["loss_percent"], "%", ""])
            if prefix == "recovery" and name == "native_progress_snapshot":
                values = row.get("native_values", {})
                for key in ("sync_successes", "pdu_accepts", "drb_context_active"):
                    if type(values.get(key)) is int and values[key] >= 0:
                        metrics.append([key + " (receipt)", values[key], "count", ""])
            self.add(context, name=name, kind=None, origin="collector observation", category=prefix,
                     mono=mono, utc=utc, source=source, line=line, fields=fields, metrics=metrics)

    def segment_clocks(self) -> int:
        self.db.execute("CREATE INDEX ordering ON events(domain,mono,id)")
        self.db.execute("CREATE INDEX event_kind ON events(category,name)")
        total = 0
        with (self.output / "clock_segments.csv").open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=["clock_domain", "segment", "first_monotonic_ns", "last_monotonic_ns",
                "paired_samples", "initial_wall_minus_mono_ns", "last_wall_minus_mono_ns", "trigger_delta_ns",
                "cross_host_uncertainty_ns"])
            writer.writeheader()
            for (domain,) in self.db.execute("SELECT DISTINCT domain FROM events ORDER BY domain"):
                segment, previous, start, last, count, initial, trigger = 0, None, None, None, 0, None, None

                def flush() -> None:
                    nonlocal total
                    if start is not None:
                        writer.writerow(dict(clock_domain=domain, segment=segment, first_monotonic_ns=start,
                            last_monotonic_ns=last, paired_samples=count, initial_wall_minus_mono_ns=initial,
                            last_wall_minus_mono_ns=previous, trigger_delta_ns=trigger, cross_host_uncertainty_ns=None))
                        total += 1

                for event_id, mono, utc in self.db.execute("SELECT id,mono,utc FROM events WHERE domain=? AND mono IS NOT NULL "
                                                          "ORDER BY mono,id", (domain,)):
                    offset = utc - mono if utc is not None else None
                    if offset is not None and previous is not None and abs(offset - previous) > self.clock_step_ns:
                        flush()
                        segment += 1
                        start, count, initial, trigger = None, 0, None, offset - previous
                    if start is None:
                        start = mono
                    last = mono
                    if offset is not None:
                        if initial is None:
                            initial = offset
                        previous = offset
                        count += 1
                    self.db.execute("UPDATE events SET segment=? WHERE id=?", (segment, event_id))
                flush()
        self.db.commit()
        return total

    def export(self) -> tuple[int, int]:
        event_count = recovery_count = 0
        state = {}
        with (self.output / "events.csv").open("w", newline="") as csv_dst, \
                (self.output / "events.jsonl").open("w") as json_dst, \
                (self.output / "recovery.csv").open("w", newline="") as recovery_dst:
            writer = csv.DictWriter(csv_dst, fieldnames=COLUMNS)
            writer.writeheader()
            recovery_writer = csv.DictWriter(recovery_dst, fieldnames=["capture", "attempt", "host", "boot_id", "clock_domain",
                "clock_segment", "loss_monotonic_ns", "milestone_monotonic_ns", "milestone", "seconds", "time_basis",
                "loss_source_file", "loss_source_line", "milestone_source_file", "milestone_source_line"])
            recovery_writer.writeheader()
            for raw in self.db.execute("SELECT * FROM events ORDER BY domain,mono IS NULL,mono,id"):
                (event_id, capture, attempt, role, host, boot, domain, mono, utc, segment, name, kind, origin,
                 category, source, line, ring, sequence, fields_json, metrics_json) = raw
                fields, metrics = json.loads(fields_json), json.loads(metrics_json)
                row = dict(zip(COLUMNS[:12], (capture, attempt, role, host, boot, domain, segment, origin, mono, utc,
                                              "host UTC, alignment unverified", kind)))
                row.update(name=name, category=category, source_file=source, source_line=line, ring=ring, sequence=sequence,
                           **{key: fields.get(key) for key in ("frame", "slot", "rnti", "sync_epoch")}, fields=fields, metrics=metrics)
                json_dst.write(packed(row) + "\n")
                writer.writerow(dict(row, fields=fields_json, metrics=metrics_json))
                event_count += 1
                for metric, value, unit, entity in metrics:
                    series = packed([capture, role, host, metric, entity])
                    family = packed([capture, role, host, metric])
                    self.db.execute("INSERT INTO points VALUES(?,?,?,?,?)", (series, family, unit, event_id, value))
                key = (capture, attempt, domain, segment)
                if state.get("key") != key:
                    # Interleaved attempts conservatively lose state rather than bridge restarts.
                    state = {"key": key, "ready": False, "accepts": 0, "loss": None, "seen": set()}
                if mono is None or boot is None:
                    continue
                milestones = []
                if name == "native_progress_snapshot":
                    values = fields.get("native_values", {})
                    accepts, active = values.get("pdu_accepts"), values.get("drb_context_active")
                    if type(accepts) is int and accepts >= 0:
                        if accepts < state["accepts"]:
                            state.update(ready=False, loss=None, seen=set())
                        if accepts > state["accepts"]:
                            milestones.append("PDU accepted counter increment observed")
                        state["accepts"] = accepts
                    if type(active) is int and active >= 0:
                        ready = active > 0 and state["accepts"] > 0
                        if state["ready"] and active == 0:
                            state["loss"] = (mono, source, line)
                            state["seen"] = set()
                        if ready and not state["ready"]:
                            milestones.append("DRB restored with PDU acceptance observed")
                        state["ready"] = ready
                elif fields.get("local_receiver_interval_valid") and fields.get("delivery") == "observed_positive":
                    milestones.append("first positive " + fields["interval"]["direction"].upper() + " receiver report observed")
                loss = state["loss"]
                for milestone in milestones:
                    if loss is not None and milestone not in state["seen"] and mono >= loss[0]:
                        recovery_writer.writerow(dict(capture=capture, attempt=attempt, host=host, boot_id=boot,
                            clock_domain=domain, clock_segment=segment, loss_monotonic_ns=loss[0], milestone_monotonic_ns=mono,
                            milestone=milestone, seconds=(mono - loss[0]) / 1e9, time_basis="collector observation",
                            loss_source_file=loss[1], loss_source_line=loss[2], milestone_source_file=source, milestone_source_line=line))
                        state["seen"].add(milestone)
                        recovery_count += 1
        self.db.execute("CREATE INDEX point_series ON points(series,event_id)")
        self.db.commit()
        return event_count, recovery_count

    def plots(self) -> tuple[list, int]:
        result = []
        total = self.db.execute("SELECT COUNT(*) FROM (SELECT series FROM points GROUP BY series)").fetchone()[0]
        minima = dict(self.db.execute("SELECT domain,MIN(mono) FROM events WHERE mono IS NOT NULL GROUP BY domain"))
        # Cover capture/metric families before spending the limit on additional
        # RNTIs. Within a family, show the most frequently observed entity first.
        groups = self.db.execute(
            "SELECT series,unit,n FROM (SELECT series,family,unit,COUNT(*) AS n,"
            "ROW_NUMBER() OVER (PARTITION BY family ORDER BY COUNT(*) DESC,series) AS entity_rank "
            "FROM points GROUP BY series,family,unit) ORDER BY entity_rank,family,series LIMIT ?",
            (self.max_plot_series,))
        for series, unit, count in groups:
            stride = max(1, math.ceil((count - 1) / (self.max_plot_points - 1)))
            points = []
            query = ("SELECT mono,utc,value,domain,segment,attempt,source,line,name FROM (SELECT e.mono,e.utc,p.value,"
                     "e.domain,e.segment,e.attempt,e.source,e.line,e.name,ROW_NUMBER() OVER (ORDER BY e.domain,e.mono,e.id) AS n "
                     "FROM points p JOIN events e ON e.id=p.event_id WHERE p.series=?) WHERE (n-1)%?=0 OR n=? ORDER BY n")
            for mono, utc, value, domain, segment, attempt, source, line, name in self.db.execute(query, (series, stride, count)):
                points.append([None if mono is None else (mono - minima[domain]) / 1e9,
                               None if utc is None else utc / 1e6, value, packed([domain, segment, attempt]),
                               str(Path(source).name) + ":" + str(line) + " " + name])
            result.append(dict(label=" / ".join(str(v) for v in json.loads(series) if v), unit=unit,
                               source_points=count, plotted_points=len(points), sampling="uniform index with first/last",
                               points=points))
        return result, max(0, total - len(result))

    def close(self) -> None:
        self.db.close()



def render_html(metadata: dict, plots: list) -> str:
    # Escape the script delimiter as well as HTML, including untrusted metadata.
    payload = packed(plots).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    facts = html.escape(json.dumps(metadata, indent=2, ensure_ascii=True))
    scopes = "".join("<li>" + html.escape(item) + "</li>" for item in SCOPES)
    return """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Offline flight timeline</title><style>
body{font:15px system-ui,sans-serif;margin:24px;max-width:1500px;color:#142534;background:#f6f8fb}
h1,h2{color:#123b55}a{color:#075985}li{margin:.4em 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}
.controls{position:sticky;top:0;background:#e4eef6;padding:12px;z-index:1}label{margin-right:15px}
article{background:white;border:1px solid #cbd5e1;border-radius:8px;padding:14px;margin:15px 0}
canvas{width:100%;height:245px;border:1px solid #e2e8f0}input[type=number]{width:130px}
.legend{font-size:13px;color:#465869}.point{min-height:1.6em;overflow-wrap:anywhere}
</style><h1>Offline flight timeline</h1><p>Full selected records: <a href="events.csv">CSV</a>,
<a href="events.jsonl">JSONL</a>. <a href="recovery.csv">Observed recovery durations</a>.
<a href="clock_segments.csv">Clock segments</a>. <a href="metadata.json">Report metadata</a>.</p>
<ul>""" + scopes + """</ul><p>Plots contain bounded uniform samples, retaining first/last. Rare events and
extrema may be omitted. Each series lists plotted/source counts. Full exports retain every selected row.</p>
<div class="controls"><label>Time axis <select id="basis"><option value="1">Host UTC, alignment unverified</option>
<option value="0">Seconds since first observation in each host boot/domain</option></select></label>
<label>From <input id="from" type="number" step="any"></label><label>To <input id="to" type="number" step="any"></label>
<button id="apply">Apply range</button> <button id="reset">Reset range</button></div><div id="plots"></div>
<details><summary>Inputs, clock policy, coverage and limitations</summary><pre>""" + facts + """</pre></details>
<script id="data" type="application/json">""" + payload + """</script><script>
'use strict';
const series=JSON.parse(document.getElementById('data').textContent),root=document.getElementById('plots');
let basis=1,limits=null;
const groups=new Map();
for(const s of series){if(!groups.has(s.unit))groups.set(s.unit,[]);groups.get(s.unit).push(s);}
function extent(){let lo=Infinity,hi=-Infinity;for(const s of series)for(const p of s.points){if(p[basis]!==null){lo=Math.min(lo,p[basis]);hi=Math.max(hi,p[basis]);}}return lo===Infinity?[0,1]:[lo,hi===lo?lo+1:hi];}
function labelTime(v){return basis===1?new Date(v).toISOString():v.toFixed(3)+' s';}
function reset(){limits=extent();document.getElementById('from').value=limits[0];document.getElementById('to').value=limits[1];draw();}
function draw(){root.replaceChildren();for(const [unit,rows]of groups){
const article=document.createElement('article'),title=document.createElement('h2'),legend=document.createElement('div'),canvas=document.createElement('canvas'),info=document.createElement('div');
title.textContent=unit;legend.className='legend';info.className='point';
legend.textContent=rows.map(s=>s.label+' ['+s.plotted_points+'/'+s.source_points+' points]').join(' | ');
article.append(title,legend,canvas,info);root.append(article);canvas.width=Math.max(600,canvas.clientWidth*devicePixelRatio);canvas.height=245*devicePixelRatio;
const c=canvas.getContext('2d'),w=canvas.width,h=canvas.height,pad=70*devicePixelRatio,right=15*devicePixelRatio,top=20*devicePixelRatio,bottom=48*devicePixelRatio;
let ymin=Infinity,ymax=-Infinity;
for(const s of rows)for(const p of s.points)if(p[basis]!==null&&p[basis]>=limits[0]&&p[basis]<=limits[1]){ymin=Math.min(ymin,p[2]);ymax=Math.max(ymax,p[2]);}
if(ymin===Infinity){c.font=14*devicePixelRatio+'px sans-serif';c.fillText('No sampled observations in range',pad,h/2);continue;}
if(ymin===ymax){ymin-=1;ymax+=1;}const gap=(ymax-ymin)*.05;ymin-=gap;ymax+=gap;
const x=v=>pad+(v-limits[0])/(limits[1]-limits[0])*(w-pad-right),y=v=>h-bottom-(v-ymin)/(ymax-ymin)*(h-top-bottom);
c.font=11*devicePixelRatio+'px sans-serif';c.strokeStyle='#cbd5e1';c.fillStyle='#334155';c.lineWidth=devicePixelRatio;
for(let i=0;i<=4;i++){let yy=top+(h-top-bottom)*i/4;c.beginPath();c.moveTo(pad,yy);c.lineTo(w-right,yy);c.stroke();c.fillText((ymax-(ymax-ymin)*i/4).toPrecision(4),2,yy+4);}
c.fillText(labelTime(limits[0]),pad,h-14*devicePixelRatio);c.textAlign='right';c.fillText(labelTime(limits[1]),w-right,h-14*devicePixelRatio);c.textAlign='left';
const hits=[];rows.forEach((s,index)=>{c.strokeStyle=c.fillStyle='hsl('+((index*137.508)%360)+' 65% 38%)';let previous=null;
for(const p of s.points){const v=p[basis];if(v===null||v<limits[0]||v>limits[1]){previous=null;continue;}const xx=x(v),yy=y(p[2]);
if(unit!=='event code'&&unit!=='core observation'&&previous&&previous[3]===p[3]){c.beginPath();c.moveTo(x(previous[basis]),y(previous[2]));c.lineTo(xx,yy);c.stroke();}
c.beginPath();c.arc(xx,yy,2.2*devicePixelRatio,0,2*Math.PI);c.fill();hits.push([xx,yy,s,p]);previous=p;}});
canvas.onmousemove=e=>{const rect=canvas.getBoundingClientRect(),xx=(e.clientX-rect.left)*canvas.width/rect.width,yy=(e.clientY-rect.top)*canvas.height/rect.height;let best=null,d=Infinity;for(const hit of hits){let z=(hit[0]-xx)**2+(hit[1]-yy)**2;if(z<d){d=z;best=hit;}}info.textContent=best&&d<1600*devicePixelRatio**2?best[2].label+' = '+best[3][2]+' '+unit+'; '+labelTime(best[3][basis])+'; '+best[3][4]:'';};
canvas.onwheel=e=>{e.preventDefault();let center=limits[0]+(e.offsetX/canvas.clientWidth)*(limits[1]-limits[0]),factor=e.deltaY>0?1.25:.8;limits=[center+(limits[0]-center)*factor,center+(limits[1]-center)*factor];document.getElementById('from').value=limits[0];document.getElementById('to').value=limits[1];draw();};
}}
document.getElementById('basis').onchange=e=>{basis=Number(e.target.value);reset();};document.getElementById('reset').onclick=reset;
document.getElementById('apply').onclick=()=>{const a=Number(document.getElementById('from').value),b=Number(document.getElementById('to').value);if(Number.isFinite(a)&&Number.isFinite(b)&&b>a){limits=[a,b];draw();}};reset();
</script></html>"""


def generate(inputs: list[tuple[str, Path]], output: Path, *, clock_step_ms: float = 5,
             max_plot_points: int = 2000, max_plot_series: int = 64) -> dict:
    if not inputs:
        raise ValueError("provide at least one --gnb, --ue or --core capture")
    if not math.isfinite(clock_step_ms) or clock_step_ms <= 0 or not 2 <= max_plot_points <= 10000 or not 1 <= max_plot_series <= 256:
        raise ValueError("positive clock threshold, 2..10000 plot points and 1..256 plot series required")
    roots = [(role, path.resolve(strict=True)) for role, path in inputs]
    if any(not path.is_dir() for _, path in roots):
        raise ValueError("capture input is not a directory")
    output = output.resolve()
    if any(output == path or path in output.parents for _, path in roots):
        raise ValueError("output must be outside every input capture")
    if len({path for _, path in roots}) != len(roots):
        raise ValueError("duplicate capture input")
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    report = Report(output, clock_step_ms, max_plot_points, max_plot_series)
    try:
        for ordinal, (role, path) in enumerate(roots, 1):
            report.ingest(path, role, f"{ordinal:03d}-{role}-{path.name}")
        segments = report.segment_clocks()
        count, recovery_count = report.export()
        plots, omitted = report.plots()
        metadata = {"schema_version": 1, "kind": "offline_flight_timeline", "status": "complete",
                    "inputs": report.inputs, "selected_native_event_ids": sorted(SELECTED), "normalized_rows": count,
                    "recovery_duration_rows": recovery_count, "clock_segment_count": segments,
                    "clock_step_threshold_ms": clock_step_ms, "cross_host_alignment": "unverified",
                    "cross_host_uncertainty_ns": None, "clock_segment_policy": "adjacent wall-minus-monotonic offset change",
                    "chrony_policy": "tracking observations retained in events; no calibrated uncertainty bound",
                    "plot_family_selection": "one representative per capture/metric family before additional entities",
                    "plot_series": len(plots), "plot_series_omitted": omitted, "maximum_points_per_series": max_plot_points,
                    "plot_source_points": sum(s["source_points"] for s in plots),
                    "plot_sampled_points": sum(s["plotted_points"] for s in plots),
                    "warnings": dict(report.warnings), "scope": SCOPES}
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
        (output / "report.html").write_text(render_html(metadata, plots))
    except Exception as error:
        (output / "INCOMPLETE.json").write_text(packed({"status": "incomplete", "error": str(error)}) + "\n")
        raise
    finally:
        report.close()
    (output / "ordering.sqlite3").unlink()
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gnb", type=Path, action="append", default=[], help="gNB attempt/session; repeatable")
    parser.add_argument("--ue", type=Path, action="append", default=[], help="UE attempt/session; repeatable")
    parser.add_argument("--core", type=Path, action="append", default=[], help="auxiliary JSONL directory; repeatable")
    parser.add_argument("--output", type=Path, required=True, help="new report directory outside captures; parent must exist")
    parser.add_argument("--clock-step-ms", type=float, default=5, help="offset jump threshold in milliseconds (default 5)")
    parser.add_argument("--max-plot-points", type=int, default=2000, help="sampled points/series (default 2000; full exports retained)")
    parser.add_argument("--max-plot-series", type=int, default=64, help="plotted series (default 64; full exports retained)")
    args = parser.parse_args()
    try:
        result = generate([("gnb", p) for p in args.gnb] + [("ue", p) for p in args.ue] + [("core", p) for p in args.core], args.output,
                          clock_step_ms=args.clock_step_ms, max_plot_points=args.max_plot_points,
                          max_plot_series=args.max_plot_series)
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(2, f"timeline: {error}\n")
    print(json.dumps({key: result[key] for key in ("status", "normalized_rows", "clock_segment_count", "plot_sampled_points", "warnings")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
