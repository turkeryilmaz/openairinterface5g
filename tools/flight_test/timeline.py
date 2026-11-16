#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Offline flight timelines: full selected CSV/JSONL and bounded standalone plots.

Example: python3 timeline.py --gnb GNB_SESSION --ue UE_SESSION --output NEWDIR
Inputs are attempts or recovery sessions containing attempts/. Input files are
read only. NEWDIR must not exist and must be outside every input. Capture's
automatic report uses a narrowly scoped opt-in for its own report/ child.
With both roles, --output GNB_CAPTURE/report publishes report.html and combined/
exports beside the preserved local gNB report, without replacing existing files.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import shutil
import sqlite3
from collections import Counter
from pathlib import Path
from string import Template
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
    "Radio/protocol events use source event clocks; host, recovery, traffic and core records use collector observations.",
    "Monotonic times compare only within a proven host boot. Unknown boots use separate attempt domains.",
    "Host UTC alignment is unverified. Shared experiment/run IDs and chrony tracking are not clock calibration.",
    "Clock segments split when consecutive wall-minus-monotonic offsets change beyond the selected threshold.",
    "No one-way latency, cross-host RACH pairing or SFN-only links are inferred. SFN wraps every 10.24 seconds.",
    "Core source timestamps remain original fields; collector receipt UTC is plotted. Packet payloads are not read.",
    "Recovery durations use collector receipts within one attempt, proven host boot and clock segment only.",
    "Positive local receiver intervals show observed delivery; idle time and absent reports are not failures.",
    "PDU acceptance and active DRB are separate from delivery; interface/TUN presence does not prove a working bearer.",
    "Gain readbacks are device settings, not calibrated RF power. Different units have separate plots.",
    "Decoding retains typed records and event-sequence ownership in memory; display limits do not bound decoding memory.",
]

# Mirror the exact PRB/channel-bandwidth pairs in nr_common.c:tables_5_3_2
# and get_supported_bw_mhz(). Do not infer a channel width by rounding a grid.
FR1_PRBS = {
    0: (25, 52, 79, 106, 133, 160, 188, 216, 242, 270),
    1: (11, 24, 38, 51, 65, 78, 92, 106, 119, 133, 162, 189, 217, 245, 273),
    2: (None, 11, 18, 24, 31, 38, 44, 51, 58, 65, 79, 93, 107, 121, 135),
}
FR1_MHZ = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100)
MCS_DISPLAY_LIMIT = 10000


def arfcn_mhz(value: int | None) -> float | None:
    """Global NR-ARFCN raster, matching nr_common.c:from_nrarfcn()."""
    if value is None or not 0 <= value <= 3279165:
        return None
    if value < 600000:
        return value * .005
    if value < 2016667:
        return 3000 + (value - 600000) * .015
    return 24250.08 + (value - 2016667) * .060


def recorded_configuration(directory: Path, identity: dict, role: str) -> dict:
    """Extract only saved scalar settings and bounded startup-log observations.

    Conflicting values remain unavailable. Includes are not expanded and current
    bench files are never consulted. Command-line scalar overrides take priority.
    """
    text = identity.get("config_snapshot", {}).get("text", "")
    if not isinstance(text, str):
        text = ""
    # libconfig strings and comments may contain example assignments. Strip
    # both before scalar matching; saved snapshots are evidence, not executed.
    text = re.sub(r'"(?:\\.|[^"\\])*"|/\*.*?\*/|//[^\n]*|\#[^\n]*', " ", text, flags=re.S)
    command = identity.get("command", [])
    command = command if isinstance(command, list) else []
    sources = [str(directory / "metadata.json") + " config_snapshot.text / command"]

    def setting(name: str) -> int | None:
        values = {int(value) for value in re.findall(r"\b" + re.escape(name) + r"\s*=\s*(-?\d+)\s*;", text)}
        overrides = {}
        for index, argument in enumerate(command):
            if not isinstance(argument, str) or not argument.startswith("--"):
                continue
            option, equals, value = argument.partition("=")
            if option.rsplit(".", 1)[-1].removeprefix("--") != name:
                continue
            value = value if equals else command[index + 1] if index + 1 < len(command) else None
            if isinstance(value, str) and re.fullmatch(r"-?\d+", value):
                overrides[option] = int(value)
            else:
                return None
        if overrides:
            distinct = set(overrides.values())
            if len(distinct) != 1 or len(values) > 1:
                return None
            return next(iter(distinct))
        return next(iter(values)) if len(values) == 1 else None

    observed: dict[str, set] = {}

    def observe(key: str, value: Any, source: str) -> None:
        observed.setdefault(key, set()).add(value)
        if source not in sources and len(sources) < 32:
            sources.append(source)

    # Startup settings are normally near the beginning. A truncated scan is
    # recorded; no unbounded stdout read or report-directory recursion occurs.
    remaining, truncated = 1024 * 1024, False
    for prefix in ("stdout", "stderr"):
        for path in sorted(directory.glob(prefix + ".*.log"), key=rotation_key):
            if not path.is_file():
                continue
            with path.open("rb") as stream:
                line_number = 0
                while True:
                    line_number += 1
                    line = stream.readline(min(65537, remaining + 1))
                    if not line:
                        break
                    remaining -= len(line)
                    if remaining < 0:
                        truncated = True
                        break
                    value = line.decode("utf-8", errors="replace")
                    source = f"{path}:{line_number}"
                    match = re.search(r"DL frequency (\d+) Hz, UL frequency (\d+) Hz", value)
                    if match:
                        observe("dl_center_mhz", int(match[1]) / 1e6, source)
                        observe("ul_center_mhz", int(match[2]) / 1e6, source)
                    match = re.search(r"duplex mode (FDD|TDD)\b", value)
                    if match:
                        observe("duplex", match[1], source)
                    match = re.search(r"nb_tx_streams (\d+), nb_rx_streams (\d+)", value)
                    if match and role == "gnb":
                        observe("gnb_rf", f"TX: {match[1]} / RX: {match[2]}", source)
                    match = re.search(r"nb_rx_antenna (\d+), nb_tx_antenna (\d+)", value)
                    if match and role == "ue":
                        observe("ue_rf", f"TX: {match[2]} / RX: {match[1]}", source)
                    match = re.search(r"pdsch_AntennaPorts N1 (\d+) N2 (\d+) XP (\d+) pusch_AntennaPorts (\d+)", value)
                    if match and role == "gnb":
                        observe("gnb_ports", f"PDSCH: {int(match[1]) * int(match[2]) * int(match[3])} / PUSCH: {match[4]}", source)
            if truncated:
                break
        if truncated:
            break
    values = {key: next(iter(items)) if len(items) == 1 else None for key, items in observed.items()}
    for key in ("dl_frequencyBand", "ul_frequencyBand", "dl_carrierBandwidth", "ul_carrierBandwidth",
                "dl_subcarrierSpacing", "ul_subcarrierSpacing", "dl_harq_round_max", "ul_harq_round_max"):
        values[key] = setting(key)
    band = values.get("dl_frequencyBand")
    # Classification of recorded operating bands from nr_common.c:nr_bandtable.
    if "duplex" not in observed:
        if band in {34, 38, 39, 40, 41, 46, 47, 48, 50, 51, 53, 77, 78, 79, 90, 96, 101, 102, 104, 257, 258, 260, 261}:
            values["duplex"] = "TDD"
        elif band in {1, 2, 3, 5, 7, 8, 12, 13, 14, 18, 20, 24, 25, 26, 28, 30, 65, 66, 70, 71, 74, 85,
                      91, 92, 93, 94, 100, 254, 255, 256, 510, 511, 512}:
            values["duplex"] = "FDD"
    for key, name in (("dl_point_a_mhz", "dl_absoluteFrequencyPointA"),
                      ("ul_point_a_mhz", "ul_absoluteFrequencyPointA"), ("ssb_reference_mhz", "absoluteFrequencySSB")):
        values[key] = arfcn_mhz(setting(name))
    # The startup get_frame_type() line records the actual band duplex type.
    # Equal/separate tuned frequencies alone do not establish the duplex mode.
    if values.get("duplex") == "FDD":
        values["duplex_timing"] = "FDD / No TDD Slot Pattern"
    elif values.get("duplex") == "TDD":
        fields = ("dl_UL_TransmissionPeriodicity", "nrofDownlinkSlots", "nrofDownlinkSymbols",
                  "nrofUplinkSlots", "nrofUplinkSymbols")
        pattern = [setting(name) for name in fields]
        if all(value is not None for value in pattern):
            values["duplex_timing"] = (f"Period Code: {pattern[0]}; DL: {pattern[1]} Slots + {pattern[2]} Symbols; "
                                       f"UL: {pattern[3]} Slots + {pattern[4]} Symbols")
    return {"values": values, "sources": sources, "startup_scan_truncated": truncated,
            "scope": "Saved configuration may omit includes; startup values and CLI overrides are used where recorded."}


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


def native_fields(event: dict, joined: dict | None = None) -> tuple[dict, list]:
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
    if joined:
        typed.update(joined)
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
    fields.update({key: typed[key] for key in ("frame", "slot", "rnti", "sync_epoch", "prach_token", "preamble_index", "channel_code",
                                              "frame_slot", "absolute_tx_slot", "absolute_rx_slot") if key in typed})
    if type(typed.get("frame_slot")) is int:
        frame_slot(typed["frame_slot"])
    if kind in (13, 16):
        fields["rnti"] = integer(event["c"])
    elif kind in (21, 23):
        fields["rnti"] = integer(event["b"])
    elif kind in (22, 24, 25, 26):
        fields["rnti"] = integer(event["a"])
    if kind == 21:
        fields["cell"] = integer(event["a"])
    if kind in (129, 130):
        identity = event["a"]
        fields.update(rnti=identity & 0xffff, cell=(identity >> 16) & 0xffff,
                      module=(identity >> 32) & 0xffff, context_generation=event["b"])
        if kind == 130:
            fields.update(frame=integer(event["c"]), dl_harq_max_attempts=integer(event["d"]),
                          ul_harq_max_attempts=integer(event["e"]))
    elif kind in (131, 132):
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
        fields["agc_acquisition"] = dict(ue=event["a"], requested_delta_db=event["b"],
            returned_delta_db=event["c"], configured_gain_minus_offset_db=event["d"], return_code=event["e"])
        add("UE acquisition gain adjustment result", kind, "event code")
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
    elif kind == 129:
        for key, direction in (("c", "DL"), ("d", "UL")):
            value = integer(event[key])
            if value is not None and 0 <= value <= 1000000:
                add(f"{direction} scheduler BLER", value / 10000, "BLER %", entity)
        fields.update(dl_last_num_sched=integer(event["e"]), ul_last_num_sched=integer(event["f"]))
    elif kind in (131, 132):
        direction = "DL" if kind == 131 else "UL"
        fields[direction.lower() + "_harq_max_attempts"] = integer(event["f"])
        for index, key in enumerate("bcde", 4):
            add(f"gNB {direction} HARQ round {index}", integer(event[key]), "count", entity)
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
    if kind == 45 and (typed.get("would_change") or typed.get("submitted")):
        # The producer also repeats hold observations once per second. Keep
        # these out of the action timeline; a proposal is not an actuation.
        action = "request submitted" if typed.get("submitted") else "proposal not submitted"
        add("RX AGC " + action, kind, "event code", typed.get("source", ""))
    elif kind == 40 and typed.get("operation") in ("SET_RX", "SET_TX") and typed.get("request_id") != 0:
        operation = "RX" if typed["operation"] == "SET_RX" else "TX"
        qualifier = " (Request Metadata Unavailable)" if typed.get("request_id") is None else ""
        add(operation + " device gain result " + typed.get("status", "unknown") + qualifier, kind, "event code")
    elif kind in (49, 58) and type(typed.get("applied")) is bool:
        applied = "result inconsistent" if typed["applied"] and typed.get("status") != "OK" else "applied" if typed["applied"] else "not applied"
        add("TX digital power " + applied, kind, "event code", typed.get("channel", ""))
    elif kind == 51:
        add("TX digital power request rejected", kind, "event code", typed.get("channel", ""))
    if kind in (80, 84, 87, 88, 90, 91, 119):
        add(decoder.NAMES[kind], kind, "event code", entity)
    if kind == 111:
        add("gNB PUSCH CRC failed", int(typed["crc_failed"]), "boolean", entity)
        add("gNB PUSCH DTX", int(typed["dtx"]), "boolean", entity)
    return fields, metrics


RECOVERY_EVENTS = {
    "recovery_stop_requested": "Recovery stop request result",
    "recovery_session_started": "Recovery session started",
    "recovery_attempt_started": "Recovery attempt prepared",
    "recovery_attempt_worker_started": "Recovery worker launched",
    "recovery_attempt_finished": "Recovery worker finished",
    "recovery_backoff": "Recovery backoff started",
    "recovery_group_fence": "Recovery group signal requested",
    "recovery_session_finished": "Recovery session finished",
    "operator_stop_forwarded": "Operator stop forwarding result",
}


def recovery_metric(row: dict) -> str | None:
    name = row["kind"]
    if name != "recovery_policy_transition":
        return RECOVERY_EVENTS.get(name)
    decision = row.get("decision")
    if isinstance(decision, dict) and decision.get("action") in ("observe", "restart", "retry", "stop"):
        return "Recovery decision " + decision["action"]
    change = row.get("change")
    if isinstance(change, dict) and isinstance(change.get("event"), str):
        return "Recovery observation " + change["event"][:96]
    return None


def point_details(fields: dict) -> dict:
    """Bound display details; full source fields remain in CSV/JSONL exports."""
    def scalars(values: dict, keys: tuple) -> dict:
        return {key: value[:512] if isinstance(value, str) else value
                for key in keys if (value := values.get(key)) is not None and type(value) in (str, int, float, bool)}

    details = scalars(fields, ("kind", "policy_phase", "policy_generation", "ordinal", "attempt_directory", "worker_pid",
        "process_group", "state", "signal", "requested_signal", "delivered_signal", "reason", "exit_code", "child_returncode",
        "recovery_stop_reason", "group_fence_state", "remaining_ns", "planned_launch_monotonic_ns", "actual_launch_monotonic_ns",
        "enabled", "delivered", "outcome", "error", "errno"))
    for key, keys in {
        "decision": ("action", "reason", "not_before_ns"), "policy_decision": ("action", "reason", "not_before_ns"),
        "change": ("event", "reason", "previous_reason", "candidate_action", "qualified", "count", "cause", "policy",
                   "wait_seconds", "device_id", "observed_duration_ns", "persistence_required_ns", "application_delivery_verified"),
        "agc_acquisition": ("ue", "requested_delta_db", "returned_delta_db", "configured_gain_minus_offset_db", "return_code"),
        "typed": ("source", "generation", "phase", "reason", "would_change", "submitted", "candidate_gain_db",
                  "reported_rx_gain_db", "input_join_status", "mean_level_dbfs", "peak_component_dbfs", "error_db",
                  "radio", "request_id", "operation", "status", "status_code", "requested_gain_db", "rx_readback_db",
                  "tx_readback_db", "rx_gain_valid", "tx_gain_valid", "time_metadata_joined", "channel", "applied",
                  "actuation_requested", "requested_power_dbm", "estimated_output_power_dbm", "amplitude_coefficient_q30",
                  "selected_nominal", "requested_digital_dbfs", "realized_digital_dbfs"),
    }.items():
        if isinstance(fields.get(key), dict):
            values = scalars(fields[key], keys)
            if values:
                details[key] = values
    return details


class Report:
    def __init__(self, output: Path, clock_step_ms: float, max_plot_points: int, max_plot_series: int,
                 min_free_bytes: int = 512 * 1024 * 1024):
        self.output = output
        self.min_free_bytes = min_free_bytes
        self.added = 0
        self.check_space()
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

    def check_space(self, required_bytes: int = 0) -> None:
        if shutil.disk_usage(self.output).free < self.min_free_bytes + required_bytes:
            raise OSError("timeline free-space reserve reached")

    def write_text(self, name: str, content: str) -> None:
        payload = content.encode("utf-8")
        self.check_space(len(payload))
        (self.output / name).write_bytes(payload)

    def add(self, context: dict, *, name: str, kind: int | None, origin: str, category: str,
            mono: int | None, utc: int | None, source: str, line: int, fields: dict, metrics: list,
            boot: str | None = None, ring: int | None = None, sequence: int | None = None) -> None:
        if self.added % 1024 == 0:
            self.check_space()
        self.added += 1
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
            info["recorded_configuration"] = recorded_configuration(attempt, identity, role)
            recorder = attempt / "recorder"
            if recorder.is_dir() and any(recorder.glob("oai-flight-recorder-*.ndjson")):
                decoded = self.output / "decoded" / capture / f"{ordinal:06d}"
                status = decoder.decode(recorder, decoded, check_space=self.check_space)
                info["decoder"] = {"directory": str(decoded.relative_to(self.output)),
                    **{key: status.get(key) for key in ("files", "events", "invalid_lines", "unsupported_schema",
                                                       "clean_footer_observed", "footers", "sampling_config")}}
                # Reuse production decoder joins by exact source location. Keep
                # this lookup on disk rather than accumulating every gain action.
                self.db.execute("CREATE TEMP TABLE native_details(source TEXT,line INTEGER,fields TEXT,PRIMARY KEY(source,line))")
                for filename in ("radio_gain.csv", "radio_rx_decisions.csv"):
                    with (decoded / filename).open(newline="") as detail_source:
                        for index, detail in enumerate(csv.DictReader(detail_source)):
                            if index % 1024 == 0:
                                self.check_space()
                            parsed = {}
                            for key, value in detail.items():
                                if value == "":
                                    parsed[key] = None
                                elif value in ("True", "False"):
                                    parsed[key] = value == "True"
                                elif re.fullmatch(r"-?\d+", value):
                                    parsed[key] = int(value)
                                else:
                                    try:
                                        parsed[key] = float(value)
                                    except ValueError:
                                        parsed[key] = value
                            self.db.execute("INSERT INTO native_details VALUES(?,?,?)",
                                (detail["source_file"], int(detail["source_line"]), packed(parsed)))
                with (decoded / "events.csv").open(newline="") as source_csv:
                    batches = {}
                    harq_limits = {"dl": set(), "ul": set()}
                    for row in csv.DictReader(source_csv):
                        event = {key: int(row[key]) for key in decoder.FIELDS[3:]}
                        if event["event"] not in SELECTED:
                            continue
                        event.update(source_file=row["source_file"], source_line=int(row["source_line"]))
                        detail = self.db.execute("SELECT fields FROM native_details WHERE source=? AND line=?",
                                                 (row["source_file"], int(row["source_line"]))).fetchone() if event["event"] in (40, 45) else None
                        fields, metrics = native_fields(event, json.loads(detail[0]) if detail else None)
                        ring = event["ring"]
                        if event["event"] == 130 and 0 <= ring < 64:
                            batches[ring] = dict(fields, last_sequence=event["sequence"], last_mono=event["mono_ns"])
                        elif event["event"] in (21, 22, 24, 25, 26, 131, 132):
                            batch = batches.get(ring)
                            # The scheduler emits one ordered batch under its existing lock.
                            # A stale/mismatched batch must not identify a reused RNTI.
                            if (batch and batch.get("rnti") == fields.get("rnti")
                                    and fields.get("cell", batch.get("cell")) == batch.get("cell")
                                    and event["sequence"] > batch["last_sequence"]
                                    and 0 <= event["mono_ns"] - batch["last_mono"] <= 100000000):
                                fields.update({key: batch[key] for key in ("module", "cell", "context_generation",
                                               "dl_harq_max_attempts", "ul_harq_max_attempts") if key in batch})
                            else:
                                fields["context_identity"] = "unverified; lifetime generation not recorded"
                        if fields.get("context_generation") is not None:
                            entity = (f"RNTI {fields['rnti']} / Cell {fields['cell']} / Module {fields['module']} / "
                                      f"Context {fields['context_generation']}")
                            metrics = [[name, value, unit, entity] for name, value, unit, _ in metrics]
                        if event["event"] == 130:
                            for direction in ("dl", "ul"):
                                key = direction + "_harq_round_max"
                                actual = fields.get(direction + "_harq_max_attempts")
                                if actual is not None:
                                    harq_limits[direction].add(actual)
                                    info["recorded_configuration"]["values"][key] = (
                                        actual if len(harq_limits[direction]) == 1 else None)
                                    info["recorded_configuration"].setdefault("observed_harq_limits", {})[direction] = sorted(harq_limits[direction])
                            source_ref = str(recorder / row["source_file"]) + ":" + row["source_line"] + " GNB_UE_CONTEXT"
                            refs = info["recorded_configuration"]["sources"]
                            if source_ref not in refs and len(refs) < 32:
                                refs.append(source_ref)
                        if "typed_decode_error" in fields:
                            self.warnings["invalid_native_typed_payloads"] += 1
                        self.add(context, name=decoder.NAMES.get(event["event"], "UNKNOWN"), kind=event["event"],
                                 origin="source event", category="native", mono=integer(event["mono_ns"]),
                                 utc=integer(event["realtime_ns"]), source=str(recorder / row["source_file"]),
                                 line=int(row["source_line"]), fields=fields, metrics=metrics,
                                 ring=event["ring"], sequence=event["sequence"])
                self.db.execute("DROP TABLE native_details")
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
            if prefix == "recovery":
                metric = recovery_metric(row)
                if metric:
                    metrics.append([metric, 1, "event code", ""])
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
        self.check_space()
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
        self.check_space()
        event_count = recovery_count = 0
        state = {}
        self.db.execute("CREATE TABLE rate_state(identity TEXT PRIMARY KEY,mono INTEGER,dl INTEGER,ul INTEGER,"
                        "generation INTEGER,epoch INTEGER,source TEXT,line INTEGER)")
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
                if event_count % 1024 == 0:
                    self.check_space()
                (event_id, capture, attempt, role, host, boot, domain, mono, utc, segment, name, kind, origin,
                 category, source, line, ring, sequence, fields_json, metrics_json) = raw
                fields, metrics = json.loads(fields_json), json.loads(metrics_json)
                if kind == 21 and mono is not None and fields.get("rnti") is not None:
                    rate_key = packed([capture, attempt, domain, segment, fields.get("module"),
                                       fields.get("cell"), fields["rnti"]])
                    previous = self.db.execute("SELECT mono,dl,ul,generation,epoch,source,line FROM rate_state WHERE identity=?",
                                               (rate_key,)).fetchone()
                    counters = {"dl": integer(fields["raw"].get("d")), "ul": integer(fields["raw"].get("e"))}
                    generation, epoch = fields.get("context_generation"), previous[4] if previous else 0
                    if previous:
                        seconds = (mono - previous[0]) / 1e9
                        reset = any(value is not None and prior is not None and value < prior
                                    for value, prior in zip(counters.values(), previous[1:3]))
                        same_context = generation == previous[3]
                        if not same_context or reset or not 0 < seconds <= 2:
                            epoch += 1
                            self.warnings["mac_rate_intervals_excluded_context_reset_or_gap"] += 1
                        else:
                            entity = next((item[3] for item in metrics), f"RNTI {fields['rnti']}")
                            for (direction, value), prior in zip(counters.items(), previous[1:3]):
                                if value is not None and prior is not None and value >= prior:
                                    label = "DL MAC SDU rate (submitted)" if direction == "dl" else "UL MAC SDU rate (received)"
                                    metrics.append([label, 8 * (value - prior) / seconds, "bit/s", entity])
                            fields["mac_rate_interval"] = {"seconds": seconds, "previous_source": previous[5],
                                                           "previous_line": previous[6], "context_verified": generation is not None}
                    fields["counter_context_epoch"] = epoch
                    self.db.execute("INSERT OR REPLACE INTO rate_state VALUES(?,?,?,?,?,?,?,?)",
                                    (rate_key, mono, counters["dl"], counters["ul"], generation, epoch, source, line))
                    # Retain derived rates and context in the ordered plot source.
                    self.db.execute("UPDATE events SET fields=?,metrics=? WHERE id=?", (packed(fields), packed(metrics), event_id))
                row = dict(zip(COLUMNS[:12], (capture, attempt, role, host, boot, domain, segment, origin, mono, utc,
                                              "host UTC, alignment unverified", kind)))
                row.update(name=name, category=category, source_file=source, source_line=line, ring=ring, sequence=sequence,
                           **{key: fields.get(key) for key in ("frame", "slot", "rnti", "sync_epoch")}, fields=fields, metrics=metrics)
                json_dst.write(packed(row) + "\n")
                writer.writerow(dict(row, fields=packed(fields), metrics=packed(metrics)))
                event_count += 1
                for metric, value, unit, entity in metrics:
                    # MCS snapshots share a bounded direction series; RNTI/cell/
                    # lifetime remain in each point rather than being dropped by
                    # a per-series entity limit.
                    series = packed([capture, role, host, metric, "" if unit == "MCS index" else entity])
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
        self.db.execute("DROP TABLE rate_state")
        self.db.commit()
        return event_count, recovery_count

    def plots(self) -> tuple[list, int]:
        self.check_space()
        result = []
        total = self.db.execute("SELECT COUNT(*) FROM (SELECT series FROM points GROUP BY series)").fetchone()[0]
        minima = dict(self.db.execute("SELECT domain,MIN(mono) FROM events WHERE mono IS NOT NULL GROUP BY domain"))
        # Cover capture/metric families before spending the limit on additional
        # RNTIs. Within a family, show the most frequently observed entity first.
        groups = self.db.execute(
            "SELECT series,unit,n FROM (SELECT series,family,unit,COUNT(*) AS n,"
            "ROW_NUMBER() OVER (PARTITION BY family ORDER BY COUNT(*) DESC,series) AS entity_rank "
            "FROM points GROUP BY series,family,unit) ORDER BY unit!='MCS index',entity_rank,"
            "unit NOT IN ('event code','core observation'),family,series LIMIT ?",
            (self.max_plot_series,))
        for series, unit, count in groups:
            self.check_space()
            point_limit = MCS_DISPLAY_LIMIT if unit == "MCS index" else self.max_plot_points
            stride = max(1, math.ceil((count - 1) / (point_limit - 1)))
            points = []
            query = ("SELECT mono,utc,value,domain,segment,attempt,source,line,name,fields FROM (SELECT e.mono,e.utc,p.value,"
                     "e.domain,e.segment,e.attempt,e.source,e.line,e.name,e.fields,ROW_NUMBER() OVER (ORDER BY e.domain,e.mono,e.id) AS n "
                     "FROM points p JOIN events e ON e.id=p.event_id WHERE p.series=?) WHERE (n-1)%?=0 OR n=? ORDER BY n")
            for mono, utc, value, domain, segment, attempt, source, line, name, fields_json in self.db.execute(query, (series, stride, count)):
                fields = json.loads(fields_json)
                context = [domain, segment, attempt, fields.get("module"), fields.get("cell"), fields.get("rnti"),
                           fields.get("context_generation"), fields.get("counter_context_epoch")]
                provenance = str(Path(source).name) + ":" + str(line) + " " + name
                provenance += "; " + ", ".join(f"{key}={fields[key]}" for key in
                    ("rnti", "cell", "module", "context_generation", "sync_epoch", "frame", "slot", "preamble_index", "channel_code",
                     "dl_harq_max_attempts", "ul_harq_max_attempts") if key in fields)
                if "mac_rate_interval" in fields:
                    provenance += "; interval " + packed(fields["mac_rate_interval"])
                points.append([None if mono is None else (mono - minima[domain]) / 1e9,
                               None if utc is None else utc / 1e6, value, packed(context), provenance,
                               point_details(fields) if unit == "event code" else {}])
            nonzero = self.db.execute("SELECT COUNT(*) FROM points WHERE series=? AND value!=0", (series,)).fetchone()[0]
            if unit == "MCS index" and stride > 1:
                self.warnings["mcs_display_limit_reached_full_export_required"] += 1
            result.append(dict(label=" / ".join(str(v) for v in json.loads(series) if v), unit=unit,
                               identity=json.loads(series), source_points=count, plotted_points=len(points),
                               nonzero_source_points=nonzero, nonzero_plotted_points=sum(p[2] != 0 for p in points),
                               sampling="all recorded observations" if stride == 1 else "uniform index with first/last; omissions explicit",
                               points=points))
        families = sorted({packed(row["identity"][:4]) for row in result})
        condition = "family NOT IN (" + ",".join("?" for _ in families) + ")" if families else "1"
        self.omitted_families, self.omitted_family_count = [], 0
        with (self.output / "omitted_families.csv").open("w", newline="") as destination:
            writer = csv.writer(destination)
            writer.writerow(["capture", "role", "host", "metric", "unit", "source_series", "source_points", "nonzero_source_points"])
            query = ("SELECT family,unit,COUNT(DISTINCT series),COUNT(*),SUM(value!=0) FROM points WHERE "
                     + condition + " GROUP BY family,unit ORDER BY family,unit")
            for family, unit, series_count, points, nonzero in self.db.execute(query, families):
                if self.omitted_family_count % 1024 == 0:
                    self.check_space()
                identity = json.loads(family)
                writer.writerow(identity + [unit, series_count, points, nonzero])
                self.omitted_family_count += 1
                if len(self.omitted_families) < 256:
                    self.omitted_families.append(dict(identity=identity, unit=unit, source_series=series_count, source_points=points))
                if unit in ("event code", "core observation"):
                    self.warnings["event_families_omitted_by_series_limit"] += 1
        return result, max(0, total - len(result))

    def display_metadata(self) -> dict:
        anchors = {}
        for domain, minimum in self.db.execute("SELECT domain,MIN(mono) FROM events WHERE mono IS NOT NULL GROUP BY domain"):
            row = self.db.execute("SELECT mono,utc FROM events WHERE domain=? AND mono IS NOT NULL AND utc IS NOT NULL "
                                  "ORDER BY mono,id LIMIT 1", (domain,)).fetchone()
            anchors[domain] = {"utc_ms": (row[1] - row[0] + minimum) / 1e6 if row else None}
        utc = self.db.execute("SELECT MIN(utc) FROM events WHERE utc IS NOT NULL").fetchone()[0]
        mapped = [value["utc_ms"] for value in anchors.values() if value["utc_ms"] is not None]
        return {"domain_anchors": anchors, "recording_start_utc_ms": min(mapped) if mapped else utc / 1e6 if utc is not None else None,
                "scenario_start_utc_ms": None, "scenario_duration_seconds": None,
                "time_alignment": "Monotonic within each domain; initial cross-domain offsets use unverified host UTC.",
                "elapsed_anchor_policy": "First paired UTC/monotonic observation per domain; later wall-clock steps do not reanchor elapsed time.",
                "clock_step_domains": [row[0] for row in self.db.execute(
                    "SELECT domain FROM events GROUP BY domain HAVING COUNT(DISTINCT segment)>1 ORDER BY domain")],
                "mac_rate_policy": "Same attempt/domain/clock segment/cell/module/RNTI/context, nondecreasing counters, 0 < interval <= 2 s. "
                                   "Legacy captures without lifetime generations cannot prove same-RNTI continuity.",
                "mcs_display_limit_per_direction": MCS_DISPLAY_LIMIT,
                "unavailable_metrics": ["Calibrated RSSI", "RSRQ", "RI", "PMI"],
                "clock_mapping_unavailable_domains": [key for key, value in anchors.items() if value["utc_ms"] is None]}

    def close(self) -> None:
        self.db.close()



def configuration_cards(metadata: dict) -> str:
    configurations = [row["recorded_configuration"] for row in metadata.get("inputs", []) if "recorded_configuration" in row]

    def frequency(value: Any) -> str:
        return f"{value:.3f} MHz" if number(value) else "Unavailable"

    def display(values: dict) -> list[str]:
        def scalar(key: str) -> str:
            value = values.get(key)
            return str(value) if value is not None else "Unavailable"

        def bandwidth(direction: str) -> str:
            prbs, scs = (values.get(direction + key) for key in ("_carrierBandwidth", "_subcarrierSpacing"))
            if type(prbs) is not int or prbs <= 0:
                return "Unavailable"
            band = values.get(direction + "_frequencyBand")
            table = FR1_PRBS.get(scs, ()) if type(band) is int and band < 257 else ()
            width = FR1_MHZ[table.index(prbs)] if prbs in table else None
            return (f"{width} MHz / " if width is not None else "Channel Width Unavailable / ") + f"{prbs} PRBs"

        def spacing(direction: str) -> str:
            value = values.get(direction + "_subcarrierSpacing")
            return f"{15 * 2 ** value} kHz" if type(value) is int and 0 <= value <= 4 else "Unavailable"

        band = values.get("dl_frequencyBand")
        band_name = "n" + str(band) if band is not None else "Band Unavailable"
        return [band_name + " / " + scalar("duplex"), bandwidth("dl"), bandwidth("ul"),
                "DL: " + spacing("dl") + " / UL: " + spacing("ul"),
                frequency(values.get("dl_center_mhz")), frequency(values.get("ul_center_mhz")),
                frequency(values.get("ssb_reference_mhz")), frequency(values.get("dl_point_a_mhz")),
                frequency(values.get("ul_point_a_mhz")), scalar("gnb_ports"), scalar("gnb_rf"), scalar("ue_rf"),
                scalar("duplex_timing"), "DL: " + scalar("dl_harq_round_max") + " / UL: " + scalar("ul_harq_round_max")]

    labels = ["Frequency Band and Duplex", "DL Channel Bandwidth", "UL Channel Bandwidth", "Subcarrier Spacing",
              "DL Centre Frequency", "UL Centre Frequency", "SSB Reference Frequency", "DL Point A Frequency",
              "UL Point A Frequency", "gNB Antenna Ports", "Active gNB RF Channels", "Active UE RF Channels",
              "Duplex Timing", "HARQ Attempt Limit"]
    variants = [display(row["values"]) for row in configurations]
    cards = []
    for index, label in enumerate(labels):
        values = sorted({row[index] for row in variants if row[index] != "Unavailable"})
        # A UE capture without the gNB's config is absence, not a conflicting value.
        values = [value for value in values if value not in ("Band Unavailable / Unavailable",
                  "DL: Unavailable / UL: Unavailable")]
        # A partial observation adds no variant when its known parts already
        # occur in a complete observation from another input role.
        values = [value for value in values if "Unavailable" not in value or not any(
            "Unavailable" not in other and all(part.strip() in other.split(" / ")
                for part in value.split(" / ") if "Unavailable" not in part)
            for other in values)]
        value = "Unavailable" if not values else " / ".join(values) if len(values) <= 3 else "Varies Across Captures (See Sources)"
        cards.append("<div><dt>" + html.escape(label) + "</dt><dd>" + html.escape(value) + "</dd></div>")
    return "".join(cards)


def render_html(metadata: dict, plots: list, export_prefix: str = "") -> str:
    def safe_json(value: Any) -> str:
        # JSON text is embedded as inert data, with script delimiters escaped.
        return packed(value).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")

    template = Template(Path(__file__).with_name("timeline_report.html").read_text())
    javascript = Path(__file__).with_name("timeline_report.js").read_text()
    scopes = "".join("<li>" + html.escape(item) + "</li>" for item in SCOPES)
    configurations = [{key: row.get(key) for key in ("capture", "attempt", "role", "recorded_configuration")}
                      for row in metadata.get("inputs", []) if "recorded_configuration" in row]
    return template.substitute(data=safe_json(plots), display=safe_json(metadata.get("display", {})),
                               script=javascript, scopes=scopes, cards=configuration_cards(metadata),
                               configurations=html.escape(json.dumps(configurations, indent=2, ensure_ascii=True)),
                               facts=html.escape(json.dumps(metadata, indent=2, ensure_ascii=True)),
                               export_prefix=html.escape(export_prefix, quote=True),
                               omitted_families=metadata.get("plot_families_omitted_count", 0),
                               clock_warning=("Detected clock steps. Elapsed time retains each domain's first UTC/monotonic pair "
                                   "to preserve monotonic continuity; later cross-domain alignment remains unverified. "
                                   "Show UTC displays recorded wall-clock changes." if metadata.get("display", {}).get("clock_step_domains") else ""),
                               omitted=metadata.get("plot_series_omitted", 0))


def generate(inputs: list[tuple[str, Path]], output: Path, *, clock_step_ms: float = 5,
             max_plot_points: int = 2000, max_plot_series: int = 128, allow_nested_output: bool = False,
             min_free_bytes: int = 512 * 1024 * 1024, _combined: bool = False) -> dict:
    if not inputs:
        raise ValueError("provide at least one --gnb, --ue or --core capture")
    if not math.isfinite(clock_step_ms) or clock_step_ms <= 0 or not 2 <= max_plot_points <= 10000 or not 1 <= max_plot_series <= 256:
        raise ValueError("positive clock threshold, 2..10000 plot points and 1..256 plot series required")
    if type(min_free_bytes) is not int or min_free_bytes < 0:
        raise ValueError("free-space reserve must be a nonnegative integer number of bytes")
    roots = [(role, path.resolve(strict=True)) for role, path in inputs]
    if any(not path.is_dir() for _, path in roots):
        raise ValueError("capture input is not a directory")
    if len({path for _, path in roots}) != len(roots):
        raise ValueError("duplicate capture input")
    # Embedded auxiliaries belong to these local captures. Discover only the
    # fixed auxiliary/ child of each attempt; never recurse into generated reports.
    known = {path for _, path in roots}
    for role, root in list(roots):
        if role not in ("gnb", "ue"):
            continue
        attempts = root / "attempts"
        runs = sorted((path for path in attempts.iterdir() if path.is_dir()), key=lambda path: path.name) if attempts.is_dir() else [root]
        for run in runs:
            auxiliary = run / "auxiliary"
            if auxiliary.is_symlink() or not auxiliary.is_dir() or not any(auxiliary.glob("auxiliary-*.log")):
                continue
            auxiliary = auxiliary.resolve(strict=True)
            if auxiliary not in known:
                roots.append(("core", auxiliary))
                known.add(auxiliary)
    output = output.resolve()
    combined_root = next((path for role, path in roots if role == "gnb"), None)
    combined_allowed = (_combined and sum(role == "gnb" for role, _ in roots) == 1
                        and any(role == "ue" for role, _ in roots) and output == combined_root / "report" / "combined")
    if _combined and not combined_allowed:
        raise ValueError("combined output requires one selected gNB and at least one UE, under gNB/report/combined")
    for role, path in roots:
        nested_report = allow_nested_output and role in ("gnb", "ue") and output == path / "report"
        if (output == path or path in output.parents) and not (nested_report or combined_allowed and path == combined_root):
            raise ValueError("output must be outside every input capture (automatic reports allow only its direct report/ child)")
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    report = Report(output, clock_step_ms, max_plot_points, max_plot_series, min_free_bytes)
    try:
        for ordinal, (role, path) in enumerate(roots, 1):
            report.ingest(path, role, f"{ordinal:03d}-{role}-{path.name}")
        segments = report.segment_clocks()
        count, recovery_count = report.export()
        plots, omitted = report.plots()
        display = report.display_metadata()
        # Keep empty directions attached to the actual capture, including when
        # another gNB does have BLER or a source family was cut by the plot cap.
        for ordinal, (role, path) in enumerate(roots, 1):
            if role != "gnb":
                continue
            capture = f"{ordinal:03d}-gnb-{path.name}"
            hosts = {row["host"] for row in report.inputs if row["capture"] == capture and row.get("host")}
            host = next(iter(hosts)) if len(hosts) == 1 else ""
            available = {}
            for family, source_count, nonzero in report.db.execute(
                "SELECT p.family,COUNT(*),SUM(p.value!=0) FROM points p JOIN events e ON e.id=p.event_id "
                "WHERE e.capture=? AND p.unit='BLER %' GROUP BY p.family", (capture,)):
                metric = json.loads(family)[3]
                prior = available.get(metric, (0, 0))
                available[metric] = (prior[0] + source_count, prior[1] + nonzero)
            for direction in ("DL", "UL"):
                metric = direction + " scheduler BLER"
                if any(row["unit"] == "BLER %" and row["identity"][0] == capture and row["identity"][3] == metric for row in plots):
                    continue
                source_count, nonzero = available.get(metric, (0, 0))
                identity = [capture, "gnb", host, metric, ""]
                plots.append(dict(label=" / ".join(value for value in identity if value), identity=identity,
                                  unit="BLER %", source_points=source_count, plotted_points=0, points=[],
                                  nonzero_source_points=nonzero, nonzero_plotted_points=0,
                                  sampling="not displayed: series limit" if source_count else
                                           "unavailable: no timestamped scheduler BLER observations in this capture"))
        metadata = {"schema_version": 1, "kind": "offline_flight_timeline", "status": "complete",
                    "inputs": report.inputs, "selected_native_event_ids": sorted(SELECTED), "normalized_rows": count,
                    "recovery_duration_rows": recovery_count, "clock_segment_count": segments,
                    "clock_step_threshold_ms": clock_step_ms, "cross_host_alignment": "unverified",
                    "cross_host_uncertainty_ns": None, "clock_segment_policy": "adjacent wall-minus-monotonic offset change",
                    "chrony_policy": "tracking observations retained in events; no calibrated uncertainty bound",
                    "plot_family_selection": "MCS direction families first; then distinct families before repeated entities, events first within each rank",
                    "plot_families_omitted_count": report.omitted_family_count,
                    "plot_families_omitted": report.omitted_families,
                    "plot_families_omitted_list_truncated": report.omitted_family_count > len(report.omitted_families),
                    "plot_series": len(plots), "plot_series_omitted": omitted, "maximum_points_per_series": max_plot_points,
                    "mcs_maximum_points_per_direction": MCS_DISPLAY_LIMIT, "display": display,
                    "minimum_free_bytes": min_free_bytes,
                    "plot_source_points": sum(s["source_points"] for s in plots),
                    "plot_sampled_points": sum(s["plotted_points"] for s in plots),
                    "warnings": dict(report.warnings), "scope": SCOPES}
        report.write_text("metadata.json", json.dumps(metadata, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
        report.write_text("report.html", render_html(metadata, plots, "combined/" if _combined else ""))
    except Exception as error:
        (output / "INCOMPLETE.json").write_text(packed({"status": "incomplete", "error": str(error)}) + "\n")
        raise
    finally:
        report.close()
    (output / "ordering.sqlite3").unlink()
    return metadata


def generate_combined(inputs: list[tuple[str, Path]], output: Path, **options: Any) -> dict:
    """Publish a paired report beside the preserved local gNB report/exports."""
    gnbs = [path.resolve(strict=True) for role, path in inputs if role == "gnb"]
    output = output.absolute()
    if (len(gnbs) != 1 or not any(role == "ue" for role, _ in inputs) or output != gnbs[0] / "report"
            or output.is_symlink() or not output.is_dir()):
        raise ValueError("combined destination must be the existing selected gNB capture's nonsymlink report/ directory")
    child, destination = output / "combined", output / "report.html"
    if child.exists() or child.is_symlink() or destination.exists() or destination.is_symlink():
        raise ValueError("combined directory and combined report.html must not already exist")
    result = generate(inputs, child, _combined=True, **options)
    # Same-filesystem, exclusive publication adds no second HTML allocation and
    # never exposes a partially copied final report.
    destination.hardlink_to(child / "report.html")
    (child / "report.html").unlink()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gnb", type=Path, action="append", default=[], help="gNB attempt/session; repeatable")
    parser.add_argument("--ue", type=Path, action="append", default=[], help="UE attempt/session; repeatable")
    parser.add_argument("--core", type=Path, action="append", default=[], help="auxiliary JSONL directory; repeatable")
    parser.add_argument("--output", type=Path, required=True,
                        help="new external report directory, or selected GNB_CAPTURE/report for paired output")
    parser.add_argument("--clock-step-ms", type=float, default=5, help="offset jump threshold in milliseconds (default 5)")
    parser.add_argument("--max-plot-points", type=int, default=2000, help="sampled points/series (default 2000; full exports retained)")
    parser.add_argument("--max-plot-series", type=int, default=128, help="plotted series (default 128; full exports retained)")
    args = parser.parse_args()
    try:
        inputs = [("gnb", p) for p in args.gnb] + [("ue", p) for p in args.ue] + [("core", p) for p in args.core]
        combined = (len(args.gnb) == 1 and bool(args.ue)
                    and args.output.absolute() == args.gnb[0].resolve() / "report")
        result = (generate_combined if combined else generate)(inputs, args.output,
                          clock_step_ms=args.clock_step_ms, max_plot_points=args.max_plot_points,
                          max_plot_series=args.max_plot_series)
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(2, f"timeline: {error}\n")
    print(json.dumps({key: result[key] for key in ("status", "normalized_rows", "clock_segment_count", "plot_sampled_points", "warnings")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
