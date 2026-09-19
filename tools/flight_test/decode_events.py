#!/usr/bin/env python3
"""Decode only numeric OAI flight events, retaining malformed/unclean indicators."""
import argparse
import csv
import json
import re
from pathlib import Path

NAMES = dict(zip([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31],
                 ['UE_SYNC', 'UE_AGC', 'UE_MEASUREMENTS', 'UE_RA', 'UE_RRC', 'UE_PDU', 'UE_TA',
                  'UE_NAS', 'UE_RRC_TIMER', 'UE_CONTROL', 'GNB_SLOT', 'GNB_UE_BYTES', 'GNB_UE_RADIO', 'GNB_RA', 'GNB_UE_LINK',
                  'GNB_DL_HARQ', 'GNB_UL_HARQ', 'UE_NAS_COUNT', 'RADIO_RX', 'RADIO_TX']))
FIELDS = ['source_file', 'source_line', 'name', 'event', 'ring', 'sequence', 'mono_ns', 'realtime_ns',
          'a', 'b', 'c', 'd', 'e', 'f']


def decode(directory, output):
    paths = sorted(directory.glob('oai-flight-recorder-*.ndjson'))
    if not paths:
        raise ValueError('expected recorder files from one process')
    identities = []
    slots = set()
    for path in paths:
        match = re.fullmatch(r'oai-flight-recorder-([0-9]+-[0-9a-f]{16})-([0-9]+)\.ndjson', path.name)
        if match is None or match[2] in slots:
            raise ValueError('invalid recorder filename or duplicate file slot')
        identities.append(match[1])
        slots.add(match[2])
    if len(set(identities)) != 1:
        raise ValueError('mixed process/capture identities; decode each run separately')
    output.mkdir(parents=True, exist_ok=True)
    result = dict(files=len(paths), events=0, invalid_lines=0, unsupported_schema=0,
                  footers=[], health=[], retention=[], event_counts={})
    with (output / 'events.csv').open('w', newline='') as dst:
        writer = csv.DictWriter(dst, fieldnames=FIELDS)
        writer.writeheader()
        for path in paths:
            with path.open('rb') as source:
                number = 0
                while True:
                    line = source.readline(4097)
                    if not line:
                        break
                    number += 1
                    if len(line) > 4096 or not line.endswith(b'\n'):
                        result['invalid_lines'] += 1
                        while line and not line.endswith(b'\n'):
                            line = source.readline(4097)
                        continue
                    try:
                        obj = json.loads(line)
                        if not isinstance(obj, dict):
                            raise ValueError()
                        if (obj.get('schema') != 'oai.flight_recorder' or type(obj.get('version')) is not int
                                or obj['version'] != 1):
                            result['unsupported_schema'] += 1
                            continue
                        kind = obj.get('kind')
                        if kind == 'event':
                            values = {k: obj[k] for k in FIELDS[3:]}
                            if any(type(v) is not int for v in values.values()):
                                raise ValueError()
                            name = NAMES.get(obj['event'], 'UNKNOWN')
                            writer.writerow(dict(source_file=path.name, source_line=number, name=name, **values))
                            result['events'] += 1
                            result['event_counts'][name] = result['event_counts'].get(name, 0) + 1
                        elif kind in ('capture_footer', 'health', 'capture_health', 'file_begin'):
                            if kind == 'capture_footer':
                                required = ('clean', 'writer_errors', 'dropped_ring_full', 'dropped_no_slot',
                                            'no_slot_threads', 'invalid_timestamp_records', 'payload_truncated', 'rings_assigned')
                                if any(type(obj.get(k)) is not int or obj[k] < 0 for k in required):
                                    raise ValueError()
                                if obj['clean'] != 1 or obj['writer_errors'] != 0 or obj['rings_assigned'] > 64:
                                    raise ValueError()
                            numeric = {k: v for k, v in obj.items() if type(v) is int}
                            key = 'footers' if kind == 'capture_footer' else 'retention' if kind == 'file_begin' else 'health'
                            if len(result[key]) < 10000:
                                result[key].append(numeric)
                    except (ValueError, KeyError, TypeError, UnicodeDecodeError):
                        result['invalid_lines'] += 1
    result['clean_footer_observed'] = len(result['footers']) == 1
    result['ordering'] = 'file order only; compare per-boot monotonic times, never assume continuous UTC'
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recorder_dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = decode(args.recorder_dir, args.output)
    except (ValueError, OSError) as error:
        parser.exit(2, str(error) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
