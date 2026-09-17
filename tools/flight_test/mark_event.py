#!/usr/bin/env python3
"""Append a small operator marker to an existing private capture run directory."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import time

EVENTS = ('takeoff', 'landing', 'leg_start', 'leg_end', 'turn', 'tracker_change',
          'polarization_change', 'coverage_loss', 'coverage_return', 'traffic_start', 'traffic_stop')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--flight', required=True)
    p.add_argument('--event', choices=EVENTS, required=True)
    args = p.parse_args()
    if not args.run.is_dir() or not re.fullmatch(r'[A-Za-z0-9_-]{1,32}', args.flight):
        p.error('run must exist; flight must be a short alphanumeric identifier')
    before = time.monotonic_ns()
    wall = time.time_ns()
    after = time.monotonic_ns()
    obj = dict(schema='oai.flight_operator', version=1, flight=args.flight, event=args.event,
               mono_before_ns=before, realtime_ns=wall, mono_after_ns=after,
               boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    data = (json.dumps(obj, separators=(',', ':')) + '\n').encode()
    fd = os.open(args.run / 'operator_events.ndjson', os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode) or os.fstat(fd).st_nlink != 1:
            raise RuntimeError('marker target must be a private regular file')
        fcntl.flock(fd, fcntl.LOCK_EX)
        if os.fstat(fd).st_size + len(data) > 1024 * 1024:
            raise RuntimeError('operator marker limit reached (1 MiB)')
        if os.write(fd, data) != len(data):
            raise OSError('incomplete marker write')
    finally:
        os.close(fd)
    print(json.dumps(obj))


if __name__ == '__main__':
    main()
