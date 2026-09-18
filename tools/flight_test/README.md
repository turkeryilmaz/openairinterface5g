# OAI flight logging

Run the normal gNB or nrUE executable from its usual build directory. Enable
logging with a top-level setting in the existing libconfig `.conf` file:

```conf
flight = "log";
```

```sh
sudo ./nr-softmodem -O flightTestTakeoff1.conf
sudo ./nr-uesoftmodem -O flightTestTakeoff1.conf
```

Alternatively, enable it from the command line:

```sh
sudo ./nr-softmodem -O flightTest.conf --flight log
sudo ./nr-uesoftmodem -O ../../../../Configs/2026-07-23_nrue_flight_tests.conf --flight log
```

CLI values override the corresponding configuration settings. `--flight off`
disables a configuration's `flight = "log"`. With no setting, flight logging is
off. The feature list accepts space-separated names (`flight = "log";` in the
file, `--flight log` on the terminal). Only `log` and `off` exist today: names
such as `recovery` and `agc` fail explicitly before starting the radio. They are
reserved for future implementations, not placeholders that silently do nothing.

No systemd unit or separate capture command is required. The executable reads
these options before background processes, logging threads, or radio setup. It
starts the bundled Python collector, which launches one OAI worker and enables
the C recorder. Python 3 and the checkout's `tools/flight_test/` directory must
be present; no Python packages need to be installed. This is independent of
USRP model and service manager. Normal OAI platform requirements still apply.

Relative `-O` paths and relative files referenced inside the configuration keep
the same meaning because the worker retains the original launch directory.
Legacy OAI current-directory statistics therefore still use that directory.
Additional flight capture files use the directory below. The collector mirrors
redacted output to the terminal through a bounded asynchronous queue; a slow
terminal can lose console copies without blocking disk capture. The final
status records console loss separately.

## Output and optional settings

With no overrides, each invocation creates:

```text
<checkout>/cmake_targets/log/FlightTests/YYYY-MM-DD/
  ue-<UTC-start>-<pid>-<unique-id>/
    metadata.json
    stdout.0000.log
    stderr.0000.log
    host.0000.log
    recorder/oai-flight-recorder-*.ndjson
    status.json
```

The gNB uses a `gnb-` prefix. `YYYY-MM-DD` is the host's local date at launch.
Event timestamps retain UTC epoch nanoseconds and monotonic nanoseconds;
changing directory naming does not synchronize clocks. Linked Git worktrees
use the main checkout's `cmake_targets/log/FlightTests` root when discoverable.
Separate invocations never reuse a run directory. Startup prints the actual
capture path. Directories are private (0700) and files are private (0600).

All settings below are optional, work for both roles, and can be supplied as
strings in the `.conf` file or as their matching CLI option:

| Config key / CLI option | Default | Purpose |
| --- | --- | --- |
| `flight-output` / `--flight-output` | date directory above | Explicit capture parent directory; relative to the launch directory if relative. |
| `flight-repo` / `--flight-repo` | discovered checkout | Relocated checkout containing the bundled collector and source metadata. |
| `flight-interface` / `--flight-interface` | `oaitun_ue1` | Optional interface observation. |
| `flight-core-ip` / `--flight-core-ip` | absent | Optional remote endpoint for host observations; not required to record OAI. |
| `flight-gpsd` / `--flight-gpsd` | absent | Optional GPSD `HOST:PORT` observations. |
| `flight-recorder-budget` / `--flight-recorder-budget` | `0` | Numeric recording byte cap; `0` means no total cap. Positive values: 8192..INT64_MAX. |
| `flight-stdout-budget` / `--flight-stdout-budget` | `0` | Combined stdout/stderr byte cap; `0` means no total cap. |
| `flight-host-budget` / `--flight-host-budget` | `0` | Host observations byte cap; `0` means no total cap. |
| `flight-min-free-bytes` / `--flight-min-free-bytes` | `536870912` (512 MiB) | Free-space reserve; `0` explicitly disables this guard. |

For example, prepare different configuration files before the take-offs:

```conf
flight = "log";
# Optional overrides; omit these to use the defaults.
flight-output = "/data/FlightTests/takeoff1";
flight-min-free-bytes = "1073741824";
```

Or override only the directory on the terminal:

```sh
sudo ./nr-uesoftmodem -O flightTest.conf --flight log --flight-output /data/FlightTests/takeoff1
```

The default retains successive files for the whole run while space is
available. Numeric files are at most 16 MiB, stdout/stderr files 8 MiB, and
host files 4 MiB. Older files are never overwritten. RAM rings and producer
sampling stay bounded independently of the amount stored on disk. The former
128 MiB maximum has been removed.

The free-space reserve is a configurable operational default, not a guarantee
of sufficient storage for a flight. Checks run in the asynchronous collectors
(once per second and on file rotation for the C recorder; cached for up to
0.5 seconds for text writers). Other processes can consume space concurrently;
this is not an exclusive filesystem reservation. Reaching a selected cap,
reserve, or write error reports incomplete capture and keeps OAI running.
Metadata/status remain separately limited to 512 KiB each. Retain and inspect
the final status and recorder footer when assessing completeness.

## What is collected

- Numeric source events for UE synchronization, AGC observations, measurements,
  random access, RRC, PDU establishment, and timing advance; gNB link/traffic,
  HARQ and random access; sampled USRP RX/TX results.
- Redacted stdout/stderr and bounded-buffer host/interface/clock observations.
- Binary, configuration, source and adjacent runtime-plugin fingerprints.
- Exit status, intentional signals, recording failures, and loss counters.

No core address or GPS daemon is required. Missing optional host facilities
are recorded as unavailable. Neither interface observations nor an ICMP result
prove application connectivity. This version never restarts the worker.
Ctrl+C and SIGTERM are forwarded to its process group; after the existing
10-second grace period, a stuck owned group is killed. Exit status is retained.

No raw configuration, environment, Git diff, packet payload, camera video,
application frame counters, or glass-to-glass latency is collected. Console
redaction suppresses recognized secret/SIM/authentication blocks, but cannot
classify every possible protocol dump: review private captures before export.
The recorder does not yet consume UHD asynchronous TX metadata, so it cannot
provide an authoritative TX-late/underflow counter.

## Extending the logging

The producer API and numeric field definitions are in
`common/utils/LOG/flight_recorder.h` and [EVENT_SCHEMA.md](EVENT_SCHEMA.md).
Add a stable event ID, a bounded call-site emission, documented units and field
meanings, and the decoder name. Never reuse an existing ID or change its field
meaning silently. An incompatible record-layout change requires a new schema
version and decoder support. Unknown event IDs in a recognized schema retain
their raw numeric fields as `UNKNOWN`, allowing later analysis without data
loss. Old captures remain readable by the current decoder.

Keep allocation, formatting, filesystem access, subprocesses and blocking work
out of radio producers. Add host observations to the asynchronous collector.
Register future `flight` features explicitly in the startup parser and provide
their implementation and tests before enabling them. The current feature
selection does not implement recovery or change AGC behavior.

Decode after stopping a run:

```sh
python3 tools/flight_test/decode_events.py <run>/recorder --output <run>/decoded
```

The optional `mark_event.py --run <run> --flight flight01 --event takeoff` adds
an operator marker. It does not detect take-off automatically.

## Software checks and legacy collector entry point

Build OAI normally through CMake, including the required configuration and radio
plugins adjacent to the executable. With `ENABLE_TESTS=ON`, build
`flight_startup_fixture` and `flight_recorder_test`, then run:

```sh
ctest --test-dir <build> -R '^flight_(startup|recorder)$' --output-on-failure
python3 -B tools/flight_test/tests/test_capture.py
python3 -B tools/flight_test/tests/test_decode_events.py
```

The startup fixture uses the real config plugin, bootstrap and recorder but
contains no radio code. It checks config/CLI precedence, relative files,
default date directories, unique runs, unsupported features, and INT/TERM.
The recorder tests include sequential retention beyond eight files, byte caps,
free-space failure, lifecycle races and producer limits.

`capture.py` remains callable directly for existing automation. That entry
point requires an absolute executable and absolute configuration paths unless
`--working-directory` is supplied. It defaults to a private per-run working
directory; the integrated softmodem path supplies the original launch
directory automatically. The old systemd example is optional legacy material,
not an installation requirement or the flight activation mechanism.
