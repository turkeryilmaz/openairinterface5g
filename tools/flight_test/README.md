# OAI flight logging and recovery instrumentation

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
file, `--flight log` on the terminal). `recovery` requires `log` and is UE-only. Unknown features such as
`agc` fail before starting the radio; existing AGC behavior is unchanged.

No systemd unit or separate capture command is required. The executable reads
these options before background processes, logging threads, or radio setup. It
starts the bundled Python collector, which owns the OAI worker and enables
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

## Automatic UE recovery

The opt-in session supervisor uses the same executable, with no separate
service or operator command. This is experimental flight recovery; hardware
validation results and remaining limitations belong to the campaign report.
Enable it with:

```conf
flight = "log recovery";
```

```sh
sudo ./nr-uesoftmodem -O flightTest.conf
# Or override the feature list on the command line:
sudo ./nr-uesoftmodem -O flightTest.conf --flight log recovery
```

The session supervisor keeps one worker at a time. Normal OAI cell search and RRC
recovery run first. A missing IP address after 30 seconds does not cause a
restart. The following defaults are flight experiment policy, not 3GPP timers:

| Condition | Action |
| --- | --- |
| Cell search continues with actual RX progress | Continue searching, without a 30-second deadline. |
| Fresh native monitor snapshots but RX samples stop increasing for 10 seconds, confirmed for another second | Request a worker restart. |
| PHY slot submissions keep advancing but DL or TX completions remain frozen for the stall interval, confirmed for another second | Request a worker restart. |
| A cell was acquired, or an established PDU path was lost, and 120 seconds pass without PDU acceptance or restoration of an earlier accepted session’s DRB context | Request a worker restart after any observed protocol hold expires. |
| Unexpected nonzero exit or signal after native monitoring started | Retry with 5, 15, 30, then at most 60 seconds of process backoff. |
| An accepted or restored DRB context and RX/slot/DL/TX progress remain continuously observed for 60 seconds | Reset process-backoff escalation. This does not clear a network wait or prove video delivery. |
| Exit zero following a supported registration rejection | Retry after the recorded protocol and process waits. |
| Authentication/permanent rejection, malformed or unsupported restriction, unclassified exit zero, or failure before native monitoring starts | Stop automatic retries; retain the reason. |
| Ctrl+C or SIGTERM to the launched command | Stop the session, including during backoff; never relaunch. |
| Missing ping, tunnel, traffic, text logs or native telemetry alone | Record the observation; do not infer a stuck radio. |

Change the two experiment thresholds using optional config strings
`flight-recovery-stall = "10";` and `flight-recovery-attempt = "120";`, or the
matching `--flight-recovery-stall` / `--flight-recovery-attempt` CLI options.
Values must be finite and at least one second. These do not override network
barring or backoff. A requested worker shutdown gets ten seconds of grace,
then its owned process group is killed and fenced before replacement.

The implemented registration-reject subset preserves decoded T3346 lower
bounds and T3502 values across worker attempts, uses a ten-second short-retry
lower bound, and applies the long wait after five eligible failures (immediately
for causes 95, 96, 97, 99 and 111). Cause 101 is included because it was observed
on this bench. Other causes and unknown optional IEs inhibit automatic retries.
This is a conservative fallback around OAI's existing terminal rejection path,
not a complete implementation of [TS 24.501 registration procedures](https://www.etsi.org/deliver/etsi_ts/124500_124599/124501/18.07.00_60/ts_124501v180700p.pdf).
Network selection, forbidden-area lists, all NAS timers and in-process retry
state/context ownership still belong in the native NAS/RRC implementation.

Each session retains numbered attempt directories, native progress, rejection
decisions, planned/actual launch times, backoff deadlines, signal/exit
classification and process-group cleanup outcomes. Eligible retries have no
lifetime attempt limit. The session status keeps only the latest 32 attempt
summaries plus total/omitted counts; individual attempt logs retain the history
subject to storage limits. Cleanup uncertainty stops replacement rather than
starting a second possible radio owner. Native snapshots use a private inherited socket and a normal-priority
C monitor thread independent of the recorder writer. Producers update bounded
lock-free counters. Sampling and socket delivery are best effort: missing data
is explicit and cannot prove the absence of a network restriction. Restrictions
persist within this supervisor session, not across a reboot or a newly launched
command. PDU acceptance is control-plane evidence; inspect traffic/application
measurements before calling an attempt successful. A resumed/configured DRB following an
earlier PDU acceptance cancels the acquisition deadline without claiming application
delivery. RRC_CONNECTED or fresh PHY samples alone do not cancel it. The native
DRB observation clears when the context is released, suspended, or reset.

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

With recovery enabled, the parent directory instead contains a
`ue-session-<UTC-start>-<pid>-<unique-id>/` directory with its own
`metadata.json`, `status.json`, `recovery.*.log` journal and
`attempts/000001/`, `attempts/000002/`, etc. Each attempt contains the usual
capture files. Session/attempt identifiers join the failure to its replacement;
separate files preserve crashes and forced-stop evidence. Stream and recorder
quotas apply per attempt, and the disk free-space reserve still applies.

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
prove application connectivity. Logging alone never restarts the worker.
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
selection keeps recovery opt-in and does not change AGC behavior.

Decode after stopping a run:

```sh
python3 tools/flight_test/decode_events.py <run>/recorder --output <run>/decoded
```

The optional `mark_event.py --run <run> --flight flight01 --event takeoff` adds
an operator marker. It does not detect take-off automatically.

## Software checks and legacy collector entry point

Build OAI normally through CMake, including the required configuration and radio
plugins adjacent to the executable. With `ENABLE_TESTS=ON`, build
`flight_startup_fixture`, `flight_recorder_test`, `flight_monitor_test` and
`nas_lib_test`, then run:

```sh
ctest --test-dir <build> -R '^(flight_(startup|recorder|monitor)|nas_lib_test)$' --output-on-failure
python3 -B tools/flight_test/tests/test_capture.py
python3 -B tools/flight_test/tests/test_decode_events.py
python3 -B tools/flight_test/tests/test_recovery.py
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
