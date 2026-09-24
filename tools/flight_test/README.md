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
file, `--flight log` on the terminal). `recovery` requires `log` and supports both UE and gNB workers. Unknown features such as
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

## gNB recovery and radio-health evidence

The same `flight = "log recovery";` setting enables a gNB session supervisor. It does not use UE attach, TUN,
IP-address, PDU-session, NAS, DRB, or cell-acquisition deadlines. A gNB may retry only an unexpected nonzero
worker exit after **two fresh, ordered, active snapshots for one device slot** show an increase in either
`tx_send_accepted_samples` or `rx_returned_samples`. The first static snapshot, requested TX samples, counters
from different slots, stale/reordered snapshots, startup failures, and clean unclassified exits do not
qualify. The same capped 5/15/30/60-second process backoff and ten-second owned-group fencing apply. Ctrl+C
and SIGTERM stop the whole session and never relaunch it. A gNB startup or replacement attempt that exits
before this qualification stops conservatively.

Each worker also records private monitor-socket `radio_health` schema-v1 datagrams in `radio_health.*.log`
JSON Lines. The stream retains accepted raw snapshots and per-device intervals, including source/collector
gaps, active/closed lifecycle, supported-but-unobserved values, value decreases or counter resets, and
first/last native event identity. It marks every late, underflow, overflow, queue, missing, or in-flight
finding with `candidate_action="observe"` and `qualified=false`. Those records are evidence for offline
analysis only; they are not restart triggers for UE or gNB and do not establish an RF fault, packet delivery,
or a 3GPP threshold.

Schema v1 currently advertises bounded device slots and TX send, TX async, RX stream, and TX queue metrics.
The collector distinguishes known counters, known gauges, and bounded future numeric names that remain opaque
until documented. Requested TX, accepted TX, and delivered TX are separate facts; v1 has no delivered-sample
metric. A supported counter observed as zero differs from an absent gauge. Device ticks are not interpreted
unless their companion `*_device_time_valid` value is observed as one. The `*_sample_rate_microhz`
gauges preserve the actual nominal rate returned by the backend, rounded to the nearest microhertz
(divide by 1,000,000 for Hz). For example, `7680000005984` means `7680000.005984 Hz`.
The older `*_sample_rate_hz` gauges are additionally present when the nominal rate is exactly an integer
in Hz. These are backend-coerced nominal rates, not measurements of oscillator accuracy or requested
configuration rates. `tx_send_inflight` and `rx_recv_inflight` preserve
pending-call evidence only when the corresponding known accepted/returned sample delta is zero; unavailable
deltas remain unavailable rather than becoming zero. Native sample counts follow UHD send/recv return
semantics per channel for a stream; they are not RF-delivery proof. Event counts aggregate per registered
device/stream, and valid last metadata includes only the latest channel rather than a per-channel event
history.

The radio-health writer shares the existing `flight-host-budget` and free-space reserve with host and recovery
journals. It uses the same bounded sequential rotation; a quota or storage loss is retained in final status
and never blocks radio work.

## Output and optional settings

With no overrides, each invocation creates:

```text
<checkout>/cmake_targets/log/FlightTests/YYYY-MM-DD/
  ue-<UTC-start>-<pid>-<unique-id>/
    metadata.json
    stdout.0000.log
    stderr.0000.log
    host.0000.log
    radio_health.0000.log
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
- Private native radio-health snapshots, per-device metric intervals, lifecycle and observer-only transport diagnostics.
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
UHD TX asynchronous metadata is collected through the private radio-health
channel in `radio_health.*.log`. Use its supported typed counters, source sequence
and timestamps to distinguish startup, steady operation and teardown. Numeric
RADIO_TX events describe sample acceptance; they do not prove RF delivery.

## Gain observations

Gain control is selected separately from flight capture. `agc` is not a valid
`flight` token. For an initial fixed-gain comparison on either NR role:

```conf
agc-mode = "observe";
agc-directions = "rx";
flight = "log recovery";
```

On the UE, omit the old `agc = 1`/`--agc` acquisition option in observe mode;
the resolver rejects that conflict. The current managed-radio integration uses
one radio, one RX stream and one TX stream, with an optional backend capability
interface. Unsupported combinations report an error instead of silently using a
second gain writer. The legacy no-option/off radio path remains available.
UE managed modes, including `--agc` alone or `agc = 1`, default to the existing
software CFO compensation `cont-fo-comp = 1` when that option is absent. Startup
logs this resolved default. Explicit modes 1/2/3 are preserved; explicit mode 0
is rejected. Thus the legacy gain-acquisition policy remains selectable, but its
former default hardware-CFO-retuning path is replaced by software compensation.
Use the same explicit CFO mode when comparing gain policies. Hardware CFO retuning after TX
admission has no qualified quiescent boundary and is rejected before streaming.

`continuous` with explicit `agc-directions = "rx"` enables the new RX controller.
On the UE, adding `agc = 1` selects legacy acquisition followed by new tracking;
without it, acquisition and tracking use the new policy. A loss of synchronization
returns to the selected acquisition policy. On the gNB, the controller starts in
listening/tracking mode and does not increase gain merely because UL is absent.
The controller is opt-in. Device/band/configuration qualification does not carry over to another radio merely because its model matches.


`observe` records hypothetical one-step RX decisions without applying them.
Raw-read observations provide sparse component peak and near-rail counts;
serving SSB or detected PUSCH provides the tracking reference. The reported
reference level is a full-grid-equivalent digital level, not the power averaged
over idle slots and not calibrated RF input power. A sparse observation does not
establish that every sample avoided clipping.

The decoder writes `radio_gain.csv`, `radio_rx_level.csv` and
`radio_rx_decisions.csv`. Decisions retain their actual input gain, generation,
sample-range endpoint, reason and proposed gain; `submitted` distinguishes a
real admitted request from an observe-only proposal. A backend result includes
reported gain and a device-time bracket when available. `agc-rx-settle-us`
defaults to 20000 and excludes transition samples from gain-dependent reports;
it is an engineering guard, not a guaranteed analog settling-time calibration.

TX request records in `ue_tx_power.csv` describe the MAC request and generator
reference for PRACH/PUSCH/PUCCH/SRS. `gnb_tx_reference.csv` describes configured
SSB reference power and generator amplitude. Neither file reports measured RF
output. `continuous` with `agc-directions = "tx"` or `"both"` also applies
channel-specific TX power, provided a matching qualified profile is supplied.
Explicitly requesting managed TX never silently enables RX-only control.

Use `agc-mode` independently from `flight = "log recovery"`; `flight` has no
`agc` token. On the UE, `--agc` selects the retained legacy acquisition policy.
Adding it to `continuous` selects legacy acquisition followed by new RX tracking;
it does not add a second hardware gain writer. On the gNB, legacy `--agc` and
`acquisition` are unsupported. `observe` computes decisions without changing
samples or gains and does not support combining legacy acquisition with it.

Managed TX holds analog gain fixed after initialization. The UE applies the
MAC's requested power to each complete active PRACH, PUSCH/Msg3, PUCCH or SRS
waveform span, including cyclic prefixes. The initial supported layout is one
TX stream, normal CP and symbol-disjoint channels. Overlapping channel spans
are rejected rather than scaled with one slot-wide multiplier. PUSCH TPC is
applied in scheduled transmission order, and a queued grant is rejected if its
BWP/power configuration changed before that transmission.

The gNB chooses one common generator amplitude from configured SSS resource-element
power and the profile. It preserves channel-relative amplitudes as allocation
changes; it does not renormalize every slot to equal total power. Composite
symbol power and converter peaks are checked before submission. Neither role
increases TX power simply because its RX gain controller sees a weak signal.

The `agc-tx-profile` config section identifies one connector and operating point:

| Field | Meaning |
|---|---|
| `id`, `provenance` | Profile identifier and qualification evidence reference |
| `device`, `antenna` | Exact backend connector identity and port readback |
| `qualified` | Explicit operator qualification; defaults to false |
| `minimum-frequency-hz`, `maximum-frequency-hz` | Qualified frequency interval |
| `sample-rate-hz`, `bandwidth-hz`, `reported-gain-db` | Actual sample rate, analog filter bandwidth and fixed TX gain |
| `component-full-scale` | OAI sample magnitude before the backend converter |
| `reference-dbm` | Connector power at unit complex RMS/full scale |
| `minimum-dbm`, `maximum-dbm`, `uncertainty-db` | Qualified active-channel output range and its uncertainty |
| `peak-limit-fs` | Maximum component peak/full scale; default 0.7 |
| `quantization-tolerance-db` | Maximum digital rounding power error; default 0.5 dB, maximum 1 dB |
| `quantization-evm-limit` | Maximum additional digital rounding EVM; default and maximum 0.03 |

A profile is not inferred from an uncalibrated gain setting. Missing qualification
or a connector/rate/filter/gain/converter mismatch rejects explicit managed TX.
An explicitly supplied UE network `p-Max` must fit the same physical output
range; unsupported alternate limits are rejected. An infeasible request is not
silently clipped while MAC reports unchanged power/headroom. A runtime mapping,
headroom or profile failure closes managed TX admission and asks the existing
ITTI shutdown path to stop the worker. This is a configuration/actuation fault,
not a new radio-recovery threshold.

`radio_tx_power.csv` joins each channel request to fixed-point scaling, input
and output sample energy/peaks, added quantization EVM and profile uncertainty.
The decoder marks missing or ambiguous records rather than inventing zeros.
`radio_tx_rejects.csv` records rejected layouts, spans, power limits and profiles.
`ue_tx_control.csv` records the serving-SSB pathloss, closed-loop adjustment
state after calculation, supplied TPC increment and configured network p-Max for
supported PUCCH calculations and managed PUSCH calculations at their target
slot. A supplied increment can be suppressed by saturation. An absent network
limit remains unavailable in the CSV; it is not converted into zero. These
records can explain changes in requests, but do not prove transmission.
`ue_pathloss_state.csv` records when a serving-SSB pathloss first becomes
available or unavailable. Missing or invalid measurements defer new UL
scheduling without changing power-control adjustment state; elapsed protocol
timers continue. A later valid measurement permits scheduling again. This is
measurement availability, not an RF-fault restart or a guarantee of freshness.

The first dedicated PUCCH support covers one PCell relation using the active
SSB, one matching P0 and pathloss-reference entry, and adjustment state i0.
Multiple relations, another reference signal, another serving cell or two
adjustment states are not supported by this controller. An unsupported MAC
power request produces `POWER_CONTROL` in `radio_tx_rejects.csv` and closes
managed TX admission. The ordinary fixed-amplitude path retains its prior
waveform behavior. This is a bounded implementation of the supported NR power
procedures, not a claim of complete 3GPP power-control conformance.

Estimated output remains a digital/profile estimate; backend acceptance and RF
emission are separate facts. Digital quantization EVM is not measured RF EVM.
A provisional conducted engineering fixture is not a calibrated flight profile.

The old T tracer UE_PHY_MEAS record has no gain-validity fields and reads legacy
averaging state. It is suppressed for managed gain contexts rather than emitting
stale values. Use the gain-aware flight records for this experimental branch.

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
ctest --test-dir <build> -R '^(flight_(startup|recorder|monitor)|nas_lib_test|test_radio_health)$' --output-on-failure
python3 -B tools/flight_test/tests/test_capture.py
python3 -B tools/flight_test/tests/test_decode_events.py
python3 -B tools/flight_test/tests/test_recovery.py
python3 -B tools/flight_test/tests/test_radio_health.py
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


With a managed `agc-mode` and flight logging, `radio_tx_level.csv` adds exact
whole-buffer digital energy/peak and converter-range counts for the first and
every64th admitted TX callback on either role. It includes the backend return
and reported TX gain when available. This is before-conversion digital evidence,
not measured RF power or proof of transmission; unselected buffers are not
covered. The feature observes only and does not change samples or TX gain.


For gain-controlled measurement debugging, `ue_ssb_measurements.csv` records each serving-SSB PHY acceptance or rejection together with raw SSS energy, generation, and noise/gain qualification. Proven adjacent context adds reported RXgain, device sample interval, sampled peak and near-rail counts. A valid driver gain readback alone does not make a clipped measurement suitable for pathloss calculation. Managed UE measurements and gNB noise averaging reject missing/malformed level summaries and observed near-rail samples while the RX controller retains raw overload evidence. The bounded radio sampling cannot prove that every sample was unclipped. Rejected UE updates retain the previous accepted scalar at MAC; this addition does not create a measurement-age policy. See `EVENT_SCHEMA.md` for the exact validity and loss-aware association rules.

Managed serving-SSB measurements are staged until the matching PBCH decode succeeds. A failed decode
cannot replace the verified MAC pathloss reference with noise or select a stronger but unconfirmed
SSB. Gain/noise qualification remains required, and raw overload evidence remains available to the
RX controller. The SSB CSV records whether PBCH confirmation was required, attempted and successful.
This is a conservative measurement-validity policy, not a claim that 3GPP requires a fresh PBCH CRC
for every RSRP observation. The absent-context baseline keeps its previous reporting order. The last
verified measurement is retained across a transient failed PBCH; that measurement policy adds no
expiry timer, TPC reset or minimum-power clamp. Decode selection keeps separate serving-PBCH failure
state, so retaining the verified reference does not prevent an eligible alternate SSB from being
tried after a serving decode failure.

A newly accepted contention-based RAR starts a fresh Msg3 power-control initialization. Its initial
adjustment uses that random-access attempt's ramp and RAR TPC, rather than inheriting the previous
connected-mode PUSCH adjustment. Subsequent retransmissions retain their accumulated state; CFRA
and ordinary C-RNTI grants do not take this initialization path.
Positive RX tracking and acquisition-search steps retain the policy deadband below
the sampled-peak overload threshold. This reserve avoids targeting the overload edge directly. The shared envelope
below also retains stronger-burst evidence across quieter receive windows.
Overload reduction remains immediate subject to the existing actuation cooldown;
the finite sample observations do not prove that all received peaks were measured.

The RX policies also share an input-referred peak envelope across gain changes.
It attacks stronger observed peaks immediately and releases at an initial
engineering rate of 3 dB/s, so a quiet receive window cannot immediately undo
headroom protection for a stronger burst. The retained envelope limits increases;
raw overload still controls reductions. `radio_rx_peak_envelope.csv` records that
constraint separately from the instantaneous peak in `radio_rx_decisions.csv`.
