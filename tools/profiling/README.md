# OAI nrUE/gNB archival profiler

This directory contains the offline utilities for the explicitly enabled
nrUE/gNB profiler. Schema 2 records boundary durations and instant events with
causal and execution context that later stage, kernel, and microarchitectural
probes can share. Profiling remains disabled unless `-P` or `OAI_PROFILE`
enables it.

## Default archive layout

When a softmodem runs from a normal OAI build tree, `-P` derives the archive
root from the parent of the repository containing the executable:

```text
<OpenAirInterface>/PerformanceProfiles/
  configs/
    gNB/
    nrUE/
  YYYY-MM-DD_HH-MM-SS_gNB_<hostname>/
  YYYY-MM-DD_HH-MM-SS_nrUE_<hostname>/
```

For the intended installations, the defaults are:

```text
Laptop: /home/turker/Documents/OpenAirInterface/PerformanceProfiles
CM5:    /mnt/ssd/Documents/OpenAirInterface/PerformanceProfiles
```

If two processes start during the same second on one host, the later run gains
`_01`, `_02`, and so on. The profiler creates both role-specific config
directories but deliberately does not copy a config into a run directory,
because nrUE configuration can contain subscriber credentials. The run-name
format sorts chronologically in ascending lexical order. Place working configs
under `configs/gNB` and `configs/nrUE`, use versioned filenames, and do not
overwrite a config needed to reproduce an archived run.

Before the first run on each host, create the config directories because OAI
must open the `-O` file before the profiler can initialize its archive:

```bash
mkdir -p /home/turker/Documents/OpenAirInterface/PerformanceProfiles/configs/{gNB,nrUE}
mkdir -p /mnt/ssd/Documents/OpenAirInterface/PerformanceProfiles/configs/{gNB,nrUE}
```

Place each host's config in the corresponding directory. This setup is needed
once; subsequent profile runs create their own output directories.

## Start a profile

From each repository's `cmake_targets/ran_build/build` directory:

```bash
sudo ./nr-softmodem -P -O /home/turker/Documents/OpenAirInterface/PerformanceProfiles/configs/gNB/gnb.conf
sudo ./nr-uesoftmodem -P -O /mnt/ssd/Documents/OpenAirInterface/PerformanceProfiles/configs/nrUE/nrue.conf
```

`-P` uses the default archive root, 131072 records per producer thread, a
100000 us writer flush period, and 1000000 us host-metric sampling. It does not
change radio configuration, including `min_rxtxtime`. Effective settings such
as `gnb.min_rxtxtime` and the USRP transmit-thread mode are recorded in
`settings.csv`.

Each process directory contains:

```text
events.csv                       Duration/instant records and causal identity
event_catalog.csv                Stable event IDs and auxiliary-field units
sync.csv                         Realtime, monotonic, and counter anchors
clock_catalog.csv                Clock domains, units, scopes, and resolutions
drops.csv                        Ring and span-stack integrity diagnostics
metadata.txt                     Schema, run, host, source, and lifecycle data
settings.csv                     Effective softmodem/profiler settings
host_metrics.csv                 Thermal, throttle, CPU, memory, and process data
pmu_catalog.csv                  Requested portable perf-event definitions
pmu_availability.csv             Per-thread PMU support and permission results
pmu_samples.csv                  Raw/scaled per-thread interval counters
pmu_read_overhead.csv            PMU collection cost and read-error counts
thread_metrics.csv               Per-thread scheduler, fault, and CPU-frequency data
kernel_activity.csv              Process-wide scheduler/kernel activity
interrupts.csv                   Per-CPU hard-IRQ deltas and descriptions
softirqs.csv                     Per-CPU soft-IRQ deltas
system_catalog.csv               System-stream metric definitions
system_read_overhead.csv         Writer-thread /proc collection cost
profiler_primitive_overhead.csv  Startup calibration for profiler primitives
external_sources.csv             Sidecar/power provenance and alignment state
```

PMU, scheduler, kernel, and host sampling are performed by the profiler writer
thread. Producer paths publish only fixed-size profiler records. A stream can
be header-only when it is disabled or unavailable; absence is not converted
into a zero measurement.

Schema 2 retains the existing numeric event IDs and names. Important event
fields are:

- `absolute_slot`: non-wrapping work position when the caller provides it;
  `-1` means unknown.
- `correlation_id`: identity shared by records belonging to one logical work
  item; `0` means that a current call site has not yet established context.
- `span_id` and `parent_id`: unique record identity and local or propagated
  ancestry. A root can therefore retain the span that dispatched it from
  another thread.
- `nesting_depth`: fixed-storage local nesting depth. The implementation does
  not allocate while entering or leaving a registered-thread span.
- `cpu_start` and `cpu_end`: CPUs observed at duration endpoints.
  `cpu_migrated=1` proves an endpoint change; zero does not prove that the
  task never migrated away and back during the interval.
- `event_kind`: `duration` or `instant`. Instant events have zero duration.

`metadata.txt` is authoritative for `schema_version`,
`event_record_size_bytes`, counter frequency, and counter semantics. The
validated schema-2 record is 120 bytes on x86-64. At the default 131072-record
capacity this is 15 MiB per registered producer thread. Buffers are allocated
lazily for active producer threads, not for all 256 registry entries. Record
the actual process RSS and drop counters when selecting a capacity for a
campaign.

On Raspberry Pi, `host_metrics.csv` reads the firmware throttling status
directly through `/dev/vcio`; no `vcgencmd` subprocess is started. Host sampling
runs in the profiler writer thread, not in a PHY/MAC producer path.

## Pair gNB and nrUE runs

With synchronized host clocks, the analyzer pairs a UE run only when exactly
one gNB run overlaps it in wall-clock time. Multiple candidates are reported as
ambiguous; the absence of a candidate is reported as unmatched.

For overlapping or automated campaigns, set the same explicit identifier on
both hosts. Because `sudo` commonly filters environment variables, pass it with
`env`:

```bash
sudo env OAI_PROFILE_EXPERIMENT_ID=campaign01 ./nr-softmodem -P -O gnb.conf
sudo env OAI_PROFILE_EXPERIMENT_ID=campaign01 ./nr-uesoftmodem -P -O nrue.conf
```

An explicit identifier takes precedence over wall-clock matching. It identifies
the experiment; each process still has its own timestamped run directory.

## Collect CM5 profiles

From the laptop repository root, preview and then collect missing nrUE runs:

```bash
./tools/profiling/oai_profile_collect.py turker@cm5 --dry-run
./tools/profiling/oai_profile_collect.py turker@cm5
```

The collector defaults to the CM5 and laptop roots shown above. Existing local
runs are skipped. Each new run is copied into a temporary directory on the
laptop. A finalized archive is verified before atomic publication; the
collector restores only the exact whole-second mtime truncation caused by the
measured SFTP transport, then verifies it again. Other integrity failures abort
publication. A manifestless partial run is published as transferred and
explicitly labeled unverified. The collector transfers timestamped nrUE process
directories only; it does not copy `configs/nrUE`. After collection it prints
a ready-to-run analyzer command with a literal, timestamped child below
`Analysis`. If that name already exists, `_01`, `_02`, and later suffixes are
tested until the recommendation is nonexistent; no shell command substitution
is left for the operator to expand.

## Run a controlled paired campaign

`campaign_laptop_cm5.example.json` defines the current Band 28, 25 PRB
laptop-gNB/CM5-nrUE protocol. Its one case, seven observer variants, and five
trials expand to 35 paired experiments and 70 process runs:

- `disabled`: profiler off; process-level observer baseline.
- `in-process`: event profiler on, PMU off; event-level observer baseline.
- `pmu-software`: in-process events plus software perf counters.
- `pmu-all`: software, hardware, and cache/TLB counter requests.
- `perf-stat`: aggregate external perf-stat sidecar.
- `perf-record`: sampled call paths in binary `perf.data`.
- `perf-sched`: binary scheduler trace with alignment left pending.

The default is a read-only, redacted dry run:

```bash
CAMPAIGN=tools/profiling/campaign_laptop_cm5.example.json
./tools/profiling/oai_profile_campaign.py "$CAMPAIGN"
```

Inspect one experiment before enabling the full matrix:

```bash
selection=(--case band28-25prb-minrxtx3 --variant in-process --trial 1)
./tools/profiling/oai_profile_campaign.py "$CAMPAIGN" "${selection[@]}"
```

Execution is opt-in. Use the same selection for the first hardware smoke run:

```bash
./tools/profiling/oai_profile_campaign.py "$CAMPAIGN" --execute "${selection[@]}"
```

When a role has `"sudo": true`, the runner deliberately requires
non-interactive `sudo -n`; it does not wait for a password after the peer radio
has started. SSH, archive-tool, sidecar, ownership, and sudo preflight complete
before either softmodem launches. Roles then start in the declared order with
their configured launch delay. Stop escalation is SIGINT, SIGTERM, then
SIGKILL. A partial launch or nonzero exit is preserved as an explicit result;
prepared run directories are not silently deleted.

Each role receives one common campaign/experiment/variant/trial identity and an
explicit `OAI_PROFILE_DIR`. The run contains `campaign_run.json`, process
stdout/stderr, optional sidecar output, profiler output when enabled, and a
final immutable manifest. Laptop-side control logs and
`campaign_results.csv` live under the configured campaign control root.
`campaign_run.json` labels launch and stop timestamps as either
`measured_host` or `orchestrator`. For remote roles, these process bounds use
the laptop orchestrator clock and are therefore left blank in the remote
`external_sources.csv`; they are not relabeled as CM5-local anchors.
`perf-record --clockid mono_raw` samples still use the remote source host's
`CLOCK_MONOTONIC_RAW` and can be aligned to that run's `sync.csv`.
Commands and environments are archived in redacted form; password/token/key
fragments and IMSI/SUPI/IMEI subscriber identifiers are never retained.
Successful execution requires every role to reach the declared duration,
return zero after the requested stop, finalize its archive, and register every
requested sidecar. An early role exit fails the experiment even when its
process wrapper returns zero.
`--keep-going` continues the matrix after a failed paired experiment; without
it, the runner stops after preserving and finalizing the failed experiment.

### Run the attach-gated loaded protocol

`campaign_laptop_cm5.loaded.example.json` keeps the same laptop/CM5 role
commands, RF configs, seven observer variants, five trials, and 120-second
declared measurement duration, but uses the separate campaign identity
`band28-25prb-cm5-loaded`. It does not set `UHD_IMAGES_DIR`; the host and sudo
environment remain responsible for the already validated UHD image policy.

The loaded protocol runs one client helper on the CM5 nrUE host. It:

- waits up to 120 seconds for `oaitun_ue1` to have exactly one usable
  `10.0.0.0/24` IPv4 address;
- refuses a pre-existing workload lease, unexpected table-9999 policy rule or
  route, a competing nrUE, or ambiguous interface state;
- creates only the two exact source/destination rules and default route needed
  for that UE address in policy table 9999;
- sends three source-bound pings to the fixed external-DN server
  `192.168.70.135`;
- runs 120 seconds of bidirectional UDP with one 1 Mbit/s stream per direction
  and 1200-byte datagrams; and
- removes only the exact state owned by its verified lease after the nrUE is
  proven stopped. It never flushes table 9999 and does not start, stop, or
  signal the external-DN iperf3 server.

Any pre-existing table-9999 entries are a preflight failure, not state that the
runner silently adopts or deletes. Inspect such residue outside a radio run,
establish its owner and purpose, and remove only entries that are independently
proved stale. The campaign must not touch an unrelated iperf3 server process.

Preview the complete loaded matrix without creating a run, changing policy
routing, or launching a binary:

```bash
CAMPAIGN=tools/profiling/campaign_laptop_cm5.loaded.example.json
./tools/profiling/oai_profile_campaign.py "$CAMPAIGN"
```

The first live loaded pilot is one in-process trial, not the 35-experiment
matrix:

```bash
selection=(--case band28-25prb-minrxtx3 --variant in-process --trial 1)
./tools/profiling/oai_profile_campaign.py "$CAMPAIGN" "${selection[@]}"
./tools/profiling/oai_profile_campaign.py "$CAMPAIGN" --execute "${selection[@]}"
```

Run the second command only after the core network, external-DN iperf3 server,
both selected binaries/configs, radios, passive host gates, and emergency-stop
procedure have been verified. The helper begins its traffic interval only
after UE-interface readiness and ping validation, so attach/setup time is not
silently counted as 120 seconds of offered load.

The workload remains evidence, not a third campaign role. Both role archives
receive the same `workload/workload_run.json` before finalization and register
it as an external source. Raw ping/iperf JSON, partial timeout output,
preflight/cleanup state, and control records remain in the workload directory.
For every remote role and helper action, start/completion records bind the
action, experiment, random token, process group, and `/proc` start ticks. A
remote signal is sent only to an exactly matching live identity.

SSH transport status and matching remote completion status are separate
evidence. Remote success requires an authoritative matching completion with
return code zero; process-group absence without completion proves shutdown but
does not invent a return code. A loaded campaign member succeeds only with
`workload_status=completed`, a registered workload artifact,
`network_cleanup_status=ok` or `already_absent`, and
`stop_reason=measurement_complete`. A legacy/unloaded member instead requires
`workload_status=not_configured` and `stop_reason=duration_elapsed`.
`campaign_runs.csv`, `campaign_completeness.csv`, and observer-effect
selection use this same distinction.

## Finalize and verify archives

Finalize only after all profiler, campaign, sidecar, and external-source files
have been written:

```bash
./tools/profiling/oai_profile_archive.py finalize /path/to/process-run
./tools/profiling/oai_profile_archive.py verify /path/to/process-run
```

`archive_manifest.csv` binds every regular file to size, nanosecond mtime,
SHA-256, run identity, and artifact class. Verification detects changed,
missing, extra, duplicated, unsafe, and symlinked artifacts. The manifest
protects archive consistency against accidental modification; it is not proof
of authorship or custody. Sign or store the manifest in a trusted external
system when authenticity is required.

M5Stack ingestion is intentionally deferred, but a raw future artifact can be
registered before finalization without inventing alignment:

```bash
RUN=/path/to/process-run
POWER=/path/to/raw-m5stack-export.csv
power_source=(
  "$RUN"
  --source-id m5stack-va
  --source-type m5stack_voltage_current
  --artifact "$POWER"
  --copy-artifact
  --clock-domain device_clock
  --clock-unit unknown
  --tool-version '<firmware and export-tool versions>'
  --status recorded
  --alignment-method unresolved
  --notes 'Raw voltage/current/power evidence; no clock transform applied'
)
./tools/profiling/oai_profile_archive.py register-external "${power_source[@]}"
```

This preserves provenance while preventing unmeasured clock offset, drift, or
resampling assumptions from entering the analysis.

## Analyze an archive

Run analysis on the laptop:

```bash
ANALYSIS_DIR=/home/turker/Documents/OpenAirInterface/PerformanceProfiles/Analysis/2026-07-24_02-15-00
./tools/profiling/oai_profile_analyze.py \
  /home/turker/Documents/OpenAirInterface/PerformanceProfiles \
  --output-dir "$ANALYSIS_DIR"
```

Choose a new literal final component for every manual invocation; the
destination must not exist. The collector command above automatically prints
such a collision-checked recommendation.

`--output-profile full` is the default and preserves all detailed outputs. For
publication-scale repeated analysis, use:

```bash
./tools/profiling/oai_profile_analyze.py \
  /home/turker/Documents/OpenAirInterface/PerformanceProfiles \
  --output-profile publication \
  --output-dir "$ANALYSIS_DIR"
```

The publication profile omits only the regenerable per-event
`event_timeline.csv` and per-duration `hierarchy.csv`. It retains canonical
summaries, integrity evidence, deadlines, correlations, and hard hierarchy
anomalies. Routine valid asynchronous edges are represented compactly in
`causal_edges_summary.csv`, keyed by profile, parent/child event, absolute-slot
delta, and temporal shape with edge counts and boundary-distance
distributions.

The analyzer writes:

- `summary.csv` and `by_thread.csv`: inclusive per-event distributions.
- `event_timeline.csv` (full profile): every normalized event plus its
  process-local clock mapping.
- `hierarchy.csv` (full profile): one schema-2 duration row with its parent relation,
  direct-child interval union, overlap, and validated exclusive duration.
- `exclusive_summary.csv`: per-event exclusive distributions using valid
  hierarchy rows only.
- `hierarchy_anomalies.csv`: full mode includes missing parents, correlation
  mismatches, duplicate span IDs, and causal children outside their parent's
  time interval. Publication mode retains the first three hard relations;
  routine causal children move to `causal_edges_summary.csv`.
- `hierarchy_integrity.csv`: per-process identity, parent, correlation,
  absolute-slot, and nesting diagnostics.
- `correlations.csv`: one process-local radio-work correlation with its slot
  range, elapsed interval, roots, threads, depth, and migrations.
- `deadline_misses.csv`: legacy realtime deadline-miss events retained for
  archive compatibility. Current binaries retain the event ID and descriptor
  but emit monotonic compute/check evidence instead, so this file can be
  header-only and must not be interpreted as evidence that no deadline was
  missed.
- `deadline_checks.csv`: one nrUE hardware-TX check per occurrence, preserving
  runtime monotonic validity, signed lateness/headroom, paired compute evidence,
  radio-anchor provenance, independent radio-tick reconstruction, and
  classification agreement.
- `deadline_summary.csv`: per-process deadline event cardinality, validity,
  miss rates, reconstruction coverage, anchor provenance, agreement, and
  lateness/headroom/bias distributions. Its local deadline `status` describes
  the persisted check/compute evidence; the independent
  `profile_coverage_status` states whether that event evidence covers a
  complete profile.
- `migrations.csv`, `runs.csv`, `pairs.csv`, and `host_summary.csv`:
  execution, pairing, and host-health reports.
- `profiler_primitive_overhead_summary.csv`: setup, warm-up, and measurement
  distributions. The reported excess is a difference of phase medians, not a
  per-record correction. Its trailing `stream_status` distinguishes recorded
  evidence from missing, zero-byte, header-only, or malformed calibration
  streams; unavailable sentinel rows leave counts and durations blank.
- `pmu_availability_summary.csv`, `pmu_summary.csv`, and `pmu_quality.csv`:
  requested/support/permission state, valid scaled rates, read errors, and
  interval multiplex quality.
- `thread_scheduler_summary.csv`: runtime, run-queue wait, timeslices, faults,
  context switches, CPU changes, and sampled frequency.
- `kernel_interference_summary.csv`: process/kernel activity and per-CPU
  hardirq/softirq rates, including USB/network relevance labels.
- `transport_summary.csv` and `transport_faults.csv`: outer RF plus nested
  UHD timing and raw short-transfer/overflow/async-event evidence.
- `collection_overhead_summary.csv`: PMU, system, and primitive collection
  cost distributions with explicit error counts and an independent
  `stream_status` for every source, including unavailable sentinels.
- `archive_integrity.csv` and `external_sources.csv`: per-artifact manifest
  verification plus source/provenance/alignment state.
- `perf_stat_summary.csv`: tolerant aggregation of registered semicolon
  perf-stat text, including unsupported/uncounted rows and running percentage.
- `campaign_runs.csv` and `campaign_completeness.csv`: disabled and profiled
  role states, anchors, exits, manifests, operational paired completeness, and
  a separate, deliberately narrow profiler-event-evidence qualifier.
- `observer_effect_summary.csv`: repeated-run process outcomes relative to
  `disabled`, and per-run event medians relative to `in-process`. For each
  event metric observed elsewhere in the same campaign/case/role cohort, the
  exclusion count includes every successful profiler-enabled incomplete run,
  even if that run emitted no row for the event.
- `analysis_inputs.csv`: discovered run identities plus presence and SHA-256
  of each input `archive_manifest.csv`.
- `analysis_provenance.csv`: the raw argument vector, parsed input arguments,
  event filter, output profile/destination, working directory, Python identity,
  and SHA-256/path for the analyzer and directly affecting clock, deadline,
  report, and archive-integrity modules. These rows are captured immediately
  after argument validation, before profile discovery and the long event pass.
- `analysis_manifest.csv`: every generated or intentionally omitted analysis
  artifact, with generated data-row count, size, and SHA-256. The manifest
  explicitly excludes its own digest to avoid self-reference.

Derived integrity fields distinguish missing evidence from a measured zero.
`drops_total`, span-stack diagnostics, and `counter_regressions` are numeric
only when the corresponding `drops.csv` columns contain structurally valid
native per-thread identities and nonnegative rows. For schema 2, every thread
observed producing an event must have a corresponding drop row; additional
registered threads that emitted no event are permitted. Missing, zero-byte,
header-only, malformed, and
legacy-partial streams remain explicit in `drop_diagnostics_status`;
unavailable numeric cells are blank. `profile_coverage_status=complete`
additionally requires schema 2 with a positive integer `counter_hz`; a
strictly valid recorded `events.csv` whose tick durations agree with the
writer's three-decimal microsecond values; a strictly valid recorded
`event_catalog.csv` whose IDs and names match the events; the complete and
internally consistent native lifecycle footer; and recorded zero-valued drop,
span, and counter-regression diagnostics. A descriptor's `default_kind` is
the catalog default, not a per-row invariant:
`PROFILER_PRIMITIVE_CALIBRATION` intentionally emits both duration and instant
records. Schema-2 instant rows must have zero duration and equal observed CPU
endpoints, and no row may parent itself.
Stream states, row counts, producer counts, and missing producer diagnostics
are emitted alongside the global status. Inclusive statistics from an
incomplete archive remain observed persisted-prefix
evidence, but event-level observer-effect medians exclude incomplete profiles
from otherwise operationally successful runs and report that exclusion count.
Per-record hierarchy `exclusive_valid` remains a local statement about
persisted parent/child interval structure; global archive coverage is reported
separately and gates observer/publication aggregation.

In `campaign_completeness.csv`, `paired_complete` and `status` retain their
operational meaning: both roles finished successfully, were finalized, and
passed archive-manifest verification. They do not by themselves prove complete
profiler evidence. The appended `profile_evidence_complete` and
`profile_evidence_status` fields additionally require every profiler-enabled
role to have `profile_coverage_status=complete`; profiler-disabled roles are
explicitly `not_applicable`. Canonical variants must agree with their declared
profile setting, and profiler artifacts in a declared-disabled role are
reported as unexpected evidence; either profiler metadata or an event artifact
is sufficient to detect that contamination even when no complete profile was
discovered. Per-role coverage and
incomplete/unknown/mismatched role sets remain in the same row. A
manifest-valid zero-byte diagnostic stream can therefore be operationally
intact while its profile evidence remains incomplete. This bounded qualifier
does not validate PMU, sidecars, host/system sources, clock alignment, RF
success, or whole-publication scientific validity; those are separate gates.

For an unfinalized schema-2 process, `runs.csv` reports
`clean_shutdown=unknown` when completion metadata is absent. A duration
bounded by the last durable `sync.csv` sample is labeled
`duration_scope=sync_prefix` and `duration_status=valid_sync_prefix`; it is not
a complete process lifetime. Native `sync.csv` rows with non-`ok` status are
not used as bounds. A clean footer that contains all required fields but whose
clock, duration, or regression values contradict one another is explicitly
`invalid_lifecycle_metadata`/`lifecycle_clock_invalid` and has no numeric
process duration.

PMU reports likewise distinguish missing, zero-byte, malformed, valid
header-only, and recorded streams. A clean PMU-off header-only stream is
`not_requested`; a crash-truncated zero-byte stream does not manufacture
requested, available, validity, or read-error zeros. Recorded PMU availability
and sample rows must contain the full native run/thread/event identities,
timestamps, raw and scaled counts, enabled/running times, deltas, intervals,
flags, status, and error codes. Samples must join an exact descriptor with
`requested=1` and `available=1`. Raw-derived/scaled values must be
nonnegative, running time cannot exceed enabled time, multiplex ratios must be
in `[0,1]`, and a usable row requires a positive interval, both validity flags,
`status=ok`, and `error_code=0`. Whenever scaling is marked valid, cumulative
scaled values are recomputed from raw count and cumulative enabled/running
time; usable delta-scaled values and multiplex ratios are independently
recomputed from their interval fields. Comparisons honor the writer's six- and
nine-decimal serialization precision. Non-rate rows must match one native
warm-up, clock-regression, counter-reset/reconfiguration, not-running,
read-error, or malformed-group-read state. A valid `not_running` row remains
explicit quality evidence but never contributes a rate, and read-error rows
cannot carry fabricated counter evidence. A semantic violation makes the
complete stream malformed rather than coercing it to zero or true.

`collection_overhead_summary.csv` applies the same structural distinction
independently to PMU-read, system-read, and primitive-calibration streams. A
missing, zero-byte, header-only, or malformed source always receives its own
sentinel row even when another source is recorded. Its unavailable counts and
durations remain blank; only a valid PMU-off header-only stream has known
not-requested zero counts.

The destination must not already exist. Output is written to a sibling partial
directory, each CSV is flushed and synchronized, and the directory is renamed
atomically only after complete generation and provenance. A failure before
rename returns nonzero and leaves the partial directory with
`ANALYSIS_INCOMPLETE.txt` labeled `publication_state=unpublished_partial`; the
requested destination is absent. After rename, the parent directory is also
synchronized. If that final synchronization fails, analysis still returns
nonzero and the visible requested destination is explicitly marked
`publication_state=published_incomplete`. If even that final marker cannot be
persisted, the analyzer makes a best-effort atomic rollback to the partial name
and marks it there. Full and publication analyses preserve the same retained
canonical summary values and ordering.

It accepts Phase 1/schema-1 and schema-2 archives. Schema-1 rows are reported
with unknown event kind, absolute slot, and CPU plus zero causal IDs, rather
than having absent fields inferred retrospectively. Hierarchy reports contain
schema-2 records only. The analyzer uses only the Python standard library.

In-process PMU rows are periodic per-thread sampling intervals, not exact
function measurements. A rate is emitted only when `delta_valid=1`,
`scaling_valid=1`, the interval is positive, and the scaled delta is finite.
Multiplex ratios use those same valid intervals. Join these interval samples to
OAI event time ranges only as interval attribution; use the separately
manifested `perf-record` call paths for sampled symbol/call-chain attribution.

Binary `perf.data` and perf-sched artifacts are inventoried and integrity
checked, but the analyzer does not silently symbolize them or transform their
clocks. Perform those host/tool-version-dependent operations explicitly and
register the derived artifact as another external source.

## nrUE hardware transmit deadline semantics

`UE_TX_DEADLINE_COMPUTE` records the end-of-radio-read sample timestamp, the
target transmit sample timestamp, samples per subframe, and the corresponding
`CLOCK_MONOTONIC_RAW` anchor. `UE_TX_DEADLINE_CHECK` records the current and
deadline `CLOCK_MONOTONIC_RAW` values, signed lateness in nanoseconds, error
code, and validity flags. A positive signed lateness is a miss; zero is on
time. Clock or checked-arithmetic failure clears validity and remains an
invalid row rather than becoming a fabricated hit or miss.

Offline reconstruction is independent of the runtime monotonic comparison. It
pairs a check only with a preceding compute event having the same positive
correlation ID and transmit frame/slot. It then selects only a receive endpoint
from that correlation: a parent-linked or temporally contained
`USRP_RX_RECV` is preferred, followed by an explicitly labeled outer
`UE_RF_READ_DRIFT` or `UE_RF_READ` fallback. The signed radio-sample offset
is converted to profiler ticks with checked integer round-to-nearest using the
archive's `counter_hz` and samples-per-subframe value.

`deadline_checks.csv` reports runtime and reconstruction validity separately.
Classification agreement and reconstruction-minus-runtime bias exist only
when both paths are valid. Missing correlations, frame/slot mismatches,
missing anchors, malformed monotonic identities, invalid rates, and arithmetic
flags remain explicit. They are never replaced with nearby events or inferred
values. A schema-1 archive containing only `UE_TX_DEADLINE_MISS` is labeled
`legacy_only` in `deadline_summary.csv`; no monotonic evidence is inferred
retrospectively.

## Phase 2B hierarchy semantics

Phase 2B adds Stage events for nrUE MAC/PBCH/PDCCH/PDSCH/DLSCH and uplink
generation, gNB MAC/DL/PDSCH and UL/PUSCH/ULSCH processing, RU front ends,
PRACH/PUCCH/SRS, and shared LDPC decoder and encoder workers. Event descriptors
in `event_catalog.csv` are authoritative for auxiliary-field names and units.
Shared `nrUE/gNB` LDPC descriptors resolve to the process role from
`metadata.txt` during analysis.

On current grouped gNB PUSCH processing, one `GNB_RX_PUSCH_FRONTEND` span
covers the shared group front end and its per-UE detection checks. Its
`layers` field is the total number of jointly processed layers and its `dtx`
flag means at least one UE in the group was classified as DTX. For a one-UE
group these retain their original single-PDU meanings. Demapping and
unscrambling remain separate per-UE child spans.

For `UE_PDSCH_MRC_MMSE`, `equalizer_mode=0` means that no separate MMSE
kernel ran in this measured block, `1` selects general multi-layer MMSE, and
`2` selects the specialized two-layer MMSE kernel. Its `ml_enabled` flag,
and the same flag on `UE_PDSCH_LLR`, identify the joint two-layer
QPSK/16QAM/64QAM ML-LLR path that executes later in the LLR stage.
`UE_PDSCH_SCOPE_COPY.copy_kind` values `0`, `1`, `2`, and `3` denote
channel-estimate, extracted-RX, compensated try-lock, and compensated fallback
copies, respectively. Scope-copy events exist only when a corresponding scope
operation is requested or available.

`UE_TX_DEADLINE_MISS`, `GNB_RU_TX_NORTH`,
`UE_PDSCH_WORKSPACE_ALLOCATION`, and `UE_PDSCH_WORKSPACE_FREE` are reserved
compatibility descriptors without current production producers. The first
preserves the legacy deadline event; the second preserves the event ID for the
NR RU north callback removed upstream; and the workspace descriptors preserve
archives from revisions that allocated and freed PDSCH scratch per slot.
Current PDSCH scratch is allocated persistently per actor outside the measured
slot path. Absence of any reserved event must not be interpreted as measured
zero.

All recorded durations are inclusive. Exclusive time is valid only when every
direct duration child has the same correlation and lies inside the parent
interval. Parallel child durations are not summed: the analyzer subtracts the
union of their intervals, records their overlap separately, and therefore does
not double-subtract concurrent PDSCH, PUSCH, or LDPC workers.

A propagated child can legitimately start after its short dispatch parent has
ended. Such an edge is reported as `causal_noncontained`; it is useful for
dispatch-to-start analysis but is not subtracted from the dispatch duration.
Likewise, a child can intentionally carry a future transmit absolute slot.
Absolute-slot deltas are preserved and do not by themselves invalidate a
causal edge. Missing parents and correlation mismatches are integrity
failures. Nonzero `drops.csv` or span-stack diagnostics invalidate claims of
complete hierarchy coverage and must be reported with any result.

The `--event` option limits `summary.csv`, `by_thread.csv`, legacy
`deadline_misses.csv`, and `migrations.csv`. The authoritative
`deadline_checks.csv` and `deadline_summary.csv`, hierarchy, and other
publication-level reports still read the complete relevant event set so
compute/check pairs, radio anchors, ancestors, children, transport faults, and
observer-control evidence are not silently discarded.

## Implemented phases and remaining hardware gates

- Phase 1 established opt-in archival boundary timing, host health, collection,
  pairing, and offline percentile analysis.
- Phase 2A established the schema-2 semantic substrate: descriptors, absolute
  work position, correlation and parent identity, nested spans, event kind,
  CPU endpoints, migration reports, and race-free producer/writer publication.
- Phase 2B instruments deep nrUE, gNB, RU, and shared LDPC processing stages
  using the same IDs and context, and adds overlap-safe hierarchy, exclusive
  time, integrity, and per-correlation analysis.
- Phase 2C/2D and Phase 3 software support now add the archival/clock contract,
  grouped portable PMU events, scheduler/IRQ evidence, nested USB/UHD
  transport, startup calibration, external perf sidecars, transactional paired
  campaigns, immutable manifests, and publication-oriented reports.

Software support is not hardware validation. Before scientific use, run the
focused paired campaign on the laptop/B210 gNB and CM5/B205mini-i nrUE and
verify: clean shutdown and zero drops, PMU availability/permissions, acceptable
multiplexing, stable primitive/collector overhead, sidecar compatibility,
USB-transfer fault semantics, scheduler/IRQ attribution, thermal/throttling
state, and observer-effect distributions across repeated trials. Cross-host
causal joins require measured clock quality; shared wall-clock timestamps alone
do not prove sub-slot synchronization.

The event elapsed-time counter is not a retired CPU-cycle counter. M5Stack
power samples remain external. Their later ingestion can use campaign identity,
clock catalogs, realtime/monotonic anchors, and explicit uncertainty without
renumbering events or changing archived event records.

Run analyzer schema-regression tests from the repository root:

```bash
PYTHONPYCACHEPREFIX=/tmp/oai-profile-pycache \
  python3 -m unittest discover -s tools/profiling/tests -v
```

## Overrides

- `--oai-profile-dir <path>` or `OAI_PROFILE_DIR` selects one exact process
  output directory. Existing profiler output is never overwritten.
- `OAI_PROFILE_ROOT` changes the automatic archive root while preserving
  timestamped run naming.
- `--oai-profile-buffer-records` and `OAI_PROFILE_BUFFER_RECORDS` change the
  producer ring capacity.
- `--oai-profile-flush-us` and `OAI_PROFILE_FLUSH_US` change writer flushing.
- `OAI_PROFILE_HOST_METRICS_US` changes host sampling; the minimum is 100000 us.
- `--oai-profile-pmu off|auto|software|hardware|all` or `OAI_PROFILE_PMU`
  selects portable PMU requests. Unsupported or denied counters remain explicit
  in `pmu_availability.csv`.
- `--oai-profile-pmu-sample-us` or `OAI_PROFILE_PMU_SAMPLE_US` changes PMU
  and per-thread system sampling; the minimum is 100000 us.
- `OAI_PROFILE_CALIBRATION_WARMUP` and
  `OAI_PROFILE_CALIBRATION_SAMPLES` change bounded startup calibration
  repetitions; defaults are 64 and 1024, and the maximum for either is 65536.
- `OAI_PROFILE_CAMPAIGN_ID`, `OAI_PROFILE_VARIANT`, and
  `OAI_PROFILE_TRIAL` preserve manual campaign identity. The campaign runner
  sets these automatically.

When setting an override under `sudo`, use `sudo env NAME=value ...` unless the
local sudo policy explicitly preserves that variable.
