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
events.csv          Per-occurrence duration/instant records and causal identity
event_catalog.csv   Stable IDs plus role, subsystem, class, kind, and aux units
sync.csv            Realtime, monotonic, and hardware-counter anchors
drops.csv           Producer drops and span-stack integrity diagnostics
metadata.txt        Schema, record, run, host, source, config, and lifecycle data
settings.csv        Effective softmodem/profiler settings
host_metrics.csv    Temperature, throttling, CPU, memory, and process metrics
```

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
laptop and atomically renamed into the archive after `scp` succeeds. The
collector transfers timestamped nrUE process directories only; it does not copy
`configs/nrUE`.

## Analyze an archive

Run analysis on the laptop:

```bash
ANALYSIS_DIR=/home/turker/Documents/OpenAirInterface/PerformanceProfiles/Analysis/$(date +%Y-%m-%d_%H-%M-%S)
./tools/profiling/oai_profile_analyze.py \
  /home/turker/Documents/OpenAirInterface/PerformanceProfiles \
  --output-dir "$ANALYSIS_DIR"
```

The analyzer writes `summary.csv`, `by_thread.csv`, `deadline_misses.csv`,
`migrations.csv`, `runs.csv`, `pairs.csv`, and `host_summary.csv`. It
accepts both Phase 1/schema-1 and Phase 2A/schema-2 archives. Schema-1 rows are
reported with unknown event kind, absolute slot, and CPU plus zero causal IDs,
rather than having absent fields inferred retrospectively. The analyzer uses
only the Python standard library.

## Profiling roadmap

- Phase 1 established opt-in archival boundary timing, host health, collection,
  pairing, and offline percentile analysis.
- Phase 2A establishes the schema-2 semantic substrate: descriptors, absolute
  work position, correlation and parent identity, nested spans, event kind,
  CPU endpoints, migration reports, and race-free producer/writer publication.
- Phase 2B will instrument deeper nrUE and gNB processing stages using the same
  IDs and context, so end-to-end slot critical paths can be decomposed without
  inventing another trace format.
- Phase 3 is the complete microarchitectural campaign: PMU
  cycles/instructions/branches/cache and memory events, scheduler/IRQ
  interference, and USB/UHD transport internals, with counter availability,
  multiplexing, scaling, and measurement overhead reported explicitly.

The elapsed-time counter in Phase 1/2A is not a retired CPU-cycle counter.
M5Stack power samples are also intentionally external for now. Future ingestion
can align them through the shared experiment ID, realtime interval, and
`sync.csv` anchors without changing event identity.

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

When setting an override under `sudo`, use `sudo env NAME=value ...` unless the
local sudo policy explicitly preserves that variable.
