# OAI nrUE/gNB archival profiler

This directory contains the offline utilities for the explicitly enabled
nrUE/gNB boundary-event profiler. Profiling remains disabled unless `-P` or
`OAI_PROFILE` enables it.

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
events.csv          Per-occurrence boundary-event durations
event_catalog.csv   Stable event ID/name mapping
sync.csv            Realtime, monotonic, and hardware-counter anchors
drops.csv           Producer-buffer drop counts
metadata.txt        Run, host, source, config-path, and lifecycle identity
settings.csv        Effective softmodem/profiler settings
host_metrics.csv    Temperature, throttling, CPU, memory, and process metrics
```

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
`runs.csv`, `pairs.csv`, and `host_summary.csv`. It uses only the Python
standard library.

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
