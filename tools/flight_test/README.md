# OAI flight capture

`capture.py` supervises exactly one gNB or nrUE executable using only the Python standard library.
It creates a private run, enables the paired numeric C recorder, redacts child output, and writes bounded host observations.

## Build and run

Build the OAI executable through the normal native OAI CMake workflow. This tool has no separate build or dependency installation step. A production runtime needs the respective executable, its required configuration and adjacent runtime plugins: `params_libconfig` and `oai_usrpdevif`; include `rfsimulator` for software tests.
Use an absolute executable after `--`; configuration paths supplied to `--config`, `-O`, or child `--config` must also be absolute.

```sh
# gNB
python3 tools/flight_test/capture.py --role gnb --output /var/lib/oai-flight/captures --repo /opt/oai/current --config /etc/oai/gnb-flight.conf --core-ip "$OAI_FLIGHT_CORE_IP" -- /opt/oai/current/cmake_targets/ran_build/build/nr-softmodem -O /etc/oai/gnb-flight.conf
# UE
python3 tools/flight_test/capture.py --role ue --output /var/lib/oai-flight/captures --repo /opt/oai/current --config /etc/oai/nrue-flight.conf --core-ip "$OAI_FLIGHT_CORE_IP" --interface oaitun_ue1 -- /opt/oai/current/cmake_targets/ran_build/build/nr-uesoftmodem -O /etc/oai/nrue-flight.conf
```

`--config` is fingerprinted only and does not inject or rewrite child arguments. There is no built-in site or core address.
The default interface is `oaitun_ue1`. Add `--probe-ping` only for a selected remote core endpoint, never the UE address; ICMP and interface counters do not prove a PDU session or application traffic.

To collect explicit GPSD observations, add `--gpsd HOST:PORT`. It retains GNSS UTC, fix mode, accuracy, and receipt clocks; it never changes the wall-clock basis after a fix.

```text
OAI_FLIGHT_RECORDER_DIR=<run>/recorder
OAI_FLIGHT_RECORDER_MAX_BYTES=134217728
```

The recorder variables are supplied by default and removed by `--disable-recorder`. When enabled, `--recorder-budget` must be in `8192..134217728` bytes. The paired C writer owns `recorder/`, limits itself to eight 16 MiB files, and never stops producers.
Every child uses `<run>/working` mode 0700, containing legacy current-directory OAI statistics such as `nrMAC_stats.txt`. Relative configuration paths are rejected instead of falling back to a shared repository CWD. The child environment prepends the resolved executable parent to `LD_LIBRARY_PATH` while preserving its prior value, allowing adjacent build plugins to load without reopening the shared CWD.

## Files, limits, and privacy

The unique run directory is mode 0700 and its files are mode 0600.

| Path | Contents | Default bound |
| --- | --- | --- |
| `metadata.json` | safe command, binary/config hashes, source identity | 512 KiB |
| `stdout.NNNN.log`, `stderr.NNNN.log` | continuously drained redacted output | combined 256 MiB, 8 MiB chunks |
| `recorder/` | numeric C recorder output | 128 MiB contract, eight files |
| `host.NNNN.log` | JSON Lines host and health observations | 64 MiB, 4 MiB chunks |
| `status.json` | final exit, loss, stop, and shutdown counters | 512 KiB |

The three data categories total 448 MiB; metadata/status are independently hard limited. ENOSPC, writer loss, quota loss, and recorder-contract excess produce `capture_unhealthy=` and a false final health flag while the supervisor continues draining child pipes.
It does not delete recorder evidence or stop radio to recover capture storage.

No raw configuration, environment, Git diff, raw GPSD message, or packet payload is retained by the numeric recorder or host collector. Source identity saves only commit, a streamed dirty-path count, and a streamed tracked-patch SHA-256; names and patch bytes are never saved. Runtime provenance fingerprints only fixed adjacent module names: `libparams_libconfig.so`, `liboai_usrpdevif.so`, `liboai_device.so`, and `librfsimulator.so`; a missing module is unavailable.
Sensitive argv values are replaced. ANSI escapes are normalized before checking every child-output line. Sensitive key/SIM/NAS/authentication blocks and their hexdump continuations are dropped until a blank line or a recognized non-sensitive OAI module diagnostic; bare or module-prefixed hexadecimal/address rows remain suppressed. Oversize unterminated lines are discarded at 8192 bytes with counters. This is not an exhaustive classification of OAI protocol dumps: keep child stdout/stderr private and review it before export. Unknown continuation text and unrelated diagnostics inside a sensitive block can be suppressed; use independent numeric PDU/RRC events and supervisor exit status when interpreting a run.
Host samples contain UTC receipt time, monotonic time, optional monotonic-raw time, and paired-clock uncertainty. These are capture receipt timestamps, not source event timestamps. Missing procfs, sysfs, chrony, route, or GPSD data remains unavailable rather than zero.
Raw video, payload, application sender/receiver counters, and end-to-end latency are not captured. A 1 Mbps UL/DL flight claim needs an external video/application statistics plan and separate time-correlation evidence.

## Observer-only health and shutdown

`flight_health.py` emits `starting`, `acquiring`, `healthy`, `degraded`, `exited`, and `unavailable`. It uses monotonic startup grace and three direct process/interface samples. For a UE, a tunnel absent from startup remains `acquiring`, not wedged. For a gNB, the configured interface is only a host observation; the default UE tunnel is not acquisition evidence. Events label direct evidence versus inference.
Recommendations expose backoff and budget fields, but current action is always `none` and restart eligibility is false. Interface activity and ping never trigger a restart.

TERM and INT are forwarded only to the child process group. After `--stop-timeout` (10 seconds by default), KILL is sent only to that same group. Normal child exit codes are preserved; signal exits return `128 + signal`, and launch errors return 127 or 126.

## Deterministic check

```sh
python3 -B tools/flight_test/tests/test_capture.py
```

The stdlib fixtures cover end-to-end exit/output propagation, launch failure, TERM forwarding, giant lines, ANSI and multi-line key-block redaction, bounded rotation, unavailable GPSD/clock tools, grace/hysteresis, and isolation from an unrelated process.

## Decode numeric events and mark operator events

Decode only the numeric recorder files after the run has stopped. The decoder writes `events.csv` and `summary.json` under its explicit output directory:

```sh
python3 tools/flight_test/decode_events.py <run>/recorder --output <run>/decoded
```

Add a bounded operator marker to an existing private run when an operator observes a flight event:

```sh
python3 tools/flight_test/mark_event.py --run <run> --flight flight01 --event takeoff
```

## Example systemd unit

`systemd/oai-flight-capture@.service` is an uninstalled template; it neither installs nor enables a service. It loads the matching role-specific `/etc/oai/flight/gnb.env` or `/etc/oai/flight/ue.env` file. Use absolute paths and no subscriber or authentication material:

```text
# gnb.env
OAI_FLIGHT_CONFIG=/etc/oai/gnb-flight.conf
OAI_FLIGHT_BINARY=/opt/oai/current/cmake_targets/ran_build/build/nr-softmodem
OAI_FLIGHT_CORE_IP=<REMOTE_CORE_IP>
OAI_FLIGHT_INTERFACE=oaitun_ue1
OAI_FLIGHT_EXTRA_ARGS=--sa
```

```text
# ue.env
OAI_FLIGHT_CONFIG=/etc/oai/nrue-flight.conf
OAI_FLIGHT_BINARY=/opt/oai/current/cmake_targets/ran_build/build/nr-uesoftmodem
OAI_FLIGHT_CORE_IP=<REMOTE_CORE_IP>
OAI_FLIGHT_INTERFACE=oaitun_ue1
OAI_FLIGHT_EXTRA_ARGS=--sa
```

The unit passes both `--config ${OAI_FLIGHT_CONFIG}` and `-O ${OAI_FLIGHT_CONFIG}`; `$OAI_FLIGHT_EXTRA_ARGS` is intentionally split into arguments by systemd. It defaults to `Restart=no`, matching the observer-only capture package. A site may opt in to `Restart=on-failure` only for true process exits after a successful ground gate; retain `RestartSec=30s` and the three-start lifetime budget, inspect evidence, and explicitly reset the budget after exhaustion. Replace the documentation-only address with the authorized remote core endpoint. Before any installation, provision the fixed `oai` user with controlled USB/USRP device permissions (for example through site udev policy), and verify the declared `CAP_SYS_NICE`, `CAP_NET_ADMIN`, `CAP_NET_RAW`, and `CAP_IPC_LOCK` capabilities plus `LimitRTPRIO=99` and unlimited memlock meet the host policy. `CAP_NET_ADMIN` is relevant to UE tunnel creation; no privilege, device, or system setting is changed by this repository template.
