# Flight Test Quick Guide

Use the same source commit for the gNB and nrUE. The n28 X310 flight examples
and capability XML are in
[`targets/PROJECTS/GENERIC-NR-5GC/CONF`](../../targets/PROJECTS/GENERIC-NR-5GC/CONF).
Adapt the radio/core addresses and example subscriber to your installation.
The core must already be running and its subscriber must match the UE.

Run from the **repository root**; the UE example references the capability XML
relative to that directory:

```sh
# gNB, with core containers on this host
sudo cmake_targets/ran_build/build/nr-softmodem \
  -O targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.n28.25prb.x310.flight.conf \
  --flight logx pcap report recovery traffic \
  --flight-run-id takeoff1 --flight-traffic-bind 192.168.70.129

# nrUE, on the other host
sudo cmake_targets/ran_build/build/nr-uesoftmodem \
  -O targets/PROJECTS/GENERIC-NR-5GC/CONF/nrue.n28.25prb.x310.flight.conf \
  --flight log pcap report recovery traffic \
  --flight-run-id takeoff1 --flight-traffic-peer 192.168.70.129 \
  --flight-traffic-direction both --flight-traffic-rate 1.1M
```

Replace `192.168.70.129` with the gNB-host address reachable through your core's
UE data path. To run your own iperf3, remove `traffic` and its `flight-traffic-*`
options from both commands. Avoid an existing iperf3 listener on port 5201 when
using automatic traffic. Both directions offer 1.1 Mbit/s each in this example.
See [traffic details](README.md#automatic-udp-traffic).

| Selection | What You Get |
|---|---|
| `log` | RAN logs, timestamped radio/protocol events and host observations. |
| `logx` | Everything in `log`, plus local AMF, SMF and UPF logs. |
| `pcap` | Bounded packet capture, using the default connection/traffic filter. |
| `report` | A standalone HTML report after the run. |
| `recovery` | Automatic process recovery under the documented conditions. |
| `traffic` | Automatic iperf3 and receiver-delivery measurements. |

Order does not matter. Every enabled run prints and saves a summary automatically.
The same selections can be stored in the top-level config, for example
`flight = "logx pcap report recovery";`. CLI values override config values.
[Feature rules](README.md#quick-start-and-flight-features).

For just AMF logs, add `--flight-core-logs amf`; explicit sources replace the
three defaults. `amf smf upf`, `amf,smf,upf` and the equivalent `docker:oai-...`
list are accepted. Any `flight-core-*` option also enables core collection.
A remote core needs the standalone collector on that machine; `logx` does not
log in remotely. [Core collection](README.md#core-logs-and-packet-capture).

`--flight-pcap-profile all` captures every packet instead of the default filter.
Radio sample traffic can fill its quota quickly. Any `flight-pcap-*` option
implicitly enables capture; custom interface/filter values override defaults.
Defaults: 512 MiB, eight files, 3600 seconds, full packets up to 262144 bytes.
Increase the byte/time limits for longer captures; a quota stops capture without
overwriting previous files. [Capture options](README.md#automatic-packet-capture).

`--flight-recovery-stall` defaults to 10 seconds: it bounds a lack of progress
in local radio processing, not a weak-signal interval. `--flight-recovery-attempt`
defaults to 120 seconds: it bounds acquiring/restoring protocol context after
synchronization or context loss. These are different from a lost-packet timer.
[Recovery conditions](README.md#automatic-ue-recovery).

Output defaults to `cmake_targets/log/FlightTests/YYYY-MM-DD`; the terminal
prints the actual capture directory. Override with `--flight-output PATH`.
Open the local `report/gNBreport.html` or `report/nrUEreport.html` in Firefox.
The automatic report contains the local role/core evidence. To compare both
endpoints, make their captures accessible on one analysis host and run:

```sh
python3 tools/flight_test/timeline.py \
  --gnb /data/FlightTests/gnb-session-directory \
  --ue /data/FlightTests/ue-session-directory \
  --core /data/FlightTests/core/auxiliary-run-directory \
  --output /data/FlightTests/takeoff1-report
```

Omit `--core` when the core logs are already inside the gNB capture. The output
directory must be new. The graphs use elapsed recording time by default, with
an option to show UTC. Enter an actual scenario start if you want scenario-relative
zero; the report does not guess it. Cross-host UTC alignment is unverified.
Expand each graph's help for metric meaning, sampling and missing-data details.
Connection and Radio Events appears first, including recorded supervisor and
gain-control decisions/results. Inspect points for reasons and requested versus
recorded results; a submitted request is not confirmed actuation. Use Full Screen
beside a graph's Hide All/Show All controls, then Escape or Exit Full Screen to
return. For a combined report beside the local gNB report, use
`--output GNB_CAPTURE/report`
with both `--gnb` and `--ue`: the combined HTML is `report/report.html`, and its
exports are under `report/combined/`. Existing combined output is never overwritten.
[Summary and report details](README.md#summaries-and-the-combined-timeline).
