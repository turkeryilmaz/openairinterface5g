# Numeric flight events, schema version 1

Capture is off unless OAI_FLIGHT_RECORDER_DIR names an existing directory at log initialization. capture.py enables it in a new private run directory. Radio algorithms are unchanged. Numeric capture is implemented for x86_64 and aarch64 with always-lock-free producer atomics. Other OAI-supported architectures compile an inert API and report capture unavailable when requested; use a 64-bit OS/build on the CM5 and verify recorder files before takeoff.

Every event has event, a–f, ring, sequence, mono_ns and realtime_ns. Sequence numbers are global across all producer rings; gaps within one ring are expected when other rings emit. Time fields are integer nanoseconds; realtime is Unix UTC epoch. Monotonic values compare only within one boot. INT64_MIN means unavailable. Ordinary stdout timestamps are receipt times; numeric timestamps are captured at the source event. File/sequence order across threads can differ from timestamp order.

| ID/name | a | b | c | d | e | f |
|---|---|---|---|---|---|---|
|10 UE_SYNC|UE instance|1 success / 0 failure|HW slot offset on success|CFO Hz on success|RX sample offset on success|reserved|
|11 UE_AGC|UE instance|requested delta dB|adjusted requested delta dB|configured gain minus offset dB|backend result|reserved|
|12 UE_MEASUREMENTS|UE instance|SFN*1000+slot|physical cell ID|SSB index|L1 RSRP dBm|L1 SINR dB|
|13 UE_RA|UE instance|SFN*1000+slot or -1|C-RNTI/temporary RNTI|1 success,2 contention fail,3 RAR fail|RA type on success/state on failure|CFRA on success|
|14 UE_RRC|UE instance|connected state enum|reserved|reserved|reserved|reserved|
|15 UE_PDU|UE instance|PDU session ID|PDU type enum|1 decoded/config-matched accept|reserved|reserved|
|16 UE_TA|UE instance|SFN*1000+slot|C-RNTI|TA command|TA type enum|reserved|
|20 GNB_SLOT|gNB module|cell ID|SFN|slot|reserved|reserved|
|21 GNB_UE_BYTES|cell ID|RNTI|SFN|DL MAC SDU bytes|UL MAC SDU bytes|UL failure flag|
|22 GNB_UE_RADIO|RNTI|DL MCS|UL MCS|DL HARQ errors|UL HARQ errors|UL DTX|
|23 GNB_RA|0 initiation,1 positive feedback,2 negative feedback,3 Msg3 exhausted|RNTI|SFN*1000+slot (slot placeholder 0 for stage 3)|RA-RNTI|RAPID or Msg3 round for stage 3|TA for stage 0, maxHARQ for stage 3|
|24 GNB_UE_LINK|RNTI|scheduler normalized PH dB|scheduler PCMAX dBm|PUCCH DTX|mean RSRP or unavailable|mean SINR*10 or unavailable|
|25 GNB_DL_HARQ|RNTI|round0 count|round1 count|round2 count|round3 count|DL MAC transport bytes|
|26 GNB_UL_HARQ|RNTI|round0 count|round1 count|round2 count|round3 count|UL MAC transport bytes|
|30 RADIO_RX|device-type enum|UHD sample ticks|requested samples|returned samples|RX error enum|has_time_spec|
|31 RADIO_TX|device-type enum|scheduled sample ticks|requested samples|returned samples|burst flags (direct path)|0 direct / 1 TX worker|

Counters are cumulative unless stated otherwise. Start new rate segments after process restart, counter reset or RNTI reuse. SFN wraps every 10.24 s. UE_AGC reports existing configured/backend values, not calibrated hardware readback. A PDU accept is not proof of working traffic. MAC bytes are not useful video payload. RRC/RA events are selected milestones, not a complete state-machine trace. ID 1 is reserved for lifecycle.

GNB snapshots run under the existing scheduler lock every 64 frames (640 ms), for at most 16 UEs/cell, with four HARQ rounds. This campaign uses one UE; larger deployments need expanded coverage. Link measurement accumulators may be empty/reset: unavailable is explicit. Scheduler PH is normalized using PRB/MCS factors; it is not the raw received PHR. PH/PCMAX can still hold initialization defaults before a PHR is received (the scheduler treats PCMAX zero as unavailable). Do not interpret these default zeros as measured headroom or power. UE serving SSB measurements follow existing L1 reporting. Successful USRP calls are sampled once per 1,024 calls using call counters independent of sample-block sizes; anomalous returns and RX errors are captured. Device type is not a serial number; multi-radio runs need an additional device identifier.

This version does not drain UHD TX asynchronous metadata. TX sample acceptance does not prove RF delivery; stderr L/U/O characters are not authoritative late-packet counts or restart triggers. Additional ordinary OAI logs are retained subject to privacy/volume limits.

The normal-priority writer drains 64 lifetime thread rings of 1,024 records to sequential files of at most 16 MiB. Files are never overwritten. Total-byte caps are optional (zero by default), with a configurable 512 MiB free-space reserve. Overflow/no-slot/loss is explicit. Use a separate capture directory for each process. The eight internal file-descriptor slots do not limit retained file count. No subscriber keys, packet payloads or IQ arrays enter numeric records.

Clock correlations bound clock-read skew, not UTC accuracy. A missing clean footer can mean crash, kill, capture failure or still-running process. Power loss can lose buffered events. Producer hooks add clocks/atomics/fixed copies, with no added allocation, locks, printf or file I/O. Native tests do not qualify the observer effect on CM5/B205mini: compare disabled/enabled stationary operation before takeoff.


## Sequential retention (September startup integration)

New captures retain successive numbered files without overwriting earlier data.
The filename's final decimal number is no longer limited to 0..7. `file_slot`
remains an internal descriptor slot and may repeat across files; the complete
filename identifies the file. New headers have `overwrites_available=0`.
The decoder also accepts older eight-slot captures and their overwrite markers.
Event IDs, payload meanings, timestamps and record schema version remain unchanged.
A missing clean footer or a recording-disabled diagnostic must not be treated as
complete recording. Unknown event IDs preserve all numeric fields for later decoding.
