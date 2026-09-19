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
|17 UE_NAS|UE instance|0 RX,1 TX,2 ITTI dispatch,3 Registration Request construction,4 reject decision|message type or -1; ITTI ID for dispatch; cause for rejection|security result for RX, outer header for TX, 5GMM mode for dispatch, security-container flag for construction, policy for reject|length or 0; registration type for construction; wait seconds for reject|5GMM state|
|18 UE_RRC_TIMER|UE instance|timer number|1 start,2 stop,3 expire|duration ms|RRC state|reserved|
|19 UE_CONTROL|UE instance|1 RRC transition,2 RLF,3 idle fallback,4 NAS reject|old state or reject cause|new state or reject policy|release cause or raw T3346 (-1 absent)|reserved or raw T3502 (-1 absent)|
|20 GNB_SLOT|gNB module|cell ID|SFN|slot|reserved|reserved|
|21 GNB_UE_BYTES|cell ID|RNTI|SFN|DL MAC SDU bytes|UL MAC SDU bytes|UL failure flag|
|22 GNB_UE_RADIO|RNTI|DL MCS|UL MCS|DL HARQ errors|UL HARQ errors|UL DTX|
|23 GNB_RA|0 initiation,1 positive feedback,2 negative feedback,3 Msg3 exhausted|RNTI|SFN*1000+slot (slot placeholder 0 for stage 3)|RA-RNTI|RAPID or Msg3 round for stage 3|TA for stage 0, maxHARQ for stage 3|
|24 GNB_UE_LINK|RNTI|scheduler normalized PH dB|scheduler PCMAX dBm|PUCCH DTX|mean RSRP or unavailable|mean SINR*10 or unavailable|
|25 GNB_DL_HARQ|RNTI|round0 count|round1 count|round2 count|round3 count|DL MAC transport bytes|
|26 GNB_UL_HARQ|RNTI|round0 count|round1 count|round2 count|round3 count|UL MAC transport bytes|
|27 UE_NAS_COUNT|UE instance|0 before RX security,1 after RX security,2 TX handoff,3 Registration Request construction|current UL NAS COUNT|current DL NAS COUNT|context presence mask: 1 integrity,2 ciphering|5GMM state|
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

## Native recovery stream, schema version 1

Each worker supplies an inherited private AF_UNIX datagram socket to a
SCHED_OTHER monitor. Once per second and at orderly shutdown it attempts one
bounded JSON snapshot containing worker PID, sequence, source CLOCK_MONOTONIC
nanoseconds, send-drop count and observed values. A separate recovery journal
records receipt time, sequence gaps, policy transitions and action outcomes.
No RT producer formats JSON, sends on the socket or writes a file.

`rx_samples` and `tx_samples` count successful UHD samples, `search_attempts`
counts completed searches, `sync_successes` counts successful synchronization,
`rrc_messages` and `nas_messages` count task dispatches. `ue_slot_inputs`
counts submissions from the synchronized main loop; `ue_dl_completed` and
`ue_tx_completed` count completed PHY worker calls (not decoded packets). `rrc_state` is the OAI
enum; `pdu_accepts` counts decoded/config-matched accepts. `pdu_active` denotes
accepted control-plane context and is cleared at RLF/idle/detach; it does not
prove a functioning user plane. `rrc_hold_until_ns` is a monotonic lower bound
from observed T302/T301/T311 holds. Fields are independently sampled, not one
compound-coherent protocol snapshot. Unobserved fields are omitted.

`nas_reject` is one coherent uint64: generation bits 63..56, cause 55..48,
T3502-presence flag 47, policy 46..40, raw T3502 39..32, minimum wait seconds 31..0. Policy 1 permits
the implemented retry subset, 2 inhibits retry, and 3 marks unsupported or
malformed restrictions and also inhibits retry. Generation is worker-local.
The session retains the last explicitly supplied T3502 across worker replacements;
omission does not replace it with the default. The initial default octet 0x42
represents twelve minutes. A value does not encode
a subscription identity. The journal decodes this word before policy decisions.

RX NAS events before type extraction have type -1 and the security result;
after successful validation/decryption a second event carries the type. TX
ciphertext is never interpreted as a type. Registration Request construction
with the security-container flag is not a second wire transmission. Numeric
metadata does not expose NAS payloads, keys, SUCI, GUTI or IMSI.

UE_NAS_COUNT records numeric counters and context-presence booleans only, never key bytes or pointers. Counts are local values at the named phase, not necessarily the sequence number of the adjacent wire message: TX generation may already have incremented UL COUNT, and RX validation may update DL COUNT. Compare before/after RX and construction/handoff events within one worker to diagnose context resets, replay or repeated registration. This is diagnostic evidence, not a security validation verdict.

`drb_context_active` is a native 0/1 observation of configured or resumed DRB
control-plane context. It clears on release, suspension, RLF and reset. Recovery
requires a prior PDU acceptance in the same worker before treating this as
restoration; it is not proof of packet or application delivery.
