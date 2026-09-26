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
|40 RADIO_GAIN|radio binding (currently 0)|result generation|requested gain mdB, or INT64_MIN unavailable|RX device readback mdB, when valid|TX device readback mdB, when valid|status bits 0..7; operation bits 8..15; valid flags from bit 16|
|41 RADIO_GAIN_TIME|radio binding (currently 0)|result generation|request ID, including 0 at startup|begin device ticks, or INT64_MIN unavailable|end device ticks, or INT64_MIN unavailable|valid flags: bit 0 RX readback, bit 1 TX readback, bit 2 device bracket|
|42 RADIO_RX_LEVEL|radio binding (currently 0)|signed gain generation: positive valid sample context, negative invalid context|first received sample tick|mean complex power mdBFS, or INT64_MIN unavailable|peak component mdBFS, or INT64_MIN unavailable|near-rail component count in upper 32 bits; sampled component count in lower 32 bits|
|43 UE_TX_POWER_REQUEST|UE instance|SFN*1000+slot|channel: 1 PRACH,2 PUSCH,3 PUCCH,4 SRS|current channel PDU request dBm|actual digital generator amplitude/reference|channel-specific packed resource|
|44 GNB_TX_REFERENCE|SFN*1000+slot|configured ssPBCH power dBm|actual TX_AMP at SSB generation|SSB index|SSB start symbol|antenna port bits 0..15; beam index 16..31; SSB start subcarrier 32..63|
|45 RADIO_RX_DECISION|source: 1 UE_SSB,2 GNB_PUSCH,3 UE_SEARCH,4 HEADROOM|sample-context gain generation|exact context end sample tick|mean level mdBFS, or INT64_MIN unavailable|candidate RX gain mdB, or INT64_MIN unavailable|reason bits 0..7; would-change bit 8; submitted bit 9; activity-valid bit 10; search-failed bit 11; phase-present bit 12; tracking bit 13 (otherwise acquisition when present); preserve unknown bits|
|46 RADIO_RX_DECISION_INPUT|same source|same generation|same exact context end sample tick|reported RX gain mdB, or INT64_MIN unavailable|peak component mdBFS, or INT64_MIN unavailable|error mdB only for TRACK_LEVEL/DEADBAND, otherwise INT64_MIN|
|49 RADIO_TX_POWER|channel: 1 PRACH,2 PUSCH,3 PUCCH,4 SRS|SFN*1000+slot|status bits 0..7 and applied bit 8|requested power mdBm|internal estimated output mdBm only for status 0, otherwise INT64_MIN|amplitude coefficient Q30 only for status 0, otherwise INT64_MIN|
|50 RADIO_TX_POWER_SAMPLES|same channel|same SFN*1000+slot|complex sample count|input sum(I²+Q²) raw integer codes|output sum(I²+Q²) only for status 0, otherwise INT64_MIN|input peak component upper 32 bits; output peak component lower 32 bits|
|51 RADIO_TX_REJECT|radio binding (currently 0)|SFN*1000+slot|channel: 1 PRACH,2 PUSCH,3 PUCCH,4 SRS or 0 generic|reason: 1 layout,2 span,3 overlap,4 power limit,5 profile|actuation requested bool|reserved 0|
|52 RADIO_TX_POWER_QUALITY|same channel|same SFN*1000+slot|quantization power error mdB only for status 0, otherwise INT64_MIN|quantization EVM parts per billion only for status 0, otherwise INT64_MIN|profile uncertainty mdB, or INT64_MIN unavailable|component full-scale integer code|
|58 RADIO_TX_RELATIVE_POWER|channel: 1 PRACH,2 PUSCH,3 PUCCH,4 SRS or 0 generic|SFN*1000+slot|status bits 0..7 and applied bit 8|selected nominal mdB|requested digital dBFS mdB only for status 0, otherwise INT64_MIN|realized digital dBFS mdB only for status 0, otherwise INT64_MIN|
|59 RADIO_TX_RELATIVE_QUALITY|same channel|same SFN*1000+slot|complex sample count|quantization power error mdB only for status 0, otherwise INT64_MIN|quantization EVM parts per billion only for status 0, otherwise INT64_MIN|amplitude coefficient Q30 only for status 0, otherwise INT64_MIN|
|60 RADIO_TX_RELATIVE_CONFIG|role: 0 UE,1 gNB|component full-scale integer code|digital reference mdBFS|minimum nominal|maximum nominal|fixed analog TX gain mdB, or INT64_MIN unavailable|
|61 RADIO_TX_RELATIVE_ERASURE|channel: 1 PRACH,2 PUSCH,3 PUCCH,4 SRS or 0 generic|SFN*1000+slot|relative mapping status that caused the erasure|zeroed complex sample count|reserved 0|reserved 0|
|62 UE_TX_RELATIVE_BOUNDS|channel: 1 PRACH,2 PUSCH,3 PUCCH,4 SRS|SFN*1000+slot or -1|effective minimum nominal|effective maximum nominal|MAC requested nominal|MAC selected nominal|
|63 RADIO_TX_RELATIVE_SAMPLES|channel: 1 PRACH,2 PUSCH,3 PUCCH,4 SRS or 0 generic|SFN*1000+slot|complex sample count|input sum(I²+Q²) raw integer codes|output sum(I²+Q²) only for status 0, otherwise INT64_MIN|input peak component upper 32 bits; output peak component lower 32 bits|

`RADIO_GAIN` status values are 0 OK, 1 BUSY, 2 UNSUPPORTED, 3 INVALID,
4 STALE, 5 CLOSED, 6 BACKEND_ERROR and 7 TX_PENDING. Operation values are
0 SET_RX, 1 SET_TX and 2 RETUNE. The valid-flag bits are carried unchanged in
both result records; a decoder must retain unknown bits rather than treating
them as a known validity state. A requested-gain field is valid only for a
SET_RX or SET_TX request with nonzero request ID. The initial startup record has
request ID 0 and its requested-gain field is INT64_MIN, meaning unavailable.

The RX/TX readbacks are device-reported gain settings, not RF-power
calibration. `RADIO_GAIN_TIME` begin/end values bracket device interaction only;
they do not establish gain settling, an on-air boundary, or an RF timestamp.
The radio-gain decoder writes generic `events.csv` as before, plus typed
`radio_gain.csv` result rows and `radio_rx_level.csv` sampler rows. It joins one
result and one time record only when their `(radio, generation)` key is unique;
missing or ambiguous metadata remains unavailable. It does not invent a source
timestamp from the bracket values. The decoder helper uses `None` for unavailable
values; the corresponding CSV cells are blank, never synthetic zeroes.

`RADIO_RX_LEVEL` is emitted only on every 32nd successful one-antenna read with
a valid buffer. It samples at most 64 evenly spaced complex samples (therefore
at most 128 components), and its near-rail count applies only to those sampled
components. It is not a claim about clipping across the entire read buffer. A
negative signed generation means the sample context was invalid; its absolute
value remains the associated gain generation. A zero generation is also an
invalid context.

RADIO_RX_DECISION records selected RX policy evaluations: a changed proposal,
a reason transition, or the periodic one-second record. Source values are 1
UE_SSB, 2 GNB_PUSCH, 3 UE_SEARCH, and 4 HEADROOM. Reasons are 0 HOLD_INVALID,
1 HOLD_UNSUPPORTED, 2 HOLD_STALE, 3 HOLD_TRANSITION, 4 HOLD_INACTIVE,
5 HOLD_DEADBAND, 6 HOLD_COOLDOWN, 7 HOLD_LIMIT, 8 REDUCE_OVERLOAD,
9 TRACK_LEVEL, and 10 SEARCH_STEP. Its packed flags retain the raw numeric word
and every unknown bit in radio_rx_decisions.csv; known bits describe a proposal
(would_change), successful policy submission (submitted), input activity,
and failed search. They do not prove a completed driver transaction, RF gain
change, settled gain, or calibrated connector level.

If phase-present is absent, the phase is unavailable (including older captures);
it must not be inferred as acquisition from a zero tracking bit.

RADIO_RX_DECISION_INPUT is the co-emitted input detail for the same source,
gain generation, and exact context-end sample tick. It carries the reported
device RX gain and sampled peak independently of whether its decision result is
valid. Its error is meaningful only when the paired decision reason is
TRACK_LEVEL or HOLD_DEADBAND; other reasons retain the raw sentinel and have a
blank typed error. The decoder writes one decision row to
radio_rx_decisions.csv and joins input fields only when records share one
recorder ring, source, generation, and end tick and form a strictly monotonic
decision/input sequence. Missing inputs are marked missing; unequal record
counts, non-monotonic sequence numbers, or any nonalternating decision/input
order are marked ambiguous. In either case, all input-derived columns remain
blank. It never constructs an input
from a similarly keyed record in another thread/ring.

The serving reference uses a raw unitary-FFT per-bin mean divided by converter
full-scale squared. It is a full-grid-equivalent level: do not divide it by N
and do not apply an occupancy normalization. Inactive and search paths use their
corresponding raw-read mean. HEADROOM evaluates 64 samples per read, independently
of the ID 42 sampler, which still emits one record in 32 successful reads with
its existing meaning unchanged. These fields are engineering references only:
they do not measure on-air RF power or establish an ADC protection guarantee.

The initial RX policy constants are target -18 dBFS, 3 dB deadband, 3 dB maximum
and search steps, 200 ms cooldown, and 0.25 filter weight. They remain
engineering settings pending RF characterization; the current flight
configuration observes decisions and does not actuate them. The
agc-rx-settle-us default of 20000 microseconds is a configurable engineering
quarantine after a device transaction. It is not a sample-exact analog-settling
measurement or a calibration.

UE_TX_POWER_REQUEST is emitted once after each PRACH, PUSCH, PUCCH, or SRS
generator completes. Its dBm value is the current channel PDU request, and its
generator value is the exact digital amplitude/reference used by that generator:
PRACH, PUCCH, and SRS carry the AMP/tx_amp argument; PUSCH carries 32767, the
fixed Q15 nr_modulation() constellation reference. Neither is a waveform power
measurement, RF gain, calibrated RF output, or a power-control actuation.
For ID 43, f packs PRACH as freq_msg1 bits 32..47, start symbol 24..31, num_ra
16..23, format 8..15, and preamble 0..7. PUSCH packs RB start 0..15, RB size
16..31, start symbol 32..39, number of symbols 40..47, modulation order 48..55,
and layers 56..63. PUCCH uses the same first four fields for PRB start, PRB size,
start symbol, and number of symbols; format is 48..55 and the active PDU index
is 56..63. SRS packs BWP start 0..15, BWP size 16..31, time start 32..39, actual
number of SRS symbols 40..47, frequency position 48..55, and actual antenna-port
count 56..63. The typed decoder labels these common columns resource_first,
resource_count, start_symbol, symbol_count, detail_0, and detail_1; their
channel-specific meanings are the packing just defined.

GNB_TX_REFERENCE is emitted after SSB generation. It carries the configured
ss_pbch_power and the generator's current TX_AMP, together with SSB placement.
It does not report calibrated, settled, measured, or on-air RF power. A record
uses no payload or IQ data. Zero is a real numeric field value; only INT64_MIN
means unavailable. The decoder writes generic events.csv plus typed
ue_tx_power.csv and gnb_tx_reference.csv files.

Counters are cumulative unless stated otherwise. Start new rate segments after process restart, counter reset or RNTI reuse. SFN wraps every 10.24 s. UE_AGC reports existing configured/backend values, not calibrated hardware readback. A PDU accept is not proof of working traffic. MAC bytes are not useful video payload. RRC/RA events are selected milestones, not a complete state-machine trace. ID 1 is reserved for lifecycle.

GNB snapshots run under the existing scheduler lock every 64 frames (640 ms), for at most 16 UEs/cell, with four HARQ rounds. This campaign uses one UE; larger deployments need expanded coverage. Link measurement accumulators may be empty/reset: unavailable is explicit. Scheduler PH is normalized using PRB/MCS factors; it is not the raw received PHR. PH/PCMAX can still hold initialization defaults before a PHR is received (the scheduler treats PCMAX zero as unavailable). Do not interpret these default zeros as measured headroom or power. UE serving SSB measurements follow existing L1 reporting. Successful USRP calls are sampled once per 1,024 calls using call counters independent of sample-block sizes; anomalous returns and RX errors are captured. Device type is not a serial number; multi-radio runs need an additional device identifier.

The private radio-health channel drains UHD TX asynchronous metadata and retains typed counters and last-event identity in `radio_health.*.log`. Numeric RADIO_TX events separately describe sample acceptance, which does not prove RF delivery. Stderr L/U/O characters are not authoritative late-packet counts or restart triggers. Additional ordinary OAI logs are retained subject to privacy/volume limits.

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


### TX sample buffers (47–48)

Managed NR gain bindings with flight logging emit these records for the first and
then every 64th admitted TX callback. ID47 scans every I/Q component of that
selected buffer before the backend converter, with an explicit 65,536-complex-
sample cap. Unselected/oversized buffers have no measurement. This is not complete
capture coverage and absence of an over-range observation does not prove absence
in other buffers. Disabled flight logging performs no scan.

- **47 RADIO_TX_LEVEL:** a=radio(0), b=requested first device-sample timestamp,
  c=complex sample count, d=converter component full scale, e=sum(I²+Q²) in raw
  integer codes over the whole buffer, f=(maximum absolute component <<32) |
  number of components outside [-full_scale, full_scale-1]. The sum uses64bits.
- **48 RADIO_TX_LEVEL_STATE:** a,b,c match47; d=backend return (including short
  count or negative error), e=pre-call hardware TX-gain readback in millidB,
  f=settings generation. Unknown e/f use INT64_MIN. The current TX-admission
  fence excludes gain-mapping retune during streaming; RX-only generations may
  advance without changing TX gain.

The decoder writes `radio_tx_level.csv`, joining only an unambiguous ordered
pair in one ring with equal radio/timestamp/count. Missing state is unavailable.
Mean digital power is e/(c*d²); this includes any silence in the selected buffer,
not only active NR symbols. A zero buffer has zero linear power and unavailable
finite dBFS. This is sample evidence before conversion, not measured RF power.
Backend acceptance (which may enqueue samples) does not prove physical emission.
Neither the scan nor logging modifies TX data, gains or recovery decisions.

### TX channel actuation evidence (49–52)

`RADIO_TX_POWER` is emitted by the managed channel-scale path for a PRACH,
PUSCH, PUCCH, or SRS request. Status values are 0 OK, 1 INVALID,
2 UNQUALIFIED, 3 MAPPING_REJECTED, 4 HEADROOM, and 5 QUANTIZATION. The
`applied` bit means that the software mapping wrote its selected samples; an OK
record with `applied=0` is an observe calculation and does not establish a
waveform mutation. It also does not prove a backend submission, RF emission,
connector power, gain settling, or calibrated output. The `estimated output`
value is the internal sample/profile mapping estimate named by the producer; it
is not a measured or emitted RF-power value. The Q30 coefficient is likewise an
internal digital scale.

`RADIO_TX_POWER_SAMPLES` records raw complex-sample energy and component peaks
for the exact channel span inspected by that invocation. Its input fields are
available for any status. Its output energy is valid only when the matching
power status is OK; a blank decoder cell is unavailable, never zero. The packed
peaks retain their raw integer values. `RADIO_TX_POWER_QUALITY` carries the
mapping's quantization fields only for an OK status. Profile uncertainty and
component full scale describe the local engineering profile used by the mapping;
they do not convert the record into a calibrated RF measurement.

`RADIO_TX_REJECT` is a separate admission/rejection record. Its reason describes
why managed TX was not admitted. `actuation requested` reports the requested
mode at the producer and can be false for observe mode; it does not establish
that RF hardware did or did not emit any samples. Unknown channels, reasons,
status bits, and boolean encodings are retained in typed CSV as raw or
`UNKNOWN_<value>` fields rather than coerced to a supported outcome.

The decoder writes one radio_tx_power.csv row for every ID 49 record and a
separate radio_tx_rejects.csv row for every ID 51 record. It joins ID 50 and
ID 52 only for an unbroken producer emission group: recorder ring, channel,
and SFN/slot context agree; sample sequence equals power sequence plus one;
and quality sequence equals power sequence plus two. The producer emits
49, then 50, then 52 synchronously, and the recorder assigns a global
sequence number to each registered-ring emit attempt before ring acceptance. A sequence gap can represent capture
loss or cross-ring interleaving; without a producer group ID either case leaves
the sidecar provenance unproven, so both evidence sets are marked ambiguous
and their typed columns remain blank. This permits sequential recorder files
when those exact sequence values continue. With no sidecars both statuses remain
missing; a partial, duplicate, or noncontiguous group is ambiguous. A negative
packed frame/slot context cannot be inverted reliably (for example, the current
generic producer uses frame and slot -1), so the decoder preserves it in
frame_slot_raw and leaves typed frame and slot blank. Generic events.csv retains
the original numeric records.

`RADIO_TX_REJECT` reason 6 (`POWER_CONTROL`) means the MAC could not calculate a supported channel-power request. The PUCCH MAC/PHY internal sentinel is `INT16_MIN`; it is not a dBm request. Absolute managed TX suppresses that channel and closes admission. Relative mode drops the unsupported PUCCH configuration before PHY; baseline waveform behavior retains its former zero-request fallback.

Event 53 `UE_TX_CONTROL` records MAC power-control context: a=channel (2 PUSCH, 3 PUCCH), b=SFN*1000+slot, c=serving-SSB pathloss dB, d=closed-loop adjustment state after calculation dB, e=provided eligible TPC delta dB, f=configured network p-Max dBm (`INT64_MIN` when absent). A provided delta can be suppressed by the existing saturation rule; it is not necessarily an applied increment. Configured p-Max is not the allocation-dependent P_CMAX. PUSCH records cover managed target-slot deferred calculations; baseline grant-time calculations have no such record. PUCCH records cover supported common and dedicated calculations. These rows are calculation evidence, not proof of PHY generation or RF emission. Compare channel/frame/slot and monotonic time with waveform records; the decoder does not invent joins across SFN wraps or producer rings.

Event 54 `UE_PATHLOSS_STATE` records the first observed serving-SSB pathloss availability and subsequent availability transitions at the UE UL scheduler. It does not record a transmitted channel or a new power request. Fields: a=availability (0/1), b=SSB index, c=RSRP dBm (`INT64_MIN` when the index or measurement is unavailable), d=configured SS-PBCH reference power dBm/RE, e=checked pathloss dB (`INT64_MIN` when unavailable), f=UE MAC state. The decoder writes `ue_pathloss_state.csv` with blank missing values. Unavailable pathloss defers new UL channel scheduling while elapsed protocol timers continue; subsequent valid measurement allows scheduling again. Recorder loss or starting capture after an observation can leave the transition history incomplete. Availability is a local arithmetic/input-validity condition, not proof of freshness, correct serving-cell association, RF emission, or calibration.


Events 55/56 record the actual serving-SSB PHY measurement qualification, including early rejection. Event55 `UE_SSB_MEASUREMENT`: a=SFN*1000+slot, b=SSBindex, c=gain generation (0 when managed context absent), d=raw mean squared SSS FFT-bin magnitude (digital units), e=accepted RSRP dBm or `INT64_MIN` on rejection, f=flags. Flags bits0..7 are respectively context present, raw gain context valid, level summary valid, generation current, gain normalization eligible, noise snapshot checked, noise snapshot current, and measurement accepted. Accepted means PHY eligible for reporting; it does not acknowledge MAC callback/table update. Noise-current has meaning only when noise-checked is true. Bits8, 9 and 10 respectively record PBCH decode checked, PBCH decode successful, and PBCH confirmation required. A managed measurement is committed only after the matching PBCH succeeds and gain/noise qualification passes. A stronger candidate that was not selected for decoding can be logged with confirmation required but not checked. Success without a check, or acceptance without required confirmation, is invalid evidence. Earlier records and the unchanged absent-context path have these three bits clear; that does not establish PBCH confirmation. Rejected RSRP stays blank, never zero or a fabricated value.

The immediately following event56 `UE_SSB_MEASUREMENT_CONTEXT` carries a=generation, b=first device sample tick, c=end-exclusive device sample tick, d=reported gain millidB, e=sampled peak component millidBFS, f=(near-rail component count<<32)|sampled component count. Missing gain/ticks/peak use `INT64_MIN`; a zero-valued peak in dBFS means full scale, not missing. These summaries come from up to64 evenly-spaced complex samples per radio read, conservatively aggregated over every intersecting whole read. Their coverage can exceed the SSB interval. Observed near-rail samples disqualify gain-normalized measurement; absence does not certify an entirely linear/unclipped RF waveform. Raw context validity is retained so overload AGC can still lower RXgain.

The decoder writes `ue_ssb_measurements.csv` and preserves every context separately in `ue_ssb_measurement_contexts.csv`. A context joins only a unique header and unique sidecar in the same ring, with sidecar sequence exactly header+1, matching generation, and nondecreasing monotonic timestamp. Sequence gaps, cross-ring interleaving, missing headers, and duplicates cannot create a false same-generation association; unproven columns stay blank. Recorder loss therefore reduces available context rather than implying clean measurements. The existing `ue_tx_control.csv` shows the MAC pathloss actually consumed later; no cross-thread/SFN-wrap join is invented.

### Event 57: RADIO_RX_PEAK_ENVELOPE

This additive event records the shared RX peak constraint when a policy decision
is logged and its envelope is valid. Payload `a` is the RX source (as in 45), `b`
is the gain generation, `c` is the context end sample tick, `d` is the retained
peak projected at the current RX gain in milli-dBFS, `e` is that reported RX gain
in milli-dB, and `f` is the envelope release rate in milli-dB per second. Existing
event 46 continues to record the current window's sampled peak; its meaning is unchanged.

The envelope holds the strongest input-referred observed peak with instantaneous
attack and a bounded gradual release (initial engineering setting 3 dB/s). RX gain
changes do not erase that evidence. The envelope limits positive gain steps;
instantaneous raw overload evidence still controls reduction. It is neither a
calibrated input-power measurement nor proof of unobserved sample headroom.

The decoder writes `radio_rx_peak_envelope.csv` independently. It does not join
across missing events or infer that a neighboring decision exists. Compare exact
ring, generation, context and sequence evidence when interpreting a recorded
constraint. `INT64_MIN` remains unavailable, not zero; historical captures without
57 have no envelope evidence.

### Relative TX events (58--64)

These additive records describe `tx-power-mode = "relative"`, a digital envelope
selected only for managed TX. They do not revise IDs 49--52 or the existing
absolute-mode `radio_tx_power.csv`. The configuration is calibration-free:
the fixed device-reported analog gain is not RF-power calibration, and the
recorded requested/realized values in event 58 are dBFS rather than RF dBm.

For the current `AMP = 512`, component full scale 2048 configuration, event 60
records the -12.041 dBFS digital reference and current nominal limits -3..17.
Nominal 23 is fixed; the 6 dB engineering backoff creates the upper limit and
the fixed-point quality/EVM constraints create the effective lower limit. These
are digital engineering constraints, not a qualified profile, an RF output
range, conformance evidence, or a universal crest-factor certificate.

Event 62 is the UE MAC decision after intersecting its channel range with the
relative limits. It records the requested and selected nominal values, so an
analysis can distinguish ordinary MAC bounding from an exact mapper failure.
A network p-Max may reduce the applicable ceiling, but cannot change nominal 23
or re-anchor the dBFS reference. Event 58 then records the per-span mapping
result; status values are 0 OK, 1 INVALID, 2 UNQUALIFIED, 3 MAPPING_REJECTED,
4 HEADROOM, and 5 QUANTIZATION. Bit 8 says the mapping wrote selected samples.

An event 61 means a valid relative active occasion was zeroed as a whole after an
unusual peak/crest or quality preflight failure. It is an RF-gate failure for
that occasion, not proof of emission and not a radio-recovery action; subsequent
occasions remain eligible. Structural/admission failures such as an unsupported
layout or span remain managed-TX failures rather than relative erasures. Event
59 carries the mapping quality and event 63 carries input/output whole-span
sample evidence. Their unavailable output fields remain `INT64_MIN`.

The decoder writes `relative_tx.csv` as a standalone union of IDs 58--64. Each
record becomes one row with `record_type` `mapping`, `quality`, `configuration`,
`erasure`, `bounds`, `samples`, or `admission_blocked`; only its applicable columns are populated.
No cross-event join is attempted. Missing neighbors and `INT64_MIN` fields stay
blank, so this CSV never invents an RF measurement, a mapping, or complete
occasion coverage. Generic `events.csv` preserves the original raw records.

TX 49/50/52 association also accepts a bounded globally interleaved group when
every sequence in its span (at most 256 events) is uniquely present and all
intervening records belong to other producer rings. Same-ring interruptions,
duplicates, missing sequences, reordered timestamps, and larger spans remain
ambiguous. The output records `evidence_join_basis`, `evidence_group_sequence_span`,
and `evidence_interleaved_events`; this proof does not infer missing records from
a clean footer. Existing global-consecutive groups remain supported.

Event 64 `RADIO_TX_RELATIVE_GATE` records persistent UE admission blocking once
per second of host monotonic time, triggered by actual blocked scheduler calls: a=0 (UE), b=SFN*1000+slot, c=configured nominal
p-Max or INT64_MIN when absent, d=alternate p-Max or INT64_MIN when absent,
e=available digital minimum nominal, f=available digital maximum nominal.
Missing digital bounds are INT64_MIN. An alternate limit is unsupported by the
current MAC calculation; a ceiling below the digital minimum cannot admit TX.
This record identifies a blocked opportunity, not an emitted or erased waveform.
The decoder preserves it as an independent `admission_blocked` row.

For event 63, an unavailable output-energy field (`e == INT64_MIN`) also makes
the packed output peak unavailable. Its raw zero bits do not establish a zero
peak, especially in observe mode where failed preflight leaves samples intact.
The decoder keeps both output fields blank without requiring a neighboring
mapping record.
