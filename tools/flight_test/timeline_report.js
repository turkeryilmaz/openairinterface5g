// SPDX-License-Identifier: LicenseRef-CSSL-1.0
'use strict';
const series = JSON.parse(document.getElementById('data').textContent);
const display = JSON.parse(document.getElementById('display-meta').textContent);
const root = document.getElementById('plots');
const sourceNames = new Map();
const roleCounts = new Map();
for (const s of series) {
  const parts = s.identity || s.label.split(' / ');
  const [capture, role, host, metric, ...entity] = parts;
  if (!sourceNames.has(capture)) {
    const n = (roleCounts.get(role) || 0) + 1;
    roleCounts.set(role, n);
    sourceNames.set(capture, {role, n});
  }
  s.identity = {capture, role, host, metric, entity: entity.join(' / ')};
  for (const point of s.points) point.context = JSON.parse(point[3]);
}
const names = {
  'GNB_PRACH_DECISION_LINK': 'PRACH detection record',
  'GNB_PRACH_WINDOW': 'PRACH search window',
  'GNB_RA Msg3 exhausted': 'Msg3 retry limit reached',
  'GNB_RA initiation': 'Random access started',
  'GNB_RA positive feedback': 'Random access: positive feedback',
  'GNB_RA negative feedback': 'Random access: negative feedback',
  'UE_CONTROL RLF': 'Radio-link failure',
  'UE_CONTROL RRC transition': 'RRC state change',
  'UE_CONTROL idle fallback': 'Return to idle',
  'UE_NAS': 'NAS event',
  'UE_PDU accept': 'PDU session accepted',
  'UE_PRACH_LUT': 'PRACH waveform table use',
  'UE_PRACH_SPAN': 'PRACH waveform interval',
  'UE_RA RAR failure': 'Random-access response not received',
  'UE_RA contention failure': 'Contention resolution failed',
  'UE_RA success': 'Random access succeeded',
  'UE_RRC': 'RRC event',
  'UE_RRC_TIMER': 'RRC timer event',
  'UE_SYNC failure': 'Synchronization failed',
  'UE_SYNC success': 'Synchronization succeeded',
  'UE_SYNC_EPOCH': 'Synchronization generation',
  'UE_TX_CHANNEL_CONTEXT': 'Transmit-channel context',
  'UE L1 SINR': 'SSB SINR',
  'UE SS-RSRP': 'SS-RSRP',
  'RX device gain': 'RX gain',
  'TX device gain': 'TX gain',
  'gNB mean RSRP': 'UE-reported RSRP at gNB',
  'gNB mean SINR': 'UE-reported SINR at gNB',
  'gNB PUSCH signal/noise ratio': 'PUSCH signal-to-noise ratio',
  'drb_context_active (receipt)': 'Active data-bearer context (observed)',
  'pdu_accepts (receipt)': 'PDU acceptances (observed)',
  'sync_successes (receipt)': 'Synchronization successes (observed)'
};
const palette = ['#146caa','#bd3d30','#21825b','#7756ac','#b16b0c','#168393','#ac4177','#5f6b35'];
for (const s of series) {
  const id = s.identity;
  const source = sourceNames.get(id.capture);
  const role = id.role === 'gnb' ? 'gNB' : id.role === 'ue' ? 'UE' : id.role;
  const prefix = role + (roleCounts.get(id.role) > 1 ? ' ' + source.n : '');
  const metric = names[id.metric] || id.metric.replace(/^(UE|gNB) /, '');
  s.shortLabel = prefix + ' · ' + titleCase(metric) + (id.entity ? ' · ' + id.entity : '');
  s.visible = true;
}
let basis = 0, limits = null, originUtc = display.recording_start_utc_ms;
let scenarioDuration = null, explicitOrigin = false;
const domainAnchors = new Map(Object.entries(display.domain_anchors));
function timeAt(point) {
  if (basis === 1) return point[1];
  if (point[0] === null) return null;
  const domain = point.context[0];
  const anchor = domainAnchors.get(domain);
  return anchor && anchor.utc_ms !== null && originUtc !== null ? point[0] + (anchor.utc_ms - originUtc) / 1000 : null;
}
const groups = new Map();
for (const s of series) {
  if (!groups.has(s.unit)) groups.set(s.unit, []);
  groups.get(s.unit).push(s);
}
const titles = {
  'event code': 'Connection and radio events',
  'core observation': 'Core Network Observations',
  'dBm': 'Received and configured power',
  'dB': 'Signal quality and power adjustments',
  'gain setting dB': 'RX and TX gain settings',
  'count': 'Recorded counters and state',
  'bytes': 'Data counters',
  'MCS index': 'Modulation and coding (MCS)',
  'raw CQI code': 'Channel Quality Indicator',
  'boolean': 'Reception status',
  'dBFS': 'Received sample levels',
  'bit/s': 'DL and UL Data Rates',
  '%': 'Receiver-reported packet loss',
  'kb/s': 'DL and UL data rates',
  'BLER %': 'DL and UL block error rate (BLER)'
};
const explanations = {
  'event code': 'Each dot marks a recorded event on its named row. Synchronization, random-access success and PDU acceptance describe different stages; none alone proves that application traffic is flowing. These are sampled records, so dots are not a complete count of separate failures.',
  'core observation': 'These points show when the collector received core-network log records. They help relate registration and session activity to radio events, but collector receipt time is not the exact time the core processed each message.',
  'dBm': 'SS-RSRP is the received synchronization-signal reference power: less-negative values mean a stronger received reference signal, but do not by themselves prove successful decoding. The gNB RSRP curve contains UE-reported measurements; PCMAX and p-Max are transmit-power limits, not received power. Hide those limits if you want to inspect only RSRP.',
  'dB': 'Higher SINR or signal-to-noise ratio generally makes decoding easier, while higher pathloss means a weaker propagation path. Power headroom (PH) and closed-loop power adjustments describe transmit-power control; PRACH energy and its noise reference describe preamble detection. These are different quantities sharing a unit, so compare matching curves and the reception results rather than treating every higher value as better.',
  'gain setting dB': 'RX gain and TX gain are settings reported by the USRP driver; they can come from configuration or gain-control adjustments. SSB context reported gain is the recorded RX-gain setting associated with the sample block used for that SSB measurement, so it need not match a driver readback taken at another time. Gain is neither received signal strength nor calibrated transmit power, and a constant TX gain does not mean constant transmitted signal power.',
  'count': 'HARQ round 0 counts initial transmission attempts; later rounds count retransmissions, and HARQ error counters count attempts that exhausted the allowed rounds. DTX counters count missing or unusable expected reception; increases are worth comparing with signal quality and recovery events. Bearer-context state and observed PDU/synchronization counters are separate indicators, not proof of traffic delivery; resets can accompany a new UE context.',
  'bytes': 'These are accumulated MAC byte counters for the labelled radio identifier (RNTI), not bit rates. SDU counters track data carried by MAC, while transport counters include transport-block overhead and padding; a rising DL counter records data prepared for transmission, not confirmation of UE reception. Counter resets and identifier changes start a new context; a flat line can mean idle traffic rather than a broken connection.',
  'MCS index': 'MCS is the modulation-and-coding index: 0 is valid and usually selects a robust, low-rate transmission, not a disconnected state. These are periodic scheduler-state observations, not a record of every transmitted block; the required signal quality depends on the MCS table and radio conditions. Every recorded observation is displayed up to the declared per-direction limit, including all recorded RNTIs. Above that limit the recording details disclose sampled and nonzero counts. Missing points do not establish a disconnection.',
  'raw CQI code': 'This is OAI\'s raw uplink quality code derived from the PUSCH signal/noise estimate, not the usual 0–15 downlink CQI. In the ordinary range, estimated SNR is approximately (code / 2 − 64) dB: 128 means about 0 dB and 132 about 2 dB. Higher generally means a stronger signal relative to noise, but check CRC and DTX too: the no-signal path can produce 128, and 255 must not be treated as a guaranteed good measurement.',
  'boolean': 'CRC failed = 1 means the gNB did not obtain a valid PUSCH block; 0 means its CRC check passed. DTX = 1 means the expected PUSCH did not pass the signal-detection threshold, not proof that the UE never transmitted. Compare them to separate an undetected signal from a detected but undecodable block; these 0/1 flags are not BLER percentages and a successful block does not prove end-to-end delivery.',
  'dBFS': 'dBFS describes the digital receive samples relative to full scale, not RF power in dBm. Sampled RX mean uses average I²+Q² power, while sampled RX peak uses the largest sampled I or Q component: a peak approaching 0 dBFS leaves little headroom and may indicate clipping risk. More-negative levels are not automatically bad; compare them with gain, SINR and decoding results. Only a subset of samples is inspected, so this graph cannot rule out peaks elsewhere.',
  'BLER %': 'The scheduler records a smoothed block-error estimate: 0.9 × previous estimate + 0.1 × the latest window estimate. It updates after at least 100 ms and is held when idle, so periodic snapshots are not independent failure probabilities. Compare it with MCS, retransmissions and traffic load. Samples without an initialized estimate remain unavailable. Legacy text BLER without precise source timestamps is not aligned here; an empty curve does not mean 0% BLER.',
  'bit/s': 'DL and UL share this graph. Local receiver interval reports measure delivery at that receiver. MAC SDU rates instead use 8 × byte-counter increase / elapsed seconds: downlink bytes submitted by the gNB and uplink bytes received by the gNB, including signalling. They are labelled separately and do not establish end-to-end goodput. MAC intervals require the same capture, attempt, clock domain, clock segment, cell, module, RNTI and recorded lifetime context, with nondecreasing counters and at most a 2 s observation gap. Legacy records without a context generation cannot prove same-RNTI lifetime continuity. Resets and excluded gaps are not filled with zeros.',
  '%': 'This is datagram loss reported by the traffic receiver for a reporting interval, not radio-block BLER. Lower is better for delivered application traffic, but missing receiver reports do not establish either zero loss or total loss.'
};
const summaries = {
  'event code': 'Follow synchronization, random access and session establishment on the same timeline. RNTI is a temporary radio identifier: it can change when the same UE reconnects.',
  'core observation': 'Registration and session messages observed at the core network. Their plotted times are collection times, not exact message-processing times.',
  'dBm': 'Compare received reference-signal power with recovery events. Less-negative RSRP means a stronger reference signal; PCMAX and p-Max are power limits and have a different meaning.',
  'dB': 'SINR and SNR describe signal quality; higher generally helps decoding. Pathloss, power headroom and power-control adjustments have different meanings despite also using dB.',
  'gain setting dB': 'These are receiver and transmitter gain settings, not measured RF power. More gain is not automatically better: it can also amplify noise or leave less room before clipping.',
  'count': 'Compare retransmission and missing-reception counters with connection events. Hide All lets you select one curve at a time; round 0 is the first transmission, not a retry.',
  'bytes': 'Accumulated MAC data, with automatic B/kB/MB/GB units. Rising counters show activity; they do not by themselves prove successful application delivery.',
  'MCS index': 'MCS selects modulation and coding. Higher indices can carry more bits under suitable conditions; index 0 is valid, and a low index does not mean disconnection.',
  'raw CQI code': 'This plot contains OAI’s raw uplink quality code, not the standard 0–15 downlink CQI. Higher usually means a better PUSCH signal/noise estimate; compare it with CRC and DTX.',
  'boolean': 'CRC Failed marks an invalid received data block. DTX means Discontinuous Transmission; here it marks expected uplink reception that the gNB did not detect reliably.',
  'dBFS': 'Digital receive-sample levels relative to full scale. Peaks near 0 dBFS leave little room before clipping; lower levels are not automatically bad.',
  'BLER %': 'Timestamped DL and UL scheduler Block Error Rate estimates are shown where recorded. They are smoothed state snapshots; unavailable estimates remain gaps.',
  'bit/s': 'Compare DL and UL local receiver rates and labelled MAC submission/reception rates, using automatic bit/s, kb/s or Mb/s units. The layers measure different outcomes.',
  '%': 'Packet loss reported by the traffic receiver. This differs from radio-block BLER; missing receiver reports are not a measured loss percentage.'
};

const eventHelp = {
  'GNB_PRACH_DECISION_LINK': 'Records the gNB’s preamble-detection decision and links it to the corresponding PRACH receive window. A dot is a detector record, not necessarily a detected UE or a new random-access attempt.',
  'GNB_PRACH_WINDOW': 'Records the radio-sample interval searched for a Physical Random Access Channel (PRACH) preamble. A search window can contain no successful detection.',
  'GNB_RA Msg3 exhausted': 'The gNB reached its allowed Msg3 retransmission limit without successfully completing that reception. Msg3 is the UE’s scheduled uplink message following the random-access response.',
  'GNB_RA initiation': 'The gNB started a random-access procedure after processing a preamble. This is earlier than contention resolution or a working data connection.',
  'GNB_RA positive feedback': 'The gNB processed positive acknowledgement feedback for Msg4/MsgB, the contention-resolution response. This is a radio-access milestone, not confirmation of application traffic.',
  'GNB_RA negative feedback': 'The gNB processed unsuccessful feedback for Msg4/MsgB. Further retries or recovery may follow.',
  'UE_CONTROL RLF': 'The UE’s Radio Resource Control (RRC) code reported radio-link failure. Several related records can belong to one disruption, so counting dots alone overcounts distinct outages.',
  'UE_CONTROL RRC transition': 'The UE changed its RRC connection state. A state change describes control-plane progress, not proof that the user-data path works.',
  'UE_CONTROL idle fallback': 'The UE returned to the idle recovery path. It may need cell selection, fresh random access and connection establishment before data resumes.',
  'UE_NAS': 'Non-Access Stratum (NAS) activity covers UE-to-core registration, security and session signalling, plus internal dispatch records. Not every dot is a radio message sent or received.',
  'UE_PDU accept': 'The UE decoded a matching Protocol Data Unit (PDU) Session Establishment Accept. An accepted session still needs a working radio bearer, IP configuration and actual traffic delivery.',
  'UE_PRACH_LUT': 'Records whether PRACH generation reused a prepared waveform table. It describes waveform preparation, not whether the gNB received the preamble.',
  'UE_PRACH_SPAN': 'Records where the PRACH waveform starts and how many samples it occupies, including its cyclic prefix. It helps check timing and waveform placement.',
  'UE_RA RAR failure': 'The UE did not obtain the expected valid Random Access Response (RAR) within the procedure’s reception window. This does not, by itself, identify whether the preamble or the response was lost.',
  'UE_RA contention failure': 'The UE did not complete contention resolution for that attempt. It may have transmitted Msg3 but failed to receive or accept the expected resolution response.',
  'UE_RA success': 'The UE completed random access successfully. Registration and session/bearer setup can still be required before application data flows.',
  'UE_RRC': 'A recorded UE RRC state observation from the connection-handling code. Use it together with transition and timer events; it is not another independently counted connection.',
  'UE_RRC_TIMER': 'An RRC procedure timer started, stopped or expired. The timer number and action in the full event record distinguish waiting from timeout-driven recovery.',
  'UE_SYNC failure': 'An initial synchronization or reacquisition attempt did not find a usable cell. Repeated search failures can belong to the same period without service.',
  'UE_SYNC success': 'The UE acquired synchronization and decoded the cell’s broadcast information needed at this stage. Random access and data-session recovery may still be pending.',
  'UE_SYNC_EPOCH': 'Marks a new synchronization generation and its timing reference. It separates timing observations before and after reacquisition; it is not a separate data connection.',
  'UE_TX_CHANNEL_CONTEXT': 'The current source emits this event only when preparing a Physical Random Access Channel (PRACH) waveform, with channel code 1. Each dot records a PRACH preparation request with its frame, slot, synchronization generation and preamble index. It is not a record of every uplink transmission and does not prove that the radio accepted samples, transmitted them, or that the gNB detected the preamble. Long gaps can separate random-access attempts or synchronization generations; tightly spaced retries can occur within an attempt. Plot selection can hide intermediate records when the display limit is exceeded. Compare the hover context and available/displayed counts with UE_PRACH_SPAN, UE_RA outcomes and UE_SYNC_EPOCH; dot spacing alone does not show continuous TX activity.'
};

const metricHelp = {
  'gNB DL HARQ errors': 'Downlink HARQ failures counted after the allowed transmission attempts are exhausted. This is a cumulative count, not a BLER percentage.',
  'gNB UL HARQ errors': 'Uplink HARQ failures counted after the allowed transmission attempts are exhausted. Compare its increase with retransmissions and DTX.',
  'gNB UL DTX': 'Expected uplink data reception that the gNB did not detect reliably. DTX expands to Discontinuous Transmission; here it is a receiver outcome, not proof that the UE deliberately remained silent.',
  'gNB PUCCH DTX': 'Missing or unreliable expected Physical Uplink Control Channel (PUCCH) reception. That channel carries control information such as acknowledgements, rather than the PUSCH data payload.',
  'gNB DL MAC SDU counter': 'Cumulative MAC Service Data Unit (SDU) bytes submitted for downlink transmission. This includes signalling and is not a count of bytes confirmed received by the UE.',
  'gNB UL MAC SDU counter': 'Cumulative MAC SDU bytes received by the gNB, including signalling. Application goodput needs a receiver measurement above the radio protocol layers.',
  'gNB DL MAC transport counter': 'Cumulative downlink transport-block bytes, including MAC overhead and padding. It is distinct from the useful SDU byte count.',
  'gNB UL MAC transport counter': 'Cumulative uplink transport-block bytes accounted by MAC, including overhead and padding. Do not treat it as application goodput.',
  'gNB PCMAX': 'The scheduler’s stored UE maximum transmit-power value, in dBm. Its initialization value can precede a real power-headroom report; it is not a received-power measurement.',
  'gNB PH': 'The scheduler’s normalized power-headroom value, in dB, based on the UE report and allocation factors. It is not the raw Power Headroom Report (PHR), and an initial default is not a measured report.',
  'gNB PRACH candidate energy': 'Preamble-correlation energy used by the gNB detector. Compare it with the matching detector noise reference and threshold; this is not calibrated received power in dBm.',
  'gNB PRACH noise reference': 'The noise/interference reference used by the PRACH detector. It helps explain detection decisions when compared with the corresponding candidate energy.',
  'gNB DL MCS': 'Periodic downlink scheduler MCS state, including valid zero and nonzero observations. It is not a log of every actual grant. RNTI, cell and lifetime context are shown when recorded; point counts disclose any display limit.',
  'gNB UL MCS': 'Periodic uplink scheduler MCS state. A higher MCS needs suitable signal quality; it is not evidence by itself that more data was delivered.',
  'gNB PUSCH CQI raw': 'OAI encodes its uplink signal/noise estimate on a raw scale: in the ordinary range, SNR ≈ code / 2 − 64 dB. This is not the standard UE-reported downlink CQI index.',
  'gNB PUSCH CRC failed': 'Cyclic Redundancy Check (CRC) failure flag: 1 means no valid PUSCH block was obtained, 0 means success. The failure path includes undetected reception; inspect DTX alongside it.',
  'gNB PUSCH DTX': 'Discontinuous Transmission indication for expected Physical Uplink Shared Channel (PUSCH) data: 1 means the signal-detection threshold was not passed. Weak reception can cause this even if the UE transmitted.',
  'gNB PUSCH signal/noise ratio': 'The gNB’s PUSCH signal-to-noise estimate. Larger values generally help decoding, but its meaning depends on the receiver estimator and detection state.',
  'gNB mean RSRP': 'An average of UE-reported reference-signal power available at the gNB. It describes the UE’s downlink observation, not an independent gNB measurement of uplink RSRP.',
  'RX device gain': 'Receive gain read back from the USRP driver. Raising it amplifies wanted signal and noise and may increase clipping risk.',
  'TX device gain': 'Transmit gain read back from the USRP driver. Transmitted RF power also depends on waveform amplitude and hardware calibration.',
  'SSB context reported gain': 'The recorded RX-gain setting associated with the samples used for an SSB measurement. It can differ from a driver readback at another time.',
  'UE L1 SINR': 'The UE’s physical-layer SSB signal-to-interference-plus-noise estimate. Higher usually makes synchronization and decoding easier; it is not an application performance guarantee.',
  'UE SS-RSRP': 'Synchronization-signal Reference Signal Received Power measured at the UE, in dBm. Less-negative values mean stronger reference-signal reception.',
  'UE accepted SSB RSRP': 'An SSB RSRP value accepted by the UE’s measurement/power-control path. It can describe the same underlying signal as the other RSRP curve, rather than an independent measurement.',
  'UE checked SSB pathloss': 'An SSB pathloss value checked by the UE power-control path. Greater pathloss generally calls for more transmit power, subject to configured limits.',
  'UE closed-loop adjustment': 'Accumulated uplink power-control correction. Positive and negative changes adjust the UE power request; this is not measured transmit power.',
  'UE configured network p-Max': 'The maximum UE transmit-power limit configured by the network, in dBm. The effective permitted power can also depend on UE and hardware constraints.',
  'UE pathloss input RSRP': 'The received reference-signal power used in a particular pathloss calculation. It is an input to power control, not the resulting pathloss.',
  'UE serving SSB pathloss': 'Serving-cell SSB pathloss used by UE power control. It combines the reference transmit-power information with received reference-signal power.',
  'drb_context_active (receipt)': 'Whether the collector observed an active Data Radio Bearer (DRB) context. An active context is necessary for traffic but does not prove delivery.',
  'pdu_accepts (receipt)': 'Cumulative PDU-session acceptance observations made by the collector. It is a setup-progress counter, not a traffic counter.',
  'sync_successes (receipt)': 'Cumulative successful synchronization observations made by the collector. Several can occur during one UE process as it loses and reacquires the cell.',
  'sampled RX mean': 'Mean digital I²+Q² sample power relative to the recorded full-scale reference. Because this is combined I/Q power, it differs from the single-component peak definition.',
  'sampled RX peak': 'Largest inspected I or Q component relative to full scale. A value near 0 dBFS indicates little headroom; sampling only part of the buffer can miss other peaks.',
  'DL MAC SDU rate (submitted)': 'Difference between consecutive gNB downlink SDU byte counters divided by elapsed time. It includes signalling and describes submission, not successful reception at the UE.',
  'UL MAC SDU rate (received)': 'Difference between consecutive gNB uplink SDU byte counters divided by elapsed time. It includes signalling; missing intervals and counter resets are not filled with zero.',
  'DL scheduler BLER': 'The scheduler’s smoothed downlink block-error estimate at this source timestamp and UE lifetime context. The 0.9/0.1 update mixes the previous estimate with the latest eligible window and holds its value while idle; snapshots are not independent trials.',
  'UL scheduler BLER': 'The scheduler’s smoothed uplink block-error estimate at this source timestamp and UE lifetime context. An uninitialized or unrecorded estimate is unavailable, not a measured 0% error rate.',
  'DL local receiver rate': 'Downlink bytes received during the local traffic receiver’s reported interval, divided by interval duration. This measures observed receiver delivery, including a measured zero when the receiver explicitly reports one.',
  'UL local receiver rate': 'Uplink bytes received during the local traffic receiver’s reported interval, divided by interval duration. A missing receiver report is unavailable, not zero.',
  'DL receiver datagram loss': 'Datagram loss reported by the local downlink traffic receiver for an interval. It differs from radio-block BLER.',
  'UL receiver datagram loss': 'Datagram loss reported by the local uplink traffic receiver for an interval. Absent reports do not establish a loss percentage.'
};

function describeSeries(s) {
  const metric=s.identity.metric;
  if(eventHelp[metric])return eventHelp[metric];
  const round=/^gNB (DL|UL) HARQ round (\d+)$/.exec(metric);
  if(round){const n=Number(round[2]);return `${round[1]} count of ${n===0?'initial transmissions':`retransmission ${n} (overall attempt ${n+1})`}. HARQ means Hybrid Automatic Repeat Request: retransmissions can reuse information from earlier failed receptions.`;}
  return metricHelp[metric] || (s.unit === 'core observation' ? 'Collector receipt of ' + metric + '. Original source fields remain in the full export; this timestamp does not establish the exact core processing time.' : explanations[s.unit]) || 'Recorded ' + metric + ' observation. Inspect the source record for its typed fields; availability alone does not establish successful traffic delivery.';
}

function titleCase(text) {
  const small=new Set(['a','an','the','and','or','nor','but','for','of','to','in','on','at','by','as','with','vs','from']);
  return text.split(' ').map((w,i)=>i>0&&small.has(w.toLowerCase())?w.toLowerCase():w.replace(/^[a-z]/,c=>c.toUpperCase())).join(' ');
}

function axisUnits(unit, peak) {
  if(unit==='bytes') {
    const names=['B','kB','MB','GB','TB'];
    const index=Math.max(0,Math.min(4,Math.floor(Math.log10(Math.max(1,peak))/3)));
    return {factor:1000**index,label:names[index],scaled:index>0};
  }
  if(unit==='kb/s'||unit==='bit/s') {
    const original=unit==='kb/s'?1000:1;
    const names=['bit/s','kb/s','Mb/s','Gb/s'];
    const index=Math.max(unit==='kb/s'?1:0,Math.min(3,Math.floor(Math.log10(Math.max(1,peak*original))/3)));
    return {factor:1000**index/original,label:names[index],scaled:true};
  }
  return {factor:1,label:unit==='gain setting dB'?'Gain (dB)':unit,scaled:false};
}

function extent() {
  let lo=Infinity, hi=-Infinity;
  for (const s of series) for (const p of s.points) if (timeAt(p) !== null) {
    lo=Math.min(lo,timeAt(p)); hi=Math.max(hi,timeAt(p));
  }
  if(basis===0) return [0,scenarioDuration || Math.max(1,hi)];
  if(explicitOrigin || scenarioDuration) return [originUtc,originUtc+(scenarioDuration || Math.max(1,(hi-originUtc)/1000))*1000];
  return lo === Infinity ? [originUtc || 0,(originUtc || 0)+1000] : [originUtc===null?lo:Math.min(lo,originUtc),hi === lo ? lo+1000 : hi];
}
function labelTime(v) {
  return basis === 1 ? new Date(v).toISOString().slice(11,23) : v.toFixed(2)+' s';
}
function fullTime(v) {
  return basis === 1 ? new Date(v).toISOString() : v.toFixed(3)+' s from '+(explicitOrigin?'the entered scenario start':'the recording start');
}
function writeRange() {
  document.getElementById('from').value=basis===1?new Date(limits[0]).toISOString():limits[0].toFixed(3);
  document.getElementById('to').value=basis===1?new Date(limits[1]).toISOString():limits[1].toFixed(3);
}
function reset() {
  limits=extent();
  writeRange();
  draw();
}
function element(tag, text, className) {
  const e=document.createElement(tag);
  if(text !== undefined) e.textContent=text;
  if(className) e.className=className;
  return e;
}
function wrap(c,text,width) {
  const out=[]; let line='';
  for(const word of text.split(' ')) {
    const trial=line ? line+' '+word : word;
    if(line && c.measureText(trial).width>width) {out.push(line);line=word;} else line=trial;
  }
  if(line)out.push(line);
  return out;
}
function draw() {
  const scrollX=window.scrollX,scrollY=window.scrollY;
  const focused=document.activeElement?.id;
  const expanded=new Set([...root.querySelectorAll('details[open]')].map(d=>d.id));
  root.style.minHeight=root.getBoundingClientRect().height+'px';
  root.replaceChildren();
  for(const [unit,allRows] of groups) {
    const eventPlot=unit==='event code'||unit==='core observation';
    const rows=allRows.filter(s=>s.visible);
    const article=element('article');
    article.append(element('h2',titleCase(titles[unit]||unit)));
    article.append(element('p',summaries[unit]||'Recorded observations; open the explanation below for their meaning.','chart-description'));
    const help=element('details',undefined,'graph-help');
    help.id='help-'+encodeURIComponent(unit);help.open=expanded.has(help.id);
    help.append(element('summary',eventPlot?'How to Read These Events':'How to Read This Graph'));
    help.append(element('p',explanations[unit]));
    if(unit==='event code')help.append(element('p','RNTI means Radio Network Temporary Identifier. The C-RNTI or temporary C-RNTI labels here identify a radio connection within a cell, not a permanent UE identity. Reconnection can assign a different value to the same UE, and an old value can later be reused. RNTI 0 in these records means no assigned identifier was recorded. Labels use decimal values; some OAI text logs use hexadecimal.'));
    if(unit==='count')help.append(element('p','HARQ means Hybrid Automatic Repeat Request. Round 0 is the initial transmission; later rounds count retransmissions. Four configured attempts allow rounds 0 through 3; eight allow rounds 0 through 7. The Radio Configuration cards and recorded per-context fields give the captured limits, including the initial transmission. New recorder versions include rounds 4 through 7 when configured; legacy versions contain only rounds 0 through 3. Missing higher-round records do not prove a configured four-attempt limit.'));
    if(unit==='count'||unit==='boolean')help.append(element('p','DTX stands for Discontinuous Transmission. In these gNB reception records it means the expected uplink signal or feedback was not detected reliably; it does not prove intentional silence at the UE. PUSCH carries uplink data; PUCCH carries uplink control information such as acknowledgements.'));
    if(['bytes','kb/s','bit/s'].includes(unit))help.append(element('p','Units use decimal SI prefixes: 1 kB = 1,000 bytes, 1 MB = 1,000,000 bytes; 1 Mb/s = 1,000 kb/s. Uppercase B means bytes and lowercase b means bits. The axis unit is selected from the visible curves in the current time range, independently for data totals and rates.'));
    const glossary=element('dl',undefined,'metric-glossary');
    if(eventPlot) {
      const guides=element('div');
      allRows.forEach((s,index)=>{
        const rowHelp=element('details',undefined,'row-help');
        rowHelp.id='row-help-'+encodeURIComponent(unit)+'-'+index;rowHelp.open=expanded.has(rowHelp.id);
        rowHelp.append(element('summary',s.shortLabel),element('p',describeSeries(s)));
        rowHelp.append(element('p',`${s.plotted_points} displayed / ${s.source_points} available records. ${s.sampling}.`));
        guides.append(rowHelp);
      });
      help.append(guides);
    } else for(const s of allRows){glossary.append(element('dt',s.shortLabel),element('dd',describeSeries(s)));}
    help.append(glossary);article.append(help);
    const reduced=allRows.filter(s=>s.plotted_points<s.source_points);
    if(reduced.length) article.append(element('p',`${reduced.length} curve(s) use a selection of recorded points. Peaks or nonzero values can be absent from this display; see Recording Details for exact counts and the full exports for every selected row.`,'chart-description'));
    const bulk=element('div',undefined,'bulk-controls');
    for(const [label,value] of [['Hide All',false],['Show All',true]]) {
      const b=element('button',label);b.type='button';b.id='plot-'+encodeURIComponent(unit)+'-'+(value?'show':'hide');
      b.onclick=()=>{allRows.forEach(s=>s.visible=value);draw();};bulk.append(b);
    }
    bulk.append(element('span',`${rows.length} of ${allRows.length} ${eventPlot?'rows':'curves'} visible`));
    article.append(bulk);
    const legend=element('div',undefined,'legend');
    for(const s of allRows) {
      const index=allRows.indexOf(s), button=element('button',s.shortLabel,'series-toggle');
      button.type='button'; button.setAttribute('aria-pressed',String(s.visible));
      button.id='series-'+encodeURIComponent(unit)+'-'+index;
      button.style.setProperty('--series-color',palette[index%palette.length]);
      button.onclick=()=>{s.visible=!s.visible;draw();};
      legend.append(button);
    }
    const canvas=element('canvas');
    canvas.setAttribute('role','img');
    canvas.setAttribute('aria-label',(titles[unit]||unit)+' versus time. Series and source counts are listed below.');
    const info=element('p','Move over or click a point to see its time and value.','point');
    info.setAttribute('aria-live','polite');
    const details=element('details');
    details.id='details-'+encodeURIComponent(unit);
    details.open=expanded.has(details.id);
    details.append(element('summary','Recording Details and Displayed-Point Counts'));
    for(const s of allRows) {
      const p=element('p');
      p.append(element('strong',s.shortLabel));
      p.append(element('span',` - ${s.plotted_points} displayed / ${s.source_points} available records; nonzero: ${s.nonzero_plotted_points} displayed / ${s.nonzero_source_points} available. ${s.sampling}.`));
      p.append(element('code',s.label,'full-label'));
      details.append(p);
    }
    if(eventPlot){const choice=element('details',undefined,'event-choices');choice.id='choices-'+encodeURIComponent(unit);choice.open=expanded.has(choice.id);choice.append(element('summary','Choose Event Rows'),legend);article.append(choice);}else article.append(legend);
    article.append(canvas,info,details); root.append(article);
    const ratio=window.devicePixelRatio||1;
    const width=Math.max(260,canvas.clientWidth);
    const height=eventPlot?Math.max(200,allRows.length*58+75):310;
    canvas.style.height=height+'px';canvas.width=width*ratio;canvas.height=height*ratio;
    const c=canvas.getContext('2d');c.scale(ratio,ratio);
    const pad=eventPlot?Math.min(350,width*0.47):72, right=18, top=26, bottom=57;
    const x=v=>pad+(v-limits[0])/(limits[1]-limits[0])*(width-pad-right);
    let ymin=Infinity,ymax=-Infinity;
    for(const s of rows)for(const p of s.points)if(timeAt(p)!==null&&timeAt(p)>=limits[0]&&timeAt(p)<=limits[1]) {
      ymin=Math.min(ymin,p[2]);ymax=Math.max(ymax,p[2]);
    }
    const hasData=ymin!==Infinity;
    const axis=axisUnits(unit,hasData?Math.max(Math.abs(ymin),Math.abs(ymax)):0);
    const integerAxis=['bytes','count','MCS index','raw CQI code','boolean'].includes(unit)&&!axis.scaled;
    const nonnegative=hasData&&ymin>=0&&['bytes','count','MCS index','raw CQI code','boolean','kb/s','BLER %','%','bit/s'].includes(unit);
    if(ymin===ymax){ymin-=1;ymax+=1;}
    if(ymin===Infinity){ymin=0;ymax=1;}
    const gap=(ymax-ymin)*.05;ymin-=gap;ymax+=gap;
    if(nonnegative)ymin=Math.max(0,ymin);
    if(unit==='boolean'){ymin=0;ymax=1;}
    if(integerAxis&&unit!=='boolean'){ymin=nonnegative?Math.max(0,Math.floor(ymin)):Math.floor(ymin);ymax=Math.ceil(ymax);}
    const y=v=>height-bottom-(v-ymin)/(ymax-ymin)*(height-top-bottom);
    c.font='12px system-ui';c.lineWidth=1;c.strokeStyle='#dce4eb';c.fillStyle='#475569';
    const ticks=width<650?2:4;
    for(let i=0;i<=ticks;i++) {
      const v=limits[0]+(limits[1]-limits[0])*i/ticks, xx=x(v);
      c.beginPath();c.moveTo(xx,top);c.lineTo(xx,height-bottom);c.stroke();
      c.textAlign=i===0?'left':i===ticks?'right':'center';
      c.fillText(labelTime(v),xx,height-bottom+20);
    }
    c.textAlign='center';
    const axisLabel=basis===1?'Time (UTC)':(explicitOrigin?'Scenario elapsed time (s)':'Elapsed time from recording start (s)');
    const axisLines=wrap(c,axisLabel,width-pad-right);
    axisLines.forEach((line,i)=>c.fillText(line,(pad+width-right)/2,height-17+i*13));
    if(!eventPlot) {
      c.textAlign='right';
      const values=integerAxis?[]:[0,1,2,3,4].map(i=>ymin+(ymax-ymin)*i/4);
      if(integerAxis){const step=Math.max(1,Math.ceil((ymax-ymin)/4));for(let v=Math.ceil(ymin/step)*step;v<=ymax;v+=step)values.push(v);}
      for(const v of values) {
        const yy=y(v);
        c.beginPath();c.moveTo(pad,yy);c.lineTo(width-right,yy);c.stroke();
        c.fillText(axis.scaled?(v/axis.factor).toFixed(2):integerAxis?v.toLocaleString('en-GB'):Number(v.toPrecision(4)).toString(),pad-10,yy+4);
      }
      c.textAlign='left';c.fillText(axis.label,8,15);
    }
    if(!hasData){c.textAlign='center';c.fillStyle='#64748b';c.fillText(!rows.length?'All Curves Hidden - Select a Label or Show All':unit==='BLER %'?'No Selected Timestamped BLER Samples in This View':'No Observations in This View',(pad+width-right)/2,(height-bottom+top)/2);}
    const hits=[];
    rows.forEach((s,index)=>{
      const colour=palette[allRows.indexOf(s)%palette.length];
      const eventY=top+(index+.5)*(height-top-bottom)/Math.max(1,rows.length);
      if(eventPlot) {
        c.fillStyle='#24374a';c.textAlign='right';
        const lines=wrap(c,s.shortLabel,pad-20);
        lines.forEach((line,i)=>c.fillText(line,pad-12,eventY+(i-(lines.length-1)/2)*14+4));
        c.strokeStyle='#e4eaf0';c.beginPath();c.moveTo(pad,eventY);c.lineTo(width-right,eventY);c.stroke();
      }
      c.strokeStyle=c.fillStyle=colour;let previous=null;
      for(const p of s.points) {
        const v=timeAt(p);if(v===null||v<limits[0]||v>limits[1]){previous=null;continue;}
        const xx=x(v),yy=eventPlot?eventY:y(p[2]);
        if(!eventPlot&&unit!=='boolean'&&unit!=='MCS index'&&previous&&previous[3]===p[3]) {
          c.beginPath();c.moveTo(x(timeAt(previous)),y(previous[2]));c.lineTo(xx,yy);c.stroke();
        }
        c.beginPath();c.arc(xx,yy,eventPlot?3.2:2.5,0,2*Math.PI);c.fill();
        hits.push([xx,yy,s,p]);previous=p;
      }
    });
    function inspect(e) {
      const rect=canvas.getBoundingClientRect(),xx=e.clientX-rect.left,yy=e.clientY-rect.top;
      let best=null,d=Infinity;
      for(const hit of hits){const distance=(hit[0]-xx)**2+(hit[1]-yy)**2;if(distance<d){d=distance;best=hit;}}
      if(best&&d<1600)info.textContent=best[2].shortLabel+(eventPlot?'':` = ${axis.scaled?(best[3][2]/axis.factor).toFixed(2):best[3][2]} ${axis.label}`)+'; '+fullTime(timeAt(best[3]))+'; '+best[3][4];
    }
    canvas.onmousemove=inspect;canvas.onclick=inspect;
    canvas.onwheel=e=>{
      // Ordinary scrolling still scrolls the page; Ctrl+wheel zooms all plots.
      if(!e.ctrlKey)return;e.preventDefault();
      const f=Math.max(0,Math.min(1,(e.offsetX-pad)/(width-pad-right)));
      const center=limits[0]+f*(limits[1]-limits[0]),factor=e.deltaY>0?1.25:.8;
      limits=[center+(limits[0]-center)*factor,center+(limits[1]-center)*factor];
      writeRange();draw();
    };
  }
  root.style.minHeight='';
  if(focused&&(focused.startsWith('series-')||focused.startsWith('plot-')))document.getElementById(focused)?.focus({preventScroll:true});
  window.scrollTo({left:scrollX,top:scrollY,behavior:'instant'});
}
document.getElementById('show-utc').onchange=e=>{
  const next=e.target.checked?1:0;
  if(next!==basis)limits=limits.map(v=>next===1?originUtc+v*1000:(v-originUtc)/1000);
  basis=next;writeRange();draw();
};
document.getElementById('reset').onclick=reset;
document.getElementById('apply').onclick=()=>{
  const parse=basis===1?Date.parse:Number;
  const a=parse(document.getElementById('from').value),b=parse(document.getElementById('to').value);
  if(Number.isFinite(a)&&Number.isFinite(b)&&b>a){limits=[a,b];draw();}
};
document.getElementById('apply-scenario').onclick=()=>{
  const start=document.getElementById('scenario-start').value.trim();
  const duration=document.getElementById('scenario-duration').value.trim();
  const parsed=start?Date.parse(start):display.recording_start_utc_ms;
  const seconds=duration?Number(duration):null;
  const status=document.getElementById('scenario-status');
  if((start&&!/Z$|[+-]\d\d:\d\d$/.test(start))||!Number.isFinite(parsed)||(seconds!==null&&(!Number.isFinite(seconds)||seconds<=0))){status.textContent='Enter a UTC timestamp ending in Z and a positive duration, or leave the fields empty.';return;}
  originUtc=parsed;scenarioDuration=seconds;explicitOrigin=!!start;
  status.textContent=explicitOrigin?'Zero uses your scenario start; each host advances using its monotonic clock. Cross-host alignment remains unverified.':'Zero uses the recording start. The actual scenario launch time is not available in these logs.';
  reset();
};
let resizeTimer;
window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(draw,150);});
if(originUtc===null) {
  document.getElementById('show-utc').disabled=true;
  document.getElementById('scenario-status').textContent='A clock mapping to UTC is unavailable. Unmapped records remain in the full export; the report does not invent a recording start.';
}
reset();
