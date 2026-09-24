#!/usr/bin/env python3
"""Decode only numeric OAI flight events, retaining malformed/unclean indicators."""
import argparse
import csv
import json
import math
import re
from pathlib import Path

NAMES = dict(zip([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31,
                  40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57],
                 ['UE_SYNC', 'UE_AGC', 'UE_MEASUREMENTS', 'UE_RA', 'UE_RRC', 'UE_PDU', 'UE_TA',
                  'UE_NAS', 'UE_RRC_TIMER', 'UE_CONTROL', 'GNB_SLOT', 'GNB_UE_BYTES', 'GNB_UE_RADIO', 'GNB_RA', 'GNB_UE_LINK',
                  'GNB_DL_HARQ', 'GNB_UL_HARQ', 'UE_NAS_COUNT', 'RADIO_RX', 'RADIO_TX',
                  'RADIO_GAIN', 'RADIO_GAIN_TIME', 'RADIO_RX_LEVEL', 'UE_TX_POWER_REQUEST', 'GNB_TX_REFERENCE',
                  'RADIO_RX_DECISION', 'RADIO_RX_DECISION_INPUT', 'RADIO_TX_LEVEL', 'RADIO_TX_LEVEL_STATE',
                  'RADIO_TX_POWER', 'RADIO_TX_POWER_SAMPLES', 'RADIO_TX_REJECT', 'RADIO_TX_POWER_QUALITY', 'UE_TX_CONTROL', 'UE_PATHLOSS_STATE',
                  'UE_SSB_MEASUREMENT', 'UE_SSB_MEASUREMENT_CONTEXT', 'RADIO_RX_PEAK_ENVELOPE']))
FIELDS = ['source_file', 'source_line', 'name', 'event', 'ring', 'sequence', 'mono_ns', 'realtime_ns',
          'a', 'b', 'c', 'd', 'e', 'f']

INT64_MIN = -(1 << 63)
UINT64_MASK = (1 << 64) - 1
RADIO_GAIN = 40
RADIO_GAIN_TIME = 41
RADIO_RX_LEVEL = 42
UE_TX_POWER_REQUEST = 43
GNB_TX_REFERENCE = 44
RADIO_RX_DECISION = 45
RADIO_RX_DECISION_INPUT = 46
RADIO_TX_POWER = 49
RADIO_TX_POWER_SAMPLES = 50
RADIO_TX_REJECT = 51
RADIO_TX_POWER_QUALITY = 52
UE_TX_CHANNEL_NAMES = {
    1: 'PRACH',
    2: 'PUSCH',
    3: 'PUCCH',
    4: 'SRS',
}
RADIO_TX_REJECT_CHANNEL_NAMES = {0: 'GENERIC', **UE_TX_CHANNEL_NAMES}
RADIO_GAIN_SET_RX = 0
RADIO_GAIN_SET_TX = 1
RADIO_GAIN_RETUNE = 2
RADIO_GAIN_STATUS_NAMES = {
    0: 'OK',
    1: 'BUSY',
    2: 'UNSUPPORTED',
    3: 'INVALID',
    4: 'STALE',
    5: 'CLOSED',
    6: 'BACKEND_ERROR',
    7: 'TX_PENDING',
}
RADIO_GAIN_OPERATION_NAMES = {
    RADIO_GAIN_SET_RX: 'SET_RX',
    RADIO_GAIN_SET_TX: 'SET_TX',
    RADIO_GAIN_RETUNE: 'RETUNE',
}
RADIO_GAIN_FIELDS = [
    'source_file', 'source_line', 'radio', 'generation', 'recorder_mono_ns', 'recorder_realtime_ns',
    'status_code', 'status', 'operation_code', 'operation',
    'result_valid_flags', 'result_unknown_valid_flags', 'rx_gain_valid', 'tx_gain_valid', 'device_time_valid',
    'time_metadata_joined', 'time_valid_flags', 'time_unknown_valid_flags', 'request_id',
    'requested_gain_db', 'rx_readback_db', 'tx_readback_db', 'begin_device_ticks', 'end_device_ticks',
]
RADIO_RX_LEVEL_FIELDS = [
    'source_file', 'source_line', 'radio', 'generation', 'context_valid', 'recorder_mono_ns', 'recorder_realtime_ns',
    'sample_start_ticks', 'mean_power_dbfs', 'peak_component_dbfs', 'near_rail_components', 'sampled_components',
]
RADIO_RX_DECISION_FIELDS = [
    'source_file', 'source_line', 'thread_ring', 'decision_sequence', 'source_code', 'source',
    'generation', 'context_end_sample_ticks', 'recorder_mono_ns', 'recorder_realtime_ns',
    'mean_level_dbfs', 'candidate_gain_db', 'reason_code', 'reason', 'flags_raw',
    'unknown_flag_bits', 'would_change', 'submitted', 'activity_valid', 'search_failed', 'phase',
    'input_join_status', 'input_source_file', 'input_source_line', 'input_sequence',
    'input_recorder_mono_ns', 'input_recorder_realtime_ns', 'reported_rx_gain_db',
    'peak_component_dbfs', 'error_raw_millidb', 'error_db',
]
UE_TX_POWER_FIELDS = [
    'source_file', 'source_line', 'ue', 'frame', 'slot', 'channel_code', 'channel',
    'requested_power_dbm', 'generator_reference', 'resource_raw',
    'resource_first', 'resource_count', 'start_symbol', 'symbol_count', 'detail_0', 'detail_1',
    'recorder_mono_ns', 'recorder_realtime_ns',
]
GNB_TX_REFERENCE_FIELDS = [
    'source_file', 'source_line', 'frame', 'slot', 'configured_ss_pbch_power_dbm', 'tx_amp',
    'ssb_index', 'ssb_start_symbol', 'antenna_port', 'beam_index', 'ssb_start_subcarrier',
    'recorder_mono_ns', 'recorder_realtime_ns',
]
RADIO_TX_POWER_FIELDS = [
    'source_file', 'source_line', 'thread_ring', 'power_sequence', 'frame_slot_raw', 'frame', 'slot', 'channel_code', 'channel',
    'recorder_mono_ns', 'recorder_realtime_ns', 'status_flags_raw', 'status_code', 'status',
    'status_unknown_flag_bits', 'applied', 'requested_power_dbm', 'estimated_output_power_dbm',
    'amplitude_coefficient_q30', 'sample_join_status', 'sample_source_file', 'sample_source_line',
    'sample_sequence', 'sample_recorder_mono_ns', 'sample_recorder_realtime_ns', 'sample_count', 'input_energy',
    'output_energy', 'input_peak_component', 'output_peak_component', 'quality_join_status',
    'quality_source_file', 'quality_source_line', 'quality_sequence', 'quality_recorder_mono_ns',
    'quality_recorder_realtime_ns', 'quantization_power_error_raw_millidb', 'quantization_power_error_db',
    'quantization_evm_raw_ppb', 'quantization_evm_ppb', 'profile_uncertainty_raw_millidb',
    'profile_uncertainty_db', 'component_full_scale',
    'evidence_group_sequence_span', 'evidence_interleaved_events', 'evidence_join_basis',
]
RADIO_TX_REJECT_FIELDS = [
    'source_file', 'source_line', 'thread_ring', 'sequence', 'radio', 'frame_slot_raw', 'frame', 'slot', 'channel_code', 'channel',
    'reason_code', 'reason', 'actuation_requested_raw', 'actuation_requested', 'recorder_mono_ns',
    'recorder_realtime_ns',
]

RADIO_RX_SOURCE_NAMES = {
    1: 'UE_SSB',
    2: 'GNB_PUSCH',
    3: 'UE_SEARCH',
    4: 'HEADROOM',
}
RADIO_RX_REASON_NAMES = {
    0: 'HOLD_INVALID',
    1: 'HOLD_UNSUPPORTED',
    2: 'HOLD_STALE',
    3: 'HOLD_TRANSITION',
    4: 'HOLD_INACTIVE',
    5: 'HOLD_DEADBAND',
    6: 'HOLD_COOLDOWN',
    7: 'HOLD_LIMIT',
    8: 'REDUCE_OVERLOAD',
    9: 'TRACK_LEVEL',
    10: 'SEARCH_STEP',
}
RADIO_RX_KNOWN_FLAG_MASK = (1 << 14) - 1
RADIO_RX_ERROR_REASONS = {5, 9}
RADIO_TX_POWER_STATUS_NAMES = {
    0: 'OK',
    1: 'INVALID',
    2: 'UNQUALIFIED',
    3: 'MAPPING_REJECTED',
    4: 'HEADROOM',
    5: 'QUANTIZATION',
}
RADIO_TX_POWER_KNOWN_STATUS_MASK = (1 << 9) - 1
RADIO_TX_REJECT_REASON_NAMES = {
    1: 'LAYOUT',
    2: 'SPAN',
    3: 'OVERLAP',
    4: 'POWER_LIMIT',
    5: 'PROFILE',
    6: 'POWER_CONTROL',
}


def _enum_name(names, value):
    return names.get(value, f'UNKNOWN_{value}')


def _optional_milli_db(value):
    return None if value == INT64_MIN else value / 1000.0


def _optional_int64(value):
    return None if value == INT64_MIN else value


def _optional_bool(value):
    if value == 0:
        return False
    if value == 1:
        return True
    return None


def _validity_flags(value):
    flags = value & UINT64_MASK
    return {
        'valid_flags': flags,
        'unknown_valid_flags': flags & ~0x7,
        'rx_gain_valid': bool(flags & 0x1),
        'tx_gain_valid': bool(flags & 0x2),
        'device_time_valid': bool(flags & 0x4),
    }


def decode_radio_gain_event(event):
    """Return typed ID 40--42 payload data, or None for another numeric event.

    Results are intentionally partial until decode() can join a unique ID 41
    metadata record by (radio, generation). In particular, a requested gain is
    unavailable until the request ID and non-retune operation are both known.
    """
    event_id = event['event']
    if event_id == RADIO_GAIN:
        packed = event['f'] & UINT64_MASK
        flags = _validity_flags(packed >> 16)
        return {
            'record_type': 'result',
            'radio': event['a'],
            'generation': event['b'],
            'recorder_mono_ns': event['mono_ns'],
            'recorder_realtime_ns': event['realtime_ns'],
            'status_code': packed & 0xff,
            'status': _enum_name(RADIO_GAIN_STATUS_NAMES, packed & 0xff),
            'operation_code': (packed >> 8) & 0xff,
            'operation': _enum_name(RADIO_GAIN_OPERATION_NAMES, (packed >> 8) & 0xff),
            'result_valid_flags': flags['valid_flags'],
            'result_unknown_valid_flags': flags['unknown_valid_flags'],
            'rx_gain_valid': flags['rx_gain_valid'],
            'tx_gain_valid': flags['tx_gain_valid'],
            'device_time_valid': flags['device_time_valid'],
            'time_metadata_joined': False,
            'time_valid_flags': None,
            'time_unknown_valid_flags': None,
            'request_id': None,
            'requested_gain_db': None,
            'rx_readback_db': _optional_milli_db(event['d']) if flags['rx_gain_valid'] else None,
            'tx_readback_db': _optional_milli_db(event['e']) if flags['tx_gain_valid'] else None,
            'begin_device_ticks': None,
            'end_device_ticks': None,
            '_requested_gain_mdb': event['c'],
        }
    if event_id == RADIO_GAIN_TIME:
        flags = _validity_flags(event['f'])
        return {
            'record_type': 'time',
            'radio': event['a'],
            'generation': event['b'],
            'request_id': event['c'],
            'begin_device_ticks': _optional_int64(event['d']) if flags['device_time_valid'] else None,
            'end_device_ticks': _optional_int64(event['e']) if flags['device_time_valid'] else None,
            **flags,
        }
    if event_id == RADIO_RX_LEVEL:
        packed = event['f'] & UINT64_MASK
        signed_generation = event['b']
        return {
            'record_type': 'rx_level',
            'radio': event['a'],
            'generation': signed_generation if signed_generation >= 0 else -signed_generation,
            'context_valid': signed_generation > 0,
            'recorder_mono_ns': event['mono_ns'],
            'recorder_realtime_ns': event['realtime_ns'],
            'sample_start_ticks': event['c'],
            'mean_power_dbfs': _optional_milli_db(event['d']),
            'peak_component_dbfs': _optional_milli_db(event['e']),
            'near_rail_components': packed >> 32,
            'sampled_components': packed & 0xffffffff,
        }
    return None


def decode_radio_rx_decision_event(event):
    """Return typed ID 45--46 data, leaving input values unjoined."""
    event_id = event['event']
    if event_id == RADIO_RX_DECISION:
        flags = event['f'] & UINT64_MASK
        reason_code = flags & 0xff
        return {
            'record_type': 'decision',
            'thread_ring': event['ring'],
            'decision_sequence': event['sequence'],
            'source_code': event['a'],
            'source': _enum_name(RADIO_RX_SOURCE_NAMES, event['a']),
            'generation': event['b'],
            'context_end_sample_ticks': event['c'],
            'recorder_mono_ns': event['mono_ns'],
            'recorder_realtime_ns': event['realtime_ns'],
            'mean_level_dbfs': _optional_milli_db(event['d']),
            'candidate_gain_db': _optional_milli_db(event['e']),
            'reason_code': reason_code,
            'reason': _enum_name(RADIO_RX_REASON_NAMES, reason_code),
            'flags_raw': flags,
            'unknown_flag_bits': flags & ~RADIO_RX_KNOWN_FLAG_MASK,
            'would_change': bool(flags & (1 << 8)),
            'submitted': bool(flags & (1 << 9)),
            'activity_valid': bool(flags & (1 << 10)),
            'search_failed': bool(flags & (1 << 11)),
            'phase': ('TRACKING' if flags & (1 << 13) else 'ACQUISITION') if flags & (1 << 12) else None,
            'input_join_status': 'missing',
        }
    if event_id == RADIO_RX_DECISION_INPUT:
        return {
            'record_type': 'input',
            'thread_ring': event['ring'],
            'input_sequence': event['sequence'],
            'source_code': event['a'],
            'generation': event['b'],
            'context_end_sample_ticks': event['c'],
            'input_recorder_mono_ns': event['mono_ns'],
            'input_recorder_realtime_ns': event['realtime_ns'],
            'reported_rx_gain_db': _optional_milli_db(event['d']),
            'peak_component_dbfs': _optional_milli_db(event['e']),
            'error_raw_millidb': event['f'],
        }
    return None


def decode_radio_rx_peak_envelope(event):
    """Decode standalone ID57 without assuming a neighboring decision survived."""
    if event['event'] != 57:
        return None
    return dict(source_file=event['source_file'], source_line=event['source_line'],
                thread_ring=event['ring'], sequence=event['sequence'],
                recorder_mono_ns=event['mono_ns'], recorder_realtime_ns=event['realtime_ns'],
                source_code=event['a'], source=_enum_name(RADIO_RX_SOURCE_NAMES, event['a']),
                generation=event['b'], context_end_sample_ticks=event['c'],
                retained_peak_at_current_gain_dbfs=_optional_milli_db(event['d']),
                reported_rx_gain_db=_optional_milli_db(event['e']),
                peak_release_db_per_second=_optional_milli_db(event['f']))


def _radio_rx_decision_key(record):
    return (record['thread_ring'], record['source_code'], record['generation'], record['context_end_sample_ticks'])


def _join_radio_rx_decision_inputs(decisions, inputs):
    """Join only an unambiguous decision/input sequence from one recorder ring."""
    by_key = {}
    for decision in decisions:
        by_key.setdefault(_radio_rx_decision_key(decision), []).append(('decision', decision))
    for input_record in inputs:
        by_key.setdefault(_radio_rx_decision_key(input_record), []).append(('input', input_record))

    joined = 0
    for records in by_key.values():
        decisions_for_key = [record for kind, record in records if kind == 'decision']
        inputs_for_key = [record for kind, record in records if kind == 'input']
        if not decisions_for_key or not inputs_for_key:
            continue
        ordered = sorted(records,
                         key=lambda item: item[1]['decision_sequence']
                         if item[0] == 'decision' else item[1]['input_sequence'])
        sequences = [record['decision_sequence'] if kind == 'decision' else record['input_sequence']
                     for kind, record in ordered]
        pairs_are_ordered = (len(decisions_for_key) == len(inputs_for_key)
                             and all(sequences[index] < sequences[index + 1] for index in range(len(sequences) - 1))
                             and all(ordered[index][0] == 'decision' and ordered[index + 1][0] == 'input'
                                     for index in range(0, len(ordered), 2)))
        if not pairs_are_ordered:
            for decision in decisions_for_key:
                decision['input_join_status'] = 'ambiguous'
            continue
        for index in range(0, len(ordered), 2):
            decision = ordered[index][1]
            input_record = ordered[index + 1][1]
            decision['input_join_status'] = 'joined'
            decision['input_source_file'] = input_record['source_file']
            decision['input_source_line'] = input_record['source_line']
            decision['input_sequence'] = input_record['input_sequence']
            decision['input_recorder_mono_ns'] = input_record['input_recorder_mono_ns']
            decision['input_recorder_realtime_ns'] = input_record['input_recorder_realtime_ns']
            decision['reported_rx_gain_db'] = input_record['reported_rx_gain_db']
            decision['peak_component_dbfs'] = input_record['peak_component_dbfs']
            decision['error_raw_millidb'] = input_record['error_raw_millidb']
            if decision['reason_code'] in RADIO_RX_ERROR_REASONS:
                decision['error_db'] = _optional_milli_db(input_record['error_raw_millidb'])
            joined += 1
    return joined


def _unpack_frame_slot(value):
    if value == INT64_MIN:
        return None, None
    return value // 1000, value % 1000


def _unpack_tx_frame_slot(value):
    if value < 0:
        return None, None
    return _unpack_frame_slot(value)


def _decode_ue_tx_resource(channel, value):
    empty = dict(resource_raw=None, resource_first=None, resource_count=None,
                 start_symbol=None, symbol_count=None, detail_0=None, detail_1=None)
    if value == INT64_MIN:
        return empty
    resource = value & UINT64_MASK
    common = dict(resource_raw=resource)
    if channel == 1:
        return dict(**common,
                    resource_first=(resource >> 32) & 0xffff,
                    resource_count=(resource >> 16) & 0xff,
                    start_symbol=(resource >> 24) & 0xff,
                    symbol_count=None,
                    detail_0=(resource >> 8) & 0xff,
                    detail_1=resource & 0xff)
    if channel == 2:
        return dict(**common,
                    resource_first=resource & 0xffff,
                    resource_count=(resource >> 16) & 0xffff,
                    start_symbol=(resource >> 32) & 0xff,
                    symbol_count=(resource >> 40) & 0xff,
                    detail_0=(resource >> 48) & 0xff,
                    detail_1=(resource >> 56) & 0xff)
    if channel == 3:
        return dict(**common,
                    resource_first=resource & 0xffff,
                    resource_count=(resource >> 16) & 0xffff,
                    start_symbol=(resource >> 32) & 0xff,
                    symbol_count=(resource >> 40) & 0xff,
                    detail_0=(resource >> 48) & 0xff,
                    detail_1=(resource >> 56) & 0xff)
    if channel == 4:
        return dict(**common,
                    resource_first=resource & 0xffff,
                    resource_count=(resource >> 16) & 0xffff,
                    start_symbol=(resource >> 32) & 0xff,
                    symbol_count=(resource >> 40) & 0xff,
                    detail_0=(resource >> 48) & 0xff,
                    detail_1=(resource >> 56) & 0xff)
    return dict(**common, resource_first=None, resource_count=None, start_symbol=None,
                symbol_count=None, detail_0=None, detail_1=None)


def decode_tx_reference_event(event):
    """Return typed ID 43--44 payload data, or None for another numeric event."""
    if event['event'] == UE_TX_POWER_REQUEST:
        frame, slot = _unpack_frame_slot(event['b'])
        return {
            'record_type': 'ue_tx_power_request',
            'ue': event['a'],
            'frame': frame,
            'slot': slot,
            'channel_code': event['c'],
            'channel': _enum_name(UE_TX_CHANNEL_NAMES, event['c']),
            'requested_power_dbm': _optional_int64(event['d']),
            'generator_reference': _optional_int64(event['e']),
            **_decode_ue_tx_resource(event['c'], event['f']),
            'recorder_mono_ns': event['mono_ns'],
            'recorder_realtime_ns': event['realtime_ns'],
        }
    if event['event'] == GNB_TX_REFERENCE:
        frame, slot = _unpack_frame_slot(event['a'])
        resource = event['f'] & UINT64_MASK if event['f'] != INT64_MIN else None
        return {
            'record_type': 'gnb_tx_reference',
            'frame': frame,
            'slot': slot,
            'configured_ss_pbch_power_dbm': _optional_int64(event['b']),
            'tx_amp': _optional_int64(event['c']),
            'ssb_index': _optional_int64(event['d']),
            'ssb_start_symbol': _optional_int64(event['e']),
            'antenna_port': resource & 0xffff if resource is not None else None,
            'beam_index': (resource >> 16) & 0xffff if resource is not None else None,
            'ssb_start_subcarrier': resource >> 32 if resource is not None else None,
            'recorder_mono_ns': event['mono_ns'],
            'recorder_realtime_ns': event['realtime_ns'],
        }
    return None


def decode_radio_tx_power_event(event):
    """Return typed ID 49--52 TX power evidence without cross-record joins."""
    event_id = event['event']
    channel = event['a']
    frame, slot = _unpack_tx_frame_slot(event['b'])
    common = {
        'thread_ring': event['ring'],
        'channel_code': channel,
        'channel': _enum_name(UE_TX_CHANNEL_NAMES, channel),
        'frame_slot_raw': event['b'],
        'frame': frame,
        'slot': slot,
        '_frame_slot_raw': event['b'],
    }
    if event_id == RADIO_TX_POWER:
        flags = event['c'] & UINT64_MASK
        status_code = flags & 0xff
        status_ok = status_code == 0
        return {
            'record_type': 'power',
            'power_sequence': event['sequence'],
            'recorder_mono_ns': event['mono_ns'],
            'recorder_realtime_ns': event['realtime_ns'],
            'status_flags_raw': flags,
            'status_code': status_code,
            'status': _enum_name(RADIO_TX_POWER_STATUS_NAMES, status_code),
            'status_unknown_flag_bits': flags & ~RADIO_TX_POWER_KNOWN_STATUS_MASK,
            'applied': bool(flags & (1 << 8)),
            'requested_power_dbm': _optional_milli_db(event['d']),
            'estimated_output_power_dbm': _optional_milli_db(event['e']) if status_ok else None,
            'amplitude_coefficient_q30': _optional_int64(event['f']) if status_ok else None,
            'sample_join_status': 'missing',
            'quality_join_status': 'missing',
            **common,
        }
    if event_id == RADIO_TX_POWER_SAMPLES:
        packed = event['f'] & UINT64_MASK
        return {
            'record_type': 'samples',
            'sample_sequence': event['sequence'],
            'sample_recorder_mono_ns': event['mono_ns'],
            'sample_recorder_realtime_ns': event['realtime_ns'],
            'sample_count': _optional_int64(event['c']),
            'input_energy': _optional_int64(event['d']),
            '_output_energy_raw': event['e'],
            'input_peak_component': packed >> 32,
            'output_peak_component': packed & 0xffffffff,
            **common,
        }
    if event_id == RADIO_TX_POWER_QUALITY:
        return {
            'record_type': 'quality',
            'quality_sequence': event['sequence'],
            'quality_recorder_mono_ns': event['mono_ns'],
            'quality_recorder_realtime_ns': event['realtime_ns'],
            '_quantization_power_error_raw_millidb': event['c'],
            '_quantization_evm_raw_ppb': event['d'],
            'profile_uncertainty_raw_millidb': event['e'],
            'profile_uncertainty_db': _optional_milli_db(event['e']),
            'component_full_scale': _optional_int64(event['f']),
            **common,
        }
    if event_id == RADIO_TX_REJECT:
        return {
            'record_type': 'reject',
            'sequence': event['sequence'],
            'radio': event['a'],
            'frame_slot_raw': event['b'],
            'frame': frame,
            'slot': slot,
            'channel_code': event['c'],
            'channel': _enum_name(RADIO_TX_REJECT_CHANNEL_NAMES, event['c']),
            'reason_code': event['d'],
            'reason': _enum_name(RADIO_TX_REJECT_REASON_NAMES, event['d']),
            'actuation_requested_raw': event['e'],
            'actuation_requested': _optional_bool(event['e']),
            'recorder_mono_ns': event['mono_ns'],
            'recorder_realtime_ns': event['realtime_ns'],
        }
    return None


def _radio_tx_power_context_key(record):
    if record['_frame_slot_raw'] < 0:
        return None
    return record['thread_ring'], record['channel_code'], record['_frame_slot_raw']


def _join_radio_tx_sample(power, sample):
    power.update(sample_join_status='joined',
                 sample_source_file=sample['source_file'],
                 sample_source_line=sample['source_line'],
                 sample_sequence=sample['sample_sequence'],
                 sample_recorder_mono_ns=sample['sample_recorder_mono_ns'],
                 sample_recorder_realtime_ns=sample['sample_recorder_realtime_ns'],
                 sample_count=sample['sample_count'],
                 input_energy=sample['input_energy'],
                 input_peak_component=sample['input_peak_component'],
                 output_peak_component=sample['output_peak_component'],
                 output_energy=_optional_int64(sample['_output_energy_raw']) if power['status_code'] == 0 else None)


def _join_radio_tx_quality(power, quality):
    status_ok = power['status_code'] == 0
    power.update(quality_join_status='joined',
                 quality_source_file=quality['source_file'],
                 quality_source_line=quality['source_line'],
                 quality_sequence=quality['quality_sequence'],
                 quality_recorder_mono_ns=quality['quality_recorder_mono_ns'],
                 quality_recorder_realtime_ns=quality['quality_recorder_realtime_ns'],
                 quantization_power_error_raw_millidb=quality['_quantization_power_error_raw_millidb'],
                 quantization_power_error_db=(_optional_milli_db(quality['_quantization_power_error_raw_millidb'])
                                              if status_ok else None),
                 quantization_evm_raw_ppb=quality['_quantization_evm_raw_ppb'],
                 quantization_evm_ppb=(_optional_int64(quality['_quantization_evm_raw_ppb']) if status_ok else None),
                 profile_uncertainty_raw_millidb=quality['profile_uncertainty_raw_millidb'],
                 profile_uncertainty_db=quality['profile_uncertainty_db'],
                 component_full_scale=quality['component_full_scale'])


def _proven_tx_group(power, sample, quality, sequence_owners):
    sequences = (power['power_sequence'], sample['sample_sequence'], quality['quality_sequence'])
    first, middle, last = sequences
    # Bound proof work even on corrupt or extremely separated sequence values.
    if not (0 <= first < middle < last and last - first < 256):
        return False
    if not (power['recorder_mono_ns'] <= sample['sample_recorder_mono_ns'] <= quality['quality_recorder_mono_ns']):
        return False
    ring = power['thread_ring']
    for sequence in range(first, last + 1):
        owner = sequence_owners.get(sequence)
        if owner is None or (owner == ring) != (sequence in sequences):
            return False
    return True


def _join_radio_tx_power_evidence(powers, samples, qualities, sequence_owners):
    """Join only a complete synchronous ID 49 -> 50 -> 52 emission group."""
    grouped = {}
    for record in [*powers, *samples, *qualities]:
        key = _radio_tx_power_context_key(record)
        if key is not None:
            grouped.setdefault(key, []).append(record)

    samples_joined = 0
    qualities_joined = 0
    for records in grouped.values():
        ordered = sorted(records,
                         key=lambda record: (record['power_sequence'] if record['record_type'] == 'power'
                                             else record['sample_sequence'] if record['record_type'] == 'samples'
                                             else record['quality_sequence']))
        sequences = [record['power_sequence'] if record['record_type'] == 'power'
                     else record['sample_sequence'] if record['record_type'] == 'samples'
                     else record['quality_sequence'] for record in ordered]
        power_positions = [index for index, record in enumerate(ordered) if record['record_type'] == 'power']
        if any(sequences[index] >= sequences[index + 1] for index in range(len(sequences) - 1)):
            for position in power_positions:
                ordered[position]['sample_join_status'] = 'ambiguous'
                ordered[position]['quality_join_status'] = 'ambiguous'
            continue

        for index, position in enumerate(power_positions):
            next_position = power_positions[index + 1] if index + 1 < len(power_positions) else len(ordered)
            power = ordered[position]
            interval = ordered[position + 1:next_position]
            matching_samples = [record for record in interval if record['record_type'] == 'samples']
            matching_qualities = [record for record in interval if record['record_type'] == 'quality']
            complete_group = (len(matching_samples) == 1
                              and len(matching_qualities) == 1
                              and _proven_tx_group(power, matching_samples[0], matching_qualities[0], sequence_owners))
            if complete_group:
                _join_radio_tx_sample(power, matching_samples[0])
                _join_radio_tx_quality(power, matching_qualities[0])
                span = matching_qualities[0]['quality_sequence'] - power['power_sequence'] + 1
                power.update(evidence_group_sequence_span=span, evidence_interleaved_events=span - 3,
                             evidence_join_basis='global_consecutive' if span == 3 else 'complete_cross_ring_coverage')
                samples_joined += 1
                qualities_joined += 1
            elif matching_samples or matching_qualities:
                # A later same-context sidecar is indistinguishable from a dropped
                # ID 49 header without this complete producer emission sequence.
                power['sample_join_status'] = 'ambiguous'
                power['quality_join_status'] = 'ambiguous'
    return samples_joined, qualities_joined


def _write_radio_tx_power_outputs(output, powers, samples, qualities, rejects, sequence_owners):
    samples_joined, qualities_joined = _join_radio_tx_power_evidence(powers, samples, qualities, sequence_owners)
    _write_csv(output / 'radio_tx_power.csv', RADIO_TX_POWER_FIELDS, powers)
    _write_csv(output / 'radio_tx_rejects.csv', RADIO_TX_REJECT_FIELDS, rejects)
    return samples_joined, qualities_joined


def _write_radio_tx_level_output(output, events):
    """A whole selected buffer, not active-symbol power or proof of RF emission."""
    levels = [e for e in events if e['event'] == 47]
    groups = {}
    for event in events:
        key = (event['ring'], event['a'], event['b'], event['c'])
        groups.setdefault(key, []).append(event)
    rows = []
    joined = 0
    for event in levels:
        count, full_scale, energy = event['c'], event['d'], event['e']
        packed = event['f'] & UINT64_MASK
        peak, over_range = packed >> 32, packed & 0xffffffff
        valid = (0 < count <= 65536 and 0 < full_scale <= 32768 and 0 <= peak <= 32768
                 and peak * peak <= energy <= 2 * count * peak * peak
                 and 0 <= over_range <= 2 * count and (over_range == 0 or peak >= full_scale))
        mean = energy / (count * full_scale * full_scale) if valid else None
        row = dict(source_file=event['source_file'], source_line=event['source_line'],
                   radio=event['a'], thread_ring=event['ring'], sequence=event['sequence'],
                   recorder_mono_ns=event['mono_ns'], sample_start_ticks=event['b'], sample_count=count,
                   component_full_scale=full_scale, sum_squared_components=energy,
                   peak_component=peak, over_range_components=over_range, level_valid=valid,
                   mean_power_fs=mean, mean_power_dbfs=10 * math.log10(mean) if mean else None,
                   peak_component_dbfs=20 * math.log10(peak/full_scale) if valid and peak else None,
                   state_joined=False, backend_return_samples=None, tx_readback_db=None, settings_generation=None)
        candidates = groups[(event['ring'], event['a'], event['b'], event['c'])]
        if len(candidates) == 2:
            metadata = next((e for e in candidates if e['event'] == 48 and e['sequence'] > event['sequence']), None)
            if metadata is not None:
                row.update(state_joined=True, backend_return_samples=metadata['d'],
                           tx_readback_db=_optional_milli_db(metadata['e']),
                           settings_generation=_optional_int64(metadata['f']))
                joined += 1
        rows.append(row)
    fields = ['source_file', 'source_line', 'radio', 'thread_ring', 'sequence', 'recorder_mono_ns',
              'sample_start_ticks', 'sample_count', 'component_full_scale', 'sum_squared_components',
              'peak_component', 'over_range_components', 'level_valid', 'mean_power_fs', 'mean_power_dbfs',
              'peak_component_dbfs', 'state_joined', 'backend_return_samples', 'tx_readback_db', 'settings_generation']
    _write_csv(output / 'radio_tx_level.csv', fields, rows)
    return len(rows), joined


def decode_ue_tx_control(event):
    if event['event'] != 53:
        return None
    return {
        'source_file': event['source_file'], 'source_line': event['source_line'],
        'thread_ring': event['ring'], 'sequence': event['sequence'],
        'recorder_mono_ns': event['mono_ns'], 'recorder_realtime_ns': event['realtime_ns'],
        'channel': _enum_name(UE_TX_CHANNEL_NAMES, event['a']),
        'frame_slot_raw': event['b'], 'frame': event['b'] // 1000 if event['b'] >= 0 else None,
        'slot': event['b'] % 1000 if event['b'] >= 0 else None,
        'ssb_pathloss_db': event['c'], 'adjustment_state_after_db': event['d'],
        'provided_tpc_delta_db': event['e'], 'configured_network_pmax_dbm': _optional_int64(event['f']),
    }


def decode_ue_pathloss_state(event):
    if event['event'] != 54:
        return None
    available = event['a']
    if available not in (0, 1):
        raise ValueError('invalid pathloss availability flag')
    pathloss = _optional_int64(event['e'])
    rsrp = _optional_int64(event['c'])
    if available and (pathloss is None or not 0 <= pathloss <= 32767 or rsrp is None):
        raise ValueError('available pathloss lacks valid measurement')
    if not available and pathloss is not None:
        raise ValueError('unavailable pathloss has a numeric value')
    return {
        'source_file': event['source_file'], 'source_line': event['source_line'],
        'thread_ring': event['ring'], 'sequence': event['sequence'],
        'recorder_mono_ns': event['mono_ns'], 'recorder_realtime_ns': event['realtime_ns'],
        'available': bool(available), 'ssb_index': event['b'], 'rsrp_dbm': rsrp,
        'ssb_reference_dbm': event['d'], 'ssb_pathloss_db': pathloss, 'mac_state': event['f'],
    }


def decode_ue_ssb_measurement(event):
    if event['event'] not in (55, 56):
        return None
    row = {key: event[key] for key in ('source_file', 'source_line')}
    row.update(thread_ring=event['ring'], sequence=event['sequence'],
               recorder_mono_ns=event['mono_ns'], recorder_realtime_ns=event['realtime_ns'])
    if event['event'] == 56:
        counts = event['f'] & UINT64_MASK
        row.update(record_type='context', generation=event['a'],
                   first_sample_ticks=_optional_int64(event['b']), end_sample_ticks=_optional_int64(event['c']),
                   reported_rx_gain_db=_optional_milli_db(event['d']), peak_component_dbfs=_optional_milli_db(event['e']),
                   sampled_components=counts & 0xffffffff, near_rail_components=counts >> 32)
        return row
    flags = event['f'] & UINT64_MASK
    accepted = bool(flags & 128)
    pbch_checked = bool(flags & 256)
    pbch_success = bool(flags & 512)
    pbch_confirmation_required = bool(flags & 1024)
    rsrp = _optional_int64(event['e'])
    if accepted != (rsrp is not None) or not 0 <= event['d'] <= 0xffffffff:
        raise ValueError('inconsistent SSB measurement acceptance or raw energy')
    if pbch_success and not pbch_checked:
        raise ValueError('SSB PBCH success without a decode check')
    if accepted and pbch_confirmation_required and not (pbch_checked and pbch_success):
        raise ValueError('SSB accepted without required PBCH confirmation')
    row.update(record_type='measurement', frame_slot_raw=event['a'],
               frame=event['a'] // 1000 if event['a'] >= 0 else None,
               slot=event['a'] % 1000 if event['a'] >= 0 else None,
               ssb_index=event['b'], generation=event['c'], raw_sss_mean_energy=event['d'],
               accepted=accepted, accepted_rsrp_dbm=rsrp, flags_raw=flags, unknown_flag_bits=flags & ~2047,
               context_present=bool(flags & 1), context_gain_valid=bool(flags & 2),
               context_level_valid=bool(flags & 4), generation_current=bool(flags & 8),
               gain_normalization_eligible=bool(flags & 16), noise_checked=bool(flags & 32),
               noise_current=bool(flags & 64), pbch_checked=pbch_checked, pbch_success=pbch_success,
               pbch_confirmation_required=pbch_confirmation_required, context_join_status='missing')
    return row


def _write_ue_ssb_measurements(output, measurements, contexts):
    by_sequence = {}
    headers = {}
    for context in contexts:
        by_sequence.setdefault((context['thread_ring'], context['sequence']), []).append(context)
    for row in measurements:
        key = (row['thread_ring'], row['sequence'])
        headers[key] = headers.get(key, 0) + 1
    joined = 0
    context_fields = ['first_sample_ticks', 'end_sample_ticks', 'reported_rx_gain_db',
                      'peak_component_dbfs', 'sampled_components', 'near_rail_components']
    for row in measurements:
        key = (row['thread_ring'], row['sequence'])
        candidates = by_sequence.get((row['thread_ring'], row['sequence'] + 1), [])
        if not candidates:
            continue
        row['context_join_status'] = 'ambiguous'
        if headers[key] != 1 or len(candidates) != 1:
            continue
        context = candidates[0]
        if context['generation'] != row['generation'] or context['recorder_mono_ns'] < row['recorder_mono_ns']:
            continue
        row.update({field: context[field] for field in context_fields})
        row.update(context_join_status='joined', context_source_file=context['source_file'],
                   context_source_line=context['source_line'], context_sequence=context['sequence'])
        joined += 1
    common = ['source_file', 'source_line', 'thread_ring', 'sequence', 'recorder_mono_ns', 'recorder_realtime_ns']
    fields = common + ['frame_slot_raw', 'frame', 'slot', 'ssb_index', 'generation', 'raw_sss_mean_energy',
                       'accepted', 'accepted_rsrp_dbm', 'flags_raw', 'unknown_flag_bits', 'context_present',
                       'context_gain_valid', 'context_level_valid', 'generation_current', 'gain_normalization_eligible',
                       'noise_checked', 'noise_current', 'pbch_checked', 'pbch_success',
                       'pbch_confirmation_required', 'context_join_status', 'context_source_file',
                       'context_source_line', 'context_sequence'] + context_fields
    _write_csv(output / 'ue_ssb_measurements.csv', fields, measurements)
    _write_csv(output / 'ue_ssb_measurement_contexts.csv', common + ['generation'] + context_fields, contexts)
    return joined


def _write_csv(path, fields, rows):
    with path.open('w', newline='') as dst:
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_radio_outputs(output, results, times, levels):
    results_by_key = {}
    times_by_key = {}
    for result in results:
        results_by_key.setdefault((result['radio'], result['generation']), []).append(result)
    for time in times:
        times_by_key.setdefault((time['radio'], time['generation']), []).append(time)

    for result in results:
        key = (result['radio'], result['generation'])
        candidates = times_by_key.get(key, [])
        # A generation can have failed/retried requests. Without a one-to-one key,
        # ordering does not prove which timing record belongs to which result.
        if len(results_by_key[key]) != 1 or len(candidates) != 1:
            continue
        timing = candidates[0]
        result['time_metadata_joined'] = True
        result['time_valid_flags'] = timing['valid_flags']
        result['time_unknown_valid_flags'] = timing['unknown_valid_flags']
        result['request_id'] = timing['request_id']
        if result['operation_code'] in (RADIO_GAIN_SET_RX, RADIO_GAIN_SET_TX) and timing['request_id'] != 0:
            result['requested_gain_db'] = _optional_milli_db(result['_requested_gain_mdb'])
        if result['device_time_valid'] and timing['device_time_valid']:
            result['begin_device_ticks'] = timing['begin_device_ticks']
            result['end_device_ticks'] = timing['end_device_ticks']

    _write_csv(output / 'radio_gain.csv', RADIO_GAIN_FIELDS, results)
    _write_csv(output / 'radio_rx_level.csv', RADIO_RX_LEVEL_FIELDS, levels)


def _write_radio_rx_decision_output(output, decisions, inputs):
    joined = _join_radio_rx_decision_inputs(decisions, inputs)
    _write_csv(output / 'radio_rx_decisions.csv', RADIO_RX_DECISION_FIELDS, decisions)
    return joined


def _write_tx_reference_outputs(output, ue_records, gnb_records):
    _write_csv(output / 'ue_tx_power.csv', UE_TX_POWER_FIELDS, ue_records)
    _write_csv(output / 'gnb_tx_reference.csv', GNB_TX_REFERENCE_FIELDS, gnb_records)


def decode(directory, output):
    paths = sorted(directory.glob('oai-flight-recorder-*.ndjson'))
    if not paths:
        raise ValueError('expected recorder files from one process')
    identities = []
    slots = set()
    for path in paths:
        match = re.fullmatch(r'oai-flight-recorder-([0-9]+-[0-9a-f]{16})-([0-9]+)\.ndjson', path.name)
        if match is None or match[2] in slots:
            raise ValueError('invalid recorder filename or duplicate file slot')
        identities.append(match[1])
        slots.add(match[2])
    if len(set(identities)) != 1:
        raise ValueError('mixed process/capture identities; decode each run separately')
    output.mkdir(parents=True, exist_ok=True)
    result = dict(files=len(paths), events=0, invalid_lines=0, unsupported_schema=0,
                  footers=[], health=[], retention=[], event_counts={})
    # Global sequence ownership proves gaps are retained events from other rings.
    # A duplicate is explicitly ambiguous, even if both copies name the same ring.
    sequence_owners = {}
    radio_results = []
    radio_times = []
    radio_levels = []
    radio_rx_decisions = []
    radio_rx_decision_inputs = []
    radio_rx_peak_envelopes = []
    ue_tx_power_records = []
    ue_tx_control_records = []
    ue_pathloss_state_records = []
    ue_ssb_measurements = []
    ue_ssb_contexts = []
    gnb_tx_reference_records = []
    radio_tx_power_records = []
    radio_tx_power_samples = []
    radio_tx_power_qualities = []
    radio_tx_rejects = []
    radio_tx_level_events = []
    with (output / 'events.csv').open('w', newline='') as dst:
        writer = csv.DictWriter(dst, fieldnames=FIELDS)
        writer.writeheader()
        for path in paths:
            with path.open('rb') as source:
                number = 0
                while True:
                    line = source.readline(4097)
                    if not line:
                        break
                    number += 1
                    if len(line) > 4096 or not line.endswith(b'\n'):
                        result['invalid_lines'] += 1
                        while line and not line.endswith(b'\n'):
                            line = source.readline(4097)
                        continue
                    try:
                        obj = json.loads(line)
                        if not isinstance(obj, dict):
                            raise ValueError()
                        if (obj.get('schema') != 'oai.flight_recorder' or type(obj.get('version')) is not int
                                or obj['version'] != 1):
                            result['unsupported_schema'] += 1
                            continue
                        kind = obj.get('kind')
                        if kind == 'event':
                            values = {k: obj[k] for k in FIELDS[3:]}
                            if any(type(v) is not int for v in values.values()):
                                raise ValueError()
                            sequence, ring = values['sequence'], values['ring']
                            sequence_owners[sequence] = ring if sequence not in sequence_owners and 0 <= ring < 64 else None
                            name = NAMES.get(obj['event'], 'UNKNOWN')
                            writer.writerow(dict(source_file=path.name, source_line=number, name=name, **values))
                            decoded = decode_radio_gain_event(values)
                            if decoded is not None:
                                decoded['source_file'] = path.name
                                decoded['source_line'] = number
                                if decoded['record_type'] == 'result':
                                    radio_results.append(decoded)
                                elif decoded['record_type'] == 'time':
                                    radio_times.append(decoded)
                                else:
                                    radio_levels.append(decoded)
                            rx_decoded = decode_radio_rx_decision_event(values)
                            if rx_decoded is not None:
                                rx_decoded['source_file'] = path.name
                                rx_decoded['source_line'] = number
                                if rx_decoded['record_type'] == 'decision':
                                    radio_rx_decisions.append(rx_decoded)
                                else:
                                    radio_rx_decision_inputs.append(rx_decoded)
                            envelope = decode_radio_rx_peak_envelope(dict(values, source_file=path.name, source_line=number))
                            if envelope is not None:
                                radio_rx_peak_envelopes.append(envelope)
                            tx_decoded = decode_tx_reference_event(values)
                            if tx_decoded is not None:
                                tx_decoded['source_file'] = path.name
                                tx_decoded['source_line'] = number
                                if tx_decoded['record_type'] == 'ue_tx_power_request':
                                    ue_tx_power_records.append(tx_decoded)
                                else:
                                    gnb_tx_reference_records.append(tx_decoded)
                            tx_power_decoded = decode_radio_tx_power_event(values)
                            if tx_power_decoded is not None:
                                tx_power_decoded['source_file'] = path.name
                                tx_power_decoded['source_line'] = number
                                if tx_power_decoded['record_type'] == 'power':
                                    radio_tx_power_records.append(tx_power_decoded)
                                elif tx_power_decoded['record_type'] == 'samples':
                                    radio_tx_power_samples.append(tx_power_decoded)
                                elif tx_power_decoded['record_type'] == 'quality':
                                    radio_tx_power_qualities.append(tx_power_decoded)
                                else:
                                    radio_tx_rejects.append(tx_power_decoded)
                            control = decode_ue_tx_control(dict(values, source_file=path.name, source_line=number))
                            if control is not None:
                                ue_tx_control_records.append(control)
                            pathloss_state = decode_ue_pathloss_state(dict(values, source_file=path.name, source_line=number))
                            if pathloss_state is not None:
                                ue_pathloss_state_records.append(pathloss_state)
                            ssb = decode_ue_ssb_measurement(dict(values, source_file=path.name, source_line=number))
                            if ssb is not None:
                                (ue_ssb_measurements if ssb['record_type'] == 'measurement' else ue_ssb_contexts).append(ssb)
                            if values['event'] in (47, 48):
                                radio_tx_level_events.append(dict(values, source_file=path.name, source_line=number))
                            result['events'] += 1
                            result['event_counts'][name] = result['event_counts'].get(name, 0) + 1
                        elif kind in ('capture_footer', 'health', 'capture_health', 'file_begin'):
                            if kind == 'capture_footer':
                                required = ('clean', 'writer_errors', 'dropped_ring_full', 'dropped_no_slot',
                                            'no_slot_threads', 'invalid_timestamp_records', 'payload_truncated', 'rings_assigned')
                                if any(type(obj.get(k)) is not int or obj[k] < 0 for k in required):
                                    raise ValueError()
                                if obj['clean'] != 1 or obj['writer_errors'] != 0 or obj['rings_assigned'] > 64:
                                    raise ValueError()
                            numeric = {k: v for k, v in obj.items() if type(v) is int}
                            key = 'footers' if kind == 'capture_footer' else 'retention' if kind == 'file_begin' else 'health'
                            if len(result[key]) < 10000:
                                result[key].append(numeric)
                    except (ValueError, KeyError, TypeError, UnicodeDecodeError):
                        result['invalid_lines'] += 1
    _write_radio_outputs(output, radio_results, radio_times, radio_levels)
    rx_decision_inputs_joined = _write_radio_rx_decision_output(output, radio_rx_decisions, radio_rx_decision_inputs)
    _write_tx_reference_outputs(output, ue_tx_power_records, gnb_tx_reference_records)
    result['radio_tx_power_samples_joined'], result['radio_tx_power_qualities_joined'] = _write_radio_tx_power_outputs(
        output, radio_tx_power_records, radio_tx_power_samples, radio_tx_power_qualities, radio_tx_rejects, sequence_owners)
    result['radio_tx_levels'], result['radio_tx_level_states_joined'] = _write_radio_tx_level_output(output, radio_tx_level_events)
    result['radio_gain_results'] = len(radio_results)
    result['radio_gain_time_metadata'] = len(radio_times)
    result['radio_rx_levels'] = len(radio_levels)
    result['radio_rx_decisions'] = len(radio_rx_decisions)
    result['radio_rx_decision_inputs'] = len(radio_rx_decision_inputs)
    result['radio_rx_decision_inputs_joined'] = rx_decision_inputs_joined
    peak_fields = ['source_file', 'source_line', 'thread_ring', 'sequence', 'recorder_mono_ns', 'recorder_realtime_ns',
                   'source_code', 'source', 'generation', 'context_end_sample_ticks',
                   'retained_peak_at_current_gain_dbfs', 'reported_rx_gain_db', 'peak_release_db_per_second']
    _write_csv(output / 'radio_rx_peak_envelope.csv', peak_fields, radio_rx_peak_envelopes)
    result['radio_rx_peak_envelopes'] = len(radio_rx_peak_envelopes)
    control_fields = ['source_file', 'source_line', 'thread_ring', 'sequence', 'recorder_mono_ns', 'recorder_realtime_ns',
                      'channel', 'frame_slot_raw', 'frame', 'slot', 'ssb_pathloss_db', 'adjustment_state_after_db',
                      'provided_tpc_delta_db', 'configured_network_pmax_dbm']
    _write_csv(output / 'ue_tx_control.csv', control_fields, ue_tx_control_records)
    result['ue_tx_control_records'] = len(ue_tx_control_records)
    pathloss_fields = ['source_file', 'source_line', 'thread_ring', 'sequence', 'recorder_mono_ns', 'recorder_realtime_ns',
                       'available', 'ssb_index', 'rsrp_dbm', 'ssb_reference_dbm', 'ssb_pathloss_db', 'mac_state']
    _write_csv(output / 'ue_pathloss_state.csv', pathloss_fields, ue_pathloss_state_records)
    result['ue_pathloss_state_records'] = len(ue_pathloss_state_records)
    result['ue_ssb_contexts_joined'] = _write_ue_ssb_measurements(output, ue_ssb_measurements, ue_ssb_contexts)
    result['ue_ssb_measurements'] = len(ue_ssb_measurements)
    result['ue_ssb_measurement_contexts'] = len(ue_ssb_contexts)
    result['ue_tx_power_requests'] = len(ue_tx_power_records)
    result['gnb_tx_references'] = len(gnb_tx_reference_records)
    result['radio_tx_powers'] = len(radio_tx_power_records)
    result['radio_tx_power_samples'] = len(radio_tx_power_samples)
    result['radio_tx_power_qualities'] = len(radio_tx_power_qualities)
    result['radio_tx_rejects'] = len(radio_tx_rejects)
    result['clean_footer_observed'] = len(result['footers']) == 1
    result['ordering'] = 'file order only; compare per-boot monotonic times, never assume continuous UTC'
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recorder_dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = decode(args.recorder_dir, args.output)
    except (ValueError, OSError) as error:
        parser.exit(2, str(error) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
