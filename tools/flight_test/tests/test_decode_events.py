#!/usr/bin/env python3
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('decoder', Path(__file__).resolve().parents[1] / 'decode_events.py')
decoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decoder)


class DecoderTests(unittest.TestCase):
    @staticmethod
    def event(event_id, sequence, *, ring=0, a=0, b=0, c=0, d=0, e=0, f=0, mono_ns=None, realtime_ns=None):
        return dict(schema='oai.flight_recorder', version=1, kind='event', event=event_id,
                    ring=ring, sequence=sequence, mono_ns=sequence if mono_ns is None else mono_ns,
                    realtime_ns=sequence if realtime_ns is None else realtime_ns,
                    a=a, b=b, c=c, d=d, e=e, f=f)

    def test_peak_envelope_stands_alone_and_preserves_unavailable_values(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = [self.event(57, 100, a=2, b=8, c=300, d=-4800, e=54000, f=3000),
                      self.event(57, 200, a=99, b=9, c=400,
                                 d=decoder.INT64_MIN, e=decoder.INT64_MIN, f=decoder.INT64_MIN)]
            (root / 'oai-flight-recorder-100-0123456789abcdef-0.ndjson').write_text(
                ''.join(json.dumps(event) + '\n' for event in events))
            result = decoder.decode(root, root / 'out')
            self.assertEqual(result['invalid_lines'], 0)
            self.assertEqual(result['radio_rx_peak_envelopes'], 2)
            self.assertEqual(result['radio_rx_decisions'], 0)
            self.assertEqual(result['event_counts']['RADIO_RX_PEAK_ENVELOPE'], 2)
            with (root / 'out/radio_rx_peak_envelope.csv').open(newline='') as source:
                valid, unknown = list(csv.DictReader(source))
            self.assertEqual(valid['source'], 'GNB_PUSCH')
            self.assertEqual(valid['retained_peak_at_current_gain_dbfs'], '-4.8')
            self.assertEqual(valid['reported_rx_gain_db'], '54.0')
            self.assertEqual(valid['peak_release_db_per_second'], '3.0')
            self.assertEqual(unknown['retained_peak_at_current_gain_dbfs'], '')
            self.assertEqual(unknown['reported_rx_gain_db'], '')
            self.assertEqual(unknown['peak_release_db_per_second'], '')
            self.assertIn('99', unknown['source'])

    def test_ssb_acceptance_and_clipped_context(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = [self.event(55, 1, a=1002, b=0, c=7, d=1024, e=-80, f=255),
                      self.event(56, 2, a=7, b=100, c=200, d=17000, e=-6000, f=128),
                      self.event(55, 3, a=1004, b=0, c=8, d=2048, e=decoder.INT64_MIN, f=15),
                      self.event(56, 4, a=8, b=200, c=300, d=74000, e=0, f=(8 << 32) | 128)]
            (root / 'oai-flight-recorder-100-0123456789abcdef-0.ndjson').write_text(
                ''.join(json.dumps(event) + '\n' for event in events))
            result = decoder.decode(root, root / 'out')
            self.assertEqual(result['invalid_lines'], 0)
            self.assertEqual(result['ue_ssb_contexts_joined'], 2)
            with (root / 'out/ue_ssb_measurements.csv').open(newline='') as source:
                valid, clipped = list(csv.DictReader(source))
            self.assertEqual(valid['accepted_rsrp_dbm'], '-80')
            self.assertEqual(valid['reported_rx_gain_db'], '17.0')
            self.assertEqual(clipped['accepted'], 'False')
            self.assertEqual(clipped['accepted_rsrp_dbm'], '')
            self.assertEqual(clipped['context_gain_valid'], 'True')
            self.assertEqual(clipped['gain_normalization_eligible'], 'False')
            self.assertEqual(clipped['near_rail_components'], '8')
            self.assertEqual(clipped['peak_component_dbfs'], '0.0')

    def test_ssb_context_loss_and_duplicate_never_fabricate_join(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = [self.event(55, 10, a=1002, c=7, d=1, e=-80, f=255),
                      self.event(56, 100, a=7, b=100, c=200, d=17000, e=-6000, f=128),
                      self.event(55, 200, a=1002, c=7, d=1, e=-80, f=255),
                      self.event(56, 201, a=7, b=100, c=200, d=17000, e=-6000, f=128),
                      self.event(56, 201, a=7, b=100, c=200, d=17000, e=-6000, f=128)]
            (root / 'oai-flight-recorder-100-0123456789abcdef-0.ndjson').write_text(
                ''.join(json.dumps(event) + '\n' for event in events))
            result = decoder.decode(root, root / 'out')
            self.assertEqual(result['ue_ssb_contexts_joined'], 0)
            with (root / 'out/ue_ssb_measurements.csv').open(newline='') as source:
                missing, duplicate = list(csv.DictReader(source))
            self.assertEqual(missing['context_join_status'], 'missing')
            self.assertEqual(duplicate['context_join_status'], 'ambiguous')
            self.assertEqual(missing['reported_rx_gain_db'], '')
            self.assertEqual(duplicate['near_rail_components'], '')

    def test_ssb_inconsistent_acceptance_is_invalid(self):
        for flags, rsrp in ((255, decoder.INT64_MIN), (15, -80)):
            with self.assertRaises(ValueError):
                decoder.decode_ue_ssb_measurement(dict(self.event(55, 1, d=1, e=rsrp, f=flags),
                                                       source_file='test', source_line=1))

    def test_ssb_pbch_confirmation_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = [self.event(55, 1, d=100, e=decoder.INT64_MIN, f=127 | 256 | 1024),
                      self.event(55, 2, d=200, e=-57, f=255 | 256 | 512 | 1024),
                      self.event(55, 3, d=100, e=-97, f=255),
                      self.event(55, 4, d=200, e=decoder.INT64_MIN, f=127 | 1024)]
            (root / 'oai-flight-recorder-100-0123456789abcdef-0.ndjson').write_text(
                ''.join(json.dumps(event) + '\n' for event in events))
            result = decoder.decode(root, root / 'out')
            self.assertEqual(result['invalid_lines'], 0)
            with (root / 'out/ue_ssb_measurements.csv').open(newline='') as source:
                failed, confirmed, historical, unselected = list(csv.DictReader(source))
            self.assertEqual(failed['accepted'], 'False')
            self.assertEqual(failed['pbch_checked'], 'True')
            self.assertEqual(failed['pbch_success'], 'False')
            self.assertEqual(failed['accepted_rsrp_dbm'], '')
            self.assertEqual(confirmed['pbch_confirmation_required'], 'True')
            self.assertEqual(confirmed['pbch_success'], 'True')
            self.assertEqual(confirmed['accepted_rsrp_dbm'], '-57')
            self.assertEqual(confirmed['unknown_flag_bits'], '0')
            self.assertEqual(historical['pbch_confirmation_required'], 'False')
            self.assertEqual(historical['accepted_rsrp_dbm'], '-97')
            self.assertEqual(unselected['pbch_checked'], 'False')
            self.assertEqual(unselected['pbch_confirmation_required'], 'True')

    def test_ssb_invalid_confirmation_is_not_accepted(self):
        for flags, rsrp in ((255 | 1024, -57), (255 | 256 | 1024, -57),
                            (127 | 512 | 1024, decoder.INT64_MIN)):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                decoder.decode_ue_ssb_measurement(dict(self.event(55, 1, d=100, e=rsrp, f=flags),
                                                       source_file='test', source_line=1))

    def test_pathloss_unavailable_then_fresh_preserves_missing_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = [self.event(54, 1, a=0, b=0, c=decoder.INT64_MIN, d=-25, e=decoder.INT64_MIN, f=1),
                      self.event(54, 2, a=1, b=0, c=-86, d=-25, e=61, f=1),
                      self.event(54, 3, a=0, b=64, c=decoder.INT64_MIN, d=-25, e=decoder.INT64_MIN, f=2)]
            (root / 'oai-flight-recorder-100-0123456789abcdef-0.ndjson').write_text(
                ''.join(json.dumps(event) + '\n' for event in events))
            result = decoder.decode(root, root / 'out')
            self.assertEqual(result['ue_pathloss_state_records'], 3)
            self.assertEqual(result['invalid_lines'], 0)
            with (root / 'out/ue_pathloss_state.csv').open(newline='') as source:
                missing, valid, bad_index = list(csv.DictReader(source))
            self.assertEqual(missing['available'], 'False')
            self.assertEqual(missing['ssb_pathloss_db'], '')
            self.assertEqual(missing['rsrp_dbm'], '')
            self.assertEqual(valid['available'], 'True')
            self.assertEqual(valid['ssb_pathloss_db'], '61')
            self.assertEqual(valid['rsrp_dbm'], '-86')
            self.assertEqual(bad_index['ssb_index'], '64')
            self.assertEqual(bad_index['ssb_pathloss_db'], '')

    def test_pathloss_inconsistent_availability_rejected(self):
        for available, pathloss, rsrp in ((2, 61, -86), (0, 61, -86),
                                         (1, decoder.INT64_MIN, -86), (1, -1, -24),
                                         (1, 61, decoder.INT64_MIN)):
            event = self.event(54, 1, a=available, c=rsrp, e=pathloss)
            with self.assertRaises(ValueError):
                decoder.decode_ue_pathloss_state(dict(event, source_file='test', source_line=1))

    def test_mac_power_control_context_preserves_absent_ceiling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = [self.event(53, 1, a=2, b=1008, c=31, d=-2, e=3, f=decoder.INT64_MIN),
                      self.event(53, 2, a=3, b=1009, c=30, d=4, e=-1, f=-20)]
            (root / 'oai-flight-recorder-100-0123456789abcdef-0.ndjson').write_text(
                ''.join(json.dumps(event) + '\n' for event in events))
            result = decoder.decode(root, root / 'out')
            self.assertEqual(result['ue_tx_control_records'], 2)
            with (root / 'out/ue_tx_control.csv').open(newline='') as source:
                pusch, pucch = list(csv.DictReader(source))
            self.assertEqual(pusch['channel'], 'PUSCH')
            self.assertEqual(pusch['configured_network_pmax_dbm'], '')
            self.assertEqual(pusch['adjustment_state_after_db'], '-2')
            self.assertEqual(pucch['provided_tpc_delta_db'], '-1')
            self.assertEqual(pucch['configured_network_pmax_dbm'], '-20')
            self.assertEqual(pucch['frame'], '1')
            self.assertEqual(pucch['slot'], '9')

    def test_mixed_capture_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for slot, pid in enumerate((100, 101)):
                (root / f'oai-flight-recorder-{pid}-0123456789abcdef-{slot}.ndjson').write_text('')
            with self.assertRaises(ValueError):
                decoder.decode(root, root / 'out')
            self.assertFalse((root / 'out').exists())

    def test_more_than_eight_sequential_files_and_unknown_event(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index in range(12):
                event = dict(schema='oai.flight_recorder', version=1, kind='event', event=999,
                             ring=0, sequence=index + 1, mono_ns=index, realtime_ns=index,
                             a=0, b=0, c=0, d=0, e=0, f=0)
                (root / f'oai-flight-recorder-100-0123456789abcdef-{index}.ndjson').write_text(json.dumps(event) + '\n')
            result = decoder.decode(root, root / 'out')
            self.assertEqual(result['events'], 12)
            self.assertEqual(result['event_counts']['UNKNOWN'], 12)

    def test_radio_gain_events_are_typed_joined_only_by_radio_and_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = [
                self.event(11, 1, a=9, b=1),
                self.event(40, 2, a=3, b=7, c=12500, d=11000, e=12000, f=7 << 16, mono_ns=100),
                self.event(41, 3, a=3, b=7, c=99, d=900001, e=900101, f=7, mono_ns=200),
                self.event(42, 4, a=3, b=-7, c=4000, d=-32000, e=-5000, f=(3 << 32) | 128, mono_ns=300),
                self.event(40, 5, a=3, b=8, c=9000, d=11000, e=decoder.INT64_MIN, f=1 << 16, mono_ns=400),
                self.event(41, 6, a=4, b=8, c=10, d=1, e=2, f=4, mono_ns=500),
                self.event(40, 7, a=3, b=1, c=decoder.INT64_MIN, d=10000, e=12000, f=7 << 16, mono_ns=600),
                self.event(41, 8, a=3, b=1, c=0, d=10, e=20, f=7, mono_ns=601),
                self.event(40, 9, a=3, b=9, c=5555, d=10000, e=12000, f=(2 << 8) | (7 << 16), mono_ns=700),
                self.event(41, 10, a=3, b=9, c=44, d=30, e=40, f=7, mono_ns=701),
            ]
            path = root / 'oai-flight-recorder-100-0123456789abcdef-0.ndjson'
            path.write_text(''.join(json.dumps(event) + '\n' for event in events))
            result = decoder.decode(root, root / 'out')

            self.assertEqual(result['event_counts']['UE_AGC'], 1)
            self.assertEqual(result['event_counts']['RADIO_GAIN'], 4)
            self.assertEqual(result['event_counts']['RADIO_GAIN_TIME'], 4)
            self.assertEqual(result['event_counts']['RADIO_RX_LEVEL'], 1)
            self.assertEqual(result['radio_gain_results'], 4)
            self.assertEqual(result['radio_gain_time_metadata'], 4)
            self.assertEqual(result['radio_rx_levels'], 1)
            with (root / 'out/events.csv').open(newline='') as source:
                event_names = [row['name'] for row in csv.DictReader(source)]
            self.assertEqual(event_names, ['UE_AGC', 'RADIO_GAIN', 'RADIO_GAIN_TIME', 'RADIO_RX_LEVEL', 'RADIO_GAIN', 'RADIO_GAIN_TIME',
                                           'RADIO_GAIN', 'RADIO_GAIN_TIME', 'RADIO_GAIN', 'RADIO_GAIN_TIME'])

            with (root / 'out/radio_gain.csv').open(newline='') as source:
                gains = list(csv.DictReader(source))
            joined = next(row for row in gains if row['generation'] == '7')
            self.assertEqual(joined['radio'], '3')
            self.assertEqual(joined['status'], 'OK')
            self.assertEqual(joined['operation'], 'SET_RX')
            self.assertEqual(joined['result_valid_flags'], '7')
            self.assertEqual(joined['result_unknown_valid_flags'], '0')
            self.assertEqual(joined['time_valid_flags'], '7')
            self.assertEqual(joined['request_id'], '99')
            self.assertEqual(joined['requested_gain_db'], '12.5')
            self.assertEqual(joined['rx_readback_db'], '11.0')
            self.assertEqual(joined['tx_readback_db'], '12.0')
            self.assertEqual(joined['begin_device_ticks'], '900001')
            self.assertEqual(joined['end_device_ticks'], '900101')
            self.assertEqual(joined['recorder_mono_ns'], '100')
            self.assertEqual(joined['time_metadata_joined'], 'True')

            unmatched = next(row for row in gains if row['generation'] == '8')
            self.assertEqual(unmatched['time_metadata_joined'], 'False')
            self.assertEqual(unmatched['request_id'], '')
            self.assertEqual(unmatched['requested_gain_db'], '')
            self.assertEqual(unmatched['begin_device_ticks'], '')
            self.assertEqual(unmatched['end_device_ticks'], '')

            startup = next(row for row in gains if row['generation'] == '1')
            self.assertEqual(startup['request_id'], '0')
            self.assertEqual(startup['requested_gain_db'], '')
            self.assertEqual(startup['begin_device_ticks'], '10')
            self.assertEqual(startup['end_device_ticks'], '20')

            retune = next(row for row in gains if row['generation'] == '9')
            self.assertEqual(retune['operation'], 'RETUNE')
            self.assertEqual(retune['request_id'], '44')
            self.assertEqual(retune['requested_gain_db'], '')

            with (root / 'out/radio_rx_level.csv').open(newline='') as source:
                levels = list(csv.DictReader(source))
            self.assertEqual(len(levels), 1)
            self.assertEqual(levels[0]['generation'], '7')
            self.assertEqual(levels[0]['context_valid'], 'False')
            self.assertEqual(levels[0]['sample_start_ticks'], '4000')
            self.assertEqual(levels[0]['mean_power_dbfs'], '-32.0')
            self.assertEqual(levels[0]['peak_component_dbfs'], '-5.0')
            self.assertEqual(levels[0]['near_rail_components'], '3')
            self.assertEqual(levels[0]['sampled_components'], '128')

    def test_radio_rx_decisions_join_only_ordered_matching_pairs_in_one_ring(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = [
                self.event(45, 100, ring=1, a=1, b=7, c=900, d=-18000, e=12000,
                           f=9 | (1 << 8) | (1 << 9) | (1 << 10) | (1 << 15)),
                self.event(46, 101, ring=1, a=1, b=7, c=900, d=9000, e=-1000, f=2000),
                # This is the same source/generation/end key, but a different recorder ring.
                self.event(45, 102, ring=2, a=1, b=7, c=900, d=-19000, e=8000, f=8 | (1 << 8)),
                self.event(46, 103, ring=2, a=1, b=7, c=900, d=8000, e=-500, f=1234),
                self.event(45, 104, ring=1, a=3, b=8, c=1000, d=decoder.INT64_MIN, e=decoder.INT64_MIN,
                           f=10 | (1 << 11)),
                # Two decisions followed by two inputs share one key; an equal sequence also breaks monotonic order.
                self.event(45, 105, ring=1, a=4, b=9, c=2000, d=-17000, e=6000, f=5),
                self.event(45, 106, ring=1, a=4, b=9, c=2000, d=-16000, e=7000, f=9),
                self.event(46, 106, ring=1, a=4, b=9, c=2000, d=6000, e=-200, f=1000),
                self.event(46, 107, ring=1, a=4, b=9, c=2000, d=7000, e=-100, f=2000),
            ]
            path = root / 'oai-flight-recorder-100-0123456789abcdef-0.ndjson'
            path.write_text(''.join(json.dumps(event) + '\n' for event in events))
            result = decoder.decode(root, root / 'out')

            self.assertEqual(result['event_counts']['RADIO_RX_DECISION'], 5)
            self.assertEqual(result['event_counts']['RADIO_RX_DECISION_INPUT'], 4)
            self.assertEqual(result['radio_rx_decisions'], 5)
            self.assertEqual(result['radio_rx_decision_inputs'], 4)
            self.assertEqual(result['radio_rx_decision_inputs_joined'], 2)
            with (root / 'out/events.csv').open(newline='') as source:
                self.assertEqual([row['name'] for row in csv.DictReader(source)],
                                 ['RADIO_RX_DECISION', 'RADIO_RX_DECISION_INPUT'] * 2
                                 + ['RADIO_RX_DECISION'] * 3 + ['RADIO_RX_DECISION_INPUT'] * 2)

            with (root / 'out/radio_rx_decisions.csv').open(newline='') as source:
                decisions = list(csv.DictReader(source))
            track = next(row for row in decisions if row['thread_ring'] == '1' and row['decision_sequence'] == '100')
            self.assertEqual(track['source'], 'UE_SSB')
            self.assertEqual(track['generation'], '7')
            self.assertEqual(track['context_end_sample_ticks'], '900')
            self.assertEqual(track['mean_level_dbfs'], '-18.0')
            self.assertEqual(track['candidate_gain_db'], '12.0')
            self.assertEqual(track['reason'], 'TRACK_LEVEL')
            self.assertEqual(track['flags_raw'], str(9 | (1 << 8) | (1 << 9) | (1 << 10) | (1 << 15)))
            self.assertEqual(track['unknown_flag_bits'], str(1 << 15))
            self.assertEqual(track['would_change'], 'True')
            self.assertEqual(track['submitted'], 'True')
            self.assertEqual(track['activity_valid'], 'True')
            self.assertEqual(track['search_failed'], 'False')
            self.assertEqual(track['input_join_status'], 'joined')
            self.assertEqual(track['input_sequence'], '101')
            self.assertEqual(track['reported_rx_gain_db'], '9.0')
            self.assertEqual(track['peak_component_dbfs'], '-1.0')
            self.assertEqual(track['error_raw_millidb'], '2000')
            self.assertEqual(track['error_db'], '2.0')

            overload = next(row for row in decisions if row['thread_ring'] == '2')
            self.assertEqual(overload['input_join_status'], 'joined')
            self.assertEqual(overload['reason'], 'REDUCE_OVERLOAD')
            self.assertEqual(overload['reported_rx_gain_db'], '8.0')
            self.assertEqual(overload['error_raw_millidb'], '1234')
            self.assertEqual(overload['error_db'], '')

            missing = next(row for row in decisions if row['source'] == 'UE_SEARCH')
            self.assertEqual(missing['input_join_status'], 'missing')
            self.assertEqual(missing['mean_level_dbfs'], '')
            self.assertEqual(missing['candidate_gain_db'], '')
            self.assertEqual(missing['search_failed'], 'True')
            self.assertEqual(missing['reported_rx_gain_db'], '')
            self.assertEqual(missing['error_raw_millidb'], '')

            ambiguous = [row for row in decisions if row['source'] == 'HEADROOM']
            self.assertEqual(len(ambiguous), 2)
            self.assertTrue(all(row['input_join_status'] == 'ambiguous' for row in ambiguous))
            self.assertTrue(all(row['input_sequence'] == '' for row in ambiguous))
            self.assertTrue(all(row['reported_rx_gain_db'] == '' for row in ambiguous))
            self.assertTrue(all(row['error_raw_millidb'] == '' for row in ambiguous))

        unknown = decoder.decode_radio_rx_decision_event(self.event(45, 109, a=99, f=64 | (1 << 14)))
        self.assertEqual(unknown['source'], 'UNKNOWN_99')
        self.assertEqual(unknown['reason'], 'UNKNOWN_64')
        self.assertEqual(unknown['unknown_flag_bits'], 1 << 14)
        self.assertIsNone(unknown['phase'])
        acquisition = decoder.decode_radio_rx_decision_event(self.event(45, 110, f=(1 << 12)))
        tracking = decoder.decode_radio_rx_decision_event(self.event(45, 111, f=(1 << 12) | (1 << 13)))
        self.assertEqual(acquisition['phase'], 'ACQUISITION')
        self.assertEqual(tracking['phase'], 'TRACKING')
        self.assertEqual(tracking['unknown_flag_bits'], 0)

    def test_tx_power_and_reference_events_decode_producer_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = [
                self.event(43, 1, a=7, b=1013, c=1, d=0, e=32767,
                           f=9 | (2 << 8) | (1 << 16) | (4 << 24) | (23 << 32)),
                self.event(43, 2, a=7, b=1013, c=2, d=18, e=32767,
                           f=4 | (64 << 16) | (2 << 32) | (12 << 40) | (4 << 48) | (2 << 56)),
                self.event(43, 3, a=7, b=1013, c=3, d=6, e=1024,
                           f=10 | (2 << 16) | (9 << 32) | (4 << 40) | (1 << 48) | (1 << 56)),
                self.event(43, 4, a=7, b=1013, c=4, d=-5, e=1024,
                           f=20 | (32 << 16) | (10 << 32) | (2 << 40) | (7 << 48) | (2 << 56)),
                self.event(44, 5, a=2000, b=0, c=0, d=3, e=0,
                           f=1 | (8 << 16) | (132 << 32)),
            ]
            path = root / 'oai-flight-recorder-100-0123456789abcdef-0.ndjson'
            path.write_text(''.join(json.dumps(event) + '\n' for event in events))
            result = decoder.decode(root, root / 'out')

            self.assertEqual(result['event_counts']['UE_TX_POWER_REQUEST'], 4)
            self.assertEqual(result['event_counts']['GNB_TX_REFERENCE'], 1)
            self.assertEqual(result['ue_tx_power_requests'], 4)
            self.assertEqual(result['gnb_tx_references'], 1)
            with (root / 'out/ue_tx_power.csv').open(newline='') as source:
                records = list(csv.DictReader(source))
            prach, pusch, pucch, srs = records
            self.assertEqual(prach['frame'], '1')
            self.assertEqual(prach['slot'], '13')
            self.assertEqual(prach['channel'], 'PRACH')
            self.assertEqual(prach['requested_power_dbm'], '0')
            self.assertEqual(prach['generator_reference'], '32767')
            self.assertEqual(prach['resource_first'], '23')
            self.assertEqual(prach['resource_count'], '1')
            self.assertEqual(prach['start_symbol'], '4')
            self.assertEqual(prach['detail_0'], '2')
            self.assertEqual(prach['detail_1'], '9')
            self.assertEqual(pusch['channel'], 'PUSCH')
            self.assertEqual(pusch['requested_power_dbm'], '18')
            self.assertEqual(pusch['resource_first'], '4')
            self.assertEqual(pusch['resource_count'], '64')
            self.assertEqual(pusch['start_symbol'], '2')
            self.assertEqual(pusch['symbol_count'], '12')
            self.assertEqual(pusch['detail_0'], '4')
            self.assertEqual(pusch['detail_1'], '2')
            self.assertEqual(pucch['channel'], 'PUCCH')
            self.assertEqual(pucch['resource_first'], '10')
            self.assertEqual(pucch['detail_0'], '1')
            self.assertEqual(pucch['detail_1'], '1')
            self.assertEqual(srs['channel'], 'SRS')
            self.assertEqual(srs['requested_power_dbm'], '-5')
            self.assertEqual(srs['resource_first'], '20')
            self.assertEqual(srs['resource_count'], '32')
            self.assertEqual(srs['start_symbol'], '10')
            self.assertEqual(srs['symbol_count'], '2')
            self.assertEqual(srs['detail_0'], '7')
            self.assertEqual(srs['detail_1'], '2')

            with (root / 'out/gnb_tx_reference.csv').open(newline='') as source:
                gnb = next(csv.DictReader(source))
            self.assertEqual(gnb['frame'], '2')
            self.assertEqual(gnb['slot'], '0')
            self.assertEqual(gnb['configured_ss_pbch_power_dbm'], '0')
            self.assertEqual(gnb['tx_amp'], '0')
            self.assertEqual(gnb['ssb_index'], '3')
            self.assertEqual(gnb['ssb_start_symbol'], '0')
            self.assertEqual(gnb['antenna_port'], '1')
            self.assertEqual(gnb['beam_index'], '8')
            self.assertEqual(gnb['ssb_start_subcarrier'], '132')

        unavailable = decoder.decode_tx_reference_event(self.event(
            43, 6, a=7, b=1013, c=99, d=decoder.INT64_MIN, e=decoder.INT64_MIN, f=decoder.INT64_MIN))
        self.assertEqual(unavailable['channel'], 'UNKNOWN_99')
        self.assertIsNone(unavailable['requested_power_dbm'])
        self.assertIsNone(unavailable['generator_reference'])
        self.assertIsNone(unavailable['resource_raw'])

    def test_radio_gain_unknown_valid_flags_and_unavailable_values(self):
        result = decoder.decode_radio_gain_event(self.event(
            40, 1, a=2, b=4, c=decoder.INT64_MIN, d=decoder.INT64_MIN, e=5000,
            f=3 | (99 << 8) | ((1 | 0x20) << 16)))
        self.assertEqual(result['status'], 'INVALID')
        self.assertEqual(result['operation'], 'UNKNOWN_99')
        self.assertEqual(result['result_unknown_valid_flags'], 0x20)
        self.assertTrue(result['rx_gain_valid'])
        self.assertFalse(result['tx_gain_valid'])
        self.assertIsNone(result['requested_gain_db'])
        self.assertIsNone(result['rx_readback_db'])
        self.assertIsNone(result['tx_readback_db'])

        timing = decoder.decode_radio_gain_event(self.event(
            41, 2, a=2, b=4, c=0, d=decoder.INT64_MIN, e=0, f=0x84))
        self.assertEqual(timing['unknown_valid_flags'], 0x80)
        self.assertTrue(timing['device_time_valid'])
        self.assertIsNone(timing['begin_device_ticks'])
        self.assertEqual(timing['end_device_ticks'], 0)

        level = decoder.decode_radio_gain_event(self.event(
            42, 3, a=2, b=-4, c=77, d=decoder.INT64_MIN, e=decoder.INT64_MIN,
            f=(5 << 32) | 6))
        self.assertEqual(level['generation'], 4)
        self.assertFalse(level['context_valid'])
        self.assertIsNone(level['mean_power_dbfs'])
        self.assertIsNone(level['peak_component_dbfs'])
        self.assertEqual(level['near_rail_components'], 5)
        self.assertEqual(level['sampled_components'], 6)

    def test_radio_tx_power_evidence_joins_across_sequential_capture_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self.event(49, 10, ring=3, a=4, b=8001, c=(1 << 8), d=-7000, e=-6900, f=1073741824)
            samples = self.event(50, 11, ring=3, a=4, b=8001, c=4, d=400, e=300, f=(20 << 32) | 18)
            quality = self.event(52, 12, ring=3, a=4, b=8001, c=-2, d=50, e=250, f=1024)
            (root / 'oai-flight-recorder-100-0123456789abcdef-0.ndjson').write_text(json.dumps(first) + '\n')
            (root / 'oai-flight-recorder-100-0123456789abcdef-1.ndjson').write_text(
                json.dumps(samples) + '\n' + json.dumps(quality) + '\n')
            result = decoder.decode(root, root / 'out')
            self.assertEqual(result['radio_tx_power_samples_joined'], 1)
            self.assertEqual(result['radio_tx_power_qualities_joined'], 1)
            with (root / 'out/radio_tx_power.csv').open(newline='') as source:
                row = next(csv.DictReader(source))
            self.assertEqual(row['channel'], 'SRS')
            self.assertEqual(row['sample_source_file'], 'oai-flight-recorder-100-0123456789abcdef-1.ndjson')
            self.assertEqual(row['quality_source_file'], 'oai-flight-recorder-100-0123456789abcdef-1.ndjson')

    def test_radio_tx_power_evidence_preserves_repeated_contexts_and_unavailable_values(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = [
                # First PUCCH in frame 4/slot 5: applied, complete evidence.
                self.event(49, 10, ring=1, a=3, b=4005, c=(1 << 8), d=-12500, e=-12400, f=1073741824),
                self.event(50, 11, ring=1, a=3, b=4005, c=16, d=1000, e=900, f=(35 << 32) | 33),
                self.event(52, 12, ring=1, a=3, b=4005, c=-15, d=2345, e=500, f=2048),
                # A second PUCCH shares the same channel and frame/slot but is an observe-only OK result.
                self.event(49, 20, ring=1, a=3, b=4005, c=0, d=-11500, e=-11400, f=536870912),
                self.event(50, 21, ring=1, a=3, b=4005, c=12, d=800, e=700, f=(30 << 32) | 28),
                self.event(52, 22, ring=1, a=3, b=4005, c=-10, d=1234, e=500, f=2048),
                # Non-OK status preserves input evidence but gates output and quantization estimates.
                self.event(49, 30, ring=1, a=2, b=4005, c=2, d=-10000,
                           e=decoder.INT64_MIN, f=decoder.INT64_MIN),
                self.event(50, 31, ring=1, a=2, b=4005, c=8, d=600, e=decoder.INT64_MIN, f=(25 << 32) | 0),
                self.event(52, 32, ring=1, a=2, b=4005, c=decoder.INT64_MIN, d=decoder.INT64_MIN, e=750, f=2048),
                # Dropped sidecars remain missing rather than joining a later same-channel record.
                self.event(49, 40, ring=1, a=2, b=4006, c=0, d=-9000, e=-8900, f=1073741824),
                # Duplicate sidecars for one context make both evidence classes ambiguous.
                self.event(49, 50, ring=1, a=1, b=4007, c=0, d=-8000, e=-7900, f=1073741824),
                self.event(50, 51, ring=1, a=1, b=4007, c=4, d=400, e=300, f=(20 << 32) | 18),
                self.event(50, 52, ring=1, a=1, b=4007, c=4, d=401, e=301, f=(21 << 32) | 19),
                self.event(52, 53, ring=1, a=1, b=4007, c=-1, d=99, e=500, f=2048),
                self.event(52, 54, ring=1, a=1, b=4007, c=-2, d=100, e=500, f=2048),
                # Retain an unknown reject field exactly rather than treating it as a known action.
                self.event(51, 60, ring=1, a=0, b=4005, c=99, d=99, e=2, f=0),
                self.event(51, 61, ring=1, a=0, b=-1001, c=0, d=5, e=1, f=0),
            ]
            (root / 'oai-flight-recorder-100-0123456789abcdef-0.ndjson').write_text(
                ''.join(json.dumps(event) + '\n' for event in events))
            result = decoder.decode(root, root / 'out')

            self.assertEqual(result['event_counts']['RADIO_TX_POWER'], 5)
            self.assertEqual(result['event_counts']['RADIO_TX_POWER_SAMPLES'], 5)
            self.assertEqual(result['event_counts']['RADIO_TX_POWER_QUALITY'], 5)
            self.assertEqual(result['event_counts']['RADIO_TX_REJECT'], 2)
            self.assertEqual(result['radio_tx_powers'], 5)
            self.assertEqual(result['radio_tx_power_samples_joined'], 3)
            self.assertEqual(result['radio_tx_power_qualities_joined'], 3)
            self.assertEqual(result['radio_tx_rejects'], 2)

            with (root / 'out/radio_tx_power.csv').open(newline='') as source:
                rows = list(csv.DictReader(source))
            self.assertEqual([row['power_sequence'] for row in rows], ['10', '20', '30', '40', '50'])
            first_pucch, observe_pucch, unqualified, missing, ambiguous = rows
            self.assertEqual(first_pucch['channel'], 'PUCCH')
            self.assertEqual(first_pucch['frame'], '4')
            self.assertEqual(first_pucch['slot'], '5')
            self.assertEqual(first_pucch['applied'], 'True')
            self.assertEqual(first_pucch['requested_power_dbm'], '-12.5')
            self.assertEqual(first_pucch['estimated_output_power_dbm'], '-12.4')
            self.assertEqual(first_pucch['amplitude_coefficient_q30'], '1073741824')
            self.assertEqual(first_pucch['sample_join_status'], 'joined')
            self.assertEqual(first_pucch['sample_sequence'], '11')
            self.assertEqual(first_pucch['output_energy'], '900')
            self.assertEqual(first_pucch['input_peak_component'], '35')
            self.assertEqual(first_pucch['output_peak_component'], '33')
            self.assertEqual(first_pucch['quality_join_status'], 'joined')
            self.assertEqual(first_pucch['quantization_power_error_raw_millidb'], '-15')
            self.assertEqual(first_pucch['quantization_power_error_db'], '-0.015')
            self.assertEqual(first_pucch['quantization_evm_ppb'], '2345')
            self.assertEqual(first_pucch['profile_uncertainty_db'], '0.5')
            self.assertEqual(first_pucch['component_full_scale'], '2048')

            self.assertEqual(observe_pucch['channel'], 'PUCCH')
            self.assertEqual(observe_pucch['power_sequence'], '20')
            self.assertEqual(observe_pucch['applied'], 'False')
            self.assertEqual(observe_pucch['status'], 'OK')
            self.assertEqual(observe_pucch['sample_sequence'], '21')
            self.assertEqual(observe_pucch['output_energy'], '700')
            self.assertEqual(observe_pucch['quality_sequence'], '22')

            self.assertEqual(unqualified['status'], 'UNQUALIFIED')
            self.assertEqual(unqualified['estimated_output_power_dbm'], '')
            self.assertEqual(unqualified['amplitude_coefficient_q30'], '')
            self.assertEqual(unqualified['sample_join_status'], 'joined')
            self.assertEqual(unqualified['input_energy'], '600')
            self.assertEqual(unqualified['output_energy'], '')
            self.assertEqual(unqualified['quality_join_status'], 'joined')
            self.assertEqual(unqualified['quantization_power_error_raw_millidb'], str(decoder.INT64_MIN))
            self.assertEqual(unqualified['quantization_power_error_db'], '')
            self.assertEqual(unqualified['quantization_evm_ppb'], '')
            self.assertEqual(unqualified['profile_uncertainty_db'], '0.75')

            self.assertEqual(missing['sample_join_status'], 'missing')
            self.assertEqual(missing['sample_sequence'], '')
            self.assertEqual(missing['quality_join_status'], 'missing')
            self.assertEqual(missing['quantization_power_error_db'], '')
            self.assertEqual(ambiguous['sample_join_status'], 'ambiguous')
            self.assertEqual(ambiguous['sample_sequence'], '')
            self.assertEqual(ambiguous['quality_join_status'], 'ambiguous')
            self.assertEqual(ambiguous['quality_sequence'], '')
            self.assertEqual(ambiguous['input_energy'], '')
            self.assertEqual(ambiguous['quantization_power_error_db'], '')

            with (root / 'out/radio_tx_rejects.csv').open(newline='') as source:
                reject, generic_reject = list(csv.DictReader(source))
            self.assertEqual(reject['frame_slot_raw'], '4005')
            self.assertEqual(reject['frame'], '4')
            self.assertEqual(reject['slot'], '5')
            self.assertEqual(reject['channel'], 'UNKNOWN_99')
            self.assertEqual(reject['reason'], 'UNKNOWN_99')
            self.assertEqual(reject['actuation_requested_raw'], '2')
            self.assertEqual(reject['actuation_requested'], '')
            self.assertEqual(generic_reject['frame_slot_raw'], '-1001')
            self.assertEqual(generic_reject['frame'], '')
            self.assertEqual(generic_reject['slot'], '')
            self.assertEqual(generic_reject['channel'], 'GENERIC')
            self.assertEqual(generic_reject['reason'], 'PROFILE')
            self.assertEqual(generic_reject['actuation_requested'], 'True')

        unsupported = decoder.decode_radio_tx_power_event(self.event(51, 69, a=0, b=4005, c=3, d=6, e=1))
        self.assertEqual(unsupported['reason'], 'POWER_CONTROL')
        self.assertEqual(unsupported['channel'], 'PUCCH')

        unknown = decoder.decode_radio_tx_power_event(self.event(49, 70, a=2, b=4005, c=99 | (1 << 12)))
        self.assertEqual(unknown['status'], 'UNKNOWN_99')
        self.assertEqual(unknown['status_unknown_flag_bits'], 1 << 12)

    def test_radio_tx_power_sidecars_require_unbroken_emission_group(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = [
                # ID 49 sequence 100 lost its own ID 50/52 sidecars (101/102).
                self.event(49, 100, ring=7, a=3, b=4005, c=0, d=-9000, e=-8900, f=1073741824),
                # A later same-context ID 49 at 1099 was dropped; its 50/52
                # sidecars must not attach to the surviving sequence-100 header.
                self.event(50, 1100, ring=7, a=3, b=4005, c=8, d=800, e=700, f=(30 << 32) | 28),
                self.event(52, 1101, ring=7, a=3, b=4005, c=-5, d=100, e=500, f=2048),
                # A separate complete triplet retains valid decoder behavior.
                self.event(49, 1200, ring=7, a=4, b=4006, c=0, d=-8000, e=-7900, f=1073741824),
                self.event(50, 1201, ring=7, a=4, b=4006, c=4, d=400, e=300, f=(20 << 32) | 18),
                self.event(52, 1202, ring=7, a=4, b=4006, c=-2, d=50, e=250, f=2048),
            ]
            (root / 'oai-flight-recorder-100-0123456789abcdef-0.ndjson').write_text(
                ''.join(json.dumps(event) + '\n' for event in events))
            result = decoder.decode(root, root / 'out')

            self.assertEqual(result['radio_tx_power_samples_joined'], 1)
            self.assertEqual(result['radio_tx_power_qualities_joined'], 1)
            with (root / 'out/radio_tx_power.csv').open(newline='') as source:
                stale, complete = list(csv.DictReader(source))
            self.assertEqual(stale['power_sequence'], '100')
            self.assertEqual(stale['sample_join_status'], 'ambiguous')
            self.assertEqual(stale['quality_join_status'], 'ambiguous')
            self.assertEqual(stale['sample_sequence'], '')
            self.assertEqual(stale['quality_sequence'], '')
            self.assertEqual(stale['input_energy'], '')
            self.assertEqual(stale['quantization_power_error_db'], '')
            self.assertEqual(complete['power_sequence'], '1200')
            self.assertEqual(complete['sample_join_status'], 'joined')
            self.assertEqual(complete['sample_sequence'], '1201')
            self.assertEqual(complete['quality_join_status'], 'joined')
            self.assertEqual(complete['quality_sequence'], '1202')

    def test_tx_group_allows_only_proven_cross_ring_interleaving(self):
        for fault in (None, 'missing', 'same_ring', 'duplicate', 'duplicate_header', 'backward_time'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                header = self.event(49, 100, ring=4, a=3, b=325001, c=256, d=-29000, e=-29001, f=2052001619)
                events = [header,
                          self.event(50, 102, ring=4, a=3, b=325001, c=548, d=3395352, e=12398999, f=1),
                          self.event(52, 104, ring=4, a=3, b=325001, c=-1, d=2697911, e=3000, f=2048),
                          self.event(27, 101, ring=4 if fault == 'same_ring' else 8),
                          self.event(17, 103, ring=8)]
                if fault == 'backward_time':
                    events[2]['mono_ns'] = 101
                elif fault == 'missing':
                    del events[3]
                elif fault == 'duplicate':
                    events.append(self.event(27, 101, ring=8))
                elif fault == 'duplicate_header':
                    events.append(dict(header))
                (root / 'oai-flight-recorder-100-0123456789abcdef-0.ndjson').write_text(
                    ''.join(json.dumps(event) + '\n' for event in events))
                result = decoder.decode(root, root / 'out')
                expected = 1 if fault is None else 0
                self.assertEqual(result['radio_tx_power_samples_joined'], expected)
                self.assertEqual(result['radio_tx_power_qualities_joined'], expected)
                with (root / 'out/radio_tx_power.csv').open(newline='') as source:
                    rows = list(csv.DictReader(source))
                if fault is None:
                    self.assertEqual(rows[0]['evidence_interleaved_events'], '2')
                    self.assertEqual(rows[0]['evidence_join_basis'], 'complete_cross_ring_coverage')
                else:
                    self.assertTrue(all(row['sample_join_status'] == 'ambiguous' for row in rows))

    def test_tx_interleaving_proof_has_bounded_span(self):
        power = dict(power_sequence=100, recorder_mono_ns=100, thread_ring=4)
        sample = dict(sample_sequence=354, sample_recorder_mono_ns=354)
        quality = dict(quality_sequence=355, quality_recorder_mono_ns=355)
        owners = dict.fromkeys(range(100, 357), 8)
        owners.update({100: 4, 354: 4, 355: 4})
        self.assertTrue(decoder._proven_tx_group(power, sample, quality, owners))
        owners[355], owners[356] = 8, 4
        quality.update(quality_sequence=356, quality_recorder_mono_ns=356)
        self.assertFalse(decoder._proven_tx_group(power, sample, quality, owners))

    def test_complete_tx_buffer_levels_and_missing_or_ambiguous_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = [
                self.event(47, 1, a=0, b=100, c=2, d=2048, e=16777218, f=(2049 << 32) | 2),
                self.event(48, 2, a=0, b=100, c=2, d=1, e=60750, f=7),
                self.event(47, 3, a=0, b=200, c=1, d=2048, e=0, f=0),
                self.event(47, 4, a=0, b=300, c=1, d=2048, e=0, f=0),
                self.event(48, 5, a=0, b=300, c=1, d=-5, e=decoder.INT64_MIN, f=decoder.INT64_MIN),
                self.event(48, 6, a=0, b=300, c=1, d=1, e=60750, f=7),
                self.event(47, 7, a=0, b=400, c=0, d=0, e=-1, f=0),
            ]
            (root / 'oai-flight-recorder-100-0123456789abcdef-0.ndjson').write_text(
                ''.join(json.dumps(event) + '\n' for event in events))
            result = decoder.decode(root, root / 'out')
            self.assertEqual(result['radio_tx_levels'], 4)
            self.assertEqual(result['radio_tx_level_states_joined'], 1)
            with (root / 'out/radio_tx_level.csv').open(newline='') as src:
                rows = list(csv.DictReader(src))
            self.assertEqual(rows[0]['over_range_components'], '2')
            self.assertEqual(rows[0]['backend_return_samples'], '1')
            self.assertEqual(rows[0]['tx_readback_db'], '60.75')
            self.assertEqual(rows[0]['settings_generation'], '7')
            self.assertGreater(float(rows[0]['mean_power_fs']), 2)
            self.assertEqual(rows[1]['mean_power_fs'], '0.0')
            self.assertEqual(rows[1]['mean_power_dbfs'], '')
            self.assertEqual(rows[1]['state_joined'], 'False')
            self.assertEqual(rows[2]['backend_return_samples'], '')
            self.assertEqual(rows[3]['level_valid'], 'False')

    def test_false_footer_and_integer_precision(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            event = dict(schema='oai.flight_recorder', version=1, kind='event', event=10,
                         ring=0, sequence=1, mono_ns=42, realtime_ns=9223372036854775806,
                         a=0, b=1, c=2, d=3, e=4, f=5)
            footer = dict(schema='oai.flight_recorder', version=1, kind='capture_footer', clean=1)
            path = root / 'oai-flight-recorder-100-0123456789abcdef-0.ndjson'
            path.write_text(json.dumps(event) + '\n' + json.dumps(footer) + '\n' + 'x' * 5000 + '\n' + '{}')
            result = decoder.decode(root, root / 'out')
            self.assertEqual(result['events'], 1)
            self.assertEqual(result['invalid_lines'], 3)
            self.assertFalse(result['clean_footer_observed'])
            self.assertIn('9223372036854775806', (root / 'out/events.csv').read_text())
            footer.update(writer_errors=0, dropped_ring_full=0, dropped_no_slot=0, no_slot_threads=0,
                          invalid_timestamp_records=0, payload_truncated=0, rings_assigned=1)
            path.write_text(json.dumps(footer) + '\n')
            self.assertTrue(decoder.decode(root, root / 'out2')['clean_footer_observed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
