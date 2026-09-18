#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('decoder', Path(__file__).resolve().parents[1] / 'decode_events.py')
decoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decoder)


class DecoderTests(unittest.TestCase):
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
