#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0

import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class ProfilerSourceContractTest(unittest.TestCase):
    def test_gnb_min_rxtxtime_is_recorded_after_mac_configuration(self) -> None:
        source = (
            REPOSITORY_ROOT / "executables" / "nr-softmodem.c"
        ).read_text(encoding="utf-8")
        recorder = 'oai_profiler_record_setting_int("gnb.min_rxtxtime"'

        self.assertEqual(source.count(recorder), 1)

        function_start = source.index("static int create_gNB_tasks(")
        function_end = source.index("int main(", function_start)
        function = source[function_start:function_end]

        configure_position = function.index("RCconfig_nr_macrlc(cfg);")
        recorder_position = function.index(recorder)
        north_position = function.index("l1_north_init_gNB()")

        self.assertLess(configure_position, recorder_position)
        self.assertLess(recorder_position, north_position)
        self.assertIn("if (RC.nb_nr_macrlc_inst > 0) {", function)
        self.assertIn(
            "if (RC.nrmac != NULL && RC.nrmac[0] != NULL)",
            function[configure_position:recorder_position],
        )


if __name__ == "__main__":
    unittest.main()
