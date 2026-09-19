#!/usr/bin/env python3
"""Exercise the production bootstrap and real OAI config plugin without any radio code."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

BINARY = Path(sys.argv.pop(1)).resolve()
ROOT = Path(__file__).resolve().parents[3] / "cmake_targets/log/FlightTests/Validation/startup"


class Startup(unittest.TestCase):
    def setUp(self):
        ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.output = self.work / "captures"
        (self.work / "relative-input.txt").write_text("fixture\n")
        self.env = dict(os.environ, LD_LIBRARY_PATH=str(BINARY.parent))
        for name in ("_OAI_FLIGHT_CAPTURE_PARENT", "OAI_FLIGHT_RECORDER_DIR", "OAI_FLIGHT_RECORDER_MAX_BYTES"):
            self.env.pop(name, None)

    def config(self, features=None, extra=""):
        text = 'fixture-input = "relative-input.txt";\n'
        if features is not None:
            text += f'flight = "{features}";\n'
        text += f'flight-output = "{self.output}";\n' + extra
        (self.work / "flight.conf").write_text(text)

    def run_fixture(self, *args):
        return subprocess.run([str(BINARY), "-O", "flight.conf", *args], cwd=self.work,
                              env=self.env, capture_output=True, text=True, timeout=15)

    def capture(self, root=None):
        runs = list((root or self.output).glob("ue-*"))
        self.assertEqual(len(runs), 1)
        run = runs[0]
        status = json.loads((run / "status.json").read_text())
        self.assertEqual(status["counters"]["launch_attempts"], 1)
        self.assertEqual(status["exit_code"], 0)
        self.assertTrue(status["capture"]["healthy"], status)
        self.assertIn('"kind":"event"', ''.join(p.read_text() for p in (run / "recorder").glob("*.ndjson")))
        metadata = json.loads((run / "metadata.json").read_text())
        self.assertEqual(metadata["child_working_directory"]["mode"], "original_launch_directory")
        self.assertIsNone(metadata["source"].get("error"))
        return run

    def test_config_only_and_relative_inputs(self):
        self.config("log")
        result = self.run_fixture()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.capture()

    def test_cli_enables_and_overrides_output(self):
        self.config("off")
        output = self.work / "cli with spaces"
        result = self.run_fixture("--flight", "log", "--flight-output", str(output))
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.capture(output)
        self.assertFalse(self.output.exists())

    def test_secret_not_echoed_before_collector(self):
        marker = "synthetic-bootstrap-key-do-not-print"
        for config_enabled in (False, True):
            self.output = self.work / f"secret-{config_enabled}"
            self.config("log" if config_enabled else "off")
            args = [] if config_enabled else ["--flight", "log"]
            result = self.run_fixture(*args, "--uicc0.key", marker)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertNotIn(marker, result.stdout + result.stderr)
            run = self.capture()
            for path in [run / "metadata.json", *run.glob("stdout.*.log"), *run.glob("stderr.*.log")]:
                self.assertNotIn(marker, path.read_text())

    def test_recovery_config_and_cli_start_one_terminal_worker(self):
        for config_enabled in (True, False):
            self.output = self.work / f"recovery-{config_enabled}"
            self.config("log recovery" if config_enabled else "off")
            args = [] if config_enabled else ["--flight", "log", "recovery"]
            result = self.run_fixture(*args)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            sessions = list(self.output.glob("ue-session-*"))
            self.assertEqual(len(sessions), 1)
            status = json.loads((sessions[0] / "status.json").read_text())
            self.assertEqual(status["attempt_count"], 1)
            self.assertEqual(status["state"], "policy_stop")
            self.assertIsNone(status["operator_stop_signal"])
            self.assertEqual(status["exit_code"], 0)
            # This fixture performs the production bootstrap/logInit then
            # exits normally. A terminal exit must not become a retry loop.
            logs = list((sessions[0] / "attempts").glob("*/stdout.*.log"))
            self.assertIn("fixture-recording=1", ''.join(f.read_text() for f in logs))

    def test_cli_disables_config(self):
        self.config("log")
        result = self.run_fixture("--flight", "off")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("fixture-recording=0", result.stdout)
        self.assertFalse(self.output.exists())

    def test_ordinary_command_remains_disabled(self):
        self.config()
        result = self.run_fixture()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("fixture-recording=0", result.stdout)
        self.assertFalse(self.output.exists())

    def test_future_features_rejected_before_capture(self):
        for features in (("recovery",), ("log", "recovery", "agc"), ("log", "off"), ()):
            self.config()
            result = self.run_fixture("--flight", *features)
            self.assertEqual(result.returncode, 2, result.stderr + result.stdout)
            self.assertFalse(self.output.exists())

    def test_invalid_capture_setting_stops_before_worker(self):
        self.config("log", 'flight-recorder-budget = "invalid";\n')
        result = self.run_fixture()
        self.assertEqual(result.returncode, 2, result.stderr + result.stdout)
        self.assertFalse(self.output.exists())

    def test_default_date_and_unique_runs(self):
        # A relocated checkout path avoids depending on /opt, /etc, or the user's HOME under sudo.
        repo = self.work / "repo"
        (repo / "tools").mkdir(parents=True)
        (repo / "tools/flight_test").symlink_to(Path(__file__).resolve().parents[1], target_is_directory=True)
        (self.work / "flight.conf").write_text(f'flight = "log"; flight-repo = "{repo}";\n')
        for _ in range(2):
            result = self.run_fixture()
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        date = time.strftime("%Y-%m-%d")
        runs = list((repo / "cmake_targets/log/FlightTests" / date).glob("ue-*"))
        self.assertEqual(len(runs), 2)
        self.assertNotEqual(runs[0], runs[1])

    def test_int_and_term_stop_without_relaunch(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.output = self.work / f"signal-{sig}"
            self.config("log", 'fixture-hold = 1;\n')
            p = subprocess.Popen([str(BINARY), "-O", "flight.conf"], cwd=self.work, env=self.env,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    logs = list(self.output.glob("ue-*/stdout.*.log"))
                    if logs and "fixture-recording=1" in ''.join(f.read_text() for f in logs):
                        break
                    if p.poll() is not None:
                        self.fail(str(p.communicate()))
                    time.sleep(0.02)
                else:
                    self.fail("worker did not start")
                p.send_signal(sig)
                stdout, stderr = p.communicate(timeout=10)
                self.assertEqual(p.returncode, 0, stderr + stdout)
                self.capture()
            finally:
                if p.poll() is None:
                    p.kill()
                    p.communicate()


if __name__ == "__main__":
    unittest.main()
