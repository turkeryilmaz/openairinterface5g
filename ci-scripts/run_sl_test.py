#!/usr/bin/env python3
#
# Automated tests for running 5G NR Sidelink SyncRef UE and/or Nearby UE.
# The following is an example to run remote machine (nearby) three times.
# The syncref UE is launched on the current machine, and a single
# nearby UE is launched on the machine specified by the --host and
# --user flags. The -r will enable this simulation to be repeated
# three times.
#
# python3 run_sl_test.py --user account --host 10.1.1.68 -r 3
#
# The following is an example to run just a Sidelink Nearby UE.
# By specifying -l nearby, only the nearby UE will be launched
# on the machine specified by the --host and --user flags.
# The -r will enable this simulation to be repeated two times.
#
# python3 run_sl_test.py -l nearby --user account --host 10.1.1.68 -r 2
#
# See `--help` for more information.
#

import os
import sys
import argparse
import logging
import time
import re
import glob
import subprocess
from subprocess import Popen
import threading
from typing import Dict
from queue import Queue
from sl_check_log import LogChecker

HOME_DIR = os.path.expanduser( '~' )

# ----------------------------------------------------------------------------
# Command line argument parsing

parser = argparse.ArgumentParser(description="""
Automated tests for 5G NR Sidelink simulations
""")

parser.add_argument('--launch', '-l', default='both', choices='syncref nearby both'.split(), help="""
Sidelink UE type to launch test scenario (default: %(default)s)
""")

parser.add_argument('--host', default='10.1.1.61', type=str, help="""
Nearby Host IP (default: %(default)s)
""")

parser.add_argument('--user', '-u', default=os.environ.get('USER'), type=str, help="""
User id in Nearby Host (default: %(default)s)
""")

parser.add_argument('--att-host', default='10.1.1.78', type=str, help="""
Host IP for adjusting attenuation (default: %(default)s)
""")

parser.add_argument('--att-user', default='zaid', type=str, help="""
User id for adjusting attenuation (default: %(default)s)
""")

parser.add_argument('--repeat', '-r', default=1, type=int, help="""
The number of repeated test iterations (default: %(default)s)
""")

parser.add_argument('--basic', '-b', action='store_true', help="""
Basic test with basic shell commands
""")

parser.add_argument('--message', '-m', type=str, default='EpiScience', help="""
The message to send from SyncRef UE to Nearby UE
""")

parser.add_argument('--commands', default='sl_cmds.txt', help="""
The USRP Commands .txt file (default: %(default)s)
""")

parser.add_argument('--duration', '-d', metavar='SECONDS', type=int, default=20, help="""
How long to run the test before stopping to examine the logs
""")

parser.add_argument('--att', type=int, default=-1, help="""
Attenuation value
""")

parser.add_argument('--nid1', type=int, default=10, help="""
Nid1 value
""")

parser.add_argument('--nid2', type=int, default=1, help="""
Nid2 value
""")

parser.add_argument('--log-dir', default=HOME_DIR, help="""
Where to store log files
""")

parser.add_argument('--compress', action='store_true', help="""
Compress the log files in the --log-dir
""")

parser.add_argument('--no-run', '-n', action='store_true', help="""
Don't run the test, only examine the logs in the --log-dir
directory from a previous run of the test
""")

parser.add_argument('--debug', action='store_true', help="""
Enable debug logging (for this script only)
""")

parser.add_argument('--save', default=None, help="""
The default Python log result with .txt extention (default: %(default)s)
""")

parser.add_argument('--sci2', action='store_true', help="""
Enable SCI2 log parsing (this will grep the logs for the SCI2 payload)
""")

parser.add_argument('--test', '-t', default='usrp_b210', choices='psbchsim psschsim rfsim usrp_b210 usrp_n310'.split(), help="""
The kind of test scenario to run. The options include psbchsim, psschsim, rfsim, or usrp_b210 usrp_n310. (default: %(default)s)
""")

OPTS = parser.parse_args()
del parser

logging.basicConfig(level=logging.DEBUG if OPTS.debug else logging.INFO,
                    format='>>> %(name)s: %(levelname)s: %(message)s')
LOGGER = logging.getLogger(os.path.basename(sys.argv[0]))

if OPTS.save:
    out_fh = logging.FileHandler(filename=OPTS.save, mode='a')
    LOGGER.addHandler(out_fh)
# ----------------------------------------------------------------------------
def redirect_output(cmd: str, filename: str) -> str:
    cmd += f' >{filename} 2>&1'
    return cmd

def thread_delay(delay: int) -> None:
    count = 0
    while count < 1:
        time.sleep(delay)
        count += 1

class Command:
    """
    Parsing USRP commands file
    """
    def __init__(self, filename) -> None:
        self.check_user()
        self.filename = self.check_file(filename)
        self.parse_commands()

    def check_user(self) -> None:
        if OPTS.test != 'usrp':
            return
        if OPTS.launch != 'syncref' and OPTS.user == '':
            LOGGER.error("--user followed by [user id] is mandatory to connect to remote machine")
            sys.exit(1)

    def check_file(self, filename) -> str:
        data_file = glob.glob(filename)
        if data_file:
            return filename
        LOGGER.error('The file %s does not exist!', filename)
        sys.exit(1)

    def parse_commands(self) -> None:
        """
        Scan the provided commands file.
        """
        self.launch_cmds: Dict[str, str] = {}
        if OPTS.test.lower() == 'usrp_b210':
            launch_cmds_re = re.compile(r'^\s*(\S*)usrp_b210\S*\s*=\s*((\S+\s*)*)')
        elif OPTS.test.lower() == 'usrp_n310':
            launch_cmds_re = re.compile(r'^\s*(\S*)usrp_n310\S*\s*=\s*((\S+\s*)*)')
        elif OPTS.test.lower() == 'rfsim':
            launch_cmds_re = re.compile(r'^\s*(\S*)rfsim\S*\s*=\s*((\S+\s*)*)')
        elif OPTS.test.lower() == 'psbchsim':
            launch_cmds_re = re.compile(r'^\s*(\S*)psbchsim\S*\s*=\s*((\S+\s*)*)')
        elif OPTS.test.lower() == 'psschsim':
            launch_cmds_re = re.compile(r'^\s*(\S*)psschsim\S*\s*=\s*((\S+\s*)*)')
        else:
            LOGGER.error("Provided test option not valid! %s", OPTS.test)
            sys.exit(1)

        with open(self.filename, 'rt') as in_fh:
            nearby_cmd_continued = False
            syncref_cmd_continued = False
            for line in in_fh:
                if line == '\n':
                    continue
                match = launch_cmds_re.match(line)
                if match:
                    host_role = match.group(1)
                    launch_cmds = match.group(2)
                    if host_role.lower().startswith('nearby'):
                        nearby_cmd_continued = True
                        continue
                    if host_role.lower().startswith('syncref'):
                        syncref_cmd_continued = True
                        continue
                elif nearby_cmd_continued:
                    launch_cmds += line
                    if not line.strip().endswith('\\'):
                        self.launch_cmds['nearby'] = launch_cmds
                        LOGGER.debug('Nearby cmd is %s', launch_cmds)
                        nearby_cmd_continued = False
                    continue
                elif syncref_cmd_continued:
                    launch_cmds += line
                    if not line.strip().endswith('\\'):
                        self.launch_cmds['syncref'] = launch_cmds
                        LOGGER.debug('Syncref cmd is %s', launch_cmds)
                        syncref_cmd_continued = False
                    continue
                else:
                    LOGGER.debug('Unmatched line %r', line)
                    continue
        if not self.launch_cmds:
            LOGGER.error('usrp commands are not found in file: %s', self.filename)
            sys.exit(1)

class TestThread(threading.Thread):
    """
    Represents TestThread
    """
    def __init__(self, queue, commands, passed, log_agent):
        threading.Thread.__init__(self)
        self.queue = queue
        self.commands = commands
        self.passed = passed
        self.delay = 0
        self.log_file = log_agent.txlog_file_path
        self.log_agent = log_agent

    def run(self):
        if self.queue.empty():
            LOGGER.error("Queue is empty!")
            sys.exit(1)
        try:
            while not self.queue.empty():
                job = self.queue.get()
                if "nearby" == job:
                    thread_delay(delay = 3)
                    self.launch_nearby(job)
                if "syncref" == job and not OPTS.no_run:
                    thread_delay(delay = self.delay)
                    self.launch_syncref(job)
            self.queue.task_done()
        except Exception as inst:
            LOGGER.info(f"Failed to operate on job with type {type(inst)} and args {inst.args}")

    def launch_syncref(self, job) -> Popen:
        LOGGER.info('Launching SyncRef UE')
        if OPTS.basic: cmd = redirect_output('uname -a', self.log_file)
        else: cmd = self.commands.launch_cmds[job]
        cmd = cmd[:-1] + f' --message {OPTS.message}'
        proc = Popen(cmd, shell=True)
        LOGGER.info(f"syncref_proc = {proc}")
        if not OPTS.basic and not OPTS.no_run:
            LOGGER.info("Process running... %s", job)
            time.sleep(OPTS.duration)
            self.kill_process("syncref", proc)

    def launch_nearby(self, job, host=OPTS.host, user=OPTS.user) -> Popen:
        LOGGER.info('#' * 42)
        LOGGER.info('Launching Nearby UE')
        if OPTS.basic: cmd = redirect_output('uname -a', self.log_file)
        else: cmd = self.commands.launch_cmds[job]
        if 'usrp' in OPTS.test:
            cmd = cmd[:-1] + f' -d {OPTS.duration} --nid1 {OPTS.nid1} --nid2 {OPTS.nid2}'
            proc = Popen(["ssh", f"{user}@{host}", cmd],
                        shell=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
            LOGGER.info(f"nearby_proc = {proc}")
            remote_output = proc.stdout.readlines()
            if remote_output == []:
                nearby_result = proc.stderr.readlines()
            else:
                nearby_result = remote_output
            self.kill_process("nearby", proc)
            if nearby_result:
                self.find_nearby_result_metric(nearby_result)
        else:
            proc = Popen(cmd, shell=True)
            LOGGER.info(f"nearby_proc = {proc}")
            if not OPTS.basic and not OPTS.no_run:
                LOGGER.info(f"Process running... {job}")
                time.sleep(OPTS.duration)
                self.kill_process("nearby", proc)
            nearby_result = self.log_agent.analyze_nearby_logs(OPTS.nid1, OPTS.nid2, OPTS.sci2)
            if nearby_result:
                self.find_nearby_result_metric(nearby_result)

    def find_nearby_result_metric(self, remote_log):
        result_metric = None
        for line in remote_log:
            if type(line) is not str:
                line = line.decode()
            if OPTS.test == 'usrp':
                LOGGER.info(line.strip())
            # 'SyncRef UE found. PSSCH-RSRP: -102 dBm/RE SSS-RSRP: -100 dBm/RE passed 99 total 100 It took {delta_time_s} seconds'
            if 'SyncRef UE found' in line:
                fields = line.split(maxsplit=20)
                if len(fields) > 6:
                    pssch_rsrp = float(fields[-13])
                    ssb_rsrp = float(fields[-10])
                    nb_decoded = int(fields[-7])
                    total_rx = int(fields[-5])
                    sync_duration = float(fields[-2])
                    counting_duration = sync_duration - self.delay
                    result_metric = (pssch_rsrp, ssb_rsrp, nb_decoded, total_rx, sync_duration, counting_duration)
                    self.passed += [result_metric]
                    return

    def kill_process(self, job: str, proc: Popen) -> None:
        # Wait for the processes to end
        LOGGER.info(f'kill main simulation processes... {job}')
        cmd = ['sudo', 'killall']
        cmd.append('-KILL')
        if proc:
            cmd.append('nr-uesoftmodem')
            if "syncref" == job:
                subprocess.run(cmd)
            LOGGER.info(f'Waiting for PID proc.pid for {job}')
            proc.kill()
            proc.wait()
        LOGGER.info(f'kill main simulation processes...done for {job}')

# ----------------------------------------------------------------------------

def set_attenuation(attenuation, host, user) -> Popen:
    if OPTS.att >= 0:
        LOGGER.info('Setting attenuation')
        cmd = f'curl http://169.254.10.10/:CHAN:3:SETATT:{attenuation}'
        Popen(["ssh", f"{user}@{host}", cmd],
              shell=False,
              stdout=subprocess.PIPE,
              stderr=subprocess.PIPE)
        LOGGER.info(f"attenuation value = {attenuation}")
        time.sleep(1)


def main() -> int:
    """
    Main function to run sidelink test repeatedly for a given attenuation value.
    """
    commands = Command(OPTS.commands)
    log_agent = LogChecker(OPTS, LOGGER)
    LOGGER.debug(f'Number of iterations {OPTS.repeat}')
    if commands.launch_cmds is not None:
        for role, cmd in commands.launch_cmds.items():
            LOGGER.debug(f'{role} UE: {cmd}')
    jobs = ['nearby', 'syncref'] if OPTS.launch == 'both' else [OPTS.launch]
    passed_metric = []
    num_tx_ssb = []
    num_passed = 0
    total_rx_list = []
    nb_decoded_list = []
    pssch_rsrp_list = []
    ssb_rsrp_list = []
    sync_duration_list = []
    set_attenuation(OPTS.att, OPTS.att_host, OPTS.att_user)
    for i in range(OPTS.repeat):
        threads = []
        queue = Queue()
        for job in jobs:
            queue.put(job)
            th = TestThread(queue, commands, passed_metric, log_agent)
            th.setDaemon(True)
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        if 'nearby' in jobs:
            if num_passed != len(passed_metric):
                # Examine the logs to determine if the test passed
                (pssch_rsrp, ssb_rsrp, nb_decoded, total_rx, sync_duration, counting_duration) = passed_metric[-1]
                num_ssb = log_agent.analyze_syncref_logs(counting_duration)
                num_tx_ssb += [num_ssb]
                total_rx_list += [total_rx]
                sync_duration_list += [sync_duration]
                nb_decoded_list += [nb_decoded]
                pssch_rsrp_list += [pssch_rsrp]
                ssb_rsrp_list += [ssb_rsrp]
                LOGGER.info(f"Trial {i+1}/{OPTS.repeat} PASSED. {num_ssb} SSB(s) were generated. Measured {ssb_rsrp} RSRP (dbm/RE)")
            else:
                LOGGER.info(f"Failure detected during {i+1}/{OPTS.repeat} trial(s).")
            num_passed = len(passed_metric)

    LOGGER.info('#' * 42)
    if 'nearby' in jobs:
        LOGGER.info(f"Number of passed = {len(passed_metric)}/{OPTS.repeat}")
        if len(num_tx_ssb) > 0:
            LOGGER.info(f"Avg number of SSB = {sum(num_tx_ssb) / len(num_tx_ssb)} ({num_tx_ssb})")
        if len(passed_metric) > 0:
            sum_nb_decoded, sum_total_rx = sum(nb_decoded_list), sum(total_rx_list)
            avg_bler = (float) (sum_total_rx - sum_nb_decoded) / sum_total_rx if sum_total_rx > 0 else 1
            LOGGER.info(f"Avg PSSCH RSRP = {sum(pssch_rsrp_list) / len(passed_metric):.2f}")
            LOGGER.info(f"Avg SSB RSRP = {sum(ssb_rsrp_list) / len(passed_metric):.2f}")
            LOGGER.info(f"Avg BLER = {avg_bler:.2f} with {sum_nb_decoded} / {sum_total_rx}")
            LOGGER.info(f"Avg Sync duration (seconds) = {sum(sync_duration_list) / len(passed_metric):.2f}")
            LOGGER.info(f"pssch_rsrp_list = {pssch_rsrp_list}")
            LOGGER.info(f"ssb_rsrp_list = {ssb_rsrp_list}")
            LOGGER.info(f"nb_decoded_list = {nb_decoded_list}")
            LOGGER.info(f"total_rx_list = {total_rx_list}")
            LOGGER.info('-' * 42)
        return 0

sys.exit(main())
