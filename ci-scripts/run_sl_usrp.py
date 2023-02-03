#!/usr/bin/env python3
#
# Automated tests for running 5G NR Sidelink SyncRef UE and/or Nearby UE.
# The following is an example to run remote machine (nearby) three times.
# The syncref UE is launched on the current machine, and a single
# nearby UE is launched on the machine specified by the --host and
# --user flags. The -r will enable this simulation to be repeated
# three times.
#
# python3 run_sl_usrp.py --user account --host 10.1.1.68 -r 3
#
# The following is an example to run just a Sidelink Nearby UE.
# By specifying -l nearby, only the nearby UE will be launched
# on the machine specified by the --host and --user flags.
# The -r will enable this simulation to be repeated two times.
#
# python3 run_sl_usrp.py -l nearby --user account --host 10.1.1.68 -r 2
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
from typing import Dict, Generator
from queue import *

HOME_DIR = os.path.expanduser( '~' )

# ----------------------------------------------------------------------------
# Command line argument parsing

parser = argparse.ArgumentParser(description="""
Automated tests for 5G NR Sidelink simulations
""")

parser.add_argument('--launch', '-l', default='both', choices='syncref nearby both'.split(), help="""
Sidelink UE type to launch test scenario (default: %(default)s)
""")

parser.add_argument('--host', default='10.1.1.80', type=str, help="""
Nearby Host IP (default: %(default)s)
""")

parser.add_argument('--user', '-u',  default='zaid', type=str, help="""
User id in Nearby Host (default: %(default)s)
""")

parser.add_argument('--repeat', '-r', default=1, type=int, help="""
The number of repeated test iterations (default: %(default)s)
""")

parser.add_argument('--basic', '-b', action='store_true', help="""
Basic test with basic shell commands
""")

parser.add_argument('--commands', default='sl_usrp_cmds.txt', help="""
The USRP Commands .txt file (default: %(default)s)
""")

parser.add_argument('--duration', '-d', metavar='SECONDS', type=int, default=30, help="""
How long to run the test before stopping to examine the logs
""")

parser.add_argument('--log-dir', default=HOME_DIR, help="""
Where to store log files
""")

parser.add_argument('--compress', action='store_true', help="""
Compress the log files in the --log-dir
""")

parser.add_argument('--debug', action='store_true', help="""
Enable debug logging (for this script only)
""")

OPTS = parser.parse_args()
del parser

logging.basicConfig(level=logging.DEBUG if OPTS.debug else logging.INFO,
                    format='>>> %(name)s: %(levelname)s: %(message)s')
LOGGER = logging.getLogger(os.path.basename(sys.argv[0]))

log_file_path = os.path.join(OPTS.log_dir, 'tx.log')

# ----------------------------------------------------------------------------
def redirect_output(cmd: str, filename: str) -> str:
    cmd += ' >{} 2>&1'.format(filename)
    return cmd

def thread_delay(thread_name: str, delay: int) -> None:
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
        if OPTS.launch != 'syncref' and OPTS.user == '':
            LOGGER.error(f'--user followed by user id is mandatory to connect to remote machine')
            sys.exit(1)

    def check_file(self, filename) -> str:
        data_file = glob.glob(filename)
        if data_file:
            return filename
        else:
            LOGGER.error(f'The file {filename} does not exist!')
            sys.exit(1)

    def parse_commands(self) -> None:
        """
        Scan the usrp commands file.
        """
        self.usrp_cmds: Dict[str, str] = {}
        usrp_cmds_re = re.compile(r'^\s*(\S*)usrp\S*\s*=\s*((\S+\s*)*)')

        with open(self.filename, 'rt') as fh:
            nearby_cmd_continued = False
            syncref_cmd_continued = False
            for line in fh:
                if line == '\n': continue
                match = usrp_cmds_re.match(line)
                if match:
                    host_role = match.group(1)
                    usrp_cmds = match.group(2)
                    if host_role.lower().startswith('nearby'):
                        nearby_cmd_continued = True
                        continue
                    if host_role.lower().startswith('syncref'):
                        syncref_cmd_continued = True
                        continue
                elif nearby_cmd_continued:
                    usrp_cmds += line
                    if not line.strip().endswith('\\'):
                        self.usrp_cmds['nearby'] = usrp_cmds
                        LOGGER.debug('Nearby cmd is %s', usrp_cmds)
                        nearby_cmd_continued = False
                    continue
                elif syncref_cmd_continued:
                    usrp_cmds += line
                    if not line.strip().endswith('\\'):
                        self.usrp_cmds['syncref'] = usrp_cmds
                        LOGGER.debug('Syncref cmd is %s', usrp_cmds)
                        syncref_cmd_continued = False
                    continue
                else:
                    LOGGER.debug('Unmatched line %r', line)
                    continue
        if self.usrp_cmds == {}:
            LOGGER.error(f'usrp commands are not found in file: {OPTS.commands} ')
            exit()

class TestThread(threading.Thread):
    """
    Represents TestThread
    """
    def __init__(self, queue, commands, passed):
        threading.Thread.__init__(self)
        self.queue = queue
        self.commands = commands
        self.passed = passed
        self.delay = 3

    def run(self):
        if self.queue.empty() == True:
            LOGGER.error("Queue is empty!")
            sys.exit(1)
        try:
            nearby_proc = None
            syncref_proc = None
            while not self.queue.empty():
                job = self.queue.get()
                if "nearby" == job:
                    thread_delay(job, delay = 0)
                    nearby_proc = self.launch_nearby(job)
                    LOGGER.info(f"nearby_proc = {nearby_proc}")
                if "syncref" == job:
                    thread_delay(job, delay = self.delay)
                    syncref_proc = self.launch_syncref(job)
                    LOGGER.info(f"syncref_proc = {syncref_proc}")
            if not OPTS.basic:
                LOGGER.info(f"Process running... {job}")
                time.sleep(OPTS.duration)
                if nearby_proc:
                    self.kill_process("nearby", nearby_proc)
                if syncref_proc:
                    self.kill_process("syncref", syncref_proc)
            self.queue.task_done()
        except Exception as inst:
            LOGGER.info(f"Failed to operate on job with type {type(inst)} and args {inst.args}")

    def launch_nearby(self, job, host=OPTS.host, user=OPTS.user) -> Popen:
        LOGGER.info('#' * 42)
        LOGGER.info('Launching Nearby UE')
        if OPTS.basic: cmd = redirect_output('uname -a', log_file_path)
        else: cmd = self.commands.usrp_cmds[job][:-1] + f' -d {OPTS.duration}\n'
        proc = Popen(["ssh", f"{user}@{host}", cmd],
                    shell=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
        remote_output = proc.stdout.readlines()
        if remote_output == []:
            remote_log = proc.stderr.readlines()
        else:
            remote_log = remote_output
        result_metric = None
        for raw_line in remote_log:
            line = raw_line.decode()
            LOGGER.info(line.strip())
            # 'SyncRef UE found. RSRP: -100 dBm/RE. It took {delta_time_s} seconds'
            if 'It took' in line and 'seconds' in line:
                fields = line.split(maxsplit=12)
                if len(fields) > 6:
                    ssb_rsrp = float(fields[-6])
                    sync_duration = float(fields[-2])
                    counting_duration = sync_duration - self.delay
                    result_metric = (ssb_rsrp, sync_duration, counting_duration)
            if 'PASSED' in line:
                self.passed += [result_metric]
        return proc

    def launch_syncref(self, job) -> Popen:
        LOGGER.info('Launching SyncRef UE')
        if OPTS.basic: cmd = redirect_output('uname -a', log_file_path)
        else: cmd = self.commands.usrp_cmds[job]
        proc = Popen(cmd, shell=True)
        time.sleep(1)
        return proc

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

def get_lines(filename: str) -> Generator[str, None, None]:
    """
    Yield each line of the given log file (.bz2 compressed log file if -c flag is used.)
    """
    fh = bz2.open(filename, 'rb') if OPTS.compress else open(filename, 'rb')
    for line_bytes in fh:
        line = line_bytes.decode('utf-8', 'backslashreplace')
        line = line.rstrip('\r\n')
        yield line

def get_analysis_messages(filename: str) -> Generator[str, None, None]:
    """
    Finding all logs in the log file with X fields for log parsing optimization
    """
    LOGGER.info('Scanning %s', filename)
    for line in get_lines(filename):
            #796811.532881 [NR_PHY] nrUE configured
            #796821.854505 [NR_PHY] PSBCH SL generation started
            fields = line.split(maxsplit=5)
            if len(fields) == 4 or len(fields) == 6 :
                yield line

def analyze_logs(counting_delta: float) -> int:
    time_start_s, time_end_s = -1, -1
    log_file = log_file_path
    sum_ssb = 0

    if OPTS.compress:
        log_file = f'{log_file_path}.bz2'
    for line in get_analysis_messages(log_file):
        #796811.532881 [NR_PHY] nrUE configured
        #796821.854505 [NR_PHY] PSBCH SL generation started
        if time_start_s == -1 and 'nrUE configured' in line:
            fields = line.split(maxsplit=2)
            time_start_s = float(fields[0])
            time_end_s = time_start_s + counting_delta
        if 'PSBCH SL generation started' in line:
            fields = line.split(maxsplit=2)
            time_st = float(''.join([ch for ch in fields[0] if ch.isnumeric() or ch =='.']))
            if time_st < time_end_s:
                sum_ssb += 1
    return sum_ssb

def main() -> int:
    commands = Command(OPTS.commands)
    LOGGER.debug(f'Number of iterations {OPTS.repeat}')
    if commands.usrp_cmds:
        for role, cmd in commands.usrp_cmds.items():
            LOGGER.debug(f'{role} UE: {cmd}')
    jobs = ['nearby', 'syncref'] if OPTS.launch == 'both' else [OPTS.launch]
    delay = 3
    passed_metric = []
    num_tx_ssb = []
    num_passed = 0
    for i in range(OPTS.repeat):
        threads = []
        queue = Queue()
        for job in jobs:
            queue.put(job)
            th = TestThread(queue, commands, passed_metric)
            th.setDaemon(True)
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        if num_passed != len(passed_metric):
            # Examine the logs to determine if the test passed
            (ssb_rsrp, sync_duration, counting_duration) = passed_metric[-1]
            num_ssb = analyze_logs(counting_duration)
            num_tx_ssb += [num_ssb]
            LOGGER.info(f'number of SSB = {num_ssb}')
            LOGGER.info(f'SSB RSRP = {ssb_rsrp} dBm/RE')
            LOGGER.info(f"Passed at the trial {i+1}/{OPTS.repeat}")
        else:
            LOGGER.info(f"Failed at the trial {i+1}/{OPTS.repeat}")
        num_passed = len(passed_metric)

    LOGGER.info('#' * 42)
    LOGGER.info(f"Number of passed = {len(passed_metric)}/{OPTS.repeat}")
    if len(num_tx_ssb) > 0:
        LOGGER.info(f"Avg number of SSB = {sum(num_tx_ssb) / len(num_tx_ssb)} ({num_tx_ssb})")
    if len(passed_metric) > 0:
        LOGGER.info(f"Avg SSB RSRP = {sum([result[0] for result in passed_metric]) / len(passed_metric)}")
        LOGGER.info(f"Avg Sync duration (seconds) = {sum([result[1] for result in passed_metric]) / len(passed_metric)}")
    return 0

sys.exit(main())
