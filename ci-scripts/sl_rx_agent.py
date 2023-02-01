#!/usr/bin/env python3
#
# Automated tests for running 5G NR Sidelink Rx simulations.
# The following is an example to run this script.
#
# python3 sl_rx_agent.py \
# --cmd 'sudo -E LD_LIBRARY_PATH=$HOME/openairinterface5g/cmake_targets/ran_build/build:$LD_LIBRARY_PATH \
# $HOME/openairinterface5g/cmake_targets/ran_build/build/nr-uesoftmodem \
# --sl-mode 2 -r 106 --numerology 1 --band 38 -C 2600000000  --ue-rxgain 90 \
# --usrp-args "type=n3xx,addr=192.168.20.2,subdev=A:0,master_clock_rate=122.88e6" \
# > ~/rx.log 2>&1' \
# --nid1 10 --nid2 1
#
# See `--help` for more information.
#

import os
import sys
import argparse
import logging
import time
import glob
import bz2
import subprocess
from subprocess import Popen
from typing import Optional, List, Generator

HOME_DIR = os.path.expanduser( '~' )
# ----------------------------------------------------------------------------
# Command line argument parsing

parser = argparse.ArgumentParser(description="""
Automated tests for 5G NR Sidelink Rx simulations
""")

parser.add_argument('--duration', '-d', metavar='SECONDS', type=int, default=25, help="""
How long to run the test before stopping to examine the logs
""")

parser.add_argument('--cmd', type=str, default='', help="""
Commands
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

parser.add_argument('--compress', '-c', action='store_true', help="""
Compress the log files in the --log-dir
""")

parser.add_argument('--no-run', '-n', action='store_true', help="""
Don't run the test, only examine the logs in the --log-dir
directory from a previous run of the test
""")

parser.add_argument('--debug', action='store_true', help="""
Enable debug logging (for this script only)
""")

OPTS = parser.parse_args()
del parser

logging.basicConfig(level=logging.DEBUG if OPTS.debug else logging.INFO,
                    format='>>> %(name)s: %(levelname)s: %(message)s')
LOGGER = logging.getLogger(os.path.basename(sys.argv[0]))

log_file_path = os.path.join(OPTS.log_dir, 'rx.log')

# ----------------------------------------------------------------------------

def compress(from_name: str, to_name: Optional[str]=None, remove_original: bool=False) -> None:
    """
    Compress the file `from_name` and store it as `to_name`.
    `to_name` defaults to `from_name` with `.bz2` appended.
    If `remove_original` is True, removes `from_name` when the compress finishes.
    """
    if to_name is None:
        to_name = from_name
    if not to_name.endswith('.bz2'):
        to_name += '.bz2'
    LOGGER.info('Compress %s to %s', from_name, to_name)
    with bz2.open(to_name, 'w') as outh, \
        open(from_name, 'rb') as inh:
        while True:
            data = inh.read(10240)
            if not data:
                break
            outh.write(data)
    if remove_original:
        LOGGER.debug('Remove %s', from_name)
        os.remove(from_name)


class CompressJobs:
    """
    Allow multiple invocations of `compress` to run in parallel
    """

    def __init__(self) -> None:
        self.kids: List[int] = []

    def compress(self, from_name: str, to_name: Optional[str]=None, remove_original: bool=False) -> None:
        if not os.path.exists(from_name):
            # It's not necessarily an error if the log file does not exist.
            # For example, if nfapi_trace never gets invoked (e.g., because
            # NFAPI_TRACE_LEVEL is set to none), then the log file nfapi.log
            # will not get created.
            LOGGER.warning('No file: %s', from_name)
            return
        kid = os.fork()
        if kid != 0:
            self.kids.append(kid)
        else:
            LOGGER.debug('in pid %d compress %s...', os.getpid(), from_name)
            compress(from_name, to_name, remove_original)
            LOGGER.debug('in pid %d compress %s...done', os.getpid(), from_name)
            sys.exit()

    def wait(self) -> None:
        LOGGER.debug('wait %s...', self.kids)
        failed = []
        for kid in self.kids:
            LOGGER.debug('waitpid %d', kid)
            _pid, status = os.waitpid(kid, 0)
            if status != 0:
                failed.append(kid)
        if failed:
            raise Exception('compression failed: %s', failed)
        LOGGER.debug('wait...done')


class TestNearby():
    """
    Represents TestNearby
    """
    def __init__(self):
        self.cmd = None
        self.delay = 3 # seconds


    def run(self, cmd: str) -> bool:
        self.cmd = cmd
        job = "nearby"
        time.sleep(self.delay)
        proc = self.launch_nearby(job)
        LOGGER.info(f"nearby_proc = {proc}")
        LOGGER.info(f"Process running... {job}")
        time.sleep(OPTS.duration)
        passed = self.kill_process("nearby", proc)
        if OPTS.compress:
            self.compress_log_file(proc)
        return passed

    def launch_nearby(self, job) -> Popen:
        LOGGER.info('Launching nearby UE: %s', log_file_path)
        cmd = self.cmd
        proc = Popen(cmd, shell=True)
        time.sleep(1)
        return proc

    def kill_process(self, job: str, proc: Popen) -> bool:
        passed = True
        if proc:
            status = proc.poll()
            if status is None:
                LOGGER.info('process is still running, which is good')
            else:
                #passed = False
                LOGGER.info('process ended early: %r', status)
        LOGGER.info(f'kill main simulation processes... {job}')
        cmd = ['sudo', 'killall']
        cmd.append('-KILL')
        if proc:
            cmd.append('nr-uesoftmodem')
            if "nearby" == job:
                subprocess.run(cmd)
            LOGGER.info(f'Waiting for PID proc.pid for {job}')
            proc.kill()
            proc.wait()
        LOGGER.info(f'kill main simulation processes...done for {job}')
        return passed

    def compress_log_file(self, proc: Popen):
        jobs = CompressJobs()
        jobs.compress(log_file_path)
        jobs.wait()

# ----------------------------------------------------------------------------

def get_lines(filename: str) -> Generator[str, None, None]:
    """
    Yield each line of the given .bz2 compressed log file
    """
    fh = bz2.open(filename, 'rb') if OPTS.compress else open(filename, 'rb')
    for line_bytes in fh:
        line = line_bytes.decode('utf-8', 'backslashreplace')
        line = line.rstrip('\r\n')
        yield line

def get_analysis_messages(filename: str) -> Generator[str, None, None]:
    """
    Find all the 4 fields or the 6 fields log messages in the given log file `filename`
    and yield them one by one.  The file is a .bz2 compressed log.
    """
    LOGGER.info('Scanning %s', filename)
    for line in get_lines(filename):
            #'[NR_PHY] SyncRefUE found with Nid1 10 and Nid2 1'
            fields = line.split(maxsplit=8)
            if len(fields) == 9:
                yield line

def analyze_logs(test_agent: TestNearby, encNid1: int, encNid2: int) -> bool:
    num_failed = 0
    found = set()
    estNid1, estNid2 = -1, -1
    log_file = log_file_path

    if OPTS.compress:
        log_file = f'{log_file_path}.bz2'
    for line in get_analysis_messages(log_file):
        #'[NR_PHY] SyncRefUE found with Nid1 10 and Nid2 1'
        if 'SyncRefUE found' in line and 'No' not in line:
            if 'Nid1' in line and 'Nid2' in line:
                fields = line.split(maxsplit=8)
                estNid1 = int(fields[5])
                estNid2 = int(fields[8])
                found.add('found')
            continue

    LOGGER.debug('found: %r', found)
    if len(found) != 1 or encNid1 != estNid1 or encNid2 != estNid2:
        LOGGER.error(f'Failed -- found SyncRefUE Nid1 {estNid1}, Ni2 {estNid2}, expecting Nid1 {encNid1}, Nid2 {encNid2}')
        return False

    LOGGER.info('SyncRefUE found')
    return True

def main(argv) -> int:
    test_agent = TestNearby()

    passed = True
    if not OPTS.no_run:
        passed = test_agent.run(OPTS.cmd)

    # Examine the logs to determine if the test passed
    if not analyze_logs(test_agent, encNid1=OPTS.nid1, encNid2=OPTS.nid2):
        passed = False

    if not passed:
        LOGGER.critical('FAILED')
        return 1

    LOGGER.info('PASSED')
    return 0

sys.exit(main(sys.argv))
