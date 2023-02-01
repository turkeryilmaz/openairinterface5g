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
from typing import Dict
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

parser.add_argument('--user', '-u',  default='', type=str, help="""
User id in Nearby Host (default: %(default)s)
""")

parser.add_argument('--repeat', '-r', default=1, type=int, help="""
The number of repeated test iterations (default: %(default)s)
""")

parser.add_argument('--basic', '-b', action='store_true', help="""
Basic test with basic shell commands
""")

parser.add_argument('--commands', '-c', default='sl_usrp_cmds.txt', help="""
The USRP Commands .txt file (default: %(default)s)
""")

parser.add_argument('--duration', '-d', metavar='SECONDS', type=int, default=25, help="""
How long to run the test before stopping to examine the logs
""")

parser.add_argument('--log-dir', default=HOME_DIR, help="""
Where to store log files
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
    def __init__(self, job, queue, commands, passed):
        threading.Thread.__init__(self)
        self.name = job
        self.queue = queue
        self.commands = commands
        self.passed = passed

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
                    thread_delay(job, delay = 3)
                    syncref_proc = self.launch_syncref(job)
                    LOGGER.info(f"syncref_proc = {syncref_proc}")
            if not OPTS.basic:
                LOGGER.info(f"Process running... {job}")
                time.sleep(OPTS.duration)
                if nearby_proc:
                    self.kill_process("nearby", nearby_proc)
                if syncref_proc:
                    #time.sleep(OPTS.duration)
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
            error = proc.stderr.readlines()
            for raw_line in error:
                line = raw_line.decode()
                LOGGER.info(line.strip())
                if 'PASSED' in line:
                    self.passed += ['passed']
        else:
            for raw_line in remote_output:
                line = raw_line.decode()
                LOGGER.info(line.strip())
                if 'PASSED' in line:
                    self.passed += ['passed']
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

def main() -> int:
    commands = Command(OPTS.commands)
    LOGGER.debug(f'Number of iterations {OPTS.repeat}')
    if commands.usrp_cmds:
        for role, cmd in commands.usrp_cmds.items():
            LOGGER.debug(f'{role} UE: {cmd}')
    jobs = ['nearby', 'syncref'] if OPTS.launch == 'both' else [OPTS.launch]
    passed = []
    num_passed = 0
    for i in range(OPTS.repeat):
        threads = []
        queue = Queue()
        for job in jobs:
            queue.put(job)
            th = TestThread(job, queue, commands, passed)
            th.setDaemon(True)
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        if num_passed != len(passed):
            LOGGER.info(f"Passed at the trial {i+1}/{OPTS.repeat}")
        else:
            LOGGER.info(f"Failed at the trial {i+1}/{OPTS.repeat}")
        num_passed = len(passed)
    LOGGER.info('#' * 42)
    LOGGER.info(f"Number of passed = {len(passed)}/{OPTS.repeat}")
    return 0

sys.exit(main())
