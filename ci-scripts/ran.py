#/*
# * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
# * contributor license agreements.  See the NOTICE file distributed with
# * this work for additional information regarding copyright ownership.
# * The OpenAirInterface Software Alliance licenses this file to You under
# * the OAI Public License, Version 1.1  (the "License"); you may not use this file
# * except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.openairinterface.org/?page_id=698
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *-------------------------------------------------------------------------------
# * For more information about the OpenAirInterface (OAI) Software Alliance:
# *      contact@openairinterface.org
# */
#---------------------------------------------------------------------
# Python for CI of OAI-eNB + COTS-UE
#
#   Required Python Version
#     Python 3.x
#
#   Required Python Package
#     pexpect
#---------------------------------------------------------------------

#-----------------------------------------------------------
# Import
#-----------------------------------------------------------
import sys		 # arg
import re		 # reg
import logging
import os
import time
from multiprocessing import Process, Lock, SimpleQueue
import yaml
import cls_cmd
from cls_containerize import CreateWorkspace

#-----------------------------------------------------------
# OAI Testing modules
#-----------------------------------------------------------
import helpreadme as HELP
import constants as CONST

#-----------------------------------------------------------
# Class Declaration
#-----------------------------------------------------------
class RANManagement():

	def __init__(self):
		
		self.prematureExit = False
		self.ranRepository = ''
		self.ranBranch = ''
		self.ranAllowMerge = False
		self.ranCommitID = ''
		self.ranTargetBranch = ''
		self.eNBIPAddress = ''
		self.eNBSourceCodePath = ''
		self.eNB1IPAddress = ''
		self.eNB1SourceCodePath = ''
		self.eNB2IPAddress = ''
		self.eNB2SourceCodePath = ''
		self.Build_eNB_args = ''
		self.backgroundBuild = False
		self.backgroundBuildTestId = ['', '', '']
		self.Build_eNB_forced_workspace_cleanup = False
		self.Initialize_eNB_args = ''
		self.imageKind = ''
		self.air_interface = ['', '', ''] #changed from 'lte' to '' may lead to side effects in main
		self.eNB_instance = 0
		self.eNB_serverId = ['', '', '']
		self.eNBLogFiles = ['', '', '']
		self.eNBOptions = ['', '', '']
		self.eNBmbmsEnables = [False, False, False]
		self.eNBstatuses = [-1, -1, -1]
		self.testCase_id = ''
		self.epcPcapFile = ''
		self.runtime_stats= ''
		self.datalog_rt_stats={}
		self.datalog_rt_stats_file='datalog_rt_stats.default.yaml'
		self.eNB_Trace = '' #if 'yes', Tshark will be launched at initialization
		self.eNB_Stats = '' #if 'yes', Statistics Monitor will be launched at initialization		
		self.USRPIPAddress = ''
		#checkers from xml
		self.ran_checkers={}
		self.cmd_prefix = '' # prefix before {lte,nr}-softmodem
		self.node = ''
		self.command = ''
		self.command_fail = False


#-----------------------------------------------------------
# RAN management functions
#-----------------------------------------------------------

	def BuildeNB(self, HTML):
		if self.ranRepository == '' or self.ranBranch == '' or self.ranCommitID == '':
			HELP.GenericHelp(CONST.Version)
			sys.exit('Insufficient Parameter')
		if self.eNB_serverId[self.eNB_instance] == '0':
			lIpAddr = self.eNBIPAddress
			lSourcePath = self.eNBSourceCodePath
		elif self.eNB_serverId[self.eNB_instance] == '1':
			lIpAddr = self.eNB1IPAddress
			lSourcePath = self.eNB1SourceCodePath
		elif self.eNB_serverId[self.eNB_instance] == '2':
			lIpAddr = self.eNB2IPAddress
			lSourcePath = self.eNB2SourceCodePath
		if lIpAddr == '' or lSourcePath == '':
			HELP.GenericHelp(CONST.Version)
			sys.exit('Insufficient Parameter')
		logging.debug('Building on server: ' + lIpAddr)
		cmd = cls_cmd.getConnection(lIpAddr)
		# Check if we build an 5G-NR gNB or an LTE eNB
		result = re.search('--RU', self.Build_eNB_args)
		if result is not None:
			self.air_interface[self.eNB_instance] = 'oairu'
		else:
			result = re.search('--gNB', self.Build_eNB_args)
			if result is not None:
				self.air_interface[self.eNB_instance] = 'nr-softmodem'
			else:
				self.air_interface[self.eNB_instance] = 'lte-softmodem'
		
		# Worakround for some servers, we need to erase completely the workspace
		if self.Build_eNB_forced_workspace_cleanup:
			cmd.run(f'sudo rm -Rf {lSourcePath}')
		self.testCase_id = HTML.testCase_id
		# on RedHat/CentOS .git extension is mandatory
		result = re.search('([a-zA-Z0-9\:\-\.\/])+\.git', self.ranRepository)
		if result is not None:
			full_ran_repo_name = self.ranRepository.replace('git/', 'git')
		else:
			full_ran_repo_name = self.ranRepository + '.git'
		CreateWorkspace(cmd, lSourcePath, full_ran_repo_name, self.ranCommitID, self.ranTargetBranch, self.ranAllowMerge)
		cmd.run('source oaienv')
		cmd.cd('cmake_targets')
		cmd.run('mkdir -p log')
		cmd.run('chmod 777 log')
		# no need to remove in log (git clean did the trick)
		if self.backgroundBuild:
			cmd.run(f'echo "./build_oai {self.Build_eNB_args}" > ./my-lte-softmodem-build.sh')
			cmd.run('chmod 775 ./my-lte-softmodem-build.sh')
			cmd.run('sudo ls')
			logging.debug(cmd.getBefore()) # print current directory contents for verification
			cmd.run('echo $USER; nohup sudo -E ./my-lte-softmodem-build.sh' + ' > ' + lSourcePath + '/cmake_targets/compile_oai_enb.log ' + ' 2>&1 &')
			cmd.close()
			HTML.CreateHtmlTestRow(self.Build_eNB_args, 'OK', CONST.ALL_PROCESSES_OK)
			self.backgroundBuildTestId[int(self.eNB_instance)] = self.testCase_id
			return
		cmd.run(f'stdbuf -o0 ./build_oai {self.Build_eNB_args} 2>&1 | stdbuf -o0 tee compile_oai_enb.log', timeout=900)
		cmd.close()
		self.checkBuildeNB(lIpAddr, lSourcePath, self.testCase_id, HTML)

	def WaitBuildeNBisFinished(self, HTML):
		if self.eNB_serverId[self.eNB_instance] == '0':
			lIpAddr = self.eNBIPAddress
			lSourcePath = self.eNBSourceCodePath
		elif self.eNB_serverId[self.eNB_instance] == '1':
			lIpAddr = self.eNB1IPAddress
			lSourcePath = self.eNB1SourceCodePath
		elif self.eNB_serverId[self.eNB_instance] == '2':
			lIpAddr = self.eNB2IPAddress
			lSourcePath = self.eNB2SourceCodePath
		if lIpAddr == '' or lSourcePath == '':
			HELP.GenericHelp(CONST.Version)
			sys.exit('Insufficient Parameter')
		logging.debug('Waiting for end of build on server: ' + lIpAddr)
		cmd = cls_cmd.getConnection(lIpAddr)
		count = 40
		buildOAIprocess = True
		while (count > 0) and buildOAIprocess:
			cmd.run('ps aux | grep --color=never build_ | grep -v grep')
			result = re.search('build_oai', cmd.getBefore())
			if result is None:
				buildOAIprocess = False
			else:
				count -= 1
				time.sleep(30)
		cmd.close()
		self.checkBuildeNB(lIpAddr, lSourcePath, self.backgroundBuildTestId[int(self.eNB_instance)], HTML)

	def CustomCommand(self, HTML):
		cmd = cls_cmd.getConnection(self.node)
		ret = cmd.run(self.command)
		cmd.close()
		logging.debug(f"Custum command : {self.command} on node : {self.node} returnCode : {ret.returncode}")
		status = 'OK'
		message = []
		if ret.returncode != 0 and not self.command_fail:
			message = [ret.stdout]
			logging.warning(f'CustomCommand output: {message}')
			status = 'Warning'
		if self.command_fail: # important command since it would make the pipeline fail, so show output in HTML
			message = [ret.stdout]
		if ret.returncode != 0 and self.command_fail:
			message = [ret.stdout]
			logging.error(f'CustomCommand failed: output: {message}')
			status = 'KO'
			self.prematureExit = True
		HTML.CreateHtmlTestRowQueue(self.command, status, message)

	def checkBuildeNB(self, lIpAddr, lSourcePath, testcaseId, HTML):
		HTML.testCase_id=testcaseId
		cmd = cls_cmd.getConnection(lIpAddr)
		cmd_enb = cls_cmd.getConnection(self.eNBIPAddress)
		cmd.cd('{lSourcePath}/cmake_targets')
		cmd.run('ls ran_build/build')
		cmd.run('ls ran_build/build')

		#check if we have the build corresponding to the air interface keywords (nr-softmode, lte-softmodem)
		logging.info(f'CHECK Build with IP={lIpAddr} SourcePath={lSourcePath}')
		result = re.search(self.air_interface[self.eNB_instance], cmd.getBefore())
		if result is None:
			buildStatus = False #if not, build failed
		else:
			buildStatus = True 
			# Generating a BUILD INFO file
		cmd.run(f'mkdir -p build_log_{testcaseId}')
		cmd.run(f'mv log/* build_log_{testcaseId}')
		cmd.run(f'mv compile_oai_enb.log build_log_{testcaseId}')
		if self.eNB_serverId[self.eNB_instance] != '0':
			cmd.cd('cmake_targets')
			cmd.run(f'if [ -e tmp_build{testcaseId}.zip ]; then rm -f tmp_build{testcaseId}.zip; fi')
			cmd.run(f'zip -r -qq tmp_build{testcaseId}.zip build_log_{testcaseId}')
			cmd.close()
			if (os.path.isfile(f'./tmp_build{testcaseId}.zip')):
				os.remove(f'./tmp_build{testcaseId}.zip')
			cmd.copyin(f'{lSourcePath}/cmake_targets/tmp_build{testcaseId}.zip', f"tmp_build{testcaseId}.zip")
			if (os.path.isfile(f'./tmp_build{testcaseId}.zip')):
				cmd_enb.copyout(f'./tmp_build{testcaseId}.zip', f'{self.eNBSourceCodePath}/cmake_targets/.')
				os.remove(f'./tmp_build{testcaseId}.zip')
				cmd_enb.cd(f'{self.eNBSourceCodePath}/cmake_targets')
				#-qq quiet / -u update orcreate files
				cmd_enb.run(f'unzip -o -u -qq -DD tmp_build{testcaseId}.zip')
				cmd_enb.run('rm -f tmp_build{testcaseId}.zip')
				cmd_enb.close()
		else:
			cmd.close()

		#generate logging info depending on buildStatus and air interface
		if buildStatus:
			logging.info(f'\u001B[1m Building OAI {self.air_interface[self.eNB_instance]} Pass\u001B[0m')
			HTML.CreateHtmlTestRow(self.Build_eNB_args, 'OK', CONST.ALL_PROCESSES_OK)
		else:
			logging.error('\u001B[1m Building OAI {self.air_interface[self.eNB_instance]} Failed\u001B[0m')
			HTML.CreateHtmlTestRow(self.Build_eNB_args, 'KO', CONST.ALL_PROCESSES_OK)
			HTML.CreateHtmlTabFooter(False)
			sys.exit(1)

	def InitializeeNB(self, HTML, EPC):
		if self.eNB_serverId[self.eNB_instance] == '0':
			lIpAddr = self.eNBIPAddress
			lSourcePath = self.eNBSourceCodePath
		elif self.eNB_serverId[self.eNB_instance] == '1':
			lIpAddr = self.eNB1IPAddress
			lSourcePath = self.eNB1SourceCodePath
		elif self.eNB_serverId[self.eNB_instance] == '2':
			lIpAddr = self.eNB2IPAddress
			lSourcePath = self.eNB2SourceCodePath
		if lIpAddr == '' or lSourcePath == '':
			HELP.GenericHelp(CONST.Version)
			sys.exit('Insufficient Parameter')
		logging.debug(f'Starting eNB/gNB on server: {lIpAddr}')

		self.testCase_id = HTML.testCase_id
		cwd = os.getcwd()
		cmd = cls_cmd.getConnection(lIpAddr)
		cmd.copyout(f"{cwd}/active_net_interfaces.awk", "/tmp/active_net_interfaces.awk")
		cmd.close()
		#Get pcap on enb and/or gnb if enabled in the xml
		if self.eNB_Trace=='yes':
			if self.air_interface[self.eNB_instance] == 'lte-softmodem':
				pcapfile_prefix="enb_"
			else:
				pcapfile_prefix="gnb_"
			eth_interface = 'any'
			fltr = 'sctp'
			logging.debug('\u001B[1m Launching tshark on xNB on interface ' + eth_interface + ' with filter "' + fltr + '"\u001B[0m')
			pcapfile = pcapfile_prefix + self.testCase_id + '_log.pcap'
			cmd = cls_cmd.getConnection(lIpAddr)
			cmd.run(f'sudo rm -f /tmp/{pcapfile}')
			cmd.run(f'echo $USER; nohup sudo -E tshark  -i {eth_interface} -f "{fltr}" -w /tmp/{pcapfile} > /dev/null 2>&1 &')
			cmd.close()

		# If tracer options is on, running tshark on EPC side and capture traffic b/ EPC and eNB
		if EPC.IPAddress != "none":
			localEpcIpAddr = EPC.IPAddress
			cmd = cls_cmd.getConnection(localEpcIpAddr)
			eth_interface = 'any'
			fltr = 'sctp'
			logging.debug(f'\u001B[1m Launching tshark on EPC on interface {eth_interface} with filter "{fltr}"\u001B[0m')
			self.epcPcapFile = 'enb_' + self.testCase_id + '_s1log.pcap'
			cmd.run(f'sudo rm -f /tmp/{self.epcPcapFile}')
			cmd.run(f'echo $USER; nohup sudo tshark -f "host {lIpAddr}" -i {eth_interface} -f "{fltr}" -w /tmp/{self.epcPcapFile} > /tmp/tshark.log 2>&1 &')
			cmd.close()

		cmd = cls_cmd.getConnection(lIpAddr)
		cmd.cd(lSourcePath)
		# Initialize_eNB_args usually start with -O and followed by the location in repository
		full_config_file = self.Initialize_eNB_args.replace('-O ','')
		extra_options = ''
		extIdx = full_config_file.find('.conf')
		if (extIdx > 0):
			extra_options = full_config_file[extIdx + 5:]
			# if tracer options is on, compiling and running T Tracer
			result = re.search('T_stdout', str(extra_options))
			if result is not None:
				logging.debug('\u001B[1m Compiling and launching T Tracer\u001B[0m')
				cmd.cd('common/utils/T/tracer')
				cmd.run('make')
				cmd.run(f'echo $USER; nohup ./record -d ../T_messages.txt -o {lSourcePath}/cmake_targets/enb_{self.testCase_id}_record.raw -ON -off VCD -off HEAVY -off LEGACY_GROUP_TRACE -off LEGACY_GROUP_DEBUG > {lSourcePath}/cmake_targets/enb_{self.testCase_id}_record.log 2>&1 &')
				cmd.cd(lSourcePath)
			full_config_file = full_config_file[:extIdx + 5]
			config_path, config_file = os.path.split(full_config_file)
		else:
			sys.exit('Insufficient Parameter')
		ci_full_config_file = config_path + '/ci-' + config_file
		rruCheck = False
		result = re.search('^rru|^rcc|^du.band', str(config_file))
		if result is not None:
			rruCheck = True
		# Make a copy and adapt to EPC / eNB IP addresses
		cmd.run('cp {full_config_file} {ci_full_config_file}')
		localMmeIpAddr = EPC.MmeIPAddress
		cmd.run(f'sed -i -e \'s/CI_MME_IP_ADDR/{localMmeIpAddr}/\' {ci_full_config_file}');
		cmd.run(f'sed -i -e \'s/CI_ENB_IP_ADDR/{lIpAddr}/\' {ci_full_config_file}');
		cmd.run(f'sed -i -e \'s/CI_GNB_IP_ADDR/{lIpAddr}/\' {ci_full_config_file}');
		cmd.run(f'sed -i -e \'s/CI_RCC_IP_ADDR/{self.eNBIPAddress}/\' {ci_full_config_file}');
		cmd.run(f'sed -i -e \'s/CI_RRU1_IP_ADDR/{self.eNB1IPAddress}/\' {ci_full_config_file}');
		cmd.run(f'sed -i -e \'s/CI_RRU2_IP_ADDR/{self.eNB2IPAddress}/\' {ci_full_config_file}');
		cmd.run(f'sed -i -e \'s/CI_FR1_CTL_ENB_IP_ADDR/{self.eNBIPAddress}/\' {ci_full_config_file}');
		self.eNBmbmsEnables[int(self.eNB_instance)] = False
		cmd.run(f'grep --colour=never enable_enb_m2 {ci_full_config_file}');
		result = re.search('yes', cmd.getBefore())
		if result is not None:
			self.eNBmbmsEnables[int(self.eNB_instance)] = True
			logging.debug('\u001B[1m MBMS is enabled on this eNB\u001B[0m')
		result = re.search('noS1', str(self.Initialize_eNB_args))
		eNBinNoS1 = False
		if result is not None:
			eNBinNoS1 = True
			logging.debug('\u001B[1m eNB is in noS1 configuration \u001B[0m')
		# Launch eNB with the modified config file
		cmd.run('source oaienv')
		cmd.cd('cmake_targets')
		if self.air_interface[self.eNB_instance] == 'nr-softmodem':
			cmd.run('if [ -e rbconfig.raw ]; then sudo rm rbconfig.raw; fi')
			cmd.run('if [ -e reconfig.raw ]; then sudo rm reconfig.raw; fi')
		# NOTE: WE SHALL do a check if the executable is present (in case build went wrong)

		#hack UHD_RFNOC_DIR variable for gNB / N310 on RHEL8 server:
		#if the USRP address is in the xml then we are using an eth USRP (N3xx)
		if self.air_interface[self.eNB_instance] == 'lte-softmodem':
			gNB = False
		else:
			gNB = True
		cmd.run(f'echo "ulimit -c unlimited && {self.cmd_prefix} ./ran_build/build/{self.air_interface[self.eNB_instance]} -O {lSourcePath}/{ci_full_config_file} {extra_options}" > ./my-lte-softmodem-run{self.eNB_instance}.sh')

		cmd.run(f'chmod 775 ./my-lte-softmodem-run{self.eNB_instance}.sh')
		cmd.run(f'sudo rm -Rf enb_{self.testCase_id}.log')
		cmd.run(f'echo $USER; nohup sudo -E stdbuf -o0 ./my-lte-softmodem-run{self.eNB_instance}.sh > {lSourcePath}/cmake_targets/enb_{self.testCase_id}.log 2>&1 &')


		#stats monitoring during runtime
		time.sleep(20)
		monitor_file='../ci-scripts/stats_monitor.py'
		conf_file='../ci-scripts/stats_monitor_conf.yaml'
		if self.eNB_Stats=='yes':
			if self.air_interface[self.eNB_instance] == 'lte-softmodem':
				cmd.run(f'echo $USER; nohup python3 {monitor_file} {conf_file} {self.testCase_id} enb 2>&1 > enb_stats_monitor_execution.log &')
			else:
				cmd.run(f'echo $USER; nohup python3 {monitor_file} {conf_file} {self.testCase_id} gnb 2>&1 > gnb_stats_monitor_execution.log &')



		self.eNBLogFiles[int(self.eNB_instance)] = f'enb_{self.testCase_id}.log'
		if extra_options != '':
			self.eNBOptions[int(self.eNB_instance)] = extra_options
		time.sleep(6)
		doLoop = True
		loopCounter = 20
		enbDidSync = False
		while (doLoop):
			loopCounter = loopCounter - 1
			if (loopCounter == 0):
				# In case of T tracer recording, we may need to kill it
				result = re.search('T_stdout', str(self.Initialize_eNB_args))
				if result is not None:
					cmd.run('killall --signal SIGKILL record')
				cmd.close()
				doLoop = False
				logging.error('\u001B[1;37;41m eNB/gNB logging system did not show got sync! \u001B[0m')
				HTML.CreateHtmlTestRow(self.air_interface[self.eNB_instance] + ' -O ' + config_file + extra_options, 'KO', CONST.ALL_PROCESSES_OK)
				# In case of T tracer recording, we need to kill tshark on EPC side
				localEpcIpAddr = EPC.IPAddress
				cmd = cls_cmd.getConnection(localEpcIpAddr)
				logging.debug('\u001B[1m Stopping tshark on EPC \u001B[0m')
				cmd.run('sudo -S killall --signal SIGKILL tshark')
				if self.epcPcapFile  != '':
					cmd.run(f'sudo chmod 666 /tmp/{self.epcPcapFile}')
				cmd.close()
				if self.epcPcapFile != '':
					copyin_res = cmd.copyin(f'/tmp/{self.epcPcapFile}', self.epcPcapFile)
					if (copyin_res == 0):
						cmd.copyout(self.epcPcapFile, f'{lSourcePath}/cmake_targets/{self.epcPcapFile}')
				self.prematureExit = True
				return
			else:
				cmd.run(f'stdbuf -o0 cat enb_{self.testCase_id}.log | egrep --text --color=never -i "wait|sync|Starting|Started"')
				if rruCheck:
					result = re.search('wait RUs', cmd.getBefore())
				else:
					result = re.search('got sync|Starting F1AP at CU', cmd.getBefore())
				if result is None:
					time.sleep(6)
				else:
					doLoop = False
					enbDidSync = True
					time.sleep(10)

		rruCheck = False
		result = re.search('^rru|^du.band', str(config_file))
		if result is not None:
			rruCheck = True
		if enbDidSync and eNBinNoS1 and not rruCheck:
			cmd.run('ifconfig oaitun_enb1')
			cmd.run('ifconfig oaitun_enb1')
			result = re.search('inet addr:1|inet 1', cmd.getBefore())
			if result is not None:
				logging.debug('\u001B[1m oaitun_enb1 interface is mounted and configured\u001B[0m')
			else:
				logging.error('\u001B[1m oaitun_enb1 interface is either NOT mounted or NOT configured\u001B[0m')
			if self.eNBmbmsEnables[int(self.eNB_instance)]:
				cmd.run('ifconfig oaitun_enm1')
				result = re.search('inet addr', cmd.getBefore())
				if result is not None:
					logging.debug('\u001B[1m oaitun_enm1 interface is mounted and configured\u001B[0m')
				else:
					logging.error('\u001B[1m oaitun_enm1 interface is either NOT mounted or NOT configured\u001B[0m')
		if enbDidSync:
			self.eNBstatuses[int(self.eNB_instance)] = int(self.eNB_serverId[self.eNB_instance])

		cmd.close()


		HTML.CreateHtmlTestRow(f'{self.cmd_prefix} {self.air_interface[self.eNB_instance]} -O {config_file} {extra_options}', 'OK', CONST.ALL_PROCESSES_OK)
		logging.debug('\u001B[1m Initialize eNB/gNB Completed\u001B[0m')

	def CheckeNBProcess(self, status_queue):
		try:
			# At least the instance 0 SHALL be on!
			if self.eNBstatuses[0] == 0:
				lIpAddr = self.eNBIPAddress
			elif self.eNBstatuses[0] == 1:
				lIpAddr = self.eNB1IPAddress
			elif self.eNBstatuses[0] == 2:
				lIpAddr = self.eNB2IPAddress
			else:
				lIpAddr = self.eNBIPAddress
			cmd = cls_cmd.getConnection(lIpAddr)
			if self.air_interface[self.eNB_instance] == '':
				pattern = 'softmodem'
			else:
				pattern = self.air_interface[self.eNB_instance]
			cmd.run(f'stdbuf -o0 ps -aux | grep --color=never {pattern} | grep -v grep')
			result = re.search(pattern, cmd.getBefore())
			if result is None:
				logging.debug('\u001B[1;37;41m eNB Process Not Found! \u001B[0m')
				status_queue.put(CONST.ENB_PROCESS_FAILED)
			else:
				status_queue.put(CONST.ENB_PROCESS_OK)
			cmd.close()
		except:
			os.kill(os.getppid(),signal.SIGUSR1)

	def TerminateeNB(self, HTML, EPC):
		if self.eNB_serverId[self.eNB_instance] == '0':
			lIpAddr = self.eNBIPAddress
			lSourcePath = self.eNBSourceCodePath
		elif self.eNB_serverId[self.eNB_instance] == '1':
			lIpAddr = self.eNB1IPAddress
			lSourcePath = self.eNB1SourceCodePath
		elif self.eNB_serverId[self.eNB_instance] == '2':
			lIpAddr = self.eNB2IPAddress
			lSourcePath = self.eNB2SourceCodePath
		if lIpAddr == '' or lSourcePath == '':
			HELP.GenericHelp(CONST.Version)
			sys.exit('Insufficient Parameter')
		logging.debug(f'Stopping eNB/gNB on server: {lIpAddr}')
		cmd = cls_cmd.getConnection(lIpAddr)
		cmd.cd(f'{lSourcePath}/cmake_targets')
		if self.air_interface[self.eNB_instance] == 'lte-softmodem':
			nodeB_prefix = 'e'
		else:
			nodeB_prefix = 'g'
		cmd.run('stdbuf -o0  ps -aux | grep --color=never -e softmodem | grep -v grep')
		result = re.search('-softmodem', cmd.getBefore())
		if result is not None:
			cmd.run('sudo killall --signal SIGINT -r .*-softmodem || true')
			time.sleep(10)
			cmd.run('stdbuf -o0  ps -aux | grep --color=never -e softmodem | grep -v grep')
			result = re.search('-softmodem', cmd.getBefore())
			if result is not None:
				cmd.run('sudo killall --signal SIGKILL -r .*-softmodem || true')
				time.sleep(5)
		cmd.run('rm -f my-lte-softmodem-run' + str(self.eNB_instance) + '.sh')
		#stopping tshark (valid if eNB and enabled in xml, will not harm otherwise)
		logging.debug('\u001B[1m Stopping tshark on xNB \u001B[0m')
		cmd.run('sudo killall --signal SIGKILL tshark')
		time.sleep(1)

		if EPC.IPAddress != "none" and EPC.IPAddress != '':
			localEpcIpAddr = EPC.IPAddress
			logging.debug(f'\u001B[1m Stopping tshark on EPC ({localEpcIpAddr}) \u001B[0m')
			cmd_epc = cls_cmd.getConnection(localEpcIpAddr)
			cmd_epc.run('sudo killall --signal SIGKILL tshark')
			time.sleep(1)
			if self.epcPcapFile != '':
				cmd_epc.run(f'sudo chmod 666 /tmp/{self.epcPcapFile}')
				cmd_epc.copyin(f'/tmp/{self.epcPcapFile}', f'{self.epcPcapFile}')
				cmd_epc.copyout(self.epcPcapFile, f'{lSourcePath}/cmake_targets/{self.epcPcapFile}')
			cmd_epc.run('killall --signal SIGKILL record')
			cmd_epc.close()
		# if T tracer was run with option 0 (no logs), analyze logs
		# from textlog, otherwise do normal analysis (e.g., option 2)
		result = re.search('T_stdout 0', str(self.Initialize_eNB_args))
		if (result is not None):
			logging.debug('\u001B[1m Replaying RAW record file\u001B[0m')
			cmd.cd(f'{lSourcePath}/common/utils/T/tracer/')
			enbLogFile = self.eNBLogFiles[int(self.eNB_instance)]
			raw_record_file = enbLogFile.replace('.log', '_record.raw')
			replay_log_file = enbLogFile.replace('.log', '_replay.log')
			extracted_txt_file = enbLogFile.replace('.log', '_extracted_messages.txt')
			extracted_log_file = enbLogFile.replace('.log', '_extracted_messages.log')
			cmd.run(f'./extract_config -i {lSourcePath}/cmake_targets/{raw_record_file} > {lSourcePath}/cmake_targets/{extracted_txt_file}')
			cmd.run(f'echo $USER; nohup ./replay -i {lSourcePath}/cmake_targets/{raw_record_file} > {lSourcePath}/cmake_targets/{replay_log_file} 2>&1 &')
			cmd.run(f'./textlog -d {lSourcePath}/cmake_targets/{extracted_txt_file} -no-gui -ON -full > {lSourcePath}/cmake_targets/{extracted_log_file}')
			cmd.copyin(f'{lSourcePath}/cmake_targets/{extracted_log_file}', extracted_log_file)
			logging.debug('\u001B[1m Analyzing eNB replay logfile \u001B[0m')
			logStatus = self.AnalyzeLogFile_eNB(extracted_log_file, HTML, self.ran_checkers)
			HTML.CreateHtmlTestRow(self.runtime_stats, 'OK', CONST.ALL_PROCESSES_OK)
			self.eNBLogFiles[int(self.eNB_instance)] = ''
		else:
			analyzeFile = False
			if self.eNBLogFiles[int(self.eNB_instance)] != '':
				analyzeFile = True
				fileToAnalyze = self.eNBLogFiles[int(self.eNB_instance)]
				self.eNBLogFiles[int(self.eNB_instance)] = ''
			if analyzeFile:
				#*stats.log files + pickle + png
				cmd.copyin(f'{lSourcePath}/cmake_targets/*stats.log', '*stats.log', recursive=True)
				cmd.copyin(f'{lSourcePath}/cmake_targets/*.pickle', '*.pickle', recursive=True)
				cmd.copyin(f'{lSourcePath}/cmake_targets/*.png', '*.png', recursive=True)
				#
				copyin_res = cmd.copyin(f'{lSourcePath}/cmake_targets/{fileToAnalyze}', fileToAnalyze)
				if (copyin_res == -1):
					logging.debug('\u001B[1;37;41m Could not copy ' + nodeB_prefix + 'NB logfile to analyze it! \u001B[0m')
					HTML.htmleNBFailureMsg='Could not copy ' + nodeB_prefix + 'NB logfile to analyze it!'
					HTML.CreateHtmlTestRow('N/A', 'KO', CONST.ENB_PROCESS_NOLOGFILE_TO_ANALYZE)
					self.eNBmbmsEnables[int(self.eNB_instance)] = False
					return
				if self.eNB_serverId[self.eNB_instance] != '0':
					#*stats.log files + pickle + png

					#debug / tentative
					cmd.copyout('./nrL1_stats.log', f'{self.eNBSourceCodePath}/cmake_targets/nrL1_stats.log')
					cmd.copyout('./nrMAC_stats.log', f'{self.eNBSourceCodePath}/cmake_targets/nrMAC_stats.log')
					cmd.copyout('./gnb_stats_monitor.pickle', f'{self.eNBSourceCodePath}/cmake_targets/gnb_stats_monitor.pickle')
					cmd.copyout('./gnb_stats_monitor.png', f'{self.eNBSourceCodePath}/cmake_targets/gnb_stats_monitor.png')
					#
					cmd.copyout(f'./{fileToAnalyze}', f'{self.eNBSourceCodePath}/cmake_targets/{fileToAnalyze}')
				logging.debug('\u001B[1m Analyzing ' + nodeB_prefix + 'NB logfile \u001B[0m ' + fileToAnalyze)
				logStatus = self.AnalyzeLogFile_eNB(fileToAnalyze, HTML, self.ran_checkers)
				if (logStatus < 0):
					HTML.CreateHtmlTestRow('N/A', 'KO', logStatus)
					#display rt stats for gNB only
					if len(self.datalog_rt_stats)!=0 and nodeB_prefix == 'g':
						HTML.CreateHtmlDataLogTable(self.datalog_rt_stats)
					self.prematureExit = True
					self.eNBmbmsEnables[int(self.eNB_instance)] = False
					return
				else:
					HTML.CreateHtmlTestRow(self.runtime_stats, 'OK', CONST.ALL_PROCESSES_OK)
			else:
				HTML.CreateHtmlTestRow(self.runtime_stats, 'OK', CONST.ALL_PROCESSES_OK)
		#display rt stats for gNB only
		if len(self.datalog_rt_stats)!=0 and nodeB_prefix == 'g':
			HTML.CreateHtmlDataLogTable(self.datalog_rt_stats)
		self.eNBmbmsEnables[int(self.eNB_instance)] = False
		self.eNBstatuses[int(self.eNB_instance)] = -1
		cmd.close()

	def LogCollecteNB(self):
		cmd = cls_cmd.getConnection(self.eNBIPAddress)
		# Copying back to xNB server any log from all the runs.
		# Should also contains ping and iperf logs
		absPath = os.path.abspath('.')
		if absPath.count('ci-scripts') == 0:
			os.chdir('./ci-scripts')

		for x in os.listdir():
			if x.endswith('.log') or x.endswith('.log.png'):
				cmd.copyout( x, f'{self.eNBSourceCodePath}/cmake_targets/', silent=True, ignorePermDenied=True)
		# Back to normal
		cmd.cd(self.eNBSourceCodePath)
		cmd.cd('cmake_targets')
		cmd.run('sudo mv /tmp/enb_*.pcap .')
		cmd.run('sudo mv /tmp/gnb_*.pcap .')
		cmd.run('sudo rm -f enb.log.zip')
		cmd.run('sudo zip enb.log.zip *.log enb_*record.raw enb_*.pcap gnb_*.pcap enb_*txt physim_*.log *stats.log *monitor.pickle *monitor*.png ping*.log* iperf*.log log/*/*.log log/*/*.pcap')
		result = re.search('core.\d+', cmd.getBefore())
		if result is not None:
			cmd.run('sudo zip enb.log.zip core* ran_build/build/{lte,nr}-softmodem') # add core and executable to zip
		cmd.run('sudo rm enb*.log core* enb_*record.raw enb_*.pcap gnb_*.pcap enb_*txt physim_*.log *stats.log *monitor.pickle *monitor*.png ping*.log* iperf*.log log/*/*.log log/*/*.pcap')
		cmd.close()

	def _analyzeUeRetx(self, rounds, checkers, regex):
		if len(rounds) == 0 or len(checkers) == 0:
			logging.warning(f'warning: rounds={rounds} checkers={checkers}')
			return []

		perc = list(0 for i in checkers) # results in %
		stats = list(False for i in checkers) # status if succeeded
		tmp = re.match(regex, rounds)
		if tmp is None:
			logging.error('_analyzeUeRetx: did not match regex for DL retx analysis')
			return stats
		retx_data = [float(x) for x in tmp.groups()]
		for i in range(0, len(perc)):
			#case where numerator > denumerator with denum ==0 is disregarded, cannot hapen in principle, will lead to 0%
			perc[i] = 0 if (retx_data[i] == 0) else 100 * retx_data[i + 1] / retx_data[i]
			#treating % > 100 , % > requirement
			stats[i] = perc[i] <= 100 and perc[i] <= checkers[i]
		return stats

	def AnalyzeLogFile_eNB(self, eNBlogFile, HTML, checkers={}):
		if (not os.path.isfile(eNBlogFile)):
			return -1
		enb_log_file = open(eNBlogFile, 'r')
		exitSignalReceived = False
		foundAssertion = False
		msgAssertion = ''
		msgLine = 0
		foundSegFault = False
		foundRealTimeIssue = False
		foundRealTimeIssue_cnt = 0
		rrcSetupComplete = 0
		rrcReleaseRequest = 0
		rrcReconfigRequest = 0
		rrcReconfigComplete = 0
		rrcReestablishRequest = 0
		rrcReestablishComplete = 0
		rrcReestablishReject = 0
		rlcDiscardBuffer = 0
		rachCanceledProcedure = 0
		uciStatMsgCount = 0
		pdcpFailure = 0
		ulschFailure = 0
		ulschAllocateCCEerror = 0
		uplinkSegmentsAborted = 0
		ulschReceiveOK = 0
		gnbRxTxWakeUpFailure = 0
		gnbTxWriteThreadEnabled = False
		cdrxActivationMessageCount = 0
		dropNotEnoughRBs = 0
		mbmsRequestMsg = 0
		htmleNBFailureMsg = ''
		isRRU = False
		isSlave = False
		slaveReceivesFrameResyncCmd = False
		X2HO_state = CONST.X2_HO_REQ_STATE__IDLE
		X2HO_inNbProcedures = 0
		X2HO_outNbProcedures = 0
		global_status = CONST.ALL_PROCESSES_OK
		# Runtime statistics
		runTime = ''
		userTime = ''
		systemTime = ''
		maxPhyMemUsage = ''
		nbContextSwitches = ''
		#NSA FR1 check
		NSA_RAPROC_PUSCH_check = 0
		#dlsch and ulsch statistics (dictionary)
		dlsch_ulsch_stats = {}
		#real time statistics (dictionary)
		real_time_stats = {}
		#count "problem receiving samples" msg
		pb_receiving_samples_cnt = 0
		#count "removing UE" msg
		removing_ue = 0
		#count"X2AP-PDU"
		x2ap_pdu = 0
		#gnb specific log markers
		gnb_markers ={'SgNBReleaseRequestAcknowledge': [],'FAILURE': [], 'scgFailureInformationNR-r15': [], 'SgNBReleaseRequest': [], 'Detected UL Failure on PUSCH':[]}
		nodeB_prefix_found = False
		RealTimeProcessingIssue = False
		retx_status = {}
		nrRrcRcfgComplete = 0
		harqFeedbackPast = 0
		showedByeMsg = False # last line is Bye. -> stopped properly
	
		line_cnt=0 #log file line counter
		for line in enb_log_file.readlines():
			line_cnt+=1
			# Detection of eNB/gNB from a container log
			result = re.search('Starting eNB soft modem', str(line))
			if result is not None:
				nodeB_prefix_found = True
				nodeB_prefix = 'e'
			result = re.search('Starting gNB soft modem', str(line))
			if result is not None:
				nodeB_prefix_found = True
				nodeB_prefix = 'g'
			result = re.search('Run time:' ,str(line))
			# Runtime statistics
			result = re.search('Run time:' ,str(line))
			if result is not None:
				runTime = str(line).strip()
			if runTime != '':
				result = re.search('Time executing user inst', str(line))
				if result is not None:
					fields=line.split(':')
					userTime = 'userTime : ' + fields[1].replace('\n','')
				result = re.search('Time executing system inst', str(line))
				if result is not None:
					fields=line.split(':')
					systemTime = 'systemTime : ' + fields[1].replace('\n','')
				result = re.search('Max. Phy. memory usage:', str(line))
				if result is not None:
					fields=line.split(':')
					maxPhyMemUsage = 'maxPhyMemUsage : ' + fields[1].replace('\n','')
				result = re.search('Number of context switch.*process origin', str(line))
				if result is not None:
					fields=line.split(':')
					nbContextSwitches = 'nbContextSwitches : ' + fields[1].replace('\n','')
			if X2HO_state == CONST.X2_HO_REQ_STATE__IDLE:
				result = re.search('target eNB Receives X2 HO Req X2AP_HANDOVER_REQ', str(line))
				if result is not None:
					X2HO_state = CONST.X2_HO_REQ_STATE__TARGET_RECEIVES_REQ
				result = re.search('source eNB receives the X2 HO ACK X2AP_HANDOVER_REQ_ACK', str(line))
				if result is not None:
					X2HO_state = CONST.X2_HO_REQ_STATE__SOURCE_RECEIVES_REQ_ACK
			if X2HO_state == CONST.X2_HO_REQ_STATE__TARGET_RECEIVES_REQ:
				result = re.search('Received LTE_RRCConnectionReconfigurationComplete from UE', str(line))
				if result is not None:
					X2HO_state = CONST.X2_HO_REQ_STATE__TARGET_RRC_RECFG_COMPLETE
			if X2HO_state == CONST.X2_HO_REQ_STATE__TARGET_RRC_RECFG_COMPLETE:
				result = re.search('issue rrc_eNB_send_PATH_SWITCH_REQ', str(line))
				if result is not None:
					X2HO_state = CONST.X2_HO_REQ_STATE__TARGET_SENDS_SWITCH_REQ
			if X2HO_state == CONST.X2_HO_REQ_STATE__TARGET_SENDS_SWITCH_REQ:
				result = re.search('received path switch ack S1AP_PATH_SWITCH_REQ_ACK', str(line))
				if result is not None:
					X2HO_state = CONST.X2_HO_REQ_STATE__IDLE
					X2HO_inNbProcedures += 1
			if X2HO_state == CONST.X2_HO_REQ_STATE__SOURCE_RECEIVES_REQ_ACK:
				result = re.search('source eNB receives the X2 UE CONTEXT RELEASE X2AP_UE_CONTEXT_RELEASE', str(line))
				if result is not None:
					X2HO_state = CONST.X2_HO_REQ_STATE__IDLE
					X2HO_outNbProcedures += 1

			if self.eNBOptions[int(self.eNB_instance)] != '':
				res1 = re.search('max_rxgain (?P<requested_option>[0-9]+)', self.eNBOptions[int(self.eNB_instance)])
				res2 = re.search('max_rxgain (?P<applied_option>[0-9]+)',  str(line))
				if res1 is not None and res2 is not None:
					requested_option = int(res1.group('requested_option'))
					applied_option = int(res2.group('applied_option'))
					if requested_option == applied_option:
						htmleNBFailureMsg += '<span class="glyphicon glyphicon-ok-circle"></span> Command line option(s) correctly applied <span class="glyphicon glyphicon-arrow-right"></span> ' + self.eNBOptions[int(self.eNB_instance)] + '\n\n'
					else:
						htmleNBFailureMsg += '<span class="glyphicon glyphicon-ban-circle"></span> Command line option(s) NOT applied <span class="glyphicon glyphicon-arrow-right"></span> ' + self.eNBOptions[int(self.eNB_instance)] + '\n\n'
			result = re.search('Exiting OAI softmodem|Caught SIGTERM, shutting down', str(line))
			if result is not None:
				exitSignalReceived = True
			result = re.search('[Ss]egmentation [Ff]ault', str(line))
			if result is not None and not exitSignalReceived:
				foundSegFault = True
			result = re.search('[Cc]ore [dD]ump', str(line))
			if result is not None and not exitSignalReceived:
				foundSegFault = True
			result = re.search('[Aa]ssertion', str(line))
			if result is not None and not exitSignalReceived:
				foundAssertion = True
			result = re.search('LLL', str(line))
			if result is not None and not exitSignalReceived:
				foundRealTimeIssue = True
				foundRealTimeIssue_cnt += 1
			if foundAssertion and (msgLine < 3):
				msgLine += 1
				msgAssertion += str(line)
			result = re.search('Setting function for RU', str(line))
			if result is not None:
				isRRU = True
			if isRRU:
				result = re.search('RU 0 is_slave=yes', str(line))
				if result is not None:
					isSlave = True
				if isSlave:
					result = re.search('Received RRU_frame_resynch command', str(line))
					if result is not None:
						slaveReceivesFrameResyncCmd = True
			result = re.search('LTE_RRCConnectionSetupComplete from UE', str(line))
			if result is not None:
				rrcSetupComplete += 1
			result = re.search('Generate LTE_RRCConnectionRelease|Generate RRCConnectionRelease', str(line))
			if result is not None:				rrcReleaseRequest += 1
			result = re.search('Generate LTE_RRCConnectionReconfiguration', str(line))
			if result is not None:
				rrcReconfigRequest += 1
			result = re.search('LTE_RRCConnectionReconfigurationComplete from UE rnti', str(line))
			if result is not None:
				rrcReconfigComplete += 1
			result = re.search('LTE_RRCConnectionReestablishmentRequest', str(line))
			if result is not None:
				rrcReestablishRequest += 1
			result = re.search('LTE_RRCConnectionReestablishmentComplete', str(line))
			if result is not None:
				rrcReestablishComplete += 1
			result = re.search('LTE_RRCConnectionReestablishmentReject', str(line))
			if result is not None:
				rrcReestablishReject += 1
			result = re.search('CDRX configuration activated after RRC Connection', str(line))
			if result is not None:
				cdrxActivationMessageCount += 1
			result = re.search('uci->stat', str(line))
			if result is not None:
				uciStatMsgCount += 1
			result = re.search('PDCP.*Out of Resources.*reason', str(line))
			if result is not None:
				pdcpFailure += 1
			result = re.search('could not wakeup gNB rxtx process', str(line))
			if result is not None:
				gnbRxTxWakeUpFailure += 1
			result = re.search('tx write thread ready', str(line))
			if result is not None:
				gnbTxWriteThreadEnabled = True
			result = re.search('ULSCH in error in round|ULSCH 0 in error', str(line))
			if result is not None:
				ulschFailure += 1
			result = re.search('ERROR ALLOCATING CCEs', str(line))
			if result is not None:
				ulschAllocateCCEerror += 1
			result = re.search('uplink segment error.*aborted [1-9] segments', str(line))
			if result is not None:
				uplinkSegmentsAborted += 1
			result = re.search('ULSCH received ok', str(line))
			if result is not None:
				ulschReceiveOK += 1
			result = re.search('BAD all_segments_received', str(line))
			if result is not None:
				rlcDiscardBuffer += 1
			result = re.search('Canceled RA procedure for UE rnti', str(line))
			if result is not None:
				rachCanceledProcedure += 1
			result = re.search('dropping, not enough RBs', str(line))
			if result is not None:
				dropNotEnoughRBs += 1
			if self.eNBmbmsEnables[int(self.eNB_instance)]:
				result = re.search('MBMS USER-PLANE.*Requesting.*bytes from RLC', str(line))
				if result is not None:
					mbmsRequestMsg += 1
			#FR1 NSA test : add new markers to make sure gNB is used
			result = re.search('\[gNB [0-9]+\]\[RAPROC\] PUSCH with TC_RNTI 0x[0-9a-fA-F]+ received correctly, adding UE MAC Context RNTI 0x[0-9a-fA-F]+', str(line))
			if result is not None:
				NSA_RAPROC_PUSCH_check = 1

			# Collect information on UE DLSCH and ULSCH statistics
			keys = {'dlsch_rounds','dlsch_total_bytes','ulsch_rounds','ulsch_total_bytes_scheduled'}
			for k in keys:
				result = re.search(k, line)
				if result is None:
					continue
				result = re.search('UE (?:RNTI )?([0-9a-f]{4})', line)
				if result is None:
					logging.error(f'did not find RNTI while matching key {k}')
					continue
				rnti = result.group(1)

				#remove 1- all useless char before relevant info (ulsch or dlsch) 2- trailing char
				if not rnti in dlsch_ulsch_stats: dlsch_ulsch_stats[rnti] = {}
				dlsch_ulsch_stats[rnti][k]=re.sub(r'^.*\]\s+', r'' , line.rstrip())

			result = re.search('Received NR_RRCReconfigurationComplete from UE', str(line))
			if result is not None:
				nrRrcRcfgComplete += 1
			result = re.search('HARQ feedback is in the past', str(line))
			if result is not None:
				harqFeedbackPast += 1


			#count "problem receiving samples" msg
			result = re.search('\[PHY\]\s+problem receiving samples', str(line))
			if result is not None:
				pb_receiving_samples_cnt += 1
			#count "Removing UE" msg
			result = re.search('\[MAC\]\s+Removing UE', str(line))
			if result is not None:
				removing_ue += 1
			#count "X2AP-PDU"
			result = re.search('X2AP-PDU', str(line))
			if result is not None:
				x2ap_pdu += 1
			#gnb markers logging
			for k in gnb_markers:
				result = re.search(k, line)
				if result is not None:
					gnb_markers[k].append(line_cnt)

			# check whether e/gNB log finishes with "Bye." message
			showedByeMsg |= re.search(r'^Bye.\n', str(line), re.MULTILINE) is not None

		enb_log_file.close()


		#the following part takes the *_stats.log files as source (not the stdout log file)

		#the datalog config file has to be loaded
		datalog_rt_stats_file=self.datalog_rt_stats_file
		if (os.path.isfile(datalog_rt_stats_file)):
			yaml_file=datalog_rt_stats_file
		elif (os.path.isfile('ci-scripts/'+datalog_rt_stats_file)):
			yaml_file='ci-scripts/'+datalog_rt_stats_file
		else:
			logging.error("Datalog RT stats yaml file cannot be found")
			sys.exit("Datalog RT stats yaml file cannot be found")

		with open(yaml_file,'r') as f:
			datalog_rt_stats = yaml.load(f,Loader=yaml.FullLoader)
		rt_keys = datalog_rt_stats['Ref'] #we use the keys from the Ref field  

		if os.path.isfile('./nrL1_stats.log') and os.path.isfile('./nrMAC_stats.log'):
			# don't use CI-nrL1_stats.log, as this will increase the processing time for
			# no reason, we just need the last occurence
			nrL1_stats = open('./nrL1_stats.log', 'r')
			nrMAC_stats = open('./nrMAC_stats.log', 'r')
			for line in nrL1_stats.readlines():
				for k in rt_keys:
					result = re.search(k, line)     
					if result is not None:
						#remove 1- all useless char before relevant info  2- trailing char
						tmp=re.match(rf'^.*?(\b{k}\b.*)',line.rstrip()) #from python 3.6 we can use literal string interpolation for the variable k, using rf' in the regex
						if tmp!=None: 
							real_time_stats[k]=tmp.group(1)
			for line in nrMAC_stats.readlines():
				for k in rt_keys:
					result = re.search(k, line)     
					if result is not None:
						#remove 1- all useless char before relevant info  2- trailing char
						tmp=re.match(rf'^.*?(\b{k}\b.*)',line.rstrip()) #from python 3.6 we can use literal string interpolation for the variable k, using rf' in the regex
						if tmp!=None: 
							real_time_stats[k]=tmp.group(1)
			nrL1_stats.close()
			nrMAC_stats.close()
		else:
			logging.debug("NR Stats files for RT analysis not found")

		#stdout log file and stat log files analysis completed
		logging.debug('   File analysis (stdout, stats) completed')

		#post processing depending on the node type
		if not nodeB_prefix_found:
			if self.air_interface[self.eNB_instance] == 'lte-softmodem':
				nodeB_prefix = 'e'
			else:
				nodeB_prefix = 'g'

		if nodeB_prefix == 'g':
			if ulschReceiveOK > 0:
				statMsg = nodeB_prefix + 'NB showed ' + str(ulschReceiveOK) + ' "ULSCH received ok" message(s)'
				logging.debug('\u001B[1;30;43m ' + statMsg + ' \u001B[0m')
				htmleNBFailureMsg += statMsg + '\n'
			if gnbRxTxWakeUpFailure > 0:
				statMsg = nodeB_prefix + 'NB showed ' + str(gnbRxTxWakeUpFailure) + ' "could not wakeup gNB rxtx process" message(s)'
				logging.debug('\u001B[1;30;43m ' + statMsg + ' \u001B[0m')
				htmleNBFailureMsg += statMsg + '\n'
			if gnbTxWriteThreadEnabled:
				statMsg = nodeB_prefix + 'NB ran with TX Write thread enabled'
				logging.debug('\u001B[1;30;43m ' + statMsg + ' \u001B[0m')
				htmleNBFailureMsg += statMsg + '\n'
			if nrRrcRcfgComplete > 0:
				statMsg = nodeB_prefix + 'NB showed ' + str(nrRrcRcfgComplete) + ' "Received NR_RRCReconfigurationComplete from UE" message(s)'
				logging.debug('\u001B[1;30;43m ' + statMsg + ' \u001B[0m')
				htmleNBFailureMsg += statMsg + '\n'
			if harqFeedbackPast > 0:
				statMsg = nodeB_prefix + 'NB showed ' + str(harqFeedbackPast) + ' "HARQ feedback is in the past" message(s)'
				logging.debug('\u001B[1;30;43m ' + statMsg + ' \u001B[0m')
				htmleNBFailureMsg += statMsg + '\n'
			#FR1 NSA test : add new markers to make sure gNB is used
			if NSA_RAPROC_PUSCH_check:
				statMsg = '[RAPROC] PUSCH with TC_RNTI message check for ' + nodeB_prefix + 'NB : PASS '
				htmlMsg = statMsg+'\n'
			else:
				statMsg = '[RAPROC] PUSCH with TC_RNTI message check for ' + nodeB_prefix + 'NB : FAIL or not relevant'
				htmlMsg = statMsg+'\n'
			logging.debug(statMsg)
			htmleNBFailureMsg += htmlMsg
			#problem receiving samples log
			statMsg = '[PHY] problem receiving samples msg count =  '+str(pb_receiving_samples_cnt)
			htmlMsg = statMsg+'\n'
			logging.debug(statMsg)
			htmleNBFailureMsg += htmlMsg
			#gnb markers
			statMsg = 'logfile line count = ' + str(line_cnt)			
			htmlMsg = statMsg+'\n'
			logging.debug(statMsg)
			htmleNBFailureMsg += htmlMsg
			if len(gnb_markers['SgNBReleaseRequestAcknowledge'])!=0:
				statMsg = 'SgNBReleaseRequestAcknowledge = ' + str(len(gnb_markers['SgNBReleaseRequestAcknowledge'])) + ' occurences , starting line ' + str(gnb_markers['SgNBReleaseRequestAcknowledge'][0])
			else:
				statMsg = 'SgNBReleaseRequestAcknowledge = ' + str(len(gnb_markers['SgNBReleaseRequestAcknowledge'])) + ' occurences' 
			htmlMsg = statMsg+'\n'
			logging.debug(statMsg)
			htmleNBFailureMsg += htmlMsg
			statMsg = 'FAILURE = ' + str(len(gnb_markers['FAILURE'])) + ' occurences'
			htmlMsg = statMsg+'\n'
			logging.debug(statMsg)
			htmleNBFailureMsg += htmlMsg
			statMsg = 'Detected UL Failure on PUSCH = ' + str(len(gnb_markers['Detected UL Failure on PUSCH'])) + ' occurences'
			htmlMsg = statMsg+'\n'
			logging.debug(statMsg)
			htmleNBFailureMsg += htmlMsg

			#ulsch and dlsch statistics and checkers
			for ue in dlsch_ulsch_stats:
				dlulstat = dlsch_ulsch_stats[ue]
				#print statistics into html
				statMsg=''
				for key in dlulstat:
					statMsg += dlulstat[key] + '\n'
					logging.debug(dlulstat[key])
				htmleNBFailureMsg += statMsg

				retx_status[ue] = {}
				dlcheckers = [] if 'd_retx_th' not in checkers else checkers['d_retx_th']
				retx_status[ue]['dl'] = self._analyzeUeRetx(dlulstat['dlsch_rounds'], dlcheckers, r'^.*dlsch_rounds\s+(\d+)\/(\d+)\/(\d+)\/(\d+),\s+dlsch_errors\s+(\d+)')
				ulcheckers = [] if 'u_retx_th' not in checkers else checkers['u_retx_th']
				retx_status[ue]['ul'] = self._analyzeUeRetx(dlulstat['ulsch_rounds'], ulcheckers, r'^.*ulsch_rounds\s+(\d+)\/(\d+)\/(\d+)\/(\d+),\s+.*,\s+ulsch_errors\s+(\d+)')


			#real time statistics
			datalog_rt_stats['Data']={}
			if len(real_time_stats)!=0: #check if dictionary is not empty
				for k in real_time_stats:
					tmp=re.match(r'^(?P<metric>.*):\s+(?P<avg>\d+\.\d+) us;\s+(?P<count>\d+);\s+(?P<max>\d+\.\d+) us;',real_time_stats[k])
					if tmp is not None:
						metric=tmp.group('metric')
						avg=float(tmp.group('avg'))
						max=float(tmp.group('max'))
						count=int(tmp.group('count'))
						datalog_rt_stats['Data'][metric]=["{:.0f}".format(avg),"{:.0f}".format(max),"{:d}".format(count),"{:.2f}".format(avg/datalog_rt_stats['Ref'][metric])]
				#once all metrics are collected, store the data as a class attribute to build a dedicated HTML table afterward
				self.datalog_rt_stats=datalog_rt_stats
				#check if there is a fail => will render the test as failed
				for k in datalog_rt_stats['Data']:
					if float(datalog_rt_stats['Data'][k][3])> datalog_rt_stats['Threshold'][k]: #condition for fail : avg/ref is greater than the fixed threshold
						logging.debug('\u001B[1;30;43m datalog_rt_stats metric ' + k + '=' + datalog_rt_stats['Data'][k][3] + ' > threshold ' + str(datalog_rt_stats['Threshold'][k]) + ' \u001B[0m')
						RealTimeProcessingIssue = True
			else:
				statMsg = 'No real time stats found in the log file\n'
				logging.debug('No real time stats found in the log file')
				htmleNBFailureMsg += statMsg

			if not showedByeMsg:
				logging.debug('\u001B[1;37;41m ' + nodeB_prefix + 'NB did not show "Bye." message at end, it likely did not stop properly! \u001B[0m')
				htmleNBFailureMsg += 'No Bye. message found, did not stop properly\n'
				global_status = CONST.ENB_SHUTDOWN_NO_BYE
			else:
				logging.debug('"Bye." message found at end.')

		else:
			#Removing UE log
			statMsg = '[MAC] Removing UE msg count =  '+str(removing_ue)
			htmlMsg = statMsg+'\n'
			logging.debug(statMsg)
			htmleNBFailureMsg += htmlMsg
			#X2AP-PDU log
			statMsg = 'X2AP-PDU msg count =  '+str(x2ap_pdu)
			htmlMsg = statMsg+'\n'
			logging.debug(statMsg)
			htmleNBFailureMsg += htmlMsg
			#nsa markers
			statMsg = 'logfile line count = ' + str(line_cnt)			
			htmlMsg = statMsg+'\n'
			logging.debug(statMsg)
			htmleNBFailureMsg += htmlMsg
			if len(gnb_markers['SgNBReleaseRequest'])!=0:
				statMsg = 'SgNBReleaseRequest = ' + str(len(gnb_markers['SgNBReleaseRequest'])) + ' occurences , starting line ' + str(gnb_markers['SgNBReleaseRequest'][0])
			else:
				statMsg = 'SgNBReleaseRequest = ' + str(len(gnb_markers['SgNBReleaseRequest'])) + ' occurences'
			htmlMsg = statMsg+'\n'
			logging.debug(statMsg)
			htmleNBFailureMsg += htmlMsg
			statMsg = 'scgFailureInformationNR-r15 = ' + str(len(gnb_markers['scgFailureInformationNR-r15'])) + ' occurences'
			htmlMsg = statMsg+'\n'
			logging.debug(statMsg)
			htmleNBFailureMsg += htmlMsg			

		for ue in retx_status:
			msg = f"retransmissions for UE {ue}: DL {retx_status[ue]['dl']} UL {retx_status[ue]['ul']}"
			if False in retx_status[ue]['dl'] or False in retx_status[ue]['ul']:
				msg = 'Failure: ' + msg
				logging.error(f'\u001B[1;37;41m {msg}\u001B[0m')
				htmleNBFailureMsg += f'{msg}\n'
				global_status = CONST.ENB_RETX_ISSUE
			else:
				logging.debug(msg)

		if RealTimeProcessingIssue:
			logging.debug('\u001B[1;37;41m ' + nodeB_prefix + 'NB ended with real time processing issue! \u001B[0m')
			htmleNBFailureMsg += 'Fail due to real time processing issue\n'
			global_status = CONST.ENB_REAL_TIME_PROCESSING_ISSUE
		if uciStatMsgCount > 0:
			statMsg = nodeB_prefix + 'NB showed ' + str(uciStatMsgCount) + ' "uci->stat" message(s)'
			logging.debug('\u001B[1;30;43m ' + statMsg + ' \u001B[0m')
			htmleNBFailureMsg += statMsg + '\n'
		if pdcpFailure > 0:
			statMsg = nodeB_prefix + 'NB showed ' + str(pdcpFailure) + ' "PDCP Out of Resources" message(s)'
			logging.debug('\u001B[1;30;43m ' + statMsg + ' \u001B[0m')
			htmleNBFailureMsg += statMsg + '\n'
		if ulschFailure > 0:
			statMsg = nodeB_prefix + 'NB showed ' + str(ulschFailure) + ' "ULSCH in error in round" message(s)'
			logging.debug('\u001B[1;30;43m ' + statMsg + ' \u001B[0m')
			htmleNBFailureMsg += statMsg + '\n'
		if ulschAllocateCCEerror > 0:
			statMsg = nodeB_prefix + 'NB showed ' + str(ulschAllocateCCEerror) + ' "eNB_dlsch_ulsch_scheduler(); ERROR ALLOCATING CCEs" message(s)'
			logging.debug('\u001B[1;30;43m ' + statMsg + ' \u001B[0m')
			htmleNBFailureMsg += statMsg + '\n'
		if uplinkSegmentsAborted > 0:
			statMsg = nodeB_prefix + 'NB showed ' + str(uplinkSegmentsAborted) + ' "uplink segment error 0/2, aborted * segments" message(s)'
			logging.debug('\u001B[1;30;43m ' + statMsg + ' \u001B[0m')
			htmleNBFailureMsg += statMsg + '\n'
		if dropNotEnoughRBs > 0:
			statMsg = 'eNB showed ' + str(dropNotEnoughRBs) + ' "dropping, not enough RBs" message(s)'
			logging.debug('\u001B[1;30;43m ' + statMsg + ' \u001B[0m')
			htmleNBFailureMsg += statMsg + '\n'
		if rrcSetupComplete > 0:
			rrcMsg = nodeB_prefix + 'NB completed ' + str(rrcSetupComplete) + ' RRC Connection Setup(s)'
			logging.debug('\u001B[1;30;43m ' + rrcMsg + ' \u001B[0m')
			htmleNBFailureMsg += rrcMsg + '\n'
			rrcMsg = ' -- ' + str(rrcSetupComplete) + ' were completed'
			logging.debug('\u001B[1;30;43m ' + rrcMsg + ' \u001B[0m')
			htmleNBFailureMsg += rrcMsg + '\n'
		if rrcReleaseRequest > 0:
			rrcMsg = nodeB_prefix + 'NB requested ' + str(rrcReleaseRequest) + ' RRC Connection Release(s)'
			logging.debug('\u001B[1;30;43m ' + rrcMsg + ' \u001B[0m')
			htmleNBFailureMsg += rrcMsg + '\n'
		if rrcReconfigRequest > 0 or rrcReconfigComplete > 0:
			rrcMsg = nodeB_prefix + 'NB requested ' + str(rrcReconfigRequest) + ' RRC Connection Reconfiguration(s)'
			logging.debug('\u001B[1;30;43m ' + rrcMsg + ' \u001B[0m')
			htmleNBFailureMsg += rrcMsg + '\n'
			rrcMsg = ' -- ' + str(rrcReconfigComplete) + ' were completed'
			logging.debug('\u001B[1;30;43m ' + rrcMsg + ' \u001B[0m')
			htmleNBFailureMsg += rrcMsg + '\n'
		if rrcReestablishRequest > 0 or rrcReestablishComplete > 0 or rrcReestablishReject > 0:
			rrcMsg = nodeB_prefix + 'NB requested ' + str(rrcReestablishRequest) + ' RRC Connection Reestablishment(s)'
			logging.debug('\u001B[1;30;43m ' + rrcMsg + ' \u001B[0m')
			htmleNBFailureMsg += rrcMsg + '\n'
			rrcMsg = ' -- ' + str(rrcReestablishComplete) + ' were completed'
			logging.debug('\u001B[1;30;43m ' + rrcMsg + ' \u001B[0m')
			htmleNBFailureMsg += rrcMsg + '\n'
			rrcMsg = ' -- ' + str(rrcReestablishReject) + ' were rejected'
			logging.debug('\u001B[1;30;43m ' + rrcMsg + ' \u001B[0m')
			htmleNBFailureMsg += rrcMsg + '\n'
		if self.eNBmbmsEnables[int(self.eNB_instance)]:
			if mbmsRequestMsg > 0:
				rrcMsg = 'eNB requested ' + str(mbmsRequestMsg) + ' times the RLC for MBMS USER-PLANE'
				logging.debug('\u001B[1;30;43m ' + rrcMsg + ' \u001B[0m')
				htmleNBFailureMsg += rrcMsg + '\n'
		if X2HO_inNbProcedures > 0:
			rrcMsg = 'eNB completed ' + str(X2HO_inNbProcedures) + ' X2 Handover Connection procedure(s)'
			logging.debug('\u001B[1;30;43m ' + rrcMsg + ' \u001B[0m')
			htmleNBFailureMsg += rrcMsg + '\n'
		if X2HO_outNbProcedures > 0:
			rrcMsg = 'eNB completed ' + str(X2HO_outNbProcedures) + ' X2 Handover Release procedure(s)'
			logging.debug('\u001B[1;30;43m ' + rrcMsg + ' \u001B[0m')
			htmleNBFailureMsg += rrcMsg + '\n'
		if self.eNBOptions[int(self.eNB_instance)] != '':
			res1 = re.search('drx_Config_present prSetup', self.eNBOptions[int(self.eNB_instance)])
			if res1 is not None:
				if cdrxActivationMessageCount > 0:
					rrcMsg = 'eNB activated the CDRX Configuration for ' + str(cdrxActivationMessageCount) + ' time(s)'
					logging.debug('\u001B[1;30;43m ' + rrcMsg + ' \u001B[0m')
					htmleNBFailureMsg += rrcMsg + '\n'
				else:
					rrcMsg = 'eNB did NOT ACTIVATE the CDRX Configuration'
					logging.debug('\u001B[1;37;43m ' + rrcMsg + ' \u001B[0m')
					htmleNBFailureMsg += rrcMsg + '\n'
		if rachCanceledProcedure > 0:
			rachMsg = nodeB_prefix + 'NB cancelled ' + str(rachCanceledProcedure) + ' RA procedure(s)'
			logging.debug('\u001B[1;30;43m ' + rachMsg + ' \u001B[0m')
			htmleNBFailureMsg += rachMsg + '\n'
		if isRRU:
			if isSlave:
				if slaveReceivesFrameResyncCmd:
					rruMsg = 'Slave RRU received the RRU_frame_resynch command from RAU'
					logging.debug('\u001B[1;30;43m ' + rruMsg + ' \u001B[0m')
					htmleNBFailureMsg += rruMsg + '\n'
				else:
					rruMsg = 'Slave RRU DID NOT receive the RRU_frame_resynch command from RAU'
					logging.debug('\u001B[1;37;41m ' + rruMsg + ' \u001B[0m')
					htmleNBFailureMsg += rruMsg + '\n'
					self.prematureExit = True
					global_status = CONST.ENB_PROCESS_SLAVE_RRU_NOT_SYNCED
		if foundSegFault:
			logging.debug('\u001B[1;37;41m ' + nodeB_prefix + 'NB ended with a Segmentation Fault! \u001B[0m')
			global_status = CONST.ENB_PROCESS_SEG_FAULT
		if foundAssertion:
			logging.debug('\u001B[1;37;41m ' + nodeB_prefix + 'NB ended with an assertion! \u001B[0m')
			htmleNBFailureMsg += msgAssertion
			global_status = CONST.ENB_PROCESS_ASSERTION
		if foundRealTimeIssue:
			logging.debug('\u001B[1;37;41m ' + nodeB_prefix + 'NB faced real time issues! \u001B[0m')
			htmleNBFailureMsg += nodeB_prefix + 'NB faced real time issues! COUNT = '+ str(foundRealTimeIssue_cnt) +' lines\n'
		if rlcDiscardBuffer > 0:
			rlcMsg = nodeB_prefix + 'NB RLC discarded ' + str(rlcDiscardBuffer) + ' buffer(s)'
			logging.debug('\u001B[1;37;41m ' + rlcMsg + ' \u001B[0m')
			htmleNBFailureMsg += rlcMsg + '\n'
			global_status = CONST.ENB_PROCESS_REALTIME_ISSUE
		HTML.htmleNBFailureMsg=htmleNBFailureMsg
		# Runtime statistics for console output and HTML
		if runTime != '':
			logging.debug(runTime)
			logging.debug(userTime)
			logging.debug(systemTime)
			logging.debug(maxPhyMemUsage)
			logging.debug(nbContextSwitches)
			self.runtime_stats='<pre>'+runTime + '\n'+ userTime + '\n' + systemTime + '\n' + maxPhyMemUsage + '\n' + nbContextSwitches+'</pre>'
		return global_status
