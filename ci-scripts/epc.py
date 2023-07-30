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
import sys	      # arg
import re	       # reg
import logging
import os
import time
import signal

from multiprocessing import Process, Lock, SimpleQueue

#-----------------------------------------------------------
# OAI Testing modules
#-----------------------------------------------------------
import helpreadme as HELP
import constants as CONST
import cls_cluster as OC
import cls_cmd
#-----------------------------------------------------------
# Class Declaration
#-----------------------------------------------------------


class EPCManagement():

	def __init__(self):

		self.IPAddress = ''
		self.SourceCodePath = ''
		self.Type = ''
		self.PcapFileName = ''
		self.testCase_id = ''
		self.MmeIPAddress = ''
		self.containerPrefix = 'prod'
		self.mmeConfFile = 'mme.conf'
		self.yamlPath = ''
		self.isMagmaUsed = False
		self.cfgDeploy = '--type start-mini --scenario 1 --capture /tmp/oai-cn5g-v1.5.pcap' #from xml, 'mini' is default normal for docker-network.py
		self.cfgUnDeploy = '--type stop-mini --scenario 1' #from xml, 'mini' is default normal for docker-network.py
		self.OCUrl = "https://api.oai.cs.eurecom.fr:6443"
		self.OCRegistry = "default-route-openshift-image-registry.apps.oai.cs.eurecom.fr/"
		self.OCUserName = ''
		self.OCPassword = ''
		self.imageToPull = ''
		self.eNBSourceCodePath = ''

#-----------------------------------------------------------
# EPC management functions
#-----------------------------------------------------------

	def InitializeHSS(self, HTML):
		if self.IPAddress == '' or self.SourceCodePath == '' or self.Type == '':
			HELP.GenericHelp(CONST.Version)
			HELP.EPCSrvHelp(self.IPAddress, self.SourceCodePath, self.Type)
			sys.exit('Insufficient EPC Parameters')
		cmd = cls_cmd.getConnection(self.IPAddress)
		if re.match('OAI-Rel14-Docker', self.Type, re.IGNORECASE):
			logging.debug('Using the OAI EPC Release 14 Cassandra-based HSS in Docker')
			cmd.run(f'if [ -d {self.SourceCodePath}/scripts ]; then sudo -S rm -Rf ' + self.SourceCodePath + '/scripts ; fi')
			cmd.run(f'mkdir -p {self.SourceCodePath}/scripts')
			cmd.run(f'docker exec -d ' + self.containerPrefix + '-oai-hss /bin/bash -c "nohup tshark -i eth0 -i eth1 -w /tmp/hss_check_run.pcap 2>&1 > /dev/null"')
			time.sleep(5)
			cmd.run(f'docker exec -d ' + self.containerPrefix + '-oai-hss /bin/bash -c "nohup ./bin/oai_hss -j ./etc/hss_rel14.json --reloadkey true > hss_check_run.log 2>&1"')
		elif re.match('OAI-Rel14-CUPS', self.Type, re.IGNORECASE):
			logging.debug('Using the OAI EPC Release 14 Cassandra-based HSS')
			cmd.cd(f'{self.SourceCodePath}/scripts')
			logging.debug('\u001B[1m Launching tshark on all interfaces \u001B[0m')
			self.PcapFileName = f'epc_{self.testCase_id}.pcap'
			cmd.run(f'sudo rm -f {self.PcapFileName}')
			cmd.run('echo $USER; nohup sudo tshark -f "tcp port not 22 and port not 53" -i any -w {self.SourceCodePath}/scripts/' + self.PcapFileName + ' > /tmp/tshark.log 2>&1 &', self.UserName, 5)
			cmd.run('sudo mkdir -p logs')
			cmd.run(f'sudo rm -f hss_{self.testCase_id}.log logs/hss*.*')
			cmd.run('echo "oai_hss -j /usr/local/etc/oai/hss_rel14.json" > ./my-hss.sh')
			cmd.run('chmod 755 ./my-hss.sh')
			cmd.run(f'sudo daemon --unsafe --name=hss_daemon --chdir={self.SourceCodePath}/scripts -o {self.SourceCodePath}/scripts/hss_{self.testCase_id}.log ./my-hss.sh')
		elif re.match('OAI', self.Type, re.IGNORECASE):
			logging.debug('Using the OAI EPC HSS')
			cmd.cd(f'{self.SourceCodePath}')
			cmd.run('source oaienv')
			cmd.cd('scripts')
			cmd.run('sudo ./run_hss 2>&1 | stdbuf -o0 awk \'{ print strftime("[%Y/%m/%d %H:%M:%S] ",systime()) $0 }\' | stdbuf -o0 tee -a hss_' + self.testCase_id + '.log &')
		elif re.match('ltebox', self.Type, re.IGNORECASE):
			logging.debug('Using the ltebox simulated HSS')
			cmd.run(f'if [ -d {self.SourceCodePath}/scripts ]; then sudo rm -Rf ' + self.SourceCodePath + '/scripts ; fi')
			cmd.run(f'mkdir -p {self.SourceCodePath}/scripts')
			result = re.search('hss_sim s6as diam_hss', cmd.getBefore())
			if result is not None:
				cmd.run('sudo killall hss_sim')
			cmd.run('ps aux | grep --colour=never xGw | grep -v grep')
			result = re.search('root.*xGw', cmd.getBefore())
			if result is not None:
				cmd.cd('/opt/ltebox/tools')
				cmd.run('sudo ./stop_ltebox')
			cmd.cd('/opt/hss_sim0609')
			cmd.run('sudo rm -f hss.log')
			cmd.run('sudo echo "Starting sudo session" && sudo su -c "screen -dm -S simulated_hss ./starthss"')
		else:
			logging.error('This option should not occur!')
		cmd.close()
		HTML.CreateHtmlTestRow(self.Type, 'OK', CONST.ALL_PROCESSES_OK)

	def InitializeMME(self, HTML):
		if self.IPAddress == '' or self.SourceCodePath == '' or self.Type == '':
			HELP.GenericHelp(CONST.Version)
			HELP.EPCSrvHelp(self.IPAddress,self.SourceCodePath, self.Type)
			sys.exit('Insufficient EPC Parameters')
		cmd = cls_cmd.getConnection(self.IPAddress)
		if re.match('OAI-Rel14-Docker', self.Type, re.IGNORECASE):
			logging.debug('Using the OAI EPC Release 14 MME in Docker')
			cmd.run(f'docker exec -d {self.containerPrefix}-oai-mme /bin/bash -c "nohup tshark -i eth0 -i lo:s10 -f "not port 2152" -w /tmp/mme_check_run.pcap 2>&1 > /dev/null"')
			time.sleep(5)
			cmd.run(f'docker exec -d {self.containerPrefix}-oai-mme /bin/bash -c "nohup ./bin/oai_mme -c ./etc/{self.mmeConfFile} > mme_check_run.log 2>&1"')
		elif re.match('OAI-Rel14-CUPS', self.Type, re.IGNORECASE):
			logging.debug('Using the OAI EPC Release 14 MME')
			cmd.cd(f'{self.SourceCodePath}/scripts')
			cmd.run(f'sudo rm -f mme_{self.testCase_id}.log')
			cmd.run('echo "./run_mme --config-file /usr/local/etc/oai/mme.conf --set-virt-if" > ./my-mme.sh')
			cmd.run('chmod 755 ./my-mme.sh')
			cmd.run(f'sudo daemon --unsafe --name=mme_daemon --chdir={self.SourceCodePath}/scripts -o {self.SourceCodePath}/scripts/mme_{self.testCase_id}.log ./my-mme.sh')
		elif re.match('OAI', self.Type, re.IGNORECASE):
			cmd.cd(self.SourceCodePath)
			cmd.run('source oaienv')
			cmd.cd('scripts')
			cmd.run('stdbuf -o0 hostname')
			result = re.search('hostname\\\\r\\\\n(?P<host_name>[a-zA-Z0-9\-\_]+)\\\\r\\\\n', cmd.getBefore())
			if result is None:
				logging.debug('\u001B[1;37;41m Hostname Not Found! \u001B[0m')
				sys.exit(1)
			host_name = result.group('host_name')
			cmd.run(f'sudo ./run_mme 2>&1 | stdbuf -o0 tee -a mme_{self.testCase_id}.log &')
		elif re.match('ltebox', self.Type, re.IGNORECASE):
			cmd.cd('/opt/ltebox/tools')
			# Clean-up the logs from previous runs
			cmd.run('sudo rm -f ../var/log/*.0')
			cmd.run('sudo ./start_mme')
		else:
			logging.error('This option should not occur!')
		cmd.close()
		HTML.CreateHtmlTestRow(self.Type, 'OK', CONST.ALL_PROCESSES_OK)

	def SetMmeIPAddress(self):
		# Not an error if we don't need an EPC
		if self.IPAddress == '' or self.SourceCodePath == '' or self.Type == '':
			return
		if self.IPAddress == 'none':
			return
		# Only in case of Docker containers, MME IP address is not the EPC HOST IP address
		if re.match('OAI-Rel14-Docker', self.Type, re.IGNORECASE):
			cmd = cls_cmd.getConnection(self.IPAddress)
			self.isMagmaUsed = False
			cmd.run('docker ps -a')
			result = re.search('magma', cmd.getBefore())
			if result is not None:
				self.isMagmaUsed = True
			if self.isMagmaUsed:
				cmd.run('docker inspect --format="MME_IP_ADDR = {{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}" ' + self.containerPrefix + '-magma-mme')
			else:
				cmd.run('docker inspect --format="MME_IP_ADDR = {{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}" ' + self.containerPrefix + '-oai-mme')
			result = re.search('MME_IP_ADDR = (?P<mme_ip_addr>[0-9\.]+)', cmd.getBefore())
			if result is not None:
				self.MmeIPAddress = result.group('mme_ip_addr')
				logging.debug('MME IP Address is ' + self.MmeIPAddress)
			cmd.close()
		else:
			self.MmeIPAddress = self.IPAddress

	def InitializeSPGW(self, HTML):
		if self.IPAddress == '' or self.SourceCodePath == '' or self.Type == '':
			HELP.GenericHelp(CONST.Version)
			HELP.EPCSrvHelp(self.IPAddress,self.SourceCodePath, self.Type)
			sys.exit('Insufficient EPC Parameters')
		cmd = cls_cmd.getConnection(self.IPAddress)
		if re.match('OAI-Rel14-Docker', self.Type, re.IGNORECASE):
			logging.debug('Using the OAI EPC Release 14 SPGW-CUPS in Docker')
			cmd.run(f'docker exec -d {self.containerPrefix}-oai-spgwc /bin/bash -c "nohup tshark -i eth0 -i lo:p5c -i lo:s5c -f "not port 2152" -w /tmp/spgwc_check_run.pcap 2>&1 > /dev/null"')
			cmd.run(f'docker exec -d {self.containerPrefix}-oai-spgwu-tiny /bin/bash -c "nohup tshark -i eth0 -f "not port 2152" -w /tmp/spgwu_check_run.pcap 2>&1 > /dev/null"')
			time.sleep(5)
			cmd.run(f'docker exec -d {self.containerPrefix}-oai-spgwc /bin/bash -c "nohup ./bin/oai_spgwc -o -c ./etc/spgw_c.conf > spgwc_check_run.log 2>&1"')
			time.sleep(5)
			cmd.run(f'docker exec -d {self.containerPrefix}-oai-spgwu-tiny /bin/bash -c "nohup ./bin/oai_spgwu -o -c ./etc/spgw_u.conf > spgwu_check_run.log 2>&1"')
		elif re.match('OAI-Rel14-CUPS', self.Type, re.IGNORECASE):
			logging.debug('Using the OAI EPC Release 14 SPGW-CUPS')
			cmd.cd(f'{self.SourceCodePath}/scripts')
			cmd.run(f'sudo rm -f spgwc_{self.testCase_id}.log spgwu_{self.testCase_id}.log')
			cmd.run('echo "spgwc -c /usr/local/etc/oai/spgw_c.conf" > ./my-spgwc.sh')
			cmd.run('chmod 755 ./my-spgwc.sh')
			cmd.run(f'sudo daemon --unsafe --name=spgwc_daemon --chdir={self.SourceCodePath}/scripts -o {self.SourceCodePath}/scripts/spgwc_{self.testCase_id}.log ./my-spgwc.sh')
			time.sleep(5)
			cmd.run('echo "spgwu -c /usr/local/etc/oai/spgw_u.conf" > ./my-spgwu.sh')
			cmd.run('chmod 755 ./my-spgwu.sh')
			cmd.run(f'sudo daemon --unsafe --name=spgwu_daemon --chdir={self.SourceCodePath}/scripts -o {self.SourceCodePath}/scripts/spgwu_{self.testCase_id}.log ./my-spgwu.sh')
		elif re.match('OAI', self.Type, re.IGNORECASE):
			cmd.cd(self.SourceCodePath)
			cmd.run('source oaienv')
			cmd.cd('scripts')
			cmd.run(f'sudo ./run_spgw 2>&1 | stdbuf -o0 tee -a spgw_{self.testCase_id}.log &')
		elif re.match('ltebox', self.Type, re.IGNORECASE):
			cmd.cd('/opt/ltebox/tools')
			cmd.run('sudo ./start_xGw')
		else:
			logging.error('This option should not occur!')
		cmd.close()
		HTML.CreateHtmlTestRow(self.Type, 'OK', CONST.ALL_PROCESSES_OK)

	def Initialize5GCN(self, HTML):
		if self.IPAddress == '' or self.Type == '':
			HELP.GenericHelp(CONST.Version)
			HELP.EPCSrvHelp(self.IPAddress,self.Type)
			sys.exit('Insufficient EPC Parameters')
		cmd = cls_cmd.getConnection(self.IPAddress)
		html_cell = ''
		if re.match('ltebox', self.Type, re.IGNORECASE):
			logging.debug('Using the SABOX simulated HSS')
			cmd.run(f'if [ -d {self.SourceCodePath}/scripts ]; then sudo rm -Rf {self.SourceCodePath}/scripts ; fi')
			cmd.run(f'mkdir -p {self.SourceCodePath}/scripts')
			cmd.cd('/opt/hss_sim0609')
			cmd.run('sudo rm -f hss.log')
			cmd.run('sudo echo "Starting sudo session" && sudo su -c "screen -dm -S simulated_5g_hss ./start_5g_hss"')
			logging.debug('Using the sabox')
			cmd.cd('/opt/ltebox/tools')
			logging.debug(cmd.run('sudo ./start_sabox', timeout=5))
			html_cell += 'N/A\n'
		elif re.match('OAICN5G', self.Type, re.IGNORECASE):
			logging.debug('Starting OAI CN5G')
			cmd.run(f'if [ -d {self.SourceCodePath}/scripts ]; then sudo rm -Rf {self.SourceCodePath}/scripts ; fi')
			cmd.run(f'mkdir -p {self.SourceCodePath}/scripts')
			cmd.cd('/opt/oai-cn5g-fed-v1.5/docker-compose')
			cmd.run('python3 ./core-network.py '+self.cfgDeploy)
			if re.search('start-mini-as-ue', self.cfgDeploy):
				dFile = 'docker-compose-mini-nrf-asue.yaml'
			else:
				dFile = 'docker-compose-mini-nrf.yaml'
			cmd.run(f'docker-compose -f {dFile} ps -a')
			if cmd.getBefore().count('Up (healthy)') != 6:
				logging.error('Not all container healthy')
			else:
				logging.debug('OK --> all containers are healthy')
			cmd.run(f'docker-compose -f {dFile} config | grep --colour=never image')
			listOfImages = cmd.getBefore()
			for imageLine in listOfImages.split('\\r\\n'):
				res1 = re.search('image: (?P<name>[a-zA-Z0-9\-/]+):(?P<tag>[a-zA-Z0-9\-]+)', str(imageLine))
				res2 = re.search('mysql', str(imageLine))
				if res1 is not None and res2 is None:
					html_cell += res1.group('name') + ':' + res1.group('tag') + ' '
					nbChars = len(res1.group('name')) + len(res1.group('tag')) + 2
					while (nbChars < 32):
						html_cell += ' '
						nbChars += 1
					cmd.run('docker image inspect --format="Size = {{.Size}} bytes" ' + res1.group('name') + ':' + res1.group('tag'))
					res3 = re.search('Size *= *(?P<size>[0-9\-]*) *bytes', cmd.getBefore())
					if res3 is not None:
						imageSize = int(res3.group('size'))
						imageSize = int(imageSize/(1024*1024))
						html_cell += str(imageSize) + ' MBytes '
					cmd.run('docker image inspect --format="Date = {{.Created}}" ' + res1.group('name') + ':' + res1.group('tag'))
					res4 = re.search('Date *= *(?P<date>[0-9\-]*)T', cmd.getBefore())
					if res4 is not None:
						html_cell += '(' + res4.group('date') + ')'
					html_cell += '\n'
		elif re.match('OC-OAI-CN5G', self.Type, re.IGNORECASE):
			self.testCase_id = HTML.testCase_id
			imageNames = ["oai-nrf", "oai-amf", "oai-smf", "oai-spgwu-tiny", "oai-ausf", "oai-udm", "oai-udr", "mysql","oai-traffic-server"]
			logging.debug('Deploying OAI CN5G on Openshift Cluster')
			lIpAddr = self.IPAddress
			lSourcePath = "/opt/oai-cn5g-fed-develop-2023-04-28-20897"
			succeeded = OC.OC_login(cmd, self.OCUserName, self.OCPassword, OC.CI_OC_CORE_NAMESPACE)
			if not succeeded:
				logging.error('\u001B[1m OC Cluster Login Failed\u001B[0m')
				HTML.CreateHtmlTestRow('N/A', 'KO', CONST.OC_LOGIN_FAIL)
				return False
			for ii in imageNames:
					cmd.run(f'helm uninstall {ii}', reportNonZero=False)
			cmd.run(f'helm spray {lSourcePath}/ci-scripts/charts/oai-5g-basic/.')
			ret = cmd.run(f'oc get pods', silent=True)
			if ret.stdout.count('Running') != 9:
				logging.error('\u001B[1m Deploying 5GCN Failed using helm chart on OC Cluster\u001B[0m')
				for ii in imageNames:
					cmd.run('helm uninstall '+ ii)
				ret = cmd.run(f'oc get pods')
				if re.search('No resources found', ret.stdout):
					logging.debug('All pods uninstalled')
					OC.OC_logout(cmd)
					cmd.close()
					HTML.CreateHtmlTestRow('N/A', 'KO', CONST.OC_PROJECT_FAIL)
					return False
			ret = cmd.run(f'oc get pods', silent=True)
			for line in ret.stdout.split('\n')[1:]:
				columns = line.strip().split()
				name = columns[0]
				status = columns[2]
				html_cell += status + '    ' + name
				html_cell += '\n'
			OC.OC_logout(cmd)
		else:
			logging.error('This option should not occur!')
		cmd.close()
		HTML.CreateHtmlTestRowQueue(self.Type, 'OK', [html_cell])

	def SetAmfIPAddress(self):
		# Not an error if we don't need an 5GCN
		if self.IPAddress == '' or self.SourceCodePath == '' or self.Type == '':
			return
		if self.IPAddress == 'none':
			return
		if re.match('ltebox', self.Type, re.IGNORECASE):
			self.MmeIPAddress = self.IPAddress
		elif re.match('OAICN5G', self.Type, re.IGNORECASE):
			cmd = cls_cmd.getConnection(self.IPAddress)
			response=cmd.run('docker container ls -f name=oai-amf', 10)
			if len(response)>1:
				response=cmd.run('docker inspect --format=\'{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}\' oai-amf', 10)
				tmp = str(response[0],'utf-8')
				self.MmeIPAddress = tmp.rstrip()
				logging.debug('AMF IP Address ' + self.MmeIPAddress)
			else:
				logging.error('no container with name oai-amf found, could not retrieve AMF IP address')
			cmd.close()
		elif re.match('OC-OAI-CN5G', self.Type, re.IGNORECASE):
			response = '172.21.6.100'

	def CheckHSSProcess(self, status_queue):
		try:
			cmd = cls_cmd.getConnection(self.IPAddress)
			if re.match('OAI-Rel14-Docker', self.Type, re.IGNORECASE):
				cmd.run(f'docker top {self.containerPrefix}-oai-hss')
			else:
				cmd.run('stdbuf -o0 ps -aux | grep --color=never hss | grep -v grep')
			if re.match('OAI-Rel14-CUPS', self.Type, re.IGNORECASE) or re.match('OAI-Rel14-Docker', self.Type, re.IGNORECASE):
				result = re.search('oai_hss -j', cmd.getBefore())
			elif re.match('OAI', self.Type, re.IGNORECASE):
				result = re.search('\/bin\/bash .\/run_', cmd.getBefore())
			elif re.match('ltebox', self.Type, re.IGNORECASE):
				result = re.search('hss_sim s6as diam_hss', cmd.getBefore())
			else:
				logging.error('This should not happen!')
			if result is None:
				logging.debug('\u001B[1;37;41m HSS Process Not Found! \u001B[0m')
				status_queue.put(CONST.HSS_PROCESS_FAILED)
			else:
				status_queue.put(CONST.HSS_PROCESS_OK)
			cmd.close()
		except:
			os.kill(os.getppid(),signal.SIGUSR1)

	def CheckMMEProcess(self, status_queue):
		try:
			cmd = cls_cmd.getConnection(self.IPAddress)
			if re.match('OAI-Rel14-Docker', self.Type, re.IGNORECASE):
				self.isMagmaUsed = False
				cmd.run('docker ps -a')
				result = re.search('magma', cmd.getBefore())
				if result is not None:
					self.isMagmaUsed = True
				if self.isMagmaUsed:
					cmd.run(f'docker top {self.containerPrefix}-magma-mme')
				else:
					cmd.run(f'docker top {self.containerPrefix}-oai-mme')
			else:
				cmd.run('stdbuf -o0 ps -aux | grep --color=never mme | grep -v grep')
			if re.match('OAI-Rel14-Docker', self.Type, re.IGNORECASE):
				result = re.search('oai_mme -c ', cmd.getBefore())
			elif re.match('OAI-Rel14-CUPS', self.Type, re.IGNORECASE):
				result = re.search('mme -c', cmd.getBefore())
			elif re.match('OAI', self.Type, re.IGNORECASE):
				result = re.search('\/bin\/bash .\/run_', cmd.getBefore())
			elif re.match('ltebox', self.Type, re.IGNORECASE):
				result = re.search('mme|amf', cmd.getBefore())
			else:
				logging.error('This should not happen!')
			if result is None:
				logging.debug('\u001B[1;37;41m MME|AMF Process Not Found! \u001B[0m')
				status_queue.put(CONST.MME_PROCESS_FAILED)
			else:
				status_queue.put(CONST.MME_PROCESS_OK)
			cmd.close()
		except:
			os.kill(os.getppid(),signal.SIGUSR1)

	def CheckSPGWProcess(self, status_queue):
		try:
			cmd = cls_cmd.getConnection(self.IPAddress)
			if re.match('OAI-Rel14-Docker', self.Type, re.IGNORECASE):
				cmd.run(f'docker top {self.containerPrefix}-oai-spgwc')
				result = re.search('oai_spgwc -', cmd.getBefore())
				if result is not None:
					cmd.run(f'docker top {self.containerPrefix}-oai-spgwu-tiny')
					result = re.search('oai_spgwu -', cmd.getBefore())
			elif re.match('OAI-Rel14-CUPS', self.Type, re.IGNORECASE):
				cmd.run('stdbuf -o0 ps -aux | grep --color=never spgw | grep -v grep')
				result = re.search('spgwu -c ', cmd.getBefore())
			elif re.match('OAI', self.Type, re.IGNORECASE):
				cmd.run('stdbuf -o0 ps -aux | grep --color=never spgw | grep -v grep')
				result = re.search('\/bin\/bash .\/run_', cmd.getBefore())
			elif re.match('ltebox', self.Type, re.IGNORECASE):
				cmd.run('stdbuf -o0 ps -aux | grep --color=never xGw | grep -v grep')
				result = re.search('xGw|upf', cmd.getBefore())
			else:
				logging.error('This should not happen!')
			if result is None:
				logging.debug('\u001B[1;37;41m SPGW|UPF Process Not Found! \u001B[0m')
				status_queue.put(CONST.SPGW_PROCESS_FAILED)
			else:
				status_queue.put(CONST.SPGW_PROCESS_OK)
			cmd.close()
		except:
			os.kill(os.getppid(),signal.SIGUSR1)

	def TerminateHSS(self, HTML):
		cmd = cls_cmd.getConnection(self.IPAddress)
		if re.match('OAI-Rel14-Docker', self.Type, re.IGNORECASE):
			cmd.run(f'docker exec -it {self.containerPrefix}-oai-hss /bin/bash -c "killall --signal SIGINT oai_hss tshark"')
			time.sleep(2)
			cmd.run(f'docker exec -it {self.containerPrefix}-oai-hss /bin/bash -c "ps aux | grep oai_hss"')
			result = re.search('oai_hss -j ', cmd.getBefore())
			if result is not None:
				cmd.run(f'docker exec -it {self.containerPrefix}-oai-hss /bin/bash -c "killall --signal SIGKILL oai_hss"')
		elif re.match('OAI-Rel14-CUPS', self.Type, re.IGNORECASE):
			cmd.run('sudo killall --signal SIGINT oai_hss || true')
			time.sleep(2)
			cmd.run('stdbuf -o0  ps -aux | grep --colour=never hss | grep -v grep')
			result = re.search('oai_hss -j', cmd.getBefore())
			if result is not None:
				cmd.run('sudo killall --signal SIGKILL oai_hss || true')
			cmd.run(f'rm -f {self.SourceCodePath}/scripts/my-hss.sh')
		elif re.match('OAI', self.Type, re.IGNORECASE):
			cmd.run('sudo killall --signal SIGINT run_hss oai_hss || true')
			time.sleep(2)
			cmd.run('stdbuf -o0  ps -aux | grep --colour=never hss | grep -v grep')
			result = re.search('\/bin\/bash .\/run_', cmd.getBefore())
			if result is not None:
				cmd.run('sudo killall --signal SIGKILL run_hss oai_hss || true')
		elif re.match('ltebox', self.Type, re.IGNORECASE):
			cmd.cd(f'{self.SourceCodePath}/scripts')
			time.sleep(1)
			cmd.run('sudo screen -S simulated_hss -X quit')
			time.sleep(5)
			cmd.run('ps aux | grep --colour=never hss_sim | grep -v grep')
			result = re.search('hss_sim s6as diam_hss', cmd.getBefore())
			if result is not None:
				cmd.run('sudo killall hss_sim')
		else:
			logging.error('This should not happen!')
		cmd.close()
		HTML.CreateHtmlTestRow('N/A', 'OK', CONST.ALL_PROCESSES_OK)

	def TerminateMME(self, HTML):
		cmd = cls_cmd.getConnection(self.IPAddress)
		if re.match('OAI-Rel14-Docker', self.Type, re.IGNORECASE):
			cmd.run(f'docker exec -it {self.containerPrefix}-oai-mme /bin/bash -c "killall --signal SIGINT oai_mme tshark"')
			time.sleep(2)
			cmd.run(f'docker exec -it {self.containerPrefix}-oai-mme /bin/bash -c "ps aux | grep oai_mme"')
			result = re.search('oai_mme -c ', cmd.getBefore())
			if result is not None:
				cmd.run(f'docker exec -it {self.containerPrefix}-oai-mme /bin/bash -c "killall --signal SIGKILL oai_mme"')
		elif re.match('OAI', self.Type, re.IGNORECASE) or re.match('OAI-Rel14-CUPS', self.Type, re.IGNORECASE):
			cmd.run('sudo killall --signal SIGINT run_mme mme || true')
			time.sleep(2)
			cmd.run('stdbuf -o0 ps -aux | grep mme | grep -v grep')
			result = re.search('mme -c', cmd.getBefore())
			if result is not None:
				cmd.run('sudo killall --signal SIGKILL run_mme mme || true')
			cmd.run(f'rm -f {self.SourceCodePath}/scripts/my-mme.sh')
		elif re.match('ltebox', self.Type, re.IGNORECASE):
			cmd.cd('/opt/ltebox/tools')
			cmd.run('sudo ./stop_mme')
			time.sleep(5)
		else:
			logging.error('This should not happen!')
		cmd.close()
		HTML.CreateHtmlTestRow('N/A', 'OK', CONST.ALL_PROCESSES_OK)

	def TerminateSPGW(self, HTML):
		cmd = cls_cmd.getConnection(self.IPAddress)
		if re.match('OAI-Rel14-Docker', self.Type, re.IGNORECASE):
			cmd.run(f'docker exec -it {self.containerPrefix}-oai-spgwc /bin/bash -c "killall --signal SIGINT oai_spgwc tshark"')
			cmd.run(f'docker exec -it {self.containerPrefix}-oai-spgwu-tiny /bin/bash -c "killall --signal SIGINT oai_spgwu tshark"')
			time.sleep(2)
			cmd.run(f'docker exec -it {self.containerPrefix}-oai-spgwc /bin/bash -c "ps aux | grep oai_spgwc"')
			result = re.search('oai_spgwc -o -c ', cmd.getBefore())
			if result is not None:
				cmd.run(f'docker exec -it {self.containerPrefix}-oai-spgwc /bin/bash -c "killall --signal SIGKILL oai_spgwc"')
			cmd.run(f'docker exec -it {self.containerPrefix}-oai-spgwu-tiny /bin/bash -c "ps aux | grep oai_spgwu"')
			result = re.search('oai_spgwu -o -c ', cmd.getBefore())
			if result is not None:
				cmd.run(f'docker exec -it {self.containerPrefix}-oai-spgwu-tiny /bin/bash -c "killall --signal SIGKILL oai_spgwu"')
		elif re.match('OAI-Rel14-CUPS', self.Type, re.IGNORECASE):
			cmd.run('sudo killall --signal SIGINT spgwc spgwu || true')
			time.sleep(2)
			cmd.run('stdbuf -o0 ps -aux | grep spgw | grep -v grep')
			result = re.search('spgwc -c |spgwu -c ', cmd.getBefore())
			if result is not None:
				cmd.run('sudo killall --signal SIGKILL spgwc spgwu || true')
			cmd.run(f'rm -f {self.SourceCodePath}/scripts/my-spgw*.sh')
			cmd.run('stdbuf -o0 ps -aux | grep tshark | grep -v grep')
			result = re.search('-w ', cmd.getBefore())
			if result is not None:
				cmd.run('sudo killall --signal SIGINT tshark || true')
				cmd.run(f'sudo chmod 666 {self.SourceCodePath}/scripts/*.pcap')
		elif re.match('OAI', self.Type, re.IGNORECASE):
			cmd.run('sudo killall --signal SIGINT run_spgw spgw || true')
			time.sleep(2)
			cmd.run('stdbuf -o0 ps -aux | grep spgw | grep -v grep')
			result = re.search('\/bin\/bash .\/run_', cmd.getBefore())
			if result is not None:
				cmd.run('sudo killall --signal SIGKILL run_spgw spgw || true')
		elif re.match('ltebox', self.Type, re.IGNORECASE):
			cmd.cd('/opt/ltebox/tools')
			cmd.run('sudo ./stop_xGw')
		else:
			logging.error('This should not happen!')
		cmd.close()
		HTML.CreateHtmlTestRow('N/A', 'OK', CONST.ALL_PROCESSES_OK)

	def Terminate5GCN(self, HTML):
		imageNames = ["mysql", "oai-nrf", "oai-amf", "oai-smf", "oai-spgwu-tiny", "oai-ausf", "oai-udm", "oai-udr", "oai-traffic-server"]
		containerInPods = ["", "-c nrf", "-c amf", "-c smf", "-c spgwu", "-c ausf", "-c udm", "-c udr", ""]
		cmd = cls_cmd.getConnection(self.IPAddress)
		message = ''
		if re.match('ltebox', self.Type, re.IGNORECASE):
			logging.debug('Terminating SA BOX')
			cmd.cd('/opt/ltebox/tools')
			cmd.run('sudo ./stop_sabox')
			time.sleep(1)
			cmd.cd(f'{self.SourceCodePath}/scripts')
			time.sleep(1)
			cmd.run('sudo screen -S simulated_5g_hss -X quit')
		elif re.match('OAICN5G', self.Type, re.IGNORECASE):
			logging.debug('OAI CN5G Collecting Log files to workspace')
			cmd.run(f'sudo rm -rf {self.SourceCodePath}/logs')
			cmd.run(f'mkdir {self.SourceCodePath}/logs')
			containers_list=['oai-smf','oai-spgwu','oai-amf','oai-nrf']
			for c in containers_list:
				cmd.run(f'docker logs {c} > {self.SourceCodePath}/logs/{c}.log')

			logging.debug('Terminating OAI CN5G')
			cmd.cd('/opt/oai-cn5g-fed-v1.5/docker-compose')
			cmd.run(f'python3 ./core-network.py {self.cfgUnDeploy}')
			cmd.run('docker volume prune --force || true')
			time.sleep(2)
			cmd.run('tshark -r /tmp/oai-cn5g-v1.5.pcap | egrep --colour=never "Tracking area update" ')
			result = re.search('Tracking area update request', cmd.getBefore())
			if result is not None:
				message = 'UE requested ' + str(cmd.getBefore().count('Tracking area update request')) + 'Tracking area update request(s)'
			else:
				message = 'No Tracking area update request'
			logging.debug(message)
		elif re.match('OC-OAI-CN5G', self.Type, re.IGNORECASE):
			logging.debug('Terminating OAI CN5G on Openshift Cluster')
			lIpAddr = self.IPAddress
			lSourcePath = self.SourceCodePath
			cmd.run(f'rm -Rf {lSourcePath}/logs')
			cmd.run(f'mkdir -p {lSourcePath}/logs')
			logging.debug('OC OAI CN5G - Collecting Log files to workspace')
			succeeded = OC.OC_login(cmd, self.OCUserName, self.OCPassword, OC.CI_OC_CORE_NAMESPACE)
			if not succeeded:
				logging.error('\u001B[1m OC Cluster Login Failed\u001B[0m')
				HTML.CreateHtmlTestRow('N/A', 'KO', CONST.OC_LOGIN_FAIL)
				return False
			cmd.run(f'oc describe pod &> {lSourcePath}/logs/describe-pods-post-test.log')
			cmd.run(f'oc get pods.metrics.k8s &> {lSourcePath}/logs/nf-resource-consumption.log')
			for ii, ci in zip(imageNames, containerInPods):
			       podName = cmd.run(f"oc get pods | grep {ii} | awk \'{{print $1}}\'").stdout.strip()
			       if not podName:
				       logging.debug(f'{ii} pod not found!')
				       HTML.CreateHtmlTestRow(self.Type, 'KO', CONST.INVALID_PARAMETER)
				       HTML.CreateHtmlTabFooter(False)
			       cmd.run(f'oc logs -f {podName} {ci} &> {lSourcePath}/logs/{ii}.log &')
			       cmd.run(f'helm uninstall {ii}')
			       podName = ''
			cmd.run(f'cd {lSourcePath}/logs && zip -r -qq test_logs_CN.zip *.log')
			cmd.copyin(f'{lSourcePath}/logs/test_logs_CN.zip','test_logs_CN.zip')
			ret = cmd.run(f'oc get pods', silent=True)
			res = re.search(f'No resources found in {OC.CI_OC_CORE_NAMESPACE} namespace.', ret.stdout)
			if res is not None:
			       logging.debug('OC OAI CN5G components uninstalled')
			       message = 'OC OAI CN5G components uninstalled'
			OC.OC_logout(cmd)
		else:
			logging.error('This should not happen!')
		cmd.close()
		HTML.CreateHtmlTestRowQueue(self.Type, 'OK', [message])

	def DeployEpc(self, HTML):
		logging.debug('Trying to deploy')
		if not re.match('OAI-Rel14-Docker', self.Type, re.IGNORECASE):
			HTML.CreateHtmlTestRow(self.Type, 'KO', CONST.INVALID_PARAMETER)
			HTML.CreateHtmlTabFooter(False)
			sys.exit('Deploy not possible with this EPC type: ' + self.Type)

		if self.IPAddress == '' or self.SourceCodePath == '' or self.Type == '':
			HELP.GenericHelp(CONST.Version)
			HELP.EPCSrvHelp(self.IPAddress,self.SourceCodePath, self.Type)
			sys.exit('Insufficient EPC Parameters')
		cmd = cls_cmd.getConnection(self.IPAddress)
		cmd.run('docker-compose --version')
		result = re.search('docker-compose version 1', cmd.getBefore())
		if result is None:
			cmd.close()
			HTML.CreateHtmlTestRow(self.Type, 'KO', CONST.INVALID_PARAMETER)
			HTML.CreateHtmlTabFooter(False)
			sys.exit('docker-compose not installed on ' + self.IPAddress)

		# Checking if it is a MAGMA deployment
		self.isMagmaUsed = False
		if os.path.isfile('./' + self.yamlPath + '/redis_extern.conf'):
			self.isMagmaUsed = True
			logging.debug('MAGMA MME is used!')

		cmd.run(f'if [ -d {self.SourceCodePath}/scripts ]; then sudo rm -Rf {self.SourceCodePath}/scripts ; fi')
		cmd.run(f'if [ -d {self.SourceCodePath}/logs ]; then sudo rm -Rf {self.SourceCodePath}/logs ; fi')
		cmd.run(f'mkdir -p {self.SourceCodePath}/scripts {self.SourceCodePath}/logs')

		# deploying and configuring the cassandra database
		# container names and services are currently hard-coded.
		# they could be recovered by:
		# - docker-compose config --services
		# - docker-compose config | grep container_name
		cmd.cd(f'{self.SourceCodePath}/scripts')
		cmd.copyout(f'./{self.yamlPath}/docker-compose.yml', f'{self.SourceCodePath}/scripts/docker-compose.yml')
		if self.isMagmaUsed:
			cmd.copyout(f'./{self.yamlPath}/entrypoint.sh', f'{self.SourceCodePath}/scripts/entrypoint.sh')
			cmd.copyout(f'./{self.yamlPath}/mme.conf', f'{self.SourceCodePath}/scripts/mme.conf')
			cmd.copyout(f'./{self.yamlPath}/mme_fd.sprint.conf', f'{self.SourceCodePath}/scripts/mme_fd.sprint.conf')
			cmd.copyout(f'./{self.yamlPath}/redis_extern.conf', f'{self.SourceCodePath}/scripts/redis_extern.conf')
			cmd.run(f'chmod a+x {self.SourceCodePath}/scripts/entrypoint.sh')
		else:
			cmd.copyout(f'./{self.yamlPath}/entrypoint.sh', f'{self.SourceCodePath}/scripts/entrypoint.sh')
			cmd.copyout(f'./{self.yamlPath}/mme.conf', f'{self.SourceCodePath}/scripts/mme.conf')
			cmd.run('chmod 775 entrypoint.sh')
		cmd.run('wget --quiet --tries=3 --retry-connrefused https://raw.githubusercontent.com/OPENAIRINTERFACE/openair-hss/develop/src/hss_rel14/db/oai_db.cql')
		cmd.run('docker-compose down')
		cmd.run('docker-compose up -d db_init')
		# databases take time...
		time.sleep(10)
		cnt = 0
		db_init_status = False
		while (cnt < 10):
			cmd.run('docker logs prod-db-init')
			result = re.search('OK', cmd.getBefore())
			if result is not None:
				cnt = 10
				db_init_status = True
			else:
				time.sleep(5)
				cnt += 1
		cmd.run('docker rm -f prod-db-init')
		if not db_init_status:
			HTML.CreateHtmlTestRow(self.Type, 'KO', CONST.INVALID_PARAMETER)
			HTML.CreateHtmlTabFooter(False)
			sys.exit('Cassandra DB deployment/configuration went wrong!')

		# deploying EPC cNFs
		cmd.run('docker-compose up -d oai_spgwu')
		if self.isMagmaUsed:
			listOfContainers = 'prod-cassandra prod-oai-hss prod-magma-mme prod-oai-spgwc prod-oai-spgwu-tiny prod-redis'
			expectedHealthyContainers = 6
		else:
			listOfContainers = 'prod-cassandra prod-oai-hss prod-oai-mme prod-oai-spgwc prod-oai-spgwu-tiny'
			expectedHealthyContainers = 5

		# Checking for additional services
		cmd.run('docker-compose config')
		configResponse = cmd.getBefore()
		if configResponse.count('trf_gen') == 1:
			cmd.run('docker-compose up -d trf_gen')
			listOfContainers += ' prod-trf-gen'
			expectedHealthyContainers += 1

		cmd.run('docker-compose config | grep --colour=never image')
		html_cell = ''
		listOfImages = cmd.getBefore()
		for imageLine in listOfImages.split('\\r\\n'):
			res1 = re.search('image: (?P<name>[a-zA-Z0-9\-]+):(?P<tag>[a-zA-Z0-9\-]+)', str(imageLine))
			res2 = re.search('cassandra|redis', str(imageLine))
			if res1 is not None and res2 is None:
				html_cell += res1.group('name') + ':' + res1.group('tag') + ' '
				nbChars = len(res1.group('name')) + len(res1.group('tag')) + 2
				while (nbChars < 32):
					html_cell += ' '
					nbChars += 1
				cmd.run('docker image inspect --format="Size = {{.Size}} bytes" ' + res1.group('name') + ':' + res1.group('tag'))
				res3 = re.search('Size *= *(?P<size>[0-9\-]*) *bytes', cmd.getBefore())
				if res3 is not None:
					imageSize = int(res3.group('size'))
					imageSize = int(imageSize/(1024*1024))
					html_cell += str(imageSize) + ' MBytes '
				cmd.run('docker image inspect --format="Date = {{.Created}}" ' + res1.group('name') + ':' + res1.group('tag'))
				res4 = re.search('Date *= *(?P<date>[0-9\-]*)T', cmd.getBefore())
				if res4 is not None:
					html_cell += '(' + res4.group('date') + ')'
				html_cell += '\n'
		# Checking if all are healthy
		cnt = 0
		while (cnt < 3):
			cmd.run('docker inspect --format=\'{{.State.Health.Status}}\' ' + listOfContainers)
			unhealthyNb = cmd.getBefore().count('unhealthy')
			healthyNb = cmd.getBefore().count('healthy') - unhealthyNb
			startingNb = cmd.getBefore().count('starting')
			if healthyNb == expectedHealthyContainers:
				cnt = 10
			else:
				time.sleep(10)
				cnt += 1
		logging.debug(' -- ' + str(healthyNb) + ' healthy container(s)')
		logging.debug(' -- ' + str(unhealthyNb) + ' unhealthy container(s)')
		logging.debug(' -- ' + str(startingNb) + ' still starting container(s)')
		if healthyNb == expectedHealthyContainers:
			cmd.run('docker exec -d prod-oai-hss /bin/bash -c "nohup tshark -i any -f \'port 9042 or port 3868\' -w /tmp/hss_check_run.pcap 2>&1 > /dev/null"')
			if self.isMagmaUsed:
				cmd.run('docker exec -d prod-magma-mme /bin/bash -c "nohup tshark -i any -f \'port 3868 or port 2123 or port 36412\' -w /tmp/mme_check_run.pcap 2>&1 > /dev/null"')
			else:
				cmd.run('docker exec -d prod-oai-mme /bin/bash -c "nohup tshark -i any -f \'port 3868 or port 2123 or port 36412\' -w /tmp/mme_check_run.pcap 2>&1 > /dev/null"')
			cmd.run('docker exec -d prod-oai-spgwc /bin/bash -c "nohup tshark -i any -f \'port 2123 or port 8805\' -w /tmp/spgwc_check_run.pcap 2>&1 > /dev/null"')
			# on SPGW-U, not capturing on SGI to avoid huge file
			cmd.run('docker exec -d prod-oai-spgwu-tiny /bin/bash -c "nohup tshark -i any -f \'port 8805\'  -w /tmp/spgwu_check_run.pcap 2>&1 > /dev/null"')
			cmd.close()
			logging.debug('Deployment OK')
			HTML.CreateHtmlTestRowQueue(self.Type, 'OK', [html_cell])
		else:
			cmd.close()
			logging.debug('Deployment went wrong')
			HTML.CreateHtmlTestRowQueue(self.Type, 'KO', [html_cell])

	def UndeployEpc(self, HTML):
		logging.debug('Trying to undeploy')
		# No check down, we suppose everything done before.
		cmd = cls_cmd.getConnection(self.IPAddress)
		# Checking if it is a MAGMA deployment.
		cmd.cd(f'{self.SourceCodePath}/scripts')
		cmd.run('docker-compose ps -a')
		self.isMagmaUsed = False
		result = re.search('magma', cmd.getBefore())
		if result is not None:
			self.isMagmaUsed = True
			logging.debug('MAGMA MME is used!')
		# Recovering logs and pcap files
		cmd.cd(f'{self.SourceCodePath}/logs')
		cmd.run('docker exec -it prod-oai-hss /bin/bash -c "killall --signal SIGINT oai_hss tshark"')
		if self.isMagmaUsed:
			cmd.run('docker exec -it prod-magma-mme /bin/bash -c "killall --signal SIGINT tshark"')
		else:
			cmd.run('docker exec -it prod-oai-mme /bin/bash -c "killall --signal SIGINT tshark"')
		cmd.run('docker exec -it prod-oai-spgwc /bin/bash -c "killall --signal SIGINT oai_spgwc tshark"')
		cmd.run('docker exec -it prod-oai-spgwu-tiny /bin/bash -c "killall --signal SIGINT tshark"')
		cmd.run(f'docker logs prod-oai-hss > hss_{self.testCase_id}.log')
		if self.isMagmaUsed:
			cmd.run(f'docker cp --follow-link prod-magma-mme:/var/log/mme.log mme_{self.testCase_id}.log')
		else:
			cmd.run(f'docker logs prod-oai-mme > mme_{self.testCase_id}.log')
		cmd.run(f'docker logs prod-oai-spgwc > spgwc_{self.testCase_id}.log')
		cmd.run(f'docker logs prod-oai-spgwu-tiny > spgwu_{self.testCase_id}.log')
		cmd.run(f'docker cp prod-oai-hss:/tmp/hss_check_run.pcap hss_{self.testCase_id}.pcap')
		if self.isMagmaUsed:
			cmd.run(f'docker cp prod-magma-mme:/tmp/mme_check_run.pcap mme_{self.testCase_id}.pcap')
		else:
			cmd.run(f'docker cp prod-oai-mme:/tmp/mme_check_run.pcap mme_{self.testCase_id}.pcap')
		cmd.run(f'tshark -r mme_{self.testCase_id}.pcap | egrep --colour=never "Tracking area update"')
		result = re.search('Tracking area update request', cmd.getBefore())
		if result is not None:
			message = 'UE requested ' + str(cmd.getBefore().count('Tracking area update request')) + 'Tracking area update request(s)'
		else:
			message = 'No Tracking area update request'
		logging.debug(message)
		cmd.run(f'docker cp prod-oai-spgwc:/tmp/spgwc_check_run.pcap spgwc_{self.testCase_id}.pcap')
		cmd.run(f'docker cp prod-oai-spgwu-tiny:/tmp/spgwu_check_run.pcap spgwu_{self.testCase_id}.pcap')
		# Remove all
		cmd.cd(f'{self.SourceCodePath}/scripts')
		if self.isMagmaUsed:
			listOfContainers = 'prod-cassandra prod-oai-hss prod-magma-mme prod-oai-spgwc prod-oai-spgwu-tiny prod-redis'
			nbContainers = 6
		else:
			listOfContainers = 'prod-cassandra prod-oai-hss prod-oai-mme prod-oai-spgwc prod-oai-spgwu-tiny'
			nbContainers = 5
		# Checking for additional services
		cmd.run('docker-compose config')
		configResponse = cmd.getBefore()
		if configResponse.count('trf_gen') == 1:
			listOfContainers += ' prod-trf-gen'
			nbContainers += 1

		cmd.run('docker-compose down')
		cmd.run('docker volume prune --force || true')
		cmd.run('docker inspect --format=\'{{.State.Health.Status}}\' ' + listOfContainers)
		noMoreContainerNb = cmd.getBefore().count('No such object')
		cmd.run('docker inspect --format=\'{{.Name}}\' prod-oai-public-net prod-oai-private-net')
		noMoreNetworkNb = cmd.getBefore().count('No such object')
		cmd.close()
		if noMoreContainerNb == nbContainers and noMoreNetworkNb == 2:
			logging.debug('Undeployment OK')
			HTML.CreateHtmlTestRowQueue(self.Type, 'OK', [message])
		else:
			logging.debug('Undeployment went wrong')
			HTML.CreateHtmlTestRowQueue(self.Type, 'KO', [message])

	def LogCollectHSS(self):
		cmd = cls_cmd.getConnection(self.IPAddress)
		cmd.cd(f'{self.SourceCodePath}/scripts')
		cmd.run('rm -f hss.log.zip')
		if re.match('OAI-Rel14-Docker', self.Type, re.IGNORECASE):
			cmd.run('docker inspect prod-oai-hss')
			result = re.search('No such object', cmd.getBefore())
			if result is not None:
				cmd.cd('../logs')
				cmd.run('rm -f hss.log.zip')
				cmd.run('zip hss.log.zip hss_*.*')
				cmd.run('mv hss.log.zip ../scripts')
			else:
				cmd.run(f'docker cp {self.containerPrefix}-oai-hss:/openair-hss/hss_check_run.log .')
				cmd.run(f'docker cp {self.containerPrefix}-oai-hss:/tmp/hss_check_run.pcap .')
				cmd.run('zip hss.log.zip hss_check_run.*')
		elif re.match('OAICN5G', self.Type, re.IGNORECASE):
			logging.debug('LogCollect is bypassed for that variant')
		elif re.match('OC-OAI-CN5G', self.Type, re.IGNORECASE):
			logging.debug('LogCollect is bypassed for that variant')
		elif re.match('OAI', self.Type, re.IGNORECASE) or re.match('OAI-Rel14-CUPS', self.Type, re.IGNORECASE):
			cmd.run('zip hss.log.zip hss*.log')
			cmd.run('sudo rm hss*.log')
			if re.match('OAI-Rel14-CUPS', self.Type, re.IGNORECASE):
				cmd.run('zip hss.log.zip logs/hss*.* *.pcap')
				cmd.run('sudo rm -f logs/hss*.* *.pcap')
		elif re.match('ltebox', self.Type, re.IGNORECASE):
			cmd.run('cp /opt/hss_sim0609/hss.log .')
			cmd.run('zip hss.log.zip hss.log')
		else:
			logging.error('This option should not occur!')
		cmd.close()

	def LogCollectMME(self):
		cmd = cls_cmd.getConnection(self.IPAddress)
		if self.Type != 'OC-OAI-CN5G':
			cmd.cd(f'{self.SourceCodePath}/scripts')
			cmd.run('rm -f mme.log.zip')
		if re.match('OAI-Rel14-Docker', self.Type, re.IGNORECASE):
			cmd.run('docker inspect prod-oai-mme')
			result = re.search('No such object', cmd.getBefore())
			if result is not None:
				cmd.cd('../logs')
				cmd.run('rm -f mme.log.zip')
				cmd.run('zip mme.log.zip mme_*.*')
				cmd.run('mv mme.log.zip ../scripts')
			else:
				cmd.run(f'docker cp {self.containerPrefix}-oai-mme:/openair-mme/mme_check_run.log .')
				cmd.run(f'docker cp {self.containerPrefix}-oai-mme:/tmp/mme_check_run.pcap .')
				cmd.run('zip mme.log.zip mme_check_run.*')
		elif re.match('OAICN5G', self.Type, re.IGNORECASE):
			cmd.cd(f'{self.SourceCodePath}/logs')
			cmd.run('cp -f /tmp/oai-cn5g-v1.5.pcap .')
			cmd.run('zip mme.log.zip oai-amf.log oai-nrf.log oai-cn5g*.pcap')
			cmd.run(f'mv mme.log.zip {self.SourceCodePath}/scripts')
		elif re.match('OC-OAI-CN5G', self.Type, re.IGNORECASE):
			cmd.run(f'cd {self.SourceCodePath}/logs')
			cmd.run('zip mme.log.zip oai-amf.log oai-nrf.log oai-cn5g*.pcap')
			cmd.run(f'mv mme.log.zip {self.SourceCodePath}/scripts')
		elif re.match('OAI', self.Type, re.IGNORECASE) or re.match('OAI-Rel14-CUPS', self.Type, re.IGNORECASE):
			cmd.run('zip mme.log.zip mme*.log')
			cmd.run('sudo rm mme*.log')
		elif re.match('ltebox', self.Type, re.IGNORECASE):
			cmd.run('cp /opt/ltebox/var/log/*Log.0 .')
			cmd.run('zip mme.log.zip mmeLog.0 s1apcLog.0 s1apsLog.0 s11cLog.0 libLog.0 s1apCodecLog.0 amfLog.0 ngapcLog.0 ngapcommonLog.0 ngapsLog.0')
		else:
			logging.error('This option should not occur!')
		cmd.close()

	def LogCollectSPGW(self):
		cmd = cls_cmd.getConnection(self.IPAddress)
		cmd.cd(f'{self.SourceCodePath}/scripts')
		cmd.run('rm -f spgw.log.zip')
		if re.match('OAI-Rel14-Docker', self.Type, re.IGNORECASE):
			cmd.run('docker inspect prod-oai-mme')
			result = re.search('No such object', cmd.getBefore())
			if result is not None:
				cmd.cd('../logs')
				cmd.run('rm -f spgw.log.zip')
				cmd.run('zip spgw.log.zip spgw*.*')
				cmd.run('mv spgw.log.zip ../scripts')
			else:
				cmd.run(f'docker cp {self.containerPrefix}-oai-spgwc:/openair-spgwc/spgwc_check_run.log .')
				cmd.run(f'docker cp {self.containerPrefix}-oai-spgwu-tiny:/openair-spgwu-tiny/spgwu_check_run.log .')
				cmd.run(f'docker cp {self.containerPrefix}-oai-spgwc:/tmp/spgwc_check_run.pcap .')
				cmd.run(f'docker cp {self.containerPrefix}-oai-spgwu-tiny:/tmp/spgwu_check_run.pcap .')
				cmd.run('zip spgw.log.zip spgw*_check_run.*')
		elif re.match('OAICN5G', self.Type, re.IGNORECASE):
			cmd.cd(f'{self.SourceCodePath}/logs')
			cmd.run('zip spgw.log.zip oai-smf.log oai-spgwu.log')
			cmd.run(f'mv spgw.log.zip {self.SourceCodePath}/scripts')
		elif re.match('OC-OAI-CN5G', self.Type, re.IGNORECASE):
			cmd.cd(f'{self.SourceCodePath}/logs')
			cmd.run('zip spgw.log.zip oai-smf.log oai-spgwu.log')
			cmd.run(f'mv spgw.log.zip {self.SourceCodePath}/scripts')
		elif re.match('OAI', self.Type, re.IGNORECASE) or re.match('OAI-Rel14-CUPS', self.Type, re.IGNORECASE):
			cmd.run('zip spgw.log.zip spgw*.log')
			cmd.run('sudo rm spgw*.log')
		elif re.match('ltebox', self.Type, re.IGNORECASE):
			cmd.run('cp /opt/ltebox/var/log/*Log.0 .')
			cmd.run('zip spgw.log.zip xGwLog.0 upfLog.0')
		else:
			logging.error('This option should not occur!')
		cmd.close()
