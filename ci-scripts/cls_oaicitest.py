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
# 
#
#   Required Python Version
#     Python 3.x
#
#   Required Python Package
#     pexpect
#---------------------------------------------------------------------


#-----------------------------------------------------------
# Import Libs
#-----------------------------------------------------------
import sys		# arg
import re		# reg
import time		# sleep
import os
import logging
import concurrent.futures
import json

#import our libs
import helpreadme as HELP
import constants as CONST

import cls_module
import cls_corenetwork
import cls_cmd

#-----------------------------------------------------------
# Helper functions used here and in other classes
#-----------------------------------------------------------
def Iperf_ComputeModifiedBW(idx, ue_num, profile, args):
	result = re.search(r'-b\s*(?P<iperf_bandwidth>[0-9\.]+)(?P<unit>[KMG])', str(args))
	if result is None:
		raise ValueError(f'requested iperf bandwidth not found in iperf options "{args}"')
	iperf_bandwidth = float(result.group('iperf_bandwidth'))
	if iperf_bandwidth == 0:
		raise ValueError('iperf bandwidth set to 0 - invalid value')
	if profile == 'balanced':
		iperf_bandwidth_new = iperf_bandwidth/ue_num
	if profile =='single-ue':
		iperf_bandwidth_new = iperf_bandwidth
	if profile == 'unbalanced':
		# residual is 2% of max bw
		residualBW = iperf_bandwidth / 50
		if idx == 0:
			iperf_bandwidth_new = iperf_bandwidth - ((ue_num - 1) * residualBW)
		else:
			iperf_bandwidth_new = residualBW
	iperf_bandwidth_str = result.group(0)
	iperf_bandwidth_unit = result.group(2)
	iperf_bandwidth_str_new = f"-b {'%.2f' % iperf_bandwidth_new}{iperf_bandwidth_unit}"
	args_new = re.sub(iperf_bandwidth_str, iperf_bandwidth_str_new, str(args))
	if iperf_bandwidth_unit == 'K':
		iperf_bandwidth_new = iperf_bandwidth_new / 1000
	return iperf_bandwidth_new, args_new

def Iperf_ComputeTime(args):
	result = re.search(r'-t\s*(?P<iperf_time>\d+)', str(args))
	if result is None:
		raise Exception('Iperf time not found!')
	return int(result.group('iperf_time'))

def Iperf_analyzeV3TCPJson(filename, iperf_tcp_rate_target):
	try:
		with open(filename) as f:
			 results = json.load(f)
		sender_bitrate   = round(results['end']['streams'][0]['sender']['bits_per_second'] / 1000000, 2)
		receiver_bitrate = round(results['end']['streams'][0]['receiver']['bits_per_second'] / 1000000, 2)
	except json.JSONDecodeError as e:
		return (False, f'Could not decode JSON log file {filename}: {e}')
	except KeyError as e:
		e_msg = results.get('error', f'error report not found in {filename}')
		return (False, f'While parsing Iperf3 results: missing key {e}, {e_msg}')
	except Exception as e:
		return (False, f'While parsing Iperf3 results: exception: {e}')
	snd_msg = f'Sender Bitrate   : {sender_bitrate} Mbps'
	rcv_msg = f'Receiver Bitrate : {receiver_bitrate} Mbps'
	success = True
	if iperf_tcp_rate_target is not None:
		success = float(receiver_bitrate) >= float(iperf_tcp_rate_target)
		if success:
			rcv_msg += f" (target: {iperf_tcp_rate_target})"
		else:
			rcv_msg += f" (too low! < {iperf_tcp_rate_target})"
	return(success, f'{snd_msg}\n{rcv_msg}')

def Iperf_analyzeV3BIDIRJson(filename):
	try:
		with open(filename) as f:
			results = json.load(f)
		sender_bitrate_ul   = round(results['end']['streams'][0]['sender']['bits_per_second'] / 1000000, 2)
		receiver_bitrate_ul = round(results['end']['streams'][0]['receiver']['bits_per_second'] / 1000000, 2)
		sender_bitrate_dl   = round(results['end']['streams'][1]['sender']['bits_per_second'] / 1000000, 2)
		receiver_bitrate_dl = round(results['end']['streams'][1]['receiver']['bits_per_second'] / 1000000, 2)
	except json.JSONDecodeError as e:
		return (False, f'Could not decode JSON log file: {e}')
	except KeyError as e:
		e_msg = results.get('error', f'error report not found in {filename}')
		return (False, f'While parsing Iperf3 results: missing key {e}, {e_msg}')
	except Exception as e:
		return (False, f'While parsing Iperf3 results: exception: {e}')

	msg = f'Sender Bitrate DL   : {sender_bitrate_dl} Mbps\n'
	msg += f'Receiver Bitrate DL : {receiver_bitrate_dl} Mbps\n'
	msg += f'Sender Bitrate UL   : {sender_bitrate_ul} Mbps\n'
	msg += f'Receiver Bitrate UL : {receiver_bitrate_ul} Mbps\n'
	return (True, msg)

def Iperf_analyzeV3UDP(filename, iperf_bitrate_threshold, iperf_packetloss_threshold, target_bitrate):
	if (not os.path.isfile(filename)):
		return (False, 'Iperf3 UDP: Log file not present')
	if (os.path.getsize(filename)==0):
		return (False, 'Iperf3 UDP: Log file is empty')
	sender_bitrate = None
	receiver_bitrate = None
	with open(filename, 'r') as server_file:
		for line in server_file.readlines():
			res_sender = re.search(r'(?P<bitrate>[0-9\.]+)\s+(?P<unit>[KMG]?bits\/sec)\s+(?P<jitter>[0-9\.]+\s+ms)\s+(?P<lostPack>-?\d+)/(?P<sentPack>-?\d+) \((?P<lost>[0-9\.]+).*?\s+(sender)', line)
			res_receiver = re.search(r'(?P<bitrate>[0-9\.]+)\s+(?P<unit>[KMG]?bits\/sec)\s+(?P<jitter>[0-9\.]+\s+ms)\s+(?P<lostPack>-?\d+)/(?P<receivedPack>-?\d+)\s+\((?P<lost>[0-9\.]+)%\).*?(receiver)', line)
			if res_sender is not None:
				sender_bitrate = res_sender.group('bitrate')
				sender_unit = res_sender.group('unit')
				sender_jitter = res_sender.group('jitter')
				sender_lostPack = res_sender.group('lostPack')
				sender_sentPack = res_sender.group('sentPack')
				sender_packetloss = res_sender.group('lost')
			if res_receiver is not None:
				receiver_bitrate = res_receiver.group('bitrate')
				receiver_unit = res_receiver.group('unit')
				receiver_jitter = res_receiver.group('jitter')
				receiver_lostPack = res_receiver.group('lostPack')
				receiver_receivedPack = res_receiver.group('receivedPack')
				receiver_packetloss = res_receiver.group('lost')

	if receiver_bitrate is not None and sender_bitrate is not None:
		if sender_unit == 'Kbits/sec':
			sender_bitrate = float(sender_bitrate) / 1000
		if receiver_unit == 'Kbits/sec':
			receiver_bitrate = float(receiver_bitrate) / 1000
		br_perf = 100 * float(receiver_bitrate) / float(target_bitrate)
		br_perf = '%.2f ' % br_perf
		sender_bitrate = '%.2f ' % float(sender_bitrate)
		receiver_bitrate = '%.2f ' % float(receiver_bitrate)
		req_msg = f'Sender Bitrate  : {sender_bitrate} Mbps'
		bir_msg = f'Receiver Bitrate: {receiver_bitrate} Mbps'
		brl_msg = f'{br_perf}%'
		jit_msg = f'Jitter          : {receiver_jitter}'
		pal_msg = f'Packet Loss     : {receiver_packetloss} %'
		if float(br_perf) < float(iperf_bitrate_threshold):
			brl_msg = f'too low! < {iperf_bitrate_threshold}%'
		if float(receiver_packetloss) > float(iperf_packetloss_threshold):
			pal_msg += f' (too high! > {iperf_packetloss_threshold}%)'
		result = float(br_perf) >= float(iperf_bitrate_threshold) and float(receiver_packetloss) <= float(iperf_packetloss_threshold)
		return (result, f'{req_msg}\n{bir_msg} ({brl_msg})\n{jit_msg}\n{pal_msg}')
	else:
		return (False, 'Could not analyze iperf report')

def Iperf_analyzeV2UDP(server_filename, iperf_bitrate_threshold, iperf_packetloss_threshold, target_bitrate):
		result = None
		if (not os.path.isfile(server_filename)):
			return (False, 'Iperf UDP: Server report not found!')
		if (os.path.getsize(server_filename)==0):
			return (False, 'Iperf UDP: Log file is empty')
		statusTemplate = r'(?:|\[ *\d+\].*) +0\.0+-\s*(?P<duration>[0-9\.]+) +sec +[0-9\.]+ [kKMG]Bytes +(?P<bitrate>[0-9\.]+) (?P<magnitude>[kKMG])bits\/sec +(?P<jitter>[0-9\.]+) ms +(\d+\/ *\d+) +(\((?P<packetloss>[0-9\.]+)%\))'
		with open(server_filename, 'r') as server_file:
			for line in server_file.readlines():
				result = re.search(statusTemplate, str(line)) or result
		if result is None:
			return (False, 'Could not parse server report!')
		bitrate = float(result.group('bitrate'))
		magn = result.group('magnitude')
		if magn == "k" or magn == "K":
			bitrate /= 1000
		elif magn == "G": # we assume bitrate in Mbps, therefore it must be G now
			bitrate *= 1000
		jitter = float(result.group('jitter'))
		packetloss = float(result.group('packetloss'))
		br_perf = float(bitrate)/float(target_bitrate) * 100
		br_perf = '%.2f ' % br_perf

		result = float(br_perf) >= float(iperf_bitrate_threshold) and float(packetloss) <= float(iperf_packetloss_threshold)
		req_msg = f'Req Bitrate : {target_bitrate}'
		bir_msg = f'Bitrate     : {bitrate}'
		brl_msg = f'Bitrate Perf: {br_perf} %'
		if float(br_perf) < float(iperf_bitrate_threshold):
			brl_msg += f' (too low! <{iperf_bitrate_threshold}%)'
		jit_msg = f'Jitter      : {jitter}'
		pal_msg = f'Packet Loss : {packetloss}'
		if float(packetloss) > float(iperf_packetloss_threshold):
			pal_msg += f' (too high! >{iperf_packetloss_threshold}%)'
		return (result, f'{req_msg}\n{bir_msg}\n{brl_msg}\n{jit_msg}\n{pal_msg}')

def Custom_Command(HTML, node, command):
    logging.info(f"Executing custom command on {node}")
    cmd = cls_cmd.getConnection(node)
    ret = cmd.run(command)
    cmd.close()
    logging.debug(f"Custom_Command: {command} on node: {node} - {'OK, command succeeded' if ret.returncode == 0 else f'Error, return code: {ret.returncode}'}")
    status = 'OK'
    message = []
    if ret.returncode != 0:
        message = [ret.stdout]
        logging.error(f'Custom_Command failed: output: {message}')
        status = 'KO'
    HTML.CreateHtmlTestRowQueue(command, status, message)
    return status == 'OK' or status == 'Warning'

def Custom_Script(HTML, node, script):
	logging.info(f"Executing custom script on {node}")
	ret = cls_cmd.runScript(node, script, 90)
	logging.debug(f"Custom_Script: {script} on node: {node} - return code {ret.returncode}, output:\n{ret.stdout}")
	status = 'OK'
	message = [ret.stdout]
	if ret.returncode != 0:
		status = 'KO'
	HTML.CreateHtmlTestRowQueue(script, status, message)
	return status == 'OK' or status == 'Warning'

def IdleSleep(HTML, idle_sleep_time):
	time.sleep(idle_sleep_time)
	HTML.CreateHtmlTestRow(f"{idle_sleep_time} sec", 'OK', CONST.ALL_PROCESSES_OK)
	return True

#-----------------------------------------------------------
# OaiCiTest Class Definition
#-----------------------------------------------------------
class OaiCiTest():
	
	def __init__(self):
		self.ranRepository = ''
		self.ranBranch = ''
		self.ranCommitID = ''
		self.ranAllowMerge = False
		self.ranTargetBranch = ''

		self.testCase_id = ''
		self.testXMLfiles = []
		self.ping_args = ''
		self.ping_packetloss_threshold = ''
		self.ping_rttavg_threshold =''
		self.iperf_args = ''
		self.iperf_packetloss_threshold = ''
		self.iperf_bitrate_threshold = ''
		self.iperf_profile = ''
		self.iperf_options = ''
		self.iperf_tcp_rate_target = ''
		self.finalStatus = False
		self.air_interface=''
		self.ue_ids = []
		self.nodes = []
		self.svr_node = None
		self.svr_id = None
		self.cmd_prefix = '' # prefix before {lte,nr}-uesoftmodem

	def InitializeUE(self, HTML):
		ues = [cls_module.Module_UE(n.strip()) for n in self.ue_ids]
		messages = []
		with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
			futures = [executor.submit(ue.initialize) for ue in ues]
			for f, ue in zip(futures, ues):
				uename = f'UE {ue.getName()}'
				messages.append(f'{uename}: initialized' if f.result() else f'{uename}: ERROR during Initialization')
			[f.result() for f in futures]
		HTML.CreateHtmlTestRowQueue('N/A', 'OK', messages)
		return True

	def AttachUE(self, HTML):
		ues = [cls_module.Module_UE(ue_id, server_name) for ue_id, server_name in zip(self.ue_ids, self.nodes)]
		with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
			futures = [executor.submit(ue.attach) for ue in ues]
			attached = [f.result() for f in futures]
			futures = [executor.submit(ue.checkMTU) for ue in ues]
			mtus = [f.result() for f in futures]
			messages = [f"UE {ue.getName()}: {ue.getIP()}" for ue in ues]
		success = all(attached) and all(mtus)
		if success:
			HTML.CreateHtmlTestRowQueue('N/A', 'OK', messages)
		else:
			logging.error(f'error attaching or wrong MTU: attached {attached}, mtus {mtus}')
			HTML.CreateHtmlTestRowQueue('N/A', 'KO', ["Could not retrieve UE IP address(es) or MTU(s) wrong!"])
		return success

	def DetachUE(self, HTML):
		ues = [cls_module.Module_UE(ue_id, server_name) for ue_id, server_name in zip(self.ue_ids, self.nodes)]
		with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
			futures = [executor.submit(ue.detach) for ue in ues]
			[f.result() for f in futures]
			messages = [f"UE {ue.getName()}: detached" for ue in ues]
		HTML.CreateHtmlTestRowQueue('NA', 'OK', messages)
		return True

	def DataDisableUE(self, HTML):
		ues = [cls_module.Module_UE(n.strip()) for n in self.ue_ids]
		with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
			futures = [executor.submit(ue.dataDisable) for ue in ues]
			status = [f.result() for f in futures]
		success = all(status)
		if success:
			messages = [f"UE {ue.getName()}: data disabled" for ue in ues]
			HTML.CreateHtmlTestRowQueue('NA', 'OK', messages)
		else:
			logging.error(f'error enabling data: {status}')
			HTML.CreateHtmlTestRowQueue('N/A', 'KO', ["Could not disable UE data!"])
		return success

	def DataEnableUE(self, HTML):
		ues = [cls_module.Module_UE(n.strip()) for n in self.ue_ids]
		logging.debug(f'disabling data for UEs {ues}')
		with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
			futures = [executor.submit(ue.dataEnable) for ue in ues]
			status = [f.result() for f in futures]
		success = all(status)
		if success:
			messages = [f"UE {ue.getName()}: data enabled" for ue in ues]
			HTML.CreateHtmlTestRowQueue('NA', 'OK', messages)
		else:
			logging.error(f'error enabling data: {status}')
			HTML.CreateHtmlTestRowQueue('N/A', 'KO', ["Could not enable UE data!"])
		return success

	def CheckStatusUE(self,HTML):
		ues = [cls_module.Module_UE(n.strip()) for n in self.ue_ids]
		logging.debug(f'checking status of UEs {ues}')
		messages = []
		with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
			futures = [executor.submit(ue.check) for ue in ues]
			messages = [f.result() for f in futures]
		HTML.CreateHtmlTestRowQueue('NA', 'OK', messages)
		return True

	def Ping_common(self, cn, ue, logPath):
		ping_status = 0
		ueIP = ue.getIP()
		if not ueIP:
			return (False, f"UE {ue.getName()} has no IP address")
		svrIP = cn.getIP()
		if not svrIP:
			return (False, f"CN {cn.getName()} has no IP address")
		ping_log_file = f'ping_{self.testCase_id}_{ue.getName()}.log'
		ping_time = re.findall(r"-c *(\d+)",str(self.ping_args))
		local_ping_log_file = f'{logPath}/{ping_log_file}'
		if re.search('%cn_ip%', self.ping_args) or re.search(r'[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+', self.ping_args):
			raise Exception(f"ping_args should not have IP address: {self.ping_args}")
		interface = f'-I {ue.getIFName()}' if ue.getIFName() else ''
		ping_cmd = f'{ue.getCmdPrefix()} ping {interface} {self.ping_args} {svrIP} 2>&1 | tee /tmp/{ping_log_file}'
		cmd = cls_cmd.getConnection(ue.getHost())
		response = cmd.run(ping_cmd, timeout=int(ping_time[0])*1.5)
		ue_header = f'UE {ue.getName()} ({ueIP})'
		if response.returncode != 0:
			message = ue_header + ': ping crashed: TIMEOUT?'
			return (False, message)

		#copy the ping log file to have it locally for analysis (ping stats)
		cmd.copyin(src=f'/tmp/{ping_log_file}', tgt=local_ping_log_file)
		cmd.close()

		with open(local_ping_log_file, 'r') as f:
			ping_output = "".join(f.readlines())
		result = re.search(r', (?P<packetloss>[0-9\.]+)% packet loss, time [0-9\.]+ms', ping_output)
		if result is None:
			message = ue_header + ': Packet Loss Not Found!'
			return (False, message)
		packetloss = result.group('packetloss')
		result = re.search(r'rtt min\/avg\/max\/mdev = (?P<rtt_min>[0-9\.]+)\/(?P<rtt_avg>[0-9\.]+)\/(?P<rtt_max>[0-9\.]+)\/[0-9\.]+ ms', ping_output)
		if result is None:
			message = ue_header + ': Ping RTT_Min RTT_Avg RTT_Max Not Found!'
			return (False, message)
		rtt_min = result.group('rtt_min')
		rtt_avg = result.group('rtt_avg')
		rtt_max = result.group('rtt_max')

		pal_msg = f'Packet Loss: {packetloss}%'
		min_msg = f'RTT(Min)   : {rtt_min} ms'
		avg_msg = f'RTT(Avg)   : {rtt_avg} ms'
		max_msg = f'RTT(Max)   : {rtt_max} ms'

		message = f'{ue_header}\n{pal_msg}\n{min_msg}\n{avg_msg}\n{max_msg}'

		#checking packet loss compliance
		if float(packetloss) > float(self.ping_packetloss_threshold):
			message += '\nPacket Loss too high'
			return (False, message)
		elif float(packetloss) > 0:
			message += '\nPacket Loss is not 0%'

		if self.ping_rttavg_threshold != '':
			if float(rtt_avg) > float(self.ping_rttavg_threshold):
				ping_rttavg_error_msg = f'RTT(Avg) too high: {rtt_avg} ms; Target: {self.ping_rttavg_threshold} ms'
				message += f'\n {ping_rttavg_error_msg}'
				return (False, message)

		return (True, message)

	def Ping(self, HTML, CONTAINERS, infra_file="ci_infra.yaml"):
		if self.ue_ids == [] or self.svr_id == None:
			raise Exception("no module names in self.ue_ids or/and self.svr_id provided")
		# Creating destination log folder if needed on the python executor workspace
		with cls_cmd.getConnection('localhost') as local:
			ymlPath = CONTAINERS.yamlPath[0].split('/')
			logPath = f'{os.getcwd()}/../cmake_targets/log/{ymlPath[-1]}'
			local.run(f'mkdir -p {logPath}', silent=True)
		ues = [cls_module.Module_UE(ue_id, server_name, infra_file) for ue_id, server_name in zip(self.ue_ids, self.nodes)]
		cn = cls_corenetwork.CoreNetwork(self.svr_id, self.svr_node, filename=infra_file)
		with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
			futures = [executor.submit(self.Ping_common, cn, ue, logPath) for ue in ues]
			results = [f.result() for f in futures]
			# each result in results is a tuple, first member goes to successes, second to messages
			successes, messages = map(list, zip(*results))

		success = len(successes) == len(ues) and all(successes)
		logger = logging.info if success else logging.error
		hcolor = "\u001B[1;37;44m" if success else "\u001B[1;37;41m"
		lcolor = "\u001B[1;34m" if success else "\u001B[1;31m"
		for m in messages:
			lines = m.split('\n')
			logger(f'{hcolor} ping result for {lines[0]} \u001B[0m')
			for l in lines[1:]:
				logger(f'{lcolor}    {l}\u001B[0m')

		if success:
			HTML.CreateHtmlTestRowQueue(self.ping_args, 'OK', messages)
		else:
			HTML.CreateHtmlTestRowQueue(self.ping_args, 'KO', messages)
		return success

	def Iperf_Module(self, cn, ue, idx, ue_num, logPath):
		ueIP = ue.getIP()
		if not ueIP:
			return (False, f"UE {ue.getName()} has no IP address")
		svrIP = cn.getIP()
		if not svrIP:
			return (False, f"Iperf server {cn.getName()} has no IP address")

		iperf_opt = self.iperf_args
		jsonReport = "--json"
		serverReport = ""
		udpIperf = re.search('-u', iperf_opt) is not None
		bidirIperf = re.search('--bidir', iperf_opt) is not None
		client_filename = f'iperf_client_{self.testCase_id}_{ue.getName()}.log'
		if udpIperf:
			target_bitrate, iperf_opt = Iperf_ComputeModifiedBW(idx, ue_num, self.iperf_profile, self.iperf_args)
			# note: for UDP testing we don't want to use json report - reports 0 Mbps received bitrate
			jsonReport = ""
			# note: enable server report collection on the UE side, no need to store and collect server report separately on the server side
			serverReport = "--get-server-output"
		iperf_time = Iperf_ComputeTime(self.iperf_args)
		# hack: the ADB UEs don't have iperf in $PATH, so we need to hardcode for the moment
		iperf_ue = '/data/local/tmp/iperf3' if re.search('adb', ue.getName()) else 'iperf3'
		ue_header = f'UE {ue.getName()} ({ueIP})'
		with cls_cmd.getConnection(ue.getHost()) as cmd_ue, cls_cmd.getConnection(cn.getHost()) as cmd_svr:
			port = 5002 + idx
			# note: some core setups start an iperf3 server automatically, indicated in ci_infra by runIperf3Server: False`
			t = iperf_time * 2.5
			cmd_ue.run(f'rm /tmp/{client_filename}', reportNonZero=False, silent=True)
			if cn.runIperf3Server():
				cmd_svr.run(f'{cn.getCmdPrefix()} timeout -vk3 {t} iperf3 -s -B {svrIP} -p {port} -1 {jsonReport} >> /dev/null &', timeout=t)
			cmd_ue.run(f'{ue.getCmdPrefix()} timeout -vk3 {t} {iperf_ue} -B {ueIP} -c {svrIP} -p {port} {iperf_opt} {jsonReport} {serverReport} -O 5 >> /tmp/{client_filename}', timeout=t)
			# note: copy iperf3 log to the current directory for log analysis and log collection
			dest_filename = f'{logPath}/{client_filename}'
			cmd_ue.copyin(f'/tmp/{client_filename}', dest_filename)
			cmd_ue.run(f'rm /tmp/{client_filename}', reportNonZero=False, silent=True)
		if udpIperf:
			status, msg = Iperf_analyzeV3UDP(dest_filename, self.iperf_bitrate_threshold, self.iperf_packetloss_threshold, target_bitrate)
		elif bidirIperf:
			status, msg = Iperf_analyzeV3BIDIRJson(dest_filename)
		else:
			status, msg = Iperf_analyzeV3TCPJson(dest_filename, self.iperf_tcp_rate_target)

		return (status, f'{ue_header}\n{msg}')

	def Iperf(self, HTML, CONTAINERS, infra_file="ci_infra.yaml"):
		logging.debug(f'Iperf: iperf_args "{self.iperf_args}" iperf_packetloss_threshold "{self.iperf_packetloss_threshold}" iperf_bitrate_threshold "{self.iperf_bitrate_threshold}" iperf_profile "{self.iperf_profile}" iperf_options "{self.iperf_options}"')

		if self.ue_ids == [] or self.svr_id == None:
			raise Exception("no module names in self.ue_ids or/and self.svr_id provided")
		# create log directory on executor node
		with cls_cmd.getConnection('localhost') as local:
			ymlPath = CONTAINERS.yamlPath[0].split('/')
			logPath = f'{os.getcwd()}/../cmake_targets/log/{ymlPath[-1]}'
			local.run(f'mkdir -p {logPath}', silent=True)
		ues = [cls_module.Module_UE(ue_id, server_name, infra_file) for ue_id, server_name in zip(self.ue_ids, self.nodes)]
		cn = cls_corenetwork.CoreNetwork(self.svr_id, self.svr_node, filename=infra_file)
		with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
			futures = [executor.submit(self.Iperf_Module, cn, ue, i, len(ues), logPath) for i, ue in enumerate(ues)]
			results = [f.result() for f in futures]
			# each result in results is a tuple, first member goes to successes, second to messages
			successes, messages = map(list, zip(*results))

		success = len(successes) == len(ues) and all(successes)
		logger = logging.info if success else logging.error
		hcolor = "\u001B[1;37;45m" if success else "\u001B[1;37;41m"
		lcolor = "\u001B[1;35m" if success else "\u001B[1;31m"
		for m in messages:
			lines = m.split('\n')
			logger(f'{hcolor} iperf result for {lines[0]} \u001B[0m')
			for l in lines[1:]:
				logger(f'{lcolor}    {l}\u001B[0m')

		if success:
			HTML.CreateHtmlTestRowQueue(self.iperf_args, 'OK', messages)
		else:
			HTML.CreateHtmlTestRowQueue(self.iperf_args, 'KO', messages)
		return success

	def Iperf2_Unidir(self, HTML, CONTAINERS, infra_file="ci_infra.yaml"):
		if self.ue_ids == [] or self.svr_id == None or len(self.ue_ids) != 1:
			raise Exception("no module names in self.ue_ids or/and self.svr_id provided, multi UE scenario not supported")
		ue = cls_module.Module_UE(self.ue_ids[0].strip(),self.nodes[0].strip(), infra_file)
		cn = cls_corenetwork.CoreNetwork(self.svr_id, self.svr_node, filename=infra_file)
		ueIP = ue.getIP()
		if not ueIP:
			return False
		svrIP = cn.getIP()
		if not svrIP:
			return False
		server_filename = f'iperf_server_{self.testCase_id}_{ue.getName()}.log'
		ymlPath = CONTAINERS.yamlPath[0].split('/')
		logPath = f'{os.getcwd()}/../cmake_targets/log/{ymlPath[-1]}'
		iperf_time = Iperf_ComputeTime(self.iperf_args)
		target_bitrate, iperf_opt = Iperf_ComputeModifiedBW(0, 1, self.iperf_profile, self.iperf_args)
		t = iperf_time*2.5
		with cls_cmd.getConnection('localhost') as local:
			local.run(f'mkdir -p {logPath}')
		with cls_cmd.getConnection(ue.getHost()) as cmd_ue, cls_cmd.getConnection(cn.getHost()) as cmd_svr:
			cmd_ue.run(f'rm /tmp/{server_filename}', reportNonZero=False)
			cmd_ue.run(f'{ue.getCmdPrefix()} timeout -vk3 {t} iperf -B {ueIP} -s -u -i1 >> /tmp/{server_filename} &', timeout=t)
			cmd_svr.run(f'{cn.getCmdPrefix()} timeout -vk3 {t} iperf -c {ueIP} -B {svrIP} {iperf_opt} -i1 >> /dev/null', timeout=t)
			localPath = f'{os.getcwd()}'
			# note: copy iperf2 log to the directory for log collection
			cmd_ue.copyin(f'/tmp/{server_filename}', f'{logPath}/{server_filename}')
			cmd_ue.run(f'rm /tmp/{server_filename}', reportNonZero=False)
		success, msg = Iperf_analyzeV2UDP(f'{logPath}/{server_filename}', self.iperf_bitrate_threshold, self.iperf_packetloss_threshold, target_bitrate)
		ue_header = f'UE {ue.getName()} ({ueIP})'
		logging.info(f'\u001B[1;37;45m iperf result for {ue_header}\u001B[0m')
		for l in msg.split('\n'):
			logging.info(f'\u001B[1;35m	{l} \u001B[0m')
		if success:
			HTML.CreateHtmlTestRowQueue(self.iperf_args, 'OK', [f'{ue_header}\n{msg}'])
		else:
			HTML.CreateHtmlTestRowQueue(self.iperf_args, 'KO', [f'{ue_header}\n{msg}'])
		return success

	def AnalyzeLogFile_UE(self, UElogFile,HTML,RAN):
		if (not os.path.isfile(f'{UElogFile}')):
			return -1
		ue_log_file = open(f'{UElogFile}', 'r')
		exitSignalReceived = False
		foundAssertion = False
		msgAssertion = ''
		msgLine = 0
		foundSegFault = False
		foundRealTimeIssue = False
		uciStatMsgCount = 0
		pdcpDataReqFailedCount = 0
		badDciCount = 0
		f1aRetransmissionCount = 0
		fatalErrorCount = 0
		macBsrTimerExpiredCount = 0
		rrcConnectionRecfgComplete = 0
		no_cell_sync_found = False
		mib_found = False
		frequency_found = False
		plmn_found = False
		nrUEFlag = False
		nrDecodeMib = 0
		nrFoundDCI = 0
		nrCRCOK = 0
		mbms_messages = 0
		nbPduSessAccept = 0
		nbPduDiscard = 0
		HTML.htmlUEFailureMsg=''
		global_status = CONST.ALL_PROCESSES_OK
		for line in ue_log_file.readlines():
			result = re.search('nr_synchro_time|Starting NR UE soft modem', str(line))
			sidelink = re.search('sl-mode', str(line))
			if result is not None:
				nrUEFlag = True
			if sidelink is not None:
				nrUEFlag = False
			if nrUEFlag:
				result = re.search('decode mib', str(line))
				if result is not None:
					nrDecodeMib += 1
				result = re.search('found 1 DCIs', str(line))
				if result is not None:
					nrFoundDCI += 1
				result = re.search('CRC OK', str(line))
				if result is not None:
					nrCRCOK += 1
				result = re.search('Received PDU Session Establishment Accept', str(line))
				if result is not None:
					nbPduSessAccept += 1
				result = re.search('warning: discard PDU, sn out of window', str(line))
				if result is not None:
					nbPduDiscard += 1
				result = re.search('--nfapi STANDALONE_PNF --node-number 2', str(line))
				if result is not None:
					frequency_found = True
			result = re.search('Exiting OAI softmodem', str(line))
			if result is not None:
				exitSignalReceived = True
			result = re.search('System error|[Ss]egmentation [Ff]ault|======= Backtrace: =========|======= Memory map: ========', str(line))
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
			if foundAssertion and (msgLine < 3):
				msgLine += 1
				msgAssertion += str(line)
			result = re.search('uci->stat', str(line))
			if result is not None and not exitSignalReceived:
				uciStatMsgCount += 1
			result = re.search('PDCP data request failed', str(line))
			if result is not None and not exitSignalReceived:
				pdcpDataReqFailedCount += 1
			result = re.search('bad DCI 1', str(line))
			if result is not None and not exitSignalReceived:
				badDciCount += 1
			result = re.search('Format1A Retransmission but TBS are different', str(line))
			if result is not None and not exitSignalReceived:
				f1aRetransmissionCount += 1
			result = re.search('FATAL ERROR', str(line))
			if result is not None and not exitSignalReceived:
				fatalErrorCount += 1
			result = re.search('MAC BSR Triggered ReTxBSR Timer expiry', str(line))
			if result is not None and not exitSignalReceived:
				macBsrTimerExpiredCount += 1
			result = re.search('Generating RRCConnectionReconfigurationComplete', str(line))
			if result is not None:
				rrcConnectionRecfgComplete += 1
			# No cell synchronization found, abandoning
			result = re.search('No cell synchronization found, abandoning', str(line))
			if result is not None:
				no_cell_sync_found = True
			if RAN.eNBmbmsEnables[0]:
				result = re.search('TRIED TO PUSH MBMS DATA', str(line))
				if result is not None:
					mbms_messages += 1
			result = re.search(r"MIB Information => ([a-zA-Z]{1,10}), ([a-zA-Z]{1,10}), NidCell (?P<nidcell>\d{1,3}), N_RB_DL (?P<n_rb_dl>\d{1,3}), PHICH DURATION (?P<phich_duration>\d), PHICH RESOURCE (?P<phich_resource>.{1,4}), TX_ANT (?P<tx_ant>\d)", str(line))
			if result is not None and (not mib_found):
				try:
					mibMsg = "MIB Information: " + result.group(1) + ', ' + result.group(2)
					HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + mibMsg + '\n'
					logging.debug(f'\033[94m{mibMsg}\033[0m')
					mibMsg = "    nidcell = " + result.group('nidcell')
					HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + mibMsg
					logging.debug(f'\033[94m{mibMsg}\033[0m')
					mibMsg = "    n_rb_dl = " + result.group('n_rb_dl')
					HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + mibMsg + '\n'
					logging.debug(f'\033[94m{mibMsg}\033[0m')
					mibMsg = "    phich_duration = " + result.group('phich_duration')
					HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + mibMsg
					logging.debug(f'\033[94m{mibMsg}\033[0m')
					mibMsg = "    phich_resource = " + result.group('phich_resource')
					HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + mibMsg + '\n'
					logging.debug(f'\033[94m{mibMsg}\033[0m')
					mibMsg = "    tx_ant = " + result.group('tx_ant')
					HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + mibMsg + '\n'
					logging.debug(f'\033[94m{mibMsg}\033[0m')
					mib_found = True
				except Exception as e:
					logging.error(f'\033[91m MIB marker was not found \033[0m')
			result = re.search("Initial sync: pbch decoded sucessfully", str(line))
			if result is not None and (not frequency_found):
				try:
					mibMsg = f"UE decoded PBCH successfully"
					HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + mibMsg + '\n'
					logging.debug(f'\033[94m{mibMsg}\033[0m')
					frequency_found = True
				except Exception as e:
					logging.error(f'\033[91m UE did not find PBCH\033[0m')
			result = re.search(r"PLMN MCC (?P<mcc>\d{1,3}), MNC (?P<mnc>\d{1,3}), TAC", str(line))
			if result is not None and (not plmn_found):
				try:
					mibMsg = f"PLMN MCC = {result.group('mcc')} MNC = {result.group('mnc')}"
					HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + mibMsg + '\n'
					logging.debug(f'\033[94m{mibMsg}\033[0m')
					plmn_found = True
				except Exception as e:
					logging.error(f'\033[91m PLMN not found \033[0m')
			result = re.search(r"Found (?P<operator>[\w,\s]{1,15}) \(name from internal table\)", str(line))
			if result is not None:
				try:
					mibMsg = f"The operator is: {result.group('operator')}"
					HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + mibMsg + '\n'
					logging.debug(f'\033[94m{mibMsg}\033[0m')
				except Exception as e:
					logging.error(f'\033[91m Operator name not found \033[0m')
			result = re.search("SIB5 InterFreqCarrierFreq element (.{1,4})/(.{1,4})", str(line))
			if result is not None:
				try:
					mibMsg = f'SIB5 InterFreqCarrierFreq element {result.group(1)}/{result.group(2)}'
					HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + mibMsg + ' -> '
					logging.debug(f'\033[94m{mibMsg}\033[0m')
				except Exception as e:
					logging.error(f'\033[91m SIB5 InterFreqCarrierFreq element not found \033[0m')
			result = re.search(r"DL Carrier Frequency/ARFCN : \-*(?P<carrier_frequency>\d{1,15}/\d{1,4})", str(line))
			if result is not None:
				try:
					freq = result.group('carrier_frequency')
					new_freq = re.sub('/[0-9]+','',freq)
					float_freq = float(new_freq) / 1000000
					HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + 'DL Freq: ' + ('%.1f' % float_freq) + ' MHz'
					logging.debug(f'\033[94m    DL Carrier Frequency is:  {freq}\033[0m')
				except Exception as e:
					logging.error(f'\033[91m    DL Carrier Frequency not found \033[0m')
			result = re.search(r"AllowedMeasBandwidth : (?P<allowed_bandwidth>\d{1,7})", str(line))
			if result is not None:
				try:
					prb = result.group('allowed_bandwidth')
					HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + ' -- PRB: ' + prb + '\n'
					logging.debug(f'\033[94m    AllowedMeasBandwidth: {prb}\033[0m')
				except Exception as e:
					logging.error(f'\033[91m    AllowedMeasBandwidth not found \033[0m')
		ue_log_file.close()
		if rrcConnectionRecfgComplete > 0:
			statMsg = f'UE connected to eNB ({rrcConnectionRecfgComplete}) RRCConnectionReconfigurationComplete message(s) generated)'
			logging.debug(f'\033[94m{statMsg}\033[0m')
			HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + statMsg + '\n'
		if nrUEFlag:
			if nrDecodeMib > 0:
				statMsg = f'UE showed {nrDecodeMib} "MIB decode" message(s)'
				logging.debug(f'\u001B[1;30;43m{statMsg}\u001B[0m')
				HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + statMsg + '\n'
			if nrFoundDCI > 0:
				statMsg = f'UE showed {nrFoundDCI} "DCI found" message(s)'
				logging.debug(f'\u001B[1;30;43m{statMsg}\u001B[0m')
				HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + statMsg + '\n'
			if nrCRCOK > 0:
				statMsg = f'UE showed {nrCRCOK} "PDSCH decoding" message(s)'
				logging.debug(f'\u001B[1;30;43m{statMsg}\u001B[0m')
				HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + statMsg + '\n'
			if not frequency_found:
				statMsg = 'NR-UE could NOT synch!'
				logging.error(f'\u001B[1;30;43m{statMsg}\u001B[0m')
				HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + statMsg + '\n'
			if nbPduSessAccept > 0:
				statMsg = f'UE showed {nbPduSessAccept} "Received PDU Session Establishment Accept" message(s)'
				logging.debug(f'\u001B[1;30;43m{statMsg}\u001B[0m')
				HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + statMsg + '\n'
			if nbPduDiscard > 0:
				statMsg = f'UE showed {nbPduDiscard} "warning: discard PDU, sn out of window" message(s)'
				logging.debug(f'\u001B[1;30;43m{statMsg}\u001B[0m')
				HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + statMsg + '\n'
		if uciStatMsgCount > 0:
			statMsg = f'UE showed {uciStatMsgCount} "uci->stat" message(s)'
			logging.debug(f'\u001B[1;30;43m{statMsg}\u001B[0m')
			HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + statMsg + '\n'
		if pdcpDataReqFailedCount > 0:
			statMsg = f'UE showed {pdcpDataReqFailedCount} "PDCP data request failed" message(s)'
			logging.debug(f'\u001B[1;30;43m{statMsg}\u001B[0m')
			HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + statMsg + '\n'
		if badDciCount > 0:
			statMsg = f'UE showed {badDciCount} "bad DCI 1(A)" message(s)'
			logging.debug(f'\u001B[1;30;43m{statMsg}\u001B[0m')
			HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + statMsg + '\n'
		if f1aRetransmissionCount > 0:
			statMsg = f'UE showed {f1aRetransmissionCount} "Format1A Retransmission but TBS are different" message(s)'
			logging.debug(f'\u001B[1;30;43m{statMsg}\u001B[0m')
			HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + statMsg + '\n'
		if fatalErrorCount > 0:
			statMsg = f'UE showed {fatalErrorCount} "FATAL ERROR:" message(s)'
			logging.debug(f'\u001B[1;30;43m{statMsg}\u001B[0m')
			HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + statMsg + '\n'
		if macBsrTimerExpiredCount > 0:
			statMsg = f'UE showed {fatalErrorCount} "MAC BSR Triggered ReTxBSR Timer expiry" message(s)'
			logging.debug(f'\u001B[1;30;43m{statMsg}\u001B[0m')
			HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + statMsg + '\n'
		if RAN.eNBmbmsEnables[0]:
			if mbms_messages > 0:
				statMsg = f'UE showed {mbms_messages} "TRIED TO PUSH MBMS DATA" message(s)'
				logging.debug(f'\u001B[1;30;43m{statMsg}\u001B[0m')
			else:
				statMsg = 'UE did NOT SHOW "TRIED TO PUSH MBMS DATA" message(s)'
				logging.debug(f'\u001B[1;30;41m{statMsg}\u001B[0m')
				global_status = CONST.OAI_UE_PROCESS_NO_MBMS_MSGS
			HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + statMsg + '\n'
		if foundSegFault:
			logging.debug('\u001B[1;37;41m UE ended with a Segmentation Fault! \u001B[0m')
			if not nrUEFlag:
				global_status = CONST.OAI_UE_PROCESS_SEG_FAULT
			else:
				if not frequency_found:
					global_status = CONST.OAI_UE_PROCESS_SEG_FAULT
		if foundAssertion:
			logging.debug('\u001B[1;30;43m UE showed an assertion! \u001B[0m')
			HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + 'UE showed an assertion!\n'
			if not nrUEFlag:
				if not mib_found or not frequency_found:
					global_status = CONST.OAI_UE_PROCESS_ASSERTION
			else:
				if not frequency_found:
					global_status = CONST.OAI_UE_PROCESS_ASSERTION
		if foundRealTimeIssue:
			logging.debug('\u001B[1;37;41m UE faced real time issues! \u001B[0m')
			HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + 'UE faced real time issues!\n'
		if nrUEFlag:
			if not frequency_found:
				global_status = CONST.OAI_UE_PROCESS_COULD_NOT_SYNC
		else:
			if no_cell_sync_found and not mib_found:
				logging.debug('\u001B[1;37;41m UE could not synchronize ! \u001B[0m')
				HTML.htmlUEFailureMsg=HTML.htmlUEFailureMsg + 'UE could not synchronize!\n'
				global_status = CONST.OAI_UE_PROCESS_COULD_NOT_SYNC
		return global_status

	def TerminateUE(self, HTML, logPath):
		ues = [cls_module.Module_UE(n.strip()) for n in self.ue_ids]
		with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
			futures = [executor.submit(ue.terminate, logPath) for ue in ues]
			archives = [f.result() for f in futures]
		archive_info = [f'Log at: {a}' if a else 'No log available' for a in archives]
		messages = [f"UE {ue.getName()}: {log}" for (ue, log) in zip(ues, archive_info)]
		HTML.CreateHtmlTestRowQueue(f'N/A', 'OK', messages)
		return True

	def DeployCoreNetwork(cn_id, ctx, HTML):
		core_name = cn_id.strip()
		cn = cls_corenetwork.CoreNetwork(core_name)
		success, output = cn.deploy()
		logging.info(f"deployment core network {core_name} success {success}, output:\n{output}")
		if success:
			msg = f"Started {cn} [{cn.getIP()}]"
			HTML.CreateHtmlTestRowQueue(core_name, 'OK', [msg])
		else:
			msg = f"deployment of core network {core_name} FAILED"
			logging.error(msg)
			HTML.CreateHtmlTestRowQueue(core_name, 'KO', [msg])
		return success

	def UndeployCoreNetwork(cn_id, ctx, HTML):
		core_name = cn_id.strip()
		cn = cls_corenetwork.CoreNetwork(core_name)
		logs, output = cn.undeploy(ctx=ctx)
		logging.info(f"undeployed core network {core_name}, logs {logs}, output:\n{output}")
		message = "Log files:\n" + "\n".join([os.path.basename(l) for l in logs])
		HTML.CreateHtmlTestRowQueue(core_name, 'OK', [message])
		return True

	def LogCollectBuild(self,RAN):
		# Some pipelines are using "none" IP / Credentials
		# In that case, just forget about it
		if RAN.eNBIPAddress == 'none':
			sys.exit(0)

		if (RAN.eNBIPAddress != '' and RAN.eNBUserName != '' and RAN.eNBPassword != ''):
			IPAddress = RAN.eNBIPAddress
			UserName = RAN.eNBUserName
			Password = RAN.eNBPassword
			SourceCodePath = RAN.eNBSourceCodePath
		else:
			sys.exit('Insufficient Parameter')
		with cls_cmd.getConnection(IPAddress) as cmd:
			d = f'{SourceCodePath}/cmake_targets'
			cmd.run(f'rm -f {d}/build.log.zip')
			cmd.run(f'cd {d} && zip -r build.log.zip build_log_*/*')
