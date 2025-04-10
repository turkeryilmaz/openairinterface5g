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
# Import Components
#-----------------------------------------------------------

import helpreadme as HELP
import constants as CONST


import cls_oaicitest		 #main class for OAI CI test framework
import cls_containerize	 #class Containerize for all container-based operations on RAN/UE objects
import cls_static_code_analysis  #class for static code analysis
import cls_physim1		 #class PhySim for physical simulators deploy and run
import cls_cluster		 # class for building/deploying on cluster
import cls_native        # class for all native/source-based operations

import ran
import cls_oai_html


#-----------------------------------------------------------
# Import Libs
#-----------------------------------------------------------
import sys		# arg
import re		# reg
import time		# sleep
import os
import subprocess
from lxml import etree as ET
import logging
import signal
import traceback
logging.basicConfig(
	level=logging.DEBUG,
	stream=sys.stdout,
	format="[%(asctime)s] %(levelname)8s: %(message)s"
)

import copy

#-----------------------------------------------------------
# General Functions
#-----------------------------------------------------------



def CheckClassValidity(xml_class_list,action,id):
	if action not in xml_class_list:
		logging.error('test-case ' + id + ' has unlisted class ' + action + ' ##CHECK xml_class_list.yml')
		resp=False
	else:
		resp=True
	return resp

#assigning parameters to object instance attributes (even if the attributes do not exist !!)
def AssignParams(params_dict):

	for key,value in params_dict.items():
		setattr(CiTestObj, key, value)
		setattr(RAN, key, value)
		setattr(HTML, key, value)

def LoadTests(file):
	if not os.path.exists(file):
		logging.error(f'Test file not found: {file}')
		sys.exit(1)

	xmlTree = ET.parse(file)
	xmlRoot = xmlTree.getroot()

	# Handle XInclude
	try:
		xmlTree.xinclude()
	except ET.XIncludeError as e:
		logging.error("XInclude failed: %s" % str(e))
		sys.exit(1)

	test_list = {}

    # Iterate through all test cases in the file
	if (HTML.nbTestXMLfiles == 1):
		HTML.htmlTabRefs.append(xmlRoot.findtext('htmlTabRef',default='test-tab-0'))
		HTML.htmlTabNames.append(xmlRoot.findtext('htmlTabName',default='Test-0'))
	for test in xmlRoot.findall('testCase'):
		test_id = test.get('id')  # Get test ID
		#check that requested tests are well formatted
		#(6 digits or less than 6 digits followed by +)
		#be verbose
		if (re.match('^[0-9]{6}$', test_id) or re.match('^[0-9]{1,5}\\+$', test)):
			logging.debug(f'test group/case requested: {test_id}')
		else:
			logging.error(f'Requested test is invalidly formatted: {test_id}')
			sys.exit(1)

		ci_test_obj = copy.deepcopy(CiTestObj)

		# Populate the CiTestObj with values from XML
		ci_test_obj.testCase_id = test_id
		ci_test_obj.desc = test.findtext('desc', default='')
		ci_test_obj.type = test.findtext('class', default='')
		ci_test_obj.eNB_instance = test.findtext('eNB_instance', default='')
		ci_test_obj.eNB_serverId = test.findtext('eNB_serverId', default='')
		ci_test_obj.rt_stats_cfg = test.findtext('rt_stats_cfg', default='')
		ci_test_obj.USRP_IPAddress = test.findtext('USRP_IPAddress', default='')
		ci_test_obj.Initialize_eNB_args = test.findtext('Initialize_eNB_args', default='')
		ci_test_obj.ping_args = test.findtext('ping_args', default='')
		logging.error(f'DEBUG LOADED {ci_test_obj.ping_args}')
		ci_test_obj.ping_packetloss_threshold = test.findtext('ping_packetloss_threshold', default='')
		ci_test_obj.ping_rttavg_threshold = test.findtext('ping_rttavg_threshold', default='')
		ci_test_obj.iperf_args = test.findtext('iperf_args', default='')
		ci_test_obj.iperf_packetloss_threshold = test.findtext('iperf_packetloss_threshold', default='')
		ci_test_obj.iperf_bitrate_threshold = test.findtext('iperf_bitrate_threshold', default='90')
		ci_test_obj.iperf_profile = test.findtext('iperf_profile', default='balanced')
		ci_test_obj.iperf_options = test.findtext('iperf_options', default='check')
		if ci_test_obj.iperf_options != 'check' and ci_test_obj.iperf_options != 'sink':
			logging.error(f'test-case {test_id} has wrong iperf_options{ci_test_obj.iperf_options}')
			ci_test_obj.iperf_options = 'check'
		ci_test_obj.iperf_tcp_rate_target = test.findtext('iperf_tcp_rate_target', default=None)
		ci_test_obj.finalStatus = test.findtext('finalStatus', default='False') == 'True'
		ci_test_obj.UEIPAddress = test.findtext('UEIPAddress', default='')
		ci_test_obj.UEUserName = test.findtext('UEUserName', default='')
		ci_test_obj.UEPassword = test.findtext('UEPassword', default='')
		ci_test_obj.UESourceCodePath = test.findtext('UESourceCodePath', default='')
		ci_test_obj.UELogFile = test.findtext('UELogFile', default='')
		ci_test_obj.air_interface = test.findtext('air_interface', default='')
		ci_test_obj.always_exec = test.findtext('always_exec', default='False') in ['True', 'true', 'Yes', 'yes']
		ci_test_obj.cmd_prefix = test.findtext('cmd_prefix', default='')
		ci_test_obj.d_retx_th = test.findtext('d_retx_th', default='')
		ci_test_obj.u_retx_th = test.findtext('u_retx_th', default='')
		ci_test_obj.ue_ids = test.findtext('id', default='').split(' ')
		ci_test_obj.yaml_path = test.findtext('yaml_path', default='')
		ci_test_obj.services = test.findtext('services', default='')
		ci_test_obj.num_attempts = test.findtext('num_attempts', default='1')
		ci_test_obj.node = test.findtext('node', default='')
		ci_test_obj.command = test.findtext('command', default='')
		ci_test_obj.command_fail = test.findtext('command_fail', default='') in ['True', 'true', 'Yes', 'yes']
		ci_test_obj.images = test.findtext('images', default='').split()
		ci_test_obj.script = test.findtext('script', default='')
		ci_test_obj.parameters = test.findtext('parameters', default='')
		ci_test_obj.args = test.findtext('args', default='')
		ci_test_obj.cn_id = test.findtext('cn_id', default='')
		ci_test_obj.idle_sleep_time_in_sec = test.findtext('idle_sleep_time_in_sec', default='5')
		ci_test_obj.option = test.findtext('option', default='')
		ci_test_obj.id = test.findtext('id', default='')
		ci_test_obj.nodes = test.findtext('nodes', default='').split(' ')
		ci_test_obj.svr_node = test.findtext('svr_node', default=None) if not force_local else 'localhost'
		ci_test_obj.svr_id = test.findtext('svr_id', default=None)
		ci_test_obj.cmd_prefix = test.findtext('cmd_prefix', default='')
		ci_test_obj.physim_time_threshold = test.findtext('physim_time_threshold', default='inf')
		ci_test_obj.physim_run_args = test.findtext('physim_run_args', default='')
		ci_test_obj.physim_test = test.findtext('physim_test', default='')
		ci_test_obj.proxy_commit = test.findtext('proxy_commit', default='')
		ci_test_obj.Build_eNB_args = test.findtext('Build_eNB_args', default='')
		ci_test_obj.kind = test.findtext('kind', default='')
		ci_test_obj.forced_workspace_cleanup = test.findtext('forced_workspace_cleanup', default='')

		# Handle postActions
		post_actions = test.findall('postActions/step')
		ci_test_obj.postActions = [step.get('id') for step in post_actions]

		# Store the test case object in the dictionary
		test_list[test_id] = ci_test_obj
		logging.info(f'Loaded test {test_id} with description: {ci_test_obj.desc}')

	logging.debug(f'Total number of test loaded from file {file}: {len(test_list)}')
	return test_list

def ExecuteActionWithParam(action, test):
	global RAN
	global HTML
	global CONTAINERS
	global SCA
	global PHYSIM
	global CLUSTER
	if action == 'Build_eNB' or action == 'Build_Image' or action == 'Build_Proxy' or action == "Build_Cluster_Image" or action == "Build_Run_Tests":
		RAN.Build_eNB_args=test.Build_eNB_args
		CONTAINERS.imageKind=test.kind
		forced_workspace_cleanup = test.forced_workspace_cleanup
		RAN.Build_eNB_forced_workspace_cleanup=False
		CONTAINERS.forcedWorkspaceCleanup=False
		CLUSTER.forcedWorkspaceCleanup = False
		if forced_workspace_cleanup is not None and re.match('true', forced_workspace_cleanup, re.IGNORECASE):
			RAN.Build_eNB_forced_workspace_cleanup = True
			CONTAINERS.forcedWorkspaceCleanup = True
			CLUSTER.forcedWorkspaceCleanup = True
		if (test.eNB_instance is None):
			RAN.eNB_instance=0
			CONTAINERS.eNB_instance=0
		else:
			RAN.eNB_instance=int(test.eNB_instance)
			CONTAINERS.eNB_instance=int(test.eNB_instance)
		if (test.eNB_serverId is None):
			RAN.eNB_serverId[RAN.eNB_instance]='0'
			CONTAINERS.eNB_serverId[RAN.eNB_instance]='0'
		else:
			RAN.eNB_serverId[RAN.eNB_instance]=test.eNB_serverId
			CONTAINERS.eNB_serverId[CONTAINERS.eNB_instance]=test.eNB_serverId
		if test.proxy_commit is not None:
			CONTAINERS.proxyCommit = test.proxy_commit
		if action == 'Build_eNB':
			success = cls_native.Native.Build(HTML.testCase_id, HTML, RAN.eNBIPAddress, RAN.eNBSourceCodePath, RAN.Build_eNB_args)
		elif action == 'Build_Image':
			success = CONTAINERS.BuildImage(HTML)
		elif action == 'Build_Proxy':
			success = CONTAINERS.BuildProxy(HTML)
		elif action == 'Build_Cluster_Image':
			success = CLUSTER.BuildClusterImage(HTML)
		elif action == 'Build_Run_Tests':
			success = CONTAINERS.BuildRunTests(HTML)

	elif action == 'Initialize_eNB':
		if test.rt_stats_cfg is None:
			RAN.datalog_rt_stats_file='datalog_rt_stats.default.yaml'
		else:
			RAN.datalog_rt_stats_file=test.rt_stats_cfg
		RAN.Initialize_eNB_args=test.Initialize_eNB_args

		if test.USRP_IPAddress is None:
			RAN.USRPIPAddress=''
		else:
			RAN.USRPIPAddress=test.USRP_IPAddress

		if (test.eNB_instance is None):
			RAN.eNB_instance=0
		else:
			RAN.eNB_instance=int(test.eNB_instance)

		if (test.eNB_serverId is None):
			RAN.eNB_serverId[RAN.eNB_instance]='0'
		else:
			RAN.eNB_serverId[RAN.eNB_instance]=test.eNB_serverId
			
		#local variable air_interface
		if (test.air_interface is None) or (test.air_interface.lower() not in ['nr','lte']):
			RAN.air_interface[RAN.eNB_instance] = 'lte-softmodem'
		else:
			RAN.air_interface[RAN.eNB_instance] = test.air_interface.lower() +'-softmodem'

		if test.cmd_prefix is not None: RAN.cmd_prefix = test.cmd_prefix
		success = RAN.InitializeeNB(HTML)

	elif action == 'Terminate_eNB':
		if (test.eNB_instance is None):
			RAN.eNB_instance=0
		else:
			RAN.eNB_instance=int(test.eNB_instance)
		if (test.eNB_serverId is None):
			RAN.eNB_serverId[RAN.eNB_instance]='0'
		else:
			RAN.eNB_serverId[RAN.eNB_instance]=test.eNB_serverId

		#retx checkers
		if (test.d_retx_th is not None):
			RAN.ran_checkers['d_retx_th'] = [
			    float(x) for x in test.d_retx_th.split(',') if x.strip() != ''
			]
		if (test.u_retx_th is not None):
			RAN.ran_checkers['u_retx_th'] = [
			    float(x) for x in test.u_retx_th.split(',') if x.strip() != ''
			]

		#local variable air_interface
		air_interface = test.air_interface		
		if (air_interface is None) or (air_interface.lower() not in ['nr','lte']):
			RAN.air_interface[RAN.eNB_instance] = 'lte-softmodem'
		else:
			RAN.air_interface[RAN.eNB_instance] = air_interface.lower() +'-softmodem'
		success = RAN.TerminateeNB(HTML)

	elif action == 'Initialize_UE' or action == 'Attach_UE' or action == 'Detach_UE' or action == 'Terminate_UE' or action == 'CheckStatusUE' or action == 'DataEnable_UE' or action == 'DataDisable_UE':
		if force_local:
			# Change all execution targets to localhost
			test.nodes = ['localhost'] * len(test.ue_ids)
		else:
			if test.findtext('nodes'):
				if len(test.ue_ids) != len(test.nodes):
					logging.error('Number of Nodes are not equal to the total number of UEs')
					sys.exit("Mismatch in number of Nodes and UIs")
			else:
				test.nodes = [None] * len(test.ue_ids)
		if action == 'Initialize_UE':
			success = test.InitializeUE(HTML)
		elif action == 'Attach_UE':
			success = test.AttachUE(HTML)
		elif action == 'Detach_UE':
			success = test.DetachUE(HTML)
		elif action == 'Terminate_UE':
			success = test.TerminateUE(HTML)
		elif action == 'CheckStatusUE':
			success = test.CheckStatusUE(HTML)
		elif action == 'DataEnable_UE':
			success = test.DataEnableUE(HTML)
		elif action == 'DataDisable_UE':
			success = test.DataDisableUE(HTML)

	elif action == 'Ping':
		if force_local:
			# Change all execution targets to localhost
			test.nodes = ['localhost'] * len(test.ue_ids)
		else:
			if test.nodes:
				if len(test.ue_ids) != len(test.nodes):
					logging.error('Number of Nodes are not equal to the total number of UEs')
					sys.exit("Mismatch in number of Nodes and UIs")
			else:
				test.nodes = [None] * len(test.ue_ids)
		ping_rttavg_threshold = test.ping_rttavg_threshold or ''
		success = test.Ping(HTML, CONTAINERS)

	elif action == 'Iperf' or action == 'Iperf2_Unidir':
		if force_local:
			# Change all execution targets to localhost
			test.nodes = ['localhost'] * len(test.ue_ids)
		else:
			if test.nodes:
				if len(test.ue_ids) != len(test.nodes):
					logging.error('Number of Nodes are not equal to the total number of UEs')
					sys.exit("Mismatch in number of Nodes and UIs")
			else:
				test.nodes = [None] * len(CiTestObj.ue_ids)
		if test.iperf_profile != 'balanced' and test.iperf_profile != 'unbalanced' and test.iperf_profile != 'single-ue':
			logging.error(f'test-case has wrong profile {test.iperf_profile}, forcing balanced')
			test.iperf_profile = 'balanced'
		if action == 'Iperf':
			success = test.Iperf(HTML, CONTAINERS)
		elif action == 'Iperf2_Unidir':
			success = test.Iperf2_Unidir(HTML, CONTAINERS)

	elif action == 'IdleSleep':
		success = cls_oaicitest.IdleSleep(HTML, int(test.idle_sleep_time_in_sec))

	elif action == 'Deploy_Run_PhySim':
		success = PHYSIM.Deploy_PhySim(HTML)

	elif action == 'DeployCoreNetwork' or action == 'UndeployCoreNetwork':
		core_op = getattr(cls_oaicitest.OaiCiTest, action)
		success = core_op(test.cn_id, HTML)

	elif action == 'Deploy_Object' or action == 'Undeploy_Object' or action == "Create_Workspace":
		if (test.eNB_instance is None):
			CONTAINERS.eNB_instance=0
		else:
			CONTAINERS.eNB_instance=int(test.eNB_instance)
		if (test.eNB_serverId is None):
			CONTAINERS.eNB_serverId[CONTAINERS.eNB_instance]='0'
		else:
			CONTAINERS.eNB_serverId[CONTAINERS.eNB_instance]=test.eNB_serverId
		if (test.yaml_path is not None):
			CONTAINERS.yamlPath[CONTAINERS.eNB_instance] = test.yaml_path
		if (test.d_retx_th is not None):
			# Split by commas and convert to floats only if the string is not empty
			CONTAINERS.ran_checkers['d_retx_th'] = [
			    float(x) for x in test.d_retx_th.split(',') if x.strip() != ''
			]
		if (test.u_retx_th is not None):
			# Split by commas and convert to floats only if the string is not empty
			CONTAINERS.ran_checkers['u_retx_th'] = [
			    float(x) for x in test.u_retx_th.split(',') if x.strip() != ''
			]
		if test.services is not None:
			CONTAINERS.services[CONTAINERS.eNB_instance] = test.services
		CONTAINERS.num_attempts = test.num_attempts
		CONTAINERS.deploymentTag = cls_containerize.CreateTag(CONTAINERS.ranCommitID, CONTAINERS.ranBranch, CONTAINERS.ranAllowMerge)
		if action == 'Deploy_Object':
			success = CONTAINERS.DeployObject(HTML)
		elif action == 'Undeploy_Object':
			success = CONTAINERS.UndeployObject(HTML, RAN)
		elif action == 'Create_Workspace':
			if force_local:
				# Do not create a working directory when running locally. Current repo directory will be used
				return True
			success = CONTAINERS.Create_Workspace(HTML)

	elif action == 'Run_Physim':
		physim_options = test.physim_run_args
		physim_test = test.physim_test
		physim_threshold = test.physim_time_threshold
		success = cls_native.Native.Run_Physim(HTML, RAN.eNBIPAddress, RAN.eNBSourceCodePath, physim_options, physim_test, physim_threshold)

	elif action == 'LicenceAndFormattingCheck':
		success = SCA.LicenceAndFormattingCheck(HTML)

	elif action == 'Cppcheck_Analysis':
		success = SCA.CppCheckAnalysis(HTML)

	elif action == 'Push_Local_Registry':
		success = CONTAINERS.Push_Image_to_Local_Registry(HTML, test.svr_id)

	elif action == 'Pull_Local_Registry' or action == 'Clean_Test_Server_Images':
		if force_local:
			# Do not pull or remove images when running locally. User is supposed to handle image creation & cleanup
			return True
		# hack: for FlexRIC, we need to overwrite the tag to use
		tag = None
		if len(test.images) == 1 and test.images[0] == "oai-flexric":
			tag = CONTAINERS.flexricTag
		if action == "Pull_Local_Registry":
			success = CONTAINERS.Pull_Image_from_Registry(HTML, test.svr_id, test.images, tag=tag)
		if action == "Clean_Test_Server_Images":
			success = CONTAINERS.Clean_Test_Server_Images(HTML, test.svr_id, test.images, tag=tag)

	elif action == 'Custom_Command':
		node = test.node
		if force_local:
			# Change all execution targets to localhost
			node = 'localhost'
		success = cls_oaicitest.Custom_Command(HTML, node, test.command, test.command_fail)

	elif action == 'Custom_Script':
		success = cls_oaicitest.Custom_Script(HTML, test.node, test.script, test.command_fail)

	elif action == 'Pull_Cluster_Image':
		success = CLUSTER.PullClusterImage(HTML, test.node, test.images)

	else:
		logging.warning(f"unknown action {action}, skip step")
		success = True # by default, we skip the step and print a warning

	return success

#check if given test is in list
#it is in list if one of the strings in 'list' is at the beginning of 'test'
def test_in_list(test, list):
	for check in list:
		check=check.replace('+','')
		if (test.startswith(check)):
			return True
	return False

def receive_signal(signum, frame):
	sys.exit(1)

def do_test(test):
	global task_set_succeeded
	task_set_succeeded = True
	logging.info(f'Executing test case: {test.testCase_id}')
	HTML.testCase_id = test.testCase_id
	HTML.desc = test.desc
	action = test.type
	if (CheckClassValidity(xml_class_list, action, id) == False):
		return
	test.ShowTestID()
	if not task_set_succeeded and not test.always_exec:
		msg = f"skipping test {test.testCase_id} due to prior error"
		logging.warning(msg)
		HTML.CreateHtmlTestRowQueue(msg, "SKIP", [])
		return
	try:
		test_succeeded = ExecuteActionWithParam(action, test)
		if not test_succeeded:
			logging.error(f"test ID {test.testCase_id} action {action} failed ({test_succeeded}), skipping next tests")
			task_set_succeeded = False
	except Exception as e:
		s = traceback.format_exc()
		logging.error(f'while running CI, an exception occurred:\n{s}')
		HTML.CreateHtmlTestRowQueue("N/A", 'KO', [f"CI test code encountered an exception:\n{s}"])
		task_set_succeeded = False
		return

#-----------------------------------------------------------
# MAIN PART
#-----------------------------------------------------------

#loading xml action list from yaml
import yaml
xml_class_list_file='xml_class_list.yml'
if (os.path.isfile(xml_class_list_file)):
	yaml_file=xml_class_list_file
elif (os.path.isfile('ci-scripts/'+xml_class_list_file)):
	yaml_file='ci-scripts/'+xml_class_list_file
else:
	logging.error("XML action list yaml file cannot be found")
	sys.exit("XML action list yaml file cannot be found")

with open(yaml_file,'r') as f:
    # The FullLoader parameter handles the conversion-$
    #from YAML scalar values to Python dictionary format$
    xml_class_list = yaml.load(f,Loader=yaml.FullLoader)

mode = ''

CiTestObj = cls_oaicitest.OaiCiTest()
 
RAN = ran.RANManagement()
HTML = cls_oai_html.HTMLManagement()
CONTAINERS = cls_containerize.Containerize()
SCA = cls_static_code_analysis.StaticCodeAnalysis()
PHYSIM = cls_physim1.PhySim()
CLUSTER = cls_cluster.Cluster()

#-----------------------------------------------------------
# Parsing Command Line Arguments
#-----------------------------------------------------------

import args_parse
# Force local execution, move all execution targets to localhost
force_local = False
py_param_file_present, py_params, mode, force_local = args_parse.ArgsParse(sys.argv,CiTestObj,RAN,HTML,CONTAINERS,HELP,SCA,PHYSIM,CLUSTER)



#-----------------------------------------------------------
# TEMPORARY params management (UNUSED)
#-----------------------------------------------------------
#temporary solution for testing:
if py_param_file_present == True:
	AssignParams(py_params)

#-----------------------------------------------------------
# mode amd XML class (action) analysis
#-----------------------------------------------------------
cwd = os.getcwd()

if re.match('^TerminateeNB$', mode, re.IGNORECASE):
	if RAN.eNBIPAddress == '' or RAN.eNBUserName == '' or RAN.eNBPassword == '':
		HELP.GenericHelp(CONST.Version)
		sys.exit('Insufficient Parameter')
	if RAN.eNBIPAddress == 'none':
		sys.exit(0)
	RAN.eNB_instance=0
	RAN.eNB_serverId[0]='0'
	RAN.eNBSourceCodePath='/tmp/'
	RAN.TerminateeNB(HTML)
elif re.match('^TerminateHSS$', mode, re.IGNORECASE):
	logging.warning("Option TerminateHSS ignored")
elif re.match('^TerminateMME$', mode, re.IGNORECASE):
	logging.warning("Option TerminateMME ignored")
elif re.match('^TerminateSPGW$', mode, re.IGNORECASE):
	logging.warning("Option TerminateSPGW ignored")
elif re.match('^LogCollectBuild$', mode, re.IGNORECASE):
	if (RAN.eNBIPAddress == '' or RAN.eNBUserName == '' or RAN.eNBPassword == '' or RAN.eNBSourceCodePath == '') and (CiTestObj.UEIPAddress == '' or CiTestObj.UEUserName == '' or CiTestObj.UEPassword == '' or CiTestObj.UESourceCodePath == ''):
		HELP.GenericHelp(CONST.Version)
		sys.exit('Insufficient Parameter')
	if RAN.eNBIPAddress == 'none':
		sys.exit(0)
	CiTestObj.LogCollectBuild(RAN)
elif re.match('^LogCollecteNB$', mode, re.IGNORECASE):
	if RAN.eNBIPAddress == '' or RAN.eNBUserName == '' or RAN.eNBPassword == '' or RAN.eNBSourceCodePath == '':
		HELP.GenericHelp(CONST.Version)
		sys.exit('Insufficient Parameter')
	if os.path.isdir('cmake_targets/log'):
		cmd = 'zip -r enb.log.' + RAN.BuildId + '.zip cmake_targets/log'
		logging.info(cmd)
		try:
			zipStatus = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, universal_newlines=True, timeout=60)
		except subprocess.CalledProcessError as e:
			logging.error("Command '{}' returned non-zero exit status {}.".format(e.cmd, e.returncode))
			logging.error("Error output:\n{}".format(e.output))
		sys.exit(0)
	RAN.LogCollecteNB()
elif re.match('^LogCollectHSS$', mode, re.IGNORECASE):
	logging.warning("Option LogCollectHSS ignored")
elif re.match('^LogCollectMME$', mode, re.IGNORECASE):
	logging.warning("Option LogCollectMME ignored")
elif re.match('^LogCollectSPGW$', mode, re.IGNORECASE):
	logging.warning("Option LogCollectSPGW ignored")
elif re.match('^LogCollectPing$', mode, re.IGNORECASE):
	logging.warning("Option LogCollectPing ignored")
elif re.match('^LogCollectIperf$', mode, re.IGNORECASE):
	logging.warning("Option LogCollectIperf ignored")
elif re.match('^LogCollectOAIUE$', mode, re.IGNORECASE):
	logging.warning("Option LogCollectOAIUE ignored")
elif re.match('^InitiateHtml$', mode, re.IGNORECASE):
	count = 0
	foundCount = 0
	while (count < HTML.nbTestXMLfiles):
		#xml_test_file = cwd + "/" + CiTestObj.testXMLfiles[count]
		xml_test_file = sys.path[0] + "/" + CiTestObj.testXMLfiles[count]
		if (os.path.isfile(xml_test_file)):
			try:
				xmlTree = ET.parse(xml_test_file)
			except Exception as e:
				print(f"Error: {e} while parsing file: {xml_test_file}.")
			xmlRoot = xmlTree.getroot()
			HTML.htmlTabRefs.append(xmlRoot.findtext('htmlTabRef',default='test-tab-' + str(count)))
			HTML.htmlTabNames.append(xmlRoot.findtext('htmlTabName',default='test-tab-' + str(count)))
			HTML.htmlTabIcons.append(xmlRoot.findtext('htmlTabIcon',default='info-sign'))
			foundCount += 1
		count += 1
	if foundCount != HTML.nbTestXMLfiles:
		HTML.nbTestXMLfiles=foundCount
	
	HTML.CreateHtmlHeader()
elif re.match('^FinalizeHtml$', mode, re.IGNORECASE):
	logging.info('\u001B[1m----------------------------------------\u001B[0m')
	logging.info('\u001B[1m  Creating HTML footer \u001B[0m')
	logging.info('\u001B[1m----------------------------------------\u001B[0m')

	HTML.CreateHtmlFooter(CiTestObj.finalStatus)
elif re.match('^TesteNB$', mode, re.IGNORECASE) or re.match('^TestUE$', mode, re.IGNORECASE):
	logging.info('\u001B[1m----------------------------------------\u001B[0m')
	logging.info('\u001B[1m  Starting Scenario: ' + CiTestObj.testXMLfiles[0] + '\u001B[0m')
	logging.info('\u001B[1m----------------------------------------\u001B[0m')
	if re.match('^TesteNB$', mode, re.IGNORECASE):
		if RAN.eNBIPAddress == '' or RAN.ranRepository == '' or RAN.ranBranch == '' or RAN.eNBUserName == '' or RAN.eNBPassword == '' or RAN.eNBSourceCodePath == '':
			HELP.GenericHelp(CONST.Version)
			if RAN.ranRepository == '':
				HELP.GitSrvHelp(RAN.ranRepository, RAN.ranBranch, RAN.ranCommitID, RAN.ranAllowMerge, RAN.ranTargetBranch)
			if RAN.eNBIPAddress == ''  or RAN.eNBUserName == '' or RAN.eNBPassword == '' or RAN.eNBSourceCodePath == '':
				HELP.eNBSrvHelp(RAN.eNBIPAddress, RAN.eNBUserName, RAN.eNBPassword, RAN.eNBSourceCodePath)
			sys.exit('Insufficient Parameter')
	else:
		if CiTestObj.UEIPAddress == '' or CiTestObj.ranRepository == '' or CiTestObj.ranBranch == '' or CiTestObj.UEUserName == '' or CiTestObj.UEPassword == '' or CiTestObj.UESourceCodePath == '':
			HELP.GenericHelp(CONST.Version)
			sys.exit('UE: Insufficient Parameter')


	file = cwd + '/xml_files/common.xml'
	common_tests = LoadTests(file)
	print(common_tests)

	#read test_case_list.xml file
	# if no parameters for XML file, use default value
	if (HTML.nbTestXMLfiles != 1):
		xml_test_file = cwd + "/test_case_list.xml"
	else:
		xml_test_file = cwd + "/" + CiTestObj.testXMLfiles[0]

	todo_tests = LoadTests(xml_test_file)

	#get the list of tests to be done

	signal.signal(signal.SIGUSR1, receive_signal)

	HTML.CreateHtmlTabHeader()

	HTML.startTime=int(round(time.time() * 1000))

	for test_id, test in todo_tests.items():
		do_test(test)
		if task_set_succeeded is not True:
			break

		# Handle postActions
		postActions = test.postActions
		if postActions is not None:
			for step in postActions:
				logging.info(f'Found postAction step: {step}')

				for common_test_id, common_test_obj in common_tests.items():
					test_id = common_test_obj.testCase_id
					if test_id != step:
						continue
					# Retrieve the corresponding common test from the dictionary
					common_test = common_tests.get(step)
					if common_test:
						do_test(common_test_obj)
						if task_set_succeeded is not True:
							break

	if not task_set_succeeded:
		logging.error('\u001B[1;37;41mScenario failed\u001B[0m')
		HTML.CreateHtmlTabFooter(False)
		sys.exit('Failed Scenario')
	else:
		logging.info('\u001B[1;37;42mScenario passed\u001B[0m')
		HTML.CreateHtmlTabFooter(True)
elif re.match('^LoadParams$', mode, re.IGNORECASE):
	pass
else:
	HELP.GenericHelp(CONST.Version)
	sys.exit('Invalid mode')
sys.exit(0)
