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

import logging
import re
import os

import cls_cmd
import cls_oai_html
import cls_analysis
import constants as CONST
from cls_ci_helper import archiveArtifact

DPDK_PATH = '/opt/dpdk-t2-22.11.0'

class Native():

	def Build(ctx, node, test_case, HTML, directory, options):
		logging.debug(f'Building on server: {node}')
		HTML.testCase_id = test_case

		with cls_cmd.getConnection(node) as ssh:
			base = f"{directory}/cmake_targets"
			ret = ssh.run(f"C_INCLUDE_PATH={DPDK_PATH}/include/ PKG_CONFIG_PATH={DPDK_PATH}/lib64/pkgconfig/ {base}/build_oai {options} > {base}/build_oai.log", timeout=900)
			success = ret.returncode == 0
			logs = ssh.run(f"cat {base}/build_oai.log", silent=True)
			logging.debug(f"build finished with code {ret.returncode}")

			archiveArtifact(ssh, ctx, f'{base}/build_oai.log')

			# check if build artifacts are there
			# NOTE: build_oai should fail with exit code if it could not build, but it does not
			build_dir = f"{base}/ran_build/build"
			build_gnb = re.search('--gNB', options) is not None
			if build_gnb:
				success = success and ssh.run(f"ls {build_dir}/nr-softmodem").returncode == 0
			build_enb = re.search('--eNB', options) is not None
			if build_enb:
				success = success and ssh.run(f"ls {build_dir}/lte-softmodem").returncode == 0

		if success:
			logging.info('\u001B[1m Building OAI Pass\u001B[0m')
			HTML.CreateHtmlTestRow(options, 'OK', CONST.ALL_PROCESSES_OK)
		else:
			logging.error('\u001B[1m Building OAI Failed\u001B[0m')
			HTML.CreateHtmlTestRow(options, 'KO', CONST.ALL_PROCESSES_OK)
		return success
