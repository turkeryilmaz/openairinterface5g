import sys
import logging
logging.basicConfig(
	level=logging.DEBUG,
	stream=sys.stdout,
	format="[%(asctime)s] %(levelname)8s: %(message)s"
)
import os
os.system(f'rm -rf cmake_targets')
os.system(f'mkdir -p cmake_targets/log')
import unittest
import tempfile

sys.path.append('./') # to find OAI imports below
import cls_oai_html
from cls_ci_helper import TestCaseCtx
import cls_oaicitest
import cls_cmd

class TestPingIperf(unittest.TestCase):
	def setUp(self):
		self.html = cls_oai_html.HTMLManagement()
		self.html.testCaseId = "000000"
		self.ci = cls_oaicitest.OaiCiTest()
		self.ci.ue_ids = ["test"]
		self.ci.nodes = ["localhost"]
		self.ctx = TestCaseCtx.Default(tempfile.mkdtemp())
	def tearDown(self):
		with cls_cmd.LocalCmd() as c:
			c.run(f'rm -rf {self.ctx.logPath}')

	def test_ping(self):
		self.ci.ping_args = "-c3"
		self.ci.ping_packetloss_threshold = "0"
		self.ci.svr_id = "test"
		infra_file = "tests/config/infra_ping_iperf.yaml"
		# TODO Should need nothing but options and UE(s) to use
		success = self.ci.Ping(self.ctx, self.html, infra_file=infra_file)
		self.assertTrue(success)

	def test_iperf(self):
		# note: needs to be five seconds because Iperf() adds -O 3, so if it is
		# too short, the server is terminated before the client loaded
		# everything
		self.ci.iperf_args = "-u -t 5 -b 1M -R"
		self.ci.svr_id = "test"
		self.ci.svr_node = "localhost"
		self.ci.iperf_packetloss_threshold = "0"
		self.ci.iperf_bitrate_threshold = "0"
		self.ci.iperf_profile = "balanced"
		infra_file = "tests/config/infra_ping_iperf.yaml"
		# TODO Should need nothing but options and UE(s) to use
		success = self.ci.Iperf(self.ctx, self.html, infra_file=infra_file)
		self.assertTrue(success)

	def test_iperf2_unidir(self):
		self.ci.iperf_args = "-u -t 5 -b 1M"
		self.ci.svr_id = "test"
		self.ci.svr_node = "localhost"
		self.ci.iperf_packetloss_threshold = "0"
		self.ci.iperf_bitrate_threshold = "0"
		self.ci.iperf_profile = "balanced"
		infra_file = "tests/config/infra_ping_iperf.yaml"
		# TODO Should need nothing but options and UE(s) to use
		success = self.ci.Iperf2_Unidir(self.ctx, self.html, infra_file=infra_file)
		self.assertTrue(success)

if __name__ == '__main__':
	unittest.main()
