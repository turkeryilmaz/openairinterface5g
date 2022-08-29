/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <netinet/in.h>
#include <netinet/sctp.h>

#include <arpa/inet.h>

#include "assertions.h"
#include "common/utils/system.h"
#include "common/utils/LOG/ss-log.h"
#include "queue.h"
#include "sctp_common.h"

#include "intertask_interface.h"
#include "common/ran_context.h"
#include "LTE_UL-CCCH-Message.h"
#include "LTE_DL-CCCH-Message.h"
#include "LTE_UL-DCCH-Message.h"
#include "LTE_DL-DCCH-Message.h"

#include "acpSysSrb.h"
#include "acpSys.h"
#include "ss_eNB_context.h"
#include "ss_eNB_vt_timer_task.h"
extern RAN_CONTEXT_t RC;
SSConfigContext_t SS_context;
static acpCtx_t ctx_srb_g = NULL;
static uint16_t rnti_g = 0;
static instance_t instance_g = 0;
uint16_t ss_rnti_g = 0;

enum MsgUserId
{
	// user defined IDs should be an int number >= 1
	MSG_SysProcess_userId = 1,
	MSG_SysSrbProcessFromSS_userId,
	MSG_SysSrbProcessToSS_userId,
};

typedef enum
{
	UndefinedMsg = 0,
	EutraRrcPduReq = 1
} sidl_msg_id;

static unsigned char *buffer = NULL;
static const size_t size = 16 * 1024;
uint8_t lttng_sdu[SDU_SIZE];
//------------------------------------------------------------------------------

/*
 * Function : ss_send_srb_data
 * Description: Function to send response to the TTCN/SIDL Client
 * In :
 * req  - Request received from the TTCN
 * Out:
 * newState: No impack on the State
 *
 */
static void ss_send_srb_data(ss_rrc_pdu_ind_t *pdu_ind)
{
	struct EUTRA_RRC_PDU_IND ind = {};
	uint32_t status = 0;
        LTE_UL_DCCH_Message_t               *ul_dcch_msg = NULL;
        LTE_UL_CCCH_Message_t               *ul_ccch_msg = NULL;

	LOG_A(ENB_SS, "[SS_SRB] Reported rrc sdu_size:%d \t srb_id %d rnti %d\n", pdu_ind->sdu_size, pdu_ind->srb_id, pdu_ind->rnti);
	DevAssert(pdu_ind != NULL);
	DevAssert(pdu_ind->sdu_size >= 0);
	DevAssert(pdu_ind->srb_id >= 0);
	rnti_g = pdu_ind->rnti;
	ss_rnti_g = rnti_g;
	size_t msgSize = size;
	memset(&ind, 0, sizeof(ind));
	ind.Common.CellId = SS_context.eutra_cellId;

	// Populated the Routing Info
	ind.Common.RoutingInfo.d = RoutingInfo_Type_RadioBearerId;
	ind.Common.RoutingInfo.v.RadioBearerId.d = RadioBearerId_Type_Srb;
	ind.Common.RoutingInfo.v.RadioBearerId.v.Srb = pdu_ind->srb_id;

	// Populated the Timing Info
	ind.Common.TimingInfo.d = TimingInfo_Type_SubFrame;
	ind.Common.TimingInfo.v.SubFrame.SFN.d = SystemFrameNumberInfo_Type_Number;
	ind.Common.TimingInfo.v.SubFrame.SFN.v.Number = pdu_ind->frame;

	ind.Common.TimingInfo.v.SubFrame.Subframe.d = SubFrameInfo_Type_Number;
	ind.Common.TimingInfo.v.SubFrame.Subframe.v.Number = pdu_ind->subframe;

	ind.Common.TimingInfo.v.SubFrame.HSFN.d = SystemFrameNumberInfo_Type_Number;
	ind.Common.TimingInfo.v.SubFrame.HSFN.v.Number = 0;

	ind.Common.TimingInfo.v.SubFrame.Slot.d = SlotTimingInfo_Type_Any;
	ind.Common.TimingInfo.v.SubFrame.Slot.v.Any = true;

	ind.Common.Status.d = IndicationStatus_Type_Ok;
	ind.Common.Status.v.Ok = true;

	ind.Common.RlcBearerRouting.d = true;
	ind.Common.RlcBearerRouting.v.d = RlcBearerRouting_Type_EUTRA;
	ind.Common.RlcBearerRouting.v.v.EUTRA = SS_context.eutra_cellId;

	/* Populate and Send the EUTRA RRC PDU IND to Client */
	if (pdu_ind->srb_id == 0)
	{
		uper_decode(
                      NULL,
                      &asn_DEF_LTE_UL_CCCH_Message,
                      (void **)&ul_ccch_msg,
                      pdu_ind->sdu,
                      pdu_ind->sdu_size,
                      0,
                      0);
		memcpy(lttng_sdu, pdu_ind->sdu, pdu_ind->sdu_size);
		LOG_P(OAILOG_DEBUG, "UL_CCCH_Message", lttng_sdu, pdu_ind->sdu_size);

		xer_fprint(stdout, &asn_DEF_LTE_UL_CCCH_Message, (void *)ul_ccch_msg);
		ind.RrcPdu.d = RRC_MSG_Indication_Type_Ccch;
		ind.RrcPdu.v.Ccch.d = pdu_ind->sdu_size;
		ind.RrcPdu.v.Ccch.v = pdu_ind->sdu;
	}
	else
	{
		uper_decode(
                      NULL,
                      &asn_DEF_LTE_UL_DCCH_Message,
                      (void **)&ul_dcch_msg,
                      pdu_ind->sdu,
                      pdu_ind->sdu_size,
                      0,
                      0);

		xer_fprint(stdout, &asn_DEF_LTE_UL_DCCH_Message, (void *)ul_dcch_msg);
		memcpy(lttng_sdu, pdu_ind->sdu, pdu_ind->sdu_size);
		LOG_P(OAILOG_DEBUG, "UL_DCCH_Message", lttng_sdu, pdu_ind->sdu_size);
		ind.RrcPdu.d = RRC_MSG_Indication_Type_Dcch;
		ind.RrcPdu.v.Dcch.d = pdu_ind->sdu_size;
		ind.RrcPdu.v.Dcch.v = pdu_ind->sdu;

	}

	/* Encode message
   */
	if (acpSysSrbProcessToSSEncSrv(ctx_srb_g, buffer, &msgSize, &ind) != 0)
	{
		LOG_A(ENB_SS, "[SS_SRB][EUTRA_RRC_PDU_IND] acpSysSrbProcessToSSEncSrv Failure\n");
		return;
	}
	LOG_A(ENB_SS, "[SS_SRB][EUTRA_RRC_PDU_IND] Buffer msgSize=%d (!!2) to EUTRACell %d", (int)msgSize,SS_context.eutra_cellId);

	/* Send message
   */
	status = acpSendMsg(ctx_srb_g, msgSize, buffer);
	if (status != 0)
	{
		LOG_A(ENB_SS, "[SS_SRB][EUTRA_RRC_PDU_IND] acpSendMsg failed. Error : %d on fd: %d\n", status, acpGetSocketFd(ctx_srb_g));
		return;
	}
	else
	{
		LOG_A(ENB_SS, "[SS_SRB][EUTRA_RRC_PDU_IND] acpSendMsg Success \n");
	}
}

/*
 * Function : ss_task_handle_rrc_pdu_req
 * Description: Function to handle SS_RRC_PDU_REQ from the TTCN
 * and further send it to RRC eNB task
 * In :
 * req  - Request received from the TTCN
 * Out:
 * newState: No impack on the State
 *
 */
static void ss_task_handle_rrc_pdu_req(struct EUTRA_RRC_PDU_REQ *req)
{
	assert(req);
        LTE_DL_DCCH_Message_t *dl_dcch_msg=NULL;
        LTE_DL_CCCH_Message_t *dl_ccch_msg=NULL;
	MessageDef *message_p = itti_alloc_new_message(TASK_RRC_ENB, 0, SS_RRC_PDU_REQ);
	assert(message_p);
	instance_g = 0;
	if (message_p)
	{
		/* Populate the message and send to SS */
		SS_RRC_PDU_REQ(message_p).srb_id = req->Common.RoutingInfo.v.RadioBearerId.v.Srb;
		memset(SS_RRC_PDU_REQ(message_p).sdu, 0, SDU_SIZE);
		if (req->RrcPdu.d == RRC_MSG_Request_Type_Ccch)
		{
			SS_RRC_PDU_REQ(message_p).sdu_size = req->RrcPdu.v.Ccch.d;
			memcpy(SS_RRC_PDU_REQ(message_p).sdu, req->RrcPdu.v.Ccch.v, req->RrcPdu.v.Ccch.d);
			uper_decode(NULL,
                                    &asn_DEF_LTE_DL_CCCH_Message,
                                    (void **)&dl_ccch_msg,
                                    (uint8_t *)SS_RRC_PDU_REQ(message_p).sdu,
                                    SS_RRC_PDU_REQ(message_p).sdu_size,0,0);

			xer_fprint(stdout,&asn_DEF_LTE_DL_CCCH_Message,(void *)dl_ccch_msg);
			memcpy(lttng_sdu, SS_RRC_PDU_REQ(message_p).sdu, SS_RRC_PDU_REQ(message_p).sdu_size);
			LOG_P(OAILOG_DEBUG, "DL_CCCH_Message", lttng_sdu, SS_RRC_PDU_REQ(message_p).sdu_size);
		}
		else
		{
			SS_RRC_PDU_REQ(message_p).sdu_size = req->RrcPdu.v.Dcch.d;
			memcpy(SS_RRC_PDU_REQ(message_p).sdu, req->RrcPdu.v.Dcch.v, req->RrcPdu.v.Dcch.d);
			uper_decode(NULL,
                                    &asn_DEF_LTE_DL_DCCH_Message,
                                    (void **)&dl_dcch_msg,
                                    (uint8_t *)SS_RRC_PDU_REQ(message_p).sdu,
                                    SS_RRC_PDU_REQ(message_p).sdu_size,0,0);

			xer_fprint(stdout,&asn_DEF_LTE_DL_DCCH_Message,(void *)dl_dcch_msg);
			memcpy(lttng_sdu, SS_RRC_PDU_REQ(message_p).sdu, SS_RRC_PDU_REQ(message_p).sdu_size);

			LOG_P(OAILOG_DEBUG, "DL_DCCH_Message", lttng_sdu, SS_RRC_PDU_REQ(message_p).sdu_size);
		}

		LOG_A(ENB_SS, "[SS_SRB][EUTRA_RRC_PDU_REQ] sending to TASK_RRC_ENB: {srb: %d, ch: %s, qty: %d rnti %d}\n",
			  SS_RRC_PDU_REQ(message_p).srb_id,
			  req->RrcPdu.d == RRC_MSG_Request_Type_Ccch ? "CCCH" : "DCCH", SS_RRC_PDU_REQ(message_p).sdu_size ,rnti_g);

		SS_RRC_PDU_REQ(message_p).rnti = rnti_g;
        uint8_t msg_queued = 0;
		if (req->Common.TimingInfo.d == TimingInfo_Type_SubFrame)
		{
			ss_set_timinfo_t tinfo, timer_tinfo;
			tinfo.sfn = req->Common.TimingInfo.v.SubFrame.SFN.v.Number;
			tinfo.sf = req->Common.TimingInfo.v.SubFrame.Subframe.v.Number;
			timer_tinfo = tinfo;
			msg_queued = msg_can_be_queued(tinfo, &timer_tinfo);

			LOG_A(ENB_SS,"VT_TIMER SRB  task received MSG for future  SFN %d , SF %d\n",tinfo.sfn,tinfo.sf);

			if(msg_queued)
			{
				 msg_queued = vt_timer_setup(timer_tinfo, TASK_RRC_ENB, instance_g,message_p);
			}
			LOG_A(ENB_SS, "RRC_PDU Queued as the scheduled SFN is %d SF: %d and curr SFN %d , SF %d",
					tinfo.sfn,tinfo.sf, SS_context.sfn,SS_context.sf);

		}
		if (!msg_queued)
		{
			int send_res = itti_send_msg_to_task(TASK_RRC_ENB, instance_g, message_p);
			if (send_res < 0)
			{
				LOG_A(ENB_SS, "[SS_SRB] Error in itti_send_msg_to_task");
			}

			LOG_A(ENB_SS, "Send res: %d", send_res);
		}
	}
}

/*
 * Function : ss_eNB_read_from_srb_socket
 * Description: Function to received message from SRB Socket
 * In :
 * req  - Request received from the TTCN
 * Out:
 * newState: No impack on the State
 *
 */
static bool isConnected = false;
static inline void
ss_eNB_read_from_srb_socket(acpCtx_t ctx)
{
	size_t msgSize = size; //2

	while (1)
	{
		int userId = acpRecvMsg(ctx, &msgSize, buffer);
		LOG_A(ENB_SS, "[SS_SRB] Received msgSize=%d, userId=%d\n", (int)msgSize, userId);

		// Error handling
		if (userId < 0)
		{
			if (userId == -ACP_ERR_SERVICE_NOT_MAPPED)
			{
				// Message not mapped to user id,
				// this error should not appear on server side for the messages received from clients
			}
			else if (userId == -ACP_ERR_SIDL_FAILURE)
			{
				// Server returned service error,
				// this error should not appear on server side for the messages received from clients
				SidlStatus sidlStatus = -1;
				acpGetMsgSidlStatus(msgSize, buffer, &sidlStatus);
			}
			else if (userId == -ACP_PEER_DISCONNECTED){
    			LOG_A(GNB_APP, "[SS_SRB] Peer ordered shutdown\n");
				isConnected = false;
            } 
            else if (userId == -ACP_PEER_CONNECTED){
	            LOG_A(GNB_APP, "[SS_SRB] Peer connection established\n");
				isConnected = true;
            } 
			else
			{
				LOG_A(ENB_SS, "[SS_SRB] Invalid userId: %d \n", userId);
				break;
			}
		}

		if (userId == 0)
		{
			// No message (timeout on socket)
			if (isConnected == true){
				break;
			}
		}
		else if (MSG_SysSrbProcessFromSS_userId == userId)
		{
			struct EUTRA_RRC_PDU_REQ *req = NULL;
			LOG_A(ENB_SS, "[SS_SRB][EUTRA_RRC_PDU_REQ] EUTRA_RRC_PDU_REQ Received \n");
			// Got the message
			if (acpSysSrbProcessFromSSDecSrv(ctx, buffer, msgSize, &req) != 0)
			{
				LOG_A(ENB_SS, "[SS_SRB][EUTRA_RRC_PDU_REQ] acpSysSrbProcessFromSSDecSrv Failed\n");
				break;
			}
			if (RC.ss.ss_cell_list[0].State >= SS_STATE_CELL_ACTIVE)
			{
				ss_task_handle_rrc_pdu_req(req);
			}
			else
			{
				LOG_A(ENB_SS, "ERROR [SS_SRB][EUTRA_RRC_PDU_REQ] received in SS state %d \n", RC.ss.ss_cell_list[0].State);
			}

			acpSysSrbProcessFromSSFreeSrv(req);
			return;
		}
		else if (MSG_SysSrbProcessToSS_userId == userId)
		{
			LOG_A(ENB_SS, "[SS_SRB][EUTRA_RRC_PDU_IND] EUTRA_RRC_PDU_IND Received; ignoring \n");
			break;
		}
	}
}

/*
 * Function : ss_eNB_srb_init
 * Description: Function handles for initilization of SRB task
 * In :
 * req :
 * Out:
 * newState: No impack on the State
 *
 */
void ss_eNB_srb_init(void)
{
	IpAddress_t ipaddr;
	LOG_A(ENB_SS, "[SS_SRB] Starting System Simulator SRB Thread \n");

	const char *hostIp;
	hostIp = RC.ss.hostIp;
	acpConvertIp(hostIp, &ipaddr);

	// Port number
	int port = RC.ss.Srbport;

	const struct acpMsgTable msgTable[] = {
		{"SysSrbProcessFromSS", MSG_SysSrbProcessFromSS_userId},
		{"SysSrbProcessToSS", MSG_SysSrbProcessToSS_userId},
		{"SysProcess", MSG_SysProcess_userId},
		// The last element should be NULL
		{NULL, 0}};

	// Arena size to decode received message
	const size_t aSize = 32 * 1024;

	// Start listening server and get ACP context,
	// after the connection is performed, we can use all services
	int ret = acpServerInitWithCtx(ipaddr, port, msgTable, aSize, &ctx_srb_g);
	if (ret < 0)
	{
		LOG_A(ENB_SS, "[SS_SRB] Connection failure err=%d\n", ret);
		return;
	}
	int fd1 = acpGetSocketFd(ctx_srb_g);
	LOG_A(ENB_SS, "[SS_SRB] Connection performed : %d\n", fd1);

	buffer = (unsigned char *)acpMalloc(size);
	assert(buffer);

	itti_subscribe_event_fd(TASK_SS_SRB, fd1);

	itti_mark_task_ready(TASK_SS_SRB);
}

/*
 * Function : ss_eNB_srb_process_itti_msg
 * Description: Funtion Handles the ITTI
 * message received from the TTCN on SYS Port
 * In :
 * req  - ITTI message received from the TTCN via PORTMAN
 * Out:
 * newState: No impact on state machine.
 *
 */
void *ss_eNB_srb_process_itti_msg(void *notUsed)
{
	MessageDef *received_msg = NULL;
	int result = 0;

	itti_receive_msg(TASK_SS_SRB, &received_msg);

	/* Check if there is a packet to handle */
	if (received_msg != NULL)
	{
		switch (ITTI_MSG_ID(received_msg))
		{
		case SS_RRC_PDU_IND:
		{
			task_id_t origin_task = ITTI_MSG_ORIGIN_ID(received_msg);

			if (origin_task == TASK_SS_PORTMAN)
			{
				LOG_D(ENB_APP, "[SS_SRB] DUMMY WAKEUP receviedfrom PORTMAN state %d \n", RC.ss.ss_cell_list[0].State);
			}
			else
			{
				LOG_A(ENB_SS, "[SS_SRB] Received SS_RRC_PDU_IND from RRC\n");
				if (RC.ss.ss_cell_list[0].State >= SS_STATE_CELL_ACTIVE)
				{
					instance_g = ITTI_MSG_DESTINATION_INSTANCE(received_msg);
					ss_send_srb_data(&received_msg->ittiMsg.ss_rrc_pdu_ind);
				}
				else
				{
					LOG_A(ENB_SS, "ERROR [SS_SRB][EUTRA_RRC_PDU_IND] received in SS state %d \n", RC.ss.ss_cell_list[0].State);
				}
			}

			result = itti_free(ITTI_MSG_ORIGIN_ID(received_msg), received_msg);
			AssertFatal(result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);
                        received_msg = NULL;
		}
		break;

		case TERMINATE_MESSAGE:
			LOG_A(ENB_SS, "[SS_SRB] Received TERMINATE_MESSAGE \n");
			itti_exit_task();
			break;

		default:
			LOG_A(ENB_SS, "[SS_SRB] Received unhandled message %d:%s\n",
				  ITTI_MSG_ID(received_msg), ITTI_MSG_NAME(received_msg));
			break;
		}
	}
	return NULL;
}

/*
 * Function : ss_eNB_srb_task
 * Description: Funtion Handles the SRB Task
 * In :
 * req :
 * Out:
 * newState: No impact on state machine.
 *
 */
void *ss_eNB_srb_task(void *arg)
{
	while (1)
	{
		//LOG_A(ENB_SS,"[SS_SRB] Inside ss_eNB_srb_task \n");
		(void)ss_eNB_srb_process_itti_msg(NULL);
	}
	//acpFree(buffer);

	return NULL;
}
void *ss_eNB_srb_acp_task(void *arg)
{
	// printf("\n SRB ACP Task\n");
	ss_eNB_srb_init();
	while (1)
	{
		// printf("\nInside while srb acp task \n");
		ss_eNB_read_from_srb_socket(ctx_srb_g);
	}
	acpFree(buffer);
}
