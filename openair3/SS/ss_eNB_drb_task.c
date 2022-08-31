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
#include "queue.h"
#include "sctp_common.h"

#include "intertask_interface.h"
#include "common/ran_context.h"

#include "acpDrb.h"
#include "ss_eNB_context.h"

extern RAN_CONTEXT_t RC;
extern uint16_t ss_rnti_g;
static acpCtx_t ctx_drb_g = NULL;
SSConfigContext_t SS_context;

static unsigned char *buffer = NULL;
static const size_t size = 16 * 1024;
static instance_t instance_g = 0;

enum MsgUserId
{
        // user defined IDs should be an int number >= 1
        MSG_DrbProcessFromSS_userId = 1,
        MSG_DrbProcessToSS_userId,
};

static void ss_send_drb_data(ss_drb_pdu_ind_t *pdu_ind){
	struct DRB_COMMON_IND ind = {};
        uint32_t status = 0;

	LOG_A(ENB_APP, "[SS_DRB] Reported drb sdu_size:%d \t drb_id %d\n", pdu_ind->sdu_size, pdu_ind->drb_id);

	DevAssert(pdu_ind != NULL);
        DevAssert(pdu_ind->sdu_size >= 0);
        DevAssert(pdu_ind->drb_id >= 0);

	size_t msgSize = size;
        memset(&ind, 0, sizeof(ind));

	ind.Common.CellId = SS_context[0].eutra_cellId;

	//Populated the Routing Info
	ind.Common.RoutingInfo.d = RoutingInfo_Type_RadioBearerId;
	ind.Common.RoutingInfo.v.RadioBearerId.d = RadioBearerId_Type_Drb;
	ind.Common.RoutingInfo.v.RadioBearerId.v.Drb = pdu_ind->drb_id;

	//Populated the Timing Info
	ind.Common.TimingInfo.d = TimingInfo_Type_SubFrame;
	ind.Common.TimingInfo.v.SubFrame.SFN.d = SystemFrameNumberInfo_Type_Number;
	ind.Common.TimingInfo.v.SubFrame.SFN.v.Number = 0; //Need to check what value needs to be sent
	
	ind.Common.TimingInfo.v.SubFrame.Subframe.d = SubFrameInfo_Type_Number;
	ind.Common.TimingInfo.v.SubFrame.Subframe.v.Number = 0; //Need to check what value needs to be sent

	ind.Common.TimingInfo.v.SubFrame.HSFN.d = SystemFrameNumberInfo_Type_Number;
        ind.Common.TimingInfo.v.SubFrame.HSFN.v.Number = 0; //Need to check what value needs to be sent

	ind.Common.TimingInfo.v.SubFrame.Slot.d = SlotTimingInfo_Type_Any;
        ind.Common.TimingInfo.v.SubFrame.Slot.v.Any = true;

	ind.Common.Status.d = IndicationStatus_Type_Ok;
        ind.Common.Status.v.Ok = true;

	ind.Common.RlcBearerRouting.d = true;
        ind.Common.RlcBearerRouting.v.d = RlcBearerRouting_Type_EUTRA;
        ind.Common.RlcBearerRouting.v.v.EUTRA = SS_context[0].eutra_cellId;

	//Populating the PDU
	ind.U_Plane.SubframeData.NoOfTTIs = 1;
	ind.U_Plane.SubframeData.PduSduList.d = L2DataList_Type_PdcpSdu;
	ind.U_Plane.SubframeData.PduSduList.v.PdcpSdu.d = 1;
	LOG_A(ENB_APP, "[SS_DRB][DRB_COMMON_IND] PDCP SDU Count: %d\n", ind.U_Plane.SubframeData.PduSduList.v.PdcpSdu.d);
	for(int i = 0; i < ind.U_Plane.SubframeData.PduSduList.v.PdcpSdu.d; i++){
        	ind.U_Plane.SubframeData.PduSduList.v.PdcpSdu.v[i].d = pdu_ind->sdu_size;
                DevAssert(ind.U_Plane.SubframeData.PduSduList.v.PdcpSdu.v != NULL);
		memcpy(ind.U_Plane.SubframeData.PduSduList.v.PdcpSdu.v[i].v, pdu_ind->sdu, pdu_ind->sdu_size); 
	}

	//Encode Message
	if (acpDrbProcessToSSEncSrv(ctx_drb_g, buffer, &msgSize, &ind) != 0)
        {
                LOG_A(ENB_APP, "[SS_DRB][DRB_COMMON_IND] acpDrbProcessToSSEncSrv Failure\n");
                return;
        }
	LOG_A(ENB_APP, "[SS_DRB][DRB_COMMON_IND] Buffer msgSize=%d (!!2) to EUTRACell %d", (int)msgSize,SS_context[0].eutra_cellId);

	//Send Message
	status = acpSendMsg(ctx_drb_g, msgSize, buffer);
	if (status != 0)
        {
                LOG_A(ENB_APP, "[SS_DRB][DRB_COMMON_IND] acpSendMsg failed. Error : %d on fd: %d\n", status, acpGetSocketFd(ctx_drb_g));
                return;
        }
	else
        {
                LOG_A(ENB_APP, "[SS_DRB][DRB_COMMON_IND] acpSendMsg Success \n");
        }

}

static void ss_task_handle_drb_pdu_req(struct DRB_COMMON_REQ *req)
{
	assert(req);
	MessageDef *message_p = itti_alloc_new_message(TASK_PDCP_ENB, instance_g, SS_DRB_PDU_REQ);
        assert(message_p);
        if (message_p)
        {
		 /* Populate the message and send to eNB */
                SS_DRB_PDU_REQ(message_p).drb_id = req->Common.RoutingInfo.v.RadioBearerId.v.Drb;
                memset(SS_DRB_PDU_REQ(message_p).sdu, 0, SDU_SIZE);

		for(int i = 0; i < req->U_Plane.SubframeDataList.d; i++){
			if(req->U_Plane.SubframeDataList.v[i].PduSduList.d == L2DataList_Type_PdcpSdu){
				LOG_A(ENB_APP, "PDCP SDU Received in DRB_COMMON_REQ");
				for(int j = 0; j < req->U_Plane.SubframeDataList.v[i].PduSduList.v.PdcpSdu.d; j++){
					SS_DRB_PDU_REQ(message_p).sdu_size = req->U_Plane.SubframeDataList.v[i].PduSduList.v.PdcpSdu.v[j].d;
					LOG_A(ENB_APP, "Length of PDCP SDU received in DRB_COMMON_REQ: %d",  req->U_Plane.SubframeDataList.v[i].PduSduList.v.PdcpSdu.v[j].d);
					memcpy(SS_DRB_PDU_REQ(message_p).sdu, req->U_Plane.SubframeDataList.v[i].PduSduList.v.PdcpSdu.v[j].v, req->U_Plane.SubframeDataList.v[i].PduSduList.v.PdcpSdu.v[j].d);
				}
			}
		}

	}
        SS_DRB_PDU_REQ(message_p).rnti = ss_rnti_g;

        int send_res = itti_send_msg_to_task(TASK_RRC_ENB, instance_g, message_p);
        if (send_res < 0)
        {
                LOG_A(ENB_APP, "[SS_DRB] Error in itti_send_msg_to_task");
        }

        LOG_A(ENB_APP, "Send res: %d", send_res);

}

ss_eNB_read_from_drb_socket(acpCtx_t ctx){

	size_t msgSize = size; //2

        while (1)
        {
                int userId = acpRecvMsg(ctx, &msgSize, buffer);

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
                        else
                        {
                                LOG_A(ENB_APP, "[SS_DRB] Invalid userId: %d \n", userId);
                                break;
                        }
                }

		if (userId == 0)
                {
                        // No message (timeout on socket)
                        //break;
                }
		else if (MSG_DrbProcessFromSS_userId == userId)
                {
                        struct DRB_COMMON_REQ *req = NULL;
                        LOG_A(ENB_APP, "[SS_DRB] DRB_COMMON_REQ Received \n");

                        if (acpDrbProcessFromSSDecSrv(ctx, buffer, msgSize, &req) != 0)
                        {
                                LOG_A(ENB_APP, "[SS_DRB][DRB_COMMON_REQ] acpDrbProcessFromSSDecSrv Failed\n");
                                break;
                        }
                        if(RC.ss.State >= SS_STATE_CELL_ACTIVE)
                        {
				LOG_A(ENB_APP, "[SS_DRB][DRB_COMMON_REQ] DRB_COMMON_REQ Received in CELL_ACTIVE\n");
                                ss_task_handle_drb_pdu_req(req);
                        }
                        else
                        {
                                LOG_W(ENB_APP, "[SS_DRB][DRB_COMMON_REQ] received in SS state %d \n", RC.ss.State);
                        }

                        acpDrbProcessFromSSFreeSrv(req);
                        return;
                }
		else if (MSG_DrbProcessToSS_userId == userId)
                {
                        LOG_A(ENB_APP, "[SS_DRB] DRB_COMMON_IND Received; ignoring \n");
                        break;
                }

	}
}

void *ss_eNB_drb_process_itti_msg(void *notUsed)
{
	MessageDef *received_msg = NULL;
        int result = 0;

        itti_receive_msg(TASK_SS_DRB, &received_msg);
		
	/* Check if there is a packet to handle */
        if (received_msg != NULL)
        {
                switch (ITTI_MSG_ID(received_msg))
                {
			case SS_DRB_PDU_IND:
			{
				task_id_t origin_task = ITTI_MSG_ORIGIN_ID(received_msg);

	    			if (origin_task == TASK_SS_PORTMAN)
				{       
			 				
                                	LOG_D(ENB_APP, "[SS_DRB] DUMMY WAKEUP recevied from PORTMAN state %d \n", RC.ss.State);
                                }
				else
	                        {
                                	LOG_A(ENB_APP, "[SS_DRB] Received SS_DRB_PDU_IND from RRC PDCP\n");
					if (RC.ss.State >= SS_STATE_CELL_ACTIVE)
	                                {
        	                                instance_g = ITTI_MSG_DESTINATION_INSTANCE(received_msg);
                	                        ss_send_drb_data(&received_msg->ittiMsg.ss_drb_pdu_ind);
                        	        }
					else
	                                {
        	                                LOG_A(ENB_APP, "ERROR [SS_DRB][SS_DRB_PDU_IND] received in SS state %d \n", RC.ss.State);
                	                }
				}

				result = itti_free(ITTI_MSG_ORIGIN_ID(received_msg), received_msg);
	                        AssertFatal(result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);
			};
			break;

			case TERMINATE_MESSAGE:
        	                LOG_A(ENB_APP, "[SS_DRB] Received TERMINATE_MESSAGE \n");
                	        itti_exit_task();
                       	 	break;

	                default:
                        	LOG_A(ENB_APP, "[SS_DRB] Received unhandled message %d:%s\n",
                                ITTI_MSG_ID(received_msg), ITTI_MSG_NAME(received_msg));
                        	break;
                }		
	}

	ss_eNB_read_from_drb_socket(ctx_drb_g);

        return NULL;

}

void ss_eNB_drb_init(void)
{
	IpAddress_t ipaddr;
        LOG_A(ENB_APP, "[SS_DRB] Starting System Simulator DRB Thread \n");

	const char *hostIp;
        hostIp = RC.ss.hostIp;
        acpConvertIp(hostIp, &ipaddr);

        // Port number
        int port = RC.ss.Drbport;
	const struct acpMsgTable msgTable[] = {
                {"DrbProcessFromSS", MSG_DrbProcessFromSS_userId},
                {"DrbProcessToSS", MSG_DrbProcessToSS_userId},
                // The last element should be NULL
                {NULL, 0}};

	// Arena size to decode received message
        const size_t aSize = 32 * 1024;

        // Start listening server and get ACP context,
        // after the connection is performed, we can use all services
        int ret = acpServerInitWithCtx(ipaddr, port, msgTable, aSize, &ctx_drb_g);
	if (ret < 0)
        {
                LOG_A(ENB_APP, "[SS_DRB] Connection failure err=%d\n", ret);
                return;
        }
        int fd1 = acpGetSocketFd(ctx_drb_g);
        LOG_A(ENB_APP, "[SS_DRB] Connection performed : %d\n", fd1);

        buffer = (unsigned char *)acpMalloc(size);
        assert(buffer);

        itti_subscribe_event_fd(TASK_SS_DRB, fd1);

        itti_mark_task_ready(TASK_SS_DRB);

}

void *ss_eNB_drb_task(void *arg)
{
        ss_eNB_drb_init();

        while (1)
        {
                (void)ss_eNB_drb_process_itti_msg(NULL);
        }
        acpFree(buffer);

        return NULL;
}

