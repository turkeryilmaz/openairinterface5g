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
#include "NR_UL-CCCH-Message.h"
#include "NR_DL-CCCH-Message.h"
#include "NR_UL-DCCH-Message.h"
#include "NR_DL-DCCH-Message.h"

#include "acpNrSysSrb.h"
#include "ss_gNB_context.h"

extern RAN_CONTEXT_t RC;
SSConfigContext_t SS_context;
static acpCtx_t ctx_srb_g = NULL;
static uint16_t rnti_g = 0;
static instance_t instance_g = 0;
uint16_t ss_rnti_g = 0;

typedef enum {
        // user defined IDs should be an int number >= 1
        MSG_TestHelloFromSS_userId = 1,
        /* MSG_SysProcess_userId, */
        MSG_NrSysSrbProcessFromSS_userId,
        MSG_NrSysSrbProcessToSS_userId,
} MSG_userId;

static unsigned char *buffer = NULL;
static const size_t size = 16 * 1024;

//------------------------------------------------------------------------------
// Function to send response to the SIDL client
static void ss_send_srb_data(ss_nrrrc_pdu_ind_t *pdu_ind)
{
        struct NR_RRC_PDU_IND ind = {};
        uint32_t status = 0;
        NR_UL_DCCH_Message_t               *ul_dcch_msg = NULL;
        NR_UL_CCCH_Message_t               *ul_ccch_msg = NULL;

        LOG_A(GNB_APP, "[SS_SRB] Reported rrc sdu_size:%d \t srb_id %d\n", pdu_ind->sdu_size, pdu_ind->srb_id);

        DevAssert(pdu_ind != NULL);
        DevAssert(pdu_ind->sdu_size >= 0);
        DevAssert(pdu_ind->srb_id >= 0);
        rnti_g = pdu_ind->rnti;
        ss_rnti_g = rnti_g;
        size_t msgSize = size;
        memset(&ind, 0, sizeof(ind));
        //TODO: Work Around till Sys port is implemented for 5G
	ind.Common.CellId = nr_Cell1;

        // Populated the Routing Info
        ind.Common.RoutingInfo.d = NR_RoutingInfo_Type_RadioBearerId;
        ind.Common.RoutingInfo.v.RadioBearerId.d = NR_RadioBearerId_Type_Srb;
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

        ind.Common.RlcBearerRouting.d = RlcBearerRouting_Type_NR;
	//TODO: Work Around till Sys port is implemented for 5G
        ind.Common.RlcBearerRouting.v.NR = nr_Cell1;

        /* Populate and Send the EUTRA RRC PDU IND to Client */
        if (pdu_ind->srb_id == 0)
        {
                uper_decode(
                      NULL,
                      &asn_DEF_NR_UL_CCCH_Message,
                      (void **)&ul_ccch_msg,
                      pdu_ind->sdu,
                      pdu_ind->sdu_size,
                      0,
                      0);

                xer_fprint(stdout, &asn_DEF_NR_UL_CCCH_Message, (void *)ul_ccch_msg);
                ind.RrcPdu.d = NR_RRC_MSG_Indication_Type_Ccch;
                ind.RrcPdu.v.Ccch.d = pdu_ind->sdu_size;
                ind.RrcPdu.v.Ccch.v = pdu_ind->sdu;
        }
        else
        {
                uper_decode(
                      NULL,
                      &asn_DEF_NR_UL_DCCH_Message,
                      (void **)&ul_dcch_msg,
                      pdu_ind->sdu,
                      pdu_ind->sdu_size,
                      0,
                      0);

                xer_fprint(stdout, &asn_DEF_NR_UL_DCCH_Message, (void *)ul_dcch_msg);
                ind.RrcPdu.d = NR_RRC_MSG_Indication_Type_Dcch;
                ind.RrcPdu.v.Dcch.d = pdu_ind->sdu_size;
                ind.RrcPdu.v.Dcch.v = pdu_ind->sdu;
        }


        /* Encode message
   */
        if (acpNrSysSrbProcessToSSEncSrv(ctx_srb_g, buffer, &msgSize, &ind) != 0)
        {
                LOG_A(GNB_APP, "[SS_SRB][NR_RRC_PDU_IND] acpNrSysSrbProcessToSSEncSrv Failure\n");
                return;
        }
        LOG_A(GNB_APP, "[SS_SRB][NR_RRC_PDU_IND] Buffer msgSize=%d (!!2) to EUTRACell %d", (int)msgSize,SS_context.nr_cellId);

        /* Send message
   */
        status = acpSendMsg(ctx_srb_g, msgSize, buffer);
        if (status != 0)
        {
                LOG_A(GNB_APP, "[SS_SRB][NR_RRC_PDU_IND] acpSendMsg failed. Error : %d on fd: %d\n", status, acpGetSocketFd(ctx_srb_g));
                return;
        }
        else
        {
                LOG_A(GNB_APP, "[SS_SRB][NR_RRC_PDU_IND] acpSendMsg Success \n");
        }
}

//------------------------------------------------------------------------------
static void ss_task_handle_rrc_pdu_req(struct NR_RRC_PDU_REQ *req)
{
        assert(req);
        NR_DL_DCCH_Message_t *dl_dcch_msg=NULL;
        NR_DL_CCCH_Message_t *dl_ccch_msg=NULL;
        MessageDef *message_p = itti_alloc_new_message(TASK_RRC_GNB, instance_g, SS_NRRRC_PDU_REQ);
        assert(message_p);
        if (message_p)
        {
                /* Populate the message and send to SS */
                SS_NRRRC_PDU_REQ(message_p).srb_id = req->Common.RoutingInfo.v.RadioBearerId.v.Srb;
                memset(SS_NRRRC_PDU_REQ(message_p).sdu, 0, SDU_SIZE);
                if (req->RrcPdu.d == NR_RRC_MSG_Request_Type_Ccch)
                {
                        SS_NRRRC_PDU_REQ(message_p).sdu_size = req->RrcPdu.v.Ccch.d;
                        memcpy(SS_NRRRC_PDU_REQ(message_p).sdu, req->RrcPdu.v.Ccch.v, req->RrcPdu.v.Ccch.d);
                        uper_decode(NULL,
                                    &asn_DEF_NR_DL_CCCH_Message,
                                    (void **)&dl_ccch_msg,
                                    (uint8_t *)SS_NRRRC_PDU_REQ(message_p).sdu,
                                    SS_NRRRC_PDU_REQ(message_p).sdu_size,0,0);

                        xer_fprint(stdout,&asn_DEF_NR_DL_CCCH_Message,(void *)dl_ccch_msg);
                }
                else
                {
                        SS_NRRRC_PDU_REQ(message_p).sdu_size = req->RrcPdu.v.Dcch.d;
                        memcpy(SS_NRRRC_PDU_REQ(message_p).sdu, req->RrcPdu.v.Dcch.v, req->RrcPdu.v.Dcch.d);
                        uper_decode(NULL,
                                    &asn_DEF_NR_DL_DCCH_Message,
                                    (void **)&dl_dcch_msg,
                                    (uint8_t *)SS_NRRRC_PDU_REQ(message_p).sdu,
                                    SS_NRRRC_PDU_REQ(message_p).sdu_size,0,0);

                        xer_fprint(stdout,&asn_DEF_NR_DL_DCCH_Message,(void *)dl_dcch_msg);
                }

                LOG_A(GNB_APP, "[SS_SRB][NR_RRC_PDU_REQ] sending to TASK_RRC_GNB: {srb: %d, ch: %s, qty: %d }",
                          SS_NRRRC_PDU_REQ(message_p).srb_id,
                          req->RrcPdu.d == NR_RRC_MSG_Request_Type_Ccch ? "CCCH" : "DCCH", SS_NRRRC_PDU_REQ(message_p).sdu_size);

                SS_NRRRC_PDU_REQ(message_p).rnti = rnti_g;
                int send_res = itti_send_msg_to_task(TASK_RRC_GNB, instance_g, message_p);
                if (send_res < 0)
                {
                        LOG_A(GNB_APP, "[SS_SRB] Error in itti_send_msg_to_task");
                }

                LOG_A(GNB_APP, "Send res: %d", send_res);
        }
}


//------------------------------------------------------------------------------
static inline void
ss_gNB_read_from_srb_socket(acpCtx_t ctx)
{
        size_t msgSize = size; //2

        while (1)
        {
                int userId = acpRecvMsg(ctx, &msgSize, buffer);
                LOG_A(GNB_APP, "[SS_SRB] Received msgSize=%d, userId=%d\n", (int)msgSize, userId);

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
                                LOG_A(GNB_APP, "[SS_SRB] Invalid userId: %d \n", userId);
                                break;
                        }
                }

                if (userId == 0)
                {
                        // No message (timeout on socket)
                        break;
                }
                else if (MSG_NrSysSrbProcessFromSS_userId == userId)
                {
                        struct NR_RRC_PDU_REQ *req = NULL;
                        LOG_A(GNB_APP, "[SS_SRB][NR_RRC_PDU_REQ] NR_RRC_PDU_REQ Received \n");
                        // Got the message
                        if (acpNrSysSrbProcessFromSSDecSrv(ctx, buffer, msgSize, &req) != 0)
                        {
                                LOG_A(GNB_APP, "[SS_SRB][NR_RRC_PDU_REQ] acpNrSysSrbProcessFromSSDecSrv Failed\n");
                                break;
                        }
                        if (RC.ss.State >= SS_STATE_CELL_ACTIVE)
                        {
                                ss_task_handle_rrc_pdu_req(req);
                        }
                        else
                        {
                                LOG_A(GNB_APP, "ERROR [SS_SRB][NR_RRC_PDU_REQ] received in SS state %d \n", RC.ss.State);
                        }

                        acpNrSysSrbProcessFromSSFreeSrv(req);
                        return;
                }
                else if (MSG_NrSysSrbProcessToSS_userId == userId)
                {
                        LOG_A(GNB_APP, "[SS_SRB][NR_RRC_PDU_IND] NR_RRC_PDU_IND Received; ignoring \n");
                        break;
                }
                else if (userId == MSG_TestHelloFromSS_userId)
                {
                        LOG_A(GNB_APP, "[SS_SRB] Hello From Client Received \n");
                        break;
                }
        }
}


//------------------------------------------------------------------------------
void ss_gNB_srb_init(void)
{
        IpAddress_t ipaddr;
        LOG_A(GNB_APP, "[SS_SRB] Starting System Simulator SRB Thread \n");

        const char *hostIp;
        hostIp = RC.ss.hostIp;
        acpConvertIp(hostIp, &ipaddr);

        // Port number
        int port = RC.ss.Srbport;
	acpInit(malloc, free, 1000);

        // Register user services/notifications in message table
        const struct acpMsgTable msgTable[] = {
                { "NrSysSrbProcessFromSS", MSG_NrSysSrbProcessFromSS_userId },
                { "NrSysSrbProcessToSS", MSG_NrSysSrbProcessToSS_userId },
                { "TestHelloFromSS", MSG_TestHelloFromSS_userId },
                /* { "SysProcess", MSG_SysProcess_userId }, */
                // The last element should be NULL
                { NULL, 0 }
        };

        // Arena size to decode received message
        const size_t aSize = 32 * 1024;

        // Start listening server and get ACP context,
        // after the connection is performed, we can use all services
        int ret = acpServerInitWithCtx(ipaddr, port, msgTable, aSize, &ctx_srb_g);
        if (ret < 0)
        {
                LOG_A(GNB_APP, "[SS_SRB] Connection failure err=%d\n", ret);
                return;
        }
        int fd1 = acpGetSocketFd(ctx_srb_g);
        LOG_A(GNB_APP, "[SS_SRB] Connection performed : %d\n", fd1);

                buffer = (unsigned char *)acpMalloc(size);
        assert(buffer);

        RC.ss.State = SS_STATE_CELL_ACTIVE;
        itti_subscribe_event_fd(TASK_SS_SRB, fd1);

        itti_mark_task_ready(TASK_SS_SRB);
}

//------------------------------------------------------------------------------
void *ss_gNB_srb_process_itti_msg(void *notUsed)
{
        MessageDef *received_msg = NULL;
        int result = 0;

        itti_receive_msg(TASK_SS_SRB, &received_msg);

        /* Check if there is a packet to handle */
        if (received_msg != NULL)
        {
                switch (ITTI_MSG_ID(received_msg))
                {
                case SS_NRRRC_PDU_IND:
                {
#if 0
                        task_id_t origin_task = ITTI_MSG_ORIGIN_ID(received_msg);

                        if (origin_task == TASK_SS_PORTMAN)
                        {
                                LOG_D(GNB_APP, "[SS_SRB] DUMMY WAKEUP receviedfrom PORTMAN state %d \n", RC.ss.State);
                        }
                        else
#endif
                        {
                                LOG_A(GNB_APP, "[SS_SRB] Received SS_NRRRC_PDU_IND from RRC\n");
                                if (RC.ss.State >= SS_STATE_CELL_ACTIVE)
                                {
                                        instance_g = ITTI_MSG_DESTINATION_INSTANCE(received_msg);
                                        ss_send_srb_data(&received_msg->ittiMsg.ss_nrrrc_pdu_ind);
                                }
                                else
                                {
                                        LOG_A(GNB_APP, "ERROR [SS_SRB][NR_RRC_PDU_IND] received in SS state %d \n", RC.ss.State);
                                }
                         }

                        result = itti_free(ITTI_MSG_ORIGIN_ID(received_msg), received_msg);
                        AssertFatal(result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);
                };
                break;
                case TERMINATE_MESSAGE:
                        LOG_A(GNB_APP, "[SS_SRB] Received TERMINATE_MESSAGE \n");
                        itti_exit_task();
                        break;

                default:
                        LOG_A(GNB_APP, "[SS_SRB] Received unhandled message %d:%s\n",
                                  ITTI_MSG_ID(received_msg), ITTI_MSG_NAME(received_msg));
                        break;
                }
        }

        ss_gNB_read_from_srb_socket(ctx_srb_g);

        return NULL;
}

static void ss_gNB_wait_hello(void)
{

        while (1)
        {
                size_t msg_sz = size;
                int ret = acpRecvMsg(ctx_srb_g, &msg_sz, buffer);
                if (ret == MSG_TestHelloFromSS_userId)
                {
                        LOG_A(GNB_APP, "[SS_SRB] Hello From Client Received (on-start) \n");
                        break;
                }
        }
}

void *ss_gNB_srb_task(void *arg)
{
        ss_gNB_srb_init();
        ss_gNB_wait_hello();

	while (1)
        {
                //LOG_A(GNB_APP,"[SS_SRB] Inside ss_gNB_srb_task \n");
                (void)ss_gNB_srb_process_itti_msg(NULL);
        }
        acpFree(buffer);

	return NULL;
}
