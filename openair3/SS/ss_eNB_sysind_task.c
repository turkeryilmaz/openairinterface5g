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

#include "intertask_interface.h"
#include "common/ran_context.h"
#include "ss_eNB_context.h"
#include "acpSys.h"
#include "acpSysInd.h"
#include "ss_eNB_sysind_task.h"

extern RAN_CONTEXT_t RC;
extern SSConfigContext_t SS_context;
static acpCtx_t ctx_sysind_g = NULL;
static unsigned char *buffer = NULL;
static const size_t size = 16 * 1024;
static instance_t instance_g = 0;

enum MsgUserId
{
        // user defined IDs should be an int number >= 1
        MSG_SysProcess_userId = 1,
	MSG_SysIndProcessToSS_userId,
};


//------------------------------------------------------------------------------

/*
 * Function : ss_send_sysind_data
 * Description: Function to send response to the TTCN/SIDL Client
 * In :
 * Out:
 * newState: No impack on the State
 *
 */
static void ss_send_sysind_data(ss_system_ind_t *p_ind)
{
	struct SYSTEM_IND ind = {};
        uint32_t status = 0;
       
        DevAssert(p_ind != NULL);
        size_t msgSize = size; 
        memset(&ind, 0, sizeof(ind));
        ind.Common.CellId = SS_context.eutra_cellId;

        // Populated the Routing Info
        ind.Common.RoutingInfo.d = RoutingInfo_Type_None;
        ind.Common.RoutingInfo.v.None = true;

        // Populated the Timing Info
        ind.Common.TimingInfo.d = TimingInfo_Type_SubFrame;
        ind.Common.TimingInfo.v.SubFrame.SFN.d = SystemFrameNumberInfo_Type_Number;
        ind.Common.TimingInfo.v.SubFrame.SFN.v.Number = p_ind->sfn;

        ind.Common.TimingInfo.v.SubFrame.Subframe.d = SubFrameInfo_Type_Number;
        ind.Common.TimingInfo.v.SubFrame.Subframe.v.Number = p_ind->sf;

        ind.Common.TimingInfo.v.SubFrame.HSFN.d = SystemFrameNumberInfo_Type_Number;
        ind.Common.TimingInfo.v.SubFrame.HSFN.v.Number = 0;

        ind.Common.TimingInfo.v.SubFrame.Slot.d = SlotTimingInfo_Type_Any;
        ind.Common.TimingInfo.v.SubFrame.Slot.v.Any = true;

        ind.Common.Status.d = IndicationStatus_Type_Ok;
        ind.Common.Status.v.Ok = true;

        ind.Common.RlcBearerRouting.d = true;
        ind.Common.RlcBearerRouting.v.d = RlcBearerRouting_Type_EUTRA;
        ind.Common.RlcBearerRouting.v.v.EUTRA = SS_context.eutra_cellId;

        LOG_A(ENB_SS,"[SS_SYSIND][SYSTEM_IND] Frame: %d, Subframe: %d, RAPID: %d, PRTPower: %d, BitMask: %d \n",p_ind->sfn,p_ind->sf,p_ind->ra_PreambleIndex,p_ind->prtPower_Type,p_ind->bitmask);

        /* Populate and Send the SYSTEM_IND to Client */
        ind.Indication.d = SystemIndication_Type_RachPreamble;
        ind.Indication.v.RachPreamble.RAPID  = p_ind->ra_PreambleIndex;
        ind.Indication.v.RachPreamble.PRTPower = p_ind->prtPower_Type;
        if (p_ind->bitmask)
        {
           ind.Indication.v.RachPreamble.RepetitionsPerPreambleAttempt.d = true;
           ind.Indication.v.RachPreamble.RepetitionsPerPreambleAttempt.v = p_ind->repetitionsPerPreambleAttempt;
        }

        /* Encode message */
        if (acpSysIndProcessToSSEncSrv(ctx_sysind_g, buffer, &msgSize, &ind) != 0)
        {
                LOG_A(ENB_SS, "[SS_SYSIND][SYSTEM_IND] acpSysIndProcessToSSEncSrv Failure\n");
                return;
        }
        LOG_A(ENB_SS, "[SS_SYSIND][SYSTEM_IND] Buffer msgSize=%d (!!2) to EUTRACell %d\n", (int)msgSize,SS_context.eutra_cellId);

        /* Send message */
        status = acpSendMsg(ctx_sysind_g, msgSize, buffer);
        if (status != 0)
        {
                LOG_A(ENB_SS, "[SS_SYSIND][SYSTEM_IND] acpSendMsg failed. Error : %d on fd: %d\n", status, acpGetSocketFd(ctx_sysind_g));
                return;
        }
        else
        {
                LOG_A(ENB_SS, "[SS_SYSIND][SYSTEM_IND] acpSendMsg Success \n");
        }

}

/*
 * Function : ss_eNB_read_from_sysind_socket
 * Description: Function to received message from SYSIND Socket
 * In :
 * req  - Request received from the TTCN
 * Out:
 * newState: No impack on the State
 *
 */
static bool isConnected = false;
static inline void
ss_eNB_read_from_sysind_socket(acpCtx_t ctx)
{
	size_t msgSize = size; //2

	while (1)
	{
		int userId = acpRecvMsg(ctx, &msgSize, buffer);
		LOG_A(ENB_SS, "[SS_SYSIND] Received msgSize=%d, userId=%d\n", (int)msgSize, userId);

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
                                LOG_A(GNB_APP, "[SS_SYSIND] Peer ordered shutdown\n");
                                isConnected = false;
                        }
                        else if (userId == -ACP_PEER_CONNECTED){
                                LOG_A(GNB_APP, "[SS_SYSIND] Peer connection established\n");
                                isConnected = true;

                        }
                        else
                        {
                                LOG_A(ENB_SS, "[SS_SYSIND] Invalid userId: %d \n", userId);
                                break;
                        }
                }
		if (userId == 0)
		{
			// No message (timeout on socket)
                        if (isConnected == true)
                        {
                                break;
                        }
	        }
        }
}

/*
 * Function : ss_eNB_sysind_init
 * Description: Function handles for initilization of SYSIND task
 * In :
 * req :
 * Out:
 * newState: No impack on the State
 *
 */
void ss_eNB_sysind_init(void)
{
	IpAddress_t ipaddr;
        LOG_A(ENB_SS, "[SS_SYSIND] Starting System Simulator SYSIND Thread \n");
        const char *hostIp;
        hostIp = RC.ss.hostIp;
        acpConvertIp(hostIp, &ipaddr);

        // Port number
        int port = RC.ss.SysIndport;

	const struct acpMsgTable msgTable[] = {
                {"SysIndProcessToSS", MSG_SysIndProcessToSS_userId},
                // The last element should be NULL
                {NULL, 0}};
        
        // Arena size to decode received message
        const size_t aSize = 32 * 1024;

        // Start listening server and get ACP context,
        // after the connection is performed, we can use all services
        int ret = acpServerInitWithCtx(ipaddr, port, msgTable, aSize, &ctx_sysind_g);
        if (ret < 0)
        {
                LOG_A(ENB_SS, "[SS_SYSIND] Connection failure err=%d\n", ret);
                return;
        }
        int fd1 = acpGetSocketFd(ctx_sysind_g);
        LOG_A(ENB_SS, "[SS_SYSIND] Connection performed : %d\n", fd1);

        buffer = (unsigned char *)acpMalloc(size);
        assert(buffer);

        itti_subscribe_event_fd(TASK_SS_SYSIND, fd1);

        itti_mark_task_ready(TASK_SS_SYSIND);

}

/*
 * Function : ss_eNB_sysind_process_itti_msg
 * Description: Funtion Handles the ITTI
 * message received from the eNB on SYSIND Port
 * In :
 * Out:
 * newState: No impact on state machine.
 *
 */
void *ss_eNB_sysind_process_itti_msg(void *notUsed)
{
	MessageDef *received_msg = NULL;
        int result = 0;

        itti_receive_msg(TASK_SS_SYSIND, &received_msg);


        /* Check if there is a packet to handle */
        if (received_msg != NULL)
        {
                switch (ITTI_MSG_ID(received_msg))
                {
                case SS_SYSTEM_IND:
                {
                        task_id_t origin_task = ITTI_MSG_ORIGIN_ID(received_msg);

                        if (origin_task == TASK_SS_PORTMAN)
                        {
                                LOG_D(ENB_APP, "[SS_SYSIND] DUMMY WAKEUP receviedfrom PORTMAN state %d \n", RC.ss.State);
                        }
                        else
                        {
                                LOG_A(ENB_SS, "[SS_SYSIND] Received SS_SYSTEM_IND\n");
                                if (RC.ss.State >= SS_STATE_CELL_CONFIGURED)
                                {
                                        instance_g = ITTI_MSG_DESTINATION_INSTANCE(received_msg);
                                        ss_send_sysind_data(&received_msg->ittiMsg.ss_system_ind);
                                }
                                else
                                {
                                        LOG_E(ENB_SS, "[SS_SYSIND][SS_SYSTEM_IND] received in SS state %d \n", RC.ss.State);
                                }
                        }
                        result = itti_free(ITTI_MSG_ORIGIN_ID(received_msg), received_msg);
                        AssertFatal(result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);
                };
                break;

		case TERMINATE_MESSAGE:
                        LOG_A(ENB_SS, "[SS_SYSIND] Received TERMINATE_MESSAGE \n");
                        itti_exit_task();
                        break;

                default:
                        LOG_A(ENB_SS, "[SS_SYSIND] Received unhandled message %d:%s\n",
                                  ITTI_MSG_ID(received_msg), ITTI_MSG_NAME(received_msg));
                        break;
                }
        }
        return NULL;
}

/*
 * Function : ss_eNB_sysind_task
 * Description: Funtion Handles the SYSIND Task
 * In :
 * req :
 * Out:
 * newState: No impact on state machine.
 *
 */
void *ss_eNB_sysind_task(void *arg)
{
        while (1)
        {
                (void)ss_eNB_sysind_process_itti_msg(NULL);
        }

        return NULL;
}

/*
 * Function : ss_eNB_sysind_acp_task
 * Description: Funtion Handles the SYSIND ACP Task
 * In :
 * req :
 * Out:
 * newState: No impact on state machine.
 *
 */
void *ss_eNB_sysind_acp_task(void *arg)
{
        ss_eNB_sysind_init();
	while (1)
	{
		ss_eNB_read_from_sysind_socket(ctx_sysind_g);
	}
	acpFree(buffer);
}
