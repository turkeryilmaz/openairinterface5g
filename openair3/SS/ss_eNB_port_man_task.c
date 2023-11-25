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

#include <pthread.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <sys/ioctl.h>
#include <net/if.h>

#include <netinet/in.h>
#include <netinet/sctp.h>

#include <arpa/inet.h>

#include "assertions.h"
#include "common/utils/system.h"
#include "queue.h"
#include "sctp_common.h"

#include "intertask_interface.h"
#include "common/ran_context.h"

#include "ss_eNB_port_man_task.h"
#include "ss_eNB_context.h"
#include "acpSys.h"

extern RAN_CONTEXT_t RC;
extern int  cell_index;
acpCtx_t ctx_g = NULL;
int cell_index_pm = 0;

enum MsgUserId
{
    MSG_SysProcess_userId = 1,
};
extern SSConfigContext_t SS_context;

pthread_cond_t sys_confirm_done_cond;
pthread_mutex_t sys_confirm_done_mutex;
int sys_confirm_done=-1;

static  void wait_sys_confirm(char *thread_name) {
  LOG_I(ENB_SS_PORTMAN_ACP, "Entry in fxn:%s\n",__FUNCTION__);
  LOG_I(ENB_SS_PORTMAN_ACP, "waiting for SYS CONFRIM DONE Indication (%s)\n",thread_name);
  pthread_mutex_lock( &sys_confirm_done_mutex );

  while ( sys_confirm_done < 0 )
    pthread_cond_wait( &sys_confirm_done_cond, &sys_confirm_done_mutex );

  pthread_mutex_unlock(&sys_confirm_done_mutex );
  LOG_I(ENB_SS_PORTMAN_ACP, "Received SYS CONFIRM (%s)\n", thread_name);
  LOG_I(ENB_SS_PORTMAN_ACP, "Exit from fxn:%s\n",__FUNCTION__);
}

/*
 * Function : ss_dumpReqMsg
 * Description: Function for print the received message
 * In :
 * req  -
 * Out:
 * newState: No impact on state machine.
 *
 */
static void ss_dumpReqMsg(struct SYSTEM_CTRL_REQ *msg)
{
    LOG_A(ENB_SS_PORTMAN, "SysProcess: received from the TTCN\n");
    LOG_A(ENB_SS_PORTMAN, "\tCommon:\n");
    LOG_A(ENB_SS_PORTMAN, "\t\tCellId=%d\n", msg->Common.CellId);
    LOG_A(ENB_SS_PORTMAN, "\t\tRoutingInfo=%d\n", msg->Common.RoutingInfo.d);
    LOG_A(ENB_SS_PORTMAN, "\t\tTimingInfo=%d\n", msg->Common.TimingInfo.d);
    LOG_A(ENB_SS_PORTMAN, "\t\tCnfFlag=%d\n", msg->Common.ControlInfo.CnfFlag);
    LOG_A(ENB_SS_PORTMAN, "\t\tFollowOnFlag=%d\n", msg->Common.ControlInfo.FollowOnFlag);
    LOG_A(ENB_SS_PORTMAN, "\tRequest=%d\n", msg->Request.d);
}

/*
 * Function : ss_port_man_send_cnf
 * Description: Function for sending the confirmation to the TTCN on the basis of
 * particular messages
 * In :
 * req  -
 * Out:
 * newState: No impact on state machine.
 *
 */
void ss_port_man_send_cnf(struct SYSTEM_CTRL_CNF recvCnf)
{
    struct SYSTEM_CTRL_CNF cnf;
    const size_t size = 16 * 1024;
    uint32_t status;

    unsigned char *buffer = (unsigned char *)acpMalloc(size);

    size_t msgSize = size;
    memset(&cnf, 0, sizeof(cnf));
    cnf.Common.CellId = recvCnf.Common.CellId;
    cnf.Common.RoutingInfo.d = RoutingInfo_Type_None;
    cnf.Common.RoutingInfo.v.None = true;
    cnf.Common.TimingInfo.d = TimingInfo_Type_Now;
    cnf.Common.TimingInfo.v.Now = true;
    cnf.Common.Result.d = recvCnf.Common.Result.d;
    cnf.Common.Result.v.Success = recvCnf.Common.Result.v.Success;
    cnf.Confirm.d = recvCnf.Confirm.d;
    LOG_A(ENB_SS_PORTMAN, "SYS_CNF received cellId %d result %d type %d \n",
                     cnf.Common.CellId,cnf.Common.Result.d, recvCnf.Confirm.d);
    switch (recvCnf.Confirm.d)
    {
    case SystemConfirm_Type_Cell:
        cnf.Confirm.v.Cell = true;
        break;

    case SystemConfirm_Type_CellAttenuationList:
        cnf.Confirm.v.CellAttenuationList = true;
        cnf.Common.CellId = eutra_Cell_NonSpecific;
        break;

    case SystemConfirm_Type_RadioBearerList:
        cnf.Confirm.v.RadioBearerList = true;
        break;

    case SystemConfirm_Type_AS_Security:
        cnf.Confirm.v.AS_Security = true;
        break;

    case SystemConfirm_Type_PdcpCount:
        cnf.Confirm.v.PdcpCount.d = recvCnf.Confirm.v.PdcpCount.d;
        cnf.Confirm.v.PdcpCount.v = recvCnf.Confirm.v.PdcpCount.v;
        break;
    case SystemConfirm_Type_UE_Cat_Info:
        cnf.Confirm.v.UE_Cat_Info = true;
        break;
    case SystemConfirm_Type_Paging:
        cnf.Confirm.v.Paging = true;
        break;
    case SystemConfirm_Type_PdcchOrder:
        cnf.Confirm.v.PdcchOrder = true;
        break;

    case SystemConfirm_Type_L1MacIndCtrl:
      LOG_A(ENB_SS_PORTMAN, "SystemConfirm_Type_L1MacIndCtrl\n");
      cnf.Confirm.v.L1MacIndCtrl = true;
      break;
    case SystemConfirm_Type_Sps:
    case SystemConfirm_Type_RlcIndCtrl:
    case SystemConfirm_Type_PdcpHandoverControl:
        cnf.Confirm.v.PdcpHandoverControl = true;
        break;
    case SystemConfirm_Type_L1_TestMode:
    case SystemConfirm_Type_ActivateScell:
    case SystemConfirm_Type_MbmsConfig:
    case SystemConfirm_Type_PDCCH_MCCH_ChangeNotification:
    case SystemConfirm_Type_MSI_Config:
    case SystemConfirm_Type_OCNG_Config:
    case SystemConfirm_Type_DirectIndicationInfo:
    default:
        LOG_A(ENB_SS_PORTMAN, "[SYS] Error not handled CNF TYPE to %d \n", recvCnf.Confirm.d);
    }

    /* Encode message
     */
    if (acpSysProcessEncSrv(ctx_g, buffer, &msgSize, &cnf) != 0)
    {
        acpFree(buffer);
        return;
    }
    /* Send message
     */
    status = acpSendMsg(ctx_g, msgSize, buffer);
    if (status != 0)
    {
        LOG_A(ENB_SS_PORTMAN, "acpSendMsg failed. Error : %d on fd: %d\n",
              status, acpGetSocketFd(ctx_g));
        acpFree(buffer);
        return;
    }
    else
    {
        LOG_A(ENB_SS_PORTMAN, "acpSendMsg Success \n");
    }
    // Free allocated buffer
    acpFree(buffer);
}

/*
 * Function : ss_port_man_send_cnf
 * Description: Function to send response to the TTCN/SIDL CLient
 * In :
 * req  -
 * Out:
 * newState: No impact on state machine.
 *
 */
void ss_port_man_send_data(
    instance_t instance,
    task_id_t task_id,
    ss_set_timinfo_t *tinfo)
{
    struct SYSTEM_CTRL_CNF cnf;
    const size_t size = 16 * 1024;
    uint32_t status;

    unsigned char *buffer = (unsigned char *)acpMalloc(size);

    DevAssert(tinfo != NULL);
    DevAssert(tinfo->sfn >= 0);
    DevAssert(tinfo->sf >= 0);

    size_t msgSize = size;
    memset(&cnf, 0, sizeof(cnf));
    cnf.Common.CellId = SS_context.SSCell_list[tinfo->cell_index].eutra_cellId;
    cnf.Common.RoutingInfo.d = RoutingInfo_Type_None;
    cnf.Common.RoutingInfo.v.None = true;
    cnf.Common.TimingInfo.d = TimingInfo_Type_Now;
    cnf.Common.TimingInfo.v.Now = true;
    cnf.Common.Result.d = ConfirmationResult_Type_Success;
    cnf.Common.Result.v.Success = true;
    cnf.Confirm.d = SystemConfirm_Type_EnquireTiming;
    cnf.Confirm.v.EnquireTiming = true;

    cell_index_pm = tinfo->cell_index;

    /**
   * FIXME: Currently filling only SFN and subframe numbers.
   */
    cnf.Common.TimingInfo.d = TimingInfo_Type_SubFrame;
    cnf.Common.TimingInfo.v.SubFrame.SFN.d = SystemFrameNumberInfo_Type_Number;
    cnf.Common.TimingInfo.v.SubFrame.SFN.v.Number = tinfo->sfn;

    cnf.Common.TimingInfo.v.SubFrame.Subframe.d = SubFrameInfo_Type_Number;
    cnf.Common.TimingInfo.v.SubFrame.Subframe.v.Number = tinfo->sf;

    cnf.Common.TimingInfo.v.SubFrame.HSFN.d = SystemFrameNumberInfo_Type_Number;
    cnf.Common.TimingInfo.v.SubFrame.HSFN.v.Number = tinfo->hsfn;

    /** TODO: Always marking as first slot, need to change this */
    cnf.Common.TimingInfo.v.SubFrame.Slot.d = SlotTimingInfo_Type_FirstSlot;
    cnf.Common.TimingInfo.v.SubFrame.Slot.v.FirstSlot = true;

    /** TODO: Always marking as any symbol, need to change this */
    cnf.Common.TimingInfo.v.SubFrame.Symbol.d = SymbolTimingInfo_Type_Any;
    cnf.Common.TimingInfo.v.SubFrame.Symbol.v.Any = true;

    /* Encode message
     */
    if (acpSysProcessEncSrv(ctx_g, buffer, &msgSize, &cnf) != 0)
    {
        acpFree(buffer);
        return;
    }

    /* Send message
     */
    status = acpSendMsg(ctx_g, msgSize, buffer);
    if (status != 0)
    {
        LOG_A(ENB_SS_PORTMAN, "acpSendMsg failed. Error : %d on fd: %d\n",
              status, acpGetSocketFd(ctx_g));
        acpFree(buffer);
        return;
    }
    else
    {
        LOG_A(ENB_SS_PORTMAN, "acpSendMsg Success \n");
    }
    // Free allocated buffer
    acpFree(buffer);
}

/*
 * Function : ss_eNB_port_man_init
 * Description: Function to initilize the portman task
 * In :
 * req  -
 * Out:
 * newState: No impact on state machine.
 *
 */
void ss_eNB_port_man_init(void)
{

    LOG_A(ENB_SS_PORTMAN_ACP, "[SS-PORTMAN] Starting System Simulator Manager\n");
    // Port number
    int port = RC.ss.Sysport;

    acpInit(malloc, free, 0);

    const struct acpMsgTable msgTable[] = {
        {"SysProcess", MSG_SysProcess_userId},

        // The last element should be NULL
        {NULL, 0}};

    // Arena size to decode received message
    const size_t aSize = 128 * 1024;

    // Start listening server and get ACP context,
    // after the connection is performed, we can use all services
    int ret = acpServerInitWithCtx(RC.ss.SysHost ? RC.ss.SysHost : "127.0.0.1", port, msgTable, aSize, &ctx_g);
    if (ret < 0)
    {
        LOG_A(ENB_SS_PORTMAN_ACP, "Connection failure err=%d\n", ret);
        return;
    }
    int fd1 = acpGetSocketFd(ctx_g);
    LOG_A(ENB_SS_PORTMAN_ACP, "Connection performed : %d\n", fd1);

    //itti_subscribe_event_fd(TASK_SS_PORTMAN, fd1);

    itti_mark_task_ready(TASK_SS_PORTMAN);
}

/*
 * Function : ss_eNB_read_from_socket
 * Description: Function to read from the Socket
 * In :
 * req  -
 * Out:
 * newState: No impact on state machine.
 *
 */
static inline void ss_eNB_read_from_socket(acpCtx_t ctx)
{
    struct SYSTEM_CTRL_REQ *req = NULL;
    const size_t size = 16 * 1024;
    size_t msgSize = size; //2
    unsigned char *buffer = (unsigned char *)acpMalloc(size);
    assert(buffer);
    LOG_D(ENB_SS_PORTMAN_ACP, "Entry in fxn:%s\n", __FUNCTION__);
    int userId = acpRecvMsg(ctx, &msgSize, buffer);

    // Error handling
    if (userId < 0)
    {
        if (userId == -ACP_ERR_SERVICE_NOT_MAPPED)
        {
            // Message not mapped to user id,
            // this error should not appear on server side for the messages
            // received from clients
          LOG_E(ENB_SS_PORTMAN_ACP, "Error: Message not mapped to user id\n");
        }
        else if (userId == -ACP_ERR_SIDL_FAILURE)
        {
            // Server returned service error,
            // this error should not appear on server side for the messages
            // received from clients
            LOG_E(ENB_SS_PORTMAN_ACP, "Error: Server returned service error \n");
            SidlStatus sidlStatus = -1;
            acpGetMsgSidlStatus(msgSize, buffer, &sidlStatus);
              acpFree(buffer);
        }
        else if (userId == -ACP_PEER_DISCONNECTED)
        {
            LOG_E(ENB_SS_PORTMAN_ACP, "Error: Peer ordered shutdown\n");
        }
        else if (userId == -ACP_PEER_CONNECTED)
        {
            LOG_A(ENB_SS_PORTMAN_ACP, " Peer connection established\n");
        }
        else
        {
            return;
        }
    }
    else if (userId == 0)
    {
        // No message (timeout on socket)
        //Send Dummy Wake up ITTI message to SRB task.
        if (RC.ss.mode >= SS_SOFTMODEM && SS_context.SSCell_list[cell_index_pm].State >= SS_STATE_CELL_ACTIVE)
        {
            LOG_D(ENB_SS_PORTMAN_ACP,"Sending Wake up signal/SS_RRC_PDU_IND (msg_Id:%d) to TASK_SS_SRB task \n", SS_RRC_PDU_IND);
            MessageDef *message_p = itti_alloc_new_message(TASK_SS_PORTMAN, 0, SS_RRC_PDU_IND);
            if (message_p)
            {
                /* Populate the message to SS */
                SS_RRC_PDU_IND(message_p).sdu_size = 1;
                SS_RRC_PDU_IND(message_p).srb_id = -1;
                SS_RRC_PDU_IND(message_p).rnti = -1;
                SS_RRC_PDU_IND(message_p).frame = -1;
                SS_RRC_PDU_IND(message_p).subframe = -1;

                int send_res = itti_send_msg_to_task(TASK_SS_SRB, 0, message_p);
                if (send_res < 0)
                {
                    LOG_E(ENB_SS_PORTMAN_ACP, "Error in sending Wake up signal /SS_RRC_PDU_IND (msg_Id:%d)  to TASK_SS_SRB\n", SS_RRC_PDU_IND);
                }
            }
        }
    }
    else
    {

      LOG_A(ENB_SS_PORTMAN_ACP, "received msg %d from the client.\n", userId);
      if (acpSysProcessDecSrv(ctx, buffer, msgSize, &req) != 0)
      {
        acpFree(buffer);
        return;
      }
      ss_dumpReqMsg(req);

      if (userId == MSG_SysProcess_userId)
      {
        bool ret_Val = false;
        struct SYSTEM_CTRL_REQ *sys_req = (struct SYSTEM_CTRL_REQ *)req;
        if (sys_req->Request.d == SystemRequest_Type_EnquireTiming)
        {
          LOG_I(ENB_SS_PORTMAN_ACP, "Received EnquireTiming\n");
          ret_Val = ss_eNB_port_man_handle_enquiryTiming(sys_req);
          if (ret_Val == false)
            LOG_E(ENB_SS_PORTMAN_ACP, "Error Sending EnquiryTiming Respone to TTCN\n");
        }
        else
        {
          int rc = 0;
          MessageDef *message_p = itti_alloc_new_message(TASK_SS_PORTMAN, 0,  SS_SYS_PORT_MSG_IND);
          if (message_p)
          {
            SS_SYS_PORT_MSG_IND(message_p).req = (struct SYSTEM_CTRL_REQ *)malloc(sizeof(struct SYSTEM_CTRL_REQ));
            if (SS_SYS_PORT_MSG_IND(message_p).req == NULL)
            {
              LOG_E(ENB_SS_PORTMAN_ACP, "Error allocating memory for SYSTEM CTRL REQ\n");
              return;
            }
            memset(SS_SYS_PORT_MSG_IND(message_p).req, 0, sizeof(struct SYSTEM_CTRL_REQ));
            memcpy(SS_SYS_PORT_MSG_IND(message_p).req, req, sizeof(struct SYSTEM_CTRL_REQ));
            SS_SYS_PORT_MSG_IND(message_p).userId = userId;
            rc = itti_send_msg_to_task(TASK_SYS, 0, message_p);
            if (rc == 0 )
            {
              sys_confirm_done = -1;
              /* wait for signal */
              wait_sys_confirm("TASK_SYS");
            }
          }
        }
        sys_req = NULL;
      }
    }
    acpSysProcessFreeSrv(req);
    LOG_D(ENB_SS_PORTMAN_ACP, "Exit from fxn:%s\n", __FUNCTION__);

    acpFree(buffer);
    return;
}

/*
 * Function : ss_port_man_process_itti_msg
 * Description: Function to process ITTI messages received from the TTCN
 * In :
 * req  - request recived from the TTCN
 * Out:
 * newState: No impact on state machine.
 *
 */
void *ss_port_man_process_itti_msg(void *notUsed)
{
    MessageDef *received_msg = NULL;
    int result = 0;
    LOG_D(ENB_SS_PORTMAN, "Entry in fxn:%s\n", __FUNCTION__);
    itti_poll_msg(TASK_SS_PORTMAN, &received_msg);
    if (received_msg != NULL)
    {
        LOG_A(ENB_SS_PORTMAN, "Received a message id : %d \n",
              ITTI_MSG_ID(received_msg));
        switch (ITTI_MSG_ID(received_msg))
        {
        case SS_SET_TIM_INFO:
        {
            LOG_A(ENB_SS_PORTMAN, "Received timing info \n");
            ss_port_man_send_data(0, 0, &received_msg->ittiMsg.ss_set_timinfo);
            result = itti_free(ITTI_MSG_ORIGIN_ID(received_msg), received_msg);
        }
        break;

        case SS_SYS_PORT_MSG_CNF:
        {
            LOG_A(ENB_SS_PORTMAN, "Received SS_SYS_PORT_MSG_CNF \n");
            ss_port_man_send_cnf(*(SS_SYS_PORT_MSG_CNF(received_msg).cnf));
            result = itti_free(ITTI_MSG_ORIGIN_ID(received_msg), received_msg);
        }
        break;

        case TERMINATE_MESSAGE:
            itti_exit_task();
            break;

        default:
            LOG_A(ENB_SS_PORTMAN, "Received unhandled message %d:%s\n",
                  ITTI_MSG_ID(received_msg), ITTI_MSG_NAME(received_msg));
            result = itti_free(ITTI_MSG_ORIGIN_ID(received_msg), received_msg);
            break;
        }

        AssertFatal(result == EXIT_SUCCESS, "Failed to free memory (%d)!\n",
                    result);
        received_msg = NULL;
    }

    /* Now handle notifications for other sockets */
    ss_eNB_read_from_socket(ctx_g);

    LOG_D(ENB_SS_PORTMAN, "Exit from fxn:%s\n", __FUNCTION__);
    return NULL;
}

/*
 * Function : ss_eNB_port_man_task
 * Description: The TASK_SS_PORTMAN main function handler. Initilizes
 * the TASK_SS_PORTMAN state machine Init_State. Invoke the itti message
 * In :
 * req  - request recived from the TTCN
 * Out:
 * newState: No impact on state machine.
 *
 */
void *ss_eNB_port_man_eNB_task(void *arg)
{
    ss_eNB_port_man_init();
    while (1)
    {
        /* Now handle notifications for other sockets */
        ss_port_man_process_itti_msg(NULL);
    }

    return NULL;
}

bool ss_eNB_port_man_handle_enquiryTiming(struct SYSTEM_CTRL_REQ *sys_req)
{
  struct SYSTEM_CTRL_CNF cnf;
  const size_t size = 16 * 1024;
  unsigned char *buffer = (unsigned char *)acpMalloc(size);
  int status = 0;

  if (!buffer)
    return false;

  size_t msgSize = size;
  memset(&cnf, 0, sizeof(cnf));
  cnf.Common.CellId = sys_req->Common.CellId;

  cnf.Common.RoutingInfo.d = RoutingInfo_Type_None;
  cnf.Common.RoutingInfo.v.None = true;

  cnf.Common.TimingInfo.d = TimingInfo_Type_SubFrame;
  cnf.Common.TimingInfo.v.SubFrame.SFN.d = SystemFrameNumberInfo_Type_Number;
  cnf.Common.TimingInfo.v.SubFrame.SFN.v.Number = SS_context.sfn;

  cnf.Common.TimingInfo.v.SubFrame.Subframe.d = SubFrameInfo_Type_Number;
  cnf.Common.TimingInfo.v.SubFrame.Subframe.v.Number = SS_context.sf;

  cnf.Common.TimingInfo.v.SubFrame.HSFN.d = SystemFrameNumberInfo_Type_Number;
  cnf.Common.TimingInfo.v.SubFrame.HSFN.v.Number = SS_context.hsfn;

  cnf.Common.TimingInfo.v.SubFrame.Slot.d = SlotTimingInfo_Type_FirstSlot;
  cnf.Common.TimingInfo.v.SubFrame.Slot.v.FirstSlot = true;

  cnf.Common.TimingInfo.v.SubFrame.Symbol.d = SymbolTimingInfo_Type_Any;
  cnf.Common.TimingInfo.v.SubFrame.Symbol.v.Any = true;

  cnf.Common.Result.d = ConfirmationResult_Type_Success;
  cnf.Common.Result.v.Success = true;

  cnf.Confirm.d = SystemConfirm_Type_EnquireTiming;
  cnf.Confirm.v.EnquireTiming = true;

  /* Encode message */
  if (acpSysProcessEncSrv(ctx_g, buffer, &msgSize, &cnf) != 0)
  {
    acpFree(buffer);
    return false;
  }

  /* Send message */
  status = acpSendMsg(ctx_g, msgSize, buffer);
  if (status != 0)
  {
    LOG_A(ENB_SS_PORTMAN, "acpSendMsg failed for EnquiryTiming.\n");
    acpFree(buffer);
    return false;
  }

  LOG_A(ENB_SS_PORTMAN, "enquiryTiming CNF sent successfully for SFN:%d SF:%d\n",
      cnf.Common.TimingInfo.v.SubFrame.SFN.v.Number,
      cnf.Common.TimingInfo.v.SubFrame.Subframe.v.Number);
  acpFree(buffer);
  return true;

}
