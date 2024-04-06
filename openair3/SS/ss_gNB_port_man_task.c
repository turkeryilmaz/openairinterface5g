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

#include "ss_gNB_port_man_task.h"
#include "ss_gNB_context.h"
#include "acpNrSys.h"

extern RAN_CONTEXT_t RC;
extern int nr_cell_index;

acpCtx_t nrctx_g = NULL;
bool ss_gNB_portman_acp_task_exit = false;
pthread_mutex_t acp_msg_mutex;

enum MsgUserId
{
    MSG_NrSysProcess_userId = 1,
};
extern SSConfigContext_t SS_context;
static void ss_dumpReqMsg(struct NR_SYSTEM_CTRL_REQ *msg)
{
    LOG_A(GNB_APP, "NrSysProcess: received from the TTCN\n");
    LOG_A(GNB_APP, "\tCommon:\n");
    LOG_A(GNB_APP, "\t\tCellId=%d\n", msg->Common.CellId);
    LOG_A(GNB_APP, "\t\tRoutingInfo=%d\n", msg->Common.RoutingInfo.d);
    LOG_A(GNB_APP, "\t\tTimingInfo=%d\n", msg->Common.TimingInfo.d);
    LOG_A(GNB_APP, "\t\tCnfFlag=%d\n", msg->Common.ControlInfo.CnfFlag);
    LOG_A(GNB_APP, "\t\tFollowOnFlag=%d\n", msg->Common.ControlInfo.FollowOnFlag);
    LOG_A(GNB_APP, "\tRequest=%d\n", msg->Request.d);
}

void ss_nr_port_man_send_cnf(struct NR_SYSTEM_CTRL_CNF recvCnf)
{
    struct NR_SYSTEM_CTRL_CNF cnf;
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
    LOG_A(GNB_APP, "[SS-PORTMAN] Attn CNF received cellId %d result %d type %d \n",
                     cnf.Common.CellId,cnf.Common.Result.d, recvCnf.Confirm.d);
    switch (recvCnf.Confirm.d)
    {
    case NR_SystemConfirm_Type_Cell:
        cnf.Confirm.v.Cell = true;
        break;
    case NR_SystemConfirm_Type_RadioBearerList:
        cnf.Confirm.v.RadioBearerList= true;
        break;
    case NR_SystemConfirm_Type_CellAttenuationList:
        cnf.Confirm.v.CellAttenuationList= true;
        break;
    case NR_SystemConfirm_Type_PdcpCount:
        cnf.Confirm.v.PdcpCount.d = recvCnf.Confirm.v.PdcpCount.d;
        cnf.Confirm.v.PdcpCount.v = recvCnf.Confirm.v.PdcpCount.v;
        break;
    case NR_SystemConfirm_Type_AS_Security:
        cnf.Confirm.v.AS_Security = true;
        break;
    case NR_SystemConfirm_Type_DeltaValues:
        memcpy(&cnf.Confirm.v.DeltaValues, &recvCnf.Confirm.v.DeltaValues, sizeof(struct UE_NR_DeltaValues_Type));
        break;
    default:
        LOG_A(GNB_APP, "[SYS] Error not handled CNF TYPE to [SS-PORTMAN] %d \n", recvCnf.Confirm.d);
    }

    /* Encode message
     */
    if (acpNrSysProcessEncSrv(nrctx_g, buffer, &msgSize, &cnf) != 0)
    {
        acpFree(buffer);
        return;
    }
    /* Send message
     */
    status = acpSendMsg(nrctx_g, msgSize, buffer);
    if (status != 0)
    {
        LOG_A(GNB_APP, "[SS-PORTMAN] acpSendMsg failed. Error : %d on fd: %d\n",
              status, acpGetSocketFd(nrctx_g));
        acpFree(buffer);
        return;
    }
    else
    {
        LOG_A(GNB_APP, "[SS-PORTMAN] acpSendMsg Success \n");
    }
    // Free allocated buffer
    acpFree(buffer);
}

//------------------------------------------------------------------------------
// Function to send response to the SIDL client
void ss_nr_port_man_send_data(
    instance_t instance,
    task_id_t task_id,
    ss_nrset_timinfo_t *tinfo)
{
    struct NR_SYSTEM_CTRL_CNF cnf;
    const size_t size = 16 * 1024;
    uint32_t status;

    unsigned char *buffer = (unsigned char *)acpMalloc(size);

    DevAssert(tinfo != NULL);
    DevAssert(tinfo->sfn >= 0);
    DevAssert(tinfo->slot >= 0);

    size_t msgSize = size;
    memset(&cnf, 0, sizeof(cnf));
	/*TODO: */
    cnf.Common.CellId = SS_context.SSCell_list[nr_cell_index].nr_cellId;
    cnf.Common.RoutingInfo.d = NR_RoutingInfo_Type_None;
    cnf.Common.RoutingInfo.v.None = true;
    cnf.Common.TimingInfo.d = TimingInfo_Type_Now;
    cnf.Common.TimingInfo.v.Now = true;
    cnf.Common.Result.d = ConfirmationResult_Type_Success;
    cnf.Common.Result.v.Success = true;
    cnf.Confirm.d = NR_SystemConfirm_Type_EnquireTiming;
    cnf.Confirm.v.EnquireTiming = true;

    /**
   * FIXME: Currently filling only SFN and subframe numbers.
   */
    cnf.Common.TimingInfo.d = TimingInfo_Type_SubFrame;
    cnf.Common.TimingInfo.v.SubFrame.SFN.d = SystemFrameNumberInfo_Type_Number;
    cnf.Common.TimingInfo.v.SubFrame.SFN.v.Number = tinfo->sfn;

    cnf.Common.TimingInfo.v.SubFrame.Subframe.d = SubFrameInfo_Type_Number;
    cnf.Common.TimingInfo.v.SubFrame.Subframe.v.Number = tinfo->slot / 2;

    /** TODO: Always filling HSFN as 0, need to change this */
    cnf.Common.TimingInfo.v.SubFrame.HSFN.d = SystemFrameNumberInfo_Type_Number;
    cnf.Common.TimingInfo.v.SubFrame.HSFN.v.Number = 0;

    cnf.Common.TimingInfo.v.SubFrame.Slot.d = SlotTimingInfo_Type_SlotOffset;
    cnf.Common.TimingInfo.v.SubFrame.Slot.v.SlotOffset.d = SlotOffset_Type_Numerology1;
    cnf.Common.TimingInfo.v.SubFrame.Slot.v.SlotOffset.v.Numerology1  = tinfo->slot % 2;

    cnf.Common.TimingInfo.v.SubFrame.Symbol.d = SymbolTimingInfo_Type_Any;
    cnf.Common.TimingInfo.v.SubFrame.Symbol.v.Any = true;

    /* Encode message
     */
    if (acpNrSysProcessEncSrv(nrctx_g, buffer, &msgSize, &cnf) != 0)
    {
        acpFree(buffer);
        return;
    }

    /* Send message
     */
    status = acpSendMsg(nrctx_g, msgSize, buffer);
    if (status != 0)
    {
        LOG_A(GNB_APP, "[SS-PORTMAN] acpSendMsg failed. Error : %d on fd: %d\n",
              status, acpGetSocketFd(nrctx_g));
        acpFree(buffer);
        return;
    }
    else
    {
        LOG_A(GNB_APP, "[SS-PORTMAN GNB] acpSendMsg Success \n");
    }
    // Free allocated buffer
    acpFree(buffer);
}
//------------------------------------------------------------------------------
void ss_gNB_port_man_init(void)
{
    // Port number
    int port = RC.ss.SysportNR;

    acpInit(malloc, free, 1000);

    const struct acpMsgTable msgTable[] = {
        {"NrSysProcess", MSG_NrSysProcess_userId},
        // The last element should be NULL
        {NULL, 0}};
    // Arena size to decode received message
    const size_t aSize = 128 * 1024;

    // Start listening server and get ACP context,
    // after the connection is performed, we can use all services
    int ret = acpServerInitWithCtx(RC.ss.SysHost, port, msgTable, aSize, &nrctx_g);
    if (ret < 0)
    {
        LOG_A(GNB_APP, "[SS-PORTMAN-GNB] Connection failure err=%d\n", ret);
        return;
    }
    int fd1 = acpGetSocketFd(nrctx_g);
    LOG_A(GNB_APP, "[SS-PORTMAN-GNB] Connection performed : %d\n", fd1);

    //itti_subscribe_event_fd(TASK_SS_PORTMAN, fd1);

    itti_mark_task_ready(TASK_SS_PORTMAN_GNB);
}

//------------------------------------------------------------------------------
static inline void ss_gNB_read_from_socket(acpCtx_t ctx)
{
  struct NR_SYSTEM_CTRL_REQ *req = NULL;
  const size_t size = 16 * 1024;
  size_t msgSize = size; //2

  pthread_mutex_lock(&acp_msg_mutex);
  unsigned char *buffer = (unsigned char *)acpMalloc(size);
  DevAssert(buffer);

  int userId = acpRecvMsg(ctx, &msgSize, buffer);


  // Error handling
  if (userId < 0)
  {
    LOG_A(GNB_APP, "[SS-PORTMAN-ACP] fxn:%s userId:%d\n", __FUNCTION__, userId);
    if (userId == -ACP_ERR_SERVICE_NOT_MAPPED)
    {
      LOG_A(GNB_APP, "[SS-PORTMAN-ACP] fxn:%s userId:-ACP_ERR_SERVICE_NOT_MAPPED \n", __FUNCTION__);
      // Message not mapped to user id,
      // this error should not appear on server side for the messages
      // received from clients
    }
    else if (userId == -ACP_ERR_SIDL_FAILURE)
    {
      LOG_A(GNB_APP, "[SS-PORTMAN-ACP] fxn:%s userId:-ACP_ERR_SIDL_FAILURE\n", __FUNCTION__);
      // Server returned service error,
      // this error should not appear on server side for the messages
      // received from clients
      SidlStatus sidlStatus = -1;
      acpGetMsgSidlStatus(msgSize, buffer, &sidlStatus);
    }
    else if (userId == -ACP_PEER_DISCONNECTED){
      LOG_A(GNB_APP, "[SS-PORTMAN-ACP] Peer ordered shutdown\n");
    }
    else if (userId == -ACP_PEER_CONNECTED){
      LOG_A(GNB_APP, "[SS-PORTMAN-ACP] Peer connection established\n");
    }
    else
    {
      LOG_A(GNB_APP, "[SS-PORTMAN-ACP] fxn:%s line:%d\n", __FUNCTION__, __LINE__);
    }
    pthread_mutex_unlock(&acp_msg_mutex);
  }
  else if (userId == 0)
  {
    // No message (timeout on socket)
    if (RC.ss.mode >= SS_SOFTMODEM && SS_context.SSCell_list[nr_cell_index].State >= SS_STATE_CELL_ACTIVE)
    {
#if 0 /* Disable by Yangxiong. There is no need to send wake up signal to TASK_SS_SRB */
      LOG_A(ENB_SS,"[SS-PORTMAN] Sending Wake up signal/SS_NRRRC_PDU_IND (msg_Id:%d) to TASK_SS_SRB task \n", SS_NRRRC_PDU_IND);
      MessageDef *message_p = itti_alloc_new_message(TASK_SS_PORTMAN, 0, SS_NRRRC_PDU_IND);
      if (message_p)
      {
        /* Populate the message to SS */
        SS_NRRRC_PDU_IND(message_p).sdu_size = 1;
        SS_NRRRC_PDU_IND(message_p).srb_id = -1;
        SS_NRRRC_PDU_IND(message_p).rnti = -1;
        SS_NRRRC_PDU_IND(message_p).frame = -1;
        SS_NRRRC_PDU_IND(message_p).subframe = -1;

        int send_res = itti_send_msg_to_task(TASK_SS_SRB_GNB, 0, message_p);
        if (send_res < 0)
        {
          LOG_A(ENB_SS, "Error in sending Wake up signal /SS_NRRRC_PDU_IND (msg_Id:%d)  to TASK_SS_SRB_GNB\n", SS_NRRRC_PDU_IND);
        }
      }
#endif
    }
    pthread_mutex_unlock(&acp_msg_mutex);
  }
  else
  {
    LOG_A(GNB_APP, "[SS-PORTMAN-ACP] received msg %d from the client.\n", userId);
    if (acpNrSysProcessDecSrv(ctx, buffer, msgSize, &req) != 0)
    {
      LOG_A(GNB_APP, "[SS-PORTMAN-ACP] fxn:%s line:%d\n", __FUNCTION__, __LINE__);
      pthread_mutex_unlock(&acp_msg_mutex);
      return;
    }

    ss_dumpReqMsg(req);

    if (userId == MSG_NrSysProcess_userId)
    {
      LOG_A(GNB_APP, "[SS-PORTMAN-ACP] fxn:%s userId: MSG_NrSysProcess_userId\n", __FUNCTION__);
      MessageDef *message_p = itti_alloc_new_message(TASK_SS_PORTMAN_GNB, INSTANCE_DEFAULT,  SS_NR_SYS_PORT_MSG_IND);
      if (message_p)
      {
        SS_NR_SYS_PORT_MSG_IND(message_p).req = req;
        SS_NR_SYS_PORT_MSG_IND(message_p).userId = userId;
        itti_send_msg_to_task(TASK_SYS_GNB, INSTANCE_DEFAULT, message_p);
        LOG_A(GNB_APP, "[SS-PORTMAN-ACP] fxn:%s line:%d Msg sent to TASK_SYS_GNB \n", __FUNCTION__, __LINE__);
      }
    } else {
      pthread_mutex_unlock(&acp_msg_mutex);
    }
  }
  acpNrSysProcessFreeSrv(req);
  return;
}

//------------------------------------------------------------------------------
void *ss_port_man_5G_NR_process_itti_msg(void *notUsed)
{
	MessageDef *received_msg = NULL;
	int result = EXIT_SUCCESS;

	/* itti_poll_msg(TASK_SS_PORTMAN_GNB, &received_msg);*/
	itti_receive_msg(TASK_SS_PORTMAN_GNB, &received_msg);

	/* Check if there is a packet to handle */
	if (received_msg != NULL) {
		LOG_A(GNB_APP, "[SS-PORTMAN-GNB] Received a message id : %d \n", ITTI_MSG_ID(received_msg));
		switch (ITTI_MSG_ID(received_msg)) {
			case SS_NRSET_TIM_INFO:
			{
				LOG_A(GNB_APP, "[SS-PORTMAN-GNB] Received NR timing info \n");
				ss_nr_port_man_send_data(0, 0, &received_msg->ittiMsg.ss_nrset_timinfo);
				result = itti_free(ITTI_MSG_ORIGIN_ID(received_msg), received_msg);
			}
			break;

			case SS_NR_SYS_PORT_MSG_CNF:
			{
				LOG_A(GNB_APP, "[SS-PORTMAN-GNB] Received SS_NR_SYS_PORT_MSG_CNF \n");
				ss_nr_port_man_send_cnf(*(SS_NR_SYS_PORT_MSG_CNF(received_msg).cnf));
				result = itti_free(ITTI_MSG_ORIGIN_ID(received_msg), received_msg);
			}
			break;

			case TERMINATE_MESSAGE:
			{
				LOG_A(GNB_APP, "[SS-PORTMAN-GNB] Received TERMINATE_MESSAGE\n");
				ss_gNB_portman_acp_task_exit = true;
				itti_exit_task();
			}
			break;

			default:
				LOG_A(GNB_APP, "[SS-PORTMAN-GNB] Received unhandled message %d:%s\n",
						ITTI_MSG_ID(received_msg), ITTI_MSG_NAME(received_msg));
				break;
		}

		AssertFatal(result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);
		received_msg = NULL;
	}

	return NULL;
}

//------------------------------------------------------------------------------
void *ss_gNB_port_man_acp_task(void *arg)
{
	LOG_A(GNB_APP, "[SS_PORTMAN_ACP] Starting System Simulator SS_PORTMAN_ACP Thread \n");
	ss_gNB_port_man_init();
	while(1) {
		if(nrctx_g) {
			ss_gNB_read_from_socket(nrctx_g);
		} else {
			sleep(10);
		}

		if(ss_gNB_portman_acp_task_exit) {
			ss_gNB_portman_acp_task_exit = false;
			LOG_A(GNB_APP, "[SS_PORTMAN_ACP] TERMINATE \n");
			pthread_exit (NULL);
		}
	}

	return NULL;
}

void *ss_gNB_port_man_task(void *arg)
{
	LOG_A(GNB_APP, "[SS-PORTMAN-GNB] Starting GNB System Simulator Manager\n");

	while (1) {
		/* Now handle notifications for other sockets */
		(void)ss_port_man_5G_NR_process_itti_msg(NULL);
	}

	return NULL;
}
