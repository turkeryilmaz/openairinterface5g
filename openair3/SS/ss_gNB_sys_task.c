#include <pthread.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <stdlib.h>

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

#include "acpNrSys.h"
#include "gnb_config.h"
#include "ss_gNB_sys_task.h"
#include "ss_gNB_context.h"

#include "common/utils/LOG/ss-log.h"
#include "msc.h"

extern RAN_CONTEXT_t RC;

extern uint16_t ss_rnti_nr_g;
extern SSConfigContext_t SS_context;
typedef enum
{
  UndefinedMsg = 0,
  EnquireTiming = 1,
  CellConfig = 2
} sidl_msg_id;

bool reqCnfFlag_g = false;


/*
 * Function : send_sys_cnf
 * Description: Funtion to build and send the SYS_CNF
 * In :
 * resType - Result type of the requested command
 * resVal  - Result value Success/Fail for the command
 * cnfType - Confirmation type for the Request received
 *           needed by TTCN to map to the Request sent.
 */
static void send_sys_cnf(enum ConfirmationResult_Type_Sel resType,
                         bool resVal,
                         enum NR_SystemConfirm_Type_Sel cnfType,
                         void *msg)
{
  struct NR_SYSTEM_CTRL_CNF *msgCnf = CALLOC(1, sizeof(struct NR_SYSTEM_CTRL_CNF));
  MessageDef *message_p = itti_alloc_new_message(TASK_SYS_GNB, INSTANCE_DEFAULT, SS_NR_SYS_PORT_MSG_CNF);

  /* The request has send confirm flag flase so do nothing in this funciton */
  if (reqCnfFlag_g == FALSE)
  {
     LOG_A(GNB_APP, "[SYS-GNB] No confirm required\n");
     return ;
  }

  if (message_p)
  {
    LOG_A(GNB_APP, "[SYS-GNB] Send SS_NR_SYS_PORT_MSG_CNF\n");
    msgCnf->Common.CellId = SS_context.eutra_cellId;
    msgCnf->Common.Result.d = resType;
    msgCnf->Common.Result.v.Success = resVal;
    msgCnf->Confirm.d = cnfType;
    switch (cnfType)
    {
    case NR_SystemConfirm_Type_Cell:
    {
      LOG_A(GNB_APP, "[SYS-GNB] Send confirm for cell configuration\n");
      msgCnf->Confirm.v.Cell = true;
      break;
    }
    default:
      LOG_A(GNB_APP, "[SYS-GNB] Error not handled CNF TYPE to [SS-PORTMAN-GNB]");
    }
    SS_NR_SYS_PORT_MSG_CNF(message_p).cnf = msgCnf;
    int send_res = itti_send_msg_to_task(TASK_SS_PORTMAN_GNB, INSTANCE_DEFAULT, message_p);
    if (send_res < 0)
    {
      LOG_A(GNB_APP, "[SYS-GNB] Error sending to [SS-PORTMAN-GNB]");
    }
  }
}
/*
 * Function : sys_handle_nr_enquire_timing
 * Description: Sends the NR enquire timing update to PORTMAN
 */
static void sys_handle_nr_enquire_timing(ss_nrset_timinfo_t *tinfo)
{
  MessageDef *message_p = itti_alloc_new_message(TASK_SYS_GNB, INSTANCE_DEFAULT, SS_NRSET_TIM_INFO);
  if (message_p)
  {
    LOG_A(GNB_APP, "[SYS-GNB] Reporting info sfn:%d\t slot:%d.\n", tinfo->sfn, tinfo->slot);
    SS_NRSET_TIM_INFO(message_p).slot = tinfo->slot;
    SS_NRSET_TIM_INFO(message_p).sfn = tinfo->sfn;

    int send_res = itti_send_msg_to_task(TASK_SS_PORTMAN_GNB, INSTANCE_DEFAULT, message_p);
    if (send_res < 0)
    {
      LOG_A(GNB_APP, "[SYS-GNB] Error sending to [SS-PORTMAN-GNB]");
    }
  }
}
/* 
 * =========================================================================================================== 
 * Function Name: ss_task_sys_nr_handle_req
 * Parameter    : SYSTEM_CTRL_REQ *req, is the message having ASP Defination of NR_SYSTEM_CTRL_REQ (38.523-3)
 *                which is received on SIDL via TTCN.
 *                ss_set_timinfo_t *tinfo, is currently not used.
 * Description  : This function handles the SYS_PORT_NR configuration command received from TTCN via the PORTMAN.
 *                It applies the configuration on RAN Context for NR and sends the confirmation message to 
 *                PORTMAN.
 * Returns      : Void
 * ==========================================================================================================
*/
static void ss_task_sys_nr_handle_req(struct NR_SYSTEM_CTRL_REQ *req, ss_nrset_timinfo_t *tinfo)
{
	int enterState = RC.ss.State;
	if(req->Common.CellId)				
		SS_context.eutra_cellId = req->Common.CellId;
	LOG_A(GNB_APP, "[SYS-GNB] Current SS_STATE %d received SystemRequest_Type %d eutra_cellId %d cnf_flag %d\n",
			RC.ss.State, req->Request.d, SS_context.eutra_cellId, req->Common.ControlInfo.CnfFlag);
	switch (RC.ss.State)
	{
		case SS_STATE_NOT_CONFIGURED:
			if (req->Request.d == NR_SystemRequest_Type_Cell)
			{
				LOG_A(GNB_APP, "[SYS-GNB] NR_SystemRequest_Type_Cell received\n");
			}
			else
			{
				LOG_E(GNB_APP, "[SYS-GNB] Error ! SS_STATE %d  Invalid SystemRequest_Type %d received\n",
						RC.ss.State, req->Request.d);
			}
			break;
                case SS_STATE_CELL_ACTIVE:
                        if (req->Request.d == NR_SystemRequest_Type_EnquireTiming)
                        {
                                sys_handle_nr_enquire_timing(tinfo);
                                LOG_A(GNB_APP, "[SYS-GNB] NR_SystemRequest_Type_EnquireTiming received\n");
                        }
			else
			{
				LOG_E(GNB_APP, "[SYS-GNB] Error ! SS_STATE %d  Invalid SystemRequest_Type %d received\n",
                                                RC.ss.State, req->Request.d);
			}
                        break;
		default:
			LOG_E(GNB_APP, "[SYS-GNB] Error ! SS_STATE %d  Invalid SystemRequest_Type %d received\n",
					RC.ss.State, req->Request.d);
			break;
	}
	LOG_A(GNB_APP, "[SYS-GNB] SS_STATE %d New SS_STATE %d received SystemRequest_Type %d\n",
			enterState, RC.ss.State, req->Request.d);
}

/* 
 * =============================================================================================================
 * Function Name: valid_nr_sys_msg
 * Parameter    : SYSTEM_CTRL_REQ *req, is the message having ASP Defination of NR_SYSTEM_CTRL_REQ (38.523-3) 
 *                which is received on SIDL via TTCN.
 * Description  : This function validates the validity of System Control Request Type. On successfull validation,
 *                this function sends the dummy confirmation to PORTMAN which is further forwareded towards TTCN.
 * Returns      : TRUE if recevied command is supported by SYS state handler
 *                FALSE if received command is not supported by SYS handler 
 * ============================================================================================================
*/

bool valid_nr_sys_msg(struct NR_SYSTEM_CTRL_REQ *req)
{
  bool valid = FALSE;
  enum ConfirmationResult_Type_Sel resType = ConfirmationResult_Type_Success;
  bool resVal = TRUE;
  bool sendDummyCnf = TRUE;
  enum NR_SystemConfirm_Type_Sel cnfType = 0;

  LOG_A(GNB_APP, "[SYS-GNB] received req : %d for cell %d RC.ss.state %d \n",
        req->Request.d, req->Common.CellId, RC.ss.State);
  switch (req->Request.d)
  {
	  case NR_SystemRequest_Type_Cell:
		  if (RC.ss.State >= SS_STATE_NOT_CONFIGURED)
		  {
			  valid = TRUE;
			  sendDummyCnf = FALSE;
			  reqCnfFlag_g = req->Common.ControlInfo.CnfFlag;
		  }
		  else
		  {
			  cnfType = NR_SystemConfirm_Type_Cell;
		  }
		  break;
          case NR_SystemRequest_Type_EnquireTiming:
		  valid = TRUE;
		  sendDummyCnf = FALSE;
		  break;
	  default:
		  valid = FALSE;
		  sendDummyCnf = FALSE;
  }
  if (sendDummyCnf)
  {
    send_sys_cnf(resType, resVal, cnfType, NULL);
    LOG_A(GNB_APP, "[SYS-GNB] Sending Dummy OK Req %d cnTfype %d ResType %d ResValue %d\n",
          req->Request.d, cnfType, resType, resVal);
  }
  return valid;
}


/* 
 * =========================================================================================================== 
 * Function Name: ss_gNB_sys_process_itti_msg
 * Parameter    : notUsed, is a dummy parameter is not being used currently
 * Description  : This function is entry point function for TASK_SYS_5G_NR. This function process the received 
 *                messages from other module and invokes respective handler function
 * Returns      : Void
 * ==========================================================================================================
*/

void *ss_gNB_sys_process_itti_msg(void *notUsed)
{
  MessageDef *received_msg = NULL;
  int result;
  static ss_nrset_timinfo_t tinfo = {.sfn = 0xFFFF, .slot = 0xFF};

  itti_receive_msg(TASK_SYS_GNB, &received_msg);

  /* Check if there is a packet to handle */
  if (received_msg != NULL)
  {
    switch (ITTI_MSG_ID(received_msg))
    {
    case SS_NRUPD_TIM_INFO:
    {
      LOG_A(GNB_APP, "TASK_SYS_GNB received SS_NRUPD_TIM_INFO with sfn=%d slot=%d\n", SS_NRUPD_TIM_INFO(received_msg).sfn, SS_NRUPD_TIM_INFO(received_msg).slot);
      tinfo.slot = SS_NRUPD_TIM_INFO(received_msg).slot;
      tinfo.sfn = SS_NRUPD_TIM_INFO(received_msg).sfn;
    }
    break;

    case SS_NR_SYS_PORT_MSG_IND:
    {

      if (valid_nr_sys_msg(SS_NR_SYS_PORT_MSG_IND(received_msg).req))
      {
        ss_task_sys_nr_handle_req(SS_NR_SYS_PORT_MSG_IND(received_msg).req, &tinfo);
      }
      else
      {
        LOG_A(GNB_APP, "TASK_SYS_GNB: Not handled SYS_PORT message received \n");
      }
    }
    break;

    case TERMINATE_MESSAGE:
    {
      itti_exit_task();
      break;
    }
    default:
      LOG_A(GNB_APP, "TASK_SYS_GNB: Received unhandled message %d:%s\n",
            ITTI_MSG_ID(received_msg), ITTI_MSG_NAME(received_msg));
      break;
    }
    result = itti_free(ITTI_MSG_ORIGIN_ID(received_msg), received_msg);
    AssertFatal(result == EXIT_SUCCESS, "[SYS] Failed to free memory (%d)!\n", result);
    received_msg = NULL;
  }
  return NULL;
}

/*
 * Function : ss_gNB_sys_task
 * Description:  The SYS_TASK main function handler. Initilizes the UDP
 * socket towards the Proxy for the configuration updates. Initilizes
 * the SYS_TASK state machine Init_State. Invoke the itti message
 * handler for the SYY_PORT.
 */
void *ss_gNB_sys_task(void *arg)
{
  if (RC.ss.configured == 0)
  {
    RCconfig_nr_ssparam();
    RC.ss.configured = 1;
  }
  // Set the state to NOT_CONFIGURED for Cell Config processing mode
  if (RC.ss.mode == SS_SOFTMODEM)
  {
    RC.ss.State = SS_STATE_NOT_CONFIGURED;
  }
  // Set the state to CELL_ACTIVE for SRB processing mode
  else if (RC.ss.mode == SS_SOFTMODEM_SRB)
  {
    RC.ss.State = SS_STATE_CELL_ACTIVE;
  }

  while (1)
  {
    (void) ss_gNB_sys_process_itti_msg(NULL);
  }

  return NULL;
}
