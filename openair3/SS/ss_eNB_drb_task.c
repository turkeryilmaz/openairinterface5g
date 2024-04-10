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
#include "conversions.h"
#include "common/utils/system.h"
#include "queue.h"
#include "sctp_common.h"

#include "intertask_interface.h"
#include "common/ran_context.h"
#include "ss_eNB_multicell_helper.h"

#include "acpDrb.h"
#include "ss_eNB_context.h"
#include "ss_eNB_vt_timer_task.h"

extern RAN_CONTEXT_t RC;
static acpCtx_t ctx_drb_g = NULL;
extern SSConfigContext_t SS_context;

static unsigned char *buffer = NULL;
static const size_t size = 16 * 1024;
static instance_t instance_g = 0;

enum MsgUserId
{
        // user defined IDs should be an int number >= 1
        MSG_DrbProcessFromSS_userId = 1,
        MSG_DrbProcessToSS_userId,
};

static void ss_send_drb_data(ss_drb_pdu_ind_t *pdu_ind, int cell_index){
	struct DRB_COMMON_IND ind = {};
        uint32_t status = 0;

	LOG_A(ENB_SS_DRB, "[SS_DRB] Reported drb sdu_size:%d \t drb_id %d\n", pdu_ind->sdu_size, pdu_ind->drb_id);

	DevAssert(pdu_ind != NULL);
        DevAssert(pdu_ind->sdu_size >= 0);
        DevAssert(pdu_ind->drb_id >= 0);

	size_t msgSize = size;
        memset(&ind, 0, sizeof(ind));

	ind.Common.CellId = SS_context.SSCell_list[cell_index].eutra_cellId;

	//Populated the Routing Info
	ind.Common.RoutingInfo.d = RoutingInfo_Type_RadioBearerId;
	ind.Common.RoutingInfo.v.RadioBearerId.d = RadioBearerId_Type_Drb;
	ind.Common.RoutingInfo.v.RadioBearerId.v.Drb = pdu_ind->drb_id;

	//Populated the Timing Info
	ind.Common.TimingInfo.d = TimingInfo_Type_SubFrame;
	ind.Common.TimingInfo.v.SubFrame.SFN.d = SystemFrameNumberInfo_Type_Number;
	ind.Common.TimingInfo.v.SubFrame.SFN.v.Number = pdu_ind->frame;

	ind.Common.TimingInfo.v.SubFrame.Subframe.d = SubFrameInfo_Type_Number;
	ind.Common.TimingInfo.v.SubFrame.Subframe.v.Number = pdu_ind->subframe;

	ind.Common.TimingInfo.v.SubFrame.HSFN.d = SystemFrameNumberInfo_Type_Number;
	ind.Common.TimingInfo.v.SubFrame.HSFN.v.Number = SS_context.hsfn;

	ind.Common.TimingInfo.v.SubFrame.Slot.d = SlotTimingInfo_Type_Any;
	ind.Common.TimingInfo.v.SubFrame.Slot.v.Any = true;

	ind.Common.TimingInfo.v.SubFrame.Symbol.d = SymbolTimingInfo_Type_Any;
	ind.Common.TimingInfo.v.SubFrame.Symbol.v.Any = true;

	ind.Common.Status.d = IndicationStatus_Type_Ok;
	ind.Common.Status.v.Ok = true;

	ind.Common.RlcBearerRouting.d = true;
        ind.Common.RlcBearerRouting.v.d = RlcBearerRouting_Type_EUTRA;
        ind.Common.RlcBearerRouting.v.v.EUTRA = SS_context.SSCell_list[cell_index].eutra_cellId;
        LOG_A(ENB_SS_DRB, "[SS_DRB][DRB_COMMON_IND] data ind type %d \n",pdu_ind->data_type);

	if(pdu_ind->data_type == DRB_PdcpSdu)
	{
	   //Populating the PDCP SDU
	   ind.U_Plane.SubframeData.NoOfTTIs = 1;
	   ind.U_Plane.SubframeData.PduSduList.d = L2DataList_Type_PdcpSdu;
	   ind.U_Plane.SubframeData.PduSduList.v.PdcpSdu.d = 1;
	   LOG_A(ENB_SS_DRB, "[SS_DRB][DRB_COMMON_IND] PDCP SDU Count: %lu\n", ind.U_Plane.SubframeData.PduSduList.v.PdcpSdu.d);
	   for(int i = 0; i < ind.U_Plane.SubframeData.PduSduList.v.PdcpSdu.d; i++){
                   ind.U_Plane.SubframeData.PduSduList.v.PdcpSdu.v = CALLOC(1,(ind.U_Plane.SubframeData.PduSduList.v.PdcpSdu.d)*(sizeof(PDCP_SDU_Type)));
                   DevAssert(ind.U_Plane.SubframeData.PduSduList.v.PdcpSdu.v != NULL);
                   ind.U_Plane.SubframeData.PduSduList.v.PdcpSdu.v[i].d = pdu_ind->sdu_size;
                   ind.U_Plane.SubframeData.PduSduList.v.PdcpSdu.v[i].v = CALLOC(1,ind.U_Plane.SubframeData.PduSduList.v.PdcpSdu.v[i].d);
		   memcpy(ind.U_Plane.SubframeData.PduSduList.v.PdcpSdu.v[i].v, pdu_ind->sdu, pdu_ind->sdu_size);
	   }
        }else{
          if(pdu_ind->data_type == DRB_MacPdu)
	  {
	      //Populating the MAC PDU
              ind.U_Plane.SubframeData.NoOfTTIs = 1;
              ind.U_Plane.SubframeData.PduSduList.d = L2DataList_Type_MacPdu;
	      /* Only single indication expected in a PDU.To do if multiple indications*/
              ind.U_Plane.SubframeData.PduSduList.v.MacPdu.d = 1;
              ind.U_Plane.SubframeData.PduSduList.v.MacPdu.v = CALLOC(1,(ind.U_Plane.SubframeData.PduSduList.v.MacPdu.d)*(sizeof(MAC_PDUList_Type)));
	      /* Only single indication expected in a PDU.To do if multiple indications*/
              ind.U_Plane.SubframeData.PduSduList.v.MacPdu.v->Header.d = 1;
              ind.U_Plane.SubframeData.PduSduList.v.MacPdu.v->Header.v = CALLOC(1,(ind.U_Plane.SubframeData.PduSduList.v.MacPdu.v->Header.d)*(sizeof(MAC_Header_Type)));

              LOG_A(ENB_SS_DRB_ACP, "[SS_DRB][DRB_COMMON_IND] MAC PDU Received\n");
              uint8_t lcid = pdu_ind->drb_id + 2;
              /*Convert integer value to bit-octet(A bit packed as Byte)*/
              UINT8_TO_BIT_OCTET(ind.U_Plane.SubframeData.PduSduList.v.MacPdu.v->Header.v[0].LCID, lcid, LCID_BIT_OCTET_SIZE);
              ind.U_Plane.SubframeData.PduSduList.v.MacPdu.v->SduList.d = true;
              /* Only single indication expected in a PDU.To do if multiple indications*/
              ind.U_Plane.SubframeData.PduSduList.v.MacPdu.v->SduList.v.d = 1;
              ind.U_Plane.SubframeData.PduSduList.v.MacPdu.v->SduList.v.v = CALLOC(1,(ind.U_Plane.SubframeData.PduSduList.v.MacPdu.v->SduList.v.d)*(sizeof(MAC_SDUList_Type)));
              ind.U_Plane.SubframeData.PduSduList.v.MacPdu.v->SduList.v.v->d = pdu_ind->sdu_size;
              ind.U_Plane.SubframeData.PduSduList.v.MacPdu.v->SduList.v.v->v = CALLOC(1,pdu_ind->sdu_size);
              memcpy(ind.U_Plane.SubframeData.PduSduList.v.MacPdu.v->SduList.v.v->v, pdu_ind->sdu,pdu_ind->sdu_size);
          }
        }

	//Encode Message
	if (acpDrbProcessToSSEncSrv(ctx_drb_g, buffer, &msgSize, &ind) != 0)
        {
                LOG_A(ENB_SS_DRB, "[SS_DRB][DRB_COMMON_IND] acpDrbProcessToSSEncSrv Failure\n");
                return;
        }
	LOG_A(ENB_SS_DRB, "[SS_DRB][DRB_COMMON_IND] Buffer msgSize=%d (!!2) to EUTRACell %d", (int)msgSize,SS_context.SSCell_list[cell_index].eutra_cellId);

	//Send Message
	status = acpSendMsg(ctx_drb_g, msgSize, buffer);
	if (status != 0)
        {
                LOG_A(ENB_SS_DRB, "[SS_DRB][DRB_COMMON_IND] acpSendMsg failed. Error : %d on fd: %d\n", status, acpGetSocketFd(ctx_drb_g));
                return;
        }
	else
        {
                LOG_A(ENB_SS_DRB, "[SS_DRB][DRB_COMMON_IND] acpSendMsg Success \n");
        }

}

static void ss_task_handle_drb_pdu_req(struct DRB_COMMON_REQ *req,int cell_index)
{
  assert(req);
  MessageDef *message_p = itti_alloc_new_message(TASK_PDCP_ENB, 0, SS_DRB_PDU_REQ);
  assert(message_p);
  if (message_p)
  {
    /* Populate the message and send to eNB */
    SS_DRB_PDU_REQ(message_p).drb_id = req->Common.RoutingInfo.v.RadioBearerId.v.Drb;
    memset(SS_DRB_PDU_REQ(message_p).sdu, 0, SDU_SIZE);

    for(int i = 0; i < req->U_Plane.SubframeDataList.d; i++){
      if(req->U_Plane.SubframeDataList.v[i].PduSduList.d == L2DataList_Type_PdcpSdu){
        LOG_A(ENB_SS_DRB_ACP, "PDCP SDU Received in DRB_COMMON_REQ \n");
        for(int j = 0; j < req->U_Plane.SubframeDataList.v[i].PduSduList.v.PdcpSdu.d; j++){
          SS_DRB_PDU_REQ(message_p).sdu_size = req->U_Plane.SubframeDataList.v[i].PduSduList.v.PdcpSdu.v[j].d;
          SS_DRB_PDU_REQ(message_p).data_type = DRB_PdcpSdu;
          LOG_A(ENB_SS_DRB_ACP, "Length of PDCP SDU received in DRB_COMMON_REQ: %lu\n",  req->U_Plane.SubframeDataList.v[i].PduSduList.v.PdcpSdu.v[j].d);
          memcpy(SS_DRB_PDU_REQ(message_p).sdu, req->U_Plane.SubframeDataList.v[i].PduSduList.v.PdcpSdu.v[j].v, req->U_Plane.SubframeDataList.v[i].PduSduList.v.PdcpSdu.v[j].d);
        }
      }else{
        if(req->U_Plane.SubframeDataList.v[i].PduSduList.d == L2DataList_Type_MacPdu){
          SS_DRB_PDU_REQ(message_p).data_type = DRB_MacPdu;
          /*Length is omitted in header of many MAC PDU. Altrnatively get from SDU size */
          SS_DRB_PDU_REQ(message_p).sdu_size = req->U_Plane.SubframeDataList.v[i].PduSduList.v.MacPdu.v->SduList.v.v->d;
          LOG_A(ENB_SS_DRB_ACP, "MAC PDU Received in DRB_COMMON_REQ size: %lu \n",SS_DRB_PDU_REQ(message_p).sdu_size);
          for(int8_t k = 0; k < req->U_Plane.SubframeDataList.v[i].PduSduList.v.MacPdu.v->Header.d; k++){
            uint8_t lcid = 0;
	    /*Convert bit-octet(A bit packed as Byte) to integer value*/
            BIT_OCTET_TO_UINT8(lcid, req->U_Plane.SubframeDataList.v[i].PduSduList.v.MacPdu.v->Header.v[k].LCID, LCID_BIT_OCTET_SIZE);
            LOG_A(ENB_SS_DRB_ACP, "MAC PDU received in lcid: %lu \n", lcid);
            /* Ignore padding bytes & its LCID --- would be taken care by eNB MAC during DLSCH Header generation*/
            if(PADDING_BYTE_LCID == lcid){
              continue;
            }else{
              SS_DRB_PDU_REQ(message_p).drb_id = lcid;
              memcpy(SS_DRB_PDU_REQ(message_p).sdu, req->U_Plane.SubframeDataList.v[i].PduSduList.v.MacPdu.v->SduList.v.v->v, SS_DRB_PDU_REQ(message_p).sdu_size);
	    }
          }	
        }
      }
    }

    SS_DRB_PDU_REQ(message_p).rnti = SS_context.SSCell_list[cell_index].ss_rnti_g;
	if (!vt_timer_push_msg(&req->Common.TimingInfo, TASK_RRC_ENB,instance_g, message_p))
	{
		itti_send_msg_to_task(TASK_RRC_ENB, instance_g, message_p);
	}

  }
}

static void
ss_eNB_read_from_drb_socket(acpCtx_t ctx){

	size_t msgSize = size; //2
	int cell_index = 0;

	LOG_A(ENB_SS_DRB_ACP, "Entry in fxn:%s\n", __FUNCTION__);
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
			else if (userId == -ACP_PEER_DISCONNECTED){
				LOG_A(GNB_APP, "[SS_DRB] Peer ordered shutdown\n");
			}
			else if (userId == -ACP_PEER_CONNECTED){
				LOG_A(GNB_APP, "[SS_DRB] Peer connection established\n");
			}
			else
			{
				LOG_A(ENB_SS_DRB_ACP, "[SS_DRB] Invalid userId: %d \n", userId);
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
			LOG_A(ENB_SS_DRB_ACP, "[SS_DRB] DRB_COMMON_REQ Received msgSize: %d\n", msgSize);

			if (acpDrbProcessFromSSDecSrv(ctx, buffer, msgSize, &req) != 0)
			{
				LOG_A(ENB_SS_DRB_ACP, "[SS_DRB][DRB_COMMON_REQ] acpDrbProcessFromSSDecSrv Failed\n");
				break;
			}
			if(req->Common.CellId){
				cell_index = get_cell_index(req->Common.CellId, SS_context.SSCell_list);
				SS_context.SSCell_list[cell_index].eutra_cellId = req->Common.CellId;
				LOG_A(ENB_SS_DRB_ACP,"[SS_DRB] cell_index: %d eutra_cellId: %d PhysicalCellId: %d \n",cell_index,SS_context.SSCell_list[cell_index].eutra_cellId,SS_context.SSCell_list[cell_index].PhysicalCellId);
			}
			if(SS_context.SSCell_list[cell_index].State >= SS_STATE_CELL_ACTIVE)
			{
				LOG_A(ENB_SS_DRB_ACP, "[SS_DRB][DRB_COMMON_REQ] DRB_COMMON_REQ Received in CELL_ACTIVE\n");
				ss_task_handle_drb_pdu_req(req,cell_index);
			}
			else
			{
				LOG_W(ENB_SS_DRB_ACP, "[SS_DRB][DRB_COMMON_REQ] received in SS state %d \n", SS_context.SSCell_list[cell_index].State);
			}

			acpDrbProcessFromSSFreeSrv(req);
			LOG_A(ENB_SS_DRB_ACP, "Exit from fxn:%s at line:%d \n", __FUNCTION__, __LINE__);
			return;
		}
		else if (MSG_DrbProcessToSS_userId == userId)
		{
			LOG_A(ENB_SS_DRB_ACP, "[SS_DRB] DRB_COMMON_IND Received; ignoring \n");
			break;
		}

	}
	LOG_A(ENB_SS_DRB_ACP, "Exit from fxn:%s at line:%d \n", __FUNCTION__, __LINE__);
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
                                int cell_index=0;
                                if(received_msg->ittiMsg.ss_drb_pdu_ind.physCellId){
                                  cell_index = get_cell_index_pci(received_msg->ittiMsg.ss_drb_pdu_ind.physCellId, SS_context.SSCell_list);
                                  LOG_A(ENB_SS_DRB,"[SS_DRB] cell_index in SS_DRB_PDU_IND: %d PhysicalCellId: %d \n",cell_index,SS_context.SSCell_list[cell_index].PhysicalCellId);
                                }
				task_id_t origin_task = ITTI_MSG_ORIGIN_ID(received_msg);

				if (origin_task == TASK_SS_PORTMAN)
				{
					LOG_D(ENB_SS_DRB, "[SS_DRB] DUMMY WAKEUP recevied from PORTMAN state %d \n", SS_context.SSCell_list[cell_index].State);
				}
				else
	            {
					LOG_A(ENB_SS_DRB, "[SS_DRB] Received SS_DRB_PDU_IND from L2/L3\n");
					if (SS_context.SSCell_list[cell_index].State >= SS_STATE_CELL_ACTIVE)
	                {
						instance_g = ITTI_MSG_DESTINATION_INSTANCE(received_msg);
						ss_send_drb_data(&received_msg->ittiMsg.ss_drb_pdu_ind,cell_index);
					}
					else
					{
						LOG_A(ENB_SS_DRB, "ERROR [SS_DRB][SS_DRB_PDU_IND] received in SS state %d \n", SS_context.SSCell_list[cell_index].State);
					}

					result = itti_free(ITTI_MSG_ORIGIN_ID(received_msg), received_msg);
					AssertFatal(result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);
				};
				break;

			case TERMINATE_MESSAGE:
				LOG_A(ENB_SS_DRB, "[SS_DRB] Received TERMINATE_MESSAGE \n");
				itti_exit_task();
				break;

			default:
				LOG_A(ENB_SS_DRB, "[SS_DRB] Received unhandled message %d:%s\n",
						ITTI_MSG_ID(received_msg), ITTI_MSG_NAME(received_msg));
				break;
                        }
		}
	}
	else
	{
	}

	return NULL;

}

void ss_eNB_drb_init(void)
{
  LOG_A(ENB_SS_DRB_ACP, "[SS_DRB] Starting System Simulator DRB Thread \n");

  // Port number
  int port = RC.ss.Drbport;
  const struct acpMsgTable msgTable[] = {
    {"DrbProcessFromSS", MSG_DrbProcessFromSS_userId},
    {"DrbProcessToSS", MSG_DrbProcessToSS_userId},
    // The last element should be NULL
    {NULL, 0}};

  // Arena size to decode received message
  const size_t aSize = 128 * 1024;

  // Start listening server and get ACP context,
  // after the connection is performed, we can use all services
  int ret = acpServerInitWithCtx(RC.ss.DrbHost ? RC.ss.DrbHost : "127.0.0.1", port, msgTable, aSize, &ctx_drb_g);
  if (ret < 0)
  {
    LOG_A(ENB_SS_DRB_ACP, "[SS_DRB] Connection failure err=%d\n", ret);
    return;
  }
  int fd1 = acpGetSocketFd(ctx_drb_g);
  LOG_A(ENB_SS_DRB_ACP, "[SS_DRB] Connection performed : %d\n", fd1);

  buffer = (unsigned char *)acpMalloc(size);
  assert(buffer);

  itti_subscribe_event_fd(TASK_SS_DRB, fd1);

  itti_mark_task_ready(TASK_SS_DRB);

}

void *ss_eNB_drb_task(void *arg)
{
  while (1)
  {
    (void)ss_eNB_drb_process_itti_msg(NULL);
  }
  acpFree(buffer);
  return NULL;
}

void *ss_eNB_drb_acp_task(void *arg)
{
  ss_eNB_drb_init();
  while (1)
  {
    ss_eNB_read_from_drb_socket(ctx_drb_g);
  }
  //acpFree(buffer);

  return NULL;
}

