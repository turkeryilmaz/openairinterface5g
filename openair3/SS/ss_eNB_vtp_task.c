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

#include "intertask_interface.h"
#include "common/ran_context.h"

#include "acpSys.h"
#include "ss_eNB_vtp_task.h"
#include "ss_eNB_context.h"

#include "ss_eNB_proxy_iface.h"
#include "ss_eNB_multicell_helper.h"
#include "SIDL_VIRTUAL_TIME_PORT.h"
#include "acpSysVT.h"
#include "enb_config.h"

extern RAN_CONTEXT_t RC;
int cellIndex = 0;

static acpCtx_t ctx_vtp_g = NULL;
extern SSConfigContext_t SS_context;
enum MsgUserId
{
    MSG_SysVTEnquireTimingAck_userId = 1,
    MSG_SysVTEnquireTimingUpd_userId = 2,
};

char *vtp_local_address = "127.0.0.1";
int vtp_proxy_send_port = 7776;
int vtp_proxy_recv_port = 7777;

int ss_eNB_vtp_init(void);

/*
 * Function : sys_send_init_udp
 * Description: Sends the UDP_INIT message to UDP_TASK to create the receiving socket
 * for the SYS_TASK from the Proxy for the configuration confirmations.
 */
static int vtp_send_init_udp(const vtp_udpSockReq_t *req)
{
  // Create and alloc new message
  MessageDef *message_p;
  message_p = itti_alloc_new_message(TASK_VTP, 0, UDP_INIT);
  if (message_p == NULL)
  {
    return -1;
  }
  UDP_INIT(message_p).port = req->port;
  //addr.s_addr = req->ss_ip_addr;
  UDP_INIT(message_p).address = req->address; //inet_ntoa(addr);
  LOG_A(ENB_SS_VTP, "Tx UDP_INIT IP addr %s (%x)\n", UDP_INIT(message_p).address, UDP_INIT(message_p).port);
  return itti_send_msg_to_task(TASK_UDP, 0, message_p);
}
//------------------------------------------------------------------------------
// Function to send response to the SIDL client
void ss_vtp_send_tinfo(
    task_id_t task_id,
    ss_set_timinfo_t *tinfo)
{
    struct VirtualTimeInfo_Type virtualTime;
    const size_t size = 16 * 1024;
    uint32_t status;

    unsigned char *buffer = (unsigned char *)acpMalloc(size);

    DevAssert(tinfo != NULL);


    size_t msgSize = size;
    memset(&virtualTime, 0, sizeof(virtualTime));
    virtualTime.Enable = true;
    virtualTime.TimingInfo.SFN.d = SystemFrameNumberInfo_Type_Number;
    virtualTime.TimingInfo.SFN.v.Number = tinfo->sfn;

    virtualTime.TimingInfo.Subframe.d = SubFrameInfo_Type_Number;
    virtualTime.TimingInfo.Subframe.v.Number = tinfo->sf;

    virtualTime.TimingInfo.HSFN.d = SystemFrameNumberInfo_Type_Number;
    virtualTime.TimingInfo.HSFN.v.Number = tinfo->hsfn;

    /** TODO: Always marking as first slot, need to check this */
    virtualTime.TimingInfo.Slot.d = SlotTimingInfo_Type_FirstSlot;
    virtualTime.TimingInfo.Slot.v.FirstSlot = 0;

    /* Encode message
     */
    if (acpSysVTEnquireTimingUpdEncSrv(ctx_vtp_g, buffer, &msgSize, &virtualTime) != 0)
    {
        acpFree(buffer);
        return;
    }

    /* Send message
     */
    status = acpSendMsg(ctx_vtp_g, msgSize, buffer);
    if (status != 0)
    {
        LOG_E(ENB_SS_VTP, "[SS-VTP] acpSendMsg failed. Error : %d on fd: %d the VTP at SS will be disabled\n",
              status, acpGetSocketFd(ctx_vtp_g));
        acpFree(buffer);
        return;
    }
    else
    {
        LOG_D(ENB_SS_VTP, "[SS-VTP] acpSendMsg VTP_Send Success SFN %d SF %d virtualTime.Enable %d\n",tinfo->sfn,tinfo->sf,virtualTime.Enable);
        SS_context.vtinfo = *tinfo;
    }
    // Free allocated buffer
    acpFree(buffer);
}
/*
 * Function : vtp_send_udp_msg
 * Description: Sends the UDP_INIT message to UDP_TASK to create the listening socket
 */
static int vtp_send_udp_msg(
    uint8_t *buffer,
    uint32_t buffer_len,
    uint32_t buffer_offset,
    uint32_t peerIpAddr,
    uint16_t peerPort)
{
  // Create and alloc new message
  MessageDef *message_p = NULL;
  udp_data_req_t *udp_data_req_p = NULL;
  message_p = itti_alloc_new_message(TASK_VTP, 0, UDP_DATA_REQ);

  if (message_p)
  {
    LOG_D(ENB_SS_VTP, "Sending UDP_DATA_REQ length %u offset %u buffer %d %d %d\n", buffer_len, buffer_offset, buffer[0], buffer[1], buffer[2]);
    udp_data_req_p = &message_p->ittiMsg.udp_data_req;
    udp_data_req_p->peer_address = peerIpAddr;
    udp_data_req_p->peer_port = peerPort;
    udp_data_req_p->buffer = buffer;
    udp_data_req_p->buffer_length = buffer_len;
    udp_data_req_p->buffer_offset = buffer_offset;
    return itti_send_msg_to_task(TASK_UDP, 0, message_p);
  }
  else
  {
    LOG_A(ENB_SS_VTP, "Failed Sending UDP_DATA_REQ length %u offset %u", buffer_len, buffer_offset);
    return -1;
  }
}

/*
 * Function : sys_send_proxy
 * Description: Sends the messages from SYS to proxy
 */
static void vtp_send_proxy(void *msg, int msgLen)
{
  LOG_D(ENB_SS_VTP, "In sys_send_proxy\n");
  uint32_t peerIpAddr;
  uint16_t peerPort = vtp_proxy_send_port;

  IPV4_STR_ADDR_TO_INT_NWBO(vtp_local_address, peerIpAddr, " BAD IP Address");

  LOG_D(ENB_SS_VTP, "\nCell Config End of Buffer\n ");

  /** Send to proxy */
  vtp_send_udp_msg((uint8_t *)msg, msgLen, 0, peerIpAddr, peerPort);
  return;
}
static inline void ss_send_vtp_resp(struct VirtualTimeInfo_Type *virtualTime)
{
      VtpCmdReq_t *req = (VtpCmdReq_t *)malloc(sizeof(VtpCmdReq_t));
      req->header.preamble = 0xFEEDC0DE;
      req->header.msg_id = SS_VTP_RESP;
      req->header.length = sizeof(proxy_ss_header_t);
      req->header.cell_id = SS_context.SSCell_list[cellIndex].PhysicalCellId;
      req->header.cell_index = cellIndex;
      req->tinfo.sfn = virtualTime->TimingInfo.SFN.v.Number;
      req->tinfo.sf = virtualTime->TimingInfo.Subframe.v.Number;

      if ((req->tinfo.sfn % 32) == 0) {
        LOG_D(ENB_SS_VTP, "VTP_ACK Command to proxy sent for cell_id: %d SFN %d SF %d\n",
                req->header.cell_id,req->tinfo.sfn ,req->tinfo.sf );
      }

      vtp_send_proxy((void *)req, sizeof(VtpCmdReq_t));
}

#define SS_VTP_ENABLE 11

static inline void ss_enable_vtp()
{
      VtpCmdReq_t *req = (VtpCmdReq_t *)malloc(sizeof(VtpCmdReq_t));
      req->header.preamble = 0xFEEDC0DE;
      req->header.msg_id = SS_VTP_ENABLE;
      req->header.length = sizeof(proxy_ss_header_t);
      req->header.cell_id = SS_context.SSCell_list[cellIndex].PhysicalCellId;

      req->tinfo.sfn = 0;
      req->tinfo.sf = 0;

      LOG_A(ENB_SS_VTP, "VTP_ENABLE Command to proxy sent for cell_id: %d SFN %d SF %d\n",
            req->header.cell_id,req->tinfo.sfn ,req->tinfo.sf);

      LOG_A(ENB_SS_VTP,"VTP_ENABLE Command to proxy sent for cell_id: %d\n",
            req->header.cell_id );
      vtp_send_proxy((void *)req, sizeof(VtpCmdReq_t));

}
//------------------------------------------------------------------------------
static bool isConnected = false;
static inline void ss_eNB_read_from_vtp_socket(acpCtx_t ctx, bool vtInit)
{
    struct VirtualTimeInfo_Type *virtualTime = NULL;
    const size_t size = 16 * 1024;
    unsigned char *buffer = (unsigned char *)acpMalloc(size);
    assert(buffer);
    size_t msgSize = size; // 2

    assert(ctx);

    while (1)
    {
        int userId = acpRecvMsg(ctx, &msgSize, buffer);
        LOG_D(ENB_SS_VTP, "[SS-VTP] Received msgSize=%d, userId=%d\n", (int)msgSize, userId);

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
                LOG_E(GNB_APP, "[SS_VTP] ACP_ERR_SIDL_FAILURE, sidlStatus: %d\n", sidlStatus);
            }
            else if (userId == -ACP_PEER_DISCONNECTED){
                LOG_E(GNB_APP, "[SS_VTP] Peer ordered shutdown\n");
                isConnected = false;
            }
            else if (userId == -ACP_PEER_CONNECTED){
	            LOG_I(GNB_APP, "[SS_VTP] Peer connection established\n");
                isConnected = true;
            } else {
                LOG_E(GNB_APP, "[SS_VTP] ACP_ERR: %d \n", userId );
                ss_eNB_vtp_init();
            }
        }

        if (isConnected == false || vtInit == true)
        {
            // No message (timeout on socket)
            RC.ss.vtp_ready = 0;
            break;
        }

        if (userId == MSG_SysVTEnquireTimingAck_userId)
        {
            LOG_D(ENB_SS_VTP, "[SS-VTP] Received VTEnquireTimingAck Request\n");


            if (acpSysVTEnquireTimingAckDecSrv(ctx, buffer, msgSize, &virtualTime) != 0)
            {
                LOG_E(ENB_SS_VTP, "[SS-VTP] acpVngProcessDecSrv failed \n");
                break;
            }


            {
                if (virtualTime->Enable) {
                    ss_send_vtp_resp(virtualTime);

                    if (virtualTime->TimingInfo.SFN.d && (virtualTime->TimingInfo.SFN.v.Number % 32) == 0) {
                        LOG_D(ENB_SS_VTP, "[SS-VTP] SFN: %d\n",
                                virtualTime->TimingInfo.SFN.v.Number);

                        if (virtualTime->TimingInfo.Subframe.d) {
                            LOG_D(ENB_SS_VTP, "[SS-VTP] SubFrame: %d\n",
                                    virtualTime->TimingInfo.Subframe.v.Number);
                        }
                    }

				} else {
					ss_send_vtp_resp(virtualTime);
					LOG_W(ENB_SS_VTP, "[SS-VTP] disabled \n");
				}
				acpSysVTEnquireTimingAckFreeSrv(virtualTime);
				// TODo forward the message to sys_task ACK
				break;
            }
        }
    }
    acpFree(buffer);
}

void *ss_eNB_vtp_process_itti_msg(void *notUsed)
{
    MessageDef *received_msg = NULL;
    int result;

    itti_receive_msg(TASK_VTP, &received_msg);

    /* Check if there is a packet to handle */
    if (received_msg != NULL)
    {
        switch (ITTI_MSG_ID(received_msg))
        {
        case SS_UPD_TIM_INFO:
        {
            ss_set_timinfo_t tinfo;
            tinfo.hsfn = SS_context.hsfn; //it is supposed that SS_UPD_TIM_INFO processed by eNB_sys task firstly
            tinfo.sf = SS_UPD_TIM_INFO(received_msg).sf;
            tinfo.sfn = SS_UPD_TIM_INFO(received_msg).sfn;
            if(SS_UPD_TIM_INFO(received_msg).physCellId) {
              cellIndex = get_cell_index_pci(SS_UPD_TIM_INFO(received_msg).physCellId, SS_context.SSCell_list);
              LOG_A(ENB_SS,"[VTP] cellIndex in SS_UPD_TIM_INFO: %d PhysicalCellId: %d \n",cellIndex,SS_context.SSCell_list[cellIndex].PhysicalCellId);
            }

            if (isConnected == true) {
                if ((tinfo.sfn % 32) == 0 && tinfo.sf == 0) {
                    LOG_W(ENB_SS_VTP,"[VTP] received VTP_UPD_TIM_INFO SFN: %d SF: %d\n", tinfo.sfn, tinfo.sf);
                }
                ss_vtp_send_tinfo(TASK_VTP, &tinfo);
            }
        }
        break;


        case TERMINATE_MESSAGE:
        {
            LOG_E(ENB_SS_VTP, "[SS-VTP] Terminate message %d:%s\n", ITTI_MSG_ID(received_msg), ITTI_MSG_NAME(received_msg));
            itti_exit_task();
        }
        break;

        default:
            LOG_E(ENB_SS_VTP, "[SS-VTP] Received unhandled message %d:%s\n",
                  ITTI_MSG_ID(received_msg), ITTI_MSG_NAME(received_msg));
        }
        result = itti_free(ITTI_MSG_ORIGIN_ID(received_msg), received_msg);
        AssertFatal(result == EXIT_SUCCESS, "[SYS] Failed to free memory (%d)!\n", result);
        received_msg = NULL;
    }
    ss_eNB_read_from_vtp_socket(ctx_vtp_g, false);

    return NULL;
}

//------------------------------------------------------------------------------
int ss_eNB_vtp_init(void)
{
    // Port number
    int port = RC.ss.Vtpport;
    if (port == 0)
    {
        port = 7780;
    }

    LOG_A(ENB_SS_VTP, "[SS-VTP] Initializing VTP Port %s:%d\n", RC.ss.VtpHost, port);
    // acpInit(malloc, free, 1000);
    const struct acpMsgTable msgTable[] = {
        {"SysVTEnquireTimingAck", MSG_SysVTEnquireTimingAck_userId},
        {"SysVTEnquireTimingUpd", MSG_SysVTEnquireTimingUpd_userId},
        // The last element should be NULL
        {NULL, 0},
    };

    // Arena size to decode received message
    const size_t aSize = 128 * 1024;

    // Start listening server and get ACP context,
    // after the connection is performed, we can use all services
    int ret = acpServerInitWithCtx(RC.ss.VtpHost, port, msgTable, aSize, &ctx_vtp_g);
    if (ret < 0)
    {
        LOG_E(ENB_SS_VTP, "[SS-VTP] Connection failure err=%d\n", ret);
        return -1;
    }
#ifdef ACP_DEBUG_DUMP_MSGS /** TODO: Need to verify */
    adbgSetPrintLogFormat(ctx, true);
#endif
    int fd1 = acpGetSocketFd(ctx_vtp_g);
    LOG_A(ENB_SS_VTP, "[SS-VTP] Connected: %d\n", fd1);

    itti_mark_task_ready(TASK_VTP);
    return 0;
}
static void ss_eNB_wait_first_msg(void)
{
    unsigned char *buffer = (unsigned char *)acpMalloc(16 * 1024);
    assert(buffer);
	while (1)
	{
        ss_eNB_read_from_vtp_socket(ctx_vtp_g, true);
        if (isConnected == true){
            LOG_A(ENB_SS_VTP, "[SS_VTP] VT-HANDSHAKE with Client Completed (on-start) \n");
            RC.ss.vtp_ready = 1;
            break;
        }
        LOG_A(ENB_SS_VTP, "[SS_VTP] Waiting for VT-HANDSHAKE with Client(on-start) \n");
	}
}
//------------------------------------------------------------------------------
void* ss_eNB_vtp_task(void *arg) {
	vtp_udpSockReq_t req;
	req.address = vtp_local_address;
	req.port = vtp_proxy_recv_port;
	vtp_send_init_udp(&req);
	sleep(5);
	int retVal = ss_eNB_vtp_init();
	if (retVal != -1) {
		LOG_A(ENB_SS_VTP, "[SS-VTP] Enabled VTP starting the itti_msg_handler \n");

		ss_eNB_wait_first_msg();

		ss_enable_vtp();
		sleep(1);
		while (1) {
			(void) ss_eNB_vtp_process_itti_msg(NULL);
		}
	} else {

		LOG_A(ENB_SS_VTP, "[SS-VTP] VTP port disabled at eNB \n");
		sleep(10);
	}

	return NULL;
}
