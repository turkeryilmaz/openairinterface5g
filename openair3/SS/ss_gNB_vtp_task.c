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
#include "ss_gNB_vtp_task.h"
#include "ss_gNB_context.h"

#include "ss_gNB_proxy_iface.h"
#include "SIDL_VIRTUAL_TIME_PORT.h"
#include "acpSysVT.h"
#include "ss_gNB_multicell_helper.h"
#include "enb_config.h"//bugz128620 gnb_config.h?
int nrcellIndex = 0;
extern RAN_CONTEXT_t RC;

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

bool ss_gNB_vtp_acp_task_ready_go = false;
bool ss_gNB_vtp_acp_task_exit = false;

static void _ss_log_vt(struct VirtualTimeInfo_Type* virtualTime, const char* prefix) {

    if (virtualTime->Enable) {
        char _msg[512] = {};
        char* _msg_end = _msg;

        if (virtualTime->TimingInfo.SFN.d) {
            _msg_end += snprintf(_msg_end, sizeof(_msg) - (_msg_end - _msg), "SFN: %d ",
                    virtualTime->TimingInfo.SFN.v.Number);
        }

        if (virtualTime->TimingInfo.HSFN.d) {
            _msg_end += snprintf(_msg_end, sizeof(_msg) - (_msg_end - _msg), "HSFN: %d ",
                    virtualTime->TimingInfo.HSFN.v.Number);
        }

        if (virtualTime->TimingInfo.Subframe.d) {
            _msg_end += snprintf(_msg_end, sizeof(_msg) - (_msg_end - _msg), "SubFrame: %d ",
                    virtualTime->TimingInfo.Subframe.v.Number);
        }

        if (virtualTime->TimingInfo.Slot.d == SlotTimingInfo_Type_SlotOffset) {
            _msg_end += snprintf(_msg_end, sizeof(_msg) - (_msg_end - _msg), "mu: %d ", virtualTime->TimingInfo.Slot.v.SlotOffset.d - 1);

            switch(virtualTime->TimingInfo.Slot.v.SlotOffset.d) {
                case SlotOffset_Type_Numerology0:  break;
                case SlotOffset_Type_Numerology1:
                    _msg_end += snprintf(_msg_end, sizeof(_msg) - (_msg_end - _msg), "slot(1): %d", virtualTime->TimingInfo.Slot.v.SlotOffset.v.Numerology1);
                break;
                case SlotOffset_Type_Numerology2:
                    _msg_end += snprintf(_msg_end, sizeof(_msg) - (_msg_end - _msg), "slot(2): %d", virtualTime->TimingInfo.Slot.v.SlotOffset.v.Numerology2);
                break;
                case SlotOffset_Type_Numerology3:
                    _msg_end += snprintf(_msg_end, sizeof(_msg) - (_msg_end - _msg), "slot(3): %d", virtualTime->TimingInfo.Slot.v.SlotOffset.v.Numerology3);
                break;
                case SlotOffset_Type_Numerology4:
                    _msg_end += snprintf(_msg_end, sizeof(_msg) - (_msg_end - _msg), "slot(4): %d", virtualTime->TimingInfo.Slot.v.SlotOffset.v.Numerology4);
                break;
                default:
                    LOG_E(GNB_APP, "Wrong MU\r\n");
                break;
            }
        }
        LOG_A(GNB_APP, "[SS-VTP] %s %s\n", prefix, _msg);
    } else {
        LOG_A(GNB_APP, "[SS-VTP] disabled \n");
    }
}

/*
 * Function : sys_send_init_udp
 * Description: Sends the UDP_INIT message to UDP_TASK to create the receiving socket
 * for the SYS_TASK from the Proxy for the configuration confirmations.
 */
static int vtp_send_init_udp(const vtp_udpSockReq_t *req)
{
  // Create and alloc new message
  MessageDef *message_p;
  message_p = itti_alloc_new_message(TASK_UDP, 0, UDP_INIT);
  if (message_p == NULL)
  {
    return -1;
  }
  UDP_INIT(message_p).port = req->port;
  //addr.s_addr = req->ss_ip_addr;
  UDP_INIT(message_p).address = req->address; //inet_ntoa(addr);
  LOG_A(GNB_APP, "Tx UDP_INIT IP addr %s (%x)\n", UDP_INIT(message_p).address, UDP_INIT(message_p).port);
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
    DevAssert(tinfo->sfn >= 0);
    DevAssert(tinfo->sf >= 0);

    size_t msgSize = size;
    memset(&virtualTime, 0, sizeof(virtualTime));
    virtualTime.Enable = true;

    virtualTime.TimingInfo.Slot.d = tinfo->mu > 0 ? SlotTimingInfo_Type_SlotOffset : SlotTimingInfo_Type_UNBOUND_VALUE;
    virtualTime.TimingInfo.Slot.v.SlotOffset.d = (enum SlotOffset_Type_Sel) (tinfo->mu + 1);

    switch(virtualTime.TimingInfo.Slot.v.SlotOffset.d) {
        case SlotOffset_Type_Numerology0:
            virtualTime.TimingInfo.Slot.v.SlotOffset.v.Numerology0 = true;
        break;
        case SlotOffset_Type_Numerology1:
            virtualTime.TimingInfo.Slot.v.SlotOffset.v.Numerology1 = tinfo->slot;
        break;
        case SlotOffset_Type_Numerology2:
            virtualTime.TimingInfo.Slot.v.SlotOffset.v.Numerology2 = tinfo->slot;
        break;
        case SlotOffset_Type_Numerology3:
            virtualTime.TimingInfo.Slot.v.SlotOffset.v.Numerology3 = tinfo->slot;
        break;
        case SlotOffset_Type_Numerology4:
            virtualTime.TimingInfo.Slot.v.SlotOffset.v.Numerology4 = tinfo->slot;
        break;;
        default:
            virtualTime.TimingInfo.Slot.d =  SlotTimingInfo_Type_UNBOUND_VALUE;
            break;;
    }

    virtualTime.TimingInfo.SFN.d = SystemFrameNumberInfo_Type_Number;
    virtualTime.TimingInfo.SFN.v.Number = tinfo->sfn;

    virtualTime.TimingInfo.Subframe.d = SubFrameInfo_Type_Number;
    virtualTime.TimingInfo.Subframe.v.Number = tinfo->sf;

    virtualTime.TimingInfo.HSFN.d = SystemFrameNumberInfo_Type_Number;
    virtualTime.TimingInfo.HSFN.v.Number = tinfo->hsfn;

    if (((tinfo->sfn % 32) == 0) && (tinfo->sf == 0) && (tinfo->slot == 0))
    {
        _ss_log_vt(&virtualTime, " <= ");
    }

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
        LOG_E(GNB_APP, "[SS-VTP] acpSendMsg failed. Error : %d on fd: %d the VTP at SS will be disabled\n",
              status, acpGetSocketFd(ctx_vtp_g));
        acpFree(buffer);
        SS_context.vtp_enabled = 0;

        return;
    }

    SS_context.vtinfo = *tinfo;

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
  message_p = itti_alloc_new_message(TASK_UDP, 0, UDP_DATA_REQ);

  if (message_p)
  {
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
    LOG_A(GNB_APP, "Failed Sending UDP_DATA_REQ length %u offset %u", buffer_len, buffer_offset);
    return -1;
  }
}

/*
 * Function : sys_send_proxy
 * Description: Sends the messages from SYS to proxy
 */
static void vtp_send_proxy(void *msg, int msgLen)
{
    uint32_t peerIpAddr;
    uint16_t peerPort = vtp_proxy_send_port;

    IPV4_STR_ADDR_TO_INT_NWBO(vtp_local_address, peerIpAddr, " BAD IP Address");

    /** Send to proxy */
    vtp_send_udp_msg((uint8_t *)msg, msgLen, 0, peerIpAddr, peerPort);
    return;
}

static void ss_send_vtp_resp(struct VirtualTimeInfo_Type *virtualTime)
{
    VtpCmdReq_t *req = (VtpCmdReq_t *)malloc(sizeof(VtpCmdReq_t));
    req->header.preamble = 0xFEEDC0DE;
    req->header.msg_id = SS_VTP_RESP;
    req->header.length = sizeof(proxy_ss_header_t);
    req->header.cell_id = SS_context.SSCell_list[nrcellIndex].PhysicalCellId;
    req->header.cell_index = 0;

    req->tinfo.mu = -1;

    if (virtualTime->TimingInfo.Slot.d != SlotTimingInfo_Type_UNBOUND_VALUE) {
        req->tinfo.mu = virtualTime->TimingInfo.Slot.d - 1;
        switch(virtualTime->TimingInfo.Slot.v.SlotOffset.d) {
            case SlotOffset_Type_Numerology0: break;
            case SlotOffset_Type_Numerology1:
                req->tinfo.slot = virtualTime->TimingInfo.Slot.v.SlotOffset.v.Numerology1;
            break;
            case SlotOffset_Type_Numerology2:
                req->tinfo.slot = virtualTime->TimingInfo.Slot.v.SlotOffset.v.Numerology2;
            break;
            case SlotOffset_Type_Numerology3:
                req->tinfo.slot = virtualTime->TimingInfo.Slot.v.SlotOffset.v.Numerology3;
            break;
            case SlotOffset_Type_Numerology4:
                req->tinfo.slot = virtualTime->TimingInfo.Slot.v.SlotOffset.v.Numerology4;
            break;
            default:
                req->tinfo.mu = -1;
            break;
        }
    }


    req->tinfo.sfn = virtualTime->TimingInfo.SFN.v.Number;
    req->tinfo.sf = virtualTime->TimingInfo.Subframe.v.Number;
    
    LOG_A(GNB_APP, "VTP_ACK Command to proxy sent for cell_id: %d SFN: %d SF: %d mu: %d slot: %d cell_index: %d\n",
        req->header.cell_id,req->tinfo.sfn ,req->tinfo.sf, req->tinfo.mu, req->tinfo.slot,req->header.cell_index);

    vtp_send_proxy((void *)req, sizeof(VtpCmdReq_t));

}

//------------------------------------------------------------------------------
static inline uint8_t ss_gNB_read_from_vtp_socket(acpCtx_t ctx)
{
    const size_t size = 16 * 1024;
    unsigned char *buffer = (unsigned char *)acpMalloc(size);
    assert(buffer);
    size_t msgSize = size; // 2

    assert(ctx);

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
                LOG_A(GNB_APP, "[SS-VTP-ACP] Peer ordered shutdown\n");
                return 1;
            }
            else if (userId == -ACP_PEER_CONNECTED){
	            LOG_A(GNB_APP, "[SS-VTP-ACP] Peer connection established\n");
            }
        }

        if (userId == 0)
        {
            // No message (timeout on socket)
            break;
        }
        else if (userId == MSG_SysVTEnquireTimingAck_userId)
        {
            struct VirtualTimeInfo_Type *virtualTime = NULL;
            if (acpSysVTEnquireTimingAckDecSrv(ctx, buffer, msgSize, &virtualTime) != 0)
            {
                LOG_E(GNB_APP, "[SS-VTP-ACP] acpVngProcessDecSrv failed \n");
                break;
            }

            AssertFatal(virtualTime != NULL, "VT struct is null (%p)", virtualTime);
            ss_send_vtp_resp(virtualTime);
            if ((virtualTime->TimingInfo.SFN.v.Number % 32) == 0 && (virtualTime->TimingInfo.Subframe.v.Number == 0)
                && (virtualTime->TimingInfo.Slot.v.SlotOffset.v.Numerology1 == 0)) {
                _ss_log_vt(virtualTime, " => ");
            }

            acpSysVTEnquireTimingAckFreeSrv(virtualTime);
            // TODo forward the message to sys_task ACK
            break;
        }
    }
    acpFree(buffer);
    return 0;
}

uint8_t ss_gNB_vtp_process_itti_msg(void)
{
	MessageDef *received_msg = NULL;
	int result;

	itti_receive_msg(TASK_VTP, &received_msg);

	/* Check if there is a packet to handle */
	if (received_msg != NULL) {
		switch (ITTI_MSG_ID(received_msg)) {
			case SS_NRUPD_TIM_INFO:
			{
				ss_set_timinfo_t tinfo;
				tinfo.mu = SS_context.mu;
				uint8_t slotsPerSubFrame = 1<<tinfo.mu;
				tinfo.slot = SS_NRUPD_TIM_INFO(received_msg).slot % slotsPerSubFrame;
				tinfo.sf = SS_NRUPD_TIM_INFO(received_msg).slot /slotsPerSubFrame;
				tinfo.sfn = SS_NRUPD_TIM_INFO(received_msg).sfn;
				tinfo.hsfn = SS_context.hsfn;

				if (SS_context.vtp_enabled == 1) {
					ss_vtp_send_tinfo(TASK_VTP, &tinfo);
				}
			}
			break;

			case TERMINATE_MESSAGE:
			{
				ss_gNB_vtp_acp_task_exit = true;
				itti_exit_task();
				break;
			}
			default:
				LOG_E(GNB_APP, "[SS-VTP] Received unhandled message %d:%s\n",
						ITTI_MSG_ID(received_msg), ITTI_MSG_NAME(received_msg));
		}
		result = itti_free(ITTI_MSG_ORIGIN_ID(received_msg), received_msg);
		AssertFatal(result == EXIT_SUCCESS, "[SYS] Failed to free memory (%d)!\n", result);
		received_msg = NULL;
	}

	return NULL;
}

//------------------------------------------------------------------------------
int ss_gNB_vtp_init(void)
{
    // Port number
    int port = RC.ss.Vtpport ? RC.ss.Vtpport : 7780;

    LOG_A(GNB_APP, "[SS-VTP] Initializing VTP Port %s:%d\n", RC.ss.hostIp, port);
    // acpInit(malloc, free, 1000);
    const struct acpMsgTable msgTable[] = {
        {"SysVTEnquireTimingAck", MSG_SysVTEnquireTimingAck_userId},
        {"SysVTEnquireTimingUpd", MSG_SysVTEnquireTimingUpd_userId},
        // The last element should be NULL
        {
            NULL, 0}};

    // Arena size to decode received message
    const size_t aSize = 128 * 1024;

    // Start listening server and get ACP context,
    // after the connection is performed, we can use all services
    LOG_W(GNB_APP, "[SS-VTP] Connecting to %s\n", RC.ss.VtpHost);
    int ret = acpServerInitWithCtx(RC.ss.VtpHost, port, msgTable, aSize, &ctx_vtp_g);
    if (ret < 0)
    {
        LOG_E(GNB_APP, "[SS-VTP] Connection failure err=%d\n", ret);
        return -1;
    }
#ifdef ACP_DEBUG_DUMP_MSGS /** TODO: Need to verify */
    adbgSetPrintLogFormat(ctx, true);
#endif
    int fd1 = acpGetSocketFd(ctx_vtp_g);
    LOG_A(GNB_APP, "[SS-VTP] Connected: %d\n", fd1);

    itti_mark_task_ready(TASK_VTP);
    return 0;
}

static void ss_gNB_vt_ena(void) {
    SS_context.vtp_enabled = 1;
    VtpCmdReq_t *req = (VtpCmdReq_t *)malloc(sizeof(VtpCmdReq_t));
    req->header.preamble = 0xFEEDC0DE;
    req->header.msg_id = SS_VTP_ENABLE;
    req->header.length = sizeof(proxy_ss_header_t);
    req->header.cell_id = SS_context.SSCell_list[nrcellIndex].PhysicalCellId;
    req->tinfo.mu = -1;
    req->tinfo.sfn = 0;
    req->tinfo.sf = 0;
    req->tinfo.slot = 0;
    req->header.cell_index = nrcellIndex;
    LOG_A(GNB_APP, "[SS-VTP] VT enable sent \n");
    vtp_send_proxy((void *)req, sizeof(VtpCmdReq_t));
}

static void ss_gNB_wait_first_msg(void)
{
	const size_t size = 16 * 1024;
	unsigned char *buffer = (unsigned char *)acpMalloc(size);
	assert(buffer);
	size_t msg_sz = size;
	while (1) {
		int ret = acpRecvMsg(ctx_vtp_g, &msg_sz, buffer);
		if (ret == MSG_SysVTEnquireTimingAck_userId || ret == -ACP_PEER_CONNECTED) {
			LOG_A(GNB_APP, "[SS_VTP] First VT-ACK From Client Received (on-start) \n");
			struct VirtualTimeInfo_Type *virtualTime = NULL;

			if (acpSysVTEnquireTimingAckDecSrv(ctx_vtp_g, buffer, msg_sz, &virtualTime) != 0) {
				LOG_E(GNB_APP, "[SS-VTP] acpVngProcessDecSrv failed \n");
				break;
			}

			if (virtualTime->Enable) {
				ss_gNB_vt_ena();
			}

			_ss_log_vt(virtualTime, " => (enable message) ");
			acpSysVTEnquireTimingAckFreeSrv(virtualTime);
			break;
		}
		LOG_A(GNB_APP, "[SS_VTP] Waiting for First VT-ACK From Client(on-start) \n");
	}
}
//------------------------------------------------------------------------------
void *ss_gNB_vtp_acp_task(void *arg)
{
	LOG_A(GNB_APP, "[SS-VTP-ACP] Starting System Simulator VTP_ACP Thread \n");

	while(ss_gNB_vtp_acp_task_ready_go == false) {
		sleep(5);
	}

	LOG_A(GNB_APP, "[SS-VTP-ACP] System Simulator VTP_ACP Thread Ready Go!\n");
	while(1) {
		if(ctx_vtp_g) {
			ss_gNB_read_from_vtp_socket(ctx_vtp_g);
		} else {
			sleep(10);
		}

		if(ss_gNB_vtp_acp_task_exit) {
			ss_gNB_vtp_acp_task_exit = false;
			ss_gNB_vtp_acp_task_ready_go = false;
			LOG_A(GNB_APP, "[SS-VTP-ACP] TERMINATE \n");
			pthread_exit (NULL);
		}
	}

	return NULL;
}

void* ss_gNB_vtp_task(void *arg) {
	vtp_udpSockReq_t req;
	req.address = vtp_local_address;
	req.port = vtp_proxy_recv_port;
	vtp_send_init_udp(&req);
	sleep(5);
	int retVal = ss_gNB_vtp_init();
	if (retVal != -1) {
		LOG_A(GNB_APP, "[SS-VTP] Enabled VTP starting the itti_msg_handler \n");

		ss_gNB_wait_first_msg();

		RC.ss.vtp_ready = 1;

		sleep(1);

		/* Ready to trigger SS-VTP-ACP task */
		ss_gNB_vtp_acp_task_ready_go = true;

		while (1) {
			if(ss_gNB_vtp_process_itti_msg()) {
				break;
			}
		}
	} else {
		LOG_A(GNB_APP, "[SS-VTP] VTP port disabled at gNB \n");
		sleep(10);
	}

	return NULL;
}
