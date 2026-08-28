/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include <string.h>
#include <stdlib.h>
#include "common/platform_types.h"
#include "common/utils/LOG/log.h"
#include "common/utils/ocp_itti/intertask_interface.h"
#include "assertions.h"
#include "openair2/COMMON/sctp_messages_types.h"
#include "openair2/COMMON/xnap_messages_types.h"
#include "xnap_default_values.h"
#include "xnap_common.h"
#include "xnap_gNB.h"

/* Create the Xn instance, bind a local SCTP listener (for incoming Xn
 * connections), and dial every configured candidate gNB */
static void xnap_gNB_handle_register_gnb(instance_t instance, xnap_register_gnb_req_t *req)
{
  xnap_create_inst(instance, &req->ng_setup_info, &req->net_config);
  xnap_gnb_inst_t *inst = xnap_get_inst(instance);

  const char *local_ip = inst->net_config.gnb_xn_interface_ip_address;
  size_t addr_len = strlen(local_ip) + 1;

  MessageDef *listen_msg = itti_alloc_new_message_sized(TASK_XNAP, instance, SCTP_INIT_MSG, sizeof(sctp_init_t) + addr_len);
  sctp_init_t *init = &SCTP_INIT_MSG(listen_msg);
  init->port = XNAP_PORT_NUMBER;
  init->ppid = XNAP_SCTP_PPID;
  char *addr_buf = (char *)(init + 1);
  init->bind_address = addr_buf;
  memcpy(addr_buf, local_ip, addr_len);
  itti_send_msg_to_task(TASK_SCTP, instance, listen_msg);

  const uint8_t nb_candidates = inst->net_config.nb_of_candidate_gNBs;
  LOG_I(XNAP, "[gNB %ld] Xn registered — listening on %s port %u, connecting to %u candidate(s)\n",
        instance, local_ip, XNAP_PORT_NUMBER, nb_candidates);

  for (uint16_t candidate_id = 0; candidate_id < nb_candidates; candidate_id++) {
    const char *remote_ip = inst->net_config.candidate_gnb_address_for_xnc[candidate_id];

    MessageDef *msg = itti_alloc_new_message(TASK_XNAP, instance, SCTP_NEW_ASSOCIATION_REQ);
    sctp_new_association_req_t *assoc_req = &msg->ittiMsg.sctp_new_association_req;

    assoc_req->ulp_cnx_id = candidate_id;
    assoc_req->port = XNAP_PORT_NUMBER;
    assoc_req->ppid = XNAP_SCTP_PPID;
    assoc_req->in_streams = inst->net_config.sctp_streams.sctp_in_streams;
    assoc_req->out_streams = inst->net_config.sctp_streams.sctp_out_streams;

    assoc_req->local_address.ipv4 = 1;
    strncpy(assoc_req->local_address.ipv4_address, local_ip, sizeof(assoc_req->local_address.ipv4_address) - 1);

    assoc_req->remote_address.ipv4 = 1;
    strncpy(assoc_req->remote_address.ipv4_address, remote_ip, sizeof(assoc_req->remote_address.ipv4_address) - 1);

    LOG_I(XNAP, "[gNB %ld] Initiating SCTP connection to candidate %u at %s port %u\n", instance, candidate_id, remote_ip, XNAP_PORT_NUMBER);

    itti_send_msg_to_task(TASK_SCTP, instance, msg);
  }
}

void *xnap_task(void *args)
{
  UNUSED(args);
  LOG_I(XNAP, "Starting XnAP task\n");
  itti_mark_task_ready(TASK_XNAP);

  while (1) {
    MessageDef *msg = NULL;
    itti_receive_msg(TASK_XNAP, &msg);
    const instance_t instance = ITTI_MSG_DESTINATION_INSTANCE(msg);
    const int msgType = ITTI_MSG_ID(msg);
    LOG_D(XNAP, "XnAP received %s for instance %ld\n", ITTI_MSG_NAME(msg), instance);

    switch (msgType) {
      case XNAP_REGISTER_GNB_REQ:
        xnap_gNB_handle_register_gnb(instance, &XNAP_REGISTER_GNB_REQ(msg));
        break;

      default:
        LOG_E(XNAP, "Unknown message type %d (%s)\n", msgType, ITTI_MSG_NAME(msg));
        break;
    }

    int result = itti_free(ITTI_MSG_ORIGIN_ID(msg), msg);
    AssertFatal(result == EXIT_SUCCESS, "Failed to free ITTI message (%d)\n", result);
    msg = NULL;
  }
}
