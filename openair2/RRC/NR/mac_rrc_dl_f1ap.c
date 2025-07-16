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

#include <netinet/in.h>
#include <netinet/sctp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "assertions.h"
#include "f1ap_messages_types.h"
#include "intertask_interface.h"
#include "mac_rrc_dl.h"
#include "nr_rrc_defs.h"
#include "lib/f1ap_rrc_message_transfer.h"
#include "lib/f1ap_interface_management.h"
#include "lib/f1ap_ue_context.h"

static void f1_reset_cu_initiated_f1ap(sctp_assoc_t assoc_id, const f1ap_reset_t *reset)
{
  MessageDef *msg = itti_alloc_new_message(TASK_RRC_GNB, 0, F1AP_RESET);
  msg->ittiMsgHeader.originInstance = assoc_id;
  f1ap_reset_t *f1ap_msg = &F1AP_RESET(msg);
  *f1ap_msg = cp_f1ap_reset(reset);
  itti_send_msg_to_task(TASK_CU_F1, 0, msg);
}

static void f1_reset_acknowledge_du_initiated_f1ap(sctp_assoc_t assoc_id, const f1ap_reset_ack_t *ack)
{
  (void)ack;
  AssertFatal(false, "%s() not implemented yet\n", __func__);
}

static void f1_setup_response_f1ap(sctp_assoc_t assoc_id, const f1ap_setup_resp_t *resp)
{
  MessageDef *msg = itti_alloc_new_message(TASK_RRC_GNB, 0, F1AP_SETUP_RESP);
  msg->ittiMsgHeader.originInstance = assoc_id;
  f1ap_setup_resp_t *f1ap_msg = &F1AP_SETUP_RESP(msg);
  *f1ap_msg = cp_f1ap_setup_response(resp);
  itti_send_msg_to_task(TASK_CU_F1, 0, msg);
}

static void f1_setup_failure_f1ap(sctp_assoc_t assoc_id, const f1ap_setup_failure_t *fail)
{
  MessageDef *msg = itti_alloc_new_message(TASK_RRC_GNB, 0, F1AP_SETUP_FAILURE);
  msg->ittiMsgHeader.originInstance = assoc_id;
  f1ap_setup_failure_t *f1ap_msg = &F1AP_SETUP_FAILURE(msg);
  *f1ap_msg = cp_f1ap_setup_failure(fail);
  itti_send_msg_to_task(TASK_CU_F1, 0, msg);
}

static void gnb_du_configuration_update_ack_f1ap(sctp_assoc_t assoc_id, const f1ap_gnb_du_configuration_update_acknowledge_t *ack)
{
  MessageDef *msg = itti_alloc_new_message(TASK_RRC_GNB, 0, F1AP_GNB_DU_CONFIGURATION_UPDATE_ACKNOWLEDGE);
  msg->ittiMsgHeader.originInstance = assoc_id;
  f1ap_gnb_du_configuration_update_acknowledge_t *f1ap_msg = &F1AP_GNB_DU_CONFIGURATION_UPDATE_ACKNOWLEDGE(msg);
  *f1ap_msg = cp_f1ap_du_configuration_update_acknowledge(ack);
  itti_send_msg_to_task(TASK_CU_F1, 0, msg);
}

static void ue_context_setup_request_f1ap(sctp_assoc_t assoc_id, const f1ap_ue_context_setup_req_t *req)
{
  MessageDef *msg = itti_alloc_new_message(TASK_RRC_GNB, 0, F1AP_UE_CONTEXT_SETUP_REQ);
  msg->ittiMsgHeader.originInstance = assoc_id;
  F1AP_UE_CONTEXT_SETUP_REQ(msg) = cp_ue_context_setup_req(req);
  itti_send_msg_to_task(TASK_CU_F1, 0, msg);
}

static void ue_context_modification_request_f1ap(sctp_assoc_t assoc_id, const f1ap_ue_context_mod_req_t *req)
{
  MessageDef *msg = itti_alloc_new_message(TASK_RRC_GNB, 0, F1AP_UE_CONTEXT_MODIFICATION_REQ);
  msg->ittiMsgHeader.originInstance = assoc_id;
  F1AP_UE_CONTEXT_MODIFICATION_REQ(msg) = cp_ue_context_mod_req(req);
  itti_send_msg_to_task(TASK_CU_F1, 0, msg);
}

static void ue_context_modification_confirm_f1ap(sctp_assoc_t assoc_id, const f1ap_ue_context_modif_confirm_t *confirm)
{
  MessageDef *msg = itti_alloc_new_message(TASK_MAC_GNB, 0, F1AP_UE_CONTEXT_MODIFICATION_CONFIRM);
  msg->ittiMsgHeader.originInstance = assoc_id;
  f1ap_ue_context_modif_confirm_t *f1ap_msg = &F1AP_UE_CONTEXT_MODIFICATION_CONFIRM(msg);
  f1ap_msg->gNB_CU_ue_id = confirm->gNB_CU_ue_id;
  f1ap_msg->gNB_DU_ue_id = confirm->gNB_DU_ue_id;
  f1ap_msg->rrc_container = NULL;
  f1ap_msg->rrc_container_length = 0;
  if (confirm->rrc_container != NULL) {
    f1ap_msg->rrc_container = calloc(1, sizeof(*f1ap_msg->rrc_container));
    AssertFatal(f1ap_msg->rrc_container != NULL, "out of memory\n");
    memcpy(f1ap_msg->rrc_container, confirm->rrc_container, confirm->rrc_container_length);
    f1ap_msg->rrc_container_length = confirm->rrc_container_length;
  }
  itti_send_msg_to_task(TASK_CU_F1, 0, msg);
}

static void ue_context_modification_refuse_f1ap(sctp_assoc_t assoc_id, const f1ap_ue_context_modif_refuse_t *refuse)
{
  MessageDef *msg = itti_alloc_new_message(TASK_MAC_GNB, 0, F1AP_UE_CONTEXT_MODIFICATION_REFUSE);
  msg->ittiMsgHeader.originInstance = assoc_id;
  f1ap_ue_context_modif_refuse_t *f1ap_msg = &F1AP_UE_CONTEXT_MODIFICATION_REFUSE(msg);
  *f1ap_msg = *refuse;
  itti_send_msg_to_task(TASK_CU_F1, 0, msg);
}

static void ue_context_release_command_f1ap(sctp_assoc_t assoc_id, const f1ap_ue_context_rel_cmd_t *cmd)
{
  MessageDef *message_p = itti_alloc_new_message (TASK_RRC_GNB, 0, F1AP_UE_CONTEXT_RELEASE_CMD);
  message_p->ittiMsgHeader.originInstance = assoc_id;
  F1AP_UE_CONTEXT_RELEASE_CMD(message_p) = cp_ue_context_rel_cmd(cmd);
  itti_send_msg_to_task (TASK_CU_F1, 0, message_p);
}

static void dl_rrc_message_transfer_f1ap(sctp_assoc_t assoc_id, const f1ap_dl_rrc_message_t *dl_rrc)
{
  /* TODO call F1AP function directly? no real-time constraint here */

  MessageDef *message_p = itti_alloc_new_message (TASK_RRC_GNB, 0, F1AP_DL_RRC_MESSAGE);
  message_p->ittiMsgHeader.originInstance = assoc_id;
  f1ap_dl_rrc_message_t *msg = &F1AP_DL_RRC_MESSAGE(message_p);
  *msg = cp_dl_rrc_message_transfer(dl_rrc);
  itti_send_msg_to_task (TASK_CU_F1, 0, message_p);
}

void mac_rrc_dl_f1ap_init(nr_mac_rrc_dl_if_t *mac_rrc)
{
  mac_rrc->f1_reset = f1_reset_cu_initiated_f1ap;
  mac_rrc->f1_reset_acknowledge = f1_reset_acknowledge_du_initiated_f1ap;
  mac_rrc->f1_setup_response = f1_setup_response_f1ap;
  mac_rrc->f1_setup_failure = f1_setup_failure_f1ap;
  mac_rrc->gnb_du_configuration_update_acknowledge = gnb_du_configuration_update_ack_f1ap;
  mac_rrc->ue_context_setup_request = ue_context_setup_request_f1ap;
  mac_rrc->ue_context_modification_request = ue_context_modification_request_f1ap;
  mac_rrc->ue_context_modification_confirm = ue_context_modification_confirm_f1ap;
  mac_rrc->ue_context_modification_refuse = ue_context_modification_refuse_f1ap;
  mac_rrc->ue_context_release_command = ue_context_release_command_f1ap;
  mac_rrc->dl_rrc_message_transfer = dl_rrc_message_transfer_f1ap;
}
