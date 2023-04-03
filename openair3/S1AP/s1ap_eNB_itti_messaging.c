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

#include "s1ap_eNB_itti_messaging.h"
#include "openair2/COMMON/sctp_messages_types.h"
  
void s1ap_eNB_itti_send_sctp_data_req(instance_t instance, int32_t assoc_id, uint8_t *buffer,
                                      uint32_t buffer_length, uint16_t stream)
{
  MessageDef *message_p = SCTP_DATA_REQ_alloc(TASK_S1AP, 0);
  sctp_data_req_t *sctp_data_req = SCTP_DATA_REQ_data( message_p ); 

  sctp_data_req->assoc_id      = assoc_id;
  sctp_data_req->buffer        = buffer;
  sctp_data_req->buffer_length = buffer_length;
  sctp_data_req->stream        = stream;

  itti_send_msg_to_task(TASK_SCTP, instance, message_p);
}

void s1ap_eNB_itti_send_nas_downlink_ind(instance_t instance,
    uint16_t ue_initial_id,
    uint32_t eNB_ue_s1ap_id,
    uint8_t *nas_pdu,
    uint32_t nas_pdu_length)
{
  MessageDef *message_p = S1AP_DOWNLINK_NAS_alloc(TASK_S1AP, 0);

  s1ap_downlink_nas_t *s1ap_downlink_nas = S1AP_DOWNLINK_NAS_data(message_p);

  s1ap_downlink_nas->ue_initial_id  = ue_initial_id;
  s1ap_downlink_nas->eNB_ue_s1ap_id = eNB_ue_s1ap_id;
  s1ap_downlink_nas->nas_pdu.buffer = malloc(sizeof(uint8_t) * nas_pdu_length);
  memcpy(s1ap_downlink_nas->nas_pdu.buffer, nas_pdu, nas_pdu_length);
  s1ap_downlink_nas->nas_pdu.length = nas_pdu_length;

  itti_send_msg_to_task(TASK_RRC_ENB, instance, message_p);
}

void s1ap_eNB_itti_send_sctp_close_association(instance_t instance, int32_t assoc_id)
{
  MessageDef *message_p = SCTP_CLOSE_ASSOCIATION_alloc(TASK_S1AP, 0);
  SCTP_CLOSE_ASSOCIATION_data(message_p->ittiMsg)->assoc_id      = assoc_id;

  itti_send_msg_to_task(TASK_SCTP, instance, message_p);
}

