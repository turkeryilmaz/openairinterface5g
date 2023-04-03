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

#include "f1ap_common.h"
#include "f1ap_itti_messaging.h"

void f1ap_itti_send_sctp_data_req(bool isCu, instance_t instance, uint8_t *buffer,
                                  uint32_t buffer_length, uint16_t stream) {
  MessageDef      *message_p;
  sctp_data_req_t *sctp_data_req;
  message_p = SCTP_DATA_REQ_alloc(isCu ? TASK_CU_F1 : TASK_DU_F1, 0);
  sctp_data_req = SCTP_DATA_REQ_data(message_p);
  sctp_data_req->assoc_id      =  f1ap_assoc_id(isCu,instance);
  sctp_data_req->buffer        = buffer;
  sctp_data_req->buffer_length = buffer_length;
  sctp_data_req->stream = stream;
  LOG_D(F1AP, "Sending ITTI message to SCTP Task\n");
  itti_send_msg_to_task(TASK_SCTP, instance, message_p);
}

void f1ap_itti_send_sctp_close_association(bool isCu, instance_t instance) {
  MessageDef               *message_p = NULL;
  sctp_close_association_t *sctp_close_association_p = NULL;
  message_p = SCTP_CLOSE_ASSOCIATION_alloc(TASK_S1AP, 0);
  sctp_close_association_p = SCTP_CLOSE_ASSOCIATION_data(message_p);
  sctp_close_association_p->assoc_id      = f1ap_assoc_id(isCu,instance);
  itti_send_msg_to_task(TASK_SCTP, instance, message_p);
}

