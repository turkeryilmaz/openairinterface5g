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

/*
 * nas_messages_types.h
 *
 *  Created on: Jan 07, 2014
 *      Author: winckel
 */

#ifndef NAS_MESSAGES_TYPES_H_
#define NAS_MESSAGES_TYPES_H_



#include "nas_message.h"

//-------------------------------------------------------------------------------------------//
#define NAS_DATA_LENGHT_MAX     256

typedef enum {
  EMM_MSG_HEADER = 1,
  EMM_MSG_ATTACH_REQUEST,
  EMM_MSG_ATTACH_ACCEPT,
  EMM_MSG_ATTACH_COMPLETE,
  EMM_MSG_ATTACH_REJECT,
  EMM_MSG_DETACH_REQUEST,
  EMM_MSG_DETACH_ACCEPT,
  EMM_MSG_TRACKING_AREA_UPDATE_REQUEST,
  EMM_MSG_TRACKING_AREA_UPDATE_ACCEPT,
  EMM_MSG_TRACKING_AREA_UPDATE_COMPLETE,
  EMM_MSG_TRACKING_AREA_UPDATE_REJECT,
  EMM_MSG_EXTENDED_SERVICE_REQUEST,
  EMM_MSG_SERVICE_REQUEST,
  EMM_MSG_SERVICE_REJECT,
  EMM_MSG_GUTI_REALLOCATION_COMMAND,
  EMM_MSG_GUTI_REALLOCATION_COMPLETE,
  EMM_MSG_AUTHENTICATION_REQUEST,
  EMM_MSG_AUTHENTICATION_RESPONSE,
  EMM_MSG_AUTHENTICATION_REJECT,
  EMM_MSG_AUTHENTICATION_FAILURE,
  EMM_MSG_IDENTITY_REQUEST,
  EMM_MSG_IDENTITY_RESPONSE,
  EMM_MSG_SECURITY_MODE_COMMAND,
  EMM_MSG_SECURITY_MODE_COMPLETE,
  EMM_MSG_SECURITY_MODE_REJECT,
  EMM_MSG_EMM_STATUS,
  EMM_MSG_EMM_INFORMATION,
  EMM_MSG_DOWNLINK_NAS_TRANSPORT,
  EMM_MSG_UPLINK_NAS_TRANSPORT,
  EMM_MSG_CS_SERVICE_NOTIFICATION,
} emm_message_ids_t;

typedef enum {
  ESM_MSG_HEADER = 1,
  ESM_MSG_ACTIVATE_DEFAULT_EPS_BEARER_CONTEXT_REQUEST,
  ESM_MSG_ACTIVATE_DEFAULT_EPS_BEARER_CONTEXT_ACCEPT,
  ESM_MSG_ACTIVATE_DEFAULT_EPS_BEARER_CONTEXT_REJECT,
  ESM_MSG_ACTIVATE_DEDICATED_EPS_BEARER_CONTEXT_REQUEST,
  ESM_MSG_ACTIVATE_DEDICATED_EPS_BEARER_CONTEXT_ACCEPT,
  ESM_MSG_ACTIVATE_DEDICATED_EPS_BEARER_CONTEXT_REJECT,
  ESM_MSG_MODIFY_EPS_BEARER_CONTEXT_REQUEST,
  ESM_MSG_MODIFY_EPS_BEARER_CONTEXT_ACCEPT,
  ESM_MSG_MODIFY_EPS_BEARER_CONTEXT_REJECT,
  ESM_MSG_DEACTIVATE_EPS_BEARER_CONTEXT_REQUEST,
  ESM_MSG_DEACTIVATE_EPS_BEARER_CONTEXT_ACCEPT,
  ESM_MSG_PDN_CONNECTIVITY_REQUEST,
  ESM_MSG_PDN_CONNECTIVITY_REJECT,
  ESM_MSG_PDN_DISCONNECT_REQUEST,
  ESM_MSG_PDN_DISCONNECT_REJECT,
  ESM_MSG_BEARER_RESOURCE_ALLOCATION_REQUEST,
  ESM_MSG_BEARER_RESOURCE_ALLOCATION_REJECT,
  ESM_MSG_BEARER_RESOURCE_MODIFICATION_REQUEST,
  ESM_MSG_BEARER_RESOURCE_MODIFICATION_REJECT,
  ESM_MSG_ESM_INFORMATION_REQUEST,
  ESM_MSG_ESM_INFORMATION_RESPONSE,
  ESM_MSG_ESM_STATUS,
} esm_message_ids_t;

typedef struct nas_raw_msg_s {
  uint32_t                        lenght;
  uint8_t                         data[NAS_DATA_LENGHT_MAX];
} nas_raw_msg_t;

typedef struct nas_emm_plain_msg_s {
  emm_message_ids_t               present;
  EMM_msg                         choice;

} nas_emm_plain_msg_t;

typedef struct nas_emm_protected_msg_s {
  nas_message_security_header_t   header;
  emm_message_ids_t               present;
  EMM_msg                         choice;
} nas_emm_protected_msg_t;

typedef struct nas_esm_plain_msg_s {
  esm_message_ids_t               present;
  ESM_msg                         choice;

} nas_esm_plain_msg_t;

typedef struct nas_esm_protected_msg_s {
  nas_message_security_header_t   header;
  esm_message_ids_t               present;
  ESM_msg                         choice;
} nas_esm_protected_msg_t;

#include "common/utils/ocp_itti/intertask_interface.h"
#define MESSAGE_DEF(iD, pRIO, sTRUCT)              \
  static inline sTRUCT *iD##_data(MessageDef *msg) \
  {                                                \
    return (sTRUCT *)msg->ittiMsg;                 \
  }
#include "openair2/COMMON/nas_messages_def.h"
#undef MESSAGE_DEF
#define MESSAGE_DEF(iD, pRIO, sTRUCT)                                                 \
  static inline MessageDef *iD##_alloc(task_id_t origintaskID, instance_t originINST) \
  {                                                                                   \
    return itti_alloc_sized(origintaskID, originINST, iD, sizeof(sTRUCT));            \
  }
#include "openair2/COMMON/nas_messages_def.h"
#undef MESSAGE_DEF

#endif /* NAS_MESSAGES_TYPES_H_ */
