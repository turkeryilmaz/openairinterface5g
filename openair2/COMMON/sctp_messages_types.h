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

#ifndef SCTP_MESSAGES_TYPES_H_
#define SCTP_MESSAGES_TYPES_H_
#include "openair2/COMMON/s1ap_messages_types.h"

enum sctp_state_e {
  SCTP_STATE_CLOSED,
  SCTP_STATE_SHUTDOWN,
  SCTP_STATE_ESTABLISHED,
  SCTP_STATE_UNREACHABLE
};

typedef struct sctp_new_association_req_s {
  /* Upper layer connection identifier */
  uint16_t         ulp_cnx_id;

  /* The port to connect to */
  uint16_t         port;
  /* Payload Protocol Identifier to use */
  uint32_t         ppid;

  /* Number of streams used for this association */
  uint16_t in_streams;
  uint16_t out_streams;

  /* Local address to bind to */
  net_ip_address_t local_address;
  /* Remote address to connect to */
  net_ip_address_t remote_address;
} sctp_new_association_req_t;

typedef struct sctp_new_association_req_multi_s {
  /* Upper layer connection identifier */
  uint16_t         ulp_cnx_id;

  /* The port to connect to */
  uint16_t         port;
  /* Payload Protocol Identifier to use */
  uint32_t         ppid;

  /* Number of streams used for this association */
  uint16_t in_streams;
  uint16_t out_streams;

  /* Local address to bind to */
  net_ip_address_t local_address;
  /* Remote address to connect to */
  net_ip_address_t remote_address;

  /* Multi-socket descriptor */
  int multi_sd;
} sctp_new_association_req_multi_t;

typedef struct sctp_init_msg_multi_cnf_s {
  int multi_sd;
} sctp_init_msg_multi_cnf_t;

typedef struct sctp_new_association_ind_s {
  /* Assoc id of the new association */
  int32_t  assoc_id;

  /* The port used by remote host */
  uint16_t port;

  /* Number of streams used for this association */
  uint16_t in_streams;
  uint16_t out_streams;
} sctp_new_association_ind_t;

typedef struct sctp_new_association_resp_s {
  /* Upper layer connection identifier */
  uint16_t ulp_cnx_id;

  /* SCTP Association ID */
  int32_t  assoc_id;

  /* Input/output streams */
  uint16_t out_streams;
  uint16_t in_streams;

  /* State of the association at SCTP level */
  enum sctp_state_e sctp_state;
} sctp_new_association_resp_t;

typedef struct sctp_data_ind_s {
  /* SCTP Association ID */
  int32_t   assoc_id;

  /* Buffer to send over SCTP */
  uint32_t  buffer_length;
  uint8_t  *buffer;

  /* Streams on which data will be sent/received */
  uint16_t  stream;
} sctp_data_ind_t;

typedef struct sctp_init_s {
  /* Request usage of ipv4 */
  unsigned  ipv4:1;
  /* Request usage of ipv6 */
  unsigned  ipv6:1;
  uint8_t   nb_ipv4_addr;
  uint32_t  ipv4_address[10];
  uint8_t   nb_ipv6_addr;
  char     *ipv6_address[10];
  uint16_t  port;
  uint32_t  ppid;
} sctp_init_t;


typedef struct sctp_close_association_s {
  uint32_t  assoc_id;
} sctp_close_association_t;


typedef sctp_data_ind_t sctp_data_req_t;

typedef struct sctp_listener_register_upper_layer_s {
  /* Port to listen to */
  uint16_t port;
  /* Payload protocol identifier
   * Any data received on PPID != will be discarded
   */
  uint32_t ppid;
} sctp_listener_register_upper_layer_t;

#define MESSAGE_DEF(iD, pRIO, sTRUCT)              \
  static inline sTRUCT *iD##_data(MessageDef *msg) \
  {                                                \
    return (sTRUCT *)msg->ittiMsg;                 \
  }
#include "openair2/COMMON/sctp_messages_def.h"
#undef MESSAGE_DEF
#define MESSAGE_DEF(iD, pRIO, sTRUCT)                                                 \
  static inline MessageDef *iD##_alloc(task_id_t origintaskID, instance_t originINST) \
  {                                                                                   \
    return itti_alloc_sized(origintaskID, originINST, iD, sizeof(sTRUCT));            \
  }
#include "openair2/COMMON/sctp_messages_def.h"
 #undef MESSAGE_DEF

#endif /* SCTP_MESSAGES_TYPES_H_ */
