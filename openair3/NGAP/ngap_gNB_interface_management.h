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

#ifndef NGAP_GNB_INTERFACE_MANAGEMENT_H_
#define NGAP_GNB_INTERFACE_MANAGEMENT_H_

#include <netinet/in.h>
#include <netinet/sctp.h>
#include <stdint.h>
#include "ngap_gNB_defs.h"
#include "ngap_msg_includes.h"

#define MAX_NUM_SERVED_GUAMI 256
#define MAX_NUM_PLMN 12

typedef struct {
  struct served_guami_s guami[MAX_NUM_SERVED_GUAMI];
  int num_guami;
  struct plmn_support_s plmn[MAX_NUM_PLMN];
  int num_plmn;
  long relative_amf_capacity;
  char *amf_name;
} ng_setup_response_t;

int encode_ng_setup_request(ngap_gNB_instance_t *instance_p, ngap_gNB_amf_data_t *amf);

int decode_ng_setup_response(ng_setup_response_t *out, const NGAP_NGSetupResponse_t *container);

int ngap_gNB_handle_overload_stop(sctp_assoc_t assoc_id, uint32_t stream, NGAP_NGAP_PDU_t *pdu);

#endif /* NGAP_GNB_INTERFACE_MANAGEMENT_H_ */
