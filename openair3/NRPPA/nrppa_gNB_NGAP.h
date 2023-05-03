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

/*! \file nrppa_gNB_NGAP.h
 * \brief NRPPA NGAP procedures for gNB
 * \author Adeel
 * \date 2023
 * \version 0.1
 * \email: adeel.malik@eurecom.fr
 */



int nrppa_gNB_process_NGAP_DOWNLINKUEASSOCIATEDNRPPA(MessageDef *msg_p, instance_t instance, mui_t *rrc_gNB_mui); //this will process the recevied nrppa pdu
void nrppa_gNB_send_NGAP_UPLINKUEASSOCIATEDNRPPA(   //this will process and send the nrppa pdu
  const protocol_ctxt_t    *const ctxt_pP,
  rrc_gNB_ue_context_t     *const ue_context_pP,
  NR_UL_DCCH_Message_t     *const ul_dcch_msg
);




int rrc_gNB_process_NGAP_DOWNLINK_NAS(MessageDef *msg_p, instance_t instance, mui_t *rrc_gNB_mui);

void rrc_gNB_send_NGAP_UPLINK_NAS(
  const protocol_ctxt_t    *const ctxt_pP,
  rrc_gNB_ue_context_t     *const ue_context_pP,
  NR_UL_DCCH_Message_t     *const ul_dcch_msg
);



#endif
