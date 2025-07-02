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

#ifndef _RRC_GNB_DRBS_H_
#define _RRC_GNB_DRBS_H_

#include <stdbool.h>
#include <stdint.h>
#include "e1ap_messages_types.h"
#include "nr_rrc_defs.h"

/// @brief retrieve the data structure representing DRB with ID drb_id of UE ue
drb_t *get_drb(seq_arr_t *seq, int id);

/// @brief retrieve PDU session of UE ue with ID id
void *find_pduSession(seq_arr_t *seq, int id);

/// @brief Add a new PDU session @param in to the list @param for sessions_ptr for UE @param rrc_ue_id
pdusession_t *add_pduSession(seq_arr_t **sessions_ptr, const int rrc_ue_id, pdusession_t *in);

/// @brief Update an established PDU Session in @param sessions_ptr (setup list) with modified @param mod session
bool update_pduSession(seq_arr_t **sessions_ptr, const pdusession_t *mod);

/// @brief get PDU session of UE ue through the DRB drb_id
pdusession_t *find_pduSession_from_drbId(gNB_RRC_UE_t *ue, seq_arr_t *seq, int drb_id);

rrc_pdusession_failed_t *add_failed_pduSession(seq_arr_t **sessions_ptr, const int rrc_ue_id, rrc_pdusession_failed_t in);

rrc_pdusession_release_t *add_pduSession_to_release(seq_arr_t **sessions_ptr, const int rrc_ue_id, rrc_pdusession_release_t in);

/// @brief set PDCP configuration in a bearer context management message
bearer_context_pdcp_config_t set_bearer_context_pdcp_config(const nr_pdcp_configuration_t pdcp, bool um_on_default_drb);

/// @brief Deep copy an instance of struct pdusession_t
void cp_pdusession(pdusession_t *dst, const pdusession_t *src);

void free_pdusession(void *ptr);

bool rm_pduSession(seq_arr_t *seq, int pdusession_id);

drb_t *add_rrc_drb(seq_arr_t **drb_ptr, drb_t in);

void release_pduSessions(gNB_RRC_INST *rrc, gNB_RRC_UE_t *ue);

void remove_drbs_by_pdu_session(seq_arr_t **drbs, int pdusession_id);

drb_t *find_drb(seq_arr_t *seq, int pdusession_id);

pdusession_t *find_active_pdu_session(gNB_RRC_UE_t *ue_p, int pdusession_id);

#endif
