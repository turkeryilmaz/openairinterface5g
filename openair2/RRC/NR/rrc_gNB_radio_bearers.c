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

#include "rrc_gNB_radio_bearers.h"
#include <stddef.h>
#include "E1AP_RLC-Mode.h"
#include "RRC/NR/nr_rrc_defs.h"
#include "T.h"
#include "asn_internal.h"
#include "assertions.h"
#include "common/platform_constants.h"
#include "common/utils/T/T.h"
#include "ngap_messages_types.h"
#include "oai_asn1.h"
#include "openair2/LAYER2/nr_pdcp/nr_pdcp_asn1_utils.h"
#include "common/utils/alg/find.h"

static bool eq_pdu_session_id(const void *vval, const void *vit)
{
  const int id = *(const int *)vval;
  const rrc_pdu_session_param_t *elem = vit;
  return elem->param.pdusession_id == id;
}

/** @brief Deep copy pdusession_t */
void cp_pdusession(pdusession_t *dst, const pdusession_t *src)
{
  // Shallow copy
  *dst = *src;
  // nas_pdu
  dst->nas_pdu = copy_byte_array(src->nas_pdu);
}

/** @brief Free pdusession_t */
void free_pdusession(void *ptr)
{
  const rrc_pdu_session_param_t *elem = ptr;
  free_byte_array(elem->param.nas_pdu);
}

/** @brief Retrieves PDU Session for the input ID
 *  @return pointer to the found PDU Session, NULL if not found */
rrc_pdu_session_param_t *find_pduSession(seq_arr_t *seq, int id)
{
  DevAssert(seq);
  elm_arr_t elm = find_if(seq, &id, eq_pdu_session_id);
  if (elm.found)
    return (rrc_pdu_session_param_t *)elm.it;
  return NULL;
}

/** @brief Adds a new PDU Session to the list
 *  @return pointer to the new PDU Session */
rrc_pdu_session_param_t *add_pduSession(seq_arr_t *sessions_ptr, const int rrc_ue_id, const pdusession_t *in)
{
  DevAssert(sessions_ptr);
  DevAssert(in);

  if (seq_arr_size(sessions_ptr) == NGAP_MAX_PDU_SESSION) {
    LOG_I(NR_RRC, "Reached maximum number of PDU Session = %ld\n", seq_arr_size(sessions_ptr));
    return NULL;
  }

  rrc_pdu_session_param_t new = {0};
  cp_pdusession(&new.param, in);
  seq_arr_push_back(sessions_ptr, &new, sizeof(rrc_pdu_session_param_t));
  rrc_pdu_session_param_t *added = find_pduSession(sessions_ptr, in->pdusession_id);
  DevAssert(added);
  LOG_I(NR_RRC, "Added PDU Session %d, total number of PDU Sessions = %ld\n", in->pdusession_id, seq_arr_size(sessions_ptr));

  return added;
}

/** @brief Add drb_t item in the UE context list for @param pdusession_id */
drb_t *nr_rrc_add_drb(seq_arr_t *drb_ptr, int pdusession_id, nr_pdcp_configuration_t *pdcp)
{
  DevAssert(drb_ptr);

  // Get next available DRB ID
  int drb_id = seq_arr_size(drb_ptr) + 1;
  if (drb_id >= MAX_DRBS_PER_UE) {
    LOG_E(NR_RRC, "Cannot set up new DRB for pdusession_id=%d - reached maximum capacity\n", pdusession_id);
    return NULL;
  }

  // Add item to the list
  drb_t in = {.drb_id = drb_id, .pdusession_id = pdusession_id, .pdcp_config = *pdcp};
  seq_arr_push_back(drb_ptr, &in, sizeof(drb_t));
  drb_t *out = get_drb(drb_ptr, drb_id);
  DevAssert(out);
  LOG_I(NR_RRC, "Added DRB %d to established list (PDU Session ID=%d, total DRBs = %ld)\n", out->drb_id, pdusession_id, seq_arr_size(drb_ptr));
  return out;
}

static bool eq_drb_id(const void *vval, const void *vit)
{
  const int id = *(const int *)vval;
  const drb_t *elem = (const drb_t *)vit;
  return elem->drb_id == id;
}

/** @brief Retrieves DRB for the input ID
 *  @return pointer to the found DRB, NULL if not found */
drb_t *get_drb(seq_arr_t *seq, int id)
{
  DevAssert(id > 0 && id <= MAX_DRBS_PER_UE);
  elm_arr_t elm = find_if(seq, &id, eq_drb_id);
  if (elm.found)
    return (drb_t *)elm.it;
  return NULL;
}

/** @brief Free pdusession_t */
void free_drb(void *ptr)
{
  // do nothing
}

rrc_pdu_session_param_t *find_pduSession_from_drbId(gNB_RRC_UE_t *ue, int drb_id)
{
  const drb_t *drb = get_drb(&ue->drbs, drb_id);
  if (!drb) {
    LOG_E(NR_RRC, "UE %d: DRB %d not found\n", ue->rrc_ue_id, drb_id);
    return NULL;
  }
  int id = drb->pdusession_id;
  return find_pduSession(&ue->pduSessions, id);
}

bearer_context_pdcp_config_t set_bearer_context_pdcp_config(const nr_pdcp_configuration_t pdcp,
                                                            bool um_on_default_drb,
                                                            const nr_redcap_ue_cap_t *redcap_cap)
{
  bearer_context_pdcp_config_t out = {0};
  if (redcap_cap && redcap_cap->support_of_redcap_r17 && !redcap_cap->pdcp_drb_long_sn_redcap_r17) {
    LOG_I(NR_RRC, "UE is RedCap without long PDCP SN support: overriding PDCP SN size to 12\n");
    out.pDCP_SN_Size_DL = NR_PDCP_Config__drb__pdcp_SN_SizeDL_len12bits;
    out.pDCP_SN_Size_UL = NR_PDCP_Config__drb__pdcp_SN_SizeUL_len12bits;
  } else {
    out.pDCP_SN_Size_DL = encode_sn_size_dl(pdcp.drb.sn_size);
    out.pDCP_SN_Size_UL = encode_sn_size_ul(pdcp.drb.sn_size);
  }
  out.discardTimer = encode_discard_timer(pdcp.drb.discard_timer);
  out.reorderingTimer = encode_t_reordering(pdcp.drb.t_reordering);
  out.rLC_Mode = um_on_default_drb ? E1AP_RLC_Mode_rlc_um_bidirectional : E1AP_RLC_Mode_rlc_am;
  out.pDCP_Reestablishment = false;
  return out;
}
