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
#include "openair3/ocp-gtpu/gtp_itf.h"

drb_t *add_rrc_drb(seq_arr_t **drb_ptr, drb_t in)
{
  if (drb_ptr == NULL) {
    LOG_E(NR_RRC, "add_drb: Invalid input\n");
    return NULL;
  }

  SEQ_ARR_INIT(drb_ptr, drb_t, MAX_DRBS_PER_UE);

  return SEQ_ARR_PUSH_BACK_AND_GET(drb_t, *drb_ptr, &in);
}

static bool eq_drb_id(const void *vval, const void *vit)
{
  const int *id = (const int *)vval;
  const drb_t *elem = (const drb_t *)vit;
  return elem->drb_id == *id;
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

static bool eq_drb_pdu_session_id(const void *vval, const void *vit)
{
  const int *id = (const int *)vval;
  const drb_t *elem = (const drb_t *)vit;
  return elem->pdusession_id == *id;
}

/** @brief Finds the first DRB with the given PDU session ID.
 *  @return Pointer to matching drb_t or NULL if not found. */
drb_t *find_drb(seq_arr_t *seq, int pdusession_id)
{
  elm_arr_t elm = find_if(seq, &pdusession_id, eq_drb_pdu_session_id);
  if (elm.found)
    return (drb_t *)elm.it;
  return NULL;
}


/** @brief Removes all DRBs associated with a given PDU session ID */
void remove_drbs_by_pdu_session(seq_arr_t **drbs, int pdusession_id)
{
  if (drbs == NULL)
    return;

  drb_t *drb;
  while ((drb = find_drb(*drbs, pdusession_id)) != NULL) {
    LOG_I(NR_RRC, "Removing DRB ID %d associated with PDU Session ID %d\n", drb->drb_id, pdusession_id);
    seq_arr_erase(*drbs, drb);
  }

  if (!seq_arr_size(*drbs)) {
    SEQ_ARR_CLEANUP_AND_FREE(*drbs, NULL);
  }
}

static bool eq_pdu_session_id(const void *vval, const void *vit)
{
  const int *id = (const int *)vval;
  const pdusession_t *elem = (const pdusession_t *)vit;
  return elem->pdusession_id == *id;
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
  pdusession_t *session = (pdusession_t *)ptr;
  FREE_AND_ZERO_BYTE_ARRAY(session->nas_pdu);
}

/** @brief Retrieves PDU Session for the input ID
 *  @return pointer to the found PDU Session, NULL if not found */
void *find_pduSession(seq_arr_t *seq, int id)
{
  elm_arr_t elm = find_if(seq, &id, eq_pdu_session_id);
  if (elm.found)
    return elm.it;
  return NULL;
}

/** @brief Adds a new PDU Session to the list (either setup or addmod)
 *  @return pointer to the new PDU Session */
pdusession_t *add_pduSession(seq_arr_t **sessions_ptr, const int rrc_ue_id, pdusession_t *in)
{
  if (sessions_ptr == NULL || in == NULL) {
    LOG_E(NR_RRC, "add_pduSession: Invalid input\n");
    return NULL;
  }
  // Initialized seq_arr if necessary
  SEQ_ARR_INIT(sessions_ptr, pdusession_t, NGAP_MAX_PDU_SESSION);
  // Add item to the list
  pdusession_t *added = SEQ_ARR_PUSH_BACK_AND_GET(pdusession_t, *sessions_ptr, in);
  return added;
}

/** @brief Update an established PDU Session (setup list) after a "modify" procedure */
bool update_pduSession(seq_arr_t **sessions_ptr, const pdusession_t *mod)
{
  if (sessions_ptr == NULL || mod == NULL) {
    LOG_E(NR_RRC, "update_pduSession: Invalid input\n");
    return false;
  }

  pdusession_t *found = (pdusession_t *)find_pduSession(*sessions_ptr, mod->pdusession_id);
  if (!found) {
    LOG_E(NR_RRC, "PDU Session not found in the setup list (UE->pduSessions)\n");
    return false;
  }

  // Update the setup PDU Session with new status
  cp_pdusession(found, mod);

  return true;
}

/** @brief Adds a new PDU Session to the list
 *  @return pointer to the new PDU Session */
rrc_pdusession_release_t *add_pduSession_to_release(seq_arr_t **sessions_ptr, const int rrc_ue_id, rrc_pdusession_release_t in)
{
  if (sessions_ptr == NULL) {
    LOG_E(NR_RRC, "add_pduSession_to_release: Invalid input\n");
    return NULL;
  }

  SEQ_ARR_INIT(sessions_ptr, rrc_pdusession_release_t, NGAP_MAX_PDU_SESSION);

  rrc_pdusession_release_t *added = SEQ_ARR_PUSH_BACK_AND_GET(rrc_pdusession_release_t, *sessions_ptr, &in);
  LOG_I(NR_RRC, "Added PDU Session %d to release (total = %ld)\n", added->pdusession_id, seq_arr_size(*sessions_ptr));

  return added;
}

/** @brief Adds a new PDU Session to the list
 *  @return pointer to the new PDU Session */
rrc_pdusession_failed_t *add_failed_pduSession(seq_arr_t **sessions_ptr, const int rrc_ue_id, rrc_pdusession_failed_t in)
{
  if (sessions_ptr == NULL) {
    LOG_E(NR_RRC, "add_failed_pduSession: Invalid input\n");
    return NULL;
  }

  SEQ_ARR_INIT(sessions_ptr, rrc_pdusession_failed_t, NGAP_MAX_PDU_SESSION);

  rrc_pdusession_failed_t *added = SEQ_ARR_PUSH_BACK_AND_GET(rrc_pdusession_failed_t, *sessions_ptr, &in);
  LOG_I(NR_RRC, "Added failed PDU Session %d (total = %ld)\n", added->pdusession_id, seq_arr_size(*sessions_ptr));

  return added;
}

pdusession_t *find_pduSession_from_drbId(gNB_RRC_UE_t *ue, seq_arr_t *seq, int drb_id)
{
  const drb_t *drb = get_drb(ue->drbs, drb_id);
  if (!drb) {
    LOG_E(NR_RRC, "UE %d: DRB %d not found\n", ue->rrc_ue_id, drb_id);
    return NULL;
  }
  int id = drb->pdusession_id;
  return (pdusession_t *)find_pduSession(seq, id);
}

/** @brief Delete all N2 GTP tunnels for PDU Session to release
 *  Context: CU/CU-CP, so deletes NG-C (N2) tunnels only */
void release_pduSessions(gNB_RRC_INST *rrc, gNB_RRC_UE_t *ue)
{
  // GTP tunnel cleanup
  gtpv1u_gnb_delete_tunnel_req_t req = {0};
  req.ue_id = ue->rnti;
  FOR_EACH_SEQ_ARR(rrc_pdusession_release_t *, release, ue->pduSessions_to_release) {
    LOG_I(NR_RRC, "Delete GTP tunnels for UE %04x, PDU Session ID %d\n", ue->rnti, release->pdusession_id);
    req.pdusession_id[req.num_pdusession++] = release->pdusession_id;
  }
  gtpv1u_delete_ngu_tunnel(rrc->module_id, &req);
}

/** @brief Removes the PDU Session with the given ID from the list.
 *  @return true if a session was removed, false if not found. */
bool rm_pduSession(seq_arr_t *seq, int pdusession_id)
{
  if (seq == NULL) {
    LOG_E(NR_RRC, "rm_pduSession: seq is NULL\n");
    return false;
  }

  elm_arr_t elm = find_if(seq, &pdusession_id, eq_pdu_session_id);
  if (!elm.found) {
    LOG_W(NR_RRC, "rm_pduSession: PDU Session %d not found\n", pdusession_id);
    return false;
  }

  seq_arr_erase(seq, elm.it);  // shallow erase
  LOG_I(NR_RRC, "Removed PDU Session %d, remaining = %ld\n", pdusession_id, seq_arr_size(seq));
  return true;
}

bearer_context_pdcp_config_t set_bearer_context_pdcp_config(const nr_pdcp_configuration_t pdcp, bool um_on_default_drb)
{
  bearer_context_pdcp_config_t out = {0};
  out.pDCP_SN_Size_UL = encode_sn_size_ul(pdcp.drb.sn_size);
  out.pDCP_SN_Size_DL = encode_sn_size_dl(pdcp.drb.sn_size);
  out.discardTimer = encode_discard_timer(pdcp.drb.discard_timer);
  out.reorderingTimer = encode_t_reordering(pdcp.drb.t_reordering);
  out.rLC_Mode = um_on_default_drb ? E1AP_RLC_Mode_rlc_um_bidirectional : E1AP_RLC_Mode_rlc_am;
  out.pDCP_Reestablishment = false;
  return out;
}
