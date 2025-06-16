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
  int id = drb->cnAssociation.sdap_config.pdusession_id;
  return (pdusession_t *)find_pduSession(seq, id);
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

void set_default_drb_pdcp_config(struct pdcp_config_s *pdcp_config,
                                 int do_drb_integrity,
                                 int do_drb_ciphering,
                                 const nr_pdcp_configuration_t *default_pdcp_config)
{
  AssertError(pdcp_config != NULL, return, "Failed to set default PDCP configuration for DRB!\n");
  pdcp_config->discardTimer = encode_discard_timer(default_pdcp_config->drb.discard_timer);
  pdcp_config->pdcp_SN_SizeDL = encode_sn_size_dl(default_pdcp_config->drb.sn_size);
  pdcp_config->pdcp_SN_SizeUL = encode_sn_size_ul(default_pdcp_config->drb.sn_size);
  pdcp_config->t_Reordering = encode_t_reordering(default_pdcp_config->drb.t_reordering);
  pdcp_config->headerCompression.present = NR_PDCP_Config__drb__headerCompression_PR_notUsed;
  pdcp_config->headerCompression.NotUsed = 0;
  pdcp_config->integrityProtection = do_drb_integrity ? NR_PDCP_Config__drb__integrityProtection_enabled : 1;
  pdcp_config->ext1.cipheringDisabled = do_drb_ciphering ? 1 : NR_PDCP_Config__ext1__cipheringDisabled_true;
}

void set_bearer_context_pdcp_config(bearer_context_pdcp_config_t *pdcp_config, drb_t *rrc_drb, bool um_on_default_drb)
{
  AssertError(rrc_drb != NULL && pdcp_config != NULL, return, "Failed to set default bearer context PDCP configuration!\n");
  pdcp_config->pDCP_SN_Size_UL = rrc_drb->pdcp_config.pdcp_SN_SizeUL;
  pdcp_config->pDCP_SN_Size_DL = rrc_drb->pdcp_config.pdcp_SN_SizeDL;
  pdcp_config->discardTimer = rrc_drb->pdcp_config.discardTimer;
  pdcp_config->reorderingTimer = rrc_drb->pdcp_config.t_Reordering;
  pdcp_config->rLC_Mode = um_on_default_drb ? E1AP_RLC_Mode_rlc_um_bidirectional : E1AP_RLC_Mode_rlc_am;
}

drb_t *generateDRB(gNB_RRC_UE_t *ue,
                   uint8_t drb_id,
                   const pdusession_t *pduSession,
                   bool enable_sdap,
                   int do_drb_integrity,
                   int do_drb_ciphering,
                   const nr_pdcp_configuration_t *pdcp_config)
{
  DevAssert(ue != NULL);

  LOG_I(NR_RRC, "UE %d: configure DRB ID %d for PDU session ID %d\n", ue->rrc_ue_id, drb_id, pduSession->pdusession_id);

  drb_t *est_drb = get_drb(ue->drbs, drb_id);
  est_drb->drb_id = drb_id;
  est_drb->cnAssociation.sdap_config.defaultDRB = true;

  /* SDAP Configuration */
  est_drb->cnAssociation.present = NR_DRB_ToAddMod__cnAssociation_PR_sdap_Config;
  est_drb->cnAssociation.sdap_config.pdusession_id = pduSession->pdusession_id;
  if (enable_sdap) {
    est_drb->cnAssociation.sdap_config.sdap_HeaderDL = NR_SDAP_Config__sdap_HeaderDL_present;
    est_drb->cnAssociation.sdap_config.sdap_HeaderUL = NR_SDAP_Config__sdap_HeaderUL_present;
  } else {
    est_drb->cnAssociation.sdap_config.sdap_HeaderDL = NR_SDAP_Config__sdap_HeaderDL_absent;
    est_drb->cnAssociation.sdap_config.sdap_HeaderUL = NR_SDAP_Config__sdap_HeaderUL_absent;
  }
  for (int qos_flow_index = 0; qos_flow_index < pduSession->nb_qos; qos_flow_index++) {
    est_drb->cnAssociation.sdap_config.mappedQoS_FlowsToAdd[qos_flow_index] = pduSession->qos[qos_flow_index].qfi;
  }
  /* PDCP Configuration */
  set_default_drb_pdcp_config(&est_drb->pdcp_config, do_drb_integrity, do_drb_ciphering, pdcp_config);

  drb_t *rrc_drb = get_drb(ue->drbs, drb_id);
  DevAssert(rrc_drb == est_drb); /* to double check that we create the same which we would retrieve */
  return rrc_drb;
}

NR_DRB_ToAddMod_t *generateDRB_ASN1(const drb_t *drb_asn1)
{
  NR_DRB_ToAddMod_t *DRB_config = CALLOC(1, sizeof(*DRB_config));
  NR_SDAP_Config_t *SDAP_config = CALLOC(1, sizeof(NR_SDAP_Config_t));

  asn1cCalloc(DRB_config->cnAssociation, association);
  asn1cCalloc(SDAP_config->mappedQoS_FlowsToAdd, sdapFlows);
  asn1cCalloc(DRB_config->pdcp_Config, pdcpConfig);
  asn1cCalloc(pdcpConfig->drb, drb);

  DRB_config->drb_Identity = drb_asn1->drb_id;
  association->present = drb_asn1->cnAssociation.present;

  /* SDAP Configuration */
  SDAP_config->pdu_Session = drb_asn1->cnAssociation.sdap_config.pdusession_id;
  SDAP_config->sdap_HeaderDL = drb_asn1->cnAssociation.sdap_config.sdap_HeaderDL;
  SDAP_config->sdap_HeaderUL = drb_asn1->cnAssociation.sdap_config.sdap_HeaderUL;
  SDAP_config->defaultDRB = drb_asn1->cnAssociation.sdap_config.defaultDRB;

  for (int qos_flow_index = 0; qos_flow_index < QOSFLOW_MAX_VALUE; qos_flow_index++) {
    if (drb_asn1->cnAssociation.sdap_config.mappedQoS_FlowsToAdd[qos_flow_index] != 0) {
      asn1cSequenceAdd(sdapFlows->list, NR_QFI_t, qfi);
      *qfi = drb_asn1->cnAssociation.sdap_config.mappedQoS_FlowsToAdd[qos_flow_index];
    }
  }

  association->choice.sdap_Config = SDAP_config;

  /* PDCP Configuration */
  asn1cCallocOne(drb->discardTimer, drb_asn1->pdcp_config.discardTimer);
  asn1cCallocOne(drb->pdcp_SN_SizeUL, drb_asn1->pdcp_config.pdcp_SN_SizeUL);
  asn1cCallocOne(drb->pdcp_SN_SizeDL, drb_asn1->pdcp_config.pdcp_SN_SizeDL);
  asn1cCallocOne(pdcpConfig->t_Reordering, drb_asn1->pdcp_config.t_Reordering);

  drb->headerCompression.present = drb_asn1->pdcp_config.headerCompression.present;
  drb->headerCompression.choice.notUsed = drb_asn1->pdcp_config.headerCompression.NotUsed;

  if (!drb_asn1->pdcp_config.integrityProtection) {
    asn1cCallocOne(drb->integrityProtection, drb_asn1->pdcp_config.integrityProtection);
  }
  if (!drb_asn1->pdcp_config.ext1.cipheringDisabled) {
    asn1cCalloc(pdcpConfig->ext1, ext1);
    asn1cCallocOne(ext1->cipheringDisabled, drb_asn1->pdcp_config.ext1.cipheringDisabled);
  }

  return DRB_config;
}

uint8_t get_next_available_drb_id(gNB_RRC_UE_t *ue)
{
  uint8_t drb_id = seq_arr_size(ue->drbs) + 1;
  if (drb_id < MAX_DRBS_PER_UE)
    return drb_id;
  /* From this point, we need to handle the case that all DRBs are already used by the UE. */
  LOG_E(RRC, "Error - All the DRBs are used - Handle this\n");
  return 0;
}
