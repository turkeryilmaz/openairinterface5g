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

#include "cucp_cuup_handler.h"
#include <netinet/in.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "NR_DRB-ToAddMod.h"
#include "NR_DRB-ToAddModList.h"
#include "NR_PDCP-Config.h"
#include "NR_QFI.h"
#include "NR_SDAP-Config.h"
#include "PHY/defs_common.h"
#include "RRC/NR/nr_rrc_common.h"
#include "SDAP/nr_sdap/nr_sdap_entity.h"
#include "asn_internal.h"
#include "assertions.h"
#include "common/utils/T/T.h"
#include "common/utils/oai_asn1.h"
#include "constr_TYPE.h"
#include "cuup_cucp_if.h"
#include "gtpv1_u_messages_types.h"
#include "nr_pdcp/nr_pdcp_asn1_utils.h"
#include "nr_pdcp/nr_pdcp_entity.h"
#include "nr_pdcp_oai_api.h"
#include "openair2/COMMON/e1ap_messages_types.h"
#include "openair2/E1AP/e1ap_common.h"
#include "openair2/E1AP/e1ap_context.h"
#include "openair2/F1AP/f1ap_common.h"
#include "openair2/F1AP/f1ap_ids.h"
#include "openair2/SDAP/nr_sdap/nr_sdap.h"
#include "openair3/ocp-gtpu/gtp_itf.h"

static instance_t get_n3_gtp_instance(void)
{
  const e1ap_upcp_inst_t *inst = getCxtE1(0);
  AssertFatal(inst, "need to have E1 instance\n");
  return inst->gtpInstN3;
}

static instance_t get_f1_gtp_instance(void)
{
  const f1ap_cudu_inst_t *inst = getCxt(0);
  if (!inst)
    return -1; // means no F1
  return inst->gtpInst;
}

static void deliver_pdcp_sdu(void *deliver_sdu_data,
                             nr_pdcp_entity_t *entity,
                             uint8_t *buf,
                             int size,
                             const nr_pdcp_integrity_data_t *msg_integrity)
{
  e1ap_cuup_context_t *e1_context = deliver_sdu_data;

  if (e1_context->sdap == NULL)
    return;

  e1_context->sdap->recv_pdu(e1_context->sdap, buf, size, entity->rb_id, entity->has_sdap_rx);
}

static void deliver_pdcp_pdu(void *deliver_pdu_data,
                             ue_id_t ue_id,
                             int rb_id,
                             uint8_t *buf,
                             int size,
                             int sdu_id)
{
  abort();
}

static void deliver_sdap_pdu(void *deliver_pdu_data, int rb_id, uint8_t *buf, int size)
{
  e1ap_cuup_context_t *e1_context = deliver_pdu_data;
  if (e1_context->drbs[rb_id - 1].pdcp == NULL) {
    LOG_E(SDAP, "no PDCP context found for DRB %d\n", rb_id);
    return;
  }
  int max_size = nr_max_pdcp_pdu_size(size);
  uint8_t pdu_buf[max_size];
  int pdu_size = e1_context->drbs[rb_id - 1].pdcp->process_sdu(e1_context->drbs[rb_id - 1].pdcp,
                                                               buf,
                                                               size,
                                                               0, /* sdu_id, not used */
                                                               pdu_buf,
                                                               max_size);
  if (pdu_size == -1) {
    LOG_E(PDCP, "failure to process SDAP PDU\n");
    return;
  }

  gtpv1uSendDirect(get_f1_gtp_instance(), e1_context->cu_up_ue_id, rb_id, pdu_buf, pdu_size, false, false);
}

static bool gtp_f1_ul_callback(void *gtp_callback_data,
                               ue_id_t ue_id,
                               int pdu_session_id,
                               int rb_id,
                               int qfi,
                               int rqi,
                               uint8_t *buf,
                               int size)
{
  DevAssert(rb_id >= 1 && rb_id <= MAX_DRBS_PER_PDU_SESSION);

  e1ap_cuup_context_t *e1_context = gtp_callback_data;

  if (e1_context->drbs[rb_id - 1].pdcp == NULL) {
    LOG_W(E1AP, "no DRB %d found for ue %ld, ignore input data of size %d\n", rb_id, ue_id, size);
    return false;
  }

  e1_context->drbs[rb_id - 1].pdcp->recv_pdu(e1_context->drbs[rb_id - 1].pdcp, buf, size);

  return true;
}

static bool gtp_n3_dl_callback(void *gtp_callback_data,
                               ue_id_t ue_id,
                               int pdu_session_id,
                               int rb_id,
                               int qfi,
                               int rqi,
                               uint8_t *buf,
                               int size)
{
  DevAssert(rb_id >= 1 && rb_id <= MAX_DRBS_PER_PDU_SESSION);

  e1ap_cuup_context_t *e1_context = gtp_callback_data;

  if (e1_context->sdap == NULL) {
    LOG_W(E1AP, "no SDAP for pdu session %d found for ue %ld, ignore input data of size %d\n", pdu_session_id, ue_id, size);
    return false;
  }

  e1_context->sdap->recv_sdu(e1_context->sdap, buf, size, rb_id, qfi);

  return true;
}

void e1_bearer_context_setup(const e1ap_bearer_setup_req_t *req)
{
  bool need_ue_id_mgmt = e1_used();

  /* mirror the CU-CP UE ID for CU-UP */
  uint32_t cu_up_ue_id = req->gNB_cu_cp_ue_id;
  f1_ue_data_t ued = {.secondary_ue = req->gNB_cu_cp_ue_id};
  if (need_ue_id_mgmt) {
    cu_add_f1_ue_data(cu_up_ue_id, &ued);
    LOG_I(E1AP, "adding UE with CU-CP UE ID %d and CU-UP UE ID %d\n", req->gNB_cu_cp_ue_id, cu_up_ue_id);
  }

  instance_t n3inst = get_n3_gtp_instance();
  instance_t f1inst = get_f1_gtp_instance();

  if (f1inst < 0) abort();

  e1ap_bearer_setup_resp_t resp = {
    .gNB_cu_cp_ue_id = req->gNB_cu_cp_ue_id,
    .gNB_cu_up_ue_id = cu_up_ue_id,
  };
  resp.numPDUSessions = req->numPDUSessions;

  /* to check that all DRB IDs are unique across all PDU sessions */
  bool drb_in_use[32] = { false };

  for (int i = 0; i < resp.numPDUSessions; ++i) {
    pdu_session_setup_t *resp_pdu = resp.pduSession + i;
    const pdu_session_to_setup_t *req_pdu = req->pduSession + i;
    resp_pdu->id = req_pdu->sessionId;

    AssertFatal(req_pdu->numDRB2Modify == 0, "DRB modification not implemented\n");
    AssertFatal(req_pdu->numDRB2Setup == 1, "can only handle one DRB per PDU session\n");
    resp_pdu->numDRBSetup = req_pdu->numDRB2Setup;
    const DRB_nGRAN_to_setup_t *req_drb = &req_pdu->DRBnGRanList[0];
    AssertFatal(req_drb->numQosFlow2Setup == 1, "can only handle one QoS Flow per DRB\n");
    DRB_nGRAN_setup_t *resp_drb = &resp_pdu->DRBnGRanList[0];
    resp_drb->id = req_drb->id;
    resp_drb->numQosFlowSetup = req_drb->numQosFlow2Setup;
    for (int k = 0; k < resp_drb->numQosFlowSetup; k++) {
      const qos_flow_to_setup_t *qosflow2Setup = &req_drb->qosFlows[k];
      qos_flow_setup_t *qosflowSetup = &resp_drb->qosFlows[k];
      qosflowSetup->qfi = qosflow2Setup->qfi;
    }

    e1ap_cuup_context_t *e1_context = new_e1ap_cuup_context(cu_up_ue_id, req_pdu->sessionId);

    /* create SDAP entity */
    e1_context->sdap = new_nr_sdap_entity2_gnb(cu_up_ue_id,
                                               req_pdu->sessionId,
                                               deliver_sdap_pdu,
                                               e1_context);

    /* check that all QFIs are unique in the PDU session */
    /* we also need all DRB IDs to be unique across all sessions
     * (this restriction may be removed if needed, as far as the
     * specifications are understood correctly it should be possible
     * to have the same DRB ID used in two different PDU sessions, but
     * as of writing this comment, the GTP module needs unique DRB IDs)
     */
    bool check_qfi[SDAP_MAX_QFI] = { false };
    for (int i = 0; i < req_pdu->numDRB2Setup; i++) {
      const DRB_nGRAN_to_setup_t *drb = &req_pdu->DRBnGRanList[i];
      DevAssert(drb->id >= 1 && drb->id <= 32);
      DevAssert(drb_in_use[drb->id - 1] == false);
      drb_in_use[drb->id - 1] = true;
      for (int j = 0; j < drb->numQosFlow2Setup; j++) {
        DevAssert(drb->qosFlows[j].qfi >= 1 && drb->qosFlows[j].qfi <= SDAP_MAX_QFI);
        DevAssert(check_qfi[drb->qosFlows[j].qfi - 1] == false);
        check_qfi[drb->qosFlows[j].qfi - 1] = true;
      }
    }

    /* create PDCP bearers, SDAP QoS flows and GTP-U UL UP endpoints */
    nr_pdcp_entity_security_keys_and_algos_t security_parameters;
    if (req_pdu->securityIndication.confidentialityProtectionIndication == SECURITY_NOT_NEEDED) {
      security_parameters.ciphering_algorithm = 0;
      memset(security_parameters.ciphering_key, 0, NR_K_KEY_SIZE);
    } else {
      security_parameters.ciphering_algorithm = req->cipheringAlgorithm;
      memcpy(security_parameters.ciphering_key, req->encryptionKey, NR_K_KEY_SIZE);
    }
    if (req_pdu->securityIndication.integrityProtectionIndication == SECURITY_NOT_NEEDED) {
      security_parameters.integrity_algorithm = 0;
      memset(security_parameters.integrity_key, 0, NR_K_KEY_SIZE);
    } else {
      security_parameters.integrity_algorithm = req->integrityProtectionAlgorithm;
      memcpy(security_parameters.integrity_key, req->integrityProtectionKey, NR_K_KEY_SIZE);
    }
    for (int i = 0; i < req_pdu->numDRB2Setup; i++) {
      const DRB_nGRAN_to_setup_t *drb = &req_pdu->DRBnGRanList[i];
      int rb_id = drb->id;
      /* only accept IDs in [1..32], to be refined if needed */
      DevAssert(rb_id >= 1 && rb_id <= MAX_DRBS_PER_PDU_SESSION);
      /* for gNB: SDAP rx is UL and SDAP tx is DL */
      /* 38.331 forces the use of UL header for the default bearer, see the definition of 'SDAP-Config' */
      /* 37.324 5.2.1 Note 2 says the same */
      AssertFatal(!drb->sdap_config.defaultDRB || drb->sdap_config.sDAP_Header_UL == NR_SDAP_Config__sdap_HeaderUL_present,
                  "default DRB must have Header UL present\n");
      bool has_sdap_rx_header = drb->sdap_config.sDAP_Header_UL == NR_SDAP_Config__sdap_HeaderUL_present;
      bool has_sdap_tx_header = drb->sdap_config.sDAP_Header_DL == NR_SDAP_Config__sdap_HeaderDL_present;
      /* accept only one default DRB */
      DevAssert(!drb->sdap_config.defaultDRB || e1_context->sdap->default_drb == 0);
      if (drb->sdap_config.defaultDRB) {
        e1_context->sdap->default_drb = rb_id;
        e1_context->sdap->default_drb_has_sdap_rx = has_sdap_rx_header;
        e1_context->sdap->default_drb_has_sdap_tx = has_sdap_tx_header;
      }
      /* PDCP entity */
      /* limitation: SN size for DL and UL must be equal (to be removed if needed) */
      DevAssert(drb->pdcp_config.pDCP_SN_Size_UL == drb->pdcp_config.pDCP_SN_Size_DL);
      e1_context->drbs[rb_id - 1].pdcp = new_nr_pdcp_entity(NR_PDCP_DRB_AM,
                                                            true, /* is_gnb */
                                                            cu_up_ue_id,
                                                            rb_id,
                                                            req_pdu->sessionId,
                                                            has_sdap_rx_header,
                                                            has_sdap_tx_header,
                                                            deliver_pdcp_sdu,
                                                            e1_context,
                                                            deliver_pdcp_pdu,
                                                            e1_context,
                                                            decode_sn_size_ul(drb->pdcp_config.pDCP_SN_Size_UL),
                                                            decode_t_reordering(drb->pdcp_config.reorderingTimer),
                                                            decode_discard_timer(drb->pdcp_config.discardTimer),
                                                            &security_parameters);
      /* SDAP QoS flows */
      for (int j = 0; j < drb->numQosFlow2Setup; j++)
        e1_context->sdap->qfi2drb_map_update(e1_context->sdap,
                                             drb->qosFlows[j].qfi,
                                             rb_id,
                                             has_sdap_rx_header,
                                             has_sdap_tx_header);
      /* GTP-U UL UP endpoint for this bearer */
      uint8_t addr[64] = { 0 };
      int addr_len = 0;
      int port = 0;
      gtpu_get_instance_address_and_port(f1inst, addr, &addr_len, &port);
      AssertFatal(addr_len == 4, "only IPv4 supported for the moment\n");
      transport_layer_addr_t dummy_addr = { 0 };
      dummy_addr.length = 32;
      teid_t teid = newGtpuCreateTunnel(f1inst,
                                        cu_up_ue_id,
                                        rb_id,
                                        rb_id, //should be req_pdu->sessionId but does not work... todo: fix
                                        0xffff, /* outgoing teid - will be sent by DU */
                                        -1,  /* QoS flow is not used for this GTP-U tunnel */
                                        dummy_addr,
                                        port,
                                        NULL,
                                        NULL);
      gtpu_set_callback2(teid, gtp_f1_ul_callback, e1_context);
      AssertFatal(teid != GTPNOK, "Unable to create GTP Tunnel for F1-U\n");
      e1_context->drbs[rb_id - 1].gtpu_tunnel_id = teid;
      memcpy(&resp_drb->UpParamList[resp_drb->numUpParam].tlAddress, addr, 4);
      resp_drb->UpParamList[resp_drb->numUpParam].teId = teid;
      resp_drb->numUpParam++;
    }

    // GTP tunnel for N3/to core
    transport_layer_addr_t n3_addr = { 0 };
    n3_addr.length = 32; /* unit: bits */
    uint32_t n3_addr_ip = /*htonl*/(req_pdu->UP_TL_information.tlAddress);
    memcpy(n3_addr.buffer, &n3_addr_ip, 4);
    int addr_len = 0;
    int port = 0;
    uint8_t dummy_addr[64] = { 0 };
    /* get the port */
    gtpu_get_instance_address_and_port(n3inst, dummy_addr, &addr_len, &port);
    teid_t teid = newGtpuCreateTunnel(n3inst,
                                      cu_up_ue_id,
                                      req_drb->id,
                                      req_pdu->sessionId, //should be req_pdu->sessionId but does not work... todo: fix
                                      req_pdu->UP_TL_information.teId,
                                      1, /* QoS flow is not used for this GTP-U tunnel */
                                      n3_addr,
                                      port,
                                      NULL,
                                      NULL);
    gtpu_set_callback2(teid, gtp_n3_dl_callback, e1_context);
    AssertFatal(teid != GTPNOK, "Unable to create GTP Tunnel for F1-U\n");
    e1_context->n3_tunnel_id = teid;
    resp_pdu->teId = teid;
    memcpy(&resp_pdu->tlAddress, dummy_addr, 4);

    // We assume all DRBs to setup have been setup successfully, so we always
    // send successful outcome in response and no failed DRBs
    resp_pdu->numDRBFailed = 0;
  }

  get_e1_if()->bearer_setup_response(&resp);
}

/**
 * @brief Fill Bearer Context Modification Response and send to callback
 */
void e1_bearer_context_modif(const e1ap_bearer_mod_req_t *req)
{
  AssertFatal(req->numPDUSessionsMod > 0, "need at least one PDU session to modify\n");

  e1ap_bearer_modif_resp_t modif = {
      .gNB_cu_cp_ue_id = req->gNB_cu_cp_ue_id,
      .gNB_cu_up_ue_id = req->gNB_cu_up_ue_id,
      .numPDUSessionsMod = req->numPDUSessionsMod,
  };

  instance_t f1inst = get_f1_gtp_instance();

  /* PDU Session Resource To Modify List (see 9.3.3.11 of TS 38.463) */
  for (int i = 0; i < req->numPDUSessionsMod; i++) {
    DevAssert(req->pduSessionMod[i].sessionId > 0);
    e1ap_cuup_context_t *e1_context = get_e1ap_cuup_context(req->gNB_cu_up_ue_id, req->pduSessionMod[i].sessionId);
    AssertFatal(e1_context != NULL, "pdu session %ld not found for UE %d\n", req->pduSessionMod[i].sessionId, req->gNB_cu_up_ue_id);
    LOG_I(E1AP,
          "UE %d: updating PDU session ID %ld (%ld bearers)\n",
          req->gNB_cu_up_ue_id,
          req->pduSessionMod[i].sessionId,
          req->pduSessionMod[i].numDRB2Modify);
    modif.pduSessionMod[i].id = req->pduSessionMod[i].sessionId;
    modif.pduSessionMod[i].numDRBModified = req->pduSessionMod[i].numDRB2Modify;
    /* DRBs to modify */
    for (int j = 0; j < req->pduSessionMod[i].numDRB2Modify; j++) {
      const DRB_nGRAN_to_mod_t *to_modif = &req->pduSessionMod[i].DRBnGRanModList[j];
      DRB_nGRAN_modified_t *modified = &modif.pduSessionMod[i].DRBnGRanModList[j];
      modified->id = to_modif->id;
      DevAssert(to_modif->id >= 1 && to_modif->id <= MAX_DRBS_PER_PDU_SESSION);

      if (to_modif->pdcp_config.pDCP_Reestablishment) {
        nr_pdcp_entity_security_keys_and_algos_t security_parameters;
        security_parameters.ciphering_algorithm = req->cipheringAlgorithm;
        security_parameters.integrity_algorithm = req->integrityProtectionAlgorithm;
        memcpy(security_parameters.ciphering_key, req->encryptionKey, NR_K_KEY_SIZE);
        memcpy(security_parameters.integrity_key, req->integrityProtectionKey, NR_K_KEY_SIZE);
        nr_pdcp_entity_t *pdcp = e1_context->drbs[to_modif->id - 1].pdcp;
        AssertFatal(pdcp != NULL, "pdu session %ld for UE %d does not have DRB %ld to modify\n",
                    req->pduSessionMod[i].sessionId, req->gNB_cu_up_ue_id, to_modif->id);
        pdcp->reestablish_entity(pdcp, &security_parameters);
      }

      if (f1inst < 0) // no F1-U?
        continue; // nothing to do

      /* Loop through DL UP Transport Layer params list
       * and update GTP tunnel outgoing addr and TEID */
      for (int k = 0; k < to_modif->numDlUpParam; k++) {
        in_addr_t addr = to_modif->DlUpParamList[k].tlAddress;
        GtpuUpdateTunnelOutgoingAddressAndTeid(f1inst, req->gNB_cu_cp_ue_id, to_modif->id, addr, to_modif->DlUpParamList[k].teId);
      }
    }
  }

  get_e1_if()->bearer_modif_response(&modif);
}

void e1_bearer_release_cmd(const e1ap_bearer_release_cmd_t *cmd)
{
  instance_t n3inst = get_n3_gtp_instance();
  instance_t f1inst = get_f1_gtp_instance();

  LOG_I(E1AP, "releasing UE %d\n", cmd->gNB_cu_up_ue_id);

  newGtpuDeleteAllTunnels(n3inst, cmd->gNB_cu_up_ue_id);
  if (f1inst >= 0)  // is there F1-U?
    newGtpuDeleteAllTunnels(f1inst, cmd->gNB_cu_up_ue_id);

  remove_e1ap_cuup_ue(cmd->gNB_cu_up_ue_id);

  e1ap_bearer_release_cplt_t cplt = {
    .gNB_cu_cp_ue_id = cmd->gNB_cu_cp_ue_id,
    .gNB_cu_up_ue_id = cmd->gNB_cu_up_ue_id,
  };

  get_e1_if()->bearer_release_complete(&cplt);
}
