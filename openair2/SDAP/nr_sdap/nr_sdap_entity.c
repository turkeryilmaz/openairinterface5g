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

#include "nr_sdap_entity.h"
#include <openair2/LAYER2/nr_pdcp/nr_pdcp_oai_api.h>
#include <openair3/ocp-gtpu/gtp_itf.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "T.h"
#include "assertions.h"
#include "common/utils/T/T.h"
#include "gtpv1_u_messages_types.h"
#include "intertask_interface.h"
#include "rlc.h"
#include "tun_if.h"
#include "nr_sdap.h"

typedef struct {
  nr_sdap_entity_t *sdap_entity_llist;
} nr_sdap_entity_info;

static nr_sdap_entity_info sdap_info;

instance_t *N3GTPUInst = NULL;

/**
 * @brief indicates whether it is a receiving SDAP entity
 *        i.e. for UE, header for DL data is present
 *             for gNB, header for UL data is present
 */
bool is_sdap_rx(bool is_gnb, NR_SDAP_Config_t *sdap_config)
{
  if (is_gnb) {
    return sdap_config->sdap_HeaderUL == NR_SDAP_Config__sdap_HeaderUL_present;
  } else {
    return sdap_config->sdap_HeaderDL == NR_SDAP_Config__sdap_HeaderDL_present;
  }
}

/**
 * @brief indicates whether it is a transmitting SDAP entity
 *        i.e. for UE, header for UL data is present
 *             for gNB, header for DL data is present
 */
bool is_sdap_tx(bool is_gnb, NR_SDAP_Config_t *sdap_config)
{
  if (is_gnb) {
    return sdap_config->sdap_HeaderDL == NR_SDAP_Config__sdap_HeaderDL_present;
  } else {
    return sdap_config->sdap_HeaderUL == NR_SDAP_Config__sdap_HeaderUL_present;
  }
}

void nr_pdcp_submit_sdap_ctrl_pdu(ue_id_t ue_id, rb_id_t sdap_ctrl_pdu_drb, nr_sdap_ul_hdr_t ctrl_pdu)
{

  protocol_ctxt_t ctxt = { .rntiMaybeUEid = ue_id };
  nr_pdcp_data_req_drb(&ctxt,
                       SRB_FLAG_NO,
                       sdap_ctrl_pdu_drb,
                       RLC_MUI_UNDEFINED,
                       SDU_CONFIRM_NO,
                       SDAP_HDR_LENGTH,
                       (unsigned char *)&ctrl_pdu,
                       PDCP_TRANSMISSION_MODE_UNKNOWN,
                       NULL,
                       NULL);
  LOG_D(SDAP, "Control PDU - Submitting Control PDU to DRB ID:  %ld\n", sdap_ctrl_pdu_drb);
  LOG_D(SDAP, "QFI: %u\n R: %u\n D/C: %u\n", ctrl_pdu.QFI, ctrl_pdu.R, ctrl_pdu.DC);
  return;
}

static bool nr_sdap_tx_entity(nr_sdap_entity_t *entity,
                              protocol_ctxt_t *ctxt_p,
                              const srb_flag_t srb_flag,
                              const rb_id_t rb_id,
                              const mui_t mui,
                              const confirm_t confirm,
                              const sdu_size_t sdu_buffer_size,
                              unsigned char *const sdu_buffer,
                              const pdcp_transmission_mode_t pt_mode,
                              const uint32_t *sourceL2Id,
                              const uint32_t *destinationL2Id,
                              const uint8_t qfi,
                              const bool rqi) {
  /* The offset of the SDAP header, it might be 0 if has_sdap_tx is not true in the pdcp entity. */
  int offset=0;
  bool ret = false;
  /*Hardcode DRB ID given from upper layer (ue/gnb_tun_read_thread rb_id), it will change if we have SDAP*/
  rb_id_t sdap_drb_id = rb_id;
  int pdcp_ent_has_sdap = 0;

  if(sdu_buffer == NULL) {
    LOG_E(SDAP, "%s:%d:%s: NULL sdu_buffer \n", __FILE__, __LINE__, __FUNCTION__);
    exit(1);
  }

  uint8_t sdap_buf[SDAP_MAX_PDU];
  rb_id_t pdcp_entity = entity->qfi2drb_map(entity, qfi);

  if(pdcp_entity){
    sdap_drb_id = pdcp_entity;
    pdcp_ent_has_sdap = entity->qfi2drb_table[qfi].has_sdap_tx;
    LOG_D(SDAP, "TX - QFI: %u is mapped to DRB ID: %ld\n", qfi, entity->qfi2drb_table[qfi].drb_id);
  }

  if(!pdcp_ent_has_sdap){
    LOG_D(SDAP, "TX - DRB ID: %ld does not have SDAP\n", entity->qfi2drb_table[qfi].drb_id);
    ret = nr_pdcp_data_req_drb(ctxt_p,
                               srb_flag,
                               sdap_drb_id,
                               mui,
                               confirm,
                               sdu_buffer_size,
                               sdu_buffer,
                               pt_mode,
                               sourceL2Id,
                               destinationL2Id);

    if(!ret)
      LOG_E(SDAP, "%s:%d:%s: PDCP refused PDU\n", __FILE__, __LINE__, __FUNCTION__);

    return ret;
  }

  if(sdu_buffer_size == 0 || sdu_buffer_size > 8999) {
    LOG_E(SDAP, "%s:%d:%s: NULL or 0 or exceeded sdu_buffer_size (over max PDCP SDU)\n", __FILE__, __LINE__, __FUNCTION__);
    return 0;
  }

  if(ctxt_p->enb_flag) { // gNB
    offset = SDAP_HDR_LENGTH;
    /*
     * TS 37.324 4.4 Functions
     * marking QoS flow ID in DL packets.
     *
     * Construct the DL SDAP data PDU.
     */
    nr_sdap_dl_hdr_t sdap_hdr;
    sdap_hdr.QFI = qfi;
    sdap_hdr.RQI = rqi;
    sdap_hdr.RDI = 0; // SDAP Hardcoded Value
    /* Add the SDAP DL Header to the buffer */
    memcpy(&sdap_buf[0], &sdap_hdr, SDAP_HDR_LENGTH);
    memcpy(&sdap_buf[SDAP_HDR_LENGTH], sdu_buffer, sdu_buffer_size);
    LOG_D(SDAP, "TX Entity QFI: %u \n", sdap_hdr.QFI);
    LOG_D(SDAP, "TX Entity RQI: %u \n", sdap_hdr.RQI);
    LOG_D(SDAP, "TX Entity RDI: %u \n", sdap_hdr.RDI);
  } else { // nrUE
    offset = SDAP_HDR_LENGTH;
    /*
     * TS 37.324 4.4 Functions
     * marking QoS flow ID in UL packets.
     *
     * 5.2.1 Uplink
     * construct the UL SDAP data PDU as specified in the subclause 6.2.2.3.
     */
    nr_sdap_ul_hdr_t sdap_hdr;
    sdap_hdr.QFI = qfi;
    sdap_hdr.R = 0;
    sdap_hdr.DC = rqi;
    /* Add the SDAP UL Header to the buffer */
    memcpy(&sdap_buf[0], &sdap_hdr, SDAP_HDR_LENGTH);
    memcpy(&sdap_buf[SDAP_HDR_LENGTH], sdu_buffer, sdu_buffer_size);
    LOG_D(SDAP, "TX Entity QFI: %u \n", sdap_hdr.QFI);
    LOG_D(SDAP, "TX Entity R:   %u \n", sdap_hdr.R);
    LOG_D(SDAP, "TX Entity DC:  %u \n", sdap_hdr.DC);
  }

  /*
   * TS 37.324 5.2 Data transfer
   * 5.2.1 Uplink UE side
   * submit the constructed UL SDAP data PDU to the lower layers
   *
   * Downlink gNB side
   */
  ret = nr_pdcp_data_req_drb(ctxt_p,
                             srb_flag,
                             sdap_drb_id,
                             mui,
                             confirm,
                             sdu_buffer_size + offset,
                             sdap_buf,
                             pt_mode,
                             sourceL2Id,
                             destinationL2Id);

  if(!ret)
    LOG_E(SDAP, "%s:%d:%s: PDCP refused PDU\n", __FILE__, __LINE__, __FUNCTION__);

  return ret;
}

static void nr_sdap_rx_entity(nr_sdap_entity_t *entity,
                              rb_id_t pdcp_entity,
                              int is_gnb,
                              bool has_sdap_rx,
                              int pdusession_id,
                              ue_id_t ue_id,
                              char *buf,
                              int size)
{
  /* The offset of the SDAP header, it might be 0 if has_sdap_rx is not true in the pdcp entity. */
  int offset=0;

  if (is_gnb) { // gNB
    if (has_sdap_rx) { // Handling the SDAP Header
      offset = SDAP_HDR_LENGTH;
      nr_sdap_ul_hdr_t *sdap_hdr = (nr_sdap_ul_hdr_t *)buf;
      LOG_D(SDAP, "RX Entity Received QFI:    %u\n", sdap_hdr->QFI);
      LOG_D(SDAP, "RX Entity Received R bit:  %u\n", sdap_hdr->R);
      LOG_D(SDAP, "RX Entity Received DC bit: %u\n", sdap_hdr->DC);

      switch (sdap_hdr->DC) {
        case SDAP_HDR_UL_DATA_PDU:
          LOG_D(SDAP, "RX Entity Received SDAP Data PDU\n");
          break;

        case SDAP_HDR_UL_CTRL_PDU:
          LOG_D(SDAP, "RX Entity Received SDAP Control PDU\n");
          break;
      }
    }

    uint8_t *gtp_buf = (uint8_t *)(buf + offset);
    size_t gtp_len = size - offset;

    // Pushing SDAP SDU to GTP-U Layer
    LOG_D(SDAP, "sending message to gtp size %ld\n", gtp_len);
    // very very dirty hack gloabl var N3GTPUInst
    instance_t inst = *N3GTPUInst;
    gtpv1uSendDirect(inst, ue_id, pdusession_id, gtp_buf, gtp_len, false, false);
  } else { //nrUE
    /*
     * TS 37.324 5.2 Data transfer
     * 5.2.2 Downlink
     * if the DRB from which this SDAP data PDU is received is configured by RRC with the presence of SDAP header.
     */
    if (has_sdap_rx) { // Handling the SDAP Header
      offset = SDAP_HDR_LENGTH;
      /*
       * TS 37.324 5.2 Data transfer
       * 5.2.2 Downlink
       * retrieve the SDAP SDU from the DL SDAP data PDU as specified in the subclause 6.2.2.2.
       */
      nr_sdap_dl_hdr_t *sdap_hdr = (nr_sdap_dl_hdr_t *)buf;
      LOG_D(SDAP, "RX Entity Received QFI : %u\n", sdap_hdr->QFI);
      LOG_D(SDAP, "RX Entity Received RQI : %u\n", sdap_hdr->RQI);
      LOG_D(SDAP, "RX Entity Received RDI : %u\n", sdap_hdr->RDI);

      /*
       * TS 37.324 5.2 Data transfer
       * 5.2.2 Downlink
       * Perform reflective QoS flow to DRB mapping as specified in the subclause 5.3.2.
       */
      if(sdap_hdr->RDI == SDAP_REFLECTIVE_MAPPING) {
        LOG_D(SDAP, "RX - Performing Reflective Mapping\n");
        /*
         * TS 37.324 5.3 QoS flow to DRB Mapping 
         * 5.3.2 Reflective mapping
         * If there is no stored QoS flow to DRB mapping rule for the QoS flow and a default DRB is configured.
         */
        if(!entity->qfi2drb_table[sdap_hdr->QFI].drb_id && entity->default_drb){
          nr_sdap_ul_hdr_t sdap_ctrl_pdu = entity->sdap_construct_ctrl_pdu(sdap_hdr->QFI);
          int sdap_ctrl_pdu_drb = entity->sdap_map_ctrl_pdu(entity, SDAP_CTRL_PDU_MAP_DEF_DRB, sdap_hdr->QFI);
          entity->sdap_submit_ctrl_pdu(ue_id, sdap_ctrl_pdu_drb, sdap_ctrl_pdu);
        }

        /*
         * TS 37.324 5.3 QoS flow to DRB mapping 
         * 5.3.2 Reflective mapping
         * if the stored QoS flow to DRB mapping rule for the QoS flow 
         * is different from the QoS flow to DRB mapping of the DL SDAP data PDU
         * and
         * the DRB according to the stored QoS flow to DRB mapping rule is configured by RRC
         * with the presence of UL SDAP header
         */
        if (pdcp_entity != entity->qfi2drb_table[sdap_hdr->QFI].drb_id) {
          nr_sdap_ul_hdr_t sdap_ctrl_pdu = entity->sdap_construct_ctrl_pdu(sdap_hdr->QFI);
          int sdap_ctrl_pdu_drb = entity->sdap_map_ctrl_pdu(entity, SDAP_CTRL_PDU_MAP_RULE_DRB, sdap_hdr->QFI);
          entity->sdap_submit_ctrl_pdu(ue_id, sdap_ctrl_pdu_drb, sdap_ctrl_pdu);
        }

        /*
         * TS 37.324 5.3 QoS flow to DRB Mapping 
         * 5.3.2 Reflective mapping
         * store the QoS flow to DRB mapping of the DL SDAP data PDU as the QoS flow to DRB mapping rule for the UL. 
         */ 
        entity->qfi2drb_table[sdap_hdr->QFI].drb_id = pdcp_entity;
      }

      /*
       * TS 37.324 5.2 Data transfer
       * 5.2.2 Downlink
       * perform RQI handling as specified in the subclause 5.4
       */
      if(sdap_hdr->RQI == SDAP_RQI_HANDLING) {
        LOG_W(SDAP, "UE - TODD 5.4\n");
      }
    } /*  else - retrieve the SDAP SDU from the DL SDAP data PDU as specified in the subclause 6.2.2.1 */

    /*
     * TS 37.324 5.2 Data transfer
     * 5.2.2 Downlink
     * deliver the retrieved SDAP SDU to the upper layer.
     */
    int len = write(entity->pdusession_sock, &buf[offset], size - offset);
    LOG_D(SDAP, "RX Entity len : %d\n", len);
    LOG_D(SDAP, "RX Entity size : %d\n", size);
    LOG_D(SDAP, "RX Entity offset : %d\n", offset);

    if (len != size-offset)
      LOG_E(SDAP, "write failed to fd %d! errno = %s\n", entity->pdusession_sock, strerror(errno));
  }
}

/** @brief Update QFI to DRB mapping rules
 * @param qfi the QoS Flow index, used as unique index of the qfi2drb mapping table
 * @param drb the DRB ID to be mapped */
static void nr_sdap_qfi2drb_map_update(nr_sdap_entity_t *entity, const sdap_config_t *sdap)
{
  for (int i = 0; i < sdap->mappedQFIs2AddCount; i++) {
    uint8_t qfi = sdap->mappedQFIs2Add[i];
    LOG_D(SDAP, "Updating QFI to DRB mapping rules: %d mapped QFIs for DRB %d\n", sdap->mappedQFIs2AddCount, sdap->drb_id);
    if (qfi < SDAP_MAX_QFI && qfi > SDAP_MAP_RULE_EMPTY && sdap->drb_id > 0 && sdap->drb_id <= MAX_DRBS_PER_UE) {
      entity->qfi2drb_map_add(entity, qfi, sdap->drb_id, sdap->sdap_rx, sdap->sdap_tx);
    } else {
      LOG_E(SDAP, "Failed to update qfi2drb mapping: QFI=%d, DRB=%d\n", qfi, sdap->drb_id);
    }
  }
  for (int i = 0; i < sdap->mappedQFIs2ReleaseCount; i++) {
    uint8_t qfi = sdap->mappedQFIs2Release[i];
    LOG_D(SDAP, "Deelting QFI to DRB mapping rules: QFI=%d for DRB=%d\n", qfi, sdap->drb_id);
    entity->qfi2drb_map_delete(entity, qfi);
  }
}

static void nr_sdap_qfi2drb_map_add(nr_sdap_entity_t *entity,
                                    const uint8_t qfi,
                                    const uint8_t drb_id,
                                    const uint8_t role_rx,
                                    const uint8_t role_tx)
{
  qfi2drb_t *qfi2drb = &entity->qfi2drb_table[qfi];
  LOG_D(SDAP, "%s mapping: QFI %u -> DRB %d \n", qfi2drb->drb_id == SDAP_NO_MAPPING_RULE ? "Add" : "Update", qfi, drb_id);
  qfi2drb->drb_id = drb_id;
  qfi2drb->has_sdap_rx = role_rx;
  qfi2drb->has_sdap_tx = role_tx;
}

static void nr_sdap_qfi2drb_map_del(nr_sdap_entity_t *entity, const uint8_t qfi)
{
  qfi2drb_t *qfi2drb = &entity->qfi2drb_table[qfi];
  qfi2drb->drb_id = SDAP_NO_MAPPING_RULE;
  LOG_D(SDAP, "Deleted mapping for QFI=%d, DRB=%d\n", qfi, qfi2drb->drb_id);
}

/**
 * @brief   get the DRB ID mapped to the QFI, for both DL and UL
 * @return  DRB that is mapped to the QFI
 *          or the default DRB if no mapping rule exists
 *          or 0 if no mapping and no default DRB exists for that QFI
 * @ref     TS 37.324, 5.2.1 Uplink
 *          If there is no stored QoS flow to DRB mapping rule for the QoS flow as specified in the subclause 5.3,
 *          map the SDAP SDU to the default DRB else, map the SDAP SDU to the DRB according to the stored QoS flow to DRB mapping rule.
 */
static int nr_sdap_qfi2drb(nr_sdap_entity_t *entity, uint8_t qfi)
{
  /* Fetch DRB ID mapped to QFI */
  int drb_id = entity->qfi2drb_table[qfi].drb_id;
  if (drb_id) {
    /* QoS flow to DRB mapping rule exists, return corresponding DRB ID */
    LOG_D(SDAP, "Existing QoS flow to DRB mapping rule: QFI %u to DRB %d\n", qfi, drb_id);
    return drb_id;
  } else if (entity->default_drb) {
    /* QoS flow to DRB mapping rule does not exist, map SDAP SDU to default DRB, e.g. return default DRB of the SDAP entity */
    LOG_D(SDAP, "QoS flow to DRB mapping rule does not exists! mapping SDU to Default DRB: %ld\n", entity->default_drb);
    return entity->default_drb;
  } else {
    /* Note: UE undefined behaviour when neither a default DRB
       nor a stored QoS flow to DRB mapping rule exists */
    LOG_E(SDAP, "Mapping rule and default DRB do not exist for QFI:%u\n", qfi);
    return SDAP_MAP_RULE_EMPTY;
  }
  return drb_id;
}

nr_sdap_ul_hdr_t nr_sdap_construct_ctrl_pdu(uint8_t qfi){
  nr_sdap_ul_hdr_t sdap_end_marker_hdr;
  sdap_end_marker_hdr.QFI = qfi;
  sdap_end_marker_hdr.R = 0;
  sdap_end_marker_hdr.DC = SDAP_HDR_UL_CTRL_PDU;
  LOG_D(SDAP, "Constructed Control PDU with QFI:%u R:%u DC:%u \n", sdap_end_marker_hdr.QFI,
                                                                   sdap_end_marker_hdr.R,
                                                                   sdap_end_marker_hdr.DC);
  return sdap_end_marker_hdr;
}

int nr_sdap_map_ctrl_pdu(nr_sdap_entity_t *entity, int map_type, uint8_t dl_qfi)
{
  int drb_of_endmarker = 0;
  if(map_type == SDAP_CTRL_PDU_MAP_DEF_DRB){
    drb_of_endmarker = entity->default_drb;
    LOG_D(SDAP, "Mapping Control PDU QFI: %u to Default DRB: %d\n", dl_qfi, drb_of_endmarker);
  }
  if(map_type == SDAP_CTRL_PDU_MAP_RULE_DRB){
    drb_of_endmarker = entity->qfi2drb_map(entity, dl_qfi);
    LOG_D(SDAP, "Mapping Control PDU QFI: %u to DRB: %d\n", dl_qfi, drb_of_endmarker);
  }
  return drb_of_endmarker;
}

/**
 * @brief Submit the end-marker control PDU to PDCP according to TS 37.324, clause 5.3
 */
void nr_sdap_submit_ctrl_pdu(ue_id_t ue_id, rb_id_t sdap_ctrl_pdu_drb, nr_sdap_ul_hdr_t ctrl_pdu)
{
  if(sdap_ctrl_pdu_drb){
    nr_pdcp_submit_sdap_ctrl_pdu(ue_id, sdap_ctrl_pdu_drb, ctrl_pdu);
    LOG_D(SDAP, "Sent Control PDU to PDCP Layer.\n");
  }
}

/** @brief UL QoS flow to DRB mapping configuration for a SDAP entity has already been established
 *         according to TS 37.324, 5.3 QoS flow to DRB Mapping, 5.3.1 Configuration Procedures.
 *         The function handles both UL QoS Flows mapping rules to add and to remove. */
void nr_sdap_ue_qfi2drb_config(nr_sdap_entity_t *entity, const ue_id_t ue_id, const sdap_config_t sdap)
{
  // handle QFIs to DRB mapping rule to add
  for (int i = 0; i < sdap.mappedQFIs2AddCount; i++) {
    uint8_t qfi = sdap.mappedQFIs2Add[i];
    /* a default DRB exists and there is no stored QFI to DRB mapping rule for the QFI */
    if (entity->default_drb && entity->qfi2drb_table[qfi].drb_id == SDAP_NO_MAPPING_RULE) {
      // construct an end-marker control PDU (6.2.3 TS 37.324)
      nr_sdap_ul_hdr_t sdap_ctrl_pdu = entity->sdap_construct_ctrl_pdu(qfi);
      // map the end-marker control PDU to the default DRB
      int sdap_ctrl_pdu_drb = entity->sdap_map_ctrl_pdu(entity, SDAP_CTRL_PDU_MAP_DEF_DRB, qfi);
      // submit the end-marker control PDU to the lower layers
      entity->sdap_submit_ctrl_pdu(ue_id, sdap_ctrl_pdu_drb, sdap_ctrl_pdu);
    }
    /* the stored UL QFI to DRB mapping rule is different from the configured one and has UL SDAP header */
    if (entity->qfi2drb_table[qfi].drb_id != sdap.drb_id && entity->qfi2drb_table[qfi].has_sdap_tx) {
      // construct an end-marker control PDU (6.2.3 TS 37.324)
      nr_sdap_ul_hdr_t sdap_ctrl_pdu = entity->sdap_construct_ctrl_pdu(qfi);
      // map the end-marker control PDU to the DRB according to the stored QoS flow to DRB mapping rule
      int sdap_ctrl_pdu_drb = entity->sdap_map_ctrl_pdu(entity, SDAP_CTRL_PDU_MAP_RULE_DRB, qfi);
      // submit the end-marker control PDU to the lower layers
      entity->sdap_submit_ctrl_pdu(ue_id, sdap_ctrl_pdu_drb, sdap_ctrl_pdu);
    }
  }

  // handle QFIs to DRB mapping rule to release
  for (int i = 0; i < sdap.mappedQFIs2ReleaseCount; i++) {
    entity->qfi2drb_map_delete(entity, sdap.mappedQFIs2Release[i]);
  }
  // store QFI to DRB mapping rules
  LOG_D(SDAP, "Storing the configured QoS flow to DRB mapping rule\n");
  entity->qfi2drb_map_update(entity, &sdap);
}

/**
 * @brief   add a new SDAP entity according to 5.1.1. of 3GPP TS 37.324
 * @note    there is one SDAP entity per PDU session
 *
 * @param   is_gnb, indicates whether it is for gNB or UE
 * @param   has_sdap_rx, indicates whether it is a receiving SDAP entity
 * @param   has_sdap_tx, indicates whether it is a transmitting SDAP entity
 * @param   ue_id, UE ID
 * @param   pdusession_id, PDU session ID
 * @param   is_defaultDRB, indicates whether the entity has a default DRB
 * @param   mapped_qfi_2_add, list of QoS flows to add/update
 * @param   mappedQFIs2AddCount, number of QoS flows to add/update
 */
nr_sdap_entity_t *new_nr_sdap_entity(const int is_gnb, const ue_id_t ue_id, const sdap_config_t sdap)
{
  /* check whether the SDAP entity already exists and
     update QFI to DRB mapping rules in that case */
  nr_sdap_entity_t *sdap_entity = nr_sdap_get_entity(ue_id, sdap.pdusession_id);
  if (sdap_entity) {
    LOG_E(SDAP, "SDAP Entity for UE already exists with RNTI/UE ID: %lu and PDU SESSION ID: %d\n", ue_id, sdap.pdusession_id);
    if (!is_gnb) {
      nr_sdap_ue_qfi2drb_config(sdap_entity, ue_id, sdap);
    } else {
      // store QFI to DRB mapping rules
      sdap_entity->qfi2drb_map_update(sdap_entity, &sdap);
    }
    return sdap_entity;
  }

  sdap_entity = calloc_or_fail(1, sizeof(*sdap_entity));

  // SDAP entity ids
  sdap_entity->ue_id = ue_id;
  sdap_entity->pdusession_id = sdap.pdusession_id;
  sdap_entity->is_gnb = is_gnb;

  // rx/tx entities
  sdap_entity->tx_entity = nr_sdap_tx_entity;
  sdap_entity->rx_entity = nr_sdap_rx_entity;

  // control pdu function pointers
  sdap_entity->sdap_construct_ctrl_pdu = nr_sdap_construct_ctrl_pdu;
  sdap_entity->sdap_map_ctrl_pdu = nr_sdap_map_ctrl_pdu;
  sdap_entity->sdap_submit_ctrl_pdu = nr_sdap_submit_ctrl_pdu;

  // QFI to DRB mapping functions pointers
  sdap_entity->qfi2drb_map_update = nr_sdap_qfi2drb_map_update;
  sdap_entity->qfi2drb_map_add = nr_sdap_qfi2drb_map_add;
  sdap_entity->qfi2drb_map_delete = nr_sdap_qfi2drb_map_del;
  sdap_entity->qfi2drb_map = nr_sdap_qfi2drb;
  sdap_entity->pdusession_sock = -1;

  // set default DRB
  if (sdap.defaultDRB) {
    sdap_entity->default_drb = sdap.drb_id;
    LOG_I(SDAP, "Default DRB for the created SDAP entity: DRB %ld \n", sdap_entity->default_drb);
  }

  // store QFI to DRB mapping rules
  sdap_entity->qfi2drb_map_update(sdap_entity, &sdap);

  // update SDAP entity list pointers
  sdap_entity->next_entity = sdap_info.sdap_entity_llist;
  sdap_info.sdap_entity_llist = sdap_entity;

  if (IS_SOFTMODEM_NOS1 && is_gnb) {
    // In NOS1 mode, terminate SDAP for the first UE on the gNB. This allows injecting/receiving
    // PDCP SDUs to/from the TUN interface.
    start_sdap_tun_gnb_first_ue_default_pdu_session(ue_id);
  }
  return sdap_entity;
}

/**
 * @brief   Fetches the SDAP entity for the give PDU session ID.
 * @note    There is one SDAP entity per PDU session.
 * @return  The pointer to the SDAP entity if existing, NULL otherwise
 */
nr_sdap_entity_t *nr_sdap_get_entity(ue_id_t ue_id, int pdusession_id)
{
  nr_sdap_entity_t *sdap_entity;
  sdap_entity = sdap_info.sdap_entity_llist;

  if(sdap_entity == NULL)
    return NULL;

  while ((sdap_entity->ue_id != ue_id || sdap_entity->pdusession_id != pdusession_id) && sdap_entity->next_entity != NULL) {
    sdap_entity = sdap_entity->next_entity;
  }

  if (sdap_entity->ue_id == ue_id && sdap_entity->pdusession_id == pdusession_id)
    return sdap_entity;

  return NULL;
}

void nr_sdap_release_drb(ue_id_t ue_id, int drb_id, int pdusession_id)
{
  // remove all QoS flow to DRB mappings associated with the released DRB
  nr_sdap_entity_t *sdap = nr_sdap_get_entity(ue_id, pdusession_id);
  if (sdap) {
    for (int i = 0; i < SDAP_MAX_QFI; i++) {
      if (sdap->qfi2drb_table[i].drb_id == drb_id)
        sdap->qfi2drb_table[i].drb_id = SDAP_NO_MAPPING_RULE;
    }
  }
  else
    LOG_E(SDAP, "Couldn't find a SDAP entity associated with PDU session ID %d\n", pdusession_id);
}

bool nr_sdap_delete_entity(ue_id_t ue_id, int pdusession_id)
{
  nr_sdap_entity_t *entityPtr = sdap_info.sdap_entity_llist;
  nr_sdap_entity_t *entityPrev = NULL;
  bool ret = false;
  int upperBound = 0;

  if (entityPtr == NULL && (pdusession_id) * (pdusession_id - NGAP_MAX_PDU_SESSION) > 0) {
    LOG_E(SDAP, "SDAP entities not established or Invalid range of pdusession_id [0, 256].\n");
    return ret;
  }
  LOG_D(SDAP, "Deleting SDAP entity for UE %lx and PDU Session id %d\n", ue_id, entityPtr->pdusession_id);

  if (entityPtr->ue_id == ue_id && entityPtr->pdusession_id == pdusession_id) {
    sdap_info.sdap_entity_llist = sdap_info.sdap_entity_llist->next_entity;
    if (entityPtr->pdusession_sock != -1)
      remove_ip_if(entityPtr);
    free(entityPtr);
    LOG_D(SDAP, "Successfully deleted Entity.\n");
    ret = true;
  } else {
    while ((entityPtr->ue_id != ue_id || entityPtr->pdusession_id != pdusession_id) && entityPtr->next_entity != NULL
           && upperBound < SDAP_MAX_NUM_OF_ENTITIES) {
      entityPrev = entityPtr;
      entityPtr = entityPtr->next_entity;
      upperBound++;
    }

    if (entityPtr->ue_id == ue_id && entityPtr->pdusession_id == pdusession_id) {
      entityPrev->next_entity = entityPtr->next_entity;
      if (entityPtr->pdusession_sock != -1) {
        remove_ip_if(entityPtr);
      }
      free(entityPtr);
      LOG_D(SDAP, "Successfully deleted Entity.\n");
      ret = true;
    }
  }
  LOG_E(SDAP, "Entity does not exist or it was not found.\n");
  return ret;
}

bool nr_sdap_delete_ue_entities(ue_id_t ue_id)
{
  nr_sdap_entity_t *entityPtr = sdap_info.sdap_entity_llist;
  nr_sdap_entity_t *entityPrev = NULL;
  int upperBound = 0;
  bool ret = false;

  if (entityPtr == NULL && (ue_id) * (ue_id - SDAP_MAX_UE_ID) > 0) {
    LOG_W(SDAP, "SDAP entities not established or Invalid range of ue_id [0, 65536]\n");
    return ret;
  }

  /* Handle scenario where ue_id matches the head of the list */
  while (entityPtr != NULL && entityPtr->ue_id == ue_id && upperBound < MAX_DRBS_PER_UE) {
    sdap_info.sdap_entity_llist = entityPtr->next_entity;
    if (entityPtr->pdusession_sock != -1)
      remove_ip_if(entityPtr);
    free(entityPtr);
    entityPtr = sdap_info.sdap_entity_llist;
    ret = true;
  }

  while (entityPtr != NULL && upperBound < SDAP_MAX_NUM_OF_ENTITIES) {
    if (entityPtr->ue_id != ue_id) {
      entityPrev = entityPtr;
      entityPtr = entityPtr->next_entity;
    } else {
      entityPrev->next_entity = entityPtr->next_entity;
      if (entityPtr->pdusession_sock != -1)
        remove_ip_if(entityPtr);
      free(entityPtr);
      entityPtr = entityPrev->next_entity;
      LOG_I(SDAP, "Successfully deleted SDAP entity for UE %ld\n", ue_id);
      ret = true;
    }
  }
  return ret;
}

/** @brief This function gets the relevant SDAP config from the received SDAP-Config */
sdap_config_t get_sdap_Config(int is_gnb, ue_id_t UEid, NR_SDAP_Config_t *sdap_Config, int drb_id)
{
  sdap_config_t sdapConfig = {0};
  sdapConfig.drb_id = drb_id;
  sdapConfig.sdap_rx = is_sdap_rx(is_gnb, sdap_Config);
  sdapConfig.sdap_tx = is_sdap_tx(is_gnb, sdap_Config);
  sdapConfig.defaultDRB = sdap_Config->defaultDRB;
  // 3GPP TS 38.331 The network sets sdap-HeaderUL to present if the field defaultDRB is set to true
  if (sdapConfig.defaultDRB && (sdap_Config->sdap_HeaderUL != NR_SDAP_Config__sdap_HeaderUL_present))
    LOG_D(SDAP, "Received SDAP-Config with defaultDRB but sdap-HeaderUL is not present\n");
  if (sdap_Config->mappedQoS_FlowsToAdd) {
    sdapConfig.mappedQFIs2AddCount = sdap_Config->mappedQoS_FlowsToAdd->list.count;
    LOG_D(SDAP, "DRB %d: mapped QFIs = %d  \n", sdapConfig.drb_id, sdapConfig.mappedQFIs2AddCount);
    for (int i = 0; i < sdapConfig.mappedQFIs2AddCount; i++){
      sdapConfig.mappedQFIs2Add[i] = *sdap_Config->mappedQoS_FlowsToAdd->list.array[i];
      LOG_D(SDAP, "Captured mappedQoS_FlowsToAdd[%d] from RRC: %ld\n", i, sdapConfig.mappedQFIs2Add[i]);
    }
  }
  sdapConfig.pdusession_id = sdap_Config->pdu_Session;
  if (sdap_Config->mappedQoS_FlowsToRelease) {
    sdapConfig.mappedQFIs2ReleaseCount = sdap_Config->mappedQoS_FlowsToRelease->list.count;
    for (int i = 0; i < sdapConfig.mappedQFIs2ReleaseCount; i++) {
      sdapConfig.mappedQFIs2Release[i] = *sdap_Config->mappedQoS_FlowsToRelease->list.array[0];
    }
  }
  return sdapConfig;
}

/**
 * @brief SDAP Entity reconfiguration at UE according to TS 37.324
 *        and triggered by RRC reconfiguration events according to clause 5.3.5.6.5 of TS 38.331.
 *        This function performs:
 *        - QoS flow to DRB mapping according to clause 5.3.1 of TS 37.324
 */
void nr_reconfigure_sdap_entity(NR_SDAP_Config_t *sdap_config, ue_id_t ue_id, int pdusession_id, int drb_id)
{
  bool is_gnb = false;
  /* fetch SDAP entity */
  nr_sdap_entity_t *sdap_entity = nr_sdap_get_entity(ue_id, pdusession_id);
  AssertError(sdap_entity != NULL,
              return,
              "Could not find SDAP Entity for RNTI/UE ID: %lu and PDU SESSION ID: %d\n",
              ue_id,
              pdusession_id);
  /* QFI to DRB mapping */
  sdap_config_t sdap = get_sdap_Config(is_gnb, ue_id, sdap_config, drb_id);
  nr_sdap_ue_qfi2drb_config(sdap_entity, ue_id, sdap);
}

void set_qfi(uint8_t qfi, uint8_t pduid, ue_id_t ue_id)
{
  nr_sdap_entity_t *entity = nr_sdap_get_entity(ue_id, pduid);
  DevAssert(entity != NULL);
  entity->qfi = qfi;
  return;
}
