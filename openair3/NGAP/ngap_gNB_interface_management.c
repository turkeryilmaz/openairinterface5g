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

#include "ngap_gNB_interface_management.h"
#include <stdint.h>
#include <stdio.h>
#include "assertions.h"
#include "conversions.h"
#include "utils.h"
#include "ngap_gNB_encoder.h"
#include "ngap_common.h"
#include "ngap_gNB_defs.h"
#include "ngap_gNB_utils.h"
#include "ngap_msg_includes.h"
#include "ngap_gNB_itti_messaging.h"
#include "oai_asn1.h"

/**************
 * NGAP SETUP *
 **************/

/** @brief NG Setup: encode NG Setup Request towards AMF (8.7.1. of 3GPP TS 38.413) */
int encode_ng_setup_request(ngap_gNB_instance_t *instance_p, ngap_gNB_amf_data_t *amf)
{
  NGAP_NGAP_PDU_t pdu;
  uint8_t *buffer = NULL;
  uint32_t len = 0;
  DevAssert(instance_p != NULL);
  DevAssert(amf != NULL);
  ngap_plmn_t *plmn = &instance_p->plmn[amf->broadcast_plmn_index[0]];

  NGAP_INFO("Generating NG setup request. PLMN=%03d%02d, TAC=%02x\n", plmn->mcc, plmn->mnc, instance_p->tac);

  amf->state = NGAP_GNB_STATE_WAITING;

  memset(&pdu, 0, sizeof(pdu));

  // Message Type (M)
  pdu.present = NGAP_NGAP_PDU_PR_initiatingMessage;
  pdu.choice.initiatingMessage = calloc_or_fail(1, sizeof(*pdu.choice.initiatingMessage));
  pdu.choice.initiatingMessage->procedureCode = NGAP_ProcedureCode_id_NGSetup;
  pdu.choice.initiatingMessage->criticality = NGAP_Criticality_reject;
  pdu.choice.initiatingMessage->value.present = NGAP_InitiatingMessage__value_PR_NGSetupRequest;
  NGAP_NGSetupRequest_t *out = &pdu.choice.initiatingMessage->value.choice.NGSetupRequest;

  // Global RAN Node ID (M)
  asn1cSequenceAdd(out->protocolIEs.list, NGAP_NGSetupRequestIEs_t, ie);
  ie->id = NGAP_ProtocolIE_ID_id_GlobalRANNodeID;
  ie->criticality = NGAP_Criticality_reject;
  ie->value.present = NGAP_NGSetupRequestIEs__value_PR_GlobalRANNodeID;
  NGAP_GlobalRANNodeID_t *id = &ie->value.choice.GlobalRANNodeID;
  id->present = NGAP_GlobalRANNodeID_PR_globalGNB_ID;
  id->choice.globalGNB_ID = calloc_or_fail(1, sizeof(*id->choice.globalGNB_ID));
  MCC_MNC_TO_PLMNID(plmn->mcc, plmn->mnc, plmn->mnc_digit_length, &(id->choice.globalGNB_ID->pLMNIdentity));
  NGAP_GNB_ID_t *gNB_ID = &id->choice.globalGNB_ID->gNB_ID;
  gNB_ID->present = NGAP_GNB_ID_PR_gNB_ID;
  MACRO_GNB_ID_TO_BIT_STRING(instance_p->gNB_id, &gNB_ID->choice.gNB_ID);
  NGAP_INFO("[gNB %u] Global RAN Node ID (gNB ID): %02x%02x%02x%02x\n",
            instance_p->gNB_id,
            gNB_ID->choice.gNB_ID.buf[0],
            gNB_ID->choice.gNB_ID.buf[1],
            gNB_ID->choice.gNB_ID.buf[2],
            gNB_ID->choice.gNB_ID.buf[3]);

  // RAN Node Name (O)
  if (instance_p->gNB_name) {
    asn1cSequenceAdd(out->protocolIEs.list, NGAP_NGSetupRequestIEs_t, ie1);
    ie1->id = NGAP_ProtocolIE_ID_id_RANNodeName;
    ie1->criticality = NGAP_Criticality_ignore;
    ie1->value.present = NGAP_NGSetupRequestIEs__value_PR_RANNodeName;
    OCTET_STRING_fromBuf(&ie1->value.choice.RANNodeName, instance_p->gNB_name, strlen(instance_p->gNB_name));
  }

  // Supported TA List (M)
  asn1cSequenceAdd(out->protocolIEs.list, NGAP_NGSetupRequestIEs_t, ie2);
  ie2->id = NGAP_ProtocolIE_ID_id_SupportedTAList;
  ie2->criticality = NGAP_Criticality_reject;
  ie2->value.present = NGAP_NGSetupRequestIEs__value_PR_SupportedTAList;
  asn1cSequenceAdd(ie2->value.choice.SupportedTAList.list, NGAP_SupportedTAItem_t, ta);
  INT24_TO_OCTET_STRING(instance_p->tac, &ta->tAC);
  for (int i = 0; i < amf->broadcast_plmn_num; ++i) {
    asn1cSequenceAdd(ta->broadcastPLMNList.list, NGAP_BroadcastPLMNItem_t, p);
    ngap_plmn_t *plmn_req = &instance_p->plmn[amf->broadcast_plmn_index[i]];
    MCC_MNC_TO_TBCD(plmn_req->mcc, plmn_req->mnc, plmn_req->mnc_digit_length, &p->pLMNIdentity);
    for (int si = 0; si < plmn_req->num_nssai; si++) {
      asn1cSequenceAdd(p->tAISliceSupportList.list, NGAP_SliceSupportItem_t, ssi);
      INT8_TO_OCTET_STRING(plmn_req->s_nssai[si].sst, &ssi->s_NSSAI.sST);
      const uint32_t sd = plmn_req->s_nssai[si].sd;
      if (sd != 0xffffff) {
        ssi->s_NSSAI.sD = calloc_or_fail(1, sizeof(*ssi->s_NSSAI.sD));
        ssi->s_NSSAI.sD->buf = calloc_or_fail(3, sizeof(*ssi->s_NSSAI.sD->buf));
        ssi->s_NSSAI.sD->size = 3;
        ssi->s_NSSAI.sD->buf[0] = (sd & 0xff0000) >> 16;
        ssi->s_NSSAI.sD->buf[1] = (sd & 0x00ff00) >> 8;
        ssi->s_NSSAI.sD->buf[2] = (sd & 0x0000ff);
      }
    }
  }

  // Default Paging DRX (M)
  asn1cSequenceAdd(out->protocolIEs.list, NGAP_NGSetupRequestIEs_t, ie3);
  ie3->id = NGAP_ProtocolIE_ID_id_DefaultPagingDRX;
  ie3->criticality = NGAP_Criticality_ignore;
  ie3->value.present = NGAP_NGSetupRequestIEs__value_PR_PagingDRX;
  ie3->value.choice.PagingDRX = instance_p->default_drx;

  if (ngap_gNB_encode_pdu(&pdu, &buffer, &len) < 0) {
    NGAP_ERROR("Failed to encode NG setup request\n");
    return -1;
  }

  /* Non UE-Associated signalling -> stream = 0 */
  ngap_gNB_itti_send_sctp_data_req(instance_p->instance, amf->assoc_id, buffer, len, 0);
  return 0;
}

/** @brief NG Setup Response decoder (9.2.6.2 3GPP TS 38.413) */
int decode_ng_setup_response(ng_setup_response_t *out, const NGAP_NGSetupResponse_t *container)
{
  NGAP_NGSetupResponseIEs_t *ie;

  NGAP_FIND_PROTOCOLIE_BY_ID(NGAP_NGSetupResponseIEs_t, ie, container, NGAP_ProtocolIE_ID_id_ServedGUAMIList, true);

  /* Served GUAMI List (M):
   * NR related guami is the first element in the list, i.e with an id of 0. */
  NGAP_DEBUG("servedGUAMIs.list.count %d\n", ie->value.choice.ServedGUAMIList.list.count);
  DevAssert(ie->value.choice.ServedGUAMIList.list.count > 0);
  DevAssert(ie->value.choice.ServedGUAMIList.list.count <= NGAP_maxnoofServedGUAMIs);

  out->num_guami = ie->value.choice.ServedGUAMIList.list.count;
  for (int i = 0; i < out->num_guami; i++) {
    struct served_guami_s *new_guami_p = &out->guami[i];
    NGAP_ServedGUAMIItem_t *guami_item_p = ie->value.choice.ServedGUAMIList.list.array[i];

    STAILQ_INIT(&new_guami_p->served_plmns);
    STAILQ_INIT(&new_guami_p->served_region_ids);
    STAILQ_INIT(&new_guami_p->amf_set_ids);
    STAILQ_INIT(&new_guami_p->amf_pointers);

    NGAP_PLMNIdentity_t *plmn_identity_p = &guami_item_p->gUAMI.pLMNIdentity;
    struct plmn_identity_s *new_plmn_identity_p = calloc_or_fail(1, sizeof(struct plmn_identity_s));
    TBCD_TO_MCC_MNC(plmn_identity_p, new_plmn_identity_p->mcc, new_plmn_identity_p->mnc, new_plmn_identity_p->mnc_digit_length);
    STAILQ_INSERT_TAIL(&new_guami_p->served_plmns, new_plmn_identity_p, next);
    new_guami_p->nb_served_plmns++;

    NGAP_AMFRegionID_t *amf_region_id_p = &guami_item_p->gUAMI.aMFRegionID;
    struct served_region_id_s *new_region_id_p = calloc_or_fail(1, sizeof(struct served_region_id_s));
    OCTET_STRING_TO_INT8(amf_region_id_p, new_region_id_p->amf_region_id);
    STAILQ_INSERT_TAIL(&new_guami_p->served_region_ids, new_region_id_p, next);
    new_guami_p->nb_region_id++;

    NGAP_AMFSetID_t *amf_set_id_p = &guami_item_p->gUAMI.aMFSetID;
    struct amf_set_id_s *new_amf_set_id_p = calloc_or_fail(1, sizeof(struct amf_set_id_s));
    OCTET_STRING_TO_INT16(amf_set_id_p, new_amf_set_id_p->amf_set_id);
    STAILQ_INSERT_TAIL(&new_guami_p->amf_set_ids, new_amf_set_id_p, next);
    new_guami_p->nb_amf_set_id++;

    NGAP_AMFPointer_t *amf_pointer_p = &guami_item_p->gUAMI.aMFPointer;
    struct amf_pointer_s *new_amf_pointer_p = calloc_or_fail(1, sizeof(struct amf_pointer_s));
    OCTET_STRING_TO_INT8(amf_pointer_p, new_amf_pointer_p->amf_pointer);
    STAILQ_INSERT_TAIL(&new_guami_p->amf_pointers, new_amf_pointer_p, next);
    new_guami_p->nb_amf_pointer++;
  }

  // Relative AMF Capacity (M)
  NGAP_FIND_PROTOCOLIE_BY_ID(NGAP_NGSetupResponseIEs_t, ie, container, NGAP_ProtocolIE_ID_id_RelativeAMFCapacity, true);

  out->relative_amf_capacity = ie->value.choice.RelativeAMFCapacity;

  // AMF Name (M)
  NGAP_FIND_PROTOCOLIE_BY_ID(NGAP_NGSetupResponseIEs_t, ie, container, NGAP_ProtocolIE_ID_id_AMFName, true);

  if (ie) {
    out->amf_name = malloc_or_fail(ie->value.choice.AMFName.size + 1);
    memcpy(out->amf_name, ie->value.choice.AMFName.buf, ie->value.choice.AMFName.size);
    out->amf_name[ie->value.choice.AMFName.size] = '\0';
  }

  /// PLMN Support List (M)
  NGAP_FIND_PROTOCOLIE_BY_ID(NGAP_NGSetupResponseIEs_t, ie, container, NGAP_ProtocolIE_ID_id_PLMNSupportList, true);

  NGAP_DEBUG("PLMNSupportList.list.count %d\n", ie->value.choice.PLMNSupportList.list.count);
  DevAssert(ie->value.choice.PLMNSupportList.list.count > 0);
  DevAssert(ie->value.choice.PLMNSupportList.list.count <= NGAP_maxnoofPLMNs);

  out->num_plmn = ie->value.choice.PLMNSupportList.list.count;
  for (int i = 0; i < out->num_plmn; i++) {
    NGAP_PLMNSupportItem_t *plmn_support_item_p = ie->value.choice.PLMNSupportList.list.array[i];

    struct plmn_support_s *new_plmn_support_p = &out->plmn[i];

    TBCD_TO_MCC_MNC(&plmn_support_item_p->pLMNIdentity,
                    new_plmn_support_p->plmn_identity.mcc,
                    new_plmn_support_p->plmn_identity.mnc,
                    new_plmn_support_p->plmn_identity.mnc_digit_length);

    NGAP_DEBUG("PLMNSupportList.list.count %d\n", plmn_support_item_p->sliceSupportList.list.count);
    DevAssert(plmn_support_item_p->sliceSupportList.list.count > 0);
    DevAssert(plmn_support_item_p->sliceSupportList.list.count <= NGAP_maxnoofSliceItems);

    STAILQ_INIT(&new_plmn_support_p->slice_supports);
    for (int j = 0; j < plmn_support_item_p->sliceSupportList.list.count; j++) {
      NGAP_SliceSupportItem_t *slice_support_item_p = plmn_support_item_p->sliceSupportList.list.array[j];

      struct slice_support_s *new_slice_support_p = calloc_or_fail(1, sizeof(*new_slice_support_p));

      OCTET_STRING_TO_INT8(&slice_support_item_p->s_NSSAI.sST, new_slice_support_p->sST);

      if (slice_support_item_p->s_NSSAI.sD != NULL) {
        new_slice_support_p->sD_flag = 1;
        new_slice_support_p->sD[0] = slice_support_item_p->s_NSSAI.sD->buf[0];
        new_slice_support_p->sD[1] = slice_support_item_p->s_NSSAI.sD->buf[1];
        new_slice_support_p->sD[2] = slice_support_item_p->s_NSSAI.sD->buf[2];
      }
      STAILQ_INSERT_TAIL(&new_plmn_support_p->slice_supports, new_slice_support_p, next);
    }
  }
  return 0;
}

/*****************
 * NGAP OVERLOAD *
 *****************/

int ngap_gNB_handle_overload_stop(sctp_assoc_t assoc_id, uint32_t stream, NGAP_NGAP_PDU_t *pdu)
{
  /* We received Overload stop message, meaning that the AMF is no more
   * overloaded. This is an empty message, with only message header and no
   * Information Element.
   */
  DevAssert(pdu != NULL);

  ngap_gNB_amf_data_t *amf_desc_p;

  /* Non UE-associated signalling -> stream 0 */
  DevCheck(stream == 0, stream, 0, 0);

  if ((amf_desc_p = ngap_gNB_get_AMF(NULL, assoc_id, 0)) == NULL) {
    /* No AMF context associated */
    return -1;
  }

  amf_desc_p->state = NGAP_GNB_STATE_CONNECTED;
  amf_desc_p->overload_state = NGAP_NO_OVERLOAD;
  return 0;
}
