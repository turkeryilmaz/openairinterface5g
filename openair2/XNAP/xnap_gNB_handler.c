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

/*! \file xnap_gNB_handler.c
 * \brief xnap handler procedures for gNB
 * \author Sreeshma Shiv <sreeshmau@iisc.ac.in>
 * \date August 2023
 * \version 1.0
 */

#include <stdint.h>
#include "intertask_interface.h"
#include "xnap_common.h"
#include "xnap_gNB_defs.h"
#include "xnap_gNB_handler.h"
#include "xnap_ids.h"
#include "xnap_gNB_interface_management.h"
#include "assertions.h"
#include "conversions.h"

static int xnap_gNB_handle_handover_preparation(instance_t instance, sctp_assoc_t assoc_id, uint32_t stream, XNAP_XnAP_PDU_t *pdu);

/* Placement of callback functions according to XNAP_ProcedureCode.h */
static const xnap_message_decoded_callback xnap_messages_callback[][3] = {
    {xnap_gNB_handle_handover_preparation, 0, 0},  /* handoverPreparation */
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {xnap_gNB_handle_xn_setup_request, xnap_gNB_handle_xn_setup_response, xnap_gNB_handle_xn_setup_failure}, /* xnSetup */
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0}};

static const char *const xnap_direction_String[] = {
    "", /* Nothing */
    "Originating message", /* originating message */
    "Successfull outcome", /* successfull outcome */
    "UnSuccessfull outcome", /* successfull outcome */
};
const char *xnap_direction2String(int xnap_dir)
{
  return (xnap_direction_String[xnap_dir]);
}

/** Commented it for now ** 
int xnap_assoc_id(F1_t isXNAP, instance_t instanceP) {
  xnap_setup_req_t *xnap_inst=xnap_req(isXNAP, instanceP);
  return xnap_inst->assoc_id;
} **/


int xnap_gNB_handle_message(instance_t instance,
                            sctp_assoc_t assoc_id,
                            int32_t stream,
                            const uint8_t *const data,
                            const uint32_t data_length)
{
  XNAP_XnAP_PDU_t pdu;
  int ret = 0;

  DevAssert(data != NULL);

  memset(&pdu, 0, sizeof(pdu));

  printf("Data length received: %d\n", data_length);
  if (xnap_gNB_decode_pdu(&pdu, data, data_length) < 0) {
    LOG_E(XNAP, "Failed to decode PDU\n");
    return -1;
  }

  switch (pdu.present) {
    case XNAP_XnAP_PDU_PR_initiatingMessage:
      LOG_I(XNAP, "xnap_gNB_decode_initiating_message!\n");
      /* Checking procedure Code and direction of message */
      if (pdu.choice.initiatingMessage->procedureCode
          >= sizeof(xnap_messages_callback) / (3 * sizeof(xnap_message_decoded_callback))) {
        //|| (pdu.present > XNAP_XnAP_PDU_PR_unsuccessfulOutcome)) {
        LOG_E(XNAP, "[SCTP %d] Either procedureCode %ld exceed expected\n", assoc_id, pdu.choice.initiatingMessage->procedureCode);
        ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_XNAP_XnAP_PDU, &pdu);
        return -1;
      }

      /* No handler present */
      if (xnap_messages_callback[pdu.choice.initiatingMessage->procedureCode][pdu.present - 1] == NULL) {
        LOG_E(XNAP,
              "[SCTP %d] No handler for procedureCode %ld in %s\n",
              assoc_id,
              pdu.choice.initiatingMessage->procedureCode,
              xnap_direction2String(pdu.present - 1));
        ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_XNAP_XnAP_PDU, &pdu);
        return -1;
      }
      /* Calling the right handler */
      ret =
          (*xnap_messages_callback[pdu.choice.initiatingMessage->procedureCode][pdu.present - 1])(instance, assoc_id, stream, &pdu);
      break;

    case XNAP_XnAP_PDU_PR_successfulOutcome:
      LOG_I(XNAP, "xnap_gNB_decode_successfuloutcome_message!\n");
      /* Checking procedure Code and direction of message */
      if (pdu.choice.successfulOutcome->procedureCode
          >= sizeof(xnap_messages_callback) / (3 * sizeof(xnap_message_decoded_callback))) {
        LOG_E(XNAP, "[SCTP %d] Either procedureCode %ld exceed expected\n", assoc_id, pdu.choice.successfulOutcome->procedureCode);
        ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_XNAP_XnAP_PDU, &pdu);
        return -1;
      }

      /* No handler present.*/
      if (xnap_messages_callback[pdu.choice.successfulOutcome->procedureCode][pdu.present - 1] == NULL) {
        LOG_E(XNAP,
              "[SCTP %d] No handler for procedureCode %ld in %s\n",
              assoc_id,
              pdu.choice.successfulOutcome->procedureCode,
              xnap_direction2String(pdu.present - 1));
        ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_XNAP_XnAP_PDU, &pdu);
        return -1;
      }
      /* Calling the right handler */
      ret =
          (*xnap_messages_callback[pdu.choice.successfulOutcome->procedureCode][pdu.present - 1])(instance, assoc_id, stream, &pdu);
      break;

    case XNAP_XnAP_PDU_PR_unsuccessfulOutcome:
      LOG_I(XNAP, "xnap_gNB_decode_unsuccessfuloutcome_message!\n");
      /* Checking procedure Code and direction of message */
      if (pdu.choice.unsuccessfulOutcome->procedureCode
          >= sizeof(xnap_messages_callback) / (3 * sizeof(xnap_message_decoded_callback))) {
        LOG_E(XNAP,
              "[SCTP %d] Either procedureCode %ld exceed expected\n",
              assoc_id,
              pdu.choice.unsuccessfulOutcome->procedureCode);
        ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_XNAP_XnAP_PDU, &pdu);
        return -1;
      }

      /* No handler present */
      if (xnap_messages_callback[pdu.choice.unsuccessfulOutcome->procedureCode][pdu.present - 1] == NULL) {
        LOG_E(XNAP,
              "[SCTP %d] No handler for procedureCode %ld in %s\n",
              assoc_id,
              pdu.choice.unsuccessfulOutcome->procedureCode,
              xnap_direction2String(pdu.present - 1));
        ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_XNAP_XnAP_PDU, &pdu);
        return -1;
      }
      /* Calling the right handler */
      ret = (*xnap_messages_callback[pdu.choice.unsuccessfulOutcome->procedureCode][pdu.present - 1])(instance,
                                                                                                      assoc_id,
                                                                                                      stream,
                                                                                                      &pdu);
      break;

    default:
      LOG_E(XNAP, "[SCTP %d] Direction %d exceed expected\n", assoc_id, pdu.present);
      break;
  }

  ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_XNAP_XnAP_PDU, &pdu);
  return ret;
}

static int xnap_gNB_handle_handover_preparation(instance_t instance, sctp_assoc_t assoc_id, uint32_t stream, XNAP_XnAP_PDU_t *pdu)
{
   
  XNAP_HandoverRequest_t             *xnHandoverRequest;
  XNAP_HandoverRequest_IEs_t         *ie;

//  XNAP_PDUSessionResourcesToBeSetup_List_t *PDUSession_ToBeSetup_ItemIEs;
//  XNAP_PDUSessionResourcesToBeSetup_Item_t   *PDUSession_ToBeSetup_Item;
//  X2AP_E_RABs_ToBeSetup_ItemIEs_t    *e_RABS_ToBeSetup_ItemIEs;
//  X2AP_E_RABs_ToBeSetup_Item_t       *e_RABs_ToBeSetup_Item;

  xnap_gNB_instance_t                *instance_p;
  xnap_gNB_data_t                    *xnap_gNB_data;
  MessageDef                         *msg;
  int                                ue_id;

  DevAssert (pdu != NULL);
  xnHandoverRequest = &pdu->choice.initiatingMessage->value.choice.HandoverRequest;

  if (stream == 0) {
    LOG_E (XNAP, "Received new xn handover request on stream == 0\n");
    /* TODO: send a xn failure response */
    return 0;
  }

  LOG_E (XNAP,"Received a new XN handover request\n");

  xnap_gNB_data = xnap_get_gNB(NULL, assoc_id, 0);
  DevAssert(xnap_gNB_data != NULL);

  instance_p = xnap_gNB_get_instance(instance);
  DevAssert(instance_p != NULL);

  msg = itti_alloc_new_message(TASK_XNAP, 0, XNAP_HANDOVER_REQ);

  XNAP_FIND_PROTOCOLIE_BY_ID(XNAP_HandoverRequest_IEs_t, ie, xnHandoverRequest,
                             XNAP_ProtocolIE_ID_id_oldNG_RANnodeUEXnAPID, true);
  if (ie == NULL ) {
    LOG_E (XNAP, "%s %d: ie is a NULL pointer \n",__FILE__,__LINE__);
    itti_free(ITTI_MSG_ORIGIN_ID(msg), msg);
    return -1;
  }

    /* allocate a new X2AP UE ID */
  ue_id = xnap_allocate_new_id(&instance_p->id_manager);
  if (ue_id == -1) {
    LOG_E (XNAP, "could not allocate a new XNAP UE ID\n");
    /* TODO: cancel handover: send HO preparation failure to source gNB */
    exit(1);
  }
  /* rnti is unknown yet, must not be set to -1, 0 is fine */
  xnap_set_ids(&instance_p->id_manager, ue_id, 0, ie->value.choice.NG_RANnodeUEXnAPID, ue_id);
  xnap_id_set_state(&instance_p->id_manager, ue_id, XNID_STATE_TARGET);

  XNAP_HANDOVER_REQ(msg).ng_node_ue_xnap_id = ue_id;

  XNAP_FIND_PROTOCOLIE_BY_ID(XNAP_HandoverRequest_IEs_t, ie, xnHandoverRequest,
                             XNAP_ProtocolIE_ID_id_GUAMI, true);
  if (ie == NULL ) {
    LOG_E (XNAP, "%s %d: ie is a NULL pointer \n",__FILE__,__LINE__);
    itti_free(ITTI_MSG_ORIGIN_ID(msg), msg);
    return -1;
  }

  PLMNID_TO_MCC_MNC(&ie->value.choice.GUAMI.plmn_ID,
		    XNAP_HANDOVER_REQ(msg).guami.plmn_id.mcc,
		    XNAP_HANDOVER_REQ(msg).guami.plmn_id.mnc,
		    XNAP_HANDOVER_REQ(msg).guami.plmn_id.mnc_digit_length);
/*  MCC_MNC_TO_PLMNID(xnHandoverRequest->GUAMI.plmn_ID.mcc,
                    xnHandoverRequest->GUAMI.plmn_ID.mnc,
                    xnHandoverRequest->GUAMI.plmn_ID.mnc_digit_length,
                    &ie->value.choice.GUAMI.plmn_ID);  **/

/** Need to understand if this search is neeeded **/
  XNAP_FIND_PROTOCOLIE_BY_ID(XNAP_HandoverRequest_IEs_t, ie, xnHandoverRequest,
                             XNAP_ProtocolIE_ID_id_UEContextInfoHORequest, true);  

  if (ie == NULL ) {
    LOG_E (XNAP, "%s %d: ie is a NULL pointer \n",__FILE__,__LINE__);
    itti_free(ITTI_MSG_ORIGIN_ID(msg), msg);
    return -1;
  }

//  XNAP_HANDOVER_REQ(msg).mme_ue_s1ap_id = ie->value.choice.UE_ContextInformation.mME_UE_S1AP_ID;

  /* TODO: properly store Target Cell ID */

  XNAP_HANDOVER_REQ(msg).ue_context.target_assoc_id = assoc_id;

  XNAP_HANDOVER_REQ(msg).ue_context.security_capabilities.encryption_algorithms =
    BIT_STRING_to_uint16(&ie->value.choice.UEContextInfoHORequest.ueSecurityCapabilities.nr_EncyptionAlgorithms);
  XNAP_HANDOVER_REQ(msg).ue_context.security_capabilities.integrity_algorithms =
    BIT_STRING_to_uint16(&ie->value.choice.UEContextInfoHORequest.ueSecurityCapabilities.nr_IntegrityProtectionAlgorithms);

  if ((ie->value.choice.UEContextInfoHORequest.securityInformation.key_NG_RAN_Star.buf) &&
          (ie->value.choice.UEContextInfoHORequest.securityInformation.key_NG_RAN_Star.size == 32)) {
    memcpy(XNAP_HANDOVER_REQ(msg).ue_context.as_security_key_ranstar, ie->value.choice.UEContextInfoHORequest.securityInformation.key_NG_RAN_Star.buf, 32);
    XNAP_HANDOVER_REQ(msg).ue_context.as_security_ncc = ie->value.choice.UEContextInfoHORequest.securityInformation.ncc;
  } else {
    LOG_E (XNAP,"Size of gNB key star does not match the expected value\n");
  }

/*  if (ie->value.choice.UEContextInfoHORequest.pduSessionResourcesToBeSetup_List.list.count > 0) {

    XNAP_HANDOVER_REQ(msg).ue_context.pdusession_tobe_setup_list.num_pdu = ie->value.choice.UEContextInfoHORequest.pduSessionResourcesToBeSetup_List.list.count;

    for (int i=0;i<ie->value.choice.UEContextInfoHORequest.pduSessionResourcesToBeSetup_List.list.count;i++) {
      PDUSession_ToBeSetup_ItemIEs = (XNAP_PDUSessionResourcesToBeSetup_Item_t *) ie->value.choice.UE_ContextInfoHORequest.pduSessionResourcesToBeSetup_List.list.array[i];
      PDUSession_ToBeSetup_Item = &XNAP_PDUSessionResourcesToBeSetup_Item->value.choice.UE_ContextInfoHORequest.XNAP_PDUSessionResourcesToBeSetup_Item;

      XNAP_HANDOVER_REQ(msg).ue_context.pdusession_tobe_setup_list.pdu[i].pdusession_id = PDUSession_ToBeSetup_Item->pduSessionId;

      memcpy(XNAP_HANDOVER_REQ(msg).e_rabs_tobesetup[i].eNB_addr.buffer,
                     e_RABs_ToBeSetup_Item->uL_GTPtunnelEndpoint.transportLayerAddress.buf,
                     e_RABs_ToBeSetup_Item->uL_GTPtunnelEndpoint.transportLayerAddress.size);

      XNAP_HANDOVER_REQ(msg).e_rabs_tobesetup[i].eNB_addr.length =
                      e_RABs_ToBeSetup_Item->uL_GTPtunnelEndpoint.transportLayerAddress.size * 8 - e_RABs_ToBeSetup_Item->uL_GTPtunnelEndpoint.transportLayerAddress.bits_unused;

      OCTET_STRING_TO_INT32(&e_RABs_ToBeSetup_Item->uL_GTPtunnelEndpoint.gTP_TEID,
                                                XNAP_HANDOVER_REQ(msg).e_rabs_tobesetup[i].gtp_teid);

      XNAP_HANDOVER_REQ(msg).e_rab_param[i].qos.qci = e_RABs_ToBeSetup_Item->e_RAB_Level_QoS_Parameters.qCI;
      XNAP_HANDOVER_REQ(msg).e_rab_param[i].qos.allocation_retention_priority.priority_level = e_RABs_ToBeSetup_Item->e_RAB_Level_QoS_Parameters.allocationAndRetentionPriority.priorityLevel;
      XNAP_HANDOVER_REQ(msg).e_rab_param[i].qos.allocation_retention_priority.pre_emp_capability = e_RABs_ToBeSetup_Item->e_RAB_Level_QoS_Parameters.allocationAndRetentionPriority.pre_emptionCapability;
      XNAP_HANDOVER_REQ(msg).e_rab_param[i].qos.allocation_retention_priority.pre_emp_vulnerability = e_RABs_ToBeSetup_Item->e_RAB_Level_QoS_Parameters.allocationAndRetentionPriority.pre_emptionVulnerability;
    }

  }
    else {
    LOG_E (XNAP,"Can't decode the qos allocation\n");
  }
*/

//  XNAP_RRC_Context_t *c = &ie->value.choice.UE_ContextInformation.rRC_Context;
     OCTET_STRING_t *c = &ie->value.choice.UEContextInfoHORequest.rrc_Context;
  if (sizeof(c) > 8192 /* TODO: this is the size of rrc_buffer in struct x2ap_handover_req_s */)
    { printf("%s:%d: fatal: buffer too big\n", __FILE__, __LINE__); abort(); }

//  memcpy(XNAP_HANDOVER_REQ(msg).ue_context.rrc_buffer, c, sizeof(c));
  XNAP_HANDOVER_REQ(msg).ue_context.rrc_buffer_size = sizeof(c);

  itti_send_msg_to_task(TASK_RRC_GNB, instance_p->instance, msg);

  return 0;
}
                                                           

