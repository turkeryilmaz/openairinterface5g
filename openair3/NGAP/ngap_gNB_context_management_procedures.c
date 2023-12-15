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

/*! \file ngap_gNB_context_management_procedures.c
 * \brief NGAP context management procedures
 * \author  Yoshio INOUE, Masayuki HARADA
 * \date 2020
 * \email: yoshio.inoue@fujitsu.com,masayuki.harada@fujitsu.com (yoshio.inoue%40fujitsu.com%2cmasayuki.harada%40fujitsu.com)
 * \version 1.0
 * @ingroup _ngap
 */
 

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "assertions.h"
#include "conversions.h"

#include "intertask_interface.h"

#include "ngap_common.h"
#include "ngap_gNB_defs.h"

#include "ngap_gNB_itti_messaging.h"

#include "ngap_gNB_encoder.h"
#include "ngap_gNB_decoder.h"
#include "ngap_gNB_nnsf.h"
#include "ngap_gNB_ue_context.h"
#include "ngap_gNB_nas_procedures.h"
#include "ngap_gNB_management_procedures.h"
#include "ngap_gNB_context_management_procedures.h"
#include "NGAP_PDUSessionResourceItemCxtRelReq.h"
#include "NGAP_PDUSessionResourceItemHORqd.h"
#include "NGAP_HandoverRequiredTransfer.h"
#include "NGAP_SourceNGRANNode-ToTargetNGRANNode-TransparentContainer.h"
#include "NGAP_PDUSessionResourceInformationList.h"
#include "NGAP_PDUSessionResourceInformationItem.h"
#include "NGAP_QosFlowInformationList.h"
#include "NGAP_QosFlowInformationItem.h"
#include "NGAP_LastVisitedCellItem.h"
#include "NGAP_UEHistoryInformation.h"
#include "NGAP_LastVisitedNGRANCellInformation.h"
#include "NGAP_Cause.h"


int ngap_ue_context_release_complete(instance_t instance,
                                     ngap_ue_release_complete_t *ue_release_complete_p)
{

  ngap_gNB_instance_t                 *ngap_gNB_instance_p = NULL;
  struct ngap_gNB_ue_context_s        *ue_context_p        = NULL;
  NGAP_NGAP_PDU_t pdu = {0};
  uint8_t  *buffer;
  uint32_t length;

  /* Retrieve the NGAP gNB instance associated with Mod_id */
  ngap_gNB_instance_p = ngap_gNB_get_instance(instance);

  DevAssert(ue_release_complete_p != NULL);
  DevAssert(ngap_gNB_instance_p != NULL);

  if ((ue_context_p = ngap_get_ue_context(ue_release_complete_p->gNB_ue_ngap_id)) == NULL) {
    /* The context for this gNB ue ngap id doesn't exist in the map of gNB UEs */
    NGAP_ERROR("Failed to find ue context associated with gNB ue ngap id: %u\n", ue_release_complete_p->gNB_ue_ngap_id);
    return -1;
  }

  /* Prepare the NGAP message to encode */
  pdu.present = NGAP_NGAP_PDU_PR_successfulOutcome;
  asn1cCalloc(pdu.choice.successfulOutcome, head);
  head->procedureCode = NGAP_ProcedureCode_id_UEContextRelease;
  head->criticality = NGAP_Criticality_reject;
  head->value.present = NGAP_SuccessfulOutcome__value_PR_UEContextReleaseComplete;
  NGAP_UEContextReleaseComplete_t *out = &head->value.choice.UEContextReleaseComplete;

  /* mandatory */
  {
    asn1cSequenceAdd(out->protocolIEs.list, NGAP_UEContextReleaseComplete_IEs_t, ie);
    ie->id = NGAP_ProtocolIE_ID_id_AMF_UE_NGAP_ID;
    ie->criticality = NGAP_Criticality_ignore;
    ie->value.present = NGAP_UEContextReleaseComplete_IEs__value_PR_AMF_UE_NGAP_ID;
    asn_uint642INTEGER(&ie->value.choice.AMF_UE_NGAP_ID, ue_context_p->amf_ue_ngap_id);
  }

  /* mandatory */
  {
    asn1cSequenceAdd(out->protocolIEs.list, NGAP_UEContextReleaseComplete_IEs_t, ie);
    ie->id = NGAP_ProtocolIE_ID_id_RAN_UE_NGAP_ID;
    ie->criticality = NGAP_Criticality_ignore;
    ie->value.present = NGAP_UEContextReleaseComplete_IEs__value_PR_RAN_UE_NGAP_ID;
    ie->value.choice.RAN_UE_NGAP_ID = ue_release_complete_p->gNB_ue_ngap_id;
  }

  if (ngap_gNB_encode_pdu(&pdu, &buffer, &length) < 0) {
    /* Encode procedure has failed... */
    NGAP_ERROR("Failed to encode UE context release complete\n");
    return -1;
  }

  /* UE associated signalling -> use the allocated stream */
  ngap_gNB_itti_send_sctp_data_req(ngap_gNB_instance_p->instance,
                                   ue_context_p->amf_ref->assoc_id, buffer,
                                   length, ue_context_p->tx_stream);

  //LG ngap_gNB_itti_send_sctp_close_association(ngap_gNB_instance_p->instance,
  //                                             ue_context_p->amf_ref->assoc_id);
  // release UE context
  ngap_gNB_ue_context_t *tmp = ngap_detach_ue_context(ue_release_complete_p->gNB_ue_ngap_id);
  if (tmp)
    free(tmp);
  return 0;
}

int ngap_ue_context_release_req(instance_t instance,
                                ngap_ue_release_req_t *ue_release_req_p)
{
  ngap_gNB_instance_t                *ngap_gNB_instance_p           = NULL;
  struct ngap_gNB_ue_context_s       *ue_context_p                  = NULL;
  NGAP_NGAP_PDU_t pdu = {0};
  uint8_t                            *buffer                        = NULL;
  uint32_t                            length;
  /* Retrieve the NGAP gNB instance associated with Mod_id */
  ngap_gNB_instance_p = ngap_gNB_get_instance(instance);

  DevAssert(ue_release_req_p != NULL);
  DevAssert(ngap_gNB_instance_p != NULL);

  if ((ue_context_p = ngap_get_ue_context(ue_release_req_p->gNB_ue_ngap_id)) == NULL) {
    /* The context for this gNB ue ngap id doesn't exist in the map of gNB UEs */
    NGAP_WARN("Failed to find ue context associated with gNB ue ngap id: %u\n",
              ue_release_req_p->gNB_ue_ngap_id);
    /* send response to free the UE: we don't know it, but it should be
     * released since RRC seems to know it (e.g., there is no AMF) */
    MessageDef *msg = itti_alloc_new_message(TASK_NGAP, 0, NGAP_UE_CONTEXT_RELEASE_COMMAND);
    NGAP_UE_CONTEXT_RELEASE_COMMAND(msg).gNB_ue_ngap_id = ue_release_req_p->gNB_ue_ngap_id;
    itti_send_msg_to_task(TASK_RRC_GNB, ngap_gNB_instance_p->instance, msg);
    return -1;
  }

  /* Prepare the NGAP message to encode */
  pdu.present = NGAP_NGAP_PDU_PR_initiatingMessage;
  asn1cCalloc(pdu.choice.initiatingMessage, head);
  head->procedureCode = NGAP_ProcedureCode_id_UEContextReleaseRequest;
  head->criticality = NGAP_Criticality_ignore;
  head->value.present = NGAP_InitiatingMessage__value_PR_UEContextReleaseRequest;
  NGAP_UEContextReleaseRequest_t *out = &head->value.choice.UEContextReleaseRequest;

  /* mandatory */
  {
    asn1cSequenceAdd(out->protocolIEs.list, NGAP_UEContextReleaseRequest_IEs_t, ie);
    ie->id = NGAP_ProtocolIE_ID_id_AMF_UE_NGAP_ID;
    ie->criticality = NGAP_Criticality_reject;
    ie->value.present = NGAP_UEContextReleaseRequest_IEs__value_PR_AMF_UE_NGAP_ID;
    asn_uint642INTEGER(&ie->value.choice.AMF_UE_NGAP_ID, ue_context_p->amf_ue_ngap_id);
  }

  /* mandatory */
  {
    asn1cSequenceAdd(out->protocolIEs.list, NGAP_UEContextReleaseRequest_IEs_t, ie);
    ie->id = NGAP_ProtocolIE_ID_id_RAN_UE_NGAP_ID;
    ie->criticality = NGAP_Criticality_reject;
    ie->value.present = NGAP_UEContextReleaseRequest_IEs__value_PR_RAN_UE_NGAP_ID;
    ie->value.choice.RAN_UE_NGAP_ID = ue_release_req_p->gNB_ue_ngap_id;
  }

  /* optional */
  if (ue_release_req_p->nb_of_pdusessions > 0) {
    asn1cSequenceAdd(out->protocolIEs.list, NGAP_UEContextReleaseRequest_IEs_t, ie);
    ie->id = NGAP_ProtocolIE_ID_id_PDUSessionResourceListCxtRelReq;
    ie->criticality = NGAP_Criticality_reject;
    ie->value.present = NGAP_UEContextReleaseRequest_IEs__value_PR_PDUSessionResourceListCxtRelReq;
    for (int i = 0; i < ue_release_req_p->nb_of_pdusessions; i++) {
      NGAP_PDUSessionResourceItemCxtRelReq_t     *item;
      item = (NGAP_PDUSessionResourceItemCxtRelReq_t *)calloc(1,sizeof(NGAP_PDUSessionResourceItemCxtRelReq_t));
      item->pDUSessionID = ue_release_req_p->pdusessions[i].pdusession_id;
      asn1cSeqAdd(&ie->value.choice.PDUSessionResourceListCxtRelReq.list, item);
    }
  }

  /* mandatory */
  {
    asn1cSequenceAdd(out->protocolIEs.list, NGAP_UEContextReleaseRequest_IEs_t, ie);
    ie->id = NGAP_ProtocolIE_ID_id_Cause;
    ie->criticality = NGAP_Criticality_ignore;
    ie->value.present = NGAP_UEContextReleaseRequest_IEs__value_PR_Cause;
    DevAssert(ue_release_req_p->cause <= NGAP_Cause_PR_choice_ExtensionS);
    switch(ue_release_req_p->cause){
      case NGAP_CAUSE_RADIO_NETWORK:
	ie->value.choice.Cause.present = NGAP_Cause_PR_radioNetwork;
	ie->value.choice.Cause.choice.radioNetwork = ue_release_req_p->cause_value;
	break;
      case NGAP_CAUSE_TRANSPORT:
	ie->value.choice.Cause.present = NGAP_Cause_PR_transport;
	ie->value.choice.Cause.choice.transport = ue_release_req_p->cause_value;
	break;
      case NGAP_CAUSE_NAS:
	ie->value.choice.Cause.present = NGAP_Cause_PR_nas;
	ie->value.choice.Cause.choice.nas = ue_release_req_p->cause_value;
	break;
      case NGAP_CAUSE_PROTOCOL:
	ie->value.choice.Cause.present = NGAP_Cause_PR_protocol;
	ie->value.choice.Cause.choice.protocol = ue_release_req_p->cause_value;
	break;
      case NGAP_CAUSE_MISC:
	ie->value.choice.Cause.present = NGAP_Cause_PR_misc;
	ie->value.choice.Cause.choice.misc = ue_release_req_p->cause_value;
	break;
      default:
        NGAP_WARN("Received NG Error indication cause NGAP_Cause_PR_choice_Extensions\n");
        break;
    }
  }

  if (ngap_gNB_encode_pdu(&pdu, &buffer, &length) < 0) {
    /* Encode procedure has failed... */
    NGAP_ERROR("Failed to encode UE context release complete\n");
    return -1;
  }

  /* UE associated signalling -> use the allocated stream */
  ngap_gNB_itti_send_sctp_data_req(ngap_gNB_instance_p->instance,
                                   ue_context_p->amf_ref->assoc_id, buffer,
                                   length, ue_context_p->tx_stream);

  return 0;
}

int ngap_gNB_handover_required(instance_t instance, ngap_handover_required_t* handover_required_p)
{
  LOG_I(NR_RRC, "HO LOG: NGAP Handover Required Preparation!\n");
  ngap_gNB_instance_t            *ngap_gNB_instance_p = NULL;
  struct ngap_gNB_ue_context_s   *ue_context_p        = NULL;
  NGAP_NGAP_PDU_t pdu;
  uint8_t  *buffer  = NULL;
  uint32_t length;
  /* Retrieve the NGAP gNB instance associated with Mod_id */
  ngap_gNB_instance_p = ngap_gNB_get_instance(instance);
  DevAssert(handover_required_p != NULL);
  DevAssert(ngap_gNB_instance_p != NULL);

  if ((ue_context_p = ngap_get_ue_context(handover_required_p->gNB_ue_ngap_id)) == NULL) {
    /* The context for this gNB ue ngap id doesn't exist in the map of gNB UEs */
    NGAP_WARN("Failed to find ue context associated with gNB ue ngap id: 0x%08x\n", handover_required_p->gNB_ue_ngap_id);
    return -1;
  }  

  /* Prepare the NGAP message to encode */
  memset(&pdu, 0, sizeof(pdu));
  pdu.present = NGAP_NGAP_PDU_PR_initiatingMessage;
  asn1cCalloc(pdu.choice.initiatingMessage, head);
  head->procedureCode = NGAP_ProcedureCode_id_HandoverPreparation;
  head->criticality = NGAP_Criticality_reject;
  head->value.present = NGAP_InitiatingMessage__value_PR_HandoverRequired;
  NGAP_HandoverRequired_t *out = &head->value.choice.HandoverRequired;

  /* mandatory */
  {
    asn1cSequenceAdd(out->protocolIEs.list, NGAP_HandoverRequiredIEs_t, ie);
    ie->id = NGAP_ProtocolIE_ID_id_AMF_UE_NGAP_ID;
    ie->criticality = NGAP_Criticality_reject;
    ie->value.present = NGAP_HandoverRequiredIEs__value_PR_AMF_UE_NGAP_ID;
    asn_uint642INTEGER(&ie->value.choice.AMF_UE_NGAP_ID, ue_context_p->amf_ue_ngap_id);
  }


    /* mandatory */
  {
    asn1cSequenceAdd(out->protocolIEs.list, NGAP_HandoverRequiredIEs_t, ie);
    ie->id = NGAP_ProtocolIE_ID_id_RAN_UE_NGAP_ID;
    ie->criticality = NGAP_Criticality_reject;
    ie->value.present = NGAP_HandoverRequiredIEs__value_PR_RAN_UE_NGAP_ID;
    ie->value.choice.RAN_UE_NGAP_ID = (int64_t)ue_context_p->gNB_ue_ngap_id;
  }

  /* mandatory */
  {
    asn1cSequenceAdd(out->protocolIEs.list, NGAP_HandoverRequiredIEs_t, ie);
    ie->id = NGAP_ProtocolIE_ID_id_HandoverType;
    ie->criticality = NGAP_Criticality_reject;
    ie->value.present = NGAP_HandoverRequiredIEs__value_PR_HandoverType;
    ie->value.choice.HandoverType = (int64_t)handover_required_p->handoverType; //NGAP_HandoverType_intra5gs
  }

  /* mandatory */
  {
    asn1cSequenceAdd(out->protocolIEs.list, NGAP_HandoverRequiredIEs_t, ie);
    ie->id = NGAP_ProtocolIE_ID_id_Cause;
    ie->criticality = NGAP_Criticality_ignore;
    ie->value.present = NGAP_HandoverRequiredIEs__value_PR_Cause;
    ie->value.choice.Cause.present = NGAP_Cause_PR_radioNetwork;
    ie->value.choice.Cause.choice.radioNetwork = handover_required_p->cause; //NGAP_CauseRadioNetwork_handover_desirable_for_radio_reason
  }

  /* mandatory */
  {
    asn1cSequenceAdd(out->protocolIEs.list, NGAP_HandoverRequiredIEs_t, ie);
    ie->id = NGAP_ProtocolIE_ID_id_TargetID;
    ie->criticality = NGAP_Criticality_reject;
    ie->value.present = NGAP_HandoverRequiredIEs__value_PR_TargetID;
    ie->value.choice.TargetID.present = NGAP_TargetID_PR_targetRANNodeID;
    asn1cCalloc(ie->value.choice.TargetID.choice.targetRANNodeID, targetRan);
    targetRan->globalRANNodeID.present = NGAP_GlobalRANNodeID_PR_globalGNB_ID;
    asn1cCalloc(targetRan->globalRANNodeID.choice.globalGNB_ID, globalGnbId);

    globalGnbId->gNB_ID.present = NGAP_GNB_ID_PR_gNB_ID;
    
    MACRO_GNB_ID_TO_BIT_STRING(handover_required_p->target_gnb_id.targetgNBId,
                                &globalGnbId->gNB_ID.choice.gNB_ID);


    MCC_MNC_TO_PLMNID(handover_required_p->target_gnb_id.plmn_identity.mcc,
                    handover_required_p->target_gnb_id.plmn_identity.mnc,
                    handover_required_p->target_gnb_id.plmn_identity.mnc_digit_length,
                    &globalGnbId->pLMNIdentity);

    /* Set TAI */
    INT24_TO_OCTET_STRING(handover_required_p->target_gnb_id.tac, &targetRan->selectedTAI.tAC);
    MCC_MNC_TO_PLMNID(handover_required_p->target_gnb_id.plmn_identity.mcc,
                    handover_required_p->target_gnb_id.plmn_identity.mnc,
                    handover_required_p->target_gnb_id.plmn_identity.mnc_digit_length,
                    &targetRan->selectedTAI.pLMNIdentity);

  }

    /* mandatory */
  {
    asn1cSequenceAdd(out->protocolIEs.list, NGAP_HandoverRequiredIEs_t, ie);
    ie->id = NGAP_ProtocolIE_ID_id_PDUSessionResourceListHORqd;
    ie->criticality = NGAP_Criticality_reject;
    ie->value.present = NGAP_HandoverRequiredIEs__value_PR_PDUSessionResourceListHORqd;
    for (int i = 0; i < handover_required_p->nb_of_pdusessions; ++i) {
      asn1cSequenceAdd(ie->value.choice.PDUSessionResourceListHORqd.list, NGAP_PDUSessionResourceItemHORqd_t, hoRequiredPduSession);
      hoRequiredPduSession->pDUSessionID = handover_required_p->pdusessions[i].pdusession_id;
      
      NGAP_HandoverRequiredTransfer_t hoRequiredTransfer = {0};
      asn1cCalloc(hoRequiredTransfer.directForwardingPathAvailability, directFwdPathAvl);
      *directFwdPathAvl = NGAP_DirectForwardingPathAvailability_direct_path_available; //set False! (?)

      void *hoRequiredTransferPduSessionBuf;
      ssize_t encoded = aper_encode_to_new_buffer(&asn_DEF_NGAP_HandoverRequiredTransfer, NULL, &hoRequiredTransfer, &hoRequiredTransferPduSessionBuf);
      AssertFatal(encoded > 0, "HO LOG: asn_DEF_NGAP_HandoverRequiredTransfer ASN1 message encoding failed !\n");
      hoRequiredPduSession->handoverRequiredTransfer.buf = CALLOC(1, encoded);
      hoRequiredPduSession->handoverRequiredTransfer.buf = hoRequiredTransferPduSessionBuf;
      hoRequiredPduSession->handoverRequiredTransfer.size = encoded;
      
      ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_NGAP_HandoverRequiredTransfer, &hoRequiredTransfer);
    }
  }

  /* optional */
  {
    asn1cSequenceAdd(out->protocolIEs.list, NGAP_HandoverRequiredIEs_t, ie);
    ie->id = NGAP_ProtocolIE_ID_id_DirectForwardingPathAvailability;
    ie->criticality = NGAP_Criticality_ignore;
    ie->value.present = NGAP_HandoverRequiredIEs__value_PR_DirectForwardingPathAvailability;
    ie->value.choice.DirectForwardingPathAvailability = NGAP_DirectForwardingPathAvailability_direct_path_available;
  }

  /* mandatory */
  {
    asn1cSequenceAdd(out->protocolIEs.list, NGAP_HandoverRequiredIEs_t, ie);
    ie->id = NGAP_ProtocolIE_ID_id_SourceToTarget_TransparentContainer;
    ie->criticality = NGAP_Criticality_reject;
    ie->value.present = NGAP_HandoverRequiredIEs__value_PR_SourceToTarget_TransparentContainer;
    

    NGAP_SourceNGRANNode_ToTargetNGRANNode_TransparentContainer_t* transparentContainer =  (NGAP_SourceNGRANNode_ToTargetNGRANNode_TransparentContainer_t*)calloc(1, sizeof(NGAP_SourceNGRANNode_ToTargetNGRANNode_TransparentContainer_t));
    
    //rRC Container
    {
      // Handover Prep Information
      transparentContainer->rRCContainer.size = handover_required_p->sourceToTargetContainer->handoverInfo.length;
      transparentContainer->rRCContainer.buf =  (uint8_t *)malloc(handover_required_p->sourceToTargetContainer->handoverInfo.length);
      memcpy(transparentContainer->rRCContainer.buf, handover_required_p->sourceToTargetContainer->handoverInfo.buffer ,transparentContainer->rRCContainer.size);
      
    }


    //PDU Session Information
    {
      asn1cCalloc(transparentContainer->pDUSessionResourceInformationList, pduSessionList);
      for (uint8_t i = 0; i < handover_required_p->nb_of_pdusessions; ++i)
      {


        asn1cSequenceAdd(pduSessionList->list, NGAP_PDUSessionResourceInformationItem_t, pduSesResourceStp);
        pduSesResourceStp->pDUSessionID = (int64_t)handover_required_p->pdusessions[i].pdusession_id;

        pduSesResourceStp->dRBsToQosFlowsMappingList = NULL; 
        pduSesResourceStp->iE_Extensions = NULL;

        for (int qosFlowId = 0; qosFlowId < handover_required_p->pdusessions[i].nb_of_qos_flow; ++qosFlowId) {
          asn1cSequenceAdd(pduSesResourceStp->qosFlowInformationList.list, NGAP_QosFlowInformationItem_t, qosFlowInfo);
          qosFlowInfo->qosFlowIdentifier = (int64_t)handover_required_p->pdusessions[i].associated_qos_flows[qosFlowId].qfi;
        }
      }
    }

    // NrCGI - Mandatory
    {
      transparentContainer->targetCell_ID.present = NGAP_NGRAN_CGI_PR_nR_CGI;
      asn1cCalloc(transparentContainer->targetCell_ID.choice.nR_CGI, tNrCGI);
      tNrCGI->iE_Extensions = NULL;
      MCC_MNC_TO_PLMNID(handover_required_p->target_gnb_id.plmn_identity.mcc,
                      handover_required_p->target_gnb_id.plmn_identity.mnc,
                      handover_required_p->target_gnb_id.plmn_identity.mnc_digit_length,
                      &tNrCGI->pLMNIdentity);

      NR_CELL_ID_TO_BIT_STRING(handover_required_p->sourceToTargetContainer->targetCellId.nrCellIdentity, &tNrCGI->nRCellIdentity);
    }

    //UE history Information - Mandatory
    {
      asn1cSequenceAdd(transparentContainer->uEHistoryInformation.list, NGAP_LastVisitedCellItem_t, lastVisitedCell);
      lastVisitedCell->iE_Extensions = NULL;
      lastVisitedCell->lastVisitedCellInformation.present = NGAP_LastVisitedCellInformation_PR_nGRANCell;
      asn1cCalloc(lastVisitedCell->lastVisitedCellInformation.choice.nGRANCell, lastVisitedNR);
      lastVisitedNR->cellType.cellSize = NGAP_CellSize_small;
      lastVisitedNR->globalCellID.present = NGAP_NGRAN_CGI_PR_nR_CGI;

      asn1cCalloc(lastVisitedNR->globalCellID.choice.nR_CGI, lastVisitedNrCGI);
      MCC_MNC_TO_PLMNID(handover_required_p->target_gnb_id.plmn_identity.mcc,
                      handover_required_p->target_gnb_id.plmn_identity.mnc,
                      handover_required_p->target_gnb_id.plmn_identity.mnc_digit_length,
                      &lastVisitedNrCGI->pLMNIdentity);

      NR_CELL_ID_TO_BIT_STRING(handover_required_p->sourceToTargetContainer->targetCellId.nrCellIdentity, &lastVisitedNrCGI->nRCellIdentity);

      asn1cCalloc(lastVisitedNR->hOCauseValue, lastVisitedCause);
      lastVisitedCause->present = NGAP_Cause_PR_radioNetwork;
      lastVisitedCause->choice.radioNetwork = NGAP_CauseRadioNetwork_handover_desirable_for_radio_reason;
      //TODO: It is mandatory adding dummy number for now
      lastVisitedNR->timeUEStayedInCell = (int64_t)500;
    }

    uint8_t *buf = NULL;
    int encoded = aper_encode_to_new_buffer(&asn_DEF_NGAP_SourceNGRANNode_ToTargetNGRANNode_TransparentContainer, NULL, transparentContainer, (void**)&buf);
    AssertFatal(encoded > 0, "HO LOG: asn_DEF_NGAP_SourceNGRANNode_ToTargetNGRANNode_TransparentContainer ASN1 message encoding failed !\n");
    LOG_I(NR_RRC, "HO LOG Encoding Result and Length of The Transparent Container: %d!\n\n", encoded);

    OCTET_STRING_fromBuf(&ie->value.choice.SourceToTarget_TransparentContainer, (const char *)buf,
                            encoded);
    
  }

  if (ngap_gNB_encode_pdu(&pdu, &buffer, &length) < 0) {
    /* Encode procedure has failed... */
    NGAP_ERROR("Failed to encode Handover Required Msg\n");
    return -1;
  }

  /* UE associated signalling -> use the allocated stream */
  ngap_gNB_itti_send_sctp_data_req(ngap_gNB_instance_p->instance,
                                   ue_context_p->amf_ref->assoc_id, buffer,
                                   length, ue_context_p->tx_stream);

  return 0;
}


