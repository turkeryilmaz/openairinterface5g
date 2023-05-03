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

/*! \file rrc_gNB_NGAP.h
 * \brief rrc NGAP procedures for gNB
 * \author Yoshio INOUE, Masayuki HARADA
 * \date 2020
 * \version 0.1
 * \email: yoshio.inoue@fujitsu.com,masayuki.harada@fujitsu.com
 *         (yoshio.inoue%40fujitsu.com%2cmasayuki.harada%40fujitsu.com)
 */

#include "nrppa_gNB_NGAP.h"

/*#include "RRC/L2_INTERFACE/openair_rrc_L2_interface.h"
#include "rrc_eNB_S1AP.h"
#include "gnb_config.h"
#include "common/ran_context.h"*/

#include "oai_asn1.h"
#include "intertask_interface.h"
#include "pdcp.h"
#include "pdcp_primitives.h"

#include "S1AP_NAS-PDU.h"
#include "executables/softmodem-common.h"
#include "UTIL/OSA/osa_defs.h"
#include "ngap_gNB_defs.h"
#include "ngap_gNB_ue_context.h"
#include "ngap_gNB_management_procedures.h"
#include "NR_ULInformationTransfer.h"
#include "RRC/NR/MESSAGES/asn1_msg.h"
#include "NR_UERadioAccessCapabilityInformation.h"
#include "NR_UE-CapabilityRAT-ContainerList.h"
#include "NGAP_CauseRadioNetwork.h"
#include "f1ap_messages_types.h"
#include "openair2/E1AP/e1ap_asnc.h"
#include "NGAP_asn_constant.h"
#include "NGAP_PDUSessionResourceSetupRequestTransfer.h"
#include "NGAP_PDUSessionResourceModifyRequestTransfer.h"
#include "NGAP_ProtocolIE-Field.h"
#include "NGAP_GTPTunnel.h"
#include "NGAP_QosFlowSetupRequestItem.h"
#include "NGAP_QosFlowAddOrModifyRequestItem.h"
#include "NGAP_NonDynamic5QIDescriptor.h"
#include "conversions.h"
#include "RRC/NR/rrc_gNB_radio_bearers.h"
extern RAN_CONTEXT_t RC;

//------------------------------------------------------------------------------
void nrppa_gNB_send_NGAP_UPLINKUEASSOCIATEDNRPPA(   //this will process and send the nrppa pdu
  const protocol_ctxt_t    *const ctxt_pP,
  rrc_gNB_ue_context_t     *const ue_context_pP,
  NR_UL_DCCH_Message_t     *const ul_dcch_msg
);{


/*
 uint32_t pdu_length;
    uint8_t *pdu_buffer;
    MessageDef *msg_p;
    NR_ULInformationTransfer_t *ulInformationTransfer = ul_dcch_msg->message.choice.c1->choice.ulInformationTransfer;
    gNB_RRC_UE_t *UE = &ue_context_pP->ue_context;

    if (ulInformationTransfer->criticalExtensions.present == NR_ULInformationTransfer__criticalExtensions_PR_ulInformationTransfer) {
        pdu_length = ulInformationTransfer->criticalExtensions.choice.ulInformationTransfer->dedicatedNAS_Message->size;
        pdu_buffer = ulInformationTransfer->criticalExtensions.choice.ulInformationTransfer->dedicatedNAS_Message->buf;




        msg_p = itti_alloc_new_message (TASK_NRPPA_GNB, 0, NGAP_UPLINKUEASSOCIATEDNRPPA);
        NGAP_UPLINKUEASSOCIATEDNRPPA(msg_p).gNB_ue_ngap_id = UE->gNB_ue_ngap_id;
        NGAP_UPLINKUEASSOCIATEDNRPPA(msg_p).nas_pdu.length = pdu_length;
        NGAP_UPLINKUEASSOCIATEDNRPPA(msg_p).nas_pdu.buffer = pdu_buffer;
        NGAP_UPLINKUEASSOCIATEDNRPPA(msg_p).routing_id = 0; //TODO ad**l
        itti_send_msg_to_task (TASK_NGAP, ctxt_pP->instance, msg_p);
        LOG_I(NR_NRPPA,"Send RRC GNB UL Information Transfer \n");
    }

*/



}


//------------------------------------------------------------------------------
int rrc_gNB_process_NGAP_DOWNLINK_NAS(MessageDef *msg_p, instance_t instance, mui_t *rrc_gNB_mui)
//------------------------------------------------------------------------------
{
  uint32_t length;
  uint8_t *buffer;
  protocol_ctxt_t ctxt = {0};
  ngap_downlink_nas_t *req = &NGAP_DOWNLINK_NAS(msg_p);
  rrc_gNB_ue_context_t *ue_context_p = rrc_gNB_get_ue_context(RC.nrrrc[instance], req->gNB_ue_ngap_id);

  if (ue_context_p == NULL) {
    /* Can not associate this message to an UE index, send a failure to NGAP and discard it! */
    MessageDef *msg_fail_p;
    LOG_W(NR_RRC, "[gNB %ld] In NGAP_DOWNLINK_NAS: unknown UE from NGAP ids (%u)\n", instance, req->gNB_ue_ngap_id);
    msg_fail_p = itti_alloc_new_message(TASK_RRC_GNB, 0, NGAP_NAS_NON_DELIVERY_IND);
    ngap_nas_non_delivery_ind_t *msg = &NGAP_NAS_NON_DELIVERY_IND(msg_fail_p);
    msg->gNB_ue_ngap_id = req->gNB_ue_ngap_id;
    msg->nas_pdu.length = req->nas_pdu.length;
    msg->nas_pdu.buffer = req->nas_pdu.buffer;
    // TODO add failure cause when defined!
    itti_send_msg_to_task(TASK_NGAP, instance, msg_fail_p);
    return (-1);
  }

  gNB_RRC_UE_t *UE = &ue_context_p->ue_context;
  PROTOCOL_CTXT_SET_BY_INSTANCE(&ctxt, instance, GNB_FLAG_YES, UE->rnti, 0, 0);

  /* Create message for PDCP (DLInformationTransfer_t) */
  length = do_NR_DLInformationTransfer(instance, &buffer, rrc_gNB_get_next_transaction_identifier(instance), req->nas_pdu.length, req->nas_pdu.buffer);
  LOG_DUMPMSG(NR_RRC, DEBUG_RRC, buffer, length, "[MSG] RRC DL Information Transfer\n");
  /*
   * switch UL or DL NAS message without RRC piggybacked to SRB2 if active.
   */
  AssertFatal(!NODE_IS_DU(RC.nrrrc[ctxt.module_id]->node_type), "illegal node type DU: receiving NGAP messages at this node\n");
  /* Transfer data to PDCP */
  rb_id_t srb_id = UE->Srb[2].Active ? DCCH1 : DCCH;
  nr_rrc_data_req(&ctxt, srb_id, (*rrc_gNB_mui)++, SDU_CONFIRM_NO, length, buffer, PDCP_TRANSMISSION_MODE_CONTROL);
  return 0;
}







//------------------------------------------------------------------------------
void
rrc_gNB_send_NGAP_UPLINK_NAS(
  const protocol_ctxt_t    *const ctxt_pP,
  rrc_gNB_ue_context_t     *const ue_context_pP,
  NR_UL_DCCH_Message_t     *const ul_dcch_msg
)
//------------------------------------------------------------------------------
{
    uint32_t pdu_length;
    uint8_t *pdu_buffer;
    MessageDef *msg_p;
    NR_ULInformationTransfer_t *ulInformationTransfer = ul_dcch_msg->message.choice.c1->choice.ulInformationTransfer;
    gNB_RRC_UE_t *UE = &ue_context_pP->ue_context;

    if (ulInformationTransfer->criticalExtensions.present == NR_ULInformationTransfer__criticalExtensions_PR_ulInformationTransfer) {
        pdu_length = ulInformationTransfer->criticalExtensions.choice.ulInformationTransfer->dedicatedNAS_Message->size;
        pdu_buffer = ulInformationTransfer->criticalExtensions.choice.ulInformationTransfer->dedicatedNAS_Message->buf;
        msg_p = itti_alloc_new_message (TASK_RRC_GNB, 0, NGAP_UPLINK_NAS);
        NGAP_UPLINK_NAS(msg_p).gNB_ue_ngap_id = UE->gNB_ue_ngap_id;
        NGAP_UPLINK_NAS (msg_p).nas_pdu.length = pdu_length;
        NGAP_UPLINK_NAS (msg_p).nas_pdu.buffer = pdu_buffer;
        // extract_imsi(NGAP_UPLINK_NAS (msg_p).nas_pdu.buffer,
        //               NGAP_UPLINK_NAS (msg_p).nas_pdu.length,
        //               ue_context_pP);
        itti_send_msg_to_task (TASK_NGAP, ctxt_pP->instance, msg_p);
        LOG_I(NR_RRC,"Send RRC GNB UL Information Transfer \n");
    }
}

