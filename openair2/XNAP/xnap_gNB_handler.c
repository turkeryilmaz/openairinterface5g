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
 * \author Konstantinos Alexandris <Konstantinos.Alexandris@eurecom.fr>, Cedric Roux <Cedric.Roux@eurecom.fr>, Navid Nikaein <Navid.Nikaein@eurecom.fr>
 * \date 2018
 * \version 1.0
 */

#include <stdint.h>

#include "intertask_interface.h"

#include "xnap_common.h"
#include "xnap_gNB_defs.h"
#include "xnap_gNB_handler.h"
#include "xnap_gNB_decoder.h"
#include "xnap_ids.h"
#include "XNAP_GlobalgNB-ID.h"
#include "xnap_gNB_management_procedures.h"
#include "xnap_gNB_generate_messages.h"

#include "assertions.h"
#include "conversions.h"
#include "XNAP_NRFrequencyBandItem.h"
#include "XNAP_GlobalNG-RANNode-ID.h"

static
int xnap_gNB_handle_xn_setup_request (instance_t instance,
                                      uint32_t assoc_id,
                                      uint32_t stream,
                                      XNAP_XnAP_PDU_t *pdu);
static
int xnap_gNB_handle_xn_setup_response (instance_t instance,
                                       uint32_t assoc_id,
                                       uint32_t stream,
                                       XNAP_XnAP_PDU_t *pdu);
/*static
int xnap_gNB_handle_xn_reset_request (instance_t instance,
                                      uint32_t assoc_id,
                                      uint32_t stream,
                                      XNAP_XnAP_PDU_t *pdu);*/
/*static
int xnap_gNB_handle_xn_reset_response (instance_t instance,
                                       uint32_t assoc_id,
                                       uint32_t stream,
                                       XNAP_XnAP_PDU_t *pdu);*/

static
int xnap_gNB_handle_xn_setup_failure (instance_t instance,
                                      uint32_t assoc_id,
                                      uint32_t stream,
                                      XNAP_XnAP_PDU_t *pdu);


/* Handlers matrix. Only gNB related procedure present here. Placement of callback functions according to XNAP_ProcedureCode.h */
static const xnap_message_decoded_callback xnap_messages_callback[][3] = {
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0}, /* loadIndication */
    {0, 0, 0}, /* errorIndication */
    {0, 0, 0}, /* snStatusTransfer */
    //{xnap_gNB_handle_ue_context_release, 0, 0}, /* uEContextRelease */
    {0, 0, 0},
    //{xnap_gNB_handle_xn_reset_request, xnap_gNB_handle_xn_reset_response, 0}, /* reset */
    {0, 0, 0},
    {0, 0, 0}, /* gNBConfigurationUpdate */
    {0, 0, 0}, /* resourceStatusReportingInitiation */
    {0, 0, 0}, /* resourceStatusReporting */
    {0, 0, 0}, /* privateMessage */
    {0, 0, 0}, /* mobilitySettingsChange */
    {0, 0, 0}, /* rLFIndication */
    {0, 0, 0}, /* handoverReport */
    {0, 0, 0}, /* cellActivation */
    {0, 0, 0}, /* xnRelease */
    {0, 0, 0},
    {xnap_gNB_handle_xn_setup_request, xnap_gNB_handle_xn_setup_response, xnap_gNB_handle_xn_setup_failure}, /* xnSetup */
    {0, 0, 0}, /* xnAPMessageTransfer */
    {0, 0, 0}, /* xnRemoval */
    //{xnap_gNB_handle_sgnb_addition_request,
     //xnap_gNB_handle_sgnb_addition_request_ack,
     //xnap_gNB_handle_sgnb_addition_request_reject}, /* sgNBAdditionPreparation */
    {0, 0, 0},
    {0, 0, 0}, /* sgNBReconfigurationCompletion */
    {0, 0, 0}, /* mgNBinitiatedSgNBModificationPreparation */
    {0, 0, 0}, /* sgNBinitiatedSgNBModification */
    {0, 0, 0}, /* mgNBinitiatedSgNBRelease */
    {0, 0, 0}, /* sgNBinitiatedSgNBRelease */
    {0, 0, 0}, /* sgNBCounterCheck */
    {0, 0, 0}, /* retrieveUEContext */
    //{xnap_gNB_handle_ENDC_sGNB_addition_request,
     //xnap_gNB_handle_ENDC_sGNB_addition_response,
     //0}, /*XNAP_ProcedureCode_id_sgNBAdditionPreparation*/
    {0, 0, 0},
    //{xnap_gNB_handle_ENDC_sGNB_reconfiguration_complete, 0, 0}, /*XNAP_ProcedureCode_id_sgNBReconfigurationCompletion*/
    {0, 0, 0}, //added sreeshma
    {0, 0, 0},
    {0, 0, 0},
    //{xnap_gNB_handle_ENDC_sGNB_release_request,
     //xnap_gNB_handle_ENDC_sGNB_release_request_acknowledge,
     //0}, /* mgNBinitiatedSgNBRelease */
    {0, 0, 0}, //added
    //{xnap_gNB_handle_ENDC_sGNB_release_required, xnap_gNB_handle_ENDC_sGNB_release_confirm, 0}, /* sgNBinitiatedSgNBRelease */
    {0, 0, 0}, //added
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    //{xnap_gNB_handle_ENDC_xn_setup_request, xnap_gNB_handle_ENDC_xn_setup_response, 0}, /*XNAP_ProcedureCode_id_endcXNSetup*/
    {0, 0, 0}, //added
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

void xnap_handle_xn_setup_message(xnap_gNB_instance_t *instance_p, xnap_gNB_data_t *gnb_desc_p, int sctp_shutdown)
{
  if (sctp_shutdown) {
    /* A previously connected gNB has been shutdown */

    /* TODO check if it was used by some gNB and send a message to inform these gNB if there is no more associated gNB */
    if (gnb_desc_p->state == XNAP_GNB_STATE_CONNECTED) {
      gnb_desc_p->state = XNAP_GNB_STATE_DISCONNECTED;

      if (instance_p-> xn_target_gnb_associated_nb > 0) {
        /* Decrease associated gNB number */
        instance_p-> xn_target_gnb_associated_nb --;
      }

      /* If there are no more associated gNB, inform gNB app */
      if (instance_p->xn_target_gnb_associated_nb == 0) {
        MessageDef                 *message_p;

        message_p = itti_alloc_new_message(TASK_XNAP, 0, XNAP_DEREGISTERED_GNB_IND);
        XNAP_DEREGISTERED_GNB_IND(message_p).nb_xn = 0;
        itti_send_msg_to_task(TASK_GNB_APP, instance_p->instance, message_p);
      }
    }
  } else {
    /* Check that at least one setup message is pending */
    DevCheck(instance_p->xn_target_gnb_pending_nb > 0,
             instance_p->instance,
             instance_p->xn_target_gnb_pending_nb, 0);

    if (instance_p->xn_target_gnb_pending_nb > 0) {
      /* Decrease pending messages number */
      instance_p->xn_target_gnb_pending_nb --;
    }

    /* If there are no more pending messages, inform gNB app */
    if (instance_p->xn_target_gnb_pending_nb == 0) {
      MessageDef                 *message_p;

      message_p = itti_alloc_new_message(TASK_XNAP, 0, XNAP_REGISTER_GNB_CNF);
      XNAP_REGISTER_GNB_CNF(message_p).nb_xn = instance_p->xn_target_gnb_associated_nb;
      itti_send_msg_to_task(TASK_GNB_APP, instance_p->instance, message_p);
    }
  }
}


int xnap_gNB_handle_message(instance_t instance, uint32_t assoc_id, int32_t stream,
                                const uint8_t *const data, const uint32_t data_length)
{
  XNAP_XnAP_PDU_t pdu;
  int ret = 0;

  DevAssert(data != NULL);

  memset(&pdu, 0, sizeof(pdu));

  //printf("Data length received: %d\n", data_length);

  if (xnap_gNB_decode_pdu(&pdu, data, data_length) < 0) {
    XNAP_ERROR("Failed to decode PDU\n");
    return -1;
  }

  switch (pdu.present) {

  case XNAP_XnAP_PDU_PR_initiatingMessage:
    /* Checking procedure Code and direction of message */
    if (pdu.choice.initiatingMessage->procedureCode >= sizeof(xnap_messages_callback) / (3 * sizeof(
          xnap_message_decoded_callback))) {
        //|| (pdu.present > XNAP_XnAP_PDU_PR_unsuccessfulOutcome)) {
      XNAP_ERROR("[SCTP %d] Either procedureCode %ld exceed expected\n",
                 assoc_id, pdu.choice.initiatingMessage->procedureCode);
      ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_XNAP_XnAP_PDU, &pdu);
      return -1;
    }

    /* No handler present.
     * This can mean not implemented or no procedure for gNB (wrong direction).
     */
    if (xnap_messages_callback[pdu.choice.initiatingMessage->procedureCode][pdu.present - 1] == NULL) {
      XNAP_ERROR("[SCTP %d] No handler for procedureCode %ld in %s\n",
                  assoc_id, pdu.choice.initiatingMessage->procedureCode,
                 xnap_direction2String(pdu.present - 1));
      ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_XNAP_XnAP_PDU, &pdu);
      return -1;
    }
    /* Calling the right handler */
    ret = (*xnap_messages_callback[pdu.choice.initiatingMessage->procedureCode][pdu.present - 1])
        (instance, assoc_id, stream, &pdu);
    break;

  case XNAP_XnAP_PDU_PR_successfulOutcome:
    /* Checking procedure Code and direction of message */
    if (pdu.choice.successfulOutcome->procedureCode >= sizeof(xnap_messages_callback) / (3 * sizeof(
          xnap_message_decoded_callback))) {
        //|| (pdu.present > XNAP_XnAP_PDU_PR_unsuccessfulOutcome)) {
      XNAP_ERROR("[SCTP %d] Either procedureCode %ld exceed expected\n",
                 assoc_id, pdu.choice.successfulOutcome->procedureCode);
      ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_XNAP_XnAP_PDU, &pdu);
      return -1;
    }

    /* No handler present.
     * This can mean not implemented or no procedure for gNB (wrong direction).
     */
    if (xnap_messages_callback[pdu.choice.successfulOutcome->procedureCode][pdu.present - 1] == NULL) {
      XNAP_ERROR("[SCTP %d] No handler for procedureCode %ld in %s\n",
                  assoc_id, pdu.choice.successfulOutcome->procedureCode,
                 xnap_direction2String(pdu.present - 1));
      ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_XNAP_XnAP_PDU, &pdu);
      return -1;
    }
    /* Calling the right handler */
    ret = (*xnap_messages_callback[pdu.choice.successfulOutcome->procedureCode][pdu.present - 1])
        (instance, assoc_id, stream, &pdu);
    break;

  case XNAP_XnAP_PDU_PR_unsuccessfulOutcome:
    /* Checking procedure Code and direction of message */
    if (pdu.choice.unsuccessfulOutcome->procedureCode >= sizeof(xnap_messages_callback) / (3 * sizeof(
          xnap_message_decoded_callback))) {
        //|| (pdu.present > XNAP_XnAP_PDU_PR_unsuccessfulOutcome)) {
      XNAP_ERROR("[SCTP %d] Either procedureCode %ld exceed expected\n",
                 assoc_id, pdu.choice.unsuccessfulOutcome->procedureCode);
      ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_XNAP_XnAP_PDU, &pdu);
      return -1;
    }

    /* No handler present.
     * This can mean not implemented or no procedure for gNB (wrong direction).
     */
    if (xnap_messages_callback[pdu.choice.unsuccessfulOutcome->procedureCode][pdu.present - 1] == NULL) {
      XNAP_ERROR("[SCTP %d] No handler for procedureCode %ld in %s\n",
                  assoc_id, pdu.choice.unsuccessfulOutcome->procedureCode,
                  xnap_direction2String(pdu.present - 1));
      ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_XNAP_XnAP_PDU, &pdu);
      return -1;
    }
    /* Calling the right handler */
    ret = (*xnap_messages_callback[pdu.choice.unsuccessfulOutcome->procedureCode][pdu.present - 1])
        (instance, assoc_id, stream, &pdu);
    break;

  default:
    XNAP_ERROR("[SCTP %d] Direction %d exceed expected\n",
               assoc_id, pdu.present);
    break;
  }

  ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_XNAP_XnAP_PDU, &pdu);
  return ret;
}

int
xnap_gNB_handle_xn_setup_request(instance_t instance,
                                 uint32_t assoc_id,
                                 uint32_t stream,
                                 XNAP_XnAP_PDU_t *pdu)
{

  XNAP_XnSetupRequest_t              *xnSetupRequest;
  XNAP_XnSetupRequest_IEs_t          *ie;
  //XNAP_ServedCells_NR_Item_t          *servedCellMember;

  xnap_gNB_instance_t                *instance_p;
  xnap_gNB_data_t                    *xnap_gNB_data;
  MessageDef                         *msg;
  uint32_t                           gNB_id = 0;

  DevAssert (pdu != NULL);
  xnSetupRequest = &pdu->choice.initiatingMessage->value.choice.XnSetupRequest;

  /*
   * We received a new valid XN Setup Request on a stream != 0.
   * * * * This should not happen -> reject gNB xn setup request.
   */

  if (stream != 0) {
    XNAP_ERROR("Received new xn setup request on stream != 0\n");
      /*
       * Send a xn setup failure with protocol cause unspecified
       */
    return xnap_gNB_generate_xn_setup_failure (instance,
                                               assoc_id,
                                               XNAP_Cause_PR_protocol,
                                               XNAP_CauseProtocol_unspecified,
                                               -1);
  }

  XNAP_DEBUG("Received a new XN setup request\n");

  XNAP_FIND_PROTOCOLIE_BY_ID(XNAP_XnSetupRequest_IEs_t, ie, xnSetupRequest,
                             XNAP_ProtocolIE_ID_id_GlobalNG_RAN_node_ID, true);
  if (ie == NULL ) {
    XNAP_ERROR("%s %d: ie is a NULL pointer \n",__FILE__,__LINE__);
    return -1;
  } else {
    if (ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.present==XNAP_GNB_ID_Choice_PR_gnb_ID) {
    //gNB ID = 28 bits
      uint8_t  *gNB_id_buf = ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID.buf;

      if (ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID.size != 28) {
      //TODO: handle case were size != 28 -> notify ? reject ?
      }
      gNB_id = (gNB_id_buf[0] << 20) + (gNB_id_buf[1] << 12) + (gNB_id_buf[2] << 4) + ((gNB_id_buf[3] & 0xf0) >> 4);
      XNAP_DEBUG("gNB id: %07x\n", gNB_id);
    } else {
    // Macro eNB = 20 bits
    //TODO if NSA setup
    }
  }

  XNAP_DEBUG("Adding gNB to the list of associated gNBs\n");

  if ((xnap_gNB_data = xnap_is_gNB_id_in_list (gNB_id)) == NULL) {
      /*
       * gNB has not been found in list of associated gNB,
       * * * * Add it to the tail of list and initialize data
       */
    if ((xnap_gNB_data = xnap_is_gNB_assoc_id_in_list (assoc_id)) == NULL) {
      /*
       * ??
       */
      return -1;
    } else {
      xnap_gNB_data->state = XNAP_GNB_STATE_RESETTING;
      xnap_gNB_data->gNB_id = gNB_id;
    }
  } else {
    xnap_gNB_data->state = XNAP_GNB_STATE_RESETTING;
    /*
     * gNB has been found in list, consider the xn setup request as a reset connection,
     * * * * reseting any previous UE state if sctp association is != than the previous one
     */
    if (xnap_gNB_data->assoc_id != assoc_id) {
      /*
       * ??: Send an overload cause...
       */
      XNAP_ERROR("Rejecting xn setup request as gNB id %d is already associated to an active sctp association" "Previous known: %d, new one: %d\n", gNB_id, xnap_gNB_data->assoc_id, assoc_id);

      xnap_gNB_generate_xn_setup_failure (instance,
                                          assoc_id,
                                          XNAP_Cause_PR_protocol,
                                          XNAP_CauseProtocol_unspecified,
                                          -1);
      return -1;
    }
    /*
     * TODO: call the reset procedure
     */
  }

  /* Set proper pci */
 /* XNAP_FIND_PROTOCOLIE_BY_ID(XNAP_XnSetupRequest_IEs_t, ie, xnSetupRequest,
                             XNAP_ProtocolIE_ID_id_ServedCells, true);
  if (ie == NULL ) {
    XNAP_ERROR("%s %d: ie is a NULL pointer \n",__FILE__,__LINE__);
    return -1;
  }

  msg = itti_alloc_new_message(TASK_XNAP, 0, XNAP_SETUP_REQ);

  XNAP_SETUP_REQ(msg).num_cc = ie->value.choice.ServedCells.list.count;

  if (ie->value.choice.ServedCells.list.count > 0) {
    xnap_gNB_data->num_cc = ie->value.choice.ServedCells.list.count;
    for (int i=0; i<ie->value.choice.ServedCells.list.count;i++) {
      servedCellMember = (ServedCells__Member *)ie->value.choice.ServedCells.list.array[i];
      xnap_gNB_data->Nid_cell[i] = servedCellMember->servedCellInfo.pCI;
      XNAP_SETUP_REQ(msg).Nid_cell[i] = xnap_gNB_data->Nid_cell[i];
    }
  }
*/
  instance_p = xnap_gNB_get_instance(instance);
  DevAssert(instance_p != NULL);

 // itti_send_msg_to_task(TASK_RRC_GNB, instance_p->instance, msg);

  return xnap_gNB_generate_xn_setup_response(instance_p, xnap_gNB_data);
  //printf("Received Xn setup request");
}

int
xnap_gNB_handle_xn_reset_response(instance_t instance,
                                  uint32_t assoc_id,
                                  uint32_t stream,
                                  XNAP_XnAP_PDU_t *pdu)
{

   return (0);
}



static
int xnap_gNB_handle_xn_setup_response(instance_t instance,
                                      uint32_t assoc_id,
                                      uint32_t stream,
                                      XNAP_XnAP_PDU_t *pdu)
{

  XNAP_XnSetupResponse_t              *xnSetupResponse;
  XNAP_XnSetupResponse_IEs_t          *ie;
  //ServedCells__Member                 *servedCellMember;

  xnap_gNB_instance_t                 *instance_p;
  xnap_gNB_data_t                     *xnap_gNB_data;
  MessageDef                          *msg;
  uint32_t                            gNB_id = 0;

  DevAssert (pdu != NULL);
  xnSetupResponse = &pdu->choice.successfulOutcome->value.choice.XnSetupResponse;

  /*
   * We received a new valid XN Setup Response on a stream != 0.
   * * * * This should not happen -> reject gNB xn setup response.
   */

  if (stream != 0) {
    XNAP_ERROR("Received new xn setup response on stream != 0\n");
  }

  if ((xnap_gNB_data = xnap_get_gNB(NULL, assoc_id, 0)) == NULL) {
    XNAP_ERROR("[SCTP %d] Received XN setup response for non existing "
               "gNB context\n", assoc_id);
    return -1;
  }

  if((xnap_gNB_data->state == XNAP_GNB_STATE_CONNECTED) ||
     (xnap_gNB_data->state == XNAP_GNB_STATE_READY))
  
  {
    XNAP_ERROR("Received Unexpexted XN Setup Response Message\n");
    return -1;
  }

  XNAP_DEBUG("Received a new XN setup response\n");

  XNAP_FIND_PROTOCOLIE_BY_ID(XNAP_XnSetupResponse_IEs_t, ie, xnSetupResponse,
                             XNAP_ProtocolIE_ID_id_GlobalNG_RAN_node_ID, true);

  if (ie == NULL ) {
    XNAP_ERROR("%s %d: ie is a NULL pointer \n",__FILE__,__LINE__);
    return -1;
  } 
  /*if (ie->value.choice.GlobalNG_RANNode_ID.gNB_ID.present == XNAP_GNB_ID_PR_home_gNB_ID) {
    // Home gNB ID = 28 bits
    uint8_t  *gNB_id_buf = ie->value.choice.GlobalNG_RANNode_ID.gNB_ID.choice.home_gNB_ID.buf;

    if (ie->value.choice.GlobalNG_RANNode_ID.gNB_ID.choice.macro_gNB_ID.size != 28) {
      //TODO: handle case were size != 28 -> notify ? reject ?
    }*/
    uint8_t *gNB_id_buf = ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID.buf;
    gNB_id = (gNB_id_buf[0] << 20) + (gNB_id_buf[1] << 12) + (gNB_id_buf[2] << 4) + ((gNB_id_buf[3] & 0xf0) >> 4);
    XNAP_DEBUG("Home gNB id: %07x\n", gNB_id);
  /*} else {
    // Macro gNB = 20 bits
    uint8_t *gNB_id_buf = ie->value.choice.GlobalNG_RANNode_ID.gNB_ID.choice.macro_gNB_ID.buf;

    if (ie->value.choice.GlobalNG_RANNode_ID.gNB_ID.choice.macro_gNB_ID.size != 20) {
      //TODO: handle case were size != 20 -> notify ? reject ?
    }

    gNB_id = (gNB_id_buf[0] << 12) + (gNB_id_buf[1] << 4) + ((gNB_id_buf[2] & 0xf0) >> 4);
    XNAP_DEBUG("macro gNB id: %05x\n", gNB_id);
  }*/

  XNAP_DEBUG("Adding gNB to the list of associated gNBs\n");

  if ((xnap_gNB_data = xnap_is_gNB_id_in_list (gNB_id)) == NULL) {
      /*
       * gNB has not been found in list of associated gNB,
       * * * * Add it to the tail of list and initialize data
       */
    if ((xnap_gNB_data = xnap_is_gNB_assoc_id_in_list (assoc_id)) == NULL) {
      /*
       * ??: Send an overload cause...
       */
      return -1;
    } else {
      xnap_gNB_data->state = XNAP_GNB_STATE_RESETTING;
      xnap_gNB_data->gNB_id = gNB_id;
    }
  } else {
    xnap_gNB_data->state = XNAP_GNB_STATE_RESETTING;
    /*
     * TODO: call the reset procedure
     */
  }

  /* Set proper pci */
  /*XNAP_FIND_PROTOCOLIE_BY_ID(XNAP_XnSetupResponse_IEs_t, ie, xnSetupResponse,
                             XNAP_ProtocolIE_ID_id_ServedCells, true);
  if (ie == NULL ) {
    XNAP_ERROR("%s %d: ie is a NULL pointer \n",__FILE__,__LINE__);
    return -1;
  }*/

  msg = itti_alloc_new_message(TASK_XNAP, 0, XNAP_SETUP_RESP);

  //XNAP_SETUP_RESP(msg).num_cc = ie->value.choice.ServedCells.list.count;

  /*if (ie->value.choice.ServedCells.list.count > 0) {
    xnap_gNB_data->num_cc = ie->value.choice.ServedCells.list.count;
    for (int i=0; i<ie->value.choice.ServedCells.list.count;i++) {
      servedCellMember = (ServedCells__Member *)ie->value.choice.ServedCells.list.array[i];
      xnap_gNB_data->Nid_cell[i] = servedCellMember->servedCellInfo.pCI;
      XNAP_SETUP_RESP(msg).Nid_cell[i] = xnap_gNB_data->Nid_cell[i];
    }
  }*/

  /* Optionaly set the target gNB name */

  /* The association is now ready as source and target gNBs know parameters of each other.
   * Mark the association as connected.
   */
  xnap_gNB_data->state = XNAP_GNB_STATE_READY;

  instance_p = xnap_gNB_get_instance(instance);
  DevAssert(instance_p != NULL);

  instance_p->xn_target_gnb_associated_nb ++;
  xnap_handle_xn_setup_message(instance_p, xnap_gNB_data, 0);

  itti_send_msg_to_task(TASK_RRC_GNB, instance_p->instance, msg);

  return 0;
}

static
int xnap_gNB_handle_xn_setup_failure(instance_t instance,
                                     uint32_t assoc_id,
                                     uint32_t stream,
                                     XNAP_XnAP_PDU_t *pdu)
{

  XNAP_XnSetupFailure_t              *xnSetupFailure;
  XNAP_XnSetupFailure_IEs_t          *ie;

  xnap_gNB_instance_t                *instance_p;
  xnap_gNB_data_t                    *xnap_gNB_data;

  DevAssert(pdu != NULL);

  xnSetupFailure = &pdu->choice.unsuccessfulOutcome->value.choice.XnSetupFailure;

  /*
   * We received a new valid XN Setup Failure on a stream != 0.
   * * * * This should not happen -> reject gNB xn setup failure.
  */

  if (stream != 0) {
    XNAP_WARN("[SCTP %d] Received xn setup failure on stream != 0 (%d)\n",
    assoc_id, stream);
  }

  if ((xnap_gNB_data = xnap_get_gNB (NULL, assoc_id, 0)) == NULL) {
    XNAP_ERROR("[SCTP %d] Received XN setup failure for non existing "
    "gNB context\n", assoc_id);
    return -1;
  }

  if((xnap_gNB_data->state == XNAP_GNB_STATE_CONNECTED) ||
     (xnap_gNB_data->state == XNAP_GNB_STATE_READY))
  
  {
    XNAP_ERROR("Received Unexpexted XN Setup Failure Message\n");
    return -1;
  }

  XNAP_DEBUG("Received a new XN setup failure\n");

  XNAP_FIND_PROTOCOLIE_BY_ID(XNAP_XnSetupFailure_IEs_t, ie, xnSetupFailure,
                             XNAP_ProtocolIE_ID_id_Cause, true);

  if (ie == NULL ) {
    XNAP_ERROR("%s %d: ie is a NULL pointer \n",__FILE__,__LINE__);
    return -1;
  } 
  // need a FSM to handle all cases
  if ((ie->value.choice.Cause.present == XNAP_Cause_PR_misc) &&
      (ie->value.choice.Cause.choice.misc == XNAP_CauseMisc_unspecified)) {
    XNAP_ERROR("Received XN setup failure for gNB ... gNB is not ready\n");
    exit(1);
  } else {
    XNAP_ERROR("Received xn setup failure for gNB... please check your parameters\n");
    exit(1);
  }

  xnap_gNB_data->state = XNAP_GNB_STATE_WAITING;

  instance_p = xnap_gNB_get_instance(instance);
  DevAssert(instance_p != NULL);

  xnap_handle_xn_setup_message(instance_p, xnap_gNB_data, 0);

  return 0;
}

