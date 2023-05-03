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

/*! \file nrppa_gNB.c
 * \brief NRPPA gNB task
 * \author  Adeel Maik
 * \date 2023
 * \email: adeel.malik@eurecom.fr
 * \version 1.0
 * @ingroup _nrppa
 */


#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <crypt.h>

#include "tree.h"
#include "queue.h"

#include "intertask_interface.h"

//#include "ngap_gNB_default_values.h"

//#include "ngap_common.h"

//#include "ngap_gNB_defs.h"
//#include "ngap_gNB.h"
//#include "ngap_gNB_encoder.h"
//#include "ngap_gNB_handlers.h"
//#include "ngap_gNB_nnsf.h"

//#include "ngap_gNB_nas_procedures.h"
//#include "ngap_gNB_management_procedures.h"
//#include "ngap_gNB_context_management_procedures.h"

//#include "ngap_gNB_itti_messaging.h"

//#include "ngap_gNB_ue_context.h" // test, to be removed
//#include "ngap_gNB_NRPPa_transport_procedures.h" //adeel nrppa

#include "assertions.h"
#include "conversions.h"


/* Start: Adeel New added For NRPPA */
#include "nrppa_gNB.h"
#include "nrppa_common.h"
#include "nrppa_gNB_management_procedures.h"
/* END: Adeel New added for NRPPA */


//Processing DownLINK UE ASSOCIATED NRPPA TRANSPORT
int nrppa_process_DownlinkUEAssociatedNRPPaTransport(instance_t instance, ngap_DownlinkUEAssociatedNRPPa_t *ngap_DownlinkUEAssociatedNRPPa_p){
/* TODO
NRPPA_NRPPA_PDU_t* nrppaPdu = &ngap_DownlinkUEAssociatedNRPPa_p->nrppa_pdu.buffer;
//IE: 9.3.1.1 Message Type
  message_type = &nrppaPdu->choice.initiatingMessage->procedureCode;*/


/*Refer to ngap_handle_message
int ngap_gNB_handle_message(uint32_t assoc_id, int32_t stream, const uint8_t *const data, const uint32_t data_length)
uint8_t *const data = &ngap_DownlinkUEAssociatedNRPPa_p->nrppa_pdu.buffer;
const uint32_t data_length= ngap_DownlinkUEAssociatedNRPPa_p->nrppa_pdu.length
{
  NRPPA_NRPPA_PDU_t pdu;
  int ret;
  DevAssert(data != NULL);
  memset(&pdu, 0, sizeof(pdu));

  if (nrppa_gNB_decode_pdu(&pdu, data, data_length) < 0) {
    NRPPA_ERROR("Failed to decode PDU\n");
    return -1;
  }

  /* Checking procedure Code and direction of message
  if (pdu.choice.initiatingMessage->procedureCode >= sizeof(nrppa_messages_callback) / (3 * sizeof(nrppa_message_decoded_callback)) || (pdu.present > NRPPA_NRPPA_PDU_PR_unsuccessfulOutcome)) {
    NRPPA_ERROR("[NGAP %d] Either procedureCode %ld or direction %d exceed expected\n", assoc_id, pdu.choice.initiatingMessage->procedureCode, pdu.present);
    ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_NRPPA_NRPPA_PDU, &pdu);
    return -1;
  }

  /* No handler present.
   * This can mean not implemented or no procedure for gNB (wrong direction).

  if (nrppa_messages_callback[pdu.choice.initiatingMessage->procedureCode][pdu.present - 1] == NULL) {
    NRPPA_ERROR("[NGAP %d] No handler for procedureCode %ld in %s\n", assoc_id, pdu.choice.initiatingMessage->procedureCode, nrppa_direction2String(pdu.present - 1));
    ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_NRPPA_NRPPA_PDU, &pdu);
    return -1;
  }

  /* Calling the right handler
  ret = (*nrppa_messages_callback[pdu.choice.initiatingMessage->procedureCode][pdu.present - 1])(assoc_id, stream, &pdu);
  ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_NRPPA_NRPPA_PDU, &pdu);
  return ret;
}
*/




}


//Processing DOWNLINK NON UE ASSOCIATED NRPPA TRANSPORT (9.2.9.4 of TS 38.413 Version 16.0.0.0 Release 16)
int nrppa_process_DownlinkNonUEAssociatedNRPPaTransport(instance_t instance, ngap_DownlinkNonUEAssociatedNRPPa_t *ngap_DownlinkNonUEAssociatedNRPPa_p){
/*TODO*/
}



void nrppa_gNB_init(void) {
  NRPPA_DEBUG("Starting NRPPA layer\n");
  //ngap_gNB_prepare_internal_data();
  itti_mark_task_ready(TASK_NRPPA);
}


void *nrppa_gNB_process_itti_msg(void *notUsed) {
  printf("Test 2 Adeel: NRPPA Waiting for message\n");
  MessageDef *received_msg = NULL;
  int         result;
  itti_receive_msg(TASK_NRPPA, &received_msg);
  if (received_msg) {
    instance_t instance = ITTI_MSG_DESTINATION_INSTANCE(received_msg);
    LOG_D(NRPPA, "Received message %s\n", ITTI_MSG_NAME(received_msg));
    switch (ITTI_MSG_ID(received_msg)) {
      case TERMINATE_MESSAGE:
        NRPPA_WARN(" *** Exiting NRPPA thread\n"); // to be implemented // ad**l
        itti_exit_task();
        break;


      case NGAP_DOWNLINKUEASSOCIATEDNRPPA:
       nrppa_process_DownlinkUEAssociatedNRPPaTransport(instance, &NGAP_DOWNLINKUEASSOCIATEDNRPPA(received_msg));   // adeel changes NRPPA
        break;

      case NGAP_DOWNLINKNONUEASSOCIATEDNRPPA:
        nrppa_process_DownlinkUEAssociatedNRPPaTransport(instance, &NGAP_DOWNLINKNONUEASSOCIATEDNRPPA(received_msg));   // adeel changes NRPPA
        break;

      default:
        NRPPA_ERROR("Received unhandled message: %d:%s\n", ITTI_MSG_ID(received_msg), ITTI_MSG_NAME(received_msg));
        break;
    }

    result = itti_free(ITTI_MSG_ORIGIN_ID(received_msg), received_msg);
    AssertFatal(result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);
  }
  return NULL;
}




void *nrppa_gNB_task(void *arg) {
printf("Test 1 Adeel: NRPPA Waiting for message\n");
  nrppa_gNB_init();

  while (1) {
    (void) nrppa_gNB_process_itti_msg(NULL);
  }

  return NULL;
}



