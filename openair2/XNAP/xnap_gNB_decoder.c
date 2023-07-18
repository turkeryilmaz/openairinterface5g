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

/*! \file xnap_gNB_decoder.c
 * \date July 2023
 * \version 1.0
 */

#include <stdio.h>

#include "assertions.h"
#include "intertask_interface.h"
#include "xnap_common.h"
#include "xnap_gNB_decoder.h"

static int xnap_gNB_decode_initiating_message(XNAP_XnAP_PDU_t *pdu)
{
  DevAssert(pdu != NULL);

  switch(pdu->choice.initiatingMessage->procedureCode) {

    case XNAP_ProcedureCode_id_xnSetup:
      //asn_encode_to_new_buffer(NULL, ATS_CANONICAL_XER, &asn_DEF_X2AP_X2AP_PDU, pdu);
      LOG_I(XNAP, "x2ap_eNB_decode_initiating_message!\n");
      break;
      
    default:
      LOG_E(XNAP, "Unknown procedure ID (%d) for initiating message\n",
                  (int)pdu->choice.initiatingMessage->procedureCode);
      AssertFatal( 0, "Unknown procedure ID (%d) for initiating message\n",
                   (int)pdu->choice.initiatingMessage->procedureCode);
      return -1;
  }

  return 0;
}

static int xnap_gNB_decode_successful_outcome(XNAP_XnAP_PDU_t *pdu)
{
  DevAssert(pdu != NULL);

  switch(pdu->choice.successfulOutcome->procedureCode) {
    case XNAP_ProcedureCode_id_xnSetup:
      //asn_encode_to_new_buffer(NULL, ATS_CANONICAL_XER, &asn_DEF_X2AP_X2AP_PDU, pdu);
      LOG_I(XNAP, "xnap_gNB_decode_successfuloutcome_message!\n");
      break;

    default:
      LOG_E(XNAP, "Unknown procedure ID (%d) for successfull outcome message\n",
                  (int)pdu->choice.successfulOutcome->procedureCode);
      return -1;
  }

  return 0;
}

static int xnap_gNB_decode_unsuccessful_outcome(XNAP_XnAP_PDU_t *pdu)
{
  DevAssert(pdu != NULL);

  switch(pdu->choice.unsuccessfulOutcome->procedureCode) {
    case XNAP_ProcedureCode_id_xnSetup:
      //asn_encode_to_new_buffer(NULL, ATS_CANONICAL_XER, &asn_DEF_X2AP_X2AP_PDU, pdu);
      LOG_I(XNAP, "xnap_gNB_decode_unsuccessfuloutcome_message!\n");
      break;

    default:
       LOG_E(XNAP, "Unknown procedure ID (%d) for unsuccessfull outcome message\n",
                  (int)pdu->choice.unsuccessfulOutcome->procedureCode);
      return -1;
  }

  return 0;
}

int xnap_gNB_decode_pdu(XNAP_XnAP_PDU_t *pdu, const uint8_t *const buffer, uint32_t length)
{
  asn_dec_rval_t dec_ret;

  DevAssert(buffer != NULL);

  dec_ret = aper_decode(NULL,
                        &asn_DEF_XNAP_XnAP_PDU,
                        (void **)&pdu,
                        buffer,
                        length,
                        0,
                        0);
  //can be removed later
  xer_fprint(stdout, &asn_DEF_XNAP_XnAP_PDU, pdu);
  

  if (dec_ret.code != RC_OK) {
    LOG_E(XNAP, "Failed to decode PDU\n");
    return -1;
  }

  switch(pdu->present) {
    case XNAP_XnAP_PDU_PR_initiatingMessage:
      return xnap_gNB_decode_initiating_message(pdu);

    case XNAP_XnAP_PDU_PR_successfulOutcome:
      return xnap_gNB_decode_successful_outcome(pdu);

    case XNAP_XnAP_PDU_PR_unsuccessfulOutcome:
      return xnap_gNB_decode_unsuccessful_outcome(pdu);

    default:
      LOG_D(XNAP, "Unknown presence (%d) or not implemented\n", (int)pdu->present);
      break;
  }


  return -1;
}
