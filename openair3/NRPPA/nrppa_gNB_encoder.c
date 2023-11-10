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

/*! \file nrppa_gNB_encoder.c
 * \brief ngap pdu encode procedures for gNB
 * \author Adeel Malik
 * \email adeel.malik@eurecom.fr
 *\date 2023
 * \version 1.0
 * @ingroup _nrppa
 */


#include <stdio.h>
#include <string.h>
#include <stdint.h>

//#include "assertions.h"
//#include "conversions.h"
//#include "intertask_interface.h"

/* TODO */
#include "nrppa_common.h"
#include "nrppa_gNB_encoder.h"
/*TODO */

static inline int nrppa_gNB_encode_initiating(NRPPA_NRPPA_PDU_t *pdu, uint8_t **buffer, uint32_t *len)
{
  DevAssert(pdu != NULL);

  const NRPPA_ProcedureCode_t tmp[] = {NRPPA_ProcedureCode_id_positioningInformationExchange,  // Parent procedure for  PositioningInformationRequest, PositioningInformationResponse, and PositioningInformationFailure
                                       NRPPA_ProcedureCode_id_positioningInformationUpdate,
                                       NRPPA_ProcedureCode_id_positioningActivation};
                                      // TODO ad**l add remaining UPLINK type NRPPA Procedure codes with message type Initiating
                                      // For other procedures  check TABLE 8.1-2 and TABLE 8.1-1 of NRPPA TS38.455 v16

  int i;
  for (i = 0; i < sizeofArray(tmp); i++)
    if (pdu->choice.initiatingMessage->procedureCode == tmp[i])
      break;
  if (i == sizeofArray(tmp)) {
    NRPPA_DEBUG("Unknown procedure ID (%d) for initiating message\n", (int)pdu->choice.initiatingMessage->procedureCode);
    return -1;
  }

  //asn_encode_to_new_buffer_result_t res = asn_encode_to_new_buffer(NULL, ATS_ALIGNED_CANONICAL_PER, &asn_DEF_NRPPA_NRPPA_PDU, pdu);
  asn_encode_to_new_buffer_result_t res = asn_encode_to_new_buffer(NULL, ATS_ALIGNED_CANONICAL_PER, &asn_DEF_NRPPA_NRPPA_PDU, pdu);
  AssertFatal(res.result.encoded > 0, "failed to encode NRPPA msg\n");
  *buffer = res.buffer;
  *len = res.result.encoded;
  return 0;
}

static inline int nrppa_gNB_encode_successfull_outcome(NRPPA_NRPPA_PDU_t *pdu, uint8_t **buffer, uint32_t *len)
{

DevAssert(pdu != NULL);
  const NRPPA_ProcedureCode_t tmp[] = {NRPPA_ProcedureCode_id_positioningInformationExchange, // Parent procedure for PositioningInformationRequest, PositioningInformationResponse, and PositioningInformationFailure
                                       NRPPA_ProcedureCode_id_positioningActivation};
                                      // TODO ad**l add remaining UPLINK type NRPPA Procedure codes with message type successful
                                      // For other procedures  check TABLE 8.1-2 and TABLE 8.1-1 of NRPPA TS38.455 v16
  int i;
  for (i = 0; i < sizeofArray(tmp); i++)
    if (pdu->choice.successfulOutcome->procedureCode == tmp[i])
      break;
  if (i == sizeofArray(tmp)) {
    NRPPA_WARN("Unknown procedure ID (%ld) for successfull outcome message\n", pdu->choice.successfulOutcome->procedureCode);
    return -1;
  }
   // printf("TEST 1 nrppa encoder for PositioningInformationResponse \n");
   // xer_fprint(stdout, &asn_DEF_NRPPA_NRPPA_PDU, pdu); // test adeel
  asn_encode_to_new_buffer_result_t res = asn_encode_to_new_buffer(NULL, ATS_ALIGNED_CANONICAL_PER, &asn_DEF_NRPPA_NRPPA_PDU, pdu);
  AssertFatal(res.result.encoded > 0, "failed to encode NRPPA msg\n");
  *buffer = res.buffer;
  *len = res.result.encoded;
  return 0;
}



static inline int nrppa_gNB_encode_unsuccessfull_outcome(NRPPA_NRPPA_PDU_t *pdu, uint8_t **buffer, uint32_t *len)
{
  DevAssert(pdu != NULL);

  const NRPPA_ProcedureCode_t tmp[] = {NRPPA_ProcedureCode_id_positioningInformationExchange, // Parent procedure for PositioningInformationRequest, PositioningInformationResponse, and PositioningInformationFailure
                                       NRPPA_ProcedureCode_id_positioningActivation};
                                      // TODO ad**l add remaining UPLINK type NRPPA Procedure codes with message type unsuccessful
                                      // For other procedures  check TABLE 8.1-2 and TABLE 8.1-1 of NRPPA TS38.455 v16

  int i;
  for (i = 0; i < sizeofArray(tmp); i++)
    if (pdu->choice.unsuccessfulOutcome->procedureCode == tmp[i])
      break;
  if (i == sizeofArray(tmp)) {
    NRPPA_WARN("Unknown procedure ID (%ld) for unsuccessfull outcome message\n", pdu->choice.unsuccessfulOutcome->procedureCode);
    return -1;
  }

  asn_encode_to_new_buffer_result_t res = asn_encode_to_new_buffer(NULL, ATS_ALIGNED_CANONICAL_PER, &asn_DEF_NRPPA_NRPPA_PDU, pdu);
  AssertFatal(res.result.encoded > 0, "failed to encode NRPPA msg\n");
  *buffer = res.buffer;
  *len = res.result.encoded;
  return 0;
}

int nrppa_gNB_encode_pdu(NRPPA_NRPPA_PDU_t *pdu, uint8_t **buffer, uint32_t *len)
{
  int ret = -1;
  DevAssert(pdu != NULL);
  DevAssert(buffer != NULL);
  DevAssert(len != NULL);
  if (asn1_xer_print) {
    xer_fprint(stdout, &asn_DEF_NRPPA_NRPPA_PDU, (void *)pdu);
  }
//  xer_fprint(stdout, &asn_DEF_NRPPA_NRPPA_PDU, (void *)pdu); // test adeel
  switch (pdu->present) {
    case NRPPA_NRPPA_PDU_PR_initiatingMessage:
      ret = nrppa_gNB_encode_initiating(pdu, buffer, len);
      break;
    case NRPPA_NRPPA_PDU_PR_successfulOutcome:
      ret = nrppa_gNB_encode_successfull_outcome(pdu, buffer, len);
      break;
    case NRPPA_NRPPA_PDU_PR_unsuccessfulOutcome:
      ret = nrppa_gNB_encode_unsuccessfull_outcome(pdu, buffer, len);
      break;

    default:
      NRPPA_DEBUG("Unknown message outcome (%d) or not implemented", (int)pdu->present);
      return -1;
  }
  ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_NRPPA_NRPPA_PDU, pdu);
  return ret;
}
