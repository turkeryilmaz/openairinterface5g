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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>


#include "TLVEncoder.h"
#include "TLVDecoder.h"
#include "fgs_service_request.h"
#include "FGSMobileIdentity.h"
#include "NasKeySetIdentifier.h"
#include "ServiceType.h"

int decode_fgs_service_request(fgs_service_request_msg_t *service_request, uint8_t *buffer, uint32_t len)
{
  uint32_t decoded = 0;
  int decoded_result = 0;

  LOG_FUNC_IN;
  LOG_TRACE(INFO, "EMM  - registration_request len = %d",
            len);

///* Decoding mandatory fields */
//if ((decoded_result = decode_5gs_registration_type(&service_request., 0, *(buffer + decoded)  & 0x0f, len - decoded)) < 0) {
//  //         return decoded_result;
//  LOG_FUNC_RETURN(decoded_result);
//}

  if ((decoded_result = decode_u8_nas_key_set_identifier(&service_request->naskeysetidentifier, 0, *(buffer + decoded) >> 4, len - decoded)) < 0) {
    //         return decoded_result;
    LOG_FUNC_RETURN(decoded_result);
  }

  decoded++;

//if ((decoded_result = decode_5gs_mobile_identity(&service_request->fgsmobileidentity, 0, buffer + decoded, len - decoded)) < 0) {
//  //         return decoded_result;
//  LOG_FUNC_RETURN(decoded_result);
//} else
//  decoded += decoded_result;


  // TODO, Decoding optional fields

  return decoded;
}

int encode_5gs_service_type(ServiceType serviceType)
{
  uint8_t bufferReturn;
  uint8_t *buffer = &bufferReturn;
  uint8_t encoded = 0;
  uint8_t iei = 0;
  *(buffer + encoded) = 0x00 | (iei & 0xf0) | (serviceType & 0xF);
  encoded++;
  return bufferReturn;
}

int encode_fgs_service_request(fgs_service_request_msg_t *service_request, uint8_t *buffer, uint32_t len)
{
  int encoded = 0;
  int encoded_rc = 0;
  // ngKSI + Service type
  *(buffer + encoded) = ((encode_u8_nas_key_set_identifier(&service_request->naskeysetidentifier) & 0x0f) << 4)
                        | (encode_5gs_service_type(service_request->serviceType) & 0x0f);
  encoded++;
  // 5G-S-TMSI
  // FGSMobileIdentity fgsmobileidentity = {0};
  // memset(&fgsmobileidentity, 0, sizeof(FGSMobileIdentity));
  // fgsmobileidentity.stmsi = service_request->fiveg_s_tmsi;

  // skip IEI and Length of 5GS mobile identity contents
  encoded = encoded + 2;
  encoded_rc = encode_stmsi_5gs_mobile_identity(&service_request->fiveg_s_tmsi, buffer + encoded);
  if (encoded_rc < 0) {
    printf("FAILED TO encode_stmsi_5gs_mobile_identity: encoded_rc %d\n", encoded_rc);
    fflush(stdout);
    return encoded_rc;
  }
  uint16_t tmp = htons(encoded + encoded_rc - 2);
  memcpy(buffer, &tmp, sizeof(tmp));
  encoded += encoded_rc;

// if ((encode_result = encode_5gs_mobile_identity(&fgsmobileidentity,
//                                                 0,
//                                                 buffer + encoded,
//                                                 len - encoded))
//     < 0) {
//       // Return in case of error
//   printf("FAILED TO encode_fgs_service_request: encode_5gs_mobile_identity: encode_result %d\n", encode_result);
//   fflush(stdout);
//   return encode_result;
// }
// else
//   encoded += encode_result;


  // TODO, Encoding optional fields
  return encoded;
}