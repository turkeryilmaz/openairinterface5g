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

/*! \file FGSServiceRequest.c

\brief service request procedures for gNB
\author
\email:
\date 2022
\version 0.1
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "nas_log.h"

#include "FGSServiceRequest.h"

int encode_fgs_service_request(fgs_service_request_msg *fgs_service_request, uint8_t *buffer, uint32_t len)
{
  int encoded = 0;
  int encode_result = 0;

  *(buffer + encoded) = ((encode_u8_service_type(&fgs_service_request->servicetype) & 0x0f) << 4) | (encode_u8_nas_key_set_identifier(&fgs_service_request->naskeysetidentifier) & 0x0f);
  encoded++;

  if ((encode_result =
         encode_5gs_mobile_identity(&fgs_service_request->fgsmobileidentity, 0, buffer +
                                    encoded, len - encoded)) < 0)        //Return in case of error
    return encode_result;
  else
    encoded += encode_result;

  if ((fgs_service_request->presencemask & FGS_SERVICE_REQUEST_NAS_MESSAGE_CONTAINER_PRESENT)
      == FGS_SERVICE_REQUEST_NAS_MESSAGE_CONTAINER_PRESENT) {
    if ((encode_result = encode_fgc_nas_message_container(&fgs_service_request->fgsnasmessagecontainer,
                         FGS_SERVICE_REQUEST_NAS_MESSAGE_CONTAINER_IEI, buffer + encoded, len -
                         encoded)) < 0)
      // Return in case of error
      return encode_result;
    else
      encoded += encode_result;
  }


  // TODO, Encoding optional fields
  return encoded;
}
