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
#include <stdint.h>


#include "TLVEncoder.h"
#include "TLVDecoder.h"
#include "EapMessage.h"

int decode_eap_message(EapMessage *eapmessage, uint8_t iei, uint8_t *buffer, uint32_t len)
{
  int decoded = 0;
  uint16_t ielen = 0;
  int decode_result;

  if (iei > 0) {
    CHECK_IEI_DECODER(iei, *buffer);
    decoded++;
  }

  ielen = (uint16_t)(*(buffer + decoded) << 8);
  ielen |= (uint16_t)(*(buffer + decoded + 1));
  decoded += 2;
  CHECK_LENGTH_DECODER(len - decoded, ielen);

  if ((decode_result = decode_octet_string(&eapmessage->eapMsg, ielen, buffer + decoded, len - decoded)) < 0)
    return decode_result;
  else
    decoded += decode_result;

#if defined (NAS_DEBUG)
  dump_eap_message_xml(eapmessage, iei);
#endif
  return decoded;
}
int encode_eap_message(EapMessage *eapmessage, uint8_t iei, uint8_t *buffer, uint32_t len)
{
  uint16_t ielen = 0;
  uint8_t *lenPtr;
  uint32_t encoded = 0;
  int encode_result;
  /* Checking IEI and pointer */
  CHECK_PDU_POINTER_AND_LENGTH_ENCODER(buffer, EAP_MESSAGE_MINIMUM_LENGTH, len);
#if defined (NAS_DEBUG)
  dump_eap_message_xml(eapmessage, iei);
#endif

  if (iei > 0) {
    *buffer = iei;
    encoded++;
  }

  lenPtr = (buffer + encoded);
  encoded += 2;

  if ((encode_result = encode_octet_string(&eapmessage->eapMsg, buffer + encoded, len - encoded)) < 0)
    return encode_result;
  else
    encoded += encode_result;

  ielen = encoded - 2 - ((iei > 0) ? 1 : 0);
  *lenPtr = (uint8_t)((ielen >> 8) & 0xFF);
  *(lenPtr + 1) = (uint8_t)(ielen & 0xFF);

  return encoded;
}

void dump_eap_message_xml(EapMessage *eapmessage, uint8_t iei)
{
  printf("<EAP Message>\n");
  dump_octet_string_xml(&eapmessage->eapMsg);
  printf("</EAP Message>\n");
}
