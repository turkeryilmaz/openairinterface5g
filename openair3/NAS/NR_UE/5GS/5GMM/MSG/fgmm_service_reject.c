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

#include "fgmm_service_reject.h"
#include <string.h>
#include <arpa/inet.h> // For htons and ntohs
#include <stdlib.h> // For malloc and free
#include "common/utils/ds/byte_array.h"
#include "common/utils/utils.h"
#include "fgmm_lib.h"

#define SERVICE_REJECT_MIN_LEN 1 // 1 octet

/** @brief Encode Service Reject (8.2.18 of 3GPP TS 24.501) */
int encode_fgs_service_reject(byte_array_t *buffer, const fgs_service_reject_msg_t *msg)
{
  if (buffer->len < SERVICE_REJECT_MIN_LEN) {
    PRINT_NAS_ERROR("Failed to encode Service Reject: missing Cause IE!\n");
    return -1;
  }

  byte_array_t ba = *buffer; // Local copy of buffer
  int encoded = 0;

  // 5GMM cause (Mandatory)
  buffer->buf[encoded++] = msg->cause;
  UPDATE_BYTE_ARRAY(ba, 1);

  // PDU Session Status (Optional)
  if (msg->has_psi_status) {
    ENCODE_NAS_IE(ba, encode_pdu_session_ie(&ba, IEI_PDU_SESSION_STATUS, msg->psi_status), encoded);
  }

  // T3446 Timer (Optional)
  if (msg->t3446) {
    ENCODE_NAS_IE(ba, encode_gprs_timer_ie(&ba, IEI_T3446_VALUE, msg->t3446), encoded);
  }

  // T3448 Timer (Optional)
  if (msg->t3448) {
    ENCODE_NAS_IE(ba, encode_gprs_timer_ie(&ba, IEI_T3448_VALUE, msg->t3448), encoded);
  }

  // EAP message (Optional)
  if (msg->eap_msg) {
    ENCODE_NAS_IE(ba, encode_eap_msg_ie(&ba, msg->eap_msg), encoded);
  }

  return encoded;
}

/** @brief Decode Service Reject (8.2.18 of 3GPP TS 24.501) */
int decode_fgs_service_reject(fgs_service_reject_msg_t *msg, const byte_array_t *buffer)
{
  if (buffer->len < SERVICE_REJECT_MIN_LEN) {
    PRINT_NAS_ERROR("Nothing to decode: missing Cause IE!\n");
    return -1;
  }

  byte_array_t ba = *buffer; // Local copy of buffer to be used in the decoding sub-functions
  uint32_t decoded = 0;

  // Cause (Mandatory)
  msg->cause = buffer->buf[decoded++];
  UPDATE_BYTE_ARRAY(ba, 1);

  // Decode Optional IEs
  while (decoded < buffer->len) {
    uint8_t iei = buffer->buf[decoded++];
    UPDATE_BYTE_ARRAY(ba, 1); // Skip IEI

    switch (iei) {
      case IEI_PDU_SESSION_STATUS:
        msg->has_psi_status = true;
        DECODE_NAS_IE(ba, decode_pdu_session_ie(msg->psi_status, &ba), decoded);
        break;

      case IEI_T3448_VALUE:
        msg->t3448 = malloc_or_fail(sizeof(*msg->t3448));
        DECODE_NAS_IE(ba, decode_gprs_timer_ie(msg->t3448, (const byte_array_t *)&ba), decoded);
        break;

      case IEI_T3446_VALUE:
        msg->t3446 = malloc_or_fail(sizeof(*msg->t3446));
        DECODE_NAS_IE(ba, decode_gprs_timer_ie(msg->t3446, (const byte_array_t *)&ba), decoded);
        break;

      case IEI_EAPMSG:
        DECODE_NAS_IE(ba, decode_eap_msg_ie(msg->eap_msg, &ba), decoded);
        break;

      case IEI_CAG_INFO_LIST: {
        int cag_list_len = (buffer->buf[decoded] << 8) | buffer->buf[decoded + 1];
        int cag_len = cag_list_len + 2; // encoded content + 2 octets for the length IE
        decoded += cag_len;
        UPDATE_BYTE_ARRAY(ba, cag_len);
        PRINT_NAS_ERROR("CAG information list IE in NAS Service Reject is not handled\n");
      } break;

      default:
        PRINT_NAS_ERROR("Uknown or invalid NAS Service Reject IEI %d\n", iei);
        return -1;
    }
  }

  return decoded;
}

/** @brief Equality check for Service Reject (enc/dec) */
bool eq_service_reject(const fgs_service_reject_msg_t *a, const fgs_service_reject_msg_t *b)
{
  _NAS_EQ_CHECK_INT(a->cause, b->cause);
  _NAS_EQ_CHECK_INT(a->has_psi_status, b->has_psi_status);
  if (a->has_psi_status && b->has_psi_status) {
    for (int i = 0; i < MAX_NUM_PSI; i++)
      _NAS_EQ_CHECK_INT(a->psi_status[i], b->psi_status[i]);
  }
  if (a->t3448 != NULL || b->t3448 != NULL) {
    if (a->t3448 == NULL || b->t3448 == NULL) {
      PRINT_NAS_ERROR("t3448 equality check failed\n");
      return false;
    }
    if (!eq_gprs_timer(a->t3448, b->t3448))
      return false;
  }
  if (a->t3446 != NULL || b->t3446 != NULL) {
    if (a->t3446 == NULL || b->t3446 == NULL) {
      PRINT_NAS_ERROR("t3448 equality check failed\n");
      return false;
    }
    if (!eq_gprs_timer(a->t3446, b->t3446))
      return false;
  }
  return true;
}

void free_fgs_service_reject(fgs_service_reject_msg_t *msg)
{
  free(msg->t3446);
  free(msg->t3448);
}
