/* SPDX-License-Identifier: LicenseRef-CSSL-1.0 */
#include "fgmm_registration_reject.h"
#include "common/platform_types.h"
#include "common/utils/eq_check.h"

/* TS 24.501 table 8.2.9.1.1, GPRS timer 2 TLVs (IEI, length=1, value). */
#define T3346_IEI 0x5f
#define T3502_IEI 0x16

int encode_fgs_registration_reject(byte_array_t *buffer, const fgs_registration_reject_msg_t *msg)
{
  const size_t required = 1 + 3 * (msg->t3346_present + msg->t3502_present);
  if (!buffer->buf || buffer->len < required || msg->unhandled_ies)
    return -1;
  int encoded = encode_fgs_nas_cause(buffer, &msg->cause);
  if (encoded < 0)
    return encoded;
  if (msg->t3346_present) {
    buffer->buf[encoded++] = T3346_IEI;
    buffer->buf[encoded++] = 1;
    buffer->buf[encoded++] = msg->t3346;
  }
  if (msg->t3502_present) {
    buffer->buf[encoded++] = T3502_IEI;
    buffer->buf[encoded++] = 1;
    buffer->buf[encoded++] = msg->t3502;
  }
  return encoded;
}

int decode_fgs_registration_reject(fgs_registration_reject_msg_t *msg, const byte_array_t *buffer)
{
  if (!buffer->buf || buffer->len < 1)
    return -1;
  fgs_registration_reject_msg_t result = {0};
  int decoded = decode_fgs_nas_cause(&result.cause, buffer);
  if (decoded < 0)
    return decoded;
  while ((size_t)decoded < buffer->len) {
    uint8_t iei = buffer->buf[decoded];
    if (iei != T3346_IEI && iei != T3502_IEI) {
      /* Unknown IEs have several wire formats. Do not guess a length and
       * accidentally interpret their payload as another timer. */
      result.unhandled_ies = true;
      break;
    }
    if (buffer->len - decoded < 3 || buffer->buf[decoded + 1] != 1) {
      /* Preserve mandatory-cause handling, as before optional IE support.
       * The caller must not infer usable retry restrictions from this body. */
      result.unhandled_ies = true;
      break;
    }
    bool *present = iei == T3346_IEI ? &result.t3346_present : &result.t3502_present;
    if (*present) {
      result.unhandled_ies = true; // Ambiguous duplicate: do not select a shorter wait.
      break;
    }
    *present = true;
    uint8_t *value = iei == T3346_IEI ? &result.t3346 : &result.t3502;
    *value = buffer->buf[decoded + 2];
    decoded += 3;
  }
  *msg = result;
  return decoded;
}

bool eq_registration_reject(const fgs_registration_reject_msg_t *a, const fgs_registration_reject_msg_t *b)
{
  _EQ_CHECK_INT(a->cause, b->cause);
  _EQ_CHECK_INT(a->t3346_present, b->t3346_present);
  _EQ_CHECK_INT(a->t3502_present, b->t3502_present);
  _EQ_CHECK_INT(a->unhandled_ies, b->unhandled_ies);
  if (a->t3346_present)
    _EQ_CHECK_INT(a->t3346, b->t3346);
  if (a->t3502_present)
    _EQ_CHECK_INT(a->t3502, b->t3502);
  return true;
}

void free_fgs_registration_reject(fgs_registration_reject_msg_t *msg)
{
  UNUSED(msg);
}
