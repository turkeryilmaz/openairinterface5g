/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef FGS_REGISTRATION_REJECT_H
#define FGS_REGISTRATION_REJECT_H

#include "fgmm_lib.h"
#include "common/utils/ds/byte_array.h"

typedef struct {
  cause_id_t cause;
  /* TS 24.501 table 8.2.9.1.1: retain the complete GPRS timer 2 octet.
   * Presence is separate: a zero-valued timer is not an absent timer. */
  bool t3346_present;
  bool t3502_present;
  uint8_t t3346;
  uint8_t t3502;
  /* Unknown, malformed or duplicate optional IE: mandatory cause remains valid,
   * but the caller must not infer absent/usable retry restrictions. */
  bool unhandled_ies;
} fgs_registration_reject_msg_t;

int decode_fgs_registration_reject(fgs_registration_reject_msg_t *msg, const byte_array_t *buffer);
int encode_fgs_registration_reject(byte_array_t *buffer, const fgs_registration_reject_msg_t *msg);
bool eq_registration_reject(const fgs_registration_reject_msg_t *a, const fgs_registration_reject_msg_t *b);
void free_fgs_registration_reject(fgs_registration_reject_msg_t *msg);

#endif /* FGS_REGISTRATION_REJECT_H */
