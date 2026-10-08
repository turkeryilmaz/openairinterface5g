/* SPDX-License-Identifier: LicenseRef-CSSL-1.0 */
#ifndef NTN_ASSISTANCE_H
#define NTN_ASSISTANCE_H

#include <stddef.h>
#include <stdint.h>

#define NTN_ASSISTANCE_VERSION 1
#define NTN_ASSISTANCE_MAX_DATAGRAM 1200

typedef enum { NTN_ASSISTANCE_HELLO, NTN_ASSISTANCE_TIME, NTN_ASSISTANCE_UPDATE, NTN_ASSISTANCE_STATUS } ntn_assistance_kind_t;

typedef struct {
  char plmn[7]; /* MCC followed by the two or three MNC digits, including zeros. */
  uint64_t nci; /* 36-bit NR cell identity, not physical cell ID. */
} ntn_assistance_cell_t;

typedef struct {
  uint64_t generation;
  uint64_t subframe; /* Extended 1 ms downlink subframe; independent of numerology. */
} ntn_assistance_epoch_t;

typedef struct {
  ntn_assistance_epoch_t epoch;
  /* TS 38.331 v17.3.0 EphemerisInfo/NTN-Config quantized integers. */
  int32_t position[3]; /* 1.3 m, Earth-centred Earth-fixed. */
  int32_t velocity[3]; /* 0.06 m/s, derivative in that rotating ECEF frame. */
  int32_t ta_common; /* 0.004072 us */
  int32_t ta_drift; /* 0.0002 us/s */
  int32_t ta_drift_variant; /* 0.00002 us/s^2; unsigned field, not arbitrary acceleration. */
  unsigned int validity_index; /* ASN.1 ntn-UlSyncValidityDuration-r17 enumeration. */
} ntn_assistance_state_t;

typedef struct {
  ntn_assistance_kind_t kind;
  uint64_t request;
  ntn_assistance_cell_t cell;
  uint64_t session;
  uint64_t sequence;
  ntn_assistance_state_t state;
} ntn_assistance_request_t;

/* Control-worker only: strict JSON decode and full validation before replacing
 * out. Returns NULL on success, otherwise a static diagnostic code. No state or
 * output is changed on failure. UINT64 fields use canonical decimal strings. */
const char *ntn_assistance_parse(const void *data, size_t length, ntn_assistance_request_t *out);

unsigned int ntn_assistance_validity_seconds(unsigned int index);

#endif
