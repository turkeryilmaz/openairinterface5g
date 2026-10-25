/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef NR_MAC_RRC_TYPES_H_
#define NR_MAC_RRC_TYPES_H_

#include <stdint.h>

typedef struct {
  uint16_t cell_id;
  uint8_t additional_bits;
  uint8_t ssb_index;
  uint8_t ssb_length;
  uint16_t ssb_start_subcarrier;
} nr_ue_mib_metadata_t;

typedef enum {
  NR_MAC_RA_START_SETUP,
  NR_MAC_RA_START_T300,
  NR_MAC_RA_START_REESTABLISHMENT,
} nr_mac_ra_start_cause_t;

#endif /* NR_MAC_RRC_TYPES_H_ */
