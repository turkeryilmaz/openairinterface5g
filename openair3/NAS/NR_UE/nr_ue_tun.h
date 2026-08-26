/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef NR_UE_TUN_H
#define NR_UE_TUN_H

#include <net/if.h>

typedef struct nas_ue_pdu_tun_s {
  int sock; /* primary TUN fd (-1 = none) */
  char ifname[IFNAMSIZ];
  int qfi; /* -1 = unset */
} nas_ue_pdu_tun_t;

#endif /* NR_UE_TUN_H */
