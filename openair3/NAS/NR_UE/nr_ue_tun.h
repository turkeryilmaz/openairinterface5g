/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef NR_UE_TUN_H
#define NR_UE_TUN_H

#include <pthread.h>
#include <stdint.h>
#include <stdbool.h>
#include <net/if.h>
#include <arpa/inet.h>

typedef struct nas_ue_pdu_tun_s {
  int sock; /* primary TUN fd (-1 = none) */
  char ifname[IFNAMSIZ];
  int qfi; /* -1 = unset */
  bool ip_config_failed;
  bool ipv4_configured;
  char ipv4[INET_ADDRSTRLEN]; /* last successful IPv4 address */
  int ipv4_route_pdu_id; /* selector used for the last successful IPv4 route */
  bool ipv4_route_repair_pending;
  char ipv4_failed_refresh[INET_ADDRSTRLEN];
  bool ipv6_configured;
  char ipv6[INET6_ADDRSTRLEN]; /* last successful IPv6 address */
  /** TUN reader (idle or connected)
   * lifecycle managed via RRC */
  pthread_t reader_thread;
} nas_ue_pdu_tun_t;

bool nr_ue_tun_create_ip_if(nas_ue_pdu_tun_t *t, const char *ipv4, const char *ipv6, int ue_id, int pdu_session_id);
bool nr_ue_tun_is_ready(const nas_ue_pdu_tun_t *t);
void nr_ue_tun_create_eth_if(nas_ue_pdu_tun_t *t, int ue_id, int pdu_session_id);
void nr_ue_tun_store_qfi(nas_ue_pdu_tun_t *t, uint8_t qfi);

#endif /* NR_UE_TUN_H */
