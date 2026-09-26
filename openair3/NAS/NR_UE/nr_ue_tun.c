/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "nr_ue_tun.h"
#include "common/utils/assertions.h"
#include "common/utils/tuntap_if.h"
#include "common/utils/LOG/log.h"
#include <arpa/inet.h>
#include <stdio.h>
#include <string.h>

void nr_ue_tun_store_qfi(nas_ue_pdu_tun_t *t, uint8_t qfi)
{
  DevAssert(t != NULL);
  t->qfi = qfi;
}

bool nr_ue_tun_is_ready(const nas_ue_pdu_tun_t *t)
{
  DevAssert(t != NULL);
  return t->sock >= 0 && !t->ip_config_failed;
}

static bool ip_configuration_succeeded(nas_ue_pdu_tun_t *t)
{
  t->ip_config_failed = false;
  return true;
}

static bool normalize_ip_address(int family, const char *address, char *normalized, size_t normalized_size)
{
  unsigned char binary[sizeof(struct in6_addr)];
  if (!address || inet_pton(family, address, binary) != 1 || !inet_ntop(family, binary, normalized, normalized_size)) {
    LOG_E(NAS, "Invalid IPv%d address in PDU session configuration: %s\n", family == AF_INET ? 4 : 6, address ? address : "(null)");
    return false;
  }
  return true;
}

static bool configure_initial_ip(nas_ue_pdu_tun_t *t, const char *ipv4, const char *ipv6, int ue_id, int pdu_session_id)
{
  /* Configure IPv4 and its policy selectors before adding IPv6. If policy
   * setup fails, retry does not repeat an IPv6 add that could return EEXIST. */
  if (ipv4 && !tun_config(t->ifname, ipv4, NULL))
    return false;
  if (ipv4 && !setup_ue_ipv4_route_checked(t->ifname, ue_id, pdu_session_id, ipv4))
    return false;
  if (ipv6 && !tun_config(t->ifname, NULL, ipv6))
    return false;

  if (ipv4) {
    snprintf(t->ipv4, sizeof(t->ipv4), "%s", ipv4);
    t->ipv4_configured = true;
    t->ipv4_route_pdu_id = pdu_session_id;
  }
  if (ipv6) {
    snprintf(t->ipv6, sizeof(t->ipv6), "%s", ipv6);
    t->ipv6_configured = true;
  }
  return true;
}

static bool repair_ipv4_route(nas_ue_pdu_tun_t *t, int ue_id, int pdu_session_id)
{
  if (!tun_config(t->ifname, t->ipv4, NULL))
    return false;
  if (!replace_ue_ipv4_route(t->ifname, ue_id, pdu_session_id, t->ipv4_failed_refresh, t->ipv4))
    return false;
  t->ipv4_route_repair_pending = false;
  t->ipv4_failed_refresh[0] = '\0';
  return true;
}

bool nr_ue_tun_create_ip_if(nas_ue_pdu_tun_t *t, const char *ipv4, const char *ipv6, int ue_id, int pdu_session_id)
{
  DevAssert(t != NULL);
  t->ip_config_failed = true;
  char normalized_ipv4[INET_ADDRSTRLEN] = {0};
  char normalized_ipv6[INET6_ADDRSTRLEN] = {0};
  if ((ipv4 && !normalize_ip_address(AF_INET, ipv4, normalized_ipv4, sizeof(normalized_ipv4)))
      || (ipv6 && !normalize_ip_address(AF_INET6, ipv6, normalized_ipv6, sizeof(normalized_ipv6))))
    return false;
  if (!ipv4 && !ipv6) {
    LOG_E(NAS, "PDU session TUN configuration has no IP address\n");
    return false;
  }

  if (t->sock < 0) {
    /* A new TUN lifecycle cannot retain successful-address state from a
     * prior fd. Configuration failures keep the fd and state for retry. */
    t->ipv4_configured = false;
    t->ipv4[0] = '\0';
    t->ipv4_route_pdu_id = -1;
    t->ipv4_route_repair_pending = false;
    t->ipv4_failed_refresh[0] = '\0';
    t->ipv6_configured = false;
    t->ipv6[0] = '\0';
    tuntap_generate_ue_ifname(t->ifname, IFF_TUN, ue_id, pdu_session_id);
    t->sock = tuntap_alloc(IFF_TUN, t->ifname);
    if (t->sock < 0)
      return false;
  }

  if (!t->ipv4_configured && !t->ipv6_configured) {
    if (!configure_initial_ip(t, ipv4 ? normalized_ipv4 : NULL, ipv6 ? normalized_ipv6 : NULL, ue_id, pdu_session_id))
      return false;
    return ip_configuration_succeeded(t);
  }

  if ((ipv4 != NULL) != t->ipv4_configured || (ipv6 != NULL) != t->ipv6_configured) {
    LOG_E(NAS, "PDU session TUN address family changed on %s; refresh is unsupported\n", t->ifname);
    return false;
  }
  if (ipv6 && strcmp(t->ipv6, normalized_ipv6) != 0) {
    LOG_E(NAS, "PDU session TUN IPv6 address changed on %s; refresh is unsupported\n", t->ifname);
    return false;
  }
  if (ipv4 && t->ipv4_route_pdu_id != pdu_session_id) {
    LOG_E(NAS, "PDU session TUN IPv4 selector changed on %s; refresh is unsupported\n", t->ifname);
    return false;
  }
  if (ipv4 && t->ipv4_route_repair_pending && !repair_ipv4_route(t, ue_id, pdu_session_id))
    return false;
  if (!ipv4 || strcmp(t->ipv4, normalized_ipv4) == 0)
    return ip_configuration_succeeded(t);

  /* tun_config may apply the address before a later netmask or flag operation
   * fails. Mark the last successful configuration dirty before that mutation. */
  t->ipv4_route_repair_pending = true;
  snprintf(t->ipv4_failed_refresh, sizeof(t->ipv4_failed_refresh), "%s", normalized_ipv4);
  if (!tun_config(t->ifname, normalized_ipv4, NULL))
    return false;
  if (!replace_ue_ipv4_route(t->ifname, ue_id, pdu_session_id, t->ipv4, normalized_ipv4)) {
    if (!tun_config(t->ifname, t->ipv4, NULL))
      LOG_E(NAS, "Could not restore prior IPv4 address on %s after route refresh failure\n", t->ifname);
    return false;
  }

  t->ipv4_route_repair_pending = false;
  t->ipv4_failed_refresh[0] = '\0';
  snprintf(t->ipv4, sizeof(t->ipv4), "%s", normalized_ipv4);
  return ip_configuration_succeeded(t);
}

void nr_ue_tun_create_eth_if(nas_ue_pdu_tun_t *t, int ue_id, int pdu_session_id)
{
  DevAssert(t != NULL);
  if (t->sock >= 0)
    return;

  tuntap_generate_ue_ifname(t->ifname, IFF_TAP, ue_id, pdu_session_id);
  t->sock = tuntap_alloc(IFF_TAP, t->ifname);
  tap_config(t->ifname);
}
