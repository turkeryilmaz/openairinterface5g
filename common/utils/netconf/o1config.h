#pragma once

#include <common/config/config_userapi.h>
#include "common_lib.h"


typedef struct {
  char *netconf_node_id;
  char *netconf_username;
  char *netconf_password;
  uint16_t netconf_port;
  char *netconf_host;
  char * ves_url;
  char *ves_nf_vendor_name;
  char *ves_nf_naming_code;
  char *ves_oam_ipv4;
  char *ves_oam_ipv6;
  uint16_t ves_ftp_server_port;
  char *ves_ftp_listen_addr;

} o1_config_t;


int o1_readconfig(o1_config_t* config);