#include <common/utils/assertions.h>
#include <common/utils/LOG/log.h>
#include "o1config.h"

#define simOpt PARAMFLAG_NOFREE|PARAMFLAG_CMDLINE_NOPREFIXENABLED
#define simOptMandatory PARAMFLAG_NOFREE|PARAMFLAG_CMDLINE_NOPREFIXENABLED|PARAMFLAG_MANDATORY

int o1_readconfig(o1_config_t* config) {

    paramdef_t o1config_params[] = {
    {"netconfNodeId",                "<ip address to connect to>\n",     simOpt,  strptr:&config->netconf_node_id,                defstrval:"gnb-test01",           TYPE_STRING,    0 },
    {"netconfUsername",              "<ip address to connect to>\n",     simOpt,  strptr:&config->netconf_username,                defstrval:"netconf",           TYPE_STRING,    0 },
    {"netconfPassword",              "<ip address to connect to>\n",     simOpt,  strptr:&config->netconf_password,                defstrval:"netconf!",           TYPE_STRING,    0 },
    {"netconfPort",                  "<port to connect to>\n",           simOpt,  u16ptr:&(config->netconf_port),    defuintval:830,                 TYPE_UINT16,    0 },
    {"netconfHost",                  "<ip address to connect to>\n",     simOpt,  strptr:&config->netconf_host,                defstrval:"0.0.0.0",           TYPE_STRING,    0 },
    {"vesUrl",                       "<ip address to connect to>\n",     simOpt,  strptr:&config->ves_url,                defstrval:"",           TYPE_STRING,    0 },
    {"vesNfVendorName",              "<ip address to connect to>\n",     simOpt,  strptr:&config->ves_nf_vendor_name,                defstrval:"OAI",           TYPE_STRING,    0 },
    {"vesNfNameingCode",             "<ip address to connect to>\n",     simOpt,  strptr:&config->ves_nf_naming_code,                defstrval:"gNodeB",           TYPE_STRING,   0 },
    {"vesOamIpv4",                   "<ip address to connect to>\n",     simOpt,  strptr:&config->ves_oam_ipv4,                defstrval:"",           TYPE_STRING,    0 },
    {"vesOamIpv6",                   "<ip address to connect to>\n",     simOpt,  strptr:&config->ves_oam_ipv6,                defstrval:"",           TYPE_STRING,   0 },
    {"vesFtpServerPort",             "<port to connect to>\n",           simOpt,  u16ptr:&(config->ves_ftp_server_port),    defuintval:21,                 TYPE_UINT16,    0 },
    {"vesFtpServerListenAddress",    "<ip address to connect to>\n",     simOpt,  strptr:&config->ves_ftp_listen_addr,                defstrval:"0.0.0.0",           TYPE_STRING,    0 }
    
  };
    
  
  int ret = config_get( o1config_params,sizeof(o1config_params)/sizeof(paramdef_t),"o1Config");
 // AssertFatal(ret >= 0, "configuration couldn't be performed");
 return ret;
}