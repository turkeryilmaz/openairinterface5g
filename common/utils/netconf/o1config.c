#include "o1config.h"
#include "netconf_log.h"
#include <common/utils/assertions.h>
#include <common/utils/LOG/log.h>

#define simOpt PARAMFLAG_NOFREE|PARAMFLAG_CMDLINE_NOPREFIXENABLED
#define simOptMandatory PARAMFLAG_NOFREE|PARAMFLAG_CMDLINE_NOPREFIXENABLED|PARAMFLAG_MANDATORY

static int o1_validate_config(o1_config_t* config);

int o1_config_read(o1_config_t* config) {
    memset(config, 0, sizeof(o1_config_t));

    paramdef_t o1config_params[] = {
        {"nfNodeId",                     "nodeId for oam for ves mount package\n",               simOpt,  strptr:&config->netconf_node_id,            defstrval:"gnb-test01",    TYPE_STRING,    0 },
        {"netconfUsername",              "username for netconf interface\n",                     simOpt,  strptr:&config->netconf_username,           defstrval:"netconf",       TYPE_STRING,    0 },
        {"netconfPassword",              "password for netconf interface\n",                     simOpt,  strptr:&config->netconf_password,           defstrval:"netconf!",      TYPE_STRING,    0 },
        {"netconfPort",                  "port on host device for netconf interface\n",          simOpt,  u16ptr:&(config->netconf_port),             defuintval:830,            TYPE_UINT16,    0 },
        {"netconfHost",                  "ip address on host device for netconf interface\n",    simOpt,  strptr:&config->netconf_host,               defstrval:"0.0.0.0",       TYPE_STRING,    0 },
        {"vesUrl",                       "ves endpoint url to send notifications to\n",          simOpt,  strptr:&config->ves_url,                    defstrval:"",              TYPE_STRING,    0 },
        {"vesBasicAuthUsername",         "ves endpoint username to authorize\n",                 simOpt,  strptr:&config->ves_basicauth_username,     defstrval:"",              TYPE_STRING,    0 },
        {"vesBasicAuthPassword",         "ves endpoint password to authorize\n",                 simOpt,  strptr:&config->ves_basicauth_password,     defstrval:"",              TYPE_STRING,    0 },
        {"vesNfVendorName",              "vendor name information for ves packages\n",           simOpt,  strptr:&config->ves_nf_vendor_name,         defstrval:"OAI",           TYPE_STRING,    0 },
        {"vesNfNamingCode",              "naming code information for ves packages\n",           simOpt,  strptr:&config->ves_nf_naming_code,         defstrval:"gNodeB",        TYPE_STRING,    0 },
        {"vesOamIpv4",                   "oam ipv4 information for ves mount package\n",         simOpt,  strptr:&config->ves_oam_ipv4,               defstrval:"",              TYPE_STRING,    0 },
        {"vesOamIpv6",                   "oam ipv6 information for ves mount package\n",         simOpt,  strptr:&config->ves_oam_ipv6,               defstrval:"",              TYPE_STRING,    0 },
        {"vesFtpServerPort",             "ftp server port to serve pm data\n",                   simOpt,  u16ptr:&(config->ves_ftp_server_port),      defuintval:21,             TYPE_UINT16,    0 },
        {"vesFtpServerListenAddress",    "ftp listen address to serve pm data\n",                simOpt,  strptr:&config->ves_ftp_listen_addr,        defstrval:"0.0.0.0",       TYPE_STRING,    0 },
        {"demoAlarmingInterval",         "ves alarming interval in seconds for demoing alarms\n",simOpt,  u16ptr:&(config->demo_alarming_interval),   defuintval:0,              TYPE_UINT16,    0 }
    };
    
#if defined(USE_TEST_CONFIG)
    (void)o1config_params;  //unused
    config->netconf_node_id = strdup("gnb-test01");
    config->netconf_username = strdup("netconf");
    config->netconf_password = strdup("netconf!");
    config->netconf_port = 830;
    config->netconf_host = strdup("192.168.2.3");
    config->ves_url = strdup("https://192.168.5.207:8443/eventListener/v7");
    config->ves_basicauth_username = strdup("sample1");
    config->ves_basicauth_password = strdup("sample1");
    config->ves_nf_vendor_name = strdup("OpenAirInterface");
    config->ves_nf_naming_code = strdup("OGNB");
    config->ves_oam_ipv4 = strdup("10.30.40.50");
    config->ves_oam_ipv6 = strdup("");
    config->ves_ftp_server_port = 21;
    config->ves_ftp_listen_addr = strdup("0.0.0.0");
    config->demo_alarming_interval = 30;
#else
    int ret = config_get(o1config_params, sizeof(o1config_params) / sizeof(paramdef_t), "o1Config");
    AssertFatal(ret >= 0, "configuration load couldn't be performed");
#endif

    return o1_validate_config(config);
}

void o1_config_free(o1_config_t* config) {
    free(config->netconf_node_id);
    free(config->netconf_username);
    free(config->netconf_password);
    free(config->netconf_host);
    free(config->ves_url);
    free(config->ves_basicauth_username);
    free(config->ves_basicauth_password);
    free(config->ves_nf_vendor_name);
    free(config->ves_nf_naming_code);
    free(config->ves_oam_ipv4);
    free(config->ves_oam_ipv6);
    free(config->ves_ftp_listen_addr);
}

void o1_config_print(o1_config_t *config) {
    netconf_log("Printing config:");
    netconf_log("netconf_node_id: %s", config->netconf_node_id);
    netconf_log("netconf_username: %s", config->netconf_username);
    netconf_log("netconf_password: %s", config->netconf_password);
    netconf_log("netconf_port = %d", config->netconf_port);
    netconf_log("netconf_host: %s", config->netconf_host);
    netconf_log("ves_url: %s", config-> ves_url);
    netconf_log("ves_username: %s", config-> ves_basicauth_username);
    netconf_log("ves_password: %s", config-> ves_basicauth_password);
    netconf_log("ves_nf_vendor_name: %s", config->ves_nf_vendor_name);
    netconf_log("ves_nf_naming_code: %s", config->ves_nf_naming_code);
    netconf_log("ves_oam_ipv4: %s", config->ves_oam_ipv4);
    netconf_log("ves_oam_ipv6: %s", config->ves_oam_ipv6);
    netconf_log("ves_ftp_server_port = %d", config->ves_ftp_server_port);
    netconf_log("ves_ftp_listen_addr: %s", config->ves_ftp_listen_addr);
    netconf_log("demo_alarming_interval: %d", config->demo_alarming_interval);
    netconf_log("=======");
}

bool o1_config_is_ves_enabled(o1_config_t *config){
    return ((config->ves_url) && (strlen(config->ves_url) > 0));
}

bool o1_config_is_demoalarming_enabled(o1_config_t *config) {
    return (config->demo_alarming_interval > 0);
}

static int o1_validate_config(o1_config_t* config) {
    if (config == NULL) {
        return 1;
    }

    // TODO

    if (o1_config_is_ves_enabled(config)) {
        //verify vesNfNamingCode (has to be exactly 4 characters long)
    }
    else {

    }

    return 0;
}