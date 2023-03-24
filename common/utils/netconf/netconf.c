#include "netconf.h"
#include "o1config.h"
#include "netconf_session.h"
#include "netconf_log.h"
#include "utils.h"
#include "ves.h"
#include "ves_demo_alarm.h"

int netconf_init(void) {
    int rc = 0;

    //get configuration
    o1_config_t config = {0};
    rc = o1_config_read(&config);
    if(rc != 0) {
        netconf_log_error("o1_config_read() failed");
        goto failed;
    }
    o1_config_print(&config);

    //init netconf
    sr_log_stderr(SR_LL_WRN);
    rc = netconf_session_init();
    if(rc != 0) {
        netconf_log_error("netconf_session_init() failed");
        goto failed;
    }

    //init ves
    if (o1_config_is_ves_enabled(&config)) {
        ves_header_t data = {
            hostname: get_hostname(),
            nf_naming_code: config.ves_nf_naming_code,
            nf_vendor_name: config.ves_nf_vendor_name,
            pnf_id: config.netconf_node_id
        };

        ves_config_t ves_config = {
            url: config.ves_url,
            username: config.ves_basicauth_username,
            password: config.ves_basicauth_password
        };

        rc = ves_init(&ves_config, &data);
        free(data.hostname);

        if(rc != 0) {
            netconf_log_error("ves_init() failed");
            goto failed;
        }

        ves_registration_data_t pnf_reg_data= {
            pnf_username: config.netconf_username,
            pnf_password: config.netconf_password,
            pnf_ip_v4_address: config.netconf_host,
            pnf_ip_v6_address: config.ves_oam_ipv6,
            pnf_port: config.netconf_port,
            mac_address: "12:34:56:78:90:AB",   //TODO get it programatically
            vendor: config.ves_nf_vendor_name,
            model: "testModel",
            type: "gNB",
            is_tls: false
        };

        rc = ves_pnf_registration_execute(&pnf_reg_data);
        if(rc != 0) {
            netconf_log_error("ves_pnf_registration_execute() failed");
            goto failed;
        }

        if(o1_config_is_demoalarming_enabled(&config)){
            ves_demo_alarm_start(&config);    
        }
    }

    rc = netconf_disable_nacm(netconf_session_running);
    if(rc != 0) {
        netconf_log("netconf_disable_nacm() failed\n");
    }

    printf("netconf_demo_populate()...\n");
    rc = netconf_demo_populate();
    if(rc != 0) {
        netconf_log_error("netconf_demo_subscribe_edit_change() failed");
    }

    printf("netconf_demo_subscribe_edit_change()...\n");
    rc = netconf_demo_subscribe_edit_change();
    if(rc != 0) {
        netconf_log_error("netconf_demo_subscribe_edit_change() failed");
    }

    printf("netconf_demo_subscribe_operational()...\n");
    rc = netconf_demo_subscribe_operational();
    if(rc != 0) {
        netconf_log_error("netconf_demo_subscribe_operational() failed");
    }

    o1_config_free(&config);
    return 0;

failed:
    ves_free();
    o1_config_free(&config);
    return 1;
}


int netconf_free(void) {
    ves_free();
    netconf_session_free();

    return 0;
}
