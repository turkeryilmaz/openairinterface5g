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
            pnf_ip_v6_address: "",
            pnf_port: config.netconf_port,
            mac_address: "12:34:56:78:90:AB",
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


int netconf_demo_populate(void) {
    int rc = 0;

    const char *list_running[] = {
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']",
        0,
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']",
        0,
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/attributes/vnfParametersList[vnfInstanceId='SDNR_NODE_ID']",
        0,
        
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Downlink']",
        0,
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Downlink']/attributes/bwpContext",
        "DL",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Downlink']/attributes/isInitialBwp",
        "INITIAL",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Downlink']/attributes/subCarrierSpacing",
        "60",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Downlink']/attributes/cyclicPrefix",
        "NORMAL",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Downlink']/attributes/startRB",
        "0",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Downlink']/attributes/numberOfRBs",
        "2",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Downlink']/ThresholdMonitor[id='Threshold']",
        0,
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Downlink']/ThresholdMonitor[id='Threshold']/attributes/administrativeState",
        "UNLOCKED",

        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Uplink']",
        0,
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Uplink']/attributes/bwpContext",
        "UL",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Uplink']/attributes/isInitialBwp",
        "INITIAL",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Uplink']/attributes/subCarrierSpacing",
        "15",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Uplink']/attributes/cyclicPrefix",
        "EXTENDED",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Uplink']/attributes/startRB",
        "0",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Uplink']/attributes/numberOfRBs",
        "2",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Uplink']/ThresholdMonitor[id='Threshold']",
        0,
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Uplink']/ThresholdMonitor[id='Threshold']/attributes/administrativeState",
        "UNLOCKED",

        //alarms
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/AlarmList[id='CUUP-alarms']",
        0,
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/AlarmList[id='CUUP-alarms']/attributes/administrativeState",
        "UNLOCKED",
        
        
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/AlarmList[id='CUUP-alarms']/attributes/alarmRecords[alarmId='alarm-id-01']",
        0,

        //mandatory nodes
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/attributes/priorityLabel",
        "5",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/attributes/priorityLabel",
        "6",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/attributes/gNBId",
        "7",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/attributes/gNBIdLength",
        "23",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/attributes/gNBDUId",
        "8",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/attributes/vnfParametersList[vnfInstanceId='SDNR_NODE_ID']/autoScalable",
        "true",

        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Downlink']/attributes/priorityLabel",
        "9",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Uplink']/attributes/priorityLabel",
        "10",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Downlink']/ThresholdMonitor[id='Threshold']/attributes/thresholdInfoList[idx='0']/thresholdLevel",
        "12",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Uplink']/ThresholdMonitor[id='Threshold']/attributes/thresholdInfoList[idx='0']/thresholdLevel",
        "13",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Downlink']/ThresholdMonitor[id='Threshold']/attributes/thresholdInfoList[idx='0']/thresholdDirection",
        "UP",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Uplink']/ThresholdMonitor[id='Threshold']/attributes/thresholdInfoList[idx='0']/thresholdDirection",
        "UP",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Downlink']/ThresholdMonitor[id='Threshold']/attributes/thresholdInfoList[idx='0']/thresholdValue",
        "14",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Uplink']/ThresholdMonitor[id='Threshold']/attributes/thresholdInfoList[idx='0']/thresholdValue",
        "15",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Downlink']/ThresholdMonitor[id='Threshold']/attributes/monitorGranularityPeriod",
        "16",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/_3gpp-nr-nrm-bwp:BWP[id='Uplink']/ThresholdMonitor[id='Threshold']/attributes/monitorGranularityPeriod",
        "17",
        

        0
    };

    const char *list_operational[] = {
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/_3gpp-nr-nrm-gnbdufunction:GNBDUFunction[id='OAI-DU']/attributes/rimRSReportConf[reportInterval='1000']/reportIndicator",
        "DISABLED",

        //alarms
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/AlarmList[id='CUUP-alarms']/attributes/operationalState",
        "ENABLED",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/AlarmList[id='CUUP-alarms']/attributes/numOfAlarmRecords",
        "1",
        "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/AlarmList[id='CUUP-alarms']/attributes/lastModification",
        "2023-02-07T08:44:26.0Z",
        0
    };

    const char **l = list_running;
    while(*l) {
        const char *xpath = *l;
        const char *value = *(l + 1);
        netconf_log("[config] populating %s with %s.. ", xpath, value);
        rc = sr_set_item_str(netconf_session_running, xpath, value, 0, 0);
        if(rc != SR_ERR_OK) {
            netconf_log_error("error on line %d", __LINE__);
        }
        
        l += 2;
    }

    rc = sr_apply_changes(netconf_session_running, 0);
    if(rc != SR_ERR_OK) {
        netconf_log_error("error on sr_apply_changes line %d", __LINE__);
    }

    l = list_operational;
    while(*l) {
        const char *xpath = *l;
        const char *value = *(l + 1);
        netconf_log("[oper] populating %s with %s.. ", xpath, value);
        rc = sr_set_item_str(netconf_session_operational, xpath, value, 0, 0);
        if(rc != SR_ERR_OK) {
            netconf_log_error("error on line %d", __LINE__);
        }
        
        l += 2;
    }

    rc = sr_apply_changes(netconf_session_operational, 0);
    if(rc != SR_ERR_OK) {
        netconf_log_error("error on sr_apply_changes line %d", __LINE__);
    }

    return 0;
}
