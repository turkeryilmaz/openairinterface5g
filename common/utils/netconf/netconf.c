#include "netconf.h"
#include "netconf_session.h"
#include "o1config.h"

int netconf_init(void) {
    int rc = 0;

    printf("Reading config...");
    o1_config_t config={};
    rc = o1_readconfig(&config);
    printf("done %d\n",rc);

    printf("Printing config:\n");
    printf("netconf_node_id: %s\n", config.netconf_node_id);
    printf("netconf_username: %s\n", config.netconf_username);
    printf("netconf_password: %s\n", config.netconf_password);
    printf("netconf_port = %d\n", config.netconf_port);
    printf("netconf_host: %s\n", config.netconf_host);
    printf("ves_url: %s\n", config. ves_url);
    printf("ves_nf_vendor_name: %s\n", config.ves_nf_vendor_name);
    printf("ves_nf_naming_code: %s\n", config.ves_nf_naming_code);
    printf("ves_oam_ipv4: %s\n", config.ves_oam_ipv4);
    printf("ves_oam_ipv6: %s\n", config.ves_oam_ipv6);
    printf("ves_ftp_server_port = %d\n", config.ves_ftp_server_port);
    printf("ves_ftp_listen_addr: %s\n", config.ves_ftp_listen_addr);
    printf("=======\n");



    // sr_log_stderr(SR_LL_WRN);
    // rc = netconf_session_init();
    // if(rc != 0) {
    //     printf("netconf_session_init() failed\n");
    // }

    return rc;
}

int netconf_test(void) {
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
        printf("populating %s with %s.. ", xpath, value);
        rc = sr_set_item_str(netconf_session_running, xpath, value, 0, 0);
        if(rc != SR_ERR_OK) {
            printf("error on line %d\n", __LINE__);
        }
        else {
            printf("done\n");
        }
        l += 2;
    }

    rc = sr_apply_changes(netconf_session_running, 0);
    if(rc != SR_ERR_OK) {
        printf("error on sr_apply_changes line %d\n", __LINE__);
    }

    l = list_operational;
    while(*l) {
        const char *xpath = *l;
        const char *value = *(l + 1);
        printf("populating %s with %s.. ", xpath, value);
        rc = sr_set_item_str(netconf_session_operational, xpath, value, 0, 0);
        if(rc != SR_ERR_OK) {
            printf("error on line %d\n", __LINE__);
        }
        else {
            printf("done\n");
        }
        l += 2;
    }

    rc = sr_apply_changes(netconf_session_operational, 0);
    if(rc != SR_ERR_OK) {
        printf("error on sr_apply_changes line %d\n", __LINE__);
    }

    return 0;
}

int netconf_free(void) {
    netconf_session_free();

    return 0;
}
