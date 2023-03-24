#include "netconf.h"
#include "netconf_log.h"
#include "netconf_session.h"
#include "ves.h"

#include <sysrepo.h>
#include <libyang/libyang.h>

#include <stdint.h>

static int test_edit_callback(sr_session_ctx_t *session, uint32_t sub_id, const char *module_name, const char *xpath, sr_event_t event, uint32_t request_id, void *private_data);
static int test_oper_callback(sr_session_ctx_t *session, uint32_t sub_id, const char *module_name, const char *path, const char *request_xpath, uint32_t request_id, struct lyd_node **parent, void *private_data);

int netconf_disable_nacm(sr_session_ctx_t *session) {
    #define IETF_NETCONF_ACM_ENABLE_NACM_SCHEMA_XPATH               "/ietf-netconf-acm:nacm/enable-nacm"
    #define IETF_NETCONF_ACM_GROUPS_SCHEMA_XPATH                    "/ietf-netconf-acm:nacm/groups"
    #define IETF_NETCONF_ACM_RULE_LIST_SCHEMA_XPATH                 "/ietf-netconf-acm:nacm/rule-list"

    int rc = 0;
    rc = sr_set_item_str(session, IETF_NETCONF_ACM_ENABLE_NACM_SCHEMA_XPATH, "true", 0, 0);
    if(rc != SR_ERR_OK) {
        netconf_log_error("sr_set_item_str() failed");
        goto failed;
    }

    rc = sr_set_item_str(session, IETF_NETCONF_ACM_GROUPS_SCHEMA_XPATH"/group[name='sudo']/user-name", "netconf", 0, 0);
    if(rc != SR_ERR_OK) {
        netconf_log_error("sr_set_item_str() failed");
        goto failed;
    }

    rc = sr_set_item_str(session, IETF_NETCONF_ACM_RULE_LIST_SCHEMA_XPATH"[name='sudo-rules']/group", "sudo", 0, 0);
    if(rc != SR_ERR_OK) {
        netconf_log_error("sr_set_item_str() failed");
        goto failed;
    }

    rc = sr_set_item_str(session, IETF_NETCONF_ACM_RULE_LIST_SCHEMA_XPATH"[name='sudo-rules']/rule[name='allow-all-sudo']/module-name", "*", 0, 0);
    if(rc != SR_ERR_OK) {
        netconf_log_error("sr_set_item_str() failed");
        goto failed;
    }

    rc = sr_set_item_str(session, IETF_NETCONF_ACM_RULE_LIST_SCHEMA_XPATH"[name='sudo-rules']/rule[name='allow-all-sudo']/path", "/", 0, 0);
    if(rc != SR_ERR_OK) {
        netconf_log_error("sr_set_item_str() failed");
        goto failed;
    }

    rc = sr_set_item_str(session, IETF_NETCONF_ACM_RULE_LIST_SCHEMA_XPATH"[name='sudo-rules']/rule[name='allow-all-sudo']/access-operations", "*", 0, 0);
    if(rc != SR_ERR_OK) {
        netconf_log_error("sr_set_item_str() failed");
        goto failed;
    }

    rc = sr_set_item_str(session, IETF_NETCONF_ACM_RULE_LIST_SCHEMA_XPATH"[name='sudo-rules']/rule[name='allow-all-sudo']/action", "permit", 0, 0);
    if(rc != SR_ERR_OK) {
        netconf_log_error("sr_set_item_str() failed");
        goto failed;
    }

    rc = sr_set_item_str(session, IETF_NETCONF_ACM_RULE_LIST_SCHEMA_XPATH"[name='sudo-rules']/rule[name='allow-all-sudo']/comment", "Corresponds all the rules under the sudo group as defined in O-RAN.WG4.MP.0-v05.00", 0, 0);
    if(rc != SR_ERR_OK) {
        netconf_log_error("sr_set_item_str() failed");
        goto failed;
    }

    rc = sr_apply_changes(session, 0);
    if(rc != SR_ERR_OK) {
        netconf_log_error("sr_apply_changes() failed");
        goto failed;
    }
    
    return 0;

failed:
    sr_discard_changes(session);
    return 1;
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

int netconf_demo_subscribe_edit_change() {
    const char *xpath = "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/attributes/userLabel";
    int rc = sr_module_change_subscribe(netconf_session_running, "_3gpp-common-managed-element", xpath, test_edit_callback, NULL, 0, 0, &netconf_session_subscription);
    if (rc != SR_ERR_OK) {
        netconf_log_error("sr_module_change_subscribe() failed");
        goto failed;
    }

    return 0;
failed:
    return 1;
}

int netconf_demo_subscribe_operational() {
    static uint32_t counter = 1;

    const char *xpath = "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/attributes";
    int rc = sr_oper_get_subscribe(netconf_session_operational, "_3gpp-common-managed-element", xpath, test_oper_callback, &counter, SR_SUBSCR_DEFAULT, &netconf_session_subscription);
    if(rc != SR_ERR_OK) {
        netconf_log_error("sr_oper_get_subscribe() failed");
        goto failed;
    }

    return 0;

failed:
    return 1;
}

int test_edit_callback(sr_session_ctx_t *session, uint32_t sub_id, const char *module_name, const char *xpath, sr_event_t event, uint32_t request_id, void *private_data) {
    sr_change_iter_t *it = NULL;
    int rc = SR_ERR_OK;
    char path[512];
    sr_change_oper_t oper;
    sr_val_t *old_value = NULL;
    sr_val_t *new_value = NULL;

    (void)sub_id;
    (void)request_id;
    (void)private_data;
    
    
    if(event == SR_EV_CHANGE) {
        if (xpath) {
            sprintf(path, "%s//.", xpath);
        }
        else {
            sprintf(path, "/%s:*//.", module_name);
        }

        rc = sr_get_changes_iter(session, path, &it);
        if (rc != SR_ERR_OK) {
            netconf_log_error("sr_get_changes_iter() failed");
            goto failed;
        }

        while ((rc = sr_get_change_next(session, it, &oper, &old_value, &new_value)) == SR_ERR_OK) {
            if((oper == SR_OP_CREATED) || (oper == SR_OP_MODIFIED)) {
                if(strcmp("/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/attributes/userLabel", new_value->xpath) == 0) {
                    if(strcmp(new_value->data.string_val, "enabled") == 0) {
                        printf("enabled\n");
                    }
                    else if(strcmp(new_value->data.string_val, "disabled") == 0) {
                        printf("disabled\n");
                    }
                    else if(strstr(new_value->data.string_val, "pmData") == new_value->data.string_val) {
                        ves_file_ready_t pm_data = {
                            .expires_on = "2099-12-31T235959.0Z",
                            .filesize = 1234,
                            .filelocation = "/ftp/test.gz"
                        };

                        int rc = ves_pnf_pmdata_fileready_execute(&pm_data);
                        if(rc != 0) {
                            netconf_log_error("ves_pnf_pmdata_fileready_execute() failed");
                        }
                    }
                    else {
                        return SR_ERR_VALIDATION_FAILED;
                    }
                }
            }


            sr_free_val(old_value);
            sr_free_val(new_value);
        }

        sr_free_change_iter(it);
    }
    return SR_ERR_OK;

failed:
    return SR_ERR_INTERNAL;
}

static int test_oper_callback(sr_session_ctx_t *session, uint32_t sub_id, const char *module_name, const char *path, const char *request_xpath, uint32_t request_id, struct lyd_node **parent, void *private_data) {
    uint32_t *counter = private_data;
    (void)session;
    (void)sub_id;
    (void)module_name;
    (void)path;
    (void)request_xpath;
    (void)request_id;

    const char *xpath = "/_3gpp-common-managed-element:ManagedElement[id='OAI gNodeB']/attributes/SupportedPerfMetricGroup/granularityPeriods";

    char value[10];
    sprintf(value, "%d", *counter);
    struct lyd_node *node;
    LY_ERR rc = lyd_new_path(*parent, 0, xpath, value, 0, &node);
    if(rc != LY_SUCCESS) {
        netconf_log_error("lyd_new_path() failed");
        goto failed;
    }

    (*counter)++;
    return SR_ERR_OK;

failed:
    return SR_ERR_INTERNAL;
}
