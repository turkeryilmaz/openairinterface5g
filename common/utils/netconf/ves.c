#define _GNU_SOURCE

#include "ves.h"
#include "utils.h"
#include "netconf_log.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <curl/curl.h>

#define VES_COMMON_HEADER_TEMPLATE "\"commonEventHeader\": {\
    \"domain\": \"@domain@\",\
    \"eventId\": \"@domain@_@eventType@\",\
    \"eventName\": \"@domain@_@eventType@\",\
    \"eventType\": \"@eventType@\",\
    \"sequence\": @seqId@,\
    \"priority\": \"@priority@\",\
    \"reportingEntityId\": \"\",\
    \"reportingEntityName\": \"@controllerName@\",\
    \"sourceId\": \"\",\
    \"sourceName\": \"@pnfId@\",\
    \"startEpochMicrosec\": @timestamp@,\
    \"lastEpochMicrosec\": @timestamp@,\
    \"nfNamingCode\": \"@nfNamingCode@\",\
    \"nfVendorName\": \"@nfVendorName@\",\
    \"timeZoneOffset\": \"+00:00\",\
    \"version\": \"4.1\",\
    \"vesEventListenerVersion\": \"7.2.1\",\
    \"stndDefinedNamespace\": \"o1-notify-pnf-registration\",\
    \"nfcNamingCode\": \"NFC\"\
}"

#define VES_PNF_REGISTRATION_TEMPLATE "\"pnfRegistrationFields\": {\
    \"pnfRegistrationFieldsVersion\": \"2.1\",\
    \"lastServiceDate\": \"2021-03-26\",\
    \"macAddress\": \"@macAddress@\",\
    \"manufactureDate\": \"2021-01-16\",\
    \"modelNumber\": \"@model@\",\
    \"oamV4IpAddress\": \"@oamIp@\",\
    @ipv6FullDefine@\
    \"serialNumber\": \"@vendor@-@type@-@oamIp@-@model@\",\
    \"softwareVersion\": \"2.3.5\",\
    \"unitFamily\": \"@vendor@-@type@\",\
    \"unitType\": \"@type@\",\
    \"vendorName\": \"@vendor@\",\
    \"additionalFields\": {\
        \"oamPort\": \"@port@\",\
        \"protocol\": \"SSH\",\
        \"username\": \"@username@\",\
        \"password\": \"@password@\",\
        \"reconnectOnChangedSchema\": \"false\",\
        \"sleep-factor\": \"1.5\",\
        \"tcpOnly\": \"false\",\
        \"connectionTimeout\": \"20000\",\
        \"maxConnectionAttempts\": \"100\",\
        \"betweenAttemptsTimeout\": \"2000\",\
        \"keepaliveDelay\": \"120\"\
    }\
}"

#define VES_PNF_ALARMING_TEMPLATE "\"stndDefinedFields\": {\
    \"schemaReference\": \"https://forge.3gpp.org/rep/sa5/MnS/-/raw/Rel-16/OpenAPI/TS28532_FaultMnS.yaml#components/schemas/NotifyNewAlarm\",\
    \"data\": {\
        \"href\": \"href1\",\
        \"notificationId\": 0,\
        \"notificationType\": \"notifyNewAlarm\",\
        \"eventTime\": \"@eventTime@\",\
        \"systemDN\": \"xyz\",\
        \"alarmId\": \"@alarm@\",\
        \"alarmType\": \"COMMUNICATIONS_ALARM\",\
        \"probableCause\": \"@alarm@\",\
        \"specificProblem\": \"@alarm@\",\
        \"perceivedSeverity\": \"@severity@\",\
        \"backedUpStatus\": true,\
        \"backUpObject\": \"xyz\",\
        \"trendIndication\": \"MORE_SEVERE\",\
        \"thresholdInfo\": {\
            \"observedMeasurement\": \"new\",\
            \"observedValue\": 123.1\
        },\
        \"correlatedNotifications\": [],\
        \"stateChangeDefinition\": [{ \"operational-state\": \"DISABLED\" }],\
        \"monitoredAttributes\": {\
            \"interface\": \"@interface@\"\
        },\
        \"proposedRepairActions\": \"Call the police!\",\
        \"additionalText\": \"O-RAN Software Community OAM\",\
        \"additionalInformation\": {\
            \"eventTime\": \"@eventTime@\",\
            \"equipType\": \"@type@\",\
            \"vendor\": \"@vendor@\",\
            \"model\": \"@model@\"\
        },\
        \"rootCauseIndicator\": false\
    },\
    \"stndDefinedFieldsVersion\": \"1.0\"\
}"

#define VES_FILE_READY_TEMPLATE "\"notificationFields\": {\
    \"changeIdentifier\":  \"PM_MEAS_FILES\",\
    \"changeType\":  \"FileReady\",\
    \"notificationFieldsVersion\":  \"2.0\",\
    \"arrayOfNamedHashMap\":  [\
        {\
            \"name\":  \"@filename@\",\
            \"hashMap\":  {\
                \"location\": \"@filelocation@\",\
                \"compression\": \"gzip\",\
                \"fileFormatType\": \"org.3GPP.32.435#measCollec\",\
                \"fileFormatVersion\": \"V5\"\
            }\
        }\
    ]\
}"

struct memory {
    char *response;
    size_t size;
};

static int ves_execute(char *priority, char *domain, char *event_type, char* main_content);
static int ves_http_request(const char *url, const char *username, const char* password, const char *method, const char *send_data, int *response_code, char **recv_data);
static size_t curl_write_cb(void *data, size_t size, size_t nmemb, void *userp);

static ves_config_t ves_config = {0};
static int ves_seq_id = 0;
static char *ves_common_header = 0;


/**
 * initialize ves component
 *   save config internally
 *   prepare ves common header as far as possible
*/
int ves_init(ves_config_t *config, ves_header_t *ves) {
    if(ves_common_header) {
        netconf_log_error("ves_init() already called");
        goto failed;
    }

    ves_common_header = strdup(VES_COMMON_HEADER_TEMPLATE);
    if(ves_common_header == 0) {
        netconf_log_error("strdup() error");
        goto failed;
    }

    ves_config.url = strdup(config->url);
    ves_config.username = strdup(config->username);
    ves_config.password = strdup(config->password);
    if(!ves_config.url || !ves_config.username || !ves_config.password) {
        netconf_log_error("strdup() error");
        goto failed;
    }

    ves_common_header = str_replace_inplace(ves_common_header, "@controllerName@", ves->hostname);
    if(ves_common_header ==  0) {
        netconf_log_error("str_replace_inplace() failed");
        goto failed;
    }

    ves_common_header = str_replace_inplace(ves_common_header, "@pnfId@", ves->pnf_id);
    if(ves_common_header ==  0) {
        netconf_log_error("str_replace_inplace() failed");
        goto failed;
    }

    ves_common_header = str_replace_inplace(ves_common_header, "@nfNamingCode@", ves->nf_naming_code);
    if(ves_common_header ==  0) {
        netconf_log_error("str_replace_inplace() failed");
        goto failed;
    }

    ves_common_header = str_replace_inplace(ves_common_header, "@nfVendorName@", ves->nf_vendor_name);
    if(ves_common_header ==  0) {
        netconf_log_error("str_replace_inplace() failed");
        goto failed;
    }

    return 0;

failed:
    ves_free();

    return 1;
}

void ves_free() {
    free(ves_common_header);
    free(ves_config.url);
    free(ves_config.username);
    free(ves_config.password);

    ves_seq_id = 0;
    ves_common_header = 0;
    memset(&ves_config, 0, sizeof(ves_config_t));
}

/**
 * send ves pnfRegistration
 * 
*/
int ves_pnf_registration_execute(ves_registration_data_t *ves) {
    char *content = strdup(VES_PNF_REGISTRATION_TEMPLATE);
    if(content == 0) {
        netconf_log_error("strdup() failed");
        goto failed;
    }

    char *port = 0;
    asprintf(&port, "%d",ves->pnf_port);
    if(port == 0) {
        netconf_log_error("asprintf() failed");
        goto failed;
    }

    // fill template
    content = str_replace_inplace(content, "@macAddress@", ves->mac_address);
    if(content == 0) {
        netconf_log_error("str_replace_inplace() failed");
        goto failed;
    }

    content = str_replace_inplace(content, "@model@", ves->model);
    if(content == 0) {
        netconf_log_error("str_replace_inplace() failed");
        goto failed;
    }

    content = str_replace_inplace(content, "@oamIp@", ves->pnf_ip_v4_address);
    if(content == 0) {
        netconf_log_error("str_replace_inplace() failed");
        goto failed;
    }

    char *ipv6 = strdup("");
    if(ipv6 == 0) {
        netconf_log_error("strdup() failed");
        goto failed;
    }    
    if(ves->pnf_ip_v6_address && strlen(ves->pnf_ip_v6_address)) {
        free(ipv6);
        asprintf(&ipv6, "\"oamV6IpAddress\": \"%s\"", ves->pnf_ip_v6_address);
        if(ipv6 == 0) {
            netconf_log_error("asprintf() failed");
            goto failed;
        }    
    }
    content = str_replace_inplace(content, "@ipv6FullDefine@", ipv6);
    free(ipv6);
    if(content == 0) {
        netconf_log_error("str_replace_inplace() failed");
        goto failed;
    }

    content = str_replace_inplace(content, "@vendor@", ves->vendor);
    if(content == 0) {
        netconf_log_error("str_replace_inplace() failed");
        goto failed;
    }

    content = str_replace_inplace(content, "@type@", ves->type);
    if(content == 0) {
        netconf_log_error("str_replace_inplace() failed");
        goto failed;
    }

    content = str_replace_inplace(content, "@port@", port);
    if(content == 0) {
        netconf_log_error("str_replace_inplace() failed");
        goto failed;
    }

    content = str_replace_inplace(content, "@username@", ves->pnf_username);
    if(content == 0) {
        netconf_log_error("str_replace_inplace() failed");
        goto failed;
    }

    content = str_replace_inplace(content, "@password@", ves->pnf_password);
    if(content == 0) {
        netconf_log_error("str_replace_inplace() failed");
        goto failed;
    }

    int rc = ves_execute(PRIORITY_PNFREGISTRATION, DOMAIN_PNFREGISTRAION, EVENTTYPE_PNFREGISTRATION, content);
    if(rc != 0) {
        netconf_log_error("ves_execute() failed");
        goto failed;
    }

    free(port);
    free(content);
    return 0;

failed:
    free(port);
    free(content);
    return 1;
}
/**
 * send ves pnf file ready event
 * 
*/
int ves_pnf_pmdata_fileready_execute(ves_file_ready_t* ves_data) {
    char *content = strdup(VES_FILE_READY_TEMPLATE);
    if(content == 0) {
        netconf_log_error("strdup() failed");
        goto failed;
    }

    // fill template
    content = str_replace_inplace(content, "@filename@", ves_data->filename);
    if(content == 0) {
        netconf_log_error("str_replace_inplace failed");
        goto failed;
    }

    content = str_replace_inplace(content, "@filelocation@", ves_data->filelocation);
    if(content == 0) {
        netconf_log_error("str_replace_inplace failed");
        goto failed;
    }

    int rc = ves_execute(PRIORITY_PMDATA, DOMAIN_PMDATA, EVENTTYPE_PNFALARMING, content);
    if(rc != 0) {
        netconf_log_error("ves_execute() failed");
        goto failed;
    }

    free(content);
    return 0;

failed:
    free(content);
    return 1;
}
/**
 * send ves pnf alarm
 * 
*/
int ves_pnf_alarm_execute(ves_alarm_t* ves_alarm){
    char *content = strdup(VES_PNF_ALARMING_TEMPLATE);
    if(content == 0) {
        netconf_log_error("strdup() failed");
        goto failed;
    }

    // fill template
    content = str_replace_inplace(content, "@eventTime@", "");
    if(content == 0) {
        netconf_log_error("str_replace_inplace failed");
        goto failed;
    }

    content = str_replace_inplace(content, "@alarm@", "");
    if(content == 0) {
        netconf_log_error("str_replace_inplace failed");
        goto failed;
    }

    content = str_replace_inplace(content, "@severity@", "");
    if(content == 0) {
        netconf_log_error("str_replace_inplace failed");
        goto failed;
    }

    content = str_replace_inplace(content, "@type@", "");
    if(content == 0) {
        netconf_log_error("str_replace_inplace failed");
        goto failed;
    }

    content = str_replace_inplace(content, "@vendor@", "");
    if(content == 0) {
        netconf_log_error("str_replace_inplace failed");
        goto failed;
    }

    content = str_replace_inplace(content, "@model@", "");
    if(content == 0) {
        netconf_log_error("str_replace_inplace failed");
        goto failed;
    }

    
    int rc = ves_execute(PRIORITY_PNFALARMING, DOMAIN_PNFALARMING, EVENTTYPE_PNFALARMING, content);
    if(rc != 0) {
        netconf_log_error("ves_execute() failed");
        goto failed;
    }

    free(content);
    return 0;

failed:
    free(content);
    return 1;
}

static int ves_execute(char *priority, char *domain, char *event_type, char* main_content) {
    char *timestamp = 0;
    char *seq_id = 0;
    char *post_data = 0;
    char *header = 0;

    asprintf(&timestamp, "%lu", get_microseconds_since_epoch());
    if(timestamp == 0) {
        netconf_log_error("asprintf() failed");
        goto failed;
    }
    
    asprintf(&seq_id, "%d", ves_seq_id);
    if(seq_id == 0) {
        netconf_log_error("asprintf() failed");
        goto failed;
    }

    //fill header
    header = strdup(ves_common_header);
    if(header == 0) {
        netconf_log_error("strdup() failed");
        goto failed;
    }

    header = str_replace_inplace(header, "@seqId@", seq_id);
    if(header == 0) {
        netconf_log_error("str_replace_inplace() failed");
        goto failed;
    }

    header = str_replace_inplace(header, "@timestamp@", timestamp);
    if(header == 0) {
        netconf_log_error("str_replace_inplace() failed");
        goto failed;
    }

    header = str_replace_inplace(header, "@priority@", priority);
    if(header == 0) {
        netconf_log_error("str_replace_inplace() failed");
        goto failed;
    }

    header = str_replace_inplace(header, "@domain@", domain);
    if(header == 0) {
        netconf_log_error("str_replace_inplace() failed");
        goto failed;
    }

    header = str_replace_inplace(header, "@eventType@", event_type);
    if(header == 0) {
        netconf_log_error("str_replace_inplace() failed");
        goto failed;
    }

    //append body
    asprintf(&post_data,"{\"event\":{%s,%s}}", header, main_content);
    if(post_data == 0) {
        netconf_log_error("asprintf() failed");
        goto failed;
    }

    //send request
    int response_code;
    char *response = 0;
    int rc = ves_http_request(ves_config.url, ves_config.username, ves_config.password, "POST", post_data, &response_code, &response);
    if(rc != 0) {
        netconf_log_error("ves_http_request() failed");
        goto failed;
    }

    netconf_log("response_code = %d", response_code);
    if(response) {
        netconf_log("response = %s", response);
    }

    free(response);
    if(response_code > 399) {
        netconf_log_error("failure http response code: %d", response_code);
        goto failed;
    }

    
    ves_seq_id++;
    free(timestamp);
    free(seq_id);
    free(post_data);
    free(header);

    return 0;

failed:
    free(timestamp);
    free(seq_id);
    free(post_data);
    free(header);

    return 1;
}

static int ves_http_request(const char *url, const char *username, const char* password, const char *method, const char *send_data, int *response_code, char **recv_data) {
    const char *send_data_good = send_data;
    if(send_data_good == 0) {
        send_data_good = "";
    }

    CURL *curl = curl_easy_init();
    if(curl == 0) {
        netconf_log_error("curl_easy_init() error");
        goto failed;
    }

    // set curl options
    struct curl_slist *header = 0;
    header = curl_slist_append(header, "Content-Type: application/json");
    if(!header) {
        netconf_log_error("curl_slist_append() failed");
        goto failed;
    }

    header = curl_slist_append(header, "Accept: application/json");
    if(!header) {
        netconf_log_error("curl_slist_append() failed");
        goto failed;
    }

    CURLcode res = CURLE_OK;
    res = curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header);
    if(res != CURLE_OK) {
        netconf_log_error("curl_easy_setopt() error");
        goto failed;
    }

    res = curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 1L);     //seconds timeout for a connection
    if(res != CURLE_OK) {
        netconf_log_error("curl_easy_setopt() error");
        goto failed;
    }

    res = curl_easy_setopt(curl, CURLOPT_TIMEOUT, 1L);            //seconds timeout for an operation
    if(res != CURLE_OK) {
        netconf_log_error("curl_easy_setopt() error");
        goto failed;
    }

    res = curl_easy_setopt(curl, CURLOPT_FRESH_CONNECT, 1L);
    if(res != CURLE_OK) {
        netconf_log_error("curl_easy_setopt() error");
        goto failed;
    }

    res = curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L);
    if(res != CURLE_OK) {
        netconf_log_error("curl_easy_setopt() error");
        goto failed;
    }


    // disable SSL verifications
    res = curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    if(res != CURLE_OK) {
        netconf_log_error("curl_easy_setopt() error");
        goto failed;
    }

    res = curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
    if(res != CURLE_OK) {
        netconf_log_error("curl_easy_setopt() error");
        goto failed;
    }

    res = curl_easy_setopt(curl, CURLOPT_PROXY_SSL_VERIFYPEER, 0L);
    if(res != CURLE_OK) {
        netconf_log_error("curl_easy_setopt() error");
        goto failed;
    }

    res = curl_easy_setopt(curl, CURLOPT_PROXY_SSL_VERIFYHOST, 0L);
    if(res != CURLE_OK) {
        netconf_log_error("curl_easy_setopt() error");
        goto failed;
    }


    res = curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, method);
    if(res != CURLE_OK) {
        netconf_log_error("curl_easy_setopt() error");
        goto failed;
    }

    res = curl_easy_setopt(curl, CURLOPT_POSTFIELDS, send_data_good);
    if(res != CURLE_OK) {
        netconf_log_error("curl_easy_setopt() error");
        goto failed;
    }

    res = curl_easy_setopt(curl, CURLOPT_URL, url);
    if(res != CURLE_OK) {
        netconf_log_error("curl_easy_setopt() error");
        goto failed;
    }

    char *credentials = 0;
    if((username) && (password)) {
        asprintf(&credentials, "%s:%s", username, password);
        if(credentials == 0) {
            netconf_log_error("asprintf() failed");
            goto failed;
        }

        res = curl_easy_setopt(curl, CURLOPT_USERPWD, credentials);
        if(res != CURLE_OK) {
            netconf_log_error("curl_easy_setopt() error");
            goto failed;
        }
    }

    struct memory response_data = {0};
    if(recv_data) {
        res = curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
        if(res != CURLE_OK) {
            netconf_log_error("curl_easy_setopt() error");
            goto failed;
        }
        res = curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&response_data);
        if(res != CURLE_OK) {
            netconf_log_error("curl_easy_setopt() error");
            goto failed;
        }
    }
    
    netconf_log("%s-ing cURL to url=\"%s\" with body=\"%s\"... ", method, url, send_data_good);
    res = curl_easy_perform(curl);
    if(res != CURLE_OK) {
        netconf_log_error("curl_easy_perform() failed");
        goto failed;
    }

    if(response_code) {
        long http_rc;
        res = curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_rc);
        if(res != CURLE_OK) {
            netconf_log_error("curl_easy_getinfo() failed");
            goto failed;
        }
        *response_code = http_rc;
    }

    if(recv_data) {
        *recv_data = response_data.response;
    }

    free(credentials);
    curl_slist_free_all(header);
    curl_easy_cleanup(curl);
    return 0;

failed:
    free(credentials);
    curl_slist_free_all(header);
    curl_easy_cleanup(curl);
    return 1;
}

static size_t curl_write_cb(void *data, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    struct memory *mem = (struct memory *)userp;

    char *ptr = realloc(mem->response, mem->size + realsize + 1);
    if(ptr == NULL) {
        //TODO log_error("realloc failed\n");
        return 0;  /* out of memory! */
    }

    mem->response = ptr;
    memcpy(&(mem->response[mem->size]), data, realsize);
    mem->size += realsize;
    mem->response[mem->size] = 0;

    return realsize;
}
