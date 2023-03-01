#define _GNU_SOURCE

#include "ves.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static const char *vesCommonEventHandler = "\"commonEventHeader\": {\
    \"domain\": \"@domain@\",\
    \"eventId\": \"@eventId@\",\
    \"eventName\": \"@domain@_@eventType@\",\
    \"eventType\": \"@eventType@\",\
    \"sequence\": 0,\
    \"priority\": \"Low\",\
    \"reportingEntityId\": \"\",\
    \"reportingEntityName\": \"@controllerName@\",\
    \"sourceId\": \"\",\
    \"sourceName\": \"@pnfId@\",\
    \"startEpochMicrosec\": \"@timestamp@\",\
    \"lastEpochMicrosec\": \"@timestamp@\",\
    \"nfNamingCode\": \"@type@\",\
    \"nfVendorName\": \"@vendor@\",\
    \"timeZoneOffset\": \"+00:00\",\
    \"version\": \"4.1\",\
    \"vesEventListenerVersion\": \"7.2.1\"\
}";

static const char *pnfRegistrationTemplate = "\"pnfRegistrationFields\": {\
    \"pnfRegistrationFieldsVersion\": \"2.1\",\
    \"lastServiceDate\": \"2021-03-26\",\
    \"macAddress\": \"@macAddress@\",\
    \"manufactureDate\": \"2021-01-16\",\
    \"modelNumber\": \"@model@\",\
    \"oamV4IpAddress\": \"@oamIp@\",\
    \"oamV6IpAddress\": \"@oamIpV6@\",\
    \"serialNumber\": \"@vendor@-@type@-@oamIp@-@model@\",\
    \"softwareVersion\": \"2.3.5\",\
    \"unitFamily\": \"@vendor@-@type@\",\
    \"unitType\": \"@type@\",\
    \"vendorName\": \"@vendor@\",\
    \"additionalFields\": {\
        \"oamPort\": \"830\",\
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
}";

static char *ves_build_commonEventsHeader(ves_instance_t *ves) {
    char *v = strdup(vesCommonEventHandler);
    if(v == 0) {
        return 0;
    }

    char *eventId = 0;
    asprintf(&eventId, "%s-%d", ves->event_type, ves->_seq_id);
    if(eventId == 0) {
        goto failed;
    }

    long useconds = get_microseconds_since_epoch();
    char *timestamp = 0;
    asprintf(&timestamp, "%lu", useconds);
   
    // v = str_replace_inplace(v, "@domain@", ves->domain);
    // if(v == 0) {
    //     goto failed;
    // }

    // v = str_replace_inplace(v, "@eventId@", eventId);
    // if(v == 0) {
    //     goto failed;
    // }
    
    // v = str_replace_inplace(v, "@eventType@", ves->event_type);
    // if(v == 0) {
    //     goto failed;
    // }
    // v = str_replace_inplace(v, "@controllerName@", ves->);
    // if(v == 0) {
    //     goto failed;
    // }
    // v = str_replace_inplace(v, "@pnfId@", ves->);
    // if(v == 0) {
    //     goto failed;
    // }
    // v = str_replace_inplace(v, "@timestamp@", timestamp);
    // if(v == 0) {
    //     goto failed;
    // }
    // v = str_replace_inplace(v, "@type@", ves->);
    // if(v == 0) {
    //     goto failed;
    // }

    return v;

failed:
    free(eventId);
    free(timestamp);
    return 0;
}

int ves_pnf_registration_execute(ves_instance_t *ves_instance) {
    char *content = 0;

    content = ves_build_commonEventsHeader(ves_instance);

    return 0;
}
