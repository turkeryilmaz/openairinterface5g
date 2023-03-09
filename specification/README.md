# Yang Specification

## Important

It is not allowed to publish any of the 3gpp yang specifications in any kind of repository or something similar.


Please use the ```get-yangs.sh```  script to download the yang files. Th
All yang spec files to implement will be in the ```yang``` folder. Based on this tree and xml files can be generated with the ```generate-parts.sh``` script.

For the O-RAN O1 interface the 3GPP yang files(see sources) are the entrypoint, especially the ```_3gpp-common-managed-element.yang``` and in there its list of ```ManagedElement```s. These are then extended dependent on their functions with e.g. ```_3gpp-nr-nrm-gnbdufunction.yang```.
n


## Mapping



## VES Messages

### pnfRegistration

![pnfVesRegistration](https://wiki.onap.org/download/temp/plantuml7552781749771260603.png?contentType=image/png)

POST /eventListener/v7
```
{
    "event": {
        "commonEventHeader": {
            "domain": "pnfRegistration",
            "eventId": "pnfRegistration_EventType5G",
            "eventName": "pnfRegistration_EventType5G",
            "eventType": "EventType5G",
            "sequence": @seqenceId@,
            "priority": "Low",
            "reportingEntityId": "",
            "reportingEntityName": "@hostname@",
            "sourceId": "",
            "sourceName": "@pnfId@",
            "startEpochMicrosec": "@timestamp@",
            "lastEpochMicrosec": "@timestamp@",
            "nfNamingCode": "@type@",
            "nfVendorName": "@vendor@",
            "timeZoneOffset": "+00:00",
            "version": "4.1",
            "vesEventListenerVersion": "7.2.1"
        },
        "pnfRegistrationFields": {
            "pnfRegistrationFieldsVersion": "2.1",
            "lastServiceDate": "2021-03-26",
            "macAddress": "@macAddress@",
            "manufactureDate": "2021-01-16",
            "modelNumber": "@model@",
            "oamV4IpAddress": "@oamIp@",
            "oamV6IpAddress": "@oamIpV6@",
            "serialNumber": "@vendor@-@type@-@oamIp@-@model@",
            "softwareVersion": "2.3.5",
            "unitFamily": "@vendor@-@type@",
            "unitType": "@type@",
            "vendorName": "@vendor@",
            "additionalFields": {
                "oamPort": "830",
                "protocol": "SSH",
                "username": "netconf",
                "password": "netconf",
                "reconnectOnChangedSchema": "false",
                "sleep-factor": "1.5",
                "tcpOnly": "false",
                "connectionTimeout": "20000",
                "maxConnectionAttempts": "100",
                "betweenAttemptsTimeout": "2000",
                "keepaliveDelay": "120"
            }
        }
    }
}
```

### File based performance monitoring management

![Performance monitoring management](https://wiki.onap.org/download/temp/plantuml8525192104339431559.png?contentType=image/png)

```
POST /eventListener/v7
{
    "event": {
        "commonEventHeader": {
            "domain": "stndDefined",
            "eventId": "@eventId@",
            "eventName": "stndDefined_performanceMeasurementStreaming_15m",
            "eventType": "performanceMeasurementStreaming_15m",
            "sequence": @seqId@,
            "priority": "Low",
            "reportingEntityId": "",
            "reportingEntityName": "@hostname@",
            "sourceId": "",
            "sourceName": "@pnfId@",
            "startEpochMicrosec": "@collectionStartTime@",
            "lastEpochMicrosec": "@collectionEndTime@",
            "internalHeaderFields": {
                "intervalStartTime": "@intervalStartTime@",
                "intervalEndTime": "@intervalEndTime@"
            },
            "version": "4.1",
            "vesEventListenerVersion": "7.2.1"
        },
        ...TBD (talk to alex)
    }
}
```

### pnf Alarming

![pnf Alarming](https://wiki.onap.org/download/temp/plantuml4369307977353937036.png?contentType=image/png)

```
POST /eventListener/v7
{
  "event": {
    "commonEventHeader": {
      "domain": "stndDefined",
      "eventId": "stndDefined_O_RAN_COMPONENT_Alarms_@alarm@",
      "eventName": "stndDefined_O_RAN_COMPONENT_Alarms_@alarm@",
      "eventType": "O_RAN_COMPONENT_Alarms",
      "sequence": @seqId@,
      "priority": "Low",
      "reportingEntityId": "",
      "reportingEntityName": "@hostname@",
      "sourceId": "",
      "sourceName": "@pnfId@",
      "startEpochMicrosec": "@timestamp@",
      "lastEpochMicrosec": "@timestamp@",
      "nfNamingCode": "@type@",
      "nfVendorName": "@vendor@",
      "timeZoneOffset": "+00:00",
      "version": "4.1",
      "stndDefinedNamespace": "3GPP-FaultSupervision",
      "vesEventListenerVersion": "7.2.1"
    },
    "stndDefinedFields": {
      "schemaReference": "https://forge.3gpp.org/rep/sa5/MnS/-/raw/Rel-16/OpenAPI/TS28532_FaultMnS.yaml#components/schemas/NotifyNewAlarm",
      "data": {
        "href": "href1",
        "notificationId": 0,
        "notificationType": "notifyNewAlarm",
        "eventTime": "@eventTime@",
        "systemDN": "xyz",
        "alarmId": "@alarm@",
        "alarmType": "COMMUNICATIONS_ALARM",
        "probableCause": "@alarm@",
        "specificProblem": "@alarm@",
        "perceivedSeverity": "@severity@",
        "backedUpStatus": true,
        "backUpObject": "xyz",
        "trendIndication": "MORE_SEVERE",
        "thresholdInfo": {
          "observedMeasurement": "new",
          "observedValue": 123.1
        },
        "correlatedNotifications": [],
        "stateChangeDefinition": [{ "operational-state": "DISABLED" }],
        "monitoredAttributes": {
          "interface": "@interface@"
        },
        "proposedRepairActions": "Call the police!",
        "additionalText": "O-RAN Software Community OAM",
        "additionalInformation": {
          "eventTime": "@eventTime@",
          "equipType": "@type@",
          "vendor": "@vendor@",
          "model": "@model@"
        },
        "rootCauseIndicator": false
      },
      "stndDefinedFieldsVersion": "1.0"
    }
  }
}
```


## Sources:

  - https://forge.3gpp.org/rep/sa5/MnS/-/tree/Rel-18
  - https://wiki.o-ran-sc.org/display/OAM/OAM+Architecture
  - https://wiki.onap.org/display/DW/SDN-R:+PNF+registration
  - https://gerrit.o-ran-sc.org/r/gitweb?p=oam.git;a=tree;f=code/client-scripts-ves-v7;h=7ab5e1ce30d5d12be1dec53e54746b234bc76e69;hb=refs/heads/g-release
  - https://netopeer.liberouter.org/doc/sysrepo/master/html/index.html
  - https://gitlab.eurecom.fr/mosaic5g/flexric

## Abbreviations

  - OAI...OpenAirInterface
  - CM....Configuration Management
  - FM....Fault Management
  - PM....Performance Management
  - CU....Centralized Unit
  - CUCP..Centralized Unit Control Plane
  - CUUP..Centralized Unit User Plane
  - DU....Distributed Unit