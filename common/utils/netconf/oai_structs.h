#pragma once

#include <stdint.h>

#define MAND
#define RO
#define RW
#define KEY

typedef struct oai_alarmRecords_thresholdInfo {
    RO MAND char *measurementType;
    RO MAND char *direction;
    RO      char *thresholdLevel;
    RO      char *thresholdValue;
    RO      char *hysteresis;
} oai_alarmRecords_thresholdInfo_t;

typedef struct oai_alarmRecords {
    RW KEY  char *alarmId; 
    RO MAND char *objectInstance;
    RO MAND char *notificationId;        //int32_t
    RO      char *alarmRaisedTime;       //yang:date-and-time
    RO      char *alarmChangedTime;      //yang:date-and-time
    RO      char *alarmClearedTime;      //yang:date-and-time
    RO      char *alarmType;             //eventType
    RO      char *probableCause;
    RO      char *specificProblem;
    RW      char *perceivedSeverity;     //severity-level
    RO      char *backedUpStatus;
    RO      char *backUpObject;
    RO      char *trendIndication;
    RO      oai_alarmRecords_thresholdInfo_t *thresholdInfo;
    RO      char *stateChangeDefinition;
    RO      char *monitoredAttributes;
    RO      char *proposedRepairActions;
    RO      char *additionalText;
    // RO      char *additionalInformation;   <anydata>
    RO      char *rootCauseIndicator;
    RO      char *ackTime;               //yang:date-and-time
    RW      char *ackUserId;
    RW      char *ackSystemId;
    RW      char *ackState;
    RW      char *clearUserId;
    RW      char *clearSystemId;
    RO      char *serviceUser;
    RO      char *serviceProvider;
    RO      char *securityAlarmDetector;
} oai_alarmRecords_t;

typedef struct oai_BWP_ThresholdMonitor {
    RW KEY  char *id;
    RW      struct attributes {
        RW MAND char *bwpContext;
        RW MAND char *isInitialBwp;
        RW MAND char *subCarrierSpacing;
        RW MAND char *cyclicPrefix;
        RW MAND char *startRB;
        RW MAND char *numberOfRBs;
    };
} oai_BWP_ThresholdMonitor_t;

typedef struct oai_BWP {
    RW KEY  char *id;
    RW      struct attributes {
        RW MAND char *bwpContext;
        RW MAND char *isInitialBwp;
        RW MAND char *subCarrierSpacing;
        RW MAND char *cyclicPrefix;
        RW MAND char *startRB;
        RW MAND char *numberOfRBs;
    };
    RW      oai_BWP_ThresholdMonitor_t *ThresholdMonitor; 

} oai_BWP_t;

