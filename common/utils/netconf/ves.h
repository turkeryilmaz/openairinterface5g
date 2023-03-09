#pragma once

#include <stdbool.h>

// common data
typedef struct ves_header {
    char *hostname;
    char *nf_naming_code;
    char *nf_vendor_name;
    char *pnf_id;
} ves_header_t;

// ves endpoint configuration 
typedef struct ves_config {
    char *url;       // endpoint url e.g. https://10.20.30.40/enventlistener/v7
    char *username;  // basic auth username
    char *password;  // basic auth password
} ves_config_t;

//pnfRegistration data
typedef struct ves_registration_data {
    char *pnf_username;
    char *pnf_password;
    char *pnf_ip_v4_address;
    char *pnf_ip_v6_address;
    int pnf_port;
    char *mac_address;
    char *vendor;
    char *model;
    char *type;
    bool is_tls;
} ves_registration_data_t;

//
typedef struct ves_file_ready {
    char *filename;
    char *filelocation;
} ves_file_ready_t;

//
typedef struct ves_alarm {
    char *alarm_name;
    char *severity;
    char *alarm_type;
    char *vendor;
    char *model;
} ves_alarm_t;


#define DOMAIN_PNFREGISTRAION           "pnfRegistration"
#define DOMAIN_PNFALARMING              "fault"
#define DOMAIN_HEARTBEAT                "heartbeat"
#define DOMAIN_PMDATA                   "measurement"
#define EVENTTYPE_PNFREGISTRATION       "EventType5G"
#define EVENTTYPE_PNFALARMING           "O_RAN_COMPONENT_Alarms"
#define EVENTTYPE_PMDATA                "Notification-gnb_Nokia-FileReady"
#define PRIORITY_PNFREGISTRATION        "Low"
#define PRIORITY_PNFALARMING            "Low"
#define PRIORITY_PMDATA                 "Low"

int ves_init(ves_config_t *config, ves_header_t *header);
void ves_free();

int ves_pnf_registration_execute(ves_registration_data_t *data);
int ves_pnf_pmdata_fileready_execute(ves_file_ready_t* ves_data);
int ves_pnf_alarm_execute(ves_alarm_t* ves_alarm);
