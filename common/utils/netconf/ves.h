#pragma once

#include <stdbool.h>

typedef struct ves_instance {
    //private id
    int _seq_id;

    //common data
    char *domain;
    char *event_type;
    char *hostname;
    int port;
    char *priority;

    //pnfRegistration data
    char *username;
    char *password;
    char *device_ip_v4_address;
    char *device_ip_v6_address;
    int device_port;
    bool is_tls;
} ves_instance_t;

typedef struct ves_pmdata {
    
} ves_pmdata_t;
typedef struct ves_alarm {

} ves_alarm_t;

int ves_pnf_registration_execute(ves_instance_t *ves_instance);
int ves_pnf_pmdata_download_execute(ves_instance_t *ves_instance, ves_pmdata_t* ves_pmdata );
int ves_pnf_alarm_execute(ves_instance_t *ves_instance, ves_alarm_t* ves_alarm);