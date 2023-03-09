#include "ves_demo_alarm.h"
#include "netconf_log.h"
#include "o1config.h"
#include "ves.h"
#include <pthread.h>

typedef struct ves_demo_alarm_params {
    int interval;
} ves_demo_alarm_params_t;

static pthread_t ves_demo_alarm_thread;

static void *ves_demo_alarm_routine(void *arg);

int ves_demo_alarm_start(o1_config_t *config) {
    ves_demo_alarm_params_t *params = (ves_demo_alarm_params_t *)malloc(sizeof(ves_demo_alarm_params_t));
    if(params == 0) {
        netconf_log_error("malloc failed");
        goto failed;
    }

    params->interval = config->demo_alarming_interval;


    int rc = pthread_create(&ves_demo_alarm_thread, 0, ves_demo_alarm_routine, params);
    if(rc != 0) {
        netconf_log_error("pthread_create() failed");
        goto failed;
    }

    return 0;

failed:
    free(params);
    return 1;
}

static void *ves_demo_alarm_routine(void *arg) {
    ves_demo_alarm_params_t *params = (ves_demo_alarm_params_t *)arg;

    int alarm_type = 0;
    while(1) {
        sleep(params->interval);
        alarm_type++;
        if(alarm_type >= 2) {
            alarm_type = 0;
        }

        ves_alarm_t alarm = {
            .alarm_name = "demoAlarm",
            .vendor = "highstreet",
            .model = "oai"
        };
        if(alarm_type) {
            netconf_log("set alarm");
            alarm.severity = "URGENT";
            alarm.alarm_type = "ALARM";
        }
        else {
            netconf_log("clear alarm");
            alarm.severity = "NONE";
            alarm.alarm_type = "NON-ALARM";
        }


        netconf_log("alarm");
        int rc = ves_pnf_alarm_execute(&alarm);
        if(rc != 0) {
            netconf_log_error("ves_pnf_alarm_execute() failed");
        }
    }

    free(arg);

    return 0;
}
