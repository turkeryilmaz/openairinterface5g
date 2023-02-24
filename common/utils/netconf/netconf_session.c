#include "netconf_session.h"
#include "netconf_log.h"

sr_conn_ctx_t *netconf_session_connection = 0;
sr_session_ctx_t *netconf_session_running = 0;
sr_session_ctx_t *netconf_session_operational = 0;
const struct ly_ctx *netconf_session_context = 0;
sr_subscription_ctx_t *netconf_session_subscription = 0;

int netconf_session_init(void) {
    int rc = SR_ERR_OK;
    
    /* connect to sysrepo */
    rc = sr_connect(0, &netconf_session_connection);
    if(SR_ERR_OK != rc) {
        netconf_log_error("sr_connect failed");
        goto netconf_session_init_cleanup;
    }

    /* start session */
    rc = sr_session_start(netconf_session_connection, SR_DS_OPERATIONAL, &netconf_session_operational);
    if (rc != SR_ERR_OK) {
        netconf_log_error("sr_session_start operational failed");
        goto netconf_session_init_cleanup;
    }

    rc = sr_session_start(netconf_session_connection, SR_DS_RUNNING, &netconf_session_running);
    if (rc != SR_ERR_OK) {
        netconf_log_error("sr_session_start running failed");
        goto netconf_session_init_cleanup;
    }

    /* get context */
    netconf_session_context = sr_acquire_context(netconf_session_connection);
    if(netconf_session_context == 0) {
        netconf_log_error("sr_acquire_context failed");
        goto netconf_session_init_cleanup;
    }

    return 0;

netconf_session_init_cleanup:
    return 1;
}

void netconf_session_free(void) {
    if(netconf_session_subscription) {
        sr_unsubscribe(netconf_session_subscription);
    }

    sr_release_context(netconf_session_connection);

    sr_session_stop(netconf_session_operational);
    sr_session_stop(netconf_session_running);

    sr_disconnect(netconf_session_connection);

    netconf_session_connection = 0;
    netconf_session_running = 0;
    netconf_session_operational = 0;
    netconf_session_context = 0;
}
