#pragma once

#include <libyang/libyang.h>
#include <sysrepo.h>

extern sr_conn_ctx_t            *netconf_session_connection;
extern sr_session_ctx_t         *netconf_session_running;
extern sr_session_ctx_t         *netconf_session_operational;
extern const struct ly_ctx      *netconf_session_context;
extern sr_subscription_ctx_t    *netconf_session_subscription;

int netconf_session_init(void);
void netconf_session_free(void);
