#pragma once

#include <sysrepo.h>

int netconf_init(void);
int netconf_free(void);

int netconf_disable_nacm(sr_session_ctx_t *session);
int netconf_demo_populate(void);
int netconf_demo_subscribe_edit_change();
int netconf_demo_subscribe_operational();
