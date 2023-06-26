#ifndef LTE_RAN_FUNC_SM_MAC_READ_WRITE_AGENT_H
#define LTE_RAN_FUNC_SM_MAC_READ_WRITE_AGENT_H

#include "openair2/E2AP/flexric/src/agent/e2_agent_api.h"
#include "common/ran_context.h"
#include "openair2/LAYER2/MAC/mac.h"
#include "openair2/E2AP/flexric/src/util/time_now_us.h"

void read_mac_sm(void*);

void read_mac_setup_sm(void*);

sm_ag_if_ans_t write_ctrl_mac_sm(void const*);

#endif
