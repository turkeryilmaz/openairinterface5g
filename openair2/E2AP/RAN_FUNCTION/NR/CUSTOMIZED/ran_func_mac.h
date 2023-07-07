#ifndef SM_MAC_READ_WRITE_AGENT_H
#define SM_MAC_READ_WRITE_AGENT_H

#include "openair2/E2AP/flexric/src/agent/e2_agent_api.h"
#include "common/ran_context.h"
#include "openair2/LAYER2/NR_MAC_gNB/mac_proto.h"
#include "openair2/E2AP/flexric/src/util/time_now_us.h"

size_t get_number_connected_ues(void);

NR_UE_info_t * connected_ues_list(void);

void read_mac_sm(void*);

void read_mac_setup_sm(void*);

sm_ag_if_ans_t write_ctrl_mac_sm(void const*);

#endif
