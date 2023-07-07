#ifndef RAN_FUNC_SM_RLC_READ_WRITE_AGENT_H
#define RAN_FUNC_SM_RLC_READ_WRITE_AGENT_H

#include "openair2/E2AP/flexric/src/agent/e2_agent_api.h"
#include "common/ran_context.h"
#include "openair2/LAYER2/NR_MAC_gNB/mac_proto.h"
#include "openair2/LAYER2/nr_rlc/nr_rlc_oai_api.h"
#include "openair2/E2AP/flexric/src/util/time_now_us.h"

size_t get_number_drbs_per_ue(NR_UE_info_t * const UE);

nr_rlc_statistics_t* rlc_stat_per_ue(NR_UE_info_t * const UE);

uint32_t num_act_rb(NR_UEs_t* const UE_info);

void active_avg_to_tx(NR_UEs_t* const UE_info);

void read_rlc_sm(void*);

void read_rlc_setup_sm(void* data);

sm_ag_if_ans_t write_ctrl_rlc_sm(void const* data);

#endif

