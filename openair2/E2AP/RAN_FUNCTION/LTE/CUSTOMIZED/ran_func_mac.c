#include "ran_func_mac.h"
#include "openair2/E2AP/flexric/test/rnd/fill_rnd_data_mac.h"
#include <assert.h>

static
const int mod_id = 0;
static
const int CC_id = 0;

void read_mac_sm(void* data)
{
  assert(data != NULL);

  mac_ind_data_t* mac = (mac_ind_data_t*)data;
  //fill_mac_ind_data(mac);

  mac->msg.tstamp = time_now_us();

  const size_t num_ues = RC.mac[mod_id]->UE_info.num_UEs;

  mac->msg.len_ue_stats = num_ues;
  if (mac->msg.len_ue_stats > 0) {
    mac->msg.ue_stats = calloc(mac->msg.len_ue_stats, sizeof(mac_ue_stats_impl_t));
    assert(mac->msg.ue_stats != NULL && "Memory exhausted" );
  }

  const UE_list_t* ue_list = &RC.mac[mod_id]->UE_info.list;
  size_t i = 0; //TODO
  for (int ue_id = ue_list->head; ue_id >= 0; ue_id = ue_list->next[ue_id]) {
    const eNB_UE_STATS* uestats = &RC.mac[mod_id]->UE_info.eNB_UE_stats[CC_id][ue_id];
    const UE_sched_ctrl_t *sched_ctrl = &RC.mac[mod_id]->UE_info.UE_sched_ctrl[ue_id];
    const UE_TEMPLATE *template = &RC.mac[mod_id]->UE_info.UE_template[CC_id][ue_id];
    mac_ue_stats_impl_t* rd = &mac->msg.ue_stats[i];

    rd->frame = 0; // TODO
    rd->slot = 0;  // TODO

    rd->dl_aggr_tbs = uestats->total_pdu_bytes;
    rd->ul_aggr_tbs = uestats->total_ulsch_TBS;

    rd->dl_curr_tbs = uestats->TBS;
    rd->dl_sched_rb = uestats->rbs_used;

    rd->ul_curr_tbs = uestats->ulsch_TBS;
    rd->ul_sched_rb = uestats->rbs_used_rx;

    rd->rnti = uestats->crnti;
    rd->dl_aggr_prb = uestats->total_rbs_used;
    rd->ul_aggr_prb = uestats->total_rbs_used_rx;
    rd->dl_aggr_retx_prb = uestats->rbs_used_retx;
    rd->ul_aggr_retx_prb = uestats->rbs_used_retx_rx;

    rd->dl_aggr_bytes_sdus = uestats->total_sdu_bytes;
    uint64_t ul_sdu_bytes = 0;
    for (int i = 0; i < NB_RB_MAX; ++i)
      ul_sdu_bytes += uestats->num_bytes_rx[i];
    rd->ul_aggr_bytes_sdus = ul_sdu_bytes;

    rd->dl_aggr_sdus = uestats->num_mac_sdu_tx;
    rd->ul_aggr_sdus = uestats->num_mac_sdu_rx;

    rd->pusch_snr = sched_ctrl->pusch_snr[CC_id];
    rd->pucch_snr = sched_ctrl->pucch1_snr[CC_id];

    rd->wb_cqi = sched_ctrl->dl_cqi[CC_id];
    rd->dl_mcs1 = uestats->dlsch_mcs1;
    rd->dl_bler = 0; // TODO
    rd->ul_mcs1 = uestats->ulsch_mcs1;
    rd->ul_bler = sched_ctrl->pusch_bler[CC_id];
    rd->dl_mcs2 = uestats->dlsch_mcs2;
    rd->ul_mcs2 = uestats->ulsch_mcs2;
    rd->phr = template->phr_info;

    const uint32_t bufferSize = template->estimated_ul_buffer - template->scheduled_ul_bytes;
    rd->bsr = bufferSize;

    const size_t numDLHarq = 4;
    rd->dl_num_harq = numDLHarq;
    for (uint8_t j = 0; j < numDLHarq; ++j)
      rd->dl_harq[j] = uestats->dlsch_rounds[j];
    rd->dl_harq[numDLHarq] = uestats->dlsch_errors;

    const size_t numUlHarq = 4;
    rd->ul_num_harq = numUlHarq;
    for (uint8_t j = 0; j < numUlHarq; ++j)
      rd->ul_harq[j] = uestats->ulsch_rounds[j];
    rd->ul_harq[numUlHarq] = uestats->ulsch_errors;

    ++i;
  }
}

void read_mac_setup_sm(void* data)
{
  assert(data != NULL);
  assert(0 !=0 && "Not supported");
}

sm_ag_if_ans_t write_ctrl_mac_sm(void const* data)
{
  assert(data != NULL);
  assert(0 !=0 && "Not supported");
}

