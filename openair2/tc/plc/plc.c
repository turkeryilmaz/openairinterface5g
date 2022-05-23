#include "plc.h"
#include "../time/time.h"
#include "../alg_ds/alg/alg.h" 
#include "../alg_ds/alg/find.h" 

#include "../alg_ds/alg/accumulate.h" 
#include "../alg_ds/ds/seq_container/seq_generic.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#define MTU 1514

typedef struct pair_tstamp_pkt_e 
{
  uint32_t pkt_bytes;
  int64_t tstamp;
} pair_tstamp_pkt_t;


plc_t* plc_init(uint32_t drop_rate_kbps, uint32_t dev_rate_kbps, uint32_t time_window_us, queue_t* dst_q, queue_t* dev_q)  
{
  plc_t* plc = calloc(1,sizeof(plc_t));
  assert(plc != NULL);

  mtr_init(&plc->m, time_window_us);

  plc->dropped_pkts = 0;
  plc->active = 1;

  plc->drop_rate_kbps = drop_rate_kbps;
  plc->dev_rate_kbps = dev_rate_kbps;

  plc->dst_q = dst_q;
  plc->dev_q = dev_q;

  return plc;
} 

void plc_free(plc_t* plc)
{
  assert(plc != NULL);
//  void* value_semantic = NULL;
//  seq_free(&plc->ts_pkt, value_semantic);
  mtr_free(&plc->m);
  free(plc);
}

void plc_bytes_fwd(plc_t* plc, uint32_t bytes)
{
  assert(plc != NULL);
  assert(bytes != 0);

  int64_t const tstamp = 0; // The tstamp will be taken from the meter
  mtr_push_back(&plc->m, tstamp, bytes);

//  pair_tstamp_pkt_t tmp = {.pkt_bytes = bytes, .tstamp = time_now_us() };
//  seq_push_back(&plc->ts_pkt, (uint8_t*)&tmp, sizeof(pair_tstamp_pkt_t));
}
/*
typedef struct pair_s
{
  const int64_t now;
  const uint32_t time_window_us;
} pair_t;

static
bool pkts_not_in_time_window(const void* p_v, const void* p_ts_pkt_v)
{
   pair_t* p = (pair_t*)p_v; 
   const pair_tstamp_pkt_t* p_ts_pkt = (pair_tstamp_pkt_t*) p_ts_pkt_v; 

   const int64_t t_passed =  p->now - p_ts_pkt->tstamp; 
   if(t_passed < p->time_window_us)
     return true;
   return false;
}

static
uint32_t calculate_rate_kbps(plc_t* plc, uint32_t pkt_bytes)
{
  void* it_start = seq_front(&plc->ts_pkt);
  void* it_end = seq_end(&plc->ts_pkt);
  uint32_t rate_kbps = pkt_bytes;
  while(it_start != it_end){
    rate_kbps += ((pair_tstamp_pkt_t*)it_start)->pkt_bytes; 
    it_start = seq_next(&plc->ts_pkt, it_start);
  }
  return rate_kbps;
}
*/

plc_act_e plc_action(plc_t* plc, uint32_t bytes)
{
  assert(plc != NULL);
  assert(bytes != 0);

  /*
  pair_t p = {.now = time_now_us(), .time_window_us = plc->time_window_us};
  void* it = find_if_r(&plc->ts_pkt, &p, pkts_not_in_time_window);
  seq_erase(&plc->ts_pkt, seq_front(&plc->ts_pkt), it);
  const uint32_t rate_kbps = calculate_rate_kbps(plc,bytes);
  printf("Policer accumulated rate %u and drop_rate = %d in the time window = %d \n", rate_kbps, plc->drop_rate_kbps * 1000, plc->time_window_us);
*/
  float const rate_kbps = mtr_bndwdth_kbps(&plc->m);
  printf("Policer rate = %f\n", rate_kbps);
//  if(rate_kbps > plc->drop_rate_kbps * 1000){
  if(rate_kbps > plc->drop_rate_kbps){
    printf("Policer Dropping packet \n");
    ++plc->dropped_pkts;
    return PLC_DROP;
  }

//  if(rate_kbps > plc->dev_rate_kbps * 1000){
  if(rate_kbps > plc->dev_rate_kbps){
    return PLC_DEVIATE;
  }

  return PLC_PASS;
}




void plc_set_dev_q(plc_t* plc, queue_t* dev_q)
{
  assert(plc != NULL);
  assert(dev_q != NULL);
  plc->dev_q = dev_q;
}


tc_plc_t plc_stat(plc_t* plc)
{
  assert(plc != NULL);

  tc_plc_t ans = {0}; 

  ans.active = plc->active;
  ans.dev_id = plc->dev_q->id; 
  ans.dst_id = plc->dst_q->id;
  //
  ans.drp.dropped_pkts = plc->dropped_pkts;

  ans.mtr.time_window_ms = plc->m.avg_wnd.wnd_ms;
  ans.mtr.bnd_flt = plc->m.avg_wnd.avg;

  ans.mrk.marked_pkts = 0; // ToDO?? 

  return ans;
/*
  uint32_t id;
  // RFC 2475

  // meter
  tc_mtr_t mtr;

  // dropper
  tc_drp_t drp;

  // marker
  tc_mrk_t mrk;

  float max_rate_kbps;
  uint32_t active;
  uint32_t dst_id;
  uint32_t dev_id;
*/

}

void plc_mod(plc_t* p, queue_t** dev_q, tc_mod_ctrl_plc_t const* ctrl)
{
  assert(p != NULL);
  assert(ctrl != NULL);
  
  p->active = ctrl->active;
  p->dev_rate_kbps = ctrl->dev_rate_kbps;
  p->drop_rate_kbps = ctrl->drop_rate_kbps;
  p->dev_q = *dev_q;

  printf("Policer modified successfully \n");
 
}

