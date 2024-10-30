/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

/*! \file       gNB_qos_aware_scheduler_dlsch.c
 * \brief       procedures related to gNB scheduling for the DLSCH transport channel
 * \author      SriHarsha Korada
 * \date        2023
 * \email:      sriharsha.korada@iis.fraunhofe.de
 * \version     1.0
 * @ingroup     _mac

 */

#include "common/utils/nr/nr_common.h"
/*MAC*/
#include "NR_MAC_COMMON/nr_mac.h"
#include "NR_MAC_COMMON/nr_mac_extern.h"
#include "LAYER2/NR_MAC_gNB/mac_proto.h"
#include "LAYER2/NR_MAC_gNB/gNB_qos_aware_scheduler_dlsch.h"

/*NFAPI*/
#include "nfapi_nr_interface.h"
/*TAG*/
#include "NR_TAG-Id.h"

/*
Algorithm:

In the propsed scheduler, it allocates resources to each logical channel of a user depending on its QoS requirements. 
The main difference of this scheduler is that the resources are assigned to each logical channel of a User of all the active Users 
jointly which means depending of the priority of a logical channel irrespective of the User it belongs to are assigned resources first 
rather than allocating resources to all logical channels of a User. The scheduler takes in to account the characteristics of each  QoS 
flow, i.e the resource type (guaranteed flow or non- guaranteed flow), priority level

The scheduler as most of them have two phases
- Time Domain Scheduler:
   - In this phase, scheduler selects the set of active users that are requesting resources and separate them to the granularity level 
     of a logical channel.
   - The logical channels of all the active users are divided in to 2 sets depending on the resource type, i.e guaranteed flow or 
     non-guaranteed flow and GBR based bearers are always given the highest priority comparted to Non GBR bearers as these are related
      to applications that might be sensitive to delay 
   - Within each set, the logical channels are sorted depending on different requirements, i.e 
     (Note: The Weighting method that is used can be varied in the future, for example can include packetDelayBudget, amount of time 
     the packet is waiting in the buffer etc)
     - The GBR based logical channels are sorted depending on priority and tocket bucket parameters
     - The Non-GBR based flows are sorted depending on number of bytes allocated until that time.
   - In order to avoid the startvation problem, 
      - for GBR bearers, tocken bucket algorithm is used to determine which GBR bearer must be serviced first
      - for Non GBR bearers, PF algorithm is used to determine the fairness among the Non GBR bearers 

- Frequency Domain Scheduler:
  - This phase is responsible to distribute the available resources among different bearers that are sorted from the previous phase. 
    This phase is again divided in to preprocessing (main) and postprocessing phase.
  - In the preprocessing phase, the distribution of resources is done under the assumption that all the available resources are continuous. 
    Basically this step determines how many bytes from this logical channel need to be prioritized first depending on  available resources
    - In this step, the FD scheduler allocates resources firstly to the GBR bearers and after serving all the GBR bearers and if still
      resources are left then the Non GBR bearers are serviced
    - The first phase takes care of startvation issues, so in this phase only resources are scheduled as per the sorted order

  - In the post processing phase, depending on the number of actual continuous resources, the number of bytes that should be prioritized are 
    updated. So if number of continuous resources are less than what is assigned in pre-processing phase, then accordingly data from 
    the logical channe are sent until all resources allocated for that user are utilized

*/

void print_ordered_lc(LCIDsched_t *lc_sched) {

  LOG_D(NR_MAC, "Order of logical channels of UEs in which resources should be reserved in pre scheduler \n");

  LC_iterator(lc_sched, lc) {
    if (lc->UE == NULL)
        break;
    
    NR_UE_info_t *UE_curr = lc->UE;
    uint16_t rnti = UE_curr->rnti;
    uint8_t lcnum = lc->curr_lcid;
    uint8_t uenum = lc->curr_ue;

    LOG_D(NR_MAC, "[UE %.5x]The lcid %d of UE %d is should be scheduled %s \n", rnti, lcnum, uenum,  (lc_sched == lc) ? "first" : "next");
  }

}

void print_ordered_ue(UEsched_t *UElist) {
  LOG_D(NR_MAC, "Order of of UEs in which resources should be allocated until resources are available\n");
  
  UEsched_t *iterator = UElist;
  while (iterator->UE != NULL) {
    rnti_t rnti = iterator->UE->rnti;
    LOG_D(NR_MAC, " The UE with rnti %.5x should be allocated resources (order id %ld) %s \n", rnti, iterator->order_id, iterator == UElist ? "first" : "next");
    iterator++;
  }
}

int lc_comparator(const void *p, const void *q)
{
  LCIDsched_t *sched_lc1 = (LCIDsched_t *)p;
  LCIDsched_t *sched_lc2 = (LCIDsched_t *)q;

  NR_UE_sched_ctrl_t *sched_ctrl1 = &sched_lc1->UE->UE_sched_ctrl;
  NR_UE_sched_ctrl_t *sched_ctrl2 = &sched_lc2->UE->UE_sched_ctrl;
  
  nr_lc_config_t *lc1 = seq_arr_at(&sched_ctrl1->lc_config, sched_lc1->curr_lcid);
  nr_lc_config_t *lc2 = seq_arr_at(&sched_ctrl2->lc_config, sched_lc2->curr_lcid);
  
  if (lc1->lcid < 4 && lc2->lcid < 4) {
    /* both are SRBs*/
    return lc1->priority - lc2->priority;
  } else if (lc1->lcid >= 4 && lc2->lcid >= 4) {
    /* both are DRBs*/
    if (lc1->guaranteed_bitrate > 0 && lc2->guaranteed_bitrate > 0) 
      return sched_lc2->coef - sched_lc1->coef;

    else if (lc1->guaranteed_bitrate == 0 && lc2->guaranteed_bitrate == 0) 
      return sched_lc1->coef < sched_lc2->coef;

    else {
      float tmp1 = lc1->guaranteed_bitrate == 0 ? (__FLT_MIN__)*(-1) : sched_lc1->coef;
      float tmp2 = lc2->guaranteed_bitrate == 0 ? (__FLT_MIN__)*(-1) : sched_lc2->coef;
      return tmp2 - tmp1;
    }
  } else {
    /* either of one is SRB */
    float tmp1 = lc1->lcid < 4 ? 0 : 1;
    float tmp2 = lc2->lcid < 4 ? 0 : 1;
    return tmp1 - tmp2;
  }
}

int ue_comparator(const void *p, const void *q)
{
  UEsched_t *sched_ue1 = (UEsched_t *)p;
  UEsched_t *sched_ue2 = (UEsched_t *)q;
  
  return sched_ue2->order_id - sched_ue1->order_id;
}

int nr_find_nb_bytes(int bytes,
                     int oh_bytes,
                     uint32_t *tbs,
                     uint16_t *rbsize,
                     uint16_t nb_rb_max,
                     uint16_t nb_rb_min,
                     NR_sched_pdsch_t *sched_pdsch,
                     NR_tda_info_t *tda_info)
{
  bool status = false;
  int num_bytes = 0;
  while (!status && (bytes + oh_bytes) > 0) {
    num_bytes = bytes--;
    status = nr_find_nb_rb(sched_pdsch->Qm,
                           sched_pdsch->R,
                           1,
                           sched_pdsch->nrOfLayers,
                           tda_info->nrOfSymbols,
                           sched_pdsch->dmrs_parms.N_PRB_DMRS * sched_pdsch->dmrs_parms.N_DMRS_SLOT,
                           num_bytes + oh_bytes,
                           nb_rb_min,
                           nb_rb_max,
                           tbs,
                           rbsize);
    LOG_D(NR_MAC, "status = %d, num_bytes_lcid = %hu\n", status, num_bytes);
  }
  return num_bytes;
}

void qos_aware_post_processor(int prbs_available, UEsched_t *ue_iterator, LCIDsched_t *lc_sched, int oh)
{
  NR_UE_info_t *ue = ue_iterator->UE;
  NR_UE_sched_ctrl_t *sched_ctrl = &ue->UE_sched_ctrl;
  NR_sched_pdsch_t *sched_pdsch = &sched_ctrl->sched_pdsch;
  NR_tda_info_t *tda_info = &sched_pdsch->tda_info;
  int num_conprbs = prbs_available;

  /* reset */
  ue_iterator->num_rbs_sched = 0;

  uint16_t tmp_bytes = 0;
  uint16_t tmp_rbs = 0;
  uint16_t tmp_tbs = 0 ;
  bool status_oh = false;

  LC_iterator(lc_sched, lc)
  {
    if (lc->UE == NULL)
      break;

    uint8_t lcnum = lc->curr_lcid;
    nr_lc_config_t *c = seq_arr_at(&sched_ctrl->lc_config, lcnum);

    if (lc->UE != ue_iterator->UE && lc->curr_lcid < 4)
      continue;

    uint16_t num_rbs_reserved_lc = lc->num_rbs_lcid_allocated;
    uint16_t allocated_bytes_lc = lc->allocated_bytes;

    if (num_rbs_reserved_lc <= num_conprbs) {
      tmp_bytes += allocated_bytes_lc;
      tmp_rbs += num_rbs_reserved_lc;
      tmp_tbs += lc->tbs;
      num_conprbs -= num_rbs_reserved_lc;
      ue_iterator->num_rbs_sched += num_rbs_reserved_lc;
      continue;
    } else if (num_conprbs <= 0) {
      LOG_D(NR_MAC,
            "[UE %.5x]%d rbs previously allocated in prescheduling for LC %d are reset because resources are not available \n",
            ue->rnti,
            num_rbs_reserved_lc,
            c->lcid
            );
      memset(lc, 0, sizeof(LCIDsched_t));
      continue;
    }

    /* reset the lc pre allocation */
    lc->num_rbs_lcid_allocated = 0;
    lc->allocated_bytes = 0;
    lc->tbs = 0;

    /* restructure the lc allocation */
    uint32_t TBS = 0;
    uint16_t rbsize;
    int num_bytes_lcid = 0;
    LOG_D(NR_MAC,
          "[UE %.5x]LC %d allocation in prescheduling step are: %d prbs are allocated with %d bytes untill this logical channel",
          ue->rnti,
          lc->curr_lcid,
          num_rbs_reserved_lc,
          allocated_bytes_lc);

    num_bytes_lcid =
        nr_find_nb_bytes(tmp_bytes + allocated_bytes_lc, status_oh ? 0: oh, &TBS, &rbsize, prbs_available, tmp_rbs, sched_pdsch, tda_info);
    lc->allocated_bytes = num_bytes_lcid - tmp_bytes;
    sched_ctrl->rlc_status[lcnum].prioritized_bytes_in_buffer = lc->allocated_bytes;
    lc->num_rbs_lcid_allocated = rbsize - tmp_rbs;
    lc->tbs = TBS - tmp_tbs;
    sched_pdsch->lc_data_thru[lcnum] = lc->tbs;

    AssertFatal(num_bytes_lcid - tmp_bytes > 0, "Should not be negative\n");
    AssertFatal(rbsize - tmp_rbs > 0, "should not be negative\n");

    ue_iterator->num_rbs_sched += lc->num_rbs_lcid_allocated;
    status_oh = true;
  }
}

int nr_get_num_prioritized_bytes(int prbs_available, UEsched_t *ue_iterator, LCIDsched_t *lc_sched, int oh)
{
  NR_UE_info_t *ue = ue_iterator->UE;
  NR_UE_sched_ctrl_t *sched_ctrl = &ue->UE_sched_ctrl;
  uint32_t total_prioritized_bytes = 0;
  
  LOG_D(NR_MAC, "[UE %.5x]Number of PRBS available for scheduling are %d and number of PRBs reserved are %d \n", ue->rnti, prbs_available, ue_iterator->num_rbs_sched);
  
  if (prbs_available < ue_iterator->num_rbs_sched) {
    LOG_D(NR_MAC, "Performing post processing of the scheduling\n");
    qos_aware_post_processor(prbs_available, ue_iterator, lc_sched, oh);
  }

  for (uint8_t i = 0; i < seq_arr_size(&sched_ctrl->lc_config); i++) {
    nr_lc_config_t *c = seq_arr_at(&sched_ctrl->lc_config, i);
    int j = c->lcid;
    total_prioritized_bytes += sched_ctrl->rlc_status[j].prioritized_bytes_in_buffer;
    LOG_D(NR_MAC, "prioritized bytes in lcid %d is %d\n", j, sched_ctrl->rlc_status[j].prioritized_bytes_in_buffer);
  }
  return total_prioritized_bytes;
}

/* -------------------------------------------------------------------------------------------------------------------------------------- */

void fill_lc_list(NR_UE_info_t *UE, frame_t frame, slot_t slot, int *lc_currid, int ue_currid, LCIDsched_t *lc_sched)
{
  NR_UE_sched_ctrl_t *sched_ctrl = &UE->UE_sched_ctrl;
  float weight_lc;

  for (uint8_t lc_idx = 0; lc_idx < seq_arr_size(&sched_ctrl->lc_config); lc_idx++) {
    /* Check DL buffer and skip this lcid in this UE if no bytes */
    nr_lc_config_t *c = seq_arr_at(&sched_ctrl->lc_config, lc_idx);
    uint8_t lcid = c->lcid;

    uint32_t num_bytes = sched_ctrl->rlc_status[lcid].bytes_in_buffer;

    /* aggregated priority level in each lcid, since since an lcid buffer can contain packets of different qos flows */
    uint64_t aggregated_pl_lcid = c->priority;
    AssertFatal(aggregated_pl_lcid >= 0,
                "Aggregated priority should be non zero positive number. should be some issue in the process_QOSConfig function\n");

    bool is_gbr_flow = c->guaranteed_bitrate > 0 || lcid < 4;
    
    if (is_gbr_flow) {    /* gbr flow */
      weight_lc = (100 - aggregated_pl_lcid) * (c->Bj > 0 ? 1 : -1);
    }
    else {                                                                     /* non-gbr flow */
      /* Calculate Throughput */
      const uint32_t b = UE->mac_stats.dl.current_bytes_lc[lcid];
      const float a = 0.01f;

      UE->dl_thr_lc[lcid] = (1 - a) * UE->dl_thr_lc[lcid] + a * b;
      LOG_D(NR_MAC, "[%d.%d]dl thru for lcid %d is %f\n",
            frame,
            slot,
            lcid,
            UE->dl_thr_lc[lcid]);

      /* calculate coeff */
      const uint8_t Qm = nr_get_Qm_dl(sched_ctrl->sched_pdsch.mcs, UE->current_DL_BWP.mcsTableIdx);
      const uint16_t R = nr_get_code_rate_dl(sched_ctrl->sched_pdsch.mcs, UE->current_DL_BWP.mcsTableIdx);
      uint32_t tbs = nr_compute_tbs(Qm,
                                    R,
                                    1, /* rbSize */
                                    10, /* hypothetical number of slots */
                                    0, /* N_PRB_DMRS * N_DMRS_SLOT */
                                    0 /* N_PRB_oh, 0 for initialBWP */,
                                    0 /* tb_scaling */,
                                    sched_ctrl->sched_pdsch.nrOfLayers) >> 3;

      weight_lc = (100 - aggregated_pl_lcid) * ((float) tbs / UE->dl_thr_lc[lcid]);
    }
    LOG_D(NR_MAC,
          "[UE %04x]weight for lcid %u (which is %s flow) of UE %d is %f with aggregated priority %lu with number of bytes in buffer %u\n",
          UE->rnti,
          lcid,
          is_gbr_flow ? "gbr" : "non-gbr",
          ue_currid,
          weight_lc,
          aggregated_pl_lcid,
          num_bytes
          );

    lc_sched[*lc_currid].coef = weight_lc;
    lc_sched[*lc_currid].UE = UE;
    lc_sched[*lc_currid].curr_lcid = lcid;
    lc_sched[*lc_currid].lcid_priority = aggregated_pl_lcid;
    lc_sched[*lc_currid].curr_ue = ue_currid;
    lc_sched[*lc_currid].lc_remaining_bytes = num_bytes;
    (*lc_currid)++;
  }
}

void nr_update_lc_bucket_parameters(frame_t frame,
                                    sub_frame_t slot,
                                    NR_UE_info_t *UE_curr
                                   ) {
  /* update Bj for all active gbr lcids before LCP procedure in every UE */ 
  uint16_t rnti = UE_curr->rnti;
  NR_UE_sched_ctrl_t *sched_ctrl = &UE_curr->UE_sched_ctrl;
  NR_UE_DL_BWP_t *dl_bwp = &UE_curr->current_DL_BWP;

  float tti_ms = (float)1/(1 << dl_bwp->scs);

  for (int i = 0; i < seq_arr_size(&sched_ctrl->lc_config); i++) {
    nr_lc_config_t *c = seq_arr_at(&sched_ctrl->lc_config, i);
    int lcnum = c->lcid;

    // max amount of data that can be buffered/accumulated in a logical channel buffer
    uint64_t bucketSize_max = c->bucket_size;
    AssertFatal(bucketSize_max >= 0, "negative bucketSize_max %ld, will never schedule lcid %d in UE %05x\n",bucketSize_max, lcnum, rnti);

    /* measure Bj - increment the value of Bj by product PBR  * T */
    float T = tti_ms; 
    int64_t bj = c->Bj; 
    bj += c->guaranteed_bitrate * T * 0.001;

    if (lcnum < 4 || c->guaranteed_bitrate == UINT64_MAX)
        bj = INT64_MAX;

    // bj > max bucket size, set bj to max bucket size, as in ts38.321 5.4.3.1 Logical Channel Prioritization
    c->Bj = min(bj, bucketSize_max);

    LOG_D(NR_MAC, "The value of Bj after incrementing is %ld and incremented value is %f\n", c->Bj, c->guaranteed_bitrate * T * 0.001);
  }
}

bool get_dataavailability_buffers(uint8_t total_active_lcids, LCIDsched_t *lc_sched_active, bool *data_status_lcbuffers)
{
  // check whether there is any data in the rlc buffer corresponding to active lcs
  LC_iterator(lc_sched_active, lc) {
    if (lc->UE == NULL) 
      break;

    uint8_t ue_id = lc->curr_ue;
    uint8_t lcnum = lc->curr_lcid; 
    if (*((data_status_lcbuffers + ue_id*NR_MAX_NUM_LCID) + lcnum)) 
      return true;
  }
  return false;
}

long get_num_bytes_to_req_tti(module_id_t module_id,
                              LCIDsched_t *lc,
                              uint8_t round_id,
                              long *target,
                              uint32_t lcid_buffer_bytes,
                              float tti_ms)
{
  /* Calculates the number of bytes the logical channel should request from the correcponding RLC buffer*/
  NR_UE_info_t *UE_curr = lc->UE;
  uint16_t rnti = UE_curr->rnti;
  NR_UE_sched_ctrl_t *sched_ctrl = &UE_curr->UE_sched_ctrl;
  uint8_t lcnum = lc->curr_lcid;
  nr_lc_config_t *c = seq_arr_at(&sched_ctrl->lc_config, lcnum);
  
  uint64_t pbr = lcnum < 4 ? UINT64_MAX : c->guaranteed_bitrate;
  *target = pbr > 0 ? pbr : lcid_buffer_bytes;

  long num_bytes_requested = 0;

  num_bytes_requested = round_id == 0 ? min(lcid_buffer_bytes, pbr * tti_ms) : lcid_buffer_bytes;
  
  LOG_D(NR_MAC, "[UE %05x][LC %d]number of bytes requested is %li\n", rnti, lcnum, num_bytes_requested);

  return num_bytes_requested;
}

/*
int update_lcconfig(module_id_t module_id,
                    frame_t frame,
                    sub_frame_t slot,
                    LCIDsched_t *lc_sched_active,
                    UEsched_t *ue_sched) 
{ // Depending on the importance of lc among all UEs, the lc order with in each UE must be ordered 

  // reset the lc configuration in each UE 
  UEsched_t *iterator = ue_sched;
  while(iterator->UE!= NULL) {
    memset(iterator->UE->UE_sched_ctrl.dl_lc_ids, 0, iterator->UE->UE_sched_ctrl.dl_lc_num);
    iterator->UE->UE_sched_ctrl.dl_lc_num = 0;
    iterator++;
  }

  // update with new lc information
  LC_iterator(lc_sched_active, lc) {
    if (lc->UE == NULL)
        break;
    
    NR_UE_info_t *UE_curr = lc->UE;
    NR_UE_sched_ctrl_t *sched_ctrl = &UE_curr->UE_sched_ctrl;
    uint8_t lcnum = lc->curr_lcid;

    sched_ctrl->dl_lc_ids[sched_ctrl->dl_lc_num] = lcnum;
    sched_ctrl->dl_lc_num++; 
    LOG_D(NR_MAC, "Total updated number of lc are %d\n", seq_arr_size(&sched_ctrl->lc_config));
  }
  return 0;
}
*/

int allocate_numrbs_per_lc(module_id_t module_id,
                           frame_t frame,
                           sub_frame_t slot,
                           LCIDsched_t *lc, 
                           UEsched_t *ue_sched,
                           uint16_t *remain_prbs,
                           uint16_t *mask,
                           int32_t round_id, 
                           bool *lcids_data_status)
{
  gNB_MAC_INST *mac = RC.nrmac[module_id];
  NR_ServingCellConfigCommon_t *scc = mac->common_channels[0].ServingCellConfigCommon;
  NR_UE_info_t *UE_curr = lc->UE;
  NR_UE_DL_BWP_t *dl_bwp = &UE_curr->current_DL_BWP;
  uint16_t rnti = UE_curr->rnti;
  NR_UE_sched_ctrl_t *sched_ctrl = &UE_curr->UE_sched_ctrl;
  uint8_t lcnum = lc->curr_lcid;
  uint8_t uenum = lc->curr_ue;
  nr_lc_config_t *c = seq_arr_at(&sched_ctrl->lc_config, lcnum);

  LOG_D(NR_MAC,
          "[UE %05x] In round %d, Allocation of resources to logical channel %d of UE %d\n",
          rnti,
          round_id,
          lcnum,
          uenum);

  uint32_t tbs = 0;
  uint16_t rb_size = 0;
  int oh = 0;
  int min_prbs = 0;
  long target = 0;
  uint32_t tbsize_lc = 0;

  const int coresetid = sched_ctrl->coreset->controlResourceSetId;

  /* MCS has been set above */
  NR_sched_pdsch_t *sched_pdsch = &sched_ctrl->sched_pdsch;
  sched_pdsch->time_domain_allocation = get_dl_tda(mac, scc, slot);
  AssertFatal(sched_pdsch->time_domain_allocation >= 0, "Unable to find PDSCH time domain allocation in list\n");

  sched_pdsch->tda_info = get_dl_tda_info(dl_bwp,
                                          sched_ctrl->search_space->searchSpaceType->present,
                                          sched_pdsch->time_domain_allocation,
                                          scc->dmrs_TypeA_Position,
                                          1,
                                          TYPE_C_RNTI_,
                                          coresetid,
                                          false);

  NR_tda_info_t *tda_info = &sched_pdsch->tda_info;

  sched_pdsch->dmrs_parms = get_dl_dmrs_params(scc, dl_bwp, tda_info, sched_pdsch->nrOfLayers);
  sched_pdsch->Qm = nr_get_Qm_dl(sched_pdsch->mcs, dl_bwp->mcsTableIdx);
  sched_pdsch->R = nr_get_code_rate_dl(sched_pdsch->mcs, dl_bwp->mcsTableIdx);

  /* 
     oh bytes should be considered only once while calculating tbsize for a UE, so once considered, for all logical channels the oh is not
     included by default it is false, for the first time it is set and then for next bytes for same or different logical channels in a UE
     it will stay on as true so that overhead bytes should not be considered
 */
  if (!ue_sched->status) {
    /*
        Fix me: currently, the RLC does not give us the total number of PDUs
        awaiting. Therefore, for the time being, we put a fixed overhead of 12
        (for 4 PDUs) and optionally + 2 for TA. Once RLC gives the number of
        PDUs, we replace with 3 * numPDUs
    */
  
    oh = 3 * 4 + 2 * (frame == (sched_ctrl->ta_frame + 10) % 1024);
    min_prbs = 5;
    ue_sched->status = true;
  } 
  lc->overhead = oh;

  float tti_ms = (float)1/(1 << dl_bwp->scs)*0.001;

  long bytes_requested = get_num_bytes_to_req_tti(module_id, lc, round_id, &target, lc->lc_remaining_bytes, tti_ms);

  /* number of bytes scheduled so far for a UE */
  uint32_t bytes_scheduled = ue_sched->num_bytes_sched;
  uint16_t rbs_scheduled = ue_sched->num_rbs_sched;
  // uint32_t tbs_size = ue_sched->tbs;
  LOG_D(NR_MAC, "Bytes scheduled: %u, bytes requested: %ld, oh: %d, remain prbs: %u, min prbs: %d, rbs scheduled: %u\n", bytes_scheduled, bytes_requested, oh, *remain_prbs, min_prbs, rbs_scheduled);
  
  uint16_t max_prb = min_prbs + rbs_scheduled < *remain_prbs ? *remain_prbs : *remain_prbs + rbs_scheduled;
  uint32_t num_bytes_reserved = nr_find_nb_bytes(bytes_requested + bytes_scheduled, 
                                                 oh, 
                                                 &tbs, 
                                                 &rb_size, 
                                                 max_prb, 
                                                 min_prbs + rbs_scheduled, 
                                                 sched_pdsch, 
                                                 tda_info);
  
  uint32_t num_rbs_reserved_lc = rb_size - ue_sched->num_rbs_sched;
  uint32_t num_bytes_reseved_lcid  = num_bytes_reserved - bytes_scheduled;

  AssertFatal(num_rbs_reserved_lc >= 0,"The numbers of rbs reserved for this lc cannot be negative\n");
  AssertFatal(num_bytes_reseved_lcid >= 0,"The numbers of bytes reserved for this lc cannot be negative\n");

  LOG_D(NR_MAC, "Bytes scheduled: %u, bytes reserved: %u, num_bytes_reseved_lcid: %u\n", bytes_scheduled, num_bytes_reserved, num_bytes_reseved_lcid);
  LOG_D(NR_MAC, "rb size: %u, ue_sched->num_rbs_sched: %u, num_rbs_reserved_lc: %u\n", rb_size, ue_sched->num_rbs_sched, num_rbs_reserved_lc);

  if (c->guaranteed_bitrate > 0 || lcnum < 4) {        // gbr flows or srbs
    /*
    Decrement Bj by the total bytes of data reserved to resources from a  logical channel
    currently the Bj is drecremented by number of bytes that are reserved resources for a logical channel,
    so by this approach there will be more chance for lower priority logical channels to be served in the next TTI
  */
    c->Bj -= num_bytes_reseved_lcid;
    LOG_D(NR_MAC,
          "[UE %05x]decrement Bj of the lcid %d by number of bytes reserved resources = %d and new Bj for lcid %d is %ld\n",
          rnti,
          lcnum,
          num_bytes_reseved_lcid,
          lcnum,
          c->Bj);
  }
  else {
    /* calculate the tbs size for non-gbr flows to determine the fairness of non gbr flows */
    uint16_t size_rb = 0;
    LOG_D(NR_MAC, "Number of bytes reserved for this lcid %d are %u\n", lcnum, num_bytes_reseved_lcid);
    
    if (num_bytes_reseved_lcid > 0) {
      bool status_lc = nr_find_nb_rb(sched_pdsch->Qm,
                                   sched_pdsch->R,
                                   1,
                                   sched_pdsch->nrOfLayers,
                                   tda_info->nrOfSymbols,
                                   sched_pdsch->dmrs_parms.N_PRB_DMRS * sched_pdsch->dmrs_parms.N_DMRS_SLOT,
                                   num_bytes_reseved_lcid,
                                   0,
                                   dl_bwp->BWPSize,
                                   &tbsize_lc,
                                   &size_rb);
      AssertFatal(status_lc, "Should be always true because, the number of bytes allocated have suffcient resources\n");
    }
  
  }

  LOG_D(NR_MAC,
        "prescheduled parameters for %s flow lcid %d: tbsize_lc: %u, rb_size: %u, oh: %d, bytes_requested: %lu, num_bytes_reserved: %u\n",
        lcnum < 4 ? "srb" : (c->guaranteed_bitrate > 0 ? "drb_gbr" : "drb_non-gbr"),
        lcnum,
        tbsize_lc,
        num_rbs_reserved_lc,
        oh,
        bytes_requested,
        num_bytes_reseved_lcid);

  lc->allocated_bytes += num_bytes_reseved_lcid;
  lc->lc_remaining_bytes -= num_bytes_reseved_lcid;
  lc->num_rbs_lcid_allocated += num_rbs_reserved_lc;
  lc->tbs = tbsize_lc;
  *remain_prbs -= num_rbs_reserved_lc;
  sched_ctrl->rlc_status[lcnum].prioritized_bytes_in_buffer += num_bytes_reseved_lcid;
  sched_pdsch->lc_data_thru[lcnum] = lc->tbs;
  ue_sched->num_rbs_sched += num_rbs_reserved_lc;
  ue_sched->num_bytes_sched += num_bytes_reseved_lcid + oh;

  if (lc->lc_remaining_bytes == 0) {
    *((lcids_data_status + uenum * NR_MAX_NUM_LCID) + lcnum) = false;
    return 0;
  }

  return 1;
}

uint8_t qos_aware_pre_processor(module_id_t module_id,
                                frame_t frame,
                                sub_frame_t slot,
                                LCIDsched_t *lc_sched,
                                UEsched_t *ue_sched,
                                int *n_rb_sched,
                                uint8_t total_lcs,
                                uint8_t total_UEs)
{
  /*
    sort out the lcids of all active UE such that GBR LCS are given higher priority followed by NON-GBR LCs. With in the group 1 of GBR LC,
    they are sorted as per their weight coefficient and witin group 2 as well, sorted as per the weight assigned for each LC 
  */
  gNB_MAC_INST *nr_mac = RC.nrmac[module_id];
  uint16_t *rballoc_mask = nr_mac->common_channels[CC_id].vrb_map[0];
  qsort((void *)lc_sched, total_lcs, sizeof(LCIDsched_t), lc_comparator);

  #ifdef DEBUG_SCHEDULER
   print_ordered_lc(lc_sched);
  #endif
  
  uint16_t *mask = calloc(ue_sched->UE->current_DL_BWP.BWPSize, sizeof(uint16_t));
  memcpy(mask, rballoc_mask, sizeof(uint16_t) * ue_sched->UE->current_DL_BWP.BWPSize);

  // selection of logical channels
  //int avail_lcids_count = 0;
  int count_index = 0;

  // variable used to build the lcids with positive Bj
  //LCIDsched_t lcids_bj_pos[total_lcs+1];
  //memset(lcids_bj_pos, 0, (total_lcs+1)*sizeof(LCIDsched_t));
  
  // this function should be removed
  //select_logical_channels(lc_sched, &avail_lcids_count, lcids_bj_pos);
   
  // This should be removed because of changes in lc structure change??
  //update_lcconfig(module_id, frame, slot, lc_sched, ue_sched); 

  // variable used to store the lcid data status during lcp
  bool lcids_data_status[total_UEs][NR_MAX_NUM_LCID];
  memset(lcids_data_status, 1, total_UEs*NR_MAX_NUM_LCID);

  uint16_t *remain_prbs = (uint16_t*)n_rb_sched;

  // Evaluate num_prbs per lcid depending on the number of guaranteed bytes available in each of the logical channel and determine how many
  // bytes in each logical channel can be sent in those rbs
  int32_t iteration = 0;
  do {
    /* go until there is space availabile in the MAC PDU and there is data available in RLC buffers of active logical channels */

    // variable used to store the total bytes read from rlc for each lcid
    uint32_t lcids_bytes_tot[total_lcs];
    memset(lcids_bytes_tot, 0, total_lcs*sizeof(uint32_t));
    
    LC_iterator(lc_sched, lc)
    {
      if (lc->UE == NULL || *remain_prbs == 0)
        break;
        
      NR_UE_info_t *UE_curr = lc->UE;
      uint16_t rnti = UE_curr->rnti;
      uint8_t lcnum = lc->curr_lcid;
      uint8_t uenum = lc->curr_ue;

      // skip the logical channel if the data in the buffer was zero because the rbs are already allocated to those bytes
      if (!*((lcids_data_status + uenum * NR_MAX_NUM_LCID) + lcnum)) {
        *((&lcids_data_status[0][0] + uenum * NR_MAX_NUM_LCID) + lcnum) = false;
        LOG_D(NR_MAC, "[UE %05x][LC %d]Skipping because of no data in the buffer\n", rnti, lcnum);
        continue;
      }

      allocate_numrbs_per_lc(module_id,
                             frame,
                             slot,
                             lc, 
                             &ue_sched[lc->curr_ue],
                             remain_prbs,
                             mask,
                             iteration,
                             &lcids_data_status[0][0]);
                             
      if (lcnum >= 4 && ue_sched[lc->curr_ue].order_id == 0) {
        ue_sched[lc->curr_ue].order_id = total_UEs - count_index++;
      }
    }
    iteration++;
  } while (*remain_prbs >0 && get_dataavailability_buffers(total_lcs, lc_sched, &lcids_data_status[0][0])); 

  for (UEsched_t *ue = ue_sched; ue->UE != NULL; ue++)
    LOG_D(NR_MAC, "[UE %04x]Allocated %u PRBs for UE\n", ue->UE->rnti, ue->num_rbs_sched);

  return 1;
}

void qos_aware_scheduler_dl(module_id_t module_id,
                            frame_t frame,
                            sub_frame_t slot,
                            NR_UE_info_t **UE_list,
                            int max_num_ue,
                            int n_rb_sched
                            )
{

  LOG_D(NR_MAC, "-------------------- Resources available for Scheduling in Frame %u and Slot %u for %d UEs are %d ----------------\n", frame, slot, n_rb_sched, max_num_ue);
  
  gNB_MAC_INST *mac = RC.nrmac[module_id];
  NR_ServingCellConfigCommon_t *scc = mac->common_channels[0].ServingCellConfigCommon;

  // UEs that could be scheduled
  UEsched_t UE_sched[MAX_MOBILES_PER_GNB] = {0};
  LCIDsched_t LCID_sched[MAX_MOBILES_PER_GNB * NR_MAX_NUM_LCID] = {0};
  int remainUEs = max_num_ue;
  int num_prbs_remain = n_rb_sched;
  int curUE = 0;
  int CC_id = 0;
  int curLCID = 0;
  
  /* ------------------------------------ TD Scheduler -------------------------------------------*/
  /* Loop UE_info->list to check retransmission */
  UE_iterator (UE_list, UE) {
    if (UE->Msg4_ACKed != true)
      continue;

    NR_UE_sched_ctrl_t *sched_ctrl = &UE->UE_sched_ctrl;
    NR_UE_DL_BWP_t *current_BWP = &UE->current_DL_BWP;

    if (sched_ctrl->ul_failure)
      continue;

    const NR_mac_dir_stats_t *stats = &UE->mac_stats.dl;
    NR_sched_pdsch_t *sched_pdsch = &sched_ctrl->sched_pdsch;

    /* get the PID of a HARQ process awaiting retrnasmission, or -1 otherwise */
    sched_pdsch->dl_harq_pid = sched_ctrl->retrans_dl_harq.head;

    if (remainUEs == 0)
      continue;

    /* update bucket parameters */
    nr_update_lc_bucket_parameters(frame, slot, UE);

    /* retransmission */
    if (sched_pdsch->dl_harq_pid >= 0) {
      /* Allocate retransmission */
      bool r = allocate_dl_retransmission(module_id, frame, slot, &n_rb_sched, UE, sched_pdsch->dl_harq_pid);

      if (!r) {
        LOG_D(NR_MAC, "[UE %04x][%4d.%2d] DL retransmission could not be allocated\n", UE->rnti, frame, slot);
        continue;
      }
      /* reduce max_num_ue once we are sure UE can be allocated, i.e., has CCE */
      remainUEs--;

    } else {
      /* skip this UE if there are no free HARQ processes. This can happen e.g.
       * if the UE disconnected in L2sim, in which case the gNB is not notified
       * (this can be considered a design flaw) */
      if (sched_ctrl->available_dl_harq.head < 0) {
        LOG_D(NR_MAC, "[UE %04x][%4d.%2d] UE has no free DL HARQ process, skipping\n", UE->rnti, frame, slot);
        continue;
      }

      /* Check DL buffer and skip this UE if no bytes and no TA necessary */
      if (sched_ctrl->num_total_bytes == 0 && frame != (sched_ctrl->ta_frame + 10) % 1024)
        continue;

      const NR_bler_options_t *bo = &mac->dl_bler;
      const int max_mcs_table = current_BWP->mcsTableIdx == 1 ? 27 : 28;
      const int max_mcs = min(sched_ctrl->dl_max_mcs, max_mcs_table);
      if (bo->harq_round_max == 1)
        sched_pdsch->mcs = max_mcs;
      else
        sched_pdsch->mcs = get_mcs_from_bler(bo, stats, &sched_ctrl->dl_bler_stats, max_mcs, frame);
      sched_pdsch->nrOfLayers = get_dl_nrOfLayers(sched_ctrl, current_BWP->dci_format);
      sched_pdsch->pm_index =
          mac->identity_pm
              ? 0
              : get_pm_index(mac, UE, current_BWP->dci_format, sched_pdsch->nrOfLayers, mac->radio_config.pdsch_AntennaPorts.XP);

      fill_lc_list(UE, frame, slot, &curLCID, curUE, LCID_sched);

      /* Create UE_sched list for UEs eligible for new transmission*/
      UE_sched[curUE].UE = UE;
      curUE++;
    }
  }

  /* ------------------------------------ FD Scheduler -------------------------------------------*/
  LOG_D(NR_MAC, "Total number of lcids: %d and ues : %d \n", curLCID, curUE);
  if (curLCID <= 0 || curUE <= 0 || num_prbs_remain < 5) {return;}

  qos_aware_pre_processor(module_id,
                          frame,
                          slot,
                          LCID_sched,
                          UE_sched,
                          &num_prbs_remain,
                          curLCID,
                          curUE);

  qsort(UE_sched, curUE, sizeof(UEsched_t), ue_comparator);

  #ifdef DEBUG_SCHEDULER
   print_ordered_ue(UE_sched);
  #endif

  UEsched_t *iterator = UE_sched;
  const int min_rbSize = 5;

  /* Loop UE_sched to find max coeff and allocate transmission */
  while (remainUEs > 0 && n_rb_sched >= min_rbSize && iterator->UE != NULL) {
    NR_UE_sched_ctrl_t *sched_ctrl = &iterator->UE->UE_sched_ctrl;
    const uint16_t rnti = iterator->UE->rnti;

    NR_UE_DL_BWP_t *dl_bwp = &iterator->UE->current_DL_BWP;
    NR_UE_UL_BWP_t *ul_bwp = &iterator->UE->current_UL_BWP;

    if (sched_ctrl->available_dl_harq.head < 0) {
      LOG_D(NR_MAC, "[UE %04x][%4d.%2d] UE has no free DL HARQ process, skipping\n", iterator->UE->rnti, frame, slot);
      iterator++;
      continue;
    }

    NR_sched_pdsch_t *sched_pdsch = &sched_ctrl->sched_pdsch;
    sched_pdsch->dl_harq_pid = sched_ctrl->available_dl_harq.head;

    /* MCS has been set above */
    sched_pdsch->time_domain_allocation = get_dl_tda(mac, scc, slot);
    AssertFatal(sched_pdsch->time_domain_allocation >= 0, "Unable to find PDSCH time domain allocation in list\n");

    const int coresetid = sched_ctrl->coreset->controlResourceSetId;
    sched_pdsch->tda_info = get_dl_tda_info(dl_bwp,
                                            sched_ctrl->search_space->searchSpaceType->present,
                                            sched_pdsch->time_domain_allocation,
                                            scc->dmrs_TypeA_Position,
                                            1,
                                            TYPE_C_RNTI_,
                                            coresetid,
                                            false);
    AssertFatal(sched_pdsch->tda_info.valid_tda, "Invalid TDA from get_dl_tda_info\n");

    NR_tda_info_t *tda_info = &sched_pdsch->tda_info;

    const uint16_t slbitmap = SL_to_bitmap(tda_info->startSymbolIndex, tda_info->nrOfSymbols);

    // TODO assuming beam 0 for now
    uint16_t *rballoc_mask = mac->common_channels[CC_id].vrb_map[0];
    dl_bwp_info_t bwp_info = get_bwp_start_size(mac, iterator->UE);
    int rbStart = 0; // WRT BWP start
    int rbStop = bwp_info.bwpSize - 1;
    int bwp_start = bwp_info.bwpStart;

    // Freq-demain allocation
    LOG_D(NR_MAC, "Before| RB start: %d, Rb stop: %d\n", rbStart, rbStop);
    while (rbStart < rbStop && (rballoc_mask[rbStart] & slbitmap) != slbitmap)
      rbStart++;

    uint16_t max_rbSize = 1;

    while (rbStart + max_rbSize < rbStop && (rballoc_mask[rbStart + max_rbSize] & slbitmap) == slbitmap)
      max_rbSize++;

    LOG_D(NR_MAC, "Rb_start = %d, MaxRbSize = %d \n", rbStart, max_rbSize);

    if (max_rbSize < min_rbSize) {
      LOG_D(NR_MAC,
            "(%d.%d) Cannot schedule RNTI %04x, rbStart %d, rbSize %d, rbStop %d\n",
            frame,
            slot,
            rnti,
            rbStart,
            max_rbSize,
            rbStop);
      iterator++;
      continue;
    }

    // TODO properly set the beam index (currently only done for RA)
    int beam = 0;

    int CCEIndex = get_cce_index(mac,
                                 CC_id,
                                 slot,
                                 iterator->UE->rnti,
                                 &sched_ctrl->aggregation_level,
                                 sched_ctrl->search_space,
                                 sched_ctrl->coreset,
                                 &sched_ctrl->sched_pdcch,
                                 false);
    if (CCEIndex < 0) {
      LOG_D(NR_MAC, "[UE %04x][%4d.%2d] could not find free CCE for DL DCI\n", rnti, frame, slot);
      iterator++;
      continue;
    }

    /* Find PUCCH occasion: if it fails, undo CCE allocation (undoing PUCCH
     * allocation after CCE alloc fail would be more complex) */

    int alloc = -1;
    if (!get_FeedbackDisabled(iterator->UE->sc_info.downlinkHARQ_FeedbackDisabled_r17, sched_pdsch->dl_harq_pid)) {
      int r_pucch = nr_get_pucch_resource(sched_ctrl->coreset, ul_bwp->pucch_Config, CCEIndex);
      alloc = nr_acknack_scheduling(mac, iterator->UE, frame, slot, r_pucch, 0);
      if (alloc<0) {
        LOG_D(NR_MAC, "[UE %04x][%4d.%2d] could not find PUCCH for DL DCI\n",
              rnti,
              frame,
              slot);
        iterator++;
        continue;
      }
    }  

    sched_ctrl->cce_index = CCEIndex;
    fill_pdcch_vrb_map(mac,
                       /* CC_id = */ 0,
                       &sched_ctrl->sched_pdcch,
                       CCEIndex,
                       sched_ctrl->aggregation_level,
                       beam);

    sched_pdsch->dmrs_parms = get_dl_dmrs_params(scc, dl_bwp, tda_info, sched_pdsch->nrOfLayers);
    sched_pdsch->Qm = nr_get_Qm_dl(sched_pdsch->mcs, dl_bwp->mcsTableIdx);
    sched_pdsch->R = nr_get_code_rate_dl(sched_pdsch->mcs, dl_bwp->mcsTableIdx);
    sched_pdsch->pucch_allocation = alloc;
    uint32_t TBS = 0;
    uint16_t rbSize;

    // Fix me: currently, the RLC does not give us the total number of PDUs
    // awaiting. Therefore, for the time being, we put a fixed overhead of 12
    // (for 4 PDUs) and optionally + 2 for TA. Once RLC gives the number of
    // PDUs, we replace with 3 * numPDUs
    const int oh = 3 * 4 + 2 * (frame == (sched_ctrl->ta_frame + 10) % 1024);

    // calculate the TBS as per the prioritized number of bytes in each LCID bytes
    uint32_t total_prioritized_bytes = nr_get_num_prioritized_bytes(max_rbSize, iterator, LCID_sched, oh);
    uint32_t num_bytes;
    num_bytes = total_prioritized_bytes;

    bool check_status = nr_find_nb_rb(sched_pdsch->Qm,
                                      sched_pdsch->R,
                                      1, // no transform precoding for DL
                                      sched_pdsch->nrOfLayers,
                                      tda_info->nrOfSymbols,
                                      sched_pdsch->dmrs_parms.N_PRB_DMRS * sched_pdsch->dmrs_parms.N_DMRS_SLOT,
                                      num_bytes + oh,
                                      min_rbSize,
                                      max_rbSize,
                                      &TBS,
                                      &rbSize);
    LOG_D(NR_MAC,
          "check status = %d, TBS = %d, max_rbsize = %d, rbsize = %d, oh = %d, num_bytes = %d\n",
          check_status,
          TBS,
          max_rbSize,
          rbSize,
          oh,
          num_bytes);
    AssertFatal(check_status == true && rbSize <= max_rbSize, "Algorithm implemenatation is not accurate\n");

    sched_pdsch->rbSize = rbSize;
    sched_pdsch->rbStart = rbStart;
    sched_pdsch->tb_size = TBS;

    /* transmissions: directly allocate */
    n_rb_sched -= sched_pdsch->rbSize;

    for (int rb = 0; rb < sched_pdsch->rbSize; rb++)
      rballoc_mask[rb + sched_pdsch->rbStart] ^= slbitmap;

    remainUEs--;
    iterator++;
  }

}