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

int lc_comparator(const void *p, const void *q)
{
  LCIDsched_t *sched_lc1 = (LCIDsched_t *)p;
  LCIDsched_t *sched_lc2 = (LCIDsched_t *)q;

  uint16_t lcid1 = sched_lc1->curr_lcid;
  uint16_t lcid2 = sched_lc2->curr_lcid;

  if (lcid1 >= 4 && lcid2 >= 4) {
    return sched_lc2->coef - sched_lc1->coef;
  } else {
    if (lcid1 < 4 && lcid2 < 4) {
      return sched_lc1->lcid_priority - sched_lc2->lcid_priority;
    } // both are srbs
    else {
      return -1;
    } // (lcid1 is srb and lcid2 is drb) or (lcid1 is drb and lcid2 is srb)
  }
}

int preallocate_numrbs_per_lc(LCIDsched_t *lc_sched, int num_rbs_data, float *remainPRBs)
{
  // proportional factors to determine number of resource blocks for each lcid of each user
  float prop_factor_total = 0;
  for (LCIDsched_t *lc = lc_sched; lc->UE != NULL; lc++) {
    uint8_t lcnum = lc->curr_lcid;

    if (lcnum < 4) {
      continue;
    }
    prop_factor_total += (float)lc->coef;
  }
  LOG_D(NR_MAC, "the prop factor total is %f\n", prop_factor_total);
  if (prop_factor_total == 0 || num_rbs_data == 0) {
    return 0;
  }

  // Determine number of resource blocks for each lcid based on the proportional factors
  uint32_t totalPRBs = 0;

  for (LCIDsched_t *lc = lc_sched; lc->UE != NULL; lc++) {
    NR_UE_info_t *UE_curr = lc->UE;
    NR_UE_sched_ctrl_t *sched_ctrl = &UE_curr->UE_sched_ctrl;
    uint16_t rnti = UE_curr->rnti;
    uint8_t lcnum = lc->curr_lcid;

    if (lcnum < 4) {
      continue;
    }

    lc->factor = (lc->coef) / prop_factor_total;
    lc->num_rbs_lcid_allocated = floor(lc->factor * num_rbs_data);

    LOG_D(NR_MAC,
          "the idx %d with lcid %d for the UE %04x with available bytes %d is allocated %d prbs initially\n",
          lc->curr_lcid,
          lcnum,
          rnti,
          sched_ctrl->rlc_status[lcnum].bytes_in_buffer,
          lc->num_rbs_lcid_allocated);

    totalPRBs = totalPRBs + lc->num_rbs_lcid_allocated;

    LOG_D(NR_MAC,
          "for lcid %d, actual number is %f and adjusted number(due to rounding) is %d\n",
          lcnum,
          lc->factor * num_rbs_data,
          lc->num_rbs_lcid_allocated);

    *remainPRBs = *remainPRBs + (lc->factor * num_rbs_data) - lc->num_rbs_lcid_allocated;

    LOG_D(NR_MAC,
          "initial remainingPRB after lcid %d that can be used if data is still available after first iteration is %f\n",
          lcnum,
          *remainPRBs);
  }
  LOG_D(NR_MAC,
        "total remainingPRBs after allocating to all data logical channels as per their data availability are %f\n",
        *remainPRBs);
  LOG_D(NR_MAC, "totalPRB are %d\n", totalPRBs);

  AssertFatal(totalPRBs + *remainPRBs - num_rbs_data < 0.05,
              "sum of rbs allocated to all logical channels is %f but the bandwidth used for allocation is %d\n",
              totalPRBs + *remainPRBs,
              num_rbs_data);

  *remainPRBs = round(*remainPRBs);
  return 1;
}

int nr_find_nb_bytes(int bytes,
                     int oh_bytes,
                     uint32_t *tbs,
                     uint16_t *rbsize,
                     uint16_t nb_rb_max,
                     NR_sched_pdsch_t *sched_pdsch,
                     NR_tda_info_t *tda_info)
{
  bool status = false;
  int num_bytes = 0;
  while (!status && bytes >= 0) {
    num_bytes = bytes--;
    status = nr_find_nb_rb(sched_pdsch->Qm,
                           sched_pdsch->R,
                           1,
                           sched_pdsch->nrOfLayers,
                           tda_info->nrOfSymbols,
                           sched_pdsch->dmrs_parms.N_PRB_DMRS * sched_pdsch->dmrs_parms.N_DMRS_SLOT,
                           num_bytes + oh_bytes,
                           0,
                           nb_rb_max,
                           tbs,
                           rbsize);
    LOG_D(NR_MAC, "status = %d, num_bytes_lcid = %hu\n", status, num_bytes);
  }
  return num_bytes;
}

int nr_get_num_prioritized_bytes(int max_rbsize, UEsched_t *ue_iterator, LCIDsched_t *lc_sched)
{
  LOG_D(NR_MAC, "-----------------------------------Entered update phase----------------------------------------\n");
  NR_UE_info_t *ue = ue_iterator->UE;
  NR_UE_sched_ctrl_t *sched_ctrl = &ue->UE_sched_ctrl;
  NR_sched_pdsch_t *sched_pdsch = &sched_ctrl->sched_pdsch;
  NR_tda_info_t *tda_info = &sched_pdsch->tda_info;

  if (max_rbsize < ue_iterator->num_rbs_sched) {
    LOG_D(NR_MAC, "-----------------------------------Entered sub update phase----------------------------------------\n");
    uint16_t num_rbs_sched_pre = ue_iterator->num_rbs_sched;
    ue_iterator->num_rbs_sched = 0;
    LC_iterator(lc_sched, lc)
    {
      if (lc->UE == NULL)
        break;

      uint8_t lcnum = lc->curr_lcid;
      uint32_t num_bytes_buffer = sched_ctrl->rlc_status[lcnum].bytes_in_buffer;

      if (lc->UE != ue_iterator->UE && lc->curr_lcid < 4 && num_bytes_buffer == 0)
        continue;

      LOG_D(NR_MAC, "-----------------------------------Entered sub sub update phase----------------------------------------\n");

      uint16_t num_rbs_used_pre = lc->num_rbs_lcid_used;
      uint16_t allocated_bytes = lc->allocated_bytes;
      lc->num_rbs_lcid_used = 0;
      lc->allocated_bytes = 0;
      LOG_D(NR_MAC, "num_rbs_used_pre before rounding = %hu, max_rbsize = %hu\n", num_rbs_used_pre, max_rbsize);
      num_rbs_used_pre = round(((float)max_rbsize / num_rbs_sched_pre) * num_rbs_used_pre);
      AssertFatal(num_rbs_sched_pre > 0, "number of allocated resource blocks should be greater than 0\n");

      uint32_t TBS = 0;
      uint16_t rbsize;
      int num_bytes_lcid = 0;
      LOG_D(NR_MAC, "num_rbs_used_pre = %hu, allocated_bytes = %hu\n", num_rbs_used_pre, allocated_bytes);

      num_bytes_lcid = nr_find_nb_bytes(allocated_bytes, lc->overhead, &TBS, &rbsize, num_rbs_used_pre, sched_pdsch, tda_info);
      lc->allocated_bytes = num_bytes_lcid;
      sched_ctrl->rlc_status[lcnum].prioritized_bytes_in_buffer = lc->allocated_bytes;
      lc->num_rbs_lcid_used = rbsize;
      ue_iterator->num_rbs_sched += lc->num_rbs_lcid_used;
    }
  }

  // calculate the TBS as per the prioritized number of bytes in each LCID bytes
  uint32_t total_prioritized_bytes = 0;
  for (uint8_t i = 0; i < sched_ctrl->dl_lc_num; i++) {
    int j = sched_ctrl->dl_lc_ids[i];
    total_prioritized_bytes += sched_ctrl->rlc_status[j].prioritized_bytes_in_buffer;
    LOG_D(NR_MAC, "prioritized bytes in lcid %d is %d\n", j, sched_ctrl->rlc_status[j].prioritized_bytes_in_buffer);
  }
  return total_prioritized_bytes;
}

uint8_t rb_allocation_lcid(module_id_t module_id,
                           frame_t frame,
                           sub_frame_t slot,
                           LCIDsched_t *lc_sched,
                           UEsched_t *ue_sched,
                           int n_rb_sched,
                           uint16_t *rballoc_mask)
{
  gNB_MAC_INST *mac = RC.nrmac[module_id];
  NR_ServingCellConfigCommon_t *scc = mac->common_channels[0].ServingCellConfigCommon;

  float remainPRBs = n_rb_sched;
  uint32_t remainData = 0;
  uint32_t ret = 0;
  bool drb_allocation_flag = true;
  bool srb_allocation_flag = true;
  bool iteration_next = false;

  // Evaluate num_prbs per lcid depending on the number of bytes available in each of the logical channel and determine how many
  // bytes in each logical channel can be sent in those rbs
  int32_t iteration = 0;
  do {
    // this iterator loops around each lcid of every UE separately
    LC_iterator(lc_sched, lc)
    {
      if (lc->UE == NULL)
        break;

      NR_UE_info_t *UE_curr = lc->UE;
      uint16_t rnti = UE_curr->rnti;
      NR_UE_sched_ctrl_t *sched_ctrl = &UE_curr->UE_sched_ctrl;
      uint8_t lcnum = lc->curr_lcid;

      uint32_t num_bytes_buffer = UE_curr->UE_sched_ctrl.rlc_status[lcnum].bytes_in_buffer;
      if (num_bytes_buffer == 0) {
        LOG_D(NR_MAC, "NO data\n");
        continue;
      }

      iteration_next = true;

      AssertFatal(iteration > 0 ? (lcnum >= 4 ? true : false) : true, "Only DRBs are allocated from iteration 1\n");
      if (lcnum < 4) {
        // SRB
        LOG_D(NR_MAC, "LC is SRB \n");
        lc->num_rbs_lcid_allocated = srb_allocation_flag ? n_rb_sched : remainPRBs;
        srb_allocation_flag = false;
      } else {
        // DRB
        LOG_D(NR_MAC, "LC is DRB\n");
        if (drb_allocation_flag) {
          LOG_D(NR_MAC, "LC is DRB and preallocated\n");
          int num_rbs_data = remainPRBs;
          remainPRBs = 0;
          if (!preallocate_numrbs_per_lc(lc_sched, num_rbs_data, &remainPRBs))
            return 0;
          drb_allocation_flag = false;
        }
      }

      LOG_D(NR_MAC,
            "------------------------------------------------------ UE RNTI %d lcid %d, coeff %f  round "
            "%d--------------------------------------------------------------------\n",
            rnti,
            lcnum,
            lc->coef,
            iteration);
      NR_UE_DL_BWP_t *dl_bwp = &UE_curr->current_DL_BWP;

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

      uint32_t TBS = 0;
      uint16_t rbSize;

      /* oh bytes should be considered only once while calculating tbsize, so once considered for all logical channels the oh is not
       included by default it is false, for the first time it is set and then for next bytes for same or different logical channels
       it will stay on as true overhead bytes should not be considered
      */
      int oh;

      if (!ue_sched[lc->curr_ue].oh_status) {
        // Fix me: currently, the RLC does not give us the total number of PDUs
        // awaiting. Therefore, for the time being, we put a fixed overhead of 12
        // (for 4 PDUs) and optionally + 2 for TA. Once RLC gives the number of
        // PDUs, we replace with 3 * numPDUs
        oh = 3 * 4 + 2 * (frame == (sched_ctrl->ta_frame + 10) % 1024);
        ue_sched[lc->curr_ue].oh_status = true;
      } else {
        oh = 0;
      }
      lc->overhead = oh;

      uint32_t num_bytes_lcid = 0;
      uint16_t num_rb_lcid = 0;

      if (iteration == 0) {
        num_bytes_lcid = num_bytes_buffer;
        num_rb_lcid = lc->num_rbs_lcid_allocated;
      } else {
        num_bytes_lcid = lc->remaining_bytes;
        num_rb_lcid = remainPRBs;
        remainPRBs = 0;
      }

      num_bytes_lcid = nr_find_nb_bytes(num_bytes_lcid, oh, &TBS, &rbSize, num_rb_lcid, sched_pdsch, tda_info);

      // bytes are either fit exactly or less in the allocated resource blocks, so update the new values
      lc->allocated_bytes += num_bytes_lcid;
      lc->remaining_bytes = num_bytes_buffer - lc->allocated_bytes;
      sched_ctrl->rlc_status[lcnum].prioritized_bytes_in_buffer = lc->allocated_bytes;
      remainData = iteration == 0 ? remainData + lc->remaining_bytes : remainData - lc->remaining_bytes;

      lc->num_rbs_lcid_allocated = (lcnum < 4) ? rbSize : lc->num_rbs_lcid_allocated;
      lc->num_rbs_lcid_used += rbSize;
      lc->num_rbs_lcid_remain = (lc->num_rbs_lcid_used <= lc->num_rbs_lcid_allocated) ? num_rb_lcid - lc->num_rbs_lcid_used : 0;
      ue_sched[lc->curr_ue].num_rbs_sched += lc->num_rbs_lcid_used;
      remainPRBs = (iteration > 0 ? remainPRBs + num_rb_lcid - rbSize : remainPRBs)
                   + (lcnum >= 4 ? lc->num_rbs_lcid_remain : -lc->num_rbs_lcid_allocated);

      LOG_D(NR_MAC,
            "[frame %u slot %u][UE %04x][iteration %d]number of bytes allocated for lcid %d are %d in %d prbs and TBS of %d and "
            "remaining bytes are %d and remaining PRBs are %f and oh bytes are %d n_rb_sched = %d\n",
            frame,
            slot,
            UE_curr->rnti,
            iteration,
            lcnum,
            lc->allocated_bytes,
            lc->num_rbs_lcid_used,
            TBS,
            lc->remaining_bytes,
            remainPRBs,
            oh,
            n_rb_sched);

      if (remainPRBs == 0 || remainData == 0) {
        iteration_next = false;
        ret = 1;
        break;
      }
    }

    LOG_D(NR_MAC,
          "[frame %u slot %u]After iteration %d,remaining PRBS is %f and remaining data bytes is %d \n",
          frame,
          slot,
          iteration,
          remainPRBs,
          remainData);
    iteration++;
  } while (iteration_next); // iteration

  for (UEsched_t *ue = ue_sched; ue->UE != NULL; ue++)
    LOG_D(NR_MAC, "Allocated %u PRBs for UE\n", ue->num_rbs_sched);

  return ret;
}

void fill_lc_sched_list(NR_UE_info_t *UE, frame_t frame, int *lc_currid, int ue_currid, LCIDsched_t *lc_sched)
{
  NR_UE_sched_ctrl_t *sched_ctrl = &UE->UE_sched_ctrl;

  for (uint8_t lc_idx = 0; lc_idx < sched_ctrl->dl_lc_num; lc_idx++) {
    /* Check DL buffer and skip this lcid in this UE if no bytes */
    uint8_t lcid = sched_ctrl->dl_lc_ids[lc_idx];

    uint32_t num_bytes = sched_ctrl->rlc_status[lcid].bytes_in_buffer;
    if (num_bytes == 0) // && frame != (sched_ctrl->ta_frame + 10) % 1024
      continue;

    /* QoS*/
    // aggregated priority level in each lcid, since since an lcid buffer can contain packets of different qos flows
    uint64_t aggregated_pl_lcid = lcid < 4 ? lcid + 1 : sched_ctrl->dl_lc_ids_priorities[lcid];
    AssertFatal(aggregated_pl_lcid >= 0,
                "Aggregated priority should be non zero positive number. should be some issue in the process_QOSConfig function\n");

    float coeff_ue_lcid = lcid < 4 ? -1 : ((float)1 / aggregated_pl_lcid) * num_bytes;
    LOG_D(NR_MAC,
          "[UE %04x]coeff for lcid %u with %u bytes in buffer is %f with aggregated priority %lu\n",
          UE->rnti,
          lcid,
          num_bytes,
          coeff_ue_lcid,
          aggregated_pl_lcid);

    lc_sched[*lc_currid].coef = coeff_ue_lcid;
    lc_sched[*lc_currid].UE = UE;
    lc_sched[*lc_currid].curr_lcid = lcid;
    lc_sched[*lc_currid].lcid_priority = aggregated_pl_lcid;
    lc_sched[*lc_currid].curr_ue = ue_currid;
    lc_sched[*lc_currid].remaining_bytes = 1;
    (*lc_currid)++;
  }
}

void qos_aware_scheduler_dl(module_id_t module_id,
                            frame_t frame,
                            sub_frame_t slot,
                            NR_UE_info_t **UE_list,
                            int max_num_ue,
                            int n_rb_sched,
                            uint16_t *rballoc_mask)
{
  gNB_MAC_INST *mac = RC.nrmac[module_id];
  NR_ServingCellConfigCommon_t *scc = mac->common_channels[0].ServingCellConfigCommon;

  // UEs that could be scheduled
  UEsched_t UE_sched[MAX_MOBILES_PER_GNB] = {0};
  LCIDsched_t LCID_sched[MAX_MOBILES_PER_GNB * NR_MAX_NUM_LCID] = {0};
  int remainUEs = max_num_ue;
  int curUE = 0;
  int CC_id = 0;
  int curLCID = 0;

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

    /* retransmission */
    if (sched_pdsch->dl_harq_pid >= 0) {
      /* Allocate retransmission */
      bool r = allocate_dl_retransmission(module_id, frame, slot, rballoc_mask, &n_rb_sched, UE, sched_pdsch->dl_harq_pid);

      if (!r) {
        LOG_I(NR_MAC, "[UE %04x][%4d.%2d] DL retransmission could not be allocated\n", UE->rnti, frame, slot);
        continue;
      }
      /* reduce max_num_ue once we are sure UE can be allocated, i.e., has CCE */
      remainUEs--;

    } else {
      /* skip this UE if there are no free HARQ processes. This can happen e.g.
       * if the UE disconnected in L2sim, in which case the gNB is not notified
       * (this can be considered a design flaw) */
      if (sched_ctrl->available_dl_harq.head < 0) {
        LOG_I(NR_MAC, "[UE %04x][%4d.%2d] UE has no free DL HARQ process, skipping\n", UE->rnti, frame, slot);
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

      // for for each UE
      fill_lc_sched_list(UE, frame, &curLCID, curUE, LCID_sched);

      /* Create UE_sched list for UEs eligible for new transmission*/
      UE_sched[curUE].UE = UE;
      curUE++;
    }
  }

  /*
    sort out the lcid depending on the lcid priority and data available from transmission
    buffer(only for DRBs and SRBs are always given higher priority before DRBs)
  */
  qsort((void *)LCID_sched, curLCID, sizeof(LCIDsched_t), lc_comparator);

  if (true && !rb_allocation_lcid(module_id, frame, slot, LCID_sched, UE_sched, n_rb_sched, rballoc_mask))
    return;

  UEsched_t *iterator = UE_sched;
  const int min_rbSize = 5;

  /* Loop UE_sched to find max coeff and allocate transmission */
  while (remainUEs > 0 && n_rb_sched >= min_rbSize && iterator->UE != NULL) {
    NR_UE_sched_ctrl_t *sched_ctrl = &iterator->UE->UE_sched_ctrl;
    const uint16_t rnti = iterator->UE->rnti;

    NR_UE_DL_BWP_t *dl_bwp = &iterator->UE->current_DL_BWP;
    NR_UE_UL_BWP_t *ul_bwp = &iterator->UE->current_UL_BWP;

    if (sched_ctrl->available_dl_harq.head < 0) {
      LOG_I(NR_MAC, "[UE %04x][%4d.%2d] UE has no free DL HARQ process, skipping\n", iterator->UE->rnti, frame, slot);
      iterator++;
      continue;
    }

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

    int r_pucch = nr_get_pucch_resource(sched_ctrl->coreset, ul_bwp->pucch_Config, CCEIndex);
    const int alloc = nr_acknack_scheduling(mac, iterator->UE, frame, slot, r_pucch, 0);

    if (alloc < 0) {
      LOG_D(NR_MAC, "[UE %04x][%4d.%2d] could not find PUCCH for DL DCI\n", rnti, frame, slot);
      iterator++;
      continue;
    }

    sched_ctrl->cce_index = CCEIndex;
    fill_pdcch_vrb_map(mac,
                       /* CC_id = */ 0,
                       &sched_ctrl->sched_pdcch,
                       CCEIndex,
                       sched_ctrl->aggregation_level);

    /* MCS has been set above */
    NR_sched_pdsch_t *sched_pdsch = &sched_ctrl->sched_pdsch;
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

    NR_tda_info_t *tda_info = &sched_pdsch->tda_info;

    const uint16_t slbitmap = SL_to_bitmap(tda_info->startSymbolIndex, tda_info->nrOfSymbols);

    int rbStop = 0;
    int rbStart = 0;
    get_start_stop_allocation(mac, iterator->UE, &rbStart, &rbStop);
    // Freq-demain allocation
    while (rbStart < rbStop && (rballoc_mask[rbStart] & slbitmap) != slbitmap)
      rbStart++;

    uint16_t max_rbSize = 1;

    while (rbStart + max_rbSize < rbStop && (rballoc_mask[rbStart + max_rbSize] & slbitmap) == slbitmap)
      max_rbSize++;

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
    // const int oh = 3 * sched_ctrl->dl_pdus_total + 2 * (frame == (sched_ctrl->ta_frame + 10) % 1024);

    // calculate the TBS as per the prioritized number of bytes in each LCID bytes
    uint32_t total_prioritized_bytes = nr_get_num_prioritized_bytes(max_rbSize, iterator, LCID_sched);
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
    LOG_D(NR_MAC, "Rb_start = %d, MaxRbSize = %d \n", rbStart, max_rbSize);
    LOG_D(NR_MAC,
          "check status = %d, TBS = %d, max_rbsize = %d, rbsize = %d, oh = %d\n",
          check_status,
          TBS,
          max_rbSize,
          rbSize,
          oh);
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

// static void qos_aware_dl_backup(module_id_t module_id, frame_t frame, sub_frame_t slot, NR_UE_info_t **UE_list, int max_num_ue,
// int n_rb_sched, uint16_t *rballoc_mask)
// {
//   gNB_MAC_INST *mac = RC.nrmac[module_id];
//   NR_ServingCellConfigCommon_t *scc = mac->common_channels[0].ServingCellConfigCommon;

//   // UEs that could be scheduled
//   UEsched_t UE_sched[MAX_MOBILES_PER_GNB] = {0};
//   LCIDsched_t LCID_sched[MAX_MOBILES_PER_GNB * NR_MAX_NUM_LCID] = {0};
//   int remainUEs = max_num_ue;
//   int curUE = 0;
//   int CC_id = 0;
//   int curLCID = 0;

//   /* Loop UE_info->list to check retransmission */
//   UE_iterator(UE_list, UE) {
//     if (UE->Msg4_ACKed != true)
//       continue;

//     NR_UE_sched_ctrl_t *sched_ctrl = &UE->UE_sched_ctrl;
//     NR_UE_DL_BWP_t *current_BWP = &UE->current_DL_BWP;

//     if (sched_ctrl->ul_failure==1)
//       continue;

//     const NR_mac_dir_stats_t *stats = &UE->mac_stats.dl;
//     NR_sched_pdsch_t *sched_pdsch = &sched_ctrl->sched_pdsch;
//     /* get the PID of a HARQ process awaiting retrnasmission, or -1 otherwise */
//     sched_pdsch->dl_harq_pid = sched_ctrl->retrans_dl_harq.head;

//     if (remainUEs == 0)
//       continue;

//     /* retransmission */
//     if (sched_pdsch->dl_harq_pid >= 0) {
//       /* Allocate retransmission */
//       bool r = allocate_dl_retransmission(module_id, frame, slot, rballoc_mask, &n_rb_sched, UE, sched_pdsch->dl_harq_pid);

//       if (!r) {
//         LOG_D(NR_MAC, "[UE %04x][%4d.%2d] DL retransmission could not be allocated\n",
//               UE->rnti,
//               frame,
//               slot);
//         continue;
//       }
//       /* reduce max_num_ue once we are sure UE can be allocated, i.e., has CCE */
//       remainUEs--;

//     } else {
//       /* skip this UE if there are no free HARQ processes. This can happen e.g.
//        * if the UE disconnected in L2sim, in which case the gNB is not notified
//        * (this can be considered a design flaw) */
//       if (sched_ctrl->available_dl_harq.head < 0) {
//         LOG_D(NR_MAC, "[UE %04x][%4d.%2d] UE has no free DL HARQ process, skipping\n",
//               UE->rnti,
//               frame,
//               slot);
//         continue;
//       }

//       /* Check DL buffer and skip this UE if no bytes and no TA necessary */
//       if (sched_ctrl->num_total_bytes == 0 && frame != (sched_ctrl->ta_frame + 10) % 1024) {
//         LOG_D(NR_MAC, "no available data for the UE %04x \n", UE->rnti);
//         continue;
//       }

//       const NR_bler_options_t *bo = &mac->dl_bler;
//       const int max_mcs_table = current_BWP->mcsTableIdx == 1 ? 27 : 28;
//       const int max_mcs = min(sched_ctrl->dl_max_mcs, max_mcs_table);
//       if (bo->harq_round_max == 1)
//         sched_pdsch->mcs = max_mcs;
//       else
//         sched_pdsch->mcs = get_mcs_from_bler(bo, stats, &sched_ctrl->dl_bler_stats, max_mcs, frame);
//       sched_pdsch->nrOfLayers = get_dl_nrOfLayers(sched_ctrl, current_BWP->dci_format);
//      sched_pdsch->pm_index =
//           mac->identity_pm ? 0 : get_pm_index(UE, sched_pdsch->nrOfLayers, mac->radio_config.pdsch_AntennaPorts.XP);

//       /* Create UE_sched list for UEs eligible for new transmission*/
//       UE_sched[curUE].UE = UE;

//       // for for each UE
//       fill_lc_sched_list(UE, frame, &curLCID, curUE, LCID_sched);
//     }
//     curUE++; // is it the right placement of this increment??
//   }

//   // sort out the lcid depending on the lcid priority and data available from transmission buffer(only for DRBs and SRBs are
//   always given higher priority before DRBs) qsort((void *)LCID_sched, curLCID, sizeof(LCIDsched_t), lc_comparator);

//   if (!rb_allocation_lcid(module_id, frame, slot, LCID_sched, UE_sched, n_rb_sched, rballoc_mask))
//     return;

//   const int min_rbSize = 0;  // changed while combining both

//   /* Loop UE_sched to find max coeff and allocate transmission */
//   UEsched_t *iterator = UE_sched;
//   while (remainUEs > 0 && n_rb_sched >= min_rbSize && iterator->UE != NULL) {
//     NR_UE_sched_ctrl_t *sched_ctrl = &iterator->UE->UE_sched_ctrl;
//     const uint16_t rnti = iterator->UE->rnti;

//     NR_UE_DL_BWP_t *dl_bwp = &iterator->UE->current_DL_BWP;
//     NR_UE_UL_BWP_t *ul_bwp = &iterator->UE->current_UL_BWP;

//     if (sched_ctrl->available_dl_harq.head < 0) {
//       LOG_D(NR_MAC, "[UE %04x][%4d.%2d] UE has no free DL HARQ process, skipping\n",
//             iterator->UE->rnti,
//             frame,
//             slot);
//       iterator++;
//       continue;
//     }

//     int CCEIndex = get_cce_index(mac,
//                                  CC_id, slot, iterator->UE->rnti,
//                                  &sched_ctrl->aggregation_level,
//                                  sched_ctrl->search_space,
//                                  sched_ctrl->coreset,
//                                  &sched_ctrl->sched_pdcch,
//                                  false);
//     if (CCEIndex<0) {
//       LOG_D(NR_MAC, "[UE %04x][%4d.%2d] could not find free CCE for DL DCI\n",
//             rnti,
//             frame,
//             slot);
//       iterator++;
//       continue;
//     }

//     /* Find PUCCH occasion: if it fails, undo CCE allocation (undoing PUCCH
//     * allocation after CCE alloc fail would be more complex) */

//     int r_pucch = nr_get_pucch_resource(sched_ctrl->coreset, ul_bwp->pucch_Config, CCEIndex);
//     const int alloc = nr_acknack_scheduling(mac, iterator->UE, frame, slot, r_pucch, 0);

//     if (alloc<0) {
//       LOG_D(NR_MAC, "[UE %04x][%4d.%2d] could not find PUCCH for DL DCI\n",
//             rnti,
//             frame,
//             slot);
//       iterator++;
//       continue;
//     }

//     sched_ctrl->cce_index = CCEIndex;
//     fill_pdcch_vrb_map(mac,
//                        /* CC_id = */ 0,
//                        &sched_ctrl->sched_pdcch,
//                        CCEIndex,
//                        sched_ctrl->aggregation_level);

//     /* MCS has been set above */
//     NR_sched_pdsch_t *sched_pdsch = &sched_ctrl->sched_pdsch;
//     sched_pdsch->time_domain_allocation = get_dl_tda(mac, scc, slot);
//     AssertFatal(sched_pdsch->time_domain_allocation>=0,"Unable to find PDSCH time domain allocation in list\n");

//     const int coresetid = sched_ctrl->coreset->controlResourceSetId;
//     sched_pdsch->tda_info = get_dl_tda_info(dl_bwp, sched_ctrl->search_space->searchSpaceType->present,
//     sched_pdsch->time_domain_allocation,
//                                             scc->dmrs_TypeA_Position, 1, NR_RNTI_C, coresetid, false);

//     NR_tda_info_t *tda_info = &sched_pdsch->tda_info;

//     const uint16_t slbitmap = SL_to_bitmap(tda_info->startSymbolIndex, tda_info->nrOfSymbols);

//     int rbStop = 0;
//     int rbStart = 0;
//     get_start_stop_allocation(mac, iterator->UE, &rbStart, &rbStop);
//     // Freq-demain allocation
//     while (rbStart < rbStop && (rballoc_mask[rbStart] & slbitmap) != slbitmap)
//       rbStart++;
//     LOG_D(NR_MAC,"RB Start = %d\n", rbStart);

//     uint16_t max_rbSize = 1;
//     while (rbStart + max_rbSize < rbStop && (rballoc_mask[rbStart + max_rbSize] & slbitmap) == slbitmap)
//       max_rbSize++;
//     LOG_D(NR_MAC,"MAX RB SIZE = %d, rbstop = %d\n", max_rbSize, rbStop);

//     sched_pdsch->dmrs_parms = get_dl_dmrs_params(scc, dl_bwp, tda_info, sched_pdsch->nrOfLayers);
//     sched_pdsch->Qm = nr_get_Qm_dl(sched_pdsch->mcs, dl_bwp->mcsTableIdx);
//     sched_pdsch->R = nr_get_code_rate_dl(sched_pdsch->mcs, dl_bwp->mcsTableIdx);
//     sched_pdsch->pucch_allocation = alloc;
//     uint32_t TBS = 0;
//     uint16_t rbSize;
//     // Fix me: currently, the RLC does not give us the total number of PDUs
//     // awaiting. Therefore, for the time being, we put a fixed overhead of 12
//     // (for 4 PDUs) and optionally + 2 for TA. Once RLC gives the number of
//     // PDUs, we replace with 3 * numPDUs
//     const int oh = 3 * 4 + 2 * (frame == (sched_ctrl->ta_frame + 10) % 1024);
//     // const int oh = 3 * sched_ctrl->dl_pdus_total + 2 * (frame == (sched_ctrl->ta_frame + 10) % 1024);

//     // calculate the TBS as per the prioritized number of bytes in each LCID bytes
//     uint32_t total_prioritized_bytes = nr_get_num_prioritized_bytes(max_rbSize, iterator, LCID_sched);

//     // check status
//     bool check_status;
//     check_status = nr_find_nb_rb(sched_pdsch->Qm,
//                                  sched_pdsch->R,
//                                  1,
//                                  sched_pdsch->nrOfLayers,
//                                  tda_info->nrOfSymbols,
//                                  sched_pdsch->dmrs_parms.N_PRB_DMRS * sched_pdsch->dmrs_parms.N_DMRS_SLOT,
//                                  total_prioritized_bytes + oh,
//                                  min_rbSize,
//                                  max_rbSize,
//                                  &TBS,
//                                  &rbSize);
//     LOG_D(NR_MAC, "Rb_start = %d, MaxRbSize = %d \n", rbStart, max_rbSize);
//     LOG_D(NR_MAC, "check status = %d, TBS = %d, max_rbsize = %d, rbsize = %d, oh = %d\n", check_status, TBS, max_rbSize, rbSize,
//     oh); AssertFatal(check_status == true && rbSize <= max_rbSize, "Algorithm implemenatation is not accurate\n");

//     sched_pdsch->rbSize = rbSize;
//     sched_pdsch->rbStart = rbStart;
//     sched_pdsch->tb_size = TBS;
//     /* transmissions: directly allocate */
//     n_rb_sched -= sched_pdsch->rbSize;

//     for (int rb = 0; rb < sched_pdsch->rbSize; rb++)
//       rballoc_mask[rb + sched_pdsch->rbStart] ^= slbitmap;

//     remainUEs--;
//     iterator++;
//   }
// }
