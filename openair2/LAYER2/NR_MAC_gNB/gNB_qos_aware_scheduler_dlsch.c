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
  int remainUEs = max_num_ue;
  int curUE = 0;
  int CC_id = 0;

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
    /* Calculate Throughput */
    const float a = 0.01f;
    const uint32_t b = UE->mac_stats.dl.current_bytes;
    UE->dl_thr_ue = (1 - a) * UE->dl_thr_ue + a * b;

    if (remainUEs == 0)
      continue;

    /* retransmission */
    if (sched_pdsch->dl_harq_pid >= 0) {
      /* Allocate retransmission */
      bool r = allocate_dl_retransmission(module_id, frame, slot, rballoc_mask, &n_rb_sched, UE, sched_pdsch->dl_harq_pid);

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

      /* QoS processing*/
      float aggregated_priority = sched_ctrl->priority_ue;

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
      const uint8_t Qm = nr_get_Qm_dl(sched_pdsch->mcs, current_BWP->mcsTableIdx);
      const uint16_t R = nr_get_code_rate_dl(sched_pdsch->mcs, current_BWP->mcsTableIdx);
      uint32_t tbs = nr_compute_tbs(Qm,
                                    R,
                                    1, /* rbSize */
                                    10, /* hypothetical number of slots */
                                    0, /* N_PRB_DMRS * N_DMRS_SLOT */
                                    0 /* N_PRB_oh, 0 for initialBWP */,
                                    0 /* tb_scaling */,
                                    sched_pdsch->nrOfLayers)
                     >> 3;
      float coeff_ue = (float)tbs * (100 - aggregated_priority) / UE->dl_thr_ue;
      LOG_D(NR_MAC,
            "[UE %04x][%4d.%2d] b %d, thr_ue %f, tbs %d, ue_priority %f, coeff_ue %f\n",
            UE->rnti,
            frame,
            slot,
            b,
            UE->dl_thr_ue,
            tbs,
            aggregated_priority,
            coeff_ue);

      /* Create UE_sched list for UEs eligible for new transmission*/
      UE_sched[curUE].coef = coeff_ue;
      UE_sched[curUE].UE = UE;
      curUE++;
    }
  }

  qsort(UE_sched, sizeofArray(UE_sched), sizeof(UEsched_t), comparator);
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
    nr_find_nb_rb(sched_pdsch->Qm,
                  sched_pdsch->R,
                  1, // no transform precoding for DL
                  sched_pdsch->nrOfLayers,
                  tda_info->nrOfSymbols,
                  sched_pdsch->dmrs_parms.N_PRB_DMRS * sched_pdsch->dmrs_parms.N_DMRS_SLOT,
                  sched_ctrl->num_total_bytes + oh,
                  min_rbSize,
                  max_rbSize,
                  &TBS,
                  &rbSize);
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
