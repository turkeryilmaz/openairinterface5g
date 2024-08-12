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

/* \file        nr_slsch_scheduler.c
 * \brief       Routines for UE SLSCH scheduling
 * \author      R. Knopp
 * \date        Aug. 2023
 * \version     0.1
 * \company     EURECOM
 * \email       raymond.knopp@eurecom.fr
 */

#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#include <common/utils/nr/nr_common.h>

#include "NR_MAC_COMMON/nr_mac.h"
#include "NR_MAC_COMMON/nr_mac_common.h"
#include "NR_MAC_UE/mac_proto.h"
#include "NR_MAC_UE/mac_extern.h"
#include "NR_MAC_UE/nr_ue_sci.h"
#include <executables/nr-uesoftmodem.h>
#include "NR_MAC_UE/mac_defs_sl.h"
#include "NR_MAC_gNB/nr_mac_gNB.h"

const uint8_t nr_rv_round_map[4] = {0, 2, 3, 1};

uint8_t scalled_mcs(uint8_t current_mcs) {
  uint8_t orig_scale_min = 1, orig_scale_max = 28;
  uint8_t new_scale_min = 1, new_scale_max = 16;
  uint8_t scaled_mcs = new_scale_min + ((current_mcs - orig_scale_min) * (new_scale_max - new_scale_min)) / (orig_scale_max - orig_scale_min);
  return scaled_mcs;
}

void reset_sl_harq_list(NR_SL_UE_sched_ctrl_t *sched_ctrl) {
  int harq;
  while ((harq = sched_ctrl->feedback_sl_harq.head) >= 0) {
    remove_front_nr_list(&sched_ctrl->feedback_sl_harq);
    add_tail_nr_list(&sched_ctrl->available_sl_harq, harq);
  }

  while ((harq = sched_ctrl->retrans_sl_harq.head) >= 0) {
    remove_front_nr_list(&sched_ctrl->retrans_sl_harq);
    add_tail_nr_list(&sched_ctrl->available_sl_harq, harq);
  }

  for (int i = 0; i < NR_MAX_HARQ_PROCESSES; i++) {
    sched_ctrl->sl_harq_processes[i].feedback_slot = -1;
    sched_ctrl->sl_harq_processes[i].round = 0;
    sched_ctrl->sl_harq_processes[i].is_waiting = false;
  }
}

static void abort_nr_ue_sl_harq(NR_UE_MAC_INST_t *mac, int8_t harq_pid)
{
  NR_SL_UE_info_t *UE_info = (NR_SL_UE_info_t *)&mac->sl_info.list[0];
  NR_SL_UE_sched_ctrl_t *sched_ctrl = &UE_info->UE_sched_ctrl;
  NR_UE_sl_harq_t *harq = &sched_ctrl->sl_harq_processes[harq_pid];

  harq->ndi ^= 1;
  harq->round = 0;
  UE_info->mac_sl_stats.sl.errors++;
  add_tail_nr_list(&sched_ctrl->available_sl_harq, harq_pid);

  /* the transmission failed: the UE won't send the data we expected initially,
   * so retrieve to correctly schedule after next BSR */
  sched_ctrl->sched_sl_bytes -= harq->sched_pssch.tb_size;
  if (sched_ctrl->sched_sl_bytes < 0)
    sched_ctrl->sched_sl_bytes = 0;
}

void handle_nr_ue_sl_harq(module_id_t mod_id,
                          frame_t frame,
                          sub_frame_t slot,
                          sl_nr_slsch_pdu_t *rx_slsch_pdu,
                          uint16_t src_id)
{
  NR_UE_MAC_INST_t *mac = get_mac_inst(mod_id);
  NR_UE_SL_SCHED_LOCK(&mac->sl_sched_lock);
  NR_SL_UE_info_t **UE_SL_temp = (NR_SL_UE_info_t *)&mac->sl_info.list, *UE;
  // TODO: update for multiple UEs
  UE=*(UE_SL_temp);
  uint8_t num_ack_rcvd = rx_slsch_pdu->num_acks_rcvd;

  NR_SL_UE_sched_ctrl_t *sched_ctrl = &UE->UE_sched_ctrl;
  NR_UE_sl_harq_t **matched_harqs = (NR_UE_sl_harq_t **) calloc(sched_ctrl->feedback_sl_harq.len, sizeof(NR_UE_sl_harq_t *));
  int k = find_nr_ue_sl_harq(frame, slot, sched_ctrl, matched_harqs);

  for (int i = 0; i < num_ack_rcvd; i++) {
    uint8_t ack_nack = rx_slsch_pdu->ack_nack_rcvd[i];
    uint8_t rx_harq_id = matched_harqs[i]->sl_harq_pid;
    NR_SL_UE_sched_ctrl_t *sched_ctrl = &UE->UE_sched_ctrl;
    int8_t harq_pid = sched_ctrl->feedback_sl_harq.head;
    LOG_D(NR_MAC, "Comparing %4u.%2u rx_harq_id vs feedback harq_pid = %d %d\n", frame, slot, rx_harq_id, harq_pid);
    while (rx_harq_id != harq_pid || harq_pid < 0) {
      LOG_W(NR_MAC,
            "Unexpected SLSCH HARQ PID %d (have %d) for src id %4d\n",
            rx_harq_id,
            harq_pid,
            src_id);
      if (harq_pid < 0) {
        NR_UE_SL_SCHED_UNLOCK(&mac->sl_sched_lock);
        return;
      }

      remove_front_nr_list(&sched_ctrl->feedback_sl_harq);
      sched_ctrl->sl_harq_processes[harq_pid].is_waiting = false;

      if(sched_ctrl->sl_harq_processes[harq_pid].round >= (HARQ_ROUND_MAX - 1)) {
        abort_nr_ue_sl_harq(mac, harq_pid);
      } else {
        sched_ctrl->sl_harq_processes[harq_pid].round++;
        add_tail_nr_list(&sched_ctrl->retrans_sl_harq, harq_pid);
      }
      harq_pid = sched_ctrl->feedback_sl_harq.head;
    }
    remove_front_nr_list(&sched_ctrl->feedback_sl_harq);
    NR_UE_sl_harq_t *harq = &sched_ctrl->sl_harq_processes[harq_pid];
    DevAssert(harq->is_waiting);
    harq->feedback_slot = -1;
    harq->is_waiting = false;
    if (!ack_nack) {
      harq->ndi ^= 1;
      harq->round = 0;
      LOG_D(NR_MAC,
            "Ulharq id %d crc passed for src id %4d\n",
            harq_pid,
            src_id);
      add_tail_nr_list(&sched_ctrl->available_sl_harq, harq_pid);
    } else if (harq->round >= (HARQ_ROUND_MAX - 1)) {
      abort_nr_ue_sl_harq(mac, harq_pid);
      LOG_D(NR_MAC,
            "src id %4d, Slharq id %d crc failed in all rounds\n",
            src_id,
            harq_pid);
    } else {
      harq->round++;
      LOG_W(NR_MAC,
            "%4u.%2u Slharq id %d crc failed for src id %4d\n",
            frame,
            slot,
            harq_pid,
            src_id);
      add_tail_nr_list(&sched_ctrl->retrans_sl_harq, harq_pid);
    }
    NR_UE_SL_SCHED_UNLOCK(&mac->sl_sched_lock);
  }
  free(matched_harqs);
  matched_harqs = NULL;
}

bool nr_schedule_slsch(NR_UE_MAC_INST_t *mac, int frameP, int slotP, nr_sci_pdu_t *sci_pdu,
                       nr_sci_pdu_t *sci2_pdu, uint8_t *slsch_pdu, nr_sci_format_t format2,
                       uint16_t *slsch_pdu_length_max, NR_UE_sl_harq_t *cur_harq,
                       mac_rlc_status_resp_t *rlc_status) {

  sl_nr_ue_mac_params_t *sl_mac = mac->SL_MAC_PARAMS;
  uint8_t mu = sl_mac->sl_phy_config.sl_config_req.sl_bwp_config.sl_scs;
  uint8_t slots_per_frame = nr_slots_per_frame[mu];

  uint8_t psfch_period = 0;
  const uint8_t psfch_periods[] = {0,1,2,4};
  psfch_period = (mac->sl_tx_res_pool->sl_PSFCH_Config_r16 &&
                  mac->sl_tx_res_pool->sl_PSFCH_Config_r16->choice.setup->sl_PSFCH_Period_r16)
                  ? psfch_periods[*mac->sl_tx_res_pool->sl_PSFCH_Config_r16->choice.setup->sl_PSFCH_Period_r16] : 0;
  *slsch_pdu_length_max = 0;
  bool csi_acq = !mac->SL_MAC_PARAMS->sl_CSI_Acquisition;
  bool csi_req_slot = !((slots_per_frame * frameP + slotP - sl_mac->slot_offset) % sl_mac->slot_periodicity);
  bool is_harq_feedback = is_feedback_scheduled(mac, frameP, slotP);
  LOG_D(NR_MAC, "frame.slot %4d.%2d bytes_in_buffer? %d, harq_feedback %d, (csi_acq && csi_req_slot) %d, sl_csi_report %p\n",
        frameP, slotP, rlc_status->bytes_in_buffer >= 0, is_harq_feedback, (csi_acq && csi_req_slot), mac->sl_csi_report);
  if (rlc_status->bytes_in_buffer >= 0 || is_harq_feedback || (csi_acq && csi_req_slot) || mac->sl_csi_report) {
     uint8_t cqi_Table = 0;
     int8_t mcs = 11, ri = 0;
     uint16_t dest = mac->dest_id != -1 ? mac->dest_id : 0xabcd;
     uint16_t indx = dest%CURRENT_NUM_UE_CONNECTIONS;
     int8_t cqi = mac->dest_id != -1 ? mac->sl_info.list[indx]->UE_sched_ctrl.csi_report.cqi : -1;
     if (cqi != -1) {
      int mcs_tb_ind = 0;
      if (sci_pdu->additional_mcs.nbits > 0)
        mcs_tb_ind = sci_pdu->additional_mcs.val;
      if (mcs_tb_ind == 0)
        cqi_Table = NR_CSI_ReportConfig__cqi_Table_table1;
      else if (mcs_tb_ind == 1)
        cqi_Table = NR_CSI_ReportConfig__cqi_Table_table2;
      else if (mcs_tb_ind == 2)
        cqi_Table = NR_CSI_ReportConfig__cqi_Table_table3;

      mcs = get_mcs_from_cqi(mcs_tb_ind, cqi_Table, cqi);
      mac->sl_info.list[indx]->UE_sched_ctrl.sl_max_mcs = scalled_mcs(mcs);
      ri = mac->sl_info.list[indx]->UE_sched_ctrl.csi_report.ri;
     }
     // Fill SCI1A
     sci_pdu->priority = 0;
     sci_pdu->frequency_resource_assignment.val = 0;
     sci_pdu->time_resource_assignment.val = 0;
     sci_pdu->resource_reservation_period.val = 0;
     sci_pdu->dmrs_pattern.val = 0;
     sci_pdu->second_stage_sci_format = 0;
     sci_pdu->number_of_dmrs_port = ri;
     sci_pdu->mcs = mac->sl_info.list[indx]->UE_sched_ctrl.sl_max_mcs;
     sci_pdu->additional_mcs.val = 0;

     /*Following code will check whether SLSCH was received before and
      its feedback has scheduled for current slot
    */
     int scs = get_softmodem_params()->numerology;
     const int nr_slots_frame = nr_slots_per_frame[scs];
     sl_nr_ue_mac_params_t *sl_mac =  mac->SL_MAC_PARAMS;
     NR_TDD_UL_DL_Pattern_t *tdd = &sl_mac->sl_TDD_config->pattern1;
     const int n_ul_slots_period = tdd ? tdd->nrofUplinkSlots + (tdd->nrofUplinkSymbols > 0 ? 1 : 0) : nr_slots_frame;

     uint16_t num_subch = sl_get_num_subch(mac->sl_tx_res_pool);
     bool is_feedback_slot = false;
     for (int i = 0; i < (n_ul_slots_period * num_subch); i++) {
        SL_sched_feedback_t  *sched_psfch = &mac->sl_info.list[0]->UE_sched_ctrl.sched_psfch[i];
        if (slotP == sched_psfch->feedback_slot) {
            LOG_D(NR_MAC, "%4d.%2d i = %d sched_psfch %p feedback slot %d\n", frameP, slotP, i, sched_psfch, sched_psfch->feedback_slot);
            is_feedback_slot = true;
            break;
        }
     }
     if ((slotP % psfch_period == 0) && (psfch_period == 2 || psfch_period == 4)) {
         if (is_feedback_slot) {
           sci_pdu->psfch_overhead.val =  1;
           LOG_D(NR_MAC, "%4d.%2d Setting psfch_overhead\n", frameP, slotP);
         } else
             sci_pdu->psfch_overhead.val = 0;
     } else if ((slotP % psfch_period != 0) && (psfch_period == 2 || psfch_period == 4))
        sci_pdu->psfch_overhead.val = 0;


     sci_pdu->reserved.val = mac->is_synced ? 1 : 0;
     sci_pdu->conflict_information_receiver.val = 0;
     sci_pdu->beta_offset_indicator = 0;
     sci2_pdu->harq_pid = cur_harq ? cur_harq->sl_harq_pid : 0;
     sci2_pdu->ndi = (1 - sci2_pdu->ndi) & 1;
     sci2_pdu->rv_index = 0;//nr_rv_round_map[cur_harq->round%4];
     sci2_pdu->source_id = mac->src_id;
     sci2_pdu->dest_id = dest;
     sci2_pdu->harq_feedback = 1; //rlc_status->bytes_in_buffer > 0 ? 1 : 0;
     LOG_D(NR_MAC, "%4d.%2d Comparing Setting harq_feedback %d bytes_in_buffer %d sl_harq_pid %d\n", frameP, slotP, sci2_pdu->harq_feedback, rlc_status->bytes_in_buffer, cur_harq ? cur_harq->sl_harq_pid : 0);
     sci2_pdu->cast_type = 1;
     if (format2 == NR_SL_SCI_FORMAT_2C || format2 == NR_SL_SCI_FORMAT_2A) {
       sci2_pdu->csi_req = (csi_acq && csi_req_slot) ? 1 : 0;
       LOG_D(NR_MAC, "%4d.%2d Setting sci2_pdu->csi_req %d\n", frameP, slotP, sci2_pdu->csi_req);
     }
     if (format2 == NR_SL_SCI_FORMAT_2B)
       sci2_pdu->zone_id = 0;
     // Fill in for R17: communication_range
     sci2_pdu->communication_range.val = 0;
     if (format2 == NR_SL_SCI_FORMAT_2C) {
       sci2_pdu->providing_req_ind = 0;
       // Fill in for R17 : resource combinations
       sci2_pdu->resource_combinations.val = 0;
       sci2_pdu->first_resource_location = 0;
       // Fill in for R17 : reference_slot_location
       sci2_pdu->reference_slot_location.val = 0;
       sci2_pdu->resource_set_type = 0;
       // Fill in for R17 : lowest_subchannel_indices
       sci2_pdu->lowest_subchannel_indices.val = 0;
     }
     // Set SLSCH
     *slsch_pdu_length_max = rlc_status->bytes_in_buffer;
     return true;
   }
  else
    LOG_D(NR_MAC, "Tx_slot_2 %4d.%2d: schedule_slsch 0\n", frameP, slotP);
  return false;
}

