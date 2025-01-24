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

#include "mac_defs.h"
#include "mac_proto.h"
#include "LAYER2/RLC/rlc.h"
#include "LAYER2/NR_MAC_COMMON/nr_mac_common.h"

sl_resource_info_t* get_resource_element(List_t* resource_list, frameslot_t sfn);

static uint16_t sl_adjust_ssb_indices(sl_ssb_timealloc_t *ssb_timealloc, uint32_t slot_in_16frames, uint16_t *ssb_slot_ptr)
{
  uint16_t ssb_slot = ssb_timealloc->sl_TimeOffsetSSB;
  uint16_t numssb = 0;
  *ssb_slot_ptr = 0;

  if (ssb_timealloc->sl_NumSSB_WithinPeriod == 0) {
    *ssb_slot_ptr = 0;
    return 0;
  }

  while (slot_in_16frames > ssb_slot) {
    numssb = numssb + 1;
    if (numssb < ssb_timealloc->sl_NumSSB_WithinPeriod)
      ssb_slot = ssb_slot + ssb_timealloc->sl_TimeInterval;
    else
      break;
  }

  *ssb_slot_ptr = ssb_slot;

  return numssb;
}

static uint8_t sl_get_elapsed_slots(uint32_t slot, uint32_t sl_slot_bitmap)
{
  uint8_t elapsed_slots = 0;

  for (int i = 0; i < slot; i++) {
    if (sl_slot_bitmap & (1 << i))
      elapsed_slots++;
  }

  return elapsed_slots;
}

static void sl_determine_slot_bitmap(sl_nr_ue_mac_params_t *sl_mac, int ue_id)
{

  sl_nr_phy_config_request_t *sl_cfg = &sl_mac->sl_phy_config.sl_config_req;

  uint8_t sl_scs = sl_cfg->sl_bwp_config.sl_scs;
  uint8_t num_slots_per_frame = 10 * (1 << sl_scs);
  uint8_t slot_type = 0;
  for (int i = 0; i < num_slots_per_frame; i++) {
    slot_type = sl_nr_ue_slot_select(sl_cfg, i, TDD);
    if (slot_type == NR_SIDELINK_SLOT) {
      sl_mac->N_SL_SLOTS_perframe += 1;
      sl_mac->sl_slot_bitmap |= (1 << i);
    }
  }

  sl_mac->future_ttis = calloc(num_slots_per_frame, sizeof(sl_stored_tti_req_t));

  LOG_I(NR_MAC,
        "[UE%d] SL-MAC: N_SL_SLOTS_perframe:%d, SL SLOT bitmap:%x\n",
        ue_id,
        sl_mac->N_SL_SLOTS_perframe,
        sl_mac->sl_slot_bitmap);
}

/* This function determines the number of sidelink slots in 1024 frames - DFN cycle
 * which can be used for determining reserved slots and REsource pool slots according to bitmap.
 * Sidelink slots are the uplink and mixed slots with sidelink support except the SSB slots.
 */
static uint32_t sl_determine_num_sidelink_slots(sl_nr_ue_mac_params_t *sl_mac, int ue_id, uint16_t *N_SSB_16frames)
{

  uint32_t N_SSB_1024frames = 0;
  uint32_t N_SL_SLOTS = 0;
  *N_SSB_16frames = 0;

  if (sl_mac->rx_sl_bch.status) {
    sl_ssb_timealloc_t *ssb_timealloc = &sl_mac->rx_sl_bch.ssb_time_alloc;
    *N_SSB_16frames += ssb_timealloc->sl_NumSSB_WithinPeriod;
    LOG_D(NR_MAC, "RX SSB Slots:%d\n", *N_SSB_16frames);
  }

  if (sl_mac->tx_sl_bch.status) {
    sl_ssb_timealloc_t *ssb_timealloc = &sl_mac->tx_sl_bch.ssb_time_alloc;
    *N_SSB_16frames += ssb_timealloc->sl_NumSSB_WithinPeriod;
    LOG_D(NR_MAC, "TX SSB Slots:%d\n", *N_SSB_16frames);
  }

  // Total SSB slots in SFN cycle (1024 frames)
  N_SSB_1024frames = SL_FRAME_NUMBER_CYCLE / SL_NR_SSB_REPETITION_IN_FRAMES * (*N_SSB_16frames);

  // Determine total number of Valid Sidelink slots which can be used for Respool in a SFN cycle (1024 frames)
  N_SL_SLOTS = (sl_mac->N_SL_SLOTS_perframe * SL_FRAME_NUMBER_CYCLE) - N_SSB_1024frames;

  LOG_I(NR_MAC,
        "[UE%d]SL-MAC:SSB slots in 1024 frames:%d, N_SL_SLOTS_perframe:%d, N_SL_SLOTs in 1024 frames:%d, SL SLOT bitmap:%x\n",
        ue_id,
        N_SSB_1024frames,
        sl_mac->N_SL_SLOTS_perframe,
        N_SL_SLOTS,
        sl_mac->sl_slot_bitmap);

  return N_SL_SLOTS;
}

/**
 * DETERMINE IF SLOT IS MARKED AS SSB SLOT
 * ACCORDING TO THE SSB TIME ALLOCATION PARAMETERS.
 * sl_numSSB_withinPeriod - NUM SSBS in 16frames
 * sl_timeoffset_SSB - time offset for first SSB at start of 16 frames cycle
 * sl_timeinterval - distance in slots between 2 SSBs
 */
uint8_t sl_determine_if_SSB_slot(uint16_t frame, uint16_t slot, uint16_t slots_per_frame, sl_bch_params_t *sl_bch)
{
  uint16_t frame_16 = frame % SL_NR_SSB_REPETITION_IN_FRAMES;
  uint32_t slot_in_16frames = (frame_16 * slots_per_frame) + slot;
  uint16_t sl_NumSSB_WithinPeriod = sl_bch->ssb_time_alloc.sl_NumSSB_WithinPeriod;
  uint16_t sl_TimeOffsetSSB = sl_bch->ssb_time_alloc.sl_TimeOffsetSSB;
  uint16_t sl_TimeInterval = sl_bch->ssb_time_alloc.sl_TimeInterval;
  uint16_t num_ssb = sl_bch->num_ssb, ssb_slot = sl_bch->ssb_slot;

#ifdef SL_DEBUG
  LOG_D(NR_MAC,
        "%d:%d. num_ssb:%d,ssb_slot:%d, %d-%d-%d, status:%d\n",
        frame,
        slot,
        sl_bch->num_ssb,
        sl_bch->ssb_slot,
        sl_NumSSB_WithinPeriod,
        sl_TimeOffsetSSB,
        sl_TimeInterval,
        sl_bch->status);
#endif

  if (sl_NumSSB_WithinPeriod && sl_bch->status) {
    if (slot_in_16frames == sl_TimeOffsetSSB) {
      num_ssb = 0;
      ssb_slot = sl_TimeOffsetSSB;
    }

    if (num_ssb < sl_NumSSB_WithinPeriod && slot_in_16frames == ssb_slot) {
      num_ssb += 1;
      ssb_slot = (num_ssb < sl_NumSSB_WithinPeriod) ? (ssb_slot + sl_TimeInterval) : sl_TimeOffsetSSB;

      sl_bch->ssb_slot = ssb_slot;
      sl_bch->num_ssb = num_ssb;

      LOG_D(NR_MAC, "%d:%d is a PSBCH SLOT. Next PSBCH Slot:%d, num_ssb:%d\n", frame, slot, sl_bch->ssb_slot, sl_bch->num_ssb);

      return 1;
    }
  }

  LOG_D(NR_MAC, "%d:%d is NOT a PSBCH SLOT. Next PSBCH Slot:%d, num_ssb:%d\n", frame, slot, sl_bch->ssb_slot, sl_bch->num_ssb);
  return 0;
}

static uint8_t sl_psbch_scheduler(sl_nr_ue_mac_params_t *sl_mac_params, int ue_id, int frame, int slot)
{

  uint8_t config_type = 0, is_psbch_rx_slot = 0, is_psbch_tx_slot = 0;

  sl_nr_phy_config_request_t *sl_cfg = &sl_mac_params->sl_phy_config.sl_config_req;
  uint16_t scs = sl_cfg->sl_bwp_config.sl_scs;
  uint16_t slots_per_frame = nr_slots_per_frame[scs];

  if (sl_mac_params->rx_sl_bch.status) {
    is_psbch_rx_slot = sl_determine_if_SSB_slot(frame, slot, slots_per_frame, &sl_mac_params->rx_sl_bch);

    if (is_psbch_rx_slot)
      config_type = SL_NR_CONFIG_TYPE_RX_PSBCH;

  } else if (sl_mac_params->tx_sl_bch.status) {
    is_psbch_tx_slot = sl_determine_if_SSB_slot(frame, slot, slots_per_frame, &sl_mac_params->tx_sl_bch);

    if (is_psbch_tx_slot)
      config_type = SL_NR_CONFIG_TYPE_TX_PSBCH;
  }

  sl_mac_params->future_ttis[slot].frame = frame;
  sl_mac_params->future_ttis[slot].slot = slot;
  sl_mac_params->future_ttis[slot].sl_action = config_type;

  LOG_D(NR_MAC, "[UE%d] SL-PSBCH SCHEDULER: %d:%d, config type:%d\n", ue_id, frame, slot, config_type);
  return config_type;
}

/*
 * This function calculates the indices based on the new timing (frame,slot)
 * acquired by the UE.
 * NUM SSB, SLOT_SSB needs to be calculated based on current timing
 */
static void sl_adjust_indices_based_on_timing(sl_nr_ue_mac_params_t *sl_mac,
                                              int ue_id,
                                              int frame, int slot,
                                              int slots_per_frame)
{

  uint8_t elapsed_slots = 0;

  elapsed_slots = sl_get_elapsed_slots(slot, sl_mac->sl_slot_bitmap);
  AssertFatal(elapsed_slots <= sl_mac->N_SL_SLOTS_perframe,
              "Elapsed slots cannot be > N_SL_SLOTS_perframe %d,%d\n",
              elapsed_slots,
              sl_mac->N_SL_SLOTS_perframe);

  uint16_t frame_16 = frame % SL_NR_SSB_REPETITION_IN_FRAMES;
  uint32_t slot_in_16frames = (frame_16 * slots_per_frame) + slot;
  LOG_I(NR_MAC,
        "[UE%d]PSBCH params adjusted based on current timing %d:%d. frame_16:%d, slot_in_16frames:%d\n",
        ue_id,
        frame,
        slot,
        frame_16,
        slot_in_16frames);

  // Adjust PSBCH Indices based on current timing
  if (sl_mac->rx_sl_bch.status) {
    sl_ssb_timealloc_t *ssb_timealloc = &sl_mac->rx_sl_bch.ssb_time_alloc;
    sl_mac->rx_sl_bch.num_ssb = sl_adjust_ssb_indices(ssb_timealloc, slot_in_16frames, &sl_mac->rx_sl_bch.ssb_slot);

    LOG_I(NR_MAC,
          "[UE%d]PSBCH RX params adjusted. NumSSB:%d, ssb_slot:%d\n",
          ue_id,
          sl_mac->rx_sl_bch.num_ssb,
          sl_mac->rx_sl_bch.ssb_slot);
  }

  if (sl_mac->tx_sl_bch.status) {
    sl_ssb_timealloc_t *ssb_timealloc = &sl_mac->tx_sl_bch.ssb_time_alloc;
    sl_mac->tx_sl_bch.num_ssb = sl_adjust_ssb_indices(ssb_timealloc, slot_in_16frames, &sl_mac->tx_sl_bch.ssb_slot);

    LOG_I(NR_MAC,
          "[UE%d]PSBCH TX params adjusted. NumSSB:%d, ssb_slot:%d\n",
          ue_id,
          sl_mac->tx_sl_bch.num_ssb,
          sl_mac->tx_sl_bch.ssb_slot);
  }
}

// Adjust indices as new timing is acquired
static void sl_actions_after_new_timing(sl_nr_ue_mac_params_t *sl_mac,
                                        int ue_id,
                                        int frame, int slot)
{

  uint8_t mu = sl_mac->sl_phy_config.sl_config_req.sl_bwp_config.sl_scs;
  uint8_t slots_per_frame = nr_slots_per_frame[mu];

  sl_determine_slot_bitmap(sl_mac, ue_id);

  sl_mac->N_SL_SLOTS = sl_determine_num_sidelink_slots(sl_mac, ue_id, &sl_mac->N_SSB_16frames);
  sl_adjust_indices_based_on_timing(sl_mac, ue_id, frame, slot, slots_per_frame);
}

static void nr_store_slsch_buffer(NR_UE_MAC_INST_t *mac, frame_t frame, sub_frame_t slot) {

  NR_SL_UEs_t *UE_info = &mac->sl_info;
  SL_UE_iterator(UE_info->list, UE) {
    NR_SL_UE_sched_ctrl_t *sched_ctrl = &UE->UE_sched_ctrl;
    sched_ctrl->num_total_bytes = 0;
    sched_ctrl->sl_pdus_total = 0;

    const int lcid = 4;
    sched_ctrl->rlc_status[lcid] = mac_rlc_status_ind(0, mac->src_id, 0, frame, slot, ENB_FLAG_NO, MBMS_FLAG_NO, 4, 0, 0);

    if (sched_ctrl->rlc_status[lcid].bytes_in_buffer == 0)
        continue;

    sched_ctrl->sl_pdus_total += sched_ctrl->rlc_status[lcid].pdus_in_buffer;
    sched_ctrl->num_total_bytes += sched_ctrl->rlc_status[lcid].bytes_in_buffer;
    LOG_D(MAC,
          "[%4d.%2d] SLSCH, RLC status for UE: %d bytes in buffer, total DL buffer size = %d bytes, %d total PDU bytes\n",
          frame,
          slot,
          sched_ctrl->rlc_status[lcid].bytes_in_buffer,
          sched_ctrl->num_total_bytes,
          sched_ctrl->sl_pdus_total);
  }
}

/*TS 38.321
A BSR shall be triggered if any of the following events occur:
- UL data, for a logical channel which belongs to an LCG, becomes available to the MAC entity; and either
   => here we don't implement exactly the same, there is no direct relation with new data came in the UE since last BSR

- this UL data belongs to a logical channel with higher priority than the priority of any logical channel
  containing available UL data which belong to any LCG; or
  => same, we don't know the last BSR content

- none of the logical channels which belong to an LCG contains any available UL data.
  in which case the BSR is referred below to as 'Regular BSR';

- UL resources are allocated and number of padding bits is equal to or larger than the size of the Buffer Status
Report MAC CE plus its subheader, in which case the BSR is referred below to as 'Padding BSR';

- retxBSR-Timer expires, and at least one of the logical channels which belong to an LCG contains UL data, in
which case the BSR is referred below to as 'Regular BSR';

- periodicBSR-Timer expires, in which case the BSR is referred below to as 'Periodic BSR'.

*/

void nr_update_bsr(NR_UE_MAC_INST_t *mac, uint32_t *LCG_bytes)
{
  bool bsr_regular_triggered = mac->scheduling_info.BSR_reporting_active & NR_BSR_TRIGGER_REGULAR;
  for (int i = 0; i < mac->lc_ordered_list.count; i++) {
    nr_lcordered_info_t *lc_info = mac->lc_ordered_list.array[i];
    int lcid = lc_info->lcid;
    NR_LC_SCHEDULING_INFO *lc_sched_info = get_scheduling_info_from_lcid(mac, lcid);
    int lcgid = lc_sched_info->LCGID;
    // check if UL data for a logical channel which belongs to a LCG becomes available for transmission
    if (lcgid != NR_INVALID_LCGID) {
      // Update waiting bytes for this LCG
      LCG_bytes[lcgid] += lc_sched_info->LCID_buffer_remain;
      if (!bsr_regular_triggered) {
        bsr_regular_triggered = true;
        trigger_regular_bsr(mac, lcid, lc_info->sr_DelayTimerApplied);
        LOG_D(NR_MAC, "[UE %d] MAC BSR Triggered\n", mac->ue_id);
      }
    }
  }
}

static bool get_control_info(NR_UE_MAC_INST_t *mac,
                             NR_SL_UE_sched_ctrl_t *sched_ctrl,
                             const int nr_slots_per_frame,
                             uint16_t frame,
                             uint16_t slot,
                             int16_t dest_id,
                             NR_SetupRelease_SL_PSFCH_Config_r16_t *configured_PSFCH) {
  int period = 0, offset = 0;
  sl_nr_ue_mac_params_t *sl_mac = mac->SL_MAC_PARAMS;
  // Determine current slot is csi-rs schedule slot
  bool csi_acq = !mac->SL_MAC_PARAMS->sl_CSI_Acquisition;
  bool is_harq_feedback = configured_PSFCH ? is_feedback_scheduled(mac, frame, slot) : false;
  NR_TDD_UL_DL_Pattern_t *tdd = &sl_mac->sl_TDD_config->pattern1;
  // Determine current slot is csi report schedule slot
  SL_CSI_Report_t *sl_csi_report = set_nr_ue_sl_csi_meas_periodicity(tdd, sched_ctrl, mac, dest_id, false);
  nr_ue_sl_csi_period_offset(sl_csi_report,
                              &period,
                              &offset);
  LOG_D(NR_MAC, "frame.slot %4d.%2d period %d offset %d\n", frame, slot, period, offset);
  bool csi_req_slot = !((nr_slots_per_frame * frame + slot - offset) % period);
  bool is_csi_report_sched_slot = ((sched_ctrl->sched_csi_report.frame == frame) &&
                                  (sched_ctrl->sched_csi_report.slot == slot));
  bool control_info = (is_harq_feedback || (csi_acq && csi_req_slot) || is_csi_report_sched_slot);

  LOG_D(NR_MAC, "frame.slot %4d.%2d harq_feedback %d, (csi_acq && csi_req_slot) %d, is_csi_report_sched_slot %d\n",
        frame, slot, is_harq_feedback, (csi_acq && csi_req_slot), is_csi_report_sched_slot);

  return control_info;
}

void preprocess(NR_UE_MAC_INST_t *mac,
                uint16_t frame,
                uint16_t slot,
                int *fb_frame,
                int *fb_slot,
                const NR_SL_BWP_ConfigCommon_r16_t *sl_bwp,
                NR_SetupRelease_SL_PSFCH_Config_r16_t *configured_PSFCH) {

  nr_store_slsch_buffer(mac, frame, slot);
  sl_nr_ue_mac_params_t *sl_mac = mac->SL_MAC_PARAMS;
  int scs = sl_mac->sl_phy_config.sl_config_req.sl_bwp_config.sl_scs;
  const int nr_slots_frame = nr_slots_per_frame[scs];

  NR_SL_UEs_t *UE_info = &mac->sl_info;
  SL_UE_iterator(UE_info->list, UE) {
    NR_SL_UE_sched_ctrl_t *sched_ctrl = &UE->UE_sched_ctrl;
    UE->mac_sl_stats.sl.current_bytes = 0;
    UE->mac_sl_stats.sl.current_rbs = 0;
    NR_sched_pssch_t *sched_pssch = &sched_ctrl->sched_pssch;
    sched_pssch->sl_harq_pid = configured_PSFCH ? sched_ctrl->retrans_sl_harq.head : -1;

    /* retransmission */
    if (sched_pssch->sl_harq_pid >= 0) {
      if (sched_ctrl->available_sl_harq.head < 0) {
        LOG_W(NR_MAC, "[UE][%4d.%2d] UE has no free SL HARQ process, skipping\n",
              frame,
              slot);
        continue;
      } else {
         sched_ctrl->sched_csi_report.active = false;
      }
    } else {
      if (sched_ctrl->available_sl_harq.head < 0) {
        LOG_W(NR_MAC, "[UE][%4d.%2d] UE has no free SL HARQ process, skipping\n",
              frame,
              slot);
        continue;
      }
      bool control_info = get_control_info(mac, sched_ctrl, nr_slots_frame, frame, slot, UE->uid, configured_PSFCH);
      LOG_D(NR_MAC, "sched_ctrl->num_total_bytes %d, control_info %d\n", sched_ctrl->num_total_bytes, control_info);
      /* Check SL buffer and control info, skip this UE if no bytes and no control info */
      if (sched_ctrl->num_total_bytes == 0) {
        if (!control_info)
          continue;
      }
    }

    /*
    * SLSCH tx computes feedback frame and slot, which will be used by transmitter of PSFCH after receiving SLSCH.
    * Transmitter of SLSCH stores the feedback frame and slot in harq process to use those in retreiving the feedback.
    */
    if (configured_PSFCH) {
      const uint8_t psfch_periods[] = {0, 1, 2, 4};
      NR_SL_PSFCH_Config_r16_t *sl_psfch_config = mac->sl_tx_res_pool->sl_PSFCH_Config_r16->choice.setup;
      long psfch_period = (sl_psfch_config->sl_PSFCH_Period_r16)
                            ? psfch_periods[*sl_psfch_config->sl_PSFCH_Period_r16] : 0;

      int rcv_tx_frame = (frame + ((slot + DURATION_RX_TO_TX) / nr_slots_frame)) % 1024;
      int rcv_tx_slot = (slot + DURATION_RX_TO_TX) % nr_slots_frame;
      int psfch_slot = get_feedback_slot(psfch_period, rcv_tx_slot);
      update_harq_lists(mac, frame, slot, UE);
      *fb_frame = rcv_tx_frame;
      *fb_slot = psfch_slot;
      LOG_D(NR_MAC, "Tx SLSCH %4d.%2d, Expected Feedback: %4d.%2d in current PSFCH: psfch_period %ld\n",
            frame,
            slot,
            *fb_frame,
            *fb_slot,
            psfch_period);
    }
    int locbw = sl_bwp->sl_BWP_Generic_r16->sl_BWP_r16->locationAndBandwidth;
    sched_pssch->mu = scs;
    sched_pssch->frame = frame;
    sched_pssch->slot = slot;
    sched_pssch->rbSize = NRRIV2BW(locbw, MAX_BWP_SIZE);
    sched_pssch->rbStart = NRRIV2PRBOFFSET(locbw, MAX_BWP_SIZE);
  }
}

bool nr_ue_sl_pssch_scheduler(NR_UE_MAC_INST_t *mac,
                              nr_sidelink_indication_t *sl_ind,
                              const NR_SL_BWP_ConfigCommon_r16_t *sl_bwp,
                              const NR_SL_ResourcePool_r16_t *sl_res_pool,
                              sl_nr_tx_config_request_t *tx_config,
                              sl_resource_info_t *resource,
                              uint8_t *config_type) {

  uint16_t slot = sl_ind->slot_tx;
  uint16_t frame = sl_ind->frame_tx;
  int feedback_frame, feedback_slot;
  int lcid = 4;
  int sdu_length = 0;
  uint16_t sdu_length_total = 0;
  uint8_t total_mac_pdu_header_len = 0;
  bool is_resource_allocated = false;
  *config_type = 0;

  sl_nr_ue_mac_params_t* sl_mac_params = mac->SL_MAC_PARAMS;
  NR_SetupRelease_SL_PSFCH_Config_r16_t *configured_PSFCH  = mac->sl_tx_res_pool->sl_PSFCH_Config_r16;
  if ((frame & 127) == 0 && slot == 0) {
    print_meas(&mac->rlc_data_req,"rlc_data_req",NULL,NULL);
  }
  if (sl_ind->slot_type != SIDELINK_SLOT_TYPE_TX) return is_resource_allocated;

  if (slot > 9 && get_nrUE_params()->sync_ref) return is_resource_allocated;

  if (slot < 10 && !get_nrUE_params()->sync_ref) return is_resource_allocated;

  LOG_D(NR_MAC,"[UE%d] SL-PSSCH SCHEDULER: Frame:SLOT %d:%d, slot_type:%d\n",
        sl_ind->module_id, frame, slot,sl_ind->slot_type);

  uint16_t slsch_pdu_length_max;
  tx_config->tx_config_list[0].tx_pscch_pssch_config_pdu.slsch_payload = mac->slsch_payload;

  NR_SL_UEs_t *UE_info = &mac->sl_info;

  if (*(UE_info->list) == NULL) {
    LOG_D(NR_MAC, "UE list is empty\n");
    return is_resource_allocated;
  }

  preprocess(mac, frame, slot, &feedback_frame, &feedback_slot, sl_bwp, configured_PSFCH);

  SL_UE_iterator(UE_info->list, UE) {
    NR_mac_dir_stats_t *sl_mac_stats = &UE->mac_sl_stats.sl;
    NR_SL_UE_sched_ctrl_t *sched_ctrl = &UE->UE_sched_ctrl;
    sl_mac_stats->current_bytes = 0;
    sl_mac_stats->current_rbs = 0;
    NR_sched_pssch_t *sched_pssch = &sched_ctrl->sched_pssch;
    int8_t harq_id = sched_pssch->sl_harq_pid;

    if (sched_pssch->rbSize <= 0)
      continue;

    NR_UE_sl_harq_t *cur_harq = NULL;

    if (harq_id < 0) {
      /* PP has not selected a specific HARQ Process, get a new one */
      harq_id = sched_ctrl->available_sl_harq.head;
      AssertFatal(harq_id >= 0,
                  "no free HARQ process available\n");
      remove_front_nr_list(&sched_ctrl->available_sl_harq);
      sched_pssch->sl_harq_pid = harq_id;
    } else {
      /* PP selected a specific HARQ process. Check whether it will be a new
      * transmission or a retransmission, and remove from the corresponding
      * list */
      if (sched_ctrl->sl_harq_processes[harq_id].round == 0)
        remove_nr_list(&sched_ctrl->available_sl_harq, harq_id);
      else
        remove_nr_list(&sched_ctrl->retrans_sl_harq, harq_id);
    }
    cur_harq = &sched_ctrl->sl_harq_processes[harq_id];
    DevAssert(!cur_harq->is_waiting);
    /* retransmission or bytes to send */
    if (configured_PSFCH && ((cur_harq->round != 0) || (sched_ctrl->num_total_bytes > 0))) {
      cur_harq->feedback_slot = feedback_slot;
      cur_harq->feedback_frame = feedback_frame;
      add_tail_nr_list(&sched_ctrl->feedback_sl_harq, harq_id);
      cur_harq->is_waiting = true;
      LOG_D(NR_MAC, "%4d.%2d Sending Data; Expecting feedback at %4d.%2d\n", frame, slot, feedback_frame, feedback_slot);
    }
    else
      add_tail_nr_list(&sched_ctrl->available_sl_harq, harq_id);
    cur_harq->sl_harq_pid = harq_id;
    /*
    The encoder checks for a change in ndi value everytime, since sci2 changes with every transmission,
    we oscillate the ndi value so the encoder treats the data as new data everytime.
    */
    cur_harq->ndi ^= 1;

    nr_schedule_slsch(mac, frame, slot, &mac->sci1_pdu, &mac->sci2_pdu, NR_SL_SCI_FORMAT_2A,
                      UE, &slsch_pdu_length_max, cur_harq, &sched_ctrl->rlc_status[lcid], resource);

    *config_type = SL_NR_CONFIG_TYPE_TX_PSCCH_PSSCH;
    tx_config->number_pdus = 1;
    tx_config->sfn = frame;
    tx_config->slot = slot;
    tx_config->tx_config_list[0].pdu_type = *config_type;
    fill_pssch_pscch_pdu(sl_mac_params,
                        &tx_config->tx_config_list[0].tx_pscch_pssch_config_pdu,
                        sl_bwp,
                        sl_res_pool,
                        &mac->sci1_pdu,
                        &mac->sci2_pdu,
                        slsch_pdu_length_max,
                        NR_SL_SCI_FORMAT_1A,
                        NR_SL_SCI_FORMAT_2A,
                        slot,
                        resource);
    sl_nr_tx_config_pscch_pssch_pdu_t *pscch_pssch_pdu = &tx_config->tx_config_list[0].tx_pscch_pssch_config_pdu;
    sched_pssch->R = pscch_pssch_pdu->target_coderate;
    sched_pssch->tb_size = pscch_pssch_pdu->tb_size;
    sched_pssch->sl_harq_pid = mac->sci2_pdu.harq_pid;
    sched_pssch->nrOfLayers = pscch_pssch_pdu->num_layers;
    sched_pssch->mcs = pscch_pssch_pdu->mcs;
    sched_pssch->Qm = pscch_pssch_pdu->mod_order;

    LOG_D(NR_MAC, "PSSCH: %4d.%2d SL sched %4d.%2d start %2d RBS %3d MCS %2d nrOfLayers %2d TBS %4d HARQ PID %2d round %d NDI %d sched %6d\n",
          frame,
          slot,
          sched_pssch->frame,
          sched_pssch->slot,
          sched_pssch->rbStart,
          sched_pssch->rbSize,
          sched_pssch->mcs,
          sched_pssch->nrOfLayers,
          sched_pssch->tb_size,
          sched_pssch->sl_harq_pid,
          cur_harq->round,
          cur_harq->ndi,
          sched_ctrl->sched_sl_bytes);

    /* Statistics */
    AssertFatal(cur_harq->round < sl_mac_params->sl_bler.harq_round_max, "Indexing ulsch_rounds[%d] is out of bounds for max harq round %d\n", cur_harq->round, sl_mac_params->sl_bler.harq_round_max);

    sl_mac_stats->rounds[cur_harq->round]++;
    if (cur_harq->round != 0) { // retransmission
      LOG_D(NR_MAC,
            "PSSCH: %d.%2d SL retransmission sched %d.%2d HARQ PID %d round %d NDI %d\n",
            frame,
            slot,
            sched_pssch->frame,
            sched_pssch->slot,
            sched_pssch->sl_harq_pid,
            cur_harq->round,
            cur_harq->ndi);
      sl_mac_stats->total_rbs_retx += sched_pssch->rbSize;
    } else { // initial transmission

      UE->mac_sl_stats.slsch_total_bytes_scheduled += sched_pssch->tb_size;
      /* save which time allocation and nrOfLayers have been used, to be used on
      * retransmissions */
      cur_harq->sched_pssch.nrOfLayers = sched_pssch->nrOfLayers;
      sched_ctrl->sched_sl_bytes += sched_pssch->tb_size;
      sl_mac_stats->total_rbs += sched_pssch->rbSize;


      int buflen = tx_config->tx_config_list[0].tx_pscch_pssch_config_pdu.tb_size;

      LOG_D(NR_MAC, "[UE%d] Initial TTI-%d:%d TX PSCCH_PSSCH REQ  TBS %d\n", sl_ind->module_id, frame, slot, buflen);

      uint8_t *pdu = (uint8_t *) cur_harq->transportBlock;
      int buflen_remain = buflen;

      NR_SLSCH_MAC_SUBHEADER_FIXED *sl_sch_subheader = (NR_SLSCH_MAC_SUBHEADER_FIXED *) pdu;
      sl_sch_subheader->V = 0;
      sl_sch_subheader->R = 0;
      sl_sch_subheader->SRC = mac->sci2_pdu.source_id;
      sl_sch_subheader->DST = mac->sci2_pdu.dest_id;
      pdu += sizeof(NR_SLSCH_MAC_SUBHEADER_FIXED);
      LOG_D(NR_MAC, "%4d.%2d Tx V %d, R %d, SRC %d, DST %d\n", frame, slot, sl_sch_subheader->V, sl_sch_subheader->R, sl_sch_subheader->SRC, sl_sch_subheader->DST);
      buflen_remain -= sizeof(NR_SLSCH_MAC_SUBHEADER_FIXED);
      LOG_D(NR_MAC, "buflen_remain after adding SL_SCH_MAC_SUBHEADER_FIXED %d\n", buflen_remain);
      const uint8_t sh_size = sizeof(NR_MAC_SUBHEADER_LONG);

      int num_sdus=0;
      if (sched_ctrl->num_total_bytes > 0) {
        if (sched_ctrl->rlc_status[lcid].bytes_in_buffer > 0) {
          while (buflen_remain > sh_size + 1) {

            // Pointer used to build the MAC sub-PDU headers in the ULSCH buffer for each SDU
            NR_MAC_SUBHEADER_LONG *header = (NR_MAC_SUBHEADER_LONG *) pdu;
            pdu += sh_size;
            buflen_remain -= sh_size;
            const rlc_buffer_occupancy_t ndata = min(sched_ctrl->rlc_status[lcid].bytes_in_buffer, buflen_remain);

            start_meas(&mac->rlc_data_req);

            sdu_length = mac_rlc_data_req(0,
                                          mac->src_id,
                                          0,
                                          frame,
                                          ENB_FLAG_NO,
                                          MBMS_FLAG_NO,
                                          lcid,
                                          ndata,
                                          (char *)pdu,
                                          0,
                                          0);
            stop_meas(&mac->rlc_data_req);
            AssertFatal(buflen_remain >= sdu_length, "In %s: LCID = 0x%02x RLC has segmented %d bytes but MAC has max %d remaining bytes\n",
                        __FUNCTION__,
                        lcid,
                        sdu_length,
                        buflen_remain);
            if (sdu_length > 0) {

              LOG_D(NR_MAC, "In %s: [UE %d] [%d.%d] SL-DXCH -> SLSCH, Generating SL MAC sub-PDU for SDU %d, length %d bytes, RB with LCID 0x%02x (buflen (TBS) %d bytes)\n",
                __FUNCTION__,
                0,
                frame,
                slot,
                num_sdus + 1,
                sdu_length,
                lcid,
                buflen);

              header->R = 0;
              header->F = 1;
              header->LCID = lcid;
              header->L = htons(sdu_length);
              pdu += sdu_length;
              sdu_length_total += sdu_length;
              total_mac_pdu_header_len += sh_size;
              buflen_remain -= sdu_length;
              LOG_D(NR_PHY, "buflen_remain %d, subtracting (sh_size + sdu_length) %d, total_mac_pdu_header_len %hhu sdu total length %d, sdu_length %d\n", buflen_remain, (sh_size + sdu_length), total_mac_pdu_header_len, sdu_length_total, sdu_length);
              num_sdus++;

            } else {
              pdu -= sh_size;
              buflen_remain += sh_size;
              LOG_D(NR_MAC, "In %s: no data to transmit for RB with LCID 0x%02x\n", __FUNCTION__, lcid);
              break;
            }
          }

          if (buflen_remain > 0) {
            NR_UE_MAC_CE_INFO *mac_ce_p = (NR_UE_MAC_CE_INFO *) pdu;
            // EpiSci TODO: Check this code block esp. tx_powerand P_CMAX
            int tx_power = 0;
            int P_CMAX = 0;
            // Call BSR procedure as described in Section 5.4.5 in 38.321
            // Check whether BSR is triggered before scheduling ULSCH
            uint32_t LCG_bytes[NR_MAX_NUM_LCGID] = {0};
            nr_update_bsr(mac, LCG_bytes);
            nr_ue_get_sdu_mac_ce_pre(mac, 0, frame, slot, 0, pdu, buflen_remain, LCG_bytes, mac_ce_p, tx_power, P_CMAX);
            buflen_remain -= (mac_ce_p->pdu_end - mac_ce_p->end_for_tailer);
            pdu += (mac_ce_p->pdu_end - mac_ce_p->end_for_tailer);
            LOG_D(NR_PHY, "buflen_remain %d, sdu_length_total %d, total_mac_pdu_header_len %d, adding tot_mac_ce_len %ld \n", buflen_remain, sdu_length_total, total_mac_pdu_header_len, (mac_ce_p->pdu_end - mac_ce_p->end_for_tailer));
          }
        }
      }
      uint8_t sizeof_csi_report = (sizeof(NR_MAC_SUBHEADER_FIXED) + sizeof(nr_sl_csi_report_t));
      LOG_D(NR_MAC, "%4d.%2d buflen_remain %d ative %d, report slots: %4d.%2d size %d\n",
            frame,
            slot,
            buflen_remain,
            sched_ctrl->sched_csi_report.active,
            sched_ctrl->sched_csi_report.frame,
            sched_ctrl->sched_csi_report.slot,
            sizeof_csi_report);

      if (sched_ctrl->sched_csi_report.active &&
          (sched_ctrl->sched_csi_report.frame == frame) &&
          (sched_ctrl->sched_csi_report.slot == slot)) {

        if (buflen_remain >= sizeof_csi_report) {
          ((NR_MAC_SUBHEADER_FIXED *) pdu)->R = 0;
          ((NR_MAC_SUBHEADER_FIXED *) pdu)->LCID = SL_SCH_LCID_SL_CSI_REPORT;
          pdu++;
          buflen_remain -= sizeof(NR_MAC_SUBHEADER_FIXED);
          ((nr_sl_csi_report_t *) pdu)->RI = sched_ctrl->sched_csi_report.ri;
          ((nr_sl_csi_report_t *) pdu)->CQI = sched_ctrl->sched_csi_report.cqi;
          ((nr_sl_csi_report_t *) pdu)->R = 0;
          if (!get_nrUE_params()->sync_ref)
            LOG_D(NR_MAC, "%4d.%2d Sending sl_csi_report with CQI %i, RI %i\n",
                 frame,
                 slot,
                 ((nr_sl_csi_report_t *) pdu)->CQI,
                 ((nr_sl_csi_report_t *) pdu)->RI);
          pdu++;
          buflen_remain -= sizeof(nr_sl_csi_report_t);
        }
        sched_ctrl->sched_csi_report.active = false;
      }

      if (buflen_remain > 0) {
        LOG_D(NR_MAC, "In %s filling remainder %d bytes to the UL PDU \n", __FUNCTION__, buflen_remain);
        ((NR_MAC_SUBHEADER_FIXED *) pdu)->R = 0;
        ((NR_MAC_SUBHEADER_FIXED *) pdu)->LCID = SL_SCH_LCID_SL_PADDING;
        pdu++;
        buflen_remain--;

        if (IS_SOFTMODEM_RFSIM) {
          for (int j = 0; j < buflen_remain; j++) {
              pdu[j] = (unsigned char) rand();
          }
        } else {
          memset(pdu, 0, buflen_remain);
        }
      }

      sl_mac_stats->current_bytes = sched_pssch->tb_size;
      sl_mac_stats->current_rbs = sched_pssch->rbSize;
      sl_mac_stats->total_bytes += pscch_pssch_pdu->tb_size;
      sl_mac_stats->num_mac_sdu += num_sdus;
      sl_mac_stats->total_sdu_bytes += sdu_length_total;

      /* Save information on MCS, TBS etc for the current initial transmission
      * so we have access to it when retransmitting */
      cur_harq->sched_pssch = *sched_pssch;
    } // end of initial transmission

    const uint32_t TBS = pscch_pssch_pdu->tb_size;
    memcpy(pscch_pssch_pdu->slsch_payload, cur_harq->transportBlock, TBS);
    // mark UE as scheduled
    sched_pssch->rbSize = 0;
    is_resource_allocated = true;
  }
  return is_resource_allocated;
}

void nr_ue_sl_pscch_rx_scheduler(nr_sidelink_indication_t *sl_ind,
                              const NR_SL_BWP_ConfigCommon_r16_t *sl_bwp,
                              const NR_SL_ResourcePool_r16_t *sl_res_pool,
                              sl_nr_rx_config_request_t *rx_config,
                              uint8_t *config_type,
                              bool sl_has_psfch) {

  *config_type = SL_NR_CONFIG_TYPE_RX_PSCCH;
  rx_config->number_pdus = 1;
  rx_config->sfn = sl_ind->frame_rx;
  rx_config->slot = sl_ind->slot_rx;
  rx_config->sl_rx_config_list[0].pdu_type = *config_type;
  config_pscch_pdu_rx(&rx_config->sl_rx_config_list[0].rx_pscch_config_pdu,
                       sl_bwp,
                       sl_res_pool,
                       sl_has_psfch);


   LOG_D(NR_MAC, "[UE%d] TTI-%d:%d RX PSCCH REQ \n", sl_ind->module_id,sl_ind->frame_rx, sl_ind->slot_rx);

}

static uint8_t sl_tx_scheduler(NR_UE_MAC_INST_t *mac,
                               int frame,
                               int slot,
                               sl_nr_tx_config_request_t *tx_config,
                               sl_resource_info_t *resource,
                               nr_sidelink_indication_t *sl_ind,
                               long psfch_period,
                               uint8_t mu) {

  //Check if reserved slot or a sidelink resource configured in Rx/Tx resource pool timeresource bitmap
  uint8_t tti_action = 0;
  bool is_resource_allocated = nr_ue_sl_pssch_scheduler(mac, sl_ind, mac->sl_bwp, mac->sl_tx_res_pool, tx_config, resource, &tti_action);

  if (is_resource_allocated && mac->sci2_pdu.csi_req) {
    nr_ue_sl_csi_rs_scheduler(mac, mu, mac->sl_bwp, tx_config, NULL, &tti_action);
    LOG_D(NR_MAC, "%4d.%2d Scheduling CSI-RS\n", frame, slot);
  }

  bool is_feedback_slot = mac->sl_tx_res_pool->sl_PSFCH_Config_r16 ? is_feedback_scheduled(mac, frame, slot) : false;

  if (is_resource_allocated && is_feedback_slot && mac->sl_tx_res_pool->sl_PSFCH_Config_r16->choice.setup) {
    if (is_feedback_slot) {
      nr_ue_sl_psfch_scheduler(mac, frame, slot, psfch_period, sl_ind, mac->sl_bwp, tx_config, &tti_action);
      reset_sched_psfch(mac, frame, slot);
    }
  }

  tx_config->sfn = frame;
  tx_config->slot = slot;
  sl_nr_ue_mac_params_t *sl_mac_params = mac->SL_MAC_PARAMS;
  sl_mac_params->future_ttis[slot].frame = frame;
  sl_mac_params->future_ttis[slot].slot = slot;
  sl_mac_params->future_ttis[slot].sl_action = tti_action;

  return tti_action;
}

static uint8_t sl_rx_scheduler(NR_UE_MAC_INST_t *mac,
                               int frame,
                               int slot,
                               sl_nr_rx_config_request_t *rx_config,
                               nr_sidelink_indication_t *sl_ind,
                               long psfch_period,
                               uint8_t mu) {
  static uint16_t prev_slot = 0;
  uint8_t tti_action = 0;
  sl_nr_ue_mac_params_t *sl_mac_params = mac->SL_MAC_PARAMS;

  if (prev_slot != slot) {
    frameslot_t fs;
    fs.frame = frame;
    fs.slot = slot;
    uint64_t rx_abs_slot = normalize(&fs, mu);
    uint8_t pool_id = 0;
    SL_ResourcePool_params_t *sl_rx_rsrc_pool = sl_mac_params->sl_RxPool[pool_id];
    uint16_t phy_map_sz = ((sl_rx_rsrc_pool->phy_sl_bitmap.size << 3) - sl_rx_rsrc_pool->phy_sl_bitmap.bits_unused);
    bool sl_has_psfch = slot_has_psfch(mac, &sl_rx_rsrc_pool->phy_sl_bitmap, rx_abs_slot, psfch_period, phy_map_sz, mac->SL_MAC_PARAMS->sl_TDD_config);
    LOG_D(NR_MAC, "%4d.%2d RX sl_has_psfch %d, psfch_period %ld\n", frame, slot, sl_has_psfch, psfch_period);
    nr_ue_sl_pscch_rx_scheduler(sl_ind, mac->sl_bwp, mac->sl_rx_res_pool, rx_config, &tti_action, sl_has_psfch);
    prev_slot = slot;
  }

  rx_config->sfn = frame;
  rx_config->slot = slot;
  sl_mac_params->future_ttis[slot].frame = frame;
  sl_mac_params->future_ttis[slot].slot = slot;
  sl_mac_params->future_ttis[slot].sl_action = tti_action;

  return tti_action;
}

static void sl_schedule_rx_actions(nr_sidelink_indication_t *sl_ind, NR_UE_MAC_INST_t *mac, sl_nr_rx_config_request_t *rx_config)
{

  sl_nr_ue_mac_params_t *sl_mac = mac->SL_MAC_PARAMS;
  int ue_id = mac->ue_id;
  int rx_action = 0;

  if (sl_ind->sci_ind != NULL) {
    // TBD..
  } else {
    rx_action = sl_mac->future_ttis[sl_ind->slot_rx].sl_action;
  }

  if (rx_action == SL_NR_CONFIG_TYPE_RX_PSBCH) {
    rx_config->number_pdus = 1;
    rx_config->sl_rx_config_list[0].pdu_type = rx_action;

    LOG_I(NR_MAC, "[UE%d] %d:%d CMD to PHY: RX PSBCH \n", ue_id, sl_ind->frame_rx, sl_ind->slot_rx);

  } else if (rx_action >= SL_NR_CONFIG_TYPE_RX_PSCCH && rx_action <= SL_NR_CONFIG_TYPE_RX_PSSCH_SLSCH) {
    // TBD..

  } else if (rx_action == SL_NR_CONFIG_TYPE_RX_PSFCH) {
    // TBD..
  }

  if (rx_config->number_pdus) {
    AssertFatal(sl_ind->slot_type == SIDELINK_SLOT_TYPE_RX || sl_ind->slot_type == SIDELINK_SLOT_TYPE_BOTH,
                "RX action cannot be scheduled in non Sidelink RX slot\n");

    nr_scheduled_response_t scheduled_response = {.sl_rx_config = rx_config,
                                                  .module_id = sl_ind->module_id,
                                                  .CC_id = sl_ind->cc_id,
                                                  .phy_data = sl_ind->phy_data,
                                                  .mac = mac};

    sl_mac->future_ttis[sl_ind->slot_rx].sl_action = 0;

    if ((mac->if_module != NULL) && (mac->if_module->scheduled_response != NULL))
      mac->if_module->scheduled_response(&scheduled_response);
  }
}

static void sl_schedule_tx_actions(nr_sidelink_indication_t *sl_ind, NR_UE_MAC_INST_t *mac, sl_nr_tx_config_request_t *tx_config)
{

  sl_nr_ue_mac_params_t *sl_mac = mac->SL_MAC_PARAMS;
  int ue_id = mac->ue_id;
  nr_scheduled_response_t scheduled_response;
  memset(&scheduled_response,0, sizeof(nr_scheduled_response_t));

  int tx_action = 0;
  tx_action = sl_mac->future_ttis[sl_ind->slot_tx].sl_action;

  if (tx_action == SL_NR_CONFIG_TYPE_TX_PSBCH) {
    tx_config->number_pdus = 1;
    tx_config->tx_config_list[0].pdu_type = tx_action;
    tx_config->tx_config_list[0].tx_psbch_config_pdu.tx_slss_id = sl_mac->tx_sl_bch.slss_id;
    tx_config->tx_config_list[0].tx_psbch_config_pdu.psbch_tx_power = 0; // TBD...
    memcpy(tx_config->tx_config_list[0].tx_psbch_config_pdu.psbch_payload, sl_mac->tx_sl_bch.sl_mib, 4);

    LOG_I(NR_MAC, "[UE%d] %d:%d CMD to PHY: TX PSBCH \n", ue_id, sl_ind->frame_tx, sl_ind->slot_tx);

  } else if ((tx_action == SL_NR_CONFIG_TYPE_TX_PSCCH_PSSCH) ||
             (tx_action == SL_NR_CONFIG_TYPE_TX_PSCCH_PSSCH_PSFCH) ||
             (tx_action == SL_NR_CONFIG_TYPE_TX_PSCCH_PSSCH_CSI_RS)) {
    tx_config->number_pdus = 1;
    fill_scheduled_response(&scheduled_response, NULL, NULL, NULL, NULL, tx_config, sl_ind->module_id, 0, sl_ind->frame_tx, sl_ind->slot_tx, sl_ind->phy_data);
  }

  if (tx_config->number_pdus == 1) {
    AssertFatal(sl_ind->slot_type == SIDELINK_SLOT_TYPE_TX || sl_ind->slot_type == SIDELINK_SLOT_TYPE_BOTH,
                "TX action cannot be scheduled in non Sidelink TX slot\n");

    nr_scheduled_response_t scheduled_response = {.sl_tx_config = tx_config,
                                                  .module_id = sl_ind->module_id,
                                                  .CC_id = sl_ind->cc_id,
                                                  .phy_data = sl_ind->phy_data,
                                                  .mac = mac};

    sl_mac->future_ttis[sl_ind->slot_tx].sl_action = 0;

    if ((mac->if_module != NULL) && (mac->if_module->scheduled_response != NULL))
      mac->if_module->scheduled_response(&scheduled_response);
  }
}

void nr_ue_sidelink_scheduler(nr_sidelink_indication_t *sl_ind, NR_UE_MAC_INST_t *mac)
{
  AssertFatal(sl_ind != NULL, "sl_indication cannot be NULL\n");
  sl_nr_ue_mac_params_t *sl_mac = mac->SL_MAC_PARAMS;
  int ue_id = mac->ue_id;

  LOG_D(NR_MAC,
        "[UE%d]SL-SCHEDULER: RX %d-%d- TX %d-%d. slot_type:%d\n",
        ue_id,
        sl_ind->frame_rx,
        sl_ind->slot_rx,
        sl_ind->frame_tx,
        sl_ind->slot_tx,
        sl_ind->slot_type);

  // Adjust indices as new timing is acquired
  if (sl_mac->timing_acquired) {
    sl_actions_after_new_timing(sl_mac, ue_id, sl_ind->frame_tx, sl_ind->slot_tx);
    sl_mac->timing_acquired = false;
  }

  sl_nr_rx_config_request_t rx_config;
  sl_nr_tx_config_request_t tx_config;

  rx_config.number_pdus = 0;
  tx_config.number_pdus = 0;

  if (sl_ind->slot_type == SIDELINK_SLOT_TYPE_TX || sl_ind->slot_type == SIDELINK_SLOT_TYPE_BOTH) {
    int frame = sl_ind->frame_tx;
    int slot = sl_ind->slot_tx;
    int is_sl_slot = 0;
    is_sl_slot = sl_mac->sl_slot_bitmap & (1 << slot);

    if (is_sl_slot) {
      frameslot_t frame_slot;
      frame_slot.frame = frame;
      frame_slot.slot = slot;

      sl_resource_info_t *resource = NULL;
      if (mac->sl_candidate_resources && mac->sl_candidate_resources->size > 0) {
        LOG_D(NR_MAC, "%4d.%2d sl_candidate_resources %p size %ld, capacity %ld slot_type %d\n", frame, slot, mac->sl_candidate_resources, mac->sl_candidate_resources->size, mac->sl_candidate_resources->capacity, sl_ind->slot_type);
        resource = get_resource_element(mac->sl_candidate_resources, frame_slot);
        if (resource) {
          LOG_D(NR_MAC, "SELECTED_RESOURCE %4d.%2d slot_type %d, num_sl_pscch_rbs %d, sl_max_num_per_reserve %d, sl_min_time_gap_psfch %d, sl_pscch_sym_start %d, \
                sl_pscch_sym_len %d, sl_psfch_period %d, sl_pssch_sym_start %d, sl_pssch_sym_len %d, sl_subchan_len %d, sl_subchan_size %d\n",
                resource->sfn.frame, resource->sfn.slot, sl_ind->slot_type,
                resource->num_sl_pscch_rbs,
                resource->sl_max_num_per_reserve,
                resource->sl_min_time_gap_psfch,
                resource->sl_pscch_sym_start,
                resource->sl_pscch_sym_len,
                resource->sl_psfch_period,
                resource->sl_pssch_sym_start,
                resource->sl_pssch_sym_len,
                resource->sl_subchan_len,
                resource->sl_subchan_size);
        }
      }

      nr_sl_transmission_params_t *sl_tx_params = &sl_mac->mac_tx_params;
      sl_nr_phy_config_request_t *sl_cfg = &sl_mac->sl_phy_config.sl_config_req;
      uint8_t mu = sl_cfg->sl_bwp_config.sl_scs;
      uint16_t p_prime_rsvp_tx = time_to_slots(mu, sl_tx_params->resel_counter);
      static int8_t is_rsrc_selected = false;

      if (mac->rsc_selection_method == c1 ||
          mac->rsc_selection_method == c4 ||
          mac->rsc_selection_method == c5 ||
          mac->rsc_selection_method == c7) {
        LOG_D(NR_MAC, "%4d.%2d is_rsrc_selected %d, reselection_timer %d, p_prime_rsvp_tx %d, slot_type %d\n",
              frame, slot, is_rsrc_selected, mac->reselection_timer, p_prime_rsvp_tx, sl_ind->slot_type);
        if(is_rsrc_selected && (mac->reselection_timer < p_prime_rsvp_tx)) {
          mac->reselection_timer++;
        } else {
          if (mac->reselection_timer < p_prime_rsvp_tx) {
            mac->sl_candidate_resources = get_candidate_resources(&frame_slot, mac, &mac->sl_sensing_data, &mac->sl_transmit_history);
            if (mac->sl_candidate_resources) {
              LOG_D(NR_MAC, "%4d.%2d Returned resources %p\n", frame, slot, mac->sl_candidate_resources);
              print_candidate_list(mac->sl_candidate_resources, __LINE__);
            }
            is_rsrc_selected = true;
          } else {
            mac->reselection_timer = 0;
            is_rsrc_selected = false;
          }
        }
      }

      uint8_t tti_action = 0;

      NR_SL_PSFCH_Config_r16_t *sl_psfch_config = mac->sl_tx_res_pool->sl_PSFCH_Config_r16 ? mac->sl_tx_res_pool->sl_PSFCH_Config_r16->choice.setup : NULL;
      const uint8_t psfch_periods[] = {0,1,2,4};
      long psfch_period = (sl_psfch_config && sl_psfch_config->sl_PSFCH_Period_r16)
                          ? psfch_periods[*sl_psfch_config->sl_PSFCH_Period_r16] : 0;

      // Check if PSBCH slot and PSBCH should be transmitted or Received
      tti_action = sl_psbch_scheduler(sl_mac, ue_id, frame, slot);

      // TBD .. Check for Actions coming out of TX resource pool
      if (resource && mac->is_synced && !tti_action && sl_mac->sl_TxPool[0])
        tti_action = sl_tx_scheduler(mac, frame, slot, &tx_config, resource, sl_ind, psfch_period, mu);

      //TBD .. Check for Actions coming out of RX resource pool
      if (!tti_action && sl_mac->sl_RxPool[0])
        tti_action = sl_rx_scheduler(mac, frame, slot, &rx_config, sl_ind, psfch_period, mu);

      LOG_D(NR_MAC, "[UE%d]SL-SCHED: TTI - %d:%d scheduled action:%d\n", ue_id, frame, slot, tti_action);

    } else {
      AssertFatal(1 == 0, "TX SLOT not a sidelink slot. Should not occur\n");
    }

    // Schedule the Tx actions if any
    sl_schedule_tx_actions(sl_ind, mac, &tx_config);
  }

  if (sl_ind->slot_type == SIDELINK_SLOT_TYPE_RX || sl_ind->slot_type == SIDELINK_SLOT_TYPE_BOTH)
    sl_schedule_rx_actions(sl_ind, mac, &rx_config);
}

void print_candidate_list(List_t *candidate_resources, int line) {
  for (int i = 0; i < candidate_resources->size; i++) {
    sl_resource_info_t *itr_rsrc = (sl_resource_info_t*)((char*)candidate_resources->data + i * candidate_resources->element_size);
    LOG_D(NR_MAC, "line %d, %4d.%2d, %ld, sl_subchan_len %d\n", line, itr_rsrc->sfn.frame, itr_rsrc->sfn.slot, normalize(&itr_rsrc->sfn, 1), itr_rsrc->sl_subchan_len);
  }
}

void print_reserved_list(List_t *candidate_resources, int line) {
  for (int i = 0; i < candidate_resources->size; i++) {
    reserved_resource_t *itr_rsrc = (reserved_resource_t*)((char*)candidate_resources->data + i * candidate_resources->element_size);
    LOG_D(NR_MAC, "line %d, %4d.%2d, %ld, sl_subchan_len %d\n", line, itr_rsrc->sfn.frame, itr_rsrc->sfn.slot, normalize(&itr_rsrc->sfn, 1), itr_rsrc->sb_ch_length);
  }
}

void print_sensing_data_list(List_t *sensing_data, int line) {
  for (int i = 0; i < sensing_data->size; i++) {
    sensing_data_t *itr_rsrc = (sensing_data_t*)((char*)sensing_data->data + i * sensing_data->element_size);
    LOG_D(NR_MAC, "line %d, %4d.%2d, %ld, sl_subchan_len %d\n", line, itr_rsrc->frame_slot.frame, itr_rsrc->frame_slot.slot, normalize(&itr_rsrc->frame_slot, 1), itr_rsrc->subch_len);
  }
}

sl_resource_info_t* get_resource_element(List_t* resource_list, frameslot_t sfn) {
  for (int i = 0; i < resource_list->size; i++) {
    sl_resource_info_t *itr_rsrc = (sl_resource_info_t*)((char*)resource_list->data + i * resource_list->element_size);
    LOG_D(NR_MAC, "%s %4d.%2d, %ld, sl_subchan_len %d, current sfn %4d.%2d\n",
          __FUNCTION__, itr_rsrc->sfn.frame, itr_rsrc->sfn.slot, normalize(&itr_rsrc->sfn, 1), itr_rsrc->sl_subchan_len, sfn.frame, sfn.slot);
    // TODO: currently, following condition is based on num_subchan = 1, needs to update for multi-subchannels
    if (itr_rsrc->sfn.frame == sfn.frame && itr_rsrc->sfn.slot == sfn.slot) {
      return itr_rsrc;
    }
  }
  return NULL;
}