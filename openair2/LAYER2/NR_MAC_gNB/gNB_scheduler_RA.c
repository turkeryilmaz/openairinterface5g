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

/*! \file     gNB_scheduler_RA.c
 * \brief     primitives used for random access
 * \author    Guido Casati
 * \date      2019
 * \email:    guido.casati@iis.fraunhofer.de
 * \version
 */

#include "common/platform_types.h"
#include "uper_decoder.h"

/* MAC */
#include "nr_mac_gNB.h"
#include "NR_MAC_gNB/mac_proto.h"
#include "openair2/COMMON/mac_messages_types.h"

/* Utils */
#include "common/utils/LOG/log.h"
#include "common/utils/LOG/vcd_signal_dumper.h"
#include "common/utils/nr/nr_common.h"
#include "UTIL/OPT/opt.h"

/* rlc */
#include "openair2/LAYER2/nr_rlc/nr_rlc_oai_api.h"

#include <executables/softmodem-common.h>

// forward declaration of functions used in this file
static void fill_msg3_pusch_pdu(nfapi_nr_pusch_pdu_t *pusch_pdu,
                                NR_ServingCellConfigCommon_t *scc,
                                NR_UE_info_t *UE,
                                int startSymbolAndLength,
                                int scs,
                                int bwp_size,
                                int bwp_start,
                                int mappingtype,
                                int fh);
static void nr_fill_rar(uint8_t Mod_idP, NR_UE_info_t *UE, uint8_t *dlsch_buffer, nfapi_nr_pusch_pdu_t *pusch_pdu);

static const float ssb_per_rach_occasion[8] = {0.125, 0.25, 0.5, 1, 2, 4, 8};

static int16_t ssb_index_from_prach(module_id_t module_idP,
                                    frame_t frameP,
                                    slot_t slotP,
                                    uint16_t preamble_index,
                                    uint8_t freq_index,
                                    uint8_t symbol)
{
  gNB_MAC_INST *gNB = RC.nrmac[module_idP];
  NR_COMMON_channels_t *cc = &gNB->common_channels[0];
  NR_ServingCellConfigCommon_t *scc = cc->ServingCellConfigCommon;
  nfapi_nr_config_request_scf_t *cfg = &RC.nrmac[module_idP]->config[0];
  NR_RACH_ConfigCommon_t *rach_ConfigCommon = scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup;
  uint8_t config_index = rach_ConfigCommon->rach_ConfigGeneric.prach_ConfigurationIndex;
  uint8_t fdm = cfg->prach_config.num_prach_fd_occasions.value;
  
  uint8_t total_RApreambles = MAX_NUM_NR_PRACH_PREAMBLES;
  if (rach_ConfigCommon->totalNumberOfRA_Preambles != NULL)
    total_RApreambles = *rach_ConfigCommon->totalNumberOfRA_Preambles;
  
  float  num_ssb_per_RO = ssb_per_rach_occasion[cfg->prach_config.ssb_per_rach.value];	
  uint16_t start_symbol_index = 0;
  uint8_t temp_start_symbol = 0;
  uint16_t RA_sfn_index = -1;
  uint16_t prach_occasion_id = -1;
  uint8_t num_active_ssb = cc->num_active_ssb;
  NR_MsgA_ConfigCommon_r16_t *msgacc = NULL;
  if (scc->uplinkConfigCommon->initialUplinkBWP->ext1 && scc->uplinkConfigCommon->initialUplinkBWP->ext1->msgA_ConfigCommon_r16)
    msgacc = scc->uplinkConfigCommon->initialUplinkBWP->ext1->msgA_ConfigCommon_r16->choice.setup;
  const int ul_mu = scc->uplinkConfigCommon->frequencyInfoUL->scs_SpecificCarrierList.list.array[0]->subcarrierSpacing;
  const int mu = nr_get_prach_or_ul_mu(msgacc, rach_ConfigCommon, ul_mu);
  frequency_range_t freq_range = get_freq_range_from_arfcn(scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencyPointA);
  get_nr_prach_sched_from_info(cc->prach_info, config_index, frameP, slotP, mu, freq_range, &RA_sfn_index, cc->frame_type);

  uint8_t index = 0, slot_index = 0;
  for (slot_index = 0; slot_index < cc->prach_info.N_RA_slot; slot_index++) {
    if (cc->prach_info.N_RA_slot <= 1) { // 1 PRACH slot in a subframe
       if((mu == 1) || (mu == 3))
         slot_index = 1; // For scs = 30khz and 120khz
    }
    for (int i = 0; i < cc->prach_info.N_t_slot; i++) {
      temp_start_symbol = (cc->prach_info.start_symbol + i * cc->prach_info.N_dur + 14 * slot_index) % 14;
      if(symbol == temp_start_symbol) {
        start_symbol_index = i;
        break;
      }
    }
  }
  if (cc->prach_info.N_RA_slot <= 1) { // 1 PRACH slot in a subframe
    if((mu == 1) || (mu == 3))
      slot_index = 0; // For scs = 30khz and 120khz
  }
  int config_period = cc->prach_info.x;
  //  prach_occasion_id = subframe_index * N_t_slot * N_RA_slot * fdm + N_RA_slot_index * N_t_slot * fdm + freq_index + fdm * start_symbol_index;
  prach_occasion_id =
      (((frameP % (cc->max_association_period * config_period)) / config_period) * cc->total_prach_occasions_per_config_period)
      + (RA_sfn_index + slot_index) * cc->prach_info.N_t_slot * fdm + start_symbol_index * fdm + freq_index;

  //one SSB have more than one continuous RO
  if(num_ssb_per_RO <= 1)
    index = (int) (prach_occasion_id / (int)(1 / num_ssb_per_RO)) % num_active_ssb;
  //one RO is shared by one or more SSB
  else if (num_ssb_per_RO > 1) {
    index = (prach_occasion_id * (int)num_ssb_per_RO) % num_active_ssb;
    for(int j = 0; j < num_ssb_per_RO; j++) {
      if(preamble_index <  ((j + 1) * cc->cb_preambles_per_ssb))
        index = index + j;
    }
  }

  LOG_D(NR_MAC, "Frame %d, Slot %d: Prach Occasion id = %d ssb per RO = %f number of active SSB %u index = %d fdm %u symbol index %u freq_index %u total_RApreambles %u\n",
        frameP, slotP, prach_occasion_id, num_ssb_per_RO, num_active_ssb, index, fdm, start_symbol_index, freq_index, total_RApreambles);

  return index;
}

//Compute Total active SSBs and RO available
void find_SSB_and_RO_available(gNB_MAC_INST *nrmac)
{
  /* already mutex protected through nr_mac_config_scc() */
  //NR_SCHED_ENSURE_LOCKED(&nrmac->sched_lock);

  NR_COMMON_channels_t *cc = &nrmac->common_channels[0];
  NR_ServingCellConfigCommon_t *scc = cc->ServingCellConfigCommon;
  nfapi_nr_config_request_scf_t *cfg = &nrmac->config[0];
  NR_RACH_ConfigCommon_t *rach_ConfigCommon = scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup;
  uint8_t config_index = rach_ConfigCommon->rach_ConfigGeneric.prach_ConfigurationIndex;
  uint16_t unused_RA_occasion, repetition = 0;
  uint8_t num_active_ssb = 0;

  struct NR_RACH_ConfigCommon__ssb_perRACH_OccasionAndCB_PreamblesPerSSB *ssb_perRACH_OccasionAndCB_PreamblesPerSSB = rach_ConfigCommon->ssb_perRACH_OccasionAndCB_PreamblesPerSSB;

  switch (ssb_perRACH_OccasionAndCB_PreamblesPerSSB->present) {
    case NR_RACH_ConfigCommon__ssb_perRACH_OccasionAndCB_PreamblesPerSSB_PR_oneEighth:
      cc->cb_preambles_per_ssb = 4 * (ssb_perRACH_OccasionAndCB_PreamblesPerSSB->choice.oneEighth + 1);
      break;
    case NR_RACH_ConfigCommon__ssb_perRACH_OccasionAndCB_PreamblesPerSSB_PR_oneFourth:
      cc->cb_preambles_per_ssb = 4 * (ssb_perRACH_OccasionAndCB_PreamblesPerSSB->choice.oneFourth + 1);
      break;
    case NR_RACH_ConfigCommon__ssb_perRACH_OccasionAndCB_PreamblesPerSSB_PR_oneHalf:
      cc->cb_preambles_per_ssb = 4 * (ssb_perRACH_OccasionAndCB_PreamblesPerSSB->choice.oneHalf + 1);
      break;
    case NR_RACH_ConfigCommon__ssb_perRACH_OccasionAndCB_PreamblesPerSSB_PR_one:
      cc->cb_preambles_per_ssb = 4 * (ssb_perRACH_OccasionAndCB_PreamblesPerSSB->choice.one + 1);
      break;
    case NR_RACH_ConfigCommon__ssb_perRACH_OccasionAndCB_PreamblesPerSSB_PR_two:
      cc->cb_preambles_per_ssb = 4 * (ssb_perRACH_OccasionAndCB_PreamblesPerSSB->choice.two + 1);
      break;
    case NR_RACH_ConfigCommon__ssb_perRACH_OccasionAndCB_PreamblesPerSSB_PR_four:
      cc->cb_preambles_per_ssb = ssb_perRACH_OccasionAndCB_PreamblesPerSSB->choice.four;
      break;
    case NR_RACH_ConfigCommon__ssb_perRACH_OccasionAndCB_PreamblesPerSSB_PR_eight:
      cc->cb_preambles_per_ssb = ssb_perRACH_OccasionAndCB_PreamblesPerSSB->choice.eight;
      break;
    case NR_RACH_ConfigCommon__ssb_perRACH_OccasionAndCB_PreamblesPerSSB_PR_sixteen:
      cc->cb_preambles_per_ssb = ssb_perRACH_OccasionAndCB_PreamblesPerSSB->choice.sixteen;
      break;
    default:
      AssertFatal(1 == 0, "Unsupported ssb_perRACH_config %d\n", ssb_perRACH_OccasionAndCB_PreamblesPerSSB->present);
      break;
  }

  // prach is scheduled according to configuration index and tables 6.3.3.2.2 to 6.3.3.2.4
  frequency_range_t freq_range = get_freq_range_from_arfcn(scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencyPointA);
  nr_prach_info_t prach_info =  get_nr_prach_occasion_info_from_index(config_index, freq_range, cc->frame_type);

  float num_ssb_per_RO = ssb_per_rach_occasion[cfg->prach_config.ssb_per_rach.value];	
  uint8_t fdm = cfg->prach_config.num_prach_fd_occasions.value;
  uint64_t L_ssb = (((uint64_t) cfg->ssb_table.ssb_mask_list[0].ssb_mask.value) << 32) | cfg->ssb_table.ssb_mask_list[1].ssb_mask.value;
  uint32_t total_RA_occasions = prach_info.N_RA_sfn * prach_info.N_t_slot * prach_info.N_RA_slot * fdm;

  for(int i = 0; i < 64; i++) {
    if ((L_ssb >> (63 - i)) & 0x01) { // only if the bit of L_ssb at current ssb index is 1
      cc->ssb_index[num_active_ssb] = i;
      num_active_ssb++;
    }
  }

  cc->total_prach_occasions_per_config_period = total_RA_occasions;
  for (int i = 1; (1 << (i - 1)) <= prach_info.max_association_period; i++) {
    cc->max_association_period = (1 << (i - 1));
    total_RA_occasions = total_RA_occasions * cc->max_association_period;
    if(total_RA_occasions >= (int) (num_active_ssb / num_ssb_per_RO)) {
      repetition = (uint16_t)((total_RA_occasions * num_ssb_per_RO) / num_active_ssb);
      break;
    }
  }

  unused_RA_occasion = total_RA_occasions - (int)((num_active_ssb * repetition) / num_ssb_per_RO);
  cc->total_prach_occasions = total_RA_occasions - unused_RA_occasion;
  cc->num_active_ssb = num_active_ssb;
  cc->prach_info = prach_info;

  LOG_D(NR_MAC,
        "Total available RO %d, num of active SSB %d: unused RO = %d association_period %u N_RA_sfn %u "
        "total_prach_occasions_per_config_period %u\n",
        cc->total_prach_occasions,
        cc->num_active_ssb,
        unused_RA_occasion,
        cc->max_association_period,
        prach_info.N_RA_sfn,
        cc->total_prach_occasions_per_config_period);
}

static void schedule_nr_MsgA_pusch(NR_UplinkConfigCommon_t *uplinkConfigCommon,
                                   gNB_MAC_INST *nr_mac,
                                   module_id_t module_idP,
                                   frame_t frameP,
                                   slot_t slotP,
                                   nfapi_nr_prach_pdu_t *prach_pdu,
                                   uint16_t dmrs_TypeA_Position,
                                   NR_PhysCellId_t physCellId)
{
  NR_SCHED_ENSURE_LOCKED(&nr_mac->sched_lock);

  NR_MsgA_PUSCH_Resource_r16_t *msgA_PUSCH_Resource = uplinkConfigCommon->initialUplinkBWP->ext1->msgA_ConfigCommon_r16->choice
                                                          .setup->msgA_PUSCH_Config_r16->msgA_PUSCH_ResourceGroupA_r16;

  const int n_slots_frame = nr_mac->frame_structure.numb_slots_frame;
  slot_t msgA_pusch_slot = (slotP + msgA_PUSCH_Resource->msgA_PUSCH_TimeDomainOffset_r16) % n_slots_frame;
  frame_t msgA_pusch_frame = (frameP + ((slotP + msgA_PUSCH_Resource->msgA_PUSCH_TimeDomainOffset_r16) / n_slots_frame)) % 1024;

  int index = ul_buffer_index((int)msgA_pusch_frame, (int)msgA_pusch_slot, n_slots_frame, nr_mac->UL_tti_req_ahead_size);
  nfapi_nr_ul_tti_request_t *UL_tti_req = &nr_mac[module_idP].UL_tti_req_ahead[0][index];

  UL_tti_req->SFN = msgA_pusch_frame;
  UL_tti_req->Slot = msgA_pusch_slot;
  AssertFatal(is_ul_slot(msgA_pusch_slot, &nr_mac->frame_structure),
              "Slot %d is not an Uplink slot, invalid msgA_PUSCH_TimeDomainOffset_r16 %ld\n",
              msgA_pusch_slot,
              msgA_PUSCH_Resource->msgA_PUSCH_TimeDomainOffset_r16);

  UL_tti_req->pdus_list[UL_tti_req->n_pdus].pdu_type = NFAPI_NR_UL_CONFIG_PUSCH_PDU_TYPE;
  UL_tti_req->pdus_list[UL_tti_req->n_pdus].pdu_size = sizeof(nfapi_nr_pusch_pdu_t);
  nfapi_nr_pusch_pdu_t *pusch_pdu = &UL_tti_req->pdus_list[UL_tti_req->n_pdus].pusch_pdu;
  memset(pusch_pdu, 0, sizeof(nfapi_nr_pusch_pdu_t));

  rnti_t ra_rnti = nr_get_ra_rnti(prach_pdu->prach_start_symbol, slotP, prach_pdu->num_ra, 0);

  // Fill PUSCH PDU
  pusch_pdu->pdu_bit_map = PUSCH_PDU_BITMAP_PUSCH_DATA;
  pusch_pdu->rnti = ra_rnti;
  pusch_pdu->handle = 0;
  pusch_pdu->rb_size = msgA_PUSCH_Resource->nrofPRBs_PerMsgA_PO_r16;
  pusch_pdu->mcs_table = 0;
  pusch_pdu->frequency_hopping =
      msgA_PUSCH_Resource->msgA_IntraSlotFrequencyHopping_r16 ? *msgA_PUSCH_Resource->msgA_IntraSlotFrequencyHopping_r16 : 0;
  pusch_pdu->dmrs_ports = 1; // 6.2.2 in 38.214 only port 0 to be used
  AssertFatal(msgA_PUSCH_Resource->startSymbolAndLengthMsgA_PO_r16,
              "Only SLIV based on startSymbolAndLengthMsgA_PO_r16 implemented\n");
  int S = 0;
  int L = 0;
  SLIV2SL(*msgA_PUSCH_Resource->startSymbolAndLengthMsgA_PO_r16, &S, &L);
  pusch_pdu->start_symbol_index = S;
  pusch_pdu->nr_of_symbols = L;
  pusch_pdu->pusch_data.new_data_indicator = 1;
  pusch_pdu->nrOfLayers = 1;
  pusch_pdu->num_dmrs_cdm_grps_no_data = L <= 2 ? 1 : 2; // no data in dmrs symbols as in 6.2.2 in 38.214
  pusch_pdu->ul_dmrs_symb_pos = get_l_prime(3, 0, pusch_dmrs_pos2, pusch_len1, 10, dmrs_TypeA_Position);
  pusch_pdu->transform_precoding = *uplinkConfigCommon->initialUplinkBWP->ext1->msgA_ConfigCommon_r16->choice.setup->msgA_PUSCH_Config_r16->msgA_TransformPrecoder_r16;
  pusch_pdu->rb_bitmap[0] = 0;
  pusch_pdu->rb_start = msgA_PUSCH_Resource->frequencyStartMsgA_PUSCH_r16; // rb_start depends on the RO
  int locationAndBandwidth = uplinkConfigCommon->initialUplinkBWP->genericParameters.locationAndBandwidth;
  pusch_pdu->bwp_size = NRRIV2BW(locationAndBandwidth, MAX_BWP_SIZE);
  pusch_pdu->bwp_start = NRRIV2PRBOFFSET(locationAndBandwidth, MAX_BWP_SIZE);
  pusch_pdu->subcarrier_spacing = 0;
  pusch_pdu->cyclic_prefix = 0;
  pusch_pdu->uplink_frequency_shift_7p5khz = 0;
  pusch_pdu->vrb_to_prb_mapping = 0;
  pusch_pdu->dmrs_config_type = 0;
  pusch_pdu->data_scrambling_id = 0;
  if (msgA_PUSCH_Resource->msgA_DMRS_Config_r16.msgA_ScramblingID0_r16) {
    pusch_pdu->ul_dmrs_scrambling_id = *msgA_PUSCH_Resource->msgA_DMRS_Config_r16.msgA_ScramblingID0_r16;
  } else {
    pusch_pdu->ul_dmrs_scrambling_id = physCellId;
  }
  pusch_pdu->scid =
      0; // DMRS sequence initialization [TS38.211, sec 6.4.1.1.1]. Should match what is sent in DCI 0_1, otherwise set to 0.
  pusch_pdu->pusch_identity = 0;
  pusch_pdu->resource_alloc = 1; // type 1
  pusch_pdu->tx_direct_current_location = 0;
  pusch_pdu->mcs_index = msgA_PUSCH_Resource->msgA_MCS_r16;
  pusch_pdu->qam_mod_order = nr_get_Qm_dl(pusch_pdu->mcs_index, pusch_pdu->mcs_table);

  int num_dmrs_symb = 0;
  for (int i = 10; i < 10 + 3; i++)
    num_dmrs_symb += (pusch_pdu->ul_dmrs_symb_pos >> i) & 1;
  AssertFatal(pusch_pdu->mcs_index <= 28, "Exceeding MCS limit for MsgA PUSCH\n");
  int R = nr_get_code_rate_ul(pusch_pdu->mcs_index, pusch_pdu->mcs_table);
  pusch_pdu->target_code_rate = R;
  int TBS = nr_compute_tbs(pusch_pdu->qam_mod_order,
                       R,
                       pusch_pdu->rb_size,
                       pusch_pdu->nr_of_symbols,
                       num_dmrs_symb * 12, // nb dmrs set for no data in dmrs symbol
                       0, // nb_rb_oh
                       0, // to verify tb scaling
                       pusch_pdu->nrOfLayers)
        >> 3;

  pusch_pdu->pusch_data.tb_size = TBS;
  pusch_pdu->maintenance_parms_v3.ldpcBaseGraph = get_BG(TBS << 3, R);

  LOG_D(NR_MAC, "Scheduling MsgA PUSCH in %d.%d\n", msgA_pusch_frame, msgA_pusch_slot);

  UL_tti_req->n_pdus += 1;
}

static void fill_vrb(const frame_t frame,
                     const slot_t slot,
                     int nb_rb,
                     int beam_idx,
                     int vrb_size,
                     int slots_frame,
                     int rb_start,
                     int start_symb,
                     int num_symb,
                     NR_COMMON_channels_t *cc)
{
  const int index = ul_buffer_index(frame, slot, slots_frame, vrb_size);
  uint16_t *vrb_map_UL = &cc->vrb_map_UL[beam_idx][index * MAX_BWP_SIZE];
  for (int i = 0; i < nb_rb; ++i) {
    AssertFatal(
        !(vrb_map_UL[rb_start + i] & SL_to_bitmap(start_symb, num_symb)),
        "PRACH resources are already occupied!\n");
    vrb_map_UL[rb_start + i] |= SL_to_bitmap(start_symb, num_symb);
  }
}

void schedule_nr_prach(module_id_t module_idP, frame_t frameP, slot_t slotP)
{
  gNB_MAC_INST *gNB = RC.nrmac[module_idP];
  /* already mutex protected: held in gNB_dlsch_ulsch_scheduler() */
  NR_SCHED_ENSURE_LOCKED(&gNB->sched_lock);

  NR_COMMON_channels_t *cc = gNB->common_channels;
  NR_ServingCellConfigCommon_t *scc = cc->ServingCellConfigCommon;
  NR_BWP_UplinkCommon_t *initialUplinkBWP = scc->uplinkConfigCommon->initialUplinkBWP;
  NR_RACH_ConfigCommon_t *rach_ConfigCommon = initialUplinkBWP->rach_ConfigCommon->choice.setup;
  NR_MsgA_ConfigCommon_r16_t *msgacc = NULL;
  if (initialUplinkBWP->ext1 && initialUplinkBWP->ext1->msgA_ConfigCommon_r16)
    msgacc = initialUplinkBWP->ext1->msgA_ConfigCommon_r16->choice.setup;
  int slots_frame = gNB->frame_structure.numb_slots_frame;
  int index = ul_buffer_index(frameP, slotP, slots_frame, gNB->UL_tti_req_ahead_size);
  nfapi_nr_ul_tti_request_t *UL_tti_req = &RC.nrmac[module_idP]->UL_tti_req_ahead[0][index];
  nfapi_nr_config_request_scf_t *cfg = &RC.nrmac[module_idP]->config[0];

  if (is_ul_slot(slotP, &RC.nrmac[module_idP]->frame_structure)) {
    const NR_RACH_ConfigGeneric_t *rach_ConfigGeneric = &rach_ConfigCommon->rach_ConfigGeneric;
    uint8_t config_index = rach_ConfigGeneric->prach_ConfigurationIndex;
    int slot_index = 0;
    uint16_t prach_occasion_id = -1;

    int bwp_start = NRRIV2PRBOFFSET(initialUplinkBWP->genericParameters.locationAndBandwidth, MAX_BWP_SIZE);

    uint8_t fdm = cfg->prach_config.num_prach_fd_occasions.value;
    const int ul_mu = scc->uplinkConfigCommon->frequencyInfoUL->scs_SpecificCarrierList.list.array[0]->subcarrierSpacing;
    const int mu = nr_get_prach_or_ul_mu(msgacc, rach_ConfigCommon, ul_mu);
    // prach is scheduled according to configuration index and tables 6.3.3.2.2 to 6.3.3.2.4
    uint16_t RA_sfn_index = -1;
    frequency_range_t freq_range = get_freq_range_from_arfcn(scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencyPointA);
    if (get_nr_prach_sched_from_info(cc->prach_info, config_index, frameP, slotP, mu, freq_range, &RA_sfn_index, cc->frame_type)) {
      uint16_t format0 = cc->prach_info.format & 0xff; // first column of format from table
      uint16_t format1 = (cc->prach_info.format >> 8) & 0xff; // second column of format from table

      if (cc->prach_info.N_RA_slot > 1) { // more than 1 PRACH slot in a subframe
        if (slotP % 2 == 1)
          slot_index = 1;
        else
          slot_index = 0;
      } else if (cc->prach_info.N_RA_slot <= 1) { // 1 PRACH slot in a subframe
        slot_index = 0;
      }

      UL_tti_req->SFN = frameP;
      UL_tti_req->Slot = slotP;
      UL_tti_req->rach_present = 1;
      NR_beam_alloc_t beam = {0};
      uint32_t N_t_slot = cc->prach_info.N_t_slot;
      uint32_t start_symb = cc->prach_info.start_symbol;
      for (int fdm_index = 0; fdm_index < fdm; fdm_index++) { // one structure per frequency domain occasion
        AssertFatal(UL_tti_req->n_pdus < sizeof(UL_tti_req->pdus_list) / sizeof(UL_tti_req->pdus_list[0]),
                    "Invalid UL_tti_req->n_pdus %d\n",
                     UL_tti_req->n_pdus);

        UL_tti_req->pdus_list[UL_tti_req->n_pdus].pdu_type = NFAPI_NR_UL_CONFIG_PRACH_PDU_TYPE;
        UL_tti_req->pdus_list[UL_tti_req->n_pdus].pdu_size = sizeof(nfapi_nr_prach_pdu_t);
        nfapi_nr_prach_pdu_t  *prach_pdu = &UL_tti_req->pdus_list[UL_tti_req->n_pdus].prach_pdu;
        memset(prach_pdu, 0, sizeof(nfapi_nr_prach_pdu_t));
        UL_tti_req->n_pdus += 1;
        int num_td_occ = 0;
        for (int td_index = 0; td_index < N_t_slot; td_index++) {
          uint32_t config_period = cc->prach_info.x;
          prach_occasion_id = (((frameP % (cc->max_association_period * config_period))/config_period) * cc->total_prach_occasions_per_config_period) +
                              (RA_sfn_index + slot_index) * N_t_slot * fdm + td_index * fdm + fdm_index;

          if (prach_occasion_id >= cc->total_prach_occasions) // to be confirmed: unused occasion?
            continue;

          num_td_occ++;
          float num_ssb_per_RO = ssb_per_rach_occasion[cfg->prach_config.ssb_per_rach.value];
          int beam_index = 0;
          if(num_ssb_per_RO <= 1) {
            // ordered ssb number
            int n_ssb = (int) (prach_occasion_id / (int)(1 / num_ssb_per_RO)) % cc->num_active_ssb;
            // fapi beam index
            beam_index = get_fapi_beamforming_index(gNB, cc->ssb_index[n_ssb]);
            // multi-beam allocation structure
            beam = beam_allocation_procedure(&gNB->beam_info, frameP, slotP, beam_index, slots_frame);
            AssertFatal(beam.idx >= 0, "Cannot allocate PRACH corresponding to %d SSB transmitted in any available beam\n", n_ssb + 1);
          } else {
            int first_ssb_index = (prach_occasion_id * (int)num_ssb_per_RO) % cc->num_active_ssb;
            for(int j = first_ssb_index; j < first_ssb_index + num_ssb_per_RO; j++) {
              // fapi beam index
              beam_index = get_fapi_beamforming_index(gNB, cc->ssb_index[j]);
              // multi-beam allocation structure
              beam = beam_allocation_procedure(&gNB->beam_info, frameP, slotP, beam_index, slots_frame);
              AssertFatal(beam.idx >= 0, "Cannot allocate PRACH corresponding to SSB %d in any available beam\n", j);
            }
          }
          if(num_td_occ == 1) {
            // filling the prach fapi structure
            prach_pdu->phys_cell_id = *scc->physCellId;
            prach_pdu->prach_start_symbol = start_symb;
            prach_pdu->num_ra = fdm_index;
            prach_pdu->num_cs = get_NCS(rach_ConfigGeneric->zeroCorrelationZoneConfig,
                                        format0,
                                        rach_ConfigCommon->restrictedSetConfig);

            // SCF PRACH PDU format field does not consider A1/B1 etc. possibilities
            // We added 9 = A1/B1 10 = A2/B2 11 A3/B3
            if (format1!=0xff) {
              switch(format0) {
                case 0xa1:
                  prach_pdu->prach_format = 11;
                  break;
                case 0xa2:
                  prach_pdu->prach_format = 12;
                  break;
                case 0xa3:
                  prach_pdu->prach_format = 13;
                  break;
              default:
                AssertFatal(1==0,"Only formats A1/B1 A2/B2 A3/B3 are valid for dual format");
              }
            }
            else{
              switch(format0) {
                case 0:
                  prach_pdu->prach_format = 0;
                  break;
                case 1:
                  prach_pdu->prach_format = 1;
                  break;
                case 2:
                  prach_pdu->prach_format = 2;
                  break;
                case 3:
                  prach_pdu->prach_format = 3;
                  break;
                case 0xa1:
                  prach_pdu->prach_format = 4;
                  break;
                case 0xa2:
                  prach_pdu->prach_format = 5;
                  break;
                case 0xa3:
                  prach_pdu->prach_format = 6;
                  break;
                case 0xb1:
                  prach_pdu->prach_format = 7;
                  break;
                case 0xb4:
                  prach_pdu->prach_format = 8;
                  break;
                case 0xc0:
                  prach_pdu->prach_format = 9;
                  break;
                case 0xc2:
                  prach_pdu->prach_format = 10;
                  break;
              default:
                AssertFatal(1==0,"Invalid PRACH format");
              }
            }
            if (initialUplinkBWP->ext1 && initialUplinkBWP->ext1->msgA_ConfigCommon_r16) {
              if (gNB->UE_info.connected_ue_list[0] == NULL)
                schedule_nr_MsgA_pusch(scc->uplinkConfigCommon,
                                       gNB,
                                       module_idP,
                                       frameP,
                                       slotP,
                                       prach_pdu,
                                       scc->dmrs_TypeA_Position,
                                       *scc->physCellId);
            }
          }
          prach_pdu->num_prach_ocas = num_td_occ;
          prach_pdu->beamforming.num_prgs = 0;
          prach_pdu->beamforming.prg_size = 0;
          prach_pdu->beamforming.dig_bf_interface = num_td_occ;
          prach_pdu->beamforming.prgs_list[0].dig_bf_interface_list[num_td_occ - 1].beam_idx = beam_index;

          LOG_D(NR_MAC,
                "Frame %d, Slot %d: Prach Occasion id = %u  fdm index = %u start symbol = %u slot index = %u subframe index = %u \n",
                frameP,
                slotP,
                prach_occasion_id,
                prach_pdu->num_ra,
                prach_pdu->prach_start_symbol,
                slot_index,
                RA_sfn_index);
        }
        if (num_td_occ == 0) // no TD PRACH occasion -> no PRACH PDU
          UL_tti_req->n_pdus -= 1;
        else
          gNB->num_scheduled_prach_rx += num_td_occ;
      }

      // block resources in vrb_map_UL
      const int mu_pusch = scc->uplinkConfigCommon->frequencyInfoUL->scs_SpecificCarrierList.list.array[0]->subcarrierSpacing;
      const int16_t n_ra_rb = get_N_RA_RB(cfg->prach_config.prach_sub_c_spacing.value, mu_pusch);
      // mark PRBs as occupied for current and future slots if prach extends beyond current slot
      int total_prach_slots;
      uint32_t N_dur = cc->prach_info.N_dur;
      if (format0 < 4) {
        N_dur = 14; // number of PRACH symbols in PRACH slot
        total_prach_slots = get_long_prach_dur(format0, mu_pusch);
        AssertFatal(slotP + total_prach_slots - 1 < slots_frame, "PRACH cannot extend across frames\n");
      } else {
        total_prach_slots = 1;
      }
      // reserve PRBs occupied by PRACH in all PRACH slot.
      for (int i = 0; i < total_prach_slots; i++) {
        fill_vrb(frameP,
                 slotP + i,
                 n_ra_rb * fdm,
                 beam.idx,
                 gNB->vrb_map_UL_size,
                 slots_frame,
                 bwp_start + rach_ConfigGeneric->msg1_FrequencyStart,
                 start_symb,
                 N_t_slot * N_dur,
                 cc);
      }
    }
  }
}

int nr_fill_successrar(const NR_UE_sched_ctrl_t *ue_sched_ctl,
                       rnti_t crnti,
                       const unsigned char *ue_cont_res_id,
                       uint8_t resource_indicator,
                       uint8_t timing_indicator,
                       unsigned char *mac_pdu,
                       int mac_pdu_length)
{
  LOG_D(NR_MAC, "mac_pdu_length = %d\n", mac_pdu_length);
  int timing_advance_cmd = ue_sched_ctl->ta_update;
  // TS 38.321 - Figure 6.1.5a-1: BI MAC subheader
  NR_RA_HEADER_BI_MSGB *bi = (NR_RA_HEADER_BI_MSGB *)&mac_pdu[mac_pdu_length];
  mac_pdu_length += sizeof(NR_RA_HEADER_BI_MSGB);

  bi->E = 1;
  bi->T1 = 0;
  bi->T2 = 0;
  bi->R = 0;
  bi->BI = 0; // BI = 0, 5ms

  // TS 38.321 - Figure 6.1.5a-3: SuccessRAR MAC subheader
  NR_RA_HEADER_SUCCESS_RAR_MSGB *SUCESS_RAR_header = (NR_RA_HEADER_SUCCESS_RAR_MSGB *)&mac_pdu[mac_pdu_length];
  mac_pdu_length += sizeof(NR_RA_HEADER_SUCCESS_RAR_MSGB);

  SUCESS_RAR_header->E = 0;
  SUCESS_RAR_header->T1 = 0;
  SUCESS_RAR_header->T2 = 1;
  SUCESS_RAR_header->S = 0;
  SUCESS_RAR_header->R = 0;

  // TS 38.321 - Figure 6.2.3a-2: successRAR
  NR_MAC_SUCCESS_RAR *successRAR = (NR_MAC_SUCCESS_RAR *)&mac_pdu[mac_pdu_length];
  mac_pdu_length += sizeof(NR_MAC_SUCCESS_RAR);

  successRAR->CONT_RES_1 = ue_cont_res_id[0];
  successRAR->CONT_RES_2 = ue_cont_res_id[1];
  successRAR->CONT_RES_3 = ue_cont_res_id[2];
  successRAR->CONT_RES_4 = ue_cont_res_id[3];
  successRAR->CONT_RES_5 = ue_cont_res_id[4];
  successRAR->CONT_RES_6 = ue_cont_res_id[5];
  successRAR->R = 0;
  successRAR->CH_ACESS_CPEXT = 1;
  successRAR->TPC = ue_sched_ctl->tpc0;
  successRAR->HARQ_FTI = timing_indicator;
  successRAR->PUCCH_RI = resource_indicator;
  successRAR->TA1 = (uint8_t)(timing_advance_cmd >> 8); // 4 MSBs of timing advance;
  successRAR->TA2 = (uint8_t)(timing_advance_cmd & 0xff); // 8 LSBs of timing advance;
  successRAR->CRNTI_1 = (uint8_t)(crnti >> 8); // 8 MSBs of rnti
  successRAR->CRNTI_2 = (uint8_t)(crnti & 0xff); // 8 LSBs of rnti

  LOG_D(NR_MAC,
        "successRAR: Contention Resolution ID 0x%02x%02x%02x%02x%02x%02x R 0x%01x CH_ACESS_CPEXT 0x%02x TPC 0x%02x HARQ_FTI 0x%03x "
        "PUCCH_RI 0x%04x TA 0x%012x CRNTI 0x%04x\n",
        successRAR->CONT_RES_1,
        successRAR->CONT_RES_2,
        successRAR->CONT_RES_3,
        successRAR->CONT_RES_4,
        successRAR->CONT_RES_5,
        successRAR->CONT_RES_6,
        successRAR->R,
        successRAR->CH_ACESS_CPEXT,
        successRAR->TPC,
        successRAR->HARQ_FTI,
        successRAR->PUCCH_RI,
        timing_advance_cmd,
        crnti);
  LOG_D(NR_MAC, "mac_pdu_length = %d\n", mac_pdu_length);
  return mac_pdu_length;
}

/** @brief find UE with RA process for given preamble */
static NR_UE_info_t *get_existing_ra(gNB_MAC_INST *nr_mac, uint16_t preamble_index)
{
  UE_iterator(nr_mac->UE_info.access_ue_list, UE) {
    NR_RA_t *ra = UE->ra;
    for (int i = 0; i < ra->preambles.num_preambles; ++i) {
      if (ra->preambles.preamble_list[i] == preamble_index)
        return UE;
    }
  }
  return NULL;
}

/** @brief add UE to list of UEs doing RA.
 *
 * Remove with nr_release_ra_UE(). */
bool add_new_UE_RA(gNB_MAC_INST *nr_mac, NR_UE_info_t *UE)
{
  DevAssert(UE->ra); // a UE in the acess_ue_list needs to have an RA process
  return add_UE_to_list(NR_NB_RA_PROC_MAX, nr_mac->UE_info.access_ue_list, UE);
}

static uint8_t nr_get_msg3_tpc(uint32_t preamble_power)
{
  // TODO not sure how to implement TPC for MSG3 to be sent in RAR
  //      maybe using preambleReceivedTargetPower as a term of comparison
  //      in any case OAI L1 sets this as invalid
  //      and Aerial report doesn't seem to be reliable (not matching preambleReceivedTargetPower)
  //      so for now we feedback 0dB TPC
  return 3; // it means 0dB
}

static unsigned int get_slot_RA(const NR_ServingCellConfigCommon_t *scc,
                                const NR_RACH_ConfigCommon_t *rach_ConfigCommon,
                                const NR_MsgA_ConfigCommon_r16_t *msgacc,
                                frame_type_t frame_type,
                                int slot)
{
  uint8_t index = rach_ConfigCommon->rach_ConfigGeneric.prach_ConfigurationIndex;
  uint16_t prach_format =
    get_nr_prach_format_from_index(index, scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencyPointA, frame_type);
  unsigned int slot_RA;
  if ((prach_format & 0xff) < 4) {
    const int ul_mu = scc->uplinkConfigCommon->frequencyInfoUL->scs_SpecificCarrierList.list.array[0]->subcarrierSpacing;
    const int mu = nr_get_prach_or_ul_mu(msgacc, rach_ConfigCommon, ul_mu);
    unsigned int slots_per_sf = (1 << mu);
    slot_RA = slot / slots_per_sf;
  } else {
    slot_RA = slot;
  }
  return slot_RA;
}

void nr_initiate_ra_proc(module_id_t module_idP,
                         int CC_id,
                         frame_t frame,
                         int slot,
                         uint16_t preamble_index,
                         uint8_t freq_index,
                         uint8_t symbol,
                         int16_t timing_offset,
                         uint32_t preamble_power)
{
  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_INITIATE_RA_PROC, 1);

  gNB_MAC_INST *nr_mac = RC.nrmac[module_idP];
  NR_SCHED_LOCK(&nr_mac->sched_lock);

  /* check if preamble exists (NSA, HO cases) */
  NR_UE_info_t *UE = get_existing_ra(nr_mac, preamble_index);
  if ((get_softmodem_params()->nsa || get_softmodem_params()->do_ra) && !UE) {
    /* we are in NSA, but no UE has been configured => we ignore it */
    LOG_W(NR_MAC, "random access with preamble %d: no pre-configured RA process found\n", preamble_index);
    NR_SCHED_UNLOCK(&nr_mac->sched_lock);
    return;
  }

  if (!UE) {
    /* in CBRA: we don't know this UE yet. There might be CFRA (e.g., HO IN SA)
     * where we know the UE */
    rnti_t rnti;
    bool rnti_found = nr_mac_get_new_rnti(&nr_mac->UE_info, &rnti);
    if (!rnti_found) {
      LOG_E(NR_MAC, "initialisation random access: no more available RNTIs for new UE\n");
      NR_SCHED_UNLOCK(&nr_mac->sched_lock);
      return;
    }

    UE = get_new_nr_ue_inst(&nr_mac->UE_info.uid_allocator, rnti, NULL);
    if (!UE) {
      LOG_E(NR_MAC, "FAILURE: %4d.%2d cannot create UE context, ignoring RA because RRC Reject not implemented yet\n", frame, slot);
      NR_SCHED_UNLOCK(&nr_mac->sched_lock);
      return;
    }
    if (!add_new_UE_RA(nr_mac, UE)) {
      LOG_E(NR_MAC, "FAILURE: %4d.%2d initiating RA procedure for preamble index %d: no free RA process\n", frame, slot, preamble_index);
      delete_nr_ue_data(UE, NULL, &nr_mac->UE_info.uid_allocator);
      NR_SCHED_UNLOCK(&nr_mac->sched_lock);
      return;
    }
  }

  NR_RA_t *ra = UE->ra;
  ra->preamble_frame = frame;
  ra->preamble_slot = slot;
  ra->preamble_index = preamble_index;
  ra->timing_offset = timing_offset;
  ra->msg3_TPC = nr_get_msg3_tpc(preamble_power);

  NR_COMMON_channels_t *cc = &nr_mac->common_channels[CC_id];
  NR_ServingCellConfigCommon_t *scc = cc->ServingCellConfigCommon;
  // 3GPP TS 38.321 Section 5.1.3(a) says t_id for RA-RNTI depends on mu as specified in clause 5.3.2 in TS 38.211
  // so mu = 0 for prach format < 4.
  NR_RACH_ConfigCommon_t *rach_ConfigCommon = scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup;
  NR_MsgA_ConfigCommon_r16_t *msgacc = NULL;
  if (scc->uplinkConfigCommon->initialUplinkBWP->ext1 && scc->uplinkConfigCommon->initialUplinkBWP->ext1->msgA_ConfigCommon_r16)
    msgacc = scc->uplinkConfigCommon->initialUplinkBWP->ext1->msgA_ConfigCommon_r16->choice.setup;
  uint8_t ul_carrier_id = 0; // 0 for NUL 1 for SUL
  uint32_t slot_RA = get_slot_RA(scc, rach_ConfigCommon, msgacc, cc->frame_type, slot);
  ra->RA_rnti = nr_get_ra_rnti(symbol, slot_RA, freq_index, ul_carrier_id);
  if (msgacc) {
    ra->ra_type = RA_2_STEP;
    ra->ra_state = nrRA_WAIT_MsgA_PUSCH;
    ra->MsgB_rnti = nr_get_MsgB_rnti(symbol, slot_RA, freq_index, ul_carrier_id);
  } else {
    ra->ra_type = RA_4_STEP;
    ra->ra_state = nrRA_Msg2;
  }

  LOG_A(NR_MAC, "%d.%d UE RA-RNTI %04x TC-RNTI %04x: initiating RA procedure\n", frame, slot, ra->RA_rnti, UE->rnti);

  // Configure RA BWP
  configure_UE_BWP(nr_mac, scc, UE, true, NR_SearchSpace__searchSpaceType_PR_common, -1, -1);
  // return current SSB order in the list of tranmitted SSBs
  int n_ssb = ssb_index_from_prach(module_idP, frame, slot, preamble_index, freq_index, symbol);
  UE->UE_beam_index = get_fapi_beamforming_index(nr_mac, cc->ssb_index[n_ssb]);

  NR_SCHED_UNLOCK(&nr_mac->sched_lock);
  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_INITIATE_RA_PROC, 0);
}

static void start_ra_contention_resolution_timer(NR_RA_t *ra, const long ra_ContentionResolutionTimer, const int K2, const int scs)
{
  // 3GPP TS 38.331 Section 6.3.2 Radio resource control information elements
  // ra-ContentionResolutionTimer ENUMERATED {sf8, sf16, sf24, sf32, sf40, sf48, sf56, sf64}
  // The initial value for the contention resolution timer.
  // Value sf8 corresponds to 8 subframes, value sf16 corresponds to 16 subframes, and so on.
  // We add 2 * K2 because the timer runs from Msg2 transmission till Msg4 ACK reception
  ra->contention_resolution_timer = ((((int)ra_ContentionResolutionTimer + 1) * 8) << scs) + 2 * K2;
  LOG_D(NR_MAC,
        "Starting RA Contention Resolution timer with %d ms + 2 * %d K2 (%d slots) duration\n",
        ((int)ra_ContentionResolutionTimer + 1) * 8,
        K2,
        ra->contention_resolution_timer);
}

static void nr_generate_Msg3_retransmission(module_id_t module_idP,
                                            int CC_id,
                                            frame_t frame,
                                            slot_t slot,
                                            NR_UE_info_t *UE,
                                            nfapi_nr_ul_dci_request_t *ul_dci_req)
{
  gNB_MAC_INST *nr_mac = RC.nrmac[module_idP];
  NR_RA_t *ra = UE->ra;
  NR_COMMON_channels_t *cc = &nr_mac->common_channels[CC_id];
  NR_ServingCellConfigCommon_t *scc = cc->ServingCellConfigCommon;
  NR_UE_UL_BWP_t *ul_bwp = &UE->current_UL_BWP;
  NR_UE_ServingCell_Info_t *sc_info = &UE->sc_info;

  NR_PUSCH_TimeDomainResourceAllocationList_t *pusch_TimeDomainAllocationList = ul_bwp->tdaList_Common;
  int mu = ul_bwp->scs;
  int slots_frame = nr_mac->frame_structure.numb_slots_frame;
  uint16_t K2 = *pusch_TimeDomainAllocationList->list.array[ra->Msg3_tda_id]->k2 + get_NTN_Koffset(scc);
  const int sched_frame = (frame + (slot + K2) / slots_frame) % MAX_FRAME_NUMBER;
  const int sched_slot = (slot + K2) % slots_frame;

  if (is_ul_slot(sched_slot, &nr_mac->frame_structure)) {
    NR_beam_alloc_t beam_ul = beam_allocation_procedure(&nr_mac->beam_info, sched_frame, sched_slot, UE->UE_beam_index, slots_frame);
    if (beam_ul.idx < 0)
      return;
    NR_beam_alloc_t beam_dci = beam_allocation_procedure(&nr_mac->beam_info, frame, slot, UE->UE_beam_index, slots_frame);
    if (beam_dci.idx < 0) {
      reset_beam_status(&nr_mac->beam_info, sched_frame, sched_slot, UE->UE_beam_index, slots_frame, beam_ul.new_beam);
      return;
    }
    int fh = 0;
    int startSymbolAndLength = pusch_TimeDomainAllocationList->list.array[ra->Msg3_tda_id]->startSymbolAndLength;
    int StartSymbolIndex, NrOfSymbols;
    SLIV2SL(startSymbolAndLength, &StartSymbolIndex, &NrOfSymbols);
    int mappingtype = pusch_TimeDomainAllocationList->list.array[ra->Msg3_tda_id]->mappingType;

    int buffer_index = ul_buffer_index(sched_frame, sched_slot, slots_frame, nr_mac->vrb_map_UL_size);
    uint16_t *vrb_map_UL = &nr_mac->common_channels[CC_id].vrb_map_UL[beam_ul.idx][buffer_index * MAX_BWP_SIZE];

    const int BWPSize = sc_info->initial_ul_BWPSize;
    const int BWPStart = sc_info->initial_ul_BWPStart;

    int rbStart = 0;
    for (int i = 0; (i < ra->msg3_nb_rb) && (rbStart <= (BWPSize - ra->msg3_nb_rb)); i++) {
      if (vrb_map_UL[rbStart + BWPStart + i]&SL_to_bitmap(StartSymbolIndex, NrOfSymbols)) {
        rbStart += i;
        i = 0;
      }
    }
    if (rbStart > (BWPSize - ra->msg3_nb_rb)) {
      // cannot find free vrb_map for msg3 retransmission in this slot
      return;
    }

    LOG_I(NR_MAC,
          "%4d%2d: RA RNTI %04x CC_id %d Scheduling retransmission of Msg3 in (%d,%d)\n",
          frame,
          slot,
          UE->rnti,
          CC_id,
          sched_frame,
          sched_slot);

    buffer_index = ul_buffer_index(sched_frame, sched_slot, slots_frame, nr_mac->UL_tti_req_ahead_size);
    nfapi_nr_ul_tti_request_t *future_ul_tti_req = &nr_mac->UL_tti_req_ahead[CC_id][buffer_index];
    AssertFatal(future_ul_tti_req->SFN == sched_frame
                && future_ul_tti_req->Slot == sched_slot,
                "future UL_tti_req's frame.slot %d.%d does not match PUSCH %d.%d\n",
                future_ul_tti_req->SFN,
                future_ul_tti_req->Slot,
                sched_frame,
                sched_slot);
    AssertFatal(future_ul_tti_req->n_pdus <
                sizeof(future_ul_tti_req->pdus_list) / sizeof(future_ul_tti_req->pdus_list[0]),
                "Invalid future_ul_tti_req->n_pdus %d\n", future_ul_tti_req->n_pdus);
    future_ul_tti_req->pdus_list[future_ul_tti_req->n_pdus].pdu_type = NFAPI_NR_UL_CONFIG_PUSCH_PDU_TYPE;
    future_ul_tti_req->pdus_list[future_ul_tti_req->n_pdus].pdu_size = sizeof(nfapi_nr_pusch_pdu_t);
    nfapi_nr_pusch_pdu_t *pusch_pdu = &future_ul_tti_req->pdus_list[future_ul_tti_req->n_pdus].pusch_pdu;
    memset(pusch_pdu, 0, sizeof(nfapi_nr_pusch_pdu_t));

    fill_msg3_pusch_pdu(pusch_pdu, scc, UE, startSymbolAndLength, mu, BWPSize, BWPStart, mappingtype, fh);
    future_ul_tti_req->n_pdus += 1;

    // generation of DCI 0_0 to schedule msg3 retransmission
    NR_SearchSpace_t *ss = UE->UE_sched_ctrl.search_space;
    NR_ControlResourceSet_t *coreset = UE->UE_sched_ctrl.coreset;
    AssertFatal(coreset, "Coreset cannot be null for RA-Msg3 retransmission\n");

    const int coresetid = coreset->controlResourceSetId;
    nfapi_nr_dl_tti_pdcch_pdu_rel15_t *pdcch_pdu_rel15 = nr_mac->pdcch_pdu_idx[CC_id][coresetid];
    if (!pdcch_pdu_rel15) {
      nfapi_nr_ul_dci_request_pdus_t *ul_dci_request_pdu = &ul_dci_req->ul_dci_pdu_list[ul_dci_req->numPdus];
      memset(ul_dci_request_pdu, 0, sizeof(nfapi_nr_ul_dci_request_pdus_t));
      ul_dci_request_pdu->PDUType = NFAPI_NR_DL_TTI_PDCCH_PDU_TYPE;
      ul_dci_request_pdu->PDUSize = (uint8_t)(2+sizeof(nfapi_nr_dl_tti_pdcch_pdu));
      pdcch_pdu_rel15 = &ul_dci_request_pdu->pdcch_pdu.pdcch_pdu_rel15;
      ul_dci_req->numPdus += 1;
      nr_configure_pdcch(pdcch_pdu_rel15, coreset, &UE->UE_sched_ctrl.sched_pdcch);
      nr_mac->pdcch_pdu_idx[CC_id][coresetid] = pdcch_pdu_rel15;
    }

    uint8_t aggregation_level;
    int CCEIndex = get_cce_index(nr_mac,
                                 CC_id, slot, 0,
                                 &aggregation_level,
                                 beam_dci.idx,
                                 ss,
                                 coreset,
                                 &UE->UE_sched_ctrl.sched_pdcch,
                                 true,
                                 0);
    if (CCEIndex < 0) {
      LOG_E(NR_MAC, "UE %04x cannot find free CCE!\n", UE->rnti);
      return;
    }

    // Fill PDCCH DL DCI PDU
    nfapi_nr_dl_dci_pdu_t *dci_pdu = &pdcch_pdu_rel15->dci_pdu[pdcch_pdu_rel15->numDlDci];
    pdcch_pdu_rel15->numDlDci++;
    dci_pdu->RNTI = UE->rnti;
    dci_pdu->ScramblingId = *scc->physCellId;
    dci_pdu->ScramblingRNTI = 0;
    dci_pdu->AggregationLevel = aggregation_level;
    dci_pdu->CceIndex = CCEIndex;
    dci_pdu->beta_PDCCH_1_0 = 0;
    dci_pdu->powerControlOffsetSS = 1;

    dci_pdu->precodingAndBeamforming.num_prgs = 0;
    dci_pdu->precodingAndBeamforming.prg_size = 0;
    dci_pdu->precodingAndBeamforming.dig_bf_interfaces = 1;
    dci_pdu->precodingAndBeamforming.prgs_list[0].pm_idx = 0;
    dci_pdu->precodingAndBeamforming.prgs_list[0].dig_bf_interface_list[0].beam_idx = UE->UE_beam_index;

    dci_pdu_rel15_t uldci_payload={0};

    config_uldci(sc_info,
                 pusch_pdu,
                 &uldci_payload,
                 NULL,
                 NULL,
                 ra->Msg3_tda_id,
                 ra->msg3_TPC,
                 1, // Not toggling NDI in msg3 retransmissions
                 ul_bwp,
                 ss->searchSpaceType->present);

    // Reset TPC to 0 dB to not request new gain multiple times before computing new value for SNR
    ra->msg3_TPC = 1;

    fill_dci_pdu_rel15(sc_info,
                       &UE->current_DL_BWP,
                       ul_bwp,
                       dci_pdu,
                       &uldci_payload,
                       NR_UL_DCI_FORMAT_0_0,
                       TYPE_TC_RNTI_,
                       ul_bwp->bwp_id,
                       ss,
                       coreset,
                       0, // parameter not needed for DCI 0_0
                       nr_mac->cset0_bwp_size);

    // Mark the corresponding RBs as used

    fill_pdcch_vrb_map(nr_mac,
                       CC_id,
                       &UE->UE_sched_ctrl.sched_pdcch,
                       CCEIndex,
                       aggregation_level,
                       beam_dci.idx);

    for (int rb = 0; rb < ra->msg3_nb_rb; rb++) {
      vrb_map_UL[rbStart + BWPStart + rb] |= SL_to_bitmap(StartSymbolIndex, NrOfSymbols);
    }

    // Restart RA contention resolution timer in Msg3 retransmission slot (current slot + K2)
    // 3GPP TS 38.321 Section 5.1.5 Contention Resolution
    start_ra_contention_resolution_timer(
        ra,
        scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->ra_ContentionResolutionTimer,
        K2,
        ul_bwp->scs);

    // reset state to wait msg3
    ra->ra_state = nrRA_WAIT_Msg3;
    ra->Msg3_frame = sched_frame;
    ra->Msg3_slot = sched_slot;
  }
}

static bool get_feasible_msg3_tda(const NR_ServingCellConfigCommon_t *scc,
                                  int mu_delta,
                                  const NR_PUSCH_TimeDomainResourceAllocationList_t *tda_list,
                                  int frame,
                                  int slot,
                                  NR_RA_t *ra,
                                  NR_beam_info_t *beam_info,
                                  int ue_beam_idx,
                                  const frame_structure_t *fs)
{
  DevAssert(tda_list != NULL);

  const int NTN_gNB_Koffset = get_NTN_Koffset(scc);

  int slots_per_frame = fs->numb_slots_frame;
  for (int i = 0; i < tda_list->list.count; i++) {
    // check if it is UL
    long k2 = *tda_list->list.array[i]->k2 + NTN_gNB_Koffset;
    int abs_slot = slot + k2 + mu_delta;
    int temp_frame = (frame + (abs_slot / slots_per_frame)) & 1023;
    int temp_slot = abs_slot % slots_per_frame; // msg3 slot according to 8.3 in 38.213
    if (fs->frame_type == TDD && !is_ul_slot(temp_slot, fs))
      continue;

    const tdd_bitmap_t *tdd_slot_bitmap = fs->period_cfg.tdd_slot_bitmap;
    int s = get_slot_idx_in_period(temp_slot, fs);
    // check if enough symbols in case of mixed slot
    bool is_mixed = is_mixed_slot(s, fs);
    // if the mixed slot has not enough symbols, skip
    if (is_mixed && tdd_slot_bitmap[s].num_ul_symbols < 3)
      continue;

    uint16_t slot_mask =
        is_mixed ? SL_to_bitmap(NR_NUMBER_OF_SYMBOLS_PER_SLOT - tdd_slot_bitmap[s].num_ul_symbols,
                                tdd_slot_bitmap[s].num_ul_symbols)
                 : 0x3fff;
    long startSymbolAndLength = tda_list->list.array[i]->startSymbolAndLength;
    int start, nr;
    SLIV2SL(startSymbolAndLength, &start, &nr);
    uint16_t msg3_mask = SL_to_bitmap(start, nr);
    LOG_D(NR_MAC, "Check Msg3 TDA %d for slot %d: k2 %ld, S %d L %d\n", i, temp_slot, k2, start, nr);
    /* if this start and length of this TDA cannot be fulfilled, skip */
    if ((slot_mask & msg3_mask) != msg3_mask)
      continue;

    // check if it is possible to allocate MSG3 in a beam in this slot
    NR_beam_alloc_t beam = beam_allocation_procedure(beam_info, temp_frame, temp_slot, ue_beam_idx, slots_per_frame);
    if (beam.idx < 0)
      continue;
      
    // is in mixed slot with more or equal than 3 symbols, or UL slot
    ra->Msg3_frame = temp_frame;
    ra->Msg3_slot = temp_slot;
    ra->Msg3_tda_id = i;
    ra->Msg3_beam = beam;
    return true;
  }

  return false;
}

static bool nr_get_Msg3alloc(gNB_MAC_INST *mac, int CC_id, int current_slot, frame_t current_frame, NR_UE_info_t *UE)
{
  NR_RA_t *ra = UE->ra;
  DevAssert(ra->Msg3_tda_id >= 0 && ra->Msg3_tda_id < 16);

  uint16_t msg3_nb_rb = max(8, mac->min_grant_prb); // sdu has 6 or 8 bytes

  NR_UE_UL_BWP_t *ul_bwp = &UE->current_UL_BWP;
  NR_UE_ServingCell_Info_t *sc_info = &UE->sc_info;

  const NR_PUSCH_TimeDomainResourceAllocationList_t *pusch_TimeDomainAllocationList = ul_bwp->tdaList_Common;

  int startSymbolAndLength = pusch_TimeDomainAllocationList->list.array[ra->Msg3_tda_id]->startSymbolAndLength;
  SLIV2SL(startSymbolAndLength, &ra->msg3_startsymb, &ra->msg3_nbSymb);

  const int buffer_index = ul_buffer_index(ra->Msg3_frame,
                                           ra->Msg3_slot,
                                           mac->frame_structure.numb_slots_frame,
                                           mac->vrb_map_UL_size);
  uint16_t *vrb_map_UL = &mac->common_channels[CC_id].vrb_map_UL[ra->Msg3_beam.idx][buffer_index * MAX_BWP_SIZE];

  int bwpSize = sc_info->initial_ul_BWPSize;
  int bwpStart = sc_info->initial_ul_BWPStart;
  if (bwpSize != ul_bwp->BWPSize || bwpStart != ul_bwp->BWPStart) {
    int act_bwp_start = ul_bwp->BWPStart;
    int act_bwp_size  = ul_bwp->BWPSize;
    if (!((bwpStart >= act_bwp_start) && ((bwpStart+bwpSize) <= (act_bwp_start+act_bwp_size))))
      bwpStart = act_bwp_start;
  }

  /* search msg3_nb_rb free RBs */
  int rbSize = 0;
  int rbStart = 0;
  while (rbSize < msg3_nb_rb) {
    rbStart += rbSize; /* last iteration rbSize was not enough, skip it */
    rbSize = 0;
    while (rbStart < bwpSize && (vrb_map_UL[rbStart + bwpStart] & SL_to_bitmap(ra->msg3_startsymb, ra->msg3_nbSymb)))
      rbStart++;
    if (rbStart + msg3_nb_rb > bwpSize) {
      LOG_D(NR_MAC, "No space to allocate Msg 3\n");
      return false;
    }
    while (rbStart + rbSize < bwpSize
           && !(vrb_map_UL[rbStart + bwpStart + rbSize] & SL_to_bitmap(ra->msg3_startsymb, ra->msg3_nbSymb)) && rbSize < msg3_nb_rb)
      rbSize++;
  }
  ra->msg3_nb_rb = msg3_nb_rb;
  ra->msg3_first_rb = rbStart;
  ra->msg3_bwp_start = bwpStart;
  LOG_I(NR_MAC,
        "UE %04x: Msg3 scheduled at %d.%d (%d.%d TDA %u) start %d RBs %d\n",
        UE->rnti,
        ra->Msg3_frame,
        ra->Msg3_slot,
        current_frame,
        current_slot,
        ra->Msg3_tda_id,
        ra->msg3_first_rb,
        ra->msg3_nb_rb);
  return true;
}

static void fill_msg3_pusch_pdu(nfapi_nr_pusch_pdu_t *pusch_pdu,
                                NR_ServingCellConfigCommon_t *scc,
                                NR_UE_info_t *UE,
                                int startSymbolAndLength,
                                int scs,
                                int bwp_size,
                                int bwp_start,
                                int mappingtype,
                                int fh)
{
  int start_symbol_index,nr_of_symbols;
  SLIV2SL(startSymbolAndLength, &start_symbol_index, &nr_of_symbols);
  int mcsindex = -1; // init value
  const NR_RA_t *ra = UE->ra;
  pusch_pdu->pdu_bit_map = PUSCH_PDU_BITMAP_PUSCH_DATA;
  pusch_pdu->rnti = UE->rnti;
  pusch_pdu->handle = 0;
  pusch_pdu->bwp_start = bwp_start;
  pusch_pdu->bwp_size = bwp_size;
  pusch_pdu->subcarrier_spacing = scs;
  pusch_pdu->cyclic_prefix = 0;
  pusch_pdu->mcs_table = 0;
  if (scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->msg3_transformPrecoder == NULL)
    pusch_pdu->transform_precoding = 1; // disabled
  else {
    pusch_pdu->transform_precoding = 0; // enabled
    pusch_pdu->dfts_ofdm.low_papr_group_number = *scc->physCellId % 30;
    pusch_pdu->dfts_ofdm.low_papr_sequence_number = 0;
    if (scc->uplinkConfigCommon->initialUplinkBWP->pusch_ConfigCommon->choice.setup->groupHoppingEnabledTransformPrecoding)
      AssertFatal(1==0,"Hopping mode is not supported in transform precoding\n");
  }
  pusch_pdu->data_scrambling_id = *scc->physCellId;
  pusch_pdu->nrOfLayers = 1;
  pusch_pdu->ul_dmrs_symb_pos = get_l_prime(nr_of_symbols,mappingtype,pusch_dmrs_pos2,pusch_len1,start_symbol_index, scc->dmrs_TypeA_Position);
  LOG_D(NR_MAC, "MSG3 start_sym:%d NR Symb:%d mappingtype:%d, ul_dmrs_symb_pos:%x\n", start_symbol_index, nr_of_symbols, mappingtype, pusch_pdu->ul_dmrs_symb_pos);
  pusch_pdu->dmrs_config_type = 0;
  pusch_pdu->ul_dmrs_scrambling_id = *scc->physCellId; //If provided and the PUSCH is not a msg3 PUSCH, otherwise, L2 should set this to physical cell id.
  pusch_pdu->pusch_identity = *scc->physCellId; //If provided and the PUSCH is not a msg3 PUSCH, otherwise, L2 should set this to physical cell id.
  pusch_pdu->scid = 0; //DMRS sequence initialization [TS38.211, sec 6.4.1.1.1]. Should match what is sent in DCI 0_1, otherwise set to 0.
  pusch_pdu->dmrs_ports = 1;  // 6.2.2 in 38.214 only port 0 to be used
  pusch_pdu->num_dmrs_cdm_grps_no_data = nr_of_symbols <= 2 ? 1 : 2;  // no data in dmrs symbols as in 6.2.2 in 38.214
  pusch_pdu->resource_alloc = 1; //type 1
  memset(pusch_pdu->rb_bitmap, 0, sizeof(pusch_pdu->rb_bitmap));
  pusch_pdu->rb_start = ra->msg3_first_rb;
  if (ra->msg3_nb_rb > pusch_pdu->bwp_size)
    AssertFatal(false, "MSG3 allocated number of RBs exceed the BWP size\n");
  else
    pusch_pdu->rb_size = ra->msg3_nb_rb;
  pusch_pdu->vrb_to_prb_mapping = 0;

  pusch_pdu->frequency_hopping = fh;
  //pusch_pdu->tx_direct_current_location;
  //The uplink Tx Direct Current location for the carrier. Only values in the value range of this field between 0 and 3299,
  //which indicate the subcarrier index within the carrier corresponding 1o the numerology of the corresponding uplink BWP and value 3300,
  //which indicates "Outside the carrier" and value 3301, which indicates "Undetermined position within the carrier" are used. [TS38.331, UplinkTxDirectCurrentBWP IE]
  pusch_pdu->uplink_frequency_shift_7p5khz = 0;
  //Resource Allocation in time domain
  pusch_pdu->start_symbol_index = start_symbol_index;
  pusch_pdu->nr_of_symbols = nr_of_symbols;
  //Optional Data only included if indicated in pduBitmap
  pusch_pdu->pusch_data.rv_index = nr_get_rv(ra->msg3_round % 4);
  pusch_pdu->pusch_data.harq_process_id = 0;
  pusch_pdu->pusch_data.new_data_indicator = (ra->msg3_round == 0) ? 1 : 0;;
  pusch_pdu->pusch_data.num_cb = 0;

  // Beamforming
  pusch_pdu->beamforming.num_prgs = 0;
  pusch_pdu->beamforming.prg_size = 0; // bwp_size;
  pusch_pdu->beamforming.dig_bf_interface = 1;
  pusch_pdu->beamforming.prgs_list[0].dig_bf_interface_list[0].beam_idx = UE->UE_beam_index;

  int num_dmrs_symb = 0;
  for(int i = start_symbol_index; i < start_symbol_index+nr_of_symbols; i++)
    num_dmrs_symb += (pusch_pdu->ul_dmrs_symb_pos >> i) & 1;
  int TBS = 0;
  while(TBS<7) {  // TBS for msg3 is 7 bytes (except for RRCResumeRequest1 currently not implemented)
    mcsindex++;
    AssertFatal(mcsindex <= 28, "Exceeding MCS limit for Msg3\n");
    int R = nr_get_code_rate_ul(mcsindex,pusch_pdu->mcs_table);
    pusch_pdu->target_code_rate = R;
    pusch_pdu->qam_mod_order = nr_get_Qm_ul(mcsindex,pusch_pdu->mcs_table);
    TBS = nr_compute_tbs(pusch_pdu->qam_mod_order,
                         R,
                         pusch_pdu->rb_size,
                         pusch_pdu->nr_of_symbols,
                         num_dmrs_symb*12, // nb dmrs set for no data in dmrs symbol
                         0, //nb_rb_oh
                         0, // to verify tb scaling
                         pusch_pdu->nrOfLayers)>>3;

    pusch_pdu->mcs_index = mcsindex;
    pusch_pdu->pusch_data.tb_size = TBS;
    pusch_pdu->maintenance_parms_v3.ldpcBaseGraph = get_BG(TBS<<3,R);
  }
}

static void nr_add_msg3(module_id_t module_idP, int CC_id, frame_t frameP, slot_t slotP, NR_UE_info_t *UE, uint8_t *RAR_pdu)
{
  gNB_MAC_INST *mac = RC.nrmac[module_idP];
  NR_COMMON_channels_t *cc = &mac->common_channels[CC_id];
  NR_ServingCellConfigCommon_t *scc = cc->ServingCellConfigCommon;
  NR_UE_UL_BWP_t *ul_bwp = &UE->current_UL_BWP;
  NR_UE_ServingCell_Info_t *sc_info = &UE->sc_info;
  NR_RA_t *ra = UE->ra;
  if (ra->ra_state == nrRA_gNB_IDLE) {
    LOG_W(NR_MAC,"RA is not active for RA %X. skipping msg3 scheduling\n", UE->rnti);
    return;
  }

  const uint16_t mask = SL_to_bitmap(ra->msg3_startsymb, ra->msg3_nbSymb);
  int slots_frame = mac->frame_structure.numb_slots_frame;
  int buffer_index = ul_buffer_index(ra->Msg3_frame, ra->Msg3_slot, slots_frame, mac->vrb_map_UL_size);
  uint16_t *vrb_map_UL = &RC.nrmac[module_idP]->common_channels[CC_id].vrb_map_UL[ra->Msg3_beam.idx][buffer_index * MAX_BWP_SIZE];
  for (int i = 0; i < ra->msg3_nb_rb; ++i) {
    AssertFatal(!(vrb_map_UL[i + ra->msg3_first_rb + ra->msg3_bwp_start] & mask),
                "RB %d in %4d.%2d is already taken, cannot allocate Msg3!\n",
                i + ra->msg3_first_rb,
                ra->Msg3_frame,
                ra->Msg3_slot);
    vrb_map_UL[i + ra->msg3_first_rb + ra->msg3_bwp_start] |= mask;
  }

  LOG_D(NR_MAC, "UE %04x: %d.%d RA is active, Msg3 in (%d,%d)\n", UE->rnti, frameP, slotP, ra->Msg3_frame, ra->Msg3_slot);
  buffer_index = ul_buffer_index(ra->Msg3_frame, ra->Msg3_slot, slots_frame, mac->UL_tti_req_ahead_size);
  nfapi_nr_ul_tti_request_t *future_ul_tti_req = &RC.nrmac[module_idP]->UL_tti_req_ahead[CC_id][buffer_index];
  AssertFatal(future_ul_tti_req->SFN == ra->Msg3_frame
              && future_ul_tti_req->Slot == ra->Msg3_slot,
              "future UL_tti_req's frame.slot %d.%d does not match PUSCH %d.%d\n",
              future_ul_tti_req->SFN,
              future_ul_tti_req->Slot,
              ra->Msg3_frame,
              ra->Msg3_slot);
  future_ul_tti_req->pdus_list[future_ul_tti_req->n_pdus].pdu_type = NFAPI_NR_UL_CONFIG_PUSCH_PDU_TYPE;
  future_ul_tti_req->pdus_list[future_ul_tti_req->n_pdus].pdu_size = sizeof(nfapi_nr_pusch_pdu_t);
  nfapi_nr_pusch_pdu_t *pusch_pdu = &future_ul_tti_req->pdus_list[future_ul_tti_req->n_pdus].pusch_pdu;
  memset(pusch_pdu, 0, sizeof(nfapi_nr_pusch_pdu_t));

  const int ibwp_size = sc_info->initial_ul_BWPSize;
  const int fh = (ul_bwp->pusch_Config && ul_bwp->pusch_Config->frequencyHopping) ? 1 : 0;
  const int startSymbolAndLength = ul_bwp->tdaList_Common->list.array[ra->Msg3_tda_id]->startSymbolAndLength;
  const int mappingtype = ul_bwp->tdaList_Common->list.array[ra->Msg3_tda_id]->mappingType;

  LOG_D(NR_MAC,
        "UE %04x: %d.%d Adding Msg3 UL Config Request for (%d,%d) : (%d,%d,%d)\n",
        UE->rnti,
        frameP,
        slotP,
        ra->Msg3_frame,
        ra->Msg3_slot,
        ra->msg3_nb_rb,
        ra->msg3_first_rb,
        ra->msg3_round);

  fill_msg3_pusch_pdu(pusch_pdu, scc, UE, startSymbolAndLength, ul_bwp->scs, ibwp_size, ra->msg3_bwp_start, mappingtype, fh);
  future_ul_tti_req->n_pdus += 1;

  // calling function to fill rar message
  nr_fill_rar(module_idP, UE, RAR_pdu, pusch_pdu);
}

static bool check_msg2_monitoring(const NR_SearchSpace_t *ss, int slots_per_frame, int current_frame, int current_slot)
{
  // check if the slot is not among the PDCCH monitored ones (38.213 10.1)
  int monitoring_slot_period, monitoring_offset;
  get_monitoring_period_offset(ss, &monitoring_slot_period, &monitoring_offset);
  if ((current_frame * slots_per_frame + current_slot - monitoring_offset) % monitoring_slot_period != 0)
    return false;
  return true;
}

static int get_response_window(e_NR_RACH_ConfigGeneric__ra_ResponseWindow response_window)
{
  int slots;
  switch (response_window) {
    case NR_RACH_ConfigGeneric__ra_ResponseWindow_sl1:
      slots = 1;
      break;
    case NR_RACH_ConfigGeneric__ra_ResponseWindow_sl2:
      slots = 2;
      break;
    case NR_RACH_ConfigGeneric__ra_ResponseWindow_sl4:
      slots = 4;
      break;
    case NR_RACH_ConfigGeneric__ra_ResponseWindow_sl8:
      slots = 8;
      break;
    case NR_RACH_ConfigGeneric__ra_ResponseWindow_sl10:
      slots = 10;
      break;
    case NR_RACH_ConfigGeneric__ra_ResponseWindow_sl20:
      slots = 20;
      break;
    case NR_RACH_ConfigGeneric__ra_ResponseWindow_sl40:
      slots = 40;
      break;
    case NR_RACH_ConfigGeneric__ra_ResponseWindow_sl80:
      slots = 80;
      break;
    default:
      AssertFatal(false, "Invalid response window value %d\n", response_window);
  }
  return slots;
}

static bool msg2_in_response_window(int rach_frame,
                                    int rach_slot,
                                    int n_slots_frame,
                                    long rrc_ra_ResponseWindow,
                                    int current_frame,
                                    int current_slot)
{
  int window_slots = get_response_window(rrc_ra_ResponseWindow);
  if (window_slots > n_slots_frame)
    LOG_E(NR_MAC, "RA-ResponseWindow need to be configured to a value lower than or equal to 10 ms\n");
  int abs_rach = n_slots_frame * rach_frame + rach_slot;
  int abs_now = n_slots_frame * current_frame + current_slot;
  int diff = (n_slots_frame * 1024 + abs_now - abs_rach) % (n_slots_frame * 1024);

  bool in_window = diff <= window_slots;
  if (!in_window) {
    LOG_W(NR_MAC,
          "exceeded RA window: preamble at %d.%2d now %d.%d (diff %d), ra_ResponseWindow %ld/%d slots\n",
          rach_frame,
          rach_slot,
          current_frame,
          current_slot,
          diff,
          rrc_ra_ResponseWindow,
          window_slots);
  }
  return in_window;
}

static void prepare_dl_pdus(gNB_MAC_INST *nr_mac,
                            NR_UE_info_t *UE,
                            NR_sched_pdsch_t *sched_pdsch,
                            nfapi_nr_dl_tti_request_body_t *dl_req,
                            NR_sched_pucch_t *pucch,
                            nr_rnti_type_t rnti_type,
                            int aggregation_level,
                            int CCEIndex,
                            int tb_size,
                            int ndi,
                            int tpc,
                            int delta_PRI,
                            int current_harq_pid,
                            int time_domain_assignment,
                            int CC_id,
                            int rnti,
                            int round,
                            int tb_scaling,
                            int pduindex,
                            long BWPStart,
                            long BWPSize)
{
  // look up the PDCCH PDU for this CC, BWP, and CORESET. If it does not exist, create it. This is especially
  // important if we have multiple RAs, and the DLSCH has to reuse them, so we need to mark them
  NR_UE_sched_ctrl_t *sched_ctrl = &UE->UE_sched_ctrl;
  NR_ControlResourceSet_t *coreset = sched_ctrl->coreset;
  const int coresetid = coreset->controlResourceSetId;
  nfapi_nr_dl_tti_pdcch_pdu_rel15_t *pdcch_pdu_rel15 = nr_mac->pdcch_pdu_idx[CC_id][coresetid];
  if (!pdcch_pdu_rel15) {
    nfapi_nr_dl_tti_request_pdu_t *dl_tti_pdcch_pdu = &dl_req->dl_tti_pdu_list[dl_req->nPDUs];
    memset(dl_tti_pdcch_pdu, 0, sizeof(nfapi_nr_dl_tti_request_pdu_t));
    dl_tti_pdcch_pdu->PDUType = NFAPI_NR_DL_TTI_PDCCH_PDU_TYPE;
    dl_tti_pdcch_pdu->PDUSize = (uint8_t)(2 + sizeof(nfapi_nr_dl_tti_pdcch_pdu));
    dl_req->nPDUs += 1;
    pdcch_pdu_rel15 = &dl_tti_pdcch_pdu->pdcch_pdu.pdcch_pdu_rel15;
    nr_configure_pdcch(pdcch_pdu_rel15, coreset, &sched_ctrl->sched_pdcch);
    nr_mac->pdcch_pdu_idx[CC_id][coresetid] = pdcch_pdu_rel15;
  }

  nfapi_nr_dl_tti_request_pdu_t *dl_tti_pdsch_pdu = &dl_req->dl_tti_pdu_list[dl_req->nPDUs];
  memset((void *)dl_tti_pdsch_pdu, 0, sizeof(nfapi_nr_dl_tti_request_pdu_t));
  dl_tti_pdsch_pdu->PDUType = NFAPI_NR_DL_TTI_PDSCH_PDU_TYPE;
  dl_tti_pdsch_pdu->PDUSize = (uint8_t)(2 + sizeof(nfapi_nr_dl_tti_pdsch_pdu));
  dl_req->nPDUs += 1;
  nfapi_nr_dl_tti_pdsch_pdu_rel15_t *pdsch_pdu_rel15 = &dl_tti_pdsch_pdu->pdsch_pdu.pdsch_pdu_rel15;

  NR_COMMON_channels_t *cc = &nr_mac->common_channels[CC_id];
  NR_ServingCellConfigCommon_t *scc = cc->ServingCellConfigCommon;
  NR_UE_DL_BWP_t *dl_bwp = &UE->current_DL_BWP;
  int mcsTableIdx = dl_bwp->mcsTableIdx;

  pdsch_pdu_rel15->pduBitmap = 0;
  pdsch_pdu_rel15->rnti = rnti;
  pdsch_pdu_rel15->pduIndex = pduindex;
  pdsch_pdu_rel15->BWPSize  = BWPSize;
  pdsch_pdu_rel15->BWPStart = BWPStart;
  pdsch_pdu_rel15->SubcarrierSpacing = dl_bwp->scs;
  pdsch_pdu_rel15->CyclicPrefix = 0;
  pdsch_pdu_rel15->NrOfCodewords = 1;
  pdsch_pdu_rel15->targetCodeRate[0] = sched_pdsch->R;
  pdsch_pdu_rel15->qamModOrder[0] = sched_pdsch->Qm;
  pdsch_pdu_rel15->TBSize[0] = tb_size;
  pdsch_pdu_rel15->mcsIndex[0] = sched_pdsch->mcs;
  pdsch_pdu_rel15->mcsTable[0] = mcsTableIdx;
  pdsch_pdu_rel15->rvIndex[0] = nr_get_rv(round % 4);
  pdsch_pdu_rel15->dataScramblingId = *scc->physCellId;
  pdsch_pdu_rel15->nrOfLayers = 1;
  pdsch_pdu_rel15->transmissionScheme = 0;
  pdsch_pdu_rel15->refPoint = 0;
  pdsch_pdu_rel15->dmrsConfigType = sched_pdsch->dmrs_parms.dmrsConfigType;
  pdsch_pdu_rel15->dlDmrsScramblingId = sched_pdsch->dmrs_parms.scrambling_id;
  pdsch_pdu_rel15->SCID = sched_pdsch->dmrs_parms.n_scid;
  pdsch_pdu_rel15->numDmrsCdmGrpsNoData = sched_pdsch->dmrs_parms.numDmrsCdmGrpsNoData;
  pdsch_pdu_rel15->dmrsPorts = 1;
  pdsch_pdu_rel15->resourceAlloc = 1;
  pdsch_pdu_rel15->rbStart = sched_pdsch->rbStart;
  pdsch_pdu_rel15->rbSize = sched_pdsch->rbSize;
  pdsch_pdu_rel15->VRBtoPRBMapping = 0;
  pdsch_pdu_rel15->StartSymbolIndex = sched_pdsch->tda_info.startSymbolIndex;
  pdsch_pdu_rel15->NrOfSymbols = sched_pdsch->tda_info.nrOfSymbols;
  pdsch_pdu_rel15->dlDmrsSymbPos = sched_pdsch->dmrs_parms.dl_dmrs_symb_pos;
  pdsch_pdu_rel15->maintenance_parms_v3.tbSizeLbrmBytes = nr_compute_tbslbrm(mcsTableIdx, UE->sc_info.dl_bw_tbslbrm, 1);
  pdsch_pdu_rel15->maintenance_parms_v3.ldpcBaseGraph = get_BG(tb_size << 3, sched_pdsch->R);

  pdsch_pdu_rel15->precodingAndBeamforming.num_prgs = 1;
  pdsch_pdu_rel15->precodingAndBeamforming.prg_size = 275;
  pdsch_pdu_rel15->precodingAndBeamforming.dig_bf_interfaces = 1;
  pdsch_pdu_rel15->precodingAndBeamforming.prgs_list[0].pm_idx = 0;
  pdsch_pdu_rel15->precodingAndBeamforming.prgs_list[0].dig_bf_interface_list[0].beam_idx = UE->UE_beam_index;

  /* Fill PDCCH DL DCI PDU */
  nfapi_nr_dl_dci_pdu_t *dci_pdu = &pdcch_pdu_rel15->dci_pdu[pdcch_pdu_rel15->numDlDci];
  pdcch_pdu_rel15->numDlDci++;
  dci_pdu->RNTI = rnti;
  dci_pdu->ScramblingId = *scc->physCellId;
  dci_pdu->ScramblingRNTI = 0;
  dci_pdu->AggregationLevel = aggregation_level;
  dci_pdu->CceIndex = CCEIndex;
  dci_pdu->beta_PDCCH_1_0 = 0;
  dci_pdu->powerControlOffsetSS = 1;

  dci_pdu->precodingAndBeamforming.num_prgs = 0;
  dci_pdu->precodingAndBeamforming.prg_size = 0;
  dci_pdu->precodingAndBeamforming.dig_bf_interfaces = 1;
  dci_pdu->precodingAndBeamforming.prgs_list[0].pm_idx = 0;
  dci_pdu->precodingAndBeamforming.prgs_list[0].dig_bf_interface_list[0].beam_idx = UE->UE_beam_index;

  dci_pdu_rel15_t dci_payload;
  dci_payload.frequency_domain_assignment.val = PRBalloc_to_locationandbandwidth0(pdsch_pdu_rel15->rbSize,
                                                                                  pdsch_pdu_rel15->rbStart,
                                                                                  BWPSize);

  dci_payload.time_domain_assignment.val = time_domain_assignment;
  dci_payload.vrb_to_prb_mapping.val = 0;
  dci_payload.mcs = pdsch_pdu_rel15->mcsIndex[0];
  dci_payload.tb_scaling = tb_scaling;
  if (rnti_type == TYPE_TC_RNTI_) {
    dci_payload.format_indicator = 1;
    dci_payload.rv = pdsch_pdu_rel15->rvIndex[0];
    dci_payload.harq_pid.val = current_harq_pid;
    dci_payload.ndi = ndi;
    dci_payload.dai[0].val = pucch ? (pucch->dai_c-1) & 3 : 0;
    dci_payload.tpc = tpc; // TPC for PUCCH: table 7.2.1-1 in 38.213
    dci_payload.pucch_resource_indicator = delta_PRI; // This is delta_PRI from 9.2.1 in 38.213
    dci_payload.pdsch_to_harq_feedback_timing_indicator.val = pucch ? pucch->timing_indicator : 0;
  }

  LOG_D(NR_MAC,
        "DCI 1_0 payload: freq_alloc %d (%d,%d,%d), time_alloc %d, vrb to prb %d, mcs %d tb_scaling %d pucchres %d harqtiming %d\n",
        dci_payload.frequency_domain_assignment.val,
        pdsch_pdu_rel15->rbStart,
        pdsch_pdu_rel15->rbSize,
        pdsch_pdu_rel15->BWPSize,
        dci_payload.time_domain_assignment.val,
        dci_payload.vrb_to_prb_mapping.val,
        dci_payload.mcs,
        dci_payload.tb_scaling,
        dci_payload.pucch_resource_indicator,
        dci_payload.pdsch_to_harq_feedback_timing_indicator.val);

  LOG_D(NR_MAC,
        "DCI params: rnti 0x%x, rnti_type %d, dci_format %d coreset params: FreqDomainResource %llx, start_symbol %d  "
        "n_symb %d, BWPsize %d\n",
        pdcch_pdu_rel15->dci_pdu[0].RNTI,
        rnti_type,
        NR_DL_DCI_FORMAT_1_0,
        (unsigned long long)pdcch_pdu_rel15->FreqDomainResource,
        pdcch_pdu_rel15->StartSymbolIndex,
        pdcch_pdu_rel15->DurationSymbols,
        pdsch_pdu_rel15->BWPSize);

  fill_dci_pdu_rel15(&UE->sc_info,
                     dl_bwp,
                     &UE->current_UL_BWP,
                     &pdcch_pdu_rel15->dci_pdu[pdcch_pdu_rel15->numDlDci - 1],
                     &dci_payload,
                     NR_DL_DCI_FORMAT_1_0,
                     rnti_type,
                     dl_bwp->bwp_id,
                     sched_ctrl->search_space,
                     coreset,
                     0, // parameter not needed for DCI 1_0
                     nr_mac->cset0_bwp_size);

  LOG_D(NR_MAC, "BWPSize: %i\n", pdcch_pdu_rel15->BWPSize);
  LOG_D(NR_MAC, "BWPStart: %i\n", pdcch_pdu_rel15->BWPStart);
  LOG_D(NR_MAC, "SubcarrierSpacing: %i\n", pdcch_pdu_rel15->SubcarrierSpacing);
  LOG_D(NR_MAC, "CyclicPrefix: %i\n", pdcch_pdu_rel15->CyclicPrefix);
  LOG_D(NR_MAC, "StartSymbolIndex: %i\n", pdcch_pdu_rel15->StartSymbolIndex);
  LOG_D(NR_MAC, "DurationSymbols: %i\n", pdcch_pdu_rel15->DurationSymbols);
  for (int n = 0; n < 6; n++)
    LOG_D(NR_MAC, "FreqDomainResource[%i]: %x\n", n, pdcch_pdu_rel15->FreqDomainResource[n]);
  LOG_D(NR_MAC, "CceRegMappingType: %i\n", pdcch_pdu_rel15->CceRegMappingType);
  LOG_D(NR_MAC, "RegBundleSize: %i\n", pdcch_pdu_rel15->RegBundleSize);
  LOG_D(NR_MAC, "InterleaverSize: %i\n", pdcch_pdu_rel15->InterleaverSize);
  LOG_D(NR_MAC, "CoreSetType: %i\n", pdcch_pdu_rel15->CoreSetType);
  LOG_D(NR_MAC, "ShiftIndex: %i\n", pdcch_pdu_rel15->ShiftIndex);
  LOG_D(NR_MAC, "precoderGranularity: %i\n", pdcch_pdu_rel15->precoderGranularity);
  LOG_D(NR_MAC, "numDlDci: %i\n", pdcch_pdu_rel15->numDlDci);
}

static void nr_generate_Msg2(module_id_t module_idP,
                             int CC_id,
                             frame_t frameP,
                             slot_t slotP,
                             NR_UE_info_t *UE,
                             nfapi_nr_dl_tti_request_t *DL_req,
                             nfapi_nr_tx_data_request_t *TX_req)
{
  gNB_MAC_INST *nr_mac = RC.nrmac[module_idP];

  // no DL -> cannot send Msg2
  if (!is_dl_slot(slotP, &nr_mac->frame_structure)) {
    return;
  }

  NR_COMMON_channels_t *cc = &nr_mac->common_channels[CC_id];
  NR_UE_DL_BWP_t *dl_bwp = &UE->current_DL_BWP;
  NR_UE_ServingCell_Info_t *sc_info = &UE->sc_info;
  NR_ServingCellConfigCommon_t *scc = cc->ServingCellConfigCommon;
  NR_RA_t *ra = UE->ra;
  long rrc_ra_ResponseWindow =
      scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->rach_ConfigGeneric.ra_ResponseWindow;
  const int n_slots_frame = nr_mac->frame_structure.numb_slots_frame;
  if (!msg2_in_response_window(ra->preamble_frame, ra->preamble_slot, n_slots_frame, rrc_ra_ResponseWindow, frameP, slotP)) {
    LOG_E(NR_MAC, "UE RA-RNTI %04x TC-RNTI %04x: exceeded RA window, cannot schedule Msg2\n", ra->RA_rnti, UE->rnti);
    nr_release_ra_UE(nr_mac, UE->rnti);
    return;
  }

  NR_UE_sched_ctrl_t *sched_ctrl = &UE->UE_sched_ctrl;
  NR_SearchSpace_t *ss = sched_ctrl->search_space;
  if (!check_msg2_monitoring(ss, n_slots_frame, frameP, slotP)) {
    LOG_E(NR_MAC, "UE RA-RNTI %04x TC-RNTI %04x: Msg2 not monitored by UE\n", ra->RA_rnti, UE->rnti);
    return;
  }
  NR_beam_alloc_t beam = beam_allocation_procedure(&nr_mac->beam_info, frameP, slotP, UE->UE_beam_index, n_slots_frame);
  if (beam.idx < 0)
    return;

  const NR_UE_UL_BWP_t *ul_bwp = &UE->current_UL_BWP;
  bool ret = get_feasible_msg3_tda(scc,
                                   get_delta_for_k2(ul_bwp->scs),
                                   ul_bwp->tdaList_Common,
                                   frameP,
                                   slotP,
                                   ra,
                                   &nr_mac->beam_info,
                                   UE->UE_beam_index,
                                   &nr_mac->frame_structure);
  if (!ret || ra->Msg3_tda_id > 15) {
    LOG_D(NR_MAC, "UE RNTI %04x %d.%d: infeasible Msg3 TDA\n", UE->rnti, frameP, slotP);
    reset_beam_status(&nr_mac->beam_info, frameP, slotP, UE->UE_beam_index, n_slots_frame, beam.new_beam);
    return;
  }

  int mcsIndex = -1; // initialization value
  int rbStart = 0;
  int rbSize = 8;
  long BWPStart = 0;
  long BWPSize = 0;
  NR_Type0_PDCCH_CSS_config_t *type0_PDCCH_CSS_config = NULL;
  if (*ss->controlResourceSetId != 0) {
    BWPStart = dl_bwp->BWPStart;
    BWPSize = sc_info->initial_dl_BWPSize;
  } else {
    type0_PDCCH_CSS_config = &nr_mac->type0_PDCCH_CSS_config[cc->ssb_index[UE->UE_beam_index]];
    BWPStart = type0_PDCCH_CSS_config->cset_start_rb;
    BWPSize = type0_PDCCH_CSS_config->num_rbs;
  }

  NR_ControlResourceSet_t *coreset = sched_ctrl->coreset;
  AssertFatal(coreset, "Coreset cannot be null for RA-Msg2\n");
  const int coresetid = coreset->controlResourceSetId;
  // Calculate number of symbols
  int time_domain_assignment = get_dl_tda(nr_mac, slotP);
  int mux_pattern = type0_PDCCH_CSS_config ? type0_PDCCH_CSS_config->type0_pdcch_ss_mux_pattern : 1;
  NR_tda_info_t tda_info = get_dl_tda_info(dl_bwp,
                                           ss->searchSpaceType->present,
                                           time_domain_assignment,
                                           scc->dmrs_TypeA_Position,
                                           mux_pattern,
                                           TYPE_RA_RNTI_,
                                           coresetid,
                                           false);
  if (!tda_info.valid_tda)
    return;

  uint16_t *vrb_map = cc[CC_id].vrb_map[beam.idx];
  for (int i = 0; (i < rbSize) && (rbStart <= (BWPSize - rbSize)); i++) {
    if (vrb_map[BWPStart + rbStart + i] & SL_to_bitmap(tda_info.startSymbolIndex, tda_info.nrOfSymbols)) {
      rbStart += i;
      i = 0;
    }
  }

  if (rbStart > (BWPSize - rbSize)) {
    LOG_W(NR_MAC, "Cannot find free vrb_map for RA RNTI %04x!\n", ra->RA_rnti);
    reset_beam_status(&nr_mac->beam_info, ra->Msg3_frame, ra->Msg3_slot, UE->UE_beam_index, n_slots_frame, ra->Msg3_beam.new_beam);
    reset_beam_status(&nr_mac->beam_info, frameP, slotP, UE->UE_beam_index, n_slots_frame, beam.new_beam);
    return;
  }

  // Checking if the DCI allocation is feasible in current subframe
  nfapi_nr_dl_tti_request_body_t *dl_req = &DL_req->dl_tti_request_body;
  if (dl_req->nPDUs > NFAPI_NR_MAX_DL_TTI_PDUS - 2) {
    LOG_W(NR_MAC, "UE %04x: %d.%d FAPI DL structure is full\n", UE->rnti, frameP, slotP);
    reset_beam_status(&nr_mac->beam_info, ra->Msg3_frame, ra->Msg3_slot, UE->UE_beam_index, n_slots_frame, ra->Msg3_beam.new_beam);
    reset_beam_status(&nr_mac->beam_info, frameP, slotP, UE->UE_beam_index, n_slots_frame, beam.new_beam);
    return;
  }

  uint8_t aggregation_level;
  int CCEIndex = get_cce_index(nr_mac, CC_id, slotP, 0, &aggregation_level, beam.idx, ss, coreset, &sched_ctrl->sched_pdcch, true, 0);

  if (CCEIndex < 0) {
    LOG_W(NR_MAC, "UE %04x: %d.%d cannot find free CCE for Msg2!\n", UE->rnti, frameP, slotP);
    reset_beam_status(&nr_mac->beam_info, ra->Msg3_frame, ra->Msg3_slot, UE->UE_beam_index, n_slots_frame, ra->Msg3_beam.new_beam);
    reset_beam_status(&nr_mac->beam_info, frameP, slotP, UE->UE_beam_index, n_slots_frame, beam.new_beam);
    return;
  }

  bool msg3_ret = nr_get_Msg3alloc(nr_mac, CC_id, slotP, frameP, UE);
  if (!msg3_ret) {
    reset_beam_status(&nr_mac->beam_info, ra->Msg3_frame, ra->Msg3_slot, UE->UE_beam_index, n_slots_frame, ra->Msg3_beam.new_beam);
    reset_beam_status(&nr_mac->beam_info, frameP, slotP, UE->UE_beam_index, n_slots_frame, beam.new_beam);
    return;
  }

  LOG_D(NR_MAC, "Msg2 startSymbolIndex.nrOfSymbols %d.%d\n", tda_info.startSymbolIndex, tda_info.nrOfSymbols);

  // Distance calculation according to SCF222.10.02 RACH.indication (table 3-74) and 38.213 4.2/38.211 4.3.1
  // T_c according to 38.211 4.1
  float T_c_ns = 0.509;
  int numerology = ul_bwp->scs;
  float rtt_ns = T_c_ns * 16 * 64 / (1 << numerology) * ra->timing_offset;
  float speed_of_light_in_meters_per_second = 299792458.0f;
  float distance_in_meters = speed_of_light_in_meters_per_second * rtt_ns / 1000 / 1000 / 1000 / 2;
  LOG_A(NR_MAC,
        "UE %04x: %d.%d Generating RA-Msg2 DCI, RA RNTI 0x%x, state %d, preamble_index(RAPID) %d, "
        "timing_offset = %d (estimated distance %.1f [m])\n",
        UE->rnti,
        frameP,
        slotP,
        ra->RA_rnti,
        ra->ra_state,
        ra->preamble_index,
        ra->timing_offset,
        distance_in_meters);

  // SCF222: PDU index incremented for each PDSCH PDU sent in TX control message. This is used to associate control
  // information to data and is reset every slot.
  const int pduindex = nr_mac->pdu_index[CC_id]++;
  uint8_t mcsTableIdx = dl_bwp->mcsTableIdx;

  NR_pdsch_dmrs_t dmrs_parms = get_dl_dmrs_params(scc, dl_bwp, &tda_info, 1);

  uint8_t tb_scaling = 0;
  int R, Qm;
  uint32_t TBS = 0;
  while (TBS < 9) { // min TBS for RAR is 9 bytes
    mcsIndex++;
    R = nr_get_code_rate_dl(mcsIndex, mcsTableIdx);
    Qm = nr_get_Qm_dl(mcsIndex, mcsTableIdx);
    TBS = nr_compute_tbs(Qm,
                         R,
                         rbSize,
                         tda_info.nrOfSymbols,
                         dmrs_parms.N_PRB_DMRS * dmrs_parms.N_DMRS_SLOT,
                         0, // overhead
                         tb_scaling, // tb scaling
                         1)
          >> 3; // layers
  }

  NR_sched_pdsch_t sched_pdsch = {
    .R = R,
    .Qm = Qm,
    .mcs = mcsIndex,
    .nrOfLayers = 1,
    .dmrs_parms = dmrs_parms,
    .tda_info = tda_info,
    .pm_index = 0,
    .rbStart = rbStart,
    .rbSize = rbSize
  };
  prepare_dl_pdus(nr_mac,
                  UE,
                  &sched_pdsch,
                  dl_req,
                  NULL,
                  TYPE_RA_RNTI_,
                  aggregation_level,
                  CCEIndex,
                  TBS,
                  0,
                  0,
                  0,
                  0,
                  time_domain_assignment,
                  CC_id,
                  ra->RA_rnti,
                  0,
                  tb_scaling,
                  pduindex,
                  BWPStart,
                  BWPSize);

  // DL TX request
  nfapi_nr_pdu_t *tx_req = &TX_req->pdu_list[TX_req->Number_of_PDUs];

  // Program UL processing for Msg3
  nr_add_msg3(module_idP, CC_id, frameP, slotP, UE, (uint8_t *)&tx_req->TLVs[0].value.direct[0]);

  // Start RA contention resolution timer in Msg3 transmission slot (current slot + K2)
  // 3GPP TS 38.321 Section 5.1.5 Contention Resolution
  start_ra_contention_resolution_timer(
      ra,
      scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->ra_ContentionResolutionTimer,
      *ul_bwp->tdaList_Common->list.array[ra->Msg3_tda_id]->k2 + get_NTN_Koffset(scc),
      ul_bwp->scs);

  LOG_D(NR_MAC,
        "UE %04x: %d.%d: Setting RA-Msg3 reception (%s) for SFN.Slot %d.%d\n",
        UE->rnti,
        frameP,
        slotP,
        ra->cfra ? "CFRA" : "CBRA",
        ra->Msg3_frame,
        ra->Msg3_slot);

  LOG_A(NR_MAC, "%d.%d Send RAR to RA-RNTI %04x\n", frameP, slotP, ra->RA_rnti);

  tx_req->PDU_index = pduindex;
  tx_req->num_TLV = 1;
  tx_req->TLVs[0].length = TBS;
  tx_req->PDU_length = compute_PDU_length(tx_req->num_TLV, TBS);
  TX_req->SFN = frameP;
  TX_req->Number_of_PDUs++;
  TX_req->Slot = slotP;

  T(T_GNB_MAC_DL_RAR_PDU_WITH_DATA,
    T_INT(module_idP),
    T_INT(CC_id),
    T_INT(ra->RA_rnti),
    T_INT(frameP),
    T_INT(slotP),
    T_INT(0),
    T_BUFFER(&tx_req->TLVs[0].value.direct[0], tx_req->TLVs[0].length));

  // Mark the corresponding symbols RBs as used
  fill_pdcch_vrb_map(nr_mac, CC_id, &sched_ctrl->sched_pdcch, CCEIndex, aggregation_level, beam.idx);
  for (int rb = 0; rb < rbSize; rb++) {
    vrb_map[BWPStart + rb + rbStart] |= SL_to_bitmap(tda_info.startSymbolIndex, tda_info.nrOfSymbols);
  }

  ra->ra_state = nrRA_WAIT_Msg3;
}

static void nr_generate_Msg4_MsgB(module_id_t module_idP,
                                  int CC_id,
                                  frame_t frameP,
                                  slot_t slotP,
                                  NR_UE_info_t *UE,
                                  nfapi_nr_dl_tti_request_t *DL_req,
                                  nfapi_nr_tx_data_request_t *TX_req)
{
  gNB_MAC_INST *nr_mac = RC.nrmac[module_idP];
  NR_COMMON_channels_t *cc = &nr_mac->common_channels[CC_id];
  NR_UE_DL_BWP_t *dl_bwp = &UE->current_DL_BWP;

  // if it is a DL slot, if the RA is in MSG4 state
  if (is_dl_slot(slotP, &nr_mac->frame_structure)) {
    NR_ServingCellConfigCommon_t *scc = cc->ServingCellConfigCommon;
    NR_UE_sched_ctrl_t *sched_ctrl = &UE->UE_sched_ctrl;
    NR_SearchSpace_t *ss = sched_ctrl->search_space;
    NR_RA_t *ra = UE->ra;
    const char *ra_type_str = ra->ra_type == RA_2_STEP ? "MsgB" : "Msg4";
    NR_ControlResourceSet_t *coreset = sched_ctrl->coreset;
    AssertFatal(coreset != NULL, "Coreset cannot be null for RA %s\n", ra_type_str);

    uint16_t mac_sdu_length = 0;
    /* get the PID of a HARQ process awaiting retrnasmission, or -1 otherwise */
    int current_harq_pid = sched_ctrl->retrans_dl_harq.head;

    logical_chan_id_t lcid = DL_SCH_LCID_CCCH;
    if (current_harq_pid < 0) {
      // Check for data on SRB0 (RRCSetup)
      mac_rlc_status_resp_t srb_status = nr_mac_rlc_status_ind(UE->rnti, frameP, lcid);

      if (srb_status.bytes_in_buffer == 0) {
        lcid = DL_SCH_LCID_DCCH;
        // Check for data on SRB1 (RRCReestablishment, RRCReconfiguration)
        srb_status = nr_mac_rlc_status_ind(UE->rnti, frameP, lcid);
      }

      // Need to wait until data for Msg4 is ready
      if (srb_status.bytes_in_buffer == 0)
        return;
      mac_sdu_length = srb_status.bytes_in_buffer;
    }

    const int n_slots_frame = nr_mac->frame_structure.numb_slots_frame;
    NR_beam_alloc_t beam = beam_allocation_procedure(&nr_mac->beam_info, frameP, slotP, UE->UE_beam_index, n_slots_frame);
    if (beam.idx < 0)
      return;

    long BWPStart = 0;
    long BWPSize = 0;
    NR_Type0_PDCCH_CSS_config_t *type0_PDCCH_CSS_config = NULL;
    if(*ss->controlResourceSetId!=0) {
      BWPStart = dl_bwp->BWPStart;
      BWPSize  = dl_bwp->BWPSize;
    } else {
      type0_PDCCH_CSS_config = &nr_mac->type0_PDCCH_CSS_config[cc->ssb_index[UE->UE_beam_index]];
      BWPStart = type0_PDCCH_CSS_config->cset_start_rb;
      BWPSize = type0_PDCCH_CSS_config->num_rbs;
    }

    // get CCEindex, needed also for PUCCH and then later for PDCCH
    uint8_t aggregation_level;
    int CCEIndex = get_cce_index(nr_mac,
                                 CC_id, slotP, 0,
                                 &aggregation_level,
                                 beam.idx,
                                 ss,
                                 coreset,
                                 &sched_ctrl->sched_pdcch,
                                 true,
                                 0);

    if (CCEIndex < 0) {
      LOG_E(NR_MAC, "Cannot find free CCE for RA RNTI 0x%04x!\n", UE->rnti);
      reset_beam_status(&nr_mac->beam_info, frameP, slotP, UE->UE_beam_index, n_slots_frame, beam.new_beam);
      return;
    }

    // Checking if the DCI allocation is feasible in current subframe
    nfapi_nr_dl_tti_request_body_t *dl_req = &DL_req->dl_tti_request_body;
    if (dl_req->nPDUs > NFAPI_NR_MAX_DL_TTI_PDUS - 2) {
      LOG_I(NR_MAC, "UE %04x: %d.%d FAPI DL structure is full\n", UE->rnti, frameP, slotP);
      reset_beam_status(&nr_mac->beam_info, frameP, slotP, UE->UE_beam_index, n_slots_frame, beam.new_beam);
      return;
    }

    uint8_t time_domain_assignment = get_dl_tda(nr_mac, slotP);
    int mux_pattern = type0_PDCCH_CSS_config ? type0_PDCCH_CSS_config->type0_pdcch_ss_mux_pattern : 1;
    NR_tda_info_t msg4_tda = get_dl_tda_info(dl_bwp,
                                             ss->searchSpaceType->present,
                                             time_domain_assignment,
                                             scc->dmrs_TypeA_Position,
                                             mux_pattern,
                                             TYPE_TC_RNTI_,
                                             coreset->controlResourceSetId,
                                             false);
    if (!msg4_tda.valid_tda)
      return;

    NR_pdsch_dmrs_t dmrs_info = get_dl_dmrs_params(scc, dl_bwp, &msg4_tda, 1);

    uint8_t mcsTableIdx = dl_bwp->mcsTableIdx;
    uint8_t mcsIndex = 0;
    int rbStart = 0;
    int rbSize = 0;
    uint8_t tb_scaling = 0;
    uint32_t tb_size = 0;
    uint16_t pdu_length;
    if(current_harq_pid >= 0) { // in case of retransmission
      NR_UE_harq_t *harq = &sched_ctrl->harq_processes[current_harq_pid];
      DevAssert(!harq->is_waiting);
      pdu_length = harq->tb_size;
    }
    else {
      uint8_t subheader_len = (mac_sdu_length < 256) ? sizeof(NR_MAC_SUBHEADER_SHORT) : sizeof(NR_MAC_SUBHEADER_LONG);
      pdu_length = mac_sdu_length + subheader_len + 13; // 13 is contention resolution length of msgB. It is divided into 3 parts:
      // BI MAC subheader (1 Oct) + SuccessRAR MAC subheader (1 Oct) + SuccessRAR (11 Oct)
    }

    // increase PRBs until we get to BWPSize or TBS is bigger than MAC PDU size
    do {
      if(rbSize < BWPSize)
        rbSize++;
      else
        mcsIndex++;
      LOG_D(NR_MAC,"Calling nr_compute_tbs with N_PRB_DMRS %d, N_DMRS_SLOT %d\n",dmrs_info.N_PRB_DMRS,dmrs_info.N_DMRS_SLOT);
      tb_size = nr_compute_tbs(nr_get_Qm_dl(mcsIndex, mcsTableIdx),
                               nr_get_code_rate_dl(mcsIndex, mcsTableIdx),
                               rbSize,
                               msg4_tda.nrOfSymbols,
                               dmrs_info.N_PRB_DMRS * dmrs_info.N_DMRS_SLOT,
                               0,
                               tb_scaling,1) >> 3;
    } while (tb_size < pdu_length && mcsIndex<=28);

    AssertFatal(tb_size >= pdu_length, "Cannot allocate %s\n", ra_type_str);

    int i = 0;
    uint16_t *vrb_map = cc[CC_id].vrb_map[beam.idx];
    while ((i < rbSize) && (rbStart + rbSize <= BWPSize)) {
      if (vrb_map[BWPStart + rbStart + i]&SL_to_bitmap(msg4_tda.startSymbolIndex, msg4_tda.nrOfSymbols)) {
        rbStart += i+1;
        i = 0;
      } else {
        i++;
      }
    }

    if (rbStart > (BWPSize - rbSize)) {
      LOG_E(NR_MAC, "Cannot find free vrb_map for RNTI %04x!\n", UE->rnti);
      reset_beam_status(&nr_mac->beam_info, frameP, slotP, UE->UE_beam_index, n_slots_frame, beam.new_beam);
      return;
    }

    const int delta_PRI = 0;
    int r_pucch = nr_get_pucch_resource(coreset, UE->current_UL_BWP.pucch_Config, CCEIndex);
    LOG_D(NR_MAC, "Msg4 r_pucch %d (CCEIndex %d, delta_PRI %d)\n", r_pucch, CCEIndex, delta_PRI);
    int alloc = nr_acknack_scheduling(nr_mac, UE, frameP, slotP, UE->UE_beam_index, r_pucch, 1);
    if (alloc < 0) {
      LOG_D(NR_MAC,"Couldn't find a pucch allocation for ack nack (msg4) in frame %d slot %d\n", frameP, slotP);
      reset_beam_status(&nr_mac->beam_info, frameP, slotP, UE->UE_beam_index, n_slots_frame, beam.new_beam);
      return;
    }

    // HARQ management
    if (current_harq_pid < 0) {
      AssertFatal(sched_ctrl->available_dl_harq.head >= 0,
                  "UE context not initialized: no HARQ processes found\n");
      current_harq_pid = sched_ctrl->available_dl_harq.head;
      remove_front_nr_list(&sched_ctrl->available_dl_harq);
    }

    NR_UE_harq_t *harq = &sched_ctrl->harq_processes[current_harq_pid];
    NR_sched_pucch_t *pucch = &sched_ctrl->sched_pucch[alloc];
    DevAssert(!harq->is_waiting);
    add_tail_nr_list(&sched_ctrl->feedback_dl_harq, current_harq_pid);
    harq->feedback_slot = pucch->ul_slot;
    harq->feedback_frame = pucch->frame;
    harq->is_waiting = true;
    ra->harq_pid = current_harq_pid;
    UE->mac_stats.dl.rounds[harq->round]++;
    harq->tb_size = tb_size;
    uint8_t *buf = allocate_transportBlock_buffer(&harq->transportBlock, tb_size);
    // Bytes to be transmitted
    if (harq->round == 0) {
      uint16_t mac_pdu_length = 0;
      if (ra->ra_type == RA_4_STEP) {
        // UE Contention Resolution Identity MAC CE
        mac_pdu_length = nr_write_ce_dlsch_pdu(module_idP, sched_ctrl, buf, 255, ra->cont_res_id);
      } else if (ra->ra_type == RA_2_STEP) {
        mac_pdu_length = nr_fill_successrar(sched_ctrl,
                                            UE->rnti,
                                            ra->cont_res_id,
                                            pucch->resource_indicator,
                                            pucch->timing_indicator,
                                            buf,
                                            mac_pdu_length);
      } else {
        AssertFatal(false, "RA type %d not implemented!\n", ra->ra_type);
      }

      uint8_t buffer[CCCH_SDU_SIZE];
      uint8_t mac_subheader_len = sizeof(NR_MAC_SUBHEADER_SHORT);
      // Get RLC data on the SRB (RRCSetup, RRCReestablishment)
      mac_sdu_length = nr_mac_rlc_data_req(module_idP, UE->rnti, true, lcid, CCCH_SDU_SIZE, (char *)buffer);

      if (mac_sdu_length < 256) {
        ((NR_MAC_SUBHEADER_SHORT *)&buf[mac_pdu_length])->R = 0;
        ((NR_MAC_SUBHEADER_SHORT *)&buf[mac_pdu_length])->F = 0;
        ((NR_MAC_SUBHEADER_SHORT *)&buf[mac_pdu_length])->LCID = lcid;
        ((NR_MAC_SUBHEADER_SHORT *)&buf[mac_pdu_length])->L = mac_sdu_length;
        ra->mac_pdu_length = mac_pdu_length + mac_sdu_length + sizeof(NR_MAC_SUBHEADER_SHORT);
      } else {
        mac_subheader_len = sizeof(NR_MAC_SUBHEADER_LONG);
        ((NR_MAC_SUBHEADER_LONG *)&buf[mac_pdu_length])->R = 0;
        ((NR_MAC_SUBHEADER_LONG *)&buf[mac_pdu_length])->F = 1;
        ((NR_MAC_SUBHEADER_LONG *)&buf[mac_pdu_length])->LCID = lcid;
        ((NR_MAC_SUBHEADER_LONG *)&buf[mac_pdu_length])->L = htons(mac_sdu_length);
        ra->mac_pdu_length = mac_pdu_length + mac_sdu_length + sizeof(NR_MAC_SUBHEADER_LONG);
      }
      memcpy(&buf[mac_pdu_length + mac_subheader_len], buffer, mac_sdu_length);
    }

    rnti_t rnti = ra->ra_type == RA_4_STEP ? UE->rnti : ra->MsgB_rnti;
    NR_sched_pdsch_t sched_pdsch = {
      .R = nr_get_code_rate_dl(mcsIndex, mcsTableIdx),
      .Qm = nr_get_Qm_dl(mcsIndex, mcsTableIdx),
      .mcs = mcsIndex,
      .nrOfLayers = 1,
      .dmrs_parms = dmrs_info,
      .tda_info = msg4_tda,
      .pm_index = 0,
      .rbStart = rbStart,
      .rbSize = rbSize
    };
    const int pduindex = nr_mac->pdu_index[CC_id]++;
    prepare_dl_pdus(nr_mac,
                    UE,
                    &sched_pdsch,
                    dl_req,
                    pucch,
                    TYPE_TC_RNTI_,
                    aggregation_level,
                    CCEIndex,
                    tb_size,
                    harq->ndi,
                    sched_ctrl->tpc1,
                    delta_PRI,
                    current_harq_pid,
                    time_domain_assignment,
                    CC_id,
                    rnti,
                    harq->round,
                    tb_scaling,
                    pduindex,
                    BWPStart,
                    BWPSize);

    // Reset TPC to 0 dB to not request new gain multiple times before computing new value for SNR
    sched_ctrl->tpc1 = 1;

    // Add padding header and zero rest out if there is space left
    if (ra->mac_pdu_length < harq->tb_size) {
      NR_MAC_SUBHEADER_FIXED *padding = (NR_MAC_SUBHEADER_FIXED *) &buf[ra->mac_pdu_length];
      padding->R = 0;
      padding->LCID = DL_SCH_LCID_PADDING;
      for(int k = ra->mac_pdu_length+1; k<harq->tb_size; k++) {
        buf[k] = 0;
      }
    }

    T(T_GNB_MAC_DL_PDU_WITH_DATA, T_INT(module_idP), T_INT(CC_id), T_INT(UE->rnti),
      T_INT(frameP), T_INT(slotP), T_INT(current_harq_pid), T_BUFFER(harq->transportBlock.buf, harq->tb_size));

    // DL TX request
    nfapi_nr_pdu_t *tx_req = &TX_req->pdu_list[TX_req->Number_of_PDUs];
    memcpy(tx_req->TLVs[0].value.direct, harq->transportBlock.buf, sizeof(uint8_t) * harq->tb_size);
    tx_req->PDU_index = pduindex;
    tx_req->num_TLV = 1;
    tx_req->TLVs[0].length =  harq->tb_size;
    tx_req->PDU_length = compute_PDU_length(tx_req->num_TLV, tx_req->TLVs[0].length);
    TX_req->SFN = frameP;
    TX_req->Number_of_PDUs++;
    TX_req->Slot = slotP;

    // Mark the corresponding symbols and RBs as used
    fill_pdcch_vrb_map(nr_mac,
                       CC_id,
                       &sched_ctrl->sched_pdcch,
                       CCEIndex,
                       aggregation_level,
                       beam.idx);
    for (int rb = 0; rb < rbSize; rb++) {
      vrb_map[BWPStart + rb + rbStart] |= SL_to_bitmap(msg4_tda.startSymbolIndex, msg4_tda.nrOfSymbols);
    }

    ra->ra_state = nrRA_WAIT_Msg4_MsgB_ACK;
    LOG_I(NR_MAC,
          "UE %04x Generate %s: feedback at %4d.%2d, payload %d bytes, next state nrRA_WAIT_Msg4_MsgB_ACK\n",
          UE->rnti,
          ra_type_str,
          pucch->frame,
          pucch->ul_slot,
          harq->tb_size);
  }
}

void nr_check_Msg4_MsgB_Ack(module_id_t module_id, frame_t frame, slot_t slot, NR_UE_info_t *UE, bool success)
{
  NR_RA_t *ra = UE->ra;
  const char *ra_type_str = ra->ra_type == RA_2_STEP ? "MsgB" : "Msg4";
  const int current_harq_pid = ra->harq_pid;

  NR_UE_sched_ctrl_t *sched_ctrl = &UE->UE_sched_ctrl;
  NR_UE_harq_t *harq = &sched_ctrl->harq_processes[current_harq_pid];

  LOG_D(NR_MAC, "ue rnti 0x%04x, round %d, frame %d %d, harq id %d\n", UE->rnti, harq->round, frame, slot, current_harq_pid);

  if (harq->round == 0) {
    if (success) {
      gNB_MAC_INST *nr_mac = RC.nrmac[module_id];
      NR_ServingCellConfigCommon_t *scc = nr_mac->common_channels[0].ServingCellConfigCommon;
      // we configure the UE using common search space with DCIX0 while waiting for a reconfiguration
      configure_UE_BWP(nr_mac, scc, UE, false, NR_SearchSpace__searchSpaceType_PR_common, -1, -1);
      transition_ra_connected_nr_ue(nr_mac, UE);
      LOG_A(NR_MAC, "%4d.%2d UE %04x: Received Ack of %s. CBRA procedure succeeded!\n", frame, slot, UE->rnti, ra_type_str);
    } else {
      LOG_I(NR_MAC, "%4d.%2d UE %04x: RA Procedure failed at %s!\n", frame, slot, UE->rnti, ra_type_str);
      nr_mac_trigger_ul_failure(sched_ctrl, UE->current_DL_BWP.scs);
    }

    if (sched_ctrl->retrans_dl_harq.head >= 0) {
      remove_nr_list(&sched_ctrl->retrans_dl_harq, current_harq_pid);
    }
  } else {
    LOG_I(NR_MAC, "(UE %04x) Received Nack in %s, preparing retransmission!\n", UE->rnti, ra_type_str);
    ra->ra_state = ra->ra_type == RA_4_STEP ? nrRA_Msg4 : nrRA_MsgB;
  }
}

/////////////////////////////////////
//    Random Access Response PDU   //
//         TS 38.213 ch 8.2        //
//        TS 38.321 ch 6.2.3       //
/////////////////////////////////////
//| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |// bit-wise
//| E | T |       R A P I D       |//
//| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |//
//| R |           T A             |//
//|       T A         |  UL grant |//
//|            UL grant           |//
//|            UL grant           |//
//|            UL grant           |//
//|         T C - R N T I         |//
//|         T C - R N T I         |//
/////////////////////////////////////
//       UL grant  (27 bits)       //
/////////////////////////////////////
//| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |// bit-wise
//|-------------------|FHF|F_alloc|//
//|        Freq allocation        |//
//|    F_alloc    |Time allocation|//
//|      MCS      |     TPC   |CSI|//
/////////////////////////////////////
// WIP
// todo:
// - handle MAC RAR BI subheader
// - sending only 1 RAR subPDU
// - UL Grant: hardcoded CSI, TPC, time alloc
// - padding
static void nr_fill_rar(uint8_t Mod_idP, NR_UE_info_t *UE, uint8_t *dlsch_buffer, nfapi_nr_pusch_pdu_t *pusch_pdu)
{
  NR_RA_t *ra = UE->ra;
  LOG_D(NR_MAC,
        "[gNB] Generate RAR MAC PDU frame %d slot %d preamble index %u TA command %d \n",
        ra->Msg2_frame,
        ra->Msg2_slot,
        ra->preamble_index,
        ra->timing_offset);
  NR_RA_HEADER_BI *rarbi = (NR_RA_HEADER_BI *) dlsch_buffer;
  NR_RA_HEADER_RAPID *rarh = (NR_RA_HEADER_RAPID *) (dlsch_buffer + 1);
  NR_MAC_RAR *rar = (NR_MAC_RAR *) (dlsch_buffer + 2);
  unsigned char csi_req = 0;

  /// E/T/R/R/BI subheader ///
  // E = 1, MAC PDU includes another MAC sub-PDU (RAPID)
  // T = 0, Back-off indicator subheader
  // R = 2, Reserved
  // BI = 0, 5ms
  rarbi->E = 1;
  rarbi->T = 0;
  rarbi->R = 0;
  rarbi->BI = 0;

  /// E/T/RAPID subheader ///
  // E = 0, one only RAR, first and last
  // T = 1, RAPID
  rarh->E = 0;
  rarh->T = 1;
  rarh->RAPID = ra->preamble_index;

  /// RAR MAC payload ///
  rar->R = 0;

  // TA command
  rar->TA1 = (uint8_t) (ra->timing_offset >> 5);    // 7 MSBs of timing advance
  rar->TA2 = (uint8_t) (ra->timing_offset & 0x1f);  // 5 LSBs of timing advance

  // TC-RNTI
  rar->TCRNTI_1 = (uint8_t) (UE->rnti >> 8);        // 8 MSBs of rnti
  rar->TCRNTI_2 = (uint8_t) (UE->rnti & 0xff);      // 8 LSBs of rnti

  // UL grant

  if (pusch_pdu->frequency_hopping)
    AssertFatal(1==0,"PUSCH with frequency hopping currently not supported");

  int bwp_size = pusch_pdu->bwp_size;
  int prb_alloc = PRBalloc_to_locationandbandwidth0(ra->msg3_nb_rb, ra->msg3_first_rb, bwp_size);
  int valid_bits = 14;
  int f_alloc = prb_alloc & ((1 << valid_bits) - 1);

  uint32_t ul_grant = csi_req | (ra->msg3_TPC << 1) | (pusch_pdu->mcs_index << 4) | (ra->Msg3_tda_id << 8) | (f_alloc << 12) | (pusch_pdu->frequency_hopping << 26);

  rar->UL_GRANT_1 = (uint8_t) (ul_grant >> 24) & 0x07;
  rar->UL_GRANT_2 = (uint8_t) (ul_grant >> 16) & 0xff;
  rar->UL_GRANT_3 = (uint8_t) (ul_grant >> 8) & 0xff;
  rar->UL_GRANT_4 = (uint8_t) ul_grant & 0xff;

#ifdef DEBUG_RAR
  LOG_I(NR_MAC, "rarh->E = 0x%x\n", rarh->E);
  LOG_I(NR_MAC, "rarh->T = 0x%x\n", rarh->T);
  LOG_I(NR_MAC, "rarh->RAPID = 0x%x (%i)\n", rarh->RAPID, rarh->RAPID);
  LOG_I(NR_MAC, "rar->R = 0x%x\n", rar->R);
  LOG_I(NR_MAC, "rar->TA1 = 0x%x\n", rar->TA1);
  LOG_I(NR_MAC, "rar->TA2 = 0x%x\n", rar->TA2);
  LOG_I(NR_MAC, "rar->UL_GRANT_1 = 0x%x\n", rar->UL_GRANT_1);
  LOG_I(NR_MAC, "rar->UL_GRANT_2 = 0x%x\n", rar->UL_GRANT_2);
  LOG_I(NR_MAC, "rar->UL_GRANT_3 = 0x%x\n", rar->UL_GRANT_3);
  LOG_I(NR_MAC, "rar->UL_GRANT_4 = 0x%x\n", rar->UL_GRANT_4);
  LOG_I(NR_MAC, "rar->TCRNTI_1 = 0x%x\n", rar->TCRNTI_1);
  LOG_I(NR_MAC, "rar->TCRNTI_2 = 0x%x\n", rar->TCRNTI_2);
#endif
  LOG_D(NR_MAC,
        "In %s: Transmitted RAR with t_alloc %d f_alloc %d ta_command %d mcs %d freq_hopping %d tpc_command %d csi_req %d t_crnti "
        "%x \n",
        __FUNCTION__,
        rar->UL_GRANT_3 & 0x0f,
        (rar->UL_GRANT_3 >> 4) | (rar->UL_GRANT_2 << 4) | ((rar->UL_GRANT_1 & 0x03) << 12),
        rar->TA2 + (rar->TA1 << 5),
        rar->UL_GRANT_4 >> 4,
        rar->UL_GRANT_1 >> 2,
        ra->msg3_TPC,
        csi_req,
        rar->TCRNTI_2 + (rar->TCRNTI_1 << 8));

  // resetting msg3 TPC to 0dB for possible retransmissions
  ra->msg3_TPC = 1;
}

/** @brief remove the UE with RNTI rnti from list of UEs doing RA.
 *
 * The corresponding function to add is add_new_UE_RA(). */
void nr_release_ra_UE(gNB_MAC_INST *mac, rnti_t rnti)
{
  NR_UEs_t *UE_info = &mac->UE_info;
  NR_SCHED_LOCK(&UE_info->mutex);
  NR_UE_info_t *UE = remove_UE_from_list(NR_NB_RA_PROC_MAX, UE_info->access_ue_list, rnti);
  NR_SCHED_UNLOCK(&UE_info->mutex);
  if (UE) {
    delete_nr_ue_data(UE, mac->common_channels, &UE_info->uid_allocator);
  } else {
    LOG_W(NR_MAC,"Call to del rnti %04x, but not existing\n", rnti);
  }
}

void nr_schedule_RA(module_id_t module_idP,
                    frame_t frameP,
                    slot_t slotP,
                    nfapi_nr_ul_dci_request_t *ul_dci_req,
                    nfapi_nr_dl_tti_request_t *DL_req,
                    nfapi_nr_tx_data_request_t *TX_req)
{
  gNB_MAC_INST *mac = RC.nrmac[module_idP];
  /* already mutex protected: held in gNB_dlsch_ulsch_scheduler() */
  NR_SCHED_ENSURE_LOCKED(&mac->sched_lock);

  start_meas(&mac->schedule_ra);
  for (int CC_id = 0; CC_id < MAX_NUM_CCs; CC_id++) {
    UE_iterator(mac->UE_info.access_ue_list, UE) {
      NR_RA_t *ra = UE->ra;
      if (ra->ra_state != nrRA_gNB_IDLE)
        LOG_D(NR_MAC, "UE %04x frame.slot %d.%d RA state: %d\n", UE->rnti, frameP, slotP, ra->ra_state);

      // Check RA Contention Resolution timer (TODO check this procedure)
      if (ra->ra_type == RA_4_STEP && ra->ra_state > nrRA_WAIT_Msg3) {
        ra->contention_resolution_timer--;
        if (ra->contention_resolution_timer == 0) {
          LOG_W(NR_MAC, "(%d.%d) RA Contention Resolution timer expired for UE 0x%04x, RA procedure failed...\n", frameP, slotP, UE->rnti);
          bool requested = nr_mac_request_release_ue(mac, UE->rnti);
          if (!requested)
            nr_release_ra_UE(mac, UE->rnti);
          continue;
        }
      }

      switch (ra->ra_state) {
        case nrRA_Msg2:
          nr_generate_Msg2(module_idP, CC_id, frameP, slotP, UE, DL_req, TX_req);
          break;
        case nrRA_Msg3_retransmission:
          nr_generate_Msg3_retransmission(module_idP, CC_id, frameP, slotP, UE, ul_dci_req);
          break;
        case nrRA_Msg4:
        case nrRA_MsgB:
          nr_generate_Msg4_MsgB(module_idP, CC_id, frameP, slotP, UE, DL_req, TX_req);
          break;
        default:
          break;
      }
    }
  }
  stop_meas(&mac->schedule_ra);
}
