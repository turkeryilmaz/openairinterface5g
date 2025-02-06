/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.0  (the "License"); you may not use this file
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

/*! \file PHY/NR_TRANSPORT/nr_ulsch.h
* \brief functions used for PUSCH/ULSCH physical and transport channels for gNB
* \author Ahmed Hussein
* \date 2019
* \version 0.1
* \company Fraunhofer IIS
* \email: ahmed.hussein@iis.fraunhofer.de
* \note
* \warning
*/

#ifndef NR_ULSCH_H_
#define NR_ULSCH_H_

#include "PHY/defs_gNB.h"
#include "common/utils/threadPool/thread-pool.h"

#define NUMBER_FRAMES_PHY_UE_INACTIVE 10

void free_gNB_ulsch(NR_gNB_ULSCH_t *ulsch, uint16_t N_RB_UL);

NR_gNB_ULSCH_t new_gNB_ulsch(uint8_t max_ldpc_iterations, uint16_t N_RB_UL);

/*! \brief Perform PUSCH decoding for the whole current received TTI. TS 38.212 V15.4.0 subclause 6.2
  @param phy_vars_gNB, Pointer to PHY data structure for gNB
  @param frame_parms, Pointer to frame descriptor structure
  @param frame, current received frame
  @param nr_tti_rx, current received TTI
  @param G
  @param ULSCH_ids, array of ULSCH ids
  @param nb_pusch, number of uplink shared channels
*/

int nr_ulsch_decoding(PHY_VARS_gNB *phy_vars_gNB,
                      NR_DL_FRAME_PARMS *frame_parms,
                      uint32_t frame,
                      uint8_t nr_tti_rx,
                      uint32_t *G,
                      uint8_t *ULSCH_ids,
                      int nb_pusch);

/*! \brief Perform PUSCH unscrambling. TS 38.211 V15.4.0 subclause 6.3.1.1
  @param llr, Pointer to llr bits
  @param size, length of llr bits
  @param q, codeword index (0,1)
  @param Nid, cell id
  @param n_RNTI, CRNTI
*/

void nr_ulsch_unscrambling(int16_t* llr, uint32_t size, uint32_t Nid, uint32_t n_RNTI);

void nr_ulsch_layer_demapping(int16_t *llr_cw, uint8_t Nl, uint8_t mod_order, uint32_t length, int16_t **llr_layers);

void dump_pusch_stats(FILE *fd,PHY_VARS_gNB *gNB);

void dump_nr_I0_stats(FILE *fd,PHY_VARS_gNB *gNB);

NR_gNB_SCH_STATS_t *get_ulsch_stats(PHY_VARS_gNB *gNB,NR_gNB_ULSCH_t *ulsch);

/// Channel estimation and compensation

/*! \brief compute average of uint32_t array
 *  \param x uint32_t array to average
 *  \param size size of the array 
 */
uint32_t average_u32(const uint32_t *x, uint16_t size);

/*! \brief get number of resource element in one symbol
 *  \param frame_parms frame parameters
 *  \param rel15_ul PUSCH PDU
 *  \param symbol symbol index
 */
int get_nb_re_pusch(NR_DL_FRAME_PARMS *frame_parms, nfapi_nr_pusch_pdu_t *rel15_ul, int symbol);

/*! \brief extract resource blocks for corresponding PUSCH PDU from symbol and channel model
 *  \param rxdataF input symbol array
 *  \param chF input channel model array
 *  \param rxFext output symbol
 *  \param chFext output channel model
 *  \param rxoffset offset to symbol in input array
 *  \param choffset offset to channel model in input array
 *  \param aarx antenna id
 *  \param is_dmrs_symbol indicates if the symbol is a DMRS symbol
 *  \param pusch_pdu PUSCH PDU to which the RBs belong
 *  \param frme_parms frame parameters
 */
void nr_ulsch_extract_rbs(c16_t* const rxdataF,
                          c16_t* const chF,
                          c16_t *rxFext,
                          c16_t *chFext,
                          int rxoffset,
                          int choffset,
                          int aarx,
                          int is_dmrs_symbol,
                          nfapi_nr_pusch_pdu_t *pusch_pdu,
                          NR_DL_FRAME_PARMS *frame_parms);

/*! \brief scales channel level on each (TX,RX) antenna pair
 *  \param size_est size of channel estimation per antenna
 *  \param ul_ch_estimates_ext channel estimation array
 *         index 1: layer and antenna (layer * total number of antennas + antenna)
 *         index 2: sample
 *  \param frame_parms frame parameters
 *  \param symbol first symbol index
 *  \param is_dmrs_symbol flag indicating if the first symbol is a DMRS symbol
 *  \param len number of RX resource elements
 *  \param nrOfLayers number of layers
 *  \param nb_rb number of resource blocks in corresponding PDU
 *  \param shift_ch_ext shift to compress channel estimation
 */
void nr_ulsch_scale_channel(int size_est,
                            int ul_ch_estimates_ext[][size_est],
                            NR_DL_FRAME_PARMS *frame_parms,
                            uint8_t symbol,
                            uint8_t is_dmrs_symbol,
                            uint32_t len,
                            uint8_t nrOfLayers,
                            unsigned short nb_rb,
                            int shift_ch_ext);

/*! \brief compute average channel level on each (TX,RX) antenna pair
 *  \param size_est size of channel estimation per antenna
 *  \param ul_ch_estimates_ext channel estimation array
 *         index 1: layer and antenna (layer * total number of antennas + antenna)
 *         index 2: sample
 *  \param frame_parms frame parameters
 *  \param avg average channel level
 *         index: layer and antenna (layer * total number of antennas + antenna)
 *  \param symbol first symbol index
 *  \param len number of resource elements
 *  \param nrOfLayers number of layers
 */
void nr_ulsch_channel_level(int size_est,
                            int ul_ch_estimates_ext[][size_est],
                            NR_DL_FRAME_PARMS *frame_parms,
                            int32_t *avg,
                            uint8_t symbol,
                            uint32_t len,
                            uint8_t nrOfLayers);

/*! \brief performs channel compensation on symbol
 *  \param rxFext input symbol after RBs extraction
 *  \param chFext input channel model after RBs extraction
 *  \param ul_ch_maga channel model aggregate A (Qm > 2), or something like that ...
 *  \param ul_ch_magb channel model aggregate B (Qm > 4), or something like that ...
 *  \param ul_ch_magc channel model aggregate C (Qm > 6), or something like that ...
 *  \param rxComp output compensated symbol
 *         index 1: layer and RX antenna (layer * total number of RX antennas + antenna)
 *         index 2: sample
 *  \param rho aggregated cross layer channel estimation correlation, or something like that ...
 *  \param frame_parms frame parameters
 *  \param buffer_length number of samples allocated per symbol in arrays
 *  \param output_shift shift to reduce quadratic channel estimate size, or something like that ...
 */
void nr_ulsch_channel_compensation(c16_t *rxFext,
                                   c16_t *chFext,
                                   c16_t *ul_ch_maga,
                                   c16_t *ul_ch_magb,
                                   c16_t *ul_ch_magc,
                                   int32_t **rxComp,
                                   c16_t *rho,
                                   NR_DL_FRAME_PARMS *frame_parms,
                                   nfapi_nr_pusch_pdu_t* rel15_ul,
                                   uint32_t symbol,
                                   uint32_t buffer_length,
                                   uint32_t output_shift);

typedef struct chestcomp_params_s {
  int frame;
  int slot;
  int beam_nb;
  int ulsch_id;
  // gNB
  /// Physical Cell ID, 𝑁_{𝐼𝐷}^{𝑐𝑒𝑙𝑙} [38.211, sec 7.4.2.1] Value: 0 ->1007
  uint16_t phy_cell_id;
  /// indicate the channel estimation technique in time domain
  int chest_time;
  /// indicate the channel estimation technique in freq domain
  int chest_freq;
  /// number of RX antennas processed within each channel estimation task
  /// Must be equal to the total number of antennas if multi-threading is disabled
  int dmrs_num_antennas_per_thread;
  // frame_parms
  /// Number of resource blocks (RB) in UL
  int N_RB_UL;
  /// Carrier offset in FFT buffer for first RE in PRB0
  uint16_t first_carrier_offset;
  /// Size of FFT
  uint16_t ofdm_symbol_size;
  /// Number of OFDM/SC-FDMA symbols in one slot
  uint16_t symbols_per_slot;
  /// Number of Receive antennas in node
  uint8_t nb_antennas_rx;
  /// Cyclic Prefix for DL (0=Normal CP, 1=Extended CP)
  lte_prefix_type_t Ncp;
  // pusch_pdu
  //BWP
  uint16_t bwp_size;
  uint16_t bwp_start;
  //pusch information always include
  uint8_t  qam_mod_order;
  uint8_t  transform_precoding;
  uint8_t  nrOfLayers;
  //DMRS
  uint16_t  ul_dmrs_symb_pos;
  uint8_t  dmrs_config_type;
  uint8_t  scid;
  uint8_t  num_dmrs_cdm_grps_no_data;
  uint16_t rb_start;
  uint16_t rb_size;
  //Resource Allocation in time domain
  uint8_t  start_symbol_index;
  uint8_t  nr_of_symbols;
  // pusch_pdu.dfts_ofdm
  uint8_t  low_papr_group_number;//Group number for Low PAPR sequence generation.
  uint16_t low_papr_sequence_number;//[TS38.211, sec 5.2.2] For DFT-S-OFDM.
} chestcomp_params_t;

chestcomp_params_t collect_channel_estimation_compensation_parameters(PHY_VARS_gNB *gNB, int frame, int slot, int beam_nb, int ulsch_id);

void set_channel_estimation_compensation_parameters(chestcomp_params_t params, PHY_VARS_gNB *gNB, int *frame, int *slot, int *beam_nb, int *ulsch_id);

#endif /* NR_ULSCH_H_ */
