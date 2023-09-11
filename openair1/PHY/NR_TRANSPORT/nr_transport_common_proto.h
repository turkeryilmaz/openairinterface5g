
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

/*! \file PHY/NR_TRANSPORT/nr_transport_common_proto.h
* \brief Some support routines
* \author
* \date 2019
* \version 0.1
* \company Eurecom
* \email:
* \note
* \warning
*/

/** @addtogroup _PHY_TRANSPORT_
 * @{
 */

#ifndef __NR_TRANSPORT_COMMON_PROTO__H__
#define __NR_TRANSPORT_COMMON_PROTO__H__

#include "PHY/defs_nr_common.h"


void nr_group_sequence_hopping(pucch_GroupHopping_t PUCCH_GroupHopping,
                               uint32_t n_id,
                               uint8_t n_hop,
                               int nr_slot_tx,
                               uint8_t *u,
                               uint8_t *v);

double nr_cyclic_shift_hopping(uint32_t n_id,
                               uint8_t m0,
                               uint8_t mcs,
                               uint8_t lnormal,
                               uint8_t lprime,
                               int nr_slot_tx);

/** \brief Computes available bits G. */
uint32_t nr_get_G(uint16_t nb_rb, uint16_t nb_symb_sch, uint8_t nb_re_dmrs, uint16_t length_dmrs, uint8_t Qm, uint8_t Nl);

uint32_t nr_get_G_sl(uint16_t nb_rb, uint16_t nb_re_sci1, uint16_t nb_re_sci2, uint16_t nb_symb_sch,uint8_t nb_re_dmrs,uint16_t length_dmrs, uint8_t Qm, uint8_t Nl);

uint32_t nr_get_E(uint32_t G, uint8_t C, uint8_t Qm, uint8_t Nl, uint8_t r);

void compute_nr_prach_seq(uint8_t short_sequence,
                          uint8_t num_sequences,
                          uint8_t rootSequenceIndex,
                          uint32_t X_u[64][839]);

void nr_fill_du(uint16_t N_ZC,uint16_t *prach_root_sequence_map);

void init_nr_prach_tables(int N_ZC);

void nr_codeword_scrambling(uint8_t *in,
                            uint32_t size,
                            uint8_t q,
                            uint32_t Nid,
                            uint32_t n_RNTI,
                            uint32_t* out);

void nr_codeword_unscrambling(int16_t* llr, uint32_t size, uint8_t q, uint32_t Nid, uint32_t n_RNTI);

void nr_codeword_unscrambling_sl(int16_t* llr, uint32_t size, uint8_t SCI2_bits, uint32_t Nid, uint8_t Nl);

/*!
\brief This function implements the idft transform precoding in PUSCH
\param z Pointer to input in frequnecy domain, and it is also the output in time domain
\param Msc_PUSCH number of allocated data subcarriers
*/
void nr_idft(int32_t *z, uint32_t Msc_PUSCH);

/** \brief This function generates log-likelihood ratios (decoder input) for single-stream QPSK received waveforms.
    @param rxdataF_comp Compensated channel output
    @param ulsch_llr llr output
    @param nb_re number of REs for this allocation
    @param symbol OFDM symbol index in sub-frame
*/
void nr_ulsch_qpsk_llr(int32_t *rxdataF_comp,
                       int16_t *ulsch_llr,
                       uint32_t nb_re,
                       uint8_t  symbol);


/** \brief This function generates log-likelihood ratios (decoder input) for single-stream 16 QAM received waveforms.
    @param rxdataF_comp Compensated channel output
    @param ul_ch_mag uplink channel magnitude multiplied by the 1st amplitude threshold in QAM 16
    @param ulsch_llr llr output
    @param nb_re number of RBs for this allocation
    @param symbol OFDM symbol index in sub-frame
*/
void nr_ulsch_16qam_llr(int32_t *rxdataF_comp,
                        int32_t *ul_ch_mag,
                        int16_t  *ulsch_llr,
                        uint32_t nb_rb,
                        uint32_t nb_re,
                        uint8_t  symbol);


/** \brief This function generates log-likelihood ratios (decoder input) for single-stream 64 QAM received waveforms.
    @param rxdataF_comp Compensated channel output
    @param ul_ch_mag  uplink channel magnitude multiplied by the 1st amplitude threshold in QAM 64
    @param ul_ch_magb uplink channel magnitude multiplied by the 2bd amplitude threshold in QAM 64
    @param ulsch_llr llr output
    @param nb_re number of REs for this allocation
    @param symbol OFDM symbol index in sub-frame
*/
void nr_ulsch_64qam_llr(int32_t *rxdataF_comp,
                        int32_t *ul_ch_mag,
                        int32_t *ul_ch_magb,
                        int16_t  *ulsch_llr,
                        uint32_t nb_rb,
                        uint32_t nb_re,
                        uint8_t  symbol);

void nr_ulsch_det_HhH(int32_t *after_mf_00,//a
                int32_t *after_mf_01,//b
                int32_t *after_mf_10,//c
                int32_t *after_mf_11,//d
                int32_t *det_fin,//1/ad-bc
                unsigned short nb_rb,
                unsigned char symbol,
                int32_t shift);

__m128i nr_ulsch_inv_comp_muli(__m128i input_x,
                         __m128i input_y);

void nr_ulsch_conjch0_mult_ch1(int *ch0,
                         int *ch1,
                         int32_t *ch0conj_ch1,
                         unsigned short nb_rb,
                         unsigned char output_shift0);

__m128i nr_ulsch_comp_muli_sum(__m128i input_x,
                         __m128i input_y,
                         __m128i input_w,
                         __m128i input_z,
                         __m128i det);

void nr_ulsch_construct_HhH_elements(int *conjch00_ch00,
                               int *conjch01_ch01,
                               int *conjch11_ch11,
                               int *conjch10_ch10,//
                               int *conjch20_ch20,
                               int *conjch21_ch21,
                               int *conjch30_ch30,
                               int *conjch31_ch31,
                               int *conjch00_ch01,//00_01
                               int *conjch01_ch00,//01_00
                               int *conjch10_ch11,//10_11
                               int *conjch11_ch10,//11_10
                               int *conjch20_ch21,
                               int *conjch21_ch20,
                               int *conjch30_ch31,
                               int *conjch31_ch30,
                               int32_t *after_mf_00,
                               int32_t *after_mf_01,
                               int32_t *after_mf_10,
                               int32_t *after_mf_11,
                               unsigned short nb_rb,
                               unsigned char symbol);


/**@}*/

void init_pucch2_luts(void);
#endif
