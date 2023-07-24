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

/*! \file PHY/NR_UE_TRANSPORT
* \brief PHY level receive procedures for the Sidelink SSB S-SSS symbols
* \author M. Elkadi, D. Kim
* \date 2023
* \version 0.1
* \company EpiSci, Episys Science LLC
* \email: melissa@episci.com, d.kim@episci.com
* \note
* \warning
*/

#include <stdio.h>
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <nr-uesoftmodem.h>

#include "PHY/defs_nr_UE.h"
#include "PHY/phy_extern.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_proto_ue.h"
#include "PHY/NR_REFSIG/ss_pbch_nr.h"
#include "PHY/NR_REFSIG/sss_nr.h"


static const int16_t phase_re_nr_sl[PHASE_HYPOTHESIS_NUMBER]
    // -pi/3 ---- pi/3
    = {16384, 20173, 23571, 26509, 28932, 30791, 32051, 32687, 32687, 32051, 30791, 28932, 26509, 23571, 20173, 16384};

static const int16_t phase_im_nr_sl[PHASE_HYPOTHESIS_NUMBER] // -pi/3 ---- pi/3
    = {-28377, -25821, -22762, -19260, -15383, -11207, -6813, -2286, 2286, 6813, 11207, 15383, 19260, 22762, 25821, 28377};

static int pss_sss_sl_extract_nr(PHY_VARS_NR_UE *ue,
                                 UE_nr_rxtx_proc_t *proc,
                                 c16_t pss0_ext[NB_ANTENNAS_RX][SL_NR_PSS_SEQUENCE_LENGTH],
                                 c16_t sss0_ext[NB_ANTENNAS_RX][SL_NR_SSS_SEQUENCE_LENGTH],
                                 c16_t pss1_ext[NB_ANTENNAS_RX][SL_NR_PSS_SEQUENCE_LENGTH],
                                 c16_t sss1_ext[NB_ANTENNAS_RX][SL_NR_SSS_SEQUENCE_LENGTH],
                                 c16_t rxdataF[][ue->SL_UE_PHY_PARAMS.sl_frame_params.samples_per_slot_wCP])
{
  NR_DL_FRAME_PARMS *frame_parms = &ue->SL_UE_PHY_PARAMS.sl_frame_params;
  unsigned int ofdm_symbol_size = frame_parms->ofdm_symbol_size;

  c16_t *pss0_rxF, *pss0_rxF_ext;
  c16_t *pss1_rxF, *pss1_rxF_ext;
  c16_t *sss0_rxF, *sss0_rxF_ext;
  c16_t *sss1_rxF, *sss1_rxF_ext;
  for (uint8_t aarx = 0; aarx < frame_parms->nb_antennas_rx; aarx++) {
    pss0_rxF = &rxdataF[aarx][PSS0_SL_SYMBOL_NB * ofdm_symbol_size];
    pss1_rxF = &rxdataF[aarx][PSS1_SL_SYMBOL_NB * ofdm_symbol_size];
    sss0_rxF = &rxdataF[aarx][SSS0_SL_SYMBOL_NB * ofdm_symbol_size];
    sss1_rxF = &rxdataF[aarx][SSS1_SL_SYMBOL_NB * ofdm_symbol_size];
    pss0_rxF_ext = &pss0_ext[aarx][0];
    pss1_rxF_ext = &pss1_ext[aarx][0];
    sss0_rxF_ext = &sss0_ext[aarx][0];
    sss1_rxF_ext = &sss1_ext[aarx][0];
    unsigned int k = frame_parms->first_carrier_offset + frame_parms->ssb_start_subcarrier + PSS_SSS_SUB_CARRIER_START_SL;
    if (k >= ofdm_symbol_size)
      k-=ofdm_symbol_size;
    for (int i = 0; i < SL_NR_PSS_SEQUENCE_LENGTH; i++) {
      pss0_rxF_ext[i] = pss0_rxF[k];
      pss1_rxF_ext[i] = pss1_rxF[k];
      k++;
      if (k == ofdm_symbol_size)
        k = 0;
    }
    k = frame_parms->first_carrier_offset + frame_parms->ssb_start_subcarrier + PSS_SSS_SUB_CARRIER_START_SL;
    if (k >= ofdm_symbol_size)
      k-=ofdm_symbol_size;
    for (int i = 0; i < SL_NR_SSS_SEQUENCE_LENGTH; i++) {
      sss0_rxF_ext[i] = sss0_rxF[k];
      sss1_rxF_ext[i] = sss1_rxF[k];
      k++;
      if (k == ofdm_symbol_size)
        k = 0;
    }
  }

  return(0);
}

static int pss_sl_ch_est_nr(PHY_VARS_NR_UE *ue,
                     c16_t pss0_ext[NB_ANTENNAS_RX][SL_NR_PSS_SEQUENCE_LENGTH],
                     c16_t sss0_ext[NB_ANTENNAS_RX][SL_NR_SSS_SEQUENCE_LENGTH],
                     c16_t pss1_ext[NB_ANTENNAS_RX][SL_NR_PSS_SEQUENCE_LENGTH],
                     c16_t sss1_ext[NB_ANTENNAS_RX][SL_NR_SSS_SEQUENCE_LENGTH])
{
  int id = ue->common_vars.nid2_sl;
  c16_t *pss = (c16_t *)&ue->SL_UE_PHY_PARAMS.init_params.sl_pss_for_sync[id];
  c16_t tmp_pss0, tmp_pss1, tmp_sss0, tmp_sss1;
  for (uint8_t aarx = 0; aarx < ue->frame_parms.nb_antennas_rx; aarx++) {
    c16_t *pss0_ext_ptr = &pss0_ext[aarx][0];
    c16_t *pss1_ext_ptr = &pss1_ext[aarx][0];
    c16_t *sss0_ext_ptr = &sss0_ext[aarx][0];
    c16_t *sss1_ext_ptr = &sss1_ext[aarx][0];
    for (uint8_t i = 0; i < SL_NR_PSS_SEQUENCE_LENGTH; i++) {
      // This is H*(PSS) = R* \cdot PSS
      tmp_pss0.r = (int16_t)((((int32_t)pss0_ext_ptr[i].r) * pss[i].r)>>15);
      tmp_pss0.i = -pss0_ext_ptr[i].i * pss[i].i;
      tmp_pss1.r = pss1_ext_ptr[i].r * pss[i].r;
      tmp_pss1.i = -pss1_ext_ptr[i].i * pss[i].i;
      int32_t amp0 = (((int32_t)tmp_pss0.r) * tmp_pss0.r) + ((int32_t)tmp_pss0.i) * tmp_pss0.i;
      int shift_pss0 = log2_approx(amp0) / 2;
      int32_t amp1 = (((int32_t)tmp_pss1.r) * tmp_pss1.r) + ((int32_t)tmp_pss1.i) * tmp_pss1.i;
      int shift_pss1 = log2_approx(amp1) / 2;
      // This is R(SSS) \cdot H*(PSS)
      tmp_sss0.r = (int16_t)(((tmp_pss0.r * (int32_t)sss0_ext_ptr[i].r) >> shift_pss0) - ((tmp_pss0.i * (int32_t)sss0_ext_ptr[i].i >> shift_pss0)));
      tmp_sss0.i = (int16_t)(((tmp_pss0.r * (int32_t)sss0_ext_ptr[i].i) >> shift_pss0) + ((tmp_pss0.i * (int32_t)sss0_ext_ptr[i].r >> shift_pss0)));
      tmp_sss1.r = (int16_t)(((tmp_pss1.r * (int32_t)sss1_ext_ptr[i].r) >> shift_pss1) - ((tmp_pss1.i * (int32_t)sss1_ext_ptr[i].i >> shift_pss1)));
      tmp_sss1.i = (int16_t)(((tmp_pss1.r * (int32_t)sss1_ext_ptr[i].i) >> shift_pss1) + ((tmp_pss1.i * (int32_t)sss1_ext_ptr[i].r >> shift_pss1)));
      // MRC on RX antennas
      if (aarx==0) {
        sss0_ext_ptr[i].r = tmp_sss0.r;
        sss0_ext_ptr[i].i = tmp_sss0.i;
        sss1_ext_ptr[i].r = tmp_sss1.r;
        sss1_ext_ptr[i].i = tmp_sss1.i;
      } else {
        sss0_ext_ptr[i].r += tmp_sss0.r;
        sss0_ext_ptr[i].i += tmp_sss0.i;
        sss1_ext_ptr[i].r += tmp_sss1.r;
        sss1_ext_ptr[i].i += tmp_sss1.i;
      }
    }
  }
  return(0);
}

/*******************************************************************
*
* NAME :         rx_sss_sl_nr
*
* PARAMETERS :   none
*
* RETURN :       Set Nid_cell in ue context
*
* DESCRIPTION :  Determine element Nid1 of cell identity
*                so Nid_cell in ue context is set according to Nid1 & Nid2
*
*********************************************************************/

int rx_sss_sl_nr(PHY_VARS_NR_UE *ue,
                 UE_nr_rxtx_proc_t *proc,
                 int32_t *tot_metric,
                 uint8_t *phase_max,
                 int *freq_offset_sss,
                 c16_t rxdataF[][ue->SL_UE_PHY_PARAMS.sl_frame_params.samples_per_slot_wCP])
{
  c16_t pss0_ext[NB_ANTENNAS_RX][SL_NR_PSS_SEQUENCE_LENGTH];
  c16_t sss0_ext[NB_ANTENNAS_RX][SL_NR_SSS_SEQUENCE_LENGTH];
  c16_t pss1_ext[NB_ANTENNAS_RX][SL_NR_PSS_SEQUENCE_LENGTH];
  c16_t sss1_ext[NB_ANTENNAS_RX][SL_NR_SSS_SEQUENCE_LENGTH];
  NR_DL_FRAME_PARMS *frame_parms = &ue->SL_UE_PHY_PARAMS.sl_frame_params;

  pss_sss_sl_extract_nr(ue, proc, pss0_ext, sss0_ext, pss1_ext, sss1_ext, rxdataF);

#ifdef DEBUG_PLOT_SSS
  write_output("rxsig0.m","rxs0", &rxdataF[0][0] ,ue->SL_UE_PHY_PARAMS.sl_frame_params.samples_per_subframe, 1, 1);
  write_output("rxdataF0_pss.m","rxF0_pss", &rxdataF[0][0], SL_UE_PHY_PARAMS.sl_frame_params->ofdm_symbol_size, 1, 1);
  write_output("rxdataF0_sss.m","rxF0_sss", &rxdataF[0][(SSS_SYMBOL_NB-PSS_SYMBOL_NB) * SL_UE_PHY_PARAMS.sl_frame_params->ofdm_symbol_size],
                                            SL_UE_PHY_PARAMS.sl_frame_params->ofdm_symbol_size, 1, 1);
  write_output("pss0_ext.m","pss0_ext", pss0_ext, SL_NR_PSS_SEQUENCE_LENGTH, 1, 1);
  write_output("pss1_ext.m","pss1_ext", pss1_ext, SL_NR_PSS_SEQUENCE_LENGTH, 1, 1);
  write_output("sss0_ext.m","sss0_ext", sss0_ext, SL_NR_SSS_SEQUENCE_LENGTH, 1, 1);
  write_output("sss1_ext.m","sss1_ext", sss1_ext, SL_NR_SSS_SEQUENCE_LENGTH, 1, 1);
#endif

  // get conjugated channel estimate from PSS, H* = R* \cdot PSS
  // and do channel estimation and compensation based on PSS
  pss_sl_ch_est_nr(ue, pss0_ext, sss0_ext, pss1_ext, sss1_ext);

  /* now do the SSS detection based on the precomputed sequences in PHY/LTE_TRANSPORT/sss.
     for phase evaluation, one uses an array of possible phase shifts
     then a correlation is done between received signal with a shift pĥase and the reference signal
     Computation of signal with shift phase is based on below formula
     cosinus cos(x + y) = cos(x)cos(y) - sin(x)sin(y)
     sinus   sin(x + y) = sin(x)cos(y) + cos(x)sin(y) */
  uint16_t Nid1;
  uint8_t Nid2 = ue->common_vars.nid2_sl;
  *tot_metric = INT_MIN;
  c16_t *sss0 = &sss0_ext[0][0];
  c16_t *sss1 = &sss1_ext[0][0];
  int16_t *d;
  for (Nid1 = 0; Nid1 < SL_NR_NUM_SLSS_IDs; Nid1++) {          // all possible Nid1 values
    for (uint8_t phase = 0; phase < PHASE_HYPOTHESIS_NUMBER; phase++) {  // phase offset between PSS and SSS
      int32_t metric = 0;
      int32_t metric_re = 0;
      d = (int16_t *)&ue->SL_UE_PHY_PARAMS.init_params.sl_sss_for_sync[Nid2][Nid1];
      // This is the inner product using one particular value of each unknown parameter
      for (int i = 0; i < SL_NR_SSS_SEQUENCE_LENGTH; i++) {
        metric_re += d[i] * (((phase_re_nr_sl[phase] * sss0[i].r) >> SCALING_METRIC_SSS_NR) - ((phase_im_nr_sl[phase] * sss0[i].i) >> SCALING_METRIC_SSS_NR)) +
                     d[i] * (((phase_re_nr_sl[phase] * sss1[i].r) >> SCALING_METRIC_SSS_NR) - ((phase_im_nr_sl[phase] * sss1[i].i) >> SCALING_METRIC_SSS_NR));
      }
      metric = metric_re;
      if (metric > *tot_metric) {
        *tot_metric = metric;
        ue->frame_parms.Nid_cell = Nid1 + NUMBER_SSS_SEQUENCE * Nid2;
        *phase_max = phase;
#ifdef DEBUG_SSS_NR
        LOG_I(NR_PHY,"(phase, Nid1) (%d,%d), metric_phase = %d, tot_metric = %d, phase_max = %d \n", phase, Nid1, metric, *tot_metric, *phase_max);
#endif
      }
    }
  }

#define SSS_SL_METRIC_FLOOR_NR   (30000)
  if (*tot_metric > SSS_SL_METRIC_FLOOR_NR) {
    Nid2 = GET_NID2_SL(frame_parms->Nid_cell);
    Nid1 = GET_NID1_SL(frame_parms->Nid_cell);
  }
#if 1
  LOG_D(NR_PHY, "Nid2 %d Nid1 %d tot_metric %d, phase_max %d \n", Nid2, Nid1, *tot_metric, *phase_max);
#endif

  if (Nid1 == N_ID_1_NUMBER) {
    LOG_I(PHY,"Failled to detect SSS after PSS\n");
    return -1;
  }

  int re = 0;
  int im = 0;
  d = (int16_t *)&ue->SL_UE_PHY_PARAMS.init_params.sl_sss_for_sync[Nid2][Nid1];
  for(int i = 0; i < SL_NR_SSS_SEQUENCE_LENGTH; i++) {
    re += d[i] * sss0[i].r;
    im += d[i] * sss0[i].i;
    re += d[i] * sss1[i].r;
    im += d[i] * sss1[i].i;
  }

  double ffo_sss = atan2(im, re) / M_PI / 4.3;
  *freq_offset_sss = (int)(ffo_sss * frame_parms->subcarrier_spacing);
  double ffo_pss = ((double)ue->common_vars.freq_offset) / frame_parms->subcarrier_spacing;
  LOG_I(NR_PHY, "ffo_pss %f (%i Hz), ffo_sss %f (%i Hz), ffo_pss+ffo_sss %f (%i Hz)\n",
         ffo_pss, (int)(ffo_pss * frame_parms->subcarrier_spacing), ffo_sss, *freq_offset_sss, ffo_pss + ffo_sss,
         (int)((ffo_pss + ffo_sss) * frame_parms->subcarrier_spacing));

  return(0);
}

