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

/**********************************************************************
*
* FILENAME    :  sss_nr.c
*
* MODULE      :  Functions for secundary synchronisation signal
*
* DESCRIPTION :  generation of sss
*                3GPP TS 38.211 7.4.2.3 Secondary synchronisation signal
*
************************************************************************/

#include <stdio.h>
#include <assert.h>
#include <errno.h>

#include "PHY/defs_nr_UE.h"
#include "PHY/MODULATION/modulation_UE.h"
#include "executables/softmodem-common.h"
#include "PHY/NR_REFSIG/ss_pbch_nr.h"

#define DEFINE_VARIABLES_SSS_NR_H
#include "PHY/NR_REFSIG/sss_nr.h"
#undef DEFINE_VARIABLES_SSS_NR_H

/*******************************************************************
*
* NAME :         init_context_sss_nr
*
* PARAMETERS :   N_ID_2 : element 2 of physical layer cell identity
*                value : { 0, 1, 2 }
*
* RETURN :       generate binary sss sequence (this is a m-sequence)
*                d_sss is a third dimension array depending on
*                Cell identity elements:
*                - N_ID_1 : value from 0 to 335
*                - N_ID_2 : value from 0 to 2
*
* DESCRIPTION :  3GPP TS 38.211 7.4.2.3 Secundary synchronisation signal
*                Sequence generation
*
*********************************************************************/

void init_context_sss_nr(int amp)
{
  int16_t x0[LENGTH_SSS_NR];
  int16_t x1[LENGTH_SSS_NR];
  const int x0_initial[INITIAL_SSS_NR] = { 1, 0, 0, 0, 0, 0, 0 };
  const int x1_initial[INITIAL_SSS_NR] = { 1, 0, 0, 0, 0, 0, 0 };
  for (int i = 0; i < INITIAL_SSS_NR; i++) {
    x0[i] = x0_initial[i];
    x1[i] = x1_initial[i];
  }
  for (int i = 0; i < (LENGTH_SSS_NR - INITIAL_SSS_NR); i++) {
    x0[i + 7] = (x0[i + 4] + x0[i]) % (2);
    x1[i + 7] = (x1[i + 1] + x1[i]) % (2);
  }

  int nid_2_num = get_softmodem_params()->sl_mode == 0 ? N_ID_2_NUMBER : N_ID_2_NUMBER_SL;
  int dss_current;
  int m0, m1;
  for (int N_ID_2 = 0; N_ID_2 < nid_2_num; N_ID_2++) {
    for (int N_ID_1 = 0; N_ID_1 < N_ID_1_NUMBER; N_ID_1++) {
      m0 = 15 * (N_ID_1 / 112) + (5 * N_ID_2);
      m1 = N_ID_1 % 112;
      for (int n = 0; n < LENGTH_SSS_NR; n++) {
        dss_current = (1 - 2 * x0 [(n + m0) % (LENGTH_SSS_NR)]) * (1 - 2 * x1[(n + m1) % (LENGTH_SSS_NR)]);
      /* Modulation of SSS is a BPSK TS 36.211 chapter 5.1.2 BPSK */
        d_sss[N_ID_2][N_ID_1][n] = dss_current;
      }
    }
  }

}

/*******************************************************************
*
* NAME :         insert_sss_nr
*
* PARAMETERS :   pointer to input buffer for which sss in inserted
*                amp amplitude applied to input
*                frame parameters for cell identity
*
* RETURN :       none
*
* DESCRIPTION :  Allow to generate a reference sss sequence according to cell identity
*
*********************************************************************/

//#define DEBUG_SSS_NR
//#define DEBUG_PLOT_SSS
void insert_sss_nr(c16_t *sss_time,
                   NR_DL_FRAME_PARMS *frame_parms)
{
  unsigned int ofdm_symbol_size = frame_parms->ofdm_symbol_size;
  int Nid2 = GET_NID2(frame_parms->Nid_cell);
  int Nid1 = GET_NID1(frame_parms->Nid_cell);

  /* call of IDFT should be done with ordered input as below
    *
    *                n input samples
    *  <------------------------------------------------>
    *  0                                                n
    *  are written into input buffer for IFFT in this order
    *   -------------------------------------------------
    *  |xxxxxxx                       N/2       xxxxxxxx|
    *  --------------------------------------------------
    *  ^      ^                 ^               ^          ^
    *  |      |                 |               |          |
    * n/2    end of            n=0            start of    n/2-1
    *         sss                               sss
    *
    *                   Frequencies
    *      positives                   negatives
    * 0                 (+N/2)(-N/2)
    * |-----------------------><-------------------------|
    *
    * sample 0 is for continuous frequency which is not used here
    */

  unsigned int k = ofdm_symbol_size - ((LENGTH_SSS_NR/2)+1);

  /* SSS is directly mapped to subcarrier */
  c16_t in[sizeof(int16_t) * ofdm_symbol_size] __attribute__((aligned(32)));
  memset(in, 0, sizeof(in));
  for (int i = 0; i < LENGTH_SSS_NR; i++) {
    in[i].r = d_sss[Nid2][Nid1][i];
    in[i].i = 0;
    k++;
    if (k >= ofdm_symbol_size) {
      k++;
      k-=ofdm_symbol_size;
    }
  }

  /* get sss in the frequency domain by applying an inverse FFT */
  c16_t out[sizeof(int16_t) * ofdm_symbol_size] __attribute__((aligned(32)));
  memset(out, 0, sizeof(out));
  memset(sss_time, 0, sizeof(int16_t) * ofdm_symbol_size);
  idft(IDFT_2048, (int16_t *)&in, (int16_t *)&out, 1);
  for (unsigned int i = 0; i < ofdm_symbol_size; i++) {
    sss_time[i] = out[i];
  }
}

/*******************************************************************
*
* NAME :         pss_ch_est
*
* PARAMETERS :   none
*
* RETURN :       none
*
* DESCRIPTION :  pss channel estimation
*
*********************************************************************/

int pss_ch_est_nr(PHY_VARS_NR_UE *ue,
                  c16_t pss_ext[NB_ANTENNAS_RX][LENGTH_PSS_NR],
                  c16_t sss_ext[NB_ANTENNAS_RX][LENGTH_SSS_NR])
{
  int id = get_softmodem_params()->sl_mode == 0 ? ue->common_vars.eNb_id : ue->common_vars.N2_id;
  c16_t *pss = primary_synchro_nr2[id];
  c16_t tmp, tmp2;
  c16_t *sss_ext3 = &sss_ext[0][0];
  for (uint8_t aarx = 0; aarx < ue->frame_parms.nb_antennas_rx; aarx++) {
    c16_t *sss_ext2 = &sss_ext[aarx][0];
    c16_t *pss_ext2 = &pss_ext[aarx][0];
    int32_t amp;
    int shift;
    for (uint8_t i = 0; i < LENGTH_PSS_NR; i++) {
      tmp.r = pss_ext2[i].r * pss[i].r;
      tmp.i = -pss_ext2[i].i * pss[i].i;
      amp = (((int32_t)tmp.r) * tmp.r) + ((int32_t)tmp.i) * tmp.i;
      shift = log2_approx(amp) / 2;
      tmp2.r = (int16_t)(((tmp.r * (int32_t)sss_ext2[i].r) >> shift) - ((tmp.i * (int32_t)sss_ext2[i].i >> shift)));
      tmp2.i = (int16_t)(((tmp.r * (int32_t)sss_ext2[i].i) >> shift) + ((tmp.i * (int32_t)sss_ext2[i].r >> shift)));
      if (aarx == 0) {
        sss_ext3[i].r = tmp2.r;
        sss_ext3[i].i = tmp2.i;
      } else {
        sss_ext3[i].r += tmp2.r;
        sss_ext3[i].i += tmp2.i;
      }
    }
  }

  return(0);
}

/*******************************************************************
*
* NAME :         do_pss_sss_extract
*
* PARAMETERS :   none
*
* RETURN :       none
*
* DESCRIPTION : it allows extracting sss from samples buffer
*
*********************************************************************/

int do_pss_sss_extract_nr(PHY_VARS_NR_UE *ue,
                          UE_nr_rxtx_proc_t *proc,
                          c16_t pss_ext[NB_ANTENNAS_RX][LENGTH_PSS_NR],
                          c16_t sss_ext[NB_ANTENNAS_RX][LENGTH_SSS_NR],
                          uint8_t doPss, uint8_t doSss,
                          uint8_t subframe,
                          c16_t rxdataF[][ue->frame_parms.samples_per_slot_wCP]) // add flag to indicate extracting only PSS, only SSS, or both
{
  uint8_t aarx;
  c16_t *pss_rxF,*pss_rxF_ext;
  c16_t *sss_rxF,*sss_rxF_ext;
  uint8_t pss_symbol = 0;
  uint8_t sss_symbol = get_softmodem_params()->sl_mode == 0 ?
                       (SSS_SYMBOL_NB - PSS_SYMBOL_NB) :
                       (SSS0_SL_SYMBOL_NB - PSS0_SL_SYMBOL_NB) ;
  NR_DL_FRAME_PARMS *frame_parms = &ue->frame_parms;

  for (aarx=0; aarx<frame_parms->nb_antennas_rx; aarx++) {
    pss_rxF = &rxdataF[aarx][pss_symbol * frame_parms->ofdm_symbol_size];
    sss_rxF = &rxdataF[aarx][sss_symbol * frame_parms->ofdm_symbol_size];
    pss_rxF_ext = &pss_ext[aarx][0];
    sss_rxF_ext = &sss_ext[aarx][0];

    unsigned int k = frame_parms->first_carrier_offset +
                     frame_parms->ssb_start_subcarrier +
                     ((get_softmodem_params()->sl_mode == 0) ?
                     PSS_SSS_SUB_CARRIER_START :
                     PSS_SSS_SUB_CARRIER_START_SL);

    if (k>= frame_parms->ofdm_symbol_size) k-=frame_parms->ofdm_symbol_size;
    for (int i=0; i < LENGTH_PSS_NR; i++) {
      if (doPss) {
        pss_rxF_ext[i] = pss_rxF[k];
      }
      if (doSss) {
        sss_rxF_ext[i] = sss_rxF[k];
      }
      k++;
      if (k == frame_parms->ofdm_symbol_size) k = 0;
    }
  }

  return(0);
}

/*******************************************************************
*
* NAME :         rx_sss_nr
*
* PARAMETERS :   none
*
* RETURN :       Set Nid_cell in ue context
*
* DESCRIPTION :  Determine element Nid1 of cell identity
*                so Nid_cell in ue context is set according to Nid1 & Nid2
*
*********************************************************************/

int rx_sss_nr(PHY_VARS_NR_UE *ue,
              UE_nr_rxtx_proc_t *proc,
              int32_t *tot_metric,
              uint8_t *phase_max,
              int *freq_offset_sss,
              c16_t rxdataF[][ue->frame_parms.samples_per_slot_wCP])
{
  c16_t pss_ext[NB_ANTENNAS_RX][LENGTH_PSS_NR];
  c16_t sss_ext[NB_ANTENNAS_RX][LENGTH_SSS_NR];
  do_pss_sss_extract_nr(ue, proc, pss_ext, sss_ext, 1 /* doPss */, 1 /* doSss */, 0 /* subframe */, rxdataF);
  pss_ch_est_nr(ue, pss_ext, sss_ext);

  *tot_metric = INT_MIN;
  c16_t *sss = &sss_ext[0][0];
  uint16_t Nid1;
  uint8_t Nid2 = get_softmodem_params()->sl_mode == 0 ? ue->common_vars.eNb_id : ue->common_vars.N2_id;
  int16_t *d;
  for (Nid1 = 0; Nid1 < N_ID_1_NUMBER; Nid1++) {
    for (uint8_t phase = 0; phase < PHASE_HYPOTHESIS_NUMBER; phase++) {
      int32_t metric = 0;
      int32_t metric_re = 0;
      d = (int16_t *)&d_sss[Nid2][Nid1];
      for (uint8_t i = 0; i < LENGTH_SSS_NR; i++) {
        metric_re += d[i] * (((phase_re_nr[phase] * sss[i].r) >> SCALING_METRIC_SSS_NR) - ((phase_im_nr[phase] * sss[i].i) >> SCALING_METRIC_SSS_NR));
      }
      metric = metric_re;
      if (metric > *tot_metric) {
        *tot_metric = metric;
        ue->frame_parms.Nid_cell = Nid2+(3*Nid1);
        *phase_max = phase;

      }
    }
  }

  if (*tot_metric > SSS_METRIC_FLOOR_NR) {
    Nid2 = GET_NID2(ue->frame_parms.Nid_cell);
    Nid1 = GET_NID1(ue->frame_parms.Nid_cell);
    LOG_D(NR_PHY,"Nid2 %d Nid1 %d tot_metric %d, phase_max %d \n", Nid2, Nid1, *tot_metric, *phase_max);
  }
  if (Nid1 == N_ID_1_NUMBER) {
    LOG_I(NR_PHY,"Failed to detect SSS after PSS\n");
    return -1;
  }

  int re = 0;
  int im = 0;
  for(int i = 0; i < LENGTH_SSS_NR; i++) {
    re += d[i] * sss[i].r;
    im += d[i] * sss[i].i;
  }

  double ffo_sss = atan2(im,re)/M_PI/4.3;
  *freq_offset_sss = (int)(ffo_sss * ue->frame_parms.subcarrier_spacing);
  double ffo_pss = ((double)ue->common_vars.freq_offset)/ue->frame_parms.subcarrier_spacing;
  LOG_I(NR_PHY, "ffo_pss %f (%i Hz), ffo_sss %f (%i Hz),  ffo_pss+ffo_sss %f (%i Hz)\n",
         ffo_pss, (int)(ffo_pss*ue->frame_parms.subcarrier_spacing), ffo_sss, *freq_offset_sss, ffo_pss+ffo_sss, (int)((ffo_pss+ffo_sss)*ue->frame_parms.subcarrier_spacing));

  return(0);
}
