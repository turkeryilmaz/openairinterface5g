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


int nr_sl_generate_sss(int32_t *txdataF,
                       int16_t amp,
                       uint8_t ssb_start_symbol,
                       NR_DL_FRAME_PARMS *frame_parms)
{
  int16_t x0[NR_SSS_LENGTH];
  int16_t x1[NR_SSS_LENGTH];
  const int x0_initial[7] = { 1, 0, 0, 0, 0, 0, 0 };
  const int x1_initial[7] = { 1, 0, 0, 0, 0, 0, 0 };

  /// Sequence generation
  int Nid = frame_parms->Nid_SL;
  int Nid1 = Nid % NUMBER_SSS_SEQUENCE;
  int Nid2 = Nid / NUMBER_SSS_SEQUENCE;

  for (int i=0; i < 7; i++) {
    x0[i] = x0_initial[i];
    x1[i] = x1_initial[i];
  }

  for (int i=0; i < NR_SSS_LENGTH - 7; i++) {
    x0[i+7] = (x0[i + 4] + x0[i]) % 2;
    x1[i+7] = (x1[i + 1] + x1[i]) % 2;
  }

  int m0 = 15 * (Nid1 / 112) + (5 * Nid2);
  int m1 = Nid1 % 112;

#ifdef NR_SSS_DEBUG
  write_output("d_sss.m", "d_sss", (void*)d_sss, NR_SSS_LENGTH, 1, 1);
#endif

  /// Resource mapping

  // SSS occupies a predefined position (subcarriers 2-129, symbol 3) within the SSB block starting from
  int k = frame_parms->first_carrier_offset + frame_parms->ssb_start_subcarrier + 2; //and
  int l = ssb_start_symbol + 3;

  for (int i = 0; i < NR_SSS_LENGTH; i++) {
    int16_t d_sss = (1 - 2 * x0[(i + m0) % NR_SSS_LENGTH] ) * (1 - 2 * x1[(i + m1) % NR_SSS_LENGTH] ) * 23170;
    ((int16_t*)txdataF)[2 * (l * frame_parms->ofdm_symbol_size + k)] = (((int16_t)amp) * d_sss) >> 15;
    k++;

    if (k >= frame_parms->ofdm_symbol_size)
      k-=frame_parms->ofdm_symbol_size;
  }

  // SSS occupies a predefined position (subcarriers 2 to 129, symbol 4) within the SSB block starting from
  k = frame_parms->first_carrier_offset + frame_parms->ssb_start_subcarrier + 2; //and
  l = ssb_start_symbol + 4;

  for (int i = 0; i < NR_SSS_LENGTH; i++) {
    int16_t d_sss = (1 - 2*x0[(i + m0) % NR_SSS_LENGTH] ) * (1 - 2*x1[(i + m1) % NR_SSS_LENGTH] ) * 23170;
    ((int16_t*)txdataF)[2*(l*frame_parms->ofdm_symbol_size + k)] = (((int16_t)amp) * d_sss) >> 15;
    k++;

    if (k >= frame_parms->ofdm_symbol_size)
      k-=frame_parms->ofdm_symbol_size;
  }
#ifdef NR_SSS_DEBUG
  //  write_output("sss_0.m", "sss_0", (void*)txdataF[0][l*frame_parms->ofdm_symbol_size], frame_parms->ofdm_symbol_size, 1, 1);
#endif

  return 0;
}

/*******************************************************************
*
* NAME :         pss_sss_sl_extract_nr
*
* PARAMETERS :   none
*
* RETURN :       none
*
* DESCRIPTION :
*
*********************************************************************/

static int pss_sss_sl_extract_nr(PHY_VARS_NR_UE *ue,
                                 UE_nr_rxtx_proc_t *proc,
                                 int32_t pss0_ext[NB_ANTENNAS_RX][LENGTH_PSS_NR],
                                 int32_t sss0_ext[NB_ANTENNAS_RX][LENGTH_SSS_NR],
                                 int32_t pss1_ext[NB_ANTENNAS_RX][LENGTH_PSS_NR],
                                 int32_t sss1_ext[NB_ANTENNAS_RX][LENGTH_SSS_NR])
{
  int32_t **rxdataF  =  ue->common_vars.common_vars_rx_data_per_thread[proc->thread_id].rxdataF;
  NR_DL_FRAME_PARMS *frame_parms = &ue->frame_parms;
  unsigned int ofdm_symbol_size = frame_parms->ofdm_symbol_size;

  int32_t *pss0_rxF, *pss0_rxF_ext;
  int32_t *sss0_rxF, *sss0_rxF_ext;
  for (uint8_t aarx = 0; aarx < frame_parms->nb_antennas_rx; aarx++) {
    pss0_rxF  =  &rxdataF[aarx][PSS0_SL_SYMBOL_NB * ofdm_symbol_size];
    sss0_rxF  =  &rxdataF[aarx][SSS0_SL_SYMBOL_NB * ofdm_symbol_size];
    pss0_rxF_ext = &pss0_ext[aarx][0];
    sss0_rxF_ext = &sss0_ext[aarx][0];
    unsigned int k = frame_parms->first_carrier_offset + frame_parms->ssb_start_subcarrier + SPSS_SSSS_SUB_CARRIER_START;
    if (k >= frame_parms->ofdm_symbol_size)
      k-=frame_parms->ofdm_symbol_size;

    for (int i = 0; i < LENGTH_PSS_NR; i++) {
      pss0_rxF_ext[i] = pss0_rxF[k];
      sss0_rxF_ext[i] = sss0_rxF[k];
      k++;
      if (k == ofdm_symbol_size) k=0;
    }
  }

  int32_t *pss1_rxF, *pss1_rxF_ext;
  int32_t *sss1_rxF, *sss1_rxF_ext;
  for (uint8_t aarx = 0; aarx < frame_parms->nb_antennas_rx; aarx++) {
    rxdataF  =  ue->common_vars.common_vars_rx_data_per_thread[proc->thread_id].rxdataF;
    pss1_rxF  =  &rxdataF[aarx][PSS1_SL_SYMBOL_NB * ofdm_symbol_size];
    sss1_rxF  =  &rxdataF[aarx][SSS1_SL_SYMBOL_NB * ofdm_symbol_size];
    pss1_rxF_ext = &pss1_ext[aarx][0];
    sss1_rxF_ext = &sss1_ext[aarx][0];
    unsigned int k = frame_parms->first_carrier_offset + frame_parms->ssb_start_subcarrier + SPSS_SSSS_SUB_CARRIER_START;
    if (k >= frame_parms->ofdm_symbol_size)
      k-=frame_parms->ofdm_symbol_size;

    for (int i=0; i < LENGTH_PSS_NR; i++) {
      pss1_rxF_ext[i] = pss1_rxF[k];
      sss1_rxF_ext[i] = sss1_rxF[k];
      k++;
      if (k == ofdm_symbol_size) k=0;
    }
  }

  return(0);
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

int pss_sl_ch_est_nr(PHY_VARS_NR_UE *ue,
                     int32_t pss0_ext[NB_ANTENNAS_RX][LENGTH_PSS_NR],
                     int32_t sss0_ext[NB_ANTENNAS_RX][LENGTH_SSS_NR],
                     int32_t pss1_ext[NB_ANTENNAS_RX][LENGTH_PSS_NR],
                     int32_t sss1_ext[NB_ANTENNAS_RX][LENGTH_SSS_NR])
{
  int16_t *pss0_ext2, *sss0_ext2, *pss1_ext2, *sss1_ext2;
  int16_t tmp_re, tmp_im, tmp_re2, tmp_im2;
  NR_DL_FRAME_PARMS *frame_parms = &ue->frame_parms;
  int16_t *pss = primary_synchro_nr2[ue->common_vars.N2_id];
  int16_t *sss0_ext3 = (int16_t*)&sss0_ext[0][0];

  for (uint8_t aarx = 0; aarx < frame_parms->nb_antennas_rx; aarx++) {
    sss0_ext2 = (int16_t*)&sss0_ext[aarx][0];
    pss0_ext2 = (int16_t*)&pss0_ext[aarx][0];
    for (uint8_t i = 0; i < LENGTH_PSS_NR; i++) {
      // This is H*(PSS) = R* \cdot PSS
      tmp_re = pss0_ext2[i*2] * pss[i];
      tmp_im = -pss0_ext2[i*2+1] * pss[i];
      int32_t amp = (((int32_t)tmp_re)*tmp_re) + ((int32_t)tmp_im)*tmp_im;
      int shift = log2_approx(amp)/2;
      // This is R(SSS) \cdot H*(PSS)
      tmp_re2 = (int16_t)(((tmp_re * (int32_t)sss0_ext2[i*2])>>shift)    - ((tmp_im * (int32_t)sss0_ext2[i*2+1]>>shift)));
      tmp_im2 = (int16_t)(((tmp_re * (int32_t)sss0_ext2[i*2+1])>>shift)  + ((tmp_im * (int32_t)sss0_ext2[i*2]>>shift)));
      // MRC on RX antennas
      if (aarx==0) {
        sss0_ext3[i<<1]      = tmp_re2;
        sss0_ext3[1+(i<<1)]  = tmp_im2;
      } else {
        sss0_ext3[i<<1]      += tmp_re2;
        sss0_ext3[1+(i<<1)]  += tmp_im2;
      }
    }
  }

  int16_t *sss1_ext3 = (int16_t*)&sss1_ext[0][0];
  for (uint8_t aarx = 0; aarx < frame_parms->nb_antennas_rx; aarx++) {
    sss1_ext2 = (int16_t*)&sss1_ext[aarx][0];
    pss1_ext2 = (int16_t*)&pss1_ext[aarx][0];
    for (uint8_t i = 0; i < LENGTH_PSS_NR; i++) {
      // This is H*(PSS) = R* \cdot PSS
      tmp_re = pss1_ext2[i*2] * pss[i];
      tmp_im = -pss1_ext2[i*2+1] * pss[i];
      int32_t amp = (((int32_t)tmp_re)*tmp_re) + ((int32_t)tmp_im)*tmp_im;
      int shift = log2_approx(amp)/2;
      // This is R(SSS) \cdot H*(PSS)
      tmp_re2 = (int16_t)(((tmp_re * (int32_t)sss1_ext2[i*2])>>shift)    - ((tmp_im * (int32_t)sss1_ext2[i*2+1]>>shift)));
      tmp_im2 = (int16_t)(((tmp_re * (int32_t)sss1_ext2[i*2+1])>>shift)  + ((tmp_im * (int32_t)sss1_ext2[i*2]>>shift)));

      // MRC on RX antennas
      if (aarx == 0) {
        sss1_ext3[i<<1]      = tmp_re2;
        sss1_ext3[1+(i<<1)]  = tmp_im2;
      } else {
        sss1_ext3[i<<1]      += tmp_re2;
        sss1_ext3[1+(i<<1)]  += tmp_im2;
      }
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

int rx_sss_sl_nr(PHY_VARS_NR_UE *ue, UE_nr_rxtx_proc_t *proc, int32_t *tot_metric, uint8_t *phase_max, int *freq_offset_sss)
{
  int32_t pss0_ext[NB_ANTENNAS_RX][LENGTH_PSS_NR];
  int32_t sss0_ext[NB_ANTENNAS_RX][LENGTH_SSS_NR];
  int32_t pss1_ext[NB_ANTENNAS_RX][LENGTH_PSS_NR];
  int32_t sss1_ext[NB_ANTENNAS_RX][LENGTH_SSS_NR];
  NR_DL_FRAME_PARMS *frame_parms = &ue->frame_parms;

  pss_sss_sl_extract_nr(ue, proc, pss0_ext, sss0_ext, pss1_ext, sss1_ext);

#ifdef DEBUG_PLOT_SSS
  write_output("rxsig0.m","rxs0",&ue->common_vars.rxdata[0][0],ue->frame_parms.samples_per_subframe,1,1);
  write_output("rxdataF0_pss.m","rxF0_pss",&ue->common_vars.common_vars_rx_data_per_thread[proc->thread_id].rxdataF[0][0],frame_parms->ofdm_symbol_size,1,1);
  write_output("rxdataF0_sss.m","rxF0_sss",&ue->common_vars.common_vars_rx_data_per_thread[proc->thread_id].rxdataF[0][(SSS_SYMBOL_NB-PSS_SYMBOL_NB)*frame_parms->ofdm_symbol_size],frame_parms->ofdm_symbol_size,1,1);
  write_output("pss0_ext.m","pss0_ext",pss0_ext,LENGTH_PSS_NR,1,1);
  write_output("pss1_ext.m","pss1_ext",pss1_ext,LENGTH_PSS_NR,1,1);
  write_output("sss0_ext.m","sss0_ext",sss0_ext,LENGTH_PSS_NR,1,1);
  write_output("sss1_ext.m","sss1_ext",sss1_ext,LENGTH_PSS_NR,1,1);
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
  uint8_t Nid2 = GET_NID2_SL(frame_parms->Nid_SL);
  *tot_metric = INT_MIN;
  int16_t *sss0 = (int16_t*)&sss0_ext[0][0];
  int16_t *sss1 = (int16_t*)&sss1_ext[0][0];
  int32_t metric = 0;
  int32_t metric_re = 0;
  int16_t *d;
  for (Nid1 = 0; Nid1 < N_ID_1_NUMBER; Nid1++) {          // all possible Nid1 values
    for (uint8_t phase = 0; phase < PHASE_HYPOTHESIS_NUMBER; phase++) {  // phase offset between PSS and SSS
      metric = 0;
      metric_re = 0;
      d = (int16_t *)&d_sss[Nid2][Nid1];
      // This is the inner product using one particular value of each unknown parameter
      for (int i = 0; i < LENGTH_SSS_NR; i++) {
        metric_re += d[i] * (((phase_re_nr[phase] * sss0[2 *i ]) >> SCALING_METRIC_SSS_NR) - ((phase_im_nr[phase] * sss0[2 * i + 1]) >> SCALING_METRIC_SSS_NR)) +
                     d[i] * (((phase_re_nr[phase] * sss1[2 *i ]) >> SCALING_METRIC_SSS_NR) - ((phase_im_nr[phase] * sss1[2 * i + 1]) >> SCALING_METRIC_SSS_NR));
      }
      metric = metric_re;
      if (metric > *tot_metric) {
        *tot_metric = metric;
        ue->frame_parms.Nid_SL = Nid1 + NUMBER_SSS_SEQUENCE * Nid2;
        *phase_max = phase;
#ifdef DEBUG_SSS_NR
        printf("(phase, Nid1) (%d,%d), metric_phase = %d, tot_metric = %d, phase_max = %d \n", phase, Nid1, metric, *tot_metric, *phase_max);
#endif
      }
    }
  }

#define SSS_SL_METRIC_FLOOR_NR   (30000)
  if (*tot_metric > SSS_SL_METRIC_FLOOR_NR) {
    Nid2 = GET_NID2_SL(frame_parms->Nid_SL);
    Nid1 = GET_NID1_SL(frame_parms->Nid_SL);
  }
#if 1
  printf("Nid2 %d Nid1 %d tot_metric %d, phase_max %d \n", Nid2, Nid1, *tot_metric, *phase_max);
#endif

  if (Nid1 == N_ID_1_NUMBER) {
    LOG_I(PHY,"Failled to detect SSS after PSS\n");
    return -1;
  }

  int re = 0;
  int im = 0;
  d = (int16_t *)&d_sss[Nid2][Nid1];
  for(int i = 0; i < LENGTH_SSS_NR; i++) {
    re += d[i] * sss0[2 * i];
    im += d[i] * sss0[2 * i + 1];
    re += d[i] * sss1[2 * i];
    im += d[i] * sss1[2 * i + 1];
  }

  double ffo_sss = atan2(im, re) / M_PI / 4.3;
  *freq_offset_sss = (int)(ffo_sss * frame_parms->subcarrier_spacing);
  double ffo_pss = ((double)ue->common_vars.freq_offset) / frame_parms->subcarrier_spacing;
  LOG_I(NR_PHY, "ffo_pss %f (%i Hz), ffo_sss %f (%i Hz), ffo_pss+ffo_sss %f (%i Hz)\n",
         ffo_pss, (int)(ffo_pss * frame_parms->subcarrier_spacing), ffo_sss, *freq_offset_sss, ffo_pss + ffo_sss,
         (int)((ffo_pss + ffo_sss) * frame_parms->subcarrier_spacing));

  return(0);
}

