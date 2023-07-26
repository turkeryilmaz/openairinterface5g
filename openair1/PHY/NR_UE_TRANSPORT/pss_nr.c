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
* FILENAME    :  pss_nr.c
*
* MODULE      :  synchronisation signal
*
* DESCRIPTION :  generation of pss
*                3GPP TS 38.211 7.4.2.2 Primary synchronisation signal
*
************************************************************************/

#include <stdio.h>
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <nr-uesoftmodem.h>

#include "PHY/defs_nr_UE.h"

#include "PHY/NR_REFSIG/ss_pbch_nr.h"
#include "common/utils/LOG/vcd_signal_dumper.h"

#define DEFINE_VARIABLES_PSS_NR_H
#include "PHY/NR_REFSIG/pss_nr.h"
#undef DEFINE_VARIABLES_PSS_NR_H

#include "PHY/NR_REFSIG/sss_nr.h"
#include "PHY/NR_UE_TRANSPORT/cic_filter_nr.h"

//#define DBG_PSS_NR

/*******************************************************************
*
* NAME :         generate_pss_nr
*
* PARAMETERS :   N_ID_2 : element 2 of physical layer cell identity
*                value : { 0, 1, 2}
*
* RETURN :       generate binary pss sequence (this is a m-sequence)
*
* DESCRIPTION :  3GPP TS 38.211 7.4.2.2 Primary synchronisation signal
*                Sequence generation
*
*********************************************************************/

void generate_pss_nr(NR_DL_FRAME_PARMS *fp, int N_ID_2, int pss_seq_offset)
{
  AssertFatal(fp->ofdm_symbol_size > 127,"Illegal ofdm_symbol_size %d\n",fp->ofdm_symbol_size);
  AssertFatal(N_ID_2>=0 && N_ID_2 <=2,"Illegal N_ID_2 %d\n",N_ID_2);
  const int x_initial[INITIAL_PSS_NR] = {0, 1, 1, 0, 1, 1, 1};
  int16_t x[LENGTH_PSS_NR];
  for (int i = 0; i < INITIAL_PSS_NR; i++)
    x[i] = x_initial[i];

  for (int i = 0; i < (LENGTH_PSS_NR - INITIAL_PSS_NR); i++)
    x[i + INITIAL_PSS_NR] = (x[i + 4] + x[i]) % (2);

  int16_t d_pss[LENGTH_PSS_NR];
  for (int n = 0; n < LENGTH_PSS_NR; n++) {
    int m = (n + 43 * N_ID_2) % (LENGTH_PSS_NR);
    d_pss[n] = 1 - 2 * x[m];
  }

  c16_t *primary_synchro = primary_synchro_nr[N_ID_2];
  c16_t *primary_synchro2 = primary_synchro_nr2[N_ID_2];
  for (int i = 0; i < LENGTH_PSS_NR; i++) {
    primary_synchro[i].r = (d_pss[i] * SHRT_MAX) >> SCALING_PSS_NR;
    primary_synchro[i].i = 0;
    primary_synchro2[i].r = d_pss[i];
    primary_synchro2[i].i = d_pss[i];
  }

  /* call of IDFT should be done with ordered input as below
  *
  *                n input samples
  *  <------------------------------------------------>
  *  0                                                n
  *  are written into input buffer for IFFT
  *   -------------------------------------------------
  *  |xxxxxxx                       N/2       xxxxxxxx|
  *  --------------------------------------------------
  *  ^      ^                 ^               ^          ^
  *  |      |                 |               |          |
  * n/2    end of            n=0            start of    n/2-1
  *         pss                               pss
  *
  *                   Frequencies
  *      positives                   negatives
  * 0                 (+N/2)(-N/2)
  * |-----------------------><-------------------------|
  *
  * sample 0 is for continuous frequency which is used here
  */

  unsigned int subcarrier_start = get_softmodem_params()->sl_mode == 0 ? PSS_SSS_SUB_CARRIER_START : PSS_SSS_SUB_CARRIER_START_SL;
  unsigned int  k = fp->first_carrier_offset + fp->ssb_start_subcarrier + subcarrier_start;
  if (k>= fp->ofdm_symbol_size) k-=fp->ofdm_symbol_size;

  LOG_I(NR_PHY,"generate_pss_nr: subcarrier_start %d, k %d\n",subcarrier_start,k);
  c16_t in[sizeof(int16_t) * fp->ofdm_symbol_size] __attribute__((aligned(32)));
  memset(in, 0, sizeof(in));
  for (int i = 0; i < LENGTH_PSS_NR; i++) {
    in[k]= primary_synchro[i];
    k++;
    if (k == fp->ofdm_symbol_size) k = 0;
  }

  c16_t out[sizeof(int16_t) * fp->ofdm_symbol_size] __attribute__((aligned(32)));
  memset(out, 0, sizeof(out));
  memset(primary_synchro_time_nr[N_ID_2], 0, sizeof(int16_t) * fp->ofdm_symbol_size);
  idft((int16_t)get_idft(fp->ofdm_symbol_size), (int16_t *)in, (int16_t *)out, 1);
  for (unsigned int i = 0; i < fp->ofdm_symbol_size; i++) {
    primary_synchro_time_nr[N_ID_2][i] = out[i];
  }

}

/*******************************************************************
*
* NAME :         init_context_pss_nr
*
* PARAMETERS :   structure NR_DL_FRAME_PARMS give frame parameters
*
* RETURN :       generate binary pss sequences (this is a m-sequence)
*
* DESCRIPTION :  3GPP TS 38.211 7.4.2.2 Primary synchronisation signal
*                Sequence generation
*
*********************************************************************/

void init_context_pss_nr(NR_DL_FRAME_PARMS *frame_parms_ue)
{

  LOG_I(NR_PHY,"Calling init_context_pss_nr\n");	
  AssertFatal(frame_parms_ue->ofdm_symbol_size > 127, "illegal frame_parms_ue->ofdm_symbol_size %d\n",
              frame_parms_ue->ofdm_symbol_size);
  c16_t *p = NULL;
  int pss_sequence = get_softmodem_params()->sl_mode == 0 ? NUMBER_PSS_SEQUENCE : NUMBER_PSS_SEQUENCE_SL;
  c16_t txdataF[3*4096];
  for (int i = 0; i < pss_sequence; i++) { 
    LOG_I(NR_PHY,"Initializing PSS %d\n",i);
    p = malloc(LENGTH_PSS_NR * sizeof(c16_t));
    if (p != NULL) {
      primary_synchro_nr[i] = p;
      bzero(primary_synchro_nr[i], LENGTH_PSS_NR * sizeof(c16_t));
    } else {
      LOG_E(PHY,"Fatal memory allocation problem \n");
      assert(0);
    }
    p = malloc(LENGTH_PSS_NR * sizeof(c16_t));
    if (p != NULL) {
      primary_synchro_nr2[i] = p;
      bzero(primary_synchro_nr2[i], LENGTH_PSS_NR * sizeof(c16_t));
    } else {
      LOG_E(PHY,"Fatal memory allocation problem \n");
      assert(0);
    }
    p = malloc(sizeof(c16_t) * frame_parms_ue->ofdm_symbol_size);
    if (p != NULL) {
      primary_synchro_time_nr[i] = p;
      bzero(primary_synchro_time_nr[i], sizeof(c16_t) * frame_parms_ue->ofdm_symbol_size);
    } else {
      LOG_E(PHY,"Fatal memory allocation problem \n");
     assert(0);
    }

    if (get_softmodem_params()->sl_mode==0 || get_softmodem_params()->sync_ref) generate_pss_nr(frame_parms_ue, i, pss_sequence);
    else {
	int tmp=frame_parms_ue->Nid_SL;
	frame_parms_ue->Nid_SL = 336*i;
	nr_sl_generate_pss(txdataF,AMP,0,frame_parms_ue);
	frame_parms_ue->Nid_SL = tmp;
    }
  }
}

/*******************************************************************
*
* NAME :         init_context_synchro_nr
*
* PARAMETERS :   none
*
* RETURN :       generate context for pss and sss
*
* DESCRIPTION :  initialise contexts and buffers for synchronisation
*
*********************************************************************/

void init_context_synchro_nr(NR_DL_FRAME_PARMS *frame_parms_ue)
{
  init_context_pss_nr(frame_parms_ue);
  init_context_sss_nr(AMP);
}

/*******************************************************************
*
* NAME :         free_context_synchro_nr
*
* PARAMETERS :   none
*
* RETURN :       free context for pss and sss
*
* DESCRIPTION :  deallocate memory of synchronisation
*
*********************************************************************/

void free_context_synchro_nr(void)
{
  for (int i = 0; i < NUMBER_PSS_SEQUENCE; i++) {
    free_and_zero(primary_synchro_nr[i]);
    free_and_zero(primary_synchro_nr2[i]);
    free_and_zero(primary_synchro_time_nr[i]);
  }
}

/*******************************************************************
*
* NAME :         set_frame_context_pss_nr
*
* PARAMETERS :   configuration for UE with new FFT size
*
* RETURN :       0 if OK else error
*
* DESCRIPTION :  initialisation of UE contexts
*
*********************************************************************/

void set_frame_context_pss_nr(NR_DL_FRAME_PARMS *frame_parms_ue, int rate_change)
{
  /* set new value according to rate_change */
  frame_parms_ue->ofdm_symbol_size = (frame_parms_ue->ofdm_symbol_size / rate_change);
  frame_parms_ue->samples_per_subframe = (frame_parms_ue->samples_per_subframe / rate_change);
  init_context_pss_nr(frame_parms_ue);
  init_context_sss_nr(AMP);
}


/********************************************************************
*
* NAME :         decimation_synchro_nr
*
* INPUT :        UE context
*                for first and second pss sequence
*                - position of pss in the received UE buffer
*                - number of pss sequence
*
* RETURN :      0 if OK else error
*
* DESCRIPTION :  detect pss sequences in the received UE buffer
*
********************************************************************/

void decimation_synchro_nr(PHY_VARS_NR_UE *PHY_vars_UE, int rate_change, int **rxdata)
{
  NR_DL_FRAME_PARMS *frame_parms = &(PHY_vars_UE->frame_parms);
  int samples_for_frame = 2*frame_parms->samples_per_frame;

#if TEST_SYNCHRO_TIMING_PSS

  opp_enabled = 1;

  start_meas(&generic_time[TIME_RATE_CHANGE]);

#endif

/* build with cic filter does not work properly. Performances are significantly deteriorated */
#ifdef CIC_DECIMATOR

  cic_decimator((int16_t *)&(PHY_vars_UE->common_vars.rxdata[0][0]), (int16_t *)&(rxdata[0][0]),
                            samples_for_frame, rate_change, CIC_FILTER_STAGE_NUMBER, 0, FIR_RATE_CHANGE);
#else

  fir_decimator((int16_t *)&(PHY_vars_UE->common_vars.rxdata[0][0]), (int16_t *)&(rxdata[0][0]),
                            samples_for_frame, rate_change, 0);

#endif

  set_frame_context_pss_nr(frame_parms, rate_change);

#if TEST_SYNCHRO_TIMING_PSS

  stop_meas(&generic_time[TIME_RATE_CHANGE]);

  printf("Rate change execution duration %5.2f \n", generic_time[TIME_RATE_CHANGE].p_time/(cpuf*1000.0));

#endif
}

/*******************************************************************
*
* NAME :         pss_synchro_nr
*
* PARAMETERS :   int rate_change
*
* RETURN :       position of detected pss
*
* DESCRIPTION :  pss search can be done with sampling decimation.*
*
*********************************************************************/

int pss_synchro_nr(PHY_VARS_NR_UE *PHY_vars_UE, int is, int rate_change)
{
  NR_DL_FRAME_PARMS *frame_parms = &(PHY_vars_UE->frame_parms);
  int synchro_position;
  c16_t **rxdata = NULL;
  int fo_flag = PHY_vars_UE->UE_fo_compensation;  // flag to enable freq offset estimation and compensation

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PSS_SYNCHRO_NR, VCD_FUNCTION_IN);
#ifdef DBG_PSS_NR

  LOG_M("rxdata0_rand.m","rxd0_rand", &PHY_vars_UE->common_vars.rxdata[0][0], frame_parms->samples_per_frame, 1, 1);

#endif

  if (rate_change != 1) {

    rxdata = (c16_t**)malloc16(frame_parms->nb_antennas_rx * sizeof(c16_t*));

    for (int aa = 0; aa < frame_parms->nb_antennas_rx; aa++) {
      rxdata[aa] = (c16_t*)malloc16_clear((frame_parms->samples_per_frame + 8192) * sizeof(c16_t));
    }
#ifdef SYNCHRO_DECIMAT

    decimation_synchro_nr(PHY_vars_UE, rate_change, rxdata);

#endif
  }
  else {

    rxdata = PHY_vars_UE->common_vars.rxdata;
  }

#ifdef DBG_PSS_NR

  LOG_M("rxdata0_des.m","rxd0_des", &rxdata[0][0], frame_parms->samples_per_frame,1,1);

#endif

#if TEST_SYNCHRO_TIMING_PSS

  opp_enabled = 1;

  start_meas(&generic_time[TIME_PSS]);

#endif

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PSS_SEARCH_TIME_NR, VCD_FUNCTION_IN);
  synchro_position = pss_search_time_nr(rxdata, PHY_vars_UE, fo_flag, is);
  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PSS_SEARCH_TIME_NR, VCD_FUNCTION_OUT);

#if TEST_SYNCHRO_TIMING_PSS

  stop_meas(&generic_time[TIME_PSS]);

  int duration_ms = generic_time[TIME_PSS].p_time/(cpuf*1000.0);

  #ifndef NR_UNIT_TEST

  LOG_I(PHY,"PSS execution duration %4d microseconds \n", duration_ms);

  #endif

#endif

#ifdef SYNCHRO_DECIMAT

  if (rate_change != 1) {

    if (rxdata[0] != NULL) {

      for (int aa=0;aa<frame_parms->nb_antennas_rx;aa++) {
        free(rxdata[aa]);
      }

      free(rxdata);
    }

    restore_frame_context_pss_nr(frame_parms, rate_change);  
  }
#endif

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PSS_SYNCHRO_NR, VCD_FUNCTION_OUT);
  return synchro_position;
}



/*******************************************************************
*
* NAME :         pss_search_time_nr
*
* PARAMETERS :   received buffer in time domain
*                frame parameters
*
* RETURN :       position of detected pss
*
* DESCRIPTION :  Synchronisation on pss sequence is based on a time domain correlation between received samples and pss sequence
*                A maximum likelihood detector finds the timing offset (position) that corresponds to the maximum correlation
*                Length of received buffer should be a minimum of 2 frames (see TS 38.213 4.1 Cell search)
*                Search pss in the received buffer is done each 4 samples which ensures a memory alignment to 128 bits (32 bits x 4).
*                This is required by SIMD (single instruction Multiple Data) Extensions of Intel processors
*                Correlation computation is based on a a dot product which is realized thank to SIMS extensions
*
*                                    (x frames)
*     <--------------------------------------------------------------------------->
*
*
*     -----------------------------------------------------------------------------
*     |                      Received UE data buffer                              |
*     ----------------------------------------------------------------------------
*                -------------
*     <--------->|    pss    |
*      position  -------------
*                ^
*                |
*            peak position
*            given by maximum of correlation result
*            position matches beginning of first ofdm symbol of pss sequence
*
*     Remark: memory position should be aligned on a multiple of 4 due to I & Q samples of int16
*             An OFDM symbol is composed of x number of received samples depending of Rf front end sample rate.
*
*     I & Q storage in memory
*
*             First samples       Second  samples
*     ------------------------- -------------------------  ...
*     |     I1     |     Q1    |     I2     |     Q2    |
*     ---------------------------------------------------  ...
*     ^    16  bits   16 bits  ^
*     |                        |
*     ---------------------------------------------------  ...
*     |         sample 1       |    sample   2          |
*    ----------------------------------------------------  ...
*     ^
*
*********************************************************************/

int pss_search_time_nr(c16_t **rxdata, PHY_VARS_NR_UE *ue, int fo_flag, int is)
{
  NR_DL_FRAME_PARMS *frame_parms = &ue->frame_parms;
  unsigned int length = is == 0 ? frame_parms->samples_per_frame + (2 * frame_parms->ofdm_symbol_size) :
                        frame_parms->samples_per_frame;
  AssertFatal(length > 0, "illegal length %d\n", length);

  int maxval = 0;
  for (int i = 0; i < frame_parms->ofdm_symbol_size; i++) {
    maxval = max(maxval, primary_synchro_time_nr[0][i].r);
    maxval = max(maxval, primary_synchro_time_nr[0][i].i);
    maxval = max(maxval, -primary_synchro_time_nr[0][i].r);
    maxval = max(maxval, -primary_synchro_time_nr[0][i].i);
    maxval = max(maxval, primary_synchro_time_nr[1][i].r);
    maxval = max(maxval, primary_synchro_time_nr[1][i].i);
    maxval = max(maxval, -primary_synchro_time_nr[1][i].r);
    maxval = max(maxval, -primary_synchro_time_nr[1][i].i);
    if (get_softmodem_params()->sl_mode == 0) {
      maxval = max(maxval, primary_synchro_time_nr[2][i].r);
      maxval = max(maxval, primary_synchro_time_nr[2][i].i);
      maxval = max(maxval, -primary_synchro_time_nr[2][i].r);
      maxval = max(maxval, -primary_synchro_time_nr[2][i].i);
    }
  }

  int avg_size = get_softmodem_params()->sl_mode == 0 ?  NUMBER_PSS_SEQUENCE : NUMBER_PSS_SEQUENCE_SL;
  int64_t avg[avg_size];
  bzero(avg, avg_size);

  int shift = log2_approx(maxval);
  int64_t peak_value = 0;
  unsigned int peak_position = 0;
  unsigned int pss_source = 0;
  /* Search pss in the received buffer each 4 samples which ensures a memory alignment on 128 bits (32 bits x 4 ) */
  /* This is required by SIMD (single instruction Multiple Data) Extensions of Intel processors. */
  /* Correlation computation is based on a a dot product which is realized thank to SIMS extensions */

  uint16_t pss_index_start = 0;
  uint16_t pss_index_end = get_softmodem_params()->sl_mode == 0 ? NUMBER_PSS_SEQUENCE : NUMBER_PSS_SEQUENCE_SL;
  if (ue->target_Nid_cell != -1) {
    pss_index_start = GET_NID2(ue->target_Nid_cell);
    pss_index_end = pss_index_start + 1;
  }
  unsigned int step = get_softmodem_params()->sl_mode == 0 ? 8 : 4;
  for (int i=0;i<10;i++) printf("pss1[%d] %d.%d\n",i,primary_synchro_time_nr[1][i].r,primary_synchro_time_nr[1][i].i);
  for (int pss_index = pss_index_start; pss_index < pss_index_end; pss_index++) {
    for (unsigned int n = 0; n < length; n += step) {
      int64_t pss_corr_ue = 0;
      for (unsigned int ar = 0; ar < frame_parms->nb_antennas_rx; ar++) {
        /* perform correlation of rx data and pss sequence ie it is a dot product */
        const c32_t result = dot_product(primary_synchro_time_nr[pss_index],
                                         &(rxdata[ar][n + is * frame_parms->samples_per_frame]),
                                         frame_parms->ofdm_symbol_size,
                                         shift);
        const c64_t r64 = {.r = result.r, .i = result.i};
        pss_corr_ue += squaredMod(r64);
	if (get_softmodem_params()->sl_mode > 0) {
          //non-coherentely combine repeition of PSS 		
          const c32_t result = dot_product(primary_synchro_time_nr[pss_index],
                                           &(rxdata[ar][n + is * frame_parms->samples_per_frame + frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples]),
                                           frame_parms->ofdm_symbol_size,
                                           shift);
          const c64_t r64 = {.r = result.r, .i = result.i};
          pss_corr_ue += squaredMod(r64);
	}
      }
      /* calculate the absolute value of sync_corr[n] */
      avg[pss_index]+=pss_corr_ue;
      if (pss_corr_ue > peak_value) {
        peak_value = pss_corr_ue;
        peak_position = n;
        pss_source = pss_index;
	LOG_I(NR_PHY,"++++ peak_value %d, peak_pos %d, pss %d\n",dB_fixed64(peak_value),peak_position,pss_source);
      }
    }
  }
  double ffo_est = 0;
  if (fo_flag) {
	  // fractional frequency offset computation according to Cross-correlation Synchronization Algorithm Using PSS
	  // Shoujun Huang, Yongtao Su, Ying He and Shan Tang, "Joint time and frequency offset estimation in LTE downlink," 7th International Conference on Communications and Networking in China, 2012.
    // Computing cross-correlation at peak on half the symbol size for first half of data
    c32_t r1 = dot_product(primary_synchro_time_nr[pss_source],
                           &(rxdata[0][peak_position + is * frame_parms->samples_per_frame]),
                           frame_parms->ofdm_symbol_size >> 1,
                           shift);
    // Computing cross-correlation at peak on half the symbol size for data shifted by half symbol size
    // as it is real and complex it is necessary to shift by a value equal to symbol size to obtain such shift
    c32_t r2 = dot_product(primary_synchro_time_nr[pss_source] + (frame_parms->ofdm_symbol_size >> 1),
                           &(rxdata[0][peak_position + is * frame_parms->samples_per_frame]) + (frame_parms->ofdm_symbol_size >> 1),
                           frame_parms->ofdm_symbol_size >> 1,
                           shift);
    cd_t r1d = {r1.r, r1.i}, r2d = {r2.r, r2.i};
    // estimation of fractional frequency offset: angle[(result1)'*(result2)]/pi
    ffo_est = atan2(r1d.r * r2d.i - r2d.r * r1d.i, r1d.r * r2d.r + r1d.i * r2d.i) / M_PI;
  }
  ue->common_vars.freq_offset = ffo_est * frame_parms->subcarrier_spacing;
  int *id = get_softmodem_params()->sl_mode == 0 ? (int *)&ue->common_vars.eNb_id : (int *)&ue->common_vars.N2_id;
  *id = pss_source;

  for (int pss_index = pss_index_start; pss_index < pss_index_end; pss_index++)
    avg[pss_index] /= (length / 4);

  LOG_I(NR_PHY, "[UE] nr_synchro_time: Sync source = %d, Peak found at pos %d, val = %llu (%d dB) avg %d dB, ffo %lf\n",
        pss_source, peak_position, (unsigned long long)peak_value, dB_fixed64(peak_value), dB_fixed64(avg[pss_source]), ffo_est);

  if (peak_value < 5 * avg[pss_source])
    return(-1);

  return(peak_position);
}

