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
*                Called by Nearby UE for pss decoding in Sidelink Rx.
*
*********************************************************************/

void generate_pss_nr(NR_DL_FRAME_PARMS *fp, int N_ID_2, int pss_seq_offset)
{
  AssertFatal(fp->ofdm_symbol_size > 127,"Illegal ofdm_symbol_size %d\n",fp->ofdm_symbol_size);
  AssertFatal(N_ID_2>=0 && N_ID_2 <=2,"Illegal N_ID_2 %d\n",N_ID_2);
  const int x_initial[INITIAL_PSS_NR] = {0, 1, 1 , 0, 1, 1, 1};
  int_least16_t x[LENGTH_PSS_NR];
  for (int i = 0; i < INITIAL_PSS_NR; i++)
    x[i] = x_initial[i];

  for (int i = 0; i < (LENGTH_PSS_NR - INITIAL_PSS_NR); i++)
    x[i + INITIAL_PSS_NR] = (x[i + 4] + x[i]) % 2;

  c16_t d_pss[LENGTH_PSS_NR];
  for (int n = 0; n < LENGTH_PSS_NR; n++) {
    int m = (n + pss_seq_offset + 43 * N_ID_2) % (LENGTH_PSS_NR);
    d_pss[n].r = (1 - 2 * x[m]);
  }

  /* PSS is directly mapped to subcarrier without modulation 38.211 */
  c16_t primary_synchro[LENGTH_PSS_NR];
  c16_t *primary_synchro2 = primary_synchro_nr[N_ID_2];
  for (int i = 0; i < LENGTH_PSS_NR; i++) {
    primary_synchro[i].r = ((d_pss[i].r * SHRT_MAX) >> SCALING_PSS_NR);
    primary_synchro2[i] = d_pss[i];
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

  c16_t in[fp->ofdm_symbol_size];
  memset(in, 0, sizeof(in));
  for (int i = 0; i < LENGTH_PSS_NR; i++) {
    in[k] = primary_synchro[i];
    k++;
    if (k == fp->ofdm_symbol_size) k = 0;
  }

  /* IFFT will give temporal signal of Pss */

  c16_t out[fp->ofdm_symbol_size];
  idft((int16_t)get_idft(fp->ofdm_symbol_size), (int16_t *)in, (int16_t *)out, 1);
  c16_t *primary_synchro_time = primary_synchro_time_nr[N_ID_2];
  for (unsigned int i = 0; i < fp->ofdm_symbol_size; i++) {
    primary_synchro_time[i] = out[i];
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
  AssertFatal(frame_parms_ue->ofdm_symbol_size > 127, "illegal frame_parms_ue->ofdm_symbol_size %d\n",
              frame_parms_ue->ofdm_symbol_size);
  c16_t *p = NULL;
  int pss_sequence = get_softmodem_params()->sl_mode == 0 ?  NUMBER_PSS_SEQUENCE : NUMBER_PSS_SEQUENCE_SL;
  for (int i = 0; i < pss_sequence; i++) {
    p = malloc16(2 * frame_parms_ue->ofdm_symbol_size);
    if (p != NULL) {
      primary_synchro_time_nr[i] = p;
      bzero(primary_synchro_time_nr[i], 2 * frame_parms_ue->ofdm_symbol_size);
    }
    p = malloc16(2 * frame_parms_ue->ofdm_symbol_size);
    if (p != NULL) {
      primary_synchro_nr[i] = p;
      bzero(primary_synchro_nr[i], 2 * frame_parms_ue->ofdm_symbol_size);
    }
    generate_pss_nr(frame_parms_ue, i, pss_sequence);
  }
  init_context_sss_nr(AMP);
}

/*******************************************************************
*
* NAME :         free_context_pss_nr
*
* PARAMETERS :   none
*
* RETURN :       none
*
* DESCRIPTION :  free context related to pss
*
*********************************************************************/

void free_context_synchro_nr(void)
{
  int pss_sequence = get_softmodem_params()->sl_mode == 0 ?  NUMBER_PSS_SEQUENCE : NUMBER_PSS_SEQUENCE_SL;
  for (int i = 0; i < pss_sequence; i++) {
    free_and_zero(primary_synchro_nr[i]);
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

  /* pss reference have to be rebuild with new parameters ie ofdm symbol size */
  init_context_synchro_nr(frame_parms_ue);

#ifdef SYNCHRO_DECIMAT
  set_pss_nr(frame_parms_ue->ofdm_symbol_size);
#endif
}

/*******************************************************************
*
* NAME :         restore_frame_context_pss_nr
*
* PARAMETERS :   configuration for UE and eNB with new FFT size
*
* RETURN :       0 if OK else error
*
* DESCRIPTION :  initialisation of UE and eNode contexts
*
*********************************************************************/

void restore_frame_context_pss_nr(NR_DL_FRAME_PARMS *frame_parms_ue, int rate_change)
{
  frame_parms_ue->ofdm_symbol_size = frame_parms_ue->ofdm_symbol_size * rate_change;
  frame_parms_ue->samples_per_subframe = frame_parms_ue->samples_per_subframe * rate_change;

  /* pss reference have to be rebuild with new parameters ie ofdm symbol size */
  init_context_synchro_nr(frame_parms_ue);
#ifdef SYNCHRO_DECIMAT
  set_pss_nr(frame_parms_ue->ofdm_symbol_size);
#endif
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
  synchro_position = pss_search_time_nr(rxdata,
                                        frame_parms,
                                        fo_flag,
                                        is,
                                        ((get_softmodem_params()->sl_mode == 0) ?
                                                (int *)&PHY_vars_UE->common_vars.eNb_id :
                                                (int *)&PHY_vars_UE->common_vars.N2_id),
                                        (int *)&PHY_vars_UE->common_vars.freq_offset);


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
* PARAMETERS :   received buffer
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

int pss_search_time_nr(c16_t **rxdata, ///rx data in time domain
                       NR_DL_FRAME_PARMS *frame_parms,
                       int fo_flag,
                       int is,
                       int *id,
                       int *f_off)
{
  unsigned int length = is == 0 ? frame_parms->samples_per_frame + (2 * frame_parms->ofdm_symbol_size) :
                        frame_parms->samples_per_frame;
  AssertFatal(length > 0, "illegal length %d\n", length);

  int maxval = 0;
  for (int i = 0; i < frame_parms->ofdm_symbol_size;i++) {
    maxval = max(maxval, primary_synchro_time_nr[0][i].r);
    maxval = max(maxval, -primary_synchro_time_nr[0][i].r);
    maxval = max(maxval, primary_synchro_time_nr[1][i].r);
    maxval = max(maxval, -primary_synchro_time_nr[1][i].r);
    maxval = max(maxval, primary_synchro_time_nr[0][i].i);
    maxval = max(maxval, -primary_synchro_time_nr[0][i].i);
    maxval = max(maxval, primary_synchro_time_nr[1][i].i);
    maxval = max(maxval, -primary_synchro_time_nr[1][i].i);
    if (get_softmodem_params()->sl_mode == 0) {
      maxval = max(maxval, primary_synchro_time_nr[2][i].r);
      maxval = max(maxval, -primary_synchro_time_nr[2][i].r);
      maxval = max(maxval, primary_synchro_time_nr[2][i].i);
      maxval = max(maxval, -primary_synchro_time_nr[2][i].i);
    }
  }

  int avg_size = get_softmodem_params()->sl_mode == 0 ?  NUMBER_PSS_SEQUENCE : NUMBER_PSS_SEQUENCE_SL;
  int64_t avg[avg_size];
  bzero(avg, avg_size);

  int shift = log2_approx(maxval);//*(frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples)*2);
  int64_t peak_value = 0;
  unsigned int peak_position = 0;
  unsigned int pss_source = 0;
  /* Search pss in the received buffer each 4 samples which ensures a memory alignment on 128 bits (32 bits x 4 ) */
  /* This is required by SIMD (single instruction Multiple Data) Extensions of Intel processors. */
  /* Correlation computation is based on a a dot product which is realized thank to SIMS extensions */
  int pss_sequence = get_softmodem_params()->sl_mode == 0 ? NUMBER_PSS_SEQUENCE : NUMBER_PSS_SEQUENCE_SL;
  unsigned int step = get_softmodem_params()->sl_mode == 0 ? 8 : 4;
  for (int pss_index = 0; pss_index < pss_sequence; pss_index++) {
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
      }
      /* calculate the absolute value of sync_corr[n] */
      avg[pss_index]+=pss_corr_ue;
      if (pss_corr_ue > peak_value) {
        peak_value = pss_corr_ue;
        peak_position = n;
        pss_source = pss_index;
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
                           frame_parms->ofdm_symbol_size,
                           shift);
    // Computing cross-correlation at peak on half the symbol size for data shifted by half symbol size
    // as it is real and complex it is necessary to shift by a value equal to symbol size to obtain such shift
    c32_t r2 = dot_product(primary_synchro_time_nr[pss_source],
                           &(rxdata[0][peak_position + is * frame_parms->samples_per_frame]),
                           frame_parms->ofdm_symbol_size,
                           shift);
    cd_t r1d = {r1.r, r1.i}, r2d = {r2.r, r2.i};
    // estimation of fractional frequency offset: angle[(result1)'*(result2)]/pi
    ffo_est = atan2(r1d.r * r2d.i - r2d.r * r1d.i, r1d.r * r2d.r + r1d.i * r2d.i) / M_PI;
  }
  *f_off = ffo_est * frame_parms->subcarrier_spacing;
  *id = pss_source;

  for (int pss_index = 0; pss_index < NUMBER_PSS_SEQUENCE; pss_index++)
    avg[pss_index] /= (length / step);

  LOG_I(NR_PHY, "[UE] nr_synchro_time: Sync source = %d, Peak found at pos %d, val = %llu (%d dB) avg %d dB, ffo %lf\n",
        pss_source, peak_position, (unsigned long long)peak_value, dB_fixed64(peak_value), dB_fixed64(avg[pss_source]), ffo_est);

  if (peak_value < 5 * avg[pss_source])
    return(-1);

  return(peak_position);
}

