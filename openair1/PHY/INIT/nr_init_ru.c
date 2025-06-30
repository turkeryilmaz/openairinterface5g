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

#include "phy_init.h"
#include "PHY/phy_extern.h"
#include "SCHED/sched_common.h"
#include "common/utils/LOG/vcd_signal_dumper.h"
#include "assertions.h"
#include <math.h>
#include "openair1/PHY/defs_RU.h"

#define DEVELOP_CIR

#ifdef DEVELOP_CIR
#include "SIMULATION/TOOLS/sim.h"
static char cir_conf_file[] = "../../../cir_conf.txt";
static const char cir_file_template[] = "../../../cir/output/binary/delayamplist";
static const char delayindexlist_template[] = "../../../cir/output/binary/delayindexlist";
#endif

void init_prach_ru_list(RU_t *ru);

#ifdef DEVELOP_CIR
void init_cir_variables(void *param)
{
  RU_t *ru = (RU_t *)param;
  FILE *fptr;
  cf_t path_loss_dB = (cf_t){0.0, 0.0};
  float amp_gain_dB = 0.0;
  cf_t noise_power_dB = (cf_t){0.0, 0.0};
  int channel_length = 1;
  char str[256];
  char cir_file_path[256];
  char delayindexlist_path[256];

  // read file: pathloss_db, noise_power_dB
  fptr = fopen(cir_conf_file, "r");
  if (fptr) {
    // pathLossLinear
    fgets(str, sizeof(str), fptr);
    sscanf(str, "%f", &path_loss_dB.r);
    // amp_gain_dB
    fgets(str, sizeof(str), fptr);
    sscanf(str, "%f", &amp_gain_dB);
    // noise_per_sample
    fgets(str, sizeof(str), fptr);
    sscanf(str, "%f", &noise_power_dB.r);
    // num of taps
    fgets(str, sizeof(str), fptr);
    sscanf(str, "%d", &channel_length);
    fclose(fptr);
  } else {
    printf("error: cir_conf.txt\n");
    fflush(stdout);
  }

  // write params
  pthread_mutex_lock(&ru->proc.mutex_mimo);
  ru->pathLossLinear.r = pow(10, (path_loss_dB.r + amp_gain_dB) / 20.0);
  ru->pathLossLinear.i = 0.0;
  ru->noise_per_sample.r = pow(10, noise_power_dB.r / 20.0) * 256; // TODO: check formula
  ru->noise_per_sample.i = 0.0; // TODO: check formula
  ru->channel_length = channel_length;
  pthread_mutex_unlock(&ru->proc.mutex_mimo);

  // read cir data
  cf_t cir_buffer[channel_length * ru->nb_tx * ru->nb_rx];
  memset(cir_buffer, 0, sizeof(cir_buffer));
  ru->cirMIMO_simulmatrix = (cf_t *)aligned_alloc(64, ru->nb_tx * ru->nb_rx * channel_length * sizeof(cf_t));
  sprintf(cir_file_path, "%s0000.b", cir_file_template);
  fptr = fopen(cir_file_path, "rb");
  if (fptr) {
    fread(cir_buffer, sizeof(cir_buffer), 1, fptr);
    fclose(fptr);
    pthread_mutex_lock(&ru->proc.mutex_mimo);
    for (int l = 0; l < channel_length; l++) {
      for (int a_tx = 0; a_tx < ru->nb_tx; a_tx++) {
        for (int a_rx = 0; a_rx < ru->nb_rx; a_rx++) {
            ru->cirMIMO_simulmatrix[a_rx*channel_length*ru->nb_tx + l*ru->nb_rx + a_tx].r = cir_buffer[l*ru->nb_tx*ru->nb_rx + ru->nb_rx*a_tx + a_rx].r;
            ru->cirMIMO_simulmatrix[a_rx*channel_length*ru->nb_tx + l*ru->nb_rx + a_tx].i = cir_buffer[l*ru->nb_tx*ru->nb_rx + ru->nb_rx*a_tx + a_rx].i;
        }
      }
    }
    pthread_mutex_unlock(&ru->proc.mutex_mimo);
  }

  // read delay index list
  int delayindexlist_tmp[channel_length];
  memset(delayindexlist_tmp, 0, sizeof(delayindexlist_tmp));
  ru->delayindexlist = (int *)aligned_alloc(64, channel_length*sizeof(int));
  sprintf(delayindexlist_path, "%s0000.b", delayindexlist_template);
  fptr = fopen(delayindexlist_path, "rb");
  if (fptr) {
    fread(delayindexlist_tmp, sizeof(delayindexlist_tmp), 1, fptr);
    fclose(fptr);
    pthread_mutex_lock(&ru->proc.mutex_mimo);
    for (int l = 0; l < channel_length; l++) {
      ru->delayindexlist[l] = delayindexlist_tmp[l];
    }
    pthread_mutex_unlock(&ru->proc.mutex_mimo);
  }
}

void init_noise(void *param)
{
  RU_t *ru = (RU_t *)param;
  NR_DL_FRAME_PARMS *fp = ru->nr_frame_parms;
  int siglen = fp->get_samples_per_slot(0, fp);
  int i, a;

  // init noise_array
  ru->noise_array = aligned_alloc(64, sizeof(cf_t) * ru->nb_tx);
  for (a = 0; a < ru->nb_tx; a++) ru->noise_array[a] = aligned_alloc(64, sizeof(cf_t) * siglen);

  // set noise -> noise_array
  for (a = 0; a < ru->nb_tx; a++) {
    for (i = 0; i < siglen; i++) {
      pthread_mutex_lock(&ru->proc.mutex_noise);
      ru->noise_array[a][i].r = (float)gaussZiggurat(0.0, 1.0);
      ru->noise_array[a][i].i = (float)gaussZiggurat(0.0, 1.0);
      pthread_mutex_unlock(&ru->proc.mutex_noise);
    }
  }
}

#endif // DEVELOP_CIR

int nr_phy_init_RU(RU_t *ru)
{
  NR_DL_FRAME_PARMS *fp = ru->nr_frame_parms;

  LOG_D(PHY, "Initializing RU signal buffers (if_south %s) nb_tx %d, nb_rx %d\n", ru_if_types[ru->if_south], ru->nb_tx, ru->nb_rx);

  nfapi_nr_config_request_scf_t *cfg = &ru->config;
  ru->nb_log_antennas = 0;
  for (int n = 0; n < ru->num_gNB; n++) {
    if (cfg->carrier_config.num_tx_ant.value > ru->nb_log_antennas)
      ru->nb_log_antennas = cfg->carrier_config.num_tx_ant.value;
  }

  // copy configuration from gNB[0] in to RU, assume that all gNB instances sharing RU use the same configuration
  // (at least the parts that are needed by the RU, numerology and PRACH)

  AssertFatal(ru->nb_log_antennas > 0 && ru->nb_log_antennas < 13, "ru->nb_log_antennas %d ! \n",ru->nb_log_antennas);

  ru->common.beam_id = malloc16_clear(MAX_NUM_BEAM_PERIODS * sizeof(int*));
  for(int i = 0; i < MAX_NUM_BEAM_PERIODS; i++)
    ru->common.beam_id[i] = malloc16_clear(fp->symbols_per_slot * fp->slots_per_frame * sizeof(int));

  if (ru->if_south <= REMOTE_IF5) { // this means REMOTE_IF5 or LOCAL_RF, so allocate memory for time-domain signals 
    // Time-domain signals
    ru->common.txdata        = (int32_t**)malloc16(ru->nb_tx*sizeof(int32_t*));
    ru->common.rxdata        = (int32_t**)malloc16(ru->nb_rx*sizeof(int32_t*) );


    for (int i = 0; i < ru->nb_tx; i++) {
      // Allocate 10 subframes of I/Q TX signal data (time) if not
      ru->common.txdata[i]  = (int32_t*)malloc16_clear((ru->sf_extension + fp->samples_per_frame)*sizeof(int32_t));
      LOG_D(PHY,
            "[INIT] common.txdata[%d] = %p (%lu bytes,sf_extension %d)\n",
            i,
            ru->common.txdata[i],
            (ru->sf_extension + fp->samples_per_frame) * sizeof(int32_t),
            ru->sf_extension);
      ru->common.txdata[i] =  &ru->common.txdata[i][ru->sf_extension];

      LOG_D(PHY, "[INIT] common.txdata[%d] = %p \n", i, ru->common.txdata[i]);
    }
    for (int i = 0; i < ru->nb_rx; i++) {
      ru->common.rxdata[i] = (int32_t*)malloc16_clear( fp->samples_per_frame*sizeof(int32_t) );
    }
  } // IF5 or local RF
  else {
    ru->common.txdata        = (int32_t**)NULL;
    ru->common.rxdata        = (int32_t**)NULL;
  }
  if (ru->function != NGFI_RRU_IF5) { // we need to do RX/TX RU processing
    LOG_D(PHY, "nb_tx %d\n", ru->nb_tx);
    ru->common.rxdata_7_5kHz = (int32_t**)malloc16(ru->nb_rx*sizeof(int32_t*) );
    for (int i = 0; i < ru->nb_rx; i++) {
      ru->common.rxdata_7_5kHz[i] = (int32_t*)malloc16_clear( 2*fp->samples_per_subframe*2*sizeof(int32_t) );
      LOG_D(PHY, "rxdata_7_5kHz[%d] %p for RU %d\n", i, ru->common.rxdata_7_5kHz[i], ru->idx);
    }
  

    // allocate precoding input buffers (TX)
    ru->common.txdataF = (int32_t **)malloc16(ru->nb_tx*sizeof(int32_t*));
    // [hna] samples_per_frame without CP
    for(int i = 0; i < ru->nb_tx; ++i)
      ru->common.txdataF[i] = (int32_t*)malloc16_clear(fp->samples_per_frame_wCP * sizeof(int32_t));

    // allocate IFFT input buffers (TX)
    ru->common.txdataF_BF = (int32_t **)malloc16(ru->nb_tx*sizeof(int32_t*));
    LOG_D(PHY, "[INIT] common.txdata_BF= %p (%lu bytes)\n", ru->common.txdataF_BF, ru->nb_tx * sizeof(int32_t *));
    for (int i = 0; i < ru->nb_tx; i++) {
      ru->common.txdataF_BF[i] = (int32_t*)malloc16_clear(fp->samples_per_subframe_wCP*sizeof(int32_t) );
      LOG_D(PHY, "txdataF_BF[%d] %p for RU %d\n", i, ru->common.txdataF_BF[i], ru->idx);
    }
    // allocate FFT output buffers (RX)
    ru->common.rxdataF     = (int32_t**)malloc16(ru->nb_rx*sizeof(int32_t*) );
    for (int i = 0; i < ru->nb_rx; i++) {
      // allocate 4 slots of I/Q signal data (frequency)
      int size = RU_RX_SLOT_DEPTH * fp->symbols_per_slot * fp->ofdm_symbol_size;
      ru->common.rxdataF[i] = (int32_t*)malloc16_clear(sizeof(**ru->common.rxdataF) * size);
      LOG_D(PHY, "rxdataF[%d] %p for RU %d\n", i, ru->common.rxdataF[i], ru->idx);
    }

    /* number of elements of an array X is computed as sizeof(X) / sizeof(X[0]) */
    //    AssertFatal(ru->nb_rx <= sizeof(ru->prach_rxsigF) / sizeof(ru->prach_rxsigF[0]),
    //		"nb_antennas_rx too large");
    for (int j = 0; j < NUMBER_OF_NR_RU_PRACH_OCCASIONS_MAX; j++) {
      ru->prach_rxsigF[j] = (int16_t**)malloc(ru->nb_rx * sizeof(int16_t*));
      
      for (int i = 0; i < ru->nb_rx; i++) {
	// largest size for PRACH FFT is 4x98304 (16*24576)
	ru->prach_rxsigF[j][i] = (int16_t*)malloc16_clear( 4*98304*2*sizeof(int16_t) );
	LOG_D(PHY,"[INIT] prach_vars->rxsigF[%d] = %p\n",i,ru->prach_rxsigF[j][i]);
      }
    }
    
    AssertFatal(ru->num_gNB <= NUMBER_OF_gNB_MAX,"gNB instances %d > %d\n",
		ru->num_gNB,NUMBER_OF_gNB_MAX);

    LOG_D(PHY, "[INIT] %s() ru->num_gNB:%d \n", __FUNCTION__, ru->num_gNB);

  } // !=IF5

  init_prach_ru_list(ru);

  #ifdef DEVELOP_CIR
  printf("INIT DEVELOP_CIR\n");
  fflush(stdout);

  // init cir variables
  init_cir_variables(ru); // init: convolution matrix, channel_length, pathLossLinear, noise_per_sample
  init_noise(ru); // init: noise array

  ru->common.buffboundary = 0;
  ru->common.circular_buff_size = fp->samples_per_slot0 * 20;

  ru->common.circular_buff = (cf_t **)aligned_alloc(64, ru->nb_rx * sizeof(cf_t));
  for (int a = 0; a < ru->nb_rx; a++) {
    ru->common.circular_buff[a] = (cf_t *)aligned_alloc(64, ru->common.circular_buff_size * sizeof(cf_t));
    memset(&ru->common.circular_buff[a][0], 0, sizeof(cf_t) * ru->common.circular_buff_size);
  }

  ru->common.noise_array = (cf_t *)aligned_alloc(64, ru->nb_rx * fp->samples_per_slot0 * sizeof(cf_t));

  // allocate MIMO temporary store fields
  ru->common.simul_input = (cf_t *)aligned_alloc(
    64,
    ru->nb_tx * ru->channel_length * fp->samples_per_slot0 * sizeof(cf_t) // TODO: ru->nb_tx -> UE->nb_tx
    ); // (nb_tx*channel_length * nsamps)
  #endif // DEVELOP_CIR

  return(0);
}

void nr_phy_free_RU(RU_t *ru)
{
  LOG_D(PHY, "Freeing RU signal buffers (if_south %s) nb_tx %d\n", ru_if_types[ru->if_south], ru->nb_tx);

  if (ru->if_south <= REMOTE_IF5) { // this means REMOTE_IF5 or LOCAL_RF, so free memory for time-domain signals
    // Hack: undo what is done at allocation
    for (int i = 0; i < ru->nb_tx; i++) {
      int32_t *p = &ru->common.txdata[i][-ru->sf_extension];
      free_and_zero(p);
    }
    free_and_zero(ru->common.txdata);

    for (int i = 0; i < ru->nb_rx; i++)
      free_and_zero(ru->common.rxdata[i]);
    free_and_zero(ru->common.rxdata);
  } // else: IF5 or local RF -> nothing to free()

  if (ru->function != NGFI_RRU_IF5) { // we need to do RX/TX RU processing
    for (int i = 0; i < ru->nb_rx; i++)
      free_and_zero(ru->common.rxdata_7_5kHz[i]);
    free_and_zero(ru->common.rxdata_7_5kHz);

    // free beamforming input buffers (TX)
    for (int i = 0; i < ru->nb_tx; i++)
      free_and_zero(ru->common.txdataF[i]);
    free_and_zero(ru->common.txdataF);

    // free IFFT input buffers (TX)
    for (int i = 0; i < ru->nb_tx; i++)
      free_and_zero(ru->common.txdataF_BF[i]);
    free_and_zero(ru->common.txdataF_BF);

    // free FFT output buffers (RX)
    for (int i = 0; i < ru->nb_rx; i++)
      free_and_zero(ru->common.rxdataF[i]);
    free_and_zero(ru->common.rxdataF);

    for (int j = 0; j < NUMBER_OF_NR_RU_PRACH_OCCASIONS_MAX; j++) {
      for (int i = 0; i < ru->nb_rx; i++)
	free_and_zero(ru->prach_rxsigF[j][i]);
      free_and_zero(ru->prach_rxsigF[j]);
    }
    for(int i = 0; i < MAX_NUM_BEAM_PERIODS; ++i)
      free_and_zero(ru->common.beam_id[i]);
    free_and_zero(ru->common.beam_id);
  }

  PHY_VARS_gNB *gNB0 = ru->gNB_list[0];
  gNB0->num_RU--;
  DevAssert(gNB0->num_RU >= 0);
#ifdef DEVELOP_CIR
  printf("terminate DEVELOP_CIR\n");
  fflush(stdout);
  free(ru->common.simul_input);
  free(ru->delayindexlist);
  free(ru->cirMIMO_simulmatrix);
  for (int a = 0; a < ru->nb_tx; a++) free(ru->noise_array[a]);
  free(ru->noise_array);

  for (int a = 0; a < ru->nb_rx; a++) free(ru->common.circular_buff[a]);
  free(ru->common.circular_buff);

  free(ru->common.noise_array);
#endif // DEVELOP_CIR
}
