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

#include <stdlib.h>
#include <time.h>

#include "radio/COMMON/common_lib.h"
#include "common/utils/load_module_shlib.h"
#include "common/utils/LOG/log.h"
#include "openair1/PHY/INIT/nr_phy_init.h"
#include "openair1/PHY/defs_nr_common.h"

int oai_exit;
int usrp_tx_thread = 0;
typedef int(*devfunc_t)(openair0_device *, openair0_config_t *, eth_params_t *);

uint32_t nr_subcarrier_spacing[MAX_NUM_SUBCARRIER_SPACING] = {15e3, 30e3, 60e3, 120e3, 240e3};
uint16_t nr_slots_per_subframe[MAX_NUM_SUBCARRIER_SPACING] = {1, 2, 4, 8, 16};

uint32_t get_samples_symbol_timestamp(const int symbol,
                                      const int slot,
                                      const NR_DL_FRAME_PARMS *fp,
                                      const int symb_ahead)
{
  int sampleCnt = 0;
  for (int i = symbol; i < symbol + symb_ahead; i++) {
    const int absSymbol = slot * fp->symbols_per_slot + i;
    const int prefix_samples = (absSymbol%(0x7<<fp->numerology_index)) ?
                               fp->nb_prefix_samples : fp->nb_prefix_samples0;
    sampleCnt += (prefix_samples + fp->ofdm_symbol_size);
  }
  return sampleCnt;
}

uint32_t get_samples_per_slot(int slot, NR_DL_FRAME_PARMS* fp)
{
  uint32_t samp_count;

  if(fp->numerology_index == 0)
    samp_count = fp->samples_per_subframe;
  else
    samp_count = (slot%(fp->slots_per_subframe/2)) ? fp->samples_per_slotN0 : fp->samples_per_slot0;

  return samp_count;
}

uint32_t get_slot_from_timestamp(openair0_timestamp timestamp_rx, NR_DL_FRAME_PARMS* fp)
{
   uint32_t slot_idx = 0;
   int samples_till_the_slot = fp->get_samples_per_slot(slot_idx,fp)-1;
   timestamp_rx = timestamp_rx%fp->samples_per_frame;

    while (timestamp_rx > samples_till_the_slot) {
        slot_idx++;
        samples_till_the_slot += fp->get_samples_per_slot(slot_idx,fp);
     }
   return slot_idx; 
}

uint32_t get_samples_slot_timestamp(int slot, NR_DL_FRAME_PARMS* fp, uint8_t sl_ahead)
{
  uint32_t samp_count = 0;

  if(!sl_ahead) {
    for(uint8_t idx_slot = 0; idx_slot < slot; idx_slot++)
      samp_count += fp->get_samples_per_slot(idx_slot, fp);
  } else {
    for(uint8_t idx_slot = slot; idx_slot < slot+sl_ahead; idx_slot++)
      samp_count += fp->get_samples_per_slot(idx_slot, fp);
  }
  return samp_count;
}

void set_scs_parameters (NR_DL_FRAME_PARMS *fp, int mu, int N_RB_DL)
{
  switch(mu) {
    case NR_MU_0: //15kHz scs
      fp->subcarrier_spacing = nr_subcarrier_spacing[NR_MU_0];
      fp->slots_per_subframe = nr_slots_per_subframe[NR_MU_0];
      break;

    case NR_MU_1: //30kHz scs
      fp->subcarrier_spacing = nr_subcarrier_spacing[NR_MU_1];
      fp->slots_per_subframe = nr_slots_per_subframe[NR_MU_1];
      break;

    case NR_MU_2: //60kHz scs
      fp->subcarrier_spacing = nr_subcarrier_spacing[NR_MU_2];
      fp->slots_per_subframe = nr_slots_per_subframe[NR_MU_2];
      break;

    case NR_MU_3:
      fp->subcarrier_spacing = nr_subcarrier_spacing[NR_MU_3];
      fp->slots_per_subframe = nr_slots_per_subframe[NR_MU_3];
      break;

    case NR_MU_4:
      fp->subcarrier_spacing = nr_subcarrier_spacing[NR_MU_4];
      fp->slots_per_subframe = nr_slots_per_subframe[NR_MU_4];
      break;

    default:
      break;
  }

  if(fp->threequarter_fs)
    fp->ofdm_symbol_size = 3 * 128;
  else
    fp->ofdm_symbol_size = 4 * 128;

  while(fp->ofdm_symbol_size < N_RB_DL * 12)
    fp->ofdm_symbol_size <<= 1;

  fp->first_carrier_offset = fp->ofdm_symbol_size - (N_RB_DL * 12 / 2);
  fp->nb_prefix_samples    = fp->ofdm_symbol_size / 128 * 9;
  fp->nb_prefix_samples0   = fp->ofdm_symbol_size / 128 * (9 + (1 << mu));
}

void nr_init_frame_parms_ue_sa(NR_DL_FRAME_PARMS *frame_parms, uint64_t downlink_frequency, int32_t delta_duplex, uint8_t mu, uint16_t nr_band) {

  frame_parms->numerology_index = mu;
  frame_parms->dl_CarrierFreq = downlink_frequency;
  frame_parms->ul_CarrierFreq = downlink_frequency + delta_duplex;
  frame_parms->freq_range = (frame_parms->dl_CarrierFreq < 6e9)? nr_FR1 : nr_FR2;
  frame_parms->N_RB_UL = frame_parms->N_RB_DL;

  frame_parms->nr_band = nr_band;
  frame_parms->frame_type = TDD;

  frame_parms->Ncp = NORMAL;
  set_scs_parameters(frame_parms, frame_parms->numerology_index, frame_parms->N_RB_DL);

  frame_parms->slots_per_frame = 10* frame_parms->slots_per_subframe;
  frame_parms->symbols_per_slot = ((frame_parms->Ncp == NORMAL)? 14 : 12); // to redefine for different slot formats
  frame_parms->samples_per_subframe_wCP = frame_parms->ofdm_symbol_size * frame_parms->symbols_per_slot * frame_parms->slots_per_subframe;
  frame_parms->samples_per_frame_wCP = 10 * frame_parms->samples_per_subframe_wCP;
  frame_parms->samples_per_slot_wCP = frame_parms->symbols_per_slot*frame_parms->ofdm_symbol_size;
  frame_parms->samples_per_slotN0 = (frame_parms->nb_prefix_samples + frame_parms->ofdm_symbol_size) * frame_parms->symbols_per_slot;
  frame_parms->samples_per_slot0 = frame_parms->nb_prefix_samples0 + ((frame_parms->symbols_per_slot-1)*frame_parms->nb_prefix_samples) + (frame_parms->symbols_per_slot*frame_parms->ofdm_symbol_size);
  frame_parms->samples_per_subframe = (frame_parms->nb_prefix_samples0 + frame_parms->ofdm_symbol_size) * 2 +
                             (frame_parms->nb_prefix_samples + frame_parms->ofdm_symbol_size) * (frame_parms->symbols_per_slot * frame_parms->slots_per_subframe - 2);
  frame_parms->get_samples_per_slot = &get_samples_per_slot;
  frame_parms->get_samples_slot_timestamp = &get_samples_slot_timestamp;
  frame_parms->samples_per_frame = 10 * frame_parms->samples_per_subframe;
}

int main (int argc, char **argv)
{
  char addr[20] = "192.168.40.2";
  int numerology = 1;
  int runSlots = 100;
  int symbolAhead = 36;
  int logLevel = 3;
  char c;
  while ((c = getopt(argc, argv, "a:n:s:x:l:h")) != -1) {
    switch (c) {
    case 'a':
      strcpy(addr, optarg);
      break;

    case 'n':
      numerology = atoi(optarg);
      break;

    case 's':
      runSlots = atoi(optarg);
      break;

    case 'x':
      symbolAhead = atoi(optarg);
      break;

    case 'l':
      logLevel = atoi(optarg);
      break;

    case 'h':
      printf("a: USRP address\n");
      printf("n: Numerology\n");
      printf("s: Number of slots\n");
      printf("x: Slots ahead for Tx\n");
      break;
    }
  }

  if ( load_configmodule(argc,argv,CONFIG_ENABLECMDLINEONLY) == NULL) {
    exit_fun("[SOFTMODEM] Error, configuration module init failed\n");
  }
  logInit();
  set_glog(logLevel);
  openair0_device device;
  openair0_config_t openair0_cfg;

  NR_DL_FRAME_PARMS frame_parms;
  const double center_freq = 3.6e9;
  char sdr_addrs[100];
  sprintf(sdr_addrs, "mgmt_addr=%s,addr=%s,clock_source=internal,time_source=internal", addr, addr);
  frame_parms.threequarter_fs = 0;
  frame_parms.nb_antennas_rx = 1;
  frame_parms.nb_antennas_tx = 1;
  frame_parms.N_RB_DL = 106;
  nr_init_frame_parms_ue_sa(&frame_parms, center_freq, 0, numerology, 78);

  openair0_cfg.Mod_id = 0;
  openair0_cfg.num_rb_dl = frame_parms.N_RB_DL;
  openair0_cfg.clock_source = 0;
  openair0_cfg.time_source = 0;
  openair0_cfg.tune_offset = 0;
  openair0_cfg.tx_num_channels = frame_parms.nb_antennas_tx;
  openair0_cfg.rx_num_channels = frame_parms.nb_antennas_rx;
  openair0_cfg.configFilename    = NULL;
  openair0_cfg.threequarter_fs   = frame_parms.threequarter_fs;
  openair0_cfg.sample_rate       = frame_parms.samples_per_subframe * 1e3;
  openair0_cfg.samples_per_frame = frame_parms.samples_per_frame;
  openair0_cfg.duplex_mode = duplex_mode_TDD;
  openair0_cfg.sdr_addrs = (char *)sdr_addrs;

  for (int i = 0; i < openair0_cfg.rx_num_channels; i++) {
    openair0_cfg.rx_freq[i] = center_freq;
    openair0_cfg.rx_gain[i] = 0;
    openair0_cfg.autocal[i] = 1;
  }

  for (int i = 0; i < openair0_cfg.tx_num_channels; i++) {
    openair0_cfg.tx_freq[i] = center_freq;
    openair0_cfg.tx_gain[i] = 0;
  }
  loader_shlibfunc_t shlib_fdesc[1];

  shlib_fdesc[0].fname="device_init";

  char devname[50] = "oai_usrpdevif";
  int ret = load_module_shlib(devname, shlib_fdesc, 1, NULL);
  if (ret > 0)
    printf("Library %s couldn't be loaded\n", devname);

  if (((devfunc_t)shlib_fdesc[0].fptr)(&device, &openair0_cfg, NULL) != 0) {
    printf("No radio. exiting\n");
    exit(-1);
  }

  int32_t *rxBuffer[frame_parms.nb_antennas_rx];
  for (int i = 0; i < frame_parms.nb_antennas_rx; i++) {
    rxBuffer[i] = malloc16(frame_parms.samples_per_frame * sizeof(int32_t));
  }
  int32_t *txBuffer[frame_parms.nb_antennas_tx];
  for (int i = 0; i < frame_parms.nb_antennas_tx; i++) {
    txBuffer[i] = malloc16(frame_parms.samples_per_frame * sizeof(int32_t));
  }

  openair0_timestamp timeStamp;

  if (device.trx_start_func(&device) != 0) printf("couldn't start radio\n");

  printf("Tx slot ahead: %d\nNumber of slots: %d\nSamples per frame: %d\n",
         symbolAhead, runSlots, frame_parms.samples_per_frame);

  uint32_t slot = 0;
  for (uint32_t symbol = 0; symbol < runSlots*frame_parms.symbols_per_slot; symbol++) {
    /* Read samples */
    const int absSymbol = symbol % (frame_parms.slots_per_frame * frame_parms.symbols_per_slot);
    int prefixSamples = (absSymbol%(0x7<<numerology)) ?
                               frame_parms.nb_prefix_samples : frame_parms.nb_prefix_samples0;
    const int readSamples = prefixSamples + frame_parms.ofdm_symbol_size;
    const clock_t startRx = clock();
    if (readSamples != device.trx_read_func(&device,
                                            &timeStamp,
                                            (void **)rxBuffer,
                                            readSamples,
                                            frame_parms.nb_antennas_rx)) {
      printf("read error\n");
    }
    const clock_t stopRx = clock();
    printf("Rx latency %0.2lfus\n", (float)(stopRx - startRx)/CLOCKS_PER_SEC*1e6);

    /* write samples */
    
    const openair0_timestamp writeTimeStamp = timeStamp + get_samples_symbol_timestamp(symbol, slot, &frame_parms, symbolAhead);
    prefixSamples = ((absSymbol+symbolAhead)%(0x7<<numerology)) ?
                           frame_parms.nb_prefix_samples : frame_parms.nb_prefix_samples0;
    const int writeSamples = prefixSamples + frame_parms.ofdm_symbol_size;
    const clock_t startTx = clock();
    if (writeSamples != device.trx_write_func(&device,
                                              writeTimeStamp,
                                              (void **)txBuffer,
                                              writeSamples,
                                              frame_parms.nb_antennas_tx,
                                              1)) {
      printf("write error\n");
    }
    const clock_t stopTx = clock();
    printf("Tx latency %0.2lfus\n", (float)(stopTx - startTx)/CLOCKS_PER_SEC*1e6);
    slot += !(symbol % frame_parms.symbols_per_slot);
  }
}
