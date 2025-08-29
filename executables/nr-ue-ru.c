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

#include "nr-ue-ru.h"
#include "PHY/defs_nr_common.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_proto_ue.h"

int nrue_cell_count;
nrUE_cell_params_t *nrue_cells;
int nrue_ru_count;
nrUE_RU_params_t *nrue_rus;

openair0_config_t openair0_cfg[MAX_CARDS];
openair0_device_t openair0_dev[MAX_CARDS];
NR_DL_FRAME_PARMS cell_fp[MAX_CARDS];

static int firstTS_initialized = 0;

void nr_ue_ru_start(void)
{
  for (int card = 0; card < MAX_CARDS; card++) {
    openair0_config_t *cfg0 = &openair0_cfg[card];
    openair0_device_t *dev0 = &openair0_dev[card];
    if (cfg0->sample_rate == 0)
      continue;
    int tmp = openair0_device_load(dev0, cfg0);
    AssertFatal(tmp == 0, "Could not load the device %d\n", card);
    dev0->host_type = RAU_HOST;
    int tmp2 = dev0->trx_start_func(dev0);
    AssertFatal(tmp2 == 0, "Could not start the device %d\n", card);
    if (usrp_tx_thread == 1)
      dev0->trx_write_init(dev0);
  }
}

void nr_ue_ru_end(void)
{
  for (int card = 0; card < MAX_CARDS; card++) {
    if (openair0_dev[card].trx_get_stats_func)
      openair0_dev[card].trx_get_stats_func(&openair0_dev[card]);
    if (openair0_dev[card].trx_end_func)
      openair0_dev[card].trx_end_func(&openair0_dev[card]);
  }
}

void nr_ue_ru_set_freq(PHY_VARS_NR_UE *UE, uint64_t ul_carrier, uint64_t dl_carrier, int freq_offset)
{
  NR_DL_FRAME_PARMS *fp0 = &cell_fp[UE->rf_map.card];
  uint64_t carrier_distance =
      (fp0->dl_CarrierFreq > dl_carrier ? fp0->dl_CarrierFreq - dl_carrier : dl_carrier - fp0->dl_CarrierFreq) +
      (fp0->ul_CarrierFreq > ul_carrier ? fp0->ul_CarrierFreq - ul_carrier : ul_carrier - fp0->ul_CarrierFreq);
  for (int card = 0; carrier_distance != 0 && card < MAX_CARDS; card++) {
    fp0 = &cell_fp[card];
    if (fp0->samples_per_subframe == 0)
      continue;
    uint64_t this_carrier_distance =
        (fp0->dl_CarrierFreq > dl_carrier ? fp0->dl_CarrierFreq - dl_carrier : dl_carrier - fp0->dl_CarrierFreq) +
        (fp0->ul_CarrierFreq > ul_carrier ? fp0->ul_CarrierFreq - ul_carrier : ul_carrier - fp0->ul_CarrierFreq);
    if (this_carrier_distance < carrier_distance) {
      UE->rf_map.card = card;
      carrier_distance = this_carrier_distance;
    }
  }

  openair0_config_t *cfg0 = &openair0_cfg[UE->rf_map.card];
  openair0_device_t *dev0 = &openair0_dev[UE->rf_map.card];
  nr_rf_card_config_freq(cfg0, ul_carrier, dl_carrier, freq_offset);
  dev0->trx_set_freq_func(dev0, cfg0);
}

int nr_ue_ru_adjust_rx_gain(PHY_VARS_NR_UE *UE, int gain_change)
{
  openair0_config_t *cfg0 = &openair0_cfg[UE->rf_map.card];
  openair0_device_t *dev0 = &openair0_dev[UE->rf_map.card];

  // Increase the RX gain by the value determined by adjust_rxgain
  cfg0->rx_gain[0] += gain_change;

  // Set new RX gain.
  int ret_gain = dev0->trx_set_gains_func(dev0, cfg0);
  // APPLY RX gain again if crossed the MAX RX gain threshold
  if (ret_gain < 0) {
    gain_change += ret_gain;
    cfg0->rx_gain[0] += ret_gain;
    ret_gain = dev0->trx_set_gains_func(dev0, cfg0);
  }

  int applied_rxgain = cfg0->rx_gain[0] - cfg0->rx_gain_offset[0];
  LOG_I(HW, "Rxgain adjusted by %d dB, RX gain: %d dB \n", gain_change, applied_rxgain);

  return gain_change;
}

int nr_ue_ru_read(PHY_VARS_NR_UE *UE, openair0_timestamp_t *ptimestamp, void **buff, int nsamps, int num_antennas)
{
  openair0_device_t *dev0 = &openair0_dev[UE->rf_map.card];
  openair0_timestamp_t tmp_timestamp;
  int ret = dev0->trx_read_func(dev0, &tmp_timestamp, buff, nsamps, num_antennas);
  if (!firstTS_initialized)
    dev0->firstTS = tmp_timestamp;
  *ptimestamp = tmp_timestamp - dev0->firstTS;

  void *tmp_buf[num_antennas];
  uint32_t tmp_samples[nsamps];
  for (int ant = 0; ant < num_antennas; ant++)
    tmp_buf[ant] = tmp_samples;

  for (int card = 0; card < MAX_CARDS; card++) {
    dev0 = &openair0_dev[card];
    if (card == UE->rf_map.card || dev0->trx_read_func == NULL)
      continue;
    dev0->trx_read_func(dev0, &tmp_timestamp, tmp_buf, nsamps, num_antennas);
    if (!firstTS_initialized)
      dev0->firstTS = tmp_timestamp;
  }

  firstTS_initialized = 1;

  return ret;
}

int nr_ue_ru_write(PHY_VARS_NR_UE *UE, openair0_timestamp_t timestamp, void **buff, int nsamps, int num_antennas, int flags)
{
  openair0_device_t *dev0 = &openair0_dev[UE->rf_map.card];
  int ret = dev0->trx_write_func(dev0, timestamp + dev0->firstTS, buff, nsamps, num_antennas, flags);

  void *tmp_buf[num_antennas];
  uint32_t tmp_samples[nsamps];
  memset(tmp_samples, 0, sizeof(tmp_samples));
  for (int ant = 0; ant < num_antennas; ant++)
    tmp_buf[ant] = tmp_samples;

  for (int card = 0; card < MAX_CARDS; card++) {
    dev0 = &openair0_dev[card];
    if (card == UE->rf_map.card || dev0->trx_write_func == NULL)
      continue;
    dev0->trx_write_func(dev0, timestamp + dev0->firstTS, tmp_buf, nsamps, num_antennas, flags);
  }
  return ret;
}

static void writerEnqueue(re_order_t *ctx, openair0_timestamp_t timestamp, void **txp, int nsamps, int nbAnt, int flags)
{
  pthread_mutex_lock(&ctx->mutex_store);
  LOG_D(HW, "Enqueue write for TS: %lu\n", timestamp);
  int i;
  for (i = 0; i < WRITE_QUEUE_SZ; i++)
    if (!ctx->queue[i].active) {
      ctx->queue[i].timestamp = timestamp;
      ctx->queue[i].active = true;
      ctx->queue[i].nsamps = nsamps;
      ctx->queue[i].nbAnt = nbAnt;
      ctx->queue[i].flags = flags;
      AssertFatal(nbAnt <= NB_ANTENNAS_TX, "");
      for (int j = 0; j < nbAnt; j++)
        ctx->queue[i].txp[j] = txp[j];
      break;
    }
  AssertFatal(i < WRITE_QUEUE_SZ, "Write queue full\n");
  pthread_mutex_unlock(&ctx->mutex_store);
}

#define MAX_GAP 100ULL
static void writerProcessWaitingQueue(PHY_VARS_NR_UE *UE)
{
  bool found = false;
  openair0_device_t *dev0 = &openair0_dev[UE->rf_map.card];
  re_order_t *ctx = &dev0->reOrder;
  do {
    found = false;
    pthread_mutex_lock(&ctx->mutex_store);
    for (int i = 0; i < WRITE_QUEUE_SZ; i++) {
      if (ctx->queue[i].active && llabs(ctx->queue[i].timestamp - ctx->nextTS) < MAX_GAP) {
        openair0_timestamp_t timestamp = ctx->queue[i].timestamp;
        LOG_D(HW, "Dequeue write for TS: %lu\n", timestamp);
        int nsamps = ctx->queue[i].nsamps;
        int nbAnt = ctx->queue[i].nbAnt;
        int flags = ctx->queue[i].flags;
        void *txp[NB_ANTENNAS_TX];
        AssertFatal(nbAnt <= NB_ANTENNAS_TX, "");
        for (int j = 0; j < nbAnt; j++)
          txp[j] = ctx->queue[i].txp[j];
        ctx->queue[i].active = false;
        pthread_mutex_unlock(&ctx->mutex_store);
        found = true;
        if (flags || IS_SOFTMODEM_RFSIM) {
          int wroteSamples = nr_ue_ru_write(UE, timestamp, txp, nsamps, nbAnt, flags);
          if (wroteSamples != nsamps)
            LOG_E(HW, "Failed to write to rf\n");
        }
        ctx->nextTS = timestamp + nsamps;
        pthread_mutex_lock(&ctx->mutex_store);
      }
    }
    pthread_mutex_unlock(&ctx->mutex_store);
  } while (found);
}

// We assume the data behind *txp are permanently allocated
// When we will go further, we can remove all RC.xxx.txdata buffers in xNB, in UE
// but to make zerocopy and agnostic design, we need to make a proper ring buffer with mutex protection
// mutex (or atomic flags) will be mandatory because this out order system root cause is there are several writer threads

int nr_ue_ru_write_reorder(PHY_VARS_NR_UE *UE, openair0_timestamp_t timestamp, void **txp, int nsamps, int nbAnt, int flags)
{
  int wroteSamples = 0;
  openair0_device_t *dev0 = &openair0_dev[UE->rf_map.card];
  re_order_t *ctx = &dev0->reOrder;
  LOG_D(HW, "received write order ts: %lu, nb samples %d, next ts %luflags %d\n", timestamp, nsamps, timestamp + nsamps, flags);
  if (!ctx->initDone) {
    ctx->nextTS = timestamp;
    pthread_mutex_init(&ctx->mutex_write, NULL);
    pthread_mutex_init(&ctx->mutex_store, NULL);
    ctx->initDone = true;
  }
  if (pthread_mutex_trylock(&ctx->mutex_write) == 0) {
    // We have the write exclusivity
    if (llabs(timestamp - ctx->nextTS) < MAX_GAP) { // We are writing in sequence of the previous write
      if (flags || IS_SOFTMODEM_RFSIM)
        wroteSamples = nr_ue_ru_write(UE, timestamp, txp, nsamps, nbAnt, flags);
      else
        wroteSamples = nsamps;
      ctx->nextTS = timestamp + nsamps;

    } else {
      writerEnqueue(ctx, timestamp, txp, nsamps, nbAnt, flags);
    }
    writerProcessWaitingQueue(UE);
    pthread_mutex_unlock(&ctx->mutex_write);
    return wroteSamples ? wroteSamples : nsamps;
  }
  writerEnqueue(ctx, timestamp, txp, nsamps, nbAnt, flags);
  if (pthread_mutex_trylock(&ctx->mutex_write) == 0) {
    writerProcessWaitingQueue(UE);
    pthread_mutex_unlock(&ctx->mutex_write);
  }
  return nsamps;
}

void nr_ue_ru_write_reorder_clear_context(PHY_VARS_NR_UE *UE)
{
  LOG_I(HW, "received write reorder clear context\n");
  openair0_device_t *dev0 = &openair0_dev[UE->rf_map.card];
  re_order_t *ctx = &dev0->reOrder;
  if (!ctx->initDone)
    return;
  if (pthread_mutex_trylock(&ctx->mutex_write) != 0)
    LOG_E(HW, "nr_ue_ru_write_reorder_clear_context call while still writing on the device\n");
  pthread_mutex_destroy(&ctx->mutex_write);
  pthread_mutex_lock(&ctx->mutex_store);
  for (int i = 0; i < WRITE_QUEUE_SZ; i++)
    ctx->queue[i].active = false;
  pthread_mutex_unlock(&ctx->mutex_store);
  pthread_mutex_destroy(&ctx->mutex_store);
  ctx->initDone = false;
}
