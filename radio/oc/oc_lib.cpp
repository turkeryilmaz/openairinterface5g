/*
 * Licensed by open cells project
 */
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <complex>
#include <fstream>
#include <cmath>
#include <time.h>
#ifdef OAI_INTEGRATION
#include "common/utils/LOG/log.h"
#include "common_lib.h"
#include "assertions.h"
#else
#define LOG_E(m, a...) printf(a)
#include "common_lib.h"
#endif
#include "system.h"
#include <sys/resource.h>

#define DEVICE_NAME_DEFAULT "/dev/xdma0_h2c_0"
#define SIZE_DEFAULT (32)

typedef struct {
  char filename[FILENAME_MAX];
  int fd;
  int num_underflows;
  int num_overflows;
  int num_seq_errors;
  int64_t tx_count;
  int64_t rx_count;
  int wait_for_first_pps;
  int use_gps;
  openair0_timestamp rx_timestamp;
} oc_state_t;

static int check_ref_locked(oc_state_t *s)
{
  return 0;
}

static int sync_to_gps(openair0_device *device)
{
  return 0;
}

static int oc_write(openair0_device *device, openair0_timestamp timestamp, void **buff, int nsamps, int cc, int flags)
{
  oc_state_t *s = (oc_state_t *)device->priv;
  timestamp -= device->openair0_cfg->command_line_sample_advance + device->openair0_cfg->tx_sample_advance;
  size_t wrote = write(s->fd, buff[0], nsamps * sizeof(c16_t));
  if (wrote != nsamps * sizeof(c16_t)) {
    LOG_E(HW, "write to SDR failed, request: %d, wrote %ld\n", nsamps, wrote / sizeof(c16_t));
    if (wrote < 0)
      LOG_E(HW, "write to %s failed, errno %d:%s\n", DEVICE_NAME_DEFAULT, errno, strerror(errno));
  }
  return wrote;
}

static int oc_read(openair0_device *device, openair0_timestamp *ptimestamp, void **buff, int nsamps, int cc)
{
  oc_state_t *s = (oc_state_t *)device->priv;
  int samples_received = nsamps;
  if (0)
    samples_received = read(s->fd, buff[0], nsamps * sizeof(c16_t));
  return samples_received;
}

static int oc_set_freq(openair0_device *device, openair0_config_t *openair0_cfg)
{
  // oc_state_t *s = (oc_state_t *)device->priv;
  printf("Setting TX Freq %f, RX Freq %f, tune_offset: %f\n",
         openair0_cfg[0].tx_freq[0],
         openair0_cfg[0].rx_freq[0],
         openair0_cfg[0].tune_offset);
  return 0;
}

static int oc_set_gains(openair0_device *device, openair0_config_t *openair0_cfg)
{
  // oc_state_t *s = (oc_state_t *)device->priv;
  LOG_I(HW, "Setting RX gain to %f \n", openair0_cfg[0].rx_gain[0]);
  return 0;
}

static int oc_stop(openair0_device *device)
{
  return 0;
}

void set_rx_gain_offset(openair0_config_t *openair0_cfg, int chain_index, int bw_gain_adjust)
{
  int i = 0;
  // loop through calibration table to find best adjustment factor for RX frequency
  double min_diff = 6e9, diff, gain_adj = 0.0;

  if (bw_gain_adjust == 1) {
    switch ((int)openair0_cfg[0].sample_rate) {
      case 46080000:
        break;

      case 30720000:
        break;

      case 23040000:
        gain_adj = 1.25;
        break;

      case 15360000:
        gain_adj = 3.0;
        break;

      case 7680000:
        gain_adj = 6.0;
        break;

      case 3840000:
        gain_adj = 9.0;
        break;

      case 1920000:
        gain_adj = 12.0;
        break;

      default:
        LOG_E(HW, "unknown sampling rate %d\n", (int)openair0_cfg[0].sample_rate);
        // exit(-1);
        break;
    }
  }

  while (openair0_cfg->rx_gain_calib_table[i].freq > 0) {
    diff = fabs(openair0_cfg->rx_freq[chain_index] - openair0_cfg->rx_gain_calib_table[i].freq);
    LOG_I(HW,
          "cal %d: freq %f, offset %f, diff %f\n",
          i,
          openair0_cfg->rx_gain_calib_table[i].freq,
          openair0_cfg->rx_gain_calib_table[i].offset,
          diff);

    if (min_diff > diff) {
      min_diff = diff;
      openair0_cfg->rx_gain_offset[chain_index] = openair0_cfg->rx_gain_calib_table[i].offset + gain_adj;
    }

    i++;
  }
}

static int oc_get_stats(openair0_device *device)
{
  return (0);
}

static int oc_reset_stats(openair0_device *device)
{
  return (0);
}

int oc_write_init(openair0_device *device)
{
  LOG_E(HW, "trx_write_init should not be called, and is a design error even for USRP\n");
  return -1;
}

static int oc_start(openair0_device *device)
{
  oc_state_t *s = (oc_state_t *)device->priv;
  s->wait_for_first_pps = 1;
  s->rx_count = 0;
  s->tx_count = 0;
  s->rx_timestamp = 0;
  s->fd = open(s->filename, O_RDWR);
  if (s->fd < 0) {
    LOG_E(HW, "Open %s failed, errno %d:%s\n", s->filename, errno, strerror(errno));
    exit(1);
  }
  oc_set_gains(device, device->openair0_cfg);
  oc_set_freq(device, device->openair0_cfg);
  sync_to_gps(device);
  check_ref_locked(s);
  // Fixme: set sampling rate, lack of API in OAI
  return 0;
}

static void oc_end(openair0_device *device)
{
  if (device == NULL)
    return;
}

extern "C" {
int device_init(openair0_device *device, openair0_config_t *openair0_cfg)
{
  LOG_I(HW, "openair0_cfg[0].sdr_addrs == '%s'\n", openair0_cfg[0].sdr_addrs);
  LOG_I(HW,
        "openair0_cfg[0].clock_source == '%d' (internal = %d, external = %d)\n",
        openair0_cfg[0].clock_source,
        internal,
        external);
  oc_state_t *s;
  if (device->priv == NULL) {
    s = (oc_state_t *)calloc(1, sizeof(oc_state_t));
    device->priv = s;
    AssertFatal(s != NULL, "OC device: memory allocation failure\n");
  } else {
    LOG_E(HW, "multiple calls to device init detected\n");
    return 0;
  }

  device->openair0_cfg = openair0_cfg;
  device->trx_start_func = oc_start;
  device->trx_get_stats_func = oc_get_stats;
  device->trx_reset_stats_func = oc_reset_stats;
  device->trx_end_func = oc_end;
  device->trx_stop_func = oc_stop;
  device->trx_set_freq_func = oc_set_freq;
  device->trx_set_gains_func = oc_set_gains;
  device->trx_write_init = oc_write_init;
  device->trx_write_func = oc_write;
  device->trx_read_func = oc_read;
  if (device->openair0_cfg->recplay_mode == RECPLAY_RECORDMODE) {
    std::cerr << "OC device initialized in subframes record mode" << std::endl;
  }

  struct {
    int sample_rate;
    int tx_sample_advance;
    double tx_bw;
    double rx_bw;
  } config_table[] = {{245760000, 15, 200e6, 200e6},
                      {184320000, 15, 100e6, 100e6},
                      {122880000, 15, 80e6, 80e6},
                      {92160000, 15, 60e6, 60e6},
                      {61440000, 15, 40e6, 40e6},
                      {46080000, 15, 40e6, 40e6},
                      {30720000, 15, 40e6, 40e6},
                      {23040000, 15, 20e6, 20e6},
                      {15360000, 15, 10e6, 10e6},
                      {7680000, 50, 5e6, 5e6},
                      {1920000, 50, 1.25e6, 1.25e6}};
  size_t i = 0;
  for (; i < sizeofArray(config_table); i++)
    if (config_table[i].sample_rate == (int)openair0_cfg[0].sample_rate) {
      openair0_cfg[0].tx_sample_advance = config_table[i].tx_sample_advance;
      openair0_cfg[0].tx_bw = config_table[i].tx_bw;
      openair0_cfg[0].rx_bw = config_table[i].rx_bw;
      break;
    }
  if (i == sizeofArray(config_table)) {
    LOG_E(HW, "unknown sampling rate: %d\n", (int)openair0_cfg[0].sample_rate);
    exit(-1);
  }

  strcpy(s->filename, DEVICE_NAME_DEFAULT);
  openair0_cfg[0].iq_txshift = 4; // shift
  openair0_cfg[0].iq_rxrescale = 15; // rescale iqs

  return 0;
}
}
