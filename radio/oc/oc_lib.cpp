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
//#define LOG_E(m, a...) printf(a)
#include "common_lib.h"
#endif
#include "system.h"
#include <sys/resource.h>

#define DEVICE_WRITE_DEFAULT "/dev/xdma0_h2c_0"
#define DEVICE_READ_DEFAULT "/dev/xdma0_c2h_0"
#define OC_BUFFER 8192 * 16 // in bytes
#define SAMPLE_BUF (OC_BUFFER / sizeof(c16_t)) // in samples

static const uint64_t magic = 0xA5A5A5A5A5A5A5A5;

typedef struct rxHeader {
  uint64_t header;
  uint64_t sdrStatus;
  int64_t timestamp;
  uint64_t trailer;
} rxHeader_t;

typedef struct {
  char filename_write[FILENAME_MAX];
  char filename_read[FILENAME_MAX];
  int fd_write;
  int fd_read;
  int num_underflows;
  int num_overflows;
  int num_seq_errors;
  int64_t tx_count;
  int64_t rx_count;
  int wait_for_first_pps;
  int use_gps;
  openair0_timestamp rx_timestamp;
  openair0_timestamp tx_ts;
  c16_t **tx_block;
  size_t tx_block_sz;
  bool first_tx;
  bool rxMagicFound;
} oc_state_t;

typedef struct {
  openair0_device *rfdevice;
  int antennas;
  int dft_sz;
} threads_t;

static int check_ref_locked(oc_state_t *s)
{
  return 0;
}

static int sync_to_gps(openair0_device *device)
{
  return 0;
}

void *write_thread(void *arg)
{
  // threads_t params = *(threads_t *)arg;

  return NULL;
}

void *read_thread(void *arg)
{
  // threads_t params = *(threads_t *)arg;

  return NULL;
}
#if 0
  //TEST 4G 20MHz
  int nsamps2 = (nsamps*4) / 8 ;
  simde__m256i buff_tx[nsamps2];
  simde__m256i *out=buff_tx;
  int *in=(int *)buff[0];
  // bring RX data into 12 LSBs for softmodem RX
  for (uint j = 0; j < nsamps/2; j++) {
    simde__m256i tmp=simde_mm256_set_epi32(in[1],in[1],in[1],in[1], in[0],in[0],in[0],in[0]);
    in+=2;
    *out++ = simde_mm256_slli_epi16(tmp, 6);
  }
#endif

static int32_t signalEnergy(int32_t *input, uint32_t length)
{
  // init
  simde__m128 mm0 = simde_mm_setzero_ps();

  // Acc
  for (uint32_t i = 0; i < (length >> 2); i++) {
    simde__m128i in = simde_mm_loadu_si128((simde__m128i *)input);
    mm0 = simde_mm_add_ps(mm0, simde_mm_cvtepi32_ps(simde_mm_madd_epi16(in, in)));
    input += 4;
  }

  // leftover
  float leftover_sum = 0;
  c16_t *leftover_input = (c16_t *)input;
  uint16_t lefover_count = length - ((length >> 2) << 2);
  for (int32_t i = 0; i < lefover_count; i++) {
    leftover_sum += leftover_input[i].r * leftover_input[i].r + leftover_input[i].i * leftover_input[i].i;
  }

  // Ave
  float sums[4];
  simde_mm_store_ps(sums, mm0);
  return (uint32_t)((sums[0] + sums[1] + sums[2] + sums[3] + leftover_sum) / (float)length);
}

// DC-filter: 0 will be done in FPGA after seeing 128-consecutive samples having the same value
static inline void write_block(oc_state_t *s)
{
  uint64_t st = rdtsc_oai();
  size_t wrote = write(s->fd_write, s->tx_block[0], SAMPLE_BUF * sizeof(c16_t));
  uint64_t end = rdtsc_oai();
  // if (end-st > 100*5000)
  // LOG_E(HW,"one write to xdma took %ld µs, ts:%lu\n", (end-st)/5000, s->tx_ts);
  /*
  static uint64_t  old;
  if (old-st > 5000*500)
    LOG_E(HW,"we come back to writer after: %ld µs\n", (old-st)/5000);
  old=st;
  */
  wrote /= sizeof(c16_t);
  if (wrote != SAMPLE_BUF)
    LOG_E(HW, "write to SDR failed, request: %lu, wrote %ld\n", SAMPLE_BUF, wrote / sizeof(c16_t));
  if (wrote < 0)
    LOG_E(HW, "write to %s failed, errno %d:%s\n", s->filename_write, errno, strerror(errno));
  s->tx_ts += wrote;
  s->tx_block_sz = 0;
  s->tx_count++;
  LOG_D(HW, "wrote at ts: %lu, energy: %u\n", s->tx_ts, signalEnergy((int32_t *)s->tx_block[0], SAMPLE_BUF));
}

static int oc_write(openair0_device *device, openair0_timestamp timestamp, void **buff, int nsamps, int cc, int flags)
{
  oc_state_t *s = (oc_state_t *)device->priv;
  timestamp -= device->openair0_cfg->command_line_sample_advance + device->openair0_cfg->tx_sample_advance;
  c16_t *in = (c16_t *)buff[0];

  if (s->first_tx) {
    s->tx_ts = timestamp;
    s->first_tx = false;
  }

  int64_t gap = timestamp - s->tx_ts;
  if (gap < 0) {
    LOG_E(HW, "out of sequence\n");
    gap = 0;
  }
  if (gap)
    LOG_D(HW, "gap of %ld\n", gap);

  while (gap) {
    int tmp = std::min(gap, (int64_t)SAMPLE_BUF - (int64_t)s->tx_block_sz);
    memset(s->tx_block[0] + s->tx_block_sz, 0, tmp * sizeof(*in));
    gap -= tmp;
    s->tx_block_sz += tmp;
    if (s->tx_block_sz == SAMPLE_BUF)
      write_block(s);
  }
  int wr_sz = nsamps;
  while (wr_sz) {
    int tmp = std::min((long unsigned int)wr_sz, SAMPLE_BUF - s->tx_block_sz);
    simde__m256i *sig = (simde__m256i *)(s->tx_block[0] + s->tx_block_sz);
    for (int j = 0; j < tmp; j += 8)
      *sig++ = simde_mm256_slli_epi16(*(simde__m256i *)(in + j), 4);
    // memcpy(s->tx_block[0] + s->tx_block_sz, in, tmp * sizeof(*in));
    wr_sz -= tmp;
    s->tx_block_sz += tmp;
    if (s->tx_block_sz == SAMPLE_BUF)
      write_block(s);
  }
  s->tx_ts = timestamp + nsamps;
  return nsamps;
}

static int oc_read(openair0_device *device, openair0_timestamp *ptimestamp, void **buff, int nsamps, int cc)
{
  oc_state_t *s = (oc_state_t *)device->priv;
  rxHeader_t rx = {0};

  if (!s->rxMagicFound) {
    while (rx.header != magic) {
      int ret = read(s->fd_read, &rx, sizeof(rx));
    }
    printf("found magic, rx can start\n");
    s->rx_timestamp = rx.timestamp;
    s->rxMagicFound = true;
  }

  // need to read, test, skip header at given rate
  // logic failure: if i read a header, i should get samples after, not a second header
  c16_t tmp[nsamps + sizeof(rxHeader_t) / sizeof(c16_t)];
  int bytes_received = read(s->fd_read, &tmp, sizeof(tmp));
  if (bytes_received % sizeof(c16_t))
    printf("Error in read, size is not a number of samples %d\n", bytes_received);
  memcpy(&rx, tmp, sizeof(rx));
  if (rx.header != magic) {
    printf("wrong header %lx, dropping packet %lu\n", rx.header, s->rx_count);
    s->rxMagicFound = false;
    return 0;
  }
  if (s->rx_timestamp != rx.timestamp)
    printf("expected ts: %lu got %lu, diff %ld\n", s->rx_timestamp, rx.timestamp, (int64_t)rx.timestamp - s->rx_timestamp);

  s->rx_count++;
  *ptimestamp = s->rx_timestamp;
  memcpy(buff[0], tmp + sizeof(rxHeader_t) / sizeof(c16_t), nsamps * sizeof(c16_t));
  s->rx_timestamp = rx.timestamp + nsamps;
  return nsamps; // fixme: return actual status
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
  s->tx_block_sz = 0;
  s->first_tx = true;
  ;
  int nb_tx = device->openair0_cfg->tx_num_channels;
  s->tx_block = (c16_t **)malloc(nb_tx * sizeof(*s->tx_block));
  for (int i = 0; i < nb_tx; i++)
    s->tx_block[i] = (c16_t *)malloc16(OC_BUFFER);
  s->wait_for_first_pps = 1;
  s->rx_count = 0;
  s->tx_count = 0;
  s->rx_timestamp = 0;
  s->fd_write = open(s->filename_write, O_RDWR);
  if (s->fd_write < 0) {
    LOG_E(HW, "Open %s failed, errno %d:%s\n", s->filename_write, errno, strerror(errno));
    exit(1);
  }
  s->fd_read = open(s->filename_read, O_RDWR);
  if (s->fd_read < 0) {
    LOG_E(HW, "Open %s failed, errno %d:%s\n", s->filename_read, errno, strerror(errno));
    exit(1);
  }
  /*
  threads_t params = (threads_t){&rfdevice, antennas, DFT};
  pthread_t w_thread;
  threadCreate(&w_thread, write_thread, &params, "write_thr", -1, OAI_PRIORITY_RT);
  pthread_t r_thread;
  threadCreate(&r_thread, read_thread, &params, "read_thr", -1, OAI_PRIORITY_RT);
  */
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
  oc_state_t *s = (oc_state_t *)device->priv;
  int nb_tx = device->openair0_cfg->tx_num_channels;
  if (s && nb_tx > 0) {
    for (int i = 0; i < nb_tx; i++)
      free(s->tx_block[i]);
    free(s->tx_block);
  }
}

extern "C" {
int device_init(openair0_device *device, openair0_config_t *openair0_cfg)
{
  LOG_I(HW, "openair0_cfg->clock_source == '%d' (internal = %d, external = %d)\n", openair0_cfg->clock_source, internal, external);
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

  device->type = USRP_X300_DEV;
  
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

  strcpy(s->filename_write, DEVICE_WRITE_DEFAULT);
  strcpy(s->filename_read, DEVICE_READ_DEFAULT);
  openair0_cfg[0].iq_txshift = 4; // shift
  openair0_cfg[0].iq_rxrescale = 15; // rescale iqs

  return 0;
}
}
