#include <stdint.h>
#include <openair1/PHY/impl_defs_top.h>
#include <radio/COMMON/common_lib.h>
#include <executables/softmodem-common.h>
#include <openair1/PHY/TOOLS/calibration_scope.h>
#include "nfapi/oai_integration/vendor_ext.h"
#include <arpa/inet.h>

int oai_exit=false;
unsigned int mmapped_dma=0;
uint32_t timing_advance;
int8_t threequarter_fs;
uint64_t downlink_frequency[MAX_NUM_CCs][4];
int32_t uplink_frequency_offset[MAX_NUM_CCs][4];
int cpu_meas_enabled;
THREAD_STRUCT thread_struct;
uint32_t target_ul_mcs = 9;
uint32_t target_dl_mcs = 9;
uint64_t dlsch_slot_bitmap = (1<<1);
uint64_t ulsch_slot_bitmap = (1<<8);
uint32_t target_ul_bw = 50;
uint32_t target_dl_bw = 50;
uint32_t target_dl_Nl;
uint32_t target_ul_Nl;
char *uecap_file;
#include <executables/nr-softmodem.h>

int read_recplayconfig(recplay_conf_t **recplay_conf, recplay_state_t **recplay_state) {return 0;}
void nfapi_setmode(nfapi_mode_t nfapi_mode) {}
void set_taus_seed(unsigned int seed_init){};
// configmodule_interface_t *uniqCfg = NULL;
const int tx_ahead = DFT * 50;
openair0_timestamp rx_timestamp = 0;
openair0_timestamp tx_timestamp = 0;

void *write_thread(void *arg)
{
  threads_t params = *(threads_t *)arg;
  c16_t **samplesTx = params.samplesTx;
  uint64_t ts = 0;
  for (int i = 0; i < params.dft_sz; i++) {
    // Better to select a frequency having an integer division with the sampling rate to avoid having DFT leakage later on
    //  .r = cos and .i = sin -> having a positive spectrum
    //  For negative spectrum -> .r = sin and .i = cos
    samplesTx[0][i].r = 16000 * cos((ts * M_PI * 2 * 30720) / 122880);
    samplesTx[0][i].i = 16000 * sin((ts * M_PI * 2 * 30720) / 122880); // samplesTx[0][i].r;
    // Hamming Window - to allow some pseudo-continuity between batches as this is not a continuously generated signal as in real
    // life
    samplesTx[0][i].r = (samplesTx[0][i].r) * (0.54 - 0.46 * cos(2 * M_PI * 30720 / 122880));
    samplesTx[0][i].i = (samplesTx[0][i].i) * (0.54 - 0.46 * cos(2 * M_PI * 30720 / 122880));
    ts++;
  }
  double avg = 0;
  for (int i = 0; i < params.dft_sz; i++) {
    avg += sqrt(squaredMod(samplesTx[0][i]));
  }
  printf("avg: %f \n", avg / params.dft_sz);
  uint64_t count = 0;
  struct timespec last_second;
  clock_gettime(CLOCK_REALTIME, &last_second);

  openair0_timestamp last_tx_timestamp = 0, new_tx = 0;

  while (!oai_exit) {
    /*
    do {
      pthread_mutex_lock(&params.txMutex);
      printf("write got lock\n");
      new_tx = tx_timestamp & ~31;
      pthread_mutex_unlock(&params.txMutex);
      if (new_tx == last_tx_timestamp)
        usleep(5);
    } while (last_tx_timestamp == new_tx);
    */
    last_tx_timestamp = new_tx & ~31;
    params.rfdevice->trx_write_func(params.rfdevice, new_tx + tx_ahead, (void **)samplesTx, params.dft_sz, params.antennas, 0);
    count++;
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    if (now.tv_sec != last_second.tv_sec) {
      printf("write thread wrote %lu times in one second\n", count);
      last_second = now;
      count = 0;
    }
  }
  return NULL;
}

void *read_thread(void *arg)
{
  threads_t params = *(threads_t *)arg;
  c16_t **samplesRx = params.samplesRx;
  uint64_t count = 0;
  struct timespec last_second;
  clock_gettime(CLOCK_REALTIME, &last_second);
  while (!oai_exit) {
    pthread_mutex_lock(&params.rxMutex);
    int ret = params.rfdevice->trx_read_func(params.rfdevice, &rx_timestamp, (void **)samplesRx, params.dft_sz, params.antennas);
    pthread_mutex_unlock(&params.rxMutex);
    if (ret != params.dft_sz)
      printf("read of :%d\n", ret);
    count++;
    pthread_mutex_lock(&params.txMutex);
    tx_timestamp = rx_timestamp;
    pthread_mutex_unlock(&params.txMutex);
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    if (now.tv_sec != last_second.tv_sec) {
      printf("read thread got %lu blocks in one second, samples per block: %d\n", count, ret);
      last_second = now;
      count = 0;
      FILE *fd = fopen("trace.iq", "w+");
      if (!fd)
        abort();

      /* We should advance +1 only if header was detected in previous steps
       * Which will make the read working even without timestamped rxdata
       */
      c16_t *s = samplesRx[0]; /* Exclude the header from the samples */
      for (int i = 0; i < ret; i++) {
        /* We need to throw the entie 256-bits word if we detect the 64-bits header.
         * This may happens when receiving a big packet size in chuncks.
         */
        fprintf(fd, "%d %d %d\n", i, s[i].r, s[i].i);
      }
    }
  }
  return NULL;
}

int main(int argc, char **argv)
{
  /// static configuration for NR at the moment
  if ((uniqCfg = load_configmodule(argc, argv, CONFIG_ENABLECMDLINEONLY)) == NULL) {
    exit_fun("[SOFTMODEM] Error, configuration module init failed\n");
  }
  set_softmodem_sighandler();
  setvbuf(stdout, NULL, _IONBF, 0);
  setvbuf(stderr, NULL, _IONBF, 0);
  logInit();
  paramdef_t cmdline_params[] = CMDLINE_PARAMS_DESC_GNB;

  CONFIG_SETRTFLAG(CONFIG_NOEXITONHELP);
  get_common_options(uniqCfg);
  config_process_cmdline(uniqCfg, cmdline_params, sizeofArray(cmdline_params), NULL);
  CONFIG_CLEARRTFLAG(CONFIG_NOEXITONHELP);
  lock_memory_to_ram();

  int sampling_rate = 30.72e6;

  int antennas = 1;
  uint64_t freq = 2420.0e6;
  int rxGain = 90;
  int txGain = 0;
  int filterBand = 40e6;

  openair0_config_t openair0_cfg = {
      .duplex_mode = 0,
      .sample_rate = sampling_rate,
      .tx_sample_advance = 0,
      .rx_num_channels = antennas,
      .tx_num_channels = antennas,
      .rx_freq = {freq, freq, freq, freq},
      .tx_freq = {freq, freq, freq, freq},
      .rx_gain_calib_table = NULL,
      .rx_gain = {rxGain, rxGain, rxGain, rxGain},
      .tx_gain = {txGain, txGain, txGain, txGain},
      .rx_bw = filterBand,
      .tx_bw = filterBand,
      .clock_source = internal, // internal gpsdo external
      .time_source = internal, // internal gpsdo external
      .autocal = {0},
      //! rf devices work with x bits iqs when oai have its own iq format
      //! the two following parameters are used to convert iqs
      .iq_txshift = 0,
      .iq_rxrescale = 0,
      .configFilename = "",
      .recplay_mode = 0,
      .recplay_conf = NULL,
  };
  //-----------------------
  openair0_device rfdevice = {
      /*!brief Type of this device */
      .type = NONE_DEV,
      /*!brief Transport protocol type that the device supports (in case I/Q samples need to be transported) */
      .transp_type = NONE_TP,
      /*!brief Type of the device's host (RAU/RRU) */
      .host_type = MIN_HOST_TYPE,
      /* !brief RF frontend parameters set by application */
      .openair0_cfg = NULL, // set by device_init
      /* !brief ETH params set by application */
      .eth_params = NULL,
      //! record player data, definition in record_player.h
      .recplay_state = NULL,
      /* !brief Indicates if device already initialized */
      .is_init = 0,
      /*!brief Can be used by driver to hold internal structure*/
      .priv = NULL,
  };

  openair0_device_load(&rfdevice, &openair0_cfg);

  printf("generate a sinus wave at middle RB");
  load_dftslib();

  c16_t **samplesRx = malloc16(antennas * sizeof(c16_t *));
  for (int i = 0; i < antennas; i++) {
    samplesRx[i] = malloc16_clear(DFT * sizeof(c16_t));
  }
  c16_t **samplesTx = malloc16(antennas * sizeof(c16_t *));
  for (int i = 0; i < antennas; i++) {
    samplesTx[i] = malloc16_clear(DFT * sizeof(c16_t));
  }

  /* scopedata shall be filled from a software FIFO and not directly from the samples */
  threads_t params = (threads_t){&rfdevice, antennas, DFT, samplesRx, samplesTx};
  pthread_mutex_init(&params.rxMutex, NULL);
  pthread_mutex_init(&params.txMutex, NULL);
  CalibrationInitScope(&params);
  rfdevice.trx_start_func(&rfdevice);
  pthread_t w_thread;
  threadCreate(&w_thread, write_thread, &params, "write_thr", -1, OAI_PRIORITY_RT);
  pthread_t r_thread;
  threadCreate(&r_thread, read_thread, &params, "read_thr", -1, OAI_PRIORITY_RT);
  (void)pthread_join(w_thread, NULL);
  (void)pthread_join(r_thread, NULL);
  return 0;
}
