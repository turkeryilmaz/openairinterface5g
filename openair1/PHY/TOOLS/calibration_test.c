#include <stdint.h>
#include <openair1/PHY/impl_defs_top.h>
#include <radio/COMMON/common_lib.h>
#include <executables/softmodem-common.h>
#include <openair1/PHY/TOOLS/calibration_scope.h>
#include "nfapi/oai_integration/vendor_ext.h"


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
int main(int argc, char **argv) {
  ///static configuration for NR at the moment
  if ((uniqCfg = load_configmodule(argc, argv, CONFIG_ENABLECMDLINEONLY)) == NULL) {
    exit_fun("[SOFTMODEM] Error, configuration module init failed\n");
  }
  set_softmodem_sighandler();
  setvbuf(stdout, NULL, _IONBF, 0);
  setvbuf(stderr, NULL, _IONBF, 0);
  logInit();
   paramdef_t cmdline_params[] = CMDLINE_PARAMS_DESC_GNB ;

  CONFIG_SETRTFLAG(CONFIG_NOEXITONHELP);
  get_common_options(uniqCfg);
  config_process_cmdline(uniqCfg, cmdline_params, sizeofArray(cmdline_params), NULL);
  CONFIG_CLEARRTFLAG(CONFIG_NOEXITONHELP);
  lock_memory_to_ram();

  int N_RB=50;
  int sampling_rate=30.72e6;
  int DFT=2048;
  int TxAdvanceInDFTSize=12;
  int antennas=1;
  uint64_t freq = 2420.0e6;
  int rxGain=90;
  int txGain = 0;
  int filterBand=40e6;
  char * usrp_addrs="type=b200";

  openair0_config_t openair0_cfg = {
      .log_level = 0,
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
      .threequarter_fs = 0,
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
  void ** samplesRx = (void **)malloc16(antennas* sizeof(c16_t *) );
  void ** samplesTx = (void **)malloc16(antennas* sizeof(c16_t *) );

  for (int i=0; i<antennas; i++) {
    samplesRx[i] = (int32_t *)malloc16_clear( DFT*sizeof(c16_t) );
    samplesTx[i] = (int32_t *)malloc16_clear( DFT*sizeof(c16_t) );
  }
  int fd = -1;
  if (getenv("rftestInputFile")) {
    fd = open(getenv("rftestInputFile"), O_RDONLY);
    AssertFatal(fd >= 0, "%s", strerror(errno));
  } else {
    printf("generate a sinus wave at middle RB");
    load_dftslib();
    c16_t frequency[DFT];
    memset(frequency, 0, sizeof(frequency));
    frequency[DFT / 4] = (c16_t){32767, 0};
    dft(DFT_2048, (int16_t *)frequency, samplesTx[0], 1);
  }

  CalibrationInitScope(samplesRx, &rfdevice);
  openair0_timestamp timestamp=0;
  rfdevice.trx_start_func(&rfdevice);
  
  while(!oai_exit) {
    if (fd >= 0)
      for (int i = 0; i < antennas; i++)
        read(fd, samplesTx[i], DFT * sizeof(c16_t));
    rfdevice.trx_read_func(&rfdevice, &timestamp, samplesRx, DFT, antennas);
    rfdevice.trx_write_func(&rfdevice, timestamp + TxAdvanceInDFTSize * DFT, samplesTx, DFT, antennas, 0);
  }

  return 0;
}
