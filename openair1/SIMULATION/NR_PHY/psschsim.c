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

#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include "common/config/config_userapi.h"
#include "common/utils/load_module_shlib.h"
#include "common/utils/LOG/log.h"
#include "common/ran_context.h"
#include "PHY/types.h"
#include "PHY/defs_nr_common.h"
#include "PHY/defs_nr_UE.h"
#include "PHY/defs_gNB.h"
#include "PHY/INIT/phy_init.h"
#include "PHY/NR_REFSIG/refsig_defs_ue.h"
#include "PHY/MODULATION/nr_modulation.h"
#include "PHY/MODULATION/modulation_eNB.h"
#include "PHY/MODULATION/modulation_UE.h"
#include "PHY/NR_TRANSPORT/nr_transport_proto.h"
#include "PHY/NR_TRANSPORT/nr_dlsch.h"
#include "PHY/NR_TRANSPORT/nr_ulsch.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_proto_ue.h"
#include "PHY/phy_vars_nr_ue.h"

#include "SCHED_NR/sched_nr.h"
#include "openair1/SIMULATION/TOOLS/sim.h"
#include "openair1/SIMULATION/RF/rf.h"
#include "openair1/SIMULATION/NR_PHY/nr_unitary_defs.h"
#include "openair1/SIMULATION/NR_PHY/nr_dummy_functions.c"
#include "common/utils/threadPool/thread-pool.h"
#include "openair2/LAYER2/NR_MAC_COMMON/nr_mac_common.h"
#include "executables/nr-uesoftmodem.h"
#include "PHY/impl_defs_top.h"
#include "PHY/MODULATION/modulation_common.h"

#define LDPC_MAX_LIMIT 31
#define DEBUG_NR_PSSCHSIM

// typedef struct {
//   uint8_t priority;
//   uint8_t freq_res;
//   uint8_t time_res;
//   uint8_t period;
//   uint16_t dmrs_pattern;
//   uint8_t mcs;
//   uint8_t beta_offset;
//   uint8_t dmrs_port;
// } SCI_1_A;

typedef struct {
  double scs;
  double bw;
  double fs;
} BW;

// THREAD_STRUCT thread_struct;
// PHY_VARS_NR_UE *txUE;
// PHY_VARS_NR_UE *rxUE;

#define HNA_SIZE 6 * 68 * 384 // [hna] 16 segments, 68*Zc
#define SCI2_LEN_SIZE 35
RAN_CONTEXT_t RC;
double cpuf;
uint16_t NB_UE_INST = 1;
openair0_config_t openair0_cfg[MAX_CARDS];
uint8_t const nr_rv_round_map[4] = {0, 2, 3, 1};

uint64_t get_softmodem_optmask(void) {return 0;}
static softmodem_params_t softmodem_params;
softmodem_params_t *get_softmodem_params(void) {
  return &softmodem_params;
}

void init_downlink_harq_status(NR_DL_UE_HARQ_t *dl_harq) {}

double snr0 = 100;
double snr1 = 2.0;
int slot = 0;
uint8_t snr1set = 0;
int n_trials = 5;
uint8_t n_tx = 1;
uint8_t n_rx = 1;
int ssb_subcarrier_offset = 0;
FILE *input_fd = NULL;
SCM_t channel_model = AWGN;
int N_RB_SL = 106;
int mu = 1;
int loglvl = OAILOG_WARNING;
int seed = 0;
int mcs = 0;
uint16_t node_id = 0;

static void get_sim_cl_opts(int argc, char **argv)
{
    char c;
    while ((c = getopt(argc, argv, "F:d:g:hIL:l:m:M:n:N:o:O:p:P:r:R:s:S:t:x:y:z:")) != -1) {
    switch (c) {
      case 'F':
        input_fd = fopen(optarg, "r");
        if (input_fd == NULL) {
          printf("Problem with filename %s. Exiting.\n", optarg);
          exit(-1);
        }
        break;

      case 'd':
        node_id = atoi(optarg);
        break;

      case 'g':
        switch((char)*optarg) {
          case 'A':
            channel_model=SCM_A;
            break;

          case 'B':
            channel_model=SCM_B;
            break;

          case 'C':
            channel_model=SCM_C;
            break;

          case 'D':
            channel_model=SCM_D;
            break;

          case 'E':
            channel_model=EPA;
            break;

          case 'F':
            channel_model=EVA;
            break;

          case 'G':
            channel_model=ETU;
            break;

          default:
            printf("Unsupported channel model! Exiting.\n");
            exit(-1);
          }
        break;

      case 'L':
        loglvl = atoi(optarg);
        break;

      case 'l':
        slot = atoi(optarg);
        break;

      case 'm':
        mu = atoi(optarg);
        break;

      case 'n':
        n_trials = atoi(optarg);
        break;

      case 'r':
        N_RB_SL = atoi(optarg);
        break;

      case 's':
        snr0 = atof(optarg);
        break;

      case 'S':
        snr1 = atof(optarg);
        snr1set = 1;
        break;

      case 't':
        mcs = atoi(optarg);
        break;

      case 'y':
        n_tx = atoi(optarg);
        if ((n_tx == 0) || (n_tx > 2)) {
          printf("Unsupported number of TX antennas %d. Exiting.\n", n_tx);
          exit(-1);
        }
        break;

      case 'z':
        n_rx = atoi(optarg);
        if ((n_rx == 0) || (n_rx > 2)) {
          printf("Unsupported number of RX antennas %d. Exiting.\n", n_rx);
          exit(-1);
        }
        break;

      default:
      case 'h':
          printf("%s -h(elp) -g channel_model -n n_frames -s snr0 -S snr1 -p(extended_prefix) -y TXant -z RXant -M -N cell_id -R -F input_filename -m -l -r\n", argv[0]);
          //printf("%s -h(elp) -p(extended_prefix) -N cell_id -f output_filename -F input_filename -g channel_model -n n_frames -t Delayspread -s snr0 -S snr1 -x transmission_mode -y TXant -z RXant -i Intefrence0 -j Interference1 -A interpolation_file -C(alibration offset dB) -N CellId\n", argv[0]);
          printf("-h This message\n");
          printf("-g [A,B,C,D,E,F,G] Use 3GPP SCM (A,B,C,D) or 36-101 (E-EPA,F-EVA,G-ETU) models (ignores delay spread and Ricean factor)\n");
          printf("-n Number of frames to simulate\n");
          printf("-s Starting SNR, runs from SNR0 to SNR0 + 5 dB.  If n_frames is 1 then just SNR is simulated\n");
          printf("-S Ending SNR, runs from SNR0 to SNR1\n");
          printf("-p Use extended prefix mode\n");
          printf("-y Number of TX antennas used in eNB\n");
          printf("-z Number of RX antennas used in UE\n");
          printf("-W number of layer\n");
          printf("-r N_RB_SL\n");
          printf("-F Input filename (.txt format) for RX conformance testing\n");
          printf("-m MCS\n");
          printf("-l number of symbol\n");
          printf("-r number of RB\n");
        exit (-1);
        break;
    }
  }
}


void nr_phy_config_request_psschsim(PHY_VARS_NR_UE *ue,
                                    int N_RB_SL,
                                    int mu,
                                    uint64_t position_in_burst)
{
  NR_DL_FRAME_PARMS *fp                 = &ue->frame_parms;
  fapi_nr_config_request_t *nrUE_config = &ue->nrUE_config;
  uint64_t rev_burst=0;
  for (int i = 0; i < 64; i++)
    rev_burst |= (((position_in_burst >> (63 - i)) & 0x01) << i);

  nrUE_config->ssb_config.scs_common               = mu;
  nrUE_config->ssb_table.ssb_subcarrier_offset     = ssb_subcarrier_offset;
  nrUE_config->ssb_table.ssb_offset_point_a        = (N_RB_SL - 11) >> 1;
  nrUE_config->ssb_table.ssb_mask_list[1].ssb_mask = (rev_burst)&(0xFFFFFFFF);
  nrUE_config->ssb_table.ssb_mask_list[0].ssb_mask = (rev_burst >> 32)&(0xFFFFFFFF);
  nrUE_config->cell_config.frame_duplex_type       = TDD;
  nrUE_config->ssb_table.ssb_period                = 1; //10ms
  nrUE_config->carrier_config.sl_grid_size[mu]     = N_RB_SL;
  nrUE_config->carrier_config.num_tx_ant           = fp->nb_antennas_tx;
  nrUE_config->carrier_config.num_rx_ant           = fp->nb_antennas_rx;
  nrUE_config->tdd_table.tdd_period                = 0;

  ue->mac_enabled = 1;
  ue->is_synchronized_sl = 1;
  if (mu == 0) {
    fp->dl_CarrierFreq = 2600000000;
    fp->ul_CarrierFreq = 2600000000;
    fp->nr_band = 38;
  } else if (mu == 1) {
    fp->dl_CarrierFreq = 3600000000;
    fp->ul_CarrierFreq = 3600000000;
    fp->sl_CarrierFreq = 2600000000;
    nrUE_config->carrier_config.sl_frequency               = fp->sl_CarrierFreq / 1000;
    fp->nr_band = 78;
  } else if (mu == 3) {
    fp->dl_CarrierFreq = 27524520000;
    fp->ul_CarrierFreq = 27524520000;
    fp->nr_band = 261;
  }

  fp->threequarter_fs = 0;
  nrUE_config->carrier_config.sl_bandwidth = config_bandwidth(mu, N_RB_SL, fp->nr_band);

  nr_init_frame_parms_ue(fp, nrUE_config, fp->nr_band);
  fp->ofdm_offset_divisor = UINT_MAX;
  ue->configured = 1;
  LOG_I(NR_PHY, "tx UE configured\n");
}

void set_sci(SCI_1_A *sci, uint8_t mcs) {
  sci->period = 0;
  sci->dmrs_pattern = 0b0001000001000; // LSB is slot 1 and MSB is slot 13
  sci->beta_offset = 0;
  sci->dmrs_port = 0;
  sci->priority = 0;
  sci->freq_res = 1;
  sci->time_res = 1;
  sci->mcs = mcs;
}

void set_fs_bw(PHY_VARS_NR_UE *UE, int mu, int N_RB, BW *bw_setting) {
  double scs = 0, fs = 0, bw = 0;
  switch (mu) {
    case 1:
      scs = 30000;
      UE->frame_parms.Lmax = 1;
      if (N_RB == 217) {
        fs = 122.88e6;
        bw = 80e6;
      }
      else if (N_RB == 245) {
        fs = 122.88e6;
        bw = 90e6;
      }
      else if (N_RB == 273) {
        fs = 122.88e6;
        bw = 100e6;
      }
      else if (N_RB == 106) {
        fs = 61.44e6;
        bw = 40e6;
      }
      else if (N_RB == 52) {
        fs = 30.72e6;
        bw = 20e6;
      }
      else AssertFatal(1 == 0, "Unsupported numerology for mu %d, N_RB %d\n", mu, N_RB);
      break;
    case 3:
      UE->frame_parms.Lmax = 64;
      scs = 120000;
      if (N_RB == 66) {
        fs = 122.88e6;
        bw = 100e6;
      }
      else AssertFatal(1 == 0, "Unsupported numerology for mu %d, N_RB %d\n", mu, N_RB);
      break;
    default:
      AssertFatal(1 == 0, "Unsupported numerology for mu %d, N_RB %d\n", mu, N_RB);
      break;
  }
  bw_setting->scs = scs;
  bw_setting->fs = fs;
  bw_setting->bw = bw;
  return;
}

nrUE_params_t nrUE_params;
nrUE_params_t *get_nrUE_params(void) {
  return &nrUE_params;
}

int main(int argc, char **argv)
{
  get_softmodem_params()->sl_mode = 2;
  if (load_configmodule(argc, argv, CONFIG_ENABLECMDLINEONLY) == 0) {
    exit_fun("[NR_PSSCHSIM] Error, configuration module init failed\n");
  }
  get_sim_cl_opts(argc, argv);
  char user_msg[128] = "EpiScience";
  get_softmodem_params()->sl_user_msg = user_msg;
  randominit(0);
  // logging initialization
  logInit();
  set_glog(loglvl);
  load_nrLDPClib(NULL);

  PHY_VARS_NR_UE *txUE = malloc(sizeof(PHY_VARS_NR_UE));
  txUE->sync_ref= true;
  txUE->frame_parms.N_RB_SL = N_RB_SL;
  txUE->frame_parms.Ncp = NORMAL;
  txUE->frame_parms.nb_antennas_tx = 1;
  txUE->frame_parms.nb_antennas_rx = n_rx;
  txUE->frame_parms.Imcs = mcs;
  txUE->max_ldpc_iterations = 5;

  PHY_VARS_NR_UE *rxUE = malloc(sizeof(PHY_VARS_NR_UE));
  rxUE->sync_ref= false;
  rxUE->frame_parms.nb_antennas_tx = n_tx;
  rxUE->frame_parms.nb_antennas_rx = 1;
  rxUE->frame_parms.Imcs = mcs;
  initTpool("n", &rxUE->threadPool, true);
  initNotifiedFIFO(&rxUE->respDecode);

  uint64_t burst_position = 0x01;
  nr_phy_config_request_psschsim(txUE, N_RB_SL, mu, burst_position);
  nr_phy_config_request_psschsim(rxUE, N_RB_SL, mu, burst_position);

  BW *bw_setting = malloc(sizeof(BW));
  set_fs_bw(txUE, mu, N_RB_SL, bw_setting);

  double DS_TDL = 300e-9; //.03;
  channel_desc_t *UE2UE = new_channel_desc_scm(n_tx, n_rx, channel_model,
                                               bw_setting->fs,
                                               bw_setting->bw,
                                               DS_TDL,
                                               0, 0, 0, 0);

  if (UE2UE == NULL) {
    printf("Problem generating channel model. Exiting.\n");
    free(bw_setting);
    exit(-1);
  }

  if (init_nr_ue_signal(txUE, 1) != 0 || init_nr_ue_signal(rxUE, 1) != 0) {
    printf("Error at UE NR initialization.\n");
    free(bw_setting);
    free(txUE);
    free(rxUE);
    exit(-1);
  }
#ifdef DEBUG_NR_PSSCHSIM
  for (int sf = 0; sf < 2; sf++) {
    txUE->slsch[sf][0] = new_nr_ue_ulsch(N_RB_SL, 8, &txUE->frame_parms);
    if (!txUE->slsch[sf][0]) {
      printf("Can't get ue ulsch structures.\n");
      exit(-1);
    }
  }
#endif
  get_softmodem_params()->sync_ref = false;
  init_nr_ue_transport(txUE);
  get_softmodem_params()->sync_ref = true;
  init_nr_ue_transport(rxUE);

  NR_UE_DLSCH_t *slsch_ue_rx = rxUE->slsch_rx[0][0][0];
  unsigned char harq_pid = 0;
  NR_DL_UE_HARQ_t *harq_process_rxUE = slsch_ue_rx->harq_processes[harq_pid];
  NR_UL_UE_HARQ_t *harq_process_txUE = txUE->slsch[0][0]->harq_processes[harq_pid];
  DevAssert(harq_process_txUE);

  int frame = 0;
  int slot = 0;
  int soffset = (slot & 3) * rxUE->frame_parms.symbols_per_slot * rxUE->frame_parms.ofdm_symbol_size;
  int32_t **txdata = txUE->common_vars.txdata;
  NR_UE_ULSCH_t *slsch_ue = txUE->slsch[0][0];
  crcTableInit();
  nr_ue_set_slsch(&txUE->frame_parms, 0, slsch_ue, frame, slot);
  nr_ue_slsch_tx_procedures(txUE, harq_pid, frame, slot);
  printf("tx is done\n");

  int32_t **rxdataF = rxUE->common_vars.common_vars_rx_data_per_thread[0].rxdataF;
  UE_nr_rxtx_proc_t proc;
  proc.thread_id = 0;

  //unsigned int errors_bit_uncoded = 0;
  unsigned int errors_bit = 0;
  unsigned int n_errors = 0;
  unsigned int n_false_positive = 0;
  unsigned int errors_bit_delta = 0;
  unsigned int num_bytes_to_check = 80;
  //double modulated_input[HNA_SIZE];
  unsigned char test_input_bit[HNA_SIZE];
  //short channel_output_uncoded[HNA_SIZE];
  unsigned char estimated_output_bit[HNA_SIZE];
  double snr_step = 2;
  snr1 = snr1set == 0 ? snr0 + snr_step * 1 : snr1;
  int frame_length_complex_samples = txUE->frame_parms.samples_per_subframe * NR_NUMBER_OF_SUBFRAMES_PER_FRAME;
  double **r_re = malloc(NR_MAX_NB_LAYERS_SL * sizeof(double*));
  double **r_im = malloc(NR_MAX_NB_LAYERS_SL * sizeof(double*));
  for (int i = 0; i < NR_MAX_NB_LAYERS_SL; i++) {
    r_re[i] = malloc16_clear(frame_length_complex_samples * sizeof(double));
    r_im[i] = malloc16_clear(frame_length_complex_samples * sizeof(double));
  }
  get_softmodem_params()->node_number = node_id;
  nr_ue_set_slsch_rx(rxUE, 0);
  for (double SNR = snr0; SNR < snr1; SNR += snr_step) {
    n_errors = 0;
    n_false_positive = 0;
    errors_bit = 0;

    for (int trial = 0; trial < n_trials; trial++) {
      for (int i = 0; i < frame_length_complex_samples; i++) {
        for (int aa = 0; aa < txUE->frame_parms.nb_antennas_tx; aa++) {
          r_re[aa][i] = ((double)(((short *)txdata[aa]))[(i << 1)]);
          r_im[aa][i] = ((double)(((short *)txdata[aa]))[(i << 1) + 1]);
        }
      }

      for (int i = 0; i < frame_length_complex_samples; i++) {
        double sigma2_dB = 20 * log10((double)AMP / 4) - SNR;
        double sigma2 = pow(10, sigma2_dB / 10);
        for (int aa = 0; aa < rxUE->frame_parms.nb_antennas_rx; aa++) {
          ((short*) rxUE->common_vars.rxdata[aa])[2 * i] = (short) ((r_re[aa][i] + sqrt(sigma2 / 2) * gaussdouble(0.0, 1.0)));
          ((short*) rxUE->common_vars.rxdata[aa])[2 * i + 1] = (short) ((r_im[aa][i] + sqrt(sigma2 / 2) * gaussdouble(0.0, 1.0)));
        }
      }
#ifdef DEBUG_NR_PSSCHSIM
      char buffer1[rxUE->frame_parms.ofdm_symbol_size * 4];
      for (int i = 0; i < 13; i++) {
        bzero(buffer1, sizeof(buffer1));
        LOG_D(PHY, "Slot %d, RXUE Symbol[%d]:  %s\n",
              slot, rxUE->frame_parms.ofdm_symbol_size * i,
              hexdump((int16_t *)&rxUE->common_vars.rxdata[0][rxUE->frame_parms.ofdm_symbol_size * i],
                      rxUE->frame_parms.ofdm_symbol_size * 4, buffer1, sizeof(buffer1)));
      }
#endif
      for (int aa = 0; aa < rxUE->frame_parms.nb_antennas_rx; aa++) {
        for (int ofdm_symbol = 0; ofdm_symbol < NR_NUMBER_OF_SYMBOLS_PER_SLOT; ofdm_symbol++) {
            nr_slot_fep_ul(&rxUE->frame_parms, rxUE->common_vars.rxdata[aa], &rxdataF[aa][soffset], ofdm_symbol, slot, 0);
        }
        apply_nr_rotation_ul(&rxUE->frame_parms, rxdataF[aa], slot, 0, NR_NUMBER_OF_SYMBOLS_PER_SLOT, link_type_sl);
      }
      uint32_t ret = nr_ue_slsch_rx_procedures(rxUE,
                                               harq_pid,
                                               frame,
                                               slot,
                                               rxdataF,
                                               harq_process_txUE->B_multiplexed,
                                               txUE->slsch[0][0]->Nidx,
                                               &proc);

      bool polar_decoded = (ret < LDPC_MAX_LIMIT) ? true : false;
      if (ret != -1) {
        errors_bit_delta = 0;
        bool payload_type_string = false;
        for (int i = 0; i < num_bytes_to_check; i++) {
          estimated_output_bit[i] = (harq_process_rxUE->b[i / 8] & (1 << (i & 7))) >> (i & 7);
          test_input_bit[i] = (txUE->slsch[0][0]->harq_processes[harq_pid]->a[i / 8] & (1 << (i & 7))) >> (i & 7); // Further correct for multiple segments
#ifdef DEBUG_NR_PSSCHSIM
          if (i % 8 == 0) {
            if (payload_type_string) {
              printf("TxByte : %c  vs  %c : RxByte\n", txUE->slsch[0][0]->harq_processes[harq_pid]->a[i / 8], harq_process_rxUE->b[i / 8]);
            } else {
              printf("TxByte : %2u  vs  %2u : RxByte\n", txUE->slsch[0][0]->harq_processes[harq_pid]->a[i / 8], harq_process_rxUE->b[i / 8]);
            }
          }
  #endif
          if (estimated_output_bit[i] != test_input_bit[i]) {
            errors_bit_delta++;
          }
        }
        if (errors_bit_delta > 0) {
          n_false_positive++;
          printf("errors_bit %u (trial %d)\n", errors_bit_delta, trial);
        }
        if ((errors_bit_delta > 0) || (polar_decoded == false)) {
          n_errors++;
        }
        errors_bit += errors_bit_delta;
      }
    } // trial

    printf("*****************************************\n");
    printf("SNR %f, BLER %f BER %f\n", SNR,
          (float) n_errors / (float) n_trials,
          (float) errors_bit / (float) (n_trials * num_bytes_to_check));
    printf("*****************************************\n");
    printf("\n");

    if (n_errors == 0) {
      printf("PSSCH test OK\n");
      printf("\n");
      break;
    }
    else {
      printf("PSSCH test NG due to number of error bits: %u\n", errors_bit);
      printf("\n");
    }
    printf("\n");

  } // snr

  term_nr_ue_transport(txUE);
  term_nr_ue_transport(rxUE);
  term_nr_ue_signal(rxUE, 1);
  term_nr_ue_signal(txUE, 1);
  free(txUE);
  free(rxUE);

  free_channel_desc_scm(UE2UE);
  free(bw_setting);

  loader_reset();
  logTerm();
  return (0);
}
