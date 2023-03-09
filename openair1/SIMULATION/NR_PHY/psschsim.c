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
#include "SCHED_NR/sched_nr.h"
#include "openair1/SIMULATION/TOOLS/sim.h"
#include "openair1/SIMULATION/RF/rf.h"
#include "openair1/SIMULATION/NR_PHY/nr_unitary_defs.h"
#include "openair1/SIMULATION/NR_PHY/nr_dummy_functions.c"
#include "common/utils/threadPool/thread-pool.h"
#include "openair2/LAYER2/NR_MAC_COMMON/nr_mac_common.h"
#include "executables/nr-uesoftmodem.h"
#include "PHY/impl_defs_top.h"

#define DEBUG_NR_SLSCHSIM
//#define DEBUG_NR_SLSCHSIM
#define HNA_SIZE 6 * 68 * 384 // [hna] 16 segments, 68*Zc
#define SCI2_LEN_SIZE 35
RAN_CONTEXT_t RC;

double cpuf;
uint16_t NB_UE_INST = 1;
openair0_config_t openair0_cfg[MAX_CARDS];
uint8_t const nr_rv_round_map[4] = {0, 2, 3, 1};
const short conjugate[8]__attribute__((aligned(16))) = {-1,1,-1,1,-1,1,-1,1};
const short conjugate2[8]__attribute__((aligned(16))) = {1,-1,1,-1,1,-1,1,-1};
PHY_VARS_NR_UE *PHY_vars_UE_g[1][1] = { { NULL } };
uint16_t n_rnti = 0x1234;
openair0_config_t openair0_cfg[MAX_CARDS];
double **s_re, **s_im, **r_re, **r_im;

uint64_t get_softmodem_optmask(void) {return 0;}
static softmodem_params_t softmodem_params;
softmodem_params_t *get_softmodem_params(void) {
  return &softmodem_params;
}

nrUE_params_t nrUE_params;
nrUE_params_t *get_nrUE_params(void) {
  return &nrUE_params;
}

bool nr_ue_sl_postDecode_sim(PHY_VARS_NR_UE *phy_vars_ue, notifiedFIFO_elt_t *req, bool last, notifiedFIFO_t *nf_p) {
  ldpcDecode_ue_t *rdata = (ldpcDecode_ue_t*) NotifiedFifoData(req);
  NR_DL_UE_HARQ_t *harq_process = rdata->harq_process;
  NR_UE_DLSCH_t *dlsch = (NR_UE_DLSCH_t *) rdata->dlsch;
  int r = rdata->segment_r;

  merge_meas(&phy_vars_ue->dlsch_deinterleaving_stats, &rdata->ts_deinterleave);
  merge_meas(&phy_vars_ue->dlsch_rate_unmatching_stats, &rdata->ts_rate_unmatch);
  merge_meas(&phy_vars_ue->dlsch_ldpc_decoding_stats, &rdata->ts_ldpc_decode);

  bool decodeSuccess = (rdata->decodeIterations < (1+dlsch->max_ldpc_iterations));

  if (decodeSuccess) {
    memcpy(harq_process->b+rdata->offset,
           harq_process->c[r],
           rdata->Kr_bytes - (harq_process->F>>3) -((harq_process->C>1)?3:0));

  } else {
    if ( !last ) {
      int nb=abortTpoolJob(&get_nrUE_params()->Tpool, req->key);
      nb+=abortNotifiedFIFOJob(nf_p, req->key);
      LOG_D(PHY,"downlink segment error %d/%d, aborted %d segments\n",rdata->segment_r,rdata->nbSegments, nb);
      LOG_D(PHY, "SLSCH %d in error\n",rdata->dlsch_id);
      last = true;
    }
  }

  // if all segments are done
  if (last) {
    if (decodeSuccess) {
      //LOG_D(PHY,"[UE %d] DLSCH: Setting ACK for nr_slot_rx %d TBS %d mcs %d nb_rb %d harq_process->round %d\n",
      //      phy_vars_ue->Mod_id,nr_slot_rx,harq_process->TBS,harq_process->mcs,harq_process->nb_rb, harq_process->round);
      harq_process->status = SCH_IDLE;
      harq_process->ack = 1;

      //LOG_D(PHY,"[UE %d] DLSCH: Setting ACK for SFN/SF %d/%d (pid %d, status %d, round %d, TBS %d, mcs %d)\n",
      //  phy_vars_ue->Mod_id, frame, subframe, harq_pid, harq_process->status, harq_process->round,harq_process->TBS,harq_process->mcs);

      //if(is_crnti) {
      //  LOG_D(PHY,"[UE %d] DLSCH: Setting ACK for nr_slot_rx %d (pid %d, round %d, TBS %d)\n",phy_vars_ue->Mod_id,nr_slot_rx,harq_pid,harq_process->round,harq_process->TBS);
      //}
      dlsch->last_iteration_cnt = rdata->decodeIterations;
      LOG_D(PHY, "SLSCH received ok \n");
    } else {
      //LOG_D(PHY,"[UE %d] DLSCH: Setting NAK for SFN/SF %d/%d (pid %d, status %d, round %d, TBS %d, mcs %d) Kr %d r %d harq_process->round %d\n",
      //      phy_vars_ue->Mod_id, frame, nr_slot_rx, harq_pid,harq_process->status, harq_process->round,harq_process->TBS,harq_process->mcs,Kr,r,harq_process->round);
      harq_process->ack = 0;

      //if(is_crnti) {
      //  LOG_D(PHY,"[UE %d] DLSCH: Setting NACK for nr_slot_rx %d (pid %d, pid status %d, round %d/Max %d, TBS %d)\n",
      //        phy_vars_ue->Mod_id,nr_slot_rx,harq_pid,harq_process->status,harq_process->round,dlsch->Mdlharq,harq_process->TBS);
      //}
      dlsch->last_iteration_cnt = dlsch->max_ldpc_iterations + 1;
      LOG_D(PHY, "SLSCH received nok \n");
    }
    return true; //stop
  }
  else
  {
	return false; //not last one
  }
}

void init_downlink_harq_status(NR_DL_UE_HARQ_t *dl_harq) {}

typedef struct {
  double scs;
  double bw;
  double fs;
} BW;

double snr0 =100;//- 2.0;
double snr1 = 2.0;
uint8_t snr1set = 0;
int n_trials = 1;
uint8_t n_tx = 1;
uint8_t n_rx = 1;
int ssb_subcarrier_offset = 0;
FILE *input_fd = NULL;
SCM_t channel_model = AWGN;
int N_RB = 106;
int N_RB_UL = 106;
int N_RB_DL = 106;
int mu = 1;
int loglvl = OAILOG_WARNING;
int seed = 0;

static void get_sim_cl_opts(int argc, char **argv)
{
    char c;
    while ((c = getopt(argc, argv, "F:g:hIL:m:M:n:N:o:O:p:P:r:R:s:S:x:y:z:")) != -1) {
    switch (c) {
      case 'F':
        input_fd = fopen(optarg, "r");
        if (input_fd == NULL) {
          printf("Problem with filename %s. Exiting.\n", optarg);
          exit(-1);
        }
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

      case 'm':
        mu = atoi(optarg);
        break;

      case 'n':
        n_trials = atoi(optarg);
        break;

      case 'R':
        N_RB_UL = atoi(optarg);
        break;

      case 'r':
        N_RB = atoi(optarg);
        break;

      case 's':
        snr0 = atof(optarg);
        break;

      case 'S':
        snr1 = atof(optarg);
        snr1set = 1;
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
          printf("-R N_RB_UL\n");
          printf("-r nb_rb\n");
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
                                    int N_RB_UL,
                                    int N_RB_DL,
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
  nrUE_config->ssb_table.ssb_offset_point_a        = (N_RB_UL - 20) >> 1;
  nrUE_config->ssb_table.ssb_mask_list[1].ssb_mask = (rev_burst)&(0xFFFFFFFF);
  nrUE_config->ssb_table.ssb_mask_list[0].ssb_mask = (rev_burst >> 32)&(0xFFFFFFFF);
  nrUE_config->cell_config.frame_duplex_type       = TDD;
  nrUE_config->ssb_table.ssb_period                = 1; //10ms
  nrUE_config->carrier_config.dl_grid_size[mu]     = N_RB_DL;
  nrUE_config->carrier_config.ul_grid_size[mu]     = N_RB_UL;
  nrUE_config->carrier_config.num_tx_ant           = fp->nb_antennas_tx;
  nrUE_config->carrier_config.num_rx_ant           = fp->nb_antennas_rx;
  nrUE_config->tdd_table.tdd_period                = 0;

  ue->mac_enabled = 1;
  if (mu == 0) {
    fp->dl_CarrierFreq = 2600000000;
    fp->ul_CarrierFreq = 2600000000;
    fp->nr_band = 38;
  } else if (mu == 1) {
    fp->dl_CarrierFreq = 3600000000;
    fp->ul_CarrierFreq = 3600000000;
    fp->sl_CarrierFreq = 2600000000;
    fp->nr_band = 78;
  } else if (mu == 3) {
    fp->dl_CarrierFreq = 27524520000;
    fp->ul_CarrierFreq = 27524520000;
    fp->nr_band = 261;
  }

  fp->threequarter_fs = 0;
  nrUE_config->carrier_config.dl_bandwidth = config_bandwidth(mu, N_RB_DL, fp->nr_band);

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
  double scs, fs, bw;
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

/// @brief
/// @param argc
/// @param argv
/// @return
int main(int argc, char **argv)
{
  // **** Initialization ****
  int i;
  double snr_step = 0.2;
  //uint8_t write_output_file = 0;
  int frame = 0, slot = 0;
  unsigned char qbits = 8;
  uint64_t burst_position = 0x01;
  get_softmodem_params()->sl_mode = 2;

  cpuf = get_cpu_freq_GHz();

  if (load_configmodule(argc, argv, CONFIG_ENABLECMDLINEONLY) == 0) {
    exit_fun("[NR_PSBCHSIM] Error, configuration module init failed\n");
  }
  get_sim_cl_opts(argc, argv);
  randominit(seed);
  logInit();
  set_glog(loglvl);
  load_nrLDPClib(NULL);

  PHY_VARS_NR_UE *nearbyUE = malloc(sizeof(PHY_VARS_NR_UE));
  nearbyUE->frame_parms.N_RB_DL = N_RB_DL;
  nearbyUE->frame_parms.N_RB_UL = N_RB_UL;
  nearbyUE->frame_parms.Ncp = NORMAL;
  nearbyUE->frame_parms.nb_antennas_tx = 1;
  nearbyUE->frame_parms.nb_antennas_rx = n_rx;
  nearbyUE->max_ldpc_iterations = 5;
  PHY_VARS_NR_UE *syncRefUE = malloc(sizeof(PHY_VARS_NR_UE));
  syncRefUE->frame_parms.nb_antennas_tx = n_tx;
  syncRefUE->frame_parms.nb_antennas_rx = 1;
  initTpool("n", &syncRefUE->threadPool, true);
  initNotifiedFIFO(&syncRefUE->respDecode);

  nr_phy_config_request_psschsim(nearbyUE, N_RB_UL, N_RB_DL, mu, burst_position);
  nr_phy_config_request_psschsim(syncRefUE, N_RB_UL, N_RB_DL, mu, burst_position);

  BW *bw_setting = malloc(sizeof(BW));
  set_fs_bw(nearbyUE, mu, N_RB_UL, bw_setting);

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

  if (init_nr_ue_signal(nearbyUE, 1) != 0 || init_nr_ue_signal(syncRefUE, 1) != 0) {
    printf("Error at UE NR initialisation.\n");
    free(bw_setting);
    free(nearbyUE);
    free(syncRefUE);
    exit(-1);
  }
  /*
  for (sf = 0; sf < 2; sf++) {
    txUE->slsch[sf][0] = new_nr_ue_ulsch(N_RB, 8, frame_parms);
    rxUE->slsch_rx[sf][0] = new_nr_ue_ulsch(N_RB, 8, frame_parms);
    if (!txUE->slsch[sf][0]) {
      printf("Can't get ue ulsch structures.\n");
      exit(-1);
    }
  }
  */
  init_nr_ue_transport(nearbyUE);
  init_nr_ue_transport(syncRefUE);

  s_re = malloc(n_tx*sizeof(double*));
  s_im = malloc(n_tx*sizeof(double*));
  r_re = malloc(n_rx*sizeof(double*));
  r_im = malloc(n_rx*sizeof(double*));

  unsigned char harq_pid = 0;
  unsigned int TBS = 8424;
  uint32_t available_bits = 0;
  uint32_t G_SCI2_bits = 0;
  uint32_t G_slsch_bits = 0;
  uint32_t M_SCI2_bits = 0;
  uint32_t M_data_bits = 0;

  uint8_t number_dmrs_symbols = 2;
  uint16_t N_RE_prime;
  uint8_t rvidx = 0;
  uint8_t UE_id = 0;

  uint8_t nb_re_dmrs = 6;
  uint8_t Nl = 1; // number of layers
  uint8_t nb_codewords = 1;
  uint8_t Imcs = 9;
  uint16_t nb_symb_sch = 12;

  NR_UE_DLSCH_t *slsch_syncRefUE = syncRefUE->slsch_rx[UE_id][0][0];
  NR_DL_UE_HARQ_t *harq_process_syncRefUE = slsch_syncRefUE->harq_processes[harq_pid];
  nfapi_nr_pssch_pdu_t *rel16_ul = &harq_process_syncRefUE->pssch_pdu;
  NR_UE_ULSCH_t *slsch_ue = nearbyUE->slsch[0][0];

  if ((Nl == 4)||(Nl == 3))
    nb_re_dmrs = nb_re_dmrs*2;

  uint8_t mod_order = nr_get_Qm_ul(Imcs, 0);//{2,6790},
  uint16_t code_rate = nr_get_code_rate_ul(Imcs, 0);//{2,6790},
  G_slsch_bits = nr_get_G(N_RB, nb_symb_sch, nb_re_dmrs, number_dmrs_symbols, mod_order, Nl);
  TBS = nr_compute_tbs(mod_order,code_rate, N_RB, nb_symb_sch, nb_re_dmrs*number_dmrs_symbols, 0, 0, Nl);
  // Available bits 27984 TBS 18432 mod_order 2
  printf("\nG_slsch_bits %u TBS %u mod_order %d TBS/8 %d\n", G_slsch_bits, TBS, mod_order, TBS / 8);


  /////////// setting rel15_ul parameters ///////////
  rel16_ul->mcs_index           = Imcs;
  rel16_ul->pssch_data.rv_index = rvidx;
  rel16_ul->target_code_rate    = code_rate;
  rel16_ul->pssch_data.tb_size  = TBS>>3;
  rel16_ul->maintenance_parms_v3.ldpcBaseGraph = get_BG(TBS, code_rate);

  /////////////////////////[adk] preparing UL harq_process parameters/////////////////////////
  ///////////
  NR_UL_UE_HARQ_t *harq_process_nearbyUE = slsch_ue->harq_processes[harq_pid];
  DevAssert(harq_process_nearbyUE);

  uint8_t N_PRB_oh   = 0; // higher layer (RRC) parameter xOverhead in PUSCH-ServingCellConfig
  N_RE_prime = NR_NB_SC_PER_RB * nb_symb_sch - nb_re_dmrs - N_PRB_oh;
  harq_process_nearbyUE->pssch_pdu.mcs_index = Imcs;
  harq_process_nearbyUE->pssch_pdu.nrOfLayers = Nl;
  harq_process_nearbyUE->pssch_pdu.rb_size = N_RB;
  harq_process_nearbyUE->pssch_pdu.nr_of_symbols = nb_symb_sch;
  harq_process_nearbyUE->num_of_mod_symbols = N_RE_prime * N_RB * nb_codewords;
  harq_process_nearbyUE->pssch_pdu.pssch_data.rv_index = rvidx;
  harq_process_nearbyUE->pssch_pdu.pssch_data.tb_size  = TBS>>3;
  harq_process_nearbyUE->pssch_pdu.pssch_data.sci2_size = SCI2_LEN_SIZE >> 3;
  harq_process_nearbyUE->pssch_pdu.target_code_rate = code_rate;
  harq_process_nearbyUE->pssch_pdu.qam_mod_order = mod_order;
  uint64_t *sci_input = harq_process_nearbyUE->a_sci2;
  //SCI_2_A *sci2 = &harq_process_nearbyUE->pssch_pdu.sci2;
  SCI_1_A *sci1 = &harq_process_nearbyUE->pssch_pdu.sci1;
  set_sci(sci1, Imcs);
  unsigned char *test_input = harq_process_nearbyUE->a;


  NR_UE_DLSCH_t *slsch_ue_rx = syncRefUE->slsch_rx[0][0][0];
  slsch_ue_rx->harq_processes[harq_pid]->Nl = Nl;
  slsch_ue_rx->harq_processes[harq_pid]->Qm = mod_order;
  slsch_ue_rx->harq_processes[harq_pid]->nb_rb = N_RB;
  slsch_ue_rx->harq_processes[harq_pid]->TBS = TBS >> 3;
  slsch_ue_rx->harq_processes[harq_pid]->n_dmrs_cdm_groups = 1;
  slsch_ue_rx->harq_processes[harq_pid]->dlDmrsSymbPos = 16;
  slsch_ue_rx->harq_processes[harq_pid]->mcs = Imcs;
  slsch_ue_rx->harq_processes[harq_pid]->dmrsConfigType = 0;
  slsch_ue_rx->harq_processes[harq_pid]->R = code_rate;
  nfapi_nr_pssch_pdu_t *rel16_sl_rx = &slsch_ue_rx->harq_processes[harq_pid]->pssch_pdu;
  rel16_sl_rx->mcs_index            = Imcs;
  rel16_sl_rx->pssch_data.rv_index  = 0;
  rel16_sl_rx->target_code_rate     = code_rate;
  rel16_sl_rx->pssch_data.tb_size   = TBS >> 3; // bytes
  rel16_sl_rx->pssch_data.sci2_size = SCI2_LEN_SIZE >> 3;
  rel16_sl_rx->maintenance_parms_v3.ldpcBaseGraph = get_BG(TBS, code_rate);

  uint64_t u = pow(2, SCI2_LEN_SIZE) - 1;
  *sci_input = u;//rand() % (u - 0 + 1);
  printf("the sci2 is:%"PRIu64"\n", *sci_input);

  crcTableInit();
  for (int i = 0; i < TBS / 8; i++)
    test_input[i] = (unsigned char) (i+3);//rand();

  ////////////////////////////////////////////////////////////////////////////////////////////

//#define DEBUG_NR_SLSCHSIM 1
#ifdef DEBUG_NR_SLSCHSIM
  for (i = 0; i < 10; i++)
   test_input[i] = i;//(unsigned char) rand();
  //for (i = 0; i < TBS / 8; i++) printf("test_input[i]=%hhu \n", test_input[i]);
  for (i = 0; i < 10; i++) printf("test_input[i]=%hhu \n", test_input[i]);
#endif

  /////////////////////////SLSCH SCI2 coding/////////////////////////
  // TODO: update the following
  PSSCH_SCI2_payload pssch_payload;
  NR_UE_PSSCH m_pssch;
  nearbyUE->pssch_vars[0] = &m_pssch;
  NR_UE_PSSCH *pssch = nearbyUE->pssch_vars[0];
  memset((void *)pssch, 0, sizeof(NR_UE_PSSCH));

  pssch->pssch_a = *((uint32_t *)&pssch_payload);
  pssch->pssch_a_interleaved = pssch->pssch_a; // skip interlevaing for Sidelink

  // Encoder reversal
  uint64_t a_reversed = 0;
  for (int i = 0; i < NR_POLAR_PSSCH_PAYLOAD_BITS; i++)
    a_reversed |= (((uint64_t)pssch->pssch_a_interleaved >> i) & 1) << (31 - i);
  uint16_t Nidx = 0;

#if 0
  Nidx = get_Nidx_from_CRC(&a_reversed, 0, 0,
                           NR_POLAR_PSSCH_MESSAGE_TYPE,
                           NR_POLAR_PSSCH_PAYLOAD_BITS,
                           NR_POLAR_PSSCH_AGGREGATION_LEVEL);

  /// CRC, coding and rate matching
  polar_encoder_fast(&a_reversed, (void*)pssch->pssch_e, 0, 0,
                     NR_POLAR_PSSCH_MESSAGE_TYPE,
                     NR_POLAR_PSSCH_PAYLOAD_BITS,
                     NR_POLAR_PSSCH_AGGREGATION_LEVEL);
  SCI2_bits = NR_POLAR_PSSCH_E;
  available_bits += NR_POLAR_PSSCH_E;
#endif

  /////////////////////////SLSCH data coding/////////////////////////
  ///////////
  G_slsch_bits = nr_get_G(N_RB, nb_symb_sch, nb_re_dmrs, number_dmrs_symbols, mod_order, Nl);
  if (input_fd == NULL) {
    //nr_slsch_encoding(nearbyUE, slsch_ue, frame_parms, harq_pid, G_slsch_bits);
  }

  //////////////////SLSCH data and control multiplexing//////////////

  // TODO: Remove the following hard coded case.
  G_SCI2_bits = 32;
  pssch->pssch_e[0] = 0x0101;//-> 0011

  G_slsch_bits = 32;
  harq_process_nearbyUE->f[0] = 0;
  harq_process_nearbyUE->f[1] = 1;
  harq_process_nearbyUE->f[2] = 0;
  harq_process_nearbyUE->f[3] = 1;
  // TODO: Remove the above hard coded case.

#ifdef DEBUG_NR_SLSCHSIM
  printf("encoded SCI2_bits[i]= ");
  for (i = 0; i < G_SCI2_bits; i++)
    printf("%u ", ((uint8_t *)pssch->pssch_e)[i]);
  printf("\n\n");
  printf("encoded SLSCH_bits[i]= ");
  for (i = 0; i < G_slsch_bits; i++)
    printf("%u ", harq_process_nearbyUE->f[i]);
  printf("\n\n");
#endif

  M_SCI2_bits = G_SCI2_bits * Nl; //assume multiple of 32 bits
  printf("M_SCI2_bits mod 32 = %u\n", M_SCI2_bits % 32);
  AssertFatal(M_SCI2_bits % 32 == 0, "M_SCI2_bits %d needs to be multiple of 32, but remained %d\n", M_SCI2_bits, M_SCI2_bits % 32);
  M_data_bits = G_slsch_bits;
  available_bits += M_SCI2_bits;
  available_bits += M_data_bits;
  uint8_t  SCI2_mod_order = 2;
  uint8_t multiplexed_output[available_bits];
  memset(multiplexed_output, 0, available_bits * sizeof(uint8_t));

  nr_pssch_data_control_multiplexing(harq_process_nearbyUE->f,
                                     (uint8_t*)pssch->pssch_e,
                                     G_slsch_bits,
                                     G_SCI2_bits,
                                     Nl,
                                     SCI2_mod_order,
                                     multiplexed_output);

#ifdef DEBUG_NR_SLSCHSIM
  printf("muxed SLSCH_bits[i]= ");
  for (i = 0; i < available_bits; i++)
    printf("%u ", multiplexed_output[i]);
  printf("\n\n");
#endif

  /////////////////////////SLSCH scrambling/////////////////////////
  uint32_t scrambled_output[(available_bits >> 5) + 1];
  memset(scrambled_output, 0, ((available_bits >> 5) + 1)*sizeof(uint32_t));

  nr_pusch_codeword_scrambling_sl(multiplexed_output,
                                  available_bits,
                                  M_SCI2_bits,
                                  Nidx,
                                  scrambled_output);

#ifdef DEBUG_NR_SLSCHSIM
  printf("Nidx= %d\n", Nidx);
  printf("scambled SCI2_bits = ");
  for (i = 0; i < (M_SCI2_bits + 31) >> 5; i++)
    printf("0x%x ", scrambled_output[i]);
  printf("\n\n");
  printf("scambled data_bits = ");
  for (i = 0; i < (M_data_bits + 31) >> 5; i++)
    printf("0x%x ", scrambled_output[i+ M_SCI2_bits]);
  printf("\n\n");
#endif

  /////////////////////////SLSCH modulation/////////////////////////

  int max_num_re = Nl * nb_symb_sch * N_RB * NR_NB_SC_PER_RB;
  int32_t d_mod[max_num_re] __attribute__ ((aligned(16)));

  // modulating for the 2nd-stage SCI bits
  nr_modulation(scrambled_output, // assume one codeword for the moment
                M_SCI2_bits,
                SCI2_mod_order,
                (int16_t *)d_mod);

  // modulating SL-SCH bits
  nr_modulation(scrambled_output + (M_SCI2_bits >> 5), // assume one codeword for the moment
                M_data_bits,
                mod_order,
                (int16_t *)(d_mod + M_SCI2_bits / SCI2_mod_order));


#ifdef DEBUG_NR_SLSCHSIM
  printf("modulated SCI2_bits[i]= ");
  for (i = 0; i < M_SCI2_bits / SCI2_mod_order; i++) {
    if (i%4 == 0) printf("\n");
    printf("0x%x ", d_mod[i]);
  }
  printf("\n\n");

  printf("modulated SLSCH_bits[i]= ");
  for (i = 0; i < M_data_bits / mod_order; i++) {
    if (i%4 == 0) printf("\n");
    printf("0x%x ", d_mod[i + M_SCI2_bits / SCI2_mod_order]);
  }
  printf("\n\n");
#endif

  /////////////////////////LLRs computation/////////////////////////

  int16_t *ulsch_llr = syncRefUE->pssch_vars[UE_id]->llr;
  int nb_re_SCI2 = M_SCI2_bits/SCI2_mod_order;
  int nb_re_slsch = M_data_bits/mod_order;
  uint8_t  symbol = 5;
  int32_t a = 1;
  int32_t b = 1;

  nr_ulsch_compute_llr(d_mod, &a, &b,
                      ulsch_llr, N_RB, nb_re_SCI2,
                      symbol, SCI2_mod_order);

  nr_ulsch_compute_llr(d_mod + (M_SCI2_bits / SCI2_mod_order), &a, &b,
                      ulsch_llr + nb_re_SCI2 * 2, N_RB, nb_re_slsch,
                      symbol, mod_order);

#ifdef DEBUG_NR_SLSCHSIM

  printf("SCI2 llr value= ");
  for (i = 0; i < M_SCI2_bits; i++) {
    if (i%8 == 0) printf("\n");
    printf("0x%x ", ulsch_llr[i] & 0xFFFF);
  }
  printf("\n\n");
  printf("SLSCH llr value= ");
  for (i = 0; i < M_data_bits; i++) {
    if (i%8 == 0) printf("\n");
    printf("0x%x ", ulsch_llr[i + M_SCI2_bits] & 0xFFFF);
  }
  printf("\n\n");
#endif


#if 0
  nr_ulsch_layer_demapping(rxUE->pssch_vars[UE_id].llr,
                           rel16_ul->nrOfLayers,
                           rel16_ul->qam_mod_order,
                           available_bits,
                           rxUE->pssch_vars[UE_id]->llr_layers);
#endif

  /////////////////////////SLSCH descrambling/////////////////////////

nr_codeword_unscrambling_sl(ulsch_llr, available_bits, M_SCI2_bits, Nidx, Nl);

#ifdef DEBUG_NR_SLSCHSIM
  printf("Unscrambled llr value= ");
  for (i = 0; i < available_bits; i++) {
    if (i % 8 == 0) printf("\n");
    printf("0x%x ", ulsch_llr[i] & 0xFFFF);
  }
  printf("\n\n");
#endif

  //////////////////SLSCH data and control demultiplexing//////////////

int16_t *llr_polar_ctrl = ulsch_llr;//len: M_SCI2_bits
int16_t *llr_ldpc_data = ulsch_llr + M_SCI2_bits; //len: M_data_bits

#ifdef DEBUG_NR_SLSCHSIM
  printf("demultiplexed SCI2_bits[i]= ");
  for (i = 0; i < M_SCI2_bits; i++) {
    if (i % 8 == 0) printf("\n");
    printf("0x%x ", llr_polar_ctrl[i] & 0xFFFF);
  }
  printf("\n\n");
  printf("demultiplexed SLSCH_bits[i]= ");
  for (i = 0; i < M_data_bits; i++) {
    if (i % 8 == 0) printf("\n");
    printf("0x%x ", llr_ldpc_data[i] & 0xFFFF);
  }
  printf("\n\n");
#endif

  #if 1
  unsigned int errors_bit_uncoded = 0;
  unsigned int errors_bit = 0;
  unsigned int n_errors = 0;
  double modulated_input[HNA_SIZE];
  unsigned char test_input_bit[HNA_SIZE];
  short channel_output_fixed[HNA_SIZE];
  short channel_output_uncoded[HNA_SIZE];
  unsigned char estimated_output_bit[HNA_SIZE];
  snr1 = snr1set == 0 ? snr0 + 10 : snr1;
  int numb_bits = available_bits;//slsch_ue->harq_processes[harq_pid]->B_sci2;

  for (double SNR = snr0; SNR < snr1; SNR += snr_step) {
    n_errors = 0;
    for (int trial = 0; trial < n_trials; trial++) {
      errors_bit_uncoded = 0;
      for (int i = 0; i < numb_bits; i++) {
        if (slsch_ue->harq_processes[harq_pid]->f[i] == 0){
          modulated_input[i] = 1.0;        ///sqrt(2);  //QPSK
        }else{
          modulated_input[i] = -1.0;        ///sqrt(2);
        }

        double SNR_lin = pow(10, SNR / 10.0);
        double sigma = 1.0 / sqrt(2 * SNR_lin);
        channel_output_fixed[i] = (short) quantize(sigma / 4.0 / 4.0,
                                                   modulated_input[i] + sigma * gaussdouble(0.0, 1.0),
                                                   qbits);
        if (channel_output_fixed[i] < 0)
          channel_output_uncoded[i] = 1;  //QPSK demod
        else
          channel_output_uncoded[i] = 0;
        if (channel_output_uncoded[i] != slsch_ue->harq_processes[harq_pid]->f[i])
          errors_bit_uncoded = errors_bit_uncoded + 1;
      }
      UE_nr_rxtx_proc_t proc;
      // this is a small hack :D
      slsch_ue_rx->harq_processes[0]->B_sci2 = slsch_ue->harq_processes[harq_pid]->B_sci2;
      uint32_t ret = nr_slsch_decoding(syncRefUE, &proc,channel_output_fixed,
                                  &syncRefUE->frame_parms, slsch_ue_rx,
                                  slsch_ue_rx->harq_processes[0], frame,
                                  nb_symb_sch, slot, harq_pid);

      if (ret)
        n_errors++;

      errors_bit = 0;
      for (int i = 0; i < TBS; i++) {
        estimated_output_bit[i] = (slsch_ue_rx->harq_processes[harq_pid]->b[i / 8] & (1 << (i & 7))) >> (i & 7);
        test_input_bit[i] = (test_input[i / 8] & (1 << (i & 7))) >> (i & 7); // Further correct for multiple segments
        printf("tx bit: %u, rx bit: %u\n",test_input_bit[i],estimated_output_bit[i]);
        if (estimated_output_bit[i] != test_input_bit[i]) {
          errors_bit++;
        }
      }

      printf("*****************************************\n");
      printf("SNR %f, BLER %f (number of erros bit %d)\n", SNR,
            (float) n_errors / (float) n_trials,
            errors_bit);
      printf("*****************************************\n");
      printf("\n");

      if (n_errors == 0) {
        printf("PSSCH test OK\n");
        printf("\n");
        break;
      }
      printf("\n");
    }
  }
  #endif
  for (int sf = 0; sf < 2; sf++) {
    free_nr_ue_slsch(&nearbyUE->slsch[sf][0], N_RB_UL, &nearbyUE->frame_parms);
    free_nr_ue_dlsch(&syncRefUE->slsch_rx[sf][0][0], N_RB_DL);
  }
  //term_nr_ue_transport(nearbyUE);
  term_nr_ue_transport(syncRefUE);
  term_nr_ue_signal(syncRefUE, 1);
  term_nr_ue_signal(nearbyUE, 1);
  free(nearbyUE);
  free(syncRefUE);

  free_channel_desc_scm(UE2UE);
  free(bw_setting);

  loader_reset();
  logTerm();
  return (0);
}
