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

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <bits/getopt_core.h>
#include "common/utils/nr/nr_common.h"
#include "common/utils/var_array.h"
#define inMicroS(a) (((double)(a)) / (get_cpu_freq_GHz() * 1000.0))
#include "SIMULATION/LTE_PHY/common_sim.h"
#include "openair2/RRC/LTE/rrc_vars.h"
#include "common/utils/assertions.h"
#include "executables/softmodem-common.h"
#include "NR_BCCH-BCH-Message.h"
#include "NR_IF_Module.h"
#include "NR_MAC_COMMON/nr_mac.h"
#include "NR_MAC_COMMON/nr_mac_common.h"
#include "NR_MAC_UE/mac_defs.h"
#include "NR_MAC_gNB/nr_mac_gNB.h"
#include "NR_PHY_INTERFACE/NR_IF_Module.h"
#include "NR_ReconfigurationWithSync.h"
#include "NR_ServingCellConfig.h"
#include "NR_UE-NR-Capability.h"
#include "PHY/CODING/nrLDPC_coding/nrLDPC_coding_interface.h"
#include "PHY/INIT/nr_phy_init.h"
#include "PHY/MODULATION/nr_modulation.h"
#include "PHY/NR_ESTIMATION/nr_ul_estimation.h"
#include "PHY/NR_REFSIG/dmrs_nr.h"
#include "PHY/NR_REFSIG/ptrs_nr.h"
#include "PHY/NR_REFSIG/ul_ref_seq_nr.h"
#include "PHY/NR_TRANSPORT/nr_transport_common_proto.h"
#include "PHY/NR_TRANSPORT/nr_ulsch.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_ue.h"
#include "PHY/TOOLS/tools_defs.h"
#include "PHY/defs_RU.h"
#include "PHY/defs_common.h"
#include "PHY/defs_gNB.h"
#include "PHY/defs_nr_UE.h"
#include "PHY/defs_nr_common.h"
#include "PHY/impl_defs_nr.h"
#include "PHY/phy_vars_nr_ue.h"
#include "SCHED_NR/fapi_nr_l1.h"
#include "SCHED_NR/sched_nr.h"
#include "SCHED_NR_UE/defs.h"
#include "SCHED_NR_UE/fapi_nr_ue_l1.h"
#include "asn_internal.h"
#include "assertions.h"
#include "common/config/config_load_configmodule.h"
#include "common/ngran_types.h"
#include "common/openairinterface5g_limits.h"
#include "common/ran_context.h"
#include "common/utils/LOG/log.h"
#include "common/utils/T/T.h"
#include "common/utils/nr/nr_common.h"
#include "common/utils/threadPool/thread-pool.h"
#include "common/utils/var_array.h"
#include "common_lib.h"
#include "e1ap_messages_types.h"
#include "executables/nr-uesoftmodem.h"
#include "fapi_nr_ue_constants.h"
#include "fapi_nr_ue_interface.h"
#include "nfapi_interface.h"
#include "nfapi_nr_interface_scf.h"
#include "nr_ue_phy_meas.h"
#include "openair1/SIMULATION/NR_PHY/nr_unitary_defs.h"
#include "openair1/SIMULATION/TOOLS/sim.h"
#include "openair2/LAYER2/NR_MAC_UE/mac_proto.h"
#include "openair2/LAYER2/NR_MAC_gNB/mac_proto.h"
#include "openair2/RRC/NR/nr_rrc_config.h"
#include "time_meas.h"
#include "utils.h"

// #define DEBUG_ULSIM

#define DUMP_CH_EST_COMP_IN_OUT

void dump_channel_estimation_compensation_in_out(PHY_VARS_gNB *gNB,
                                                 int beam_nb,
                                                 int ulsch_id,
                                                 char *rxdataF_file,
                                                 char *rxdataF_comp_file)
{
  FILE *fd;

  fd = fopen(rxdataF_file, "w");
  int nb_samples_per_antenna = gNB->frame_parms.symbols_per_slot * gNB->frame_parms.ofdm_symbol_size;
  for (int antenna = 0; antenna < gNB->frame_parms.nb_antennas_rx; antenna++) {
    c16_t *rxdataF = &gNB->common_vars.rxdataF[beam_nb][antenna][0];
    fwrite((void *)rxdataF, sizeof(c16_t), nb_samples_per_antenna, fd);
  }
  fclose(fd);

  fd = fopen(rxdataF_comp_file, "w");
  nfapi_nr_pusch_pdu_t *rel15_ul = &gNB->ulsch[ulsch_id].harq_process->ulsch_pdu;
  int buffer_length = ceil_mod(rel15_ul->rb_size * NR_NB_SC_PER_RB, 16);
  int nb_samples_per_antenna_layer = rel15_ul->nr_of_symbols * buffer_length;
  for (int layer = 0; layer < rel15_ul->nrOfLayers; layer++) {
    int32_t *rxdataF_comp =
        &gNB->pusch_vars[ulsch_id]
             .rxdataF_comp[layer * gNB->frame_parms.nb_antennas_rx][rel15_ul->start_symbol_index * buffer_length];
    fwrite((void *)rxdataF_comp, sizeof(int32_t), nb_samples_per_antenna_layer, fd);
  }
  fclose(fd);
}

void read_channel_estimation_compensation_parameters(PHY_VARS_gNB *gNB,
                                                     int *frame,
                                                     int *slot,
                                                     int *beam_nb,
                                                     int *ulsch_id,
                                                     chestcomp_params_t *params,
                                                     char *params_file)
{
  FILE *fd;

  fd = fopen(params_file, "r");
  AssertFatal(fread((void *)params, sizeof(chestcomp_params_t), 1, fd) == 1, "error reading file: %s\n", params_file);
  fclose(fd);
}
void read_channel_estimation_compensation_in_out(PHY_VARS_gNB *gNB,
                                                 int beam_nb,
                                                 int ulsch_id,
                                                 char *rxdataF_file,
                                                 char *rxdataF_comp_file,
                                                 int32_t *rxdataF_comp_ref)
{
  FILE *fd;

  fd = fopen(rxdataF_file, "r");
  AssertFatal(fd != NULL, "error opening file: %s\n", rxdataF_file);
  int nb_samples_per_antenna = gNB->frame_parms.symbols_per_slot * gNB->frame_parms.ofdm_symbol_size;
  for (int antenna = 0; antenna < gNB->frame_parms.nb_antennas_rx; antenna++) {
    c16_t *rxdataF = &gNB->common_vars.rxdataF[beam_nb][antenna][0];
    AssertFatal(fread((void *)rxdataF, sizeof(c16_t), nb_samples_per_antenna, fd) == nb_samples_per_antenna,
                "error reading file: %s\n",
                rxdataF_file);
  }
  fclose(fd);

  fd = fopen(rxdataF_comp_file, "r");
  AssertFatal(fd != NULL, "error opening file: %s\n", rxdataF_comp_file);
  nfapi_nr_pusch_pdu_t *rel15_ul = &gNB->ulsch[ulsch_id].harq_process->ulsch_pdu;
  int buffer_length = ceil_mod(rel15_ul->rb_size * NR_NB_SC_PER_RB, 16);
  int nb_samples_per_antenna_layer = rel15_ul->nr_of_symbols * buffer_length;
  for (int layer = 0; layer < rel15_ul->nrOfLayers; layer++) {
    int32_t *rxdataF_comp = &rxdataF_comp_ref[layer * nb_samples_per_antenna_layer];
    AssertFatal(fread((void *)rxdataF_comp, sizeof(int32_t), nb_samples_per_antenna_layer, fd) == nb_samples_per_antenna_layer,
                "error reading file: %s\n",
                rxdataF_comp_file);
  }
  fclose(fd);
}

int channel_estimation_compensation_test(PHY_VARS_gNB *gNB, int frame, int slot)
{
  start_meas(&gNB->rx_pusch_stats);

  int ret = 0;

  int ulsch_id = 0;
  int beam_nb = 0;
  nfapi_nr_pusch_pdu_t *rel15_ul = &gNB->ulsch[ulsch_id].harq_process->ulsch_pdu;
  NR_gNB_PUSCH *pusch_vars = &gNB->pusch_vars[ulsch_id];
  NR_DL_FRAME_PARMS *frame_parms = &gNB->frame_parms;

  //----------------------------------------------------------
  //------------------- Channel estimation -------------------
  //----------------------------------------------------------

  start_meas(&gNB->ulsch_channel_estimation_stats);

  uint32_t bwp_start_subcarrier = ((rel15_ul->rb_start + rel15_ul->bwp_start) * NR_NB_SC_PER_RB + frame_parms->first_carrier_offset)
                                  % frame_parms->ofdm_symbol_size;

  int max_ch = 0;
  uint32_t nvar = 0;
  int end_symbol = rel15_ul->start_symbol_index + rel15_ul->nr_of_symbols;
  for (uint8_t symbol = rel15_ul->start_symbol_index; symbol < end_symbol; symbol++) {
    uint8_t dmrs_symbol_flag = (rel15_ul->ul_dmrs_symb_pos >> symbol) & 0x01;
    LOG_D(PHY, "symbol %d, dmrs_symbol_flag :%d\n", symbol, dmrs_symbol_flag);

    if (dmrs_symbol_flag == 1) {
      for (int nl = 0; nl < rel15_ul->nrOfLayers; nl++) {
        uint32_t nvar_tmp = 0;
        nr_pusch_channel_estimation(gNB,
                                    slot,
                                    nl,
                                    get_dmrs_port(nl, rel15_ul->dmrs_ports),
                                    symbol,
                                    ulsch_id,
                                    beam_nb,
                                    bwp_start_subcarrier,
                                    rel15_ul,
                                    &max_ch,
                                    &nvar_tmp);
        nvar += nvar_tmp;
      }
    }
  }

  nvar /= (rel15_ul->nr_of_symbols * rel15_ul->nrOfLayers * frame_parms->nb_antennas_rx);

  // averaging time domain channel estimates
  if (gNB->chest_time == 1)
    nr_chest_time_domain_avg(frame_parms,
                             pusch_vars->ul_ch_estimates,
                             rel15_ul->nr_of_symbols,
                             rel15_ul->start_symbol_index,
                             rel15_ul->ul_dmrs_symb_pos,
                             rel15_ul->rb_size);

  stop_meas(&gNB->ulsch_channel_estimation_stats);

  // first the computation of channel levels

  int nb_re_pusch = 0, meas_symbol = -1;
  for (meas_symbol = rel15_ul->start_symbol_index; meas_symbol < end_symbol; meas_symbol++)
    if ((nb_re_pusch = get_nb_re_pusch(frame_parms, rel15_ul, meas_symbol)) > 0)
      break;

  AssertFatal(nb_re_pusch > 0 && meas_symbol >= 0,
              "nb_re_pusch %d cannot be 0 or meas_symbol %d cannot be negative here\n",
              nb_re_pusch,
              meas_symbol);

  // extract the first dmrs for the channel level computation
  // extract the data in the OFDM frame, to the start of the array
  int soffset = (slot % RU_RX_SLOT_DEPTH) * frame_parms->symbols_per_slot * frame_parms->ofdm_symbol_size;

  nb_re_pusch = ceil_mod(nb_re_pusch, 16);
  int dmrs_symbol_id;
  if (gNB->chest_time == 0)
    dmrs_symbol_id = get_valid_dmrs_idx_for_channel_est(rel15_ul->ul_dmrs_symb_pos, meas_symbol);
  else // average of channel estimates stored in first symbol
    dmrs_symbol_id = get_next_dmrs_symbol_in_slot(rel15_ul->ul_dmrs_symb_pos, rel15_ul->start_symbol_index, end_symbol);
  int size_est = nb_re_pusch * frame_parms->symbols_per_slot;
  __attribute__((aligned(32))) int ul_ch_estimates_ext[rel15_ul->nrOfLayers * frame_parms->nb_antennas_rx][size_est];
  memset(ul_ch_estimates_ext, 0, sizeof(ul_ch_estimates_ext));
  c16_t temp_rxFext[frame_parms->nb_antennas_rx][rel15_ul->rb_size * NR_NB_SC_PER_RB] __attribute__((aligned(32)));
  for (int aarx = 0; aarx < frame_parms->nb_antennas_rx; aarx++)
    for (int nl = 0; nl < rel15_ul->nrOfLayers; nl++)
      nr_ulsch_extract_rbs(gNB->common_vars.rxdataF[beam_nb][aarx],
                           (c16_t *)pusch_vars->ul_ch_estimates[nl * frame_parms->nb_antennas_rx + aarx],
                           temp_rxFext[aarx],
                           (c16_t *)&ul_ch_estimates_ext[nl * frame_parms->nb_antennas_rx + aarx][meas_symbol * nb_re_pusch],
                           soffset + meas_symbol * frame_parms->ofdm_symbol_size,
                           dmrs_symbol_id * frame_parms->ofdm_symbol_size,
                           aarx,
                           (rel15_ul->ul_dmrs_symb_pos >> meas_symbol) & 0x01,
                           rel15_ul,
                           frame_parms);

  int avgs = 0;
  int avg[frame_parms->nb_antennas_rx * rel15_ul->nrOfLayers];
  uint8_t shift_ch_ext = rel15_ul->nrOfLayers > 1 ? log2_approx(max_ch >> 11) : 0;

  //----------------------------------------------------------
  //--------------------- Channel Scaling --------------------
  //----------------------------------------------------------
  nr_ulsch_scale_channel(size_est,
                         ul_ch_estimates_ext,
                         frame_parms,
                         meas_symbol,
                         (rel15_ul->ul_dmrs_symb_pos >> meas_symbol) & 0x01,
                         nb_re_pusch,
                         rel15_ul->nrOfLayers,
                         rel15_ul->rb_size,
                         shift_ch_ext);

  nr_ulsch_channel_level(size_est,
                         ul_ch_estimates_ext,
                         frame_parms,
                         avg,
                         meas_symbol, // index of the start symbol
                         nb_re_pusch, // number of the re in pusch
                         rel15_ul->nrOfLayers);

  for (int nl = 0; nl < rel15_ul->nrOfLayers; nl++)
    for (int aarx = 0; aarx < frame_parms->nb_antennas_rx; aarx++)
      avgs = cmax(avgs, avg[nl * frame_parms->nb_antennas_rx + aarx]);

  if (rel15_ul->nrOfLayers == 2 && rel15_ul->qam_mod_order > 6)
    pusch_vars->log2_maxh = (log2_approx(avgs) >> 1) - 3; // for MMSE
  else if (rel15_ul->nrOfLayers == 2)
    pusch_vars->log2_maxh = (log2_approx(avgs) >> 1) - 2 + log2_approx(frame_parms->nb_antennas_rx >> 1);
  else
    pusch_vars->log2_maxh = (log2_approx(avgs) >> 1) + 1 + log2_approx(frame_parms->nb_antennas_rx >> 1);

  if (pusch_vars->log2_maxh < 0)
    pusch_vars->log2_maxh = 0;

  //----------------------------------------------------------
  //------------------ Channel compensation ------------------
  //----------------------------------------------------------

  start_meas(&gNB->ulsch_channel_compensation_stats);

  c16_t **rxF = gNB->common_vars.rxdataF[beam_nb];
  int output_shift = gNB->pusch_vars[ulsch_id].log2_maxh;

  for (int symbol = rel15_ul->start_symbol_index; symbol < rel15_ul->start_symbol_index + rel15_ul->nr_of_symbols; symbol++) {
    gNB->pusch_vars[ulsch_id].ul_valid_re_per_slot[symbol] = get_nb_re_pusch(frame_parms, rel15_ul, symbol);
    if (gNB->pusch_vars[ulsch_id].ul_valid_re_per_slot[symbol] == 0)
      continue;

    int nb_layer = rel15_ul->nrOfLayers;
    int nb_rx_ant = frame_parms->nb_antennas_rx;
    int dmrs_symbol_flag = (rel15_ul->ul_dmrs_symb_pos >> symbol) & 0x01;
    int buffer_length = ceil_mod(rel15_ul->rb_size * NR_NB_SC_PER_RB, 16);
    c16_t rxFext[nb_rx_ant][buffer_length] __attribute__((aligned(32)));
    c16_t chFext[nb_layer][nb_rx_ant][buffer_length] __attribute__((aligned(32)));

    memset(rxFext, 0, sizeof(c16_t) * nb_rx_ant * buffer_length);
    memset(chFext, 0, sizeof(c16_t) * nb_layer * nb_rx_ant * buffer_length);
    int dmrs_symbol;
    if (gNB->chest_time == 0)
      dmrs_symbol = dmrs_symbol_flag ? symbol : get_valid_dmrs_idx_for_channel_est(rel15_ul->ul_dmrs_symb_pos, symbol);
    else { // average of channel estimates stored in first symbol
      int end_symbol = rel15_ul->start_symbol_index + rel15_ul->nr_of_symbols;
      dmrs_symbol = get_next_dmrs_symbol_in_slot(rel15_ul->ul_dmrs_symb_pos, rel15_ul->start_symbol_index, end_symbol);
    }

    for (int aarx = 0; aarx < nb_rx_ant; aarx++) {
      for (int aatx = 0; aatx < nb_layer; aatx++) {
        nr_ulsch_extract_rbs(rxF[aarx],
                             (c16_t *)pusch_vars->ul_ch_estimates[aatx * nb_rx_ant + aarx],
                             rxFext[aarx],
                             chFext[aatx][aarx],
                             soffset + (symbol * frame_parms->ofdm_symbol_size),
                             dmrs_symbol * frame_parms->ofdm_symbol_size,
                             aarx,
                             dmrs_symbol_flag,
                             rel15_ul,
                             frame_parms);
      }
    }
    c16_t rho[nb_layer][nb_layer][buffer_length] __attribute__((aligned(32)));
    c16_t rxF_ch_maga[nb_layer][buffer_length] __attribute__((aligned(32)));
    c16_t rxF_ch_magb[nb_layer][buffer_length] __attribute__((aligned(32)));
    c16_t rxF_ch_magc[nb_layer][buffer_length] __attribute__((aligned(32)));

    memset(rho, 0, sizeof(c16_t) * nb_layer * nb_layer * buffer_length);
    memset(rxF_ch_maga, 0, sizeof(c16_t) * nb_layer * buffer_length);
    memset(rxF_ch_magb, 0, sizeof(c16_t) * nb_layer * buffer_length);
    memset(rxF_ch_magc, 0, sizeof(c16_t) * nb_layer * buffer_length);
    for (int i = 0; i < nb_layer; i++)
      memset(&pusch_vars->rxdataF_comp[i * nb_rx_ant][symbol * buffer_length], 0, sizeof(int32_t) * buffer_length);

    nr_ulsch_channel_compensation((c16_t *)rxFext,
                                  (c16_t *)chFext,
                                  (c16_t *)rxF_ch_maga,
                                  (c16_t *)rxF_ch_magb,
                                  (c16_t *)rxF_ch_magc,
                                  pusch_vars->rxdataF_comp,
                                  (nb_layer == 1) ? NULL : (c16_t *)rho,
                                  frame_parms,
                                  rel15_ul,
                                  symbol,
                                  buffer_length,
                                  output_shift);
  }

  stop_meas(&gNB->ulsch_channel_compensation_stats);

  stop_meas(&gNB->rx_pusch_stats);

  return ret;
}

const char *__asan_default_options()
{
  /* don't do leak checking in nr_ulsim, not finished yet */
  return "detect_leaks=0";
}
PHY_VARS_gNB *gNB;
PHY_VARS_NR_UE *UE;
RAN_CONTEXT_t RC;
char *uecap_file;
int32_t uplink_frequency_offset[MAX_NUM_CCs][4];

uint64_t downlink_frequency[MAX_NUM_CCs][4];
THREAD_STRUCT thread_struct;

void e1_bearer_context_setup(const e1ap_bearer_setup_req_t *req)
{
  abort();
}
void e1_bearer_context_modif(const e1ap_bearer_setup_req_t *req)
{
  abort();
}
void e1_bearer_release_cmd(const e1ap_bearer_release_cmd_t *cmd)
{
  abort();
}

configmodule_interface_t *uniqCfg = NULL;
int main(int argc, char *argv[])
{
  int i;
  int slot = 8, frame = 1;
  // uint8_t write_output_file = 0;
  int trial, n_trials = 1;
  uint8_t n_rx = 1;
  uint16_t N_RB_UL = 106;

  // unsigned char frame_type = 0;
  NR_DL_FRAME_PARMS *frame_parms;
  int loglvl = OAILOG_WARNING;
  uint8_t precod_nbr_layers = 1;
  int UE_id = 0;
  int print_perf = 0;

  int threadCnt = 0;
  if ((uniqCfg = load_configmodule(argc, argv, CONFIG_ENABLECMDLINEONLY)) == 0) {
    exit_fun("[NR_ULSIM] Error, configuration module init failed\n");
  }
  // logInit();
  randominit(0);

  /* initialize the sin-cos table */
  InitSinLUT();

  int c;
  while ((c = getopt(argc, argv, "--:O:h:n:C:L:P")) != -1) {
    /* ignore long options starting with '--', option '-O' and their arguments that are handled by configmodule */
    /* with this opstring getopt returns 1 for non-option arguments, refer to 'man 3 getopt' */
    if (c == 1 || c == '-' || c == 'O')
      continue;

    printf("handling optarg %c\n", c);
    switch (c) {
      case 'n':
        n_trials = atoi(optarg);
        break;

      case 'C':
        threadCnt = atoi(optarg);
        break;

      case 'L':
        loglvl = atoi(optarg);
        break;

      case 'P':
        print_perf = 1;
        cpu_meas_enabled = 1;
        break;

      default:
      case 'h':
        printf("%s -h(elp)\n", argv[0]);
        printf("-h This message\n");
        printf("-n Number of trials to simulate\n");
        printf("-C Specify the number of threads for the simulation\n");
        printf("-L <log level, 0(errors), 1(warning), 2(info) 3(debug) 4 (trace)>\n");
        printf("-P Print ULSCH performances\n");
        exit(-1);
        break;
    }
  }

  char *params_in_file = NULL;
  char *rxdataF_in_file = NULL;
  char *rxdataF_comp_in_file = NULL;
  char *rxdataF_out_file = NULL;
  char *rxdataF_comp_out_file = NULL;
  paramdef_t LoaderParams[] = {{"params_in_file",
                                NULL,
                                0,
                                .strptr = &params_in_file,
                                .defstrval = "dump_channel_estimation_compensation_params.log",
                                TYPE_STRING,
                                0,
                                NULL},
                               {"rxdataF_in_file",
                                NULL,
                                0,
                                .strptr = &rxdataF_in_file,
                                .defstrval = "dump_channel_estimation_compensation_rxdataF.log",
                                TYPE_STRING,
                                0,
                                NULL},
                               {"rxdataF_comp_in_file",
                                NULL,
                                0,
                                .strptr = &rxdataF_comp_in_file,
                                .defstrval = "dump_channel_estimation_compensation_rxdataF_comp.log",
                                TYPE_STRING,
                                0,
                                NULL},
                               {"rxdataF_out_file",
                                NULL,
                                0,
                                .strptr = &rxdataF_out_file,
                                .defstrval = "test_dump_channel_estimation_compensation_rxdataF.log",
                                TYPE_STRING,
                                0,
                                NULL},
                               {"rxdataF_comp_out_file",
                                NULL,
                                0,
                                .strptr = &rxdataF_comp_out_file,
                                .defstrval = "test_dump_channel_estimation_compensation_rxdataF_comp.log",
                                TYPE_STRING,
                                0,
                                NULL}};
  config_get(config_get_if(), LoaderParams, sizeofArray(LoaderParams), "nr_chestcompsim");

  chestcomp_params_t chestcomp_params = {0};
  int beam_nb = 0;
  int ulsch_id = 0;
  read_channel_estimation_compensation_parameters(gNB, &frame, &slot, &beam_nb, &ulsch_id, &chestcomp_params, params_in_file);

  N_RB_UL = chestcomp_params.N_RB_UL;

  n_rx = chestcomp_params.nb_antennas_rx;
  if ((n_rx == 0) || (n_rx > 8)) {
    printf("Unsupported number of rx antennas %d\n", n_rx);
    exit(-1);
  }

  precod_nbr_layers = chestcomp_params.nrOfLayers;

  logInit();
  set_glog(loglvl);

  get_softmodem_params()->phy_test = 1;
  get_softmodem_params()->do_ra = 0;
  get_softmodem_params()->usim_test = 1;

  RC.gNB = (PHY_VARS_gNB **)malloc(sizeof(PHY_VARS_gNB *));
  RC.gNB[0] = calloc(1, sizeof(PHY_VARS_gNB));
  gNB = RC.gNB[0];

  initFloatingCoresTpool(threadCnt, &gNB->threadPool, false, "gNB-tpool");

  //---------------
  int ret = 1;

  reset_meas(&gNB->rx_pusch_stats);
  reset_meas(&gNB->ulsch_channel_estimation_stats);
  reset_meas(&gNB->pusch_channel_estimation_antenna_processing_stats);
  reset_meas(&gNB->ulsch_channel_compensation_stats);

  load_dftslib();

  gNB->max_nb_pusch = 1;
  gNB->ulsch = (NR_gNB_ULSCH_t *)malloc16(1 * sizeof(NR_gNB_ULSCH_t));
  for (int i = 0; i < gNB->max_nb_pusch; i++) {
    LOG_D(PHY, "Allocating Transport Channel Buffers for ULSCH %d/%d\n", i, gNB->max_nb_pusch);
    gNB->ulsch[i] = new_gNB_ulsch(8, N_RB_UL);

    nfapi_nr_pusch_pdu_t *rel15_ul = &gNB->ulsch[i].harq_process->ulsch_pdu;
    rel15_ul->dmrs_ports = ((1 << precod_nbr_layers) - 1);
  }

  set_channel_estimation_compensation_parameters(chestcomp_params, gNB, &frame, &slot, &beam_nb, &ulsch_id);

  frame_parms = &gNB->frame_parms;

  init_delay_table(frame_parms->ofdm_symbol_size, MAX_DELAY_COMP, NR_MAX_OFDM_SYMBOL_SIZE, frame_parms->delay_table);

  /* Do NOT allocate per-antenna rxdataF: the gNB gets a pointer to the
   * RU to copy/recover freq-domain memory from there */
  gNB->common_vars.rxdataF = (c16_t ***)malloc16(1 * sizeof(c16_t **));
  for (int i = 0; i < 1; i++)
    gNB->common_vars.rxdataF[i] = (c16_t **)malloc16(n_rx * sizeof(c16_t *));

  /* RU handles rxdataF, and gNB just has a pointer. Here, we don't have an RU,
   * so we need to allocate that memory as well. */
  int nb_samples_per_antenna = gNB->frame_parms.symbols_per_slot * gNB->frame_parms.ofdm_symbol_size;
  for (i = 0; i < n_rx; i++)
    gNB->common_vars.rxdataF[0][i] = malloc16_clear(nb_samples_per_antenna * sizeof(c16_t));

  int max_ul_mimo_layers = 4;
  int n_buf = n_rx * max_ul_mimo_layers;
  int nb_re_pusch = N_RB_UL * NR_NB_SC_PER_RB;
  int nb_re_pusch2 = ceil_mod(nb_re_pusch, 16);
  gNB->pusch_vars = (NR_gNB_PUSCH *)malloc16_clear(1 * sizeof(NR_gNB_PUSCH));
  for (int ULSCH_id = 0; ULSCH_id < gNB->max_nb_pusch; ULSCH_id++) {
    NR_gNB_PUSCH *pusch = &gNB->pusch_vars[ULSCH_id];
    pusch->ul_ch_estimates = (int32_t **)malloc16(n_buf * sizeof(int32_t *));
    // pusch->ptrs_phase_per_slot = (int32_t **)malloc16(n_buf * sizeof(int32_t *));
    pusch->ul_ch_estimates_time = (int32_t **)malloc16(n_buf * sizeof(int32_t *));
    pusch->rxdataF_comp = (int32_t **)malloc16(n_buf * sizeof(int32_t *));
    for (int i = 0; i < n_buf; i++) {
      pusch->ul_ch_estimates[i] =
          (int32_t *)malloc16_clear(sizeof(int32_t) * frame_parms->ofdm_symbol_size * frame_parms->symbols_per_slot);
      pusch->ul_ch_estimates_time[i] = (int32_t *)malloc16_clear(sizeof(int32_t) * frame_parms->ofdm_symbol_size);
      // pusch->ptrs_phase_per_slot[i] = (int32_t *)malloc16_clear(sizeof(int32_t) * frame_parms->symbols_per_slot); // symbols per
      // slot
      pusch->rxdataF_comp[i] = (int32_t *)malloc16_clear(sizeof(int32_t) * nb_re_pusch2 * frame_parms->symbols_per_slot);
    }
    pusch->ul_valid_re_per_slot = (int16_t *)malloc16_clear(sizeof(int16_t) * frame_parms->symbols_per_slot);
  } // ulsch_id

  //----------------------------------------------------------
  //------------------- gNB phy procedures -------------------
  //----------------------------------------------------------

  bool chestcomperror = false;
  int chestcomperror_count = 0;

  nfapi_nr_pusch_pdu_t *rel15_ul = &gNB->ulsch[0].harq_process->ulsch_pdu;
  int buffer_length = ceil_mod(rel15_ul->rb_size * NR_NB_SC_PER_RB, 16);
  int nb_samples_per_antenna_layer = rel15_ul->nr_of_symbols * buffer_length;
  int32_t rxdataF_comp_ref[rel15_ul->nrOfLayers][nb_samples_per_antenna_layer];
  memset(&rxdataF_comp_ref[0][0], 0, rel15_ul->nrOfLayers * nb_samples_per_antenna_layer * sizeof(int32_t));

  read_channel_estimation_compensation_in_out(gNB, 0, 0, rxdataF_in_file, rxdataF_comp_in_file, &rxdataF_comp_ref[0][0]);

  for (int layer = 0; layer < rel15_ul->nrOfLayers; layer++) {
    int32_t *rxdataF_comp =
        &gNB->pusch_vars[0].rxdataF_comp[layer * gNB->frame_parms.nb_antennas_rx][rel15_ul->start_symbol_index * buffer_length];
    memset(rxdataF_comp, 0, nb_samples_per_antenna_layer * sizeof(int32_t));
  }

  for (trial = 0; trial < n_trials; trial++) {
    channel_estimation_compensation_test(gNB, frame, slot);

    for (int layer = 0; layer < rel15_ul->nrOfLayers; layer++) {
      int32_t *rxdataF_comp =
          &gNB->pusch_vars[0].rxdataF_comp[layer * gNB->frame_parms.nb_antennas_rx][rel15_ul->start_symbol_index * buffer_length];
      if (memcmp(rxdataF_comp, &rxdataF_comp_ref[layer][0], nb_samples_per_antenna_layer * sizeof(int32_t)) != 0) {
        if (!chestcomperror) {
          LOG_E(PHY, "Error in channel compensation\n");
          chestcomperror = true;
        }
        chestcomperror_count++;
        break;
      }
    }
  }

#ifdef DUMP_CH_EST_COMP_IN_OUT
  dump_channel_estimation_compensation_in_out(gNB, 0, 0, rxdataF_out_file, rxdataF_comp_out_file);
#endif

  LOG_M("rxsigF0.m", "rxsF0", gNB->common_vars.rxdataF[0][0], 14 * frame_parms->ofdm_symbol_size, 1, 1);
  if (precod_nbr_layers > 1) {
    LOG_M("rxsigF1.m", "rxsF1", gNB->common_vars.rxdataF[0][1], 14 * frame_parms->ofdm_symbol_size, 1, 1);
    if (precod_nbr_layers == 4) {
      LOG_M("rxsigF2.m", "rxsF2", gNB->common_vars.rxdataF[0][2], 14 * frame_parms->ofdm_symbol_size, 1, 1);
      LOG_M("rxsigF3.m", "rxsF3", gNB->common_vars.rxdataF[0][3], 14 * frame_parms->ofdm_symbol_size, 1, 1);
    }
  }

  NR_gNB_PUSCH *pusch_vars = &gNB->pusch_vars[UE_id];
  __attribute__((unused)) int off = ((rel15_ul->rb_size & 1) == 1) ? 4 : 0;

  LOG_M("chestF0.m",
        "chF0",
        &pusch_vars->ul_ch_estimates[0][rel15_ul->start_symbol_index * frame_parms->ofdm_symbol_size],
        frame_parms->ofdm_symbol_size,
        1,
        1);

  LOG_M("rxsigF0_comp.m",
        "rxsF0_comp",
        &pusch_vars->rxdataF_comp[0][rel15_ul->start_symbol_index * (off + (NR_NB_SC_PER_RB * rel15_ul->rb_size))],
        rel15_ul->nr_of_symbols * (off + (NR_NB_SC_PER_RB * rel15_ul->rb_size)),
        1,
        1);

  if (precod_nbr_layers == 2) {
    LOG_M("chestF3.m",
          "chF3",
          &pusch_vars->ul_ch_estimates[3][rel15_ul->start_symbol_index * frame_parms->ofdm_symbol_size],
          frame_parms->ofdm_symbol_size,
          1,
          1);

    LOG_M("rxsigF2_comp.m",
          "rxsF2_comp",
          &pusch_vars->rxdataF_comp[2][rel15_ul->start_symbol_index * (off + (NR_NB_SC_PER_RB * rel15_ul->rb_size))],
          rel15_ul->nr_of_symbols * (off + (NR_NB_SC_PER_RB * rel15_ul->rb_size)),
          1,
          1);
  }

  if (precod_nbr_layers == 4) {
    LOG_M("chestF5.m",
          "chF5",
          &pusch_vars->ul_ch_estimates[5][rel15_ul->start_symbol_index * frame_parms->ofdm_symbol_size],
          frame_parms->ofdm_symbol_size,
          1,
          1);
    LOG_M("chestF10.m",
          "chF10",
          &pusch_vars->ul_ch_estimates[10][rel15_ul->start_symbol_index * frame_parms->ofdm_symbol_size],
          frame_parms->ofdm_symbol_size,
          1,
          1);
    LOG_M("chestF15.m",
          "chF15",
          &pusch_vars->ul_ch_estimates[15][rel15_ul->start_symbol_index * frame_parms->ofdm_symbol_size],
          frame_parms->ofdm_symbol_size,
          1,
          1);

    LOG_M("rxsigF4_comp.m",
          "rxsF4_comp",
          &pusch_vars->rxdataF_comp[4][rel15_ul->start_symbol_index * (off + (NR_NB_SC_PER_RB * rel15_ul->rb_size))],
          rel15_ul->nr_of_symbols * (off + (NR_NB_SC_PER_RB * rel15_ul->rb_size)),
          1,
          1);
    LOG_M("rxsigF8_comp.m",
          "rxsF8_comp",
          &pusch_vars->rxdataF_comp[8][rel15_ul->start_symbol_index * (off + (NR_NB_SC_PER_RB * rel15_ul->rb_size))],
          rel15_ul->nr_of_symbols * (off + (NR_NB_SC_PER_RB * rel15_ul->rb_size)),
          1,
          1);
    LOG_M("rxsigF12_comp.m",
          "rxsF12_comp",
          &pusch_vars->rxdataF_comp[12][rel15_ul->start_symbol_index * (off + (NR_NB_SC_PER_RB * rel15_ul->rb_size))],
          rel15_ul->nr_of_symbols * (off + (NR_NB_SC_PER_RB * rel15_ul->rb_size)),
          1,
          1);
  }

  printf("\n");

  //----------------------------------------------------------
  //----------------- count and print errors -----------------
  //----------------------------------------------------------

  // TODO

  if (print_perf == 1) {
    printf("gNB RX\n");
    printStatIndent(&gNB->rx_pusch_stats, "RX PUSCH time");
    printStatIndent2(&gNB->ulsch_channel_estimation_stats, "ULSCH channel estimation time");
    printStatIndent3(&gNB->pusch_channel_estimation_antenna_processing_stats, "Antenna Processing time");
    printStatIndent2(&gNB->ulsch_channel_compensation_stats, "ULSCH channel compensation time");
    printf("\n");
  }

  if (!chestcomperror) {
    printf("*****************************************\n");
    printf("Channel estimation & compensation test OK\n");
    printf("*****************************************\n");
    ret = 0;
  } else {
    printf("Channel estimation & compensation test NOK\n");
    printf("number of errors / number of trials: %d / %d\n", chestcomperror_count, n_trials);
    ret = chestcomperror_count;
  }

  printf("\n");
  printf(
      "Num RB:\t%d\n"
      "Num symbols:\t%d\n"
      "DMRS config type:\t%d\n"
      "DMRS symb pos:\t%04x\n"
      "DMRS length:\t%d\n"
      "DMRS CDM gr w/o data:\t%d\n",
      rel15_ul->rb_size,
      rel15_ul->nr_of_symbols,
      rel15_ul->dmrs_config_type,
      rel15_ul->ul_dmrs_symb_pos,
      pusch_len1,
      rel15_ul->num_dmrs_cdm_grps_no_data);

  for (int i = 0; i < gNB->max_nb_pusch; i++) {
    free_gNB_ulsch(&gNB->ulsch[i], chestcomp_params.N_RB_UL);
  }
  free(gNB->ulsch);

  for (i = 0; i < n_rx; i++)
    free(gNB->common_vars.rxdataF[0][i]);
  for (int i = 0; i < 1; i++)
    free(gNB->common_vars.rxdataF[i]);
  free(gNB->common_vars.rxdataF);

  for (int ULSCH_id = 0; ULSCH_id < gNB->max_nb_pusch; ULSCH_id++) {
    NR_gNB_PUSCH *pusch = &gNB->pusch_vars[ULSCH_id];
    for (int i = 0; i < n_buf; i++) {
      free(pusch->ul_ch_estimates[i]);
      free(pusch->ul_ch_estimates_time[i]);
      // free(pusch->ptrs_phase_per_slot[i]);
      free(pusch->rxdataF_comp[i]);
    }
    free(pusch->ul_ch_estimates);
    // free(pusch->ptrs_phase_per_slot);
    free(pusch->ul_ch_estimates_time);
    free(pusch->rxdataF_comp);
    free(pusch->ul_valid_re_per_slot);
  } // ulsch_id
  free(gNB->pusch_vars);

  return ret;
}
