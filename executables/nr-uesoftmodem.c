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


#define _GNU_SOURCE             /* See feature_test_macros(7) */
#include <sched.h>
#include <stdbool.h>
#include <signal.h>
#include <errno.h>

#include "T.h"
#include "common/oai_version.h"
#include "assertions.h"
#include "PHY/types.h"
#include "PHY/defs_nr_UE.h"
#include "SCHED_NR_UE/defs.h"
#include "common/ran_context.h"
#include "common/config/config_userapi.h"
#include "common/utils/load_module_shlib.h"
//#undef FRAME_LENGTH_COMPLEX_SAMPLES //there are two conflicting definitions, so we better make sure we don't use it at all
#include "common/utils/nr/nr_common.h"

#include "radio/COMMON/common_lib.h"
#include "radio/ETHERNET/if_defs.h"

//#undef FRAME_LENGTH_COMPLEX_SAMPLES //there are two conflicting definitions, so we better make sure we don't use it at all
#include "openair1/PHY/MODULATION/nr_modulation.h"
#include "PHY/CODING/nrLDPC_coding/nrLDPC_coding_interface.h"
#include "PHY/phy_vars_nr_ue.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_proto_ue.h"
#include "PHY/NR_TRANSPORT/nr_dlsch.h"
//#include "../../SIMU/USER/init_lte.h"

#include "PHY_INTERFACE/phy_interface_vars.h"
#include "NR_IF_Module.h"
#include "openair1/SIMULATION/TOOLS/sim.h"
#include "openair2/RRC/NR_UE/L2_interface_ue.h"

#ifdef SMBV
#include "PHY/TOOLS/smbv.h"
unsigned short config_frames[4] = {2,9,11,13};
#endif
#include "common/utils/LOG/log.h"
#include "common/utils/time_manager/time_manager.h"
#include "common/utils/LOG/vcd_signal_dumper.h"

#include "UTIL/OPT/opt.h"
#include "LAYER2/nr_pdcp/nr_pdcp_oai_api.h"

#include "intertask_interface.h"

#include "PHY/INIT/nr_phy_init.h"
#include "system.h"
#include <openair2/RRC/NR_UE/rrc_proto.h>
#include <openair2/LAYER2/NR_MAC_UE/mac_defs.h>
#include <openair2/LAYER2/NR_MAC_UE/mac_proto.h>
#include <openair2/NR_UE_PHY_INTERFACE/NR_IF_Module.h>
#include <openair1/SCHED_NR_UE/fapi_nr_ue_l1.h>
#include "nr_rlc/nr_rlc_oai_api.h"
/* Callbacks, globals and object handlers */

//#include "stats.h"
// current status is that every UE has a DL scope for a SINGLE eNB (eNB_id=0)
#include "PHY/TOOLS/phy_scope_interface.h"
#include "PHY/TOOLS/nr_phy_scope.h"
#include <executables/nr-ue-ru.h>
#include <executables/nr-uesoftmodem.h>
#include "executables/softmodem-common.h"
#include "executables/thread-common.h"

#include "nr_nas_msg.h"
#include <openair1/PHY/MODULATION/nr_modulation.h>
#include "openair2/GNB_APP/gnb_paramdef.h"
#include "actor.h"

THREAD_STRUCT thread_struct;
nrUE_params_t nrUE_params = {0};

// Thread variables
pthread_cond_t nfapi_sync_cond;
pthread_mutex_t nfapi_sync_mutex;
int nfapi_sync_var=-1; //!< protected by mutex \ref nfapi_sync_mutex
pthread_cond_t sync_cond;
pthread_mutex_t sync_mutex;
int sync_var=-1; //!< protected by mutex \ref sync_mutex.
int config_sync_var=-1;

// not used in UE
instance_t CUuniqInstance=0;
instance_t DUuniqInstance=0;

int get_node_type() {return -1;}

RAN_CONTEXT_t RC;
int oai_exit = 0;


uint64_t        downlink_frequency[MAX_NUM_CCs][4];
int32_t         uplink_frequency_offset[MAX_NUM_CCs][4];
uint64_t        sidelink_frequency[MAX_NUM_CCs][4];

// UE and OAI config variables
extern openair0_config_t openair0_cfg[MAX_CARDS];
extern openair0_device_t openair0_dev[MAX_CARDS];
extern NR_DL_FRAME_PARMS cell_fp[MAX_CARDS];
int16_t           node_synch_ref[MAX_NUM_CCs];
int               otg_enabled;
double            cpuf;

int create_tasks_nrue(uint32_t ue_nb) {
  LOG_D(NR_RRC, "%s(ue_nb:%d)\n", __FUNCTION__, ue_nb);
  itti_wait_ready(1);

  if (ue_nb > 0) {
    LOG_I(NR_RRC,"create TASK_RRC_NRUE \n");
    const ittiTask_parms_t parmsRRC = {NULL, rrc_nrue};
    if (itti_create_task(TASK_RRC_NRUE, rrc_nrue_task, &parmsRRC) < 0) {
      LOG_E(NR_RRC, "Create task for RRC UE failed\n");
      return -1;
    }
    if (get_softmodem_params()->nsa) {
      init_connections_with_lte_ue();
      if (itti_create_task (TASK_RRC_NSA_NRUE, recv_msgs_from_lte_ue, NULL) < 0) {
        LOG_E(NR_RRC, "Create task for RRC NSA nr-UE failed\n");
        return -1;
      }
    }
    const ittiTask_parms_t parmsNAS = {NULL, nas_nrue};
    if (itti_create_task(TASK_NAS_NRUE, nas_nrue_task, &parmsNAS) < 0) {
      LOG_E(NR_RRC, "Create task for NAS UE failed\n");
      return -1;
    }
  }

  itti_wait_ready(0);

  return 0;
}

void exit_function(const char *file, const char *function, const int line, const char *s, const int assert)
{
  if (s != NULL) {
    printf("%s:%d %s() Exiting OAI softmodem: %s\n",file,line, function, s);
  }

  oai_exit = 1;

  nr_ue_ru_end();

  if (assert) {
    abort();
  } else {
    sleep(1); // allow lte-softmodem threads to exit first
    exit(EXIT_SUCCESS);
  }
}

uint64_t get_nrUE_optmask(void) {
  return nrUE_params.optmask;
}

uint64_t set_nrUE_optmask(uint64_t bitmask) {
  nrUE_params.optmask = nrUE_params.optmask | bitmask;
  return nrUE_params.optmask;
}

nrUE_params_t *get_nrUE_params(void) {
  return &nrUE_params;
}

static int get_nrUE_RU_params(configmodule_interface_t *cfg, nrUE_RU_params_t **RUs)
{
  paramdef_t RUParams[] = NRUE_RU_PARAMS_DESC;
  paramlist_def_t RUParamList = {CONFIG_STRING_NRUE_RU_LIST, NULL, 0};
  config_getlist(cfg, &RUParamList, RUParams, sizeofArray(RUParams), NULL);

  if (RUParamList.numelt <= 0) {
    *RUs = (nrUE_RU_params_t *)calloc_or_fail(1, sizeof(nrUE_RU_params_t));
    (*RUs)[0] = (nrUE_RU_params_t){
      .nb_tx          = nrUE_params.nb_antennas_tx,
      .nb_rx          = nrUE_params.nb_antennas_rx,
      .att_tx         = nrUE_params.tx_gain,
      .att_rx         = 0,
      .max_rxgain     = nrUE_params.rx_gain,
      .sdr_addrs      = nrUE_params.usrp_args,
      .tx_subdev      = nrUE_params.tx_subdev,
      .rx_subdev      = nrUE_params.rx_subdev,
      .clock_source   = get_softmodem_params()->clock_source,
      .time_source    = get_softmodem_params()->timing_source,
      .tune_offset    = get_softmodem_params()->tune_offset,
      .if_frequency   = nrUE_params.if_freq,
      .if_freq_offset = nrUE_params.if_freq_off,
      .used_by_cell   = -1
    };
    return 1;
  }

  *RUs = (nrUE_RU_params_t *)calloc_or_fail(RUParamList.numelt, sizeof(nrUE_RU_params_t));

  for (int j = 0; j < RUParamList.numelt; j++) {
    nrUE_RU_params_t *RU = &(*RUs)[j];

    RU->nb_tx          = *(RUParamList.paramarray[j][NRUE_RU_NB_TX_IDX].uptr);
    RU->nb_rx          = *(RUParamList.paramarray[j][NRUE_RU_NB_RX_IDX].uptr);
    RU->att_tx         = *(RUParamList.paramarray[j][NRUE_RU_ATT_TX_IDX].uptr);
    RU->att_rx         = *(RUParamList.paramarray[j][NRUE_RU_ATT_RX_IDX].uptr);
    RU->max_rxgain     = *(RUParamList.paramarray[j][NRUE_RU_MAX_RXGAIN_IDX].iptr);
    RU->tune_offset    = *(RUParamList.paramarray[j][NRUE_RU_TUNE_OFFSET_IDX].dblptr);
    RU->if_frequency   = *(RUParamList.paramarray[j][NRUE_RU_IF_FREQUENCY_IDX].u64ptr);
    RU->if_freq_offset = *(RUParamList.paramarray[j][NRUE_RU_IF_FREQ_OFFSET_IDX].iptr);
    RU->rfsim_serverport = *(RUParamList.paramarray[j][NRUE_RU_RFSIM_SERVPORT_IDX].u16ptr);
    RU->rfsim_prop_delay = *(RUParamList.paramarray[j][NRUE_RU_RFSIM_DELAY_IDX].dblptr);
    RU->used_by_cell   = -1;

    if (RUParamList.paramarray[j][NRUE_RU_RFSIM_SERVADDR_IDX].strptr != NULL) {
      RU->rfsim_serveraddr = strdup(*(RUParamList.paramarray[j][NRUE_RU_RFSIM_SERVADDR_IDX].strptr));
    }

    if (config_isparamset(RUParamList.paramarray[j], NRUE_RU_SDR_ADDRS_IDX)) {
      RU->sdr_addrs = strdup(*(RUParamList.paramarray[j][NRUE_RU_SDR_ADDRS_IDX].strptr));
    }

    if (config_isparamset(RUParamList.paramarray[j], NRUE_RU_TX_SUBDEV_IDX)) {
      RU->tx_subdev = strdup(*(RUParamList.paramarray[j][NRUE_RU_TX_SUBDEV_IDX].strptr));
    }

    if (config_isparamset(RUParamList.paramarray[j], NRUE_RU_RX_SUBDEV_IDX)) {
      RU->rx_subdev = strdup(*(RUParamList.paramarray[j][NRUE_RU_RX_SUBDEV_IDX].strptr));
    }

    if (config_isparamset(RUParamList.paramarray[j], NRUE_RU_CLOCK_SRC_IDX)) {
      if (strcmp(*(RUParamList.paramarray[j][NRUE_RU_CLOCK_SRC_IDX].strptr), "internal") == 0) {
        RU->clock_source = internal;
      } else if (strcmp(*(RUParamList.paramarray[j][NRUE_RU_CLOCK_SRC_IDX].strptr), "external") == 0) {
        RU->clock_source = external;
      } else if (strcmp(*(RUParamList.paramarray[j][NRUE_RU_CLOCK_SRC_IDX].strptr), "gpsdo") == 0) {
        RU->clock_source = gpsdo;
      } else {
        RU->clock_source = internal;
        LOG_E(PHY, "Erroneous RU clock source in the provided configuration file: '%s'\n", *(RUParamList.paramarray[j][NRUE_RU_CLOCK_SRC_IDX].strptr));
      }
    } else {
      RU->clock_source = internal;
    }

    if (config_isparamset(RUParamList.paramarray[j], NRUE_RU_TIME_SRC_IDX)) {
      if (strcmp(*(RUParamList.paramarray[j][NRUE_RU_TIME_SRC_IDX].strptr), "internal") == 0) {
        RU->time_source = internal;
      } else if (strcmp(*(RUParamList.paramarray[j][NRUE_RU_TIME_SRC_IDX].strptr), "external") == 0) {
        RU->time_source = external;
      } else if (strcmp(*(RUParamList.paramarray[j][NRUE_RU_TIME_SRC_IDX].strptr), "gpsdo") == 0) {
        RU->time_source = gpsdo;
      } else {
        RU->time_source = internal;
        LOG_E(PHY, "Erroneous RU time source in the provided configuration file: '%s'\n", *(RUParamList.paramarray[j][NRUE_RU_TIME_SRC_IDX].strptr));
      }
    } else {
      RU->time_source = internal;
    }
  }

  return RUParamList.numelt;
}

static int get_nrUE_cell_params(configmodule_interface_t *cfg, nrUE_cell_params_t **cells)
{
  paramdef_t cellParams[] = NRUE_CELL_PARAMS_DESC;
  paramlist_def_t cellParamList = {CONFIG_STRING_NRUE_CELL_LIST, NULL, 0};
  config_getlist(cfg, &cellParamList, cellParams, sizeofArray(cellParams), NULL);

  if (cellParamList.numelt <= 0) {
    *cells = (nrUE_cell_params_t *)calloc_or_fail(1, sizeof(nrUE_cell_params_t));
    (*cells)[0] = (nrUE_cell_params_t){
      .rf_port        = 0,
      .band           = get_softmodem_params()->band,
      .rf_frequency   = downlink_frequency[0][0],
      .rf_freq_offset = uplink_frequency_offset[0][0],
      .numerology     = get_softmodem_params()->numerology,
      .N_RB_DL        = nrUE_params.N_RB_DL,
      .ssb_start      = nrUE_params.ssb_start_subcarrier,
      .used_by_ue     = -1
    };
    return 1;
  }

  *cells = (nrUE_cell_params_t *)calloc_or_fail(cellParamList.numelt, sizeof(nrUE_cell_params_t));

  for (int j = 0; j < cellParamList.numelt; j++) {
    nrUE_cell_params_t *cell = &(*cells)[j];

    cell->rf_port        = *(cellParamList.paramarray[j][NRUE_CELL_RF_PORT_IDX].iptr);
    cell->band           = *(cellParamList.paramarray[j][NRUE_CELL_BAND_IDX].iptr);
    cell->rf_frequency   = *(cellParamList.paramarray[j][NRUE_CELL_RF_FREQUENCY_IDX].u64ptr);
    cell->rf_freq_offset = *(cellParamList.paramarray[j][NRUE_CELL_RF_FREQ_OFFSET_IDX].iptr);
    cell->numerology     = *(cellParamList.paramarray[j][NRUE_CELL_NUMEROLOGY_IDX].iptr);
    cell->N_RB_DL        = *(cellParamList.paramarray[j][NRUE_CELL_N_RB_DL_IDX].iptr);
    cell->ssb_start      = *(cellParamList.paramarray[j][NRUE_CELL_SSB_START_IDX].iptr);
    cell->used_by_ue     = -1;
  }

  return cellParamList.numelt;
}

static void get_options(configmodule_interface_t *cfg)
{
  paramdef_t cmdline_params[] = CMDLINE_NRUEPARAMS_DESC;
  int numparams = sizeofArray(cmdline_params);
  config_get(cfg, cmdline_params, numparams, NULL);
  if (nrUE_params.vcdflag > 0)
    ouput_vcd = 1;
  AssertFatal(nrUE_params.extra_pdu_id != get_softmodem_params()->default_pdu_session_id,
              "Default PDU ID (%d) and Extra PDU ID (%d) must be different!\n",
              get_softmodem_params()->default_pdu_session_id,
              nrUE_params.extra_pdu_id);
}

// set PHY vars from command line
static void set_UE_options(int CC_id, PHY_VARS_NR_UE *UE, int card, const nrUE_RU_params_t *RU)
{
  // Set UE variables
  UE->rx_total_gain_dB = RU->max_rxgain - RU->att_rx;
  UE->tx_total_gain_dB = RU->att_tx;
  UE->if_freq          = RU->if_frequency;
  UE->if_freq_off      = RU->if_freq_offset;
  UE->rf_map.card      = card;
  UE->rf_map.chain     = CC_id;

  UE->tx_power_max_dBm     = nrUE_params.tx_max_power;
  UE->max_ldpc_iterations  = nrUE_params.max_ldpc_iterations;
  UE->UE_scan_carrier      = nrUE_params.UE_scan_carrier;
  UE->UE_fo_compensation   = nrUE_params.UE_fo_compensation;
  UE->chest_freq           = nrUE_params.chest_freq;
  UE->chest_time           = nrUE_params.chest_time;
  UE->no_timing_correction = nrUE_params.no_timing_correction;
  UE->initial_fo           = nrUE_params.initial_fo;
  UE->cont_fo_comp         = nrUE_params.cont_fo_comp;

  LOG_I(PHY,"Set UE_fo_compensation %d, UE_scan_carrier %d, UE_no_timing_correction %d \n, chest-freq %d, chest-time %d\n",
        UE->UE_fo_compensation, UE->UE_scan_carrier, UE->no_timing_correction, UE->chest_freq, UE->chest_time);
}

static void set_fp_options(NR_DL_FRAME_PARMS *fp, const nrUE_RU_params_t *RU)
{
  // Set FP variables
  fp->nb_antennas_rx = RU->nb_rx;
  fp->nb_antennas_tx = RU->nb_tx;

  fp->threequarter_fs     = get_softmodem_params()->threequarter_fs;
  fp->ofdm_offset_divisor = nrUE_params.ofdm_offset_divisor;

  LOG_I(PHY, "Set UE nb_rx_antenna %d, nb_tx_antenna %d, threequarter_fs %d, ofdm_offset_divisor %d\n", fp->nb_antennas_rx, fp->nb_antennas_tx, fp->threequarter_fs, fp->ofdm_offset_divisor);
}

static void init_openair0(const nrUE_RU_params_t *RUs, int max_cards)
{
  int freq_off = 0;
  bool is_sidelink = (get_softmodem_params()->sl_mode) ? true : false;

  for (int card = 0; card < max_cards; card++) {
    NR_DL_FRAME_PARMS *frame_parms;
    if (is_sidelink)
      frame_parms = &PHY_vars_UE_g[0][0]->SL_UE_PHY_PARAMS.sl_frame_params;
    else {
      int cell_id = RUs[card].used_by_cell;
      if (cell_id == -1) {
        LOG_W(PHY, "Skipping initialization of RU %d because it is not used by any cell!\n", card);
        continue;
      }
      frame_parms = &cell_fp[cell_id];
    }

    openair0_config_t *cfg = &openair0_cfg[card];
    cfg->configFilename    = NULL;
    cfg->sample_rate       = frame_parms->samples_per_subframe * 1e3;
    cfg->samples_per_frame = frame_parms->samples_per_frame;

    if (frame_parms->frame_type == TDD)
      cfg->duplex_mode = duplex_mode_TDD;
    else
      cfg->duplex_mode = duplex_mode_FDD;

    cfg->Mod_id = 0;
    cfg->num_rb_dl = frame_parms->N_RB_DL;
    cfg->tx_num_channels = min(4, frame_parms->nb_antennas_tx);
    cfg->rx_num_channels = min(4, frame_parms->nb_antennas_rx);

    LOG_I(PHY,
          "HW: Configuring card %d, sample_rate %f, tx/rx num_channels %d/%d, duplex_mode %s\n",
          card,
          cfg->sample_rate,
          cfg->tx_num_channels,
          cfg->rx_num_channels,
          duplex_mode_txt[cfg->duplex_mode]);

    uint64_t dl_carrier, ul_carrier;
    if (is_sidelink || RUs[card].if_frequency == 0) {
      dl_carrier = frame_parms->dl_CarrierFreq;
      ul_carrier = frame_parms->ul_CarrierFreq;
    } else {
      dl_carrier = RUs[card].if_frequency;
      ul_carrier = RUs[card].if_frequency + RUs[card].if_freq_offset;
    }

    nr_rf_card_config_freq(cfg, ul_carrier, dl_carrier, freq_off);
    nr_rf_card_config_gain(cfg);

    cfg->configFilename = get_softmodem_params()->rf_config_file;

    cfg->sdr_addrs = RUs[card].sdr_addrs;
    cfg->tx_subdev = RUs[card].tx_subdev;
    cfg->rx_subdev = RUs[card].rx_subdev;
    cfg->clock_source = RUs[card].clock_source;
    cfg->time_source = RUs[card].time_source;
    cfg->tune_offset = RUs[card].tune_offset;

    openair0_device_t *dev = &openair0_dev[card];
    dev->rfsim_params.card = card;
    dev->rfsim_params.serveraddr = RUs[card].rfsim_serveraddr;
    dev->rfsim_params.serverport = RUs[card].rfsim_serverport;
    dev->rfsim_params.prop_delay_ms = RUs[card].rfsim_prop_delay;
  }
}

static void init_pdcp(int ue_id)
{
  if (get_softmodem_params()->nsa && nr_rlc_module_init(NR_RLC_OP_MODE_UE) != 0) {
    LOG_I(RLC, "Problem at RLC initiation \n");
  }
  nr_pdcp_layer_init();
}

// Stupid function addition because UE itti messages queues definition is common with eNB
void *rrc_enb_process_msg(void *notUsed) {
  return NULL;
}

static bool stop_immediately = false;
static void trigger_stop(int sig)
{
  if (!oai_exit)
    itti_wait_tasks_unblock();
}
static void trigger_deregistration(int sig)
{
  if (!stop_immediately && IS_SA_MODE(get_softmodem_params())) {
    for (int ue_inst = 0; ue_inst < NB_UE_INST; ue_inst++) {
      MessageDef *msg = itti_alloc_new_message(TASK_NAS_NRUE, ue_inst, NAS_DEREGISTRATION_REQ);
      NAS_DEREGISTRATION_REQ(msg).cause = AS_DETACH;
      itti_send_msg_to_task(TASK_NAS_NRUE, ue_inst, msg);
    }
    stop_immediately = true;
    static const char m[] = "Press ^C again to trigger immediate shutdown\n";
    __attribute__((unused)) int unused = write(STDOUT_FILENO, m, sizeof(m) - 1);
    signal(SIGALRM, trigger_stop);
    alarm(5);
  } else {
    itti_wait_tasks_unblock();
  }
}

static void get_channel_model_mode(configmodule_interface_t *cfg)
{
  paramdef_t GNBParams[]  = GNBPARAMS_DESC;
  config_get(cfg, GNBParams, sizeofArray(GNBParams), NULL);
  int num_xp_antennas = *GNBParams[GNB_PDSCH_ANTENNAPORTS_XP_IDX].iptr;

  if (num_xp_antennas == 2)
    init_nr_bler_table("NR_MIMO2x2_AWGN_RESULTS_DIR");
  else
    init_nr_bler_table("NR_AWGN_RESULTS_DIR");
}

void start_oai_nrue_threads()
{
    init_queue(&nr_rach_ind_queue);
    init_queue(&nr_rx_ind_queue);
    init_queue(&nr_crc_ind_queue);
    init_queue(&nr_uci_ind_queue);
    init_queue(&nr_sfn_slot_queue);
    init_queue(&nr_chan_param_queue);
    init_queue(&nr_dl_tti_req_queue);
    init_queue(&nr_tx_req_queue);
    init_queue(&nr_ul_dci_req_queue);
    init_queue(&nr_ul_tti_req_queue);

    if (sem_init(&sfn_slot_semaphore, 0, 0) != 0)
    {
      LOG_E(MAC, "sem_init() error\n");
      abort();
    }

    init_nrUE_standalone_thread(ue_id_g);
}

int NB_UE_INST = 1;
configmodule_interface_t *uniqCfg = NULL;
nrLDPC_coding_interface_t nrLDPC_coding_interface = {0};

int main(int argc, char **argv)
{
  start_background_system();

  if ((uniqCfg = load_configmodule(argc, argv, CONFIG_ENABLECMDLINEONLY)) == NULL) {
    exit_fun("[SOFTMODEM] Error, configuration module init failed\n");
  }
  //set_softmodem_sighandler();
  CONFIG_SETRTFLAG(CONFIG_NOEXITONHELP);
  memset(openair0_cfg, 0, sizeof(openair0_config_t) * MAX_CARDS);
  memset(openair0_dev, 0, sizeof(openair0_device_t) * MAX_CARDS);
  // initialize logging
  logInit();
  // get options and fill parameters from configuration file

  get_options(uniqCfg); // Command-line options specific for NRUE
  IS_SOFTMODEM_5GUE = true;
  get_common_options(uniqCfg);
  CONFIG_CLEARRTFLAG(CONFIG_NOEXITONHELP);

  softmodem_verify_mode(get_softmodem_params());

#if T_TRACER
  T_Config_Init();
#endif
  initTpool(get_softmodem_params()->threadPoolConfig, &(nrUE_params.Tpool), cpumeas(CPUMEAS_GETSTATE));
  //randominit (0);
  set_taus_seed (0);

  if (!has_cap_sys_nice())
    LOG_W(UTIL,
          "no SYS_NICE capability: cannot set thread priority and affinity, consider running with sudo for optimum performance\n");

  cpuf=get_cpu_freq_GHz();
  itti_init(TASK_MAX, tasks_info);

  init_opt();

  int ret_loader = load_nrLDPC_coding_interface(NULL, &nrLDPC_coding_interface);
  AssertFatal(ret_loader == 0, "error loading LDPC library\n");

  if (ouput_vcd) {
    vcd_signal_dumper_init("/tmp/openair_dump_nrUE.vcd");
  }
  // strdup to put the sring in the core file for post mortem identification
  char *pckg = strdup(OAI_PACKAGE_VERSION);
  LOG_I(HW, "Version: %s\n", pckg);

  PHY_vars_UE_g = calloc_or_fail(NB_UE_INST, sizeof(*PHY_vars_UE_g));
  for (int inst = 0; inst < NB_UE_INST; inst++) {
    PHY_vars_UE_g[inst] = calloc_or_fail(MAX_NUM_CCs, sizeof(*PHY_vars_UE_g[inst]));
    for (int CC_id = 0; CC_id < MAX_NUM_CCs; CC_id++) {
      PHY_vars_UE_g[inst][CC_id] = calloc_or_fail(1, sizeof(*PHY_vars_UE_g[inst][CC_id]));
      // All instances use the same coding interface
      PHY_vars_UE_g[inst][CC_id]->nrLDPC_coding_interface = nrLDPC_coding_interface;
    }
  }

  if (create_tasks_nrue(1) < 0) {
    printf("cannot create ITTI tasks\n");
    exit(-1); // need a softer mode
  }

  int mode_offset = get_softmodem_params()->nsa ? NUMBER_OF_UE_MAX : 1;
  uint16_t node_number = get_softmodem_params()->node_number;
  ue_id_g = (node_number == 0) ? 0 : node_number - 2;
  AssertFatal(ue_id_g >= 0, "UE id is expected to be nonnegative.\n");

  if (node_number == 0)
    init_pdcp(0);
  else
    init_pdcp(mode_offset + ue_id_g);
  nas_init_nrue(NB_UE_INST);

  nrUE_RU_params_t *RUs;
  int nrue_ru_count = get_nrUE_RU_params(uniqCfg, &RUs);

  nrUE_cell_params_t *cells;
  int nrue_cell_count = get_nrUE_cell_params(uniqCfg, &cells);

  init_NR_UE(NB_UE_INST, get_nrUE_params()->uecap_file, get_nrUE_params()->reconfig_file, get_nrUE_params()->rbconfig_file, cells[0].numerology);

  if (get_softmodem_params()->emulate_l1) {
    RCconfig_nr_ue_macrlc();
    get_channel_model_mode(uniqCfg);
  }

  // start time manager with some reasonable default for the running mode
  // (may be overwritten in configuration file or command line)
  void nr_pdcp_ms_tick(void);
  void nr_rlc_ms_tick(void);
  time_manager_tick_function_t tick_functions[] = {
    nr_pdcp_ms_tick,
    nr_rlc_ms_tick
  };
  int tick_functions_count = 2;
  time_manager_start(tick_functions, tick_functions_count,
                     // iq_samples time source for rfsim,
                     // realtime time source if not
                     IS_SOFTMODEM_RFSIM ? TIME_SOURCE_IQ_SAMPLES
                                        : TIME_SOURCE_REALTIME);

  if (!get_softmodem_params()->nsa && get_softmodem_params()->emulate_l1)
    start_oai_nrue_threads();

  // initialize per-cell frame parameters
  for (int cell_id = 0; cell_id < nrue_cell_count; cell_id++) {
    int rf_port = cells[cell_id].rf_port;
    AssertFatal(rf_port >= 0 && rf_port < nrue_ru_count, "Invalid rf_port (%d) for cell %d. Should be >= 0 and < %d\n", rf_port, cell_id, nrue_ru_count);
    AssertFatal(RUs[rf_port].used_by_cell == -1, "RU %d is already used by cell %d and therefore cannot also be used by cell %d\n", rf_port, RUs[rf_port].used_by_cell, cell_id);
    RUs[rf_port].used_by_cell = cell_id;

    set_fp_options(&cell_fp[cell_id], &RUs[rf_port]);
    if (IS_SA_MODE(get_softmodem_params()) || get_softmodem_params()->sl_mode)
      nr_init_frame_parms_ue_sa(&cell_fp[cell_id], &cells[cell_id]);
  }

  if (!get_softmodem_params()->emulate_l1) {
    for (int inst = 0; inst < NB_UE_INST; inst++) {
      int cell_id = inst; // initially connect each UE to its own cell
      AssertFatal(cell_id >= 0 && cell_id < nrue_cell_count,
                  "There are not enough cell definitions for all UEs! NB_UE_INST = %d, nrue_cell_count = %d\n",
                  NB_UE_INST,
                  nrue_cell_count);
      nrUE_cell_params_t *cell = &cells[cell_id];
      AssertFatal(cell->used_by_ue == -1,
                  "Cell %d is already used by UE %d and therefore cannot also be used by UE %d\n",
                  cell_id,
                  cell->used_by_ue,
                  inst);
      cell->used_by_ue = inst;

      int rf_port = cell->rf_port;
      nrUE_RU_params_t *RU = &RUs[rf_port];

      for (int CC_id = 0; CC_id < MAX_NUM_CCs; CC_id++) {
        PHY_VARS_NR_UE *UE_CC = PHY_vars_UE_g[inst][CC_id];

        set_UE_options(CC_id, UE_CC, rf_port, RU);
        UE_CC->frame_parms = cell_fp[cell_id];
        NR_UE_MAC_INST_t *mac = get_mac_inst(inst);
        init_nr_ue_phy_cpu_stats(&UE_CC->phy_cpu_stats);

        if (IS_SA_MODE(get_softmodem_params()) || get_softmodem_params()->sl_mode) {
          mac->nr_band = cell->band;
          mac->ssb_start_subcarrier = cell->ssb_start;
          mac->dl_frequency = cell->rf_frequency;
        } else {
          do {
            notifiedFIFO_elt_t *elt = pollNotifiedFIFO(&mac->input_nf);
            if (!elt) {
              break;
            }
            process_msg_rcc_to_mac(NotifiedFifoData(elt), inst);
            delNotifiedFIFO_elt(elt);
          } while (true);
          fapi_nr_config_request_t *nrUE_config = &UE_CC->nrUE_config;
          nr_init_frame_parms_ue(&UE_CC->frame_parms, nrUE_config, mac->nr_band);
        }

        UE_CC->sl_mode = get_softmodem_params()->sl_mode;
        init_actor(&UE_CC->sync_actor, "SYNC_", -1);
        if (get_nrUE_params()->num_dl_actors > 0) {
          UE_CC->dl_actors = calloc_or_fail(get_nrUE_params()->num_dl_actors, sizeof(*UE_CC->dl_actors));
          for (int i = 0; i < get_nrUE_params()->num_dl_actors; i++) {
            init_actor(&UE_CC->dl_actors[i], "DL_", -1);
          }
        }
        if (get_nrUE_params()->num_ul_actors > 0) {
          UE_CC->ul_actors = calloc_or_fail(get_nrUE_params()->num_ul_actors, sizeof(*UE_CC->ul_actors));
          for (int i = 0; i < get_nrUE_params()->num_ul_actors; i++) {
            init_actor(&UE_CC->ul_actors[i], "UL_", -1);
          }
        }
        init_nr_ue_vars(UE_CC, inst);

        if (UE_CC->sl_mode) {
          AssertFatal(UE_CC->sl_mode == 2, "Only Sidelink mode 2 supported. Mode 1 not yet supported\n");
          DevAssert(mac->if_module != NULL && mac->if_module->sl_phy_config_request != NULL);
          nr_sl_phy_config_t *phycfg = &mac->SL_MAC_PARAMS->sl_phy_config;
          phycfg->sl_config_req.sl_carrier_config.sl_num_rx_ant = get_nrUE_params()->nb_antennas_rx;
          phycfg->sl_config_req.sl_carrier_config.sl_num_tx_ant = get_nrUE_params()->nb_antennas_tx;
          mac->if_module->sl_phy_config_request(phycfg);
          sl_nr_ue_phy_params_t *sl_phy = &UE_CC->SL_UE_PHY_PARAMS;
          nr_init_frame_parms_ue_sl(&sl_phy->sl_frame_params,
                                    &sl_phy->sl_config,
                                    get_softmodem_params()->threequarter_fs,
                                    get_nrUE_params()->ofdm_offset_divisor);
          sl_ue_phy_init(UE_CC);
        }
      }
    }

    init_openair0(RUs, nrue_ru_count);

    lock_memory_to_ram();

    if (IS_SOFTMODEM_DOSCOPE) {
      load_softscope("nr", PHY_vars_UE_g[0][0]);
    }
    if (IS_SOFTMODEM_IMSCOPE_ENABLED) {
      load_softscope("im", PHY_vars_UE_g[0][0]);
    }
    AssertFatal(!(IS_SOFTMODEM_IMSCOPE_ENABLED && IS_SOFTMODEM_IMSCOPE_RECORD_ENABLED),
                "Data recoding and ImScope cannot be enabled at the same time\n");
    if (IS_SOFTMODEM_IMSCOPE_RECORD_ENABLED) {
      load_module_shlib("imscope_record", NULL, 0, PHY_vars_UE_g[0][0]);
    }

    nr_ue_ru_start();

    for (int inst = 0; inst < NB_UE_INST; inst++) {
      LOG_I(PHY,"Intializing UE Threads for instance %d ...\n", inst);
      init_NR_UE_threads(PHY_vars_UE_g[inst][0]);
    }
    printf("UE threads created by %ld\n", gettid());
  }

  // wait for end of program
  printf("TYPE <CTRL-C> TO TERMINATE\n");

  // Sleep a while before checking all parameters have been used
  // Some are used directly in external threads, asynchronously
  sleep(2);
  config_check_unknown_cmdlineopt(uniqCfg, CONFIG_CHECKALLSECTIONS);

  // wait for end of program
  printf("Entering ITTI signals handler\n");
  printf("TYPE <CTRL-C> TO TERMINATE\n");
  itti_wait_tasks_end(trigger_deregistration);
  printf("Returned from ITTI signal handler\n");
  oai_exit=1;
  printf("oai_exit=%d\n",oai_exit);

  if (ouput_vcd)
    vcd_signal_dumper_close();

  if (PHY_vars_UE_g && PHY_vars_UE_g[0]) {
    for (int CC_id = 0; CC_id < MAX_NUM_CCs; CC_id++) {
      PHY_VARS_NR_UE *phy_vars = PHY_vars_UE_g[0][CC_id];
      if (phy_vars) {
        for (int i = 0; i < get_nrUE_params()->num_ul_actors; i++) {
          shutdown_actor(&phy_vars->ul_actors[i]);
        }
        for (int i = 0; i < get_nrUE_params()->num_dl_actors; i++) {
          shutdown_actor(&phy_vars->dl_actors[i]);
        }
        int ret = pthread_join(phy_vars->main_thread, NULL);
        AssertFatal(ret == 0, "pthread_join error %d, errno %d (%s)\n", ret, errno, strerror(errno));
        if (!IS_SOFTMODEM_NOSTATS) {
          ret = pthread_join(phy_vars->stat_thread, NULL);
          AssertFatal(ret == 0, "pthread_join error %d, errno %d (%s)\n", ret, errno, strerror(errno));
        }
      }
    }
  }

  nr_ue_ru_end();

  free_nrLDPC_coding_interface(&nrLDPC_coding_interface);

  time_manager_finish();

  free(pckg);
  return 0;
}

