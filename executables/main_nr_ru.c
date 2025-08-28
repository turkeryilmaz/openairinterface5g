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

/*! \file oairu.c
 * \brief Top-level threads for radio-unit
 * \author R. Knopp
 * \date 2020
 * \version 0.1
 * \company Eurecom
 * \email: knopp@eurecom.fr
 * \note
 * \warning
 */


#define _GNU_SOURCE             /* See feature_test_macros(7) */
#include <sched.h>
#include "assertions.h"
#include "PHY/types.h"
#include "PHY/defs_RU.h"
#include "common/oai_version.h"
#include "common/config/config_userapi.h"
#include "common/utils/load_module_shlib.h"
#include "common/ran_context.h"
#include "radio/COMMON/common_lib.h"
#include "radio/ETHERNET/if_defs.h"
#include "PHY/phy_vars.h"
#include "PHY/phy_extern.h"
#include "PHY/TOOLS/phy_scope_interface.h"
#include "common/utils/LOG/log.h"
#include "common/utils/LOG/vcd_signal_dumper.h"
//#include "PHY/INIT/phy_init.h"
#include "openair2/ENB_APP/enb_paramdef.h"
#include "system.h"
#include "nfapi/oai_integration/vendor_ext.h"
#include <executables/softmodem-common.h>
#include <executables/thread-common.h>
#include "openair1/PHY/INIT/nr_phy_init.h"

pthread_cond_t sync_cond;
pthread_mutex_t sync_mutex;
int sync_var=-1; //!< protected by mutex \ref sync_mutex.
int config_sync_var=-1;

int oai_exit = 0;
int sf_ahead = 4;
int emulate_rf = 0;

RAN_CONTEXT_t RC;

extern void kill_NR_RU_proc(int inst);
extern void set_function_spec_param(RU_t *ru);
extern void start_NR_RU(RU_t *ru);
extern void init_NR_RU(configmodule_interface_t *cfg, char *);

int32_t uplink_frequency_offset[MAX_NUM_CCs][4];


void nfapi_setmode(nfapi_mode_t nfapi_mode) { return; }
void exit_function(const char *file, const char *function, const int line, const char *s, const int assert) {

  if (s != NULL) {
    printf("%s:%d %s() Exiting OAI softmodem: %s\n",file,line, function, s);
  }
  close_log_mem();
  oai_exit = 1;
  RU_t *ru = RC.ru[0];

  if (ru->rfdevice.trx_end_func) {
    ru->rfdevice.trx_end_func(&ru->rfdevice);
    ru->rfdevice.trx_end_func = NULL;
  }

  if (ru->ifdevice.trx_end_func) {
    ru->ifdevice.trx_end_func(&ru->ifdevice);
    ru->ifdevice.trx_end_func = NULL;
  }

  pthread_mutex_destroy(ru->ru_mutex);
  pthread_cond_destroy(ru->ru_cond);
  if (assert) {
    abort();
  } else {
    sleep(1); // allow lte-softmodem threads to exit first
    exit(EXIT_SUCCESS);
  }
}

static void get_options(configmodule_interface_t *cfg)
{
  CONFIG_SETRTFLAG(CONFIG_NOEXITONHELP);
  get_common_options(cfg);
  CONFIG_CLEARRTFLAG(CONFIG_NOEXITONHELP);

//  NRCConfig();
}


nfapi_mode_t nfapi_getmode(void) {
  return(NFAPI_MODE_PNF);
}

void oai_nfapi_rach_ind(nfapi_rach_indication_t *rach_ind) {

  AssertFatal(1==0,"This is bad ... please check why we get here\n");
}

void wait_eNBs(void){ return; }
void wait_gNBs(void){ return; }

struct timespec timespec_add(struct timespec,struct timespec) {struct timespec t={0}; return t;};
struct timespec timespec_sub(struct timespec,struct timespec) {struct timespec t={0}; return t;};

void perform_symbol_rotation(NR_DL_FRAME_PARMS *fp, double f0, c16_t *symbol_rotation) {return;}
void init_timeshift_rotation(NR_DL_FRAME_PARMS *fp) {return;};
int beam_index_allocation(int fapi_beam_index, NR_gNB_COMMON *common_vars, int slot, int symbols_per_slot, int bitmap_symbols) {int i=0; return i;};
void nr_fill_du(uint16_t N_ZC, const uint16_t *prach_root_sequence_map, uint16_t nr_du[NR_PRACH_SEQ_LEN_L - 1]) {return;};
uint16_t nr_du[838];

uint64_t                 downlink_frequency[MAX_NUM_CCs][4];

configmodule_interface_t *uniqCfg = NULL;
THREAD_STRUCT thread_struct;

void *oru_north_read_thread(void *arg);
void NRRCconfig_RU(configmodule_interface_t *cfg);

int main ( int argc, char **argv )
{
  memset(&RC,0,sizeof(RC));
  if ((uniqCfg = load_configmodule(argc, argv, 0)) == NULL) {
    exit_fun("[SOFTMODEM] Error, configuration module init failed\n");
  }

  logInit();
  printf("Reading in command-line options\n");
  get_options(uniqCfg);

  if (CONFIG_ISFLAGSET(CONFIG_ABORT) ) {
    fprintf(stderr,"Getting configuration failed\n");
    exit(-1);
  }

#if T_TRACER
  T_Config_Init();
#endif
  printf("configuring for RRU\n");
  // strdup to put the sring in the core file for post mortem identification
  LOG_I(HW, "Version: %s\n", strdup(OAI_PACKAGE_VERSION));


  /* Read configuration */

  printf("About to Init RU threads\n");

  lock_memory_to_ram();

  RC.nb_RU = 1;
  NRRCconfig_RU(config_get_if());
  RU_t *ru = RC.ru[0];
  nr_ru_init_frame_parms(ru);
  load_dftslib();

  ORU_t oru;
  oru.ru = ru;
  pthread_create(&oru.thread, NULL, oru_north_read_thread, (void *)&oru);

  while (oai_exit==0) sleep(1);
  // stop threads

  kill_NR_RU_proc(0);

  end_configmodule(uniqCfg);

  if (ru->rfdevice.trx_end_func) {
    ru->rfdevice.trx_end_func(&ru->rfdevice);
    ru->rfdevice.trx_end_func = NULL;
  }

  if (ru->ifdevice.trx_end_func) {
    ru->ifdevice.trx_end_func(&ru->ifdevice);
    ru->ifdevice.trx_end_func = NULL;
  }

  logClean();
  printf("Bye.\n");
  return 0;
}
