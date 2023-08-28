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

#include <sys/types.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#define TELNETSERVERCODE
#include "telnetsrv.h"

#include "openair2/LAYER2/NR_MAC_gNB/nr_mac_gNB.h"

#define ERROR_MSG_RET(mSG, aRGS...) do { prnt(mSG, ##aRGS); return 1; } while (0)

int get_stats(char *buf, int debug, telnet_printfunc_t prnt)
{
  if (buf)
    ERROR_MSG_RET("no parameter allowed\n");

  const gNB_MAC_INST *mac = RC.nrmac[0];
  const NR_ServingCellConfigCommon_t *scc = mac->common_channels[0].ServingCellConfigCommon;
  const NR_FrequencyInfoDL_t *frequencyInfoDL = scc->downlinkConfigCommon->frequencyInfoDL;
  prnt("=== begin O1 stats ===\n");
  prnt("ssb scs %ld\n", *scc->ssbSubcarrierSpacing);
  prnt("band %ld\n", *frequencyInfoDL->frequencyBandList.list.array[0]);
  prnt("prb dl %ld\n", frequencyInfoDL->scs_SpecificCarrierList.list.array[0]->carrierBandwidth);
  prnt("===   end O1 stats ===\n");
  return 0;
}

static telnetshell_cmddef_t o1cmds[] = {
  {"stats", "", get_stats},
  {"", "", NULL},
};

static telnetshell_vardef_t o1vars[] = {

  {"", 0, 0, NULL}
};

void add_o1_cmds(void) {
  add_telnetcmd("o1", o1vars, o1cmds);
}
