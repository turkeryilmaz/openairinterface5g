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
#include <ctype.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#define TELNETSERVERCODE
#include "telnetsrv.h"

#include "openair2/RRC/NR/nr_rrc_defs.h"
#include "openair2/LAYER2/NR_MAC_gNB/nr_mac_gNB.h"
#include "common/utils/nr/nr_common.h"

#define ERROR_MSG_RET(mSG, aRGS...) do { prnt("ERROR: " mSG, ##aRGS); return 1; } while (0)

#define ISINITBWP "bwp3gpp:isInitialBwp"
//#define CYCLPREF  "bwp3gpp:cyclicPrefix"
#define NUMRBS    "bwp3gpp:numberOfRBs"
#define STARTRB   "bwp3gpp:startRB"
#define BWPSCS    "bwp3gpp:subCarrierSpacing"

#define SSBFREQ "nrcelldu3gpp:ssbFrequency"
#define ARFCNDL "nrcelldu3gpp:arfcnDL"
#define BWDL    "nrcelldu3gpp:bSChannelBwDL"
#define ARFCNUL "nrcelldu3gpp:arfcnUL"
#define BWUL    "nrcelldu3gpp:bSChannelBwUL"
#define PCI     "nrcelldu3gpp:nRPCI"
#define TAC     "nrcelldu3gpp:nRTAC"
//#define MCC     "nrcelldu3gpp:mcc"
//#define MNC     "nrcelldu3gpp:mnc"

static int get_stats(char *buf, int debug, telnet_printfunc_t prnt)
{
  if (buf)
    ERROR_MSG_RET("no parameter allowed\n");

  const gNB_MAC_INST *mac = RC.nrmac[0];
  AssertFatal(mac != NULL, "need MAC\n");

  const gNB_RRC_INST *rrc = RC.nrrrc[0];
  const gNB_RrcConfigurationReq *conf = &rrc->configuration;
  AssertFatal(rrc != NULL, "need RRC\n");

  const NR_ServingCellConfigCommon_t *scc = mac->common_channels[0].ServingCellConfigCommon;
  const NR_FrequencyInfoDL_t *frequencyInfoDL = scc->downlinkConfigCommon->frequencyInfoDL;
  const NR_FrequencyInfoUL_t *frequencyInfoUL = scc->uplinkConfigCommon->frequencyInfoUL;
  frame_type_t frame_type = get_frame_type(*frequencyInfoDL->frequencyBandList.list.array[0], *scc->ssbSubcarrierSpacing);
  const NR_BWP_t *initialDL = &scc->downlinkConfigCommon->initialDownlinkBWP->genericParameters;

  int num_ues = 0;
  UE_iterator((NR_UE_info_t **)mac->UE_info.list, it) {
    num_ues++;
  }

  prnt("=== begin O1 stats ===\n");
  prnt("{\n");

    prnt("  \"BWP\": {\n");
    prnt("    \"" ISINITBWP "\": true,\n");
    //prnt("    \"" CYCLPREF "\": %ld,\n", *initialDL->cyclicPrefix);
    prnt("    \"" NUMRBS "\": %ld,\n", NRRIV2BW(initialDL->locationAndBandwidth, MAX_BWP_SIZE));
    prnt("    \"" STARTRB "\": %ld,\n", NRRIV2PRBOFFSET(initialDL->locationAndBandwidth, MAX_BWP_SIZE));
    prnt("    \"" BWPSCS "\": %ld\n", initialDL->subcarrierSpacing);
    prnt("  },\n");

    prnt("  \"NRCELLDU\": {\n");
    prnt("    \"" SSBFREQ "\": %ld,\n", *scc->ssbSubcarrierSpacing);
    prnt("    \"" ARFCNDL "\": %ld,\n", frequencyInfoDL->absoluteFrequencyPointA);
    prnt("    \"" BWDL "\": %ld,\n", frequencyInfoDL->scs_SpecificCarrierList.list.array[0]->carrierBandwidth);
    prnt("    \"" ARFCNUL "\": %ld,\n", frequencyInfoUL->absoluteFrequencyPointA ? *frequencyInfoUL->absoluteFrequencyPointA : frequencyInfoDL->absoluteFrequencyPointA);
    prnt("    \"" BWUL "\": %ld,\n", frequencyInfoUL->scs_SpecificCarrierList.list.array[0]->carrierBandwidth);
    prnt("    \"" PCI "\": %ld,\n", *scc->physCellId);
    prnt("    \"" TAC "\": %ld\n", conf->tac);
    prnt("  }\n");
  prnt("}\n");
  prnt("===   end O1 stats ===\n");
  prnt("=== begin add stats ===\n");
  prnt("frame type %s\n", frame_type == TDD ? "tdd" : "fdd");
  prnt("band number %ld\n", *frequencyInfoDL->frequencyBandList.list.array[0]);
  prnt("no UEs %d\n", num_ues);
  prnt("===   end add stats ===\n");
  return 0;
}

static int read_long(const char *buf, const char *end, const char *id, long *val)
{
  const char *curr = buf;
  while (isspace(*curr) && curr < end) // skip leading spaces
    curr++;
  int len = strlen(id);
  if (curr + len >= end)
    return -1;
  if (strncmp(curr, id, len) != 0) // check buf has id
    return -1;
  curr += len;
  while (isspace(*curr) && curr < end) // skip middle spaces
    curr++;
  if (curr >= end)
    return -1;
  int nread = sscanf(curr, "%ld", val);
  if (nread != 1)
    return -1;
  while (isdigit(*curr) && curr < end) // skip all digits read above
    curr++;
  if (curr > end)
    return -1;
  return curr - buf;
}

static int set_config(char *buf, int debug, telnet_printfunc_t prnt)
{
  if (!buf)
    ERROR_MSG_RET("need param: o1 config param1 val1 [param2 val2 ...]\n");
  const char *end = buf + strlen(buf);

  int processed = 0;
  int pos = 0;
  long arfcnDL;
  processed = read_long(buf + pos, end, ARFCNDL, &arfcnDL);
  if (processed < 0)
    ERROR_MSG_RET("could not read " ARFCNDL " at index %d\n", pos + processed);
  pos += processed;
  prnt("setting " ARFCNDL " %ld len %d\n", arfcnDL, pos);

  long arfcnUL;
  processed = read_long(buf + pos, end, ARFCNUL, &arfcnUL);
  if (processed < 0)
    ERROR_MSG_RET("could not read " ARFCNUL " at index %d\n", pos + processed);
  pos += processed;
  prnt("setting " ARFCNUL " %ld len %d\n", arfcnUL, pos);

  return 0;
}

static telnetshell_cmddef_t o1cmds[] = {
  {"stats", "", get_stats},
  {"config", "?", set_config},
  {"", "", NULL},
};

static telnetshell_vardef_t o1vars[] = {
  {"", 0, 0, NULL}
};

void add_o1_cmds(void) {
  add_telnetcmd("o1", o1vars, o1cmds);
}
