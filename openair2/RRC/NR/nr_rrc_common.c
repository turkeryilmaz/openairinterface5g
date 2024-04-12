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

/*! \file nr_rrc_common.c
 * \brief rrc common procedures for gNB
 * \author Navid Nikaein and Raymond Knopp, WEI-TAI CHEN
 * \date 2011 - 2014, 2018
 * \version 1.0
 * \company Eurecom, NTUST
 * \email:  navid.nikaein@eurecom.fr and raymond.knopp@eurecom.fr, kroempa@gmail.com
 */

#include "nr_rrc_extern.h"
#include "LAYER2/NR_MAC_COMMON/nr_mac_extern.h"
#include "COMMON/openair_defs.h"
#include "common/platform_types.h"
#include "RRC/L2_INTERFACE/openair_rrc_L2_interface.h"
#include "LAYER2/RLC/rlc.h"
#include "COMMON/mac_rrc_primitives.h"
#include "common/utils/LOG/log.h"
#include "asn1_msg.h"
#include "common/utils/LOG/vcd_signal_dumper.h"
#include "common/ran_context.h"

#define DEBUG_NR_RRC 1

extern RAN_CONTEXT_t RC;
extern UE_MAC_INST *UE_mac_inst;
extern mui_t rrc_gNB_mui;

//-----------------------------------------------------------------------------
void rrc_init_nr_srb_param(NR_LCHAN_DESC *chan)
{
  chan->transport_block_size = 4;
  chan->max_transport_blocks = 16;
  chan->Delay_class = 1;
  return;
}


//-----------------------------------------------------------------------------

void nrrrc_init_global_cc_context(int CC_id, module_id_t module_id)
{
  for(int i=0;i<MAX_RBS;i++)
  {
    {
      //RC.NR_RB_Config[CC_id][i].isRBConfigValid = false;
      //memset(&RC.NR_RB_Config[CC_id][i],0,sizeof(NRRBConfig));
    }

    /*SRB1 Default Config for RLC and MAC*/
    /*RC.NR_RB_Config[CC_id][1].RlcBearer->rlc_Config->present = 1;
    RC.NR_RB_Config[CC_id][1].RlcBearer->rlc_Config->choice.am.ul_AM_RLC.sn_FieldLength;
    RC.NR_RB_Config[CC_id][1].RlcBearer->rlc_Config->choice.am.ul_AM_RLC.t_PollRetransmit = 15;
    RC.NR_RB_Config[CC_id][1].RlcBearer->rlc_Config->choice.am.ul_AM_RLC.pollPDU = 0;
    RC.NR_RB_Config[CC_id][1].RlcBearer->rlc_Config->choice.am.ul_AM_RLC.pollByte = 14;
    RC.NR_RB_Config[CC_id][1].RlcBearer->rlc_Config->choice.am.ul_AM_RLC.maxRetxThreshold = 3;
    RC.NR_RB_Config[CC_id][1].RlcBearer->rlc_Config->choice.am.dl_AM_RLC.sn_FieldLength;
    RC.NR_RB_Config[CC_id][1].RlcBearer->rlc_Config->choice.am.dl_AM_RLC.t_Reassembly;
    RC.NR_RB_Config[CC_id][1].RlcBearer->rlc_Config->choice.am.dl_AM_RLC.t_StatusProhibit = 0;
    RC.NR_RB_Config[CC_id][1].RlcBearer->mac_LogicalChannelConfig->ul_SpecificParameters = CALLOC(1, sizeof(struct NR_LogicalChannelConfig__ul_SpecificParameters));
    RC.NR_RB_Config[CC_id][1].RlcBearer->mac_LogicalChannelConfig->ul_SpecificParameters->priority = 1;
    RC.NR_RB_Config[CC_id][1].RlcBearer->mac_LogicalChannelConfig->ul_SpecificParameters->prioritisedBitRate = 7;*/

    /*SRB2 Default Config for RLC and MAC*/
    /*RC.NR_RB_Config[CC_id][2].RlcBearer->rlc_Config->present = NR_RLC_Config_PR_am;
    RC.NR_RB_Config[CC_id][2].RlcBearer->rlc_Config->choice.am.ul_AM_RLC.sn_FieldLength;
    RC.NR_RB_Config[CC_id][2].RlcBearer->rlc_Config->choice.am.ul_AM_RLC.t_PollRetransmit = NR_T_PollRetransmit_ms15;
    RC.NR_RB_Config[CC_id][2].RlcBearer->rlc_Config->choice.am.ul_AM_RLC.pollPDU = NR_PollPDU_p8;
    RC.NR_RB_Config[CC_id][2].RlcBearer->rlc_Config->choice.am.ul_AM_RLC.pollByte = 14;
    RC.NR_RB_Config[CC_id][2].RlcBearer->rlc_Config->choice.am.ul_AM_RLC.maxRetxThreshold = 3;
    RC.NR_RB_Config[CC_id][2].RlcBearer->rlc_Config->choice.am.dl_AM_RLC.sn_FieldLength;
    RC.NR_RB_Config[CC_id][2].RlcBearer->rlc_Config->choice.am.dl_AM_RLC.t_Reassembly;
    RC.NR_RB_Config[CC_id][2].RlcBearer->rlc_Config->choice.am.dl_AM_RLC.t_StatusProhibit = 0;
    RC.NR_RB_Config[CC_id][2].RlcBearer->mac_LogicalChannelConfig->ul_SpecificParameters = CALLOC(1, sizeof(struct NR_LogicalChannelConfig__ul_SpecificParameters));
    RC.NR_RB_Config[CC_id][2].RlcBearer->mac_LogicalChannelConfig->ul_SpecificParameters->priority = 1;
    RC.NR_RB_Config[CC_id][2].RlcBearer->mac_LogicalChannelConfig->ul_SpecificParameters->prioritisedBitRate = 7;*/

    /*DRB Default Config for PDCP,RLC(AM) and MAC*/
    /*RC.NR_RB_Config[CC_id][3].RlcCfg.present = LTE_RLC_Config_PR_am;
    RC.NR_RB_Config[CC_id][3].RlcCfg.choice.am.ul_AM_RLC.t_PollRetransmit = LTE_T_PollRetransmit_ms50;
    RC.NR_RB_Config[CC_id][3].RlcCfg.choice.am.ul_AM_RLC.pollPDU = LTE_PollPDU_p16;
    RC.NR_RB_Config[CC_id][3].RlcCfg.choice.am.ul_AM_RLC.pollByte = LTE_PollByte_kBinfinity;
    RC.NR_RB_Config[CC_id][3].RlcCfg.choice.am.ul_AM_RLC.maxRetxThreshold = LTE_UL_AM_RLC__maxRetxThreshold_t8;
    RC.NR_RB_Config[CC_id][3].RlcCfg.choice.am.dl_AM_RLC.t_Reordering = LTE_T_Reordering_ms35;
    RC.NR_RB_Config[CC_id][3].RlcCfg.choice.am.dl_AM_RLC.t_StatusProhibit = LTE_T_StatusProhibit_ms25;
    RC.NR_RB_Config[CC_id][3].PdcpCfg.rlc_AM = CALLOC(1, sizeof(struct LTE_PDCP_Config__rlc_AM));
    RC.NR_RB_Config[CC_id][3].PdcpCfg.rlc_AM->statusReportRequired = false;
    RC.NR_RB_Config[CC_id][3].PdcpCfg.headerCompression.present = LTE_PDCP_Config__headerCompression_PR_notUsed;
    RC.NR_RB_Config[CC_id][3].Mac.ul_SpecificParameters = CALLOC(1, sizeof(struct LTE_LogicalChannelConfig__ul_SpecificParameters));
    RC.NR_RB_Config[CC_id][3].Mac.ul_SpecificParameters->priority = 12;
    RC.NR_RB_Config[CC_id][3].Mac.ul_SpecificParameters->prioritisedBitRate = LTE_LogicalChannelConfig__ul_SpecificParameters__prioritisedBitRate_kBps8;
    RC.NR_RB_Config[CC_id][3].PdcpCfg.discardTimer = CALLOC(1, sizeof(long));
    *(RC.NR_RB_Config[CC_id][3].PdcpCfg.discardTimer) = LTE_PDCP_Config__discardTimer_infinity;*/

    /*DRB Default Config for PDCP,RLC(UM) and MAC*/
    /*RC.NR_RB_Config[CC_id][4].RlcCfg.present = LTE_RLC_Config_PR_um_Bi_Directional;
    RC.NR_RB_Config[CC_id][4].RlcCfg.choice.um_Bi_Directional.ul_UM_RLC.sn_FieldLength = LTE_SN_FieldLength_size10;
    RC.NR_RB_Config[CC_id][4].RlcCfg.choice.um_Bi_Directional.dl_UM_RLC.sn_FieldLength = LTE_SN_FieldLength_size10;
    RC.NR_RB_Config[CC_id][4].RlcCfg.choice.um_Bi_Directional.dl_UM_RLC.t_Reordering = LTE_T_Reordering_ms35;
    RC.NR_RB_Config[CC_id][4].PdcpCfg.rlc_UM = CALLOC(1, sizeof(struct LTE_PDCP_Config__rlc_UM));
    RC.NR_RB_Config[CC_id][4].PdcpCfg.rlc_UM->pdcp_SN_Size = LTE_PDCP_Config__rlc_UM__pdcp_SN_Size_len12bits;
    RC.NR_RB_Config[CC_id][4].PdcpCfg.headerCompression.present = LTE_PDCP_Config__headerCompression_PR_notUsed;
    RC.NR_RB_Config[CC_id][4].Mac.ul_SpecificParameters = CALLOC(1, sizeof(struct LTE_LogicalChannelConfig__ul_SpecificParameters));
    RC.NR_RB_Config[CC_id][4].Mac.ul_SpecificParameters->priority = 12;
    RC.NR_RB_Config[CC_id][4].Mac.ul_SpecificParameters->prioritisedBitRate = LTE_LogicalChannelConfig__ul_SpecificParameters__prioritisedBitRate_kBps8;
    RC.NR_RB_Config[CC_id][4].PdcpCfg.discardTimer = CALLOC(1, sizeof(long));
    *(RC.NR_RB_Config[CC_id][4].PdcpCfg.discardTimer) = LTE_PDCP_Config__discardTimer_infinity;*/

  }
}
