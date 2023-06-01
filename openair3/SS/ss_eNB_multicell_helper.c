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

#include <pthread.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <stdlib.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <sys/ioctl.h>
#include <net/if.h>

#include <netinet/in.h>
#include <netinet/sctp.h>

#include <arpa/inet.h>

#include "assertions.h"
#include "common/utils/system.h"
#include "queue.h"
#include "sctp_common.h"

#include "intertask_interface.h"
#include "common/ran_context.h"
#include "common/utils/LOG/ss-log.h"
#include "ss_eNB_context.h"
#include "ss_eNB_multicell_helper.h"
extern RAN_CONTEXT_t RC;


int get_cell_index(uint16_t cell_id, SS_Cell_Context_t SSCell_list[]){
  for(int Cell_idx = 0; Cell_idx < 8; Cell_idx++){
    if((SSCell_list[Cell_idx].cell_configured_flag == true)&&(SSCell_list[Cell_idx].eutra_cellId == cell_id)){
      return Cell_idx;
    }
  }
  for(int Cell_idx = 0; Cell_idx < 8; Cell_idx++){
    if(SSCell_list[Cell_idx].cell_configured_flag == false){
      SSCell_list[Cell_idx].cell_configured_flag = true;
      SSCell_list[Cell_idx].eutra_cellId = cell_id;
      printf("CC-MGMT cc_idx %d\n",Cell_idx);
      return Cell_idx;
    }
  }
  return -1;
}

int get_cell_index_pci(uint16_t physCellId, SS_Cell_Context_t SSCell_list[]){
  for(int Cell_idx = 0; Cell_idx < 8; Cell_idx++){
    if(SSCell_list[Cell_idx].PhysicalCellId == physCellId){
      return Cell_idx;
    }
  }
  return -1;
}
void init_ss_context(SS_Cell_Context_t SSCell_list[]){
  memset(SSCell_list, 0, (sizeof(SS_Cell_Context_t) * 8));
  memset(RC.ss.l1macind,0,sizeof(RC.ss.l1macind));
  memset(RC.ss.CC_update_flag,0, (sizeof(RC.ss.CC_update_flag)));
  memset(RC.ss.CC_conf_flag,0, (sizeof(RC.ss.CC_conf_flag)));
  for(int Cell_idx = 0; Cell_idx < 8; Cell_idx++){
    SSCell_list[Cell_idx].eutra_cellId = -1;
  }
}
void init_cell_context(int cell_index, int enb_id, MessageDef *msg_p )
{
  int init_cell_id = 0;
  if (!cell_index)
  return;
  //Need to add the initialization of all parameters from RCconfig_RRC() function (enb_config.c )
  //RRC_CONFIGURATION_REQ(msg_p) = RC.rrc[enb_id]->configuration;
LOG_A(ENB_APP, "[SYS] CC-MGMT init_cell_context cell_index %d \n",cell_index);

    // Cell params, MIB/SIB1 in DU
    RRC_CONFIGURATION_REQ(msg_p).tdd_config[cell_index] = RRC_CONFIGURATION_REQ(msg_p).tdd_config[init_cell_id];
    RRC_CONFIGURATION_REQ(msg_p).tdd_config_s[cell_index] = RRC_CONFIGURATION_REQ(msg_p).tdd_config_s[init_cell_id];

    RRC_CONFIGURATION_REQ(msg_p).prefix_type[cell_index] = RRC_CONFIGURATION_REQ(msg_p).prefix_type[init_cell_id];
    RRC_CONFIGURATION_REQ(msg_p).pbch_repetition[cell_index] = RRC_CONFIGURATION_REQ(msg_p).pbch_repetition[init_cell_id];

  RRC_CONFIGURATION_REQ(msg_p).eutra_band[cell_index] = RRC_CONFIGURATION_REQ(msg_p).eutra_band[init_cell_id] ;
  RRC_CONFIGURATION_REQ(msg_p).downlink_frequency[cell_index] = RRC_CONFIGURATION_REQ(msg_p).downlink_frequency[init_cell_id] ;
  RRC_CONFIGURATION_REQ(msg_p).uplink_frequency_offset[cell_index] = RRC_CONFIGURATION_REQ(msg_p).uplink_frequency_offset[init_cell_id] ;
  RRC_CONFIGURATION_REQ(msg_p).Nid_cell[cell_index] = RRC_CONFIGURATION_REQ(msg_p).Nid_cell[init_cell_id] ;



  RRC_CONFIGURATION_REQ(msg_p).N_RB_DL[cell_index] =RRC_CONFIGURATION_REQ(msg_p).N_RB_DL[init_cell_id];



  RRC_CONFIGURATION_REQ(msg_p).frame_type[cell_index] = RRC_CONFIGURATION_REQ(msg_p).frame_type[init_cell_id];
  //if (config_check_band_frequencies(cell_index,
    //                                RRC_CONFIGURATION_REQ(msg_p).eutra_band[cell_index],
      //                              RRC_CONFIGURATION_REQ(msg_p).downlink_frequency[cell_index],
        //                            RRC_CONFIGURATION_REQ(msg_p).uplink_frequency_offset[cell_index],
          //                          RRC_CONFIGURATION_REQ(msg_p).frame_type[cell_index]))
  //{
    //AssertFatal(0, "error calling enb_check_band_frequencies\n");
 // }

  RRC_CONFIGURATION_REQ(msg_p).nb_antenna_ports[cell_index] = RRC_CONFIGURATION_REQ(msg_p).nb_antenna_ports[init_cell_id];

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].prach_root = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].prach_root;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].prach_config_index = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].prach_config_index;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].prach_high_speed = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].prach_high_speed;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].prach_zero_correlation = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].prach_zero_correlation;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].prach_freq_offset = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].prach_freq_offset;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].pucch_delta_shift = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].pucch_delta_shift;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].pucch_nRB_CQI = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].pucch_nRB_CQI;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].pucch_nCS_AN = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].pucch_nCS_AN;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].pucch_n1_AN = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].pucch_n1_AN;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].pdsch_referenceSignalPower = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].pdsch_referenceSignalPower;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].pdsch_p_b = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].pdsch_p_b;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].pusch_n_SB = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].pusch_n_SB;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].pusch_hoppingMode = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].pusch_hoppingMode;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].pusch_hoppingOffset = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].pusch_hoppingOffset;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].pusch_enable64QAM = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].pusch_enable64QAM;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].pusch_groupHoppingEnabled = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].pusch_groupHoppingEnabled;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].pusch_groupAssignment = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].pusch_groupAssignment;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].pusch_sequenceHoppingEnabled = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].pusch_sequenceHoppingEnabled;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].pusch_nDMRS1 = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].pusch_nDMRS1;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].phich_duration = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].phich_duration;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].phich_resource = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].phich_resource;
  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].phich_resource = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].phich_resource ;
  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].srs_enable =   RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].srs_enable;

  if (RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].srs_enable == true)
  {
    RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].srs_BandwidthConfig = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].srs_BandwidthConfig ;

    RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].srs_SubframeConfig = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].srs_SubframeConfig;
    RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].srs_ackNackST = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].srs_ackNackST;
  }

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].pusch_p0_Nominal = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].pusch_p0_Nominal;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].pusch_alpha = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].pusch_alpha;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].pucch_p0_Nominal = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].pucch_p0_Nominal ;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].msg3_delta_Preamble = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].msg3_delta_Preamble;
  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].pucch_deltaF_Format1 = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].pucch_deltaF_Format1;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].pucch_deltaF_Format1b = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].pucch_deltaF_Format1b;
  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].pucch_deltaF_Format2 = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].pucch_deltaF_Format2 ;
  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].pucch_deltaF_Format2a = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].pucch_deltaF_Format2a;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].pucch_deltaF_Format2b = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].pucch_deltaF_Format2b;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].rach_numberOfRA_Preambles = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].rach_numberOfRA_Preambles;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].rach_preamblesGroupAConfig = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].rach_preamblesGroupAConfig;
  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].rach_sizeOfRA_PreamblesGroupA = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].rach_sizeOfRA_PreamblesGroupA;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].rach_messageSizeGroupA =  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].rach_messageSizeGroupA ;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].rach_messagePowerOffsetGroupB = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].rach_messagePowerOffsetGroupB;
  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].rach_preamblesGroupAConfig =   RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].rach_preamblesGroupAConfig;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].rach_preambleInitialReceivedTargetPower = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].rach_preambleInitialReceivedTargetPower;
  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].rach_powerRampingStep =  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].rach_powerRampingStep;
  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].rach_preambleTransMax =  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].rach_preambleTransMax;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].rach_raResponseWindowSize =  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].rach_raResponseWindowSize;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].rach_macContentionResolutionTimer = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].rach_macContentionResolutionTimer;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].rach_maxHARQ_Msg3Tx = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].rach_maxHARQ_Msg3Tx;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].pcch_defaultPagingCycle = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].pcch_defaultPagingCycle ;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].pcch_nB =  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].pcch_nB;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].drx_Config_present = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].drx_Config_present ;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].drx_onDurationTimer = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].drx_onDurationTimer;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].drx_InactivityTimer = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].drx_InactivityTimer ;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].ue_multiple_max = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].ue_multiple_max;
  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].mbms_dedicated_serving_cell = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].mbms_dedicated_serving_cell;
  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].drx_RetransmissionTimer = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].drx_RetransmissionTimer;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].drx_longDrx_CycleStartOffset_present = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].drx_longDrx_CycleStartOffset_present;
  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].drx_longDrx_CycleStartOffset = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].drx_longDrx_CycleStartOffset ;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].drx_shortDrx_Cycle = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].drx_shortDrx_Cycle;
  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].drx_shortDrx_ShortCycleTimer = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].drx_shortDrx_ShortCycleTimer;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].bcch_modificationPeriodCoeff = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].bcch_modificationPeriodCoeff;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].ue_TimersAndConstants_t300 = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].ue_TimersAndConstants_t300 ;
  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].ue_TimersAndConstants_t301 =  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].ue_TimersAndConstants_t301;
  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].ue_TimersAndConstants_t310 =  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].ue_TimersAndConstants_t310;
  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].ue_TimersAndConstants_t311 =  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].ue_TimersAndConstants_t311;
  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].ue_TimersAndConstants_n310 =  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].ue_TimersAndConstants_n310;
  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].ue_TimersAndConstants_n311 =  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].ue_TimersAndConstants_n311;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].ue_TransmissionMode = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].ue_TransmissionMode;

  RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].ue_multiple_max = RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[init_cell_id].ue_multiple_max;
}

