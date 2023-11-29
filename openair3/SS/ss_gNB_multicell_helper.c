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

//#include "ss_gNB_context.h"
//#include "ss_gNB_multicell_helper.h"
//#include "ss_messages_types.h"

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
#include "ss_gNB_context.h"
#include "ss_gNB_multicell_helper.h"

int get_gNB_cell_index(uint16_t cell_id, SS_Cell_Context_t SSCell_list[]){
  for(int Cell_idx = 0; Cell_idx < 8; Cell_idx++){
    if((SSCell_list[Cell_idx].cell_configured_flag == true)&&(SSCell_list[Cell_idx].nr_cellId == cell_id)){
      return Cell_idx;
    }
  }
  for(int Cell_idx = 0; Cell_idx < 8; Cell_idx++){
    if(SSCell_list[Cell_idx].cell_configured_flag == false){
      SSCell_list[Cell_idx].cell_configured_flag = true;
      SSCell_list[Cell_idx].nr_cellId = cell_id;
      printf("CC-MGMT cc_idx %d\n",Cell_idx);
      return Cell_idx;
    }
  }
  return -1;
}

int get_gNB_cell_index_pci(uint16_t physCellId, SS_Cell_Context_t SSCell_list[]){
  for(int Cell_idx = 0; Cell_idx < 8; Cell_idx++){
    if(SSCell_list[Cell_idx].PhysicalCellId == physCellId){
      return Cell_idx;
    }
  }
  return -1;
}

void init_ss_gNB_context(SS_Cell_Context_t SSCell_list[]){
  memset(SSCell_list, 0, (sizeof(SS_Cell_Context_t) * 8));
  for(int Cell_idx = 0; Cell_idx < 8; Cell_idx++){
    SSCell_list[Cell_idx].nr_cellId = -1;
  }
}

void init_gnb_cell_context(int cell_index)
{
  int init_cell_id = 0;
  if(!cell_index)
  {
  	 return;
  }

  LOG_I(GNB_APP,"init_gnb_cell_context incoming cell_index %d, initialized cell_index %x\n",
        cell_index,init_cell_id);
  /* Prepare configuration */
 // memcpy(&RC.nrrrc[0]->carrier[cell_index],&RC.nrrrc[0]->carrier[init_cell_id],sizeof(rrc_gNB_carrier_data_t));
 /*bugz128620 rebase: carrier pdcch_configsib1 seems not being used, but scc->*/
#if 0 
  memcpy(RC.nrrrc[0]->carrier[cell_index].pdcch_ConfigSIB1,RC.nrrrc[0]->carrier[init_cell_id].pdcch_ConfigSIB1,sizeof(NR_PDCCH_ConfigSIB1_t));
  LOG_I(GNB_APP,"controlResourceSetZero: %d controlResourceSetZero1: %d searchSpaceZero: %d searchSpaceZero1: %d\n",RC.nrrrc[0]->carrier[cell_index].pdcch_ConfigSIB1->controlResourceSetZero,RC.nrrrc[0]->carrier[init_cell_id].pdcch_ConfigSIB1->controlResourceSetZero,RC.nrrrc[0]->carrier[cell_index].pdcch_ConfigSIB1->searchSpaceZero,RC.nrrrc[0]->carrier[init_cell_id].pdcch_ConfigSIB1->searchSpaceZero);
#endif
  /* Prepare scc */
//  memcpy(&RC.nrrrc[0]->configuration[cell_index].scc,&RC.nrrrc[0]->configuration[init_cell_id].scc,sizeof(NR_ServingCellConfigCommon_t));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->physCellId, RC.nrrrc[0]->configuration[init_cell_id].scc->physCellId, sizeof(NR_PhysCellId_t));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->downlinkConfigCommon, RC.nrrrc[0]->configuration[init_cell_id].scc->downlinkConfigCommon, sizeof(struct NR_DownlinkConfigCommon));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->downlinkConfigCommon->frequencyInfoDL, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->downlinkConfigCommon->frequencyInfoDL, 
    sizeof(struct NR_FrequencyInfoDL));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencySSB, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencySSB, 
    sizeof(NR_ARFCN_ValueNR_t));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->downlinkConfigCommon->initialDownlinkBWP, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->downlinkConfigCommon->initialDownlinkBWP, 
    sizeof(struct NR_BWP_DownlinkCommon));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->downlinkConfigCommon->initialDownlinkBWP->pdcch_ConfigCommon, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->downlinkConfigCommon->initialDownlinkBWP->pdcch_ConfigCommon, 
    sizeof(struct NR_SetupRelease_PDCCH_ConfigCommon));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->downlinkConfigCommon->initialDownlinkBWP->pdcch_ConfigCommon->choice.setup, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->downlinkConfigCommon->initialDownlinkBWP->pdcch_ConfigCommon->choice.setup, 
    sizeof(struct NR_PDCCH_ConfigCommon));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->downlinkConfigCommon->initialDownlinkBWP->pdcch_ConfigCommon->choice.setup->controlResourceSetZero, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->downlinkConfigCommon->initialDownlinkBWP->pdcch_ConfigCommon->choice.setup->controlResourceSetZero, 
    sizeof(long));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->downlinkConfigCommon->initialDownlinkBWP->pdcch_ConfigCommon->choice.setup->searchSpaceZero, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->downlinkConfigCommon->initialDownlinkBWP->pdcch_ConfigCommon->choice.setup->searchSpaceZero, 
    sizeof(long));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->downlinkConfigCommon->initialDownlinkBWP->pdsch_ConfigCommon, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->downlinkConfigCommon->initialDownlinkBWP->pdsch_ConfigCommon, 
    sizeof(struct NR_SetupRelease_PDSCH_ConfigCommon));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->downlinkConfigCommon->initialDownlinkBWP->pdsch_ConfigCommon->choice.setup, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->downlinkConfigCommon->initialDownlinkBWP->pdsch_ConfigCommon->choice.setup,
    sizeof(struct NR_PDSCH_ConfigCommon));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->downlinkConfigCommon->initialDownlinkBWP->pdsch_ConfigCommon->choice.setup->pdsch_TimeDomainAllocationList, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->downlinkConfigCommon->initialDownlinkBWP->pdsch_ConfigCommon->choice.setup->pdsch_TimeDomainAllocationList,
    sizeof(struct NR_PDSCH_TimeDomainResourceAllocationList));


  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->uplinkConfigCommon, RC.nrrrc[0]->configuration[init_cell_id].scc->uplinkConfigCommon, sizeof(struct NR_UplinkConfigCommon));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->uplinkConfigCommon->frequencyInfoUL, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->uplinkConfigCommon->frequencyInfoUL, 
    sizeof(struct NR_FrequencyInfoUL));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->uplinkConfigCommon->frequencyInfoUL->frequencyBandList, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->uplinkConfigCommon->frequencyInfoUL->frequencyBandList,
    sizeof(struct NR_MultiFrequencyBandListNR));
#if 0 //bygz128620  TODO:  uncomment here,  a memeory  access issue hit, to figure  out  where resetting this point
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->uplinkConfigCommon->frequencyInfoUL->absoluteFrequencyPointA, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->uplinkConfigCommon->frequencyInfoUL->absoluteFrequencyPointA,
    sizeof(NR_ARFCN_ValueNR_t));
#endif
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->uplinkConfigCommon->frequencyInfoUL->p_Max, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->uplinkConfigCommon->frequencyInfoUL->p_Max,
    sizeof(NR_P_Max_t));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->uplinkConfigCommon->initialUplinkBWP, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->uplinkConfigCommon->initialUplinkBWP, 
    sizeof(struct NR_BWP_UplinkCommon));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon, 
    sizeof(NR_SetupRelease_RACH_ConfigCommon_t));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup,
    sizeof(struct NR_RACH_ConfigCommon));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->ssb_perRACH_OccasionAndCB_PreamblesPerSSB, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->ssb_perRACH_OccasionAndCB_PreamblesPerSSB,
    sizeof(struct NR_RACH_ConfigCommon__ssb_perRACH_OccasionAndCB_PreamblesPerSSB));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->rsrp_ThresholdSSB, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->rsrp_ThresholdSSB,
    sizeof(NR_RSRP_Range_t));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->msg1_SubcarrierSpacing, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->msg1_SubcarrierSpacing,
    sizeof(NR_SubcarrierSpacing_t));
#if 0
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->msg3_transformPrecoder, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->msg3_transformPrecoder,
    sizeof(long));
#endif
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->uplinkConfigCommon->initialUplinkBWP->pusch_ConfigCommon, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->uplinkConfigCommon->initialUplinkBWP->pusch_ConfigCommon, 
    sizeof(NR_SetupRelease_PUSCH_ConfigCommon_t));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->uplinkConfigCommon->initialUplinkBWP->pusch_ConfigCommon->choice.setup, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->uplinkConfigCommon->initialUplinkBWP->pusch_ConfigCommon->choice.setup,
    sizeof(struct NR_PUSCH_ConfigCommon));
  /*memcpy(RC.nrrrc[0]->configuration[cell_index].scc->uplinkConfigCommon->initialUplinkBWP->pusch_ConfigCommon->choice.setup->groupHoppingEnabledTransformPrecoding, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->uplinkConfigCommon->initialUplinkBWP->pusch_ConfigCommon->choice.setup->groupHoppingEnabledTransformPrecoding,
    sizeof(long));*/
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->uplinkConfigCommon->initialUplinkBWP->pusch_ConfigCommon->choice.setup->pusch_TimeDomainAllocationList, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->uplinkConfigCommon->initialUplinkBWP->pusch_ConfigCommon->choice.setup->pusch_TimeDomainAllocationList,
    sizeof(struct NR_PUSCH_TimeDomainResourceAllocationList));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->uplinkConfigCommon->initialUplinkBWP->pusch_ConfigCommon->choice.setup->msg3_DeltaPreamble, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->uplinkConfigCommon->initialUplinkBWP->pusch_ConfigCommon->choice.setup->msg3_DeltaPreamble,
    sizeof(long));
      memcpy(RC.nrrrc[0]->configuration[cell_index].scc->uplinkConfigCommon->initialUplinkBWP->pusch_ConfigCommon->choice.setup->p0_NominalWithGrant, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->uplinkConfigCommon->initialUplinkBWP->pusch_ConfigCommon->choice.setup->p0_NominalWithGrant,
    sizeof(long));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->uplinkConfigCommon->initialUplinkBWP->pucch_ConfigCommon, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->uplinkConfigCommon->initialUplinkBWP->pucch_ConfigCommon,
    sizeof(struct NR_SetupRelease_PUCCH_ConfigCommon));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->uplinkConfigCommon->initialUplinkBWP->pucch_ConfigCommon->choice.setup, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->uplinkConfigCommon->initialUplinkBWP->pucch_ConfigCommon->choice.setup,
    sizeof(struct NR_PUCCH_ConfigCommon));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->uplinkConfigCommon->initialUplinkBWP->pucch_ConfigCommon->choice.setup->p0_nominal, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->uplinkConfigCommon->initialUplinkBWP->pucch_ConfigCommon->choice.setup->p0_nominal,
    sizeof(long));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->uplinkConfigCommon->initialUplinkBWP->pucch_ConfigCommon->choice.setup->pucch_ResourceCommon, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->uplinkConfigCommon->initialUplinkBWP->pucch_ConfigCommon->choice.setup->pucch_ResourceCommon,
    sizeof(long));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->uplinkConfigCommon->initialUplinkBWP->pucch_ConfigCommon->choice.setup->hoppingId, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->uplinkConfigCommon->initialUplinkBWP->pucch_ConfigCommon->choice.setup->hoppingId,
    sizeof(long));
    
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->ssb_periodicityServingCell, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->ssb_periodicityServingCell, 
    sizeof(long));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->ssbSubcarrierSpacing, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->ssbSubcarrierSpacing, 
    sizeof(NR_SubcarrierSpacing_t));
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->tdd_UL_DL_ConfigurationCommon, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->tdd_UL_DL_ConfigurationCommon, 
    sizeof(struct NR_TDD_UL_DL_ConfigCommon));
#if 0
  memcpy(RC.nrrrc[0]->configuration[cell_index].scc->tdd_UL_DL_ConfigurationCommon->pattern2, 
    RC.nrrrc[0]->configuration[init_cell_id].scc->tdd_UL_DL_ConfigurationCommon->pattern2, 
    sizeof(struct NR_TDD_UL_DL_Pattern));
#endif



#if 0
  /* Prepare scd */
  memcpy(RC.nrrrc[0]->configuration[cell_index].scd,RC.nrrrc[0]->configuration[init_cell_id].scd,sizeof(NR_ServingCellConfig_t));

  LOG_I(GNB_APP,"OUTPUT cell_index: %d\n",cell_index);     
  LOG_I(GNB_APP,"Address: %ld %ld %ld %ld %ld %ld\n",RC.nrrrc[0]->carrier[cell_index].pdcch_ConfigSIB1,RC.nrrrc[0]->carrier[init_cell_id].pdcch_ConfigSIB1,RC.nrrrc[0]->configuration[cell_index].scc,RC.nrrrc[0]->configuration[init_cell_id].scc,RC.nrrrc[0]->configuration[cell_index].scd,RC.nrrrc[0]->configuration[init_cell_id].scd);     
  LOG_I(GNB_APP,"absoluteFrequencySSB: %d\n",*RC.nrrrc[0]->configuration[cell_index].scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencySSB);
  LOG_I(GNB_APP,"absoluteFrequencyPointA: %d\n",RC.nrrrc[0]->configuration[cell_index].scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencyPointA);
  LOG_I(GNB_APP,"controlResourceSetZero: %d\n",*RC.nrrrc[0]->configuration[cell_index].scc->downlinkConfigCommon->initialDownlinkBWP->pdcch_ConfigCommon->choice.setup->controlResourceSetZero);
  LOG_I(GNB_APP,"hoppingId: %d\n",*RC.nrrrc[0]->configuration[cell_index].scc->uplinkConfigCommon->initialUplinkBWP->pucch_ConfigCommon->choice.setup->hoppingId);
  LOG_I(GNB_APP," scc->ssb_PositionsInBurst->present: %d\n", RC.nrrrc[0]->configuration[cell_index].scc->ssb_PositionsInBurst->present);
  LOG_I(GNB_APP,"Read in pdcch_ConfigSIB1ParamList controlResourceSetZero:%ld searchSpaceZero:%ld \n",
             (unsigned long)RC.nrrrc[0]->carrier[cell_index].pdcch_ConfigSIB1->controlResourceSetZero,
             (unsigned long)RC.nrrrc[0]->carrier[cell_index].pdcch_ConfigSIB1->searchSpaceZero
         );
#endif

}

