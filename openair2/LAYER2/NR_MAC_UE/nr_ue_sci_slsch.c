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

/* \file nr_ue_sci.c
 * \brief handling of sci/slsch procedures for Sidelink UE
 * \author R. Knopp
 * \date 2023
 * \version 0.1
 * \company Eurecom
 * \email: knopp@eurecom.fr
 * \note
 * \warning
 */


#include <stdio.h>
#include <math.h>

/* exe */
#include "executables/nr-softmodem.h"

/* RRC*/
#include "RRC/NR_UE/rrc_proto.h"
#include "NR_SL-BWP-ConfigCommon-r16.h"

/* MAC */
#include "NR_MAC_UE/nr_ue_sci.h"
#include "NR_MAC_COMMON/nr_mac.h"
#include "NR_MAC_UE/mac_proto.h"
#include "NR_MAC_UE/mac_extern.h"
#include "NR_MAC_COMMON/nr_mac_extern.h"
#include "common/utils/nr/nr_common.h"
#include "openair2/NR_UE_PHY_INTERFACE/NR_Packet_Drop.h"

/* PHY */
#include "executables/softmodem-common.h"
#include "openair1/PHY/defs_nr_UE.h"

/* utils */
#include "assertions.h"
#include "oai_asn1.h"
#include "common/utils/LOG/log.h"
#include "common/utils/LOG/vcd_signal_dumper.h"


const int sl_dmrs_mask2[2][8] = { {34,34,34,264,264,1032,1032,1032},
                                  {34,34,34,272,272,1040,1040,1040}};
const int sl_dmrs_mask3[5]    = {146,146,546,546,2114};
const int sl_dmrs_mask4[3]    = {1170,1170,1170};
const int pscch_rb_table[5] = {10,12,15,20,25};
const int pscch_tda[2] = {2,3};

const int subch_to_rb[8] = {10,12,15,20,25,50,75,100};

uint32_t nr_sci_size(const NR_SL_ResourcePool_r16_t *sl_res_pool,
	             nr_sci_pdu_t *sci_pdu,
	             const nr_sci_format_t format) {
			     
  int size=0;

  switch(format) {
    case NR_SL_SCI_FORMAT_1A:
	    // priority
	    size+=3;
	    // frequency resource assignment
	    long Nsc = *sl_res_pool->sl_NumSubchannel_r16;
            if (sl_res_pool->sl_UE_SelectedConfigRP_r16 && 
                sl_res_pool->sl_UE_SelectedConfigRP_r16->sl_MaxNumPerReserve_r16 &&
                *sl_res_pool->sl_UE_SelectedConfigRP_r16->sl_MaxNumPerReserve_r16 == NR_SL_UE_SelectedConfigRP_r16__sl_MaxNumPerReserve_r16_n2)
	      sci_pdu->frequency_resource_assignment.nbits =  (uint8_t)ceil(log2((Nsc * (Nsc + 1)) >>1));  
	    else
	      sci_pdu->frequency_resource_assignment.nbits =  (uint8_t)ceil(log2((Nsc * (Nsc + 1) * (2*Nsc + 1)) /6));  
            size += sci_pdu->frequency_resource_assignment.nbits;
	    // time-domain-assignment
            if (*sl_res_pool->sl_UE_SelectedConfigRP_r16->sl_MaxNumPerReserve_r16 == NR_SL_UE_SelectedConfigRP_r16__sl_MaxNumPerReserve_r16_n2)
		sci_pdu->time_resource_assignment.nbits = 5;
	    else
		sci_pdu->time_resource_assignment.nbits = 9;
            size += sci_pdu->time_resource_assignment.nbits;
	    
	    // resource reservation period
	    
	    if (1 /*!sl_res_pool->sl_MultiReserveResource*/) // not defined in 17.4 RRC
	       sci_pdu->resource_reservation_period.nbits = 0;
            size += sci_pdu->resource_reservation_period.nbits;

	    // DMRS pattern
	    int dmrs_pattern_num = sl_res_pool->sl_PSSCH_Config_r16->choice.setup->sl_PSSCH_DMRS_TimePatternList_r16->list.count;
	    sci_pdu->dmrs_pattern.nbits = (uint8_t)ceil(log2(dmrs_pattern_num));
	    size += sci_pdu->dmrs_pattern.nbits;

            // second_stage_sci_format // 2 bits - Table 8.3.1.1-1
	    size += 2;
	    // beta_offset_indicator // 2 bits - depending sl-BetaOffsets2ndSCI and Table 8.3.1.1-2
	    size += 2;
	    // number_of_dmrs_port // 1 bit - Table 8.3.1.1-3
	    size += 1;
            // mcs // 5 bits
	    size += 5;
           
	    // additional_mcs; // depending on sl-Additional-MCS-Table
	    if (sl_res_pool->sl_Additional_MCS_Table_r16) 
	       sci_pdu->additional_mcs.nbits = (*sl_res_pool->sl_Additional_MCS_Table_r16 < 2) ? 1 : 2;
	    else sci_pdu->additional_mcs.nbits=0;
	    size += sci_pdu->additional_mcs.nbits;

	    // psfch_overhead; // depending on sl-PSFCH-Period
	    if (sl_res_pool->sl_PSFCH_Config_r16 && sl_res_pool->sl_PSFCH_Config_r16->choice.setup->sl_PSFCH_Period_r16 && *sl_res_pool->sl_PSFCH_Config_r16->choice.setup->sl_PSFCH_Period_r16>1)
		sci_pdu->psfch_overhead.nbits=1;
	    else sci_pdu->psfch_overhead.nbits=0;
	    size += sci_pdu->psfch_overhead.nbits;

   	    // reserved; // depending on N_reserved (sl-NumReservedBits) and sl-IndicationUE-B
	    // note R17 dependence no sl_IndicationUE-B needs to be added here
	    AssertFatal(sl_res_pool->sl_PSCCH_Config_r16!=NULL,"sl_res_pool->sl_PSCCH_Config_r16 is null\n");
	    AssertFatal(sl_res_pool->sl_PSCCH_Config_r16->choice.setup!=NULL,"sl_res_pool->sl_PSCCH_Config_r16->choice.setup is null\n");
	    AssertFatal(sl_res_pool->sl_PSCCH_Config_r16->choice.setup->sl_NumReservedBits_r16!=NULL, "sl_res_pool->sl_PSCCH_Config_r16->choice.setup->sl_NumReservedBits_r16 is null\n");
            sci_pdu->reserved.nbits = *sl_res_pool->sl_PSCCH_Config_r16->choice.setup->sl_NumReservedBits_r16;

            // conflict_information_receiver; // depending on sl-IndicationUE-B 
	    // note: R17 field not included here
	    sci_pdu->conflict_information_receiver.nbits=0;
	    break;
    case NR_SL_SCI_FORMAT_2A:
    case NR_SL_SCI_FORMAT_2B:
    case NR_SL_SCI_FORMAT_2C:
	    // common components
	    //harq_pid; // 4 bits
	    //ndi; // 1 bit
	    //rv_index; // 2 bits
	    //source_id; // 8 bits
	    //dest_id; // 16 bits
	    //harq_feedback; //1 bit
	    size += (4+1+2+8+16+1);
	    if (format==NR_SL_SCI_FORMAT_2A)
	      //cast_type // 2 bits formac 2A
	      size += 2;
	    if (format==NR_SL_SCI_FORMAT_2C || format==NR_SL_SCI_FORMAT_2A)
	      // csi_req // 1 bit format 2A, format 2C
              size +=1;
	    if (format==NR_SL_SCI_FORMAT_2B) { 
	      // zone_id // 12 bits format 2B
	      size +=12;

	      // communication_range; // 4 bits depending on sl-ZoneConfigMCR-Index, format 2B
	      // note fill in for R17
	      if (0) size +=4;
	    }
	    else if (format==NR_SL_SCI_FORMAT_2C) {

       	     // providing_req_ind; // 1 bit, format 2C
	     size += 1;
             // resource_combinations; // depending on n_subChannel^SL (sl-NumSubchennel), N_rsv_period (sl-ResourceReservePeriodList) and sl-MultiReservedResource, format 2C
	     // first_resource_location; // 8 bits, format 2C
	     size += 8;
	     // reference_slot_location; // depending on mu, format 2C
             // resource_set_type; // 1 bit, format 2C
	     size += 1;
	     //	lowest_subchannel_indices; // depending on n_subChannel^SL, format 2C
	     //
	    }
	    break;
  }
  return(size);
}


int get_nREDMRS(const NR_SL_ResourcePool_r16_t *sl_res_pool) {

  int cnt = sl_res_pool->sl_PSSCH_Config_r16->choice.setup->sl_PSSCH_DMRS_TimePatternList_r16->list.count;
  int nREDMRS = 0;
  for (int i=0;i<cnt;i++)
    nREDMRS += *sl_res_pool->sl_PSSCH_Config_r16->choice.setup->sl_PSSCH_DMRS_TimePatternList_r16->list.array[i] * 6;
  return(nREDMRS/cnt);
}

#define MAX_EL_213_9_3_2 19
const float tab38_213_9_3_2[MAX_EL_213_9_3_2] = {1.125,1.250,1.375,1.625,1.750,2.000,2.250,2.500,2.875,3.125,3.500,4.000,5.000,6.250,8.000,10.000,12.625,15.875,20.000};
const float alpha_tab[4] = {0.5,0.65,0.8,1.0};

int get_NREsci2(const NR_SL_ResourcePool_r16_t *sl_res_pool,
                const sl_nr_tx_config_pscch_pssch_pdu_t *nr_sl_pssch_pscch_pdu,
                const int mcs_tb_ind) {

  float Osci2 = (float)nr_sl_pssch_pscch_pdu->sci2_payload_len;
  AssertFatal(nr_sl_pssch_pscch_pdu->sci2_beta_offset < MAX_EL_213_9_3_2, "illegal sci2_beta_offset %d\n",nr_sl_pssch_pscch_pdu->sci2_beta_offset);
  float beta_offset_sci2 = tab38_213_9_3_2[nr_sl_pssch_pscch_pdu->sci2_beta_offset];


  uint32_t R10240 = nr_get_code_rate_ul(nr_sl_pssch_pscch_pdu->mcs,mcs_tb_ind); 

  uint32_t tmp  = (uint32_t)ceil((Osci2 + 24)*beta_offset_sci2/(R10240/5120));
  float tmp2 = 12.0*nr_sl_pssch_pscch_pdu->pssch_numsym;
  int N_REsci1  = 12*nr_sl_pssch_pscch_pdu->pscch_numrbs*nr_sl_pssch_pscch_pdu->pscch_numsym;
  tmp2 *= nr_sl_pssch_pscch_pdu->l_subch*nr_sl_pssch_pscch_pdu->subchannel_size;
  tmp2 -= N_REsci1;
  AssertFatal(*sl_res_pool->sl_PSSCH_Config_r16->choice.setup->sl_Scaling_r16 < 4, "Illegal index %d to alpha table\n",(int)*sl_res_pool->sl_PSSCH_Config_r16->choice.setup->sl_Scaling_r16);
  tmp2 *= alpha_tab[*sl_res_pool->sl_PSSCH_Config_r16->choice.setup->sl_Scaling_r16];
  return min(tmp,(int)ceil(tmp2)); 
 
}

void fill_pssch_pscch_pdu(sl_nr_tx_config_pscch_pssch_pdu_t *nr_sl_pssch_pscch_pdu,
			  const NR_SL_BWP_ConfigCommon_r16_t *sl_bwp, 
                          const NR_SL_ResourcePool_r16_t *sl_res_pool,
		          nr_sci_pdu_t *sci_pdu, 
		          nr_sci_pdu_t *sci2_pdu, 
                          uint16_t slsch_pdu_length,
		          const nr_sci_format_t format1,
		          const nr_sci_format_t format2)  {
  int pos=0,fsize;
  uint64_t *sci_payload  =  (uint64_t *)nr_sl_pssch_pscch_pdu->pscch_sci_payload;
  uint64_t *sci2_payload =  (uint64_t *)nr_sl_pssch_pscch_pdu->sci2_payload;
  nr_sl_pssch_pscch_pdu->pscch_sci_payload_len  = nr_sci_size(sl_res_pool,sci_pdu,format1);
  nr_sl_pssch_pscch_pdu->sci2_payload_len = nr_sci_size(sl_res_pool,sci2_pdu,format2);
  int sci_size  = nr_sl_pssch_pscch_pdu->pscch_sci_payload_len;
  int sci2_size = nr_sl_pssch_pscch_pdu->sci2_payload_len;


  // freq domain allocation starts
  nr_sl_pssch_pscch_pdu->startrb=*sl_res_pool->sl_StartRB_Subchannel_r16;
  // Number of symbols used for PSCCH
  nr_sl_pssch_pscch_pdu->pscch_numsym = pscch_tda[*sl_res_pool->sl_PSCCH_Config_r16->choice.setup->sl_TimeResourcePSCCH_r16];
  // Number of  RBS used for PSCCH
  nr_sl_pssch_pscch_pdu->pscch_numrbs = pscch_rb_table[*sl_res_pool->sl_PSCCH_Config_r16->choice.setup->sl_FreqResourcePSCCH_r16];
  // Scrambling Id used for Generation of PSCCH DMRS Symbols
  nr_sl_pssch_pscch_pdu->pscch_dmrs_scrambling_id = *sl_res_pool->sl_PSCCH_Config_r16->choice.setup->sl_DMRS_ScrambleID_r16;
  // num subchannels in a resource pool
  nr_sl_pssch_pscch_pdu->num_subch = *sl_res_pool->sl_NumSubchannel_r16;
  // Size of subchannels in RBs
  nr_sl_pssch_pscch_pdu->subchannel_size = subch_to_rb[*sl_res_pool->sl_SubchannelSize_r16];
  //_PSCCH PSSCH TX: Size of subchannels in a PSSCH resource (l_subch)
  AssertFatal(sci_pdu->time_resource_assignment.val == 0, "need to handle a non-zero time_resource_assignment (2 or 3 time hops, N=2,3)\n");
  convNRFRIV(sci_pdu->frequency_resource_assignment.val,
             nr_sl_pssch_pscch_pdu->num_subch,
             *sl_res_pool->sl_UE_SelectedConfigRP_r16->sl_MaxNumPerReserve_r16,
             &nr_sl_pssch_pscch_pdu->l_subch,
             NULL,NULL);
  //number of symbols for Sidelink transmission on PSSCH/PSCCH
  //(Total Sidelink symbols available - number of psfch symbols configured - 2)
  //Guard symbol + AGC symbol are also excluded
  //Indicates the number of symbols for PSCCH+PSSCH txn
  int num_psfch_symbols=0;
  if (sl_res_pool->sl_PSFCH_Config_r16 && sl_res_pool->sl_PSFCH_Config_r16->choice.setup->sl_PSFCH_Period_r16 && *sl_res_pool->sl_PSFCH_Config_r16->choice.setup->sl_PSFCH_Period_r16>0) {
     num_psfch_symbols = *sl_res_pool->sl_PSFCH_Config_r16->choice.setup->sl_PSFCH_Period_r16;
     if (num_psfch_symbols == 3) num_psfch_symbols++;
  }
  nr_sl_pssch_pscch_pdu->pssch_numsym=7+*sl_bwp->sl_BWP_Generic_r16->sl_LengthSymbols_r16-num_psfch_symbols-2;
  nr_sl_pssch_pscch_pdu->pssch_startsym = *sl_bwp->sl_BWP_Generic_r16->sl_StartSymbol_r16;

  nr_sl_pssch_pscch_pdu->sci2_beta_offset = *sl_res_pool->sl_PSSCH_Config_r16->choice.setup->sl_BetaOffsets2ndSCI_r16->list.array[sci_pdu->beta_offset_indicator];
  if (sl_res_pool->sl_PowerControl_r16)
    nr_sl_pssch_pscch_pdu->sci2_alpha_times_100 = (*sl_res_pool->sl_PowerControl_r16->sl_Alpha_PSSCH_PSCCH_r16 == 0) ? 0 : (3+(*sl_res_pool->sl_PowerControl_r16->sl_Alpha_PSSCH_PSCCH_r16))*100;

  else nr_sl_pssch_pscch_pdu->sci2_alpha_times_100 = 100;

  switch(format1) {
    case NR_SL_SCI_FORMAT_1A:
	    // priority 3 bits
	    fsize=3;
	    for (int i = 0; i < fsize; i++)
		   *sci_payload |= (((uint64_t)sci_pdu->priority >> (fsize - i - 1)) & 1) << (sci_size - pos++);
	    
	    // frequency resource assignment
	    fsize = sci_pdu->frequency_resource_assignment.nbits;  
	    for (int i = 0; i < fsize; i++)
		   *sci_payload |= (((uint64_t)sci_pdu->frequency_resource_assignment.val >> (fsize - i - 1)) & 1) << (sci_size - pos++);
	    // time-domain-assignment
	    fsize = sci_pdu->time_resource_assignment.nbits;
	    for (int i = 0; i < fsize; i++)
		   *sci_payload |= (((uint64_t)sci_pdu->time_resource_assignment.val >> (fsize - i - 1)) & 1) << (sci_size - pos++);
	    // resource reservation period
            fsize = sci_pdu->resource_reservation_period.nbits;
	    for (int i = 0; i < fsize; i++)
		   *sci_payload |= (((uint64_t)sci_pdu->resource_reservation_period.val >> (fsize - i - 1)) & 1) << (sci_size - pos++);
	    // DMRS pattern
	    fsize = sci_pdu->dmrs_pattern.nbits;
	    for (int i = 0; i < fsize; i++)
		   *sci_payload |= (((uint64_t)sci_pdu->dmrs_pattern.val >> (fsize - i - 1)) & 1) << (sci_size - pos++);
            // second_stage_sci_format // 2 bits - Table 8.3.1.1-1
	    fsize = 2;
	    for (int i = 0; i < fsize; i++)
		   *sci_payload |= (((uint64_t)sci_pdu->second_stage_sci_format >> (fsize - i - 1)) & 1) << (sci_size - pos++);
	    // beta_offset_indicator // 2 bits - depending sl-BetaOffsets2ndSCI and Table 8.3.1.1-2
	    fsize = 2;
	    for (int i = 0; i < fsize; i++)
		   *sci_payload |= (((uint64_t)sci_pdu->beta_offset_indicator >> (fsize - i - 1)) & 1) << (sci_size - pos++);
	    // number_of_dmrs_port // 1 bit - Table 8.3.1.1-3
	    fsize = 1;
	    *sci_payload |= (((uint64_t)sci_pdu->number_of_dmrs_port&1)) << (sci_size - pos++);
            // mcs // 5 bits
	    fsize = 5;
	    for (int i = 0; i < fsize; i++)
		   *sci_payload |= (((uint64_t)sci_pdu->mcs >> (fsize - i - 1)) & 1) << (sci_size - pos++);
	    // additional_mcs; // depending on sl-Additional-MCS-Table
	    fsize = sci_pdu->additional_mcs.nbits;
	    for (int i = 0; i < fsize; i++)
		   *sci_payload |= (((uint64_t)sci_pdu->additional_mcs.val >> (fsize - i - 1)) & 1) << (sci_size - pos++);
	    // psfch_overhead; // depending on sl-PSFCH-Period
   	    fsize = sci_pdu->psfch_overhead.nbits;
	    for (int i = 0; i < fsize; i++)
		   *sci_payload |= (((uint64_t)sci_pdu->psfch_overhead.val >> (fsize - i - 1)) & 1) << (sci_size - pos++);

   	    // reserved; // depending on N_reserved (sl-NumReservedBits) and sl-IndicationUE-B
            fsize = sci_pdu->reserved.nbits;
	    for (int i = 0; i < fsize; i++)
		   *sci_payload |= (((uint64_t)sci_pdu->reserved.val >> (fsize - i - 1)) & 1) << (sci_size - pos++);
            // conflict_information_receiver; // depending on sl-IndicationUE-B 
	    fsize = sci_pdu->conflict_information_receiver.nbits;
	    for (int i = 0; i < fsize; i++)
		   *sci_payload |= (((uint64_t)sci_pdu->conflict_information_receiver.val >> (fsize - i - 1)) & 1) << (sci_size - pos++);
	    break;
    default:
            AssertFatal(1==0,"Unknown format1 %d\n",format1);
            break;
  }

  int mcs_tb_ind = 0;
  if (sci_pdu->additional_mcs.nbits > 0)
    mcs_tb_ind = sci_pdu->additional_mcs.val;

  int nohPRB    = (sl_res_pool->sl_X_Overhead_r16) ? 3*(*sl_res_pool->sl_X_Overhead_r16) : 0;
  int nREDMRS   = get_nREDMRS(sl_res_pool);  
  int N_REprime = 12*nr_sl_pssch_pscch_pdu->pssch_numsym - nohPRB - nREDMRS;
  int N_REsci1  = 12*nr_sl_pssch_pscch_pdu->pscch_numrbs*nr_sl_pssch_pscch_pdu->pscch_numsym;
  int N_REsci2  = get_NREsci2(sl_res_pool,nr_sl_pssch_pscch_pdu,mcs_tb_ind);
  int N_RE      = N_REprime*nr_sl_pssch_pscch_pdu->l_subch*nr_sl_pssch_pscch_pdu->subchannel_size - N_REsci1 - N_REsci2;

  nr_sl_pssch_pscch_pdu->mod_order = nr_get_Qm_ul(sci_pdu->mcs,mcs_tb_ind);
  nr_sl_pssch_pscch_pdu->target_coderate = nr_get_code_rate_ul(sci_pdu->mcs,mcs_tb_ind);
  nr_sl_pssch_pscch_pdu->tbslbrm = nr_compute_tbs_sl(nr_sl_pssch_pscch_pdu->mod_order,
                                                     nr_sl_pssch_pscch_pdu->target_coderate,
						     N_RE,1+(sci_pdu->number_of_dmrs_port&1));

  nr_sl_pssch_pscch_pdu->mcs = sci_pdu->mcs;
  nr_sl_pssch_pscch_pdu->num_layers = sci_pdu->number_of_dmrs_port+1;
  nr_sl_pssch_pscch_pdu->mcs_table=mcs_tb_ind;
  nr_sl_pssch_pscch_pdu->rv_index = sci2_pdu->rv_index;
  nr_sl_pssch_pscch_pdu->ndi = sci2_pdu->ndi;
  int num_dmrs_symbols;
  AssertFatal(sci_pdu->dmrs_pattern.val < sl_res_pool->sl_PSSCH_Config_r16->choice.setup->sl_PSSCH_DMRS_TimePatternList_r16->list.count,"dmrs.pattern %d out of bounds for list size %d\n",sci_pdu->dmrs_pattern.val,sl_res_pool->sl_PSSCH_Config_r16->choice.setup->sl_PSSCH_DMRS_TimePatternList_r16->list.count);
  num_dmrs_symbols = *sl_res_pool->sl_PSSCH_Config_r16->choice.setup->sl_PSSCH_DMRS_TimePatternList_r16->list.array[sci_pdu->dmrs_pattern.val];
  if (num_dmrs_symbols == 2) {
    AssertFatal(nr_sl_pssch_pscch_pdu->pssch_numsym>5, "num_pssch_ymbols %d is not ok for 2 DMRS (min 6)\n",nr_sl_pssch_pscch_pdu->pssch_numsym);
    nr_sl_pssch_pscch_pdu->dmrs_symbol_position = sl_dmrs_mask2[nr_sl_pssch_pscch_pdu->pscch_numsym-2][nr_sl_pssch_pscch_pdu->pssch_numsym-6];
  } else if (num_dmrs_symbols == 3) {
    AssertFatal(nr_sl_pssch_pscch_pdu->pssch_numsym>8, "num_pssch_ymbols %d is not ok for 3 DMRS (min 9)\n",nr_sl_pssch_pscch_pdu->pssch_numsym);
    nr_sl_pssch_pscch_pdu->dmrs_symbol_position = sl_dmrs_mask3[nr_sl_pssch_pscch_pdu->pssch_numsym-9];
  } else if (num_dmrs_symbols == 4) {
    AssertFatal(nr_sl_pssch_pscch_pdu->pssch_numsym>10, "num_pssch_ymbols %d is not ok for 4 DMRS (min 11)\n",nr_sl_pssch_pscch_pdu->pssch_numsym);
    nr_sl_pssch_pscch_pdu->dmrs_symbol_position = sl_dmrs_mask4[nr_sl_pssch_pscch_pdu->pssch_numsym-11];
  }
  
  pos=0;
  switch(format2) {
    case NR_SL_SCI_FORMAT_2A:
    case NR_SL_SCI_FORMAT_2B:
    case NR_SL_SCI_FORMAT_2C:
	    // common components
	    //harq_pid; // 4 bits
            fsize = 4;
	    for (int i = 0; i < fsize; i++)
		   *sci2_payload |= (((uint64_t)sci2_pdu->harq_pid >> (fsize - i - 1)) & 1) << (sci2_size - pos++);
	    //ndi; // 1 bit
	    *sci2_payload |= ((uint64_t)sci2_pdu->ndi  & 1) << (sci2_size - pos++);
	    //rv_index; // 2 bits
            fsize = 2;
	    for (int i = 0; i < fsize; i++)
		   *sci2_payload |= (((uint64_t)sci2_pdu->rv_index >> (fsize - i - 1)) & 1) << (sci2_size - pos++);
	    //source_id; // 8 bits
            fsize = 8;
	    for (int i = 0; i < fsize; i++)
		   *sci2_payload |= (((uint64_t)sci2_pdu->source_id >> (fsize - i - 1)) & 1) << (sci2_size - pos++);
	    //dest_id; // 16 bits
            fsize = 16;
	    for (int i = 0; i < fsize; i++)
		   *sci2_payload |= (((uint64_t)sci2_pdu->dest_id >> (fsize - i - 1)) & 1) << (sci2_size - pos++);
	    //harq_feedback; //1 bit
	    *sci2_payload |= ((uint64_t)sci2_pdu->harq_feedback  & 1) << (sci2_size - pos++);
	    if (format2==NR_SL_SCI_FORMAT_2A) {
	      //cast_type // 2 bits formac 2A
	      fsize = 2;
	      for (int i = 0; i < fsize; i++)
	  	   *sci2_payload |= (((uint64_t)sci2_pdu->cast_type >> (fsize - i - 1)) & 1) << (sci2_size - pos++);
            }
	    if (format2==NR_SL_SCI_FORMAT_2C || format2==NR_SL_SCI_FORMAT_2A)
	      // csi_req // 1 bit format 2A, format 2C
	      *sci2_payload |= ((uint64_t)sci2_pdu->csi_req  & 1) << (sci2_size - pos++);
              
	    if (format2==NR_SL_SCI_FORMAT_2B) { 
	      // zone_id // 12 bits format 2B
	      fsize = 12;
	      for (int i = 0; i < fsize; i++)
	  	   *sci2_payload |= (((uint64_t)sci2_pdu->zone_id >> (fsize - i - 1)) & 1) << (sci2_size - pos++);

	      // communication_range; // 4 bits depending on sl-ZoneConfigMCR-Index, format 2B
	      // note fill in for R17
	      if (0) {
	        fsize = 4;
	        for (int i = 0; i < fsize; i++)
	  	     *sci2_payload |= (((uint64_t)sci2_pdu->communication_range.val >> (fsize - i - 1)) & 1) << (sci2_size - pos++);
              } 
	    }
	    else if (format2==NR_SL_SCI_FORMAT_2C) {

       	     // providing_req_ind; // 1 bit, format 2C
	     *sci2_payload |= ((uint64_t)sci2_pdu->providing_req_ind  & 1) << (sci2_size - pos++);
             // resource_combinations; // depending on n_subChannel^SL (sl-NumSubchennel), N_rsv_period (sl-ResourceReservePeriodList) and sl-MultiReservedResource, format 2C
             if (0) {
               fsize = 0; 
	       for (int i = 0; i < fsize; i++)
	          *sci2_payload |= (((uint64_t)sci2_pdu->resource_combinations.val >> (fsize - i - 1)) & 1) << (sci2_size - pos++);
             }
	     // first_resource_location; // 8 bits, format 2C
	     fsize = 8;
	     for (int i = 0; i < fsize; i++)
	        *sci2_payload |= (((uint64_t)sci2_pdu->first_resource_location >> (fsize - i - 1)) & 1) << (sci2_size - pos++);
	     // reference_slot_location; // depending on mu, format 2C
	     if (0) {
               fsize = 0;
	       for (int i = 0; i < fsize; i++)
	          *sci2_payload |= (((uint64_t)sci2_pdu->reference_slot_location.val >> (fsize - i - 1)) & 1) << (sci2_size - pos++);
             }
             // resource_set_type; // 1 bit, format 2C
	     *sci2_payload |= ((uint64_t)sci2_pdu->resource_set_type  & 1) << (sci2_size - pos++);
	     //	lowest_subchannel_indices; // depending on n_subChannel^SL, format 2C
	     if (0) {
	       fsize = 0;
	       for (int i = 0; i < fsize; i++)
	          *sci2_payload |= (((uint64_t)sci2_pdu->lowest_subchannel_indices.val >> (fsize - i - 1)) & 1) << (sci2_size - pos++);
             }
	     //
	    }
	    break;
        default:
          AssertFatal(1==0,"Unknown format %d for sci2\n",format2);
          break;
  }
  nr_sl_pssch_pscch_pdu->slsch_payload_length = slsch_pdu_length;
};



void config_pscch_pdu_rx(sl_nr_rx_config_pscch_pdu_t *nr_sl_pscch_pdu,
	  	         const NR_SL_BWP_ConfigCommon_r16_t *sl_bwp, 
                         const NR_SL_ResourcePool_r16_t *sl_res_pool){

  nr_sci_pdu_t dummy_sci;
  // Starting RE of the lowest subchannel in a resource where PSCCH
  // freq domain allocation starts
  nr_sl_pscch_pdu->pscch_startrb=*sl_res_pool->sl_StartRB_Subchannel_r16;
  // Number of symbols used for PSCCH
  nr_sl_pscch_pdu->pscch_numsym=pscch_tda[*sl_res_pool->sl_PSCCH_Config_r16->choice.setup->sl_TimeResourcePSCCH_r16];
  // Number of  RBS used for PSCCH
  nr_sl_pscch_pdu->pscch_numrbs=pscch_rb_table[*sl_res_pool->sl_PSCCH_Config_r16->choice.setup->sl_FreqResourcePSCCH_r16];
  // Scrambling Id used for Generation of PSCCH DMRS Symbols
  nr_sl_pscch_pdu->pscch_dmrs_scrambling_id=*sl_res_pool->sl_PSCCH_Config_r16->choice.setup->sl_DMRS_ScrambleID_r16;
;
  // num subchannels in a resource pool
  nr_sl_pscch_pdu->num_subch=*sl_res_pool->sl_NumSubchannel_r16;
  // Size of subchannels in RBs
  nr_sl_pscch_pdu->subchannel_size=subch_to_rb[*sl_res_pool->sl_SubchannelSize_r16];
  // PSCCH PSSCH RX: this is set to 1 - Blind decoding for SCI1A done on every subchannel
  // PSCCH SENSING: this is equal to number of subchannels forming a resource.

  nr_sl_pscch_pdu->l_subch=1;
  //number of symbols for Sidelink transmission on PSSCH/PSCCH
  //(Total Sidelink symbols available - number of psfch symbols configured - 2)
  //Guard symbol + AGC symbol are also excluded
  //Indicates the number of symbols for PSCCH+PSSCH txn
  int num_psfch_symbols=0;
  if (sl_res_pool->sl_PSFCH_Config_r16 && sl_res_pool->sl_PSFCH_Config_r16->choice.setup->sl_PSFCH_Period_r16 && *sl_res_pool->sl_PSFCH_Config_r16->choice.setup->sl_PSFCH_Period_r16>0) {
     num_psfch_symbols = *sl_res_pool->sl_PSFCH_Config_r16->choice.setup->sl_PSFCH_Period_r16;
     if (num_psfch_symbols == 3) num_psfch_symbols++;
  }
  nr_sl_pscch_pdu->pssch_numsym=7+*sl_bwp->sl_BWP_Generic_r16->sl_LengthSymbols_r16-num_psfch_symbols-2;
  //sci 1A length used to decode on PSCCH.
  nr_sl_pscch_pdu->sci_1a_length = nr_sci_size(sl_res_pool,&dummy_sci,NR_SL_SCI_FORMAT_1A);
  //This paramter is set if PSCCH RX is triggered on TX resource pool
  // as part of TX pool sensing procedure.
  nr_sl_pscch_pdu->sense_pscch=0;

  LOG_I(NR_MAC,"Programming PSCCH reception (sci_1a_length %d)\n",nr_sl_pscch_pdu->sci_1a_length);

}

void extract_pscch_pdu(uint64_t *sci1_payload,
	  	       const NR_SL_BWP_ConfigCommon_r16_t *sl_bwp, 
                       const NR_SL_ResourcePool_r16_t *sl_res_pool,
		       nr_sci_pdu_t *sci_pdu) { 
  int pos=0,fsize;
  int sci1_size = nr_sci_size(sl_res_pool,sci_pdu,NR_SL_SCI_FORMAT_1A);


    // priority 3 bits
  fsize=3;
  pos=fsize;
  sci_pdu->priority = *sci1_payload>>(sci1_size-pos)&((1<<fsize)-1);

  // frequency resource assignment
  fsize = sci_pdu->frequency_resource_assignment.nbits;  
  pos+=fsize;
  sci_pdu->frequency_resource_assignment.val = *sci1_payload>>(sci1_size-pos)&((1<<fsize)-1);

  // time-domain-assignment
  fsize = sci_pdu->time_resource_assignment.nbits;
  pos+=fsize;
  sci_pdu->time_resource_assignment.val = *sci1_payload>>(sci1_size-pos)&((1<<fsize)-1);

  // resource reservation period
  fsize = sci_pdu->resource_reservation_period.nbits;
  pos+=fsize;
  sci_pdu->resource_reservation_period.val = *sci1_payload>>(sci1_size-pos)&((1<<fsize)-1);
	    
  // DMRS pattern
  fsize = sci_pdu->dmrs_pattern.nbits;
  pos+=fsize;
  sci_pdu->dmrs_pattern.val = *sci1_payload>>(sci1_size-pos)&((1<<fsize)-1);

  // second_stage_sci_format // 2 bits - Table 8.3.1.1-1
  fsize = 2;
  pos+=fsize;
  sci_pdu->second_stage_sci_format = *sci1_payload>>(sci1_size-pos)&((1<<fsize)-1); 

  // beta_offset_indicator // 2 bits - depending sl-BetaOffsets2ndSCI and Table 8.3.1.1-2
  fsize = 2;
  pos+=fsize;
  sci_pdu->beta_offset_indicator = *sci1_payload>>(sci1_size-pos)&((1<<fsize)-1); 

  // number_of_dmrs_port // 1 bit - Table 8.3.1.1-3
  fsize = 1;
  pos+=fsize;
  sci_pdu->number_of_dmrs_port = *sci1_payload>>(sci1_size-pos)&((1<<fsize)-1);

  // mcs // 5 bits
  fsize = 5;
  pos+=fsize;
  sci_pdu->mcs = *sci1_payload>>(sci1_size-pos)&((1<<fsize)-1);

  // additional_mcs; // depending on sl-Additional-MCS-Table
  fsize = sci_pdu->additional_mcs.nbits;
  pos+=fsize;
  sci_pdu->additional_mcs.val = *sci1_payload>>(sci1_size-pos)&((1<<fsize)-1);

  // psfch_overhead; // depending on sl-PSFCH-Period
  fsize = sci_pdu->psfch_overhead.nbits;
  pos+=fsize;
  sci_pdu->psfch_overhead.val = *sci1_payload>>(sci1_size-pos)&((1<<fsize)-1); 

  // reserved; // depending on N_reserved (sl-NumReservedBits) and sl-IndicationUE-B
  fsize = sci_pdu->reserved.nbits;
  pos+=fsize;
  sci_pdu->reserved.val = *sci1_payload>>(sci1_size-pos)&((1<<fsize)-1); 

  // conflict_information_receiver; // depending on sl-IndicationUE-B 
  fsize = sci_pdu->conflict_information_receiver.nbits;
  pos+=fsize;
  sci_pdu->conflict_information_receiver.val = *sci1_payload>>(sci1_size-pos)&((1<<fsize)-1); 
}

int nr_ue_process_sci1_indication_pdu(NR_UE_MAC_INST_t *mac,frame_t frame, int slot, sl_nr_sci_indication_pdu_t *sci) {

  nr_sci_pdu_t sci_pdu;  //&mac->def_sci_pdu[slot][sci->sci_format_type];
  sl_nr_rx_config_pssch_sci_pdu_t nr_sl_pssch_sci_pdu;
  const NR_SL_BWP_ConfigCommon_r16_t *sl_bwp = mac->sl_bwp;
  const NR_SL_ResourcePool_r16_t *sl_res_pool = mac->sl_rx_res_pool; 

  LOG_D(MAC,"Received sci indication (sci format %d, Nid %x, subChannelIndex %d, payloadSize %d,payload %llx)\n",
        sci->sci_format_type,sci->Nid,sci->subch_index,sci->sci_payloadlen,*(unsigned long long*)sci->sci_payloadBits);
  AssertFatal(sci->sci_format_type == SL_SCI_FORMAT_1A_ON_PSCCH, "need to have format 1A here only\n");
  extract_pscch_pdu((uint64_t *)sci->sci_payloadBits, sl_bwp, sl_res_pool, &sci_pdu);
  config_pssch_sci_pdu_rx(&nr_sl_pssch_sci_pdu,
                          NR_SL_SCI_FORMAT_2A,
                          &sci_pdu,
                          sci->Nid,
                          sci->subch_index,
                          sl_bwp,
                          sl_res_pool);
  // send schedule response
  return 1;
}



void config_pssch_sci_pdu_rx(sl_nr_rx_config_pssch_sci_pdu_t *nr_sl_pssch_sci_pdu,
                             nr_sci_format_t sci2_format,
			     nr_sci_pdu_t *sci_pdu,
			     uint32_t pscch_Nid,
			     int pscch_subchannel_index,
	  	             const NR_SL_BWP_ConfigCommon_r16_t *sl_bwp, 
                             const NR_SL_ResourcePool_r16_t *sl_res_pool){

  AssertFatal(sci2_format>NR_SL_SCI_FORMAT_1A,"cannot use format 1A with this function\n");
  // Expected Length of SCI2 in bits
  nr_sl_pssch_sci_pdu->sci2_len = nr_sci_size(sl_res_pool,sci_pdu,sci2_format);
  // Used to determine number of SCI2 modulated symbols
  nr_sl_pssch_sci_pdu->sci2_beta_offset = *sl_res_pool->sl_PSSCH_Config_r16->choice.setup->sl_BetaOffsets2ndSCI_r16->list.array[sci_pdu->beta_offset_indicator];
  // Used to determine number of SCI2 modulated symbols
 //Values will be sl-scaling*100 (sl-scaling values 0.5, 0.65, 0.8, 1)
  nr_sl_pssch_sci_pdu->sci2_alpha_times_100=(*sl_res_pool->sl_PowerControl_r16->sl_Alpha_PSSCH_PSCCH_r16 == 0) ? 0 : (3+(*sl_res_pool->sl_PowerControl_r16->sl_Alpha_PSSCH_PSCCH_r16))*100;

  int mcs_tb_ind = 0;
  if (sci_pdu->additional_mcs.nbits > 0)
    mcs_tb_ind = sci_pdu->additional_mcs.val;

  nr_sl_pssch_sci_pdu->targetCodeRate = nr_get_code_rate_ul(sci_pdu->mcs,mcs_tb_ind);
  nr_sl_pssch_sci_pdu->mod_order      = nr_get_Qm_ul(sci_pdu->mcs,mcs_tb_ind);
  nr_sl_pssch_sci_pdu->num_layers     = 1+sci_pdu->number_of_dmrs_port;

  // Derived from PSCCH CRC Refer 38.211 section 8.3.1.1
  // to be used for PSSCH DMRS and PSSCH 38.211 Scrambling
  nr_sl_pssch_sci_pdu->Nid = pscch_Nid;

  // Starting RE of the lowest subchannel.
  //In Sym with PSCCH - Start of PSCCH
  //In Sym without PSCCH - Start of PSSCH
  // freq domain allocation starts
  nr_sl_pssch_sci_pdu->startrb = pscch_subchannel_index*12*(*sl_res_pool->sl_SubchannelSize_r16);
  // Number of symbols used for PSCCH
  nr_sl_pssch_sci_pdu->pscch_numsym = pscch_tda[*sl_res_pool->sl_PSCCH_Config_r16->choice.setup->sl_TimeResourcePSCCH_r16];
  // Number of  RBS used for PSCCH
  nr_sl_pssch_sci_pdu->pscch_numrbs = pscch_rb_table[*sl_res_pool->sl_PSCCH_Config_r16->choice.setup->sl_FreqResourcePSCCH_r16];
 ;
  // num subchannels in a resource pool
  nr_sl_pssch_sci_pdu->num_subch = *sl_res_pool->sl_NumSubchannel_r16;
  // Size of subchannels in RBs
  nr_sl_pssch_sci_pdu->subchannel_size = subch_to_rb[*sl_res_pool->sl_SubchannelSize_r16];
  // In case of PSCCH PSSCH RX: this is always 1. Blind decoding done for every channel
  // In case of RESOURCE SENSING: this is equal to number of subchannels forming a resource.
  nr_sl_pssch_sci_pdu->l_subch = 1;
  //number of symbols for Sidelink transmission on PSSCH/PSCCH
  //(Total Sidelink symbols available - number of psfch symbols configured - 2)
  //Guard symbol + AGC symbol are also excluded
  //Indicates the number of symbols for PSCCH+PSSCH txn
  int num_psfch_symbols=0;
  if (sl_res_pool->sl_PSFCH_Config_r16 && sl_res_pool->sl_PSFCH_Config_r16->choice.setup->sl_PSFCH_Period_r16 && *sl_res_pool->sl_PSFCH_Config_r16->choice.setup->sl_PSFCH_Period_r16>0) {
     num_psfch_symbols = *sl_res_pool->sl_PSFCH_Config_r16->choice.setup->sl_PSFCH_Period_r16;
     if (num_psfch_symbols == 3) num_psfch_symbols++;
  }
  nr_sl_pssch_sci_pdu->pssch_numsym = 7+*sl_bwp->sl_BWP_Generic_r16->sl_LengthSymbols_r16-num_psfch_symbols-2;;
  
  //DMRS SYMBOL MASK. If bit set to 1 indicates it is a DMRS symbol. LSB is symbol 0
  // Table from SPEC 38.211, Table 8.4.1.1.2-1
  int num_dmrs_symbols;
  AssertFatal(sci_pdu->dmrs_pattern.val < sl_res_pool->sl_PSSCH_Config_r16->choice.setup->sl_PSSCH_DMRS_TimePatternList_r16->list.count,"dmrs.pattern %d out of bounds for list size %d\n",sci_pdu->dmrs_pattern.val,sl_res_pool->sl_PSSCH_Config_r16->choice.setup->sl_PSSCH_DMRS_TimePatternList_r16->list.count);
  num_dmrs_symbols = *sl_res_pool->sl_PSSCH_Config_r16->choice.setup->sl_PSSCH_DMRS_TimePatternList_r16->list.array[sci_pdu->dmrs_pattern.val];
  if (num_dmrs_symbols == 2) {
    AssertFatal(nr_sl_pssch_sci_pdu->pssch_numsym>5, "num_pssch_ymbols %d is not ok for 2 DMRS (min 6)\n",nr_sl_pssch_sci_pdu->pssch_numsym);
    nr_sl_pssch_sci_pdu->dmrs_symbol_position = sl_dmrs_mask2[nr_sl_pssch_sci_pdu->pscch_numsym-2][nr_sl_pssch_sci_pdu->pssch_numsym-6];
  } else if (num_dmrs_symbols == 3) {
    AssertFatal(nr_sl_pssch_sci_pdu->pssch_numsym>8, "num_pssch_ymbols %d is not ok for 3 DMRS (min 9)\n",nr_sl_pssch_sci_pdu->pssch_numsym);
    nr_sl_pssch_sci_pdu->dmrs_symbol_position = sl_dmrs_mask3[nr_sl_pssch_sci_pdu->pssch_numsym-9];
  } else if (num_dmrs_symbols == 4) {
    AssertFatal(nr_sl_pssch_sci_pdu->pssch_numsym>10, "num_pssch_ymbols %d is not ok for 4 DMRS (min 11)\n",nr_sl_pssch_sci_pdu->pssch_numsym);
    nr_sl_pssch_sci_pdu->dmrs_symbol_position = sl_dmrs_mask4[nr_sl_pssch_sci_pdu->pssch_numsym-11];
  }

  //This paramter is set if PSSCH sensing (PSSCH DMRS RSRP measurement)
  // is triggred as part of TX pool sensing procedure.
  nr_sl_pssch_sci_pdu->sense_pssch = 0;

}


