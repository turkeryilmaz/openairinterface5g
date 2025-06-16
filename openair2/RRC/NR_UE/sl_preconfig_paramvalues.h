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

#ifndef __SL_PRECONFIG_PARAMSVALUES__H__
#define __SL_PRECONFIG_PARAMSVALUES__H__

#include "NR_SL-PreconfigurationNR-r16.h"
#include "common/config/config_userapi.h"

#define SL_CONFIG_STRING_SL_PRECONFIGURATION                   "SIDELINK_PRECONFIGURATION"

/*   Sidelink Frequency common configuration in SL-Preconfig */
#define SL_CONFIG_STRING_SL_FCC_LIST                           "sl_FrequencyCommonConfig"
#define SL_CONFIG_STRING_SLOFFSETTOCARRIER                     "sl_offstToCarrier"
#define SL_CONFIG_STRING_FCC_SUBCARRIERSPACING                 "sl_subcarrierSpacing"
#define SL_CONFIG_STRING_SLCARRIERBANDWIDTH                    "sl_carrierBandwidth"
#define SL_CONFIG_STRING_SLABSOLUTEFREQUENCYSSB                "sl_absoluteFrequencySSB"
#define SL_CONFIG_STRING_SLABSOLUEFREQUENCYPOINTA              "sl_absoluteFrequencyPointA"

// Sidelink TDD ULDL configuration parameters in SL-Preconfig
#define SL_CONFIG_STRING_DLULTRANSMISSIONPERIODICITY           "sl_dl_UL_TransmissionPeriodicity"
#define SL_CONFIG_STRING_NROFDOWNLINKSLOTS                     "sl_nrofDownlinkSlots"
#define SL_CONFIG_STRING_NROFDOWNLINKSYMBOLS                   "sl_nrofDownlinkSymbols"
#define SL_CONFIG_STRING_NROFUPLINKSLOTS                       "sl_nrofUplinkSlots"
#define SL_CONFIG_STRING_NROFUPLINKSYMBOLS                     "sl_nrofUplinkSymbols"


// Sidelink sync config parameters in SL-Preconfig.
// 3 sets possible per sync config list entry
#define SL_CONFIG_STRING_SL_SYNCCONFIG_LIST                    "sl_syncCfg"
#define SL_CONFIG_STRING_SL_NUMSSB_WITHINPERIOD_0              "sl_NumSSB_WithinPeriod_0"
#define SL_CONFIG_STRING_SL_TIMEOFFSET_SSB_0                   "sl_TimeOffsetSSB_0"
#define SL_CONFIG_STRING_SL_TIMEINTERVAL_0                     "sl_TimeInterval_0"
#define SL_CONFIG_STRING_SL_NUMSSB_WITHINPERIOD_1              "sl_NumSSB_WithinPeriod_1"
#define SL_CONFIG_STRING_SL_TIMEOFFSET_SSB_1                   "sl_TimeOffsetSSB_1"
#define SL_CONFIG_STRING_SL_TIMEINTERVAL_1                     "sl_TimeInterval_1"
#define SL_CONFIG_STRING_SL_NUMSSB_WITHINPERIOD_2              "sl_NumSSB_WithinPeriod_2"
#define SL_CONFIG_STRING_SL_TIMEOFFSET_SSB_2                   "sl_TimeOffsetSSB_2"
#define SL_CONFIG_STRING_SL_TIMEINTERVAL_2                     "sl_TimeInterval_2"


/*Sidelink Bandwidth related parameters in SL-Preconfig */
#define SL_CONFIG_STRING_SL_BWP_LIST                           "sl_BWP"
#define SL_CONFIG_STRING_SL_BWP_START_AND_SIZE                 "sl_locationAndBandwidth"
#define SL_CONFIG_STRING_SL_BWP_NUM_SYMBOLS                    "sl_LengthSymbols"
#define SL_CONFIG_STRING_SL_BWP_START_SYMBOL                   "sl_StartSymbol"



/*Sidelink Resource pool related parameters in SL-Preconfig */
#define SL_CONFIG_STRING_SL_RX_RPOOL_LIST                     "sl_RxResPools"
#define SL_CONFIG_STRING_SL_TX_RPOOL_LIST                     "sl_TxResPools"
#define SL_CONFIG_STRING_RESPOOL_PSCCH_NUMSYM                 "sl_TimeResourcePSCCH"
#define SL_CONFIG_STRING_RESPOOL_PSCCH_NUMRBS                 "sl_FreqResourcePSCCH"
#define SL_CONFIG_STRING_RESPOOL_SUBCH_SIZE_IN_RBS            "sl_SubchannelSize"
#define SL_CONFIG_STRING_RESPOOL_SUBCH_START_RB               "sl_StartRB_Subchannel"
#define SL_CONFIG_STRING_RESPOOL_NUM_RBS                      "sl_RB_Number"
#define SL_CONFIG_STRING_RESPOOL_NUM_SUBCHS                   "sl_NumSubchannel"
#define SL_CONFIG_STRING_RESPOOL_PSFCH_PERIOD                 "sl_PSFCH_Period"
#define SL_CONFIG_STRING_RESPOOL_PSFCH_RB_SET                 "sl_PSFCH_RB_Set"
#define SL_CONFIG_STRING_RESPOOL_PSFCH_NUMMUXCS_PAIR          "sl_NumMuxCS_Pair"
#define SL_CONFIG_STRING_RESPOOL_PSFCH_MINTIMEGAP             "sl_MinTimeGapPSFCH"
#define SL_CONFIG_STRING_RESPOOL_PSFCH_HOPID                  "sl_PSFCH_HopID"
#define SL_CONFIG_STRING_RESPOOL_PSFCH_CANDIDATERESOURCETYPE  "sl_PSFCH_CandidateResourceType"
#define SL_CONFIG_STRING_RSRC_SEL_PRIORITY                    "sl_Priority"

#define SL_CONFIG_STRING_RSRC_SEL_PARAMS_LIST                 "rsrc_selection_params"
#define SL_CONFIG_STRING_RSRC_SEL_SELECTION_WINDOW            "sl_SelectionWindow"
#define SL_CONFIG_STRING_RSRC_SEL_SENSING_WINDOW              "sl_SensingWindow"
#define SL_CONFIG_STRING_RSRC_SEL_TRESHOLD_RSRP               "sl_Thres_RSRP"
#define SL_CONFIG_STRING_RSRC_SEL_MAXNUM_PER_RESERVE          "sl_MaxNumPerReserve"
#define SL_CONFIG_STRING_RSRC_SEL_RESOURCE_RESERVED_PERIOD    "sl_ResourceReservePeriod"
#define SL_CONFIG_STRING_RSRC_SEL_RS_FOR_SENSING              "sl_RS_ForSensing"
#define SL_CONFIG_STRING_RSRC_SEL_TX_PERCENTAGE               "sl_TxPercentage"

#define SL_CONFIG_STRING_UEINFO                               "sl_UEINFO"
#define SL_CONFIG_STRING_UEINFO_SRCID                         "srcid"
#define SL_CONFIG_STRING_UEINFO_IPV4ADDR_THIRD_OCTET          "thirdOctet"
#define SL_CONFIG_STRING_UEINFO_IPV4ADDR_FOURTH_OCTET         "fourthOctet"
#define SL_CONFIG_STRING_UEINFO_REMOTE_UE_ID                  "remote_ue_id"
#define SL_CONFIG_STRING_UEINFO_IS_RELAY_UE                   "is_relay_ue"
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*             Sidelink Frequency common Cell Config parameters                                                                                                     */
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/* Please refer to 3GPP SPEC 38.331 (RRC Specification) for details about these parameters
   sl_AbsoluteFrequencyPointA_r16 - ARFCN of the lowest subcarrier for sidelink operation
   sl_AbsoluteFrequencySSB_r16 - ARFCN of the center RE of sidelink SSB of length 11 RBs
*/
#define SL_FCCPARAMS_DESC(sl_fcc) { \
{SL_CONFIG_STRING_SLOFFSETTOCARRIER,NULL,0,.i64ptr=&sl_fcc->sl_SCS_SpecificCarrierList_r16.list.array[0]->offsetToCarrier,.defint64val=0,TYPE_INT64,0}, \
{SL_CONFIG_STRING_FCC_SUBCARRIERSPACING,NULL,0,.i64ptr=&sl_fcc->sl_SCS_SpecificCarrierList_r16.list.array[0]->subcarrierSpacing,.defint64val=NR_SubcarrierSpacing_kHz30,TYPE_INT64,0},\
{SL_CONFIG_STRING_SLCARRIERBANDWIDTH,NULL,0,.i64ptr=&sl_fcc->sl_SCS_SpecificCarrierList_r16.list.array[0]->carrierBandwidth,.defint64val=106,TYPE_INT64,0}, \
{SL_CONFIG_STRING_SLABSOLUEFREQUENCYPOINTA,NULL,0,.i64ptr=&sl_fcc->sl_AbsoluteFrequencyPointA_r16,.defint64val=792000,TYPE_INT64,0},\
{SL_CONFIG_STRING_SLABSOLUTEFREQUENCYSSB,NULL,0,.i64ptr=sl_fcc->sl_AbsoluteFrequencySSB_r16,.defint64val=792372,TYPE_INT64,0}}


/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*             Sidelink TDD ULDL Config parameters                                                                                                     */
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/* Please refer to 3GPP SPEC 38.331 (RRC Specification) for details about these parameters
   TDD ul dl configuration used for sidelink operation set in Sidelink Preconfiguration IE
*/
#define SL_TDDCONFIGPARAMS_DESC(sl_tdd_ul_dl_cfg) { \
{SL_CONFIG_STRING_DLULTRANSMISSIONPERIODICITY,NULL,0,.i64ptr=&sl_tdd_ul_dl_cfg->pattern1.dl_UL_TransmissionPeriodicity,.defint64val=6,TYPE_INT64,0}, \
{SL_CONFIG_STRING_NROFDOWNLINKSLOTS,NULL,0,.i64ptr=&sl_tdd_ul_dl_cfg->pattern1.nrofDownlinkSlots,.defint64val=7,TYPE_INT64,0},\
{SL_CONFIG_STRING_NROFDOWNLINKSYMBOLS,NULL,0,.i64ptr=&sl_tdd_ul_dl_cfg->pattern1.nrofDownlinkSymbols,.defint64val=10,TYPE_INT64,0}, \
{SL_CONFIG_STRING_NROFUPLINKSLOTS,NULL,0,.i64ptr=&sl_tdd_ul_dl_cfg->pattern1.nrofUplinkSlots,.defint64val=2,TYPE_INT64,0},\
{SL_CONFIG_STRING_NROFUPLINKSYMBOLS,NULL,0,.i64ptr=&sl_tdd_ul_dl_cfg->pattern1.nrofUplinkSymbols,.defint64val=4,TYPE_INT64,0}}


/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*             Sidelink Sync config parameters                                                                                                     */
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/* Please refer to 3GPP SPEC 38.331 (RRC Specification) for details about these parameters
   These parameters indicate the repetition of Sidelink SSB
   sl_NumSSB_WithinPeriod_r16 - number of SSB transmissions within 16 frames
   sl_TimeOffsetSSB_r16 - timeoffset in slots for the first SL-SSB transmission within 16 frames
   sl_TimeInterval_r16 - interval in slots between SL-SSB transmissions
*/
#define SL_SYNCPARAMS_DESC(sl_syncconfig) { \
{SL_CONFIG_STRING_SL_NUMSSB_WITHINPERIOD_0,NULL,0,.i64ptr=sl_syncconfig->sl_SSB_TimeAllocation1_r16->sl_NumSSB_WithinPeriod_r16,.defint64val=1,TYPE_INT64,0}, \
{SL_CONFIG_STRING_SL_TIMEOFFSET_SSB_0,NULL,0,.i64ptr=sl_syncconfig->sl_SSB_TimeAllocation1_r16->sl_TimeOffsetSSB_r16,.defint64val=8,TYPE_INT64,0},\
{SL_CONFIG_STRING_SL_TIMEINTERVAL_0,NULL,0,.i64ptr=sl_syncconfig->sl_SSB_TimeAllocation1_r16->sl_TimeInterval_r16,.defint64val=60,TYPE_INT64,0}}


/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*             Sidelink BWP Config parameters                                                                                                     */
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/* Please refer to 3GPP SPEC 38.331 (RRC Specification) for details about these parameters
   Sidelink BandWidthPart parameters
   locationAndBandwidth - determines the start and size of Sidelink BWP
   sl_LengthSymbols_r16 - number of symbols in a slot used for sidelink operation
   sl_StartSymbol_r16 - start symbol for sidelink operation in a slot
*/
#define SL_BWPPARAMS_DESC(sl_bwp) { \
{SL_CONFIG_STRING_SL_BWP_START_AND_SIZE,NULL,0,.i64ptr=&sl_bwp->sl_BWP_Generic_r16->sl_BWP_r16->locationAndBandwidth,.defint64val=28875,TYPE_INT64,0}, \
{SL_CONFIG_STRING_SL_BWP_NUM_SYMBOLS,NULL,0,.i64ptr=sl_bwp->sl_BWP_Generic_r16->sl_LengthSymbols_r16,.defint64val=7,TYPE_INT64,0}, \
{SL_CONFIG_STRING_SL_BWP_START_SYMBOL,NULL,0,.i64ptr=sl_bwp->sl_BWP_Generic_r16->sl_StartSymbol_r16,.defint64val=0,TYPE_INT64,0}}

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*             Sidelink Resource pool parameters                                                                                                     */
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/* Please refer to 3GPP SPEC 38.331 (RRC Specification) for details about these parameters
   Sidelink TX/RX resource pool parameters
   sl_TimeResourcePSCCH_r16 - number of symbols used for PSCCH
   sl_SubchannelSize_r16 - number of PRBs in every subchannel in a resource pool
   sl_StartRB_Subchannel_r16 - lowest RB of a lowest subchannel in a resource pool
   sl_RB_Number_r16 - number of RBs in a resource pool
*/
#define SL_RESPOOLPARAMS_DESC(sl_res_pool) { \
{SL_CONFIG_STRING_RESPOOL_PSCCH_NUMSYM,NULL,0,.i64ptr=sl_res_pool->sl_PSCCH_Config_r16->choice.setup->sl_TimeResourcePSCCH_r16,.defint64val=1,TYPE_INT64,0}, \
{SL_CONFIG_STRING_RESPOOL_PSCCH_NUMRBS,NULL,0,.i64ptr=sl_res_pool->sl_PSCCH_Config_r16->choice.setup->sl_FreqResourcePSCCH_r16,.defint64val=1,TYPE_INT64,0}, \
{SL_CONFIG_STRING_RESPOOL_SUBCH_SIZE_IN_RBS,NULL,0,.i64ptr=sl_res_pool->sl_SubchannelSize_r16,.defint64val=0,TYPE_INT64,0},\
{SL_CONFIG_STRING_RESPOOL_SUBCH_START_RB,NULL,0,.i64ptr=sl_res_pool->sl_StartRB_Subchannel_r16,.defint64val=0,TYPE_INT64,0},\
{SL_CONFIG_STRING_RESPOOL_NUM_RBS,NULL,0,.i64ptr=sl_res_pool->sl_RB_Number_r16,.defint64val=106,TYPE_INT64,0},\
{SL_CONFIG_STRING_RESPOOL_NUM_SUBCHS,NULL,0,.i64ptr=sl_res_pool->sl_NumSubchannel_r16,.defint64val=10,TYPE_INT64,0},\
{SL_CONFIG_STRING_RESPOOL_PSFCH_PERIOD,NULL,0,.i64ptr=sl_res_pool->sl_PSFCH_Config_r16->choice.setup->sl_PSFCH_Period_r16,.defint64val=3,TYPE_INT64,0}, \
{SL_CONFIG_STRING_RESPOOL_PSFCH_NUMMUXCS_PAIR,NULL,0,.i64ptr=sl_res_pool->sl_PSFCH_Config_r16->choice.setup->sl_NumMuxCS_Pair_r16,.defint64val=1,TYPE_INT64,0}, \
{SL_CONFIG_STRING_RESPOOL_PSFCH_MINTIMEGAP,NULL,0,.i64ptr=sl_res_pool->sl_PSFCH_Config_r16->choice.setup->sl_MinTimeGapPSFCH_r16,.defint64val=1,TYPE_INT64,0}, \
{SL_CONFIG_STRING_RESPOOL_PSFCH_HOPID,NULL,0,.i64ptr=sl_res_pool->sl_PSFCH_Config_r16->choice.setup->sl_PSFCH_HopID_r16,.defint64val=1,TYPE_INT64,0}, \
{SL_CONFIG_STRING_RESPOOL_PSFCH_CANDIDATERESOURCETYPE,NULL,0,.i64ptr=sl_res_pool->sl_PSFCH_Config_r16->choice.setup->sl_PSFCH_CandidateResourceType_r16,.defint64val=0,TYPE_INT64,0}}

#define SL_RSRCSELPARAMS_DESC(sl_rsrc_sel_pool) { \
{SL_CONFIG_STRING_RSRC_SEL_PRIORITY,NULL,0,.i64ptr=&sl_res_pool->sl_UE_SelectedConfigRP_r16->sl_SelectionWindowList_r16->list.array[0]->sl_Priority_r16,.defint64val=7,TYPE_INT64,0}, \
{SL_CONFIG_STRING_RSRC_SEL_SELECTION_WINDOW,NULL,0,.i64ptr=&sl_res_pool->sl_UE_SelectedConfigRP_r16->sl_SelectionWindowList_r16->list.array[0]->sl_SelectionWindow_r16,.defint64val=5,TYPE_INT64,0}, \
{SL_CONFIG_STRING_RSRC_SEL_SENSING_WINDOW,NULL,0,.i64ptr=sl_res_pool->sl_UE_SelectedConfigRP_r16->sl_SensingWindow_r16,.defint64val=6,TYPE_INT64,0}, \
{SL_CONFIG_STRING_RSRC_SEL_TRESHOLD_RSRP,NULL,0,.i64ptr=sl_res_pool->sl_UE_SelectedConfigRP_r16->sl_Thres_RSRP_List_r16->list.array[0],.defint64val=-128,TYPE_INT64,0}, \
{SL_CONFIG_STRING_RSRC_SEL_MAXNUM_PER_RESERVE,NULL,0,.i64ptr=sl_res_pool->sl_UE_SelectedConfigRP_r16->sl_MaxNumPerReserve_r16,.defint64val=1,TYPE_INT64,0}, \
{SL_CONFIG_STRING_RSRC_SEL_RESOURCE_RESERVED_PERIOD,NULL,0,.i64ptr=&sl_res_pool->sl_UE_SelectedConfigRP_r16->sl_ResourceReservePeriodList_r16->list.array[0]->choice.sl_ResourceReservePeriod1_r16,.defint64val=100,TYPE_INT64,0}, \
{SL_CONFIG_STRING_RSRC_SEL_RS_FOR_SENSING,NULL,0,.i64ptr=&sl_res_pool->sl_UE_SelectedConfigRP_r16->sl_RS_ForSensing_r16,.defint64val=1,TYPE_INT64,0}, \
{SL_CONFIG_STRING_RSRC_SEL_TX_PERCENTAGE,NULL,0,.i64ptr=&sl_res_pool->sl_TxPercentageList_r16->list.array[0]->sl_TxPercentage_r16,.defint64val=0,TYPE_INT64,0}, \
}

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*             Sidelink Top-Level UE Info                                                                                                     */
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/* 
   sl_srcid - 16-bit source ID used for LCID information and indexing in L2 stack 
   sl_ipv4addr - string containing ipv4 address for default SLRB 
*/
#define SL_UEINFO_DESC(sl_ueinfo) { \
{SL_CONFIG_STRING_UEINFO_SRCID,NULL,0,.iptr=&sl_ueinfo.srcid,.defintval=1,TYPE_INT,0}, \
{SL_CONFIG_STRING_UEINFO_IPV4ADDR_THIRD_OCTET,NULL,0,.iptr=&sl_ueinfo.thirdOctet,.defintval=0,TYPE_INT,0}, \
{SL_CONFIG_STRING_UEINFO_IPV4ADDR_FOURTH_OCTET,NULL,0,.iptr=&sl_ueinfo.fourthOctet,.defintval=1,TYPE_INT,0}, \
{SL_CONFIG_STRING_UEINFO_REMOTE_UE_ID,NULL,0,.u8ptr=&sl_ueinfo.remote_ue_id,.defintval=0,TYPE_UINT8,0}, \
{SL_CONFIG_STRING_UEINFO_IS_RELAY_UE,NULL,0,.u8ptr=&sl_ueinfo.is_relay_ue,.defintval=0,TYPE_UINT8,0}}
#endif
