/*
 * Copyright 2022 Sequans Communications.
 *
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.0  (the "License"); you may not use this file
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
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

#pragma once

#include "SidlCompiler.h"
#include "SidlASN1.h"
#include "SidlASN1Base.h"
#include "SidlBase.h"
#include "SidlCommon.h"
#include "SidlCommonBase.h"
#include "SidlNrASN1.h"
#include "SidlNrCommon.h"

SIDL_BEGIN_C_INTERFACE

struct NR_ASN1_TDD_UL_DL_SlotConfig_Type_NR_TDD_UL_DL_SlotConfigList_Type_Dynamic {
	size_t d;
	struct NR_ASN1_TDD_UL_DL_SlotConfig_Type* v;
};

typedef struct NR_ASN1_TDD_UL_DL_SlotConfig_Type_NR_TDD_UL_DL_SlotConfigList_Type_Dynamic NR_TDD_UL_DL_SlotConfigList_Type;

struct NR_ASN1_TDD_UL_DL_ConfigCommon_Type_Common_Optional {
	bool d;
	struct NR_ASN1_TDD_UL_DL_ConfigCommon_Type v;
};

struct NR_TDD_UL_DL_SlotConfigList_Type_Dedicated_Optional {
	bool d;
	NR_TDD_UL_DL_SlotConfigList_Type v;
};

struct NR_TDD_Config_Type {
	struct NR_ASN1_TDD_UL_DL_ConfigCommon_Type_Common_Optional Common;
	struct NR_TDD_UL_DL_SlotConfigList_Type_Dedicated_Optional Dedicated;
};

struct NR_FDD_Info_Type {
};

enum NR_TDD_Info_Type_Sel {
	NR_TDD_Info_Type_UNBOUND_VALUE = 0,
	NR_TDD_Info_Type_Config = 1,
	NR_TDD_Info_Type_FullFlexible = 2,
};

union NR_TDD_Info_Type_Value {
	struct NR_TDD_Config_Type Config;
	Null_Type FullFlexible;
};

struct NR_TDD_Info_Type {
	enum NR_TDD_Info_Type_Sel d;
	union NR_TDD_Info_Type_Value v;
};

enum NR_DuplexMode_Type_Sel {
	NR_DuplexMode_Type_UNBOUND_VALUE = 0,
	NR_DuplexMode_Type_FDD = 1,
	NR_DuplexMode_Type_TDD = 2,
};

union NR_DuplexMode_Type_Value {
	struct NR_FDD_Info_Type FDD;
	struct NR_TDD_Info_Type TDD;
};

struct NR_DuplexMode_Type {
	enum NR_DuplexMode_Type_Sel d;
	union NR_DuplexMode_Type_Value v;
};

struct SQN_NR_PhysCellId_PhysicalCellId_Optional {
	bool d;
	SQN_NR_PhysCellId v;
};

struct NR_DuplexMode_Type_DuplexMode_Optional {
	bool d;
	struct NR_DuplexMode_Type v;
};

struct NR_CellConfigPhysicalLayerCommon_Type {
	struct SQN_NR_PhysCellId_PhysicalCellId_Optional PhysicalCellId;
	struct NR_DuplexMode_Type_DuplexMode_Optional DuplexMode;
};

typedef SQN_NR_ServingCellConfigCommon_ssb_periodicityServingCell_e NR_SSB_Periodicity_Type;

enum NR_SS_BlockPattern_Type {
	NR_SS_BlockPattern_caseA = 0,
	NR_SS_BlockPattern_caseB = 1,
	NR_SS_BlockPattern_caseC = 2,
	NR_SS_BlockPattern_caseD = 3,
	NR_SS_BlockPattern_caseE = 4,
};

typedef enum NR_SS_BlockPattern_Type NR_SS_BlockPattern_Type;

typedef struct SQN_NR_ServingCellConfigCommon_ssb_PositionsInBurst NR_SSB_PositionsInBurst_Type;

struct int32_t_NR_SSB_Beam_Type_SsbIndex_Optional {
	bool d;
	int32_t v;
};

struct int32_t_NR_SSB_Beam_Type_Attenuation_Optional {
	bool d;
	int32_t v;
};

struct NR_SSB_Beam_Type {
	struct int32_t_NR_SSB_Beam_Type_SsbIndex_Optional SsbIndex;
	struct int32_t_NR_SSB_Beam_Type_Attenuation_Optional Attenuation;
};

struct NR_SSB_Beam_Type_NR_SSB_BeamArray_Type_Dynamic {
	size_t d;
	struct NR_SSB_Beam_Type* v;
};

typedef struct NR_SSB_Beam_Type_NR_SSB_BeamArray_Type_Dynamic NR_SSB_BeamArray_Type;

struct NR_SS_BlockPattern_Type_BlockPattern_Optional {
	bool d;
	NR_SS_BlockPattern_Type v;
};

struct NR_SSB_PositionsInBurst_Type_PositionsInBurst_Optional {
	bool d;
	NR_SSB_PositionsInBurst_Type v;
};

struct NR_SSB_BeamArray_Type_BeamArray_Optional {
	bool d;
	NR_SSB_BeamArray_Type v;
};

struct NR_SSB_BurstConfig_Type {
	struct NR_SS_BlockPattern_Type_BlockPattern_Optional BlockPattern;
	struct NR_SSB_PositionsInBurst_Type_PositionsInBurst_Optional PositionsInBurst;
	struct NR_SSB_BeamArray_Type_BeamArray_Optional BeamArray;
};

typedef int32_t NR_EPRE_Ratio_Type;

struct NR_EPRE_Ratio_Type_PbchToDmrs_Optional {
	bool d;
	NR_EPRE_Ratio_Type v;
};

struct NR_EPRE_Ratio_Type_PssToSss_Optional {
	bool d;
	NR_EPRE_Ratio_Type v;
};

struct NR_EPRE_Ratio_Type_SssToSsbBeam_Optional {
	bool d;
	NR_EPRE_Ratio_Type v;
};

struct NR_EPRE_Ratio_Type_DmrsToSss_Optional {
	bool d;
	NR_EPRE_Ratio_Type v;
};

struct NR_SSB_EPREs_Type {
	struct NR_EPRE_Ratio_Type_PbchToDmrs_Optional PbchToDmrs;
	struct NR_EPRE_Ratio_Type_PssToSss_Optional PssToSss;
	struct NR_EPRE_Ratio_Type_SssToSsbBeam_Optional SssToSsbBeam;
	struct NR_EPRE_Ratio_Type_DmrsToSss_Optional DmrsToSss;
};

struct SQN_NR_SubcarrierSpacing_e_SubCarrierSpacing_Optional {
	bool d;
	SQN_NR_SubcarrierSpacing_e v;
};

struct int32_t_NR_SSB_Config_Type_SubcarrierOffset_Optional {
	bool d;
	int32_t v;
};

struct NR_SSB_Periodicity_Type_Periodicity_Optional {
	bool d;
	NR_SSB_Periodicity_Type v;
};

struct int32_t_NR_SSB_Config_Type_HalfFrameOffset_Optional {
	bool d;
	int32_t v;
};

struct NR_SSB_BurstConfig_Type_BurstConfig_Optional {
	bool d;
	struct NR_SSB_BurstConfig_Type v;
};

struct NR_SSB_EPREs_Type_RelativeTxPower_Optional {
	bool d;
	struct NR_SSB_EPREs_Type v;
};

struct NR_SSB_Config_Type {
	struct SQN_NR_SubcarrierSpacing_e_SubCarrierSpacing_Optional SubCarrierSpacing;
	struct int32_t_NR_SSB_Config_Type_SubcarrierOffset_Optional SubcarrierOffset;
	struct NR_SSB_Periodicity_Type_Periodicity_Optional Periodicity;
	struct int32_t_NR_SSB_Config_Type_HalfFrameOffset_Optional HalfFrameOffset;
	struct NR_SSB_BurstConfig_Type_BurstConfig_Optional BurstConfig;
	struct NR_SSB_EPREs_Type_RelativeTxPower_Optional RelativeTxPower;
};

typedef SQN_NR_MIB_dmrs_TypeA_Position_e NR_PDSCH_DMRS_TypeA_Position_Type;

struct NR_ASN1_RateMatchPattern_Type_NR_RateMatchPatternList_Type_Dynamic {
	size_t d;
	struct NR_ASN1_RateMatchPattern_Type* v;
};

typedef struct NR_ASN1_RateMatchPattern_Type_NR_RateMatchPatternList_Type_Dynamic NR_RateMatchPatternList_Type;

struct NR_ASN1_RateMatchPatternLTE_CRS_Type_NR_RateMatchPatternLteCrsList_Type_Dynamic {
	size_t d;
	struct NR_ASN1_RateMatchPatternLTE_CRS_Type* v;
};

typedef struct NR_ASN1_RateMatchPatternLTE_CRS_Type_NR_RateMatchPatternLteCrsList_Type_Dynamic NR_RateMatchPatternLteCrsList_Type;

struct NR_RateMatchPatternList_Type_PatternList_Optional {
	bool d;
	NR_RateMatchPatternList_Type v;
};

struct NR_RateMatchPatternLteCrsList_Type_PatternListLteCrs_Optional {
	bool d;
	NR_RateMatchPatternLteCrsList_Type v;
};

struct NR_CellLevelRateMatchPattern_Type {
	struct NR_RateMatchPatternList_Type_PatternList_Optional PatternList;
	struct NR_RateMatchPatternLteCrsList_Type_PatternListLteCrs_Optional PatternListLteCrs;
};

struct NR_PDSCH_DMRS_TypeA_Position_Type_DMRS_TypeA_Position_Optional {
	bool d;
	NR_PDSCH_DMRS_TypeA_Position_Type v;
};

struct NR_CellLevelRateMatchPattern_Type_RateMatchPattern_Optional {
	bool d;
	struct NR_CellLevelRateMatchPattern_Type v;
};

struct NR_ASN1_PDSCH_ServingCellConfig_Type_ServingCellConfig_Optional {
	bool d;
	struct NR_ASN1_PDSCH_ServingCellConfig_Type v;
};

struct NR_PDSCH_CellLevelConfig_Type {
	struct NR_PDSCH_DMRS_TypeA_Position_Type_DMRS_TypeA_Position_Optional DMRS_TypeA_Position;
	struct NR_CellLevelRateMatchPattern_Type_RateMatchPattern_Optional RateMatchPattern;
	struct NR_ASN1_PDSCH_ServingCellConfig_Type_ServingCellConfig_Optional ServingCellConfig;
};

typedef UInt_Type NR_SearchSpaceCandidatePriority_Type;

struct NR_SearchSpaceTypeAndPriority_Type {
	NR_SearchSpaceType_Type Type;
	NR_SearchSpaceCandidatePriority_Type CandidatePriority;
};

struct NR_SearchSpaceTypeAndPriority_Type_NR_SearchSpaceTypeAndPriorityList_Type_Dynamic {
	size_t d;
	struct NR_SearchSpaceTypeAndPriority_Type* v;
};

typedef struct NR_SearchSpaceTypeAndPriority_Type_NR_SearchSpaceTypeAndPriorityList_Type_Dynamic NR_SearchSpaceTypeAndPriorityList_Type;

enum NR_PDCCH_CCE_AggregationLevel_Type {
	NR_PDCCH_CCE_AggregationLevel_AggregationLevel1 = 0,
	NR_PDCCH_CCE_AggregationLevel_AggregationLevel2 = 1,
	NR_PDCCH_CCE_AggregationLevel_AggregationLevel4 = 2,
	NR_PDCCH_CCE_AggregationLevel_AggregationLevel8 = 3,
	NR_PDCCH_CCE_AggregationLevel_AggregationLevel16 = 4,
};

typedef enum NR_PDCCH_CCE_AggregationLevel_Type NR_PDCCH_CCE_AggregationLevel_Type;

struct NR_BWP_SearchSpaceConfig_Type {
	NR_SearchSpaceTypeAndPriorityList_Type TypeAndPriorityList;
	NR_PDCCH_CCE_AggregationLevel_Type AggregationLevel;
	struct NR_ASN1_SearchSpace_Type SearchSpaceConfigAtUE;
};

struct NR_BWP_SearchSpaceConfig_Type_NR_BWP_SearchSpaceList_Type_Dynamic {
	size_t d;
	struct NR_BWP_SearchSpaceConfig_Type* v;
};

typedef struct NR_BWP_SearchSpaceConfig_Type_NR_BWP_SearchSpaceList_Type_Dynamic NR_BWP_SearchSpaceList_Type;

struct NR_ASN1_ControlResourceSet_Type_NR_BWP_CoresetList_Type_Dynamic {
	size_t d;
	struct NR_ASN1_ControlResourceSet_Type* v;
};

typedef struct NR_ASN1_ControlResourceSet_Type_NR_BWP_CoresetList_Type_Dynamic NR_BWP_CoresetList_Type;

struct NR_EPRE_Ratio_Type_PdcchToCell_Optional {
	bool d;
	NR_EPRE_Ratio_Type v;
};

struct NR_EPRE_Ratio_Type_PdcchToDmrs_Optional {
	bool d;
	NR_EPRE_Ratio_Type v;
};

struct NR_PDCCH_EPREs_Type {
	struct NR_EPRE_Ratio_Type_PdcchToCell_Optional PdcchToCell;
	struct NR_EPRE_Ratio_Type_PdcchToDmrs_Optional PdcchToDmrs;
};

struct NR_BWP_SearchSpaceList_Type_SearchSpaceArray_Optional {
	bool d;
	NR_BWP_SearchSpaceList_Type v;
};

struct NR_BWP_CoresetList_Type_CoresetArray_Optional {
	bool d;
	NR_BWP_CoresetList_Type v;
};

struct NR_PDCCH_EPREs_Type_RelativeTxPower_Optional {
	bool d;
	struct NR_PDCCH_EPREs_Type v;
};

struct int32_t_NR_BWP_PDCCH_Configuration_Type_Coreset0_OffsetRBs_Optional {
	bool d;
	int32_t v;
};

struct NR_BWP_PDCCH_Configuration_Type {
	struct NR_BWP_SearchSpaceList_Type_SearchSpaceArray_Optional SearchSpaceArray;
	struct NR_BWP_CoresetList_Type_CoresetArray_Optional CoresetArray;
	struct NR_PDCCH_EPREs_Type_RelativeTxPower_Optional RelativeTxPower;
	struct int32_t_NR_BWP_PDCCH_Configuration_Type_Coreset0_OffsetRBs_Optional Coreset0_OffsetRBs;
};

struct NR_EPRE_Ratio_Type_PdschToCell_Optional {
	bool d;
	NR_EPRE_Ratio_Type v;
};

struct NR_EPRE_Ratio_Type_PdschToDmrs_Optional {
	bool d;
	NR_EPRE_Ratio_Type v;
};

struct NR_EPRE_Ratio_Type_PdschToPtrs_Optional {
	bool d;
	NR_EPRE_Ratio_Type v;
};

struct NR_PDSCH_EPREs_Type {
	struct NR_EPRE_Ratio_Type_PdschToCell_Optional PdschToCell;
	struct NR_EPRE_Ratio_Type_PdschToDmrs_Optional PdschToDmrs;
	struct NR_EPRE_Ratio_Type_PdschToPtrs_Optional PdschToPtrs;
};

struct NR_ASN1_PDSCH_ConfigCommon_Type_ConfigCommon_Optional {
	bool d;
	struct NR_ASN1_PDSCH_ConfigCommon_Type v;
};

struct NR_ASN1_PDSCH_Config_Type_ConfigDedicated_Optional {
	bool d;
	struct NR_ASN1_PDSCH_Config_Type v;
};

struct NR_PDSCH_EPREs_Type_RelativeTxPower_Optional {
	bool d;
	struct NR_PDSCH_EPREs_Type v;
};

struct NR_BWP_PDSCH_Configuration_Type {
	struct NR_ASN1_PDSCH_ConfigCommon_Type_ConfigCommon_Optional ConfigCommon;
	struct NR_ASN1_PDSCH_Config_Type_ConfigDedicated_Optional ConfigDedicated;
	struct NR_PDSCH_EPREs_Type_RelativeTxPower_Optional RelativeTxPower;
};

struct SQN_NR_BWP_Id_NR_DownlinkBWP_Type_Id_Optional {
	bool d;
	SQN_NR_BWP_Id v;
};

struct NR_ASN1_BWP_Type_BWP_Optional {
	bool d;
	struct NR_ASN1_BWP_Type v;
};

struct NR_BWP_PDCCH_Configuration_Type_Pdcch_Optional {
	bool d;
	struct NR_BWP_PDCCH_Configuration_Type v;
};

struct NR_BWP_PDSCH_Configuration_Type_Pdsch_Optional {
	bool d;
	struct NR_BWP_PDSCH_Configuration_Type v;
};

struct NR_ASN1_SPS_Config_Type_Sps_Optional {
	bool d;
	struct NR_ASN1_SPS_Config_Type v;
};

struct NR_DownlinkBWP_Type {
	struct SQN_NR_BWP_Id_NR_DownlinkBWP_Type_Id_Optional Id;
	struct NR_ASN1_BWP_Type_BWP_Optional BWP;
	struct NR_BWP_PDCCH_Configuration_Type_Pdcch_Optional Pdcch;
	struct NR_BWP_PDSCH_Configuration_Type_Pdsch_Optional Pdsch;
	struct NR_ASN1_SPS_Config_Type_Sps_Optional Sps;
};

struct NR_DownlinkBWP_Type_NR_DownlinkBWP_List_Type_Dynamic {
	size_t d;
	struct NR_DownlinkBWP_Type* v;
};

typedef struct NR_DownlinkBWP_Type_NR_DownlinkBWP_List_Type_Dynamic NR_DownlinkBWP_List_Type;

struct SQN_NR_BWP_Id_ActiveBWP_Optional {
	bool d;
	SQN_NR_BWP_Id v;
};

struct NR_DownlinkBWP_List_Type_BwpArray_Optional {
	bool d;
	NR_DownlinkBWP_List_Type v;
};

struct NR_DownlinkBWPs_Type {
	struct SQN_NR_BWP_Id_ActiveBWP_Optional ActiveBWP;
	struct NR_DownlinkBWP_List_Type_BwpArray_Optional BwpArray;
};

enum NR_CSI_RS_Periodicity_Type_Sel {
	NR_CSI_RS_Periodicity_Type_UNBOUND_VALUE = 0,
	NR_CSI_RS_Periodicity_Type_PeriodicityAndOffset = 1,
};

union NR_CSI_RS_Periodicity_Type_Value {
	struct SQN_NR_CSI_ResourcePeriodicityAndOffset PeriodicityAndOffset;
};

struct NR_CSI_RS_Periodicity_Type {
	enum NR_CSI_RS_Periodicity_Type_Sel d;
	union NR_CSI_RS_Periodicity_Type_Value v;
};

struct SQN_NR_ScramblingId_ScramblingId_Optional {
	bool d;
	SQN_NR_ScramblingId v;
};

struct SQN_NR_CSI_RS_ResourceMapping_ResourceMapping_Optional {
	bool d;
	struct SQN_NR_CSI_RS_ResourceMapping v;
};

struct int32_t_NR_NZP_CSI_RS_Config_Type_OffsetToFreqBand_Optional {
	bool d;
	int32_t v;
};

struct NR_CSI_RS_Periodicity_Type_Periodicity_Optional {
	bool d;
	struct NR_CSI_RS_Periodicity_Type v;
};

struct int32_t_NR_NZP_CSI_RS_Config_Type_Attenuation_Optional {
	bool d;
	int32_t v;
};

struct NR_NZP_CSI_RS_Config_Type {
	struct SQN_NR_ScramblingId_ScramblingId_Optional ScramblingId;
	struct SQN_NR_CSI_RS_ResourceMapping_ResourceMapping_Optional ResourceMapping;
	struct int32_t_NR_NZP_CSI_RS_Config_Type_OffsetToFreqBand_Optional OffsetToFreqBand;
	struct NR_CSI_RS_Periodicity_Type_Periodicity_Optional Periodicity;
	struct int32_t_NR_NZP_CSI_RS_Config_Type_Attenuation_Optional Attenuation;
};

struct NR_NZP_CSI_RS_Config_Type_NR_NZP_CSI_RS_ConfigList_Type_Dynamic {
	size_t d;
	struct NR_NZP_CSI_RS_Config_Type* v;
};

typedef struct NR_NZP_CSI_RS_Config_Type_NR_NZP_CSI_RS_ConfigList_Type_Dynamic NR_NZP_CSI_RS_ConfigList_Type;

enum NR_CSI_Config_Type_Sel {
	NR_CSI_Config_Type_UNBOUND_VALUE = 0,
	NR_CSI_Config_Type_CSI_RS = 1,
	NR_CSI_Config_Type_None = 2,
};

union NR_CSI_Config_Type_Value {
	NR_NZP_CSI_RS_ConfigList_Type CSI_RS;
	Null_Type None;
};

struct NR_CSI_Config_Type {
	enum NR_CSI_Config_Type_Sel d;
	union NR_CSI_Config_Type_Value v;
};

struct NR_ASN1_FrequencyInfoDL_Type_FrequencyInfoDL_Optional {
	bool d;
	struct NR_ASN1_FrequencyInfoDL_Type v;
};

struct NR_SSB_Config_Type_SSPbchBlock_Optional {
	bool d;
	struct NR_SSB_Config_Type v;
};

struct NR_PDSCH_CellLevelConfig_Type_PdschCellLevelConfig_Optional {
	bool d;
	struct NR_PDSCH_CellLevelConfig_Type v;
};

struct NR_DownlinkBWPs_Type_BWPs_Optional {
	bool d;
	struct NR_DownlinkBWPs_Type v;
};

struct NR_CSI_Config_Type_CsiConfig_Optional {
	bool d;
	struct NR_CSI_Config_Type v;
};

struct NR_CellConfigPhysicalLayerDownlink_Type {
	struct NR_ASN1_FrequencyInfoDL_Type_FrequencyInfoDL_Optional FrequencyInfoDL;
	struct NR_SSB_Config_Type_SSPbchBlock_Optional SSPbchBlock;
	struct NR_PDSCH_CellLevelConfig_Type_PdschCellLevelConfig_Optional PdschCellLevelConfig;
	struct NR_DownlinkBWPs_Type_BWPs_Optional BWPs;
	struct NR_CSI_Config_Type_CsiConfig_Optional CsiConfig;
};

enum NR_ActiveUplinkBWP_Id_Type_Sel {
	NR_ActiveUplinkBWP_Id_Type_UNBOUND_VALUE = 0,
	NR_ActiveUplinkBWP_Id_Type_Explicit = 1,
	NR_ActiveUplinkBWP_Id_Type_SameIdAsDL = 2,
};

union NR_ActiveUplinkBWP_Id_Type_Value {
	SQN_NR_BWP_Id Explicit;
	Null_Type SameIdAsDL;
};

struct NR_ActiveUplinkBWP_Id_Type {
	enum NR_ActiveUplinkBWP_Id_Type_Sel d;
	union NR_ActiveUplinkBWP_Id_Type_Value v;
};

struct SQN_NR_BWP_Id_NR_UplinkBWP_Type_Id_Optional {
	bool d;
	SQN_NR_BWP_Id v;
};

struct NR_ASN1_BWP_UplinkCommon_Type_Common_Optional {
	bool d;
	struct NR_ASN1_BWP_UplinkCommon_Type v;
};

struct NR_ASN1_BWP_UplinkDedicated_Type_Dedicated_Optional {
	bool d;
	struct NR_ASN1_BWP_UplinkDedicated_Type v;
};

struct NR_UplinkBWP_Type {
	struct SQN_NR_BWP_Id_NR_UplinkBWP_Type_Id_Optional Id;
	struct NR_ASN1_BWP_UplinkCommon_Type_Common_Optional Common;
	struct NR_ASN1_BWP_UplinkDedicated_Type_Dedicated_Optional Dedicated;
};

struct NR_UplinkBWP_Type_NR_UplinkBWP_List_Type_Dynamic {
	size_t d;
	struct NR_UplinkBWP_Type* v;
};

typedef struct NR_UplinkBWP_Type_NR_UplinkBWP_List_Type_Dynamic NR_UplinkBWP_List_Type;

struct NR_ActiveUplinkBWP_Id_Type_ActiveBWP_Optional {
	bool d;
	struct NR_ActiveUplinkBWP_Id_Type v;
};

struct NR_UplinkBWP_List_Type_BwpArray_Optional {
	bool d;
	NR_UplinkBWP_List_Type v;
};

struct NR_UplinkBWPs_Type {
	struct NR_ActiveUplinkBWP_Id_Type_ActiveBWP_Optional ActiveBWP;
	struct NR_UplinkBWP_List_Type_BwpArray_Optional BwpArray;
};

struct NR_ASN1_FrequencyInfoUL_Type_FrequencyInfoUL_Optional {
	bool d;
	struct NR_ASN1_FrequencyInfoUL_Type v;
};

struct NR_UplinkBWPs_Type_BWPs_Optional {
	bool d;
	struct NR_UplinkBWPs_Type v;
};

struct NR_ASN1_RACH_ConfigDedicated_Type_RACH_ConfigDedicated_Optional {
	bool d;
	struct NR_ASN1_RACH_ConfigDedicated_Type v;
};

struct NR_ASN1_SI_RequestConfig_Type_SI_RequestConfig_Optional {
	bool d;
	struct NR_ASN1_SI_RequestConfig_Type v;
};

struct NR_UplinkConfig_Type {
	struct NR_ASN1_FrequencyInfoUL_Type_FrequencyInfoUL_Optional FrequencyInfoUL;
	struct NR_UplinkBWPs_Type_BWPs_Optional BWPs;
	struct NR_ASN1_RACH_ConfigDedicated_Type_RACH_ConfigDedicated_Optional RACH_ConfigDedicated;
	struct NR_ASN1_SI_RequestConfig_Type_SI_RequestConfig_Optional SI_RequestConfig;
};

enum NR_Uplink_Type_Sel {
	NR_Uplink_Type_UNBOUND_VALUE = 0,
	NR_Uplink_Type_Config = 1,
	NR_Uplink_Type_None = 2,
};

union NR_Uplink_Type_Value {
	struct NR_UplinkConfig_Type Config;
	Null_Type None;
};

struct NR_Uplink_Type {
	enum NR_Uplink_Type_Sel d;
	union NR_Uplink_Type_Value v;
};

enum NR_SS_TimingAdvanceConfig_Type_Sel {
	NR_SS_TimingAdvanceConfig_Type_UNBOUND_VALUE = 0,
	NR_SS_TimingAdvanceConfig_Type_InitialValue = 1,
	NR_SS_TimingAdvanceConfig_Type_Relative = 2,
};

union NR_SS_TimingAdvanceConfig_Type_Value {
	NR_RACH_TimingAdvance_Type InitialValue;
	NR_TimingAdvanceIndex_Type Relative;
};

struct NR_SS_TimingAdvanceConfig_Type {
	enum NR_SS_TimingAdvanceConfig_Type_Sel d;
	union NR_SS_TimingAdvanceConfig_Type_Value v;
};

struct NR_Uplink_Type_Uplink_Optional {
	bool d;
	struct NR_Uplink_Type v;
};

struct NR_Uplink_Type_SupplementaryUplink_Optional {
	bool d;
	struct NR_Uplink_Type v;
};

struct NR_SS_TimingAdvanceConfig_Type_TimingAdvance_Optional {
	bool d;
	struct NR_SS_TimingAdvanceConfig_Type v;
};

struct NR_ASN1_PUSCH_ServingCellConfig_Type_PUSCH_ServingCellConfig_Optional {
	bool d;
	struct NR_ASN1_PUSCH_ServingCellConfig_Type v;
};

struct NR_ASN1_PUSCH_ServingCellConfig_Type_PUSCH_ServingCellConfigSUL_Optional {
	bool d;
	struct NR_ASN1_PUSCH_ServingCellConfig_Type v;
};

struct NR_CellConfigPhysicalLayerUplink_Type {
	struct NR_Uplink_Type_Uplink_Optional Uplink;
	struct NR_Uplink_Type_SupplementaryUplink_Optional SupplementaryUplink;
	struct NR_SS_TimingAdvanceConfig_Type_TimingAdvance_Optional TimingAdvance;
	struct NR_ASN1_PUSCH_ServingCellConfig_Type_PUSCH_ServingCellConfig_Optional PUSCH_ServingCellConfig;
	struct NR_ASN1_PUSCH_ServingCellConfig_Type_PUSCH_ServingCellConfigSUL_Optional PUSCH_ServingCellConfigSUL;
};

struct NR_CellConfigPhysicalLayerCommon_Type_Common_Optional {
	bool d;
	struct NR_CellConfigPhysicalLayerCommon_Type v;
};

struct NR_CellConfigPhysicalLayerDownlink_Type_Downlink_Optional {
	bool d;
	struct NR_CellConfigPhysicalLayerDownlink_Type v;
};

struct NR_CellConfigPhysicalLayerUplink_Type_Uplink_Optional {
	bool d;
	struct NR_CellConfigPhysicalLayerUplink_Type v;
};

struct NR_CellConfigPhysicalLayer_Type {
	struct NR_CellConfigPhysicalLayerCommon_Type_Common_Optional Common;
	struct NR_CellConfigPhysicalLayerDownlink_Type_Downlink_Optional Downlink;
	struct NR_CellConfigPhysicalLayerUplink_Type_Uplink_Optional Uplink;
};

SIDL_END_C_INTERFACE
