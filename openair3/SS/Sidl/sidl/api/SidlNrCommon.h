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

SIDL_BEGIN_C_INTERFACE

typedef UInt16_Type RNTI_Value_Type;

typedef uint16_t NR_RACH_TimingAdvance_Type;

typedef uint8_t NR_TimingAdvanceIndex_Type;

typedef uint16_t NR_TimingAdvance_Period_Type;

typedef int32_t NR_ServingCellIndex_Type;

enum NR_RadioBearerId_Type_Sel {
	NR_RadioBearerId_Type_UNBOUND_VALUE = 0,
	NR_RadioBearerId_Type_Srb = 1,
	NR_RadioBearerId_Type_Drb = 2,
};

union NR_RadioBearerId_Type_Value {
	SRB_Identity_Type Srb;
	SQN_NR_DRB_Identity Drb;
};

struct NR_RadioBearerId_Type {
	enum NR_RadioBearerId_Type_Sel d;
	union NR_RadioBearerId_Type_Value v;
};

enum NR_RoutingInfo_Type_Sel {
	NR_RoutingInfo_Type_UNBOUND_VALUE = 0,
	NR_RoutingInfo_Type_None = 1,
	NR_RoutingInfo_Type_RadioBearerId = 2,
	NR_RoutingInfo_Type_QosFlow = 3,
};

union NR_RoutingInfo_Type_Value {
	Null_Type None;
	struct NR_RadioBearerId_Type RadioBearerId;
	struct QosFlow_Identification_Type QosFlow;
};

struct NR_RoutingInfo_Type {
	enum NR_RoutingInfo_Type_Sel d;
	union NR_RoutingInfo_Type_Value v;
};

typedef Null_Type NR_RoutingInfoSUL_Type;

struct MacBearerRouting_Type {
	NR_CellId_Type NR;
};

struct NR_RoutingInfoSUL_Type_RoutingInfoSUL_Optional {
	bool d;
	NR_RoutingInfoSUL_Type v;
};

struct MacBearerRouting_Type_NR_IndAspCommonPart_Type_MacBearerRouting_Optional {
	bool d;
	struct MacBearerRouting_Type v;
};

struct NR_IndAspCommonPart_Type {
	NR_CellId_Type CellId;
	struct NR_RoutingInfo_Type RoutingInfo;
	struct NR_RoutingInfoSUL_Type_RoutingInfoSUL_Optional RoutingInfoSUL;
	struct RlcBearerRouting_Type RlcBearerRouting;
	struct MacBearerRouting_Type_NR_IndAspCommonPart_Type_MacBearerRouting_Optional MacBearerRouting;
	struct TimingInfo_Type TimingInfo;
	struct IndicationStatus_Type Status;
};

struct MacBearerRouting_Type_NR_ReqAspCommonPart_Type_MacBearerRouting_Optional {
	bool d;
	struct MacBearerRouting_Type v;
};

struct NR_ReqAspCommonPart_Type {
	NR_CellId_Type CellId;
	struct NR_RoutingInfo_Type RoutingInfo;
	struct RlcBearerRouting_Type RlcBearerRouting;
	struct MacBearerRouting_Type_NR_ReqAspCommonPart_Type_MacBearerRouting_Optional MacBearerRouting;
	struct TimingInfo_Type TimingInfo;
	struct ReqAspControlInfo_Type ControlInfo;
};

struct NR_CnfAspCommonPart_Type {
	NR_CellId_Type CellId;
	struct NR_RoutingInfo_Type RoutingInfo;
	struct TimingInfo_Type TimingInfo;
	struct ConfirmationResult_Type Result;
};

enum NR_PdcpCountFormat_Type {
	NR_PdcpCount_Srb = 0,
	NR_PdcpCount_DrbSQN12 = 1,
	NR_PdcpCount_DrbSQN18 = 2,
};

typedef enum NR_PdcpCountFormat_Type NR_PdcpCountFormat_Type;

struct NR_PdcpCount_Type {
	NR_PdcpCountFormat_Type Format;
	PdcpCountValue_Type Value;
};

struct NR_PdcpCount_Type_UL_Optional {
	bool d;
	struct NR_PdcpCount_Type v;
};

struct NR_PdcpCount_Type_DL_Optional {
	bool d;
	struct NR_PdcpCount_Type v;
};

struct NR_PdcpCountInfo_Type {
	struct NR_RadioBearerId_Type RadioBearerId;
	struct NR_PdcpCount_Type_UL_Optional UL;
	struct NR_PdcpCount_Type_DL_Optional DL;
};

struct NR_PdcpCountInfo_Type_NR_PdcpCountInfoList_Type_Dynamic {
	size_t d;
	struct NR_PdcpCountInfo_Type* v;
};

typedef struct NR_PdcpCountInfo_Type_NR_PdcpCountInfoList_Type_Dynamic NR_PdcpCountInfoList_Type;

enum NR_PDCP_CountCnf_Type_Sel {
	NR_PDCP_CountCnf_Type_UNBOUND_VALUE = 0,
	NR_PDCP_CountCnf_Type_Get = 1,
	NR_PDCP_CountCnf_Type_Set = 2,
};

union NR_PDCP_CountCnf_Type_Value {
	NR_PdcpCountInfoList_Type Get;
	Null_Type Set;
};

struct NR_PDCP_CountCnf_Type {
	enum NR_PDCP_CountCnf_Type_Sel d;
	union NR_PDCP_CountCnf_Type_Value v;
};

struct DeltaValues_Type {
	int32_t DeltaNRf1;
	int32_t DeltaNRf2;
	int32_t DeltaNRf3;
	int32_t DeltaNRf4;
};

struct UE_NR_DeltaValues_Type {
	struct DeltaValues_Type DeltaPrimaryBand;
	struct DeltaValues_Type DeltaSecondaryBand;
};

enum NR_SearchSpaceType_Type {
	NR_SearchSpaceType_cssType0 = 0,
	NR_SearchSpaceType_cssType0A = 1,
	NR_SearchSpaceType_cssType1 = 2,
	NR_SearchSpaceType_cssType2 = 3,
	NR_SearchSpaceType_cssType3 = 4,
	NR_SearchSpaceType_ussDL = 5,
	NR_SearchSpaceType_ussUL = 6,
};

typedef enum NR_SearchSpaceType_Type NR_SearchSpaceType_Type;

enum NR_Attenuation_Type_Sel {
	NR_Attenuation_Type_UNBOUND_VALUE = 0,
	NR_Attenuation_Type_Value = 1,
	NR_Attenuation_Type_Off = 2,
};

union NR_Attenuation_Type_Value {
	uint8_t Value;
	Null_Type Off;
};

struct NR_Attenuation_Type {
	enum NR_Attenuation_Type_Sel d;
	union NR_Attenuation_Type_Value v;
};

enum NR_ASN1_TDD_UL_DL_SlotConfig_Type_Sel {
	NR_ASN1_TDD_UL_DL_SlotConfig_Type_UNBOUND_VALUE = 0,
	NR_ASN1_TDD_UL_DL_SlotConfig_Type_R15 = 1,
};

union NR_ASN1_TDD_UL_DL_SlotConfig_Type_Value {
	struct SQN_NR_TDD_UL_DL_SlotConfig R15;
};

struct NR_ASN1_TDD_UL_DL_SlotConfig_Type {
	enum NR_ASN1_TDD_UL_DL_SlotConfig_Type_Sel d;
	union NR_ASN1_TDD_UL_DL_SlotConfig_Type_Value v;
};

enum NR_ASN1_TDD_UL_DL_ConfigCommon_Type_Sel {
	NR_ASN1_TDD_UL_DL_ConfigCommon_Type_UNBOUND_VALUE = 0,
	NR_ASN1_TDD_UL_DL_ConfigCommon_Type_R15 = 1,
};

union NR_ASN1_TDD_UL_DL_ConfigCommon_Type_Value {
	struct SQN_NR_TDD_UL_DL_ConfigCommon R15;
};

struct NR_ASN1_TDD_UL_DL_ConfigCommon_Type {
	enum NR_ASN1_TDD_UL_DL_ConfigCommon_Type_Sel d;
	union NR_ASN1_TDD_UL_DL_ConfigCommon_Type_Value v;
};

enum NR_ASN1_FrequencyInfoDL_Type_Sel {
	NR_ASN1_FrequencyInfoDL_Type_UNBOUND_VALUE = 0,
	NR_ASN1_FrequencyInfoDL_Type_R15 = 1,
};

union NR_ASN1_FrequencyInfoDL_Type_Value {
	struct SQN_NR_FrequencyInfoDL R15;
};

struct NR_ASN1_FrequencyInfoDL_Type {
	enum NR_ASN1_FrequencyInfoDL_Type_Sel d;
	union NR_ASN1_FrequencyInfoDL_Type_Value v;
};

enum NR_ASN1_PDSCH_ServingCellConfig_Type_Sel {
	NR_ASN1_PDSCH_ServingCellConfig_Type_UNBOUND_VALUE = 0,
	NR_ASN1_PDSCH_ServingCellConfig_Type_R15 = 1,
};

union NR_ASN1_PDSCH_ServingCellConfig_Type_Value {
	struct SQN_NR_PDSCH_ServingCellConfig R15;
};

struct NR_ASN1_PDSCH_ServingCellConfig_Type {
	enum NR_ASN1_PDSCH_ServingCellConfig_Type_Sel d;
	union NR_ASN1_PDSCH_ServingCellConfig_Type_Value v;
};

enum NR_ASN1_RateMatchPattern_Type_Sel {
	NR_ASN1_RateMatchPattern_Type_UNBOUND_VALUE = 0,
	NR_ASN1_RateMatchPattern_Type_R15 = 1,
};

union NR_ASN1_RateMatchPattern_Type_Value {
	struct SQN_NR_RateMatchPattern R15;
};

struct NR_ASN1_RateMatchPattern_Type {
	enum NR_ASN1_RateMatchPattern_Type_Sel d;
	union NR_ASN1_RateMatchPattern_Type_Value v;
};

enum NR_ASN1_RateMatchPatternLTE_CRS_Type_Sel {
	NR_ASN1_RateMatchPatternLTE_CRS_Type_UNBOUND_VALUE = 0,
	NR_ASN1_RateMatchPatternLTE_CRS_Type_R15 = 1,
};

union NR_ASN1_RateMatchPatternLTE_CRS_Type_Value {
	struct SQN_NR_RateMatchPatternLTE_CRS R15;
};

struct NR_ASN1_RateMatchPatternLTE_CRS_Type {
	enum NR_ASN1_RateMatchPatternLTE_CRS_Type_Sel d;
	union NR_ASN1_RateMatchPatternLTE_CRS_Type_Value v;
};

enum NR_ASN1_BWP_Type_Sel {
	NR_ASN1_BWP_Type_UNBOUND_VALUE = 0,
	NR_ASN1_BWP_Type_R15 = 1,
};

union NR_ASN1_BWP_Type_Value {
	struct SQN_NR_BWP R15;
};

struct NR_ASN1_BWP_Type {
	enum NR_ASN1_BWP_Type_Sel d;
	union NR_ASN1_BWP_Type_Value v;
};

struct SQN_NR_SearchSpaceExt_r16_R16_Optional {
	bool d;
	struct SQN_NR_SearchSpaceExt_r16 v;
};

struct NR_ASN1_SearchSpace_Type {
	struct SQN_NR_SearchSpace R15;
	struct SQN_NR_SearchSpaceExt_r16_R16_Optional R16;
};

enum NR_ASN1_ControlResourceSet_Type_Sel {
	NR_ASN1_ControlResourceSet_Type_UNBOUND_VALUE = 0,
	NR_ASN1_ControlResourceSet_Type_R15 = 1,
};

union NR_ASN1_ControlResourceSet_Type_Value {
	struct SQN_NR_ControlResourceSet R15;
};

struct NR_ASN1_ControlResourceSet_Type {
	enum NR_ASN1_ControlResourceSet_Type_Sel d;
	union NR_ASN1_ControlResourceSet_Type_Value v;
};

enum NR_ASN1_PDSCH_ConfigCommon_Type_Sel {
	NR_ASN1_PDSCH_ConfigCommon_Type_UNBOUND_VALUE = 0,
	NR_ASN1_PDSCH_ConfigCommon_Type_R15 = 1,
};

union NR_ASN1_PDSCH_ConfigCommon_Type_Value {
	struct SQN_NR_PDSCH_ConfigCommon R15;
};

struct NR_ASN1_PDSCH_ConfigCommon_Type {
	enum NR_ASN1_PDSCH_ConfigCommon_Type_Sel d;
	union NR_ASN1_PDSCH_ConfigCommon_Type_Value v;
};

enum NR_ASN1_PDSCH_Config_Type_Sel {
	NR_ASN1_PDSCH_Config_Type_UNBOUND_VALUE = 0,
	NR_ASN1_PDSCH_Config_Type_R15 = 1,
};

union NR_ASN1_PDSCH_Config_Type_Value {
	struct SQN_NR_PDSCH_Config R15;
};

struct NR_ASN1_PDSCH_Config_Type {
	enum NR_ASN1_PDSCH_Config_Type_Sel d;
	union NR_ASN1_PDSCH_Config_Type_Value v;
};

enum NR_ASN1_SPS_Config_Type_Sel {
	NR_ASN1_SPS_Config_Type_UNBOUND_VALUE = 0,
	NR_ASN1_SPS_Config_Type_R15 = 1,
};

union NR_ASN1_SPS_Config_Type_Value {
	struct SQN_NR_SPS_Config R15;
};

struct NR_ASN1_SPS_Config_Type {
	enum NR_ASN1_SPS_Config_Type_Sel d;
	union NR_ASN1_SPS_Config_Type_Value v;
};

enum NR_ASN1_FrequencyInfoUL_Type_Sel {
	NR_ASN1_FrequencyInfoUL_Type_UNBOUND_VALUE = 0,
	NR_ASN1_FrequencyInfoUL_Type_R15 = 1,
};

union NR_ASN1_FrequencyInfoUL_Type_Value {
	struct SQN_NR_FrequencyInfoUL R15;
};

struct NR_ASN1_FrequencyInfoUL_Type {
	enum NR_ASN1_FrequencyInfoUL_Type_Sel d;
	union NR_ASN1_FrequencyInfoUL_Type_Value v;
};

enum NR_ASN1_BWP_UplinkCommon_Type_Sel {
	NR_ASN1_BWP_UplinkCommon_Type_UNBOUND_VALUE = 0,
	NR_ASN1_BWP_UplinkCommon_Type_R15 = 1,
};

union NR_ASN1_BWP_UplinkCommon_Type_Value {
	struct SQN_NR_BWP_UplinkCommon R15;
};

struct NR_ASN1_BWP_UplinkCommon_Type {
	enum NR_ASN1_BWP_UplinkCommon_Type_Sel d;
	union NR_ASN1_BWP_UplinkCommon_Type_Value v;
};

enum NR_ASN1_BWP_UplinkDedicated_Type_Sel {
	NR_ASN1_BWP_UplinkDedicated_Type_UNBOUND_VALUE = 0,
	NR_ASN1_BWP_UplinkDedicated_Type_R15 = 1,
};

union NR_ASN1_BWP_UplinkDedicated_Type_Value {
	struct SQN_NR_BWP_UplinkDedicated R15;
};

struct NR_ASN1_BWP_UplinkDedicated_Type {
	enum NR_ASN1_BWP_UplinkDedicated_Type_Sel d;
	union NR_ASN1_BWP_UplinkDedicated_Type_Value v;
};

enum NR_ASN1_RACH_ConfigDedicated_Type_Sel {
	NR_ASN1_RACH_ConfigDedicated_Type_UNBOUND_VALUE = 0,
	NR_ASN1_RACH_ConfigDedicated_Type_R15 = 1,
};

union NR_ASN1_RACH_ConfigDedicated_Type_Value {
	struct SQN_NR_RACH_ConfigDedicated R15;
};

struct NR_ASN1_RACH_ConfigDedicated_Type {
	enum NR_ASN1_RACH_ConfigDedicated_Type_Sel d;
	union NR_ASN1_RACH_ConfigDedicated_Type_Value v;
};

enum NR_ASN1_SI_RequestConfig_Type_Sel {
	NR_ASN1_SI_RequestConfig_Type_UNBOUND_VALUE = 0,
	NR_ASN1_SI_RequestConfig_Type_R15 = 1,
};

union NR_ASN1_SI_RequestConfig_Type_Value {
	struct SQN_NR_SI_RequestConfig R15;
};

struct NR_ASN1_SI_RequestConfig_Type {
	enum NR_ASN1_SI_RequestConfig_Type_Sel d;
	union NR_ASN1_SI_RequestConfig_Type_Value v;
};

enum NR_ASN1_PUSCH_ServingCellConfig_Type_Sel {
	NR_ASN1_PUSCH_ServingCellConfig_Type_UNBOUND_VALUE = 0,
	NR_ASN1_PUSCH_ServingCellConfig_Type_R15 = 1,
};

union NR_ASN1_PUSCH_ServingCellConfig_Type_Value {
	struct SQN_NR_PUSCH_ServingCellConfig R15;
};

struct NR_ASN1_PUSCH_ServingCellConfig_Type {
	enum NR_ASN1_PUSCH_ServingCellConfig_Type_Sel d;
	union NR_ASN1_PUSCH_ServingCellConfig_Type_Value v;
};

enum NR_ASN1_DRX_Config_Type_Sel {
	NR_ASN1_DRX_Config_Type_UNBOUND_VALUE = 0,
	NR_ASN1_DRX_Config_Type_R15 = 1,
};

union NR_ASN1_DRX_Config_Type_Value {
	struct SQN_NR_DRX_Config R15;
};

struct NR_ASN1_DRX_Config_Type {
	enum NR_ASN1_DRX_Config_Type_Sel d;
	union NR_ASN1_DRX_Config_Type_Value v;
};

enum NR_ASN1_MeasGapConfig_Type_Sel {
	NR_ASN1_MeasGapConfig_Type_UNBOUND_VALUE = 0,
	NR_ASN1_MeasGapConfig_Type_R15 = 1,
};

union NR_ASN1_MeasGapConfig_Type_Value {
	struct SQN_NR_MeasGapConfig R15;
};

struct NR_ASN1_MeasGapConfig_Type {
	enum NR_ASN1_MeasGapConfig_Type_Sel d;
	union NR_ASN1_MeasGapConfig_Type_Value v;
};

enum NR_ASN1_MAC_CellGroupConfig_Type_Sel {
	NR_ASN1_MAC_CellGroupConfig_Type_UNBOUND_VALUE = 0,
	NR_ASN1_MAC_CellGroupConfig_Type_R15 = 1,
};

union NR_ASN1_MAC_CellGroupConfig_Type_Value {
	struct SQN_NR_MAC_CellGroupConfig R15;
};

struct NR_ASN1_MAC_CellGroupConfig_Type {
	enum NR_ASN1_MAC_CellGroupConfig_Type_Sel d;
	union NR_ASN1_MAC_CellGroupConfig_Type_Value v;
};

enum NR_ASN1_PhysicalCellGroupConfig_Type_Sel {
	NR_ASN1_PhysicalCellGroupConfig_Type_UNBOUND_VALUE = 0,
	NR_ASN1_PhysicalCellGroupConfig_Type_R15 = 1,
};

union NR_ASN1_PhysicalCellGroupConfig_Type_Value {
	struct SQN_NR_PhysicalCellGroupConfig R15;
};

struct NR_ASN1_PhysicalCellGroupConfig_Type {
	enum NR_ASN1_PhysicalCellGroupConfig_Type_Sel d;
	union NR_ASN1_PhysicalCellGroupConfig_Type_Value v;
};

enum NR_ASN1_ARFCN_ValueNR_Type_Sel {
	NR_ASN1_ARFCN_ValueNR_Type_UNBOUND_VALUE = 0,
	NR_ASN1_ARFCN_ValueNR_Type_R15 = 1,
};

union NR_ASN1_ARFCN_ValueNR_Type_Value {
	SQN_NR_ARFCN_ValueNR R15;
};

struct NR_ASN1_ARFCN_ValueNR_Type {
	enum NR_ASN1_ARFCN_ValueNR_Type_Sel d;
	union NR_ASN1_ARFCN_ValueNR_Type_Value v;
};

struct SQN_NR_BWP_Id_NR_BWP_Id_List_Type_Dynamic {
	size_t d;
	SQN_NR_BWP_Id* v;
};

typedef struct SQN_NR_BWP_Id_NR_BWP_Id_List_Type_Dynamic NR_BWP_Id_List_Type;

struct Null_Type_ActiveBWP_Optional {
	bool d;
	Null_Type v;
};

struct Null_Type_InitialBWP_Optional {
	bool d;
	Null_Type v;
};

struct Null_Type_InitialBWP_RedCap_Optional {
	bool d;
	Null_Type v;
};

struct NR_AssignedBWPs_Type {
	struct Null_Type_ActiveBWP_Optional ActiveBWP;
	struct Null_Type_InitialBWP_Optional InitialBWP;
	NR_BWP_Id_List_Type DedicatedBWPs;
	struct Null_Type_InitialBWP_RedCap_Optional InitialBWP_RedCap;
};

enum NR_ResourceAllocationType_Type {
	NR_ResourceAllocation_Type0 = 0,
	NR_ResourceAllocation_Type1 = 1,
};

typedef enum NR_ResourceAllocationType_Type NR_ResourceAllocationType_Type;

struct NR_FreqDomainSchedulCommonDL_Type {
	int32_t FirstRbIndex;
	int32_t MaxRbCnt;
};

struct NR_FreqDomainSchedulExplicit_Type {
	int32_t FirstRbIndex;
	int32_t Nprb;
};

enum NR_FreqDomainResourceAssignmentDL_Type_Sel {
	NR_FreqDomainResourceAssignmentDL_Type_UNBOUND_VALUE = 0,
	NR_FreqDomainResourceAssignmentDL_Type_Automatic = 1,
	NR_FreqDomainResourceAssignmentDL_Type_Explicit = 2,
};

union NR_FreqDomainResourceAssignmentDL_Type_Value {
	struct NR_FreqDomainSchedulCommonDL_Type Automatic;
	struct NR_FreqDomainSchedulExplicit_Type Explicit;
};

struct NR_FreqDomainResourceAssignmentDL_Type {
	enum NR_FreqDomainResourceAssignmentDL_Type_Sel d;
	union NR_FreqDomainResourceAssignmentDL_Type_Value v;
};

enum NR_DciCommon_TimeDomainResourceAssignment_Type_Sel {
	NR_DciCommon_TimeDomainResourceAssignment_Type_UNBOUND_VALUE = 0,
	NR_DciCommon_TimeDomainResourceAssignment_Type_Index = 1,
};

union NR_DciCommon_TimeDomainResourceAssignment_Type_Value {
	BIT_STRING Index;
};

struct NR_DciCommon_TimeDomainResourceAssignment_Type {
	enum NR_DciCommon_TimeDomainResourceAssignment_Type_Sel d;
	union NR_DciCommon_TimeDomainResourceAssignment_Type_Value v;
};

enum NR_ModulationSchemePDSCH_Type {
	NR_ModulationSchemePDSCH_qpsk = 0,
	NR_ModulationSchemePDSCH_qam16 = 1,
	NR_ModulationSchemePDSCH_qam64 = 2,
	NR_ModulationSchemePDSCH_qam256 = 3,
};

typedef enum NR_ModulationSchemePDSCH_Type NR_ModulationSchemePDSCH_Type;

typedef uint8_t NR_RedundancyVersion_Type;

struct NR_RedundancyVersion_Type_NR_RedundancyVersionList_Type_Dynamic {
	size_t d;
	NR_RedundancyVersion_Type* v;
};

typedef struct NR_RedundancyVersion_Type_NR_RedundancyVersionList_Type_Dynamic NR_RedundancyVersionList_Type;

struct NR_ModulationSchemePDSCH_Type_TransportBlock2_Optional {
	bool d;
	NR_ModulationSchemePDSCH_Type v;
};

struct NR_RedundancyVersionList_Type_RedundancyVersionList_Optional {
	bool d;
	NR_RedundancyVersionList_Type v;
};

struct NR_TransportBlockSchedulingDL_Automatic_Type {
	NR_ModulationSchemePDSCH_Type TransportBlock1;
	struct NR_ModulationSchemePDSCH_Type_TransportBlock2_Optional TransportBlock2;
	struct NR_RedundancyVersionList_Type_RedundancyVersionList_Optional RedundancyVersionList;
};

enum RetransmissionTiming_Type_Sel {
	RetransmissionTiming_Type_UNBOUND_VALUE = 0,
	RetransmissionTiming_Type_SlotOffset = 1,
	RetransmissionTiming_Type_SubframeOffset = 2,
	RetransmissionTiming_Type_AnyTime = 3,
};

union RetransmissionTiming_Type_Value {
	int32_t SlotOffset;
	int32_t SubframeOffset;
	Null_Type AnyTime;
};

struct RetransmissionTiming_Type {
	enum RetransmissionTiming_Type_Sel d;
	union RetransmissionTiming_Type_Value v;
};

enum TransmissionTimingOffset_Type_Sel {
	TransmissionTimingOffset_Type_UNBOUND_VALUE = 0,
	TransmissionTimingOffset_Type_None = 1,
	TransmissionTimingOffset_Type_Retransmission = 2,
};

union TransmissionTimingOffset_Type_Value {
	Null_Type None;
	struct RetransmissionTiming_Type Retransmission;
};

struct TransmissionTimingOffset_Type {
	enum TransmissionTimingOffset_Type_Sel d;
	union TransmissionTimingOffset_Type_Value v;
};

typedef uint8_t NR_ImcsValue_Type;

struct NR_TransportBlockSingleTransmission_Type {
	struct TransmissionTimingOffset_Type TimingOffset;
	NR_ImcsValue_Type ImcsValue;
	NR_RedundancyVersion_Type RedundancyVersion;
	bool ToggleNDI;
};

struct NR_TransportBlockSingleTransmission_Type_NR_TransportBlockRetransmissionList_Type_Dynamic {
	size_t d;
	struct NR_TransportBlockSingleTransmission_Type* v;
};

typedef struct NR_TransportBlockSingleTransmission_Type_NR_TransportBlockRetransmissionList_Type_Dynamic NR_TransportBlockRetransmissionList_Type;

struct NR_TransportBlockRetransmissionList_Type_TransportBlock2_Optional {
	bool d;
	NR_TransportBlockRetransmissionList_Type v;
};

struct NR_TransportBlockSchedulingDL_Explicit_Type {
	NR_TransportBlockRetransmissionList_Type TransportBlock1;
	struct NR_TransportBlockRetransmissionList_Type_TransportBlock2_Optional TransportBlock2;
};

enum NR_TransportBlockSchedulingDL_Type_Sel {
	NR_TransportBlockSchedulingDL_Type_UNBOUND_VALUE = 0,
	NR_TransportBlockSchedulingDL_Type_Automatic = 1,
	NR_TransportBlockSchedulingDL_Type_Explicit = 2,
};

union NR_TransportBlockSchedulingDL_Type_Value {
	struct NR_TransportBlockSchedulingDL_Automatic_Type Automatic;
	struct NR_TransportBlockSchedulingDL_Explicit_Type Explicit;
};

struct NR_TransportBlockSchedulingDL_Type {
	enum NR_TransportBlockSchedulingDL_Type_Sel d;
	union NR_TransportBlockSchedulingDL_Type_Value v;
};

typedef int32_t NR_HarqProcessId_Type;

struct NR_HarqProcessId_Type_NR_HarqProcessList_Type_Dynamic {
	size_t d;
	NR_HarqProcessId_Type* v;
};

typedef struct NR_HarqProcessId_Type_NR_HarqProcessList_Type_Dynamic NR_HarqProcessList_Type;

enum NR_HarqProcessConfig_Type_Sel {
	NR_HarqProcessConfig_Type_UNBOUND_VALUE = 0,
	NR_HarqProcessConfig_Type_None = 1,
	NR_HarqProcessConfig_Type_Broadcast = 2,
	NR_HarqProcessConfig_Type_AnyProcess = 3,
	NR_HarqProcessConfig_Type_SpecificSubset = 4,
};

union NR_HarqProcessConfig_Type_Value {
	Null_Type None;
	Null_Type Broadcast;
	Null_Type AnyProcess;
	NR_HarqProcessList_Type SpecificSubset;
};

struct NR_HarqProcessConfig_Type {
	enum NR_HarqProcessConfig_Type_Sel d;
	union NR_HarqProcessConfig_Type_Value v;
};

struct NR_ResourceAllocationType_Type_NR_DciFormat_1_X_ResourceAssignment_Type_ResourceAllocationType_Optional {
	bool d;
	NR_ResourceAllocationType_Type v;
};

struct NR_FreqDomainResourceAssignmentDL_Type_FreqDomain_Optional {
	bool d;
	struct NR_FreqDomainResourceAssignmentDL_Type v;
};

struct NR_DciCommon_TimeDomainResourceAssignment_Type_NR_DciFormat_1_X_ResourceAssignment_Type_TimeDomain_Optional {
	bool d;
	struct NR_DciCommon_TimeDomainResourceAssignment_Type v;
};

struct NR_TransportBlockSchedulingDL_Type_TransportBlockScheduling_Optional {
	bool d;
	struct NR_TransportBlockSchedulingDL_Type v;
};

struct NR_HarqProcessConfig_Type_NR_DciFormat_1_X_ResourceAssignment_Type_HarqProcessConfig_Optional {
	bool d;
	struct NR_HarqProcessConfig_Type v;
};

struct NR_DciFormat_1_X_ResourceAssignment_Type {
	struct NR_ResourceAllocationType_Type_NR_DciFormat_1_X_ResourceAssignment_Type_ResourceAllocationType_Optional ResourceAllocationType;
	struct NR_FreqDomainResourceAssignmentDL_Type_FreqDomain_Optional FreqDomain;
	struct NR_DciCommon_TimeDomainResourceAssignment_Type_NR_DciFormat_1_X_ResourceAssignment_Type_TimeDomain_Optional TimeDomain;
	struct NR_TransportBlockSchedulingDL_Type_TransportBlockScheduling_Optional TransportBlockScheduling;
	struct NR_HarqProcessConfig_Type_NR_DciFormat_1_X_ResourceAssignment_Type_HarqProcessConfig_Optional HarqProcessConfig;
};

enum NR_DciCommon_VrbPrbMapping_Type_Sel {
	NR_DciCommon_VrbPrbMapping_Type_UNBOUND_VALUE = 0,
	NR_DciCommon_VrbPrbMapping_Type_None = 1,
	NR_DciCommon_VrbPrbMapping_Type_Index = 2,
};

union NR_DciCommon_VrbPrbMapping_Type_Value {
	Null_Type None;
	B1_Type Index;
};

struct NR_DciCommon_VrbPrbMapping_Type {
	enum NR_DciCommon_VrbPrbMapping_Type_Sel d;
	union NR_DciCommon_VrbPrbMapping_Type_Value v;
};

enum DAI_B2_Type_Sel {
	DAI_B2_Type_UNBOUND_VALUE = 0,
	DAI_B2_Type_Index = 1,
	DAI_B2_Type_Automatic = 2,
};

union DAI_B2_Type_Value {
	B2_Type Index;
	Null_Type Automatic;
};

struct DAI_B2_Type {
	enum DAI_B2_Type_Sel d;
	union DAI_B2_Type_Value v;
};

enum NR_DciFormat_1_0_DAI_Type_Sel {
	NR_DciFormat_1_0_DAI_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_1_0_DAI_Type_Index = 1,
};

union NR_DciFormat_1_0_DAI_Type_Value {
	struct DAI_B2_Type Index;
};

struct NR_DciFormat_1_0_DAI_Type {
	enum NR_DciFormat_1_0_DAI_Type_Sel d;
	union NR_DciFormat_1_0_DAI_Type_Value v;
};

enum NR_DciCommon_TpcCommand_Type_Sel {
	NR_DciCommon_TpcCommand_Type_UNBOUND_VALUE = 0,
	NR_DciCommon_TpcCommand_Type_Value = 1,
};

union NR_DciCommon_TpcCommand_Type_Value {
	B2_Type Value;
};

struct NR_DciCommon_TpcCommand_Type {
	enum NR_DciCommon_TpcCommand_Type_Sel d;
	union NR_DciCommon_TpcCommand_Type_Value v;
};

enum NR_DciFormat_1_X_PucchResourceIndicator_Type_Sel {
	NR_DciFormat_1_X_PucchResourceIndicator_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_1_X_PucchResourceIndicator_Type_Value = 1,
};

union NR_DciFormat_1_X_PucchResourceIndicator_Type_Value {
	B3_Type Value;
};

struct NR_DciFormat_1_X_PucchResourceIndicator_Type {
	enum NR_DciFormat_1_X_PucchResourceIndicator_Type_Sel d;
	union NR_DciFormat_1_X_PucchResourceIndicator_Type_Value v;
};

enum NR_DciFormat_1_X_PdschHarqTimingIndicator_Type_Sel {
	NR_DciFormat_1_X_PdschHarqTimingIndicator_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_1_X_PdschHarqTimingIndicator_Type_Value = 1,
};

union NR_DciFormat_1_X_PdschHarqTimingIndicator_Type_Value {
	BIT_STRING Value;
};

struct NR_DciFormat_1_X_PdschHarqTimingIndicator_Type {
	enum NR_DciFormat_1_X_PdschHarqTimingIndicator_Type_Sel d;
	union NR_DciFormat_1_X_PdschHarqTimingIndicator_Type_Value v;
};

enum NR_DciFormat_X_0_ChannelAccessCPext_Type_Sel {
	NR_DciFormat_X_0_ChannelAccessCPext_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_X_0_ChannelAccessCPext_Type_None = 1,
	NR_DciFormat_X_0_ChannelAccessCPext_Type_Value = 2,
};

union NR_DciFormat_X_0_ChannelAccessCPext_Type_Value {
	Null_Type None;
	B2_Type Value;
};

struct NR_DciFormat_X_0_ChannelAccessCPext_Type {
	enum NR_DciFormat_X_0_ChannelAccessCPext_Type_Sel d;
	union NR_DciFormat_X_0_ChannelAccessCPext_Type_Value v;
};

struct NR_DciFormat_1_0_DAI_Type_DAI_Optional {
	bool d;
	struct NR_DciFormat_1_0_DAI_Type v;
};

struct NR_DciCommon_TpcCommand_Type_NR_DciFormat_1_0_SpecificInfo_Type_TpcCommandPucch_Optional {
	bool d;
	struct NR_DciCommon_TpcCommand_Type v;
};

struct NR_DciFormat_1_X_PucchResourceIndicator_Type_NR_DciFormat_1_0_SpecificInfo_Type_PucchResourceIndicator_Optional {
	bool d;
	struct NR_DciFormat_1_X_PucchResourceIndicator_Type v;
};

struct NR_DciFormat_1_X_PdschHarqTimingIndicator_Type_NR_DciFormat_1_0_SpecificInfo_Type_PdschHarqTimingIndicator_Optional {
	bool d;
	struct NR_DciFormat_1_X_PdschHarqTimingIndicator_Type v;
};

struct NR_DciFormat_X_0_ChannelAccessCPext_Type_NR_DciFormat_1_0_SpecificInfo_Type_ChannelAccessCPext_Optional {
	bool d;
	struct NR_DciFormat_X_0_ChannelAccessCPext_Type v;
};

struct NR_DciFormat_1_0_SpecificInfo_Type {
	struct NR_DciFormat_1_0_DAI_Type_DAI_Optional DAI;
	struct NR_DciCommon_TpcCommand_Type_NR_DciFormat_1_0_SpecificInfo_Type_TpcCommandPucch_Optional TpcCommandPucch;
	struct NR_DciFormat_1_X_PucchResourceIndicator_Type_NR_DciFormat_1_0_SpecificInfo_Type_PucchResourceIndicator_Optional PucchResourceIndicator;
	struct NR_DciFormat_1_X_PdschHarqTimingIndicator_Type_NR_DciFormat_1_0_SpecificInfo_Type_PdschHarqTimingIndicator_Optional PdschHarqTimingIndicator;
	struct NR_DciFormat_X_0_ChannelAccessCPext_Type_NR_DciFormat_1_0_SpecificInfo_Type_ChannelAccessCPext_Optional ChannelAccessCPext;
};

struct B2_Type_NR_DciFormat_1_0_P_RNTI_SpecificInfo_Type_ShortMessageIndicator_Optional {
	bool d;
	B2_Type v;
};

struct B8_Type_NR_DciFormat_1_0_P_RNTI_SpecificInfo_Type_ShortMessages_Optional {
	bool d;
	B8_Type v;
};

struct B2_Type_NR_DciFormat_1_0_P_RNTI_SpecificInfo_Type_TbScaling_Optional {
	bool d;
	B2_Type v;
};

struct NR_DciFormat_1_0_P_RNTI_SpecificInfo_Type {
	struct B2_Type_NR_DciFormat_1_0_P_RNTI_SpecificInfo_Type_ShortMessageIndicator_Optional ShortMessageIndicator;
	struct B8_Type_NR_DciFormat_1_0_P_RNTI_SpecificInfo_Type_ShortMessages_Optional ShortMessages;
	struct B2_Type_NR_DciFormat_1_0_P_RNTI_SpecificInfo_Type_TbScaling_Optional TbScaling;
};

struct B1_Type_NR_DciFormat_1_0_SI_RNTI_SpecificInfo_Type_SystemInfoIndicator_Optional {
	bool d;
	B1_Type v;
};

struct NR_DciFormat_1_0_SI_RNTI_SpecificInfo_Type {
	struct B1_Type_NR_DciFormat_1_0_SI_RNTI_SpecificInfo_Type_SystemInfoIndicator_Optional SystemInfoIndicator;
};

enum NR_DciFormat_1_0_LSBsOfSFN_Type_Sel {
	NR_DciFormat_1_0_LSBsOfSFN_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_1_0_LSBsOfSFN_Type_None = 1,
	NR_DciFormat_1_0_LSBsOfSFN_Type_Automatic = 2,
};

union NR_DciFormat_1_0_LSBsOfSFN_Type_Value {
	Null_Type None;
	Null_Type Automatic;
};

struct NR_DciFormat_1_0_LSBsOfSFN_Type {
	enum NR_DciFormat_1_0_LSBsOfSFN_Type_Sel d;
	union NR_DciFormat_1_0_LSBsOfSFN_Type_Value v;
};

struct B2_Type_NR_DciFormat_1_0_RA_RNTI_SpecificInfo_Type_TbScaling_Optional {
	bool d;
	B2_Type v;
};

struct NR_DciFormat_1_0_LSBsOfSFN_Type_LSBsOfSFN_Optional {
	bool d;
	struct NR_DciFormat_1_0_LSBsOfSFN_Type v;
};

struct NR_DciFormat_1_0_RA_RNTI_SpecificInfo_Type {
	struct B2_Type_NR_DciFormat_1_0_RA_RNTI_SpecificInfo_Type_TbScaling_Optional TbScaling;
	struct NR_DciFormat_1_0_LSBsOfSFN_Type_LSBsOfSFN_Optional LSBsOfSFN;
};

enum NR_DciCommon_CarrierIndicator_Type_Sel {
	NR_DciCommon_CarrierIndicator_Type_UNBOUND_VALUE = 0,
	NR_DciCommon_CarrierIndicator_Type_None = 1,
	NR_DciCommon_CarrierIndicator_Type_CellIndex = 2,
};

union NR_DciCommon_CarrierIndicator_Type_Value {
	Null_Type None;
	B3_Type CellIndex;
};

struct NR_DciCommon_CarrierIndicator_Type {
	enum NR_DciCommon_CarrierIndicator_Type_Sel d;
	union NR_DciCommon_CarrierIndicator_Type_Value v;
};

enum NR_DciCommon_BWPIndicator_Type_Sel {
	NR_DciCommon_BWPIndicator_Type_UNBOUND_VALUE = 0,
	NR_DciCommon_BWPIndicator_Type_Index = 1,
};

union NR_DciCommon_BWPIndicator_Type_Value {
	BIT_STRING Index;
};

struct NR_DciCommon_BWPIndicator_Type {
	enum NR_DciCommon_BWPIndicator_Type_Sel d;
	union NR_DciCommon_BWPIndicator_Type_Value v;
};

enum NR_DciFormat_1_1_PrbBundlingSizeIndicator_Type_Sel {
	NR_DciFormat_1_1_PrbBundlingSizeIndicator_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_1_1_PrbBundlingSizeIndicator_Type_None = 1,
	NR_DciFormat_1_1_PrbBundlingSizeIndicator_Type_Dynamic = 2,
};

union NR_DciFormat_1_1_PrbBundlingSizeIndicator_Type_Value {
	Null_Type None;
	B1_Type Dynamic;
};

struct NR_DciFormat_1_1_PrbBundlingSizeIndicator_Type {
	enum NR_DciFormat_1_1_PrbBundlingSizeIndicator_Type_Sel d;
	union NR_DciFormat_1_1_PrbBundlingSizeIndicator_Type_Value v;
};

enum NR_DciFormat_1_1_RateMatchingIndicator_Type_Sel {
	NR_DciFormat_1_1_RateMatchingIndicator_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_1_1_RateMatchingIndicator_Type_Bitmap = 1,
};

union NR_DciFormat_1_1_RateMatchingIndicator_Type_Value {
	BIT_STRING Bitmap;
};

struct NR_DciFormat_1_1_RateMatchingIndicator_Type {
	enum NR_DciFormat_1_1_RateMatchingIndicator_Type_Sel d;
	union NR_DciFormat_1_1_RateMatchingIndicator_Type_Value v;
};

enum NR_DciFormat_1_1_ZP_CSI_RS_Trigger_Type_Sel {
	NR_DciFormat_1_1_ZP_CSI_RS_Trigger_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_1_1_ZP_CSI_RS_Trigger_Type_Index = 1,
};

union NR_DciFormat_1_1_ZP_CSI_RS_Trigger_Type_Value {
	BIT_STRING Index;
};

struct NR_DciFormat_1_1_ZP_CSI_RS_Trigger_Type {
	enum NR_DciFormat_1_1_ZP_CSI_RS_Trigger_Type_Sel d;
	union NR_DciFormat_1_1_ZP_CSI_RS_Trigger_Type_Value v;
};

enum DAI_B1_Type_Sel {
	DAI_B1_Type_UNBOUND_VALUE = 0,
	DAI_B1_Type_Index = 1,
	DAI_B1_Type_Automatic = 2,
};

union DAI_B1_Type_Value {
	B1_Type Index;
	Null_Type Automatic;
};

struct DAI_B1_Type {
	enum DAI_B1_Type_Sel d;
	union DAI_B1_Type_Value v;
};

enum DAI_B4_Type_Sel {
	DAI_B4_Type_UNBOUND_VALUE = 0,
	DAI_B4_Type_Index = 1,
	DAI_B4_Type_Automatic = 2,
};

union DAI_B4_Type_Value {
	B4_Type Index;
	Null_Type Automatic;
};

struct DAI_B4_Type {
	enum DAI_B4_Type_Sel d;
	union DAI_B4_Type_Value v;
};

enum DAI_B6_Type_Sel {
	DAI_B6_Type_UNBOUND_VALUE = 0,
	DAI_B6_Type_Index = 1,
	DAI_B6_Type_Automatic = 2,
};

union DAI_B6_Type_Value {
	B6_Type Index;
	Null_Type Automatic;
};

struct DAI_B6_Type {
	enum DAI_B6_Type_Sel d;
	union DAI_B6_Type_Value v;
};

enum NR_DciFormat_1_1_DAI_Type_Sel {
	NR_DciFormat_1_1_DAI_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_1_1_DAI_Type_None = 1,
	NR_DciFormat_1_1_DAI_Type_TwoBits = 2,
	NR_DciFormat_1_1_DAI_Type_FourBits = 3,
	NR_DciFormat_1_1_DAI_Type_SixBits = 4,
};

union NR_DciFormat_1_1_DAI_Type_Value {
	Null_Type None;
	struct DAI_B2_Type TwoBits;
	struct DAI_B4_Type FourBits;
	struct DAI_B6_Type SixBits;
};

struct NR_DciFormat_1_1_DAI_Type {
	enum NR_DciFormat_1_1_DAI_Type_Sel d;
	union NR_DciFormat_1_1_DAI_Type_Value v;
};

enum NR_DciFormat_1_1_OneShotHarqAckRequest_Type_Sel {
	NR_DciFormat_1_1_OneShotHarqAckRequest_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_1_1_OneShotHarqAckRequest_Type_None = 1,
	NR_DciFormat_1_1_OneShotHarqAckRequest_Type_Value = 2,
};

union NR_DciFormat_1_1_OneShotHarqAckRequest_Type_Value {
	Null_Type None;
	B1_Type Value;
};

struct NR_DciFormat_1_1_OneShotHarqAckRequest_Type {
	enum NR_DciFormat_1_1_OneShotHarqAckRequest_Type_Sel d;
	union NR_DciFormat_1_1_OneShotHarqAckRequest_Type_Value v;
};

enum NR_DciFormat_1_1_PdschGroupIndex_Type_Sel {
	NR_DciFormat_1_1_PdschGroupIndex_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_1_1_PdschGroupIndex_Type_None = 1,
	NR_DciFormat_1_1_PdschGroupIndex_Type_Value = 2,
};

union NR_DciFormat_1_1_PdschGroupIndex_Type_Value {
	Null_Type None;
	B1_Type Value;
};

struct NR_DciFormat_1_1_PdschGroupIndex_Type {
	enum NR_DciFormat_1_1_PdschGroupIndex_Type_Sel d;
	union NR_DciFormat_1_1_PdschGroupIndex_Type_Value v;
};

enum NR_DciFormat_1_1_NewFeedbackIndicator_Type_Sel {
	NR_DciFormat_1_1_NewFeedbackIndicator_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_1_1_NewFeedbackIndicator_Type_None = 1,
	NR_DciFormat_1_1_NewFeedbackIndicator_Type_Value = 2,
};

union NR_DciFormat_1_1_NewFeedbackIndicator_Type_Value {
	Null_Type None;
	BIT_STRING Value;
};

struct NR_DciFormat_1_1_NewFeedbackIndicator_Type {
	enum NR_DciFormat_1_1_NewFeedbackIndicator_Type_Sel d;
	union NR_DciFormat_1_1_NewFeedbackIndicator_Type_Value v;
};

enum NR_DciFormat_1_1_NumberRequestedPdschGroup_Type_Sel {
	NR_DciFormat_1_1_NumberRequestedPdschGroup_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_1_1_NumberRequestedPdschGroup_Type_None = 1,
	NR_DciFormat_1_1_NumberRequestedPdschGroup_Type_Value = 2,
};

union NR_DciFormat_1_1_NumberRequestedPdschGroup_Type_Value {
	Null_Type None;
	B1_Type Value;
};

struct NR_DciFormat_1_1_NumberRequestedPdschGroup_Type {
	enum NR_DciFormat_1_1_NumberRequestedPdschGroup_Type_Sel d;
	union NR_DciFormat_1_1_NumberRequestedPdschGroup_Type_Value v;
};

enum NR_DciFormat_1_1_AntennaPorts_Type_Sel {
	NR_DciFormat_1_1_AntennaPorts_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_1_1_AntennaPorts_Type_Index = 1,
};

union NR_DciFormat_1_1_AntennaPorts_Type_Value {
	BIT_STRING Index;
};

struct NR_DciFormat_1_1_AntennaPorts_Type {
	enum NR_DciFormat_1_1_AntennaPorts_Type_Sel d;
	union NR_DciFormat_1_1_AntennaPorts_Type_Value v;
};

enum NR_DciFormat_1_1_TCI_Type_Sel {
	NR_DciFormat_1_1_TCI_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_1_1_TCI_Type_None = 1,
	NR_DciFormat_1_1_TCI_Type_Value = 2,
};

union NR_DciFormat_1_1_TCI_Type_Value {
	Null_Type None;
	B3_Type Value;
};

struct NR_DciFormat_1_1_TCI_Type {
	enum NR_DciFormat_1_1_TCI_Type_Sel d;
	union NR_DciFormat_1_1_TCI_Type_Value v;
};

enum NR_DciFormat_X_1_SrsRequest_Type_Sel {
	NR_DciFormat_X_1_SrsRequest_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_X_1_SrsRequest_Type_SingleUL = 1,
	NR_DciFormat_X_1_SrsRequest_Type_UL_SUL = 2,
};

union NR_DciFormat_X_1_SrsRequest_Type_Value {
	B2_Type SingleUL;
	B3_Type UL_SUL;
};

struct NR_DciFormat_X_1_SrsRequest_Type {
	enum NR_DciFormat_X_1_SrsRequest_Type_Sel d;
	union NR_DciFormat_X_1_SrsRequest_Type_Value v;
};

enum NR_DciFormat_1_1_CBGTI_Type_Sel {
	NR_DciFormat_1_1_CBGTI_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_1_1_CBGTI_Type_Bitmap = 1,
};

union NR_DciFormat_1_1_CBGTI_Type_Value {
	BIT_STRING Bitmap;
};

struct NR_DciFormat_1_1_CBGTI_Type {
	enum NR_DciFormat_1_1_CBGTI_Type_Sel d;
	union NR_DciFormat_1_1_CBGTI_Type_Value v;
};

enum NR_DciFormat_1_1_CBGFI_Type_Sel {
	NR_DciFormat_1_1_CBGFI_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_1_1_CBGFI_Type_None = 1,
	NR_DciFormat_1_1_CBGFI_Type_Flag = 2,
};

union NR_DciFormat_1_1_CBGFI_Type_Value {
	Null_Type None;
	B1_Type Flag;
};

struct NR_DciFormat_1_1_CBGFI_Type {
	enum NR_DciFormat_1_1_CBGFI_Type_Sel d;
	union NR_DciFormat_1_1_CBGFI_Type_Value v;
};

enum NR_DciFormat_X_1_DmrsSequenceInit_Type_Sel {
	NR_DciFormat_X_1_DmrsSequenceInit_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_X_1_DmrsSequenceInit_Type_None = 1,
	NR_DciFormat_X_1_DmrsSequenceInit_Type_Value = 2,
};

union NR_DciFormat_X_1_DmrsSequenceInit_Type_Value {
	Null_Type None;
	B1_Type Value;
};

struct NR_DciFormat_X_1_DmrsSequenceInit_Type {
	enum NR_DciFormat_X_1_DmrsSequenceInit_Type_Sel d;
	union NR_DciFormat_X_1_DmrsSequenceInit_Type_Value v;
};

enum NR_DciFormat_X_1_PriorityIndicator_Type_Sel {
	NR_DciFormat_X_1_PriorityIndicator_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_X_1_PriorityIndicator_Type_None = 1,
	NR_DciFormat_X_1_PriorityIndicator_Type_Value = 2,
};

union NR_DciFormat_X_1_PriorityIndicator_Type_Value {
	Null_Type None;
	B1_Type Value;
};

struct NR_DciFormat_X_1_PriorityIndicator_Type {
	enum NR_DciFormat_X_1_PriorityIndicator_Type_Sel d;
	union NR_DciFormat_X_1_PriorityIndicator_Type_Value v;
};

enum NR_DciFormat_1_1_ChannelAccessCPext_Type_Sel {
	NR_DciFormat_1_1_ChannelAccessCPext_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_1_1_ChannelAccessCPext_Type_None = 1,
	NR_DciFormat_1_1_ChannelAccessCPext_Type_Value = 2,
};

union NR_DciFormat_1_1_ChannelAccessCPext_Type_Value {
	Null_Type None;
	BIT_STRING Value;
};

struct NR_DciFormat_1_1_ChannelAccessCPext_Type {
	enum NR_DciFormat_1_1_ChannelAccessCPext_Type_Sel d;
	union NR_DciFormat_1_1_ChannelAccessCPext_Type_Value v;
};

enum NR_DciFormat_X_1_MinimumApplicableSchedulingOffset_Type_Sel {
	NR_DciFormat_X_1_MinimumApplicableSchedulingOffset_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_X_1_MinimumApplicableSchedulingOffset_Type_None = 1,
	NR_DciFormat_X_1_MinimumApplicableSchedulingOffset_Type_Value = 2,
};

union NR_DciFormat_X_1_MinimumApplicableSchedulingOffset_Type_Value {
	Null_Type None;
	B1_Type Value;
};

struct NR_DciFormat_X_1_MinimumApplicableSchedulingOffset_Type {
	enum NR_DciFormat_X_1_MinimumApplicableSchedulingOffset_Type_Sel d;
	union NR_DciFormat_X_1_MinimumApplicableSchedulingOffset_Type_Value v;
};

enum NR_DciFormat_X_1_SCellDormancyIndication_Type_Sel {
	NR_DciFormat_X_1_SCellDormancyIndication_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_X_1_SCellDormancyIndication_Type_None = 1,
	NR_DciFormat_X_1_SCellDormancyIndication_Type_Value = 2,
};

union NR_DciFormat_X_1_SCellDormancyIndication_Type_Value {
	Null_Type None;
	BIT_STRING Value;
};

struct NR_DciFormat_X_1_SCellDormancyIndication_Type {
	enum NR_DciFormat_X_1_SCellDormancyIndication_Type_Sel d;
	union NR_DciFormat_X_1_SCellDormancyIndication_Type_Value v;
};

struct NR_DciCommon_CarrierIndicator_Type_NR_DciFormat_1_1_SpecificInfo_Type_CarrierIndicator_Optional {
	bool d;
	struct NR_DciCommon_CarrierIndicator_Type v;
};

struct NR_DciCommon_BWPIndicator_Type_NR_DciFormat_1_1_SpecificInfo_Type_BWPIndicator_Optional {
	bool d;
	struct NR_DciCommon_BWPIndicator_Type v;
};

struct NR_DciFormat_1_1_PrbBundlingSizeIndicator_Type_PrbBundlingSizeIndicator_Optional {
	bool d;
	struct NR_DciFormat_1_1_PrbBundlingSizeIndicator_Type v;
};

struct NR_DciFormat_1_1_RateMatchingIndicator_Type_RateMatchingIndicator_Optional {
	bool d;
	struct NR_DciFormat_1_1_RateMatchingIndicator_Type v;
};

struct NR_DciFormat_1_1_ZP_CSI_RS_Trigger_Type_ZP_CSI_RS_Trigger_Optional {
	bool d;
	struct NR_DciFormat_1_1_ZP_CSI_RS_Trigger_Type v;
};

struct NR_DciFormat_1_1_DAI_Type_DAI_Optional {
	bool d;
	struct NR_DciFormat_1_1_DAI_Type v;
};

struct NR_DciCommon_TpcCommand_Type_NR_DciFormat_1_1_SpecificInfo_Type_TpcCommandPucch_Optional {
	bool d;
	struct NR_DciCommon_TpcCommand_Type v;
};

struct NR_DciFormat_1_X_PucchResourceIndicator_Type_NR_DciFormat_1_1_SpecificInfo_Type_PucchResourceIndicator_Optional {
	bool d;
	struct NR_DciFormat_1_X_PucchResourceIndicator_Type v;
};

struct NR_DciFormat_1_X_PdschHarqTimingIndicator_Type_NR_DciFormat_1_1_SpecificInfo_Type_PdschHarqTimingIndicator_Optional {
	bool d;
	struct NR_DciFormat_1_X_PdschHarqTimingIndicator_Type v;
};

struct NR_DciFormat_1_1_OneShotHarqAckRequest_Type_OneShotHarqAckRequest_Optional {
	bool d;
	struct NR_DciFormat_1_1_OneShotHarqAckRequest_Type v;
};

struct NR_DciFormat_1_1_PdschGroupIndex_Type_PdschGroupIndex_Optional {
	bool d;
	struct NR_DciFormat_1_1_PdschGroupIndex_Type v;
};

struct NR_DciFormat_1_1_NewFeedbackIndicator_Type_NewFeedbackIndicator_Optional {
	bool d;
	struct NR_DciFormat_1_1_NewFeedbackIndicator_Type v;
};

struct NR_DciFormat_1_1_NumberRequestedPdschGroup_Type_NumberRequestedPdschGroup_Optional {
	bool d;
	struct NR_DciFormat_1_1_NumberRequestedPdschGroup_Type v;
};

struct NR_DciFormat_1_1_AntennaPorts_Type_AntennaPorts_Optional {
	bool d;
	struct NR_DciFormat_1_1_AntennaPorts_Type v;
};

struct NR_DciFormat_1_1_TCI_Type_TCI_Optional {
	bool d;
	struct NR_DciFormat_1_1_TCI_Type v;
};

struct NR_DciFormat_X_1_SrsRequest_Type_NR_DciFormat_1_1_SpecificInfo_Type_SrsRequest_Optional {
	bool d;
	struct NR_DciFormat_X_1_SrsRequest_Type v;
};

struct NR_DciFormat_1_1_CBGTI_Type_CBGTI_Optional {
	bool d;
	struct NR_DciFormat_1_1_CBGTI_Type v;
};

struct NR_DciFormat_1_1_CBGFI_Type_CBGFI_Optional {
	bool d;
	struct NR_DciFormat_1_1_CBGFI_Type v;
};

struct NR_DciFormat_X_1_DmrsSequenceInit_Type_NR_DciFormat_1_1_SpecificInfo_Type_DmrsSequenceInit_Optional {
	bool d;
	struct NR_DciFormat_X_1_DmrsSequenceInit_Type v;
};

struct NR_DciFormat_X_1_PriorityIndicator_Type_NR_DciFormat_1_1_SpecificInfo_Type_PriorityIndicator_Optional {
	bool d;
	struct NR_DciFormat_X_1_PriorityIndicator_Type v;
};

struct NR_DciFormat_1_1_ChannelAccessCPext_Type_ChannelAccessCPext_Optional {
	bool d;
	struct NR_DciFormat_1_1_ChannelAccessCPext_Type v;
};

struct NR_DciFormat_X_1_MinimumApplicableSchedulingOffset_Type_NR_DciFormat_1_1_SpecificInfo_Type_MinimumApplicableSchedulingOffset_Optional {
	bool d;
	struct NR_DciFormat_X_1_MinimumApplicableSchedulingOffset_Type v;
};

struct NR_DciFormat_X_1_SCellDormancyIndication_Type_NR_DciFormat_1_1_SpecificInfo_Type_SCellDormancyIndication_Optional {
	bool d;
	struct NR_DciFormat_X_1_SCellDormancyIndication_Type v;
};

struct NR_DciFormat_1_1_SpecificInfo_Type {
	struct NR_DciCommon_CarrierIndicator_Type_NR_DciFormat_1_1_SpecificInfo_Type_CarrierIndicator_Optional CarrierIndicator;
	struct NR_DciCommon_BWPIndicator_Type_NR_DciFormat_1_1_SpecificInfo_Type_BWPIndicator_Optional BWPIndicator;
	struct NR_DciFormat_1_1_PrbBundlingSizeIndicator_Type_PrbBundlingSizeIndicator_Optional PrbBundlingSizeIndicator;
	struct NR_DciFormat_1_1_RateMatchingIndicator_Type_RateMatchingIndicator_Optional RateMatchingIndicator;
	struct NR_DciFormat_1_1_ZP_CSI_RS_Trigger_Type_ZP_CSI_RS_Trigger_Optional ZP_CSI_RS_Trigger;
	struct NR_DciFormat_1_1_DAI_Type_DAI_Optional DAI;
	struct NR_DciCommon_TpcCommand_Type_NR_DciFormat_1_1_SpecificInfo_Type_TpcCommandPucch_Optional TpcCommandPucch;
	struct NR_DciFormat_1_X_PucchResourceIndicator_Type_NR_DciFormat_1_1_SpecificInfo_Type_PucchResourceIndicator_Optional PucchResourceIndicator;
	struct NR_DciFormat_1_X_PdschHarqTimingIndicator_Type_NR_DciFormat_1_1_SpecificInfo_Type_PdschHarqTimingIndicator_Optional PdschHarqTimingIndicator;
	struct NR_DciFormat_1_1_OneShotHarqAckRequest_Type_OneShotHarqAckRequest_Optional OneShotHarqAckRequest;
	struct NR_DciFormat_1_1_PdschGroupIndex_Type_PdschGroupIndex_Optional PdschGroupIndex;
	struct NR_DciFormat_1_1_NewFeedbackIndicator_Type_NewFeedbackIndicator_Optional NewFeedbackIndicator;
	struct NR_DciFormat_1_1_NumberRequestedPdschGroup_Type_NumberRequestedPdschGroup_Optional NumberRequestedPdschGroup;
	struct NR_DciFormat_1_1_AntennaPorts_Type_AntennaPorts_Optional AntennaPorts;
	struct NR_DciFormat_1_1_TCI_Type_TCI_Optional TCI;
	struct NR_DciFormat_X_1_SrsRequest_Type_NR_DciFormat_1_1_SpecificInfo_Type_SrsRequest_Optional SrsRequest;
	struct NR_DciFormat_1_1_CBGTI_Type_CBGTI_Optional CBGTI;
	struct NR_DciFormat_1_1_CBGFI_Type_CBGFI_Optional CBGFI;
	struct NR_DciFormat_X_1_DmrsSequenceInit_Type_NR_DciFormat_1_1_SpecificInfo_Type_DmrsSequenceInit_Optional DmrsSequenceInit;
	struct NR_DciFormat_X_1_PriorityIndicator_Type_NR_DciFormat_1_1_SpecificInfo_Type_PriorityIndicator_Optional PriorityIndicator;
	struct NR_DciFormat_1_1_ChannelAccessCPext_Type_ChannelAccessCPext_Optional ChannelAccessCPext;
	struct NR_DciFormat_X_1_MinimumApplicableSchedulingOffset_Type_NR_DciFormat_1_1_SpecificInfo_Type_MinimumApplicableSchedulingOffset_Optional MinimumApplicableSchedulingOffset;
	struct NR_DciFormat_X_1_SCellDormancyIndication_Type_NR_DciFormat_1_1_SpecificInfo_Type_SCellDormancyIndication_Optional SCellDormancyIndication;
};

enum NR_DciFormat_1_X_SpecificInfo_Type_Sel {
	NR_DciFormat_1_X_SpecificInfo_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_1_X_SpecificInfo_Type_Format_1_0 = 1,
	NR_DciFormat_1_X_SpecificInfo_Type_Format_1_0_P_RNTI = 2,
	NR_DciFormat_1_X_SpecificInfo_Type_Format_1_0_SI_RNTI = 3,
	NR_DciFormat_1_X_SpecificInfo_Type_Format_1_0_RA_RNTI = 4,
	NR_DciFormat_1_X_SpecificInfo_Type_Format_1_1 = 5,
};

union NR_DciFormat_1_X_SpecificInfo_Type_Value {
	struct NR_DciFormat_1_0_SpecificInfo_Type Format_1_0;
	struct NR_DciFormat_1_0_P_RNTI_SpecificInfo_Type Format_1_0_P_RNTI;
	struct NR_DciFormat_1_0_SI_RNTI_SpecificInfo_Type Format_1_0_SI_RNTI;
	struct NR_DciFormat_1_0_RA_RNTI_SpecificInfo_Type Format_1_0_RA_RNTI;
	struct NR_DciFormat_1_1_SpecificInfo_Type Format_1_1;
};

struct NR_DciFormat_1_X_SpecificInfo_Type {
	enum NR_DciFormat_1_X_SpecificInfo_Type_Sel d;
	union NR_DciFormat_1_X_SpecificInfo_Type_Value v;
};

struct NR_DciFormat_1_X_ResourceAssignment_Type_ResoureAssignment_Optional {
	bool d;
	struct NR_DciFormat_1_X_ResourceAssignment_Type v;
};

struct NR_DciCommon_VrbPrbMapping_Type_VrbPrbMapping_Optional {
	bool d;
	struct NR_DciCommon_VrbPrbMapping_Type v;
};

struct NR_DciFormat_1_X_SpecificInfo_Type_Format_Optional {
	bool d;
	struct NR_DciFormat_1_X_SpecificInfo_Type v;
};

struct NR_DciDlInfo_Type {
	struct NR_DciFormat_1_X_ResourceAssignment_Type_ResoureAssignment_Optional ResoureAssignment;
	struct NR_DciCommon_VrbPrbMapping_Type_VrbPrbMapping_Optional VrbPrbMapping;
	struct NR_DciFormat_1_X_SpecificInfo_Type_Format_Optional Format;
};

struct NR_AssignedBWPs_Type_NR_SearchSpaceDlDciAssignment_Type_AssignedBWPs_Optional {
	bool d;
	struct NR_AssignedBWPs_Type v;
};

struct NR_SearchSpaceType_Type_NR_SearchSpaceDlDciAssignment_Type_SearchSpaceType_Optional {
	bool d;
	NR_SearchSpaceType_Type v;
};

struct NR_DciDlInfo_Type_DciInfo_Optional {
	bool d;
	struct NR_DciDlInfo_Type v;
};

struct NR_SearchSpaceDlDciAssignment_Type {
	struct NR_AssignedBWPs_Type_NR_SearchSpaceDlDciAssignment_Type_AssignedBWPs_Optional AssignedBWPs;
	struct NR_SearchSpaceType_Type_NR_SearchSpaceDlDciAssignment_Type_SearchSpaceType_Optional SearchSpaceType;
	struct NR_DciDlInfo_Type_DciInfo_Optional DciInfo;
};

struct NR_ResourceAllocationType_Type_NR_DciFormat_0_X_ResourceAssignment_Type_ResourceAllocationType_Optional {
	bool d;
	NR_ResourceAllocationType_Type v;
};

struct NR_FreqDomainSchedulExplicit_Type_FreqDomain_Optional {
	bool d;
	struct NR_FreqDomainSchedulExplicit_Type v;
};

struct NR_DciCommon_TimeDomainResourceAssignment_Type_NR_DciFormat_0_X_ResourceAssignment_Type_TimeDomain_Optional {
	bool d;
	struct NR_DciCommon_TimeDomainResourceAssignment_Type v;
};

struct NR_TransportBlockRetransmissionList_Type_TransportBlockScheduling_Optional {
	bool d;
	NR_TransportBlockRetransmissionList_Type v;
};

struct NR_HarqProcessConfig_Type_NR_DciFormat_0_X_ResourceAssignment_Type_HarqProcessConfig_Optional {
	bool d;
	struct NR_HarqProcessConfig_Type v;
};

struct NR_DciFormat_0_X_ResourceAssignment_Type {
	struct NR_ResourceAllocationType_Type_NR_DciFormat_0_X_ResourceAssignment_Type_ResourceAllocationType_Optional ResourceAllocationType;
	struct NR_FreqDomainSchedulExplicit_Type_FreqDomain_Optional FreqDomain;
	struct NR_DciCommon_TimeDomainResourceAssignment_Type_NR_DciFormat_0_X_ResourceAssignment_Type_TimeDomain_Optional TimeDomain;
	struct NR_TransportBlockRetransmissionList_Type_TransportBlockScheduling_Optional TransportBlockScheduling;
	struct NR_HarqProcessConfig_Type_NR_DciFormat_0_X_ResourceAssignment_Type_HarqProcessConfig_Optional HarqProcessConfig;
};

enum NR_DciFormat_0_X_PuschHoppingCtrl_Type_Sel {
	NR_DciFormat_0_X_PuschHoppingCtrl_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_0_X_PuschHoppingCtrl_Type_None = 1,
	NR_DciFormat_0_X_PuschHoppingCtrl_Type_Flag = 2,
};

union NR_DciFormat_0_X_PuschHoppingCtrl_Type_Value {
	Null_Type None;
	B1_Type Flag;
};

struct NR_DciFormat_0_X_PuschHoppingCtrl_Type {
	enum NR_DciFormat_0_X_PuschHoppingCtrl_Type_Sel d;
	union NR_DciFormat_0_X_PuschHoppingCtrl_Type_Value v;
};

enum NR_DciCommon_UL_SUL_Indicator_Type_Sel {
	NR_DciCommon_UL_SUL_Indicator_Type_UNBOUND_VALUE = 0,
	NR_DciCommon_UL_SUL_Indicator_Type_None = 1,
	NR_DciCommon_UL_SUL_Indicator_Type_Value = 2,
};

union NR_DciCommon_UL_SUL_Indicator_Type_Value {
	Null_Type None;
	B1_Type Value;
};

struct NR_DciCommon_UL_SUL_Indicator_Type {
	enum NR_DciCommon_UL_SUL_Indicator_Type_Sel d;
	union NR_DciCommon_UL_SUL_Indicator_Type_Value v;
};

struct NR_DciFormat_X_0_ChannelAccessCPext_Type_NR_DciFormat_0_0_SpecificInfo_Type_ChannelAccessCPext_Optional {
	bool d;
	struct NR_DciFormat_X_0_ChannelAccessCPext_Type v;
};

struct NR_DciFormat_0_0_SpecificInfo_Type {
	struct NR_DciFormat_X_0_ChannelAccessCPext_Type_NR_DciFormat_0_0_SpecificInfo_Type_ChannelAccessCPext_Optional ChannelAccessCPext;
};

enum NR_DciFormat_0_1_DfiFlag_Type_Sel {
	NR_DciFormat_0_1_DfiFlag_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_0_1_DfiFlag_Type_None = 1,
	NR_DciFormat_0_1_DfiFlag_Type_Flag = 2,
};

union NR_DciFormat_0_1_DfiFlag_Type_Value {
	Null_Type None;
	B1_Type Flag;
};

struct NR_DciFormat_0_1_DfiFlag_Type {
	enum NR_DciFormat_0_1_DfiFlag_Type_Sel d;
	union NR_DciFormat_0_1_DfiFlag_Type_Value v;
};

enum NR_DciFormat_0_1_FirstDAI_Type_Sel {
	NR_DciFormat_0_1_FirstDAI_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_0_1_FirstDAI_Type_SemiStaticCodebook = 1,
	NR_DciFormat_0_1_FirstDAI_Type_DynamicCodebook = 2,
	NR_DciFormat_0_1_FirstDAI_Type_EnhancedDynamicCodebook = 3,
};

union NR_DciFormat_0_1_FirstDAI_Type_Value {
	struct DAI_B1_Type SemiStaticCodebook;
	struct DAI_B2_Type DynamicCodebook;
	struct DAI_B4_Type EnhancedDynamicCodebook;
};

struct NR_DciFormat_0_1_FirstDAI_Type {
	enum NR_DciFormat_0_1_FirstDAI_Type_Sel d;
	union NR_DciFormat_0_1_FirstDAI_Type_Value v;
};

enum NR_DciFormat_0_1_SecondDAI_Type_Sel {
	NR_DciFormat_0_1_SecondDAI_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_0_1_SecondDAI_Type_None = 1,
	NR_DciFormat_0_1_SecondDAI_Type_DynamicCodebook = 2,
	NR_DciFormat_0_1_SecondDAI_Type_EnhancedDynamicCodebook = 3,
};

union NR_DciFormat_0_1_SecondDAI_Type_Value {
	Null_Type None;
	struct DAI_B2_Type DynamicCodebook;
	struct DAI_B4_Type EnhancedDynamicCodebook;
};

struct NR_DciFormat_0_1_SecondDAI_Type {
	enum NR_DciFormat_0_1_SecondDAI_Type_Sel d;
	union NR_DciFormat_0_1_SecondDAI_Type_Value v;
};

enum NR_DciFormat_0_1_SrsResourceIndicator_Type_Sel {
	NR_DciFormat_0_1_SrsResourceIndicator_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_0_1_SrsResourceIndicator_Type_NonCodeBook = 1,
	NR_DciFormat_0_1_SrsResourceIndicator_Type_CodeBook = 2,
};

union NR_DciFormat_0_1_SrsResourceIndicator_Type_Value {
	BIT_STRING NonCodeBook;
	BIT_STRING CodeBook;
};

struct NR_DciFormat_0_1_SrsResourceIndicator_Type {
	enum NR_DciFormat_0_1_SrsResourceIndicator_Type_Sel d;
	union NR_DciFormat_0_1_SrsResourceIndicator_Type_Value v;
};

enum NR_DciFormat_0_1_PrecodingInfo_Type_Sel {
	NR_DciFormat_0_1_PrecodingInfo_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_0_1_PrecodingInfo_Type_NonCodeBook = 1,
	NR_DciFormat_0_1_PrecodingInfo_Type_CodeBook = 2,
};

union NR_DciFormat_0_1_PrecodingInfo_Type_Value {
	Null_Type NonCodeBook;
	BIT_STRING CodeBook;
};

struct NR_DciFormat_0_1_PrecodingInfo_Type {
	enum NR_DciFormat_0_1_PrecodingInfo_Type_Sel d;
	union NR_DciFormat_0_1_PrecodingInfo_Type_Value v;
};

enum NR_DciFormat_0_1_AntennaPorts_Type_Sel {
	NR_DciFormat_0_1_AntennaPorts_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_0_1_AntennaPorts_Type_Index = 1,
};

union NR_DciFormat_0_1_AntennaPorts_Type_Value {
	BIT_STRING Index;
};

struct NR_DciFormat_0_1_AntennaPorts_Type {
	enum NR_DciFormat_0_1_AntennaPorts_Type_Sel d;
	union NR_DciFormat_0_1_AntennaPorts_Type_Value v;
};

enum NR_DciFormat_0_1_CsiRequest_Type_Sel {
	NR_DciFormat_0_1_CsiRequest_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_0_1_CsiRequest_Type_Index = 1,
};

union NR_DciFormat_0_1_CsiRequest_Type_Value {
	BIT_STRING Index;
};

struct NR_DciFormat_0_1_CsiRequest_Type {
	enum NR_DciFormat_0_1_CsiRequest_Type_Sel d;
	union NR_DciFormat_0_1_CsiRequest_Type_Value v;
};

enum NR_DciFormat_0_1_CBGTI_Type_Sel {
	NR_DciFormat_0_1_CBGTI_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_0_1_CBGTI_Type_Index = 1,
};

union NR_DciFormat_0_1_CBGTI_Type_Value {
	BIT_STRING Index;
};

struct NR_DciFormat_0_1_CBGTI_Type {
	enum NR_DciFormat_0_1_CBGTI_Type_Sel d;
	union NR_DciFormat_0_1_CBGTI_Type_Value v;
};

enum NR_DciFormat_0_1_PtrsDmrsAssociation_Type_Sel {
	NR_DciFormat_0_1_PtrsDmrsAssociation_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_0_1_PtrsDmrsAssociation_Type_None = 1,
	NR_DciFormat_0_1_PtrsDmrsAssociation_Type_Value = 2,
};

union NR_DciFormat_0_1_PtrsDmrsAssociation_Type_Value {
	Null_Type None;
	B2_Type Value;
};

struct NR_DciFormat_0_1_PtrsDmrsAssociation_Type {
	enum NR_DciFormat_0_1_PtrsDmrsAssociation_Type_Sel d;
	union NR_DciFormat_0_1_PtrsDmrsAssociation_Type_Value v;
};

enum NR_DciFormat_0_1_BetaOffsetIndicator_Type_Sel {
	NR_DciFormat_0_1_BetaOffsetIndicator_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_0_1_BetaOffsetIndicator_Type_None = 1,
	NR_DciFormat_0_1_BetaOffsetIndicator_Type_Value = 2,
};

union NR_DciFormat_0_1_BetaOffsetIndicator_Type_Value {
	Null_Type None;
	B2_Type Value;
};

struct NR_DciFormat_0_1_BetaOffsetIndicator_Type {
	enum NR_DciFormat_0_1_BetaOffsetIndicator_Type_Sel d;
	union NR_DciFormat_0_1_BetaOffsetIndicator_Type_Value v;
};

enum NR_DciFormat_0_1_UlschIndicator_Type_Sel {
	NR_DciFormat_0_1_UlschIndicator_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_0_1_UlschIndicator_Type_None = 1,
	NR_DciFormat_0_1_UlschIndicator_Type_Value = 2,
};

union NR_DciFormat_0_1_UlschIndicator_Type_Value {
	Null_Type None;
	B1_Type Value;
};

struct NR_DciFormat_0_1_UlschIndicator_Type {
	enum NR_DciFormat_0_1_UlschIndicator_Type_Sel d;
	union NR_DciFormat_0_1_UlschIndicator_Type_Value v;
};

enum NR_DciFormat_0_1_ChannelAccessCPextCAPC_Type_Sel {
	NR_DciFormat_0_1_ChannelAccessCPextCAPC_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_0_1_ChannelAccessCPextCAPC_Type_None = 1,
	NR_DciFormat_0_1_ChannelAccessCPextCAPC_Type_Value = 2,
};

union NR_DciFormat_0_1_ChannelAccessCPextCAPC_Type_Value {
	Null_Type None;
	BIT_STRING Value;
};

struct NR_DciFormat_0_1_ChannelAccessCPextCAPC_Type {
	enum NR_DciFormat_0_1_ChannelAccessCPextCAPC_Type_Sel d;
	union NR_DciFormat_0_1_ChannelAccessCPextCAPC_Type_Value v;
};

enum NR_DciFormat_0_1_OpenLoopPowerControl_Type_Sel {
	NR_DciFormat_0_1_OpenLoopPowerControl_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_0_1_OpenLoopPowerControl_Type_None = 1,
	NR_DciFormat_0_1_OpenLoopPowerControl_Type_Value = 2,
};

union NR_DciFormat_0_1_OpenLoopPowerControl_Type_Value {
	Null_Type None;
	BIT_STRING Value;
};

struct NR_DciFormat_0_1_OpenLoopPowerControl_Type {
	enum NR_DciFormat_0_1_OpenLoopPowerControl_Type_Sel d;
	union NR_DciFormat_0_1_OpenLoopPowerControl_Type_Value v;
};

enum NR_DciFormat_0_1_InvalidSymbolPatternIndicator_Type_Sel {
	NR_DciFormat_0_1_InvalidSymbolPatternIndicator_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_0_1_InvalidSymbolPatternIndicator_Type_None = 1,
	NR_DciFormat_0_1_InvalidSymbolPatternIndicator_Type_Value = 2,
};

union NR_DciFormat_0_1_InvalidSymbolPatternIndicator_Type_Value {
	Null_Type None;
	B1_Type Value;
};

struct NR_DciFormat_0_1_InvalidSymbolPatternIndicator_Type {
	enum NR_DciFormat_0_1_InvalidSymbolPatternIndicator_Type_Sel d;
	union NR_DciFormat_0_1_InvalidSymbolPatternIndicator_Type_Value v;
};

enum NR_DciFormat_0_1_SidelinkAssignmentIndex_Type_Sel {
	NR_DciFormat_0_1_SidelinkAssignmentIndex_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_0_1_SidelinkAssignmentIndex_Type_None = 1,
	NR_DciFormat_0_1_SidelinkAssignmentIndex_Type_Value = 2,
};

union NR_DciFormat_0_1_SidelinkAssignmentIndex_Type_Value {
	Null_Type None;
	BIT_STRING Value;
};

struct NR_DciFormat_0_1_SidelinkAssignmentIndex_Type {
	enum NR_DciFormat_0_1_SidelinkAssignmentIndex_Type_Sel d;
	union NR_DciFormat_0_1_SidelinkAssignmentIndex_Type_Value v;
};

struct NR_DciCommon_CarrierIndicator_Type_NR_DciFormat_0_1_SpecificInfo_Type_CarrierIndicator_Optional {
	bool d;
	struct NR_DciCommon_CarrierIndicator_Type v;
};

struct NR_DciFormat_0_1_DfiFlag_Type_DfiFlag_Optional {
	bool d;
	struct NR_DciFormat_0_1_DfiFlag_Type v;
};

struct NR_DciCommon_BWPIndicator_Type_NR_DciFormat_0_1_SpecificInfo_Type_BWPIndicator_Optional {
	bool d;
	struct NR_DciCommon_BWPIndicator_Type v;
};

struct NR_DciFormat_0_1_FirstDAI_Type_FirstDAI_Optional {
	bool d;
	struct NR_DciFormat_0_1_FirstDAI_Type v;
};

struct NR_DciFormat_0_1_SecondDAI_Type_SecondDAI_Optional {
	bool d;
	struct NR_DciFormat_0_1_SecondDAI_Type v;
};

struct NR_DciFormat_0_1_SrsResourceIndicator_Type_SrsResourceIndicator_Optional {
	bool d;
	struct NR_DciFormat_0_1_SrsResourceIndicator_Type v;
};

struct NR_DciFormat_0_1_PrecodingInfo_Type_PrecodingInfo_Optional {
	bool d;
	struct NR_DciFormat_0_1_PrecodingInfo_Type v;
};

struct NR_DciFormat_0_1_AntennaPorts_Type_AntennaPorts_Optional {
	bool d;
	struct NR_DciFormat_0_1_AntennaPorts_Type v;
};

struct NR_DciFormat_X_1_SrsRequest_Type_NR_DciFormat_0_1_SpecificInfo_Type_SrsRequest_Optional {
	bool d;
	struct NR_DciFormat_X_1_SrsRequest_Type v;
};

struct NR_DciFormat_0_1_CsiRequest_Type_CsiRequest_Optional {
	bool d;
	struct NR_DciFormat_0_1_CsiRequest_Type v;
};

struct NR_DciFormat_0_1_CBGTI_Type_CBGTI_Optional {
	bool d;
	struct NR_DciFormat_0_1_CBGTI_Type v;
};

struct NR_DciFormat_0_1_PtrsDmrsAssociation_Type_PtrsDmrsAssociation_Optional {
	bool d;
	struct NR_DciFormat_0_1_PtrsDmrsAssociation_Type v;
};

struct NR_DciFormat_0_1_BetaOffsetIndicator_Type_BetaOffsetIndicator_Optional {
	bool d;
	struct NR_DciFormat_0_1_BetaOffsetIndicator_Type v;
};

struct NR_DciFormat_X_1_DmrsSequenceInit_Type_NR_DciFormat_0_1_SpecificInfo_Type_DmrsSequenceInit_Optional {
	bool d;
	struct NR_DciFormat_X_1_DmrsSequenceInit_Type v;
};

struct NR_DciFormat_0_1_UlschIndicator_Type_UlschIndicator_Optional {
	bool d;
	struct NR_DciFormat_0_1_UlschIndicator_Type v;
};

struct NR_DciFormat_0_1_ChannelAccessCPextCAPC_Type_ChannelAccessCPextCAPC_Optional {
	bool d;
	struct NR_DciFormat_0_1_ChannelAccessCPextCAPC_Type v;
};

struct NR_DciFormat_0_1_OpenLoopPowerControl_Type_OpenLoopPowerControl_Optional {
	bool d;
	struct NR_DciFormat_0_1_OpenLoopPowerControl_Type v;
};

struct NR_DciFormat_X_1_PriorityIndicator_Type_NR_DciFormat_0_1_SpecificInfo_Type_PriorityIndicator_Optional {
	bool d;
	struct NR_DciFormat_X_1_PriorityIndicator_Type v;
};

struct NR_DciFormat_0_1_InvalidSymbolPatternIndicator_Type_InvalidSymbolPatternIndicator_Optional {
	bool d;
	struct NR_DciFormat_0_1_InvalidSymbolPatternIndicator_Type v;
};

struct NR_DciFormat_X_1_MinimumApplicableSchedulingOffset_Type_NR_DciFormat_0_1_SpecificInfo_Type_MinimumApplicableSchedulingOffset_Optional {
	bool d;
	struct NR_DciFormat_X_1_MinimumApplicableSchedulingOffset_Type v;
};

struct NR_DciFormat_X_1_SCellDormancyIndication_Type_NR_DciFormat_0_1_SpecificInfo_Type_SCellDormancyIndication_Optional {
	bool d;
	struct NR_DciFormat_X_1_SCellDormancyIndication_Type v;
};

struct NR_DciFormat_0_1_SidelinkAssignmentIndex_Type_SidelinkAssignmentIndex_Optional {
	bool d;
	struct NR_DciFormat_0_1_SidelinkAssignmentIndex_Type v;
};

struct NR_DciFormat_0_1_SpecificInfo_Type {
	struct NR_DciCommon_CarrierIndicator_Type_NR_DciFormat_0_1_SpecificInfo_Type_CarrierIndicator_Optional CarrierIndicator;
	struct NR_DciFormat_0_1_DfiFlag_Type_DfiFlag_Optional DfiFlag;
	struct NR_DciCommon_BWPIndicator_Type_NR_DciFormat_0_1_SpecificInfo_Type_BWPIndicator_Optional BWPIndicator;
	struct NR_DciFormat_0_1_FirstDAI_Type_FirstDAI_Optional FirstDAI;
	struct NR_DciFormat_0_1_SecondDAI_Type_SecondDAI_Optional SecondDAI;
	struct NR_DciFormat_0_1_SrsResourceIndicator_Type_SrsResourceIndicator_Optional SrsResourceIndicator;
	struct NR_DciFormat_0_1_PrecodingInfo_Type_PrecodingInfo_Optional PrecodingInfo;
	struct NR_DciFormat_0_1_AntennaPorts_Type_AntennaPorts_Optional AntennaPorts;
	struct NR_DciFormat_X_1_SrsRequest_Type_NR_DciFormat_0_1_SpecificInfo_Type_SrsRequest_Optional SrsRequest;
	struct NR_DciFormat_0_1_CsiRequest_Type_CsiRequest_Optional CsiRequest;
	struct NR_DciFormat_0_1_CBGTI_Type_CBGTI_Optional CBGTI;
	struct NR_DciFormat_0_1_PtrsDmrsAssociation_Type_PtrsDmrsAssociation_Optional PtrsDmrsAssociation;
	struct NR_DciFormat_0_1_BetaOffsetIndicator_Type_BetaOffsetIndicator_Optional BetaOffsetIndicator;
	struct NR_DciFormat_X_1_DmrsSequenceInit_Type_NR_DciFormat_0_1_SpecificInfo_Type_DmrsSequenceInit_Optional DmrsSequenceInit;
	struct NR_DciFormat_0_1_UlschIndicator_Type_UlschIndicator_Optional UlschIndicator;
	struct NR_DciFormat_0_1_ChannelAccessCPextCAPC_Type_ChannelAccessCPextCAPC_Optional ChannelAccessCPextCAPC;
	struct NR_DciFormat_0_1_OpenLoopPowerControl_Type_OpenLoopPowerControl_Optional OpenLoopPowerControl;
	struct NR_DciFormat_X_1_PriorityIndicator_Type_NR_DciFormat_0_1_SpecificInfo_Type_PriorityIndicator_Optional PriorityIndicator;
	struct NR_DciFormat_0_1_InvalidSymbolPatternIndicator_Type_InvalidSymbolPatternIndicator_Optional InvalidSymbolPatternIndicator;
	struct NR_DciFormat_X_1_MinimumApplicableSchedulingOffset_Type_NR_DciFormat_0_1_SpecificInfo_Type_MinimumApplicableSchedulingOffset_Optional MinimumApplicableSchedulingOffset;
	struct NR_DciFormat_X_1_SCellDormancyIndication_Type_NR_DciFormat_0_1_SpecificInfo_Type_SCellDormancyIndication_Optional SCellDormancyIndication;
	struct NR_DciFormat_0_1_SidelinkAssignmentIndex_Type_SidelinkAssignmentIndex_Optional SidelinkAssignmentIndex;
};

enum NR_DciFormat_0_X_SpecificInfo_Type_Sel {
	NR_DciFormat_0_X_SpecificInfo_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_0_X_SpecificInfo_Type_Format_0_0 = 1,
	NR_DciFormat_0_X_SpecificInfo_Type_Format_0_1 = 2,
};

union NR_DciFormat_0_X_SpecificInfo_Type_Value {
	struct NR_DciFormat_0_0_SpecificInfo_Type Format_0_0;
	struct NR_DciFormat_0_1_SpecificInfo_Type Format_0_1;
};

struct NR_DciFormat_0_X_SpecificInfo_Type {
	enum NR_DciFormat_0_X_SpecificInfo_Type_Sel d;
	union NR_DciFormat_0_X_SpecificInfo_Type_Value v;
};

struct NR_DciFormat_0_X_ResourceAssignment_Type_ResoureAssignment_Optional {
	bool d;
	struct NR_DciFormat_0_X_ResourceAssignment_Type v;
};

struct NR_DciFormat_0_X_PuschHoppingCtrl_Type_PuschHoppingCtrl_Optional {
	bool d;
	struct NR_DciFormat_0_X_PuschHoppingCtrl_Type v;
};

struct NR_DciCommon_TpcCommand_Type_TpcCommandPusch_Optional {
	bool d;
	struct NR_DciCommon_TpcCommand_Type v;
};

struct NR_DciCommon_UL_SUL_Indicator_Type_UL_SUL_Indicator_Optional {
	bool d;
	struct NR_DciCommon_UL_SUL_Indicator_Type v;
};

struct NR_DciFormat_0_X_SpecificInfo_Type_Format_Optional {
	bool d;
	struct NR_DciFormat_0_X_SpecificInfo_Type v;
};

struct NR_DciUlInfo_Type {
	struct NR_DciFormat_0_X_ResourceAssignment_Type_ResoureAssignment_Optional ResoureAssignment;
	struct NR_DciFormat_0_X_PuschHoppingCtrl_Type_PuschHoppingCtrl_Optional PuschHoppingCtrl;
	struct NR_DciCommon_TpcCommand_Type_TpcCommandPusch_Optional TpcCommandPusch;
	struct NR_DciCommon_UL_SUL_Indicator_Type_UL_SUL_Indicator_Optional UL_SUL_Indicator;
	struct NR_DciFormat_0_X_SpecificInfo_Type_Format_Optional Format;
};

struct NR_AssignedBWPs_Type_NR_SearchSpaceUlDciAssignment_Type_AssignedBWPs_Optional {
	bool d;
	struct NR_AssignedBWPs_Type v;
};

struct NR_SearchSpaceType_Type_NR_SearchSpaceUlDciAssignment_Type_SearchSpaceType_Optional {
	bool d;
	NR_SearchSpaceType_Type v;
};

struct NR_DciUlInfo_Type_DciInfo_Optional {
	bool d;
	struct NR_DciUlInfo_Type v;
};

struct NR_SearchSpaceUlDciAssignment_Type {
	struct NR_AssignedBWPs_Type_NR_SearchSpaceUlDciAssignment_Type_AssignedBWPs_Optional AssignedBWPs;
	struct NR_SearchSpaceType_Type_NR_SearchSpaceUlDciAssignment_Type_SearchSpaceType_Optional SearchSpaceType;
	struct NR_DciUlInfo_Type_DciInfo_Optional DciInfo;
};

SIDL_END_C_INTERFACE
