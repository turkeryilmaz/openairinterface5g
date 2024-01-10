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
#include "SidlParts.h"
#include "SidlVals.h"
#include "TtcnCommon.h"
#include "SidlCommon_BcchConfig.h"
#include "SidlCommon_Bcch_BRConfig.h"
#include "SidlCommon_CcchDcchDtchConfig.h"
#include "SidlCommon_InitialCellPower.h"
#include "SidlCommon_PhysicalLayerConfigDL.h"
#include "SidlCommon_PhysicalLayerConfigUL.h"
#include "SidlCommon_RachProcedureConfig.h"
#include "SidlCommon_SciSchedulingConfig.h"
#include "SidlCommon_ServingCellConfig.h"
#include "SidlCommon_StaticCellInfo.h"
#include "SidlCommon_NR_BcchConfig_Type.h"
#include "SidlCommon_NR_CellConfigCommon_Type.h"
#include "SidlCommon_NR_CellConfigPhysicalLayer_Type.h"
#include "SidlCommon_NR_DcchDtchConfig_Type.h"
#include "SidlCommon_NR_PcchConfig_Type.h"
#include "SidlCommon_NR_RachProcedureConfig_Type.h"
#include "SidlCommon_NR_SS_StaticCellResourceConfig_Type.h"
#include "SidlCommon_NR_ServingCellConfig_Type.h"
#include "SidlCommon_CellConfigRequest.h"
#include "SidlCommon_NR_CellConfigRequest.h"

SIDL_BEGIN_C_INTERFACE

enum NR_HarqProcessAssignment_Type_Sel {
	NR_HarqProcessAssignment_Type_UNBOUND_VALUE = 0,
	NR_HarqProcessAssignment_Type_Id = 1,
	NR_HarqProcessAssignment_Type_Automatic = 2,
};

union NR_HarqProcessAssignment_Type_Value {
	NR_HarqProcessId_Type Id;
	Null_Type Automatic;
};

struct NR_HarqProcessAssignment_Type {
	enum NR_HarqProcessAssignment_Type_Sel d;
	union NR_HarqProcessAssignment_Type_Value v;
};

typedef B16_Type RNTI_B16_Type;

typedef O1_Type NR_MAC_LongBSR_BufferSize_Type;

struct NR_MAC_LongBSR_BufferSize_Type_NR_MAC_LongBSR_BufferSizeList_Type_Dynamic {
	size_t d;
	NR_MAC_LongBSR_BufferSize_Type* v;
};

typedef struct NR_MAC_LongBSR_BufferSize_Type_NR_MAC_LongBSR_BufferSizeList_Type_Dynamic NR_MAC_LongBSR_BufferSizeList_Type;

typedef B8_Type NR_MAC_CE_DuplicationActDeact_Type;

typedef B48_Type NR_MAC_CE_ContentionResolutionId_Type;

struct NR_MAC_CE_TimingAdvance_Type {
	B2_Type TAG_ID;
	B6_Type TimingAdvanceCommand;
};

struct B8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex15_8_Optional {
	bool d;
	B8_Type v;
};

struct B8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex23_16_Optional {
	bool d;
	B8_Type v;
};

struct B8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex31_24_Optional {
	bool d;
	B8_Type v;
};

struct NR_MAC_CE_SCellFlags_Type {
	B8_Type SCellIndex7_1;
	struct B8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex15_8_Optional SCellIndex15_8;
	struct B8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex23_16_Optional SCellIndex23_16;
	struct B8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex31_24_Optional SCellIndex31_24;
};

typedef struct NR_MAC_CE_SCellFlags_Type NR_MAC_CE_SCellActDeact_Type;

struct NR_MAC_CE_ServCellId_BwpId_Type {
	B1_Type Field1;
	B5_Type ServCellId;
	B2_Type BwpId;
};

struct NR_MAC_CE_SP_ResourceSetActDeact_Octet2_Type {
	B1_Type Reserved;
	B1_Type IM;
	B6_Type CSI_RS_ResourcesetId;
};

struct NR_MAC_CE_SP_ResourceSetActDeact_Octet3_Type {
	B2_Type Reserved;
	B6_Type CSI_IM_ResourcesetId;
};

struct NR_MAC_CE_SP_ResourceSetActDeact_TciStateId_Type {
	B1_Type Reserved;
	B7_Type Id;
};

struct NR_MAC_CE_SP_ResourceSetActDeact_TciStateId_Type_NR_MAC_CE_SP_ResourceSetActDeact_TciStateIdList_Type_Dynamic {
	size_t d;
	struct NR_MAC_CE_SP_ResourceSetActDeact_TciStateId_Type* v;
};

typedef struct NR_MAC_CE_SP_ResourceSetActDeact_TciStateId_Type_NR_MAC_CE_SP_ResourceSetActDeact_TciStateIdList_Type_Dynamic NR_MAC_CE_SP_ResourceSetActDeact_TciStateIdList_Type;

struct NR_MAC_CE_SP_ResourceSetActDeact_Octet3_Type_NR_MAC_CE_SP_ResourceSetActDeact_Type_Octet3_Optional {
	bool d;
	struct NR_MAC_CE_SP_ResourceSetActDeact_Octet3_Type v;
};

struct NR_MAC_CE_SP_ResourceSetActDeact_Type {
	struct NR_MAC_CE_ServCellId_BwpId_Type Octet1;
	struct NR_MAC_CE_SP_ResourceSetActDeact_Octet2_Type Octet2;
	struct NR_MAC_CE_SP_ResourceSetActDeact_Octet3_Type_NR_MAC_CE_SP_ResourceSetActDeact_Type_Octet3_Optional Octet3;
	NR_MAC_CE_SP_ResourceSetActDeact_TciStateIdList_Type IdList;
};

struct NR_MAC_CE_CSI_TriggerStateSubselection_Type {
	struct NR_MAC_CE_ServCellId_BwpId_Type Octet1;
	B8_List_Type Selection;
};

struct NR_MAC_CE_TCI_StatesActDeact_Type {
	struct NR_MAC_CE_ServCellId_BwpId_Type Octet1;
	B8_List_Type Status;
};

struct NR_MAC_CE_TCI_StateIndication_Type {
	B5_Type ServCellId;
	B4_Type CoresetId;
	B7_Type TciStateId;
};

struct NR_MAC_CE_SP_CSI_ReportingActDeact_Type {
	struct NR_MAC_CE_ServCellId_BwpId_Type Octet1;
	B4_Type Reserved;
	B4_Type ConfigState;
};

struct NR_MAC_CE_SP_SRS_ActDeact_Octet2_Type {
	B2_Type Reserved;
	B1_Type C;
	B1_Type SUL;
	B4_Type SRS_ResourcesetId;
};

struct NR_MAC_CE_SP_SRS_ActDeact_ResourceId_Type {
	B1_Type F;
	B7_Type Id;
};

struct NR_MAC_CE_SP_SRS_ActDeact_ResourceInfo_Type {
	B1_Type Reserved;
	B5_Type ServingCellId;
	B2_Type BwpId;
};

struct NR_MAC_CE_SP_SRS_ActDeact_ResourceId_Type_NR_MAC_CE_SP_SRS_ActDeact_ResourceIdList_Type_Dynamic {
	size_t d;
	struct NR_MAC_CE_SP_SRS_ActDeact_ResourceId_Type* v;
};

typedef struct NR_MAC_CE_SP_SRS_ActDeact_ResourceId_Type_NR_MAC_CE_SP_SRS_ActDeact_ResourceIdList_Type_Dynamic NR_MAC_CE_SP_SRS_ActDeact_ResourceIdList_Type;

struct NR_MAC_CE_SP_SRS_ActDeact_ResourceInfo_Type_NR_MAC_CE_SP_SRS_ActDeact_ResourceInfoList_Type_Dynamic {
	size_t d;
	struct NR_MAC_CE_SP_SRS_ActDeact_ResourceInfo_Type* v;
};

typedef struct NR_MAC_CE_SP_SRS_ActDeact_ResourceInfo_Type_NR_MAC_CE_SP_SRS_ActDeact_ResourceInfoList_Type_Dynamic NR_MAC_CE_SP_SRS_ActDeact_ResourceInfoList_Type;

struct NR_MAC_CE_SP_SRS_ActDeact_Type {
	struct NR_MAC_CE_ServCellId_BwpId_Type Octet1;
	struct NR_MAC_CE_SP_SRS_ActDeact_Octet2_Type Octet2;
	NR_MAC_CE_SP_SRS_ActDeact_ResourceIdList_Type ResourceIdList;
	NR_MAC_CE_SP_SRS_ActDeact_ResourceInfoList_Type ResourceInfoList;
};

struct NR_MAC_CE_PUCCH_SpatialRelationActDeact_Octet2_Type {
	B1_Type Reserved;
	B7_Type ResourceId;
};

struct NR_MAC_CE_PUCCH_SpatialRelationActDeact_Type {
	struct NR_MAC_CE_ServCellId_BwpId_Type Octet1;
	struct NR_MAC_CE_PUCCH_SpatialRelationActDeact_Octet2_Type Octet2;
	B8_Type ActivationStatus;
};

struct NR_MAC_CE_SP_ZP_ResourceSetActDeact_Octet2_Type {
	B4_Type Reserved;
	B4_Type Id;
};

struct NR_MAC_CE_SP_ZP_ResourceSetActDeact_Type {
	struct NR_MAC_CE_ServCellId_BwpId_Type Octet1;
	struct NR_MAC_CE_SP_ZP_ResourceSetActDeact_Octet2_Type Octet2;
};

struct NR_MAC_CE_ShortBSR_Type {
	B3_Type LCG;
	B5_Type BufferSize;
};

struct NR_MAC_CE_LongBSR_Type {
	B8_Type LCG_Presence;
	NR_MAC_LongBSR_BufferSizeList_Type BufferSizeList;
};

struct B2_Type_NR_MAC_CE_PH_Record_Type_MPE_or_R_Optional {
	bool d;
	B2_Type v;
};

struct B6_Type_NR_MAC_CE_PH_Record_Type_PCMaxc_Optional {
	bool d;
	B6_Type v;
};

struct NR_MAC_CE_PH_Record_Type {
	B1_Type P_Bit;
	B1_Type V_Bit;
	B6_Type Value;
	struct B2_Type_NR_MAC_CE_PH_Record_Type_MPE_or_R_Optional MPE_or_R;
	struct B6_Type_NR_MAC_CE_PH_Record_Type_PCMaxc_Optional PCMaxc;
};

typedef struct NR_MAC_CE_PH_Record_Type NR_MAC_CE_SingleEntryPHR_Type;

struct NR_MAC_CE_PH_Record_Type_PH_Record_Dynamic {
	size_t d;
	struct NR_MAC_CE_PH_Record_Type* v;
};

struct NR_MAC_CE_MultiEntryPHR_Type {
	struct NR_MAC_CE_SCellFlags_Type PHFieldPresentForSCell;
	struct NR_MAC_CE_PH_Record_Type_PH_Record_Dynamic PH_Record;
};

struct NR_MAC_CE_RecommendedBitrate_Type {
	B6_Type LCID;
	B1_Type UL_DL;
	B6_Type Bitrate;
	B1_Type X;
	B2_Type Reserved;
};

enum NR_MAC_ControlElementDL_Type_Sel {
	NR_MAC_ControlElementDL_Type_UNBOUND_VALUE = 0,
	NR_MAC_ControlElementDL_Type_ContentionResolutionID = 1,
	NR_MAC_ControlElementDL_Type_TimingAdvance = 2,
	NR_MAC_ControlElementDL_Type_SCellActDeact = 3,
	NR_MAC_ControlElementDL_Type_DuplicationActDeact = 4,
	NR_MAC_ControlElementDL_Type_SP_ResourceSetActDeact = 5,
	NR_MAC_ControlElementDL_Type_CSI_TriggerStateSubselection = 6,
	NR_MAC_ControlElementDL_Type_TCI_StatesActDeact = 7,
	NR_MAC_ControlElementDL_Type_TCI_StateIndication = 8,
	NR_MAC_ControlElementDL_Type_SP_CSI_ReportingActDeact = 9,
	NR_MAC_ControlElementDL_Type_SP_SRS_ActDeact = 10,
	NR_MAC_ControlElementDL_Type_PUCCH_SpatialRelationActDeact = 11,
	NR_MAC_ControlElementDL_Type_SP_ZP_ResourceSetActDeact = 12,
	NR_MAC_ControlElementDL_Type_RecommendatdBitrate = 13,
};

union NR_MAC_ControlElementDL_Type_Value {
	NR_MAC_CE_ContentionResolutionId_Type ContentionResolutionID;
	struct NR_MAC_CE_TimingAdvance_Type TimingAdvance;
	NR_MAC_CE_SCellActDeact_Type SCellActDeact;
	NR_MAC_CE_DuplicationActDeact_Type DuplicationActDeact;
	struct NR_MAC_CE_SP_ResourceSetActDeact_Type SP_ResourceSetActDeact;
	struct NR_MAC_CE_CSI_TriggerStateSubselection_Type CSI_TriggerStateSubselection;
	struct NR_MAC_CE_TCI_StatesActDeact_Type TCI_StatesActDeact;
	struct NR_MAC_CE_TCI_StateIndication_Type TCI_StateIndication;
	struct NR_MAC_CE_SP_CSI_ReportingActDeact_Type SP_CSI_ReportingActDeact;
	struct NR_MAC_CE_SP_SRS_ActDeact_Type SP_SRS_ActDeact;
	struct NR_MAC_CE_PUCCH_SpatialRelationActDeact_Type PUCCH_SpatialRelationActDeact;
	struct NR_MAC_CE_SP_ZP_ResourceSetActDeact_Type SP_ZP_ResourceSetActDeact;
	struct NR_MAC_CE_RecommendedBitrate_Type RecommendatdBitrate;
};

struct NR_MAC_ControlElementDL_Type {
	enum NR_MAC_ControlElementDL_Type_Sel d;
	union NR_MAC_ControlElementDL_Type_Value v;
};

enum NR_MAC_ControlElementUL_Type_Sel {
	NR_MAC_ControlElementUL_Type_UNBOUND_VALUE = 0,
	NR_MAC_ControlElementUL_Type_ShortBSR = 1,
	NR_MAC_ControlElementUL_Type_LongBSR = 2,
	NR_MAC_ControlElementUL_Type_C_RNTI = 3,
	NR_MAC_ControlElementUL_Type_SingleEntryPHR = 4,
	NR_MAC_ControlElementUL_Type_MultiEntryPHR = 5,
	NR_MAC_ControlElementUL_Type_RecommendedBitrate = 6,
};

union NR_MAC_ControlElementUL_Type_Value {
	struct NR_MAC_CE_ShortBSR_Type ShortBSR;
	struct NR_MAC_CE_LongBSR_Type LongBSR;
	RNTI_B16_Type C_RNTI;
	NR_MAC_CE_SingleEntryPHR_Type SingleEntryPHR;
	struct NR_MAC_CE_MultiEntryPHR_Type MultiEntryPHR;
	struct NR_MAC_CE_RecommendedBitrate_Type RecommendedBitrate;
};

struct NR_MAC_ControlElementUL_Type {
	enum NR_MAC_ControlElementUL_Type_Sel d;
	union NR_MAC_ControlElementUL_Type_Value v;
};

struct BIT_STRING_NR_MAC_PDU_SubHeader_Type_ELCID_Optional {
	bool d;
	BIT_STRING v;
};

struct BIT_STRING_NR_MAC_PDU_SubHeader_Type_Length_Optional {
	bool d;
	BIT_STRING v;
};

struct NR_MAC_PDU_SubHeader_Type {
	B1_Type Reserved;
	B1_Type Format;
	B6_Type LCID;
	struct BIT_STRING_NR_MAC_PDU_SubHeader_Type_ELCID_Optional ELCID;
	struct BIT_STRING_NR_MAC_PDU_SubHeader_Type_Length_Optional Length;
};

typedef OCTET_STRING NR_MAC_SDU_Type;

struct NR_MAC_ControlElementDL_Type_NR_MAC_CE_SubPDU_DL_Type_ControlElement_Optional {
	bool d;
	struct NR_MAC_ControlElementDL_Type v;
};

struct NR_MAC_CE_SubPDU_DL_Type {
	struct NR_MAC_PDU_SubHeader_Type SubHeader;
	struct NR_MAC_ControlElementDL_Type_NR_MAC_CE_SubPDU_DL_Type_ControlElement_Optional ControlElement;
};

struct NR_MAC_SDU_SubPDU_Type {
	struct NR_MAC_PDU_SubHeader_Type SubHeader;
	NR_MAC_SDU_Type SDU;
};

struct NR_MAC_ControlElementUL_Type_NR_MAC_CE_SubPDU_UL_Type_ControlElement_Optional {
	bool d;
	struct NR_MAC_ControlElementUL_Type v;
};

struct NR_MAC_CE_SubPDU_UL_Type {
	struct NR_MAC_PDU_SubHeader_Type SubHeader;
	struct NR_MAC_ControlElementUL_Type_NR_MAC_CE_SubPDU_UL_Type_ControlElement_Optional ControlElement;
};

struct NR_MAC_CE_SubPDU_DL_Type_NR_MAC_CE_SubPDU_DL_List_Type_Dynamic {
	size_t d;
	struct NR_MAC_CE_SubPDU_DL_Type* v;
};

typedef struct NR_MAC_CE_SubPDU_DL_Type_NR_MAC_CE_SubPDU_DL_List_Type_Dynamic NR_MAC_CE_SubPDU_DL_List_Type;

struct NR_MAC_SDU_SubPDU_Type_NR_MAC_SDU_SubPDU_List_Type_Dynamic {
	size_t d;
	struct NR_MAC_SDU_SubPDU_Type* v;
};

typedef struct NR_MAC_SDU_SubPDU_Type_NR_MAC_SDU_SubPDU_List_Type_Dynamic NR_MAC_SDU_SubPDU_List_Type;

struct NR_MAC_CE_SubPDU_UL_Type_NR_MAC_CE_SubPDU_UL_List_Type_Dynamic {
	size_t d;
	struct NR_MAC_CE_SubPDU_UL_Type* v;
};

typedef struct NR_MAC_CE_SubPDU_UL_Type_NR_MAC_CE_SubPDU_UL_List_Type_Dynamic NR_MAC_CE_SubPDU_UL_List_Type;

struct NR_MAC_Padding_SubPDU_Type {
	struct NR_MAC_PDU_SubHeader_Type SubHeader;
	OCTET_STRING Padding;
};

struct NR_MAC_CE_SubPDU_DL_List_Type_NR_MAC_PDU_DL_Type_CE_SubPDUList_Optional {
	bool d;
	NR_MAC_CE_SubPDU_DL_List_Type v;
};

struct NR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_DL_Type_SDU_SubPDUList_Optional {
	bool d;
	NR_MAC_SDU_SubPDU_List_Type v;
};

struct NR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_DL_Type_Padding_SubPDU_Optional {
	bool d;
	struct NR_MAC_Padding_SubPDU_Type v;
};

struct NR_MAC_PDU_DL_Type {
	struct NR_MAC_CE_SubPDU_DL_List_Type_NR_MAC_PDU_DL_Type_CE_SubPDUList_Optional CE_SubPDUList;
	struct NR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_DL_Type_SDU_SubPDUList_Optional SDU_SubPDUList;
	struct NR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_DL_Type_Padding_SubPDU_Optional Padding_SubPDU;
};

struct NR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_UL_Type_SDU_SubPDUList_Optional {
	bool d;
	NR_MAC_SDU_SubPDU_List_Type v;
};

struct NR_MAC_CE_SubPDU_UL_List_Type_NR_MAC_PDU_UL_Type_CE_SubPDUList_Optional {
	bool d;
	NR_MAC_CE_SubPDU_UL_List_Type v;
};

struct NR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_UL_Type_Padding_SubPDU_Optional {
	bool d;
	struct NR_MAC_Padding_SubPDU_Type v;
};

struct NR_MAC_PDU_UL_Type {
	struct NR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_UL_Type_SDU_SubPDUList_Optional SDU_SubPDUList;
	struct NR_MAC_CE_SubPDU_UL_List_Type_NR_MAC_PDU_UL_Type_CE_SubPDUList_Optional CE_SubPDUList;
	struct NR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_UL_Type_Padding_SubPDU_Optional Padding_SubPDU;
};

enum NR_MAC_PDU_Type_Sel {
	NR_MAC_PDU_Type_UNBOUND_VALUE = 0,
	NR_MAC_PDU_Type_DL = 1,
	NR_MAC_PDU_Type_UL = 2,
};

union NR_MAC_PDU_Type_Value {
	struct NR_MAC_PDU_DL_Type DL;
	struct NR_MAC_PDU_UL_Type UL;
};

struct NR_MAC_PDU_Type {
	enum NR_MAC_PDU_Type_Sel d;
	union NR_MAC_PDU_Type_Value v;
};

typedef B2_Type NR_RLC_SegmentationInfo_Type;

typedef B1_Type NR_RLC_Status_ExtensionBit1_Type;

typedef B1_Type NR_RLC_Status_ExtensionBit2_Type;

typedef B1_Type NR_RLC_Status_ExtensionBit3_Type;

typedef B16_Type NR_RLC_SegmentOffset_Type;

typedef OCTET_STRING NR_RLC_TMD_PDU_Type;

struct NR_RLC_UMD_HeaderNoSN_Type {
	NR_RLC_SegmentationInfo_Type SegmentationInfo;
	B6_Type Reserved;
};

typedef OCTET_STRING NR_RLC_UMD_Data_Type;

struct NR_RLC_UMD_PduNoSN_Type {
	struct NR_RLC_UMD_HeaderNoSN_Type Header;
	NR_RLC_UMD_Data_Type Data;
};

struct NR_RLC_SegmentOffset_Type_NR_RLC_UMD_HeaderSN6Bit_Type_SegmentOffset_Optional {
	bool d;
	NR_RLC_SegmentOffset_Type v;
};

struct NR_RLC_UMD_HeaderSN6Bit_Type {
	NR_RLC_SegmentationInfo_Type SegmentationInfo;
	B6_Type SequenceNumber;
	struct NR_RLC_SegmentOffset_Type_NR_RLC_UMD_HeaderSN6Bit_Type_SegmentOffset_Optional SegmentOffset;
};

struct NR_RLC_UMD_PduSN6Bit_Type {
	struct NR_RLC_UMD_HeaderSN6Bit_Type Header;
	NR_RLC_UMD_Data_Type Data;
};

struct NR_RLC_SegmentOffset_Type_NR_RLC_UMD_HeaderSN12Bit_Type_SegmentOffset_Optional {
	bool d;
	NR_RLC_SegmentOffset_Type v;
};

struct NR_RLC_UMD_HeaderSN12Bit_Type {
	NR_RLC_SegmentationInfo_Type SegmentationInfo;
	B2_Type Reserved;
	B12_Type SequenceNumber;
	struct NR_RLC_SegmentOffset_Type_NR_RLC_UMD_HeaderSN12Bit_Type_SegmentOffset_Optional SegmentOffset;
};

struct NR_RLC_UMD_PduSN12Bit_Type {
	struct NR_RLC_UMD_HeaderSN12Bit_Type Header;
	NR_RLC_UMD_Data_Type Data;
};

enum NR_RLC_UMD_PDU_Type_Sel {
	NR_RLC_UMD_PDU_Type_UNBOUND_VALUE = 0,
	NR_RLC_UMD_PDU_Type_NoSN = 1,
	NR_RLC_UMD_PDU_Type_SN6Bit = 2,
	NR_RLC_UMD_PDU_Type_SN12Bit = 3,
};

union NR_RLC_UMD_PDU_Type_Value {
	struct NR_RLC_UMD_PduNoSN_Type NoSN;
	struct NR_RLC_UMD_PduSN6Bit_Type SN6Bit;
	struct NR_RLC_UMD_PduSN12Bit_Type SN12Bit;
};

struct NR_RLC_UMD_PDU_Type {
	enum NR_RLC_UMD_PDU_Type_Sel d;
	union NR_RLC_UMD_PDU_Type_Value v;
};

struct NR_RLC_SegmentOffset_Type_NR_RLC_AMD_HeaderSN12Bit_Type_SegmentOffset_Optional {
	bool d;
	NR_RLC_SegmentOffset_Type v;
};

struct NR_RLC_AMD_HeaderSN12Bit_Type {
	B1_Type D_C;
	B1_Type Poll;
	NR_RLC_SegmentationInfo_Type SegmentationInfo;
	B12_Type SequenceNumber;
	struct NR_RLC_SegmentOffset_Type_NR_RLC_AMD_HeaderSN12Bit_Type_SegmentOffset_Optional SegmentOffset;
};

typedef OCTET_STRING NR_RLC_AMD_Data_Type;

struct NR_RLC_AMD_PduSN12Bit_Type {
	struct NR_RLC_AMD_HeaderSN12Bit_Type Header;
	NR_RLC_AMD_Data_Type Data;
};

struct NR_RLC_SegmentOffset_Type_NR_RLC_AMD_HeaderSN18Bit_Type_SegmentOffset_Optional {
	bool d;
	NR_RLC_SegmentOffset_Type v;
};

struct NR_RLC_AMD_HeaderSN18Bit_Type {
	B1_Type D_C;
	B1_Type Poll;
	NR_RLC_SegmentationInfo_Type SegmentationInfo;
	B2_Type Reserved;
	B18_Type SequenceNumber;
	struct NR_RLC_SegmentOffset_Type_NR_RLC_AMD_HeaderSN18Bit_Type_SegmentOffset_Optional SegmentOffset;
};

struct NR_RLC_AMD_PduSN18Bit_Type {
	struct NR_RLC_AMD_HeaderSN18Bit_Type Header;
	NR_RLC_AMD_Data_Type Data;
};

enum NR_RLC_AMD_PDU_Type_Sel {
	NR_RLC_AMD_PDU_Type_UNBOUND_VALUE = 0,
	NR_RLC_AMD_PDU_Type_SN12Bit = 1,
	NR_RLC_AMD_PDU_Type_SN18Bit = 2,
};

union NR_RLC_AMD_PDU_Type_Value {
	struct NR_RLC_AMD_PduSN12Bit_Type SN12Bit;
	struct NR_RLC_AMD_PduSN18Bit_Type SN18Bit;
};

struct NR_RLC_AMD_PDU_Type {
	enum NR_RLC_AMD_PDU_Type_Sel d;
	union NR_RLC_AMD_PDU_Type_Value v;
};

struct NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN12Bit_Type_SOstart_Optional {
	bool d;
	NR_RLC_SegmentOffset_Type v;
};

struct NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN12Bit_Type_SOstop_Optional {
	bool d;
	NR_RLC_SegmentOffset_Type v;
};

struct B8_Type_NR_RLC_Status_NackSN12Bit_Type_NACKrange_Optional {
	bool d;
	B8_Type v;
};

struct NR_RLC_Status_NackSN12Bit_Type {
	B12_Type SequenceNumberNACK;
	NR_RLC_Status_ExtensionBit1_Type E1;
	NR_RLC_Status_ExtensionBit2_Type E2;
	NR_RLC_Status_ExtensionBit3_Type E3;
	B1_Type Reserved;
	struct NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN12Bit_Type_SOstart_Optional SOstart;
	struct NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN12Bit_Type_SOstop_Optional SOstop;
	struct B8_Type_NR_RLC_Status_NackSN12Bit_Type_NACKrange_Optional NACKrange;
};

struct NR_RLC_Status_NackSN12Bit_Type_NR_RLC_Status_NackListSN12Bit_Type_Dynamic {
	size_t d;
	struct NR_RLC_Status_NackSN12Bit_Type* v;
};

typedef struct NR_RLC_Status_NackSN12Bit_Type_NR_RLC_Status_NackListSN12Bit_Type_Dynamic NR_RLC_Status_NackListSN12Bit_Type;

struct NR_RLC_Status_NackListSN12Bit_Type_NackList_Optional {
	bool d;
	NR_RLC_Status_NackListSN12Bit_Type v;
};

struct NR_RLC_StatusPduSN12Bit_Type {
	B1_Type D_C;
	B3_Type CPT;
	B12_Type SequenceNumberACK;
	NR_RLC_Status_ExtensionBit1_Type E1;
	B7_Type Reserved;
	struct NR_RLC_Status_NackListSN12Bit_Type_NackList_Optional NackList;
};

struct NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN18Bit_Type_SOstart_Optional {
	bool d;
	NR_RLC_SegmentOffset_Type v;
};

struct NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN18Bit_Type_SOstop_Optional {
	bool d;
	NR_RLC_SegmentOffset_Type v;
};

struct B8_Type_NR_RLC_Status_NackSN18Bit_Type_NACKrange_Optional {
	bool d;
	B8_Type v;
};

struct NR_RLC_Status_NackSN18Bit_Type {
	B18_Type SequenceNumberNACK;
	NR_RLC_Status_ExtensionBit1_Type E1;
	NR_RLC_Status_ExtensionBit2_Type E2;
	NR_RLC_Status_ExtensionBit3_Type E3;
	B3_Type Reserved;
	struct NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN18Bit_Type_SOstart_Optional SOstart;
	struct NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN18Bit_Type_SOstop_Optional SOstop;
	struct B8_Type_NR_RLC_Status_NackSN18Bit_Type_NACKrange_Optional NACKrange;
};

struct NR_RLC_Status_NackSN18Bit_Type_NR_RLC_Status_NackListSN18Bit_Type_Dynamic {
	size_t d;
	struct NR_RLC_Status_NackSN18Bit_Type* v;
};

typedef struct NR_RLC_Status_NackSN18Bit_Type_NR_RLC_Status_NackListSN18Bit_Type_Dynamic NR_RLC_Status_NackListSN18Bit_Type;

struct NR_RLC_Status_NackListSN18Bit_Type_NackList_Optional {
	bool d;
	NR_RLC_Status_NackListSN18Bit_Type v;
};

struct NR_RLC_StatusPduSN18Bit_Type {
	B1_Type D_C;
	B3_Type CPT;
	B18_Type SequenceNumberACK;
	NR_RLC_Status_ExtensionBit1_Type E1;
	B1_Type Reserved;
	struct NR_RLC_Status_NackListSN18Bit_Type_NackList_Optional NackList;
};

enum NR_RLC_AM_StatusPDU_Type_Sel {
	NR_RLC_AM_StatusPDU_Type_UNBOUND_VALUE = 0,
	NR_RLC_AM_StatusPDU_Type_SN12Bit = 1,
	NR_RLC_AM_StatusPDU_Type_SN18Bit = 2,
};

union NR_RLC_AM_StatusPDU_Type_Value {
	struct NR_RLC_StatusPduSN12Bit_Type SN12Bit;
	struct NR_RLC_StatusPduSN18Bit_Type SN18Bit;
};

struct NR_RLC_AM_StatusPDU_Type {
	enum NR_RLC_AM_StatusPDU_Type_Sel d;
	union NR_RLC_AM_StatusPDU_Type_Value v;
};

enum NR_RLC_PDU_Type_Sel {
	NR_RLC_PDU_Type_UNBOUND_VALUE = 0,
	NR_RLC_PDU_Type_TMD = 1,
	NR_RLC_PDU_Type_UMD = 2,
	NR_RLC_PDU_Type_AMD = 3,
	NR_RLC_PDU_Type_Status = 4,
};

union NR_RLC_PDU_Type_Value {
	NR_RLC_TMD_PDU_Type TMD;
	struct NR_RLC_UMD_PDU_Type UMD;
	struct NR_RLC_AMD_PDU_Type AMD;
	struct NR_RLC_AM_StatusPDU_Type Status;
};

struct NR_RLC_PDU_Type {
	enum NR_RLC_PDU_Type_Sel d;
	union NR_RLC_PDU_Type_Value v;
};

typedef B3_Type NR_PDCP_CtrlPduType_Type;

typedef OCTET_STRING SDAP_SDU_Type;

typedef OCTET_STRING NR_RLC_SDU_Type;

typedef OCTET_STRING NR_PDCP_SDU_Type;

struct B32_Type_NR_PDCP_DataPduSN12Bits_Type_MAC_I_Optional {
	bool d;
	B32_Type v;
};

struct NR_PDCP_DataPduSN12Bits_Type {
	B1_Type D_C;
	B3_Type Reserved;
	B12_Type SequenceNumber;
	NR_PDCP_SDU_Type SDU;
	struct B32_Type_NR_PDCP_DataPduSN12Bits_Type_MAC_I_Optional MAC_I;
};

struct B32_Type_NR_PDCP_DataPduSN18Bits_Type_MAC_I_Optional {
	bool d;
	B32_Type v;
};

struct NR_PDCP_DataPduSN18Bits_Type {
	B1_Type D_C;
	B5_Type Reserved;
	B18_Type SequenceNumber;
	NR_PDCP_SDU_Type SDU;
	struct B32_Type_NR_PDCP_DataPduSN18Bits_Type_MAC_I_Optional MAC_I;
};

struct OCTET_STRING_NR_PDCP_CtrlPduStatus_Type_Bitmap_Optional {
	bool d;
	OCTET_STRING v;
};

struct NR_PDCP_CtrlPduStatus_Type {
	B1_Type D_C;
	NR_PDCP_CtrlPduType_Type PDU_Type;
	B4_Type Reserved;
	B32_Type FirstMissingCount;
	struct OCTET_STRING_NR_PDCP_CtrlPduStatus_Type_Bitmap_Optional Bitmap;
};

struct NR_PDCP_CtrlPduRohcFeedback_Type {
	B1_Type D_C;
	NR_PDCP_CtrlPduType_Type PDU_Type;
	B4_Type Reserved;
	OCTET_STRING RohcFeedback;
};

enum NR_PDCP_PDU_Type_Sel {
	NR_PDCP_PDU_Type_UNBOUND_VALUE = 0,
	NR_PDCP_PDU_Type_DataPduSN12Bits = 1,
	NR_PDCP_PDU_Type_DataPduSN18Bits = 2,
	NR_PDCP_PDU_Type_CtrlPduStatus = 3,
	NR_PDCP_PDU_Type_CtrlPduRohcFeedback = 4,
};

union NR_PDCP_PDU_Type_Value {
	struct NR_PDCP_DataPduSN12Bits_Type DataPduSN12Bits;
	struct NR_PDCP_DataPduSN18Bits_Type DataPduSN18Bits;
	struct NR_PDCP_CtrlPduStatus_Type CtrlPduStatus;
	struct NR_PDCP_CtrlPduRohcFeedback_Type CtrlPduRohcFeedback;
};

struct NR_PDCP_PDU_Type {
	enum NR_PDCP_PDU_Type_Sel d;
	union NR_PDCP_PDU_Type_Value v;
};

struct SDAP_DL_PduHeader_Type {
	B1_Type RDI;
	B1_Type RQI;
	B6_Type QFI;
};

struct SDAP_DL_PduHeader_Type_SDAP_PDU_DL_Type_Header_Optional {
	bool d;
	struct SDAP_DL_PduHeader_Type v;
};

struct SDAP_PDU_DL_Type {
	struct SDAP_DL_PduHeader_Type_SDAP_PDU_DL_Type_Header_Optional Header;
	SDAP_SDU_Type Data;
};

struct SDAP_UL_PduHeader_Type {
	B1_Type DC;
	B1_Type R;
	B6_Type QFI;
};

struct SDAP_UL_PduHeader_Type_SDAP_PDU_UL_Type_Header_Optional {
	bool d;
	struct SDAP_UL_PduHeader_Type v;
};

struct SDAP_SDU_Type_SDAP_PDU_UL_Type_Data_Optional {
	bool d;
	SDAP_SDU_Type v;
};

struct SDAP_PDU_UL_Type {
	struct SDAP_UL_PduHeader_Type_SDAP_PDU_UL_Type_Header_Optional Header;
	struct SDAP_SDU_Type_SDAP_PDU_UL_Type_Data_Optional Data;
};

enum SDAP_PDU_Type_Sel {
	SDAP_PDU_Type_UNBOUND_VALUE = 0,
	SDAP_PDU_Type_DL = 1,
	SDAP_PDU_Type_UL = 2,
};

union SDAP_PDU_Type_Value {
	struct SDAP_PDU_DL_Type DL;
	struct SDAP_PDU_UL_Type UL;
};

struct SDAP_PDU_Type {
	enum SDAP_PDU_Type_Sel d;
	union SDAP_PDU_Type_Value v;
};

struct SDAP_PDU_Type_SDAP_PDUList_Type_Dynamic {
	size_t d;
	struct SDAP_PDU_Type* v;
};

typedef struct SDAP_PDU_Type_SDAP_PDUList_Type_Dynamic SDAP_PDUList_Type;

struct SDAP_SDU_Type_SDAP_SDUList_Type_Dynamic {
	size_t d;
	SDAP_SDU_Type* v;
};

typedef struct SDAP_SDU_Type_SDAP_SDUList_Type_Dynamic SDAP_SDUList_Type;

struct NR_MAC_PDU_Type_NR_MAC_PDUList_Type_Dynamic {
	size_t d;
	struct NR_MAC_PDU_Type* v;
};

typedef struct NR_MAC_PDU_Type_NR_MAC_PDUList_Type_Dynamic NR_MAC_PDUList_Type;

struct NR_RLC_PDU_Type_NR_RLC_PDUList_Type_Dynamic {
	size_t d;
	struct NR_RLC_PDU_Type* v;
};

typedef struct NR_RLC_PDU_Type_NR_RLC_PDUList_Type_Dynamic NR_RLC_PDUList_Type;

struct NR_RLC_SDU_Type_NR_RLC_SDUList_Type_Dynamic {
	size_t d;
	NR_RLC_SDU_Type* v;
};

typedef struct NR_RLC_SDU_Type_NR_RLC_SDUList_Type_Dynamic NR_RLC_SDUList_Type;

struct NR_PDCP_PDU_Type_NR_PDCP_PDUList_Type_Dynamic {
	size_t d;
	struct NR_PDCP_PDU_Type* v;
};

typedef struct NR_PDCP_PDU_Type_NR_PDCP_PDUList_Type_Dynamic NR_PDCP_PDUList_Type;

struct NR_PDCP_SDU_Type_NR_PDCP_SDUList_Type_Dynamic {
	size_t d;
	NR_PDCP_SDU_Type* v;
};

typedef struct NR_PDCP_SDU_Type_NR_PDCP_SDUList_Type_Dynamic NR_PDCP_SDUList_Type;

enum NR_L2DataList_Type_Sel {
	NR_L2DataList_Type_UNBOUND_VALUE = 0,
	NR_L2DataList_Type_MacPdu = 1,
	NR_L2DataList_Type_RlcPdu = 2,
	NR_L2DataList_Type_RlcSdu = 3,
	NR_L2DataList_Type_PdcpPdu = 4,
	NR_L2DataList_Type_PdcpSdu = 5,
	NR_L2DataList_Type_SdapPdu = 6,
	NR_L2DataList_Type_SdapSdu = 7,
};

union NR_L2DataList_Type_Value {
	NR_MAC_PDUList_Type MacPdu;
	NR_RLC_PDUList_Type RlcPdu;
	NR_RLC_SDUList_Type RlcSdu;
	NR_PDCP_PDUList_Type PdcpPdu;
	NR_PDCP_SDUList_Type PdcpSdu;
	SDAP_PDUList_Type SdapPdu;
	SDAP_SDUList_Type SdapSdu;
};

struct NR_L2DataList_Type {
	enum NR_L2DataList_Type_Sel d;
	union NR_L2DataList_Type_Value v;
};

struct NR_HarqProcessAssignment_Type_HarqProcess_Optional {
	bool d;
	struct NR_HarqProcessAssignment_Type v;
};

struct NR_DRB_DataPerSlot_DL_Type {
	int32_t SlotOffset;
	struct NR_HarqProcessAssignment_Type_HarqProcess_Optional HarqProcess;
	struct NR_L2DataList_Type PduSduList;
};

struct NR_DRB_DataPerSlot_DL_Type_NR_DRB_DataPerSlotList_DL_Type_Dynamic {
	size_t d;
	struct NR_DRB_DataPerSlot_DL_Type* v;
};

typedef struct NR_DRB_DataPerSlot_DL_Type_NR_DRB_DataPerSlotList_DL_Type_Dynamic NR_DRB_DataPerSlotList_DL_Type;

struct NR_L2Data_Request_Type {
	NR_DRB_DataPerSlotList_DL_Type SlotDataList;
};

struct NR_DRB_DataPerSlot_UL_Type {
	struct NR_L2DataList_Type PduSduList;
	int32_t NoOfTTIs;
};

struct NR_L2Data_Indication_Type {
	struct NR_DRB_DataPerSlot_UL_Type SlotData;
};

struct Null_Type_NR_DRB_COMMON_REQ_SuppressPdcchForC_RNTI_Optional {
	bool d;
	Null_Type v;
};

struct NR_DRB_COMMON_REQ {
	struct NR_ReqAspCommonPart_Type Common;
	struct NR_L2Data_Request_Type U_Plane;
	struct Null_Type_NR_DRB_COMMON_REQ_SuppressPdcchForC_RNTI_Optional SuppressPdcchForC_RNTI;
};

struct NR_DRB_COMMON_IND {
	struct NR_IndAspCommonPart_Type Common;
	struct NR_L2Data_Indication_Type U_Plane;
};

SIDL_END_C_INTERFACE
