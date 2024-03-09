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

struct TimingInfo_Type_NR_TimingInfo_Optional {
	bool d;
	struct TimingInfo_Type v;
};

struct NR_CellAttenuationConfig_Type {
	NR_CellId_Type CellId;
	struct NR_Attenuation_Type Attenuation;
	struct TimingInfo_Type_NR_TimingInfo_Optional TimingInfo;
};

enum NR_PdcpCountGetReq_Type_Sel {
	NR_PdcpCountGetReq_Type_UNBOUND_VALUE = 0,
	NR_PdcpCountGetReq_Type_AllRBs = 1,
	NR_PdcpCountGetReq_Type_SingleRB = 2,
};

union NR_PdcpCountGetReq_Type_Value {
	Null_Type AllRBs;
	struct NR_RadioBearerId_Type SingleRB;
};

struct NR_PdcpCountGetReq_Type {
	enum NR_PdcpCountGetReq_Type_Sel d;
	union NR_PdcpCountGetReq_Type_Value v;
};

enum NR_PDCP_CountReq_Type_Sel {
	NR_PDCP_CountReq_Type_UNBOUND_VALUE = 0,
	NR_PDCP_CountReq_Type_Get = 1,
	NR_PDCP_CountReq_Type_Set = 2,
};

union NR_PDCP_CountReq_Type_Value {
	struct NR_PdcpCountGetReq_Type Get;
	NR_PdcpCountInfoList_Type Set;
};

struct NR_PDCP_CountReq_Type {
	enum NR_PDCP_CountReq_Type_Sel d;
	union NR_PDCP_CountReq_Type_Value v;
};

enum SDAP_Header_Type {
	SDAP_Header_Type_Present = 0,
	SDAP_Header_Type_Absent = 1,
};

typedef enum SDAP_Header_Type SDAP_Header_Type;

struct int32_t_QFI_List_Type_Dynamic {
	size_t d;
	int32_t* v;
};

typedef struct int32_t_QFI_List_Type_Dynamic QFI_List_Type;

struct SDAP_Header_Type_Sdap_HeaderDL_Optional {
	bool d;
	SDAP_Header_Type v;
};

struct QFI_List_Type_MappedQoS_Flows_Optional {
	bool d;
	QFI_List_Type v;
};

struct SdapConfig_Type {
	int32_t Pdu_SessionId;
	struct SDAP_Header_Type_Sdap_HeaderDL_Optional Sdap_HeaderDL;
	struct QFI_List_Type_MappedQoS_Flows_Optional MappedQoS_Flows;
};

struct SdapTransparentMode_Type {
	SDAP_Header_Type Sdap_HeaderUL;
};

enum SdapConfigInfo_Type_Sel {
	SdapConfigInfo_Type_UNBOUND_VALUE = 0,
	SdapConfigInfo_Type_SdapConfig = 1,
	SdapConfigInfo_Type_TransparentMode = 2,
};

union SdapConfigInfo_Type_Value {
	struct SdapConfig_Type SdapConfig;
	struct SdapTransparentMode_Type TransparentMode;
};

struct SdapConfigInfo_Type {
	enum SdapConfigInfo_Type_Sel d;
	union SdapConfigInfo_Type_Value v;
};

enum SDAP_Configuration_Type_Sel {
	SDAP_Configuration_Type_UNBOUND_VALUE = 0,
	SDAP_Configuration_Type_None = 1,
	SDAP_Configuration_Type_Config = 2,
};

union SDAP_Configuration_Type_Value {
	Null_Type None;
	struct SdapConfigInfo_Type Config;
};

struct SDAP_Configuration_Type {
	enum SDAP_Configuration_Type_Sel d;
	union SDAP_Configuration_Type_Value v;
};

enum NR_PDCP_SN_Size_Type {
	NR_PDCP_SNLength12 = 0,
	NR_PDCP_SNLength18 = 1,
};

typedef enum NR_PDCP_SN_Size_Type NR_PDCP_SN_Size_Type;

enum NR_PDCP_DRB_HeaderCompression_Type_Sel {
	NR_PDCP_DRB_HeaderCompression_Type_UNBOUND_VALUE = 0,
	NR_PDCP_DRB_HeaderCompression_Type_None = 1,
};

union NR_PDCP_DRB_HeaderCompression_Type_Value {
	Null_Type None;
};

struct NR_PDCP_DRB_HeaderCompression_Type {
	enum NR_PDCP_DRB_HeaderCompression_Type_Sel d;
	union NR_PDCP_DRB_HeaderCompression_Type_Value v;
};

struct NR_PDCP_DRB_Config_Parameters_Type {
	NR_PDCP_SN_Size_Type SN_SizeUL;
	NR_PDCP_SN_Size_Type SN_SizeDL;
	struct NR_PDCP_DRB_HeaderCompression_Type HeaderCompression;
	bool IntegrityProtectionEnabled;
};

enum NR_PDCP_RB_Config_Parameters_Type_Sel {
	NR_PDCP_RB_Config_Parameters_Type_UNBOUND_VALUE = 0,
	NR_PDCP_RB_Config_Parameters_Type_Srb = 1,
	NR_PDCP_RB_Config_Parameters_Type_Drb = 2,
};

union NR_PDCP_RB_Config_Parameters_Type_Value {
	Null_Type Srb;
	struct NR_PDCP_DRB_Config_Parameters_Type Drb;
};

struct NR_PDCP_RB_Config_Parameters_Type {
	enum NR_PDCP_RB_Config_Parameters_Type_Sel d;
	union NR_PDCP_RB_Config_Parameters_Type_Value v;
};

struct NR_PDCP_Config_Parameters_Type {
	struct NR_PDCP_RB_Config_Parameters_Type Rb;
};

struct NR_PDCP_TransparentMode {
	NR_PDCP_SN_Size_Type SN_Size;
};

enum NR_PDCP_RbConfig_Type_Sel {
	NR_PDCP_RbConfig_Type_UNBOUND_VALUE = 0,
	NR_PDCP_RbConfig_Type_Params = 1,
	NR_PDCP_RbConfig_Type_TransparentMode = 2,
};

union NR_PDCP_RbConfig_Type_Value {
	struct NR_PDCP_Config_Parameters_Type Params;
	struct NR_PDCP_TransparentMode TransparentMode;
};

struct NR_PDCP_RbConfig_Type {
	enum NR_PDCP_RbConfig_Type_Sel d;
	union NR_PDCP_RbConfig_Type_Value v;
};

struct NR_PDCP_RbConfig_Type_RbConfig_Optional {
	bool d;
	struct NR_PDCP_RbConfig_Type v;
};

struct RlcBearerRouting_Type_LinkToOtherCellGroup_Optional {
	bool d;
	struct RlcBearerRouting_Type v;
};

struct NR_PDCP_RBTerminating_Type {
	struct NR_PDCP_RbConfig_Type_RbConfig_Optional RbConfig;
	struct RlcBearerRouting_Type_LinkToOtherCellGroup_Optional LinkToOtherCellGroup;
};

struct NR_PDCP_Proxy_Type {
	struct RlcBearerRouting_Type LinkToOtherNode;
};

enum NR_PDCP_Configuration_Type_Sel {
	NR_PDCP_Configuration_Type_UNBOUND_VALUE = 0,
	NR_PDCP_Configuration_Type_None = 1,
	NR_PDCP_Configuration_Type_RBTerminating = 2,
	NR_PDCP_Configuration_Type_Proxy = 3,
};

union NR_PDCP_Configuration_Type_Value {
	Null_Type None;
	struct NR_PDCP_RBTerminating_Type RBTerminating;
	struct NR_PDCP_Proxy_Type Proxy;
};

struct NR_PDCP_Configuration_Type {
	enum NR_PDCP_Configuration_Type_Sel d;
	union NR_PDCP_Configuration_Type_Value v;
};

enum NR_ASN1_UL_AM_RLC_Type_Sel {
	NR_ASN1_UL_AM_RLC_Type_UNBOUND_VALUE = 0,
	NR_ASN1_UL_AM_RLC_Type_R15 = 1,
};

union NR_ASN1_UL_AM_RLC_Type_Value {
	struct SQN_NR_UL_AM_RLC R15;
};

struct NR_ASN1_UL_AM_RLC_Type {
	enum NR_ASN1_UL_AM_RLC_Type_Sel d;
	union NR_ASN1_UL_AM_RLC_Type_Value v;
};

enum NR_ASN1_DL_AM_RLC_Type_Sel {
	NR_ASN1_DL_AM_RLC_Type_UNBOUND_VALUE = 0,
	NR_ASN1_DL_AM_RLC_Type_R15 = 1,
};

union NR_ASN1_DL_AM_RLC_Type_Value {
	struct SQN_NR_DL_AM_RLC R15;
};

struct NR_ASN1_DL_AM_RLC_Type {
	enum NR_ASN1_DL_AM_RLC_Type_Sel d;
	union NR_ASN1_DL_AM_RLC_Type_Value v;
};

struct NR_ASN1_UL_AM_RLC_Type_Tx_Optional {
	bool d;
	struct NR_ASN1_UL_AM_RLC_Type v;
};

struct NR_ASN1_DL_AM_RLC_Type_Rx_Optional {
	bool d;
	struct NR_ASN1_DL_AM_RLC_Type v;
};

struct NR_SS_RLC_AM_Type {
	struct NR_ASN1_UL_AM_RLC_Type_Tx_Optional Tx;
	struct NR_ASN1_DL_AM_RLC_Type_Rx_Optional Rx;
};

enum NR_ASN1_UL_UM_RLC_Type_Sel {
	NR_ASN1_UL_UM_RLC_Type_UNBOUND_VALUE = 0,
	NR_ASN1_UL_UM_RLC_Type_R15 = 1,
};

union NR_ASN1_UL_UM_RLC_Type_Value {
	struct SQN_NR_UL_UM_RLC R15;
};

struct NR_ASN1_UL_UM_RLC_Type {
	enum NR_ASN1_UL_UM_RLC_Type_Sel d;
	union NR_ASN1_UL_UM_RLC_Type_Value v;
};

enum NR_ASN1_DL_UM_RLC_Type_Sel {
	NR_ASN1_DL_UM_RLC_Type_UNBOUND_VALUE = 0,
	NR_ASN1_DL_UM_RLC_Type_R15 = 1,
};

union NR_ASN1_DL_UM_RLC_Type_Value {
	struct SQN_NR_DL_UM_RLC R15;
};

struct NR_ASN1_DL_UM_RLC_Type {
	enum NR_ASN1_DL_UM_RLC_Type_Sel d;
	union NR_ASN1_DL_UM_RLC_Type_Value v;
};

struct NR_ASN1_UL_UM_RLC_Type_Tx_Optional {
	bool d;
	struct NR_ASN1_UL_UM_RLC_Type v;
};

struct NR_ASN1_DL_UM_RLC_Type_Rx_Optional {
	bool d;
	struct NR_ASN1_DL_UM_RLC_Type v;
};

struct NR_SS_RLC_UM_Type {
	struct NR_ASN1_UL_UM_RLC_Type_Tx_Optional Tx;
	struct NR_ASN1_DL_UM_RLC_Type_Rx_Optional Rx;
};

typedef Null_Type NR_SS_RLC_TM_Type;

enum NR_RLC_RbConfig_Type_Sel {
	NR_RLC_RbConfig_Type_UNBOUND_VALUE = 0,
	NR_RLC_RbConfig_Type_AM = 1,
	NR_RLC_RbConfig_Type_UM = 2,
	NR_RLC_RbConfig_Type_TM = 3,
};

union NR_RLC_RbConfig_Type_Value {
	struct NR_SS_RLC_AM_Type AM;
	struct NR_SS_RLC_UM_Type UM;
	NR_SS_RLC_TM_Type TM;
};

struct NR_RLC_RbConfig_Type {
	enum NR_RLC_RbConfig_Type_Sel d;
	union NR_RLC_RbConfig_Type_Value v;
};

enum NR_RLC_ACK_Prohibit_Type {
	NR_RLC_ACK_Prohibit_Type_Prohibit = 0,
	NR_RLC_ACK_Prohibit_Type_Continue = 1,
};

typedef enum NR_RLC_ACK_Prohibit_Type NR_RLC_ACK_Prohibit_Type;

enum NR_RLC_NotACK_NextRLC_PDU_Type {
	NR_RLC_NotACK_NextRLC_PDU_Type_Start = 0,
};

typedef enum NR_RLC_NotACK_NextRLC_PDU_Type NR_RLC_NotACK_NextRLC_PDU_Type;

enum NR_RLC_TransparentMode_Sel {
	NR_RLC_TransparentMode_UNBOUND_VALUE = 0,
	NR_RLC_TransparentMode_Umd = 1,
	NR_RLC_TransparentMode_Amd = 2,
};

union NR_RLC_TransparentMode_Value {
	SQN_NR_SN_FieldLengthUM_e Umd;
	SQN_NR_SN_FieldLengthUM_e Amd;
};

struct NR_RLC_TransparentMode {
	enum NR_RLC_TransparentMode_Sel d;
	union NR_RLC_TransparentMode_Value v;
};

enum NR_RLC_TestModeInfo_Type_Sel {
	NR_RLC_TestModeInfo_Type_UNBOUND_VALUE = 0,
	NR_RLC_TestModeInfo_Type_AckProhibit = 1,
	NR_RLC_TestModeInfo_Type_NotACK_NextRLC_PDU = 2,
	NR_RLC_TestModeInfo_Type_TransparentMode = 3,
};

union NR_RLC_TestModeInfo_Type_Value {
	NR_RLC_ACK_Prohibit_Type AckProhibit;
	NR_RLC_NotACK_NextRLC_PDU_Type NotACK_NextRLC_PDU;
	struct NR_RLC_TransparentMode TransparentMode;
};

struct NR_RLC_TestModeInfo_Type {
	enum NR_RLC_TestModeInfo_Type_Sel d;
	union NR_RLC_TestModeInfo_Type_Value v;
};

enum NR_RLC_TestModeConfig_Type_Sel {
	NR_RLC_TestModeConfig_Type_UNBOUND_VALUE = 0,
	NR_RLC_TestModeConfig_Type_None = 1,
	NR_RLC_TestModeConfig_Type_Info = 2,
};

union NR_RLC_TestModeConfig_Type_Value {
	Null_Type None;
	struct NR_RLC_TestModeInfo_Type Info;
};

struct NR_RLC_TestModeConfig_Type {
	enum NR_RLC_TestModeConfig_Type_Sel d;
	union NR_RLC_TestModeConfig_Type_Value v;
};

struct NR_RLC_RbConfig_Type_Rb_Optional {
	bool d;
	struct NR_RLC_RbConfig_Type v;
};

struct NR_RLC_TestModeConfig_Type_TestMode_Optional {
	bool d;
	struct NR_RLC_TestModeConfig_Type v;
};

struct NR_RLC_Configuration_Type {
	struct NR_RLC_RbConfig_Type_Rb_Optional Rb;
	struct NR_RLC_TestModeConfig_Type_TestMode_Optional TestMode;
};

typedef SQN_NR_LogicalChannelConfig_ul_SpecificParameters_prioritisedBitRate_e NR_PrioritizedBitRate_Type;

struct NR_MAC_LogicalChannelConfig_Type {
	int32_t Priority;
	NR_PrioritizedBitRate_Type PrioritizedBitRate;
};

typedef UInt_Type NR_LogicalChannelId_Type;

struct NR_MAC_Test_DLLogChID_Type {
	NR_LogicalChannelId_Type LogChId;
	Null_Type ConfigLchId;
};

enum NR_MAC_Test_SCH_NoHeaderManipulation_Type {
	NR_MAC_Test_SCH_NoHeaderManipulation_Type_NormalMode = 0,
	NR_MAC_Test_SCH_NoHeaderManipulation_Type_DL_SCH_Only = 1,
	NR_MAC_Test_SCH_NoHeaderManipulation_Type_DL_UL_SCH = 2,
};

typedef enum NR_MAC_Test_SCH_NoHeaderManipulation_Type NR_MAC_Test_SCH_NoHeaderManipulation_Type;

struct NR_MAC_TestModeInfo_Type {
	struct NR_MAC_Test_DLLogChID_Type DiffLogChId;
	NR_MAC_Test_SCH_NoHeaderManipulation_Type No_HeaderManipulation;
};

enum NR_MAC_TestModeConfig_Type_Sel {
	NR_MAC_TestModeConfig_Type_UNBOUND_VALUE = 0,
	NR_MAC_TestModeConfig_Type_None = 1,
	NR_MAC_TestModeConfig_Type_Info = 2,
};

union NR_MAC_TestModeConfig_Type_Value {
	Null_Type None;
	struct NR_MAC_TestModeInfo_Type Info;
};

struct NR_MAC_TestModeConfig_Type {
	enum NR_MAC_TestModeConfig_Type_Sel d;
	union NR_MAC_TestModeConfig_Type_Value v;
};

struct NR_MAC_LogicalChannelConfig_Type_LogicalChannel_Optional {
	bool d;
	struct NR_MAC_LogicalChannelConfig_Type v;
};

struct NR_MAC_TestModeConfig_Type_TestMode_Optional {
	bool d;
	struct NR_MAC_TestModeConfig_Type v;
};

struct NR_MAC_Configuration_Type {
	struct NR_MAC_LogicalChannelConfig_Type_LogicalChannel_Optional LogicalChannel;
	struct NR_MAC_TestModeConfig_Type_TestMode_Optional TestMode;
};

struct NR_RLC_Configuration_Type_Rlc_Optional {
	bool d;
	struct NR_RLC_Configuration_Type v;
};

struct NR_LogicalChannelId_Type_LogicalChannelId_Optional {
	bool d;
	NR_LogicalChannelId_Type v;
};

struct NR_MAC_Configuration_Type_Mac_Optional {
	bool d;
	struct NR_MAC_Configuration_Type v;
};

struct bool_NR_RlcBearerConfigInfo_Type_DiscardULData_Optional {
	bool d;
	bool v;
};

struct NR_RlcBearerConfigInfo_Type {
	struct NR_RLC_Configuration_Type_Rlc_Optional Rlc;
	struct NR_LogicalChannelId_Type_LogicalChannelId_Optional LogicalChannelId;
	struct NR_MAC_Configuration_Type_Mac_Optional Mac;
	struct bool_NR_RlcBearerConfigInfo_Type_DiscardULData_Optional DiscardULData;
};

enum NR_RlcBearerConfig_Type_Sel {
	NR_RlcBearerConfig_Type_UNBOUND_VALUE = 0,
	NR_RlcBearerConfig_Type_Config = 1,
	NR_RlcBearerConfig_Type_None = 2,
};

union NR_RlcBearerConfig_Type_Value {
	struct NR_RlcBearerConfigInfo_Type Config;
	Null_Type None;
};

struct NR_RlcBearerConfig_Type {
	enum NR_RlcBearerConfig_Type_Sel d;
	union NR_RlcBearerConfig_Type_Value v;
};

struct SDAP_Configuration_Type_Sdap_Optional {
	bool d;
	struct SDAP_Configuration_Type v;
};

struct NR_PDCP_Configuration_Type_Pdcp_Optional {
	bool d;
	struct NR_PDCP_Configuration_Type v;
};

struct NR_RlcBearerConfig_Type_RlcBearer_Optional {
	bool d;
	struct NR_RlcBearerConfig_Type v;
};

struct NR_RadioBearerConfigInfo_Type {
	struct SDAP_Configuration_Type_Sdap_Optional Sdap;
	struct NR_PDCP_Configuration_Type_Pdcp_Optional Pdcp;
	struct NR_RlcBearerConfig_Type_RlcBearer_Optional RlcBearer;
};

enum NR_RadioBearerConfig_Type_Sel {
	NR_RadioBearerConfig_Type_UNBOUND_VALUE = 0,
	NR_RadioBearerConfig_Type_AddOrReconfigure = 1,
	NR_RadioBearerConfig_Type_Release = 2,
};

union NR_RadioBearerConfig_Type_Value {
	struct NR_RadioBearerConfigInfo_Type AddOrReconfigure;
	Null_Type Release;
};

struct NR_RadioBearerConfig_Type {
	enum NR_RadioBearerConfig_Type_Sel d;
	union NR_RadioBearerConfig_Type_Value v;
};

struct NR_RadioBearer_Type {
	struct NR_RadioBearerId_Type Id;
	struct NR_RadioBearerConfig_Type Config;
};

struct NR_PdcpSQN_Type {
	NR_PdcpCountFormat_Type Format;
	int32_t Value;
};

enum NR_PDCP_ActTime_Type_Sel {
	NR_PDCP_ActTime_Type_UNBOUND_VALUE = 0,
	NR_PDCP_ActTime_Type_None = 1,
	NR_PDCP_ActTime_Type_SQN = 2,
};

union NR_PDCP_ActTime_Type_Value {
	Null_Type None;
	struct NR_PdcpSQN_Type SQN;
};

struct NR_PDCP_ActTime_Type {
	enum NR_PDCP_ActTime_Type_Sel d;
	union NR_PDCP_ActTime_Type_Value v;
};

struct NR_SecurityActTime_Type {
	struct NR_RadioBearerId_Type RadioBearerId;
	struct NR_PDCP_ActTime_Type UL;
	struct NR_PDCP_ActTime_Type DL;
};

struct NR_SecurityActTime_Type_NR_SecurityActTimeList_Type_Dynamic {
	size_t d;
	struct NR_SecurityActTime_Type* v;
};

typedef struct NR_SecurityActTime_Type_NR_SecurityActTimeList_Type_Dynamic NR_SecurityActTimeList_Type;

struct B128_Key_Type_KUPint_Optional {
	bool d;
	B128_Key_Type v;
};

struct NR_SecurityActTimeList_Type_ActTimeList_Optional {
	bool d;
	NR_SecurityActTimeList_Type v;
};

struct NR_AS_IntegrityInfo_Type {
	SQN_NR_IntegrityProtAlgorithm_e Algorithm;
	B128_Key_Type KRRCint;
	struct B128_Key_Type_KUPint_Optional KUPint;
	struct NR_SecurityActTimeList_Type_ActTimeList_Optional ActTimeList;
};

struct NR_AS_CipheringInfo_Type {
	SQN_NR_CipheringAlgorithm_e Algorithm;
	B128_Key_Type KRRCenc;
	B128_Key_Type KUPenc;
	NR_SecurityActTimeList_Type ActTimeList;
};

struct NR_AS_IntegrityInfo_Type_Integrity_Optional {
	bool d;
	struct NR_AS_IntegrityInfo_Type v;
};

struct NR_AS_CipheringInfo_Type_Ciphering_Optional {
	bool d;
	struct NR_AS_CipheringInfo_Type v;
};

struct NR_AS_SecStartRestart_Type {
	struct NR_AS_IntegrityInfo_Type_Integrity_Optional Integrity;
	struct NR_AS_CipheringInfo_Type_Ciphering_Optional Ciphering;
};

enum NR_AS_Security_Type_Sel {
	NR_AS_Security_Type_UNBOUND_VALUE = 0,
	NR_AS_Security_Type_StartRestart = 1,
	NR_AS_Security_Type_Release = 2,
};

union NR_AS_Security_Type_Value {
	struct NR_AS_SecStartRestart_Type StartRestart;
	Null_Type Release;
};

struct NR_AS_Security_Type {
	enum NR_AS_Security_Type_Sel d;
	union NR_AS_Security_Type_Value v;
};

struct int32_t_NR_SlotOffsetList_Type_Dynamic {
	size_t d;
	int32_t* v;
};

typedef struct int32_t_NR_SlotOffsetList_Type_Dynamic NR_SlotOffsetList_Type;

struct NR_SlotOffsetList_Type_NR_PagingTrigger_Type_SlotOffsetList_Optional {
	bool d;
	NR_SlotOffsetList_Type v;
};

struct NR_PagingTrigger_Type {
	struct SQN_NR_PCCH_Message Paging;
	struct NR_SlotOffsetList_Type_NR_PagingTrigger_Type_SlotOffsetList_Optional SlotOffsetList;
};

struct NR_CellAttenuationConfig_Type_NR_CellAttenuationList_Type_Dynamic {
	size_t d;
	struct NR_CellAttenuationConfig_Type* v;
};

typedef struct NR_CellAttenuationConfig_Type_NR_CellAttenuationList_Type_Dynamic NR_CellAttenuationList_Type;

struct NR_RadioBearer_Type_NR_RadioBearerList_Type_Dynamic {
	size_t d;
	struct NR_RadioBearer_Type* v;
};

typedef struct NR_RadioBearer_Type_NR_RadioBearerList_Type_Dynamic NR_RadioBearerList_Type;

struct NR_ASN1_ARFCN_ValueNR_Type_Ssb_NRf1_Optional {
	bool d;
	struct NR_ASN1_ARFCN_ValueNR_Type v;
};

struct NR_ASN1_ARFCN_ValueNR_Type_Ssb_NRf2_Optional {
	bool d;
	struct NR_ASN1_ARFCN_ValueNR_Type v;
};

struct NR_ASN1_ARFCN_ValueNR_Type_Ssb_NRf3_Optional {
	bool d;
	struct NR_ASN1_ARFCN_ValueNR_Type v;
};

struct NR_ASN1_ARFCN_ValueNR_Type_Ssb_NRf4_Optional {
	bool d;
	struct NR_ASN1_ARFCN_ValueNR_Type v;
};

struct Band_SsbInfo_Type {
	SQN_NR_FreqBandIndicatorNR DeltaBand;
	struct NR_ASN1_ARFCN_ValueNR_Type_Ssb_NRf1_Optional Ssb_NRf1;
	struct NR_ASN1_ARFCN_ValueNR_Type_Ssb_NRf2_Optional Ssb_NRf2;
	struct NR_ASN1_ARFCN_ValueNR_Type_Ssb_NRf3_Optional Ssb_NRf3;
	struct NR_ASN1_ARFCN_ValueNR_Type_Ssb_NRf4_Optional Ssb_NRf4;
};

struct NR_Band_SsbForDelta_Type {
	struct Band_SsbInfo_Type DeltaPrimary;
	struct Band_SsbInfo_Type DeltaSecondary;
};

struct NR_PDCCH_Order_Type {
	B6_Type RA_PreambleIndex;
	struct NR_DciCommon_UL_SUL_Indicator_Type UL_SUL_Indicator;
	B6_Type SSB_Index;
	B4_Type PrachMaskIndex;
};

struct NR_SlotOffsetList_Type_NR_DciWithShortMessageOnly_Type_SlotOffsetList_Optional {
	bool d;
	NR_SlotOffsetList_Type v;
};

struct NR_DciWithShortMessageOnly_Type {
	B2_Type ShortMessageIndicator;
	B8_Type ShortMessages;
	struct NR_SlotOffsetList_Type_NR_DciWithShortMessageOnly_Type_SlotOffsetList_Optional SlotOffsetList;
};

struct BIT_STRING_NR_DciFormat_2_0_SfiList_Type_Dynamic {
	size_t d;
	BIT_STRING* v;
};

typedef struct BIT_STRING_NR_DciFormat_2_0_SfiList_Type_Dynamic NR_DciFormat_2_0_SfiList_Type;

struct NR_DciFormat_2_0_Type {
	NR_DciFormat_2_0_SfiList_Type SfiList;
};

struct B14_Type_NR_DciFormat_2_1_IntValueList_Type_Dynamic {
	size_t d;
	B14_Type* v;
};

typedef struct B14_Type_NR_DciFormat_2_1_IntValueList_Type_Dynamic NR_DciFormat_2_1_IntValueList_Type;

struct NR_DciFormat_2_1_Type {
	NR_DciFormat_2_1_IntValueList_Type IntValueList;
};

struct NR_DciFormat_2_2_ClosedLoopIndicator_Type {
	Null_Type None;
	B1_Type Index;
};

enum NR_DciFormat_2_2_TpcBlock_Type_Sel {
	NR_DciFormat_2_2_TpcBlock_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_2_2_TpcBlock_Type_ClosedLoopIndicator = 1,
	NR_DciFormat_2_2_TpcBlock_Type_TpcCommand = 2,
};

union NR_DciFormat_2_2_TpcBlock_Type_Value {
	struct NR_DciFormat_2_2_ClosedLoopIndicator_Type ClosedLoopIndicator;
	struct NR_DciCommon_TpcCommand_Type TpcCommand;
};

struct NR_DciFormat_2_2_TpcBlock_Type {
	enum NR_DciFormat_2_2_TpcBlock_Type_Sel d;
	union NR_DciFormat_2_2_TpcBlock_Type_Value v;
};

struct NR_DciFormat_2_2_TpcBlock_Type_NR_DciFormat_2_2_TpcBlockList_Type_Dynamic {
	size_t d;
	struct NR_DciFormat_2_2_TpcBlock_Type* v;
};

typedef struct NR_DciFormat_2_2_TpcBlock_Type_NR_DciFormat_2_2_TpcBlockList_Type_Dynamic NR_DciFormat_2_2_TpcBlockList_Type;

struct NR_DciFormat_2_2_Type {
	NR_DciFormat_2_2_TpcBlockList_Type TpcBlockList;
};

struct NR_DciFormat_2_3_SrsRequest_Type {
	Null_Type None;
	B2_Type SrsRequestValue;
};

struct NR_DciCommon_TpcCommand_Type_NR_DciCommon_TpcCommandList_Type_Dynamic {
	size_t d;
	struct NR_DciCommon_TpcCommand_Type* v;
};

typedef struct NR_DciCommon_TpcCommand_Type_NR_DciCommon_TpcCommandList_Type_Dynamic NR_DciCommon_TpcCommandList_Type;

struct NR_DciFormat_2_3_TypeA_Type {
	struct NR_DciFormat_2_3_SrsRequest_Type SrsRequest;
	NR_DciCommon_TpcCommandList_Type TpcCommandList;
};

struct NR_DciFormat_2_3_SingleBlockTypeB_Type {
	struct NR_DciFormat_2_3_SrsRequest_Type SrsRequest;
	struct NR_DciCommon_TpcCommand_Type TpcCommand;
};

struct NR_DciFormat_2_3_SingleBlockTypeB_Type_NR_DciFormat_2_3_TypeB_Type_Dynamic {
	size_t d;
	struct NR_DciFormat_2_3_SingleBlockTypeB_Type* v;
};

typedef struct NR_DciFormat_2_3_SingleBlockTypeB_Type_NR_DciFormat_2_3_TypeB_Type_Dynamic NR_DciFormat_2_3_TypeB_Type;

enum NR_DciFormat_2_3_TypeA_B_Type_Sel {
	NR_DciFormat_2_3_TypeA_B_Type_UNBOUND_VALUE = 0,
	NR_DciFormat_2_3_TypeA_B_Type_TypeA = 1,
	NR_DciFormat_2_3_TypeA_B_Type_TypeB = 2,
};

union NR_DciFormat_2_3_TypeA_B_Type_Value {
	struct NR_DciFormat_2_3_TypeA_Type TypeA;
	NR_DciFormat_2_3_TypeB_Type TypeB;
};

struct NR_DciFormat_2_3_TypeA_B_Type {
	enum NR_DciFormat_2_3_TypeA_B_Type_Sel d;
	union NR_DciFormat_2_3_TypeA_B_Type_Value v;
};

struct NR_DciFormat_2_3_Type {
	struct NR_DciFormat_2_3_TypeA_B_Type TypeA_B;
};

struct NR_DciFormat_2_6_Block_Type {
	B1_Type WakeupIndication;
	struct NR_DciFormat_X_1_SCellDormancyIndication_Type SCellDormacyIndication;
};

struct NR_DciFormat_2_6_Block_Type_NR_DciFormat_2_6_Block_List_Type_Dynamic {
	size_t d;
	struct NR_DciFormat_2_6_Block_Type* v;
};

typedef struct NR_DciFormat_2_6_Block_Type_NR_DciFormat_2_6_Block_List_Type_Dynamic NR_DciFormat_2_6_Block_List_Type;

struct NR_DciFormat_2_6_Type {
	NR_DciFormat_2_6_Block_List_Type BlockList;
};

enum NR_DCI_TriggerFormat_Type_Sel {
	NR_DCI_TriggerFormat_Type_UNBOUND_VALUE = 0,
	NR_DCI_TriggerFormat_Type_PdcchOrder = 1,
	NR_DCI_TriggerFormat_Type_ShortMessage = 2,
	NR_DCI_TriggerFormat_Type_DciFormat_2_0 = 3,
	NR_DCI_TriggerFormat_Type_DciFormat_2_1 = 4,
	NR_DCI_TriggerFormat_Type_DciFormat_2_2 = 5,
	NR_DCI_TriggerFormat_Type_DciFormat_2_3 = 6,
	NR_DCI_TriggerFormat_Type_DciFormat_2_6 = 7,
};

union NR_DCI_TriggerFormat_Type_Value {
	struct NR_PDCCH_Order_Type PdcchOrder;
	struct NR_DciWithShortMessageOnly_Type ShortMessage;
	struct NR_DciFormat_2_0_Type DciFormat_2_0;
	struct NR_DciFormat_2_1_Type DciFormat_2_1;
	struct NR_DciFormat_2_2_Type DciFormat_2_2;
	struct NR_DciFormat_2_3_Type DciFormat_2_3;
	struct NR_DciFormat_2_6_Type DciFormat_2_6;
};

struct NR_DCI_TriggerFormat_Type {
	enum NR_DCI_TriggerFormat_Type_Sel d;
	union NR_DCI_TriggerFormat_Type_Value v;
};

struct NR_DCI_Trigger_Type {
	struct NR_AssignedBWPs_Type AssignedBWPs;
	NR_SearchSpaceType_Type SearchSpaceType;
	struct NR_DCI_TriggerFormat_Type DciFormat;
};

enum NR_SystemRequest_Type_Sel {
	NR_SystemRequest_Type_UNBOUND_VALUE = 0,
	NR_SystemRequest_Type_Cell = 1,
	NR_SystemRequest_Type_CellAttenuationList = 2,
	NR_SystemRequest_Type_RadioBearerList = 3,
	NR_SystemRequest_Type_EnquireTiming = 4,
	NR_SystemRequest_Type_AS_Security = 5,
	NR_SystemRequest_Type_PdcpCount = 7,
	NR_SystemRequest_Type_DciTrigger = 8,
	NR_SystemRequest_Type_Paging = 9,
	NR_SystemRequest_Type_DeltaValues = 13,
};

union NR_SystemRequest_Type_Value {
	struct NR_CellConfigRequest_Type Cell;
	NR_CellAttenuationList_Type CellAttenuationList;
	NR_RadioBearerList_Type RadioBearerList;
	Null_Type EnquireTiming;
	struct NR_AS_Security_Type AS_Security;
	struct NR_PDCP_CountReq_Type PdcpCount;
	struct NR_DCI_Trigger_Type DciTrigger;
	struct NR_PagingTrigger_Type Paging;
	struct NR_Band_SsbForDelta_Type DeltaValues;
};

struct NR_SystemRequest_Type {
	enum NR_SystemRequest_Type_Sel d;
	union NR_SystemRequest_Type_Value v;
};

enum NR_SystemConfirm_Type_Sel {
	NR_SystemConfirm_Type_UNBOUND_VALUE = 0,
	NR_SystemConfirm_Type_Cell = 1,
	NR_SystemConfirm_Type_CellAttenuationList = 2,
	NR_SystemConfirm_Type_RadioBearerList = 3,
	NR_SystemConfirm_Type_EnquireTiming = 4,
	NR_SystemConfirm_Type_AS_Security = 5,
	NR_SystemConfirm_Type_SystemIndCtrl = 6,
	NR_SystemConfirm_Type_PdcpCount = 7,
	NR_SystemConfirm_Type_DciTrigger = 8,
	NR_SystemConfirm_Type_MacCommandTrigger = 9,
	NR_SystemConfirm_Type_L1_TestMode = 10,
	NR_SystemConfirm_Type_PdcpHandoverControl = 11,
	NR_SystemConfirm_Type_DeltaValues = 12,
	NR_SystemConfirm_Type_SpsCg = 13,
};

union NR_SystemConfirm_Type_Value {
	Null_Type Cell;
	Null_Type CellAttenuationList;
	Null_Type RadioBearerList;
	Null_Type EnquireTiming;
	Null_Type AS_Security;
	Null_Type SystemIndCtrl;
	struct NR_PDCP_CountCnf_Type PdcpCount;
	Null_Type DciTrigger;
	Null_Type MacCommandTrigger;
	Null_Type L1_TestMode;
	Null_Type PdcpHandoverControl;
	struct UE_NR_DeltaValues_Type DeltaValues;
	Null_Type SpsCg;
};

struct NR_SystemConfirm_Type {
	enum NR_SystemConfirm_Type_Sel d;
	union NR_SystemConfirm_Type_Value v;
};

struct NR_SYSTEM_CTRL_REQ {
	struct NR_ReqAspCommonPart_Type Common;
	struct NR_SystemRequest_Type Request;
};

struct NR_SYSTEM_CTRL_CNF {
	struct NR_CnfAspCommonPart_Type Common;
	struct NR_SystemConfirm_Type Confirm;
};

SIDL_END_C_INTERFACE
