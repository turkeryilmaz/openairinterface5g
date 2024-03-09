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

#include "adbgUtilsNrSys.h"

const char* adbgUtilsNrSysNR_PdcpCountGetReq_TypeToStr(int select)
{
	switch (select) {
		case NR_PdcpCountGetReq_Type_AllRBs: return "AllRBs";
		case NR_PdcpCountGetReq_Type_SingleRB: return "SingleRB";
		default: return "unknown";
	}
}

const char* adbgUtilsNrSysNR_PDCP_CountReq_TypeToStr(int select)
{
	switch (select) {
		case NR_PDCP_CountReq_Type_Get: return "Get";
		case NR_PDCP_CountReq_Type_Set: return "Set";
		default: return "unknown";
	}
}

const char* adbgUtilsNrSysSdapConfigInfo_TypeToStr(int select)
{
	switch (select) {
		case SdapConfigInfo_Type_SdapConfig: return "SdapConfig";
		case SdapConfigInfo_Type_TransparentMode: return "TransparentMode";
		default: return "unknown";
	}
}

const char* adbgUtilsNrSysSDAP_Configuration_TypeToStr(int select)
{
	switch (select) {
		case SDAP_Configuration_Type_None: return "None";
		case SDAP_Configuration_Type_Config: return "Config";
		default: return "unknown";
	}
}

const char* adbgUtilsNrSysNR_PDCP_DRB_HeaderCompression_TypeToStr(int select)
{
	switch (select) {
		case NR_PDCP_DRB_HeaderCompression_Type_None: return "None";
		default: return "unknown";
	}
}

const char* adbgUtilsNrSysNR_PDCP_RB_Config_Parameters_TypeToStr(int select)
{
	switch (select) {
		case NR_PDCP_RB_Config_Parameters_Type_Srb: return "Srb";
		case NR_PDCP_RB_Config_Parameters_Type_Drb: return "Drb";
		default: return "unknown";
	}
}

const char* adbgUtilsNrSysNR_PDCP_RbConfig_TypeToStr(int select)
{
	switch (select) {
		case NR_PDCP_RbConfig_Type_Params: return "Params";
		case NR_PDCP_RbConfig_Type_TransparentMode: return "TransparentMode";
		default: return "unknown";
	}
}

const char* adbgUtilsNrSysNR_PDCP_Configuration_TypeToStr(int select)
{
	switch (select) {
		case NR_PDCP_Configuration_Type_None: return "None";
		case NR_PDCP_Configuration_Type_RBTerminating: return "RBTerminating";
		case NR_PDCP_Configuration_Type_Proxy: return "Proxy";
		default: return "unknown";
	}
}

const char* adbgUtilsNrSysNR_ASN1_UL_AM_RLC_TypeToStr(int select)
{
	switch (select) {
		case NR_ASN1_UL_AM_RLC_Type_R15: return "R15";
		default: return "unknown";
	}
}

const char* adbgUtilsNrSysNR_ASN1_DL_AM_RLC_TypeToStr(int select)
{
	switch (select) {
		case NR_ASN1_DL_AM_RLC_Type_R15: return "R15";
		default: return "unknown";
	}
}

const char* adbgUtilsNrSysNR_ASN1_UL_UM_RLC_TypeToStr(int select)
{
	switch (select) {
		case NR_ASN1_UL_UM_RLC_Type_R15: return "R15";
		default: return "unknown";
	}
}

const char* adbgUtilsNrSysNR_ASN1_DL_UM_RLC_TypeToStr(int select)
{
	switch (select) {
		case NR_ASN1_DL_UM_RLC_Type_R15: return "R15";
		default: return "unknown";
	}
}

const char* adbgUtilsNrSysNR_RLC_RbConfig_TypeToStr(int select)
{
	switch (select) {
		case NR_RLC_RbConfig_Type_AM: return "AM";
		case NR_RLC_RbConfig_Type_UM: return "UM";
		case NR_RLC_RbConfig_Type_TM: return "TM";
		default: return "unknown";
	}
}

const char* adbgUtilsNrSysNR_RLC_TransparentModeToStr(int select)
{
	switch (select) {
		case NR_RLC_TransparentMode_Umd: return "Umd";
		case NR_RLC_TransparentMode_Amd: return "Amd";
		default: return "unknown";
	}
}

const char* adbgUtilsNrSysNR_RLC_TestModeInfo_TypeToStr(int select)
{
	switch (select) {
		case NR_RLC_TestModeInfo_Type_AckProhibit: return "AckProhibit";
		case NR_RLC_TestModeInfo_Type_NotACK_NextRLC_PDU: return "NotACK_NextRLC_PDU";
		case NR_RLC_TestModeInfo_Type_TransparentMode: return "TransparentMode";
		default: return "unknown";
	}
}

const char* adbgUtilsNrSysNR_RLC_TestModeConfig_TypeToStr(int select)
{
	switch (select) {
		case NR_RLC_TestModeConfig_Type_None: return "None";
		case NR_RLC_TestModeConfig_Type_Info: return "Info";
		default: return "unknown";
	}
}

const char* adbgUtilsNrSysNR_MAC_TestModeConfig_TypeToStr(int select)
{
	switch (select) {
		case NR_MAC_TestModeConfig_Type_None: return "None";
		case NR_MAC_TestModeConfig_Type_Info: return "Info";
		default: return "unknown";
	}
}

const char* adbgUtilsNrSysNR_RlcBearerConfig_TypeToStr(int select)
{
	switch (select) {
		case NR_RlcBearerConfig_Type_Config: return "Config";
		case NR_RlcBearerConfig_Type_None: return "None";
		default: return "unknown";
	}
}

const char* adbgUtilsNrSysNR_RadioBearerConfig_TypeToStr(int select)
{
	switch (select) {
		case NR_RadioBearerConfig_Type_AddOrReconfigure: return "AddOrReconfigure";
		case NR_RadioBearerConfig_Type_Release: return "Release";
		default: return "unknown";
	}
}

const char* adbgUtilsNrSysNR_PDCP_ActTime_TypeToStr(int select)
{
	switch (select) {
		case NR_PDCP_ActTime_Type_None: return "None";
		case NR_PDCP_ActTime_Type_SQN: return "SQN";
		default: return "unknown";
	}
}

const char* adbgUtilsNrSysNR_AS_Security_TypeToStr(int select)
{
	switch (select) {
		case NR_AS_Security_Type_StartRestart: return "StartRestart";
		case NR_AS_Security_Type_Release: return "Release";
		default: return "unknown";
	}
}

const char* adbgUtilsNrSysNR_DciFormat_2_2_TpcBlock_TypeToStr(int select)
{
	switch (select) {
		case NR_DciFormat_2_2_TpcBlock_Type_ClosedLoopIndicator: return "ClosedLoopIndicator";
		case NR_DciFormat_2_2_TpcBlock_Type_TpcCommand: return "TpcCommand";
		default: return "unknown";
	}
}

const char* adbgUtilsNrSysNR_DciFormat_2_3_TypeA_B_TypeToStr(int select)
{
	switch (select) {
		case NR_DciFormat_2_3_TypeA_B_Type_TypeA: return "TypeA";
		case NR_DciFormat_2_3_TypeA_B_Type_TypeB: return "TypeB";
		default: return "unknown";
	}
}

const char* adbgUtilsNrSysNR_DCI_TriggerFormat_TypeToStr(int select)
{
	switch (select) {
		case NR_DCI_TriggerFormat_Type_PdcchOrder: return "PdcchOrder";
		case NR_DCI_TriggerFormat_Type_ShortMessage: return "ShortMessage";
		case NR_DCI_TriggerFormat_Type_DciFormat_2_0: return "DciFormat_2_0";
		case NR_DCI_TriggerFormat_Type_DciFormat_2_1: return "DciFormat_2_1";
		case NR_DCI_TriggerFormat_Type_DciFormat_2_2: return "DciFormat_2_2";
		case NR_DCI_TriggerFormat_Type_DciFormat_2_3: return "DciFormat_2_3";
		case NR_DCI_TriggerFormat_Type_DciFormat_2_6: return "DciFormat_2_6";
		default: return "unknown";
	}
}

const char* adbgUtilsNrSysNR_SystemRequest_TypeToStr(int select)
{
	switch (select) {
		case NR_SystemRequest_Type_Cell: return "Cell";
		case NR_SystemRequest_Type_CellAttenuationList: return "CellAttenuationList";
		case NR_SystemRequest_Type_RadioBearerList: return "RadioBearerList";
		case NR_SystemRequest_Type_EnquireTiming: return "EnquireTiming";
		case NR_SystemRequest_Type_AS_Security: return "AS_Security";
		case NR_SystemRequest_Type_PdcpCount: return "PdcpCount";
		case NR_SystemRequest_Type_DciTrigger: return "DciTrigger";
		case NR_SystemRequest_Type_Paging: return "Paging";
		case NR_SystemRequest_Type_DeltaValues: return "DeltaValues";
		default: return "unknown";
	}
}

const char* adbgUtilsNrSysNR_SystemConfirm_TypeToStr(int select)
{
	switch (select) {
		case NR_SystemConfirm_Type_Cell: return "Cell";
		case NR_SystemConfirm_Type_CellAttenuationList: return "CellAttenuationList";
		case NR_SystemConfirm_Type_RadioBearerList: return "RadioBearerList";
		case NR_SystemConfirm_Type_EnquireTiming: return "EnquireTiming";
		case NR_SystemConfirm_Type_AS_Security: return "AS_Security";
		case NR_SystemConfirm_Type_SystemIndCtrl: return "SystemIndCtrl";
		case NR_SystemConfirm_Type_PdcpCount: return "PdcpCount";
		case NR_SystemConfirm_Type_DciTrigger: return "DciTrigger";
		case NR_SystemConfirm_Type_MacCommandTrigger: return "MacCommandTrigger";
		case NR_SystemConfirm_Type_L1_TestMode: return "L1_TestMode";
		case NR_SystemConfirm_Type_PdcpHandoverControl: return "PdcpHandoverControl";
		case NR_SystemConfirm_Type_DeltaValues: return "DeltaValues";
		case NR_SystemConfirm_Type_SpsCg: return "SpsCg";
		default: return "unknown";
	}
}
