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

#include "adbgUtilsSys.h"

const char* adbgUtilsSysPdcpCountGetReq_TypeToStr(int select)
{
	switch (select) {
		case PdcpCountGetReq_Type_AllRBs: return "AllRBs";
		case PdcpCountGetReq_Type_SingleRB: return "SingleRB";
		default: return "unknown";
	}
}

const char* adbgUtilsSysPDCP_CountReq_TypeToStr(int select)
{
	switch (select) {
		case PDCP_CountReq_Type_Get: return "Get";
		case PDCP_CountReq_Type_Set: return "Set";
		default: return "unknown";
	}
}

const char* adbgUtilsSysPDCP_TestModeInfo_TypeToStr(int select)
{
	switch (select) {
		case PDCP_TestModeInfo_Type_PDCP_ROHC_Mode: return "PDCP_ROHC_Mode";
		case PDCP_TestModeInfo_Type_PDCP_NonROHC_Mode: return "PDCP_NonROHC_Mode";
		default: return "unknown";
	}
}

const char* adbgUtilsSysPDCP_TestModeConfig_TypeToStr(int select)
{
	switch (select) {
		case PDCP_TestModeConfig_Type_None: return "None";
		case PDCP_TestModeConfig_Type_Info: return "Info";
		default: return "unknown";
	}
}

const char* adbgUtilsSysPDCP_Config_TypeToStr(int select)
{
	switch (select) {
		case PDCP_Config_Type_R8: return "R8";
		default: return "unknown";
	}
}

const char* adbgUtilsSysPDCP_RBConfig_TypeToStr(int select)
{
	switch (select) {
		case PDCP_RBConfig_Type_Srb: return "Srb";
		case PDCP_RBConfig_Type_Drb: return "Drb";
		case PDCP_RBConfig_Type_Transparent: return "Transparent";
		default: return "unknown";
	}
}

const char* adbgUtilsSysPDCP_Configuration_TypeToStr(int select)
{
	switch (select) {
		case PDCP_Configuration_Type_None: return "None";
		case PDCP_Configuration_Type_Config: return "Config";
		default: return "unknown";
	}
}

const char* adbgUtilsSysUL_AM_RLC_TypeToStr(int select)
{
	switch (select) {
		case UL_AM_RLC_Type_R8: return "R8";
		default: return "unknown";
	}
}

const char* adbgUtilsSysDL_AM_RLC_TypeToStr(int select)
{
	switch (select) {
		case DL_AM_RLC_Type_R8: return "R8";
		default: return "unknown";
	}
}

const char* adbgUtilsSysUL_UM_RLC_TypeToStr(int select)
{
	switch (select) {
		case UL_UM_RLC_Type_R8: return "R8";
		default: return "unknown";
	}
}

const char* adbgUtilsSysDL_UM_RLC_TypeToStr(int select)
{
	switch (select) {
		case DL_UM_RLC_Type_R8: return "R8";
		default: return "unknown";
	}
}

const char* adbgUtilsSysRLC_RbConfig_TypeToStr(int select)
{
	switch (select) {
		case RLC_RbConfig_Type_AM: return "AM";
		case RLC_RbConfig_Type_UM: return "UM";
		case RLC_RbConfig_Type_UM_OnlyUL: return "UM_OnlyUL";
		case RLC_RbConfig_Type_UM_OnlyDL: return "UM_OnlyDL";
		case RLC_RbConfig_Type_TM: return "TM";
		default: return "unknown";
	}
}

const char* adbgUtilsSysRLC_TestModeInfo_TypeToStr(int select)
{
	switch (select) {
		case RLC_TestModeInfo_Type_AckProhibit: return "AckProhibit";
		case RLC_TestModeInfo_Type_NotACK_NextRLC_PDU: return "NotACK_NextRLC_PDU";
		case RLC_TestModeInfo_Type_ModifyVTS: return "ModifyVTS";
		case RLC_TestModeInfo_Type_TransparentMode_UMDwith5BitSN: return "TransparentMode_UMDwith5BitSN";
		case RLC_TestModeInfo_Type_TransparentMode_UMDwith10BitSN: return "TransparentMode_UMDwith10BitSN";
		case RLC_TestModeInfo_Type_TransparentMode_AMD: return "TransparentMode_AMD";
		default: return "unknown";
	}
}

const char* adbgUtilsSysRLC_TestModeConfig_TypeToStr(int select)
{
	switch (select) {
		case RLC_TestModeConfig_Type_None: return "None";
		case RLC_TestModeConfig_Type_Info: return "Info";
		default: return "unknown";
	}
}

const char* adbgUtilsSysMAC_Test_DLLogChID_TypeToStr(int select)
{
	switch (select) {
		case MAC_Test_DLLogChID_Type_LogChId: return "LogChId";
		case MAC_Test_DLLogChID_Type_ConfigLchId: return "ConfigLchId";
		default: return "unknown";
	}
}

const char* adbgUtilsSysMAC_TestModeConfig_TypeToStr(int select)
{
	switch (select) {
		case MAC_TestModeConfig_Type_None: return "None";
		case MAC_TestModeConfig_Type_Info: return "Info";
		default: return "unknown";
	}
}

const char* adbgUtilsSysRadioBearerConfig_TypeToStr(int select)
{
	switch (select) {
		case RadioBearerConfig_Type_AddOrReconfigure: return "AddOrReconfigure";
		case RadioBearerConfig_Type_Release: return "Release";
		default: return "unknown";
	}
}

const char* adbgUtilsSysPDCP_ActTime_TypeToStr(int select)
{
	switch (select) {
		case PDCP_ActTime_Type_None: return "None";
		case PDCP_ActTime_Type_SQN: return "SQN";
		default: return "unknown";
	}
}

const char* adbgUtilsSysAS_Security_TypeToStr(int select)
{
	switch (select) {
		case AS_Security_Type_StartRestart: return "StartRestart";
		case AS_Security_Type_Release: return "Release";
		default: return "unknown";
	}
}

const char* adbgUtilsSysPDCP_HandoverControlReq_TypeToStr(int select)
{
	switch (select) {
		case PDCP_HandoverControlReq_Type_HandoverInit: return "HandoverInit";
		case PDCP_HandoverControlReq_Type_HandoverComplete: return "HandoverComplete";
		default: return "unknown";
	}
}

const char* adbgUtilsSysSystemRequest_TypeToStr(int select)
{
	switch (select) {
		case SystemRequest_Type_Cell: return "Cell";
		case SystemRequest_Type_CellAttenuationList: return "CellAttenuationList";
		case SystemRequest_Type_RadioBearerList: return "RadioBearerList";
		case SystemRequest_Type_EnquireTiming: return "EnquireTiming";
		case SystemRequest_Type_AS_Security: return "AS_Security";
		case SystemRequest_Type_Paging: return "Paging";
		case SystemRequest_Type_L1MacIndCtrl: return "L1MacIndCtrl";
		case SystemRequest_Type_PdcpCount: return "PdcpCount";
		case SystemRequest_Type_PdcpHandoverControl: return "PdcpHandoverControl";
		case SystemRequest_Type_PdcchOrder: return "PdcchOrder";
		case SystemRequest_Type_UE_Cat_Info: return "UE_Cat_Info";
		default: return "unknown";
	}
}

const char* adbgUtilsSysSystemConfirm_TypeToStr(int select)
{
	switch (select) {
		case SystemConfirm_Type_Cell: return "Cell";
		case SystemConfirm_Type_CellAttenuationList: return "CellAttenuationList";
		case SystemConfirm_Type_RadioBearerList: return "RadioBearerList";
		case SystemConfirm_Type_EnquireTiming: return "EnquireTiming";
		case SystemConfirm_Type_AS_Security: return "AS_Security";
		case SystemConfirm_Type_Sps: return "Sps";
		case SystemConfirm_Type_Paging: return "Paging";
		case SystemConfirm_Type_L1MacIndCtrl: return "L1MacIndCtrl";
		case SystemConfirm_Type_RlcIndCtrl: return "RlcIndCtrl";
		case SystemConfirm_Type_PdcpCount: return "PdcpCount";
		case SystemConfirm_Type_PdcpHandoverControl: return "PdcpHandoverControl";
		case SystemConfirm_Type_L1_TestMode: return "L1_TestMode";
		case SystemConfirm_Type_PdcchOrder: return "PdcchOrder";
		case SystemConfirm_Type_ActivateScell: return "ActivateScell";
		case SystemConfirm_Type_MbmsConfig: return "MbmsConfig";
		case SystemConfirm_Type_PDCCH_MCCH_ChangeNotification: return "PDCCH_MCCH_ChangeNotification";
		case SystemConfirm_Type_MSI_Config: return "MSI_Config";
		case SystemConfirm_Type_UE_Cat_Info: return "UE_Cat_Info";
		case SystemConfirm_Type_OCNG_Config: return "OCNG_Config";
		case SystemConfirm_Type_DirectIndicationInfo: return "DirectIndicationInfo";
		default: return "unknown";
	}
}
