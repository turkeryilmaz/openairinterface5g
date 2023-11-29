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

#include "adbgUtilsNrDrb.h"

const char* adbgUtilsNrDrbNR_HarqProcessAssignment_TypeToStr(int select)
{
	switch (select) {
		case NR_HarqProcessAssignment_Type_Id: return "Id";
		case NR_HarqProcessAssignment_Type_Automatic: return "Automatic";
		default: return "unknown";
	}
}

const char* adbgUtilsNrDrbNR_MAC_ControlElementDL_TypeToStr(int select)
{
	switch (select) {
		case NR_MAC_ControlElementDL_Type_ContentionResolutionID: return "ContentionResolutionID";
		case NR_MAC_ControlElementDL_Type_TimingAdvance: return "TimingAdvance";
		case NR_MAC_ControlElementDL_Type_SCellActDeact: return "SCellActDeact";
		case NR_MAC_ControlElementDL_Type_DuplicationActDeact: return "DuplicationActDeact";
		case NR_MAC_ControlElementDL_Type_SP_ResourceSetActDeact: return "SP_ResourceSetActDeact";
		case NR_MAC_ControlElementDL_Type_CSI_TriggerStateSubselection: return "CSI_TriggerStateSubselection";
		case NR_MAC_ControlElementDL_Type_TCI_StatesActDeact: return "TCI_StatesActDeact";
		case NR_MAC_ControlElementDL_Type_TCI_StateIndication: return "TCI_StateIndication";
		case NR_MAC_ControlElementDL_Type_SP_CSI_ReportingActDeact: return "SP_CSI_ReportingActDeact";
		case NR_MAC_ControlElementDL_Type_SP_SRS_ActDeact: return "SP_SRS_ActDeact";
		case NR_MAC_ControlElementDL_Type_PUCCH_SpatialRelationActDeact: return "PUCCH_SpatialRelationActDeact";
		case NR_MAC_ControlElementDL_Type_SP_ZP_ResourceSetActDeact: return "SP_ZP_ResourceSetActDeact";
		case NR_MAC_ControlElementDL_Type_RecommendatdBitrate: return "RecommendatdBitrate";
		default: return "unknown";
	}
}

const char* adbgUtilsNrDrbNR_MAC_ControlElementUL_TypeToStr(int select)
{
	switch (select) {
		case NR_MAC_ControlElementUL_Type_ShortBSR: return "ShortBSR";
		case NR_MAC_ControlElementUL_Type_LongBSR: return "LongBSR";
		case NR_MAC_ControlElementUL_Type_C_RNTI: return "C_RNTI";
		case NR_MAC_ControlElementUL_Type_SingleEntryPHR: return "SingleEntryPHR";
		case NR_MAC_ControlElementUL_Type_MultiEntryPHR: return "MultiEntryPHR";
		case NR_MAC_ControlElementUL_Type_RecommendedBitrate: return "RecommendedBitrate";
		default: return "unknown";
	}
}

const char* adbgUtilsNrDrbNR_MAC_PDU_TypeToStr(int select)
{
	switch (select) {
		case NR_MAC_PDU_Type_DL: return "DL";
		case NR_MAC_PDU_Type_UL: return "UL";
		default: return "unknown";
	}
}

const char* adbgUtilsNrDrbNR_RLC_UMD_PDU_TypeToStr(int select)
{
	switch (select) {
		case NR_RLC_UMD_PDU_Type_NoSN: return "NoSN";
		case NR_RLC_UMD_PDU_Type_SN6Bit: return "SN6Bit";
		case NR_RLC_UMD_PDU_Type_SN12Bit: return "SN12Bit";
		default: return "unknown";
	}
}

const char* adbgUtilsNrDrbNR_RLC_AMD_PDU_TypeToStr(int select)
{
	switch (select) {
		case NR_RLC_AMD_PDU_Type_SN12Bit: return "SN12Bit";
		case NR_RLC_AMD_PDU_Type_SN18Bit: return "SN18Bit";
		default: return "unknown";
	}
}

const char* adbgUtilsNrDrbNR_RLC_AM_StatusPDU_TypeToStr(int select)
{
	switch (select) {
		case NR_RLC_AM_StatusPDU_Type_SN12Bit: return "SN12Bit";
		case NR_RLC_AM_StatusPDU_Type_SN18Bit: return "SN18Bit";
		default: return "unknown";
	}
}

const char* adbgUtilsNrDrbNR_RLC_PDU_TypeToStr(int select)
{
	switch (select) {
		case NR_RLC_PDU_Type_TMD: return "TMD";
		case NR_RLC_PDU_Type_UMD: return "UMD";
		case NR_RLC_PDU_Type_AMD: return "AMD";
		case NR_RLC_PDU_Type_Status: return "Status";
		default: return "unknown";
	}
}

const char* adbgUtilsNrDrbNR_PDCP_PDU_TypeToStr(int select)
{
	switch (select) {
		case NR_PDCP_PDU_Type_DataPduSN12Bits: return "DataPduSN12Bits";
		case NR_PDCP_PDU_Type_DataPduSN18Bits: return "DataPduSN18Bits";
		case NR_PDCP_PDU_Type_CtrlPduStatus: return "CtrlPduStatus";
		case NR_PDCP_PDU_Type_CtrlPduRohcFeedback: return "CtrlPduRohcFeedback";
		default: return "unknown";
	}
}

const char* adbgUtilsNrDrbSDAP_PDU_TypeToStr(int select)
{
	switch (select) {
		case SDAP_PDU_Type_DL: return "DL";
		case SDAP_PDU_Type_UL: return "UL";
		default: return "unknown";
	}
}

const char* adbgUtilsNrDrbNR_L2DataList_TypeToStr(int select)
{
	switch (select) {
		case NR_L2DataList_Type_MacPdu: return "MacPdu";
		case NR_L2DataList_Type_RlcPdu: return "RlcPdu";
		case NR_L2DataList_Type_RlcSdu: return "RlcSdu";
		case NR_L2DataList_Type_PdcpPdu: return "PdcpPdu";
		case NR_L2DataList_Type_PdcpSdu: return "PdcpSdu";
		case NR_L2DataList_Type_SdapPdu: return "SdapPdu";
		case NR_L2DataList_Type_SdapSdu: return "SdapSdu";
		default: return "unknown";
	}
}
