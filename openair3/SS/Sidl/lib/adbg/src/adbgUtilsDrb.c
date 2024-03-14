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

#include "adbgUtilsDrb.h"

const char* adbgUtilsDrbHarqProcessAssignment_TypeToStr(int select)
{
	switch (select) {
		case HarqProcessAssignment_Type_Id: return "Id";
		case HarqProcessAssignment_Type_Automatic: return "Automatic";
		default: return "unknown";
	}
}

const char* adbgUtilsDrbRLC_LI_List_TypeToStr(int select)
{
	switch (select) {
		case RLC_LI_List_Type_LI11: return "LI11";
		case RLC_LI_List_Type_LI15: return "LI15";
		default: return "unknown";
	}
}

const char* adbgUtilsDrbRLC_UMD_PDU_TypeToStr(int select)
{
	switch (select) {
		case RLC_UMD_PDU_Type_ShortSN: return "ShortSN";
		case RLC_UMD_PDU_Type_LongSN: return "LongSN";
		default: return "unknown";
	}
}

const char* adbgUtilsDrbRLC_PDU_TypeToStr(int select)
{
	switch (select) {
		case RLC_PDU_Type_TMD: return "TMD";
		case RLC_PDU_Type_UMD: return "UMD";
		case RLC_PDU_Type_AMD: return "AMD";
		case RLC_PDU_Type_AMD_Ext: return "AMD_Ext";
		case RLC_PDU_Type_AMD_SegExt: return "AMD_SegExt";
		case RLC_PDU_Type_Status: return "Status";
		case RLC_PDU_Type_Status_Ext: return "Status_Ext";
		default: return "unknown";
	}
}

const char* adbgUtilsDrbPDCP_PDU_TypeToStr(int select)
{
	switch (select) {
		case PDCP_PDU_Type_DataLongSN: return "DataLongSN";
		case PDCP_PDU_Type_DataShortSN: return "DataShortSN";
		case PDCP_PDU_Type_DataExtSN: return "DataExtSN";
		case PDCP_PDU_Type_Data_18bitSN: return "Data_18bitSN";
		case PDCP_PDU_Type_RohcFeedback: return "RohcFeedback";
		case PDCP_PDU_Type_StatusReport: return "StatusReport";
		case PDCP_PDU_Type_StatusReportExt: return "StatusReportExt";
		case PDCP_PDU_Type_StatusReport_18bitSN: return "StatusReport_18bitSN";
		case PDCP_PDU_Type_LWA_StatusReport: return "LWA_StatusReport";
		case PDCP_PDU_Type_LWA_StatusReportExt: return "LWA_StatusReportExt";
		case PDCP_PDU_Type_LWA_StatusReport_18bitSN: return "LWA_StatusReport_18bitSN";
		case PDCP_PDU_Type_DataSLRB: return "DataSLRB";
		case PDCP_PDU_Type_DataSLRB_1to1: return "DataSLRB_1to1";
		case PDCP_PDU_Type_LWA_EndMarker: return "LWA_EndMarker";
		case PDCP_PDU_Type_LWA_EndMarkerExt: return "LWA_EndMarkerExt";
		case PDCP_PDU_Type_LWA_EndMarker_18bitSN: return "LWA_EndMarker_18bitSN";
		case PDCP_PDU_Type_DataLongSN_UDC: return "DataLongSN_UDC";
		case PDCP_PDU_Type_DataExtSN_UDC: return "DataExtSN_UDC";
		case PDCP_PDU_Type_Data18bitSN_UDC: return "Data18bitSN_UDC";
		case PDCP_PDU_Type_UdcFeedback: return "UdcFeedback";
		case PDCP_PDU_Type_EhcFeedback_ShortCID: return "EhcFeedback_ShortCID";
		case PDCP_PDU_Type_EhcFeedback_LongCID: return "EhcFeedback_LongCID";
		default: return "unknown";
	}
}

const char* adbgUtilsDrbL2DataList_TypeToStr(int select)
{
	switch (select) {
		case L2DataList_Type_MacPdu: return "MacPdu";
		case L2DataList_Type_RlcPdu: return "RlcPdu";
		case L2DataList_Type_PdcpPdu: return "PdcpPdu";
		case L2DataList_Type_PdcpSdu: return "PdcpSdu";
		case L2DataList_Type_NrPdcpSdu: return "NrPdcpSdu";
		case L2DataList_Type_RlcSdu: return "RlcSdu";
		default: return "unknown";
	}
}
