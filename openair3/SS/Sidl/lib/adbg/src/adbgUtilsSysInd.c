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

#include "adbgUtilsSysInd.h"

const char* adbgUtilsSysIndBSR_TypeToStr(int select)
{
	switch (select) {
		case BSR_Type_Short: return "Short";
		case BSR_Type_Truncated: return "Truncated";
		case BSR_Type_Long: return "Long";
		case BSR_Type_Sidelink: return "Sidelink";
		default: return "unknown";
	}
}

const char* adbgUtilsSysIndHarqError_TypeToStr(int select)
{
	switch (select) {
		case HarqError_Type_UL: return "UL";
		case HarqError_Type_DL: return "DL";
		default: return "unknown";
	}
}

const char* adbgUtilsSysIndSystemIndication_TypeToStr(int select)
{
	switch (select) {
		case SystemIndication_Type_Error: return "Error";
		case SystemIndication_Type_RachPreamble: return "RachPreamble";
		case SystemIndication_Type_SchedReq: return "SchedReq";
		case SystemIndication_Type_BSR: return "BSR";
		case SystemIndication_Type_UL_HARQ: return "UL_HARQ";
		case SystemIndication_Type_C_RNTI: return "C_RNTI";
		case SystemIndication_Type_PHR: return "PHR";
		case SystemIndication_Type_HarqError: return "HarqError";
		case SystemIndication_Type_RlcDiscardInd: return "RlcDiscardInd";
		case SystemIndication_Type_PeriodicRI: return "PeriodicRI";
		case SystemIndication_Type_EPHR: return "EPHR";
		case SystemIndication_Type_CqiInd: return "CqiInd";
		case SystemIndication_Type_SrsInd: return "SrsInd";
		case SystemIndication_Type_DC_PHR: return "DC_PHR";
		default: return "unknown";
	}
}
