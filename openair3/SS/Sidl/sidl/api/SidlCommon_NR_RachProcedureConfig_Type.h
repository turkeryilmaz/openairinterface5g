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

typedef uint8_t NR_RAR_BackoffIndicator_Type;

struct NR_RAR_RapIdOnly_Type {
	struct RAR_RapIdCtrl_Type RapId;
};

struct NR_RAR_UplinkGrant_Type {
	B1_Type HoppingFlag;
	B14_Type Msg3FrequencyResourceAllocation;
	B4_Type Msg3TimeResourceAllocation;
	B4_Type MCS;
	B3_Type TPC_Command;
	B1_Type CQI_Req;
};

enum NR_TempC_RNTI_Type_Sel {
	NR_TempC_RNTI_Type_UNBOUND_VALUE = 0,
	NR_TempC_RNTI_Type_SameAsC_RNTI = 1,
	NR_TempC_RNTI_Type_Explicit = 2,
};

union NR_TempC_RNTI_Type_Value {
	Null_Type SameAsC_RNTI;
	RNTI_Value_Type Explicit;
};

struct NR_TempC_RNTI_Type {
	enum NR_TempC_RNTI_Type_Sel d;
	union NR_TempC_RNTI_Type_Value v;
};

struct NR_RAR_Payload_Type {
	NR_RACH_TimingAdvance_Type TimingAdvance;
	struct NR_RAR_UplinkGrant_Type UplinkGrant;
	struct NR_TempC_RNTI_Type TempC_RNTI;
};

struct NR_RAR_RapIdAndPayload_Type {
	struct RAR_RapIdCtrl_Type RapId;
	struct NR_RAR_Payload_Type Payload;
};

enum NR_RAR_SubPdu_Type_Sel {
	NR_RAR_SubPdu_Type_UNBOUND_VALUE = 0,
	NR_RAR_SubPdu_Type_BackoffIndicator = 1,
	NR_RAR_SubPdu_Type_RapIdOnly = 2,
	NR_RAR_SubPdu_Type_RapIdAndPayload = 3,
};

union NR_RAR_SubPdu_Type_Value {
	NR_RAR_BackoffIndicator_Type BackoffIndicator;
	struct NR_RAR_RapIdOnly_Type RapIdOnly;
	struct NR_RAR_RapIdAndPayload_Type RapIdAndPayload;
};

struct NR_RAR_SubPdu_Type {
	enum NR_RAR_SubPdu_Type_Sel d;
	union NR_RAR_SubPdu_Type_Value v;
};

struct NR_RAR_SubPdu_Type_NR_RAR_SubPduList_Type_Dynamic {
	size_t d;
	struct NR_RAR_SubPdu_Type* v;
};

typedef struct NR_RAR_SubPdu_Type_NR_RAR_SubPduList_Type_Dynamic NR_RAR_SubPduList_Type;

struct NR_RAR_SubPduList_Type_SubPduList_Optional {
	bool d;
	NR_RAR_SubPduList_Type v;
};

struct bool_NR_RAR_MacPdu_Type_CrcError_Optional {
	bool d;
	bool v;
};

struct NR_RAR_MacPdu_Type {
	struct NR_RAR_SubPduList_Type_SubPduList_Optional SubPduList;
	struct bool_NR_RAR_MacPdu_Type_CrcError_Optional CrcError;
};

struct NR_SearchSpaceDlDciAssignment_Type_NR_RandomAccessResponseConfig_Type_SearchSpaceAndDci_Optional {
	bool d;
	struct NR_SearchSpaceDlDciAssignment_Type v;
};

struct NR_RAR_MacPdu_Type_MacPdu_Optional {
	bool d;
	struct NR_RAR_MacPdu_Type v;
};

struct NR_RandomAccessResponseConfig_Type {
	struct NR_SearchSpaceDlDciAssignment_Type_NR_RandomAccessResponseConfig_Type_SearchSpaceAndDci_Optional SearchSpaceAndDci;
	struct NR_RAR_MacPdu_Type_MacPdu_Optional MacPdu;
};

enum NR_ContentionResolutionId_Type_Sel {
	NR_ContentionResolutionId_Type_UNBOUND_VALUE = 0,
	NR_ContentionResolutionId_Type_XorMask = 1,
	NR_ContentionResolutionId_Type_None = 2,
};

union NR_ContentionResolutionId_Type_Value {
	B48_Type XorMask;
	Null_Type None;
};

struct NR_ContentionResolutionId_Type {
	enum NR_ContentionResolutionId_Type_Sel d;
	union NR_ContentionResolutionId_Type_Value v;
};

enum NR_RachProcedureMsg4RrcMsg_Type_Sel {
	NR_RachProcedureMsg4RrcMsg_Type_UNBOUND_VALUE = 0,
	NR_RachProcedureMsg4RrcMsg_Type_RrcCcchMsg = 1,
	NR_RachProcedureMsg4RrcMsg_Type_RrcDcchMsg = 2,
	NR_RachProcedureMsg4RrcMsg_Type_None = 3,
};

union NR_RachProcedureMsg4RrcMsg_Type_Value {
	OCTET_STRING RrcCcchMsg;
	OCTET_STRING RrcDcchMsg;
	Null_Type None;
};

struct NR_RachProcedureMsg4RrcMsg_Type {
	enum NR_RachProcedureMsg4RrcMsg_Type_Sel d;
	union NR_RachProcedureMsg4RrcMsg_Type_Value v;
};

struct NR_SearchSpaceDlDciAssignment_Type_NR_RachProcedureMsg4_Type_SearchSpaceAndDci_Optional {
	bool d;
	struct NR_SearchSpaceDlDciAssignment_Type v;
};

struct NR_ContentionResolutionId_Type_ContentionResolutionId_Optional {
	bool d;
	struct NR_ContentionResolutionId_Type v;
};

struct NR_RachProcedureMsg4RrcMsg_Type_RrcPdu_Optional {
	bool d;
	struct NR_RachProcedureMsg4RrcMsg_Type v;
};

struct bool_NR_RachProcedureMsg4_Type_CrcError_Optional {
	bool d;
	bool v;
};

struct NR_RachProcedureMsg4_Type {
	struct NR_SearchSpaceDlDciAssignment_Type_NR_RachProcedureMsg4_Type_SearchSpaceAndDci_Optional SearchSpaceAndDci;
	struct NR_ContentionResolutionId_Type_ContentionResolutionId_Optional ContentionResolutionId;
	struct NR_RachProcedureMsg4RrcMsg_Type_RrcPdu_Optional RrcPdu;
	struct bool_NR_RachProcedureMsg4_Type_CrcError_Optional CrcError;
};

enum NR_ContentionResolutionCtrl_Type_Sel {
	NR_ContentionResolutionCtrl_Type_UNBOUND_VALUE = 0,
	NR_ContentionResolutionCtrl_Type_None = 1,
	NR_ContentionResolutionCtrl_Type_CRNTI_Based = 2,
	NR_ContentionResolutionCtrl_Type_Msg4_Based = 3,
};

union NR_ContentionResolutionCtrl_Type_Value {
	Null_Type None;
	struct NR_SearchSpaceUlDciAssignment_Type CRNTI_Based;
	struct NR_RachProcedureMsg4_Type Msg4_Based;
};

struct NR_ContentionResolutionCtrl_Type {
	enum NR_ContentionResolutionCtrl_Type_Sel d;
	union NR_ContentionResolutionCtrl_Type_Value v;
};

struct NR_RandomAccessResponseConfig_Type_RandomAccessResponse_Optional {
	bool d;
	struct NR_RandomAccessResponseConfig_Type v;
};

struct NR_ContentionResolutionCtrl_Type_ContentionResolution_Optional {
	bool d;
	struct NR_ContentionResolutionCtrl_Type v;
};

struct NR_RachProcedure_Type {
	struct NR_RandomAccessResponseConfig_Type_RandomAccessResponse_Optional RandomAccessResponse;
	struct NR_ContentionResolutionCtrl_Type_ContentionResolution_Optional ContentionResolution;
};

struct NR_RachProcedure_Type_NR_RachProcedureList_Type_Dynamic {
	size_t d;
	struct NR_RachProcedure_Type* v;
};

typedef struct NR_RachProcedure_Type_NR_RachProcedureList_Type_Dynamic NR_RachProcedureList_Type;

struct NR_RachProcedureList_Type_RachProcedureList_Optional {
	bool d;
	NR_RachProcedureList_Type v;
};

struct NR_RachProcedureConfig_Type {
	struct NR_RachProcedureList_Type_RachProcedureList_Optional RachProcedureList;
};

SIDL_END_C_INTERFACE
