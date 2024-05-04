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

typedef Null_Type NR_BcchToPbchConfig_Type;

struct int32_t_IntegerList_Type_Dynamic {
	size_t d;
	int32_t* v;
};

typedef struct int32_t_IntegerList_Type_Dynamic IntegerList_Type;

struct NR_SearchSpaceDlDciAssignment_Type_NR_Sib1Schedul_Type_SearchSpaceAndDci_Optional {
	bool d;
	struct NR_SearchSpaceDlDciAssignment_Type v;
};

struct int32_t_NR_Sib1Schedul_Type_Periodicity_Optional {
	bool d;
	int32_t v;
};

struct IntegerList_Type_SlotOffsetList_Optional {
	bool d;
	IntegerList_Type v;
};

struct NR_Sib1Schedul_Type {
	struct NR_SearchSpaceDlDciAssignment_Type_NR_Sib1Schedul_Type_SearchSpaceAndDci_Optional SearchSpaceAndDci;
	struct int32_t_NR_Sib1Schedul_Type_Periodicity_Optional Periodicity;
	struct IntegerList_Type_SlotOffsetList_Optional SlotOffsetList;
};

typedef SQN_NR_SI_SchedulingInfo_si_WindowLength_e NR_SiWindowLength_Type;

typedef SQN_NR_SchedulingInfo_si_Periodicity_e NR_SiPeriodicity_Type;

struct NR_SearchSpaceDlDciAssignment_Type_NR_SingleSiSchedul_Type_SearchSpaceAndDci_Optional {
	bool d;
	struct NR_SearchSpaceDlDciAssignment_Type v;
};

struct int32_t_NR_SingleSiSchedul_Type_SlotOffset_Optional {
	bool d;
	int32_t v;
};

struct NR_SingleSiSchedul_Type {
	struct NR_SearchSpaceDlDciAssignment_Type_NR_SingleSiSchedul_Type_SearchSpaceAndDci_Optional SearchSpaceAndDci;
	struct int32_t_NR_SingleSiSchedul_Type_SlotOffset_Optional SlotOffset;
};

struct NR_SiPeriodicity_Type_Periodicity_Optional {
	bool d;
	NR_SiPeriodicity_Type v;
};

struct NR_SingleSiSchedul_Type_Window_Dynamic {
	size_t d;
	struct NR_SingleSiSchedul_Type* v;
};

struct NR_SingleSiSchedul_Type_Window_DynamicOptional {
	bool d;
	struct NR_SingleSiSchedul_Type_Window_Dynamic v;
};

struct NR_OtherSiSchedul_Type {
	struct NR_SiPeriodicity_Type_Periodicity_Optional Periodicity;
	struct NR_SingleSiSchedul_Type_Window_DynamicOptional Window;
};

struct NR_OtherSiSchedul_Type_NR_OtherSiSchedulList_Type_Dynamic {
	size_t d;
	struct NR_OtherSiSchedul_Type* v;
};

typedef struct NR_OtherSiSchedul_Type_NR_OtherSiSchedulList_Type_Dynamic NR_OtherSiSchedulList_Type;

struct NR_SiWindowLength_Type_WindowLength_Optional {
	bool d;
	NR_SiWindowLength_Type v;
};

struct NR_OtherSiSchedulList_Type_SiList_Optional {
	bool d;
	NR_OtherSiSchedulList_Type v;
};

struct NR_OtherSiSchedulList_Type_SegmentedSiList_Optional {
	bool d;
	NR_OtherSiSchedulList_Type v;
};

struct NR_AllOtherSiSchedul_Type {
	struct NR_SiWindowLength_Type_WindowLength_Optional WindowLength;
	struct NR_OtherSiSchedulList_Type_SiList_Optional SiList;
	struct NR_OtherSiSchedulList_Type_SegmentedSiList_Optional SegmentedSiList;
};

struct NR_Sib1Schedul_Type_Sib1Schedul_Optional {
	bool d;
	struct NR_Sib1Schedul_Type v;
};

struct NR_AllOtherSiSchedul_Type_SiSchedul_Optional {
	bool d;
	struct NR_AllOtherSiSchedul_Type v;
};

struct NR_BcchToPdschConfig_Type {
	struct NR_Sib1Schedul_Type_Sib1Schedul_Optional Sib1Schedul;
	struct NR_AllOtherSiSchedul_Type_SiSchedul_Optional SiSchedul;
};

struct SQN_NR_BCCH_DL_SCH_Message_NR_SI_List_Type_Dynamic {
	size_t d;
	struct SQN_NR_BCCH_DL_SCH_Message* v;
};

typedef struct SQN_NR_BCCH_DL_SCH_Message_NR_SI_List_Type_Dynamic NR_SI_List_Type;

struct NR_SI_List_Type_NR_SegmentedSI_List_Type_Dynamic {
	size_t d;
	NR_SI_List_Type* v;
};

typedef struct NR_SI_List_Type_NR_SegmentedSI_List_Type_Dynamic NR_SegmentedSI_List_Type;

enum NR_PosSI_Type_Sel {
	NR_PosSI_Type_UNBOUND_VALUE = 0,
	NR_PosSI_Type_PosSI = 1,
	NR_PosSI_Type_SegmentedPosSIs = 2,
};

union NR_PosSI_Type_Value {
	struct SQN_NR_BCCH_DL_SCH_Message PosSI;
	NR_SI_List_Type SegmentedPosSIs;
};

struct NR_PosSI_Type {
	enum NR_PosSI_Type_Sel d;
	union NR_PosSI_Type_Value v;
};

struct NR_PosSI_Type_NR_PosSI_List_Type_Dynamic {
	size_t d;
	struct NR_PosSI_Type* v;
};

typedef struct NR_PosSI_Type_NR_PosSI_List_Type_Dynamic NR_PosSI_List_Type;

struct SQN_NR_BCCH_BCH_Message_MIB_Optional {
	bool d;
	struct SQN_NR_BCCH_BCH_Message v;
};

struct SQN_NR_BCCH_DL_SCH_Message_SIB1_Optional {
	bool d;
	struct SQN_NR_BCCH_DL_SCH_Message v;
};

struct NR_SI_List_Type_SIs_Optional {
	bool d;
	NR_SI_List_Type v;
};

struct NR_SegmentedSI_List_Type_SegmentedSIs_Optional {
	bool d;
	NR_SegmentedSI_List_Type v;
};

struct NR_PosSI_List_Type_PosSIs_Optional {
	bool d;
	NR_PosSI_List_Type v;
};

struct NR_BcchInfo_Type {
	struct SQN_NR_BCCH_BCH_Message_MIB_Optional MIB;
	struct SQN_NR_BCCH_DL_SCH_Message_SIB1_Optional SIB1;
	struct NR_SI_List_Type_SIs_Optional SIs;
	struct NR_SegmentedSI_List_Type_SegmentedSIs_Optional SegmentedSIs;
	struct NR_PosSI_List_Type_PosSIs_Optional PosSIs;
};

struct NR_BcchToPbchConfig_Type_Pbch_Optional {
	bool d;
	NR_BcchToPbchConfig_Type v;
};

struct NR_BcchToPdschConfig_Type_Pdsch_Optional {
	bool d;
	struct NR_BcchToPdschConfig_Type v;
};

struct NR_BcchInfo_Type_BcchInfo_Optional {
	bool d;
	struct NR_BcchInfo_Type v;
};

struct NR_BcchConfig_Type {
	struct NR_BcchToPbchConfig_Type_Pbch_Optional Pbch;
	struct NR_BcchToPdschConfig_Type_Pdsch_Optional Pdsch;
	struct NR_BcchInfo_Type_BcchInfo_Optional BcchInfo;
};

SIDL_END_C_INTERFACE
