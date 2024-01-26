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

struct NR_SearchSpaceDlDciAssignment_Type_NR_DcchDtchConfigDL_Type_SearchSpaceAndDci_Optional {
	bool d;
	struct NR_SearchSpaceDlDciAssignment_Type v;
};

struct NR_DcchDtchConfigDL_Type {
	struct NR_SearchSpaceDlDciAssignment_Type_NR_DcchDtchConfigDL_Type_SearchSpaceAndDci_Optional SearchSpaceAndDci;
};

struct NR_UplinkTimeAlignment_AutoSynch_Type {
	NR_TimingAdvanceIndex_Type TimingAdvance;
	NR_TimingAdvance_Period_Type TA_Period;
};

enum NR_UplinkTimeAlignment_Synch_Type_Sel {
	NR_UplinkTimeAlignment_Synch_Type_UNBOUND_VALUE = 0,
	NR_UplinkTimeAlignment_Synch_Type_None = 1,
	NR_UplinkTimeAlignment_Synch_Type_Auto = 2,
};

union NR_UplinkTimeAlignment_Synch_Type_Value {
	Null_Type None;
	struct NR_UplinkTimeAlignment_AutoSynch_Type Auto;
};

struct NR_UplinkTimeAlignment_Synch_Type {
	enum NR_UplinkTimeAlignment_Synch_Type_Sel d;
	union NR_UplinkTimeAlignment_Synch_Type_Value v;
};

struct NR_SearchSpaceUlDciAssignment_Type_SearchSpaceAndDci_Optional {
	bool d;
	struct NR_SearchSpaceUlDciAssignment_Type v;
};

struct NR_UplinkTimeAlignment_Synch_Type_PUCCH_Synch_Optional {
	bool d;
	struct NR_UplinkTimeAlignment_Synch_Type v;
};

struct UL_GrantConfig_Type_GrantConfig_Optional {
	bool d;
	struct UL_GrantConfig_Type v;
};

struct NR_DcchDtchConfigUL_Type {
	struct NR_SearchSpaceUlDciAssignment_Type_SearchSpaceAndDci_Optional SearchSpaceAndDci;
	struct NR_UplinkTimeAlignment_Synch_Type_PUCCH_Synch_Optional PUCCH_Synch;
	struct UL_GrantConfig_Type_GrantConfig_Optional GrantConfig;
};

enum NR_DrxCtrl_Type_Sel {
	NR_DrxCtrl_Type_UNBOUND_VALUE = 0,
	NR_DrxCtrl_Type_None = 1,
	NR_DrxCtrl_Type_Config = 2,
};

union NR_DrxCtrl_Type_Value {
	Null_Type None;
	struct NR_ASN1_DRX_Config_Type Config;
};

struct NR_DrxCtrl_Type {
	enum NR_DrxCtrl_Type_Sel d;
	union NR_DrxCtrl_Type_Value v;
};

enum NR_MeasGapCtrl_Type_Sel {
	NR_MeasGapCtrl_Type_UNBOUND_VALUE = 0,
	NR_MeasGapCtrl_Type_None = 1,
	NR_MeasGapCtrl_Type_Config = 2,
};

union NR_MeasGapCtrl_Type_Value {
	Null_Type None;
	struct NR_ASN1_MeasGapConfig_Type Config;
};

struct NR_MeasGapCtrl_Type {
	enum NR_MeasGapCtrl_Type_Sel d;
	union NR_MeasGapCtrl_Type_Value v;
};

struct NR_DcchDtchConfigDL_Type_DL_Optional {
	bool d;
	struct NR_DcchDtchConfigDL_Type v;
};

struct NR_DcchDtchConfigUL_Type_UL_Optional {
	bool d;
	struct NR_DcchDtchConfigUL_Type v;
};

struct NR_DrxCtrl_Type_DrxCtrl_Optional {
	bool d;
	struct NR_DrxCtrl_Type v;
};

struct NR_MeasGapCtrl_Type_MeasGapCtrl_Optional {
	bool d;
	struct NR_MeasGapCtrl_Type v;
};

struct NR_DcchDtchConfig_Type {
	struct NR_DcchDtchConfigDL_Type_DL_Optional DL;
	struct NR_DcchDtchConfigUL_Type_UL_Optional UL;
	struct NR_DrxCtrl_Type_DrxCtrl_Optional DrxCtrl;
	struct NR_MeasGapCtrl_Type_MeasGapCtrl_Optional MeasGapCtrl;
};

SIDL_END_C_INTERFACE
