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

typedef int32_t NR_AbsoluteCellPower_Type;

typedef struct NR_Attenuation_Type NR_InitialAttenuation_Type;

struct NR_InitialCellPower_Type {
	NR_AbsoluteCellPower_Type MaxReferencePower;
	NR_InitialAttenuation_Type Attenuation;
};

struct RNTI_Value_Type_C_RNTI_Optional {
	bool d;
	RNTI_Value_Type v;
};

struct CellTimingInfo_Type_CellTimingInfo_Optional {
	bool d;
	struct CellTimingInfo_Type v;
};

struct NR_InitialCellPower_Type_InitialCellPower_Optional {
	bool d;
	struct NR_InitialCellPower_Type v;
};

struct NR_CellConfigCommon_Type {
	struct RNTI_Value_Type_C_RNTI_Optional C_RNTI;
	struct CellTimingInfo_Type_CellTimingInfo_Optional CellTimingInfo;
	struct NR_InitialCellPower_Type_InitialCellPower_Optional InitialCellPower;
};

SIDL_END_C_INTERFACE
