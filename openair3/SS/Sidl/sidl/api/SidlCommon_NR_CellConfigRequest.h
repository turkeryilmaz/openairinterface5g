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
#include "SidlCommon_NR_BcchConfig_Type.h"
#include "SidlCommon_NR_CellConfigCommon_Type.h"
#include "SidlCommon_NR_CellConfigPhysicalLayer_Type.h"
#include "SidlCommon_NR_DcchDtchConfig_Type.h"
#include "SidlCommon_NR_PcchConfig_Type.h"
#include "SidlCommon_NR_RachProcedureConfig_Type.h"
#include "SidlCommon_NR_SS_StaticCellResourceConfig_Type.h"
#include "SidlCommon_NR_ServingCellConfig_Type.h"

SIDL_BEGIN_C_INTERFACE

struct NR_SS_StaticCellResourceConfig_Type_StaticResourceConfig_Optional {
	bool d;
	struct NR_SS_StaticCellResourceConfig_Type v;
};

struct NR_CellConfigCommon_Type_CellConfigCommon_Optional {
	bool d;
	struct NR_CellConfigCommon_Type v;
};

struct NR_CellConfigPhysicalLayer_Type_PhysicalLayer_Optional {
	bool d;
	struct NR_CellConfigPhysicalLayer_Type v;
};

struct NR_BcchConfig_Type_BcchConfig_Optional {
	bool d;
	struct NR_BcchConfig_Type v;
};

struct NR_PcchConfig_Type_PcchConfig_Optional {
	bool d;
	struct NR_PcchConfig_Type v;
};

struct NR_RachProcedureConfig_Type_RachProcedureConfig_Optional {
	bool d;
	struct NR_RachProcedureConfig_Type v;
};

struct NR_DcchDtchConfig_Type_DcchDtchConfig_Optional {
	bool d;
	struct NR_DcchDtchConfig_Type v;
};

struct NR_ServingCellConfig_Type_ServingCellConfig_Optional {
	bool d;
	struct NR_ServingCellConfig_Type v;
};

struct NR_CellConfigInfo_Type {
	struct NR_SS_StaticCellResourceConfig_Type_StaticResourceConfig_Optional StaticResourceConfig;
	struct NR_CellConfigCommon_Type_CellConfigCommon_Optional CellConfigCommon;
	struct NR_CellConfigPhysicalLayer_Type_PhysicalLayer_Optional PhysicalLayer;
	struct NR_BcchConfig_Type_BcchConfig_Optional BcchConfig;
	struct NR_PcchConfig_Type_PcchConfig_Optional PcchConfig;
	struct NR_RachProcedureConfig_Type_RachProcedureConfig_Optional RachProcedureConfig;
	struct NR_DcchDtchConfig_Type_DcchDtchConfig_Optional DcchDtchConfig;
	struct NR_ServingCellConfig_Type_ServingCellConfig_Optional ServingCellConfig;
};

enum NR_CellConfigRequest_Type_Sel {
	NR_CellConfigRequest_Type_UNBOUND_VALUE = 0,
	NR_CellConfigRequest_Type_AddOrReconfigure = 1,
	NR_CellConfigRequest_Type_Release = 2,
};

union NR_CellConfigRequest_Type_Value {
	struct NR_CellConfigInfo_Type AddOrReconfigure;
	Null_Type Release;
};

struct NR_CellConfigRequest_Type {
	enum NR_CellConfigRequest_Type_Sel d;
	union NR_CellConfigRequest_Type_Value v;
};

SIDL_END_C_INTERFACE
