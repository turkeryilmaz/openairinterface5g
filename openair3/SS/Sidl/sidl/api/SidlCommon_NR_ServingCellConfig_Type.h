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

struct NR_CellId_Type_NR_CellIdList_Type_Dynamic {
	size_t d;
	NR_CellId_Type* v;
};

typedef struct NR_CellId_Type_NR_CellIdList_Type_Dynamic NR_CellIdList_Type;

struct SQN_NR_CellGroupId_CellGroupId_Optional {
	bool d;
	SQN_NR_CellGroupId v;
};

struct NR_CellIdList_Type_SCellList_Optional {
	bool d;
	NR_CellIdList_Type v;
};

struct NR_ASN1_MAC_CellGroupConfig_Type_MAC_CellGroupConfig_Optional {
	bool d;
	struct NR_ASN1_MAC_CellGroupConfig_Type v;
};

struct NR_ASN1_PhysicalCellGroupConfig_Type_PhysicalCellGroupConfig_Optional {
	bool d;
	struct NR_ASN1_PhysicalCellGroupConfig_Type v;
};

struct NR_SpCell_CellGroupConfig_Type {
	struct SQN_NR_CellGroupId_CellGroupId_Optional CellGroupId;
	struct NR_CellIdList_Type_SCellList_Optional SCellList;
	struct NR_ASN1_MAC_CellGroupConfig_Type_MAC_CellGroupConfig_Optional MAC_CellGroupConfig;
	struct NR_ASN1_PhysicalCellGroupConfig_Type_PhysicalCellGroupConfig_Optional PhysicalCellGroupConfig;
};

struct NR_ServingCellIndex_Type_NR_SpCellConfig_Type_ServingCellIndex_Optional {
	bool d;
	NR_ServingCellIndex_Type v;
};

struct NR_SpCell_CellGroupConfig_Type_CellGroupConfig_Optional {
	bool d;
	struct NR_SpCell_CellGroupConfig_Type v;
};

struct NR_SpCellConfig_Type {
	struct NR_ServingCellIndex_Type_NR_SpCellConfig_Type_ServingCellIndex_Optional ServingCellIndex;
	struct NR_SpCell_CellGroupConfig_Type_CellGroupConfig_Optional CellGroupConfig;
};

struct NR_ServingCellIndex_Type_NR_SCellConfig_Type_ServingCellIndex_Optional {
	bool d;
	NR_ServingCellIndex_Type v;
};

struct SQN_NR_TAG_Id_TAG_Id_Optional {
	bool d;
	SQN_NR_TAG_Id v;
};

struct NR_SCellConfig_Type {
	struct NR_ServingCellIndex_Type_NR_SCellConfig_Type_ServingCellIndex_Optional ServingCellIndex;
	struct SQN_NR_TAG_Id_TAG_Id_Optional TAG_Id;
};

enum NR_ServingCellConfig_Type_Sel {
	NR_ServingCellConfig_Type_UNBOUND_VALUE = 0,
	NR_ServingCellConfig_Type_SpCell = 1,
	NR_ServingCellConfig_Type_SCell = 2,
	NR_ServingCellConfig_Type_None = 3,
};

union NR_ServingCellConfig_Type_Value {
	struct NR_SpCellConfig_Type SpCell;
	struct NR_SCellConfig_Type SCell;
	Null_Type None;
};

struct NR_ServingCellConfig_Type {
	enum NR_ServingCellConfig_Type_Sel d;
	union NR_ServingCellConfig_Type_Value v;
};

SIDL_END_C_INTERFACE
