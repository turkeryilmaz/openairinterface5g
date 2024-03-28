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

SIDL_BEGIN_C_INTERFACE

typedef SQN_MAC_MainConfig_mac_MainConfig_v1020_sCellDeactivationTimer_r10_e MAC_MainConfig_ScellDeactivationTimer_Type;

typedef struct SQN_CrossCarrierSchedulingConfig_r10_schedulingCellInfo_r10_other_r10 CrossSchedulingCarrierInfo_Type;

typedef SQN_TDD_Config_subframeAssignment_e TDD_SubframeAssignment_Type;

enum Scell_Capability_Type {
	Scell_Cap_DlOnly = 0,
	Scell_Cap_UL_DL = 1,
};

typedef enum Scell_Capability_Type Scell_Capability_Type;

struct SQN_CrossCarrierSchedulingConfig_r10_crossCarrierSchedulingConfig_r10_Optional {
	bool d;
	struct SQN_CrossCarrierSchedulingConfig_r10 v;
};

struct SQN_CrossCarrierSchedulingConfig_r13_crossCarrierSchedulingConfig_r13_Optional {
	bool d;
	struct SQN_CrossCarrierSchedulingConfig_r13 v;
};

struct CrossCarrierScheduledCellsList_Type {
	struct SQN_CrossCarrierSchedulingConfig_r10_crossCarrierSchedulingConfig_r10_Optional crossCarrierSchedulingConfig_r10;
	struct SQN_CrossCarrierSchedulingConfig_r13_crossCarrierSchedulingConfig_r13_Optional crossCarrierSchedulingConfig_r13;
};

struct SQN_MeasSubframePatternPCell_r10_MeasSubframePatternPCell_Optional {
	bool d;
	struct SQN_MeasSubframePatternPCell_r10 v;
};

struct CrossCarrierScheduledCellsList_Type_CrossCarrierScheduledCellsList_Optional {
	bool d;
	struct CrossCarrierScheduledCellsList_Type v;
};

struct PrimaryCellInfo_Type {
	EUTRA_CellIdList_Type AssociatedScellList;
	struct SQN_MeasSubframePatternPCell_r10_MeasSubframePatternPCell_Optional MeasSubframePatternPCell;
	struct CrossCarrierScheduledCellsList_Type_CrossCarrierScheduledCellsList_Optional CrossCarrierScheduledCellsList;
};

enum ScellDeactivationTimer_Type_Sel {
	ScellDeactivationTimer_Type_UNBOUND_VALUE = 0,
	ScellDeactivationTimer_Type_NumberOfRadioFrames = 1,
	ScellDeactivationTimer_Type_Infinity = 2,
};

union ScellDeactivationTimer_Type_Value {
	MAC_MainConfig_ScellDeactivationTimer_Type NumberOfRadioFrames;
	Null_Type Infinity;
};

struct ScellDeactivationTimer_Type {
	enum ScellDeactivationTimer_Type_Sel d;
	union ScellDeactivationTimer_Type_Value v;
};

enum SchedulingCarrierConfig_Type_Sel {
	SchedulingCarrierConfig_Type_UNBOUND_VALUE = 0,
	SchedulingCarrierConfig_Type_Own = 1,
	SchedulingCarrierConfig_Type_CrossScheduled = 2,
};

union SchedulingCarrierConfig_Type_Value {
	struct CrossCarrierScheduledCellsList_Type Own;
	CrossSchedulingCarrierInfo_Type CrossScheduled;
};

struct SchedulingCarrierConfig_Type {
	enum SchedulingCarrierConfig_Type_Sel d;
	union SchedulingCarrierConfig_Type_Value v;
};

enum CrossCarrierSchedulingConfig_Type_Sel {
	CrossCarrierSchedulingConfig_Type_UNBOUND_VALUE = 0,
	CrossCarrierSchedulingConfig_Type_Config = 1,
	CrossCarrierSchedulingConfig_Type_None = 2,
};

union CrossCarrierSchedulingConfig_Type_Value {
	struct SchedulingCarrierConfig_Type Config;
	Null_Type None;
};

struct CrossCarrierSchedulingConfig_Type {
	enum CrossCarrierSchedulingConfig_Type_Sel d;
	union CrossCarrierSchedulingConfig_Type_Value v;
};

enum Pcell_Mode_Type_Sel {
	Pcell_Mode_Type_UNBOUND_VALUE = 0,
	Pcell_Mode_Type_FDD = 1,
	Pcell_Mode_Type_TDD = 2,
};

union Pcell_Mode_Type_Value {
	Null_Type FDD;
	TDD_SubframeAssignment_Type TDD;
};

struct Pcell_Mode_Type {
	enum Pcell_Mode_Type_Sel d;
	union Pcell_Mode_Type_Value v;
};

struct Scell_Capability_Type_Scell_Capability_Optional {
	bool d;
	Scell_Capability_Type v;
};

struct ScellDeactivationTimer_Type_ScellDeactivationTimer_Optional {
	bool d;
	struct ScellDeactivationTimer_Type v;
};

struct CrossCarrierSchedulingConfig_Type_SecondaryCellInfo_Type_CrossCarrierSchedulingConfig_Optional {
	bool d;
	struct CrossCarrierSchedulingConfig_Type v;
};

struct SQN_STAG_Id_r11_STAG_Id_Optional {
	bool d;
	SQN_STAG_Id_r11 v;
};

struct Pcell_Mode_Type_SecondaryCellInfo_Type_Pcell_Mode_Optional {
	bool d;
	struct Pcell_Mode_Type v;
};

struct SecondaryCellInfo_Type {
	EUTRA_CellId_Type AssociatedPcellId;
	SQN_SCellIndex_r10 SCellIndex;
	struct Scell_Capability_Type_Scell_Capability_Optional Scell_Capability;
	struct ScellDeactivationTimer_Type_ScellDeactivationTimer_Optional ScellDeactivationTimer;
	struct CrossCarrierSchedulingConfig_Type_SecondaryCellInfo_Type_CrossCarrierSchedulingConfig_Optional CrossCarrierSchedulingConfig;
	struct SQN_STAG_Id_r11_STAG_Id_Optional STAG_Id;
	struct Pcell_Mode_Type_SecondaryCellInfo_Type_Pcell_Mode_Optional Pcell_Mode;
};

struct CrossCarrierSchedulingConfig_Type_PSCellInfo_Type_CrossCarrierSchedulingConfig_Optional {
	bool d;
	struct CrossCarrierSchedulingConfig_Type v;
};

struct Pcell_Mode_Type_PSCellInfo_Type_Pcell_Mode_Optional {
	bool d;
	struct Pcell_Mode_Type v;
};

struct PSCellInfo_Type {
	EUTRA_CellId_Type AssociatedPcellId;
	SQN_SCellIndex_r10 SCellIndex;
	EUTRA_CellIdList_Type AssociatedScellList;
	struct CrossCarrierSchedulingConfig_Type_PSCellInfo_Type_CrossCarrierSchedulingConfig_Optional CrossCarrierSchedulingConfig;
	struct Pcell_Mode_Type_PSCellInfo_Type_Pcell_Mode_Optional Pcell_Mode;
};

enum ServingCellConfig_Type_Sel {
	ServingCellConfig_Type_UNBOUND_VALUE = 0,
	ServingCellConfig_Type_PCell = 1,
	ServingCellConfig_Type_SCell = 2,
	ServingCellConfig_Type_PSCell = 3,
	ServingCellConfig_Type_Release = 4,
};

union ServingCellConfig_Type_Value {
	struct PrimaryCellInfo_Type PCell;
	struct SecondaryCellInfo_Type SCell;
	struct PSCellInfo_Type PSCell;
	Null_Type Release;
};

struct ServingCellConfig_Type {
	enum ServingCellConfig_Type_Sel d;
	union ServingCellConfig_Type_Value v;
};

SIDL_END_C_INTERFACE
