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

#include "adbgNrDrb.h"

static const char* adbgNrDrb__NR_CellId_Type__ToString(NR_CellId_Type v)
{
	switch(v) {
		case nr_Cell_NonSpecific: return "nr_Cell_NonSpecific";
		case nr_Cell1: return "nr_Cell1";
		case nr_Cell2: return "nr_Cell2";
		case nr_Cell3: return "nr_Cell3";
		case nr_Cell4: return "nr_Cell4";
		case nr_Cell6: return "nr_Cell6";
		case nr_Cell10: return "nr_Cell10";
		case nr_Cell11: return "nr_Cell11";
		case nr_Cell12: return "nr_Cell12";
		case nr_Cell13: return "nr_Cell13";
		case nr_Cell14: return "nr_Cell14";
		case nr_Cell23: return "nr_Cell23";
		case nr_Cell28: return "nr_Cell28";
		case nr_Cell29: return "nr_Cell29";
		case nr_Cell30: return "nr_Cell30";
		case nr_Cell31: return "nr_Cell31";
		default: return "Unknown";
	}
}

static void _adbgNrDrb__NR_RadioBearerId_Type_Value(acpCtx_t _ctx, const union NR_RadioBearerId_Type_Value* p, enum NR_RadioBearerId_Type_Sel d)
{
	if (d == NR_RadioBearerId_Type_Srb) {
		adbgPrintLog(_ctx, "Srb := %u", (unsigned int)p->Srb);
		return;
	}
	if (d == NR_RadioBearerId_Type_Drb) {
		adbgPrintLog(_ctx, "Drb := %u", (unsigned int)p->Drb);
		return;
	}
	adbgPrintLog(_ctx, "INVALID");
}

static void _adbgNrDrb__NR_RadioBearerId_Type(acpCtx_t _ctx, const struct NR_RadioBearerId_Type* p)
{
	_adbgNrDrb__NR_RadioBearerId_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrDrb__QosFlow_Identification_Type(acpCtx_t _ctx, const struct QosFlow_Identification_Type* p)
{
	adbgPrintLog(_ctx, "PDU_SessionId := %d", (int)p->PDU_SessionId);
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "QFI := %d", (int)p->QFI);
}

static void _adbgNrDrb__NR_RoutingInfo_Type_Value(acpCtx_t _ctx, const union NR_RoutingInfo_Type_Value* p, enum NR_RoutingInfo_Type_Sel d)
{
	if (d == NR_RoutingInfo_Type_None) {
		adbgPrintLog(_ctx, "None := %s", (p->None ? "true" : "false"));
		return;
	}
	if (d == NR_RoutingInfo_Type_RadioBearerId) {
		adbgPrintLog(_ctx, "RadioBearerId := { ");
		_adbgNrDrb__NR_RadioBearerId_Type(_ctx, &p->RadioBearerId);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_RoutingInfo_Type_QosFlow) {
		adbgPrintLog(_ctx, "QosFlow := { ");
		_adbgNrDrb__QosFlow_Identification_Type(_ctx, &p->QosFlow);
		adbgPrintLog(_ctx, " }");
		return;
	}
	adbgPrintLog(_ctx, "INVALID");
}

static void _adbgNrDrb__NR_RoutingInfo_Type(acpCtx_t _ctx, const struct NR_RoutingInfo_Type* p)
{
	_adbgNrDrb__NR_RoutingInfo_Type_Value(_ctx, &p->v, p->d);
}

static const char* adbgNrDrb__EUTRA_CellId_Type__ToString(EUTRA_CellId_Type v)
{
	switch(v) {
		case eutra_Cell_NonSpecific: return "eutra_Cell_NonSpecific";
		case eutra_Cell1: return "eutra_Cell1";
		case eutra_Cell2: return "eutra_Cell2";
		case eutra_Cell3: return "eutra_Cell3";
		case eutra_Cell4: return "eutra_Cell4";
		case eutra_Cell6: return "eutra_Cell6";
		case eutra_Cell10: return "eutra_Cell10";
		case eutra_Cell11: return "eutra_Cell11";
		case eutra_Cell12: return "eutra_Cell12";
		case eutra_Cell13: return "eutra_Cell13";
		case eutra_Cell14: return "eutra_Cell14";
		case eutra_Cell23: return "eutra_Cell23";
		case eutra_Cell28: return "eutra_Cell28";
		case eutra_Cell29: return "eutra_Cell29";
		case eutra_Cell30: return "eutra_Cell30";
		case eutra_Cell31: return "eutra_Cell31";
		case eutra_CellA: return "eutra_CellA";
		case eutra_CellB: return "eutra_CellB";
		case eutra_CellC: return "eutra_CellC";
		case eutra_CellD: return "eutra_CellD";
		case eutra_CellE: return "eutra_CellE";
		case eutra_CellG: return "eutra_CellG";
		case eutra_CellH: return "eutra_CellH";
		case eutra_CellI: return "eutra_CellI";
		case eutra_CellJ: return "eutra_CellJ";
		case eutra_CellK: return "eutra_CellK";
		case eutra_CellL: return "eutra_CellL";
		case eutra_CellM: return "eutra_CellM";
		default: return "Unknown";
	}
}

static void _adbgNrDrb__RlcBearerRouting_Type_Value(acpCtx_t _ctx, const union RlcBearerRouting_Type_Value* p, enum RlcBearerRouting_Type_Sel d)
{
	if (d == RlcBearerRouting_Type_EUTRA) {
		adbgPrintLog(_ctx, "EUTRA := %s (%d)", adbgNrDrb__EUTRA_CellId_Type__ToString(p->EUTRA), (int)p->EUTRA);
		return;
	}
	if (d == RlcBearerRouting_Type_NR) {
		adbgPrintLog(_ctx, "NR := %s (%d)", adbgNrDrb__NR_CellId_Type__ToString(p->NR), (int)p->NR);
		return;
	}
	if (d == RlcBearerRouting_Type_None) {
		adbgPrintLog(_ctx, "None := %s", (p->None ? "true" : "false"));
		return;
	}
	adbgPrintLog(_ctx, "INVALID");
}

static void _adbgNrDrb__RlcBearerRouting_Type(acpCtx_t _ctx, const struct RlcBearerRouting_Type* p)
{
	_adbgNrDrb__RlcBearerRouting_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrDrb__MacBearerRouting_Type(acpCtx_t _ctx, const struct MacBearerRouting_Type* p)
{
	adbgPrintLog(_ctx, "NR := %s (%d)", adbgNrDrb__NR_CellId_Type__ToString(p->NR), (int)p->NR);
}

static void _adbgNrDrb__MacBearerRouting_Type_NR_ReqAspCommonPart_Type_MacBearerRouting_Optional(acpCtx_t _ctx, const struct MacBearerRouting_Type_NR_ReqAspCommonPart_Type_MacBearerRouting_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	_adbgNrDrb__MacBearerRouting_Type(_ctx, &p->v);
}

static void _adbgNrDrb__SystemFrameNumberInfo_Type_Value(acpCtx_t _ctx, const union SystemFrameNumberInfo_Type_Value* p, enum SystemFrameNumberInfo_Type_Sel d)
{
	if (d == SystemFrameNumberInfo_Type_Number) {
		adbgPrintLog(_ctx, "Number := %u", (unsigned int)p->Number);
		return;
	}
	if (d == SystemFrameNumberInfo_Type_Any) {
		adbgPrintLog(_ctx, "Any := %s", (p->Any ? "true" : "false"));
		return;
	}
	adbgPrintLog(_ctx, "INVALID");
}

static void _adbgNrDrb__SystemFrameNumberInfo_Type(acpCtx_t _ctx, const struct SystemFrameNumberInfo_Type* p)
{
	_adbgNrDrb__SystemFrameNumberInfo_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrDrb__SubFrameInfo_Type_Value(acpCtx_t _ctx, const union SubFrameInfo_Type_Value* p, enum SubFrameInfo_Type_Sel d)
{
	if (d == SubFrameInfo_Type_Number) {
		adbgPrintLog(_ctx, "Number := %u", (unsigned int)p->Number);
		return;
	}
	if (d == SubFrameInfo_Type_Any) {
		adbgPrintLog(_ctx, "Any := %s", (p->Any ? "true" : "false"));
		return;
	}
	adbgPrintLog(_ctx, "INVALID");
}

static void _adbgNrDrb__SubFrameInfo_Type(acpCtx_t _ctx, const struct SubFrameInfo_Type* p)
{
	_adbgNrDrb__SubFrameInfo_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrDrb__HyperSystemFrameNumberInfo_Type(acpCtx_t _ctx, const HyperSystemFrameNumberInfo_Type* p)
{
	_adbgNrDrb__SystemFrameNumberInfo_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrDrb__SlotOffset_Type_Value(acpCtx_t _ctx, const union SlotOffset_Type_Value* p, enum SlotOffset_Type_Sel d)
{
	if (d == SlotOffset_Type_Numerology0) {
		adbgPrintLog(_ctx, "Numerology0 := %s", (p->Numerology0 ? "true" : "false"));
		return;
	}
	if (d == SlotOffset_Type_Numerology1) {
		adbgPrintLog(_ctx, "Numerology1 := %u", (unsigned int)p->Numerology1);
		return;
	}
	if (d == SlotOffset_Type_Numerology2) {
		adbgPrintLog(_ctx, "Numerology2 := %u", (unsigned int)p->Numerology2);
		return;
	}
	if (d == SlotOffset_Type_Numerology3) {
		adbgPrintLog(_ctx, "Numerology3 := %u", (unsigned int)p->Numerology3);
		return;
	}
	if (d == SlotOffset_Type_Numerology4) {
		adbgPrintLog(_ctx, "Numerology4 := %u", (unsigned int)p->Numerology4);
		return;
	}
	adbgPrintLog(_ctx, "INVALID");
}

static void _adbgNrDrb__SlotOffset_Type(acpCtx_t _ctx, const struct SlotOffset_Type* p)
{
	_adbgNrDrb__SlotOffset_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrDrb__SlotTimingInfo_Type_Value(acpCtx_t _ctx, const union SlotTimingInfo_Type_Value* p, enum SlotTimingInfo_Type_Sel d)
{
	if (d == SlotTimingInfo_Type_SlotOffset) {
		adbgPrintLog(_ctx, "SlotOffset := { ");
		_adbgNrDrb__SlotOffset_Type(_ctx, &p->SlotOffset);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == SlotTimingInfo_Type_FirstSlot) {
		adbgPrintLog(_ctx, "FirstSlot := %s", (p->FirstSlot ? "true" : "false"));
		return;
	}
	if (d == SlotTimingInfo_Type_Any) {
		adbgPrintLog(_ctx, "Any := %s", (p->Any ? "true" : "false"));
		return;
	}
	adbgPrintLog(_ctx, "INVALID");
}

static void _adbgNrDrb__SlotTimingInfo_Type(acpCtx_t _ctx, const struct SlotTimingInfo_Type* p)
{
	_adbgNrDrb__SlotTimingInfo_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrDrb__SymbolTimingInfo_Type_Value(acpCtx_t _ctx, const union SymbolTimingInfo_Type_Value* p, enum SymbolTimingInfo_Type_Sel d)
{
	if (d == SymbolTimingInfo_Type_SymbolOffset) {
		adbgPrintLog(_ctx, "SymbolOffset := %u", (unsigned int)p->SymbolOffset);
		return;
	}
	if (d == SymbolTimingInfo_Type_FirstSymbol) {
		adbgPrintLog(_ctx, "FirstSymbol := %s", (p->FirstSymbol ? "true" : "false"));
		return;
	}
	if (d == SymbolTimingInfo_Type_Any) {
		adbgPrintLog(_ctx, "Any := %s", (p->Any ? "true" : "false"));
		return;
	}
	adbgPrintLog(_ctx, "INVALID");
}

static void _adbgNrDrb__SymbolTimingInfo_Type(acpCtx_t _ctx, const struct SymbolTimingInfo_Type* p)
{
	_adbgNrDrb__SymbolTimingInfo_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrDrb__SubFrameTiming_Type(acpCtx_t _ctx, const struct SubFrameTiming_Type* p)
{
	adbgPrintLog(_ctx, "SFN := { ");
	_adbgNrDrb__SystemFrameNumberInfo_Type(_ctx, &p->SFN);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Subframe := { ");
	_adbgNrDrb__SubFrameInfo_Type(_ctx, &p->Subframe);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "HSFN := { ");
	_adbgNrDrb__HyperSystemFrameNumberInfo_Type(_ctx, &p->HSFN);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Slot := { ");
	_adbgNrDrb__SlotTimingInfo_Type(_ctx, &p->Slot);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Symbol := { ");
	_adbgNrDrb__SymbolTimingInfo_Type(_ctx, &p->Symbol);
	adbgPrintLog(_ctx, " }");
}

static void _adbgNrDrb__TimingInfo_Type_Value(acpCtx_t _ctx, const union TimingInfo_Type_Value* p, enum TimingInfo_Type_Sel d)
{
	if (d == TimingInfo_Type_SubFrame) {
		adbgPrintLog(_ctx, "SubFrame := { ");
		_adbgNrDrb__SubFrameTiming_Type(_ctx, &p->SubFrame);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == TimingInfo_Type_Now) {
		adbgPrintLog(_ctx, "Now := %s", (p->Now ? "true" : "false"));
		return;
	}
	if (d == TimingInfo_Type_None) {
		adbgPrintLog(_ctx, "None := %s", (p->None ? "true" : "false"));
		return;
	}
	adbgPrintLog(_ctx, "INVALID");
}

static void _adbgNrDrb__TimingInfo_Type(acpCtx_t _ctx, const struct TimingInfo_Type* p)
{
	_adbgNrDrb__TimingInfo_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrDrb__ReqAspControlInfo_Type(acpCtx_t _ctx, const struct ReqAspControlInfo_Type* p)
{
	adbgPrintLog(_ctx, "CnfFlag := %s", (p->CnfFlag ? "true" : "false"));
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "FollowOnFlag := %s", (p->FollowOnFlag ? "true" : "false"));
}

static void _adbgNrDrb__NR_ReqAspCommonPart_Type(acpCtx_t _ctx, const struct NR_ReqAspCommonPart_Type* p)
{
	adbgPrintLog(_ctx, "CellId := %s (%d)", adbgNrDrb__NR_CellId_Type__ToString(p->CellId), (int)p->CellId);
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "RoutingInfo := { ");
	_adbgNrDrb__NR_RoutingInfo_Type(_ctx, &p->RoutingInfo);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "RlcBearerRouting := { ");
	_adbgNrDrb__RlcBearerRouting_Type(_ctx, &p->RlcBearerRouting);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "MacBearerRouting := ");
	if (p->MacBearerRouting.d) { adbgPrintLog(_ctx, "{ "); };
	_adbgNrDrb__MacBearerRouting_Type_NR_ReqAspCommonPart_Type_MacBearerRouting_Optional(_ctx, &p->MacBearerRouting);
	if (p->MacBearerRouting.d) { adbgPrintLog(_ctx, " }"); };
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "TimingInfo := { ");
	_adbgNrDrb__TimingInfo_Type(_ctx, &p->TimingInfo);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "ControlInfo := { ");
	_adbgNrDrb__ReqAspControlInfo_Type(_ctx, &p->ControlInfo);
	adbgPrintLog(_ctx, " }");
}

static void _adbgNrDrb__NR_HarqProcessAssignment_Type_Value(acpCtx_t _ctx, const union NR_HarqProcessAssignment_Type_Value* p, enum NR_HarqProcessAssignment_Type_Sel d)
{
	if (d == NR_HarqProcessAssignment_Type_Id) {
		adbgPrintLog(_ctx, "Id := %d", (int)p->Id);
		return;
	}
	if (d == NR_HarqProcessAssignment_Type_Automatic) {
		adbgPrintLog(_ctx, "Automatic := %s", (p->Automatic ? "true" : "false"));
		return;
	}
	adbgPrintLog(_ctx, "INVALID");
}

static void _adbgNrDrb__NR_HarqProcessAssignment_Type(acpCtx_t _ctx, const struct NR_HarqProcessAssignment_Type* p)
{
	_adbgNrDrb__NR_HarqProcessAssignment_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrDrb__NR_HarqProcessAssignment_Type_HarqProcess_Optional(acpCtx_t _ctx, const struct NR_HarqProcessAssignment_Type_HarqProcess_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	_adbgNrDrb__NR_HarqProcessAssignment_Type(_ctx, &p->v);
}

static void _adbgNrDrb__BIT_STRING_NR_MAC_PDU_SubHeader_Type_ELCID_Optional(acpCtx_t _ctx, const struct BIT_STRING_NR_MAC_PDU_SubHeader_Type_ELCID_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	adbgPrintLog(_ctx, "'");
	for (size_t i4 = 0; i4 < p->v.d; i4++) {
		adbgPrintLog(_ctx, "%02X", p->v.v[i4]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__BIT_STRING_NR_MAC_PDU_SubHeader_Type_Length_Optional(acpCtx_t _ctx, const struct BIT_STRING_NR_MAC_PDU_SubHeader_Type_Length_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	adbgPrintLog(_ctx, "'");
	for (size_t i4 = 0; i4 < p->v.d; i4++) {
		adbgPrintLog(_ctx, "%02X", p->v.v[i4]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_MAC_PDU_SubHeader_Type(acpCtx_t _ctx, const struct NR_MAC_PDU_SubHeader_Type* p)
{
	adbgPrintLog(_ctx, "Reserved := '");
	for (size_t i4 = 0; i4 < 1; i4++) {
		adbgPrintLog(_ctx, "%02X", p->Reserved[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Format := '");
	for (size_t i4 = 0; i4 < 1; i4++) {
		adbgPrintLog(_ctx, "%02X", p->Format[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "LCID := '");
	for (size_t i4 = 0; i4 < 6; i4++) {
		adbgPrintLog(_ctx, "%02X", p->LCID[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "ELCID := ");
	if (p->ELCID.d) { adbgPrintLog(_ctx, "{ "); };
	_adbgNrDrb__BIT_STRING_NR_MAC_PDU_SubHeader_Type_ELCID_Optional(_ctx, &p->ELCID);
	if (p->ELCID.d) { adbgPrintLog(_ctx, " }"); };
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Length := ");
	if (p->Length.d) { adbgPrintLog(_ctx, "{ "); };
	_adbgNrDrb__BIT_STRING_NR_MAC_PDU_SubHeader_Type_Length_Optional(_ctx, &p->Length);
	if (p->Length.d) { adbgPrintLog(_ctx, " }"); };
}

static void _adbgNrDrb__NR_MAC_CE_TimingAdvance_Type(acpCtx_t _ctx, const struct NR_MAC_CE_TimingAdvance_Type* p)
{
	adbgPrintLog(_ctx, "TAG_ID := '");
	for (size_t i4 = 0; i4 < 2; i4++) {
		adbgPrintLog(_ctx, "%02X", p->TAG_ID[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "TimingAdvanceCommand := '");
	for (size_t i4 = 0; i4 < 6; i4++) {
		adbgPrintLog(_ctx, "%02X", p->TimingAdvanceCommand[i4]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__B8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex15_8_Optional(acpCtx_t _ctx, const struct B8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex15_8_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	adbgPrintLog(_ctx, "'");
	for (size_t i4 = 0; i4 < 8; i4++) {
		adbgPrintLog(_ctx, "%02X", p->v[i4]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__B8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex23_16_Optional(acpCtx_t _ctx, const struct B8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex23_16_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	adbgPrintLog(_ctx, "'");
	for (size_t i4 = 0; i4 < 8; i4++) {
		adbgPrintLog(_ctx, "%02X", p->v[i4]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__B8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex31_24_Optional(acpCtx_t _ctx, const struct B8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex31_24_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	adbgPrintLog(_ctx, "'");
	for (size_t i4 = 0; i4 < 8; i4++) {
		adbgPrintLog(_ctx, "%02X", p->v[i4]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_MAC_CE_SCellActDeact_Type(acpCtx_t _ctx, const NR_MAC_CE_SCellActDeact_Type* p)
{
	adbgPrintLog(_ctx, "SCellIndex7_1 := '");
	for (size_t i4 = 0; i4 < 8; i4++) {
		adbgPrintLog(_ctx, "%02X", p->SCellIndex7_1[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SCellIndex15_8 := ");
	_adbgNrDrb__B8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex15_8_Optional(_ctx, &p->SCellIndex15_8);
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SCellIndex23_16 := ");
	_adbgNrDrb__B8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex23_16_Optional(_ctx, &p->SCellIndex23_16);
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SCellIndex31_24 := ");
	_adbgNrDrb__B8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex31_24_Optional(_ctx, &p->SCellIndex31_24);
}

static void _adbgNrDrb__NR_MAC_CE_ServCellId_BwpId_Type(acpCtx_t _ctx, const struct NR_MAC_CE_ServCellId_BwpId_Type* p)
{
	adbgPrintLog(_ctx, "Field1 := '");
	for (size_t i4 = 0; i4 < 1; i4++) {
		adbgPrintLog(_ctx, "%02X", p->Field1[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "ServCellId := '");
	for (size_t i4 = 0; i4 < 5; i4++) {
		adbgPrintLog(_ctx, "%02X", p->ServCellId[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "BwpId := '");
	for (size_t i4 = 0; i4 < 2; i4++) {
		adbgPrintLog(_ctx, "%02X", p->BwpId[i4]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_MAC_CE_SP_ResourceSetActDeact_Octet2_Type(acpCtx_t _ctx, const struct NR_MAC_CE_SP_ResourceSetActDeact_Octet2_Type* p)
{
	adbgPrintLog(_ctx, "Reserved := '");
	for (size_t i4 = 0; i4 < 1; i4++) {
		adbgPrintLog(_ctx, "%02X", p->Reserved[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "IM := '");
	for (size_t i4 = 0; i4 < 1; i4++) {
		adbgPrintLog(_ctx, "%02X", p->IM[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "CSI_RS_ResourcesetId := '");
	for (size_t i4 = 0; i4 < 6; i4++) {
		adbgPrintLog(_ctx, "%02X", p->CSI_RS_ResourcesetId[i4]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_MAC_CE_SP_ResourceSetActDeact_Octet3_Type(acpCtx_t _ctx, const struct NR_MAC_CE_SP_ResourceSetActDeact_Octet3_Type* p)
{
	adbgPrintLog(_ctx, "Reserved := '");
	for (size_t i4 = 0; i4 < 2; i4++) {
		adbgPrintLog(_ctx, "%02X", p->Reserved[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "CSI_IM_ResourcesetId := '");
	for (size_t i4 = 0; i4 < 6; i4++) {
		adbgPrintLog(_ctx, "%02X", p->CSI_IM_ResourcesetId[i4]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_MAC_CE_SP_ResourceSetActDeact_Octet3_Type_NR_MAC_CE_SP_ResourceSetActDeact_Type_Octet3_Optional(acpCtx_t _ctx, const struct NR_MAC_CE_SP_ResourceSetActDeact_Octet3_Type_NR_MAC_CE_SP_ResourceSetActDeact_Type_Octet3_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	_adbgNrDrb__NR_MAC_CE_SP_ResourceSetActDeact_Octet3_Type(_ctx, &p->v);
}

static void _adbgNrDrb__NR_MAC_CE_SP_ResourceSetActDeact_TciStateId_Type(acpCtx_t _ctx, const struct NR_MAC_CE_SP_ResourceSetActDeact_TciStateId_Type* p)
{
	adbgPrintLog(_ctx, "Reserved := '");
	for (size_t i5 = 0; i5 < 1; i5++) {
		adbgPrintLog(_ctx, "%02X", p->Reserved[i5]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Id := '");
	for (size_t i5 = 0; i5 < 7; i5++) {
		adbgPrintLog(_ctx, "%02X", p->Id[i5]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_MAC_CE_SP_ResourceSetActDeact_Type(acpCtx_t _ctx, const struct NR_MAC_CE_SP_ResourceSetActDeact_Type* p)
{
	adbgPrintLog(_ctx, "Octet1 := { ");
	_adbgNrDrb__NR_MAC_CE_ServCellId_BwpId_Type(_ctx, &p->Octet1);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Octet2 := { ");
	_adbgNrDrb__NR_MAC_CE_SP_ResourceSetActDeact_Octet2_Type(_ctx, &p->Octet2);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Octet3 := ");
	if (p->Octet3.d) { adbgPrintLog(_ctx, "{ "); };
	_adbgNrDrb__NR_MAC_CE_SP_ResourceSetActDeact_Octet3_Type_NR_MAC_CE_SP_ResourceSetActDeact_Type_Octet3_Optional(_ctx, &p->Octet3);
	if (p->Octet3.d) { adbgPrintLog(_ctx, " }"); };
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "IdList := { ");
	for (size_t i4 = 0; i4 < p->IdList.d; i4++) {
		adbgPrintLog(_ctx, "{ ");
		_adbgNrDrb__NR_MAC_CE_SP_ResourceSetActDeact_TciStateId_Type(_ctx, &p->IdList.v[i4]);
		adbgPrintLog(_ctx, " }");
		if (i4 != p->IdList.d - 1) { adbgPrintLog(_ctx, ", "); }
	}
	adbgPrintLog(_ctx, " }");
}

static void _adbgNrDrb__NR_MAC_CE_CSI_TriggerStateSubselection_Type(acpCtx_t _ctx, const struct NR_MAC_CE_CSI_TriggerStateSubselection_Type* p)
{
	adbgPrintLog(_ctx, "Octet1 := { ");
	_adbgNrDrb__NR_MAC_CE_ServCellId_BwpId_Type(_ctx, &p->Octet1);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "{");
	for (size_t i4 = 0; i4 < p->Selection.d; i4++) {
		adbgPrintLog(_ctx, "'");
		for (size_t i5 = 0; i5 < 8; i5++) {
			adbgPrintLog(_ctx, "%02X", p->Selection.v[i4][i5]);
		}
		adbgPrintLog(_ctx, "'O");
		if (i4 != p->Selection.d - 1) { adbgPrintLog(_ctx, ", "); }
	}
	adbgPrintLog(_ctx, " }");
}

static void _adbgNrDrb__NR_MAC_CE_TCI_StatesActDeact_Type(acpCtx_t _ctx, const struct NR_MAC_CE_TCI_StatesActDeact_Type* p)
{
	adbgPrintLog(_ctx, "Octet1 := { ");
	_adbgNrDrb__NR_MAC_CE_ServCellId_BwpId_Type(_ctx, &p->Octet1);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "{");
	for (size_t i4 = 0; i4 < p->Status.d; i4++) {
		adbgPrintLog(_ctx, "'");
		for (size_t i5 = 0; i5 < 8; i5++) {
			adbgPrintLog(_ctx, "%02X", p->Status.v[i4][i5]);
		}
		adbgPrintLog(_ctx, "'O");
		if (i4 != p->Status.d - 1) { adbgPrintLog(_ctx, ", "); }
	}
	adbgPrintLog(_ctx, " }");
}

static void _adbgNrDrb__NR_MAC_CE_TCI_StateIndication_Type(acpCtx_t _ctx, const struct NR_MAC_CE_TCI_StateIndication_Type* p)
{
	adbgPrintLog(_ctx, "ServCellId := '");
	for (size_t i4 = 0; i4 < 5; i4++) {
		adbgPrintLog(_ctx, "%02X", p->ServCellId[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "CoresetId := '");
	for (size_t i4 = 0; i4 < 4; i4++) {
		adbgPrintLog(_ctx, "%02X", p->CoresetId[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "TciStateId := '");
	for (size_t i4 = 0; i4 < 7; i4++) {
		adbgPrintLog(_ctx, "%02X", p->TciStateId[i4]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_MAC_CE_SP_CSI_ReportingActDeact_Type(acpCtx_t _ctx, const struct NR_MAC_CE_SP_CSI_ReportingActDeact_Type* p)
{
	adbgPrintLog(_ctx, "Octet1 := { ");
	_adbgNrDrb__NR_MAC_CE_ServCellId_BwpId_Type(_ctx, &p->Octet1);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Reserved := '");
	for (size_t i4 = 0; i4 < 4; i4++) {
		adbgPrintLog(_ctx, "%02X", p->Reserved[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "ConfigState := '");
	for (size_t i4 = 0; i4 < 4; i4++) {
		adbgPrintLog(_ctx, "%02X", p->ConfigState[i4]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_MAC_CE_SP_SRS_ActDeact_Octet2_Type(acpCtx_t _ctx, const struct NR_MAC_CE_SP_SRS_ActDeact_Octet2_Type* p)
{
	adbgPrintLog(_ctx, "Reserved := '");
	for (size_t i4 = 0; i4 < 2; i4++) {
		adbgPrintLog(_ctx, "%02X", p->Reserved[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "C := '");
	for (size_t i4 = 0; i4 < 1; i4++) {
		adbgPrintLog(_ctx, "%02X", p->C[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SUL := '");
	for (size_t i4 = 0; i4 < 1; i4++) {
		adbgPrintLog(_ctx, "%02X", p->SUL[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SRS_ResourcesetId := '");
	for (size_t i4 = 0; i4 < 4; i4++) {
		adbgPrintLog(_ctx, "%02X", p->SRS_ResourcesetId[i4]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_MAC_CE_SP_SRS_ActDeact_ResourceId_Type(acpCtx_t _ctx, const struct NR_MAC_CE_SP_SRS_ActDeact_ResourceId_Type* p)
{
	adbgPrintLog(_ctx, "F := '");
	for (size_t i5 = 0; i5 < 1; i5++) {
		adbgPrintLog(_ctx, "%02X", p->F[i5]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Id := '");
	for (size_t i5 = 0; i5 < 7; i5++) {
		adbgPrintLog(_ctx, "%02X", p->Id[i5]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_MAC_CE_SP_SRS_ActDeact_ResourceInfo_Type(acpCtx_t _ctx, const struct NR_MAC_CE_SP_SRS_ActDeact_ResourceInfo_Type* p)
{
	adbgPrintLog(_ctx, "Reserved := '");
	for (size_t i5 = 0; i5 < 1; i5++) {
		adbgPrintLog(_ctx, "%02X", p->Reserved[i5]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "ServingCellId := '");
	for (size_t i5 = 0; i5 < 5; i5++) {
		adbgPrintLog(_ctx, "%02X", p->ServingCellId[i5]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "BwpId := '");
	for (size_t i5 = 0; i5 < 2; i5++) {
		adbgPrintLog(_ctx, "%02X", p->BwpId[i5]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_MAC_CE_SP_SRS_ActDeact_Type(acpCtx_t _ctx, const struct NR_MAC_CE_SP_SRS_ActDeact_Type* p)
{
	adbgPrintLog(_ctx, "Octet1 := { ");
	_adbgNrDrb__NR_MAC_CE_ServCellId_BwpId_Type(_ctx, &p->Octet1);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Octet2 := { ");
	_adbgNrDrb__NR_MAC_CE_SP_SRS_ActDeact_Octet2_Type(_ctx, &p->Octet2);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "ResourceIdList := { ");
	for (size_t i4 = 0; i4 < p->ResourceIdList.d; i4++) {
		adbgPrintLog(_ctx, "{ ");
		_adbgNrDrb__NR_MAC_CE_SP_SRS_ActDeact_ResourceId_Type(_ctx, &p->ResourceIdList.v[i4]);
		adbgPrintLog(_ctx, " }");
		if (i4 != p->ResourceIdList.d - 1) { adbgPrintLog(_ctx, ", "); }
	}
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "ResourceInfoList := { ");
	for (size_t i4 = 0; i4 < p->ResourceInfoList.d; i4++) {
		adbgPrintLog(_ctx, "{ ");
		_adbgNrDrb__NR_MAC_CE_SP_SRS_ActDeact_ResourceInfo_Type(_ctx, &p->ResourceInfoList.v[i4]);
		adbgPrintLog(_ctx, " }");
		if (i4 != p->ResourceInfoList.d - 1) { adbgPrintLog(_ctx, ", "); }
	}
	adbgPrintLog(_ctx, " }");
}

static void _adbgNrDrb__NR_MAC_CE_PUCCH_SpatialRelationActDeact_Octet2_Type(acpCtx_t _ctx, const struct NR_MAC_CE_PUCCH_SpatialRelationActDeact_Octet2_Type* p)
{
	adbgPrintLog(_ctx, "Reserved := '");
	for (size_t i4 = 0; i4 < 1; i4++) {
		adbgPrintLog(_ctx, "%02X", p->Reserved[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "ResourceId := '");
	for (size_t i4 = 0; i4 < 7; i4++) {
		adbgPrintLog(_ctx, "%02X", p->ResourceId[i4]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_MAC_CE_PUCCH_SpatialRelationActDeact_Type(acpCtx_t _ctx, const struct NR_MAC_CE_PUCCH_SpatialRelationActDeact_Type* p)
{
	adbgPrintLog(_ctx, "Octet1 := { ");
	_adbgNrDrb__NR_MAC_CE_ServCellId_BwpId_Type(_ctx, &p->Octet1);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Octet2 := { ");
	_adbgNrDrb__NR_MAC_CE_PUCCH_SpatialRelationActDeact_Octet2_Type(_ctx, &p->Octet2);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "ActivationStatus := '");
	for (size_t i4 = 0; i4 < 8; i4++) {
		adbgPrintLog(_ctx, "%02X", p->ActivationStatus[i4]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_MAC_CE_SP_ZP_ResourceSetActDeact_Octet2_Type(acpCtx_t _ctx, const struct NR_MAC_CE_SP_ZP_ResourceSetActDeact_Octet2_Type* p)
{
	adbgPrintLog(_ctx, "Reserved := '");
	for (size_t i4 = 0; i4 < 4; i4++) {
		adbgPrintLog(_ctx, "%02X", p->Reserved[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Id := '");
	for (size_t i4 = 0; i4 < 4; i4++) {
		adbgPrintLog(_ctx, "%02X", p->Id[i4]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_MAC_CE_SP_ZP_ResourceSetActDeact_Type(acpCtx_t _ctx, const struct NR_MAC_CE_SP_ZP_ResourceSetActDeact_Type* p)
{
	adbgPrintLog(_ctx, "Octet1 := { ");
	_adbgNrDrb__NR_MAC_CE_ServCellId_BwpId_Type(_ctx, &p->Octet1);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Octet2 := { ");
	_adbgNrDrb__NR_MAC_CE_SP_ZP_ResourceSetActDeact_Octet2_Type(_ctx, &p->Octet2);
	adbgPrintLog(_ctx, " }");
}

static void _adbgNrDrb__NR_MAC_CE_RecommendedBitrate_Type(acpCtx_t _ctx, const struct NR_MAC_CE_RecommendedBitrate_Type* p)
{
	adbgPrintLog(_ctx, "LCID := '");
	for (size_t i4 = 0; i4 < 6; i4++) {
		adbgPrintLog(_ctx, "%02X", p->LCID[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "UL_DL := '");
	for (size_t i4 = 0; i4 < 1; i4++) {
		adbgPrintLog(_ctx, "%02X", p->UL_DL[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Bitrate := '");
	for (size_t i4 = 0; i4 < 6; i4++) {
		adbgPrintLog(_ctx, "%02X", p->Bitrate[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "X := '");
	for (size_t i4 = 0; i4 < 1; i4++) {
		adbgPrintLog(_ctx, "%02X", p->X[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Reserved := '");
	for (size_t i4 = 0; i4 < 2; i4++) {
		adbgPrintLog(_ctx, "%02X", p->Reserved[i4]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_MAC_ControlElementDL_Type_Value(acpCtx_t _ctx, const union NR_MAC_ControlElementDL_Type_Value* p, enum NR_MAC_ControlElementDL_Type_Sel d)
{
	if (d == NR_MAC_ControlElementDL_Type_ContentionResolutionID) {
		adbgPrintLog(_ctx, "ContentionResolutionID := '");
		for (size_t i4 = 0; i4 < 48; i4++) {
			adbgPrintLog(_ctx, "%02X", p->ContentionResolutionID[i4]);
		}
		adbgPrintLog(_ctx, "'O");
		return;
	}
	if (d == NR_MAC_ControlElementDL_Type_TimingAdvance) {
		adbgPrintLog(_ctx, "TimingAdvance := { ");
		_adbgNrDrb__NR_MAC_CE_TimingAdvance_Type(_ctx, &p->TimingAdvance);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_MAC_ControlElementDL_Type_SCellActDeact) {
		adbgPrintLog(_ctx, "SCellActDeact := { ");
		_adbgNrDrb__NR_MAC_CE_SCellActDeact_Type(_ctx, &p->SCellActDeact);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_MAC_ControlElementDL_Type_DuplicationActDeact) {
		adbgPrintLog(_ctx, "DuplicationActDeact := '");
		for (size_t i4 = 0; i4 < 8; i4++) {
			adbgPrintLog(_ctx, "%02X", p->DuplicationActDeact[i4]);
		}
		adbgPrintLog(_ctx, "'O");
		return;
	}
	if (d == NR_MAC_ControlElementDL_Type_SP_ResourceSetActDeact) {
		adbgPrintLog(_ctx, "SP_ResourceSetActDeact := { ");
		_adbgNrDrb__NR_MAC_CE_SP_ResourceSetActDeact_Type(_ctx, &p->SP_ResourceSetActDeact);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_MAC_ControlElementDL_Type_CSI_TriggerStateSubselection) {
		adbgPrintLog(_ctx, "CSI_TriggerStateSubselection := { ");
		_adbgNrDrb__NR_MAC_CE_CSI_TriggerStateSubselection_Type(_ctx, &p->CSI_TriggerStateSubselection);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_MAC_ControlElementDL_Type_TCI_StatesActDeact) {
		adbgPrintLog(_ctx, "TCI_StatesActDeact := { ");
		_adbgNrDrb__NR_MAC_CE_TCI_StatesActDeact_Type(_ctx, &p->TCI_StatesActDeact);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_MAC_ControlElementDL_Type_TCI_StateIndication) {
		adbgPrintLog(_ctx, "TCI_StateIndication := { ");
		_adbgNrDrb__NR_MAC_CE_TCI_StateIndication_Type(_ctx, &p->TCI_StateIndication);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_MAC_ControlElementDL_Type_SP_CSI_ReportingActDeact) {
		adbgPrintLog(_ctx, "SP_CSI_ReportingActDeact := { ");
		_adbgNrDrb__NR_MAC_CE_SP_CSI_ReportingActDeact_Type(_ctx, &p->SP_CSI_ReportingActDeact);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_MAC_ControlElementDL_Type_SP_SRS_ActDeact) {
		adbgPrintLog(_ctx, "SP_SRS_ActDeact := { ");
		_adbgNrDrb__NR_MAC_CE_SP_SRS_ActDeact_Type(_ctx, &p->SP_SRS_ActDeact);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_MAC_ControlElementDL_Type_PUCCH_SpatialRelationActDeact) {
		adbgPrintLog(_ctx, "PUCCH_SpatialRelationActDeact := { ");
		_adbgNrDrb__NR_MAC_CE_PUCCH_SpatialRelationActDeact_Type(_ctx, &p->PUCCH_SpatialRelationActDeact);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_MAC_ControlElementDL_Type_SP_ZP_ResourceSetActDeact) {
		adbgPrintLog(_ctx, "SP_ZP_ResourceSetActDeact := { ");
		_adbgNrDrb__NR_MAC_CE_SP_ZP_ResourceSetActDeact_Type(_ctx, &p->SP_ZP_ResourceSetActDeact);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_MAC_ControlElementDL_Type_RecommendatdBitrate) {
		adbgPrintLog(_ctx, "RecommendatdBitrate := { ");
		_adbgNrDrb__NR_MAC_CE_RecommendedBitrate_Type(_ctx, &p->RecommendatdBitrate);
		adbgPrintLog(_ctx, " }");
		return;
	}
	adbgPrintLog(_ctx, "INVALID");
}

static void _adbgNrDrb__NR_MAC_ControlElementDL_Type(acpCtx_t _ctx, const struct NR_MAC_ControlElementDL_Type* p)
{
	_adbgNrDrb__NR_MAC_ControlElementDL_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrDrb__NR_MAC_ControlElementDL_Type_NR_MAC_CE_SubPDU_DL_Type_ControlElement_Optional(acpCtx_t _ctx, const struct NR_MAC_ControlElementDL_Type_NR_MAC_CE_SubPDU_DL_Type_ControlElement_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	_adbgNrDrb__NR_MAC_ControlElementDL_Type(_ctx, &p->v);
}

static void _adbgNrDrb__NR_MAC_CE_SubPDU_DL_Type(acpCtx_t _ctx, const struct NR_MAC_CE_SubPDU_DL_Type* p)
{
	adbgPrintLog(_ctx, "SubHeader := { ");
	_adbgNrDrb__NR_MAC_PDU_SubHeader_Type(_ctx, &p->SubHeader);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "ControlElement := ");
	if (p->ControlElement.d) { adbgPrintLog(_ctx, "{ "); };
	_adbgNrDrb__NR_MAC_ControlElementDL_Type_NR_MAC_CE_SubPDU_DL_Type_ControlElement_Optional(_ctx, &p->ControlElement);
	if (p->ControlElement.d) { adbgPrintLog(_ctx, " }"); };
}

static void _adbgNrDrb__NR_MAC_CE_SubPDU_DL_List_Type_NR_MAC_PDU_DL_Type_CE_SubPDUList_Optional(acpCtx_t _ctx, const struct NR_MAC_CE_SubPDU_DL_List_Type_NR_MAC_PDU_DL_Type_CE_SubPDUList_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	for (size_t i3 = 0; i3 < p->v.d; i3++) {
		_adbgNrDrb__NR_MAC_CE_SubPDU_DL_Type(_ctx, &p->v.v[i3]);
		if (i3 != p->v.d - 1) { adbgPrintLog(_ctx, ", "); }
	}
}

static void _adbgNrDrb__NR_MAC_SDU_SubPDU_Type(acpCtx_t _ctx, const struct NR_MAC_SDU_SubPDU_Type* p)
{
	adbgPrintLog(_ctx, "SubHeader := { ");
	_adbgNrDrb__NR_MAC_PDU_SubHeader_Type(_ctx, &p->SubHeader);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SDU := '");
	for (size_t i4 = 0; i4 < p->SDU.d; i4++) {
		adbgPrintLog(_ctx, "%02X", p->SDU.v[i4]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_DL_Type_SDU_SubPDUList_Optional(acpCtx_t _ctx, const struct NR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_DL_Type_SDU_SubPDUList_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	for (size_t i3 = 0; i3 < p->v.d; i3++) {
		_adbgNrDrb__NR_MAC_SDU_SubPDU_Type(_ctx, &p->v.v[i3]);
		if (i3 != p->v.d - 1) { adbgPrintLog(_ctx, ", "); }
	}
}

static void _adbgNrDrb__NR_MAC_Padding_SubPDU_Type(acpCtx_t _ctx, const struct NR_MAC_Padding_SubPDU_Type* p)
{
	adbgPrintLog(_ctx, "SubHeader := { ");
	_adbgNrDrb__NR_MAC_PDU_SubHeader_Type(_ctx, &p->SubHeader);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Padding := '");
	for (size_t i3 = 0; i3 < p->Padding.d; i3++) {
		adbgPrintLog(_ctx, "%02X", p->Padding.v[i3]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_DL_Type_Padding_SubPDU_Optional(acpCtx_t _ctx, const struct NR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_DL_Type_Padding_SubPDU_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	_adbgNrDrb__NR_MAC_Padding_SubPDU_Type(_ctx, &p->v);
}

static void _adbgNrDrb__NR_MAC_PDU_DL_Type(acpCtx_t _ctx, const struct NR_MAC_PDU_DL_Type* p)
{
	adbgPrintLog(_ctx, "CE_SubPDUList := ");
	if (p->CE_SubPDUList.d) { adbgPrintLog(_ctx, "{ "); };
	_adbgNrDrb__NR_MAC_CE_SubPDU_DL_List_Type_NR_MAC_PDU_DL_Type_CE_SubPDUList_Optional(_ctx, &p->CE_SubPDUList);
	if (p->CE_SubPDUList.d) { adbgPrintLog(_ctx, " }"); };
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SDU_SubPDUList := ");
	if (p->SDU_SubPDUList.d) { adbgPrintLog(_ctx, "{ "); };
	_adbgNrDrb__NR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_DL_Type_SDU_SubPDUList_Optional(_ctx, &p->SDU_SubPDUList);
	if (p->SDU_SubPDUList.d) { adbgPrintLog(_ctx, " }"); };
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Padding_SubPDU := ");
	if (p->Padding_SubPDU.d) { adbgPrintLog(_ctx, "{ "); };
	_adbgNrDrb__NR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_DL_Type_Padding_SubPDU_Optional(_ctx, &p->Padding_SubPDU);
	if (p->Padding_SubPDU.d) { adbgPrintLog(_ctx, " }"); };
}

static void _adbgNrDrb__NR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_UL_Type_SDU_SubPDUList_Optional(acpCtx_t _ctx, const struct NR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_UL_Type_SDU_SubPDUList_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	for (size_t i3 = 0; i3 < p->v.d; i3++) {
		_adbgNrDrb__NR_MAC_SDU_SubPDU_Type(_ctx, &p->v.v[i3]);
		if (i3 != p->v.d - 1) { adbgPrintLog(_ctx, ", "); }
	}
}

static void _adbgNrDrb__NR_MAC_CE_ShortBSR_Type(acpCtx_t _ctx, const struct NR_MAC_CE_ShortBSR_Type* p)
{
	adbgPrintLog(_ctx, "LCG := '");
	for (size_t i4 = 0; i4 < 3; i4++) {
		adbgPrintLog(_ctx, "%02X", p->LCG[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "BufferSize := '");
	for (size_t i4 = 0; i4 < 5; i4++) {
		adbgPrintLog(_ctx, "%02X", p->BufferSize[i4]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_MAC_CE_LongBSR_Type(acpCtx_t _ctx, const struct NR_MAC_CE_LongBSR_Type* p)
{
	adbgPrintLog(_ctx, "LCG_Presence := '");
	for (size_t i4 = 0; i4 < 8; i4++) {
		adbgPrintLog(_ctx, "%02X", p->LCG_Presence[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "{");
	for (size_t i4 = 0; i4 < p->BufferSizeList.d; i4++) {
		adbgPrintLog(_ctx, "'");
		for (size_t i5 = 0; i5 < 1; i5++) {
			adbgPrintLog(_ctx, "%02X", p->BufferSizeList.v[i4][i5]);
		}
		adbgPrintLog(_ctx, "'O");
		if (i4 != p->BufferSizeList.d - 1) { adbgPrintLog(_ctx, ", "); }
	}
	adbgPrintLog(_ctx, " }");
}

static void _adbgNrDrb__B2_Type_NR_MAC_CE_PH_Record_Type_MPE_or_R_Optional(acpCtx_t _ctx, const struct B2_Type_NR_MAC_CE_PH_Record_Type_MPE_or_R_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	adbgPrintLog(_ctx, "'");
	for (size_t i4 = 0; i4 < 2; i4++) {
		adbgPrintLog(_ctx, "%02X", p->v[i4]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__B6_Type_NR_MAC_CE_PH_Record_Type_PCMaxc_Optional(acpCtx_t _ctx, const struct B6_Type_NR_MAC_CE_PH_Record_Type_PCMaxc_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	adbgPrintLog(_ctx, "'");
	for (size_t i4 = 0; i4 < 6; i4++) {
		adbgPrintLog(_ctx, "%02X", p->v[i4]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_MAC_CE_SingleEntryPHR_Type(acpCtx_t _ctx, const NR_MAC_CE_SingleEntryPHR_Type* p)
{
	adbgPrintLog(_ctx, "P_Bit := '");
	for (size_t i4 = 0; i4 < 1; i4++) {
		adbgPrintLog(_ctx, "%02X", p->P_Bit[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "V_Bit := '");
	for (size_t i4 = 0; i4 < 1; i4++) {
		adbgPrintLog(_ctx, "%02X", p->V_Bit[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Value := '");
	for (size_t i4 = 0; i4 < 6; i4++) {
		adbgPrintLog(_ctx, "%02X", p->Value[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "MPE_or_R := ");
	_adbgNrDrb__B2_Type_NR_MAC_CE_PH_Record_Type_MPE_or_R_Optional(_ctx, &p->MPE_or_R);
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "PCMaxc := ");
	_adbgNrDrb__B6_Type_NR_MAC_CE_PH_Record_Type_PCMaxc_Optional(_ctx, &p->PCMaxc);
}

static void _adbgNrDrb__NR_MAC_CE_SCellFlags_Type(acpCtx_t _ctx, const struct NR_MAC_CE_SCellFlags_Type* p)
{
	adbgPrintLog(_ctx, "SCellIndex7_1 := '");
	for (size_t i4 = 0; i4 < 8; i4++) {
		adbgPrintLog(_ctx, "%02X", p->SCellIndex7_1[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SCellIndex15_8 := ");
	_adbgNrDrb__B8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex15_8_Optional(_ctx, &p->SCellIndex15_8);
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SCellIndex23_16 := ");
	_adbgNrDrb__B8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex23_16_Optional(_ctx, &p->SCellIndex23_16);
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SCellIndex31_24 := ");
	_adbgNrDrb__B8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex31_24_Optional(_ctx, &p->SCellIndex31_24);
}

static void _adbgNrDrb__NR_MAC_CE_PH_Record_Type(acpCtx_t _ctx, const struct NR_MAC_CE_PH_Record_Type* p)
{
	adbgPrintLog(_ctx, "P_Bit := '");
	for (size_t i5 = 0; i5 < 1; i5++) {
		adbgPrintLog(_ctx, "%02X", p->P_Bit[i5]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "V_Bit := '");
	for (size_t i5 = 0; i5 < 1; i5++) {
		adbgPrintLog(_ctx, "%02X", p->V_Bit[i5]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Value := '");
	for (size_t i5 = 0; i5 < 6; i5++) {
		adbgPrintLog(_ctx, "%02X", p->Value[i5]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "MPE_or_R := ");
	_adbgNrDrb__B2_Type_NR_MAC_CE_PH_Record_Type_MPE_or_R_Optional(_ctx, &p->MPE_or_R);
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "PCMaxc := ");
	_adbgNrDrb__B6_Type_NR_MAC_CE_PH_Record_Type_PCMaxc_Optional(_ctx, &p->PCMaxc);
}

static void _adbgNrDrb__NR_MAC_CE_MultiEntryPHR_Type(acpCtx_t _ctx, const struct NR_MAC_CE_MultiEntryPHR_Type* p)
{
	adbgPrintLog(_ctx, "PHFieldPresentForSCell := { ");
	_adbgNrDrb__NR_MAC_CE_SCellFlags_Type(_ctx, &p->PHFieldPresentForSCell);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "PH_Record := { ");
	for (size_t i4 = 0; i4 < p->PH_Record.d; i4++) {
		adbgPrintLog(_ctx, "{ ");
		_adbgNrDrb__NR_MAC_CE_PH_Record_Type(_ctx, &p->PH_Record.v[i4]);
		adbgPrintLog(_ctx, " }");
		if (i4 != p->PH_Record.d - 1) { adbgPrintLog(_ctx, ", "); }
	}
	adbgPrintLog(_ctx, " }");
}

static void _adbgNrDrb__NR_MAC_ControlElementUL_Type_Value(acpCtx_t _ctx, const union NR_MAC_ControlElementUL_Type_Value* p, enum NR_MAC_ControlElementUL_Type_Sel d)
{
	if (d == NR_MAC_ControlElementUL_Type_ShortBSR) {
		adbgPrintLog(_ctx, "ShortBSR := { ");
		_adbgNrDrb__NR_MAC_CE_ShortBSR_Type(_ctx, &p->ShortBSR);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_MAC_ControlElementUL_Type_LongBSR) {
		adbgPrintLog(_ctx, "LongBSR := { ");
		_adbgNrDrb__NR_MAC_CE_LongBSR_Type(_ctx, &p->LongBSR);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_MAC_ControlElementUL_Type_C_RNTI) {
		adbgPrintLog(_ctx, "C_RNTI := '");
		for (size_t i4 = 0; i4 < 16; i4++) {
			adbgPrintLog(_ctx, "%02X", p->C_RNTI[i4]);
		}
		adbgPrintLog(_ctx, "'O");
		return;
	}
	if (d == NR_MAC_ControlElementUL_Type_SingleEntryPHR) {
		adbgPrintLog(_ctx, "SingleEntryPHR := { ");
		_adbgNrDrb__NR_MAC_CE_SingleEntryPHR_Type(_ctx, &p->SingleEntryPHR);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_MAC_ControlElementUL_Type_MultiEntryPHR) {
		adbgPrintLog(_ctx, "MultiEntryPHR := { ");
		_adbgNrDrb__NR_MAC_CE_MultiEntryPHR_Type(_ctx, &p->MultiEntryPHR);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_MAC_ControlElementUL_Type_RecommendedBitrate) {
		adbgPrintLog(_ctx, "RecommendedBitrate := { ");
		_adbgNrDrb__NR_MAC_CE_RecommendedBitrate_Type(_ctx, &p->RecommendedBitrate);
		adbgPrintLog(_ctx, " }");
		return;
	}
	adbgPrintLog(_ctx, "INVALID");
}

static void _adbgNrDrb__NR_MAC_ControlElementUL_Type(acpCtx_t _ctx, const struct NR_MAC_ControlElementUL_Type* p)
{
	_adbgNrDrb__NR_MAC_ControlElementUL_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrDrb__NR_MAC_ControlElementUL_Type_NR_MAC_CE_SubPDU_UL_Type_ControlElement_Optional(acpCtx_t _ctx, const struct NR_MAC_ControlElementUL_Type_NR_MAC_CE_SubPDU_UL_Type_ControlElement_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	_adbgNrDrb__NR_MAC_ControlElementUL_Type(_ctx, &p->v);
}

static void _adbgNrDrb__NR_MAC_CE_SubPDU_UL_Type(acpCtx_t _ctx, const struct NR_MAC_CE_SubPDU_UL_Type* p)
{
	adbgPrintLog(_ctx, "SubHeader := { ");
	_adbgNrDrb__NR_MAC_PDU_SubHeader_Type(_ctx, &p->SubHeader);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "ControlElement := ");
	if (p->ControlElement.d) { adbgPrintLog(_ctx, "{ "); };
	_adbgNrDrb__NR_MAC_ControlElementUL_Type_NR_MAC_CE_SubPDU_UL_Type_ControlElement_Optional(_ctx, &p->ControlElement);
	if (p->ControlElement.d) { adbgPrintLog(_ctx, " }"); };
}

static void _adbgNrDrb__NR_MAC_CE_SubPDU_UL_List_Type_NR_MAC_PDU_UL_Type_CE_SubPDUList_Optional(acpCtx_t _ctx, const struct NR_MAC_CE_SubPDU_UL_List_Type_NR_MAC_PDU_UL_Type_CE_SubPDUList_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	for (size_t i3 = 0; i3 < p->v.d; i3++) {
		_adbgNrDrb__NR_MAC_CE_SubPDU_UL_Type(_ctx, &p->v.v[i3]);
		if (i3 != p->v.d - 1) { adbgPrintLog(_ctx, ", "); }
	}
}

static void _adbgNrDrb__NR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_UL_Type_Padding_SubPDU_Optional(acpCtx_t _ctx, const struct NR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_UL_Type_Padding_SubPDU_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	_adbgNrDrb__NR_MAC_Padding_SubPDU_Type(_ctx, &p->v);
}

static void _adbgNrDrb__NR_MAC_PDU_UL_Type(acpCtx_t _ctx, const struct NR_MAC_PDU_UL_Type* p)
{
	adbgPrintLog(_ctx, "SDU_SubPDUList := ");
	if (p->SDU_SubPDUList.d) { adbgPrintLog(_ctx, "{ "); };
	_adbgNrDrb__NR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_UL_Type_SDU_SubPDUList_Optional(_ctx, &p->SDU_SubPDUList);
	if (p->SDU_SubPDUList.d) { adbgPrintLog(_ctx, " }"); };
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "CE_SubPDUList := ");
	if (p->CE_SubPDUList.d) { adbgPrintLog(_ctx, "{ "); };
	_adbgNrDrb__NR_MAC_CE_SubPDU_UL_List_Type_NR_MAC_PDU_UL_Type_CE_SubPDUList_Optional(_ctx, &p->CE_SubPDUList);
	if (p->CE_SubPDUList.d) { adbgPrintLog(_ctx, " }"); };
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Padding_SubPDU := ");
	if (p->Padding_SubPDU.d) { adbgPrintLog(_ctx, "{ "); };
	_adbgNrDrb__NR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_UL_Type_Padding_SubPDU_Optional(_ctx, &p->Padding_SubPDU);
	if (p->Padding_SubPDU.d) { adbgPrintLog(_ctx, " }"); };
}

static void _adbgNrDrb__NR_MAC_PDU_Type_Value(acpCtx_t _ctx, const union NR_MAC_PDU_Type_Value* p, enum NR_MAC_PDU_Type_Sel d)
{
	if (d == NR_MAC_PDU_Type_DL) {
		adbgPrintLog(_ctx, "DL := { ");
		_adbgNrDrb__NR_MAC_PDU_DL_Type(_ctx, &p->DL);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_MAC_PDU_Type_UL) {
		adbgPrintLog(_ctx, "UL := { ");
		_adbgNrDrb__NR_MAC_PDU_UL_Type(_ctx, &p->UL);
		adbgPrintLog(_ctx, " }");
		return;
	}
	adbgPrintLog(_ctx, "INVALID");
}

static void _adbgNrDrb__NR_MAC_PDU_Type(acpCtx_t _ctx, const struct NR_MAC_PDU_Type* p)
{
	_adbgNrDrb__NR_MAC_PDU_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrDrb__NR_RLC_UMD_HeaderNoSN_Type(acpCtx_t _ctx, const struct NR_RLC_UMD_HeaderNoSN_Type* p)
{
	adbgPrintLog(_ctx, "SegmentationInfo := '");
	for (size_t i3 = 0; i3 < 2; i3++) {
		adbgPrintLog(_ctx, "%02X", p->SegmentationInfo[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Reserved := '");
	for (size_t i3 = 0; i3 < 6; i3++) {
		adbgPrintLog(_ctx, "%02X", p->Reserved[i3]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_RLC_UMD_PduNoSN_Type(acpCtx_t _ctx, const struct NR_RLC_UMD_PduNoSN_Type* p)
{
	adbgPrintLog(_ctx, "Header := { ");
	_adbgNrDrb__NR_RLC_UMD_HeaderNoSN_Type(_ctx, &p->Header);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Data := '");
	for (size_t i3 = 0; i3 < p->Data.d; i3++) {
		adbgPrintLog(_ctx, "%02X", p->Data.v[i3]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_RLC_SegmentOffset_Type_NR_RLC_UMD_HeaderSN6Bit_Type_SegmentOffset_Optional(acpCtx_t _ctx, const struct NR_RLC_SegmentOffset_Type_NR_RLC_UMD_HeaderSN6Bit_Type_SegmentOffset_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	adbgPrintLog(_ctx, "'");
	for (size_t i3 = 0; i3 < 16; i3++) {
		adbgPrintLog(_ctx, "%02X", p->v[i3]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_RLC_UMD_HeaderSN6Bit_Type(acpCtx_t _ctx, const struct NR_RLC_UMD_HeaderSN6Bit_Type* p)
{
	adbgPrintLog(_ctx, "SegmentationInfo := '");
	for (size_t i3 = 0; i3 < 2; i3++) {
		adbgPrintLog(_ctx, "%02X", p->SegmentationInfo[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SequenceNumber := '");
	for (size_t i3 = 0; i3 < 6; i3++) {
		adbgPrintLog(_ctx, "%02X", p->SequenceNumber[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SegmentOffset := ");
	_adbgNrDrb__NR_RLC_SegmentOffset_Type_NR_RLC_UMD_HeaderSN6Bit_Type_SegmentOffset_Optional(_ctx, &p->SegmentOffset);
}

static void _adbgNrDrb__NR_RLC_UMD_PduSN6Bit_Type(acpCtx_t _ctx, const struct NR_RLC_UMD_PduSN6Bit_Type* p)
{
	adbgPrintLog(_ctx, "Header := { ");
	_adbgNrDrb__NR_RLC_UMD_HeaderSN6Bit_Type(_ctx, &p->Header);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Data := '");
	for (size_t i3 = 0; i3 < p->Data.d; i3++) {
		adbgPrintLog(_ctx, "%02X", p->Data.v[i3]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_RLC_SegmentOffset_Type_NR_RLC_UMD_HeaderSN12Bit_Type_SegmentOffset_Optional(acpCtx_t _ctx, const struct NR_RLC_SegmentOffset_Type_NR_RLC_UMD_HeaderSN12Bit_Type_SegmentOffset_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	adbgPrintLog(_ctx, "'");
	for (size_t i3 = 0; i3 < 16; i3++) {
		adbgPrintLog(_ctx, "%02X", p->v[i3]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_RLC_UMD_HeaderSN12Bit_Type(acpCtx_t _ctx, const struct NR_RLC_UMD_HeaderSN12Bit_Type* p)
{
	adbgPrintLog(_ctx, "SegmentationInfo := '");
	for (size_t i3 = 0; i3 < 2; i3++) {
		adbgPrintLog(_ctx, "%02X", p->SegmentationInfo[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Reserved := '");
	for (size_t i3 = 0; i3 < 2; i3++) {
		adbgPrintLog(_ctx, "%02X", p->Reserved[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SequenceNumber := '");
	for (size_t i3 = 0; i3 < 12; i3++) {
		adbgPrintLog(_ctx, "%02X", p->SequenceNumber[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SegmentOffset := ");
	_adbgNrDrb__NR_RLC_SegmentOffset_Type_NR_RLC_UMD_HeaderSN12Bit_Type_SegmentOffset_Optional(_ctx, &p->SegmentOffset);
}

static void _adbgNrDrb__NR_RLC_UMD_PduSN12Bit_Type(acpCtx_t _ctx, const struct NR_RLC_UMD_PduSN12Bit_Type* p)
{
	adbgPrintLog(_ctx, "Header := { ");
	_adbgNrDrb__NR_RLC_UMD_HeaderSN12Bit_Type(_ctx, &p->Header);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Data := '");
	for (size_t i3 = 0; i3 < p->Data.d; i3++) {
		adbgPrintLog(_ctx, "%02X", p->Data.v[i3]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_RLC_UMD_PDU_Type_Value(acpCtx_t _ctx, const union NR_RLC_UMD_PDU_Type_Value* p, enum NR_RLC_UMD_PDU_Type_Sel d)
{
	if (d == NR_RLC_UMD_PDU_Type_NoSN) {
		adbgPrintLog(_ctx, "NoSN := { ");
		_adbgNrDrb__NR_RLC_UMD_PduNoSN_Type(_ctx, &p->NoSN);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_RLC_UMD_PDU_Type_SN6Bit) {
		adbgPrintLog(_ctx, "SN6Bit := { ");
		_adbgNrDrb__NR_RLC_UMD_PduSN6Bit_Type(_ctx, &p->SN6Bit);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_RLC_UMD_PDU_Type_SN12Bit) {
		adbgPrintLog(_ctx, "SN12Bit := { ");
		_adbgNrDrb__NR_RLC_UMD_PduSN12Bit_Type(_ctx, &p->SN12Bit);
		adbgPrintLog(_ctx, " }");
		return;
	}
	adbgPrintLog(_ctx, "INVALID");
}

static void _adbgNrDrb__NR_RLC_UMD_PDU_Type(acpCtx_t _ctx, const struct NR_RLC_UMD_PDU_Type* p)
{
	_adbgNrDrb__NR_RLC_UMD_PDU_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrDrb__NR_RLC_SegmentOffset_Type_NR_RLC_AMD_HeaderSN12Bit_Type_SegmentOffset_Optional(acpCtx_t _ctx, const struct NR_RLC_SegmentOffset_Type_NR_RLC_AMD_HeaderSN12Bit_Type_SegmentOffset_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	adbgPrintLog(_ctx, "'");
	for (size_t i3 = 0; i3 < 16; i3++) {
		adbgPrintLog(_ctx, "%02X", p->v[i3]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_RLC_AMD_HeaderSN12Bit_Type(acpCtx_t _ctx, const struct NR_RLC_AMD_HeaderSN12Bit_Type* p)
{
	adbgPrintLog(_ctx, "D_C := '");
	for (size_t i3 = 0; i3 < 1; i3++) {
		adbgPrintLog(_ctx, "%02X", p->D_C[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Poll := '");
	for (size_t i3 = 0; i3 < 1; i3++) {
		adbgPrintLog(_ctx, "%02X", p->Poll[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SegmentationInfo := '");
	for (size_t i3 = 0; i3 < 2; i3++) {
		adbgPrintLog(_ctx, "%02X", p->SegmentationInfo[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SequenceNumber := '");
	for (size_t i3 = 0; i3 < 12; i3++) {
		adbgPrintLog(_ctx, "%02X", p->SequenceNumber[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SegmentOffset := ");
	_adbgNrDrb__NR_RLC_SegmentOffset_Type_NR_RLC_AMD_HeaderSN12Bit_Type_SegmentOffset_Optional(_ctx, &p->SegmentOffset);
}

static void _adbgNrDrb__NR_RLC_AMD_PduSN12Bit_Type(acpCtx_t _ctx, const struct NR_RLC_AMD_PduSN12Bit_Type* p)
{
	adbgPrintLog(_ctx, "Header := { ");
	_adbgNrDrb__NR_RLC_AMD_HeaderSN12Bit_Type(_ctx, &p->Header);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Data := '");
	for (size_t i3 = 0; i3 < p->Data.d; i3++) {
		adbgPrintLog(_ctx, "%02X", p->Data.v[i3]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_RLC_SegmentOffset_Type_NR_RLC_AMD_HeaderSN18Bit_Type_SegmentOffset_Optional(acpCtx_t _ctx, const struct NR_RLC_SegmentOffset_Type_NR_RLC_AMD_HeaderSN18Bit_Type_SegmentOffset_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	adbgPrintLog(_ctx, "'");
	for (size_t i3 = 0; i3 < 16; i3++) {
		adbgPrintLog(_ctx, "%02X", p->v[i3]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_RLC_AMD_HeaderSN18Bit_Type(acpCtx_t _ctx, const struct NR_RLC_AMD_HeaderSN18Bit_Type* p)
{
	adbgPrintLog(_ctx, "D_C := '");
	for (size_t i3 = 0; i3 < 1; i3++) {
		adbgPrintLog(_ctx, "%02X", p->D_C[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Poll := '");
	for (size_t i3 = 0; i3 < 1; i3++) {
		adbgPrintLog(_ctx, "%02X", p->Poll[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SegmentationInfo := '");
	for (size_t i3 = 0; i3 < 2; i3++) {
		adbgPrintLog(_ctx, "%02X", p->SegmentationInfo[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Reserved := '");
	for (size_t i3 = 0; i3 < 2; i3++) {
		adbgPrintLog(_ctx, "%02X", p->Reserved[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SequenceNumber := '");
	for (size_t i3 = 0; i3 < 18; i3++) {
		adbgPrintLog(_ctx, "%02X", p->SequenceNumber[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SegmentOffset := ");
	_adbgNrDrb__NR_RLC_SegmentOffset_Type_NR_RLC_AMD_HeaderSN18Bit_Type_SegmentOffset_Optional(_ctx, &p->SegmentOffset);
}

static void _adbgNrDrb__NR_RLC_AMD_PduSN18Bit_Type(acpCtx_t _ctx, const struct NR_RLC_AMD_PduSN18Bit_Type* p)
{
	adbgPrintLog(_ctx, "Header := { ");
	_adbgNrDrb__NR_RLC_AMD_HeaderSN18Bit_Type(_ctx, &p->Header);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Data := '");
	for (size_t i3 = 0; i3 < p->Data.d; i3++) {
		adbgPrintLog(_ctx, "%02X", p->Data.v[i3]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_RLC_AMD_PDU_Type_Value(acpCtx_t _ctx, const union NR_RLC_AMD_PDU_Type_Value* p, enum NR_RLC_AMD_PDU_Type_Sel d)
{
	if (d == NR_RLC_AMD_PDU_Type_SN12Bit) {
		adbgPrintLog(_ctx, "SN12Bit := { ");
		_adbgNrDrb__NR_RLC_AMD_PduSN12Bit_Type(_ctx, &p->SN12Bit);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_RLC_AMD_PDU_Type_SN18Bit) {
		adbgPrintLog(_ctx, "SN18Bit := { ");
		_adbgNrDrb__NR_RLC_AMD_PduSN18Bit_Type(_ctx, &p->SN18Bit);
		adbgPrintLog(_ctx, " }");
		return;
	}
	adbgPrintLog(_ctx, "INVALID");
}

static void _adbgNrDrb__NR_RLC_AMD_PDU_Type(acpCtx_t _ctx, const struct NR_RLC_AMD_PDU_Type* p)
{
	_adbgNrDrb__NR_RLC_AMD_PDU_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrDrb__NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN12Bit_Type_SOstart_Optional(acpCtx_t _ctx, const struct NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN12Bit_Type_SOstart_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	adbgPrintLog(_ctx, "'");
	for (size_t i4 = 0; i4 < 16; i4++) {
		adbgPrintLog(_ctx, "%02X", p->v[i4]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN12Bit_Type_SOstop_Optional(acpCtx_t _ctx, const struct NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN12Bit_Type_SOstop_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	adbgPrintLog(_ctx, "'");
	for (size_t i4 = 0; i4 < 16; i4++) {
		adbgPrintLog(_ctx, "%02X", p->v[i4]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__B8_Type_NR_RLC_Status_NackSN12Bit_Type_NACKrange_Optional(acpCtx_t _ctx, const struct B8_Type_NR_RLC_Status_NackSN12Bit_Type_NACKrange_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	adbgPrintLog(_ctx, "'");
	for (size_t i4 = 0; i4 < 8; i4++) {
		adbgPrintLog(_ctx, "%02X", p->v[i4]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_RLC_Status_NackSN12Bit_Type(acpCtx_t _ctx, const struct NR_RLC_Status_NackSN12Bit_Type* p)
{
	adbgPrintLog(_ctx, "SequenceNumberNACK := '");
	for (size_t i4 = 0; i4 < 12; i4++) {
		adbgPrintLog(_ctx, "%02X", p->SequenceNumberNACK[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "E1 := '");
	for (size_t i4 = 0; i4 < 1; i4++) {
		adbgPrintLog(_ctx, "%02X", p->E1[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "E2 := '");
	for (size_t i4 = 0; i4 < 1; i4++) {
		adbgPrintLog(_ctx, "%02X", p->E2[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "E3 := '");
	for (size_t i4 = 0; i4 < 1; i4++) {
		adbgPrintLog(_ctx, "%02X", p->E3[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Reserved := '");
	for (size_t i4 = 0; i4 < 1; i4++) {
		adbgPrintLog(_ctx, "%02X", p->Reserved[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SOstart := ");
	_adbgNrDrb__NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN12Bit_Type_SOstart_Optional(_ctx, &p->SOstart);
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SOstop := ");
	_adbgNrDrb__NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN12Bit_Type_SOstop_Optional(_ctx, &p->SOstop);
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "NACKrange := ");
	_adbgNrDrb__B8_Type_NR_RLC_Status_NackSN12Bit_Type_NACKrange_Optional(_ctx, &p->NACKrange);
}

static void _adbgNrDrb__NR_RLC_Status_NackListSN12Bit_Type_NackList_Optional(acpCtx_t _ctx, const struct NR_RLC_Status_NackListSN12Bit_Type_NackList_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	for (size_t i3 = 0; i3 < p->v.d; i3++) {
		_adbgNrDrb__NR_RLC_Status_NackSN12Bit_Type(_ctx, &p->v.v[i3]);
		if (i3 != p->v.d - 1) { adbgPrintLog(_ctx, ", "); }
	}
}

static void _adbgNrDrb__NR_RLC_StatusPduSN12Bit_Type(acpCtx_t _ctx, const struct NR_RLC_StatusPduSN12Bit_Type* p)
{
	adbgPrintLog(_ctx, "D_C := '");
	for (size_t i3 = 0; i3 < 1; i3++) {
		adbgPrintLog(_ctx, "%02X", p->D_C[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "CPT := '");
	for (size_t i3 = 0; i3 < 3; i3++) {
		adbgPrintLog(_ctx, "%02X", p->CPT[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SequenceNumberACK := '");
	for (size_t i3 = 0; i3 < 12; i3++) {
		adbgPrintLog(_ctx, "%02X", p->SequenceNumberACK[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "E1 := '");
	for (size_t i3 = 0; i3 < 1; i3++) {
		adbgPrintLog(_ctx, "%02X", p->E1[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Reserved := '");
	for (size_t i3 = 0; i3 < 7; i3++) {
		adbgPrintLog(_ctx, "%02X", p->Reserved[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "NackList := ");
	if (p->NackList.d) { adbgPrintLog(_ctx, "{ "); };
	_adbgNrDrb__NR_RLC_Status_NackListSN12Bit_Type_NackList_Optional(_ctx, &p->NackList);
	if (p->NackList.d) { adbgPrintLog(_ctx, " }"); };
}

static void _adbgNrDrb__NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN18Bit_Type_SOstart_Optional(acpCtx_t _ctx, const struct NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN18Bit_Type_SOstart_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	adbgPrintLog(_ctx, "'");
	for (size_t i4 = 0; i4 < 16; i4++) {
		adbgPrintLog(_ctx, "%02X", p->v[i4]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN18Bit_Type_SOstop_Optional(acpCtx_t _ctx, const struct NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN18Bit_Type_SOstop_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	adbgPrintLog(_ctx, "'");
	for (size_t i4 = 0; i4 < 16; i4++) {
		adbgPrintLog(_ctx, "%02X", p->v[i4]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__B8_Type_NR_RLC_Status_NackSN18Bit_Type_NACKrange_Optional(acpCtx_t _ctx, const struct B8_Type_NR_RLC_Status_NackSN18Bit_Type_NACKrange_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	adbgPrintLog(_ctx, "'");
	for (size_t i4 = 0; i4 < 8; i4++) {
		adbgPrintLog(_ctx, "%02X", p->v[i4]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_RLC_Status_NackSN18Bit_Type(acpCtx_t _ctx, const struct NR_RLC_Status_NackSN18Bit_Type* p)
{
	adbgPrintLog(_ctx, "SequenceNumberNACK := '");
	for (size_t i4 = 0; i4 < 18; i4++) {
		adbgPrintLog(_ctx, "%02X", p->SequenceNumberNACK[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "E1 := '");
	for (size_t i4 = 0; i4 < 1; i4++) {
		adbgPrintLog(_ctx, "%02X", p->E1[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "E2 := '");
	for (size_t i4 = 0; i4 < 1; i4++) {
		adbgPrintLog(_ctx, "%02X", p->E2[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "E3 := '");
	for (size_t i4 = 0; i4 < 1; i4++) {
		adbgPrintLog(_ctx, "%02X", p->E3[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Reserved := '");
	for (size_t i4 = 0; i4 < 3; i4++) {
		adbgPrintLog(_ctx, "%02X", p->Reserved[i4]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SOstart := ");
	_adbgNrDrb__NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN18Bit_Type_SOstart_Optional(_ctx, &p->SOstart);
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SOstop := ");
	_adbgNrDrb__NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN18Bit_Type_SOstop_Optional(_ctx, &p->SOstop);
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "NACKrange := ");
	_adbgNrDrb__B8_Type_NR_RLC_Status_NackSN18Bit_Type_NACKrange_Optional(_ctx, &p->NACKrange);
}

static void _adbgNrDrb__NR_RLC_Status_NackListSN18Bit_Type_NackList_Optional(acpCtx_t _ctx, const struct NR_RLC_Status_NackListSN18Bit_Type_NackList_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	for (size_t i3 = 0; i3 < p->v.d; i3++) {
		_adbgNrDrb__NR_RLC_Status_NackSN18Bit_Type(_ctx, &p->v.v[i3]);
		if (i3 != p->v.d - 1) { adbgPrintLog(_ctx, ", "); }
	}
}

static void _adbgNrDrb__NR_RLC_StatusPduSN18Bit_Type(acpCtx_t _ctx, const struct NR_RLC_StatusPduSN18Bit_Type* p)
{
	adbgPrintLog(_ctx, "D_C := '");
	for (size_t i3 = 0; i3 < 1; i3++) {
		adbgPrintLog(_ctx, "%02X", p->D_C[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "CPT := '");
	for (size_t i3 = 0; i3 < 3; i3++) {
		adbgPrintLog(_ctx, "%02X", p->CPT[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SequenceNumberACK := '");
	for (size_t i3 = 0; i3 < 18; i3++) {
		adbgPrintLog(_ctx, "%02X", p->SequenceNumberACK[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "E1 := '");
	for (size_t i3 = 0; i3 < 1; i3++) {
		adbgPrintLog(_ctx, "%02X", p->E1[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Reserved := '");
	for (size_t i3 = 0; i3 < 1; i3++) {
		adbgPrintLog(_ctx, "%02X", p->Reserved[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "NackList := ");
	if (p->NackList.d) { adbgPrintLog(_ctx, "{ "); };
	_adbgNrDrb__NR_RLC_Status_NackListSN18Bit_Type_NackList_Optional(_ctx, &p->NackList);
	if (p->NackList.d) { adbgPrintLog(_ctx, " }"); };
}

static void _adbgNrDrb__NR_RLC_AM_StatusPDU_Type_Value(acpCtx_t _ctx, const union NR_RLC_AM_StatusPDU_Type_Value* p, enum NR_RLC_AM_StatusPDU_Type_Sel d)
{
	if (d == NR_RLC_AM_StatusPDU_Type_SN12Bit) {
		adbgPrintLog(_ctx, "SN12Bit := { ");
		_adbgNrDrb__NR_RLC_StatusPduSN12Bit_Type(_ctx, &p->SN12Bit);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_RLC_AM_StatusPDU_Type_SN18Bit) {
		adbgPrintLog(_ctx, "SN18Bit := { ");
		_adbgNrDrb__NR_RLC_StatusPduSN18Bit_Type(_ctx, &p->SN18Bit);
		adbgPrintLog(_ctx, " }");
		return;
	}
	adbgPrintLog(_ctx, "INVALID");
}

static void _adbgNrDrb__NR_RLC_AM_StatusPDU_Type(acpCtx_t _ctx, const struct NR_RLC_AM_StatusPDU_Type* p)
{
	_adbgNrDrb__NR_RLC_AM_StatusPDU_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrDrb__NR_RLC_PDU_Type_Value(acpCtx_t _ctx, const union NR_RLC_PDU_Type_Value* p, enum NR_RLC_PDU_Type_Sel d)
{
	if (d == NR_RLC_PDU_Type_TMD) {
		adbgPrintLog(_ctx, "TMD := '");
		for (size_t i3 = 0; i3 < p->TMD.d; i3++) {
			adbgPrintLog(_ctx, "%02X", p->TMD.v[i3]);
		}
		adbgPrintLog(_ctx, "'O");
		return;
	}
	if (d == NR_RLC_PDU_Type_UMD) {
		adbgPrintLog(_ctx, "UMD := { ");
		_adbgNrDrb__NR_RLC_UMD_PDU_Type(_ctx, &p->UMD);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_RLC_PDU_Type_AMD) {
		adbgPrintLog(_ctx, "AMD := { ");
		_adbgNrDrb__NR_RLC_AMD_PDU_Type(_ctx, &p->AMD);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_RLC_PDU_Type_Status) {
		adbgPrintLog(_ctx, "Status := { ");
		_adbgNrDrb__NR_RLC_AM_StatusPDU_Type(_ctx, &p->Status);
		adbgPrintLog(_ctx, " }");
		return;
	}
	adbgPrintLog(_ctx, "INVALID");
}

static void _adbgNrDrb__NR_RLC_PDU_Type(acpCtx_t _ctx, const struct NR_RLC_PDU_Type* p)
{
	_adbgNrDrb__NR_RLC_PDU_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrDrb__B32_Type_NR_PDCP_DataPduSN12Bits_Type_MAC_I_Optional(acpCtx_t _ctx, const struct B32_Type_NR_PDCP_DataPduSN12Bits_Type_MAC_I_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	adbgPrintLog(_ctx, "'");
	for (size_t i3 = 0; i3 < 32; i3++) {
		adbgPrintLog(_ctx, "%02X", p->v[i3]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_PDCP_DataPduSN12Bits_Type(acpCtx_t _ctx, const struct NR_PDCP_DataPduSN12Bits_Type* p)
{
	adbgPrintLog(_ctx, "D_C := '");
	for (size_t i3 = 0; i3 < 1; i3++) {
		adbgPrintLog(_ctx, "%02X", p->D_C[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Reserved := '");
	for (size_t i3 = 0; i3 < 3; i3++) {
		adbgPrintLog(_ctx, "%02X", p->Reserved[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SequenceNumber := '");
	for (size_t i3 = 0; i3 < 12; i3++) {
		adbgPrintLog(_ctx, "%02X", p->SequenceNumber[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SDU := '");
	for (size_t i3 = 0; i3 < p->SDU.d; i3++) {
		adbgPrintLog(_ctx, "%02X", p->SDU.v[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "MAC_I := ");
	_adbgNrDrb__B32_Type_NR_PDCP_DataPduSN12Bits_Type_MAC_I_Optional(_ctx, &p->MAC_I);
}

static void _adbgNrDrb__B32_Type_NR_PDCP_DataPduSN18Bits_Type_MAC_I_Optional(acpCtx_t _ctx, const struct B32_Type_NR_PDCP_DataPduSN18Bits_Type_MAC_I_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	adbgPrintLog(_ctx, "'");
	for (size_t i3 = 0; i3 < 32; i3++) {
		adbgPrintLog(_ctx, "%02X", p->v[i3]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_PDCP_DataPduSN18Bits_Type(acpCtx_t _ctx, const struct NR_PDCP_DataPduSN18Bits_Type* p)
{
	adbgPrintLog(_ctx, "D_C := '");
	for (size_t i3 = 0; i3 < 1; i3++) {
		adbgPrintLog(_ctx, "%02X", p->D_C[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Reserved := '");
	for (size_t i3 = 0; i3 < 5; i3++) {
		adbgPrintLog(_ctx, "%02X", p->Reserved[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SequenceNumber := '");
	for (size_t i3 = 0; i3 < 18; i3++) {
		adbgPrintLog(_ctx, "%02X", p->SequenceNumber[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SDU := '");
	for (size_t i3 = 0; i3 < p->SDU.d; i3++) {
		adbgPrintLog(_ctx, "%02X", p->SDU.v[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "MAC_I := ");
	_adbgNrDrb__B32_Type_NR_PDCP_DataPduSN18Bits_Type_MAC_I_Optional(_ctx, &p->MAC_I);
}

static void _adbgNrDrb__OCTET_STRING_NR_PDCP_CtrlPduStatus_Type_Bitmap_Optional(acpCtx_t _ctx, const struct OCTET_STRING_NR_PDCP_CtrlPduStatus_Type_Bitmap_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	adbgPrintLog(_ctx, "'");
	for (size_t i3 = 0; i3 < p->v.d; i3++) {
		adbgPrintLog(_ctx, "%02X", p->v.v[i3]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_PDCP_CtrlPduStatus_Type(acpCtx_t _ctx, const struct NR_PDCP_CtrlPduStatus_Type* p)
{
	adbgPrintLog(_ctx, "D_C := '");
	for (size_t i3 = 0; i3 < 1; i3++) {
		adbgPrintLog(_ctx, "%02X", p->D_C[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "PDU_Type := '");
	for (size_t i3 = 0; i3 < 3; i3++) {
		adbgPrintLog(_ctx, "%02X", p->PDU_Type[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Reserved := '");
	for (size_t i3 = 0; i3 < 4; i3++) {
		adbgPrintLog(_ctx, "%02X", p->Reserved[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "FirstMissingCount := '");
	for (size_t i3 = 0; i3 < 32; i3++) {
		adbgPrintLog(_ctx, "%02X", p->FirstMissingCount[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Bitmap := ");
	if (p->Bitmap.d) { adbgPrintLog(_ctx, "{ "); };
	_adbgNrDrb__OCTET_STRING_NR_PDCP_CtrlPduStatus_Type_Bitmap_Optional(_ctx, &p->Bitmap);
	if (p->Bitmap.d) { adbgPrintLog(_ctx, " }"); };
}

static void _adbgNrDrb__NR_PDCP_CtrlPduRohcFeedback_Type(acpCtx_t _ctx, const struct NR_PDCP_CtrlPduRohcFeedback_Type* p)
{
	adbgPrintLog(_ctx, "D_C := '");
	for (size_t i3 = 0; i3 < 1; i3++) {
		adbgPrintLog(_ctx, "%02X", p->D_C[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "PDU_Type := '");
	for (size_t i3 = 0; i3 < 3; i3++) {
		adbgPrintLog(_ctx, "%02X", p->PDU_Type[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Reserved := '");
	for (size_t i3 = 0; i3 < 4; i3++) {
		adbgPrintLog(_ctx, "%02X", p->Reserved[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "RohcFeedback := '");
	for (size_t i3 = 0; i3 < p->RohcFeedback.d; i3++) {
		adbgPrintLog(_ctx, "%02X", p->RohcFeedback.v[i3]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__NR_PDCP_PDU_Type_Value(acpCtx_t _ctx, const union NR_PDCP_PDU_Type_Value* p, enum NR_PDCP_PDU_Type_Sel d)
{
	if (d == NR_PDCP_PDU_Type_DataPduSN12Bits) {
		adbgPrintLog(_ctx, "DataPduSN12Bits := { ");
		_adbgNrDrb__NR_PDCP_DataPduSN12Bits_Type(_ctx, &p->DataPduSN12Bits);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_PDCP_PDU_Type_DataPduSN18Bits) {
		adbgPrintLog(_ctx, "DataPduSN18Bits := { ");
		_adbgNrDrb__NR_PDCP_DataPduSN18Bits_Type(_ctx, &p->DataPduSN18Bits);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_PDCP_PDU_Type_CtrlPduStatus) {
		adbgPrintLog(_ctx, "CtrlPduStatus := { ");
		_adbgNrDrb__NR_PDCP_CtrlPduStatus_Type(_ctx, &p->CtrlPduStatus);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_PDCP_PDU_Type_CtrlPduRohcFeedback) {
		adbgPrintLog(_ctx, "CtrlPduRohcFeedback := { ");
		_adbgNrDrb__NR_PDCP_CtrlPduRohcFeedback_Type(_ctx, &p->CtrlPduRohcFeedback);
		adbgPrintLog(_ctx, " }");
		return;
	}
	adbgPrintLog(_ctx, "INVALID");
}

static void _adbgNrDrb__NR_PDCP_PDU_Type(acpCtx_t _ctx, const struct NR_PDCP_PDU_Type* p)
{
	_adbgNrDrb__NR_PDCP_PDU_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrDrb__SDAP_DL_PduHeader_Type(acpCtx_t _ctx, const struct SDAP_DL_PduHeader_Type* p)
{
	adbgPrintLog(_ctx, "RDI := '");
	for (size_t i3 = 0; i3 < 1; i3++) {
		adbgPrintLog(_ctx, "%02X", p->RDI[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "RQI := '");
	for (size_t i3 = 0; i3 < 1; i3++) {
		adbgPrintLog(_ctx, "%02X", p->RQI[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "QFI := '");
	for (size_t i3 = 0; i3 < 6; i3++) {
		adbgPrintLog(_ctx, "%02X", p->QFI[i3]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__SDAP_DL_PduHeader_Type_SDAP_PDU_DL_Type_Header_Optional(acpCtx_t _ctx, const struct SDAP_DL_PduHeader_Type_SDAP_PDU_DL_Type_Header_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	_adbgNrDrb__SDAP_DL_PduHeader_Type(_ctx, &p->v);
}

static void _adbgNrDrb__SDAP_PDU_DL_Type(acpCtx_t _ctx, const struct SDAP_PDU_DL_Type* p)
{
	adbgPrintLog(_ctx, "Header := ");
	if (p->Header.d) { adbgPrintLog(_ctx, "{ "); };
	_adbgNrDrb__SDAP_DL_PduHeader_Type_SDAP_PDU_DL_Type_Header_Optional(_ctx, &p->Header);
	if (p->Header.d) { adbgPrintLog(_ctx, " }"); };
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Data := '");
	for (size_t i3 = 0; i3 < p->Data.d; i3++) {
		adbgPrintLog(_ctx, "%02X", p->Data.v[i3]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__SDAP_UL_PduHeader_Type(acpCtx_t _ctx, const struct SDAP_UL_PduHeader_Type* p)
{
	adbgPrintLog(_ctx, "DC := '");
	for (size_t i3 = 0; i3 < 1; i3++) {
		adbgPrintLog(_ctx, "%02X", p->DC[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "R := '");
	for (size_t i3 = 0; i3 < 1; i3++) {
		adbgPrintLog(_ctx, "%02X", p->R[i3]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "QFI := '");
	for (size_t i3 = 0; i3 < 6; i3++) {
		adbgPrintLog(_ctx, "%02X", p->QFI[i3]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__SDAP_UL_PduHeader_Type_SDAP_PDU_UL_Type_Header_Optional(acpCtx_t _ctx, const struct SDAP_UL_PduHeader_Type_SDAP_PDU_UL_Type_Header_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	_adbgNrDrb__SDAP_UL_PduHeader_Type(_ctx, &p->v);
}

static void _adbgNrDrb__SDAP_SDU_Type_SDAP_PDU_UL_Type_Data_Optional(acpCtx_t _ctx, const struct SDAP_SDU_Type_SDAP_PDU_UL_Type_Data_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	adbgPrintLog(_ctx, "'");
	for (size_t i3 = 0; i3 < p->v.d; i3++) {
		adbgPrintLog(_ctx, "%02X", p->v.v[i3]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgNrDrb__SDAP_PDU_UL_Type(acpCtx_t _ctx, const struct SDAP_PDU_UL_Type* p)
{
	adbgPrintLog(_ctx, "Header := ");
	if (p->Header.d) { adbgPrintLog(_ctx, "{ "); };
	_adbgNrDrb__SDAP_UL_PduHeader_Type_SDAP_PDU_UL_Type_Header_Optional(_ctx, &p->Header);
	if (p->Header.d) { adbgPrintLog(_ctx, " }"); };
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Data := ");
	if (p->Data.d) { adbgPrintLog(_ctx, "{ "); };
	_adbgNrDrb__SDAP_SDU_Type_SDAP_PDU_UL_Type_Data_Optional(_ctx, &p->Data);
	if (p->Data.d) { adbgPrintLog(_ctx, " }"); };
}

static void _adbgNrDrb__SDAP_PDU_Type_Value(acpCtx_t _ctx, const union SDAP_PDU_Type_Value* p, enum SDAP_PDU_Type_Sel d)
{
	if (d == SDAP_PDU_Type_DL) {
		adbgPrintLog(_ctx, "DL := { ");
		_adbgNrDrb__SDAP_PDU_DL_Type(_ctx, &p->DL);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == SDAP_PDU_Type_UL) {
		adbgPrintLog(_ctx, "UL := { ");
		_adbgNrDrb__SDAP_PDU_UL_Type(_ctx, &p->UL);
		adbgPrintLog(_ctx, " }");
		return;
	}
	adbgPrintLog(_ctx, "INVALID");
}

static void _adbgNrDrb__SDAP_PDU_Type(acpCtx_t _ctx, const struct SDAP_PDU_Type* p)
{
	_adbgNrDrb__SDAP_PDU_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrDrb__NR_L2DataList_Type_Value(acpCtx_t _ctx, const union NR_L2DataList_Type_Value* p, enum NR_L2DataList_Type_Sel d)
{
	if (d == NR_L2DataList_Type_MacPdu) {
		adbgPrintLog(_ctx, "MacPdu := { ");
		for (size_t i2 = 0; i2 < p->MacPdu.d; i2++) {
			adbgPrintLog(_ctx, "{ ");
			_adbgNrDrb__NR_MAC_PDU_Type(_ctx, &p->MacPdu.v[i2]);
			adbgPrintLog(_ctx, " }");
			if (i2 != p->MacPdu.d - 1) { adbgPrintLog(_ctx, ", "); }
		}
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_L2DataList_Type_RlcPdu) {
		adbgPrintLog(_ctx, "RlcPdu := { ");
		for (size_t i2 = 0; i2 < p->RlcPdu.d; i2++) {
			adbgPrintLog(_ctx, "{ ");
			_adbgNrDrb__NR_RLC_PDU_Type(_ctx, &p->RlcPdu.v[i2]);
			adbgPrintLog(_ctx, " }");
			if (i2 != p->RlcPdu.d - 1) { adbgPrintLog(_ctx, ", "); }
		}
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_L2DataList_Type_RlcSdu) {
		adbgPrintLog(_ctx, "RlcSdu := { ");
		for (size_t i2 = 0; i2 < p->RlcSdu.d; i2++) {
			adbgPrintLog(_ctx, "'");
			for (size_t i3 = 0; i3 < p->RlcSdu.v[i2].d; i3++) {
				adbgPrintLog(_ctx, "%02X", p->RlcSdu.v[i2].v[i3]);
			}
			adbgPrintLog(_ctx, "'O");
			if (i2 != p->RlcSdu.d - 1) { adbgPrintLog(_ctx, ", "); }
		}
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_L2DataList_Type_PdcpPdu) {
		adbgPrintLog(_ctx, "PdcpPdu := { ");
		for (size_t i2 = 0; i2 < p->PdcpPdu.d; i2++) {
			adbgPrintLog(_ctx, "{ ");
			_adbgNrDrb__NR_PDCP_PDU_Type(_ctx, &p->PdcpPdu.v[i2]);
			adbgPrintLog(_ctx, " }");
			if (i2 != p->PdcpPdu.d - 1) { adbgPrintLog(_ctx, ", "); }
		}
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_L2DataList_Type_PdcpSdu) {
		adbgPrintLog(_ctx, "PdcpSdu := { ");
		for (size_t i2 = 0; i2 < p->PdcpSdu.d; i2++) {
			adbgPrintLog(_ctx, "'");
			for (size_t i3 = 0; i3 < p->PdcpSdu.v[i2].d; i3++) {
				adbgPrintLog(_ctx, "%02X", p->PdcpSdu.v[i2].v[i3]);
			}
			adbgPrintLog(_ctx, "'O");
			if (i2 != p->PdcpSdu.d - 1) { adbgPrintLog(_ctx, ", "); }
		}
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_L2DataList_Type_SdapPdu) {
		adbgPrintLog(_ctx, "SdapPdu := { ");
		for (size_t i2 = 0; i2 < p->SdapPdu.d; i2++) {
			adbgPrintLog(_ctx, "{ ");
			_adbgNrDrb__SDAP_PDU_Type(_ctx, &p->SdapPdu.v[i2]);
			adbgPrintLog(_ctx, " }");
			if (i2 != p->SdapPdu.d - 1) { adbgPrintLog(_ctx, ", "); }
		}
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_L2DataList_Type_SdapSdu) {
		adbgPrintLog(_ctx, "SdapSdu := { ");
		for (size_t i2 = 0; i2 < p->SdapSdu.d; i2++) {
			adbgPrintLog(_ctx, "'");
			for (size_t i3 = 0; i3 < p->SdapSdu.v[i2].d; i3++) {
				adbgPrintLog(_ctx, "%02X", p->SdapSdu.v[i2].v[i3]);
			}
			adbgPrintLog(_ctx, "'O");
			if (i2 != p->SdapSdu.d - 1) { adbgPrintLog(_ctx, ", "); }
		}
		adbgPrintLog(_ctx, " }");
		return;
	}
	adbgPrintLog(_ctx, "INVALID");
}

static void _adbgNrDrb__NR_L2DataList_Type(acpCtx_t _ctx, const struct NR_L2DataList_Type* p)
{
	_adbgNrDrb__NR_L2DataList_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrDrb__NR_DRB_DataPerSlot_DL_Type(acpCtx_t _ctx, const struct NR_DRB_DataPerSlot_DL_Type* p)
{
	adbgPrintLog(_ctx, "SlotOffset := %d", (int)p->SlotOffset);
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "HarqProcess := ");
	if (p->HarqProcess.d) { adbgPrintLog(_ctx, "{ "); };
	_adbgNrDrb__NR_HarqProcessAssignment_Type_HarqProcess_Optional(_ctx, &p->HarqProcess);
	if (p->HarqProcess.d) { adbgPrintLog(_ctx, " }"); };
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "PduSduList := { ");
	_adbgNrDrb__NR_L2DataList_Type(_ctx, &p->PduSduList);
	adbgPrintLog(_ctx, " }");
}

static void _adbgNrDrb__NR_L2Data_Request_Type(acpCtx_t _ctx, const struct NR_L2Data_Request_Type* p)
{
	adbgPrintLog(_ctx, "SlotDataList := { ");
	for (size_t i1 = 0; i1 < p->SlotDataList.d; i1++) {
		adbgPrintLog(_ctx, "{ ");
		_adbgNrDrb__NR_DRB_DataPerSlot_DL_Type(_ctx, &p->SlotDataList.v[i1]);
		adbgPrintLog(_ctx, " }");
		if (i1 != p->SlotDataList.d - 1) { adbgPrintLog(_ctx, ", "); }
	}
	adbgPrintLog(_ctx, " }");
}

static void _adbgNrDrb__Null_Type_NR_DRB_COMMON_REQ_SuppressPdcchForC_RNTI_Optional(acpCtx_t _ctx, const struct Null_Type_NR_DRB_COMMON_REQ_SuppressPdcchForC_RNTI_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	adbgPrintLog(_ctx, "%s", (p->v ? "true" : "false"));
}

static void _adbgNrDrb__NR_DRB_COMMON_REQ(acpCtx_t _ctx, const struct NR_DRB_COMMON_REQ* p)
{
	adbgPrintLog(_ctx, "Common := { ");
	_adbgNrDrb__NR_ReqAspCommonPart_Type(_ctx, &p->Common);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "U_Plane := { ");
	_adbgNrDrb__NR_L2Data_Request_Type(_ctx, &p->U_Plane);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "SuppressPdcchForC_RNTI := ");
	_adbgNrDrb__Null_Type_NR_DRB_COMMON_REQ_SuppressPdcchForC_RNTI_Optional(_ctx, &p->SuppressPdcchForC_RNTI);
}

void adbgNrDrbProcessFromSSLogIn(acpCtx_t _ctx, const struct NR_DRB_COMMON_REQ* FromSS)
{
	adbgPrintLog(_ctx, "@NrDrbProcessFromSS In Args : { ");

	adbgPrintLog(_ctx, "FromSS := { ");
	_adbgNrDrb__NR_DRB_COMMON_REQ(_ctx, FromSS);
	adbgPrintLog(_ctx, " }");

	adbgPrintLog(_ctx, " }");
	adbgPrintFormatLog(_ctx);
}

static void _adbgNrDrb__NR_RoutingInfoSUL_Type_RoutingInfoSUL_Optional(acpCtx_t _ctx, const struct NR_RoutingInfoSUL_Type_RoutingInfoSUL_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	adbgPrintLog(_ctx, "%s", (p->v ? "true" : "false"));
}

static void _adbgNrDrb__MacBearerRouting_Type_NR_IndAspCommonPart_Type_MacBearerRouting_Optional(acpCtx_t _ctx, const struct MacBearerRouting_Type_NR_IndAspCommonPart_Type_MacBearerRouting_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	_adbgNrDrb__MacBearerRouting_Type(_ctx, &p->v);
}

static void _adbgNrDrb__IntegrityErrorIndication_Type(acpCtx_t _ctx, const struct IntegrityErrorIndication_Type* p)
{
	adbgPrintLog(_ctx, "Nas := %s", (p->Nas ? "true" : "false"));
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Pdcp := %s", (p->Pdcp ? "true" : "false"));
}

static void _adbgNrDrb__ErrorIndication_Type(acpCtx_t _ctx, const struct ErrorIndication_Type* p)
{
	adbgPrintLog(_ctx, "Integrity := { ");
	_adbgNrDrb__IntegrityErrorIndication_Type(_ctx, &p->Integrity);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "System := %d", (int)p->System);
}

static void _adbgNrDrb__IndicationStatus_Type_Value(acpCtx_t _ctx, const union IndicationStatus_Type_Value* p, enum IndicationStatus_Type_Sel d)
{
	if (d == IndicationStatus_Type_Ok) {
		adbgPrintLog(_ctx, "Ok := %s", (p->Ok ? "true" : "false"));
		return;
	}
	if (d == IndicationStatus_Type_Error) {
		adbgPrintLog(_ctx, "Error := { ");
		_adbgNrDrb__ErrorIndication_Type(_ctx, &p->Error);
		adbgPrintLog(_ctx, " }");
		return;
	}
	adbgPrintLog(_ctx, "INVALID");
}

static void _adbgNrDrb__IndicationStatus_Type(acpCtx_t _ctx, const struct IndicationStatus_Type* p)
{
	_adbgNrDrb__IndicationStatus_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrDrb__NR_IndAspCommonPart_Type(acpCtx_t _ctx, const struct NR_IndAspCommonPart_Type* p)
{
	adbgPrintLog(_ctx, "CellId := %s (%d)", adbgNrDrb__NR_CellId_Type__ToString(p->CellId), (int)p->CellId);
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "RoutingInfo := { ");
	_adbgNrDrb__NR_RoutingInfo_Type(_ctx, &p->RoutingInfo);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "RoutingInfoSUL := ");
	_adbgNrDrb__NR_RoutingInfoSUL_Type_RoutingInfoSUL_Optional(_ctx, &p->RoutingInfoSUL);
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "RlcBearerRouting := { ");
	_adbgNrDrb__RlcBearerRouting_Type(_ctx, &p->RlcBearerRouting);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "MacBearerRouting := ");
	if (p->MacBearerRouting.d) { adbgPrintLog(_ctx, "{ "); };
	_adbgNrDrb__MacBearerRouting_Type_NR_IndAspCommonPart_Type_MacBearerRouting_Optional(_ctx, &p->MacBearerRouting);
	if (p->MacBearerRouting.d) { adbgPrintLog(_ctx, " }"); };
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "TimingInfo := { ");
	_adbgNrDrb__TimingInfo_Type(_ctx, &p->TimingInfo);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Status := { ");
	_adbgNrDrb__IndicationStatus_Type(_ctx, &p->Status);
	adbgPrintLog(_ctx, " }");
}

static void _adbgNrDrb__NR_DRB_DataPerSlot_UL_Type(acpCtx_t _ctx, const struct NR_DRB_DataPerSlot_UL_Type* p)
{
	adbgPrintLog(_ctx, "PduSduList := { ");
	_adbgNrDrb__NR_L2DataList_Type(_ctx, &p->PduSduList);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "NoOfTTIs := %d", (int)p->NoOfTTIs);
}

static void _adbgNrDrb__NR_L2Data_Indication_Type(acpCtx_t _ctx, const struct NR_L2Data_Indication_Type* p)
{
	adbgPrintLog(_ctx, "SlotData := { ");
	_adbgNrDrb__NR_DRB_DataPerSlot_UL_Type(_ctx, &p->SlotData);
	adbgPrintLog(_ctx, " }");
}

static void _adbgNrDrb__NR_DRB_COMMON_IND(acpCtx_t _ctx, const struct NR_DRB_COMMON_IND* p)
{
	adbgPrintLog(_ctx, "Common := { ");
	_adbgNrDrb__NR_IndAspCommonPart_Type(_ctx, &p->Common);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "U_Plane := { ");
	_adbgNrDrb__NR_L2Data_Indication_Type(_ctx, &p->U_Plane);
	adbgPrintLog(_ctx, " }");
}

void adbgNrDrbProcessToSSLogOut(acpCtx_t _ctx, const struct NR_DRB_COMMON_IND* ToSS)
{
	adbgPrintLog(_ctx, "@NrDrbProcessToSS Out Args : { ");

	adbgPrintLog(_ctx, "ToSS := { ");
	_adbgNrDrb__NR_DRB_COMMON_IND(_ctx, ToSS);
	adbgPrintLog(_ctx, " }");

	adbgPrintLog(_ctx, " }");
	adbgPrintFormatLog(_ctx);
}
