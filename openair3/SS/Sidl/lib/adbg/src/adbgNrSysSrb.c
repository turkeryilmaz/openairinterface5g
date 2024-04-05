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

#include "adbgNrSysSrb.h"

static const char* adbgNrSysSrb__NR_CellId_Type__ToString(NR_CellId_Type v)
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

static void _adbgNrSysSrb__NR_RadioBearerId_Type_Value(acpCtx_t _ctx, const union NR_RadioBearerId_Type_Value* p, enum NR_RadioBearerId_Type_Sel d)
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

static void _adbgNrSysSrb__NR_RadioBearerId_Type(acpCtx_t _ctx, const struct NR_RadioBearerId_Type* p)
{
	_adbgNrSysSrb__NR_RadioBearerId_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrSysSrb__QosFlow_Identification_Type(acpCtx_t _ctx, const struct QosFlow_Identification_Type* p)
{
	adbgPrintLog(_ctx, "PDU_SessionId := %d", (int)p->PDU_SessionId);
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "QFI := %d", (int)p->QFI);
}

static void _adbgNrSysSrb__NR_RoutingInfo_Type_Value(acpCtx_t _ctx, const union NR_RoutingInfo_Type_Value* p, enum NR_RoutingInfo_Type_Sel d)
{
	if (d == NR_RoutingInfo_Type_None) {
		adbgPrintLog(_ctx, "None := %s", (p->None ? "true" : "false"));
		return;
	}
	if (d == NR_RoutingInfo_Type_RadioBearerId) {
		adbgPrintLog(_ctx, "RadioBearerId := { ");
		_adbgNrSysSrb__NR_RadioBearerId_Type(_ctx, &p->RadioBearerId);
		adbgPrintLog(_ctx, " }");
		return;
	}
	if (d == NR_RoutingInfo_Type_QosFlow) {
		adbgPrintLog(_ctx, "QosFlow := { ");
		_adbgNrSysSrb__QosFlow_Identification_Type(_ctx, &p->QosFlow);
		adbgPrintLog(_ctx, " }");
		return;
	}
	adbgPrintLog(_ctx, "INVALID");
}

static void _adbgNrSysSrb__NR_RoutingInfo_Type(acpCtx_t _ctx, const struct NR_RoutingInfo_Type* p)
{
	_adbgNrSysSrb__NR_RoutingInfo_Type_Value(_ctx, &p->v, p->d);
}

static const char* adbgNrSysSrb__EUTRA_CellId_Type__ToString(EUTRA_CellId_Type v)
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

static void _adbgNrSysSrb__RlcBearerRouting_Type_Value(acpCtx_t _ctx, const union RlcBearerRouting_Type_Value* p, enum RlcBearerRouting_Type_Sel d)
{
	if (d == RlcBearerRouting_Type_EUTRA) {
		adbgPrintLog(_ctx, "EUTRA := %s (%d)", adbgNrSysSrb__EUTRA_CellId_Type__ToString(p->EUTRA), (int)p->EUTRA);
		return;
	}
	if (d == RlcBearerRouting_Type_NR) {
		adbgPrintLog(_ctx, "NR := %s (%d)", adbgNrSysSrb__NR_CellId_Type__ToString(p->NR), (int)p->NR);
		return;
	}
	if (d == RlcBearerRouting_Type_None) {
		adbgPrintLog(_ctx, "None := %s", (p->None ? "true" : "false"));
		return;
	}
	adbgPrintLog(_ctx, "INVALID");
}

static void _adbgNrSysSrb__RlcBearerRouting_Type(acpCtx_t _ctx, const struct RlcBearerRouting_Type* p)
{
	_adbgNrSysSrb__RlcBearerRouting_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrSysSrb__MacBearerRouting_Type(acpCtx_t _ctx, const struct MacBearerRouting_Type* p)
{
	adbgPrintLog(_ctx, "NR := %s (%d)", adbgNrSysSrb__NR_CellId_Type__ToString(p->NR), (int)p->NR);
}

static void _adbgNrSysSrb__MacBearerRouting_Type_NR_ReqAspCommonPart_Type_MacBearerRouting_Optional(acpCtx_t _ctx, const struct MacBearerRouting_Type_NR_ReqAspCommonPart_Type_MacBearerRouting_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	_adbgNrSysSrb__MacBearerRouting_Type(_ctx, &p->v);
}

static void _adbgNrSysSrb__SystemFrameNumberInfo_Type_Value(acpCtx_t _ctx, const union SystemFrameNumberInfo_Type_Value* p, enum SystemFrameNumberInfo_Type_Sel d)
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

static void _adbgNrSysSrb__SystemFrameNumberInfo_Type(acpCtx_t _ctx, const struct SystemFrameNumberInfo_Type* p)
{
	_adbgNrSysSrb__SystemFrameNumberInfo_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrSysSrb__SubFrameInfo_Type_Value(acpCtx_t _ctx, const union SubFrameInfo_Type_Value* p, enum SubFrameInfo_Type_Sel d)
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

static void _adbgNrSysSrb__SubFrameInfo_Type(acpCtx_t _ctx, const struct SubFrameInfo_Type* p)
{
	_adbgNrSysSrb__SubFrameInfo_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrSysSrb__HyperSystemFrameNumberInfo_Type(acpCtx_t _ctx, const HyperSystemFrameNumberInfo_Type* p)
{
	_adbgNrSysSrb__SystemFrameNumberInfo_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrSysSrb__SlotOffset_Type_Value(acpCtx_t _ctx, const union SlotOffset_Type_Value* p, enum SlotOffset_Type_Sel d)
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

static void _adbgNrSysSrb__SlotOffset_Type(acpCtx_t _ctx, const struct SlotOffset_Type* p)
{
	_adbgNrSysSrb__SlotOffset_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrSysSrb__SlotTimingInfo_Type_Value(acpCtx_t _ctx, const union SlotTimingInfo_Type_Value* p, enum SlotTimingInfo_Type_Sel d)
{
	if (d == SlotTimingInfo_Type_SlotOffset) {
		adbgPrintLog(_ctx, "SlotOffset := { ");
		_adbgNrSysSrb__SlotOffset_Type(_ctx, &p->SlotOffset);
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

static void _adbgNrSysSrb__SlotTimingInfo_Type(acpCtx_t _ctx, const struct SlotTimingInfo_Type* p)
{
	_adbgNrSysSrb__SlotTimingInfo_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrSysSrb__SymbolTimingInfo_Type_Value(acpCtx_t _ctx, const union SymbolTimingInfo_Type_Value* p, enum SymbolTimingInfo_Type_Sel d)
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

static void _adbgNrSysSrb__SymbolTimingInfo_Type(acpCtx_t _ctx, const struct SymbolTimingInfo_Type* p)
{
	_adbgNrSysSrb__SymbolTimingInfo_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrSysSrb__SubFrameTiming_Type(acpCtx_t _ctx, const struct SubFrameTiming_Type* p)
{
	adbgPrintLog(_ctx, "SFN := { ");
	_adbgNrSysSrb__SystemFrameNumberInfo_Type(_ctx, &p->SFN);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Subframe := { ");
	_adbgNrSysSrb__SubFrameInfo_Type(_ctx, &p->Subframe);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "HSFN := { ");
	_adbgNrSysSrb__HyperSystemFrameNumberInfo_Type(_ctx, &p->HSFN);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Slot := { ");
	_adbgNrSysSrb__SlotTimingInfo_Type(_ctx, &p->Slot);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Symbol := { ");
	_adbgNrSysSrb__SymbolTimingInfo_Type(_ctx, &p->Symbol);
	adbgPrintLog(_ctx, " }");
}

static void _adbgNrSysSrb__TimingInfo_Type_Value(acpCtx_t _ctx, const union TimingInfo_Type_Value* p, enum TimingInfo_Type_Sel d)
{
	if (d == TimingInfo_Type_SubFrame) {
		adbgPrintLog(_ctx, "SubFrame := { ");
		_adbgNrSysSrb__SubFrameTiming_Type(_ctx, &p->SubFrame);
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

static void _adbgNrSysSrb__TimingInfo_Type(acpCtx_t _ctx, const struct TimingInfo_Type* p)
{
	_adbgNrSysSrb__TimingInfo_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrSysSrb__ReqAspControlInfo_Type(acpCtx_t _ctx, const struct ReqAspControlInfo_Type* p)
{
	adbgPrintLog(_ctx, "CnfFlag := %s", (p->CnfFlag ? "true" : "false"));
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "FollowOnFlag := %s", (p->FollowOnFlag ? "true" : "false"));
}

static void _adbgNrSysSrb__NR_ReqAspCommonPart_Type(acpCtx_t _ctx, const struct NR_ReqAspCommonPart_Type* p)
{
	adbgPrintLog(_ctx, "CellId := %s (%d)", adbgNrSysSrb__NR_CellId_Type__ToString(p->CellId), (int)p->CellId);
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "RoutingInfo := { ");
	_adbgNrSysSrb__NR_RoutingInfo_Type(_ctx, &p->RoutingInfo);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "RlcBearerRouting := { ");
	_adbgNrSysSrb__RlcBearerRouting_Type(_ctx, &p->RlcBearerRouting);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "MacBearerRouting := ");
	if (p->MacBearerRouting.d) { adbgPrintLog(_ctx, "{ "); };
	_adbgNrSysSrb__MacBearerRouting_Type_NR_ReqAspCommonPart_Type_MacBearerRouting_Optional(_ctx, &p->MacBearerRouting);
	if (p->MacBearerRouting.d) { adbgPrintLog(_ctx, " }"); };
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "TimingInfo := { ");
	_adbgNrSysSrb__TimingInfo_Type(_ctx, &p->TimingInfo);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "ControlInfo := { ");
	_adbgNrSysSrb__ReqAspControlInfo_Type(_ctx, &p->ControlInfo);
	adbgPrintLog(_ctx, " }");
}

static void _adbgNrSysSrb__NR_RRC_MSG_Request_Type_Value(acpCtx_t _ctx, const union NR_RRC_MSG_Request_Type_Value* p, enum NR_RRC_MSG_Request_Type_Sel d)
{
	if (d == NR_RRC_MSG_Request_Type_Ccch) {
		adbgPrintLog(_ctx, "Ccch := '");
		for (size_t i1 = 0; i1 < p->Ccch.d; i1++) {
			adbgPrintLog(_ctx, "%02X", p->Ccch.v[i1]);
		}
		adbgPrintLog(_ctx, "'O");
		return;
	}
	if (d == NR_RRC_MSG_Request_Type_Dcch) {
		adbgPrintLog(_ctx, "Dcch := '");
		for (size_t i1 = 0; i1 < p->Dcch.d; i1++) {
			adbgPrintLog(_ctx, "%02X", p->Dcch.v[i1]);
		}
		adbgPrintLog(_ctx, "'O");
		return;
	}
	adbgPrintLog(_ctx, "INVALID");
}

static void _adbgNrSysSrb__NR_RRC_MSG_Request_Type(acpCtx_t _ctx, const struct NR_RRC_MSG_Request_Type* p)
{
	_adbgNrSysSrb__NR_RRC_MSG_Request_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrSysSrb__NR_RRC_PDU_REQ(acpCtx_t _ctx, const struct NR_RRC_PDU_REQ* p)
{
	adbgPrintLog(_ctx, "Common := { ");
	_adbgNrSysSrb__NR_ReqAspCommonPart_Type(_ctx, &p->Common);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "RrcPdu := { ");
	_adbgNrSysSrb__NR_RRC_MSG_Request_Type(_ctx, &p->RrcPdu);
	adbgPrintLog(_ctx, " }");
}

void adbgNrSysSrbProcessFromSSLogIn(acpCtx_t _ctx, const struct NR_RRC_PDU_REQ* FromSS)
{
	adbgPrintLog(_ctx, "@NrSysSrbProcessFromSS In Args : { ");

	adbgPrintLog(_ctx, "FromSS := { ");
	_adbgNrSysSrb__NR_RRC_PDU_REQ(_ctx, FromSS);
	adbgPrintLog(_ctx, " }");

	adbgPrintLog(_ctx, " }");
	adbgPrintFormatLog(_ctx);
}

static void _adbgNrSysSrb__NR_RoutingInfoSUL_Type_RoutingInfoSUL_Optional(acpCtx_t _ctx, const struct NR_RoutingInfoSUL_Type_RoutingInfoSUL_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	adbgPrintLog(_ctx, "%s", (p->v ? "true" : "false"));
}

static void _adbgNrSysSrb__MacBearerRouting_Type_NR_IndAspCommonPart_Type_MacBearerRouting_Optional(acpCtx_t _ctx, const struct MacBearerRouting_Type_NR_IndAspCommonPart_Type_MacBearerRouting_Optional* p)
{
	if (!p->d) { adbgPrintLog(_ctx, "omit"); return; }
	_adbgNrSysSrb__MacBearerRouting_Type(_ctx, &p->v);
}

static void _adbgNrSysSrb__IntegrityErrorIndication_Type(acpCtx_t _ctx, const struct IntegrityErrorIndication_Type* p)
{
	adbgPrintLog(_ctx, "Nas := %s", (p->Nas ? "true" : "false"));
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Pdcp := %s", (p->Pdcp ? "true" : "false"));
}

static void _adbgNrSysSrb__ErrorIndication_Type(acpCtx_t _ctx, const struct ErrorIndication_Type* p)
{
	adbgPrintLog(_ctx, "Integrity := { ");
	_adbgNrSysSrb__IntegrityErrorIndication_Type(_ctx, &p->Integrity);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "System := %d", (int)p->System);
}

static void _adbgNrSysSrb__IndicationStatus_Type_Value(acpCtx_t _ctx, const union IndicationStatus_Type_Value* p, enum IndicationStatus_Type_Sel d)
{
	if (d == IndicationStatus_Type_Ok) {
		adbgPrintLog(_ctx, "Ok := %s", (p->Ok ? "true" : "false"));
		return;
	}
	if (d == IndicationStatus_Type_Error) {
		adbgPrintLog(_ctx, "Error := { ");
		_adbgNrSysSrb__ErrorIndication_Type(_ctx, &p->Error);
		adbgPrintLog(_ctx, " }");
		return;
	}
	adbgPrintLog(_ctx, "INVALID");
}

static void _adbgNrSysSrb__IndicationStatus_Type(acpCtx_t _ctx, const struct IndicationStatus_Type* p)
{
	_adbgNrSysSrb__IndicationStatus_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrSysSrb__NR_IndAspCommonPart_Type(acpCtx_t _ctx, const struct NR_IndAspCommonPart_Type* p)
{
	adbgPrintLog(_ctx, "CellId := %s (%d)", adbgNrSysSrb__NR_CellId_Type__ToString(p->CellId), (int)p->CellId);
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "RoutingInfo := { ");
	_adbgNrSysSrb__NR_RoutingInfo_Type(_ctx, &p->RoutingInfo);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "RoutingInfoSUL := ");
	_adbgNrSysSrb__NR_RoutingInfoSUL_Type_RoutingInfoSUL_Optional(_ctx, &p->RoutingInfoSUL);
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "RlcBearerRouting := { ");
	_adbgNrSysSrb__RlcBearerRouting_Type(_ctx, &p->RlcBearerRouting);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "MacBearerRouting := ");
	if (p->MacBearerRouting.d) { adbgPrintLog(_ctx, "{ "); };
	_adbgNrSysSrb__MacBearerRouting_Type_NR_IndAspCommonPart_Type_MacBearerRouting_Optional(_ctx, &p->MacBearerRouting);
	if (p->MacBearerRouting.d) { adbgPrintLog(_ctx, " }"); };
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "TimingInfo := { ");
	_adbgNrSysSrb__TimingInfo_Type(_ctx, &p->TimingInfo);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "Status := { ");
	_adbgNrSysSrb__IndicationStatus_Type(_ctx, &p->Status);
	adbgPrintLog(_ctx, " }");
}

static void _adbgNrSysSrb__NR_RRC_MSG_Indication_Type_Value(acpCtx_t _ctx, const union NR_RRC_MSG_Indication_Type_Value* p, enum NR_RRC_MSG_Indication_Type_Sel d)
{
	if (d == NR_RRC_MSG_Indication_Type_Ccch) {
		adbgPrintLog(_ctx, "Ccch := '");
		for (size_t i1 = 0; i1 < p->Ccch.d; i1++) {
			adbgPrintLog(_ctx, "%02X", p->Ccch.v[i1]);
		}
		adbgPrintLog(_ctx, "'O");
		return;
	}
	if (d == NR_RRC_MSG_Indication_Type_Ccch1) {
		adbgPrintLog(_ctx, "Ccch1 := '");
		for (size_t i1 = 0; i1 < p->Ccch1.d; i1++) {
			adbgPrintLog(_ctx, "%02X", p->Ccch1.v[i1]);
		}
		adbgPrintLog(_ctx, "'O");
		return;
	}
	if (d == NR_RRC_MSG_Indication_Type_Dcch) {
		adbgPrintLog(_ctx, "Dcch := '");
		for (size_t i1 = 0; i1 < p->Dcch.d; i1++) {
			adbgPrintLog(_ctx, "%02X", p->Dcch.v[i1]);
		}
		adbgPrintLog(_ctx, "'O");
		return;
	}
	adbgPrintLog(_ctx, "INVALID");
}

static void _adbgNrSysSrb__NR_RRC_MSG_Indication_Type(acpCtx_t _ctx, const struct NR_RRC_MSG_Indication_Type* p)
{
	_adbgNrSysSrb__NR_RRC_MSG_Indication_Type_Value(_ctx, &p->v, p->d);
}

static void _adbgNrSysSrb__NR_RRC_PDU_IND(acpCtx_t _ctx, const struct NR_RRC_PDU_IND* p)
{
	adbgPrintLog(_ctx, "Common := { ");
	_adbgNrSysSrb__NR_IndAspCommonPart_Type(_ctx, &p->Common);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "RrcPdu := { ");
	_adbgNrSysSrb__NR_RRC_MSG_Indication_Type(_ctx, &p->RrcPdu);
	adbgPrintLog(_ctx, " }");
}

void adbgNrSysSrbProcessToSSLogOut(acpCtx_t _ctx, const struct NR_RRC_PDU_IND* ToSS)
{
	adbgPrintLog(_ctx, "@NrSysSrbProcessToSS Out Args : { ");

	adbgPrintLog(_ctx, "ToSS := { ");
	_adbgNrSysSrb__NR_RRC_PDU_IND(_ctx, ToSS);
	adbgPrintLog(_ctx, " }");

	adbgPrintLog(_ctx, " }");
	adbgPrintFormatLog(_ctx);
}
