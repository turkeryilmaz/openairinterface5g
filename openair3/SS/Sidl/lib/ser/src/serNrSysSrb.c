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

#include <string.h>
#include "serNrSysSrb.h"
#include "serMem.h"
#include "serUtils.h"

void serNrSysSrbProcessFromSSInitClt(unsigned char* _arena, size_t _aSize, struct NR_RRC_PDU_REQ** FromSS)
{
	serMem_t _mem = serMemInit(_arena, _aSize);

	*FromSS = (struct NR_RRC_PDU_REQ*)serMalloc(_mem, sizeof(struct NR_RRC_PDU_REQ));
	memset(*FromSS, 0, sizeof(struct NR_RRC_PDU_REQ));
}

static int _serNrSysSrbEncNR_RadioBearerId_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union NR_RadioBearerId_Type_Value* p, enum NR_RadioBearerId_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_RadioBearerId_Type_Srb) {
		HTON_8(&_buffer[*_lidx], p->Srb, _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == NR_RadioBearerId_Type_Drb) {
		HTON_8(&_buffer[*_lidx], p->Drb, _lidx);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrSysSrbEncNR_RadioBearerId_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RadioBearerId_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrSysSrbEncNR_RadioBearerId_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbEncQosFlow_Identification_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct QosFlow_Identification_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_32(&_buffer[*_lidx], p->PDU_SessionId, _lidx);
	HTON_32(&_buffer[*_lidx], p->QFI, _lidx);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbEncNR_RoutingInfo_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union NR_RoutingInfo_Type_Value* p, enum NR_RoutingInfo_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_RoutingInfo_Type_None) {
		HTON_8(&_buffer[*_lidx], p->None, _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == NR_RoutingInfo_Type_RadioBearerId) {
		_serNrSysSrbEncNR_RadioBearerId_Type(_buffer, _size, _lidx, &p->RadioBearerId);
		return SIDL_STATUS_OK;
	}
	if (d == NR_RoutingInfo_Type_QosFlow) {
		_serNrSysSrbEncQosFlow_Identification_Type(_buffer, _size, _lidx, &p->QosFlow);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrSysSrbEncNR_RoutingInfo_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RoutingInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrSysSrbEncNR_RoutingInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbEncRlcBearerRouting_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union RlcBearerRouting_Type_Value* p, enum RlcBearerRouting_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == RlcBearerRouting_Type_EUTRA) {
		{
			size_t _tmp = (size_t)p->EUTRA;
			HTON_32(&_buffer[*_lidx], _tmp, _lidx);
		}
		return SIDL_STATUS_OK;
	}
	if (d == RlcBearerRouting_Type_NR) {
		{
			size_t _tmp = (size_t)p->NR;
			HTON_32(&_buffer[*_lidx], _tmp, _lidx);
		}
		return SIDL_STATUS_OK;
	}
	if (d == RlcBearerRouting_Type_None) {
		HTON_8(&_buffer[*_lidx], p->None, _lidx);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrSysSrbEncRlcBearerRouting_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct RlcBearerRouting_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrSysSrbEncRlcBearerRouting_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbEncMacBearerRouting_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct MacBearerRouting_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->NR;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbEncMacBearerRouting_Type_NR_ReqAspCommonPart_Type_MacBearerRouting_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct MacBearerRouting_Type_NR_ReqAspCommonPart_Type_MacBearerRouting_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	_serNrSysSrbEncMacBearerRouting_Type(_buffer, _size, _lidx, &p->v);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbEncSystemFrameNumberInfo_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union SystemFrameNumberInfo_Type_Value* p, enum SystemFrameNumberInfo_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == SystemFrameNumberInfo_Type_Number) {
		HTON_16(&_buffer[*_lidx], p->Number, _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == SystemFrameNumberInfo_Type_Any) {
		HTON_8(&_buffer[*_lidx], p->Any, _lidx);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrSysSrbEncSystemFrameNumberInfo_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct SystemFrameNumberInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrSysSrbEncSystemFrameNumberInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbEncSubFrameInfo_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union SubFrameInfo_Type_Value* p, enum SubFrameInfo_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == SubFrameInfo_Type_Number) {
		HTON_8(&_buffer[*_lidx], p->Number, _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == SubFrameInfo_Type_Any) {
		HTON_8(&_buffer[*_lidx], p->Any, _lidx);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrSysSrbEncSubFrameInfo_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct SubFrameInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrSysSrbEncSubFrameInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbEncHyperSystemFrameNumberInfo_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const HyperSystemFrameNumberInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrSysSrbEncSystemFrameNumberInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbEncSlotOffset_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union SlotOffset_Type_Value* p, enum SlotOffset_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == SlotOffset_Type_Numerology0) {
		HTON_8(&_buffer[*_lidx], p->Numerology0, _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == SlotOffset_Type_Numerology1) {
		HTON_8(&_buffer[*_lidx], p->Numerology1, _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == SlotOffset_Type_Numerology2) {
		HTON_8(&_buffer[*_lidx], p->Numerology2, _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == SlotOffset_Type_Numerology3) {
		HTON_8(&_buffer[*_lidx], p->Numerology3, _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == SlotOffset_Type_Numerology4) {
		HTON_8(&_buffer[*_lidx], p->Numerology4, _lidx);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrSysSrbEncSlotOffset_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct SlotOffset_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrSysSrbEncSlotOffset_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbEncSlotTimingInfo_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union SlotTimingInfo_Type_Value* p, enum SlotTimingInfo_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == SlotTimingInfo_Type_SlotOffset) {
		_serNrSysSrbEncSlotOffset_Type(_buffer, _size, _lidx, &p->SlotOffset);
		return SIDL_STATUS_OK;
	}
	if (d == SlotTimingInfo_Type_FirstSlot) {
		HTON_8(&_buffer[*_lidx], p->FirstSlot, _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == SlotTimingInfo_Type_Any) {
		HTON_8(&_buffer[*_lidx], p->Any, _lidx);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrSysSrbEncSlotTimingInfo_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct SlotTimingInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrSysSrbEncSlotTimingInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbEncSymbolTimingInfo_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union SymbolTimingInfo_Type_Value* p, enum SymbolTimingInfo_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == SymbolTimingInfo_Type_SymbolOffset) {
		HTON_8(&_buffer[*_lidx], p->SymbolOffset, _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == SymbolTimingInfo_Type_FirstSymbol) {
		HTON_8(&_buffer[*_lidx], p->FirstSymbol, _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == SymbolTimingInfo_Type_Any) {
		HTON_8(&_buffer[*_lidx], p->Any, _lidx);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrSysSrbEncSymbolTimingInfo_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct SymbolTimingInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrSysSrbEncSymbolTimingInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbEncSubFrameTiming_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct SubFrameTiming_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrSysSrbEncSystemFrameNumberInfo_Type(_buffer, _size, _lidx, &p->SFN);
	_serNrSysSrbEncSubFrameInfo_Type(_buffer, _size, _lidx, &p->Subframe);
	_serNrSysSrbEncHyperSystemFrameNumberInfo_Type(_buffer, _size, _lidx, &p->HSFN);
	_serNrSysSrbEncSlotTimingInfo_Type(_buffer, _size, _lidx, &p->Slot);
	_serNrSysSrbEncSymbolTimingInfo_Type(_buffer, _size, _lidx, &p->Symbol);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbEncTimingInfo_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union TimingInfo_Type_Value* p, enum TimingInfo_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == TimingInfo_Type_SubFrame) {
		_serNrSysSrbEncSubFrameTiming_Type(_buffer, _size, _lidx, &p->SubFrame);
		return SIDL_STATUS_OK;
	}
	if (d == TimingInfo_Type_Now) {
		HTON_8(&_buffer[*_lidx], p->Now, _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == TimingInfo_Type_None) {
		HTON_8(&_buffer[*_lidx], p->None, _lidx);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrSysSrbEncTimingInfo_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct TimingInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrSysSrbEncTimingInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbEncReqAspControlInfo_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct ReqAspControlInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->CnfFlag, _lidx);
	HTON_8(&_buffer[*_lidx], p->FollowOnFlag, _lidx);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbEncNR_ReqAspCommonPart_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_ReqAspCommonPart_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->CellId;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrSysSrbEncNR_RoutingInfo_Type(_buffer, _size, _lidx, &p->RoutingInfo);
	_serNrSysSrbEncRlcBearerRouting_Type(_buffer, _size, _lidx, &p->RlcBearerRouting);
	_serNrSysSrbEncMacBearerRouting_Type_NR_ReqAspCommonPart_Type_MacBearerRouting_Optional(_buffer, _size, _lidx, &p->MacBearerRouting);
	_serNrSysSrbEncTimingInfo_Type(_buffer, _size, _lidx, &p->TimingInfo);
	_serNrSysSrbEncReqAspControlInfo_Type(_buffer, _size, _lidx, &p->ControlInfo);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbEncNR_RRC_MSG_Request_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union NR_RRC_MSG_Request_Type_Value* p, enum NR_RRC_MSG_Request_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_RRC_MSG_Request_Type_Ccch) {
		HTON_32(&_buffer[*_lidx], p->Ccch.d, _lidx);
		for (size_t i1 = 0; i1 < p->Ccch.d; i1++) {
			HTON_8(&_buffer[*_lidx], p->Ccch.v[i1], _lidx);
		}
		return SIDL_STATUS_OK;
	}
	if (d == NR_RRC_MSG_Request_Type_Dcch) {
		HTON_32(&_buffer[*_lidx], p->Dcch.d, _lidx);
		for (size_t i1 = 0; i1 < p->Dcch.d; i1++) {
			HTON_8(&_buffer[*_lidx], p->Dcch.v[i1], _lidx);
		}
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrSysSrbEncNR_RRC_MSG_Request_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RRC_MSG_Request_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrSysSrbEncNR_RRC_MSG_Request_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbEncNR_RRC_PDU_REQ(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RRC_PDU_REQ* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrSysSrbEncNR_ReqAspCommonPart_Type(_buffer, _size, _lidx, &p->Common);
	_serNrSysSrbEncNR_RRC_MSG_Request_Type(_buffer, _size, _lidx, &p->RrcPdu);

	return SIDL_STATUS_OK;
}

int serNrSysSrbProcessFromSSEncClt(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RRC_PDU_REQ* FromSS)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrSysSrbEncNR_RRC_PDU_REQ(_buffer, _size, _lidx, FromSS);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbDecNR_RadioBearerId_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, union NR_RadioBearerId_Type_Value* p, enum NR_RadioBearerId_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_RadioBearerId_Type_Srb) {
		NTOH_8(p->Srb, &_buffer[*_lidx], _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == NR_RadioBearerId_Type_Drb) {
		NTOH_8(p->Drb, &_buffer[*_lidx], _lidx);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrSysSrbDecNR_RadioBearerId_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_RadioBearerId_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum NR_RadioBearerId_Type_Sel)_tmp;
	}
	_serNrSysSrbDecNR_RadioBearerId_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbDecQosFlow_Identification_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct QosFlow_Identification_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_32(p->PDU_SessionId, &_buffer[*_lidx], _lidx);
	NTOH_32(p->QFI, &_buffer[*_lidx], _lidx);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbDecNR_RoutingInfo_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, union NR_RoutingInfo_Type_Value* p, enum NR_RoutingInfo_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_RoutingInfo_Type_None) {
		NTOH_8(p->None, &_buffer[*_lidx], _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == NR_RoutingInfo_Type_RadioBearerId) {
		_serNrSysSrbDecNR_RadioBearerId_Type(_buffer, _size, _lidx, &p->RadioBearerId);
		return SIDL_STATUS_OK;
	}
	if (d == NR_RoutingInfo_Type_QosFlow) {
		_serNrSysSrbDecQosFlow_Identification_Type(_buffer, _size, _lidx, &p->QosFlow);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrSysSrbDecNR_RoutingInfo_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_RoutingInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum NR_RoutingInfo_Type_Sel)_tmp;
	}
	_serNrSysSrbDecNR_RoutingInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbDecRlcBearerRouting_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, union RlcBearerRouting_Type_Value* p, enum RlcBearerRouting_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == RlcBearerRouting_Type_EUTRA) {
		{
			size_t _tmp;
			NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
			p->EUTRA = (EUTRA_CellId_Type)_tmp;
		}
		return SIDL_STATUS_OK;
	}
	if (d == RlcBearerRouting_Type_NR) {
		{
			size_t _tmp;
			NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
			p->NR = (NR_CellId_Type)_tmp;
		}
		return SIDL_STATUS_OK;
	}
	if (d == RlcBearerRouting_Type_None) {
		NTOH_8(p->None, &_buffer[*_lidx], _lidx);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrSysSrbDecRlcBearerRouting_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct RlcBearerRouting_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum RlcBearerRouting_Type_Sel)_tmp;
	}
	_serNrSysSrbDecRlcBearerRouting_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbDecMacBearerRouting_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct MacBearerRouting_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->NR = (NR_CellId_Type)_tmp;
	}

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbDecMacBearerRouting_Type_NR_ReqAspCommonPart_Type_MacBearerRouting_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct MacBearerRouting_Type_NR_ReqAspCommonPart_Type_MacBearerRouting_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	_serNrSysSrbDecMacBearerRouting_Type(_buffer, _size, _lidx, &p->v);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbDecSystemFrameNumberInfo_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, union SystemFrameNumberInfo_Type_Value* p, enum SystemFrameNumberInfo_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == SystemFrameNumberInfo_Type_Number) {
		NTOH_16(p->Number, &_buffer[*_lidx], _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == SystemFrameNumberInfo_Type_Any) {
		NTOH_8(p->Any, &_buffer[*_lidx], _lidx);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrSysSrbDecSystemFrameNumberInfo_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct SystemFrameNumberInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum SystemFrameNumberInfo_Type_Sel)_tmp;
	}
	_serNrSysSrbDecSystemFrameNumberInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbDecSubFrameInfo_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, union SubFrameInfo_Type_Value* p, enum SubFrameInfo_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == SubFrameInfo_Type_Number) {
		NTOH_8(p->Number, &_buffer[*_lidx], _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == SubFrameInfo_Type_Any) {
		NTOH_8(p->Any, &_buffer[*_lidx], _lidx);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrSysSrbDecSubFrameInfo_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct SubFrameInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum SubFrameInfo_Type_Sel)_tmp;
	}
	_serNrSysSrbDecSubFrameInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbDecHyperSystemFrameNumberInfo_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, HyperSystemFrameNumberInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum SystemFrameNumberInfo_Type_Sel)_tmp;
	}
	_serNrSysSrbDecSystemFrameNumberInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbDecSlotOffset_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, union SlotOffset_Type_Value* p, enum SlotOffset_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == SlotOffset_Type_Numerology0) {
		NTOH_8(p->Numerology0, &_buffer[*_lidx], _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == SlotOffset_Type_Numerology1) {
		NTOH_8(p->Numerology1, &_buffer[*_lidx], _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == SlotOffset_Type_Numerology2) {
		NTOH_8(p->Numerology2, &_buffer[*_lidx], _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == SlotOffset_Type_Numerology3) {
		NTOH_8(p->Numerology3, &_buffer[*_lidx], _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == SlotOffset_Type_Numerology4) {
		NTOH_8(p->Numerology4, &_buffer[*_lidx], _lidx);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrSysSrbDecSlotOffset_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct SlotOffset_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum SlotOffset_Type_Sel)_tmp;
	}
	_serNrSysSrbDecSlotOffset_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbDecSlotTimingInfo_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, union SlotTimingInfo_Type_Value* p, enum SlotTimingInfo_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == SlotTimingInfo_Type_SlotOffset) {
		_serNrSysSrbDecSlotOffset_Type(_buffer, _size, _lidx, &p->SlotOffset);
		return SIDL_STATUS_OK;
	}
	if (d == SlotTimingInfo_Type_FirstSlot) {
		NTOH_8(p->FirstSlot, &_buffer[*_lidx], _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == SlotTimingInfo_Type_Any) {
		NTOH_8(p->Any, &_buffer[*_lidx], _lidx);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrSysSrbDecSlotTimingInfo_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct SlotTimingInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum SlotTimingInfo_Type_Sel)_tmp;
	}
	_serNrSysSrbDecSlotTimingInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbDecSymbolTimingInfo_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, union SymbolTimingInfo_Type_Value* p, enum SymbolTimingInfo_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == SymbolTimingInfo_Type_SymbolOffset) {
		NTOH_8(p->SymbolOffset, &_buffer[*_lidx], _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == SymbolTimingInfo_Type_FirstSymbol) {
		NTOH_8(p->FirstSymbol, &_buffer[*_lidx], _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == SymbolTimingInfo_Type_Any) {
		NTOH_8(p->Any, &_buffer[*_lidx], _lidx);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrSysSrbDecSymbolTimingInfo_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct SymbolTimingInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum SymbolTimingInfo_Type_Sel)_tmp;
	}
	_serNrSysSrbDecSymbolTimingInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbDecSubFrameTiming_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct SubFrameTiming_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrSysSrbDecSystemFrameNumberInfo_Type(_buffer, _size, _lidx, &p->SFN);
	_serNrSysSrbDecSubFrameInfo_Type(_buffer, _size, _lidx, &p->Subframe);
	_serNrSysSrbDecHyperSystemFrameNumberInfo_Type(_buffer, _size, _lidx, &p->HSFN);
	_serNrSysSrbDecSlotTimingInfo_Type(_buffer, _size, _lidx, &p->Slot);
	_serNrSysSrbDecSymbolTimingInfo_Type(_buffer, _size, _lidx, &p->Symbol);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbDecTimingInfo_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, union TimingInfo_Type_Value* p, enum TimingInfo_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == TimingInfo_Type_SubFrame) {
		_serNrSysSrbDecSubFrameTiming_Type(_buffer, _size, _lidx, &p->SubFrame);
		return SIDL_STATUS_OK;
	}
	if (d == TimingInfo_Type_Now) {
		NTOH_8(p->Now, &_buffer[*_lidx], _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == TimingInfo_Type_None) {
		NTOH_8(p->None, &_buffer[*_lidx], _lidx);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrSysSrbDecTimingInfo_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct TimingInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum TimingInfo_Type_Sel)_tmp;
	}
	_serNrSysSrbDecTimingInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbDecReqAspControlInfo_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct ReqAspControlInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->CnfFlag, &_buffer[*_lidx], _lidx);
	NTOH_8(p->FollowOnFlag, &_buffer[*_lidx], _lidx);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbDecNR_ReqAspCommonPart_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_ReqAspCommonPart_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->CellId = (NR_CellId_Type)_tmp;
	}
	_serNrSysSrbDecNR_RoutingInfo_Type(_buffer, _size, _lidx, &p->RoutingInfo);
	_serNrSysSrbDecRlcBearerRouting_Type(_buffer, _size, _lidx, &p->RlcBearerRouting);
	_serNrSysSrbDecMacBearerRouting_Type_NR_ReqAspCommonPart_Type_MacBearerRouting_Optional(_buffer, _size, _lidx, &p->MacBearerRouting);
	_serNrSysSrbDecTimingInfo_Type(_buffer, _size, _lidx, &p->TimingInfo);
	_serNrSysSrbDecReqAspControlInfo_Type(_buffer, _size, _lidx, &p->ControlInfo);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbDecNR_RRC_MSG_Request_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, union NR_RRC_MSG_Request_Type_Value* p, enum NR_RRC_MSG_Request_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_RRC_MSG_Request_Type_Ccch) {
		NTOH_32(p->Ccch.d, &_buffer[*_lidx], _lidx);
		p->Ccch.v = (uint8_t*)serMalloc(_mem, p->Ccch.d * sizeof(uint8_t));
		for (size_t i1 = 0; i1 < p->Ccch.d; i1++) {
			NTOH_8(p->Ccch.v[i1], &_buffer[*_lidx], _lidx);
		}
		return SIDL_STATUS_OK;
	}
	if (d == NR_RRC_MSG_Request_Type_Dcch) {
		NTOH_32(p->Dcch.d, &_buffer[*_lidx], _lidx);
		p->Dcch.v = (uint8_t*)serMalloc(_mem, p->Dcch.d * sizeof(uint8_t));
		for (size_t i1 = 0; i1 < p->Dcch.d; i1++) {
			NTOH_8(p->Dcch.v[i1], &_buffer[*_lidx], _lidx);
		}
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrSysSrbDecNR_RRC_MSG_Request_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_RRC_MSG_Request_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum NR_RRC_MSG_Request_Type_Sel)_tmp;
	}
	_serNrSysSrbDecNR_RRC_MSG_Request_Type_Value(_buffer, _size, _lidx, _mem, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbDecNR_RRC_PDU_REQ(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_RRC_PDU_REQ* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrSysSrbDecNR_ReqAspCommonPart_Type(_buffer, _size, _lidx, &p->Common);
	_serNrSysSrbDecNR_RRC_MSG_Request_Type(_buffer, _size, _lidx, _mem, &p->RrcPdu);

	return SIDL_STATUS_OK;
}

int serNrSysSrbProcessFromSSDecSrv(const unsigned char* _buffer, size_t _size, unsigned char* _arena, size_t _aSize, struct NR_RRC_PDU_REQ** FromSS)
{
	(void)_size; // TODO: generate boundaries checking

	serMem_t _mem = serMemInit(_arena, _aSize);

	size_t __lidx = 0;
	size_t* _lidx = &__lidx;

	*FromSS = (struct NR_RRC_PDU_REQ*)serMalloc(_mem, sizeof(struct NR_RRC_PDU_REQ));
	_serNrSysSrbDecNR_RRC_PDU_REQ(_buffer, _size, _lidx, _mem, *FromSS);

	return SIDL_STATUS_OK;
}

static void _serNrSysSrbFreeNR_RRC_MSG_Request_Type_Value(union NR_RRC_MSG_Request_Type_Value* p, enum NR_RRC_MSG_Request_Type_Sel d)
{
	if (d == NR_RRC_MSG_Request_Type_Ccch) {
		if (p->Ccch.v) {
			serFree(p->Ccch.v);
		}
		return;
	}
	if (d == NR_RRC_MSG_Request_Type_Dcch) {
		if (p->Dcch.v) {
			serFree(p->Dcch.v);
		}
		return;
	}
}

static void _serNrSysSrbFreeNR_RRC_MSG_Request_Type(struct NR_RRC_MSG_Request_Type* p)
{
	_serNrSysSrbFreeNR_RRC_MSG_Request_Type_Value(&p->v, p->d);
}

static void _serNrSysSrbFreeNR_RRC_PDU_REQ(struct NR_RRC_PDU_REQ* p)
{
	_serNrSysSrbFreeNR_RRC_MSG_Request_Type(&p->RrcPdu);
}

void serNrSysSrbProcessFromSSFree0Srv(struct NR_RRC_PDU_REQ* FromSS)
{
	if (FromSS) {
		_serNrSysSrbFreeNR_RRC_PDU_REQ(FromSS);
	}
}

void serNrSysSrbProcessFromSSFreeSrv(struct NR_RRC_PDU_REQ* FromSS)
{
	if (FromSS) {
		_serNrSysSrbFreeNR_RRC_PDU_REQ(FromSS);
		serFree(FromSS);
	}
}

void serNrSysSrbProcessToSSInitSrv(unsigned char* _arena, size_t _aSize, struct NR_RRC_PDU_IND** ToSS)
{
	serMem_t _mem = serMemInit(_arena, _aSize);

	*ToSS = (struct NR_RRC_PDU_IND*)serMalloc(_mem, sizeof(struct NR_RRC_PDU_IND));
	memset(*ToSS, 0, sizeof(struct NR_RRC_PDU_IND));
}

static int _serNrSysSrbEncNR_RoutingInfoSUL_Type_RoutingInfoSUL_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RoutingInfoSUL_Type_RoutingInfoSUL_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	HTON_8(&_buffer[*_lidx], p->v, _lidx);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbEncMacBearerRouting_Type_NR_IndAspCommonPart_Type_MacBearerRouting_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct MacBearerRouting_Type_NR_IndAspCommonPart_Type_MacBearerRouting_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	_serNrSysSrbEncMacBearerRouting_Type(_buffer, _size, _lidx, &p->v);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbEncIntegrityErrorIndication_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct IntegrityErrorIndication_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->Nas, _lidx);
	HTON_8(&_buffer[*_lidx], p->Pdcp, _lidx);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbEncErrorIndication_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct ErrorIndication_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrSysSrbEncIntegrityErrorIndication_Type(_buffer, _size, _lidx, &p->Integrity);
	HTON_32(&_buffer[*_lidx], p->System, _lidx);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbEncIndicationStatus_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union IndicationStatus_Type_Value* p, enum IndicationStatus_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == IndicationStatus_Type_Ok) {
		HTON_8(&_buffer[*_lidx], p->Ok, _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == IndicationStatus_Type_Error) {
		_serNrSysSrbEncErrorIndication_Type(_buffer, _size, _lidx, &p->Error);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrSysSrbEncIndicationStatus_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct IndicationStatus_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrSysSrbEncIndicationStatus_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbEncNR_IndAspCommonPart_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_IndAspCommonPart_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->CellId;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrSysSrbEncNR_RoutingInfo_Type(_buffer, _size, _lidx, &p->RoutingInfo);
	_serNrSysSrbEncNR_RoutingInfoSUL_Type_RoutingInfoSUL_Optional(_buffer, _size, _lidx, &p->RoutingInfoSUL);
	_serNrSysSrbEncRlcBearerRouting_Type(_buffer, _size, _lidx, &p->RlcBearerRouting);
	_serNrSysSrbEncMacBearerRouting_Type_NR_IndAspCommonPart_Type_MacBearerRouting_Optional(_buffer, _size, _lidx, &p->MacBearerRouting);
	_serNrSysSrbEncTimingInfo_Type(_buffer, _size, _lidx, &p->TimingInfo);
	_serNrSysSrbEncIndicationStatus_Type(_buffer, _size, _lidx, &p->Status);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbEncNR_RRC_MSG_Indication_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union NR_RRC_MSG_Indication_Type_Value* p, enum NR_RRC_MSG_Indication_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_RRC_MSG_Indication_Type_Ccch) {
		HTON_32(&_buffer[*_lidx], p->Ccch.d, _lidx);
		for (size_t i1 = 0; i1 < p->Ccch.d; i1++) {
			HTON_8(&_buffer[*_lidx], p->Ccch.v[i1], _lidx);
		}
		return SIDL_STATUS_OK;
	}
	if (d == NR_RRC_MSG_Indication_Type_Ccch1) {
		HTON_32(&_buffer[*_lidx], p->Ccch1.d, _lidx);
		for (size_t i1 = 0; i1 < p->Ccch1.d; i1++) {
			HTON_8(&_buffer[*_lidx], p->Ccch1.v[i1], _lidx);
		}
		return SIDL_STATUS_OK;
	}
	if (d == NR_RRC_MSG_Indication_Type_Dcch) {
		HTON_32(&_buffer[*_lidx], p->Dcch.d, _lidx);
		for (size_t i1 = 0; i1 < p->Dcch.d; i1++) {
			HTON_8(&_buffer[*_lidx], p->Dcch.v[i1], _lidx);
		}
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrSysSrbEncNR_RRC_MSG_Indication_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RRC_MSG_Indication_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrSysSrbEncNR_RRC_MSG_Indication_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbEncNR_RRC_PDU_IND(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RRC_PDU_IND* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrSysSrbEncNR_IndAspCommonPart_Type(_buffer, _size, _lidx, &p->Common);
	_serNrSysSrbEncNR_RRC_MSG_Indication_Type(_buffer, _size, _lidx, &p->RrcPdu);

	return SIDL_STATUS_OK;
}

int serNrSysSrbProcessToSSEncSrv(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RRC_PDU_IND* ToSS)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrSysSrbEncNR_RRC_PDU_IND(_buffer, _size, _lidx, ToSS);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbDecNR_RoutingInfoSUL_Type_RoutingInfoSUL_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_RoutingInfoSUL_Type_RoutingInfoSUL_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	NTOH_8(p->v, &_buffer[*_lidx], _lidx);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbDecMacBearerRouting_Type_NR_IndAspCommonPart_Type_MacBearerRouting_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct MacBearerRouting_Type_NR_IndAspCommonPart_Type_MacBearerRouting_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	_serNrSysSrbDecMacBearerRouting_Type(_buffer, _size, _lidx, &p->v);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbDecIntegrityErrorIndication_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct IntegrityErrorIndication_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->Nas, &_buffer[*_lidx], _lidx);
	NTOH_8(p->Pdcp, &_buffer[*_lidx], _lidx);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbDecErrorIndication_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct ErrorIndication_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrSysSrbDecIntegrityErrorIndication_Type(_buffer, _size, _lidx, &p->Integrity);
	NTOH_32(p->System, &_buffer[*_lidx], _lidx);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbDecIndicationStatus_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, union IndicationStatus_Type_Value* p, enum IndicationStatus_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == IndicationStatus_Type_Ok) {
		NTOH_8(p->Ok, &_buffer[*_lidx], _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == IndicationStatus_Type_Error) {
		_serNrSysSrbDecErrorIndication_Type(_buffer, _size, _lidx, &p->Error);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrSysSrbDecIndicationStatus_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct IndicationStatus_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum IndicationStatus_Type_Sel)_tmp;
	}
	_serNrSysSrbDecIndicationStatus_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbDecNR_IndAspCommonPart_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_IndAspCommonPart_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->CellId = (NR_CellId_Type)_tmp;
	}
	_serNrSysSrbDecNR_RoutingInfo_Type(_buffer, _size, _lidx, &p->RoutingInfo);
	_serNrSysSrbDecNR_RoutingInfoSUL_Type_RoutingInfoSUL_Optional(_buffer, _size, _lidx, &p->RoutingInfoSUL);
	_serNrSysSrbDecRlcBearerRouting_Type(_buffer, _size, _lidx, &p->RlcBearerRouting);
	_serNrSysSrbDecMacBearerRouting_Type_NR_IndAspCommonPart_Type_MacBearerRouting_Optional(_buffer, _size, _lidx, &p->MacBearerRouting);
	_serNrSysSrbDecTimingInfo_Type(_buffer, _size, _lidx, &p->TimingInfo);
	_serNrSysSrbDecIndicationStatus_Type(_buffer, _size, _lidx, &p->Status);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbDecNR_RRC_MSG_Indication_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, union NR_RRC_MSG_Indication_Type_Value* p, enum NR_RRC_MSG_Indication_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_RRC_MSG_Indication_Type_Ccch) {
		NTOH_32(p->Ccch.d, &_buffer[*_lidx], _lidx);
		p->Ccch.v = (uint8_t*)serMalloc(_mem, p->Ccch.d * sizeof(uint8_t));
		for (size_t i1 = 0; i1 < p->Ccch.d; i1++) {
			NTOH_8(p->Ccch.v[i1], &_buffer[*_lidx], _lidx);
		}
		return SIDL_STATUS_OK;
	}
	if (d == NR_RRC_MSG_Indication_Type_Ccch1) {
		NTOH_32(p->Ccch1.d, &_buffer[*_lidx], _lidx);
		p->Ccch1.v = (uint8_t*)serMalloc(_mem, p->Ccch1.d * sizeof(uint8_t));
		for (size_t i1 = 0; i1 < p->Ccch1.d; i1++) {
			NTOH_8(p->Ccch1.v[i1], &_buffer[*_lidx], _lidx);
		}
		return SIDL_STATUS_OK;
	}
	if (d == NR_RRC_MSG_Indication_Type_Dcch) {
		NTOH_32(p->Dcch.d, &_buffer[*_lidx], _lidx);
		p->Dcch.v = (uint8_t*)serMalloc(_mem, p->Dcch.d * sizeof(uint8_t));
		for (size_t i1 = 0; i1 < p->Dcch.d; i1++) {
			NTOH_8(p->Dcch.v[i1], &_buffer[*_lidx], _lidx);
		}
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrSysSrbDecNR_RRC_MSG_Indication_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_RRC_MSG_Indication_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum NR_RRC_MSG_Indication_Type_Sel)_tmp;
	}
	_serNrSysSrbDecNR_RRC_MSG_Indication_Type_Value(_buffer, _size, _lidx, _mem, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrSysSrbDecNR_RRC_PDU_IND(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_RRC_PDU_IND* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrSysSrbDecNR_IndAspCommonPart_Type(_buffer, _size, _lidx, &p->Common);
	_serNrSysSrbDecNR_RRC_MSG_Indication_Type(_buffer, _size, _lidx, _mem, &p->RrcPdu);

	return SIDL_STATUS_OK;
}

int serNrSysSrbProcessToSSDecClt(const unsigned char* _buffer, size_t _size, unsigned char* _arena, size_t _aSize, struct NR_RRC_PDU_IND** ToSS)
{
	(void)_size; // TODO: generate boundaries checking

	serMem_t _mem = serMemInit(_arena, _aSize);

	size_t __lidx = 0;
	size_t* _lidx = &__lidx;

	*ToSS = (struct NR_RRC_PDU_IND*)serMalloc(_mem, sizeof(struct NR_RRC_PDU_IND));
	_serNrSysSrbDecNR_RRC_PDU_IND(_buffer, _size, _lidx, _mem, *ToSS);

	return SIDL_STATUS_OK;
}

static void _serNrSysSrbFreeNR_RRC_MSG_Indication_Type_Value(union NR_RRC_MSG_Indication_Type_Value* p, enum NR_RRC_MSG_Indication_Type_Sel d)
{
	if (d == NR_RRC_MSG_Indication_Type_Ccch) {
		if (p->Ccch.v) {
			serFree(p->Ccch.v);
		}
		return;
	}
	if (d == NR_RRC_MSG_Indication_Type_Ccch1) {
		if (p->Ccch1.v) {
			serFree(p->Ccch1.v);
		}
		return;
	}
	if (d == NR_RRC_MSG_Indication_Type_Dcch) {
		if (p->Dcch.v) {
			serFree(p->Dcch.v);
		}
		return;
	}
}

static void _serNrSysSrbFreeNR_RRC_MSG_Indication_Type(struct NR_RRC_MSG_Indication_Type* p)
{
	_serNrSysSrbFreeNR_RRC_MSG_Indication_Type_Value(&p->v, p->d);
}

static void _serNrSysSrbFreeNR_RRC_PDU_IND(struct NR_RRC_PDU_IND* p)
{
	_serNrSysSrbFreeNR_RRC_MSG_Indication_Type(&p->RrcPdu);
}

void serNrSysSrbProcessToSSFree0Clt(struct NR_RRC_PDU_IND* ToSS)
{
	if (ToSS) {
		_serNrSysSrbFreeNR_RRC_PDU_IND(ToSS);
	}
}

void serNrSysSrbProcessToSSFreeClt(struct NR_RRC_PDU_IND* ToSS)
{
	if (ToSS) {
		_serNrSysSrbFreeNR_RRC_PDU_IND(ToSS);
		serFree(ToSS);
	}
}
