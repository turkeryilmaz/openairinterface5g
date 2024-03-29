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
#include "serNrDrb.h"
#include "serMem.h"
#include "serUtils.h"

void serNrDrbProcessFromSSInitClt(unsigned char* _arena, size_t _aSize, struct NR_DRB_COMMON_REQ** FromSS)
{
	serMem_t _mem = serMemInit(_arena, _aSize);

	*FromSS = (struct NR_DRB_COMMON_REQ*)serMalloc(_mem, sizeof(struct NR_DRB_COMMON_REQ));
	memset(*FromSS, 0, sizeof(struct NR_DRB_COMMON_REQ));
}

static int _serNrDrbEncNR_RadioBearerId_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union NR_RadioBearerId_Type_Value* p, enum NR_RadioBearerId_Type_Sel d)
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

static int _serNrDrbEncNR_RadioBearerId_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RadioBearerId_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrDrbEncNR_RadioBearerId_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncQosFlow_Identification_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct QosFlow_Identification_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_32(&_buffer[*_lidx], p->PDU_SessionId, _lidx);
	HTON_32(&_buffer[*_lidx], p->QFI, _lidx);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RoutingInfo_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union NR_RoutingInfo_Type_Value* p, enum NR_RoutingInfo_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_RoutingInfo_Type_None) {
		HTON_8(&_buffer[*_lidx], p->None, _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == NR_RoutingInfo_Type_RadioBearerId) {
		_serNrDrbEncNR_RadioBearerId_Type(_buffer, _size, _lidx, &p->RadioBearerId);
		return SIDL_STATUS_OK;
	}
	if (d == NR_RoutingInfo_Type_QosFlow) {
		_serNrDrbEncQosFlow_Identification_Type(_buffer, _size, _lidx, &p->QosFlow);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrDrbEncNR_RoutingInfo_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RoutingInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrDrbEncNR_RoutingInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncRlcBearerRouting_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union RlcBearerRouting_Type_Value* p, enum RlcBearerRouting_Type_Sel d)
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

static int _serNrDrbEncRlcBearerRouting_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct RlcBearerRouting_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrDrbEncRlcBearerRouting_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncMacBearerRouting_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct MacBearerRouting_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->NR;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncMacBearerRouting_Type_NR_ReqAspCommonPart_Type_MacBearerRouting_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct MacBearerRouting_Type_NR_ReqAspCommonPart_Type_MacBearerRouting_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	_serNrDrbEncMacBearerRouting_Type(_buffer, _size, _lidx, &p->v);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncSystemFrameNumberInfo_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union SystemFrameNumberInfo_Type_Value* p, enum SystemFrameNumberInfo_Type_Sel d)
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

static int _serNrDrbEncSystemFrameNumberInfo_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct SystemFrameNumberInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrDrbEncSystemFrameNumberInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncSubFrameInfo_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union SubFrameInfo_Type_Value* p, enum SubFrameInfo_Type_Sel d)
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

static int _serNrDrbEncSubFrameInfo_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct SubFrameInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrDrbEncSubFrameInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncHyperSystemFrameNumberInfo_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const HyperSystemFrameNumberInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrDrbEncSystemFrameNumberInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncSlotOffset_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union SlotOffset_Type_Value* p, enum SlotOffset_Type_Sel d)
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

static int _serNrDrbEncSlotOffset_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct SlotOffset_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrDrbEncSlotOffset_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncSlotTimingInfo_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union SlotTimingInfo_Type_Value* p, enum SlotTimingInfo_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == SlotTimingInfo_Type_SlotOffset) {
		_serNrDrbEncSlotOffset_Type(_buffer, _size, _lidx, &p->SlotOffset);
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

static int _serNrDrbEncSlotTimingInfo_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct SlotTimingInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrDrbEncSlotTimingInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncSymbolTimingInfo_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union SymbolTimingInfo_Type_Value* p, enum SymbolTimingInfo_Type_Sel d)
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

static int _serNrDrbEncSymbolTimingInfo_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct SymbolTimingInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrDrbEncSymbolTimingInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncSubFrameTiming_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct SubFrameTiming_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncSystemFrameNumberInfo_Type(_buffer, _size, _lidx, &p->SFN);
	_serNrDrbEncSubFrameInfo_Type(_buffer, _size, _lidx, &p->Subframe);
	_serNrDrbEncHyperSystemFrameNumberInfo_Type(_buffer, _size, _lidx, &p->HSFN);
	_serNrDrbEncSlotTimingInfo_Type(_buffer, _size, _lidx, &p->Slot);
	_serNrDrbEncSymbolTimingInfo_Type(_buffer, _size, _lidx, &p->Symbol);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncTimingInfo_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union TimingInfo_Type_Value* p, enum TimingInfo_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == TimingInfo_Type_SubFrame) {
		_serNrDrbEncSubFrameTiming_Type(_buffer, _size, _lidx, &p->SubFrame);
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

static int _serNrDrbEncTimingInfo_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct TimingInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrDrbEncTimingInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncReqAspControlInfo_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct ReqAspControlInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->CnfFlag, _lidx);
	HTON_8(&_buffer[*_lidx], p->FollowOnFlag, _lidx);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_ReqAspCommonPart_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_ReqAspCommonPart_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->CellId;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrDrbEncNR_RoutingInfo_Type(_buffer, _size, _lidx, &p->RoutingInfo);
	_serNrDrbEncRlcBearerRouting_Type(_buffer, _size, _lidx, &p->RlcBearerRouting);
	_serNrDrbEncMacBearerRouting_Type_NR_ReqAspCommonPart_Type_MacBearerRouting_Optional(_buffer, _size, _lidx, &p->MacBearerRouting);
	_serNrDrbEncTimingInfo_Type(_buffer, _size, _lidx, &p->TimingInfo);
	_serNrDrbEncReqAspControlInfo_Type(_buffer, _size, _lidx, &p->ControlInfo);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_HarqProcessAssignment_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union NR_HarqProcessAssignment_Type_Value* p, enum NR_HarqProcessAssignment_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_HarqProcessAssignment_Type_Id) {
		HTON_32(&_buffer[*_lidx], p->Id, _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == NR_HarqProcessAssignment_Type_Automatic) {
		HTON_8(&_buffer[*_lidx], p->Automatic, _lidx);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrDrbEncNR_HarqProcessAssignment_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_HarqProcessAssignment_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrDrbEncNR_HarqProcessAssignment_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_HarqProcessAssignment_Type_HarqProcess_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_HarqProcessAssignment_Type_HarqProcess_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	_serNrDrbEncNR_HarqProcessAssignment_Type(_buffer, _size, _lidx, &p->v);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncBIT_STRING_NR_MAC_PDU_SubHeader_Type_ELCID_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct BIT_STRING_NR_MAC_PDU_SubHeader_Type_ELCID_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	HTON_32(&_buffer[*_lidx], p->v.d, _lidx);
	for (size_t i4 = 0; i4 < p->v.d; i4++) {
		HTON_8(&_buffer[*_lidx], p->v.v[i4], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncBIT_STRING_NR_MAC_PDU_SubHeader_Type_Length_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct BIT_STRING_NR_MAC_PDU_SubHeader_Type_Length_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	HTON_32(&_buffer[*_lidx], p->v.d, _lidx);
	for (size_t i4 = 0; i4 < p->v.d; i4++) {
		HTON_8(&_buffer[*_lidx], p->v.v[i4], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_PDU_SubHeader_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_PDU_SubHeader_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 1; i4++) {
		HTON_8(&_buffer[*_lidx], p->Reserved[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		HTON_8(&_buffer[*_lidx], p->Format[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 6; i4++) {
		HTON_8(&_buffer[*_lidx], p->LCID[i4], _lidx);
	}
	_serNrDrbEncBIT_STRING_NR_MAC_PDU_SubHeader_Type_ELCID_Optional(_buffer, _size, _lidx, &p->ELCID);
	_serNrDrbEncBIT_STRING_NR_MAC_PDU_SubHeader_Type_Length_Optional(_buffer, _size, _lidx, &p->Length);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_TimingAdvance_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_TimingAdvance_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 2; i4++) {
		HTON_8(&_buffer[*_lidx], p->TAG_ID[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 6; i4++) {
		HTON_8(&_buffer[*_lidx], p->TimingAdvanceCommand[i4], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncB8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex15_8_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct B8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex15_8_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i4 = 0; i4 < 8; i4++) {
		HTON_8(&_buffer[*_lidx], p->v[i4], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncB8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex23_16_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct B8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex23_16_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i4 = 0; i4 < 8; i4++) {
		HTON_8(&_buffer[*_lidx], p->v[i4], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncB8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex31_24_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct B8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex31_24_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i4 = 0; i4 < 8; i4++) {
		HTON_8(&_buffer[*_lidx], p->v[i4], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_SCellActDeact_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const NR_MAC_CE_SCellActDeact_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 8; i4++) {
		HTON_8(&_buffer[*_lidx], p->SCellIndex7_1[i4], _lidx);
	}
	_serNrDrbEncB8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex15_8_Optional(_buffer, _size, _lidx, &p->SCellIndex15_8);
	_serNrDrbEncB8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex23_16_Optional(_buffer, _size, _lidx, &p->SCellIndex23_16);
	_serNrDrbEncB8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex31_24_Optional(_buffer, _size, _lidx, &p->SCellIndex31_24);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_ServCellId_BwpId_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_ServCellId_BwpId_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 1; i4++) {
		HTON_8(&_buffer[*_lidx], p->Field1[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 5; i4++) {
		HTON_8(&_buffer[*_lidx], p->ServCellId[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 2; i4++) {
		HTON_8(&_buffer[*_lidx], p->BwpId[i4], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_SP_ResourceSetActDeact_Octet2_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_SP_ResourceSetActDeact_Octet2_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 1; i4++) {
		HTON_8(&_buffer[*_lidx], p->Reserved[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		HTON_8(&_buffer[*_lidx], p->IM[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 6; i4++) {
		HTON_8(&_buffer[*_lidx], p->CSI_RS_ResourcesetId[i4], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_SP_ResourceSetActDeact_Octet3_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_SP_ResourceSetActDeact_Octet3_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 2; i4++) {
		HTON_8(&_buffer[*_lidx], p->Reserved[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 6; i4++) {
		HTON_8(&_buffer[*_lidx], p->CSI_IM_ResourcesetId[i4], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_SP_ResourceSetActDeact_Octet3_Type_NR_MAC_CE_SP_ResourceSetActDeact_Type_Octet3_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_SP_ResourceSetActDeact_Octet3_Type_NR_MAC_CE_SP_ResourceSetActDeact_Type_Octet3_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	_serNrDrbEncNR_MAC_CE_SP_ResourceSetActDeact_Octet3_Type(_buffer, _size, _lidx, &p->v);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_SP_ResourceSetActDeact_TciStateId_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_SP_ResourceSetActDeact_TciStateId_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i5 = 0; i5 < 1; i5++) {
		HTON_8(&_buffer[*_lidx], p->Reserved[i5], _lidx);
	}
	for (size_t i5 = 0; i5 < 7; i5++) {
		HTON_8(&_buffer[*_lidx], p->Id[i5], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_SP_ResourceSetActDeact_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_SP_ResourceSetActDeact_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncNR_MAC_CE_ServCellId_BwpId_Type(_buffer, _size, _lidx, &p->Octet1);
	_serNrDrbEncNR_MAC_CE_SP_ResourceSetActDeact_Octet2_Type(_buffer, _size, _lidx, &p->Octet2);
	_serNrDrbEncNR_MAC_CE_SP_ResourceSetActDeact_Octet3_Type_NR_MAC_CE_SP_ResourceSetActDeact_Type_Octet3_Optional(_buffer, _size, _lidx, &p->Octet3);
	HTON_32(&_buffer[*_lidx], p->IdList.d, _lidx);
	for (size_t i4 = 0; i4 < p->IdList.d; i4++) {
		_serNrDrbEncNR_MAC_CE_SP_ResourceSetActDeact_TciStateId_Type(_buffer, _size, _lidx, &p->IdList.v[i4]);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_CSI_TriggerStateSubselection_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_CSI_TriggerStateSubselection_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncNR_MAC_CE_ServCellId_BwpId_Type(_buffer, _size, _lidx, &p->Octet1);
	HTON_32(&_buffer[*_lidx], p->Selection.d, _lidx);
	for (size_t i4 = 0; i4 < p->Selection.d; i4++) {
		for (size_t i5 = 0; i5 < 8; i5++) {
			HTON_8(&_buffer[*_lidx], p->Selection.v[i4][i5], _lidx);
		}
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_TCI_StatesActDeact_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_TCI_StatesActDeact_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncNR_MAC_CE_ServCellId_BwpId_Type(_buffer, _size, _lidx, &p->Octet1);
	HTON_32(&_buffer[*_lidx], p->Status.d, _lidx);
	for (size_t i4 = 0; i4 < p->Status.d; i4++) {
		for (size_t i5 = 0; i5 < 8; i5++) {
			HTON_8(&_buffer[*_lidx], p->Status.v[i4][i5], _lidx);
		}
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_TCI_StateIndication_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_TCI_StateIndication_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 5; i4++) {
		HTON_8(&_buffer[*_lidx], p->ServCellId[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 4; i4++) {
		HTON_8(&_buffer[*_lidx], p->CoresetId[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 7; i4++) {
		HTON_8(&_buffer[*_lidx], p->TciStateId[i4], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_SP_CSI_ReportingActDeact_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_SP_CSI_ReportingActDeact_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncNR_MAC_CE_ServCellId_BwpId_Type(_buffer, _size, _lidx, &p->Octet1);
	for (size_t i4 = 0; i4 < 4; i4++) {
		HTON_8(&_buffer[*_lidx], p->Reserved[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 4; i4++) {
		HTON_8(&_buffer[*_lidx], p->ConfigState[i4], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_SP_SRS_ActDeact_Octet2_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_SP_SRS_ActDeact_Octet2_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 2; i4++) {
		HTON_8(&_buffer[*_lidx], p->Reserved[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		HTON_8(&_buffer[*_lidx], p->C[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		HTON_8(&_buffer[*_lidx], p->SUL[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 4; i4++) {
		HTON_8(&_buffer[*_lidx], p->SRS_ResourcesetId[i4], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_SP_SRS_ActDeact_ResourceId_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_SP_SRS_ActDeact_ResourceId_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i5 = 0; i5 < 1; i5++) {
		HTON_8(&_buffer[*_lidx], p->F[i5], _lidx);
	}
	for (size_t i5 = 0; i5 < 7; i5++) {
		HTON_8(&_buffer[*_lidx], p->Id[i5], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_SP_SRS_ActDeact_ResourceInfo_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_SP_SRS_ActDeact_ResourceInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i5 = 0; i5 < 1; i5++) {
		HTON_8(&_buffer[*_lidx], p->Reserved[i5], _lidx);
	}
	for (size_t i5 = 0; i5 < 5; i5++) {
		HTON_8(&_buffer[*_lidx], p->ServingCellId[i5], _lidx);
	}
	for (size_t i5 = 0; i5 < 2; i5++) {
		HTON_8(&_buffer[*_lidx], p->BwpId[i5], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_SP_SRS_ActDeact_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_SP_SRS_ActDeact_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncNR_MAC_CE_ServCellId_BwpId_Type(_buffer, _size, _lidx, &p->Octet1);
	_serNrDrbEncNR_MAC_CE_SP_SRS_ActDeact_Octet2_Type(_buffer, _size, _lidx, &p->Octet2);
	HTON_32(&_buffer[*_lidx], p->ResourceIdList.d, _lidx);
	for (size_t i4 = 0; i4 < p->ResourceIdList.d; i4++) {
		_serNrDrbEncNR_MAC_CE_SP_SRS_ActDeact_ResourceId_Type(_buffer, _size, _lidx, &p->ResourceIdList.v[i4]);
	}
	HTON_32(&_buffer[*_lidx], p->ResourceInfoList.d, _lidx);
	for (size_t i4 = 0; i4 < p->ResourceInfoList.d; i4++) {
		_serNrDrbEncNR_MAC_CE_SP_SRS_ActDeact_ResourceInfo_Type(_buffer, _size, _lidx, &p->ResourceInfoList.v[i4]);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_PUCCH_SpatialRelationActDeact_Octet2_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_PUCCH_SpatialRelationActDeact_Octet2_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 1; i4++) {
		HTON_8(&_buffer[*_lidx], p->Reserved[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 7; i4++) {
		HTON_8(&_buffer[*_lidx], p->ResourceId[i4], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_PUCCH_SpatialRelationActDeact_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_PUCCH_SpatialRelationActDeact_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncNR_MAC_CE_ServCellId_BwpId_Type(_buffer, _size, _lidx, &p->Octet1);
	_serNrDrbEncNR_MAC_CE_PUCCH_SpatialRelationActDeact_Octet2_Type(_buffer, _size, _lidx, &p->Octet2);
	for (size_t i4 = 0; i4 < 8; i4++) {
		HTON_8(&_buffer[*_lidx], p->ActivationStatus[i4], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_SP_ZP_ResourceSetActDeact_Octet2_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_SP_ZP_ResourceSetActDeact_Octet2_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 4; i4++) {
		HTON_8(&_buffer[*_lidx], p->Reserved[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 4; i4++) {
		HTON_8(&_buffer[*_lidx], p->Id[i4], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_SP_ZP_ResourceSetActDeact_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_SP_ZP_ResourceSetActDeact_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncNR_MAC_CE_ServCellId_BwpId_Type(_buffer, _size, _lidx, &p->Octet1);
	_serNrDrbEncNR_MAC_CE_SP_ZP_ResourceSetActDeact_Octet2_Type(_buffer, _size, _lidx, &p->Octet2);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_RecommendedBitrate_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_RecommendedBitrate_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 6; i4++) {
		HTON_8(&_buffer[*_lidx], p->LCID[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		HTON_8(&_buffer[*_lidx], p->UL_DL[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 6; i4++) {
		HTON_8(&_buffer[*_lidx], p->Bitrate[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		HTON_8(&_buffer[*_lidx], p->X[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 2; i4++) {
		HTON_8(&_buffer[*_lidx], p->Reserved[i4], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_ControlElementDL_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union NR_MAC_ControlElementDL_Type_Value* p, enum NR_MAC_ControlElementDL_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_MAC_ControlElementDL_Type_ContentionResolutionID) {
		for (size_t i4 = 0; i4 < 48; i4++) {
			HTON_8(&_buffer[*_lidx], p->ContentionResolutionID[i4], _lidx);
		}
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementDL_Type_TimingAdvance) {
		_serNrDrbEncNR_MAC_CE_TimingAdvance_Type(_buffer, _size, _lidx, &p->TimingAdvance);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementDL_Type_SCellActDeact) {
		_serNrDrbEncNR_MAC_CE_SCellActDeact_Type(_buffer, _size, _lidx, &p->SCellActDeact);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementDL_Type_DuplicationActDeact) {
		for (size_t i4 = 0; i4 < 8; i4++) {
			HTON_8(&_buffer[*_lidx], p->DuplicationActDeact[i4], _lidx);
		}
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementDL_Type_SP_ResourceSetActDeact) {
		_serNrDrbEncNR_MAC_CE_SP_ResourceSetActDeact_Type(_buffer, _size, _lidx, &p->SP_ResourceSetActDeact);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementDL_Type_CSI_TriggerStateSubselection) {
		_serNrDrbEncNR_MAC_CE_CSI_TriggerStateSubselection_Type(_buffer, _size, _lidx, &p->CSI_TriggerStateSubselection);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementDL_Type_TCI_StatesActDeact) {
		_serNrDrbEncNR_MAC_CE_TCI_StatesActDeact_Type(_buffer, _size, _lidx, &p->TCI_StatesActDeact);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementDL_Type_TCI_StateIndication) {
		_serNrDrbEncNR_MAC_CE_TCI_StateIndication_Type(_buffer, _size, _lidx, &p->TCI_StateIndication);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementDL_Type_SP_CSI_ReportingActDeact) {
		_serNrDrbEncNR_MAC_CE_SP_CSI_ReportingActDeact_Type(_buffer, _size, _lidx, &p->SP_CSI_ReportingActDeact);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementDL_Type_SP_SRS_ActDeact) {
		_serNrDrbEncNR_MAC_CE_SP_SRS_ActDeact_Type(_buffer, _size, _lidx, &p->SP_SRS_ActDeact);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementDL_Type_PUCCH_SpatialRelationActDeact) {
		_serNrDrbEncNR_MAC_CE_PUCCH_SpatialRelationActDeact_Type(_buffer, _size, _lidx, &p->PUCCH_SpatialRelationActDeact);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementDL_Type_SP_ZP_ResourceSetActDeact) {
		_serNrDrbEncNR_MAC_CE_SP_ZP_ResourceSetActDeact_Type(_buffer, _size, _lidx, &p->SP_ZP_ResourceSetActDeact);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementDL_Type_RecommendatdBitrate) {
		_serNrDrbEncNR_MAC_CE_RecommendedBitrate_Type(_buffer, _size, _lidx, &p->RecommendatdBitrate);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrDrbEncNR_MAC_ControlElementDL_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_ControlElementDL_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrDrbEncNR_MAC_ControlElementDL_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_ControlElementDL_Type_NR_MAC_CE_SubPDU_DL_Type_ControlElement_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_ControlElementDL_Type_NR_MAC_CE_SubPDU_DL_Type_ControlElement_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	_serNrDrbEncNR_MAC_ControlElementDL_Type(_buffer, _size, _lidx, &p->v);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_SubPDU_DL_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_SubPDU_DL_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncNR_MAC_PDU_SubHeader_Type(_buffer, _size, _lidx, &p->SubHeader);
	_serNrDrbEncNR_MAC_ControlElementDL_Type_NR_MAC_CE_SubPDU_DL_Type_ControlElement_Optional(_buffer, _size, _lidx, &p->ControlElement);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_SubPDU_DL_List_Type_NR_MAC_PDU_DL_Type_CE_SubPDUList_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_SubPDU_DL_List_Type_NR_MAC_PDU_DL_Type_CE_SubPDUList_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	HTON_32(&_buffer[*_lidx], p->v.d, _lidx);
	for (size_t i3 = 0; i3 < p->v.d; i3++) {
		_serNrDrbEncNR_MAC_CE_SubPDU_DL_Type(_buffer, _size, _lidx, &p->v.v[i3]);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_SDU_SubPDU_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_SDU_SubPDU_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncNR_MAC_PDU_SubHeader_Type(_buffer, _size, _lidx, &p->SubHeader);
	HTON_32(&_buffer[*_lidx], p->SDU.d, _lidx);
	for (size_t i4 = 0; i4 < p->SDU.d; i4++) {
		HTON_8(&_buffer[*_lidx], p->SDU.v[i4], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_DL_Type_SDU_SubPDUList_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_DL_Type_SDU_SubPDUList_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	HTON_32(&_buffer[*_lidx], p->v.d, _lidx);
	for (size_t i3 = 0; i3 < p->v.d; i3++) {
		_serNrDrbEncNR_MAC_SDU_SubPDU_Type(_buffer, _size, _lidx, &p->v.v[i3]);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_Padding_SubPDU_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_Padding_SubPDU_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncNR_MAC_PDU_SubHeader_Type(_buffer, _size, _lidx, &p->SubHeader);
	HTON_32(&_buffer[*_lidx], p->Padding.d, _lidx);
	for (size_t i3 = 0; i3 < p->Padding.d; i3++) {
		HTON_8(&_buffer[*_lidx], p->Padding.v[i3], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_DL_Type_Padding_SubPDU_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_DL_Type_Padding_SubPDU_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	_serNrDrbEncNR_MAC_Padding_SubPDU_Type(_buffer, _size, _lidx, &p->v);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_PDU_DL_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_PDU_DL_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncNR_MAC_CE_SubPDU_DL_List_Type_NR_MAC_PDU_DL_Type_CE_SubPDUList_Optional(_buffer, _size, _lidx, &p->CE_SubPDUList);
	_serNrDrbEncNR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_DL_Type_SDU_SubPDUList_Optional(_buffer, _size, _lidx, &p->SDU_SubPDUList);
	_serNrDrbEncNR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_DL_Type_Padding_SubPDU_Optional(_buffer, _size, _lidx, &p->Padding_SubPDU);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_UL_Type_SDU_SubPDUList_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_UL_Type_SDU_SubPDUList_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	HTON_32(&_buffer[*_lidx], p->v.d, _lidx);
	for (size_t i3 = 0; i3 < p->v.d; i3++) {
		_serNrDrbEncNR_MAC_SDU_SubPDU_Type(_buffer, _size, _lidx, &p->v.v[i3]);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_ShortBSR_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_ShortBSR_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 3; i4++) {
		HTON_8(&_buffer[*_lidx], p->LCG[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 5; i4++) {
		HTON_8(&_buffer[*_lidx], p->BufferSize[i4], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_LongBSR_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_LongBSR_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 8; i4++) {
		HTON_8(&_buffer[*_lidx], p->LCG_Presence[i4], _lidx);
	}
	HTON_32(&_buffer[*_lidx], p->BufferSizeList.d, _lidx);
	for (size_t i4 = 0; i4 < p->BufferSizeList.d; i4++) {
		for (size_t i5 = 0; i5 < 1; i5++) {
			HTON_8(&_buffer[*_lidx], p->BufferSizeList.v[i4][i5], _lidx);
		}
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncB2_Type_NR_MAC_CE_PH_Record_Type_MPE_or_R_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct B2_Type_NR_MAC_CE_PH_Record_Type_MPE_or_R_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i4 = 0; i4 < 2; i4++) {
		HTON_8(&_buffer[*_lidx], p->v[i4], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncB6_Type_NR_MAC_CE_PH_Record_Type_PCMaxc_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct B6_Type_NR_MAC_CE_PH_Record_Type_PCMaxc_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i4 = 0; i4 < 6; i4++) {
		HTON_8(&_buffer[*_lidx], p->v[i4], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_SingleEntryPHR_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const NR_MAC_CE_SingleEntryPHR_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 1; i4++) {
		HTON_8(&_buffer[*_lidx], p->P_Bit[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		HTON_8(&_buffer[*_lidx], p->V_Bit[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 6; i4++) {
		HTON_8(&_buffer[*_lidx], p->Value[i4], _lidx);
	}
	_serNrDrbEncB2_Type_NR_MAC_CE_PH_Record_Type_MPE_or_R_Optional(_buffer, _size, _lidx, &p->MPE_or_R);
	_serNrDrbEncB6_Type_NR_MAC_CE_PH_Record_Type_PCMaxc_Optional(_buffer, _size, _lidx, &p->PCMaxc);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_SCellFlags_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_SCellFlags_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 8; i4++) {
		HTON_8(&_buffer[*_lidx], p->SCellIndex7_1[i4], _lidx);
	}
	_serNrDrbEncB8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex15_8_Optional(_buffer, _size, _lidx, &p->SCellIndex15_8);
	_serNrDrbEncB8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex23_16_Optional(_buffer, _size, _lidx, &p->SCellIndex23_16);
	_serNrDrbEncB8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex31_24_Optional(_buffer, _size, _lidx, &p->SCellIndex31_24);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_PH_Record_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_PH_Record_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i5 = 0; i5 < 1; i5++) {
		HTON_8(&_buffer[*_lidx], p->P_Bit[i5], _lidx);
	}
	for (size_t i5 = 0; i5 < 1; i5++) {
		HTON_8(&_buffer[*_lidx], p->V_Bit[i5], _lidx);
	}
	for (size_t i5 = 0; i5 < 6; i5++) {
		HTON_8(&_buffer[*_lidx], p->Value[i5], _lidx);
	}
	_serNrDrbEncB2_Type_NR_MAC_CE_PH_Record_Type_MPE_or_R_Optional(_buffer, _size, _lidx, &p->MPE_or_R);
	_serNrDrbEncB6_Type_NR_MAC_CE_PH_Record_Type_PCMaxc_Optional(_buffer, _size, _lidx, &p->PCMaxc);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_MultiEntryPHR_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_MultiEntryPHR_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncNR_MAC_CE_SCellFlags_Type(_buffer, _size, _lidx, &p->PHFieldPresentForSCell);
	HTON_32(&_buffer[*_lidx], p->PH_Record.d, _lidx);
	for (size_t i4 = 0; i4 < p->PH_Record.d; i4++) {
		_serNrDrbEncNR_MAC_CE_PH_Record_Type(_buffer, _size, _lidx, &p->PH_Record.v[i4]);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_ControlElementUL_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union NR_MAC_ControlElementUL_Type_Value* p, enum NR_MAC_ControlElementUL_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_MAC_ControlElementUL_Type_ShortBSR) {
		_serNrDrbEncNR_MAC_CE_ShortBSR_Type(_buffer, _size, _lidx, &p->ShortBSR);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementUL_Type_LongBSR) {
		_serNrDrbEncNR_MAC_CE_LongBSR_Type(_buffer, _size, _lidx, &p->LongBSR);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementUL_Type_C_RNTI) {
		for (size_t i4 = 0; i4 < 16; i4++) {
			HTON_8(&_buffer[*_lidx], p->C_RNTI[i4], _lidx);
		}
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementUL_Type_SingleEntryPHR) {
		_serNrDrbEncNR_MAC_CE_SingleEntryPHR_Type(_buffer, _size, _lidx, &p->SingleEntryPHR);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementUL_Type_MultiEntryPHR) {
		_serNrDrbEncNR_MAC_CE_MultiEntryPHR_Type(_buffer, _size, _lidx, &p->MultiEntryPHR);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementUL_Type_RecommendedBitrate) {
		_serNrDrbEncNR_MAC_CE_RecommendedBitrate_Type(_buffer, _size, _lidx, &p->RecommendedBitrate);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrDrbEncNR_MAC_ControlElementUL_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_ControlElementUL_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrDrbEncNR_MAC_ControlElementUL_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_ControlElementUL_Type_NR_MAC_CE_SubPDU_UL_Type_ControlElement_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_ControlElementUL_Type_NR_MAC_CE_SubPDU_UL_Type_ControlElement_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	_serNrDrbEncNR_MAC_ControlElementUL_Type(_buffer, _size, _lidx, &p->v);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_SubPDU_UL_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_SubPDU_UL_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncNR_MAC_PDU_SubHeader_Type(_buffer, _size, _lidx, &p->SubHeader);
	_serNrDrbEncNR_MAC_ControlElementUL_Type_NR_MAC_CE_SubPDU_UL_Type_ControlElement_Optional(_buffer, _size, _lidx, &p->ControlElement);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_CE_SubPDU_UL_List_Type_NR_MAC_PDU_UL_Type_CE_SubPDUList_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_CE_SubPDU_UL_List_Type_NR_MAC_PDU_UL_Type_CE_SubPDUList_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	HTON_32(&_buffer[*_lidx], p->v.d, _lidx);
	for (size_t i3 = 0; i3 < p->v.d; i3++) {
		_serNrDrbEncNR_MAC_CE_SubPDU_UL_Type(_buffer, _size, _lidx, &p->v.v[i3]);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_UL_Type_Padding_SubPDU_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_UL_Type_Padding_SubPDU_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	_serNrDrbEncNR_MAC_Padding_SubPDU_Type(_buffer, _size, _lidx, &p->v);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_PDU_UL_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_PDU_UL_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncNR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_UL_Type_SDU_SubPDUList_Optional(_buffer, _size, _lidx, &p->SDU_SubPDUList);
	_serNrDrbEncNR_MAC_CE_SubPDU_UL_List_Type_NR_MAC_PDU_UL_Type_CE_SubPDUList_Optional(_buffer, _size, _lidx, &p->CE_SubPDUList);
	_serNrDrbEncNR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_UL_Type_Padding_SubPDU_Optional(_buffer, _size, _lidx, &p->Padding_SubPDU);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_MAC_PDU_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union NR_MAC_PDU_Type_Value* p, enum NR_MAC_PDU_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_MAC_PDU_Type_DL) {
		_serNrDrbEncNR_MAC_PDU_DL_Type(_buffer, _size, _lidx, &p->DL);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_PDU_Type_UL) {
		_serNrDrbEncNR_MAC_PDU_UL_Type(_buffer, _size, _lidx, &p->UL);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrDrbEncNR_MAC_PDU_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_MAC_PDU_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrDrbEncNR_MAC_PDU_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_UMD_HeaderNoSN_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_UMD_HeaderNoSN_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i3 = 0; i3 < 2; i3++) {
		HTON_8(&_buffer[*_lidx], p->SegmentationInfo[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 6; i3++) {
		HTON_8(&_buffer[*_lidx], p->Reserved[i3], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_UMD_PduNoSN_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_UMD_PduNoSN_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncNR_RLC_UMD_HeaderNoSN_Type(_buffer, _size, _lidx, &p->Header);
	HTON_32(&_buffer[*_lidx], p->Data.d, _lidx);
	for (size_t i3 = 0; i3 < p->Data.d; i3++) {
		HTON_8(&_buffer[*_lidx], p->Data.v[i3], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_SegmentOffset_Type_NR_RLC_UMD_HeaderSN6Bit_Type_SegmentOffset_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_SegmentOffset_Type_NR_RLC_UMD_HeaderSN6Bit_Type_SegmentOffset_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i3 = 0; i3 < 16; i3++) {
		HTON_8(&_buffer[*_lidx], p->v[i3], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_UMD_HeaderSN6Bit_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_UMD_HeaderSN6Bit_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i3 = 0; i3 < 2; i3++) {
		HTON_8(&_buffer[*_lidx], p->SegmentationInfo[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 6; i3++) {
		HTON_8(&_buffer[*_lidx], p->SequenceNumber[i3], _lidx);
	}
	_serNrDrbEncNR_RLC_SegmentOffset_Type_NR_RLC_UMD_HeaderSN6Bit_Type_SegmentOffset_Optional(_buffer, _size, _lidx, &p->SegmentOffset);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_UMD_PduSN6Bit_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_UMD_PduSN6Bit_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncNR_RLC_UMD_HeaderSN6Bit_Type(_buffer, _size, _lidx, &p->Header);
	HTON_32(&_buffer[*_lidx], p->Data.d, _lidx);
	for (size_t i3 = 0; i3 < p->Data.d; i3++) {
		HTON_8(&_buffer[*_lidx], p->Data.v[i3], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_SegmentOffset_Type_NR_RLC_UMD_HeaderSN12Bit_Type_SegmentOffset_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_SegmentOffset_Type_NR_RLC_UMD_HeaderSN12Bit_Type_SegmentOffset_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i3 = 0; i3 < 16; i3++) {
		HTON_8(&_buffer[*_lidx], p->v[i3], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_UMD_HeaderSN12Bit_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_UMD_HeaderSN12Bit_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i3 = 0; i3 < 2; i3++) {
		HTON_8(&_buffer[*_lidx], p->SegmentationInfo[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 2; i3++) {
		HTON_8(&_buffer[*_lidx], p->Reserved[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 12; i3++) {
		HTON_8(&_buffer[*_lidx], p->SequenceNumber[i3], _lidx);
	}
	_serNrDrbEncNR_RLC_SegmentOffset_Type_NR_RLC_UMD_HeaderSN12Bit_Type_SegmentOffset_Optional(_buffer, _size, _lidx, &p->SegmentOffset);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_UMD_PduSN12Bit_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_UMD_PduSN12Bit_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncNR_RLC_UMD_HeaderSN12Bit_Type(_buffer, _size, _lidx, &p->Header);
	HTON_32(&_buffer[*_lidx], p->Data.d, _lidx);
	for (size_t i3 = 0; i3 < p->Data.d; i3++) {
		HTON_8(&_buffer[*_lidx], p->Data.v[i3], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_UMD_PDU_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union NR_RLC_UMD_PDU_Type_Value* p, enum NR_RLC_UMD_PDU_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_RLC_UMD_PDU_Type_NoSN) {
		_serNrDrbEncNR_RLC_UMD_PduNoSN_Type(_buffer, _size, _lidx, &p->NoSN);
		return SIDL_STATUS_OK;
	}
	if (d == NR_RLC_UMD_PDU_Type_SN6Bit) {
		_serNrDrbEncNR_RLC_UMD_PduSN6Bit_Type(_buffer, _size, _lidx, &p->SN6Bit);
		return SIDL_STATUS_OK;
	}
	if (d == NR_RLC_UMD_PDU_Type_SN12Bit) {
		_serNrDrbEncNR_RLC_UMD_PduSN12Bit_Type(_buffer, _size, _lidx, &p->SN12Bit);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrDrbEncNR_RLC_UMD_PDU_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_UMD_PDU_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrDrbEncNR_RLC_UMD_PDU_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_SegmentOffset_Type_NR_RLC_AMD_HeaderSN12Bit_Type_SegmentOffset_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_SegmentOffset_Type_NR_RLC_AMD_HeaderSN12Bit_Type_SegmentOffset_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i3 = 0; i3 < 16; i3++) {
		HTON_8(&_buffer[*_lidx], p->v[i3], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_AMD_HeaderSN12Bit_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_AMD_HeaderSN12Bit_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i3 = 0; i3 < 1; i3++) {
		HTON_8(&_buffer[*_lidx], p->D_C[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 1; i3++) {
		HTON_8(&_buffer[*_lidx], p->Poll[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 2; i3++) {
		HTON_8(&_buffer[*_lidx], p->SegmentationInfo[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 12; i3++) {
		HTON_8(&_buffer[*_lidx], p->SequenceNumber[i3], _lidx);
	}
	_serNrDrbEncNR_RLC_SegmentOffset_Type_NR_RLC_AMD_HeaderSN12Bit_Type_SegmentOffset_Optional(_buffer, _size, _lidx, &p->SegmentOffset);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_AMD_PduSN12Bit_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_AMD_PduSN12Bit_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncNR_RLC_AMD_HeaderSN12Bit_Type(_buffer, _size, _lidx, &p->Header);
	HTON_32(&_buffer[*_lidx], p->Data.d, _lidx);
	for (size_t i3 = 0; i3 < p->Data.d; i3++) {
		HTON_8(&_buffer[*_lidx], p->Data.v[i3], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_SegmentOffset_Type_NR_RLC_AMD_HeaderSN18Bit_Type_SegmentOffset_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_SegmentOffset_Type_NR_RLC_AMD_HeaderSN18Bit_Type_SegmentOffset_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i3 = 0; i3 < 16; i3++) {
		HTON_8(&_buffer[*_lidx], p->v[i3], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_AMD_HeaderSN18Bit_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_AMD_HeaderSN18Bit_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i3 = 0; i3 < 1; i3++) {
		HTON_8(&_buffer[*_lidx], p->D_C[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 1; i3++) {
		HTON_8(&_buffer[*_lidx], p->Poll[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 2; i3++) {
		HTON_8(&_buffer[*_lidx], p->SegmentationInfo[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 2; i3++) {
		HTON_8(&_buffer[*_lidx], p->Reserved[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 18; i3++) {
		HTON_8(&_buffer[*_lidx], p->SequenceNumber[i3], _lidx);
	}
	_serNrDrbEncNR_RLC_SegmentOffset_Type_NR_RLC_AMD_HeaderSN18Bit_Type_SegmentOffset_Optional(_buffer, _size, _lidx, &p->SegmentOffset);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_AMD_PduSN18Bit_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_AMD_PduSN18Bit_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncNR_RLC_AMD_HeaderSN18Bit_Type(_buffer, _size, _lidx, &p->Header);
	HTON_32(&_buffer[*_lidx], p->Data.d, _lidx);
	for (size_t i3 = 0; i3 < p->Data.d; i3++) {
		HTON_8(&_buffer[*_lidx], p->Data.v[i3], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_AMD_PDU_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union NR_RLC_AMD_PDU_Type_Value* p, enum NR_RLC_AMD_PDU_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_RLC_AMD_PDU_Type_SN12Bit) {
		_serNrDrbEncNR_RLC_AMD_PduSN12Bit_Type(_buffer, _size, _lidx, &p->SN12Bit);
		return SIDL_STATUS_OK;
	}
	if (d == NR_RLC_AMD_PDU_Type_SN18Bit) {
		_serNrDrbEncNR_RLC_AMD_PduSN18Bit_Type(_buffer, _size, _lidx, &p->SN18Bit);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrDrbEncNR_RLC_AMD_PDU_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_AMD_PDU_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrDrbEncNR_RLC_AMD_PDU_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN12Bit_Type_SOstart_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN12Bit_Type_SOstart_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i4 = 0; i4 < 16; i4++) {
		HTON_8(&_buffer[*_lidx], p->v[i4], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN12Bit_Type_SOstop_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN12Bit_Type_SOstop_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i4 = 0; i4 < 16; i4++) {
		HTON_8(&_buffer[*_lidx], p->v[i4], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncB8_Type_NR_RLC_Status_NackSN12Bit_Type_NACKrange_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct B8_Type_NR_RLC_Status_NackSN12Bit_Type_NACKrange_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i4 = 0; i4 < 8; i4++) {
		HTON_8(&_buffer[*_lidx], p->v[i4], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_Status_NackSN12Bit_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_Status_NackSN12Bit_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 12; i4++) {
		HTON_8(&_buffer[*_lidx], p->SequenceNumberNACK[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		HTON_8(&_buffer[*_lidx], p->E1[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		HTON_8(&_buffer[*_lidx], p->E2[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		HTON_8(&_buffer[*_lidx], p->E3[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		HTON_8(&_buffer[*_lidx], p->Reserved[i4], _lidx);
	}
	_serNrDrbEncNR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN12Bit_Type_SOstart_Optional(_buffer, _size, _lidx, &p->SOstart);
	_serNrDrbEncNR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN12Bit_Type_SOstop_Optional(_buffer, _size, _lidx, &p->SOstop);
	_serNrDrbEncB8_Type_NR_RLC_Status_NackSN12Bit_Type_NACKrange_Optional(_buffer, _size, _lidx, &p->NACKrange);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_Status_NackListSN12Bit_Type_NackList_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_Status_NackListSN12Bit_Type_NackList_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	HTON_32(&_buffer[*_lidx], p->v.d, _lidx);
	for (size_t i3 = 0; i3 < p->v.d; i3++) {
		_serNrDrbEncNR_RLC_Status_NackSN12Bit_Type(_buffer, _size, _lidx, &p->v.v[i3]);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_StatusPduSN12Bit_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_StatusPduSN12Bit_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i3 = 0; i3 < 1; i3++) {
		HTON_8(&_buffer[*_lidx], p->D_C[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 3; i3++) {
		HTON_8(&_buffer[*_lidx], p->CPT[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 12; i3++) {
		HTON_8(&_buffer[*_lidx], p->SequenceNumberACK[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 1; i3++) {
		HTON_8(&_buffer[*_lidx], p->E1[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 7; i3++) {
		HTON_8(&_buffer[*_lidx], p->Reserved[i3], _lidx);
	}
	_serNrDrbEncNR_RLC_Status_NackListSN12Bit_Type_NackList_Optional(_buffer, _size, _lidx, &p->NackList);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN18Bit_Type_SOstart_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN18Bit_Type_SOstart_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i4 = 0; i4 < 16; i4++) {
		HTON_8(&_buffer[*_lidx], p->v[i4], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN18Bit_Type_SOstop_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN18Bit_Type_SOstop_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i4 = 0; i4 < 16; i4++) {
		HTON_8(&_buffer[*_lidx], p->v[i4], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncB8_Type_NR_RLC_Status_NackSN18Bit_Type_NACKrange_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct B8_Type_NR_RLC_Status_NackSN18Bit_Type_NACKrange_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i4 = 0; i4 < 8; i4++) {
		HTON_8(&_buffer[*_lidx], p->v[i4], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_Status_NackSN18Bit_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_Status_NackSN18Bit_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 18; i4++) {
		HTON_8(&_buffer[*_lidx], p->SequenceNumberNACK[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		HTON_8(&_buffer[*_lidx], p->E1[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		HTON_8(&_buffer[*_lidx], p->E2[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		HTON_8(&_buffer[*_lidx], p->E3[i4], _lidx);
	}
	for (size_t i4 = 0; i4 < 3; i4++) {
		HTON_8(&_buffer[*_lidx], p->Reserved[i4], _lidx);
	}
	_serNrDrbEncNR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN18Bit_Type_SOstart_Optional(_buffer, _size, _lidx, &p->SOstart);
	_serNrDrbEncNR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN18Bit_Type_SOstop_Optional(_buffer, _size, _lidx, &p->SOstop);
	_serNrDrbEncB8_Type_NR_RLC_Status_NackSN18Bit_Type_NACKrange_Optional(_buffer, _size, _lidx, &p->NACKrange);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_Status_NackListSN18Bit_Type_NackList_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_Status_NackListSN18Bit_Type_NackList_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	HTON_32(&_buffer[*_lidx], p->v.d, _lidx);
	for (size_t i3 = 0; i3 < p->v.d; i3++) {
		_serNrDrbEncNR_RLC_Status_NackSN18Bit_Type(_buffer, _size, _lidx, &p->v.v[i3]);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_StatusPduSN18Bit_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_StatusPduSN18Bit_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i3 = 0; i3 < 1; i3++) {
		HTON_8(&_buffer[*_lidx], p->D_C[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 3; i3++) {
		HTON_8(&_buffer[*_lidx], p->CPT[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 18; i3++) {
		HTON_8(&_buffer[*_lidx], p->SequenceNumberACK[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 1; i3++) {
		HTON_8(&_buffer[*_lidx], p->E1[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 1; i3++) {
		HTON_8(&_buffer[*_lidx], p->Reserved[i3], _lidx);
	}
	_serNrDrbEncNR_RLC_Status_NackListSN18Bit_Type_NackList_Optional(_buffer, _size, _lidx, &p->NackList);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_AM_StatusPDU_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union NR_RLC_AM_StatusPDU_Type_Value* p, enum NR_RLC_AM_StatusPDU_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_RLC_AM_StatusPDU_Type_SN12Bit) {
		_serNrDrbEncNR_RLC_StatusPduSN12Bit_Type(_buffer, _size, _lidx, &p->SN12Bit);
		return SIDL_STATUS_OK;
	}
	if (d == NR_RLC_AM_StatusPDU_Type_SN18Bit) {
		_serNrDrbEncNR_RLC_StatusPduSN18Bit_Type(_buffer, _size, _lidx, &p->SN18Bit);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrDrbEncNR_RLC_AM_StatusPDU_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_AM_StatusPDU_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrDrbEncNR_RLC_AM_StatusPDU_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_RLC_PDU_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union NR_RLC_PDU_Type_Value* p, enum NR_RLC_PDU_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_RLC_PDU_Type_TMD) {
		HTON_32(&_buffer[*_lidx], p->TMD.d, _lidx);
		for (size_t i3 = 0; i3 < p->TMD.d; i3++) {
			HTON_8(&_buffer[*_lidx], p->TMD.v[i3], _lidx);
		}
		return SIDL_STATUS_OK;
	}
	if (d == NR_RLC_PDU_Type_UMD) {
		_serNrDrbEncNR_RLC_UMD_PDU_Type(_buffer, _size, _lidx, &p->UMD);
		return SIDL_STATUS_OK;
	}
	if (d == NR_RLC_PDU_Type_AMD) {
		_serNrDrbEncNR_RLC_AMD_PDU_Type(_buffer, _size, _lidx, &p->AMD);
		return SIDL_STATUS_OK;
	}
	if (d == NR_RLC_PDU_Type_Status) {
		_serNrDrbEncNR_RLC_AM_StatusPDU_Type(_buffer, _size, _lidx, &p->Status);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrDrbEncNR_RLC_PDU_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RLC_PDU_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrDrbEncNR_RLC_PDU_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncB32_Type_NR_PDCP_DataPduSN12Bits_Type_MAC_I_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct B32_Type_NR_PDCP_DataPduSN12Bits_Type_MAC_I_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i3 = 0; i3 < 32; i3++) {
		HTON_8(&_buffer[*_lidx], p->v[i3], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_PDCP_DataPduSN12Bits_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_PDCP_DataPduSN12Bits_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i3 = 0; i3 < 1; i3++) {
		HTON_8(&_buffer[*_lidx], p->D_C[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 3; i3++) {
		HTON_8(&_buffer[*_lidx], p->Reserved[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 12; i3++) {
		HTON_8(&_buffer[*_lidx], p->SequenceNumber[i3], _lidx);
	}
	HTON_32(&_buffer[*_lidx], p->SDU.d, _lidx);
	for (size_t i3 = 0; i3 < p->SDU.d; i3++) {
		HTON_8(&_buffer[*_lidx], p->SDU.v[i3], _lidx);
	}
	_serNrDrbEncB32_Type_NR_PDCP_DataPduSN12Bits_Type_MAC_I_Optional(_buffer, _size, _lidx, &p->MAC_I);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncB32_Type_NR_PDCP_DataPduSN18Bits_Type_MAC_I_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct B32_Type_NR_PDCP_DataPduSN18Bits_Type_MAC_I_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i3 = 0; i3 < 32; i3++) {
		HTON_8(&_buffer[*_lidx], p->v[i3], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_PDCP_DataPduSN18Bits_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_PDCP_DataPduSN18Bits_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i3 = 0; i3 < 1; i3++) {
		HTON_8(&_buffer[*_lidx], p->D_C[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 5; i3++) {
		HTON_8(&_buffer[*_lidx], p->Reserved[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 18; i3++) {
		HTON_8(&_buffer[*_lidx], p->SequenceNumber[i3], _lidx);
	}
	HTON_32(&_buffer[*_lidx], p->SDU.d, _lidx);
	for (size_t i3 = 0; i3 < p->SDU.d; i3++) {
		HTON_8(&_buffer[*_lidx], p->SDU.v[i3], _lidx);
	}
	_serNrDrbEncB32_Type_NR_PDCP_DataPduSN18Bits_Type_MAC_I_Optional(_buffer, _size, _lidx, &p->MAC_I);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncOCTET_STRING_NR_PDCP_CtrlPduStatus_Type_Bitmap_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct OCTET_STRING_NR_PDCP_CtrlPduStatus_Type_Bitmap_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	HTON_32(&_buffer[*_lidx], p->v.d, _lidx);
	for (size_t i3 = 0; i3 < p->v.d; i3++) {
		HTON_8(&_buffer[*_lidx], p->v.v[i3], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_PDCP_CtrlPduStatus_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_PDCP_CtrlPduStatus_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i3 = 0; i3 < 1; i3++) {
		HTON_8(&_buffer[*_lidx], p->D_C[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 3; i3++) {
		HTON_8(&_buffer[*_lidx], p->PDU_Type[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 4; i3++) {
		HTON_8(&_buffer[*_lidx], p->Reserved[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 32; i3++) {
		HTON_8(&_buffer[*_lidx], p->FirstMissingCount[i3], _lidx);
	}
	_serNrDrbEncOCTET_STRING_NR_PDCP_CtrlPduStatus_Type_Bitmap_Optional(_buffer, _size, _lidx, &p->Bitmap);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_PDCP_CtrlPduRohcFeedback_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_PDCP_CtrlPduRohcFeedback_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i3 = 0; i3 < 1; i3++) {
		HTON_8(&_buffer[*_lidx], p->D_C[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 3; i3++) {
		HTON_8(&_buffer[*_lidx], p->PDU_Type[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 4; i3++) {
		HTON_8(&_buffer[*_lidx], p->Reserved[i3], _lidx);
	}
	HTON_32(&_buffer[*_lidx], p->RohcFeedback.d, _lidx);
	for (size_t i3 = 0; i3 < p->RohcFeedback.d; i3++) {
		HTON_8(&_buffer[*_lidx], p->RohcFeedback.v[i3], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_PDCP_PDU_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union NR_PDCP_PDU_Type_Value* p, enum NR_PDCP_PDU_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_PDCP_PDU_Type_DataPduSN12Bits) {
		_serNrDrbEncNR_PDCP_DataPduSN12Bits_Type(_buffer, _size, _lidx, &p->DataPduSN12Bits);
		return SIDL_STATUS_OK;
	}
	if (d == NR_PDCP_PDU_Type_DataPduSN18Bits) {
		_serNrDrbEncNR_PDCP_DataPduSN18Bits_Type(_buffer, _size, _lidx, &p->DataPduSN18Bits);
		return SIDL_STATUS_OK;
	}
	if (d == NR_PDCP_PDU_Type_CtrlPduStatus) {
		_serNrDrbEncNR_PDCP_CtrlPduStatus_Type(_buffer, _size, _lidx, &p->CtrlPduStatus);
		return SIDL_STATUS_OK;
	}
	if (d == NR_PDCP_PDU_Type_CtrlPduRohcFeedback) {
		_serNrDrbEncNR_PDCP_CtrlPduRohcFeedback_Type(_buffer, _size, _lidx, &p->CtrlPduRohcFeedback);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrDrbEncNR_PDCP_PDU_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_PDCP_PDU_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrDrbEncNR_PDCP_PDU_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncSDAP_DL_PduHeader_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct SDAP_DL_PduHeader_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i3 = 0; i3 < 1; i3++) {
		HTON_8(&_buffer[*_lidx], p->RDI[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 1; i3++) {
		HTON_8(&_buffer[*_lidx], p->RQI[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 6; i3++) {
		HTON_8(&_buffer[*_lidx], p->QFI[i3], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncSDAP_DL_PduHeader_Type_SDAP_PDU_DL_Type_Header_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct SDAP_DL_PduHeader_Type_SDAP_PDU_DL_Type_Header_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	_serNrDrbEncSDAP_DL_PduHeader_Type(_buffer, _size, _lidx, &p->v);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncSDAP_PDU_DL_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct SDAP_PDU_DL_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncSDAP_DL_PduHeader_Type_SDAP_PDU_DL_Type_Header_Optional(_buffer, _size, _lidx, &p->Header);
	HTON_32(&_buffer[*_lidx], p->Data.d, _lidx);
	for (size_t i3 = 0; i3 < p->Data.d; i3++) {
		HTON_8(&_buffer[*_lidx], p->Data.v[i3], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncSDAP_UL_PduHeader_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct SDAP_UL_PduHeader_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i3 = 0; i3 < 1; i3++) {
		HTON_8(&_buffer[*_lidx], p->DC[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 1; i3++) {
		HTON_8(&_buffer[*_lidx], p->R[i3], _lidx);
	}
	for (size_t i3 = 0; i3 < 6; i3++) {
		HTON_8(&_buffer[*_lidx], p->QFI[i3], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncSDAP_UL_PduHeader_Type_SDAP_PDU_UL_Type_Header_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct SDAP_UL_PduHeader_Type_SDAP_PDU_UL_Type_Header_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	_serNrDrbEncSDAP_UL_PduHeader_Type(_buffer, _size, _lidx, &p->v);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncSDAP_SDU_Type_SDAP_PDU_UL_Type_Data_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct SDAP_SDU_Type_SDAP_PDU_UL_Type_Data_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	HTON_32(&_buffer[*_lidx], p->v.d, _lidx);
	for (size_t i3 = 0; i3 < p->v.d; i3++) {
		HTON_8(&_buffer[*_lidx], p->v.v[i3], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncSDAP_PDU_UL_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct SDAP_PDU_UL_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncSDAP_UL_PduHeader_Type_SDAP_PDU_UL_Type_Header_Optional(_buffer, _size, _lidx, &p->Header);
	_serNrDrbEncSDAP_SDU_Type_SDAP_PDU_UL_Type_Data_Optional(_buffer, _size, _lidx, &p->Data);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncSDAP_PDU_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union SDAP_PDU_Type_Value* p, enum SDAP_PDU_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == SDAP_PDU_Type_DL) {
		_serNrDrbEncSDAP_PDU_DL_Type(_buffer, _size, _lidx, &p->DL);
		return SIDL_STATUS_OK;
	}
	if (d == SDAP_PDU_Type_UL) {
		_serNrDrbEncSDAP_PDU_UL_Type(_buffer, _size, _lidx, &p->UL);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrDrbEncSDAP_PDU_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct SDAP_PDU_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrDrbEncSDAP_PDU_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_L2DataList_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union NR_L2DataList_Type_Value* p, enum NR_L2DataList_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_L2DataList_Type_MacPdu) {
		HTON_32(&_buffer[*_lidx], p->MacPdu.d, _lidx);
		for (size_t i2 = 0; i2 < p->MacPdu.d; i2++) {
			_serNrDrbEncNR_MAC_PDU_Type(_buffer, _size, _lidx, &p->MacPdu.v[i2]);
		}
		return SIDL_STATUS_OK;
	}
	if (d == NR_L2DataList_Type_RlcPdu) {
		HTON_32(&_buffer[*_lidx], p->RlcPdu.d, _lidx);
		for (size_t i2 = 0; i2 < p->RlcPdu.d; i2++) {
			_serNrDrbEncNR_RLC_PDU_Type(_buffer, _size, _lidx, &p->RlcPdu.v[i2]);
		}
		return SIDL_STATUS_OK;
	}
	if (d == NR_L2DataList_Type_RlcSdu) {
		HTON_32(&_buffer[*_lidx], p->RlcSdu.d, _lidx);
		for (size_t i2 = 0; i2 < p->RlcSdu.d; i2++) {
			HTON_32(&_buffer[*_lidx], p->RlcSdu.v[i2].d, _lidx);
			for (size_t i3 = 0; i3 < p->RlcSdu.v[i2].d; i3++) {
				HTON_8(&_buffer[*_lidx], p->RlcSdu.v[i2].v[i3], _lidx);
			}
		}
		return SIDL_STATUS_OK;
	}
	if (d == NR_L2DataList_Type_PdcpPdu) {
		HTON_32(&_buffer[*_lidx], p->PdcpPdu.d, _lidx);
		for (size_t i2 = 0; i2 < p->PdcpPdu.d; i2++) {
			_serNrDrbEncNR_PDCP_PDU_Type(_buffer, _size, _lidx, &p->PdcpPdu.v[i2]);
		}
		return SIDL_STATUS_OK;
	}
	if (d == NR_L2DataList_Type_PdcpSdu) {
		HTON_32(&_buffer[*_lidx], p->PdcpSdu.d, _lidx);
		for (size_t i2 = 0; i2 < p->PdcpSdu.d; i2++) {
			HTON_32(&_buffer[*_lidx], p->PdcpSdu.v[i2].d, _lidx);
			for (size_t i3 = 0; i3 < p->PdcpSdu.v[i2].d; i3++) {
				HTON_8(&_buffer[*_lidx], p->PdcpSdu.v[i2].v[i3], _lidx);
			}
		}
		return SIDL_STATUS_OK;
	}
	if (d == NR_L2DataList_Type_SdapPdu) {
		HTON_32(&_buffer[*_lidx], p->SdapPdu.d, _lidx);
		for (size_t i2 = 0; i2 < p->SdapPdu.d; i2++) {
			_serNrDrbEncSDAP_PDU_Type(_buffer, _size, _lidx, &p->SdapPdu.v[i2]);
		}
		return SIDL_STATUS_OK;
	}
	if (d == NR_L2DataList_Type_SdapSdu) {
		HTON_32(&_buffer[*_lidx], p->SdapSdu.d, _lidx);
		for (size_t i2 = 0; i2 < p->SdapSdu.d; i2++) {
			HTON_32(&_buffer[*_lidx], p->SdapSdu.v[i2].d, _lidx);
			for (size_t i3 = 0; i3 < p->SdapSdu.v[i2].d; i3++) {
				HTON_8(&_buffer[*_lidx], p->SdapSdu.v[i2].v[i3], _lidx);
			}
		}
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrDrbEncNR_L2DataList_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_L2DataList_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrDrbEncNR_L2DataList_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_DRB_DataPerSlot_DL_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_DRB_DataPerSlot_DL_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_32(&_buffer[*_lidx], p->SlotOffset, _lidx);
	_serNrDrbEncNR_HarqProcessAssignment_Type_HarqProcess_Optional(_buffer, _size, _lidx, &p->HarqProcess);
	_serNrDrbEncNR_L2DataList_Type(_buffer, _size, _lidx, &p->PduSduList);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_L2Data_Request_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_L2Data_Request_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_32(&_buffer[*_lidx], p->SlotDataList.d, _lidx);
	for (size_t i1 = 0; i1 < p->SlotDataList.d; i1++) {
		_serNrDrbEncNR_DRB_DataPerSlot_DL_Type(_buffer, _size, _lidx, &p->SlotDataList.v[i1]);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNull_Type_NR_DRB_COMMON_REQ_SuppressPdcchForC_RNTI_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct Null_Type_NR_DRB_COMMON_REQ_SuppressPdcchForC_RNTI_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	HTON_8(&_buffer[*_lidx], p->v, _lidx);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_DRB_COMMON_REQ(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_DRB_COMMON_REQ* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncNR_ReqAspCommonPart_Type(_buffer, _size, _lidx, &p->Common);
	_serNrDrbEncNR_L2Data_Request_Type(_buffer, _size, _lidx, &p->U_Plane);
	_serNrDrbEncNull_Type_NR_DRB_COMMON_REQ_SuppressPdcchForC_RNTI_Optional(_buffer, _size, _lidx, &p->SuppressPdcchForC_RNTI);

	return SIDL_STATUS_OK;
}

int serNrDrbProcessFromSSEncClt(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_DRB_COMMON_REQ* FromSS)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncNR_DRB_COMMON_REQ(_buffer, _size, _lidx, FromSS);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RadioBearerId_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, union NR_RadioBearerId_Type_Value* p, enum NR_RadioBearerId_Type_Sel d)
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

static int _serNrDrbDecNR_RadioBearerId_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_RadioBearerId_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum NR_RadioBearerId_Type_Sel)_tmp;
	}
	_serNrDrbDecNR_RadioBearerId_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecQosFlow_Identification_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct QosFlow_Identification_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_32(p->PDU_SessionId, &_buffer[*_lidx], _lidx);
	NTOH_32(p->QFI, &_buffer[*_lidx], _lidx);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RoutingInfo_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, union NR_RoutingInfo_Type_Value* p, enum NR_RoutingInfo_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_RoutingInfo_Type_None) {
		NTOH_8(p->None, &_buffer[*_lidx], _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == NR_RoutingInfo_Type_RadioBearerId) {
		_serNrDrbDecNR_RadioBearerId_Type(_buffer, _size, _lidx, &p->RadioBearerId);
		return SIDL_STATUS_OK;
	}
	if (d == NR_RoutingInfo_Type_QosFlow) {
		_serNrDrbDecQosFlow_Identification_Type(_buffer, _size, _lidx, &p->QosFlow);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrDrbDecNR_RoutingInfo_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_RoutingInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum NR_RoutingInfo_Type_Sel)_tmp;
	}
	_serNrDrbDecNR_RoutingInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecRlcBearerRouting_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, union RlcBearerRouting_Type_Value* p, enum RlcBearerRouting_Type_Sel d)
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

static int _serNrDrbDecRlcBearerRouting_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct RlcBearerRouting_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum RlcBearerRouting_Type_Sel)_tmp;
	}
	_serNrDrbDecRlcBearerRouting_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecMacBearerRouting_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct MacBearerRouting_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->NR = (NR_CellId_Type)_tmp;
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecMacBearerRouting_Type_NR_ReqAspCommonPart_Type_MacBearerRouting_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct MacBearerRouting_Type_NR_ReqAspCommonPart_Type_MacBearerRouting_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	_serNrDrbDecMacBearerRouting_Type(_buffer, _size, _lidx, &p->v);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecSystemFrameNumberInfo_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, union SystemFrameNumberInfo_Type_Value* p, enum SystemFrameNumberInfo_Type_Sel d)
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

static int _serNrDrbDecSystemFrameNumberInfo_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct SystemFrameNumberInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum SystemFrameNumberInfo_Type_Sel)_tmp;
	}
	_serNrDrbDecSystemFrameNumberInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecSubFrameInfo_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, union SubFrameInfo_Type_Value* p, enum SubFrameInfo_Type_Sel d)
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

static int _serNrDrbDecSubFrameInfo_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct SubFrameInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum SubFrameInfo_Type_Sel)_tmp;
	}
	_serNrDrbDecSubFrameInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecHyperSystemFrameNumberInfo_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, HyperSystemFrameNumberInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum SystemFrameNumberInfo_Type_Sel)_tmp;
	}
	_serNrDrbDecSystemFrameNumberInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecSlotOffset_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, union SlotOffset_Type_Value* p, enum SlotOffset_Type_Sel d)
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

static int _serNrDrbDecSlotOffset_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct SlotOffset_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum SlotOffset_Type_Sel)_tmp;
	}
	_serNrDrbDecSlotOffset_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecSlotTimingInfo_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, union SlotTimingInfo_Type_Value* p, enum SlotTimingInfo_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == SlotTimingInfo_Type_SlotOffset) {
		_serNrDrbDecSlotOffset_Type(_buffer, _size, _lidx, &p->SlotOffset);
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

static int _serNrDrbDecSlotTimingInfo_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct SlotTimingInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum SlotTimingInfo_Type_Sel)_tmp;
	}
	_serNrDrbDecSlotTimingInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecSymbolTimingInfo_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, union SymbolTimingInfo_Type_Value* p, enum SymbolTimingInfo_Type_Sel d)
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

static int _serNrDrbDecSymbolTimingInfo_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct SymbolTimingInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum SymbolTimingInfo_Type_Sel)_tmp;
	}
	_serNrDrbDecSymbolTimingInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecSubFrameTiming_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct SubFrameTiming_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecSystemFrameNumberInfo_Type(_buffer, _size, _lidx, &p->SFN);
	_serNrDrbDecSubFrameInfo_Type(_buffer, _size, _lidx, &p->Subframe);
	_serNrDrbDecHyperSystemFrameNumberInfo_Type(_buffer, _size, _lidx, &p->HSFN);
	_serNrDrbDecSlotTimingInfo_Type(_buffer, _size, _lidx, &p->Slot);
	_serNrDrbDecSymbolTimingInfo_Type(_buffer, _size, _lidx, &p->Symbol);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecTimingInfo_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, union TimingInfo_Type_Value* p, enum TimingInfo_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == TimingInfo_Type_SubFrame) {
		_serNrDrbDecSubFrameTiming_Type(_buffer, _size, _lidx, &p->SubFrame);
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

static int _serNrDrbDecTimingInfo_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct TimingInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum TimingInfo_Type_Sel)_tmp;
	}
	_serNrDrbDecTimingInfo_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecReqAspControlInfo_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct ReqAspControlInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->CnfFlag, &_buffer[*_lidx], _lidx);
	NTOH_8(p->FollowOnFlag, &_buffer[*_lidx], _lidx);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_ReqAspCommonPart_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_ReqAspCommonPart_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->CellId = (NR_CellId_Type)_tmp;
	}
	_serNrDrbDecNR_RoutingInfo_Type(_buffer, _size, _lidx, &p->RoutingInfo);
	_serNrDrbDecRlcBearerRouting_Type(_buffer, _size, _lidx, &p->RlcBearerRouting);
	_serNrDrbDecMacBearerRouting_Type_NR_ReqAspCommonPart_Type_MacBearerRouting_Optional(_buffer, _size, _lidx, &p->MacBearerRouting);
	_serNrDrbDecTimingInfo_Type(_buffer, _size, _lidx, &p->TimingInfo);
	_serNrDrbDecReqAspControlInfo_Type(_buffer, _size, _lidx, &p->ControlInfo);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_HarqProcessAssignment_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, union NR_HarqProcessAssignment_Type_Value* p, enum NR_HarqProcessAssignment_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_HarqProcessAssignment_Type_Id) {
		NTOH_32(p->Id, &_buffer[*_lidx], _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == NR_HarqProcessAssignment_Type_Automatic) {
		NTOH_8(p->Automatic, &_buffer[*_lidx], _lidx);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrDrbDecNR_HarqProcessAssignment_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_HarqProcessAssignment_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum NR_HarqProcessAssignment_Type_Sel)_tmp;
	}
	_serNrDrbDecNR_HarqProcessAssignment_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_HarqProcessAssignment_Type_HarqProcess_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_HarqProcessAssignment_Type_HarqProcess_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	_serNrDrbDecNR_HarqProcessAssignment_Type(_buffer, _size, _lidx, &p->v);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecBIT_STRING_NR_MAC_PDU_SubHeader_Type_ELCID_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct BIT_STRING_NR_MAC_PDU_SubHeader_Type_ELCID_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	NTOH_32(p->v.d, &_buffer[*_lidx], _lidx);
	p->v.v = (BIT_STRING_ELEMENT*)serMalloc(_mem, p->v.d * sizeof(BIT_STRING_ELEMENT));
	for (size_t i4 = 0; i4 < p->v.d; i4++) {
		NTOH_8(p->v.v[i4], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecBIT_STRING_NR_MAC_PDU_SubHeader_Type_Length_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct BIT_STRING_NR_MAC_PDU_SubHeader_Type_Length_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	NTOH_32(p->v.d, &_buffer[*_lidx], _lidx);
	p->v.v = (BIT_STRING_ELEMENT*)serMalloc(_mem, p->v.d * sizeof(BIT_STRING_ELEMENT));
	for (size_t i4 = 0; i4 < p->v.d; i4++) {
		NTOH_8(p->v.v[i4], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_PDU_SubHeader_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_MAC_PDU_SubHeader_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 1; i4++) {
		NTOH_8(p->Reserved[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		NTOH_8(p->Format[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 6; i4++) {
		NTOH_8(p->LCID[i4], &_buffer[*_lidx], _lidx);
	}
	_serNrDrbDecBIT_STRING_NR_MAC_PDU_SubHeader_Type_ELCID_Optional(_buffer, _size, _lidx, _mem, &p->ELCID);
	_serNrDrbDecBIT_STRING_NR_MAC_PDU_SubHeader_Type_Length_Optional(_buffer, _size, _lidx, _mem, &p->Length);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_TimingAdvance_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_MAC_CE_TimingAdvance_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 2; i4++) {
		NTOH_8(p->TAG_ID[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 6; i4++) {
		NTOH_8(p->TimingAdvanceCommand[i4], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecB8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex15_8_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct B8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex15_8_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i4 = 0; i4 < 8; i4++) {
		NTOH_8(p->v[i4], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecB8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex23_16_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct B8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex23_16_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i4 = 0; i4 < 8; i4++) {
		NTOH_8(p->v[i4], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecB8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex31_24_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct B8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex31_24_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i4 = 0; i4 < 8; i4++) {
		NTOH_8(p->v[i4], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_SCellActDeact_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, NR_MAC_CE_SCellActDeact_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 8; i4++) {
		NTOH_8(p->SCellIndex7_1[i4], &_buffer[*_lidx], _lidx);
	}
	_serNrDrbDecB8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex15_8_Optional(_buffer, _size, _lidx, &p->SCellIndex15_8);
	_serNrDrbDecB8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex23_16_Optional(_buffer, _size, _lidx, &p->SCellIndex23_16);
	_serNrDrbDecB8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex31_24_Optional(_buffer, _size, _lidx, &p->SCellIndex31_24);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_ServCellId_BwpId_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_MAC_CE_ServCellId_BwpId_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 1; i4++) {
		NTOH_8(p->Field1[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 5; i4++) {
		NTOH_8(p->ServCellId[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 2; i4++) {
		NTOH_8(p->BwpId[i4], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_SP_ResourceSetActDeact_Octet2_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_MAC_CE_SP_ResourceSetActDeact_Octet2_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 1; i4++) {
		NTOH_8(p->Reserved[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		NTOH_8(p->IM[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 6; i4++) {
		NTOH_8(p->CSI_RS_ResourcesetId[i4], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_SP_ResourceSetActDeact_Octet3_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_MAC_CE_SP_ResourceSetActDeact_Octet3_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 2; i4++) {
		NTOH_8(p->Reserved[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 6; i4++) {
		NTOH_8(p->CSI_IM_ResourcesetId[i4], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_SP_ResourceSetActDeact_Octet3_Type_NR_MAC_CE_SP_ResourceSetActDeact_Type_Octet3_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_MAC_CE_SP_ResourceSetActDeact_Octet3_Type_NR_MAC_CE_SP_ResourceSetActDeact_Type_Octet3_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	_serNrDrbDecNR_MAC_CE_SP_ResourceSetActDeact_Octet3_Type(_buffer, _size, _lidx, &p->v);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_SP_ResourceSetActDeact_TciStateId_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_MAC_CE_SP_ResourceSetActDeact_TciStateId_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i5 = 0; i5 < 1; i5++) {
		NTOH_8(p->Reserved[i5], &_buffer[*_lidx], _lidx);
	}
	for (size_t i5 = 0; i5 < 7; i5++) {
		NTOH_8(p->Id[i5], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_SP_ResourceSetActDeact_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_MAC_CE_SP_ResourceSetActDeact_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecNR_MAC_CE_ServCellId_BwpId_Type(_buffer, _size, _lidx, &p->Octet1);
	_serNrDrbDecNR_MAC_CE_SP_ResourceSetActDeact_Octet2_Type(_buffer, _size, _lidx, &p->Octet2);
	_serNrDrbDecNR_MAC_CE_SP_ResourceSetActDeact_Octet3_Type_NR_MAC_CE_SP_ResourceSetActDeact_Type_Octet3_Optional(_buffer, _size, _lidx, &p->Octet3);
	NTOH_32(p->IdList.d, &_buffer[*_lidx], _lidx);
	p->IdList.v = (struct NR_MAC_CE_SP_ResourceSetActDeact_TciStateId_Type*)serMalloc(_mem, p->IdList.d * sizeof(struct NR_MAC_CE_SP_ResourceSetActDeact_TciStateId_Type));
	for (size_t i4 = 0; i4 < p->IdList.d; i4++) {
		_serNrDrbDecNR_MAC_CE_SP_ResourceSetActDeact_TciStateId_Type(_buffer, _size, _lidx, &p->IdList.v[i4]);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_CSI_TriggerStateSubselection_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_MAC_CE_CSI_TriggerStateSubselection_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecNR_MAC_CE_ServCellId_BwpId_Type(_buffer, _size, _lidx, &p->Octet1);
	NTOH_32(p->Selection.d, &_buffer[*_lidx], _lidx);
	p->Selection.v = (B8_Type*)serMalloc(_mem, p->Selection.d * sizeof(B8_Type));
	for (size_t i4 = 0; i4 < p->Selection.d; i4++) {
		for (size_t i5 = 0; i5 < 8; i5++) {
			NTOH_8(p->Selection.v[i4][i5], &_buffer[*_lidx], _lidx);
		}
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_TCI_StatesActDeact_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_MAC_CE_TCI_StatesActDeact_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecNR_MAC_CE_ServCellId_BwpId_Type(_buffer, _size, _lidx, &p->Octet1);
	NTOH_32(p->Status.d, &_buffer[*_lidx], _lidx);
	p->Status.v = (B8_Type*)serMalloc(_mem, p->Status.d * sizeof(B8_Type));
	for (size_t i4 = 0; i4 < p->Status.d; i4++) {
		for (size_t i5 = 0; i5 < 8; i5++) {
			NTOH_8(p->Status.v[i4][i5], &_buffer[*_lidx], _lidx);
		}
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_TCI_StateIndication_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_MAC_CE_TCI_StateIndication_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 5; i4++) {
		NTOH_8(p->ServCellId[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 4; i4++) {
		NTOH_8(p->CoresetId[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 7; i4++) {
		NTOH_8(p->TciStateId[i4], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_SP_CSI_ReportingActDeact_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_MAC_CE_SP_CSI_ReportingActDeact_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecNR_MAC_CE_ServCellId_BwpId_Type(_buffer, _size, _lidx, &p->Octet1);
	for (size_t i4 = 0; i4 < 4; i4++) {
		NTOH_8(p->Reserved[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 4; i4++) {
		NTOH_8(p->ConfigState[i4], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_SP_SRS_ActDeact_Octet2_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_MAC_CE_SP_SRS_ActDeact_Octet2_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 2; i4++) {
		NTOH_8(p->Reserved[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		NTOH_8(p->C[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		NTOH_8(p->SUL[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 4; i4++) {
		NTOH_8(p->SRS_ResourcesetId[i4], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_SP_SRS_ActDeact_ResourceId_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_MAC_CE_SP_SRS_ActDeact_ResourceId_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i5 = 0; i5 < 1; i5++) {
		NTOH_8(p->F[i5], &_buffer[*_lidx], _lidx);
	}
	for (size_t i5 = 0; i5 < 7; i5++) {
		NTOH_8(p->Id[i5], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_SP_SRS_ActDeact_ResourceInfo_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_MAC_CE_SP_SRS_ActDeact_ResourceInfo_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i5 = 0; i5 < 1; i5++) {
		NTOH_8(p->Reserved[i5], &_buffer[*_lidx], _lidx);
	}
	for (size_t i5 = 0; i5 < 5; i5++) {
		NTOH_8(p->ServingCellId[i5], &_buffer[*_lidx], _lidx);
	}
	for (size_t i5 = 0; i5 < 2; i5++) {
		NTOH_8(p->BwpId[i5], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_SP_SRS_ActDeact_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_MAC_CE_SP_SRS_ActDeact_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecNR_MAC_CE_ServCellId_BwpId_Type(_buffer, _size, _lidx, &p->Octet1);
	_serNrDrbDecNR_MAC_CE_SP_SRS_ActDeact_Octet2_Type(_buffer, _size, _lidx, &p->Octet2);
	NTOH_32(p->ResourceIdList.d, &_buffer[*_lidx], _lidx);
	p->ResourceIdList.v = (struct NR_MAC_CE_SP_SRS_ActDeact_ResourceId_Type*)serMalloc(_mem, p->ResourceIdList.d * sizeof(struct NR_MAC_CE_SP_SRS_ActDeact_ResourceId_Type));
	for (size_t i4 = 0; i4 < p->ResourceIdList.d; i4++) {
		_serNrDrbDecNR_MAC_CE_SP_SRS_ActDeact_ResourceId_Type(_buffer, _size, _lidx, &p->ResourceIdList.v[i4]);
	}
	NTOH_32(p->ResourceInfoList.d, &_buffer[*_lidx], _lidx);
	p->ResourceInfoList.v = (struct NR_MAC_CE_SP_SRS_ActDeact_ResourceInfo_Type*)serMalloc(_mem, p->ResourceInfoList.d * sizeof(struct NR_MAC_CE_SP_SRS_ActDeact_ResourceInfo_Type));
	for (size_t i4 = 0; i4 < p->ResourceInfoList.d; i4++) {
		_serNrDrbDecNR_MAC_CE_SP_SRS_ActDeact_ResourceInfo_Type(_buffer, _size, _lidx, &p->ResourceInfoList.v[i4]);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_PUCCH_SpatialRelationActDeact_Octet2_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_MAC_CE_PUCCH_SpatialRelationActDeact_Octet2_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 1; i4++) {
		NTOH_8(p->Reserved[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 7; i4++) {
		NTOH_8(p->ResourceId[i4], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_PUCCH_SpatialRelationActDeact_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_MAC_CE_PUCCH_SpatialRelationActDeact_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecNR_MAC_CE_ServCellId_BwpId_Type(_buffer, _size, _lidx, &p->Octet1);
	_serNrDrbDecNR_MAC_CE_PUCCH_SpatialRelationActDeact_Octet2_Type(_buffer, _size, _lidx, &p->Octet2);
	for (size_t i4 = 0; i4 < 8; i4++) {
		NTOH_8(p->ActivationStatus[i4], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_SP_ZP_ResourceSetActDeact_Octet2_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_MAC_CE_SP_ZP_ResourceSetActDeact_Octet2_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 4; i4++) {
		NTOH_8(p->Reserved[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 4; i4++) {
		NTOH_8(p->Id[i4], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_SP_ZP_ResourceSetActDeact_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_MAC_CE_SP_ZP_ResourceSetActDeact_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecNR_MAC_CE_ServCellId_BwpId_Type(_buffer, _size, _lidx, &p->Octet1);
	_serNrDrbDecNR_MAC_CE_SP_ZP_ResourceSetActDeact_Octet2_Type(_buffer, _size, _lidx, &p->Octet2);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_RecommendedBitrate_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_MAC_CE_RecommendedBitrate_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 6; i4++) {
		NTOH_8(p->LCID[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		NTOH_8(p->UL_DL[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 6; i4++) {
		NTOH_8(p->Bitrate[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		NTOH_8(p->X[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 2; i4++) {
		NTOH_8(p->Reserved[i4], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_ControlElementDL_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, union NR_MAC_ControlElementDL_Type_Value* p, enum NR_MAC_ControlElementDL_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_MAC_ControlElementDL_Type_ContentionResolutionID) {
		for (size_t i4 = 0; i4 < 48; i4++) {
			NTOH_8(p->ContentionResolutionID[i4], &_buffer[*_lidx], _lidx);
		}
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementDL_Type_TimingAdvance) {
		_serNrDrbDecNR_MAC_CE_TimingAdvance_Type(_buffer, _size, _lidx, &p->TimingAdvance);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementDL_Type_SCellActDeact) {
		_serNrDrbDecNR_MAC_CE_SCellActDeact_Type(_buffer, _size, _lidx, &p->SCellActDeact);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementDL_Type_DuplicationActDeact) {
		for (size_t i4 = 0; i4 < 8; i4++) {
			NTOH_8(p->DuplicationActDeact[i4], &_buffer[*_lidx], _lidx);
		}
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementDL_Type_SP_ResourceSetActDeact) {
		_serNrDrbDecNR_MAC_CE_SP_ResourceSetActDeact_Type(_buffer, _size, _lidx, _mem, &p->SP_ResourceSetActDeact);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementDL_Type_CSI_TriggerStateSubselection) {
		_serNrDrbDecNR_MAC_CE_CSI_TriggerStateSubselection_Type(_buffer, _size, _lidx, _mem, &p->CSI_TriggerStateSubselection);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementDL_Type_TCI_StatesActDeact) {
		_serNrDrbDecNR_MAC_CE_TCI_StatesActDeact_Type(_buffer, _size, _lidx, _mem, &p->TCI_StatesActDeact);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementDL_Type_TCI_StateIndication) {
		_serNrDrbDecNR_MAC_CE_TCI_StateIndication_Type(_buffer, _size, _lidx, &p->TCI_StateIndication);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementDL_Type_SP_CSI_ReportingActDeact) {
		_serNrDrbDecNR_MAC_CE_SP_CSI_ReportingActDeact_Type(_buffer, _size, _lidx, &p->SP_CSI_ReportingActDeact);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementDL_Type_SP_SRS_ActDeact) {
		_serNrDrbDecNR_MAC_CE_SP_SRS_ActDeact_Type(_buffer, _size, _lidx, _mem, &p->SP_SRS_ActDeact);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementDL_Type_PUCCH_SpatialRelationActDeact) {
		_serNrDrbDecNR_MAC_CE_PUCCH_SpatialRelationActDeact_Type(_buffer, _size, _lidx, &p->PUCCH_SpatialRelationActDeact);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementDL_Type_SP_ZP_ResourceSetActDeact) {
		_serNrDrbDecNR_MAC_CE_SP_ZP_ResourceSetActDeact_Type(_buffer, _size, _lidx, &p->SP_ZP_ResourceSetActDeact);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementDL_Type_RecommendatdBitrate) {
		_serNrDrbDecNR_MAC_CE_RecommendedBitrate_Type(_buffer, _size, _lidx, &p->RecommendatdBitrate);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrDrbDecNR_MAC_ControlElementDL_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_MAC_ControlElementDL_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum NR_MAC_ControlElementDL_Type_Sel)_tmp;
	}
	_serNrDrbDecNR_MAC_ControlElementDL_Type_Value(_buffer, _size, _lidx, _mem, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_ControlElementDL_Type_NR_MAC_CE_SubPDU_DL_Type_ControlElement_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_MAC_ControlElementDL_Type_NR_MAC_CE_SubPDU_DL_Type_ControlElement_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	_serNrDrbDecNR_MAC_ControlElementDL_Type(_buffer, _size, _lidx, _mem, &p->v);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_SubPDU_DL_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_MAC_CE_SubPDU_DL_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecNR_MAC_PDU_SubHeader_Type(_buffer, _size, _lidx, _mem, &p->SubHeader);
	_serNrDrbDecNR_MAC_ControlElementDL_Type_NR_MAC_CE_SubPDU_DL_Type_ControlElement_Optional(_buffer, _size, _lidx, _mem, &p->ControlElement);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_SubPDU_DL_List_Type_NR_MAC_PDU_DL_Type_CE_SubPDUList_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_MAC_CE_SubPDU_DL_List_Type_NR_MAC_PDU_DL_Type_CE_SubPDUList_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	NTOH_32(p->v.d, &_buffer[*_lidx], _lidx);
	p->v.v = (struct NR_MAC_CE_SubPDU_DL_Type*)serMalloc(_mem, p->v.d * sizeof(struct NR_MAC_CE_SubPDU_DL_Type));
	for (size_t i3 = 0; i3 < p->v.d; i3++) {
		_serNrDrbDecNR_MAC_CE_SubPDU_DL_Type(_buffer, _size, _lidx, _mem, &p->v.v[i3]);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_SDU_SubPDU_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_MAC_SDU_SubPDU_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecNR_MAC_PDU_SubHeader_Type(_buffer, _size, _lidx, _mem, &p->SubHeader);
	NTOH_32(p->SDU.d, &_buffer[*_lidx], _lidx);
	p->SDU.v = (OCTET_STRING_ELEMENT*)serMalloc(_mem, p->SDU.d * sizeof(OCTET_STRING_ELEMENT));
	for (size_t i4 = 0; i4 < p->SDU.d; i4++) {
		NTOH_8(p->SDU.v[i4], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_DL_Type_SDU_SubPDUList_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_DL_Type_SDU_SubPDUList_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	NTOH_32(p->v.d, &_buffer[*_lidx], _lidx);
	p->v.v = (struct NR_MAC_SDU_SubPDU_Type*)serMalloc(_mem, p->v.d * sizeof(struct NR_MAC_SDU_SubPDU_Type));
	for (size_t i3 = 0; i3 < p->v.d; i3++) {
		_serNrDrbDecNR_MAC_SDU_SubPDU_Type(_buffer, _size, _lidx, _mem, &p->v.v[i3]);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_Padding_SubPDU_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_MAC_Padding_SubPDU_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecNR_MAC_PDU_SubHeader_Type(_buffer, _size, _lidx, _mem, &p->SubHeader);
	NTOH_32(p->Padding.d, &_buffer[*_lidx], _lidx);
	p->Padding.v = (OCTET_STRING_ELEMENT*)serMalloc(_mem, p->Padding.d * sizeof(OCTET_STRING_ELEMENT));
	for (size_t i3 = 0; i3 < p->Padding.d; i3++) {
		NTOH_8(p->Padding.v[i3], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_DL_Type_Padding_SubPDU_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_DL_Type_Padding_SubPDU_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	_serNrDrbDecNR_MAC_Padding_SubPDU_Type(_buffer, _size, _lidx, _mem, &p->v);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_PDU_DL_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_MAC_PDU_DL_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecNR_MAC_CE_SubPDU_DL_List_Type_NR_MAC_PDU_DL_Type_CE_SubPDUList_Optional(_buffer, _size, _lidx, _mem, &p->CE_SubPDUList);
	_serNrDrbDecNR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_DL_Type_SDU_SubPDUList_Optional(_buffer, _size, _lidx, _mem, &p->SDU_SubPDUList);
	_serNrDrbDecNR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_DL_Type_Padding_SubPDU_Optional(_buffer, _size, _lidx, _mem, &p->Padding_SubPDU);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_UL_Type_SDU_SubPDUList_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_UL_Type_SDU_SubPDUList_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	NTOH_32(p->v.d, &_buffer[*_lidx], _lidx);
	p->v.v = (struct NR_MAC_SDU_SubPDU_Type*)serMalloc(_mem, p->v.d * sizeof(struct NR_MAC_SDU_SubPDU_Type));
	for (size_t i3 = 0; i3 < p->v.d; i3++) {
		_serNrDrbDecNR_MAC_SDU_SubPDU_Type(_buffer, _size, _lidx, _mem, &p->v.v[i3]);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_ShortBSR_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_MAC_CE_ShortBSR_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 3; i4++) {
		NTOH_8(p->LCG[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 5; i4++) {
		NTOH_8(p->BufferSize[i4], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_LongBSR_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_MAC_CE_LongBSR_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 8; i4++) {
		NTOH_8(p->LCG_Presence[i4], &_buffer[*_lidx], _lidx);
	}
	NTOH_32(p->BufferSizeList.d, &_buffer[*_lidx], _lidx);
	p->BufferSizeList.v = (NR_MAC_LongBSR_BufferSize_Type*)serMalloc(_mem, p->BufferSizeList.d * sizeof(NR_MAC_LongBSR_BufferSize_Type));
	for (size_t i4 = 0; i4 < p->BufferSizeList.d; i4++) {
		for (size_t i5 = 0; i5 < 1; i5++) {
			NTOH_8(p->BufferSizeList.v[i4][i5], &_buffer[*_lidx], _lidx);
		}
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecB2_Type_NR_MAC_CE_PH_Record_Type_MPE_or_R_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct B2_Type_NR_MAC_CE_PH_Record_Type_MPE_or_R_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i4 = 0; i4 < 2; i4++) {
		NTOH_8(p->v[i4], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecB6_Type_NR_MAC_CE_PH_Record_Type_PCMaxc_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct B6_Type_NR_MAC_CE_PH_Record_Type_PCMaxc_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i4 = 0; i4 < 6; i4++) {
		NTOH_8(p->v[i4], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_SingleEntryPHR_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, NR_MAC_CE_SingleEntryPHR_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 1; i4++) {
		NTOH_8(p->P_Bit[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		NTOH_8(p->V_Bit[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 6; i4++) {
		NTOH_8(p->Value[i4], &_buffer[*_lidx], _lidx);
	}
	_serNrDrbDecB2_Type_NR_MAC_CE_PH_Record_Type_MPE_or_R_Optional(_buffer, _size, _lidx, &p->MPE_or_R);
	_serNrDrbDecB6_Type_NR_MAC_CE_PH_Record_Type_PCMaxc_Optional(_buffer, _size, _lidx, &p->PCMaxc);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_SCellFlags_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_MAC_CE_SCellFlags_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 8; i4++) {
		NTOH_8(p->SCellIndex7_1[i4], &_buffer[*_lidx], _lidx);
	}
	_serNrDrbDecB8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex15_8_Optional(_buffer, _size, _lidx, &p->SCellIndex15_8);
	_serNrDrbDecB8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex23_16_Optional(_buffer, _size, _lidx, &p->SCellIndex23_16);
	_serNrDrbDecB8_Type_NR_MAC_CE_SCellFlags_Type_SCellIndex31_24_Optional(_buffer, _size, _lidx, &p->SCellIndex31_24);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_PH_Record_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_MAC_CE_PH_Record_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i5 = 0; i5 < 1; i5++) {
		NTOH_8(p->P_Bit[i5], &_buffer[*_lidx], _lidx);
	}
	for (size_t i5 = 0; i5 < 1; i5++) {
		NTOH_8(p->V_Bit[i5], &_buffer[*_lidx], _lidx);
	}
	for (size_t i5 = 0; i5 < 6; i5++) {
		NTOH_8(p->Value[i5], &_buffer[*_lidx], _lidx);
	}
	_serNrDrbDecB2_Type_NR_MAC_CE_PH_Record_Type_MPE_or_R_Optional(_buffer, _size, _lidx, &p->MPE_or_R);
	_serNrDrbDecB6_Type_NR_MAC_CE_PH_Record_Type_PCMaxc_Optional(_buffer, _size, _lidx, &p->PCMaxc);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_MultiEntryPHR_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_MAC_CE_MultiEntryPHR_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecNR_MAC_CE_SCellFlags_Type(_buffer, _size, _lidx, &p->PHFieldPresentForSCell);
	NTOH_32(p->PH_Record.d, &_buffer[*_lidx], _lidx);
	p->PH_Record.v = (struct NR_MAC_CE_PH_Record_Type*)serMalloc(_mem, p->PH_Record.d * sizeof(struct NR_MAC_CE_PH_Record_Type));
	for (size_t i4 = 0; i4 < p->PH_Record.d; i4++) {
		_serNrDrbDecNR_MAC_CE_PH_Record_Type(_buffer, _size, _lidx, &p->PH_Record.v[i4]);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_ControlElementUL_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, union NR_MAC_ControlElementUL_Type_Value* p, enum NR_MAC_ControlElementUL_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_MAC_ControlElementUL_Type_ShortBSR) {
		_serNrDrbDecNR_MAC_CE_ShortBSR_Type(_buffer, _size, _lidx, &p->ShortBSR);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementUL_Type_LongBSR) {
		_serNrDrbDecNR_MAC_CE_LongBSR_Type(_buffer, _size, _lidx, _mem, &p->LongBSR);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementUL_Type_C_RNTI) {
		for (size_t i4 = 0; i4 < 16; i4++) {
			NTOH_8(p->C_RNTI[i4], &_buffer[*_lidx], _lidx);
		}
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementUL_Type_SingleEntryPHR) {
		_serNrDrbDecNR_MAC_CE_SingleEntryPHR_Type(_buffer, _size, _lidx, &p->SingleEntryPHR);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementUL_Type_MultiEntryPHR) {
		_serNrDrbDecNR_MAC_CE_MultiEntryPHR_Type(_buffer, _size, _lidx, _mem, &p->MultiEntryPHR);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_ControlElementUL_Type_RecommendedBitrate) {
		_serNrDrbDecNR_MAC_CE_RecommendedBitrate_Type(_buffer, _size, _lidx, &p->RecommendedBitrate);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrDrbDecNR_MAC_ControlElementUL_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_MAC_ControlElementUL_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum NR_MAC_ControlElementUL_Type_Sel)_tmp;
	}
	_serNrDrbDecNR_MAC_ControlElementUL_Type_Value(_buffer, _size, _lidx, _mem, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_ControlElementUL_Type_NR_MAC_CE_SubPDU_UL_Type_ControlElement_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_MAC_ControlElementUL_Type_NR_MAC_CE_SubPDU_UL_Type_ControlElement_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	_serNrDrbDecNR_MAC_ControlElementUL_Type(_buffer, _size, _lidx, _mem, &p->v);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_SubPDU_UL_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_MAC_CE_SubPDU_UL_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecNR_MAC_PDU_SubHeader_Type(_buffer, _size, _lidx, _mem, &p->SubHeader);
	_serNrDrbDecNR_MAC_ControlElementUL_Type_NR_MAC_CE_SubPDU_UL_Type_ControlElement_Optional(_buffer, _size, _lidx, _mem, &p->ControlElement);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_CE_SubPDU_UL_List_Type_NR_MAC_PDU_UL_Type_CE_SubPDUList_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_MAC_CE_SubPDU_UL_List_Type_NR_MAC_PDU_UL_Type_CE_SubPDUList_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	NTOH_32(p->v.d, &_buffer[*_lidx], _lidx);
	p->v.v = (struct NR_MAC_CE_SubPDU_UL_Type*)serMalloc(_mem, p->v.d * sizeof(struct NR_MAC_CE_SubPDU_UL_Type));
	for (size_t i3 = 0; i3 < p->v.d; i3++) {
		_serNrDrbDecNR_MAC_CE_SubPDU_UL_Type(_buffer, _size, _lidx, _mem, &p->v.v[i3]);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_UL_Type_Padding_SubPDU_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_UL_Type_Padding_SubPDU_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	_serNrDrbDecNR_MAC_Padding_SubPDU_Type(_buffer, _size, _lidx, _mem, &p->v);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_PDU_UL_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_MAC_PDU_UL_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecNR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_UL_Type_SDU_SubPDUList_Optional(_buffer, _size, _lidx, _mem, &p->SDU_SubPDUList);
	_serNrDrbDecNR_MAC_CE_SubPDU_UL_List_Type_NR_MAC_PDU_UL_Type_CE_SubPDUList_Optional(_buffer, _size, _lidx, _mem, &p->CE_SubPDUList);
	_serNrDrbDecNR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_UL_Type_Padding_SubPDU_Optional(_buffer, _size, _lidx, _mem, &p->Padding_SubPDU);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_MAC_PDU_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, union NR_MAC_PDU_Type_Value* p, enum NR_MAC_PDU_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_MAC_PDU_Type_DL) {
		_serNrDrbDecNR_MAC_PDU_DL_Type(_buffer, _size, _lidx, _mem, &p->DL);
		return SIDL_STATUS_OK;
	}
	if (d == NR_MAC_PDU_Type_UL) {
		_serNrDrbDecNR_MAC_PDU_UL_Type(_buffer, _size, _lidx, _mem, &p->UL);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrDrbDecNR_MAC_PDU_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_MAC_PDU_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum NR_MAC_PDU_Type_Sel)_tmp;
	}
	_serNrDrbDecNR_MAC_PDU_Type_Value(_buffer, _size, _lidx, _mem, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_UMD_HeaderNoSN_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_RLC_UMD_HeaderNoSN_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i3 = 0; i3 < 2; i3++) {
		NTOH_8(p->SegmentationInfo[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 6; i3++) {
		NTOH_8(p->Reserved[i3], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_UMD_PduNoSN_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_RLC_UMD_PduNoSN_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecNR_RLC_UMD_HeaderNoSN_Type(_buffer, _size, _lidx, &p->Header);
	NTOH_32(p->Data.d, &_buffer[*_lidx], _lidx);
	p->Data.v = (OCTET_STRING_ELEMENT*)serMalloc(_mem, p->Data.d * sizeof(OCTET_STRING_ELEMENT));
	for (size_t i3 = 0; i3 < p->Data.d; i3++) {
		NTOH_8(p->Data.v[i3], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_SegmentOffset_Type_NR_RLC_UMD_HeaderSN6Bit_Type_SegmentOffset_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_RLC_SegmentOffset_Type_NR_RLC_UMD_HeaderSN6Bit_Type_SegmentOffset_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i3 = 0; i3 < 16; i3++) {
		NTOH_8(p->v[i3], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_UMD_HeaderSN6Bit_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_RLC_UMD_HeaderSN6Bit_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i3 = 0; i3 < 2; i3++) {
		NTOH_8(p->SegmentationInfo[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 6; i3++) {
		NTOH_8(p->SequenceNumber[i3], &_buffer[*_lidx], _lidx);
	}
	_serNrDrbDecNR_RLC_SegmentOffset_Type_NR_RLC_UMD_HeaderSN6Bit_Type_SegmentOffset_Optional(_buffer, _size, _lidx, &p->SegmentOffset);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_UMD_PduSN6Bit_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_RLC_UMD_PduSN6Bit_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecNR_RLC_UMD_HeaderSN6Bit_Type(_buffer, _size, _lidx, &p->Header);
	NTOH_32(p->Data.d, &_buffer[*_lidx], _lidx);
	p->Data.v = (OCTET_STRING_ELEMENT*)serMalloc(_mem, p->Data.d * sizeof(OCTET_STRING_ELEMENT));
	for (size_t i3 = 0; i3 < p->Data.d; i3++) {
		NTOH_8(p->Data.v[i3], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_SegmentOffset_Type_NR_RLC_UMD_HeaderSN12Bit_Type_SegmentOffset_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_RLC_SegmentOffset_Type_NR_RLC_UMD_HeaderSN12Bit_Type_SegmentOffset_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i3 = 0; i3 < 16; i3++) {
		NTOH_8(p->v[i3], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_UMD_HeaderSN12Bit_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_RLC_UMD_HeaderSN12Bit_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i3 = 0; i3 < 2; i3++) {
		NTOH_8(p->SegmentationInfo[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 2; i3++) {
		NTOH_8(p->Reserved[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 12; i3++) {
		NTOH_8(p->SequenceNumber[i3], &_buffer[*_lidx], _lidx);
	}
	_serNrDrbDecNR_RLC_SegmentOffset_Type_NR_RLC_UMD_HeaderSN12Bit_Type_SegmentOffset_Optional(_buffer, _size, _lidx, &p->SegmentOffset);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_UMD_PduSN12Bit_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_RLC_UMD_PduSN12Bit_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecNR_RLC_UMD_HeaderSN12Bit_Type(_buffer, _size, _lidx, &p->Header);
	NTOH_32(p->Data.d, &_buffer[*_lidx], _lidx);
	p->Data.v = (OCTET_STRING_ELEMENT*)serMalloc(_mem, p->Data.d * sizeof(OCTET_STRING_ELEMENT));
	for (size_t i3 = 0; i3 < p->Data.d; i3++) {
		NTOH_8(p->Data.v[i3], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_UMD_PDU_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, union NR_RLC_UMD_PDU_Type_Value* p, enum NR_RLC_UMD_PDU_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_RLC_UMD_PDU_Type_NoSN) {
		_serNrDrbDecNR_RLC_UMD_PduNoSN_Type(_buffer, _size, _lidx, _mem, &p->NoSN);
		return SIDL_STATUS_OK;
	}
	if (d == NR_RLC_UMD_PDU_Type_SN6Bit) {
		_serNrDrbDecNR_RLC_UMD_PduSN6Bit_Type(_buffer, _size, _lidx, _mem, &p->SN6Bit);
		return SIDL_STATUS_OK;
	}
	if (d == NR_RLC_UMD_PDU_Type_SN12Bit) {
		_serNrDrbDecNR_RLC_UMD_PduSN12Bit_Type(_buffer, _size, _lidx, _mem, &p->SN12Bit);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrDrbDecNR_RLC_UMD_PDU_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_RLC_UMD_PDU_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum NR_RLC_UMD_PDU_Type_Sel)_tmp;
	}
	_serNrDrbDecNR_RLC_UMD_PDU_Type_Value(_buffer, _size, _lidx, _mem, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_SegmentOffset_Type_NR_RLC_AMD_HeaderSN12Bit_Type_SegmentOffset_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_RLC_SegmentOffset_Type_NR_RLC_AMD_HeaderSN12Bit_Type_SegmentOffset_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i3 = 0; i3 < 16; i3++) {
		NTOH_8(p->v[i3], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_AMD_HeaderSN12Bit_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_RLC_AMD_HeaderSN12Bit_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i3 = 0; i3 < 1; i3++) {
		NTOH_8(p->D_C[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 1; i3++) {
		NTOH_8(p->Poll[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 2; i3++) {
		NTOH_8(p->SegmentationInfo[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 12; i3++) {
		NTOH_8(p->SequenceNumber[i3], &_buffer[*_lidx], _lidx);
	}
	_serNrDrbDecNR_RLC_SegmentOffset_Type_NR_RLC_AMD_HeaderSN12Bit_Type_SegmentOffset_Optional(_buffer, _size, _lidx, &p->SegmentOffset);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_AMD_PduSN12Bit_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_RLC_AMD_PduSN12Bit_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecNR_RLC_AMD_HeaderSN12Bit_Type(_buffer, _size, _lidx, &p->Header);
	NTOH_32(p->Data.d, &_buffer[*_lidx], _lidx);
	p->Data.v = (OCTET_STRING_ELEMENT*)serMalloc(_mem, p->Data.d * sizeof(OCTET_STRING_ELEMENT));
	for (size_t i3 = 0; i3 < p->Data.d; i3++) {
		NTOH_8(p->Data.v[i3], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_SegmentOffset_Type_NR_RLC_AMD_HeaderSN18Bit_Type_SegmentOffset_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_RLC_SegmentOffset_Type_NR_RLC_AMD_HeaderSN18Bit_Type_SegmentOffset_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i3 = 0; i3 < 16; i3++) {
		NTOH_8(p->v[i3], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_AMD_HeaderSN18Bit_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_RLC_AMD_HeaderSN18Bit_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i3 = 0; i3 < 1; i3++) {
		NTOH_8(p->D_C[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 1; i3++) {
		NTOH_8(p->Poll[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 2; i3++) {
		NTOH_8(p->SegmentationInfo[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 2; i3++) {
		NTOH_8(p->Reserved[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 18; i3++) {
		NTOH_8(p->SequenceNumber[i3], &_buffer[*_lidx], _lidx);
	}
	_serNrDrbDecNR_RLC_SegmentOffset_Type_NR_RLC_AMD_HeaderSN18Bit_Type_SegmentOffset_Optional(_buffer, _size, _lidx, &p->SegmentOffset);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_AMD_PduSN18Bit_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_RLC_AMD_PduSN18Bit_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecNR_RLC_AMD_HeaderSN18Bit_Type(_buffer, _size, _lidx, &p->Header);
	NTOH_32(p->Data.d, &_buffer[*_lidx], _lidx);
	p->Data.v = (OCTET_STRING_ELEMENT*)serMalloc(_mem, p->Data.d * sizeof(OCTET_STRING_ELEMENT));
	for (size_t i3 = 0; i3 < p->Data.d; i3++) {
		NTOH_8(p->Data.v[i3], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_AMD_PDU_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, union NR_RLC_AMD_PDU_Type_Value* p, enum NR_RLC_AMD_PDU_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_RLC_AMD_PDU_Type_SN12Bit) {
		_serNrDrbDecNR_RLC_AMD_PduSN12Bit_Type(_buffer, _size, _lidx, _mem, &p->SN12Bit);
		return SIDL_STATUS_OK;
	}
	if (d == NR_RLC_AMD_PDU_Type_SN18Bit) {
		_serNrDrbDecNR_RLC_AMD_PduSN18Bit_Type(_buffer, _size, _lidx, _mem, &p->SN18Bit);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrDrbDecNR_RLC_AMD_PDU_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_RLC_AMD_PDU_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum NR_RLC_AMD_PDU_Type_Sel)_tmp;
	}
	_serNrDrbDecNR_RLC_AMD_PDU_Type_Value(_buffer, _size, _lidx, _mem, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN12Bit_Type_SOstart_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN12Bit_Type_SOstart_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i4 = 0; i4 < 16; i4++) {
		NTOH_8(p->v[i4], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN12Bit_Type_SOstop_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN12Bit_Type_SOstop_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i4 = 0; i4 < 16; i4++) {
		NTOH_8(p->v[i4], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecB8_Type_NR_RLC_Status_NackSN12Bit_Type_NACKrange_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct B8_Type_NR_RLC_Status_NackSN12Bit_Type_NACKrange_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i4 = 0; i4 < 8; i4++) {
		NTOH_8(p->v[i4], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_Status_NackSN12Bit_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_RLC_Status_NackSN12Bit_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 12; i4++) {
		NTOH_8(p->SequenceNumberNACK[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		NTOH_8(p->E1[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		NTOH_8(p->E2[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		NTOH_8(p->E3[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		NTOH_8(p->Reserved[i4], &_buffer[*_lidx], _lidx);
	}
	_serNrDrbDecNR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN12Bit_Type_SOstart_Optional(_buffer, _size, _lidx, &p->SOstart);
	_serNrDrbDecNR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN12Bit_Type_SOstop_Optional(_buffer, _size, _lidx, &p->SOstop);
	_serNrDrbDecB8_Type_NR_RLC_Status_NackSN12Bit_Type_NACKrange_Optional(_buffer, _size, _lidx, &p->NACKrange);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_Status_NackListSN12Bit_Type_NackList_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_RLC_Status_NackListSN12Bit_Type_NackList_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	NTOH_32(p->v.d, &_buffer[*_lidx], _lidx);
	p->v.v = (struct NR_RLC_Status_NackSN12Bit_Type*)serMalloc(_mem, p->v.d * sizeof(struct NR_RLC_Status_NackSN12Bit_Type));
	for (size_t i3 = 0; i3 < p->v.d; i3++) {
		_serNrDrbDecNR_RLC_Status_NackSN12Bit_Type(_buffer, _size, _lidx, &p->v.v[i3]);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_StatusPduSN12Bit_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_RLC_StatusPduSN12Bit_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i3 = 0; i3 < 1; i3++) {
		NTOH_8(p->D_C[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 3; i3++) {
		NTOH_8(p->CPT[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 12; i3++) {
		NTOH_8(p->SequenceNumberACK[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 1; i3++) {
		NTOH_8(p->E1[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 7; i3++) {
		NTOH_8(p->Reserved[i3], &_buffer[*_lidx], _lidx);
	}
	_serNrDrbDecNR_RLC_Status_NackListSN12Bit_Type_NackList_Optional(_buffer, _size, _lidx, _mem, &p->NackList);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN18Bit_Type_SOstart_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN18Bit_Type_SOstart_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i4 = 0; i4 < 16; i4++) {
		NTOH_8(p->v[i4], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN18Bit_Type_SOstop_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN18Bit_Type_SOstop_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i4 = 0; i4 < 16; i4++) {
		NTOH_8(p->v[i4], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecB8_Type_NR_RLC_Status_NackSN18Bit_Type_NACKrange_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct B8_Type_NR_RLC_Status_NackSN18Bit_Type_NACKrange_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i4 = 0; i4 < 8; i4++) {
		NTOH_8(p->v[i4], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_Status_NackSN18Bit_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_RLC_Status_NackSN18Bit_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i4 = 0; i4 < 18; i4++) {
		NTOH_8(p->SequenceNumberNACK[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		NTOH_8(p->E1[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		NTOH_8(p->E2[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 1; i4++) {
		NTOH_8(p->E3[i4], &_buffer[*_lidx], _lidx);
	}
	for (size_t i4 = 0; i4 < 3; i4++) {
		NTOH_8(p->Reserved[i4], &_buffer[*_lidx], _lidx);
	}
	_serNrDrbDecNR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN18Bit_Type_SOstart_Optional(_buffer, _size, _lidx, &p->SOstart);
	_serNrDrbDecNR_RLC_SegmentOffset_Type_NR_RLC_Status_NackSN18Bit_Type_SOstop_Optional(_buffer, _size, _lidx, &p->SOstop);
	_serNrDrbDecB8_Type_NR_RLC_Status_NackSN18Bit_Type_NACKrange_Optional(_buffer, _size, _lidx, &p->NACKrange);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_Status_NackListSN18Bit_Type_NackList_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_RLC_Status_NackListSN18Bit_Type_NackList_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	NTOH_32(p->v.d, &_buffer[*_lidx], _lidx);
	p->v.v = (struct NR_RLC_Status_NackSN18Bit_Type*)serMalloc(_mem, p->v.d * sizeof(struct NR_RLC_Status_NackSN18Bit_Type));
	for (size_t i3 = 0; i3 < p->v.d; i3++) {
		_serNrDrbDecNR_RLC_Status_NackSN18Bit_Type(_buffer, _size, _lidx, &p->v.v[i3]);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_StatusPduSN18Bit_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_RLC_StatusPduSN18Bit_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i3 = 0; i3 < 1; i3++) {
		NTOH_8(p->D_C[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 3; i3++) {
		NTOH_8(p->CPT[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 18; i3++) {
		NTOH_8(p->SequenceNumberACK[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 1; i3++) {
		NTOH_8(p->E1[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 1; i3++) {
		NTOH_8(p->Reserved[i3], &_buffer[*_lidx], _lidx);
	}
	_serNrDrbDecNR_RLC_Status_NackListSN18Bit_Type_NackList_Optional(_buffer, _size, _lidx, _mem, &p->NackList);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_AM_StatusPDU_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, union NR_RLC_AM_StatusPDU_Type_Value* p, enum NR_RLC_AM_StatusPDU_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_RLC_AM_StatusPDU_Type_SN12Bit) {
		_serNrDrbDecNR_RLC_StatusPduSN12Bit_Type(_buffer, _size, _lidx, _mem, &p->SN12Bit);
		return SIDL_STATUS_OK;
	}
	if (d == NR_RLC_AM_StatusPDU_Type_SN18Bit) {
		_serNrDrbDecNR_RLC_StatusPduSN18Bit_Type(_buffer, _size, _lidx, _mem, &p->SN18Bit);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrDrbDecNR_RLC_AM_StatusPDU_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_RLC_AM_StatusPDU_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum NR_RLC_AM_StatusPDU_Type_Sel)_tmp;
	}
	_serNrDrbDecNR_RLC_AM_StatusPDU_Type_Value(_buffer, _size, _lidx, _mem, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RLC_PDU_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, union NR_RLC_PDU_Type_Value* p, enum NR_RLC_PDU_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_RLC_PDU_Type_TMD) {
		NTOH_32(p->TMD.d, &_buffer[*_lidx], _lidx);
		p->TMD.v = (OCTET_STRING_ELEMENT*)serMalloc(_mem, p->TMD.d * sizeof(OCTET_STRING_ELEMENT));
		for (size_t i3 = 0; i3 < p->TMD.d; i3++) {
			NTOH_8(p->TMD.v[i3], &_buffer[*_lidx], _lidx);
		}
		return SIDL_STATUS_OK;
	}
	if (d == NR_RLC_PDU_Type_UMD) {
		_serNrDrbDecNR_RLC_UMD_PDU_Type(_buffer, _size, _lidx, _mem, &p->UMD);
		return SIDL_STATUS_OK;
	}
	if (d == NR_RLC_PDU_Type_AMD) {
		_serNrDrbDecNR_RLC_AMD_PDU_Type(_buffer, _size, _lidx, _mem, &p->AMD);
		return SIDL_STATUS_OK;
	}
	if (d == NR_RLC_PDU_Type_Status) {
		_serNrDrbDecNR_RLC_AM_StatusPDU_Type(_buffer, _size, _lidx, _mem, &p->Status);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrDrbDecNR_RLC_PDU_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_RLC_PDU_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum NR_RLC_PDU_Type_Sel)_tmp;
	}
	_serNrDrbDecNR_RLC_PDU_Type_Value(_buffer, _size, _lidx, _mem, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecB32_Type_NR_PDCP_DataPduSN12Bits_Type_MAC_I_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct B32_Type_NR_PDCP_DataPduSN12Bits_Type_MAC_I_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i3 = 0; i3 < 32; i3++) {
		NTOH_8(p->v[i3], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_PDCP_DataPduSN12Bits_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_PDCP_DataPduSN12Bits_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i3 = 0; i3 < 1; i3++) {
		NTOH_8(p->D_C[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 3; i3++) {
		NTOH_8(p->Reserved[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 12; i3++) {
		NTOH_8(p->SequenceNumber[i3], &_buffer[*_lidx], _lidx);
	}
	NTOH_32(p->SDU.d, &_buffer[*_lidx], _lidx);
	p->SDU.v = (OCTET_STRING_ELEMENT*)serMalloc(_mem, p->SDU.d * sizeof(OCTET_STRING_ELEMENT));
	for (size_t i3 = 0; i3 < p->SDU.d; i3++) {
		NTOH_8(p->SDU.v[i3], &_buffer[*_lidx], _lidx);
	}
	_serNrDrbDecB32_Type_NR_PDCP_DataPduSN12Bits_Type_MAC_I_Optional(_buffer, _size, _lidx, &p->MAC_I);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecB32_Type_NR_PDCP_DataPduSN18Bits_Type_MAC_I_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct B32_Type_NR_PDCP_DataPduSN18Bits_Type_MAC_I_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	for (size_t i3 = 0; i3 < 32; i3++) {
		NTOH_8(p->v[i3], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_PDCP_DataPduSN18Bits_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_PDCP_DataPduSN18Bits_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i3 = 0; i3 < 1; i3++) {
		NTOH_8(p->D_C[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 5; i3++) {
		NTOH_8(p->Reserved[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 18; i3++) {
		NTOH_8(p->SequenceNumber[i3], &_buffer[*_lidx], _lidx);
	}
	NTOH_32(p->SDU.d, &_buffer[*_lidx], _lidx);
	p->SDU.v = (OCTET_STRING_ELEMENT*)serMalloc(_mem, p->SDU.d * sizeof(OCTET_STRING_ELEMENT));
	for (size_t i3 = 0; i3 < p->SDU.d; i3++) {
		NTOH_8(p->SDU.v[i3], &_buffer[*_lidx], _lidx);
	}
	_serNrDrbDecB32_Type_NR_PDCP_DataPduSN18Bits_Type_MAC_I_Optional(_buffer, _size, _lidx, &p->MAC_I);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecOCTET_STRING_NR_PDCP_CtrlPduStatus_Type_Bitmap_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct OCTET_STRING_NR_PDCP_CtrlPduStatus_Type_Bitmap_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	NTOH_32(p->v.d, &_buffer[*_lidx], _lidx);
	p->v.v = (OCTET_STRING_ELEMENT*)serMalloc(_mem, p->v.d * sizeof(OCTET_STRING_ELEMENT));
	for (size_t i3 = 0; i3 < p->v.d; i3++) {
		NTOH_8(p->v.v[i3], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_PDCP_CtrlPduStatus_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_PDCP_CtrlPduStatus_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i3 = 0; i3 < 1; i3++) {
		NTOH_8(p->D_C[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 3; i3++) {
		NTOH_8(p->PDU_Type[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 4; i3++) {
		NTOH_8(p->Reserved[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 32; i3++) {
		NTOH_8(p->FirstMissingCount[i3], &_buffer[*_lidx], _lidx);
	}
	_serNrDrbDecOCTET_STRING_NR_PDCP_CtrlPduStatus_Type_Bitmap_Optional(_buffer, _size, _lidx, _mem, &p->Bitmap);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_PDCP_CtrlPduRohcFeedback_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_PDCP_CtrlPduRohcFeedback_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i3 = 0; i3 < 1; i3++) {
		NTOH_8(p->D_C[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 3; i3++) {
		NTOH_8(p->PDU_Type[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 4; i3++) {
		NTOH_8(p->Reserved[i3], &_buffer[*_lidx], _lidx);
	}
	NTOH_32(p->RohcFeedback.d, &_buffer[*_lidx], _lidx);
	p->RohcFeedback.v = (OCTET_STRING_ELEMENT*)serMalloc(_mem, p->RohcFeedback.d * sizeof(OCTET_STRING_ELEMENT));
	for (size_t i3 = 0; i3 < p->RohcFeedback.d; i3++) {
		NTOH_8(p->RohcFeedback.v[i3], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_PDCP_PDU_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, union NR_PDCP_PDU_Type_Value* p, enum NR_PDCP_PDU_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_PDCP_PDU_Type_DataPduSN12Bits) {
		_serNrDrbDecNR_PDCP_DataPduSN12Bits_Type(_buffer, _size, _lidx, _mem, &p->DataPduSN12Bits);
		return SIDL_STATUS_OK;
	}
	if (d == NR_PDCP_PDU_Type_DataPduSN18Bits) {
		_serNrDrbDecNR_PDCP_DataPduSN18Bits_Type(_buffer, _size, _lidx, _mem, &p->DataPduSN18Bits);
		return SIDL_STATUS_OK;
	}
	if (d == NR_PDCP_PDU_Type_CtrlPduStatus) {
		_serNrDrbDecNR_PDCP_CtrlPduStatus_Type(_buffer, _size, _lidx, _mem, &p->CtrlPduStatus);
		return SIDL_STATUS_OK;
	}
	if (d == NR_PDCP_PDU_Type_CtrlPduRohcFeedback) {
		_serNrDrbDecNR_PDCP_CtrlPduRohcFeedback_Type(_buffer, _size, _lidx, _mem, &p->CtrlPduRohcFeedback);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrDrbDecNR_PDCP_PDU_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_PDCP_PDU_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum NR_PDCP_PDU_Type_Sel)_tmp;
	}
	_serNrDrbDecNR_PDCP_PDU_Type_Value(_buffer, _size, _lidx, _mem, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecSDAP_DL_PduHeader_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct SDAP_DL_PduHeader_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i3 = 0; i3 < 1; i3++) {
		NTOH_8(p->RDI[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 1; i3++) {
		NTOH_8(p->RQI[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 6; i3++) {
		NTOH_8(p->QFI[i3], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecSDAP_DL_PduHeader_Type_SDAP_PDU_DL_Type_Header_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct SDAP_DL_PduHeader_Type_SDAP_PDU_DL_Type_Header_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	_serNrDrbDecSDAP_DL_PduHeader_Type(_buffer, _size, _lidx, &p->v);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecSDAP_PDU_DL_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct SDAP_PDU_DL_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecSDAP_DL_PduHeader_Type_SDAP_PDU_DL_Type_Header_Optional(_buffer, _size, _lidx, &p->Header);
	NTOH_32(p->Data.d, &_buffer[*_lidx], _lidx);
	p->Data.v = (OCTET_STRING_ELEMENT*)serMalloc(_mem, p->Data.d * sizeof(OCTET_STRING_ELEMENT));
	for (size_t i3 = 0; i3 < p->Data.d; i3++) {
		NTOH_8(p->Data.v[i3], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecSDAP_UL_PduHeader_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct SDAP_UL_PduHeader_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i3 = 0; i3 < 1; i3++) {
		NTOH_8(p->DC[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 1; i3++) {
		NTOH_8(p->R[i3], &_buffer[*_lidx], _lidx);
	}
	for (size_t i3 = 0; i3 < 6; i3++) {
		NTOH_8(p->QFI[i3], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecSDAP_UL_PduHeader_Type_SDAP_PDU_UL_Type_Header_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct SDAP_UL_PduHeader_Type_SDAP_PDU_UL_Type_Header_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	_serNrDrbDecSDAP_UL_PduHeader_Type(_buffer, _size, _lidx, &p->v);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecSDAP_SDU_Type_SDAP_PDU_UL_Type_Data_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct SDAP_SDU_Type_SDAP_PDU_UL_Type_Data_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	NTOH_32(p->v.d, &_buffer[*_lidx], _lidx);
	p->v.v = (OCTET_STRING_ELEMENT*)serMalloc(_mem, p->v.d * sizeof(OCTET_STRING_ELEMENT));
	for (size_t i3 = 0; i3 < p->v.d; i3++) {
		NTOH_8(p->v.v[i3], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecSDAP_PDU_UL_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct SDAP_PDU_UL_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecSDAP_UL_PduHeader_Type_SDAP_PDU_UL_Type_Header_Optional(_buffer, _size, _lidx, &p->Header);
	_serNrDrbDecSDAP_SDU_Type_SDAP_PDU_UL_Type_Data_Optional(_buffer, _size, _lidx, _mem, &p->Data);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecSDAP_PDU_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, union SDAP_PDU_Type_Value* p, enum SDAP_PDU_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == SDAP_PDU_Type_DL) {
		_serNrDrbDecSDAP_PDU_DL_Type(_buffer, _size, _lidx, _mem, &p->DL);
		return SIDL_STATUS_OK;
	}
	if (d == SDAP_PDU_Type_UL) {
		_serNrDrbDecSDAP_PDU_UL_Type(_buffer, _size, _lidx, _mem, &p->UL);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrDrbDecSDAP_PDU_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct SDAP_PDU_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum SDAP_PDU_Type_Sel)_tmp;
	}
	_serNrDrbDecSDAP_PDU_Type_Value(_buffer, _size, _lidx, _mem, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_L2DataList_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, union NR_L2DataList_Type_Value* p, enum NR_L2DataList_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == NR_L2DataList_Type_MacPdu) {
		NTOH_32(p->MacPdu.d, &_buffer[*_lidx], _lidx);
		p->MacPdu.v = (struct NR_MAC_PDU_Type*)serMalloc(_mem, p->MacPdu.d * sizeof(struct NR_MAC_PDU_Type));
		for (size_t i2 = 0; i2 < p->MacPdu.d; i2++) {
			_serNrDrbDecNR_MAC_PDU_Type(_buffer, _size, _lidx, _mem, &p->MacPdu.v[i2]);
		}
		return SIDL_STATUS_OK;
	}
	if (d == NR_L2DataList_Type_RlcPdu) {
		NTOH_32(p->RlcPdu.d, &_buffer[*_lidx], _lidx);
		p->RlcPdu.v = (struct NR_RLC_PDU_Type*)serMalloc(_mem, p->RlcPdu.d * sizeof(struct NR_RLC_PDU_Type));
		for (size_t i2 = 0; i2 < p->RlcPdu.d; i2++) {
			_serNrDrbDecNR_RLC_PDU_Type(_buffer, _size, _lidx, _mem, &p->RlcPdu.v[i2]);
		}
		return SIDL_STATUS_OK;
	}
	if (d == NR_L2DataList_Type_RlcSdu) {
		NTOH_32(p->RlcSdu.d, &_buffer[*_lidx], _lidx);
		p->RlcSdu.v = (NR_RLC_SDU_Type*)serMalloc(_mem, p->RlcSdu.d * sizeof(NR_RLC_SDU_Type));
		for (size_t i2 = 0; i2 < p->RlcSdu.d; i2++) {
			NTOH_32(p->RlcSdu.v[i2].d, &_buffer[*_lidx], _lidx);
			p->RlcSdu.v[i2].v = (OCTET_STRING_ELEMENT*)serMalloc(_mem, p->RlcSdu.v[i2].d * sizeof(OCTET_STRING_ELEMENT));
			for (size_t i3 = 0; i3 < p->RlcSdu.v[i2].d; i3++) {
				NTOH_8(p->RlcSdu.v[i2].v[i3], &_buffer[*_lidx], _lidx);
			}
		}
		return SIDL_STATUS_OK;
	}
	if (d == NR_L2DataList_Type_PdcpPdu) {
		NTOH_32(p->PdcpPdu.d, &_buffer[*_lidx], _lidx);
		p->PdcpPdu.v = (struct NR_PDCP_PDU_Type*)serMalloc(_mem, p->PdcpPdu.d * sizeof(struct NR_PDCP_PDU_Type));
		for (size_t i2 = 0; i2 < p->PdcpPdu.d; i2++) {
			_serNrDrbDecNR_PDCP_PDU_Type(_buffer, _size, _lidx, _mem, &p->PdcpPdu.v[i2]);
		}
		return SIDL_STATUS_OK;
	}
	if (d == NR_L2DataList_Type_PdcpSdu) {
		NTOH_32(p->PdcpSdu.d, &_buffer[*_lidx], _lidx);
		p->PdcpSdu.v = (NR_PDCP_SDU_Type*)serMalloc(_mem, p->PdcpSdu.d * sizeof(NR_PDCP_SDU_Type));
		for (size_t i2 = 0; i2 < p->PdcpSdu.d; i2++) {
			NTOH_32(p->PdcpSdu.v[i2].d, &_buffer[*_lidx], _lidx);
			p->PdcpSdu.v[i2].v = (OCTET_STRING_ELEMENT*)serMalloc(_mem, p->PdcpSdu.v[i2].d * sizeof(OCTET_STRING_ELEMENT));
			for (size_t i3 = 0; i3 < p->PdcpSdu.v[i2].d; i3++) {
				NTOH_8(p->PdcpSdu.v[i2].v[i3], &_buffer[*_lidx], _lidx);
			}
		}
		return SIDL_STATUS_OK;
	}
	if (d == NR_L2DataList_Type_SdapPdu) {
		NTOH_32(p->SdapPdu.d, &_buffer[*_lidx], _lidx);
		p->SdapPdu.v = (struct SDAP_PDU_Type*)serMalloc(_mem, p->SdapPdu.d * sizeof(struct SDAP_PDU_Type));
		for (size_t i2 = 0; i2 < p->SdapPdu.d; i2++) {
			_serNrDrbDecSDAP_PDU_Type(_buffer, _size, _lidx, _mem, &p->SdapPdu.v[i2]);
		}
		return SIDL_STATUS_OK;
	}
	if (d == NR_L2DataList_Type_SdapSdu) {
		NTOH_32(p->SdapSdu.d, &_buffer[*_lidx], _lidx);
		p->SdapSdu.v = (SDAP_SDU_Type*)serMalloc(_mem, p->SdapSdu.d * sizeof(SDAP_SDU_Type));
		for (size_t i2 = 0; i2 < p->SdapSdu.d; i2++) {
			NTOH_32(p->SdapSdu.v[i2].d, &_buffer[*_lidx], _lidx);
			p->SdapSdu.v[i2].v = (OCTET_STRING_ELEMENT*)serMalloc(_mem, p->SdapSdu.v[i2].d * sizeof(OCTET_STRING_ELEMENT));
			for (size_t i3 = 0; i3 < p->SdapSdu.v[i2].d; i3++) {
				NTOH_8(p->SdapSdu.v[i2].v[i3], &_buffer[*_lidx], _lidx);
			}
		}
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrDrbDecNR_L2DataList_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_L2DataList_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum NR_L2DataList_Type_Sel)_tmp;
	}
	_serNrDrbDecNR_L2DataList_Type_Value(_buffer, _size, _lidx, _mem, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_DRB_DataPerSlot_DL_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_DRB_DataPerSlot_DL_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_32(p->SlotOffset, &_buffer[*_lidx], _lidx);
	_serNrDrbDecNR_HarqProcessAssignment_Type_HarqProcess_Optional(_buffer, _size, _lidx, &p->HarqProcess);
	_serNrDrbDecNR_L2DataList_Type(_buffer, _size, _lidx, _mem, &p->PduSduList);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_L2Data_Request_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_L2Data_Request_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_32(p->SlotDataList.d, &_buffer[*_lidx], _lidx);
	p->SlotDataList.v = (struct NR_DRB_DataPerSlot_DL_Type*)serMalloc(_mem, p->SlotDataList.d * sizeof(struct NR_DRB_DataPerSlot_DL_Type));
	for (size_t i1 = 0; i1 < p->SlotDataList.d; i1++) {
		_serNrDrbDecNR_DRB_DataPerSlot_DL_Type(_buffer, _size, _lidx, _mem, &p->SlotDataList.v[i1]);
	}

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNull_Type_NR_DRB_COMMON_REQ_SuppressPdcchForC_RNTI_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct Null_Type_NR_DRB_COMMON_REQ_SuppressPdcchForC_RNTI_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	NTOH_8(p->v, &_buffer[*_lidx], _lidx);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_DRB_COMMON_REQ(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_DRB_COMMON_REQ* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecNR_ReqAspCommonPart_Type(_buffer, _size, _lidx, &p->Common);
	_serNrDrbDecNR_L2Data_Request_Type(_buffer, _size, _lidx, _mem, &p->U_Plane);
	_serNrDrbDecNull_Type_NR_DRB_COMMON_REQ_SuppressPdcchForC_RNTI_Optional(_buffer, _size, _lidx, &p->SuppressPdcchForC_RNTI);

	return SIDL_STATUS_OK;
}

int serNrDrbProcessFromSSDecSrv(const unsigned char* _buffer, size_t _size, unsigned char* _arena, size_t _aSize, struct NR_DRB_COMMON_REQ** FromSS)
{
	(void)_size; // TODO: generate boundaries checking

	serMem_t _mem = serMemInit(_arena, _aSize);

	size_t __lidx = 0;
	size_t* _lidx = &__lidx;

	*FromSS = (struct NR_DRB_COMMON_REQ*)serMalloc(_mem, sizeof(struct NR_DRB_COMMON_REQ));
	_serNrDrbDecNR_DRB_COMMON_REQ(_buffer, _size, _lidx, _mem, *FromSS);

	return SIDL_STATUS_OK;
}

static void _serNrDrbFreeBIT_STRING_NR_MAC_PDU_SubHeader_Type_ELCID_Optional(struct BIT_STRING_NR_MAC_PDU_SubHeader_Type_ELCID_Optional* p)
{
	if (!p->d) return;
	if (p->v.v) {
		serFree(p->v.v);
	}
}

static void _serNrDrbFreeBIT_STRING_NR_MAC_PDU_SubHeader_Type_Length_Optional(struct BIT_STRING_NR_MAC_PDU_SubHeader_Type_Length_Optional* p)
{
	if (!p->d) return;
	if (p->v.v) {
		serFree(p->v.v);
	}
}

static void _serNrDrbFreeNR_MAC_PDU_SubHeader_Type(struct NR_MAC_PDU_SubHeader_Type* p)
{
	_serNrDrbFreeBIT_STRING_NR_MAC_PDU_SubHeader_Type_ELCID_Optional(&p->ELCID);
	_serNrDrbFreeBIT_STRING_NR_MAC_PDU_SubHeader_Type_Length_Optional(&p->Length);
}

static void _serNrDrbFreeNR_MAC_CE_SP_ResourceSetActDeact_Type(struct NR_MAC_CE_SP_ResourceSetActDeact_Type* p)
{
	if (p->IdList.v) {
		for (size_t i4 = 0; i4 < p->IdList.d; i4++) {
		}
		serFree(p->IdList.v);
	}
}

static void _serNrDrbFreeNR_MAC_CE_CSI_TriggerStateSubselection_Type(struct NR_MAC_CE_CSI_TriggerStateSubselection_Type* p)
{
	if (p->Selection.v) {
		serFree(p->Selection.v);
	}
}

static void _serNrDrbFreeNR_MAC_CE_TCI_StatesActDeact_Type(struct NR_MAC_CE_TCI_StatesActDeact_Type* p)
{
	if (p->Status.v) {
		serFree(p->Status.v);
	}
}

static void _serNrDrbFreeNR_MAC_CE_SP_SRS_ActDeact_Type(struct NR_MAC_CE_SP_SRS_ActDeact_Type* p)
{
	if (p->ResourceIdList.v) {
		for (size_t i4 = 0; i4 < p->ResourceIdList.d; i4++) {
		}
		serFree(p->ResourceIdList.v);
	}
	if (p->ResourceInfoList.v) {
		for (size_t i4 = 0; i4 < p->ResourceInfoList.d; i4++) {
		}
		serFree(p->ResourceInfoList.v);
	}
}

static void _serNrDrbFreeNR_MAC_ControlElementDL_Type_Value(union NR_MAC_ControlElementDL_Type_Value* p, enum NR_MAC_ControlElementDL_Type_Sel d)
{
	if (d == NR_MAC_ControlElementDL_Type_SP_ResourceSetActDeact) {
		_serNrDrbFreeNR_MAC_CE_SP_ResourceSetActDeact_Type(&p->SP_ResourceSetActDeact);
		return;
	}
	if (d == NR_MAC_ControlElementDL_Type_CSI_TriggerStateSubselection) {
		_serNrDrbFreeNR_MAC_CE_CSI_TriggerStateSubselection_Type(&p->CSI_TriggerStateSubselection);
		return;
	}
	if (d == NR_MAC_ControlElementDL_Type_TCI_StatesActDeact) {
		_serNrDrbFreeNR_MAC_CE_TCI_StatesActDeact_Type(&p->TCI_StatesActDeact);
		return;
	}
	if (d == NR_MAC_ControlElementDL_Type_SP_SRS_ActDeact) {
		_serNrDrbFreeNR_MAC_CE_SP_SRS_ActDeact_Type(&p->SP_SRS_ActDeact);
		return;
	}
}

static void _serNrDrbFreeNR_MAC_ControlElementDL_Type(struct NR_MAC_ControlElementDL_Type* p)
{
	_serNrDrbFreeNR_MAC_ControlElementDL_Type_Value(&p->v, p->d);
}

static void _serNrDrbFreeNR_MAC_ControlElementDL_Type_NR_MAC_CE_SubPDU_DL_Type_ControlElement_Optional(struct NR_MAC_ControlElementDL_Type_NR_MAC_CE_SubPDU_DL_Type_ControlElement_Optional* p)
{
	if (!p->d) return;
	_serNrDrbFreeNR_MAC_ControlElementDL_Type(&p->v);
}

static void _serNrDrbFreeNR_MAC_CE_SubPDU_DL_Type(struct NR_MAC_CE_SubPDU_DL_Type* p)
{
	_serNrDrbFreeNR_MAC_PDU_SubHeader_Type(&p->SubHeader);
	_serNrDrbFreeNR_MAC_ControlElementDL_Type_NR_MAC_CE_SubPDU_DL_Type_ControlElement_Optional(&p->ControlElement);
}

static void _serNrDrbFreeNR_MAC_CE_SubPDU_DL_List_Type_NR_MAC_PDU_DL_Type_CE_SubPDUList_Optional(struct NR_MAC_CE_SubPDU_DL_List_Type_NR_MAC_PDU_DL_Type_CE_SubPDUList_Optional* p)
{
	if (!p->d) return;
	if (p->v.v) {
		for (size_t i3 = 0; i3 < p->v.d; i3++) {
			_serNrDrbFreeNR_MAC_CE_SubPDU_DL_Type(&p->v.v[i3]);
		}
		serFree(p->v.v);
	}
}

static void _serNrDrbFreeNR_MAC_SDU_SubPDU_Type(struct NR_MAC_SDU_SubPDU_Type* p)
{
	_serNrDrbFreeNR_MAC_PDU_SubHeader_Type(&p->SubHeader);
	if (p->SDU.v) {
		serFree(p->SDU.v);
	}
}

static void _serNrDrbFreeNR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_DL_Type_SDU_SubPDUList_Optional(struct NR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_DL_Type_SDU_SubPDUList_Optional* p)
{
	if (!p->d) return;
	if (p->v.v) {
		for (size_t i3 = 0; i3 < p->v.d; i3++) {
			_serNrDrbFreeNR_MAC_SDU_SubPDU_Type(&p->v.v[i3]);
		}
		serFree(p->v.v);
	}
}

static void _serNrDrbFreeNR_MAC_Padding_SubPDU_Type(struct NR_MAC_Padding_SubPDU_Type* p)
{
	_serNrDrbFreeNR_MAC_PDU_SubHeader_Type(&p->SubHeader);
	if (p->Padding.v) {
		serFree(p->Padding.v);
	}
}

static void _serNrDrbFreeNR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_DL_Type_Padding_SubPDU_Optional(struct NR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_DL_Type_Padding_SubPDU_Optional* p)
{
	if (!p->d) return;
	_serNrDrbFreeNR_MAC_Padding_SubPDU_Type(&p->v);
}

static void _serNrDrbFreeNR_MAC_PDU_DL_Type(struct NR_MAC_PDU_DL_Type* p)
{
	_serNrDrbFreeNR_MAC_CE_SubPDU_DL_List_Type_NR_MAC_PDU_DL_Type_CE_SubPDUList_Optional(&p->CE_SubPDUList);
	_serNrDrbFreeNR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_DL_Type_SDU_SubPDUList_Optional(&p->SDU_SubPDUList);
	_serNrDrbFreeNR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_DL_Type_Padding_SubPDU_Optional(&p->Padding_SubPDU);
}

static void _serNrDrbFreeNR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_UL_Type_SDU_SubPDUList_Optional(struct NR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_UL_Type_SDU_SubPDUList_Optional* p)
{
	if (!p->d) return;
	if (p->v.v) {
		for (size_t i3 = 0; i3 < p->v.d; i3++) {
			_serNrDrbFreeNR_MAC_SDU_SubPDU_Type(&p->v.v[i3]);
		}
		serFree(p->v.v);
	}
}

static void _serNrDrbFreeNR_MAC_CE_LongBSR_Type(struct NR_MAC_CE_LongBSR_Type* p)
{
	if (p->BufferSizeList.v) {
		serFree(p->BufferSizeList.v);
	}
}

static void _serNrDrbFreeNR_MAC_CE_MultiEntryPHR_Type(struct NR_MAC_CE_MultiEntryPHR_Type* p)
{
	if (p->PH_Record.v) {
		for (size_t i4 = 0; i4 < p->PH_Record.d; i4++) {
		}
		serFree(p->PH_Record.v);
	}
}

static void _serNrDrbFreeNR_MAC_ControlElementUL_Type_Value(union NR_MAC_ControlElementUL_Type_Value* p, enum NR_MAC_ControlElementUL_Type_Sel d)
{
	if (d == NR_MAC_ControlElementUL_Type_LongBSR) {
		_serNrDrbFreeNR_MAC_CE_LongBSR_Type(&p->LongBSR);
		return;
	}
	if (d == NR_MAC_ControlElementUL_Type_MultiEntryPHR) {
		_serNrDrbFreeNR_MAC_CE_MultiEntryPHR_Type(&p->MultiEntryPHR);
		return;
	}
}

static void _serNrDrbFreeNR_MAC_ControlElementUL_Type(struct NR_MAC_ControlElementUL_Type* p)
{
	_serNrDrbFreeNR_MAC_ControlElementUL_Type_Value(&p->v, p->d);
}

static void _serNrDrbFreeNR_MAC_ControlElementUL_Type_NR_MAC_CE_SubPDU_UL_Type_ControlElement_Optional(struct NR_MAC_ControlElementUL_Type_NR_MAC_CE_SubPDU_UL_Type_ControlElement_Optional* p)
{
	if (!p->d) return;
	_serNrDrbFreeNR_MAC_ControlElementUL_Type(&p->v);
}

static void _serNrDrbFreeNR_MAC_CE_SubPDU_UL_Type(struct NR_MAC_CE_SubPDU_UL_Type* p)
{
	_serNrDrbFreeNR_MAC_PDU_SubHeader_Type(&p->SubHeader);
	_serNrDrbFreeNR_MAC_ControlElementUL_Type_NR_MAC_CE_SubPDU_UL_Type_ControlElement_Optional(&p->ControlElement);
}

static void _serNrDrbFreeNR_MAC_CE_SubPDU_UL_List_Type_NR_MAC_PDU_UL_Type_CE_SubPDUList_Optional(struct NR_MAC_CE_SubPDU_UL_List_Type_NR_MAC_PDU_UL_Type_CE_SubPDUList_Optional* p)
{
	if (!p->d) return;
	if (p->v.v) {
		for (size_t i3 = 0; i3 < p->v.d; i3++) {
			_serNrDrbFreeNR_MAC_CE_SubPDU_UL_Type(&p->v.v[i3]);
		}
		serFree(p->v.v);
	}
}

static void _serNrDrbFreeNR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_UL_Type_Padding_SubPDU_Optional(struct NR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_UL_Type_Padding_SubPDU_Optional* p)
{
	if (!p->d) return;
	_serNrDrbFreeNR_MAC_Padding_SubPDU_Type(&p->v);
}

static void _serNrDrbFreeNR_MAC_PDU_UL_Type(struct NR_MAC_PDU_UL_Type* p)
{
	_serNrDrbFreeNR_MAC_SDU_SubPDU_List_Type_NR_MAC_PDU_UL_Type_SDU_SubPDUList_Optional(&p->SDU_SubPDUList);
	_serNrDrbFreeNR_MAC_CE_SubPDU_UL_List_Type_NR_MAC_PDU_UL_Type_CE_SubPDUList_Optional(&p->CE_SubPDUList);
	_serNrDrbFreeNR_MAC_Padding_SubPDU_Type_NR_MAC_PDU_UL_Type_Padding_SubPDU_Optional(&p->Padding_SubPDU);
}

static void _serNrDrbFreeNR_MAC_PDU_Type_Value(union NR_MAC_PDU_Type_Value* p, enum NR_MAC_PDU_Type_Sel d)
{
	if (d == NR_MAC_PDU_Type_DL) {
		_serNrDrbFreeNR_MAC_PDU_DL_Type(&p->DL);
		return;
	}
	if (d == NR_MAC_PDU_Type_UL) {
		_serNrDrbFreeNR_MAC_PDU_UL_Type(&p->UL);
		return;
	}
}

static void _serNrDrbFreeNR_MAC_PDU_Type(struct NR_MAC_PDU_Type* p)
{
	_serNrDrbFreeNR_MAC_PDU_Type_Value(&p->v, p->d);
}

static void _serNrDrbFreeNR_RLC_UMD_PduNoSN_Type(struct NR_RLC_UMD_PduNoSN_Type* p)
{
	if (p->Data.v) {
		serFree(p->Data.v);
	}
}

static void _serNrDrbFreeNR_RLC_UMD_PduSN6Bit_Type(struct NR_RLC_UMD_PduSN6Bit_Type* p)
{
	if (p->Data.v) {
		serFree(p->Data.v);
	}
}

static void _serNrDrbFreeNR_RLC_UMD_PduSN12Bit_Type(struct NR_RLC_UMD_PduSN12Bit_Type* p)
{
	if (p->Data.v) {
		serFree(p->Data.v);
	}
}

static void _serNrDrbFreeNR_RLC_UMD_PDU_Type_Value(union NR_RLC_UMD_PDU_Type_Value* p, enum NR_RLC_UMD_PDU_Type_Sel d)
{
	if (d == NR_RLC_UMD_PDU_Type_NoSN) {
		_serNrDrbFreeNR_RLC_UMD_PduNoSN_Type(&p->NoSN);
		return;
	}
	if (d == NR_RLC_UMD_PDU_Type_SN6Bit) {
		_serNrDrbFreeNR_RLC_UMD_PduSN6Bit_Type(&p->SN6Bit);
		return;
	}
	if (d == NR_RLC_UMD_PDU_Type_SN12Bit) {
		_serNrDrbFreeNR_RLC_UMD_PduSN12Bit_Type(&p->SN12Bit);
		return;
	}
}

static void _serNrDrbFreeNR_RLC_UMD_PDU_Type(struct NR_RLC_UMD_PDU_Type* p)
{
	_serNrDrbFreeNR_RLC_UMD_PDU_Type_Value(&p->v, p->d);
}

static void _serNrDrbFreeNR_RLC_AMD_PduSN12Bit_Type(struct NR_RLC_AMD_PduSN12Bit_Type* p)
{
	if (p->Data.v) {
		serFree(p->Data.v);
	}
}

static void _serNrDrbFreeNR_RLC_AMD_PduSN18Bit_Type(struct NR_RLC_AMD_PduSN18Bit_Type* p)
{
	if (p->Data.v) {
		serFree(p->Data.v);
	}
}

static void _serNrDrbFreeNR_RLC_AMD_PDU_Type_Value(union NR_RLC_AMD_PDU_Type_Value* p, enum NR_RLC_AMD_PDU_Type_Sel d)
{
	if (d == NR_RLC_AMD_PDU_Type_SN12Bit) {
		_serNrDrbFreeNR_RLC_AMD_PduSN12Bit_Type(&p->SN12Bit);
		return;
	}
	if (d == NR_RLC_AMD_PDU_Type_SN18Bit) {
		_serNrDrbFreeNR_RLC_AMD_PduSN18Bit_Type(&p->SN18Bit);
		return;
	}
}

static void _serNrDrbFreeNR_RLC_AMD_PDU_Type(struct NR_RLC_AMD_PDU_Type* p)
{
	_serNrDrbFreeNR_RLC_AMD_PDU_Type_Value(&p->v, p->d);
}

static void _serNrDrbFreeNR_RLC_Status_NackListSN12Bit_Type_NackList_Optional(struct NR_RLC_Status_NackListSN12Bit_Type_NackList_Optional* p)
{
	if (!p->d) return;
	if (p->v.v) {
		for (size_t i3 = 0; i3 < p->v.d; i3++) {
		}
		serFree(p->v.v);
	}
}

static void _serNrDrbFreeNR_RLC_StatusPduSN12Bit_Type(struct NR_RLC_StatusPduSN12Bit_Type* p)
{
	_serNrDrbFreeNR_RLC_Status_NackListSN12Bit_Type_NackList_Optional(&p->NackList);
}

static void _serNrDrbFreeNR_RLC_Status_NackListSN18Bit_Type_NackList_Optional(struct NR_RLC_Status_NackListSN18Bit_Type_NackList_Optional* p)
{
	if (!p->d) return;
	if (p->v.v) {
		for (size_t i3 = 0; i3 < p->v.d; i3++) {
		}
		serFree(p->v.v);
	}
}

static void _serNrDrbFreeNR_RLC_StatusPduSN18Bit_Type(struct NR_RLC_StatusPduSN18Bit_Type* p)
{
	_serNrDrbFreeNR_RLC_Status_NackListSN18Bit_Type_NackList_Optional(&p->NackList);
}

static void _serNrDrbFreeNR_RLC_AM_StatusPDU_Type_Value(union NR_RLC_AM_StatusPDU_Type_Value* p, enum NR_RLC_AM_StatusPDU_Type_Sel d)
{
	if (d == NR_RLC_AM_StatusPDU_Type_SN12Bit) {
		_serNrDrbFreeNR_RLC_StatusPduSN12Bit_Type(&p->SN12Bit);
		return;
	}
	if (d == NR_RLC_AM_StatusPDU_Type_SN18Bit) {
		_serNrDrbFreeNR_RLC_StatusPduSN18Bit_Type(&p->SN18Bit);
		return;
	}
}

static void _serNrDrbFreeNR_RLC_AM_StatusPDU_Type(struct NR_RLC_AM_StatusPDU_Type* p)
{
	_serNrDrbFreeNR_RLC_AM_StatusPDU_Type_Value(&p->v, p->d);
}

static void _serNrDrbFreeNR_RLC_PDU_Type_Value(union NR_RLC_PDU_Type_Value* p, enum NR_RLC_PDU_Type_Sel d)
{
	if (d == NR_RLC_PDU_Type_TMD) {
		if (p->TMD.v) {
			serFree(p->TMD.v);
		}
		return;
	}
	if (d == NR_RLC_PDU_Type_UMD) {
		_serNrDrbFreeNR_RLC_UMD_PDU_Type(&p->UMD);
		return;
	}
	if (d == NR_RLC_PDU_Type_AMD) {
		_serNrDrbFreeNR_RLC_AMD_PDU_Type(&p->AMD);
		return;
	}
	if (d == NR_RLC_PDU_Type_Status) {
		_serNrDrbFreeNR_RLC_AM_StatusPDU_Type(&p->Status);
		return;
	}
}

static void _serNrDrbFreeNR_RLC_PDU_Type(struct NR_RLC_PDU_Type* p)
{
	_serNrDrbFreeNR_RLC_PDU_Type_Value(&p->v, p->d);
}

static void _serNrDrbFreeNR_PDCP_DataPduSN12Bits_Type(struct NR_PDCP_DataPduSN12Bits_Type* p)
{
	if (p->SDU.v) {
		serFree(p->SDU.v);
	}
}

static void _serNrDrbFreeNR_PDCP_DataPduSN18Bits_Type(struct NR_PDCP_DataPduSN18Bits_Type* p)
{
	if (p->SDU.v) {
		serFree(p->SDU.v);
	}
}

static void _serNrDrbFreeOCTET_STRING_NR_PDCP_CtrlPduStatus_Type_Bitmap_Optional(struct OCTET_STRING_NR_PDCP_CtrlPduStatus_Type_Bitmap_Optional* p)
{
	if (!p->d) return;
	if (p->v.v) {
		serFree(p->v.v);
	}
}

static void _serNrDrbFreeNR_PDCP_CtrlPduStatus_Type(struct NR_PDCP_CtrlPduStatus_Type* p)
{
	_serNrDrbFreeOCTET_STRING_NR_PDCP_CtrlPduStatus_Type_Bitmap_Optional(&p->Bitmap);
}

static void _serNrDrbFreeNR_PDCP_CtrlPduRohcFeedback_Type(struct NR_PDCP_CtrlPduRohcFeedback_Type* p)
{
	if (p->RohcFeedback.v) {
		serFree(p->RohcFeedback.v);
	}
}

static void _serNrDrbFreeNR_PDCP_PDU_Type_Value(union NR_PDCP_PDU_Type_Value* p, enum NR_PDCP_PDU_Type_Sel d)
{
	if (d == NR_PDCP_PDU_Type_DataPduSN12Bits) {
		_serNrDrbFreeNR_PDCP_DataPduSN12Bits_Type(&p->DataPduSN12Bits);
		return;
	}
	if (d == NR_PDCP_PDU_Type_DataPduSN18Bits) {
		_serNrDrbFreeNR_PDCP_DataPduSN18Bits_Type(&p->DataPduSN18Bits);
		return;
	}
	if (d == NR_PDCP_PDU_Type_CtrlPduStatus) {
		_serNrDrbFreeNR_PDCP_CtrlPduStatus_Type(&p->CtrlPduStatus);
		return;
	}
	if (d == NR_PDCP_PDU_Type_CtrlPduRohcFeedback) {
		_serNrDrbFreeNR_PDCP_CtrlPduRohcFeedback_Type(&p->CtrlPduRohcFeedback);
		return;
	}
}

static void _serNrDrbFreeNR_PDCP_PDU_Type(struct NR_PDCP_PDU_Type* p)
{
	_serNrDrbFreeNR_PDCP_PDU_Type_Value(&p->v, p->d);
}

static void _serNrDrbFreeSDAP_PDU_DL_Type(struct SDAP_PDU_DL_Type* p)
{
	if (p->Data.v) {
		serFree(p->Data.v);
	}
}

static void _serNrDrbFreeSDAP_SDU_Type_SDAP_PDU_UL_Type_Data_Optional(struct SDAP_SDU_Type_SDAP_PDU_UL_Type_Data_Optional* p)
{
	if (!p->d) return;
	if (p->v.v) {
		serFree(p->v.v);
	}
}

static void _serNrDrbFreeSDAP_PDU_UL_Type(struct SDAP_PDU_UL_Type* p)
{
	_serNrDrbFreeSDAP_SDU_Type_SDAP_PDU_UL_Type_Data_Optional(&p->Data);
}

static void _serNrDrbFreeSDAP_PDU_Type_Value(union SDAP_PDU_Type_Value* p, enum SDAP_PDU_Type_Sel d)
{
	if (d == SDAP_PDU_Type_DL) {
		_serNrDrbFreeSDAP_PDU_DL_Type(&p->DL);
		return;
	}
	if (d == SDAP_PDU_Type_UL) {
		_serNrDrbFreeSDAP_PDU_UL_Type(&p->UL);
		return;
	}
}

static void _serNrDrbFreeSDAP_PDU_Type(struct SDAP_PDU_Type* p)
{
	_serNrDrbFreeSDAP_PDU_Type_Value(&p->v, p->d);
}

static void _serNrDrbFreeNR_L2DataList_Type_Value(union NR_L2DataList_Type_Value* p, enum NR_L2DataList_Type_Sel d)
{
	if (d == NR_L2DataList_Type_MacPdu) {
		if (p->MacPdu.v) {
			for (size_t i2 = 0; i2 < p->MacPdu.d; i2++) {
				_serNrDrbFreeNR_MAC_PDU_Type(&p->MacPdu.v[i2]);
			}
			serFree(p->MacPdu.v);
		}
		return;
	}
	if (d == NR_L2DataList_Type_RlcPdu) {
		if (p->RlcPdu.v) {
			for (size_t i2 = 0; i2 < p->RlcPdu.d; i2++) {
				_serNrDrbFreeNR_RLC_PDU_Type(&p->RlcPdu.v[i2]);
			}
			serFree(p->RlcPdu.v);
		}
		return;
	}
	if (d == NR_L2DataList_Type_RlcSdu) {
		if (p->RlcSdu.v) {
			for (size_t i2 = 0; i2 < p->RlcSdu.d; i2++) {
				if (p->RlcSdu.v[i2].v) {
					serFree(p->RlcSdu.v[i2].v);
				}
			}
			serFree(p->RlcSdu.v);
		}
		return;
	}
	if (d == NR_L2DataList_Type_PdcpPdu) {
		if (p->PdcpPdu.v) {
			for (size_t i2 = 0; i2 < p->PdcpPdu.d; i2++) {
				_serNrDrbFreeNR_PDCP_PDU_Type(&p->PdcpPdu.v[i2]);
			}
			serFree(p->PdcpPdu.v);
		}
		return;
	}
	if (d == NR_L2DataList_Type_PdcpSdu) {
		if (p->PdcpSdu.v) {
			for (size_t i2 = 0; i2 < p->PdcpSdu.d; i2++) {
				if (p->PdcpSdu.v[i2].v) {
					serFree(p->PdcpSdu.v[i2].v);
				}
			}
			serFree(p->PdcpSdu.v);
		}
		return;
	}
	if (d == NR_L2DataList_Type_SdapPdu) {
		if (p->SdapPdu.v) {
			for (size_t i2 = 0; i2 < p->SdapPdu.d; i2++) {
				_serNrDrbFreeSDAP_PDU_Type(&p->SdapPdu.v[i2]);
			}
			serFree(p->SdapPdu.v);
		}
		return;
	}
	if (d == NR_L2DataList_Type_SdapSdu) {
		if (p->SdapSdu.v) {
			for (size_t i2 = 0; i2 < p->SdapSdu.d; i2++) {
				if (p->SdapSdu.v[i2].v) {
					serFree(p->SdapSdu.v[i2].v);
				}
			}
			serFree(p->SdapSdu.v);
		}
		return;
	}
}

static void _serNrDrbFreeNR_L2DataList_Type(struct NR_L2DataList_Type* p)
{
	_serNrDrbFreeNR_L2DataList_Type_Value(&p->v, p->d);
}

static void _serNrDrbFreeNR_DRB_DataPerSlot_DL_Type(struct NR_DRB_DataPerSlot_DL_Type* p)
{
	_serNrDrbFreeNR_L2DataList_Type(&p->PduSduList);
}

static void _serNrDrbFreeNR_L2Data_Request_Type(struct NR_L2Data_Request_Type* p)
{
	if (p->SlotDataList.v) {
		for (size_t i1 = 0; i1 < p->SlotDataList.d; i1++) {
			_serNrDrbFreeNR_DRB_DataPerSlot_DL_Type(&p->SlotDataList.v[i1]);
		}
		serFree(p->SlotDataList.v);
	}
}

static void _serNrDrbFreeNR_DRB_COMMON_REQ(struct NR_DRB_COMMON_REQ* p)
{
	_serNrDrbFreeNR_L2Data_Request_Type(&p->U_Plane);
}

void serNrDrbProcessFromSSFree0Srv(struct NR_DRB_COMMON_REQ* FromSS)
{
	if (FromSS) {
		_serNrDrbFreeNR_DRB_COMMON_REQ(FromSS);
	}
}

void serNrDrbProcessFromSSFreeSrv(struct NR_DRB_COMMON_REQ* FromSS)
{
	if (FromSS) {
		_serNrDrbFreeNR_DRB_COMMON_REQ(FromSS);
		serFree(FromSS);
	}
}

void serNrDrbProcessToSSInitSrv(unsigned char* _arena, size_t _aSize, struct NR_DRB_COMMON_IND** ToSS)
{
	serMem_t _mem = serMemInit(_arena, _aSize);

	*ToSS = (struct NR_DRB_COMMON_IND*)serMalloc(_mem, sizeof(struct NR_DRB_COMMON_IND));
	memset(*ToSS, 0, sizeof(struct NR_DRB_COMMON_IND));
}

static int _serNrDrbEncNR_RoutingInfoSUL_Type_RoutingInfoSUL_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_RoutingInfoSUL_Type_RoutingInfoSUL_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	HTON_8(&_buffer[*_lidx], p->v, _lidx);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncMacBearerRouting_Type_NR_IndAspCommonPart_Type_MacBearerRouting_Optional(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct MacBearerRouting_Type_NR_IndAspCommonPart_Type_MacBearerRouting_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->d, _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	_serNrDrbEncMacBearerRouting_Type(_buffer, _size, _lidx, &p->v);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncIntegrityErrorIndication_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct IntegrityErrorIndication_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->Nas, _lidx);
	HTON_8(&_buffer[*_lidx], p->Pdcp, _lidx);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncErrorIndication_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct ErrorIndication_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncIntegrityErrorIndication_Type(_buffer, _size, _lidx, &p->Integrity);
	HTON_32(&_buffer[*_lidx], p->System, _lidx);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncIndicationStatus_Type_Value(unsigned char* _buffer, size_t _size, size_t* _lidx, const union IndicationStatus_Type_Value* p, enum IndicationStatus_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == IndicationStatus_Type_Ok) {
		HTON_8(&_buffer[*_lidx], p->Ok, _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == IndicationStatus_Type_Error) {
		_serNrDrbEncErrorIndication_Type(_buffer, _size, _lidx, &p->Error);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrDrbEncIndicationStatus_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct IndicationStatus_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->d;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrDrbEncIndicationStatus_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_IndAspCommonPart_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_IndAspCommonPart_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp = (size_t)p->CellId;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}
	_serNrDrbEncNR_RoutingInfo_Type(_buffer, _size, _lidx, &p->RoutingInfo);
	_serNrDrbEncNR_RoutingInfoSUL_Type_RoutingInfoSUL_Optional(_buffer, _size, _lidx, &p->RoutingInfoSUL);
	_serNrDrbEncRlcBearerRouting_Type(_buffer, _size, _lidx, &p->RlcBearerRouting);
	_serNrDrbEncMacBearerRouting_Type_NR_IndAspCommonPart_Type_MacBearerRouting_Optional(_buffer, _size, _lidx, &p->MacBearerRouting);
	_serNrDrbEncTimingInfo_Type(_buffer, _size, _lidx, &p->TimingInfo);
	_serNrDrbEncIndicationStatus_Type(_buffer, _size, _lidx, &p->Status);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_DRB_DataPerSlot_UL_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_DRB_DataPerSlot_UL_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncNR_L2DataList_Type(_buffer, _size, _lidx, &p->PduSduList);
	HTON_32(&_buffer[*_lidx], p->NoOfTTIs, _lidx);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_L2Data_Indication_Type(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_L2Data_Indication_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncNR_DRB_DataPerSlot_UL_Type(_buffer, _size, _lidx, &p->SlotData);

	return SIDL_STATUS_OK;
}

static int _serNrDrbEncNR_DRB_COMMON_IND(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_DRB_COMMON_IND* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncNR_IndAspCommonPart_Type(_buffer, _size, _lidx, &p->Common);
	_serNrDrbEncNR_L2Data_Indication_Type(_buffer, _size, _lidx, &p->U_Plane);

	return SIDL_STATUS_OK;
}

int serNrDrbProcessToSSEncSrv(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct NR_DRB_COMMON_IND* ToSS)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbEncNR_DRB_COMMON_IND(_buffer, _size, _lidx, ToSS);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_RoutingInfoSUL_Type_RoutingInfoSUL_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_RoutingInfoSUL_Type_RoutingInfoSUL_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	NTOH_8(p->v, &_buffer[*_lidx], _lidx);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecMacBearerRouting_Type_NR_IndAspCommonPart_Type_MacBearerRouting_Optional(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct MacBearerRouting_Type_NR_IndAspCommonPart_Type_MacBearerRouting_Optional* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->d, &_buffer[*_lidx], _lidx);
	if (!p->d) return SIDL_STATUS_OK;
	_serNrDrbDecMacBearerRouting_Type(_buffer, _size, _lidx, &p->v);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecIntegrityErrorIndication_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct IntegrityErrorIndication_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->Nas, &_buffer[*_lidx], _lidx);
	NTOH_8(p->Pdcp, &_buffer[*_lidx], _lidx);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecErrorIndication_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct ErrorIndication_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecIntegrityErrorIndication_Type(_buffer, _size, _lidx, &p->Integrity);
	NTOH_32(p->System, &_buffer[*_lidx], _lidx);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecIndicationStatus_Type_Value(const unsigned char* _buffer, size_t _size, size_t* _lidx, union IndicationStatus_Type_Value* p, enum IndicationStatus_Type_Sel d)
{
	(void)_size; // TODO: generate boundaries checking

	if (d == IndicationStatus_Type_Ok) {
		NTOH_8(p->Ok, &_buffer[*_lidx], _lidx);
		return SIDL_STATUS_OK;
	}
	if (d == IndicationStatus_Type_Error) {
		_serNrDrbDecErrorIndication_Type(_buffer, _size, _lidx, &p->Error);
		return SIDL_STATUS_OK;
	}

	return SIDL_STATUS_ERROR;
}

static int _serNrDrbDecIndicationStatus_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct IndicationStatus_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->d = (enum IndicationStatus_Type_Sel)_tmp;
	}
	_serNrDrbDecIndicationStatus_Type_Value(_buffer, _size, _lidx, &p->v, p->d);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_IndAspCommonPart_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct NR_IndAspCommonPart_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->CellId = (NR_CellId_Type)_tmp;
	}
	_serNrDrbDecNR_RoutingInfo_Type(_buffer, _size, _lidx, &p->RoutingInfo);
	_serNrDrbDecNR_RoutingInfoSUL_Type_RoutingInfoSUL_Optional(_buffer, _size, _lidx, &p->RoutingInfoSUL);
	_serNrDrbDecRlcBearerRouting_Type(_buffer, _size, _lidx, &p->RlcBearerRouting);
	_serNrDrbDecMacBearerRouting_Type_NR_IndAspCommonPart_Type_MacBearerRouting_Optional(_buffer, _size, _lidx, &p->MacBearerRouting);
	_serNrDrbDecTimingInfo_Type(_buffer, _size, _lidx, &p->TimingInfo);
	_serNrDrbDecIndicationStatus_Type(_buffer, _size, _lidx, &p->Status);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_DRB_DataPerSlot_UL_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_DRB_DataPerSlot_UL_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecNR_L2DataList_Type(_buffer, _size, _lidx, _mem, &p->PduSduList);
	NTOH_32(p->NoOfTTIs, &_buffer[*_lidx], _lidx);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_L2Data_Indication_Type(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_L2Data_Indication_Type* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecNR_DRB_DataPerSlot_UL_Type(_buffer, _size, _lidx, _mem, &p->SlotData);

	return SIDL_STATUS_OK;
}

static int _serNrDrbDecNR_DRB_COMMON_IND(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct NR_DRB_COMMON_IND* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serNrDrbDecNR_IndAspCommonPart_Type(_buffer, _size, _lidx, &p->Common);
	_serNrDrbDecNR_L2Data_Indication_Type(_buffer, _size, _lidx, _mem, &p->U_Plane);

	return SIDL_STATUS_OK;
}

int serNrDrbProcessToSSDecClt(const unsigned char* _buffer, size_t _size, unsigned char* _arena, size_t _aSize, struct NR_DRB_COMMON_IND** ToSS)
{
	(void)_size; // TODO: generate boundaries checking

	serMem_t _mem = serMemInit(_arena, _aSize);

	size_t __lidx = 0;
	size_t* _lidx = &__lidx;

	*ToSS = (struct NR_DRB_COMMON_IND*)serMalloc(_mem, sizeof(struct NR_DRB_COMMON_IND));
	_serNrDrbDecNR_DRB_COMMON_IND(_buffer, _size, _lidx, _mem, *ToSS);

	return SIDL_STATUS_OK;
}

static void _serNrDrbFreeNR_DRB_DataPerSlot_UL_Type(struct NR_DRB_DataPerSlot_UL_Type* p)
{
	_serNrDrbFreeNR_L2DataList_Type(&p->PduSduList);
}

static void _serNrDrbFreeNR_L2Data_Indication_Type(struct NR_L2Data_Indication_Type* p)
{
	_serNrDrbFreeNR_DRB_DataPerSlot_UL_Type(&p->SlotData);
}

static void _serNrDrbFreeNR_DRB_COMMON_IND(struct NR_DRB_COMMON_IND* p)
{
	_serNrDrbFreeNR_L2Data_Indication_Type(&p->U_Plane);
}

void serNrDrbProcessToSSFree0Clt(struct NR_DRB_COMMON_IND* ToSS)
{
	if (ToSS) {
		_serNrDrbFreeNR_DRB_COMMON_IND(ToSS);
	}
}

void serNrDrbProcessToSSFreeClt(struct NR_DRB_COMMON_IND* ToSS)
{
	if (ToSS) {
		_serNrDrbFreeNR_DRB_COMMON_IND(ToSS);
		serFree(ToSS);
	}
}
