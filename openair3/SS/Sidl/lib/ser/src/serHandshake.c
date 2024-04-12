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
#include "serHandshake.h"
#include "serMem.h"
#include "serUtils.h"

void serHandshakeProcessInitClt(unsigned char* _arena, size_t _aSize, struct AcpHandshakeReq** FromSS)
{
	serMem_t _mem = serMemInit(_arena, _aSize);

	*FromSS = (struct AcpHandshakeReq*)serMalloc(_mem, sizeof(struct AcpHandshakeReq));
	memset(*FromSS, 0, sizeof(struct AcpHandshakeReq));
}

static int _serHandshakeEncAcpHandshakeVersion(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct AcpHandshakeVersion* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i1 = 0; i1 < 32; i1++) {
		HTON_8(&_buffer[*_lidx], p->release[i1], _lidx);
	}
	for (size_t i1 = 0; i1 < 64; i1++) {
		HTON_8(&_buffer[*_lidx], p->checksum[i1], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serHandshakeEncAcpHandshakeReqControlInfo(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct AcpHandshakeReqControlInfo* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_8(&_buffer[*_lidx], p->productMode, _lidx);
	{
		size_t _tmpLen = (strlen(p->desc ? p->desc : "")) + 1;
		memcpy(&_buffer[*_lidx], (p->desc ? p->desc : ""), _tmpLen);
		*_lidx += _tmpLen;
	}

	return SIDL_STATUS_OK;
}

static int _serHandshakeEncAcpHandshakeReqCommon(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct AcpHandshakeReqCommon* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serHandshakeEncAcpHandshakeVersion(_buffer, _size, _lidx, &p->version);
	_serHandshakeEncAcpHandshakeReqControlInfo(_buffer, _size, _lidx, &p->control);

	return SIDL_STATUS_OK;
}

static int _serHandshakeEncAcpHandshakeService(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct AcpHandshakeService* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_32(&_buffer[*_lidx], p->id, _lidx);
	{
		size_t _tmpLen = (strlen(p->name ? p->name : "")) + 1;
		memcpy(&_buffer[*_lidx], (p->name ? p->name : ""), _tmpLen);
		*_lidx += _tmpLen;
	}

	return SIDL_STATUS_OK;
}

static int _serHandshakeEncAcpHandshakeReqService(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct AcpHandshakeReqService* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_32(&_buffer[*_lidx], p->services.d, _lidx);
	for (size_t i1 = 0; i1 < p->services.d; i1++) {
		_serHandshakeEncAcpHandshakeService(_buffer, _size, _lidx, &p->services.v[i1]);
	}

	return SIDL_STATUS_OK;
}

static int _serHandshakeEncAcpHandshakeReq(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct AcpHandshakeReq* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serHandshakeEncAcpHandshakeReqCommon(_buffer, _size, _lidx, &p->common);
	_serHandshakeEncAcpHandshakeReqService(_buffer, _size, _lidx, &p->request);

	return SIDL_STATUS_OK;
}

int serHandshakeProcessEncClt(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct AcpHandshakeReq* FromSS)
{
	(void)_size; // TODO: generate boundaries checking

	_serHandshakeEncAcpHandshakeReq(_buffer, _size, _lidx, FromSS);

	return SIDL_STATUS_OK;
}

static int _serHandshakeDecAcpHandshakeVersion(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct AcpHandshakeVersion* p)
{
	(void)_size; // TODO: generate boundaries checking

	for (size_t i1 = 0; i1 < 32; i1++) {
		NTOH_8(p->release[i1], &_buffer[*_lidx], _lidx);
	}
	for (size_t i1 = 0; i1 < 64; i1++) {
		NTOH_8(p->checksum[i1], &_buffer[*_lidx], _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serHandshakeDecAcpHandshakeReqControlInfo(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct AcpHandshakeReqControlInfo* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_8(p->productMode, &_buffer[*_lidx], _lidx);
	{
		size_t _tmpLen = strlen((const char*)&_buffer[*_lidx]) + 1;
		p->desc = (char*)serMalloc(_mem, _tmpLen);
		memcpy(p->desc, &_buffer[*_lidx], _tmpLen);
		*_lidx += _tmpLen;
	}

	return SIDL_STATUS_OK;
}

static int _serHandshakeDecAcpHandshakeReqCommon(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct AcpHandshakeReqCommon* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serHandshakeDecAcpHandshakeVersion(_buffer, _size, _lidx, &p->version);
	_serHandshakeDecAcpHandshakeReqControlInfo(_buffer, _size, _lidx, _mem, &p->control);

	return SIDL_STATUS_OK;
}

static int _serHandshakeDecAcpHandshakeService(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct AcpHandshakeService* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_32(p->id, &_buffer[*_lidx], _lidx);
	{
		size_t _tmpLen = strlen((const char*)&_buffer[*_lidx]) + 1;
		p->name = (char*)serMalloc(_mem, _tmpLen);
		memcpy(p->name, &_buffer[*_lidx], _tmpLen);
		*_lidx += _tmpLen;
	}

	return SIDL_STATUS_OK;
}

static int _serHandshakeDecAcpHandshakeReqService(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct AcpHandshakeReqService* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_32(p->services.d, &_buffer[*_lidx], _lidx);
	p->services.v = (struct AcpHandshakeService*)serMalloc(_mem, p->services.d * sizeof(struct AcpHandshakeService));
	for (size_t i1 = 0; i1 < p->services.d; i1++) {
		_serHandshakeDecAcpHandshakeService(_buffer, _size, _lidx, _mem, &p->services.v[i1]);
	}

	return SIDL_STATUS_OK;
}

static int _serHandshakeDecAcpHandshakeReq(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct AcpHandshakeReq* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serHandshakeDecAcpHandshakeReqCommon(_buffer, _size, _lidx, _mem, &p->common);
	_serHandshakeDecAcpHandshakeReqService(_buffer, _size, _lidx, _mem, &p->request);

	return SIDL_STATUS_OK;
}

int serHandshakeProcessDecSrv(const unsigned char* _buffer, size_t _size, unsigned char* _arena, size_t _aSize, struct AcpHandshakeReq** FromSS)
{
	(void)_size; // TODO: generate boundaries checking

	serMem_t _mem = serMemInit(_arena, _aSize);

	size_t __lidx = 0;
	size_t* _lidx = &__lidx;

	*FromSS = (struct AcpHandshakeReq*)serMalloc(_mem, sizeof(struct AcpHandshakeReq));
	_serHandshakeDecAcpHandshakeReq(_buffer, _size, _lidx, _mem, *FromSS);

	return SIDL_STATUS_OK;
}

static void _serHandshakeFreeAcpHandshakeReqControlInfo(struct AcpHandshakeReqControlInfo* p)
{
	if (p->desc) {
		serFree(p->desc);
	}
}

static void _serHandshakeFreeAcpHandshakeReqCommon(struct AcpHandshakeReqCommon* p)
{
	_serHandshakeFreeAcpHandshakeReqControlInfo(&p->control);
}

static void _serHandshakeFreeAcpHandshakeService(struct AcpHandshakeService* p)
{
	if (p->name) {
		serFree(p->name);
	}
}

static void _serHandshakeFreeAcpHandshakeReqService(struct AcpHandshakeReqService* p)
{
	if (p->services.v) {
		for (size_t i1 = 0; i1 < p->services.d; i1++) {
			_serHandshakeFreeAcpHandshakeService(&p->services.v[i1]);
		}
		serFree(p->services.v);
	}
}

static void _serHandshakeFreeAcpHandshakeReq(struct AcpHandshakeReq* p)
{
	_serHandshakeFreeAcpHandshakeReqCommon(&p->common);
	_serHandshakeFreeAcpHandshakeReqService(&p->request);
}

void serHandshakeProcessFree0Srv(struct AcpHandshakeReq* FromSS)
{
	if (FromSS) {
		_serHandshakeFreeAcpHandshakeReq(FromSS);
	}
}

void serHandshakeProcessFreeSrv(struct AcpHandshakeReq* FromSS)
{
	if (FromSS) {
		_serHandshakeFreeAcpHandshakeReq(FromSS);
		serFree(FromSS);
	}
}

void serHandshakeProcessInitSrv(unsigned char* _arena, size_t _aSize, struct AcpHandshakeCnf** ToSS)
{
	serMem_t _mem = serMemInit(_arena, _aSize);

	*ToSS = (struct AcpHandshakeCnf*)serMalloc(_mem, sizeof(struct AcpHandshakeCnf));
	memset(*ToSS, 0, sizeof(struct AcpHandshakeCnf));
}

static int _serHandshakeEncAcpHandshakeCnfCommon(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct AcpHandshakeCnfCommon* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serHandshakeEncAcpHandshakeVersion(_buffer, _size, _lidx, &p->version);
	{
		size_t _tmp = (size_t)p->result;
		HTON_32(&_buffer[*_lidx], _tmp, _lidx);
	}

	return SIDL_STATUS_OK;
}

static int _serHandshakeEncAcpHandshakeCnfService(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct AcpHandshakeCnfService* p)
{
	(void)_size; // TODO: generate boundaries checking

	HTON_32(&_buffer[*_lidx], p->services.d, _lidx);
	for (size_t i1 = 0; i1 < p->services.d; i1++) {
		_serHandshakeEncAcpHandshakeService(_buffer, _size, _lidx, &p->services.v[i1]);
	}

	return SIDL_STATUS_OK;
}

static int _serHandshakeEncAcpHandshakeCnf(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct AcpHandshakeCnf* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serHandshakeEncAcpHandshakeCnfCommon(_buffer, _size, _lidx, &p->common);
	_serHandshakeEncAcpHandshakeCnfService(_buffer, _size, _lidx, &p->confirm);

	return SIDL_STATUS_OK;
}

int serHandshakeProcessEncSrv(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct AcpHandshakeCnf* ToSS)
{
	(void)_size; // TODO: generate boundaries checking

	_serHandshakeEncAcpHandshakeCnf(_buffer, _size, _lidx, ToSS);

	return SIDL_STATUS_OK;
}

static int _serHandshakeDecAcpHandshakeCnfCommon(const unsigned char* _buffer, size_t _size, size_t* _lidx, struct AcpHandshakeCnfCommon* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serHandshakeDecAcpHandshakeVersion(_buffer, _size, _lidx, &p->version);
	{
		size_t _tmp;
		NTOH_32(_tmp, &_buffer[*_lidx], _lidx);
		p->result = (AcpHandshakeCnfResult)_tmp;
	}

	return SIDL_STATUS_OK;
}

static int _serHandshakeDecAcpHandshakeCnfService(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct AcpHandshakeCnfService* p)
{
	(void)_size; // TODO: generate boundaries checking

	NTOH_32(p->services.d, &_buffer[*_lidx], _lidx);
	p->services.v = (struct AcpHandshakeService*)serMalloc(_mem, p->services.d * sizeof(struct AcpHandshakeService));
	for (size_t i1 = 0; i1 < p->services.d; i1++) {
		_serHandshakeDecAcpHandshakeService(_buffer, _size, _lidx, _mem, &p->services.v[i1]);
	}

	return SIDL_STATUS_OK;
}

static int _serHandshakeDecAcpHandshakeCnf(const unsigned char* _buffer, size_t _size, size_t* _lidx, serMem_t _mem, struct AcpHandshakeCnf* p)
{
	(void)_size; // TODO: generate boundaries checking

	_serHandshakeDecAcpHandshakeCnfCommon(_buffer, _size, _lidx, &p->common);
	_serHandshakeDecAcpHandshakeCnfService(_buffer, _size, _lidx, _mem, &p->confirm);

	return SIDL_STATUS_OK;
}

int serHandshakeProcessDecClt(const unsigned char* _buffer, size_t _size, unsigned char* _arena, size_t _aSize, struct AcpHandshakeCnf** ToSS)
{
	(void)_size; // TODO: generate boundaries checking

	serMem_t _mem = serMemInit(_arena, _aSize);

	size_t __lidx = 0;
	size_t* _lidx = &__lidx;

	*ToSS = (struct AcpHandshakeCnf*)serMalloc(_mem, sizeof(struct AcpHandshakeCnf));
	_serHandshakeDecAcpHandshakeCnf(_buffer, _size, _lidx, _mem, *ToSS);

	return SIDL_STATUS_OK;
}

static void _serHandshakeFreeAcpHandshakeCnfService(struct AcpHandshakeCnfService* p)
{
	if (p->services.v) {
		for (size_t i1 = 0; i1 < p->services.d; i1++) {
			_serHandshakeFreeAcpHandshakeService(&p->services.v[i1]);
		}
		serFree(p->services.v);
	}
}

static void _serHandshakeFreeAcpHandshakeCnf(struct AcpHandshakeCnf* p)
{
	_serHandshakeFreeAcpHandshakeCnfService(&p->confirm);
}

void serHandshakeProcessFree0Clt(struct AcpHandshakeCnf* ToSS)
{
	if (ToSS) {
		_serHandshakeFreeAcpHandshakeCnf(ToSS);
	}
}

void serHandshakeProcessFreeClt(struct AcpHandshakeCnf* ToSS)
{
	if (ToSS) {
		_serHandshakeFreeAcpHandshakeCnf(ToSS);
		serFree(ToSS);
	}
}
