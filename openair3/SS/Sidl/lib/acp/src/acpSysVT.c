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

#include "acpSysVT.h"
#include "acpCtx.h"
#include "acpProto.h"
#include "acpMsgIds.h"
#include "serSysVT.h"

void acpSysVTEnquireTimingAckInitClt(acpCtx_t _ctx, struct VirtualTimeInfo_Type** FromSS)
{
	if (!acpCtxIsValid(_ctx)) {
		SIDL_ASSERT(_ctx != _ctx);
	}
	serSysVTEnquireTimingAckInitClt(ACP_CTX_CAST(_ctx)->arena, ACP_CTX_CAST(_ctx)->aSize, FromSS);
}

int acpSysVTEnquireTimingAckEncClt(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct VirtualTimeInfo_Type* FromSS)
{
	if (!acpCtxIsValid(_ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	size_t _lidx = ACP_HEADER_SIZE;
	int _ret = serSysVTEnquireTimingAckEncClt(_buffer, *_size, &_lidx, FromSS);
	if (_ret == SIDL_STATUS_OK) {
		acpBuildHeader(_ctx, ACP_LID_SysVTEnquireTimingAck, _lidx, _buffer);
	}
	*_size = _lidx;
	return _ret;
}

int acpSysVTEnquireTimingAckDecSrv(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct VirtualTimeInfo_Type** FromSS)
{
	if (!acpCtxIsValid(_ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	return serSysVTEnquireTimingAckDecSrv(_buffer + ACP_HEADER_SIZE, _size - ACP_HEADER_SIZE, ACP_CTX_CAST(_ctx)->arena, ACP_CTX_CAST(_ctx)->aSize, FromSS);
}

void acpSysVTEnquireTimingAckFree0Srv(struct VirtualTimeInfo_Type* FromSS)
{
	serSysVTEnquireTimingAckFree0Srv(FromSS);
}

void acpSysVTEnquireTimingAckFreeSrv(struct VirtualTimeInfo_Type* FromSS)
{
	serSysVTEnquireTimingAckFreeSrv(FromSS);
}

void acpSysVTEnquireTimingAckFree0CltSrv(struct VirtualTimeInfo_Type* FromSS)
{
	serSysVTEnquireTimingAckFree0Srv(FromSS);
}

void acpSysVTEnquireTimingAckFreeCltSrv(struct VirtualTimeInfo_Type* FromSS)
{
	serSysVTEnquireTimingAckFreeSrv(FromSS);
}

void acpSysVTEnquireTimingUpdInitSrv(acpCtx_t _ctx, struct VirtualTimeInfo_Type** ToSS)
{
	if (!acpCtxIsValid(_ctx)) {
		SIDL_ASSERT(_ctx != _ctx);
	}
	serSysVTEnquireTimingUpdInitSrv(ACP_CTX_CAST(_ctx)->arena, ACP_CTX_CAST(_ctx)->aSize, ToSS);
}

int acpSysVTEnquireTimingUpdEncSrv(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct VirtualTimeInfo_Type* ToSS)
{
	if (!acpCtxIsValid(_ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	size_t _lidx = ACP_HEADER_SIZE;
	int _ret = serSysVTEnquireTimingUpdEncSrv(_buffer, *_size, &_lidx, ToSS);
	if (_ret == SIDL_STATUS_OK) {
		acpBuildHeader(_ctx, ACP_LID_SysVTEnquireTimingUpd, _lidx, _buffer);
	}
	*_size = _lidx;
	return _ret;
}

int acpSysVTEnquireTimingUpdDecClt(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct VirtualTimeInfo_Type** ToSS)
{
	if (!acpCtxIsValid(_ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	return serSysVTEnquireTimingUpdDecClt(_buffer + ACP_HEADER_SIZE, _size - ACP_HEADER_SIZE, ACP_CTX_CAST(_ctx)->arena, ACP_CTX_CAST(_ctx)->aSize, ToSS);
}

void acpSysVTEnquireTimingUpdFree0Clt(struct VirtualTimeInfo_Type* ToSS)
{
	serSysVTEnquireTimingUpdFree0Clt(ToSS);
}

void acpSysVTEnquireTimingUpdFreeClt(struct VirtualTimeInfo_Type* ToSS)
{
	serSysVTEnquireTimingUpdFreeClt(ToSS);
}

void acpSysVTEnquireTimingUpdFree0SrvClt(struct VirtualTimeInfo_Type* ToSS)
{
	serSysVTEnquireTimingUpdFree0Clt(ToSS);
}

void acpSysVTEnquireTimingUpdFreeSrvClt(struct VirtualTimeInfo_Type* ToSS)
{
	serSysVTEnquireTimingUpdFreeClt(ToSS);
}
