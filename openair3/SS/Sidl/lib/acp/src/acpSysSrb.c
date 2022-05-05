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

#include "acpSysSrb.h"
#include "acpCtx.h"
#include "acpProto.h"
#include "acpMsgIds.h"
#include "serSysSrb.h"

void acpSysSrbProcessFromSSInitClt(acpCtx_t _ctx, struct EUTRA_RRC_PDU_REQ** FromSS)
{
	if (!acpCtxIsValid(_ctx)) {
		SIDL_ASSERT(_ctx != _ctx);
	}
	serSysSrbProcessFromSSInitClt(ACP_CTX_CAST(_ctx)->arena, ACP_CTX_CAST(_ctx)->aSize, FromSS);
}

int acpSysSrbProcessFromSSEncClt(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct EUTRA_RRC_PDU_REQ* FromSS)
{
	if (!acpCtxIsValid(_ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	size_t _lidx = ACP_HEADER_SIZE;
	int _ret = serSysSrbProcessFromSSEncClt(_buffer, *_size, &_lidx, FromSS);
	if (_ret == SIDL_STATUS_OK) {
		acpBuildHeader(_ctx, ACP_LID_SysSrbProcessFromSS, _lidx, _buffer);
	}
	*_size = _lidx;
	return _ret;
}

int acpSysSrbProcessFromSSDecSrv(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct EUTRA_RRC_PDU_REQ** FromSS)
{
	if (!acpCtxIsValid(_ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	return serSysSrbProcessFromSSDecSrv(_buffer + ACP_HEADER_SIZE, _size - ACP_HEADER_SIZE, ACP_CTX_CAST(_ctx)->arena, ACP_CTX_CAST(_ctx)->aSize, FromSS);
}

void acpSysSrbProcessFromSSFree0Srv(struct EUTRA_RRC_PDU_REQ* FromSS)
{
	serSysSrbProcessFromSSFree0Srv(FromSS);
}

void acpSysSrbProcessFromSSFreeSrv(struct EUTRA_RRC_PDU_REQ* FromSS)
{
	serSysSrbProcessFromSSFreeSrv(FromSS);
}

void acpSysSrbProcessFromSSFree0CltSrv(struct EUTRA_RRC_PDU_REQ* FromSS)
{
	serSysSrbProcessFromSSFree0Srv(FromSS);
}

void acpSysSrbProcessFromSSFreeCltSrv(struct EUTRA_RRC_PDU_REQ* FromSS)
{
	serSysSrbProcessFromSSFreeSrv(FromSS);
}

void acpSysSrbProcessToSSInitSrv(acpCtx_t _ctx, struct EUTRA_RRC_PDU_IND** ToSS)
{
	if (!acpCtxIsValid(_ctx)) {
		SIDL_ASSERT(_ctx != _ctx);
	}
	serSysSrbProcessToSSInitSrv(ACP_CTX_CAST(_ctx)->arena, ACP_CTX_CAST(_ctx)->aSize, ToSS);
}

int acpSysSrbProcessToSSEncSrv(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct EUTRA_RRC_PDU_IND* ToSS)
{
	if (!acpCtxIsValid(_ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	size_t _lidx = ACP_HEADER_SIZE;
	int _ret = serSysSrbProcessToSSEncSrv(_buffer, *_size, &_lidx, ToSS);
	if (_ret == SIDL_STATUS_OK) {
		acpBuildHeader(_ctx, ACP_LID_SysSrbProcessToSS, _lidx, _buffer);
	}
	*_size = _lidx;
	return _ret;
}

int acpSysSrbProcessToSSDecClt(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct EUTRA_RRC_PDU_IND** ToSS)
{
	if (!acpCtxIsValid(_ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	return serSysSrbProcessToSSDecClt(_buffer + ACP_HEADER_SIZE, _size - ACP_HEADER_SIZE, ACP_CTX_CAST(_ctx)->arena, ACP_CTX_CAST(_ctx)->aSize, ToSS);
}

void acpSysSrbProcessToSSFree0Clt(struct EUTRA_RRC_PDU_IND* ToSS)
{
	serSysSrbProcessToSSFree0Clt(ToSS);
}

void acpSysSrbProcessToSSFreeClt(struct EUTRA_RRC_PDU_IND* ToSS)
{
	serSysSrbProcessToSSFreeClt(ToSS);
}

void acpSysSrbProcessToSSFree0SrvClt(struct EUTRA_RRC_PDU_IND* ToSS)
{
	serSysSrbProcessToSSFree0Clt(ToSS);
}

void acpSysSrbProcessToSSFreeSrvClt(struct EUTRA_RRC_PDU_IND* ToSS)
{
	serSysSrbProcessToSSFreeClt(ToSS);
}
