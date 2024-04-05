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

#include "SIDL_Handshake.h"
#include "acp.h"

SIDL_BEGIN_C_INTERFACE

void acpHandshakeProcessInitClt(acpCtx_t _ctx, struct AcpHandshakeReq** FromSS);

int acpHandshakeProcessEncClt(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct AcpHandshakeReq* FromSS);

int acpHandshakeProcessDecSrv(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct AcpHandshakeReq** FromSS);

void acpHandshakeProcessFree0Srv(struct AcpHandshakeReq* FromSS);

void acpHandshakeProcessFreeSrv(struct AcpHandshakeReq* FromSS);

void acpHandshakeProcessFree0CltSrv(struct AcpHandshakeReq* FromSS);

void acpHandshakeProcessFreeCltSrv(struct AcpHandshakeReq* FromSS);

void acpHandshakeProcessInitSrv(acpCtx_t _ctx, struct AcpHandshakeCnf** ToSS);

int acpHandshakeProcessEncSrv(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct AcpHandshakeCnf* ToSS);

int acpHandshakeProcessDecClt(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct AcpHandshakeCnf** ToSS);

void acpHandshakeProcessFree0Clt(struct AcpHandshakeCnf* ToSS);

void acpHandshakeProcessFreeClt(struct AcpHandshakeCnf* ToSS);

void acpHandshakeProcessFree0SrvClt(struct AcpHandshakeCnf* ToSS);

void acpHandshakeProcessFreeSrvClt(struct AcpHandshakeCnf* ToSS);

SIDL_END_C_INTERFACE
