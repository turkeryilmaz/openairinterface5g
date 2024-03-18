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
#include "SidlCompiler.h"

SIDL_BEGIN_C_INTERFACE

void serHandshakeProcessInitClt(unsigned char* _arena, size_t _aSize, struct AcpHandshakeReq** FromSS);

int serHandshakeProcessEncClt(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct AcpHandshakeReq* FromSS);

int serHandshakeProcessDecSrv(const unsigned char* _buffer, size_t _size, unsigned char* _arena, size_t _aSize, struct AcpHandshakeReq** FromSS);

void serHandshakeProcessFree0Srv(struct AcpHandshakeReq* FromSS);

void serHandshakeProcessFreeSrv(struct AcpHandshakeReq* FromSS);

void serHandshakeProcessInitSrv(unsigned char* _arena, size_t _aSize, struct AcpHandshakeCnf** ToSS);

int serHandshakeProcessEncSrv(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct AcpHandshakeCnf* ToSS);

int serHandshakeProcessDecClt(const unsigned char* _buffer, size_t _size, unsigned char* _arena, size_t _aSize, struct AcpHandshakeCnf** ToSS);

void serHandshakeProcessFree0Clt(struct AcpHandshakeCnf* ToSS);

void serHandshakeProcessFreeClt(struct AcpHandshakeCnf* ToSS);

SIDL_END_C_INTERFACE
