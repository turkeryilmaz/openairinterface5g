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

#include "SIDL_NASEMU_EUTRA_SYSTEM_PORT.h"
#include "SidlCompiler.h"

SIDL_BEGIN_C_INTERFACE

void serSysSrbProcessFromSSInitClt(unsigned char* _arena, size_t _aSize, struct EUTRA_RRC_PDU_REQ** FromSS);

int serSysSrbProcessFromSSEncClt(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct EUTRA_RRC_PDU_REQ* FromSS);

int serSysSrbProcessFromSSDecSrv(const unsigned char* _buffer, size_t _size, unsigned char* _arena, size_t _aSize, struct EUTRA_RRC_PDU_REQ** FromSS);

void serSysSrbProcessFromSSFree0Srv(struct EUTRA_RRC_PDU_REQ* FromSS);

void serSysSrbProcessFromSSFreeSrv(struct EUTRA_RRC_PDU_REQ* FromSS);

void serSysSrbProcessToSSInitSrv(unsigned char* _arena, size_t _aSize, struct EUTRA_RRC_PDU_IND** ToSS);

int serSysSrbProcessToSSEncSrv(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct EUTRA_RRC_PDU_IND* ToSS);

int serSysSrbProcessToSSDecClt(const unsigned char* _buffer, size_t _size, unsigned char* _arena, size_t _aSize, struct EUTRA_RRC_PDU_IND** ToSS);

void serSysSrbProcessToSSFree0Clt(struct EUTRA_RRC_PDU_IND* ToSS);

void serSysSrbProcessToSSFreeClt(struct EUTRA_RRC_PDU_IND* ToSS);

SIDL_END_C_INTERFACE
