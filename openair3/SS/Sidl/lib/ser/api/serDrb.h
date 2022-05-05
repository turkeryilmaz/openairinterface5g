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

#include "SIDL_EUTRA_DRB_PORT.h"
#include "SidlCompiler.h"

SIDL_BEGIN_C_INTERFACE

void serDrbProcessFromSSInitClt(unsigned char* _arena, size_t _aSize, struct DRB_COMMON_REQ** FromSS);

int serDrbProcessFromSSEncClt(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct DRB_COMMON_REQ* FromSS);

int serDrbProcessFromSSDecSrv(const unsigned char* _buffer, size_t _size, unsigned char* _arena, size_t _aSize, struct DRB_COMMON_REQ** FromSS);

void serDrbProcessFromSSFree0Srv(struct DRB_COMMON_REQ* FromSS);

void serDrbProcessFromSSFreeSrv(struct DRB_COMMON_REQ* FromSS);

void serDrbProcessToSSInitSrv(unsigned char* _arena, size_t _aSize, struct DRB_COMMON_IND** ToSS);

int serDrbProcessToSSEncSrv(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct DRB_COMMON_IND* ToSS);

int serDrbProcessToSSDecClt(const unsigned char* _buffer, size_t _size, unsigned char* _arena, size_t _aSize, struct DRB_COMMON_IND** ToSS);

void serDrbProcessToSSFree0Clt(struct DRB_COMMON_IND* ToSS);

void serDrbProcessToSSFreeClt(struct DRB_COMMON_IND* ToSS);

SIDL_END_C_INTERFACE
