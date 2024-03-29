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

#include "SIDL_VIRTUAL_TIME_PORT.h"
#include "SidlCompiler.h"

SIDL_BEGIN_C_INTERFACE

void serSysVTEnquireTimingAckInitClt(unsigned char* _arena, size_t _aSize, struct VirtualTimeInfo_Type** FromSS);

int serSysVTEnquireTimingAckEncClt(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct VirtualTimeInfo_Type* FromSS);

int serSysVTEnquireTimingAckDecSrv(const unsigned char* _buffer, size_t _size, unsigned char* _arena, size_t _aSize, struct VirtualTimeInfo_Type** FromSS);

void serSysVTEnquireTimingAckFree0Srv(struct VirtualTimeInfo_Type* FromSS);

void serSysVTEnquireTimingAckFreeSrv(struct VirtualTimeInfo_Type* FromSS);

void serSysVTEnquireTimingUpdInitSrv(unsigned char* _arena, size_t _aSize, struct VirtualTimeInfo_Type** ToSS);

int serSysVTEnquireTimingUpdEncSrv(unsigned char* _buffer, size_t _size, size_t* _lidx, const struct VirtualTimeInfo_Type* ToSS);

int serSysVTEnquireTimingUpdDecClt(const unsigned char* _buffer, size_t _size, unsigned char* _arena, size_t _aSize, struct VirtualTimeInfo_Type** ToSS);

void serSysVTEnquireTimingUpdFree0Clt(struct VirtualTimeInfo_Type* ToSS);

void serSysVTEnquireTimingUpdFreeClt(struct VirtualTimeInfo_Type* ToSS);

SIDL_END_C_INTERFACE
