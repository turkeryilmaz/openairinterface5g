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
#include "acp.h"

SIDL_BEGIN_C_INTERFACE

void acpSysVTEnquireTimingAckInitClt(acpCtx_t _ctx, struct VirtualTimeInfo_Type** FromSS);

int acpSysVTEnquireTimingAckEncClt(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct VirtualTimeInfo_Type* FromSS);

int acpSysVTEnquireTimingAckDecSrv(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct VirtualTimeInfo_Type** FromSS);

void acpSysVTEnquireTimingAckFree0Srv(struct VirtualTimeInfo_Type* FromSS);

void acpSysVTEnquireTimingAckFreeSrv(struct VirtualTimeInfo_Type* FromSS);

void acpSysVTEnquireTimingAckFree0CltSrv(struct VirtualTimeInfo_Type* FromSS);

void acpSysVTEnquireTimingAckFreeCltSrv(struct VirtualTimeInfo_Type* FromSS);

void acpSysVTEnquireTimingUpdInitSrv(acpCtx_t _ctx, struct VirtualTimeInfo_Type** ToSS);

int acpSysVTEnquireTimingUpdEncSrv(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct VirtualTimeInfo_Type* ToSS);

int acpSysVTEnquireTimingUpdDecClt(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct VirtualTimeInfo_Type** ToSS);

void acpSysVTEnquireTimingUpdFree0Clt(struct VirtualTimeInfo_Type* ToSS);

void acpSysVTEnquireTimingUpdFreeClt(struct VirtualTimeInfo_Type* ToSS);

void acpSysVTEnquireTimingUpdFree0SrvClt(struct VirtualTimeInfo_Type* ToSS);

void acpSysVTEnquireTimingUpdFreeSrvClt(struct VirtualTimeInfo_Type* ToSS);

SIDL_END_C_INTERFACE
