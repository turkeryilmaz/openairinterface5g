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

#include "SIDL_NR_SYSTEM_PORT.h"
#include "acp.h"

SIDL_BEGIN_C_INTERFACE

void acpNrSysProcessInitClt(acpCtx_t _ctx, struct NR_SYSTEM_CTRL_REQ** FromSS);

int acpNrSysProcessEncClt(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct NR_SYSTEM_CTRL_REQ* FromSS);

int acpNrSysProcessDecSrv(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct NR_SYSTEM_CTRL_REQ** FromSS);

void acpNrSysProcessFree0Srv(struct NR_SYSTEM_CTRL_REQ* FromSS);

void acpNrSysProcessFreeSrv(struct NR_SYSTEM_CTRL_REQ* FromSS);

void acpNrSysProcessFree0CltSrv(struct NR_SYSTEM_CTRL_REQ* FromSS);

void acpNrSysProcessFreeCltSrv(struct NR_SYSTEM_CTRL_REQ* FromSS);

void acpNrSysProcessInitSrv(acpCtx_t _ctx, struct NR_SYSTEM_CTRL_CNF** ToSS);

int acpNrSysProcessEncSrv(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct NR_SYSTEM_CTRL_CNF* ToSS);

int acpNrSysProcessDecClt(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct NR_SYSTEM_CTRL_CNF** ToSS);

void acpNrSysProcessFree0Clt(struct NR_SYSTEM_CTRL_CNF* ToSS);

void acpNrSysProcessFreeClt(struct NR_SYSTEM_CTRL_CNF* ToSS);

void acpNrSysProcessFree0SrvClt(struct NR_SYSTEM_CTRL_CNF* ToSS);

void acpNrSysProcessFreeSrvClt(struct NR_SYSTEM_CTRL_CNF* ToSS);

SIDL_END_C_INTERFACE
