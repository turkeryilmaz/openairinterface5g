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

#include "SIDL_NASEMU_NR_SYSTEM_PORT.h"
#include "acp.h"

SIDL_BEGIN_C_INTERFACE

void acpNrSysSrbProcessFromSSInitClt(acpCtx_t _ctx, struct NR_RRC_PDU_REQ** FromSS);

int acpNrSysSrbProcessFromSSEncClt(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct NR_RRC_PDU_REQ* FromSS);

int acpNrSysSrbProcessFromSSDecSrv(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct NR_RRC_PDU_REQ** FromSS);

void acpNrSysSrbProcessFromSSFree0Srv(struct NR_RRC_PDU_REQ* FromSS);

void acpNrSysSrbProcessFromSSFreeSrv(struct NR_RRC_PDU_REQ* FromSS);

void acpNrSysSrbProcessFromSSFree0CltSrv(struct NR_RRC_PDU_REQ* FromSS);

void acpNrSysSrbProcessFromSSFreeCltSrv(struct NR_RRC_PDU_REQ* FromSS);

void acpNrSysSrbProcessToSSInitSrv(acpCtx_t _ctx, struct NR_RRC_PDU_IND** ToSS);

int acpNrSysSrbProcessToSSEncSrv(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct NR_RRC_PDU_IND* ToSS);

int acpNrSysSrbProcessToSSDecClt(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct NR_RRC_PDU_IND** ToSS);

void acpNrSysSrbProcessToSSFree0Clt(struct NR_RRC_PDU_IND* ToSS);

void acpNrSysSrbProcessToSSFreeClt(struct NR_RRC_PDU_IND* ToSS);

void acpNrSysSrbProcessToSSFree0SrvClt(struct NR_RRC_PDU_IND* ToSS);

void acpNrSysSrbProcessToSSFreeSrvClt(struct NR_RRC_PDU_IND* ToSS);

SIDL_END_C_INTERFACE
