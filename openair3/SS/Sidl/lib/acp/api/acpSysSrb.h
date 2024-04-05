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
#include "acp.h"

SIDL_BEGIN_C_INTERFACE

void acpSysSrbProcessFromSSInitClt(acpCtx_t _ctx, struct EUTRA_RRC_PDU_REQ** FromSS);

int acpSysSrbProcessFromSSEncClt(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct EUTRA_RRC_PDU_REQ* FromSS);

int acpSysSrbProcessFromSSDecSrv(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct EUTRA_RRC_PDU_REQ** FromSS);

void acpSysSrbProcessFromSSFree0Srv(struct EUTRA_RRC_PDU_REQ* FromSS);

void acpSysSrbProcessFromSSFreeSrv(struct EUTRA_RRC_PDU_REQ* FromSS);

void acpSysSrbProcessFromSSFree0CltSrv(struct EUTRA_RRC_PDU_REQ* FromSS);

void acpSysSrbProcessFromSSFreeCltSrv(struct EUTRA_RRC_PDU_REQ* FromSS);

void acpSysSrbProcessToSSInitSrv(acpCtx_t _ctx, struct EUTRA_RRC_PDU_IND** ToSS);

int acpSysSrbProcessToSSEncSrv(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct EUTRA_RRC_PDU_IND* ToSS);

int acpSysSrbProcessToSSDecClt(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct EUTRA_RRC_PDU_IND** ToSS);

void acpSysSrbProcessToSSFree0Clt(struct EUTRA_RRC_PDU_IND* ToSS);

void acpSysSrbProcessToSSFreeClt(struct EUTRA_RRC_PDU_IND* ToSS);

void acpSysSrbProcessToSSFree0SrvClt(struct EUTRA_RRC_PDU_IND* ToSS);

void acpSysSrbProcessToSSFreeSrvClt(struct EUTRA_RRC_PDU_IND* ToSS);

SIDL_END_C_INTERFACE
