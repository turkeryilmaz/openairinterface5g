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

// Internal includes
#include "acp.h"

SIDL_BEGIN_C_INTERFACE

enum adbgMsgLogDir {
	ADBG_MSG_LOG_RECV_DIR,
	ADBG_MSG_LOG_SEND_DIR
};

void adbgMsgLog(acpCtx_t ctx, enum adbgMsgLogDir dir, size_t size, const unsigned char* buffer);

SIDL_END_C_INTERFACE
