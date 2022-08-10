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

// System includes
#include <stdbool.h>

// Internal includes
#include "acp.h"

SIDL_BEGIN_C_INTERFACE

#pragma pack(push, 1)
struct acpWklmService {
	unsigned int id;	   // Service ID
	/*const char name[];*/ // NULL terminated service name
};
#pragma pack(pop)

#pragma pack(push, 1)
struct acpWklmServicePushMessage {
	unsigned int serviceQty;			  // Number of pushed services
	/*struct dcxWklmService services[];*/ // Array of services
};
#pragma pack(pop)

/** Process push message and resolve ids. */
void acpProcessPushMsg(acpCtx_t ctx, size_t size, const unsigned char* buffer);

/** Add ACP header in the begining of the buffer. */
void acpBuildHeader(acpCtx_t ctx, unsigned int type, size_t size, unsigned char* buffer);

SIDL_END_C_INTERFACE
