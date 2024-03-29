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
#include "SidlCompiler.h"

SIDL_BEGIN_C_INTERFACE

enum acpKind {
	ACP_NTF = 0,
	ACP_ONEWAY = 1,
	ACP_CMD = 2
};

/** Defines ACP Context. */
struct acpIdMapService {
	/** Service name. */
	const char* name;
	/** Service static local ID. */
	unsigned int local_id;
	/** Service remote ID. */
	unsigned int remote_id;
	/** Service kind (0 - NTF, 1 - ONEWAY, 2 - CMD). */
	int kind;
};

struct acpIdMapItf {
	/** SIDL interface name. */
	const char* name;
	/** Interface ID. */
	int id;
	/** Start index. */
	int startIndex;
	/** Service quantity. */
	int servicesQty;
};

/** Interface IDs mapping. */
extern const struct acpIdMapItf acpItfMap[];
extern const unsigned int acpItfMapSize;

/** Service IDs mapping. */
extern struct acpIdMapService acpIdMap[];
extern const unsigned int acpIdMapSize;

SIDL_END_C_INTERFACE
