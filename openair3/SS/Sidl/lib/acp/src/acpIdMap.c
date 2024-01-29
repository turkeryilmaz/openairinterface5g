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

#include "acpIdMap.h"

const struct acpIdMapItf acpItfMap[] = {
	{ "Test", 1, 0, 7 },
	{ "Sys", 2, 7, 1 },
	{ "SysVT", 3, 8, 2 },
	{ "Srb", 4, 10, 2 },
	{ "Vng", 5, 12, 1 },
	{ "Drb", 6, 13, 2 },
	{ "Handshake", 7, 15, 1 },
	{ "SysInd", 8, 16, 1 },
	{ "NrSysSrb", 9, 17, 2 },
	{ "NrSys", 10, 19, 1 },
	{ "NrDrb", 11, 20, 2 },
};

const unsigned int acpItfMapSize = sizeof(acpItfMap) / sizeof(acpItfMap[0]);

// local_id (second field) has the following format:
// 0x9004XXYY,
// where XX is the interface number (from acpItfMap second field),
// and YY is the number of service (sequence of 00..NN) in the interface.
struct acpIdMapService acpIdMap[] = {
	// Test part
	{ "TestHelloFromSS", 0x90040100, (unsigned int)-1, ACP_ONEWAY },
	{ "TestHelloToSS", 0x90040101, (unsigned int)-1, ACP_NTF },
	{ "TestPing", 0x90040102, (unsigned int)-1, ACP_CMD },
	{ "TestEcho", 0x90040103, (unsigned int)-1, ACP_CMD },
	{ "TestTest1", 0x90040104, (unsigned int)-1, ACP_CMD },
	{ "TestTest2", 0x90040105, (unsigned int)-1, ACP_CMD },
	{ "TestOther", 0x90040106, (unsigned int)-1, ACP_CMD },

	// Sys part
	{ "SysProcess", 0x90040200, (unsigned int)-1, ACP_CMD },

	// SysVT (internal) part
	{ "SysVTEnquireTimingAck", 0x90040300, (unsigned int)-1, ACP_ONEWAY },
	{ "SysVTEnquireTimingUpd", 0x90040301, (unsigned int)-1, ACP_NTF },

	// SysSrb part
	{ "SysSrbProcessFromSS", 0x90040400, (unsigned int)-1, ACP_ONEWAY },
	{ "SysSrbProcessToSS", 0x90040401, (unsigned int)-1, ACP_NTF },

	// Vng part
	{ "VngProcess", 0x90040500, (unsigned int)-1, ACP_CMD },

	// Drb part
	{ "DrbProcessFromSS", 0x90040600, (unsigned int)-1, ACP_ONEWAY },
	{ "DrbProcessToSS", 0x90040601, (unsigned int)-1, ACP_NTF },

	// Handshake (internal) part
	{ "HandshakeProcess", 0x90040700, (unsigned int)-1, ACP_ONEWAY },

	// SysInd part
	{ "SysIndProcessToSS", 0x90040800, (unsigned int)-1, ACP_NTF },

	// NrSysSrb part
	{ "NrSysSrbProcessFromSS", 0x90040900, (unsigned int)-1, ACP_ONEWAY },
	{ "NrSysSrbProcessToSS", 0x90040901, (unsigned int)-1, ACP_NTF },

	// NrSys part
	{ "NrSysProcess", 0x90040A00, (unsigned int)-1, ACP_CMD },

	// NrDrb part
	{ "NrDrbProcessFromSS", 0x90040B00, (unsigned int)-1, ACP_ONEWAY },
	{ "NrDrbProcessToSS", 0x90040B01, (unsigned int)-1, ACP_NTF },
};

const unsigned int acpIdMapSize = sizeof(acpIdMap) / sizeof(acpIdMap[0]);
