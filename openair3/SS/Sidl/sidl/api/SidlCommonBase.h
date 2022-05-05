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

#include "SidlCompiler.h"
#include "SidlBase.h"

SIDL_BEGIN_C_INTERFACE

typedef bool Null_Type;

typedef bool Dummy_Type;

typedef uint32_t UInt_Type;

typedef uint16_t UInt16_Type;

typedef BIT_STRING_ELEMENT B1_Type[1];

typedef BIT_STRING_ELEMENT B2_Type[2];

typedef BIT_STRING_ELEMENT B3_Type[3];

typedef BIT_STRING_ELEMENT B4_Type[4];

typedef BIT_STRING_ELEMENT B5_Type[5];

typedef BIT_STRING_ELEMENT B6_Type[6];

typedef BIT_STRING_ELEMENT B7_Type[7];

typedef BIT_STRING_ELEMENT B8_Type[8];

typedef BIT_STRING_ELEMENT B10_Type[10];

typedef BIT_STRING_ELEMENT B11_Type[11];

typedef BIT_STRING_ELEMENT B12_Type[12];

typedef BIT_STRING_ELEMENT B14_Type[14];

typedef BIT_STRING_ELEMENT B15_Type[15];

typedef BIT_STRING_ELEMENT B16_Type[16];

typedef BIT_STRING_ELEMENT B18_Type[18];

typedef BIT_STRING_ELEMENT B32_Type[32];

typedef BIT_STRING_ELEMENT B48_Type[48];

typedef BIT_STRING_ELEMENT B128_Type[128];

struct BIT_STRING_ELEMENT_B7_15_Type_Dynamic {
	size_t d;
	BIT_STRING_ELEMENT* v;
};

typedef struct BIT_STRING_ELEMENT_B7_15_Type_Dynamic B7_15_Type;

typedef OCTET_STRING_ELEMENT O1_Type[1];

typedef OCTET_STRING_ELEMENT O4_Type[4];

struct B8_Type_B8_List_Type_Dynamic {
	size_t d;
	B8_Type* v;
};

typedef struct B8_Type_B8_List_Type_Dynamic B8_List_Type;

SIDL_END_C_INTERFACE
