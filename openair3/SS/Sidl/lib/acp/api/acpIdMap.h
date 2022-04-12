/*
 *****************************************************************
 *
 * Module : Asynchronous Communication Protocol
 * Purpose: Message ID Mapping
 *
 *****************************************************************
 *
 *  Copyright (c) 2019-2021 SEQUANS Communications.
 *  All rights reserved.
 *
 *  This is confidential and proprietary source code of SEQUANS
 *  Communications. The use of the present source code and all
 *  its derived forms is exclusively governed by the restricted
 *  terms and conditions set forth in the SEQUANS
 *  Communications' EARLY ADOPTER AGREEMENT and/or LICENCE
 *  AGREEMENT. The present source code and all its derived
 *  forms can ONLY and EXCLUSIVELY be used with SEQUANS
 *  Communications' products. The distribution/sale of the
 *  present source code and all its derived forms is EXCLUSIVELY
 *  RESERVED to regular LICENCE holder and otherwise STRICTLY
 *  PROHIBITED.
 *
 *****************************************************************
 */

#pragma once

// Internal includes
#include "SidlCompiler.h"

SIDL_BEGIN_C_INTERFACE

/** Defines ACP Context. */
struct acpIdMapService {
	/** Service name. */
	const char* name;
	/** Service static local id. */
	unsigned int local_id;
	/** Service remote id. */
	unsigned int remote_id;
	/** Service kind (0 - NTF, 1 - ONEWAY, 2 - CMD). */
	int kind;
};

struct acpIdMapItf {
	/** SIDL interface name. */
	const char* name;
	/** interface id. */
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
