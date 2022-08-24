/*
 *****************************************************************
 *
 * Module : ACP TTCN mapper
 * Purpose: TTCN to ACP mapper
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

// Internal includes
#include "attcn.hh"

// TTCN includes
#if defined(PROJECT_HAS_RAT_EUTRA)
#include "EUTRA_CommonDefs.hh"
#endif // defined(PROJECT_HAS_RAT_EUTRA)
#if defined(PROJECT_HAS_RAT_NR)
#include "NR_CommonDefs.hh"
#endif // defined(PROJECT_HAS_RAT_NR)

// Internal includes
#if defined(PROJECT_HAS_RAT_EUTRA)
#include "SIDL_NASEMU_EUTRA_SYSTEM_PORT.h"
#endif // defined(PROJECT_HAS_RAT_EUTRA)
#if defined(PROJECT_HAS_RAT_NR)
#include "SIDL_NASEMU_NR_SYSTEM_PORT.h"
#endif // defined(PROJECT_HAS_RAT_NR)

#if defined(PROJECT_HAS_RAT_EUTRA)

void attcnConvert(EUTRA__CommonDefs::RRC__MSG__Request__Type& dst, const RRC_MSG_Request_Type& src)
{
	// TODO
}

void attcnConvert(EUTRA__CommonDefs::RRC__MSG__Indication__Type& dst, const RRC_MSG_Indication_Type& src)
{
	// TODO
}

void attcnConvert(RRC_MSG_Request_Type& src, const EUTRA__CommonDefs::RRC__MSG__Request__Type& dst)
{
	// TODO
}

void attcnConvert(RRC_MSG_Indication_Type& src, const EUTRA__CommonDefs::RRC__MSG__Indication__Type& dst)
{
	// TODO
}

#endif // defined(PROJECT_HAS_RAT_EUTRA)

#if defined(PROJECT_HAS_RAT_NR)

void attcnConvert(NR__CommonDefs::NR__RRC__MSG__Request__Type& dst, const NR_RRC_MSG_Request_Type& src)
{
	// TODO
}

void attcnConvert(NR__CommonDefs::NR__RRC__MSG__Indication__Type& dst, const NR_RRC_MSG_Indication_Type& src)
{
	// TODO
}

void attcnConvert(NR_RRC_MSG_Request_Type& src, const NR__CommonDefs::NR__RRC__MSG__Request__Type& dst)
{
	// TODO
}

void attcnConvert(NR_RRC_MSG_Indication_Type& src, const NR__CommonDefs::NR__RRC__MSG__Indication__Type& dst)
{
	// TODO
}

#endif // defined(PROJECT_HAS_RAT_NR)
