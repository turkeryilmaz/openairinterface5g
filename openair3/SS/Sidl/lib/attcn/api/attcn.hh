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

#pragma once

// System includes
#include <TTCN3.hh>

// --- Basic TTCN conversions

template<typename T1, typename T2>
void attcnConvert(T1& dst, const T2& src)
{
	dst = src;
}

template<typename T1>
void attcnConvert(T1& dst, const Enum_Type& src)
{
	dst = static_cast<T1>(src.as_int());
}

void attcnConvert(ASN_NULL& dst, const bool& src);
void attcnConvert(bool& dst, const ASN_NULL& src);

void attcnConvert(OCTETSTRING_ELEMENT&& dst, const unsigned char& src);
void attcnConvert(unsigned char& dst, const OCTETSTRING_ELEMENT& src);

void attcnConvert(BITSTRING_ELEMENT&& dst, const unsigned char& src);
void attcnConvert(unsigned char& dst, const BITSTRING_ELEMENT& src);

void attcnConvert(CHARSTRING_ELEMENT&& dst, const char& src);
void attcnConvert(char& dst, const CHARSTRING_ELEMENT& src);

// --- SysSrb internal conversions

// Forward declarations
namespace EUTRA__CommonDefs {
class RRC__MSG__Request__Type;
class RRC__MSG__Indication__Type;
} // namespace EUTRA__CommonDefs
class RRC_MSG_Request_Type;
class RRC_MSG_Indication_Type;

void attcnConvert(EUTRA__CommonDefs::RRC__MSG__Request__Type& dst, const RRC_MSG_Request_Type& src);
void attcnConvert(EUTRA__CommonDefs::RRC__MSG__Indication__Type& dst, const RRC_MSG_Indication_Type& src);
void attcnConvert(RRC_MSG_Request_Type& src, const EUTRA__CommonDefs::RRC__MSG__Request__Type& dst);
void attcnConvert(RRC_MSG_Indication_Type& src, const EUTRA__CommonDefs::RRC__MSG__Indication__Type& dst);

// --- NrSysSrb internal conversions

// Forward declarations
namespace NR__CommonDefs {
class NR__RRC__MSG__Request__Type;
class NR__RRC__MSG__Indication__Type;
} // namespace NR__CommonDefs
class NR_RRC_MSG_Request_Type;
class NR_RRC_MSG_Indication_Type;
void attcnConvert(NR__CommonDefs::NR__RRC__MSG__Request__Type& dst, const NR_RRC_MSG_Request_Type& src);
void attcnConvert(NR__CommonDefs::NR__RRC__MSG__Indication__Type& dst, const NR_RRC_MSG_Indication_Type& src);
void attcnConvert(NR_RRC_MSG_Request_Type& src, const NR__CommonDefs::NR__RRC__MSG__Request__Type& dst);
void attcnConvert(NR_RRC_MSG_Indication_Type& src, const NR__CommonDefs::NR__RRC__MSG__Indication__Type& dst);
