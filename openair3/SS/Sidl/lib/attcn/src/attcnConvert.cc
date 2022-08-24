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
#include "attcnConvert.hh"

void attcnConvert(ASN_NULL& dst, const bool& src)
{
	dst = src ? ASN_NULL(ASN_NULL_VALUE) : ASN_NULL();
}

void attcnConvert(bool& dst, const ASN_NULL& src)
{
	dst = src.is_value();
}

void attcnConvert(OCTETSTRING_ELEMENT&& dst, const unsigned char& src)
{
	dst = OCTETSTRING(1, &src);
}

void attcnConvert(unsigned char& dst, const OCTETSTRING_ELEMENT& src)
{
	dst = src.get_octet();
}

void attcnConvert(BITSTRING_ELEMENT&& dst, const unsigned char& src)
{
	dst = BITSTRING(1, &src);
}

void attcnConvert(unsigned char& dst, const BITSTRING_ELEMENT& src)
{
	dst = src.get_bit() ? 1 : 0;
}

void attcnConvert(CHARSTRING_ELEMENT&& dst, const char& src)
{
	dst = CHARSTRING(1, &src);
}

void attcnConvert(char& dst, const CHARSTRING_ELEMENT& src)
{
	dst = src.get_char();
}
