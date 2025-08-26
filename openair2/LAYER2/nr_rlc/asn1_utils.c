/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
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
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

#include "nr_rlc_asn1_utils.h"
#include "rlc.h"
#include "nr_rlc_configuration.h"

#define ENCODE_DECODE(a, b)                         \
int decode_##a(int v)                               \
{                                                   \
  static const int tab[] = { VALUES_NR_RLC_##b };   \
  AssertFatal(v >= 0 && v < SIZEOF_NR_RLC_##b,      \
              "bad encoded value " #a " %d\n", v);  \
  return tab[v];                                    \
}                                                   \
                                                    \
int encode_##a(int v)                               \
{                                                   \
  static const int tab[] = { VALUES_NR_RLC_##b };   \
  for (int ret = 0; ret < SIZEOF_NR_RLC_##b; ret++) \
    if (tab[ret] == v)                              \
      return ret;                                   \
  AssertFatal(0, "bad " #a " value %d\n", v);       \
}

ENCODE_DECODE(t_reassembly, T_REASSEMBLY)
ENCODE_DECODE(t_status_prohibit, T_STATUS_PROHIBIT)
ENCODE_DECODE(t_poll_retransmit, T_POLL_RETRANSMIT)
ENCODE_DECODE(poll_pdu, POLL_PDU)
ENCODE_DECODE(poll_byte, POLL_BYTE)
ENCODE_DECODE(max_retx_threshold, MAX_RETX_THRESHOLD)
ENCODE_DECODE(sn_field_length_am, SN_FIELD_LENGTH_AM)
ENCODE_DECODE(sn_field_length_um, SN_FIELD_LENGTH_UM)
