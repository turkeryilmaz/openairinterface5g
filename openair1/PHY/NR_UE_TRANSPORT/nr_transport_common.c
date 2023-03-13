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

#include "PHY/defs_UE.h"
#include "PHY/phy_extern_ue.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_proto_ue.h"
#include "PHY/CODING/coding_defs.h"
#include "PHY/CODING/coding_extern.h"
#include "PHY/CODING/lte_interleaver_inline.h"
#include "PHY/CODING/nrLDPC_extern.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_ue.h"
#include "common/utils/LOG/vcd_signal_dumper.h"
#include <openair2/UTIL/OPT/opt.h>

void nr_attach_crc_to_payload(unsigned char *in, uint8_t *out, int max_payload_bytes, uint32_t in_size, uint32_t *out_size) {

    unsigned int crc = 1;
    if (in_size > NR_MAX_PSSCH_TBS) {
      // Add 24-bit crc (polynomial A) to payload
      crc = crc24a(in, in_size) >> 8;
      in[in_size >> 3] = ((uint8_t*)&crc)[2];
      in[1 + (in_size >> 3)] = ((uint8_t*)&crc)[1];
      in[2 + (in_size >> 3)] = ((uint8_t*)&crc)[0];
      *out_size = in_size + 24;

      AssertFatal((in_size / 8) + 4 <= max_payload_bytes,
                  "A %d is too big (A / 8 + 4 = %d > %d)\n", in_size, (in_size / 8) + 4, max_payload_bytes);

      memcpy(out, in, (in_size / 8) + 4);
    } else {
      // Add 16-bit crc (polynomial A) to payload
      crc = crc16(in, in_size) >> 16;
      in[in_size >> 3] = ((uint8_t*)&crc)[1];
      in[1 + (in_size >> 3)] = ((uint8_t*)&crc)[0];
      *out_size = in_size + 16;

      AssertFatal((in_size / 8) + 3 <= max_payload_bytes,
                  "A %d is too big (A / 8 + 3 = %d > %d)\n", in_size, (in_size / 8) + 3, max_payload_bytes);

      memcpy(out, in, (in_size / 8) + 3);  // using 3 bytes to mimic the case of 24 bit crc
    }
}
