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

#include "openair3/SECU/aes_128_ctr.h"
#include "common/utils/assertions.h"

#include "nr_pdcp_security_nea2.h"

#include <arpa/inet.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

void *nr_pdcp_security_nea2_init(unsigned char *ciphering_key)
{
  // This is a hack, IMO init, cipher and free functions should be reduced to cipher.
  // Test show a ~10% more processing time
  return ciphering_key;
}

void nr_pdcp_security_nea2_cipher(void *security_context, unsigned char *buffer, int length, int bearer, int count, int direction)
{
  unsigned char t[16] = {0};

  t[0] = (count >> 24) & 255;
  t[1] = (count >> 16) & 255;
  t[2] = (count >>  8) & 255;
  t[3] = (count      ) & 255;
  t[4] = ((bearer-1) << 3) | (direction << 2);

  memcpy(buffer, out, length);
}

void nr_pdcp_security_nea2_free_security(void *security_context)
{
  (void)security_context;
}
