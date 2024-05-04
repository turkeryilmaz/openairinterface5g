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

#include "common/utils/assertions.h"

#include "nr_pdcp_security_nea1.h"
#include "openair3/SECU/secu_defs.h"
#include "openair3/SECU/key_nas_deriver.h"

#include <arpa/inet.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>


void *nr_pdcp_security_nea1_init(unsigned char *ciphering_key)
{
  // This is a hack, IMO init, cipher and free functions should be reduced to cipher.
  // Test show a ~10% more processing time
  return ciphering_key;
}

void nr_pdcp_security_nea1_cipher(void *security_context, unsigned char *buffer, int length, int bearer, int count, int direction)
{
  DevAssert(security_context != NULL);
  DevAssert(buffer != NULL);
  DevAssert(length > 0);
  DevAssert(bearer > -1 && bearer < 32);
  DevAssert(direction > -1 && direction < 2);
  DevAssert(count > -1);

  uint8_t *ciphering_key = (uint8_t *)security_context;
  uint8_t out[length];
  memset(out, 0, length);

  nas_stream_cipher_t stream_cipher;

  stream_cipher.key_length = 16;
  stream_cipher.count      = count;
  stream_cipher.bearer     = bearer - 1;
  stream_cipher.direction  = direction;

  stream_cipher.key        = ciphering_key;
  stream_cipher.message    = buffer;
  /* length in bits */
  stream_cipher.blength    = length << 3;

  // out will be set to stream_cipher.message, no need to free it
  stream_compute_encrypt(EEA1_128_ALG_ID, &stream_cipher, out);

  memmove(buffer, out, length);
}

void nr_pdcp_security_nea1_free_security(void *security_context)
{
  (void)security_context;
}
