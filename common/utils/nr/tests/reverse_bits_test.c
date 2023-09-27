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

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include "common/utils/assertions.h"
#include "../nr_common.c"

int main()
{
  uint64_t in = 0x1;
  uint64_t out = reverse_bits(in, 0);
  if (out != 0x0)
    return -1;

  in = 0x1;
  out = reverse_bits(in, 1);
  if (out != 0x1)
    return -1;

  in = 0x1;
  out = reverse_bits(in, 64);
  if (out != 0x8000000000000000)
    return -1;

  in = 0x20F;
  out = reverse_bits(in, 10);
  if (out != 0x3C1)
    return -1;

  return 0;
}
