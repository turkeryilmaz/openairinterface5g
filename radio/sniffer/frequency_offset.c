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

#include <math.h>
#include <complex.h>

#include "common_lib.h"

float complex *init_frequency_offset(int table_size)
{
  float complex *table = malloc_or_fail(table_size * sizeof(float complex));
  int i;

  for (i = 0; i < table_size; i++)
    table[i] = cexpf(I * 2 * M_PI * i / table_size);

  return table;
}

/* returns next cur_pos to use */
int frequency_offset(float complex *table, int samplerate,
                     int cur_pos,
                     float complex *in, int in_size,
                     int freq_offset)
{
  int i;

  for (i = 0; i < in_size; i++) {
    in[i] *= table[cur_pos];
    cur_pos -= freq_offset;
    if (cur_pos < 0) cur_pos += samplerate;
    if (cur_pos > samplerate) cur_pos -= samplerate;
  }

  return cur_pos;
}
