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

#include <stdint.h>
#include <math.h>
#include "power_reference.h"

#define FULLSCALE INT16_MAX

float calculate_average_rx_power(float rms_squared, float rx_power_reference)
{
  const float rms_fullscale = 0.707 * FULLSCALE;
  return rx_power_reference + 10 * log10(rms_squared / (rms_fullscale * rms_fullscale));
}

void get_amp_and_power_reference(float requested_power_per_subcarrier, int amp_backoff_db, uint16_t *amp, float *new_power_reference)
{
  *new_power_reference = requested_power_per_subcarrier + amp_backoff_db;
  *amp = (int16_t)fmax((FULLSCALE / pow(10.0, amp_backoff_db / 20.0)), 128);
}
