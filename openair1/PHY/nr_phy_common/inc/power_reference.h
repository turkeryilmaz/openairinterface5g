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

#ifndef POWER_REFERENCE_H
#define POWER_REFERENCE_H
#include <stdint.h>

#define TARGET_MINIMUM_AMPLITUDE (1 << 9)
#define TARGET_MAXIMUM_AMPLITUDE (1 << 12)

/// @brief Calculate received power based on rms and rx_power reference
///        This is acheived by comparing a full scale sine wave RMS^2 with
//         RMS^2 of the signal.
//
/// @param rms_squared RMS^2 of the signal (Sum(I^2 + Q^2)/N)
/// @param rx_power_reference power of a 0dBFS single carrier sine wave
///
/// @return received power in dBm
float calculate_average_rx_power(float rms_squared, float rx_power_reference);

/// @brief Calculate amp and power_reference value. This assumes that both can be changed at the same time,
///        even though this is most likely not the case
/// @param requested_power_per_subcarrier Power to transmit per subcarrier
/// @param amp_backoff_db Amp backoff in dB
/// @param amp new amp
/// @param new_power_reference new power reference
void get_amp_and_power_reference(float requested_power_per_subcarrier, int amp_backoff_db, uint16_t *amp, float *new_power_reference);

#endif
