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

#ifndef RADIO_COMMON_SNIFFER_CONFIG_H
#define RADIO_COMMON_SNIFFER_CONFIG_H

#include <stdbool.h>

#define SNIFFER_SECTION "device.sniffer"

#define SNIFFER_ENABLED_PARAMS_DESC { \
  { "sniffer", "use UE as a sniffer\n", PARAMFLAG_BOOL, .iptr = &do_sniff, .defintval = 0, TYPE_INT, 0 } \
}

#define SNIFFER_PARAMS_DESC { \
  { "file", "IQ file to read samples from\n", 0, .strptr = &filename, .defstrval = 0, TYPE_STRING, 0 }, \
  { "gain", "gain to apply to input signal\n", 0, .dblptr = &gain, .defdblval = 1, TYPE_DOUBLE, 0 }, \
  { "s16", "input signal is s16, not float\n", PARAMFLAG_BOOL, .iptr = &s16, .defintval = 0, TYPE_INT, 0 }, \
  { "resampler-interpolation", "resampler interpolation value\n", 0, .iptr = &resampler_interpolation, .defintval = -1, TYPE_INT, 0 }, \
  { "resampler-decimation", "resampler decimation value\n", 0, .iptr = &resampler_decimation, .defintval = -1, TYPE_INT, 0 }, \
  { "frequency-offset", "frequency offset (Hz) to apply to input signal\n", 0, .dblptr = &frequency_offset, .defdblval = 0, TYPE_DOUBLE, 0 }, \
  { "delay", "delay to apply in trx_sniffer_read()\n", 0, .iptr = &delay, .defintval = 0, TYPE_INT, 0 }, \
  { "trace-file", "IQ file to write samples to in trx_sniffer_read()\n", 0, .strptr = &trace_filename, .defstrval = 0, TYPE_STRING, 0 }, \
  { "skip", "number of samples to skip at startup\n", 0, .iptr = &skip, .defintval = 0, TYPE_INT, 0 }, \
}
typedef struct {
  char *filename;
  int delay;      /* delay to apply in trx_sniffer_read(), unit: microsecond */
  float gain;
  bool s16;
  bool do_resample;
  int resampler_interpolation;
  int resampler_decimation;
  bool do_frequency_offset;
  float frequency_offset;
  char *trace_filename;
  int skip;
} sniffer_configuration_t;

sniffer_configuration_t *read_sniffer_configuration(void);

#endif /* RADIO_COMMON_SNIFFER_CONFIG_H */
