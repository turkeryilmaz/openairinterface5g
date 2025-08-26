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

#include "sniffer_config.h"

#include "common/utils/LOG/log.h"
#include "common/config/config_userapi.h"

sniffer_configuration_t *read_sniffer_configuration(void)
{
  /* nothing to configure if the sniffer mode is not activated */
  int do_sniff = 0;

  paramdef_t sniffer_enabled_params[] = SNIFFER_ENABLED_PARAMS_DESC;

  config_get(config_get_if(), sniffer_enabled_params, sizeofArray(sniffer_enabled_params), NULL);

  if (!do_sniff)
    return 0;

  /* get parameters */
  char *filename = 0;
  double gain = 1;
  int s16 = false;
  int resampler_interpolation = -1;
  int resampler_decimation = -1;
  double frequency_offset = 0;
  int delay = 0;
  char *trace_filename = 0;
  int skip = 0;

  paramdef_t sniffer_params[] = SNIFFER_PARAMS_DESC;

  config_get(config_get_if(), sniffer_params, sizeofArray(sniffer_params), SNIFFER_SECTION);

  /* no configuration if filename is not given
   * sniffer works in realtime
   */
  if (!filename)
    return 0;

  if (!s16) gain *= 32767;

  LOG_I(HW, "sniffer configuration:\n");
  LOG_I(HW, "  filename '%s'\n", filename);
  LOG_I(HW, "  gain %g\n", gain);
  LOG_I(HW, "  s16 %d\n", s16);
  LOG_I(HW, "  resampler_interpolation %d\n", resampler_interpolation);
  LOG_I(HW, "  resampler_decimation %d\n", resampler_decimation);
  LOG_I(HW, "  frequency_offset %g\n", frequency_offset);
  LOG_I(HW, "  delay %d\n", delay);
  LOG_I(HW, "  trace_filename '%s'\n", trace_filename);
  LOG_I(HW, "  skip %d\n", skip);

  AssertFatal((resampler_interpolation == -1 && resampler_decimation == -1)
              || (resampler_interpolation != -1 && resampler_decimation != -1),
             "sniffer: both (or none) resampler_interpolation and resampler_decimation must be passed\n");

  sniffer_configuration_t *ret = calloc_or_fail(1, sizeof(*ret));

  ret->filename = filename;
  ret->delay = delay;
  ret->gain = gain;
  ret->s16 = s16;
  ret->do_resample = resampler_interpolation != -1;
  ret->resampler_interpolation = resampler_interpolation;
  ret->resampler_decimation = resampler_decimation;
  ret->do_frequency_offset = frequency_offset != 0;
  ret->frequency_offset = frequency_offset;
  ret->trace_filename = trace_filename;
  ret->skip = skip;

  return ret;
}
