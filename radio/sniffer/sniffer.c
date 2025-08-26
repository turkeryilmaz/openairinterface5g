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

#include "common_lib.h"
#include "common/utils/LOG/log.h"
#include "frequency_offset.h"

typedef struct {
  FILE *f;
  char *name;
  uint64_t next_sample;
  int samplerate;
  int delay;        /* delay to apply in trx_sniffer_read(), unit: microsecond */
  float gain;
  bool s16;
  bool do_resample;
  int resampler_interpolation;
  int resampler_decimation;
  bool do_frequency_offset;
  int frequency_offset;
  float complex *freq_offset_table;
  int freq_offset_pos;
  /* for debugging */
  FILE *trace;
  char *tracename;
} file_replay_t;

static int trx_sniffer_get_stats(openair0_device *device)
{
  LOG_I(HW, "trx_sniffer_get_stats() called, not implemented\n");
  return 0;
}

static int trx_sniffer_reset_stats(openair0_device *device)
{
  LOG_I(HW, "trx_sniffer_reset_stats() called, not implemented\n");
  return 0;
}

static int trx_sniffer_stop(openair0_device *device)
{
  LOG_I(HW, "trx_sniffer_stop() called, not implemented\n");
  return 0;
}

static int trx_sniffer_set_freq(openair0_device *device, openair0_config_t *openair0_cfg)
{
  LOG_I(HW, "trx_sniffer_set_freq() called, not implemented\n");
  return 0;
}

static int trx_sniffer_set_gains(openair0_device *device, openair0_config_t *openair0_cfg)
{
  LOG_I(HW, "trx_sniffer_set_gains() called, not implemented\n");
  return 0;
}

static int trx_sniffer_start(openair0_device *device)
{
  return 0;
}

static void trx_sniffer_end(openair0_device *device)
{
}

static int trx_sniffer_write(openair0_device *device, openair0_timestamp timestamp, void **buff, int nsamps, int cc, int flags)
{
  return nsamps;
}

static int16_t to_short(float f, float gain)
{
  /* for srs: 10000, for oai: 60000 */
  float v = f * gain;
  if (v < -32767) v = 32767;
  if (v > 32767) v = 32767;
  return v;
}

static int trx_sniffer_read(openair0_device *device, openair0_timestamp *ptimestamp, void **buff, int nsamps, int cc)
{
  DevAssert(cc == 1);
  file_replay_t *r = device->priv;
  *ptimestamp = r->next_sample;
  r->next_sample += nsamps;
  short *out = buff[0];
  if (!r->s16) {
    float in[nsamps * 2];
    memset(in, 0, sizeof(float) * 2 * nsamps);
    fread(in, nsamps * 2 * sizeof(float), 1, r->f);
    for (int i = 0; i < nsamps * 2; i++) out[i] = to_short(in[i], r->gain);
  } else {
    memset(out, 0, 2 * 2 * nsamps);
    fread(out, nsamps * 2 * 2, 1, r->f);
    for (int i = 0; i < nsamps * 2; i++) {
      float v = out[i] * r->gain;
      if (v < -32767) v = 32767;
      if (v > 32767) v = 32767;
      out[i] = v;
    }
  }
  if (r->do_frequency_offset) {
    float complex in[nsamps];
    for (int i = 0; i < nsamps; i++) in[i] = out[i*2] + I * out[i*2+1];
    r->freq_offset_pos = frequency_offset(r->freq_offset_table, 3072000, r->freq_offset_pos, in, nsamps, r->frequency_offset);
    for (int i = 0; i < nsamps; i++) {
      out[i*2] = crealf(in[i]);
      out[i*2+1] = cimagf(in[i]);
    }
  }
  if (r->trace) {
    fwrite(out, nsamps*4, 1, r->trace);
    fflush(r->trace);
  }
  usleep(r->delay);
  return nsamps;
}

int device_init(openair0_device *device, openair0_config_t *openair0_cfg) {
  device->openair0_cfg = openair0_cfg;
  device->trx_start_func = trx_sniffer_start;
  device->trx_get_stats_func = trx_sniffer_get_stats;
  device->trx_reset_stats_func = trx_sniffer_reset_stats;
  device->trx_end_func = trx_sniffer_end;
  device->trx_stop_func = trx_sniffer_stop;
  device->trx_set_freq_func = trx_sniffer_set_freq;
  device->trx_set_gains_func = trx_sniffer_set_gains;
  device->trx_write_func = trx_sniffer_write;
  device->trx_read_func  = trx_sniffer_read;
  device->type = 3; /* USRP N300 */

  //recplay_conf_t  *c = openair0_cfg->recplay_conf;
  file_replay_t *r = calloc_or_fail(1, sizeof(*r));
  device->priv = r;

  sniffer_configuration_t *s = openair0_cfg->sniffer_conf;
  r->name = strdup(s->filename);
  DevAssert(r->name);
  r->f = fopen(r->name, "r");
  if (!r->f) {
    LOG_E(HW, "could not open file %s: %s\n", r->name, strerror(errno));
    exit(1);
  }
  r->samplerate = openair0_cfg->sample_rate;
  r->delay = s->delay;
  r->gain = s->gain;
  r->s16 = s->s16;
  if (s->trace_filename) {
    r->tracename = strdup(s->trace_filename);
    DevAssert(r->tracename);
    r->trace = fopen(r->tracename, "w");
    if (!r->trace) {
      LOG_E(HW, "cannot create trace file %s: %s\n", r->tracename, strerror(errno));
      exit(1);
    }
  }
  r->do_resample = s->do_resample;
  r->resampler_interpolation = s->resampler_interpolation;
  r->resampler_decimation = s->resampler_decimation;
  r->frequency_offset = s->frequency_offset / r->samplerate * 3072000.;
  r->do_frequency_offset = s->do_frequency_offset && r->frequency_offset;

  if (s->skip) {
    if (r->s16) {
      short b[s->skip * 2];
      fread(b, s->skip * 2 * sizeof(short), 1, r->f);
    } else {
      float b[s->skip * 2];
      fread(b, s->skip * 2 * sizeof(float), 1, r->f);
    }
  }

  if (r->do_frequency_offset)
    r->freq_offset_table = init_frequency_offset(3072000);

  return 0;
}
/*@}*/
