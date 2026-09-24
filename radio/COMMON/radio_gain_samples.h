/* SPDX-License-Identifier: LicenseRef-CSSL-1.0 */
#ifndef OAI_RADIO_GAIN_SAMPLES_H
#define OAI_RADIO_GAIN_SAMPLES_H
#include "radio_gain.h"

#define RADIO_GAIN_SAMPLE_HISTORY 256U
#define RADIO_TX_LEVEL_MAX_SAMPLES 65536U

typedef struct {
  bool valid;
  uint32_t sample_count;
  uint32_t component_full_scale;
  uint64_t sum_squared_components;
  uint32_t peak_component;
  uint32_t over_range_components;
} radio_tx_sample_level_t;

/* Complete selected interleaved sc16 buffer before backend conversion. The
 * component interval is [-full_scale, full_scale-1]. Buffers above the explicit
 * bound are unavailable, never silently subsampled or treated as unclipped. */
radio_tx_sample_level_t radio_tx_sample_level(const int16_t *iq, uint32_t count, uint32_t full_scale);

typedef struct {
  bool present;
  bool valid;
  uint64_t generation;
  double rx_gain_db;
  int64_t first_sample;
  int64_t end_sample;
  /* Bounded, evenly-spaced observations across the raw read, never a calibrated
   * RF measurement or proof that every component was observed. mean_power_fs is
   * E[I^2+Q^2]/component_full_scale^2 and peak_component_fs is
   * max(abs(I), abs(Q))/component_full_scale. near_rail_components counts
   * observed components at or above 0.98 full scale; zero does not certify an
   * unclipped raw read. observation_ns is host CLOCK_MONOTONIC at read return. */
  bool level_valid;
  double mean_power_fs;
  double peak_component_fs;
  uint32_t sampled_components;
  uint32_t near_rail_components;
  uint64_t observation_ns;
} radio_gain_sample_context_t;

typedef struct radio_gain_sample_history radio_gain_sample_history_t;
/* Allocate/prefault before streaming; destroy only after all producers/readers stop. */
radio_gain_sample_history_t *radio_gain_sample_history_create(void);
void radio_gain_sample_history_destroy(radio_gain_sample_history_t *history);
/* One RX reader publishes each returned contiguous range in device sample ticks.
 * Invalid transitions are retained, not converted to a zero gain. */
void radio_gain_sample_publish(radio_gain_sample_history_t *history, const radio_gain_sample_context_t *context);
/* Bounded lookup supports ranges crossing reads, including UE first-symbol
 * lookahead. Delayed consumers receive unavailable if history was overwritten.
 * Level summaries are conservatively aggregated over every intersecting whole
 * read, so their coverage can be larger than the requested sample interval. */
radio_gain_sample_context_t radio_gain_sample_lookup(const radio_gain_sample_history_t *history,
                                                     int64_t first_sample,
                                                     int64_t end_sample);
/* API readback is not a settled-sample guarantee. settled_after is an explicitly
 * qualified device-time boundary supplied by the integration/profile. */
radio_gain_sample_context_t radio_gain_sample_context(const radio_gain_result_t *before,
                                                      const radio_gain_result_t *after,
                                                      int64_t first_sample,
                                                      int64_t end_sample,
                                                      bool settle_boundary_valid,
                                                      int64_t settled_after);
/* True only when a present immutable RX context can be used for gain-normalized
 * measurements. Raw context validity remains unchanged so an overload remains
 * visible to the gain controller. */
bool radio_gain_sample_measurement_valid(const radio_gain_sample_context_t *context);
#endif
