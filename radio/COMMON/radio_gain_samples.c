/* SPDX-License-Identifier: LicenseRef-CSSL-1.0 */
#include "radio_gain_samples.h"
#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  _Atomic(uint64_t) sequence;
  _Atomic(uint64_t) generation;
  _Atomic(uint64_t) gain_bits;
  _Atomic(int64_t) first;
  _Atomic(int64_t) end;
  _Atomic(bool) valid;
  _Atomic(bool) level_valid;
  _Atomic(uint64_t) mean_power_bits;
  _Atomic(uint64_t) peak_component_bits;
  _Atomic(uint32_t) sampled_components;
  _Atomic(uint32_t) near_rail_components;
  _Atomic(uint64_t) observation_ns;
} sample_entry_t;

struct radio_gain_sample_history {
  _Atomic(uint64_t) written;
  sample_entry_t entries[RADIO_GAIN_SAMPLE_HISTORY];
};

static void radio_gain_sample_prefault(void *memory, size_t size)
{
  /* This runs before any atomic object is initialized. Do not write raw bytes
   * to an atomic representation after initialization. */
  volatile unsigned char *bytes = memory;
  for (size_t offset = 0; offset < size; offset += 4096U)
    bytes[offset] = 0;
  if (size != 0)
    bytes[size - 1] = 0;
}

static void radio_gain_sample_entry_init(sample_entry_t *entry)
{
  atomic_init(&entry->sequence, 0);
  atomic_init(&entry->generation, 0);
  atomic_init(&entry->gain_bits, 0);
  atomic_init(&entry->first, 0);
  atomic_init(&entry->end, 0);
  atomic_init(&entry->valid, false);
  atomic_init(&entry->level_valid, false);
  atomic_init(&entry->mean_power_bits, 0);
  atomic_init(&entry->peak_component_bits, 0);
  atomic_init(&entry->sampled_components, 0);
  atomic_init(&entry->near_rail_components, 0);
  atomic_init(&entry->observation_ns, 0);
}

static bool radio_gain_sample_entry_lock_free(const sample_entry_t *entry)
{
  return atomic_is_lock_free(&entry->sequence) && atomic_is_lock_free(&entry->generation) && atomic_is_lock_free(&entry->gain_bits)
         && atomic_is_lock_free(&entry->first) && atomic_is_lock_free(&entry->end) && atomic_is_lock_free(&entry->valid)
         && atomic_is_lock_free(&entry->level_valid) && atomic_is_lock_free(&entry->mean_power_bits)
         && atomic_is_lock_free(&entry->peak_component_bits) && atomic_is_lock_free(&entry->sampled_components)
         && atomic_is_lock_free(&entry->near_rail_components) && atomic_is_lock_free(&entry->observation_ns);
}

radio_gain_sample_history_t *radio_gain_sample_history_create(void)
{
  radio_gain_sample_history_t *history = malloc(sizeof(*history));
  if (!history)
    return NULL;
  radio_gain_sample_prefault(history, sizeof(*history));
  atomic_init(&history->written, 0);
  for (size_t index = 0; index < RADIO_GAIN_SAMPLE_HISTORY; ++index)
    radio_gain_sample_entry_init(&history->entries[index]);

  if (!atomic_is_lock_free(&history->written)) {
    free(history);
    return NULL;
  }
  for (size_t index = 0; index < RADIO_GAIN_SAMPLE_HISTORY; ++index) {
    if (!radio_gain_sample_entry_lock_free(&history->entries[index])) {
      free(history);
      return NULL;
    }
  }
  return history;
}

void radio_gain_sample_history_destroy(radio_gain_sample_history_t *history)
{
  free(history);
}

static bool level_summary_valid(const radio_gain_sample_context_t *context)
{
  if (!context || !context->level_valid || !isfinite(context->mean_power_fs) || context->mean_power_fs < 0
      || context->mean_power_fs > 2 || !isfinite(context->peak_component_fs) || context->peak_component_fs < 0
      || context->peak_component_fs > 1 || context->sampled_components == 0 || (context->sampled_components & 1U) != 0
      || context->near_rail_components > context->sampled_components
      || context->mean_power_fs > 2 * context->peak_component_fs * context->peak_component_fs)
    return false;
  if ((context->mean_power_fs == 0) != (context->peak_component_fs == 0))
    return false;
  if ((context->near_rail_components != 0 && context->peak_component_fs < 0.98)
      || (context->peak_component_fs >= 0.98 && context->near_rail_components == 0))
    return false;
  return true;
}

bool radio_gain_sample_measurement_valid(const radio_gain_sample_context_t *context)
{
  return context && context->present && context->valid && isfinite(context->rx_gain_db) && level_summary_valid(context)
         && context->near_rail_components == 0;
}

void radio_gain_sample_publish(radio_gain_sample_history_t *history, const radio_gain_sample_context_t *context)
{
  if (!history || !context || context->end_sample <= context->first_sample)
    return;
  const uint64_t n = atomic_load_explicit(&history->written, memory_order_relaxed);
  /* Sequence wrap is not a meaningful supported process lifetime. */
  if (n >= UINT64_MAX / 2 - 1)
    return;
  sample_entry_t *entry = &history->entries[n % RADIO_GAIN_SAMPLE_HISTORY];
  uint64_t gain_bits;
  uint64_t mean_power_bits;
  uint64_t peak_component_bits;
  memcpy(&gain_bits, &context->rx_gain_db, sizeof(gain_bits));
  memcpy(&mean_power_bits, &context->mean_power_fs, sizeof(mean_power_bits));
  memcpy(&peak_component_bits, &context->peak_component_fs, sizeof(peak_component_bits));
  const bool level_valid = level_summary_valid(context);
  /* All payload fields are atomic. A bounded reader can race ring overwrite
   * without either a C data race or a speculative plain-struct snapshot. */
  atomic_store(&entry->sequence, n * 2 + 1);
  atomic_store(&entry->generation, context->generation);
  atomic_store(&entry->gain_bits, gain_bits);
  atomic_store(&entry->first, context->first_sample);
  atomic_store(&entry->end, context->end_sample);
  atomic_store(&entry->valid, context->present && context->valid && isfinite(context->rx_gain_db));
  atomic_store(&entry->level_valid, level_valid);
  atomic_store(&entry->mean_power_bits, mean_power_bits);
  atomic_store(&entry->peak_component_bits, peak_component_bits);
  atomic_store(&entry->sampled_components, context->sampled_components);
  atomic_store(&entry->near_rail_components, context->near_rail_components);
  atomic_store(&entry->observation_ns, context->observation_ns);
  atomic_store(&entry->sequence, n * 2 + 2);
  atomic_store_explicit(&history->written, n + 1, memory_order_release);
}

static bool read_entry(const sample_entry_t *entry, uint64_t index, radio_gain_sample_context_t *result)
{
  const uint64_t expected = index * 2 + 2;
  if (atomic_load(&entry->sequence) != expected)
    return false;
  result->generation = atomic_load(&entry->generation);
  const uint64_t bits = atomic_load(&entry->gain_bits);
  memcpy(&result->rx_gain_db, &bits, sizeof(bits));
  result->first_sample = atomic_load(&entry->first);
  result->end_sample = atomic_load(&entry->end);
  result->valid = atomic_load(&entry->valid);
  result->level_valid = atomic_load(&entry->level_valid);
  const uint64_t mean_power_bits = atomic_load(&entry->mean_power_bits);
  const uint64_t peak_component_bits = atomic_load(&entry->peak_component_bits);
  memcpy(&result->mean_power_fs, &mean_power_bits, sizeof(mean_power_bits));
  memcpy(&result->peak_component_fs, &peak_component_bits, sizeof(peak_component_bits));
  result->sampled_components = atomic_load(&entry->sampled_components);
  result->near_rail_components = atomic_load(&entry->near_rail_components);
  result->observation_ns = atomic_load(&entry->observation_ns);
  result->present = true;
  return atomic_load(&entry->sequence) == expected;
}

radio_gain_sample_context_t radio_gain_sample_lookup(const radio_gain_sample_history_t *history,
                                                     int64_t first_sample,
                                                     int64_t end_sample)
{
  radio_gain_sample_context_t result = {.present = history != NULL, .first_sample = first_sample, .end_sample = end_sample};
  if (!history || end_sample <= first_sample)
    return result;

  const uint64_t written = atomic_load_explicit(&history->written, memory_order_acquire);
  const uint64_t count = written < RADIO_GAIN_SAMPLE_HISTORY ? written : RADIO_GAIN_SAMPLE_HISTORY;
  int64_t uncovered_end = end_sample;
  bool range_covered = false;
  bool range_impossible = false;
  bool gain_consistent = true;
  bool have_gain = false;
  bool level_consistent = true;
  bool have_level = false;
  uint64_t total_components = 0;
  uint64_t total_near_rail = 0;
  double weighted_mean_sum = 0;
  double peak_component_fs = 0;
  uint64_t newest_observation_ns = 0;

  for (uint64_t offset = 0; offset < count; ++offset) {
    const uint64_t index = written - offset - 1;
    radio_gain_sample_context_t entry;
    if (!read_entry(&history->entries[index % RADIO_GAIN_SAMPLE_HISTORY], index, &entry))
      return result;

    const bool intersects = entry.first_sample < end_sample && entry.end_sample > first_sample;
    if (intersects) {
      if (!level_summary_valid(&entry) || UINT64_MAX - total_components < entry.sampled_components
          || UINT64_MAX - total_near_rail < entry.near_rail_components) {
        level_consistent = false;
      } else {
        const double weighted_mean = entry.mean_power_fs * entry.sampled_components;
        if (!isfinite(weighted_mean) || !isfinite(weighted_mean_sum + weighted_mean)) {
          level_consistent = false;
        } else {
          total_components += entry.sampled_components;
          total_near_rail += entry.near_rail_components;
          weighted_mean_sum += weighted_mean;
          if (entry.peak_component_fs > peak_component_fs)
            peak_component_fs = entry.peak_component_fs;
          if (entry.observation_ns > newest_observation_ns)
            newest_observation_ns = entry.observation_ns;
          have_level = true;
        }
      }
    }

    if (range_covered || range_impossible || entry.first_sample >= uncovered_end)
      continue;
    if (entry.end_sample < uncovered_end || entry.end_sample <= entry.first_sample) {
      range_impossible = true;
      continue;
    }
    if (!entry.valid) {
      gain_consistent = false;
    } else if (!have_gain) {
      result.generation = entry.generation;
      result.rx_gain_db = entry.rx_gain_db;
      have_gain = true;
    } else if (result.generation != entry.generation || result.rx_gain_db != entry.rx_gain_db) {
      gain_consistent = false;
    }

    if (entry.first_sample <= first_sample) {
      range_covered = true;
    } else {
      uncovered_end = entry.first_sample;
    }
  }

  if (!range_covered || range_impossible)
    return result;
  if (gain_consistent && have_gain)
    result.valid = true;
  if (level_consistent && have_level && total_components != 0 && total_components <= UINT32_MAX && total_near_rail <= UINT32_MAX
      && total_near_rail <= total_components) {
    const double mean_power_fs = weighted_mean_sum / total_components;
    result.level_valid = isfinite(mean_power_fs) && mean_power_fs >= 0 && mean_power_fs <= 2 && isfinite(peak_component_fs)
                         && peak_component_fs >= 0 && peak_component_fs <= 1
                         && mean_power_fs <= 2 * peak_component_fs * peak_component_fs;
    if (result.level_valid) {
      result.mean_power_fs = mean_power_fs;
      result.peak_component_fs = peak_component_fs;
      result.sampled_components = total_components;
      result.near_rail_components = total_near_rail;
      result.observation_ns = newest_observation_ns;
    }
  }
  return result;
}

radio_gain_sample_context_t radio_gain_sample_context(const radio_gain_result_t *before,
                                                      const radio_gain_result_t *after,
                                                      int64_t first_sample,
                                                      int64_t end_sample,
                                                      bool settle_boundary_valid,
                                                      int64_t settled_after)
{
  radio_gain_sample_context_t result = {.present = true, .first_sample = first_sample, .end_sample = end_sample};
  if (!before || !after)
    return result;
  result.generation = after->generation;
  result.rx_gain_db = after->reported_rx_db;
  result.valid = end_sample > first_sample && before->generation == after->generation && before->rx_gain_valid
                 && after->rx_gain_valid && isfinite(after->reported_rx_db) && before->reported_rx_db == after->reported_rx_db
                 && settle_boundary_valid && first_sample >= settled_after;
  return result;
}

radio_tx_sample_level_t radio_tx_sample_level(const int16_t *iq, uint32_t count, uint32_t full_scale)
{
  radio_tx_sample_level_t result = {0};
  if (!iq || count == 0 || count > RADIO_TX_LEVEL_MAX_SAMPLES || full_scale == 0 || full_scale > 32768U)
    return result;
  result.valid = true;
  result.sample_count = count;
  result.component_full_scale = full_scale;
  for (uint32_t index = 0; index < 2 * count; ++index) {
    const int32_t sample = iq[index];
    const uint32_t magnitude = sample < 0 ? -sample : sample;
    result.sum_squared_components += (uint64_t)((int64_t)sample * sample);
    if (magnitude > result.peak_component)
      result.peak_component = magnitude;
    result.over_range_components += sample < -(int32_t)full_scale || sample >= (int32_t)full_scale;
  }
  return result;
}
