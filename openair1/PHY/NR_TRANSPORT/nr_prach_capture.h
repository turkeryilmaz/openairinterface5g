/* SPDX-License-Identifier: LicenseRef-CSSL-1.0 */
#ifndef NR_PRACH_CAPTURE_H
#define NR_PRACH_CAPTURE_H

#include <stdbool.h>
#include <stdint.h>
#include <limits.h>

/* One RU producer owns this span. It describes outer RF reads, not internal
 * backend fragments or a sample-exact analog settling guarantee. */
typedef struct {
  bool valid;
  int frame;
  int64_t zero_tick;
  int64_t buffer_first;
  int64_t buffer_end;
  int64_t read_end;
  uint64_t sequence;
} nr_prach_rx_span_t;

typedef struct {
  int64_t token; /* Zero is unavailable; positive tokens are process-lifetime unique. */
  int64_t buffer_first;
  int ncp;
  int dftlen;
  int reps;
  int n_ta_offset;
  int k;
  uint16_t first_antenna;
  uint16_t antenna_count;
  bool valid;
} nr_prach_capture_layout_t;

static inline void nr_prach_capture_read(nr_prach_rx_span_t *span,
                                         int frame,
                                         int64_t offset,
                                         int64_t requested,
                                         int received,
                                         int64_t tick,
                                         int64_t frame_samples,
                                         bool discarded)
{
  if (discarded || frame < 0 || frame > 1023 || offset < 0 || requested <= 0 || received != requested || tick < 0
      || frame_samples < requested || offset > frame_samples - requested || tick > INT64_MAX - requested) {
    span->valid = false;
    return;
  }
  const int64_t zero_tick = tick - offset;
  if (!span->valid || span->frame != frame || span->buffer_end != offset || span->read_end != tick
      || span->zero_tick != zero_tick) {
    span->buffer_first = offset;
  }
  span->valid = true;
  span->frame = frame;
  span->zero_tick = zero_tick;
  span->buffer_end = offset + requested;
  span->read_end = tick + requested;
}

static inline bool nr_prach_capture_window(const nr_prach_rx_span_t *span,
                                           int frame,
                                           int64_t offset,
                                           int64_t count,
                                           int64_t *first,
                                           int64_t *end)
{
  *first = INT64_MIN;
  *end = INT64_MIN;
  if (!span->valid || span->frame != frame || offset < 0 || count <= 0 || offset < span->buffer_first || offset > span->buffer_end
      || count > span->buffer_end - offset || span->zero_tick > INT64_MAX - offset || span->zero_tick > INT64_MAX - offset - count)
    return false;
  const int64_t mapped_first = span->zero_tick + offset;
  if (mapped_first < 0)
    return false;
  *first = mapped_first;
  *end = mapped_first + count;
  return true;
}

/* The low eight bits name the RU; the high 55 bits are a single-producer
 * sequence starting at one. Reject exhaustion instead of reusing a token. */
static inline int64_t nr_prach_capture_token(nr_prach_rx_span_t *span, int ru)
{
  if (ru < 0 || ru > UINT8_MAX || span->sequence >= ((uint64_t)INT64_MAX >> 8))
    return 0;
  return (int64_t)((++span->sequence << 8) | (uint8_t)ru);
}

#endif
