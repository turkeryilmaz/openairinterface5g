/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef USRP_TX_BURST_H
#define USRP_TX_BURST_H

#include <stdbool.h>
#include <stdint.h>

#if __GCC_ATOMIC_LLONG_LOCK_FREE != 2
#error "USRP TX error notifications require lock-free 64-bit atomics"
#endif

/* One async reader publishes; the serialized sender only takes snapshots.
 * UINT64_MAX is outside the valid device timestamp range, including zero. */
typedef struct {
  uint64_t latest_ticks;
} usrp_tx_error_mailbox_t;

static inline void usrp_tx_error_init(usrp_tx_error_mailbox_t *mailbox)
{
  __atomic_store_n(&mailbox->latest_ticks, UINT64_MAX, __ATOMIC_RELAXED);
}

static inline void usrp_tx_error_publish(usrp_tx_error_mailbox_t *mailbox, uint64_t ticks)
{
  if (ticks > INT64_MAX)
    return;
  const uint64_t previous = __atomic_load_n(&mailbox->latest_ticks, __ATOMIC_RELAXED);
  if (previous == UINT64_MAX || ticks > previous)
    __atomic_store_n(&mailbox->latest_ticks, ticks, __ATOMIC_RELEASE);
}

static inline uint64_t usrp_tx_error_snapshot(const usrp_tx_error_mailbox_t *mailbox)
{
  return __atomic_load_n(&mailbox->latest_ticks, __ATOMIC_ACQUIRE);
}

/* Owned by the serialized sender, then by teardown after the sender has joined.
 * An unknown end requires an EOB before another timed payload can start. */
typedef struct {
  int64_t start_timestamp;
  int64_t end_timestamp;
  int64_t padding_timestamp;
  uint64_t consumed_error;
  bool open;
  bool start_valid;
  bool end_valid;
  bool padding_valid;
  bool consumed_error_valid;
} usrp_tx_burst_state_t;

typedef struct {
  bool close;
  bool payload;
  bool start;
} usrp_tx_burst_plan_t;

static inline usrp_tx_burst_plan_t usrp_tx_burst_plan(const usrp_tx_burst_state_t *state,
                                                      int64_t timestamp,
                                                      int samples,
                                                      bool start,
                                                      bool end)
{
  usrp_tx_burst_plan_t plan = {false, false, false};
  if (samples <= 0) {
    /* Do not cache a zero-sample SOB in UHD: it can override a later EOB.
     * An empty END closes an existing burst; other empty controls do nothing. */
    plan.close = samples == 0 && end && state->open;
    return plan;
  }

  plan.close = state->open && (start || !state->end_valid || timestamp != state->end_timestamp);
  plan.payload = true;
  plan.start = start || !state->open || plan.close;
  return plan;
}

static inline void usrp_tx_burst_closed(usrp_tx_burst_state_t *state)
{
  state->open = false;
  state->start_valid = false;
  state->end_valid = false;
}

static inline void usrp_tx_burst_uncertain(usrp_tx_burst_state_t *state)
{
  state->open = true;
  state->end_valid = false;
}

static inline bool usrp_tx_burst_error(usrp_tx_burst_state_t *state, uint64_t ticks)
{
  if (ticks > INT64_MAX || (state->consumed_error_valid && ticks <= state->consumed_error))
    return false;
  state->consumed_error = ticks;
  state->consumed_error_valid = true;
  /* Receipt order cannot identify the burst: an old report can arrive after a
   * successful restart. Compare its device time with the first accepted sample. */
  if (!state->open || (state->start_valid && (int64_t)ticks < state->start_timestamp))
    return false;
  usrp_tx_burst_uncertain(state);
  return true;
}

/* A separate EOB carries one padding sample after the accepted payload. Keep
 * its position across rejected requests and failed closes; do not retime data. */
static inline void usrp_tx_burst_note_padding(usrp_tx_burst_state_t *state, int64_t timestamp)
{
  if (!state->padding_valid || timestamp > state->padding_timestamp)
    state->padding_timestamp = timestamp;
  state->padding_valid = true;
}

static inline bool usrp_tx_burst_follows_padding(const usrp_tx_burst_state_t *state, int64_t timestamp)
{
  return !state->padding_valid || timestamp > state->padding_timestamp;
}

static inline bool usrp_tx_burst_close_result(usrp_tx_burst_state_t *state, uint64_t accepted)
{
  if (accepted != 1) {
    usrp_tx_burst_uncertain(state);
    return false;
  }
  usrp_tx_burst_closed(state);
  return true;
}

static inline void usrp_tx_burst_accepted(usrp_tx_burst_state_t *state, int64_t timestamp, int requested, int accepted, bool end)
{
  if (requested <= 0 || accepted < 0 || accepted > requested) {
    usrp_tx_burst_uncertain(state);
    return;
  }
  if (accepted == 0)
    return;
  state->padding_valid = false;
  /* UHD places EOB on the final fragment, which a short send may not reach. */
  if (end && accepted == requested) {
    usrp_tx_burst_closed(state);
    return;
  }

  if (!state->open) {
    state->start_timestamp = timestamp;
    state->start_valid = true;
  }
  state->open = true;
  state->end_valid = timestamp <= INT64_MAX - accepted;
  if (state->end_valid)
    state->end_timestamp = timestamp + accepted;
}

#endif /* USRP_TX_BURST_H */
