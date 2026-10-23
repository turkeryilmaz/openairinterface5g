/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef USRP_TX_BURST_H
#define USRP_TX_BURST_H

#include <stdbool.h>
#include <stdint.h>

/* Owned by the serialized sender, then by teardown after the sender has joined.
 * An unknown end requires an EOB before another timed payload can start. */
typedef struct {
  int64_t end_timestamp;
  bool open;
  bool end_valid;
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
  state->end_valid = false;
}

static inline void usrp_tx_burst_uncertain(usrp_tx_burst_state_t *state)
{
  state->open = true;
  state->end_valid = false;
}

static inline void usrp_tx_burst_accepted(usrp_tx_burst_state_t *state, int64_t timestamp, int requested, int accepted, bool end)
{
  if (requested <= 0 || accepted < 0 || accepted > requested) {
    usrp_tx_burst_uncertain(state);
    return;
  }
  if (accepted == 0)
    return;
  /* UHD places EOB on the final fragment, which a short send may not reach. */
  if (end && accepted == requested) {
    usrp_tx_burst_closed(state);
    return;
  }

  state->open = true;
  state->end_valid = timestamp <= INT64_MAX - accepted;
  if (state->end_valid)
    state->end_timestamp = timestamp + accepted;
}

#endif /* USRP_TX_BURST_H */
