/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef __NR_PHY_MEAS__H__
#define __NR_PHY_MEAS__H__

#include "common/platform_types.h"
#include "time_meas.h"
#include "PHY/impl_defs_nr.h"

// FDD and mixed TDD share the slot value; only FDD admits both full-slot directions.
#define START_MEAS_FULL_SLOT(ts, current, type, duplex) \
  do {                                                  \
    if ((duplex) == FDD || (current) == (type))         \
      start_meas(ts);                                   \
  } while (0)

#define STOP_MEAS_FULL_SLOT(ts, current, type, duplex) \
  do {                                                 \
    if ((duplex) == FDD || (current) == (type))        \
      stop_meas(ts);                                   \
  } while (0)

#define MERGE_MEAS_FULL_SLOT(dst, src, current, type, duplex) \
  do {                                                        \
    if ((duplex) == FDD || (current) == (type))               \
      merge_meas(dst, src);                                   \
  } while (0)

#endif /* __NR_PHY_MEAS__H__ */
