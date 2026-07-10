/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef TASK_WORK_STEALING_THREAD_POOL_H
#define TASK_WORK_STEALING_THREAD_POOL_H

#include "common/utils/oai_profiler.h"

typedef struct {
  void* args;
  void (*func)(void* args);
  oai_profile_work_t profile_work;
} task_t;

#endif
