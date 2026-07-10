/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include <assert.h>

#include "common/utils/oai_profiler.h"
#include "log.h"
#include "task_ans.h"
#include "thread-pool.h"

typedef struct {
  task_ans_t *answer;
  oai_profile_context_t observed;
  oai_profile_context_t poison;
} context_task_t;

static void observe_and_poison_context(void *arg)
{
  context_task_t *task = arg;
  task->observed = oai_profiler_get_context();
  oai_profiler_set_context(task->poison);
  completed_task_ans(task->answer);
}

static oai_profile_context_t run_context_task(tpool_t *pool, oai_profile_context_t context)
{
  task_ans_t answer;
  init_task_ans(&answer, 1);
  context_task_t task_data = {
      .answer = &answer,
      .poison = {.absolute_slot = 9999, .correlation_id = 9999, .parent_id = 9999},
  };

  oai_profiler_set_context(context);
  pushTpool(pool, (task_t){.args = &task_data, .func = observe_and_poison_context});
  join_task_ans(&answer);
  return task_data.observed;
}

static void assert_context_equal(oai_profile_context_t actual, oai_profile_context_t expected)
{
  assert(actual.absolute_slot == expected.absolute_slot);
  assert(actual.correlation_id == expected.correlation_id);
  assert(actual.parent_id == expected.parent_id);
}

int main(void)
{
  logInit();
  __atomic_store_n(&oai_profiler_enabled, 1, __ATOMIC_RELEASE);

  const oai_profile_context_t root = {.absolute_slot = 101, .correlation_id = 202, .parent_id = 303};
  oai_profiler_set_context(root);
  const oai_profile_work_t work = oai_profiler_capture_work(404);
  assert(work.dispatch_tick != 0);
  assert_context_equal(work.context, (oai_profile_context_t){.absolute_slot = 404, .correlation_id = 202, .parent_id = 303});
  const oai_profile_context_t previous = oai_profiler_enter_work(work);
  assert_context_equal(oai_profiler_get_context(), work.context);
  oai_profiler_leave_work(previous);
  assert_context_equal(oai_profiler_get_context(), root);

  tpool_t asynchronous_pool;
  char asynchronous_params[] = "-1";
  initNamedTpool(asynchronous_params, &asynchronous_pool, false, "prof-ctx");

  const oai_profile_context_t first = {.absolute_slot = 11, .correlation_id = 12, .parent_id = 13};
  assert_context_equal(run_context_task(&asynchronous_pool, first), first);
  const oai_profile_context_t second = {.absolute_slot = 21, .correlation_id = 22, .parent_id = 23};
  assert_context_equal(run_context_task(&asynchronous_pool, second), second);
  oai_profiler_clear_context();
  const oai_profile_context_t empty = {
      .absolute_slot = OAI_PROFILE_ABSOLUTE_SLOT_UNKNOWN,
  };
  assert_context_equal(run_context_task(&asynchronous_pool, empty), empty);

  __atomic_store_n(&oai_profiler_enabled, 0, __ATOMIC_RELEASE);
  const oai_profile_context_t disabled_context = {.absolute_slot = 41, .correlation_id = 42, .parent_id = 43};
  assert_context_equal(run_context_task(&asynchronous_pool, disabled_context), empty);
  __atomic_store_n(&oai_profiler_enabled, 1, __ATOMIC_RELEASE);
  abortTpool(&asynchronous_pool);

  tpool_t inline_pool;
  char inline_params[] = "n";
  initNamedTpool(inline_params, &inline_pool, false, "prof-inline");
  const oai_profile_context_t inline_context = {.absolute_slot = 31, .correlation_id = 32, .parent_id = 33};
  assert_context_equal(run_context_task(&inline_pool, inline_context), inline_context);
  assert_context_equal(oai_profiler_get_context(), inline_context);
  abortTpool(&inline_pool);

  __atomic_store_n(&oai_profiler_enabled, 0, __ATOMIC_RELEASE);
  int metadata_evaluations = 0;
  OAI_PROFILE_START(disabled_span);
  assert(disabled_span.start_tick == 0);
  OAI_PROFILE_STOP(OAI_PROFILE_EVENT_UE_RF_READ, disabled_span, 0, 0, ++metadata_evaluations, 0, 0, 0, 0);
  OAI_PROFILE_MARK(OAI_PROFILE_EVENT_UE_TX_DEADLINE_MISS, 0, 0, ++metadata_evaluations, 0, 0, 0, 0);
  assert(metadata_evaluations == 0);
  return 0;
}
