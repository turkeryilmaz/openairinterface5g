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

#include "marl/defer.h"
#include "marl/event.h"
#include "marl/scheduler.h"
#include "marl/waitgroup.h"
#include "marl_if.h"

namespace marl_if {
static marl::Scheduler* scheduler = nullptr;
}

extern "C" void MarlInit(int num_threads)
{
  if (marl_if::scheduler == nullptr) {
    marl_if::scheduler = new marl::Scheduler(marl::Scheduler::Config::allCores().setWorkerThreadCount(num_threads));
  }
}

extern "C" void MarlDestroy(void)
{
  delete marl_if::scheduler;
}

extern "C" void MarlBind(void)
{
  marl_if::scheduler->bind();
}

extern "C" void MarlUnbind(void)
{
  marl_if::scheduler->unbind();
}

extern "C" void MarlExecuteAndWait(notifiedFIFO_elt_t *tasks, int num_tasks)
{
  // Create a WaitGroup with an initial count of numTasks.
  marl::WaitGroup task_wait_group(num_tasks);
  for (auto task_index = 0; task_index < num_tasks; task_index++) {
    marl::schedule([=] { // All marl primitives are capture-by-value.
        // Decrement the WaitGroup counter when the task has finished.
        defer(task_wait_group.done());
        tasks[task_index].processingFunc(tasks[task_index].msgData);
    });
  }
  task_wait_group.wait();
}
