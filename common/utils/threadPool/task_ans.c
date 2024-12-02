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

#include "task_ans.h"
#include "assertions.h"
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include "pthread_utils.h"

void init_task_ans(task_ans_t* ans, uint num_jobs) {
  ans->counter = num_jobs;
  sem_init(&ans->sem, 0, 0);
}

void completed_task_ans(task_ans_t* ans)
{
  DevAssert(ans != NULL);
  int num_jobs = atomic_fetch_sub_explicit(&ans->counter, 1, memory_order_relaxed);
  if (num_jobs == 1) {
    sem_post(&ans->sem);
  }
}

void completed_many_task_ans(task_ans_t* ans, uint num_completed_jobs)
{
  DevAssert(ans != NULL);
  int num_jobs = atomic_fetch_sub_explicit(&ans->counter, num_completed_jobs, memory_order_relaxed);
  if (num_jobs == num_completed_jobs) {
    sem_post(&ans->sem);
  }
}

void join_task_ans(task_ans_t* ans)
{
  sem_wait(&ans->sem);
  sem_destroy(&ans->sem);
}
