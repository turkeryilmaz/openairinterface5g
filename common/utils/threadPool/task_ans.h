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

#ifndef TASK_ANSWER_THREAD_POOL_H
#define TASK_ANSWER_THREAD_POOL_H
#include "pthread_utils.h"
#ifdef __cplusplus
extern "C" {
#endif

#ifndef __cplusplus
#include <stdalign.h>
#include <stdatomic.h>
#else
#include <atomic>
#ifndef _Atomic
#define _Atomic(X) std::atomic<X>
#endif
#define _Alignas(X) alignas(X)
#endif

#include <stddef.h>
#include <stdint.h>
#include <semaphore.h>

#if defined(__i386__) || defined(__x86_64__)
#define LEVEL1_DCACHE_LINESIZE 64
#elif defined(__aarch64__)
// This is not always true for ARM
// in linux, you can obtain the size at runtime using sysconf (_SC_LEVEL1_DCACHE_LINESIZE)
// or from the bash with the command $ getconf LEVEL1_DCACHE_LINESIZE
// in c++ using std::hardware_destructive_interference_size
#define LEVEL1_DCACHE_LINESIZE 64
#else
#error Unknown CPU architecture
#endif

/** @brief
 * A multi-producer - single-consumer synchronization mechanism built for efficiency under
 * contention.
 *
 * @param sem semaphore to wait on
 * @param counter atomic counter to keep track of the number of tasks completed. Atomic counter
 * is used for efficiency under contention.
 */
typedef struct {
  sem_t sem;
  _Alignas(LEVEL1_DCACHE_LINESIZE) _Atomic(int) counter;
} task_ans_t;

typedef struct {
  uint8_t* buf;
  size_t len;
  size_t cap; // capacity
  task_ans_t* ans;
} thread_info_tm_t;

/// @brief Initialize a task_ans_t struct
///
/// @param ans task_ans_t struct
/// @param num_jobs number of tasks to wait for
void init_task_ans(task_ans_t* ans, unsigned int num_jobs);

/// @brief Wait for all tasks to complete
/// @param ans task_ans_t struct
void join_task_ans(task_ans_t* arr);

/// @brief Mark a number of tasks as completed.
///
/// @param ans task_ans_t struct
/// @param num_completed_jobs number of tasks to mark as completed
void completed_many_task_ans(task_ans_t* ans, unsigned int num_completed_jobs);

/// @brief Mark 1 tasks as completed.
///
/// @param ans task_ans_t struct
static inline void completed_task_ans(task_ans_t* ans)
{
  completed_many_task_ans(ans, 1);
}

#ifdef __cplusplus
}
#endif

#endif
