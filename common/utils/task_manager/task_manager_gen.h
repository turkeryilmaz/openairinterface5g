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

#ifndef TASK_MANAGER_GENERIC_H
#define TASK_MANAGER_GENERIC_H

#include "task.h"
#include "thread_pool/task_manager.h"
#include "threadPool/thread-pool.h"
#include "C-Thread-Pool/thpool.h"

#define THREAD_POOL_WORK_STEALING 1
#define THREAD_POOL_SINGLE_QUEUE  2
#define THREAD_POOL_GITHUB        3

#ifndef THREAD_POOL_IMPLEMENTATION
/* uncomment the version to use if none is defined (in make/cmake) */
#define THREAD_POOL_IMPLEMENTATION THREAD_POOL_WORK_STEALING
//#define THREAD_POOL_IMPLEMENTATION THREAD_POOL_SINGLE_QUEUE
//#define THREAD_POOL_IMPLEMENTATION THREAD_POOL_GITHUB
#endif

#if THREAD_POOL_IMPLEMENTATION == THREAD_POOL_WORK_STEALING

/* Work stealing thread pool */
#define task_manager_t     ws_task_manager_t
#define init_task_manager  init_ws_task_manager
#define free_task_manager  free_ws_task_manager
#define async_task_manager async_ws_task_manager

#elif THREAD_POOL_IMPLEMENTATION == THREAD_POOL_SINGLE_QUEUE

/* Previous single queue OAI thread pool */
#define task_manager_t tpool_t
#define init_task_manager  init_sq_task_manager
#define free_task_manager  free_sq_task_manager
#define async_task_manager async_sq_task_manager

#elif THREAD_POOL_IMPLEMENTATION == THREAD_POOL_GITHUB

/* Most rated C thread pool in github */
#define task_manager_t threadpool
#define init_task_manager  init_c_tp
#define free_task_manager  free_c_tp
#define async_task_manager async_tp_task_manager

#else
#error unknown threadpool implmentation
#endif

#endif /* TASK_MANAGER_GENERIC_H */
