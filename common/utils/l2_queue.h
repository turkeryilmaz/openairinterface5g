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

#ifndef COMMON_UTILS_DS_L2_QUEUE
#define COMMON_UTILS_DS_L2_QUEUE

#include <pthread.h>
#include <stdint.h>

typedef struct {
  uint8_t *buffer;
  int size;
} l2_buffer_t;

typedef struct {
  int packets_capacity;
  int packets_count;
  int bytes_count;
} l2_queue_size_t;

typedef struct {
  l2_buffer_t *array;
  int capacity;
  int read_index;
  int write_index;
  int packets_count;
  int bytes_count;
  pthread_mutex_t mutex;
  pthread_cond_t cond;
} l2_queue_t;

l2_queue_t *new_l2_queue(int capacity);
void delete_l2_queue(l2_queue_t *queue);

void l2_enqueue(l2_queue_t *queue, uint8_t *buffer, int size);
l2_buffer_t l2_dequeue(l2_queue_t *queue);
l2_buffer_t l2_dequeue_wait(l2_queue_t *queue);

l2_queue_size_t l2_queue_size(l2_queue_t *queue);

#endif /* COMMON_UTILS_DS_QUEUE */
