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

#include "l2_queue.h"

#include "utils.h"

l2_queue_t *new_l2_queue(int capacity)
{
  l2_queue_t *q = calloc_or_fail(1, sizeof(*q));
  q->array = malloc_or_fail(capacity * sizeof(*q->array));
  q->capacity = capacity;
  int ret = pthread_mutex_init(&q->mutex, NULL);
  DevAssert(ret == 0);
  ret = pthread_cond_init(&q->cond, NULL);
  DevAssert(ret == 0);
  return q;
}

void delete_l2_queue(l2_queue_t *q)
{
  free(q->array);
  free(q);
}

void l2_enqueue(l2_queue_t *q, uint8_t *buffer, int size)
{
  int ret = pthread_mutex_lock(&q->mutex);
  DevAssert(ret == 0);

  AssertFatal(q->packets_count < q->capacity, "l2 queue is full\n");

  q->array[q->write_index].buffer = buffer;
  q->array[q->write_index].size = size;

  q->packets_count++;
  q->bytes_count += size;

  q->write_index++;
  q->write_index %= q->capacity;

  ret = pthread_cond_broadcast(&q->cond);
  DevAssert(ret == 0);

  ret = pthread_mutex_unlock(&q->mutex);
  DevAssert(ret == 0);
}

l2_buffer_t l2_dequeue_wait(l2_queue_t *q)
{
  int ret = pthread_mutex_lock(&q->mutex);
  DevAssert(ret == 0);

  while (q->packets_count == 0) {
    ret = pthread_cond_wait(&q->cond, &q->mutex);
    DevAssert(ret == 0);
  }

  l2_buffer_t l2_buf = q->array[q->read_index];

  q->packets_count--;
  q->bytes_count -= l2_buf.size;

  q->read_index++;
  q->read_index %= q->capacity;

  ret = pthread_mutex_unlock(&q->mutex);
  DevAssert(ret == 0);

  return l2_buf;
}

l2_buffer_t l2_dequeue(l2_queue_t *q)
{
  int ret = pthread_mutex_lock(&q->mutex);
  DevAssert(ret == 0);

  AssertFatal(q->packets_count != 0, "l2 queue is empty\n");

  l2_buffer_t l2_buf = q->array[q->read_index];

  q->packets_count--;
  q->bytes_count -= l2_buf.size;

  q->read_index++;
  q->read_index %= q->capacity;

  ret = pthread_mutex_unlock(&q->mutex);
  DevAssert(ret == 0);

  return l2_buf;
}

l2_queue_size_t l2_queue_size(l2_queue_t *q)
{
  int ret = pthread_mutex_lock(&q->mutex);
  DevAssert(ret == 0);

  l2_queue_size_t l2_size = {
    .packets_count = q->packets_count,
    .bytes_count = q->bytes_count,
    .packets_capacity = q->capacity
  };

  ret = pthread_mutex_unlock(&q->mutex);
  DevAssert(ret == 0);

  return l2_size;
}

void l2_queue_double_capacity(l2_queue_t *q)
{
  int ret = pthread_mutex_lock(&q->mutex);
  DevAssert(ret == 0);

  l2_buffer_t *array = malloc_or_fail(q->capacity * 2);
  int n = q->capacity - q->read_index;
  int n2 = 0;
  if (n > q->packets_count) {
    n = q->packets_count;
  } else {
    n2 = q->packets_count - n;
  }
  memcpy(array, q->array + q->read_index, n * sizeof(l2_buffer_t));
  memcpy(array + n, q->array, n2 * sizeof(l2_buffer_t));

  free(q->array);
  q->array = array;
  q->capacity *= 2;
  q->read_index = 0;
  q->write_index = q->packets_count;

  ret = pthread_mutex_unlock(&q->mutex);
  DevAssert(ret == 0);
}
