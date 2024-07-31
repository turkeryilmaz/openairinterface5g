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

#include "mac_fapi_transaction.h"
#include <stdatomic.h>

// Use power of 2 for fast modulo
#define MAX_CONCURRENT_TRANSACTIONS (1 << 8)

typedef struct fapi_transactions_t {
  _Atomic(uint32_t) transaction_id;
  fapi_transaction_data_t transaction_data[MAX_CONCURRENT_TRANSACTIONS];
} fapi_transactions_t;

static inline uint32_t diff_timestamp_u32(uint32_t a, uint32_t b)
{
  uint32_t diff;
  if (a < b) {
    // basic case: a before b
    // ---|---a-----b---|----
    diff = b - a;
    if (diff > UINT32_MAX / 2) {
      // special case where a is too far from b, therefore b or a is assumed to have wrapped around
      // ---|-a--------b|-a'--
      diff = a + (UINT32_MAX - b);
    }
  } else {
    // basic case: b before a
    // ---|---b-----a---|----
    diff = a - b;
    if (diff > UINT32_MAX / 2) {
      // special case where a is too far from b, therefore b or a is assumed to have wrapped around
      // ---|-b--------a|-b'--
      diff = b + (UINT32_MAX - a);
    }
  }
  return diff;
}

fapi_transaction_data_t *get_transaction_data(struct fapi_transactions_t *transactions, uint32_t transaction_id)
{
  uint32_t diff = diff_timestamp_u32(transaction_id, transactions->transaction_id);
  // TODO: Consider changing to a LOG_E and return NULL. Would have to deal with possible null pointer return value in callers
  AssertFatal(diff < MAX_CONCURRENT_TRANSACTIONS,
              "Too many concurrent transactions. Current transaction id is %u but requested data for transaction %u. Consider "
              "increasing the size of the transaction array.\n",
              transactions->transaction_id,
              transaction_id);
  return &transactions->transaction_data[transaction_id % MAX_CONCURRENT_TRANSACTIONS];
}

int32_t get_transaction_id(struct fapi_transactions_t *transactions)
{
  return transactions->transaction_id++;
}

struct fapi_transactions_t *init_fapi_transaction_data(void)
{
  fapi_transactions_t *ret = (fapi_transactions_t *)malloc(sizeof(*ret));
  ret->transaction_id = 0;
  return ret;
}
