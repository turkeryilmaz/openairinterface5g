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

#ifndef __MAC_FAPI_TRANSACTION_H_
#define __MAC_FAPI_TRANSACTION_H_

#include <stdint.h>
#include <assertions.h>

struct fapi_transactions_t;

typedef union {
  struct {
    int16_t P_CMAX;
  } ulsch_alloc_data;
} fapi_transaction_data_t;

fapi_transaction_data_t *get_transaction_data(struct fapi_transactions_t *transactions, uint32_t transaction_id);
int32_t get_transaction_id(struct fapi_transactions_t *transactions);
struct fapi_transactions_t *init_fapi_transaction_data(void);

#endif
