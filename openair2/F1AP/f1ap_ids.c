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

/* "standalone" module to store a "secondary" UE ID for each UE in DU/CU.
 * Separate from the rest of F1, as it is also relevant for monolithic. */

#include "f1ap_ids.h"

#include <pthread.h>
#include "common/utils/hashtable/hashtable.h"
#include "common/utils/assertions.h"


/* we have separate versions for CU and DU, as both CU&DU might coexist in the
 * same process */
hash_table_t *cu2du_ue_mapping;
void cu_init_f1_ue_data(void)
{
  DevAssert(cu2du_ue_mapping == NULL);
  cu2du_ue_mapping = hashtable_create(1319, NULL, free); // 1319 is prime, default hash func (unit), free()
  DevAssert(cu2du_ue_mapping != NULL);
}
bool cu_add_f1_ue_data(uint32_t ue_id, const f1_ue_data_t *data)
{
  DevAssert(cu2du_ue_mapping != NULL);
  uint64_t key = ue_id;
  if (hashtable_is_key_exists(cu2du_ue_mapping, key) == HASH_TABLE_OK)
    return false;
  f1_ue_data_t *idata = malloc(sizeof(*idata));
  AssertFatal(idata != NULL, "cannot allocate memory\n");
  *idata = *data;
  hashtable_rc_t ret = hashtable_insert(cu2du_ue_mapping, key, idata);
  return ret == HASH_TABLE_OK;
}
const f1_ue_data_t *cu_get_f1_ue_data(uint32_t ue_id)
{
  DevAssert(cu2du_ue_mapping != NULL);
  uint64_t key = ue_id;
  void *data = NULL;
  hashtable_rc_t ret = hashtable_get(cu2du_ue_mapping, key, &data);
  return ret == HASH_TABLE_OK ? data : NULL;
}
bool cu_remove_f1_ue_data(uint32_t ue_id)
{
  DevAssert(cu2du_ue_mapping != NULL);
  uint64_t key = ue_id;
  hashtable_rc_t ret = hashtable_remove(cu2du_ue_mapping, key);
  return ret == HASH_TABLE_OK;
}

static hash_table_t *du2cu_ue_mapping;
void du_init_f1_ue_data(void)
{
  DevAssert(du2cu_ue_mapping == NULL);
  du2cu_ue_mapping = hashtable_create(1319, NULL, free); // 1319 is prime, default hash func (unit), free()
  DevAssert(du2cu_ue_mapping != NULL);
}
bool du_add_f1_ue_data(uint32_t ue_id, const f1_ue_data_t *data)
{
  DevAssert(du2cu_ue_mapping != NULL);
  uint64_t key = ue_id;
  if (hashtable_is_key_exists(du2cu_ue_mapping, key) == HASH_TABLE_OK)
    return false;
  f1_ue_data_t *idata = malloc(sizeof(*idata));
  AssertFatal(idata != NULL, "cannot allocate memory\n");
  *idata = *data;
  hashtable_rc_t ret = hashtable_insert(du2cu_ue_mapping, key, idata);
  return ret == HASH_TABLE_OK;
}
const f1_ue_data_t *du_get_f1_ue_data(uint32_t ue_id)
{
  DevAssert(du2cu_ue_mapping != NULL);
  uint64_t key = ue_id;
  void *data = NULL;
  hashtable_rc_t ret = hashtable_get(du2cu_ue_mapping, key, &data);
  return ret == HASH_TABLE_OK ? data : NULL;
}
bool du_remove_f1_ue_data(uint32_t ue_id)
{
  DevAssert(du2cu_ue_mapping != NULL);
  uint64_t key = ue_id;
  hashtable_rc_t ret = hashtable_remove(du2cu_ue_mapping, key);
  return ret == HASH_TABLE_OK;
}
