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

#ifndef _NR_PDCP_SECURITY_H_
#define _NR_PDCP_SECURITY_H_

#define NR_PDCP_DBG

#ifdef NR_PDCP_DBG

#include "LOG/log.h"

#define FNIN  LOG_D(PDCP, ">>> %s:%d\n",__FUNCTION__, __LINE__);
#define FNOUT LOG_D(PDCP, "<<< %s:%d\n",__FUNCTION__, __LINE__);
#define LOG_MSG(B, S, ...)  LOG_UDUMPMSG(PDCP, B, S, LOG_DUMP_CHAR, __VA_ARGS__);

#else // NR_PDCP_DBG

#define FNIN    printf(">>> %s:%d\n", __FUNCTION__, __LINE__);
#define FNOUT   printf("<<< %s:%d\n", __FUNCTION__, __LINE__);
#define LOG_MSG(B, S, ...) do {printf(__VA_ARGS__); for (int i = 0; i < S; ++i) printf("%02X", (uint8_t)B[i]); printf("\n"); } while(0);

#endif // NR_PDCP_DBG

#endif /* _NR_PDCP_SECURITY_H_ */
