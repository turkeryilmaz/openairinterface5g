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
* Author and copyright: Laurent Thomas, open-cells.com
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

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SYNC_SERVICE_DISABLED,
    SYNC_SERVICE_SERVER,
    SYNC_SERVICE_CLIENT
} sync_service_mode_t;

typedef struct sync_service_t sync_service_t;

// Initialize sync service as server or client.
// For server, bind_addr and port specify where to listen.
// For client, server_addr and port specify where to connect.
sync_service_t* sync_service_init(sync_service_mode_t mode, const char* addr, uint16_t port);

// Destroy sync service and free resources.
void sync_service_destroy(sync_service_t* svc);

// Server: sends sync message with 'time' to client(s).
// Client: blocks until a sync message with at least 'time' is received.
int sync_service_sync(sync_service_t* svc, uint64_t time);

#ifdef __cplusplus
}
#endif
