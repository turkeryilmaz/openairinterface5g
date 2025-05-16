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

#include "sync_service.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/socket.h>
#include <errno.h>
#include "common/utils/utils.h"
#include <fcntl.h>
#include "log.h"

#define MAX_CLIENTS 10

struct sync_service_t {
  sync_service_mode_t mode;
  int sockfd;
  struct sockaddr_in addr;
  uint64_t last_received_time;
  int client_sockfd[MAX_CLIENTS];
};

// Initialize sync service as server or client.
// For server, bind_addr and port specify where to listen.
// For client, server_addr and port specify where to connect.
sync_service_t* sync_service_init(sync_service_mode_t mode, const char* addr, uint16_t port)
{
  sync_service_t* svc = calloc_or_fail(1, sizeof(sync_service_t));
  svc->mode = mode;
  svc->sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (svc->sockfd < 0) {
    free(svc);
    return NULL;
  }
  memset(&svc->addr, 0, sizeof(svc->addr));
  svc->addr.sin_family = AF_INET;
  svc->addr.sin_port = htons(port);
  svc->addr.sin_addr.s_addr = inet_addr(addr);
  if (mode == SYNC_SERVICE_SERVER) {
    LOG_I(HW, "Initializing sync_service as SERVER on %s:%d\n", addr, port);
  } else {
    LOG_I(HW, "Initializing sync_service as CLIENT connecting to %s:%d\n", addr, port);
  }

  if (mode == SYNC_SERVICE_SERVER) {
    if (bind(svc->sockfd, (struct sockaddr*)&svc->addr, sizeof(svc->addr)) < 0) {
      close(svc->sockfd);
      free(svc);
      return NULL;
    }
    if (listen(svc->sockfd, 1) < 0) {
      close(svc->sockfd);
      free(svc);
      return NULL;
    }
    int flags = fcntl(svc->sockfd, F_GETFL, 0);
    if (flags >= 0) {
        fcntl(svc->sockfd, F_SETFL, flags | O_NONBLOCK);
    }
    memset(svc->client_sockfd, 0, sizeof(svc->client_sockfd));
  } else {
    int retries = 5;
    while (retries-- > 0) {
      if (connect(svc->sockfd, (struct sockaddr*)&svc->addr, sizeof(svc->addr)) == 0) {
        break;
      }
      if (retries == 0) {
        close(svc->sockfd);
        free(svc);
        return NULL;
      }
      sleep(1);
    }
    // As client, receive the first sync message to initialize last_received_time
    uint64_t initial_time = 0;
    ssize_t recvd = recv(svc->sockfd, &initial_time, sizeof(initial_time), 0);
    if (recvd != sizeof(initial_time)) {
      close(svc->sockfd);
      free(svc);
      return NULL;
    }
    svc->last_received_time = initial_time;
  }
  return svc;
}

// Destroy sync service and free resources.
void sync_service_destroy(sync_service_t* svc)
{
  if (!svc)
    return;
  close(svc->sockfd);
  free(svc);
}

// Server: sends sync message with 'time' to client(s).
// Client: blocks until a sync message with at least 'time' is received.
int sync_service_sync(sync_service_t* svc, uint64_t time)
{
  if (!svc) {
    AssertFatal(0, "recv() failed, errno(%d)\n", errno);
  }
  if (svc->mode == SYNC_SERVICE_SERVER) {
    // Accept 1 new client per call
    socklen_t addr_len = sizeof(svc->addr);
    int client_fd = accept(svc->sockfd, (struct sockaddr*)&svc->addr, &addr_len);
    if (client_fd >= 0) {
      for (int i = 0; i < MAX_CLIENTS; i++) {
        if (svc->client_sockfd[i] == 0) {
          svc->client_sockfd[i] = client_fd;
          break;
        }
      }
    }

    for (int i = 0; i < MAX_CLIENTS; i++) {
      if (svc->client_sockfd[i] > 0) {
        // Send sync message to each client
        ssize_t sent = send(svc->client_sockfd[i], &time, sizeof(time), 0);
        if (sent != sizeof(time)) {
          close(svc->client_sockfd[i]);
          svc->client_sockfd[i] = 0;
        }
      }
    }
    return 0;
  } else {
    // Wait for a time value >= requested time
    if (svc->last_received_time >= time)
      return 0;
    uint64_t recv_time = 0;
    while (1) {
      ssize_t recvd = recv(svc->sockfd, &recv_time, sizeof(recv_time), 0);
      if (recvd == sizeof(recv_time) && recv_time >= time) {
        svc->last_received_time = recv_time;
        return 0;

        if (recvd < 0 && errno != EINTR) {
          AssertFatal(0, "recv() failed, errno(%d)\n", errno);
          LOG_E(HW, "recv() failed, errno(%d)\n", errno);
          close(svc->sockfd);
          free(svc);
          return -1;
        }
      }
    }
  }
}
