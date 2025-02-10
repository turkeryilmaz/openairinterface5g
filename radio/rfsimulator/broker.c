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
* Author and copyright: Abdelkhalek Beraoud, Amine Adjou, eurecom.fr
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

#include <zmq.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    void *context = zmq_ctx_new();
    int io_threads = 4;
    zmq_ctx_set(context, ZMQ_IO_THREADS, io_threads);
    printf("Running with %d I/O threads, %s\n",
      io_threads,
      (zmq_ctx_get(context, ZMQ_IO_THREADS) == io_threads) ? "success" : "fail");
    // XSUB socket for publishers
    void *xsub_socket = zmq_socket(context, ZMQ_XSUB);
    zmq_bind(xsub_socket, "tcp://0.0.0.0:5555");
    
    // XPUB socket for subscribers
    void *xpub_socket = zmq_socket(context, ZMQ_XPUB);
    zmq_bind(xpub_socket, "tcp://0.0.0.0:5556");
    printf("Broker is running using zmq proxy\n");
    // Proxy between XSUB and XPUB sockets
    zmq_proxy(xsub_socket, xpub_socket, NULL);

    // Clean up
    zmq_close(xsub_socket);
    zmq_close(xpub_socket);
    zmq_ctx_destroy(context);

    return 0;
}
