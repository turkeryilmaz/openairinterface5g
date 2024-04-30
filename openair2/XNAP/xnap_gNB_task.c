/* Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
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

/*! \file XNAP/xnap_gNB_task.c
 * \brief XNAP tasks and functions definitions
 * \author Sreeshma Shiv
 * \date Aug 2023
 * \version 1.0
 * \email: sreeshmau@iisc.ac.in
 */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <arpa/inet.h>
#include "intertask_interface.h"
#include <openair3/ocp-gtpu/gtp_itf.h>
#include "xnap_gNB_task.h"
#include "xnap_gNB_defs.h"
#include "xnap_gNB_management_procedures.h"
#include "xnap_gNB_handler.h"
#include "queue.h"
#include "assertions.h"
#include "conversions.h"

RB_PROTOTYPE(xnap_gnb_tree, xnap_gNB_data_t, entry, xnap_gNB_compare_assoc_id);

void *xnap_task(void *arg)
{
  MessageDef *received_msg = NULL;
  int result;
  LOG_D(XNAP, "Starting XNAP layer\n");
  itti_mark_task_ready(TASK_XNAP);
  const int instance = 0;
  xnap_net_config_t xn_net_config = Read_IPconfig_Xn();
  createXninst(instance, NULL, xn_net_config);

  while (1) {
    itti_receive_msg(TASK_XNAP, &received_msg);
    LOG_D(XNAP, "Received message %d:%s\n", ITTI_MSG_ID(received_msg), ITTI_MSG_NAME(received_msg));
    switch (ITTI_MSG_ID(received_msg)) {
      case TERMINATE_MESSAGE:
        LOG_W(XNAP, "Exiting XNAP thread\n");
        itti_exit_task();
        break;

      case XNAP_REGISTER_GNB_REQ: {
        xnap_gNB_init_sctp(ITTI_MSG_DESTINATION_INSTANCE(received_msg), xn_nc);
      } break;

      case XNAP_SETUP_FAILURE: // from rrc/xnap
        xnap_gNB_generate_xn_setup_failure(ITTI_MSG_ORIGIN_INSTANCE(received_msg), &XNAP_SETUP_FAILURE(received_msg));
        break;

      case XNAP_SETUP_RESP: // from rrc
        xnap_gNB_generate_xn_setup_response(ITTI_MSG_ORIGIN_INSTANCE(received_msg), &XNAP_SETUP_RESP(received_msg));
        break;

      case SCTP_INIT_MSG_MULTI_CNF:
        xnap_gNB_handle_sctp_init_msg_multi_cnf(ITTI_MSG_DESTINATION_INSTANCE(received_msg),
                                                &received_msg->ittiMsg.sctp_init_msg_multi_cnf);
        break;

      case SCTP_NEW_ASSOCIATION_RESP:
        xnap_gNB_handle_sctp_association_resp(ITTI_MSG_DESTINATION_INSTANCE(received_msg),
                                              &received_msg->ittiMsg.sctp_new_association_resp);
        break;

      case SCTP_NEW_ASSOCIATION_IND:
        xnap_gNB_handle_sctp_association_ind(ITTI_MSG_DESTINATION_INSTANCE(received_msg),
                                             &received_msg->ittiMsg.sctp_new_association_ind);
        break;

      case SCTP_DATA_IND:
        xnap_gNB_handle_sctp_data_ind(ITTI_MSG_DESTINATION_INSTANCE(received_msg), &received_msg->ittiMsg.sctp_data_ind);
        break;

      default:
        LOG_E(XNAP, "Received unhandled message: %d:%s\n", ITTI_MSG_ID(received_msg), ITTI_MSG_NAME(received_msg));
        break;
    }

    result = itti_free(ITTI_MSG_ORIGIN_ID(received_msg), received_msg);
    AssertFatal(result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);
    received_msg = NULL;
  }

  return NULL;
}

#include "common/config/config_userapi.h"

int is_xnap_enabled(void)
{
  static volatile int config_loaded = 0;
  static volatile int enabled = 0;
  static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

  if (pthread_mutex_lock(&mutex))
    goto mutex_error;
  if (config_loaded) {
    if (pthread_mutex_unlock(&mutex))
      goto mutex_error;
    return enabled;
  }

  char *enable_xn = NULL;
  paramdef_t p[] = {{"enable_xn", "yes/no", 0, .strptr = &enable_xn, .defstrval = "", TYPE_STRING, 0}};

  /* TODO: do it per module - we check only first gNB */
  config_get(config_get_if(), p, sizeofArray(p), "gNBs.[0]");
  if (enable_xn != NULL && strcmp(enable_xn, "yes") == 0) {
    enabled = 1;
  }

  /*Consider also the case of enabling XnAP for a gNB by parsing a gNB configuration file*/

  config_get(config_get_if(), p, sizeofArray(p), "gNBs.[0]");
  if (enable_xn != NULL && strcmp(enable_xn, "yes") == 0) {
    enabled = 1;
  }

  config_loaded = 1;

  if (pthread_mutex_unlock(&mutex))
    goto mutex_error;
  return enabled;

mutex_error:
  LOG_E(XNAP, "mutex error\n");
  exit(1);
}
