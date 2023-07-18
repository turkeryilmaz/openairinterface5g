#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <arpa/inet.h>

#include "intertask_interface.h"

#include "xnap_common.h"
#include <openair3/ocp-gtpu/gtp_itf.h>
#include "xnap_gNB_task.h"
#include "xnap_gNB_defs.h"
#include "xnap_gNB_management_procedures.h"
#include "xnap_gNB_handler.h"
#include "xnap_gNB_generate_messages.h"
#include "xnap_common.h"
#include "xnap_ids.h"
#include "xnap_timers.h"

#include "queue.h"
#include "assertions.h"
#include "conversions.h"

struct xnap_gnb_map;
struct xnap_gNB_data_s;

RB_PROTOTYPE(xnap_gnb_map, xnap_gNB_data_s, entry, xnap_gNB_compare_assoc_id);

static
void xnap_gNB_handle_sctp_data_ind(instance_t instance, sctp_data_ind_t *sctp_data_ind);

static
void xnap_gNB_handle_sctp_association_ind(instance_t instance, sctp_new_association_ind_t *sctp_new_association_ind);

static
void xnap_gNB_handle_register_gNB(instance_t instance, xnap_register_gnb_req_t *xnap_register_gNB);

static
void xnap_gNB_register_gNB(xnap_gNB_instance_t *instance_p,
                           net_ip_address_t    *target_gNB_ip_addr,
                           net_ip_address_t    *local_ip_addr,
                           uint16_t             in_streams,
                           uint16_t             out_streams,
                           uint32_t             gnb_port_for_XNC);
static
void xnap_gNB_handle_sctp_association_resp(instance_t instance, sctp_new_association_resp_t *sctp_new_association_resp);

static
void xnap_gNB_handle_sctp_data_ind(instance_t instance, sctp_data_ind_t *sctp_data_ind) {
  int result;
  DevAssert(sctp_data_ind != NULL);
  xnap_gNB_handle_message(instance, sctp_data_ind->assoc_id, sctp_data_ind->stream,
                          sctp_data_ind->buffer, sctp_data_ind->buffer_length);
  result = itti_free(TASK_UNKNOWN, sctp_data_ind->buffer);
  AssertFatal (result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);
}

static
void xnap_gNB_handle_sctp_association_resp(instance_t instance, sctp_new_association_resp_t *sctp_new_association_resp) {
  xnap_gNB_instance_t *instance_p;
  xnap_gNB_data_t *xnap_gnb_data_p;
  DevAssert(sctp_new_association_resp != NULL);
  xnap_dump_trees();
  instance_p = xnap_gNB_get_instance(instance); //managementproc
  DevAssert(instance_p != NULL);

  /* if the assoc_id is already known, it is certainly because an IND was received
   * before. In this case, just update streams and return
   */
  if (sctp_new_association_resp->assoc_id != -1) {
    xnap_gnb_data_p = xnap_get_gNB(instance_p, sctp_new_association_resp->assoc_id,
                                   sctp_new_association_resp->ulp_cnx_id);

    if (xnap_gnb_data_p != NULL) {
      /* some sanity check - to be refined at some point */
      if (sctp_new_association_resp->sctp_state != SCTP_STATE_ESTABLISHED) {
        XNAP_ERROR("xnap_gnb_data_p not NULL and sctp state not SCTP_STATE_ESTABLISHED?\n");
        if (sctp_new_association_resp->sctp_state == SCTP_STATE_SHUTDOWN){
          RB_REMOVE(xnap_gnb_map, &instance_p->xnap_gnb_head, xnap_gnb_data_p);
          return;
        }

        exit(1);
      }

      xnap_gnb_data_p->in_streams  = sctp_new_association_resp->in_streams;
      xnap_gnb_data_p->out_streams = sctp_new_association_resp->out_streams;
      return;
    }
  }

  xnap_gnb_data_p = xnap_get_gNB(instance_p, -1,
                                 sctp_new_association_resp->ulp_cnx_id);
  DevAssert(xnap_gnb_data_p != NULL);
  xnap_dump_trees();

  /* gNB: exit if connection to gNB failed - to be modified if needed.
   * We may want to try to connect over and over again until we succeed
   * but the modifications to the code to get this behavior are complex.
   * Exit on error is a simple solution that can be caught by a script
   * for example.
   */
  if (instance_p->cell_type == CELL_MACRO_GNB  
      && sctp_new_association_resp->sctp_state == SCTP_STATE_UNREACHABLE) {
    XNAP_ERROR("association with gNB failed, is it running? If no, run it first. If yes, check IP addresses in your configuration file.\n");
    exit(1);
  }
  //cell_macro_gnb already using in x2ap,ngap,f1

  if (sctp_new_association_resp->sctp_state != SCTP_STATE_ESTABLISHED) {
    XNAP_WARN("Received unsuccessful result for SCTP association (%u), instance %ld, cnx_id %u\n",
              sctp_new_association_resp->sctp_state,
              instance,
              sctp_new_association_resp->ulp_cnx_id);
    xnap_handle_xn_setup_message(instance_p, xnap_gnb_data_p,
                                 sctp_new_association_resp->sctp_state == SCTP_STATE_SHUTDOWN);
    return;
  }

  xnap_dump_trees();
  /* Update parameters */
  xnap_gnb_data_p->assoc_id    = sctp_new_association_resp->assoc_id;
  xnap_gnb_data_p->in_streams  = sctp_new_association_resp->in_streams;
  xnap_gnb_data_p->out_streams = sctp_new_association_resp->out_streams;
  xnap_dump_trees();
  /* Prepare new xn Setup Request */
  if(instance_p->cell_type == CELL_MACRO_GNB)
	  //xnap_gNB_generate_ENDC_xn_setup_request(instance_p, xnap_gnb_data_p); //not defined i guess
	  LOG_E(XNAP, "CELL_MACRO_GNB ENDC not defined");
  else
	  xnap_gNB_generate_xn_setup_request(instance_p, xnap_gnb_data_p);
}

static
void xnap_gNB_handle_register_gNB(instance_t instance,
                                  xnap_register_gnb_req_t *xnap_register_gNB) {
  xnap_gNB_instance_t *new_instance;
  DevAssert(xnap_register_gNB != NULL);
  /* Look if the provided instance already exists */
  new_instance = xnap_gNB_get_instance(instance);

  if (new_instance != NULL) {
    /* Checks if it is a retry on the same gNB */
    DevCheck(new_instance->gNB_id == xnap_register_gNB->gNB_id, new_instance->gNB_id, xnap_register_gNB->gNB_id, 0);
    DevCheck(new_instance->cell_type == xnap_register_gNB->cell_type, new_instance->cell_type, xnap_register_gNB->cell_type, 0);
    DevCheck(new_instance->tac == xnap_register_gNB->tac, new_instance->tac, xnap_register_gNB->tac, 0);
    DevCheck(new_instance->mcc == xnap_register_gNB->mcc, new_instance->mcc, xnap_register_gNB->mcc, 0);
    DevCheck(new_instance->mnc == xnap_register_gNB->mnc, new_instance->mnc, xnap_register_gNB->mnc, 0);
    XNAP_WARN("gNB[%ld] already registered\n", instance);
  } else {
    new_instance = calloc(1, sizeof(xnap_gNB_instance_t));
    DevAssert(new_instance != NULL);
    RB_INIT(&new_instance->xnap_gnb_head);
    /* Copy usefull parameters */
    new_instance->instance         = instance;
    new_instance->gNB_name         = xnap_register_gNB->gNB_name;
    new_instance->gNB_id           = xnap_register_gNB->gNB_id;
    new_instance->cell_type        = xnap_register_gNB->cell_type;
    new_instance->tac              = xnap_register_gNB->tac;
    new_instance->mcc              = xnap_register_gNB->mcc;
    new_instance->mnc              = xnap_register_gNB->mnc;
    new_instance->mnc_digit_length = xnap_register_gNB->mnc_digit_length;
    new_instance->num_cc           = xnap_register_gNB->num_cc;

    xnap_id_manager_init(&new_instance->id_manager);
    xnap_timers_init(&new_instance->timers,
                     xnap_register_gNB->t_reloc_prep,
                     xnap_register_gNB->txn_reloc_overall,
                     xnap_register_gNB->t_dc_prep,
                     xnap_register_gNB->t_dc_overall);

    for (int i = 0; i< xnap_register_gNB->num_cc; i++) {
      if(new_instance->cell_type == CELL_MACRO_GNB){
        new_instance->nr_band[i]              = xnap_register_gNB->nr_band[i];
        new_instance->tdd_nRARFCN[i]             = xnap_register_gNB->nrARFCN[i];
      }
      else{
        new_instance->eutra_band[i]              = xnap_register_gNB->eutra_band[i];
        new_instance->downlink_frequency[i]      = xnap_register_gNB->downlink_frequency[i];
        new_instance->fdd_earfcn_DL[i]           = xnap_register_gNB->fdd_earfcn_DL[i];
        new_instance->fdd_earfcn_UL[i]           = xnap_register_gNB->fdd_earfcn_UL[i];
      }

      new_instance->uplink_frequency_offset[i] = xnap_register_gNB->uplink_frequency_offset[i];
      new_instance->Nid_cell[i]                = xnap_register_gNB->Nid_cell[i];
      new_instance->N_RB_DL[i]                 = xnap_register_gNB->N_RB_DL[i];
      new_instance->frame_type[i]              = xnap_register_gNB->frame_type[i];
    }

    DevCheck(xnap_register_gNB->nb_xn <= XNAP_MAX_NB_GNB_IP_ADDRESS,
             XNAP_MAX_NB_GNB_IP_ADDRESS, xnap_register_gNB->nb_xn, 0);
    memcpy(new_instance->target_gnb_xn_ip_address,
           xnap_register_gNB->target_gnb_xn_ip_address,
           xnap_register_gNB->nb_xn * sizeof(net_ip_address_t));
    new_instance->nb_xn             = xnap_register_gNB->nb_xn;
    new_instance->gnb_xn_ip_address = xnap_register_gNB->gnb_xn_ip_address;
    new_instance->sctp_in_streams   = xnap_register_gNB->sctp_in_streams;
    new_instance->sctp_out_streams  = xnap_register_gNB->sctp_out_streams;
    new_instance->gnb_port_for_XNC  = xnap_register_gNB->gnb_port_for_XNC;
    /* Add the new instance to the list of gNB (meaningfull in virtual mode) */
    xnap_gNB_insert_new_instance(new_instance);
    XNAP_INFO("Registered new gNB[%ld] and %s gNB id %u\n",
              instance,
              xnap_register_gNB->cell_type == CELL_MACRO_GNB ? "macro" : "home",
              xnap_register_gNB->gNB_id);

    /* initiate the SCTP listener */
    if (xnap_gNB_init_sctp(new_instance,&xnap_register_gNB->gnb_xn_ip_address,xnap_register_gNB->gnb_port_for_XNC) <  0 ) {
      XNAP_ERROR ("Error while sending SCTP_INIT_MSG to SCTP \n");
      return;
    }

    XNAP_INFO("gNB[%ld] gNB id %u acting as a listner (server)\n",
              instance, xnap_register_gNB->gNB_id);
  }
}

int xnap_gNB_init_sctp (xnap_gNB_instance_t *instance_p,
                        net_ip_address_t    *local_ip_addr,
                        uint32_t gnb_port_for_XNC) {
  // Create and alloc new message
  MessageDef                             *message;
  sctp_init_t                            *sctp_init  = NULL;
  DevAssert(instance_p != NULL);
  DevAssert(local_ip_addr != NULL);
  message = itti_alloc_new_message (TASK_XNAP, 0, SCTP_INIT_MSG_MULTI_REQ);
  sctp_init = &message->ittiMsg.sctp_init_multi;
  sctp_init->port = gnb_port_for_XNC;
  sctp_init->ppid = XNAP_SCTP_PPID;
  sctp_init->ipv4 = 1;
  sctp_init->ipv6 = 0;
  sctp_init->nb_ipv4_addr = 1;
#if 0
  memcpy(&sctp_init->ipv4_address,
         local_ip_addr,
         sizeof(*local_ip_addr));
#endif
  sctp_init->ipv4_address[0] = inet_addr(local_ip_addr->ipv4_address);
  /*
   * SR WARNING: ipv6 multi-homing fails sometimes for localhost.
   * * * * Disable it for now.
   */
  sctp_init->nb_ipv6_addr = 0;
  sctp_init->ipv6_address[0] = "0:0:0:0:0:0:0:1";
  return itti_send_msg_to_task (TASK_SCTP, instance_p->instance, message);
}

static void xnap_gNB_register_gNB(xnap_gNB_instance_t *instance_p,
                                  net_ip_address_t    *target_gNB_ip_address,
                                  net_ip_address_t    *local_ip_addr,
                                  uint16_t             in_streams,
                                  uint16_t             out_streams,
                                  uint32_t         gnb_port_for_XNC) {
  MessageDef                       *message                   = NULL;
  xnap_gNB_data_t                  *xnap_gnb_data             = NULL;
  DevAssert(instance_p != NULL);
  DevAssert(target_gNB_ip_address != NULL);
  message = itti_alloc_new_message(TASK_XNAP, 0, SCTP_NEW_ASSOCIATION_REQ);
   sctp_new_association_req_t *sctp_new_association_req = &message->ittiMsg.sctp_new_association_req;
  sctp_new_association_req->port = gnb_port_for_XNC;
  sctp_new_association_req->ppid = XNAP_SCTP_PPID;
  sctp_new_association_req->in_streams  = in_streams;
  sctp_new_association_req->out_streams = out_streams;
  memcpy(&sctp_new_association_req->remote_address,
         target_gNB_ip_address,
         sizeof(*target_gNB_ip_address));
  memcpy(&sctp_new_association_req->local_address,
         local_ip_addr,
         sizeof(*local_ip_addr));
  /* Create new gNB descriptor */
  xnap_gnb_data = calloc(1, sizeof(*xnap_gnb_data));
  DevAssert(xnap_gnb_data != NULL);
  xnap_gnb_data->cnx_id                = xnap_gNB_fetch_add_global_cnx_id();
  sctp_new_association_req->ulp_cnx_id = xnap_gnb_data->cnx_id;
  xnap_gnb_data->assoc_id          = -1;
  xnap_gnb_data->xnap_gNB_instance = instance_p;
  /* Insert the new descriptor in list of known gNB
   * but not yet associated.
   */
  RB_INSERT(xnap_gnb_map, &instance_p->xnap_gnb_head, xnap_gnb_data);
  xnap_gnb_data->state = XNAP_GNB_STATE_WAITING;
  instance_p->xn_target_gnb_nb ++;
  instance_p->xn_target_gnb_pending_nb ++;
  itti_send_msg_to_task(TASK_SCTP, instance_p->instance, message);
}


static
void xnap_gNB_handle_sctp_init_msg_multi_cnf(
  instance_t instance_id,
  sctp_init_msg_multi_cnf_t *m) {
  xnap_gNB_instance_t *instance;
  int index;
  DevAssert(m != NULL);
  instance = xnap_gNB_get_instance(instance_id);
  DevAssert(instance != NULL);
  instance->multi_sd = m->multi_sd;

  /* Exit if CNF message reports failure.
   * Failure means multi_sd < 0.
   */
  if (instance->multi_sd < 0) {
    XNAP_ERROR("Error: be sure to properly configure XN in your configuration file.\n");
    DevAssert(instance->multi_sd >= 0);
  }

  /* Trying to connect to the provided list of gNB ip address */

  for (index = 0; index < instance->nb_xn; index++) {
    XNAP_INFO("gNB[%ld] gNB id %u acting as an initiator (client)\n",instance_id, instance->gNB_id);
    xnap_gNB_register_gNB(instance,
                          &instance->target_gnb_xn_ip_address[index],
                          &instance->gnb_xn_ip_address,
                          instance->sctp_in_streams,
                          instance->sctp_out_streams,
                          instance->gnb_port_for_XNC);
  }
}


static
void xnap_gNB_handle_sctp_association_ind(instance_t instance, sctp_new_association_ind_t *sctp_new_association_ind) {
  xnap_gNB_instance_t *instance_p;
  xnap_gNB_data_t *xnap_gnb_data_p;
  printf("xnap_gNB_handle_sctp_association_ind at 1 (called for instance %ld)\n", instance);
  xnap_dump_trees();
  DevAssert(sctp_new_association_ind != NULL);
  instance_p = xnap_gNB_get_instance(instance);
  DevAssert(instance_p != NULL);
  xnap_gnb_data_p = xnap_get_gNB(instance_p, sctp_new_association_ind->assoc_id, -1);

  if (xnap_gnb_data_p != NULL) abort();

  //  DevAssert(xnap_gnb_data_p != NULL);
  if (xnap_gnb_data_p == NULL) {
    /* Create new gNB descriptor */
    xnap_gnb_data_p = calloc(1, sizeof(*xnap_gnb_data_p));
    DevAssert(xnap_gnb_data_p != NULL);
    xnap_gnb_data_p->cnx_id                = xnap_gNB_fetch_add_global_cnx_id();
    xnap_gnb_data_p->xnap_gNB_instance = instance_p;
    /* Insert the new descriptor in list of known gNB
     * but not yet associated.
     */
    RB_INSERT(xnap_gnb_map, &instance_p->xnap_gnb_head, xnap_gnb_data_p);
    xnap_gnb_data_p->state = XNAP_GNB_STATE_CONNECTED;
    instance_p->xn_target_gnb_nb++;

    if (instance_p->xn_target_gnb_pending_nb > 0) {
      instance_p->xn_target_gnb_pending_nb--;
    }
  } else {
    XNAP_WARN("xnap_gnb_data_p already exists\n");
  }

  printf("xnap_gNB_handle_sctp_association_ind at 2\n");
  xnap_dump_trees();
  /* Update parameters */
  xnap_gnb_data_p->assoc_id    = sctp_new_association_ind->assoc_id;
  xnap_gnb_data_p->in_streams  = sctp_new_association_ind->in_streams;
  xnap_gnb_data_p->out_streams = sctp_new_association_ind->out_streams;
  printf("xnap_gNB_handle_sctp_association_ind at 3\n");
  xnap_dump_trees();
}



void *xnap_task(void *arg) {
  MessageDef *received_msg = NULL;
  int         result;
  XNAP_DEBUG("Starting XNAP layer\n");
  xnap_gNB_prepare_internal_data();   //management procedures
  itti_mark_task_ready(TASK_XNAP);

  while (1) {
    itti_receive_msg(TASK_XNAP, &received_msg);
    LOG_D(XNAP, "Received message %d:%s\n",
	       ITTI_MSG_ID(received_msg), ITTI_MSG_NAME(received_msg));
    switch (ITTI_MSG_ID(received_msg)) {
      case TERMINATE_MESSAGE:
        XNAP_WARN(" *** Exiting XNAP thread\n");
        itti_exit_task();
        break;

      /*case XNAP_SUBFRAME_PROCESS:
        xnap_check_timers(ITTI_MSG_DESTINATION_INSTANCE(received_msg)); //to be added in xnap_timers
        break; */

      case XNAP_REGISTER_GNB_REQ:
        xnap_gNB_handle_register_gNB(ITTI_MSG_DESTINATION_INSTANCE(received_msg),
                                     &XNAP_REGISTER_GNB_REQ(received_msg));
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
        xnap_gNB_handle_sctp_data_ind(ITTI_MSG_DESTINATION_INSTANCE(received_msg),
                                      &received_msg->ittiMsg.sctp_data_ind);
        break;

      default:
        XNAP_ERROR("Received unhandled message: %d:%s\n",
                   ITTI_MSG_ID(received_msg), ITTI_MSG_NAME(received_msg));
        break;
    }

    result = itti_free (ITTI_MSG_ORIGIN_ID(received_msg), received_msg);
    AssertFatal (result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);
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

  if (pthread_mutex_lock(&mutex)) goto mutex_error;
  if (config_loaded) {
    if (pthread_mutex_unlock(&mutex)) goto mutex_error;
    return enabled;
  }

  char *enable_xn = NULL;
  paramdef_t p[] = {
   { "enable_xn", "yes/no", 0, .strptr=&enable_xn, .defstrval="", TYPE_STRING, 0 }
  };

  /* TODO: do it per module - we check only first eNB */
  config_get(p, sizeof(p)/sizeof(paramdef_t), "eNBs.[0]");
  if (enable_xn != NULL && strcmp(enable_xn, "yes") == 0){
	  enabled = 1;
  }

  /*Consider also the case of enabling XnAP for a gNB by parsing a gNB configuration file*/

  config_get(p, sizeof(p)/sizeof(paramdef_t), "gNBs.[0]");
    if (enable_xn!= NULL && strcmp(enable_xn, "yes") == 0){
  	  enabled = 1;
    }

  config_loaded = 1;

  if (pthread_mutex_unlock(&mutex)) goto mutex_error;
  return enabled;

mutex_error:
  LOG_E(XNAP, "mutex error\n");
  exit(1);
}


