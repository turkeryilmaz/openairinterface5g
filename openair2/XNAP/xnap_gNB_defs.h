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


#include <stdint.h>
#include "queue.h"
#include "tree.h"
#include "sctp_eNB_defs.h"
#include "xnap_messages_types.h"

#ifndef XNAP_GNB_DEFS_H_
#define XNAP_GNB_DEFS_H_

#define XNAP_GNB_NAME_LENGTH_MAX    (150)

typedef enum {
  /* Disconnected state: initial state for any association. */
  XNAP_GNB_STATE_DISCONNECTED = 0x0,

  /* State waiting for xn Setup response message if the target gNB accepts or
   * Xn Setup failure if rejects the gNB.
   */
  XNAP_GNB_STATE_WAITING     = 0x1,

  /* The gNB is successfully connected to another gNB. */
  XNAP_GNB_STATE_CONNECTED   = 0x2,

  /* XnAP is ready, and the gNB is successfully connected to another gNB. */
  XNAP_GNB_STATE_READY             = 0x3,

  XNAP_GNB_STATE_OVERLOAD          = 0x4,

  XNAP_GNB_STATE_RESETTING         = 0x5,

  /* Max number of states available */
  XNAP_GNB_STATE_MAX,
} xnap_gNB_state_t;

/* Served PLMN identity element */
/*struct plmn_identity_s {
  uint16_t mcc;
  uint16_t mnc;
  uint8_t  mnc_digit_length;
  STAILQ_ENTRY(plmn_identity_s) next;
};*/

struct gnb_code_s {
  uint8_t gnb_code;
  STAILQ_ENTRY(gnb_code_s) next;
};

struct xnap_gNB_instance_s;

/* This structure describes association of a eNB to another eNB */
typedef struct xnap_gNB_data_s {
  /* eNB descriptors tree, ordered by sctp assoc id */
  RB_ENTRY(xnap_gNB_data_s) entry;

  /* This is the optional name provided by the MME */
  char *gNB_name;

  /*  target eNB ID */
  uint32_t gNB_id;

  /* Current gNB load information (if any). */
  //xnap_load_state_t overload_state;

  /* Current gNB->gNB XnAP association state */
  xnap_gNB_state_t state;

  /* Next usable stream for UE signalling */
  int32_t nextstream;

  /* Number of input/ouput streams */
  uint16_t in_streams;
  uint16_t out_streams;

  /* Connexion id used between SCTP/X2AP */
  uint16_t cnx_id;

  /* SCTP association id */
  int32_t  assoc_id;

  /* Nid cells */
  uint32_t                Nid_cell;
  int                     num_cc;
  /*Frequency band of NR neighbor cell supporting ENDC NSA */
  uint32_t                servedNrCell_band;

  /* Only meaningfull in virtual mode */
  struct xnap_gNB_instance_s *xnap_gNB_instance;
} xnap_gNB_data_t;

typedef struct xnap_gNB_instance_s {
  /* used in simulation to store multiple gNB instances*/
  STAILQ_ENTRY(xnap_gNB_instance_s) xnap_gNB_entries;

  /* Number of target gNBs requested by gNB (tree size) */
  uint32_t xn_target_gnb_nb;
  /* Number of target gNBs for which association is pending */
  uint32_t xn_target_gnb_pending_nb;
  /* Number of target gNB successfully associated to gNB */
  uint32_t xn_target_gnb_associated_nb;
  /* Tree of XNAP gNB associations ordered by association ID */
  RB_HEAD(xnap_gnb_map, xnap_gNB_data_s) xnap_gnb_head;

  instance_t instance;

  /* Displayable name of gNB */
  char *gNB_name;

  /* Unique gNB_id to identify the gNB within core.
   * In our case the gNB id will be 28 bits long.
   */
  uint32_t gNB_id;

  /* Tracking area code */
  uint16_t tac;  //octet string of size 3

  /* Mobile Country Code
   * Mobile Network Code
   */
  uint16_t  mcc;
  uint16_t  mnc;
  uint8_t   mnc_digit_length;

  /* CC params */
  uint32_t                downlink_frequency;
  int32_t                 uplink_frequency_offset;
  uint32_t                Nid_cell;
  int16_t                 N_RB_DL;
  int16_t                 N_RB_UL;
  frame_type_t            frame_type;
  uint32_t                fdd_earfcn_DL;
  uint32_t                fdd_earfcn_UL;
  uint32_t                subframeAssignment;
  uint32_t                specialSubframe;
  uint32_t                 nr_band;
  uint32_t		  tdd_nRARFCN;
  uint32_t		  nrARFCN;
  int16_t                 nr_SCS;
  int16_t                 eutra_band;
  int                     num_cc;
  gnb_ip_address_t target_gnb_xn_ip_address[XNAP_MAX_NB_GNB_IP_ADDRESS];
  uint8_t          nb_xn;
  gnb_ip_address_t gnb_xn_ip_address;
  uint16_t         sctp_in_streams;
  uint16_t         sctp_out_streams;
  uint32_t         gnb_port_for_XNC;
  int              multi_sd;
} xnap_gNB_instance_t;

typedef struct {
  /* List of served gNBs*/
  STAILQ_HEAD(xnap_gNB_instances_head_s, xnap_gNB_instance_s) xnap_gNB_instances_head;
  /* Nb of registered gNBs */
  uint8_t nb_registered_gNBs;

  /* Generate a unique connexion id used between XnAP and SCTP */
  uint16_t global_cnx_id;
} xnap_gNB_internal_data_t;

int xnap_gNB_compare_assoc_id(struct xnap_gNB_data_s *p1, struct xnap_gNB_data_s *p2);

/* Generate the tree management functions */
struct xnap_gNB_map;
struct xnap_gNB_data_s;
RB_PROTOTYPE(xnap_gNB_map, xnap_gNB_data_s, entry, xnap_gNB_compare_assoc_id);


#endif /* XNAP_GNB_DEFS_H_ */
