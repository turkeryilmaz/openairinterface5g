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

/*! \file xnap_messages_types.h
 * \author Sreeshma Shiv <sreeshmau@iisc.ac.in>
 * \date August 2023
 * \version 1.0
 */
#ifndef XNAP_MESSAGES_TYPES_H_
#define XNAP_MESSAGES_TYPES_H_

// Defines to access message fields.

#define XNAP_REGISTER_GNB_REQ(mSGpTR)     (mSGpTR)->ittiMsg.xnap_register_gnb_req
#define XNAP_SETUP_REQ(mSGpTR)            (mSGpTR)->ittiMsg.xnap_setup_req
#define XNAP_SETUP_RESP(mSGpTR)           (mSGpTR)->ittiMsg.xnap_setup_resp
#define XNAP_REGISTER_GNB_CNF(mSGpTR)     (mSGpTR)->ittiMsg.xnap_register_gnb_cnf
#define XNAP_DEREGISTERED_GNB_IND(mSGpTR) (mSGpTR)->ittiMsg.xnap_deregistered_gnb_ind

#define XNAP_MAX_NB_GNB_IP_ADDRESS 6

// gNB application layer -> XNAP messages

typedef struct xnap_setup_req_s {
  uint32_t Nid_cell;
} xnap_setup_req_t;

typedef struct xnap_setup_resp_s {
  uint32_t Nid_cell;
} xnap_setup_resp_t;

typedef struct gnb_ip_address_s {
  unsigned ipv4: 1;
  unsigned ipv6: 1;
  char     ipv4_address[16];
  char     ipv6_address[46];
} gnb_ip_address_t;

typedef struct xnap_register_gnb_req_s {
  uint32_t gNB_id;
  /* Optional name for the cell
   * NOTE: the name can be NULL (i.e no name) and will be cropped to 150
   * characters.
   */
  char *gNB_name;

  /* Tracking area code */
  uint16_t tac;

  /* Mobile Country Code
   * Mobile Network Code
   */
  uint16_t mcc;
  uint16_t mnc;
  uint8_t  mnc_digit_length;

  /*
   * CC Params
   */
  int16_t      eutra_band;
  int32_t      nr_band;
  int32_t      nrARFCN;
  uint32_t     downlink_frequency;
  int32_t      uplink_frequency_offset;
  uint32_t     Nid_cell;
  int16_t      N_RB_DL;
  frame_type_t frame_type;
  uint32_t     fdd_earfcn_DL;
  uint32_t     fdd_earfcn_UL;
  uint32_t     subframeAssignment;
  uint32_t     specialSubframe;
  uint16_t     tdd_nRARFCN;
  uint16_t     tdd_Transmission_Bandwidth;

  /* The local gNB IP address to bind */
  gnb_ip_address_t gnb_xn_ip_address;

  /* Nb of GNB to connect to */
  uint8_t nb_xn;

  /* List of target gNB to connect to for Xn*/
  gnb_ip_address_t target_gnb_xn_ip_address[XNAP_MAX_NB_GNB_IP_ADDRESS];

  /* Number of SCTP streams used for associations */
  uint16_t sctp_in_streams;
  uint16_t sctp_out_streams;

  /*gNB port for XNC*/
  uint32_t gnb_port_for_XNC;

  /* timers (unit: millisecond) */
  int t_reloc_prep;
  int txn_reloc_overall;
  int t_dc_prep;
  int t_dc_overall;
} xnap_register_gnb_req_t;

// XNAP -> gNB application layer messages
typedef struct xnap_register_gnb_cnf_s {
  /* Nb of connected gNBs*/
  uint8_t nb_xn;
} xnap_register_gnb_cnf_t;

typedef struct xnap_deregistered_gnb_ind_s {
  /* Nb of connected gNBs */
  uint8_t nb_xn;
} xnap_deregistered_gnb_ind_t;

// XNAP <-> RRC
typedef struct xnap_guami_s {
  uint16_t mcc;
  uint16_t mnc;
  uint8_t  mnc_len;
  uint8_t  amf_region_id;
  uint16_t amf_set_id;
  uint8_t  amf_pointer;
} xnap_guami_t;

#endif /* XNAP_MESSAGES_TYPES_H_ */
