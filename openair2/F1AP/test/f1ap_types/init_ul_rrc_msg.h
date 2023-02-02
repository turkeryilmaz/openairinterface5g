#ifndef INIT_UL_RRC_MSG_TRANS_F1AP_H
#define INIT_UL_RRC_MSG_TRANS_F1AP_H

#include <stdbool.h>

#include "nr_cgi.h"

#include "../byte_array.h"

typedef enum{

  TRUE_SUL_ACCESS_IND,

  END_SUL_ACCESS_IND

} sul_access_ind_e;


typedef struct{

  // Message Type
  // Mandatory
  // 9.3.1.1

  //  gNB-DU UE F1AP ID
  // Mandatory
  // 9.3.1.5
  uint32_t gnb_du_ue_id; // (0 .. 2^32 -1)

  // NR CGI
  // Mandatory
  // 9.3.1.12
  nr_cgi_t nr_cgi;

  // C-RNTI
  // Mandatory
  // 9.3.1.32
  uint16_t c_rnti; // (0..65535)

  //RRC-Container
  //Mandatory
  //9.3.1.6
  byte_array_t rrc_contnr;  

  //DU to CU RRC Container
  // Optional
  // CellGroupConfig IE as defined in subclause 6.3.2 in TS 38.331 [8].
  byte_array_t* du_to_cu_rrc;

  //SUL Access Indication
  // Optional
  sul_access_ind_e* sul_access_ind;

  //Transaction ID
  // Mandatory
  // 9.3.1.23
  uint8_t trans_id;

  // RAN UE ID
  // Optional
  byte_array_t* ran_ue_id; 

  //  RRC-Container-RRCSetupComplete
  // Optional
  // 9.3.1.6
  byte_array_t* rrc_cntnr_rrc_setup;

} init_ul_rrc_msg_t; 

void free_init_ul_rrc_msg(init_ul_rrc_msg_t* src); 

bool eq_init_ul_rrc_msg(init_ul_rrc_msg_t const* m0, init_ul_rrc_msg_t const* m1);

#endif
