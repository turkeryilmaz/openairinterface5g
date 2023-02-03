#ifndef UL_RRC_MSG_MIR_F1AP_H
#define UL_RRC_MSG_MIR_F1AP_H

#include "../byte_array.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct{

  //  Message Type
  // 9.3.1.1
  // Mandatory

  // gNB-CU UE F1AP ID
  // 9.3.1.4
  // Mandatory
  uint32_t gnb_cu_ue; // [0-2^32-1] 

  // gNB-DU UE F1AP ID
  // Mandatory
  // 9.3.1.5
  uint32_t gnb_du_ue; // [0-2^32-1] 

  //SRB ID
  //Mandatory
  //9.3.1.7
  uint8_t srb_id; // [0-3]

  //RRC-Container
  //Mandatory
  //9.3.1.6
  byte_array_t rrc_cntnr;

  //Selected PLMN ID
  //Optional
  // PLMN Identity 9.3.1.14
  uint8_t* plmn_id; // (size(3) i.e., plmn_id[3])

  // New gNB-DU UE F1AP ID
  // Optional
  // gNB-DU UE F1AP ID 9.3.1.5
  uint32_t* new_gnb_du_ue; 

} ul_rrc_msg_t; 

void free_ul_rrc_msg(ul_rrc_msg_t* src);

bool eq_ul_rrc_msg(ul_rrc_msg_t const* m0, ul_rrc_msg_t const* m1);

#endif

