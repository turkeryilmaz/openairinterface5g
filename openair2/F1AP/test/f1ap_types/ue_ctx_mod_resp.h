#ifndef UE_CONTEXT_MODIFIED_RESPONSE_F1AP_H
#define UE_CONTEXT_MODIFIED_RESPONSE_F1AP_H 

#include <stdlib.h>

#include "../byte_array.h"

#include "bh_rlc_chn_failed_tbs_lst.h"
#include "drb_setup_item.h"
#include "drb_failed_setup_item.h"
#include "drb_mod.h"
#include "du_to_cu_rrc_information_f1ap.h"

#include "scell_failed_setup_item.h"
#include "sl_drb_failed.h"
#include "srb_failed_setup_item.h"
#include "srb_mod_item.h"


typedef enum{
  TRUE_full_conf_f1ap,

  END_full_conf_f1ap

} full_conf_f1ap_e;

typedef struct{

  // Message Type
  // Mandatory
  // 9.3.1.1

  // gNB-CU UE F1AP ID
  // Mandatory
  // 9.3.1.4
  uint32_t gnb_cu_ue; 

  // gNB-DU UE F1AP ID
  // Mandatory
  // 9.3.1.5
  uint32_t gnb_du_ue; 

  // Resource Coordination Transfer Container
  // Optional
  byte_array_t* res_coord_trans; 

  // DU To CU RRC Information
  // Optional
  // 9.3.1.26
  du_to_cu_rrc_information_f1ap_t* du_to_cu_rrc;

  // DRB Setup List
  // [0 - 64]
  size_t sz_drb_setup;
  drb_setup_item_f1ap_t* drb_setup; 
 
  // DRB Modified List
  // [0 - 64]
  size_t sz_drb_mod;
  drb_mod_t* drb_mod;

  // SRB Failed to be Setup List
  // [0 - 8] 
  size_t sz_srb_failed;
  srb_failed_setup_item_t* srb_failed_setup;

  // SCell Failed To Setup List
  // [0 - 32]
  size_t sz_scell_failed_setup;
  scell_failed_setup_item_t*  scell_failed_setup;

  // DRB Failed to be Modified List
  // [0 - 64]
  size_t sz_drb_fail;
  drb_failed_setup_item_t* drb_failed_setup;

  // SRB Modified List
  // [0 - 8]
  size_t sz_srb_mod;
  srb_mod_item_t*  srb_mod;

  // Full Configuration
  // Optional
  full_conf_f1ap_e* full_conf;

  // BH RLC Channel Setup List
  // [0 - 65536]
  // 9.3.1.113
  // BIT STRING (SIZE(16))
  size_t sz_bh_rlc_chn_stp;
  byte_array_t* bh_rlc_chn_stp; 

  // BH RLC Channel Failed to be Setup List
  // [0 - 65536]
  size_t sz_bh_rlc_chn_failed_tbs;
  bh_rlc_chn_failed_tbs_lst_t* bh_rlc_chn_failed_tbs;

  // BH RLC Channel Modified List
  // [0 - 65536]
  // 9.3.1.113
  // BIT STRING (SIZE(16))
  size_t sz_bh_rlc_chn_mod;
  byte_array_t* bh_rlc_chn_mod; 
  
  // BH RLC Channel Failed to be Modified List
  // [0 - 65536]
  size_t sz_bh_rlc_chn_failed_mod;
  bh_rlc_chn_failed_tbs_lst_t* bh_rlc_chn_failed_mod;

  // SL DRB Setup List
  // [0 - 512]
  // 9.3.1.120 
  size_t sz_sl_drb_stp;  
  uint16_t* sl_drb_stp; // [1-512]

  // SL DRB Modified List
  // [0 - 512]
  // 9.3.1.120 
  size_t sz_sl_drb_mod;  
  uint16_t* sl_drb_mod; // [1-512]

  // SL DRB Failed To Setup List
  // [0 - 512]
  size_t sz_sl_drb_failed_stp; 
  sl_drb_failed_t* sl_drb_failed_stp;

  //SL DRB Failed To be Modified List
  // [0 - 512]
  size_t sz_sl_drb_fail_mod; 
  sl_drb_failed_t* sl_drb_fail_mod;

  // Requested Target Cell ID
  // Mandatory
  nr_cgi_t* req_target_cell_id;

} ue_ctx_mod_resp_t;

void free_ue_ctx_mod_resp(ue_ctx_mod_resp_t* src);

bool eq_ue_ctx_mod_resp(ue_ctx_mod_resp_t* m0, ue_ctx_mod_resp_t* m1);

#endif

