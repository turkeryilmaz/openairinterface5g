#ifndef UE_CONTEXT_MODIFICATION_REQUEST_F1AP_H
#define UE_CONTEXT_MODIFICATION_REQUEST_F1AP_H 

#include <stdbool.h>
#include <stdint.h>

#include "bh_rlc_chn_to_be_mod.h"
#include "bh_rlc_chn_to_be_rel.h"
#include "cell_ul_conf.h"
#include "cu_to_du_rrc_info.h"

#include "drb_to_be_mod.h"
#include "drb_to_be_rel.h"
#include "drb_to_be_setup.h"
#include "drx_cycle.h"
#include "nr_cgi.h"
#include "f1_c_trans_path.h"
#include "intra_du_mob_info.h"
#include "trans_act_ind.h"
#include "rrc_reconf_compl_ind.h"
#include "scell_to_be_setup.h"
#include "sl_drb_to_be_setup.h"
#include "sl_drb_to_be_mod.h"
#include "sl_drb_to_be_rel.h"
#include "srb_to_be_setup.h"
#include "srb_to_be_released.h"


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

  // SpCell ID
  // Optional
  // NR CGI 9.3.1.12
  nr_cgi_t* sp_cell_id;

  // ServCellIndex
  // Optional
  // [0-31]
  uint8_t* serv_cell_idx;

  // SpCell UL Configured
  // Optional
  // Cell UL Configured 9.3.1.33
  cell_ul_conf_e* cell_ul_conf;

  // DRX Cycle
  // Optional
  // DRX Cycle 9.3.1.24
  drx_cycle_t* drx_cycle;

  // CU to DU RRC Information
  // Optional
  // 9.3.1.25
  cu_to_du_rrc_info_t* cu_to_du_rrc_info;

  // Transmission Action Indicator
  // Optional
  // 9.3.1.11
  trans_act_ind_f1ap_e* trans_act_ind;

  // Resource Coordination Transfer Container
  // Optional
  byte_array_t* res_coord_trans_cntnr;

  // RRC Reconfiguration Complete Indicator
  // Optional
  // 9.3.1.30
  rrc_reconf_cmplt_ind_e* rrc_reconf_compl_ind;

  // RRC-Container 
  // Optional
  // 9.3.1.6
  byte_array_t* rrc_cntnr;

  // SCell To Be Setup List
  // [0 - 32]
  size_t sz_scell_to_be_setup;
  scell_to_be_setup_t* scell_to_be_setup;

  // SCell To Be Removed List
  // [0 - 32]
  size_t sz_scell_to_be_rm;
  nr_cgi_t* scell_to_be_rm;

  // SRB to Be Setup List
  // [0-8]
  size_t sz_srb_to_be_setup;
  srb_to_be_setup_t* srb_to_be_setup;

  // DRB to Be Setup List
  // [0- 64]
  size_t sz_drb_to_be_setup; 
  drb_to_be_setup_t* drb_to_be_setup;

  // DRB to Be Modified List
  // [0- 64]
  size_t sz_drb_to_be_mod; 
  drb_to_be_mod_t* drb_to_be_mod;

  // SRB To Be Released List
  // [0 - 8]
  size_t sz_srb_to_be_rel;
  srb_to_be_rel_t* srb_to_be_rel;

  // DRB to Be Released List
  // [0- 64]
  size_t sz_drb_to_be_rel;
  drb_to_be_rel_t* drb_to_be_rel;

  // BH RLC Channel to be Modified List
  // [0 - 65536]
  size_t sz_bh_rlc_chn_to_be_mod;
  bh_rlc_chn_to_be_mod_t* bh_rlc_chn_to_be_mod;

  // BH RLC Channel to be Released List
  // [0 - 65536]
  size_t sz_bh_rlc_chn_to_be_rel;
  bh_rlc_chn_to_be_rel_t* bh_rlc_chn_to_be_rel;

  // SL DRB to Be Setup List
  // [0 - 512]
  size_t sz_sl_drb_to_be_setup;
  sl_drb_to_be_setup_t* sl_drb_to_be_setup;

  // SL DRB to Be Modified List
  // [0 - 512]
  size_t sz_sl_drb_to_be_mod;
  sl_drb_to_be_mod_t* sl_drb_to_be_mod;

  // SL DRB to Be Released List
  // [0 - 512]
  size_t sz_sl_drb_to_be_rel;
  sl_drb_to_be_rel_t* sl_drb_to_be_rel;

  // Conditional Intra-DU Mobility Information
  // Optional
  intra_du_mob_info_t* intra_du_mob_info;

  // F1-C Transfer Path
  // Optional
  // 9.3.1.207
  f1_c_path_nsa_e* f1_c_path_nsa;

} ue_ctx_mod_req_t;

void free_ue_ctx_mod_req(ue_ctx_mod_req_t* src);

bool eq_ue_ctx_mod_req(ue_ctx_mod_req_t const* m0, ue_ctx_mod_req_t const* m1);

#endif

