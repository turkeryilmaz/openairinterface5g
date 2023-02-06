#include "ue_ctx_mod_request.h"

#include <assert.h>

void free_ue_ctx_mod_req(ue_ctx_mod_req_t* src)
{
  assert(src != NULL);

  // Message Type
  // Mandatory
  // 9.3.1.1

  // gNB-CU UE F1AP ID
  // Mandatory
  // 9.3.1.4
  //uint32_t gnb_cu_ue; 

  // gNB-DU UE F1AP ID
  // Mandatory
  // 9.3.1.5
  //uint32_t gnb_du_ue;

  // SpCell ID
  // Optional
  // NR CGI 9.3.1.12
  assert(src->sp_cell_id == NULL && "Not implemented");

  // ServCellIndex
  // Optional
  // [0-31]
  // uint8_t serv_cell_idx;

  // SpCell UL Configured
  // Optional
  // Cell UL Configured 9.3.1.33
  assert(src->cell_ul_conf == NULL && "Not implemented");

  // DRX Cycle
  // Optional
  // DRX Cycle 9.3.1.24
  assert(src->drx_cycle == NULL && "Not implemented");

  // CU to DU RRC Information
  // Optional
  // 9.3.1.25
  assert(src->cu_to_du_rrc_info== NULL && "Not implemented");

  // Transmission Action Indicator
  // Optional
  // 9.3.1.11
  assert(src->trans_act_ind == NULL && "Not implemented");

  // Resource Coordination Transfer Container
  // Optional
  assert(src->res_coord_trans_cntnr == NULL && "Not implemented");

  // RRC Reconfiguration Complete Indicator
  // Optional
  // 9.3.1.30
  assert(src->rrc_reconf_compl_ind == NULL && "Not implemented");

  // RRC-Container 
  // Optional
  // 9.3.1.6
  assert(src->rrc_cntnr == NULL && "Not implemented");

  // SCell To Be Setup List
  // [0 - 32]
  assert(src-> sz_scell_to_be_setup == 0 && "Not implemented");
  assert(src-> scell_to_be_setup == NULL && "Not implemented");

  // SCell To Be Removed List
  // [0 - 32]
  assert(src-> sz_scell_to_be_rm == 0 && "Not implemented");
  assert(src-> scell_to_be_rm == NULL && "Not implemented");

  // SRB to Be Setup List
  // [0-8]
  assert(src-> sz_srb_to_be_setup == 0 && "Not implemented");
  assert(src-> srb_to_be_setup == NULL && "Not implemented");

  // DRB to Be Setup List
  // [0- 64]
  assert(src-> sz_drb_to_be_setup == 0 && "Not implemented");
  assert(src-> drb_to_be_setup == NULL && "Not implemented");

  // DRB to Be Modified List
  // [0- 64]
  assert(src-> sz_drb_to_be_mod == 0 && "Not implemented");
  assert(src-> drb_to_be_mod == NULL && "Not implemented");

  // SRB To Be Released List
  // [0 - 8]
  assert(src-> sz_srb_to_be_rel == 0 && "Not implemented");
  assert(src-> srb_to_be_rel == NULL && "Not implemented");

  // DRB to Be Released List
  // [0- 64]
  assert(src-> sz_drb_to_be_rel == 0 && "Not implemented");
  assert(src-> drb_to_be_rel == NULL && "Not implemented");

  // BH RLC Channel to be Modified List
  // [0 - 65536]
  assert(src-> sz_bh_rlc_chn_to_be_mod == 0 && "Not implemented");
  assert(src-> bh_rlc_chn_to_be_mod == NULL && "Not implemented");

  // BH RLC Channel to be Released List
  // [0 - 65536]
  assert(src-> sz_bh_rlc_chn_to_be_rel == 0 && "Not implemented");
  assert(src-> bh_rlc_chn_to_be_rel == NULL && "Not implemented");

  // SL DRB to Be Setup List
  // [0 - 512]
  assert(src-> sz_sl_drb_to_be_setup == 0 && "Not implemented");
  assert(src-> sl_drb_to_be_setup == NULL && "Not implemented");

  // SL DRB to Be Modified List
  // [0 - 512]
  assert(src->sz_sl_drb_to_be_mod== 0 && "Not implemented");
  assert(src-> sl_drb_to_be_mod == NULL && "Not implemented");

  // SL DRB to Be Released List
  // [0 - 512]
  assert(src-> sz_sl_drb_to_be_rel == 0 && "Not implemented");
  assert(src-> sl_drb_to_be_rel == NULL && "Not implemented");

  // Conditional Intra-DU Mobility Information
  // Optional
  assert(src-> intra_du_mob_info == NULL && "Not implemented");

  // F1-C Transfer Path
  // Optional
  // 9.3.1.207
  assert(src-> f1_c_path_nsa == NULL && "Not implemented");
}


bool eq_ue_ctx_mod_req(ue_ctx_mod_req_t const* m0, ue_ctx_mod_req_t const* m1)
{
  if(m0 == m1)
    return true;

  if(m0 == NULL || m1 == NULL)
    return false;

  // Message Type
  // Mandatory
  // 9.3.1.1

  // gNB-CU UE F1AP ID
  // Mandatory
  // 9.3.1.4
  if(m0->gnb_cu_ue != m1->gnb_cu_ue) 
    return false;

  // gNB-DU UE F1AP ID
  // Mandatory
  // 9.3.1.5
  if(m0-> gnb_du_ue != m1-> gnb_du_ue )
    return false;

  // SpCell ID
  // Optional
  // NR CGI 9.3.1.12
  assert(m0->sp_cell_id == NULL && "Not implemented" );
  assert(m1->sp_cell_id == NULL && "Not implemented" );

  // ServCellIndex
  // Optional
  // [0-31]
  assert(m0-> serv_cell_idx == NULL && "Not implemented" );
  assert(m1-> serv_cell_idx == NULL && "Not implemented" );

  // SpCell UL Configured
  // Optional
  // Cell UL Configured 9.3.1.33
  assert(m0-> cell_ul_conf == NULL && "Not implemented" );
  assert(m1-> cell_ul_conf == NULL && "Not implemented" );

  // DRX Cycle
  // Optional
  // DRX Cycle 9.3.1.24
  assert(m0-> drx_cycle == NULL && "Not implemented" );
  assert(m1-> drx_cycle == NULL && "Not implemented" );

  // CU to DU RRC Information
  // Optional
  // 9.3.1.25
  assert(m0-> cu_to_du_rrc_info == NULL && "Not implemented" );
  assert(m1-> cu_to_du_rrc_info == NULL && "Not implemented" );

  // Transmission Action Indicator
  // Optional
  // 9.3.1.11
  assert(m0-> trans_act_ind == NULL && "Not implemented" );
  assert(m1-> trans_act_ind == NULL && "Not implemented" );

  // Resource Coordination Transfer Container
  // Optional
  assert(m0-> res_coord_trans_cntnr == NULL && "Not implemented" );
  assert(m1-> res_coord_trans_cntnr == NULL && "Not implemented" );


  // RRC Reconfiguration Complete Indicator
  // Optional
  // 9.3.1.30
  assert(m0->rrc_reconf_compl_ind == NULL && "Not implemented" );
  assert(m1->rrc_reconf_compl_ind  == NULL && "Not implemented" );


  // RRC-Container 
  // Optional
  // 9.3.1.6
  assert(m0-> rrc_cntnr == NULL && "Not implemented" );
  assert(m1-> rrc_cntnr == NULL && "Not implemented" );

  // SCell To Be Setup List
  // [0 - 32]
  assert(m0-> sz_scell_to_be_setup == 0 && "Not implemented" );
  assert(m1-> sz_scell_to_be_setup == 0 && "Not implemented" );
  assert(m0-> scell_to_be_setup == NULL && "Not implemented" );
  assert(m1-> scell_to_be_setup == NULL && "Not implemented" );


  // SCell To Be Removed List
  // [0 - 32]
  assert(m0-> sz_scell_to_be_rm == 0 && "Not implemented" );
  assert(m1-> sz_scell_to_be_rm == 0 && "Not implemented" );
  assert(m0-> scell_to_be_rm == NULL && "Not implemented" );
  assert(m1->  scell_to_be_rm== NULL && "Not implemented" );


  // SRB to Be Setup List
  // [0-8]
  assert(m0-> sz_srb_to_be_setup == 0 && "Not implemented" );
  assert(m1-> sz_srb_to_be_setup == 0 && "Not implemented" );
  assert(m0-> srb_to_be_setup == NULL && "Not implemented" );
  assert(m1-> srb_to_be_setup == NULL && "Not implemented" );


  // DRB to Be Setup List
  // [0- 64]
  assert(m0-> sz_drb_to_be_setup == 0 && "Not implemented" );
  assert(m1-> sz_drb_to_be_setup == 0 && "Not implemented" );
  assert(m0-> drb_to_be_setup == NULL && "Not implemented" );
  assert(m1-> drb_to_be_setup == NULL && "Not implemented" );


  // DRB to Be Modified List
  // [0- 64]
  assert(m0-> sz_drb_to_be_mod == 0 && "Not implemented" );
  assert(m1-> sz_drb_to_be_mod == 0 && "Not implemented" );
  assert(m0-> drb_to_be_mod == NULL && "Not implemented" );
  assert(m1-> drb_to_be_mod == NULL && "Not implemented" );


  // SRB To Be Released List
  // [0 - 8]
  assert(m0-> sz_srb_to_be_rel == 0 && "Not implemented" );
  assert(m1-> sz_srb_to_be_rel == 0 && "Not implemented" );
  assert(m0-> srb_to_be_rel == NULL && "Not implemented" );
  assert(m1-> srb_to_be_rel == NULL && "Not implemented" );


  // DRB to Be Released List
  // [0- 64]
  assert(m0-> sz_drb_to_be_rel == 0 && "Not implemented" );
  assert(m1-> sz_drb_to_be_rel == 0 && "Not implemented" );
  assert(m0-> drb_to_be_rel == NULL && "Not implemented" );
  assert(m1-> drb_to_be_rel == NULL && "Not implemented" );


  // BH RLC Channel to be Modified List
  // [0 - 65536]
  assert(m0-> sz_bh_rlc_chn_to_be_mod == 0 && "Not implemented" );
  assert(m1-> sz_bh_rlc_chn_to_be_mod == 0 && "Not implemented" );
  assert(m0->  bh_rlc_chn_to_be_mod== NULL && "Not implemented" );
  assert(m1->  bh_rlc_chn_to_be_mod== NULL && "Not implemented" );


  // BH RLC Channel to be Released List
  // [0 - 65536]
  assert(m0-> sz_bh_rlc_chn_to_be_rel == 0 && "Not implemented" );
  assert(m1-> sz_bh_rlc_chn_to_be_rel == 0 && "Not implemented" );
  assert(m0->  bh_rlc_chn_to_be_rel== NULL && "Not implemented" );
  assert(m1->  bh_rlc_chn_to_be_rel== NULL && "Not implemented" );

  // SL DRB to Be Setup List
  // [0 - 512]
  assert(m0-> sz_sl_drb_to_be_setup == 0 && "Not implemented" );
  assert(m1-> sz_sl_drb_to_be_setup == 0 && "Not implemented" );
  assert(m0->  sl_drb_to_be_setup== NULL && "Not implemented" );
  assert(m1->  sl_drb_to_be_setup== NULL && "Not implemented" );


  // SL DRB to Be Modified List
  // [0 - 512]
  assert(m0-> sz_sl_drb_to_be_mod == 0 && "Not implemented" );
  assert(m1-> sz_sl_drb_to_be_mod == 0 && "Not implemented" );
  assert(m0->  sl_drb_to_be_mod== NULL && "Not implemented" );
  assert(m1->  sl_drb_to_be_mod== NULL && "Not implemented" );


  // SL DRB to Be Released List
  // [0 - 512]
  assert(m0-> sz_sl_drb_to_be_rel == 0 && "Not implemented" );
  assert(m1-> sz_sl_drb_to_be_rel == 0 && "Not implemented" );
  assert(m0->  sl_drb_to_be_rel== NULL && "Not implemented" );
  assert(m1-> sl_drb_to_be_rel == NULL && "Not implemented" );

  // Conditional Intra-DU Mobility Information
  // Optional
  assert(m0-> intra_du_mob_info == NULL && "Not implemented" );
  assert(m1-> intra_du_mob_info == NULL && "Not implemented" );

  // F1-C Transfer Path
  // Optional
  // 9.3.1.207
  assert(m0->f1_c_path_nsa == NULL && "Not implemented" );
  assert(m1->f1_c_path_nsa == NULL && "Not implemented" );

  return true;
}


