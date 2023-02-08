#include "ue_ctx_mod_resp.h"

#include <assert.h>

void free_ue_ctx_mod_resp(ue_ctx_mod_resp_t* src)
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

  // Resource Coordination Transfer Container
  // Optional
  assert(src->res_coord_trans == NULL && "Not implemented"); 

  // DU To CU RRC Information
  // Optional
  // 9.3.1.26
  assert(src->du_to_cu_rrc == NULL && "Not implemented");

  // DRB Setup List
  // [0 - 64]
  assert(src->sz_drb_setup == 0 && "Not implemented");
  assert(src->drb_setup == NULL && "Not implemented"); 

  // DRB Modified List
  // [0 - 64]
  assert(src->sz_drb_mod == 0 && "Not implemented");
  assert(src->drb_mod == NULL && "Not implemented");

  // SRB Failed to be Setup List
  // [0 - 8] 
  assert(src->sz_srb_failed == 0 && "Not implemented");
  assert(src->srb_failed_setup == NULL && "Not implemented");

  // SCell Failed To Setup List
  // [0 - 32]
  assert(src->sz_scell_failed_setup == 0 && "Not implemented");
  assert(src-> scell_failed_setup == NULL && "Not implemented");

  // DRB Failed to be Modified List
  // [0 - 64]
  assert(src->sz_drb_fail == 0 && "Not implemented");
  assert(src->drb_failed_setup == NULL && "Not implemented");

  // SRB Modified List
  // [0 - 8]
  assert(src->sz_srb_mod == 0 && "Not implemented");
  assert(src-> srb_mod == NULL && "Not implemented");

  // Full Configuration
  // Optional
  assert(src->full_conf == NULL && "Not implemented");

  // BH RLC Channel Setup List
  // [0 - 65536]
  // 9.3.1.113
  // BIT STRING (SIZE(16))
  assert(src->sz_bh_rlc_chn_stp == 0 && "Not implemented");
  assert(src->bh_rlc_chn_stp == NULL && "Not implemented"); 

  // BH RLC Channel Failed to be Setup List
  // [0 - 65536]
  assert(src->sz_bh_rlc_chn_failed_tbs == 0 && "Not implemented");
  assert(src->bh_rlc_chn_failed_tbs == NULL && "Not implemented");

  // BH RLC Channel Modified List
  // [0 - 65536]
  // 9.3.1.113
  // BIT STRING (SIZE(16))
  assert(src->sz_bh_rlc_chn_mod == 0 && "Not implemented");
  assert(src->bh_rlc_chn_mod == NULL && "Not implemented"); 

  // BH RLC Channel Failed to be Modified List
  // [0 - 65536]
  assert(src->sz_bh_rlc_chn_failed_mod == 0 && "Not implemented");
  assert(src->bh_rlc_chn_failed_mod == NULL && "Not implemented");

  // SL DRB Setup List
  // [0 - 512]
  // 9.3.1.120 
  assert(src->sz_sl_drb_stp == 0 && "Not implemented");  
  assert(src->sl_drb_stp == NULL && "Not implemented"); // [1-512]

  // SL DRB Modified List
  // [0 - 512]
  // 9.3.1.120 
  assert(src->sz_sl_drb_mod == 0 && "Not implemented");  
  assert(src->sl_drb_mod == NULL && "Not implemented"); // [1-512]

  // SL DRB Failed To Setup List
  // [0 - 512]
  assert(src->sz_sl_drb_failed_stp == 0 && "Not implemented"); 
  assert(src-> sl_drb_failed_stp == NULL && "Not implemented");

  //SL DRB Failed To be Modified List
  // [0 - 512]
  assert(src->sz_sl_drb_fail_mod == 0 && "Not implemented"); 
  assert(src->sl_drb_fail_mod == NULL && "Not implemented");

  // Requested Target Cell ID
  // Mandatory
  assert(src->req_target_cell_id == NULL && "Not implemented");

}

bool eq_ue_ctx_mod_resp(ue_ctx_mod_resp_t* m0, ue_ctx_mod_resp_t* m1)
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
  if(m0->gnb_cu_ue != m1-> gnb_cu_ue)
    return false;

  // gNB-DU UE F1AP ID
  // Mandatory
  // 9.3.1.5
  if(m0->gnb_du_ue != m1-> gnb_du_ue)
    return false;

  // Resource Coordination Transfer Container
  // Optional
  assert(m0->res_coord_trans == NULL && "Not implemented"); 
  assert(m1->res_coord_trans == NULL && "Not implemented"); 

  // DU To CU RRC Information
  // Optional
  // 9.3.1.26
  assert(m0->du_to_cu_rrc == NULL && "Not implemented");
  assert(m1->du_to_cu_rrc == NULL && "Not implemented");

  // DRB Setup List
  // [0 - 64]
  assert(m0->sz_drb_setup == 0 && "Not implemented");
  assert(m0->drb_setup == NULL && "Not implemented"); 
  assert(m1->sz_drb_setup == 0 && "Not implemented");
  assert(m1->drb_setup == NULL && "Not implemented"); 

  // DRB Modified List
  // [0 - 64]
  assert(m0->sz_drb_mod == 0 && "Not implemented");
  assert(m0->drb_mod == NULL && "Not implemented");
  assert(m1->sz_drb_mod == 0 && "Not implemented");
  assert(m1->drb_mod == NULL && "Not implemented");

  // SRB Failed to be Setup List
  // [0 - 8] 
  assert(m0->sz_srb_failed == 0 && "Not implemented");
  assert(m0->srb_failed_setup == NULL && "Not implemented");
  assert(m1->sz_srb_failed == 0 && "Not implemented");
  assert(m1->srb_failed_setup == NULL && "Not implemented");

  // SCell Failed To Setup List
  // [0 - 32]
  assert(m0->sz_scell_failed_setup == 0 && "Not implemented");
  assert(m0-> scell_failed_setup == NULL && "Not implemented");
  assert(m1->sz_scell_failed_setup == 0 && "Not implemented");
  assert(m1-> scell_failed_setup == NULL && "Not implemented");

  // DRB Failed to be Modified List
  // [0 - 64]
  assert(m0->sz_drb_fail == 0 && "Not implemented");
  assert(m0->drb_failed_setup == NULL && "Not implemented");
  assert(m1->sz_drb_fail == 0 && "Not implemented");
  assert(m1->drb_failed_setup == NULL && "Not implemented");

  // SRB Modified List
  // [0 - 8]
  assert(m0->sz_srb_mod == 0 && "Not implemented");
  assert(m0-> srb_mod == NULL && "Not implemented");
  assert(m1->sz_srb_mod == 0 && "Not implemented");
  assert(m1-> srb_mod == NULL && "Not implemented");

  // Full Configuration
  // Optional
  assert(m0->full_conf == NULL && "Not implemented");
  assert(m1->full_conf == NULL && "Not implemented");

  // BH RLC Channel Setup List
  // [0 - 65536]
  // 9.3.1.113
  // BIT STRING (SIZE(16))
  assert(m0->sz_bh_rlc_chn_stp == 0 && "Not implemented");
  assert(m0->bh_rlc_chn_stp == NULL && "Not implemented"); 
  assert(m1->sz_bh_rlc_chn_stp == 0 && "Not implemented");
  assert(m1->bh_rlc_chn_stp == NULL && "Not implemented"); 

  // BH RLC Channel Failed to be Setup List
  // [0 - 65536]
  assert(m0->sz_bh_rlc_chn_failed_tbs == 0 && "Not implemented");
  assert(m0->bh_rlc_chn_failed_tbs == NULL && "Not implemented");
  assert(m1->sz_bh_rlc_chn_failed_tbs == 0 && "Not implemented");
  assert(m1->bh_rlc_chn_failed_tbs == NULL && "Not implemented");

  // BH RLC Channel Modified List
  // [0 - 65536]
  // 9.3.1.113
  // BIT STRING (SIZE(16))
  assert(m0->sz_bh_rlc_chn_mod == 0 && "Not implemented");
  assert(m0->bh_rlc_chn_mod == NULL && "Not implemented"); 
  assert(m1->sz_bh_rlc_chn_mod == 0 && "Not implemented");
  assert(m1->bh_rlc_chn_mod == NULL && "Not implemented"); 

  // BH RLC Channel Failed to be Modified List
  // [0 - 65536]
  assert(m0->sz_bh_rlc_chn_failed_mod == 0 && "Not implemented");
  assert(m0->bh_rlc_chn_failed_mod == NULL && "Not implemented");
  assert(m1->sz_bh_rlc_chn_failed_mod == 0 && "Not implemented");
  assert(m1->bh_rlc_chn_failed_mod == NULL && "Not implemented");

  // SL DRB Setup List
  // [0 - 512]
  // 9.3.1.120 
  assert(m0->sz_sl_drb_stp == 0 && "Not implemented");  
  assert(m0->sl_drb_stp == NULL && "Not implemented"); // [1-512]
  assert(m1->sz_sl_drb_stp == 0 && "Not implemented");  
  assert(m1->sl_drb_stp == NULL && "Not implemented"); // [1-512]

  // SL DRB Modified List
  // [0 - 512]
  // 9.3.1.120 
  assert(m0->sz_sl_drb_mod == 0 && "Not implemented");  
  assert(m0->sl_drb_mod == NULL && "Not implemented"); // [1-512]
  assert(m1->sz_sl_drb_mod == 0 && "Not implemented");  
  assert(m1->sl_drb_mod == NULL && "Not implemented"); // [1-512]

  // SL DRB Failed To Setup List
  // [0 - 512]
  assert(m0->sz_sl_drb_failed_stp == 0 && "Not implemented"); 
  assert(m0-> sl_drb_failed_stp == NULL && "Not implemented");
  assert(m1->sz_sl_drb_failed_stp == 0 && "Not implemented"); 
  assert(m1-> sl_drb_failed_stp == NULL && "Not implemented");

  //SL DRB Failed To be Modified List
  // [0 - 512]
  assert(m0->sz_sl_drb_fail_mod == 0 && "Not implemented"); 
  assert(m0->sl_drb_fail_mod == NULL && "Not implemented");
  assert(m1->sz_sl_drb_fail_mod == 0 && "Not implemented"); 
  assert(m1->sl_drb_fail_mod == NULL && "Not implemented");

  // Requested Target Cell ID
  // Mandatory
  assert(m0->req_target_cell_id == NULL && "Not implemented");
  assert(m1->req_target_cell_id == NULL && "Not implemented");

  return true;
}

