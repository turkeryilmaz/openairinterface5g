#ifndef F1AP_GNB_CU_TNL_ASSOCIATION_FAILED_SETUP_H
#define F1AP_GNB_CU_TNL_ASSOCIATION_FAILED_SETUP_H

#include "cp_trans_layer_info.h"
#include "cause_f1ap.h"

typedef struct{

  // Mandatory
  // 9.3.2.4
  // TNL Association Transport Layer Address
  cp_trans_layer_info_t tnl_assoc_trans_layer_addr;

  // Mandatory
  // 9.3.1.2
  // Cause   
  cause_f1ap_t cause;

} gnb_cu_tnl_assoc_failed_stp_t;

void free_gnb_cu_tnl_assoc_failed_stp( gnb_cu_tnl_assoc_failed_stp_t* src);

bool eq_gnb_cu_tnl_assoc_failed_stp(gnb_cu_tnl_assoc_failed_stp_t const* m0, gnb_cu_tnl_assoc_failed_stp_t const* m1);

#endif

