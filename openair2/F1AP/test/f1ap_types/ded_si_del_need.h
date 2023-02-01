#ifndef F1AP_DEDICATED_SI_DELIVERY_NEEDED_UE_LIST_H 
#define F1AP_DEDICATED_SI_DELIVERY_NEEDED_UE_LIST_H 

#include "nr_cgi.h"

typedef struct{

  // Mnadatory
  // 9.3.1.4
  // gNB-CU UE F1AP ID
  uint32_t gnb_cu_ue_id; // [0, 2^32 -1]

  // Mandatory
  // 9.3.1.12
  // NR CGI
  nr_cgi_t nr_cgi;

} ded_si_del_need_t;

void free_ded_si_del_need(ded_si_del_need_t* src);

#endif

