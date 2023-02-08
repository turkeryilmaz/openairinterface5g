#ifndef SL_DRB_FAILED_F1AP_H
#define SL_DRB_FAILED_F1AP_H 

#include <stdlib.h>
#include <stdint.h>
#include "cause_f1ap.h"

typedef struct{

  // SL DRB ID
  // Mandatory 
  // 9.3.1.120
  // [1-512]
  uint16_t sl_drb_id;  

  // Cause
  // Optional
  // 9.3.1.2
  cause_f1ap_t* cause;

} sl_drb_failed_t;

#endif

