#ifndef SRB_MODIFIED_ITEM_F1AP_H
#define SRB_MODIFIED_ITEM_F1AP_H 

#include <stdint.h>

typedef struct{

  // SRB ID
  // 9.3.1.7
  // [0-3 ]
  uint8_t srb_id;

  // LCID
  // 9.3.1.35
  // [1 - 32]
  uint8_t lc_id;

} srb_mod_item_t; 

#endif
