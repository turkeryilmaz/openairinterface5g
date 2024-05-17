#include "bap_entity.h"

bap_entity_t *new_bap_entity(uint16_t bap_address){
  bap_entity_t *ret;
  ret = calloc(1, sizeof(bap_entity_t));

  if (ret == NULL) {
    LOG_E(, "%s:%d:%s: out of memory\n", __FILE__, __LINE__, __FUNCTION__);
    exit(1);
  }

  ret->bap_address = bap_address;
  return (bap_entity_t *)ret;
}

void bap_add_entity(int rnti, uint16_t bap_address)
{
  
}
