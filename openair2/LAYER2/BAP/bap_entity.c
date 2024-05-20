#include "bap_entity.h"

bap_entity_t *new_bap_entity(uint16_t bap_address, bool is_du){
  bap_entity_t *ret;
  ret = calloc(1, sizeof(bap_entity_t));
  if (ret == NULL) {
    // Change to BAP log
    LOG_E(RLC, "%s:%d:%s: out of memory\n", __FILE__, __LINE__, __FUNCTION__);
    exit(1);
  }
  
  if (pthread_mutex_init(&ret->lock, NULL)) abort();

  ret->is_du = is_du;
  ret->bap_address = bap_address;
  return (bap_entity_t *)ret;
}
