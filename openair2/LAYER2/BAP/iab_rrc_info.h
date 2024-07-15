#include <stdbool.h>
#include <stdint.h>

typedef struct bap_routingId_s{
  uint16_t bap_address;
  uint16_t bap_pathId;
} bap_routingId_t;

typedef struct bap_config_s{
  uint16_t bap_address;
  bap_routingId_t defaultUL_bap_routingID;
  uint16_t defaultUL_bhch;
  
  enum{
    perBH_RLC_Channel,
    perRoutingID,
    both,
  }flowControlFeedbackType;
} bap_config_t;
