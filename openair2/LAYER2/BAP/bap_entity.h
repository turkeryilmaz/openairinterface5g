
#include <stdint.h>

typedef struct{
    uint16_t bap_address;
    uint16_t bap_path_id;
}bap_routing_id_t;

typedef struct bap_entity_t{
    uint16_t bap_address;
    bap_routing_id_t defaultUL_bap_routing_id;
    uint16_t befaultUL_bh_rlc_channel_id;
    enum{
        PER_BH_RLC_Channel,
        PER_ROUTING_ID,
        BOTH
    }flow_control_feedback_type;

}bap_entity_t;