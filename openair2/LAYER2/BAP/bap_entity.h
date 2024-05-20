
#include <stdint.h>
#include <pthread.h>

typedef struct{
    uint16_t bap_address;
    uint16_t bap_path_id;
}bap_routing_id_t;

typedef struct{
    uint16_t priorHop_bap_address;
    uint16_t ingress_bhch_ID;
    uint16_t nextHop_bap_address;
    uint16_t egress_bhch_ID;
} bap_bhch_mapping_info_t;

typedef struct bap_entity_t{
    pthread_mutex_t lock;
    bool is_du; // true if DU, false if MT
    uint16_t bap_address;
    bap_routing_id_t defaultUL_bap_routing_id;
    uint16_t befaultUL_bh_rlc_channel_id;
    enum{
        PER_BH_RLC_Channel,
        PER_ROUTING_ID,
        BOTH
    }flow_control_feedback_type;

    // From BAPlayerBHRLCchannelMappingInfo
    // Use mappingInformationIndex to navigate the list
    bap_bhch_mapping_info_t *bhch_mapping_info_list;
}bap_entity_t;

bap_entity_t *new_bap_entity(uint16_t bap_address, bool is_du);