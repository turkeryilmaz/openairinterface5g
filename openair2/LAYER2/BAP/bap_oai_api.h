#include <stdint.h>

void bap_add_bhch(int rnti, int bhch_id, const NR_BH_RLC_ChannelConfig_r16_t *rlc_bhchConfig);
void nr_bap_layer_init(bool is_du);
void init_bap_entity(uint16_t bap_addr, bool is_du);