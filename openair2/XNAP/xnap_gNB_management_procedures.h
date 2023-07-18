
#ifndef __XNAP_GNB_MANAGEMENT_PROCEDURES__H__
#define __XNAP_GNB_MANAGEMENT_PROCEDURES__H__

void xnap_gNB_prepare_internal_data(void);

void dump_trees(void);

void xnap_gNB_insert_new_instance(xnap_gNB_instance_t *new_instance_p);

xnap_gNB_instance_t *xnap_gNB_get_instance(uint8_t mod_id);

uint16_t xnap_gNB_fetch_add_global_cnx_id(void);

void xnap_gNB_prepare_internal_data(void);

xnap_gNB_data_t* xnap_is_gNB_id_in_list(uint32_t gNB_id);

xnap_gNB_data_t* xnap_is_gNB_assoc_id_in_list(uint32_t sctp_assoc_id);

xnap_gNB_data_t* xnap_is_gNB_pci_in_list (const uint32_t pci);

struct xnap_gNB_data_s *xnap_get_gNB(xnap_gNB_instance_t *instance_p,
                                     int32_t assoc_id,
                                     uint16_t cnx_id);

#endif /* __XNAP_GNB_MANAGEMENT_PROCEDURES__H__ */
