
#include <stdbool.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/sctp.h>

#ifndef BAP_INFO_H_
#define BAP_INFO_H_

typedef struct bhch_t{
  uint16_t bhch_id;
  rlc_mode_t rlc_mode;
  long lcid;
  long priority;
  uint16_t donor_bap_address;
  uint16_t node_bap_address[30];
  uint32_t number_of_nodes;
}bhch_t;

typedef struct iab_donor_du_t {
  uint64_t du_id;
  uint16_t bap_address;
} iab_donor_du_t;

typedef struct iab_node_du_t {
  uint64_t du_id;
} iab_node_du_t;

typedef struct iab_mt_t {
  uint64_t mt_id; // Don't know if this will useful
  uint32_t rrc_ue_id;
} iab_mt_t;

typedef struct iab_node_t {
  uint16_t bap_address;
  iab_node_du_t iab_node_du;
  iab_mt_t iab_mt;
  uint16_t donor_bap_address;
} iab_node_t;

typedef struct iab_cu_t {
  // Max 10 for now
  iab_donor_du_t iab_donor_du[10];
  int number_of_iab_donor_dus;
  // max 30 for now
  iab_node_t iab_node[30];
  int number_of_iab_nodes;
  uint16_t last_given_bap_address;
  bhch_t *bhch_list;
  int number_of_bhchs;
} iab_cu_t;

typedef struct gNB_IAB_INFO_s {
  union {
    iab_cu_t iab_cu;
    iab_donor_du_t iab_donor_du;
    iab_node_du_t iab_node_du;
    iab_mt_t iab_mt;
  };
} gNB_IAB_INFO;


#endif // BAP_INFO_H_

