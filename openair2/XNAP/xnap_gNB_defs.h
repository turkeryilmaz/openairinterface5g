#include <stdint.h>

#include "queue.h"
#include "tree.h"

#include "sctp_eNB_defs.h"
#include "s1ap_messages_types.h"
#include "xnap_messages_types.h"

#include "xnap_ids.h"
#include "xnap_timers.h"

#ifndef XNAP_GNB_DEFS_H_
#define XNAP_GNB_DEFS_H_

#define XNAP_GNB_NAME_LENGTH_MAX    (150)

typedef enum {
  /* Disconnected state: initial state for any association. */
  XNAP_GNB_STATE_DISCONNECTED = 0x0,

  /* State waiting for xn Setup response message if the target gNB accepts or
   * Xn Setup failure if rejects the gNB.
   */
  XNAP_GNB_STATE_WAITING     = 0x1,

  /* The gNB is successfully connected to another gNB. */
  XNAP_GNB_STATE_CONNECTED   = 0x2,

  /* XnAP is ready, and the gNB is successfully connected to another gNB. */
  XNAP_GNB_STATE_READY             = 0x3,

  XNAP_GNB_STATE_OVERLOAD          = 0x4,

  XNAP_GNB_STATE_RESETTING         = 0x5,

  /* Max number of states available */
  XNAP_GNB_STATE_MAX,
} xnap_gNB_state_t;

/* Served PLMN identity element */
/*struct plmn_identity_s {
  uint16_t mcc;
  uint16_t mnc;
  uint8_t  mnc_digit_length;
  STAILQ_ENTRY(plmn_identity_s) next;
};*/

/* Served group id element */
/*struct served_group_id_s {
  uint16_t gnb_group_id;
  STAILQ_ENTRY(served_group_id_s) next;
};*/

/* Served enn code for a particular gNB */
struct gnb_code_s {
  uint8_t gnb_code;
  STAILQ_ENTRY(gnb_code_s) next;
};

struct xnap_gNB_instance_s;

/* This structure describes association of a eNB to another eNB */
typedef struct xnap_gNB_data_s {
  /* eNB descriptors tree, ordered by sctp assoc id */
  RB_ENTRY(xnap_gNB_data_s) entry;

  /* This is the optional name provided by the MME */
  char *gNB_name;

  /*  target eNB ID */
  uint32_t gNB_id;

  /* Current eNB load information (if any). */
  //x2ap_load_state_t overload_state;

  /* Current eNB->eNB X2AP association state */
  xnap_gNB_state_t state;

  /* Next usable stream for UE signalling */
  int32_t nextstream;

  /* Number of input/ouput streams */
  uint16_t in_streams;
  uint16_t out_streams;

  /* Connexion id used between SCTP/X2AP */
  uint16_t cnx_id;

  /* SCTP association id */
  int32_t  assoc_id;

  /* Nid cells */
  uint32_t                Nid_cell[MAX_NUM_CCs];
  int                     num_cc;
  /*Frequency band of NR neighbor cell supporting ENDC NSA */
  uint32_t                servedNrCell_band[MAX_NUM_CCs];

  /* Only meaningfull in virtual mode */
  struct xnap_gNB_instance_s *xnap_gNB_instance;
} xnap_gNB_data_t;

typedef struct xnap_gNB_instance_s {
  /* used in simulation to store multiple gNB instances*/
  STAILQ_ENTRY(xnap_gNB_instance_s) xnap_gNB_entries;

  /* Number of target gNBs requested by gNB (tree size) */
  uint32_t xn_target_gnb_nb;
  /* Number of target gNBs for which association is pending */
  uint32_t xn_target_gnb_pending_nb;
  /* Number of target gNB successfully associated to gNB */
  uint32_t xn_target_gnb_associated_nb;
  /* Tree of XNAP gNB associations ordered by association ID */
  RB_HEAD(xnap_gnb_map, xnap_gNB_data_s) xnap_gnb_head;

  /* Tree of UE ordered by eNB_ue_x2ap_id's */
  //  RB_HEAD(x2ap_ue_map, x2ap_eNB_ue_context_s) x2ap_ue_head;

  /* For virtual mode, mod_id as defined in the rest of the L1/L2 stack */
  instance_t instance;

  /* Displayable name of eNB */
  char *gNB_name;

  /* Unique eNB_id to identify the eNB within EPC.
   * In our case the eNB is a macro eNB so the id will be 20 bits long.
   * For Home eNB id, this field should be 28 bits long.
   */
  uint32_t gNB_id;
  /* The type of the cell */
  cell_type_t cell_type;

  /* Tracking area code */
  uint16_t tac;

  /* Mobile Country Code
   * Mobile Network Code
   */
  uint16_t  mcc;
  uint16_t  mnc;
  uint8_t   mnc_digit_length;

  /* CC params */
  int16_t                 eutra_band[MAX_NUM_CCs];
  uint32_t                downlink_frequency[MAX_NUM_CCs];
  int32_t                 uplink_frequency_offset[MAX_NUM_CCs];
  uint32_t                Nid_cell[MAX_NUM_CCs];;
  int16_t                 N_RB_DL[MAX_NUM_CCs];
  int16_t                 N_RB_UL[MAX_NUM_CCs];
  frame_type_t            frame_type[MAX_NUM_CCs];
  uint32_t                fdd_earfcn_DL[MAX_NUM_CCs];
  uint32_t                fdd_earfcn_UL[MAX_NUM_CCs];
  uint32_t                subframeAssignment[MAX_NUM_CCs];
  uint32_t                specialSubframe[MAX_NUM_CCs];
  uint32_t                 nr_band[MAX_NUM_CCs];
  uint32_t		  tdd_nRARFCN[MAX_NUM_CCs];
  uint32_t		  nrARFCN[MAX_NUM_CCs];
  int16_t                 nr_SCS[MAX_NUM_CCs];

  int                     num_cc;

  net_ip_address_t target_gnb_xn_ip_address[XNAP_MAX_NB_GNB_IP_ADDRESS];
  uint8_t          nb_xn;
  net_ip_address_t gnb_xn_ip_address;
  uint16_t         sctp_in_streams;
  uint16_t         sctp_out_streams;
  uint32_t         gnb_port_for_XNC;
  int              multi_sd;

  xnap_id_manager  id_manager;
  xnap_timers_t    timers;
} xnap_gNB_instance_t;

typedef struct {
  /* List of served eNBs
   * Only used for virtual mode
   */
  STAILQ_HEAD(xnap_gNB_instances_head_s, xnap_gNB_instance_s) xnap_gNB_instances_head;
  /* Nb of registered eNBs */
  uint8_t nb_registered_gNBs;

  /* Generate a unique connexion id used between X2AP and SCTP */
  uint16_t global_cnx_id;
} xnap_gNB_internal_data_t;

int xnap_gNB_compare_assoc_id(struct xnap_gNB_data_s *p1, struct xnap_gNB_data_s *p2);

/* Generate the tree management functions */
struct xnap_gNB_map;
struct xnap_gNB_data_s;
RB_PROTOTYPE(xnap_gNB_map, xnap_gNB_data_s, entry, xnap_gNB_compare_assoc_id);


#endif /* XNAP_GNB_DEFS_H_ */
