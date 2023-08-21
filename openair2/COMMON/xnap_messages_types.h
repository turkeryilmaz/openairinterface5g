#ifndef XNAP_MESSAGES_TYPES_H_
#define XNAP_MESSAGES_TYPES_H_

//#include "ngap_messages_types.h"
#include "s1ap_messages_types.h"
#include "LTE_PhysCellId.h"

typedef enum {
  XNAP_CAUSE_T_DC_PREP_TIMEOUT,
  XNAP_CAUSE_T_DC_OVERALL_TIMEOUT,
  XNAP_CAUSE_RADIO_CONNECTION_WITH_UE_LOST,
} XNAP_cause_t;

//-------------------------------------------------------------------------------------------//
// Defines to access message fields.


#define XNAP_REGISTER_GNB_REQ(mSGpTR)                           (mSGpTR)->ittiMsg.xnap_register_gnb_req
#define XNAP_SETUP_REQ(mSGpTR)                                  (mSGpTR)->ittiMsg.xnap_setup_req
#define XNAP_SETUP_RESP(mSGpTR)                                 (mSGpTR)->ittiMsg.xnap_setup_resp
#define XNAP_RESET_REQ(mSGpTR)                                  (mSGpTR)->ittiMsg.xnap_reset_req
#define XNAP_RESET_RESP(mSGpTR)                                 (mSGpTR)->ittiMsg.xnap_reset_resp

#define XNAP_REGISTER_GNB_CNF(mSGpTR)                           (mSGpTR)->ittiMsg.xnap_register_gnb_cnf
#define XNAP_DEREGISTERED_GNB_IND(mSGpTR)                       (mSGpTR)->ittiMsg.xnap_deregistered_gnb_ind


#define XNAP_MAX_NB_GNB_IP_ADDRESS 6

// gNB application layer -> XNAP messages

typedef struct xnap_setup_req_s {
  uint32_t Nid_cell[MAX_NUM_CCs];
  int num_cc;
} xnap_setup_req_t;

typedef struct xnap_setup_resp_s {
  uint32_t Nid_cell[MAX_NUM_CCs];
  int num_cc;
} xnap_setup_resp_t;

typedef struct xnap_reset_req_s {
  uint32_t cause;
} xnap_reset_req_t;

typedef struct xnap_reset_resp_s {
  int dummy;
} xnap_reset_resp_t;



typedef struct xnap_register_gnb_req_s {
  /* Unique gNB_id to identify the gNB within EPC.
   * For macro gNB ids this field should be 20 bits long. true?
   * For home gNB ids this field should be 28 bits long.
   */
  uint32_t gNB_id;
  /* The type of the cell */
  enum cell_type_e cell_type;

  /* Optional name for the cell
   * NOTE: the name can be NULL (i.e no name) and will be cropped to 150
   * characters.
   */
  char *gNB_name;

  /* Tracking area code */
  uint16_t tac;

  /* Mobile Country Code
   * Mobile Network Code
   */
  uint16_t mcc;
  uint16_t mnc;
  uint8_t  mnc_digit_length;

  /*
   * CC Params
   */
  int16_t                 eutra_band[MAX_NUM_CCs];
  int32_t                 nr_band[MAX_NUM_CCs];
  int32_t                 nrARFCN[MAX_NUM_CCs];
  uint32_t                downlink_frequency[MAX_NUM_CCs];
  int32_t                 uplink_frequency_offset[MAX_NUM_CCs];
  uint32_t                Nid_cell[MAX_NUM_CCs];
  int16_t                 N_RB_DL[MAX_NUM_CCs];
  frame_type_t            frame_type[MAX_NUM_CCs];
  uint32_t                fdd_earfcn_DL[MAX_NUM_CCs];
  uint32_t                fdd_earfcn_UL[MAX_NUM_CCs];
  uint32_t                subframeAssignment[MAX_NUM_CCs];
  uint32_t                specialSubframe[MAX_NUM_CCs];
  int                     num_cc;

  /* To be considered for TDD */
  //uint16_t tdd_EARFCN;
  //uint16_t tdd_Transmission_Bandwidth;

  /* The local gNB IP address to bind */
  net_ip_address_t gnb_xn_ip_address;

  /* Nb of GNB to connect to */
  uint8_t          nb_xn;

  /* List of target gNB to connect to for Xn*/
  net_ip_address_t target_gnb_xn_ip_address[XNAP_MAX_NB_GNB_IP_ADDRESS];

  /* Number of SCTP streams used for associations */
  uint16_t sctp_in_streams;
  uint16_t sctp_out_streams;

  /*gNB port for XNC*/
  uint32_t gnb_port_for_XNC;

  /* timers (unit: millisecond) */
  int t_reloc_prep;
  int txn_reloc_overall;
  int t_dc_prep;
  int t_dc_overall;
} xnap_register_gnb_req_t;

typedef struct xnap_subframe_process_s {
  /* nothing, we simply use the module ID in the header */
  // This dummy element is to avoid CLANG warning: empty struct has size 0 in C, size 1 in C++
  // To be removed if the structure is filled
  uint32_t dummy;
} xnap_subframe_process_t;

//-------------------------------------------------------------------------------------------//
// XNAP -> gNB application layer messages
typedef struct xnap_register_gnb_cnf_s {
  /* Nb of connected gNBs*/
  uint8_t          nb_xn;
} xnap_register_gnb_cnf_t;

typedef struct xnap_deregistered_gnb_ind_s {
  /* Nb of connected gNBs */
  uint8_t          nb_xn;
} xnap_deregistered_gnb_ind_t;

//-------------------------------------------------------------------------------------------//
// XNAP <-> RRC
typedef struct xnap_guami_s {
  uint16_t mcc;
  uint16_t mnc;
  uint8_t  mnc_len;
  uint8_t  mme_code;
  uint16_t mme_group_id;
} xnap_guami_t;


typedef struct XNAP_ENDC_setup_req_s {
  uint32_t Nid_cell[MAX_NUM_CCs];
  int num_cc;
  uint32_t servedNrCell_band[MAX_NUM_CCs];
} XNAP_ENDC_setup_req_t;







#endif /* XNAP_MESSAGES_TYPES_H_ */
