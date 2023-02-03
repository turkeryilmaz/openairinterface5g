#ifndef COPY_ASN_TO_PDU_MIR_H
#define COPY_ASN_TO_PDU_MIR_H 

#include "f1ap_types/f1_setup.h"
#include "f1ap_types/f1_setup_response.h"
#include "f1ap_types/f1_setup_failure.h"
#include "f1ap_types/ue_ctx_setup_request.h"
#include "f1ap_types/ue_ctx_setup_response.h"
#include "f1ap_types/gnb_cu_conf_update.h"
#include "f1ap_types/gnb_cu_conf_update_ack.h"
#include "f1ap_types/init_ul_rrc_msg.h"
#include "f1ap_types/ul_rrc_msg.h"

#include "../../../cmake_targets/ran_build/build/CMakeFiles/F1AP_R16.3.1/F1AP_F1AP-PDU.h"

////////////////
// F1AP Messages
////////////////

f1_setup_t cp_f1_setup_ir(F1AP_F1AP_PDU_t const* src);

f1_setup_response_t cp_f1_setup_response_ir(F1AP_F1AP_PDU_t const* src);

f1_setup_failure_t cp_f1_setup_failure_ir(F1AP_F1AP_PDU_t const* src);

ue_ctx_setup_request_t cp_ue_ctx_setup_request_ir(F1AP_F1AP_PDU_t const* src);

ue_ctx_setup_response_t cp_ue_ctx_setup_response_ir(F1AP_F1AP_PDU_t const* src);

gnb_cu_conf_update_t cp_gnb_cu_conf_update_ir(F1AP_F1AP_PDU_t const* src);

gnb_cu_conf_update_ack_t cp_gnb_cu_conf_update_ack_ir(F1AP_F1AP_PDU_t const* src);

init_ul_rrc_msg_t cp_init_ul_rrc_msg_ir(F1AP_F1AP_PDU_t const* src_pdu);

ul_rrc_msg_t cp_ul_rrc_msg_ir(F1AP_F1AP_PDU_t const* src_pdu); 

////////////////
////////////////
////////////////

#endif

