
/* gNB application layer -> XNAP messages */
/* ITTI LOG messages */
/* ENCODER */
MESSAGE_DEF(XNAP_RESET_REQUST_LOG               , MESSAGE_PRIORITY_MED, IttiMsgText                      , xnap_reset_request_log)
MESSAGE_DEF(XNAP_RESOURCE_STATUS_RESPONSE_LOG   , MESSAGE_PRIORITY_MED, IttiMsgText                      , xnap_resource_status_response_log)
MESSAGE_DEF(XNAP_RESOURCE_STATUS_FAILURE_LOG    , MESSAGE_PRIORITY_MED, IttiMsgText                      , xnap_resource_status_failure_log)

/* Messages for XNAP logging */
MESSAGE_DEF(XNAP_SETUP_REQUEST_LOG              , MESSAGE_PRIORITY_MED, IttiMsgText                      , xnap_setup_request_log)


/* gNB application layer -> XNAP messages */
MESSAGE_DEF(XNAP_REGISTER_GNB_REQ               , MESSAGE_PRIORITY_MED, xnap_register_gnb_req_t          , xnap_register_gnb_req)
MESSAGE_DEF(XNAP_SUBFRAME_PROCESS               , MESSAGE_PRIORITY_MED, xnap_subframe_process_t          , xnap_subframe_process)
MESSAGE_DEF(XNAP_RESET_REQ                      , MESSAGE_PRIORITY_MED, xnap_reset_req_t                 , xnap_reset_req)
/* XNAP -> gNB application layer messages */
MESSAGE_DEF(XNAP_REGISTER_GNB_CNF               , MESSAGE_PRIORITY_MED, xnap_register_gnb_cnf_t          , xnap_register_gnb_cnf)
MESSAGE_DEF(XNAP_DEREGISTERED_GNB_IND           , MESSAGE_PRIORITY_MED, xnap_deregistered_gnb_ind_t      , xnap_deregistered_gnb_ind)

/* handover messages XNAP <-> RRC */
MESSAGE_DEF(XNAP_SETUP_REQ                      , MESSAGE_PRIORITY_MED, xnap_setup_req_t                 , xnap_setup_req)
MESSAGE_DEF(XNAP_SETUP_RESP                     , MESSAGE_PRIORITY_MED, xnap_setup_resp_t                , xnap_setup_resp)
/*MESSAGE_DEF(XNAP_HANDOVER_REQ                   , MESSAGE_PRIORITY_MED, xnap_handover_req_t              , xnap_handover_req)
MESSAGE_DEF(XNAP_HANDOVER_REQ_ACK               , MESSAGE_PRIORITY_MED, xnap_handover_req_ack_t          , xnap_handover_req_ack)
MESSAGE_DEF(XNAP_HANDOVER_CANCEL                , MESSAGE_PRIORITY_MED, xnap_handover_cancel_t           , xnap_handover_cancel) */

/* handover messages XNAP <-> S1AP */
/*MESSAGE_DEF(XNAP_UE_CONTEXT_RELEASE             , MESSAGE_PRIORITY_MED, xnap_ue_context_release_t        , xnap_ue_context_release) */

/* Sgnb bearer addition messages XNAP <-> RRC */
/*MESSAGE_DEF(XNAP_SGNB_ADDITION_REQ              , MESSAGE_PRIORITY_MED, xnap_sgnb_addition_req_t         , xnap_sgnb_addition_req)*/

/* SGnb bearer addition messages XNAP <-> RRC */
/*MESSAGE_DEF(XNAP_ENDC_SETUP_REQ                 , MESSAGE_PRIORITY_MED, xnap_ENDC_setup_req_t            , xnap_ENDC_setup_req)
MESSAGE_DEF(XNAP_ENDC_SGNB_ADDITION_REQ         , MESSAGE_PRIORITY_MED, xnap_ENDC_sgnb_addition_req_t    , xnap_ENDC_sgnb_addition_req)
MESSAGE_DEF(XNAP_ENDC_SGNB_ADDITION_REQ_ACK     , MESSAGE_PRIORITY_MED, xnap_ENDC_sgnb_addition_req_ACK_t, xnap_ENDC_sgnb_addition_req_ACK)
MESSAGE_DEF(XNAP_ENDC_SGNB_RECONF_COMPLETE      , MESSAGE_PRIORITY_MED, xnap_ENDC_reconf_complete_t      , xnap_ENDC_sgnb_reconf_complete) */

/* SGnb UE releases ('request' is for gnb starting the process, 'required' for gnb) */
/*MESSAGE_DEF(XNAP_ENDC_SGNB_RELEASE_REQUEST      , MESSAGE_PRIORITY_MED, xnap_ENDC_sgnb_release_request_t , xnap_ENDC_sgnb_release_request)
MESSAGE_DEF(XNAP_ENDC_SGNB_RELEASE_REQUIRED     , MESSAGE_PRIORITY_MED, xnap_ENDC_sgnb_release_required_t, xnap_ENDC_sgnb_release_required)*/

/* ENDC timers' timeout XNAP <-> RRC */
/*MESSAGE_DEF(XNAP_ENDC_DC_PREP_TIMEOUT           , MESSAGE_PRIORITY_MED, xnap_ENDC_dc_prep_timeout_t      , xnap_ENDC_dc_prep_timeout)
MESSAGE_DEF(XNAP_ENDC_DC_OVERALL_TIMEOUT        , MESSAGE_PRIORITY_MED, xnap_ENDC_dc_overall_timeout_t   , xnap_ENDC_dc_overall_timeout) */
