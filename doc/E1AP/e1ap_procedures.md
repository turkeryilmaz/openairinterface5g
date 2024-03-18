<table style="border-collapse: collapse; border: none;">
  <tr style="border-collapse: collapse; border: none;">
    <td style="border-collapse: collapse; border: none;">
      <a href="http://www.openairinterface.org/">
         <img src="./images/oai_final_logo.png" alt="" border=3 height=50 width=150>
         </img>
      </a>
    </td>
    <td style="border-collapse: collapse; border: none; vertical-align: center;">
      <b><font size = "5">E1AP Procedures</font></b>
    </td>
  </tr>
</table>

[[_TOC_]]

# Introduction
The E1 interface is between the gNB-CU-CP (Central Unit - Control Plane) and gNB-CU-UP (Central Unit - User Plane) nodes. This interface is governed by the E1 Application Protocol (E1AP) outlined in the 3GPP release 16 specifications, specifically in the documents:
* 3GPP TS 38.463 - E1 Application Protocol (E1AP)
* 3GPP TS 38.460 - E1 general aspects and principles
* 3GPP TS 38.461 - E1 interface: layer 1 
* 3GPP TS 38.462 - E1 interface: signaling transport

The E1AP protocol consists of the following sets of functions:
* E1 Interface Management functions
* E1 Bearer Context Management functions
* TEID allocation function

## E1 Bearer Context Management function
This function handles the establishment, modification, and release of E1 bearer contexts.
* E1 Bearer Context Establishment: initiation of E1 bearer context is by gNB-CU-CP and acceptance or rejection is determined by gNB-CU-UP based on admission control criteria (e.g., resource availability).
* E1 Bearer Context Modification: can be initiated by either gNB-CU-CP or gNB-CU-UP, with the receiving node having the authority to accept or reject the modification.
* Release of Bearer Context: can be triggered either directly by gNB-CU-CP or following a request from gNB-CU-UP.
* QoS-Flow to DRB Mapping Configuration: responsible for setting up and modifying the QoS-flow to DRB mapping configuration. gNB-CU-CP decides the flow-to-DRB mapping, generates SDAP and PDCP configurations, and provides them to gNB-CU-UP.

# OAI implementation

For the E1AP design in OAI, please refer to the [E1 Design](./E1-design.md) document.

## E1 re-establishment

The purpose of this procedure is to follow-up the re-establishment of RRC connection over the E1 interface. For all activated DRBs a Bearer Context Modification from CU-CP to CU-UP is necessary, according to clause 9.2.2.4 of 3GPP TS 38.463. If any modification to the bearer context is required, the CU-CP informs the CU-UP with the relevant IEs (e.g. in case of PDCP re-establishment, PDCP Configuration IE clause 9.3.1.38), Current implementation in OAI:

```mermaid
sequenceDiagram
  participant e as UE
  participant d as DU
  participant c as CUCP
  participant u as CUUP

  Note over e,c: RRCReestablishment Procedure
  e->>d: RRCReestablishmentRequest
  Note over d: initial_ul_rrc_message_transfer_f1ap
  d->>+c: Initial UL RRC Message Transfer (CCCH, new C-RNTI)
  Note over c: rrc_gNB_process_initial_ul_rrc_message
  Note over c: rrc_handle_RRCReestablishmentRequests
  c-->>+d: DL RRC Message Transfer
  note right of d: fallback to RRC establishment
  d-->>-e: RRCSetup
  e-->>d: RRCSetupComplete
  Note over c: rrc_gNB_generate_RRCReestablishment
  Note over c: cuup_notify_reestablishment
  Note over c: e1apCUCP_send_BEARER_CONTEXT_MODIFICATION_REQUEST
  c->>u: BEARER CONTEXT MODIFICATION REQUEST
  Note over u: e1apCUUP_handle_BEARER_CONTEXT_MODIFICATION_REQUEST
  Note over u: e1_bearer_context_modif
  Note over u: nr_pdcp_reestablishment
  c->>-d: DL RRC Message Transfer (old gNB DU F1AP UE ID)
  d->>e: RRCReestablishment
  Note over d: Fetch UE Context with old gNB DU F1AP UE ID and update C-RNTI
  e->>d: RRCReestablishmentComplete
  u->>c: BEARER CONTEXT MODIFICATION RESPONSE
  Note over c: e1apCUCP_handle_BEARER_CONTEXT_MODIFICATION_RESPONSE
```

## DRB Setup over E1

```mermaid
sequenceDiagram
    participant Network
    participant CUCP
    participant CUUP
    participant DU
    participant UE
    Network->>CUCP: NGAP_PDUSESSION_SETUP_REQ
    Note over CUCP: ngap_gNB_handle_pdusession_setup_request
    CUCP->>CUCP: decodePDUSessionResourceSetup
    CUCP->>CUCP: rrc_gNB_process_NGAP_PDUSESSION_SETUP_REQ
    Note over CUCP: cp_pdusession_resource_item_to_pdusession
    CUCP->>CUCP: E1AP: trigger_bearer_setup
    Note over CUCP: fill Bearer Context Setup Request
    loop nb_pdusessions_tosetup
        CUCP->>CUCP: add_pduSession (&UE->pduSessions_to_addmod)
        loop numDRB2Setup
            CUCP->>CUCP: setup_rrc_drb_for_pdu_session
            Note over CUCP: fill DRB ID, PDU Session, QoS in RRC DRB
            CUCP->>CUCP: add_rrc_drb
            CUCP->>CUCP: fill_drb_ngran_tosetup
            Note over CUCP: fill SDAP, PDCP, QoS in E1 message
        end
    end
    Note over CUCP: e1_bearer_context_setup
    CUCP->>CUUP: BEARER CONTEXT SETUP REQUEST
    Note over CUUP: e1apCUUP_handle_BEARER_CONTEXT_SETUP_REQUEST
    Note over CUUP: set up DRBs, PDCP/SDAP entities, GTP tunnel
    CUUP->>CUCP: E1AP_BEARER_CONTEXT_SETUP_RESP
    Note over CUCP: rrc_gNB_process_e1_bearer_context_setup_resp
    loop numPDUSessions
        CUCP->>CUCP: find_pduSession (&UE->pduSessions_to_addmod)
        Note over CUCP: Update N3 tunnel info in pdusession_t
        loop numDRBSetup
            CUCP->>CUCP: get_drb (from RRC)
            Note over CUCP: update DRB F1 tunnel info
        end
    end
    CUCP->>DU: F1 UE Context Setup Request / ue_context_modification_request
    DU->>CUCP: F1 UE Context Setup Response
    CUCP->>CUCP: rrc_CU_process_ue_context_setup_response / rrc_CU_process_ue_context_modification_response
    Note over CUCP: e1_send_bearer_updates
    CUCP->>CUUP: E1 BEARER CONTEXT MODIFICATION REQUEST
    CUUP->>CUCP: E1 BEARER CONTEXT MODIFICATION RESPONSE
    CUCP->>CUCP: rrc_gNB_generate_dedicatedRRCReconfiguration
    Note over CUCP: RRC_PDUSESSION_ESTABLISH
    CUCP->>DU: rrc_deliver_dl_rrc_message
    DU->>UE: RRCReconfiguration
    UE->>DU: RRCReconfigurationComplete
    DU->>CUCP: F1AP_UL_RRC_MESSAGE
    Note over CUCP: rrc_gNB_decode_dcch
    Note over CUCP: handle_rrcReconfigurationComplete
    CUCP->>CUCP: rrc_gNB_send_NGAP_INITIAL_CONTEXT_SETUP_RESP / rrc_gNB_send_NGAP_PDUSESSION_SETUP_RESP
    loop pduSessions_to_addmod
        Note over CUCP: Fill NGAP message (setup)
        CUCP->>CUCP: add_pduSession (&UE->pduSessions)
    end
    loop UE->pduSessions_failed
        Note over CUCP: Fill NGAP message (failed)
    end
    Note over CUCP: free UE->pduSessions_to_addmod
    Note over CUCP: free UE->pduSessions_failed
    CUCP->>Network: NGAP_PDUSESSION_SETUP_RESP
    CUCP->>DU: ue_context_modification_request
```

# PDU Session Modification

```mermaid
sequenceDiagram
    participant AMF
    participant CUCP
    participant DU
    participant UE

    AMF->>CUCP: PDUSessionResourceModifyRequest
    Note over CUCP: ngap_gNB_handle_pdusession_modify_request
    CUCP->>CUCP: decodePDUSessionResourceModify
    CUCP->>CUCP: NGAP_PDUSESSION_MODIFY_REQ
    CUCP->>CUCP: rrc_gNB_process_NGAP_PDUSESSION_MODIFY_REQ
    opt UE not found or AMF_UE_ID mismatch
        CUCP->>AMF: NGAP_PDUSESSION_MODIFY_RESP (Failed PDU Session)
        Note over CUCP: stop further processing
    end

    loop nb_pdusessions_tomodify
        CUCP->>CUCP: find_pduSession(UE->pduSessions)
        alt Session not found
            CUCP->>CUCP: fill pdusessions_failed
        else Session found
            CUCP->>CUCP: cp_pdusession_resource_item_to_pdusession
            Note over CUCP: copy to pdusession_t item (including QoS add/modify/release)
            CUCP->>CUCP: update PDU Session in RRC (&UE->pduSessions)
            Note over CUCP: Update QoS mapping in DRB
            CUCP->>CUCP: add_pduSession(&UE->pduSessions_to_addmod)
        end
    end

    alt seq_arr_size(UE->pduSessions_to_addmod)
        CUCP->>CUCP: rrc_gNB_modify_dedicatedRRCReconfiguration
        Note over CUCP: RRC_PDUSESSION_MODIFY
        CUCP->>CUCP: Build DRB list and NAS list
        CUCP->>DU: rrc_deliver_dl_rrc_message
        DU->>UE: RRCReconfiguration (DCCH)
        UE->>DU: RRCReconfigurationComplete
        DU->>CUCP: F1AP_UL_RRC_MESSAGE
        Note over CUCP: rrc_gNB_decode_dcch
        Note over CUCP: handle_rrcReconfigurationComplete
        CUCP->>CUCP: rrc_gNB_send_NGAP_PDUSESSION_MODIFY_RESP
        loop pduSessions_to_addmod
            Note over CUCP: Fill NGAP message (modified)
        end
        loop UE->pduSessions_failed
            Note over CUCP: Fill NGAP message (failed tp modify)
        end
        CUCP->>AMF: NGAP_PDUSESSION_MODIFY_RESP
        Note over CUCP: free UE->pduSessions_to_addmod
        Note over CUCP: free UE->pduSessions_failed
        CUCP->>DU: ue_context_modification_request
    else msg->nb_of_pdusessions_failed > 0
        Note over CUCP: PDU Session failed to modify
        CUCP->>AMF: NGAP_PDUSESSION_MODIFY_RESP
    end
```

# PDU Session Release

```mermaid
sequenceDiagram
    participant AMF
    participant CUCP
    participant CUUP
    participant DU
    participant UE

    AMF->>CUCP: PDUSessionResourceReleaseCommand
    Note over CUCP: ngap_gNB_handle_pdusession_release_command
    CUCP->>CUCP: NGAP_PDUSESSION_RELEASE_COMMAND
    CUCP->>CUCP: rrc_gNB_process_NGAP_PDUSESSION_RELEASE_COMMAND

    loop nb_pdusessions_torelease
        CUCP->>CUCP: find_pduSession(UE->pduSessions_failed)
        alt PDU Session is in the failed list
            CUCP->>CUCP: add_pduSession_to_release()
        else
            CUCP->>CUCP: find_pduSession(UE->pduSessions)
            alt PDU Session not found in setup list
                CUCP->>CUCP: add_failed_pduSession()
            else PDU is setup, add to addmod list
                CUCP->>CUCP: add_pduSession_to_addmod()
                CUCP->>CUCP: rm_pduSession()
                Note over CUCP: Add to addmod and remove from setup list
                Note over CUCP: Cleanup established DRBs
                CUCP->>CUCP: fill E1AP message (numPDUSessionsRem++)
            end
        end
    end

    alt req.numPDUSessionsRem > 0
        opt E1AP available
            CUCP->>CUUP: e1_bearer_context_mod
            CUUP->>CUUP: newGtpuDeleteOneTunnel
            CUUP->>CUUP: nr_pdcp_release_drbs
            CUUP->>CUUP: nr_sdap_delete_entity
        end
        Note over CUCP: RRC_PDUSESSION_RELEASE
        CUCP->>CUCP: rrc_gNB_generate_dedicatedRRCReconfiguration_release
        Note over CUCP: Build DRB_ReleaseList and NAS message (if present)
        CUCP->>DU: rrc_deliver_dl_rrc_message
        CUCP->>UE: RRCReconfiguration (DRB release + optional NAS)
        UE->>DU: RRCReconfigurationComplete
        DU->>CUCP: F1AP_UL_RRC_MESSAGE
        Note over CUCP: handle_rrcReconfigurationComplete()

        loop UE->pduSessions_to_addmod
            Note over CUCP: move to pduSessions_to_release
            Note over CUCP: cleanup pduSessions_to_addmod
        end

        CUCP->>CUCP: rrc_gNB_send_NGAP_PDUSESSION_RELEASE_RESPONSE
        Note over CUCP: loop through pduSessions_to_release and pduSessions_failed

    else
        # CUCP->>CUCP: release_pduSessions
        # Note over CUCP: GTP tunnel deletion
        CUCP->>CUCP: rrc_gNB_send_NGAP_PDUSESSION_RELEASE_RESPONSE
        Note over CUCP: loop through pduSessions_to_release and pduSessions_failed
    end

```
