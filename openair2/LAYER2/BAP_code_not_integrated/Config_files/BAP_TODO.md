# BAP TODO List:

NOTICE

This software was produced for the U. S. Government
under Basic Contract No. W56KGU-18-D-0004, and is
subject to the Rights in Noncommercial Computer Software
and Noncommercial Computer Software Documentation
Clause 252.227-7014 (FEB 2014)

(C) 2024 The MITRE Corporation.

### Author Information:

- Developer: Surajit Dey (MITRE)
- Developer: Danny Nsouli (MITRE)

### File Descriptions:

All figures can be referenced in the [Backhaul Adaptation Protocol (BAP) 3GPP Specification TS 38.340.](https://www.etsi.org/deliver/etsi_ts/138300_138399/138340/16.04.00_60/ts_138340v160400p.pdf)

#### 1. Integrate BAP Control PDU for Flow Control Feedback Polling (Figure 6.2.3.2-1)
- It was determined that a new function (i.e. `int bap_fc_polling_pdu(rntiP, tb_sizeP)`) could be written to assemble the aforementioned polling control PDU in `openair2/LAYER2/nr_rlc/nr_rlc_oai_api.c`.
-   A global variable, `char BAP_FC_POLLING_PDU`, could be created to initialize the variable, `buffer_pP`, a pointer to a buffer that is used to store the MAC PDU (Protocol Data Unit) data. This new variable is necessary as a different value is required for the assembly of this specific PDU, hence why `MAC_PDU_SIZE` could be replaced.

- Within `radio/rfsimulator/simulator.c`, a thread may be required to both control and avoid rapid erroneous PDU polling activity. Therefore, it was determined that a BAP polling thread could be created within `bool flushInput(*t, timeout, nsamps_for_initial)`. A new worker function (i.e. `void *worker2`) should be added to assist the thread's creation as a required parameter (i.e. `threadCreate(&th, worker2, NULL, "BAP polling", -1, OAI_PRIORITY_RT_LOW);`).

#### 2. Integrate BAP Control PDU format for flow control feedback per BAP routing ID (Figure 6.2.3.1-2)

- `bap_cpdu_routing.c` contains a function that can assemble the BAP header for the aforementioned PDU.
- It has not been determined how this functionality should be integrated whether it be in an existing OAI file or a new one.

#### 3. Integrate BAP Control PDU format for flow control feedback per BH RLC channel (Figure 6.2.3.1-1)

- `bap_cpdu.c` contains a function that can assemble the BAP header for the aforementioned PDU.
- It has not been determined how this functionality should be integrated whether it be in an existing OAI file or a new one.

#### 4. Integrate BAP Control PDU format for BH RLF indication (Figure 6.2.3.3-1)

- `bap_control_indic.c` contains a function that can assemble the BAP header for the aforementioned PDU.
- It has not been determined how this functionality should be integrated whether it be in an existing OAI file or a new one.

#### 5. Integrate BAP Control PDU format for BH RLF detection indication (Figure 6.2.3.4-1)

- `bap_control_detec.c` contains a function that can assemble the BAP header for the aforementioned PDU.
- It has not been determined how this functionality should be integrated whether it be in an existing OAI file or a new one.

#### 6. Integrate BAP Control PDU format for BH RLF recovery indication (Figure 6.2.3.5-1)

- `bap_control_recov.c` contains a function that can assemble the BAP header for the aforementioned PDU.
- It has not been determined how this functionality should be integrated whether it be in an existing OAI file or a new one.

Each of the mentioned c files from points 2-6 resides in: `oai5g/bap_and_oai_modified_files_NOV2023/bap_files_not_integrated/

#### 7. Integrate all control PDUs in OAI code and test for the following functions:
- Flow control per backhaul RLC channel
- Flow control per BAP routing ID
- Flow control feedback polling
- Backhaul Radio Link Failure indication
- Backhaul Radio Link Failure detection indication
- Backhaul Radio Link Failure recovery indication

#### 8. Make BAP ID and path (routing ID) dynamic.
- In the current version BAP ID and path are hard coded for the prototype test. Those need to be configured dynamically using RRC messages.
- Process RRC signaling from donor-CU-CP to get the BAP ID and path or routing ID. Use those parameters in the BAP data PDU header for communicating to the next hop IAB node.

#### 9. Test for performance with max bandwidth to compare the impact of BAP layer addition in the user plane traffic throughput and latency.
