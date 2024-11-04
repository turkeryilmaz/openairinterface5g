# OAI File Modification Documentation:

NOTICE

This software was produced for the U. S. Government
under Basic Contract No. W56KGU-18-D-0004, and is
subject to the Rights in Noncommercial Computer Software
and Noncommercial Computer Software Documentation
Clause 252.227-7014 (FEB 2014)

(C) 2024 The MITRE Corporation.

#### Author Information:

- Developer: Surajit Dey (MITRE)
- Developer: Danny Nsouli (MITRE)

#### Modified Code Files Documentation:

All figures can be referenced in the [Backhaul Adaptation Protocol (BAP) Specification TS 38.340](https://www.etsi.org/deliver/etsi_ts/138300_138399/138340/16.04.00_60/ts_138340v160400p.pdf). The following are files modified in November 2023 in an attempt to integrate a BAP layer:

`openair2/LAYER2/nr_pdcp/nr_pdcp_oai_api.h`

- Added BAP data structure (`struct bap_data`), BAP Data PDU (Figure 6.2.2-1) assembly function declarations: `int bap_pdu_func(dest, path)`

`openair2/LAYER2/nr_pdcp/nr_pdcp_oai_api.c`

- Integrated BAP Data PDU (Figure 6.2.2-1) assembly function (`int bap_pdu_func(dest, path)`). `dest` is a 10-bit field that identifies the destination BAP address, which could be the address of an IAB-node or IAB-donor-DU. `path` carries a 10-bit path identity that is used for routing the BAP Data PDU.
- deliver_pdu_drb() was modified to insert the BAP data PDU header in the transmit direction of user plane in both UE and gNB.
- deliver_pdu_srb_rlc() was modified to insert the BAP data PDU header in the transmit direction of signaling plane in both UE and gNB.
- do_pdcp_data_ind() was modified to remove the BAP data PDU header in the receive direction of user plane in both UE and gNB.
- deliver_sdu_srb() was modified to remove the BAP data PDU header in the receive direction of signaling plane in both UE and gNB.

`openair2/RRC/NR/rrc_gNB.c`
- DURecvCb() was modified to insert the BAP data PDU header in the transmit direction of signaling plane in both UE and gNB.
- rrc_DU_process_ue_context_setup_request() was modified to insert the BAP data PDU header in the transmit direction of signaling plane in both UE and gNB.

#### New Non-Integrated Code Files:

The following files were tested with various edge cases in a local environment seperate from the OAI codebase. They contain assembly functions for each of the BAP Data and Control PDU formats:

`bap_pdu.c`
- Assembles Figure 6.2.2-1: BAP Data PDU format

`bap_cpdu_routing.c`
- Assembles Figure 6.2.3.1-2: BAP Control PDU format for flow control feedback per BAP routing ID

`bap_cpdu.c:`
- Assembles Figure 6.2.3.1-1: BAP Control PDU format for flow control feedback per BH RLC channel

`bap_control_detec.c`
- Assembles Figure 6.2.3.4-1: BAP Control PDU format for BH RLF detection indication

`bap_control_indic.c`
- Assembles Figure 6.2.3.3-1: BAP Control PDU format for BH RLF indication 

`bap_control_polling.c`
- Assembles Figure 6.2.3.2-1: BAP Control PDU format for flow control feedback polling

`bap_control_recov.c`
- Assembles Figure 6.2.3.5-1: BAP Control PDU format for BH RLF recovery indication

Each of these c files resides in: `oai5g/bap_and_oai_modified_files_NOV2023/bap_files_not_integrated/


Backhaul Adaptation Protocol (BAP) layer code has been added using 2023.w16 tag of the OAI develop branch.

BAP data PDU is added in the control plane and user plane of gNB and UE.
Packet structure of BAP data PDU is taken from 3GPP spec TS 38.340 section 6.2.2.
Verified BAP data PDU header insertion and removal in gNB & UE using RF simulation feature of Open Air Interface. UE, gNB and core all were simulated in one Ubuntu host machine running 20.04 Ubuntu.
Verified BAP layer functionality with CU-DU split architecture of gNB (using RF simulator).

How to run BAP code:

Start 5G core simulator first:
Use the following link to download core container images.
https://gitlab.eurecom.fr/oai/openairinterface5g/-/tree/develop/ci-scripts/yaml_files/5g_rfsimulator

Start only the core NFs. Remove or comment out the gNB and UE sections in the docker-compose.yaml file.
The docker-compose file is located in:
~/ci-scripts/yaml_files/5g_rfsimulator directory. 

$ cd ci-scripts/yaml_files/5g_rfsimulator
$ sudo docker-compose up -d

CU-DU split:

Start CU:
sudo RFSIMULATOR=server ./nr-softmodem --rfsim --sa -O ../../../ci-scripts/conf_files/gNB_SA_CU.conf
 
DU:
sudo RFSIMULATOR=server BAP=yes ./nr-softmodem --rfsim --sa -O ../../../ci-scripts/conf_files/gNB_SA_DU.conf

UE:
sudo RFSIMULATOR=IAB-DU_IP_addr BAP=yes ./nr-uesoftmodem -r 106 --numerology 1 --band 78 -C 3619200000 --rfsim --sa --nokrnmod -O ~/conf-files/ue1.conf

Note: The "BAP" environment variable was used for testing the BAP functionality in UE/IAB-MT and DU.

Example configuration files for CU, DU and UE can be found in this directory.

Use ping or iperf in UE and rfsim5g-oai-ext-dn for end to end data test.
Monitor traffic with wireshark on rfsim5g-public interface with filters set to "udp or sctp".

Pcap files from lab test are available in this directory. A png file shows how BAP header (3 bytes with known pattern 0x8eafcd) are added in the uplink direction.

Note 1: MITRE did end to end test with moderate amount of traffic. Stress test or performance test was not done under full bandwidth scenarios.

Note 2: RF simulator airlink data is sent between UE and gNB as TCP payload. In order to dissect the headers below IP layer - PDCP/RLC/MAC, the following procedure was used.

To run gNB with T tracer, use this option with nr-softmodem:
--T_stdout 2

cd to common/utils/T/tracer and run
./macpdu2wireshark -d ../T_messages.txt -live

Capture required data in pcap file:
sudo tcpdump -n -iany "udp port 9999 or udp port 2152 or sctp or port 38462 or port 38472 or port 38412 or port 2153" -w /tmp/rec.pcap

Open the pcap file on wireshark.

https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/common/utils/T/DOC/T/wireshark.md
https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/common/utils/T/DOC/T/basic.md
