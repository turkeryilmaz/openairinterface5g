# 1. TTCN 4.1-287 CONTENT

## 1.1 OAI (RAN)

| **openairinterface**   | sqn/3GPP_TTCN_System_Simulator | 27fb26af121da865010b56c3308d0aff6cb419f4 |    Mon Mar 18 11:50:30 2024  |  bug #130447 Support of Paging on USRP setups | Jerome Peraldi | **PASS** |

# 2. SOFTWARE UPDATES FROM PREVIOUS RELEASE

## 2.1 Generic updates

### 2.1.1 Code alignment with OAI

- Sequans TTCN SS code is aligned with OAI from W50 2023

### 2.1.2 Support of Ubuntu 22 and Debian 12

---
**Note1:** Debian 12 was already supported on previous release

**Note2: Ubuntu 20 and Debian 10 are DEPRECATED. Supported OS are Ubuntu 22 and Debian 12 from such release**

---

|     Bug Id      |     Summary |
| :----------- | :-------------- |
| 131749  |   [L3 SIMU] Support of Ubuntu 22 for Simulation |

## 2.2 Main updates: 5G

### 2.2.1 New features tested with OAI UE

#### 2.2.1.1 Support of Radio link failure

Radio link failure is supported and tested with 3GPP 38523 in front of OAI UE

|     Bug Id      |     Summary |
| :----------- | :-------------- |
| 133320  |  [L3 SIMU] [5G] support of radio link failure: 8.1.5.6.3 Radio link failure / RRC connection re-establishment success |

The test has been added in SQNS CI.

Statistic of test execution during W11 2024 provided below. (Ubuntu 22)
- 18% fail (run 11x) for P2 NR5GC_8.1.5.6.3_VT_ENABLED_NOSECU_SSMODE1 0/3,18/8 mn

### 2.2.3 TTCN Hardware setup (B210 and N310) in front of Fibocom 5G device

#### 2.2.3.1 Support of 5G SA preamble

It's possible to execute 3GPP 38523 8.1.5.1.1 preamble (5G SA attachment) on Sequans TTCN hardware setup based on B210 or N310 setups, up to rrc release message.

#### 2.2.3.2 Support of paging (N310) in front of Fibocom 5G device

Paging is now supported with Fibocom 5G device and tested on N310 board. (need to test on B210)
It has been tested with 3GPP 38523 8.1.5.1.1: step 2 of the test is reached but Failed (issue on mismatch between capabilities sent by UE and expected by TTCN).

#### 2.2.3.3 USRP list of fixed bugs

|     Bug Id      |     Summary |
| :----------- | :-------------- |
|129514	| [L3 SIMU] USRP setup SS_MODEM=2 (SRB only) up to rrc release |
|129513 | Run TTCN + gNB + OAI UE + Proxy without VT on HW setup up to AUTH procedure HW MODE=2 (SRB + SYS PORTS) N310 board |
|130540	| TTCN adapter failure on RRC Setup Complete due to replayedNASMsgContainerValue inside NAS part |
|130447|[L3 SIMU] Support of paging: Hardware setup (USRP N310)|
| 132027 | [TTCN SETUP] validate USRP setup  with B210 device |
| 132054 |	[USRP] B210 install doc |
| 132462 | [USRP] ran_setup/FC-Scripts/build_oai unable to build all libs for USRP mode |
| 126580| [L3 SIMU] preparing platform for SI HW testing |

### 2.2.4 Other

#### 2.2.4.1 support of Babeltrace2 C++ plugins for decoding NAS/RRC

All NAS/RRC/PDCP/RLC/MAC/FAPI messages are decoded with Babeltrace2 C++ plugins in replacement of python plugins.

logs decoding is 10 time faster with C++ plugins

|     Bug Id      |     Summary |
| :----------- | :-------------- |
|127920 | [L3 SIMU] 5G Taurus lttng logger to be added in TTCN environment |

# 3. IMPORTANT UPDATES 4G

## 3.1 New 3GPP 36523 supported test cases.

The following test cases are now supported:

- 3GPP 36523 8.2.1.5 RRC connection reconfiguration / Radio bearer establishment for transition 
from RRC_Idle to RRC CONNECTED / Success / Latency check 
- 3GPP 36523 8.2.1.6 RRC connection reconfiguration / Radio bearer establishment for transition  from RRC_Idle to RRC CONNECTED / Success / Latency check / SecurityModeCommand and RRCConnectionReconfiguration transmitted in the same TTI 
- 3GPP 36523 8.5.1.3 Radio link failure / T311 expiry
- 3GPP 36523 9.2.1.1.17 Attach / Rejected / No suitable cells in tracking area 
- 3GPP 36523 8.1.5.6.1 Radio link failure / RRC connection re-establishment success
- 3GPP 36523 8.5.1.6: Radio link failure / T311 expiry / Dedicated RLF timer
- 3GPP 36523 7.1.2.2 Correct selection of RACH parameters / Random access preamble and PRACH resource
- TDD support: 3GPP 36523 8.2.1.5 PASSED in TDD mode

|     Bug Id         |     Summary |
| :--------------- | :-------------- |
| 122145 | [L3 SIMU] support of 3GPP 36523 8.2.1.5 RRC connection reconfiguration / Radio bearer establishment for transition from RRC_Idle to RRC CONNECTED / Success / Latency check |
| 125065 |[L3 SIMU] RLF 3GPP 36523 8.5.1.3 Radio link failure / T311 expiry support |
| 127702 | [L3 SIMU] 3GPP 9.2.1.1.17  Attach / Rejected / No suitable cells in tracking area |
| 130121 | [L3 SIMU] [5G] support of radio link failure: 8.1.5.6.1 Radio link failure / RRC connection re-establishment success |
|130147 |[L3 SIMU] [4G] [TDD] support of TDD mode on TTCN test platform |
|130948 | [L3 SIMU] we need to support 3GPP 36523 8.5.1.6: Radio link failure / T311 expiry / Dedicated RLF timer |
|131535 | [L3 SIMU] [4G] [TDD] TC8_2_1_5 support of TDD mode on TTCN test platform after attach procedure |
|131600 | [L3 SIMU] support of 7.1.2.2 Correct selection of RACH parameters / Random access preamble and PRACH resource

# 4. Test results 5G

## 4.1 Ubuntu 22

### 5G RAN

- https://gitlab-shared.sequans.com/sequans/system/ttcn/adapter/-/pipelines/42922


|           TC name                   |   Main parameters   |   status   |  duration  |
| :---------------------------------- | :------------------ | :--------: | :--------: |
| **SYS_P1_38523-3_TTCN3_NR5GC_7.1.2.2.1_VT_ENABLED_BASIC** | Platform1 (stub)  | **PASS** | 0 |
| **SYS_P1_38523-3_TTCN3_NR5GC_8.1.5.1.1_BASIC** | Platform1 (stub)  | **PASS** | 0 |
| **SYS_P1_38523-3_TTCN3_NR5GC_8.1.5.1.1_VT_ENABLED_BASIC** | Platform1 (stub)  | **PASS** | 0 |
| **SYS_P2_38523-3_TTCN3_NR5GC_6.1.2.3_OAI_UE_SECU_AES_SSMODE1** | Platform2 (OAI UE)  | **PASS** | 5 |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.2.2.1_OAI_UE_BASIC_SSMODE1** | Platform2 (OAI UE)  | **PASS** | 0 |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.3.2.1_OAI_UE_SECU_SNOW3G_SSMODE1** | Platform2 (OAI UE)  | **PASS** | 5 |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.3.2.2_OAI_UE_SECU_AES_SSMODE1** | Platform2 (OAI UE)  | **PASS** | 5 |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.3.3.1_OAI_UE_SECU_SNOW3G_SSMODE1** | Platform2 (OAI UE)  | **PASS** | 5 |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.3.3.1_OAI_UE_SECU_SNOW3G_SSMODE1_AUTH_MILENAGE** | Platform2 (OAI UE)  | **PASS** | 5 |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.3.3.2_OAI_UE_SECU_AES_SSMODE1** | Platform2 (OAI UE)  | **PASS** | 5 |
| **SYS_P2_38523-3_TTCN3_NR5GC_8.1.5.1.1_OAI_UE_SECU_SNOW3G_SSMODE1** | Platform2 (OAI UE)  | **FAIL** | 0 |
| **SYS_P2_38523-3_TTCN3_NR5GC_8.1.5.1.1_OAI_UE_SECU_EAP_AKAP_AES_SSMODE1** | Platform2 (OAI UE)  | **PASS** | 5 |
| **SYS_P2_38523-3_TTCN3_NR5GC_8.1.5.6.3_VT_ENABLED_NOSECU_SSMODE1** | Platform2 (OAI UE)  | **PASS** | 5 |
| **SYS_P2_38523-3_TTCN3_NR5GC_8.1.5.1.1_OAI_UE_SECU_AES_SSMODE1** | Platform2 (OAI UE)  | **PASS** | 5 |
| **SYS_P2_38523-3_TTCN3_NR5GC_8.1.5.1.1_OAI_UE_NOSECU_SSMODE1** | Platform2 (OAI UE)  | **PASS** | 5 |
| **SYS_P2_38523-3_TTCN3_NR5GC_1.1.1_OAI_UE_IPV4_PING_VT_SSMODE1** | Platform2 (OAI UE)  | **PASS** | 5 |
| **SYS_P2_38523-3_TTCN3_NR5GC_1.1.1_OAI_UE_IP_LOOP_VT_SSMODE1** | Platform2 (OAI UE)  | **PASS** | 5 |
| **SYS_P2_38523-3_TTCN3_NR5GC_1.1.2_OAI_UE_IPV4_PING_NOPAGING_VT_SSMODE1** | Platform2 (OAI UE)  | **PASS** | 5 |
| **SYS_P2_38523-3_TTCN3_NR5GC_1.1.2_OAI_UE_IP_LOOP_NOPAGING_VT_SSMODE1** | Platform2 (OAI UE)  | **PASS** | 5 |
| **SYS_P2_38523-3_TTCN3_NR5GC_1.1.2_OAI_UE_IPV4_PING_NOPAGING_VT_SECU_AES_SSMODE1** | Platform2 (OAI UE)  | **PASS** | 5 |
| **SYS_P3_38523-3_TTCN3_NR5GC_1.1.2_IPV4_PING_NOPAGING_VT_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_38523-3_TTCN3_NR5GC_1.1.2_IPV4_LOOP_NOPAGING_VT_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_38523-3_TTCN3_NR5GC_7.1.2.2.4.SEQUANS_NO_PAGING_VT_SECU_AES_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 7 |
| **SYS_P3_38523-3_TTCN3_NR5GC_7.1.2.2.1.SEQUANS_NO_PAGING_VT_SECU_AES_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_38523-3_TTCN3_NR5GC_7.1.2.3.1.SEQUANS_NO_PAGING_VT_SECU_AES_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_38523-3_TTCN3_NR5GC_7.1.2.2.2.SEQUANS_NO_PAGING_VT_SECU_AES_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_38523-3_TTCN3_NR5GC_7.1.2.3.2.SEQUANS_NO_PAGING_VT_SECU_AES_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_38523-3_TTCN3_NR5GC_7.1.2.2.3.SEQUANS_NO_PAGING_VT_SECU_AES_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.SEQUANS_SECU_SNOW3G_VT_ENABLED_SSMODE1_0** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.SEQUANS_SECU_AES_VT_ENABLED_SSMODE1_0** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.SEQUANS_SECU_SNOW3G_VT_ENABLED_SSMODE1_1** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.SEQUANS_SECU_AES_VT_ENABLED_SSMODE1_1** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.SEQUANS_SECU_SNOW3G_VT_ENABLED_SSMODE1_2** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.SEQUANS_SECU_AES_VT_ENABLED_SSMODE1_2** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.SEQUANS_SECU_SNOW3G_SSMODE1_0** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.SEQUANS_SECU_AES_SSMODE1_0** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.SEQUANS_SECU_EAP_AKAP_AES_VT_ENABLED_SSMODE1_0** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.SEQUANS_SECU_SNOW3G_SSMODE1_1** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.SEQUANS_SECU_AES_SSMODE1_1** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.SEQUANS_SECU_EAP_AKAP_AES_VT_ENABLED_SSMODE1_1** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.SEQUANS_SECU_SNOW3G_SSMODE1_2** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.SEQUANS_SECU_AES_SSMODE1_2** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.SEQUANS_SECU_EAP_AKAP_AES_VT_ENABLED_SSMODE1_2** | Platform3 (SQNS UE)  | **FAIL (inconclusive)** | 5 |

#### Number of test cases: 43


#### Number of passed test cases: 41


#### Number of failed test cases: 2


#### Pass rate rating for 5G: **95%**

## 4.2 Debian 12


## 4.3 PASSED rate on executed tests (Debian 12)

W10 statistics:

### 5G RAN

|           TC name                   |  run  | Pass rate | Duration in min (min/av/max) |
| :---------------------------------- | :---: | :-------: | :--------------------------: |
| P2 NR5GC_1.1.2_OAI_UE_IP_LOOP_NOPAGING_VT_SSMODE1 3/3,00/3 mn | 21x | 100 % |  |
| P2 NR5GC_1.1.2_OAI_UE_IPV4_PING_NOPAGING_VT_SSMODE1 3/3,00/3 mn | 21x | 100 % |  |
| P2 NR5GC_7.1.3.2.2_OAI_UE_SECU_AES_SSMODE1 3/3,00/3 mn | 21x | 100 % |  |
| P3 NR5GC_8.1.5.1.1.SEQUANS_SECU_AES_VT_ENABLED_SSMODE1 0/0,05/1 mn | 21x | 100 % |  |
| P3 NR5GC_8.1.5.1.1.SEQUANS_SECU_EAP_AKAP_AES_VT_ENABLED_SSMODE1 0/0,05/1 mn | 21x | 100 % |  |
| P3 NR5GC_8.1.5.1.1.SEQUANS_SECU_SNOW3G_SSMODE1 0/0,00/0 mn | 21x | 100 % |  |
| P3 NR5GC_8.1.5.1.1.SEQUANS_SECU_SNOW3G_VT_ENABLED_SSMODE1 0/0,00/0 mn | 21x | 100 % |  |
| P3 NR5GC_1.1.2_IPV4_LOOP_NOPAGING_VT_SSMODE1 0/0,14/1 mn | 7x | 100 % |  |
| P3 NR5GC_1.1.2_IPV4_PING_NOPAGING_VT_SSMODE1 0/0,00/0 mn | 7x | 100 % |  |
| P3 NR5GC_7.1.2.2.1.SEQUANS_NO_PAGING_VT_SECU_AES_SSMODE1 0/0,00/0 mn | 7x | 100 % |  |
| P3 NR5GC_7.1.2.2.2.SEQUANS_NO_PAGING_VT_SECU_AES_SSMODE1 0/0,00/0 mn | 7x | 100 % |  |
| P3 NR5GC_7.1.2.2.4.SEQUANS_NO_PAGING_VT_SECU_AES_SSMODE1 1/6,43/11 mn | 7x | 100 % |  |
| P3 NR5GC_8.1.5.1.1.SEQUANS_SECU_AES_SSMODE1_1_noroot 0/0,00/0 mn | 7x | 100 % |  |
| P3 NR5GC_8.1.5.1.1.SEQUANS_SECU_AES_SSMODE1_2_noroot 0/0,00/0 mn | 7x | 100 % |  |
| P3 NR5GC_8.1.5.1.1.SEQUANS_SECU_AES_VT_ENABLED_SSMODE1_0_noroot 0/0,14/1 mn | 7x | 100 % |  |
| P3 NR5GC_8.1.5.1.1.SEQUANS_SECU_AES_VT_ENABLED_SSMODE1_1_noroot 0/0,29/1 mn | 7x | 100 % |  |
| P3 NR5GC_8.1.5.1.1.SEQUANS_SECU_AES_VT_ENABLED_SSMODE1_2_noroot 0/0,14/1 mn | 7x | 100 % |  |
| P3 NR5GC_8.1.5.1.1.SEQUANS_SECU_EAP_AKAP_AES_VT_ENABLED_SSMODE1_0_noroot 0/0,14/1 mn | 7x | 100 % |  |
| P3 NR5GC_8.1.5.1.1.SEQUANS_SECU_EAP_AKAP_AES_VT_ENABLED_SSMODE1_2_noroot 0/0,00/0 mn | 7x | 100 % |  |
| P3 NR5GC_8.1.5.1.1.SEQUANS_SECU_SNOW3G_SSMODE1_0_noroot 0/0,00/0 mn | 7x | 100 % |  |
| P3 NR5GC_8.1.5.1.1.SEQUANS_SECU_SNOW3G_SSMODE1_1_noroot 0/0,00/0 mn | 7x | 100 % |  |
| P3 NR5GC_8.1.5.1.1.SEQUANS_SECU_SNOW3G_SSMODE1_2_noroot 0/0,00/0 mn | 7x | 100 % |  |
| P3 NR5GC_8.1.5.1.1.SEQUANS_SECU_SNOW3G_VT_ENABLED_SSMODE1_0_noroot 0/0,00/0 mn | 7x | 100 % |  |
| P3 NR5GC_8.1.5.1.1.SEQUANS_SECU_SNOW3G_VT_ENABLED_SSMODE1_1_noroot 0/0,00/0 mn | 7x | 100 % |  |
| P3 NR5GC_8.1.5.1.1.SEQUANS_SECU_SNOW3G_VT_ENABLED_SSMODE1_2_noroot 0/0,00/0 mn | 7x | 100 % |  |
| P2 NR5GC_1.1.2_OAI_UE_IPV4_PING_NOPAGING_VT_SECU_AES_SSMODE1 0/2,86/3 mn | 21x | 96 % |  |
| P2 NR5GC_7.1.3.2.1_OAI_UE_SECU_SNOW3G_SSMODE1 3/3,00/3 mn | 21x | 96 % |  |
| P2 NR5GC_7.1.3.3.1_OAI_UE_SECU_SNOW3G_SSMODE1 3/3,00/3 mn | 21x | 96 % |  |
| P2 NR5GC_8.1.5.1.1_OAI_UE_SECU_SNOW3G_SSMODE1 3/3,00/3 mn | 21x | 96 % |  |
| P3 NR5GC_8.1.5.1.1.SEQUANS_SECU_AES_SSMODE1 0/0,24/5 mn | 21x | 96 % |  |
| P2 NR5GC_7.1.3.3.1_OAI_UE_SECU_SNOW3G_SSMODE1_AUTH_MILENAGE 3/3,00/3 mn | 21x | 91 % |  |
| P2 NR5GC_7.1.3.3.2_OAI_UE_SECU_AES_SSMODE1 0/2,86/3 mn | 21x | 91 % |  |
| P2 NR5GC_1.1.1_OAI_UE_IP_LOOP_VT_SSMODE1 1/3,95/12 mn | 21x | 86 % |  |
| P2 NR5GC_1.1.1_OAI_UE_IPV4_PING_VT_SSMODE1 3/3,62/10 mn | 21x | 86 % |  |
| P2 NR5GC_8.1.5.6.3_VT_ENABLED_NOSECU_SSMODE1 2/3,67/8 mn | 21x | 86 % |  |
| P3 NR5GC_7.1.2.2.3.SEQUANS_NO_PAGING_VT_SECU_AES_SSMODE1 1/2,29/4 mn | 7x | 86 % |  |
| P3 NR5GC_8.1.5.1.1.SEQUANS_SECU_AES_SSMODE1_0_noroot 0/0,71/5 mn | 7x | 86 % |  |
| P3 NR5GC_8.1.5.1.1.SEQUANS_SECU_EAP_AKAP_AES_VT_ENABLED_SSMODE1_1_noroot 0/2,57/16 mn | 7x | 86 % |  |
| P2 NR5GC_7.1.2.2.1_OAI_UE_BASIC_SSMODE1 0/0,71/3 mn | 21x | 77 % |  |
| P2 NR5GC_8.1.5.1.1_OAI_UE_NOSECU_SSMODE1 3/3,00/3 mn | 21x | 77 % |  |
| P2 NR5GC_6.1.2.3_OAI_UE_SECU_AES_SSMODE1 0/2,57/3 mn | 7x | 72 % |  |
| P2 NR5GC_8.1.5.1.1_OAI_UE_SECU_AES_SSMODE1 0/2,86/3 mn | 21x | 67 % |  |

---
**Note**: there are no sufficient stats for evaluating RLC AM test cases: 3GPP 38523 7.1.2.3.1 and 7.1.2.3.2

---
 
# 5. Test results 4G

## 5.1 Ubuntu 22

### 4G RAN

|           TC name                   |   Main parameters   |   status   |  duration  |
| :---------------------------------- | :------------------ | :--------: | :--------: |
| **SYS_P1_36523-3_TTCN3_LTE_6.1.2.2_BASIC** | Platform1 (stub)  | **PASS** | 1 |
| **SYS_P1_36523-3_TTCN3_LTE_6.1.2.2_VT_ENABLED_BASIC** | Platform1 (stub)  | **PASS** | 1 |
| **SYS_P1_36523-3_TTCN3_LTE_6.1.2.2a_BASIC** | Platform1 (stub)  | **PASS** | 1 |
| **SYS_P1_36523-3_TTCN3_LTE_8.1.1.2_BASIC** | Platform1 (stub)  | **PASS** | 2 |
| **SYS_P1_36523-3_TTCN3_LTE_8.1.2.2_BASIC** | Platform1 (stub)  | **PASS** | 1 |
| **SYS_P1_36523-3_TTCN3_LTE_9.1.5.1_BASIC** | Platform1 (stub)  | **PASS** | 0 |
| **SYS_P1_36523-3_TTCN3_LTE_9.3.2.1_BASIC** | Platform1 (stub)  | **PASS** | 0 |
| **SYS_P1_36523-3_TTCN3_LTE_9.4.3_BASIC** | Platform1 (stub)  | **PASS** | 0 |
| **SYS_P1_36523-3_TTCN3_LTE_6.1.2.2_VT_ENABLED_BASIC_GENUML** | Platform1 (stub) -- tools | **PASS** | 1 |
| **SYS_P1_36523-3_TTCN3_LTE_6.1.2.2_BASIC_RESOLVE_LOG_BACKTRACE** | Platform1 (stub) -- tools | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_10.2.1_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_36523-3_TTCN3_LTE_10.3.1_VT_ENABLED_NOSECU_SSMODE1** | Platform3 (SQNS UE)  | **FAIL** | 16 |
| **SYS_P3_36523-3_TTCN3_LTE_10.4.1_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 3 |
| **SYS_P3_36523-3_TTCN3_LTE_10.5.1_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_10.5.3_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 2 |
| **SYS_P3_36523-3_TTCN3_LTE_10.6.1_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_10.6.1_VT_ENABLED_NOSECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_10.7.1_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_10.7.2_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 2 |
| **SYS_P3_36523-3_TTCN3_LTE_10.7.3_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_10.7.4_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 7 |
| **SYS_P3_36523-3_TTCN3_LTE_10.7.5_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_10.8.1_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_10.8.2_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_10.8.3_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_10.8.4_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_10.8.5_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_10.8.6_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_10.8.7_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 8 |
| **SYS_P3_36523-3_TTCN3_LTE_13.3.1.1_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_13.3.1.1_VT_ENABLED_SECU_SNOW3G_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_13.3.1.1_VT_ENABLED_SECU_AES_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 2 |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.1.3_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 10 |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.1.4_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 2 |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.1.3b_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 38 |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.3a_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.7_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **FAIL (inconclusive)** | 47 |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.10_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.17_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.4_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.7a_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 38 |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.11_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.18_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 2 |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.2_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 7 |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.2a_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 7 |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.5_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 2 |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.8_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.12_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 12 |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.3_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.6_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 7 |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.8a_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 20 |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.13_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **FAIL (inconclusive)** | 0 |
| **SYS_P3_36523-3_TTCN3_LTE_7.1.1.1_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 6 |
| **SYS_P3_36523-3_TTCN3_LTE_7.1.1.2_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 8 |
| **SYS_P3_36523-3_TTCN3_LTE_7.1.2.1_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 2 |
| **SYS_P3_36523-3_TTCN3_LTE_7.1.2.2_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_36523-3_TTCN3_LTE_8.1.1.2_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 5 |
| **SYS_P3_36523-3_TTCN3_LTE_8.1.1.3_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 2 |
| **SYS_P3_36523-3_TTCN3_LTE_8.1.1.4_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_8.1.1.6_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 2 |
| **SYS_P3_36523-3_TTCN3_LTE_8.1.2.2_VT_ENABLED_NOSECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 2 |
| **SYS_P3_36523-3_TTCN3_LTE_8.1.2.2_VT_ENABLED_SECU_SNOW3G_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 2 |
| **SYS_P3_36523-3_TTCN3_LTE_8.1.2.2_VT_ENABLED_SECU_AES_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 2 |
| **SYS_P3_36523-3_TTCN3_LTE_8.1.2.3_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_8.1.2.5_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **FAIL** | 16 |
| **SYS_P3_36523-3_TTCN3_LTE_8.1.2.6_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 6 |
| **SYS_P3_36523-3_TTCN3_LTE_8.1.2.14_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 4 |
| **SYS_P3_36523-3_TTCN3_LTE_8.1.3.4_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_8.1.3.5_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.1.1_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.1.3_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.1.5_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.1.5_TDD_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **FAIL (inconclusive)** | 0 |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.1.6_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **FAIL (inconclusive)** | 2 |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.1.7_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.1.8_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.2.1_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.2.2_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.3.1_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.4.3_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.4.4_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.4.9_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 0 |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.4.13_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 2 |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.4_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **FAIL (inconclusive)** | 16 |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.8_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 7 |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.14_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 4 |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.1_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 6 |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.5_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 2 |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.9_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 3 |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.12_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.2_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 7 |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.6_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 4 |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.9a_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 2 |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.23_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 12 |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.3_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 4 |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.7_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 4 |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.10_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 3 |
| **SYS_P3_36523-3_TTCN3_LTE_8.5.1.1_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 3 |
| **SYS_P3_36523-3_TTCN3_LTE_8.5.1.2_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 3 |
| **SYS_P3_36523-3_TTCN3_LTE_8.5.1.3_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 2 |
| **SYS_P3_36523-3_TTCN3_LTE_8.5.1.4_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 2 |
| **SYS_P3_36523-3_TTCN3_LTE_8.5.1.5_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 2 |
| **SYS_P3_36523-3_TTCN3_LTE_8.5.1.6_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 7 |
| **SYS_P3_36523-3_TTCN3_LTE_8.5.4.1_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 3 |
| **SYS_P3_36523-3_TTCN3_LTE_9.1.2.3_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 5 |
| **SYS_P3_36523-3_TTCN3_LTE_9.1.2.4_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_9.1.2.5_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_9.1.2.7_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_9.1.3.1_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 2 |
| **SYS_P3_36523-3_TTCN3_LTE_9.1.3.2_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_9.1.3.3_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 2 |
| **SYS_P3_36523-3_TTCN3_LTE_9.1.5.1_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 4 |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.2_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.13_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 12 |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.17_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 9 |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.25_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.2_VT_ENABLED_SECU_SSMODE1_AUTH_MILENAGE** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.13a_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 37 |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.19_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 3 |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.26_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 3 |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.9_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 13 |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.14_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 9 |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.22_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 71 |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.1_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.10_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 12 |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.15_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 13 |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.2.1.1_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **FAIL (inconclusive)** | 10 |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.2.1.2_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.2.1.6_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 16 |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.2.1.7_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 3 |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.2.1.8_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 4 |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.2.2.1_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.2.2.2_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.2.2.14_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 11 |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.3.1.8_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 2 |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.3.1.26_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 42 |
| **SYS_P3_36523-3_TTCN3_LTE_9.3.1.1_VT_ENABLED_NOSECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_9.3.1.1_VT_ENABLED_SECU_SNOW3G_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 2 |
| **SYS_P3_36523-3_TTCN3_LTE_9.3.1.1_VT_ENABLED_SECU_AES_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_9.3.1.7_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_9.3.1.7a_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_9.3.1.16_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_9.3.1.17_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_9.3.2.1_VT_ENABLED_NOSECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_9.3.2.1_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |
| **SYS_P3_36523-3_TTCN3_LTE_9.4.1_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 3 |
| **SYS_P3_36523-3_TTCN3_LTE_9.4.2_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 3 |
| **SYS_P3_36523-3_TTCN3_LTE_9.4.3_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 3 |
| **SYS_P3_36523-3_TTCN3_LTE_9.4.4_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 2 |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.2.1.2.SEQUANS_VT_ENABLED_SECU_SSMODE1** | Platform3 (SQNS UE)  | **PASS** | 1 |

#### Number of test cases: 150


#### Number of passed test cases: 142


#### Number of failed test cases: 8


#### Pass rate rating for 4G: **94%**