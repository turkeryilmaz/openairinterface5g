# 1. TTCN_4.1-298 CONTENT

## 1.2 OPEN AIR INTERFACE GITLAB HASHKEY

|     Component      |     branch      | commit |     date      |   commit title   |  commit author  | status |
| :----------------- | :-------------- | :----- | :-----------: | :--------------- | :-------------- | :----: |
| **openairinterface**   | 3GPP_TTCN_System_Simulator | ba7ba4f0033ca909dbf8d8daf521749f0e0a8bb5 |    Mon Aug 26 17:00:12 2024 +0200  |  Merge tag 'TTCN_4.1-298' into 3GPP_TTCN_System_Simulator | Jerome Peraldi | **PASS** |

# 2. SOFTWARE UPDATES FROM PREVIOUS RELEASE (TTCN 4.1-284)

## 2.1 Generic updates

### 2.1.1 Code alignment with OAI

- TTCN code is aligned with OAI from **W13 2024.**

## 2.2 5G TTCN simulation test platform updates

### 2.2.1 full supports of 3GPP 38523 RLC AM test cases.

|Bug ID|Summary                                                                                                               |Component   |Status        |Resolution|
|------|----------------------------------------------------------------------------------------------------------------------|------------|--------------|----------|
|133529|[L3 SIMU] need to check that guard time on test is running in virtual time on simulation                             |TTCN adapter|INTEGRATION   | ---      |
|133483|[L3 SIMU] [CI] needs to integrate AM RLC test cases 7.1.2.3.3 and 7.1.2.3.4. into our CI                              |TTCN adapter|RESOLVED      |FIXED     |
|133385|[L3 SIMU] needs to deliver 7.1.2.3.3 and 7.1.2.3.4 test cases from 3GPP 38523                                         |TTCN adapter|RESOLVED      |FIXED     |
|133321|[L3 SIMU] needs to support first RLC AM test cases from 3GPP 38523: 7.1.2.3.3 and 7.1.2.3.4                           |TTCN adapter|RESOLVED      |DUPLICATE |

### 2.2.2 partial support of 3GPP 38523 7.4.1 SDAP Data Transfer and PDU Header Handling UL/DL

The test has been validated up to step 2. We cannot go deeper as above such step, reflective QOS is needed and we did not find any UE supporting properly such feature.

- **Fixed bugs**

|Bug ID|Summary                                                                                                               |Component   |Status        |Resolution|
|------|----------------------------------------------------------------------------------------------------------------------|------------|--------------|----------|
|132815|[L3 SIMU] [5G] support of 3GPP 38523 7.1.4.1 SDAP Data Transfer and PDU Header Handling UL/DL                         |TTCN adapter|INTEG_ASSIGNED| ---      |


## 2.3 5G TTCN hardware test platform updates

### 2.3.1 TTCN hardware support with National Instrument N310 and B210

SI TTCN setup is now working also over hardware in front of Fibocom device.
Radio frontends are National Instrument USRP B210 or N310 boards.
15 test cases from 3GPP 38523 are now supported.
For B210 board, 40MHZ bandwidth is supported. For N310 board, 40MHZ is supported and 100MHZ deployment is in progress.

CI from server is under deployment: for this release 2 test cases can be executed. (not stable)
Local CI is working properly on N310 setup, with a PASS rate oscillating between 80% and 95%.

- **List of supported test cases in front of Fibocom device is provided below (40 Mhz configuration):**

|           TC name                   |   Main parameters   |
| :---------------------------------- | :------------------ |
| **SYS_P4_38523-3_TTCN3_NR5GC_8.1.1.2.4_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  |
| **SYS_P4_38523-3_TTCN3_NR5GC_8.1.5.6.3_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.1.1_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.1.4_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.1.5_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.5.1.6_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.2.2_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.2.3_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.2.4_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.2.6_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.2.7_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.6.1.1_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.6.1.1_SSMODE2_2_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.6.2.1_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.6.2.1_SSMODE2_2_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.6.2.2_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  |
| **SYS_P4_38523-3_TTCN3_NR5GC_10.1.5.1_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  |

**Fixed bugs**

|Bug ID|Summary                                                                                                               |Component   |Status        |Resolution|
|------|----------------------------------------------------------------------------------------------------------------------|------------|--------------|----------|
|133439|[USRP SETUPS] enabling all messages coming on SYSport in SSMODE2 ie. NO FILTER on TTCN messages                       |TTCN Setup  |RESOLVED      |FIXED     |
|133418|[USRP SETUP] 3GPP 38523 9.1.2.2 on USRP setup                                                                         |TTCN Setup  |RESOLVED      |FIXED     |
|133376|[USRP SETUP] need to add support of IDENTITY_REQ/RESP for NR                                                          |TTCN Setup  |RESOLVED      |FIXED     |
|133374|[USRP SETUP] support of 3GPP 38523 9.1.5.1.6 Initial registration / Rejected / Illegal UE                             |TTCN Setup  |RESOLVED      |FIXED     |
|133373|[USRP SETUP] Support of 3GPP 38523 9.1.1.5 5G AKA based primary authentication and key agreement / Reject             |TTCN Setup  |RESOLVED      |FIXED     |
|133372|[USRP SETUP] support of 3GPP 9.1.1.4 5G AKA based primary authentication and key agreement / 5G-AKA related procedures|TTCN Setup  |VERIFIED      |FIXED     |

### 2.3.2 5G L1 TTCN test case: support of 10MHZ (TTCN proprietary test cases based on 3GPP 38523-3 TTCN model)

10 MHZ TTCN test case has been produced to L1 team to prepare their own integration:

- Fixed bug

|Bug ID|Summary                                                                                                               |Component   |Status        |Resolution|
|------|----------------------------------------------------------------------------------------------------------------------|------------|--------------|----------|
|133445| [USRP SETUPS] support of 10MHZ configuration both for simulation and Hardware                                                                           |TTCN setup|New      |---     |

- limitations
  - The configuration was not tested in front of any DUT as Fibocom device was not able to attach with 10MHZ bandwidth configuration in front of R&S and Anritsu testers
  - It's possible to test it in front of Amarisoft, so logs will be compared with such setup, to ensure that what is provided by TTCN configuration is OK


## 2.4 lttng logger

lttng logger fully supports post processing log decoding and live sessions.
Important improvements on 3GPP/FAPI decoding done.


## 2.5 Other

bugs below have been fixed, and improve test setup stability:

|Bug ID|Summary                                                                                                               |Component   |Status        |Resolution|
|------|----------------------------------------------------------------------------------------------------------------------|------------|--------------|----------|
|133756|[L3 SIMU] [L3 SIMU] better management of TTCN ACK CNF messages                                                        |TTCN adapter|RESOLVED   | FIXED      |
|133529|[L3 SIMU] need to check that guard timer on test is running in virtual time on simulation                            |TTCN adapter|INTEGRATION   | ---      |
|133456|[L3 SIMU] milenage AUTH config shall be set by default to false in CI tc-devXXX.json                                  |TTCN adapter|RESOLVED      |FIXED     |
|133317|[L3 SIMU] SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1_SECU_EAP_AKAP_AES_VT_ENABLED_SSMODE1 not stable                |TTCN adapter|RESOLVED      |FIXED     |
|133172|[L3 SIMU] upgrade openssl lib to the same one than Debian Bookworm image                                              |TTCN adapter|RESOLVED      |FIXED     |
|132638|[L3 SIMU] [OAI] [4G] support of timing alignment on eNB                                                               |OAI         |RESOLVED      |FIXED     |
|129806|[L3 SIMU] sometimes paging is not processed by OAI UE                                                                 |TTCN adapter|RESOLVED      |FIXED     |

## 2.6 UPDATES for 4G

## 2.6.1 New 3GPP 36523 supported test cases.

The following test cases are now supported:

- Timing alignment

|     Bug Id         |     Summary |
| :--------------- | :-------------- |
| 132638 | [L3 SIMU] [OAI] [4G] support of timing alignment on eNB |

# 3. Test results 5G

## 3.1 Ubuntu 22

### 3.1.1 5G 3GPP 38523 TTCN simulation

### 5G RAN

|           TC name                   |   UE type   |   status   |  duration  | Test type |
| :---------------------------------- | :------------------ | :--------: | :--------: | :--------: |
| **SYS_P1_38523-3_TTCN3_NR5GC_7.1.2.2.1_VT_BASIC_1** | Platform1 (stub)  | **PASS** | 0 | 3GPP |
| **SYS_P1_38523-3_TTCN3_NR5GC_8.1.5.1.1_BASIC_1** | Platform1 (stub)  | **PASS** | 0 | 3GPP |
| **SYS_P1_38523-3_TTCN3_NR5GC_8.1.5.1.1_VT_BASIC_1** | Platform1 (stub)  | **PASS** | 0 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_6.1.2.3_OAI_UE_SECU_AES_VT_SSMODE1_1** | Platform2 (OAI UE)  | **FAIL** | 0 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_6.1.2.3_OAI_UE_SECU_AES_VT_SSMODE1_2** | Platform2 (OAI UE)  | **PASS_RETRY** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.2.2.1_OAI_UE_BASIC_VT_SSMODE1_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.3.2.1_OAI_UE_SECU_SNOW3G_VT_SSMODE1_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.3.2.2_OAI_UE_SECU_AES_VT_SSMODE1_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.3.3.1_OAI_UE_SECU_SNOW3G_VT_SSMODE1_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.3.3.1_OAI_UE_SECU_SNOW3G_VT_SSMODE1_AUTH_MILENAGE_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.3.3.2_OAI_UE_SECU_AES_VT_SSMODE1_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_8.1.5.1.1_OAI_UE_SECU_SNOW3G_VT_SSMODE1_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_8.1.5.1.1_OAI_UE_SECU_EAP_AKAP_AES_VT_SSMODE1_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_8.1.5.1.1_OAI_UE_SECU_AES_VT_SSMODE1_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_8.1.5.1.1_OAI_UE_NOSECU_VT_SSMODE1_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_1.1.1_OAI_UE_IPV4_PING_VT_SSMODE1_1** | Platform2 (OAI UE)  | **PASS** | 5 | CUSTOM |
| **SYS_P2_38523-3_TTCN3_NR5GC_1.1.1_OAI_UE_IP_LOOP_VT_SSMODE1_1** | Platform2 (OAI UE)  | **PASS** | 5 | CUSTOM |
| **SYS_P2_38523-3_TTCN3_NR5GC_1.1.2_OAI_UE_IPV4_PING_NOPAGING_VT_SSMODE1_1** | Platform2 (OAI UE)  | **PASS** | 5 | CUSTOM |
| **SYS_P2_38523-3_TTCN3_NR5GC_1.1.2_OAI_UE_IP_LOOP_NOPAGING_VT_SSMODE1_1** | Platform2 (OAI UE)  | **PASS** | 5 | CUSTOM |
| **SYS_P2_38523-3_TTCN3_NR5GC_1.1.2_OAI_UE_IPV4_PING_NOPAGING_SECU_AES_VT_SSMODE1_1** | Platform2 (OAI UE)  | **PASS** | 5 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_1.1.2_IPV4_PING_NOPAGING_VT_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_1.1.2_IPV4_LOOP_NOPAGING_VT_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_7.1.2.2.4.CUSTOM_NO_PAGING_SECU_AES_VT_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 13 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_7.1.2.3.4.CUSTOM_NO_PAGING_SECU_AES_VT_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 18 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_7.1.2.2.1.CUSTOM_NO_PAGING_SECU_AES_VT_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_7.1.2.3.1.CUSTOM_NO_PAGING_SECU_AES_VT_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_7.1.2.2.2.CUSTOM_NO_PAGING_SECU_AES_VT_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_7.1.2.3.2.CUSTOM_NO_PAGING_SECU_AES_VT_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_7.1.4.1.CUSTOM_NO_PAGING_VT_SECU_AES_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_7.1.2.2.3.CUSTOM_NO_PAGING_SECU_AES_VT_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_7.1.2.3.3.CUSTOM_NO_PAGING_SECU_AES_VT_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_SNOW3G_VT_SSMODE1_0_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_EAP_AKAP_AES_VT_SSMODE1_0_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_SNOW3G_VT_SSMODE1_1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_EAP_AKAP_AES_VT_SSMODE1_1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_SNOW3G_VT_SSMODE1_2_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_EAP_AKAP_AES_VT_SSMODE1_2_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_AES_VT_SSMODE1_0_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_AES_VT_SSMODE1_1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_AES_VT_SSMODE1_2_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | CUSTOM |

#### Number of test cases: 40


#### Number of passed test cases: 39


#### Number of failed test cases: 1


#### Pass rate rating for 5G: **97%**


### 6.1.2 5G 3GPP 38523 TTCN hardware setup with National Instrument N310 board

### 5G RAN

|           TC name                   |   Main parameters   |   status   |  duration  |
| :---------------------------------- | :------------------ | :--------: | :--------: |
| **SYS_P4_38523-3_TTCN3_NR5GC_8.1.1.2.4_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  | **PASS** | 0 |
| **SYS_P4_38523-3_TTCN3_NR5GC_8.1.5.6.3_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  | **PASS** | 0 |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.1.1_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  | **PASS** | 0 |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.1.4_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  | **PASS** | 0 |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.1.5_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  | **PASS** | 0 |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.5.1.6_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  | **PASS** | 1 |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.2.2_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  | **PASS** | 0 |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.2.3_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  | **PASS** | 0 |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.2.4_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  | **PASS** | 0 |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.2.6_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  | **PASS** | 0 |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.2.7_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  | **PASS** | 0 |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.6.1.1_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  | **FAIL** | 0 |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.6.1.1_SSMODE2_2_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  | **PASS_RETRY** | 0 |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.6.2.1_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  | **FAIL** | 0 |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.6.2.1_SSMODE2_2_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  | **PASS_RETRY** | 0 |
| **SYS_P4_38523-3_TTCN3_NR5GC_9.1.6.2.2_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  | **PASS** | 2 |
| **SYS_P4_38523-3_TTCN3_NR5GC_10.1.5.1_SSMODE2_1_noroot** | Platform4 (CUSTOM 3GPP 38523 HARDWARE platform)  | **PASS** | 0 |

#### Number of test cases: 17


#### Number of passed test cases: 15


#### Number of failed test cases: 2


#### Pass rate rating for 5G: **88%**


## 3.2 Debian 12

### 3.2.1 5G 3GPP 38523 TTCN simulation

### 5G RAN

|           TC name                   |   UE type   |   status   |  duration  | Test type |
| :---------------------------------- | :------------------ | :--------: | :--------: | :--------: |
| **SYS_P2_38523-3_TTCN3_NR5GC_6.1.2.3_OAI_UE_SECU_AES_VT_SSMODE1_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.2.2.1_OAI_UE_BASIC_VT_SSMODE1_0_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.3.2.1_OAI_UE_SECU_SNOW3G_VT_SSMODE1_0_1** | Platform2 (OAI UE)  | **PASS** | 6 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.3.2.2_OAI_UE_SECU_AES_VT_SSMODE1_0_1** | Platform2 (OAI UE)  | **PASS** | 6 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.3.3.1_OAI_UE_SECU_SNOW3G_VT_SSMODE1_0_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.3.3.1_OAI_UE_SECU_SNOW3G_VT_SSMODE1_AUTH_MILENAGE_0_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.3.3.2_OAI_UE_SECU_AES_VT_SSMODE1_0_1** | Platform2 (OAI UE)  | **PASS** | 1 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.2.2.1_OAI_UE_BASIC_VT_SSMODE1_1_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.3.2.1_OAI_UE_SECU_SNOW3G_VT_SSMODE1_1_1** | Platform2 (OAI UE)  | **FAIL** | 0 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.3.2.1_OAI_UE_SECU_SNOW3G_VT_SSMODE1_1_2** | Platform2 (OAI UE)  | **PASS_RETRY** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.3.2.2_OAI_UE_SECU_AES_VT_SSMODE1_1_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.3.3.1_OAI_UE_SECU_SNOW3G_VT_SSMODE1_1_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.3.3.1_OAI_UE_SECU_SNOW3G_VT_SSMODE1_AUTH_MILENAGE_1_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.3.3.2_OAI_UE_SECU_AES_VT_SSMODE1_1_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.2.2.1_OAI_UE_BASIC_VT_SSMODE1_2_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.3.2.1_OAI_UE_SECU_SNOW3G_VT_SSMODE1_2_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.3.2.2_OAI_UE_SECU_AES_VT_SSMODE1_2_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.3.3.1_OAI_UE_SECU_SNOW3G_VT_SSMODE1_2_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.3.3.1_OAI_UE_SECU_SNOW3G_VT_SSMODE1_AUTH_MILENAGE_2_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_7.1.3.3.2_OAI_UE_SECU_AES_VT_SSMODE1_2_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_8.1.5.1.1_OAI_UE_SECU_EAP_AKAP_AES_VT_SSMODE1_0_1** | Platform2 (OAI UE)  | **PASS** | 6 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_8.1.5.1.1_OAI_UE_SECU_EAP_AKAP_AES_VT_SSMODE1_1_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_8.1.5.1.1_OAI_UE_SECU_EAP_AKAP_AES_VT_SSMODE1_2_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_8.1.5.1.1_OAI_UE_SECU_AES_VT_SSMODE1_0_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_8.1.5.1.1_OAI_UE_SECU_AES_VT_SSMODE1_1_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_8.1.5.1.1_OAI_UE_SECU_AES_VT_SSMODE1_2_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_8.1.5.1.1_OAI_UE_SECU_SNOW3G_VT_SSMODE1_0_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_8.1.5.1.1_OAI_UE_SECU_SNOW3G_VT_SSMODE1_1_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_8.1.5.1.1_OAI_UE_SECU_SNOW3G_VT_SSMODE1_2_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_8.1.5.1.1_OAI_UE_NOSECU_VT_SSMODE1_0_1** | Platform2 (OAI UE)  | **PASS** | 6 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_8.1.5.1.1_OAI_UE_NOSECU_VT_SSMODE1_1_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_8.1.5.1.1_OAI_UE_NOSECU_VT_SSMODE1_2_1** | Platform2 (OAI UE)  | **PASS** | 5 | 3GPP |
| **SYS_P2_38523-3_TTCN3_NR5GC_1.1.1_OAI_UE_IPV4_PING_VT_SSMODE1_0_1** | Platform2 (OAI UE)  | **PASS** | 5 | CUSTOM |
| **SYS_P2_38523-3_TTCN3_NR5GC_1.1.2_OAI_UE_IPV4_PING_NOPAGING_SECU_AES_VT_SSMODE1_0_1** | Platform2 (OAI UE)  | **PASS** | 5 | CUSTOM |
| **SYS_P2_38523-3_TTCN3_NR5GC_1.1.1_OAI_UE_IPV4_PING_VT_SSMODE1_1_1** | Platform2 (OAI UE)  | **FAIL** | 1 | CUSTOM |
| **SYS_P2_38523-3_TTCN3_NR5GC_1.1.1_OAI_UE_IPV4_PING_VT_SSMODE1_1_2** | Platform2 (OAI UE)  | **PASS_RETRY** | 5 | CUSTOM |
| **SYS_P2_38523-3_TTCN3_NR5GC_1.1.2_OAI_UE_IPV4_PING_NOPAGING_SECU_AES_VT_SSMODE1_1_1** | Platform2 (OAI UE)  | **PASS** | 5 | CUSTOM |
| **SYS_P2_38523-3_TTCN3_NR5GC_1.1.1_OAI_UE_IPV4_PING_VT_SSMODE1_2_1** | Platform2 (OAI UE)  | **PASS** | 5 | CUSTOM |
| **SYS_P2_38523-3_TTCN3_NR5GC_1.1.2_OAI_UE_IPV4_PING_NOPAGING_SECU_AES_VT_SSMODE1_2_1** | Platform2 (OAI UE)  | **PASS** | 5 | CUSTOM |
| **SYS_P2_38523-3_TTCN3_NR5GC_1.1.1_OAI_UE_IP_LOOP_VT_SSMODE1_0_1** | Platform2 (OAI UE)  | **PASS** | 5 | CUSTOM |
| **SYS_P2_38523-3_TTCN3_NR5GC_1.1.1_OAI_UE_IP_LOOP_VT_SSMODE1_1_1** | Platform2 (OAI UE)  | **PASS** | 5 | CUSTOM |
| **SYS_P2_38523-3_TTCN3_NR5GC_1.1.1_OAI_UE_IP_LOOP_VT_SSMODE1_2_1** | Platform2 (OAI UE)  | **PASS** | 5 | CUSTOM |
| **SYS_P2_38523-3_TTCN3_NR5GC_1.1.2_OAI_UE_IPV4_PING_NOPAGING_VT_SSMODE1_0_1** | Platform2 (OAI UE)  | **PASS** | 5 | CUSTOM |
| **SYS_P2_38523-3_TTCN3_NR5GC_1.1.2_OAI_UE_IPV4_PING_NOPAGING_VT_SSMODE1_1_1** | Platform2 (OAI UE)  | **PASS** | 5 | CUSTOM |
| **SYS_P2_38523-3_TTCN3_NR5GC_1.1.2_OAI_UE_IPV4_PING_NOPAGING_VT_SSMODE1_2_1** | Platform2 (OAI UE)  | **PASS** | 5 | CUSTOM |
| **SYS_P2_38523-3_TTCN3_NR5GC_1.1.2_OAI_UE_IP_LOOP_NOPAGING_VT_SSMODE1_0_1** | Platform2 (OAI UE)  | **PASS** | 5 | CUSTOM |
| **SYS_P2_38523-3_TTCN3_NR5GC_1.1.2_OAI_UE_IP_LOOP_NOPAGING_VT_SSMODE1_1_1** | Platform2 (OAI UE)  | **PASS** | 5 | CUSTOM |
| **SYS_P2_38523-3_TTCN3_NR5GC_1.1.2_OAI_UE_IP_LOOP_NOPAGING_VT_SSMODE1_2_1** | Platform2 (OAI UE)  | **PASS** | 5 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_1.1.2_IPV4_LOOP_NOPAGING_VT_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_1.1.2_IPV4_PING_NOPAGING_VT_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_7.1.2.2.4.CUSTOM_NO_PAGING_SECU_AES_VT_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 7 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_7.1.2.3.4.CUSTOM_NO_PAGING_SECU_AES_VT_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 12 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_7.1.2.2.1.CUSTOM_NO_PAGING_SECU_AES_VT_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_7.1.2.3.1.CUSTOM_NO_PAGING_SECU_AES_VT_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_7.1.2.2.2.CUSTOM_NO_PAGING_SECU_AES_VT_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_7.1.2.3.2.CUSTOM_NO_PAGING_SECU_AES_VT_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_7.1.4.1.CUSTOM_NO_PAGING_VT_SECU_AES_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 5 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_7.1.2.2.3.CUSTOM_NO_PAGING_SECU_AES_VT_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_7.1.2.3.3.CUSTOM_NO_PAGING_SECU_AES_VT_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 3 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_SNOW3G_VT_SSMODE1_0_1_noroot** | Platform3 (CUSTOM UE)  | **PASS** | 1 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_EAP_AKAP_AES_VT_SSMODE1_0_1_noroot** | Platform3 (CUSTOM UE)  | **PASS** | 3 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_SNOW3G_VT_SSMODE1_1_1_noroot** | Platform3 (CUSTOM UE)  | **PASS** | 1 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_EAP_AKAP_AES_VT_SSMODE1_1_1_noroot** | Platform3 (CUSTOM UE)  | **PASS** | 1 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_SNOW3G_VT_SSMODE1_2_1_noroot** | Platform3 (CUSTOM UE)  | **PASS** | 1 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_EAP_AKAP_AES_VT_SSMODE1_2_1_noroot** | Platform3 (CUSTOM UE)  | **PASS** | 1 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_SNOW3G_VT_SSMODE1_0_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_EAP_AKAP_AES_VT_SSMODE1_0_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_SNOW3G_VT_SSMODE1_1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_EAP_AKAP_AES_VT_SSMODE1_1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_SNOW3G_VT_SSMODE1_2_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_EAP_AKAP_AES_VT_SSMODE1_2_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_AES_VT_SSMODE1_0_1_noroot** | Platform3 (CUSTOM UE)  | **PASS** | 0 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_AES_VT_SSMODE1_1_1_noroot** | Platform3 (CUSTOM UE)  | **PASS** | 0 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_AES_VT_SSMODE1_2_1_noroot** | Platform3 (CUSTOM UE)  | **PASS** | 0 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_AES_VT_SSMODE1_0_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_AES_VT_SSMODE1_1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | CUSTOM |
| **SYS_P3_38523-3_TTCN3_NR5GC_8.1.5.1.1.CUSTOM_SECU_AES_VT_SSMODE1_2_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | CUSTOM |

#### Number of test cases: 77


#### Number of passed test cases: 75


#### Number of failed test cases: 2


#### Pass rate rating for 5G: **97%**

 
# 4. Test results 4G

## 4.1 Ubuntu 22


### 4G RAN

|           TC name                   |   UE type   |   status   |  duration  | Test type |
| :---------------------------------- | :------------------ | :--------: | :--------: | :--------: |
| **SYS_P1_36523-3_TTCN3_LTE_6.1.2.2_BASIC_1** | Platform1 (stub)  | **PASS** | 1 | 3GPP |
| **SYS_P1_36523-3_TTCN3_LTE_6.1.2.2_VT_ENABLED_BASIC_1** | Platform1 (stub)  | **PASS** | 1 | 3GPP |
| **SYS_P1_36523-3_TTCN3_LTE_6.1.2.2a_BASIC_1** | Platform1 (stub)  | **PASS** | 1 | 3GPP |
| **SYS_P1_36523-3_TTCN3_LTE_8.1.1.2_BASIC_1** | Platform1 (stub)  | **PASS** | 2 | 3GPP |
| **SYS_P1_36523-3_TTCN3_LTE_8.1.2.2_BASIC_1** | Platform1 (stub)  | **PASS** | 1 | 3GPP |
| **SYS_P1_36523-3_TTCN3_LTE_9.1.5.1_BASIC_1** | Platform1 (stub)  | **PASS** | 0 | 3GPP |
| **SYS_P1_36523-3_TTCN3_LTE_9.3.2.1_BASIC_1** | Platform1 (stub)  | **PASS** | 0 | 3GPP |
| **SYS_P1_36523-3_TTCN3_LTE_9.4.3_BASIC_1** | Platform1 (stub)  | **PASS** | 0 | 3GPP |
| **SYS_P1_36523-3_TTCN3_LTE_6.1.2.2_VT_ENABLED_BASIC_GENUML_1** | Platform1 (stub) -- tools | **PASS** | 1 | 3GPP |
| **SYS_P1_36523-3_TTCN3_LTE_6.1.2.2_BASIC_RESOLVE_LOG_BACKTRACE_1** | Platform1 (stub) -- tools | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_10.2.1_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_10.3.1_VT_ENABLED_NOSECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_10.4.1_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 3 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_10.5.1_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_10.5.3_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_10.6.1_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_10.6.1_VT_ENABLED_NOSECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_10.7.1_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_10.7.2_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_10.7.3_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 3 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_10.7.4_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 8 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_10.7.5_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_10.8.1_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_10.8.2_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_10.8.3_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_10.8.4_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_10.8.5_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_10.8.6_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_10.8.7_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 8 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_13.3.1.1_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_13.3.1.1_VT_ENABLED_SECU_SNOW3G_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_13.3.1.1_VT_ENABLED_SECU_AES_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.1.3_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 9 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.1.4_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.1.3b_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 35 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.3a_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.7_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 46 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.10_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.17_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.4_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.7a_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 35 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.11_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.18_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.2_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 6 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.2a_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 6 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.5_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.8_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.12_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 11 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.3_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.6_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 6 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.8a_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 19 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_6.1.2.13_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_7.1.1.1_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 6 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_7.1.1.2_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 8 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_7.1.2.1_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_7.1.2.2_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.1.1.2_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 4 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.1.1.3_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.1.1.4_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.1.1.6_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.1.2.2_VT_ENABLED_NOSECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.1.2.2_VT_ENABLED_SECU_SNOW3G_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.1.2.2_VT_ENABLED_SECU_AES_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.1.2.3_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.1.2.5_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 7 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.1.2.6_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 6 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.1.2.14_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 4 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.1.3.4_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.1.3.5_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.1.1_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.1.3_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.1.5_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **FAIL (inconclusive)** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.1.5_VT_ENABLED_SECU_SSMODE1_2** | Platform3 (CUSTOM UE)  | **PASS_RETRY** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.1.5_TDD_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **FAIL (inconclusive)** | 0 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.1.5_TDD_VT_ENABLED_SECU_SSMODE1_2** | Platform3 (CUSTOM UE)  | **PASS_RETRY** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.1.6_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.1.7_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.1.8_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.2.1_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.2.2_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.3.1_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.4.3_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.4.4_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.4.9_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.4.13_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **FAIL** | 16 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.4.13_VT_ENABLED_SECU_SSMODE1_2** | Platform3 (CUSTOM UE)  | **FAIL_RETRY** | 16 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.2.4.13_VT_ENABLED_SECU_SSMODE1_3** | Platform3 (CUSTOM UE)  | **FAIL_RETRY_SYSTEMATIC** | 16 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.4_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **FAIL (inconclusive)** | 15 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.4_VT_ENABLED_SECU_SSMODE1_2** | Platform3 (CUSTOM UE)  | **PASS_RETRY** | 23 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.8_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 3 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.14_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 4 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.1_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 6 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.5_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.9_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.12_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.2_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 6 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.6_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 4 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.9a_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.23_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 11 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.3_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 4 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.7_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 4 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.3.1.10_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.5.1.1_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.5.1.2_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 3 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.5.1.3_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.5.1.4_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.5.1.5_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.5.1.6_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 7 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_8.5.4.1_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.1.2.3_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 4 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.1.2.4_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.1.2.5_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.1.2.7_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.1.3.1_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.1.3.2_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 0 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.1.3.3_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.1.5.1_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.2_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.13_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 12 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.17_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 9 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.25_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.2_VT_ENABLED_SECU_SSMODE1_AUTH_MILENAGE_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.13a_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 39 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.19_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 3 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.26_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 3 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.9_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 12 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.14_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 9 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.22_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 71 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.1_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.10_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 13 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.15_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **FAIL** | 16 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.1.1.15_VT_ENABLED_SECU_SSMODE1_2** | Platform3 (CUSTOM UE)  | **PASS_RETRY** | 13 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.2.1.1_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.2.1.2_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.2.1.6_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 15 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.2.1.7_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 3 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.2.1.8_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 4 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.2.2.1_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.2.2.2_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.2.2.14_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 11 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.3.1.8_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.3.1.26_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 41 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.3.1.1_VT_ENABLED_NOSECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.3.1.1_VT_ENABLED_SECU_SNOW3G_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.3.1.1_VT_ENABLED_SECU_AES_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.3.1.7_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.3.1.7a_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.3.1.16_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **FAIL** | 16 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.3.1.16_VT_ENABLED_SECU_SSMODE1_2** | Platform3 (CUSTOM UE)  | **PASS_RETRY** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.3.1.17_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.3.2.1_VT_ENABLED_NOSECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.3.2.1_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.4.1_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.4.2_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.4.3_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.4.4_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 2 | 3GPP |
| **SYS_P3_36523-3_TTCN3_LTE_9.2.2.1.2.CUSTOM_VT_ENABLED_SECU_SSMODE1_1** | Platform3 (CUSTOM UE)  | **PASS** | 1 | CUSTOM |

#### Number of test cases: 157


#### Number of passed test cases: 149


#### Number of failed test cases: 8


#### Pass rate rating for 4G: **94%**
