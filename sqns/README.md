# **TTCN 3.0-240 CONTENT**

## **General**

- based on OAI W20 source code
- Improvements on synchronisation between TTCN SS (openair3/SS) and RRC RAN module
- add support of RRC restablishment procedure partially supported: better management of UE RNTI if SS configured temp rnti exists
- fixing NAS security issue (memory leaks)
- cleaning up of FAPI interfaces: full alignment between UE, gNB, proxy. Removing all unuseful compilation flags 
- improvements of Sequans virtual time mechanism
- support of unix sockets (improving jobs processing on Sequans CI servers)
- solving segfauls issue@RAN (eNB and gNB)

## **5G USRP setup**

- Sequans TTCN environment running on USRP (N310) setup (pre tested with 3GPP 38523 8.1.5.1.1 UE capability transfer / Success)

## **4G**

- Support of 120 test cases from 3GPP 36523
- 4G environment can be launched without being root. (mandatory for launching CI on any server having strong security rules.)

## **5G**

- 16 3GPP 38523 supported test cases ie tested with OAI UE or Sequans simulator
- support of loopback mode B with Sequans and OAI UEs simulators
- Support of SDAP loopback with OAI UE simulator
- support of AS security features for Control and Data plane (AES and SNOW3G)
- support of EAP-AKAP NAS algorithm for Authentication
- adding support of SSB fields in FAPI interface
- solving slot processing delay issue related to virtual time
- solving SRB id issues

## **CI**

- Sequans CI PASS rate (150 test cases, 4G and 5G): 90%

