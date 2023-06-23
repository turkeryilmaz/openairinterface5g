# 5G NSA in L2-Sim Setup

## Table of Contents

[[_TOC_]]

## 1 Overview

This document tells about the reference guide for the setup of the environment for NSA 5G configuration using the openair interface code base. This shall be the reference setup before we start the testing on the SequansRD . Since the NSA was not tested by OAI in the CI, there are few things to be taken into account to bring up the setup. This document is being created to identify the issues and resolve them and provide the steps to easily re-create the NSA setup. Once this is achieved the next step is to have the SequansRD based NSA setup.

The NSA mode functional is shown in the following image:
![](Images/Multi-UE-Proxy.png)

## 2 Message flow for NSA setup
Following slides show the message flow between the LTE-UE and NR-UE of OAI and the communication between eNB/gNB for NSA setup.
Also the important logs to be verified for the NSA setup are documented in the document

![](Images/NSA_Mode.png)

![](Images/NSA_Sockets_Routing.png)

![](Images/NSA_Mode_FlowDiagram.png)

4: LOG_A(RRC, "Initial ueCapabilityEnquiry sent to NR UE with size %zu\n", requestedFreqBandsNR->size);\
7: LOG_A(NR_RRC, "Sent initial NRUE Capability response to LTE UE\n");\
9:LOG_A(RRC,"NR_UECapabilityInformation Encoded %zd bits (%zd bytes)\n",enc_rval.encoded,(enc_rval.encoded+7)/8);\

![](Images/NSA_Mode_FlowDiagram_2.png)

11: LOG_A(RRC, "Generating RRCCConnectionReconfigurationRequest (NRUE Measurement Report Request).\n");\
14: LOG_A(RRC, "Encoded measurement object %zu bits (%zu bytes) and sent to NR UE\n", enc_rval.encoded, (enc_rval.encoded + 7)/8);\
23: LOG_A(NR_RRC, "Populated NR_UE_RRC_MEASUREMENT information and sent to LTE UE\n");\

![](Images/NSA_Mode_FlowDiagram_3.png)

27: LOG_A(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Logical Channel DL-DCCH, Generate NR UECapabilityEnquiry (bytes %d)\n" PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP), size);\
31: LOG_A(RRC, "Second ueCapabilityEnquiry (request for NR capabilities) sent to NR UE with size %zu\n" requestedFreqBandsNR->size);\
33:  LOG_A(NR_RRC, "[NR_RRC] NRUE Capability encoded, %d bytes (%zd bits)\n", UECap->sdu_size, enc_rval.encoded + 7);\
35:  LOG_A(RRC,"NR_UECapabilityInformation Encoded %zd bits (%zd bytes)\n", enc_rval.encoded, (enc_rval.encoded+7)/8);\
36: LOG_A(RRC, "got nrUE capabilities for UE %x\n", ctxt_pP->rnti);\
37: LOG_A(RRC, "Successfully decoded UE NR capabilities (NR and MRDC)\n");\
38: LOG_A(RRC, "[eNB %d] frame %d subframe %d: UE rnti %x switching to NSA mode\n", ctxt_pP->module_id, ctxt_pP->frame, ctxt_pP->subframe, ctxt_pP->rnti);\
39: LOG_A(NR_RRC, "Successfully parced CG_ConfigInfo of size %zu bits. (%zu bytes)\n", dec_rval.consumed, (dec_rval.consumed +7/8));\
45: LOG_A(RRC, "Sent RRC_CONFIG_COMPLETE_REQ to the NR UE\n");\
47:  LOG_A(NR_RRC, "rrcReconfigurationComplete Encoded %zd bits (%zd bytes)\n", enc_rval.encoded, (enc_rval.encoded+7) 8);\
50: LOG_A(RRC, "Sent rrcReconfigurationComplete to gNB\n");\
51: LOG_A(NR_RRC, "Handling of reconfiguration complete message at RRC gNB is pending \n");\
49. After 49, eNB needs to decode this RRCConnectionReconfigurationComplete message\
50. eNB needs to send RRCConnectionReconfigurationComplete message to gNB (Then gNB knows that the NRUE is expecting traffic) After this gNB will switch user plane to DRB on 5G side\

## 3 Deployment

### 3.1 Deploy RAN/UE/Proxy

1. Open a terminal and clone [openairinterface5G](https://gitlab.eurecom.fr/oai/openairinterface5g)  repo using below command
```bash
git clone https://gitlab.eurecom.fr/oai/openairinterface5g.git
```
2. cd openairinterface5g
3. git checkout 5G-NSA-Configuration-WithEPC
4. Open a terminal and clone [oai-lte-5g-multi-ue-proxy](https://github.com/EpiSci/oai-lte-5g-multi-ue-proxy) repo using below command
```bash
git clone https://github.com/EpiSci/oai-lte-5g-multi-ue-proxy.git
```
5. cd oai-lte-5g-multi-ue-proxy
6. git checkout to below commit ID for multi ue proxy repo:  a6169d8cb8ce4773ac03934b6723b26e70793b3e
7. If you run the proxy in loopback mode, add the following loopback interface for the VNF in the gNB.
```bash
sudo ifconfig lo: 127.0.0.2 netmask 255.0.0.0 up
```
6. Go to path where openairinterface5g is cloned and do below changes in below files. Please note this changes are as per HSS in the EPC.
     a. episci_ue_test_sfr.conf
```bash
vi ci-scripts/conf_files/episci/episci_ue_test_sfr.conf
MSIN="0000059449"
USIM_API_K="fec86ba6eb707ed08905757b1bb44b8f"
OPC="c42449363bbad02b66d16bc975d77cc1";
HPLMN= "00101"
```
     b. proxy_nr-ue.nfapi.conf
```bash
vi ci-scripts/conf_files/episci/proxy_nr-ue.nfapi.conf
imsi = "001010000059449"
```
     c. proxy_rcc.band7.tm1.nfapi.conf
```bash
vi ci-scripts/conf_files/episci/proxy_rcc.band7.tm1.nfapi.conf
tracking_area_code = 1
plmn_list = ( { mcc = 001; mnc = 01; mnc_length = 2; } )
mme_ip_address      = ( { ipv4       = "192.168.61.149";
                          ipv6       = "192:168:30::17";
                          port       = 36412 ;
                          active     = "yes";
                          preference = "ipv4";
                         }
                       );
NETWORK_INTERFACES :
{
   ENB_INTERFACE_NAME_FOR_S1_MME            = "lo";
   ENB_IPV4_ADDRESS_FOR_S1_MME              = "127.0.0.1";
   ENB_INTERFACE_NAME_FOR_S1U               = "lo";
   ENB_IPV4_ADDRESS_FOR_S1U                 = "127.0.0.1";
   ENB_PORT_FOR_S1U                         = 2152; # Spec 2152
   ENB_IPV4_ADDRESS_FOR_X2C                 = "127.0.0.1";
   ENB_PORT_FOR_X2C                         = 36422; # Spec 36422
    };

   #scheduler_mode = "fairRR
```
     d. proxy_rcc.band78.tm1.106PRB.nfapi.conf
```bash
vi ci-scripts/conf_files/episci/proxy_rcc.band78.tm1.106PRB.nfapi.conf
plmn_list = ({mcc = 001; mnc = 01; mnc_length = 2;})
```

_Note: Above configuration is with reference to workiong configuraton on Firecell's setup. On different machine, this configuration shall be udpated w.r.t new setup's environment. _

### 3.2 Generate NV files
```bash
cd openairinterface5g/cmake_targets
1. ./nas_sim_tools/build/conf2uedata -c ../ci-scripts/conf_files/episci/episci_ue_test_sfr.conf -o .
2. ./nas_sim_tools/build/usim -g -c ../ci-scripts/conf_files/episci/episci_ue_test_sfr.conf -o .
3. ./nas_sim_tools/build/nvram -g -c ../ci-scripts/conf_files/episci/episci_ue_test_sfr.conf -o .
```
### 3.3 Build Proxy
```bash
cd .../oai-lte-multi-ue-proxy
make
```
### 3.4 Deploy EPC
For running the System Simulator this step is not mandatory since it is not used when running with the TTCN environment.
The ﬁrst step is to install some mandatory packages from Ubuntu distribution:
- apt-transport-https
- ca-certificates
- curl
- gnupg
- lsb-release

This can be easily done by using apt-get command:

```bash
$sudo apt-get install apt-transport-https ca-certificates curl gnupg lsb-release
```
Now it is possible to download the docker package and install it
```bash
$sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg--dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
```
Update apt-get list of packages
```bash
$echo \ "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg]
https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker list > /dev/null
$sudo apt-get update
```
Install docker
```bash
$sudo apt-get install docker-ce docker-ce-cli containerd.io
```
Once the package is installed the user should have the proper permissions to run docker
```bash
$sudo usermod -a -G docker "yourusername"
```
where "yourusername" shall be replaced with the user login information of the user running the docker instances.

**Now you have to reboot the machine in order to have the changes applied to the system.**

Run the command "groups" after rebooting in order to conﬁrm docker was added to the list.

```bash
$groups
$dpkg --list | grep docker
ii  docker-ce  5:20.10.7~3-0~ubuntu-bionic amd64 Docker: the open-source application container engine
ii  docker-ce-cli 5:20.10.7~3-0~ubuntu-bionic amd64 Docker CLI: the open-source application container engine
ii  docker-ce-rootless-extras 5:20.10.7~3-0~ubuntu-bionic amd64 Rootless support for Docker.
ii  docker-scan-plugin 0.8.0~ubuntu-bionic amd64 Docker scan cli plugin.
```
Now we should download and install docker compose for managing multiple docker instances:

```bash
$sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
$sudo chmod +x /usr/local/bin/docker-compose
```
If not already registered, you need to create the user account in and make the login to pull the ubuntu:bionic and cassandra:2.1 images from docker store:
```bash
$docker login
Login with your Docker ID to push and pull images from Docker Hub. If you don't
have a Docker ID, head over to https://hub.docker.com to create one.
Username: oai01
Password:
WARNING! Your password will be stored unencrypted in
/home/hscuser/.docker/config.json.
Configure a credential helper to remove this warning. See
https://docs.docker.com/engine/reference/commandline/login/#credentials-store

Login Succeeded

$ docker pull ubuntu:bionic
$ docker pull cassandra:2.1
```
**NOTE: By default, traﬃc from containers connected to the default bridge network is NOT forwarded to the outside world. To enable the forwarding, it is necessary to change the following setting:**

- Configure the linux kernel to allow IP forwarding
```bash
$sudo sysctl net.ipv4.conf.all.forwarding=1
```
- Change the policy for the iptables FORWARD policy from DROP to ACCEPT
```bash
$ sudo iptables -P FORWARD ACCEPT
```
Pulling container images:
```bash
$docker pull rdefosseoai/oai-hss
$docker pull rdefosseoai/magma-mme
$docker pull rdefosseoai/oai-spgwc
$docker pull rdefosseoai/oai-spgwu-tiny
```
Re-tag the container images:
```bash
$docker image tag rdefosseoai/oai-hss:latest oai-hss:latest
$docker image tag rdefosseoai/magma-mme:latest magma-mme:latest
$docker image tag rdefosseoai/oai-spgwc:latest oai-spgwc:latest
$docker image tag rdefosseoai/oai-spgwu-tiny:latest oai-spgwu-tiny:latest
$docker logout

$cd <PATHTOEPC>
$cd docker-compose/magma-mme-demo
$docker-compose up -d db_init
Creating network "demo-oai-private-net" with the default driver
Creating network "demo-oai-public-net" with the default driver
Creating demo-cassandra ... done
Creating demo-db-init   ... done
$docker logs demo-db-init --follow
Connection error: ('Unable to connect to any servers', {'192.168.61.2': error(111, "Tried connecting to [('192.168.61.2', 9042)]. Last error: Connection refused")})
Connection error: ('Unable to connect to any servers', {'192.168.61.2': error(111, "Tried connecting to [('192.168.61.2', 9042)]. Last error: Connection refused")})
Connection error: ('Unable to connect to any servers', {'192.168.61.2': error(111, "Tried connecting to [('192.168.61.2', 9042)]. Last error: Connection refused")})
Connection error: ('Unable to connect to any servers', {'192.168.61.2': error(111, "Tried connecting to [('192.168.61.2', 9042)]. Last error: Connection refused")})
Connection error: ('Unable to connect to any servers', {'192.168.61.2': error(111, "Tried connecting to [('192.168.61.2', 9042)]. Last error: Connection refused")})
Connection error: ('Unable to connect to any servers', {'192.168.61.2': error(111, "Tried connecting to [('192.168.61.2', 9042)]. Last error: Connection refused")})
Connection error: ('Unable to connect to any servers', {'192.168.61.2': error(111, "Tried connecting to [('192.168.61.2', 9042)]. Last error: Connection refused")})
OK
```
You must have the "OK" message in the demo-db-init container logs, in order to proceed with the next steps. If OK is not displayed, please repeat the command.
**NOTE: the error messages are normal. They do not prevent the system from working properly.**
Remove prod-db-init container:
```bash
$docker rm -f demo-db-init
```
On your EPC docker host, check the routing table and identify the default gateway:
```bash
$ route -n Kernel IP routing table
Destination Gateway Genmask Flags Metric Ref Use Iface 0.0.0.0 192.168.253.1 0.0.0.0 UG 100 0 0 enp1s0 ...
```
Use the default gateway as DNS in the magma-mme-demo and Edit/modify the docker-compose ﬁle in order to:
```bash
$cd <PATHTOEPC>/sequansRD/EPC/docker-compose/magma-mme-demo
$vi docker-compose.yml
```
- DEFAULT_DNS_IPV4_ADDRESS: 192.168.253.1
- In services, go to magma_mme and change the name of image to "magma-mme:master" and container_name to "demo-magma-mme" e.g.
```bash
            magma_mme:
        	image: magma-mme:master
        	container_name: demo-magma-mme

```
**NOTE: Please make sure you add the proper route in the eNB server as shown earlier**
Now we can start EPC:
```bash
$docker-compose up -d oai_spgwu
Creating demo-redis   ... done
Creating demo-oai-hss ... done
Creating demo-magma-mme ... done
Creating demo-oai-spgwc ... done
Creating demo-oai-spgwu-tiny ... done
```
Check that EPC is running
```bash
$ docker ps -a
CONTAINER ID   IMAGE          		             COMMAND                  CREATED          STATUS                      PORTS                                         NAMES
d54cc50fd891   oai-spgwu-tiny:production   "/openair-spgwu-tiny…"   3 hours ago   Up 3 hours (healthy)       2152/udp, 8805/udp                            demo-oai-spgwu-tiny
b77887b63d47   oai-spgwc:production        "/openair-spgwc/bin/…"   3 hours ago   Up 3 hours (healthy)       2123/udp, 8805/udp                            demo-oai-spgwc
74c208df5dd4   magma-mme:master            "/bin/bash -c 'cd /m…"   3 hours ago   Up 3 hours (healthy)       3870/tcp, 2123/udp, 5870/tcp                  demo-magma-mme
7d86bf3316c7   oai-hss:production          "/openair-hss/bin/en…"   3 hours ago   Up 3 hours (healthy)       5868/tcp, 9042/tcp, 9080-9081/tcp             demo-oai-hss
7a234e8d1d7c   cassandra:2.1               "docker-entrypoint.s…"   3 hours ago   Up 3 hours (healthy)       7000-7001/tcp, 7199/tcp, 9042/tcp, 9160/tcp   demo-cassandra
```
In order to access the EPC components from the eNB we need to know the IP address of MME and SP-GW. In order to retrieve those IPs you can run the following commands:
For MME:
```bash
$ docker inspect --format="{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}" demo-magma-mme 192.168.61.4
```
For SP-GW:
```bash
$ docker inspect --format="{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}" demo-oai-spgwu-tiny 192.168.61.6
```

**NOTE: MME has a short life. It is better to compose it every time before using it. Each time you modify the docker-compose ﬁle, you have to repeat docker-compose up -d oai_spgwu. Each time you reboot your machine or you see with "docker ps -a" that an EPC component has exited you have to restart from scratch following these steps:**

```bash
$cd <PATHTOEPC>/docker-compose/magma-mme-demo
$docker-compose down
$docker ps -a
```
make sure that linux kernel allows IP forwarding:

```bash
$sudo sysctl net.ipv4.conf.all.forwarding=1
$sudo iptables -P FORWARD ACCEPT
$sudo iptables \--list
```
Restart Database
```bash
$docker-compose up -d db_init
$docker logs demo-db-init  --follow
$docker rm -f demo-db-init
```
please make sure the last line is OK otherwise you need to repeat the previous steps
Restart **EPC**
```bash
$docker-compose up -d oai_spgwu
$docker ps -a
```
For restart from scratch, run the below shell script also :
```bash
$cd <PATHTOEPC>/sequansRD/EPC/docker-compose/magma-mme-demo
$./startEPC.sh
```
In order to check errors in MME and HSS you can do the follow:
```bash
$ docker logs demo-oai-hss 2>&1 > hss.log
$ docker logs demo-magma-mme 2>&1 > mme.log
```
### 3.5 Build OAI
```bash
cd .../openairinterface5g
source oaienv
cd cmake_targets
sudo ./build_oai --UE --eNB --nrUE --gNB
```
### 3.6 Run NSA mode with the script
```bash
./proxy_testscript.py --num-ues 1 --mode=nsa --duration=300
```
Duration is optional.
Note: This step is for checking the sanity of the setup and is optional. 
### 3.7 Run without the proxy_testscript.py
The launch order is important, as follows.
#### 3.7.1 Launch eNB in Terminal 1
```bash
cd .../openairinterface5g
source oaienv
cd cmake_targets
node_id=1
sudo -E LD_LIBRARY_PATH=/home/firecell/openairinterface5G/openairinterface5g/cmake_targets/ran_build/build: taskset --cpu-list 1 ./ran_build/build/lte-softmodem -O ../ci-scripts/conf_files/episci/proxy_rcc.band7.tm1.nfapi.conf --noS1 --emulate-l1 --log_config.global_log_options level,nocolor,time,thread_id --nsa | tee eNB.log 2>&1
```
#### 3.7.2 Launch Proxy in Terminal 2
```bash
cd .../oai-lte-multi-ue-proxy
sudo -E ./build/proxy 1 --nsa enb_ipaddr gnb_ipaddr proxy_ipaddr ue_ipaddr
or
sudo ./build/proxy 1 --nsa
```
If you do not specify the parameters ending with ipaddr, the default IP addresses are as follows:
- enb_ipaddr = 127.0.0.1
- gnb_ipaddr = 127.0.0.2
- proxy_ipaddr = 127.0.0.1
- ue_ipaddr = 127.0.0.1
#### 3.7.3 Launch lteUE in Terminal 3
```bash
cd .../openairinterface5g
source oaienv
cd cmake_targets
node_id=2
sudo -E LD_LIBRARY_PATH=/home/firecell/openairinterface5G/openairinterface5g/cmake_targets/ran_build/build: taskset --cpu-list 2 ./ran_build/build/lte-uesoftmodem -O ../ci-scripts/conf_files/episci/proxy_ue.nfapi.conf --L2-emul 5 --nokrnmod 1 --noS1 --num-ues 1 --node-number 2 --log_config.global_log_options level,nocolor,time,thread_id --nsa | tee ue.log 2>&1
```
#### 3.7.4 Launch gNB in Terminal 4
```bash
cd .../openairinterface5g
source oaienv
cd cmake_targets
node_id=3
sudo -E LD_LIBRARY_PATH=/home/firecell/openairinterface5G/openairinterface5g/cmake_targets/ran_build/build: taskset --cpu-list 1 ./ran_build/build/nr-softmodem -O ../ci-scripts/conf_files/episci/proxy_rcc.band78.tm1.106PRB.nfapi.conf --nfapi VNF --noS1 --emulate-l1 --log_config.global_log_options level,nocolor,time,thread_id --nsa | tee gNB.log 2>&1
```
#### 3.7.5 Launch nrUE in Terminal 5
```bash
cd .../openairinterface5g
source oaienv
cd cmake_targets
node_id=2
sudo -E LD_LIBRARY_PATH=/home/firecell/openairinterface5G/openairinterface5g/cmake_targets/ran_build/build: taskset --cpu-list 2 ./ran_build/build/nr-uesoftmodem -O ../ci-scripts/conf_files/episci/proxy_nr-ue.nfapi.conf --nokrnmod 1 --nfapi STANDALONE_PNF --node-number 2 --emulate-l1 --log_config.global_log_options level,nocolor,time,thread_id --nsa | tee nrUE.log 2>&1
```

### 3.8 Checking the log result
After running the programs for 30 seconds or more, stop the processes using Ctrl-C. Open the log files and check the following logs to verify the run results.
- gNB.log : search for "CFRA procedure succeeded" log message for each UE.
- eNB.log : search for "Sent rrcReconfigurationComplete to gNB".
- nrUE.log : search for "Found RAR with the intended RAPID".
- ue.log : search for "Sent RRC_CONFIG_COMPLETE_REQ to the NR UE".
