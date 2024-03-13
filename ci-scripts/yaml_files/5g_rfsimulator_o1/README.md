<table style="border-collapse: collapse; border: none;">
  <tr style="border-collapse: collapse; border: none;">
    <td style="border-collapse: collapse; border: none;">
      <a href="http://www.openairinterface.org/">
         <img src="../../../doc/images/oai_final_logo.png" alt="" border=3 height=50 width=150>
         </img>
      </a>
    </td>
    <td style="border-collapse: collapse; border: none; vertical-align: center;">
      <b><font size = "5">Testing O1 in containers</font></b>
    </td>
  </tr>
</table>

**Table of Contents**

[[_TOC_]]


# 1. Deploy containers #

## 2.1. Deploy OAI 5G Core Network ##

```bash
docker-compose up -d mysql oai-amf oai-smf oai-upf oai-ext-dn
[+] Running 11/11
 ⠿ Network rfsim5g-oai-core-net     Created                                                                                                                                                                   0.1s
 ⠿ Network rfsim5g-oai-ue-net       Created                                                                                                                                                                   0.1s
 ⠿ Network rfsim5g-oai-traffic-net  Created                                                                                                                                                                   0.1s
 ⠿ Network rfsim5g-oai-f1c-net      Created                                                                                                                                                                   0.1s
 ⠿ Network rfsim5g-oai-f1u-1-net    Created                                                                                                                                                                   0.1s
 ⠿ Network rfsim5g-oai-o1-net       Created                                                                                                                                                                   0.1s
 ⠿ Network rfsim5g-oai-e1-net       Created                                                                                                                                                                   0.1s
 ⠿ Container rfsim5g-mysql          Started                                                                                                                                                                   0.6s
 ⠿ Container rfsim5g-oai-amf        Started                                                                                                                                                                   0.8s
 ⠿ Container rfsim5g-oai-smf        Started                                                                                                                                                                   1.0s
 ⠿ Container rfsim5g-oai-upf        Started                                                                                                                                                                   1.3s
 ⠿ Container rfsim5g-oai-ext-dn     Started                                                                                                                                                                   1.5s
```

```bash
docker-compose ps -a
       Name                     Command                  State                  Ports
-------------------------------------------------------------------------------------------------
rfsim5g-mysql        "docker-entrypoint.s…"   mysql               running (healthy)   3306/tcp, 33060/tcp
rfsim5g-oai-amf      "/openair-amf/bin/oa…"   oai-amf             running (healthy)   80/tcp, 9090/tcp, 38412/sctp
rfsim5g-oai-ext-dn   "/bin/bash -c ' ipta…"   oai-ext-dn          running (healthy)   
rfsim5g-oai-smf      "/openair-smf/bin/oa…"   oai-smf             running (healthy)   80/tcp, 8080/tcp, 8805/udp
rfsim5g-oai-upf      "/openair-upf/bin/oa…"   oai-upf             running (healthy)   2152/udp, 8805/udp
```

## 2.2. Deploy OAI CU-CP, CU-UP and DU ##


```bash
docker-compose up -d oai-cu-cp oai-cu-up oai-du
[+] Running 6/6
 ⠿ Container rfsim5g-mysql       Running                                                                                                                                                                      0.0s
 ⠿ Container rfsim5g-oai-amf     Running                                                                                                                                                                      0.0s
 ⠿ Container rfsim5g-oai-smf     Running                                                                                                                                                                      0.0s
 ⠿ Container rfsim5g-oai-upf     Running                                                                                                                                                                      0.0s
 ⠿ Container rfsim5g-oai-ext-dn  Running                                                                                                                                                                      0.0s
 ⠿ Container rfsim5g-oai-gnb     Started                                                                                                                                                                      0.4s
```

## Check Stats via telnet O1 Module ##

Make sure you have installed `netcat` in your host machine or container. 

In case you don't have then install in the gnb contianer 

```bash
docker exec -it rfsim5g-oai-gnb bash
apt update -y && apt install netcat -y
```

```bash
echo o1 stats | nc -N 192.168.74.2 9090
```

<details>
<summary>The output is similar to:</summary>

```console
{
  "o1-config": {
    "BWP": {
      "dl": [{
        "bwp3gpp:isInitialBwp": true,
        "bwp3gpp:numberOfRBs": 51,
        "bwp3gpp:startRB": 0,
        "bwp3gpp:subCarrierSpacing": 30
      }],
      "ul": [{
        "bwp3gpp:isInitialBwp": true,
        "bwp3gpp:numberOfRBs": 51,
        "bwp3gpp:startRB": 0,
        "bwp3gpp:subCarrierSpacing": 30
      }]
    },
    "NRCELLDU": {
      "nrcelldu3gpp:ssbFrequency": 627360,
      "nrcelldu3gpp:arfcnDL": 626748,
      "nrcelldu3gpp:bSChannelBwDL": 20,
      "nrcelldu3gpp:arfcnUL": 626748,
      "nrcelldu3gpp:bSChannelBwUL": 20,
      "nrcelldu3gpp:nRPCI": 0,
      "nrcelldu3gpp:nRTAC": 1,
      "nrcelldu3gpp:mcc": "208",
      "nrcelldu3gpp:mnc": "99",
      "nrcelldu3gpp:sd": 16777215,
      "nrcelldu3gpp:sst": 1
    },
    "device": {
      "gnbId": 3584,
      "gnbName": "du-rfsim",
      "vendor": "OpenAirInterface"
    }
  },
  "O1-Operational": {
    "frame-type": "tdd",
    "band-number": 78,
    "num-ues": 0,
    "ues": [    ],
    "load": 0,
    "ues-thp": [
    ]
  }
}
```
</details>

Notice the block `O1-Operational` you will see the number of connected UEs, present load at the gNB and the DL/UL RLC `throughput` in `Kbps`.

## Deploy OAI NR-UE in RF simulator mode and in Standalone Mode ##

```bash
docker-compose up -d oai-nr-ue
[+] Running 7/7
 ⠿ Container rfsim5g-mysql       Running                                                                                                                                                                      0.0s
 ⠿ Container rfsim5g-oai-amf     Running                                                                                                                                                                      0.0s
 ⠿ Container rfsim5g-oai-smf     Running                                                                                                                                                                      0.0s
 ⠿ Container rfsim5g-oai-upf     Running                                                                                                                                                                      0.0s
 ⠿ Container rfsim5g-oai-ext-dn  Running                                                                                                                                                                      0.0s
 ⠿ Container rfsim5g-oai-gnb     Running                                                                                                                                                                      0.0s
 ⠿ Container rfsim5g-oai-nr-ue   Started                                                                                                                                                                      0.4s
```

Wait for a bit.

```bash
docker-compose ps -a
       Name                     Command                  State                  Ports
-------------------------------------------------------------------------------------------------
rfsim5g-mysql        "docker-entrypoint.s…"   mysql               running (healthy)   3306/tcp, 33060/tcp
rfsim5g-oai-amf      "/openair-amf/bin/oa…"   oai-amf             running (healthy)   80/tcp, 9090/tcp, 38412/sctp
rfsim5g-oai-ext-dn   "/bin/bash -c ' ipta…"   oai-ext-dn          running (healthy)   
rfsim5g-oai-gnb      "/opt/oai-gnb/bin/en…"   oai-gnb             running (healthy)   
rfsim5g-oai-nr-ue    "/opt/oai-nr-ue/bin/…"   oai-nr-ue           running (healthy)   
rfsim5g-oai-smf      "/openair-smf/bin/oa…"   oai-smf             running (healthy)   80/tcp, 8080/tcp, 8805/udp
rfsim5g-oai-upf      "/openair-upf/bin/oa…"   oai-upf             running (healthy)   2152/udp, 8805/udp
```

Making sure the OAI UE is connected:

```bash
docker exec -it rfsim5g-oai-nr-ue ifconfig oaitun_ue1
oaitun_ue1: flags=4305<UP,POINTOPOINT,RUNNING,NOARP,MULTICAST>  mtu 1500
        inet 12.1.1.2  netmask 255.255.255.0  destination 12.1.1.2
        unspec 00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00  txqueuelen 500  (UNSPEC)
        RX packets 0  bytes 0 (0.0 B)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 0  bytes 0 (0.0 B)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
```

Now if you will re-check the stats you will see 


```bash
echo o1 stats | nc -N 192.168.74.2 9090
```

<details>
<summary>The output is similar to:</summary>

```console
{
  "o1-config": {
    "BWP": {
      "dl": [{
        "bwp3gpp:isInitialBwp": true,
        "bwp3gpp:numberOfRBs": 51,
        "bwp3gpp:startRB": 0,
        "bwp3gpp:subCarrierSpacing": 30
      }],
      "ul": [{
        "bwp3gpp:isInitialBwp": true,
        "bwp3gpp:numberOfRBs": 51,
        "bwp3gpp:startRB": 0,
        "bwp3gpp:subCarrierSpacing": 30
      }]
    },
    "NRCELLDU": {
      "nrcelldu3gpp:ssbFrequency": 627360,
      "nrcelldu3gpp:arfcnDL": 626748,
      "nrcelldu3gpp:bSChannelBwDL": 20,
      "nrcelldu3gpp:arfcnUL": 626748,
      "nrcelldu3gpp:bSChannelBwUL": 20,
      "nrcelldu3gpp:nRPCI": 0,
      "nrcelldu3gpp:nRTAC": 1,
      "nrcelldu3gpp:mcc": "208",
      "nrcelldu3gpp:mnc": "99",
      "nrcelldu3gpp:sd": 16777215,
      "nrcelldu3gpp:sst": 1
    },
    "device": {
      "gnbId": 3584,
      "gnbName": "du-rfsim",
      "vendor": "OpenAirInterface"
    }
  },
  "O1-Operational": {
    "frame-type": "tdd",
    "band-number": 78,
    "num-ues": 1,
    "ues": [39874    ],
    "load": 0,
    "ues-thp": [
      {"rnti": 39874, "dl": 0, "ul": 0}
    ]
  }
}
```
</details>

You can do some throughput test and check load for this you need to install `iperf3` in the UE container 

```bash
docker exec -it rfsim5g-oai-nr-ue bash
apt update -y && apt install iperf3 -y
```
Iperf3 server is already running you can start Iperf3 client

```bash
docker exec -it rfsim5g-oai-nr-ue iperf3 -B 12.1.1.2 -b 60M -c 192.168.72.135 -R -t 100
```

Measure the load


```bash
echo o1 stats | nc -N 192.168.74.2 9090 
```

<details>
<summary>The output is similar to:</summary>

```console
  "O1-Operational": {
    "frame-type": "tdd",
    "band-number": 78,
    "num-ues": 1,
    "ues": [39874    ],
    "load": 95,
    "ues-thp": [
      {"rnti": 39874, "dl": 52596, "ul": 913}
    ]
  }
```
</details>

Now to re-configure the bandwidth you have to follow the below steps 

1. Stop the simulated UE

```bash
docker kill rfsim5g-oai-nr-ue
```
2. Stop the L1
```bash
echo o1 stop_modem | nc -N 192.168.74.2 9090
```
3. Reconfigure bandwidth
```bash
echo o1 bwconfig 40 | nc -N 192.168.74.2 9090
```
4. Start L1
```bash
echo o1 start_modem | nc -N 192.168.74.2 9090
```

5. Re-connect UE