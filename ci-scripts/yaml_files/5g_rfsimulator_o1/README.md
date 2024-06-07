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
[+] Running 12/12
 ✔ Network rfsim5g-oai-core-net     Created                                                                                                                                                                   0.1s 
 ✔ Network rfsim5g-oai-traffic-net  Created                                                                                                                                                                   0.1s 
 ✔ Network rfsim5g-oai-e1-net       Created                                                                                                                                                                   0.1s 
 ✔ Network rfsim5g-oai-f1u-1-net    Created                                                                                                                                                                   0.1s 
 ✔ Network rfsim5g-oai-f1c-net      Created                                                                                                                                                                   0.1s 
 ✔ Network rfsim5g-oai-ue-net       Created                                                                                                                                                                   0.1s 
 ✔ Network rfsim5g-oai-o1-net       Created                                                                                                                                                                   0.1s 
 ✔ Container rfsim5g-mysql          Started                                                                                                                                                                   0.3s 
 ✔ Container rfsim5g-oai-amf        Started                                                                                                                                                                   0.5s 
 ✔ Container rfsim5g-oai-smf        Started                                                                                                                                                                   0.7s 
 ✔ Container rfsim5g-oai-upf        Started                                                                                                                                                                   0.9s 
 ✔ Container rfsim5g-oai-ext-dn     Started                                                                                                                                                                   1.1s 
```

```bash
docker-compose ps -a
NAME                 IMAGE                                    COMMAND                  SERVICE      CREATED              STATUS                        PORTS
rfsim5g-mysql        mysql:8.0                                "docker-entrypoint.s…"   mysql        About a minute ago   Up About a minute (healthy)   3306/tcp, 33060/tcp
rfsim5g-oai-amf      oaisoftwarealliance/oai-amf:v2.0.0       "/openair-amf/bin/oa…"   oai-amf      About a minute ago   Up About a minute (healthy)   80/tcp, 9090/tcp, 38412/sctp
rfsim5g-oai-ext-dn   oaisoftwarealliance/trf-gen-cn5g:focal   "/bin/bash -c ' ipta…"   oai-ext-dn   About a minute ago   Up About a minute (healthy)   
rfsim5g-oai-smf      oaisoftwarealliance/oai-smf:v2.0.0       "/openair-smf/bin/oa…"   oai-smf      About a minute ago   Up About a minute (healthy)   80/tcp, 8080/tcp, 8805/udp
rfsim5g-oai-upf      oaisoftwarealliance/oai-upf:v2.0.0       "/openair-upf/bin/oa…"   oai-upf      About a minute ago   Up About a minute (healthy)   2152/udp, 8805/udp
```

## 2.2. Deploy OAI CU-CP, CU-UP and DU ##


```bash
docker-compose up -d oai-cu-cp oai-cu-up oai-du
[+] Running 7/7
 ✔ Container rfsim5g-mysql     Running                                                                                                                                                                        0.0s 
 ✔ Container rfsim5g-oai-amf   Running                                                                                                                                                                        0.0s 
 ✔ Container rfsim5g-oai-smf   Running                                                                                                                                                                        0.0s 
 ✔ Container rfsim5g-oai-upf   Running                                                                                                                                                                        0.0s 
 ✔ Container rfsim5g-oai-cucp  Started                                                                                                                                                                        0.4s 
 ✔ Container rfsim5g-oai-cuup  Started                                                                                                                                                                        0.8s 
 ✔ Container rfsim5g-oai-du    Started                                                                                                                                                                        1.2s 
```

## Check Stats via telnet O1 Module ##

Make sure you have installed `netcat` in your host machine or container. 

In case you don't have then install in the gnb container

```bash
docker exec -it rfsim5g-oai-du bash
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
      "nrcelldu3gpp:ssbFrequency": 630048,
      "nrcelldu3gpp:arfcnDL": 629436,
      "nrcelldu3gpp:bSChannelBwDL": 20,
      "nrcelldu3gpp:arfcnUL": 629436,
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
[+] Running 8/8
 ✔ Container rfsim5g-mysql      Running                                                                                                                                                                       0.0s 
 ✔ Container rfsim5g-oai-amf    Running                                                                                                                                                                       0.0s 
 ✔ Container rfsim5g-oai-smf    Running                                                                                                                                                                       0.0s 
 ✔ Container rfsim5g-oai-upf    Running                                                                                                                                                                       0.0s 
 ✔ Container rfsim5g-oai-cucp   Running                                                                                                                                                                       0.0s 
 ✔ Container rfsim5g-oai-cuup   Running                                                                                                                                                                       0.0s 
 ✔ Container rfsim5g-oai-du     Running                                                                                                                                                                       0.0s 
 ✔ Container rfsim5g-oai-nr-ue  Started                                                                                                                                                                       0.2s
```

Wait for a bit.

```bash
docker-compose ps -a
NAME                 IMAGE                                      COMMAND                  SERVICE      CREATED          STATUS                    PORTS
rfsim5g-mysql        mysql:8.0                                  "docker-entrypoint.s…"   mysql        30 minutes ago   Up 30 minutes (healthy)   3306/tcp, 33060/tcp
rfsim5g-oai-amf      oaisoftwarealliance/oai-amf:v2.0.0         "/openair-amf/bin/oa…"   oai-amf      30 minutes ago   Up 30 minutes (healthy)   80/tcp, 9090/tcp, 38412/sctp
rfsim5g-oai-cucp     arorasagar/testing-images:oai-gnb-telnet   "/opt/oai-gnb/bin/en…"   oai-cu-cp    29 minutes ago   Up 29 minutes (healthy)   
rfsim5g-oai-cuup     oaisoftwarealliance/oai-nr-cuup:v2.1.0     "/opt/oai-gnb/bin/en…"   oai-cu-up    29 minutes ago   Up 29 minutes (healthy)   
rfsim5g-oai-du       arorasagar/testing-images:oai-gnb-telnet   "/opt/oai-gnb/bin/en…"   oai-du       5 minutes ago    Up 5 minutes (healthy)    
rfsim5g-oai-ext-dn   oaisoftwarealliance/trf-gen-cn5g:focal     "/bin/bash -c ' ipta…"   oai-ext-dn   30 minutes ago   Up 30 minutes (healthy)   
rfsim5g-oai-nr-ue    oaisoftwarealliance/oai-nr-ue:v2.1.0       "/opt/oai-nr-ue/bin/…"   oai-nr-ue    4 minutes ago    Up 4 minutes (healthy)    
rfsim5g-oai-smf      oaisoftwarealliance/oai-smf:v2.0.0         "/openair-smf/bin/oa…"   oai-smf      30 minutes ago   Up 30 minutes (healthy)   80/tcp, 8080/tcp, 8805/udp
rfsim5g-oai-upf      oaisoftwarealliance/oai-upf:v2.0.0         "/openair-upf/bin/oa…"   oai-upf      30 minutes ago   Up 30 minutes (healthy)   2152/udp, 8805/udp
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
exit
```
Iperf3 server is already running you can start Iperf3 client

```bash
docker exec -it rfsim5g-oai-nr-ue iperf3 -B 12.1.1.2 -b 80M -c 192.168.72.135 -R -t 100
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

Now re-configure the bandwidth 

1. Stop the L1

```bash
echo o1 stop_modem | nc -N 192.168.74.2 9090
```

2. Reconfigure bandwidth

```bash
echo o1 bwconfig 40 | nc -N 192.168.74.2 9090
```

3. Start L1

```bash
echo o1 start_modem | nc -N 192.168.74.2 9090
```

4. Check the reconfigured bandwidth 

```bash
echo o1 stats | nc -N 192.168.74.2 9090 
```

```bash
{
  "o1-config": {
    "BWP": {
      "dl": [{
        "bwp3gpp:isInitialBwp": true,
        "bwp3gpp:numberOfRBs": 106,
        "bwp3gpp:startRB": 0,
        "bwp3gpp:subCarrierSpacing": 30
      }],
      "ul": [{
        "bwp3gpp:isInitialBwp": true,
        "bwp3gpp:numberOfRBs": 106,
        "bwp3gpp:startRB": 0,
        "bwp3gpp:subCarrierSpacing": 30
      }]
    },
    "NRCELLDU": {
      "nrcelldu3gpp:ssbFrequency": 641280,
      "nrcelldu3gpp:arfcnDL": 640008,
      "nrcelldu3gpp:bSChannelBwDL": 40,
      "nrcelldu3gpp:arfcnUL": 640008,
      "nrcelldu3gpp:bSChannelBwUL": 40,
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
    "load": 14,
    "ues-thp": [
    ]
  }
}
```

You can see `nrcelldu3gpp:bSChannelBwUL` as 40.

5. Re-connect UE

In the [docker-compose.yaml](./docker-compose.yaml) file in `oai-nr-ue` section you need to change the UE command to 40MHz bandwidth.

```
           # #           20 MHz
           #  USE_ADDITIONAL_OPTIONS: --sa --rfsim --log_config.global_log_options level,nocolor,time
           #                          -r 51 --numerology 1 -C 3450720000 --ssb 186
           #                          --uicc0.imsi 208990100001100
           #                          --rfsimulator.serveraddr 192.168.78.2
           #            40 MHz
            USE_ADDITIONAL_OPTIONS: --sa -E --rfsim --log_config.global_log_options level,nocolor,time
                                   -r 106 --numerology 1 -C 3619200000
                                   --uicc0.imsi 208990100001100
                                   --rfsimulator.serveraddr 192.168.78.2
```

```bash
docker rm rfsim5g-oai-nr-ue
docker-compose up -d oai-nr-ue
```

After this you can deploy the UE and again do an iperf3 test you will see the load is almost half now

```
  "O1-Operational": {
    "frame-type": "tdd",
    "band-number": 78,
    "num-ues": 1,
    "ues": [25333    ],
    "load": 52,
    "ues-thp": [
      {"rnti": 25333, "dl": 83266, "ul": 1123}
    ]
  }
```