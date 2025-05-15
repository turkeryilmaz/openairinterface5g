# Joint sidelink and Uu communication (JSUC)
This tutorial will discuss the setup for a hybrid communication mode using sidelink and Uu.

## Setup
The idea is to have 4 UEs, two running sidelink and two running normal mode. A logical UE supporting hybrid communication mode is represented by 2 UEs, one running sidelink and the other running normal mode.

The following repos will be used for each technology:
- Uu: `mkdir normal-mode`, `cd normal-mode` then clone `https://github.com/beraoudabdelkhalek/rfsim-Uu-metrics`
- Sidelink: `mkdir sidelink`, `cd sidelink` then clone `https://gitlab.eurecom.fr/oai/openairinterface5g` and `git checkout sl-eurecom4`

- Build both projects, and build the telnet server for each project `./build_oai --build-lib telnetsrv`
- Run the OAI core network.
- Create 4 namespaces: `sudo ~/normal-mode/rfsim-Uu-metrics/tools/script/multi-ue.sh -c x` where x is from 1-4
- Run the gNB executable on the host using the project inside the `normal-mode` directory:
```sudo ./nr-softmodem -O ../../../targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf --gNBs.[0].min_rxtxtime 6 --rfsim --rfsimulator.options chanmod --telnetsrv```
- Run the broker on the host using the project inside `sidelink` directory.
- Run UE 1 in ns1 (normal mode) using `normal-mode` directory:
`sudo ~/normal-mode/rfsim-Uu-metrics/tools/script/multi-ue.sh -o 1`

```sudo ./nr-uesoftmodem -r 106 --numerology 1 --band 78 -C 3619200000 --rfsim --rfsimulator.serveraddr 10.201.1.100 --uicc0.imsi 001010000000001 -O ../../../ci-scripts/conf_files/nrue.uicc.conf --rfsimulator.options chanmod --telnetsrv --telnetsrv.listenport 9091```

- Run UE2 in ns2 (sidelink mode) using `sidelink` directory:
`sudo ~/normal-mode/rfsim-Uu-metrics/tools/script/multi-ue.sh -o 2`

```sudo RFSIMULATOR=server ./nr-uesoftmodem --rfsim -O ../../../targets/PROJECTS/NR-SIDELINK/CONF/sl_sync_ref.conf --sl-mode 2 --sa --sync-ref --brokerip 10.202.1.100```

- Run UE 3 in ns3 (sidelink mode) using `sidelink` directory:
`sudo ~/normal-mode/rfsim-Uu-metrics/tools/script/multi-ue.sh -o 3`

```sudo RFSIMULATOR=127.0.0.1 ./nr-uesoftmodem --rfsim -O ../../../targets/PROJECTS/NR-SIDELINK/CONF/sl_ue1.conf --sl-mode 2 --sa --brokerip 10.203.1.100 --device_id 1```

- Run UE 4 in ns4 (normal mode) using `normal-mode` directory:
`sudo ~/normal-mode/rfsim-Uu-metrics/tools/script/multi-ue.sh -o 4`

```sudo ./nr-uesoftmodem -r 106 --numerology 1 --band 78 -C 3619200000 --rfsim --rfsimulator.serveraddr 10.204.1.100 --uicc0.imsi 001010000000002 -O ../../../ci-scripts/conf_files/nrue.uicc.conf --rfsimulator.options chanmod --telnetsrv --telnetsrv.listenport 9091```

## Connectivity test

### Uu Connectivity:
From UE 1 ping UE 4:
- `sudo ~/normal-mode/rfsim-Uu-metrics/tools/script/multi-ue.sh -o 1`
- `ping 10.0.0.3`

### Sidelink Connectivity:
From UE 2 ping UE 3:
- `sudo ~/normal-mode/rfsim-Uu-metrics/tools/script/multi-ue.sh -o 2`
- `ping 10.0.0.2`
