# Procedure to install and run RTT based ranging using OAI basestation (gNB) and the user equipment (nrUE)

Open a terminal and clone the ran repository
```bash
git clone https://gitlab.eurecom.fr/oai/openairinterface5g.git
```
compile the gNB and nrUE

```bash
cd openairinterface5g/
git checkout RTT_based_ranging
source oaienv
cd cmake_targets/
./build_oai -I  
./build_oai -w USRP --gNB --nrUE --ninja --build-lib telnetsrv
```
NOTE: We assume that you have already installed USRP drivers. If not refer to Section 3.1 in 
 [NR_SA_Tutorial_OAI_nrUE.md](NR_SA_Tutorial_OAI_nrUE.md)

# Verify RTT based range estimation in RFSIMULATOR mode
Run the gNB

```bash
cd ~/openairinterface5g/cmake_targets/ran_build/build
sudo ./nr-softmodem --phy-test -E --gNBs.[0].min_rxtxtime 6 --rfsim -D 0 -U 0 -O ../../../targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210_RTT.conf --srs_threshold 0.3
```


Run the UE from a second terminal:

```bash
cd ~/openairinterface5g/cmake_targets/ran_build/build
sudo ./nr-uesoftmodem -E --phy-test --rfsim -O targets/PROJECTS/GENERIC-NR-5GC/CONF/ue.nr.prs.fr1.106prb_RTT.conf --telnetsrv --rfsimulator.options chanmod --prs_threshold 0.3
```

Note that the options `--prs_threshold` and `--srs_threshold` lies between (0, 1] is the normalized theshold to detect the peak of the channel impulse response.

Verify that it is connected: you should see the following output at gNB:

```
[NR_PHY]   rxAnt 0, SRS ToA peak estimator 0, srs threshold 0.300000
[NR_PHY]   Distance between gNB and UE 0.0

```
and nrUE:

```
[PHY]   [gNB 0][rsc 0][Rx 0][sfn 211][slot 1] DL PRS ToA ==> 0.0 / 1536 samples, peak channel power -14.1 dBm, SNR +4.0 dB, rsrp -41.1 dBm

```

This log SRS ToA peak estimator provides the RTT in samples

Convert it to range using (SRS ToA peak estimator)/(2*fs)

fs is the sampling rate

---
# Verify by varying the distances using telnet server
Open a new terminal

```
telnet 0 9090
```
 you will see the following log

 ```
 Trying 0.0.0.0...
Connected to 0.
Escape character is '^]'.

softmodem_5Gue>
 ```

set the distance using `setdistance` command from telnet

```
softmodem_5Gue> rfsimu setdistance rfsimu_channel_enB0 100
```

The interpretation of setdistance can be taken as 2*distance between the gNB and UE in meters.

you should see the following log at the telnet server
```
Trying 0.0.0.0...
Connected to 0.
Escape character is '^]'.

softmodem_5Gue> rfsimu setdistance rfsimu_channel_enB0 100
softmodem_5Gue> 
rfsimu_setdistance_cmd: new_offset 15, new (exact) distance 97.589 m, new delay 0.000326 ms
```

correspondingly you should see the RTT in samples at the gNB:

```
[NR_PHY]   rxAnt 0, SRS ToA peak estimator 15, srs threshold 0.300000
[NR_PHY]   Distance between gNB and UE 48.828125
```

The distance between the gNB and UE = (SRS ToA peak estimator)/(2*fs)

fs is the sampling rate

# Range estimation using USRP

Run the gNB:

```bash
cd ~/openairinterface5g/cmake_targets/ran_build/build
sudo ./nr-softmodem --phy-test -E --gNBs.[0].min_rxtxtime 6 -D 0 -U 0 -O ../../../targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210_RTT.conf --srs_threshold 0.3
```

Run the UE:

```bash
cd ~/openairinterface5g/cmake_targets/ran_build/build
sudo ./nr-uesoftmodem -E --phy-test -O targets/PROJECTS/GENERIC-NR-5GC/CONF/ue.nr.prs.fr1.106prb_RTT.conf --prs_threshold 0.3
```

NOTE: While taking the measurements using hardware, use `-A <samples>` option at the UE to calibrate the distance between the gNB and UE.

NOTE: Keep the gNB and UE at a known distance and calibrate it using `-A` option.

# Scripts

The scripts to record and analyze the data can be found in the [link](link)

Steps to record and analyze the required data can also be found in reference [2]

# Datasets

Datasets collected at the Northeastern university burlington campus can be found in [link](link)

# References
[1]. R. Mundlamuri, R. Gangula, F. Kaltenberger and R. Knopp, "Novel Round Trip Time Estimation in 5G NR," GLOBECOM 2024 - 2024 IEEE Global Communications Conference, Cape Town, South Africa, 2024, pp. 3069-3074, doi: 10.1109/GLOBECOM52923.2024.10901749.

[2]. R. Mundlamuri, R. Gangula, F. Kaltenberger and R. Knopp, "5G NR Positioning with OpenAirInterface: Tools and Methodologies," 2025 20th Wireless On-Demand Network Systems and Services Conference (WONS), Hintertux, Austria, 2025, pp. 1-7.

[3]. Link to datasets
