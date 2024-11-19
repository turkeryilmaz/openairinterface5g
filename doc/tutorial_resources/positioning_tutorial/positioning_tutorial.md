# UL-TDoA Positioning Testing Setup

## Part 1: single gNB

### Step 1 Pull Core Network Functions
```
docker pull oaisoftwarealliance/ims:latest
docker pull oaisoftwarealliance/oai-amf:develop
docker pull oaisoftwarealliance/oai-nrf:develop
docker pull oaisoftwarealliance/oai-smf:develop
docker pull oaisoftwarealliance/oai-udr:develop
docker pull oaisoftwarealliance/oai-upf:develop
docker pull oaisoftwarealliance/oai-udm:develop
docker pull oaisoftwarealliance/oai-ausf:develop
docker pull oaisoftwarealliance/oai-lmf:develop
```

Utility image to generate traffic

`docker pull oaisoftwarealliance/trf-gen-cn5g:latest`

### Step 2 Pull RAN Repository and Bulid gNB and nrUE

Create working folder for Example OAI_RAN

```
cd OAI_RAN
git clone https://gitlab.eurecom.fr/oai/openairinterface5g.git 
cd openairinterface5g/cmake_targets
git checkout NRPPA_Procedures
./build_oai -I # install dependencies
./build_oai --gNB --nrUE -w SIMU # compile gNB and nrUE
```

### Step 3 Start the 5G core

```
cd OAI_RAN/openairinterface5g/doc/tutorial_resources/oai-cn5g
sudo docker compose -f docker-compose.yaml up -d
```

Check the status of the core if all NF are healthy 

`docker ps -a`

To stop the core

`sudo docker compose -f docker-compose.yaml down`

### Step 4 Starting gnb

Open a terminal 

```
cd OAI_RAN/openairinterface5g/cmake_targets/ran_build/build 
sudo ./nr-softmodem --rfsim --rfsimulator.serveraddr server --sa -O ~/OAI_RAN/doc/tutorial_resources/positioning_tutorial/conf/gnb1.sa.band78.106prb.rfsim4x4.conf
```

Make sure to have channelmod conf file in conf folder for rfsimulator testing

### Step 5 Starting UE 

Open a terminal 

```
cd OAI_RAN/openairinterface5g/cmake_targets/ran_build/build
sudo ./nr-uesoftmodem -r 106 --numerology 1 --band 78 -C 3619200000 --rfsim --rfsimulator.serveraddr 127.0.0.1 --sa --uicc0.imsi 001010000000001
```

TODO: create ue.conf and include channel model 

### Step 6 External API to initiate Positioning request

Send the http post request at LMF address (as per our docker compose file http://192.168.70.141:80) to LMF determine location API to initiate the Positioning procedure

```
cd OAI_RAN/doc/tutorial_resources/positioning_tutorial/conf
curl --http2-prior-knowledge -H "Content-Type: application/json" -d "@InputData.json" -X POST http://192.168.70.141:8080/nlmf-loc/v1/determine-location
```

## Part 2: multi gNB (on same host)

### Deploy 5G core

same as above

### setup IP addresses for gNBs

For the 2nd and 3rd gNB we need to add IP addresses. For convenience we add them directly to the docker network that has been created for the core network. 

```
sudo ip address add 192.168.70.142/26 dev oai-cn5g
sudo ip address add 192.168.70.143/26 dev oai-cn5g
```

### start UE

In the multi-gNB setup we need to invert the roles of client and server for the rfsimulator to make sure the gNBs are synchronized among themselves. I.e. the UE becomes the rfsimulator server and the gNBs the rfsimulator clients. 

```
 sudo ./nr-uesoftmodem -r 106 --numerology 1 --band 78 -C 3619200000 --rfsim --sa --rfsimulator.serveraddr server --uicc0.imsi 001010000000001
```

### start gNBs

Once the core is up and IPs are set we can 

```
sudo ./nr-softmodem --rfsim --rfsimulator.serveraddr 127.0.0.1 --sa -O ~/OAI_RAN/doc/tutorial_resources/positioning_tutorial/conf/gnb1.sa.band78.106prb.rfsim4x4.conf
sudo ./nr-softmodem --rfsim --rfsimulator.serveraddr 127.0.0.1 --sa -O ~/OAI_RAN/doc/tutorial_resources/positioning_tutorial/conf/gnb2.sa.band78.106prb.rfsim2x2.conf
sudo ./nr-softmodem --rfsim --rfsimulator.serveraddr 127.0.0.1 --sa -O ~/OAI_RAN/doc/tutorial_resources/positioning_tutorial/conf/gnb3.sa.band78.106prb.rfsim.conf
```

Here in this example, just to check multiple TRPs is working fine, we are running gNB 1 with 4 TRPs, gNB 2 with 2 TRPs, and gNB3 with 1 TRP. So in our measurement response, each gNB should be sending ToA as per their TRP number.


### initiate the localization procedure

```
curl --http2-prior-knowledge -H "Content-Type: application/json" -d "@InputData.json" -X POST http://192.168.70.141:8080/nlmf-loc/v1/determine-location
```

