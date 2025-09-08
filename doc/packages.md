# Create packages

## Basic commands

    cd openairinterface/cmake_targets && \
    ./build_oai -I --install-optional-packages [ADD OPTIONS]


## Options

 - \-\-LTE-packaging => Create a package with everything for lte
 - \-\-NR-packaging => Create a package with everything for nr
 - \-\-RPM => Set the packages to be RPM ones, if not used packages will be DEB ones 
 - \-\-OAI\-LDPC\-T2\-packaging => Create a package for T2 accelerator
 - \-\-OAI\-COMMON\-packaging => Create a package with all the required shared libraries
 - \-\-OAI\-USRP\-packaging => Create a package with the libraries required for USRP
 - \-\-OAI\-PHYSIM\-packaging => Create a package with the simulators for OpenAirInterface

 ## Install debian packages with apt

```
sudo cat <<EOF | sudo tee /etc/apt/sources.list.d/oai.sources
Types: deb
URIs: https://debian-packages-a7fe24.eurecom.io
Suites: stable
Components: liboai-common liboai-usrp oai-lte oai-nr oai-physim
Trusted: yes
EOF
sudo apt update
sudo apt install liboai-common liboai-usrp oai-lte oai-nr oai-physim
```

 ## Install rpm packages with dnf

 ```
sudo cat <<EOF > /etc/yum.repos.d/oai.repo
[gitlab-rpms]
name=GitLab RPM Repo
baseurl=https://redhat-packages-d0e1f3.eurecom.io/
enabled=1
gpgcheck=0
EOF
sudo dnf clean all
sudo dnf makecache
sudo dnf install openairinterface-liboai-common-1.0.0-1.x86_64 openairinterface-liboai-usrp-1.0.0-1.x86_64 openairinterface-oai-lte-1.0.0-1.x86_64 openairinterface-oai-nr-1.0.0-1.x86_64 openairinterface-oai-physim-1.0.0-1.x86_64
```

## Systemd

Once installed, oai-nr and oai-lte have services which runs in systemd (nr-softmodem and nr-cuup for oai-nr and lte-softmodem for oai-lte), here is a list of useful commands:

- start
- stop
- restart
- reload
- status
- enable (enable service at boot)
- disable (disable service at boot)
- is-enabled

To use them:

```
sudo systemctl [command] [service]
```

By default configuration file are:
- **nr-softmodem**: gnb.sa.band78.fr1.106PRB.usrpb210.conf (check [OAI gNB with COTS UE](./NR_SA_Tutorial_COTS_UE.md))
- **nr-cuup**: gnb-cuup.sa.f1.conf
- **lte-softmodem**: enb.band7.tm1.50PRB.usrpb210.conf