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

 - touch /etc/apt/sources.list.d/oai.sources
 - cat "deb [trusted=yes] https://debian-packages-a7fe24.eurecom.io stable liboai-common liboai-usrp oai-lte oai-nr oai-physim" >> /etc/apt/sources.list.d/oai.sources
 - apt update
 - apt install liboai-common liboai-usrp oai-lte oai-nr oai-physim

 ## Install rpm packages with dnf

 - touch /etc/yum.repos.d/oai.list
 - cat <<EOF >> /etc/yum.repos.d/oai.list
    [gitlab-rpms]
    name=GitLab RPM Repo
    baseurl=https://redhat-packages-d0e1f3.eurecom.io/rpms/
    enabled=1
    gpgcheck=0
    EOF
 - dnf clean all
 - dnf makecache
 - dnf install liboai-common liboai-usrp oai-lte oai-nr oai-physim
