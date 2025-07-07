# Create packages

## Basic commands

    cd openairinterface/cmake_targets && \
    ./build_oai -I --install-optional-packages -w USRP [ADD OPTIONS]


## Options

 - \-\-UE \-\-eNB  \-\-LTE-eNB-packaging => Create packages with oairu, lte-uesoftmodem and lte-softmodem
 - \-\-nrUE \-\-NR-eNB-packaging => Create package with nr-uesoftmodem
 - \-\-gNB \-\-NR-gNB-packaging => Create packages with nr-cuup and nr-softmodem
 - \-\-RPM => Set the packages to be RPM ones, if not used packages will be DEB ones 
