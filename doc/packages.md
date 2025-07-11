# Create packages

## Basic commands

    cd openairinterface/cmake_targets && \
    ./build_oai -I --install-optional-packages -w USRP [ADD OPTIONS]


## Options

 - \-\-LTE-packaging => Create a package with everything for lte
 - \-\-NR-packaging => Create a package with everything for nr
 - \-\-RPM => Set the packages to be RPM ones, if not used packages will be DEB ones 
 - \-\-OAI\-LDPC\-T2\-packaging => Create a package for T2 accelerator
 - \-\-OAI\-COMMON\-packaging
 - \-\-OAI\-USRP\-packaging
 - \-\-OAI\-PHYSIM\-packaging

