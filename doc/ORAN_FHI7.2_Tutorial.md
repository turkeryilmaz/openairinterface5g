**Table of Contents**

## Prerequisites to run Platforms

* DPDK (Data Plane Development Kit)

  Download DPDK version 20.05 (https://core.dpdk.org/download/)

  Compile DPDK 

```
tar xJf dpdk-<version>.tar.xz
cd dpdk-<version>

meson build
cd build
sudo ninja
sudo ninja install

make install T=x86_64-native-linuxapp-gcc
```

* Setup Configuration 

  https://docs.o-ran-sc.org/projects/o-ran-sc-o-du-phy/en/latest/Setup-Configuration_fh.html

* PTP configuration

  https://docs.o-ran-sc.org/projects/o-ran-sc-o-du-phy/en/latest/PTP-configuration_fh.html

* Bind devices 
```
sudo modprobe vfio_pci
sudo /usr/local/bin/dpdk-devbind.py --bind vfio-pci 51:0e.0
sudo /usr/local/bin/dpdk-devbind.py --bind vfio-pci 51:0e.1
```

## ORAN Fronthaul Library 

  To get the ORAN FHI library and installation, follow the instructions

 * Download O-RAN FHI PHY library
 
```
git clone https://gerrit.o-ran-sc.org/r/o-du/phy.git
cd phy
git checkout oran_release_bronze_v1.1
```

 * Apply patches (available in oai_folder/cmake_targets/tools/oran_fhi_integration_patches): 
    
```
git apply oran-fhi-1-compile-libxran-using-gcc-and-disable-avx512.patch
git apply oran-fhi-2-return-correct-slot_id.patch
git apply oran-fhi-3-disable-pkt-validate-at-process_mbuf.patch
git apply oran-fhi-4-process_all_rx_ring.patch
git apply oran-fhi-5-remove-not-used-dependencies.patch
```

 * Set up the environment (change the path if you use different folders)
   
```
export XRAN_LIB_DIR=~/phy/fhi_lib/lib/build
export XRAN_DIR=~//phy/fhi_lib
export RTE_SDK=~/dpdk-20.05
export RTE_TARGET=x86_64-native-linuxapp-gcc
export RTE_INCLUDE=${RTE_SDK}/${RTE_TARGET}/include
```

 * Compile phy/fhi_lib:
   
```
./phy/fhi_lib/build.sh
```

## OAI-FHI Build and Compilation 

```
git clone https://gitlab.eurecom.fr/oai/openairinterface5g.git
cd openairinterface5g
git checkout develop
source oaienv
cd cmake_targets
./build_oai --gNB --ninja -t oran_fhlib_5g (Add, -I as well if it is first time to use for external dependencies)
```

## Run OAI with ORAN FHI config 

   ```
   cd ran_build/build
   cp ../../tools/oran_fhi_integration_patches/conf.json .
   ```
 * Change to MAC address of DU/RU and PCI address of your setup in conf.json

```
sudo ./nr-softmodem -O ../../../targets/PROJECTS/GENERIC-NR-5GC/CONF/oran_fh.conf --sa --reorder-thread-disable 1
```


