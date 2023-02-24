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

  Update Linux Boot arguments 
```
BOOT_IMAGE=(hd0,gpt2)/vmlinuz-4.18.0-425.10.1.rt7.220.el8_7.x86_64 root=/dev/mapper/rhel_skylark-root ro crashkernel=auto resume=/dev/mapper/rhel_skylark-swap rd.lvm.lv=rhel_skylark/root rd.lvm.lv=rhel_skylark/swap rhgb quiet igb.max_vfs=2 intel_iommu=on iommu=pt intel_pstate=disable nosoftlockup tsc=nowatchdog mitigations=off cgroup_memory=1 cgroup_enable=memory mce=off idle=poll hugepagesz=1G hugepages=40 hugepagesz=2M hugepages=0 default_hugepagesz=1G selinux=0 enforcing=0 nmi_watchdog=0 softlockup_panic=0 audit=0 skew_tick=1 skew_tick=1 isolcpus=managed_irq,domain,0-2,8-17 intel_pstate=disable nosoftlockup tsc=reliable
```
  Use isolated CPU 0-2 for DPDK/ORAN, CPU 8 for ru_thread in our example config 

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
export XRAN_DIR=~/phy/fhi_lib
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
   cp ../../../targets/PROJECTS/GENERIC-NR-5GC/CONF/oran.conf.json .
   ```
 * Update MAC address of DU/RU and PCIe address of your setup in oran.conf.json

```
sudo ./nr-softmodem -O ../../../targets/PROJECTS/GENERIC-NR-5GC/CONF/oran.fh.band78.fr1.273PRB.conf --sa --reorder-thread-disable 
```


