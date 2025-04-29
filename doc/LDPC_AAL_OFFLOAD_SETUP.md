<table style="border-collapse: collapse; border: none;">
  <tr style="border-collapse: collapse; border: none;">
    <td style="border-collapse: collapse; border: none;">
      <a href="http://www.openairinterface.org/">
         <img src="./images/oai_final_logo.png" alt="" border=3 height=50 width=150>
         </img>
      </a>
    </td>
    <td style="border-collapse: collapse; border: none; vertical-align: center;">
      <b><font size = "5">OAI LDPC offload (O-RAN AAL)</font></b>
    </td>
  </tr>
</table>

**Table of Contents**

[[_TOC_]]

This documentation describes the integration of LDPC coding for lookaside acceleration using O-RAN AAL (i.e., DPDK BBDEV) in OAI, along with its usage.

# Requirements

In principle, any lookaside LDPC accelerator supporting the O-RAN AAL/ DPDK BBDEV should work.

## Tested Devices/ Environments

### Xilinx T2

- TODO. To be validated.

### Intel ACC100

- TODO. To be validated.

### Intel ACC200 (also known as VRB1)
- DPDK22.11.7 on Ubuntu 24.04.
- DPDK23.11.3 on Ubuntu 24.04.

# System Setup
## DPDK installation

> Note: If you are using the Xilinx T2 card, you will need to apply the corresponding patches before compiling DPDK. 

Refer to the guide [here](https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/doc/ORAN_FHI7.2_Tutorial.md?ref_type=heads#dpdk-data-plane-development-kit) to install, and then validate your DPDK installation.

## System configuration

### Setting up Hugepages

First, we must setup hugepages on the system.
In our setup, we setup 16 of the 1G hugepages.
Apart from 1G, 2MB hugepages works too, but make sure to allocate a sufficient number of them.

```
# sudo dpdk-hugepages.py -p 1G --setup 16G
```

### Locating the Accelerator

Next, we check whether our system can detect our accelerator using `dpdk-devbind.py`.
You should see Baseband devices detected by DPDK, as follows:
```
# sudo dpdk-devbind.py -s
...
Baseband devices using DPDK-compatible driver
=============================================
0000:f7:00.0 'Device 57c0' unused=vfio-pci
...
```

As you can see here, our Intel ACC200 has the address of `0000:f7:00.0`.
Depending on the accelerator you are using, the address may vary.

### Loading VFIO-PCI and enabling SR-IOV
Following, make sure to load the `vfio-pci` kernel modules and ensure that SR-IOV is enabled.

```
# sudo modprobe vfio-pci enable_sriov=1 disable_idle_d3=1
```

### Binding the Accelerator with `vfio-pci`

Lastly, we bind our accelerator with the `vfio-pci` driver.
```
# sudo dpdk-devbind.py --bind=vfio-pci 0000:f7:00.0
```

> Note: For the Xilinx T2, we can use this device directly.
If you use an Intel vRAN accelerator, read on.

### Additional Steps for Intel vRAN Accelerators

> IMPORTANT NOTE: 
> - Currently, we only support using the Virtual Functions (VFs) of the Intel vRAN accelerators, but not the Physical Function (PF). 
> - One key advantage of using VFs is that this allows us to share the accelerator with other DU instances one the same machine, which is common in practice.

If you are using an Intel vRAN accelerator, you will need to use the the [pf_bb_config](https://github.com/intel/pf-bb-config) tool to configure the accelerator beforehand. 

#### pf_bb_config
For more details, please consult the `pf_bb_config` README.

```
# git clone https://github.com/intel/pf-bb-config
# cd ~/pf-bb-config
# ./build.sh
```
This clones and builds the `pf_bb_config` binary.

Next, we show an example for the Intel ACC200.
We use an existing configuration located at `./vrb1/vrb1_config_16vf.cfg`.

Here, it is necessary to specific a VFIO token (in this case, we use the UUID `00112233-4455-6677-8899-aabbccddeeff`).
Note that in practice, a random UUID should be used.
```
# sudo ./pf_bb_config VRB1 -v 00112233-4455-6677-8899-aabbccddeeff -c vrb1/vrb1_config_16vf.cfg
== pf_bb_config Version v25.01-0-g812e032 ==
VRB1 PF [0000:f7:00.0] configuration complete!
Log file = /var/log/pf_bb_cfg_0000:f7:00.0.log
```

#### Creating VFs

Finally, we create the VF(s) for our accelerator. 
In this example, we only create one SR-IOV VF.
```
# echo 1 | sudo tee /sys/bus/pci/devices/0000:f7:00.0/sriov_numvfs
```

If you encounter any errors when creating the VF(s), e.g., `tee: '/sys/bus/pci/devices/0000:f7:00.0/sriov_numvfs': No such file or directory`, try enabling SR-IOV again.
```
# echo 1 | sudo tee /sys/module/vfio_pci/parameters/enable_sriov
```

After you have successfully created the VF, you should see an additional baseband device, in our case, it is `0000:f7:00.1`. 
We will then use this device with OAI later.
```
# sudo dpdk-devbind.py -s
...
Baseband devices using DPDK-compatible driver
=============================================
0000:f7:00.0 'Device 57c0' drv= unused=vfio-pci
0000:f7:00.1 'Device 57c0' drv= unused=vfio-pci
...
```

# Building OAI with ORAN-AAL
OTA deployment is precisely described in the following tutorial:
- [NR_SA_Tutorial_COTS_UE](https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/doc/NR_SA_Tutorial_COTS_UE.md)
Instead of section *3.2 Build OAI gNB* from the tutorial, run following commands:

```
# Get openairinterface5g source code
git clone https://gitlab.eurecom.fr/oai/openairinterface5g.git ~/openairinterface5g
cd ~/openairinterface5g
git checkout develop

# Install OAI dependencies
cd ~/openairinterface5g/cmake_targets
./build_oai -I

# Build OAI gNB
cd ~/openairinterface5g
source oaienv
cd cmake_targets
./build_oai -w USRP --ninja --gNB -P --build-lib "ldpc_aal" -C
```

A shared object file *libldpc_aal.so* will created during the compilation. 
This object is conditionally compiled. 
The selection of the library to compile is done using *--build-lib ldpc_aal*.

> Note: The required DPDK poll mode driver has to be present on the host machine and required DPDK version has to be installed on the host, prior to building OAI.

# O-RAN AAL DPDK EAL parameters
To configure O-RAN AAL-related DPDK Environment Abstraction Layer (EAL) parameters, you can set the following parameters via the command line of PHY simulators or softmodem:

- `nrLDPC_coding_aal.dpdk_dev` - **mandatory** parameter, this specifies PCI address of our accelerator. It must follow the format `0000:XX:YY.Z`.

- `nrLDPC_coding_aal.dpdk_core_list` - **mandatory** parameter, specifies CPU cores assigned to DPDK . 
Ensure that the CPU cores specified in *nrLDPC_coding_aal.dpdk_core_list* are available and not used by other processes to avoid conflicts.

- `nrLDPC_coding_aal.dpdk_prefix` - optional parameter, DPDK shared data file prefix, by default set to *b6*.

- `nrLDPC_coding_aal.vfio_vf_token` - optional parameter, VFIO token set for the VF, if applicable.

**Note:** These parameters can also be provided in a configuration file:
```
nrLDPC_coding_aal : {
  dpdk_dev : "0000:f7:00.1";
  dpdk_core_list : "14-15";
  vfio_vf_token: "00112233-4455-6677-8899-aabbccddeeff";
};

loader : {
  ldpc : {
    shlibversion : "_aal";
  };
};
```

# Running OAI with O-RAN AAL

## 5G PHY simulators

### nr_ulsim
Offload of the channel decoding to the LDPC accelerator is in nr_ulsim specified by *--loader.ldpc.shlibversion _aal* option. 

Example command:
```
cd ~/openairinterface5g
source oaienv
cd cmake_targets/ran_build/build
sudo ./nr_ulsim -n100 -s20 -m20 -r273 -R273 --loader.ldpc.shlibversion _aal --nrLDPC_coding_aal.dpdk_dev 0000:f7:00.1 --nrLDPC_coding_aal.dpdk_core_list 0-1 --nrLDPC_coding_aal.vfio_vf_token 00112233-4455-6677-8899-aabbccddeeff
```
### nr_dlsim
Offload of the channel encoding to the LDPC accelerator is in nr_dlsim specified by *-c* option.

Example command:
```
cd ~/openairinterface5g
source oaienv
cd cmake_targets/ran_build/build
sudo ./nr_dlsim -n300 -s30 -R 106 -e 27 --loader.ldpc.shlibversion _aal --nrLDPC_coding_aal.dpdk_dev 0000:f7:00.1 --nrLDPC_coding_aal.dpdk_core_list 0-1 --nrLDPC_coding_aal.vfio_vf_token 00112233-4455-6677-8899-aabbccddeeff
```

## OTA test
Offload of the LDPC channel encoding and decoding to the LDPC accelerator is enabled by *--loader.ldpc.shlibversion _aal* option.

### Running OAI gNB with USRP B210

Example command:
```
# cd ~/openairinterface5g
# source oaienv
# cd cmake_targets/ran_build/build
# sudo ./nr-softmodem -O ../../../targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf --loader.ldpc.shlibversion _aal --nrLDPC_coding_aal.dpdk_dev 0000:f7:00.1 --nrLDPC_coding_aal.dpdk_core_list 0-1 --nrLDPC_coding_aal.vfio_vf_token 00112233-4455-6677-8899-aabbccddeeff
```

# Known Limitations

TODO. To be validated.
