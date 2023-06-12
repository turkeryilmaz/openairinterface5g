<table style="border-collapse: collapse; border: none;">
  <tr style="border-collapse: collapse; border: none;">
    <td style="border-collapse: collapse; border: none;">
      <a href="http://www.openairinterface.org/">
         <img src="./images/oai_final_logo.png" alt="" border=3 height=50 width=150>
         </img>
      </a>
    </td>
    <td style="border-collapse: collapse; border: none; vertical-align: center;">
      <b><font size = "5">OAI T1/T2 LDPC offload</font></b>
    </td>
  </tr>
</table>

**Table of Contents**

[[_TOC_]]

This documentation aims to provide a tutorial for AMD Xilinx T1 and T2 Telco cards integration into OAI and its usage.
## Prerequisites
Offload of the channel decoding  was tested with T1 and T2 card in following setup:

**AMD Xilinx T1 Telco card**

 - bitstream image and drivers provided by VVDN
 - DPDK 20.05 version used
 - tested on RHEL7.9, RHEL9.1

**AMD Xilinx T2 card**
 - bitstream image and PMD driver provided by AccelerComm
 - DPDK 20.11.3 version used
 - tested on RHEL7.9, RHEL9.1, Ubuntu 20.04


## DPDK setup

 - check the presence of the card on the host
	 - `lspci | grep "Xilinx"`
 - binding of the device with igb_uio driver
	 - `./dpdk-devbind.py -b igb_uio <address of the PCI of the card>`
 - hugepages setup (10 x 1GB hugepages)
	 - `./dpdk-hugepages.py -p 1G --setup 10G`

*Note: Commands to run from dpdk/usertool folder*

## Compilation
Deployment of the OAI is precisely described in the following tutorials.
 - [BUILD](https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/doc/BUILD.md)
 - [NR_SA_CN5G_gNB_USRP_COTS_UE_Tutorial](https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/doc/NR_SA_Tutorial_COTS_UE.md)

Shared object files *libldpc_offload_t1.so* or/and *libldpc_offload_t2.so* are created  during the compilation. These objects are conditionally compiled. Selection of the library to compile is done using *--build-lib ldpc_offload_t1 ldpc_offload_t2*. Sample command to build OAI with support for LDPC offload to T1 is:

`./build_oai -P --gNB -w USRP --build-lib ldpc_offload_t1 --ninja`

*Required DPDK version  has to be installed on the host, prior to the build of OAI*
## nr_ulsim
Offload of the channel decoding is in nr_ulsim specified by *-o* option followed by the version of the card *t1* or *t2*. Example command for running nr_ulsim with offload to the T1 card:

`./nr_ulsim -n100 -s20 -m20 -r273 -o t1`

## nr-softmodem
Offload of the channel decoding in nr-softmodem is specified by *--ldpc-offload-enable 1* option, library for the LDPC decoder is selected by *--loader.ldpc_offload.shlibversion* followed by the version of the card *_t1* or *_t2*. Sample command for running nr_ulsim with offload to the T1 card:

`sudo numactl --cpunodebind=1 --membind=1 ./nr-softmodem -O config_file.conf --sa --ldpc-offload-enable 1 --loader.ldpc_offload.shlibversion _t1`
