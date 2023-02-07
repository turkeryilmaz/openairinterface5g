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
	 - DPDK path `/usr/local/lib64`
	 - libdpdk.pc path `/usr/local/lib64/pkgconfig`

**AMD Xilinx T2 card**
 - bitstream image and PMD driver provided by AccelerComm
 - DPDK 20.11.3 version used
	 - DPDK path `/usr/lib64/accelercomm/dpdklibs`
	 - libdpdk.pc path `/usr/lib64/accelercomm/dpdklibs/pkgconfig`

libdpdk.pc modification with correct paths to the directories.

    prefix=/usr
    libdir=${prefix}/lib64/accelercomm/dpdklibs
    includedir=/opt/accelercomm/ACL_BBDEV_DPDK20.11.3_r2p0_int_96_shared_RHEL_7.9_virtual_gcc_meson/dpdk/build/include

    Name: DPDK
    Description: The Data Plane Development Kit (DPDK).
    Note that CFLAGS might contain an -march flag higher than typical baseline.
    This is required for a number of static inline functions in the public headers.
    Version: 20.11.3
    Requires.private: libcrypto, zlib, libelf, jansson
    Libs: -L${libdir} -lrte_node -lrte_graph -lrte_bpf -lrte_flow_classify -lrte_pipeline -lrte_table -lrte_port -lrte_fib -lrte_ipsec -lrte_vhost -lrte_stack -lrte_security -lrte_sched -lrte_reorder -lrte_rib -lrte_regexdev -lrte_rawdev -lrte_pdump -lrte_power -lrte_member -lrte_lpm -lrte_latencystats -lrte_kni -lrte_jobstats -lrte_ip_frag -lrte_gso -lrte_gro -lrte_eventdev -lrte_efd -lrte_distributor -lrte_cryptodev -lrte_compressdev -lrte_cfgfile -lrte_bitratestats -lrte_bbdev -lrte_acl -lrte_timer -lrte_hash -lrte_metrics -lrte_cmdline -lrte_pci -lrte_ethdev -lrte_meter -lrte_net -lrte_mbuf -lrte_mempool -lrte_rcu -lrte_ring -lrte_eal -lrte_telemetry -lrte_kvargs
    Libs.private: -Wl,--whole-archive -L${libdir} -l:librte_common_cpt.a -l:librte_common_dpaax.a -l:librte_common_iavf.a -l:librte_common_octeontx.a -l:librte_common_octeontx2.a -l:librte_common_sfc_efx.a -l:librte_bus_dpaa.a -l:librte_bus_fslmc.a -l:librte_bus_ifpga.a -l:librte_bus_pci.a -l:librte_bus_vdev.a -l:librte_bus_vmbus.a -l:librte_common_qat.a -l:librte_mempool_bucket.a -l:librte_mempool_dpaa.a -l:librte_mempool_dpaa2.a -l:librte_mempool_octeontx.a -l:librte_mempool_octeontx2.a -l:librte_mempool_ring.a -l:librte_mempool_stack.a -l:librte_net_af_packet.a -l:librte_net_ark.a -l:librte_net_atlantic.a -l:librte_net_avp.a -l:librte_net_axgbe.a -l:librte_net_bond.a -l:librte_net_bnx2x.a -l:librte_net_bnxt.a -l:librte_net_cxgbe.a -l:librte_net_dpaa.a -l:librte_net_dpaa2.a -l:librte_net_e1000.a -l:librte_net_ena.a -l:librte_net_enetc.a -l:librte_net_enic.a -l:librte_net_failsafe.a -l:librte_net_fm10k.a -l:librte_net_i40e.a -l:librte_net_hinic.a -l:librte_net_hns3.a -l:librte_net_iavf.a -l:librte_net_ice.a -l:librte_net_igc.a -l:librte_net_ipn3ke.a -l:librte_net_ixgbe.a -l:librte_net_kni.a -l:librte_net_liquidio.a -l:librte_net_memif.a -l:librte_net_netvsc.a -l:librte_net_nfp.a -l:librte_net_null.a -l:librte_net_octeontx.a -l:librte_net_octeontx2.a -l:librte_net_pcap.a -l:librte_net_pfe.a -l:librte_net_qdma.a -l:librte_net_qede.a -l:librte_net_ring.a -l:librte_net_sfc.a -l:librte_net_softnic.a -l:librte_net_tap.a -l:librte_net_thunderx.a -l:librte_net_txgbe.a -l:librte_net_vdev_netvsc.a -l:librte_net_vhost.a -l:librte_net_virtio.a -l:librte_net_vmxnet3.a -l:librte_raw_dpaa2_cmdif.a -l:librte_raw_dpaa2_qdma.a -l:librte_raw_ifpga.a -l:librte_raw_ioat.a -l:librte_raw_ntb.a -l:librte_raw_octeontx2_dma.a -l:librte_raw_octeontx2_ep.a -l:librte_raw_skeleton.a -l:librte_crypto_bcmfs.a -l:librte_crypto_caam_jr.a -l:librte_crypto_ccp.a -l:librte_crypto_dpaa_sec.a -l:librte_crypto_dpaa2_sec.a -l:librte_crypto_nitrox.a -l:librte_crypto_null.a -l:librte_crypto_octeontx.a -l:librte_crypto_octeontx2.a -l:librte_crypto_openssl.a -l:librte_crypto_scheduler.a -l:librte_crypto_virtio.a -l:librte_compress_octeontx.a -l:librte_compress_zlib.a -l:librte_regex_octeontx2.a -l:librte_vdpa_ifc.a -l:librte_event_dlb.a -l:librte_event_dlb2.a -l:librte_event_dpaa.a -l:librte_event_dpaa2.a -l:librte_event_octeontx2.a -l:librte_event_opdl.a -l:librte_event_skeleton.a -l:librte_event_sw.a -l:librte_event_dsw.a -l:librte_event_octeontx.a -l:librte_baseband_null.a -l:librte_baseband_turbo_sw.a -l:librte_baseband_fpga_lte_fec.a -l:librte_baseband_fpga_5gnr_fec.a -l:librte_baseband_acc100.a -l:librte_baseband_accl_ldpc.a -l:librte_node.a -l:librte_graph.a -l:librte_bpf.a -l:librte_flow_classify.a -l:librte_pipeline.a -l:librte_table.a -l:librte_port.a -l:librte_fib.a -l:librte_ipsec.a -l:librte_vhost.a -l:librte_stack.a -l:librte_security.a -l:librte_sched.a -l:librte_reorder.a -l:librte_rib.a -l:librte_regexdev.a -l:librte_rawdev.a -l:librte_pdump.a -l:librte_power.a -l:librte_member.a -l:librte_lpm.a -l:librte_latencystats.a -l:librte_kni.a -l:librte_jobstats.a -l:librte_ip_frag.a -l:librte_gso.a -l:librte_gro.a -l:librte_eventdev.a -l:librte_efd.a -l:librte_distributor.a -l:librte_cryptodev.a -l:librte_compressdev.a -l:librte_cfgfile.a -l:librte_bitratestats.a -l:librte_bbdev.a -l:librte_acl.a -l:librte_timer.a -l:librte_hash.a -l:librte_metrics.a -l:librte_cmdline.a -l:librte_pci.a -l:librte_ethdev.a -l:librte_meter.a -l:librte_net.a -l:librte_mbuf.a -l:librte_mempool.a -l:librte_rcu.a -l:librte_ring.a -l:librte_eal.a -l:librte_telemetry.a -l:librte_kvargs.a -Wl,--no-whole-archive -Wl,--export-dynamic -lpcap -pthread -lm -ldl -lnuma -lfdt -lpcap /lib/../lib64/librt.so
    Cflags: -I${includedir} -I${includedir} -include rte_config.h -march=broadwell

*Note: Setup with T1 and T2 cards was deployed on RHEL 7.9 with installation files provided by VVDN/AccelerComm. Modification of the libdpdk.pc file could be required in order to include correct path to DPDK/driver libraries.*
## DPDK setup

 - check the presence of the card on the host
	 - `lspci | grep "Xilinx"`
 - binding of the device with igb_uio driver
	 - `./dpdk-devbind.py -b igb_uio <address of the PCI of the card>`
 - hugepages setup (32 x 1GB hugepages)
	 - `./dpdk-hugepages.py -p 1G --setup 32G`

*Note: Commands to run from dpdk/usertool folder*

## Compilation
Deployment of the OAI is precisely described in the following tutorials.
[BUILD](https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/doc/BUILD.md)
[NR_SA_CN5G_gNB_USRP_COTS_UE_Tutorial](https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/doc/NR_SA_CN5G_gNB_USRP_COTS_UE_Tutorial.md)

Shared object files *libldpc_offload_T1.so* or/and *libldpc_offload_T2.so* are created  during the compilation. These objects are conditionally compiled based on the boolean variable *BUILD_T1_OFFLOAD* for T1 card and *BUILD_T2_OFFLOAD* for T2 card.  Selection of the library to compile can be done directly in *CMakeList.txt* in section LDPC offload library or from the command line adding following option: `--cmake-opt -DBUILD_T1_OFFLOAD=True` or `--cmake-opt -DBUILD_T2_OFFLOAD=True`.

## nr_ulsim
Offload of the channel decoding is in nr_ulsim specified by *-o* option followed by the version of the card *T1* or *T2*. Example command for running nr_ulsim with offload to the T1 card:

    ./nr_ulsim -n100 -s20 -m20 -r273 -o T1
## nr-softmodem
Offload of the channel decoding in nr-softmodem is specified by *--ldpc-offload-enable 1* option, library for operating the card is selected by *--loader.ldpc_offload.shlibversion* followed by the version of the card *_T1* or *_T2*. Exemple command for running nr-ulsim with offload to the T1 card:

    sudo numactl --cpunodebind=1 --membind=1 ./nr-softmodem -O .config_file.conf --sa --ldpc-offload-enable 1 --loader.ldpc_offload.shlibversion _T1
