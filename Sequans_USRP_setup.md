# Sequans USRP N310 Setup

This document describes how to connect the USRP N310 device and build the software and run gNB.

## 1 N310 Connection Options

The N310 comes with 4 ports that can be used for connection to the PC.
Note that some ports can be used for device management only while others for data streaming only.

* USB Console port implements 2 serial interfaces that can be used for device management only:
    1. The Linux console of the ARM CPU.
    2. The console of the STM32. For example, it can be used to reboot the device remotely.

    The baud rate is 115200 for both interfaces. This example shows how to connect to the Linux console (`if00` interface):

        $ ls /dev/serial/by-id
        usb-Silicon_Labs_CP2105_Dual_USB_to_UART_Bridge_Controller_007F6CB5-if00-port0
        usb-Silicon_Labs_CP2105_Dual_USB_to_UART_Bridge_Controller_007F6CB5-if01-port0

        $ sudo screen /dev/serial/by-id/usb-Silicon_Labs_CP2105_Dual_USB_to_UART_Bridge_Controller_007F6CB5-if00-port0 115200

* Ethernet port provides access to the Linux console via SSH. Can be used as a management port only.
  User `root`, no password.
* SFP0 can be used as a data port. The port speed depends on the variant of the FPGA firmware 
  (1Gb for the HG variant or 10Gb for the XG).
* SFP1 is a 10 Gb port and can be used as a data port.

Default network configuration:

| Interface | Configuration | Configuration File           | Configuration File (old image ver.) |
|-----------|---------------|------------------------------|-------------------------------------|
| eth0      | dhcp          | `/data/network/eth0.network` | `/etc/systemd/network/eth0.network` |
| sfp0      | 192.168.10.2  | `/data/network/sfp0.network` | `/etc/systemd/network/sfp0.network` |
| sfp1      | 192.168.20.2  | `/data/network/sfp1.network` | `/etc/systemd/network/sfp1.network` |

## 2 Connect N310

In current setup we use 10Gb SFP1 port that is connected to the `enp1s0f0` 10Gb interface of the lab PC.

1. Insert the 10Gb copper adapter to the SFP1 port and connect the port to the PC via Cat6e ethernet cable.
2. Configure the static IP address on the interface connected to the N310:

        sudo ip addr add 192.168.20.1/24 dev enp1s0f0

2. Try to ping N310:

        ping 192.168.20.2


## 3 Build and Install UHD Driver

UHD is a USRP host driver. We'll need its apps to update N310 firmware and libraries to build gNB.

1. Install required packages.

        sudo apt install -y libboost-all-dev libusb-1.0-0-dev doxygen python3-docutils python3-mako python3-numpy python3-requests python3-ruamel.yaml python3-setuptools cmake build-essential

2. Download sources.

        git clone https://github.com/EttusResearch/uhd.git ~/uhd -b v4.3.0.0

    Current version is 4.3.0.0.

3. Build and install UHD.

        cd ~/uhd/host
        mkdir build
        cd build
        cmake ../
        make -j

        sudo make install
        sudo ldconfig

4. Check if UHD can find N310.

        $ uhd_find_devices 
        [INFO] [UHD] linux; GNU C++ version 9.4.0; Boost_107100; UHD_4.3.0.HEAD-0-g1f8fd345
        --------------------------------------------------
        -- UHD Device 0
        --------------------------------------------------
        Device Address:
            serial: 327B212
            addr: 192.168.10.2
            claimed: False
            mgmt_addr: 192.168.10.2
            product: n310
            type: n3xx

## 4 Update N310 Image.

It's important to update the USRP image so that it matches the UHD driver version.
The image contains the embedded Linux file system.
The easiest way is to update the file system via Mender.

1. Download image.
    The `uhd_images_downloader` will download the version that matches the currently installed UHD.

        sudo uhd_images_downloader -t mender -t n3xx --yes -v

    If the command fails, the image can be downloaded manually.
    Check the `uhd_images_downloader` output and find the following lines:

        [INFO] Using base URL: https://files.ettus.com/binaries/cache/
        ...
        [DEBUG] URLs to download:
        n3xx/meta-ettus-v4.3.0.0/n3xx_common_mender_default-v4.3.0.0.zip

    Combine those URLs:

        wget https://files.ettus.com/binaries/cache/n3xx/meta-ettus-v4.3.0.0/n3xx_common_mender_default-v4.3.0.0.zip
        unzip n3xx_common_mender_default-v4.3.0.0.zip

2. Copy image to N310.
    Images downloaded with `uhd_images_downloader` are located in `/usr/local/share/uhd/images/`.

        scp n3xx_common_mender_default-v4.3.0.0.zip root@192.168.10.2:/home/root

3. SSH into N310 and run the installation procedure.

        ssh root@192.168.10.2

    Note, command examples that must be executed on N310 contain `root@ni-n3xx-327B212:~#` prompt.

    The installation is tricky because it depends on what Mender version is currently installed on N310.
    First, try modern version:

        root@ni-n3xx-327B212:~# mender install /home/root/usrp_n3xx_fs.mender
    
    If it didn't work, try to run Mender using old-style command line arguments:

        root@ni-n3xx-327B212:~# mender -rootfs /home/root/usrp_n3xx_fs.mender -f

4. Reboot and Validate the Installation.

        root@ni-n3xx-327B212:~# reboot

    After device booted, check the version:

        root@ni-n3xx-327B212:~# uhd_config_info --version
        root@ni-n3xx-327B212:~# cat /etc/mender/artifact_info

    Mender uses A+B partitioning scheme, where one partition is active and another one is used for update.
    After upgrade, the next boot runs from the newly-updated partition once.
    However, we need to let Mender know that upgraded FS work fine. Otherwise, it will fall back to
    the previous version after reboot:

        root@ni-n3xx-327B212:~# mender -commit

## 5 Update FPGA firmware

1. Download FPGA firmware.

        sudo uhd_images_downloader -t n310_fpga -v
    
    If download failes, use the same workaround as for image downloading and unzip files to UHD inventory location.

        wget https://files.ettus.com/binaries/cache/n3xx/uhd-dd757ac/n3xx_n310_fpga_default-gdd757ac.zip
        sudo unzip n3xx_n310_fpga_default-gdd757ac.zip -d /usr/local/share/uhd/images

2. Update FPGA firmware

        uhd_image_loader --args type=n3xx,addr=ni-n3xx-<Device_serial_addr>

    where `<Device_serial_addr>` is the serial address of the USRP, which can be obtained by running `uhd_find_devices`.

## 6 Build gNB with USRP support

1. Install `lttng`.

        sudo apt-add-repository ppa:lttng/stable-2.13
        sudo apt update
        sudo apt install -y lttng-tools lttng-modules-dkms  liblttng-ust-dev

2. Obtain sources and install required packages.

        git clone https://gitlab-shared.sequans.com/sequans/system/ttcn/firecell_sequansrd -b Latest_3GPP_FRD_299
        cd firecell_sequansrd
        ./FirecellRD/components/RAN/cmake_targets/build_oai -I

    If build fails while installing ASN1 because `gitlab.eurecom.fr` is inaccessible,
    configure git to use gitlab-shared.sequans.com instead:

        git config --global url.https://gitlab-shared.sequans.com/sequans/system/ttcn/asn1c.insteadof https://gitlab.eurecom.fr/oai/asn1c.git

3. Build gNB.

        cd FirecellRD/components/RAN
        . oaienv
        cd cmake_targets
        ./build_oai -I
        ./build_oai -w USRP --ninja --nrUE --gNB --build-lib all -c

## 7 Run gNB

1. Run gNB (from the OAI dir):

        ./cmake_targets/ran_build/build/nr-softmodem -O ci-scripts/conf_files/gNB.N78.fr1.40MHz.N310.conf --sa --usrp-tx-thread-config 1 --log_config.global_log_options wall_clock

2. *Opional step, for Fibocom UE only*. Connect to the AT port:
   Add `udev` rule for the Fibocom modem to prevent the ModemManager to interfere with the AT console.
   Create the `/etc/udev/rules.d/99-fibocom.rules` file with following content:

        ATTRS{idVendor}=="0403" ATTRS{idProduct}=="6010", ENV{ID_MM_DEVICE_IGNORE}="1"

   Reload the rules:

        sudo udevadm control --reload-rules

   If the above method doesn't help, try to stop `ModemManager` or even to disable it:

        sudo systemctl stop ModemManager
        sudo systemctl disable ModemManager

   Connect to the AT console:

        sudo minicom -D /dev/serial/by-id/usb-Fibocom_Fibocom_FM160_Modem_SN:5033FA31_5033fa31-if02-port0

   Check that band n78 is enabled

        at+gtact?
        +GTACT: 14,,,5078

        OK

   in the response we have:
   - 14: enabled RATs, NR only.
   - 5078: NR band n78

   Set `cfun=1`:

        at+cfun=1
        OK

        +SIM: Inserted

        +C5GREG: 2

        +CEREG: 2

        +SIM READY

   Check that 5G registration status changed from 0 to 2 (it may take up to half a minute):
        
        at+c5greg?
        +C5GREG: 2,2

        OK

3. Check gNB log, it should contain the `UE State = NR_RRC_CONNECTED` message (at the end of the example below).

        1692024360.127029 [NR_MAC]   Frame.Slot 128.0
        UE RNTI c723 (1) PH 49 dB PCMAX 21 dBm, average RSRP 0 (0 meas)
        UE c723: CQI 0, RI 1, PMI (0,0)
        UE c723: dlsch_rounds 178/27/27/27, dlsch_errors 27, pucch0_DTX 108, BLER 0.93539 MCS 6
        UE c723: dlsch_total_bytes 20734
        UE c723: ulsch_rounds 177/25/25/25, ulsch_DTX 100, ulsch_errors 25, BLER 0.92023 MCS 6
        UE c723: ulsch_total_bytes_scheduled 19373, ulsch_total_bytes_received 17356
        UE c723: LCID 1: 3 bytes TX

        1692024360.749749 [NR_PHY]   [gNB 0][RAPROC] Frame 189, slot 19 Initiating RA procedure with preamble 19, energy 46.6 dB (I0 78, thres 120), delay 7 start symbol 4 freq index 0
        1692024360.749807 [MAC]   UL_info[Frame 189, Slot 19] Calling initiate_ra_proc RACH:SFN/SLOT:189/19
        1692024360.749814 [NR_MAC]   Search for not existing rnti (ignore for RA): 37d2
        1692024360.749816 [NR_MAC]   [gNB 0][RAPROC] CC_id 0 Frame 189 Activating Msg2 generation in frame 190, slot 7 using RA rnti 10f SSB, new rnti 37d2 index 0 RA index 0
        1692024360.750433 [NR_MAC]   [gNB 0][RAPROC] CC_id 0 Frame 190, slotP 7: Generating RA-Msg2 DCI, rnti 0x10f, state 1, CoreSetType 2
        1692024360.750449 [NR_MAC]   [RAPROC] Msg3 slot 17: current slot 7 Msg3 frame 190 k2 7 Msg3_tda_id 3
        1692024360.750456 [NR_MAC]   [gNB 0][RAPROC] Frame 190, Subframe 7: rnti 37d2 RA state 2
        1692024360.758679 [NR_MAC]   Search for not existing rnti (ignore for RA): 37d2
        1692024360.758684 [NR_MAC]   Adding UE with rnti 0x37d2
        1692024360.759265 [NR_MAC]   [gNB 0][RAPROC] PUSCH with TC_RNTI 0x37d2 received correctly, adding UE MAC Context RNTI 0x37d2
        1692024360.759271 [NR_MAC]   [RAPROC] RA-Msg3 received (sdu_lenP 7)
        1692024360.759275 [RLC]   activated srb0 for UE with RNTI 0x37d2
        1692024360.759279 [MAC]   [RAPROC] Received SDU for CCCH length 6 for UE 37d2
        1692024360.759344 [NR_MAC]   Activating scheduling RA-Msg4 for TC_RNTI 0x37d2 (state 2)
        1692024360.759346 [NR_MAC]   Unexpected ULSCH HARQ PID 0 (have -1) for RNTI 0x37d2 (ignore this warning for RA)
        1692024360.759353 [RRC]   initial UL RRC message nr_cellid 0 does not match RRC's 12345678
        1692024360.759379 [NR_RRC]   Decoding CCCH: RNTI 37d2, inst 0, payload_size 6
        1692024360.759655 [NR_RRC]   rrc_gNB_generate_RRCSetup for RNTI 37d2
        1692024360.759700 [PDCP]   nr_pdcp_entity_set_security: rb_id=1 INT=0, CIP=0
        1692024360.759737 [NR_MAC]   Adding SchedulingRequestconfig
        1692024360.759739 [NR_MAC]   Adding BSR config
        1692024360.759741 [NR_MAC]   Adding TAG config
        1692024360.759741 [NR_MAC]   Adding PHR config
        1692024360.759757 [RLC]   /home/labo/dev/lmeyer/ttcn/openairinterface5g/openair2/LAYER2/nr_rlc/nr_rlc_oai_api.c:961:add_rlc_srb: added srb 1 to UE with RNTI 0x37d2
        1692024360.759986 [NR_MAC]   ( 191. 6) SRB0 has 136 bytes
        1692024360.760019 [NR_MAC]   Generate msg4, rnti: 37d2
        1692024360.760022 [NR_MAC]   Encoded RRCSetup Piggyback (136 + 2 bytes), mac_pdu_length 145
        1692024360.764081 [NR_MAC]   (UE RNTI 0x37d2) Received Ack of RA-Msg4. CBRA procedure succeeded!
        1692024360.764089 [NR_MAC]   (191.14) Activating RRC processing timer for UE 37d2 with 10 ms
        1692024360.774173 [NR_MAC]   (192.14) De-activating RRC processing timer for UE 37d2
        1692024360.774260 [NR_MAC]   Modified rnti 37d2 with CellGroup
        1692024360.774263 [NR_MAC]   Adding SchedulingRequestconfig
        1692024360.774264 [NR_MAC]   Adding BSR config
        1692024360.774268 [NR_MAC]   Adding TAG config
        1692024360.774269 [NR_MAC]   Adding PHR config
        1692024360.918633 [PDCP]   nr_pdcp_entity_recv_pdu: Entity security status(0): ciphering -1, integrity check -1
        1692024360.918692 [PDCP]   RLC => PDCP: rcvd_count=0, rcvd_sn=0: 00 00 10 00 05 df 80 10 5e 40 03 40 40 be 0a 7c 3f c0 00 04 00 00 04 65 cb 80 bc 1c 00 00 00 00 00 
        1692024360.918698 [PDCP]   nr_pdcp_entity_recv_pdu: deciphering did not apply
        1692024360.918813 [NR_RRC]   [FRAME 00000][gNB][MOD 00][RNTI 37d2] RLC RB 01 --- RLC_DATA_IND 27 bytes (RRCSetupComplete) ---> RRC_gNB
        1692024360.918821 [NR_RRC]   [FRAME 00000][gNB][MOD 00][RNTI 37d2] [RAPROC] Logical Channel UL-DCCH, processing NR_RRCSetupComplete from UE (SRB1 Active)
        1692024360.918826 [NR_RRC]   [FRAME 00000][gNB][MOD 00][RNTI 37d2] UE State = NR_RRC_CONNECTED 
        1692024360.918833 [NGAP]   No AMF is associated to the gNB

