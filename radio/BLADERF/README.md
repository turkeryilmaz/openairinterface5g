# BladeRF 2.0 Micro Documentation

---

[TOC]

---

As of **2025**, this documentation provides instructions for setting up and using the **BladeRF 2.0 Micro** with OpenAirInterface (OAI). If you encounter any discrepancies, contact the support team or check the latest resources on Nuand's GitHub.

---

### **1. Install BladeRF 2.0 Micro Libraries**

   - Install the latest **bladeRF 2.0 Micro** libraries and utilities.  
     It's recommended to build from source for the most recent updates.

      Follow this link :point_right: to find the installation procedure based on your Linux package Manager.

      #### Debian

      ##### Install the required packages

      ```bash
      sudo apt update
      sudo apt install -y build-essential cmake git pkg-config libusb-1.0-0-dev libtecla-dev \
                                                               libncurses5-dev libncursesw5-dev libcurl4-openssl-dev
      ```   

      ##### [Building bladeRF libraries and tools from source](https://github.com/Nuand/bladeRF/wiki/Getting-Started%3A-Linux#building-bladerf-libraries-and-tools-from-source)

      ###### a). Clone the **bladeRF** source code

      ```bash
      git clone https://github.com/Nuand/bladeRF.git && cd bladeRF
      ```

      ###### b). Compile and Install the binaries

      ```bash
      mkdir build
      cd build
      cmake ..
      make -j$(nproc)
      sudo make install
      sudo ldconfig
      ```
      #### c). **Reload udev rules**

      ```bash
      sudo udevadm control --reload-rules
      sudo udevadm trigger
      ```

      * Log out and log back in
---

### :flashlight:  **2. Flashing the FPGA Image**

#### :abacus: [Configuration File](https://nuand.com/libbladeRF-doc/v2.5.0/configfile.html)

This configuration will help the bladeRF board to load the bitstream images automatically.

- create a directory

```sh
mkdir -p ~/.config/Nuand/bladeRF
```

- Create the `bladeRF` config file:

```sh
touch `~/.config/Nuand/bladeRF/bladerf.conf
```

:speaking_head: Adjust to your `HOME` folder,  i.e. `/Users/me`

```
vi ~/.config/Nuand/bladeRF/bladerf.conf 
```
> Returns
```powershell
# Load the hostedx40.rbf FPGA image, set the trim DAC, and
# configure the frequency
fpga /Users/me/.config/Nuand/bladeRF/v0.16.0/hostedxA4.rbf
# trimdac 592
# frequency 2.4G
```

#### :roll_of_paper: Download the board images

- [ ] Pick a bladeRF version where to install the FPGA bitstream

```sh
export BLADERF_VERSION=v0.16.0
```

```sh
mkdir -p ~/.config/Nuand/bladeRF/${BLADERF_VERSION} && cd ~/.config/Nuand/bladeRF/${BLADERF_VERSION}
```


To download the FPGA bitstream file for the bladeRF 2.0 micro xA4 using curl, execute the following command in your terminal:

```sh
curl -O https://www.nuand.com/fpga/${BLADERF_VERSION}/hostedxA4.rbf
```

This command retrieves the file hostedxA4-latest.rbf from the specified URL and saves it in your current directory. The -O option instructs curl to save the file with its original name.

After downloading, you can load the FPGA image onto your bladeRF device using the bladeRF-cli tool:

```
bladeRF-cli --load-fpga hostedxA4-latest.rbf
```
> Returns
```powershell
Loading FPGA...
[INFO @ /privatehost/libraries/libbladeRF/src/helpers/version.c:106] FPGA version (v0.15.3) is newer than entries in libbladeRF's compatibility table. Please update libbladeRF if problems arise.
Successfully loaded FPGA bitstream!
```

This command loads the FPGA image into the bladeRF device, preparing it for operation. ￼

---


#### &#x1F4E6; [Upgrading FX3 Firmware](https://github.com/Nuand/bladeRF/wiki/Upgrading-bladeRF-FX3-Firmware)

[&#x1F4DC; FX3 images](https://www.nuand.com/fx3_images)

- [ ] Pick a firmware version where to install the upgrade

```sh
export BLADERF_FX3_VERSION=v2.6.0
```

```sh
mkdir -p ~/.config/Nuand/bladeRF/fx3/${BLADERF_FX3_VERSION} && cd ~/.config/Nuand/bladeRF/fx3/${BLADERF_FX3_VERSION}
```


To download the firmware file for the bladeRF 2.0 micro xA4 using curl, execute the following command in your terminal:

```sh
curl -O https://www.nuand.com/fx3/bladeRF_fw_${BLADERF_FX3_VERSION}.img
```
> Returns
```powershell
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  120k    0  120k    0     0   372k      0 --:--:-- --:--:-- --:--:--  372k
```

This command retrieves the file bladeRF_fw_latest.img from the specified URL and saves it in your current directory. The -O option instructs curl to save the file with its original name.

After downloading, you can load the firmware onto your bladeRF device using the bladeRF-cli tool:

```sh
sudo bladeRF-cli -f bladeRF_fw_${BLADERF_FX3_VERSION}.img
```
This command loads the firmware image into the bladeRF device, preparing it for operation. ￼

#### :warning: Reboot the server

- [ ] after rebooting check the version:

```sh
bladeRF-cli --exec version
```
>
```powershell

  bladeRF-cli version:        1.10.0-git-7d7d87f6
  libbladeRF version:         2.6.0-git-7d7d87f6

  Firmware version:           2.6.0-git-09c82087
  FPGA version:               0.16.0 (configured by USB host)
```

- [ ] after rebooting get the device info:

```bash
bladeRF-cli --exec info
```
>
```powershell
[WARNING @ host/libraries/libbladeRF/src/board/bladerf2/bladerf2.c:486] FPGA bitstream file not found.
[WARNING @ host/libraries/libbladeRF/src/board/bladerf2/bladerf2.c:487] Skipping further initialization...

  Board:                    Nuand bladeRF 2.0 (bladerf2)
  Serial #:                 b1f3da6672d54d37a2cd7c4d3e3c2e25
  VCTCXO DAC calibration:   0x1f91
  FPGA size:                49 KLE
  FPGA loaded:              yes
  Flash size:               32 Mbit
  USB bus:                  2
  USB address:              2
  USB speed:                SuperSpeed
  Backend:                  libusb
  Instance:                 0
```

```bash
bladeRF-cli --probe
```
>
```powershell

  Description:    Nuand bladeRF 2.0
  Backend:        libusb
  Serial:         b1f3da6672d54d37a2cd7c4d3e3c2e25
  USB Bus:        2
  USB Address:    2
```
---

### **3. Automatic Calibration (AD9361)**

   With the **AD9361** RF transceiver in the **BladeRF 2.0 Micro**, calibration is now **automatic**. This includes calibration of:
   - LMS parameters (IQ imbalance correction).
   - DC offset correction for both RX and TX paths.
   
   There is no longer a need to manually run multiple calibration commands (`cal lms`, `cal dc rxtx`), as was required with previous models. The transceiver performs dynamic calibration automatically during operation.

   However, you can still manually verify the configuration using:

   ```bash
   bladeRF-cli --interactive
   print
   ```

---

### **4. OAI and the bladeRF Libraries**

1. Install the OAI libraries

   **Note:** Run OAI's `./build_oai -I` to install the OAI libraries, the bladeRF dependencies were installed in the previous steps.

   ```
   sudo cmake_targets/build_oai -I
   ```


2. Building gNodeB Executables, along with the bladeRF module

   ```
   sudo cmake_targets/build_oai --gNB -w BLADERF
   ```

---

### **5. Tune the RX Gain Using the gNB (NR)**

   Run the softmodem and monitor the **gNB (NR)** logs for input signal strength and performance.  
   Instructions are available on the OAI wiki:  
   [OAI NR Documentation](https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/common/utils/T/DOC/T.md).

   In the logs, check the **input signal** values for time-domain signal power. This signal level should be around **30**. If the power level deviates, modify the `max_rxgain` parameter in your configuration file.

   **Configuration file:**  
   `targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.51PRB.bladerf20xa0.conf`

   Update the following parameters in the file:
   - `tracking_area_code`
   - `plmn_list` (MCC, MNC, and MNC length)
   - `mme_ip_address` (IP address of the EPC or 5GC core)
   - `NETWORK_INTERFACES` (Set gNB interface addresses to match your EPC or 5GC setup)

   ```
   CONFIG_FILE=targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.51PRB.bladerf20xa0.conf; \
     sudo cmake_targets/ran_build/build/nr-softmodem -O ${CONFIG_FILE} --sa --continuous-tx -E
   ```

---

### **6. Test the Setup**

   Connect a UE and run traffic tests (e.g., **iperf**).  
   Expected throughput:
   - **Downlink:** Over **16 Mbps**  
   - **Uplink:** Over **8 Mbps**

   In the OAI logs, you should observe entries with **PHR 40** and **CQI 15** if the setup is correctly optimized.

---

### **7. Troubleshooting**

   Ensure you are using appropriate radio equipment, including:
   - **Duplexers**
   - **Proper antennas**
   - **A low-interference environment**

   Check USB connections for stability and ensure the device is not overheating.

---

### **8. Additional Notes**

   - For high sample rates, ensure your system has sufficient USB 3.1 bandwidth and CPU resources.
   - Monitor **RX/TX overruns** using logs and adjust parameters (e.g., sample rate, buffer size, and gain) as needed.
   - Stay updated with the latest firmware and driver releases from Nuand's GitHub repository.


