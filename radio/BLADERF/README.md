# BladeRF 2.0 Micro Documentation

---

[TOC]

---

As of **2025**, this documentation provides instructions for setting up and using the **BladeRF 2.0 Micro** with OpenAirInterface (OAI). If you encounter any discrepancies, contact the support team or check the latest resources on Nuand's GitHub.

---

### **1. Install BladeRF 2.0 Micro Libraries**

   - Install the latest **bladeRF 2.0 Micro** libraries and utilities.  
     It's recommended to build from source for the most recent updates.

      Follow this link :point_right: [1.Installing](DOC/1.Installing) to find the installation procedure based on your Linux package Manager.
   

---

### **2. Flashing the FPGA Image**

   - Download the latest FPGA images from Nuand's GitHub.


      1. Follow this link :point_right: [2.Flashing](DOC/2.Flashing#flashlight-flashing-board) to flash the board.

      2. Make sure to create the booting :point_right: [Nuand directory](DOC/2.Flashing#abacus-configuration-file) that will automatically load the FPGA image onto the board

      3. Finally, configure :point_right: [3.Configuring](DOC/3.Configuring) your setup to access the BladeRF device without requiring elevated privileges.

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


