# :flashlight: Flashing Board

---

[TOC]

---

## :roll_of_paper: Download the board images

To download the FPGA bitstream file for the bladeRF 2.0 micro xA4 using curl, execute the following command in your terminal:

Note: `curl -O https://www.nuand.com/fpga/hostedxA4-latest.rbf` can be used be prefer using specific version number

```
curl -O https://www.nuand.com/fpga/v0.15.3/hostedxA4.rbf
```
> Returns
```powershell

  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 2570k    0 2570k    0     0   555k      0 --:--:--  0:00:04 --:--:--  607k
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


### Version

```
bladeRF-cli --exec version
```
> Returns
```powershell

  bladeRF-cli version:        1.9.0-git-41ef6346-dirty
  libbladeRF version:         2.5.0-git-41ef6346-dirty

  Firmware version:           2.4.0-git-a3d5c55f
  FPGA version:               0.11.1 (configured by USB host)
```

## :abacus: [Configuration File](https://nuand.com/libbladeRF-doc/v2.5.0/configfile.html)

This configuration will help the bladeRF board to load the bitstream images automatically.

- create a directory

```
mkdir -p ~/.config/Nuand/bladeRF
```

- Create a config file `~/.config/Nuand/bladeRF/bladerf.conf `

:speaking_head: Adjust the config file to your bladeRF `HOME` folder,  i.e. `/home/myuser/bladeRF`

```
cat ~/.config/Nuand/bladeRF/bladerf.conf 
```
> Returns
```powershell
# Load the hostedx40.rbf FPGA image, set the trim DAC, and
# configure the frequency
fpga /home/myuser/bladeRF/hostedxA4-latest.rbf
# trimdac 592
# frequency 2.4G
```

### That's it

The rest of this section is provided as a Reference ONLY

---

# :bookmark: References

The `bladeRF-cli` is a command-line interface tool for interacting with the bladeRF Software Defined Radio. It allows users to perform tasks like updating firmware, loading FPGA bitstreams, configuring the radio, and running diagnostics.

Here is a guide to using `bladeRF-cli`:

---

### **General Syntax**
```bash
bladeRF-cli [options] [commands]
```

### **Common Tasks**

#### **1. Launch the CLI**
To start the interactive mode:
```bash
bladeRF-cli
```

Once inside, you can type commands interactively. Use `help` for a list of available commands.

#### **2. Display Device Information**
To view details about your bladeRF device:
```bash
bladeRF-cli -i
```
Or, from interactive mode:
```bash
info
```

#### **3. Load an FPGA Bitstream**
To load an FPGA bitstream onto the device:
```bash
bladeRF-cli -l <path_to_bitstream.rbf>
```
Example:
```bash
bladeRF-cli -l hostedxA4-latest.rbf
```

#### **4. Update Firmware**
To update the FX3 firmware:
```bash
bladeRF-cli -f <path_to_firmware.img>
```
Example:
```bash
bladeRF-cli -f bladerf_fw_v2.3.1.img
```

#### **5. Configure Frequencies**
Set the RX and TX frequencies:
```bash
set frequency rx <frequency_in_hz>
set frequency tx <frequency_in_hz>
```
Example:
```bash
set frequency rx 915000000  # 915 MHz
set frequency tx 2400000000 # 2.4 GHz
```

#### **6. Configure Gains**
Set RX and TX gains:
```bash
set gain rx <gain_value>
set gain tx <gain_value>
```
Example:
```bash
set gain rx 30  # Set RX gain to 30
set gain tx 20  # Set TX gain to 20
```

#### **7. Run Transceive or Receive**
Start receiving or transmitting data:
```bash
rx config file=<file> format=<format>
tx config file=<file> format=<format>
```
Supported formats include `sc16q11` and `csv`.

#### **8. Reset the Device**
To reset the bladeRF:
```bash
bladeRF-cli -r
```

#### **9. Execute a Script**
Run a predefined script containing bladeRF CLI commands:
```bash
bladeRF-cli -s <script_file>
```
Example:
```bash
bladeRF-cli -s config_script.txt
```

#### **10. Test Device**
Run diagnostic tests to verify the bladeRF's functionality:
```bash
bladeRF-cli -t
```

---

### **Help and Documentation**
To see the full list of commands and options, use:
```bash
bladeRF-cli -h
```
Or, in interactive mode:
```bash
help
```

```
bladeRF-cli --exec help
```
> Returns
```powershell    
 | Command name     | Description                                                               | Device | FPGA  |
 |------------------+---------------------------------------------------------------------------+--------+-------|
 | calibrate        | Perform transceiver and device calibrations                               |   Y    |   Y   |
 | clear            | Clear the screen                                                          |        |       |
 | echo             | Echo each argument on a new line                                          |        |       |
 | erase            | Erase specified erase blocks of SPI flash                                 |   Y    |       |
 | flash_backup     | Back up flash data to a file with metadata.                               |   Y    |       |
 |------------------+---------------------------------------------------------------------------+--------+-------|
 | flash_image      | Print a flash image`s metadata or create a new flash image                |        |       |
 | flash_init_cal   | Write new calibration data to a device or to a file                       |   Y    |       |
 | flash_restore    | Restore flash data from a file                                            |   Y    |       |
 | fw_log           | Read firmware log contents                                                |   Y    |       |
 | generate         | Generates signals such as an offset CW                                    |        |       |
 |------------------+---------------------------------------------------------------------------+--------+-------|
 | help             | Provide information about specified command                               |        |       |
 | info             | Print information about the currently opened device                       |   Y    |       |
 | jump_to_boot     | Clear FW signature in flash and jump to FX3 bootloader                    |   Y    |       |
 | load             | Load FPGA or FX3                                                          |   Y    |       |
 | mimo             | Modify device MIMO operation                                              |   Y    |   Y   |
 |------------------+---------------------------------------------------------------------------+--------+-------|
 | open             | Open a bladeRF device                                                     |        |       |
 | peek             | Peek a memory location                                                    |   Y    |   Y   |
 | poke             | Poke a memory location                                                    |   Y    |   Y   |
 | print            | Print information about the device                                        |   Y    |   Y   |
 | probe            | List attached bladeRF devices                                             |        |       |
 |------------------+---------------------------------------------------------------------------+--------+-------|
 | quit             | Exit the CLI                                                              |        |       |
 | recover          | Load firmware when operating in FX3 bootloader mode                       |        |       |
 | run              | Run a script                                                              |        |       |
 | rx               | Receive IQ samples                                                        |   Y    |   Y   |
 | set              | Set device settings                                                       |   Y    |   Y   |
 |------------------+---------------------------------------------------------------------------+--------+-------|
 | trigger          | Control triggering                                                        |   Y    |   Y   |
 | tx               | Transmit IQ samples                                                       |   Y    |   Y   |
 | version          | Host software and device version information                              |        |       |
 | xb               | Enable and configure expansion boards                                     |   Y    |   Y   |

The letter Y in the "Device" and "FPGA" columns indicates the command
  needs the device to open, and the FPGA to be loaded, respectively.
```

## difference between **`adsbxA4.rbf`** and **`hostedxA4-latest.rbf`** 

The difference between **`adsbxA4.rbf`** and **`hostedxA4-latest.rbf`** lies in their specific configurations and intended use cases for the bladeRF FPGA:

---

### **1. `adsbxA4.rbf`**
- **Purpose**: This FPGA image is optimized for **Automatic Dependent Surveillance-Broadcast (ADS-B)** applications. ADS-B is a system used in aviation for aircraft tracking.
- **Configuration**:
  - Tailored to process ADS-B signals at **1090 MHz** (Mode S).
  - Likely optimized for lower latency, specific decoders, and efficient handling of the ADS-B protocol.
  - Includes pre-configured settings and signal processing blocks specific to ADS-B operations.
- **Use Case**:
  - Use this image when working with ADS-B receivers or applications like **dump1090** or other aircraft-tracking software.
- **Limitations**:
  - May lack general-purpose features or full configurability for other SDR applications.

---

### **2. `hostedxA4-latest.rbf`**
- **Purpose**: This is the general-purpose FPGA image for the bladeRF 2.0 micro xA4, intended for a wide range of SDR applications.
- **Configuration**:
  - Provides the full functionality of the bladeRF hardware, including support for various sampling rates, bandwidths, and frequencies.
  - Designed to work with most software tools, including GNU Radio, SoapySDR, and bladeRF-cli.
  - Includes general-purpose DSP blocks, filters, and interfaces for both RX and TX.
- **Use Case**:
  - Use this image for general SDR development and experimentation.
  - Ideal for applications that require custom configurations or broad frequency coverage.
- **Limitations**:
  - May not be optimized for specific protocols like ADS-B, resulting in lower efficiency for such tasks compared to protocol-specific images.

---

### **3. Key Differences**

| Feature                 | `adsbxA4.rbf`                            | `hostedxA4-latest.rbf`                  |
|-------------------------|------------------------------------------|-----------------------------------------|
| **Purpose**             | ADS-B signal processing at 1090 MHz     | General-purpose SDR functionality       |
| **Optimization**        | Tailored for ADS-B applications          | Supports a wide range of SDR tasks      |
| **Frequency Range**     | Limited to ADS-B frequencies (e.g., 1090 MHz) | Full bladeRF frequency range (70 MHz–6 GHz) |
| **Protocol Support**    | ADS-B specific                          | None specific; fully configurable       |
| **Use Case**            | Aircraft tracking, ADS-B decoding       | General SDR tasks (radio, LTE, etc.)    |

---

### **4. When to Use Each**
- Use **`adsbxA4.rbf`** if:
  - You're specifically working on ADS-B or Mode S decoding.
  - You need optimized performance for 1090 MHz signal processing.

- Use **`hostedxA4-latest.rbf`** if:
  - You're experimenting with SDR in general.
  - You need flexibility for tasks like FM radio, LTE, Wi-Fi, or custom waveforms.

---

### **5. Download the `adsbxA4.rbf`**

To download the `adsbxA4.rbf` FPGA bitstream file for the bladeRF 2.0 micro xA4 using curl, execute the following command in your terminal:

```
curl -O https://www.nuand.com/fpga/adsbxA4.rbf
```
> Returns
```powershell
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 2570k    0 2570k    0     0   767k      0 --:--:--  0:00:03 --:--:--  767k
```

This command retrieves the `adsbxA4.rbf` file from the specified URL and saves it in your current directory. The -O option instructs curl to save the file with its original name.

After downloading, you can load the FPGA image onto your bladeRF device using the bladeRF-cli tool:

```
 bladeRF-cli --load-fpga adsbxA4.rbf
```
> Returns
```powershell
Loading FPGA...
Successfully loaded FPGA bitstream!
```

This command loads the FPGA image into the bladeRF device, preparing it for operation. ￼

#### :x: Warning: ‼️  Loading the TWO FPGA Images cause the buffer to be waiting!

ONLY Load one FPGA image, unless you know what you are doing

```powershell
[DEBUG @ host/libraries/libbladeRF/src/streaming/sync.c:465] sync_rx: Worker is idle. Going to reset buf mgmt. 
[DEBUG @ host/libraries/libbladeRF/src/streaming/sync.c:485] sync_rx: Reset buf_mgmt consumer index 
[DEBUG @ host/libraries/libbladeRF/src/streaming/sync.c:500] sync_rx: Worker is now running. 
[ERROR @ host/libraries/libbladeRF/src/streaming/sync.c:352] wait_for_buffer: Timed out waiting for buf_ready after 500 ms 
[ERROR @ host/libraries/libbladeRF/src/streaming/sync.c:352] wait_for_buffer: Timed out waiting for buf_ready after 500 ms 
[ERROR @ host/libraries/libbladeRF/src/backend/usb/libusb.c:1090] Transfer timed out for buffer 0x12f011a00
```




### **6. Switching FPGA Images**
To switch between these images, use the following command with the `bladeRF-cli`:
```bash
bladeRF-cli -e "load /path/to/fpga_image.rbf"
```

Example:
```bash
bladeRF-cli -e "load adsbxA4.rbf"
```

Verify the loaded image:
```bash
bladeRF-cli -e "print info"
```

## [Upgrading Firmware](https://github.com/Nuand/bladeRF/wiki/Upgrading-bladeRF-FX3-Firmware)

Let me know if you have any specific use case questions or need further clarification!
