# :gear: Configuring

--- 

[TOC]

----

## :a: [Frequency Tuning Modes](https://nuand.com/libbladeRF-doc/v2.5.0/group___f_n___t_u_n_i_n_g___m_o_d_e.html)

The `libbladeRF's compatibility table` issue may cause a perturbation that needs to set the Frequency Tuning Mode manually.

```
bladeRF-cli -e version
```
> Returns
```powershell
[INFO @ /privatehost/libraries/libbladeRF/src/helpers/version.c:106] FPGA version (v0.15.3) is newer than entries in libbladeRF''s compatibility table. Please update libbladeRF if problems arise.

  bladeRF-cli version:        1.9.0-git-41ef6346-dirty
  libbladeRF version:         2.5.0-git-41ef6346-dirty

  Firmware version:           2.4.0-git-a3d5c55f
  FPGA version:               0.11.1 (configured by USB host)
```

- [ ] [BladeRF becomes unusable after changing sample rate](https://github.com/Nuand/bladeRF/issues/778)

[](https://github.com/Nuand/bladeRF/issues/778#issuecomment-619869067)

I didn't know about those tuning modes but it turned out that it was set to host by default even though it should be FPGA according to [this](https://nuand.com/libbladeRF-doc/v2.2.1/group___f_n___t_u_n_i_n_g___m_o_d_e.html) with the FPGA/lib version I am using, shouldn't it?

So `export BLADERF_DEFAULT_TUNING_MODE=fpga` does fix the issue for the Rx only app like pdsch_ue. But interestingly, the full LTE UE which apparently does Rx and Tx works better with host mode albeit printing errors.

- Check it is set properly

```
bladeRF-cli -e "print hardware"
```
> Returns
```powershell

  Hardware status:
    RFIC status:
      Tuning Mode:  FPGA
      CTRL_OUT:     0xf8 (0x035=0x00, 0x036=0xff)
    Power source:   USB Bus
    Power monitor:  4.808 V, 0.56 A, 2.68 W
    RF routing:
      RX1: RFIC 0xff (N/A    ) <= SW 0x0 (OPEN   )
      RX2: RFIC 0xff (N/A    ) <= SW 0x0 (OPEN   )
      TX1: RFIC 0xff (N/A    ) => SW 0x0 (OPEN   )
      TX2: RFIC 0xff (N/A    ) => SW 0x0 (OPEN   )

```

## :b: `plugdev` GROUP (if not already present)

### 1.  Check the `plugdev` group entry

   You can list all the available groups on your Linux system using the following command:

   ```bash
   cat /etc/group
   ```

   This will display a list of all groups, along with their group IDs and members.

   ### Example Output
   ```
   root:x:0:
   daemon:x:1:
   bin:x:2:
   sys:x:3:
   adm:x:4:yourusername
   plugdev:x:46:yourusername
   ```

   ### Additional Commands
   1. **List groups your user belongs to:**
      ```bash
      groups $(whoami)
      ```

   2. **Search for a specific group:**
      If you are checking for a specific group (e.g., `plugdev`):
      ```bash
      getent group plugdev
      ```

   3. **Filter group names for readability:**
      If you only want group names, not full details:
      ```bash
      cut -d: -f1 /etc/group
      ```

### 2. Add Your User to the Group

Add your user to the `plugdev` group (or create it if it doesn’t exist):
	
   1.	Check if plugdev exists:

```
getent group plugdev
```

If not, create it:

```
sudo groupadd plugdev
```

	2.	Add your user to the group:

```
sudo usermod -aG plugdev $(whoami)
```

	3.	Log out and log back in to apply group changes.


## :ab: Setting `UDEV Rules`

To configure **udev rules** on Debian (or any Debian-based system) for BladeRF or other devices, follow these steps. Udev rules allow non-root users to access the BladeRF device without requiring elevated privileges.

---

### **1. Identify the Device**
Plug in your BladeRF device and identify its details using the `lsusb` command:
```bash
lsusb
```
> Returns
```powershell
Bus 004 Device 002: ID 2cf0:5250 Nuand LLC bladeRF 2.0 micro
Bus 004 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
Bus 003 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
Bus 001 Device 002: ID 1997:2433 Shenzhen Riitek Technology Co., Ltd wireless mini keyboard with touchpad
Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
```


Look for an entry corresponding to your BladeRF device. For example:
```
Bus 004 Device 002: ID 2cf0:5250 Nuand LLC bladeRF 2.0 micro
```

Take note of the `Vendor ID` (`2cf0`) and `Product ID` (`5250`).

---

### **2. Create a Udev Rule**
1. **Create a new udev rules file**:
   ```bash
   sudo vi /etc/udev/rules.d/88-nuand-bladerf.rules
   ```

2. **Add the following rules**:
   Replace `2cf0` and `5250` with the actual Vendor ID and Product ID for your BladeRF device if different:
   ```bash
   # Nuand bladeRF
   SUBSYSTEM=="usb", ATTR{idVendor}=="2cf0", ATTR{idProduct}=="5250", MODE="0666", GROUP="plugdev"
   ```

   - `MODE="0666"`: Grants read and write permissions to all users.
   - `GROUP="plugdev"`: Restricts access to users in the `plugdev` group (recommended for security).

```
groups
```
> pi adm dialout cdrom sudo audio video plugdev games users input render netdev lpadmin gpio i2c spi

---

### **3. Reload Udev Rules**
Reload the udev rules so that they take effect immediately:
```bash
sudo udevadm control --reload
sudo udevadm trigger
```

---

### **4. Add User to `plugdev` Group**
If you used the `GROUP="plugdev"` setting, ensure your user is part of the `plugdev` group:
1. Add your user to the group:
   ```bash
   sudo usermod -aG plugdev $USER
   ```

2. Log out and back in for the changes to take effect.

---

### **5. Verify the Configuration**
1. Unplug and replug the BladeRF device.
2. Check the permissions of the device node:
   ```bash
   ls -l /dev/bus/usb/004/002
   ```
   Replace `004/002` with the actual bus and device number for your BladeRF.

   The output should reflect the permissions and group assignment:
   ```
   crw-rw---- 1 root plugdev 189, 385 Jan  3 08:12 /dev/bus/usb/004/002
   ```

3. Test access using the BladeRF CLI:

Note: The device needs flashing first

   ```bash
   bladeRF-cli --exec info
   ```
   > Returns
   ```
  Board:                    Nuand bladeRF 2.0 (bladerf2)
  Serial #:                 b1f3da6672d54d37a2cd7c4d3e3c2e25
  VCTCXO DAC calibration:   0x1f91
  FPGA size:                49 KLE
  FPGA loaded:              yes
  Flash size:               32 Mbit
  USB bus:                  4
  USB address:              2
  USB speed:                SuperSpeed
  Backend:                  libusb
  Instance:                 0
```

---

### **6. Troubleshooting**
- **Device Not Found**: Ensure the Vendor ID and Product ID in the udev rule match the output of `lsusb`.
- **Permissions Not Applied**: Double-check the udev rules file location and syntax. Use `sudo udevadm control --reload` to reload the rules.

