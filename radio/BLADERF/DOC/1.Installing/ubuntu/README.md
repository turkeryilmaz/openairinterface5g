# Ubuntu

---

[TOC]

---

To install the bladeRF-cli on Ubuntu, follow these steps:

## 1. Install Prerequisites

Ensure that your system is up-to-date and has the necessary dependencies:

```
sudo apt update
sudo apt upgrade
sudo apt install build-essential cmake libusb-1.0-0-dev libglib2.0-dev
```
> Returns
```powershell
Hit:1 http://ports.ubuntu.com/ubuntu-ports noble InRelease
<long ...>
Processing triggers for install-info (7.1-3build2) ...
```

## 2. Install bladeRF from Ubuntu Repositories

You can use the default Ubuntu repositories to install bladeRF-cli (though the version might not be the latest):

```
sudo apt install bladerf
```
> Returns
```powershell
Reading package lists... Done
...
Processing triggers for libc-bin (2.39-0ubuntu8.3) ...
```

- [ ] Check the installation:

```
bladeRF-cli --interactive
```
> Returns
```powershell
[WARNING @ host/libraries/libbladeRF/src/backend/usb/libusb.c:529] Found a bladeRF via VID/PID, but could not open it due to insufficient permissions.

No bladeRF device(s) available.

If one is attached, ensure it is not in use by another program
and that the current user has permission to access it.

bladeRF> version

  bladeRF-cli version:        1.9.0-0.2023.02-4build1
  libbladeRF version:         2.5.0-0.2023.02-4build1

  Device version information unavailable: No device attached.
```

```
sudo bladeRF-cli --interactive
```
> Returns
```powershell
[WARNING @ host/libraries/libbladeRF/src/board/bladerf2/bladerf2.c:461] FPGA bitstream file not found.
[WARNING @ host/libraries/libbladeRF/src/board/bladerf2/bladerf2.c:462] Skipping further initialization...
bladeRF> version

  bladeRF-cli version:        1.9.0-0.2023.02-4build1
  libbladeRF version:         2.5.0-0.2023.02-4build1

  Firmware version:           2.4.0-git-a3d5c55f
  FPGA version:               Unknown (FPGA not loaded)

bladeRF> 
```

## 3. Install via PPA (Optional)

Some Ubuntu versions may have a Personal Package Archive (PPA) available for BladeRF:

```
sudo add-apt-repository ppa:nuand/bladerf
sudo apt update
sudo apt install bladerf
```

Summary
-	Use sudo apt install bladerf for the easiest installation.
-	Build from source if you need the latest version.
-	Verify the installation with bladeRF-cli -i.

## 4. Purge

If you previously installed bladeRF through your package manager, remove it:
```bash
sudo apt-get purge bladerf
dpkg -l | grep -i blade  # Check for remaining packages and remove them
```


