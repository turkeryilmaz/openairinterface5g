# Debian

---

[TOC]

---

## Installation

### Install the required packages

```bash
sudo apt update
sudo apt install -y build-essential cmake git pkg-config libusb-1.0-0-dev libtecla-dev libcurl4-openssl-dev
```
> Returns
```html
Hit:1 http://deb.debian.org/debian bookworm InRelease
Hit:2 http://deb.debian.org/debian-security bookworm-security InRelease
Hit:3 http://deb.debian.org/debian bookworm-updates InRelease
Hit:4 http://archive.raspberrypi.com/debian bookworm InRelease
...
Processing triggers for libc-bin (2.36-9+rpt2+deb12u9) ...
```

### [Building bladeRF libraries and tools from source](https://github.com/Nuand/bladeRF/wiki/Getting-Started%3A-Linux#building-bladerf-libraries-and-tools-from-source)

#### a). Clone the **bladeRF** source code

```bash
git clone https://github.com/Nuand/bladeRF.git
```
> Returns
```terminal
cd bladeRF
Cloning into 'bladeRF'...
remote: Enumerating objects: 38356, done.
remote: Counting objects: 100% (1889/1889), done.
remote: Compressing objects: 100% (308/308), done.
remote: Total 38356 (delta 1682), reused 1634 (delta 1577), pack-reused 36467 (from 3)
Receiving objects: 100% (38356/38356), 13.24 MiB | 4.90 MiB/s, done.
Resolving deltas: 100% (24111/24111), done.
```

#### b). Compile and Install the binaries

```
mkdir build
cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig
```
> Returns
```powershell
-- The CXX compiler identification is GNU 12.2.0
-- The C compiler identification is GNU 12.2.0
...
-- Installing: /usr/local/lib/libbladeRF.so.2
-- Up-to-date: /usr/local/lib/libbladeRF.so
-- Installing: /usr/local/share/cmake/bladeRF/bladeRFConfig.cmake
-- Installing: /usr/local/share/cmake/bladeRF/bladeRFConfigVersion.cmake
-- Installing: /etc/udev/rules.d/88-nuand-bladerf1.rules
-- Installing: /etc/udev/rules.d/88-nuand-bladerf2.rules
-- Installing: /etc/udev/rules.d/88-nuand-bootloader.rules
-- Installing: /usr/local/bin/bladeRF-cli
-- Set runtime path of "/usr/local/bin/bladeRF-cli" to ""
-- Installing: /usr/local/bin/bladeRF-update
-- Set runtime path of "/usr/local/bin/bladeRF-update" to ""
-- Installing: /usr/local/bin/bladeRF-power
-- Set runtime path of "/usr/local/bin/bladeRF-power" to ""
-- Installing: /usr/local/bin/bladeRF-fsk
-- Set runtime path of "/usr/local/bin/bladeRF-fsk" to ""
```

