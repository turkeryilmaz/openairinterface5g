# dnf (Dandified YUM)

---

[TOC]

---

## Installation

### Install the required packages

```
sudo dnf install gcc gcc-c++ make cmake libusb1-devel ncurses-devel
```
> Returns
```powershell
Updating Subscription Management repositories.
<very long ...>
  ncurses-libs-6.2-10.20210508.el9.x86_64             

Complete!
```

### [Building bladeRF libraries and tools from source](https://github.com/Nuand/bladeRF/wiki/Getting-Started%3A-Linux#building-bladerf-libraries-and-tools-from-source)

```
git clone https://github.com/Nuand/bladeRF.git ./bladeRF
```
> Returns
```powershell
Cloning into './bladeRF'...
remote: Enumerating objects: 38305, done.
remote: Counting objects: 100% (4712/4712), done.
remote: Compressing objects: 100% (1576/1576), done.
remote: Total 38305 (delta 3186), reused 4436 (delta 3016), pack-reused 33593 (from 1)
Receiving objects: 100% (38305/38305), 13.02 MiB | 21.58 MiB/s, done.
Resolving deltas: 100% (24168/24168), done.
```

```
cd host && mkdir build && cd build
```

```
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DINSTALL_UDEV_RULES=ON ../
```
>Returns
```powershell
-- The C compiler identification is GNU 11.4.1
-- Detecting C compiler ABI info
<long ...>
-- Found Curses: /usr/lib64/libncursesw.so  
-- Configuring done (40.0s)
-- Generating done (0.1s)
-- Build files have been written to: /home/myuser/Developer/bladeRF/host/build
```

### Compile && install

```
make && sudo make install && sudo ldconfig
```
> Returns
```powershell
[  0%] Building C object common/thirdparty/ad936x/CMakeFiles/ad936x.dir/ad9361.c.o
[  0%] Building C object common/thirdparty/ad936x/CMakeFiles/ad936x.dir/ad9361_api.c.o
<very long ...>
-- Installing: /usr/local/bin/bladeRF-power
-- Set runtime path of "/usr/local/bin/bladeRF-power" to ""
```

## Set Group

```
$ groups
```
> myuser

```
sudo groupadd bladerf
```

```
sudo usermod -a -G bladerf ${USER}
```

### Now log out and log back in...

```
$ groups
```
> myuser bladerf



# References


##### epel-release

```
sudo yum install epel-release
```

:x: This didn't work