# OpenAirInterface Cross-Compiler User Guide

## Environment

- docker with ubnutu:20.04
- branch: new-aarch64-build-cross
- commit: 32df7bcf80a1c9c

### Set up the environment

Set up for install the package for aarch64.

```shell
sudo dpkg --add-architecture arm64

echo -e \
"deb [arch=arm64] http://ports.ubuntu.com/ focal main restricted\n"\
"deb [arch=arm64] http://ports.ubuntu.com/ focal-updates main restricted\n"\
"deb [arch=arm64] http://ports.ubuntu.com/ focal universe\n"\
"deb [arch=arm64] http://ports.ubuntu.com/ focal-updates universe\n"\
"deb [arch=arm64] http://ports.ubuntu.com/ focal multiverse\n"\
"deb [arch=arm64] http://ports.ubuntu.com/ focal-updates multiverse\n"\
"deb [arch=arm64] http://ports.ubuntu.com/ focal-backports main restricted universe multiverse"\
| sudo tee /etc/apt/sources.list.d/arm-cross-compile-sources.list

sudo cp /etc/apt/sources.list "/etc/apt/sources.list.`date`.backup"
sudo sed -i -E "s/(deb)\ (http:.+)/\1\ [arch=amd64]\ \2/" /etc/apt/sources.list

sudo apt install -y gcc-9-aarch64-linux-gnu \
                    g++-9-aarch64-linux-gnu
```

## Install and Build

### Install the Requirement Package

Use the host compiler to install some dependencies.

```shell
cd cmake_targets
./build_oai -I

sudo apt-get install -y \
libatlas-base-dev:arm64 \
libblas-dev:arm64 \
liblapack-dev:arm64 \
liblapacke-dev:arm64 \
libreadline-dev:arm64 \
libgnutls28-dev:arm64 \
libconfig-dev:arm64 \
libsctp-dev:arm64 \
libssl-dev:arm64 \
libtool:arm64 \
zlib1g-dev:arm64
```

### Build the ldpc_generators

Use the x86 compiler to build the `ldpc_generators` and generate the header file in the `ran_build/build` folder.

```shell
rm -r ran_build
mkdir ran_build
mkdir ran_build/build
mkdir ran_build/build-cross

cd ran_build/build
cmake ../../..
make -j`nproc` ldpc_generators
```

### Build the Other Executables for aarch64

Switch to the `ran_build/build-cross` folder to build other executables.

```shell
cd ../build-cross
cmake ../../.. -DCMAKE_TOOLCHAIN_FILE=../../../cross-arm.cmake

make -j`nproc` dlsim ulsim ldpctest polartest smallblocktest nr_pbchsim nr_dlschsim nr_ulschsim nr_dlsim nr_ulsim nr_pucchsim nr_prachsim
make -j`nproc` lte-softmodem nr-softmodem nr-cuup oairu lte-uesoftmodem nr-uesoftmodem
make -j`nproc` params_libconfig coding rfsimulator
```

