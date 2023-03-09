#!/bin/bash

INSTALL_PATH="/opt/dev/install"

# exit when any command fails
set -e

BUILD_MODE="Release"

# supress warnings from git
git config --global advice.detachedHead false

mkdir -p $INSTALL_PATH

# build and install libssh-dev
cd $INSTALL_PATH
git clone --single-branch --branch libssh-0.9.2 https://git.libssh.org/projects/libssh.git
cd $INSTALL_PATH/libssh
mkdir -p build
cd build 
cmake -DWITH_EXAMPLES=OFF .. 
make -j4
make install
ldconfig

# build and install libyang
cd $INSTALL_PATH
git clone --single-branch --branch v2.1.30 https://github.com/CESNET/libyang.git
cd $INSTALL_PATH/libyang
mkdir -p build
cd build 
cmake -DCMAKE_BUILD_TYPE:String="$BUILD_MODE" -DGEN_LANGUAGE_BINDINGS=ON -DENABLE_BUILD_TESTS=OFF ..
make -j4 
make install
ldconfig

# build and install sysrepo
cd $INSTALL_PATH
git clone --single-branch --branch v2.2.36 https://github.com/sysrepo/sysrepo.git
cd $INSTALL_PATH/sysrepo
mkdir -p build
cd build 
cmake -DCMAKE_BUILD_TYPE:String="$BUILD_MODE" -DGEN_LANGUAGE_BINDINGS=ON -DGEN_CPP_BINDINGS=ON -DGEN_PYTHON_BINDINGS=OFF -DENABLE_TESTS=OFF -DREPOSITORY_LOC:PATH=/etc/sysrepo -DREQUEST_TIMEOUT=60 -DOPER_DATA_PROVIDE_TIMEOUT=60 ..
make -j4
make install
ldconfig

# build and install libnetconf2
cd $INSTALL_PATH
git clone --single-branch --branch v2.1.28 https://github.com/CESNET/libnetconf2.git
cd $INSTALL_PATH/libnetconf2
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE:String="$BUILD_MODE" -DENABLE_BUILD_TESTS=OFF ..
make -j4
make install
ldconfig

# build and install netopeer2
cd $INSTALL_PATH
git clone --single-branch --branch v2.1.49 https://github.com/CESNET/netopeer2.git
cd $INSTALL_PATH/netopeer2
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE:String="$BUILD_MODE" -DGENERATE_HOSTKEY=OFF -DMERGE_LISTEN_CONFIG=OFF ..
make -j4
make install

# build and install cURL
cd $INSTALL_PATH
git clone --single-branch --branch curl-7_87_0 https://github.com/curl/curl.git
cd $INSTALL_PATH/curl
mkdir -p build
cd build
cmake -DBUILD_TESTING=OFF ..
make -j4
make install
ldconfig
