# LDPC coding implementation
This document gives an overview of the different LDPC coding implementations (functional or not) available with Open Air Interface.

[[_TOC_]]

The LDPC encoder and decoder are implemented in a shared library, dynamically loaded at run-time using the [oai shared library loader](file://../../../../common/utils/DOC/loader.md).
Two types of library are available with two different interfaces. There are libraries implementing the encoder and decoder of slots and libraries implementing the encoder and decoder of code segments.

## LDPC slot coding
The interface of the library is defined in [nrLDPC_coding_interface.h](file://../nrLDPC_coding/nrLDPC_coding_interface.h).
The code loading the LDPC library is in [nrLDPC_coding_interface_load.c](file://../nrLDPC_coding/nrLDPC_coding_interface_load.c), in function `load_nrLDPC_coding_interface`, which must be called at init time.

### Selecting the LDPC library at run time

By default the function `int load_nrLDPC_coding_interface(void)` looks for `libldpc.so`.\
This default behavior can be changed using the oai loader configuration options in the configuration file or from the command line as shown below:

#### Examples of ldpc shared lib selection when running nr softmodem's:

loading default `libldpc.so`:

```
./nr-softmodem -O libconfig:gnb.band78.tm1.106PRB.usrpx300.conf:dbgl5
```

`libldpc.so` has its decoder implemented in [nrLDPC_coding_segment_decoder.c](file://../nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_coding_segment_decoder.c).\
Its encoder is implemented in [nrLDPC_coding_segment_encoder.c](file://../nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_coding_segment_encoder.c).

loading `libldpc_aal.so` instead of `libldpc.so`:

`make ldpc_aal`

This command creates the `libldpc_aal.so` shared library.

```
Building C object CMakeFiles/ldpc_aal.dir/openair1/PHY/CODING/nrLDPC_coding/nrLDPC_coding_aal/nrLDPC_coding_aal.c.o
Linking C shared module libldpc_aal.so
```

At runtime, to successfully use LDPC accelerators (e.g., Xilinx T2/Intel ACCs), you will need to install the corresponding drivers and tools.
Please refer to the dedicated documentation at [LDPC_OFFLOAD_SETUP.md](file://../../../../doc/LDPC_OFFLOAD_SETUP.md).

<<<<<<< HEAD
```
./nr-softmodem -O  libconfig:gnb.band78.sa.fr1.106PRB.usrpb210.conf:dbgl5 --rfsim --rfsimulator.serveraddr server  --log_config.gtpu_log_level info  --loader.ldpc.shlibversion _aal --nrLDPC_coding_aal.dpdk_dev 01:00.0 --nrLDPC_coding_aal.dpdk_core_list 0-1
```

`libldpc_aal.so` has its decoder and its encoder implemented in [nrLDPC_coding_aal.c](file://../nrLDPC_coding/nrLDPC_coding_aal/nrLDPC_coding_aal.c).

loading `libldpc_xdma.so` instead of `libldpc.so`:

`make ldpc_xdma` or `ninja ldpc_xdma`

This command creates the `libldpc_xdma.so` shared library.

```
ninja ldpc_xdma
[2/2] Linking C shared module libldpc_xdma.so
```

At runtime, to successfully use the xdma, you need to install vendor specific drivers and tools.\
Please refer to the dedicated documentation at [LDPC_XDMA_OFFLOAD_SETUP.md](file://../../../../doc/LDPC_XDMA_OFFLOAD_SETUP.md).

```
./nr-softmodem -O libconfig:gnb.band78.sa.fr1.106PRB.usrpb210.conf:dbgl5 --rfsim --rfsimulator.serveraddr server --log_config.gtpu_log_level info --loader.ldpc.shlibversion _xdma --nrLDPC_coding_xdma.num_threads_prepare 2
```

`libldpc_xdma.so` has its decoder implemented in [nrLDPC_coding_xdma.c](file://../nrLDPC_coding/nrLDPC_coding_xdma/nrLDPC_coding_xdma.c).\
Its encoder is implemented in [nrLDPC_coding_segment_encoder.c](file://../nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_coding_segment_encoder.c).

*Note: `libldpc_xdma.so` relies on a segment coding library for encoding.*
*The segment coding library is `libldpc.so` by default but it can be chosen with option `--nrLDPC_coding_xdma.encoder_shlibversion` followed by the library version - like with `--loder.ldpc.shlibversion` in the segment coding case above -*

#### Examples of ldpc shared lib selection when running ldpctest:

Slot coding libraries cannot be used yet within ldpctest.

But they can be used within nr_ulsim, nr_dlsim, nr_ulschsim and nr_dlschsim.\
In these PHY simulators, using the slot coding libraries is enabled in the exact same way as in nr-softmodem.

### LDPC libraries
Libraries implementing the slotwise LDPC coding must be named `libldpc<_version>.so`. They must implement four functions: `nrLDPC_coding_init`, `nrLDPC_coding_shutdown`, `nrLDPC_coding_decoder` and `nrLDPC_coding_encoder`. The prototypes for these functions is defined in [nrLDPC_coding_interface.h](file://../nrLDPC_coding/nrLDPC_coding_interface.h).

`libldpc.so` is completed.

`libldpc_aal.so` is completed.

`libldpc_xdma.so` is completed.

## LDPC segment coding
The interface of the library is defined in [nrLDPC_defs.h](file://../nrLDPC_defs.h) as typedefs of the functions of the interface.
The name of the functions implementing these typedefs can be found in [nrLDPC_extern.h](file://../nrLDPC_extern.h).

LDPC segment coding libraries are not loaded directly by OAI but are statically linked in libraries implementing the LDPC slot coding interface using an adaptation layer between the slot coding interface and the segment coding interface.  
This way, these libraries can be loaded to implement the slot coding interface as well as the segment coding interface in `ldpctest`.  
`ldpctest` therefore loads directly the segment coding interface with the loader implemented in [nrLDPC_load.c](file://../nrLDPC_load.c) as function `load_nrLDPClib`.

### Selecting the LDPC library at run time

By default the function `int load_nrLDPClib(void)` looks for `libldpc.so`, this default behavior can be changed using the oai loader configuration options in the configuration file or from the command line as shown below:

#### Examples of ldpc shared lib selection when running nr softmodem's:

loading `libldpc_orig.so` instead of `libldpc.so`:

```
./nr-softmodem -O libconfig:gnb.band78.tm1.106PRB.usrpx300.conf:dbgl5  --loader.ldpc.shlibversion _orig
```

loading `libldpc_cl.so` instead of `libldpc.so`:

`make ldpc_cl`

This command creates the `libldpc_cl.so` shared library. To perform this build successfully, only the OpenCL header `/usr/include/CL/opencl.h` and library `/usr/lib/x86_64-linux-gnu/libOpenCL.so` are required, they implement OpenCL API support which is not hardware dependent.

```
Scanning dependencies of target nrLDPC_decoder_kernels_CL
Built target nrLDPC_decoder_kernels_CL
Scanning dependencies of target ldpc_cl
Building C object CMakeFiles/ldpc_cl.dir/usr/local/oai/oai-develop/openairinterface5g/openair1/PHY/CODING/nrLDPC_decoder/nrLDPC_decoder_CL.c.o
In file included from /usr/include/CL/cl.h:32,
                 from /usr/include/CL/opencl.h:38,
                 from /usr/local/oai/oai-develop/openairinterface5g/openair1/PHY/CODING/nrLDPC_decoder/nrLDPC_decoder_CL.c:49:
/usr/include/CL/cl_version.h:34:9: note: #pragma message: cl_version.h: CL_TARGET_OPENCL_VERSION is not defined. Defaulting to 220 (OpenCL 2.2)
 #pragma message("cl_version.h: CL_TARGET_OPENCL_VERSION is not defined. Defaulting to 220 (OpenCL 2.2)")
         ^~~~~~~

Building C object CMakeFiles/ldpc_cl.dir/usr/local/oai/oai-develop/openairinterface5g/openair1/PHY/CODING/nrLDPC_encoder/ldpc_encoder_optim8segmulti.c.o
Linking C shared module libldpc_cl.so
Built target ldpc_cl

```

At runtime, to successfully use hardware acceleration via OpenCL, you need to install vendor specific packages which deliver the required drivers and tools to make use of their GPU (Nvidia, Intel...) , fpga (Xilinx, Intel) or CPU (Intel, AMD, ARM...) through OpenCL. 

```
./nr-softmodem -O  libconfig:gnb.band78.sa.fr1.106PRB.usrpb210.conf:dbgl5 --rfsim --rfsimulator.serveraddr server  --log_config.gtpu_log_level info  --loader.ldpc.shlibversion _cl
```

```
./nr-uesoftmodem -r 106 --numerology 1 --band 78 -C 3619200000 --rfsim -O libconfig:/usr/local/oai/conf/nrue_sim.conf:dbgl5 --loader.ldpc.shlibversion _cl --log_config.hw_log_level inf
```

A mechanism to select ldpc implementation is also available in the `ldpctest` phy simulator via the `-v` option, which can be used to specify the version of the ldpc shared library to be used.

#### Examples of ldpc shared lib selection when running ldpctest:

Loading libldpc_cuda.so, the cuda implementation of the ldpc decoder:

```
$ ./ldpctest -v _cuda
```

Loading libldpc_cl.so, the opencl implementation of the ldpc decoder:

`make ldpc_cl`

```
$ ./ldpctest -v _cl
```

### LDPC libraries
The prototypes for the functions of the interface are defined in [nrLDPC_defs.h](file://nrLDPC_defs.h) as typedefs.
Libraries implementing the LDPC algorithms must be named `libldpc<_version>.so`, they must implement four functions with the names and types:
* `LDPCinit` implementing type `LDPC_initfunc_t *`
* `LDPCshutdown` implementing type `LDPC_shutdownfunc_t *`
* `LDPCdecoder` implementing type `LDPC_decoderfunc_t *`
* `LDPCencoder` implementing type `LDPC_encoderfunc_t *`

`libldpc_cuda.so` has been tested with the `ldpctest` executable, usage from the softmodem's has to be tested.

`libldpc_cl.so` is under development.

[oai Wikis home](https://gitlab.eurecom.fr/oai/openairinterface5g/wikis/home)
