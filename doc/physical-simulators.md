# Physical Simulators ("Physims") User Guide

This document provides an overview and usage guide for the **physical simulators**, also referred to as **unitary
simulators** or simply **physims**, used in the OpenAirInterface (OAI) project for LTE and 5G PHY layer validation.

[[_TOC_]]

# Introduction

**Unitary simulations** are standalone test programs designed to validate individual physical layer (L1) transport or
control channels. These simulations are based on **Monte Carlo techniques**, enabling statistical evaluation of
performance metrics such as block error rate (BLER), hybrid automatic repeat request (HARQ) throughput, and decoding
accuracy under various signal conditions.

Physims are essential for:

* Debugging and evaluating new PHY code in isolation
* Regression testing
* Ensuring correctness before merging new contributions into the repository

These tests are run automatically as part of the following
pipelines:

- [RAN-PhySim-Cluster-4G](https://jenkins-oai.eurecom.fr/job/RAN-PhySim-Cluster-4G/)
- [RAN-PhySim-Cluster-5G](https://jenkins-oai.eurecom.fr/job/RAN-PhySim-Cluster-5G/)

## Examples of Simulators

| Technology | Simulators                                | Description                      |
|------------|-------------------------------------------|----------------------------------|
| 4G LTE     | `dlsim`, `ulsim`                          | Downlink and uplink simulators   |
| 5G NR      | `nr_dlsim`, `nr_ulsim`                    | Downlink and uplink simulators   |
|            | `nr_dlschsim`, `nr_ulschsim`              | HARQ and TB throughput tests     |
|            | `nr_pucchsim`                             | Control channel simulation       |
|            | `nr_pbchsim`                              | Broadcast channel simulation     |
|            | `nr_prachsim`                             | PRACH simulation                 |
|            | `nr_psbchsim`                             | Sidelink simulation              |
| Coding     | `ldpctest`, `polartest`, `smallblocktest` | LDPC, Polar, and other FEC tests |

## Source Locations

* 4G PHY simulators: `openair1/SIMULATION/LTE_PHY/`
* 5G PHY simulators: `openair1/SIMULATION/NR_PHY/`
* Coding unit tests: `openair1/PHY/CODING/TESTBENCH/`

Example:

```bash
# 5G Downlink simulator
openair1/SIMULATION/NR_PHY/dlsim.c
```

# How to Run Simulators Using `ctest`

## Option 1: Using CMake

Build the simulators using the dedicated switch, then run `ctest` which will
run all registered tests.

```bash
cd openairinterface5g
mkdir build && cd build
cmake .. -GNinja -DENABLE_PHYSIM_TESTS=ON
ninja tests
ctest
```

## Option 2: Using the `build_oai` script

This method simplifies the process by automatically building the simulators with ctest support.

```bash
cd openairinterface5g/cmake_targets
./build_oai --ninja --phy_simulators
cd ran_build/build
ctest
```

## `ctest` Usage Tips

Use the following options to customize test execution:

| Option           | Description                                          |
|------------------|------------------------------------------------------|
| `-R <regex>`     | Run tests matching the regex pattern (by name)       |
| `-L <regex>`     | Run tests with labels matching the regex pattern     |
| `-E <regex>`     | Exclude tests matching the regex pattern (by name)   |
| `-LE <regex>`    | Exclude tests with labels matching the regex pattern |
| `--print-labels` | Display all available test labels                    |
| `-j <jobs>`      | Run tests in parallel using specified number of jobs |

For the complete list of `ctest` options, refer to the manual:

    man ctest

For instance, to run only all run NR ULSCH simulator tests, with 4 jobs in
parallel, type

    ctest -L nr_ulschsim -j 4

# Adding a New Physim Test

To define a new test or modify existing ones, update the following file:

```
openair1/SIMULATION/tests/CMakeLists.txt
```

## `add_physim_test()`

Use the `add_physim_test()` macro with the following arguments:

    add_physim_test(<test_name> <test_description> <test_exec> <test_options>)

where:
- `<test_name>` can be any name, but the canonical, historical format is to put
  it `physim.<gen>.<test_exec>.test<XYZ>` where `<gen>` is 4g/5g, and `<XYZ>`
  is an increasing number
- `<test_description>` is a human-readable description of the test
- `<test_exec> <test_options>` is the test invocation, where `<test_exec>` must
  be a target built by OAI cmake (e.g., `nr_prachsim`), followed by any options.

For instance, a PRACHsim looks like this:

    add_physim_test(physim.5g.nr_prachsim.test8 "15kHz SCS, 25 PRBs" nr_prachsim -a -s -30 -n 300 -p 99 -R 25 -m 0)

## `add_timed_physim_test()`

Use the `add_timed_physim_test()` macro to add a test the same way as with
`add_physim_test()` above. Additionally, it allows to check for thresholds with
`check_physim_threshold()`:

    check_physim_threshold(<test_name> <threshold> <condition>)

where:
- `<test_name>` is any test that must have been added with
  `add_timed_physim_test()`
- `<threshold>` is a threshold to check for, e.g., `PHY tx proc`, and
- `<condition>` a condition to check, e.g. `< 200`

Thus upon execution of the test, the test will be run, but additionally ctest
will check for a match of `PHY tx proc <NUMBER>` (where `<NUMBER> is of format
`[0-9]+(\.[0-9]+)?`), and a matching number will be checked against condition
`< 200`.

For instance, this could look like this:

    add_timed_physim_test(physim.5g.nr_dlsim.test3 "Some description" nr_dlsim -P)
    check_physim_threshold(physim.5g.nr_dlsim.test3 "DLSCH encoding time" "< 50")

# How to rerun failed CI tests using `ctest`

Ctest automatically logs the failed tests in LastTestsFailed.log. This log is archived in
the CI artifacts and can be reused locally to rerun only those failed tests.

```bash
# 1. Navigate to the build directory, build physims
cd ~/openairinterface5g/build
cmake .. -GNinja -DENABLE_PHYSIM_TESTS=ON && ninja tests

# 2. Create ctest directory structure
ctest -N --quiet

# 3. Unzip the test logs artifact from the CI run
unzip /path/to/test_logs_123.zip

# 4. Copy the LastTestsFailed.log file to the expected location
cp /path/to/test_logs_123/test_logs/LastTestsFailed.log ./Testing/Temporary/

# 5. Rerun only the failed tests using ctest
ctest --rerun-failed
```

# Legacy Bash Autotest (Deprecated)

> **Note:** Autotest script, configuration, and documentation, are no longer maintained.

For legacy support or archival purposes, you can still find this implementation by checking out the historical tag:

```bash
git checkout 2025.w18
```

# Unmaintained tests

A few tests dedicated to 4G are unmaintained:

- `mbmssim`
- `scansim`
- all simulators of format `www-tmyyyy` (for instance, `dlsim_tm4`)
