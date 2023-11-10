# Running 5G Sidelink mode 2 tests
To launch tests, a python script `run_sl_test.py` is provided. It provides 2 test modes: one is launching RFSIM, the other is launching USRP. The configurations of different tests are specified in `sl_net_config.json`.
The python run script and configurations files are located under `~/openairinterface5g/ci-scripts`.

The following examples show three nodes tests for each mode.

## Launching in RFSIM test mode ##
Open a terminal and run the following commands.
```
cd ~/openairinterface5g/ci-scripts
```
To run rfsim test with the net 2 specified in `sl_net_config.json`, the following command is used.
```
python3 run_sl_test.py --test rfsim --net 2
```

## Launching in USRP test mode ##
The following is an example to run remote machine (nearby).
The syncref UE is launched on the current machine, and a single
nearby UE is launched on the machine specified by the `--host` and
`--user` flags. The `-r` will enable this simulation to be repeated
three times.

```
python3 run_sl_test.py --user account --host 10.1.1.68 -r 3 --test usrp
```
The following is an example to run just a Sidelink Nearby UE.
By specifying `-l` nearby, only the nearby UE will be launched
on the machine specified by the `--host` and `--user` flags.
The `-r` will enable this simulation to be repeated two times.

```
python3 run_sl_test.py -l nearby --user account --host 10.1.1.68 -r 2 --test usrp
```
To run usrp test in the net 3 specified in `sl_net_config.json`, the following command is used.

```
python3 run_sl_test.py --test usrp --net 3
```
#
# See `--help` for more information.
#

# Checking launching result #

All results will be logged into log files in home folder by default with the file name following `-l` flag.

In case of above example, the following files will be created after the test.

```
Shell> ls ~/*.log
~/syncref1.log
~/nearby2.log
~/nearby3.log
```