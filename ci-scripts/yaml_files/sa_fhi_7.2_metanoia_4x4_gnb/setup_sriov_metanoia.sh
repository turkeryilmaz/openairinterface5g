set -e
sudo cpupower idle-set -D 0 > /dev/null
sudo ethtool -G ens2f1 rx 8160
sudo ethtool -G ens2f1 tx 8160
sudo sh -c 'echo 0 > /sys/class/net/ens2f1/device/sriov_numvfs'
sudo sh -c 'echo 2 > /sys/class/net/ens2f1/device/sriov_numvfs'
sudo modprobe -r iavf
sudo modprobe iavf
# this next 2 lines is for C/U planes
sudo ip link set ens2f1 vf 0 mac 00:11:22:33:44:54 vlan 3 spoofchk off mtu 9216
sudo ip link set ens2f1 vf 1 mac 00:11:22:33:44:65 vlan 3 spoofchk off mtu 9216
sleep 1
# These are the DPDK bindings for C/U-planes on vlan 4
sudo /usr/local/bin/dpdk-devbind.py --unbind 41:11.0
sudo /usr/local/bin/dpdk-devbind.py --unbind 41:11.1
sudo modprobe vfio-pci
sudo /usr/local/bin/dpdk-devbind.py --bind vfio-pci 41:11.0
sudo /usr/local/bin/dpdk-devbind.py --bind vfio-pci 41:11.1
# These are the DPDK bindings for T2 card
sudo dpdk-devbind.py -b vfio-pci 01:00.0
exit 0
