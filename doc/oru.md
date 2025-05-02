# O-RU setup on a device without a valid ethernet device

## Setup

Create two peer veth interfaces
```
sudo ip link add veth0 type veth peer veth1
sudo ip link set dev veth0 up
sudo ip link set dev veth1 up
sudo ip addr add 192.168.10.1/24 dev veth0
sudo ip addr add 192.168.10.2/24 dev veth1
```

### Test your setup
```
ping 192.168.10.1
ping 192.168.10.2
```

## Config your applications

use `vdev=net_pcap0,iface=veth0` and `vdev=net_pcap1,iface=veth1` for your DPDK device configs



