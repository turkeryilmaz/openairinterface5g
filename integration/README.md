# Integration

## How to run

### Prerequisites

 * docker (sudo apt install docker.io)
 * docker-compose (pip3 install docker-compose)


### Start SDN-Controller and its components

 * edit ```.env``` file to set image versions and public available ports
 * run cmd:
```
docker-compose up -d
```


### Start gNodeB


```
docker-compose -f docker-compose.gnb.yml up -d
```

### Go to ODLUX gui

  * Open URL in the browser: http://your-deployed-ip-address:8181
  * login with credentials defined in .env file (default: admin:admin) 
  * find out ip address of the gnodeb container (```docker inspect gnodeb01-test | grep -i ipaddress)
  * mount the device in the connectApp 
    * node-id: whatever you like
    * host: ip address you found out
    * port: 830
    * username: netconf
    * password: netconf!
  * now the node should get into the connected state



## Troubleshooting

### Node does not get into connected state

  * check if the gnodeb container is running
  * get inside of the gnodeb container and check if the netopeer2 server is running

```
  docker exec -ti gnodeb-test bash
  ps -ef | grep netopeer
```
  * check if exceptions are thrown inside of the SDN-Controller
```
  docker exec -ti sdnc bash
  less /opt/opendaylight/data/log/karaf.log
  go to the end of the log (SHIFT+G) and then upwards to search for exceptions/stacktraces (e.g. wrong username or password)
```
