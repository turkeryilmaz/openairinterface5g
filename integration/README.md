# Integration

## How to build gNodeB image

We created a script which is doing the same as the instructions inside of the docker folder.
```
./build-gnb-docker.sh --base --build --gnb
```

If the base is already created once, only ```--build``` and ```--gnb``` args are neccessary.


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

depending on if VES is configured or not you have to mount the device manually.

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


## Using ODLUX Gui

By default this is available on ```http://localhost:8080```. Default user credentials are username ```admin``` and password ```admin```. The password is configured in the .env file ```ADMIN_PASSWORD```. 

On an already started OAI gNodeB you should already see the an entry in the ConnectApp in state ```Connected```. The node-id is corresponding with the gnb.conf of the device. By right-click on it you can see in the context menu ```Info``` all the supported yang specifications, this device is supporting.

If the gnb.conf property ```demoAlarmingInterval``` is set, you should also see an alarm inside of the FaultApp. There you have also a tab ```Alarm Log``` which shows the complete history of incoming alarms into the SDNC.

HINT: If you want to see Notifications as they are coming into the SDNC you have to enable Notifications. Therefore you have to click on your username on the top right corner and then go to settings. There you can enable and disable it. It is disabled by default to prevent flooding the browser with incoming messages, which can cause some instability.

In the ConfigurationApp you are able to read and write configuration from and to the device. Sadly up to now there is a bug in the opendaylight project which is preventing to access a List of objects in the root structure, which is the way 3GPP has designed their Models. Since we are just implementing one root element this list has now the size of one, we have to access this by url and cannot click through from beginning.
```
http://localhost:8080/odlux/index.html#/configuration/{node-id}/_3gpp-common-managed-element:ManagedElement[OAI%20gNodeB]

```

But from there on you have access to all the containers and leafs inside.

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

### Node does not even show up after started up

  * check the gnb.conf if ```vesUrl``` is correct
  * check the VES-Collector if message is coming in

```
$ docker logs -f dcae-ves-collector
```
  * there should be a log with something like 
```
2023-03-20 13:12:59.697  INFO 28 [nio-8443-exec-5] o.o.d.c.p.MessageRouterHttpStatusMapper  : Successfully send event to MR
```
  * if not please check the gnb logs
  * if yes please make sure that all components of the docker-compose.yml are up running
```
$ docker-compose ps -a
        Name                      Command                  State                             Ports                       
-------------------------------------------------------------------------------------------------------------------------
dcae-dfc               java -jar /opt/app/datafil ...   Up             8100/tcp, 8433/tcp                                
dcae-pm-mapper         java -jar pm-mapper-1.10.0.jar   Up                                                               
dcae-ves-collector     /bin/sh -c bin/docker-entry.sh   Up             8080/tcp, 8443/tcp                                
dmaap-dr-mariadb       docker-entrypoint.sh mysqld      Up (healthy)   3306/tcp                                          
dmaap-dr-node          sh startup.sh                    Up             8080/tcp, 8443/tcp                                
dmaap-dr-prov          sh startup.sh                    Up (healthy)   8080/tcp, 8443/tcp                                
dmaap-kafka            /etc/confluent/docker/run        Up             9092/tcp, 9093/tcp                                
dmaap-message-router   sh startup.sh                    Up             3904/tcp, 3905/tcp                                
dmaap-zookeeper        /etc/confluent/docker/run        Up             2181/tcp, 2888/tcp, 3888/tcp                      
sdnc                   /bin/sh -c /opt/onap/sdnc/ ...   Up             1090/tcp, 1099/tcp, 8181/tcp                      
sdnc-web               /opt/bitnami/scripts/nginx ...   Up             0.0.0.0:8080->8080/tcp,:::8080->8080/tcp, 8443/tcp
sdnrdb                 /tini -- /usr/local/bin/do ...   Up             9200/tcp, 9300/tcp            
```

