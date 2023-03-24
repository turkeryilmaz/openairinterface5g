# Netconf Server for O1 interface

## Specification

For this part please look into the [specification](../../../../specification/README.md) folder.

## Configuration

Example

```
o1Config :
{
  nfNodeId =  "gnb-test01";
  netconfUsername = "netconf";
  netconfPassword = "netconf!";
  netconfPort = 58300;
  netconfHost = "192.168.5.207";
  vesUrl = "https://dcae-ves-collector:8443/eventListener/v7";
  vesBasicAuthUsername  = "sample1";
  vesBasicAuthPassword = "sample1";
  vesNfVendorName = "OpenAirInterface";
  vesNfNamingCode = "OGNB";
  vesOamIpv6 = "";
  vesFtpServerPort = 21;
  vesFtpServerListenAddress = "0.0.0.0";
  vesFtpServerUrl = "ftp://netconfr:netconf@gnodeb01-test:21";
  demoAlarmingInterval = 30;
};
```

Properties:

| property | type | mandatory | description |
| -------- | ---- | --------- | ----------- |
| nfNodeId | string |true | network function node-id to get mounted via netconf |
| netconfUsername | string | true | username to access netconf server(at the moment hardcoded)|
| netconfPassword | string | true | password to access netconf server(at the moment hardcoded)|
| netconfPort | uint_16 | true | port to listen on for netconf server(at the moment hardcoded to 830).|
| netconfHost | string | true | |
| vesUrl| string | false | url to send VES messages to. If not set, complete VES functionality is disabled |
| vesBasicAuthUsername| string | if vesUrl | basic auth username for the VES collector |
| vesBasicAuthPassword | string | if vesUrl | basic auth password for the VES collector |
| vesNfVendorName | string | if vesUrl | vendor name to be sent in the VES messages |
| vesNfNamingCode | string | if vesUrl |  nfNamingCode to be sent in the VES messages |
| vesOamIpv6 | string | false | |
| vesFtpServerPort | uint_16 | false ||
| vesFtpServerListenAddress| string | false | |
| vesFtpServerUrl | string | false | |
| demoAlarmingInterval | unit_16| false | interval in seconds to send and clear a demo alarm sent via VES |



The VES functionality can be disabled by just not setting the ```vesUrl```. With this all of the ```vesXXX``` properties are obsolete.


## Software Architecture

```
|----------------------------------------------------------------------------------------------|
| OAI container              |  Netopeer2 server  |             |   FTPes-server   |           |
|                            |--------------------|             |------------------|           |
|                                       |                                 |                    |
|                                       |                                 |                    |
|                            |--------------------|                       |                    |
|                            |      sysrepo       |                       |                    |
|                            |    (datastore)     |                       |                    |
|                            |--------------------|                       |                    |
|      gnb.conf                         |                                 |                    |
|         |                             |                                 |                    |
|  ------------------------------------------------------------------------------------------  |
|  | nr-softmodem            |               |                                              |  |
|  |                         |               |                                              |  |
|  |                  read-oper-        edit-config-                                        |  |
|  |                   callbacks         callbacks                                          |  |
|  |                                                                        ves-client      |  |
|  |                                                                                        |  |
|  |                                                                                        |  |
|  |                                                                                        |  |
|  |                                                                                        |  |
|  |                                                                                        |  |
|----------------------------------------------------------------------------------------------|
```


## Minimal Deployment Architecture

This picture is only showing the directly connected components to the OAI gNodeB. This is NOT a complete deployment since there a some more components needed to make everything work together.

```
    smo-network-----|------------------------------------------------------------------------
                    |                         |                      |
           |-------------------|    |--------------------|   |-----------------|
           |       SDNC        |    |    VES-Collector   |   |  FileCollector  |  
           |                   |    |                    |   |                 |
           |-------------------|    |--------------------|   |-----------------|
                    |                         |                      |                           
    oam-network-----|-------------------------|----------------------|-------------------
                    |                         |                      |                           
           |--------------------------------------------------------------------|                     
           | | NETCONF-Server |        | VES-Client |         | FTPes-Server |  |                     
           | |----------------|        |------------|         |--------------|  |                     
           |                                                                    |                     
           |                    OAI gNodeB                                      |           
           |                                                                    |                     
           |--------------------------------------------------------------------|                     
                                                                                              
```


## Integration Deployment And Usage

for this documentation please check the [integration](../../../../integration/README.md) folder.

### PNF Registration

By starting up the gNodeB should automatically sends a pnfRegistration message to the VES-Collector. This message will be received by the SDNC and creates then a NETCONF connection between the itself and the device. 

### PNF Alarming

If the ```demoAlarmingInterval``` is set, every n seconds there will be a alarm notification sent via VES to raise and clear the demoAlarm.


### PNF Performance Data

To transmit performance data to the ONAP system there is a ```file-ready``` event sent to the VES-Collector which contains the information of the file to transmit. The will be collected by the File-Collector via FTPes or SFTP.

For demo purpose we have now created a trigger inside which is sending such a file ready event. To do this you have to do the following steps.

  * Open the ConfigApp with the gNodeB
```
http://localhost:8080/odlux/index.html#/configuration/{node-id}/_3gpp-common-managed-element:ManagedElement[OAI%20gNodeB]
```
  * _3gpp-common-managed-element > Attributes
  * click edit on the top right corner
  * set the userLabel property to ```pmData``` and click save to trigger the ```file-ready``` event once.
  * afterwards you have to reset it to ```disabled``` and back if you want to trigger it again
  * up to now the file and its data is hardcoded. But it shows that the workflow is working

