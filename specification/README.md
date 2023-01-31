# Yang Specification

## Important

It is not allowed to publish any of the 3gpp yang specifications in any kind of repository or something similar.


Please use the ```get-yangs.sh```  script to download the yang files. Th
All yang spec files to implement will be in the ```yang``` folder. Based on this tree and xml files can be generated with the ```generate-parts.sh``` script.

For the O-RAN O1 interface the 3GPP yang files(see sources) are the entrypoint, especially the ```_3gpp-common-managed-element.yang``` and in there its list of ```ManagedElement```s. These are then extended dependent on their functions with e.g. ```_3gpp-nr-nrm-gnbdufunction.yang```.
n


## Mapping

