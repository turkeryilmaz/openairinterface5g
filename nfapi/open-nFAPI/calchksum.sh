#!/bin/bash

#Copy the native include files to tmp_chksum folder
mkdir /tmp/chksum
cp ./nfapi/public_inc/nfapi_interface.h ./nfapi/public_inc/nfapi_nr_interface.h ./nfapi/public_inc/nfapi_nr_interface_scf.h ./sim_common/inc/vendor_ext.h /tmp/chksum/.

#Calculate checksum of native nfapi include files
new_chksum=$(find /tmp/chksum/nfapi_interface.h /tmp/chksum/nfapi_nr_interface.h /tmp/chksum/nfapi_nr_interface_scf.h /tmp/chksum/vendor_ext.h -type f -exec md5sum {} \; | sort -k 2 | md5sum | cut -c1-32)

echo "The checksum: $new_chksum"

#Remove the temporary folder
rm -rf /tmp/chksum

#Check whether version file exist else exit
 [ -f ./version.json ] && echo "Checking version.json file" || { echo "version.json file does not exist" ; exit 1; }

#Get the version and checksum from version.json file
old_chksum=$(jq '.checksum' version.json | cut -c2-33)
version=$(jq '.version' version.json | sed 's/\"//g' | sed 's/\.//g')

#Compare the calculated checksum with old checksum(version.json). If value is different, update the version.json file
if [[ "$new_chksum" != "$old_chksum" ]]; then
        echo "{" >version.json;
	version=$((version+1));
	ver1=$((version/100));
	ver2=$(((version%100)/10));
	ver3=$((version%10));
	echo "        \"version\": \"$ver1.$ver2.$ver3\"," >>version.json;
        echo "        \"checksum\": \"$new_chksum\"" >>version.json;            
        echo "}" >>version.json;
	echo "Updated the version.json file";
else
	echo "No update required to version.json file";
fi

