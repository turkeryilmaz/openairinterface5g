#!/bin/bash

DIRBIN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DIR_AVAILABLE_YANGS="$DIRBIN/available-yangs"
DIR_YANGS="$DIRBIN/yang"
FILE_DOWNLOADZIP="$DIRBIN/yangs.zip"
TMPFOLDER="/tmp/o1-yangs"
UNZIP=$(which unzip)
if [ -z "$UNZIP" ]; then
    echo "unable to find unzip. please install."
    exit 1
fi

# download
if [ ! -f "$FILE_DOWNLOADZIP" ]; then
    wget -O "$FILE_DOWNLOADZIP" "https://forge.3gpp.org/rep/sa5/MnS/-/archive/Rel-18/MnS-Rel-18.zip?path=yang-models"
fi
if [ ! -d "$DIR_AVAILABLE_YANGS" ]; then
    mkdir "$DIR_AVAILABLE_YANGS"
fi
# cleanup yang folders
rm -rf "$DIR_AVAILABLE_YANGS/"*
rm "$DIR_YANGS/*"
unzip -uj yangs.zip -d "$TMPFOLDER"
cp -r "$TMPFOLDER/"* "$DIR_AVAILABLE_YANGS/"

rm "$FILE_DOWNLOADZIP"

# fill yang folder
cp "$DIR_AVAILABLE_YANGS/"_3gpp-common*.yang "$DIR_YANGS"
cp "$DIR_AVAILABLE_YANGS/"ietf-*.yang "$DIR_YANGS/"

declare special_files=(
    "_3gpp-5g-common-yang-types.yang"
    "_3gpp-5gc-nrm-configurable5qiset.yang"
    "_3gpp-nr-nrm-gnbcucpfunction.yang"
    "_3gpp-nr-nrm-gnbcuupfunction.yang"
    "_3gpp-nr-nrm-gnbdufunction.yang"
)

for file in "${special_files[@]}"
do
   cp "$DIR_AVAILABLE_YANGS/$file" "$DIR_YANGS/"
done