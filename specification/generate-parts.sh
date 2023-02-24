#!/bin/bash
DIRBIN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
pyang=$(which pyang)
yanglint=$(which yanglint)

OUTPUT_GNODEB="$DIRBIN/xml/gnodeb"
OUTPUT_CU="$DIRBIN/xml/cu"
OUTPUT_DU="$DIRBIN/xml/du"

if [ -z "$pyang" ]; then
    echo "no executable for pyang forund. please install with pip3 install pyang"
    exit 1
fi
if [ ! -d "$DIRBIN/xml" ]; then
    mkdir "$DIRBIN/xml"
fi

case "$1" in

    validate)
        $pyang --strict "$DIRBIN/yang/"*.yang
        if [ $? -eq 0 ]; then
            echo "yang files are consistent"
        else
            echo "there is an inconsistency in the yang files"
        fi
        $yanglint --strict -p "$DIRBIN/yang/" "$DIRBIN/xml/gnodeb_filled.xml"
        #$pyang --lint "$DIRBIN/yang/"*.yang
        
    ;;
    cu)
        $pyang "$DIRBIN/yang/"*common*.yang \
            "$DIRBIN/yang/"ietf*.yang \
            "$DIRBIN/yang/_3gpp-nr-nrm-gnbcucpfunction.yang" \
            "$DIRBIN/yang/_3gpp-nr-nrm-gnbcuupfunction.yang" \
             -f tree > $OUTPUT_CU.tree
        $pyang "$DIRBIN/yang/"*common*.yang \
            "$DIRBIN/yang/"ietf*.yang \
            "$DIRBIN/yang/_3gpp-nr-nrm-gnbcucpfunction.yang" \
            "$DIRBIN/yang/_3gpp-nr-nrm-gnbcuupfunction.yang" \
            -f sample-xml-skeleton \
            --sample-xml-skeleton-defaults \
            --sample-xml-skeleton-annotations > $OUTPUT_CU.xml
    ;;
    du)
        $pyang "$DIRBIN/yang/"*common*.yang \
            "$DIRBIN/yang/"ietf*.yang \
            "$DIRBIN/yang/_3gpp-nr-nrm-gnbdufunction.yang" \
             -f tree > $OUTPUT_DU.tree
        $pyang "$DIRBIN/yang/"*common*.yang \
            "$DIRBIN/yang/"ietf*.yang \
            "$DIRBIN/yang/_3gpp-nr-nrm-gnbdufunction.yang" \
            -f sample-xml-skeleton \
            --sample-xml-skeleton-defaults \
            --sample-xml-skeleton-annotations > $OUTPUT_DU.xml
    ;;
    gnodeb|*)
        $pyang "$DIRBIN/yang/"*.yang -f tree > $OUTPUT_GNODEB.tree
        $pyang "$DIRBIN/yang/"*.yang  \
            -f sample-xml-skeleton \
            --sample-xml-skeleton-defaults \
            --sample-xml-skeleton-annotations > $OUTPUT_GNODEB.xml
    ;;

esac;


