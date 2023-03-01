#!/bin/bash
OSBASE="ubuntu22"
TIMESTAMP=$(date -u +'%Y%m%dT%H%M%SZ')
TIMESTAMP_TAG="$OSBASE-$TIMESTAMP"
ONLY_DEV_BUILD="false"
ALL_STAGES="false"
until [ -z "$1" ]
do
    case "$1" in 
        --all)
            ALL_STAGES="true"
            shift
            ;;

        --dev)
            ONLY_DEV_BUILD="true"
            shift
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;

        *)
            shift
            ;;
    esac
done



if $ALL_STAGES; then
    docker build $NO_CACHE --target ran-base --tag ran-base:$OSBASE-latest --file docker/Dockerfile.base.$OSBASE .
    if [ ! $? -eq 0 ]; then
        exit 1
    fi
fi
if $ONLY_DEV_BUILD; then
    docker build $NO_CACHE --target ran-dev --build-arg BASETAG=$OSBASE-latest --tag ran-dev:$OSBASE-latest --file docker/Dockerfile.dev.$OSBASE .
    if [ ! $? -eq 0 ]; then
        exit 1
    fi
    exit 0
fi
docker build $NO_CACHE --target ran-build --build-arg BASETAG=$OSBASE-latest --tag ran-build:$OSBASE-latest --file docker/Dockerfile.build.$OSBASE .
if [ ! $? -eq 0 ]; then
    exit 1
fi
docker build $NO_CACHE --target oai-gnb --build-arg BASETAG=$OSBASE-latest --tag oai-gnb:$OSBASE-latest --tag oai-gnb:$TIMESTAMP_TAG --file docker/Dockerfile.gNB.ubuntu22 .
if [ ! $? -eq 0 ]; then
        exit 1
fi
