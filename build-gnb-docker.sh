#!/bin/bash
OSBASE="ubuntu22"
TIMESTAMP=$(date -u +'%Y%m%dT%H%M%SZ')
TIMESTAMP_TAG="$OSBASE-$TIMESTAMP"
BUILD_RAN_BASE="false"
BUILD_RAN_BUILD="false"
BUILD_RAN_DEV="false"
BUILD_RAN_GNB="false"

until [ -z "$1" ]
do
    case "$1" in 
        --all)
            BUILD_RAN_BASE="true"
            BUILD_RAN_BUILD="true"
            BUILD_RAN_DEV="true"
            BUILD_RAN_GNB="true"
            shift
            ;;

        --base)
            BUILD_RAN_BASE="true"
            shift
            ;;

        --build)
            BUILD_RAN_BUILD="true"
            shift
            ;;

        --dev)
            BUILD_RAN_DEV="true"
            shift
            ;;

        --gnb)
            BUILD_RAN_GNB="true"
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

if ! $BUILD_RAN_BUILD && ! $BUILD_RAN_GNB && ! $BUILD_RAN_DEV && ! $BUILD_RAN_GNB; then
    BUILD_RAN_BUILD="true"
    BUILD_RAN_GNB="true"
fi

if $BUILD_RAN_BASE; then
    docker build $NO_CACHE --target ran-base --tag ran-base:$OSBASE-latest --file docker/Dockerfile.base.$OSBASE .
    if [ ! $? -eq 0 ]; then
        exit 1
    fi
fi
if $BUILD_RAN_BUILD; then
    docker build $NO_CACHE --target ran-build --build-arg BASETAG=$OSBASE-latest --tag ran-build:$OSBASE-latest --file docker/Dockerfile.build.$OSBASE .
    if [ ! $? -eq 0 ]; then
        exit 1
    fi
fi
if $BUILD_RAN_DEV; then
    docker build $NO_CACHE --target ran-dev --build-arg BASETAG=$OSBASE-latest --tag ran-dev:$OSBASE-latest --file docker/Dockerfile.dev.$OSBASE .
    if [ ! $? -eq 0 ]; then
        exit 1
    fi
fi
if $BUILD_RAN_GNB; then
    docker build $NO_CACHE --target oai-gnb --build-arg BASETAG=$OSBASE-latest --tag oai-gnb:$OSBASE-latest --tag oai-gnb:$TIMESTAMP_TAG --file docker/Dockerfile.gNB.ubuntu22 .
    if [ ! $? -eq 0 ]; then
            exit 1
    fi
fi
