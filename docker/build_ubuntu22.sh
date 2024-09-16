#!/bin/bash

set -ex

# TODO get branch/tag
export OAI_TAG=imagebuilder-5f57c5e4

# TODO environment var for registry path?
# TODO where to put logs?

# TODO detect if we need to rebuild ran-base
docker compose -f docker-compose.ubuntu.yml up build-ran-base --build
# TODO incremental build?
# TODO should be able to reuse previous image, if requested
# TODO should be able to use host ccache
docker compose -f docker-compose.ubuntu.yml up build-ran-build --pull=never --build
#docker compose -f docker-compose.ubuntu.yml up build-gnb-image --pull=never --build
