#!/bin/bash

set -e

function die() { echo $@; exit 1; }
[ $# -eq 3 ] || die "usage: $0 <path-to-dir> <namespace> <image-tag>"

OC_DIR=${1}
OC_NS=${2}
IMAGE_TAG=${3}
OC_RELEASE=$(basename "${OC_DIR}")

cat ${OC_DIR}/oc-password | oc login -u oaicicd --server https://api.oai.cs.eurecom.fr:6443 > /dev/null
oc project ${OC_NS} > /dev/null
helm install --wait --timeout 120s ${OC_RELEASE} --set nfimage.version=${IMAGE_TAG} --hide-notes ${OC_DIR}/.
oc logout > /dev/null
