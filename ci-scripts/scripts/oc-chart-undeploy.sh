#!/bin/bash

set -e

function die() { echo $@; exit 1; }
[ $# -eq 3 ] || die "usage: $0 <path-to-dir> <namespace> <log-dir>"

OC_DIR=${1}
OC_NS=${2}
LOG_DIR=${3}
OC_RELEASE=$(basename "${OC_DIR}")

cat ${OC_DIR}/oc-password | oc login -u oaicicd --server https://api.oai.cs.eurecom.fr:6443 > /dev/null
oc project ${OC_NS} > /dev/null
oc describe pod > ${LOG_DIR}/describe-pods-post-test.log
oc get pods.metrics.k8s &> ${LOG_DIR}/nf-resource-consumption.log
oc logs -l app.kubernetes.io/name=${OC_RELEASE} --tail=-1 > ${LOG_DIR}/${OC_RELEASE}.log
helm uninstall ${OC_RELEASE} --wait
oc logout > /dev/null
