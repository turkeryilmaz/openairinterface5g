#!/bin/bash

function die() {
  echo $@
  exit 1
}

[ $# -ge 3 -a $# -le 4 ] || die "usage: $0 <directory> <repository> <ref> [<merge-ref>]"

set -ex

dir=$1
repo=$2
ref=$3
merge=$4

rm -rf ${dir}
git clone --filter=blob:none -n ${repo} ${dir}
cd ${dir}
git config user.email "jenkins@openairinterface.org"
git config user.name "OAI Jenkins"
git config advice.detachedHead false
mkdir -p cmake_targets/log
git checkout -f ${ref}
if [ -n "${merge}" ]; then
  git fetch origin ${merge} || die "Failed to fetch ${merge}"
  git merge --ff FETCH_HEAD -m "Temporary merge for CI"
fi
exit 0
