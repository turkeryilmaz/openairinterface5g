#!/bin/bash

function die() { echo $@ 1>&2; exit 1; }

[ $# -gt 1 ] && die "error: only one target file supported"

targetfile=targets.ninja
[ $# -eq 1 ] && targetfile=$1
[ ! -f $targetfile ] && die "file $targetfile: no such file or directory"
ALL_TARGETS=$(cat $targetfile)
echo "using targets in file $targetfile"

rm -rf results{,.txt}
mkdir results

function build_target() {
  [ $# -eq 1 ] || die "need target"
  local target=$1
  local dir=$(mktemp -p. -d)
  ( cmake -GNinja -B ${dir} && ninja -C ${dir} ${target} ) > results/${target}.log 2>&1
  local result=$([ $? -eq 0 ] && echo built || echo failed)
  echo -e "$result\t$t" | tee -a results.txt
  rm -rf ${dir}
}
export -f build_target

parallel -j4 build_target ::: "${ALL_TARGETS}"
