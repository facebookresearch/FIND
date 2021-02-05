# Copyright (c) Facebook, Inc. and its affiliates.

for task in add-or-mul count-or-mem hierar-or-linear comp-or-mem; do
    pushd $task
    sh generate.sh
    popd
done
