# Copyright (c) Facebook, Inc. and its affiliates.

for N in 5 10 20 ; do
  python generate_data.py --train_N=$N
done
