# Copyright (c) Facebook, Inc. and its affiliates.

for len in 10 20 30 40 ; do
  python generate_data.py --train_length=$len --test_span=10
done
