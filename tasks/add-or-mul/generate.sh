# Copyright (c) Facebook, Inc. and its affiliates.

for len in 5 10 15 20; do
  python generate_data.py --train_length=$len --test_span=3
done
