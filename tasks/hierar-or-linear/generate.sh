# Copyright (c) Facebook, Inc. and its affiliates.

for d in 4; do
  python generate_data.py --train_depth=$d --test_span=2
done
