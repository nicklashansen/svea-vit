#!/bin/bash

GPU=0
SEEDS=5

for ((SEED=1;SEED<=SEEDS;SEED++)); do
    CUDA_VISIBLE_DEVICES=$GPU python3 src/train.py --svea_aug none --exp_suffix none --seed $SEED
done
