#!/usr/bin/env bash
python -m nips.round2_train \
    --num-timesteps=1000000 \
    --num-steps=128 \
    --num-minibatches=16 \
    --num-cpus=20 --num-casks=4 \
    --num-gpus=4 \
    --save-interval=5 \
    --seed=30 --repeat=2
