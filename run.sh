#!/usr/bin/env bash
python -m round2.round2_train \
    --num-timesteps=10000000 \
    --num-steps=128 \
    --num-minibatches=24 \
    --num-cpus=30 --num-casks=6 \
    --save-interval=5 \
    --seed=30