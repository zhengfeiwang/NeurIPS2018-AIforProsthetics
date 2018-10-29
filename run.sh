#!/usr/bin/env bash
python -m nips.round2_train \
    --num-timesteps=10000000 \
    --num-steps=512 \
    --num-minibatches=24 \
    --num-cpus=30 --num-casks=6 \
    --save-interval=5 \
    --seed=30 \
    --checkpoint-path=checkpoints/origin/00000