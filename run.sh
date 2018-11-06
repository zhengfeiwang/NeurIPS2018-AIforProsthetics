#!/usr/bin/env bash
python -m nips.round2_train \
    --num-timesteps=1000000 \
    --num-steps=256 \
    --num-minibatches=16 \
    --num-cpus=24 --num-casks=8 \
    --num-gpus=1 \
    --save-interval=1 \
    --seed=30 --repeat=2 \
    --checkpoint-path=checkpoints/00000
