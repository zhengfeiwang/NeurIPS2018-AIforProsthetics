#!/usr/bin/env bash
python -m round2.round2_train \
    --num-timesteps=1000000 \
    --num-steps=1024 \
    --num-minibatches=16 \
    --num-cpus=24 --num-casks=8 \
    --seed=60730