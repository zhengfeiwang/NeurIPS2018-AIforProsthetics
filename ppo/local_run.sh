#!/bin/bash
python train.py --num-workers=48 --num-cpus=48 \
--reward-type=hopper --gpu --num-gpus=4 \
--gamma=0.995 --kl-coeff=1.0 \
--action-repeat=5 \
--timesteps-per-batch=1024 \
--epoch=20 --batch-size=256 --stepsize=0.0001 \
--validation-interval=2 --checkpoint-interval=5 \
--seed=60730