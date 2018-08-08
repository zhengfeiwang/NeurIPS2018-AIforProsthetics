#!/bin/bash
python train.py \
--seed=60730 \
--validation-interval=2 --checkpoint-interval=5 \
--num-cpus=16 --num-workers=16 \
--gpu --num-gpus=8 \
--integrator-accuracy=1e-3 \
--action-repeat=5 \
--reward-type=shaped \
--gamma=0.995 --kl-coeff=1.0 \
--timesteps-per-batch=1024 \
--epoch=10 --batch-size=256 --stepsize=0.0001
