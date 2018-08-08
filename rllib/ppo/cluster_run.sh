#!/bin/bash
python train.py \
--seed=60730 \
--validation-interval=2 --checkpoint-interval=5 \
--cluster --num-workers=120 \
--gpu --num-gpus=4 \
--integrator-accuracy=1e-3 \
--action-repeat=5 \
--reward-type=shaped \
--gamma=0.995 --kl-coeff=1.0 \
--timesteps-per-batch=2048 \
--epoch=10 --batch-size=256 --stepsize=0.0001
