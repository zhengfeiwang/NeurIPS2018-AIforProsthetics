#!/bin/bash
python train.py \
# train parameters
--seed=60730 \
--validation-interval=2 --checkpoint-interval=5 \
# cluster
--cluster --num-workers=120 \
# local machine
--num-cpus=48 --num-workers=48 \
--gpu --num-gpus=4 \
# environment settings
--integrator-accuracy=1e-3 \
# human reaction limit is 0.1s, 10steps
--action-repeat=5 \
# algorithms parameters
--reward-type=shaped \
--gamma=0.995 --kl-coeff=1.0 \
--timesteps-per-batch=1024 \
--epoch=20 --batch-size=256 --stepsize=0.0001
