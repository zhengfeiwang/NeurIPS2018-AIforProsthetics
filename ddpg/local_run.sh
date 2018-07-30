#!/bin/bash
python train.py --num-workers=48 --num-cpus=48 \
--actor-activation=selu --critic-activation=selu \
--actor-learning-rate=3e-4 --critic-learning-rate=3e-4 \
--reward-type=shaped --gpu \
--checkpoint-interval=10 \