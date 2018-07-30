#!/bin/bash
python train.py --num-workers=48 --num-cpus=48 \
--actor-activation=selu --critic-activation=selu \
--noise-level=0.3 \
--reward-type=shaped --gpu \
--checkpoint-interval=10 \