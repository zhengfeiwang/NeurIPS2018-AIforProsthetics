#!/bin/bash
python train.py --num-workers=48 --num-cpus=48 \
--actor-activation=selu --critic-activation=selu \
--reward-shaping 