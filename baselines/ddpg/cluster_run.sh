#!/bin/bash
python train.py --redis-address=192.168.1.137:16379 --num-workers=72 \
--actor-activation=selu --critic-activation=selu \
--reward-shaping