#!/bin/bash
python train.py --num-workers=24 --num-cpus=28 \
--actor-hiddens=800-400 --critic-hiddens=800-400 \
--actor-activation=selu --critic-activation=selu \
--reward-shaping 