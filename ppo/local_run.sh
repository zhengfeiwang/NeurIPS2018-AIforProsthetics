#!/bin/bash
python train.py --num-workers=48 --num-cpus=48 \
--reward-type=shaped --gpu --num-gpus=4 \
--action-repeat=5 \
--validation-interval=2 --checkpoint-interval=5 \
--seed=60730