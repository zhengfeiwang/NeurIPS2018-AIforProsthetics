#!/bin/bash
python train.py --num-workers=48 --num-cpus=48 \
--reward-type=shaped --gpu --num-gpus=4 \
--binary-action \
--checkpoint-interval=10 \