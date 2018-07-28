#!/bin/bash
python train.py --num-workers=24 --num-cpus=28 \
--reward-type=shaped --gpu \
--checkpoint-interval=100 \