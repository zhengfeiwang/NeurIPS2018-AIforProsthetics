#!/bin/bash
python train.py --action-repeat=4 --actor-activation="selu" --critic-activation="selu" --num-cpus=4