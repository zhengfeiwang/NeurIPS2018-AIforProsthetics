#!/bin/bash
python main.py \
--frameskip=5 --accuracy=1e-3 \
--num-workers=2 --num-cpus=2 --sample=1024 \
--reward=shaped \
--epochs=1 --hiddens=256-256 --activations=relu \
--seed=60730 --iterations=1 --checkpoint-interval=1 --validation-interval=1 \
