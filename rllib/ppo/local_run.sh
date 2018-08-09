#!/bin/bash
python main.py \
--frameskip=5 --accuracy=1e-3 \
--num-workers=16 --num-cpus=16 \
--num-gpus=8 \
--sample=1024 \
--reward=standing \
--epochs=10 --hiddens=256-256 --activations=relu \
--batch-size=256 --learning-rate=0.0001 \
--seed=60730 \
--iterations=10000 --checkpoint-interval=50 --validation-interval=1
