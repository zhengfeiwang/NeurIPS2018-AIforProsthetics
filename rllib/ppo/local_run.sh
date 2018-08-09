#!/bin/bash
python main.py \
--frameskip=5 --accuracy=1e-3 \
--num-workers=24 --num-cpus=24 \
--num-gpus=4 \
--sample=2048  --sample-batch=16 \
--reward=standing \
--epochs=10 --hiddens=256-256 --activations=relu \
--batch-size=256 --learning-rate=0.0001 \
--seed=60730 \
--iterations=500 --checkpoint-interval=50 --validation-interval=1
