#!/bin/bash
python main.py \
--frameskip=5 --accuracy=1e-3 \
--num-workers=2 --num-cpus=2 \
--sample=512 --sample-batch=16 \
--warmup=100 \
--reward=shaped \
--gamma=0.95 \
--batch-size=128 --actor-lr=3e-4 --critic-lr=3e-3 \
--seed=60730 \
--iterations=2 --checkpoint-interval=1 --validation-interval=1
