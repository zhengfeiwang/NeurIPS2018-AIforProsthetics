#!/bin/bash
python main.py \
--frameskip=5 --accuracy=1e-3 \
--num-workers=60 \
--cluster --gpu \
--sample=1024 --sample-batch=16 \
--warmup=1000 \
--reward=velocity \
--gamma=0.95 \
--batch-size=128 --actor-lr=3e-4 --critic-lr=3e-3 \
--seed=60730 \
--iterations=100 --checkpoint-interval=5 --validation-interval=1
