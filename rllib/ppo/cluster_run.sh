#!/bin/bash
python main.py \
--frameskip=5 --accuracy=1e-3 \
--num-workers=60 --num-cpus=60 \
--cluster \
--gpu --num-gpus=4 \
--sample=1024  --sample-batch=16 \
--reward=velocity \
--gamma=0.95 \
--epochs=30 --hiddens=256-256 --activations=selu \
--batch-size=256 --learning-rate=5e-5 \
--seed=60730 \
--iterations=500 --checkpoint-interval=5 --validation-interval=1
