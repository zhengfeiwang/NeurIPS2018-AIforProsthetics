#!/bin/bash
python main.py \
--frameskip=5 --accuracy=1e-3 \
--num-workers=120 --num-cpus=120 \
--cluster \
--gpu --num-gpus=4 \
--sample=1024  --sample-batch=10 \
--reward=shaped \
--gamma=0.997 \
--epochs=10 --hiddens=256-256 --activations=relu \
--batch-size=256 --learning-rate=1e-5 \
--seed=60730 \
--iterations=100 --checkpoint-interval=5 --validation-interval=1
