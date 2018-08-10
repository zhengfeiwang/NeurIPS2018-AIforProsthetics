#!/bin/bash
python main.py \
--frameskip=5 --accuracy=1e-3 \
--num-workers=80 --num-cpus=80 \
--cluster \
--num-gpus=4 \
--sample=1024  --sample-batch=16 \
--reward=shaped \
--epochs=10 --hiddens=256-256 --activations=relu \
--batch-size=256 --learning-rate=5e-5 \
--seed=60730 \
--iterations=100 --checkpoint-interval=10 --validation-interval=1
