# NIPS 2018: AI for Prosthetics Challenge Solution

A solution based on OpenAI baselines ([repo](https://github.com/openai/baselines)) for 9th place 
[NIPS 2018: AI for Prosthetics Challenge](https://www.crowdai.org/challenges/nips-2018-ai-for-prosthetics-challenge)

## Table of Contents
* [Prerequisites](#prerequisites)
* [Training](#training)
* [Evaluation](#evaluation)
* [Contributing](#contributing)

## Prerequisites

First of all, you should follow the challenge environment's instruction to set up the simulator, 
details described in: 
https://github.com/stanfordnmbl/osim-rl/  
```bash
source activate opensim-rl
```
Other dependencies is required as follow:
- Ray
- TensorFlow
- mpi4py, cloudpickle, joblib (required by baselines)

## Training

To train the model from scratch, you need to comment the `--checkpoint-path` in `run.sh`, 
or type commands as follow:
```bash
python -m nips.round2_train \
    --num-timesteps=10000000 \
    --num-steps=128 \
    --num-minibatches=16 \
    --num-cpus=24 --num-casks=8 \
    --save-interval=5 \
    --repeat=2
```
The baseline code shall be the minimal for this challenge, 
and we select PPO ([paper](https://arxiv.org/abs/1707.06347)) as the algorithm.  
You should specify your available CPUs and GPUs in the bash script. The training will be deployed 
distributed by Ray ([docs](https://ray.readthedocs.io/en/latest/index.html)).  

## Evaluation

To evaluate your model, store your checkpoints file into `checkpoints/` and change 
the arguments in script.  
```bash
# remember to place your checkpoints in your script file first
./run.sh
```
As the comparison, our model finally get 9813.224 in the Round 2 of challenge.

## Contributing

- [wangzhengfei0730](https://github.com/wangzhengfei0730)
- [QPHutu](https://github.com/QPHutu)
