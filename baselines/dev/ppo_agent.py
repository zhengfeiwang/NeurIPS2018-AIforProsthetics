#!/usr/bin/env python
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
import baselines.ppo2.ppo2 as ppo2
import baselines.ppo2.policies as policies
from dev.custom_env import make_env
from baselines.logger import configure
from baselines.common import set_global_seeds

import tensorflow as tf


def train(num_timesteps, seed):
    num_cpus = 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=num_cpus,
                            inter_op_parallelism_threads=num_cpus)
    tf.Session(config=config).__enter__()
    env = SubprocVecEnv([make_env] * num_cpus)
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = policies.MlpPolicy
    ppo2.learn(policy=policy,
               env=env,
               nsteps=256,
               nminibatches=2,
               lam=0.9,
               gamma=0.99,
               noptepochs=10,
               log_interval=1,
               ent_coef=0.0,
               lr=3e-4,
               cliprange=0.2,
               save_interval=10,
               load_path='logs/checkpoints/00000',
               total_timesteps=num_timesteps)


if __name__ == "__main__":
    configure(dir='./logs')
    train(int(1e6), 2)
