#!/usr/bin/env python

"""
Train an agent on Sonic using PPO2 from OpenAI Baselines.
"""

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import baselines.ppo2.ppo2 as ppo2
import baselines.ppo2.policies as policies
from custom_env import make_env
from baselines.logger import configure
from baselines import bench, logger
from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_normalize import VecNormalize

import tensorflow as tf


def validate(num_timesteps, seed):
    """Run PPO until the environment throws an exception."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    set_global_seeds(seed)
    with tf.Session(config=config):
        # Take more timesteps than we need to be sure that
        # we stop due to an exception.
        ppo2.learn(policy=policies.MlpPolicy,
                   env=DummyVecEnv([make_env]),
                   nsteps=128,
                   nminibatches=16,
                   lam=0.95,
                   gamma=0.99,
                   noptepochs=5,
                   log_interval=1,
                   vf_coef=0.5,
                   ent_coef=0.0,
                   lr=lambda _: 3e-4,
                   cliprange=lambda _: 0.2,
                   save_interval=10,
                   load_path="./logs/history-0/00330",
                   total_timesteps=num_timesteps
                   )


def train(num_timesteps, seed):
    ncpu = 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    env = SubprocVecEnv([make_env] * ncpu)
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = policies.MlpPolicy
    ppo2.learn(policy=policy,
               env=env,
               nsteps=128,
               nminibatches=16,
               lam=0.95,
               gamma=0.99,
               noptepochs=5,
               log_interval=1,
               ent_coef=0.0,
               lr=3e-4,
               cliprange=0.2,
               save_interval=10,
               load_path="./logs/history-0/00330",
               total_timesteps=num_timesteps)


if __name__ == '__main__':
    configure(dir="./logs")
    train(int(1e6), 2)
