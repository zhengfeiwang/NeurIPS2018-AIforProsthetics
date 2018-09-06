#!/usr/bin/env python

import baselines.ppo2.ppo2 as ppo2
import baselines.ppo2.policies as policies
from nips.custom_env import make_env
from nips.remote_vec_env import RemoteVecEnv
from baselines.logger import configure
from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_normalize import VecNormalize

import tensorflow as tf
import ray


def train(num_timesteps, seed):
    num_cpus = 1
    num_casks = 1
    num_cpus += num_casks

    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=num_cpus,
                            inter_op_parallelism_threads=num_cpus)
    tf.Session(config=config).__enter__()

    gamma = 0.95

    env = RemoteVecEnv([make_env] * num_cpus)
    env = VecNormalize(env, ret=True, gamma=gamma)

    set_global_seeds(seed)
    policy = policies.MlpPolicy
    ppo2.learn(policy=policy,
               env=env,
               nsteps=128,
               nminibatches=num_cpus-num_casks,
               lam=0.95,
               gamma=gamma,
               noptepochs=4,
               log_interval=1,
               vf_coef=0.5,
               ent_coef=0.0,
               lr=3e-5,
               cliprange=0.2,
               save_interval=2,
               load_path="./logs/course_5/00114",
               total_timesteps=num_timesteps,
               num_casks=num_casks)


if __name__ == "__main__":
    ray.init()
    configure(dir="./logs")
    train(int(1e6), 60730)
