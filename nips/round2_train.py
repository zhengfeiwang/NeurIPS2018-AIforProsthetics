#!/usr/bin/env python

import argparse
import ray
import tensorflow as tf
from baselines.logger import configure
from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_normalize import VecNormalize
import baselines.ppo2.policies as policies
import baselines.ppo2.ppo2 as ppo2
from nips.round2_env import CustomEnv, CustomActionWrapper
from nips.remote_vec_env import RemoteVecEnv


def create_env():
    env = CustomEnv(visualization=args.vis, integrator_accuracy=args.accuracy)
    env = CustomActionWrapper(env)
    return env


def train():
    config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=args.num_cpus,
        inter_op_parallelism_threads=args.num_cpus
    )
    tf.Session(config=config).__enter__()

    env = RemoteVecEnv([create_env] * args.num_cpus)
    env = VecNormalize(env, ret=True, gamma=args.gamma)

    ppo2.learn(
        policy=policies.MlpPolicy, env=env,
        total_timesteps=args.num_timesteps, nminibatches=args.num_minibatches,
        nsteps=args.num_steps, noptepochs=args.num_epochs, lr=args.learning_rate,
        gamma=args.gamma,
        lam=args.lam, ent_coef=args.ent_coef, vf_coef=args.vf_coef, cliprange=args.clip_range,
        log_interval=args.log_interval, save_interval=args.save_interval,
        load_path=args.checkpoint_path,
        num_casks=args.num_casks
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO training for NIPS 2018: AI for Prosthetics Challenge, Round 2')
    parser.add_argument('--num-timesteps', default=1e6, type=int, help='number of timesteps')
    # batch_size = num_steps * num_envs
    parser.add_argument('--num-steps', default=128, type=int, help='number of steps per update')
    # valid number of envs
    parser.add_argument('--num-minibatches', default=1, type=int, help='number of training minibatches per update')
    parser.add_argument('--num-epochs', default=4, type=int, help='number of training epochs per update')
    parser.add_argument('--learning-rate', default=3e-4, type=float, help='learning rate')
    # RL domain
    parser.add_argument('--gamma', default=0.99, type=float, help='discounting factor')
    # PPO specific
    parser.add_argument('--lam', default=0.95, type=float, help='advantage estimation discounting factor')
    parser.add_argument('--ent-coef', default=0.001, type=float, help='policy entropy coefficient')
    parser.add_argument('--vf-coef', default=0.5, type=float, help='value function loss coefficient')
    parser.add_argument('--clip-range', default=0.2, type=float, help='clipping range')
    # env related
    parser.add_argument('--seed', default=60730, type=int, help='random seed')
    parser.add_argument('--accuracy', default=5e-5, type=float, help='simulator integrator accuracy')
    parser.add_argument('--vis', default=False, action='store_true', help='visualization option')
    # training settings
    parser.add_argument('--num-cpus', default=1, type=int, help='number of cpus')
    parser.add_argument('--num-casks', default=0, type=int, help='number of casks, for acceleration')
    parser.add_argument('--log-dir', default='./logs', type=str, help='logging events output directory')
    parser.add_argument('--log-interval', default=1, type=int, help='number of timesteps between logging events')
    parser.add_argument('--save-interval', default=1, type=int, help='number of timesteps between saving events')
    parser.add_argument('--checkpoint-path', default=None, type=str, help='path to load the model checkpoint from')
    args = parser.parse_args()
    print(args)

    ray.init()
    set_global_seeds(args.seed)
    configure(dir=args.log_dir)
    train()
