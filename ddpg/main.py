import os
import time
import random
import argparse
import logging
import numpy as np
import torch
import gym
from tensorboardX import SummaryWriter
from ddpg import DDPG
from train import train
from my_env import ACTION_SPACE, MY_OBSERVATION_SPACE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDPG implemented by PyTorch')
    parser.add_argument('--discount', default=0.99, type=float, help='bellman discount')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--memory-size', default=1000000, type=int, help='memory size')
    parser.add_argument('--action-repeat', default=5, type=int, help='repeat times for each action')
    parser.add_argument('--reward-type', default='2018', type=str, help='reward type')
    parser.add_argument("--accuracy", default=5e-5, type=float, help="simulator integrator accuracy")

    parser.add_argument('--hidden1', default=400, type=int, help='number of first fully connected layer')
    parser.add_argument('--hidden2', default=300, type=int, help='number of second fully connected layer')
    parser.add_argument('--actor-lr', default=1e-4, type=float, help='actor learning rate')
    parser.add_argument('--critic-lr', default=1e-3, type=float, help='critic learning rate')
    parser.add_argument('--batch-size', default=64, type=int, help='minibatch size')
    parser.add_argument('--nb-train-steps', default=50, type=int, help='training times per episode')
    parser.add_argument('--max-episode-length', default=300, type=int, help='maximum episode length')

    parser.add_argument('--nb-iterations', default=2000000, type=int, help='iterations during training')
    parser.add_argument('--warmup', default=50, type=int, help='iteration without training to fill the replay buffer')
    parser.add_argument('--apply-noise', default=False, action='store_true', help='apply noise to the action')
    parser.add_argument('--noise-level', default=0.1, type=float, help='noise level for action')
    parser.add_argument('--validation-interval', default=10, type=int, help='episode interval for validation')
    parser.add_argument('--checkpoint-interval', default=100, type=int, help='episode interval for checkpoint')
    parser.add_argument('--validation-episodes', default=1, type=int, help='validation episodes')

    parser.add_argument('--resume', default=None, type=str, help='resuming model path')
    parser.add_argument('--resume-num', default=-1, type=int, help='number of the weight to load')
    parser.add_argument('--output', default='output', type=str, help='summary and logs output directory')
    parser.add_argument('--cuda', default=False, action='store_true', help='use CUDA to acceleration')
    parser.add_argument('--num-workers', default=1, type=int, help='number of workers for parallelism')

    parser.add_argument('--seed', default=-1, type=int, help='random seed')

    args = parser.parse_args()

    # summary file
    timestruct = time.localtime(time.time())
    timestamp = time.strftime('%Y-%m-%d %H-%M-%S', timestruct)
    writer = SummaryWriter(os.path.join(args.output, timestamp))

    # logger file
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(os.path.join(args.output, timestamp + '.log'))
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s,%(name)s,%(levelname)s,%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)

    # set random seed
    if args.seed is -1:
        args.seed = np.random.randint(0, 2**32)
    logger.info('random seed: {}'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # states and actions space
    nb_states = MY_OBSERVATION_SPACE
    nb_actions = ACTION_SPACE

    agent = DDPG(nb_states, nb_actions, args)

    logger.debug('---------------- Experiment Setup ----------------')
    logger.debug('discount factor: {}, tau: {}'.format(args.discount, args.tau))
    logger.debug('replay buffer size: {}'.format(args.memory_size))
    logger.debug('action repeat: {}'.format(args.action_repeat))
    logger.debug('reward type: {}'.format(args.reward_type))
    logger.debug('--------------------------------------------------')
    logger.debug('actor learning rate: {}, critic learning rate: {}'.format(args.actor_lr, args.critic_lr))
    logger.debug('training epochs: {}, batch size: {}'.format(args.nb_train_steps, args.batch_size))
    logger.debug('--------------------------------------------------')
    logger.debug('warmup: {}, iteration: {}'.format(args.warmup, args.nb_iterations))
    if args.apply_noise is True:
        logger.debug('apply noise, noise level: {}'.format(args.noise_level))
    logger.debug('--------------------------------------------------')

    # resume train
    if args.resume is not None and args.resume_num is not -1:
        logger.info('resume train, load weight file: {}...'.format(args.resume_num))
        agent.load_model(args.output, args.resume_num)

    train(agent, writer, logger, args)
