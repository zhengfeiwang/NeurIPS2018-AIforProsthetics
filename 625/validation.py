import os
import time
import random
import argparse
import numpy as np
import torch
from osim.env import ProstheticsEnv
from ddpg import DDPG
from evaluator import Evaluator


def val(nb_iterations, agent, env, evaluator):
    step = 0

    while step <= nb_iterations:
        validation_reward = evaluator(env, agent.select_action, visualize=False)
        print('[validation] reward={}'.format(np.mean(validation_reward)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDPG implemented by PyTorch')
    parser.add_argument('--discount', default=0.99, type=float, help='bellman discount')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--memory_size', default=1000000, type=int, help='memory size')

    parser.add_argument('--hidden1', default=400, type=int, help='number of first fully connected layer')
    parser.add_argument('--hidden2', default=300, type=int, help='number of second fully connected layer')
    parser.add_argument('--actor_lr', default=1e-4, type=float, help='actor learning rate')
    parser.add_argument('--critic_lr', default=1e-3, type=float, help='critic learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='minibatch size')

    parser.add_argument('--iterations', default=2000000, type=int, help='iterations during training')
    parser.add_argument('--warmup', default=100, type=int, help='timestep without training to fill the replay buffer')
    parser.add_argument('--apply_noise', dest='apply_noise', default=True, action='store_true', help='apply noise to the action')
    parser.add_argument('--validate_interval', default=10, type=int, help='how many episodes to validate')
    parser.add_argument('--save_interval', default=100, type=int, help='how many episodes to save model')

    parser.add_argument('--validation_episodes', default=1, type=int, help='number of episodes during validation')
    parser.add_argument('--checkpoint_interval', default=100, type=int, help='episodes interval to save model')
    parser.add_argument('--output', default='output', type=str)
    parser.add_argument('--visualization', dest='visualization', action='store_true')
    parser.add_argument('--cuda', dest='cuda', action='store_true')

    parser.add_argument('--seed', default=-1, type=int, help='random seed')

    parser.add_argument('--id', default=-1, type=int, help='id of loading weights')

    args = parser.parse_args()

    env = ProstheticsEnv(visualize=True)

    # set random seed
    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        env.seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    # states and actions space
    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]

    evaluator = Evaluator(args)

    agent = DDPG(nb_states, nb_actions, args)
    assert args.id is not -1
    agent.load_model(args.output, args.id)
    val(args.iterations, agent, env, evaluator)
