import os
import sys
import time
import random
import argparse
import numpy as np
import torch
from osim.env import ProstheticsEnv
from tensorboardX import SummaryWriter
from ddpg import DDPG
from training import train
from evaluator import Evaluator
from observation_process import obs_process


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--discount', default=0.99, type=float, help='bellman discount')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--memory_size', default=1000000, type=int, help='memory size')
    parser.add_argument('--action_repeat', default=3, type=int, help='repeat times for each action')

    parser.add_argument('--hidden1', default=64, type=int, help='number of first fully connected layer')
    parser.add_argument('--hidden2', default=64, type=int, help='number of second fully connected layer')
    parser.add_argument('--actor_lr', default=1e-4, type=float, help='actor learning rate')
    parser.add_argument('--critic_lr', default=1e-3, type=float, help='critic learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')
    parser.add_argument('--nb_train_steps', default=50, type=int, help='training times per train')

    parser.add_argument('--iterations', default=1000000, type=int, help='iterations during training')
    parser.add_argument('--warmup', default=200, type=int, help='timestep without training to fill the replay buffer')
    parser.add_argument('--apply_noise', dest='apply_noise', default=True, action='store_true', help='apply noise to the action')
    parser.add_argument('--validation_interval', default=20, type=int, help='episode interval to validation')
    parser.add_argument('--save_interval', default=100, type=int, help='episode interval to save model')
    parser.add_argument('--validation_episodes', default=10, type=int, help='how many episodes to validation')
    parser.add_argument('--max_episode_length', default=500, type=int, help='maximum episode length')

    parser.add_argument('--resume', default=None, type=str, help='resuming model path')
    parser.add_argument('--resume_num', default=1, type=int, help='number of the weight to load')
    parser.add_argument('--output', default='output', type=str)
    parser.add_argument('--cuda', dest='cuda', action='store_true')

    parser.add_argument('--seed', default=-1, type=int, help='random seed')

    args = parser.parse_args()

    # TensorBoardX summary file
    timestamp = time.time()
    timestruct = time.localtime(timestamp)
    writer = SummaryWriter(os.path.join(args.output,  'Prosthetics@' + time.strftime('%Y-%m-%d %H-%M-%S', timestruct)))

    env = ProstheticsEnv(visualize=False)

    # set random seed
    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    # states and actions space
    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]

    evaluator = Evaluator(args)

    agent = DDPG(nb_states, nb_actions, args)

    # resume train
    if args.resume is not None:
        print('resume train, load weight file: {}...'.format(args.resume_num))
        agent.load_model(args.output, args.resume_num)

    train(agent, env, evaluator, writer, args)
