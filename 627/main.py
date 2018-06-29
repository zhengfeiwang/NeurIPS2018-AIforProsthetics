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
from observation_process import obs_process
from evaluator import Evaluator


def train(nb_iterations, agent, env, evaluator):
    visualization = args.visualization
    log = step = episode = episode_steps = 0
    episode_reward = 0.
    obs = observation = None
    apply_noise = args.apply_noise
    save_times = 0 if args.resume is None else args.resume_num
    action_repeat = args.action_repeat
    max_episode_length = args.max_episode_length // action_repeat
    # max_episode_length = args.max_episode_length
    time_stamp = time.time()

    while step <= nb_iterations:
        if observation is None:
            # get dict version observation from environment
            obs = env.reset(project=False)
            observation = obs_process(obs)
            agent.reset(observation)

        if step <= args.warmup and args.resume is None:
            action = agent.random_action()
        else:
            action = agent.select_action(observation, apply_noise=apply_noise)

        # action repeat
        total_reward = 0.
        for _ in range(action_repeat):
            obs, reward, done, _ = env.step(action, project=False)
            #observation = obs_process(obs)
            total_reward += reward
            if visualization:
                env.render()
            if done:
                break
        observation = obs_process(obs)
        reward = total_reward

        """
        # action per frame
        obs, reward, done, _ = env.step(action, project=False)
        observation = obs_process(obs)
        if visualization:
            env.render()
        """
        
        agent.observe(reward, observation, done)

        step += 1
        episode_steps += 1
        episode_reward += reward
        if done or (episode_steps >= max_episode_length and max_episode_length):
            if step > args.warmup:
                # checkpoint
                if episode > 0 and episode % args.save_interval == 0:
                    save_times += 1
                    print('[save model] #{} in {}'.format(save_times, args.output))
                    agent.save_model(args.output, save_times)
                    
                # validation
                if episode > 0 and episode % args.validate_interval == 0:
                    validation_reward = evaluator(env, agent.select_action, visualize=False)
                    print('[validation] episode #{}, reward={}'.format(episode, np.mean(validation_reward)))
                    writer.add_scalar('validation/reward', np.mean(validation_reward), episode)

            writer.add_scalar('train/train_reward', episode_reward, episode)

            # log
            episode_time = time.time() - time_stamp
            time_stamp = time.time()
            print('episode #{}: reward={}, steps={}, time={:.2f}'.format(
                    episode, episode_reward, episode_steps, episode_time
            ))

            for _ in range(episode_steps):
                log += 1
                Q, critic_loss = agent.update_policy()
                writer.add_scalar('train/Q', Q, log)
                writer.add_scalar('train/critic loss', critic_loss, log)

            observation = None
            episode_steps = 0
            episode_reward = 0
            episode += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDPG implemented by PyTorch')
    parser.add_argument('--discount', default=0.99, type=float, help='bellman discount')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--memory_size', default=1000000, type=int, help='memory size')
    parser.add_argument('--action_repeat', default=5, type=int, help='repeat times for each action')

    parser.add_argument('--hidden1', default=400, type=int, help='number of first fully connected layer')
    parser.add_argument('--hidden2', default=300, type=int, help='number of second fully connected layer')
    parser.add_argument('--actor_lr', default=1e-4, type=float, help='actor learning rate')
    parser.add_argument('--critic_lr', default=1e-3, type=float, help='critic learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='minibatch size')

    parser.add_argument('--iterations', default=2000000, type=int, help='iterations during training')
    parser.add_argument('--warmup', default=100, type=int, help='timestep without training to fill the replay buffer')
    parser.add_argument('--apply_noise', dest='apply_noise', default=True, action='store_true', help='apply noise to the action')
    parser.add_argument('--validate_interval', default=10, type=int, help='episode interval to validate')
    parser.add_argument('--save_interval', default=20, type=int, help='episode interval to save model')
    parser.add_argument('--validate_episodes', default=1, type=int, help='how many episodes to validate')
    parser.add_argument('--max_episode_length', default=500, type=int, help='maximum episode length')

    parser.add_argument('--resume', default=None, type=str, help='resuming model path')
    parser.add_argument('--resume_num', default=1, type=int, help='number of the weight to load')
    parser.add_argument('--output', default='output', type=str)
    parser.add_argument('--visualization', dest='visualization', action='store_true')
    parser.add_argument('--cuda', dest='cuda', action='store_true')

    parser.add_argument('--seed', default=-1, type=int, help='random seed')

    args = parser.parse_args()

    # TensorBoardX summary file
    timestamp = time.time()
    timestruct = time.localtime(timestamp)
    writer = SummaryWriter(os.path.join(args.output,  'Prosthetics@' + time.strftime('%Y-%m-%d %H:%M:%S', timestruct)))

    env = ProstheticsEnv(visualize=False)

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

    # resume train
    if args.resume is not None:
        print('resume train, load weight file: {}...'.format(args.resume_num))
        agent.load_model(args.output, args.resume_num)

    train(args.iterations, agent, env, evaluator)
