import os
import argparse
import logging
from osim.env import ProstheticsEnv
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env

MAX_STEPS_PER_ITERATION = 1000
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def env_creator(env_config):
    from custom_env import CustomEnv
    env = CustomEnv(action_repeat=args.action_repeat)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLlib version AI for Prosthetics Challenge")
    # hyperparameters
    parser.add_argument("--action-repeat", default=1, type=int, help="repeat time for each action")
    # checkpoint
    parser.add_argument("--checkpoint-dir", default="output", type=str, help="checkpoint output directory")
    parser.add_argument("--checkpoint-id", default=None, type=str, help="id of checkpoint file")
    parser.add_argument("--visualization", default=False, action="store_true", help="visualization for evaluation")
    
    args = parser.parse_args()

    ray.init(num_cpus=2) # PPO needs at least 2 CPUs

    register_env("ProstheticsEnv", env_creator)
    config = ppo.DEFAULT_CONFIG.copy()

    agent = ppo.PPOAgent(env="ProstheticsEnv", config=config)
    checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint-" + str(args.checkpoint_id))
    agent.restore(checkpoint_path=checkpoint_path)

    env = ProstheticsEnv(visualize=args.visualization)
    observation = env.reset()

    episode_reward = 0.0
    steps = 0
    done = False

    while steps < MAX_STEPS_PER_ITERATION and not done:
        action = agent.compute_action(observation)
        for _ in range(args.action_repeat):
            observation, reward, done, _ = env.step(action)
            logger.debug('step #{}: action={}'.format(steps, action))
            logger.debug('  reward={}'.format(reward))
            steps += 1
            episode_reward += reward
            if done or steps >= MAX_STEPS_PER_ITERATION:
                break
    
    logger.info('score: {}'.format(episode_reward))
    logger.debug('episode length: {}'.format(steps * args.action_repeat))
    
    env.close()
