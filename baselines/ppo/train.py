import os
import argparse
import logging
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env

MAX_STEPS_PER_ITERATION = 1000
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def env_creator(env_config):
    from custom_env import CustomEnv
    env = CustomEnv(action_repeat=args.action_repeat, 
                    integrator_accuracy=args.integrator_accuracy,
                    reward_shaping=args.reward_shaping)
    return env


def configure(args):
    config = ppo.DEFAULT_CONFIG.copy()

    # common
    config["model"]["squash_to_range"] = True # action clip

    config["horizon"] = MAX_STEPS_PER_ITERATION // args.action_repeat
    config["num_workers"] = args.num_workers

    # PPO specific
    config["num_gpus"] = args.num_gpus
    
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLlib version AI for Prosthetics Challenge")
    # Ray
    parser.add_argument("--redis-address", default="192.168.1.137:16379", type=str, help="address of the Redis server")
    parser.add_argument("--num-workers", default=1, type=int, help="number of workers for parallelism")
    parser.add_argument("--num-cpus", default=1, type=int, help="number of local cpus")
    parser.add_argument("--num-gpus", default=0, type=int, help="number of GPUs for SGD")
    parser.add_argument("--cluster", default=False, action="store_true", help="whether use cluster or local computer")
    # environment
    parser.add_argument("--action-repeat", default=4, type=int, help="repeat time for each action")
    parser.add_argument("--integrator-accuracy", default=1e-3, type=float, help="simulator integrator accuracy")
    parser.add_argument("--reward-shaping", default=False, action="store_true", help="add extra items to reward")
    # checkpoint
    parser.add_argument("--checkpoint-dir", default="output", type=str, help="checkpoint output directory")
    parser.add_argument("--checkpoint-interval", default=100, type=int, help="iteration interval for checkpoint")
    
    args = parser.parse_args()

    if args.cluster is True:
        ray.init(redis_address=args.redis_address)
    else:
        ray.init(num_cpus=args.num_cpus)

    register_env("ProstheticsEnv", env_creator)
    config = configure(args)

    agent = ppo.PPOAgent(env="ProstheticsEnv", config=config)

    # verify checkpoint directory
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    # agent training
    while True:
        train_result = agent.train()
        # log out useful information
        logger.info('training iteration: #{}'.format(train_result.training_iteration))
        logger.info('time this iteration: {}'.format(train_result.time_this_iter_s))
        logger.debug('timestep number this iteration: {}'.format(train_result.timesteps_this_iter))
        logger.debug('total timesteps: {}'.format(train_result.timesteps_total))
        logger.debug('episode number this iteration: {}'.format(train_result.episodes_total))
        logger.debug('episode mean length: {} (x{})'.format(train_result.episode_len_mean, args.action_repeat))
        logger.debug('episode reward:')
        logger.debug('  [mean] {}'.format(train_result.episode_reward_mean))
        logger.debug('  [max] {}'.format(train_result.episode_reward_max))
        logger.debug('  [min] {}'.format(train_result.episode_reward_min))
        logger.debug('--------------------------------------------------')

        if train_result.training_iteration % args.checkpoint_interval == 0:
            save_result = agent.save(args.checkpoint_dir)
            logger.info('[checkpoint] iteration #{} at {}'.format(train_result.training_iteration, save_result))
