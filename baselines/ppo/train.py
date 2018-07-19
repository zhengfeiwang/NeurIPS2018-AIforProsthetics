import argparse
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env

MAX_STEPS_PER_ITERATION = 1000


def env_creator(env_config):
    from custom_env import CustomEnv
    env = CustomEnv(action_repeat=args.action_repeat)
    return env


def configure(args):
    config = ppo.DEFAULT_CONFIG.copy()

    # general - hard code
    config["horizon"] = MAX_STEPS_PER_ITERATION // args.action_repeat

    # general - according to arguments
    config["sample_batch_size"] = args.batch_size
    config["learning_starts"] = args.warmup
    
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLlib version AI for Prosthetics Challenge")
    # Ray
    parser.add_argument("--redis-address", default=None, type=str, help="address of the Redis server")
    parser.add_argument("--num-cpus", default=2, type=int, help="number of local cpus")
    # hyperparameters
    parser.add_argument("--batch-size", default=256, type=int, help="minibatch size")
    parser.add_argument("--action-repeat", default=4, type=int, help="repeat time for each action")
    parser.add_argument("--warmup", default=10000, type=int, help="number of random action before training")
    # checkpoint
    parser.add_argument("--checkpoint-dir", default="output", type=str, help="checkpoint output directory")
    parser.add_argument("--checkpoint-interval", default=10, type=int, help="iteration interval for checkpoint")
    
    args = parser.parse_args()

    if args.redis_address is not None:
        ray.init(redis_address=args.redis_address)
    else:
        ray.init(num_cpus=args.num_cpus)

    register_env("ProstheticsEnv", env_creator)
    config = configure(args)

    agent = ppo.PPOAgent(env="ProstheticsEnv", config=config)

    # agent training
    n_iteration = 1
    while (True):
        agent.train()
        print('training step: #{}'.format(n_iteration))

        n_iteration += 1

        if n_iteration % args.checkpoint_interval == 0:
            checkpoint = agent.save(args.checkpoint_dir)
            print('[checkpoint] No.{}'.format(n_iteration))
