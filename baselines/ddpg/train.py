import os
import argparse
import logging
import ray
import ray.rllib.agents.ddpg as ddpg
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
    config = ddpg.DEFAULT_CONFIG.copy()

    # common
    config["horizon"] = MAX_STEPS_PER_ITERATION // args.action_repeat
    config["num_workers"] = args.num_workers

    # DDPG specific
    # config["gpu"] = True
    # config["train_batch_size"] = args.batch_size
    # config["learning_starts"] = args.warmup

    actor_hiddens = []
    actor_layers = args.actor_hiddens.split('-')
    for actor_layer in actor_layers:
        actor_hiddens.append(int(actor_layer))
    critic_hiddens = []
    critic_layers = args.critic_hiddens.split('-')
    for critic_layer in critic_layers:
        critic_hiddens.append(int(critic_layer))
    
    config["actor_hiddens"] = actor_hiddens
    config["actor_hidden_activation"] = args.actor_activation
    config["critic_hiddens"] = critic_hiddens
    config["critic_hidden_activation"] = args.critic_activation
    
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLlib version AI for Prosthetics Challenge")
    # Ray
    parser.add_argument("--redis-address", default=None, type=str, help="address of the Redis server")
    parser.add_argument("--num-workers", default=1, type=int, help="number of workers for parallelism")
    parser.add_argument("--num-cpus", default=1, type=int, help="number of local cpus")
    # model
    parser.add_argument("--actor-hiddens", default="400-300", type=str, help="Actor architecture")
    parser.add_argument("--critic-hiddens", default="400-300", type=str, help="Critic architecture")
    parser.add_argument("--actor-activation", default="relu", type=str, help="Actor activation function")
    parser.add_argument("--critic-activation", default="relu", type=str, help="Critic activation function")
    # hyperparameters
    parser.add_argument("--batch-size", default=256, type=int, help="minibatch size")
    parser.add_argument("--actor-learning-rate", default=1e-4, type=float, help="Actor learning rate")
    parser.add_argument("--critic-learning-rate", default=1e-3, type=float, help="Critic learning rate")
    parser.add_argument("--action-repeat", default=5, type=int, help="repeat time for each action")
    parser.add_argument("--warmup", default=10000, type=int, help="number of random action before training")
    parser.add_argument("--reward-shaping", default=False, action="store_true", help="add extra items to reward")
    # environment
    parser.add_argument("--integrator-accuracy", default=1e-3, type=float, help="simulator integrator accuracy")
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

    agent = ddpg.DDPGAgent(env="ProstheticsEnv", config=config)

    # verify checkpoint directory
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    # agent training
    while True:
        train_result = agent.train()
        
        # log out useful information
        logger.info('training iteration: #{}'.format(train_result.training_iteration))
        logger.info('time this iteration: {}'.format(train_result.time_this_iter_s))
        logger.debug('  sample time: {}'.format(train_result.info["sample_time_ms"] / 1000))
        logger.debug('  replay time: {}'.format(train_result.info["replay_time_ms"] / 1000))
        logger.debug('  gradient time: {}'.format(train_result.info["grad_time_ms"] / 1000))
        logger.debug('  update time: {}'.format(train_result.info["update_time_ms"] / 1000))
        logger.debug('step number this iteration: {}'.format(train_result.timesteps_this_iter))
        logger.debug('total steps: {}'.format(train_result.timesteps_total))
        logger.debug('episode number this iteration: {}'.format(train_result.episodes_total))
        logger.debug('episode mean length: {}'.format(train_result.episode_len_mean * args.action_repeat))
        logger.debug('episode reward:')
        logger.debug('  [mean] {}'.format(train_result.episode_reward_mean))
        logger.debug('  [max] {}'.format(train_result.episode_reward_max))
        logger.debug('  [min] {}'.format(train_result.episode_reward_min))
        logger.debug('------------------------------------------------------------')

        if train_result.training_iteration % args.checkpoint_interval == 0:
            save_result = agent.save(args.checkpoint_dir)
            print('[checkpoint] iteration #{} at {}'.format(train_result.training_iteration, save_result))
