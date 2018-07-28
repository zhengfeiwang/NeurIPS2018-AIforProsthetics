import os
import time
import argparse
import logging
import ray
import ray.rllib.agents.ddpg as ddpg
from ray.tune.registry import register_env
from tensorboardX import SummaryWriter
from evaluator import Evaluator

MAX_STEPS_PER_ITERATION = 300
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def env_creator(env_config):
    from custom_env import CustomEnv
    env = CustomEnv(action_repeat=args.action_repeat, 
                    integrator_accuracy=args.integrator_accuracy,
                    reward_type=args.reward_type)
    return env


def configure(args):
    config = ddpg.DEFAULT_CONFIG.copy()

    # common
    config["horizon"] = MAX_STEPS_PER_ITERATION // args.action_repeat
    config["num_workers"] = args.num_workers
    config["model"]["squash_to_range"] = True # action clip

    # DDPG specific
    config["noise_scale"] = args.noise_level
    config["clip_rewards"] = False  # essential for reward shaping
    config["learning_starts"] = args.warmup
    config["train_batch_size"] = args.batch_size
    config["gpu"] = args.gpu

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
    parser.add_argument("--redis-address", default="192.168.1.137:16379", type=str, help="address of the Redis server")
    parser.add_argument("--num-workers", default=1, type=int, help="number of workers for parallelism")
    parser.add_argument("--num-cpus", default=1, type=int, help="number of local cpus")
    parser.add_argument("--cluster", default=False, action="store_true", help="whether use cluster or local computer")
    # model
    parser.add_argument("--actor-hiddens", default="400-300", type=str, help="Actor architecture")
    parser.add_argument("--critic-hiddens", default="400-300", type=str, help="Critic architecture")
    parser.add_argument("--actor-activation", default="relu", type=str, help="Actor activation function")
    parser.add_argument("--critic-activation", default="relu", type=str, help="Critic activation function")
    # hyperparameters
    parser.add_argument("--batch-size", default=256, type=int, help="minibatch size")
    parser.add_argument("--actor-learning-rate", default=1e-4, type=float, help="Actor learning rate")
    parser.add_argument("--critic-learning-rate", default=1e-3, type=float, help="Critic learning rate")
    parser.add_argument("--action-repeat", default=4, type=int, help="repeat time for each action")
    parser.add_argument("--warmup", default=10000, type=int, help="number of random action before training")
    parser.add_argument("--reward-type", default="2018", type=str, help="reward type")
    parser.add_argument("--noise-level", default=0.1, type=float, help="noise level")
    # environment
    parser.add_argument("--integrator-accuracy", default=1e-3, type=float, help="simulator integrator accuracy")
    parser.add_argument("--gpu", default=False, action="store_true", help="use GPU for optimization")
    # checkpoint and validation
    parser.add_argument("--checkpoint-dir", default="output", type=str, help="checkpoint output directory")
    parser.add_argument("--checkpoint-interval", default=5, type=int, help="iteration interval for checkpoint")
    parser.add_argument("--validation-interval", default=5, type=int, help="iteration interval for validation")
    
    args = parser.parse_args()

    if args.cluster is True:
        ray.init(redis_address=args.redis_address)
    else:
        ray.init(num_cpus=args.num_cpus)

    register_env("ProstheticsEnv", env_creator)
    config = configure(args)

    agent = ddpg.DDPGAgent(env="ProstheticsEnv", config=config)

    # verify checkpoint directory
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    
    # initialize evaluator
    evaluator = Evaluator(action_repeat=args.action_repeat)

    # tensorboard for validation reward
    timestamp = time.time()
    timestruct = time.localtime(timestamp)
    writer = SummaryWriter(os.path.join(args.checkpoint_dir, time.strftime('%Y-%m-%d_%H-%M-%S', timestruct)))

    # agent training
    while True:
        train_result = agent.train()
        # log out useful information
        logger.info('training iteration: #{}'.format(train_result.training_iteration))
        logger.info('time this iteration: {}'.format(train_result.time_this_iter_s))
        logger.debug('  sample time: {}'.format(train_result.info["sample_time_ms"] / 1000 * train_result.timesteps_this_iter))
        logger.debug('  replay time: {}'.format(train_result.info["replay_time_ms"] / 1000 * train_result.timesteps_this_iter))
        logger.debug('  gradient time: {}'.format(train_result.info["grad_time_ms"] / 1000 * train_result.timesteps_this_iter))
        logger.debug('  update time: {}'.format(train_result.info["update_time_ms"] / 1000 * train_result.timesteps_this_iter))
        logger.debug('timestep number this iteration: {}'.format(train_result.timesteps_this_iter))
        logger.debug('total timesteps: {}'.format(train_result.timesteps_total))
        logger.debug('episode number this iteration: {}'.format(train_result.episodes_total))
        logger.debug('episode mean length: {} (x{})'.format(train_result.episode_len_mean, args.action_repeat))
        logger.debug('episode reward:')
        logger.debug('  [mean] {}'.format(train_result.episode_reward_mean))
        logger.debug('  [max] {}'.format(train_result.episode_reward_max))
        logger.debug('  [min] {}'.format(train_result.episode_reward_min))
        logger.debug('--------------------------------------------------')

        # record train information in private tensorboard
        writer.add_scalar('train/mean_reward', train_result.episode_reward_mean, train_result.training_iteration)
        writer.add_scalar('train/mean_steps', train_result.episode_len_mean, train_result.training_iteration)
        writer.add_scalar('train/time', train_result.time_this_iter_s, train_result.training_iteration)

        # validation
        if train_result.training_iteration % args.validation_interval == 0:
            validation_reward, validation_steps = evaluator(agent)
            logger.info(' > validation at iteration: #{}: reward = {}, episode length = {}'.format(
                train_result.training_iteration, validation_reward, validation_steps
            ))
            # record validation score/steps in private tensorboard
            writer.add_scalar('validation/reward', validation_reward, train_result.training_iteration)
            writer.add_scalar('validation/steps', validation_steps, train_result.training_iteration)

        # checkpoint
        if train_result.training_iteration % args.checkpoint_interval == 0:
            save_result = agent.save(args.checkpoint_dir)
            logger.info('[checkpoint] iteration #{} at {}'.format(train_result.training_iteration, save_result))
