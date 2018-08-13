import os
import time
import random
import argparse
import logging
import numpy as np
import tensorflow as tf
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from tensorboardX import SummaryWriter
from evaluator import Evaluator
from train import train
from utils import OBSERVATION_DIM


def env_creator(env_config):
    from custom_env import CustomEnv
    env = CustomEnv(episode_length=args.episode_length, 
                    dim=OBSERVATION_DIM, 
                    frameskip=args.frameskip, 
                    integrator_accuracy=args.accuracy,
                    reward_type=args.reward)
    return env


def configure(args):
    config = ppo.DEFAULT_CONFIG.copy()
    # common configs
    config["gamma"] = args.gamma
    config["horizon"] = args.episode_length // args.frameskip
    config["num_workers"] = args.num_workers
    config["sample_batch_size"] = args.sample_batch
    config["batch_mode"] = "truncate_episodes"

    # model configs
    hiddens = []
    hidden_layers = args.hiddens.split('-')
    for hidden_layer in hidden_layers:
        hiddens.append(int(hidden_layer))
    config["model"]["fcnet_hiddens"] = hiddens
    config["model"]["fcnet_activation"] = args.activations
    config["model"]["squash_to_range"] = True # action clip

    # PPO specific
    config["kl_coeff"] = args.kl_coeff
    config["timesteps_per_batch"] = args.sample
    config["num_sgd_iter"] = args.epochs * (args.sample // args.batch_size)
    config["sgd_stepsize"] = args.learning_rate
    config["sgd_batchsize"] = args.batch_size
    if args.gpu is True and args.num_gpus is not None:
        config["num_gpus"] = args.num_gpus
    
    return config


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="RLlib version AI for Prosthetics Challenge")
    # parallel
    parser.add_argument("--num-workers", default=1, type=int, help="number of workers for parallelism")
    parser.add_argument("--num-cpus", default=1, type=int, help="number of local cpus")
    parser.add_argument("--num-gpus", default=None, type=int, help="number of gpus")
    parser.add_argument("--cluster", default=False, action="store_true", help="whether use cluster or local computer")
    parser.add_argument("--redis-address", default="192.168.1.137:16379", type=str, help="address of the Redis server")
    parser.add_argument("--sample", default=4000, type=int, help="number of samples per iteration")
    parser.add_argument("--sample-batch", default=200, type=int, help="sample batch size")
    # train setting
    parser.add_argument("--seed", default=-1, type=int, help="random seed")
    parser.add_argument("--gpu", default=False, action="store_true", help="use GPU for optimization")
    parser.add_argument("--iterations", default=None, type=int, help="number of training iterations")
    parser.add_argument("--checkpoint-dir", default="output", type=str, help="checkpoint output directory")
    parser.add_argument("--checkpoint-interval", default=5, type=int, help="iteration interval for checkpoint")
    parser.add_argument("--validation-interval", default=5, type=int, help="iteration interval for validation")
    parser.add_argument("--resume", default=False, action="store_true", help="resume to previous training")
    parser.add_argument("--resume-id", default=None, type=int, help="checkpoint id for training resume")
    # environment
    parser.add_argument("--reward", default="2018", type=str, help="reward type")
    parser.add_argument("--frameskip", default=1, type=int, help="number of frames to skip")
    parser.add_argument("--accuracy", default=5e-5, type=float, help="simulator integrator accuracy")
    parser.add_argument("--episode-length", default=300, type=int, help="max length for episode")
    # hyperparameters
    parser.add_argument("--epochs", default=30, type=int, help="number of training epochs")
    parser.add_argument("--batch-size", default=128, type=int, help="minibatch size")
    parser.add_argument("--learning-rate", default=5e-5, type=float, help="stepsize for optimization")
    parser.add_argument("--gamma", default=0.99, type=float, help="discount factor of the MDP")
    parser.add_argument("--kl-coeff", default=0.2, type=float, help="initial coefficient for KL divergence")
    parser.add_argument("--hiddens", default="256-256", type=str, help="hidden layer architecture")
    parser.add_argument("--activations", default="tanh", type=str, help="hidden layer activation")
    
    args = parser.parse_args()

    if args.seed == -1:
        args.seed = np.random.randint(0, 2**32)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    if args.cluster is True:
        ray.init(redis_address=args.redis_address)
    else:
        ray.init(num_cpus=args.num_cpus)

    register_env("CustomEnv", env_creator)
    config = configure(args)
    agent = ppo.PPOAgent(env="CustomEnv", config=config)
    evaluator = Evaluator(frameskip=args.frameskip)

    # verify checkpoint directory
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    
    # resume training
    if args.resume and args.resume_id is not None:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint-' + str(args.resume_id))
    agent.restore(checkpoint_path=checkpoint_path)

    # summary file
    timestruct = time.localtime(start_time)
    timestamp = time.strftime('%Y-%m-%d %H-%M-%S', timestruct)
    writer = SummaryWriter(os.path.join(args.checkpoint_dir, timestamp))
    # logger file
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(os.path.join(args.checkpoint_dir, timestamp + '.log'))
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s, %(levelname)s, %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)

    logger.info('start time: {}'.format(start_time))

    logger.info('<---------- Settings ---------->')
    logger.info('<--- Experiment --->')
    if args.cluster is True:
        logger.info('running on cluster, # workers = {}, # cpus = {}'.format(args.num_workers, args.num_cpus))
        logger.info('redis address: {}'.format(args.redis_address))
    else:
        logger.info('running on local machine, # workers = {}, # cpus = {}'.format(args.num_workers, args.num_cpus))
    logger.info('# gpus = {}'.format(args.num_gpus))
    logger.info('# training iterations = {}'.format(args.iterations))
    logger.info('# samples per iteration = {}, sample batch size = {}'.format(args.sample, args.sample_batch))
    logger.info('random seed: {}'.format(args.seed))
    logger.info('validation interval: {}, checkpoint interval: {}'.format(args.validation_interval, args.checkpoint_interval))
    logger.info('<--- Environment --->')
    logger.info('observation dimension is {}, reward type is {}'.format(OBSERVATION_DIM, args.reward))
    logger.info('frameskip: {}, episode max length: {}'.format(args.frameskip, args.episode_length))
    logger.info('simulator accuracy: {}'.format(args.accuracy))
    logger.info('<--- PPO --->')
    logger.info('gamma = {}, KL divergence = {}'.format(args.gamma, args.kl_coeff))
    logger.info('learning rate = {}, batch size = {}, # epochs = {}'.format(args.learning_rate, args.batch_size, args.epochs))
    logger.info('network architecture: {}, activation: {}'.format(args.hiddens, args.activations))
    logger.debug('RLlib agent config:')
    logger.debug(config)
    if args.resume and args.resume_id is not None:
        logger.debug('resume previous training, checkpoint id is {}'.format(args.resume_id))
    logger.info('<------------------------------>')

    train(agent, evaluator, logger, writer, args)

    end_time = time.time()
    duration = end_time - start_time
    logger.info('end time: {}'.format(end_time))
    logger.info('training duration: {}'.format(duration))
