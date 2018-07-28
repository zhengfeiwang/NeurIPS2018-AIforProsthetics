import os
import argparse
import logging
import numpy as np
from osim.env import ProstheticsEnv
import ray
import ray.rllib.agents.ddpg as ddpg
from ray.tune.registry import register_env
from evaluator import Evaluator

MAX_STEPS_PER_ITERATION = 300
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def custom_observation(observation):
    # custom observation space 33 + 3 + 17 + 17 + 4 = 74D
    res = []

    BODY_PARTS = ['femur_r', 'pros_tibia_r', 'pros_foot_r', 'femur_l', 'tibia_l', 'talus_l', 'calcn_l', 'toes_l', 'torso', 'head']
    JOINTS = ['ground_pelvis', 'hip_r', 'knee_r', 'ankle_r', 'hip_l', 'knee_l', 'ankle_l', 'back']
        
    # body parts positions relative to pelvis - 3 + 3 * 10D
    # pelvis relative position
    res += [0.0, 0.0, 0.0]
    pelvis_pos = observation["body_pos"]["pelvis"]
    for body_part in BODY_PARTS:
        # x, y, z - axis
        for axis in range(3):
            res += [observation["body_pos"][body_part][axis] - pelvis_pos[axis]]

    # pelvis velocity - 3D
    pelvis_vel = observation["body_vel"]["pelvis"]
    res += pelvis_vel

    """
    # body parts velocity relative to pelvis - 3 + 3 * 10D
    # pelvis relative velocity
    res += [0.0, 0.0, 0.0]
    pelvis_vel = observation["body_vel"]["pelvis"]
    for body_part in BODY_PARTS:
        # x, y, z - axis
        for axis in range(3):
            res += [observation["body_vel"][body_part][axis] - pelvis_vel[axis]]
    """
        
    # joints absolute angle - 6 + 3 + 1 + 1 + 3 + 1 + 1 + 1D
    for joint in JOINTS:
        for i in range(len(observation["joint_pos"][joint])):
            res += [observation["joint_pos"][joint][i]]
        
    # joints absolute angular velocity - 6 + 3 + 1 + 1 + 3 + 1 + 1 + 1D
    for joint in JOINTS:
        for i in range(len(observation["joint_vel"][joint])):
            res += [observation["joint_vel"][joint][i]]
        
    # center of mass position and velocity - 2 + 2D
    for axis in range(2):
        res += [observation["misc"]["mass_center_pos"][axis] - pelvis_pos[axis]]
    for axis in range(2):
        res += [observation["misc"]["mass_center_vel"][axis] - pelvis_vel[axis]]
            
    return res


def env_creator(env_config):
    from custom_env import CustomEnv
    env = CustomEnv(action_repeat=args.action_repeat)
    return env


def configure(args):
    config = ddpg.DEFAULT_CONFIG.copy()

    # DDPG specific - according to arguments
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
    # model
    parser.add_argument("--actor-hiddens", default="400-300", type=str, help="Actor architecture")
    parser.add_argument("--critic-hiddens", default="400-300", type=str, help="Critic architecture")
    parser.add_argument("--actor-activation", default="relu", type=str, help="Actor activation function")
    parser.add_argument("--critic-activation", default="relu", type=str, help="Critic activation function")
    # hyperparameters
    parser.add_argument("--action-repeat", default=4, type=int, help="repeat time for each action")
    # checkpoint
    parser.add_argument("--checkpoint-dir", default="output", type=str, help="checkpoint output directory")
    parser.add_argument("--checkpoint-id", default=None, type=str, help="id of checkpoint file")
    parser.add_argument("--no-render", default=False, action="store_true", help="no visualization for evaluation")
    
    args = parser.parse_args()

    ray.init(num_cpus=1)

    register_env("ProstheticsEnv", env_creator)
    config = configure(args)

    evaluator = Evaluator(args.action_repeat, render=True if args.no_render is False else False)

    agent = ddpg.DDPGAgent(env="ProstheticsEnv", config=config)
    checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint-" + str(args.checkpoint_id))
    agent.restore(checkpoint_path=checkpoint_path)

    evaluation_reward, evaluation_steps = evaluator(agent)
    logger.info('score: {}'.format(evaluation_reward))
    logger.info('steps: {}'.format(evaluation_steps))

    evaluator.close()
