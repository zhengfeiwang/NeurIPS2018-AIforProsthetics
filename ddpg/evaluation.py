import os
import argparse
import logging
import numpy as np
from osim.env import ProstheticsEnv
import ray
import ray.rllib.agents.ddpg as ddpg
from ray.tune.registry import register_env

MAX_STEPS_PER_ITERATION = 1000
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def custom_observation(observation):
    BODY_PARTS = ['femur_r', 'pros_tibia_r', 'pros_foot_r', 'femur_l', 'tibia_l', 'talus_l', 'calcn_l', 'toes_l', 'torso', 'head']
    JOINTS = ['ground_pelvis', 'hip_r', 'knee_r', 'ankle_r', 'hip_l', 'knee_l', 'ankle_l', 'back']

    # custom observation space 33 + 33 + 17 + 17 + 4 = 104D
    res = []
        
    # body parts positions relative to pelvis - 3 + 3 * 10D
    # pelvis relative position
    res += [0.0, 0.0, 0.0]
    pelvis_pos = observation["body_pos"]["pelvis"]
    for body_part in BODY_PARTS:
        # x, y, z - axis
        for axis in range(3):
            res += [observation["body_pos"][body_part][axis] - pelvis_pos[axis]]

    # body parts velocity relative to pelvis - 3 + 3 * 10D
    # pelvis relative velocity
    res += [0.0, 0.0, 0.0]
    pelvis_vel = observation["body_vel"]["pelvis"]
    for body_part in BODY_PARTS:
        # x, y, z - axis
        for axis in range(3):
            res += [observation["body_vel"][body_part][axis] - pelvis_vel[axis]]
        
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
    parser.add_argument("--no-visualization", default=False, action="store_true", help="no visualization for evaluation")
    
    args = parser.parse_args()

    ray.init(num_cpus=1)

    register_env("ProstheticsEnv", env_creator)
    config = configure(args)

    agent = ddpg.DDPGAgent(env="ProstheticsEnv", config=config)
    checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint-" + str(args.checkpoint_id))
    agent.restore(checkpoint_path=checkpoint_path)

    env = ProstheticsEnv(visualize=True if args.no_visualization is False else False)
    observation = custom_observation(env.reset(project=False))

    episode_reward = 0.0
    steps = 0
    done = False

    while steps < MAX_STEPS_PER_ITERATION and not done:
        action = agent.compute_action(observation)
        # action clip
        action = np.clip(action, 0.0, 1.0)

        for _ in range(args.action_repeat):
            observation, reward, done, _ = env.step(action, project=False)
            logger.debug('step #{}: action={}'.format(steps, action))
            logger.debug('  reward={}'.format(reward))
            steps += 1
            episode_reward += reward
            if done or steps >= MAX_STEPS_PER_ITERATION:
                break
        
        # transform dictionary to 1D vector
        observation = custom_observation(observation)
    
    logger.info('score: {}'.format(episode_reward))
    logger.debug('episode length: {}'.format(steps * args.action_repeat))
    
    env.close()
