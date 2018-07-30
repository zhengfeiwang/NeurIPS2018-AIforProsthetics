from osim.env import ProstheticsEnv
import numpy as np
from gym.spaces import Box
from utils import process_observation

MAX_STEPS_PER_EPISODE = 300
CUSTOM_OBSERVATION_SPACE = 85


class CustomEnv(ProstheticsEnv):
    def __init__(self, action_repeat, integrator_accuracy=5e-5, reward_type="2018"):
        self.env = ProstheticsEnv(visualize=False)
        self.env.integrator_accuracy = integrator_accuracy
        self.action_repeat = action_repeat
        # self.observation_space = self.env.observation_space
        # custom observation space
        self.observation_space = Box(low=-3, high=+3, shape=(CUSTOM_OBSERVATION_SPACE,), dtype=np.float32)
        self.action_space = self.env.action_space

        # reward shaping
        self.reward_type = reward_type
        self.prev_pelvis_pos = 0.0
        self.episode_steps = 0

    def step(self, action):
        cumulative_reward = 0.0

        # action clip to [0, 1]        
        action = np.clip(action, 0.0, 1.0)

        for _ in range(self.action_repeat):
            observation, reward, done, info = self.env.step(action, project=False)

            if self.reward_type == "2018":
                reward = reward
            elif self.reward_type == "2017":
                reward = observation["body_pos"]["pelvis"][0] - self.prev_pelvis_pos
                self.prev_pelvis_pos = observation["body_pos"]["pelvis"][0]
            elif self.reward_type == "shaped":
                # essential: consider reward clip to [-1, 1]
                velocity = observation["body_vel"]["pelvis"][0]
                # hzwer penalty
                lean = min(0.3, max(0, observation["body_pos"]["pelvis"][0] - observation["body_pos"]["head"][0] - 0.15)) * 0.05
                joint = sum([max(0, knee - 0.1) for knee in [observation["joint_pos"]["knee_l"][0], observation["joint_pos"]["knee_r"][0]]]) * 0.03
                penalty = lean + joint
                # survival
                survival = 0.02
                reward = velocity - penalty + survival
            else:
                assert False, 'unknown reward type...'

            cumulative_reward += reward
            if done:
                # punish for failure
                if self.episode_steps < MAX_STEPS_PER_EPISODE:
                    cumulative_reward -= 0.2
                break
        # transform dictionary to 1D vector
        observation = process_observation(observation)
        # clip rewards to [-1, 1]
        clipped_reward = np.clip(cumulative_reward, -1.0, 1.0)
        return observation, clipped_reward, done, info

    def reset(self):
        observation = self.env.reset(project=False)
        self.prev_pelvis_pos = 0.0
        self.episode_steps = 0
        return process_observation(observation)
