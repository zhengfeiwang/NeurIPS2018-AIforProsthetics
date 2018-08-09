from osim.env import ProstheticsEnv
import numpy as np
from gym.spaces import Box
from utils import process_observation


class CustomEnv(ProstheticsEnv):
    def __init__(self, episode_length=300, dim=158, frameskip=1, integrator_accuracy=5e-5, reward_type="2018"):
        self.env = ProstheticsEnv(visualize=False)
        self.episode_length = episode_length
        self.frameskip = frameskip
        self.env.integrator_accuracy = integrator_accuracy
        self.observation_space = Box(low=-10, high=+10, shape=(dim,), dtype=np.float32)
        self.action_space = self.env.action_space
        # reward shaping
        self.reward_type = reward_type
        self.prev_pelvis_pos = 0.0
        self.episode_steps = 0

    def step(self, action):
        cumulative_reward = 0.0
        action = np.clip(action, 0.0, 1.0)

        for _ in range(self.frameskip):
            observation, reward, done, info = self.env.step(action, project=False)
            self.episode_steps += 1

            if self.reward_type == "2018":
                reward = reward
            elif self.reward_type == "2017":
                reward = observation["body_pos"]["pelvis"][0] - self.prev_pelvis_pos
                self.prev_pelvis_pos = observation["body_pos"]["pelvis"][0]
            elif self.reward_type == "shaped":
                survival = 0.01
                lean = min(0.3, max(0, observation["body_pos"]["pelvis"][0] - observation["body_pos"]["head"][0] - 0.15)) * 0.05
                joint = sum([max(0, knee - 0.1) for knee in [observation["joint_pos"]["knee_l"][0], observation["joint_pos"]["knee_r"][0]]]) * 0.03
                penalty = lean + joint
                reward = 0.02 * reward + survival - penalty

            cumulative_reward += reward
            if done or self.episode_steps >= self.episode_length:
                # penalty for early failure
                if self.episode_steps < self.episode_length:
                    cumulative_reward = -1
                break
        # transform dictionary to 1D vector
        observation = process_observation(observation)
        # clip reward
        clip_reward = np.clip(cumulative_reward, -1.0, 1.0)
        return observation, clip_reward, done, info

    def reset(self):
        observation = self.env.reset(project=False)
        self.prev_pelvis_pos = 0.0
        self.episode_steps = 0
        return process_observation(observation)
