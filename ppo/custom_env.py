from osim.env import ProstheticsEnv
import numpy as np
from gym.spaces import Box
from utils import process_observation

MAX_STEPS_PER_EPISODE = 300
CUSTOM_OBSERVATION_SPACE = 85


class CustomEnv(ProstheticsEnv):
    def __init__(self, action_repeat, integrator_accuracy=5e-5, reward_type="2018", binary_action=False):
        self.env = ProstheticsEnv(visualize=False)
        self.env.integrator_accuracy = integrator_accuracy
        self.action_repeat = action_repeat
        self.binary_action = binary_action
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

        if self.binary_action:
            for i in range(len(action)):
                action[i] = 1.0 if action[i] > 0.5 else 0.0
        else:
            action = np.clip(action, 0.0, 1.0)

        for _ in range(self.action_repeat):
            observation, reward, done, info = self.env.step(action, project=False)
            self.episode_steps += 1

            if self.reward_type == "2018":
                reward = reward
            elif self.reward_type == "2017":
                reward = observation["body_pos"]["pelvis"][0] - self.prev_pelvis_pos
                self.prev_pelvis_pos = observation["body_pos"]["pelvis"][0]
            elif self.reward_type == "shaped":
                # translation
                translation = observation["body_pos"]["pelvis"][0] - self.prev_pelvis_pos
                self.prev_pelvis_pos = observation["body_pos"]["pelvis"][0]
                # survival
                survival = 0.05
                # shaped reward
                reward = reward * 0.05 + min(translation * 10, 0.3) + survival
            else:
                assert False, 'unknown reward type...'

            cumulative_reward += reward
            if done or self.episode_steps >= MAX_STEPS_PER_EPISODE:
                # punish for failure
                if self.episode_steps < MAX_STEPS_PER_EPISODE:
                    cumulative_reward -= 0.5
                break
        # transform dictionary to 1D vector
        observation = process_observation(observation)
        # reward rescale and clip
        cumulative_reward /= self.action_repeat
        clipped_reward = -1.0 if cumulative_reward < -1.0 else cumulative_reward
        return observation, clipped_reward, done, info

    def reset(self):
        observation = self.env.reset(project=False)
        self.prev_pelvis_pos = 0.0
        self.episode_steps = 0
        return process_observation(observation)
