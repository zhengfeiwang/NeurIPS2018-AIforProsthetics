from osim.env import ProstheticsEnv
import numpy as np
import gym
from gym.spaces import Box


class CustomEnv():
    def __init__(self, action_repeat, integrator_accuracy=5e-5, reward_shaping=False):
        self.env = ProstheticsEnv(visualize=False)
        self.env.integrator_accuracy = integrator_accuracy
        self.action_repeat = action_repeat
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # reward shaping
        self.reward_shaping = reward_shaping
        self.prev_pelvis_pos = 0.0

    def step(self, action):
        cumulative_reward = 0.0
        for _ in range(self.action_repeat):
            observation, reward, done, info = self.env.step(action, project=False)

            if self.reward_shaping:
                penalty = 0.0
                prev_reward = 0.0
                steps_reward = 1.0 # extend number of steps in an episode
                # lean penalty - offset between head and pelvis on x-axis and z-axis
                head_pos = observation["body_pos"]["head"]
                pelvis_pos = observation["body_pos"]["pelvis"]
                penalty += min(0.3, max(0, abs(pelvis_pos[0] - head_pos[0]) - 0.15)) * 0.05
                penalty += min(0.3, max(0, abs(pelvis_pos[2] - head_pos[2]) - 0.15)) * 0.05
                # reward in NIPS 2017 Learning to Run
                prev_reward = observation["body_pos"]["pelvis"][0] - self.prev_pelvis_pos
                self.prev_pelvis_pos = observation["body_pos"]["pelvis"][0]
                # add penalty and previous reward to current reward
                reward = reward * 0.5 + penalty + prev_reward + steps_reward * 0.05

            cumulative_reward += reward
            if done:
                break
        # transform dictionary to 1D vector
        observation = self.observation_process(observation)
        return observation, cumulative_reward, done, info

    def reset(self, project = True):
        observation = self.env.reset()
        self.prev_pelvis_pos = 0.0
        return observation
    
    # referenced from osim-rl
    def observation_process(self, state_desc):
        res = []
        pelvis = None

        for body_part in ["pelvis", "head","torso","toes_l","toes_r","talus_l","talus_r"]:
            if body_part in ["toes_r","talus_r"]:
                res += [0] * 9
                continue
            cur = []
            cur += state_desc["body_pos"][body_part][0:2]
            cur += state_desc["body_vel"][body_part][0:2]
            cur += state_desc["body_acc"][body_part][0:2]
            cur += state_desc["body_pos_rot"][body_part][2:]
            cur += state_desc["body_vel_rot"][body_part][2:]
            cur += state_desc["body_acc_rot"][body_part][2:]
            if body_part == "pelvis":
                pelvis = cur
                res += cur[1:]
            else:
                cur_upd = cur
                cur_upd[:2] = [cur[i] - pelvis[i] for i in range(2)]
                cur_upd[6:7] = [cur[i] - pelvis[i] for i in range(6,7)]
                res += cur

        for joint in ["ankle_l","ankle_r","back","hip_l","hip_r","knee_l","knee_r"]:
            res += state_desc["joint_pos"][joint]
            res += state_desc["joint_vel"][joint]
            res += state_desc["joint_acc"][joint]

        for muscle in sorted(state_desc["muscles"].keys()):
            res += [state_desc["muscles"][muscle]["activation"]]
            res += [state_desc["muscles"][muscle]["fiber_length"]]
            res += [state_desc["muscles"][muscle]["fiber_velocity"]]

        cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(2)]
        res = res + cm_pos + state_desc["misc"]["mass_center_vel"] + state_desc["misc"]["mass_center_acc"]

        return res
