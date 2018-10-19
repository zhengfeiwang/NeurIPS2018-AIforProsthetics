import numpy as np
import gym
from gym.spaces import Box
from osim.env import ProstheticsEnv

OBSERVATION_SPACE = 226


class Round2Env(ProstheticsEnv):
    def __init__(self, visualization=True, integrator_accuracy=5e-5):
        # difficulty = 1 for round 2 environment
        super().__init__(visualization, integrator_accuracy, difficulty=1)
        self.episode_length = 0
        self.episode_original_reward = 0.0
        self.episode_shaped_reward = 0.0

        self.observation_space = Box(low=-10, high=+10, shape=[OBSERVATION_SPACE])

    def step(self, action, project=True):
        obs, r, done, info = super(Round2Env, self).step(action)
        self.episode_length += 1

        # early termination penalty
        if done and self.episode_length < self.time_limit:
            r -= 2

        original_reward = super(Round2Env, self).reward()
        self.episode_original_reward += original_reward
        self.episode_shaped_reward += r

        state_desc = self.get_state_desc()

        # log information
        # print('timestamp:', self.episode_length, 'pros_foot_y:', state_desc["body_pos"]["pros_foot_r"][1])

        if done:
            info['episode'] = {
                'r': self.episode_original_reward,
                'l': self.episode_length,
                "pelvis_x": state_desc["body_pos"]["pelvis"][0],
                "shaped_reward": self.episode_shaped_reward
            }

        return obs, r, done, info

    def reset(self, project=True):
        super().reset(project=project)
        self.episode_length = 0
        self.episode_original_reward = 0.0
        self.episode_shaped_reward = 0.0
        obs = self.get_observation()
        return obs

    def get_observation_space_size(self):
        return OBSERVATION_SPACE

    def get_observation(self):
        state_desc = self.get_state_desc()

        res = []
        pelvis = None

        for body_part in ["pelvis", "head", "torso", "toes_l", "talus_l", "pros_foot_r", "pros_tibia_r"]:
            cur = []
            cur += state_desc["body_pos"][body_part]
            cur += state_desc["body_vel"][body_part]
            cur += state_desc["body_acc"][body_part]
            cur += state_desc["body_pos_rot"][body_part]
            cur += state_desc["body_vel_rot"][body_part]
            cur += state_desc["body_acc_rot"][body_part]
            if body_part == "pelvis":
                pelvis = cur
                res += cur[1:]  # make sense, pelvis.x is not important
            else:
                cur[0] -= pelvis[0]
                cur[2] -= pelvis[2]     # relative position work for x / z axis
                res += cur

        for joint in ["ankle_l", "ankle_r", "back", "hip_l", "hip_r", "knee_l", "knee_r"]:
            res += state_desc["joint_pos"][joint]
            res += state_desc["joint_vel"][joint]
            res += state_desc["joint_acc"][joint]

        for muscle in sorted(state_desc["muscles"].keys()):
            res += [state_desc["muscles"][muscle]["activation"]]
            res += [state_desc["muscles"][muscle]["fiber_length"]]
            res += [state_desc["muscles"][muscle]["fiber_velocity"]]

        cm_pos = state_desc["misc"]["mass_center_pos"]  # relative x / z axis center of mass position
        cm_pos[0] -= pelvis[0]
        cm_pos[2] -= pelvis[0]

        res = \
            res + \
            cm_pos + \
            state_desc["misc"]["mass_center_vel"] + \
            state_desc["misc"]["mass_center_acc"] + \
            [state_desc["target_vel"][0]] + [state_desc["target_vel"][2]]

        return res

    def reward(self):
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        if not prev_state_desc:
            return 0

        # course 1 - basic walk
        pelvis_vx = state_desc["body_vel"]["pelvis"][0]
        reward = pelvis_vx * 4 + 2

        return reward * 0.05


class CustomActionWrapper(gym.ActionWrapper):

    def step(self, action):
        action = self.action(action)
        rew = 0
        for i in range(2):
            obs, r, done, info = self.env.step(action)
            rew += r
            if done:
                break
        info["action"] = action
        return obs, rew, done, info

    def action(self, action):
        return np.clip(action, 0.0, 1.0)    # clip for valid action
