from osim.env import ProstheticsEnv
import gym
import numpy as np
from gym.spaces import Box

OBSERVATION_SPACE = 224


class CustomEnv(ProstheticsEnv):
    def __init__(self, visualization=True, integrator_accuracy=5e-5):
        # difficulty = 1 for round 2 environment
        super().__init__(visualization, integrator_accuracy, difficulty=1)
        self.episode_length = 0
        self.episode_original_reward = 0.0
        self.episode_shaped_reward = 0.0
        self.episode_activation_penalty = 0.0
        self.episode_vx_penalty = 0.0
        self.episode_vz_penalty = 0.0

        self.observation_space = Box(low=-10, high=+10, shape=[OBSERVATION_SPACE])

    def step(self, action, project=True):
        obs, r, done, info = super(CustomEnv, self).step(np.clip(np.array(action), 0.0, 1.0))
        self.episode_length += 1

        # early termination penalty
        if done and self.episode_length < self.time_limit:
            r -= 2

        original_reward = super(CustomEnv, self).reward()
        self.episode_original_reward += original_reward
        self.episode_shaped_reward += r

        state_desc = self.get_state_desc()

        # activation penalty
        self.episode_activation_penalty += np.sum(np.array(self.osim_model.get_activations()) ** 2) * 0.001
        # velocity matching penalty on X, Z direction
        self.episode_vx_penalty += (state_desc["body_vel"]["pelvis"][0] - state_desc["target_vel"][0]) ** 2
        self.episode_vz_penalty += (state_desc["body_vel"]["pelvis"][2] - state_desc["target_vel"][2]) ** 2

        if done:
            info['episode'] = {
                'r': self.episode_original_reward,
                'l': self.episode_length,
                "shaped_reward": self.episode_shaped_reward,
                "activation_penalty": self.episode_activation_penalty,
                "vx_penalty": self.episode_vx_penalty,
                "vz_penalty": self.episode_vz_penalty
            }

        return obs, r, done, info

    def reset(self, project=True):
        super().reset(project=project)
        self.episode_length = 0
        self.episode_original_reward = 0.0
        self.episode_shaped_reward = 0.0
        self.episode_activation_penalty = 0.0
        self.episode_vx_penalty = 0.0
        self.episode_vz_penalty = 0.0
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
        res = res + cm_pos

        # information about target velocity
        target_vx, target_vz = state_desc["target_vel"][0], state_desc["target_vel"][2]
        current_vx, current_vz = state_desc["body_vel"]["pelvis"][0], state_desc["body_vel"]["pelvis"][2]
        diff_vx, diff_vz = current_vx - target_vx, current_vz - target_vz
        res = res + [diff_vz, target_vx, diff_vx, diff_vx, target_vz, diff_vz]

        return res

    def reward(self):
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        if not prev_state_desc:
            return 0

        target_vx, target_vz = state_desc["target_vel"][0], state_desc["target_vel"][2]
        current_vx, current_vz = state_desc["body_vel"]["pelvis"][0], state_desc["body_vel"]["pelvis"][2]
        pelvis_y = state_desc["body_pos"]["pelvis"][1]

        reward_x = np.exp(-abs(target_vx - current_vx))
        reward_z = np.exp(-abs(target_vz - current_vz))
        reward = reward_x + reward_z

        penalty = 0.0
        # too low pelvis
        low_pelvis = max(0, 0.7 - pelvis_y)
        penalty += low_pelvis * 20
        # activation penalty
        penalty += np.sum(np.array(self.osim_model.get_activations()) ** 2) * 0.001 * 2
        # velocity matching penalty on X, Z direction
        penalty += abs(current_vx - target_vx)
        penalty += abs(current_vz - target_vz)

        reward -= penalty

        return reward * 0.5


class CustomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, action_repeat):
        super(CustomActionWrapper, self).__init__(env)
        self.action_repeat = action_repeat

    def step(self, action):
        action = self.action(action)
        rew = 0
        for i in range(self.action_repeat):
            obs, r, done, info = self.env.step(action)
            rew += r
            if done:
                break
        info["action"] = action
        return obs, rew, done, info

    def action(self, action):
        return action
