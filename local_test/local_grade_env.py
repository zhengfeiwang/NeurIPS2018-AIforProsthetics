import numpy as np
import gym
from gym.spaces import Box
from osim.env import ProstheticsEnv

OBSERVATION_SPACE = 224


class LocalGradeEnv(ProstheticsEnv):
    def __init__(self, random_seeds, visualization=True, integrator_accuracy=5e-5):
        super().__init__(visualization, integrator_accuracy, difficulty=1)
        self.episode_length = 0
        self.original_reward = 0.0
        self.shaped_reward = 0.0
        self.activation_penalty = 0.0
        self.vx_penalty = 0.0
        self.vz_penalty = 0.0
        self.observation_space = Box(low=-10, high=+10, shape=[OBSERVATION_SPACE])
        # local grade
        self.random_seeds = random_seeds
        self.num_seeds = len(self.random_seeds)
        self.seed_idx = 0
        self.total_length = 0
        self.total_reward = 0.0

    def reset(self):
        if self.seed_idx > 0:
            print('random seed:', self.random_seeds[self.seed_idx - 1])
            print('score:', self.original_reward, 'length:', self.episode_length)

        self.total_length += self.episode_length
        self.total_reward += self.original_reward

        if self.seed_idx == self.num_seeds:
            print('Local Grade Finished!')
            print('mean reward:', self.total_reward / self.num_seeds)
            print('mean length:', self.total_length / self.num_seeds)
            import sys
            sys.exit(0)
        super().reset(project=False, seed=self.random_seeds[self.seed_idx])
        self.seed_idx += 1

        self.episode_length = 0
        self.original_reward = 0.0
        self.shaped_reward = 0.0
        self.activation_penalty = 0.0
        self.vx_penalty = 0.0
        self.vz_penalty = 0.0

        obs = self.get_observation()
        return obs

    def step(self, action):
        obs, rew, done, info = super().step(np.clip(np.array(action), 0.0, 1.0))
        self.episode_length += 1
        self.original_reward += super().reward_round2()
        self.shaped_reward += rew

        state_desc = self.get_state_desc()
        self.activation_penalty += np.sum(np.array(self.osim_model.get_activations()) ** 2) * 0.001
        self.vx_penalty += (state_desc["body_vel"]["pelvis"][0] - state_desc["target_vel"][0]) ** 2
        self.vz_penalty += (state_desc["body_vel"]["pelvis"][2] - state_desc["target_vel"][2]) ** 2

        # target_vx, target_vz = state_desc["target_vel"][0], state_desc["target_vel"][2]
        # pelvis_vx, pelvis_vz = state_desc['body_vel']['pelvis'][0], state_desc['body_vel']['pelvis'][2]
        # print(f'timestamp={self.episode_length:3d} score={self.original_reward:5.2f}')
        # print(f'    target_vx={target_vx:3.2f} current_vx={pelvis_vx:3.2f}')
        # print(f'    target_vz={target_vz:3.2f} current_vz={pelvis_vz:3.2f}')
        return obs, rew, done, info

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


class LocalGradeRepeatActionEnv(gym.ActionWrapper):
    def __init__(self, env, repeat=2):
        super().__init__(env)
        self._repeat = repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._repeat):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info
