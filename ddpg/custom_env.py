from osim.env import ProstheticsEnv
import numpy as np
from gym.spaces import Box

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
                # 2018 reward
                original_reward = reward * 0.01
                # hzwer penalty
                lean = min(0.3, max(0, observation["body_pos"]["pelvis"][0] - observation["body_pos"]["head"][0] - 0.15)) * 0.05
                joint = sum([max(0, knee - 0.1) for knee in [observation["joint_pos"]["knee_l"][0], observation["joint_pos"]["knee_r"][0]]]) * 0.03
                penalty = lean + joint
                # survival
                survival = 0.01
                reward = original_reward - penalty + survival
            else:
                assert False, 'unknown reward type...'

            cumulative_reward += reward
            if done:
                break
        # transform dictionary to 1D vector
        observation = self.custom_observation(observation)
        # clip rewards to [-1, 1]
        clipped_reward = np.clip(cumulative_reward, -1.0, 1.0)
        return observation, clipped_reward, done, info

    def reset(self):
        observation = self.env.reset(project=False)
        self.prev_pelvis_pos = 0.0
        return self.custom_observation(observation)
    
    def custom_observation(self, observation):
        # custom observation space 44 + 3 + 17 + 17 + 4 = 85D
        res = []

        BODY_PARTS = ['femur_r', 'pros_tibia_r', 'pros_foot_r', 'femur_l', 'tibia_l', 'talus_l', 'calcn_l', 'toes_l', 'torso', 'head']
        JOINTS = ['ground_pelvis', 'hip_r', 'knee_r', 'ankle_r', 'hip_l', 'knee_l', 'ankle_l', 'back']
        
        # body parts positions relative to pelvis - (3 + 1) + (3 + 1) * 10D
        # pelvis relative position
        res += [0.0, 0.0, 0.0]
        res += [observation["body_pos"]["pelvis"][1]]   # absolute pelvis.y
        pelvis_pos = observation["body_pos"]["pelvis"]
        for body_part in BODY_PARTS:
            # x, y, z - axis
            for axis in range(3):
                res += [observation["body_pos"][body_part][axis] - pelvis_pos[axis]]
            res += [observation["body_pos"][body_part][1]] # absolute height

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
