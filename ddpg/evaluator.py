import numpy as np
from osim.env import ProstheticsEnv


class Evaluator(object):
    def __init__(self, action_repeat):
        self.env = ProstheticsEnv(visualize=False)
        self.action_repeat = action_repeat
        self.episode_length_max = 1000 // self.action_repeat

    def __call__(self, agent):
        # the environment is deterministic, so only need to evaluate once
        episode_reward = 0.0
        episode_steps = 0

        observation = self.env.reset(project=False)
        observation = self.custom_observation(observation)

        done = False

        while not done and episode_steps <= self.episode_length_max:
            # compute and clip action
            action = agent.compute_action(observation)
            action = np.clip(action, 0.0, 1.0)

            for _ in range(self.action_repeat):
                observation, reward, done, _ = self.env.step(action, project=False)
                episode_steps += 1
                episode_reward += reward

                if done:
                    break
            
            # transform dictionary to 1D vector
            observation = self.custom_observation(observation)
        
        return episode_reward, episode_steps
    
    def custom_observation(self, observation):
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
