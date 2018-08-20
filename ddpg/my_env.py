import gym
from gym.spaces import Box
from osim.env import ProstheticsEnv

ACTION_SPACE = 19
MY_OBSERVATION_SPACE = 72   # may change due to the observation processing method


def observation_process(observation):
    # observation processing, output dimension is 72D
    res = []
    BODY_PARTS = [
        'femur_r', 'pros_tibia_r', 'pros_foot_r', 
        'femur_l', 'tibia_l', 'talus_l', 'calcn_l', 'toes_l', 
        'torso', 'head'
    ]
    JOINTS = [
        'ground_pelvis', 
        'hip_r', 'knee_r', 'ankle_r', 
        'hip_l', 'knee_l', 'ankle_l', 
        'back'
    ]
    
    # body parts positions relative to pelvis - 31D
    res += [observation['body_pos']['pelvis'][1]]   # absolute pelvis.y
    pelvis_pos = observation['body_pos']['pelvis']
    for body_part in BODY_PARTS:
        # x, y, z - axis
        for axis in range(3):
            res += [observation['body_pos'][body_part][axis] - pelvis_pos[axis]]

    # pelvis velocity - 3D
    pelvis_vel = observation['body_vel']['pelvis']
    res += pelvis_vel
        
    # joints absolute angle - 17D
    for joint in JOINTS:
        for i in range(len(observation['joint_pos'][joint])):
            res += [observation['joint_pos'][joint][i]]
        
    # joints absolute angular velocity - 17D
    for joint in JOINTS:
        for i in range(len(observation['joint_vel'][joint])):
            res += [observation['joint_vel'][joint][i]]
        
    # center of mass position and velocity - 4D
    for axis in range(2):
        res += [observation['misc']['mass_center_pos'][axis] - pelvis_pos[axis]]
    for axis in range(2):
        res += [observation['misc']['mass_center_vel'][axis] - pelvis_vel[axis]]
            
    return res


class MyEnv:
    def __init__(self, accuracy=5e-5, action_repeat=1, reward_type='2018'):
        self.env = ProstheticsEnv(visualize=False, integrator_accuracy=accuracy)
        self.action_repeat = action_repeat
        self.reward_type = reward_type
        self.steps_count = 0
    
    def step(self, action):
        action = [float(action[i]) for i in range(len(action))]
        reward = 0.0

        for _ in range(self.action_repeat):
            self.steps_count += 1
            obs, r, done, _ = self.env.step(action, project=False)
            observation = observation_process(obs)

            if self.reward_type == '2018':
                reward += r
            else:
                assert False, 'undefined reward type...'

            if done:
                break
        
        return observation, reward, done
    
    def reset(self):
        obs = self.env.reset(project=False)
        self.steps_count = 0
        observation = observation_process(obs)
        return observation

    def close(self):
        self.env.close()
