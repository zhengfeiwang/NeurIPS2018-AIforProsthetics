import gym
from gym.spaces import Box
from osim.env import ProstheticsEnv

ACTION_SPACE = 19
MY_OBSERVATION_SPACE = 194


# referenced from official repo
def observation_process(state_desc):
    res = []
    pelvis = None

    for body_part in ['pelvis', 'head', 'torso', 'femur_l', 'femur_r', 'calcn_l', 'tibia_l', 'toes_l', 'talus_l', 'pros_foot_r', 'pros_tibia_r']:
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
            res += cur_upd

    for joint in ["ankle_l","ankle_r","back","hip_l","hip_r","knee_l","knee_r"]:
        res += state_desc["joint_pos"][joint]
        res += state_desc["joint_vel"][joint]
        res += state_desc["joint_acc"][joint]

    for muscle in ['abd_r', 'add_r', 'hamstrings_r', 'bifemsh_r', 'glut_max_r', 'iliopsoas_r', 'rect_fem_r', 'vasti_r', 'abd_l', 'add_l', 'hamstrings_l', 'bifemsh_l', 'glut_max_l', 'iliopsoas_l', 'rect_fem_l', 'vasti_l', 'gastroc_l', 'soleus_l', 'tib_ant_l']:
        res += [state_desc["muscles"][muscle]["activation"]]
        res += [state_desc["muscles"][muscle]["fiber_length"]]
        res += [state_desc["muscles"][muscle]["fiber_velocity"]]

    cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(2)]
    res = res + cm_pos + state_desc["misc"]["mass_center_vel"] + state_desc["misc"]["mass_center_acc"]

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
            elif self.reward_type == 'shaped':
                pelvis_velocity = obs["body_vel"]["pelvis"][0]
                survival = 0.01
                lean = min(0, obs["body_pos"]["head"][0] - obs["body_pos"]["pelvis"][0]) * 0.1
                joint = sum([max(0, knee - 0.1) for knee in [obs["joint_pos"]["knee_l"][0], obs["joint_pos"]["knee_r"][0]]]) * 0.02
                reward = pelvis_velocity * 0.01 + survival + lean - joint
                # pelvis too low
                if obs["body_pos"]["pelvis"][1] < 0.75:
                    reward -= 2 * survival

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
