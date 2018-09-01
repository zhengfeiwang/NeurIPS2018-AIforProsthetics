from osim.env import ProstheticsEnv
import gym
from gym.spaces import Box
import random

OBSERVATION_SPACE = 224


class CustomEnv(ProstheticsEnv):

    def __init__(self, visualize=True, integrator_accuracy=5e-5):
        super().__init__(visualize, integrator_accuracy)
        self.observation_space = Box(low=-10, high=+10, shape=[OBSERVATION_SPACE])
    
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
        res = res + cm_pos + state_desc["misc"]["mass_center_vel"] + state_desc["misc"]["mass_center_acc"]

        return res

    def reward(self):
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        # print("-------------------------------------")
        # print(state_desc["body_pos"]["pelvis"])
        # print(state_desc["body_pos"]["pros_foot_r"][1])
        if not prev_state_desc:
            return 0

        # original reward
        # reward = 9.0 - (state_desc["body_vel"]["pelvis"][0] - 3.0) ** 2
        # reward = state_desc["body_vel"]["pelvis"][0] * 2 + 2
        reward = state_desc["body_vel"]["pelvis"][0] * 2 + 2
        original_reward = reward

        pros_foot_r = state_desc["body_pos"]["pros_foot_r"][1]
        reward -= max(0, 2 * (pros_foot_r - 0.3))
        # print("pros foot", pros_foot_r)

        lean = min(0.3, max(0, state_desc["body_pos"]["pelvis"][0] - state_desc["body_pos"]["head"][0]))
        reward -= lean * 10

        # joint = sum([max(0, knee - 0.1) for knee in
        #              [state_desc["joint_pos"]["knee_l"][0], state_desc["joint_pos"]["knee_r"][0]]]) * 0.03

        knee_l = state_desc["joint_pos"]["knee_l"][0]
        knee_r = state_desc["joint_pos"]["knee_r"][0]
        # print("knee joint", knee_r)
        # reward -= knee_r * 0.5

        pelvis = state_desc["body_pos"]["pelvis"][1]
        reward -= max(0, 0.75 - pelvis) * 10
        # print("pelvis", pelvis)

        pros_tibia = state_desc["body_pos"]["pros_tibia_r"][1]
        reward -= max(0, pros_tibia - 0.6) * 5
        print(knee_l, knee_r, lean, " --> ", pros_tibia, pelvis)

        front_foot = max(state_desc["body_pos"]["toes_l"][0], state_desc["body_pos"]["pros_foot_r"][0])
        reward -= max(0, state_desc["body_pos"]["pelvis"][0] - front_foot) * 10

        reward -= max(0, - 1.5 - knee_l) * 5

        print(original_reward, " - ", reward, " = ", original_reward - reward)
        return reward * 0.1


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

        # 5 kua gu, 6 da tui qian ce
        # 2 da tui hou ce, 3 xiao tui wan qu, 4 da tun ji

        # for i in range(19):
        #     action[i] = min(1, max(0, action[i]))

        # state_desc = self.env.get_state_desc()
        # pros_foot_r = state_desc["body_pos"]["pros_foot_r"][1]
        # # print("pros foot ", pros_foot_r)
        # offset = pros_foot_r - 0.2
        # action[2] += random.random() * offset
        # action[3] += random.random() * offset
        # action[4] += random.random() * offset
        #
        # action[5] -= random.random() * offset
        # action[6] -= random.random() * offset

        # print("action ", action[2], action[3], action[4], "|", action[5], action[6])
        # print(pros_foot_r)

        # if random.random() < 1:
        #     action[2] = 1
        #     action[3] = 1
        #     action[4] = 0.5
        #
        #     action[5] = 0
        #     action[6] = 0

        # for i in range(19):
        #     if i != 4:
        #         action[i] = 0
        #     else:
        #         action[i] = 1
        return action


def make_env():
    env = CustomEnv(visualize=True, integrator_accuracy=1e-3)
    env = CustomActionWrapper(env)
    return env
