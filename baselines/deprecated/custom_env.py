import gym
from osim.env import ProstheticsEnv


class CustomEnv(ProstheticsEnv):
    def __init__(self, visualize=False, integrator_accuracy=1e-3):
        super(CustomEnv, self).__init__(visualize, integrator_accuracy)
        self.episode_length = 0
        self.episode_original_reward = 0.0

    def step(self, action):
        super(CustomEnv, self).step(action)
        self.episode_length += 1

        original_reward = super(CustomEnv, self).reward()
        self.episode_original_reward += original_reward
        shaped_reward, penalty = self.reward()

        obs = self.get_observation()
        done = self.is_done() or (self.osim_model.istep >= self.spec.timestep_limit)

        info = {}
        info['step'] = {'original_reward': original_reward, 'timestamp': self.episode_length}
        info['penalty'] = penalty
        if done:
            info['episode'] = {'r': self.episode_original_reward, 'l': self.episode_length}

        return obs, shaped_reward, done, info

    def reward(self):
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        if not prev_state_desc:
            return 0

        penalty = {}

        pelvis_vx = state_desc['body_vel']['pelvis'][0]

        pros_foot_y = state_desc['body_pos']['pros_foot_r'][1]
        penalty['pros_foot_too_high'] = max(0, 2 * (pros_foot_y - 0.3))

        pelvis_x = state_desc['body_pos']['pelvis'][0]
        head_x = state_desc['body_pos']['head'][0]
        penalty['lean_back'] = 10 * min(0.3, max(0, pelvis_x - head_x))

        pelvis_y = state_desc['body_pos']['pelvis'][1]
        penalty['pelvis_too_low'] = 10 * max(0, 0.75 - pelvis_y)

        pros_tibia_y = state_desc['body_pos']['pros_tibia_r'][1]
        penalty['pros_tibia_too_high'] = 5 * max(0, pros_tibia_y - 0.6)

        head_z = state_desc['body_pos']['head'][2]
        pelvis_z = state_desc['body_pos']['pelvis'][2]
        penalty['lean_side'] = 10 * max(0, abs(head_z - pelvis_z) - 0.3)

        reward = pelvis_vx * 2 + 2
        """
        for key in penalty.keys():
            reward -= penalty[key]
        reward = 0.1 * reward
        """

        return reward, penalty

    def reset(self):
        super().reset()
        self.episode_length = 0
        self.episode_original_reward = 0.0
        return self.get_observation()

    def get_observation(self):
        state_desc = self.get_state_desc()

        res = []
        pelvis = None

        for body_part in ["pelvis", "head", "torso", "toes_l", "toes_r", "talus_l", "talus_r"]:
            if self.prosthetic and body_part in ["toes_r", "talus_r"]:
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
                cur_upd[6:7] = [cur[i] - pelvis[i] for i in range(6, 7)]
                res += cur_upd  # use relative position

        for joint in ["ankle_l", "ankle_r", "back", "hip_l", "hip_r", "knee_l", "knee_r"]:
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


class RepeatActionEnv(gym.ActionWrapper):
    def __init__(self, env, repeat=2):
        super(RepeatActionEnv, self).__init__(env)
        self._repeat = repeat

    def step(self, action):
        total_reward = total_original_reward = 0.0
        total_penalty = None
        for _ in range(self._repeat):
            obs, reward, done, info = self.env.step(action)
            total_original_reward += info['step']['original_reward']
            total_reward += reward
            if total_penalty is None:
                total_penalty = info['penalty']
            else:
                for key in info['penalty'].keys():
                    total_penalty[key] += info['penalty'][key]
            if done:
                break
        info['step']['original_reward'] = total_original_reward
        info['step']['shaped_reward'] = reward
        info.pop('penalty')
        info['step']['penalty'] = total_penalty

        state_desc = self.env.get_state_desc()
        info['action'] = action
        info['position'] = {
            'head': state_desc['body_pos']['head'],
            'pelvis': state_desc['body_pos']['pelvis'],
            'calcn_l': state_desc['body_pos']['calcn_l'],
            'pros_foot_r': state_desc['body_pos']['pros_foot_r']
        }

        # episode done information
        if done:
            info['episode']['pelvis_x'] = state_desc['body_pos']['pelvis'][0]
            info['episode']['done'] = []
            # fall forward
            if state_desc['body_pos']['head'][0] - state_desc['body_pos']['pelvis'][0] > 0.5:
                info['episode']['done'].append('forward')
            # fall behind
            if state_desc['body_pos']['pelvis'][0] - state_desc['body_pos']['head'][0] > 0.5:
                info['episode']['done'].append('backward')
            # fall side
            if abs(state_desc['body_pos']['head'][2] - state_desc['body_pos']['pelvis'][2]) > 0.5:
                info['episode']['done'].append('side')

        return obs, total_reward, done, info


def make_env():
    env = CustomEnv(visualize=True, integrator_accuracy=1e-3)
    env = RepeatActionEnv(env=env, repeat=2)
    return env
