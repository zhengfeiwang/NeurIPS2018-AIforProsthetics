import gym
from gym.spaces import Box


class SubmitEnv:
    def __init__(self):
        from osim.http.client import Client
        remote_base = "http://grader.crowdai.org:1729"
        self.crowdai_token = "e47cb9f7fd533dc036dbd5d65d0d68c3"
        self.client = Client(remote_base)
        self.first_reset = True
        self.action_space = Box(low=0, high=1, shape=[19])
        self.observation_space = Box(low=-3, high=3, shape=[158])
        self.episodic_length = 0
        self.score = 0.0

        self.reward_range = None
        self.metadata = None

    def reset(self):
        self.episodic_length = 0
        self.score = 0
        if self.first_reset:
            self.first_reset = False
            return self.get_observation(self.client.env_create(self.crowdai_token, env_id="ProstheticsEnv"))
        else:
            obs = self.client.env_reset()
            if obs is None:
                self.client.submit()
                print('SUBMITTED')
                import sys
                sys.exit(0)
            return self.get_observation(obs)

    def step(self, action):
        [obs, rew, done, info] = self.client.env_step(action.tolist(), True)
        self.episodic_length += 1
        self.score += rew
        print(f'timestamp={self.episodic_length:4d} score={self.score:5.2f}')
        import sys
        sys.stdout.flush()
        return self.get_observation(obs), rew, done, info

    def close(self):
        pass

    def get_observation(self, state_desc):
        res = []
        pelvis = None

        for body_part in ["pelvis", "head", "torso", "toes_l", "toes_r", "talus_l", "talus_r"]:
            if body_part in ["toes_r", "talus_r"]:
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


class SubmitRepeatActionEnv(gym.ActionWrapper):
    def __init__(self, env, repeat=2):
        super(SubmitRepeatActionEnv, self).__init__(env)
        self._repeat = repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._repeat):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        info['mode'] = 'submit'
        return obs, total_reward, done, info


def make_submit_env():
    env = SubmitEnv()
    env = SubmitRepeatActionEnv(env=env, repeat=2)
    return env
