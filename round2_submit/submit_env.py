import gym
from gym.spaces import Box


class Round2SubmitEnv:
    def __init__(self):
        from osim.http.client import Client
        remote_base = "http://grader.crowdai.org:1730"
        self.crowdai_token = "e47cb9f7fd533dc036dbd5d65d0d68c3"
        self.client = Client(remote_base)
        self.first_reset = True
        self.action_space = Box(low=0, high=1, shape=[19])
        self.observation_space = Box(low=-3, high=3, shape=[224])
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
        pelvis_vx = obs['body_vel']['pelvis'][0]
        print(f'timestamp={self.episodic_length:3d} score={self.score:5.2f} velocity={pelvis_vx:3.2f}')
        import sys
        sys.stdout.flush()
        return self.get_observation(obs), rew, done, info

    def close(self):
        pass

    def get_observation(self, state_desc):
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

        return obs, total_reward, done, info
        

def make_submit_env():
    env = Round2SubmitEnv()
    env = SubmitRepeatActionEnv(env=env, repeat=2)
    return env
