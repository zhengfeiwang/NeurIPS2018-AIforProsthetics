from osim.env import ProstheticsEnv


class CustomEnv(ProstheticsEnv):
    def __init__(self, action_repeat, integrator_accuracy):
        self.env = ProstheticsEnv(visualize=False)
        self.env.integrator_accuracy = integrator_accuracy
        self.action_repeat = action_repeat
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        cumulative_reward = 0.0
        for _ in range(self.action_repeat):
            observation, reward, done, info = self.env.step(action)
            cumulative_reward += reward
            if done:
                break
        return observation, cumulative_reward, done, info

    def reset(self, project = True):
        observation = self.env.reset()
        return observation
