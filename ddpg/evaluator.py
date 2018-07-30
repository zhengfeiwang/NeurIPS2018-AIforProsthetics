import numpy as np
from osim.env import ProstheticsEnv
from utils import process_observation


class Evaluator(object):
    def __init__(self, action_repeat, render=False):
        self.env = ProstheticsEnv(visualize=render)
        self.action_repeat = action_repeat
        self.episode_length_max = 300 // self.action_repeat

    def __call__(self, agent):
        # the environment is deterministic, so only need to evaluate once
        episode_reward = 0.0
        episode_steps = 0

        observation = self.env.reset(project=False)
        observation = process_observation(observation)

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
            observation = process_observation(observation)
        
        return episode_reward, episode_steps
    
    def close(self):
        self.env.close()
