import numpy as np
from osim.env import ProstheticsEnv
from utils import process_observation


class Evaluator(object):
    def __init__(self, frameskip, episode_length=300, render=False):
        self.env = ProstheticsEnv(visualize=render)
        self.frameskip = frameskip
        self.episode_length = episode_length

    def __call__(self, agent, check):
        episode_reward = 0.0
        episode_steps = 0

        observation = self.env.reset(project=False)
        observation = process_observation(observation)

        done = False
        while not done and episode_steps < self.episode_length:
            action = agent.compute_action(observation)
            action = np.clip(action, 0.0, 1.0)

            for _ in range(self.frameskip):
                observation, reward, done, _ = self.env.step(action, project=False)
                episode_steps += 1
                episode_reward += reward

                if check:
                    print("action: {}".format(action))
                    print("step #{}:".format(episode_steps))
                    print("pelvis")
                    print(" - height: {}".format(observation["body_pos"]["pelvis"][1]))
                    print(" - velocity: {}".format(observation["body_vel"]["pelvis"]))
                    print("environment reward: {}".format(reward))
                if done:
                    break

            observation = process_observation(observation)

        return episode_reward, episode_steps
    
    def close(self):
        self.env.close()
