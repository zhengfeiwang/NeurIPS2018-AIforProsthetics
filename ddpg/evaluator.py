from osim.env import ProstheticsEnv
from my_env import observation_process


class Evaluator(object):
    def __init__(self, args, render=False):
        self.env = ProstheticsEnv(visualize=render)
        self.validation_episodes = args.validation_episodes
        self.action_repeat = args.action_repeat
        self.max_episode_length = args.max_episode_length

    def __call__(self, policy, logger):
        rewards = []
        steps = []

        for _ in range(self.validation_episodes):
            obs = self.env.reset(project=False)
            observation = observation_process(obs)
            episode_steps = 0
            episode_rewards = 0.

            done = False
            while not done and episode_steps < self.max_episode_length:
                action = policy(observation)
                # action repeat
                for _ in range(self.action_repeat):
                    episode_steps += 1
                    obs, reward, done, _ = self.env.step(action, project=False)
                    episode_rewards += reward
                    observation = observation_process(obs)

                    if done:
                        logger.debug('-------------------- validation --------------------')
                        logger.debug('validation episode done...')
                        logger.debug('scores = {}, steps = {}'.format(episode_rewards, episode_steps))
                        logger.debug('pelvis position: {}'.format(obs['body_pos']['pelvis']))
                        logger.debug('----------------------------------------------------')
                        break
            
            rewards.append(episode_rewards)
            steps.append(episode_steps)

        return rewards, steps
