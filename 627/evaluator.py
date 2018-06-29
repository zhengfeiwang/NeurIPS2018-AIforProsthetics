class Evaluator(object):
    def __init__(self, args):
        self.validate_episodes = args.validate_episodes
        self.action_repeat = args.action_repeat

    def __call__(self, env, policy, visualize=False):
        result = []
        for _ in range(self.validate_episodes):
            observation = env.reset()
            episode_rewards = 0.

            done = False
            while not done:
                action = policy(observation)
                # action repeat
                for _ in range(self.action_repeat):
                    observation, reward, done, _ = env.step(action)
                    episode_rewards += reward
                    if visualize:
                        env.render()
                    if done:
                        break
            result.append(episode_rewards)

        return result
