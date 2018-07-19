import argparse
import os
from osim.env import ProstheticsEnv
import ray
import ray.rllib.agents.ddpg as ddpg
from ray.tune.registry import register_env

MAX_STEPS_PER_ITERATION = 1000


def env_creator(env_config):
    from custom_env import CustomEnv
    env = CustomEnv(action_repeat=args.action_repeat)
    return env


def configure(args):
    config = ddpg.DEFAULT_CONFIG.copy()

    # hard code
    # Nothing now...

    # according to arguments
    """
    actor_hiddens = []
    actor_layers = args.actor_hiddens.split('-')
    for actor_layer in actor_layers:
        actor_hiddens.append(int(actor_layer))
    critic_hiddens = []
    critic_layers = args.critic_hiddens.split('-')
    for critic_layer in critic_layers:
        critic_hiddens.append(int(critic_layer))
    
    config["actor_hiddens"] = actor_hiddens
    config["actor_hidden_activation"] = args.actor_activation
    config["critic_hiddens"] = critic_hiddens
    config["critic_hidden_activation"] = args.critic_activation
    """
    
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLlib version AI for Prosthetics Challenge")
    # model
    parser.add_argument("--actor-hiddens", default="400-300", type=str, help="Actor architecture")
    parser.add_argument("--critic-hiddens", default="400-300", type=str, help="Critic architecture")
    parser.add_argument("--actor-activation", default="relu", type=str, help="Actor activation function")
    parser.add_argument("--critic-activation", default="relu", type=str, help="Critic activation function")
    # hyperparameters
    parser.add_argument("--action-repeat", default=1, type=int, help="repeat time for each action")
    # checkpoint
    parser.add_argument("--checkpoint-dir", default="output", type=str, help="checkpoint output directory")
    parser.add_argument("--checkpoint-id", default=None, type=str)
    
    args = parser.parse_args()

    ray.init(num_cpus=1)

    register_env("ProstheticsEnv", env_creator)
    config = configure(args)

    agent = ddpg.DDPGAgent(env="ProstheticsEnv", config=config)
    checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint-" + str(args.checkpoint_id))
    agent.restore(checkpoint_path=checkpoint_path)

    env = ProstheticsEnv(visualize=True)
    observation = env.reset()

    episode_reward = 0.
    steps = 0
    done = False

    while steps < MAX_STEPS_PER_ITERATION and not done:
        action = agent.compute_action(observation)
        for _ in range(args.action_repeat):
            observation, reward, done, _ = env.step(action)
            steps += 1
            episode_reward += reward
            if done or steps >= MAX_STEPS_PER_ITERATION:
                break
    
    print('reward:', episode_reward)
