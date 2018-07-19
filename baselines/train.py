import argparse
import ray
import ray.rllib.agents.ddpg as ddpg
from ray.tune.registry import register_env


def env_creator(env_config):
    from custom_env import CustomEnv
    env = CustomEnv(action_repeat=args.action_repeat)
    return env


def configure(args):
    config = ddpg.DEFAULT_CONFIG.copy()

    # hard code
    # Nothing now...

    # according to arguments
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
    config["learning_starts"] = args.warmup
    
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLlib version AI for Prosthetics Challenge")
    # Ray
    parser.add_argument("--redis-address", default=None, type=str, help="address of the Redis server")
    parser.add_argument("--num-cpus", default=24, type=int, help="number of local cpus")
    # model
    parser.add_argument("--actor-hiddens", default="400-300", type=str, help="Actor architecture")
    parser.add_argument("--critic-hiddens", default="400-300", type=str, help="Critic architecture")
    parser.add_argument("--actor-activation", default="relu", type=str, help="Actor activation function")
    parser.add_argument("--critic-activation", default="relu", type=str, help="Critic activation function")
    # hyperparameters
    parser.add_argument("--batch-size", default=256, type=int, help="minibatch size")
    parser.add_argument("--actor-learning-rate", default=1e-4, type=float, help="Actor learning rate")
    parser.add_argument("--critic-learning-rate", default=1e-3, type=float, help="Critic learning rate")
    parser.add_argument("--action-repeat", default=4, type=int, help="repeat time for each action")
    parser.add_argument("--warmup", default=10000, type=int, help="number of random action before training")
    # checkpoint
    parser.add_argument("--checkpoint-dir", default="output", type=str, help="checkpoint output directory")
    parser.add_argument('--checkpoint-interval', default=10, type=int, help="iteration interval for checkpoint")
    
    args = parser.parse_args()

    if args.redis_address is not None:
        ray.init(redis_address=args.redis_address)
    else:
        ray.init(num_cpus=args.num_cpus)

    register_env("ProstheticsEnv", env_creator)
    config = configure(args)

    agent = ddpg.DDPGAgent(env="ProstheticsEnv", config=config)

    # agent training
    n_iteration = 1
    while (True):
        agent.train()
        print('training step: #{}'.format(n_iteration))

        n_iteration += 1

        if n_iteration % args.checkpoint_interval == 0:
            checkpoint = agent.save(args.checkpoint_dir)
            print('[checkpoint] No.{}'.format(n_iteration))
