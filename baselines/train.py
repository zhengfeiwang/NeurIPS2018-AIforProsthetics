import argparse
import ray
import ray.rllib.agents.ddpg as ddpg
from ray.tune.registry import register_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLlib version AI for Prosthetics Challenge")
    parser.add_argument()


def env_creator():
    from custom_env import CustomEnv
    env = CustomEnv(action_repeat=1)
    return env

config = ddpg.DEFAULT_CONFIG.copy()
config["actor_hiddens"] = [800, 400]
config["actor_hidden_activation"] = "selu"
config["critic_hiddens"] = [800, 400]
config["critic_hidden_activation"] = "selu"
config["schedule_max_timesteps"] = 1000
config["num_workers"] = 1
config["gpu"] = True
config["devices"] = ["/gpu:%d" % i for i in range(4)]
config["tf_session_args"]["device_count"] = {"GPU": 4}
horizon = 100
register_env("ProstheticsEnv", env_creator)
ray.init(num_cpus=1)
trainer = ddpg.DDPGAgent(env="ProstheticsEnv", config=config)

i = 1
while (True):
    trainer.train()
    print('training step: #{}'.format(i))

    i += 1

    if i % 100 == 0:
        checkpoint = trainer.save()
        print('[checkpoint] No.{}'.format(i))
