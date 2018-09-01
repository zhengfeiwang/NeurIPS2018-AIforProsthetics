from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import baselines.ppo2.ppo2 as ppo2
import baselines.ppo2.policies as policies
from dev.submit_env import make_submit_env
from baselines.logger import configure
from baselines.common import set_global_seeds

import tensorflow as tf


def submit(num_timesteps, seed):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    set_global_seeds(seed)
    with tf.Session(config=config):
        env = DummyVecEnv([make_submit_env])
        env = VecNormalize(env)
        ppo2.learn(policy=policies.MlpPolicy,
                   env=env,
                   nsteps=10000,
                   nminibatches=16,
                   lam=0.95,
                   gamma=0.99,
                   noptepochs=5,
                   log_interval=1,
                   vf_coef=0.5,
                   ent_coef=0.0,
                   lr=lambda _: 3e-4,
                   cliprange=lambda _: 0.2,
                   save_interval=10,
                   load_path='logs/checkpoints/branch0_4steps',
                   total_timesteps=num_timesteps
                   )

if __name__ == '__main__':
    configure(dir="./logs")
    submit(int(1e6), 60730)
