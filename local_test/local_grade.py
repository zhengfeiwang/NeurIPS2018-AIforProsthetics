import random
from local_test.baselines.common.vec_env.vec_normalize import VecNormalize
from local_test.baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import local_test.baselines.ppo2.ppo2 as ppo2
import local_test.baselines.ppo2.policies as policies
from local_test.local_grade_env import LocalGradeEnv, LocalGradeRepeatActionEnv
from local_test.baselines.logger import configure

import tensorflow as tf


def make_local_grade_env():
    random_seeds = [random.randint(0, 2 ** 32 - 1) for _ in range(10)]
    local_env = LocalGradeEnv(random_seeds, visualization=False)
    repeated_local_env = LocalGradeRepeatActionEnv(env=local_env, repeat=2)
    return repeated_local_env


configure(dir="./logs")
config = tf.ConfigProto()
with tf.Session(config=config):
    env = DummyVecEnv([make_local_grade_env])
    env = VecNormalize(env)
    ppo2.learn(
        policy=policies.MlpPolicy,
        env=env,
        nsteps=100000,
        nminibatches=1,
        lam=0.95,
        gamma=0.99,
        noptepochs=5,
        log_interval=1,
        vf_coef=0.5,
        ent_coef=0.001,
        lr=3e-4,
        cliprange=0.2,
        save_interval=10,
        load_path="../checkpoints/00160",
        total_timesteps=100000
    )
