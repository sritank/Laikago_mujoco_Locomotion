import gym
from ddpg import DDPG
import ipdb
import math

from stable_baselines.common.policies import MlpPolicy

from stable_baselines import TRPO

import numpy as np



# # Create environment
env = gym.make('QuadGym-v0')

model = TRPO(MlpPolicy, env, verbose=1, gamma=0.99, timesteps_per_batch=102400, max_kl=0.01, cg_iters=10, lam=0.98, entcoeff=0.0, cg_damping=0.01, vf_stepsize=0.0003, vf_iters=3, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1)

# model = TRPO(MlpPolicy, env, verbose=1, gamma=0.91, timesteps_per_batch=1000, max_kl=0.05, cg_iters=10, lam=0.9, entcoeff=0.001, cg_damping=0.05, vf_stepsize=0.0003, vf_iters=3, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1)
model.learn(total_timesteps=14200000)
model.save("trpo_quad")

# model=TRPO.load("trpo_quad")

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(action)
    print(obs[2])
    print(info['z'])
    # print(i)
    # print(dones)
    env.render()