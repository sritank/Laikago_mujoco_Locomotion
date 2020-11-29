import gym
import ipdb
import math
from stable_baselines.sac.policies import MlpPolicy
MlpPolicy1=MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy
MlpPolicy2=MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from stable_baselines import SAC



from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1

from stable_baselines import PPO2

from stable_baselines import TRPO

import numpy as np

import math
import matplotlib.pyplot as plt


# # Create environment
env = gym.make('QuadGym-v2')
# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.ones(n_actions)*0.15, sigma=float(0.1) * np.ones(n_actions))

model = DDPG(MlpPolicy2, env, verbose=1, param_noise=param_noise, action_noise=action_noise, render=False, buffer_size=1500000, random_exploration=0.0, tensorboard_log="./tensorboard/ddpg" )
# model = DDPG(MlpPolicy2, env, gamma=0.99, memory_policy=None, nb_train_steps=500, nb_rollout_steps=50, nb_eval_steps=300, param_noise=None, action_noise=action_noise, normalize_observations=False, tau=0.002, batch_size=250, normalize_returns=False, enable_popart=False, observation_range=(-10.0,10.0), critic_l2_reg=0.0, actor_lr=0.0005, critic_lr=0.0005, clip_norm=None, render=False, render_eval=False, buffer_size=1000000, verbose=1, _init_setup_model=True)

# model.learn(total_timesteps=1000000)
# model.save("ddpg_quad2")
# qpos0_hist=np.ones((1,49))
# qpos0_hist=np.ones((1,28))
qpos0_hist=np.ones((1,109))
# model = SAC(MlpPolicy1, env, verbose=1, gamma=0.99, learning_rate=0.0003, buffer_size=1000000, learning_starts=100, train_freq=1, batch_size=64, tau=0.005)
# model = SAC(MlpPolicy1, env, verbose=1, gamma=0.9, learning_rate=0.03, buffer_size=5000, learning_starts=100, train_freq=10, batch_size=100, tau=0.05)

# model.learn(total_timesteps=3000000, log_interval=100)
# model.save("sac_quad2")

# model = PPO2(MlpPolicy, env, verbose=1, max_grad_norm=0.5, n_steps=400, learning_rate=0.00025, nminibatches=200, noptepochs=40)
# model.learn(total_timesteps=5000000)
# model.save("ppo2_quad")


# model = TRPO(MlpPolicy, env, verbose=1, gamma=0.99, timesteps_per_batch=102400, max_kl=0.01, cg_iters=10, lam=0.98, entcoeff=0.0, cg_damping=0.01, vf_stepsize=0.0003, vf_iters=3, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1)

# model = TRPO(MlpPolicy, env, verbose=1, gamma=0.91, timesteps_per_batch=1000, max_kl=0.05, cg_iters=10, lam=0.9, entcoeff=0.001, cg_damping=0.05, vf_stepsize=0.0003, vf_iters=3, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1)
# model.learn(total_timesteps=14200000)
# model.save("trpo_quad")

# model = PPO1.load("ppo1_quad")

# model=TRPO.load("trpo_quad")
# model=SAC.load("sac_quad2")
model=DDPG.load("ddpg_quad2")
# model=DDPG.load("ddpg_cheetah")
# Enjoy trained agent
obs = env.reset()
for i in range(4000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(action)
    # print(obs[2])
    print(info['z'])
    # print(i)
    # print(dones)
    qpos0_hist = np.vstack((qpos0_hist,obs))
    env.render()
qpos0_hist = np.delete(qpos0_hist,0,0)

plt.plot(qpos0_hist[:,2])
plt.show()
# ipdb.set_trace()