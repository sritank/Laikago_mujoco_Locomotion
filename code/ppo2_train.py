import gym
import ipdb
import math


from stable_baselines.common.policies import MlpPolicy

from stable_baselines import PPO2

import numpy as np

import math
import matplotlib.pyplot as plt



# # Create environment
env = gym.make('QuadGym-v2')

qpos0_hist=np.ones((1,109))
act_hist=np.ones((1,8))
# model = PPO2(MlpPolicy, env, verbose=1, max_grad_norm=0.5, n_steps=128, learning_rate=0.00025, nminibatches=4, noptepochs=4, tensorboard_log="./tensorboard/PPO_runs/", full_tensorboard_log=False, n_cpu_tf_sess=8)

# model.learn(total_timesteps=5000000, log_interval=100)
# model.save("./trained_models/ppo2_quad_working")

model = PPO2.load("./trained_models/ppo2_quad_working")

obs = env.reset()
for i in range(5000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    # print(action)
    # print(obs[2])
    # print(info['z'])
    # print(i)
    # print(dones)
    qpos0_hist = np.vstack((qpos0_hist,obs))
    act_hist = np.vstack((act_hist,action))
    env.render()
qpos0_hist = np.delete(qpos0_hist,0,0)
act_hist = np.delete(act_hist,0,0)

plt.plot(qpos0_hist[:,2])
plt.show()

plt.plot(act_hist[:,[1,3,5,7]])
plt.show()
# ipdb.set_trace()