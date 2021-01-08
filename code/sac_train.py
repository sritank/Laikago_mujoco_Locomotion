import gym
import ipdb
import math
from stable_baselines.sac.policies import MlpPolicy
MlpPolicy1=MlpPolicy

from stable_baselines import SAC



from stable_baselines.common.policies import MlpPolicy

import numpy as np

import math
import matplotlib.pyplot as plt


# # Create environment
env = gym.make('QuadGym-v2')
# the noise objects for DDPG

# qpos0_hist=np.ones((1,49))
# qpos0_hist=np.ones((1,28))
qpos0_hist=np.ones((1,121))

act_hist=np.ones((1,12))
# model = SAC(MlpPolicy1, env, verbose=1, gamma=0.99, learning_rate=0.0003, buffer_size=1500000, learning_starts=100, train_freq=1, batch_size=64, tau=0.005, tensorboard_log="./tensorboard/", full_tensorboard_log=True, n_cpu_tf_sess=8)
model3 = SAC(MlpPolicy1, env, verbose=1, gamma=0.99, learning_rate=0.0003, buffer_size=1500000, learning_starts=100, train_freq=1, batch_size=64, tau=0.005, tensorboard_log="./tensorboard/", full_tensorboard_log=True, n_cpu_tf_sess=8)

model.learn(total_timesteps=3000000, log_interval=50)
model.save("./trained_models/sac_quad")

model=SAC.load("./trained_models/sac_quad")

# trained agent
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