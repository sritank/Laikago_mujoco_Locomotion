import gym
import ipdb
import math
import matplotlib.pyplot as plt

import numpy as np

# qpos0_hist=np.ones((1,109))
qpos0_hist=np.ones((1,137))

# qpos0_hist=np.ones((1,29))
# qpos0_hist=np.ones((1,17))
i=0
env = gym.make('QuadGym-v2')


for i_episode in range(2):
    observation = env.reset()
    for t in range(0,2000):
        env.render()
        #printing the observation space
        # print(observation) 
        #storing the number of possible actions
        # ipdb.set_trace()
        action = np.zeros(12)#env.action_space.sample() 
        # action = np.zeros(2)
        # action = np.zeros(6)
        # action[2]= 35 + math.sin(t/50)* 10 
        # action[5]= 35 + math.sin(t/50)*10
        # action[8]= 30 + math.cos(t/50)*8
        # action[11]= 30 + math.cos(t/50)*8

        action[2]= math.sin(t/30)
        action[5]= math.sin(t/30)
        action[8]= math.cos(t/30)
        action[11]= math.cos(t/30)

        # action[2]= -2#*math.sin(t/20)
        # action[5]= -2#*math.sin(t/20)
        # action[8]= -2#*math.cos(t/20)
        # action[11]= -2#*math.cos(t/20)

        action[1]= math.sin(t/20)
        action[4]= math.sin(t/20)
        action[7]= math.cos(t/20)
        action[10]= math.cos(t/20)

        # action[1]= -9#*(50-15*math.sin(t/20))
        # action[4]= -9#*(50-15*math.sin(t/20))
        # action[7]= 7#+15*math.cos(t/20)
        # action[10]= 7#+15*math.cos(t/20)

        action[0]= 0#-5*math.sin(t/80)
        action[3]= 0#+5*math.sin(t/80)
        action[6]= 0#-5-2*math.sin(t/20)
        action[9]= 0#5+2*math.sin(t/20)

        observation, reward, done, info = env.step(action)
        
        # qpos0_hist = np.append(qpos0_hist,reward)
        # qpos0_hist = np.vstack((qpos0_hist,observation))
        # if t%10==0:
        #     print(info)
        # if done:
        #     print("Episode finished after {} timesteps".format(t+1))
            # break
        # print("done=")
        # print(done)
        print(info['z'])
        # print(observation[2])
        print(t)
env.close()

qpos0_hist = np.delete(qpos0_hist,0,0)

# plt.plot(qpos0_hist[:,2])
# plt.show()
# ipdb.set_trace()
