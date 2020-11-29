import gym
import ipdb
import math
import matplotlib.pyplot as plt

import numpy as np

qpos0_hist=np.ones((1,106))
# qpos0_hist=np.ones((1,29))
# qpos0_hist=np.ones((1,17))
i=0
env = gym.make('QuadGym-v2')
# env = gym.make('HalfCheetah-v2')

for i_episode in range(2):
    observation = env.reset()
    for t in range(0,1000):
        env.render()
        #printing the observation space
        # print(observation) 
        #storing the number of possible actions
        # ipdb.set_trace()
        action = np.zeros(8)#env.action_space.sample() 
        # action = np.zeros(2)
        # action = np.zeros(6)
        # action[2]= 35 + math.sin(t/50)* 10 
        # action[5]= 35 + math.sin(t/50)*10
        # action[8]= 30 + math.cos(t/50)*8
        # action[11]= 30 + math.cos(t/50)*8

        # action[2]= 3*math.sin(t/30)
        # action[5]= 3*math.sin(t/30)
        # action[8]= 3*math.cos(t/30)
        # action[11]= 3*math.cos(t/30)

        action[1]= 1*math.sin(t/50)
        action[3]= 1*math.sin(t/50)
        action[5]= 1*math.cos(t/50)
        action[7]= 1*math.cos(t/50)

        # action[1]= -1
        # action[3]= -1
        # action[5]= -1
        # action[7]= -1
        # action[0] = 1*math.cos(t/50)
        # action[1] = 1*math.sin(t/50)
        # action[0] = 1*math.cos(t/15)
        # action[1] = 1*math.sin(t/15)
        # action[0] = 1
        # action[1] = 1

        # action[1]= .5
        # action[3]= .5
        # action[5]= .5
        # action[7]= .5



        # action[1]= 25 #+ math.sin(t/50)* 10
        # action[4]= 25 #+ math.sin(t/50)*10
        # action[7]= 10 #+math.sin(t/50)*10
        # action[10]= 10 #+math.sin(t/50)*10

        # action[0]= -1#+math.cos(t/150)* 20
        # action[3]= 1#+math.cos(t/150)*20
        # action[6]= -1#+math.cos(t/150)*20
        # action[9]= 1#+math.cos(t/150)*20

        observation, reward, done, info = env.step(action)
        
        # qpos0_hist = np.append(qpos0_hist,reward)
        qpos0_hist = np.vstack((qpos0_hist,observation))
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

plt.plot(qpos0_hist[:,12])
plt.show()
# ipdb.set_trace()