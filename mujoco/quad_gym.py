import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym import spaces
import ipdb

# class LaikagoEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    # def __init__(self):
    #     mujoco_env.MujocoEnv.__init__(self, 'laikago.xml', 2)
    #     utils.EzPickle.__init__(self)

    # def step(self, action):
    #     xposbefore = self.sim.data.qpos[0]
    #     self.do_simulation(action*5, self.frame_skip)
    #     xposafter = self.sim.data.qpos[0]
    #     ob = self._get_obs()
    #     reward_ctrl =  100.0 * np.square(action).sum()
    #     reward_run = 0*(xposafter - xposbefore)/self.dt
    #     #print(ob)
    #     reward = reward_ctrl + reward_run
    #     done = False
    #     return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    # def _get_obs(self):
    #     return np.concatenate([
    #         self.sim.data.qpos.flat[1:],
    #         self.sim.data.qvel.flat,
    #     ])

    # def reset(self):
    #     qpos = self.init_qpos# + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
    #     qvel = self.init_qvel #+ self.np_random.randn(self.model.nv) * .1
    #     return np.concatenate([
    #         self.sim.data.qpos.flat[1:],
    #         self.sim.data.qvel.flat,
    #     ])


    # def reset_model(self):
    #     qpos = self.init_qpos# + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
    #     qvel = self.init_qvel #+ self.np_random.randn(self.model.nv) * .1
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()

    # def viewer_setup(self):
    #     self.viewer.cam.distance = self.model.stat.extent * 0.5


class LaikagoEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'laikago.xml', 5)
        utils.EzPickle.__init__(self)
        # self.action_space = spaces.Box(low=np.ones(12)*-1, high=np.ones(12)*1, dtype=np.float16)
        low_array = np.array([-15.1, -.1, 25, 14.9, -0.1, 25, -10.1, -0.1, 22, 9.9, -0.1, 22])
        high_array = np.array([-14.9, .1, 45, 15.1, 0.1, 45, -9.9, 0.1, 38, 10.1, 0.1, 38])
        # self.action_space = spaces.Box(low=np.ones(12)*-1, high=np.ones(12))

        # self.action_space = spaces.Box(low=np.ones(8)*-1, high=np.ones(8)*1, dtype=np.float16)
        self.action_space = spaces.Box(low=np.ones(2)*-1, high=np.ones(2)*1, dtype=np.float16)

        # low_obs = np.concatenate([np.ones(15)*-15,np.ones(14)*-15, np.ones(8)*-4, np.ones(8)*-10, np.ones(3)*-5, np.ones(3)*-10, np.ones(3)*-10, np.ones(36)*-1000000, np.ones(4)*-1])
        # high_obs = np.concatenate([np.ones(15)*15,np.ones(14)*15, np.ones(8)*4, np.ones(8)*10, np.ones(3)*5, np.ones(3)*10, np.ones(3)*10, np.ones(36)*1000000, np.ones(4)*1] )
        # low_obs = np.ones(82)*-2
        # high_obs = np.ones(82)*2
        low_obs = np.ones(49)*-10
        high_obs = np.ones(49)*10
        # low_obs = np.concatenate([np.ones(15)*-15, np.ones(8)*-4, np.ones(8)*-10, np.ones(3)*-5, np.ones(3)*-10, np.ones(3)*-10, np.ones(24)*-1000, np.ones(4)*-1])
        # high_obs = np.concatenate([np.ones(15)*15, np.ones(8)*4, np.ones(8)*10, np.ones(3)*5, np.ones(3)*10, np.ones(3)*10, np.ones(24)*1000, np.ones(4)*-1])

        # self.observation_space = spaces.Box(low=-1e4*np.ones(81), high=1e4*np.ones(81), dtype=np.float16)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs)


        self.reward_range = (0, 1000)

    
    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        # low_array = np.array([-25.05, -.1, 25, 24.95, -0.1, 25, -12.05, -0.1, 22, 11.95, -0.1, 22])
        # high_array = np.array([-24.95, .1, 45, 25.05, 0.1, 45, -11.95, 0.1, 38, 12.05, 0.1, 38])

        low_array = np.array([-.1, 25, -0.1, 25, -0.1, 22, -0.1, 22])
        high_array = np.array([.1, 45, 0.1, 45, 0.1, 38, 0.1, 38])

        # low_array = -2*np.array([.1, 45, 0.1, 45, 0.1, 38, 0.1, 38]) + 2*np.array([-.1, 25, -0.1, 25, -0.1, 22, -0.1, 22])
        # high_array = 2*np.array([.1, 45, 0.1, 45, 0.1, 38, 0.1, 38]) - 2*np.array([-.1, 25, -0.1, 25, -0.1, 22, -0.1, 22])

        reward_ctrl =  -1*1e-3  * np.square(action).sum()
        # action = np.multiply(low_array,(-action+1*np.ones(12)))/2 + np.multiply(high_array,(action+1*np.ones(12)))/2
        

        action2 = np.array([0, action[0], 0, action[0], 0, action[1], 0, action[1]])

        action=action2*2
        action = np.multiply(low_array,(-action+1*np.ones(8)))/2 + np.multiply(high_array,(action+1*np.ones(8)))/2
        
        
        
        # print(action)
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_run = 1*1*(xposafter - xposbefore)/self.dt
        reward_dist = 0*10*(self.sim.data.qpos[0] - self.init_qpos[0])
        # reward_ht = -1*1000*(self.sim.data.qpos[2] + 0.08244029956850837)**2
        reward_ht = 0/(10+1000*(self.sim.data.qpos[2] + 0.0894850981117138)**2)
        reward_eplength = 0.05 + 0*self.sim.data.time
        # print("qpos")
        # print(self.sim.data.qpos)
        # print("eptime")
        # print(self.sim.data.time)
        #print(ob)
        # print(action)
        # ipdb.set_trace()
        reward = reward_ctrl + reward_run + reward_dist + reward_ht + reward_eplength
        done = False
        # if self.sim.data.qpos[2]<-0.27 or self.sim.data.qpos[2]> 0.001:
        #     done=True
        #     reward = -100
        
        #frame quaternion gives pitch and roll penalty    
        if np.abs(self.sim.data.sensordata[16])<0.94:
            done=True
            reward=reward-10
            # reward = -100
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl, reward_dist=reward_dist, z=self.sim.data.qpos[2])

    def reset(self):
        qpos = self.init_qpos# + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel #+ self.np_random.randn(self.model.nv) * .1
        # qacc = self.init_qacc*0
        self.set_state(qpos, qvel)
        # return np.concatenate([
        #     self.sim.data.qpos.flat[1:],
        #     self.sim.data.qvel.flat,
        # ])
        obs=np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat, self.sim.data.sensordata])
        qposlimit=15; qvellimit=4; anglelimit = 3.14; omegalimit=10; vellimit=10; acclimit=20; torqlimit=100; flimit=100000; 
        obs[0:14]=obs[0:14]#/qposlimit;
        obs[15:28]=obs[15:28]#/qvellimit;
        obs[29:36]=obs[29:36]/anglelimit;
        obs[37:44]=obs[37:44]/omegalimit;
        # obs[45:47]=obs[45:47]/vellimit;
        # obs[48:50]=obs[48:50]/acclimit;
        # obs[51:53]=obs[51:53]/omegalimit;
        # obs[54:]=obs[54:]/torqlimit;
        # obs[54:89]=obs[54:89]/torqlimit;
        return obs#np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat])
    
    def _get_obs(self):
        # return np.concatenate([
        #     self.sim.data.qpos.flat[1:],
        #     self.sim.data.qvel.flat,
        # ])

        obs=np.concatenate([self.sim.data.qpos.flat,self.sim.data.qvel.flat, self.sim.data.sensordata])
        qposlimit=15; qvellimit=4; anglelimit = 6.28; omegalimit=10; vellimit=10; acclimit=20; torqlimit=100; flimit=100000; 
        obs[0:14]=obs[0:14]/qposlimit;
        obs[15:28]=obs[15:28]/qvellimit;
        obs[29:36]=obs[29:36]/anglelimit;
        obs[37:44]=obs[37:44]/omegalimit;
        # obs[45:47]=obs[45:47]/vellimit;
        # obs[48:50]=obs[48:50]/acclimit;
        # obs[51:53]=obs[51:53]/omegalimit;
        # obs[54:]=obs[54:]/torqlimit;
        return obs#np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat])

    def viewer_setup(self):
            self.viewer.cam.distance = self.model.stat.extent * 0.5
    
    def close(self):
        pass