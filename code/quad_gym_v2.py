import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym import spaces
import ipdb

class LaikagoEnv2(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'laikago2.xml', 2)
        utils.EzPickle.__init__(self)

        self.action_space = spaces.Box(low=np.ones(12)*-1, high=np.ones(12)*1, dtype=np.float16)

        # low_obs = np.ones(109)*-100
        # high_obs = np.ones(109)*100

        low_obs = np.ones(137)*-100
        high_obs = np.ones(137)*100        

        self.observation_space = spaces.Box(low=low_obs, high=high_obs)


        self.reward_range = (0, 2000)

    
    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
    
        reward_ctrl =  -1*5e-3  * np.square(action).sum()
        

        action2 = np.array([action[0], -8+2*action[1], 4.5*action[2]+2, action[3], -8+2*action[4], 4.5*action[5]+2, action[6], 8-2*action[7], 4.5*action[8]+2, action[9], 8-2*action[10], 4.5*action[11]+2])


        action=action2
        
        self.do_simulation(action[:], self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_run = (xposafter - xposbefore)/self.dt

        
        reward_dist = 0*10*(self.sim.data.qpos[0] - self.init_qpos[0])
        reward_ht = -1*0*((self.sim.data.qpos[2] + 0.125)/0.125)**2

        reward_eplength = 0.2*2.5 + 0*self.sim.data.time

        # if(self.sim.data.sensordata[71]>0.95):
        #     reward_pitch = 0.5*(self.sim.data.sensordata[63]) #rewarding for staying upright
        # else:
        #     reward_pitch = -.5


        reward_pitch = 0.5*.1*10*((self.sim.data.sensordata[83])**16)

        if(self.sim.data.sensordata[63]>0.96):
            reward_yaw = 0.5*.1*10*((self.sim.data.sensordata[91])**22) #rewarding for heading direction matching x axis (0 yaw)
        else:
            reward_yaw = -.5

        reward = reward_ctrl + reward_run + reward_dist + reward_ht + reward_eplength + reward_pitch + reward_yaw
        done = False
        # if self.sim.data.qpos[2]<-0.27 or self.sim.data.qpos[2]> 0.001:
        #     done=True
        #     reward = -100
        
        #frame quaternion gives pitch and roll penalty    
        
        # if np.abs(self.sim.data.sensordata[67])<0.94:
        #     done=True
        #     reward=reward-10
        #     # reward = -100

        if np.abs(self.sim.data.sensordata[83])<0.95: #using Z axis projection. To use X axis quaternion, use 87 element
            done=True
            reward=reward-10
            # reward = -100    
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl, reward_dist=reward_dist, z=self.sim.data.qpos[2])

    def reset(self):
        qpos = self.init_qpos# + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq) #randomized start state
        qvel = self.init_qvel #+ self.np_random.randn(self.model.nv) * .1 #randomized start state
        self.set_state(qpos, qvel)

        obs = self.sim.data.sensordata
        qposlimit=15; qvellimit=4; anglelimit = 3.14; omegalimit=10; vellimit=10; acclimit=20; torqlimit=100; flimit=100000; 

        return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat, obs])
    
    def _get_obs(self):
        # return np.concatenate([
        #     self.sim.data.qpos.flat[1:],
        #     self.sim.data.qvel.flat,
        # ])

        obs = self.sim.data.sensordata
        qposlimit=15; qvellimit=4; anglelimit = 6.28; omegalimit=10; vellimit=10; acclimit=20; torqlimit=100; flimit=100000; 

        return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat, obs])

    def viewer_setup(self):
            self.viewer.cam.distance = self.model.stat.extent * 0.5
    
    def close(self):
        pass