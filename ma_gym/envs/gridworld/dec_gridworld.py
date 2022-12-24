from ma_gym.envs.gridworld.gridworld import GridWorld
from ma_gym.wrapper import Wrapper
import gym
import numpy as np
from scipy.stats import entropy
from collections import defaultdict, OrderedDict

def Dec_GridWorld(config):
    return DecGridWorldWrapper(GridWorld, config)

def DecBF_GridWorld(config):
    return DecBFGridWorldWrapper(GridWorld, config)

class DecGridWorldWrapper(Wrapper):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.sensing_range = config.get("sensing_range",1)
        self.observation_space = gym.spaces.Dict({src: 
                                                  gym.spaces.Dict({dest: gym.spaces.Box(np.array([-1,
                                                                                                  -self.num_spaces,
                                                                                                  -self.num_spaces]),
                                                                                        np.array([self.num_features,
                                                                                                 self.num_spaces,
                                                                                                 self.num_spaces]),
                                                                                        (3,),
                                                                                        dtype=int)
                                                                   for dest in range(self.num_agents)})
                                                  for src in range(self.num_agents)})
    def _get_observation(self, src, dest):
        if self.within(src,dest):
            dist = self.get_dist(self.latent_state_space[src],self.latent_state_space[dest])
            if self.check_group(src,dest):
                return np.array([self.measurement_space[self.latent_state_space[dest]], *dist], dtype=int)
            else:
                return np.array([-1, *dist],dtype=int)
        return np.array([-1,0,0])
    def get_observation(self):
        return OrderedDict({src: OrderedDict({dest: self._get_observation(src,dest)
                                  for dest in range(self.num_agents)})
                            for src in range(self.num_agents)})
    def within(self, src, dest):
        src = self.latent_state_space[src]
        dest = self.latent_state_space[dest]
        return self.chebyshev(*self.state2coord(src),*self.state2coord(dest)) <= self.sensing_range
    def within_sensing_range(self, src, dest, sensing_range=1):
        src = self.latent_state_space[src]
        dest = self.latent_state_space[dest]
        return self.chebyshev(*self.state2coord(src),*self.state2coord(dest)) <= sensing_range
    def check_group(self, src, dest):
        group = self.groups[src]
        return dest in self.igroups[group]
    def get_dist(self, src, dest):
        return np.subtract(self.state2coord(src),self.state2coord(dest))
    def check_dist(self, src, dest, z):
        return np.equal(self.get_dist(src,dest),z).all()
    def mask(self, src, dest):
        return not (self.check_group(src,dest) and self.within(src,dest))
    def chebyshev(self,x0,y0,x1,y1):
        return max(abs(x0-x1),abs(y0-y1))
    def render(self, agent, h = 128, w = 128, show_all = False):
        if show_all:
            exclude = []
        else:
            exclude = [i for i in range(self.num_agents) if not self.within(agent, i)]
        return super().render(h, w, exclude)

class DecBFGridWorldWrapper(DecGridWorldWrapper):
    def __init__(self, env, config):
        super().__init__(env, config)  
        self.num_messagePassing_rounds = config.get('num_messagePassing_rounds',1)
        self.observation_space = gym.spaces.Dict({src: 
                                                  gym.spaces.Dict({dest: gym.spaces.Box(np.ones(self.num_spaces)*-1,
                                                                                        np.ones(self.num_spaces),
                                                                                        (self.num_spaces,),
                                                                                        dtype=np.float64)
                                                                   for dest in range(self.num_agents)})
                                                  for src in range(self.num_agents)})
    
    def reset(self):
        self.bel = np.ones((self.num_agents,self.num_agents, self.num_spaces))/self.num_spaces
        return super().reset()
    
    def step(self, action):
        self.control_update(action)
        _, reward, done, info = super().step(action)
        self.measurement_update()
        self.belief_sharing()
        return self.get_observation(), reward, done, info

    def get_observation(self):
        return {src: {dest: self.bel[src][dest]
                      for dest in range(self.num_agents)}
                for src in range(self.num_agents)}
    
    def get_sensing_emission_prob(self, z):
        out = np.array([[self.check_dist(src, dest, z) * 1. 
                        for src in np.arange(0,self.num_spaces)]
                       for dest in np.arange(0,self.num_spaces)], dtype=np.float64)
        return out
    
    def control_update(self, action):
        Tu = np.array([[self.get_transition_matrix(src, True)[u if not self.mask(src,dest) else -1].transpose(1,0)
                        for src, u in zip(range(self.num_agents), action)]
                       for dest in range(self.num_agents)])
        self.bel_hat = np.einsum("bijk,bik->bij", Tu, self.bel)
        
    def measurement_update(self):
        for src in range(self.num_agents):
            for dest in range(self.num_agents):
                if self.within(src,dest):
                    z = self.measurement_space[self.latent_state_space[dest]] if self.check_group(src,dest) else -1
                    Pz = self.get_emission_prob(z) + 1e-8
                    
                    s = self.get_dist(self.latent_state_space[src], self.latent_state_space[dest])
                    Ps = self.get_sensing_emission_prob(s)
                    Ps = np.matmul(Ps.T, self.bel_hat[src][src]) + 1e-8
                    P = Ps * Pz
                else:
                    P = self.get_emission_prob(-1)
                self.bel[src][dest] = self.bel_hat[src][dest] * P
                self.bel[src][dest] /= sum(self.bel[src][dest])+1e-8
                
    def belief_sharing(self):
        neighbors = {src:[dest for dest in range(self.num_agents) if not self.mask(src,dest)]
                         for src in range(self.num_agents)}
        for _ in range(self.num_messagePassing_rounds):
            bel_c = self.bel.copy()
            for src, dests in neighbors.items():
                bel_dest = np.array([self.bel[dest].copy() for dest in dests])
                min_H = np.argmin(entropy(bel_dest,axis=-1),axis=0)
                bel_c[src] = bel_dest[min_H,np.arange(self.num_agents)]
            for src in range(self.num_agents):
                self.bel[src] = bel_c[src]