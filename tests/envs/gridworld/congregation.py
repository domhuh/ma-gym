from ma_gym.envs.gridworld.dec_gridworld import Dec_GridWorld, DecBF_GridWorld
import gym
from ma_gym.utils.gridworld_render import *
from matplotlib import cm
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import copy
import random

class Congregation(gym.Env):
    def __init__(self, config, use_bf = True):
        super().__init__()        
        self.sim = DecBF_GridWorld(config) if use_bf else Dec_GridWorld(config)
        for k,v in self.sim.__dict__.items():
            self.__dict__[k] = v
        self.horizon = 30
    def reset(self):
        self.t = 0
        return self.sim.reset()
    def calculate_reward(self):
        reward = 0
        for src in range(self.num_agents):
            for dest in range(self.num_agents):
                if src!=dest:
                    reward += self.sim.within_sensing_range(src, dest, 1) * 1.
        return reward
    def step(self, action):
        self.t+=1
        state, reward, done, info = self.sim.step(action)
        reward = self.calculate_reward()
        done = self.t == self.horizon
        return state, reward, done, info
    def render(self, agent, h = 128, w = 128, show_all = False):
        return self.sim.render(agent, h, w, show_all)
    
#----- POLICIES -----#

import numpy as np
from collections import defaultdict
from functools import partial

def cg_policy(env, policy):
    actions = np.zeros(env.num_agents)
    for agent, action in policy().items():
        actions[agent] = action
    return actions

def random_policy(sim, group = 0):
    def thunk():
        return {agent: np.random.choice(sim.control_space) for agent in sim.igroups[group]}
    return thunk

def global_policy(sim, group = 0, theta = 0.5):
    from gridworld import CHEBYSHEV_MOVEMENT
    action_space = [CHEBYSHEV_MOVEMENT[k] for k in range(9)]    
    def thunk():
        group_avgloc = np.mean([sim.state2coord(sim.latent_state_space[agent])
                                for agent in sim.igroups[group]],0)
        target = {}
        for agent in sim.igroups[group]:
            group_dir = np.subtract(group_avgloc, sim.state2coord(sim.latent_state_space[agent]))
            _dir = np.ceil(group_dir).astype(int).clip(-1,1)
            target[agent] = np.argwhere((action_space == _dir).all(-1)).item()
        return target
    return thunk

def local_policy(sim, group = 0, theta = 0.5):
    from gridworld import CHEBYSHEV_MOVEMENT
    action_space = [CHEBYSHEV_MOVEMENT[k] for k in range(9)]
    def thunk():
        locs = np.argwhere(np.isclose(sim.bel, sim.bel.max(-1)[:,:,np.newaxis]))
        ret = defaultdict(partial(defaultdict, list))
        for src, dest, i in locs:
            ret[src][dest].append(i)
        bel = defaultdict(list)
        for src, _bel in ret.items():
            for dest in range(sim.num_agents):
                bel[src].append(sim.state2coord(np.random.choice(_bel[dest])))
        lis = []
        for agent in sim.igroups[group]:
            lis.extend(bel[agent])
        group_avgloc = np.mean(lis,0)
        target = {}
        for agent in sim.igroups[group]:
            group_dir = np.subtract(group_avgloc,sim.state2coord(sim.latent_state_space[agent]))
            _dir = np.ceil(group_dir).astype(int).clip(-1,1)
            target[agent] = np.argwhere((action_space == _dir).all(-1)).item()
        return target
    return thunk