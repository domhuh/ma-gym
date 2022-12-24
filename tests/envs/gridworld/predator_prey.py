from ma_gym.envs.gridworld.dec_gridworld import Dec_GridWorld, DecBF_GridWorld
import gym
from ma_gym.utils.gridworld_render import *
from matplotlib import cm
import numpy as np

class PredatorPrey(gym.Env):
    def __init__(self, config, use_bf = True):
        super().__init__()
        num_agents = config['num_predators'] + config['num_preys']
        groups = {i:0 if i < config['num_predators'] else 1 for i in range(num_agents)}        
        
        config['num_agents'] = num_agents
        config['groups'] = groups
        
        self.sim = DecBF_GridWorld(config) if use_bf else Dec_GridWorld(config)
        for k,v in self.sim.__dict__.items():
            self.__dict__[k] = v
        self.horizon = 30
    def reset(self):
        self.t = 0
        return self.sim.reset()
    def calculate_reward(self):
        reward = 0
        for predator in self.igroups[0]:
            for prey in self.igroups[1]:
                reward += self.sim.within_sensing_range(predator, prey, 1) * 1.
        return reward
    def step(self, action):
        self.t+=1
        state, reward, done, info = self.sim.step(action)
        reward = self.calculate_reward()
        done = self.t == self.horizon
        return state, reward, done, info
    def render(self, agent, h = 128, w = 128, show_all = False):
        if show_all:
            exclude = []
        else:
            exclude = [i for i in range(self.num_agents) if not self.sim.within(agent, i)]
        colorize_features = lambda x: tuple(int(i*255) for i in cm.Greys((x+1)/(self.num_features+2))[:3])
        colorize_agents = lambda x: tuple(int(i*255) for i in cm.hsv((self.groups[x])/len(self.groups))[:3])
        grid, cell_size = draw_grid_fixed(self.num_rows, self.num_cols, (h,w))
        for pos, obs in enumerate(self.sim.measurement_space):
            fill_cell(grid, self.sim.state2coord(pos), cell_size, colorize_features(obs), margin=0.1)
        for agent, pos in enumerate(self.sim.latent_state_space):
            if agent not in exclude:
                draw_circle(grid, self.sim.state2coord(pos), cell_size, colorize_agents(agent))
        return np.array(grid)       
    
    
    
#----- POLICIES -----#

import numpy as np
from collections import defaultdict
from functools import partial

def pp_policy(env, predator_policy, prey_policy):
    actions = np.zeros(env.num_agents)
    prey_actions = prey_policy()
    predator_actions = predator_policy()
    for agent, action in predator_actions.items():
        actions[agent] = action
    for agent, action in prey_actions.items():
        actions[agent] = action
    return actions

def random_policy(sim, group = 0):
    def thunk():
        return {agent: np.random.choice(sim.control_space) for agent in sim.igroups[group]}
    return thunk

def global_policy(sim, group = 0, theta = 0.5):
    from gridworld import CHEBYSHEV_MOVEMENT
    action_space = [CHEBYSHEV_MOVEMENT[k] for k in range(9)]
    
    not_group = [i for i in sim.igroups.keys() if i!=group]
    def thunk():
        group_avgloc = np.mean([sim.state2coord(sim.latent_state_space[agent])
                                for agent in sim.igroups[group]],0)
        lis = []
        for ng in not_group:
            for agent in sim.igroups[ng]:
                lis.append(sim.state2coord(sim.latent_state_space[agent]))
        notgroup_avgloc = np.mean(lis,0)
        
        target = {}
        for agent in sim.igroups[group]:
            group_dir = np.subtract(sim.state2coord(sim.latent_state_space[agent]), group_avgloc)
            notgroup_dir = np.subtract(notgroup_avgloc, sim.state2coord(sim.latent_state_space[agent]))
            _dir = np.ceil(theta * group_dir + (1 - theta) * notgroup_dir).astype(int).clip(-1,1)
            target[agent] = np.argwhere((action_space == _dir).all(-1)).item()
        return target
    return thunk

def local_policy(sim, group = 0, theta = 0.5):
    from gridworld import CHEBYSHEV_MOVEMENT
    action_space = [CHEBYSHEV_MOVEMENT[k] for k in range(9)]
    not_group = [i for i in sim.igroups.keys() if i!=group]
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
        
        lis = []
        for ng in not_group:
            for agent in sim.igroups[ng]:
                lis.extend(bel[agent])
                
        notgroup_avgloc = np.mean(lis,0)
        
        target = {}
        for agent in sim.igroups[group]:
            group_dir = np.subtract(sim.state2coord(sim.latent_state_space[agent]), group_avgloc)
            notgroup_dir = np.subtract(notgroup_avgloc, sim.state2coord(sim.latent_state_space[agent]))
            _dir = np.ceil(theta * group_dir + (1 - theta) * notgroup_dir).astype(int).clip(-1,1)
            target[agent] = np.argwhere((action_space == _dir).all(-1)).item()
        return target
    return thunk