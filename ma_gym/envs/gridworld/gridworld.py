from ma_gym.utils.gridworld_render import *
import gym
from matplotlib import cm
import numpy as np
from ma_gym.spaces import MultiDiscreteSet
from collections import defaultdict

def clip(v, _min, _max):
    return min(_max,max(_min,v))

CHEBYSHEV_MOVEMENT = {0 : [-1, -1],
                      1 : [-1, 0],
                      2 : [-1, 1],
                      3 : [0, -1],
                      4 : [0, 0],
                      5 : [0, 1],
                      6 : [1, -1],
                      7 : [1, 0],
                      8 : [1, 1]}

class GridWorld(gym.Env):
    def __init__(self, config = dict(num_rows=5,
                                     num_cols=5, 
                                     num_features=5, 
                                     num_agents=5,
                                     groups = {})):
        """
        GridWorld Base Env
        """
        self.num_rows = config.get("num_rows",5)
        self.num_cols = config.get("num_cols",5)
        self.num_spaces = self.num_rows * self.num_cols
        self.num_features = config.get("num_features",5)
        self.num_agents = config.get("num_agents",5)
        self.groups = config.get("groups", {agent:0 for agent in range(self.num_agents)})
        self.igroups = defaultdict(list)
        for agent, group in self.groups.items():
            self.igroups[group].append(agent)
            
        self.measurement_space = np.random.randint(0, self.num_features, size = self.num_spaces)
        self.latent_state_space = np.random.choice(np.arange(0, self.num_spaces), replace=False, size = self.num_agents)
    
        self.control_space = np.arange(0,9)
        self.action_space = MultiDiscreteSet(self.control_space, self.num_agents)
        self.observation_space = gym.spaces.Dict({src: 
                                                  gym.spaces.Dict({dest: gym.spaces.Box(0,self.num_features,(1,), dtype=np.uint8)
                                                                   for dest in range(self.num_agents)})
                                                  for src in range(self.num_agents)})
        
    def step(self, action, order=None):
        if order==None:
            order = np.arange(self.num_agents)
        for agent, u in zip(order, action):
            #T = self.get_transition_matrix(agent)
            #self.latent_state_space[agent] = np.random.choice(np.arange(self.num_spaces), p = T[u][_z])
            dr, dc = CHEBYSHEV_MOVEMENT[u]
            _z = self.latent_state_space[agent]
            _r, _c = self.state2coord(_z)
            c = clip(_c + dc, 0, self.num_cols-1)
            r = clip(_r + dr, 0, self.num_rows-1)
            z = self.coord2state(r,c)
            if z in np.delete(self.latent_state_space, agent):
                z = _z
            self.latent_state_space[agent] = z
        return self.get_observation(), 0, False, {}
    
    def reset(self, hard_reset=False):
        if hard_reset:
            self.measurement_space = np.random.randint(0, self.num_features, size = self.num_spaces)
        self.latent_state_space = np.random.choice(np.arange(0, self.num_spaces), replace=False, size = self.num_agents)
        return self.get_observation()
    
    def get_observation(self):
        return {src: {dest: self.measurement_space[self.latent_state_space[dest]]
                      for dest in range(self.num_agents)}
                for src in range(self.num_agents)}
    
    def get_state(self, use_coord = False):
        if use_coord:
            return {agent : self.state2coord(self.latent_state_space[agent]) for agent in range(self.num_agents)}
        return {agent : self.latent_state_space[agent] for agent in range(self.num_agents)}
    
    def state2coord(self, pos):
        row = pos // self.num_cols
        col = pos % self.num_cols
        return [row, col]
    
    def coord2state(self, row, col):
        return row*self.num_cols+col

    def get_transition_matrix(self, agent, mask=False):
        T = {}
        for i, u in enumerate(self.control_space):
            dr, dc = CHEBYSHEV_MOVEMENT[u]
            Pt = np.zeros(shape=(self.num_spaces,self.num_spaces))
            for _z in range(self.num_spaces):
                _r, _c = self.state2coord(_z)
                c = clip(_c + dc, 0, self.num_cols-1)
                r = clip(_r + dr, 0, self.num_rows-1)
                z = self.coord2state(r,c)
                if z in np.delete(self.latent_state_space, agent) and not mask:
                    z = _z
                if mask:
                    Pt[_z][_z] = 0.5
                    Pt[_z][z] = 0.5
                else:
                    Pt[_z][z] = 1.
            T[u] = Pt

        Pt = np.zeros(shape=(self.num_spaces,self.num_spaces))
        for v in T.values():
            Pt = np.logical_or(Pt, v)
        T[-1] = Pt/Pt.sum(-1)[:, np.newaxis]
        
        return T
    
    def get_emission_prob(self,z):
        if z == -1:
            out = np.ones(self.num_spaces)
        else:
            out = (self.measurement_space==z).astype(np.float64)
        val = sum(out)
        return out/val if val!=0 else out
    
    def render(self, h = 128, w = 128, exclude = []):
        colorize_features = lambda x: tuple(int(i*255) for i in cm.Greys((x+1)/(self.num_features+2))[:3])
        colorize_agents = lambda x: tuple(int(i*255) for i in cm.hsv((x)/(self.num_agents))[:3])
        grid, cell_size = draw_grid_fixed(self.num_rows, self.num_cols, (h,w))
        for pos, obs in enumerate(self.measurement_space):
            fill_cell(grid, self.state2coord(pos), cell_size, colorize_features(obs), margin=0.1)
            #draw_cell_outline(grid, self.state2coord(p), cell_size, width=1)
        for agent, pos in enumerate(self.latent_state_space):
            if agent not in exclude:
                draw_circle(grid, self.state2coord(pos), cell_size, colorize_agents(agent))
        return np.array(grid)       