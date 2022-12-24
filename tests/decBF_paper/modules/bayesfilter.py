from collections import defaultdict
import numpy as np

class MABayesFilter():
    def __init__(self, env):
        self.env = env
        self.n_agents = self.env.n_agents
        self.Pt = self.env.Pt
        self.Pz = self.env.Pz
        self.bel = np.ones((self.n_agents,self.n_agents, self.env.N))/self.env.N

    def control_update(self, u):
        self.Pu = np.array([[self.Pt[_u] for _u in u] for src in range(self.n_agents)])
        self.bel_hat = np.einsum("bijk,bik->bik",self.Pu,self.bel)
        
    def measurement_update(self, z):
        self.bel = self.bel_hat * np.array([self.Pz(_z) for _z in z])
        
class DecMABayesFilter():
    def __init__(self, env):
        self.env = env
        self.n_agents = self.env.n_agents
        self.Pt = self.env.Pt
        self.Pz = self.env.Pz
        self.bel = np.ones((self.n_agents,self.n_agents, self.env.N))/self.env.N

    def mask_control(self, u):
        return np.array([[self.Pt[_u if self.env.within(src,dest) else -1].transpose(1,0)
                          for dest,_u in zip(self.env.locs,u)]
                         for src in self.env.locs])
    
    def mask_measurement(self, z):
        return np.array([self.Pz(_z) for _z in z])
    
    def control_update(self, u):
        self.Pu = self.mask_control(u)
        self.bel_hat = np.einsum("bijk,bik->bij",self.Pu,self.bel)
        
    def measurement_update(self, z):
        for i, (src, bel_hat) in enumerate(zip(self.env.locs, self.bel_hat)):
            for j, (dest, bel_hat_i) in enumerate(zip(self.env.locs, bel_hat)):
                self.bel[i][j] = bel_hat_i * self.Pz(z[j] if self.env.within(src,dest) else -1)
                self.bel[i][j] /= sum(self.bel[i][j])
                
    def belief_sharing(self):
        from scipy.stats import entropy
        connected = [[j for j,dest in enumerate(self.env.locs) if self.env.within(src,dest)]
                     for i,src in enumerate(self.env.locs)]
        for i, agents in enumerate(connected):
            bel = np.array([self.bel[j].copy() for j in agents])
            ret = np.argmin(entropy(bel,axis=-1),axis=0)
            self.bel[i] = bel[ret,np.arange(self.n_agents)]