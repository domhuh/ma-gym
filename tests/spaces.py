import gym

class MultiDiscreteSet(gym.spaces.MultiDiscrete):
    def __init__(self, _range, size=1):
        super().__init__([len(_range) for i in range(size)])
        self._range = _range
    def sample(self):
        out = super().sample()
        return self._range[out]