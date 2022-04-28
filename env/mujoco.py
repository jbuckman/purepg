import random

import numpy as np

try:
    import gym
except:
    print("Mujoco not loaded.")

from env.base import BaseEnv

class Hopper(BaseEnv):
    state_shape = (11+4,)
    tokens = 256
    state_continuous = True
    # action_options = [-.99, -.75, -.50, -.25, .00, .25, .50, .75, .99]
    action_options = [-.5, 0., .5]
    action_options_count = len(action_options)
    action_count = action_options_count ** 3
    discount = 1.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        env = gym.make("Hopper-v3")
        self.env = env
        self.steps = 0
        self.action_set = list(range(self.action_count))
        self._legal_actions = [True]*self.action_count
        self._legal_action_set = self.action_set
        self.reset()

    def reset(self):
        self.steps = 0
        self._recent_obs = np.zeros(self.state_shape)
        self._obs = self.env.reset()
        self._obs = self._preprocess_obs(self._obs)
        self.terminated = False

    def _preprocess_obs(self, obs):
        return np.concatenate([obs, np.array([self.steps / (10**i) % 10 for i in range(4)])], 0) / 10

    def state_rep(self):
        return self._obs

    def obs_rep(self):
        return self._obs

    @property
    def legal_actions(self):
        return self._legal_actions

    @property
    def legal_action_set(self):
        return self._legal_action_set

    def step(self, action):
        if self.terminated:
            raise Exception(f"Attempted action {action} on terminated game.")
        action = [self.action_options[(action // self.action_options_count ** 2) % self.action_options_count],
                  self.action_options[(action // self.action_options_count ** 1) % self.action_options_count],
                  self.action_options[(action // self.action_options_count ** 0) % self.action_options_count], ]
        self._obs, reward, self.terminated, _ = self.env.step(action)
        self.steps += 1
        self._obs = self._preprocess_obs(self._obs)
        return reward

if __name__ == '__main__':
    import time
    _t = time.time()
    steps = 0
    for _ in range(20):
        game = Hopper()
        while not game.terminated:
            state = game.state_rep()
            r = game.step(random.choice(game.legal_action_set))
        steps += game.steps

    timing = time.time() - _t

    print(timing, steps)
    print(steps/timing)
