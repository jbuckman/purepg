import random

import numpy as np

try:
    import gym
except:
    print("gym not loaded.")

from env.base import BaseEnv

class CartPole(BaseEnv):
    state_shape = (4,)
    tokens = 256
    state_continuous = True
    action_count = 2
    discount = 1.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        env = gym.make("CartPole-v1")
        self.env = env
        self.steps = 0
        self.action_set = list(range(self.action_count))
        self._legal_actions = [True]*self.action_count
        self._legal_action_set = self.action_set
        self.reset()

    def reset(self):
        self.steps = 0
        self._obs = self.env.reset()
        self.terminated = False

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
        self._obs, reward, self.terminated, _ = self.env.step(action)
        self.steps += 1
        return reward

if __name__ == '__main__':
    import time
    _t = time.time()
    steps = 0
    for _ in range(20):
        game = CartPole()
        while not game.terminated:
            state = game.state_rep()
            r = game.step(random.choice(game.legal_action_set))
        steps += game.steps

    timing = time.time() - _t

    print(timing, steps)
    print(steps/timing)
