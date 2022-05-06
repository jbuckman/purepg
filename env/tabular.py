import random

import numpy as np

from env.base import BaseEnv

class DontDiscountMe(BaseEnv):
    binary_steps = 3
    linear_steps = 20
    state_shape = (2**binary_steps + linear_steps + 1,)
    action_count = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.steps = 0
        self.action_set = list(range(self.action_count))
        self._legal_actions = [True]*self.action_count
        self._legal_action_set = self.action_set
        self._state_reps = np.identity(2**self.binary_steps + self.linear_steps + 1)
        # self._state_reps = np.random.uniform(0, 1, (2**self.binary_steps + self.linear_steps + 1, *self.state_shape))
        self.reset()

    def reset(self):
        self.steps = 0
        self._state = 0
        self.terminated = False

    def state_rep(self):
        return self._state_reps[self._state]

    def obs_rep(self):
        return self.state_rep()

    @property
    def legal_actions(self):
        return self._legal_actions

    @property
    def legal_action_set(self):
        return self._legal_action_set

    def step(self, action):
        if self.terminated:
            raise Exception(f"Attempted action {action} on terminated game.")

        if self._state == 0:
            reward = 0
            if action == 0:
                self._state = 2**self.binary_steps
            else:
                self._state = 1

        ## Linear path
        elif self._state >= 2**self.binary_steps:
            if self._state == 2**self.binary_steps + 20:
                reward = 5
                self.terminated = True
            else:
                reward = 0
                self._state += 1

        ## Binary path
        else:
            stepsize = 2**self.binary_steps // 2**self.steps
            if action == 0:
                reward = 0
                self._state += 1
            else:
                reward = 0
                self._state += stepsize
            if self.steps == self.binary_steps-1:
                reward = 1 if self._state == self.binary_steps else -.1
                self.terminated = True

        self.steps += 1
        return reward


class DontDiscountMe2(BaseEnv):
    state_shape = (4,)
    state_count = 10+3+10
    action_count = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.steps = 0
        self.action_set = list(range(self.action_count))
        self._legal_actions = [True]*self.action_count
        self._legal_action_set = self.action_set
        self._state_reps = np.random.uniform(0, 1, (self.state_count, *self.state_shape))
        self.reset()

    def reset(self):
        self.steps = 0
        self._state = 0
        self.terminated = False

    def state_rep(self):
        return self._state_reps[self._state]

    def obs_rep(self):
        return self.state_rep()

    @property
    def legal_actions(self):
        return self._legal_actions

    @property
    def legal_action_set(self):
        return self._legal_action_set

    def step(self, action):
        if self.terminated:
            raise Exception(f"Attempted action {action} on terminated game.")

        ## Hard part
        if self._state < 10:
            reward = action == self._state % 2
            self._state += 1

        ## Choose
        elif self._state == 10:
            if action == 0:
                reward = 0
                self._state = 11
            else:
                reward = 0
                self._state = 13

        ## Endgame
        elif self._state == 12:
            reward = random.choice([0, 2])
            self.terminated = True
        elif self._state == 22:
            reward = random.choice([0, 10])
            self.terminated = True

        ## Easy states
        else:
            reward = 0
            self._state += 1

        self.steps += 1
        return reward

if __name__ == '__main__':
    import time
    _t = time.time()
    steps = 0
    for _ in range(20):
        game = DontDiscountMe()
        while not game.terminated:
            state = game.state_rep()
            r = game.step(random.choice(game.legal_action_set))
        steps += game.steps

    timing = time.time() - _t

    print(timing, steps)
    print(steps/timing)
