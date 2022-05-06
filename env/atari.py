import random
try:
    from ale_py import ALEInterface, LoggerMode
    ALEInterface.setLoggerMode(LoggerMode.Warning)
    import ale_py.roms
except:
    print("Atari not loaded.")
try:
    import cv2
except:
    print("Atari not loaded because cv2 is missing.")
import numpy as np

from env import BaseEnv

class classproperty(object):
    def __init__(self, f):
        self.f = f
    def __get__(self, obj, owner):
        return self.f(owner)

_ATARI_DATA = {'Alien': (227.8, 7127.7), 'Amidar': (5.8, 1719.5), 'Assault': (222.4, 742.0), 'Asterix': (210.0, 8503.3), 'Asteroids': (719.1, 47388.7), 'Atlantis': (12850.0, 29028.1), 'BankHeist': (14.2, 753.1), 'BattleZone': (2360.0, 37187.5), 'BeamRider': (363.9, 16926.5), 'Berzerk': (123.7, 2630.4), 'Bowling': (23.1, 160.7), 'Boxing': (0.1, 12.1), 'Breakout': (1.7, 30.5), 'Centipede': (2090.9, 12017.0), 'ChopperCommand': (811.0, 7387.8), 'CrazyClimber': (10780.5, 35829.4), 'Defender': (2874.5, 18688.9), 'DemonAttack': (152.1, 1971.0), 'DoubleDunk': (-18.6, -16.4), 'Enduro': (0.0, 860.5), 'FishingDerby': (-91.7, -38.7), 'Freeway': (0.0, 29.6), 'Frostbite': (65.2, 4334.7), 'Gopher': (257.6, 2412.5), 'Gravitar': (173.0, 3351.4), 'Hero': (1027.0, 30826.4), 'IceHockey': (-11.2, 0.9), 'Jamesbond': (29.0, 302.8), 'Kangaroo': (52.0, 3035.0), 'Krull': (1598.0, 2665.5), 'KungFuMaster': (258.5, 22736.3), 'MontezumaRevenge': (0.0, 4753.3), 'MsPacman': (307.3, 6951.6), 'NameThisGame': (2292.3, 8049.0), 'Phoenix': (761.4, 7242.6), 'Pitfall': (-229.4, 6463.7), 'Pong': (-20.7, 14.6), 'PrivateEye': (24.9, 69571.3), 'Qbert': (163.9, 13455.0), 'Riverraid': (1338.5, 17118.0), 'RoadRunner': (11.5, 7845.0), 'Robotank': (2.2, 11.9), 'Seaquest': (68.4, 42054.7), 'Skiing': (-17098.1, -4336.9), 'Solaris': (1236.3, 12326.7), 'SpaceInvaders': (148.0, 1668.7), 'StarGunner': (664.0, 10250.0), 'Surround': (-10.0, 6.5), 'Tennis': (-23.8, -8.3), 'TimePilot': (3568.0, 5229.2), 'Tutankham': (11.4, 167.6), 'UpNDown': (533.4, 11693.2), 'Venture': (0.0, 1187.5), 'VideoPinball': (16256.9, 17667.9), 'WizardOfWor': (563.5, 4756.5), 'YarsRevenge': (3092.9, 54576.9), 'Zaxxon': (32.5, 9173.3)}

class Atari(BaseEnv):
    state_shape = (4,84,84)
    tokens = 256
    state_continuous = True
    action_count = 18
    game_name = None
    seed = None

    discount = 1.
    random_termination_prob = 0.
    frame_skip = 4
    sticky_action_prob = 1/4
    max_steps = None
    terminate_on_first_reward = False
    terminate_on_life_loss = False
    initial_fire = True

    def render(self):
        return self.ale.getScreenRGB()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ale = ALEInterface()
        ale.setInt('random_seed', random.randint(0, 10000000) if self.seed is None else self.seed)
        ale.setFloat('repeat_action_probability', 0.)
        ale.setInt('max_num_frames_per_episode', 108_000)
        ale.setBool('display_screen', False)
        ale.loadROM(getattr(ale_py.roms, self.game_name))
        ale.reset_game()
        self.ale = ale
        self.steps = 0
        self.initial_lives = self.current_lives = ale.lives()
        self.action_set = ale.getLegalActionSet()
        self._legal_actions = [i in self.ale.getMinimalActionSet() for i in range(self.action_count)]
        self._legal_action_set = [i for i in range(self.action_count) if self._legal_actions[i]]
        self.terminated = False
        self.stuck_action = None
        if self.initial_fire and self.legal_actions[1]: ale.act(1)
        csgs = self.ale.getScreenGrayscale()
        self.current_obs_pair = [np.zeros_like(csgs), csgs]
        obs = self.preproc_obs(self.current_obs_pair)
        self.recent_obs_buffer = (self.state_shape[0]-1) * [np.zeros_like(obs)] + [obs]

    @classproperty
    def random_score(self) -> float: return _ATARI_DATA[self.game_name][0]
    @classproperty
    def human_score(self) -> float: return _ATARI_DATA[self.game_name][1]

    def reset(self):
        self.steps = 0
        self.ale.reset_game()
        self.initial_lives = self.current_lives = self.ale.lives()
        self.terminated = False
        self.stuck_action = None
        if self.initial_fire and self.legal_actions[1]: self.ale.act(1)
        csgs = self.ale.getScreenGrayscale()
        self.current_obs_pair = [np.zeros_like(csgs), csgs]
        obs = self.preproc_obs(self.current_obs_pair)
        self.recent_obs_buffer = (self.state_shape[0]-1) * [np.zeros_like(obs)] + [obs]

    def preproc_obs(self, obs_pair):
        obs = np.maximum(obs_pair[0], obs_pair[1])
        obs = cv2.resize(obs, (self.state_shape[-1], self.state_shape[-2]))
        return obs

    def state_rep(self):
        return np.cast[np.uint8](self.recent_obs_buffer)

    def obs_rep(self):
        return np.cast[np.uint8](self.recent_obs_buffer[-1])

    @property
    def legal_actions(self):
        return self._legal_actions

    @property
    def legal_action_set(self):
        return self._legal_action_set

    def step(self, action):
        if self.terminated:
            raise Exception(f"Attempted action {action} on terminated game.")
        reward = 0
        for i in range(self.frame_skip):
            if self.stuck_action is None or random.random() > self.sticky_action_prob: self.stuck_action = action
            reward += self.ale.act(self.action_set[self.stuck_action])
            if i == self.frame_skip - 2: self.current_obs_pair[0] = self.ale.getScreenGrayscale()
            if i == self.frame_skip - 1: self.current_obs_pair[1] = self.ale.getScreenGrayscale()
        if self.ale.lives() < self.current_lives:
            if self.initial_fire and self.legal_actions[1]: reward += self.ale.act(1)
            self.current_lives = self.ale.lives()
        obs = self.preproc_obs(self.current_obs_pair)
        self.recent_obs_buffer.pop(0)
        self.recent_obs_buffer.append(obs)
        self.steps += 1
        self.terminated = self.ale.game_over() or \
                          (self.max_steps is not None and self.steps == self.max_steps) or \
                          (random.random() < self.random_termination_prob) or \
                          (self.terminate_on_life_loss and self.ale.lives() < self.initial_lives) or \
                          (self.terminate_on_first_reward and reward != 0)
        return reward

class AtariPong(Atari): game_name = "Pong"

if __name__ == '__main__':
    import time
    _t = time.time()
    steps = 0
    for _ in range(20):
        game = AtariPong()
        options = [a for a in range(18) if game.legal_actions[a]]
        while not game.terminated:
            obs = game.ale.getScreenGrayscale()
            r = game.step(random.choice(options))
        steps += game.steps

    timing = time.time() - _t

    print(timing, steps)
    print(steps/timing)
