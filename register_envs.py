import gym
import pybulletgym
from gym.envs.registration import register
import numpy as np

class TimeAwareWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._env = env
        self.observation_space = gym.spaces.Box(low=np.concatenate((env.observation_space.low, [0])),
                                                high=np.concatenate((env.observation_space.high, [1])))
    def observation(self, obs):
        return np.concatenate((obs, [self.env._elapsed_steps / self.env._max_episode_steps]))

class DiscretizationWrapper(gym.ActionWrapper):
    def __init__(self, env, n=2):
        super().__init__(env)
        self.discretization = n
        self.action_dims = env.action_space.shape[0]
        self.action_space = gym.spaces.Discrete(self.discretization ** self.action_dims)

    def action(self, act):
        components = np.array([act // self.discretization ** i % self.discretization for i in range(self.action_dims)])
        return components / (self.discretization - 1) * 2 - 1

def create_disc_env(env_id: str, n : int):
    def make_env():
        env = gym.make(env_id)
        env = TimeAwareWrapper(env)
        env = DiscretizationWrapper(env, n)
        return env
    return make_env

register(
    id="CartPoleTimed-v1",
    entry_point=lambda : TimeAwareWrapper(gym.make("CartPole-v1"))
)

for env_id in ["HalfCheetahPyBulletEnv-v0", "AntPyBulletEnv-v0", "Walker2DPyBulletEnv-v0", "HopperPyBulletEnv-v0"]:
    for disc_level in [2,3,4,5]:
        name, version = env_id.split("-v")
        register(
            id=f"{name}Disc{disc_level}-v{version}",
            entry_point=create_disc_env(env_id, disc_level),
        )