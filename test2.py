from ipdb import set_trace
import envpool
import numpy as np

if __name__ == '__main__':
    EPISODES_PER_GRADIENT = 1
    BATCH_SIZE = 1
    LR = .0001

    ## Initialize environments
    env = envpool.make("CartPole-v1", env_type="gym", num_envs=EPISODES_PER_GRADIENT, batch_size=BATCH_SIZE)

    for round in range(10000000):
        print(f"==> Round {round: 4}.", end="\r")
        ## Reset all environments
        env.async_reset()

        ## Play exactly 100 episodes
        for i in range(100):
            state, rew, done, info = env.recv()
            env_id = info["env_id"]
            env.send(np.random.randint(env.action_space.n, size=BATCH_SIZE), env_id)
