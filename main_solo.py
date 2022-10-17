from ipdb import set_trace
from collections import defaultdict
from functools import partial
from itertools import count
import jax
from jax import random, numpy as jnp                # JAX NumPy
from flax import linen as nn           # The Linen API
import numpy as np                     # Ordinary NumPy
import optax                           # Optimizers
import gym
import register_envs

def front_broadcast(base, to):
    return base.reshape(base.shape[0], *[1]*(len(to.shape) - 1))

@partial(jax.jit, static_argnums=1)
@partial(jax.grad, has_aux=True)
def act_on(params, agent, state, rng):
    logits = agent.apply(params, state)
    action = jax.random.categorical(rng, logits, -1)
    log_probs = jax.nn.log_softmax(logits, -1)
    action_log_probs = -jnp.take(log_probs, action, -1)
    entropy = -jnp.sum(log_probs * jnp.exp(log_probs), -1)
    return action_log_probs, (action_log_probs, logits, action, entropy)

@jax.jit
def update_gradients(episode, running_gradients, grad):
    return jax.tree_map(lambda rg, g: rg.at[episode].set(rg[episode] + g), running_gradients, grad)

@jax.jit
def average_gradients(results, running_gradients):
    return jax.tree_map(lambda rg: (rg * front_broadcast(results, rg)).mean(0), running_gradients)

if __name__ == '__main__':
    EPISODES_PER_GRADIENT = 1000
    LR = .01
    OPT = optax.adam(LR)

    ## Initialize environments
    env = gym.make("HalfCheetahPyBulletEnvDisc2-v0")
    # env = gym.make("CartPole-v1")

    ## Model
    class MLP(nn.Module):
        @nn.compact
        def __call__(self, x):
            for feat in [256, 256, 256, 256]:
                x = nn.relu(nn.Dense(feat)(x))
            x = nn.Dense(env.action_space.n)(x)
            return x

    ## Initialize agent
    rng = jax.random.PRNGKey(0)
    agent = MLP()
    rng, _rng = random.split(rng, 2)
    params = agent.init(_rng, jnp.ones([*env.observation_space.shape]))
    tx = optax.chain(optax.adam(LR), optax.clip_by_global_norm(0.5))
    opt_state = tx.init(params)

    for step in count(1, 1):
        print(f"==> Step {step: 4}.", end="\r")

        ## Reset accumulators
        running_gradients = jax.tree_map(lambda x: jnp.zeros((EPISODES_PER_GRADIENT, *x.shape)), params)
        returns = np.zeros(EPISODES_PER_GRADIENT)
        lengths = np.zeros(EPISODES_PER_GRADIENT)
        entropies = np.zeros(EPISODES_PER_GRADIENT)
        ## Begin interacting
        for episode in range(EPISODES_PER_GRADIENT):
            state, done = env.reset(), False
            while not done:
                rng, _rng = random.split(rng, 2)
                grad, (loss, logits, action, entropy) = act_on(params, agent, jnp.array(state), _rng)
                state, rew, done, _ = env.step(np.array(action))
                lengths[episode] += 1
                returns[episode] += rew
                entropies[episode] += entropy
                running_gradients = update_gradients(episode, running_gradients, grad)
            print(f"==> ({episode / EPISODES_PER_GRADIENT:.1%}) Step {step: 4} | Score: {returns[:episode+1].mean():.2f} | Length: {lengths[:episode+1].mean():.2f} | Entropy: {(entropies / lengths)[:episode+1].mean():.5f}                 ", end="\r")
        print(f"==> Step {step: 4} | Score: {returns.mean():.2f} | Length: {lengths.mean():.2f} | Entropy: {(entropies / lengths).mean():.5f}                                                                           ")
        print(' '.join([str(int(x)) for x in reversed(sorted(returns.tolist()))][::len(returns)//10]))
        ## Update model
        overall_gradient = average_gradients(returns, running_gradients)
        updates, opt_state = tx.update(overall_gradient, opt_state)
        params = optax.apply_updates(params, updates)
