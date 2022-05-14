from ipdb import set_trace
from collections import defaultdict
import random as pyrand
from functools import partial
import jax
from jax import random, numpy as jnp                # JAX NumPy
from flax import linen as nn           # The Linen API
import numpy as np                     # Ordinary NumPy
import optax                           # Optimizers
import envpool

def front_broadcast(base, to):
    return base.reshape(base.shape[0], *[1]*(len(to.shape) - 1))

@partial(jax.jit, static_argnums=1)
@partial(jax.vmap, in_axes=(None, None, 0, 0))
@partial(jax.grad, has_aux=True)
def act_on(params, agent, state, rng):
    logits = agent.apply(params, state)
    action = jax.random.categorical(rng, logits, -1)
    action_log_probs = -jnp.take(jax.nn.log_softmax(logits, -1), action, -1)
    return action_log_probs, (action_log_probs, logits, action)

@jax.jit
def update_gradients(episodes_ongoing, running_gradients, grad):
    return jax.tree_map(lambda rg, g: rg + g * front_broadcast(episodes_ongoing, g), running_gradients, grad)

@jax.jit
def average_gradients(results, running_gradients):
    return jax.tree_map(lambda rg: (rg * front_broadcast(results, rg)).mean(0), running_gradients)

if __name__ == '__main__':
    EPISODES_PER_GRADIENT = 50
    LR = .00001

    ## Initialize environments
    env = envpool.make("Hopper-v4", env_type="gym", num_envs=EPISODES_PER_GRADIENT)

    ## Model
    class MLP(nn.Module):
        width = 1024
        depth = 4

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(self.width)(x)
            for layer in range(self.depth):
                x += nn.Dense(self.width)(nn.relu(x))
            x = nn.Dense(env.action_space.n)(x)
            return x

    ## Initialize agent
    rng = jax.random.PRNGKey(0)
    agent = MLP()
    rng, _rng = random.split(rng, 2)
    params = agent.init(_rng, jnp.ones([*env.observation_space.shape]))
    tx = optax.adam(LR)
    opt_state = tx.init(params)

    for step in range(10000000):
        print(f"==> Step {step: 4}.", end="\r")

        ## Reset accumulators
        running_gradients = jax.tree_map(lambda x: jnp.zeros((EPISODES_PER_GRADIENT, *x.shape)), params)
        results = np.zeros(EPISODES_PER_GRADIENT)
        episodes_ongoing = np.ones(EPISODES_PER_GRADIENT)
        ## Begin interacting
        state = env.reset()
        while episodes_ongoing.sum() > 0:
            rng = random.split(rng, EPISODES_PER_GRADIENT+1)
            rng, _rng = rng[0], rng[1:]
            grad, (loss, logits, action) = act_on(params, agent, state, _rng)
            state, rew, done, info = env.step(np.array(action))
            results += rew * episodes_ongoing
            episodes_ongoing *= 1 - done.astype(float)
            running_gradients = update_gradients(episodes_ongoing, running_gradients, grad)
            if pyrand.random() < .001: print(jax.nn.softmax(logits,-1)[0])

            print(f"==> Step {step: 4}. ({1 - episodes_ongoing.sum() / EPISODES_PER_GRADIENT:.1%})               ", end="\r")
        print(f"==> Step {step: 4}. Score: {results.mean()}                                                            ")
        ## Update model
        overall_gradient = average_gradients(results, running_gradients)
        updates, opt_state = tx.update(overall_gradient, opt_state)
        params = optax.apply_updates(params, updates)
