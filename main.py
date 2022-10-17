from ipdb import set_trace
from collections import defaultdict
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
@partial(jax.vmap, in_axes=(None, None, 0, None))
@partial(jax.grad, has_aux=True)
def act_on(params, agent, state, rng):
    logits = agent.apply(params, state)
    action = jax.random.categorical(rng, logits, -1)
    action_log_probs = -jnp.take(jax.nn.log_softmax(logits, -1), action, -1)
    return action_log_probs, (action_log_probs, logits, action)

@jax.jit
def update_gradients(env_id, episodes_ongoing, running_gradients, grad):
    ongoing_mask = episodes_ongoing[env_id]
    return jax.tree_map(lambda rg, g: rg.at[env_id].set(rg[env_id] + g * front_broadcast(ongoing_mask, g)), running_gradients, grad)

@jax.jit
def average_gradients(results, running_gradients):
    return jax.tree_map(lambda rg: (rg * front_broadcast(results, rg)).mean(0), running_gradients)

if __name__ == '__main__':
    EPISODES_PER_GRADIENT = 100
    BATCH_SIZE = 10
    LR = .0001
    OPT = optax.adam(LR)

    ## Initialize environments
    env = envpool.make("CartPole-v1", env_type="gym", num_envs=EPISODES_PER_GRADIENT, batch_size=BATCH_SIZE)

    ## Model
    class MLP(nn.Module):
        @nn.compact
        def __call__(self, x):
            for feat in [32, 64, 64]:
                x = nn.relu(nn.Dense(feat)(x))
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
        env.async_reset()
        while episodes_ongoing.sum() > 0:
            state, rew, done, info = env.recv()
            env_id = info["env_id"]
            results[env_id] += rew * episodes_ongoing[env_id]
            episodes_ongoing[env_id] *= 1 - done.astype(float)

            rng, _rng = random.split(rng, 2)
            grad, (loss, logits, action) = act_on(params, agent, state, _rng)
            env.send(np.array(action), env_id)
            running_gradients = update_gradients(env_id, episodes_ongoing, running_gradients, grad)

            print(f"==> Step {step: 4}. ({1 - episodes_ongoing.sum() / EPISODES_PER_GRADIENT:.1%})               ", end="\r")
        print(f"==> Step {step: 4}. Score: {results.mean()}                                                            ")
        ## Update model
        normalized_results = (results - results.min()) / (results.max() - results.min())
        overall_gradient = average_gradients(normalized_results, running_gradients)
        updates, opt_state = tx.update(overall_gradient, opt_state)
        params = optax.apply_updates(params, updates)
