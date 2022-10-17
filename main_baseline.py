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
def act_on(params, policy, state, rng):
    logits = policy.apply(params, state)
    action = jax.random.categorical(rng, logits, -1)
    log_probs = jax.nn.log_softmax(logits, -1)
    action_log_probs = -jnp.take(log_probs, action, -1)
    entropy = -jnp.sum(log_probs * jnp.exp(log_probs), -1)
    return action_log_probs, (action_log_probs, logits, action, entropy)

@partial(jax.jit, static_argnums=1)
@partial(jax.jacrev, has_aux=True)
def value_of(params, vf, state):
    out = vf.apply(params, state)
    value, stdev = out[..., 0], out[..., 1]
    return out, (value, stdev)

@jax.jit
def extend_gradient_accumulator(gradient_accumulator):
    return jax.tree_map(lambda ga: jnp.concatenate([ga, jnp.zeros([2*ga.shape[0], *ga.shape[1:]])], 0), gradient_accumulator)

@jax.jit
def append_gradient(gradient_accumulator, gradient, n):
    return jax.tree_map(lambda ga, g, n=n: ga.at[n].set(g), gradient_accumulator, gradient)

@jax.jit
def compute_episode_gradient_p(p_gradients, adv):
    return jax.tree_map(lambda g, adv=adv: np.sum(g[:adv.shape[0]] * front_broadcast(adv, g), 0), p_gradients)

@jax.jit
def gaussian_neg_log_prob(x, mu, stdev):
    # stdev = jnp.ones_like(stdev)
    return -(-0.5 * jnp.log(2 * jnp.pi * stdev**2) - (x - mu)**2 / (2 * stdev**2))

@jax.jit
@partial(jax.grad, argnums=(1,2), has_aux=True)
def grad_avg_gaussian_neg_log_prob(x, mu, stdev):
    agnlp = gaussian_neg_log_prob(x, mu, stdev).mean()
    return agnlp, (agnlp,)

@jax.jit
def compute_episode_gradient_v(v_gradients, returns, values, stdevs):
    (error_grad_μ, error_grad_σ), (loss,) = grad_avg_gaussian_neg_log_prob(returns, values, stdevs)
    def compute_grad(g, error_grad_μ=error_grad_μ, error_grad_σ=error_grad_σ):
        return np.mean(g[:error_grad_μ.shape[0],0] * front_broadcast(error_grad_μ, g[:,0]) +
                       g[:error_grad_σ.shape[0],1] * front_broadcast(error_grad_σ, g[:,1]), 0)
    return jax.tree_map(compute_grad, v_gradients), loss

if __name__ == '__main__':
    LR_P = .0001
    OPT_P = optax.adam
    LR_V = .001
    OPT_V = optax.adam

    ## Initialize environments
    # env = gym.make("HalfCheetahPyBulletEnvDisc2-v0")
    # env = gym.make("CartPole-v1")
    env = gym.make("CartPoleTimed-v1")

    ## Model
    class mlp_policy(nn.Module):
        @nn.compact
        def __call__(self, x):
            for feat in [256, 256, 256, 256]:
                x = nn.relu(nn.Dense(feat)(x))
            x = nn.Dense(env.action_space.n)(x)
            return x

    class mlp_value(nn.Module):
        @nn.compact
        def __call__(self, x):
            for feat in [256, 256, 256, 256]:
                x = nn.relu(nn.Dense(feat)(x))
            x = nn.Dense(2)(x)
            x = x.at[..., 1].set(jnp.exp(x[..., 1]))
            return x

    ## Initialize agent
    rng = jax.random.PRNGKey(0)
    agent = {"p": mlp_policy(), "v": mlp_value()}
    rng, _rng = random.split(rng, 2)
    params_p = agent["p"].init(_rng, jnp.ones([*env.observation_space.shape]))
    params_v = agent["v"].init(_rng, jnp.ones([*env.observation_space.shape]))
    tx_p = optax.chain(OPT_P(LR_P), optax.clip_by_global_norm(0.5))
    opt_state_p = tx_p.init(params_p)
    tx_v = optax.chain(OPT_V(LR_V), optax.clip_by_global_norm(0.5))
    opt_state_v = tx_v.init(params_v)

    for step in count(1, 1):
        print(f"==> Step {step: 4}.", end="\r")
        length = 0
        mem_alloc = 1024
        p_gradients = jax.tree_map(lambda x: jnp.zeros((mem_alloc, *x.shape)), params_p)
        v_gradients = jax.tree_map(lambda x: jnp.zeros((mem_alloc, 2, *x.shape)), params_v)
        rewards = []
        entropies = []
        values = []
        value_stdevs = []
        state, done = env.reset(), False
        while not done:
            rng, _rng = random.split(rng, 2)
            state_j = jnp.array(state)
            p_grad, (loss, logits, action, entropy) = act_on(params_p, agent["p"], state_j, _rng)
            v_grad, (value, stdev) = value_of(params_v, agent["v"], state_j)
            if length >= mem_alloc:
                p_gradients = extend_gradient_accumulator(p_gradients)
                v_gradients = extend_gradient_accumulator(v_gradients)
            p_gradients = append_gradient(p_gradients, p_grad, length)
            v_gradients = append_gradient(v_gradients, v_grad, length)
            state, rew, done, _ = env.step(np.array(action))
            length += 1
            entropies.append(entropy)
            values.append(value)
            value_stdevs.append(stdev)
            rewards.append(rew)
        episode_returns = np.cumsum(rewards[::-1])[::-1]
        episode_values = np.array(values)
        episode_value_stdevs = np.array(value_stdevs)
        episode_gradient_p = compute_episode_gradient_p(p_gradients, episode_returns - episode_values)
        episode_gradient_v, v_loss = compute_episode_gradient_v(v_gradients, episode_returns, episode_values, episode_value_stdevs)
        print(f"==> Step {step: 4} | Score: {sum(rewards):.2f} | Length: {length} | Entropy: {np.mean(entropies):.5f} | Value Loss: {v_loss:.5f}")
        # print(list(zip(episode_returns.astype(int), episode_values.astype(int), episode_value_stdevs.astype(int))))

        ## Update model
        updates_p, opt_state_p = tx_p.update(episode_gradient_p, opt_state_p)
        params_p = optax.apply_updates(params_p, updates_p)
        updates_v, opt_state_v = tx_v.update(episode_gradient_v, opt_state_v)
        params_v = optax.apply_updates(params_v, updates_v)
