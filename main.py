import ray
from env.mujoco import Hopper
from env.cartpole import CartPole
from typing import Sequence
from ipdb import set_trace
from collections import defaultdict
from functools import partial
import jax
from jax import random, numpy as jnp                # JAX NumPy
from flax import linen as nn           # The Linen API
import numpy as np                     # Ordinary NumPy
import optax                           # Optimizers
# ray.init(num_cpus=4)

ENV = CartPole

@ray.remote(num_cpus=.1)
class Environment(object):
    def __init__(self, env_id):
        self.env_id = env_id

    def reset(self, game_id):
        self.env = ENV()
        self.score = 0
        self.game_id = game_id
        return self.env_id, self.game_id, self.env.state_rep(), None

    def step(self, action):
        self.score += self.env.step(action)
        return self.env_id, self.game_id, self.env.state_rep(), None if not self.env.terminated else self.score

class MLP(nn.Module):
  @nn.compact
  def __call__(self, x):
    for feat in [32, 64, 64]:
      x = nn.relu(nn.Dense(feat)(x))
    x = nn.Dense(ENV.action_count)(x)
    return x

@partial(jax.jit, static_argnums=1)
@partial(jax.grad, has_aux=True)
def act_on(params, agent, state, rng):
    logits = agent.apply(params, state)
    action = jax.random.categorical(rng, logits, -1)
    action_log_probs = -jnp.take(jax.nn.log_softmax(logits, -1), action, -1)
    return action_log_probs, (action_log_probs, logits, action)

if __name__ == '__main__':
    GAMES_PER_PHASE = 100
    ENVIRONMENTS = 100
    LR = .0001

    ## Initialize environments
    environments = {env_id: Environment.remote(env_id) for env_id in range(ENVIRONMENTS)}

    ## Initialize agent
    rng = jax.random.PRNGKey(0)
    agent = MLP()
    rng, _rng = random.split(rng, 2)
    params = agent.init(_rng, jnp.ones([*ENV.state_shape]))
    tx = optax.adam(LR)
    opt_state = tx.init(params)

    for step in range(10000000):
        print(f"\n==> Step {step: 4}.", end="\r")
        ## Reset accumulators
        game_gradients = defaultdict(lambda :jax.tree_map(lambda x: jnp.zeros_like(x), params))
        results = np.zeros(GAMES_PER_PHASE)
        ## Begin interacting
        games_deployed = 0
        games_completed = 0
        futures = []
        for env_id in environments:
            futures.append(environments[env_id].reset.remote(games_deployed))
            games_deployed += 1
        while games_completed < GAMES_PER_PHASE:
            ready, futures = ray.wait(futures, num_returns=min(2, len(futures)))
            for env_id, game_id, env_state, score in [ray.get(f) for f in ready]:
                if score is None: ## Game is ongoing
                    rng, _rng = random.split(rng, 2)
                    grad, (loss, logits, action) = act_on(params, agent, env_state, _rng)
                    futures.append(environments[env_id].step.remote(action))
                    game_gradients[game_id] = jax.tree_map(lambda gg, g: gg + g, game_gradients[game_id], grad)
                else: ## Game has terminated
                    results[game_id] = score
                    games_completed += 1
                    print(f"==> Step {step: 4}. Score: {np.sum(results)/games_completed:.2f} ({games_completed/GAMES_PER_PHASE:.1%})               ", end="\r")
                    if games_deployed < GAMES_PER_PHASE:
                        futures.append(environments[env_id].reset.remote(games_deployed))
                        games_deployed += 1

        ## Update model
        rescaled_gradients = {game_id: jax.tree_map(lambda g: results[game_id] * g, game_gradients[game_id]) for game_id in game_gradients}
        overall_gradient = jax.tree_map(lambda *gs: sum(gs), *list(rescaled_gradients.values()))
        updates, opt_state = tx.update(overall_gradient, opt_state)
        params = optax.apply_updates(params, updates)