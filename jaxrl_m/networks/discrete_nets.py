from jaxrl_m.common.typing import *
import jax.numpy as jnp

import flax.linen as nn

from jaxrl_m.common.common import MLP, default_init
from jaxrl_m.networks.actor_critic_nets import get_encoding
import distrax


class DiscreteQ(nn.Module):
    encoder: nn.Module
    network: nn.Module

    def __call__(self, observations):
        latents = get_encoding(self.encoder, observations)
        return self.network(latents)


class DiscreteCriticHead(nn.Module):
    hidden_dims: Sequence[int]
    n_actions: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish

    def setup(self):
        self.q = MLP((*self.hidden_dims, self.n_actions), activation=self.activation)

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        return self.q(observations)


class DiscretePolicy(nn.Module):
    hidden_dims: Sequence[int]
    n_actions: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish

    @nn.compact
    def __call__(self, observations: jnp.ndarray, temperature=1.0) -> jnp.ndarray:
        logits = MLP((*self.hidden_dims, self.n_actions), activation=self.activation)(
            observations
        )
        return distrax.Categorical(logits=logits / temperature)
