from typing import Any

import distrax
import jax
import jax.numpy as jnp
import flax.linen as nn

from jaxrl_m.common.typing import Callable
from jaxrl_m.common.typing import Dict
from jaxrl_m.common.typing import Tuple
from jaxrl_m.common.typing import Optional
from jaxrl_m.common.typing import Sequence
from jaxrl_m.common.common import default_init
from jaxrl_m.common.common import MLP


class ActorCriticWrapper(nn.Module):
    """
    Generic wrapper for all the networks involved in actor-critic type methods.
    This includes an actor, a critic (optional), and a value network (optional).
    The critic network takes in observations and actions, while the actor and
    value networks only take in observations.

    During initialization, the default `__call__` method can be used, which runs
    all of the networks. Later on, the networks can be called individually using
    `module_def.apply({"params": params}, ..., method="...")`.
    """

    encoders: Dict[str, nn.Module]
    networks: Dict[str, nn.Module]

    def setup(self):
        assert self.encoders.keys() == self.networks.keys()
        assert self.networks.keys() in [
            {"actor"},
            {"actor", "critic"},
            {"actor", "critic", "value"},
        ]

    def actor(self, observations, temperature=1.0, train=False):
        return self.networks["actor"](
            self.encoders["actor"](observations), temperature=temperature, train=train
        )

    def critic(self, observations, actions, train=False):
        return self.networks["critic"](
            self.encoders["critic"](observations),
            actions,
            train=train,
        )

    def value(self, observations, train=False):
        return self.networks["value"](self.encoders["value"](observations), train=train)

    def __call__(self, observations, actions):
        rets = {"actor": self.actor(observations)}
        if "critic" in self.networks:
            rets["critic"] = self.critic(observations, actions)
        if "value" in self.networks:
            rets["value"] = self.value(observations)
        return rets


class MultimodalActorCriticWrapper(nn.Module):
    """
    Actor-critic wrapper that maintains different encoders for different modalities.
    """

    encoders: Dict[str, Tuple[nn.Module, Dict[str, nn.Module]]]
    networks: Dict[str, nn.Module]
    share_encoders: bool = False

    def setup(self):
        assert self.encoders.keys() == self.networks.keys()
        assert self.networks.keys() in [
            {"actor"},
            {"actor", "critic"},
            {"actor", "critic", "value"},
        ]
        self.contrastive_temp = self.param(
            "contrastive_temp", nn.initializers.constant(jnp.log(0.07)), ()
        )

    def actor(self, observations, modality, temperature=1.0, freeze_mask=None, train=False):
        
        if not self.share_encoders:
            enc_modality = modality
        else:
            enc_modality = "image"
        action_enc, task_enc = self.encoders["actor"][1][enc_modality](
            observations, self.encoders["actor"][0], override_modality=modality
        )
        if action_enc is not None:
            detach_action_enc = jax.lax.stop_gradient(action_enc)
            if freeze_mask is None:
                mask = 0
            else:
                mask = freeze_mask[:, None] 
            action_enc = mask * detach_action_enc + (1 - mask) * action_enc
            return (
                self.networks["actor"](
                    action_enc, temperature=temperature, train=train
                ),
                task_enc,
            )
        else:
            return None, task_enc

    def critic(self, observations, actions, modality, train=False):
        return self.networks["critic"](
            self.encoders["critic"][modality](observations),
            actions,
            train=train,
        )

    def value(self, observations, modality, train=False):
        return self.networks["value"](
            self.encoders["value"][modality](observations), train=train
        )

    def __call__(self, observations, actions, modalities, **kwargs):
        rets = {}
        for modality in modalities:
            rets[modality] = {"actor": self.actor(observations, modality, **kwargs)}
            
            
            
            
        return rets


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]
    dropout: float = 0.0

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        critic = MLP(
            (*self.hidden_dims, 1),
            activate_final=False,
            small_init_final=False,
            dropout=self.dropout,
        )(observations, train=train)
        return jnp.squeeze(critic, -1)


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish
    dropout: float = 0.0

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, train: bool = False
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP(
            (*self.hidden_dims, 1),
            activate_final=False,
            small_init_final=False,
            activation=self.activation,
            dropout=self.dropout,
        )(inputs, train=train)
        return jnp.squeeze(critic, -1)


def ensemblize(cls, num_qs, out_axes=0):
    return nn.vmap(
        cls,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
    )


class Policy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    tanh_squash_distribution: bool = False
    fixed_std: Optional[jnp.ndarray] = None
    state_dependent_std: bool = True
    dropout: float = 0.0

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, temperature: float = 1.0, train: bool = False
    ) -> distrax.Distribution:
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
            small_init_final=False,
            dropout=self.dropout,
        )(observations, train=train)

        means = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)
        if self.fixed_std is None:
            if self.state_dependent_std:
                log_stds = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(
                    outputs
                )
            else:
                log_stds = self.param(
                    "log_stds", nn.initializers.zeros, (self.action_dim,)
                )
        else:
            log_stds = jnp.log(jnp.array(self.fixed_std))

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max) / temperature

        if self.tanh_squash_distribution:
            distribution = TanhMultivariateNormalDiag(
                loc=means, scale_diag=jnp.exp(log_stds)
            )
        else:
            distribution = distrax.MultivariateNormalDiag(
                loc=means, scale_diag=jnp.exp(log_stds)
            )

        return distribution


class TanhMultivariateNormalDiag(distrax.Transformed):
    def __init__(
        self,
        loc: jnp.ndarray,
        scale_diag: jnp.ndarray,
        low: Optional[jnp.ndarray] = None,
        high: Optional[jnp.ndarray] = None,
    ):
        distribution = distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        layers = []

        if not (low is None or high is None):

            def rescale_from_tanh(x):
                x = (x + 1) / 2  
                return x * (high - low) + low

            def forward_log_det_jacobian(x):
                high_ = jnp.broadcast_to(high, x.shape)
                low_ = jnp.broadcast_to(low, x.shape)
                return jnp.sum(jnp.log(0.5 * (high_ - low_)), -1)

            layers.append(
                distrax.Lambda(
                    rescale_from_tanh,
                    forward_log_det_jacobian=forward_log_det_jacobian,
                    event_ndims_in=1,
                    event_ndims_out=1,
                )
            )

        layers.append(distrax.Block(distrax.Tanh(), 1))

        bijector = distrax.Chain(layers)

        super().__init__(distribution=distribution, bijector=bijector)

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())
