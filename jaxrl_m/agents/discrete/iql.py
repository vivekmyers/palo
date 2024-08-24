"""Implementations of IQL (w/ no Q function) in discrete action spaces."""
import functools
import jax
import jax.numpy as jnp
import numpy as np
import flax
import flax.linen as nn
import optax
import distrax

from jaxrl_m.common.typing import *
from jaxrl_m.common.common import TrainState, nonpytree_field, target_update
from jaxrl_m.networks.discrete_nets import DiscreteCriticHead
from jaxrl_m.networks.actor_critic_nets import get_latent, ValueCritic

import ml_collections

def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

def iql_value_loss(q, v, expectile):
    value_loss = expectile_loss(q - v, expectile)
    return value_loss.mean(), {
        "value_loss": value_loss.mean(),
        "v": v.mean(),
    }

def iql_critic_loss(q, q_target):
    critic_loss = jnp.square(q - q_target)
    return critic_loss.mean(), {
        "critic_loss": critic_loss.mean(),
        "q": q.mean(),
    }

def iql_actor_loss(q, v, dist, actions, temperature=1.0):
    a = q - v

    exp_a = jnp.exp(a / temperature)
    exp_a = jnp.minimum(exp_a, 100.0)

    log_probs = dist.log_prob(actions)
    actor_loss = -(exp_a * log_probs).mean()

    behavior_accuracy = jnp.mean(jnp.equal(dist.mode(), actions))

    return actor_loss, {
        "actor_loss": actor_loss,
        "behavior_logprob": log_probs.mean(),
        "behavior_accuracy": behavior_accuracy,
        "mean a": a.mean(),
        "max a": a.max(),
        "min a": a.min(),
    }

class DiscreteIQLMultiplexer(nn.Module):
    encoder: nn.Module
    networks: Dict[str, nn.Module]

    def __call__(self, observations):
        latents = get_latent(self.encoder, observations)
        return {k: net(latents) for k, net in self.networks.items()}

    def actor(self, observations):
        latents = get_latent(self.encoder, observations)
        return self.networks["actor"](latents)

    def value(self, observations):
        latents = get_latent(self.encoder, observations)
        return self.networks["value"](latents)

    def critic(self, observations):
        latents = get_latent(self.encoder, observations)
        return self.networks["critic"](latents)

class IQLAgent(flax.struct.PyTreeNode):
    model: TrainState
    target_model: TrainState
    config: dict = nonpytree_field()

    value_method: ModuleMethod = nonpytree_field(default="value")
    actor_method: ModuleMethod = nonpytree_field(default="actor")
    critic_method: ModuleMethod = nonpytree_field(
        default="critic"
    )  

    @functools.partial(jax.pmap, axis_name="pmap")
    def update(agent, batch: Batch):
        def value_loss_fn(params):
            nv = agent.target_model(
                batch["next_observations"], method=agent.value_method
            )
            target_q = batch["rewards"] + agent.config["discount"] * nv * batch["masks"]
            v = agent.model(
                batch["observations"], params=params, method=agent.value_method
            )
            return iql_value_loss(target_q, v, agent.config["expectile"])

        def actor_loss_fn(params):
            nv = agent.target_model(
                batch["next_observations"], method=agent.value_method
            )
            target_q = batch["rewards"] + agent.config["discount"] * nv * batch["masks"]

            v = agent.model(batch["observations"], method=agent.value_method)

            logits = agent.model(
                batch["observations"], params=params, method=agent.actor_method
            )
            dist = distrax.Categorical(logits=logits)

            return iql_actor_loss(
                target_q, v, dist, batch["actions"], agent.config["temperature"]
            )

        def loss_fn(params):
            value_loss, value_info = value_loss_fn(params)
            actor_loss, actor_info = actor_loss_fn(params)

            return value_loss + actor_loss, {**value_info, **actor_info}

        new_model, info = agent.model.apply_loss_fn(
            loss_fn=loss_fn, has_aux=True, pmap_axis="pmap"
        )
        new_target_model = target_update(
            agent.model, agent.target_model, agent.config["target_update_rate"]
        )

        return agent.replace(model=new_model, target_model=new_target_model), info

    @functools.partial(jax.jit, static_argnames=("argmax"))
    def sample_actions(agent, observations, *, seed, temperature=1.0, argmax=False):
        logits = agent.model(observations, method=agent.actor_method)
        dist = distrax.Categorical(logits=logits / temperature)

        if argmax:
            return dist.mode()
        else:
            return dist.sample(seed=seed)

def create_iql_learner(
    seed: int,
    observations: jnp.ndarray,
    n_actions: int,
    
    encoder_def: nn.Module,
    network_kwargs: dict = {
        "hidden_dims": [256, 256],
    },
    
    optim_kwargs: dict = {
        "learning_rate": 6e-5,
    },
    
    discount=0.95,
    expectile=0.9,
    temperature=1.0,
    target_update_rate=0.002,
    **kwargs
):

    print("Extra kwargs:", kwargs)

    rng = jax.random.PRNGKey(seed)

    model_def = DiscreteIQLMultiplexer(
        encoder=encoder_def,
        networks={
            "critic": DiscreteCriticHead(n_actions=n_actions, **network_kwargs),
            "actor": DiscreteCriticHead(n_actions=n_actions, **network_kwargs),
            "value": ValueCritic(**network_kwargs),
        },
    )

    tx = optax.adam(**optim_kwargs)

    params = model_def.init(rng, observations)["params"]
    model = TrainState.create(model_def, params, tx=tx)
    target_model = TrainState.create(model_def, params)

    config = flax.core.FrozenDict(
        dict(
            discount=discount,
            temperature=temperature,
            target_update_rate=target_update_rate,
            expectile=expectile,
        )
    )

    return IQLAgent(model, target_model, config)

def get_default_config():
    config = ml_collections.ConfigDict(
        {
            "algo": "iql",
            "optim_kwargs": {"learning_rate": 6e-5, "eps": 0.00015},
            "network_kwargs": {
                "hidden_dims": (256, 256),
            },
            "discount": 0.95,
            "expectile": 0.9,
            "temperature": 1.0,
            "target_update_rate": 0.002,
        }
    )
    return config
