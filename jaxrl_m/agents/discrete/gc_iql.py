"""Implementations of goal-conditioned IQL (w/ no Q function) in discrete action spaces."""

import functools
import copy

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
import distrax

from jaxrl_m.common.typing import *
from jaxrl_m.common.common import TrainState, nonpytree_field, target_update
from jaxrl_m.networks.discrete_nets import DiscreteCriticHead
from jaxrl_m.networks.actor_critic_nets import get_encoding, ValueCritic
from jaxrl_m.agents.discrete.iql import iql_value_loss, iql_critic_loss, iql_actor_loss
import ml_collections

class DiscreteGCIQLMultiplexer(nn.Module):
    encoder: nn.Module
    goal_encoder: nn.Module
    networks: Dict[str, nn.Module]

    def get_sg_latent(self, observations, goals):
        latents = get_encoding(self.encoder, observations)
        goal_latents = get_encoding(self.goal_encoder, goals)
        return jnp.concatenate([latents, goal_latents], axis=-1)

    def __call__(self, observations, goals):
        latents = self.get_sg_latent(observations, goals)
        return {k: net(latents) for k, net in self.networks.items()}

    def actor(self, observations, goals):
        latents = self.get_sg_latent(observations, goals)
        return self.networks["actor"](latents)

    def value(self, observations, goals):
        latents = self.get_sg_latent(observations, goals)
        return self.networks["value"](latents)

    def critic(self, observations, goals):
        latents = self.get_sg_latent(observations, goals)
        return self.networks["critic"](latents)

class GoalConditionedIQLAgent(flax.struct.PyTreeNode):
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
                batch["next_observations"], batch["goals"], method=agent.value_method
            )
            target_q = batch["rewards"] + agent.config["discount"] * nv * batch["masks"]
            v = agent.model(
                batch["observations"],
                batch["goals"],
                params=params,
                method=agent.value_method,
            )
            return iql_value_loss(target_q, v, agent.config["expectile"])

        def actor_loss_fn(params):
            nv = agent.target_model(
                batch["next_observations"], batch["goals"], method=agent.value_method
            )
            target_q = batch["rewards"] + agent.config["discount"] * nv * batch["masks"]

            v = agent.model(
                batch["observations"], batch["goals"], method=agent.value_method
            )

            logits = agent.model(
                batch["observations"],
                batch["goals"],
                params=params,
                method=agent.actor_method,
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
    def sample_actions(
        agent, observations, goals, *, seed, temperature=1.0, argmax=False
    ):
        dist = agent.model(
            observations, goals, temperature=temperature, method=agent.actor_method
        )
        if argmax:
            return dist.mode()
        else:
            return dist.sample(seed=seed)

def create_discrete_iql_learner(
    seed: int,
    observations: jnp.ndarray,
    goals: jnp.ndarray,
    n_actions: int,
    
    encoder_def: nn.Module,
    shared_goal_encoder: bool = False,
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

    if shared_goal_encoder:
        goal_encoder_def = encoder_def
    else:
        goal_encoder_def = copy.deepcopy(encoder_def)

    model_def = DiscreteGCIQLMultiplexer(
        encoder=encoder_def,
        goal_encoder=goal_encoder_def,
        networks={
            "actor": DiscreteCriticHead(n_actions=n_actions, **network_kwargs),
            "value": ValueCritic(**network_kwargs),
        },
    )
    tx = optax.adam(**optim_kwargs)

    params = model_def.init(rng, observations, goals)["params"]
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
    return GoalConditionedIQLAgent(model, target_model, config)

def get_default_config():
    config = ml_collections.ConfigDict(
        {
            "algo": "gc_iql",
            "optim_kwargs": {"learning_rate": 6e-5, "eps": 0.00015},
            "network_kwargs": {
                "hidden_dims": (256, 256),
            },
            "discount": 0.95,
            "expectile": 0.9,
            "temperature": 1.0,
            "target_update_rate": 0.002,
            "shared_goal_encoder": False,
        }
    )
    return config
