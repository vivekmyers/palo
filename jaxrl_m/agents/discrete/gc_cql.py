"""Implementations of goal-conditioned CQL in discrete action spaces."""
import functools
import copy

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
from jaxrl_m.networks.actor_critic_nets import get_encoding
from jaxrl_m.agents.discrete.cql import target_update, cql_loss_fn

import ml_collections

class DiscreteGCQ(nn.Module):
    encoder: nn.Module
    goal_encoder: nn.Module
    network: nn.Module

    def __call__(self, observations, goals):
        latents = get_encoding(self.encoder, observations)
        goal_latents = get_encoding(self.goal_encoder, goals)
        return self.network(latents, goal_latents)

class GCAdaptor(nn.Module):
    network: nn.Module

    def __call__(self, observations, goals):
        combined = jnp.concatenate([observations, goals], axis=-1)
        return self.network(combined)

class GoalConditionedCQLAgent(flax.struct.PyTreeNode):
    model: TrainState
    target_model: TrainState
    config: dict = nonpytree_field()

    
    method: ModuleMethod = nonpytree_field(default=None)

    @functools.partial(jax.pmap, axis_name="pmap")
    def update(agent, batch: Batch):
        def loss_fn(params):
            
            nq = agent.target_model(
                batch["next_observations"], batch["goals"], method=agent.method
            )
            nv = jnp.max(nq, axis=-1)
            q_target = batch["rewards"] + agent.config["discount"] * nv * batch["masks"]

            
            q = agent.model(
                batch["observations"],
                batch["goals"],
                params=params,
                method=agent.method,
            )
            q_pred = q[jnp.arange(len(batch["actions"])), batch["actions"]]

            
            critic_loss, info = cql_loss_fn(
                q,
                q_pred,
                q_target,
                cql_temperature=agent.config["temperature"],
                cql_alpha=agent.config["cql_alpha"],
            )
            return critic_loss, info

        new_model, info = agent.model.apply_loss_fn(
            loss_fn=loss_fn, pmap_axis="pmap", has_aux=True
        )
        new_target_model = target_update(
            agent.model, agent.target_model, agent.config["target_update_rate"]
        )

        return agent.replace(model=new_model, target_model=new_target_model), info

    @functools.partial(jax.jit, static_argnames=("argmax"))
    def sample_actions(
        agent, observations, goals, *, seed, temperature=1.0, argmax=False
    ):
        logits = agent.model(observations, goals, method=agent.method)
        if argmax:
            return jnp.argmax(logits, axis=-1)
        else:
            dist = distrax.Categorical(logits=logits / temperature)
            return dist.sample(seed=seed)

def create_cql_learner(
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
    cql_alpha=1.0,
    temperature=1.0,
    target_update_rate=0.002,
    **kwargs
):

    print("Extra kwargs:", kwargs)

    rng = jax.random.PRNGKey(seed)
    rng, model_key = jax.random.split(rng)

    if network_def is None:
        network_def = GCAdaptor(
            DiscreteCriticHead(n_actions=n_actions, **network_kwargs)
        )

    if shared_goal_encoder:
        goal_encoder_def = encoder_def
    else:
        goal_encoder_def = copy.deepcopy(encoder_def)

    model_def = DiscreteGCQ(
        encoder=encoder_def,
        goal_encoder=goal_encoder_def,
        network=network_def,
    )

    print(model_def)

    if tx is None:
        tx = optax.adam(**optim_kwargs)

    params = model_def.init(model_key, observations, goals)["params"]

    model = TrainState.create(model_def, params, tx=tx)
    target_model = TrainState.create(model_def, params=params)

    config = flax.core.FrozenDict(
        dict(
            discount=discount,
            cql_alpha=cql_alpha,
            temperature=temperature,
            target_update_rate=target_update_rate,
        )
    )
    return GoalConditionedCQLAgent(rng, model, target_model, config)

def get_default_config():
    config = ml_collections.ConfigDict(
        {
            "algo": "gccql",
            "shared_goal_encoder": False,
            "optim_kwargs": {"learning_rate": 6e-5, "eps": 0.00015},
            "network_kwargs": {
                "hidden_dims": (256, 256),
            },
            "discount": 0.95,
            "cql_alpha": 0.5,
            "temperature": 1.0,
            "target_update_rate": 0.002,
        }
    )
    return config
