"""Implementations of CQL in discrete action spaces."""
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
from jaxrl_m.networks.discrete_nets import DiscreteQ, DiscreteCriticHead

import ml_collections

def cql_loss_fn(q, q_pred, q_target, cql_temperature=1.0, cql_alpha=1.0):
    td_loss = jnp.square(q_pred - q_target)
    cql_loss = (
        jax.scipy.special.logsumexp(q / cql_temperature, axis=-1)
        - q_pred / cql_temperature
    )
    critic_loss = td_loss + cql_alpha * cql_loss

    dist = distrax.Categorical(logits=q / cql_temperature)
    q_sorted = jnp.sort(q, axis=-1)

    return critic_loss.mean(), {
        "critic_loss": critic_loss.mean(),
        "td_loss": td_loss.mean(),
        "cql_loss": cql_loss.mean(),
        "td_loss max": td_loss.max(),
        "td_loss min": td_loss.min(),
        "entropy": dist.entropy().mean(),
        "q": q_pred.mean(),
        "q_pi": jnp.max(q, axis=-1).mean(),
        "target_q": q_target.mean(),
        "q_gap": jnp.mean(q_sorted[:, -1] - q_sorted[:, -2]),
        "q_gap max": jnp.max(q_sorted[:, -1] - q_sorted[:, -2]),
        "q_gap min": jnp.min(q_sorted[:, -1] - q_sorted[:, -2]),
    }

class CQLAgent(flax.struct.PyTreeNode):
    model: TrainState
    target_model: TrainState
    config: dict = nonpytree_field()
    
    method: ModuleMethod = nonpytree_field(default=None)

    @functools.partial(jax.pmap, axis_name="pmap")
    def update(agent, batch: Batch):
        def loss_fn(params):
            
            nq = agent.target_model(batch["next_observations"], method=agent.method)
            nv = jnp.max(nq, axis=-1)
            q_target = batch["rewards"] + agent.config["discount"] * nv * batch["masks"]

            
            q = agent.model(batch["observations"], params=params, method=agent.method)
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
    def sample_actions(agent, observations, *, seed, temperature=1.0, argmax=False):
        logits = agent.model(observations, method=agent.method)
        dist = distrax.Categorical(logits=logits / temperature)

        if argmax:
            return dist.mode()
        else:
            return dist.sample(seed=seed)

def create_cql_learner(
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
    cql_alpha=1.0,
    temperature=1.0,
    target_update_rate=0.002,
    **kwargs
):

    print("Extra kwargs:", kwargs)

    rng = jax.random.PRNGKey(seed)

    network_def = DiscreteCriticHead(n_actions=n_actions, **network_kwargs)
    model_def = DiscreteQ(encoder=encoder_def, network=network_def)

    tx = optax.adam(**optim_kwargs)
    params = model_def.init(rng, observations)["params"]

    model = TrainState.create(model_def, params, tx=tx)
    target_model = TrainState.create(model_def, params)

    config = flax.core.FrozenDict(
        dict(
            discount=discount,
            cql_alpha=cql_alpha,
            temperature=temperature,
            target_update_rate=target_update_rate,
        )
    )
    return CQLAgent(model, target_model, config)

def get_default_config():
    config = ml_collections.ConfigDict(
        {
            "algo": "cql",
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
