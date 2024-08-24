"""Affordance model."""
import functools

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax

from jaxrl_m.common.typing import Batch
from jaxrl_m.common.typing import Dict
from flax.core import FrozenDict
from jaxrl_m.common.common import TrainState
from jaxrl_m.common.common import nonpytree_field
from jaxrl_m.networks.actor_critic_nets import get_encoding
from jaxrl_m.vision.cvae import CVAE
from jaxlib.xla_extension import DeviceArray

def elbo_loss(reconstruction, goal, mean, logvar, affordance_beta=0.02):
    pred_loss = jnp.mean(jnp.square(reconstruction - goal), axis=-1)
    kld = -0.5 * jnp.sum(1.0 + logvar - mean**2.0 - jnp.exp(logvar), axis=-1)
    assert pred_loss.shape == kld.shape
    elbo_loss = pred_loss + affordance_beta * kld
    return elbo_loss.mean(), {
        "pred_loss": pred_loss.mean(),
        "kld": kld.mean(),
        "elbo_loss": elbo_loss.mean(),
    }

class Affordance(nn.Module):
    networks: Dict[str, nn.Module]

    def affordance(self, observations, goals, seed, return_latents=False):
        latents = get_encoding(
            self.networks["encoder"],
            observations,
            use_proprio=False,
            early_goal_concat=False,
            goals=None,
            stop_gradient=True,
        )
        goal_latents = get_encoding(
            self.networks["encoder"],
            goals,
            use_proprio=False,
            early_goal_concat=False,
            goals=None,
            stop_gradient=True,
        )
        reconstruction = self.networks["affordance"](latents, goal_latents, seed)

        if return_latents:
            info = {"latents": latents, "goal_latents": goal_latents}
            return reconstruction, info
        else:
            return reconstruction

    def __call__(self, observations, goals, seed):
        rets = dict()
        rets["affordance"] = self.affordance(observations, goals, seed=seed)
        return rets

class AffordanceAgent(flax.struct.PyTreeNode):
    model: TrainState
    config: dict = nonpytree_field()

    @functools.partial(jax.pmap, axis_name="pmap")
    def update(agent, batch: Batch, seed: DeviceArray):
        def loss_fn(params):
            rets, latent_dict = agent.model(
                batch["observations"],
                batch["goals"],
                seed=seed,
                return_latents=True,
                params=params,
                method="affordance",
            )
            loss, info = elbo_loss(
                rets["reconstruction"],
                latent_dict["goal_latents"],
                rets["mean"],
                rets["logvar"],
                affordance_beta=agent.config["affordance_beta"],
            )
            return loss, info

        new_model, info = agent.model.apply_loss_fn(
            loss_fn=loss_fn, has_aux=True, pmap_axis="pmap"
        )

        return agent.replace(model=new_model), info

    @jax.jit
    def get_debug_metrics(agent, batch, seed):
        rets, latent_dict = agent.model(
            batch["observations"],
            batch["goals"],
            seed=seed,
            return_latents=True,
            
            method="affordance",
        )
        loss, info = elbo_loss(
            rets["reconstruction"],
            latent_dict["goal_latents"],
            rets["mean"],
            rets["logvar"],
            affordance_beta=agent.config["affordance_beta"],
        )

        return info

def create_affordance_learner(
    seed: int,
    observations: FrozenDict,
    goals: FrozenDict,
    encoder_def: nn.Module,
    
    affordance_kwargs: dict = {
        "encoder_hidden_dims": [256, 256, 256],
        "latent_dim": 8,
        "decoder_hidden_dims": [256, 256, 256],
    },
    
    optim_kwargs: dict = {
        "learning_rate": 3e-4,
    },
    
    affordance_beta=0.02,
    **kwargs
):
    print("Extra kwargs:", kwargs)

    rng = jax.random.PRNGKey(seed)
    affordance_def = CVAE(**affordance_kwargs)

    networks = {
        "encoder": encoder_def,
        "affordance": affordance_def,
    }

    model_def = Affordance(
        networks=networks,
    )

    tx = optax.adam(**optim_kwargs)

    params = model_def.init(rng, observations, goals, rng)["params"]

    model = TrainState.create(model_def, params, tx=tx)

    config = flax.core.FrozenDict(dict(affordance_beta=affordance_beta))

    return AffordanceAgent(model, config)
