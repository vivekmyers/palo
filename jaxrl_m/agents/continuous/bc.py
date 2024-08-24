import copy
from functools import partial
from typing import Any
import jax
import jax.numpy as jnp
from jaxrl_m.common.encoding import EncodingWrapper
import numpy as np
import flax
import flax.linen as nn
import optax

from flax.core import FrozenDict
from jaxrl_m.common.typing import Batch
from jaxrl_m.common.typing import PRNGKey
from jaxrl_m.common.common import JaxRLTrainState, nonpytree_field
from jaxrl_m.networks.actor_critic_nets import Policy
from jaxrl_m.networks.actor_critic_nets import ActorCriticWrapper

class BCAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    lr_schedule: Any = nonpytree_field()

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        new_rng, dropout_rng = jax.random.split(self.state.rng)

        def loss_fn(params):
            dist = self.state.apply_fn(
                {"params": params},
                batch["observations"],
                temperature=1.0,
                train=True,
                rngs={"dropout": dropout_rng},
                method="actor",
            )
            pi_actions = dist.mode()
            log_probs = dist.log_prob(batch["actions"])
            mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)
            actor_loss = -(log_probs).mean()
            actor_std = dist.stddev().mean(axis=1)

            return actor_loss, {
                "actor_loss": actor_loss,
                "mse": mse.mean(),
                "log_probs": log_probs,
                "pi_actions": pi_actions,
                "mean_std": actor_std.mean(),
                "max_std": actor_std.max(),
            }

        
        new_state, info = self.state.apply_loss_fns(
            loss_fn, pmap_axis=pmap_axis, has_aux=True
        )
        
        new_state = new_state.replace(rng=new_rng)
        
        info["lr"] = self.lr_schedule(self.state.step)

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames="argmax")
    def sample_actions(
        self,
        observations: np.ndarray,
        *,
        seed: PRNGKey,
        temperature: float = 1.0,
        argmax=False
    ) -> jnp.ndarray:
        dist = self.state.apply_fn(
            {"params": self.state.params},
            observations,
            temperature=temperature,
            method="actor",
        )
        if argmax:
            actions = dist.mode()
        else:
            actions = dist.sample(seed=seed)
        return actions

    @jax.jit
    def get_debug_metrics(self, batch):
        dist = self.state.apply_fn(
            {"params": self.state.params},
            batch["observations"],
            temperature=1.0,
            method="actor",
        )
        pi_actions = dist.mode()
        log_probs = dist.log_prob(batch["actions"])
        mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)

        return {
            "mse": mse,
            "log_probs": log_probs,
            "pi_actions": pi_actions,
        }

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: FrozenDict,
        actions: jnp.ndarray,
        
        encoder_def: nn.Module,
        use_proprio: bool = False,
        network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": False,
            "state_dependent_std": False,
            "dropout": 0.0,
        },
        
        learning_rate: float = 3e-4,
        warmup_steps: int = 1000,
        decay_steps: int = 1000000,
    ):
        encoder_def = EncodingWrapper(
            encoder=encoder_def,
            use_proprio=use_proprio,
            stop_gradient=False,
        )

        encoders = {"actor": encoder_def}
        networks = {
            "actor": Policy(
                action_dim=actions.shape[-1], **network_kwargs, **policy_kwargs
            )
        }

        model_def = ActorCriticWrapper(
            encoders=encoders,
            networks=networks,
        )

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=0.0,
        )
        tx = optax.adam(lr_schedule)

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(init_rng, observations, actions)["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=tx,
            target_params=params,
            rng=create_rng,
        )

        return cls(state, lr_schedule)
