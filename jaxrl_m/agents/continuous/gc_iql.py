import copy
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import flax
import flax.linen as nn
import optax

from jaxrl_m.agents.continuous.iql import iql_value_loss
from jaxrl_m.agents.continuous.iql import iql_actor_loss
from jaxrl_m.agents.continuous.iql import iql_critic_loss
from jaxrl_m.agents.continuous.iql import expectile_loss
from flax.core import FrozenDict
from jaxrl_m.common.typing import Batch
from jaxrl_m.common.typing import PRNGKey
from jaxrl_m.common.common import nonpytree_field
from jaxrl_m.common.common import JaxRLTrainState
from jaxrl_m.common.encoding import GCEncodingWrapper
from jaxrl_m.networks.actor_critic_nets import ValueCritic
from jaxrl_m.networks.actor_critic_nets import Policy
from jaxrl_m.networks.actor_critic_nets import Critic
from jaxrl_m.networks.actor_critic_nets import ActorCriticWrapper

class GCIQLAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    config: dict = nonpytree_field()
    lr_schedules: dict = nonpytree_field()

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        def critic_loss_fn(params, rng):
            rng, key = jax.random.split(rng)
            next_v = self.state.apply_fn(
                {"params": self.state.target_params},
                (batch["next_observations"], batch["goals"]),
                train=self.config["dropout_target_networks"],
                rngs={"dropout": key},
                method="value",
            )
            target_q = (
                batch["rewards"] + self.config["discount"] * next_v * batch["masks"]
            )
            rng, key = jax.random.split(rng)
            q = self.state.apply_fn(
                {"params": params},  
                (batch["observations"], batch["goals"]),
                batch["actions"],
                train=True,
                rngs={"dropout": key},
                method="critic",
            )
            return iql_critic_loss(q, target_q)

        def value_loss_fn(params, rng):
            rng, key = jax.random.split(rng)
            q = self.state.apply_fn(
                {"params": self.state.params},  
                (batch["observations"], batch["goals"]),
                batch["actions"],
                train=self.config["dropout_target_networks"],
                rngs={"dropout": key},
                method="critic",
            )
            rng, key = jax.random.split(rng)
            v = self.state.apply_fn(
                {"params": params},  
                (batch["observations"], batch["goals"]),
                train=True,
                rngs={"dropout": key},
                method="value",
            )
            return iql_value_loss(q, v, self.config["expectile"])

        def actor_loss_fn(params, rng):
            rng, key = jax.random.split(rng)
            next_v = self.state.apply_fn(
                {"params": self.state.target_params},
                (batch["next_observations"], batch["goals"]),
                train=self.config["dropout_target_networks"],
                rngs={"dropout": key},
                method="value",
            )
            target_q = (
                batch["rewards"] + self.config["discount"] * next_v * batch["masks"]
            )

            rng, key = jax.random.split(rng)
            v = self.state.apply_fn(
                {"params": self.state.params},  
                (batch["observations"], batch["goals"]),
                train=self.config["dropout_target_networks"],
                rngs={"dropout": key},
                method="value",
            )

            rng, key = jax.random.split(rng)
            dist = self.state.apply_fn(
                {"params": params},  
                (batch["observations"], batch["goals"]),
                train=True,
                rngs={"dropout": key},
                method="actor",
            )
            mask = batch.get("actor_loss_mask", None)
            return iql_actor_loss(
                target_q,
                v,
                dist,
                batch["actions"],
                self.config["temperature"],
                mask=mask,
            )

        loss_fns = {
            "critic": critic_loss_fn,
            "value": value_loss_fn,
            "actor": actor_loss_fn,
        }

        
        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        
        new_state = new_state.target_update(self.config["target_update_rate"])

        
        info["actor_lr"] = self.lr_schedules["actor"](self.state.step)

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames="argmax")
    def sample_actions(
        self,
        observations: np.ndarray,
        goals: np.ndarray,
        *,
        seed: PRNGKey,
        temperature: float = 1.0,
        argmax=False,
    ) -> jnp.ndarray:
        dist = self.state.apply_fn(
            {"params": self.state.params},
            (observations, goals),
            temperature=temperature,
            method="actor",
        )
        if argmax:
            actions = dist.mode()
        else:
            actions = dist.sample(seed=seed)
        return actions

    @jax.jit
    def get_debug_metrics(self, batch, gripper_close_val=None):
        dist = self.state.apply_fn(
            {"params": self.state.params},
            (batch["observations"], batch["goals"]),
            temperature=1.0,
            method="actor",
        )
        pi_actions = dist.mode()
        log_probs = dist.log_prob(batch["actions"])
        mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)

        v = self.state.apply_fn(
            {"params": self.state.params},
            (batch["observations"], batch["goals"]),
            method="value",
        )
        next_v = self.state.apply_fn(
            {"params": self.state.target_params},
            (batch["next_observations"], batch["goals"]),
            method="value",
        )
        target_q = batch["rewards"] + self.config["discount"] * next_v * batch["masks"]
        q = self.state.apply_fn(
            {"params": self.state.params},
            (batch["observations"], batch["goals"]),
            batch["actions"],
            method="critic",
        )

        metrics = {
            "log_probs": log_probs,
            "mse": ((dist.mode() - batch["actions"]) ** 2).sum(-1),
            "pi_actions": pi_actions,
            "online_v": v,
            "online_q": q,
            "target_q": target_q,
            "value_err": expectile_loss(target_q - v, self.config["expectile"]),
            "td_err": jnp.square(target_q - q),
            "advantage": target_q - v,
            "qf_advantage": q - v,
        }

        if gripper_close_val is not None:
            gripper_close_q = self.state.apply_fn(
                {"params": self.state.params},
                (batch["observations"], batch["goals"]),
                jnp.broadcast_to(gripper_close_val, batch["actions"].shape),
                method="critic",
            )
            metrics.update(
                {
                    "gripper_close_q": gripper_close_q,
                    "gripper_close_adv": gripper_close_q - v,
                }
            )

        return metrics

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: FrozenDict,
        goals: FrozenDict,
        actions: jnp.ndarray,
        
        encoder_def: nn.Module,
        shared_encoder: bool = True,
        shared_goal_encoder: bool = True,
        early_goal_concat: bool = False,
        use_proprio: bool = False,
        network_kwargs: dict = {
            "hidden_dims": [256, 256],
            "dropout": 0.0,
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": False,
            "state_dependent_std": False,
        },
        
        learning_rate: float = 3e-4,
        warmup_steps: int = 2000,
        actor_decay_steps: Optional[int] = None,
        
        discount=0.95,
        expectile=0.9,
        temperature=1.0,
        target_update_rate=0.002,
        dropout_target_networks=True,
    ):
        if early_goal_concat:
            
            goal_encoder_def = None
        else:
            if shared_goal_encoder:
                goal_encoder_def = encoder_def
            else:
                goal_encoder_def = copy.deepcopy(encoder_def)

        encoder_def = GCEncodingWrapper(
            encoder=encoder_def,
            goal_encoder=goal_encoder_def,
            use_proprio=use_proprio,
            stop_gradient=False,
        )

        if shared_encoder:
            encoders = {
                "actor": encoder_def,
                "value": encoder_def,
                "critic": encoder_def,
            }
        else:
            
            
            encoders = {
                "actor": encoder_def,
                "value": copy.deepcopy(encoder_def),
                "critic": copy.deepcopy(encoder_def),
            }

        networks = {
            "actor": Policy(
                action_dim=actions.shape[-1], **network_kwargs, **policy_kwargs
            ),
            "value": ValueCritic(**network_kwargs),
            "critic": Critic(**network_kwargs),
        }

        model_def = ActorCriticWrapper(
            encoders=encoders,
            networks=networks,
        )

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(init_rng, (observations, goals), actions)["params"]

        
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=warmup_steps + 1,
            end_value=learning_rate,
        )
        lr_schedules = {
            "actor": lr_schedule,
            "value": lr_schedule,
            "critic": lr_schedule,
        }
        if actor_decay_steps is not None:
            lr_schedules["actor"] = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=actor_decay_steps,
                end_value=0.0,
            )
        txs = {k: optax.adam(v) for k, v in lr_schedules.items()}

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        config = flax.core.FrozenDict(
            dict(
                discount=discount,
                temperature=temperature,
                target_update_rate=target_update_rate,
                expectile=expectile,
                dropout_target_networks=dropout_target_networks,
            )
        )
        return cls(state, config, lr_schedules)
