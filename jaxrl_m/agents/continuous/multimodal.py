import copy
from functools import partial
from typing import Any, Dict, List, Callable
import itertools
import jax
import jax.numpy as jnp
from jaxrl_m.common.encoding import MultimodalEncodingWrapper
import numpy as np
import flax
import flax.linen as nn
import optax

from flax.core import FrozenDict
from flax.core.frozen_dict import freeze
from jaxrl_m.common.typing import Batch
from jaxrl_m.common.typing import PRNGKey
from jaxrl_m.common.common import JaxRLTrainState, nonpytree_field
from jaxrl_m.networks.actor_critic_nets import Policy
from jaxrl_m.networks.actor_critic_nets import MultimodalActorCriticWrapper
from jaxrl_m.agents.continuous.gc_bc import GCBCAgent
from jaxrl_m.data.bridge_dataset import multi_embed
from jaxrl_m.data.language import lang_decode
from jaxrl_m.vision.clip import process_image, process_text
from flax import traverse_util

def find_and_replace(params, key, replacement):
    for k in params.keys():
        if k == key:
            params[k] = replacement
            print(f"Replaced {key} in params")
            return
        if isinstance(params[k], type(params)):
            find_and_replace(params[k], key, replacement)

from flax.core.frozen_dict import freeze

class MultimodalAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    lr_schedule: Any = nonpytree_field()
    modalities: List[str] = nonpytree_field()
    metric: str = nonpytree_field()
    alignment: float
    other_alignment: float
    flatten_task_reps: bool = nonpytree_field()
    freeze_task_B: bool = nonpytree_field()

    @classmethod
    def process_goals(cls, modality, goals):
        
        
        if modality == "image":
            reps = goals[modality]
            mask = jnp.ones(goals[modality].shape[0], dtype=bool)

        
        
        
        
        
        if modality == "language":
            reps1 = goals[modality]
            reps2 = jnp.zeros_like(reps1)
            reps = jnp.concatenate((reps1, reps2), axis=1)
            mask = goals["language_mask"]
        if modality =="language_low_level":
            reps2 = goals[modality] 
            reps1 = jnp.zeros_like(reps2)
            reps = jnp.concatenate((reps1, reps2), axis=1)
            mask = goals["language_low_level_mask"]
        if modality == "language_joint": 
            reps = goals[modality]
            mask = goals["language_mask"] & goals["language_low_level_mask"]
        goals = dict(goals)
        goals.update({modality: reps})
        return goals, mask

    @partial(jax.jit, static_argnames=["modality"])
    def compute_bc_loss(
        self,
        params,
        rng,
        modality,
        mask,
        goals,
        observations,
        actions,
        initial_obs,
        freeze_mask,
    ):
        rng, key = jax.random.split(rng)
        dist, task = self.state.apply_fn(
            {"params": params},
            (observations, goals, initial_obs),
            modality=modality,
            temperature=1.0,
            freeze_mask=freeze_mask,
            train=True,
            rngs={"dropout": key},
            method="actor",
        )

        def mask_reduce(values):
            assert values.shape == mask.shape
            return jnp.sum(values * mask) / jnp.sum(mask)

        pi_actions = dist.mode()
        log_probs = dist.log_prob(actions)
        mse = ((pi_actions - actions) ** 2).sum(-1)
        actor_loss = mask_reduce(-log_probs)
        actor_std = dist.stddev().mean(axis=1)

        return (
            actor_loss,
            task,
            {
                f"{modality}/actor_loss": actor_loss,
                f"{modality}/mse": mask_reduce(mse),
                f"{modality}/log_probs": mask_reduce(log_probs),
                f"{modality}/pi_actions": mask_reduce(pi_actions.mean(-1)),
                f"{modality}/mean_std": mask_reduce(actor_std),
                f"{modality}/max_std": (mask * actor_std).max(),
            },
        )

    @partial(jax.jit, static_argnames="name")
    def compute_alignment_loss(
        self, rep1, rep2, mask1, mask2, temp, name, inter_dataset_mask
    ):
        rep1_ = rep1[:, None, ...]
        rep2_ = rep2[None, :, ...]

        if self.metric == "l2":
            sim = -jnp.sum(jnp.square(rep1_ - rep2_), axis=-1)
        elif self.metric == "downscaled_l2":
            
            sim = -jnp.sum(jnp.square(rep1_ - rep2_), axis=-1) / 64
        elif self.metric == "cosine":
            sim = jnp.sum(rep1_ * rep2_, axis=-1) / (
                jnp.linalg.norm(rep1_, ord=2, axis=-1)
                * jnp.linalg.norm(rep2_, ord=2, axis=-1)
            )
        elif self.metric == "mse":
            sim = -jnp.mean(jnp.square(rep1_ - rep2_), axis=-1)
        elif self.metric == "cosine_loss":
            sim = jnp.sum(rep1_ * rep2_, axis=-1) / (
                jnp.linalg.norm(rep1_, ord=2, axis=-1)
                * jnp.linalg.norm(rep2_, ord=2, axis=-1)
            )
            valid = mask1 & mask2
            unscaled_loss = -jnp.sum(jnp.diag(sim) * valid) / jnp.sum(valid)
            loss = self.alignment * unscaled_loss

            return loss, {
                f"{name}/alignment_loss_unscaled": unscaled_loss,
                f"{name}/alignment_loss": loss,
            }

        else:
            raise ValueError("invalid metric")

        valid = mask1 & mask2
        skip = jnp.all(~valid)
        mask1 = mask1 | skip
        mask2 = mask2 | skip

        weight = sim / jnp.clip(jnp.exp(temp), a_min=0.01, a_max=100)
        weight = weight - inter_dataset_mask * 1e9  
        class1 = jax.nn.log_softmax(weight, where=mask1[:, None], initial=0.0, axis=0)
        class2 = jax.nn.log_softmax(weight, where=mask2[None, :], initial=0.0, axis=1)

        pred1 = jnp.argmax(jnp.exp(class1) * mask1[:, None], axis=0)
        pred2 = jnp.argmax(jnp.exp(class2) * mask2[None, :], axis=1)
        acc1 = jnp.sum((pred1 == jnp.arange(*pred1.shape)) & valid) / jnp.sum(valid)
        acc2 = jnp.sum((pred2 == jnp.arange(*pred2.shape)) & valid) / jnp.sum(valid)

        logits1 = jnp.diag(class1) * valid
        logits2 = jnp.diag(class2) * valid
        avg_logits1 = jnp.sum(logits1) / jnp.sum(valid)
        avg_logits2 = jnp.sum(logits2) / jnp.sum(valid)
        loss = -(self.alignment * avg_logits1 + self.other_alignment * avg_logits2) / 2

        return loss, {
            f"{name}/class1_logits": avg_logits1,
            f"{name}/class2_logits": avg_logits2,
            f"{name}/class1_accuracy": acc1,
            f"{name}/class2_accuracy": acc2,
            f"{name}/alignment_loss": loss,
            f"{name}/temp": temp,
        }

    def update(self, batch: Batch, pmap_axis: str = None):
        goals = batch["goals"]
        masks = []
        
        for mod in self.modalities:
            goals, mask = self.process_goals(mod, goals)
            masks.append(mask)
        new_state, info = self._update(batch, pmap_axis, goals, masks)
        return self.replace(state=new_state), info

    def get_metrics(self, batch: Batch, pmap_axis: str = None):
        goals = batch["goals"]
        masks = []
        for mod in self.modalities:
            goals, mask = self.process_goals(mod, goals)
            masks.append(mask)
        new_state, info = self._update(batch, pmap_axis, goals, masks)
        return info

    @jax.jit
    def _update(self, batch: Batch, pmap_axis, goals, masks):
        def loss_fn(param, rng):
            total_loss = 0.0
            info = {}
            outputs = []

            for mod, mask in zip(self.modalities, masks): 
                observations = batch["observations"]
                actions = batch["actions"]
                freeze_mask = (sum(masks) <= 1) & self.freeze_task_B
                loss, task, aux = self.compute_bc_loss(
                    param,
                    rng,
                    modality=mod,
                    goals=goals,
                    observations=observations,
                    actions=actions,
                    mask=mask * batch["bc_mask"],
                    initial_obs=batch["initial_obs"],
                    freeze_mask=freeze_mask,
                )
                total_loss += loss
                info.update(aux)

                if self.flatten_task_reps:
                    
                    task = [t.reshape((t.shape[0], -1)) for t in task]
                    task = jnp.concatenate(task, axis=-1)
                
                for mod_, mask_, task_ in zip(self.modalities, masks, outputs):
                    name = f"{mod_}_{mod}"
                    temp = param["contrastive_temp"]

                    loss_, aux_ = self.compute_alignment_loss(
                        rep1=task_,
                        rep2=task,
                        mask1=mask_,
                        mask2=mask,
                        temp=temp,
                        name=name,
                        inter_dataset_mask=batch["inter_dataset_mask"],
                    )
                    total_loss += loss_ * 0 
                    info.update(aux_)

                outputs.append(task)
                masks.append(mask)

            return total_loss, info

        return self.state.apply_loss_fns(loss_fn, pmap_axis=pmap_axis, has_aux=True)
        new_state, info = self.minimize_losses(bc_data, alignment_data, pmap_axis)

        
        info["lr"] = self.lr_schedule(self.state.step)

        return new_state, info

    @partial(jax.jit, static_argnames=["argmax", "modality"])
    def sample_actions(
        self,
        observations: np.ndarray,
        goals: np.ndarray,
        initial_obs: np.ndarray,
        modality: str,
        *,
        seed: PRNGKey,
        temperature: float = 1.0,
        argmax=False,
    ) -> jnp.ndarray:
        dist, _ = self.state.apply_fn(
            {"params": self.state.params},
            (observations, goals, initial_obs),
            modality=modality,
            temperature=temperature,
            method="actor",
        )
        if argmax:
            actions = dist.mode()
        else:
            actions = dist.sample(seed=seed)
        return actions

    def get_debug_metrics(self, batch, modality):
        goals, mask = self.process_goals(modality, batch["goals"])
        return self._get_debug_metrics(batch, modality, goals, mask)

    @partial(jax.jit, static_argnames="modality")
    def _get_debug_metrics(self, batch, modality, goals, mask):
        observations = batch["observations"]
        actions = batch["actions"]
        dist, _ = self.state.apply_fn(
            {"params": self.state.params},
            (observations, goals),
            modality=modality,
            temperature=1.0,
            method="actor",
        )

        def mask_reduce(values):
            assert values.shape == mask.shape
            return jnp.sum(values * mask) / jnp.sum(mask)

        pi_actions = dist.mode()
        log_probs = dist.log_prob(actions)
        mse = ((pi_actions - actions) ** 2).sum(-1)

        return {
            f"{modality}/mse": mask_reduce(mse),
            f"{modality}/log_probs": mask_reduce(log_probs),
            f"{modality}/pi_actions": mask_reduce(pi_actions.mean(-1)),
        }

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: FrozenDict,
        actions: jnp.ndarray,
        initial_obs: FrozenDict,
        goals: FrozenDict,
        
        encoder_def: nn.Module,
        task_encoder_defs: dict,
        early_fusion: bool,
        use_proprio: bool = False,
        alignment: float = 1.0,
        other_alignment: float = None,
        metric: str = "cosine",
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
        pretrained_params: dict = {},
        flatten_task_reps: bool = False,
        share_encoders: bool = False,
        early_fuse_initial_obs: bool = False,
        freeze_task_B=False,
        clip_encoder_lr_multiplier: float = 1.0,
        no_initial=False,
    ):
        task_encoder_defs = {
            mod: MultimodalEncodingWrapper(
                task_encoder=task_encoder_defs[mod],
                modality=mod,
                early_fusion=early_fusion,
                use_proprio=use_proprio,
                stop_gradient=False,
                early_fuse_initial_obs=early_fuse_initial_obs,
                no_initial=no_initial,
            )
            for mod in task_encoder_defs
        }
        modalities = list(task_encoder_defs)

        encoders = {"actor": (encoder_def, task_encoder_defs)}
        networks = {
            "actor": Policy(
                action_dim=actions.shape[-1], **network_kwargs, **policy_kwargs
            )
        }

        model_def = MultimodalActorCriticWrapper(
            encoders=encoders,
            networks=networks,
            share_encoders=share_encoders,
        )

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=0.0,
        )
        lr_schedule_enc = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate * clip_encoder_lr_multiplier,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=0.0,
        )

        partition_optimizers = {
            "encoder": optax.adam(lr_schedule_enc),
            "actor": optax.adam(lr_schedule),
        }

        observations = dict(observations)
        initial_obs = dict(initial_obs)
        for mod in modalities:
            goals, mask = cls.process_goals(mod, goals)

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            (observations, goals, initial_obs),
            actions,
            modalities,
            freeze_mask=~mask,
        )["params"]

        params = flax.core.frozen_dict.FrozenDict(params).unfreeze()
        for key in pretrained_params:
            find_and_replace(params, key, pretrained_params[key])
        params = freeze(params)

        def is_clip_encoder(path):
            components = [
                "clip_visual_projection",
                "clip_vision_model",
                "clip_text_projection",
                "clip_text_model",
                "contrastive_temp",
            ]
            for comp in components:
                if comp in path:
                    return True

        param_partitions = freeze(
            traverse_util.path_aware_map(
                lambda path, v: "encoder" if is_clip_encoder(path) else "actor", params
            )
        )

        
        tx = optax.multi_transform(partition_optimizers, param_partitions)

        flat = list(traverse_util.flatten_dict(param_partitions).items())
        print("optimizer partitions:")
        print(freeze(traverse_util.unflatten_dict(dict(flat[:2] + flat[-2:]))))
        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=tx,
            target_params=params,
            rng=create_rng,
        )

        if other_alignment is None:
            other_alignment = alignment

        return cls(
            state,
            lr_schedule,
            modalities,
            metric,
            alignment,
            other_alignment,
            flatten_task_reps,
            freeze_task_B,
        )
