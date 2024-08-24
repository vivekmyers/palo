import flax.linen as nn
from typing import Dict, Optional, Tuple
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import functools
import inspect


class EncodingWrapper(nn.Module):
    """
    Encodes observations into a single flat encoding, adding additional
    functionality for adding proprioception and stopping the gradient.

    Args:
        encoder: The encoder network.
        use_proprio: Whether to concatenate proprioception (after encoding).
        stop_gradient: Whether to stop the gradient after the encoder.
    """

    encoder: nn.Module
    use_proprio: bool
    stop_gradient: bool

    def __call__(self, observations: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        encoding = self.encoder(observations["image"])
        if self.use_proprio:
            encoding = jnp.concatenate([encoding, observations["proprio"]], axis=-1)
        if self.stop_gradient:
            encoding = jax.lax.stop_gradient(encoding)
        return encoding


class GCEncodingWrapper(nn.Module):
    """
    Encodes observations and goals into a single flat encoding. Handles all the
    logic about when/how to combine observations and goals.

    Takes a tuple (observations, goals) as input.

    Args:
        encoder: The encoder network for observations.
        goal_encoder: The encoder to use for goals (optional). If None, early
            goal concatenation is used, i.e. the goal is concatenated to the
            observation channel-wise before passing it through the encoder.
        use_proprio: Whether to concatenate proprioception (after encoding).
        stop_gradient: Whether to stop the gradient after the encoder.
    """

    encoder: nn.Module
    goal_encoder: Optional[nn.Module]
    use_proprio: bool
    stop_gradient: bool

    def __call__(
        self,
        observations_and_goals: Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
    ) -> jnp.ndarray:
        observations, goals = observations_and_goals
        if self.goal_encoder is None:
            
            encoder_inputs = jnp.concatenate(
                [observations["image"], goals["image"]], axis=-1
            )
            encoding = self.encoder(encoder_inputs)
        else:
            
            encoding = self.encoder(observations["image"])
            goal_encoding = self.goal_encoder(goals["image"])
            encoding = jnp.concatenate([encoding, goal_encoding], axis=-1)

        if self.use_proprio:
            encoding = jnp.concatenate([encoding, observations["proprio"]], axis=-1)

        if self.stop_gradient:
            encoding = jax.lax.stop_gradient(encoding)

        return encoding


def task_expand(task, obs):
    if len(task.shape) + 2 == len(obs.shape):
        x = jnp.array(task)
        x = jnp.expand_dims(x, (-2, -3))
        x = jnp.repeat(x, obs.shape[-2], axis=-2)
        x = jnp.repeat(x, obs.shape[-3], axis=-3)
    elif len(task.shape) == len(obs.shape):
        x = task
    else:
        assert False, (task.shape, obs.shape)
    return jnp.concatenate([obs, x], axis=-1)


class MultimodalEncodingWrapper(nn.Module):
    task_encoder: nn.Module
    modality: str
    early_fusion: bool
    use_proprio: bool
    stop_gradient: bool
    early_fuse_initial_obs: bool
    no_initial: bool

    def __call__(
        self,
        observations_and_goals: Tuple[
            Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]
        ],
        encoder=None,
        override_modality=None,
    ) -> jnp.ndarray:
        if override_modality is not None:
            modality = override_modality
        else:
            modality = self.modality

        observations, goals, initial_obs = observations_and_goals

        if modality == "image" and "image_embed" in goals:
            task_embed = goals["image_embed"]
            task_embed = task_embed / jnp.linalg.norm(
                task_embed, axis=-1, keepdims=True
            )
        elif modality == "language" and "text_embed" in goals:
            task_embed = goals["text_embed"]
            task_embed = task_embed / jnp.linalg.norm(
                task_embed, axis=-1, keepdims=True
            )
        else:
            if self.no_initial:
                init_obs = observations["image"]
                if "raw" in inspect.signature(self.task_encoder).parameters:
                    init_obs_unprocessed = observations["unprocessed_image"]
            else:
                init_obs = initial_obs["image"]
                if "raw" in inspect.signature(self.task_encoder).parameters:
                    init_obs_unprocessed = initial_obs["unprocessed_image"]

            if "raw" in inspect.signature(self.task_encoder).parameters:
                task_embed = self.task_encoder(
                    init_obs, goals[modality], raw=init_obs_unprocessed
                )
            else:
                task_embed = self.task_encoder(init_obs, goals[modality])

        if encoder == None:
            return None, task_embed

        
        
        if "unprocessed_image" in observations:
            obs_img = observations["unprocessed_image"]
        else:
            obs_img = observations["image"]

        if "unprocessed_image" in initial_obs:
            init_obs_img = initial_obs["unprocessed_image"]
        else:
            init_obs_img = initial_obs["image"]

        if self.early_fuse_initial_obs:
            obs = jnp.concatenate([init_obs_img, obs_img], axis=-1)
        else:
            obs = obs_img

        if self.early_fusion:
            encoding = encoder(obs, task_embed)
        else:
            encoding = encoder(obs)
            encoding = jnp.concatenate([encoding, task_embed], axis=-1)

        if self.use_proprio:
            encoding = jnp.concatenate([encoding, observations["proprio"]], axis=-1)

        if self.stop_gradient:
            encoding = jax.lax.stop_gradient(encoding)

        return encoding, task_embed
