import os
import pickle
from functools import partial
import cv2
from PIL import Image
import tensorflow as tf
from flax.training import checkpoints
from flax.core import FrozenDict
from jaxrl_m.vision import encoders
from jaxrl_m.agents import agents
from jaxrl_m.data.language import load_mapping
import jax
import numpy as np
import jax.numpy as jnp


def squash(imarr, im_size):
    im = Image.fromarray(imarr)
    im = im.resize((im_size, im_size), Image.LANCZOS) # pylint: disable=no-member
    out = np.asarray(im).astype(np.uint8)
    return out


def pad_actions(arr, max_len=150):
    return jnp.concatenate((arr, np.zeros((max_len - arr.shape[0], 7))), axis=0)


def process_actions(path):
    fp = tf.io.gfile.join(path, "policy_out.pkl")
    with tf.io.gfile.GFile(fp, "rb") as f:
        act_list = pickle.load(f)
    if isinstance(act_list[0], dict):
        act_list = [np.array(x["actions"]) for x in act_list]

    return jnp.array(act_list)


def process_state(path):
    fp = tf.io.gfile.join(path, "obs_dict.pkl")
    with tf.io.gfile.GFile(fp, "rb") as f:
        x = pickle.load(f)
    return x["full_state"][:-1] 


def get_trajectories(path, im_size, max_episode_steps):
    trajs = [os.path.join(path, i) for i in os.listdir(path)]
    states = [process_state(i) for i in trajs]
    actions = [pad_actions(process_actions(i), max_episode_steps) for i in trajs]
    ims = [os.path.join(i, "images0") for i in trajs]
    paths = [[os.path.join(p, f"im_{i}.jpg") for i in range(len(os.listdir(p)) - 1)] for p in ims]
    obs = [[squash(cv2.imread(path, cv2.COLOR_BGR2RGB), im_size) for path in p] for p in paths]
    start = paths[0][0]

    return states, actions, obs, start


@jax.jit
def mask_subtask(subtask, start, end):
    padding = jnp.zeros_like(subtask)
    idx = jnp.repeat(jnp.arange(padding.shape[0])[:, None], padding.shape[1], axis=-1)
    mask = (idx >= start) & (idx < end)
    masked = jnp.where(mask, subtask, padding)
    return masked, mask


@partial(jax.jit, static_argnames=["plan_len", "fixed_bounds"])
def sample_partition_boundaries(action_len, actions_padded, plan_len, key, fixed_bounds=False):
    """Sample partition boundaries to be used with a decomposition (`u` in eq. (3) of the paper)."""
    key, subkey = jax.random.split(key)
    if fixed_bounds:
        bounds = (jnp.arange(1, plan_len) * action_len).astype(jnp.int32)
    else:
        bounds = jnp.sort(jax.random.randint(subkey, shape=[plan_len - 1], minval=0, maxval=action_len))
    bounds = jnp.concatenate((jnp.array([0]), bounds, jnp.array([action_len])))
    sizes = bounds[1:] - bounds[:-1]
    subtasks = []
    masks = []
    subtasks, masks = jax.vmap(mask_subtask)(actions_padded, bounds[:-1], bounds[1:])
    subtasks = jnp.stack(subtasks, axis=0)
    masks = jnp.stack(masks, axis=0)
    return subtasks, masks, bounds, sizes


def process_batch(batch):
    """Add attributes to batch for agent processing."""
    if not type(batch) == FrozenDict:
        batch = FrozenDict(batch)
    lang_ids = batch["goals"]["language"]
    lang_mask = jnp.array(lang_ids >= 0)
    lang_inputs = jnp.zeros((len(lang_ids), 512))

    obs_img = batch["observations"]["image"]
    goal_img = batch["goals"]["image"]
    init_img = batch["initial_obs"]["image"]

    add_or_replace = {
        "observations": {"image": obs_img},
        "goals": {
            "image": goal_img,
            "language": lang_inputs,
            "language_mask": lang_mask,
        },
        "initial_obs": {
            "image": init_img,
        },
    }

    add_or_replace["actions"] = batch["actions"]
    add_or_replace["bc_mask"] = jnp.ones(batch["observations"]["image"].shape[0])

    return batch.copy(add_or_replace=add_or_replace)


def initialize_agent(config, checkpoint_path):
    encoder_def = encoders[config["encoder"]](**config["encoder_kwargs"])
    task_encoder_defs = {
        k: encoders[config["task_encoders"][k]](**config["task_encoder_kwargs"][k])
        for k in ("image", "language")
        if k in config["task_encoders"]
    }

    load_mapping("./agent/")

    example_batch = {
        "observations": {"image": np.zeros((10, 224, 224, 3), dtype=np.uint8)},
        "initial_obs": {"image": np.zeros((10, 224, 224, 3), dtype=np.uint8)},
        "goals": {
            "image": np.zeros((10, 224, 224, 3), dtype=np.uint8),
            "language": np.zeros(10, dtype=np.int64),
            "language_mask": np.ones(10),
        },
        "actions": np.zeros((10, 7), dtype=np.float32),
    }

    example_batch = process_batch(example_batch)

    print("\n\nRun config kwargs: ", config["agent_kwargs"])

    rng = jax.random.PRNGKey(0)
    rng, construct_rng = jax.random.split(rng)
    agent = agents[config["agent"]].create(
        rng=construct_rng,
        observations=example_batch["observations"],
        initial_obs=example_batch["initial_obs"],
        goals=example_batch["goals"],
        actions=example_batch["actions"],
        encoder_def=encoder_def,
        task_encoder_defs=task_encoder_defs,
        **config["agent_kwargs"],
    )

    agent = checkpoints.restore_checkpoint(checkpoint_path, agent)

    return agent, rng


