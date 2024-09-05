from absl import app, flags, logging
import os
import pickle
import tqdm
from functools import partial

import numpy as np
import cv2
from PIL import Image
import jax
import jax.numpy as jnp
import tensorflow as tf
from flax.training import checkpoints
from flax.core import FrozenDict

from jaxrl_m.vision import encoders
from jaxrl_m.agents import agents
from jaxrl_m.data.bridge_dataset import multi_embed
from jaxrl_m.data.language import load_mapping
from jaxrl_m.common.common import shard_batch
import palo.query_vlm as query_vlm
import json

np.set_printoptions(suppress=True)
logging.set_verbosity(logging.WARNING)


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

    checkpoint_local_path = os.path.join("/tmp/cache/", "/".join(checkpoint_path.split("/")[3:]))
    agent = checkpoints.restore_checkpoint(checkpoint_path, agent)

    return agent, rng


def compute_cost(
    key,
    sharding,
    policy,
    observations,
    states,
    candidate_plan_low,
    candidate_plan_high,
    demo_actions,
    num_partition_samples,
    unnormalize,
    fixed_bounds,
):
    best_total_cost = jnp.inf
    best_total_segment = None
    num_samples_total = num_partition_samples
    chunk_sz = 2000
    info = {"segments": [], "costs": []}

    for chunk_idx in tqdm.tqdm(range(num_samples_total // chunk_sz), desc="sample segment"):
        initial_image_obs = [{"image": observations[i][0], "proprio": states[i][0]} for i in range(len(observations))]
        unrolled_actions = []
        for x in range(len(observations)):
            acts = []
            for y in range(len(candidate_plan_low)):
                im_obs = jnp.array(observations[x])
                proprio = jnp.array(states[x])
                obs = {
                    "image": shard_batch(im_obs, sharding.replicate()),
                    "proprio": shard_batch(proprio, sharding.replicate()),
                }
                lang_l = multi_embed(candidate_plan_low[y])
                lang_h = multi_embed(candidate_plan_high[y])
                lang = shard_batch(jnp.concatenate((lang_h, lang_l), axis=0), sharding.replicate())
                goal_obs = {"language_mask": jnp.ones(1), "language": lang}
                sample = lambda obs, rpg: policy.sample_actions(
                    obs, goal_obs, initial_image_obs[x], seed=rpg, modality="language", argmax=True
                )
                key, subkey = jax.random.split(key)
                rand = jax.random.split(subkey, im_obs.shape[0])
                res = jax.vmap(sample, in_axes=({"image": 0, "proprio": 0}, 0), out_axes=0)(obs, rand)
                res = jax.vmap(unnormalize)(res)
                acts.append(res)
            unrolled_actions.append(jnp.array(acts))

        action_lens = jnp.array([i.shape[1] for i in unrolled_actions])
        plan_len = len(candidate_plan_low)

        unrolled_actions = jnp.stack([jax.vmap(pad_actions)(i) for i in unrolled_actions], axis=0)

        key, next_key = jax.random.split(key)

        num_samples_inner = chunk_sz if chunk_idx < num_samples_total // chunk_sz - 1 else num_samples_total % chunk_sz
        num_samples_inner = num_samples_inner or chunk_sz
        keys = jax.random.split(next_key, num_samples_inner)
        segmented_actions, segment_masks, bounds, sizes = jnp.vectorize(
            lambda l, u, k: sample_partition_boundaries(l, u, plan_len, k, fixed_bounds=fixed_bounds),
            signature="(),(k,i,j),(m)->(k,i,j),(k,i,j),(n),(k)",
        )(action_lens, unrolled_actions, keys[:, None])

        segmented_actions = jnp.moveaxis(segmented_actions, 2, 1)
        segment_masks = jnp.moveaxis(segment_masks, 2, 1)
        bounds = jnp.moveaxis(bounds, 2, 1)
        sizes = jnp.moveaxis(sizes, 2, 1)

        costs = jnp.sum(jnp.square(demo_actions - segmented_actions) * segment_masks, axis=(-1, -2))
        costs = jnp.nan_to_num(costs / sizes)
        costs = jnp.sum(costs, axis=(1, 2))
        best_cost = jnp.min(jnp.array(costs))
        best_idx = jnp.argmin(jnp.array(costs))
        best_segment = bounds[best_idx].T
        if best_cost < best_total_cost:
            best_total_segment = best_segment
            best_total_cost = best_cost

        info["segments"].append(bounds)
        info["costs"].append(costs)

    info["segments"] = jnp.concatenate(info["segments"], axis=0)
    info["costs"] = jnp.concatenate(info["costs"], axis=0)

    return best_total_cost, best_total_segment, info


def optimize_language(
    key,
    policy,
    instruction,
    trajectories,
    unnormalize,
    fixed_bounds=False,
    num_partition_samples=8000,
    num_language_samples=15,
):
    devices = jax.local_devices()

    sharding = jax.sharding.PositionalSharding(devices)
    states, actions, observations, start = trajectories

    policy = jax.device_put(jax.tree_map(jnp.array, policy), sharding.replicate())

    best_segment_per_inst_set = []
    decomp_tree = query_vlm.make_multiple_response(start, instruction, num_language_samples)
    candidates_json, (decomp_high, decomp_low) = query_vlm.process_candidates(decomp_tree)
    info = {}

    candidate_plan_costs = np.zeros([len(decomp_low)], dtype=float)
    for plan_id, (candidate_plan_low, candidate_plan_high) in enumerate(
        tqdm.tqdm(list(zip(decomp_low, decomp_high)), desc="sample plans")
    ):
        cost, best_segment, info_ = compute_cost(
            key,
            sharding,
            policy,
            observations,
            states,
            candidate_plan_low,
            candidate_plan_high,
            actions,
            num_partition_samples,
            unnormalize=unnormalize,
            fixed_bounds=fixed_bounds,
        )

        candidate_plan_costs[plan_id] = cost
        best_segment_per_inst_set.append(best_segment)
        info[(candidate_plan_low, candidate_plan_high)] = info_

    best_plan_id = np.argmin(candidate_plan_costs)
    best_plan = candidates_json[best_plan_id]

    return decomp_tree[best_plan_id], decomp_high[best_plan_id], decomp_low[best_plan_id], info


if __name__ == "__main__":

    # !python palo/optimize_language.py --instruction "First open the drawer, and then put the sweet potato into the opened drawer" --trajectory_path "./data" --checkpoint_path "./agent/checkpoint/" --im_size 224 --config_dir "./agent/config.pkl"

    FLAGS = flags.FLAGS
    flags.DEFINE_string(
        "instruction",
        "First open the drawer, and then put the sweet potato into the opened drawer",
        "Instruction to be optimized",
    )
    flags.DEFINE_string("trajectory_path", "./data", "Path to the trajectories")
    flags.DEFINE_string("checkpoint_path", "./agent/checkpoint/", "Path to the checkpoint")
    flags.DEFINE_string("config_dir", "./agent/config.pkl", "Directory of config")
    flags.DEFINE_integer("im_size", 224, "Size of the image")
    flags.DEFINE_integer("max_subtask_steps", 18, "Maximum number of steps per subtask")
    flags.DEFINE_integer("min_subtask_steps", 2, "Minimum number of steps per subtask")
    flags.DEFINE_integer("max_episode_steps", 150, "Maximum number of steps in an episode")
    flags.DEFINE_integer("num_language_samples", 15, "Number of plans to be sampled")
    flags.DEFINE_integer("num_partition_samples", 8000, "Number of segment samples")
    flags.DEFINE_string("output", "best_plan.json", "Output file name")
    flags.DEFINE_bool("no_sample", False, "Whether to sample the partition boundaries")

    def main(_):
        with open("./agent/config.pkl", "rb") as f:
            config = pickle.load(f)

        action_metadata = config["bridgedata_config"]["action_metadata"]
        action_mean = jnp.array(action_metadata["mean"])
        action_std = jnp.array(action_metadata["std"])
        unnormalize = lambda x: x * action_std + action_mean

        trajectories = get_trajectories(FLAGS.trajectory_path, FLAGS.im_size, FLAGS.max_episode_steps)
        policy, key = initialize_agent(config, FLAGS.checkpoint_path)
        _, high, low, info = optimize_language(
            key,
            policy,
            FLAGS.instruction,
            trajectories,
            unnormalize,
            FLAGS.num_partition_samples,
            FLAGS.num_language_samples,
        )
        data = {"high": high, "low": low}
        with open(FLAGS.output, "w") as f:
            f.write(json.dumps(data))

    app.run(main)
