from absl import app, flags, logging
import os
import pickle
import time
import tqdm
from functools import partial

import numpy as np
import cv2
import wandb
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
import experiments.call_api
import json

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS
flags.DEFINE_string("instruction", None, "High-level task instruction", required=True)
flags.DEFINE_string(
    "trajectory_path",
    None,
    None,
    "Trajectories collected",
)

flags.DEFINE_string("checkpoint_path", None, "Checkpoint of the agent", required=True)
flags.DEFINE_string("wandb_run_name", None, "Name of wandb run", required=True)
flags.DEFINE_integer("im_size", None, "Name of wandb run", required=True)
flags.DEFINE_integer("max_subtask_steps", 18, "Maximum number of steps per subtask")
flags.DEFINE_integer("min_subtask_steps", 2, "Minimum number of steps per subtask")
flags.DEFINE_integer("max_episode_steps", 150, "Maximum number of steps in an episode")
flags.DEFINE_integer("num_plan_samples", 3, "Number of plans to be sampled")
flags.DEFINE_integer("num_segment_samples", 8000, "Number of segment samples")
flags.DEFINE_string("img_dir", None, "Image prompt directory")
flags.DEFINE_string("output", "best_plan.json", "Output file name")
flags.DEFINE_bool("no_sample", False, "Whether to sample or not")

def pad(arr, max_len=150):
    return jnp.concatenate((arr, np.zeros((max_len - arr.shape[0], 7))), axis=0)

def squash(im):
    im = Image.fromarray(im)
    im = im.resize((FLAGS.im_size, FLAGS.im_size), Image.LANCZOS)
    out = np.asarray(im).astype(np.uint8)
    return out

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
    return x["full_state"][:-1], x["full_state"][1:]

def make_trajectories(path):
    if None in path:
        trajs = [tf.io.gfile.join(path, i) for i in tf.io.gfile.listdir(path)]
        states = [process_state(i)[0] for i in trajs]
        actions = [pad(process_actions(i), FLAGS.max_episode_steps) for i in trajs]
        ims = [tf.io.gfile.join(i, "images0") for i in trajs]
        obs = [
            [
                squash(tf.io.decode_jpeg(tf.io.read_file(tf.io.gfile.join(p, f"im_{i}.jpg")), channels=3).numpy())
                for i in range(len(tf.io.gfile.listdir(p)) - 1)
            ]
            for p in ims
        ]
    else:
        trajs = [os.path.join(path, i) for i in os.listdir(path)]
        states = [process_state(i)[0] for i in trajs]
        actions = [pad(process_actions(i), FLAGS.max_episode_steps) for i in trajs]
        ims = [os.path.join(i, "images0") for i in trajs]
        obs = [
            [
                squash(cv2.cvtColor(cv2.imread(os.path.join(p, f"im_{i}.jpg")), cv2.COLOR_BGR2RGB))
                for i in range(len(os.listdir(p)) - 1)
            ]
            for p in ims
        ]

    return states, jnp.array(actions), obs

@jax.jit
def slice_subtask(subtask, start, end):
    padding = jnp.zeros_like(subtask)
    rolled = jnp.roll(subtask, -start, axis=0)
    masked = jnp.where(jnp.arange(padding.shape[0])[:, None] < end - start, rolled, padding)
    return masked

@jax.jit
def mask_subtask(subtask, start, end):
    padding = jnp.zeros_like(subtask)
    idx = jnp.repeat(jnp.arange(padding.shape[0])[:, None], padding.shape[1], axis=-1)
    mask = (idx >= start) & (idx < end)
    masked = jnp.where(mask, subtask, padding)
    return masked, mask

@partial(jax.jit, static_argnames="plan_len")
def sample_boundaries(action_len, actions_padded, plan_len, key):
    key, subkey = jax.random.split(key)
    if FLAGS.no_sample:
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

def slice_actions(unrolled_actions, sampled_boundaries):

    all_predicted_actions = []

    for sample_id, boundaries in enumerate(sampled_boundaries):
        predicted_actions = np.zeros_like(unrolled_actions[0])

        num_subtasks = len(boundaries) - 1
        for subtask_id in range(num_subtasks):
            start = boundaries[subtask_id]
            end = boundaries[subtask_id + 1]
            if subtask_id == num_subtasks - 1:
                assert end == unrolled_actions.shape[1]
            predicted_actions[start:end] = unrolled_actions[subtask_id][start:end]

        all_predicted_actions.append(predicted_actions)

    all_predicted_actions = np.stack(all_predicted_actions, axis=0)

    return all_predicted_actions

def unnormalize_action(action, mean, std):
    return action * std + mean

def process_batch(batch):
    """
    A hacky method from GRIF, see if we can improve it.
    """
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

def initialize_agent(run, checkpoint_path):
    encoder_def = encoders[run.config["encoder"]](**run.config["encoder_kwargs"])
    task_encoder_defs = {
        k: encoders[run.config["task_encoders"][k]](**run.config["task_encoder_kwargs"][k])
        for k in ("image", "language")
        if k in run.config["task_encoders"]
    }

    load_mapping(run.config["data_path"])

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

    print("\n\nRun config kwargs: ", run.config["agent_kwargs"])

    rng = jax.random.PRNGKey(0)
    rng, construct_rng = jax.random.split(rng)
    agent = agents[run.config["agent"]].create(
        rng=construct_rng,
        observations=example_batch["observations"],
        initial_obs=example_batch["initial_obs"],
        goals=example_batch["goals"],
        actions=example_batch["actions"],
        encoder_def=encoder_def,
        task_encoder_defs=task_encoder_defs,
        **run.config["agent_kwargs"],
    )

    checkpoint_local_path = os.path.join("/tmp/cache/", "/".join(checkpoint_path.split("/")[3:]))

    if not tf.io.gfile.exists(checkpoint_local_path):
        tf.io.gfile.makedirs(checkpoint_local_path)
        tf.io.gfile.copy(os.path.join(checkpoint_path, "checkpoint"), os.path.join(checkpoint_local_path, "checkpoint"))

    agent = checkpoints.restore_checkpoint(checkpoint_path, agent)

    return agent, rng

def evaluate_plan(
    policy,
    observations,
    states,
    candidate_plan,
    candidate_plan_high_level,
    gt_actions,
    action_mean,
    action_std,
    sharding,
    key,
):
    best_total_cost = jnp.inf
    best_total_segment = None
    num_samples_total = 1 if FLAGS.no_sample else FLAGS.num_segment_samples
    chunk_sz = 2000
    for chunk_idx in tqdm.tqdm(range(num_samples_total // chunk_sz), desc="sample segment"):

        initial_image_obs = [{"image": observations[i][0], "proprio": states[i][0]} for i in range(len(observations))]
        unrolled_actions = []
        for x in range(len(observations)):
            acts = []
            for y in range(len(candidate_plan)):
                im_obs = jnp.array(observations[x])
                proprio = jnp.array(states[x])
                obs = {
                    "image": shard_batch(im_obs, sharding.replicate()),
                    "proprio": shard_batch(proprio, sharding.replicate()),
                }
                lang_l = multi_embed(candidate_plan[y])
                lang_h = multi_embed(candidate_plan_high_level[y])
                lang = shard_batch(jnp.concatenate((lang_h, lang_l), axis=0), sharding.replicate())
                goal_obs = {"language_mask": jnp.ones(1), "language": lang}
                sample = lambda obs, rpg: policy.sample_actions(
                    obs, goal_obs, initial_image_obs[x], seed=rpg, modality="language", argmax=True
                )
                key, subkey = jax.random.split(key)
                rand = jax.random.split(subkey, im_obs.shape[0])
                res = jax.vmap(sample, in_axes=({"image": 0, "proprio": 0}, 0), out_axes=0)(obs, rand)
                res = jax.vmap(unnormalize_action, in_axes=(0, None, None), out_axes=0)(res, action_mean, action_std)
                acts.append(res)
            unrolled_actions.append(jnp.array(acts))

        action_lens = jnp.array([i.shape[1] for i in unrolled_actions])
        plan_len = len(candidate_plan)

        unrolled_actions = jnp.stack([jax.vmap(pad)(i) for i in unrolled_actions], axis=0)

        key, next_key = jax.random.split(key)

        num_samples_inner = chunk_sz if chunk_idx < num_samples_total // chunk_sz - 1 else num_samples_total % chunk_sz
        num_samples_inner = num_samples_inner or chunk_sz
        keys = jax.random.split(next_key, num_samples_inner)
        segmented_actions, segment_masks, bounds, sizes = jnp.vectorize(
            lambda l, u, k: sample_boundaries(l, u, plan_len, k),
            signature="(),(k,i,j),(m)->(k,i,j),(k,i,j),(n),(k)",
        )(action_lens, unrolled_actions, keys[:, None])

        segmented_actions = jnp.moveaxis(segmented_actions, 2, 1)
        segment_masks = jnp.moveaxis(segment_masks, 2, 1)
        bounds = jnp.moveaxis(bounds, 2, 1)
        sizes = jnp.moveaxis(sizes, 2, 1)

        costs = jnp.sum(jnp.square(gt_actions - segmented_actions) * segment_masks, axis=(-1, -2))
        costs = jnp.nan_to_num(costs / sizes)
        costs = jnp.sum(costs, axis=(1, 2))
        best_cost = jnp.min(jnp.array(costs))
        best_idx = jnp.argmin(jnp.array(costs))
        best_segment = bounds[best_idx].T
        if best_cost < best_total_cost:
            best_total_segment = best_segment
            best_total_cost = best_cost
            print(best_total_cost)

    return best_total_cost, best_total_segment

def evaluate_plans():
    t = time.time()
    devices = jax.local_devices()

    sharding = jax.sharding.PositionalSharding(devices)
    states, actions, observations = make_trajectories(FLAGS.trajectory_path)

    api = wandb.Api()
    print("wandb run name, ", FLAGS.wandb_run_name)
    run = api.run(FLAGS.wandb_run_name)
    action_metadata = run.config["bridgedata_config"]["action_metadata"]
    action_mean = jnp.array(action_metadata["mean"])
    action_std = jnp.array(action_metadata["std"])

    policy, key = initialize_agent(run, FLAGS.checkpoint_path)
    policy = jax.device_put(jax.tree_map(jnp.array, policy), sharding.replicate())

    best_segment_per_inst_set = []
    plans_json = call_api.make_multiple_response(
        FLAGS.img_dir,
        "Put the purple thing in the drawer after opening it. You also don't have to close the drawer",
        15,
    )
    for p in plans_json:
        print(p)
    candidates_json, (candidate_plans_high_level, candidate_plans) = call_api.process_candidates(plans_json)

    candidate_plan_costs = np.zeros([len(candidate_plans)], dtype=float)
    for plan_id, (candidate_plan, candidate_plan_high_level) in enumerate(
        tqdm.tqdm(list(zip(candidate_plans, candidate_plans_high_level)), desc="sample plans")
    ):
        cost, best_segment = evaluate_plan(
            policy,
            observations,
            states,
            candidate_plan,
            candidate_plan_high_level,
            actions,
            action_mean,
            action_std,
            sharding,
            key,
        )

        candidate_plan_costs[plan_id] = cost
        best_segment_per_inst_set.append(best_segment)

    best_plan_id = np.argmin(candidate_plan_costs)
    best_plan = candidates_json[best_plan_id]

    t_plus = time.time()
    print("The costs per trajectory are: ", candidate_plan_costs)
    print("Time consumption: ", t_plus - t, " seconds")
    print("Best Plan: ")
    print(best_plan)
    print("Best Segment:")
    print(best_segment_per_inst_set[best_plan_id])
    return plans_json[best_plan_id], (candidate_plans_high_level[best_plan_id], candidate_plans[best_plan_id])

def main(_):
    _, high, low = evaluate_plans()
    data = {"high": high, "low": low}
    with open(FLAGS.output, "w") as f:
        f.write(json.dumps(data))

if __name__ == "__main__":
    app.run(main)
