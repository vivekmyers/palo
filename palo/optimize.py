import numpy as np
import jax
import jax.numpy as jnp

from absl import app, flags, logging
import tqdm
import json
import pickle


from jaxrl_m.data.bridge_dataset import multi_embed
from jaxrl_m.common.common import shard_batch
from palo import query_vlm
from palo import utils


np.set_printoptions(suppress=True)
logging.set_verbosity(logging.WARNING)


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
    chunk_sz = 2000
    info = {"segments": [], "costs": []}
    batched_itrs = int((num_partition_samples - 1) / chunk_sz + 1)

    for chunk_idx in tqdm.tqdm(range(batched_itrs), desc="sample segment"):
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

        unrolled_actions = jnp.stack([jax.vmap(utils.pad_actions)(i) for i in unrolled_actions], axis=0)

        key, next_key = jax.random.split(key)

        num_samples_inner = chunk_sz if chunk_idx < batched_itrs else num_partition_samples % chunk_sz
        num_samples_inner = num_samples_inner or chunk_sz
        keys = jax.random.split(next_key, num_samples_inner)
        segmented_actions, segment_masks, bounds, sizes = jnp.vectorize(
            lambda l, u, k: utils.sample_partition_boundaries(l, u, plan_len, k, fixed_bounds=fixed_bounds),
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
    actions = jnp.array(actions)

    policy = jax.device_put(jax.tree_map(jnp.array, policy), sharding.replicate())

    best_segment_per_inst_set = []
    decomp_tree = query_vlm.make_multiple_response(start, instruction, num_language_samples)
    _, (decomp_high, decomp_low) = query_vlm.process_candidates(decomp_tree)
    info = []

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
        info += [(candidate_plan_low, candidate_plan_high, info_)]

    best_plan_id = np.argmin(candidate_plan_costs)

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

        trajectories = utils.get_trajectories(FLAGS.trajectory_path, FLAGS.im_size, FLAGS.max_episode_steps)
        policy, key = utils.initialize_agent(config, FLAGS.checkpoint_path)
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
