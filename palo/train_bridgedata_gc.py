from functools import partial
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.common.common import shard_batch
from jaxrl_m.data.bridge_dataset import BridgeDataset, glob_to_path_list, multi_embed
from jaxrl_m.data.ss2 import SS2Dataset
from jaxrl_m.utils.timer_utils import Timer
from jaxrl_m.vision import encoders
from jaxrl_m.agents import agents
from flax.core import FrozenDict
import tensorflow as tf

import tqdm
import jax
import jax.numpy as jnp
from jaxrl_m.data.language import load_mapping, lang_decode, lang_encodings
from jaxrl_m.data.ss2_language import (
    load_mapping as load_ss2_mapping,
    lang_decode as lang_decode_ss2,
)
from absl import app, flags, logging
from ml_collections import config_flags
import numpy as np
from flax.training import checkpoints
import os

from jaxrl_m.vision.clip import (
    process_text,
    process_image,
    clip_module_vars,
    load_params_from_contrastive_checkpoint,
)

try:
    from jax_smi import initialise_tracking

    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config")
flags.DEFINE_bool("finetune_val_scene", False, "Finetune on val scene")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "bridgedata_config",
    None,
    "File path to the bridgedata configuration.",
    lock_config=False,
)


def recursive_concat(batch1, batch2):
    if isinstance(batch1, dict) or isinstance(batch1, FrozenDict):
        ret = {}
        for k in batch1:
            if k == "inter_dataset_mask":
                A = batch1[k].shape[0]
                B = batch2[k].shape[0]

                mask = jnp.zeros((A + B, A + B))
                ret[k] = mask
            else:
                ret[k] = recursive_concat(batch1[k], batch2[k])
        return ret

    elif isinstance(batch1, list):
        return [recursive_concat(b1, b2) for b1, b2 in zip(batch1, batch2)]
    else:
        return jnp.concatenate([batch1, batch2], axis=0)


def recursive_concat2(batch1, batch2):
    if isinstance(batch1, dict) or isinstance(batch1, FrozenDict):
        ret = {}
        for k in batch1:
            if k == "inter_dataset_mask":
                A = batch1[k].shape[0]
                B = batch2[k].shape[0]

                mask = jnp.zeros((A + B, A + B))
                ret[k] = mask
            else:
                ret[k] = recursive_concat(batch1[k], batch2[k])
        return ret

    elif isinstance(batch1, list):
        return [recursive_concat(b1, b2) for b1, b2 in zip(batch1, batch2)]
    else:
        return jnp.concatenate([batch1, batch2], axis=1)


def main(_):
    devices = jax.local_devices()
    num_devices = len(devices)
    assert FLAGS.config.batch_size % num_devices == 0
    print(FLAGS.config)

    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": "jaxrl_m_bridgedata",
            "exp_descriptor": FLAGS.name,
            "entity": "widowx-gcrl",
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=FLAGS.config.to_dict(),
        debug=FLAGS.debug,
    )

    save_dir = tf.io.gfile.join(
        FLAGS.config.save_dir,
        wandb_logger.config.project,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
    )

    load_mapping(FLAGS.config.data_path, augmented=FLAGS.config.augment_language)

    train_data_iters = {}
    val_datasets = {}

    assert type(FLAGS.bridgedata_config.include[0]) == list
    task_paths = [
        glob_to_path_list(path, prefix=FLAGS.config.data_path, exclude=FLAGS.bridgedata_config.exclude)
        for path in FLAGS.bridgedata_config.include
    ]

    train_paths = [[os.path.join(path, "train/out.tfrecord") for path in sub_list] for sub_list in task_paths]
    val_paths = [[os.path.join(path, "val/out.tfrecord") for path in sub_list] for sub_list in task_paths]
    train_paths = [[path for path in sub_list if tf.io.gfile.exists(path)] for sub_list in train_paths]
    val_paths = [[path for path in sub_list if tf.io.gfile.exists(path)] for sub_list in val_paths]

    align_task_paths = [
        glob_to_path_list(path, prefix=FLAGS.config.data_path, exclude=FLAGS.bridgedata_config.exclude)
        for path in FLAGS.bridgedata_config.align_include
    ]
    align_val_paths = [[os.path.join(path, "val/out.tfrecord") for path in sub_list] for sub_list in align_task_paths]
    align_val_paths = [[path for path in sub_list if tf.io.gfile.exists(path)] for sub_list in align_val_paths]

    scene_task_paths = [
        glob_to_path_list(path, prefix=FLAGS.config.data_path, exclude=FLAGS.bridgedata_config.exclude)
        for path in FLAGS.bridgedata_config.val_scene_include
    ]
    scene_train_paths = [
        [os.path.join(path, "train/out.tfrecord") for path in sub_list] for sub_list in scene_task_paths
    ]
    scene_train_paths = [[path for path in sub_list if tf.io.gfile.exists(path)] for sub_list in scene_train_paths]
    scene_val_paths = [[os.path.join(path, "val/out.tfrecord") for path in sub_list] for sub_list in scene_task_paths]
    scene_val_paths = [[path for path in sub_list if tf.io.gfile.exists(path)] for sub_list in scene_val_paths]
    assert len(scene_val_paths[0]) > 0

    lang_ids = list(lang_encodings().keys())
    lang_ids = sorted(lang_ids)
    np.random.RandomState(42).shuffle(lang_ids)
    if FLAGS.config.num_annotations is not None:
        lang_ids = lang_ids[: FLAGS.config.num_annotations]

    if FLAGS.config.domain_weight is not None:

        sample_weights = [(1 - FLAGS.config.domain_weight) / len(train_paths)] * len(train_paths) + [
            FLAGS.config.domain_weight
        ]

        train_data = BridgeDataset(
            train_paths + scene_train_paths,
            FLAGS.config.seed,
            batch_size=FLAGS.config.batch_size,
            num_devices=num_devices,
            train=True,
            filter_lang_ids=lang_ids,
            action_metadata=FLAGS.bridgedata_config.action_metadata,
            sample_weights=sample_weights,
            **FLAGS.config.dataset_kwargs,
        )
    else:
        train_data = BridgeDataset(
            train_paths,
            FLAGS.config.seed,
            batch_size=FLAGS.config.batch_size,
            num_devices=num_devices,
            train=True,
            filter_lang_ids=lang_ids,
            action_metadata=FLAGS.bridgedata_config.action_metadata,
            sample_weights=FLAGS.bridgedata_config.sample_weights,
            **FLAGS.config.dataset_kwargs,
        )

    train_data_iters["bridgedata"] = train_data.get_iterator()

    if not FLAGS.finetune_val_scene:
        val_data = BridgeDataset(
            val_paths,
            FLAGS.config.seed,
            batch_size=FLAGS.config.batch_size,
            action_metadata=FLAGS.bridgedata_config.action_metadata,
            train=False,
            **FLAGS.config.dataset_kwargs,
        )
        val_datasets["bridgedata_val"] = val_data

    val_scene_data = BridgeDataset(
        scene_val_paths,
        FLAGS.config.seed,
        batch_size=64,
        action_metadata=FLAGS.bridgedata_config.action_metadata,
        train=False,
        **FLAGS.config.dataset_kwargs,
    )
    val_datasets["bridgedata_val_scene"] = val_scene_data

    if FLAGS.finetune_val_scene:
        FLAGS.config.eval_interval = 100
        FLAGS.config.save_interval = 100

    if FLAGS.config.ss2_batch_size > 0:

        load_ss2_mapping(FLAGS.config.ss2_labels_path)

        train_data = SS2Dataset(
            root_data_path=FLAGS.config.ss2_train_path,
            seed=FLAGS.config.seed,
            batch_size=FLAGS.config.ss2_batch_size,
            train=True,
            **FLAGS.config.ss2_dataset_kwargs,
        )
        train_data_iters["ss2"] = train_data.get_iterator()

        val_data = SS2Dataset(
            root_data_path=FLAGS.config.ss2_val_path,
            seed=FLAGS.config.seed,
            batch_size=FLAGS.config.ss2_val_batch_size,
            train=False,
            **FLAGS.config.ss2_dataset_kwargs,
        )
        val_datasets["ss2_val"] = val_data

    def process_batch(batch, data_split):
        if not type(batch) == FrozenDict:
            batch = FrozenDict(batch)

        lang_ids = batch["goals"]["language"]
        lang_ids_low_lvl = batch["language_low_level"]
        lang_mask = jnp.array(lang_ids >= 0)
        if "bridgedata" in data_split:
            sents = [lang_decode(x, aug=FLAGS.config.augment_language) for x in lang_ids]
            sents_low_lvl = [lang_decode(x, aug=FLAGS.config.augment_language) for x in lang_ids_low_lvl]
        elif "ss2" in data_split:
            sents = [lang_decode_ss2(x) for x in lang_ids]
            sents_low_lvl = [lang_decode_ss2(x) for x in lang_ids_low_lvl]
        sents = [s if s is not None else "" for s in sents]
        sents_low_lvl = [s if s is not None else "" for s in sents_low_lvl]
        lang_mask_low_lvl = jnp.array([x != "" for x in sents_low_lvl])
        if not FLAGS.config.use_text_embeddings and not FLAGS.config.use_text_embeds_as_inputs:
            if "clip" in FLAGS.config.task_encoders["language"]:
                lang_inputs = process_text(sents)
                lang_inputs_low_lvl = process_text(sents_low_lvl)
            else:
                lang_inputs = jnp.array([multi_embed(x) for x in sents])
                lang_inputs_low_lvl = jnp.array([multi_embed(x) for x in sents_low_lvl])
        else:

            lang_inputs = jnp.ones((len(sents), 64))
            lang_inputs_low_lvl = jnp.ones((len(sents), 64))

        if (
            (not FLAGS.config.dataset_kwargs.clip_preprocessing)
            and "clip" in FLAGS.config.task_encoders.get("image", {})
            and not "image_embed" in batch
        ):
            obs_img = process_image(batch["observations"]["image"])
            goal_img = process_image(batch["goals"]["image"])
            init_img = process_image(batch["initial_obs"]["image"])
        else:
            obs_img = batch["observations"]["image"]
            goal_img = batch["goals"]["image"]
            init_img = batch["initial_obs"]["image"]

        processed_batch = {
            "observations": {"image": obs_img},
            "goals": {
                "image": goal_img,
                "language": lang_inputs,
                "language_low_level": lang_inputs_low_lvl,
                "language_joint": recursive_concat2(lang_inputs, lang_inputs_low_lvl),
                "language_mask": lang_mask,
                "language_low_level": lang_mask_low_lvl,
            },
            "initial_obs": {"image": init_img},
            "inter_dataset_mask": jnp.zeros((obs_img.shape[0], obs_img.shape[0])),
        }

        for key in ["observations", "initial_obs"]:
            if "unprocessed_image" in batch[key]:
                processed_batch[key]["unprocessed_image"] = batch[key]["unprocessed_image"]

        if "image_embed" in batch:
            if FLAGS.config.use_image_embeds_as_inputs:
                processed_batch["goals"]["image"] = batch["image_embed"]
            elif FLAGS.config.use_image_embeddings:
                processed_batch["goals"]["image_embed"] = batch["image_embed"]
        if "text_embed" in batch:
            if FLAGS.config.use_text_embeds_as_inputs:

                processed_batch["goals"]["language"] = batch["text_embed"]
            elif FLAGS.config.use_text_embeddings:
                processed_batch["goals"]["text_embed"] = batch["text_embed"]

        if "ss2" in data_split:
            processed_batch["actions"] = jnp.ones(
                (batch["observations"]["image"].shape[0], 7),
            )
            processed_batch["bc_mask"] = jnp.zeros(batch["observations"]["image"].shape[0])
        else:
            processed_batch["actions"] = batch["actions"]
            processed_batch["bc_mask"] = jnp.ones(batch["observations"]["image"].shape[0])

        return FrozenDict(processed_batch)

    processed_train_iters = {}
    if FLAGS.config.ss2_batch_size > 0:
        processed_train_iters["ss2"] = map(partial(process_batch, data_split="ss2"), train_data_iters["ss2"])
    processed_train_iters["bridgedata"] = map(
        partial(process_batch, data_split="bridgedata"), train_data_iters["bridgedata"]
    )

    def mixed_data_iter(train_data_iters):
        while True:
            try:
                batch1 = next(processed_train_iters["bridgedata"])
                if FLAGS.config.ss2_batch_size > 0:
                    batch2 = next(processed_train_iters["ss2"])
                    batch = recursive_concat(batch1, batch2)
                    batch = FrozenDict(batch)
                    yield batch
                else:
                    yield batch1
            except:
                print("Error in mixed data iter")
                continue

    train_data_iter = mixed_data_iter(train_data_iters)

    example_batch = next(train_data_iter)
    logging.info(f"Batch size: {example_batch['observations']['image'].shape[0]}")
    logging.info(f"Number of devices: {num_devices}")
    logging.info(f"Batch size per device: {example_batch['observations']['image'].shape[0] // num_devices}")

    sharding = jax.sharding.PositionalSharding(devices)
    example_batch = shard_batch(example_batch, sharding)

    encoder_def = encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)
    task_encoder_defs = {
        mod: encoders[arch](**FLAGS.config.task_encoder_kwargs[mod]) for mod, arch in FLAGS.config.task_encoders.items()
    }

    if "drop_encoders" in FLAGS.config and FLAGS.config.drop_encoders:

        task_encoder_defs = {mod: lambda x: x for mod in FLAGS.config.task_encoders.keys()}

    pretrained_params = {}
    if "clip" in FLAGS.config.task_encoders.get("image", {}) or "clip" in FLAGS.config.task_encoders.get(
        "language", {}
    ):
        global clip_module_vars
        if FLAGS.config.clip_resume_path:
            clip_module_vars = load_params_from_contrastive_checkpoint(FLAGS.config.clip_resume_path)

        for module_key in clip_module_vars:

            if module_key == "temperature":
                continue
            pretrained_params["clip_" + module_key] = clip_module_vars[module_key]

    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, construct_rng = jax.random.split(rng)
    agent = agents[FLAGS.config.agent].create(
        rng=construct_rng,
        observations=example_batch["observations"],
        initial_obs=example_batch["initial_obs"],
        goals=example_batch["goals"],
        actions=example_batch["actions"],
        encoder_def=encoder_def,
        task_encoder_defs=task_encoder_defs,
        pretrained_params=pretrained_params,
        **FLAGS.config.agent_kwargs,
    )
    if FLAGS.config.resume_path is not None:
        agent = checkpoints.restore_checkpoint(FLAGS.config.resume_path, target=agent)

    agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())

    timer = Timer()
    for i in tqdm.tqdm(range(int(FLAGS.config.num_steps))):

        timer.tick("dataset")
        batch = shard_batch(next(train_data_iter), sharding)
        timer.tock("dataset")

        timer.tick("train")
        agent, update_info = agent.update(batch)
        timer.tock("train")

        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_logger.log(update_info, step=i)
            wandb_logger.log({"timer": timer.get_average_times()}, step=i)

        if (i + 1) % FLAGS.config.eval_interval == 0:
            logging.info("Evaluating...")
            timer.tick("val")
            for mod in FLAGS.config.task_encoders:
                for split, val_data in val_datasets.items():
                    metrics = []
                    process_batch_ = partial(process_batch, data_split=split)
                    val_data_iter = map(process_batch_, val_data.get_iterator())

                    for batch in tqdm.tqdm(val_data_iter, total=len(val_paths), leave=False):
                        batch = shard_batch(batch, sharding)
                        metrics.append(agent.get_metrics(batch))
                    metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)
                    wandb_logger.log({split: metrics}, step=i)
            timer.tock("val")

        if (i + 1) % FLAGS.config.save_interval == 0:
            logging.info("Saving checkpoint...")
            checkpoint_path = checkpoints.save_checkpoint(save_dir, agent, step=i + 1, keep=1e6)
            logging.info("Saved checkpoint to %s", checkpoint_path)


if __name__ == "__main__":
    app.run(main)
