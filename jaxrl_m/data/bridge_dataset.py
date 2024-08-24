from typing import Iterator, List, Union, Optional, Iterable
import fnmatch

import numpy as np
from absl import logging
from flax.core import FrozenDict
import tensorflow as tf
from jaxrl_m.data.tf_augmentations import augment
import jaxrl_m.data.language as language
import tensorflow_hub as hub
import tensorflow_text
import functools

MULTI_MODULE = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
MULTI_MODEL = hub.load(MULTI_MODULE)

@functools.lru_cache(maxsize=None)
def multi_embed(x):
    if not x:
        return np.zeros([512])
    with tf.device("/cpu:0"):
        return MULTI_MODEL(x).numpy()[0]

def glob_to_path_list(
    glob_strs: Union[str, List[str]], prefix: str = "", exclude: Iterable[str] = ()
):
    """Converts a glob string or list of glob strings to a list of paths."""
    if isinstance(glob_strs, str):
        glob_strs = [glob_strs]
    path_list = []
    for glob_str in glob_strs:
        paths = tf.io.gfile.glob(f"{prefix}/{glob_str}")
        filtered_paths = []
        for path in paths:
            if not any(fnmatch.fnmatch(path, e) for e in exclude):
                filtered_paths.append(path)
            else:
                logging.info(f"Excluding {path}")
        print(f"found {len(filtered_paths)} files in {prefix}/{glob_str}")
        path_list += filtered_paths
    return path_list

@tf.function(jit_compile=True)
def _binarize_gripper_actions(actions):
    """Converts gripper actions from continous to binary values (0 and 1).

    We exploit that fact that most of the time, the gripper is fully open (near
    1.0) or fully closed (near 0.0). As it transitions between the two, it
    sometimes passes through a few intermediate values. We relabel those
    intermediate values based on the state that is reached _after_ those
    intermediate values.

    In the edge case that the trajectory ends with an intermediate value, we
    give up on binarizing and relabel that chunk of intermediate values as
    the last action in the trajectory.

    The scan implements the following code:

    new_actions = np.empty_like(actions)
    carry = actions[-1]
    for i in reversed(range(actions.shape[0])):
        if in_between_mask[i]:
            carry = carry
        else:
            carry = float(open_mask[i])
        new_actions[i] = carry
    """
    open_mask = actions > 0.95
    closed_mask = actions < 0.05
    in_between_mask = tf.logical_not(tf.logical_or(open_mask, closed_mask))

    is_open_float = tf.cast(open_mask, tf.float32)

    def scan_fn(carry, i):
        return tf.cond(
            in_between_mask[i],
            lambda: tf.cast(carry, tf.float32),
            lambda: is_open_float[i],
        )

    new_actions = tf.scan(
        scan_fn, tf.range(tf.shape(actions)[0]), actions[-1], reverse=True
    )
    return new_actions

class BridgeDataset:
    """
    Fast parallel tf.data.Dataset-based dataloader for a dataset in the
    BridgeData format. This format consists of TFRecords where each example
    is one trajectory. See `PROTO_TYPE_SPEC` below for the expected format
    for each example in more detail. See `_process_trajectory` below for
    the output format.

    Includes goal relabeling, image augmentations, and sampling from multiple
    datasets with different weights. Goal relabeling uses a 0/-1 reward scheme:
    0 when the next_obs is labeled as the goal, -1 otherwise.

    Args:
        data_paths: List of paths to the data files. If a list of list of paths
            is provided, the data will be sampled from each sub-list according
            to "sample_weights".
        seed: Random seed.
        action_metadata: Dictionary containing metadata (mean and standard
            deviation) of the actions. If provided, actions will be normalized.
        relabel_actions: Whether to relabel the actions with reached states
            (based on proprioception). Also binarizes gripper actions.
        goal_relabel_reached_proportion: Proportion of the transitions that
            will have the next observation labeled as the goal.
        sample_weights: If data_paths is a list of list of paths, this is a
            list of weights with which to sample from each sub-list.
        batch_size: Batch size.
        shuffle_buffer_size: Size of the shuffle buffer. It is split between
            sub-datasets by `sample_weights`.
        prefetch_num_batches: Number of batches to prefetch.
        cache: Whether to cache the dataset in memory.
        train: Whether this dataset is intended for training
            (if set to `False`, will disable shuffling and augmentations).
        augment: Whether to apply image augmentations.
        augment_next_obs_goal_differently: Whether to use different random seeds
            for augmenting the obs, next_obs, and goal image.
        augment_kwargs: Keyword arguments for image augmentations. See
            `jaxrl_m.data.tf_augmentations.augment` for more details.
    """

    def __init__(
        self,
        data_paths: List[Union[str, List[str]]],
        seed: int,
        action_metadata: Optional[dict] = None,
        relabel_actions: bool = True,
        goal_relabel_reached_proportion: float = 0.1,
        sample_weights: Optional[List[float]] = None,
        batch_size: int = 256,
        shuffle_buffer_size: int = 10000,
        prefetch_num_batches: int = 5,
        cache: bool = False,
        train: bool = True,
        augment: bool = False,
        augment_next_obs_goal_differently: bool = False,
        augment_kwargs: Optional[dict] = None,
        take_every=None,
        filter_lang_ids=None,
        labeled_ony=False,
        simple_goal=False,
        drop_remainder=True,
        load_frozen_embeddings=False,
        clip_preprocessing=False,
        goal_relabel_last_proportion=0.5,
        language_keep_proportion=1.0,
        **kwargs,
    ):
        
        self.load_frozen_embeddings = load_frozen_embeddings
        self.PROTO_TYPE_SPEC = {
            "observations/images0": tf.uint8,
            "observations/state": tf.float32,
            "next_observations/images0": tf.uint8,
            "next_observations/state": tf.float32,
            "actions": tf.float32,
            "terminals": tf.bool,
            "truncates": tf.bool,
            "language": tf.int64,
            "language_low_level": tf.int64
        }
        if load_frozen_embeddings:
            self.PROTO_TYPE_SPEC["text_embed"] = tf.float32
            self.PROTO_TYPE_SPEC["image_embed"] = tf.float32

        logging.warning("Extra kwargs passed to BridgeDataset: %s", kwargs)
        if isinstance(data_paths[0], str):
            data_paths = [data_paths]
        if sample_weights is None:
            
            sample_weights = [1 / len(data_paths)] * len(data_paths)
        assert len(data_paths) == len(sample_weights)
        assert np.isclose(sum(sample_weights), 1.0)

        self.relabel_actions = relabel_actions
        self.action_metadata = action_metadata
        self.goal_relabel_reached_proportion = goal_relabel_reached_proportion
        self.cache = cache
        self.augment_kwargs = augment_kwargs
        self.augment_next_obs_goal_differently = augment_next_obs_goal_differently
        self.take_every = take_every
        self.simple_goal = simple_goal
        self.clip_preprocessing = clip_preprocessing
        self.goal_relabel_last_proportion = goal_relabel_last_proportion
        self.language_keep_proportion = language_keep_proportion
        
        datasets = []
        for sub_data_paths in data_paths:
            
            sub_data_paths = [p for p in sub_data_paths if tf.io.gfile.exists(p)]
            datasets.append(self._construct_tf_dataset(sub_data_paths, seed))

        if train:
            
            
            for i in range(len(datasets)):
                datasets[i] = (
                    datasets[i]
                    .repeat()
                    .shuffle(
                        max(int(shuffle_buffer_size * sample_weights[i]), 1), seed + i
                    )
                )

        
        
        
        
        dataset = tf.data.Dataset.sample_from_datasets(
            datasets, sample_weights, seed=seed, stop_on_empty_dataset=train
        )

        if train and augment:
            
            
            dataset = dataset.enumerate(start=seed)
            dataset = dataset.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)

        if self.take_every is not None:
            dataset = dataset.shard(self.take_every, index=0)

        if filter_lang_ids is not None:
            dataset = dataset.filter(
                lambda x: tf.reduce_any(
                    tf.equal(x["goals"]["language"], filter_lang_ids)
                )
            )

        if labeled_ony:
            dataset = dataset.filter(
                lambda x: tf.reduce_any(x["goals"]["language"] != -1)
            )

        dataset = dataset.batch(
            batch_size,
            num_parallel_calls=tf.data.AUTOTUNE,
            drop_remainder=drop_remainder,
            deterministic=not train,
        )

        if clip_preprocessing:
            dataset = dataset.map(
                self._clip_preprocess, num_parallel_calls=tf.data.AUTOTUNE
            )

        
        self.pre_fetch_dataset = dataset

        prefetch_dataset = dataset.prefetch(prefetch_num_batches)

        self.tf_dataset = prefetch_dataset

    def _construct_tf_dataset(
        self,
        paths: List[str],
        seed: int,
    ) -> tf.data.Dataset:
        """
        Constructs a tf.data.Dataset from a list of paths.
        The dataset yields a dictionary of tensors for each transition.
        """
        
        
        dataset = tf.data.Dataset.from_tensor_slices(paths).shuffle(len(paths), seed)

        
        dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=tf.data.AUTOTUNE)

        
        dataset = dataset.map(self._decode_example, num_parallel_calls=tf.data.AUTOTUNE)

        
        dataset = dataset.map(
            self._process_actions, num_parallel_calls=tf.data.AUTOTUNE
        )

        
        if self.cache:
            dataset = dataset.cache()

        
        if self.simple_goal:
            dataset = dataset.map(self._add_goals, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            dataset = dataset.interleave(
                self._add_goals, num_parallel_calls=tf.data.AUTOTUNE
            )

        return dataset

    def _decode_example(self, example_proto):
        
        features = {
            key: tf.io.FixedLenFeature([], tf.string)
            for key in self.PROTO_TYPE_SPEC.keys()
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        parsed_tensors = {
            key: tf.io.parse_tensor(parsed_features[key], dtype)
            for key, dtype in self.PROTO_TYPE_SPEC.items()
        }

        
        ret = {
            "observations": {
                "image": parsed_tensors["observations/images0"],
                "proprio": parsed_tensors["observations/state"],
            },
            "next_observations": {
                "image": parsed_tensors["next_observations/images0"],
                "proprio": parsed_tensors["next_observations/state"],
            },
            "actions": parsed_tensors["actions"],
            "terminals": parsed_tensors["terminals"],
            "truncates": parsed_tensors["truncates"],
            "language": parsed_tensors["language"],
            "language_low_level": parsed_tensors["language_low_level"]
        }

        if self.load_frozen_embeddings:
            ret["text_embed"] = parsed_tensors["text_embed"]
            ret["image_embed"] = parsed_tensors["image_embed"]
        return ret

    def _process_actions(self, traj):
        if self.relabel_actions:
            
            
            movement_actions = (
                traj["next_observations"]["proprio"][:, :6]
                - traj["observations"]["proprio"][:, :6]
            )
            
            continuous_gripper_actions = traj["actions"][:, 6]
            binarized_gripper_actions = _binarize_gripper_actions(
                continuous_gripper_actions
            )

            traj["actions"] = tf.concat(
                [movement_actions, binarized_gripper_actions[:, None]],
                axis=1,
            )

        
        if self.action_metadata is not None:
            traj["actions"] = (
                traj["actions"] - self.action_metadata["mean"]
            ) / self.action_metadata["std"]

        return traj

    def _add_goals(self, traj) -> tf.data.Dataset:
        traj_len = tf.shape(traj["terminals"])[0]

        if self.simple_goal:
            example = {
                "observations": {
                    "image": traj["observations"]["image"][0],
                },
                "next_observations": {"image": traj["next_observations"]["image"][-1]},
                "goals": {
                    "image": traj["observations"]["image"][-1],
                    "language": traj["language"][0],
                },
                "initial_obs": {
                    "image": traj["observations"]["image"][0],
                },
                "actions": traj["actions"][-1],  
            }

            if self.load_frozen_embeddings:
                example["image_embed"] = traj["image_embed"][-1]
                example["text_embed"] = traj["text_embed"][-1]

            return example

        
        future_idxs = tf.cast(
            tf.random.uniform([traj_len])
            * tf.cast(traj_len - tf.range(traj_len), dtype=tf.float32),
            dtype=tf.int32,
        )

        
        goal_reached_mask = (
            tf.random.uniform([traj_len]) < self.goal_relabel_reached_proportion
        )
        goal_last_mask = tf.where(
            traj["language"] == language.NONE,
            tf.random.uniform([traj_len]) < self.goal_relabel_last_proportion,
            tf.random.uniform([traj_len]) < self.language_keep_proportion,
        )
        
        goal_reached_mask = tf.logical_or(
            goal_reached_mask, tf.range(traj_len) == traj_len - 1
        )

        
        future_idxs = tf.where(goal_reached_mask, 0, future_idxs)
        future_idxs = tf.where(
            goal_last_mask, traj_len - tf.range(traj_len) - 1, future_idxs
        )

        
        future_idxs = future_idxs + tf.range(traj_len)

        
        future_idxs = tf.minimum(future_idxs, traj_len - 1)

        
        traj["goals"] = {}
        for key in traj["observations"].keys():
            traj["goals"][key] = tf.gather(traj["next_observations"][key], future_idxs)
        traj["goals"]["language"] = tf.where(
            goal_last_mask, traj["language"], language.NONE
        )

        traj["initial_obs"] = {}
        for key in traj["observations"].keys():
            traj["initial_obs"][key] = tf.repeat(
                tf.expand_dims(traj["observations"][key][0], axis=0), traj_len, axis=0
            )

        
        traj["rewards"] = tf.cast(tf.where(goal_reached_mask, 0, -1), tf.int32)

        
        
        
        
        traj["masks"] = tf.logical_not(traj["terminals"])

        return tf.data.Dataset.from_tensor_slices(traj)

    def _augment(self, seed, image):
        if self.augment_next_obs_goal_differently:
            sub_seeds = tf.unstack(
                tf.random.stateless_uniform(
                    [3, 2], seed=[seed, seed], minval=None, maxval=None, dtype=tf.int32
                )
            )
        else:
            
            sub_seeds = [[seed, seed]] * 4

        for key, sub_seed in zip(
            ["observations", "next_observations", "goals", "initial_obs"], sub_seeds
        ):
            image[key]["image"] = augment(
                image[key]["image"], sub_seed, **self.augment_kwargs
            )
        return image

    CLIP_MEANS = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STDS = [0.26862954, 0.26130258, 0.27577711]

    def _clip_preprocess(self, batch):
        for key in [
            "observations",
            "goals",
            "initial_obs",
        ]:  
            images = batch[key]["image"]
            batch[key]["unprocessed_image"] = tf.identity(images)
            
            images.set_shape([None, 128, 128, 3])
            images = tf.image.resize(images, (224, 224), method="bicubic")
            images = tf.image.convert_image_dtype(images, tf.float32)
            images = images / 255.0
            images = (images - self.CLIP_MEANS) / self.CLIP_STDS
            batch[key]["image"] = images
        return batch

    def get_iterator(self) -> Iterator[FrozenDict]:
        
        
        iterator = map(FrozenDict, self.tf_dataset.as_numpy_iterator())
        return iterator
