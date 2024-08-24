from absl import app, flags, logging
import flax
import jax
import optax
import tensorflow as tf
import tqdm
import wandb
import jax.numpy as jnp
from PIL import Image
import pickle
import numpy as np
import os
import cv2

from octo.model.components.action_heads import L1ActionHead
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import (
    freeze_weights,
    TrainState,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("pretrained_path", None, "Path to pre-trained Octo checkpoint directory.")
flags.DEFINE_string("data_dir", None, "Path to finetuning dataset.")
flags.DEFINE_string("save_dir", None, "Directory for saving finetuning checkpoints.")
flags.DEFINE_integer("batch_size", 64, "Batch size for finetuning.")
flags.DEFINE_string("task_name", "Put the purple object in the drawer", "Task to fine tune on. ")
flags.DEFINE_integer("im_size", 256, "Image Size")

flags.DEFINE_bool(
    "freeze_transformer",
    False,
    "Whether pre-trained transformer weights should be frozen.",
)

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
    return jnp.array(x["full_state"][:-1]), x["full_state"][1:]

def make_trajectories(path):
    if None in path:
        trajs = [tf.io.gfile.join(path, i) for i in tf.io.gfile.listdir(path)]
        states = [process_state(i)[0] for i in trajs]
        actions = [process_actions(i) for i in trajs]
        ims = [tf.io.gfile.join(i, "images0") for i in trajs]
        obs = [
            jnp.array(
                [
                    squash(
                        tf.io.decode_jpeg(
                            tf.io.read_file(tf.io.gfile.join(p, f"im_{i}.jpg")),
                            channels=3,
                        ).numpy()
                    )
                    for i in range(len(tf.io.gfile.listdir(p)) - 1)
                ]
            )
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
    return (
        jnp.concatenate(states, axis=0),
        jnp.concatenate(actions, axis=0),
        jnp.concatenate(obs, axis=0),
    )

def main(_):
    assert FLAGS.batch_size % jax.device_count() == 0, "Batch size must be divisible by device count."

    initialize_compilation_cache()
    
    tf.config.set_visible_devices([], "GPU")

    
    wandb.init(name="finetune_instr", project="octo")

    
    logging.info("Loading pre-trained model...")
    model = OctoModel.load_pretrained(FLAGS.pretrained_path)

    logging.info("Loading finetuning dataset...")

    states, actions, obs = make_trajectories(FLAGS.data_dir)
    batch = states, actions, obs
    task = model.create_tasks(texts=FLAGS.batch_size * [FLAGS.task_name])  
    frozen_keys = ("octo_transformer.*",)

    
    config = model.config
    del config["model"]["observation_tokenizers"]["wrist"]
    
    config["model"]["heads"]["action"] = ModuleSpec.create(
        L1ActionHead,
        action_horizon=50,
        action_dim=14,
        readout_key="readout_action",
    )

    
    
    logging.info("Updating model for new observation & action space...")

    learning_rate = optax.join_schedules([optax.linear_schedule(0, 3e-5, 100), optax.constant_schedule(3e-5)], [100])
    tx = optax.adamw(learning_rate)
    tx = freeze_weights(tx, model.params, frozen_keys)
    train_state = TrainState.create(
        rng=jax.random.PRNGKey(1234),
        model=model,
        tx=tx,
    )

    def inference(params, obs):
        nonlocal model
        obs = obs[None]
        observation = {
            "image_primary": obs,
            "timestep_pad_mask": jnp.full((1, 1), True, dtype=bool),
        }
        bound_model = model.module.bind({"params": params})
        chunk = bound_model.octo_transformer(observation, task, observation["timestep_pad_mask"], train=True)

        return chunk

    def all_inference(params, obs):
        rg = jnp.arange(obs.shape[0])
        func = lambda obs, i: inference(params, jax.lax.dynamic_slice(obs, (i, 0, 0, 0), (1, *obs[i].shape)))
        l = jax.vmap(func, in_axes=(None, 0), out_axes=0)
        return l(obs, rg)

    def loss_fn(params, batch, rng, train=True):

        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task_inputs"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )

        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,  
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=train,
        )
        return action_loss, action_metrics

    @jax.jit
    def train_step(state: TrainState, batch):
        _, actions, obs = batch
        rng, key = jax.random.split(state.rng)

        idx = jax.random.permutation(key, jnp.arange(obs.shape[0]))[: FLAGS.batch_size]

        obs = jnp.repeat(obs[:, None, :], 4, axis=1)[idx]
        actions = jnp.repeat(actions[:, None, None, :], 4, axis=2)[idx]

        timestep_pad = jnp.ones((obs.shape[0], 4), dtype=bool)
        action_pad = jnp.ones((obs.shape[0], 7), dtype=bool)
        action_pad = jnp.repeat(action_pad[:, None, None, :], 4, axis=2)

        batch = {
            "observation": {
                "image_primary": obs,
                "timestep_pad_mask": timestep_pad,
            },
            "action": actions,
            "action_pad_mask": action_pad,
            "task_inputs": task,
        }

        (l, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.model.params, batch, rng)
        grad_norm = optax.global_norm(grads)
        updates, _ = state.tx.update(grads, state.opt_state, state.model.params)
        update_norm = optax.global_norm(updates)

        rng, key = jax.random.split(key)
        new_state = state.apply_gradients(grads=grads, rng=rng)
        info.update({"grad_norm": grad_norm, "update_norm": update_norm})
        return new_state, info

    
    logging.info("Starting finetuning...")
    for i in tqdm.tqdm(range(1000), total=1000, dynamic_ncols=True):
        train_state, update_info = train_step(train_state, batch)
        if (i + 1) % 100 == 0:
            update_info = jax.device_get(update_info)
            wandb.log(
                flax.traverse_util.flatten_dict({"training": update_info}, sep="/"),
                step=i,
            )
        if (i + 1) % 100 == 0:
            
            train_state.model.save_pretrained(step=i, checkpoint_path=FLAGS.save_dir)

if __name__ == "__main__":
    app.run(main)
