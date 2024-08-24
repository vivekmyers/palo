import sys
import os
import numpy as np
from PIL import Image
from flax.training import checkpoints
import traceback
import wandb
from jaxrl_m.vision import encoders
from jaxrl_m.agents import agents
from jaxrl_m.data.bridge_dataset import multi_embed
from jaxrl_m.data.language import load_mapping, lang_encode
import matplotlib
from absl import app, flags, logging
import matplotlib.pyplot as plt
import time
from datetime import datetime
import jax
import time
import tensorflow as tf
import jax.numpy as jnp
from flax.core import FrozenDict
import pickle
from jaxrl_m.vision.clip import process_image, process_text, create_from_checkpoint
from multiprocessing import Manager, Pool

import cv2
import io



np.set_printoptions(suppress=True)


FLAGS = flags.FLAGS


flags.DEFINE_string("checkpoint_path", "XXXX", "Path to checkpoint")
flags.DEFINE_string("wandb_run_name", "widowx-gcrl/jaxrl_m_bridgedata/all_multimodal_lcbc_20240425_003502", "Name of wandb run")
flags.DEFINE_integer("im_size", 224, "Image size")
FLAGS(sys.argv)




STEP_DURATION_MESSAGE = """
Bridge data was collected with non-blocking control and a step duration of 0.2s.
However, we relabel the actions to make it look like the data was collected with
blocking control and we evaluate with blocking control.
We also use a step duration of 0.4s to reduce the jerkiness of the policy.
Be sure to change the step duration back to 0.2 if evaluating with non-blocking control.
"""



def unnormalize_action(action, mean, std):
    return action * std + mean


def squash(im):
    im = Image.open(io.BytesIO(im))
    im = im.resize((FLAGS.im_size, FLAGS.im_size), Image.LANCZOS)
    out = np.asarray(im).astype(np.uint8)
    return out

def process_state(path):
    fp = tf.io.gfile.join(path, "obs_dict.pkl")
    with tf.io.gfile.GFile(fp, "rb") as f:
        x = pickle.load(f)
    return x["full_state"], x["full_state"][1:]

def process_actions(path):  
    fp = tf.io.gfile.join(path, "policy_out.pkl")
    with tf.io.gfile.GFile(fp, "rb") as f:
        act_list = pickle.load(f)
    if isinstance(act_list[0], dict):
        act_list = [x["actions"] for x in act_list]
    return act_list


def process_batch(batch):
    data_split = "bridgedata"
    if not type(batch) == FrozenDict:
        batch = FrozenDict(batch)
    lang_ids = batch["goals"]["language"]
    lang_mask = jnp.array(lang_ids >= 0)
    sents = ["placeholder" for x in lang_ids]
    

    
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

clip = None
run = None

def make_agent():
    """
    Makes an agent, easier for using the same weights
    """
    assert tf.io.gfile.exists(FLAGS.checkpoint_path)
    api = wandb.Api() 
    print("wandb run name, ", FLAGS.wandb_run_name)
    run = api.run(FLAGS.wandb_run_name)

    action_metadata = run.config["bridgedata_config"]["action_metadata"]
    action_mean = np.array(action_metadata["mean"])
    action_std = np.array(action_metadata["std"])

    encoder_def = encoders[run.config["encoder"]](**run.config["encoder_kwargs"])
    task_encoder_defs = {
        k: encoders[run.config["task_encoders"][k]](
            **run.config["task_encoder_kwargs"][k]
        )
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
        **run.config["agent_kwargs"]
    )

    checkpoint_local_path = os.path.join(
        "/tmp/cache/", "/".join(FLAGS.checkpoint_path.split("/")[3:])
    )

    if not tf.io.gfile.exists(checkpoint_local_path):
        tf.io.gfile.makedirs(checkpoint_local_path)
        tf.io.gfile.copy(os.path.join(FLAGS.checkpoint_path, 'checkpoint'), os.path.join(checkpoint_local_path, 'checkpoint'))

    
    agent = checkpoints.restore_checkpoint(FLAGS.checkpoint_path, agent)

    return agent, {"mean": action_mean, "std": action_std}

def eval_on_traj(plan_h, plan_l, agent, config, path, ims_to_eval):
    
    """
    Plan is a tuple of a list of tuples of index and tuple of commands:
    Format:
    (u_1a, L_1): from t = 0 to t = u_1 (exclusive), use L_1 for EMBEDDED high level instruction
    (u_1b, L_2)

    Note: this has to be 
    """
    cout_h, cout_l = 0, 0 
    
    

    
    """
    Cutoff
    """
    modality='l' 
    rng = jax.random.PRNGKey(0)

    action_mean, action_std = config["mean"], config["std"]
    all_actions = []
    state, _ = process_state(path)
    im_dir = tf.io.gfile.join(path, "images0/")
    
    
    im0 = bytearray(tf.io.gfile.GFile(im_dir+"im_0.jpg", "rb").read())
    im0 = squash(im0)
    im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
    ims = []

    
    

    
    initial_image_obs = (
        im0.reshape(3, 224, 224).transpose(1, 2, 0) * 255
    ).astype(np.uint8)
    ims.append(Image.fromarray(im0))
    
    initial_obs = {
        "image": initial_image_obs,
        "proprio": state[0]
    }
    goal_obs = {}
    actions = []
    proprio = state[0]
    for i in ims_to_eval:
        
        if i != 0:
            cout_h, cout_l = min(len([j[0] for j in plan_h if int(j[0])<i])-1, 0), min(len([j[0] for j in plan_l if int(j[0])<i])-1, 0) 
            
        else:
            cout_h, cout_l = 0, 0
        
        prompt_l, prompt_h = plan_l[cout_l][1], plan_h[cout_h][1]
        lang_inputs_low_level = prompt_l
        lang_inputs_high_level = prompt_h 


        lang_inputs = jnp.concatenate((lang_inputs_high_level, lang_inputs_low_level), axis=0)
        goal_obs.update({"language": lang_inputs, "language_mask": jnp.ones(1)})
        
        obs = bytearray(tf.io.gfile.GFile(im_dir+f"im_{i}.jpg", "rb").read())
        obs_pol = squash(obs)
        obs_pol = cv2.cvtColor(obs_pol, cv2.COLOR_BGR2RGB)
        image_obs = (
            obs_pol.reshape(3, 224, 224).transpose(1, 2, 0) * 255
        ).astype(np.uint8)

        ims.append(Image.fromarray(obs))

        obs = {"image": image_obs, "proprio": proprio}
        

        rng, key = jax.random.split(rng)
        action = np.array(
            agent.sample_actions(
                obs,
                goal_obs,
                initial_obs,
                seed=key,
                modality="language",
                argmax=True,
            )
        )[0]
        action = unnormalize_action(action, action_mean, action_std)
        actions.append(action)
        proprio = state[i]
    all_actions = np.array(actions) 
    return all_actions, ims_to_eval

if __name__ == "__main__":
    
    agent, config = make_agent()
    eval_on_traj([""], [""], agent, config, 
                 "XXXX")
