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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from datetime import datetime
import jax
import time
import tensorflow as tf
import jax.numpy as jnp
from flax.core import FrozenDict

from jaxrl_m.vision.clip import process_image, process_text, create_from_checkpoint

from widowx_env import convert_obs, state_to_eep, wait_for_obs, WidowXGym
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs, WidowXStatus

from octo.utils.gym_wrappers import (
    HistoryWrapper,
    RHCWrapper,
    TemporalEnsembleWrapper,
)

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_path", None, "Path to checkpoint", required=True)
flags.DEFINE_string("wandb_run_name", None, "Name of wandb run", required=True)
flags.DEFINE_string("video_save_path", None, "Path to save video")
flags.DEFINE_string(
    "goal_image_path",
    None,
    "Path to a single goal image",
)
flags.DEFINE_integer("num_timesteps", 120, "num timesteps")
flags.DEFINE_bool("blocking", True, "Use the blocking controller")
flags.DEFINE_spaceseplist("goal_eep", None, "Goal position")
flags.DEFINE_spaceseplist("initial_eep", None, "Initial position")
flags.DEFINE_float("gripper_thresh", 0.5, "-")

flags.DEFINE_bool("high_res", False, "Save high-res video and goal")

STEP_DURATION = 0.2
INITIAL_STATE_I = 0  
FINAL_GOAL_I = -1  
NO_PITCH_ROLL = False
NO_YAW = False 
STICKY_GRIPPER_NUM_STEPS = 2

FIXED_STD = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

STEP_DURATION_MESSAGE = """
Bridge data was collected with non-blocking control and a step duration of 0.2s.
However, we relabel the actions to make it look like the data was collected with
blocking control and we evaluate with blocking control.
We also use a step duration of 0.4s to reduce the jerkiness of the policy.
Be sure to change the step duration back to 0.2 if evaluating with non-blocking control.
"""
STEP_DURATION = 0.2
STICKY_GRIPPER_NUM_STEPS = 2

WORKSPACE_BOUNDS = [[0.2, -0.13, 0.06, -1.57, 0], [0.33, 0.13, 0.25, 1.57, 0]] 
CAMERA_TOPICS = [{"name": "/blue/image_raw"}]
ENV_PARAMS = {
    "camera_topics": CAMERA_TOPICS,
    "override_workspace_boundaries": WORKSPACE_BOUNDS,
    "move_duration": STEP_DURATION,
}

def unnormalize_action(action, mean, std):
    return action * std + mean

def process_batch(batch):
    data_split = "bridgedata"
    if not type(batch) == FrozenDict:
        batch = FrozenDict(batch)
    lang_ids = batch["goals"]["language"]
    lang_mask = jnp.array(lang_ids >= 0)
    sents = ["placeholder" for x in lang_ids]
    

    
    
    lang_inputs = process_text(sents)

    obs_img = process_image(batch["observations"]["image"])
    goal_img = process_image(batch["goals"]["image"])
    init_img = process_image(batch["initial_obs"]["image"])

    pixel_input = jnp.concatenate([init_img, goal_img], axis=-1)
    clip_outputs = clip(pixel_values=pixel_input, **lang_inputs)
    text_embed = clip_outputs["text_embeds"]
    image_embed = clip_outputs["image_embeds"]

    add_or_replace = {
        "observations": {"image": obs_img, "unprocessed_image": obs_img},
        "goals": {
            "image": goal_img,
            "language": lang_inputs,
            "language_mask": lang_mask,
            
            
        },
        "initial_obs": {
            "image": init_img,
            "unprocessed_image": init_img,
        },
    }

    if (
        "use_image_embeds_as_inputs" in run.config
        and run.config["use_image_embeds_as_inputs"]
    ):
        add_or_replace["goals"]["image"] = image_embed
    elif run.config["use_image_embeddings"]:
        add_or_replace["goals"]["image_embed"] = image_embed

    if (
        "use_text_embeds_as_inputs" in run.config
        and run.config["use_text_embeds_as_inputs"]
    ):
        add_or_replace["goals"]["language"] = text_embed
    elif run.config["use_text_embeddings"]:
        add_or_replace["goals"]["text_embed"] = text_embed

    add_or_replace["actions"] = batch["actions"]
    add_or_replace["bc_mask"] = jnp.ones(batch["observations"]["image"].shape[0])

    return batch.copy(add_or_replace=add_or_replace)

clip = None
run = None

def main(_):
    assert tf.io.gfile.exists(FLAGS.checkpoint_path)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    if FLAGS.initial_eep is not None:
        assert isinstance(FLAGS.initial_eep, list)
        initial_eep = [float(e) for e in FLAGS.initial_eep]
        start_state = np.concatenate([initial_eep, [0, 0, 0, 1]])
    else:
        start_state = None

    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update(ENV_PARAMS)
    env_params["start_state"] = list(start_state)
    widowx_client = WidowXClient(host=FLAGS.ip, port=FLAGS.port)
    widowx_client.init(env_params, image_size=FLAGS.im_size)
    env = WidowXGym(
        widowx_client, FLAGS.im_size, FLAGS.blocking, STICKY_GRIPPER_NUM_STEPS
    )

    if not FLAGS.blocking:
        assert STEP_DURATION == 0.2, STEP_DURATION_MESSAGE

    env = HistoryWrapper(env, FLAGS.horizon)
    env = TemporalEnsembleWrapper(env, FLAGS.pred_horizon)

    
    if FLAGS.goal_image_path is not None:
        image_goal = np.array(Image.open(FLAGS.goal_image_path))
    else:
        image_goal = None

    
    api = wandb.Api()
    global run
    run = api.run(FLAGS.wandb_run_name)

    
    encoder_def = encoders[run.config["encoder"]](**run.config["encoder_kwargs"])
    task_encoder_defs = {
        k: encoders[run.config["task_encoders"][k]](
            **run.config["task_encoder_kwargs"][k]
        )
        for k in ("image", "language")
        if k in run.config["task_encoders"]
    }

    global clip
    clip_path = None
    clip_local_path = os.path.join("/tmp/cache/", "/".join(clip_path.split("/")[3:]))

    if not tf.io.gfile.exists(clip_local_path):
        tf.io.gfile.makedirs(clip_local_path)
        tf.io.gfile.copy(os.path.join(clip_path, 'checkpoint'), os.path.join(clip_local_path, 'checkpoint'))

    clip = create_from_checkpoint(
        clip_local_path,
    )

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

    
    action_metadata = run.config["bridgedata_config"]["action_metadata"]
    action_mean = np.array(action_metadata["mean"])
    action_std = np.array(action_metadata["std"])

    checkpoint_local_path = os.path.join(
        "/tmp/cache/", "/".join(FLAGS.checkpoint_path.split("/")[3:])
    )

    if not tf.io.gfile.exists(checkpoint_local_path):
        tf.io.gfile.makedirs(checkpoint_local_path)
        tf.io.gfile.copy(os.path.join(FLAGS.checkpoint_path, 'checkpoint'), os.path.join(checkpoint_local_path, 'checkpoint'))

    
    agent = checkpoints.restore_checkpoint(checkpoint_local_path, agent)

    
    while True:
        
        prompt = input("new goal?")
        if prompt == "":
            if FLAGS.goal_eep is not None:
                assert isinstance(FLAGS.goal_eep, list)
                goal_eep = [float(e) for e in FLAGS.goal_eep]
            else:
                low_bound = [0.24, -0.1, 0.05, -1.57, 0]
                high_bound = [0.4, 0.20, 0.15, 1.57, 0]
                goal_eep = np.random.uniform(low_bound[:3], high_bound[:3])
            _eep = [float(e) for e in goal_eep]
            goal_eep = state_to_eep(_eep, 0)
            widowx_client.move_gripper(1.0)  
            move_status = None
            while move_status != WidowXStatus.SUCCESS:
                move_status = widowx_client.move(goal_eep, duration=1.5)
            input("Press [Enter] when ready for taking the goal image. ")
            obs = wait_for_obs(widowx_client)
            obs = convert_obs(obs, FLAGS.im_size)

            image_goal = obs["image_primary"]

        input("start?")

        
        
        
        
        
        
        
        
        
        obs, _ = env.reset()
        goal_obs = {}
        if not prompt:
            goal_obs.update({"image": image_goal[None, ...]})
        else:
            lang_inputs = process_text(prompt)
            text_embed = clip(
                pixel_values=jnp.zeros((1, 224, 224, 6)), **lang_inputs
            ).text_embeds

            if run.config["use_text_embeds_as_inputs"]:
                goal_obs.update(
                    {
                        "language": text_embed,
                        "language_mask": jnp.ones((1, 1)),
                    }
                )
            elif run.config["use_text_embeddings"]:
                goal_obs.update(
                    {
                        "language": lang_inputs,  
                        "language_mask": jnp.ones((1, 1)),
                        "text_embed": text_embed,
                    }
                )
            else:
                goal_obs.update(
                    {
                        "language": lang_inputs,
                        "language_mask": jnp.ones((1, 1)),
                    }
                )
        modality = list(goal_obs)[0]
        if modality == "language_mask":
            modality = "language"
        
        
        obs = env._get_obs()
        last_tstep = time.time()
        images = []
        full_images = []
        t = 0
        
        is_gripper_closed = False
        num_consecutive_gripper_change_actions = 0
        try:
            initial_image_obs = (
                obs["image"].reshape(3, 224, 224).transpose(1, 2, 0) * 255
            ).astype(np.uint8)
            if "image" in goal_obs:
                pixel_input = np.concatenate(
                    (
                        process_image(initial_image_obs),
                        process_image(goal_obs["image"]),
                    ),
                    axis=-1,
                )
                image_embed = clip(
                    pixel_values=pixel_input, **process_text("placeholder")
                ).image_embeds
                if run.config["use_image_embeddings"]:
                    goal_obs["image_embed"] = image_embed
                elif run.config["use_image_embeds_as_inputs"]:
                    goal_obs["image"] = image_embed
                elif run.config["dataset_kwargs"]["clip_preprocessing"]:
                    goal_obs["image"] = process_image(goal_obs["image"])

            if run.config["dataset_kwargs"]["clip_preprocessing"]:
                initial_obs = {
                    "image": process_image(initial_image_obs),
                    "unprocessed_image": initial_image_obs[None, ...],
                    "proprio": obs["state"],
                }
            else:
                initial_obs = {
                    "image": initial_image_obs[None, ...],
                    "proprio": obs["state"],
                }
            while t < FLAGS.num_timesteps:
                if time.time() > last_tstep + STEP_DURATION or FLAGS.blocking:
                    image_obs = (
                        obs["image"].reshape(3, 224, 224).transpose(1, 2, 0) * 255
                    ).astype(np.uint8)

                    if FLAGS.high_res:
                        full_images.append(Image.fromarray(obs["full_image"][0]))
                    if run.config["dataset_kwargs"]["clip_preprocessing"]:
                        obs = {
                            "image": process_image(image_obs),
                            "unprocessed_image": image_obs[None, ...],
                            "proprio": obs["state"],
                        }
                    else:
                        obs = {"image": image_obs[None, ...], "proprio": obs["state"]}

                    last_tstep = time.time()

                    print(goal_obs.keys())

                    rng, key = jax.random.split(rng)
                    action = np.array(
                        agent.sample_actions(
                            obs,
                            goal_obs,
                            initial_obs,
                            seed=key,
                            modality=modality,
                            argmax=True,
                        )
                    )[0]
                    action = unnormalize_action(action, action_mean, action_std)
                    action += np.random.normal(0, FIXED_STD)

                    print(action[-1])
                    
                    if (action[-1] < FLAGS.gripper_thresh) != is_gripper_closed:
                        num_consecutive_gripper_change_actions += 1
                    else:
                        num_consecutive_gripper_change_actions = 0

                    if (
                        num_consecutive_gripper_change_actions
                        >= STICKY_GRIPPER_NUM_STEPS
                    ):
                        is_gripper_closed = not is_gripper_closed
                        num_consecutive_gripper_change_actions = 0

                    action[-1] = 0.0 if is_gripper_closed else 1.0

                    
                    if NO_PITCH_ROLL:
                        action[3] = 0
                        action[4] = 0
                    if NO_YAW:
                        action[5] = 0

                    
                    obs, rew, done, info = env.step(
                        action, last_tstep + STEP_DURATION, blocking=FLAGS.blocking
                    )

                    
                    if image_goal is None:
                        image_formatted = image_obs
                    else:
                        image_formatted = np.concatenate(
                            (image_goal, image_obs), axis=0
                        )
                    images.append(Image.fromarray(image_formatted))

                    t += 1
        except Exception as e:
            print(traceback.format_exc(), file=sys.stderr)

        
        if FLAGS.video_save_path is not None:
            os.makedirs(FLAGS.video_save_path, exist_ok=True)
            checkpoint_name = "_".join(FLAGS.checkpoint_path.split("/")[-2:])
            prompt = prompt.replace(" ", "-")
            print(prompt)
            save_path = os.path.join(
                FLAGS.video_save_path,
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{checkpoint_name}_{prompt}.gif",
            )
            print(f"Saving Video at {save_path}")
            images[0].save(
                save_path,
                format="GIF",
                append_images=images[1:],
                save_all=True,
                duration=200,
                loop=0,
            )

            if FLAGS.high_res:
                base_path = os.path.join(FLAGS.video_save_path, "high_res")
                os.makedirs(base_path, exist_ok=True)
                print(f"Saving Video and Goal at {base_path}")
                curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                video_path = os.path.join(base_path, f"{curr_time}_{checkpoint_name}_{prompt}.gif")
                full_images[0].save(
                    video_path,
                    format="GIF",
                    append_images=full_images[1:],
                    save_all=True,
                    duration=200,
                    loop=0,
                )

if __name__ == "__main__":
    app.run(main)
