
from datetime import datetime
from functools import partial
import os
import time

from absl import app, flags, logging
import click
import cv2
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from widowx_env import convert_obs, state_to_eep, wait_for_obs, WidowXGym
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs, WidowXStatus
import pickle
import random
from PIL import Image
from flax.training import checkpoints
import traceback
import wandb
from jaxrl_m.vision import encoders
from jaxrl_m.agents import agents
from jaxrl_m.data.bridge_dataset import multi_embed
from jaxrl_m.data.language import load_mapping, lang_encode
import json
import utils

import matplotlib.pyplot as plt
import skimage.transform as tfr

from flax.core import FrozenDict

from jaxrl_m.vision.clip import process_image, process_text

from octo.utils.gym_wrappers import (
    HistoryWrapper,
    RHCWrapper,
    TemporalEnsembleWrapper,
)

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

import warnings
warnings.filterwarnings('ignore')

FLAGS = flags.FLAGS

flags.DEFINE_string("ip", "localhost", "IP address of the robot")
flags.DEFINE_integer("port", 5556, "Port of the robot")
flags.DEFINE_spaceseplist("goal_eep", [0.3, 0.0, 0.15], "Goal position")
flags.DEFINE_spaceseplist("initial_eep", [0.3, 0.0, 0.15], "Initial position")
flags.DEFINE_bool("blocking", True, "Use the blocking controller")

flags.DEFINE_bool("add_jaxrlm_baseline", False, "Also compare to jaxrl_m baseline")
flags.DEFINE_string(
    "modality",
    "l",
    "Either 'g', 'goal', 'l', 'language' (leave empty to prompt when running)",
)
flags.DEFINE_integer("im_size", 224, "Image size")
flags.DEFINE_string("video_save_path", None, "Path to save video")
flags.DEFINE_string("trajectory_save_path", None, "Path to save video")
flags.DEFINE_integer("primitive_interval", 8, "Number of steps per primitive")

flags.DEFINE_string("checkpoint_path", None, "Path to checkpoint", required=True)
flags.DEFINE_string("wandb_run_name", None, "Name of wandb run", required=True)

flags.DEFINE_integer("num_timesteps", 120, "num timesteps")
flags.DEFINE_integer("horizon", 1, "Observation history length")
flags.DEFINE_integer("pred_horizon", 1, "Length of action sequence from model")
flags.DEFINE_integer("exec_horizon", 1, "Length of action sequence to execute")
flags.DEFINE_float("temperature", 1.0, "Temperature for sampling actions")
flags.DEFINE_bool("deterministic", False, "Whether to sample action deterministically")
flags.DEFINE_bool("use_both", True, "Whether to sample action deterministically")
flags.DEFINE_list("items", ["gripper"], "The items you want to be included in grounding")

flags.DEFINE_bool("show_image", False, "Show image")

flags.DEFINE_bool("use_low_level", False, "Whether to use low level")

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

instr = [""]

def unnormalize_action(action, mean, std):
    return action * std + mean

def process_batch(batch):
    """
    A hacky method from GRIF, see if we can improve it.
    """
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

def main(_):
    print("The primitive interval is: ", FLAGS.primitive_interval)
    print("The items are: ", FLAGS.items)
    if FLAGS.use_low_level:
        low_lvl = True
    if FLAGS.use_both:
        use_both = True
    
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
    """
    new code
    """
    
    api = wandb.Api()
    print("wandb run name, ", FLAGS.wandb_run_name)
    run = api.run(FLAGS.wandb_run_name)

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

    
    action_metadata = run.config["bridgedata_config"]["action_metadata"]
    action_mean = np.array(action_metadata["mean"])
    action_std = np.array(action_metadata["std"])

    checkpoint_local_path = os.path.join(
        "/tmp/cache/", "/".join(FLAGS.checkpoint_path.split("/")[3:])
    )

    if not tf.io.gfile.exists(checkpoint_local_path):
        tf.io.gfile.makedirs(checkpoint_local_path)
        tf.io.gfile.copy(os.path.join(FLAGS.checkpoint_path, 'checkpoint'), os.path.join(checkpoint_local_path, 'checkpoint'))

    
    agent = checkpoints.restore_checkpoint(FLAGS.checkpoint_path, agent)

    """
    
    Cutoff
    
    """

    goal_image = jnp.zeros((FLAGS.im_size, FLAGS.im_size, 3), dtype=np.uint8)
    goal_instruction = ""

    modality = FLAGS.modality[:1]
    if modality not in ["g", "l", ""]:
        modality = ""
    
    while True:
        if modality == "language" or modality == "l":
            modality = "language_joint"

        traj = []
        prompt = input("new goal?")
        print(prompt)

        
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

        
        
        low_lvl_insts = []
        goal_obs = {}
        if not prompt:
            goal_obs.update({"image": image_goal})
        else:
            lang_inputs = multi_embed(prompt)
            goal_obs.update({"language": lang_inputs, "language_mask": jnp.ones(1)})
            obs = wait_for_obs(widowx_client)
        
        if modality == "language_joint":
            obs = wait_for_obs(widowx_client)
            
            goal_obs = {}
            if use_both:
                lang_inputs_high_level = multi_embed(prompt)
                lang_inputs_low_level = multi_embed("")
                lang_inputs = jnp.concatenate((lang_inputs_high_level, lang_inputs_low_level), axis=0)
            elif not low_lvl:
                lang_inputs_high_level = multi_embed(prompt)
                lang_inputs_low_level = jnp.zeros_like(lang_inputs_high_level)
                lang_inputs = jnp.concatenate((lang_inputs_high_level, lang_inputs_low_level), axis=0)
            else:
                lang_inputs_low_level = multi_embed(prompt)
                lang_inputs_high_level = jnp.zeros_like(lang_inputs_low_level)
                lang_inputs = jnp.concatenate((lang_inputs_high_level, lang_inputs_low_level), axis=0)
            goal_obs.update({"language": lang_inputs, "language_mask": jnp.ones(1)})

        modality = list(goal_obs)[0]
        if modality == "language_mask":
            modality = "language"

        
        full_images=[]
        

        input("Press [Enter] to start.")

        
        obs, _ = env.reset()

        full_images.append(env.raw_obs)

        if FLAGS.initial_eep is not None:
            assert isinstance(FLAGS.initial_eep, list)
            initial_eep = [float(e) for e in FLAGS.initial_eep]
            try:
                move_status = None
                while move_status != WidowXStatus.SUCCESS:
                    move_status = widowx_client.move(goal_eep, duration=1.5)
            except Exception as e:
                pass
        
        
        time.sleep(2.0)

        
        last_tstep = time.time()
        images = []
        goals = []
        t = 0
        initial_image_obs = obs["image_primary"][0]
        initial_obs = {
            "image": initial_image_obs,
            "proprio": obs["state"]
        }
        list_of_instrs = []
        try:
            while True:
                if time.time() > last_tstep + STEP_DURATION:
                    if use_both:
                        if t == 0:
                            print(env.raw_obs.shape, "shape of raw observation")
                            
                            
                            low_lvl_instr, im = utils.query_w_bbox(env.raw_obs, prompt, FLAGS.items) 
                            print(low_lvl_instr)
                            idx = low_lvl_instr.find("`python") + 7
                            instr_lst = eval(low_lvl_instr[idx:-3])
                            list_of_instrs.append(instr_lst[0])
                            lang_inputs_low_level = multi_embed(instr_lst[0])
                            lang_inputs = jnp.concatenate((lang_inputs_high_level, lang_inputs_low_level), axis=0)
                            goal_obs.update({"language": lang_inputs, "language_mask": jnp.ones(1)})

                    
                    image_obs = obs["image_primary"][-1]

                    last_tstep = time.time()

                    obs_x = {
                        "image": image_obs,
                        "proprio": obs["state"]
                    }

                    
                    images.append(obs["image_primary"][-1])
                    goals.append(goal_image)

                    if FLAGS.show_image:
                        bgr_img = cv2.cvtColor(obs["image_primary"][-1], cv2.COLOR_RGB2BGR)
                        cv2.imshow("img_view", bgr_img)
                        cv2.waitKey(20)

                    rng, key = jax.random.split(rng)
                    
                    forward_pass_time = time.time()
                    action = np.array(
                        agent.sample_actions(
                            obs_x,
                            goal_obs,
                            initial_obs,
                            seed=key,
                            modality=modality,
                            argmax=True,
                        )
                    )
                    
                    action = unnormalize_action(action, action_mean, action_std)
                    

                    print("forward pass time: ", time.time() - forward_pass_time)
                    traj.append(
                        dict(obs=obs, action=action, goal_instruction=goal_instruction, goal_image=goal_image)
                    )

                    
                    start_time = time.time()
                    obs, _, _, truncated, _ = env.step(action)
                    print("step time: ", time.time() - start_time)

                    t += 1
                    
                    

                    if not t % FLAGS.primitive_interval:
                        goal_obs = {}
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        try:
                            low_lvl_instr = instr_lst[t // FLAGS.primitive_interval]
                        except:
                            raise KeyboardInterrupt("Planning commands all done, defaulting to reset trajectory.")
                        print(low_lvl_instr)
                        list_of_instrs.append(low_lvl_instr)
                        
                        lang_inputs_low_level = multi_embed(low_lvl_instr)
                        lang_inputs = jnp.concatenate((lang_inputs_high_level, lang_inputs_low_level), axis=0)
                        goal_obs.update({"language": lang_inputs, "language_mask": jnp.ones(1)})
                        full_images.append(env.raw_obs)
                    
                    
        except KeyboardInterrupt:
            print(list_of_instrs)
            
            
            
            obs, _ = env.reset()
            full_images = [env.raw_obs]

            curr_time = datetime.now()
            policy_name = "lcbc_test"
            if FLAGS.trajectory_save_path is not None:
                save_path = os.path.join(FLAGS.trajectory_save_path, curr_time.strftime("%Y-%m-%d"), '{curr_time}_{policy_name}.pkl'.format(curr_time=curr_time.strftime("%Y-%m-%d_%H-%M-%S"), policy_name=policy_name))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'wb') as f:
                    pickle.dump(traj, f)
                print(f"Saved trajectory to {save_path}")
            traj = []
            
            if FLAGS.video_save_path is not None:
                os.makedirs(FLAGS.video_save_path, exist_ok=True)
                curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_path = os.path.join(
                    FLAGS.video_save_path,
                    f"{curr_time}_{prompt}.mp4",
                )
                video = np.concatenate([np.stack(goals), np.stack(images)], axis=1)
                imageio.mimsave(save_path, video, fps=1.0 / STEP_DURATION * 3)
                print(f"Saved video to {save_path}")

if __name__ == "__main__":
    
    
    app.run(main)
