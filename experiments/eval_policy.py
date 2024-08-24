import sys
from widowx_envs.widowx.widowx_env import BridgeDataRailRLPrivateWidowX
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
from widowx_envs.utils.multicam_server_rospkg.src.topic_utils import IMTopic
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

from jaxrl_m.vision.clip import process_image, process_text

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

flags.DEFINE_bool("high_res", False, "Save high-res video and goal")

STEP_DURATION = 0.2
INITIAL_STATE_I = 0  
FINAL_GOAL_I = -1  
NO_PITCH_ROLL = False
NO_YAW = False
STICKY_GRIPPER_NUM_STEPS = 2

FIXED_STD = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

def unnormalize_action(action, mean, std):
    return action * std + mean

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

def main(_):
    assert tf.io.gfile.exists(FLAGS.checkpoint_path)

    
    env_params = {
        "fix_zangle": 0.1,
        "move_duration": 0.2,
        "adaptive_wait": True,
        "move_to_rand_start_freq": 1,
        "override_workspace_boundaries": [
            [0.23, -0.11, 0, -1.57, 0],
            [0.44, 0.23, 0.18, 1.57, 0],
        ],
        "action_clipping": "xyz",
        "catch_environment_except": False,
        "start_state": None,
        "camera_topics": [IMTopic("/blue/image_raw", flip=False)],
        "return_full_image": FLAGS.high_res,
    }
    env = BridgeDataRailRLPrivateWidowX(env_params, fixed_image_size=224)

    
    if FLAGS.goal_image_path is not None:
        image_goal = np.array(Image.open(FLAGS.goal_image_path))
    else:
        image_goal = None

    
    api = wandb.Api()
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

    
    agent = checkpoints.restore_checkpoint(FLAGS.checkpoint_path, agent)

    
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
            env._controller.open_gripper(True)
            env._controller.move_to_state(goal_eep, 0, duration=1.5)
            env._reset_previous_qpos()
            try:
                env._reset_previous_qpos()
            except Exception as e:
                continue
            input("take image?")
            obs = env._get_obs()
            image_goal = (
                obs["image"].reshape(3, 224, 224).transpose(1, 2, 0) * 255
            ).astype(np.uint8)

        input("start?")
        try:
            env.reset()
            env.start()
        except Exception as e:
            continue

        
        if FLAGS.initial_eep is not None:
            assert isinstance(FLAGS.initial_eep, list)
            initial_eep = [float(e) for e in FLAGS.initial_eep]
            try:
                env._controller.move_to_state(initial_eep, 0, duration=1.5)
                env._reset_previous_qpos()
            except Exception as e:
                continue

        goal_obs = {}
        if not prompt:
            goal_obs.update({"image": image_goal})
        else:
            lang_inputs = multi_embed(prompt)
            goal_obs.update({"language": lang_inputs, "language_mask": jnp.ones(1)})
        modality = list(goal_obs)[0]
        if modality == "language_mask":
            modality = "language"

        
        obs = env._get_obs()
        last_tstep = time.time()
        images = []
        full_images=[]
        t = 0
        
        is_gripper_closed = False
        num_consecutive_gripper_change_actions = 0
        try:
            initial_image_obs = (
                obs["image"].reshape(3, 224, 224).transpose(1, 2, 0) * 255
            ).astype(np.uint8)
            initial_obs = {
                "image": initial_image_obs,
                "proprio": obs["state"],
            }
            while t < FLAGS.num_timesteps:
                if time.time() > last_tstep + STEP_DURATION or FLAGS.blocking:
                    image_obs = (
                        obs["image"].reshape(3, 224, 224).transpose(1, 2, 0) * 255
                    ).astype(np.uint8)
                    if FLAGS.high_res:
                        full_images.append(Image.fromarray(obs["full_image"][0]))
                    obs = {"image": image_obs, "proprio": obs["state"]}

                    last_tstep = time.time()

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
                    )
                    action = unnormalize_action(action, action_mean, action_std)
                    action += np.random.normal(0, FIXED_STD)

                    
                    if (action[-1] < 0.5) != is_gripper_closed:
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
