import os
from jaxrl_m.common.evaluation import supply_rng, evaluate_gc
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.envs.wrappers.roboverse import GCRoboverseWrapper
from jaxrl_m.envs.wrappers.action_norm import UnnormalizeAction
from jaxrl_m.envs.wrappers.video_recorder import VideoRecorder
from jaxrl_m.vision import encoders as vision_encoders
from jaxrl_m.agents.continuous.gc_bc import GCBCAgent, GCActor, GCAdaptor, Policy
from jaxrl_m.common.common import TrainState

import gym
import tqdm
import glob
from absl import app, flags
import numpy as np
from flax.training import checkpoints
import roboverse
import wandb
from functools import partial

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_dir", "./log/", "Dir. with checkpoints")
flags.DEFINE_string("wandb_run", "run", "Name of wandb run to get config")
flags.DEFINE_string("data_path", None, "Location of dataset", required=True)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("eval_interval", 1000, "Eval interval.")
flags.DEFINE_integer("max_steps", int(5e5), "Number of steps.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("deterministic_eval", True, "Take mode of action dist. for eval")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_string("save_dir", "./log/", "Video/buffer logging dir.")

def wrap(env: gym.Env, action_metadata: dict):
    eval_goals = np.load(
        os.path.join(FLAGS.data_path, "val/eval_goals.npy"), allow_pickle=True
    ).item()
    env = GCRoboverseWrapper(env, eval_goals)
    env = UnnormalizeAction(env, action_metadata)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=45)
    return env

def main(_):
    api = wandb.Api()
    run = api.run(FLAGS.wandb_run)
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": "jaxrl_m_roboverse",
            "exp_prefix": "gc_roboverse_offline",
            "exp_descriptor": f"{run.config['env_name']}",
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=run.config["model_config"],
    )

    FLAGS.save_dir = os.path.join(
        FLAGS.save_dir,
        wandb_logger.config.project,
        wandb_logger.config.exp_prefix,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
    )

    action_metadata = np.load(
        os.path.join(FLAGS.data_path, "train/metadata.npy"), allow_pickle=True
    ).item()

    eval_env = roboverse.make(run.config["env_name"], transpose_image=False)
    eval_env = wrap(eval_env, action_metadata)
    if FLAGS.save_video:
        eval_env = VideoRecorder(
            eval_env, os.path.join(FLAGS.save_dir, "videos"), goal_conditioned=True
        )
    eval_env.reset(seed=FLAGS.seed + 42)

    encoder_def = vision_encoders[run.config["encoder"]](
        **run.config["model_config"]["encoder_kwargs"]
    )

    encoders = {"actor": encoder_def}
    goal_encoders = encoders

    model_def = GCActor(
        encoders=encoders,
        goal_encoders=goal_encoders,
        networks={
            "actor": GCAdaptor(
                Policy(
                    action_dim=7,
                    **run.config["model_config"]["agent_kwargs"]["actor_kwargs"],
                )
            )
        },
    )

    for i in tqdm.tqdm(
        range(0, FLAGS.max_steps, FLAGS.eval_interval),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
    ):
        agent = GCBCAgent(TrainState.create(model_def, None, None))
        agent = checkpoints.restore_checkpoint(
            os.path.join(FLAGS.checkpoint_dir, f"checkpoint_{i}"), target=agent
        )

        policy_fn = supply_rng(
            partial(agent.sample_actions, argmax=FLAGS.deterministic_eval)
        )
        eval_info = evaluate_gc(policy_fn, eval_env, num_episodes=FLAGS.eval_episodes)
        for k, v in eval_info.items():
            wandb_logger.log({f"evaluation/{k}": v}, step=i)

if __name__ == "__main__":
    app.run(main)
