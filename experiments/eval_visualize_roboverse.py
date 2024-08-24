from functools import partial
from absl import app, flags
from jaxrl_m.agents.continuous.gc_iql import create_iql_learner
from jaxrl_m.vision import encoders as vision_encoders
from jaxrl_m.envs.bridge import visualization
from flax.training import checkpoints
import numpy as np
import wandb
import matplotlib.pyplot as plt
import os
import gym
import roboverse
import glob
import h5py
from collections import defaultdict

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "checkpoint_path", None, "Path to checkpoint to resume from.", required=True
)
flags.DEFINE_string(
    "wandb_run_name", None, "Name of wandb run to resume from.", required=True
)
flags.DEFINE_string("data_path", None, "Location of dataset", required=True)
flags.DEFINE_integer("demo_id", None, "ID of demo to visualize.", required=True)

def get_demo():
    paths = glob.glob(f"{FLAGS.data_path}/train/*.hdf5")
    path = paths[0]
    specs = defaultdict(list)
    with h5py.File(path, "r") as f:
        file_len = len(f["actions"])
        start = FLAGS.demo_id - (49 - f["steps_remaining"][FLAGS.demo_id])
        end = FLAGS.demo_id + f["steps_remaining"][FLAGS.demo_id] + 1
        action_metadata = np.load(
            os.path.join(FLAGS.data_path, "train/metadata.npy"), allow_pickle=True
        ).item()
        actions = (f["actions"][start:end] - action_metadata["mean"]) / action_metadata[
            "std"
        ]
        demo_batched = {
            "observations": {
                "image": f["observations/images0"][start:end],
                "proprio": f["observations/state"][start:end].astype(np.float32),
            },
            "next_observations": {
                "image": f["next_observations/images0"][start:end],
                "proprio": f["next_observations/state"][start:end].astype(np.float32),
            },
            "actions": actions,
            "goals": {
                "image": np.array(
                    [f["observations/images0"][end - 1] for _ in range(len(actions))]
                ),
            },
            "rewards": np.array(
                [0 if i == len(actions) - 1 else -1 for i in range(len(actions))]
            ),
            "masks": np.ones(len(actions)),
        }

        return demo_batched

def main(_):
    
    api = wandb.Api()
    run = api.run(FLAGS.wandb_run_name)

    assert run.config["model_constructor"] == "create_iql_learner"
    model_config = run.config["model_config"]
    encoder_def = vision_encoders[model_config["encoder"]](
        **model_config["encoder_kwargs"]
    )
    agent = create_iql_learner(
        seed=0,
        encoder_def=encoder_def,
        observations=np.zeros((128, 128, 3), dtype=np.uint8),
        goals=np.zeros((128, 128, 3), dtype=np.uint8),
        actions=np.zeros(7, dtype=np.float32),
        **model_config["agent_kwargs"],
    )
    params = checkpoints.restore_checkpoint(FLAGS.checkpoint_path, target=None)[
        "model"
    ]["params"]
    agent = agent.replace(model=agent.model.replace(params=params))

    demo_batched = get_demo()

    
    metrics = agent.get_debug_metrics(demo_batched)

    
    what_to_visualize = [
        partial(visualization.visualize_metric, metric_name="mse"),
        partial(visualization.visualize_metric, metric_name="v"),
        partial(visualization.visualize_metric, metric_name="target_q"),
        partial(visualization.visualize_metric, metric_name="advantage"),
        partial(visualization.visualize_metric, metric_name="value_err"),
        partial(visualization.visualize_metric, metric_name="td_err"),
    ]
    image = visualization.make_visual(
        demo_batched["observations"]["image"],
        metrics,
        what_to_visualize=what_to_visualize,
    )
    plt.imsave("visualization.png", image)

if __name__ == "__main__":
    app.run(main)
