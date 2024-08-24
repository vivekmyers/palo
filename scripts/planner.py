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
from __future__ import annotations
import eval_on_traj
import optimizer
import utils



class Planner:
    
    def __init__(self, instr) -> None:
        self.config = optimizer.make_init_config()
        self.instr = instr
        self.optim = optimizer.Low_Level_Optimizer(self.instr)
        self.agent = eval_on_traj.make_agent()


    def make_plan(self, obs):
        plans = utils.query_long_horizon(obs, self.instr)
        return plans
