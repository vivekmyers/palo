
from __future__ import annotations
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
from jaxrl_m.common.wandb import WandBLogger
import matplotlib
from absl import app, flags, logging
import time
from datetime import datetime
import jax
import time
import tensorflow as tf
import jax.numpy as jnp
from flax.core import FrozenDict
import pickle
import eval_on_traj
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

def make_init_config():
    return dict(init_temp=1, 
                final_temp=0.1,
                decay=0.995,
                t=250,
                max_steps = 1500,
                path_root="XXXX",)

def process_actions_gfile(path):  
    fp = tf.io.gfile.join(path, "policy_out.pkl")
    with tf.io.gfile.GFile(fp, "rb") as f:
        act_list = pickle.load(f)
    if isinstance(act_list[0], dict):
        act_list = [x["actions"] for x in act_list]
    return act_list

def process_state_gfile(path):
    fp = tf.io.gfile.join(path, "obs_dict.pkl")
    with tf.io.gfile.GFile(fp, "rb") as f:
        x = pickle.load(f)
    return x["full_state"][:-1], x["full_state"][1:]

def process_actions(path):  
    fp = os.path.join(path, "policy_out.pkl")
    with open(fp, "rb") as f:
        act_list = pickle.load(f)
    if isinstance(act_list[0], dict):
        act_list = [x["actions"] for x in act_list]
    return act_list

def process_state(path):
    fp = os.path.join(path, "obs_dict.pkl")
    with open(fp, "rb") as f:
        x = pickle.load(f)
    return x["full_state"][:-1], x["full_state"][1:]

@jax.jit
def replace(old_action: jax.Array, ind: jax.Array, new_action: jax.Array) -> jax.Array:
    
    
    
    if len(ind.shape) > 1:
        ind = ind.flatten()
    new_arr = old_action.copy()
    new_arr = new_arr.at[ind].set(new_action)
    return new_arr

@jax.jit
def replace_batched(old_action, new_action, ind):
    foo = jax.vmap(replace, in_axes=(0, 0, 0))
    return foo(old_action, ind, new_action)

@jax.jit
def perturb(key, mx:int, t: jnp.ndarray):
    MIN, MAX = 0, mx-1
    t += jax.random.choice(key, a=jnp.array([-1, 0, 1]), shape=t.shape)
    t = jnp.clip(t, MIN, MAX)
    t = t.at[0].set(0) 
    return jnp.sort(t), jax.random.split(key)[0]


def change_curr(obj: Low_Level_Optimizer, energy: float, ind: int, new_switch: jnp.ndarray, new_action: jnp.ndarray, previous_aciton: jnp.ndarray, prev_cost: float):
    obj.current_energy = energy
    obj.current_switch[ind] = new_switch
    obj.current_high_level_switch[ind] = obj.new_switch[obj.h_ind]
    return new_action


def best_state(obj: Low_Level_Optimizer, energy: float, ind: int, new_switch: jnp.ndarray, new_action: jnp.ndarray, previous_action: jnp.ndarray, prev_cost: float):
    obj.best_energy = energy
    obj.switch[ind] = new_switch
    obj.high_level_switch[ind] = new_switch[obj.h_ind]
    return previous_action


def default(**kwargs) -> jnp.ndarray:
    return kwargs.get("previous_action")


def best_state_case(obj: Low_Level_Optimizer, energy: float, ind: int, new_switch: jnp.ndarray, new_action: jnp.ndarray, previous_action: jnp.ndarray, prev_cost: float):

    return jax.lax.cond(energy < prev_cost and energy < obj.best_energy, best_state, default, obj, energy, ind, new_switch, new_action, previous_action, prev_cost)


def optim_wrapper(obj: Low_Level_Optimizer, energy: float, ind: int, new_switch: jnp.ndarray, new_action: jnp.ndarray, previous_action: jnp.ndarray, rv: float, prev_cost: float):
    return jax.lax.cond(energy > rv, change_curr, best_state_case, obj, energy, ind, new_switch, new_action, previous_action, prev_cost)

class BaseOptimizer:
    
    def __init__(self, config):
        self.init_temp = config["init_temp"]
        self.tmp = self.init_temp
        self.final_temp = config["final_temp"]
        self.decay = config["decay"]
        self.max_steps = config["max_steps"]
        self.act_len = config['t'] 
        self.path_root = config["path_root"]
        self.paths = [self.path_root + i for i in tf.io.gfile.listdir(self.path_root)[:5]] 
        self.key = jax.random.PRNGKey(193)
        self.counter = 0
    
    def add_padding(self, data: list) -> jax.Array: 
        f = lambda action: jnp.concatenate((action, jnp.zeros((self.act_len-action.shape[0], 7))), axis=0) 
        action_oracle = jax.tree_util.tree_map(f, data) 
        return jnp.array(action_oracle)
    
    
    
    def perturb(self, mx:int, t: jnp.ndarray):
        MIN, MAX = 0, mx-1
        t += jax.random.choice(self.key, a=jnp.array([-1, 0, 1]), shape=t.shape)
        t = jnp.clip(t, MIN, MAX)
        t = t.at[0].set(0) 
        return jnp.sort(t),  
    
    def perturb_batched(self, t: jax.Array):
        foo = jax.vmap(self.perturb, in_axes=0)
        return foo(t)
    
    def get_index_to_change(self, old_t: jnp.ndarray, new_t: jnp.ndarray):
        all_t = jnp.concatenate((old_t, new_t))
        sorted_t = jnp.sort(all_t)
        return sorted_t[jnp.concatenate((jnp.array([True]), sorted_t[1:] != sorted_t[:-1]))]
    
    def get_index_to_change_batch(self, old_t: jnp.ndarray, new_t: jnp.ndarray):
        foo = jax.vmap(self.get_index_to_change, in_axes=(0, 0))
        return foo(old_t, new_t)
    
    def modify(self): 
        pass
    
    def modify_batched(self):
        pass
    

class Low_Level_Optimizer(BaseOptimizer):
    

    def __init__(self, instr_h: list[str], instr_l: list[str], batched: bool, index: list):
        
        config = make_init_config()
        self.h_ind = jnp.array(index)
        super().__init__(config)
        self.action_oracle = [jnp.array(process_actions_gfile(path)) for path in self.paths]
        self.action_oracle_lens = [i.shape[0] for i in self.action_oracle]
        self.action_oracle_padded = self.add_padding(self.action_oracle) 
        self.all_inds = jnp.arange(len(self.action_oracle_lens))
        self.states = [jnp.array(process_state_gfile(path)) for path in self.paths]
        self.instr_h = instr_h
        self.instr_l = instr_l
        self.best_energy = jnp.inf
        
        self.switch: list[jnp.ndarray] = [jnp.array([int(self.action_oracle_lens[v] * i / len(instr_l))
                    for i in range(len(instr_l))]) for v in range(len(self.action_oracle))] 
        self.high_level_switch = [i[self.h_ind] for i in self.switch]
        self.current_switch = [i.copy() for i in self.switch]
        self.current_high_level_switch = [i.copy() for i in self.current_switch]
        self.batched = batched
        self.current_energy = jnp.inf

        self.agent, self.config = eval_on_traj.make_agent()

        self.instr_h_embed = jax.tree_util.tree_map(lambda x: multi_embed(x), instr_h)
        self.instr_l_embed = jax.tree_util.tree_map(lambda x: multi_embed(x), instr_l)
        

    def modify(self, ind: jnp.ndarray):
        t_p, self.key = perturb(self.key, self.action_oracle_lens[ind], self.current_switch[ind].copy())
        
        index_to_change = self.get_index_to_change(self.current_switch[ind], t_p)
        return t_p, index_to_change
    
    def modify_batched(self):
        foobar = jax.tree_util.tree_map(self.modify, self.all_inds)
        
        return foobar 
    
    def __cost(self, actions: jnp.ndarray, ind: int):
        assert actions.ndim == 2, f"action should be in the form of (num_act, dim_acts) but has shape {actions.shape}"
        actions_padded = self.add_padding(actions)
        actions_oracle = self.action_oracle_padded[ind]
        return jnp.sum(jnp.square(actions_padded - actions_oracle))
    
    
    
    
    

    def __energy(self, action, new_action, ind):
        assert action.ndim == 2, f"action should be in the form of (num_act, dim_acts) but has shape {action.shape}"
        assert new_action.shape == action.shape, f"New action has shape {new_action.shape}, but action has shape {action.shape}"
        prev_energy = self.__cost(action, ind)
        new_energy = self.__cost(new_action, ind)
        return jnp.exp((prev_energy - new_energy)/self.tmp), prev_energy, new_energy
    
    
    def optimize_one_step(self, prev_action: jnp.ndarray, new_action: jnp.ndarray, new_switch: jnp.ndarray, ind: int):
        
        energy, prev_cost, new_cost = self.__energy(prev_action, new_action, ind)
        
        
        
        
        x = jax.random.uniform(self.key)
        self.key, _ = jax.random.split(self.key)
        if energy > x:
            
            self.current_energy = energy
            self.current_switch[ind] = new_switch
            self.current_high_level_switch[ind] = new_switch[self.h_ind]
            return new_action
        if new_cost < prev_cost and new_cost < self.best_energy:
            
            self.best_energy = new_cost
            self.switch[ind] = new_switch
            self.high_level_switch[ind] = new_switch[self.h_ind]
            pass
        if not self.batched: 
            self.tmp *= self.decay
            self.counter += 1
        return prev_action

    
    
    
    
    
    
    def optimize_one_step_batched(self, prev_action: jnp.ndarray, new_action: jnp.ndarray, new_switch: jnp.ndarray):
        
        all_ind = jnp.arange(len(self.action_oracle_lens))
        foo = jax.vmap(self.optimize_one_step, in_axes=(0, 0, 0, 0))
        foo(prev_action, new_action, new_switch, all_ind)
        if self.batched: 
            self.tmp *= self.decay
            self.counter += 1

    def run_optim(self, ind: int):
        
        s, b, e, t = [], [], [], []
        time_start = time.time()
        actions = jnp.zeros_like(self.action_oracle_padded[ind])
        while self.tmp >= self.final_temp and self.counter <= self.max_steps:
            t_p, ind_to_change = self.modify(ind)
            
            plan_l = [(t_p[i], self.instr_l_embed[i]) for i in range(len(self.instr_l))]
            plan_h = [(t_p[self.h_ind[j]], self.instr_h_embed[j]) for j in range(len(self.instr_h))]
            
            inds = ind_to_change if self.counter > 0 else jnp.arange(self.action_oracle_lens[ind])
            new_actions, _ = eval_on_traj.eval_on_traj(plan_h, plan_l, self.agent, self.config, self.paths[ind], inds)
            actions_hat = replace(actions, inds, new_actions)
            actions = self.optimize_one_step(actions, actions_hat, t_p, ind)
            if not self.counter % 100:
                s.append(self.counter)
                b.append(self.current_energy)
                e.append(self.best_energy)
                t.append(self.tmp)
                print(f"Step {self.counter} done!")

            self.key, _ = jax.random.split(self.key)

        time_end = time.time()

        print("Time for optimization to finish (s): ", time_end-time_start)
        return s,b,e,t
    
    def run_optim_batched(self):
        
        foo = jax.vmap(self.run_optim, in_axes=0)
        s, b, e, t = foo(self.all_inds)
        return s, b, e, t
    

class Low_Level_Optimizer_Batched(Low_Level_Optimizer):
    
    def __init__(self, instr_h: list[list[str]], instr_l: list[list[str]], batched : bool = True):
        super().__init__(instr_h, instr_l, batched)
        self.instr_h = instr_h
        self.instr_l = instr_l

    def optim_language_batched(self):
        foo = jax.vmap()


class High_Level_Optimzier(Low_Level_Optimizer):
    def __init__(self, instr: list, paths: list[str]):
        config = make_init_config()
        super().__init__(config)

class Joint_Optimizer(High_Level_Optimzier):
    def __init__(self, instr: list, paths: list[str]):
        config = make_init_config()
        super().__init__(instr, config)


class SAModule:
    def __init__(self) -> None:
        return
    


if __name__ =="__main__":
    low = [
        "move the gripper right towards the mushroom",
        "move the gripper down towards the mushroom",
        "close the gripper on the mushroom",
        "move the gripper up",
        "move the gripper left towards the pot",
        "move the gripper down towards the pot",
        "open the gripper on the pot",
        "move the gripper up",
        "move the gripper right towards the spoon",
        "move the gripper down towards the spoon",
        "close the gripper on the spoon",
        "move the gripper up",
        "move the gripper right and forward towards the golden confetti",
        "move the gripper down towards the golden confetti",
        "move the gripper right towards the golden confetti while holding the spoon",
        "open the gripper"
    ]
    opt = Low_Level_Optimizer(["put the mushroom in the pot", "use the spoon to sweep away the golden confetti"], low, False, [0, 8])
    path = None
    
    s, b, e, t = opt.run_optim(0)
    print(s)
    print(b)
    print(e)
    
    print(opt.switch[0])
