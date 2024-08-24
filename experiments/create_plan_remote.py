
from absl import app, flags, logging
import os
import pickle
import time

import numpy as np
import cv2
import wandb
from PIL import Image
import jax
import jax.numpy as jnp
import tensorflow as tf
from flax.training import checkpoints
from flax.core import FrozenDict

from jaxrl_m.vision import encoders
from jaxrl_m.agents import agents

from jaxrl_m.data.bridge_dataset import multi_embed
from jaxrl_m.data.language import load_mapping
from jaxrl_m.common.common import shard_batch

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

flags.DEFINE_bool('show_image', False, 'Show image')

flags.DEFINE_bool('instruction_mode', None, 'Whether to sample action deterministically')
flags.DEFINE_string('instruction', None, 'High-level task instruction', required=True)
flags.DEFINE_string('trajectory_path', None, 
                    "Trajectories collected")
flags.DEFINE_string('checkpoint_path', None, "Checkpoint of the agent", required=True)
flags.DEFINE_string('wandb_run_name', None, 'Name of wandb run', required=True)
flags.DEFINE_integer('im_size', None, 'Name of wandb run', required=True)
flags.DEFINE_integer("max_subtask_steps", 18, "Maximum number of steps per subtask")
flags.DEFINE_integer("min_subtask_steps", 2, "Minimum number of steps per subtask")
flags.DEFINE_integer("max_episode_steps", 250, "Maximum number of steps in an episode")
flags.DEFINE_integer("num_plan_samples", 3, "Number of plans to be sampled")

def pad(arr, max_len=250):
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
    return x["full_state"][:-1], x["full_state"][1:]

def make_trajectories(path):
    trajs = [tf.io.gfile.join(path, i) for i in tf.io.gfile.listdir(path)]
    states = [process_state(i)[0] for i in trajs]
    actions = [pad(process_actions(i), FLAGS.max_episode_steps) for i in trajs]
    ims = [tf.io.gfile.join(i, "images0") for i in trajs]
    obs = [[squash(tf.io.decode_jpeg(tf.io.read_file(tf.io.gfile.join(p, f"im_{i}.jpg")), channels=3).numpy()) for i in range(len(tf.io.gfile.listdir(p)) - 1)] for p in ims] 
    return states, jnp.array(actions), obs

def propose_plans(obs, task_instr, num_samples):
    candidate_plans = []

    return candidate_plans

def sample_boundaries(episode_steps_in_demo, 
                     max_subtask_steps,
                     min_subtask_steps,
                     plans,
                     steps):
    
    each_subtask = []
    bb = [jnp.inf]
    
    
    for i in range(len(episode_steps_in_demo)): 
        while np.max(bb) >= episode_steps_in_demo[i]:
            db = np.random.randint(min_subtask_steps, max_subtask_steps + 1, len(plans)-1)
            bb = np.cumsum(np.concatenate((np.array([0]), db))) 

            
        bb = np.concatenate([bb, np.array([episode_steps_in_demo[i]])])
        print(bb)
        ls = np.concatenate([steps[i][j,bb[j]:bb[j+1],:] for j in range(len(bb)-1)])
        
        each_subtask.append(pad(ls))
    return jnp.array(each_subtask)

@jax.jit
def make_bb():
    return

@jax.jit
def sample_one_step(info: list):
    subtask, episode_steps_in_demo, max_subtask_steps, min_subtask_steps, plans, key = info
    
    return 

@jax.jit
def sample_boundaries_jax(episode_steps_in_demo, 
                     max_subtask_steps,
                     min_subtask_steps,
                     num_samples,
                     plans, 
                     key):
    each_subtask = jnp.array(jax.lax.fori_loop(0, num_samples, sample_one_step, [[], episode_steps_in_demo, max_subtask_steps, min_subtask_steps, plans, key]))[0]
    return each_subtask

def slice_actions(unrolled_actions, sampled_boundaries):

    all_predicted_actions = []

    for sample_id, boundaries in enumerate(sampled_boundaries):
        predicted_actions = np.zeros_like(unrolled_actions[0])

        num_subtasks = len(boundaries) - 1
        for subtask_id in range(num_subtasks):
            start = boundaries[subtask_id]
            end = boundaries[subtask_id + 1]
            if subtask_id == num_subtasks - 1:
                assert end == unrolled_actions.shape[1]
            predicted_actions[start:end] = (
                unrolled_actions[subtask_id][start:end])

        all_predicted_actions.append(predicted_actions)

    all_predicted_actions = np.stack(all_predicted_actions, axis=0)

    return all_predicted_actions

def unnormalize_action(action, mean, std):
    return action * std + mean

def process_batch(batch):
    """
    A hacky method from GRIF, see if we can improve it.
    """
    if not type(batch) == FrozenDict:
        batch = FrozenDict(batch)
    lang_ids = batch['goals']['language']
    lang_mask = jnp.array(lang_ids >= 0)
    lang_inputs = jnp.zeros((len(lang_ids), 512))

    obs_img = batch['observations']['image']
    goal_img = batch['goals']['image']
    init_img = batch['initial_obs']['image']

    add_or_replace = {
        'observations': {'image': obs_img},
        'goals': {
            'image': goal_img,
            'language': lang_inputs,
            'language_mask': lang_mask,
        },
        'initial_obs': {
            'image': init_img,
        },
    }

    add_or_replace['actions'] = batch['actions']
    add_or_replace['bc_mask'] = jnp.ones(
        batch['observations']['image'].shape[0])

    return batch.copy(add_or_replace=add_or_replace)

def initialize_agent(run, checkpoint_path):
    encoder_def = encoders[run.config['encoder']](
        **run.config['encoder_kwargs'])
    task_encoder_defs = {
        k: encoders[run.config['task_encoders'][k]](
            **run.config['task_encoder_kwargs'][k]
        )
        for k in ('image', 'language')
        if k in run.config['task_encoders']
    }

    load_mapping(run.config['data_path'])

    example_batch = {
        'observations': {'image': np.zeros((10, 224, 224, 3), dtype=np.uint8)},
        'initial_obs': {'image': np.zeros((10, 224, 224, 3), dtype=np.uint8)},
        'goals': {
            'image': np.zeros((10, 224, 224, 3), dtype=np.uint8),
            'language': np.zeros(10, dtype=np.int64),
            'language_mask': np.ones(10),
        },
        'actions': np.zeros((10, 7), dtype=np.float32),
    }

    example_batch = process_batch(example_batch)

    print('\n\nRun config kwargs: ', run.config['agent_kwargs'])
    
    rng = jax.random.PRNGKey(0)
    rng, construct_rng = jax.random.split(rng)
    agent = agents[run.config['agent']].create(
        rng=construct_rng,
        observations=example_batch['observations'],
        initial_obs=example_batch['initial_obs'],
        goals=example_batch['goals'],
        actions=example_batch['actions'],
        encoder_def=encoder_def,
        task_encoder_defs=task_encoder_defs,
        **run.config['agent_kwargs']
    )

    checkpoint_local_path = os.path.join(
        '/tmp/cache/', '/'.join(checkpoint_path.split('/')[3:])
    )

    if not tf.io.gfile.exists(checkpoint_local_path):
        tf.io.gfile.makedirs(checkpoint_local_path)
        tf.io.gfile.copy(
            os.path.join(checkpoint_path, 'checkpoint'), os.path.join(
                checkpoint_local_path, 'checkpoint'))

    
    agent = checkpoints.restore_checkpoint(checkpoint_path, agent)

    return agent, rng

def main(_):  
    t = time.time()
    devices = jax.local_devices()
    num_devices = len(devices)
    task_instr = FLAGS.instruction
    sharding = jax.sharding.PositionalSharding(devices)
    states, actions, observation = make_trajectories(FLAGS.trajectory_path) 
    
    
    api = wandb.Api()
    print('wandb run name, ', FLAGS.wandb_run_name)
    run = api.run(FLAGS.wandb_run_name)
    action_metadata = run.config['bridgedata_config']['action_metadata']
    action_mean = jnp.array(action_metadata['mean'])
    action_std = jnp.array(action_metadata['std'])

    policy, key = initialize_agent(run, FLAGS.checkpoint_path)  
    policy = jax.device_put(jax.tree_map(jnp.array, policy), sharding.replicate())
    goal_obs = {"language_mask": jnp.ones(1)}
    while True:
        
        candidate_plans = [["move the gripper right towards the mushroom",
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
        "open the gripper"], 
                           [
                            "Move the gripper forward and down towards the mushroom",
                            "close the gripper to pick up the mushroom",
                            "Move the gripper left and up towards the mushroom",
                            "Move the gripper left and forward towards the mushroom",
                            "Move the gripper down towards the mushroom",
                            "Open the gripper to release the mushroom",
                            "Move the gripper right towards the spoon",
                            "Move the gripper down towards the spoon",
                            "Close the gripper to pick up the spoon",
                            "Move the gripper right and down toward the edge of the table",
                            "Move the gripper right and down",
                            "Move the gripper right",
                            "Open the gripper",
                           ], 
                           [
                            "sdhjiosjo;nxl;",
                            "sdhsdhsd",
                            "shdshs",
                            "cxvmcvxm",
                            "vh,ghkgh,"
                           ],
                           [""]]

        candidate_plan_high_level = [

        ]
        
        candidate_plan_costs = np.zeros([len(candidate_plans)],
                                        dtype=float)
        initial_image_obs = [{"image":observation[i][0], "proprio":states[i][0]} for i in range(len(observation))]
        for plan_id, candidate_plan in enumerate(candidate_plans):
            
            unrolled_actions = []
            
            
            for x in range(len(observation)): 
                acts = []
                for y in range(len(candidate_plan)):
                    im_obs = jnp.array(observation[x]) 
                    proprio = jnp.array(states[x]) 
                    obs = {
                        "image": shard_batch(im_obs, sharding.replicate()),
                        "proprio": shard_batch(proprio, sharding.replicate())
                    }
                    lang_l = multi_embed(candidate_plan[y])
                    lang_h = jnp.zeros_like(lang_l)
                    lang = shard_batch(jnp.concatenate((lang_h, lang_h), axis=0), sharding.replicate())
                    goal_obs.update({"language": lang})
                    sample = lambda obs, rpg: policy.sample_actions(
                        obs, 
                        goal_obs,
                        initial_image_obs[x],
                        seed=rpg,
                        modality="language",
                        argmax=True
                    )
                    key, subkey = jax.random.split(key)
                    rand = jax.random.split(subkey, im_obs.shape[0])
                    foo = jax.vmap(sample, in_axes=({'image': 0, 'proprio': 0}, 0), out_axes=0)
                    foo2 = jax.vmap(unnormalize_action, in_axes=(0,None,None), out_axes=0)
                    res = foo(obs, rand)
                    res = foo2(res, action_mean, action_std)
                    acts.append(res)
                    
                unrolled_actions.append(jnp.array(acts))
                
            for _ in range(30):
                
                
                this_unrolled_actions = sample_boundaries([i.shape[1] for i in unrolled_actions], FLAGS.max_subtask_steps, FLAGS.min_subtask_steps, candidate_plan, unrolled_actions)
                
                print(this_unrolled_actions.shape)
                costs = jnp.mean(jnp.sum(jnp.square(actions - this_unrolled_actions), axis=[1, 2]))
                
                print(costs)
                
                if costs < candidate_plan_costs[plan_id]:
                    candidate_plan_costs[plan_id] = costs

        best_plan_id = np.argmin(candidate_plan_costs)
        best_plan = candidate_plans[best_plan_id]
        t_plus = time.time()
        break

    
    print("Time consumption: ", t_plus - t, " seconds")
    print('Best Plan: ')
    print(best_plan)

if __name__ == '__main__':
    app.run(main)
