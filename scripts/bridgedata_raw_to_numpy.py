import copy
from functools import partial
import tensorflow as tf
from datetime import datetime
import glob
import os
from collections import defaultdict
from PIL import Image
import pickle
import numpy as np
from absl import app, flags, logging
import tqdm
import random
from multiprocessing import Pool
from traj_relabel import *
import logging
import openai
import wandb

logging.getLogger('openai').setLevel(logging.ERROR)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

FLAGS = flags.FLAGS

flags.DEFINE_string("input_path", None, "Input path", required=True)
flags.DEFINE_string("output_path", None, "Output path", required=True)
flags.DEFINE_integer(
    "depth",
    5,
    "Number of directories deep to traverse to the dated directory. Looks for"
    "{input_path}/dir_1/dir_2/.../dir_{depth-1}/2022-01-01_00-00-00/...",
)
flags.DEFINE_bool("overwrite", False, "Overwrite existing files")
flags.DEFINE_float(
    "train_proportion", 0.9, "Proportion of data to use for training (rather than val)"
)
flags.DEFINE_integer("num_workers", 15, "Number of threads to use")


def squash(path):  
    im = Image.open(path)
    im = im.resize((224, 224), Image.LANCZOS)
    out = np.asarray(im).astype(np.uint8)
    return out


def process_images(path):  
    names = sorted(
        [x for x in os.listdir(path) if "images" in x and not "depth" in x],
        key=lambda x: int(x.split("images")[1]),
    )
    image_path = [
        os.path.join(path, x)
        for x in os.listdir(path)
        if "images" in x and not "depth" in x
    ]
    image_path = sorted(image_path, key=lambda x: int(x.split("images")[1]))

    images_out = defaultdict(list)

    tlen = len(glob.glob(image_path[0] + "/im_*.jpg"))

    for i, name in enumerate(names):
        for t in range(tlen):
            images_out[name].append(squash(image_path[i] + "/im_{}.jpg".format(t)))

    images_out = dict(images_out)

    obs, next_obs = dict(), dict()

    for n in names:
        obs[n] = images_out[n][:-1]
        next_obs[n] = images_out[n][1:]
    return obs, next_obs


def process_state(path):
    fp = os.path.join(path, "obs_dict.pkl")
    with open(fp, "rb") as f:
        x = pickle.load(f)
    return x["full_state"][:-1], x["full_state"][1:]


def process_time(path):
    fp = os.path.join(path, "obs_dict.pkl")
    with open(fp, "rb") as f:
        x = pickle.load(f)
    return x["time_stamp"][:-1], x["time_stamp"][1:]


def process_actions(path):  
    fp = os.path.join(path, "policy_out.pkl")
    with open(fp, "rb") as f:
        act_list = pickle.load(f)
    if isinstance(act_list[0], dict):
        act_list = [x["actions"] for x in act_list]
    return act_list




def process_dc(path, train_ratio=0.9):
    
    if "lmdb" in path:
        logging.warning(f"Skipping {path} because uhhhh lmdb?")
        return [], [], [], [], {}

    all_dicts_train = list()
    all_dicts_test = list()
    all_rews_train = list()
    all_rews_test = list()

    
    date_time = datetime.strptime(path.split("/")[-1], "%Y-%m-%d_%H-%M-%S")
    latency_shift = date_time < datetime(2021, 7, 23)

    search_path = os.path.join(path, "raw", "traj_group*", "traj*")
    all_traj = glob.glob(search_path)
    if all_traj == []:
        logging.info(f"no trajs found in {search_path}")
        return [], [], [], [], {}

    random.shuffle(all_traj)

    num_traj = len(all_traj)
    for itraj, tp in tqdm.tqdm(enumerate(all_traj)):
        ld = os.listdir(tp)
        tasks = [None]

        if "lang.txt" in ld:
            with open(os.path.join(tp, "lang.txt")) as f:
                lines = list(f)
                if lines:
                    tasks = [line.strip() for line in lines]

                
        elif "bridge_data_v1" in path:
            tasks = [path.split("/")[-2].replace("_", " ")]

        for task in tasks:
            try:
                out = dict()

                assert "obs_dict.pkl" in ld, tp + ":" + str(ld)
                assert "policy_out.pkl" in ld, tp + ":" + str(ld)
                

                obs, next_obs = process_images(tp)
                acts = process_actions(tp)
                state, next_state = process_state(tp)
                time_stamp, next_time_stamp = process_time(tp)
                term = [0] * len(acts)
                lang = [task] * len(acts)

                out["observations"] = obs
                out["observations"]["state"] = state
                out["observations"]["time_stamp"] = time_stamp
                out["next_observations"] = next_obs
                out["next_observations"]["state"] = next_state
                out["next_observations"]["time_stamp"] = next_time_stamp

                out["observations"] = [
                    dict(zip(out["observations"], t))
                    for t in zip(*out["observations"].values())
                ]
                out["next_observations"] = [
                    dict(zip(out["next_observations"], t))
                    for t in zip(*out["next_observations"].values())
                ]

                out["actions"] = acts
                out["terminals"] = term
                out["language"] = lang

                if latency_shift:
                    out["observations"] = out["observations"][1:]
                    out["next_observations"] = out["next_observations"][1:]
                    out["actions"] = out["actions"][:-1]
                    out["terminals"] = term[:-1]
                    out["language"] = lang[:-1]

                labeled_rew = copy.deepcopy(out["terminals"])[:]
                labeled_rew[-2:] = [1, 1]

                traj_len = len(out["observations"])
                assert len(out["next_observations"]) == traj_len
                assert len(out["actions"]) == traj_len
                assert len(out["terminals"]) == traj_len
                assert len(labeled_rew) == traj_len

                
                

                
                low_lvl_instr = caption_relabel(out, tp)

                info = {}
                for instr in low_lvl_instr:
                    info[instr] = info.get(instr, 0) + 1

                out["language_low_level"] = low_lvl_instr

                if itraj < int(num_traj * train_ratio):
                    all_dicts_train.append(out)
                    all_rews_train.append(labeled_rew)
                else:
                    all_dicts_test.append(out)
                    all_rews_test.append(labeled_rew)
            except FileNotFoundError as e:
                logging.error(e)
                continue
            except AssertionError as e:
                logging.error(e)
                continue

    return all_dicts_train, all_dicts_test, all_rews_train, all_rews_test, info


def make_numpy(x, train_proportion):
    itr, path = x
    dirname = os.path.abspath(path)
    outpath = os.path.join(
        FLAGS.output_path, *dirname.split(os.sep)[-(FLAGS.depth - 1) :]
    )
    outpath_train = tf.io.gfile.join(outpath, "train")
    outpath_val = tf.io.gfile.join(outpath, "val")
    wandb.log({"path": outpath}, step=itr)
    wandb.log({"itr": itr}, step=itr)

    outpath_train_check = tf.io.gfile.join(outpath_train, "out.npy")
    outpath_val_check = tf.io.gfile.join(outpath_val, "out.npy")
    if tf.io.gfile.exists(outpath_train_check) and tf.io.gfile.exists(outpath_val_check):
        if FLAGS.overwrite:
            logging.info(f"Deleting {outpath_train_check}")
            tf.io.gfile.rmtree(outpath)
        else:
            logging.info(f"Skipping {outpath}")
            return

    tf.io.gfile.makedirs(outpath_train)
    tf.io.gfile.makedirs(outpath_val)

    lst_train = []
    lst_val = []
    rew_train_l = []
    rew_val_l = []

    results = {}

    for dated_folder in os.listdir(path):
        curr_train, curr_val, rew_train, rew_val, info = process_dc(
            os.path.join(path, dated_folder), train_ratio=train_proportion
        )
        lst_train.extend(curr_train)
        lst_val.extend(curr_val)
        rew_train_l.extend(rew_train)
        rew_val_l.extend(rew_val)

        for key, val in info.items():
            results[key] = results.get(key, 0) + val

    with tf.io.gfile.GFile(tf.io.gfile.join(outpath_train, "out.npy"), "wb") as f:
        np.save(f, lst_train)
        
    with tf.io.gfile.GFile(tf.io.gfile.join(outpath_val, "out.npy"), "wb") as f:
        np.save(f, lst_val)
        

    return results
    
    

    
    
    


def main(_):
    assert FLAGS.depth >= 1

    wandb.init(project="bridgedata_raw_to_numpy")
    

    
    paths = glob.glob(os.path.join(FLAGS.input_path, *("*" * (FLAGS.depth - 1))))

    worker_fn = partial(make_numpy, train_proportion=FLAGS.train_proportion)

    results = {}

    with Pool(FLAGS.num_workers) as p:
        for info in tqdm.tqdm(p.imap(worker_fn, enumerate(paths))):
            if info is None:
                continue
            for key, val in info.items():
                results[key] = results.get(key, 0) + val
            wandb.log(results)



if __name__ == "__main__":
    app.run(main)
