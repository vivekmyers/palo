from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from openai import OpenAI
import cv2
import os
import base64
import json
import time
from PIL import Image
import io
import tensorflow as tf

client = OpenAI()

examples = """
Move the mushroom to the pot;
open the door;
place the spoon on the towel;
place the spoon on the burner
"""

bad_examples = """
move the gripper down towards the mushroom
open the gripper
"""
def encode_image(im):
    
    if isinstance(im, np.ndarray):
        im = Image.fromarray(im)
        buf = io.BytesIO()
        im.save(buf, format='JPEG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    elif isinstance(im, str):
        with open(im, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    else:
        raise TypeError("Image should be either a filename string or a np array")
    
    
def query2(im, x):
    messages = [
        {
            "role": "system",
            "content": "You are a planner who specializes in motion planning for robots."
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg:base64,{im}"}},
                {"type": "text",
                 "text": x}
            ]
        }
    ]

    params = {
        "model": "gpt-4o-2024-05-13",
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.2,
    }
    result = client.chat.completions.create(**params)
    
    
    

    return result.choices[0].message.content 

def process_candidates(candidates: list[str]):
    candidates_json = []
    for cand in candidates:
        try:
            this_candidate = json.loads(cand[7:-3])
            candidates_json.append(this_candidate)
        except:
            pass
    results_all_hl = []
    results_all_ll = []
    for cand in candidates_json:
        result_hl = []
        result_ll = []
        for k, v in cand.items():
            result_hl.extend([k] * len(v))
            result_ll.extend(v)
        results_all_hl.append(result_hl)
        results_all_ll.append(result_ll)
    return candidates_json, [results_all_hl, results_all_ll]

def make_plan(obs, big_instr):
    cx = f"""
    Here is an image observed by a robot in a robot manipulation task. The task is to {big_instr}. Now give me the steps in which the robot must take in order to accomplish this task. 

    Assume that robots can take in command in the form of "put x into y". First start by identifying the items that are relevant, and then form the commands. Return a python list only
    """
    return query2(obs, cx)

def sample_from_queries(lst_of_responses):
    """
    The general gist of sampling:
    The first response should be twice as likely as the second command, and the second command should be longer than that of the third command, etc.
    We then normalize it
    """
    num_responses = len(lst_of_responses)
    p_base = np.array([(1/2) ** i for i in range(num_responses)])
    p_base = p_base/np.sum(p_base)

    return lst_of_responses[np.random.choice(num_responses, p=p_base)]

def query_long_horizon(obs, instrs):
    obs = encode_image(obs)
    prompt=f"""
    Here is an image observed by the robot in a robot manipulation environment. The gripper is at the top left corner in the image.
    Now plan for the the list of subtasks and skills the robot needs to perform in order to {instrs}. 
    A subtask is one set of actions connected in a logical manner. For example, \"put an object to some place\", \"Sweep some objects to the right\", and \"Open the drawer\" are valid subtasks.
    
    For each subtask, you need to plan for the steps the robot needs to take in order to complete the subtask. Each step in the plan can be selected from the available skills below:

    *movement direction:
        *forward. This skill moves the robot gripper away from the camera by a small distance.
        *backward. This skill moves the robot gripper towards the camera by a small distance.
        *left. This skill moves the robot gripper to the left of the image by a small distance.
        *right. This skill moves the robot gripper to the right of the image by a small distance.
        *up. This skill moves the robot gripper upward until a safe height.
        *down. This skill moves the robot gripper downward to the table surface.

    *gripper movement:
        *close the gripper. This skill controls the robot gripper to close to grasp an object.
        *open the gripper. This skill controls the robot gripper to open and release the object in hand.

    You may choose between using one of movement direction or gripper movement. 
    If you were to choose to use movement direction, you may use one or two directions and include a target object, and you should format it like this:
    \"move the gripper x towards z\" or \"move the gripper x and y towards z\" where x and y are the directions and z is the target object. 
    You also must start your command with \"move the gripper\". 
    Therefore, instead of saying something like \"down\" or \"up\", you should phrase it like \"move the gripper down\" and \"move the gripper up\". 
    Make sure to include at least one direction in your command since otherwise this command format won't make sense.

    If you were to choose to use gripper movement, you should format the command as \"close the gripper on x\" or \"open the gripper to release x\", where x is the target object. 
    You may discard the target object if necessary. In that case use \"close the gripper\" or \"open the gripper\".
    If you think the gripper is close to the target object, then you must choose to use gripper movement to grasp the target object to maintain efficiency.
    If the task is related to sweeping, after you close the gripper, do not move up until after you have released the gripper. 

    Pay close attention to these factors: 
    *Which task are you doing. If you are sweeping, then don't move up as the towel needs to be consistently on the surface.
    *Whether the gripper is closed. 
    *Whether the gripper is holding the target object.
    *How far the two target objects are. If they are across the table, then duplicate the commands with a copy of it.
    *Where the gripper is.
    
    Especially pay attention to the actual direction between the gripper and the target object. Remember that the robot's angle is roughly the same as the camera's angle. 
    If the target object is closer to the edge of the table that is near the top of the image, you should move forward, otherwise you should move backward.

    Start by looking at what objects are in the image, and then plan with the direction of the objects in mind. The tasks should be completed sequentially, therefore you need to consider the position of the gripper after each task before planning the next task. 
    You should return a json dictionary with the following fields:
     - subtask: this should be the key of the dictionary. It should contain the only the verbal description of the subtask the robot needs to perform sequentially in order to finish the task, and they should be ordered in the same way the task is completed.
     - list of skills for each subtask: this should be the value of the dictionary. It should be a list of skills the robot needs to perform in order to finish the corresponding subtask.
    Do not start the subtask with \"subtask: \". Do not return any other comments other than the dictionary mentioned above, and make the list as concise as possible.
    """
    return query2(obs, prompt)

def make_multiple_response(im, instr, num_res):
    res = []
    for _ in range(num_res):
        res.append(query_long_horizon(im,instr))

    return res

def proc_im_color(im: np.ndarray):
    
    h, w, d = im.shape
    im_new = np.zeros((h+40, w+40, d), dtype=np.uint8)
    im_new[20:-20,20:-20,:]=im
    im_new[:,:20,1] = 255
    im_new[:,-20:,0] = 255
    im_new[:20,:,2]=255
    return im_new

if __name__ == "__main__":
    im_dim = "./data/traj0/images0/im_0.jpg"
    instr = "Put the sweet potato in the closed drawer"
    res = make_multiple_response(im_dim, instr, 10)
    for i in res:
        print(i)
    
    
