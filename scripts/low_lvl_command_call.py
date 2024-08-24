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
from eval_utils import proc_n_encode




client = OpenAI()


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
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.2,
    }
    result = client.chat.completions.create(**params)
    
    
    

    return result.choices[0].message.content 

def query3(obs_start, obs_curr, x):
    
    messages = [
        {
            "role": "system",
            "content": "You are an expert at planning robot motion from images and give language instruction to robots."
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg:base64,{obs_start}"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg:base64,{obs_curr}"}},
                {"type": "text",
                 "text": x}
            ]
        }
    ]

    params = {
        "model": "gpt-4-vision-preview",
        "messages": messages,
        "max_tokens": 500,
    }
    result = client.chat.completions.create(**params)
    
    return [result.choices[i].message.content for i in range(len(result.choices))]


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
    return [results_all_hl, results_all_ll]

def make_plan(obs, big_instr):
    cx = f"""
    Here is an image observed by a robot in a robot manipulation task. The task is to {big_instr}. Now give me the steps in which the robot must take in order to accomplish this task. 

    Assume that robots can take in command in the form of "put x into y". First start by identifying the items that are relevant, and then form the commands. Return a python list only
    """
    return query2(obs, cx)

def query_mid_way(start_obs, obs, instr, lst_instr, plan: list, plan_idx: int)->list:
    
    im_start = encode_image(start_obs)
    im_curr = encode_image(obs)

    prompt = f"""Here is an image of a robot manipualtion environment observed at the beginning and the current observation of the robot. 
    
    There is a preexisting plan: {plan} that maps out the motion the robot should take.

    Currently the robot is just done executing subplan {plan[plan_idx]},
    Now plan for the subsequent subtasks the robot needs to perform in order to {instr}.
    
    Each step in the plan can be selected from the available skills below:
    
    *reach:
        *reach(x). This skill moves the robot gripper towards a target object x.

    *movement direction:
        *forward. This skill moves the robot gripper away from the camera by a small distance.
        *backward. This skill moves the robot gripper towards the camera by a small distance.
        *left. This skill moves the robot gripper to the left of the image by a small distance.
        *right. This skill moves the robot gripper to the right of the image by a small distance.
        *up. This skill moves the robot gripper upward until a safe height.
        *down. This skill moves the robot gripper downward to the table surface.
        *clockwise. This skill rotates the robot gripper away from the camera by a small degree.
        *counterclockwise. This skill rotates the robot gripper towards the camera by a small degree.

    *gripper movement:
        *close the gripper. This skill controls the robot gripper to close to grasp an object.
        *open the gripper. This skill controls the robot gripper to open and release the object in hand.

    You may choose between using one of reach, movement direction ,and gripper movement. 
    If you were to choose to use reach, you must format your command in the form of \"Move the gripper towards x\", where x is the target object.
    If you were to choose to use movement direction, you may use up to two directions and include a target object, and you should format it like this:
    \"Move the gripper x and y towards z\", where x and y are directions and z is the target object. You also must start your command with \"Move the gripper\"
    x should be more important than y in terms of direction. If the second direction and the target object are not necessary, you should discard it, but make sure to include at least one direction in your command.

    If you were to choose to use gripper movement, you should format the command as \"close the gripper to grasp x\" or \"open the gripper to release x\", where x is the target object. You may also choose to disregard the target object too, in which case you should use only the skill

    Pay close attention to the actual direction between the gripper and the target object, for example, if the gripper is on the right of the target object, then you should tell the gripper to move left.

    The relevant objects in this scenarios are {lst_instr} and they should all be in the image.

    Return only the subgoals as a python list, without any other additional comments.

    """
    return query3(im_start, im_curr, prompt)


def query_w_bbox(obs, instr, lst_instr):
    if "gripper" not in lst_instr:
        lst_instr.append("gripper")
    new_im_b64, _ = proc_n_encode(obs, lst_instr)

    prompt = f"""Here is an image observed by the robot in a robot manipulation environment with a green band on the right of the image and a red band on the left of the image. 
    Now plan for the the list of subtasks the robot needs to perform in order to {instr}.

    Each step in the plan can be selected from the available skills below:

    *movement direction:
        *forward. This skill moves the robot gripper away from the camera by a small distance.
        *backward. This skill moves the robot gripper towards the camera by a small distance.
        *left. This skill moves the robot gripper to the left of the image by a small distance.
        *right. This skill moves the robot gripper to the right of the image by a small distance.
        *up. This skill moves the robot gripper upward until a safe height.
        *down. This skill moves the robot gripper downward to the table surface.
        *clockwise. This skill rotates the robot gripper away from the camera by a small degree.
        *counterclockwise. This skill rotates the robot gripper towards the camera by a small degree.

    *gripper movement:
        *close the gripper. This skill controls the robot gripper to close to grasp an object.
        *open the gripper. This skill controls the robot gripper to open and release the object in hand.

    You may choose between using one of movement direction or gripper movement. 
    If you were to choose to use movement direction, you may use up to two directions and include a target object, and you should format it like this:
    \"Move the gripper x and y towards z\", where x and y are directions and z is the target object. You also must start your command with \"Move the gripper\"
    X should be more important than y in terms of direction. If the second direction and the target object are not necessary, you should discard it, but make sure to include at least one direction in your command.

    If you were to choose to use gripper movement, you should format the command as \"close the gripper\" or \"open the gripper\", where x is the target object.
    If you think the gripper is close to the target object, then you must choose to use gripper movement to grasp the target object to maintain efficiency.

    Pay close attention to these factors: 
    *Whether the gripper is closed. 
    *Whether the gripper is holding the target object.
    
    Especially pay attention to the actual direction between the gripper and the target object. Reference the color bands on the edge of the image and make logical decisions. The robot's direction is the same as the camera's direction also.

    The relevant objects have been segmented and annotated, with the name being shown and each unique object has a different color.
    
    Think this through step by step. However, you should return only the subgoals as a python list, without any other additional comments.
    """
    return query2(new_im_b64, prompt), new_im_b64

def query_sweep(obs, instr):
    x = f"""
    Here is an image observed by the robot. Now plan for the list of subtasks the robot needs to perform in order to {instr}.
    
    Each step in the plan can be selected from the available skills below:

    *movement direction:
        *forward. This skill moves the robot gripper away from the camera by a small distance.
        *backward. This skill moves the robot gripper towards the camera by a small distance.
        *left. This skill moves the robot gripper to the left of the image by a small distance.
        *right. This skill moves the robot gripper to the right of the image by a small distance.
        *up. This skill moves the robot gripper upward until a safe height.
        *down. This skill moves the robot gripper downward to the table surface.
        *clockwise. This skill rotates the robot gripper away from the camera by a small degree.
        *counterclockwise. This skill rotates the robot gripper towards the camera by a small degree.
    
    *rotate the gripper:
        *tilt. This skill moves the 

    *gripper movement:
        *close the gripper. This skill controls the robot gripper to close to grasp an object.
        *open the gripper. This skill controls the robot gripper to open and release the object in hand.

    You may choose between using one of movement direction or gripper movement. 
    If you were to choose to use movement direction, you may use up to two directions and include a target object, and you should format it like this:
    \"Move the gripper x and y towards z\", where x and y are directions and z is the target object. You also must start your command with \"Move the gripper\"
    X should be more important than y in terms of direction. If the second direction and the target object are not necessary, you should discard it, but make sure to include at least one direction in your command.
    Once the robot is holding the object in which to sweep with, you must also order the gripper to go down as a secondary command to prevent the gripper is too high above the table.
    
    Since this instruction is a sweeping task, you need to think about the object in which you use to sweep: if it has a handle, you must identify where the handle of the object is and tell the robot to reach for that spot.

    If you were to choose to use gripper movement, you should format the command as \"close the gripper\" or \"open the gripper\", where x is the target object.
    If you think the gripper is close to the target object, then you must choose to use gripper movement to grasp the target object to maintain efficiency.

    Pay close attention to these factors: 
    *Where in the object do you want to grasp. Since this is a sweeping task, you should locate and grasp the handle of the object instead of anywhere on the object.
    *Height of the gripper. You should keep the gripper as close to the table as possible.
    *Whether the gripper is closed. 
    *Whether the gripper is holding the target object.
    
    Especially pay attention to the actual direction between the gripper and the target object. Reference the color bands on the edge of the image and make logical decisions. The robot's direction is the same as the camera's direction also.

    The relevant objects have been segmented and annotated, with the name being shown and each unique object has a different color.
    
    Think this through step by step. However, you should return only the subgoals as a python list, without any other additional comments.
    """
    return query2(obs, x)


def query_w_bbox_midway(obs, instr, lst_instr, prev_instr):
    if "gripper" not in lst_instr:
        lst_instr.append("tip of gripper")
    new_im_b64, _ = proc_n_encode(obs, lst_instr)

    prompt = f"""Here is an image observed by the robot. Now plan for the the list of subtasks the robot needs to perform in order to {instr}.

    Each step in the plan can be selected from the available skills below:

    *movement direction:
        *forward. This skill moves the robot gripper away from the camera by a small distance.
        *backward. This skill moves the robot gripper towards the camera by a small distance.
        *left. This skill moves the robot gripper to the left of the image by a small distance.
        *right. This skill moves the robot gripper to the right of the image by a small distance.
        *up. This skill moves the robot gripper upward until a safe height.
        *down. This skill moves the robot gripper downward to the table surface.
        *clockwise. This skill rotates the robot gripper away from the camera by a small degree.
        *counterclockwise. This skill rotates the robot gripper towards the camera by a small degree.

    *gripper movement:
        *close the gripper. This skill controls the robot gripper to close to grasp an object.
        *open the gripper. This skill controls the robot gripper to open and release the object in hand.

    You may choose between using one of movement direction and gripper movement. 
    If you were to choose to use movement direction, you may use up to two directions and include a target object, and you should format it like this:
    \"Move the gripper x and y towards z\", where x and y are directions and z is the target object. You also must start your command with \"Move the gripper\"
    X should be more important than y in terms of direction. If the second direction and the target object are not necessary, you should discard it, but make sure to include at least one direction in your command.

    If you were to choose to use gripper movement, you should format the command as \"Close the gripper to pick up x\" or \"open the gripper to release x\", where x is the target object. If there is no target object applicable, you can formulate your command as \"close the gripper\" and \"Open the gripper\"

    Pay close attention to the actual direction between the gripper and the target object, for example, if the gripper is on the right of the target object, then you should tell the gripper to move left.
    To aid you in this, there is a green band at the left of the image and a red band at the right of the image. Use these to reference the direction of your commands.

    The relevant objects have been segmented and annotated, with the name (and an additional confidence metric that you can ignore) being shown and each unique object has a different color.

    Return only the subgoals as a python list, without any other additional comments.
    """
    return query2(new_im_b64, prompt), new_im_b64

def query_wo_bbox(obs, instr, lst_instr=None):
    new_im_b64 = encode_image(obs)
    w_color_prompt = "with a green band on the right of the image and a red band on the left of the image"
    prompt = f"""Here is an image observed by the robot in a robot manipulation environment. The gripper is situated above the bottom center of the table.
    Now plan for the the list of subtasks the robot needs to perform in order to {instr}.

    Each step in the plan can be selected from the available skills below:

    *movement direction:
        *forward. This skill moves the robot gripper away from the camera by a small distance.
        *backward. This skill moves the robot gripper towards the camera by a small distance.
        *left. This skill moves the robot gripper to the left of the image by a small distance.
        *right. This skill moves the robot gripper to the right of the image by a small distance.
        *up. This skill moves the robot gripper upward until a safe height.
        *down. This skill moves the robot gripper downward to the table surface.
        *clockwise. This skill rotates wrist of the robot gripper clockwise with respect to the camera by a small degree.
        *counterclockwise. This skill rotates the robot gripper counterclockwise with respect to the camera by a small degree.

    *gripper movement:
        *close the gripper. This skill controls the robot gripper to close to grasp an object.
        *open the gripper. This skill controls the robot gripper to open and release the object in hand.

    You may choose between using one of movement direction or gripper movement. 
    If you were to choose to use movement direction, you may use one or two directions and include a target object, and you should format it like this:
    \"move the gripper x towards z\" or \"move the gripper x and y towards z\" where x and y are the directions and z is the target object. You also must start your command with \"move the gripper\". 
    Therefore, instead of saying something like \"left\" or \"up\", you should phrase it like \"move the gripper left\" and \"move the gripper up\"
    Make sure to include at least one direction in your command.

    If you were to choose to use gripper movement, you should format the command as \"close the gripper to pick up x\" or \"open the gripper to release x\", where x is the target object. 
    If you think the gripper is very close to the target object, then you must choose to use gripper movement to grasp the target object to maintain efficiency.
    If the task is related to sweeping, after you close the gripper, do not move up until after you have released the gripper. 

    Pay close attention to these factors: 
    *Which task are you doing.
    *Whether the gripper is closed. 
    *Whether the gripper is holding the target object.
    *How far the two target objects are. If they are across the table, then duplicate the commands with a copy of it.
    
    Especially pay attention to the actual direction between the gripper and the target object. Remember that the robot's angle is roughly the same as the camera's angle. If the target object is closer to the edge of the table that is near the top of the image, you should move forward, otherwise you should move backward.
    
    Start by looking at what objects are in the image, and then plan with the direction of the objects in mind. You should return only the subgoals as a python list, without any other additional comments.
    """
    return query2(new_im_b64, prompt), new_im_b64

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
    prompt = f"""
    Here is an image observed by the robot in a robot manipulation environment. The gripper is at the top left corner in the image.
    Now plan for the the list of subtasks the robot needs to perform in order to {instrs}

    Each step in the plan can be selected from the available skills below:

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
    You also must start your command with \"move the gripper\". Therefore, instead of saying something like \"down\" or \"up\", you should phrase it like \"move the gripper down\" and \"move the gripper up\". Make sure to include at least one direction in your command since otherwise this command format won't make sense.

    If you were to choose to use gripper movement, you should format the command as \"close the gripper to pick up x\" or \"open the gripper to release x\", where x is the target object. 
    You may discard the target object if necessary. In that case use \"close the gripper\" or \"open the gripper\".
    If you think the gripper is close to the target object, then you must choose to use gripper movement to grasp the target object to maintain efficiency.
    If the task is related to sweeping, after you close the gripper, do not move up until after you have released the gripper. 

    Pay close attention to these factors: 
    *Which task are you doing. 
    *Whether the gripper is closed. 
    *Whether the gripper is holding the target object.
    *How far the two target objects are. If they are across the table, then duplicate the commands with a copy of it.
    *Where the gripper is. After the end of each subtask, it is reasonable to assume that the gripper will not be at where it originally was in the image, but somewhere close to the last target object.
    
    Especially pay attention to the actual direction between the gripper and the target object. Remember that the robot's angle is roughly the same as the camera's angle. 
    If the target object is closer to the edge of the table that is near the top of the image, you should move forward, otherwise you should move backward.

    Start by looking at what objects are in the image, and then plan with the direction of the objects in mind. The tasks should be completed sequentially, therefore you need to consider the position of the gripper after each task before planning the next task. 
    You should return a json dictionary in which each key is the subtask to perform, and each value is the list of skills to perform in order to complete the subtask.
    """

    return query2(obs, prompt)

def make_multiple_response(im, instr, num_res):
    res = []
    for _ in range(num_res):
        res.append(query_long_horizon(im,instr))

    return res

def sample_wrapper(im, instr, instr_lst):
    res, _ = query_w_bbox(im, instr, instr_lst)
    return sample_from_queries(res)

def proc_im_color(im: np.ndarray):
    
    h, w, d = im.shape
    im_new = np.zeros((h+40, w+40, d), dtype=np.uint8)
    im_new[20:-20,20:-20,:]=im
    im_new[:,:20,1] = 255
    im_new[:,-20:,0] = 255
    im_new[:20,:,2]=255
    return im_new

if __name__ == "__main__":
    im_dim = "/mount/harddrive/robonetv2/bridge_data_v2/few_shot_instr/clear_center/2024-05-17_20-08-42/raw/traj_group0/traj5/images0/im_0.jpg"
    im = encode_image(im_dim)
    instr = "Sweep the golden confetti with the green spoon, after moving the mushroom into the metal pot."
    res = make_multiple_response(im, instr, 10)
    plans_lst = process_candidates(res)
    print(plans_lst)
    
    
    
    

    
    

    
    
    
    
    
    
    
    
    
    
    
<<<<<<< HEAD
    
    
=======
    
    

    

    im_dim = "/mount/harddrive/robonetv2/bridge_data_v2/few_shot_instr/clear_center/2024-05-17_20-08-42/raw/traj_group0/traj0/images0/im_0.jpg"
    im = encode_image(im_dim)
    instr = "Put the mushroom in the pot and use the spoon to sweep the confetti to the right"
    res = make_multiple_response(im, instr, 10)
    for result in res:
        print(result)
>>>>>>> e086dccacef5e2e9a6748af39387795af28a2458
