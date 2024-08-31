import numpy as np
import tqdm
import tensorflow as tf
from openai import OpenAI


ROTATION_MEAN = 8.024e-5
ROTATION_STD = 0.08579

STD = np.array([0.0073178, 0.01133177, 0.01211691, 0.01779856, 0.02181558, 0.08579306])


client = OpenAI()


tasks = [
    "Pick and Place",
    "Pushing objects",
    "Wipe",
    "Sweep",
    "Stacking",
    "Fold",
    "Opening and Closing",
    "Twisting",
]


def query(x):
    messages = [
        {"role": "system", "content": "You are an expert keyword extractor."},
        {
            "role": "user",
            "content": x,
        },
    ]

    result = client.chat.completions.create(model="gpt-3.5-turbo-0125", messages=messages)

    return result.choices[0].message.content


def query_lang_instr(instr: str):

    instruction = instr.split("\n")
    instruction = max(instruction[:-1] if len(instruction) > 1 else instruction, key=len)

    res = query(
        "You are presented with a text for high level instruction for a robot, and you need to extract keywords in the task description text. "
        + "In this instruction, the first keyword is the object being moved, and the second keyword, if applicable, what is the moving taking this to (either another object or a location) within the instruction."
        + " Only return the first and second keyword, and they should be separated by a comma. If the instruction is in another language, write your response in English."
        + ' For example, if the text instruction says "Pick up the silver lid on the left to the middle of two burners", return "silver lid, middle of two burners". '
        + 'Or if the instruction says: "Move the object to the top middle side of the table.", your responsed should be "object, top middle side of the table". '
        + 'Or if the instruction says : "Move the red greenish thing on the towel to the right.", return "red greendish thing on the towel, the right". '
        + 'Try your best to find the two key phrases, but if you can\'t find the second keyword within the instruction sentence, write "N/A".'
        + ' For example, if the instruction is "Move the pot lid.", the response should be "pot lid, N/A".'
        + f"There might be some other description regarding confidence at the end, you are safe to ignore it.\n The specific task description for you to analyze is: \n {instruction} \n "
    )
    return res


def query_lang_instr_fold(instr: str):
    instruction = instr.split("\n")
    instruction = max(instruction[:-1] if len(instruction) > 1 else instruction, key=len)
    res = query(
        "You are presented with a text for high level instruction for a robot, and you need to extract keywords in the task description text. "
        + "\n\nIn this instruction, the instruction tells the robot to either fold or unfold an item. Regardless of whether it is folding or unfolding the item, you should return the name of the item that is being folded or unfolded, and return that name only and nothing else in all lower case."
        + '\n\nFor example, if the instruction is: "Unfold the yellow tablecloth", return "yellow tablecloth" only and nothing else. Or if the instruction is: "Fold that thing", you should return "that thing".'
        + '\n\nTry your best to find the keyword, but if you are uncertain what the item is, return nothing like this: ""'
        + '\n\nFor example, if the instruction says: "Unfold this", you should return "", or if the instruction has nothing at all, you should also only return "". For example, if the instruction is \n\n, you should return "". \n\nThere might be some other description regarding confidence at the end, but you are safe to ignore it.'
        + f"The instruction for you to extract keyword is: \n{instruction}\n"
    )
    return res


def query_lang_instr_open(instr: str):
    instruction = instr.split("\n")
    instruction = max(instruction[:-1] if len(instruction) > 1 else instruction, key=len)
    res = query(
        "You are presented with a text for high level instruction for a robot, and you need to extract keywords in the task description text. "
        + "\n\nIn this instruction, the instruction tells the robot to either open or close an item. Regardless whether the item is being opened or being closed, you should return the name of the item only and nothing else in all lower case. If the keyword you have returned is in another language, return this in English."
        + '\n\nFor example, if the instruction is: "Open the fridge on the left", you should return only "the fridge" and nothing else, or if another instruction says: "Close the drawer on the bottom left", you should return "the drawer" and nothing else. \n\nTry your best to find the keyword, but if you are uncertain what the item is, return an empty string like this: ""'
        + '\n\nHere are a couple more examples: "Close the drawer containing strawberries" should return "the drawer"; and "Close the faucet" should return "the faucet"'
        + 'For example, if the instruction says: "Open that", you should return "", or if the instruction has nothing at all, you should also return "". For example, if the instruction is \n\n, then you should return "". \n\nThere might be some other description regarding confidence at the end, but you are safe to ignore it.'
        + f"The instruction for you to extract keyword is: \n{instruction}\n"
    )
    return res


def caption_relabel(traj: dict, path, primitive_len=4):
    """
    Draft code:
    Given a trajectory of observations, next_observations, actions, terminals, language, use the same semantics
    in traj_relabel to generate language instruction primitives that can be used downstream. Will update this later


    Generally speaking, the frame will be chunked in 4, which gives an estimated 20000x9=1800000 different primitives
    """

    instr_map = {0: ["backward", "forward"], 1: ["right", "left"], 2: ["down", "up"]}
    rot_map = [["right", "left"], ["down", "up"], ["counterclockwise", "clockwise"]]
    len_episode = len(traj["actions"])
    high_level_instruction = traj["language"][0]
    if high_level_instruction:
        has_info = high_level_instruction.strip("\n ")
    else:
        has_info = False
    if high_level_instruction:
        verb = high_level_instruction.split()[0].lower()
    else:
        verb = ""

    actions = np.array(traj["actions"])

    dPos = actions
    num_primitives = len_episode // primitive_len + 1 if len_episode % primitive_len else len_episode // primitive_len
    instr = []
    gripperChanges = np.diff(actions[:, -1])

    gripperChangesAbs = np.abs(gripperChanges)
    if any(gripperChangesAbs > 1e-1):
        isPickandPlace = True
    else:
        isPickandPlace = False
    gripped = not isPickandPlace
    released = not isPickandPlace
    if "fold" in verb:
        response = query_lang_instr_fold(high_level_instruction)
        sub = response

        if sub == "N/A":
            sub = ""
        obj = sub
        if "I'm" in sub or "keyword" in sub or "I " in sub:
            sub, obj = "", ""
    elif ("open" in verb or "close" in verb) and isPickandPlace:
        response = query_lang_instr_open(high_level_instruction)
        sub = response
        if sub == "N/A":
            sub = ""
        if "I'm" in sub or "keyword" in sub or "I " in sub:
            sub, obj = "", ""

        obj = sub

    elif isPickandPlace and has_info:
        response = query_lang_instr(high_level_instruction)
        collection_objs = response.split(",")
        collection_objs = [i.strip() for i in collection_objs][:2]

        sub = collection_objs[0]
        try:
            obj = collection_objs[1]
        except:
            obj = ""
        obj = obj if not released else None
        if "I'm" in sub or "Please" in sub or "keyword" in sub or "task" in sub:
            sub = ""
            obj = ""
        if obj == "N/A":
            obj = ""
        if sub == "N/A":
            sub = ""
    else:
        sub, obj = "", ""

    if "description" in sub or "task" in sub:
        sub, obj = "", ""
    if "flap" in path:

        sub, obj = "", ""

    for i in tqdm.trange(num_primitives, disable=True):
        dPosChunk = dPos[i * primitive_len : (i + 1) * primitive_len]
        actChange = np.mean(dPosChunk[..., :6], axis=0)
        normChange = np.abs(actChange) / STD
        if np.any(normChange[3:] > 1.5):
            is_rotate = True
        else:
            is_rotate = False

        if np.all(normChange < 0.5):
            stay_put = True
        else:
            stay_put = False
        low_level_instruction = ""
        num_actions = dPosChunk.shape[0]
        dXYZ = np.sum(dPosChunk[..., :3], axis=0)
        dMask = dXYZ > 0
        dDir = np.abs(dXYZ)
        gripperChange = gripperChanges[i * primitive_len : (i + 1) * primitive_len]
        if len(gripperChange) == 0:
            gripper_close, gripper_open = False, False
        else:
            gripperMean = np.mean(gripperChange)
            gripper_close, gripper_open = (
                gripperMean < -0.15,
                gripperMean > 0.15,
            )

        top_dir = np.argpartition(dDir, 2)[::-1]

        planar_delta = np.linalg.norm(dXYZ[top_dir[:2]])

        if planar_delta > 0.01:
            if dDir[top_dir[0]] / (dDir[top_dir[1]] + 1e-9) > np.sqrt(3):
                dir1, dir2 = instr_map[top_dir[0]][dMask[top_dir[0]]], None
                if obj:
                    low_level_instruction = f"move the gripper {dir1} towards {sub if not gripped else obj}"
                else:
                    if not gripped:
                        low_level_instruction = (
                            f"move the gripper {dir1} towards {sub}" if sub else f"move the gripper {dir1}"
                        )
                    else:
                        low_level_instruction = f"move the gripper {dir1}"
            else:
                dir1, dir2 = instr_map[top_dir[0]][dMask[top_dir[0]]], instr_map[top_dir[1]][dMask[top_dir[1]]]
                if obj:
                    low_level_instruction = f"move the gripper {dir1} and {dir2} towards {sub if not gripped else obj}"
                else:
                    if not gripped:
                        low_level_instruction = (
                            f"move the gripper {dir1} and {dir2} towards {sub}"
                            if sub
                            else f"move the gripper {dir1} and {dir2}"
                        )
                    else:
                        low_level_instruction = f"move the gripper {dir1} and {dir2}"
        if is_rotate:
            biggest_rot = np.argmax(normChange[3:])
            low_level_instruction = f"rotate the gripper {rot_map[biggest_rot][0] if actChange[biggest_rot + 3] > 0 else rot_map[biggest_rot][1]}"

        if planar_delta > 0.01 and is_rotate:
            low_level_instruction = ""

        if gripper_close:
            low_level_instruction = f"close the gripper on {sub}" if sub else "close the gripper"
            gripped = True
        elif gripper_open:
            low_level_instruction = f"open the gripper to release {sub}" if sub else "open the gripper"
            released = True
        elif stay_put:
            low_level_instruction = "stay put"

        instr.extend([low_level_instruction] * num_actions)

    """
    Thoughts on more work to be done here: we can selectively querying VLMs in this stage
    """

    return np.array(instr)


if __name__ == "__main__":

    instr = [
        "Close the drawer on the bottom",
        "open the faucet",
        "open the drawer on the bottom right, next to the sink",
    ]
    print([query_lang_instr_open(i) for i in instr])

    data_path = "XXXX"

    with tf.io.gfile.GFile(data_path, "rb") as f:
        arr = np.load(f, allow_pickle=True)

    for i in range(10):
        print(len(arr[i]["actions"]))
        caption = caption_relabel(arr[i], "hello")

        print(caption)
