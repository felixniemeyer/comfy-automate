import os
import time

import json
import random

from PIL import Image 

from common import queue_prompt

workflow_file = "workflows/new_next_frame.json"

ckpt = "sd_xl_base_1.0.safetensors"

comfy_path = os.path.expanduser('~/work/ai/ComfyUI')
comfy_output_path = os.path.join(comfy_path, "output")
comfy_input_path = os.path.join(comfy_path, "input")

output_folder_name = "frame_by_frame"

model_out_size = 768

prompt_texts = [
    ["the first time I smoked weed, 30 years ago, boy, bong", 0], 
    ["bong, smoke, weed", 14], 
    ["bong, smoke, table, third person experience, people, group of people, chairs", 30], 
    ["food, bread, biscuits, cheese, eating", 34], 
    ["night, darkness falling, bicycle, ride, street, forest", 44],
    ["night, garden gate, crash, scar on leg, injury", 40],
    ["garden gate, sneaker, trainer lost, buckled garden gate, next day, sun", 60],
    ["weed, 420, lifelong friend, companion", 70],
    ["weed, 420, lifelong friend, companion", 74],
]

# reverse
prompt_texts = prompt_texts[::-1]
for i in range(len(prompt_texts)):
    prompt_texts[i][1] = 74 - prompt_texts[i][1]

fps = 7.5

prompt_prefix = "cannabis, marjuana, cannabis leaf, hemp plant"
prompt_postfix = "photography, photorealistic"

# ensure the output_folder is empty and exists
answer = None
postfix = 1
absolute_output_folder = os.path.join(comfy_output_path, output_folder_name)
while os.path.exists(absolute_output_folder):
    # promt the user to delete the folder
    if answer == None: 
        answer = input(f"Folder {absolute_output_folder} already exists. (i)ncrement, (d)elete, (q)uit? Default is increment.")
    if answer.lower() == 'y':
        # go through all files in the folder and delete them
        for file in os.listdir(absolute_output_folder):
            os.remove(os.path.join(absolute_output_folder, file))
        os.rmdir(absolute_output_folder)
    if answer.lower() == 'q':
        print("Exiting")
        exit()
    else:
        postfix += 1
        absolute_output_folder = os.path.join(comfy_output_path, output_folder_name + str(postfix)) 

if postfix > 1:
    output_folder_name = output_folder_name + str(postfix)

print("output folder:", absolute_output_folder)

# load next_fram_gen.json workflow and find relevant nodes
ckpt_loader = None
image_loader = None
prompt_from = None
prompt_to = None
conditioning_average = None
gen_size = None
k_sampler = None
with open(workflow_file) as f:
    workflow = json.load(f)
    # find CheckpointLoaderSimple in node dictionary
    for node_id in workflow:
        node = workflow[node_id]
        class_type = node["class_type"]
        if class_type == "CheckpointLoaderSimple":
            ckpt_loader = node
        if class_type == "LoadImage":
            image_loader = node
        if class_type == "ConditioningAverage":
            conditioning_average = node
            from_node_id = conditioning_average["inputs"]["conditioning_from"][0]
            print(from_node_id)
            to_node_id = conditioning_average["inputs"]["conditioning_to"][0]
            print(to_node_id)
            prompt_from = workflow[from_node_id]
            prompt_to = workflow[to_node_id]
        if class_type == "YANC.IntegerCaster":
            gen_size = node
        if class_type == "SaveImage":
            save_image = node
        if class_type == "KSampler": 
            k_sampler = node

if ckpt_loader == None:
    print("Could not find CheckpointLoaderSimple in workflow", workflow_file)
    exit()
if save_image == None:
    print("Could not find SaveImage in workflow", workflow_file)
    exit()
if k_sampler == None:
    print("Could not find KSampler in workflow", workflow_file)
    exit()
if conditioning_average == None:
    print("Could not find ConditioningAverage in workflow", workflow_file)
    exit()
if gen_size == None:
    print("Could not find EmptyLatentImage in workflow", workflow_file)
    exit()
if image_loader == None: 
    print("Could not find LoadImage in workflow", workflow_file)
    exit()
if prompt_from == None: 
    print("Could not find prompt_from in workflow", workflow_file)
    exit()
if prompt_to == None:
    print("Could not find prompt_to in workflow", workflow_file)
    exit()

ckpt_loader["inputs"]["ckpt_name"] = ckpt
image_loader["inputs"]["image"] = os.path.join(comfy_input_path, "previous_frame.png")
gen_size["inputs"]["value"] = model_out_size
save_image["inputs"]["filename_prefix"] = output_folder_name + '/f'

def setPrompts(from_p, to_p):
    prompt_from["inputs"]["text"] = from_p + prompt_prefix
    prompt_to["inputs"]["text"] = to_p + prompt_prefix

def setProgress(progress):
    conditioning_average["inputs"]["conditioning_to_strength"] = progress

def setSeed(seed):
    k_sampler["inputs"]["seed"] = seed


if not os.path.exists(absolute_output_folder):
    os.makedirs(absolute_output_folder)

previous_generation = "grey.png"

num_prompts = len(prompt_texts)

upscale_out_size = model_out_size * 2

for prompt_index in range(num_prompts - 1):
    prompt_text_from = prompt_texts[prompt_index]
    prompt_text_to = prompt_texts[prompt_index + 1]
    pf = prompt_text_from[0]
    pt = prompt_text_to[0]
    steps = int((prompt_text_to[1] - prompt_text_from[1]) * fps)

    setPrompts(pf, pt)

    print(f"Interpolating between {pf} and {pt} in {steps} steps")

    for step in range(steps):

        # load previous_generation image
        image = Image.open(previous_generation)
        # zoom in 1 % and rotate 1 degree
        image = image.rotate(1)
        crop_border_px = 0.02 * upscale_out_size
        image = image.crop((
            crop_border_px, 
            crop_border_px, 
            upscale_out_size - crop_border_px, 
            upscale_out_size - crop_border_px
        ))

        # save
        image.save(os.path.join(comfy_input_path, "previous_frame.png"))
        
        # count number of files in output folder
        files_before = os.listdir(absolute_output_folder)

        print(f"Step {step} of {steps}")

        setProgress(step / steps)
        setSeed(random.randint(0, 10000000000))

        queue_prompt(json.dumps(workflow))

        # wait until a file appears in comy output folder
        files_after = []
        while len(files_after) <= len(files_before):
            time.sleep(0.1)
            files_after = os.listdir(absolute_output_folder)

        # find the new file
        new_file = None
        for file in files_after:
            if file not in files_before:
                previous_generation = os.path.join(absolute_output_folder, file)
                break

        # wait for file to be written to disk
        time.sleep(1.5) 

