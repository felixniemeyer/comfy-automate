import os
import time

import json
import random

from PIL import Image 

from prompt_interpolation_workflow import PromptInterpolationWorkflow
from common import queue_prompt

ckpt = "sd_xl_base_1.0.safetensors"

comfy_path = os.path.expanduser('~/work/ai/ComfyUI')
comfy_output_path = os.path.join(comfy_path, "output")
comfy_input_path = os.path.join(comfy_path, "input")

output_folder_name = "weed_loop"

model_out_size = 768


# 32 fps
# 13 sekunden pro block
# 4 wiederholungen
prompt_texts = [
    ["marjuana, cannabis leaves, joint, tip, stoner", 3],
    ["weed, travelling, countries, airplane", 3], 
    ["festival, joint, blunt, marjuana", 3], 
    ["fines, cash, police, airport", 4], 
]

prompt_prefix = "cannabis leaf, hemp plant, marjuana, "

fps = 32
for prompt_text in prompt_texts:
    prompt_text[1] = prompt_text[1] * fps

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

print("writing to", absolute_output_folder)
if not os.path.exists(absolute_output_folder):
    os.makedirs(absolute_output_folder)

workflow = PromptInterpolationWorkflow()
workflow.set_output_folder(output_folder_name)
workflow.set_gen_size(model_out_size)
workflow.set_ckpt(ckpt)
seed = random.randint(0, 10000000000)
seed = 4089594200 # override with a good one
print(f"Seed: {seed}")
workflow.set_seed(seed)
# write seed to absolute_output_folder
workflow.set_seed(4089594200)

with open(os.path.join(absolute_output_folder, "seed.txt"), "w") as f:
    f.write(str(seed))



previous_generation = "first_frame.png"

num_prompts = len(prompt_texts)

upscale_out_size = model_out_size * 2

for prompt_index in range(num_prompts):
    prompt_from = prompt_texts[prompt_index][0]
    prompt_to = prompt_texts[(prompt_index + 1) % num_prompts][0]
    steps = prompt_texts[prompt_index][1]

    workflow.set_prompts(prompt_prefix + prompt_from, prompt_prefix + prompt_to)

    print(f"Interpolating between {prompt_from} and {prompt_to} in {steps} steps")

    for step in range(steps):

        # load previous_generation image
        image = Image.open(previous_generation)
        # zoom in 1 % and rotate 1 degree
        image = image.rotate(1)
        crop_border_px = 0.02 * model_out_size
        image = image.crop((crop_border_px, crop_border_px, upscale_out_size - crop_border_px, upscale_out_size - crop_border_px))
        # save
        image.save(os.path.join(comfy_input_path, "previous_frame.png"))
        
        # count number of files in output folder
        files_before = os.listdir(absolute_output_folder)

        print(f"Step {step + 1} of {steps}")
        workflow.set_progress(step / steps)

        workflow_json = workflow.get_json()
        # write to file as debug
        with open("temp_recent_workflow.json", "w") as f:
            f.write(workflow_json)
        queue_prompt(workflow_json)

        print("waiting for comfy to finish")
        # wait until a file appears in comy output folder
        files_after = []
        while len(files_after) <= len(files_before):
            time.sleep(0.1)
            files_after = os.listdir(absolute_output_folder)

        print("comfy finished")
        # find the new file
        new_file = None
        for file in files_after:
            if file not in files_before:
                previous_generation = os.path.join(absolute_output_folder, file)
                break


        time.sleep(1.5) 

