import os
import time

import json
import random

from PIL import Image 

from flythrough import flythrough

from common import queue_prompt

ckpt = "sd_xl_base_1.0.safetensors"

comfy_path = os.path.expanduser('~/work/ai/ComfyUI')
comfy_output_path = os.path.join(comfy_path, "output")
comfy_input_path = os.path.join(comfy_path, "input")

output_folder_name = "weed_loop"

model_out_size = 768

prompt_texts = [
    ["marjuana, cannabis leaves, joint, tip, stoner", 3],
    ["countries, airplane, airport, weed", 3], 
    ["festival, joint, marjuana", 3], 
    ["fines, police, airport, drug smuggling", 3], 
]

prompt_prefix = "travelling, cannabis, marjuana, airplane, airport"

factor = 3
for prompt_text in prompt_texts:
    prompt_text[1] = prompt_text[1] * factor


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

previous_generation = "first_frame.png"

num_prompts = len(prompt_texts)

upscale_out_size = model_out_size * 2

for prompt_index in range(num_prompts):
    prompt_from = prompt_texts[prompt_index][0]
    prompt_to = prompt_texts[(prompt_index + 1) % num_prompts][0]
    steps = prompt_texts[prompt_index][1]

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

        print(f"Step {step} of {steps}")
        prompt = flythrough.substitute(
            ckpt_name=ckpt,
            size=model_out_size,
            previous_frame="previous_frame.png",
            prompt_from=prompt_from,
            prompt_to=prompt_to,
            progress=step / steps, 
            output_folder=output_folder_name, 
            control_strength=1,
            style_weight=0,
            composition_weight=0,
        )
        queue_prompt(prompt)

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

        time.sleep(1.5) # wait for file to be written to disk

#         file_size = os.path.getsize(previous_generation)
#         new_size = 0 
#         # ensure file is not being written to
#         while new_size != file_size:
#             print(f"Waiting for file to be written to: {previous_generation}")
#             time.sleep(0.1)
#             new_size = os.path.getsize(previous_generation)
# 
        

