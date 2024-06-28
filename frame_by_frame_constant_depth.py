import os
import shutil
import time

import math

import json
import random

from PIL import Image 

from common import queue_prompt

def run(
    output_folder_name, prompt_texts, fps, 
    prompt_prefix, prompt_postfix, 
    neg_prompt_prefix, neg_prompt_postfix,
    depth_img_path, 
    ckpt=None, 
    resolution=768, 
    first_frame_ip_weight = 0, 
    first_frame_canny_strength = 0,
    rotation_amount = 0
    ): 
    workflow_file = "workflows/next_frame_const_depth_v2.json"

    comfy_path = os.path.expanduser('~/work/ai/ComfyUI')
    comfy_output_path = os.path.join(comfy_path, "output")
    comfy_input_path = os.path.join(comfy_path, "input")

    # ensure the output_folder is empty and exists
    answer = None
    postfix = 1
    absolute_output_folder = os.path.join(comfy_output_path, output_folder_name)
    while os.path.exists(absolute_output_folder):
        # promt the user to delete the folder
        if answer == None: 
            answer = 'i' # input(f"Folder {absolute_output_folder} already exists. (i)ncrement, (d)elete, (q)uit? Default is increment.")
        if answer.lower() == 'y':
            # go through all files in the folder and delete them
            for file in os.listdir(absolute_output_folder):
                os.remove(os.path.join(absolute_output_folder, file))
            os.rmdir(absolute_output_folder)
        if answer.lower() == 'q':
            print("Exiting")
            exit()
        else:
            print("Incrementing output folder name")
            postfix += 1
            absolute_output_folder = os.path.join(comfy_output_path, output_folder_name + str(postfix)) 

    if postfix > 1:
        output_folder_name = output_folder_name + str(postfix)

    print("output folder:", absolute_output_folder)

    # load next_fram_gen.json workflow and find relevant nodes
    ckpt_loader = None
    image_loader = None
    prompt_from_node = None
    prompt_to_node = None
    # TBD: negative_prompts interpolation
    neg_prompt_from_node = None
    neg_prompt_to_node = None
    conditioning_positive = None
    conditioning_negative = None
    gen_size = None
    k_sampler = None
    canny_control_net_node = None
    depth_control_net_node = None
    depth_img_loader = None

    original_ip_weight = None
    original_canny_strength = None

    with open(workflow_file) as f:
        workflow = json.load(f)
        # find CheckpointLoaderSimple in node dictionary
        for node_id in workflow:
            node = workflow[node_id]
            class_type = node["class_type"]
            if class_type == "CheckpointLoaderSimple":
                ckpt_loader = node
            if class_type == "ImageResizeAndCropNode":
                image_loader_id = node["inputs"]["image"][0]
                image_loader = workflow[image_loader_id]
            if class_type == "YANC.IntegerCaster":
                gen_size = node
            if class_type == "SaveImage":
                save_image = node
            if class_type == "KSampler": 
                k_sampler = node

                canny_control_net_node = workflow[k_sampler["inputs"]["positive"][0]]
                original_canny_strength = canny_control_net_node["inputs"]["strength"]
                canny_control_net_node["inputs"]["strength"] = 0

                # had 2 control nets, now just one xxx switched back to 2 (including canny)
                depth_control_net_node = workflow[canny_control_net_node["inputs"]["positive"][0]] 
                conditioning_positive = workflow[depth_control_net_node["inputs"]["positive"][0]]
                conditioning_negative = workflow[depth_control_net_node["inputs"]["negative"][0]]

                depth_img_loader = workflow[depth_control_net_node["inputs"]["image"][0]]

                prompt_from_node = workflow[conditioning_positive["inputs"]["conditioning_from"][0]]
                prompt_to_node = to_node_id = workflow[conditioning_positive["inputs"]["conditioning_to"][0]]

                neg_prompt_from_node = workflow[conditioning_negative["inputs"]["conditioning_from"][0]]
                neg_prompt_to_node = to_node_id = workflow[conditioning_negative["inputs"]["conditioning_to"][0]]
            if class_type == "IPAdapterAdvanced":
                ip_adapter = node
                original_ip_weight = ip_adapter["inputs"]["weight"]
                ip_adapter["inputs"]["weight"] = first_frame_ip_weight

    if ckpt_loader == None:
        print("Could not find CheckpointLoaderSimple in workflow", workflow_file)
        exit()
    if save_image == None:
        print("Could not find SaveImage in workflow", workflow_file)
        exit()
    if k_sampler == None:
        print("Could not find KSampler in workflow", workflow_file)
        exit()
    if conditioning_positive == None:
        print("Could not find ConditioningAverage in workflow", workflow_file)
        exit()
    if conditioning_negative == None:
        print("Could not find ConditioningAverage in workflow", workflow_file)
        exit()
    if gen_size == None:
        print("Could not find EmptyLatentImage in workflow", workflow_file)
        exit()
    if image_loader == None: 
        print("Could not find LoadImage in workflow", workflow_file)
        exit()
    if prompt_from_node == None: 
        print("Could not find prompt_from in workflow", workflow_file)
        exit()
    if prompt_to_node == None:
        print("Could not find prompt_to in workflow", workflow_file)
        exit()
    if neg_prompt_from_node == None:
        print("Could not find neg_prompt_from in workflow", workflow_file)
        exit()
    if neg_prompt_to_node == None:
        print("Could not find neg_prompt_to in workflow", workflow_file)
        exit()
    if ip_adapter == None:
        print("Could not find IPAdapterAdvanced in workflow", workflow_file)
        exit()
    if canny_control_net_node == None:
        print("Could not find CannyControlNet in workflow", workflow_file)
        exit()

    if(ckpt != None):
        ckpt_loader["inputs"]["ckpt_name"] = ckpt
    image_loader["inputs"]["image"] = os.path.join(comfy_input_path, "previous_frame.png")

    depth_img_loader["inputs"]["image"] = depth_img_path

    gen_size["inputs"]["value"] = resolution
    save_image["inputs"]["filename_prefix"] = output_folder_name + '/f'

    def setPrompts(from_p, to_p, negfp, negtp, total_steps=0):
        prompt_from = prompt_prefix + from_p + prompt_postfix
        prompt_to = prompt_prefix + to_p + prompt_postfix
        prompt_from_node["inputs"]["text"] = prompt_from
        prompt_to_node["inputs"]["text"] = prompt_to

        neg_prompt_from = neg_prompt_prefix + negfp + neg_prompt_postfix
        neg_prompt_to = neg_prompt_prefix + negtp + neg_prompt_postfix

        print("prompt_from", prompt_from)
        print("neg prompt_from", neg_prompt_from)
        print("prompt_to", prompt_to)
        print("neg prompt_to", neg_prompt_to)

    def setProgress(progress):
        conditioning_positive["inputs"]["conditioning_to_strength"] = progress
        conditioning_negative["inputs"]["conditioning_to_strength"] = progress

    def setSeed(seed):
        k_sampler["inputs"]["seed"] = seed


    if not os.path.exists(absolute_output_folder):
        os.makedirs(absolute_output_folder)

    previous_generation = depth_img_path

    num_prompts = len(prompt_texts)

    total_steps = 0

    for prompt_index in range(num_prompts - 1):
        prompt_text_from = prompt_texts[prompt_index]
        prompt_text_to = prompt_texts[prompt_index + 1]
        pf = prompt_text_from[0][0]
        pt = prompt_text_to[0][0]
        npf = prompt_text_from[0][1]
        npt = prompt_text_to[0][1]
        steps = int((prompt_text_to[1] - prompt_text_from[1]) * fps)


        print(f"Interpolating between {pf} and {pt} in {steps} steps")

        for step in range(steps):


            setPrompts(pf, pt, npf, npt, total_steps)

            # copy previous generation to comfy input folder
            prev_frame_path = os.path.join(comfy_input_path, "previous_frame.png")
            shutil.copy(previous_generation, prev_frame_path)
            
            # count number of files in output folder
            files_before = os.listdir(absolute_output_folder)

            print(f"Step {step + 1} of {steps}")

            setProgress(step / steps)
            setSeed(random.randint(0, 10000000000))

            workflow_string = json.dumps(workflow)

            # save to file
            with open('./fbf_recent_workflow.json', "w") as f:
                f.write(workflow_string)

            queue_prompt(workflow_string)

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

            total_steps += 1

            ip_adapter["inputs"]["weight"] = original_ip_weight # reset ip weight
            canny_control_net_node["inputs"]["strength"] = original_canny_strength
