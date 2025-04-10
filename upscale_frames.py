import os
import time

import json

from common import queue_prompt

import argparse

def run(
    input_folder, 
    output_prefix,
    comfy_output_folder,
    workflow_file, 
    continue_processing=False, 
    ):

    # load next_fram_gen.json workflow and find relevant nodes
    save_image = None
    image_loader = None

    with open(workflow_file) as f:
        workflow = json.load(f)
        # find CheckpointLoaderSimple in node dictionary
        for node_id in workflow:
            node = workflow[node_id]
            class_type = node["class_type"]
            if class_type == "LoadImage":
                image_loader = node
            if class_type == "SaveImage":
                save_image = node

    if save_image == None:
        print("Could not find SaveImage in workflow", workflow_file)
        exit()
    if image_loader == None: 
        print("Could not find LoadImage in workflow", workflow_file)
        exit()

    # get absolute path of input folder
    input_folder = os.path.abspath(input_folder)

    save_image["inputs"]["filename_prefix"] = output_prefix + "/f"
    
    abs_output_folder = os.path.join(comfy_output_folder, output_prefix)

    input_frames = os.listdir(input_folder)
    input_frames.sort()

    number_of_existing_frames = len(os.listdir(abs_output_folder))
    
    # prompt for continue
    if number_of_existing_frames > 0 and not continue_processing:
        print(f"Output folder {abs_output_folder} already exists with {number_of_existing_frames} frames.")
        print("Do you want to continue or exit? (c/e)")
        choice = input()
        if choice == "e":
            print("Exiting...")
            exit()
        elif choice != "c":
            print("Invalid choice, exiting...")
            exit()

    max_queue_size = 3
    out_file_count = number_of_existing_frames
    frame = 0
    for input_frame in input_frames:
        if frame < number_of_existing_frames: 
            frame += 1
            continue
    
        frame_path = os.path.join(input_folder, input_frame)
        image_loader["inputs"]["image"] = frame_path
        workflow_string = json.dumps(workflow)

        # save to file
        with open('./f2f_recent_workflow.json', "w") as f:
            f.write(workflow_string)

        print(f"enqueuing {input_frame}...")
        queue_prompt(workflow_string)

        # wait for file to appear
        while not out_file_count + max_queue_size > frame : 
            print(f'waiting for jobs to finish. {out_file_count} + {max_queue_size} > {frame}')
            out_file_count = len(os.listdir(abs_output_folder))
            time.sleep(1)
            
        frame += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Frame2Frame')
    parser.add_argument('input_folder', type=str, help='Input folder containing frames')
    parser.add_argument('output_prefix', type=str, help='Comfy will output to upscale/<output_prefix>/f_')
    parser.add_argument('comfy_output_folder', type=str, help='folder with comfy output')

    parser.add_argument('--workflow', '-w', type=str, default="workflows/upscale/default.json", help='Workflow file to use')
    parser.add_argument('--continue_processing', help='Continue from last frame', default=False, action='store_true')

    args = parser.parse_args()
    
    args.comfy_output_folder = os.path.abspath(args.comfy_output_folder)

    run(
        args.input_folder, 
        args.output_prefix, 
        args.comfy_output_folder,
        args.workflow, 
        args.continue_processing,
    )