import os
import time

import json

from common import queue_prompt

import argparse

def run(
    input_folder, 
    output_prefix,
    workflow_file, 
    pause=10.0
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

    # get input folder name
    input_folder_name = os.path.basename(input_folder)

    save_image["inputs"]["filename_prefix"] = output_prefix

    input_frames = os.listdir(input_folder)
    input_frames.sort()

    for input_frame in input_frames:
        frame_path = os.path.join(input_folder, input_frame)
        image_loader["inputs"]["image"] = frame_path

        print(f"enqueuing {input_frame}...")

        workflow_string = json.dumps(workflow)

        # save to file
        with open('./f2f_recent_workflow.json', "w") as f:
            f.write(workflow_string)

        queue_prompt(workflow_string)

        # wait for file to be written to disk
        time.sleep(pause) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Frame2Frame')
    parser.add_argument('input_folder', type=str, help='Input folder containing frames')
    parser.add_argument('output_prefix', type=str, help='Comfy will output to upscale/<output_prefix>/f_')
    parser.add_argument('--workflow', '-w', type=str, default="workflows/upscale/default.json", help='Workflow file to use')
    parser.add_argument('--pause', '-p', help='Pause in seconds between enqueuing frames', default=10.0)
    args = parser.parse_args()

    run(args.input_folder, args.output_prefix, args.workflow, args.pause)

