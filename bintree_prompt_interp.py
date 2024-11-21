'''
This script generates frames from one prompt to another. 
It generates that frame that halfs the biggest gap. 
'''

import os
import time

import json

from common import queue_prompt

import argparse

import numpy as np
from skimage.io import imread

class Frame:
    def __init__(self, path, p):
        self.path = path
        self.p = p

class Interpolator:

    def __init__(
        self,
        output_prefix,
        comfy_output_folder,
        prompt_from, 
        prompt_to,
        workflow_file, 
        ):

        # load next_fram_gen.json workflow and find relevant nodes
        self.conditioning_average = None
        self.comfy_output_folder = comfy_output_folder

        with open(workflow_file) as f:
            workflow = json.load(f)
            # find CheckpointLoaderSimple in node dictionary
            for node_id in workflow:
                node = workflow[node_id]
                class_type = node["class_type"]
                if class_type == "SaveImage":
                    node["inputs"]["filename_prefix"] = output_prefix
                if class_type == "ConditioningAverage": 
                    self.conditioning_average = node
                    from_prompt_id = node["inputs"]["conditioning_from"]
                    to_prompt_id = node["inputs"]["conditioning_to"]
                    workflow[from_prompt_id]["inputs"]["text"] = prompt_from
                    workflow[to_prompt_id]["inputs"]["text"] = prompt_to
            self.workflow = workflow

        if save_image == None:
            raise Exception("Could not find SaveImage in workflow", workflow_file)
        if self.conditioning_average == None:
            raise Exception("Could not find ConditioningAverage in workflow", workflow_file)

        # get absolute path of output folder
        output_folder = os.path.abspath(comfy_output_folder)

        # get absolute output path
        i = 0
        ok = False
        while not ok:
            self.output_folder = os.path.join(output_folder, output_prefix)
            self.output_folder = os.path.dirname(self.output_folder) + str(i)
            if os.path.exists(self.output_folder):
                i += 1
            else:
                ok = True

    runned = False
    def run(self, frame_count):
        if(frame_count < 3):
            raise Exception("frame_count must be at least 3")
        if self.runned:
            raise Exception("run can only be called once")
        self.runned = True

        self.initialize_tree()

        for i in range(frame_count - 3):
            # find largest gap
            largest_gap_size = 0
            largest_gap_id = 0
            for i in range(len(self.gaps)):
                if self.gaps[i] > largest_gap_size:
                    largest_gap_size = self.gaps[i]
                    largest_gap_id = i

            p = 0.5 * (self.frames[largest_gap_id].p + self.frames[largest_gap_id + 1].p)

            frame = get_frame_at(p)

            # insert frame at the right position
            self.frames.insert(largest_gap_id + 1, frame)

            # update gaps
            gap_left = img_distance(self.frames[largest_gap_id], frame) 
            gap_right = img_distance(frame, self.frames[largest_gap_id + 1])
            self.gaps[largest_gap_id] = gap_left
            self.gaps.insert(largest_gap_id + 1, gap_right)

        # rename all files to be in order
        for i in range(len(self.frames)):
            frame = self.frames[i]
            new_path = os.path.join(self.output_folder, f"{i:05}.png")
            os.rename(frame.path, new_path)
            frame.path = new_path

    def initialize_tree(self):
        # run workflow for conditioning_to_strength 0, 0.5, 1 
        self.frames = [
            get_frame_at(0.0, workflow), 
            get_frame_at(0.5, workflow), 
            get_frame_at(1.0, workflow)
        ]
        self.gaps = [
            img_distance(tree[0], tree[1]), 
            img_distance(tree[1], tree[2])
        ]
    
    out_id = 0
    def get_frame_at(p):
        print(f"generating frame at {p}")

        self.conditioning_average["inputs"]["conditioning_to_strength"] = p

        workflow_string = json.dumps(workflow)

        # save to file
        with open('./last_enqueued_workflow.json', "w") as f:
            f.write(workflow_string)

        expected_filepath = os.path.join(self.comfy_output_folder, f"{output_prefix}_{out_id:05}_.png")

        queue_prompt(workflow_string)

        # wait for file to appear
        while not os.path.exists(expected_filepath):
            time.sleep(0.5)

        return Frame(
            expected_filepath, 
            p
        )

    def img_distance(frame1, frame2):
        # calculate Mean Squared Error 
        img1 = imread(frame1.path)
        img2 = imread(frame2.path)

        # Calculate MSE across all channels
        return = np.mean((img1 - img2) ** 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bintree prompt travel generates frames from one prompt to another always filling the gaps with the largest distance')

    parser.add_argument('prompt_from', type=str, help='Prompt from')
    parser.add_argument('prompt_to', type=str, help='Prompt to')

    parser.add_argument('comfy_output_folder', type=str, help='Comfy output folder')
    parser.add_argument('--output_prefix', '-p', type=str, help='like in ImageSave node', default="bininterp/v/f")
    parser.add_argument('--workflow', '-w', type=str, default="workflows/bintree_prompt_interp/v0.json", help='Workflow file to use')

    parser.add_argument('--frame_count', '-n', type=int, default=10, help='Number of frames to generate')

    args = parser.parse_args()

    interp = Interpolator(
        args.output_prefix,
        args.comfy_output_folder,
        args.prompt_from,
        args.prompt_to,
        args.workflow,
    )

    interp.run(args.frame_count)
