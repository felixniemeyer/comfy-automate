'''
This script generates frames from one prompt to another. 
It generates that frame that halfs the biggest gap. 
'''

import os
import time
import math

import json

from common import queue_prompt

import argparse

import numpy as np
from skimage.io import imread
# from skimage.measure import block_reduce
from skimage.metrics import structural_similarity as ssim

# import imagehash
# from PIL import Image

class Frame:
    def __init__(self, path, p): #, img_hash):
        self.path = path
        self.p = p
        # self.img_hash = img_hash

class Interpolator:

    def __init__(
        self,
        subfolder,
        comfy_out_abs_path,
        prompt_from, 
        prompt_to,
        workflow_file, 
        img_distance, 
        ):

        # load next_fram_gen.json workflow and find relevant nodes
        self.conditioning_average = None
        self.img_distance = img_distance

        save_image = None

        with open(workflow_file) as f:
            workflow = json.load(f)
            # find CheckpointLoaderSimple in node dictionary
            for node_id in workflow:
                node = workflow[node_id]
                class_type = node["class_type"]
                if class_type == "SaveImage":
                    save_image = node
                if class_type == "ConditioningAverage": 
                    self.conditioning_average = node
                    from_prompt_id = node["inputs"]["conditioning_from"][0]
                    to_prompt_id = node["inputs"]["conditioning_to"][0]
                    workflow[from_prompt_id]["inputs"]["text"] = prompt_from
                    workflow[to_prompt_id]["inputs"]["text"] = prompt_to
            self.workflow = workflow

        if self.conditioning_average == None:
            raise Exception("Could not find ConditioningAverage in workflow", workflow_file)
        if save_image == None:
            raise Exception("Could not find SaveImage in workflow", workflow_file)

        # get absolute path of output folder
        self.output_folder = os.path.join(comfy_out_abs_path, subfolder)

        save_image["inputs"]["filename_prefix"] = f"{subfolder}/f"

    runned = False
    def run(
        self, 
        frame_count, 
        gap_size, 
        append_p, 
        filename_prefix="",
        first_frame=None):
        if(frame_count < 2):
            raise Exception("frame_count must be at least 2")
        if self.runned:
            raise Exception("run can only be called once")
        self.runned = True

        self.out_id = 1
        if first_frame != None:
            self.frames = [first_frame]
        else:
            self.frames = [self.get_frame_at(0.0)]

        last_frame = self.get_frame_at(1.0)
        self.frames.append(last_frame)

        self.gaps = [
            self.img_distance(self.frames[0], self.frames[1])
        ]

        frame_id = 0 
        while frame_id < frame_count - 2:
            # find largest gap
            largest_gap_size = 0
            largest_gap_id = 0
            for i in range(len(self.gaps)):
                if self.gaps[i] > largest_gap_size:
                    largest_gap_size = self.gaps[i]
                    largest_gap_id = i

            if largest_gap_size < gap_size:
                print('reached gap target, stopping')
                break

            left_frame = self.frames[largest_gap_id]
            right_frame = self.frames[largest_gap_id + 1]

            if right_frame.p - left_frame.p < 0.001:
                print('p to low, ignoring this gap')
                self.gaps[largest_gap_id] = 0
                continue

            p = 0.5 * (left_frame.p + right_frame.p)

            frame = self.get_frame_at(p)

            # insert frame at the right position
            self.frames.insert(largest_gap_id + 1, frame)

            # update gaps
            gap_left = self.img_distance(left_frame, frame) 
            gap_right = self.img_distance(frame, right_frame)
            self.gaps[largest_gap_id] = gap_left
            self.gaps.insert(largest_gap_id + 1, gap_right)

            print('updated gaps', [round(g, 3) for g in self.gaps])

            frame_id += 1

        # rename all files to be in order
        for i in range(len(self.frames)):
            frame = self.frames[i]
            filename = f"{filename_prefix}{i:05}"
            if append_p:
                filename += '-' + str(frame.p)
            new_path = os.path.join(self.output_folder, f"{filename}.png")
            os.rename(frame.path, new_path)
            frame.path = new_path

        return last_frame

    out_id = 1
    def get_frame_at(self, p):
        print(f"generating frame at {p}")

        self.conditioning_average["inputs"]["conditioning_to_strength"] = p

        workflow_string = json.dumps(self.workflow)

        # save to file
        with open('./last_enqueued_workflow.json', "w") as f:
            f.write(workflow_string)

        expected_filepath = os.path.join(self.output_folder, f"f_{self.out_id:05}_.png")
        self.out_id += 1

        queue_prompt(workflow_string)

        # wait for file to appear
        print('waiting for', expected_filepath)
        while not os.path.exists(expected_filepath):
            time.sleep(0.5)

        # wait for file to be written
        time.sleep(1)

        # load image and get hash
        # img = imread(expected_filepath)
        # img_pil = Image.fromarray(img)
        # img_hash = imagehash.average_hash(img_pil, hash_size=32)

        # load image and create thumbnail for comparison
        img = imread(expected_filepath)
        img = img[::4, ::4]

        return Frame(
            expected_filepath, 
            p, 
            #    img_hash
        )

def mse_img_distance(frame1, frame2):
    # calculate Mean Squared Error 
    img1 = imread(frame1.path)
    img2 = imread(frame2.path)

    # Calculate MSE across all channels
    distance = np.mean((img1 - img2) ** 2)

    return distance.item()

def hamming_img_distance(frame1, frame2):
    # calculate Hamming distance
    return frame1.img_hash - frame2.img_hash

def ssim_img_distance(frame1, frame2):
    # calculate SSIM
    img1 = imread(frame1.path)
    img2 = imread(frame2.path)

    similarity = ssim(img1, img2, channel_axis=2)
    return 1 - similarity.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bintree prompt travel generates frames from one prompt to another always filling the gaps with the largest distance')

    parser.add_argument('--out_path', '-p', type=str, help='folder within comfy output folder', default="bininterp")
    parser.add_argument('--run_name', '-r', type=str, help='run name, default auto increment', default=None)

    parser.add_argument('--workflow', '-w', type=str, default="workflows/bintree_prompt_interp/v0.json", help='Workflow file to use')

    parser.add_argument('--frame_count', '-n', type=int, default=10, help='Max number of frames to generate')
    parser.add_argument('--gap_size', '-g', type=float, default=0.05, help='Stop when no gap exceeds this size')

    parser.add_argument('--append_p', action='store_true', help='Append p values in file name', default=True)

    parser.add_argument('comfy_output_folder', type=str, help='Comfy output folder')

    parser.add_argument('prompts', type=str, nargs='+', help='Prompts to interpolate between')

    args = parser.parse_args()

    # stick together path
    comfy_out_abs_path = os.path.abspath(args.comfy_output_folder)
    bininterp_folder = os.path.join(comfy_out_abs_path, args.out_path)
    if args.run_name is not None:
        run_folder = os.path.join(bininterp_folder, args.run_name)
    else: 
        i = 0
        ok = False
        while not ok:
            run_folder = os.path.join(bininterp_folder, f"{i}") 
            if not os.path.exists(run_folder):
                ok = True
            else:
                i += 1

    interpolations = len(args.prompts) - 1

    print(f"running {interpolations} interpolations")

    padding = math.floor(math.log10(interpolations))
    last_frame = None
    for i in range(interpolations):
        print(f"interpolating between {args.prompts[i]} and {args.prompts[i + 1]}")
        interp = Interpolator(
            run_folder,
            args.comfy_output_folder,
            args.prompts[i],
            args.prompts[i + 1],
            args.workflow,
            ssim_img_distance, 
        )

        last_frame = interp.run(
            args.frame_count,
            args.gap_size, 
            args.append_p, 
            f"{i:0{padding}}-",
            first_frame = last_frame
        )

        last_frame.p = 0.0

