'''
this script takes input images from a folder and processes them frame by frame
'''

import os
import time
import math

import json

from common import queue_prompt

import argparse

import numpy as np

class KeyFrame: 
    def __init__(self, prompt: str, frame: int):
        self.prompt = prompt
        self.frame = frame  

class Img2ImgTransformer:

    def __init__(
        self,
        workflow_file: str,
        ):
        
        with open(workflow_file) as f:
            workflow = json.load(f)
            # find CheckpointLoaderSimple in node dictionary
            for node_id in workflow:
                node = workflow[node_id]
                class_type = node["class_type"]
                if class_type == "SaveImage":
                    self.save_image = node
                #image loader
                if class_type == "LoadImage":
                    self.image_loader = node
                if class_type == "ConditioningAverage": 
                    self.conditioning_average = node
                    from_prompt_id = node["inputs"]["conditioning_from"][0]
                    to_prompt_id = node["inputs"]["conditioning_to"][0]
                    self.prompt_from = workflow[from_prompt_id]
                    self.prompt_to = workflow[to_prompt_id]
            self.workflow = workflow

        if self.conditioning_average == None:
            raise Exception("Could not find ConditioningAverage in workflow", workflow_file)
        if self.save_image == None:
            raise Exception("Could not find SaveImage in workflow", workflow_file)
        if self.prompt_from == None:
            raise Exception("Could not find ConditioningFrom in workflow", workflow_file)
        if self.prompt_to == None:  
            raise Exception("Could not find ConditioningTo in workflow", workflow_file)
        if self.image_loader == None:
            raise Exception("Could not find ImageLoader in workflow", workflow_file)
    
    def updatePrompts(self, keyframes, frame_number: int):
        (prompt_from, prompt_to, weight) = self.getPromptsAndWeight(keyframes, frame_number)
        self.conditioning_average["inputs"]["conditioning_to_strength"] = weight
        self.prompt_from["inputs"]["text"] = prompt_from
        self.prompt_to["inputs"]["text"] = prompt_to
        
    def getPromptsAndWeight(self, keyframes, frame_number: int): 
        if(len(keyframes) == 1):
            return keyframes[0].prompt, keyframes[0].prompt, 1.0
        else: 
            # find the two keyframes that the frame_number is between
            keyframe_a = None
            keyframe_b = None    
            for i in range(1, len(keyframes)):
                if(keyframes[i].frame > frame_number):
                    keyframe_a = keyframes[i - 1]
                    keyframe_b = keyframes[i]
                    break
            
            weight = (frame_number - keyframe_a.frame) / (keyframe_b.frame - keyframe_a.frame)
            
            return keyframe_a.prompt, keyframe_b.prompt, weight

    def run(
        self,
        # image_in_folder,
        # comfy_output_folder,
        # out_file_prefix,
        image_in_folder: str,
        comfy_output_folder: str, 
        out_sub_path: str,
        keyframes: list[KeyFrame],
    ):
        
        if(len(keyframes) < 1):
            # throw error
            raise Exception("At least one keyframe is required")

        image_in_folder = os.path.abspath(image_in_folder)

        # get a list of images from input folder
        image_list = os.listdir(image_in_folder)
        image_list.sort()

        abs_output_folder = os.path.join(comfy_output_folder, out_sub_path)
        # throw error if exists else create
        if os.path.exists(abs_output_folder): 
            raise Exception("Output folder already exists")
        else:
            os.makedirs(abs_output_folder, exist_ok=True)
        
        self.save_image["inputs"]["filename_prefix"] = f"{out_sub_path}/f"
        
        max_queue_size = 3
        out_file_count = 0 # folder empty in the beginning
        frame = 0
        # go through all images
        for i in range(len(image_list)):

            # make abs path
            image_path = os.path.join(image_in_folder, image_list[i])
            self.image_loader["inputs"]["image"] = image_path
            self.updatePrompts(keyframes, frame)
            frame += 1
            
            workflow_string = json.dumps(self.workflow)

            # save to file
            with open('./last_enqueued_workflow.json', "w") as f:
                f.write(workflow_string)

            queue_prompt(workflow_string)

            # debug, stop after 2
            # if frame > 1: 
            #     break

            # wait for file to appear
            while i - out_file_count > max_queue_size - 2:
                print('waiting for jobs to finish')
                out_file_count = len(os.listdir(abs_output_folder))
                time.sleep(5)

            # wait for file to be written
            time.sleep(1)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='transforms vids frame by frame')

    parser.add_argument('image_in_folder', type=str, help='folder with images to interpolate between')
    
    # tell me the location of comyUI out folder
    parser.add_argument('comfy_output_folder', type=str, help='folder with comfy output')
    parser.add_argument('out_sub_path', type=str, help='folder within comfy output folder')

    # workflow file
    parser.add_argument('--workflow', '-w', type=str, default="workflows/vid2vid_fbf/v0.json", help='Workflow file to use for each image')

    # TODO: allow specifying prompts per keyframe
    # parser.add_argument('--prompts', type=str, nargs='+', help='Prompts to interpolate between')
    # hardcode for now
    keyframes = [
        KeyFrame("a house full of cats and gongs", 0),
    ]

    args = parser.parse_args()

    vid2vid = Img2ImgTransformer(args.workflow)
    
    vid2vid.run(
        args.image_in_folder,
        args.comfy_output_folder,
        args.out_sub_path,
        keyframes
    )

    #interpolations = len(args.prompts) - 1

    #print(f"running {interpolations} interpolations")

    #padding = math.floor(math.log10(interpolations))
    #last_frame = None
    #for i in range(interpolations):
    #    print(f"interpolating between {args.prompts[i]} and {args.prompts[i + 1]}")
    #    interp = Interpolator(
    #        run_folder,
    #        args.comfy_output_folder,
    #        args.prompts[i],
    #        args.prompts[i + 1],
    #        args.workflow,
    #        ssim_img_distance, 
    #    )

    #    last_frame = interp.run(
    #        args.frame_count,
    #        args.gap_size, 
    #        args.append_p, 
    #        f"{i:0{padding}}-",
    #        first_frame = last_frame
    #    )

    #    last_frame.p = 0.0
    


