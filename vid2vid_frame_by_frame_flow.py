'''
this script takes input images from a folder and processes them frame by frame
'''

import os
import time

import cv2
import torch 
import torch.nn.functional as F

import json

from common import queue_prompt

import argparse

import os
import argparse
from PIL import Image
from PIL.ExifTags import TAGS

def warp_image(image, flow):
    """
    Warp image2 to align with image1 using the flow from image1 to image2.

    Args:
        image2 (torch.Tensor): Image to warp, shape [B, C, H, W] or [C, H, W]
        flow (torch.Tensor): Flow from image1 to image2, shape [B, 2, H, W] or [2, H, W]
    
    Returns:
        torch.Tensor: Warped image aligned with image1, shape [B, C, H, W]
    """
    # Ensure inputs have a batch dimension
    if image.ndim == 3:
        image = image.unsqueeze(0)  # [C, H, W] -> [B, C, H, W]
    if flow.ndim == 3:
        flow = flow.unsqueeze(0)  # [2, H, W] -> [B, 2, H, W]

    B, C, H, W = image.shape

    # Generate coordinate grid for image1 (target coordinates)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device='cpu'),
        torch.arange(W, device='cpu'),
        indexing='ij'
    )
    coords = torch.stack([grid_x, grid_y], dim=0).float()  # [2, H, W]
    coords = coords.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]

    # Add flow to get sampling coordinates in image2
    sampling_coords = coords + flow  # [B, 2, H, W]

    # Normalize coordinates to [-1, 1] for grid_sample
    sampling_coords[:, 0, :, :] = 2.0 * sampling_coords[:, 0, :, :] / (W - 1) - 1.0  # x
    sampling_coords[:, 1, :, :] = 2.0 * sampling_coords[:, 1, :, :] / (H - 1) - 1.0  # y

    # Warp image2 using grid_sample
    warped_image = F.grid_sample(
        image,
        sampling_coords.permute(0, 2, 3, 1),  # [B, H, W, 2]
        mode='bilinear',
        padding_mode='border', 
        align_corners=True
    )  # [B, C, H, W]

    return warped_image

def get_strip_metadata(image_path):
    """Extract the 'Strip' metadata from a PNG image."""
    try:
        img = Image.open(image_path)
        metadata = img.info  # Get metadata dictionary
        return metadata.get("Strip", None)  # Return the Strip value if available
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

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
                    self.save_image_node = node
                #image loader
                if class_type == "LoadImage":
                    self.image_loader_node = node
                if class_type == "RandomNoise":
                    self.random_noise_node = node
                if class_type == "BasicScheduler":
                    self.basic_scheduler_node = node
                if class_type == "ConditioningAverage": 
                    self.conditioning_average_node = node
                    from_prompt_id = node["inputs"]["conditioning_from"][0]
                    to_prompt_id = node["inputs"]["conditioning_to"][0]
                    self.prompt_from_node = workflow[from_prompt_id]
                    self.prompt_to_node = workflow[to_prompt_id]
            self.workflow = workflow

        if self.conditioning_average_node == None:
            raise Exception("Could not find ConditioningAverage in workflow", workflow_file)
        if self.save_image_node == None:
            raise Exception("Could not find SaveImage in workflow", workflow_file)
        if self.prompt_from_node == None:
            raise Exception("Could not find ConditioningFrom in workflow", workflow_file)
        if self.prompt_to_node == None:  
            raise Exception("Could not find ConditioningTo in workflow", workflow_file)
        if self.image_loader_node == None:
            raise Exception("Could not find ImageLoader in workflow", workflow_file)

        self.random_noise_node["inputs"]["noise_seed"] = 1337
    
        
    def setBothPromptsTo(self, prompt: str):
        self.prompt_from_node["inputs"]["text"] = prompt
        self.prompt_to_node["inputs"]["text"] = prompt
        self.conditioning_average_node["inputs"]["conditioning_to_strength"] = 1.0

    def updatePrompts(self, keyframes, frame_number: int):
        (prompt_from, prompt_to, weight) = self.getPromptsAndWeight(keyframes, frame_number)
        self.conditioning_average_node["inputs"]["conditioning_to_strength"] = weight
        self.prompt_from_node["inputs"]["text"] = prompt_from
        self.prompt_to_node["inputs"]["text"] = prompt_to
        
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
        temp_image_file: str = "tmp.png",
        prompts: list[str] = None,
    ):
        print('len prompts', len(prompts))

        image_in_folder = os.path.abspath(image_in_folder)

        # get a list of images from input folder
        image_list = os.listdir(image_in_folder)
        image_list.sort()
        # filter only png or jpg
        image_list = [f for f in image_list if f.endswith('.png') or f.endswith('.jpg')]


        abs_output_folder = os.path.join(comfy_output_folder, out_sub_path)
        # throw error if exists else create
        if os.path.exists(abs_output_folder): 
            raise Exception("Output folder already exists")
        else:
            os.makedirs(abs_output_folder, exist_ok=True)
        
        self.save_image_node["inputs"]["filename_prefix"] = f"{out_sub_path}/f"

        frame = 0
        previous_strip = None
        previous_file = None
        
        prompt_index = 0

        # go through all images
        for i in range(len(image_list)):

            image_name = image_list[i]
            image_path = os.path.join(image_in_folder, image_name)
            current_strip = get_strip_metadata(image_path)

            if previous_file == None or current_strip != previous_strip:
                print(f"Cut detected at {image_name}")
                self.basic_scheduler_node['inputs']['denoise'] = 0.78
                self.basic_scheduler_node['inputs']['steps'] = 15
                input_path = image_path
                print('switching to next prompt:', prompts[prompt_index])
                self.setBothPromptsTo(prompts[prompt_index])
                prompt_index += 1
            else: 
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
                image = image[None]

                previous_frame = cv2.imread(previous_file)
                previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2RGB)
                print(previous_frame.shape, image.shape)
                previous_frame = cv2.resize(previous_frame, (image.shape[3], image.shape[2]), interpolation=cv2.INTER_LINEAR)
                print(previous_frame.shape)
                previous_frame = torch.tensor(previous_frame, dtype=torch.float32).permute(2, 0, 1)
                previous_frame = previous_frame[None]

                # load flow image
                flow_path = os.path.join(image_in_folder, "flows", f"{i:05d}.pt")
                flow = torch.load(flow_path, map_location='cpu')
                print('flow shape', flow.shape)

                # combine images
                warped_image = warp_image(previous_frame, flow)

                # blend warped with actual image
                # flow is uv values
                # calculate the average length of the uv vector for every
                # flow shape is ch.Size([1, 2, 1280, 720])

                average_length = torch.mean(torch.sqrt(torch.sum(flow.pow(2), dim=1)))
                average_length = average_length.item()
                print('avg flow length', average_length)

                image_weight = (1 - 1 / (average_length * 0.25 + 1)) * 0.1
                image = image * image_weight + warped_image * (1.0 - image_weight)

                # save image 
                image = cv2.cvtColor(image[0].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    temp_image_file, 
                    image
                )

                denoise = 0.4 # (1 - 1 / (average_length * 0.25 + 1)) * 0.25 + 0.3
                self.basic_scheduler_node['inputs']['denoise'] = denoise
                self.basic_scheduler_node['inputs']['steps'] = int((18 * denoise) + 1)
                input_path = temp_image_file
                
            
            self.random_noise_node["inputs"]["noise_seed"] += 1
            self.image_loader_node["inputs"]["image"] = input_path
            
            workflow_string = json.dumps(self.workflow)

            # save to file
            with open('./last_enqueued_workflow.json', "w") as f:
                f.write(workflow_string)

            queue_prompt(workflow_string)

            # wait for file to appear
            expected_filename = os.path.join(
                comfy_output_folder, 
                out_sub_path, 
                f"f_{1 + frame:05d}_.png"
            )
            while os.path.exists(os.path.join(abs_output_folder, expected_filename)) == False:
                print('waiting for file to appear', expected_filename)
                time.sleep(1)

            # wait for file to be written
            time.sleep(1)
            
            previous_strip = current_strip  # Update for next iteration
            previous_file = expected_filename

            frame += 1
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='transforms vids frame by frame')

    parser.add_argument('image_in_folder', type=str, help='folder with images to interpolate between')
    
    # tell me the location of comyUI out folder
    parser.add_argument('comfy_output_folder', type=str, help='folder with comfy output')
    parser.add_argument('out_sub_path', type=str, help='folder within comfy output folder')

    # workflow file
    parser.add_argument('--workflow', '-w', type=str, default="workflows/vid2vid_fbf/v0.json", help='Workflow file to use for each image')
    
    parser.add_argument('--temp_img_file', type=str, default="tmp.png", help='Temporary image file to use for each image')

    parser.add_argument('--prompt-per-clip-schedule', '-p', help='File for prompts', required=True)

    args = parser.parse_args()
    
    # read the json file
    with open(args.prompt_per_clip_schedule) as f:
        # it contains an array of prompts (strings)
        prompts = json.load(f)
        # reverse prompts
        prompts.reverse()
        # append general prompt to every prompt
        for i in range(len(prompts)):
            prompts[i] += ". artificial superintelligence has arrived on earth and is merging with nature. cyborgs, cables and silicon chips intertwined with beautiful nature. photorealistic masterpiece."
        print(prompts)
    
    args.temp_img_file = os.path.abspath(args.temp_img_file)

    vid2vid = Img2ImgTransformer(args.workflow)
    
    vid2vid.run(
        args.image_in_folder,
        args.comfy_output_folder,
        args.out_sub_path,
        args.temp_img_file,
        prompts
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
    


