import os
import json

class PromptInterpolationWorkflow:
    def __init__(self):
        workflow_file = "workflows/prompt_interpolation_upscale.json"
        with open(workflow_file) as f:
            self.workflow = json.load(f)

        for node_id in self.workflow:
            node = self.workflow[node_id]
            class_type = node["class_type"]
            if class_type == "CheckpointLoaderSimple":
                self.ckpt_loader = node
            if class_type == "ConditioningAverage":
                self.conditioning_average = node
                self.prompt_from = workflow[conditioning_average["inputs"]["conditioning_from"][0]]
                self.prompt_to = workflow[conditioning_average["inputs"]["conditioning_to"][0]]
            if class_type == "EmptyLatentImage":
                self.gen_size = node
            if class_type == "SaveImage":
                self.save_image = node
            if class_type == "KSampler": 
                self.k_sampler = node

        missing = []
        if self.ckpt_loader == None: 
            missing.append("CheckpointLoaderSimple")
        if self.prompt_from == None:
            missing.append("prompt_from")
        if self.prompt_to == None:
            missing.append("prompt_to")
        if self.conditioning_average == None:
            missing.append("ConditioningAverage")
        if self.gen_size == None:
            missing.append("EmptyLatentImage")
        if self.save_image == None:
            missing.append("SaveImage")
        if self.k_sampler == None:
            missing.append("KSampler")

        if len(missing) > 0:
            print("Could not find all nodes in workflow", missing)
            exit()

    def set_ckpt(self, ckpt):
        self.ckpt_loader["inputs"]["ckpt_name"] = ckpt

    def set_gen_size(self, model_out_size):
        self.gen_size["inputs"]["width"] = model_out_size
        self.gen_size["inputs"]["height"] = model_out_size

    def set_output_folder(self, output_folder_name):
        self.save_image["inputs"]["filename_prefix"] = output_folder_name + '/f'

    def set_prompts(self, from_p, to_p):
        self.prompt_from["inputs"]["text"] = from_p
        self.prompt_to["inputs"]["text"] = to_p

    def set_progress(self, progress):
        self.conditioning_average["inputs"]["conditioning_to_strength"] = progress

    def set_seed(self, seed):
        self.k_sampler["inputs"]["seed"] = seed

    def get_json(self):
        return json.dumps(self.workflow)
