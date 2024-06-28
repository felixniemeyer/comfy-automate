import os
import json

class PromptInterpolationWorkflow:
    def __init__(self):
        workflow_file = "workflows/depth-based-prompt-interpolation.json"

        with open(workflow_file) as f:
            self.workflow = json.load(f)

        for node_id in self.workflow:
            node = self.workflow[node_id]
            class_type = node["class_type"]
            if class_type == "CheckpointLoaderSimple":
                self.ckpt_loader = node
            if class_type == "EmptyLatentImage":
                self.gen_size = node
            if class_type == "SaveImage":
                self.save_image = node
            if class_type == "KSampler": 
                self.k_sampler = node
                self.control_net_node = self.workflow[self.k_sampler["inputs"]["positive"][0]]
                self.positive_conditioning_average = self.workflow[self.control_net_node["inputs"]["positive"][0]]
                self.negative_conditioning_average = self.workflow[self.control_net_node["inputs"]["negative"][0]]
                self.positive_prompt_from = self.workflow[self.positive_conditioning_average["inputs"]["conditioning_from"][0]]
                self.positive_prompt_to = self.workflow[self.positive_conditioning_average["inputs"]["conditioning_to"][0]]
                self.negative_prompt_from = self.workflow[self.negative_conditioning_average["inputs"]["conditioning_from"][0]]
                self.negative_prompt_to = self.workflow[self.negative_conditioning_average["inputs"]["conditioning_to"][0]]

    def set_ckpt(self, ckpt):
        self.ckpt_loader["inputs"]["ckpt_name"] = ckpt

    def set_gen_size(self, model_out_size):
        self.gen_size["inputs"]["width"] = model_out_size
        self.gen_size["inputs"]["height"] = model_out_size

    def set_output_folder(self, output_folder_name):
        self.save_image["inputs"]["filename_prefix"] = output_folder_name + '/f'

    def set_prompts(self, from_p, to_p, from_neg_p, to_neg_p, postfix, neg_postfix):
        self.positive_prompt_from["inputs"]["text"] = ', '.join([from_p, postfix])
        self.positive_prompt_to["inputs"]["text"] = ', '.join([to_p, postfix])
        self.negative_prompt_from["inputs"]["text"] = ', '.join([from_neg_p, neg_postfix])
        self.negative_prompt_to["inputs"]["text"] = ', '.join([to_neg_p, neg_postfix])

    def set_progress(self, progress):
        self.positive_conditioning_average["inputs"]["conditioning_to_strength"] = progress
        self.negative_conditioning_average["inputs"]["conditioning_to_strength"] = progress

    def set_seed(self, seed):
        self.k_sampler["inputs"]["seed"] = seed

    def get_json(self):
        return json.dumps(self.workflow)

    def get_steps(self):
        return self.k_sampler["inputs"]["steps"]


