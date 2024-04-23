import math

import random

from common import queue_prompt

from prompt_interpolation_workflow import PromptInterpolationWorkflow


prompt_texts = [
    "hemp, cannabis plant",
    "cannabis leaves, harvested leaves",
    "cannabis joint, marjuana",
]

num_prompts = len(prompt_texts)

workflow = PromptInterpolationWorkflow()
workflow.set_ckpt("sd_xl_base_1.0.safetensors")
workflow.set_gen_size(768)
workflow.set_output_folder("smoke3")
workflow.set_seed(random.randint(0, 1000000))

steps = 40

for prompt_index in range(num_prompts):
    prompt_from = prompt_texts[prompt_index]
    prompt_to = prompt_texts[(prompt_index + 1) % num_prompts]

    print(f"Interpolating between {prompt_from} and {prompt_to}")

    for step in range(steps):
        smoke_amount = (1 - math.cos(2 * math.pi * step / steps)) / 2
        smoke_postfix = f", (smoke: {smoke_amount})"
        print(f"Step {step} of {steps}. {smoke_postfix}")
        workflow.set_prompts(
            prompt_from + smoke_postfix,
            prompt_to + smoke_postfix,
        )
        print(f"prompt_from {prompt_from + smoke_postfix} -> prompt_to {prompt_to + smoke_postfix}")
        workflow.set_progress(step / steps)
        queue_prompt(workflow.get_json())

