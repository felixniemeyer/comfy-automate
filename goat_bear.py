
import json
from urllib import request, parse
import random

from prompt_interpolation import prompt_fade

def queue_prompt(prompt):
    p = f'{{"prompt": {prompt}}}'
    req =  request.Request("http://localhost:8188/prompt", data=p.encode('utf-8'), method='POST')
    request.urlopen(req).read()

ckpt = "sd_xl_base_1.0.safetensors"

prompt_texts = [
    "a goat living in berlin, taking a walk, hipster goat",
    "ice bear drifting on a floe, photorealistic",
    "a cat riding a rocket to mars", 
    "a parrot painting"
]

num_prompts = len(prompt_texts)

steps = 30 * 4

for prompt_index in range(num_prompts):
    prompt_from = prompt_texts[prompt_index]
    prompt_to = prompt_texts[(prompt_index + 1) % num_prompts]
    print(f"Interpolating between {prompt_from} and {prompt_to}")

    for step in range(steps):
        print(f"Step {step} of {steps}")
        prompt = prompt_fade.substitute(
            ckpt_name=ckpt,
            width=768,
            height=768,
            prompt_from=prompt_from,
            prompt_to=prompt_to,
            progress=step / steps, 
            output_prefix="goat_bear3/i", 
            cfg=7, 
            steps=48
        )
        queue_prompt(prompt)

