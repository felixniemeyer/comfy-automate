
import json
from urllib import request, parse
import random

from magical_apple_heart import prompt_fade

def queue_prompt(prompt):
    p = f'{{"prompt": {prompt}}}'
    req =  request.Request("http://localhost:8188/prompt", data=p.encode('utf-8'), method='POST')
    request.urlopen(req).read()

prompt_texts = [
    "magical apple",
    "open heart",
    "artesanial bow", 
    "alien garden", 
    "reciprocated love", 
    "argentinian tango", 
    "juicy mango", 
    "beach sunset",
    "mountain sunrise",
    "pale moonlight",
    "amazing starlight",
    "colorful rainbow",
    "blue rain",
    "white snow",
    "first spring",
    "last summer",
    "this fall",
    "coldest winter",
    "thin forest",
    "rocky mountain",
    "vast ocean",
    "wild river",
    "huge lake",
]

num_prompts = len(prompt_texts)

steps = 20

for prompt_index in range(num_prompts):
    prompt_from = prompt_texts[prompt_index]
    prompt_to = prompt_texts[(prompt_index + 1) % num_prompts]
    print(f"Interpolating between {prompt_from} and {prompt_to}")

    for step in range(steps):
        print(f"Step {step} of {steps}")
        prompt = prompt_fade.substitute(
            ckpt_name = "sd_xl_turbo_1.0_fp16.safetensors", 
            width=512,
            height=512,
            prompt_from=prompt_from,
            prompt_to=prompt_to,
            progress=step / steps, 
            output_prefix="ai_loop/i"
        )
        queue_prompt(prompt)

