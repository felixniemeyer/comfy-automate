import math

import json
from urllib import request, parse
import random

from prompt_interpolation import prompt_fade

def queue_prompt(prompt):
    p = f'{{"prompt": {prompt}}}'
    req =  request.Request("http://localhost:8188/prompt", data=p.encode('utf-8'), method='POST')
    request.urlopen(req).read()

prompt_texts = [
    "cannabis plant",
    "cannabis leaves",
    "one cannabis leaf",
    "cannabis joint",
]

num_prompts = len(prompt_texts)

steps = 40

for prompt_index in range(num_prompts):
    prompt_from = prompt_texts[prompt_index]
    prompt_to = prompt_texts[(prompt_index + 1) % num_prompts]

    print(f"Interpolating between {prompt_from} and {prompt_to}")

    for step in range(steps):
        smoke_amount = (1 - math.cos(2 * math.pi * step / steps)) / 2 * 2
        smoke_postfix = f", (smoke: {smoke_amount})"
        print(f"Step {step} of {steps}. {smoke_postfix}")
        prompt = prompt_fade.substitute(
            prompt_from=prompt_from + smoke_postfix,
            prompt_to=prompt_to + smoke_postfix,
            progress=step / steps, 
            output_prefix="smoke/i"
        )
        queue_prompt(prompt)

