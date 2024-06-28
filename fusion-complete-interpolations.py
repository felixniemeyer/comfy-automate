import os 
import math

from depth_based_frame_interpolation import PromptInterpolationWorkflow 

from common import queue_prompt

dry_run = False

comfy_path = os.path.expanduser('~/work/ai/ComfyUI')
comfy_output_path = os.path.join(comfy_path, "output")
comfy_input_path = os.path.join(comfy_path, "input")

folder_name_without_number = "fusion-bg-interpolations/run"
run_number = 1
abs_output_folder = os.path.join(comfy_output_path, folder_name_without_number + str(run_number))
while os.path.exists(abs_output_folder):
    run_number += 1
    abs_output_folder = os.path.join(comfy_output_path, folder_name_without_number + str(run_number))

output_folder = folder_name_without_number + str(run_number) 


prompts = [
    "solarpunk building, vault, glass, solar panels, green plants, alien fauna, roots, cathedral, high saturation",
    # "ruins, magic forest, colorful plants, mushrooms, tree trunks in the front, peace, strong sunlight",
    "huge cyborg brain, building, fluids in tubes, processor chip, conducting paths", 
    # "die shot, CPU, mainboard, high tech, microchips, 6nm, computer, wires and cables, colorful strands of cables, transistors, coils, microchips", 
    "high security server room, computers, pins, boards, leds, switches, bright reflections",
    "a gate to the future, information flows, data highway, bit ones and zeros, (cyan light:0.75), rays and ripples, (red shadows:0.3), reflections, high contrast, ornaments on the backplate",
    "building, metal plates, lasers, poles, industrial, sci-fi, cyberpunk, welding, high contrast, (equally lit:0.7), electric power",
    "building, red bricks, (ivy on walls:0.6), (sunlight frontal:0.3)",
    "sci-fi, space station, (solar panels:0.2), antennas, satellites, stars", 
    # "space ship, rocket, metal, engines, starship, sci-fi city, tokyo, skyscrapers, neon lights, skyline, night, LED advertising, cyberpunk, bright buildings, dark sky, red, high contrast",
    "a building in which a machine lives with a head in the middle, high contrast, even lighting, bright lights, white columns"
]

neg_prompts = [
    "", 
    # "",
    "",
    # "",
    "",
    "text, numbers, letters, blur",
    "",
    "",
    "",
    # "",
    "",
]


# prompt_postfix = ", dark sky, high contrast, well lit, good lighting, strong shadows, bright highlights, intense colors"
prompt_postfix = "good lighting, shadows"
neg_prompt_postfix = ""

fps = 12
interpolation_duration = 8 # = 4 bars at 120 BPM

frame_count = fps * interpolation_duration

# run

workflow = PromptInterpolationWorkflow()

workflow.set_seed(1337)


# checks

assert len(prompts) == len(neg_prompts), "prompts and neg_prompts have to have the same length"
assert frame_count > 0, "frame count has to be at least 2 for 1 intermeidate frame"

combinations = len(prompts) * (len(prompts) - 1) // 2
total_frames = frame_count * combinations
spi = 1.3
total_seconds = total_frames * workflow.get_steps() * spi
hours = total_seconds / 3600

print("Estimated time to generate all frames:", hours, "hours")
print("Continue? y/n")
answer = input()
if answer.lower() != 'y':
    exit()

# go through all combinations of prompts

for i in range(len(prompts)): 
    workflow.set_prompts(
        prompts[i], prompts[i], 
        neg_prompts[i], neg_prompts[i], 
        prompt_postfix, neg_prompt_postfix
    )

    workflow.set_output_folder(output_folder + '/stills/' + str(i).zfill(3))

    workflow.set_progress(0.0)

    if dry_run:
        print (f"would generate still for {i}")
        # print('would send request:', workflow.get_json())
    else:
        queue_prompt(workflow.get_json())

    for j in range(i + 1, len(prompts)):

        workflow.set_prompts(
            prompts[i], prompts[j], 
            neg_prompts[i], neg_prompts[j], 
            prompt_postfix, neg_prompt_postfix
        )

        # pad with zeros 
        workflow.set_output_folder(output_folder + '/' + str(i).zfill(3) + 'to' + str(j).zfill(3))

        for f in range(1, frame_count):
            progress = (f / frame_count)

            # make slow around the middle
            weight = 0.5 * (progress * 2 + math.sin(progress * 2 * math.pi) * 0.24)

            workflow.set_progress(progress)
            # print(f"Progress: {progress}")
            if dry_run:
                print (f"would generate interpolation between {i} and {j} at progress {progress}")
                # print('would send request:', workflow.get_json())
            else:
                queue_prompt(workflow.get_json())

