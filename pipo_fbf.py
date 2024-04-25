from frame_by_frame import run 

output_folder_name = "pipo"

resolution = 768

scenes = [
    "tomatoes, squeezing tomatoes out of pussy",
    "janice griffith, deep inside, natural light, penis deep inside, penis fully in pussy, penis all the way in the pussy, vaginal sex, (pussy closeup:1.45), creampie, dick slip out",
    "underwater, sexy woman, deep water, porn, dark blue water, gracious body, sexy pose, penis penetrating, skin friction, vagina penetration",
    "face sitting, drooling pussy, cum, creampie, pussy, piss",
    "girls swimming naked, dark blue water, pussies exposed, many girls naked, swimming, pussy", 
    "orange bodies in blue water, orgy, intercourse, 3 couples, (mycelium: 0.3)",
    "penis, vagina, penis in vagina, penetration, pool, dark blue water, merging, sex, porn",
    "vagina closeup, penetration, dick in pussy, inserted dick",
    "doggy style, huge dick slip out, cum, squirt, happy woman, satisfied woman, passionate look",
    "creampie, cum, pussy, close up, dripping, drooling", 
]

prompt_texts = []
for i, scene in enumerate(scenes):
    print(i)
    prompt_texts.append([
        scene, 
        i * 3
    ])

print("Scenes:", prompt_texts)


fps = 6

prompt_prefix = ""
prompt_postfix = "slim, juicy pussy, beautiful, sexy, seductive, janice griffith, natural lighting, cock, girl, woman, babe, dark blue background, orange skin tones, warm bodies, underwater, photorealistic"

first_frame = '/home/felix/work/ai/ComfyUI/output/first_frame_768x2_00287_.png'
# '/home/felix/work/mediaworks/ai-gen/collected/0000167.png'
# '/home/felix/work/ai/ComfyUI/output/first_frame_768x2_00176_.png'
# '/home/felix/work/ai/ComfyUI/output/pipo28/f_00004_.png'
# '/home/felix/work/ai/ComfyUI/output/first_frame_768x2_00274_.png'
# '/home/felix/work/ai/ComfyUI/output/first_frame_768x2_00175_.png'

run(output_folder_name, prompt_texts, fps, prompt_prefix, prompt_postfix, None, first_frame, resolution)
