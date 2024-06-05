from frame_by_frame import run 

output_folder_name = "pipo_neg"

resolution = 768

scenes = [
]

prompt_texts = []
for i, scene in enumerate(scenes):
    print(i)
    prompt_texts.append([
        scene, 
        i * 30
    ])

print("Scenes:", prompt_texts)


fps = 6

prompt_prefix = ""
prompt_postfix = ""

neg_prompt_prefix = "text, letters, watermark, missing limbs, extra limbs, deformed limbs, CGI, cartoon"
neg_prompt_postfix = "text, letters, watermark, blurred"

first_frame = '/home/felix/work/ai/ComfyUI/output/first_frame_768x2_00401_.png'
# '/home/felix/work/ai/ComfyUI/output/first_frame_768x2_00382_.png'
# '/home/felix/work/ai/ComfyUI/output/first_frame_768x2_00287_.png'
# '/home/felix/work/mediaworks/ai-gen/collected/0000167.png'
# '/home/felix/work/ai/ComfyUI/output/first_frame_768x2_00176_.png'
# '/home/felix/work/ai/ComfyUI/output/pipo28/f_00004_.png'
# '/home/felix/work/ai/ComfyUI/output/first_frame_768x2_00274_.png'
# '/home/felix/work/ai/ComfyUI/output/first_frame_768x2_00175_.png'

run(output_folder_name, prompt_texts, fps, 
    prompt_prefix, prompt_postfix, 
    neg_prompt_prefix, neg_prompt_postfix, 
    None, first_frame, resolution)
