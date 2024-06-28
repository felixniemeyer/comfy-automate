from frame_by_frame_constant_depth import run 

output_folder_name = "fusion"

resolution = 768

scenes = [
    ['robot head, digital information processing, led eyes, shiny metal', ''],
    ['pale , forest druid, wise, glowing eyes, fierce look', ''],
    ['rusty robot, gears, steam, clockwork, orange glow', ''],
    ['cyberpunk girl, aether, neon lights, soft skin, hologram, futuristic, wires and venes, muscle fibers', ''],
    ['terminator, skull, metal, red eyes', ''],
    ['gentle healer herbal remedies, light aura', ''],
    ['batteldroid, metal, gears, nuclear power, oil spilling, flames, scratches', ''],

    # ab hier chatGPT
    # extend this list: 
    # also use alternatingly: 
    # A) something female, natural, friendly
    # B) something male, technical, powerful

    ['enchanted fairy, wings, glowing dust', ''],
    ['cyborg warrior, exoskeleton, laser guns', ''],
    ['mermaid, ocean waves, coral crown', ''],
    ['robotic engineer, circuit board, holographic display', ''],
    ['mystic shaman, feathers, earth magic', ''],
    ['space marine, jetpack, blasters', ''],
    ['goddess of nature, vines, blooming flowers', ''],
    ['android assassin, stealth mode, nanotech', ''],
    ['moonlit witch, spellcasting, crescent moon', ''],
    ['cybernetic samurai, katana, augmented reality', ''],
    ['ethereal elf, silver hair, forest guardian', ''],
    ['steampunk inventor, goggles, mechanical arm', ''],
    ['beautiful islandic ghost, white skin, tribal face painting', ''],

    # ['magical unicorn, rainbow mane, sparkling horn', ''],

    # ['forest nymph, floral dress, serene', ''],
    # ['gentle healer, herbal remedies, light aura', ''],
]

scenes.append(scenes[0]) # make it loop

alternating_post = ['male', 'female']

prompt_texts = []
for i, scene in enumerate(scenes):
    print(i)
    prompt_texts.append([
        [
            alternating_post[i % 2] + ", " + scene[0],
            scene[1],
        ], 
        i * 5 # seconds
    ])

print("Scenes:", prompt_texts)


fps = 6

prompt_prefix = "cyborg head, full lips, open mouth, "

prompt_postfix = ", bio-mechanical, human-machine fusion"

neg_prompt_prefix = "text, watermark, blur, "
neg_prompt_postfix = ", teeth"

first_frame_ip_weight = 0.

depth_img_path = '/home/felix/work/projects/fusion-visuals/4K/head-ai-video/head-depth-v5-stretch110.png'

run(output_folder_name, prompt_texts, fps, 
    prompt_prefix, prompt_postfix, 
    neg_prompt_prefix, neg_prompt_postfix, 
    depth_img_path, 
    None, resolution, first_frame_ip_weight
)

